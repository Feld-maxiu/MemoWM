"""The reader-side losses, in one place so the two trainers cannot diverge.

``train_reader_qa`` and ``train_qformer_joint`` optimize the same thing and
differ only in where the latent comes from -- a cached ``xbar`` in one, a
learnable resampler in the other. Keeping the loss in both files is how the
distillation term ended up mis-scaled in one of them for a whole run:

``F.kl_div(..., reduction="batchmean")`` divides by ``input.shape[0]``. On a
``(1, answer_len, vocab)`` tensor that is a **sum** over the answer positions,
not a mean, so ``distill_weight=1.0`` actually weighted the KL by roughly the
answer length -- 10 to 30x. Flattening the positions into the batch dimension
first is what makes the weight mean what it says.
"""
from __future__ import annotations

import inspect

import torch
from torch.nn import functional as F

# The reader builds exactly this at inference (``Qwen35LatentReader.answer``);
# training on a different shape is what made the first connector answer and then
# keep going, narrating the observation.
INSTRUCTION = (
    "Answer the question from the retrieved memory. Keep the answer concise. "
    "If absent, say exactly 'Not mentioned in memory.'.\nQuestion: "
)

# A floor on the student's log-probability inside every forward KL here. e^-30
# is 1e-13, far below anything a converging model produces, so this changes no
# ordinary value; what it removes is the unbounded tail that turns one confident
# mistake into a loss of thousands and a non-finite gradient.
LOGPROB_FLOOR = -30.0


def topk_kl(
    student_logits: torch.Tensor,
    topk_index: torch.Tensor,
    topk_logprob: torch.Tensor,
) -> torch.Tensor:
    """KL(teacher || student) over the teacher's top-k support, per position.

    The only truncated-KL implementation; both distillation terms call it.

    Truncating at the *teacher's* top-k is principled for this direction rather
    than merely expedient. ``F.kl_div(input=student, target=teacher,
    log_target=True)`` computes ``sum_v p_t(v) [log p_t(v) - log p_s(v)]``, so
    every term is weighted by a teacher probability; dropping the tail drops the
    smallest-weighted terms, and the residual mass is a direct error bound. The
    precompute records that mass and refuses to write a cache whose median falls
    below a threshold.

    Both distributions are renormalized over the retained support, so this is
    the KL between two honest distributions rather than a truncated sum that
    quietly stops summing to one.

    ``train_reader_qa.py:19-22`` rejects caching teacher logits: "cheaper than
    the storage and simpler than a top-k approximation". That holds for its
    teacher -- 89 tokens of caption, no vision. It does not hold for a teacher
    that reads the screenshot: ~1172 tokens through the vision tower, recomputed
    every micro-batch. The premise changed, so the conclusion does.
    """
    if topk_index.shape != topk_logprob.shape:
        raise ValueError(
            f"top-k index {tuple(topk_index.shape)} and logprob "
            f"{tuple(topk_logprob.shape)} disagree"
        )
    if student_logits.shape[:-1] != topk_index.shape[:-1]:
        raise ValueError(
            f"student logits {tuple(student_logits.shape)} cover "
            f"{student_logits.shape[:-1]} positions but the teacher covers "
            f"{topk_index.shape[:-1]}"
        )
    student = F.log_softmax(student_logits.float(), -1)
    # The cache lives on CPU -- 1.6 GB of it, with only a few spans live at a
    # time. Moving here rather than at each call site is deliberate: both
    # distillation paths funnel through this function, and the one that forgot
    # died on a device mismatch instead of doing something quietly wrong.
    gathered = student.gather(-1, topk_index.to(student.device))
    teacher = topk_logprob.float().to(student.device)
    # Floor the student *before* renormalizing. Placing it after was close to a
    # no-op: renormalizing over the top-k support subtracts a logsumexp that the
    # extreme values themselves dominate, so a student log-prob of -1000 came
    # out around log(1/k) and the clamp never fired. The unbounded quantity is
    # the raw log-prob, and that is what has to be bounded -- forward KL's term
    # is p_t * (log p_t - log p_s), with no floor under log p_s. Measured before
    # any floor: 2090 nats from one such token.
    gathered = gathered.clamp_min(LOGPROB_FLOOR)
    teacher = teacher - teacher.logsumexp(-1, keepdim=True)
    gathered = gathered - gathered.logsumexp(-1, keepdim=True)
    per_position = (teacher.exp() * (teacher - gathered)).sum(-1)
    return per_position.mean()


def observation_distill_kl(
    model,
    processor,
    latent: torch.Tensor,
    valid: torch.Tensor,
    probe: str,
    continuation_ids: torch.Tensor,
    topk_index: torch.Tensor,
    topk_logprob: torch.Tensor,
) -> torch.Tensor:
    """Make the latent reproduce what the raw observation would have produced.

    The teacher read the screenshot, the AXTree and ``probe``, and generated
    ``continuation_ids`` greedily; ``topk_*`` are its per-position distributions
    over that span, precomputed. The student sees only the latent and the same
    probe, and is scored on the same span.

    Nothing here touches a benchmark label. The span is the teacher's own
    output, so this is a pure fidelity objective: it asks whether a handful of
    soft tokens is a behavioural substitute for the observation itself. That is
    the distinction from ``answer_ce_and_distill_kl``, whose CE targets a gold
    answer and whose text teacher reads a caption already embedded verbatim in
    the student's own DOM span.

    Prefix lengths differ wildly between the two sides -- ~1172 tokens against
    ~50 -- but only the suffix has to line up, and it does: the last
    ``len(continuation)`` logit positions correspond because both sequences end
    with the same probe and the same continuation. Measured, padding the student
    out to the teacher's absolute position moves the resulting gap by 0.0002,
    so the twenty-fold RoPE offset is not a factor.

    Returns the KL as a tensor carrying the graph.
    """
    device = latent.device
    embed = model.get_input_embeddings()
    dtype = embed.weight.dtype

    probe_ids = processor.tokenizer(
        probe, return_tensors="pt", add_special_tokens=True
    )["input_ids"].to(device)
    continuation_ids = continuation_ids.to(device)
    length = int(continuation_ids.shape[1])
    if length == 0:
        raise ValueError(f"empty teacher continuation for probe {probe!r}")

    inputs = torch.cat((
        latent.to(dtype), embed(probe_ids).detach(), embed(continuation_ids).detach()
    ), 1)
    attention = torch.cat((
        valid.to(torch.long), torch.ones_like(probe_ids), torch.ones_like(continuation_ids)
    ), 1)
    student = model(inputs_embeds=inputs, attention_mask=attention, use_cache=False)
    return topk_kl(
        student.logits[:, -length - 1 : -1],
        topk_index.to(device),
        topk_logprob.to(device),
    )


def answer_ce_and_distill_kl(
    model,
    processor,
    latent: torch.Tensor,
    valid: torch.Tensor,
    question: str,
    answer: str,
    memory_text: str,
    *,
    max_answer_tokens: int = 64,
    weight: float = 1.0,
    max_memory_tokens: int = 256,
    instruction: str = INSTRUCTION,
    teacher_topk: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Teacher-forced CE on the gold answer, plus KL to a teacher trajectory.

    ``latent`` is ``(1, slots, model_dim)`` soft tokens and carries the graph;
    ``valid`` is ``(1, slots)``. Returns ``(loss, answer_ce, kl)`` with the last
    two as floats. **The arity is load-bearing** -- three trainers and five test
    assertions destructure it, one of them in an untracked file -- so a second
    distillation term lives in its own function rather than widening this one.

    ``teacher_topk`` is an optional ``(index, logprob)`` pair precomputed from a
    teacher that read the *raw observation*. Without it the teacher is
    recomputed from ``memory_text``, which reproduces every run to date.
    """
    device = latent.device
    embed = model.get_input_embeddings()
    dtype = embed.weight.dtype

    question_ids = processor.tokenizer(
        instruction + question, return_tensors="pt", add_special_tokens=True
    )["input_ids"].to(device)
    answer_ids = processor.tokenizer(
        answer, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=max_answer_tokens,
    )["input_ids"].to(device)
    question_embeds = embed(question_ids).detach()
    answer_embeds = embed(answer_ids).detach()
    answer_len = int(answer_ids.shape[1])
    if answer_len == 0:
        raise ValueError(f"empty answer for question {question!r}")

    def run(memory_embeds, memory_mask):
        inputs = torch.cat((memory_embeds.to(dtype), question_embeds, answer_embeds), 1)
        attention = torch.cat(
            (memory_mask, torch.ones_like(question_ids), torch.ones_like(answer_ids)), 1
        )
        labels = torch.cat((
            torch.full(memory_mask.shape, -100, device=device, dtype=torch.long),
            torch.full_like(question_ids, -100),
            answer_ids,
        ), 1)
        return model(inputs_embeds=inputs, attention_mask=attention,
                     labels=labels, use_cache=False)

    student = run(latent, valid.to(torch.long))
    loss = student.loss
    if weight <= 0 or (teacher_topk is None and not memory_text):
        return loss, float(student.loss), 0.0

    student_logits = student.logits[:, -answer_len - 1 : -1]
    if teacher_topk is not None:
        # The precomputed teacher read the screenshot and the AXTree, not the
        # caption. That matters: the caption is embedded verbatim in the
        # student's own DOM span, so the text teacher's entire input is a subset
        # of the student's and this term was teaching it to reproduce text it
        # already had. Same span, same alignment, a teacher that actually knows
        # something the student has to work for.
        index, logprob = teacher_topk
        if int(index.shape[-2]) != answer_len:
            raise ValueError(
                f"cached teacher covers {index.shape[-2]} answer positions but "
                f"this answer tokenizes to {answer_len} -- the cache and the "
                "pairs file disagree"
            )
        kl = topk_kl(student_logits, index, logprob)
        return loss + weight * kl, float(student.loss), float(kl)

    with torch.no_grad():
        text_ids = processor.tokenizer(
            memory_text, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=max_memory_tokens,
        )["input_ids"].to(device)
        teacher = run(embed(text_ids), torch.ones_like(text_ids))

    # The two sequences differ only in how the memory reached the model, so the
    # last answer_len logit positions correspond.
    teacher_logits = teacher.logits[:, -answer_len - 1 : -1]
    # Same floor as topk_kl, for the same reason: this is the path every
    # divergence to date was actually running.
    kl = F.kl_div(
        F.log_softmax(student_logits.float(), -1).clamp_min(
            LOGPROB_FLOOR).reshape(answer_len, -1),
        F.log_softmax(teacher_logits.float(), -1).reshape(answer_len, -1),
        reduction="batchmean", log_target=True,
    )
    return loss + weight * kl, float(student.loss), float(kl)


# Every answer in ``qformer-qa-pairs.npz`` tokenizes to at most 102 tokens, so
# 104 scores the whole gold answer. The 64 that ``answer_ce_and_distill_kl``
# defaults to -- correct for training, where it bounds the step cost -- truncates
# 43.4% of them, and a truncated span makes variants comparable only on a prefix.
FULL_ANSWER_TOKENS = 104


def answer_variant_scores(
    model,
    processor,
    latents: torch.Tensor,
    question: str,
    answer: str,
    *,
    max_answer_tokens: int = FULL_ANSWER_TOKENS,
    reference: int = 0,
    kl_chunk: int = 8,
    instruction: str = INSTRUCTION,
) -> dict:
    """Score B memory variants of one question in a single forward.

    ``latents`` is ``(B, slots, model_dim)``; row ``reference`` is the intact
    memory and the rest are ablations of it. Returns, per variant, the total
    teacher-forced gold-answer NLL and the KL from the reference's answer
    distribution, both in **bits**, plus the teacher-forced argmax so a caller can
    ask whether the answer actually changed.

    Three things here are not stylistic.

    **Every variant must be scored inside the same forward.** In bfloat16 the
    same variant scored at B=1 and at B=33 differs by 0.64 nats, and the
    difference is per-row rather than a common offset (std 0.30), against an
    ablation signal of order 1 nat. Scoring the reference separately from its
    ablations, or splitting the ablations into chunks, injects noise at roughly
    half the signal. In float32 the same comparison is 7.6e-5 nats, which is why
    the labelling pass runs float32 -- but the invariant is cheap to keep either
    way, so this function takes all the variants at once and refuses to be handed
    a reference index outside the batch.

    **``valid`` is not a parameter.** It is built all-ones. Masking a slot is a
    different operation at different batch sizes: ``apply_mask_to_padding_states``
    zeroes masked states in the 24 linear-attention layers only when
    ``attention_mask.shape[0] > 1`` (``modeling_qwen3_5.py:207-215``), so a
    ``valid=0`` ablation does not mean the same thing at B=1 as at B=33. Omission
    is expressed by substituting the decoder's estimate into ``latents``, which
    keeps the sequence, the positions and the attendability fixed and changes only
    the vector -- which is also what equation (23) describes.

    **``rope_deltas`` is cleared.** It is cached instance state written by any
    prior image forward and folded into the position ids whenever ``input_ids`` is
    None (``modeling_qwen3_5.py:1509-1519``). It cancels within a batch, so the
    KL and the NLL differences are safe either way, but it shifts the absolute
    NLL by ~0.055 nats and would make values incomparable across questions.

    The KL is not floored, unlike ``topk_kl``. There the teacher is a different
    model and the tail is genuinely unbounded; here both sides are the same model
    on near-identical inputs, so a large per-position term means the ablation
    destroyed a token the reference was confident about -- the measurement, not an
    artifact. ``kl_max_position`` is returned so an outlier stays visible.
    """
    if latents.dim() != 3:
        raise ValueError(f"latents must be (B, slots, dim), got {tuple(latents.shape)}")
    variants = int(latents.shape[0])
    if not 0 <= reference < variants:
        raise ValueError(f"reference {reference} outside a batch of {variants}")

    device = latents.device
    embed = model.get_input_embeddings()
    dtype = embed.weight.dtype

    question_ids = processor.tokenizer(
        instruction + question, return_tensors="pt", add_special_tokens=True
    )["input_ids"].to(device)
    answer_ids = processor.tokenizer(
        answer, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=max_answer_tokens,
    )["input_ids"].to(device)
    answer_len = int(answer_ids.shape[1])
    if answer_len == 0:
        raise ValueError(f"empty answer for question {question!r}")

    inner = getattr(model, "model", None)
    if hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None

    with torch.no_grad():
        question_embeds = embed(question_ids).expand(variants, -1, -1)
        answer_embeds = embed(answer_ids).expand(variants, -1, -1)
        memory_mask = torch.ones(latents.shape[:2], dtype=torch.long, device=device)
        inputs = torch.cat(
            (latents.to(dtype), question_embeds, answer_embeds), 1)
        attention = torch.cat((
            memory_mask,
            torch.ones((variants, question_ids.shape[1]), dtype=torch.long, device=device),
            torch.ones((variants, answer_len), dtype=torch.long, device=device),
        ), 1)

        keep = answer_len + 1
        accepted = "logits_to_keep" in inspect.signature(model.forward).parameters
        extra = {"logits_to_keep": keep} if accepted else {}
        out = model(inputs_embeds=inputs, attention_mask=attention,
                    use_cache=False, **extra)

        # logits[:, t] predicts token t+1, so answer token j is predicted at
        # S + QL + j - 1: the last answer_len positions offset by one. Identical
        # slice whether or not logits_to_keep trimmed the sequence.
        answer_logits = out.logits[:, -answer_len - 1 : -1, :]
        targets = answer_ids.expand(variants, -1)

        nll_nats = torch.zeros(variants, dtype=torch.float32, device=device)
        token_nll = torch.empty((variants, answer_len), dtype=torch.float32,
                                device=device)
        kl_nats = torch.zeros(variants, dtype=torch.float32, device=device)
        kl_max = torch.zeros(variants, dtype=torch.float32, device=device)
        argmax = torch.empty((variants, answer_len), dtype=torch.long, device=device)

        for start in range(0, answer_len, kl_chunk):
            stop = min(start + kl_chunk, answer_len)
            # .float() before log_softmax mirrors ForCausalLMLoss, so the NLL is
            # on the same footing as the loss HF would have reported.
            logprob = F.log_softmax(answer_logits[:, start:stop].float(), -1)
            gathered = logprob.gather(
                -1, targets[:, start:stop].unsqueeze(-1)).squeeze(-1)
            nll_nats -= gathered.sum(-1)
            token_nll[:, start:stop] = -gathered
            argmax[:, start:stop] = logprob.argmax(-1)
            anchor = logprob[reference : reference + 1]
            per_position = (anchor.exp() * (anchor - logprob)).sum(-1)
            kl_nats += per_position.sum(-1)
            kl_max = torch.maximum(kl_max, per_position.max(-1).values)

    ln2 = float(torch.log(torch.tensor(2.0)))
    return {
        "nll_bits": (nll_nats / ln2).cpu(),
        # Per answer token, so a caller can ask whether an ablation disturbs a
        # few tokens or all of them. That decides whether summing over the answer
        # is length-invariant (localized) or carries an answer-length scale
        # (diffuse) -- and therefore whether the sum or the per-token mean is the
        # quantity commensurate with the code bits.
        "token_nll_bits": (token_nll / ln2).cpu(),
        "kl_bits": (kl_nats / ln2).cpu(),
        "kl_max_position_bits": (kl_max / ln2).cpu(),
        "teacher_forced_argmax": argmax.cpu(),
        "answer_len": answer_len,
        "question_len": int(question_ids.shape[1]),
        "reference": reference,
    }

