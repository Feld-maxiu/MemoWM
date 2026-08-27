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
