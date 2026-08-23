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
):
    """Teacher-forced CE on the gold answer, plus KL to the text-fed trajectory.

    ``latent`` is ``(1, slots, model_dim)`` soft tokens and carries the graph;
    ``valid`` is ``(1, slots)``. The frozen model is shared by both conditions,
    so the only difference between student and teacher is how the memory
    arrived. Returns ``(loss, answer_ce, kl)`` with the last two as floats.
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
    if weight <= 0 or not memory_text:
        return loss, float(student.loss), 0.0

    with torch.no_grad():
        text_ids = processor.tokenizer(
            memory_text, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=max_memory_tokens,
        )["input_ids"].to(device)
        teacher = run(embed(text_ids), torch.ones_like(text_ids))

    # The two sequences differ only in how the memory reached the model, so the
    # last answer_len logit positions correspond.
    student_logits = student.logits[:, -answer_len - 1 : -1]
    teacher_logits = teacher.logits[:, -answer_len - 1 : -1]
    kl = F.kl_div(
        F.log_softmax(student_logits.float(), -1).reshape(answer_len, -1),
        F.log_softmax(teacher_logits.float(), -1).reshape(answer_len, -1),
        reduction="batchmean", log_target=True,
    )
    return loss + weight * kl, float(student.loss), float(kl)
