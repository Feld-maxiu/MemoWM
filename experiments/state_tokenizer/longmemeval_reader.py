"""Native reader adapter for LongMemEval-V2.

The WMA reader remains untouched.  This adapter reuses its connector and
segment embedding implementation but renders LongMemEval's ``\\boxed{}``
contract and can carry the question image through the Qwen multimodal
processor.  Retrieved historical screenshots are not re-injected; only the
question image is.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from residualmem.latent.instruct_bridge import MemorySegment, Qwen35LatentReader

from .prompts import ABS_CONTRACT_SYSTEM_PROMPT


LME_WEB_SYSTEM_PROMPT = (
    "You are an experienced colleague in a web browsing environment that has "
    "a customized magento-based shopping website, a customized magento-based "
    "shopping admin cms website, as well as a customized forum website based "
    "on reddit/postmill. Answer based on your memory of the environment. "
    "If you do not know the answer, output exactly \\boxed{UNKNOWN}. "
    "Do not guess. Never attempt to guess an answer if you are not sure. "
    "If you believe the question's construction/premise is wrong, provide an "
    "explanation in \\boxed{} explaining why the question is flawed."
)

                                                                    
                                                                            
                                                                       
                         
LME_WEB_SYSTEM_PROMPT_OFFICIAL_BYTES = LME_WEB_SYSTEM_PROMPT.replace(
    "in \\boxed{}", "in \boxed{}"
)

                                                                     
                                                                                
                                                                             
                                                                                
                                                                            
                                                                      
PREMISE_EXPLANATION_CLAUSE = (
    " If the premise is false, say what is missing in \\boxed{} instead of "
    "\\boxed{UNKNOWN}."
)
LONGMEMEVAL_SYSTEM_PROMPT = LME_WEB_SYSTEM_PROMPT + PREMISE_EXPLANATION_CLAUSE
                                                                               
                                                          
longmemeval_system_prompt = LONGMEMEVAL_SYSTEM_PROMPT

SYSTEM_PROMPTS = {
    "longmemeval": LONGMEMEVAL_SYSTEM_PROMPT,
    "official": LME_WEB_SYSTEM_PROMPT_OFFICIAL_BYTES,
                                                                             
                                                                              
                                                                
    "abscontract": ABS_CONTRACT_SYSTEM_PROMPT,
}

LME_USER_TEMPLATE = """### Memory context:
{context}

### Question to answer:
{question}"""


def reader_sampling_kwargs(*, temperature: float = 0.6, top_p: float = 0.95,
                           top_k: int = 20) -> dict:
    """Official Qwen reader defaults, not the retrieval/controller settings.

    Source: LongMemEval-V2/evaluation/run_eval.py (--reader-* arguments).
    Temperature zero is an explicit opt-in for replaying legacy greedy controls.
    """
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("reader-temperature must be finite and nonnegative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("reader-top-p must be in (0, 1]")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("reader-top-k must be a nonnegative integer")
    if temperature == 0:
                                                                        
        return {"do_sample": False}
    return {"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k}


def decode_reader_completion(tokenizer, continuation, *, enable_thinking: bool,
                             max_new_tokens: int):
    """Keep reasoning separate from the final text passed to the official scorer.

    Qwen's thinking template already opens the reasoning block in the prompt.
    A completion without its closing delimiter has not produced a final answer;
    in particular, a boxed candidate inside unfinished reasoning is not an answer.
    """
    ids = continuation.tolist() if hasattr(continuation, "tolist") else list(continuation)
    end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if not end_ids:
        raise ValueError("empty thinking delimiter encoding")
    boundaries = [
        i for i in range(len(ids) - len(end_ids) + 1)
        if ids[i:i + len(end_ids)] == end_ids
    ]
    if boundaries:
        boundary = boundaries[0]
        final_start = boundary + len(end_ids)
        reasoning_ids, answer_ids = ids[:boundary], ids[final_start:]
        thinking_completed = True
        fallback_from_reasoning = False
    elif enable_thinking:
                                                                          
                                                                        
        reasoning_ids, answer_ids = ids, ids
        thinking_completed = False
        fallback_from_reasoning = True
    else:
        reasoning_ids, answer_ids = [], ids
        thinking_completed = None
        fallback_from_reasoning = False
    decode = lambda values: tokenizer.decode(
        values, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    ).strip()
    return decode(answer_ids), {
        "completion_tokens": len(ids),
        "reasoning_tokens": len(reasoning_ids),
        "answer_tokens": len(answer_ids),
        "thinking_completed": thinking_completed,
        "fallback_from_reasoning": fallback_from_reasoning,
        "generation_limit_reached": len(ids) >= max_new_tokens,
        "reasoning_text": decode(reasoning_ids),
    }


class LongMemEvalReader(Qwen35LatentReader):
    """Qwen reader using LongMemEval's answer contract."""

    def __init__(self, *args, system_prompt: str = LME_WEB_SYSTEM_PROMPT_OFFICIAL_BYTES, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = system_prompt

    @staticmethod
    def _memory_marker(token_count: int) -> str:
        if token_count < 1:
            raise ValueError("token_count must be positive")
                                                                            
                                                                              
        return "<LME_MEMORY_SLOT>"

    def answer(
        self,
        question: str,
        segments,
        *,
        question_image: str | Path | None = None,
        max_new_tokens: int = 20000,
        enable_thinking: bool = True,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
    ) -> str:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        sampling = reader_sampling_kwargs(temperature=temperature, top_p=top_p, top_k=top_k)
        self.last_generation = {}
        segments = [MemorySegment(**item) if isinstance(item, dict) else item for item in segments]
        if not segments:
            return r"\boxed{UNKNOWN}"
        device = next(self.model.parameters()).device
        pieces, masks, _spans = self._encode_segments(segments, device)
        memory = torch.cat([piece.to(self.model.get_input_embeddings().weight.dtype) for piece in pieces], dim=1)
        memory_mask = torch.cat([mask.to(torch.long) for mask in masks], dim=1)
        marker = self._memory_marker(memory.shape[1])
        user_text = LME_USER_TEMPLATE.format(context=marker, question=question)
        if any(segment.text and '<observation trajectory=' in segment.text for segment in segments):
            user_text = (
                'Each observation block binds its compressed semantic state and sparse exact-value sidecar. '
                'The sidecar deliberately omits common UI semantics; missing sidecar text does not mean an element is absent. '
                'Combine both channels. Steps are chronological only within the same trajectory. '
                'Incoming action describes the transition into that observation; gaps mean intermediate states were not supplied. '
                'Tree positions describe accessibility-tree order, not guaranteed screen coordinates.\n\n'
                + user_text
            )
        if question_image is not None:
                                                                             
                                                                              
                                                                         
                                                
            rendered = self.processor.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": [
                        {"type": "image"}, {"type": "text", "text": user_text}
                    ]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        else:
            rendered = self.processor.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        marker_ids = self.processor.tokenizer(
            marker, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].tolist()
        if not marker_ids:
            raise ValueError("memory marker tokenized to an empty sequence")

        query_image_obj = None
        if question_image is not None:
            with Image.open(Path(question_image)) as handle:
                query_image_obj = handle.convert("RGB")
            inputs = self.processor(
                images=[query_image_obj],
                text=[rendered],
                return_tensors="pt",
                return_mm_token_type_ids=True,
            )
        else:
            inputs = self.processor.tokenizer(
                rendered, return_tensors="pt", add_special_tokens=False
            )
        input_ids = inputs["input_ids"].to(device)
        ids = input_ids[0].tolist()
        hits = [
            i for i in range(len(ids) - len(marker_ids) + 1)
            if ids[i : i + len(marker_ids)] == marker_ids
        ]
        if len(hits) != 1:
            raise ValueError(f"expected one memory marker in rendered prompt, found {hits}")
        start = hits[0]
        end = start + len(marker_ids)
        embed = self.model.get_input_embeddings()
        token_embeds = embed(input_ids)
        prefix = token_embeds[:, :start]
        suffix = token_embeds[:, end:]
        inputs_embeds = torch.cat((prefix, memory, suffix), dim=1)
                                                                             
                                                                             
                                                                             
                                                                             
                                                
        spliced_ids = torch.cat((input_ids[:, :start], input_ids[:, end:]), dim=1)
        attention = inputs["attention_mask"].to(device)
        full_attention = torch.cat((attention[:, :start], memory_mask.to(device), attention[:, end:]), dim=1)
        context_limit = getattr(self.model.config.text_config, "max_position_embeddings", None)
        if context_limit and inputs_embeds.shape[1] + max_new_tokens > context_limit:
            raise ValueError("memory plus generation budget exceeds the reader context limit")

        model_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": full_attention,
            "max_new_tokens": max_new_tokens,
            **sampling,
            "logits_to_keep": 1,
        }
                                                                         
                                                                           
                                                                            
                                                                            
                                                                              
        for name in ("pixel_values", "image_grid_thw"):
            if name in inputs:
                model_kwargs[name] = inputs[name].to(device)
        with torch.inference_mode():
                                                                           
                                                                           
                                                                              
                                                  
            generated = self.model.generate(spliced_ids, **model_kwargs)
                                                                          
                                                                             
                                                                            
                                                           
        continuation = generated[0, spliced_ids.shape[1]:]
        answer, diagnostics = decode_reader_completion(
            self.processor.tokenizer, continuation,
            enable_thinking=enable_thinking, max_new_tokens=max_new_tokens,
        )
        self.last_generation = {
            **diagnostics, "prompt_tokens": int(inputs_embeds.shape[1]),
            "enable_thinking": enable_thinking, "max_new_tokens": max_new_tokens,
            **sampling,
        }
        return answer


def segments_from_cache(xbar: np.ndarray, valid: np.ndarray, anchor_text: str | None = None):
    """Create a reader segment pair for one cached observation."""
    segments = [MemorySegment(latent=(np.asarray(xbar, np.float32), np.asarray(valid, np.bool_)))]
    if anchor_text:
        segments.append(MemorySegment(text=str(anchor_text)))
    return segments
