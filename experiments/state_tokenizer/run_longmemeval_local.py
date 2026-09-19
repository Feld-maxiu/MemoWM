"""Run the local ResidualMem reader on LongMemEval-V2 Web Small.

This is a separate runner, not a modification of LongMemEval's official
OpenAI-compatible harness.  It preserves the official question selection and
haystack mapping, while replacing only the memory context with latent/anchor
segments that cannot be serialized through the remote chat API.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

from residualmem.benchmarks.longmemeval_compact import (
    PROTOCOL as COMPACT_PROTOCOL,
    compact_axtree_text,
)
from residualmem.benchmarks.longmemeval_index import LocalVLQueryEncoder, LongMemEvalLatentIndex, diverse_hits
from residualmem.benchmarks.longmemeval_text_index import (
    LongMemEvalTextIndex,
    TextEmbeddingEncoder,
)
from residualmem.latent.instruct_bridge import MemorySegment

from .longmemeval_reader import SYSTEM_PROMPTS, LongMemEvalReader, reader_sampling_kwargs
from .longmemeval_runtime import LongMemEvalQFormerRuntime


_ENVIRONMENT_NETLOC = {
    "webarena-reddit": "localhost:9080",
    "webarena-onestopshop": "localhost:9082",
    "webarena-cms": "localhost:9083",
}


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def ordered_context(context):
    groups = {}
    for item in context:
        groups.setdefault(item.trajectory_id, []).append(item)
    return [item for group in groups.values() for item in sorted(
        group, key=lambda row: (int(row.record.get("step_idx", row.record_index)), row.record_index))]


def question_netloc(question: dict) -> str | None:
    """Resolve the site namespace mentioned by the benchmark question."""
    environment = str(question.get("environment") or "").strip().casefold()
    resolved = _ENVIRONMENT_NETLOC.get(environment)
    if resolved is not None:
        return resolved
    text = str(question.get("question") or "").casefold()
    if "reddit" in text or "custom forum" in text:
        return "localhost:9080"
    if "one stop market" in text:
        return "localhost:9082"
    if "magento" in text or "shopping admin" in text:
        return "localhost:9083"
    return None


def _memory_segments(context: Iterable, *, include_anchor: bool, framed: bool = False) -> list[MemorySegment]:
    segments: list[MemorySegment] = []
    context = ordered_context(context) if framed else context
    for item in context:
        if framed:
            record = item.record
            header = (
                f'\n<observation trajectory="{html.escape(item.trajectory_id, quote=True)}" '
                f'state="{record.get("state_index", item.record_index)}" '
                f'step="{record.get("step_idx", item.record_index)}">\n'
                f'URL: {record.get("url", "<unknown>")}\n'
                f'Page: {record.get("page_title", "<not supplied>")}\n'
                f'Incoming action (previous state -> this state): '
                f'{record.get("incoming_action_text", record.get("action") or "<none>")}\n'
                'Semantic state (compressed observation):\n'
            )
            segments.append(MemorySegment(text=header))
        segments.append(MemorySegment(latent=(item.xbar, item.valid)))
        if include_anchor and item.anchor_text:
            text = item.anchor_text
            if framed:
                text = '\nSparse exact-value sidecar (not a complete list of UI elements):\n' + text
            segments.append(MemorySegment(text=text))
        if framed:
            segments.append(MemorySegment(text='\n</observation>\n'))
    return segments


STATE_SNAPSHOT_PROTOCOL = "longmemeval-text-state-snapshots-v1"


def text_state_snapshot_memory(hits, index: LongMemEvalLatentIndex):
    """Bind every distinct state in the retrieved windows to its own latent."""
    hits = list(hits)
    snapshots = {}
    source_ranks = {}
    trajectory_info = {}
    for rank, hit in enumerate(hits, 1):
        if (hit.slice_start < 0 or hit.slice_end <= hit.slice_start
                or not hit.slice_start <= hit.center_index < hit.slice_end):
            raise ValueError(f"invalid state window: {hit.trajectory_id}/{hit.center_index}")
                                                                           
                                                                             
                                                                             
        prefix, separator, remainder = hit.context_text.partition("\nFull action sequence\n")
        actions, end_separator, _ = remainder.partition("\nLocal slice action sequence\n")
        if (not separator or not end_separator or not actions.strip()
                or f"- Trajectory: {hit.trajectory_id}" not in prefix.splitlines()
                or f"- Center state index: {hit.center_index}" not in prefix.splitlines()):
            raise ValueError(f"invalid trajectory context: {hit.trajectory_id}/{hit.center_index}")
        info = (hit.goal, actions)
        if hit.trajectory_id in trajectory_info and trajectory_info[hit.trajectory_id] != info:
            raise ValueError(f"conflicting trajectory context: {hit.trajectory_id}")
        trajectory_info[hit.trajectory_id] = info
        for position in range(hit.slice_start, hit.slice_end):
            identity = (hit.trajectory_id, position)
            source_ranks.setdefault(identity, []).append(rank)
            if identity in snapshots:
                continue
            observation = index.observation(*identity)
            if observation is None:
                raise ValueError(f"missing snapshot latent: {identity}")
            if (observation.trajectory_id, observation.record_index) != identity:
                raise ValueError(f"snapshot identity mismatch: {identity}")
            if (observation.record.get("trajectory_id", hit.trajectory_id) != hit.trajectory_id
                    or observation.xbar.ndim != 2
                    or observation.valid.shape != (observation.xbar.shape[0],)
                    or not observation.valid.any()):
                raise ValueError(f"invalid snapshot observation: {identity}")
            snapshots[identity] = replace(
                observation,
                score=hit.score,
                anchor_text=(compact_axtree_text(observation.record)
                             or "No exact-value lines retained for this state."),
            )

    segments = []
    context = []
    if snapshots:
        segments.append(MemorySegment(text=(
            "The observations below are independent sparse snapshots, not deltas. "
            "Each semantic state and its following sidecar belong to that same observation.\n"
        )))
    for trajectory_id, (goal, actions) in trajectory_info.items():
        segments.append(MemorySegment(text=(
            f'\n<trajectory id="{html.escape(trajectory_id, quote=True)}">\n'
            f"Goal: {goal}\nFull action sequence\n{actions}\n"
        )))
        observations = ordered_context(
            row for row in snapshots.values() if row.trajectory_id == trajectory_id
        )
        for observation in observations:
            ranks = source_ranks[(trajectory_id, observation.record_index)]
            segments.append(MemorySegment(text=(
                "\nRetrieved in window ranks: " + ", ".join(map(str, ranks)) + "\n"
            )))
            segments.extend(_memory_segments([observation], include_anchor=True, framed=True))
            context.append(observation)
        segments.append(MemorySegment(text="\n</trajectory>\n"))
    diagnostics = {
        "layout_protocol": STATE_SNAPSHOT_PROTOCOL,
        "retrieved_windows": len(hits),
        "window_state_references": sum(hit.slice_end - hit.slice_start for hit in hits),
        "unique_states": len(context),
        "latent_blocks": len(context),
        "latent_tokens": sum(int(row.xbar.shape[0]) for row in context),
    }
    return segments, context, diagnostics


def _question_image(data_root: Path, value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = data_root / candidate
        if not candidate.exists():
            candidate = data_root / "question_screenshots" / value
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return str(candidate.resolve())


def _load_eval_function():
                                                                        
                                                                           
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "LongMemEval-V2",                                       
        here.parents[3] / "LongMemEval-V2",                            
    ]
    lme_root = next((path for path in candidates if path.is_dir()), candidates[0])
    if str(lme_root) not in sys.path:
        sys.path.insert(0, str(lme_root))
    from evaluation.qa_eval_metrics import (
        eval_from_spec,
        eval_name,
        extract_boxed_answer,
        score_to_bool,
    )
    return eval_from_spec, eval_name, extract_boxed_answer, score_to_bool


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument(
        "--retrieval-mode", choices=("latent", "hybrid", "text"), default="hybrid",
        help="text is the official-style global raw-state AXTree retrieval",
    )
    parser.add_argument("--text-index", type=Path, default=None)
    parser.add_argument(
        "--text-embedding-model", type=Path, default=Path("models/Qwen3-Embedding-8B")
    )
    parser.add_argument(
        "--sidecar-source",
        choices=("stored", "compact"),
        default="stored",
        help=(
            "text payload bound to each latent row: the stored sparse exact-value "
            "anchor, or the rules-1-3 compact exact-value view of the same "
            "observation (latent retrieval keys are unchanged)"
        ),
    )
    parser.add_argument(
        "--latent-payload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "text retrieval mode: prepend the Q-Former latent of each hit as soft "
            "tokens so the reader receives [latent; exact-value text] per retrieved "
            "observation, matching the technical-report read path"
        ),
    )
    parser.add_argument(
        "--text-memory-layout",
        choices=("legacy-windows", "state-snapshots"),
        default="legacy-windows",
        help=(
            "state-snapshots binds each distinct state in compact text windows "
            "to its own latent and independent sidecar; requires --latent-payload"
        ),
    )
    parser.add_argument(
        "--bm25-weight", type=float, default=0.5,
        help="weight of standardized anchor-BM25 score; latent gets 1-weight",
    )
    parser.add_argument(
        "--trajectory-top-k", type=int, default=1,
        help="number of trajectories retained before returning observations",
    )
    parser.add_argument(
        "--trajectory-aggregate-k", type=int, default=3,
        help="top fused observations used to score one trajectory",
    )
    parser.add_argument(
        "--trajectory-prior-weight", type=float, default=0.4,
        help="weight of the goal-summary prior when ranking trajectories",
    )
    parser.add_argument(
        "--trajectory-hub-weight", type=float, default=0.1,
        help="penalty for trajectories with many near-global-best observations",
    )
    parser.add_argument(
        "--site-filter", action=argparse.BooleanOptionalAction, default=True,
        help="restrict candidates to the site named by the question",
    )
    parser.add_argument("--diverse-retrieval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--relevance-weight", type=float, default=.9)
    parser.add_argument("--max-hits-per-trajectory", type=int, default=2)
    parser.add_argument("--structured-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--max-observations", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reader-enable-thinking", action=argparse.BooleanOptionalAction, default=True,
        help="use Qwen thinking mode; legacy replay also needs --reader-temperature 0",
    )
    parser.add_argument(
        "--max-new-tokens", "--max-completion-tokens", type=int, default=20000,
        help="combined reasoning and answer budget, matching the official default",
    )
    parser.add_argument("--reader-temperature", type=float, default=0.6,
                        help="official reader default; 0 explicitly selects legacy greedy decoding")
    parser.add_argument("--reader-top-p", type=float, default=0.95)
    parser.add_argument("--reader-top-k", type=int, default=20,
                        help="token sampling top-k, independent of retrieval --top-k")
    parser.add_argument(
        "--reader-system-prompt",
        choices=tuple(SYSTEM_PROMPTS),
        default="longmemeval",
        help=(
            "answer-time system prompt: 'longmemeval' (default) is "
            "longmemeval_system_prompt, i.e. the official instructions plus the "
            "rule that a flawed premise must be explained in \\boxed{} instead of "
            "answering UNKNOWN; 'official' reproduces the byte sequence the "
            "official harness sends"
        ),
    )
    parser.add_argument(
        "--anchor", action=argparse.BooleanOptionalAction, default=True,
        help="append the deterministic AXTree anchor beside each retrieved latent",
    )
    parser.add_argument(
        "--evaluator-model",
        default="Qwen3.5-9B",
        help="Local/OpenAI-compatible model used for abstention and gotchas judging.",
    )
    parser.add_argument(
        "--evaluator-base-url",
        default="http://127.0.0.1:8023/v1",
        help="OpenAI-compatible base URL for the official LLM judge.",
    )
    parser.add_argument("--evaluator-api-key", default="EMPTY")
    parser.add_argument("--evaluator-max-completion-tokens", type=int, default=2048)
    parser.add_argument("--evaluator-timeout-seconds", type=float, default=43200.0)
    return parser


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        parser.error("require world-size >= 1 and 0 <= rank < world-size")
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    try:
        sampling = reader_sampling_kwargs(
            temperature=args.reader_temperature, top_p=args.reader_top_p, top_k=args.reader_top_k,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.candidate_k < args.top_k:
        parser.error("candidate-k must be at least top-k")
    if not 0.0 <= args.bm25_weight <= 1.0:
        parser.error("bm25-weight must be in [0, 1]")
    if args.trajectory_top_k < 1 or args.trajectory_aggregate_k < 1:
        parser.error("trajectory-top-k and trajectory-aggregate-k must be positive")
    if not 0.0 <= args.trajectory_prior_weight <= 1.0:
        parser.error("trajectory-prior-weight must be in [0, 1]")
    if not 0.0 <= args.trajectory_hub_weight <= 1.0:
        parser.error("trajectory-hub-weight must be in [0, 1]")
    state_snapshots = args.text_memory_layout == "state-snapshots"
    if state_snapshots and not (
        args.retrieval_mode == "text" and args.latent_payload
        and args.anchor and args.structured_memory
    ):
        parser.error("state-snapshots requires text retrieval, latent payload, anchor, and structured memory")
    text_index = None
    if args.retrieval_mode == "text":
        if args.text_index is None or not args.text_index.is_file():
            parser.error("--text-index is required for retrieval-mode=text")
        if not args.text_embedding_model.is_dir():
            parser.error("--text-embedding-model must point to a local model directory")
        if state_snapshots:
            text_index = LongMemEvalTextIndex(args.text_index)
            if text_index.metadata.get("compression") != "compact_exact":
                parser.error("state-snapshots requires a compact_exact text index")
            source = text_index.metadata.get("cache")
            if not isinstance(source, str) or Path(source).resolve() != args.cache.resolve():
                parser.error("state-snapshots text index and latent cache must have the same source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generation = {
        "enable_thinking": args.reader_enable_thinking,
        "max_new_tokens": args.max_new_tokens,
        **sampling,
        "answer_source": "final_only_after_thinking",
    }
    memory_config = {
        "cache": str(args.cache.resolve()), "anchor": args.anchor,
        "structured_memory": args.structured_memory, "candidate_k": args.candidate_k,
        "top_k": args.top_k, "diverse_retrieval": args.diverse_retrieval,
        "retrieval_mode": args.retrieval_mode, "bm25_weight": args.bm25_weight,
        "trajectory_top_k": args.trajectory_top_k,
        "trajectory_aggregate_k": args.trajectory_aggregate_k,
        "trajectory_prior_weight": args.trajectory_prior_weight,
        "trajectory_hub_weight": args.trajectory_hub_weight,
        "site_filter": args.site_filter,
        "text_index": str(args.text_index.resolve()) if args.text_index else None,
        "text_embedding_model": str(args.text_embedding_model.resolve()),
        "sidecar_source": args.sidecar_source,
        "latent_payload": bool(args.latent_payload),
        "relevance_weight": args.relevance_weight,
        "max_hits_per_trajectory": args.max_hits_per_trajectory,
        "radius": args.radius, "max_observations": args.max_observations,
    }
    if state_snapshots:
        memory_config.update({
            "text_memory_layout": args.text_memory_layout,
            "text_memory_layout_protocol": STATE_SNAPSHOT_PROTOCOL,
            "compact_protocol": COMPACT_PROTOCOL,
        })

    questions_path = args.data_root / "questions.jsonl"
    haystack_path = args.data_root / "haystacks" / "lme_v2_small.json"
    questions = [row for row in _jsonl(questions_path) if row.get("domain") == "web"]
    haystacks = json.loads(haystack_path.read_text(encoding="utf-8"))
    if args.limit is not None:
        questions = questions[: args.limit]
    candidate_questions = len(questions)
    questions = [
        question for position, question in enumerate(questions)
        if position % args.world_size == args.rank
    ]
    config = {
        "generation": generation, "memory": memory_config,
        "reader_system_prompt": args.reader_system_prompt,
        "reader_system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPTS[args.reader_system_prompt].encode("utf-8")
        ).hexdigest(),
        "question_ids": [q['id'] for q in questions],
        "model": str(args.model.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "embedding_model": str(
            (args.text_embedding_model if args.retrieval_mode == "text"
             else args.embedding_model).resolve()
        ),
        "text_index": str(args.text_index.resolve()) if args.text_index else None,
        "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
        "evaluator_model": args.evaluator_model,
        "evaluator_base_url": args.evaluator_base_url,
    }
    config_path = args.output.with_suffix('.config.json')
    rows = []
    if args.output.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('resume configuration changed')
        rows = list(_jsonl(args.output))
    else:
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    completed = {row['question_id'] for row in rows}
    if len(completed) != len(rows) or not completed.issubset(set(config['question_ids'])):
        raise ValueError('invalid resumed question coverage')
    index = None
    query_encoder = None
    text_encoder = None
    trajectory_goal_embeddings = None
    if args.retrieval_mode == "text":
        if text_index is None:
            text_index = LongMemEvalTextIndex(args.text_index)
        text_encoder = TextEmbeddingEncoder(
            args.text_embedding_model, device=args.device
        )
        if args.latent_payload:
            index = LongMemEvalLatentIndex(args.cache, sidecar=args.sidecar_source)
    else:
        index = LongMemEvalLatentIndex(args.cache, sidecar=args.sidecar_source)
        query_encoder = LocalVLQueryEncoder(args.embedding_model, device=args.device)
        trajectory_goal_texts = index.trajectory_goal_texts()
        if args.trajectory_prior_weight > 0.0:
            goal_ids = sorted(trajectory_goal_texts)
            goal_vectors = query_encoder.encode_many(
                [trajectory_goal_texts[trajectory_id] for trajectory_id in goal_ids]
            )
            trajectory_goal_embeddings = {
                trajectory_id: goal_vectors[position]
                for position, trajectory_id in enumerate(goal_ids)
            }
    runtime = LongMemEvalQFormerRuntime(
        model_path=args.model, checkpoint=args.checkpoint, device=args.device
    )
    reader = LongMemEvalReader(
        runtime.model, runtime.processor, runtime.reader.connector, mode="input",
        system_prompt=SYSTEM_PROMPTS[args.reader_system_prompt],
    )
    eval_from_spec, eval_name, extract_boxed_answer, score_to_bool = _load_eval_function()
    for number, question in enumerate(questions, 1):
        qid = str(question["id"])
        if qid in completed:
            continue
        allowed = haystacks.get(qid)
        if not isinstance(allowed, list) or not allowed:
            raise ValueError(f"{qid}: missing official haystack")
        image = _question_image(args.data_root, question.get("image"))
        question_text = str(question["question"])
        started = time.perf_counter()
        requested_site = None
        effective_site = None
        memory_diagnostics = None
        if args.retrieval_mode == "text":
            query_key = text_encoder.encode_query(question_text)
            hits = text_index.query(
                query_key, trajectory_ids=allowed, top_k=args.top_k
            )
            context = hits
            context_observations = len(hits)
            context_ids = [[hit.trajectory_id, hit.center_index] for hit in hits]
            hits_json = [
                {
                    "trajectory_id": hit.trajectory_id,
                    "record_index": hit.center_index,
                    "score": hit.score,
                    "text_score": hit.score,
                }
                for hit in hits
            ]
            context_blocks = []
            for rank, hit in enumerate(hits, 1):
                context_blocks.append(
                    f"### Raw state result {rank}\n"
                    f"- Retrieval rank: {rank}\n"
                    f"- Similarity: {hit.score:.4f}\n\n"
                    f"{hit.context_text}"
                )
            if state_snapshots:
                segments, context, memory_diagnostics = text_state_snapshot_memory(hits, index)
                context_observations = len(context)
                context_ids = [[item.trajectory_id, item.record_index] for item in context]
            elif args.latent_payload:
                                                                           
                                                                               
                                                                       
                segments = []
                for rank, hit in enumerate(hits, 1):
                    segments.append(MemorySegment(
                        text=(
                            f"### Raw state result {rank}\n"
                            f"- Retrieval rank: {rank}\n"
                            f"- Similarity: {hit.score:.4f}\n\n"
                        )
                    ))
                    observation = index.observation(hit.trajectory_id, hit.center_index)
                    if observation is not None:
                        segments.append(MemorySegment(
                            latent=(observation.xbar, observation.valid)
                        ))
                    segments.append(MemorySegment(text=hit.context_text))
            else:
                segments = (
                    [MemorySegment(text="\n\n".join(context_blocks))]
                    if context_blocks else []
                )
        else:
            query_key = query_encoder.encode(question_text, image=image)
            requested_site = question_netloc(question) if args.site_filter else None
            effective_site = requested_site
            if args.retrieval_mode == "hybrid":
                if effective_site is not None:
                    site_trajectories = index.trajectory_ids_for_netloc(effective_site)
                    scoped = [trajectory for trajectory in allowed if trajectory in site_trajectories]
                    if not scoped:
                        effective_site = None
                    else:
                        allowed = scoped
                hits = index.query_hybrid(
                    query_key, question_text,
                    trajectory_ids=allowed, site=effective_site,
                    top_k=args.top_k, bm25_weight=args.bm25_weight,
                    trajectory_top_k=args.trajectory_top_k,
                    trajectory_aggregate_k=args.trajectory_aggregate_k,
                    trajectory_prior=(
                        {
                            trajectory_id: float(query_key @ embedding)
                            for trajectory_id, embedding in trajectory_goal_embeddings.items()
                        }
                        if trajectory_goal_embeddings is not None else None
                    ),
                    trajectory_prior_weight=args.trajectory_prior_weight,
                    trajectory_hub_weight=args.trajectory_hub_weight,
                )
            else:
                candidates = index.query(query_key, trajectory_ids=allowed, top_k=args.candidate_k)
                hits = (diverse_hits(candidates, top_k=args.top_k, relevance_weight=args.relevance_weight,
                                    max_per_trajectory=args.max_hits_per_trajectory,
                                    min_index_distance=2 * args.radius)
                        if args.diverse_retrieval else candidates[:args.top_k])
            context = index.expanded_context(
                hits, radius=args.radius, max_observations=args.max_observations
            )
            if args.structured_memory:
                context = ordered_context(context)
            segments = _memory_segments(context, include_anchor=args.anchor, framed=args.structured_memory)
            context_observations = len(context)
            context_ids = [[item.trajectory_id, item.record_index] for item in context]
            hits_json = [
                {"trajectory_id": item.trajectory_id,
                 "record_index": item.record_index,
                 "score": item.score,
                 "latent_score": item.latent_score,
                 "bm25_score": item.bm25_score}
                for item in hits
            ]
        print(f'[lme-start] {number}/{len(questions)} {qid} observations={len(context)}', flush=True)
        raw = reader.answer(
            question_text, segments,
            question_image=image,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.reader_enable_thinking,
            temperature=args.reader_temperature,
            top_p=args.reader_top_p,
            top_k=args.reader_top_k,
        )
        elapsed = time.perf_counter() - started
        score = None
        score_error = None
        try:
            scorer_name = eval_name(question["eval_function"])
            evaluator_kwargs = {}
            if scorer_name in {"llm_abstention_checker", "llm_gotchas_checker"}:
                evaluator_kwargs = {
                    "question_item": question,
                    "parsed_prediction": extract_boxed_answer(raw),
                    "model_response": raw,
                    "evaluator_model": args.evaluator_model,
                    "evaluator_base_url": args.evaluator_base_url,
                    "evaluator_api_key": args.evaluator_api_key,
                    "evaluator_max_completion_tokens": (
                        args.evaluator_max_completion_tokens
                    ),
                    "evaluator_timeout_seconds": args.evaluator_timeout_seconds,
                }
            value = eval_from_spec(
                question["eval_function"],
                extract_boxed_answer(raw),
                question["answer"],
                **evaluator_kwargs,
            )
            score = bool(score_to_bool(value))
        except Exception as exc:
            score_error = str(exc)
        result = {
            "question_id": qid,
            "question_type": question.get("question_type"),
            "eval_name": eval_name(question["eval_function"]),
            "question": question["question"],
            "answer_gold": question["answer"],
            "response_raw": raw,
            "generation": generation,
            "memory_config": memory_config,
            "generation_diagnostics": reader.last_generation,
            "score": score,
            "score_error": score_error,
            "query_seconds": elapsed,
            "hit_count": len(hits),
            "context_observations": context_observations,
            "context_ids": context_ids,
            "retrieval_diagnostics": {
                "mode": args.retrieval_mode,
                "requested_site": requested_site,
                "effective_site": effective_site,
                "selected_trajectories": sorted({item.trajectory_id for item in hits}),
                "text_index": str(args.text_index.resolve()) if args.text_index else None,
            },
            "hits": hits_json,
        }
        if memory_diagnostics is not None:
            result["memory_diagnostics"] = memory_diagnostics
        rows.append(result)
        with args.output.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(rows[-1], ensure_ascii=False) + '\n')
            handle.flush()
        error_suffix = f" error={score_error}" if score_error else ""
        print(
            f"[lme] {number}/{len(questions)} {qid} score={score} "
            f"hits={len(hits)}{error_suffix}",
            flush=True,
        )
    scored = [row["score"] for row in rows if isinstance(row["score"], bool)]
    summary = {
        "protocol": "longmemeval-local-run-v2",
        "domain": "web",
        "tier": "small",
        "questions": len(rows),
        "candidate_questions": candidate_questions,
        "rank": args.rank,
        "world_size": args.world_size,
        "scored_questions": len(scored),
        "accuracy": float(np.mean(scored)) if scored else None,
        "anchor_enabled": bool(args.anchor),
        "generation": generation,
        "memory": memory_config,
        "cache": str(args.cache.resolve()),
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
