"""Run the frozen QFormer -> retrieval -> Bridge -> Qwen3-32B AMA-WEB path.

The command is intentionally split into two processes. ``cache-shard`` keeps
the 9B document encoder and query embedder on one GPU and writes resumable,
per-episode caches. ``answer`` then loads the 32B Reader across the three
visible GPUs after the cache workers have exited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from xt_ama_adapter.adapter import adapt_ama_trajectory
from xt_ama_adapter.axtree_clean import (
    ANCHOR_STYLE_ELEMENT,
    ANCHOR_STYLE_IDS,
    AXTREE_CLEAN_PROTOCOL,
    clean_trajectory,
    element_lines,
    ids_from,
)
from xt_ama_adapter.qwen32_bridge import (
    Qwen32LatentReader,
    file_sha256,
    latent_position_ids,
    load_qwen32_bridge,
    render_ama_openend_parts,
)
from xt_ama_adapter.runtime import (
    QFormerRuntimeConfig,
    QFormerXTDocumentEncoder,
    QwenVLQueryEncoder,
    validate_artifact_pair,
)

CACHE_PROTOCOL = "ama_web_qformer_latent_cache_v2"
# v1 payloads predate step_texts; they stay readable for latent-only and
# question-only reruns but cannot feed text memory rows.
CACHE_PROTOCOL_V1 = "ama_web_qformer_latent_cache_v1"
# v3 payloads encode AXTree observations that were passed through the
# deterministic R1-R4 redundancy filter (axtree_clean). Same schema as v2, so
# text-bearing arms work on it unchanged; the protocol string isolates the arm.
CLEAN_CACHE_PROTOCOL = "ama_web_qformer_latent_cache_v3-clean"
ANSWER_PROTOCOL = "ama_web_qwen32_latent_answers_v1"
# Memory modes whose prompts carry step-text rows (latent+anchor appends the
# verbatim element anchor of the injected rows); these require step_texts in
# the cache, i.e. protocol v2/v3.
TEXT_MEMORY_MODES = ("text-only", "matched", "shuffled", "latent+anchor")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class BM25Index:
    """Minimal per-episode BM25 (k1=1.2, b=0.75) over lowercase alnum tokens.

    Built once per memory episode from its ``step_texts``; ``score(query)``
    returns one BM25 score per memory row so keyword hits (element names,
    actions, step numbers, verbatim strings the question quotes) can be fused
    with the semantic retrieval channel.
    """

    _K1 = 1.2
    _B = 0.75

    def __init__(self, texts: list[str]):
        self.n = len(texts)
        self._tfs: list[Counter] = []
        self._lens: list[int] = []
        total = 0
        for text in texts:
            toks = _TOKEN_RE.findall((text or "").lower())
            self._tfs.append(Counter(toks))
            self._lens.append(len(toks))
            total += len(toks)
        self.avgdl = total / max(1, self.n)
        df: Counter = Counter()
        for counter in self._tfs:
            df.update(counter.keys())
        self._idf = {
            tok: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for tok, freq in df.items()
        }

    def score(self, query: str) -> np.ndarray:
        terms = _TOKEN_RE.findall((query or "").lower())
        out = np.zeros(self.n, dtype=np.float64)
        if not terms or self.n == 0:
            return out
        for term in set(terms):
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for index, counter in enumerate(self._tfs):
                tf = counter.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self._K1 * (
                    1 - self._B + self._B * self._lens[index] / self.avgdl
                )
                out[index] += idf * tf * (self._K1 + 1) / denom
        return out


def _rrf_merge(semantic_order: np.ndarray, lexical_order: np.ndarray,
               k: int = 60) -> list[int]:
    """Reciprocal-rank fusion of two descending index orderings."""
    fused: dict[int, float] = {}
    for rank, index in enumerate(semantic_order.tolist()):
        fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank + 1)
    for rank, index in enumerate(lexical_order.tolist()):
        fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=lambda index: -fused[index])


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def web_episodes(path: Path) -> list[dict[str, Any]]:
    rows = [row for row in _jsonl(path) if str(row.get("domain", "")).upper() == "WEB"]
    rows.sort(key=lambda row: int(row["episode_id"]))
    if not rows:
        raise ValueError("AMA test file contains no WEB episodes")
    return rows


def episodes_in_domains(path: Path, domains: Sequence[str]) -> list[dict[str, Any]]:
    """Every episode whose domain is in ``domains``, in episode order.

    The memory study needs the whole benchmark, not just the web arm: the
    official questions of every episode are the only distribution the bridge and
    reader were trained on, so utility fitted on a web-only caches would not
    cover the corpus the gate is applied to.
    """
    wanted = {str(domain).upper() for domain in domains}
    rows = [row for row in _jsonl(path)
            if str(row.get("domain", "")).upper() in wanted]
    rows.sort(key=lambda row: int(row["episode_id"]))
    if not rows:
        raise ValueError(f"AMA test file contains no episodes in {sorted(wanted)}")
    return rows


def shard_episodes(rows: list[dict[str, Any]], index: int, count: int):
    if count < 1 or index < 0 or index >= count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")
    return [row for offset, row in enumerate(rows) if offset % count == index]


def _episode_ids(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _shuffled_episode_partners(
    episode_ids: list[int], seed: int = 35,
) -> dict[int, int]:
    """Return a deterministic balanced derangement of episode IDs."""
    if len(episode_ids) < 2:
        raise ValueError("shuffled memory requires at least two episodes")
    order = np.random.default_rng(seed).permutation(
        np.asarray(sorted(episode_ids), dtype=np.int64)
    ).tolist()
    return {
        int(episode_id): int(order[(index + 1) % len(order)])
        for index, episode_id in enumerate(order)
    }


def _atomic_torch_save(payload: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, target)


EMPTY_STEP_TEXT = "(empty observation)"


def _fill_empty_steps(trajectory: Any) -> list[Any]:
    """Give turns that carry neither action nor observation a placeholder.

    51 episodes (mostly spider2) contain such turns. ``adapt_ama_step`` rejects
    them, which would leave those episodes without a cache and silently drop
    them from the utility corpus. The world-model builder already substitutes
    this exact placeholder, so the reader cache stays consistent with the state
    texts the gate is fitted on.
    """
    filled = []
    for turn in trajectory:
        if isinstance(turn, Mapping):
            action = str(turn.get("action", "") or "").strip()
            observation = str(turn.get("observation", "") or "").strip()
            if not action and not observation:
                turn = {**turn, "observation": EMPTY_STEP_TEXT}
        filled.append(turn)
    return filled


def cache_shard(args: argparse.Namespace) -> None:
    test_path = Path(args.test_file).resolve()
    qformer_path = Path(args.qformer).resolve()
    head_path = Path(args.retrieval_head).resolve()
    output = Path(args.output_dir).resolve()
    config = QFormerRuntimeConfig(
        residualmem_root=str(Path(args.residualmem_root).resolve()),
        qwen35_model_path=str(Path(args.qwen35_model).resolve()),
        qformer_checkpoint=str(qformer_path),
        retrieval_head_checkpoint=str(head_path),
        query_model_path=str(Path(args.query_model).resolve()),
        device=args.device,
        queries=32,
        qk_norm=True,
        self_attention=False,
        layer=16,
        max_length=args.max_length,
        # AMA episodes reach 43k characters; the cache stage keeps them by
        # truncating the DOM (audit recorded per step in truncated_steps).
        allow_truncate=True,
    )
    validated = validate_artifact_pair(config)
    test_hash = file_sha256(test_path)
    qformer_hash = file_sha256(qformer_path)
    head_hash = file_sha256(head_path)
    skipped = _episode_ids(args.skip_episode_ids)
    clean_mode = args.clean_axtree
    cache_protocol = CLEAN_CACHE_PROTOCOL if clean_mode != "none" else CACHE_PROTOCOL
    eligible = [row for row in episodes_in_domains(test_path, args.domains)
                if int(row["episode_id"]) not in skipped]
    rows = shard_episodes(eligible, args.shard_index, args.shard_count)
    print(json.dumps({
        "event": "cache_shard_start", "shard": args.shard_index,
        "episodes": len(rows), "test_sha256": test_hash,
        "skipped_episode_ids": sorted(skipped), **validated,
        "cache_protocol": cache_protocol, "axtree_clean": clean_mode,
    }, sort_keys=True), flush=True)
    document_encoder = QFormerXTDocumentEncoder(config)
    query_encoder = QwenVLQueryEncoder(config)
    started = time.time()
    for offset, episode in enumerate(rows, 1):
        episode_id = int(episode["episode_id"])
        target = output / f"episode-{episode_id:06d}.pt"
        if target.exists() and not args.overwrite:
            existing = torch.load(target, map_location="cpu", weights_only=True)
            metadata = existing.get("metadata") or {}
            if (existing.get("protocol") == cache_protocol
                    and metadata.get("test_sha256") == test_hash
                    and metadata.get("qformer_sha256") == qformer_hash
                    and metadata.get("retrieval_head_sha256") == head_hash
                    and metadata.get("axtree_clean", "none") == clean_mode):
                print(json.dumps({"event": "cache_skip", "episode_id": episode_id}), flush=True)
                continue
            raise ValueError(f"stale cache exists: {target}")
        trajectory = _fill_empty_steps(episode["trajectory"])
        clean_audit = None
        if clean_mode != "none":
            trajectory, clean_audit = clean_trajectory(episode, clean_mode)
        records = adapt_ama_trajectory(
            trajectory, episode_id=str(episode_id), task=episode.get("task", "")
        )
        doc_vectors, latents, truncation_flags = document_encoder.encode_with_latents(records)
        questions = [str(pair["question"]) for pair in episode["qa_pairs"]]
        query_vectors = query_encoder.encode(questions)
        xbar = torch.stack([torch.from_numpy(value) for value, _ in latents]).to(torch.bfloat16)
        valid = torch.stack([torch.from_numpy(value) for _, value in latents]).bool()
        truncated_steps = [int(record.step_index)
                           for record, flag in zip(records, truncation_flags) if flag]
        payload = {
            "protocol": cache_protocol,
            "metadata": {
                "episode_id": episode_id,
                "domain": str(episode.get("domain", "")),
                "test_sha256": test_hash,
                "qformer_sha256": qformer_hash,
                "retrieval_head_sha256": head_hash,
                "steps": len(records),
                "questions": len(questions),
                "truncated_steps": truncated_steps,
                "axtree_clean": clean_mode,
                "axtree_clean_protocol": (AXTREE_CLEAN_PROTOCOL
                                          if clean_audit else None),
                "axtree_clean_audit": clean_audit,
            },
            "step_indices": torch.tensor([record.step_index for record in records]),
            "step_texts": [record.step_text for record in records],
            "document_embeddings": torch.from_numpy(doc_vectors),
            "xbar": xbar,
            "valid": valid,
            "query_embeddings": torch.from_numpy(query_vectors),
            "questions": questions,
            "gold_answers": [str(pair.get("answer", "")) for pair in episode["qa_pairs"]],
            "qa_types": [pair.get("type") for pair in episode["qa_pairs"]],
            "task": str(episode.get("task", "")),
            "task_type": str(episode.get("task_type", "")),
        }
        _atomic_torch_save(payload, target)
        event = {
            "event": "cache_episode_complete", "episode_id": episode_id,
            "episode": offset, "episode_total": len(rows), "steps": len(records),
            "questions": len(questions), "truncated_steps": len(truncated_steps),
            "elapsed_seconds": round(time.time() - started, 1),
        }
        if clean_audit:
            event["axtree_clean_audit"] = clean_audit
        print(json.dumps(event, sort_keys=True), flush=True)
    print(json.dumps({"event": "cache_shard_complete", "shard": args.shard_index,
                      "elapsed_seconds": round(time.time() - started, 1)}), flush=True)


def _clean_answer(text: str) -> str:
    value = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"Answer\[1\]:\s*(.+)", value, flags=re.DOTALL)
    return (match.group(1) if match else value).strip()


def _cap_anchor_name(line: str, max_name_chars: int) -> str:
    """Truncate a long quoted name inside an ``[id] role 'name'`` anchor row.

    Id and role are never touched; if the quote pair cannot be located the row
    is returned unchanged (safer to keep full text than to corrupt the row).
    """
    if max_name_chars <= 0 or len(line) <= max_name_chars:
        return line
    first = line.find("'")
    last = line.rfind("'")
    if first < 0 or last <= first:
        return line
    name = line[first + 1:last]
    if len(name) <= max_name_chars:
        return line
    return f"{line[:first + 1]}{name[:max_name_chars]}…{line[last:]}"


def build_anchor_block(
    screens: list[tuple[int | None, str]],
    *,
    style: str = ANCHOR_STYLE_ELEMENT,
    max_lines_per_screen: int = 300,
    max_name_chars: int = 160,
    max_total_chars: int = 24000,
) -> tuple[str, dict]:
    """Verbatim anchor text for the injected top-k screens.

    ``screens`` is rank-ordered ``(step_index, step_text)``.  ``element`` style
    emits ``[id] role 'name'`` rows for anchor-eligible nodes (P1 granularity);
    ``ids`` emits only the ``[id]`` tokens (P2).  Lower-rank screens are dropped
    first when the total would exceed ``max_total_chars`` so the highest-ranked
    screens always stay complete.
    """
    if style not in (ANCHOR_STYLE_ELEMENT, ANCHOR_STYLE_IDS):
        raise ValueError(f"unknown anchor style: {style!r}")
    parts: list[str] = []
    per_screen: list[dict] = []
    total = 0
    dropped_screens = 0
    for step_index, text in screens:
        if style == ANCHOR_STYLE_ELEMENT:
            rows = [_cap_anchor_name(row, max_name_chars)
                    for row in element_lines(text)]
        else:
            rows = ids_from(text)
        rows = rows[:max_lines_per_screen]
        step_label = f"step {step_index}" if step_index is not None else "rank"
        block = f"Retrieved screen ({step_label}):\n" + "\n".join(rows)
        if parts and total + len(block) > max_total_chars:
            dropped_screens += 1
            continue
        parts.append(block)
        total += len(block)
        per_screen.append({"step_index": step_index, "rows": len(rows)})
    return "\n\n".join(parts), {
        "style": style,
        "screens": len(parts),
        "rows_total": sum(item["rows"] for item in per_screen),
        "chars": total,
        "screens_dropped": dropped_screens,
    }


def _left_pad_reader_inputs(rows: list[dict[str, torch.Tensor]]):
    width = max(row["inputs_embeds"].shape[1] for row in rows)
    model_width = rows[0]["inputs_embeds"].shape[-1]
    device = rows[0]["inputs_embeds"].device
    dtype = rows[0]["inputs_embeds"].dtype
    embeds, masks = [], []
    for row in rows:
        length = row["inputs_embeds"].shape[1]
        pad = width - length
        embeds.append(torch.cat((
            torch.zeros((1, pad, model_width), device=device, dtype=dtype),
            row["inputs_embeds"],
        ), dim=1))
        masks.append(torch.cat((
            torch.zeros((1, pad), device=device, dtype=torch.bool),
            row["attention_mask"].bool(),
        ), dim=1))
    attention_mask = torch.cat(masks, dim=0)
    return {
        "inputs_embeds": torch.cat(embeds, dim=0),
        "attention_mask": attention_mask,
        "position_ids": latent_position_ids(attention_mask),
    }


def _question_only_inputs(reader: Qwen32LatentReader, question: str):
    prefix, suffix, _ = render_ama_openend_parts(
        reader.tokenizer, question, enable_thinking=reader.enable_thinking
    )
    token_ids = reader.tokenizer.encode(prefix, add_special_tokens=False)
    token_ids += reader.tokenizer.encode(suffix, add_special_tokens=False)
    embedding = reader.model.get_input_embeddings()
    ids = torch.tensor([token_ids], dtype=torch.long, device=embedding.weight.device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    return {
        "inputs_embeds": embedding(ids),
        "attention_mask": mask,
        "position_ids": latent_position_ids(mask),
    }


def _load_done(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    return {(int(row["episode_id"]), int(row["qa_index"])): row for row in _jsonl(path)}


def _load_cache_payloads(files, *, qformer_hash: str | None = None,
                         head_hash: str | None = None,
                         needs_texts: bool = False) -> dict[int, dict[str, Any]]:
    """Load per-episode caches, refusing any coordinate or protocol mismatch.

    Diagnosis callers may omit the expected hashes to inspect caches without
    knowing the producing artifacts; the answer path always passes both.
    """
    cache_payloads: dict[int, dict[str, Any]] = {}
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        protocol = payload.get("protocol")
        if protocol not in (CACHE_PROTOCOL, CACHE_PROTOCOL_V1, CLEAN_CACHE_PROTOCOL):
            raise ValueError(f"unexpected cache protocol: {path}")
        if needs_texts and (protocol not in (CACHE_PROTOCOL, CLEAN_CACHE_PROTOCOL)
                            or "step_texts" not in payload):
            raise ValueError(
                f"cache {path.name} lacks step_texts required for text memory rows; "
                f"re-run cache-shard to build {CACHE_PROTOCOL} or "
                f"{CLEAN_CACHE_PROTOCOL}"
            )
        episode_id = int(payload["metadata"]["episode_id"])
        if qformer_hash is not None and payload["metadata"]["qformer_sha256"] != qformer_hash:
            raise ValueError(f"QFormer cache mismatch: {path}")
        if head_hash is not None and payload["metadata"]["retrieval_head_sha256"] != head_hash:
            raise ValueError(f"retrieval-head cache mismatch: {path}")
        cache_payloads[episode_id] = payload
    return cache_payloads


def answer(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = Path(args.cache_dir).resolve()
    files = sorted(cache_dir.glob("episode-*.pt"))
    skipped = _episode_ids(args.skip_episode_ids)
    expected = sum(int(row["episode_id"]) not in skipped
                   for row in web_episodes(Path(args.test_file)))
    if len(files) != expected and not args.allow_partial_cache:
        raise ValueError(f"expected {expected} WEB episode caches, found {len(files)}")
    qformer_hash = file_sha256(args.qformer)
    head_hash = file_sha256(args.retrieval_head)
    bridge_hash = file_sha256(args.bridge)
    bridge, bridge_metadata = load_qwen32_bridge(
        args.bridge, expected_qformer_sha256=qformer_hash,
        expected_retrieval_head_sha256=head_hash,
    )
    max_ranks = int(bridge_metadata["max_memory_ranks"])
    if args.top_k_latent > max_ranks:
        print(json.dumps({"event": "clamp_top_k_latent",
                          "requested": args.top_k_latent, "clamped": max_ranks}),
              flush=True)
        args.top_k_latent = max_ranks
    tokenizer = AutoTokenizer.from_pretrained(args.reader_model, trust_remote_code=True)
    _, _, prompt_hash = render_ama_openend_parts(
        tokenizer, "{QUESTION}", enable_thinking=bool(bridge_metadata["enable_thinking"])
    )
    if prompt_hash != bridge_metadata["prompt_sha256"]:
        raise ValueError("Reader prompt does not match the trained bridge")
    if bool(args.enable_thinking) != bool(bridge_metadata["enable_thinking"]):
        raise ValueError(
            f"--enable-thinking={args.enable_thinking} does not match the bridge "
            f"training setting (enable_thinking={bool(bridge_metadata['enable_thinking'])})"
        )
    max_memory = {index: f"{args.max_memory_gib}GiB" for index in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        args.reader_model, torch_dtype=torch.bfloat16, device_map="balanced",
        max_memory=max_memory, low_cpu_mem_usage=True, trust_remote_code=True,
    ).eval()
    model.config.use_cache = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    reader = Qwen32LatentReader(
        model, tokenizer, bridge,
        enable_thinking=bool(args.enable_thinking),
    )
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(checkpoint)
    needs_texts = (args.memory_mode in TEXT_MEMORY_MODES
                   or args.retrieval != "semantic")
    cache_payloads = _load_cache_payloads(
        files, qformer_hash=qformer_hash, head_hash=head_hash,
        needs_texts=needs_texts,
    )

    # Keyword (BM25) channel over each memory episode's step texts.  Built
    # once per episode; only rows' orderings (not their latent embeddings) are
    # affected when ``--retrieval hybrid|lexical`` is requested.
    lexical_indexes = {}
    if args.retrieval != "semantic":
        for episode_id, payload in cache_payloads.items():
            lexical_indexes[episode_id] = BM25Index(
                [str(text) for text in payload["step_texts"]]
            )

    shuffled_partners = (
        _shuffled_episode_partners(list(cache_payloads), seed=args.shuffle_seed)
        if args.memory_mode == "shuffled" else {}
    )
    work = []
    for episode_id in sorted(cache_payloads):
        payload = cache_payloads[episode_id]
        queries = payload["query_embeddings"].float().numpy()
        for qa_index, question in enumerate(payload["questions"]):
            key = (episode_id, qa_index)
            if key in done:
                continue
            if args.memory_mode == "question-only":
                work.append((episode_id, qa_index, question, None, ()))
                continue
            source_episode_id = (
                shuffled_partners[episode_id]
                if args.memory_mode == "shuffled" else episode_id
            )
            source_payload = cache_payloads[source_episode_id]
            docs = source_payload["document_embeddings"].float().numpy()
            scores = docs @ queries[qa_index]
            if args.memory_mode == "latent-only":
                # Inject the top-``top_k_latent`` latent memory rows so the
                # evaluation can probe any k (not just top-1).  Retrieval is
                # still ranked by the document@query score; the bridge must
                # have been trained with max_memory_ranks >= top_k_latent.
                needed_latent, needed_text = args.top_k_latent, 0
            elif args.memory_mode == "latent+anchor":
                # Same retrieval and latent rows as latent-only, plus a
                # verbatim element anchor (``[id] role 'name'``) of those same
                # rows so the reader can copy exact element IDs / text instead
                # of recalling them from lossy latents.  The anchor is a single
                # text block and does not consume memory-rank slots.
                needed_latent, needed_text = args.top_k_latent, 0
            elif args.memory_mode == "text-only":
                needed_latent, needed_text = 0, args.top_k
            else:  # matched / shuffled
                needed_latent, needed_text = args.top_k_latent, args.top_k
            needed = max(needed_latent, needed_text)
            if args.retrieval == "semantic":
                order = np.argsort(-scores)
            else:
                # Keyword (BM25) channel over the step texts, and optional RRF
                # fusion.  ``ranked`` always carries the *semantic* score per
                # row for audits; only the row ordering is changed by the
                # lexical channel.
                lexical = lexical_indexes[source_episode_id]
                lex_scores = lexical.score(question)
                if args.retrieval == "hybrid":
                    order = np.asarray(
                        _rrf_merge(np.argsort(-scores), np.argsort(-lex_scores))
                    )
                else:  # lexical-only
                    order = np.argsort(-lex_scores)
            order = order[:needed]
            ranked = tuple((int(index), float(scores[index])) for index in order)
            work.append((episode_id, qa_index, question, source_episode_id, ranked))
    if args.limit:
        work = work[:args.limit]
    print(json.dumps({"event": "answer_start", "pending": len(work),
                      "already_complete": len(done), "batch_size": args.batch_size,
                      "memory_mode": args.memory_mode, "top_k": args.top_k,
                      "top_k_latent": args.top_k_latent,
                      "enable_thinking": bool(args.enable_thinking),
                      "max_model_len": args.max_model_len,
                      "max_new_tokens": args.max_new_tokens,
                      "bridge_sha256": bridge_hash}), flush=True)
    started = time.time()
    with checkpoint.open("a", encoding="utf-8") as handle:
        for start in range(0, len(work), args.batch_size):
            batch = work[start:start + args.batch_size]
            inputs = []
            audits = []
            for (episode_id, _qa_index, question, source_episode_id,
                 ranked) in batch:
                if args.memory_mode == "question-only":
                    inputs.append(_question_only_inputs(reader, question))
                    audits.append({"prompt_truncated": False})
                    continue
                source_payload = cache_payloads[source_episode_id]
                if args.memory_mode == "latent-only":
                    # Inject the top-``top_k_latent`` rows (multi-rank latent).
                    segments = [
                        ("latent", (
                            source_payload["xbar"][position].float(),
                            source_payload["valid"][position],
                        ))
                        for position, _score in ranked[:args.top_k_latent]
                    ]
                elif args.memory_mode == "latent+anchor":
                    # Latent rows identical to latent-only, then one verbatim
                    # element anchor block built from the SAME rows' step text.
                    positions = [position for position, _score in ranked[:args.top_k_latent]]
                    segments = [
                        ("latent", (
                            source_payload["xbar"][position].float(),
                            source_payload["valid"][position],
                        ))
                        for position in positions
                    ]
                    anchor_text, anchor_stats = build_anchor_block(
                        [(int(source_payload["step_indices"][position]),
                          str(source_payload["step_texts"][position]))
                         for position in positions],
                        style=args.anchor_style,
                    )
                    if anchor_text:
                        segments.append(("text", anchor_text))
                elif args.memory_mode == "text-only":
                    segments = [("text", source_payload["step_texts"][position])
                                for position, _score in ranked]
                else:  # matched / shuffled: latent rows then text rows, rank order
                    segments = [
                        ("latent", (
                            source_payload["xbar"][position].float(),
                            source_payload["valid"][position],
                        ))
                        for position, _score in ranked[:args.top_k_latent]
                    ] + [
                        ("text", source_payload["step_texts"][position])
                        for position, _score in ranked[:args.top_k]
                    ]
                built, audit = reader.build_inputs_segments(
                    question, segments, max_model_len=args.max_model_len,
                    max_new_tokens=args.max_new_tokens,
                )
                if args.memory_mode == "latent+anchor":
                    audit = dict(audit)
                    audit["anchor"] = anchor_stats
                inputs.append(built)
                audits.append(audit)
            kwargs = _left_pad_reader_inputs(inputs)
            # ``position_ids`` are required for the teacher-forced training
            # forward, but passing the complete prompt positions into
            # ``generate`` can leave some Transformers/Qwen versions reusing
            # stale positions during cached decoding.  The generation helper
            # correctly derives and advances them from this attention mask.
            kwargs.pop("position_ids", None)
            with torch.inference_mode():
                generated = model.generate(
                    **kwargs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    use_cache=True, eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                                  else tokenizer.eos_token_id),
                )
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            for offset, (item, token_ids) in enumerate(zip(batch, sequences)):
                (episode_id, qa_index, question, source_episode_id, ranked) = item
                if len(token_ids) > args.max_new_tokens:
                    token_ids = token_ids[-args.max_new_tokens:]
                prediction = _clean_answer(tokenizer.decode(token_ids, skip_special_tokens=True))
                payload = cache_payloads[episode_id]
                record = {
                    "protocol": ANSWER_PROTOCOL, "episode_id": episode_id,
                    "bridge_sha256": bridge_hash,
                    "memory_mode": args.memory_mode,
                    "memory_source_episode_id": source_episode_id,
                    "qa_index": qa_index, "question": question,
                    "predicted_answer": prediction,
                    "golden_answer": payload["gold_answers"][qa_index],
                    "qa_type": payload["qa_types"][qa_index],
                    "retrieved_step_indices": [
                        int(cache_payloads[source_episode_id]["step_indices"][position])
                        for position, _score in ranked
                    ] if source_episode_id is not None else [],
                    "retrieval_scores": [score for _position, score in ranked],
                    "prompt_truncated": bool(
                        audits[offset].get("prompt_truncated", False)
                    ),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                done[(episode_id, qa_index)] = record
            print(json.dumps({
                "event": "answer_progress", "complete": len(done),
                "batch_complete": min(start + len(batch), len(work)),
                "batch_total": len(work), "elapsed_seconds": round(time.time() - started, 1),
            }), flush=True)
    if not args.limit and len(files) == expected:
        answers_path = Path(args.answers_output).resolve()
        answers_path.parent.mkdir(parents=True, exist_ok=True)
        with answers_path.open("w", encoding="utf-8") as handle:
            for episode_id in sorted(cache_payloads):
                payload = cache_payloads[episode_id]
                answer_list = [done[(episode_id, index)]["predicted_answer"]
                               for index in range(len(payload["questions"]))]
                handle.write(json.dumps({"episode_id": episode_id, "answer_list": answer_list},
                                        ensure_ascii=False) + "\n")
        print(json.dumps({"event": "answer_complete", "answers": str(answers_path),
                          "questions": len(done), "skipped_episode_ids": sorted(skipped),
                          "elapsed_seconds": round(time.time() - started, 1)}),
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache-shard")
    cache.add_argument("--test-file", required=True)
    cache.add_argument("--residualmem-root", required=True)
    cache.add_argument("--qwen35-model", required=True)
    cache.add_argument("--qformer", required=True)
    cache.add_argument("--retrieval-head", required=True)
    cache.add_argument("--query-model", required=True)
    cache.add_argument("--output-dir", required=True)
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--shard-index", type=int, required=True)
    cache.add_argument("--shard-count", type=int, required=True)
    cache.add_argument("--max-length", type=int, default=8192)
    cache.add_argument("--domains", nargs="+", default=["WEB"],
                       help="AMA domains to build caches for; the default keeps "
                            "the original web-only behaviour")
    cache.add_argument("--overwrite", action="store_true")
    cache.add_argument("--skip-episode-ids", default="")
    cache.add_argument(
        "--clean-axtree", choices=("none", "medium"), default="none",
        help="apply the deterministic R1-R4 AXTree redundancy filter (axtree_clean "
             "medium) to each observation before encoding; caches get the "
             "v3-clean protocol so dirty and clean arms never share a file")
    cache.set_defaults(func=cache_shard)

    generate = sub.add_parser("answer")
    generate.add_argument("--test-file", required=True)
    generate.add_argument("--cache-dir", required=True)
    generate.add_argument("--qformer", required=True)
    generate.add_argument("--retrieval-head", required=True)
    generate.add_argument("--bridge", required=True)
    generate.add_argument("--reader-model", required=True)
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--answers-output", required=True)
    generate.add_argument("--top-k", type=int, default=5,
                          help="number of retrieved step-text rows (official RAG uses 5)")
    generate.add_argument("--top-k-latent", type=int, default=1,
                          help="number of latent memory rows; clamped to the bridge's "
                               "max_memory_ranks")
    generate.add_argument("--batch-size", type=int, default=4)
    generate.add_argument("--max-new-tokens", type=int, default=8192,
                          help="response budget; official baseline uses 8192")
    generate.add_argument("--max-model-len", type=int, default=32000,
                          help="official qwen3-32B.yaml context length; prompt budget "
                               "is max_model_len - max_new_tokens")
    generate.add_argument("--enable-thinking",
                          type=lambda value: value.strip().lower() == "true",
                          default=True,
                          help="must match the bridge training setting")
    generate.add_argument("--max-memory-gib", type=int, default=34)
    generate.add_argument("--memory-mode",
                          choices=("question-only", "text-only", "latent-only",
                                   "latent+anchor", "matched", "shuffled"),
                          default="matched")
    generate.add_argument(
        "--anchor-style", choices=(ANCHOR_STYLE_ELEMENT, ANCHOR_STYLE_IDS),
        default=ANCHOR_STYLE_ELEMENT,
        help="verbatim anchor granularity for latent+anchor: 'element' emits "
             "[id] role 'name' rows of the injected screens (P1); 'ids' emits "
             "only the [id] tokens (P2)")
    generate.add_argument(
        "--retrieval", choices=("semantic", "hybrid", "lexical"),
        default="semantic",
        help="retrieval channel(s) selecting which memory rows are injected: "
             "'semantic' is the document@query head score (previous behavior); "
             "'lexical' re-ranks rows by BM25 over the step texts vs the "
             "question; 'hybrid' fuses the two orderings with reciprocal-rank "
             "fusion so keyword hits (element names, actions, quoted strings, "
             "step numbers) can surface rows semantic search misses")
    generate.add_argument("--shuffle-seed", type=int, default=35)
    generate.add_argument("--limit", type=int, default=0)
    generate.add_argument("--allow-partial-cache", action="store_true")
    generate.add_argument("--skip-episode-ids", default="")
    generate.set_defaults(func=answer)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
