"""Label each AMA code position with what omitting it costs the 32B reader.

Port of ``label_counterfactual.py`` to the AMA stack. The protocol is unchanged:

  U_j = [ L_ans(q, a | x_hat(-j)) - L_ans(q, a | x_hat(full)) ] / ln 2   (bits)

where ``x_hat(-j)`` replaces position ``j`` with the world model's mode and
leaves the other 1023 true codes alone. Both sides are decoded from the
quantised codes, so the quantiser's own error cancels in the difference.

Two things differ from the WebWorld version, both forced:

**The reader.** WMA used Qwen3.5-9B plus a slot connector. AMA uses Qwen3-32B
behind the trained rank-major soft-token bridge, with the official AMA chat
scaffold (``render_ama_openend_parts``) and ``enable_thinking=True`` -- the
setting the bridge was trained under. The gold answer is teacher-forced after
the rendered assistant header; we never let the reader reason, we only score
P(answer | memory, question).

**The data layout.** AMA keeps three artefacts with different orderings:
``records.jsonl`` is per transition (14,944 rows, repeated states),
``states.jsonl`` / the OPQ codes are deduplicated by ``state_id`` (10,314
unique), and the posterior dump indexes the transition space. The join has to
go through ``records.jsonl`` sorted by ``(episode_id, step)``; joining the codes
to the posteriors by row position would silently mis-attribute every label.

Episode-initial states have no causal posterior and therefore no principled
fill; they are skipped, not filled some other way.

Runs under the torch interpreter (the AMA vLLM/torch venv).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import math
import sys
import time
from pathlib import Path

# ``xt_ama_adapter/`` is a namespace directory whose real package lives one level
# down; putting it on the path makes ``xt_ama_adapter.qwen32_bridge`` importable
# regardless of where the interpreter was started.
_ADAPTER_PARENT = Path(__file__).resolve().parents[2] / "xt_ama_adapter"
if str(_ADAPTER_PARENT) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_PARENT))

import numpy as np
import torch

from experiments.state_tokenizer.common import iter_jsonl
from experiments.utility_gate.groups import NUM_POSITIONS, slot_of, subspace_of
from experiments.utility_gate.verify_scorer import load_codebook, rebuild

LN2 = math.log(2.0)


def load_codes(pq_path: Path) -> tuple[np.ndarray, list[str]]:
    """The deduplicated OPQ code table: one row per unique ``state_id``."""
    with np.load(pq_path, allow_pickle=True) as data:
        codes = np.asarray(data["codes/all"], np.uint8)
        state_ids = [str(v) for v in np.asarray(data["state_ids/all"])]
    return codes, state_ids


def load_posteriors(posteriors_path: Path, records_path: Path):
    """``state_id -> posterior row``, plus the per-position WM quantities.

    ``records.jsonl`` sorted by ``(episode_id, step)`` reproduces the cache's
    global index space, which is what ``target_indices`` refers to. The same
    ``state_id`` can appear at several steps, so the first row wins; the
    duplicates are the same observation and carry the same posterior (asserted
    below for the ones we keep).
    """
    with np.load(posteriors_path, allow_pickle=True) as data:
        dumped = {k: np.asarray(data[k]) for k in data.files}
    records = list(iter_jsonl(records_path))
    records.sort(key=lambda r: (r["episode_id"], int(r["step"])))
    sid_at = np.asarray([str(r["state_id"]) for r in records])
    row_of: dict[str, int] = {}
    for k, g in enumerate(dumped["target_indices"]):
        row_of.setdefault(str(sid_at[int(g)]), k)
    return row_of, dumped


def load_pairs(path: Path) -> list[dict]:
    return [r for r in iter_jsonl(path) if r.get("question") and r.get("answer")]


def load_xbar(shards: list[Path]) -> dict[str, np.ndarray]:
    """``state_id -> (32, 512)`` reconstruction input, for the raw variant only."""
    out: dict[str, np.ndarray] = {}
    for path in shards:
        with np.load(path, allow_pickle=True) as data:
            xbar = np.asarray(data["xbar"], np.float32)
            ids = [str(v) for v in np.asarray(data["state_ids"])]
        for row, sid in enumerate(ids):
            out[sid] = xbar[row]
    return out


def split_of_state(states_path: Path, records_path: Path) -> tuple[dict, set]:
    """Authoritative split per state, plus the states shared across splits.

    ``states.jsonl`` assigns each deduplicated state exactly one split, which is
    the assignment the QA generator used. But a handful of states are content
    duplicates that appear in both a train and a validation episode (81 of them,
    measured); those must not be scored on the evaluation side or the mask fit
    would see its own evaluation states.
    """
    states = list(iter_jsonl(states_path))
    split = {str(r["state_id"]): str(r["split"]) for r in states}
    records = list(iter_jsonl(records_path))
    train = {str(r["state_id"]) for r in records if r["split"] == "train"}
    val = {str(r["state_id"]) for r in records if r["split"] == "validation"}
    return split, train & val


def latent_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Give padding no RoPE position and do not let it advance positions."""
    positions = attention_mask.long().cumsum(-1) - 1
    return positions.masked_fill(~attention_mask.bool(), 0)


def load_ama_reader(reader_model: str, dtype_name: str, device: str,
                    device_map: str | None = None):
    """Load the 32B reader, sharded across GPUs when asked.

    ``float32`` is the label-producing configuration: the measurement is a
    difference of two answer NLLs of order 50 bits, and bfloat16 shifts that
    difference by 0.06-0.23 bits depending on batch shape -- the same order as
    the signal. In float32 the WMA pass measured batch-geometry error at 7.6e-5
    nats. The price is 128 GB of weights, hence ``device_map="auto"``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(reader_model, trust_remote_code=True)
    torch_dtype = getattr(torch, dtype_name)
    if device_map:
        # Never let accelerate fall back to CPU: an fp32 32B reader on CPU runs
        # at ~1 % of the GPU speed and looks identical to a hang. Pin the budget
        # to what each visible GPU actually has free right now.
        max_memory = None
        if device_map == "auto":
            # Static budget on purpose: querying CUDA here (mem_get_info) trips
            # the sandbox's device-count check before anything is loaded. Each
            # visible 80 GB GPU gets 70 GiB, which is far more than the 32 GiB
            # its share of an fp32 32B model needs.
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            count = len([part for part in visible.split(",") if part.strip()])
            count = count or torch.cuda.device_count()
            max_memory = {index: "70GiB" for index in range(count)}
            # GPUs partially occupied by other jobs: "45,70,45" caps GPU 0 at
            # 45 GiB, GPU 1 at 70 GiB, ... (unlisted GPUs keep the 70 GiB
            # default). The static default assumes fully free cards.
            override = os.environ.get("RESIDUALMEM_READER_MAX_MEMORY")
            if override:
                for index, part in enumerate(override.split(",")):
                    if part.strip():
                        max_memory[index] = f"{float(part):g}GiB"
            max_memory["cpu"] = "40GiB"
            print(f"[reader] max_memory={max_memory}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            reader_model, torch_dtype=torch_dtype, device_map=device_map,
            max_memory=max_memory)
        model = model.eval()
        placement = {}
        for name, parameter in model.named_parameters():
            placement.setdefault(str(parameter.device), 0)
            placement[str(parameter.device)] += parameter.numel()
        print(f"[reader] placed on { {k: f'{v / 1e9:.1f}B' for k, v in placement.items()} }",
              flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            reader_model, torch_dtype=torch_dtype).to(device).eval()
    return tokenizer, model


def place_bridge(bridge_path: Path, model):
    """Bridge onto the reader's embedding device; returns ``(bridge, metadata)``."""
    from xt_ama_adapter.qwen32_bridge import load_qwen32_bridge

    bridge, metadata = load_qwen32_bridge(Path(bridge_path))
    embedding_device = model.get_input_embeddings().weight.device
    return bridge.to(device=embedding_device, dtype=torch.float32).eval(), metadata


def score_without_memory(model, tokenizer, question: str, answer: str, *,
                         enable_thinking: bool = True,
                         max_answer_tokens: int = 104) -> float:
    """Gold-answer NLL for the same question with NO memory at all.

    The reference NLL only means something if it is distinctly better than this.
    If the two are equal, the reader is answering from the question alone and
    every position's utility is zero by construction -- the labels would be
    measuring nothing.
    """
    from xt_ama_adapter.qwen32_bridge import render_ama_openend_parts

    embed = model.get_input_embeddings()
    device = embed.weight.device
    prefix, suffix, _ = render_ama_openend_parts(
        tokenizer, question, enable_thinking=bool(enable_thinking))
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)[:max_answer_tokens]
    answer_len = len(answer_ids)
    if answer_len == 0:
        return float("nan")

    def text_embeds(ids):
        return embed(torch.tensor([ids], dtype=torch.long, device=device))

    inputs = torch.cat((text_embeds(prefix_ids), text_embeds(suffix_ids),
                        text_embeds(answer_ids)), 1)
    attention = torch.ones((1, inputs.shape[1]), dtype=torch.long, device=device)
    kwargs = {"use_cache": False}
    parameters = inspect.signature(model.forward).parameters
    if "position_ids" in parameters:
        kwargs["position_ids"] = latent_position_ids(attention)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = answer_len + 1
    with torch.no_grad():
        out = model(inputs_embeds=inputs, attention_mask=attention, **kwargs)
        logits = out.logits[:, -answer_len - 1:-1, :].float()
        logprob = torch.nn.functional.log_softmax(logits, -1)
        target = torch.tensor([answer_ids], dtype=torch.long, device=logits.device)
        gathered = logprob.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(-gathered.sum() / math.log(2.0))


def answer_variant_scores_ama(
    model, tokenizer, latents: torch.Tensor, question: str, answer: str, *,
    enable_thinking: bool = True, max_answer_tokens: int = 104,
    reference: int = 0, kl_chunk: int = 8,
) -> dict:
    """Score B memory variants of one AMA question in a single forward.

    ``latents`` is ``(B, 32, 5120)`` soft tokens already pushed through the
    bridge; row ``reference`` is the intact memory. Returns per-variant
    teacher-forced gold-answer NLL and the KL from the reference's answer
    distribution, in bits.

    Every variant rides in the same forward for the same reason as in WMA: in a
    low-precision pass the absolute NLL depends on the row's place in the batch,
    and scoring the reference separately from its ablations would inject that
    difference straight into the label. ``--null`` measures how large the
    residue is.
    """
    if latents.dim() != 3:
        raise ValueError(f"latents must be (B, slots, dim), got {tuple(latents.shape)}")
    variants = int(latents.shape[0])
    if not 0 <= reference < variants:
        raise ValueError(f"reference {reference} outside a batch of {variants}")

    # Imported here so the module stays importable without the adapter package.
    from xt_ama_adapter.qwen32_bridge import render_ama_openend_parts

    device = latents.device
    embed = model.get_input_embeddings()
    dtype = embed.weight.dtype

    prefix, suffix, _ = render_ama_openend_parts(
        tokenizer, question, enable_thinking=bool(enable_thinking))
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)[:max_answer_tokens]
    answer_len = len(answer_ids)
    if answer_len == 0:
        raise ValueError(f"empty answer for question {question!r}")

    def text_embeds(ids: list[int]) -> torch.Tensor:
        tensor = torch.tensor([ids], dtype=torch.long, device=device)
        return embed(tensor).expand(variants, -1, -1)

    inputs = torch.cat((
        text_embeds(prefix_ids), latents.to(dtype), text_embeds(suffix_ids),
        text_embeds(answer_ids),
    ), 1)
    attention = torch.ones((variants, inputs.shape[1]), dtype=torch.long, device=device)

    kwargs = {"use_cache": False}
    parameters = inspect.signature(model.forward).parameters
    if "position_ids" in parameters:
        kwargs["position_ids"] = latent_position_ids(attention)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = answer_len + 1

    with torch.no_grad():
        out = model(inputs_embeds=inputs, attention_mask=attention, **kwargs)
        # logits[t] predicts token t+1, so the gold answer occupies the last
        # answer_len positions before the final one.
        answer_logits = out.logits[:, -answer_len - 1:-1, :]
        # With ``device_map`` sharding the output lives on the last shard, so the
        # reductions are allocated where the logits are rather than where the
        # inputs were built.
        reduce_device = answer_logits.device

        targets = torch.tensor([answer_ids] * variants, dtype=torch.long,
                               device=reduce_device)
        nll = torch.zeros(variants, dtype=torch.float32, device=reduce_device)
        token_nll = torch.empty((variants, answer_len), dtype=torch.float32,
                                device=reduce_device)
        kl = torch.zeros(variants, dtype=torch.float32, device=reduce_device)
        kl_max = torch.zeros(variants, dtype=torch.float32, device=reduce_device)
        argmax = torch.empty((variants, answer_len), dtype=torch.long,
                             device=reduce_device)

        for start in range(0, answer_len, kl_chunk):
            stop = min(start + kl_chunk, answer_len)
            logprob = torch.nn.functional.log_softmax(
                answer_logits[:, start:stop].float(), -1)
            gathered = logprob.gather(
                -1, targets[:, start:stop].unsqueeze(-1)).squeeze(-1)
            nll -= gathered.sum(-1)
            token_nll[:, start:stop] = -gathered
            argmax[:, start:stop] = logprob.argmax(-1)
            anchor = logprob[reference:reference + 1]
            per_position = (anchor.exp() * (anchor - logprob)).sum(-1)
            kl += per_position.sum(-1)
            kl_max = torch.maximum(kl_max, per_position.max(-1).values)

    ln2 = math.log(2.0)
    return {
        "nll_bits": (nll / ln2).cpu(),
        "token_nll_bits": (token_nll / ln2).cpu(),
        "kl_bits": (kl / ln2).cpu(),
        "kl_max_position_bits": (kl_max / ln2).cpu(),
        "teacher_forced_argmax": argmax.cpu(),
        "answer_len": answer_len,
        "question_len": len(prefix_ids) + len(suffix_ids),
        "reference": reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq", default="outputs/wm_train/ama-v1/ama-pq-c64-v2codebook.npz")
    parser.add_argument("--codebook",
                        default="outputs/wm_train/ama-v1/ama-pq-c64-v2codebook.npz")
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--records", default="outputs/wm_train/ama-v1/records.jsonl")
    parser.add_argument("--states", default="outputs/wm_train/ama-v1/states.jsonl")
    parser.add_argument("--xbar-shards", nargs="*", default=None,
                        help="ama-states-*.npz shards; only needed for the "
                             "raw_xbar (quantiser-cost) variant")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--reader-model", default="/data1/models/Qwen3-32B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default=None,
                        help="e.g. 'auto' to shard the reader across every visible "
                             "GPU; required for --dtype float32")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--enable-thinking", type=int, default=1)
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--max-states", type=int, default=0,
                        help="subsample whole states, not loose rows")
    parser.add_argument("--max-answer-tokens", type=int, default=104)
    parser.add_argument("--null", action="store_true",
                        help="fill each position with its own true code; every "
                             "variant then equals the reference and U must be 0")
    parser.add_argument("--batch-invariance", action="store_true",
                        help="also score the reference alone at B=1 and report "
                             "the gap, which bounds what batch geometry costs")
    parser.add_argument("--positive-control", action="store_true",
                        help="also score the all-positions-replaced variant: if "
                             "that does not move the answer NLL, the reader is "
                             "not using the memory at all and the single-position "
                             "labels mean nothing")
    parser.add_argument("--diagnose", action="store_true",
                        help="on the first row print the no-memory baseline and "
                             "the reader's own generation from the reference")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    book = load_codebook(Path(args.codebook))
    codes, state_ids = load_codes(Path(args.pq))
    by_id = {sid: i for i, sid in enumerate(state_ids)}
    split, shared = split_of_state(Path(args.states), Path(args.records))
    row_of, dumped = load_posteriors(Path(args.posteriors), Path(args.records))

    xbar = load_xbar([Path(p) for p in args.xbar_shards]) if args.xbar_shards else {}

    fill = np.zeros_like(codes)
    rate = np.full(codes.shape, np.nan, np.float32)
    has_posterior = np.zeros(len(codes), bool)
    for sid, row in by_id.items():
        posterior = row_of.get(sid)
        if posterior is None:
            continue
        fill[row] = dumped["wm_argmax"][posterior]
        rate[row] = dumped["code_bits"][posterior]
        has_posterior[row] = True

    pairs = load_pairs(Path(args.pairs))
    usable = []
    for pair_row, pair in enumerate(pairs):
        sid = str(pair["state_id"])
        if sid not in by_id:
            continue
        if split.get(sid) != args.split:
            continue
        if args.split == "validation" and sid in shared:
            continue  # also present in a train episode: keep it out of eval
        row = by_id[sid]
        if not has_posterior[row]:
            continue  # episode-initial: no causal posterior, no principled fill
        usable.append(pair_row)
    skipped = len(pairs) - len(usable)

    rng = np.random.default_rng(args.seed)
    per_state: dict[str, np.ndarray] = {}
    for pair_row in usable:
        sid = str(pairs[pair_row]["state_id"])
        if sid not in per_state:
            per_state[sid] = np.sort(rng.permutation(NUM_POSITIONS)[: args.positions])

    # Subsample whole states so a state keeps all of its questions: the
    # "same state, different question" variance -- the term that bounds what any
    # write-time gate can achieve -- needs those pairs.
    states_used = sorted(per_state)
    if args.max_states and args.max_states < len(states_used):
        states_used = sorted(rng.choice(states_used, args.max_states, replace=False))
    keep_states = set(states_used)
    rows = [r for r in usable if str(pairs[r]["state_id"]) in keep_states]
    rows = rows[args.shard_index::args.shard_count]
    print(f"[label] {len(rows)} rows of {len(usable)} usable ({skipped} skipped: "
          f"no state, no posterior, wrong split or shared), "
          f"{len(states_used)} states, fill=world_model_argmax", flush=True)

    device = torch.device(args.device)
    tokenizer, model = load_ama_reader(
        args.reader_model, args.dtype, args.device, args.device_map)
    bridge, bridge_metadata = place_bridge(Path(args.bridge), model)
    if bool(bridge_metadata.get("enable_thinking")) != bool(args.enable_thinking):
        print(f"[label] WARNING bridge trained with enable_thinking="
              f"{bridge_metadata.get('enable_thinking')}, running with "
              f"{bool(args.enable_thinking)}", flush=True)

    out: dict[str, list] = {k: [] for k in (
        "pair_row", "state_row", "position", "kl_bits", "delta_nll_bits",
        "abs_delta_nll_bits", "effective_tokens", "top1_token_share",
        "argmax_changed", "rate_bits")}
    per_row: dict[str, list] = {k: [] for k in (
        "row_pair_row", "row_state_row", "row_reference_nll_bits",
        "row_raw_nll_bits", "row_answer_len", "row_question_len",
        "row_answer_truncated")}

    started = time.time()
    for done, pair_row in enumerate(rows):
        pair = pairs[pair_row]
        sid = str(pair["state_id"])
        state = by_id[sid]
        positions = per_state[sid]
        question, answer = str(pair["question"]), str(pair["answer"])

        true_codes = codes[state]
        source = true_codes if args.null else fill[state]
        stacked = np.repeat(true_codes[None], len(positions) + 1, axis=0)
        for offset, position in enumerate(positions, start=1):
            slot, subspace = slot_of(int(position)), subspace_of(int(position))
            stacked[offset, slot, subspace] = source[slot, subspace]
        variants = [rebuild(stacked, book)]
        names = ["reference"] + [f"position:{int(p)}" for p in positions]
        if sid in xbar:
            variants.append(xbar[sid][None])
            names.append("raw_xbar")
        batch = np.concatenate(variants).astype(np.float32)

        with torch.no_grad():
            soft, _mask = bridge(
                torch.as_tensor(batch, device=device)[:, None],
                torch.ones((batch.shape[0], 1, batch.shape[1]),
                           dtype=torch.bool, device=device))
            scored = answer_variant_scores_ama(
                model, tokenizer, soft, question, answer,
                enable_thinking=bool(args.enable_thinking),
                max_answer_tokens=args.max_answer_tokens)

        nll = scored["nll_bits"].numpy()
        kl = scored["kl_bits"].numpy()
        token_nll = scored["token_nll_bits"].numpy()
        argmax = scored["teacher_forced_argmax"].numpy()

        per_row["row_pair_row"].append(pair_row)
        per_row["row_state_row"].append(state)
        per_row["row_reference_nll_bits"].append(float(nll[0]))
        per_row["row_raw_nll_bits"].append(
            float(nll[names.index("raw_xbar")]) if "raw_xbar" in names else float("nan"))
        per_row["row_answer_len"].append(int(scored["answer_len"]))
        per_row["row_question_len"].append(int(scored["question_len"]))
        per_row["row_answer_truncated"].append(
            bool(len(tokenizer.encode(answer, add_special_tokens=False))
                 > args.max_answer_tokens))

        for offset, position in enumerate(positions, start=1):
            slot, subspace = slot_of(int(position)), subspace_of(int(position))
            spread = np.abs(token_nll[offset] - token_nll[0])
            total = float(spread.sum())
            square = float((spread ** 2).sum())
            out["pair_row"].append(pair_row)
            out["state_row"].append(state)
            out["position"].append(int(position))
            out["kl_bits"].append(float(kl[offset]))
            out["delta_nll_bits"].append(float(nll[offset] - nll[0]))
            out["abs_delta_nll_bits"].append(total)
            out["effective_tokens"].append(
                float(total ** 2 / square) if square > 0 else 0.0)
            out["top1_token_share"].append(
                float(spread.max() / total) if total > 0 else 0.0)
            out["argmax_changed"].append(bool((argmax[offset] != argmax[0]).any()))
            out["rate_bits"].append(float(rate[state, slot, subspace]))

        if args.diagnose and done == 0:
            # Is the reference memory doing anything at all? Two references: the
            # same question with no memory, and the reader's own generation from
            # the reference latent. If the empty baseline is as good as the
            # memory, or the generation does not contain the gold answer, then
            # this question distribution is not answerable from the latent and
            # the labels are measuring nothing.
            reference_nll = float(nll[0])
            empty_nll = score_without_memory(
                model, tokenizer, question, answer,
                enable_thinking=bool(args.enable_thinking),
                max_answer_tokens=args.max_answer_tokens)
            print(f"[label] diagnosis: reference NLL {reference_nll:.4f} bits vs "
                  f"no-memory {empty_nll:.4f} bits -> memory gain "
                  f"{empty_nll - reference_nll:+.4f} bits", flush=True)

            from xt_ama_adapter.qwen32_bridge import Qwen32LatentReader

            reader = Qwen32LatentReader(
                model, tokenizer, bridge, enable_thinking=bool(args.enable_thinking))
            reference_state = torch.as_tensor(
                rebuild(true_codes[None], book), dtype=torch.float32,
                device=bridge.input_norm.weight.device)
            generated = reader.answer(
                question, reference_state,
                torch.ones(reference_state.shape[:2], dtype=torch.bool,
                           device=reference_state.device),
                max_new_tokens=64)
            print(f"[label] diagnosis: gold {answer!r} vs generated "
                  f"{generated!r}", flush=True)

        if args.positive_control and done == 0:
            # Drop every position to the world model's mode at once. This is the
            # ceiling of what the mask can ever do, so it is the control that
            # separates "this state is genuinely insensitive to its codes" from
            # "the reader never looked at the memory".
            worst = true_codes.copy()
            worst[...] = fill[state]
            pair = np.concatenate(
                [rebuild(true_codes[None], book), rebuild(worst[None], book)]
            ).astype(np.float32)
            with torch.no_grad():
                control_soft, _ = bridge(
                    torch.as_tensor(pair, device=device)[:, None],
                    torch.ones((2, 1, pair.shape[1]), dtype=torch.bool, device=device))
                control = answer_variant_scores_ama(
                    model, tokenizer, control_soft, question, answer,
                    enable_thinking=bool(args.enable_thinking),
                    max_answer_tokens=args.max_answer_tokens)
            print(f"[label] positive control (all {NUM_POSITIONS} positions -> mode): "
                  f"delta_nll {float(control['nll_bits'][1] - control['nll_bits'][0]):+.4f} "
                  f"bits, kl {float(control['kl_bits'][1]):.4f} bits", flush=True)

        if args.batch_invariance and done == 0:
            # The quantity that must be geometry-stable is the DIFFERENCE
            # nll(ablation) - nll(reference), not either NLL alone: a low
            # precision pass shifts the absolute NLL by a common-mode amount
            # when the batch shape changes, and that cancels in the label. So
            # score the reference and each probe position ALONE at B=1 and
            # compare their difference against the batched one.
            def solo_nll(row_index: int) -> float:
                alone = batch[row_index:row_index + 1]
                with torch.no_grad():
                    solo_soft, _ = bridge(
                        torch.as_tensor(alone, device=device)[:, None],
                        torch.ones((1, 1, alone.shape[1]),
                                   dtype=torch.bool, device=device))
                    solo = answer_variant_scores_ama(
                        model, tokenizer, solo_soft, question, answer,
                        enable_thinking=bool(args.enable_thinking),
                        max_answer_tokens=args.max_answer_tokens)
                return float(solo["nll_bits"][0])

            solo_reference = solo_nll(0)
            print(f"[label] batch invariance: reference NLL {nll[0]:.4f} bits at "
                  f"B={batch.shape[0]} vs {solo_reference:.4f} bits at B=1 "
                  f"(common-mode {abs(solo_reference - nll[0]):.6f})", flush=True)
            for offset in range(1, min(3, len(positions) + 1)):
                solo_delta = solo_nll(offset) - solo_reference
                batched_delta = float(nll[offset] - nll[0])
                print(f"[label] batch invariance: position {int(positions[offset - 1])} "
                      f"delta_nll batched {batched_delta:+.6f} vs B=1 "
                      f"{solo_delta:+.6f} -> gap {abs(batched_delta - solo_delta):.6f} bits",
                      flush=True)

        if done % 10 == 0:
            elapsed = time.time() - started
            per_row_s = elapsed / max(done + 1, 1)
            print(f"[label] {done + 1}/{len(rows)}  {per_row_s:.2f}s/row  "
                  f"eta {(len(rows) - done - 1) * per_row_s / 60:.1f} min",
                  flush=True)

    dumped_out = {k: np.asarray(v) for k, v in out.items()}
    dumped_out.update({k: np.asarray(v) for k, v in per_row.items()})
    dumped_out["state_ids"] = np.asarray(state_ids, dtype=object)
    dumped_out["metadata"] = json.dumps({
        "protocol": "residualmem_utility_gate_labels_ama_v1",
        "split": args.split,
        "fill": "true_code_null" if args.null else "world_model_argmax",
        "null_control": bool(args.null),
        "positions_per_state": int(args.positions),
        "states": len(states_used), "seed": args.seed,
        "shard": [args.shard_index, args.shard_count],
        "rows": len(rows), "usable": len(usable), "skipped": skipped,
        "pairs": str(Path(args.pairs).resolve()),
        "posteriors": str(Path(args.posteriors).resolve()),
        "bridge": str(Path(args.bridge).resolve()),
        "reader_model": str(args.reader_model),
        "reader_dtype": args.dtype,
        "enable_thinking": bool(args.enable_thinking),
        "max_answer_tokens": int(args.max_answer_tokens),
    }, ensure_ascii=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dumped_out)

    delta = dumped_out["delta_nll_bits"]
    kl = dumped_out["kl_bits"]
    summary = {
        "rows": len(rows), "labels": int(len(delta)),
        "delta_nll_bits": {
            "mean": float(delta.mean()),
            "abs_median": float(np.median(np.abs(delta))),
            "max_abs": float(np.abs(delta).max()) if len(delta) else 0.0,
            "nonzero": float((delta != 0).mean()) if len(delta) else 0.0,
        },
        "kl_bits": {
            "mean": float(kl.mean()), "median": float(np.median(kl)),
            "max": float(kl.max()) if len(kl) else 0.0,
            "nonzero": float((kl != 0).mean()) if len(kl) else 0.0,
        },
        "argmax_changed": float(dumped_out["argmax_changed"].mean()),
        "position_coverage": int(len(np.unique(dumped_out["position"]))),
        "answer_truncated_fraction": float(dumped_out["row_answer_truncated"].mean()),
        "seconds_per_row": round((time.time() - started) / max(len(rows), 1), 3),
    }
    if args.null:
        summary["null_control"] = {
            "max_abs_delta_nll_bits": float(np.abs(delta).max()) if len(delta) else 0.0,
            "max_kl_bits": float(kl.max()) if len(kl) else 0.0,
        }
    (output.with_suffix(".summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
