"""Stage C: is a latent as good to answer from as the text it replaces?

End-to-end QA on WorldMemArena cannot answer this. Only 0.30 of the ten rows
retrieved per question is an observation row -- the only kind that carries a
latent -- so whatever the tokenizer does is drowned by the round-text rows that
surround it. This measures the question directly by taking retrieval out of the
loop: one observation is handed to the reader twice, once as the text the row
holds and once as the soft tokens produced from it. Same question, same frozen
Qwen3.5, same official judge.

Two question sets:

``gold`` pairs each question with the observations its **own gold evidence**
points at, joining ``gold_evidence_memory_ids`` (an official field, which
carries image ids alongside memory-point ids) to the ``image_ids`` each
extracted observation records. 1,184 of the 1,459 web questions match at least
one observation, for 2,248 pairs.

``retrieval`` keeps the original protocol -- the top-ranked retrieved
observation per question, 233 questions -- so earlier numbers stay comparable.
It is the weaker set: it measures whichever row retrieval happened to surface,
which is why the text condition itself only answers 29.6% of them and why the
subset that actually tests fidelity was only 69 questions.

Two latent sources: a connector reading the pooled ``xbar`` off disk, or a
Q-Former reader that re-runs the frozen trunk over the observation's own inputs.

This does not claim a token saving. 64 soft tokens stand in for a median
88-token observation, a 0.72x ratio, because the connector emits one token per
slot with no pooling anywhere.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from residualmem.latent.instruct_bridge import (
    READER_BRIDGE_PROTOCOL,
    InputSoftTokenConnector,
    MemorySegment,
    Qwen35LatentReader,
    load_bridge,
)
from residualmem.latent.qformer import (
    QFORMER_PROTOCOL,
    QFormerStateReader,
    StateQFormer,
)


@dataclasses.dataclass(frozen=True)
class Observation:
    xbar: np.ndarray
    valid: np.ndarray
    text: str
    screenshot: str
    axtree: str
    image_ids: tuple[str, ...]


def load_observations(directory: Path) -> dict[tuple[str, str], Observation]:
    """``(sample_id, memory_id) -> Observation`` from an extraction."""
    table: dict[tuple[str, str], Observation] = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata"])))
            keys = sorted(k for k in data.files if k.startswith("m11/xbar/"))
            records = metadata.get("records") or []
            if len(records) != len(keys):
                raise ValueError(f"{path.name}: {len(records)} records, {len(keys)} states")
            for key, record in zip(keys, records):
                memory_id = record.get("memory_id")
                if not memory_id:
                    continue
                index = key.rsplit("/", 1)[1]
                table[(path.stem, str(memory_id))] = Observation(
                    xbar=np.asarray(data[key], np.float32),
                    valid=np.asarray(data[f"m11/valid/{index}"], bool),
                    text=str(record.get("fused_text", "")),
                    screenshot=str(record.get("screenshot", "")),
                    axtree=str(record.get("synthetic_axtree", "")),
                    image_ids=tuple(str(x) for x in (record.get("image_ids") or [])),
                )
    if not table:
        raise FileNotFoundError(f"no observations under {directory}")
    return table


def questions_from_retrieval(pipeline: Path, table: dict) -> list[dict]:
    """The original protocol: the top-ranked retrieved observation per question."""
    chosen = []
    for line in pipeline.open():
        record = json.loads(line)
        for item in record["retrieval"]["items"]:
            key = (record["sample_id"], item["memory_id"])
            if key in table:
                chosen.append({
                    "sample_id": record["sample_id"],
                    "question": record["question"],
                    "gold_answer": record.get("gold_answer", ""),
                    "gold_contents": record.get("gold_evidence_contents", []),
                    "rank": int(item["rank"]),
                    "key": key,
                })
                break
    return chosen


def questions_from_gold(pipeline: Path, table: dict) -> list[dict]:
    """Each question against the observations its own gold evidence names.

    ``gold_evidence_memory_ids`` mixes memory-point ids with image ids; only the
    latter resolve here, and they resolve at observation level rather than the
    benchmark matcher's session-level fallback.
    """
    by_image: dict[tuple[str, str], tuple[str, str]] = {}
    for key, observation in table.items():
        for image_id in observation.image_ids:
            by_image[(key[0], image_id)] = key
    chosen = []
    for line in pipeline.open():
        record = json.loads(line)
        seen = set()
        for evidence in record.get("gold_evidence_memory_ids") or []:
            key = by_image.get((record["sample_id"], str(evidence)))
            if key is None or key in seen:
                continue
            seen.add(key)
            chosen.append({
                "sample_id": record["sample_id"],
                "question": record["question"],
                "gold_answer": record.get("gold_answer", ""),
                "gold_contents": record.get("gold_evidence_contents", []),
                "rank": -1,
                "key": key,
            })
    return chosen


class LatentSource:
    """Turns an observation into soft-token input, either way it can be built."""

    def __init__(self, connector, qformer_reader, processor, model, device, layer):
        self.connector = connector
        self.reader = qformer_reader
        self.processor = processor
        self.model = model
        self.device = device
        self.layer = layer

    def segment(self, observation: Observation) -> MemorySegment:
        if self.reader is None:
            return MemorySegment(latent=(observation.xbar, observation.valid))
        from .trunk_states import collate, trunk_states

        if not observation.axtree:
            raise ValueError(
                "the extraction did not persist synthetic_axtree; a Q-Former has "
                "to re-run the trunk over the identical input"
            )
        with Image.open(observation.screenshot) as handle:
            image = handle.convert("RGB")
        states = trunk_states(self.processor, self.model, image, observation.axtree,
                              layer=self.layer, device=self.device)
        with torch.no_grad():
            xbar, valid = self.reader.encode(*collate([states]))
        return MemorySegment(latent=(xbar[0].float().cpu().numpy(),
                                     valid[0].cpu().numpy()))

    @property
    def slots(self) -> int:
        if self.reader is not None:
            return self.reader.qformer.num_queries
        return int(self.connector.rank_embedding.shape[0])


def answer_with_image(processor, model, question: str, text: str,
                      screenshot: str | None, device, max_new_tokens: int = 96) -> str:
    """The chat-template path, with or without the screenshot attached.

    ``Qwen3-VL-Embedding-8B-FusedObs-RAG`` sets ``mm_mode: "image"``
    (``eval_framework/config.yaml:71``), so the official reader receives the
    screenshot base64-inlined next to the row's text. Comparing a latent that
    encodes the screenshot against the caption alone measures pixels against a
    caption, not a latent against the memory.

    ``screenshot=None`` runs the same wrapper with no image, which is the only
    way to read the image's contribution: the text and latent conditions use the
    bare tokenizer the reader was built around, and this one *must* use the chat
    template to place the image token. Measured, that wrapper is not neutral --
    templated answers refuse far more often -- so image-versus-no-image is only
    interpretable inside the template.

    ``enable_thinking=False`` matters and is not cosmetic: with the default the
    model spends its whole budget on a reasoning preamble and the answer never
    appears inside ``max_new_tokens``, which the judge then scores as an
    omission. That is the setting the answer server uses too
    (``local_openai_server.py:77``).
    """
    prompt = (
        "Answer the question from the retrieved memory. Keep the answer concise. "
        "If absent, say exactly 'Not mentioned in memory.'.\nQuestion: " + question
    )
    content = [{"type": "text", "text": f"{text}\n\n{prompt}"}]
    images = []
    if screenshot:
        with Image.open(screenshot) as handle:
            images.append(handle.convert("RGB"))
        content.insert(0, {"type": "image"})
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = processor(text=[rendered], images=images or None,
                       return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                   do_sample=False)
    trimmed = generated[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(trimmed, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True, help="pipeline_qa.jsonl from a run")
    parser.add_argument("--xbar-dir", required=True)
    parser.add_argument("--connector", help="an InputSoftTokenConnector over cached xbar")
    parser.add_argument("--qformer", help="a QFormerStateReader; re-runs the trunk")
    parser.add_argument("--queries", type=int, default=16,
                        help="query count the --qformer checkpoint was trained with")
    parser.add_argument("--qformer-hidden", type=int, default=1024)
    parser.add_argument("--qformer-heads", type=int, default=8)
    parser.add_argument("--qformer-layers", type=int, default=4)
    parser.add_argument("--self-attention", action="store_true",
                        help="only for checkpoints trained with it; off by default")
    parser.add_argument("--questions", choices=("gold", "retrieval"), default="gold")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="0 = every eligible pair")
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.connector) == bool(args.qformer):
        parser.error("pass exactly one of --connector or --qformer")

    import sys
    sys.path.insert(0, str(Path(args.pipeline).resolve().parents[2]))
    from eval_framework.judges import evaluate_qa_llm

    table = load_observations(Path(args.xbar_dir))
    build = questions_from_gold if args.questions == "gold" else questions_from_retrieval
    questions = build(Path(args.pipeline), table)
    if args.limit and len(questions) > args.limit:
        # Subsample rather than truncate: the pipeline is ordered by sample, so
        # the first N would come from a handful of websites.
        index = np.random.default_rng(args.seed).choice(
            len(questions), args.limit, replace=False
        )
        questions = [questions[int(i)] for i in sorted(index)]
    print(f"[stageC] {args.questions} protocol: {len(questions)} (question, observation) "
          f"pairs over {len({q['question'] for q in questions})} questions", flush=True)

    from .extract_qwen import _load_model

    device = torch.device(args.device)
    processor, model = _load_model(args.model, device, False)
    if args.qformer:
        module = QFormerStateReader(
            StateQFormer(num_queries=args.queries, hidden=args.qformer_hidden,
                         heads=args.qformer_heads, layers=args.qformer_layers,
                         modalities=4, self_attention=args.self_attention),
            InputSoftTokenConnector(slots=args.queries),
        )
        payload = torch.load(args.qformer, map_location="cpu", weights_only=True)
        if payload.get("protocol") != QFORMER_PROTOCOL:
            raise ValueError(f"{args.qformer}: protocol {payload.get('protocol')!r}")
        # The retrieval head is only present when the run used L_sem; it plays
        # no part in answering, so it is dropped rather than required.
        state = {k: v for k, v in payload["state_dict"].items()
                 if not k.startswith("retrieval_head.")}
        module.load_state_dict(state, strict=True)
        module.to(device).eval()
        source = LatentSource(None, module, processor, model, device, args.layer)
        answering = Qwen35LatentReader(model, processor, module.connector, mode="input")
    else:
        connector = InputSoftTokenConnector()
        load_bridge(args.connector, connector, expected_protocol=READER_BRIDGE_PROTOCOL)
        connector.to(device).eval()
        source = LatentSource(connector, None, processor, model, device, args.layer)
        answering = Qwen35LatentReader(model, processor, connector, mode="input")

    rows = []
    conditions_order = ("text", "text@template", "text+image", "latent", "latent-mismatched")
    labels = {name: collections.Counter() for name in conditions_order}
    # A latent from a *different* observation, same question. If the latent
    # condition scores just as well on these, its answers come from the question
    # and the model's priors rather than from the state -- which would make the
    # whole comparison meaningless. Drawn once, deterministically, from an
    # observation in another sample so it cannot be coincidentally relevant.
    others = np.random.default_rng(args.seed).permutation(len(questions))
    for position, entry in enumerate(questions, 1):
        observation = table[entry["key"]]
        decoy_key = questions[int(others[position - 1])]["key"]
        if decoy_key[0] == entry["key"][0]:
            decoy_key = questions[int(others[(position * 7 + 3) % len(questions)])]["key"]
        answers = {
            "text": answering.answer(
                entry["question"], [MemorySegment(text=observation.text)]
            ),
            "text@template": answer_with_image(
                processor, model, entry["question"], observation.text, None, device,
            ),
            "text+image": answer_with_image(
                processor, model, entry["question"], observation.text,
                observation.screenshot, device,
            ),
            "latent": answering.answer(
                entry["question"], [source.segment(observation)]
            ),
            "latent-mismatched": answering.answer(
                entry["question"], [source.segment(table[decoy_key])]
            ),
        }
        judged = {}
        for name in conditions_order:
            verdict = evaluate_qa_llm(
                entry["question"], entry["gold_answer"],
                "\n".join(str(x) for x in entry["gold_contents"]), answers[name],
            )
            label = str(verdict.get("evaluation_result", "Omission"))
            labels[name][label] += 1
            judged[name] = {"answer": answers[name], "label": label}
        text_tokens = int(processor.tokenizer(
            observation.text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].shape[1])
        rows.append({
            **{k: entry[k] for k in ("sample_id", "question", "rank")},
            "memory_id": entry["key"][1],
            "decoy_memory_id": f"{decoy_key[0]}/{decoy_key[1]}",
            "text_tokens": text_tokens,
            "latent_tokens": source.slots,
            "judged": judged,
        })
        if position % 25 == 0:
            print(f"[stageC] {position}/{len(questions)}", flush=True)

    report = {
        "protocol": "connector_fidelity_v3",
        "question_set": args.questions,
        "latent_source": "qformer" if args.qformer else "connector",
        "checkpoint": args.qformer or args.connector,
        "queries": source.slots,
        "pairs": len(rows),
        "labels": {name: dict(counter) for name, counter in labels.items()},
        "rows": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # The image costs roughly 880 tokens on a 1280x720 screenshot; that is the
    # number the latent is actually competing against, not the caption's 87.
    image_tokens = 880
    print("\n" + "=" * 70)
    print(f"{'condition':20s}{'Correct':>10s}{'Halluc':>10s}{'Omission':>10s}{'tokens':>12s}")
    print("-" * 70)
    average_text = float(np.mean([r["text_tokens"] for r in rows]))
    costs = {"text": average_text, "text@template": average_text,
             "text+image": average_text + image_tokens,
             "latent": float(source.slots), "latent-mismatched": float(source.slots)}
    for name in conditions_order:
        counter = labels[name]
        total = max(sum(counter.values()), 1)
        print(f"{name:20s}{counter['Correct'] / total:10.4f}"
              f"{counter['Hallucination'] / total:10.4f}"
              f"{counter['Omission'] / total:10.4f}{costs[name]:12.1f}")
    print("=" * 70)
    print("bare tokenizer:  text | latent | latent-mismatched")
    print("chat template :  text@template | text+image")
    print("compare only inside a wrapper -- the wrapper is not neutral")
    for reference in ("text", "text+image"):
        answerable = [r for r in rows if r["judged"][reference]["label"] == "Correct"]
        if answerable:
            recovered = sum(1 for r in answerable
                            if r["judged"]["latent"]["label"] == "Correct")
            print(f"{reference} answers {len(answerable)}/{len(rows)}; "
                  f"latent recovers {recovered}/{len(answerable)} = "
                  f"{recovered / len(answerable):.1%} of them")
    real = labels["latent"]["Correct"]
    decoy = labels["latent-mismatched"]["Correct"]
    print(f"\nlatent {real}/{len(rows)} correct against {decoy}/{len(rows)} on a "
          f"mismatched state: {real - decoy:+d}. Whatever the decoy scores is what "
          "the question and the model's priors supply without the memory.")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
