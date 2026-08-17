"""Does the state say *which element* a target label belongs to?

``overlap_words`` shows a literal survived somewhere in the state. Completing
"Select bg and click Submit" needs more than that: the agent has to know which
checkbox is the one called ``bg``. Static Key64 makes this a live risk by design
-- detail slots hold literal spans, while ``ref``/``parent``/``box``/``id`` are
explicitly kept out of raw detail and structure is carried by the pooled
full-node context. So a representation can score well on word presence while the
label-to-element binding is gone, and an agent reading it still cannot click.

The probe is conditioned on the *specific* literal:

    P(ref | x_t, T, target literal)

Conditioning matters for more than realism. Once the selector prioritises
instruction-mentioned candidates, "which detail slot" leaks "is a target", and a
probe asked only for the target ref set could score well from slot order alone.
Asking which ref belongs to *this* literal removes that shortcut: in
click-checkboxes every target is prioritised, so slot order cannot separate them.

Three reference numbers are reported, and the important one is the ablation.

``dom_oracle`` -- picking the clickable element nearest the literal in DOM order
-- scores 1.0 by construction, because that is exactly how the label is derived.
It is therefore *not* a competitive baseline; it is a sanity check that the
labels are well posed, and an upper bound: the binding is fully determined by the
raw DOM, so whatever the probe loses was lost by the tokenizer.

``literal_ablated`` is the real control. The same probe is trained and evaluated
with the literal input zeroed, so it must name a target ref knowing only the
state. If that scores near the conditioned probe, the conditioning is doing no
work and slot order is carrying the answer -- which is exactly the leak that
instruction-priority selection would introduce.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from .common import iter_jsonl, write_json
from .key_pooling import parse_dom_spans
from .slot_probe import TokenProvider

REPRESENTATION = "key64_static_pca"
CLICKABLE = {"input_checkbox", "input_radio", "option", "button",
             "input_submit", "input_button", "a", "label"}
BINDING_TASKS = ("miniwob/click-option-v1", "miniwob/click-checkboxes-v1")
# Upper bound on the element id the probe can name. Derived from the data, not
# fixed: MiniWoB's own refs top out around 17, but BrowserGym's bids reach 36 in
# the same tasks, and a hardcoded 24 silently dropped 48% of the clickable
# elements -- along with every example whose answer was one of them. The probe
# then trained on a truncated candidate set and scored at its literal-ablated
# control, which reads as "binding is absent" rather than "the labels were cut".
DEFAULT_MAX_REF = 24
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"


def instruction_targets(instruction: str) -> set[str]:
    """The literals named by a "Select a, b and click Submit" instruction."""
    if not instruction.startswith("Select"):
        return set()
    head = instruction.split(" and ")[0].replace("Select", "")
    return {word for word in re.findall(r"[A-Za-z0-9]{2,8}", head)}


def binding_labels(dom: str, targets: set[str]) -> dict[str, tuple[int, float]]:
    """Map each target literal to the ref of its clickable element.

    The literal sits on a ``t`` node; the control it names is a sibling (or the
    parent) under the same ``label``. Also returns the literal's node index so
    the positional control can be computed without the model.
    """
    nodes = parse_dom_spans(dom)
    by_ref = {node.ref: node for node in nodes if node.ref}
    children: dict[str, list] = defaultdict(list)
    for node in nodes:
        if node.parent:
            children[node.parent].append(node)

    out: dict[str, tuple[int, float]] = {}
    for node in nodes:
        text = next((a.value for a in node.attributes if a.key == "text"), None)
        if text is None or text.strip() not in targets:
            continue
        literal = text.strip()
        if node.tag in CLICKABLE and node.ref:
            out[literal] = (int(node.ref), float(node.index))
            continue
        sibling = next(
            (s for s in children.get(node.parent, ()) if s.tag in CLICKABLE and s.ref),
            None,
        )
        if sibling is not None:
            out[literal] = (int(sibling.ref), float(node.index))
            continue
        parent = by_ref.get(node.parent)
        if parent is not None and parent.tag in CLICKABLE and parent.ref:
            out[literal] = (int(parent.ref), float(node.index))
    return out


def clickable_refs(dom: str, max_ref: int) -> list[tuple[int, float]]:
    """Every clickable ref with its node index, for the controls."""
    return [
        (int(n.ref), float(n.index))
        for n in parse_dom_spans(dom)
        if n.tag in CLICKABLE and n.ref and 0 <= int(n.ref) < max_ref
    ]


def infer_max_ref(records, instructions) -> int:
    """Smallest bound that keeps every binding example the data actually offers."""
    highest = 0
    for record in records:
        if record["task"] not in BINDING_TASKS:
            continue
        targets = instruction_targets(instructions.get(record["state_id"], ""))
        if not targets:
            continue
        for node in parse_dom_spans(record["dom"]):
            if node.tag in CLICKABLE and node.ref and node.ref.lstrip("-").isdigit():
                highest = max(highest, int(node.ref))
    return highest + 1


def encode_literal(literal: str, width: int = 12) -> np.ndarray:
    """Fixed-width character indices; targets are short random strings."""
    codes = np.zeros((width,), np.int64)
    for position, char in enumerate(literal[:width]):
        codes[position] = CHARS.find(char) + 1
    return codes


class BindingReader(nn.Module):
    """Cross-attention from a literal-conditioned query onto the state slots."""

    def __init__(self, input_dim: int, max_ref: int, hidden: int = 256, heads: int = 4):
        super().__init__()
        self.slots = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden))
        self.slot_position = nn.Linear(1, hidden)
        self.characters = nn.Embedding(len(CHARS) + 1, hidden, padding_idx=0)
        self.literal = nn.GRU(hidden, hidden, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, max_ref)

    def forward(self, tokens, positions, valid, literal):
        values = self.slots(tokens) + self.slot_position(positions.unsqueeze(-1))
        _, query = self.literal(self.characters(literal))
        query = query[-1].unsqueeze(1)
        attended, _ = self.attention(
            query, values, values, key_padding_mask=~valid
        )
        return self.head(self.norm(attended.squeeze(1)))


def build_examples(records, instructions, max_ref: int):
    """One example per (state, target literal) pair."""
    examples = []
    dropped = 0
    for row, record in enumerate(records):
        if record["task"] not in BINDING_TASKS:
            continue
        targets = instruction_targets(instructions.get(record["state_id"], ""))
        if not targets:
            continue
        bindings = binding_labels(record["dom"], targets)
        options = clickable_refs(record["dom"], max_ref)
        if not options:
            continue
        for literal, (ref, node_index) in bindings.items():
            if ref >= max_ref:
                dropped += 1
                continue
            examples.append({
                "row": row, "split": record["split"], "literal": literal,
                "ref": ref, "node_index": node_index, "options": options,
            })
    if dropped:
        raise ValueError(
            f"{dropped} binding examples name an element beyond max_ref={max_ref}; "
            "the probe would train on a truncated candidate set"
        )
    return examples


def controls(examples) -> dict[str, float]:
    """Random guess, plus the DOM oracle that the labels are derived from."""
    random_hits, nearest_hits = 0.0, 0
    for item in examples:
        random_hits += 1.0 / len(item["options"])
        nearest = min(item["options"], key=lambda o: abs(o[1] - item["node_index"]))
        nearest_hits += int(nearest[0] == item["ref"])
    total = max(len(examples), 1)
    return {
        "random_guess": random_hits / total,
        # 1.0 by construction -- same rule the label uses. Reported as an upper
        # bound (binding is fully determined by the raw DOM), not a baseline.
        "dom_oracle": nearest_hits / total,
        "examples": len(examples),
    }


def run(args: argparse.Namespace) -> dict:
    records = list(iter_jsonl(args.records))
    instructions = {
        r["state_id"]: r.get("instruction", "")
        for r in iter_jsonl(args.instruction_records)
    }
    max_ref = args.max_ref or infer_max_ref(records, instructions)
    examples = build_examples(records, instructions, max_ref)
    by_split = defaultdict(list)
    for item in examples:
        by_split[item["split"]].append(item)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    provider = TokenProvider(REPRESENTATION, records, features=args.features, full_h=None)

    def batches(items, size, shuffle, generator=None):
        order = (torch.randperm(len(items), generator=generator).tolist()
                 if shuffle else range(len(items)))
        for start in range(0, len(items), size):
            picked = [items[i] for i in list(order)[start:start + size]]
            tok, pos, val = [], [], []
            for item in picked:
                tokens, _, positions, valid = provider.get(item["row"])
                tok.append(tokens.float()); pos.append(positions.float()); val.append(valid)
            yield (
                torch.stack(tok).to(device), torch.stack(pos).to(device),
                torch.stack(val).to(device),
                torch.tensor(np.stack([encode_literal(i["literal"]) for i in picked])).to(device),
                torch.tensor([i["ref"] for i in picked]).to(device),
                picked,
            )

    def fit_and_score(seed: int, ablate_literal: bool) -> float:
        torch.manual_seed(seed)
        model = BindingReader(provider.input_dim, max_ref).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=1e-4)
        generator = torch.Generator().manual_seed(seed)
        blank = (lambda t: torch.zeros_like(t)) if ablate_literal else (lambda t: t)
        for _ in range(args.epochs):
            model.train()
            for tokens, positions, valid, literal, ref, _ in batches(
                by_split["train"], args.batch_size, True, generator
            ):
                loss = nn.functional.cross_entropy(
                    model(tokens, positions, valid, blank(literal)), ref
                )
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        hits = total = 0
        with torch.inference_mode():
            for tokens, positions, valid, literal, ref, _ in batches(
                by_split[args.split], args.batch_size, False
            ):
                predicted = model(tokens, positions, valid, blank(literal)).argmax(-1)
                hits += int((predicted == ref).sum()); total += len(ref)
        return hits / max(total, 1)

    seeds = [fit_and_score(s, False) for s in range(args.seeds)]
    ablated = [fit_and_score(s, True) for s in range(args.seeds)]

    report = {
        "protocol": "target_element_binding_v1",
        "max_ref": max_ref,
        "tasks": list(BINDING_TASKS),
        "split": args.split,
        "train_examples": len(by_split["train"]),
        "eval_examples": len(by_split[args.split]),
        "probe_accuracy_per_seed": seeds,
        "probe_accuracy_mean": float(np.mean(seeds)),
        "probe_accuracy_sd": float(np.std(seeds, ddof=1)) if len(seeds) > 1 else 0.0,
        "literal_ablated_per_seed": ablated,
        "literal_ablated_mean": float(np.mean(ablated)),
        "controls": controls(by_split[args.split]),
        "note": "the conditioned probe must beat literal_ablated; dom_oracle is an "
                "upper bound derived by the same rule as the label, not a baseline",
    }
    write_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--instruction-records", required=True,
                        help="collection manifest with the real task instruction; the "
                             "feature manifest carries the fixed observation prompt")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation",
                        choices=("train", "validation", "test"))
    parser.add_argument("--max-ref", type=int,
                        help="element-id upper bound; inferred from the data when "
                             "omitted. MiniWoB refs and BrowserGym bids occupy "
                             "different ranges for the same pages")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:0")
    report = run(parser.parse_args())
    print(json.dumps({
        "eval_examples": report["eval_examples"],
        "probe": f"{report['probe_accuracy_mean']:.4f} ± {report['probe_accuracy_sd']:.4f}",
        "literal_ablated": f"{report['literal_ablated_mean']:.4f}",
        "controls": report["controls"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
