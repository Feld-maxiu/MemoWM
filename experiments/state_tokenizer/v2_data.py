"""Data definitions for the Full-H / slot-aware tokenizer diagnostic.

All primary labels in this module are derived from the compact DOM string that
was actually included in the Qwen input.  Browser-only sidecars are never used
for the v2 gate.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .common import WORD_RE, sha256_file, write_json


STATIC_LABELS: tuple[str, ...] = (
    "has_button",
    "has_checkbox",
    "has_radio",
    "has_textbox",
    "has_select",
    "has_dialog",
)

DYNAMIC_LABELS: tuple[str, ...] = (
    *(f"checkbox_{index}_checked" for index in range(3)),
    *(f"radio_{index}_selected" for index in range(3)),
    *(f"textbox_{index}_nonempty" for index in range(3)),
    *(f"textbox_{index}_focused" for index in range(3)),
    "interactive_tampered",
    "has_random_value",
)

UNSUPPORTED_LABELS: dict[str, str] = {
    "button_disabled": "the existing manifest has zero positive examples",
    "dialog_open_dynamic": "dialog states occur only in the open/template state",
    "specific_selected_option": "option sidecar details were not stored in the manifest",
    "success_failure": "terminal observations and rewards were not persisted",
}

TEXTBOX_TAGS = {"input_text", "input_password", "textarea"}
INTERACTIVE_TAGS = {
    "input_checkbox", "input_radio", "input_text", "input_password",
    "textarea", "button", "input_button", "input_submit", "select", "option",
}
RANDOM_VALUE_RE = re.compile(r"state-(\d{5})")


def parse_compact_dom(dom: str) -> list[dict]:
    """Parse the stable ``compact_dom`` wire format without executing text."""
    elements: list[dict] = []
    for line_number, raw in enumerate(dom.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if not (line.startswith("<") and line.endswith("/>")):
            raise ValueError(f"invalid compact DOM row {line_number}: {line[:80]!r}")
        item: dict = {}
        for token in shlex.split(line[1:-2], posix=True):
            if "=" not in token:
                raise ValueError(f"invalid DOM attribute in row {line_number}: {token!r}")
            key, value = token.split("=", 1)
            item[key] = value
        flags = item.get("flags", "")
        item["flags"] = [int(value) for value in flags.split(",")] if flags else [0, 0, 0, 0]
        if len(item["flags"]) != 4:
            raise ValueError(f"expected four flags in row {line_number}")
        elements.append(item)
    if not elements:
        raise ValueError("compact DOM is empty")
    return elements


def _instruction_words(instruction: str) -> set[str]:
    return {match.group(0).lower() for match in WORD_RE.finditer(instruction)}


def derive_v2_targets(record: dict) -> dict:
    """Build masked static, dynamic, value, and leakage-aware text targets."""
    elements = parse_compact_dom(record["dom"])
    tags = [element.get("tag", "").lower() for element in elements]
    checkboxes = [element for element in elements if element.get("tag") == "input_checkbox"]
    radios = [element for element in elements if element.get("tag") == "input_radio"]
    textboxes = [element for element in elements if element.get("tag") in TEXTBOX_TAGS]
    interactive = [element for element in elements if element.get("tag") in INTERACTIVE_TAGS]

    static = {
        "has_button": bool(set(tags) & {"button", "input_button", "input_submit"}),
        "has_checkbox": bool(checkboxes),
        "has_radio": bool(radios),
        "has_textbox": bool(textboxes),
        "has_select": bool(set(tags) & {"select", "option"}),
        "has_dialog": any(
            "ui-dialog" in element.get("classes", "") or "dialog" in element.get("id", "")
            for element in elements
        ),
    }
    dynamic = {label: False for label in DYNAMIC_LABELS}
    dynamic_mask = {label: False for label in DYNAMIC_LABELS}

    for prefix, candidates in (("checkbox", checkboxes), ("radio", radios)):
        suffix = "checked" if prefix == "checkbox" else "selected"
        for index in range(3):
            label = f"{prefix}_{index}_{suffix}"
            if index < len(candidates):
                dynamic_mask[label] = True
                dynamic[label] = candidates[index].get("value", "").lower() == "true"

    random_digits: list[int] | None = None
    for index in range(3):
        for suffix in ("nonempty", "focused"):
            label = f"textbox_{index}_{suffix}"
            if index >= len(textboxes):
                continue
            dynamic_mask[label] = True
            if suffix == "nonempty":
                dynamic[label] = bool(textboxes[index].get("value", ""))
            else:
                dynamic[label] = bool(textboxes[index]["flags"][0])
        if index < len(textboxes) and random_digits is None:
            match = RANDOM_VALUE_RE.search(textboxes[index].get("value", ""))
            if match:
                random_digits = [int(value) for value in match.group(1)]

    dynamic_mask["interactive_tampered"] = bool(interactive)
    dynamic["interactive_tampered"] = any(bool(element["flags"][1]) for element in interactive)
    dynamic_mask["has_random_value"] = bool(textboxes)
    dynamic["has_random_value"] = random_digits is not None

    visible_words = set(record["probe"]["visible_words"])
    instruction_words = _instruction_words(record["instruction"])
    return {
        "static": {label: bool(static[label]) for label in STATIC_LABELS},
        "dynamic": {label: bool(dynamic[label]) for label in DYNAMIC_LABELS},
        "dynamic_mask": {label: bool(dynamic_mask[label]) for label in DYNAMIC_LABELS},
        "random_digits": random_digits if random_digits is not None else [-1] * 5,
        "random_value_mask": random_digits is not None,
        "instruction_overlap_words": sorted(visible_words & instruction_words),
        "dom_only_words": sorted(visible_words - instruction_words),
    }


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def _state_signature(record: dict) -> tuple:
    target = record["v2"]
    return (
        min(int(record["step"]), 3),
        sum(target["dynamic"].get(f"checkbox_{index}_checked", False) for index in range(3)),
        sum(target["dynamic"].get(f"radio_{index}_selected", False) for index in range(3)),
        sum(target["dynamic"].get(f"textbox_{index}_nonempty", False) for index in range(3)),
        sum(target["dynamic"].get(f"textbox_{index}_focused", False) for index in range(3)),
        int(target["dynamic"]["interactive_tampered"]),
        int(target["random_value_mask"]),
    )


def _proportional_bucket_sample(candidates: Sequence[dict], quota: int, rng: np.random.Generator) -> list[dict]:
    if quota > len(candidates):
        raise ValueError(f"quota {quota} exceeds {len(candidates)} episode representatives")
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for record in candidates:
        buckets[_state_signature(record)].append(record)
    for values in buckets.values():
        rng.shuffle(values)
    keys = sorted(buckets)
    raw = {key: quota * len(buckets[key]) / len(candidates) for key in keys}
    counts = {key: min(len(buckets[key]), int(math.floor(raw[key]))) for key in keys}
    remaining = quota - sum(counts.values())
    ranked = sorted(keys, key=lambda key: (-(raw[key] - counts[key]), key))
    while remaining:
        progressed = False
        for key in ranked:
            if counts[key] < len(buckets[key]):
                counts[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("could not fill proportional bucket quota")
    selected = [record for key in keys for record in buckets[key][:counts[key]]]
    rng.shuffle(selected)
    return selected


def _capped_task_quotas(capacities: dict[str, int], total: int) -> dict[str, int]:
    """Balanced water-filling that never exceeds unique-episode capacity."""
    if total > sum(capacities.values()):
        raise ValueError(f"requested {total} states from only {sum(capacities.values())} episodes")
    quotas = {task: 0 for task in capacities}
    remaining = total
    while remaining:
        candidates = [task for task in capacities if quotas[task] < capacities[task]]
        if not candidates:
            raise RuntimeError("task quota water-filling ran out of capacity")
        task = min(candidates, key=lambda name: (quotas[name], name))
        quotas[task] += 1
        remaining -= 1
    return quotas


def select_subset(records: Sequence[dict], split_sizes: dict[str, int], seed: int) -> list[dict]:
    """Select exact split sizes with at most one state per episode."""
    annotated = []
    for global_index, raw in enumerate(records):
        record = dict(raw)
        record["global_index"] = global_index
        record["v2"] = derive_v2_targets(record)
        annotated.append(record)
    tasks = sorted({record["task"] for record in annotated})
    selected: list[dict] = []
    for split, total in split_sizes.items():
        task_episodes: dict[str, dict[str, list[dict]]] = {}
        for task in tasks:
            group = [record for record in annotated if record["split"] == split and record["task"] == task]
            episodes: dict[str, list[dict]] = defaultdict(list)
            for record in group:
                episodes[record["episode_id"]].append(record)
            task_episodes[task] = episodes
        quotas = _capped_task_quotas(
            {task: len(episodes) for task, episodes in task_episodes.items()}, total
        )
        for task in tasks:
            episodes = task_episodes[task]
            rng = np.random.default_rng(_stable_seed(seed, split, task))
            representatives = []
            for episode_id in sorted(episodes):
                values = sorted(episodes[episode_id], key=lambda record: record["step"])
                # Dynamic diagnostics need both untouched and post-action states.
                # Give the initial state half of the mass and distribute the
                # other half uniformly over all available post-action states.
                if len(values) == 1 or float(rng.random()) < 0.5:
                    representatives.append(values[0])
                else:
                    representatives.append(values[1 + int(rng.integers(0, len(values) - 1))])
            selected.extend(_proportional_bucket_sample(representatives, quotas[task], rng))
    selected.sort(key=lambda record: record["global_index"])
    if len({record["episode_id"] for record in selected}) != len(selected):
        raise RuntimeError("subset contains repeated episodes")
    actual = Counter(record["split"] for record in selected)
    if dict(actual) != split_sizes:
        raise RuntimeError(f"subset split mismatch: {dict(actual)} != {split_sizes}")
    return selected


def _word_vocab(records: Sequence[dict], key: str, limit: int, minimum: int = 20) -> list[str]:
    train = [record for record in records if record["split"] == "train"]
    counts = Counter(word for record in train for word in set(record["v2"][key]))
    eligible = [
        (word, count) for word, count in counts.items()
        if count >= minimum and len(train) - count >= minimum
    ]
    return [word for word, _ in sorted(eligible, key=lambda item: (-item[1], item[0]))[:limit]]


def label_coverage(records: Sequence[dict]) -> dict:
    coverage = {}
    for label in DYNAMIC_LABELS:
        per_split = {}
        for split in ("train", "validation", "test"):
            values = [
                int(record["v2"]["dynamic"][label])
                for record in records
                if record["split"] == split and record["v2"]["dynamic_mask"][label]
            ]
            positives = sum(values)
            per_split[split] = {
                "examples": len(values),
                "positives": positives,
                "negatives": len(values) - positives,
            }
        varying_tasks = []
        for task in sorted({record["task"] for record in records}):
            values = {
                record["v2"]["dynamic"][label]
                for record in records
                if record["task"] == task and record["v2"]["dynamic_mask"][label]
            }
            if len(values) > 1:
                varying_tasks.append(task)
        eligible = (
            per_split["train"]["positives"] >= 50
            and per_split["train"]["negatives"] >= 50
            and per_split["validation"]["positives"] >= 15
            and per_split["validation"]["negatives"] >= 15
            and per_split["test"]["positives"] >= 30
            and per_split["test"]["negatives"] >= 30
            and bool(varying_tasks)
        )
        coverage[label] = {
            "eligible": eligible,
            "varying_tasks": varying_tasks,
            "splits": per_split,
        }
    return coverage


def write_subset(
    records: Sequence[dict],
    output: str | Path,
    *,
    source_manifest: str | Path,
    seed: int,
) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "records": len(records),
        "split_counts": dict(Counter(record["split"] for record in records)),
        "task_counts": dict(Counter(record["task"] for record in records)),
        "unique_episodes": len({record["episode_id"] for record in records}),
        "seed": seed,
        "source_manifest": str(Path(source_manifest).resolve()),
        "source_sha256": sha256_file(source_manifest),
        "subset_sha256": sha256_file(output),
        "dynamic_coverage": label_coverage(records),
        "instruction_overlap_vocab": _word_vocab(records, "instruction_overlap_words", 64),
        "dom_only_vocab": _word_vocab(records, "dom_only_words", 128),
        "unsupported_labels": UNSUPPORTED_LABELS,
    }
    write_json(output.with_suffix(".summary.json"), summary)
    return summary
