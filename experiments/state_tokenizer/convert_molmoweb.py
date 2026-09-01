"""Convert MolmoWeb-HumanTrajs parquet shards into ResidualMem state records.

The output is what ``WebObservation`` needs, not a serialized tree. The
tokenizer's own ``encode`` builds the tree itself --
``qformer_runtime.py:144`` calls ``synthetic_axtree(observation)`` inside
``encode`` -- so its only input is
``WebObservation(screenshot, user_text, captions, image_ids)``. Emitting a
pre-built ``dom`` here would either be ignored or force a bypass of ``encode``,
and bypassing it means MolmoWeb and WorldMemArena stop going through the same
function, which is the one property this conversion exists to preserve.

So the text modality is carried as its raw materials and composed into
``captions`` at encode time. That also keeps the two arms switchable without
re-converting:

  Arm A  page title + url        -- free, but not what the resampler was trained on
  Arm B  a generated caption     -- the shape WorldMemArena actually has

Arm B is the structural match. In the K32e training inputs
(``wma-xbar-fitcorpus-axtree/*.npz``) every record's ``synthetic_axtree`` holds
an image caption -- ``<ref=2 ... value="A Windows 10 desktop is shown with many
app shortcuts..."/>`` -- so a caption is what the DOM modality contained during
training, and WorldMemArena carries exactly one of them per observation.

Three things here are load-bearing and were established empirically against the
released data rather than assumed:

1. ``screenshot_t`` is the observation *before* ``action_t``.  Verified visually:
   step 1 of the Apple trajectory is the Chrome Incognito start page while its
   action is ``goto(apple.com)``, and step 2 is the Apple homepage.  So
   ``(o_t, a_t, o_{t+1})`` is a clean MDP triple with no off-by-one.

2. ``other_obs_t`` is aligned with ``screenshot_t`` -- also pre-action.  Tested
   by asking which action explains a url change: attributing it to ``action_t``
   gives precision 0.90, attributing it to ``action_{t+1}`` gives 0.50.  This
   matters because the url/title feed the text modality; had ``other_obs`` been
   the post-action state, the text would have announced the outcome of ``a_t``
   before it happened and the world model's conditional code length would be
   spuriously short.

3. **Step 1 is the sole exception.**  Every trajectory opens with a
   template-prepended ``goto`` whose ``other_obs.url`` is already the
   destination, not the start page.  Its text is therefore blanked; keeping it
   would leak exactly the outcome point 2 rules out.

``user_text`` is always empty, matching WorldMemArena web where it is empty in
all 956 observations. The real instruction is kept in ``instruction_raw`` for
ablations only, and nothing on the tokenizer path reads that field.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq

from ..world_model.schema_web import (
    ACTION_TYPE_NAMES,
    merge_consecutive_scrolls,
    parse_molmoweb_action,
)


PROTOCOL = "residualmem_molmoweb_records_v1"

# Actions that terminate a trajectory rather than change the world. Their
# observation is still a valid state; the transition leaving it is not.
TERMINAL_ACTIONS = frozenset({"ANSWER"})


def _split_for_sample(sample_id: str) -> str:
    """Deterministic 7:2:1 split, keyed by trajectory so none is ever split."""
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 10
    if bucket < 7:
        return "train"
    return "validation" if bucket < 9 else "test"


def _page_text(step: dict, *, is_first: bool) -> tuple[str, str]:
    """Arm A's raw materials: ``(page_title, url)``, both pre-action.

    Returned separately rather than pre-composed so the arm can be changed
    without re-converting. Blanked on step 1 -- see the module docstring.
    """
    if is_first:
        return "", ""
    other = step.get("other_obs") or {}
    titles = other.get("open_pages_titles") or []
    index = other.get("page_index") or 0
    title = ""
    if isinstance(titles, list) and 0 <= index < len(titles) and titles[index]:
        title = str(titles[index]).strip()
    return title, str(other.get("url") or "").strip()


def convert_shard(
    shard: Path, output: Path, *, images_dir: Path,
    limit: int | None = None, batch_size: int = 8,
) -> dict:
    """Stream one parquet shard to JSONL + PNGs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    reader = pq.ParquetFile(shard)

    stats = {
        "trajectories": 0, "states": 0, "transitions": 0, "dropped_missing_image": 0,
        "actions": {name: 0 for name in ACTION_TYPE_NAMES},
        "scroll_raw": 0, "scroll_merged": 0, "splits": {},
    }
    episode_index = 0
    with output.open("w", encoding="utf-8") as handle:
        for batch in reader.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                if limit is not None and stats["trajectories"] >= limit:
                    return stats
                written = _convert_row(row, handle, images_dir, episode_index, stats)
                if written:
                    episode_index += 1
                    stats["trajectories"] += 1
    return stats


def _convert_row(row, handle, images_dir: Path, episode_index: int, stats: dict) -> bool:
    sample_id = str(row["sample_id"])
    try:
        trajectory = json.loads(row["trajectory"])
        instruction = json.loads(row["instruction"])
    except (json.JSONDecodeError, TypeError):
        logging.warning("unreadable trajectory %s", sample_id)
        return False

    by_name = dict(zip(row["image_paths"], row["images"]))
    keys = sorted(trajectory, key=int)
    if len(keys) < 2:
        return False

    split = _split_for_sample(sample_id)
    actions = [parse_molmoweb_action(trajectory[k]) for k in keys]
    stats["scroll_raw"] += sum(a.type_name == "SCROLL" for a in actions)

    # Conversion is lossless. Every observation and every raw transition is
    # kept; the scroll runs are only *labelled*, not collapsed.
    #
    # Collapsing here was the first design and it was wrong: folding 11 scroll
    # steps into one action discards the 10 observations in between, which on
    # the preview shard threw away 35% of all states. The corpus is the single
    # biggest measured lever on this line (+297.42 bits per doubling, no
    # saturation), so destroying a third of it to reshape an action histogram is
    # a bad trade -- and an irreversible one, since undoing it means re-running
    # the whole conversion.
    #
    # ``scroll_run_*`` lets the sampler merge or downweight runs at training
    # time, which is reversible and can be ablated without touching this file.
    runs = _scroll_runs(actions)
    stats["scroll_merged"] += sum(1 for run in runs.values() if run[1] == 0)

    episode_id = f"molmoweb-ep{episode_index:06d}-{sample_id[:12]}"
    emitted = 0
    for step, key in enumerate(keys):
        node = trajectory[key]
        action = actions[step]
        name = node.get("screenshot")
        blob = by_name.get(name)
        if not blob:
            stats["dropped_missing_image"] += 1
            continue

        image_rel = Path(episode_id) / f"{step:04d}.png"
        target = images_dir / image_rel
        # The bytes come straight out of the parquet and never change, so a
        # re-run only needs to rewrite the JSONL. Skipping saves 58K writes to a
        # network filesystem.
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)

        page_title, url = _page_text(node, is_first=(step == 0))
        run_id, run_pos, run_len, run_dx, run_dy = runs.get(
            step, (-1, 0, 1, 0.0, 0.0)
        )
        record = {
            "state_id": f"{episode_id}-s{step:04d}",
            "task": "molmoweb",
            "episode_id": episode_id,
            "episode_index": episode_index,
            "step": step,
            "split": split,
            # WorldMemArena web has user_text empty in all 956 observations, so
            # the synthetic tree always carries "<no_instruction>". Emitting a
            # real instruction here would be a train/test modality difference.
            "user_text": "",
            "screenshot": image_rel.as_posix(),
            "image_id": f"{episode_id}-s{step:04d}",
            "image_w": int(node.get("image_w") or 0),
            "image_h": int(node.get("image_h") or 0),
            # Arm A's raw materials. Composed into ``captions`` at encode time;
            # Arm B's caption is joined in by state_id from a separate file.
            "page_title": page_title,
            "source_url": url,
            "action": _action_json(action),
            "is_terminal": action.type_name in TERMINAL_ACTIONS,
            # Scroll-run grouping for the sampler. ``run_id`` is -1 outside a
            # run; inside one, ``run_dx``/``run_dy`` is the whole run's summed
            # delta, so the sampler can emit a single merged transition from
            # position 0 to the observation after the run ends.
            "scroll_run_id": run_id,
            "scroll_run_pos": run_pos,
            "scroll_run_len": run_len,
            "scroll_run_dx": run_dx,
            "scroll_run_dy": run_dy,
            # Ablation-only; never read by the tokenizer path.
            "instruction_raw": instruction.get("high_level", ""),
        }
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats["actions"][action.type_name] += 1
        stats["states"] += 1
        emitted += 1

    if emitted:
        stats["transitions"] += max(emitted - 1, 0)
        stats["splits"][split] = stats["splits"].get(split, 0) + 1
    return emitted > 0


def _scroll_runs(actions) -> dict[int, tuple[int, int, int, float, float]]:
    """Label maximal same-direction scroll runs.

    Returns ``step -> (run_id, position_in_run, run_length, run_dx, run_dy)``.
    Boundaries follow ``merge_consecutive_scrolls``: a direction reversal or an
    axis change starts a new run, because a scroll down then up is a real round
    trip rather than one movement.
    """
    labels: dict[int, tuple[int, int, int, float, float]] = {}
    run_id = 0
    index = 0
    while index < len(actions):
        if actions[index].type_name != "SCROLL":
            index += 1
            continue
        end = index + 1
        while end < len(actions) and actions[end].type_name == "SCROLL":
            if len(merge_consecutive_scrolls([actions[end - 1], actions[end]])) != 1:
                break
            end += 1
        span = list(range(index, end))
        total_dx = float(sum(actions[i].dx for i in span))
        total_dy = float(sum(actions[i].dy for i in span))
        for position, step in enumerate(span):
            labels[step] = (run_id, position, len(span), total_dx, total_dy)
        run_id += 1
        index = end
    return labels


def _action_json(action) -> dict:
    payload = dataclasses.asdict(action)
    payload["payload"] = action.payload.decode("utf-8", "replace")
    payload["type_name"] = action.type_name
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    totals: dict = {}
    for index, shard in enumerate(args.shard):
        out = args.output.with_suffix(f".{index:02d}.jsonl")
        logging.info("converting %s -> %s", shard, out)
        stats = convert_shard(
            shard, out, images_dir=args.images,
            limit=args.limit, batch_size=args.batch_size,
        )
        logging.info("%s", json.dumps(stats, ensure_ascii=False))
        totals[str(shard)] = stats
    summary = args.output.with_suffix(".summary.json")
    summary.write_text(
        json.dumps({"protocol": PROTOCOL, "shards": totals}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("wrote %s", summary)


if __name__ == "__main__":
    main()
