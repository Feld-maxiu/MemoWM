"""Collect MiniWoB state records through BrowserGym, with AXTree as the state text.

The v7 collector (``collect_miniwob``) is kept untouched so that line stays
reproducible; this is a sibling, not a replacement. What changes is the source of
the state text and, consequently, the action space: BrowserGym drives Playwright
and addresses elements by ``bid``, where Farama's MiniWoB drove Selenium and
addressed them by MiniWoB's own ``ref``.

The exploration policy is deliberately identical to v7's -- the same scripted
cascade, the same epsilon-mixed random draw -- so that a difference in the
resulting representation is attributable to the observation and not to a
different distribution of states.

Runs under ``browsergym-venv``, which has Playwright but no torch or jax. Nothing
imported here may pull those in.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from unittest.mock import patch

import gymnasium
import numpy as np
from PIL import Image

import browsergym.miniwob  # noqa: F401  -- registers the browsergym/miniwob.* ids
from browsergym.core.env import _get_global_playwright
from browsergym.utils.obs import flatten_dom_to_str

from .common import (
    PILOT_TASKS,
    axtree_rows,
    compact_axtree,
    derive_probe_labels_axtree,
    split_for_episode,
    write_json,
)

# Roles worth acting on, named by the tags ``compact_axtree`` maps them to.
INTERACTIVE_TAGS = {
    "input_checkbox", "input_radio", "input_text", "input_password", "textarea",
    "button", "a", "label", "input_button", "input_submit", "option", "select",
}
TYPING_TAGS = {"input_text", "input_password", "textarea"}


def _enable_browser_reuse(env) -> None:
    """Reuse both BrowserGym Chromium processes while replacing their contexts.

    BrowserGym 0.14.3 launches one Chromium for the task and another for its
    chat UI on every reset.  MiniWoB teardown has no browser-level state, and a
    fresh incognito context still isolates each episode.  This wrapper keeps
    the two processes but lets BrowserGym execute its unchanged reset path.
    It is opt-in because production use requires exact overlap validation.
    """
    base = env.unwrapped
    original_reset = base.reset

    def reset_with_reused_browsers(*args, **kwargs):
        main_browser = base.browser
        chat_browser = base.chat.browser if base.chat is not None else None
        if (
            main_browser is None
            or chat_browser is None
            or not main_browser.is_connected()
            or not chat_browser.is_connected()
        ):
            return original_reset(*args, **kwargs)

        chromium = _get_global_playwright().chromium
        # BrowserEnv.reset launches task Chromium first; Chat.__init__ launches
        # chat Chromium second.  Closing either old context remains untouched.
        with (
            patch.object(main_browser, "close", return_value=None),
            patch.object(chat_browser, "close", return_value=None),
            patch.object(chromium, "launch", side_effect=(main_browser, chat_browser)),
        ):
            return original_reset(*args, **kwargs)

    base.reset = reset_with_reused_browsers


def _quote(value: str) -> str:
    """BrowserGym actions are python source, so values must be literals."""
    return json.dumps(str(value))


def _typed_text(rng: np.random.Generator) -> str:
    """The five-digit payload the value-exact metric is measured on."""
    return f"state-{int(rng.integers(0, 100000)):05d}"


def _actable(rows):
    return [
        row for row in rows
        if row["tag"] in INTERACTIVE_TAGS and row["ref"] not in ("-1", "0")
    ]


def _describe(row, kind, text, policy):
    return {"type": kind, "ref": int(row["ref"]), "text": text,
            "tag": row["tag"], "policy": policy}


# Everything ``axtree_rows`` reads, and nothing else. Chrome ships each name with
# a ``sources`` chain recording which attribute it was computed from, which is
# most of the raw payload's size and is never consulted.
_AXTREE_KEEP = ("nodeId", "parentId", "childIds", "browsergym_id", "ignored")


def _prunable_axtree(axtree_object: dict, sidecar) -> dict:
    nodes = []
    for node in axtree_object.get("nodes", ()):
        kept = {key: node[key] for key in _AXTREE_KEEP if key in node}
        for key in ("role", "name", "value"):
            field = node.get(key)
            if isinstance(field, dict) and field.get("value") not in (None, ""):
                kept[key] = {"value": field["value"]}
        properties = [
            {"name": item["name"], "value": {"value": item["value"].get("value")}}
            for item in node.get("properties") or ()
            if isinstance(item, dict) and isinstance(item.get("value"), dict)
        ]
        if properties:
            kept["properties"] = properties
        nodes.append(kept)
    return {"nodes": nodes}


def _action_for(row, rng, policy):
    """Build the BrowserGym action string plus the record of what it was."""
    bid = row["ref"]
    if row["tag"] in TYPING_TAGS:
        text = _typed_text(rng)
        return (f"fill({_quote(bid)}, {_quote(text)})",
                _describe(row, "FILL", text, policy))
    if row["tag"] == "option" and row["text"]:
        # select_option addresses the <select>, not the <option>
        return (f"select_option({_quote(row['parent'])}, {_quote(row['text'])})",
                _describe(row, "SELECT_OPTION", row["text"], policy))
    return f"click({_quote(bid)})", _describe(row, "CLICK", None, policy)


def _random_action(rows, rng: np.random.Generator):
    """Uniformly sample one legal action over the page's interactive elements.

    The scripted policy is a deterministic priority cascade, so ``u_t`` is very
    nearly a function of ``y_t`` -- which makes "does the action carry
    information beyond the state?" unanswerable no matter how much data is
    collected. Sampling uniformly over legal actions breaks that dependence.
    """
    candidates = _actable(rows)
    if not candidates:
        return None, None
    return _action_for(candidates[int(rng.integers(len(candidates)))], rng, "random")


def _make_action(rows, rng: np.random.Generator, step: int):
    """v7's scripted cascade, re-expressed over AXTree rows."""
    candidates = _actable(rows)
    if not candidates:
        return None, None

    def shuffled(items):
        items = list(items)
        rng.shuffle(items)
        return items

    unchecked = [
        row for row in candidates
        if row["tag"] in {"input_checkbox", "input_radio"} and row["value"] != "true"
    ]
    if unchecked:
        return _action_for(unchecked[0], rng, "scripted")

    textboxes = shuffled(row for row in candidates if row["tag"] in TYPING_TAGS)
    if textboxes and step == 0:
        return _action_for(textboxes[0], rng, "scripted")

    options = shuffled(row for row in candidates if row["tag"] == "option")
    if options:
        return _action_for(options[0], rng, "scripted")

    clickable = shuffled(
        row for row in candidates
        if row["tag"] in {"button", "a", "label", "input_button", "input_submit"}
    )
    if clickable:
        return _action_for(clickable[0], rng, "scripted")
    return None, None


def _choose_action(rows, rng, step, random_prob: float):
    """Scripted cascade with probability ``1 - random_prob``, else a random action.

    Pure exploration would maximise action entropy but terminate multi-step tasks
    early (a wrong click ends form-sequence), starving exactly the deep states the
    history question needs. Mixing keeps trajectory depth while still making half
    the transitions interventional.
    """
    if random_prob > 0.0 and float(rng.random()) < random_prob:
        action, described = _random_action(rows, rng)
        if action is not None:
            return action, described
    return _make_action(rows, rng, step)


def _save_state(
    output: Path,
    task: str,
    task_index: int,
    episode_index: int,
    step: int,
    observation: dict,
    rows: list[dict],
    tampered: set[str],
    *,
    prefix: str = "",
    force_split: str | None = None,
    environment_seed: int | None = None,
) -> dict | None:
    """One record. ``dom`` holds the AXTree text; ``obs_kind`` says so.

    The field keeps its v7 name because fifteen call sites read it and the
    contents play the same role -- the state text handed to Qwen. ``obs_kind`` is
    what makes a store self-describing rather than requiring the reader to
    remember which collection produced it.

    ``prefix`` prepends to ``state_id``/``episode_id`` so an appended collection
    sorts after everything already extracted. ``merge_records`` sorts by
    ``state_id`` and writes the row position as ``global_index``, and the 55 GB
    feature store is keyed by that index, so a prefix that sorts *last* is what
    makes appending incremental instead of a full re-extraction.

    ``environment_seed`` records the seed actually passed to ``env.reset``. It
    is written so that "the new collection did not regenerate old episodes" can
    be asserted against the data rather than argued from the code.
    """
    screenshot = np.asarray(observation.get("screenshot"))
    if not rows or screenshot.ndim != 3 or screenshot.size == 0:
        return None
    # MiniWoB's own ``tampered`` bit is unreachable here: BrowserGym removes the
    # click listener that sets it (``remove_human_display``). Our own actions are
    # the only interactions with the page, so tracking them in Python is exact.
    marked = [{"bid": bid, "tampered": True} for bid in sorted(tampered)]
    state_id = f"{prefix}task{task_index:02d}-ep{episode_index:06d}-t{step:02d}"
    image_rel = Path("images") / f"{state_id}.png"
    image_path = output / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(screenshot.astype(np.uint8), mode="RGB").save(image_path, optimize=True)
    marked_rows = axtree_rows(observation["axtree_object"], marked)
    record = {
        "state_id": state_id,
        "task": task,
        "episode_id": f"{prefix}task{task_index:02d}-ep{episode_index:06d}",
        "episode_index": episode_index,
        "step": step,
        "split": force_split or split_for_episode(episode_index),
        "instruction": str(observation.get("goal", "")),
        "fields": [],
        "obs_kind": "axtree",
        "dom": compact_axtree(observation["axtree_object"], marked),
        # Same states, serialized the old way: lets the AXTree-vs-DOM comparison
        # be run within one environment instead of across two.
        "dom_control": flatten_dom_to_str(observation["dom_object"]),
        # The pruned tree the serializer actually reads. Keeping it makes a change
        # to the serialization rules a re-serialize plus re-pool, rather than a
        # re-collection -- Chrome's ``sources`` chains are what make the raw
        # payload bulky and nothing downstream looks at them.
        "axtree_raw": _prunable_axtree(observation["axtree_object"], marked),
        "screenshot": image_rel.as_posix(),
        "probe": derive_probe_labels_axtree(marked_rows),
    }
    if environment_seed is not None:
        record["environment_seed"] = int(environment_seed)
    return record


def _resume_state(path: Path, episode_stride: int) -> tuple[dict[str, dict], list[dict]]:
    """Per-task progress from an existing shard, minus its last episode.

    A run that spans hours will be interrupted -- the session container that owns
    the process can be rebuilt, and every worker dies at once with no manifest.
    Episode indices are a deterministic function of (task, worker, iteration), so
    resuming is just "continue from the next index"; the only care needed is the
    final episode, which is flushed as a unit and can therefore be short if the
    kill landed mid-write. That episode is dropped and re-collected rather than
    trusted.
    """
    if not path.exists():
        return {}, []
    kept: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            kept.append(json.loads(line))
        except json.JSONDecodeError:
            # only ever the final line, if the process died mid-write
            break

    last_episode: dict[str, int] = {}
    for record in kept:
        task = record["task"]
        index = int(record["episode_index"])
        if index > last_episode.get(task, -1):
            last_episode[task] = index

    surviving = [
        record for record in kept
        if int(record["episode_index"]) != last_episode.get(record["task"])
    ]
    progress: dict[str, dict] = {}
    episodes: dict[str, set[int]] = {}
    for record in surviving:
        entry = progress.setdefault(
            record["task"], {"states": 0, "episodes": 0, "next_episode": 0}
        )
        entry["states"] += 1
        episodes.setdefault(record["task"], set()).add(int(record["episode_index"]))
        entry["next_episode"] = max(
            entry["next_episode"], int(record["episode_index"]) + episode_stride
        )
    for task, values in episodes.items():
        progress[task]["episodes"] = len(values)
    for task, index in last_episode.items():
        entry = progress.setdefault(
            task, {"states": 0, "episodes": 0, "next_episode": 0}
        )
        # re-collect the dropped episode
        entry["next_episode"] = max(entry["next_episode"], index)
    return progress, surviving


def collect(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = tuple(args.tasks or PILOT_TASKS)
    # ``--only-tasks`` filters *without* renumbering. ``task_index`` feeds both
    # the environment seed (``seed + task_index * 1e6 + episode_index``) and the
    # ``task{NN}`` id, so re-enumerating a filtered list would silently move
    # every later task into a different seed band and change what ``task03``
    # means relative to an earlier collection. Filter the assignment; never the
    # index source.
    chosen = set(args.only_tasks) if args.only_tasks else None
    if chosen is not None:
        unknown = sorted(chosen - set(tasks))
        if unknown:
            raise SystemExit(f"--only-tasks names unknown tasks: {unknown}")
    lane_mode = args.target_episodes is not None
    if args.assigned_task_index is None:
        assigned = [
            (index, task) for index, task in enumerate(tasks)
            if index % args.num_workers == args.worker_id
            and (chosen is None or task in chosen)
        ]
    else:
        if not 0 <= args.assigned_task_index < len(tasks):
            raise SystemExit(
                f"--assigned-task-index={args.assigned_task_index} is outside "
                f"the canonical task list of length {len(tasks)}"
            )
        selected = tasks[args.assigned_task_index]
        if chosen is not None and selected not in chosen:
            raise SystemExit("--assigned-task-index is excluded by --only-tasks")
        assigned = [(args.assigned_task_index, selected)]
    # Budget is per *collected* task, so skipping tasks concentrates the target
    # on the ones that remain rather than silently under-collecting.
    per_task = int(np.ceil(args.target_states / len(chosen or tasks)))
    shard_name = args.shard_name or f"worker{args.worker_id:02d}"
    records_path = output / f"records-{shard_name}.jsonl"
    counts = {"train": 0, "validation": 0, "test": 0}
    task_counts: dict[str, int] = {}
    task_episode_counts: dict[str, int] = {}
    aborted: list[str] = []
    total = 0

    progress: dict[str, dict] = {}
    if args.resume:
        progress, surviving = _resume_state(records_path, args.episode_stride)
        if progress:
            with records_path.open("w", encoding="utf-8") as handle:
                for item in surviving:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            total = len(surviving)
            for item in surviving:
                counts[item["split"]] += 1
            logging.info("worker=%d resuming with %d states across %d tasks",
                         args.worker_id, total, len(progress))

    mode = "a" if args.resume and progress else "w"
    with records_path.open(mode, encoding="utf-8") as handle:
        for task_index, task in assigned:
            task_total = progress.get(task, {}).get("states", 0)
            episode_total = progress.get(task, {}).get("episodes", 0)
            episode_index = progress.get(task, {}).get(
                "next_episode", args.episode_start
            )

            def complete() -> bool:
                return (
                    episode_total >= args.target_episodes
                    if lane_mode else task_total >= per_task
                )

            if complete():
                logging.info("worker=%d task=%s already complete (%d/%d)",
                             args.worker_id, task,
                             episode_total if lane_mode else task_total,
                             args.target_episodes if lane_mode else per_task)
                task_counts[task] = task_total
                task_episode_counts[task] = episode_total
                continue
            # "miniwob/click-button-v1" -> "browsergym/miniwob.click-button"
            slug = re.sub(r"^miniwob/|-v1$", "", task)
            env = gymnasium.make(
                f"browsergym/miniwob.{slug}",
                headless=True,
                wait_for_user_message=False,
                timeout=args.timeout_ms,
                pre_observation_delay=args.pre_observation_delay,
            )
            if args.reuse_browser:
                _enable_browser_reuse(env)
            try:
                consecutive_failures = 0
                while not complete():
                    seed = args.seed + task_index * 1_000_000 + episode_index
                    episode: list[dict] = []
                    tampered: set[str] = set()
                    episode_ok = False
                    try:
                        observation, _ = env.reset(seed=seed)
                        rng = np.random.default_rng(seed)
                        # The action is only known after its state has been built,
                        # so the episode is buffered and flushed once complete:
                        # record[t]["action"] is what led to record[t+1].
                        for step in range(args.max_steps + 1):
                            rows = axtree_rows(observation["axtree_object"])
                            record = _save_state(
                                output, task, task_index, episode_index, step,
                                observation, rows, tampered,
                                prefix=args.state_id_prefix,
                                force_split=args.force_split,
                                environment_seed=seed,
                            )
                            if record is not None:
                                record["action"] = None
                                episode.append(record)
                                if not lane_mode:
                                    counts[record["split"]] += 1
                                    task_total += 1
                                    total += 1
                                    if total % args.log_every == 0:
                                        logging.info(
                                            "worker=%d records=%d current_task=%s "
                                            "task_records=%d/%d",
                                            args.worker_id, total, task, task_total, per_task,
                                        )
                                if not lane_mode and task_total >= per_task:
                                    break
                            action, described = _choose_action(
                                rows, rng, step, args.random_action_prob
                            )
                            if action is None:
                                break
                            if record is not None:
                                record["action"] = described
                            observation, _, terminated, truncated, _ = env.step(action)
                            tampered.add(str(described["ref"]))
                            if observation.get("last_action_error"):
                                logging.debug("action rejected: %s -> %s",
                                              action, observation["last_action_error"])
                                break
                            if terminated or truncated:
                                break
                        if lane_mode and not episode:
                            raise RuntimeError(
                                f"lane episode produced no state: task={task} "
                                f"episode={episode_index}"
                            )
                    except Exception:
                        logging.exception("Episode failed: task=%s episode=%d", task, episode_index)
                        consecutive_failures += 1
                        # A dead browser cannot recover on its own: BrowserGym's
                        # reset() closes the previous context first, so once the
                        # context is gone every later reset raises too. Without
                        # this guard the loop keeps incrementing episode_index and
                        # burns CPU forever while writing nothing -- the process
                        # looks alive and the record count simply stops moving.
                        if consecutive_failures >= args.max_consecutive_failures:
                            logging.error(
                                "worker=%d task=%s aborting after %d consecutive episode "
                                "failures; collected %d/%d",
                                args.worker_id, task, consecutive_failures,
                                episode_total if lane_mode else task_total,
                                args.target_episodes if lane_mode else per_task,
                            )
                            break
                    else:
                        consecutive_failures = 0
                        episode_ok = True
                        episode_total += 1
                        if lane_mode:
                            for item in episode:
                                counts[item["split"]] += 1
                            task_total += len(episode)
                            total += len(episode)
                            if episode_total % args.log_every == 0:
                                logging.info(
                                    "shard=%s episodes=%d/%d records=%d task=%s",
                                    shard_name, episode_total, args.target_episodes,
                                    total, task,
                                )
                    finally:
                        if episode_ok or not lane_mode:
                            for item in episode:
                                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                            handle.flush()
                    if lane_mode:
                        if episode_ok:
                            episode_index += args.episode_stride
                    else:
                        episode_index += args.episode_stride
            finally:
                env.close()
            task_counts[task] = task_total
            task_episode_counts[task] = episode_total
            if not complete():
                aborted.append(task)

    summary = {
        "worker_id": args.worker_id,
        "obs_kind": "axtree",
        "random_action_prob": args.random_action_prob,
        "max_steps": args.max_steps,
        "pre_observation_delay": args.pre_observation_delay,
        "reuse_browser": args.reuse_browser,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "shard_name": shard_name,
        "lane_mode": lane_mode,
        "assigned_task_index": args.assigned_task_index,
        "episode_start": args.episode_start,
        "episode_stride": args.episode_stride,
        "target_episodes": args.target_episodes,
        "records": total,
        "split_counts": counts,
        "task_counts": task_counts,
        # Named so a retry wrapper can tell "finished" from "gave up"; main()
        # exits non-zero when this is non-empty.
        "aborted_tasks": aborted,
        "records_path": str(records_path),
        "task_episode_counts": {
            task: task_episode_counts.get(task, 0) for task in task_counts
        },
    }
    write_json(output / f"collect-{shard_name}.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-states", type=int, default=10_000)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--assigned-task-index", type=int,
        help="canonical task index for deterministic episode-lane collection",
    )
    parser.add_argument(
        "--episode-start", type=int,
        help="first episode index; defaults to worker-id in the legacy protocol",
    )
    parser.add_argument(
        "--episode-stride", type=int,
        help="episode-index stride; defaults to num-workers in the legacy protocol",
    )
    parser.add_argument(
        "--target-episodes", type=int,
        help="collect this many complete episodes for the assigned lane instead "
             "of stopping after target-states",
    )
    parser.add_argument(
        "--shard-name",
        help="unique [A-Za-z0-9._-]+ shard suffix; defaults to workerNN",
    )
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--random-action-prob", type=float, default=0.5,
                        help="probability of replacing the scripted action with a "
                             "uniform draw over the page's legal actions; the "
                             "scripted policy is near-deterministic given the "
                             "observation, so a positive value is what makes the "
                             "action carry information beyond the state")
    parser.add_argument("--timeout-ms", type=int, default=5_000)
    parser.add_argument(
        "--pre-observation-delay", type=float, default=0.5,
        help="seconds BrowserGym waits before extracting each observation; the "
             "historical protocol is 0.5, and a faster value must pass exact "
             "record/screenshot overlap validation before production use",
    )
    parser.add_argument(
        "--reuse-browser", action=argparse.BooleanOptionalAction, default=False,
        help="reuse BrowserGym's task/chat Chromium processes across episodes "
             "while creating fresh contexts; opt-in pending exact overlap validation",
    )
    parser.add_argument("--max-consecutive-failures", type=int, default=20,
                        help="give up on a task after this many episodes fail in a "
                             "row. A dead browser makes every later reset fail, so "
                             "without a cap the loop spins forever writing nothing")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="continue an interrupted shard instead of truncating it; "
                             "the last episode of each task is dropped and redone "
                             "because a kill mid-flush can leave it short")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument(
        "--only-tasks", nargs="*",
        help="collect just these tasks, keeping every task's canonical index. "
             "Use this rather than --tasks when appending to an existing "
             "collection: --tasks replaces the index source, which moves "
             "task_index and therefore both the seed band and the taskNN id",
    )
    parser.add_argument(
        "--state-id-prefix", default="",
        help="prepended to state_id/episode_id. Must sort AFTER 'task' so an "
             "appended collection occupies a contiguous block at the end of the "
             "merged manifest and leaves every existing global_index untouched",
    )
    parser.add_argument(
        "--force-split", choices=("train", "validation", "test"), default=None,
        help="assign every collected episode to this split instead of using "
             "episode_index %% 10. Appending to train only is what keeps the "
             "held-out sets byte-identical to the earlier collection",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    args.episode_start = (
        args.worker_id if args.episode_start is None else args.episode_start
    )
    args.episode_stride = (
        args.num_workers if args.episode_stride is None else args.episode_stride
    )
    if not 0 <= args.worker_id < args.num_workers:
        parser.error("worker-id must be in [0, num-workers)")
    if args.episode_start < 0 or args.episode_stride < 1:
        parser.error("episode-start must be non-negative and episode-stride positive")
    if args.target_episodes is not None and args.target_episodes < 1:
        parser.error("target-episodes must be positive")
    lane_values = (
        args.assigned_task_index, args.target_episodes, args.shard_name,
    )
    if any(value is not None for value in lane_values) and not all(
        value is not None for value in lane_values
    ):
        parser.error(
            "lane collection requires assigned-task-index, target-episodes, "
            "and shard-name together"
        )
    lane_mode = args.target_episodes is not None
    if args.shard_name and not re.fullmatch(r"[A-Za-z0-9._-]+", args.shard_name):
        parser.error("shard-name must match [A-Za-z0-9._-]+")
    if lane_mode:
        if args.state_id_prefix or args.force_split:
            parser.error("lane collection forbids state-id-prefix and force-split")
        if args.assigned_task_index is not None:
            if args.episode_start % len(PILOT_TASKS) != args.assigned_task_index:
                parser.error(
                    "episode-start must preserve the original task_index + 12k lattice"
                )
            if args.episode_stride % len(PILOT_TASKS):
                parser.error(
                    "episode-stride must be a multiple of the original 12-worker stride"
                )
    if not 0.0 <= args.random_action_prob <= 1.0:
        parser.error("random-action-prob must be in [0, 1]")
    if args.pre_observation_delay < 0.0:
        parser.error("pre-observation-delay must be non-negative")
    # The block property this whole scheme rests on: merge_records sorts by
    # state_id, so a prefixed id sorts after every unprefixed one only if its
    # first character sorts after 't'. Checked rather than assumed, because a
    # prefix like "b9" would quietly interleave and re-key the feature store.
    if args.state_id_prefix and args.state_id_prefix[0] <= "t":
        parser.error(
            f"--state-id-prefix {args.state_id_prefix!r} must start with a "
            "character sorting after 't', otherwise the appended block does not "
            "sort last and existing global_index values move"
        )
    # Each worker's episode indices are worker_id + k * num_workers, and
    # split_for_episode buckets on index % 10. When num_workers is a multiple of
    # 5 a worker only ever reaches ceil(10/num_workers) buckets, so whole workers
    # -- and therefore whole tasks -- land in a single split. Nothing downstream
    # notices; the split file just quietly stops covering all twelve tasks.
    # --force-split bypasses split_for_episode entirely, so the hazard cannot
    # arise and the constraint does not apply.
    if not lane_mode and args.num_workers % 5 == 0 and args.force_split is None:
        parser.error(
            f"num-workers={args.num_workers} is a multiple of 5: each worker would "
            "reach too few split buckets and tasks would be confined to one split"
        )
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    summary = collect(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["aborted_tasks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
