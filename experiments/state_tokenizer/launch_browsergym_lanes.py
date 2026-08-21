"""Launch the high-concurrency deterministic v8 BrowserGym lane plan."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from math import ceil
from pathlib import Path


# Derived from the first ~150 episodes/task of the canonical collector.  Two
# known one-state tasks use the exact worst-case episode count; the rest include
# 35% headroom.  If assembly still finds a short task, increasing its target and
# relaunching resumes every lane without changing any episode id or seed.
DEFAULT_PLAN: dict[int, tuple[int, int]] = {
    0: (24, 251),
    1: (32, 261),
    2: (12, 146),
    3: (12, 148),
    4: (12, 144),
    5: (24, 236),
    6: (32, 261),
    7: (16, 230),
    8: (16, 301),
    9: (24, 233),
    10: (12, 233),
    11: (12, 215),
}
REFERENCE_TARGET_STATES = 100_000
REFERENCE_STATES_PER_TASK = ceil(REFERENCE_TARGET_STATES / len(DEFAULT_PLAN))


def _alive(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _complete(summary_path: Path, task: int, target: int) -> bool:
    if not summary_path.exists():
        return False
    try:
        value = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    counts = value.get("task_episode_counts", {})
    return (
        value.get("assigned_task_index") == task
        and not value.get("aborted_tasks")
        and len(counts) == 1
        and next(iter(counts.values())) >= target
    )


def _plan_for_target(target_states: int) -> dict[int, tuple[int, int]]:
    """Scale only episode budgets; lane identities/lattices remain unchanged."""
    if target_states < 1:
        raise ValueError("--target-states must be positive")
    states_per_task = ceil(target_states / len(DEFAULT_PLAN))
    ratio = states_per_task / REFERENCE_STATES_PER_TASK
    return {
        task: (lanes, max(1, ceil(reference_episodes * ratio)))
        for task, (lanes, reference_episodes) in DEFAULT_PLAN.items()
    }


def _round_robin_entries(plan: dict[int, tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """Spread every launch wave across tasks instead of filling one task first."""
    entries = []
    max_lanes = max(lanes for lanes, _ in plan.values())
    for lane in range(max_lanes):
        for task, (lanes, target) in plan.items():
            if lane < lanes:
                entries.append((task, lane, lanes, target))
    return entries


def launch(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[2]
    output = root / "outputs/state_tokenizer/v8-lanes"
    pid_root = root / "outputs/state_tokenizer/v8-lane-pids"
    output.mkdir(parents=True, exist_ok=True)
    pid_root.mkdir(parents=True, exist_ok=True)
    script = root / "scripts_v8_collect_lane.sh"
    plan = _plan_for_target(args.target_states)
    requested = sum(lanes for lanes, _ in plan.values())
    if requested > args.max_active:
        raise ValueError(f"plan needs {requested} lanes, above --max-active={args.max_active}")
    if args.max_running < 1:
        raise ValueError("--max-running must be positive")

    launched = skipped_alive = skipped_complete = pending = 0
    entries = []
    for task, lane, lanes, target in _round_robin_entries(plan):
        shard = f"lane-t{task:02d}-l{lane:02d}of{lanes:02d}"
        summary = output / f"collect-{shard}.json"
        pid_path = pid_root / f"{shard}.pid"
        if _complete(summary, task, target):
            status = "complete"
            skipped_complete += 1
        elif _alive(pid_path):
            status = "alive"
            skipped_alive += 1
        else:
            status = "pending"
            pending += 1
        entries.append({
            "task": task, "lane": lane, "lanes": lanes,
            "target_episodes": target, "shard": shard, "status": status,
            "pid_path": pid_path,
        })

    slots = max(0, args.max_running - skipped_alive)
    if not args.dry_run:
        for entry in entries:
            if entry["status"] != "pending" or slots == 0:
                continue
            process = subprocess.Popen(
                [
                    "/bin/bash", str(script), str(entry["task"]),
                    str(entry["lane"]), str(entry["lanes"]),
                    str(entry["target_episodes"]),
                ],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            entry["pid_path"].write_text(f"{process.pid}\n")
            entry["status"] = "launched"
            launched += 1
            pending -= 1
            slots -= 1

    for entry in entries:
        entry["pid_path"] = str(entry["pid_path"])
    result = {
        "protocol": "browsergym_deterministic_episode_lanes_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requested_lanes": requested,
        "target_states_requested": args.target_states,
        "states_per_task": ceil(args.target_states / len(plan)),
        "states_selected_after_assembly": ceil(args.target_states / len(plan)) * len(plan),
        "max_running": args.max_running,
        "launched": launched,
        "skipped_alive": skipped_alive,
        "skipped_complete": skipped_complete,
        "pending": pending,
        "running_after_launch": skipped_alive + launched,
        "dry_run": args.dry_run,
        "entries": entries,
    }
    manifest = root / "outputs/state_tokenizer/v8-lane-launch.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-active", type=int, default=240)
    parser.add_argument("--max-running", type=int, default=64)
    parser.add_argument("--target-states", type=int, default=REFERENCE_TARGET_STATES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.watch and args.dry_run:
        parser.error("--watch and --dry-run cannot be combined")
    while True:
        result = launch(args)
        print(
            json.dumps({key: value for key, value in result.items() if key != "entries"}),
            flush=True,
        )
        if not args.watch or result["skipped_complete"] == result["requested_lanes"]:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
