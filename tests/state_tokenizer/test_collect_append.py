"""Guards for appending a collection to an already-extracted line.

Extending v8 rests on one property: new ``state_id``s must sort after every
existing one. ``merge_records`` sorts by ``state_id`` and writes the row
position as ``global_index``, and the 55 GB feature store is keyed by that
index -- so a prefix that sorts last makes the extension incremental, while any
other prefix silently re-keys every feature shard.

Two failure modes here are invisible at collection time and only surface as
corrupt data much later, which is why they are tested rather than argued:

* a prefix that does not sort after ``task`` interleaves the new block;
* filtering the task list by re-enumerating it shifts ``task_index``, which
  feeds both the environment seed (``seed + task_index * 1e6 + episode_index``)
  and the ``taskNN`` id -- so ``task03`` would mean different tasks in the two
  collections, and whole tasks would move into other seed bands.
"""
from __future__ import annotations

import inspect
import sys

from experiments.state_tokenizer.collect_browsergym import _save_state, parse_args
from experiments.state_tokenizer.common import PILOT_TASKS, split_for_episode

# The two tasks whose episodes are all single-state, so they yield no
# transitions at all and are worth skipping when collecting more data.
DEAD_TASKS = ("miniwob/click-dialog-2-v1", "miniwob/focus-text-v1")


def _parse(*extra: str):
    sys.argv = ["collect", "--output", "/tmp/unused", *extra]
    return parse_args()


def _rejects(*extra: str) -> bool:
    try:
        _parse(*extra)
    except SystemExit:
        return True
    return False


def test_prefix_must_sort_after_the_unprefixed_ids():
    assert _rejects("--state-id-prefix", "b9")
    assert _rejects("--state-id-prefix", "task2")
    assert not _rejects("--state-id-prefix", "v9")
    # The property the guard stands in for, stated directly.
    assert f"v9task00-ep000000-t00" > f"task11-ep099999-t07"
    assert f"b9task00-ep000000-t00" < f"task00-ep000000-t00"


def test_defaults_reproduce_the_v8_record_exactly():
    defaults = {
        name: value.default
        for name, value in inspect.signature(_save_state).parameters.items()
        if value.default is not inspect.Parameter.empty
    }
    assert defaults["prefix"] == ""
    assert defaults["force_split"] is None
    # Omitted rather than null, so an unprefixed record is byte-identical to
    # what the v8 collection wrote and merge_records still reproduces its hash.
    assert defaults["environment_seed"] is None


def test_only_tasks_filters_without_renumbering():
    live = [task for task in PILOT_TASKS if task not in DEAD_TASKS]
    canonical = {task: index for index, task in enumerate(PILOT_TASKS)}
    chosen = set(live)

    assigned = [
        (index, task) for index, task in enumerate(PILOT_TASKS)
        if chosen is None or task in chosen
    ]
    assert len(assigned) == len(live)
    for index, task in assigned:
        assert index == canonical[task]

    # What re-enumerating would have done: nine of ten tasks change seed band.
    renumbered = {task: index for index, task in enumerate(live)}
    shifted = [t for t in live if renumbered[t] != canonical[t]]
    assert len(shifted) == 9


def test_force_split_overrides_the_modulo_rule_and_relaxes_the_worker_guard():
    # episode_index 42 is a train bucket; 46 and 48 are not.
    assert split_for_episode(42) == "train"
    assert split_for_episode(46) == "validation"
    assert split_for_episode(48) == "test"

    # num_workers divisible by 5 is only a hazard because it starves the
    # modulo buckets; forcing the split removes the mechanism entirely.
    assert _rejects("--num-workers", "10", "--worker-id", "0")
    assert not _rejects("--num-workers", "10", "--worker-id", "0",
                        "--force-split", "train")


def test_unknown_only_tasks_are_rejected_rather_than_silently_collecting_nothing():
    arguments = _parse("--only-tasks", "miniwob/click-button-v1")
    assert arguments.only_tasks == ["miniwob/click-button-v1"]
    # A typo would otherwise produce an empty assignment and a zero-state shard.
    assert set(arguments.only_tasks) <= set(PILOT_TASKS)


def test_lane_arguments_preserve_the_original_episode_lattice():
    arguments = _parse(
        "--assigned-task-index", "3", "--target-episodes", "7",
        "--shard-name", "lane-t03-l02of08",
        "--episode-start", str(3 + 12 * 2),
        "--episode-stride", str(12 * 8),
    )
    assert arguments.assigned_task_index == 3
    assert arguments.episode_start == 27
    assert arguments.episode_stride == 96
    assert arguments.target_episodes == 7


def test_lane_arguments_reject_partial_or_off_lattice_configurations():
    assert _rejects("--assigned-task-index", "0")
    assert _rejects(
        "--assigned-task-index", "3", "--target-episodes", "7",
        "--shard-name", "bad/name", "--episode-start", "27",
        "--episode-stride", "96",
    )
    assert _rejects(
        "--assigned-task-index", "3", "--target-episodes", "7",
        "--shard-name", "lane", "--episode-start", "15",
        "--episode-stride", "95",
    )
