from __future__ import annotations

from experiments.state_tokenizer.v2_data import derive_v2_targets, parse_compact_dom, select_subset


def _record(index: int, split: str, task: str, episode: str, step: int, value: str = ""):
    flags = "1,1,0,1" if step else "0,0,0,1"
    dom = (
        '<ref=1 parent=0 tag=body box=0,0,1,1/>\n'
        f'<ref=2 parent=1 tag=input_checkbox box=0,0,1,1 value={"True" if step else "False"} flags={flags}/>'
        f'\n<ref=3 parent=1 tag=input_text box=0,0,1,1 value="{value}" flags={flags}/>'
    )
    return {
        "state_id": f"s{index}", "episode_id": episode, "step": step,
        "split": split, "task": task, "instruction": "Select alpha",
        "dom": dom, "probe": {"visible_words": ["alpha", "beta"]},
    }


def test_parse_and_targets_capture_local_dynamic_state():
    record = _record(0, "train", "task-a", "episode-a", 1, "state-12345")
    elements = parse_compact_dom(record["dom"])
    assert elements[1]["flags"] == [1, 1, 0, 1]
    target = derive_v2_targets(record)
    assert target["dynamic"]["checkbox_0_checked"]
    assert target["dynamic"]["textbox_0_nonempty"]
    assert target["dynamic"]["textbox_0_focused"]
    assert target["dynamic"]["interactive_tampered"]
    assert target["random_digits"] == [1, 2, 3, 4, 5]
    assert target["instruction_overlap_words"] == ["alpha"]
    assert target["dom_only_words"] == ["beta"]


def test_subset_has_exact_splits_and_unique_episodes():
    records = []
    index = 0
    for split, episodes in (("train", 8), ("validation", 6), ("test", 7)):
        for task in ("task-a", "task-b"):
            for episode in range(episodes):
                for step in (0, 1):
                    records.append(_record(
                        index, split, task, f"{split}-{task}-{episode}", step,
                        "state-12345" if step else "",
                    ))
                    index += 1
    subset = select_subset(records, {"train": 10, "validation": 6, "test": 8}, seed=7)
    assert len(subset) == 24
    assert len({record["episode_id"] for record in subset}) == 24
    assert sum(record["split"] == "train" for record in subset) == 10
    assert sum(record["split"] == "validation" for record in subset) == 6
    assert sum(record["split"] == "test" for record in subset) == 8
