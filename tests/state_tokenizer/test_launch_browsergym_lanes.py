from experiments.state_tokenizer.launch_browsergym_lanes import (
    _plan_for_target,
    _round_robin_entries,
)


def test_50k_plan_keeps_lane_lattices_and_halves_episode_budgets():
    plan = _plan_for_target(50_000)
    assert sum(lanes for lanes, _ in plan.values()) == 228
    assert plan == {
        0: (24, 126), 1: (32, 131), 2: (12, 73), 3: (12, 74),
        4: (12, 72), 5: (24, 118), 6: (32, 131), 7: (16, 115),
        8: (16, 151), 9: (24, 117), 10: (12, 117), 11: (12, 108),
    }
    entries = _round_robin_entries(plan)
    assert [task for task, lane, _, _ in entries[:12]] == list(range(12))
    assert [lane for _, lane, _, _ in entries[:24]] == [0] * 12 + [1] * 12
