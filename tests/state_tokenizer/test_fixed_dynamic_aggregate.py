import math

from experiments.state_tokenizer.aggregate_fixed_dynamic import LEAKAGE_BASELINES, mean_std


def test_fixed_dynamic_aggregate_ignores_nan_diagnostics():
    result = mean_std([0.4, float("nan"), 0.6])
    assert result["finite"] == 2
    assert math.isclose(result["mean"], 0.5)


def test_contextualized_fixed_prompt_tokens_are_not_a_leakage_baseline():
    assert "instruction_only" not in LEAKAGE_BASELINES
    assert set(LEAKAGE_BASELINES) == {"task_only", "task_step"}
