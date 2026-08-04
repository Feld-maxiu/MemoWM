from experiments.state_tokenizer.fixed_prompt import (
    OBSERVATION_PROMPT,
    build_fixed_prompt_records,
    has_random_five_digit_value,
)


def test_fixed_prompt_removes_task_instruction_without_mutating_source():
    source = [{"instruction": "Click Submit", "state_id": "s0"}]
    fixed = build_fixed_prompt_records(source)
    assert source[0]["instruction"] == "Click Submit"
    assert fixed[0]["instruction"] == OBSERVATION_PROMPT
    assert fixed[0]["instruction_protocol"] == "fixed_task_independent_observation_v1"


def test_random_value_filter_requires_exactly_five_valid_digits():
    assert has_random_five_digit_value({"v2": {"random_digits": [1, 2, 3, 4, 5]}})
    assert not has_random_five_digit_value({"v2": {"random_digits": [-1] * 5}})
    assert not has_random_five_digit_value({"v2": {"random_digits": [1, 2, 3]}})
