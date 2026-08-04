import math

from experiments.state_tokenizer.aggregate_v2 import paired_retention


def _result(dynamic, value, prevalence=0.25):
    labels = {
        f"label_{index}": {
            "task_prevalence": prevalence,
            "task_macro_average_precision": dynamic,
        }
        for index in range(2)
    }
    return {
        "test": {
            "dynamic": {
                "eligible_labels": list(labels),
                "eligible_task_macro_average_precision": dynamic,
                "per_label": labels,
            },
            "value": {"macro_position_accuracy": value},
        }
    }


def test_paired_retention_subtracts_the_correct_chance_baseline():
    oracle = [_result(0.85, 0.90) for _ in range(3)]
    candidate = [_result(0.73, 0.74) for _ in range(3)]
    dynamic = paired_retention(candidate, oracle, "dynamic")
    value = paired_retention(candidate, oracle, "value")
    assert math.isclose(dynamic["mean"], (0.73 - 0.25) / (0.85 - 0.25))
    assert math.isclose(value["mean"], (0.74 - 0.10) / (0.90 - 0.10))
