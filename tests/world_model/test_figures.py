from __future__ import annotations

from experiments.world_model.figures import (
    causal_gains,
    model_vs_baselines,
    scale_and_rollout,
)


def _comparison(point, low, high):
    return {
        "point_mean": point,
        "micro": {"ci95": [low, high]},
    }


def test_three_figure_families_render_png_and_pdf(tmp_path):
    baseline = {
        "bits_per_transition": {
            "marginal": {"total_bits_per_transition": 12.0},
            "copy": {"total_bits_per_transition": 10.0},
            "source": {"total_bits_per_transition": 9.0},
        }
    }
    runs = [
        {
            "variant": "full",
            "best_selection": {"total_bits_per_transition": 8.5},
        },
        {
            "variant": "full",
            "best_selection": {"total_bits_per_transition": 8.7},
        },
    ]
    statistics = {
        "comparisons": {
            "history_gain": _comparison(0.4, 0.1, 0.7),
            "structural_action_gain_random_policy": _comparison(0.8, 0.2, 1.2),
            "payload_gain": _comparison(-0.1, -0.4, 0.2),
        }
    }
    scale_runs = [
        {
            "data": {"train_transitions": size},
            "best_selection": {"total_bits_per_transition": rate},
        }
        for size, rate in ((10_000, 10.0), (30_000, 9.0), (39_365, 8.5))
    ]
    rollout = {
        "summary_by_horizon": {
            "1": {
                "cumulative_bits": 8.5,
                "copy_cumulative_bits": 10.0,
                "source_cumulative_bits": 9.0,
            },
            "2": {
                "cumulative_bits": 18.0,
                "copy_cumulative_bits": 20.0,
                "source_cumulative_bits": 18.5,
            },
        }
    }
    model_vs_baselines(baseline, runs, tmp_path)
    causal_gains(statistics, tmp_path)
    scale_and_rollout(scale_runs, [rollout], tmp_path)
    for stem in (
        "model_vs_baselines",
        "history_action_payload_gains",
        "data_scale_and_closed_loop",
    ):
        assert (tmp_path / f"{stem}.png").stat().st_size > 0
        assert (tmp_path / f"{stem}.pdf").stat().st_size > 0
