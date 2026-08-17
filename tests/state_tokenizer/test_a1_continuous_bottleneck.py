"""End-to-end A1 continuous bottleneck runner checks on miniature artifacts.

Exercises the real runner (not a stub) against temporary records, Static-PCA
shards and a normalizer, and asserts the protocol guarantees the experiment
depends on: split isolation, BF16 -> FP32 loading, best-selection parameter
choice, test-split silence by default, and strict checkpoint identity.
"""
from __future__ import annotations

import argparse
import json

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from experiments.state_tokenizer import a1_continuous_bottleneck as A1
from residualmem.encoders.normalization import GroupChannelNormalizer
from residualmem.world_model import continuous_bottleneck as C

TASKS = ("miniwob/task-a", "miniwob/task-b", "miniwob/task-c")
LAYOUT = (32, 12, 16, 4)
SPLIT_SIZES = {"train": 12, "validation": 6, "test": 6}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("a1-dataset")
    rng = np.random.default_rng(0)
    records = []
    for split, size in SPLIT_SIZES.items():
        for offset in range(size):
            index = len(records)
            records.append({
                "global_index": index,
                "split": split,
                "task": TASKS[offset % len(TASKS)],
            })
    total = len(records)

    values = rng.normal(size=(total, 64, 512)).astype(np.float32) * 0.5
    valid = np.ones((total, 64), bool)
    valid[:, 60:] = False  # a few structurally invalid prompt slots
    values = values * valid[..., None]

    worker = root / "features" / "worker00"
    worker.mkdir(parents=True)
    np.save(worker / "key64-static-pca-bf16.npy",
            values.astype(ml_dtypes.bfloat16).view(np.uint16))
    np.save(worker / "key64-static-valid.npy", valid)
    np.save(worker / "record_indices.npy", np.arange(total, dtype=np.int64))
    np.save(worker / "done.npy", np.ones((total,), bool))
    np.save(worker / "key64-static-pca-done.npy", np.ones((total,), bool))

    records_path = root / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    normalization = root / "normalization.npz"
    np.savez(
        normalization,
        mean=rng.normal(size=(4, 512)).astype(np.float32) * 0.1,
        scale=(np.abs(rng.normal(size=(4, 512))).astype(np.float32) + 0.5),
        layout=np.asarray(LAYOUT, np.int32),
        group_names=np.asarray(["image", "detail", "context", "prompt"]),
        pca_sha256=np.asarray("0" * 64),
    )
    return {
        "root": root,
        "records": records_path,
        "features": root / "features",
        "normalization": normalization,
        "values": values,
        "valid": valid,
        "records_list": records,
    }


def _args(dataset, tmp_path, **overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        records=str(dataset["records"]),
        features=str(dataset["features"]),
        normalization=str(dataset["normalization"]),
        output=str(tmp_path / "result.json"),
        label="test",
        num_e_tokens=4,
        e_dim=32,
        num_heads=2,
        ffn_hidden=32,
        overfit_states=0,
        init_checkpoint=None,
        seed=0,
        stage_steps=[4],
        stage_learning_rates=[1e-3],
        batch_size=4,
        eval_batch_size=8,
        eval_every=2,
        diagnostic_states=4,
        diagnostics_npz=None,
        weight_decay=1e-4,
        clip_norm=10.0,
        early_stop_mse=0.0,
        evaluate_test=False,
        gate_profile="report_only",
        baseline_mse=A1.A0_SLOTKEY_MSE,
        structural_mse_ratio=3.0,
        total_mse_gate=1e-4,
        total_r2_gate=0.999,
        group_mse_gate=1e-3,
        group_r2_gate=0.99,
        raw_rmse_gate=1e-2,
        compressed_r2_gate=0.90,
        compressed_group_r2_gate=0.80,
        max_r2_gap=0.10,
        platform="cpu",
        device_index=0,
        matmul_precision="highest",
        patience_evals=0,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_feature_store_reproduces_bf16_to_fp32_and_zero_padding(dataset):
    store = A1.FeatureStore(dataset["features"])
    indices = [3, 1, 20]
    x_t, valid = store.load(indices)
    expected = (
        dataset["values"][indices].astype(ml_dtypes.bfloat16).astype(np.float32)
    )
    assert np.array_equal(x_t, expected)
    assert np.array_equal(valid, dataset["valid"][indices])

    normalizer = GroupChannelNormalizer.from_npz(dataset["normalization"])
    xbar = normalizer.normalize(x_t, valid)
    assert np.isfinite(xbar).all()
    assert int(np.count_nonzero(xbar[~valid])) == 0

    with pytest.raises(KeyError):
        store.load([10_000])


def test_split_rows_are_disjoint_and_task_balanced(dataset):
    records = dataset["records_list"]
    splits = {name: A1.select_split_rows(records, name) for name in SPLIT_SIZES}
    assert [len(splits[name]) for name in SPLIT_SIZES] == list(SPLIT_SIZES.values())
    combined = [row for rows in splits.values() for row in rows]
    assert len(combined) == len(set(combined))
    for name, rows in splits.items():
        assert all(records[row]["split"] == name for row in rows)

    selected = A1.select_task_balanced_rows(records, 6, seed=0)
    assert len(set(selected)) == 6
    assert set(selected) <= set(splits["train"])
    tasks = [records[row]["task"] for row in selected]
    assert len(set(tasks)) == len(TASKS)  # round-robin covers every task
    assert selected == A1.select_task_balanced_rows(records, 6, seed=0)
    with pytest.raises(ValueError):
        A1.select_task_balanced_rows(records, len(records) + 1, seed=0)


def test_overfit_run_reports_protocol_and_improves(dataset, tmp_path):
    args = _args(dataset, tmp_path, overfit_states=6, stage_steps=[6],
                 eval_every=3, output=str(tmp_path / "overfit.json"))
    result = A1.run(args)

    assert result["protocol"] == C.PROTOCOL
    assert result["mode"] == "overfit"
    assert result["data"]["selection_split"] == "train"
    assert result["optimizer"]["steps_ran"] == 6
    assert result["capacity"]["compression_factor"] == pytest.approx(64 * 512 / (4 * 32))
    assert result["best_selection_mse"] < result["initial_selection_mse"]
    assert result["gates"]["training_improved"]
    assert result["gates"]["encoder_invalid_attention_zero"]
    assert result["gates"]["invalid_output_zero"]
    assert result["gates"]["initial_gradients_finite"]
    assert result["gates"]["passed"]  # report_only: integrity gates only
    assert result["wiring"]["encoder_value"] == "xbar_safe"
    assert result["wiring"]["decoder_key"] == "latent_index_address"
    assert (tmp_path / "overfit.npz").exists()
    assert "z_t" not in json.dumps(result)


def test_split_mode_hides_test_split_unless_requested(dataset, tmp_path):
    args = _args(dataset, tmp_path, output=str(tmp_path / "split.json"))
    result = A1.run(args)
    assert result["mode"] == "split"
    assert set(result["splits"]) == {"train", "validation"}
    assert result["data"]["selection_split"] == "validation"
    assert result["data"]["evaluate_test"] is False
    assert result["history"], "expected periodic validation history"

    with_test = A1.run(
        _args(dataset, tmp_path, evaluate_test=True, output=str(tmp_path / "with_test.json"))
    )
    assert set(with_test["splits"]) == {"train", "validation", "test"}
    assert with_test["splits"]["test"]["states"] == float(SPLIT_SIZES["test"])


def test_multi_stage_schedule_resets_optimizer_and_records_stages(dataset, tmp_path):
    args = _args(dataset, tmp_path, stage_steps=[3, 3],
                 stage_learning_rates=[1e-3, 1e-4], eval_every=3,
                 output=str(tmp_path / "stages.json"))
    result = A1.run(args)
    assert result["optimizer"]["steps_ran"] == 6
    assert result["optimizer"]["optimizer_state_reset_per_stage"] is True
    assert sorted({entry["stage"] for entry in result["history"]}) == [0, 1]


def test_checkpoint_roundtrips_and_rejects_foreign_checkpoints(dataset, tmp_path):
    args = _args(dataset, tmp_path, overfit_states=6, stage_steps=[4], eval_every=2,
                 output=str(tmp_path / "ckpt.json"))
    result = A1.run(args)
    config = C.ContinuousBottleneckConfig(
        num_e_tokens=args.num_e_tokens, e_dim=args.e_dim,
        num_heads=args.num_heads, ffn_hidden=args.ffn_hidden,
    )
    params = A1.load_checkpoint(tmp_path / "ckpt.npz", config)
    assert set(params) == set(C.expected_param_shapes(config))

    store = A1.FeatureStore(dataset["features"])
    normalizer = GroupChannelNormalizer.from_npz(dataset["normalization"])
    x_t, valid = store.load(result["data"]["train_global_indices"])
    xbar = normalizer.normalize(x_t, valid)
    metrics = C.group_metrics(
        C.reconstruct(params, xbar, valid, config), xbar, valid, config
    )
    assert metrics["all/mse"] == pytest.approx(result["splits"]["train"]["all/mse"], rel=1e-4)

    # a different capacity is a different model, never a partial load
    with pytest.raises(ValueError):
        A1.load_checkpoint(
            tmp_path / "ckpt.npz",
            C.ContinuousBottleneckConfig(
                num_e_tokens=args.num_e_tokens + 1, e_dim=args.e_dim,
                num_heads=args.num_heads, ffn_hidden=args.ffn_hidden,
            ),
        )
    # an A0 checkpoint must not initialize A1
    a0_like = tmp_path / "a0_like.npz"
    np.savez_compressed(
        a0_like,
        slot_embedding=np.zeros((64, 512), np.float32),
        metadata=np.asarray(json.dumps({"protocol": "a0_slot_reconstruction_v1"})),
    )
    with pytest.raises(ValueError):
        A1.load_checkpoint(a0_like, config)
    bare = tmp_path / "bare.npz"
    np.savez_compressed(bare, weight=np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError):
        A1.load_checkpoint(bare, config)


def test_resume_from_checkpoint_preserves_parameters(dataset, tmp_path):
    first = _args(dataset, tmp_path, overfit_states=6, stage_steps=[4], eval_every=2,
                  output=str(tmp_path / "first.json"))
    A1.run(first)
    config = C.ContinuousBottleneckConfig(
        num_e_tokens=first.num_e_tokens, e_dim=first.e_dim,
        num_heads=first.num_heads, ffn_hidden=first.ffn_hidden,
    )
    saved = A1.load_checkpoint(tmp_path / "first.npz", config)
    resumed = A1.run(
        _args(dataset, tmp_path, overfit_states=6, stage_steps=[1], eval_every=1,
              init_checkpoint=str(tmp_path / "first.npz"),
              output=str(tmp_path / "resumed.json"))
    )
    assert resumed["init_checkpoint"].endswith("first.npz")
    reloaded = A1.load_checkpoint(tmp_path / "resumed.npz", config)
    # one extra update means the parameters moved, but the graph is identical
    assert set(reloaded) == set(saved)
    assert not jnp.array_equal(reloaded["encoder/e_queries"], saved["encoder/e_queries"])


def test_numerics_protocol_is_recorded_and_defaults_to_highest(dataset, tmp_path):
    result = A1.run(
        _args(dataset, tmp_path, stage_steps=[2], eval_every=2,
              output=str(tmp_path / "numerics.json"))
    )
    numerics = result["numerics"]
    assert numerics["matmul_precision"] == "highest"
    assert numerics["activations"] == "float32"
    assert numerics["metric_accumulation"] == "float64_host"
    assert numerics["eval_batch_size"] == 8
    assert result["device"]

    # the CLI default must stay 'highest'; a throughput run has to be explicit
    parser = A1.build_parser()
    assert parser.get_default("matmul_precision") == "highest"
    assert parser.get_default("evaluate_test") is False


def test_metric_accumulation_is_batch_split_invariant(dataset, tmp_path):
    """FP64 host accumulation must make the reported MSE independent of batching."""
    whole = A1.run(
        _args(dataset, tmp_path, stage_steps=[2], eval_every=2, eval_batch_size=64,
              output=str(tmp_path / "whole.json"))
    )
    chunked = A1.run(
        _args(dataset, tmp_path, stage_steps=[2], eval_every=2, eval_batch_size=2,
              output=str(tmp_path / "chunked.json"))
    )
    for split in ("train", "validation"):
        assert whole["splits"][split]["all/mse"] == pytest.approx(
            chunked["splits"][split]["all/mse"], rel=1e-12
        )
        assert whole["splits"][split]["denormalized/rmse"] == pytest.approx(
            chunked["splits"][split]["denormalized/rmse"], rel=1e-12
        )


def test_patience_stops_early_and_records_reason(dataset, tmp_path):
    """Patience is opt-in; A1 and A2 must share it for a fair comparison."""
    default = A1.run(
        _args(dataset, tmp_path, stage_steps=[6], eval_every=2,
              output=str(tmp_path / "no_patience.json"))
    )
    assert default["optimizer"]["steps_ran"] == 6
    assert default["optimizer"]["stop_reason"] == "budget_exhausted"
    assert default["optimizer"]["patience_evals"] == 0

    # patience=1 on an already-converged toy run stops at the first non-improving eval
    patient = A1.run(
        _args(dataset, tmp_path, stage_steps=[60], eval_every=2, patience_evals=1,
              output=str(tmp_path / "patience.json"))
    )
    assert patient["optimizer"]["steps_ran"] < 60
    assert patient["optimizer"]["stopped_early"] is True
    assert patient["optimizer"]["stop_reason"] == "validation_patience"
    assert A1.build_parser().get_default("patience_evals") == 0


def test_diagnostics_npz_holds_attention_not_parameters(dataset, tmp_path):
    diagnostics = tmp_path / "diag.npz"
    args = _args(dataset, tmp_path, stage_steps=[2], eval_every=2,
                 diagnostic_states=3, diagnostics_npz=str(diagnostics),
                 output=str(tmp_path / "diag.json"))
    A1.run(args)
    with np.load(diagnostics, allow_pickle=False) as data:
        assert data["encoder_attention"].shape == (3, args.num_e_tokens, 64)
        assert data["decoder_attention"].shape == (64, args.num_e_tokens)
        assert data["global_indices"].shape == (3,)
        assert str(data["split"]) == "validation"
        assert "encoder/e_queries" not in data.files
