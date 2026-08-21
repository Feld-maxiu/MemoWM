"""End-to-end A2 (PQ-8) runner checks on miniature artifacts.

Drives the real runner: A1 warm start, per-subspace K-means, temperature
calibration, patience stopping, code health, the de-duplicated validation subset
and the strict A2 checkpoint protocol.
"""
from __future__ import annotations

import argparse
import json

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from experiments.state_tokenizer import a1_continuous_bottleneck as A1
from experiments.state_tokenizer import a2_categorical_bottleneck as A2
from residualmem.world_model import categorical_bottleneck as Q
from residualmem.world_model import continuous_bottleneck as C

TASKS = ("miniwob/task-a", "miniwob/task-b", "miniwob/task-c")
SPLIT_SIZES = {"train": 24, "validation": 8, "test": 8}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("a2-dataset")
    rng = np.random.default_rng(0)
    records = []
    for split, size in SPLIT_SIZES.items():
        for offset in range(size):
            records.append({
                "global_index": len(records),
                "split": split,
                "task": TASKS[offset % len(TASKS)],
            })
    total = len(records)

    values = rng.normal(size=(total, 64, 512)).astype(np.float32) * 0.5
    valid = np.ones((total, 64), bool)
    valid[:, 60:] = False
    values = values * valid[..., None]
    # one validation state is a near-duplicate of a train state, so the de-dup
    # subset has something to remove
    values[SPLIT_SIZES["train"]] = values[0]

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
        scale=np.abs(rng.normal(size=(4, 512))).astype(np.float32) + 0.5,
        layout=np.asarray((32, 12, 16, 4), np.int32),
        group_names=np.asarray(["image", "detail", "context", "prompt"]),
        pca_sha256=np.asarray("0" * 64),
    )
    return {"root": root, "records": records_path,
            "features": root / "features", "normalization": normalization}


def _args(dataset, tmp_path, **overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        records=str(dataset["records"]),
        features=str(dataset["features"]),
        normalization=str(dataset["normalization"]),
        output=str(tmp_path / "a2.json"),
        label="test",
        init_a1_checkpoint=None,
        init_checkpoint=None,
        init_pq_checkpoint=None,
        assignment="pq",
        a1_reference=None,
        num_e_tokens=4, e_dim=8, num_subspaces=2, num_categories=4,
        num_heads=2, ffn_hidden=16,
        temperature=1.0, calibrate_temperature=True, target_max_prob=0.8,
        calibration_states=8, kmeans_iterations=3, kmeans_problem_batch=3,
        overfit_states=0, seed=0, max_steps=4, batch_size=4,
        eval_batch_size=8, eval_every=2, patience_evals=0, init_batch_size=8,
        codebook_learning_rate=3e-4, backbone_learning_rate=3e-5,
        freeze_encoder=False, freeze_codebook=False,
        weight_decay=1e-4, clip_norm=10.0, diagnostic_states=4,
        detail_weight=0.0,
        dedup_threshold=0.01, codes_npz=None, evaluate_test=False,
        gate_profile="report_only", r2_gate=0.90, group_r2_gate=0.80,
        platform="cpu", device_index=0, matmul_precision="highest",
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _a1_args(dataset, tmp_path, **overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        records=str(dataset["records"]), features=str(dataset["features"]),
        normalization=str(dataset["normalization"]),
        output=str(tmp_path / "a1.json"), label="a1",
        num_e_tokens=4, e_dim=8, num_heads=2, ffn_hidden=16,
        overfit_states=0, init_checkpoint=None, seed=0,
        stage_steps=[2], stage_learning_rates=[1e-3],
        batch_size=4, eval_batch_size=8, eval_every=2, diagnostic_states=4,
        diagnostics_npz=None, weight_decay=1e-4, clip_norm=10.0,
        early_stop_mse=0.0, patience_evals=0, evaluate_test=False,
        gate_profile="report_only", baseline_mse=A1.A0_SLOTKEY_MSE,
        structural_mse_ratio=3.0, total_mse_gate=1e-4, total_r2_gate=0.999,
        group_mse_gate=1e-3, group_r2_gate=0.99, raw_rmse_gate=1e-2,
        compressed_r2_gate=0.90, compressed_group_r2_gate=0.80, max_r2_gap=0.10,
        platform="cpu", device_index=0, matmul_precision="highest",
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_run_reports_capacity_wiring_and_code_health(dataset, tmp_path):
    result = A2.run(_args(dataset, tmp_path, output=str(tmp_path / "base.json")))

    assert result["protocol"] == Q.PROTOCOL
    capacity = result["capacity"]
    assert capacity["codes_per_state"] == 4 * 2
    assert capacity["code_bits_per_state"] == pytest.approx(4 * 2 * 2.0)  # log2(4)
    assert capacity["mask_bits_per_state"] == 64
    assert "NOT a measured bitrate" in capacity["note"]

    wiring = result["wiring"]
    assert wiring["forward"] == "hard_gather_only"
    assert wiring["rng"] == "none"
    assert wiring["soft_path"] == "diagnostic_only"
    assert wiring["losses"] == ["masked_reconstruction_mse"]
    assert wiring["detail_weight"] == 0.0

    metrics = result["splits"]["validation"]
    assert metrics["code/bits_per_state"] == pytest.approx(capacity["code_bits_per_state"])
    assert 1.0 <= metrics["code/perplexity_median"] <= 4.0
    assert "soft/mse" in metrics and "hard_soft_ratio" in metrics
    assert result["gates"]["invalid_output_zero"]
    assert result["gates"]["codebook_gradient_nonzero"]
    assert result["gates"]["latent_gradient_nonzero"]
    assert result["gates"]["encoder_params_moved"]
    assert result["gates"]["codebook_moved"]
    assert (tmp_path / "base.npz").exists()


def test_detail_weight_changes_the_run_and_is_archived_with_it(dataset, tmp_path):
    """The loss definition has to travel with the result, not just the command."""
    plain = A2.run(_args(dataset, tmp_path, output=str(tmp_path / "plain.json")))
    weighted = A2.run(
        _args(dataset, tmp_path, detail_weight=1.0,
              output=str(tmp_path / "weighted.json"))
    )

    assert weighted["wiring"]["detail_weight"] == 1.0
    assert any("detail_group_mse_global_denominator" in name
               for name in weighted["wiring"]["losses"])

    # Same seed, same data, same K-means and temperature: initialisation is
    # untouched, so the objective is the only thing that differs.
    for key in ("kmeans", "temperature"):
        assert weighted["init"][key] == plain["init"][key]
    # ...and it really does reach the parameters, at step 0 already
    assert weighted["init"]["initial_gradient_norms"] != \
        plain["init"]["initial_gradient_norms"]
    assert weighted["splits"]["validation"]["all/mse"] != \
        plain["splits"]["validation"]["all/mse"]


def test_kmeans_and_temperature_calibration_are_recorded(dataset, tmp_path):
    result = A2.run(_args(dataset, tmp_path, output=str(tmp_path / "init.json")))
    kmeans = result["init"]["kmeans"]
    assert kmeans["clusters"] == 4
    assert kmeans["problems"] == 4 * 2
    assert kmeans["samples"] == SPLIT_SIZES["train"]
    for key in ("empty_clusters", "singleton_clusters", "occupancy_min",
                "occupancy_median", "occupancy_max"):
        assert key in kmeans

    calibration = result["init"]["temperature_calibration"]
    assert calibration["achieved_median_max_prob"] == pytest.approx(0.8, abs=0.1)
    assert result["init"]["temperature"] > 0
    assert result["config"]["temperature"] == result["init"]["temperature"]
    assert result["gates"]["temperature_calibrated"]

    fixed = A2.run(_args(dataset, tmp_path, calibrate_temperature=False,
                         temperature=2.5, output=str(tmp_path / "fixed.json")))
    assert fixed["init"]["temperature_calibration"] is None
    assert fixed["config"]["temperature"] == pytest.approx(2.5)


def test_kmeans_problem_blocking_is_numerically_equivalent():
    """Changing only the resident problem count must not change centroids."""
    points = np.random.default_rng(7).normal(size=(7, 20, 3)).astype(np.float32)
    whole, whole_stats, whole_occupancy = A2.kmeans_codebook(
        points, num_clusters=4, seed=11, iterations=3, chunk=7,
        return_occupancy=True,
    )

    blocked = []
    occupancies = []
    for start in range(0, len(points), 3):
        centers, _, occupancy = A2.kmeans_codebook(
            points[start:start + 3], num_clusters=4, seed=11, iterations=3,
            chunk=3, problem_offset=start, return_occupancy=True,
        )
        blocked.append(np.asarray(centers))
        occupancies.append(occupancy)

    np.testing.assert_array_equal(np.concatenate(blocked), np.asarray(whole))
    np.testing.assert_array_equal(np.concatenate(occupancies), whole_occupancy)
    assert whole_stats["problems"] == len(points)


def test_centroids_never_see_held_out_states(dataset, tmp_path):
    """K-means must be fitted on train only."""
    args = _args(dataset, tmp_path)
    result = A2.run(args)
    assert result["init"]["kmeans"]["samples"] == SPLIT_SIZES["train"]

    config = Q.CategoricalBottleneckConfig(
        num_e_tokens=args.num_e_tokens, e_dim=args.e_dim,
        num_heads=args.num_heads, ffn_hidden=args.ffn_hidden,
        num_subspaces=args.num_subspaces, num_categories=args.num_categories,
    )
    params = Q.initialize_params(config, 0)
    store = A2.FeatureStore(args.features)
    from residualmem.encoders.normalization import GroupChannelNormalizer
    normalizer = GroupChannelNormalizer.from_npz(args.normalization)
    records = list(A2.iter_jsonl(args.records))
    import jax
    device = jax.devices("cpu")[0]
    validation = A2.build_split(
        "validation", A2.select_split_rows(records, "validation"),
        records, store, normalizer, device,
    )
    with pytest.raises(ValueError):
        A2.fit_codebook(params, validation, config, seed=0, iterations=1, batch_size=8)


def test_dedup_subset_removes_near_duplicates(dataset, tmp_path):
    result = A2.run(_args(dataset, tmp_path, output=str(tmp_path / "dedup.json")))
    dedup = result["data"]["dedup"]
    assert dedup["removed"] >= 1  # the planted duplicate
    assert dedup["kept"] == SPLIT_SIZES["validation"] - dedup["removed"]
    assert result["splits"]["validation_dedup"]["states"] == float(dedup["kept"])
    assert result["data"]["scope"].startswith("same 12 tasks")


def test_warm_start_inherits_a1_backbone_only(dataset, tmp_path):
    A1.run(_a1_args(dataset, tmp_path, output=str(tmp_path / "a1.json")))
    result = A2.run(_args(
        dataset, tmp_path,
        init_a1_checkpoint=str(tmp_path / "a1.npz"),
        a1_reference=str(tmp_path / "a1.json"),
        output=str(tmp_path / "warm.json"),
    ))
    inherited = result["init"]["warm_started_keys"]
    assert Q.CODEBOOK not in inherited
    assert "encoder/e_queries" in inherited and "decoder/output_queries" in inherited

    baseline = result["baseline"]
    assert baseline["rho_disc"] > 0
    assert set(baseline["rho_disc_per_group"]) == set(C.GROUP_NAMES)

    config = Q.CategoricalBottleneckConfig(
        num_e_tokens=4, e_dim=8, num_heads=2, ffn_hidden=16,
        num_subspaces=2, num_categories=4,
        temperature=result["init"]["temperature"],
    )
    shared = A2.load_a1_warm_start(tmp_path / "a1.npz", config)
    a1_params = A1.load_checkpoint(tmp_path / "a1.npz", config.continuous)
    assert set(shared) == set(a1_params)
    with pytest.raises(ValueError):
        A2.load_a1_warm_start(tmp_path / "warm.npz", config)  # A2 npz is not A1


def test_checkpoint_roundtrip_replays_from_codes(dataset, tmp_path):
    args = _args(dataset, tmp_path, codes_npz=str(tmp_path / "codes.npz"),
                 output=str(tmp_path / "ckpt.json"))
    result = A2.run(args)
    config = Q.CategoricalBottleneckConfig(
        num_e_tokens=args.num_e_tokens, e_dim=args.e_dim,
        num_heads=args.num_heads, ffn_hidden=args.ffn_hidden,
        num_subspaces=args.num_subspaces, num_categories=args.num_categories,
        temperature=result["init"]["temperature"],
    )
    params = A2.load_checkpoint(tmp_path / "ckpt.npz", config)

    store = A2.FeatureStore(args.features)
    from residualmem.encoders.normalization import GroupChannelNormalizer
    normalizer = GroupChannelNormalizer.from_npz(args.normalization)
    with np.load(tmp_path / "codes.npz", allow_pickle=False) as saved:
        codes = jnp.asarray(saved["codes"])
        valid = np.asarray(saved["valid"])
        x_t, loaded_valid = store.load([int(i) for i in saved["global_indices"]])
    assert np.array_equal(valid, loaded_valid)
    xbar = normalizer.normalize(x_t, valid)

    full = Q.reconstruct(params, xbar, valid, config)
    replay = Q.decode(params, Q.embed_codes(params, codes, config), valid, config.continuous)
    assert jnp.array_equal(full, replay)

    with pytest.raises(ValueError):
        A2.load_checkpoint(tmp_path / "ckpt.npz",
                           Q.CategoricalBottleneckConfig(
                               num_e_tokens=args.num_e_tokens, e_dim=args.e_dim,
                               num_heads=args.num_heads, ffn_hidden=args.ffn_hidden,
                               num_subspaces=args.num_subspaces,
                               num_categories=args.num_categories + 1,
                               temperature=result["init"]["temperature"]))
    with pytest.raises(ValueError):
        A2.load_checkpoint(tmp_path / "a1.npz", config) if (tmp_path / "a1.npz").exists() \
            else A2.load_checkpoint(tmp_path / "ckpt.npz", config.__class__())


def test_numerics_protocol_and_batch_split_invariance(dataset, tmp_path):
    """Invariance is a property of evaluation at fixed parameters.

    It cannot be asserted across two whole training runs: best-checkpoint
    selection compares validation MSEs that differ in the last fp64 ulp, so on a
    flat curve the two runs can legitimately keep different steps.
    """
    args = _args(dataset, tmp_path, eval_batch_size=32,
                 output=str(tmp_path / "numerics.json"))
    result = A2.run(args)
    numerics = result["numerics"]
    assert numerics["matmul_precision"] == "highest"
    assert numerics["metric_accumulation"] == "float64_host"
    assert numerics["eval_batch_size"] == 32
    assert A2.build_parser().get_default("matmul_precision") == "highest"
    assert A2.build_parser().get_default("evaluate_test") is False

    import jax
    from residualmem.encoders.normalization import GroupChannelNormalizer

    config = Q.CategoricalBottleneckConfig(
        num_e_tokens=args.num_e_tokens, e_dim=args.e_dim,
        num_heads=args.num_heads, ffn_hidden=args.ffn_hidden,
        num_subspaces=args.num_subspaces, num_categories=args.num_categories,
        temperature=result["init"]["temperature"],
    )
    params = A2.load_checkpoint(tmp_path / "numerics.npz", config)
    normalizer = GroupChannelNormalizer.from_npz(args.normalization)
    records = list(A2.iter_jsonl(args.records))
    split = A2.build_split(
        "validation", A2.select_split_rows(records, "validation"), records,
        A2.FeatureStore(args.features), normalizer, jax.devices("cpu")[0],
    )
    forward, soft_forward = A2.make_forward(config)

    def evaluate(batch_size):
        return A2.evaluate_split(
            params, split, normalizer, config, batch_size=batch_size,
            diagnostic_states=4, forward=forward, soft_forward=soft_forward,
        )

    whole, chunked = evaluate(32), evaluate(2)
    for key in ("all/mse", "denormalized/rmse", "soft/mse",
                "code/perplexity_median", "code/dominant_share_max"):
        assert whole["metrics"][key] == pytest.approx(chunked["metrics"][key], rel=1e-12)
    assert np.array_equal(whole["histogram"], chunked["histogram"])


def test_test_split_stays_closed_unless_requested(dataset, tmp_path):
    default = A2.run(_args(dataset, tmp_path, output=str(tmp_path / "closed.json")))
    assert set(default["splits"]) == {"train", "validation", "validation_dedup"}
    opened = A2.run(_args(dataset, tmp_path, evaluate_test=True,
                          output=str(tmp_path / "opened.json")))
    assert opened["splits"]["test"]["states"] == float(SPLIT_SIZES["test"])


def test_resume_keeps_codebook_and_temperature(dataset, tmp_path):
    """Resuming must not refit K-means or recalibrate: both would discard training."""
    first = A2.run(_args(dataset, tmp_path, output=str(tmp_path / "first.json")))
    resumed = A2.run(_args(
        dataset, tmp_path, init_checkpoint=str(tmp_path / "first.npz"),
        max_steps=2, output=str(tmp_path / "resumed.json"),
    ))
    assert resumed["init"]["resumed_from"].endswith("first.npz")
    assert resumed["init"]["kmeans"] is None
    assert resumed["init"]["temperature_calibration"] is None
    assert resumed["init"]["temperature"] == pytest.approx(first["init"]["temperature"])
    # picking up where the first run stopped, not from a fresh codebook
    assert resumed["init"]["initial_selection_mse"] == pytest.approx(
        first["splits"]["validation"]["all/mse"], rel=1e-9
    )

    with pytest.raises(ValueError):
        A2.run(_args(dataset, tmp_path, init_checkpoint=str(tmp_path / "first.npz"),
                     num_categories=8, output=str(tmp_path / "bad.json")))
    with pytest.raises(ValueError):
        A2.run(_args(dataset, tmp_path, init_checkpoint=str(tmp_path / "first.npz"),
                     init_a1_checkpoint=str(tmp_path / "first.npz"),
                     output=str(tmp_path / "both.json")))


def test_freezing_holds_groups_fixed_and_reports_it(dataset, tmp_path):
    """With encoder and codebook frozen the codes are fixed: a pure capacity probe."""
    result = A2.run(_args(
        dataset, tmp_path, freeze_encoder=True, freeze_codebook=True,
        max_steps=6, eval_every=3, output=str(tmp_path / "frozen.json"),
    ))
    assert result["optimizer"]["freeze_encoder"] is True
    assert result["optimizer"]["freeze_codebook"] is True
    assert result["gates"]["encoder_stayed_frozen"]
    assert result["gates"]["codebook_stayed_frozen"]
    assert "encoder_params_moved" not in result["gates"]
    assert "codebook_moved" not in result["gates"]

    config = Q.CategoricalBottleneckConfig(
        num_e_tokens=4, e_dim=8, num_heads=2, ffn_hidden=16,
        num_subspaces=2, num_categories=4,
        temperature=result["init"]["temperature"],
    )
    trained = A2.load_checkpoint(tmp_path / "frozen.npz", config)
    # the decoder is the only thing that may have moved
    baseline = A2.run(_args(dataset, tmp_path, max_steps=0,
                            output=str(tmp_path / "frozen_ref.json")))
    reference = A2.load_checkpoint(tmp_path / "frozen_ref.npz", config)
    for name in reference:
        if name.startswith("encoder/") or name == Q.CODEBOOK:
            assert jnp.array_equal(trained[name], reference[name]), name
    assert baseline["optimizer"]["stop_reason"] == "evaluation_only"


def test_evaluation_only_run_trains_nothing(dataset, tmp_path):
    result = A2.run(_args(dataset, tmp_path, max_steps=0,
                          output=str(tmp_path / "eval_only.json")))
    assert result["optimizer"]["steps_ran"] == 0
    assert result["optimizer"]["stop_reason"] == "evaluation_only"
    assert result["history"] == []
    # no movement gates are asserted for a run that never stepped
    assert "training_improved" not in result["gates"]
    assert result["gates"]["gradients_finite"]
    assert result["best_selection_mse"] == pytest.approx(
        result["init"]["initial_selection_mse"]
    )


def test_quantization_geometry_is_reported(dataset, tmp_path):
    result = A2.run(_args(dataset, tmp_path, max_steps=0,
                          output=str(tmp_path / "quant.json")))
    for split in ("train", "validation"):
        metrics = result["splits"][split]
        assert 0.0 <= metrics["quant/relative_error"] <= 2.0
        assert metrics["quant/latent_rms"] > 0.0
        assert metrics["quant/codeword_rms"] > 0.0


def test_learned_assignment_starts_from_pq_equivalence(dataset, tmp_path):
    """A learned run must begin as the PQ model it was handed, then diverge."""
    pq = A2.run(_args(dataset, tmp_path, max_steps=0,
                      output=str(tmp_path / "pq_src.json")))
    learned = A2.run(_args(
        dataset, tmp_path, assignment="learned_full",
        init_pq_checkpoint=str(tmp_path / "pq_src.npz"),
        max_steps=0, output=str(tmp_path / "learned0.json"),
    ))
    assert learned["config"]["assignment"] == "learned_full"
    assert learned["init"]["pq_equivalence"]["source"].endswith("pq_src.npz")
    assert learned["init"]["kmeans"] is None
    assert learned["init"]["temperature_calibration"] is None
    # step 0 reproduces the PQ model, so the metrics must agree
    for split in ("train", "validation"):
        assert learned["splits"][split]["all/mse"] == pytest.approx(
            pq["splits"][split]["all/mse"], rel=1e-5
        )
    # the geometry-only diagnostic disappears; the logit-scale one appears
    assert "quant/relative_error" in pq["splits"]["validation"]
    assert "quant/relative_error" not in learned["splits"]["validation"]
    assert 0.0 < learned["splits"]["validation"]["code/max_prob_mean"] <= 1.0

    trained = A2.run(_args(
        dataset, tmp_path, assignment="learned_full",
        init_pq_checkpoint=str(tmp_path / "pq_src.npz"),
        max_steps=6, eval_every=3, output=str(tmp_path / "learned.json"),
    ))
    assert trained["gates"]["codebook_moved"]
    assert trained["gates"]["codebook_gradient_nonzero"]
    assert trained["capacity"]["parameters"] > pq["capacity"]["parameters"]


def test_learned_checkpoint_is_not_interchangeable_with_pq(dataset, tmp_path):
    A2.run(_args(dataset, tmp_path, max_steps=0, output=str(tmp_path / "pq.json")))
    A2.run(_args(dataset, tmp_path, assignment="learned_slice",
                 init_pq_checkpoint=str(tmp_path / "pq.npz"),
                 max_steps=0, output=str(tmp_path / "ls.json")))

    def config(mode):
        return Q.CategoricalBottleneckConfig(
            num_e_tokens=4, e_dim=8, num_heads=2, ffn_hidden=16,
            num_subspaces=2, num_categories=4, assignment=mode,
            temperature=json.loads(
                (tmp_path / ("pq.json" if mode == "pq" else "ls.json")).read_text()
            )["init"]["temperature"],
        )

    assert set(A2.load_checkpoint(tmp_path / "ls.npz", config("learned_slice"))) >= {
        Q.SELECTOR_WEIGHT, Q.SELECTOR_BIAS, Q.EMBEDDING_TABLE
    }
    with pytest.raises(ValueError):
        A2.load_checkpoint(tmp_path / "ls.npz", config("pq"))
    with pytest.raises(ValueError):
        A2.load_checkpoint(tmp_path / "pq.npz", config("learned_slice"))
    # a PQ source is required for the equivalence transform
    with pytest.raises(ValueError):
        A2.load_pq_for_equivalence(tmp_path / "ls.npz", config("learned_slice"))


def test_pre_assignment_checkpoints_still_load(dataset, tmp_path):
    """The twelve completed PQ runs predate the field and must keep loading."""
    A2.run(_args(dataset, tmp_path, max_steps=0, output=str(tmp_path / "legacy.json")))
    path = tmp_path / "legacy.npz"
    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    metadata = json.loads(str(np.asarray(payload["metadata"]).item()))
    metadata["config"].pop("assignment")
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    legacy = tmp_path / "legacy_no_field.npz"
    np.savez_compressed(legacy, **payload)

    config = Q.CategoricalBottleneckConfig(
        num_e_tokens=4, e_dim=8, num_heads=2, ffn_hidden=16,
        num_subspaces=2, num_categories=4,
        temperature=json.loads((tmp_path / "legacy.json").read_text())["init"]["temperature"],
    )
    assert Q.CODEBOOK in A2.load_checkpoint(legacy, config)
    assert A2.read_checkpoint_config(legacy)["assignment"] == "pq"


def test_patience_stops_and_records_reason(dataset, tmp_path):
    result = A2.run(_args(dataset, tmp_path, max_steps=40, eval_every=2,
                          patience_evals=1, output=str(tmp_path / "patience.json")))
    assert result["optimizer"]["stop_reason"] in ("validation_patience", "budget_exhausted")
    if result["optimizer"]["stop_reason"] == "validation_patience":
        assert result["optimizer"]["steps_ran"] < 40
    assert result["optimizer"]["patience_evals"] == 1
