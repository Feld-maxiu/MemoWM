"""Optional 5k-step predicted-prefix curriculum after a frozen base checkpoint."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.state_tokenizer.common import sha256_file, write_json

from .cache import CACHE_FILES, FrozenCache
from .config import load_config
from .model import actions_from_batch, predict
from .schema import ACTION_TYPE_IDS, REF_PAD_ID, TAG_PAD_ID
from .train import (
    DeterministicSampler,
    model_batch_keys,
    _decay_mask,
    _device_batch,
    _save_checkpoint,
    evaluate_model,
    load_checkpoint,
    make_update,
)


def _make_greedy(variant, model_config):
    def greedy(params, batch):
        actions = actions_from_batch(batch, model_config)
        mask_logits, code_logits = predict(
            params, batch["history_codes"], batch["history_valid"], actions,
            batch["task_ids"], variant, model_config,
            history_present=batch["history_present"], train=False,
        )
        return (jnp.argmax(code_logits, -1).astype(jnp.uint8), mask_logits >= 0)
    return jax.jit(greedy)


def scheduled_sample_batch(params, host, ratio, key, greedy, batch_keys):
    """Replace eligible history states by recursive greedy model predictions."""
    codes = host["history_codes"].copy()
    valid = host["history_valid"].copy()
    present = host["history_present"]
    batch_size, history = present.shape
    for target_time in range(1, history):
        eligible = present[:, target_time] & present[:, target_time - 1]
        if not np.any(eligible):
            continue
        prefix = {name: np.asarray(value).copy() for name, value in host.items()}
        prefix["history_codes"].fill(0)
        prefix["history_valid"].fill(False)
        prefix["history_present"].fill(False)
        prefix["action_types"].fill(ACTION_TYPE_IDS["PAD"])
        prefix["action_tags"].fill(TAG_PAD_ID)
        prefix["action_refs"].fill(REF_PAD_ID)
        prefix["action_payloads"].fill(0)
        prefix["action_lengths"].fill(0)
        for row in range(batch_size):
            source_positions = np.flatnonzero(present[row, :target_time])
            if not len(source_positions):
                # A legal dummy prefix for rows whose prediction is ignored.
                source_positions = np.asarray([history - 1])
            width = min(len(source_positions), history)
            source_positions = source_positions[-width:]
            destination = np.arange(history - width, history)
            prefix["history_codes"][row, destination] = codes[row, source_positions]
            prefix["history_valid"][row, destination] = valid[row, source_positions]
            prefix["history_present"][row, destination] = True
            for name in (
                "action_types", "action_tags", "action_refs",
                "action_payloads", "action_lengths",
            ):
                prefix[name][row, destination] = host[name][row, source_positions]
        predicted_codes, predicted_valid = jax.device_get(
            greedy(params, _device_batch(prefix, batch_keys))
        )
        key, draw_key = jax.random.split(key)
        replace = np.asarray(
            jax.random.bernoulli(draw_key, ratio, (batch_size,)), np.bool_
        ) & eligible
        codes[replace, target_time] = np.asarray(predicted_codes)[replace]
        valid[replace, target_time] = np.asarray(predicted_valid)[replace]
    host = dict(host)
    host["history_codes"] = codes
    host["history_valid"] = valid
    return host, key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", default="configs/world_model/v8_discrete.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--start-ratio", type=float, default=0.1)
    parser.add_argument("--end-ratio", type=float, default=0.5)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    if args.steps != 5_000 or args.start_ratio != 0.1 or args.end_ratio != 0.5:
        raise ValueError("formal curriculum is fixed at 5000 steps and ratio 0.1 -> 0.5")

    cache = FrozenCache(args.cache)
    config = load_config(args.config, num_tasks=len(cache.task_names))
    checkpoint = load_checkpoint(args.checkpoint)
    variant = checkpoint["metadata"]["variant"]
    seed = int(checkpoint["metadata"]["seed"])
    device = jax.devices(args.platform)[args.device_index]
    params = jax.device_put(checkpoint["params"], device)
    opt_state = jax.device_put(checkpoint["opt_state"], device)
    key = jax.device_put(checkpoint["key"], device)
    sampler = DeterministicSampler.from_state(checkpoint["sampler"])
    base_step = int(checkpoint["step"])
    train_rows = cache.indices_for_split("train")
    if not np.array_equal(np.sort(sampler.rows), train_rows):
        raise ValueError("curriculum checkpoint was not trained on full train transitions")

    original_schedule = optax.warmup_cosine_decay_schedule(
        0.0, config.training.learning_rate, config.training.warmup_steps,
        config.training.max_steps, 0.0,
    )
    start_lr = float(original_schedule(base_step))

    def continuation_schedule(count):
        progress = jnp.clip((count - base_step) / args.steps, 0.0, 1.0)
        return start_lr * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.training.gradient_clip),
        optax.adamw(
            continuation_schedule,
            b1=config.training.adam_beta1,
            b2=config.training.adam_beta2,
            eps=config.training.adam_epsilon,
            weight_decay=config.training.weight_decay,
            mask=_decay_mask(params),
        ),
    )
    # Optax state structure is unchanged; only the scalar schedule is replaced.
    update = make_update(optimizer, variant, config.model, overfit=False)
    greedy = _make_greedy(variant, config.model)
    batch_keys = model_batch_keys(config.model)
    validation_rows = cache.indices_for_split("validation")
    before, _ = evaluate_model(
        cache, validation_rows, params, variant, config, keep_per_transition=False
    )
    for local_step in range(args.steps):
        ratio = args.start_ratio + (
            args.end_ratio - args.start_ratio
        ) * local_step / max(args.steps - 1, 1)
        host = cache.batch(sampler.next())
        host, key = scheduled_sample_batch(
                params, host, ratio, key, greedy, batch_keys
            )
        key, update_key = jax.random.split(key)
        params, opt_state, _loss, _metrics = update(
            params, opt_state, _device_batch(host, batch_keys), update_key
        )
    after, per_transition = evaluate_model(
        cache, validation_rows, params, variant, config, keep_per_transition=True
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    candidate = output / "candidate.pkl"
    metadata = dict(checkpoint["metadata"])
    metadata["curriculum"] = {
        "steps": args.steps, "start_ratio": args.start_ratio,
        "end_ratio": args.end_ratio, "base_checkpoint_sha256": sha256_file(args.checkpoint),
    }
    _save_checkpoint(
        candidate, params=params, opt_state=opt_state, key=key, sampler=sampler,
        step=base_step + args.steps,
        best_metric=after["total_bits_per_transition"],
        best_step=base_step + args.steps, evals_without_improvement=0,
        metadata=metadata,
    )
    archive = output / "per_transition.npz"
    np.savez_compressed(archive, **per_transition)
    one_step_relative_degradation = (
        after["total_bits_per_transition"] / before["total_bits_per_transition"] - 1.0
    )
    report = {
        "protocol": "v8_wm_predicted_prefix_curriculum_v1",
        "variant": variant,
        "seed": seed,
        "base_step": base_step,
        "curriculum_steps": args.steps,
        "ratio": [args.start_ratio, args.end_ratio],
        "learning_rate": {
            "start": start_lr,
            "end": 0.0,
            "schedule": "cosine_continuation_no_rewarmup",
        },
        "teacher_forced_before": before,
        "teacher_forced_after": after,
        "one_step_relative_degradation": one_step_relative_degradation,
        "one_step_within_1pct": one_step_relative_degradation <= 0.01,
        "candidate_checkpoint": candidate.name,
        "candidate_checkpoint_sha256": sha256_file(candidate),
        "retention_decision": (
            "pending_three_seed_closed_loop_bootstrap; one-step criterion alone is insufficient"
        ),
    }
    write_json(output / "curriculum.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
