"""Stage-2 (report) training for the probabilistic RSSM world model.

Objective (report Eq. 40, minus the task/value terms that Section 8 gating would
add): ``L = recon_weight * ||x_hat - x||^2 + kl_beta * KL(q || p)`` with free bits.
Trains teacher-forced (advance with the posterior sample) then optionally
closed-loop (advance with the prior mode) to harden the prior the codec relies on.
There is deliberately **no value head / counterfactual-utility term**.

Training windows are cut at ``segment_length`` boundaries with a reset at each
window start, mirroring how the codec resets ``h=0`` at every segment anchor.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..latent.types import DomainId
from ..types import Trajectory
from . import rssm as R
from .latent_predictor import LatentPredictor


def build_config(encoder, num_groups: int, num_categories: int, **overrides) -> R.RSSMConfig:
    """Construct an RSSMConfig whose ``x_dim`` matches the encoder's token target."""
    x_dim = encoder.spec.token_dim * encoder.spec.num_tokens
    return R.RSSMConfig(
        num_groups=num_groups, num_categories=num_categories, x_dim=x_dim, **overrides)


def _segment_examples(trajectory: Trajectory, encoder, segment_length: int):
    x_all = np.stack([encoder.encode(state).flat() for state in trajectory.states])
    actions = np.asarray(trajectory.actions, np.int32)
    examples = []
    for start in range(0, len(actions), segment_length):
        stop = min(start + segment_length, len(actions))
        examples.append((
            x_all[start:stop + 1].astype(np.float32),
            actions[start:stop],
        ))
    return examples


def train_rssm(
    trajectories: list[Trajectory],
    encoder,
    domain: DomainId,
    config: R.RSSMConfig,
    output,
    *,
    seed: int = 0,
    teacher_epochs: int = 40,
    closed_loop_epochs: int = 10,
    learning_rate: float = 3e-4,
    segment_length: int = 64,
    init_params=None,
) -> dict[str, float]:
    if not trajectories:
        raise ValueError("at least one trajectory required")
    x_dim = encoder.spec.token_dim * encoder.spec.num_tokens
    if config.x_dim != x_dim:
        raise ValueError(f"config.x_dim={config.x_dim} != encoder x_dim={x_dim}")
    latent_schema = _latent_schema(config)
    domain_idx = domain.index

    examples = [ex for traj in trajectories
                for ex in _segment_examples(traj, encoder, segment_length)]
    if not examples:
        raise ValueError("training needs at least one transition")

    params = init_params if init_params is not None else R.initialize_params(config, seed)
    optimizer = optax.chain(optax.clip_by_global_norm(100.0), optax.adam(learning_rate))
    opt_state = optimizer.init(params)

    def loss_fn(p, x, actions, key, closed_loop):
        return R.rssm_loss(p, x, actions, domain_idx, config, key, closed_loop)

    grad_tf = jax.jit(jax.value_and_grad(lambda p, x, a, k: loss_fn(p, x, a, k, False)))
    grad_cl = jax.jit(jax.value_and_grad(lambda p, x, a, k: loss_fn(p, x, a, k, True)))

    base = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    counter = 0
    losses: list[float] = []
    for phase, epochs, grad in (("tf", teacher_epochs, grad_tf),
                                ("cl", closed_loop_epochs, grad_cl)):
        for _ in range(epochs):
            for index in rng.permutation(len(examples)):
                x, actions = examples[index]
                key = jax.random.fold_in(base, counter)
                counter += 1
                loss, grads = grad(params, jnp.asarray(x), jnp.asarray(actions), key)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                losses.append(float(loss))

    R.save_rssm_checkpoint(output, params, config, latent_schema, domain.name)
    metrics = evaluate_rssm(params, config, encoder, trajectories, domain,
                            segment_length=segment_length)
    metrics["final_loss"] = float(np.mean(losses[-max(1, len(examples)):]))
    Path(output).with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def evaluate_rssm(params, config: R.RSSMConfig, encoder, trajectories, domain: DomainId,
                  *, segment_length: int = 64) -> dict[str, float]:
    latent_schema = _latent_schema(config)
    domain_idx = domain.index
    predictor = LatentPredictor(params, config, latent_schema, domain)
    total_groups = matched = 0
    recon_sq = 0.0
    recon_count = 0
    for traj in trajectories:
        for x, actions in _segment_examples(traj, encoder, segment_length):
            x = jnp.asarray(x)
            actions_j = jnp.asarray(actions)
            codes, xhat = R.posterior_rollout(params, x, actions_j, domain_idx, config)
            codes = np.asarray(codes)
            recon_sq += float(np.sum((np.asarray(xhat) - np.asarray(x)) ** 2))
            recon_count += int(np.asarray(x).size)
            # prior predictive accuracy along the exact codec path
            predictor.reset()
            prev = latent_schema.make_state([int(v) for v in codes[0]])
            for step in range(1, codes.shape[0]):
                default = predictor.predict_next(prev, int(actions[step - 1])).default_state
                realized = [int(v) for v in codes[step]]
                matched += sum(int(a == b) for a, b in zip(default.values, realized))
                total_groups += config.num_groups
                prev = latent_schema.make_state(realized)
    acc = matched / max(1, total_groups)
    return {
        "prior_predictive_accuracy": acc,
        "mean_residual_groups_per_step": (1.0 - acc) * config.num_groups,
        "recon_mse": recon_sq / max(1, recon_count),
    }


def latentize_trajectory(params, config: R.RSSMConfig, encoder, trajectory: Trajectory,
                         domain: DomainId) -> Trajectory:
    """Encode a canonical trajectory to a single-sequence latent Trajectory (z^+).

    Stage-1 validation encodes this with one segment (segment_length >= T). The
    Stage-2 latent codec latentizes per segment internally instead."""
    latent_schema = _latent_schema(config)
    x = np.stack([encoder.encode(state).flat() for state in trajectory.states]).astype(np.float32)
    actions = np.asarray(trajectory.actions, np.int32)
    codes = np.asarray(
        R.latentize_codes(params, jnp.asarray(x), jnp.asarray(actions), domain.index, config))
    states = tuple(latent_schema.make_state([int(v) for v in row]) for row in codes)
    return Trajectory(states, tuple(int(a) for a in actions),
                      episode_id=trajectory.episode_id,
                      metadata={"latent_of": trajectory.episode_id, "domain": domain.name})


def _latent_schema(config: R.RSSMConfig):
    from ..latent.types import LatentSpec
    return LatentSpec(config.num_groups, config.num_categories, token_dim=1).make_schema()


def joint_finetune(
    checkpoint,
    trajectories: list[Trajectory],
    encoder,
    domain: DomainId,
    output,
    *,
    kl_beta: float,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    segment_length: int = 64,
    seed: int = 0,
) -> dict[str, float]:
    """Fixed-budget fine-tune (report Stage 6 / Eq. 45 minus the value term).

    The learning-free analytic codec has no trainable rate, so the budget dial is
    the WM's ``kl_beta``: a larger beta pulls the prior toward the posterior, which
    lowers the expected residual bits. Continues closed-loop training from a saved
    checkpoint and re-saves. There is deliberately no task/utility (value) term.
    """
    params, config, _ = R.load_rssm_checkpoint(checkpoint)
    config = dataclasses.replace(config, kl_beta=float(kl_beta))
    metrics = train_rssm(
        trajectories, encoder, domain, config, output, seed=seed,
        teacher_epochs=0, closed_loop_epochs=epochs, learning_rate=learning_rate,
        segment_length=segment_length, init_params=params)
    metrics["kl_beta"] = float(kl_beta)
    return metrics
