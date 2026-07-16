from __future__ import annotations

import argparse
import json
from pathlib import Path

from residualmem.adapters.crafter import collect_crafter, load_emembench
from residualmem.codec.segment import decode_memory, encode_memory
from residualmem.data import load_trajectory, save_trajectory
from residualmem.evaluation.runner import plot_results, run_codec_suite
from residualmem.schemas import crafter_schema
from residualmem.world_model import (
    GRUConfig,
    GRUPredictor,
    IgnoreActionPredictor,
    PersistencePredictor,
    load_gru_checkpoint,
)
from residualmem.world_model.train import evaluate_gru, train_gru


def _json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _schema(profile: str):
    return crafter_schema(profile)


def _predictor(checkpoint: str | None, schema):
    if checkpoint is None:
        return PersistencePredictor()
    return load_gru_checkpoint(checkpoint, schema, allow_policy_rebind=True)


def _load_for_schema(path: str | Path, schema):
    try:
        return load_trajectory(path, schema)
    except ValueError as exc:
        if schema.schema_id.endswith("-exact"):
            raise
        exact = crafter_schema("exact")
        trajectory = load_trajectory(path, exact)
        for state in trajectory.states:
            schema.validate(state)
        return type(trajectory)(
            trajectory.states,
            trajectory.actions,
            trajectory.episode_id,
            {**trajectory.metadata, "schema_profile": schema.schema_id},
        )


def cmd_collect(args) -> None:
    schema = _schema(args.profile)
    trajectory = collect_crafter(schema, args.seed, args.steps, args.policy)
    save_trajectory(args.output, trajectory, schema)
    print(f"saved {len(trajectory.actions)} transitions to {args.output}")


def cmd_import(args) -> None:
    schema = _schema(args.profile)
    trajectory = load_emembench(args.input, schema)
    save_trajectory(args.output, trajectory, schema)
    print(f"saved {len(trajectory.actions)} transitions to {args.output}")


def cmd_train(args) -> None:
    schema = _schema(args.profile)
    trajectories = [_load_for_schema(path, schema) for path in args.input]
    metrics = train_gru(
        trajectories,
        schema,
        args.output,
        config=GRUConfig(
            hidden_size=args.hidden_size,
            action_values=args.action_values,
        ),
        seed=args.seed,
        epochs=args.epochs,
        closed_loop_epochs=args.closed_loop_epochs,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def cmd_evaluate(args) -> None:
    schema = _schema(args.profile)
    trajectories = [_load_for_schema(path, schema) for path in args.input]
    predictor = load_gru_checkpoint(
        args.checkpoint, schema, allow_policy_rebind=True
    )
    metrics = evaluate_gru(predictor, trajectories, schema)
    _json(args.output, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def cmd_encode(args) -> None:
    schema = _schema(args.profile)
    trajectory = _load_for_schema(args.input, schema)
    predictor = _predictor(args.checkpoint, schema)
    accounting = encode_memory(
        path=args.output,
        trajectory=trajectory,
        predictor=predictor,
        schema=schema,
        lambda_=args.lambda_rd,
        segment_length=args.segment_length,
    )
    decoded, _ = decode_memory(args.output, predictor, schema)
    if args.profile == "exact" and decoded.states != trajectory.states:
        raise AssertionError("exact mode failed to reconstruct the input trajectory")
    print(json.dumps(accounting.as_dict(), indent=2, sort_keys=True))


def cmd_decode(args) -> None:
    schema = _schema(args.profile)
    predictor = _predictor(args.checkpoint, schema)
    trajectory, accounting = decode_memory(args.input, predictor, schema)
    save_trajectory(args.output, trajectory, schema)
    print(json.dumps(accounting.as_dict(), indent=2, sort_keys=True))


def cmd_sanity(args) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    exact = crafter_schema("exact")
    trajectory = collect_crafter(exact, args.seed, args.steps)
    trajectory_path = output / "crafter_sanity.npz"
    save_trajectory(trajectory_path, trajectory, exact)

    checkpoint = output / "gru_sanity.npz"
    train_metrics = train_gru(
        [trajectory],
        exact,
        checkpoint,
        config=GRUConfig(hidden_size=args.hidden_size),
        seed=args.seed,
        epochs=args.epochs,
        closed_loop_epochs=args.closed_loop_epochs,
    )
    gru = load_gru_checkpoint(checkpoint, exact)
    exact_rows = run_codec_suite(
        trajectory,
        exact,
        [
            {"name": "persistence_exact", "predictor": PersistencePredictor()},
            {"name": "gru_exact", "predictor": gru},
            {
                "name": "gru_no_action_exact",
                "predictor": IgnoreActionPredictor(gru),
            },
        ],
        output / "exact",
        segment_length=args.segment_length,
    )
    plot_results(exact_rows, output / "exact")

    rd = crafter_schema("rd_uniform")
    rd_trajectory = type(trajectory)(
        trajectory.states, trajectory.actions, trajectory.episode_id,
        {**trajectory.metadata, "schema_profile": "rd_uniform"}
    )
    rd_gru = GRUPredictor(gru.params, rd, gru.config)
    rd_rows = run_codec_suite(
        rd_trajectory,
        rd,
        [
            {"name": "gru_lambda_0", "predictor": rd_gru, "lambda": 0.0},
            {"name": "gru_lambda_0.01", "predictor": rd_gru, "lambda": 0.01},
            {"name": "gru_lambda_0.1", "predictor": rd_gru, "lambda": 0.1},
            {"name": "wm_only", "predictor": rd_gru, "lambda": 1e9},
        ],
        output / "rd",
        segment_length=args.segment_length,
    )
    plot_results(rd_rows, output / "rd")
    summary = {
        "seed": args.seed,
        "steps": len(trajectory.actions),
        "checkpoint": str(checkpoint),
        "train": train_metrics,
        "exact": exact_rows,
        "rd": rd_rows,
    }
    _json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="residualmem",
        description="ResidualMem v0.2 reference implementation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect-crafter")
    collect.add_argument("--output", required=True)
    collect.add_argument("--steps", type=int, default=1000)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--policy", default="random", choices=["random"])
    collect.add_argument("--profile", default="exact", choices=["exact"])
    collect.set_defaults(func=cmd_collect)

    imported = sub.add_parser("import-emembench")
    imported.add_argument("--input", required=True)
    imported.add_argument("--output", required=True)
    imported.add_argument("--profile", default="exact", choices=["exact"])
    imported.set_defaults(func=cmd_import)

    train = sub.add_parser("train-wm")
    train.add_argument("--input", nargs="+", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--profile", default="exact", choices=["exact"])
    train.add_argument("--hidden-size", type=int, default=512)
    train.add_argument("--action-values", type=int, default=17)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--closed-loop-epochs", type=int, default=10)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate-wm")
    evaluate.add_argument("--input", nargs="+", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument(
        "--profile", default="exact", choices=["exact", "rd_uniform", "rd_task"]
    )
    evaluate.set_defaults(func=cmd_evaluate)

    encode = sub.add_parser("encode")
    encode.add_argument("--input", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--checkpoint")
    encode.add_argument(
        "--profile", default="exact", choices=["exact", "rd_uniform", "rd_task"]
    )
    encode.add_argument("--lambda-rd", type=float, default=0.0)
    encode.add_argument("--segment-length", type=int, default=64)
    encode.set_defaults(func=cmd_encode)

    decode = sub.add_parser("decode")
    decode.add_argument("--input", required=True)
    decode.add_argument("--output", required=True)
    decode.add_argument("--checkpoint")
    decode.add_argument(
        "--profile", default="exact", choices=["exact", "rd_uniform", "rd_task"]
    )
    decode.set_defaults(func=cmd_decode)

    sanity = sub.add_parser("sanity")
    sanity.add_argument("--output", default="outputs/residualmem_sanity")
    sanity.add_argument("--steps", type=int, default=64)
    sanity.add_argument("--seed", type=int, default=0)
    sanity.add_argument("--hidden-size", type=int, default=32)
    sanity.add_argument("--epochs", type=int, default=3)
    sanity.add_argument("--closed-loop-epochs", type=int, default=1)
    sanity.add_argument("--segment-length", type=int, default=32)
    sanity.set_defaults(func=cmd_sanity)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
