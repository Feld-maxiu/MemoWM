from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from residualmem.adapters.crafter import collect_crafter, load_emembench
from residualmem.codec import MemoryFile, decode_memory, encode_memory
from residualmem.data import load_trajectory, save_trajectory
from residualmem.evaluation.runner import plot_results, run_codec_suite
from residualmem.query import (
    Answer,
    EvidenceRef,
    Expand,
    Open,
    QueryEngine,
    QueryPlan,
    Reveal,
    ScriptedReaderPolicy,
    Switch,
)
from residualmem.retrieval import (
    MemoryIndex,
    build_memory_index,
    export_index_documents,
    extract_segment_events,
)
from residualmem.schemas import crafter_schema
from residualmem.world_model import (
    GRUConfig,
    IgnoreActionPredictor,
    PersistencePredictor,
    load_gru_checkpoint,
)
from residualmem.world_model.train import evaluate_gru, train_gru

from residualmem.encoders.structured import StructuredStateEncoder
from residualmem.latent.types import DomainId, LatentSpec
from residualmem.world_model.rssm import load_rssm_checkpoint
from residualmem.world_model.rssm_train import build_config, latentize_trajectory, train_rssm
from residualmem.codec import (
    ExactMaskPolicy,
    LatentMemoryFile,
    RateDistortionMaskPolicy,
    decode_latent_memory,
    encode_latent_memory,
)
from residualmem.evaluation.runner import plot_rate_distortion, run_latent_rate_distortion


def _json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _predictor(checkpoint: str | None, schema):
    if checkpoint is None:
        return PersistencePredictor()
    return load_gru_checkpoint(checkpoint, schema)


def cmd_collect(args) -> None:
    schema = crafter_schema()
    trajectory = collect_crafter(schema, args.seed, args.steps, args.policy)
    save_trajectory(args.output, trajectory, schema)
    print(f"saved {len(trajectory.actions)} transitions to {args.output}")


def cmd_import(args) -> None:
    schema = crafter_schema()
    trajectory = load_emembench(args.input, schema)
    save_trajectory(args.output, trajectory, schema)
    print(f"saved {len(trajectory.actions)} transitions to {args.output}")


def cmd_train(args) -> None:
    schema = crafter_schema()
    trajectories = [load_trajectory(path, schema) for path in args.input]
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
        learning_rate=args.learning_rate,
        segment_length=args.segment_length,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def cmd_evaluate(args) -> None:
    schema = crafter_schema()
    trajectories = [load_trajectory(path, schema) for path in args.input]
    predictor = load_gru_checkpoint(args.checkpoint, schema)
    metrics = evaluate_gru(predictor, trajectories, schema)
    _json(args.output, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def cmd_encode(args) -> None:
    schema = crafter_schema()
    trajectory = load_trajectory(args.input, schema)
    predictor = _predictor(args.checkpoint, schema)
    accounting = encode_memory(
        path=args.output,
        trajectory=trajectory,
        predictor=predictor,
        schema=schema,
        segment_length=args.segment_length,
    )
    decoded, _ = decode_memory(args.output, predictor, schema)
    if decoded.states != trajectory.states or decoded.actions != trajectory.actions:
        raise AssertionError("v0.3 exact stream failed to reconstruct its trajectory")
    print(json.dumps(accounting.as_dict(), indent=2, sort_keys=True))


def cmd_decode(args) -> None:
    schema = crafter_schema()
    predictor = _predictor(args.checkpoint, schema)
    trajectory, accounting = decode_memory(args.input, predictor, schema)
    save_trajectory(args.output, trajectory, schema)
    print(json.dumps(accounting.as_dict(), indent=2, sort_keys=True))


def cmd_export_index_docs(args) -> None:
    schema = crafter_schema()
    trajectory = load_trajectory(args.input, schema)
    events = extract_segment_events(trajectory, schema, args.segment_length)
    export_index_documents(events, args.output)
    print(f"saved {len(events)} Segment documents to {args.output}")


def cmd_build_index(args) -> None:
    schema = crafter_schema()
    trajectory = load_trajectory(args.trajectory, schema)
    events = extract_segment_events(trajectory, schema, args.segment_length)
    embeddings = np.load(args.embeddings, allow_pickle=False)
    build_memory_index(
        args.output,
        args.memory,
        events,
        schema,
        embeddings,
        args.embedding_model_id,
    )
    print(f"saved {len(events)} indexed Segments to {args.output}")


def cmd_retrieve(args) -> None:
    schema = crafter_schema()
    plan = _plan_from_dict(_read_json(args.plan))
    query_embedding = _optional_vector(args.query_embedding)
    index = MemoryIndex(args.index, schema, args.memory)
    try:
        candidates = index.retrieve(
            plan,
            query_embedding,
            args.embedding_model_id,
            args.top_k,
        )
        result = [dataclasses.asdict(candidate) for candidate in candidates]
    finally:
        index.close()
    _json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


def cmd_replay_query(args) -> None:
    schema = crafter_schema()
    script = _read_json(args.script)
    plan = _plan_from_dict(script["plan"])
    actions = [_reader_action(value) for value in script["actions"]]
    reader = ScriptedReaderPolicy(plan, actions)
    predictor = _predictor(args.checkpoint, schema)
    memory = MemoryFile(args.memory, predictor, schema)
    index = MemoryIndex(args.index, schema, args.memory)
    try:
        result = QueryEngine(
            memory,
            index,
            schema,
            top_k=args.top_k,
            max_tool_calls=args.max_tool_calls,
            default_expand_steps=args.expand_steps,
        ).run(
            script.get("query", plan.query),
            reader,
            _optional_vector(args.query_embedding),
            args.embedding_model_id,
        )
    finally:
        index.close()
    _json(args.output, result.as_dict())
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False, sort_keys=True))


def cmd_sanity(args) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    schema = crafter_schema()
    trajectory = collect_crafter(schema, args.seed, args.steps)
    trajectory_path = output / "crafter_sanity.npz"
    save_trajectory(trajectory_path, trajectory, schema)
    checkpoint = output / "gru_sanity.npz"
    train_metrics = train_gru(
        [trajectory],
        schema,
        checkpoint,
        config=GRUConfig(hidden_size=args.hidden_size),
        seed=args.seed,
        epochs=args.epochs,
        segment_length=args.segment_length,
    )
    gru = load_gru_checkpoint(checkpoint, schema)
    rows = run_codec_suite(
        trajectory,
        schema,
        [
            {"name": "persistence_exact", "predictor": PersistencePredictor()},
            {"name": "gru_exact", "predictor": gru},
            {"name": "gru_no_action_exact", "predictor": IgnoreActionPredictor(gru)},
        ],
        output / "exact",
        segment_length=args.segment_length,
    )
    plot_results(rows, output / "exact")
    summary = {
        "seed": args.seed,
        "steps": len(trajectory.actions),
        "checkpoint": str(checkpoint),
        "train": train_metrics,
        "exact": rows,
    }
    _json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _optional_vector(path: str | None):
    if path is None:
        return None
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)


def _plan_from_dict(value: dict) -> QueryPlan:
    tuple_fields = (
        "actions",
        "changed_fields",
        "literals",
        "entities",
        "initial_fields",
    )
    normalized = dict(value)
    for name in tuple_fields:
        normalized[name] = tuple(normalized.get(name, ()))
    return QueryPlan(**normalized)


def _reader_action(value: dict):
    kind = value["type"].upper()
    values = {key: item for key, item in value.items() if key != "type"}
    if "fields" in values:
        values["fields"] = tuple(values["fields"])
    if kind == "ANSWER":
        values["evidence"] = tuple(
            EvidenceRef(
                item["segment_id"],
                item["start_step"],
                item["end_step"],
                tuple(item["fields"]),
            )
            for item in values.get("evidence", ())
        )
    constructors = {
        "OPEN": Open,
        "EXPAND": Expand,
        "REVEAL": Reveal,
        "SWITCH": Switch,
        "ANSWER": Answer,
    }
    if kind not in constructors:
        raise ValueError(f"unknown Reader action: {kind}")
    return constructors[kind](**values)


def _latent_setup(checkpoint: str):
    """Load an RSSM checkpoint and rebuild (params, config, domain, latent_schema, encoder)."""
    params, config, metadata = load_rssm_checkpoint(checkpoint)
    domain = DomainId(metadata["domain"])
    latent_schema = LatentSpec(config.num_groups, config.num_categories, token_dim=1).make_schema()
    encoder = StructuredStateEncoder(
        crafter_schema(), LatentSpec(config.num_groups, config.num_categories, 1), domain)
    return params, config, domain, latent_schema, encoder


def _mask_policy(args):
    if getattr(args, "mask", "exact") == "exact":
        return ExactMaskPolicy()
    return RateDistortionMaskPolicy(lam=args.lam)


def _synthetic_trajectory(schema, seed: int, steps: int):
    rng = np.random.default_rng(seed)
    inv = [0] * 16
    ach = [0] * 22
    x, y = 10, 10
    states, actions = [], []
    for _ in range(steps):
        states.append(schema.make_state([x, y] + inv + [bool(a) for a in ach]))
        actions.append(int(rng.integers(0, 17)))
        x = min(63, max(0, x + int(rng.integers(-1, 2))))
        y = min(63, max(0, y + int(rng.integers(-1, 2))))
        if rng.random() < 0.12:
            inv[int(rng.integers(0, 16))] = min(9, inv[int(rng.integers(0, 16))] + 1)
        if rng.random() < 0.05:
            ach[int(rng.integers(0, 22))] = 1
    states.append(schema.make_state([x, y] + inv + [bool(a) for a in ach]))
    from residualmem.types import Trajectory
    return Trajectory(tuple(states), tuple(actions))


def cmd_train_rssm(args) -> None:
    schema = crafter_schema()
    domain = DomainId(args.domain)
    encoder = StructuredStateEncoder(schema, LatentSpec(args.groups, args.categories, 1), domain)
    trajectories = [load_trajectory(path, schema) for path in args.input]
    config = build_config(
        encoder, num_groups=args.groups, num_categories=args.categories,
        hidden_size=args.hidden_size)
    metrics = train_rssm(
        trajectories, encoder, domain, config, args.output, seed=args.seed,
        teacher_epochs=args.teacher_epochs, closed_loop_epochs=args.closed_loop_epochs,
        segment_length=args.segment_length)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def cmd_encode_latent(args) -> None:
    params, config, domain, latent_schema, encoder = _latent_setup(args.checkpoint)
    trajectory = load_trajectory(args.input, crafter_schema())
    accounting, _ = encode_latent_memory(
        args.output, trajectory, params, config, encoder, domain, latent_schema,
        mask_policy=_mask_policy(args), segment_length=args.segment_length)
    print(json.dumps(accounting.as_dict(), indent=2, sort_keys=True))


def cmd_decode_latent(args) -> None:
    params, config, domain, latent_schema, _ = _latent_setup(args.checkpoint)
    trajectory, accounting = decode_latent_memory(args.input, params, config, domain, latent_schema)
    save_trajectory(args.output, trajectory, latent_schema)
    print(json.dumps(accounting.as_dict(), indent=2, sort_keys=True))


def cmd_eval_rate_distortion(args) -> None:
    params, config, domain, latent_schema, encoder = _latent_setup(args.checkpoint)
    trajectory = load_trajectory(args.input, crafter_schema())
    rows = run_latent_rate_distortion(
        trajectory, params, config, encoder, domain, latent_schema, args.output,
        lambdas=tuple(args.lambdas), segment_length=args.segment_length)
    plot_rate_distortion(rows, args.output)
    print(json.dumps(rows, indent=2, sort_keys=True))


def cmd_latent_sanity(args) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    schema = crafter_schema()
    domain = DomainId("crafter")
    trajectory = _synthetic_trajectory(schema, args.seed, args.steps)
    save_trajectory(output / "synthetic.npz", trajectory, schema)
    encoder = StructuredStateEncoder(schema, LatentSpec(args.groups, args.categories, 1), domain)
    config = build_config(encoder, num_groups=args.groups, num_categories=args.categories,
                          hidden_size=args.hidden_size, embed_size=args.hidden_size,
                          post_hidden=args.hidden_size, dec_hidden=args.hidden_size)
    ckpt = output / "rssm.npz"
    train_metrics = train_rssm(
        [trajectory], encoder, domain, config, ckpt, seed=args.seed,
        teacher_epochs=args.epochs, closed_loop_epochs=max(1, args.epochs // 4),
        segment_length=args.segment_length)
    params, config2, _ = load_rssm_checkpoint(ckpt)
    latent_schema = LatentSpec(args.groups, args.categories, 1).make_schema()
    rows = run_latent_rate_distortion(
        trajectory, params, config2, encoder, domain, latent_schema, output,
        lambdas=(1.0, 2.0, 4.0, 8.0), segment_length=args.segment_length)
    plot_rate_distortion(rows, output)
    summary = {"train": train_metrics, "rate_distortion": rows}
    _json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="residualmem",
        description="ResidualMem: v0.3 exact + v0.4 latent (technical-report) memory",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect-crafter")
    collect.add_argument("--output", required=True)
    collect.add_argument("--steps", type=int, default=1000)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--policy", default="random", choices=["random"])
    collect.set_defaults(func=cmd_collect)

    imported = sub.add_parser("import-emembench")
    imported.add_argument("--input", required=True)
    imported.add_argument("--output", required=True)
    imported.set_defaults(func=cmd_import)

    train = sub.add_parser("train-wm")
    train.add_argument("--input", nargs="+", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--hidden-size", type=int, default=512)
    train.add_argument("--action-values", type=int, default=17)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--segment-length", type=int, default=64)
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate-wm")
    evaluate.add_argument("--input", nargs="+", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(func=cmd_evaluate)

    encode = sub.add_parser("encode")
    encode.add_argument("--input", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--checkpoint")
    encode.add_argument("--segment-length", type=int, default=64)
    encode.set_defaults(func=cmd_encode)

    decode = sub.add_parser("decode")
    decode.add_argument("--input", required=True)
    decode.add_argument("--output", required=True)
    decode.add_argument("--checkpoint")
    decode.set_defaults(func=cmd_decode)

    docs = sub.add_parser("export-index-docs")
    docs.add_argument("--input", required=True, help="canonical trajectory")
    docs.add_argument("--output", required=True)
    docs.add_argument("--segment-length", type=int, default=64)
    docs.set_defaults(func=cmd_export_index_docs)

    index = sub.add_parser("build-index")
    index.add_argument("--memory", required=True)
    index.add_argument("--trajectory", required=True)
    index.add_argument("--embeddings", required=True, help="precomputed .npy matrix")
    index.add_argument("--embedding-model-id", required=True)
    index.add_argument("--output", required=True)
    index.add_argument("--segment-length", type=int, default=64)
    index.set_defaults(func=cmd_build_index)

    retrieve = sub.add_parser("retrieve")
    retrieve.add_argument("--index", required=True)
    retrieve.add_argument("--memory", required=True)
    retrieve.add_argument("--plan", required=True)
    retrieve.add_argument("--output", required=True)
    retrieve.add_argument("--query-embedding")
    retrieve.add_argument("--embedding-model-id")
    retrieve.add_argument("--top-k", type=int, default=8)
    retrieve.set_defaults(func=cmd_retrieve)

    replay = sub.add_parser("replay-query")
    replay.add_argument("--memory", required=True)
    replay.add_argument("--index", required=True)
    replay.add_argument("--script", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--checkpoint")
    replay.add_argument("--query-embedding")
    replay.add_argument("--embedding-model-id")
    replay.add_argument("--top-k", type=int, default=8)
    replay.add_argument("--max-tool-calls", type=int, default=32)
    replay.add_argument("--expand-steps", type=int, default=8)
    replay.set_defaults(func=cmd_replay_query)

    sanity = sub.add_parser("sanity")
    sanity.add_argument("--output", default="outputs/residualmem_sanity")
    sanity.add_argument("--steps", type=int, default=64)
    sanity.add_argument("--seed", type=int, default=0)
    sanity.add_argument("--hidden-size", type=int, default=32)
    sanity.add_argument("--epochs", type=int, default=3)
    sanity.add_argument("--segment-length", type=int, default=32)
    sanity.set_defaults(func=cmd_sanity)

    # --- v0.4 latent (technical-report) commands ---
    train_rssm_p = sub.add_parser("train-rssm")
    train_rssm_p.add_argument("--input", nargs="+", required=True)
    train_rssm_p.add_argument("--output", required=True)
    train_rssm_p.add_argument("--domain", default="crafter")
    train_rssm_p.add_argument("--groups", type=int, default=16)
    train_rssm_p.add_argument("--categories", type=int, default=16)
    train_rssm_p.add_argument("--hidden-size", type=int, default=256)
    train_rssm_p.add_argument("--teacher-epochs", type=int, default=40)
    train_rssm_p.add_argument("--closed-loop-epochs", type=int, default=10)
    train_rssm_p.add_argument("--segment-length", type=int, default=64)
    train_rssm_p.add_argument("--seed", type=int, default=0)
    train_rssm_p.set_defaults(func=cmd_train_rssm)

    enc_l = sub.add_parser("encode-latent")
    enc_l.add_argument("--input", required=True)
    enc_l.add_argument("--output", required=True)
    enc_l.add_argument("--checkpoint", required=True)
    enc_l.add_argument("--mask", default="exact", choices=["exact", "rd"])
    enc_l.add_argument("--lam", type=float, default=4.0)
    enc_l.add_argument("--segment-length", type=int, default=64)
    enc_l.set_defaults(func=cmd_encode_latent)

    dec_l = sub.add_parser("decode-latent")
    dec_l.add_argument("--input", required=True)
    dec_l.add_argument("--output", required=True)
    dec_l.add_argument("--checkpoint", required=True)
    dec_l.set_defaults(func=cmd_decode_latent)

    eval_rd = sub.add_parser("eval-rate-distortion")
    eval_rd.add_argument("--input", required=True)
    eval_rd.add_argument("--checkpoint", required=True)
    eval_rd.add_argument("--output", required=True)
    eval_rd.add_argument("--lambdas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    eval_rd.add_argument("--segment-length", type=int, default=64)
    eval_rd.set_defaults(func=cmd_eval_rate_distortion)

    latent_sanity = sub.add_parser("latent-sanity")
    latent_sanity.add_argument("--output", default="outputs/residualmem_latent_sanity")
    latent_sanity.add_argument("--steps", type=int, default=128)
    latent_sanity.add_argument("--seed", type=int, default=0)
    latent_sanity.add_argument("--groups", type=int, default=8)
    latent_sanity.add_argument("--categories", type=int, default=16)
    latent_sanity.add_argument("--hidden-size", type=int, default=64)
    latent_sanity.add_argument("--epochs", type=int, default=8)
    latent_sanity.add_argument("--segment-length", type=int, default=32)
    latent_sanity.set_defaults(func=cmd_latent_sanity)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
