from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .qformer_pq import encode, load_states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--states", action="append", required=True,
                        help="glob over encoded xbar npz files; repeatable")
    parser.add_argument("--label", required=True,
                        help="split name to write the codes under")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = np.load(args.artifact, allow_pickle=True)
    mean, scale = np.asarray(artifact["mean"]), np.asarray(artifact["scale"])
    centroids = np.asarray(artifact["centroids"])
    bases = np.asarray(artifact["rotation_bases"]) if "rotation_bases" in artifact else None
    order = np.asarray(artifact["rotation_order"]) if "rotation_order" in artifact else None

    states, ids, _checkpoint = load_states(args.states)
    if states.shape[1:] != mean.shape:
        raise SystemExit(
            f"states are {states.shape[1:]} but the artifact standardises {mean.shape}"
        )
    codes = encode(states, mean, scale, centroids, bases=bases, order=order)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **{f"codes/{args.label}": codes,
           f"state_ids/{args.label}": np.asarray(ids, dtype=object)},
    )
    print(json.dumps({
        "artifact": str(args.artifact.resolve()),
        "label": args.label,
        "states": int(len(ids)),
        "codes": list(codes.shape),
        "rotation": "shared" if bases is not None else "none",
    }, indent=2))


if __name__ == "__main__":
    main()
