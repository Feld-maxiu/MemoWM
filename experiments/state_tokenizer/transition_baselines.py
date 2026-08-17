"""Compatibility entry point for the frozen-cache v8 transition baselines.

The old implementation encoded states on every invocation, fitted and evaluated
on the same split, hard-coded the obsolete ``(32,12,16,4)`` layout, and sliced
distributed A2 latent tokens as if they were observation slots. The authoritative
implementation now lives in :mod:`experiments.world_model.baselines`.
"""

from experiments.world_model.baselines import main


if __name__ == "__main__":
    main()
