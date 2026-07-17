# Mastering Diverse Domains through World Models

## ResidualMem v0.3

This fork adds the paper prototype as a separate `residualmem/` package. The
original DreamerV3 agent, configs, environments, and training entry point remain
available and unchanged so that later RL integration is still possible.

Create or reproduce the environment:

```sh
conda env create -f environment.yml
conda activate ResidualMem
pip install -e .
```

The environment has already been created in this workspace. Run the deterministic
Crafter sanity pipeline (collect, train a small 2-layer GRU, exact encode,
decode, verify, and plot):

```sh
conda run -n ResidualMem python -m residualmem sanity \
  --output outputs/residualmem_sanity
```

The individual stages are also exposed:

```sh
python -m residualmem collect-crafter --steps 1000 --seed 0 \
  --output data/crafter_seed0.npz
python -m residualmem train-wm --input data/crafter_seed0.npz \
  --output checkpoints/residualmem_gru.npz
python -m residualmem encode --input data/crafter_seed0.npz \
  --checkpoint checkpoints/residualmem_gru.npz --output memory/crafter.rsm
python -m residualmem decode --input memory/crafter.rsm \
  --checkpoint checkpoints/residualmem_gru.npz --output data/reconstructed.npz
```

Use `import-emembench` for official JSONL trajectories with `action_id` and
`info.{player_pos,inventory,achievements}`. Codec outputs carry schema and
model hashes, per-segment CRCs and final-state hashes, and a complete byte
account. Every canonical field is exact by construction; there is no weighted
or rate-distortion mode in v0.3.

Retrieval is Segment-level. A SQLite event index provides primary recall over
actions, changed fields, literals, entities, and time ranges. Exactly one dense
embedding is stored per Segment as semantic fallback. The repository deliberately
does not install an embedding model or LLM: export deterministic Segment documents,
embed them externally, then import the resulting matrix.

```sh
python -m residualmem export-index-docs --input data/crafter_seed0.npz \
  --output memory/segment_docs.jsonl
# Produce memory/segment_embeddings.npy with an external embedding provider.
python -m residualmem build-index --memory memory/crafter.rsm \
  --trajectory data/crafter_seed0.npz \
  --embeddings memory/segment_embeddings.npy \
  --embedding-model-id your-model-version --output memory/crafter.rmi
python -m residualmem retrieve --memory memory/crafter.rsm \
  --index memory/crafter.rmi --plan query_plan.json --output candidates.json
```

`ReaderPolicy` is an abstract policy interface. The engine automatically shows
the first state of the top Segment using Reader-selected fields. The Reader can
then `EXPAND` to the next residual (or at most 8 steps), `REVEAL` cached states,
or `SWITCH` Segment. Answers must cite fields and steps that were actually shown.
See `ResidualMem_v0.3_exact_progressive.md` for the implementation contract.

A reimplementation of [DreamerV3][paper], a scalable and general reinforcement
learning algorithm that masters a wide range of applications with fixed
hyperparameters.

![DreamerV3 Tasks](https://user-images.githubusercontent.com/2111293/217647148-cbc522e2-61ad-4553-8e14-1ecdc8d9438b.gif)

If you find this code useful, please reference in your paper:

```
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

To learn more:

- [Research paper][paper]
- [Project website][website]
- [Twitter summary][tweet]

## DreamerV3

DreamerV3 learns a world model from experiences and uses it to train an actor
critic policy from imagined trajectories. The world model encodes sensory
inputs into categorical representations and predicts future representations and
rewards given actions.

![DreamerV3 Method Diagram](https://user-images.githubusercontent.com/2111293/217355673-4abc0ce5-1a4b-4366-a08d-64754289d659.png)

DreamerV3 masters a wide range of domains with a fixed set of hyperparameters,
outperforming specialized methods. Removing the need for tuning reduces the
amount of expert knowledge and computational resources needed to apply
reinforcement learning.

![DreamerV3 Benchmark Scores](https://github.com/danijar/dreamerv3/assets/2111293/0fe8f1cf-6970-41ea-9efc-e2e2477e7861)

Due to its robustness, DreamerV3 shows favorable scaling properties. Notably,
using larger models consistently increases not only its final performance but
also its data-efficiency. Increasing the number of gradient steps further
increases data efficiency.

![DreamerV3 Scaling Behavior](https://user-images.githubusercontent.com/2111293/217356063-0cf06b17-89f0-4d5f-85a9-b583438c98dd.png)

# Instructions

The code has been tested on Linux and Mac and requires Python 3.11+.

## Docker

You can either use the provided `Dockerfile` that contains instructions or
follow the manual instructions below.

## Manual

Install [JAX][jax] and then the other dependencies:

```sh
pip install -U -r requirements.txt
```

Training script:

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs crafter \
  --run.train_ratio 32
```

To reproduce results, train on the desired task using the corresponding config,
such as `--configs atari --task atari_pong`.

View results:

```sh
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

Scalar metrics are also writting as JSONL files.

# Tips

- All config options are listed in `dreamerv3/configs.yaml` and you can
  override them as flags from the command line.
- The `debug` config block reduces the network size, batch size, duration
  between logs, and so on for fast debugging (but does not learn a good model).
- By default, the code tries to run on GPU. You can switch to CPU or TPU using
  the `--jax.platform cpu` flag.
- You can use multiple config blocks that will override defaults in the
  order they are specified, for example `--configs crafter size50m`.
- By default, metrics are printed to the terminal, appended to a JSON lines
  file, and written as Scope summaries. Other outputs like WandB and
  TensorBoard can be enabled in the training script.
- If you get a `Too many leaves for PyTreeDef` error, it means you're
  reloading a checkpoint that is not compatible with the current config. This
  often happens when reusing an old logdir by accident.
- If you are getting CUDA errors, scroll up because the cause is often just an
  error that happened earlier, such as out of memory or incompatible JAX and
  CUDA versions. Try `--batch_size 1` to rule out an out of memory error.
- Many environments are included, some of which require installing additional
  packages. See the `Dockerfile` for reference.
- To continue stopped training runs, simply run the same command line again and
  make sure that the `--logdir` points to the same directory.

# Disclaimer

This repository contains a reimplementation of DreamerV3 based on the open
source DreamerV2 code base. It is unrelated to Google or DeepMind. The
implementation has been tested to reproduce the official results on a range of
environments.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://arxiv.org/pdf/2301.04104
[website]: https://danijar.com/dreamerv3
[tweet]: https://twitter.com/danijarh/status/1613161946223677441
