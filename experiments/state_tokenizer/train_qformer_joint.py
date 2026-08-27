"""Train the Q-Former tokenizer and the reader connector together.

The objective is the same two terms ``train_reader_qa`` uses -- answer CE plus
distillation KL against the frozen generator reading the observation text -- so
the only variable between the two runs is where the latent comes from. That is
the comparison Stage C measures.

**No anchor losses.** Report Eq. (9a) proposes a mixed frozen target
(semantic / OCR / visual / state subspaces) so the latent cannot drift. Three
reasons it is not here:

* The frozen observation prompt already asks for all four -- "preserve visible
  text, input values, control types, selected/focused/enabled state and spatial
  relations". In the pooled pipeline that request reaches nothing: the prompt is
  last in the sequence, causal attention hides it from the image and DOM
  positions, and its own band is ``PROMPT_SLOTS = 0``. Cross-attention has no
  such barrier, so the prompt starts working the moment the resampler does.
* Report 5.4's collapse scenario is a tokenizer trained *with the world model*,
  which is rewarded for making residuals small. Both terms here reward keeping
  information instead.
* A text-reconstruction anchor is precisely the objective that made the first
  connector answer and then narrate the observation.

What replaces them is a monitor, not a loss: every eval reports pairwise cosine
and effective rank against the *fixed-pooling* xbar of the same observations. If
the learned states are less distinguishable than the pooling they replace, the
run has failed and the specific anchor for the failing metric goes back in.

Micro-batch is 1 with gradient accumulation rather than a padded batch. The
student's memory is 64 soft tokens and the teacher's is variable-length text, so
a real batch means aligning an answer span across two different paddings -- the
same class of bug as the KL reduction this file's loss was just fixed for. The
accumulated version reuses the single-sample forward unchanged and costs ~7%.

``L_sem`` is the exception, and has to be. Its first version was
``1 - cos(project(xbar), teacher)`` computed inside the micro-batch, which at
batch 1 has no negative to contrast against -- an InfoNCE term written there
would have been *identically* zero, since cross-entropy of a single logit
against label 0 is zero. Cosine alone only asks the state to point at its
teacher, and teachers point at each other too: measured, that projection scored
0.8199 against its own teacher and 0.6491 against everyone else's, a gap of
0.17, and reached R@1 of 0.045. So the term now runs as its own step with its
own batch drawn from ONE session, which is the discrimination WorldMemArena
actually asks for. The QA path keeps micro-batch 1 and keeps its
observation-uniform sampler, so an improvement is attributable to the new loss
rather than to session-correlated QA gradients.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from residualmem.latent.instruct_bridge import (
    OBSERVATION_TEACHER_PROTOCOL,
    InputSoftTokenConnector,
    MaskedAttentionRetrievalHead,
    load_bridge,
    save_bridge,
)
from residualmem.latent.qformer import (
    QFORMER_PROTOCOL,
    QFormerStateReader,
    StateQFormer,
    qformer_hash,
)

from .extract_qwen import _load_model
from .observation_kl_precheck import PROBES
from .reader_losses import answer_ce_and_distill_kl, observation_distill_kl
from .train_retrieval_bridge import symmetric_infonce
from .trunk_states import NUM_MODALITIES, collate, trunk_states

PAIRS_PROTOCOL = "wma_qformer_qa_v1"


class ObservationStore:
    """``(sample_id, index)`` -> the inputs the trunk saw, plus the pooled xbar."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._records: dict[str, list[dict]] = {}
        self._pooled: dict[str, np.ndarray] = {}

    def _load(self, sample_id: str) -> list[dict]:
        if sample_id not in self._records:
            path = self._directory / f"{sample_id}.npz"
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(np.asarray(data["metadata"])))
            self._records[sample_id] = metadata.get("records") or []
        return self._records[sample_id]

    def observation(self, sample_id: str, index: int) -> dict:
        record = self._load(sample_id)[index]
        if not record.get("synthetic_axtree"):
            raise ValueError(
                f"{sample_id}[{index}] has no synthetic_axtree; re-run the extraction"
            )
        return record

    def pooled_xbar(self, sample_id: str, index: int) -> np.ndarray:
        """The fixed pooling's own output for this observation -- the baseline."""
        key = f"{sample_id}/{index}"
        if key not in self._pooled:
            with np.load(self._directory / f"{sample_id}.npz", allow_pickle=False) as data:
                self._pooled[key] = np.asarray(data[f"m11/xbar/{index:04d}"], np.float32)
        return self._pooled[key]


class TeacherStore:
    """``(sample_id, index)`` -> the fused-observation embedding, for L_sem.

    Rows are positionally aligned with the extraction's records, the same
    contract ``wma_build_bridge_cache`` relies on. ``image_ids`` travels with
    the teacher, so the alignment is checked rather than assumed -- a silent
    off-by-one here would train the tokenizer to match the wrong observation.
    """

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._cache: dict[str, tuple[np.ndarray, list]] = {}

    def get(self, sample_id: str, index: int, image_ids: list[str]) -> np.ndarray:
        if sample_id not in self._cache:
            with np.load(self._directory / f"{sample_id}.npz", allow_pickle=False) as data:
                protocol = str(np.asarray(data["protocol"]))
                if protocol != "qwen3_vl_fused_observation_v1":
                    raise ValueError(f"{sample_id}: teacher protocol {protocol!r}")
                self._cache[sample_id] = (
                    np.asarray(data["teacher"], np.float32),
                    json.loads(str(np.asarray(data["image_ids"]))) if
                    np.asarray(data["image_ids"]).dtype.kind in "US" and
                    np.asarray(data["image_ids"]).ndim == 0
                    else np.asarray(data["image_ids"]).tolist(),
                )
        teacher, ids = self._cache[sample_id]
        if index >= len(teacher):
            raise ValueError(f"{sample_id}[{index}]: only {len(teacher)} teacher rows")
        expected = [str(x) for x in (ids[index] if index < len(ids) else [])]
        if image_ids and expected and sorted(expected) != sorted(str(x) for x in image_ids):
            raise ValueError(
                f"{sample_id}[{index}] teacher covers {expected} but the observation "
                f"covers {image_ids} -- the two extractions are not aligned"
            )
        return teacher[index]


class ObservationTeacherStore:
    """The precomputed teacher that read the screenshot, keyed like the pairs file.

    Two lookups, one cache. ``continuation`` serves ``L_obs`` -- the teacher's
    own greedy output under a task-independent probe, so that term touches no
    benchmark label at all. ``answer_topk`` serves ``L_q``, replacing a teacher
    whose entire input (the caption) was already embedded verbatim in the
    student's DOM span.

    Held on CPU and moved per use: the whole corpus is 1.6 GB and only a few
    spans are live at a time.
    """

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def _load(self, sample_id: str) -> dict[str, np.ndarray]:
        if sample_id not in self._cache:
            path = self._directory / f"{sample_id}.npz"
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(np.asarray(data["metadata"])))
                if metadata.get("protocol") != OBSERVATION_TEACHER_PROTOCOL:
                    raise ValueError(f"{sample_id}: teacher protocol {metadata.get('protocol')!r}")
                self._cache[sample_id] = {name: np.asarray(data[name]) for name in data.files
                                          if name != "metadata"}
        return self._cache[sample_id]

    def continuation(self, sample_id: str, index: int, probe: str):
        arrays = self._load(sample_id)
        key = f"obs/{index:04d}/{probe}"
        if f"{key}/ids" not in arrays:
            raise ValueError(f"{sample_id}[{index}] has no cached probe {probe!r}")
        return (
            torch.as_tensor(arrays[f"{key}/ids"], dtype=torch.long)[None],
            torch.as_tensor(arrays[f"{key}/index"], dtype=torch.long)[None],
            torch.as_tensor(arrays[f"{key}/logprob"], dtype=torch.float32)[None],
        )

    def answer_topk(self, sample_id: str, row: int):
        """``(index, logprob)`` for one QA pair.

        Keyed by ``sample_id`` rather than searched across loaded samples: the
        row index is global to the pairs file, so scanning whatever happens to
        be cached would find it only by luck and return None the rest of the
        time -- a silent fallback to the caption teacher on most steps.
        """
        arrays = self._load(sample_id)
        key = f"qa/{row:05d}/index"
        if key not in arrays:
            raise ValueError(
                f"{sample_id} has no cached teacher for pairs row {row}; the "
                "cache and the pairs file were built from different inputs"
            )
        return (torch.as_tensor(arrays[key], dtype=torch.long)[None],
                torch.as_tensor(arrays[f"qa/{row:05d}/logprob"], dtype=torch.float32)[None])


def spread(states: np.ndarray) -> dict[str, float]:
    """Report 5.4's monitors: how distinguishable are these states from each other.

    Every quantity is computed **after removing the mean across observations**.
    Without that, the cosine measures a shared offset rather than collapse and
    the two representations are not comparable: the pooled xbar is near
    zero-mean by construction (frozen group/channel normalization) while a
    learned resampler's output is not. Measured at step 400 the raw cosine read
    1.0000 for the Q-Former against 0.5739 for the pooling, which looks like
    total collapse; centred, the same states read 0.0389 against -0.0225, i.e.
    close to orthogonal in both. ``mean_to_deviation`` is what the raw cosine
    was actually reporting, kept as its own number -- 251.5 against 1.20 says
    the informative part is 0.4% of the learned state's magnitude, which is a
    real pathology but a different one from collapse.
    """
    flat = states.reshape(len(states), -1).astype(np.float64)
    centred = flat - flat.mean(0, keepdims=True)
    unit = centred / np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-12)
    gram = unit @ unit.T
    upper = gram[np.triu_indices(len(unit), 1)]
    singular = np.linalg.svdvals(centred)
    share = singular / max(singular.sum(), 1e-12)
    share = share[share > 0]
    deviation = np.linalg.norm(centred, axis=1).mean()
    return {
        "pairwise_cosine": float(upper.mean()),
        "effective_rank": float(np.exp(-(share * np.log(share)).sum())),
        "mean_to_deviation": float(np.linalg.norm(flat.mean(0)) / max(deviation, 1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="build_qformer_qa_pairs output")
    parser.add_argument("--xbar-dir", required=True,
                        help="the extraction the pairs reference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--qformer-hidden", type=int, default=1024)
    parser.add_argument("--qformer-heads", type=int, default=8)
    parser.add_argument("--qformer-layers", type=int, default=4)
    parser.add_argument("--self-attention", action="store_true",
                        help="ADD a latent self-attention sublayer (off by default). It lets "
                             "queries divide work but is also a mixing operator "
                             "that pulls them together; the JAX resample() this "
                             "is ported from has none, and enabling it collapsed the "
                             "queries to rank 4 at identical CE")
    parser.add_argument("--teacher-dir",
                        help="fused-observation teacher embeddings, enabling L_sem")
    parser.add_argument("--sem-weight", type=float, default=0.0,
                        help="weight on the same-session contrastive term. The "
                             "anchor against representation collapse: the answer "
                             "losses need very little information, so nothing "
                             "else stops the states from becoming interchangeable")
    parser.add_argument("--teacher-cache",
                        help="build_observation_teacher output. Required by "
                             "--obs-weight and by --distill-teacher observation")
    parser.add_argument("--obs-weight", type=float, default=0.0,
                        help="weight on the observation-distillation term: make "
                             "the latent reproduce what the raw screenshot would "
                             "have produced under a task-independent probe. "
                             "Default 0 reproduces every run to date")
    parser.add_argument("--distill-teacher", choices=("fused_text", "observation"),
                        default="fused_text",
                        help="what L_q's teacher reads. 'fused_text' is the "
                             "caption, which is embedded verbatim in the "
                             "student's own DOM span -- the teacher's whole "
                             "input is a subset of the student's, so the term "
                             "teaches reproducing text it already holds. "
                             "'observation' is the screenshot. Default keeps "
                             "the old behaviour so this change is measurable")
    parser.add_argument("--trunk-fp32", action="store_true",
                        help="load the frozen trunk in fp32. Doubles memory; "
                             "exists only to test whether the non-finite "
                             "gradients are bf16 numerics in the backward")
    parser.add_argument("--min-lr-fraction", type=float, default=0.01,
                        help="floor for the learning rate decayed on each "
                             "non-finite gradient, as a fraction of the initial "
                             "rate. Without a floor a bad stretch decays the run "
                             "into a no-op that still burns GPU hours")
    parser.add_argument("--lr-recover-steps", type=int, default=0,
                        help="consecutive clean steps that restore the learning "
                             "rate by 2x, capped at the scheduled rate. This is "
                             "the half of GradScaler's logic the decay above was "
                             "missing -- it backs off on every skip and grows "
                             "again once the run is behaving, so one bad stretch "
                             "cannot hold the rate down for the rest of training. "
                             "0 keeps the decay-only behaviour of every run to "
                             "date. Measured on the w=1.0 arm this fires for no "
                             "value above 46: its 286 skips arrive a median 5 "
                             "steps apart and never once leave a 50-step gap, so "
                             "on that failure it is a no-op and the input sweep "
                             "below is the diagnostic that applies")
    parser.add_argument("--init-from",
                        help="load Q-Former + connector (+ head) weights before "
                             "training or sweeping. Needed to interrogate a "
                             "checkpoint saved mid-failure rather than only "
                             "reproduce the failure from scratch")
    parser.add_argument("--sweep-nonfinite", type=int, default=0,
                        help="skip training; instead push N training rows through "
                             "the exact loss path one at a time, on frozen "
                             "weights, and report which produce a non-finite "
                             "gradient. 0 disables. Every previous hunt needed the "
                             "failure to happen live and lost the state when the "
                             "run died; --init-from a bad checkpoint makes it "
                             "static and repeatable")
    parser.add_argument("--drop-microbatches", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="test each micro-batch's gradient on its own and "
                             "drop only the offending one, instead of discarding "
                             "the whole step. On the w=1.0 arm 2.1% of rows were "
                             "pathological and they cost 13% of steps; dropping "
                             "per micro-batch turns that into 2.1% of samples. "
                             "--no-drop-microbatches restores the old behaviour "
                             "for a like-for-like comparison")
    parser.add_argument("--microbatch-max-norm", type=float, default=1e3,
                        help="gradient norm above which a micro-batch is "
                             "dropped. Not redundant with a finiteness test: two "
                             "measured offenders were a *finite* 1.85e19 and "
                             "1.9e15, and a finite 1e19 survives clipping to "
                             "consume the entire step budget in its own "
                             "direction. Clean micro-batches measured 1.9 to "
                             "20.8 and the offenders 1e15 and above, so the "
                             "threshold sits in a thirteen-decade empty gap "
                             "rather than on a distribution's shoulder")
    parser.add_argument("--dissect-rows", type=int, nargs="*", default=None,
                        help="with --sweep-nonfinite, take these specific rows "
                             "apart instead of sweeping: CE alone against CE+KL, "
                             "the latent and teacher magnitudes, and the "
                             "parameter carrying the largest gradient")
    parser.add_argument("--sweep-repeats", type=int, default=3,
                        help="times each offending row is re-run during the "
                             "sweep. Identical verdicts every time means the "
                             "trigger is the input; verdicts that vary on "
                             "unchanged weights and inputs mean it is the kernels")
    parser.add_argument("--locate-nonfinite", action="store_true",
                        help="sample the gradient norm between the three "
                             "backward calls so a non-finite total can be "
                             "attributed to a term instead of guessed at")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="linear warmup on the learning rate. 0 reproduces "
                             "every run to date, which had no scheduler at all")
    parser.add_argument("--max-skipped-steps", type=int, default=50,
                        help="tolerated non-finite gradients before giving up. "
                             "A handful is a bad batch; dozens means the loss "
                             "or the data is wrong and skipping hides it")
    parser.add_argument("--probe-observations", type=int, default=24,
                        help="observations scored for the held-out probe gap. "
                             "Far fewer than --validation-observations on "
                             "purpose: the gap is a difference of two means and "
                             "converges quickly, while each one costs a vision "
                             "forward plus two scoring passes")
    parser.add_argument("--held-out-probe", default="P4", choices=sorted(PROBES),
                        help="never sampled for training; its KL is the eval "
                             "number that separates a general substitute from a "
                             "memorised continuation")
    parser.add_argument("--sem-mode", choices=("same-session", "cosine"),
                        default="same-session",
                        help="'cosine' reproduces the original term exactly: "
                             "1 - cos(project(xbar), teacher) inside each QA "
                             "micro-batch, where batch 1 leaves no negative. It "
                             "is kept only so the same-session variant has a "
                             "control run from this same script -- reverting the "
                             "file to get one would vary more than the objective")
    parser.add_argument("--sem-batch", type=int, default=4,
                        help="observations drawn from ONE session per step for "
                             "L_sem. This batch is sampled independently of the "
                             "QA path so the two changes stay separable")
    parser.add_argument("--sem-extra-negatives", type=int, default=-1,
                        help="extra same-session teachers used as negatives on "
                             "top of the batch. They are free -- the teachers "
                             "are frozen and already cached, only the students "
                             "cost a resampler forward -- so the default -1 "
                             "means 'all of them'. 0 reproduces the run that "
                             "tied the two counts together and had 3 negatives")
    parser.add_argument("--sem-temperature", type=float, default=0.05,
                        help="matches train_retrieval_bridge, so the jointly "
                             "trained projection and the separately trained head "
                             "optimize the same objective")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--accumulate", type=int, default=8,
                        help="micro-batches of 1 per optimizer step")
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-observations", type=int, default=96,
                        help="distinct observations scored per eval; every "
                             "question of each is averaged inside it first")
    parser.add_argument("--patience-evals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()

    pairs = np.load(args.pairs, allow_pickle=False)
    metadata = json.loads(str(np.asarray(pairs["metadata"])))
    if metadata.get("protocol") != PAIRS_PROTOCOL:
        raise ValueError(f"pairs protocol {metadata.get('protocol')!r}")
    if metadata.get("excluded") != "web":
        raise ValueError("the evaluation subcategory must be excluded from the pairs")

    device = torch.device(args.device)
    processor, model = _load_model(
        args.model, device, False,
        dtype=torch.float32 if args.trunk_fp32 else torch.bfloat16,
    )
    # Seed before construction; both modules draw from the global generator.
    torch.manual_seed(args.seed)
    joint = QFormerStateReader(
        StateQFormer(
            num_queries=args.queries, hidden=args.qformer_hidden,
            heads=args.qformer_heads, layers=args.qformer_layers,
            modalities=NUM_MODALITIES, self_attention=args.self_attention,
        ),
        InputSoftTokenConnector(slots=args.queries),
        MaskedAttentionRetrievalHead() if args.sem_weight > 0 else None,
    ).to(device)
    if args.init_from:
        loaded = load_bridge(args.init_from, joint, expected_protocol=QFORMER_PROTOCOL)
        print(f"[qformer] loaded {args.init_from}: "
              f"step {loaded.metadata.get('best_step')}, "
              f"val CE {loaded.metadata.get('validation_answer_ce')}, "
              f"gap {loaded.metadata.get('held_out_probe_gap')}, "
              f"selected_by {loaded.metadata.get('selected_by')!r}", flush=True)
    optimizer = torch.optim.AdamW(joint.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    store = ObservationStore(args.xbar_dir)
    if args.sem_weight > 0 and not args.teacher_dir:
        raise ValueError("--sem-weight needs --teacher-dir")
    teachers = TeacherStore(args.teacher_dir) if args.teacher_dir else None
    needs_cache = args.obs_weight > 0 or args.distill_teacher == "observation"
    if needs_cache and not args.teacher_cache:
        raise ValueError("--obs-weight / --distill-teacher observation need --teacher-cache")
    teachers_obs = ObservationTeacherStore(args.teacher_cache) if needs_cache else None
    train_probes = [n for n in sorted(PROBES) if n != args.held_out_probe]
    if args.obs_weight > 0 and not train_probes:
        raise ValueError("every probe is held out, so L_obs has nothing to train on")

    split = pairs["split"].astype(str)
    # Group by observation before sampling. The encoder is observation-level, so
    # drawing QA pairs uniformly weights an observation by how many questions it
    # happens to carry: 695 observations carry one question and 1,432 carry the
    # four-question cap, so 45% of observations would consume 63% of the steps.
    # That optimizes the question distribution rather than the observation
    # distribution -- and a state tokenizer that fits the question distribution
    # is exactly the shortcut the collapse monitor is watching for.
    #
    # The split is already by sample_id (strictly coarser than by observation),
    # so every question of an observation lands on the same side and validation
    # measures generalization to unseen states rather than unseen questions.
    def grouped(rows):
        table = collections.defaultdict(list)
        for row in rows:
            key = (str(pairs["sample_id"][row]), int(pairs["record_index"][row]))
            table[key].append(int(row))
        return [np.asarray(value) for value in table.values()]

    train_groups = grouped(np.flatnonzero(split == "train"))
    val_groups = grouped(np.flatnonzero(split == "validation"))

    # L_sem's own index: session -> its train observations. Only sessions with
    # two or more qualify, since a batch of one has no negative and the InfoNCE
    # term would be identically zero -- which is exactly the defect this
    # replaces. The QA sampler above is untouched.
    by_session: dict[str, set] = collections.defaultdict(set)
    for row in np.flatnonzero(split == "train"):
        by_session[str(pairs["sample_id"][row])].add(int(pairs["record_index"][row]))
    sem_sessions = [
        (name, np.asarray(sorted(indices)))
        for name, indices in sorted(by_session.items()) if len(indices) >= 2
    ]
    if args.sem_weight > 0 and not sem_sessions:
        raise ValueError("no training session has two observations, so L_sem has no negatives")
    picker = np.random.default_rng(args.seed + 1)
    chosen_val = (
        val_groups if len(val_groups) <= args.validation_observations
        else [val_groups[i] for i in picker.choice(
            len(val_groups), args.validation_observations, replace=False)]
    )
    trainable = sum(p.numel() for p in joint.parameters() if p.requires_grad)
    print(f"[qformer] {len(train_groups)} train / {len(val_groups)} validation observations "
          f"({int((split == 'train').sum())}/{int((split == 'validation').sum())} pairs), "
          f"{len(chosen_val)} scored per eval", flush=True)
    print(f"[qformer] {trainable/1e6:.1f}M trainable, {args.queries} queries, "
          f"accumulate {args.accumulate}, distill weight {args.distill_weight}", flush=True)
    if args.sem_weight > 0 and args.sem_mode == "same-session":
        sizes = np.asarray([len(indices) for _, indices in sem_sessions])
        pool = (sizes if args.sem_extra_negatives < 0
                else np.minimum(sizes, args.sem_batch + args.sem_extra_negatives))
        print(f"[qformer] L_sem weight {args.sem_weight}, batch {args.sem_batch} drawn from "
              f"one of {len(sem_sessions)} sessions ({sizes.min()}-{sizes.max()} observations "
              f"each, {int((sizes >= args.sem_batch).sum())} can fill the batch), "
              f"temperature {args.sem_temperature}, "
              f"{pool.mean() - 1:.1f} negatives on average "
              f"(extra_negatives={args.sem_extra_negatives})", flush=True)
    elif args.sem_weight > 0:
        print(f"[qformer] L_sem weight {args.sem_weight}, mode cosine: one teacher per QA "
              f"micro-batch, no negatives (the control)", flush=True)

    def latents_for(row: int):
        sample_id = str(pairs["sample_id"][row])
        index = int(pairs["record_index"][row])
        record = store.observation(sample_id, index)
        with Image.open(record["screenshot"]) as handle:
            image = handle.convert("RGB")
        states = trunk_states(
            processor, model, image, record["synthetic_axtree"],
            layer=args.layer, device=device,
        )
        # The trunk is frozen and its output carries no graph, so the two terms
        # that need this observation can share one 9B vision forward. Not doing
        # so doubled the cost of every micro-batch with L_obs on -- the same
        # saving sem_step already makes, and for the same reason.
        trunk_cache[(sample_id, index)] = states
        soft, xbar, valid = joint(*collate([states]))
        return soft, xbar, valid, record, sample_id, index

    trunk_cache: dict[tuple[str, int], object] = {}

    head_norm: dict[str, float] = {}
    sem_forward: dict[str, object] = {}
    if joint.retrieval_head is not None:
        def _watch(_module, _inputs, output):
            # Both ends, and finiteness. Recording only the minimum norm was a
            # blind spot: an infinite row makes norm(dim=-1) infinite for that
            # row and .min() then reports the healthy ones, so a head that had
            # already produced inf read as "denominator 12.01, fine".
            norms = output.detach().norm(dim=-1)
            finite = bool(torch.isfinite(norms).all())
            value = float(norms[torch.isfinite(norms)].min()) if finite else 0.0
            head_norm["min"] = min(head_norm.get("min", value), value)
            head_norm["max"] = max(head_norm.get("max", 0.0), float(norms[torch.isfinite(norms)].max()) if finite else float("inf"))
            head_norm["finite"] = head_norm.get("finite", True) and finite
        joint.retrieval_head.projection.register_forward_hook(_watch)

    def grad_norm_now() -> float:
        """Total gradient norm accumulated so far, without touching the grads.

        The guard below only learns that the *sum* went non-finite, which after
        five wrong diagnoses is not enough. Sampling between the three backward
        calls says which term produced it -- ``CE+KL_q``, ``L_obs`` or
        ``L_sem`` -- and the last of those runs the retrieval head, whose
        ``F.normalize`` has no eps and whose pre-normalisation norm was measured
        at 2.71 for a trained K=32 against 4.13 for the K=16 that never
        diverged. A norm that dips toward zero there is an unbounded gradient
        nobody has bounded.
        """
        total = 0.0
        for parameter in joint.parameters():
            if parameter.grad is not None:
                value = float(parameter.grad.detach().norm())
                if not math.isfinite(value):
                    return value
                total += value * value
        return math.sqrt(total)

    def loss_for(row: int, weight: float, sem_weight: float = 0.0):
        soft, xbar, valid, record, sample_id, index = latents_for(row)
        loss, ce, kl = answer_ce_and_distill_kl(
            model, processor, soft, valid,
            str(pairs["question"][row]), str(pairs["answer"][row]),
            str(record.get("fused_text", "")),
            max_answer_tokens=args.max_answer_tokens, weight=weight,
            teacher_topk=(teachers_obs.answer_topk(sample_id, row)
                          if args.distill_teacher == "observation" and weight > 0
                          else None),
        )
        sem = 0.0
        if sem_weight > 0:
            # --sem-mode cosine only. Pool the slots to one vector first, then
            # compare; asking each slot to match the 4096-d target separately is
            # what homogenizes them. There is no negative here -- that is the
            # defect this mode exists to be the control for.
            target = torch.as_tensor(
                teachers.get(sample_id, index, record.get("image_ids") or []),
                dtype=torch.float32, device=device,
            )[None]
            projected = joint.project(xbar, valid)
            sem_loss = (1.0 - F.cosine_similarity(projected, F.normalize(target, dim=-1))).mean()
            loss = loss + sem_weight * sem_loss
            sem = float(sem_loss)
        return loss, ce, kl, sem, xbar

    def sem_step(sem_weight: float) -> float:
        """One same-session contrastive step, sampled independently of the QA path.

        Two things are deliberate here. The batch is drawn from a *single*
        session, because the negatives that matter are the ones WorldMemArena
        retrieval faces -- round 3 against round 7 of one session, not one
        website against another. And it is drawn independently of the QA
        sampler, which stays observation-uniform: bundling "QA microbatches
        become session-correlated" into the same run would make an improvement
        unattributable between the new loss and the new sampling.

        The QA path still backpropagates one observation at a time, so the
        answer-span alignment that forced micro-batch 1 is untouched. This runs
        its own forward: ``soft`` feeds a backward *through the frozen 9B*, and
        retaining that graph across the accumulation loop would hold B copies of
        it. Re-running the resampler costs 79M instead, two orders of magnitude
        less, and the trunk states it consumes are ``no_grad`` either way.
        """
        session, indices = sem_sessions[int(rng.integers(len(sem_sessions)))]
        chosen = rng.choice(indices, size=min(args.sem_batch, len(indices)), replace=False)
        states, targets = [], []
        for index in chosen:
            record = store.observation(session, int(index))
            with Image.open(record["screenshot"]) as handle:
                image = handle.convert("RGB")
            states.append(trunk_states(
                processor, model, image, record["synthetic_axtree"],
                layer=args.layer, device=device,
            ))
            targets.append(teachers.get(session, int(index), record.get("image_ids") or []))
        # Every *other* observation of this session is a negative that costs
        # nothing: the students must go through the resampler, but the teachers
        # are frozen, precomputed, and already resident (TeacherStore caches the
        # whole sample's array on first touch). Tying the two counts together at
        # sem_batch left the term with three negatives when ~24 more were
        # sitting in memory, and they are the negatives the benchmark poses --
        # other rounds of the same session.
        extra = [int(i) for i in indices if int(i) not in set(int(c) for c in chosen)]
        if args.sem_extra_negatives >= 0:
            extra = extra[: args.sem_extra_negatives] if args.sem_extra_negatives else []
        extra_targets = [teachers.get(session, i, []) for i in extra]
        xbar, valid = joint.encode(*collate(states))
        student = joint.project(xbar, valid)
        target = F.normalize(torch.as_tensor(
            np.stack(targets), dtype=torch.float32, device=device), dim=-1)
        # Strictly a superset of what train_retrieval_bridge does: the
        # student->teacher direction gets the extra same-session teachers as
        # additional negatives, while the teacher->student direction and the
        # cosine stay on the square block, unchanged. Keeping the square block
        # intact is what makes the two objectives comparable rather than merely
        # similar; only teachers with a student in this batch can name one.
        loss = symmetric_infonce(student, target, args.sem_temperature)
        if extra_targets:
            negatives = F.normalize(torch.as_tensor(
                np.stack(extra_targets), dtype=torch.float32, device=device), dim=-1)
            logits = student @ torch.cat([target, negatives]).T / args.sem_temperature
            labels = torch.arange(len(student), device=device)
            forward = F.cross_entropy(logits, labels)
            backward = F.cross_entropy(
                (student @ target.T / args.sem_temperature).T, labels
            )
            loss = 0.5 * (forward + backward)
        loss = loss + (1.0 - (student * target).sum(-1)).mean()
        if args.locate_nonfinite:
            # The forward values at the moment of failure. If these are all
            # finite and the gradient is not, the fault is in the backward
            # arithmetic rather than in anything this function computed.
            report = {
                "xbar": bool(torch.isfinite(xbar).all()),
                "student": bool(torch.isfinite(student).all()),
                "target": bool(torch.isfinite(target).all()),
                "loss": bool(torch.isfinite(loss).all()),
                "xbar_absmax": float(xbar.abs().max()),
                "student_absmax": float(student.abs().max()),
                "loss_value": float(loss),
                "lengths": [int(len(x)) for x in states],
            }
            if not all(report[k] for k in ("xbar", "student", "target", "loss")):
                print(f"[qformer] L_sem forward already non-finite: {report}", flush=True)
            sem_forward.clear()
            sem_forward.update(report)
        (sem_weight * loss).backward()
        return float(loss)

    def obs_step(row: int, obs_weight: float, scale: float = 1.0) -> float:
        """One observation-distillation step: no benchmark label anywhere.

        The teacher read the screenshot, the AXTree and a task-independent
        probe, then generated its own continuation; the student sees the latent
        and the same probe and is scored on the same span. The span is the
        teacher's own output, so unlike ``CE_gold`` this term never consults an
        annotation.

        One probe per micro-batch, drawn from the training set only. ``P4`` is
        held out for eval: reporting a gap on a probe the run trained on would
        not distinguish "behavioural substitute for the observation" from
        "memorised one prompt's continuation".
        """
        sample_id = str(pairs["sample_id"][row])
        index = int(pairs["record_index"][row])
        probe = train_probes[int(rng.integers(len(train_probes)))]
        continuation, topk_index, topk_logprob = teachers_obs.continuation(
            sample_id, index, probe
        )
        states = trunk_cache.get((sample_id, index))
        if states is None:
            record = store.observation(sample_id, index)
            with Image.open(record["screenshot"]) as handle:
                image = handle.convert("RGB")
            states = trunk_states(processor, model, image, record["synthetic_axtree"],
                                  layer=args.layer, device=device)
        soft, _xbar, valid = joint(*collate([states]))
        kl = observation_distill_kl(
            model, processor, soft, valid, PROBES[probe],
            continuation, topk_index, topk_logprob,
        )
        # ``scale`` is 1/accumulate. sem_step runs once per optimizer step so it
        # needs no scaling; this one runs inside the accumulation loop, and
        # backpropagating full weight on every micro-batch made --obs-weight 1.0
        # behave as 4.0. That is why the arms diverged in order of their weight
        # -- 1.0 at step 532, 0.8 at 756, 0.5 not at all -- and why w=2.0's val
        # CE collapsed to 2.71: it was running at an effective 8.0.
        (obs_weight * scale * kl).backward()
        return float(kl)

    if args.sweep_nonfinite:
        # Every earlier hunt for these gradients needed the failure to happen
        # live, and the run died with the state in it. The w=1.0 arm ended
        # differently: the rate hit its floor at step 1911 and 280 further skips
        # arrived over the next 2088 steps with the weights effectively frozen,
        # so the failing state is not a transient -- it is a checkpoint on disk.
        # Loading it and pushing rows through one at a time turns a stochastic
        # event into a static question: which inputs, and always the same ones?
        rows = [int(row) for group in train_groups for row in group]
        order = rng.permutation(len(rows))[:args.sweep_nonfinite]
        print(f"[sweep] {len(order)} of {len(rows)} training rows on frozen "
              f"weights, obs_weight {args.obs_weight}, repeats {args.sweep_repeats}",
              flush=True)

        def verdict(row: int) -> dict:
            """Finiteness after each backward, in training's exact order."""
            optimizer.zero_grad(set_to_none=True)
            trunk_cache.clear()
            result = {"row": row,
                      "sample_id": str(pairs["sample_id"][row]),
                      "record_index": int(pairs["record_index"][row])}
            loss, _ce, _kl, _sem, _xbar = loss_for(
                row, args.distill_weight,
                args.sem_weight if args.sem_mode == "cosine" else 0.0,
            )
            (loss / args.accumulate).backward()
            result["after_qa"] = grad_norm_now()
            if args.obs_weight > 0:
                obs_step(row, args.obs_weight, 1.0 / args.accumulate)
                result["after_obs"] = grad_norm_now()
            result["finite"] = all(
                math.isfinite(result[k]) for k in ("after_qa", "after_obs")
                if k in result
            )
            return result

        if args.dissect_rows:
            # The sweep answers "which rows"; this answers "what about them".
            # 19 of 21 offenders were already non-finite after CE+KL_q, so the
            # new L_obs is not the term to instrument -- and one of them carried
            # a *finite* 1.85e19, whose square (3.4e38) is exactly where fp32
            # ends. That is a different failure from an infinite gradient: the
            # norm reduction overflows on a large-but-representable gradient,
            # and clip_grad_norm_ then reports inf for something the guard could
            # in principle have clipped.
            for row in args.dissect_rows:
                optimizer.zero_grad(set_to_none=True)
                trunk_cache.clear()
                sample_id = str(pairs["sample_id"][row])
                answer = str(pairs["answer"][row])
                question = str(pairs["question"][row])
                print(f"\n=== row {row}  {sample_id}[{int(pairs['record_index'][row])}]")
                print(f"    Q {question[:110]!r}")
                print(f"    A {answer[:110]!r}")

                soft, xbar, valid, record, sid, index = latents_for(row)
                states = trunk_cache[(sid, index)]
                flat = states.hidden.float()
                # input_projection is the first thing the trunk output meets, and
                # it is where every offender's gradient blows up. grad_W is
                # grad_out^T @ input, so a normal loss and a huge input give a
                # huge weight gradient with nothing non-finite anywhere in the
                # forward -- which is exactly what the loss values here show.
                print(f"    trunk   |max| {float(flat.abs().max()):.6g}  "
                      f"p99.9 {float(flat.abs().flatten().quantile(0.999)):.4g}  "
                      f"median {float(flat.abs().median()):.4g}  "
                      f"shape {tuple(flat.shape)}")
                print(f"    latent  |max| {float(soft.abs().max()):.4g}  "
                      f"mean {float(soft.mean()):+.4g}  "
                      f"finite {bool(torch.isfinite(soft).all())}")
                print(f"    xbar    |max| {float(xbar.abs().max()):.4g}  "
                      f"finite {bool(torch.isfinite(xbar).all())}")

                cached = (teachers_obs.answer_topk(sid, row)
                          if args.distill_teacher == "observation" else None)
                if cached is not None:
                    tindex, tlogprob = cached
                    lp = tlogprob.float()
                    print(f"    teacher {tuple(tindex.shape)}  logprob "
                          f"[{float(lp.min()):.2f}, {float(lp.max()):.2f}]  "
                          f"retained mass {float(lp.exp().sum(-1).median()):.6f}  "
                          f"finite {bool(torch.isfinite(lp).all())}")

                # CE alone, then CE+KL, each from a clean slate, so the term
                # that carries the magnitude is named rather than inferred.
                # Parameter gradients say which module is extreme but not where
                # the amplification happened -- a huge weight gradient can be a
                # huge incoming gradient or a huge input. Hooking the tensors
                # between modules ranks the backward path itself, so the step
                # where the norm jumps is read off rather than inferred.
                traces: dict[str, float] = {}

                def watch(name):
                    def hook(_module, grad_input, grad_output):
                        value = grad_output[0]
                        if value is not None:
                            traces[name] = float(value.detach().norm())
                    return hook

                handles = [joint.connector.register_full_backward_hook(watch("connector"))]
                q = joint.qformer
                handles.append(q.output_projection.register_full_backward_hook(
                    watch("qformer.output_projection")))
                for depth, block in enumerate(q.blocks):
                    handles.append(block.register_full_backward_hook(
                        watch(f"qformer.blocks.{depth}")))
                handles.append(q.input_norm.register_full_backward_hook(
                    watch("qformer.input_norm")))
                handles.append(q.input_projection.register_full_backward_hook(
                    watch("qformer.input_projection")))
                optimizer.zero_grad(set_to_none=True)
                # The graph has to be built *after* the hooks exist: backward
                # hooks fire only for forwards that ran while they were
                # registered, and the first attempt reused the ``soft`` computed
                # above, so every trace came back empty.
                trunk_cache.clear()
                soft2, _xbar2, valid2, _rec2, _sid2, _idx2 = latents_for(row)
                loss, ce, _kl = answer_ce_and_distill_kl(
                    model, processor, soft2, valid2, question, answer,
                    str(record.get("fused_text", "")),
                    max_answer_tokens=args.max_answer_tokens, weight=0.0,
                )
                loss.backward()
                for handle in handles:
                    handle.remove()
                print("    反传路径（梯度流向：从下往上）")
                order = ["qformer.input_projection", "qformer.input_norm",
                         "qformer.blocks.0", "qformer.blocks.1", "qformer.blocks.2",
                         "qformer.blocks.3", "qformer.output_projection", "connector"]
                previous = None
                for name in reversed(order):
                    if name not in traces:
                        continue
                    value = traces[name]
                    jump = "" if previous is None or previous == 0 else f"  x{value / previous:.3g}"
                    print(f"        |grad_out| {value:12.4g}  {name}{jump}")
                    previous = value

                # LayerNorm's backward carries a 1/std factor, so the amplifier
                # the trace points at is a token whose projection is nearly
                # constant across the hidden dimension. Rank the per-token
                # variance to see whether such a token exists and what it is.
                with torch.no_grad():
                    pre = joint.qformer.input_projection(
                        states.hidden.to(joint.qformer.input_projection.weight.dtype))
                    variance = pre.var(-1, unbiased=False)
                    worst = int(variance.argmin())
                    names = {0: "IMAGE", 1: "DOM", 2: "INSTRUCTION", 3: "WRAPPER"}
                    print(f"    input_norm 输入方差  min {float(variance.min()):.4g}  "
                          f"p1 {float(variance.quantile(0.01)):.4g}  "
                          f"median {float(variance.median()):.4g}")
                    print(f"        最小方差 token #{worst}/{len(variance)}  "
                          f"modality {names.get(int(states.modality_ids[worst]), '?')}  "
                          f"1/sqrt(var+1e-5) = "
                          f"{1.0 / math.sqrt(float(variance[worst]) + 1e-5):.4g}  "
                          f"trunk |x| {float(states.hidden[worst].abs().max()):.4g}")
                    below = int((variance < 1e-5).sum())
                    print(f"        方差 < eps(1e-5) 的 token 数: {below}/{len(variance)}")

                for label, weight in (("CE only", 0.0), ("CE+KL_q", args.distill_weight)):
                    optimizer.zero_grad(set_to_none=True)
                    loss, ce, kl = answer_ce_and_distill_kl(
                        model, processor, soft, valid, question, answer,
                        str(record.get("fused_text", "")),
                        max_answer_tokens=args.max_answer_tokens, weight=weight,
                        teacher_topk=cached if weight > 0 else None,
                    )
                    loss.backward(retain_graph=True)
                    total = grad_norm_now()
                    biggest = max(
                        ((n, float(p.grad.detach().abs().max()))
                         for n, p in joint.named_parameters() if p.grad is not None),
                        key=lambda item: (math.isfinite(item[1]), item[1]),
                        default=("-", 0.0),
                    )
                    print(f"    {label:8s}  loss {float(loss):.4f}  CE {ce:.4f}  "
                          f"KL {kl:.4f}  |grad| {total:.4g}  "
                          f"largest {biggest[0]} {biggest[1]:.4g}")
                    if label == "CE only":
                        # input_projection is the *first* layer, so a huge weight
                        # gradient there means the gradient arriving at its output
                        # was already huge -- the explosion is downstream and
                        # propagated back. Ranking every parameter says how far
                        # downstream: if only the earliest layers are extreme the
                        # amplifier sits between them, and input_norm is a
                        # LayerNorm whose backward divides by a standard
                        # deviation nobody has bounded away from zero.
                        ranked = sorted(
                            ((n, float(p.grad.detach().norm()))
                             for n, p in joint.named_parameters() if p.grad is not None),
                            key=lambda item: -item[1],
                        )
                        for name, value in ranked[:8]:
                            print(f"        {value:12.4g}  {name}")
                        print(f"        ... {len(ranked) - 8} more, smallest "
                              f"{ranked[-1][1]:.4g} ({ranked[-1][0]})")
            return

        offenders, checked = [], []
        for position, index in enumerate(order, 1):
            found = verdict(rows[int(index)])
            checked.append(found)
            if not found["finite"]:
                offenders.append(found)
            if position % 100 == 0 or position == len(order):
                print(f"[sweep] {position}/{len(order)}  non-finite so far "
                      f"{len(offenders)} ({100 * len(offenders) / position:.1f}%)",
                      flush=True)

        # Same weights, same input, N more times. Identical verdicts implicate
        # the input; verdicts that vary implicate the kernels.
        repeats = {}
        for found in offenders:
            repeats[found["row"]] = [
                verdict(found["row"])["finite"] for _ in range(args.sweep_repeats)
            ]
        stable = sum(1 for votes in repeats.values() if not any(votes))
        report_path = Path(args.output).with_suffix(".sweep.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "init_from": args.init_from,
            "swept": len(checked),
            "non_finite": len(offenders),
            "rate": len(offenders) / max(len(checked), 1),
            "reproduced_every_repeat": stable,
            "sweep_repeats": args.sweep_repeats,
            "obs_weight": args.obs_weight,
            "distill_weight": args.distill_weight,
            "offenders": offenders,
            "repeat_votes": {str(k): v for k, v in repeats.items()},
        }, indent=2, sort_keys=True))
        print(f"\n[sweep] {len(offenders)}/{len(checked)} rows non-finite "
              f"({100 * len(offenders) / max(len(checked), 1):.2f}%); "
              f"{stable}/{len(offenders)} reproduced on all "
              f"{args.sweep_repeats} repeats -> {report_path}", flush=True)
        return


    def scheduled_lr(step: int) -> float:
        """The rate warmup alone would set -- the ceiling recovery may climb to.

        Capping at this rather than at ``args.learning_rate`` matters only in one
        case, but it is a case the decay-only code got wrong: warmup below stops
        applying once anything has been skipped, so a skip inside the warmup
        window would otherwise let recovery restore the *full* rate at a step
        warmup had deliberately held down.
        """
        if not args.warmup_steps:
            return args.learning_rate
        return args.learning_rate * min(1.0, step / args.warmup_steps)

    best, best_step, stale, history, skipped = math.inf, 0, 0, [], 0
    clean_run = 0
    dropped = dropped_sem = 0
    drop_log: list[tuple] = []
    accumulator = None
    if args.drop_microbatches:
        # One buffer mirroring the trainable parameters (78.9M x 4 bytes ~ 316
        # MB). Each micro-batch backwards into a cleared ``.grad``, is tested on
        # its own, and is added here only if it passes; the mean at the end is
        # over the survivors.
        accumulator = [torch.zeros_like(p) for p in joint.parameters()]
        print(f"[qformer] per-micro-batch guard on: drop above norm "
              f"{args.microbatch_max_norm:g} "
              f"({sum(b.numel() for b in accumulator) * 4 / 2**20:.0f} MB buffer)",
              flush=True)

    def microbatch_norm() -> float:
        """Total norm of the gradient currently in ``.grad``, in float64.

        float64 for one reason: an offender measured 1.85e19, whose square is
        3.4e38 -- exactly where float32 ends. Summing squares in float32 turns a
        large-but-finite gradient into inf, so the guard could not tell "the
        gradient is infinite" from "my own reduction overflowed". In float64 the
        same sum is 3.4e38 with 270 orders of magnitude to spare, and the value
        that comes back is the real one.
        """
        total = 0.0
        for parameter in joint.parameters():
            if parameter.grad is None:
                continue
            value = float(parameter.grad.detach().double().norm())
            if not math.isfinite(value):
                return value
            total += value * value
        return math.sqrt(total)
    best_gap, best_gap_step = -math.inf, 0
    for step in range(1, args.max_steps + 1):
        if args.warmup_steps and step <= args.warmup_steps and skipped == 0:
            # There was no scheduler at all: the first update took the full
            # learning rate against a resampler initialised at std 0.02. The
            # divergences all share a shape -- hundreds of clean steps, then a
            # cluster of infinite gradients from which the run never recovers --
            # which is what an optimizer that has already walked somewhere bad
            # looks like. Intermediate activations were separately measured
            # climbing monotonically from 4.3 to 59.5 over 900 steps.
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * min(1.0, step / args.warmup_steps)
        joint.train()
        optimizer.zero_grad(set_to_none=True)
        # Entries hold a (~1100, 4096) bf16 trunk output each. They are only
        # useful within one step, and keeping them would be ~9 MB x 4 x 9000.
        trunk_cache.clear()
        totals = collections.Counter()
        drawn = []
        kept = 0
        if accumulator is not None:
            # Allocated once and reused, so it has to be cleared here. Without
            # this the buffer sums every micro-batch of every step and the run
            # walks off after a few dozen updates.
            for buffer in accumulator:
                buffer.zero_()
        for _ in range(args.accumulate):
            # Uniform over observations, then uniform over that observation's
            # questions -- P(o) = 1/N, q ~ Q(o). Identical in both sem modes.
            group = train_groups[int(rng.integers(len(train_groups)))]
            row = int(rng.choice(group))
            drawn.append((str(pairs["sample_id"][row]), int(pairs["record_index"][row])))
            loss, ce, kl, sem, _ = loss_for(
                row, args.distill_weight,
                args.sem_weight if args.sem_mode == "cosine" else 0.0,
            )
            # Unscaled: the mean is taken at the end over the micro-batches that
            # survived, not over the ones that were drawn. Dividing by a fixed
            # ``accumulate`` here would quietly shrink every step that dropped
            # something, which is a learning-rate cut applied exactly to the
            # batches containing the pathological rows.
            loss.backward()
            if args.locate_nonfinite:
                after_qa = grad_norm_now()
                if not math.isfinite(after_qa):
                    print(f"[qformer] step {step:5d}  NON-FINITE after CE+KL_q  "
                          f"({after_qa}); {drawn[-1]}", flush=True)
            if args.obs_weight > 0:
                obs_value = obs_step(row, args.obs_weight, 1.0)
                if args.locate_nonfinite:
                    after_obs = grad_norm_now()
                    if math.isfinite(after_qa) and not math.isfinite(after_obs):
                        print(f"[qformer] step {step:5d}  NON-FINITE after L_obs  "
                              f"({after_obs}); {drawn[-1]}", flush=True)
            if accumulator is not None:
                # Drop the micro-batch, not the step. 2.1% of rows produce a
                # gradient 13 orders of magnitude above the rest; under the old
                # guard one of them discarded all four micro-batches, which cost
                # 13% of steps on the w=1.0 arm. The magnitude test is not
                # redundant with the finiteness test: two of the offenders
                # measured 1.85e19 and 1.9e15, both *finite*, and a finite 1e19
                # survives clipping to take the entire step budget in its own
                # direction while annihilating the other micro-batches to 1e-26.
                norm = microbatch_norm()
                if math.isfinite(norm) and norm <= args.microbatch_max_norm:
                    for buffer, parameter in zip(accumulator, joint.parameters()):
                        if parameter.grad is not None:
                            buffer.add_(parameter.grad)
                    kept += 1
                    totals["ce"] += ce
                    totals["kl"] += kl
                    totals["obs"] += obs_value if args.obs_weight > 0 else 0.0
                    if args.sem_mode == "cosine":
                        totals["sem"] += sem
                else:
                    dropped += 1
                    drop_log.append((step, drawn[-1], norm))
                optimizer.zero_grad(set_to_none=True)
            else:
                kept += 1
                totals["ce"] += ce
                totals["kl"] += kl
                totals["obs"] += obs_value if args.obs_weight > 0 else 0.0
                if args.sem_mode == "cosine":
                    totals["sem"] += sem
        if accumulator is not None:
            if kept == 0:
                skipped += 1
                clean_run = 0
                print(f"[qformer] step {step:5d}  every micro-batch dropped; "
                      f"drawn={drawn}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                if skipped > args.max_skipped_steps:
                    raise ValueError(
                        f"{skipped} steps lost every micro-batch: this is not a "
                        "handful of bad rows, it is the objective or the data"
                    )
                continue
        for key in ("ce", "kl", "obs"):
            if totals[key]:
                totals[key] /= max(kept, 1)
        if args.sem_mode == "cosine" and totals["sem"]:
            totals["sem"] /= max(kept, 1)
        if args.sem_weight > 0 and args.sem_mode == "same-session":
            # L_sem runs once per step rather than per micro-batch, so it gets
            # the same treatment on its own terms: computed into a cleared
            # buffer, tested, and admitted or dropped by itself. Letting it
            # backward into the surviving average would put the whole step back
            # at the mercy of the one term this guard cannot otherwise isolate.
            if accumulator is not None:
                optimizer.zero_grad(set_to_none=True)
            before_sem = grad_norm_now() if args.locate_nonfinite else 0.0
            sem_value = sem_step(args.sem_weight)
            if args.locate_nonfinite:
                after_sem = grad_norm_now()
                if math.isfinite(before_sem) and not math.isfinite(after_sem):
                    print(f"[qformer] step {step:5d}  NON-FINITE after L_sem  "
                          f"({after_sem}); head pre-norm "
                          f"min {head_norm.get('min', float('nan')):.4f} "
                          f"max {head_norm.get('max', float('nan')):.4f} "
                          f"finite {head_norm.get('finite')}; forward {sem_forward}",
                          flush=True)
            if accumulator is not None:
                norm = microbatch_norm()
                if math.isfinite(norm) and norm <= args.microbatch_max_norm:
                    totals["sem"] += sem_value
                    for buffer, parameter in zip(accumulator, joint.parameters()):
                        if parameter.grad is not None:
                            buffer.add_(parameter.grad, alpha=float(kept))
                else:
                    dropped_sem += 1
                    drop_log.append((step, "L_sem", norm))
            else:
                totals["sem"] += sem_value
        if accumulator is not None:
            for buffer, parameter in zip(accumulator, joint.parameters()):
                parameter.grad = buffer / kept
        grad_norm = torch.nn.utils.clip_grad_norm_(joint.parameters(), args.clip_norm)
        if not torch.isfinite(grad_norm):
            # Every divergence in this project has looked the same: losses in
            # normal ranges at the last eval, then "xbar contains non-finite
            # values" at a step nobody can predict. That shape says one
            # micro-batch produced a bad gradient, the weights went non-finite,
            # and the *next* forward reported it -- so the traceback has always
            # pointed at the symptom rather than the cause. Skipping the update
            # is what every mixed-precision trainer does, and it turns a fatal
            # run into a logged event with the offending observations attached.
            skipped += 1
            # Skipping alone has no escape: the weights do not move, so the next
            # batch meets the same model and the run stalls in a cluster of
            # skips until it hits the cap. Halving on each skip is what a
            # GradScaler does, and it works without knowing the cause -- which
            # matters here, because six hypotheses for that cause have now been
            # falsified (slot count, L_sem's form, the learning rate, unbounded
            # forward KL, the missing warmup, the retrieval head's normalise)
            # and the event reproduces only stochastically: the same config and
            # seed diverged at step 447 once and ran clean past 450 the next
            # time. The floor keeps a run from decaying into a no-op.
            #
            # It did not work, and the w=1.0 log says why: 7 skips carried the
            # rate from 1e-4 to the 1e-6 floor in 63 steps, and then 280 more
            # skips arrived at that floor over the next 2088 steps, one every
            # 7.5. A 100x cut in the rate left the skip rate unchanged. At 1e-6
            # the weights move by at most ~2e-3 in total, so the model is frozen
            # and the only thing varying between steps is the input -- which is
            # what --sweep-nonfinite exists to interrogate. Keep the decay
            # (skipping still needs *some* escape and it costs little), but stop
            # believing it is the fix.
            clean_run = 0
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"] * 0.5, args.learning_rate * args.min_lr_fraction)
            print(f"[qformer] step {step:5d}  non-finite gradient "
                  f"({float(grad_norm)}), update skipped, lr -> "
                  f"{optimizer.param_groups[0]['lr']:.3g}; drawn={drawn}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            if skipped > args.max_skipped_steps:
                raise ValueError(
                    f"{skipped} non-finite gradients: this is not an occasional "
                    "bad batch, it is the objective or the data"
                )
            continue
        optimizer.step()
        clean_run += 1
        if args.lr_recover_steps and clean_run >= args.lr_recover_steps:
            # The other half of GradScaler: back off immediately on a bad step,
            # grow again only after the run has behaved for a while. The
            # asymmetry is the point -- decay is one step, recovery is
            # --lr-recover-steps of them -- so an occasional bad batch costs a
            # transient dip instead of holding the rate down permanently.
            ceiling = scheduled_lr(step)
            current = optimizer.param_groups[0]["lr"]
            if current < ceiling:
                for group in optimizer.param_groups:
                    group["lr"] = min(group["lr"] * 2.0, ceiling)
                print(f"[qformer] step {step:5d}  {clean_run} clean steps, "
                      f"lr -> {optimizer.param_groups[0]['lr']:.3g}", flush=True)
            clean_run = 0

        if step % args.eval_every and step != args.max_steps:
            continue

        joint.eval()
        trunk_cache.clear()
        losses, learned, pooled, held_out = [], [], [], []
        held_out_mismatched: list[float] = []
        mismatch_soft = mismatch_valid = None
        with torch.no_grad():
            for group in chosen_val:
                # Every question of the observation, averaged inside it before
                # averaging across observations: an unbiased observation-level
                # estimate with no "which question got drawn" noise. The state
                # is identical for all of them, so it enters the monitor once.
                per_observation, xbar = [], None
                for row in group:
                    _loss, ce, _kl, _sem, xbar = loss_for(int(row), 0.0)
                    per_observation.append(ce)
                losses.append(float(np.mean(per_observation)))
                learned.append(xbar[0].float().cpu().numpy())
                pooled.append(store.pooled_xbar(
                    str(pairs["sample_id"][group[0]]), int(pairs["record_index"][group[0]])
                ))
                if teachers_obs is not None and len(held_out) < args.probe_observations:
                    # The held-out probe, never sampled during training. This is
                    # the first KL in this block: loss_for is called with
                    # weight=0.0 above, so the distillation term is off at eval.
                    sample_id = str(pairs["sample_id"][group[0]])
                    index = int(pairs["record_index"][group[0]])
                    continuation, ti, tl = teachers_obs.continuation(
                        sample_id, index, args.held_out_probe
                    )
                    states = trunk_cache.get((sample_id, index))
                    if states is None:
                        record = store.observation(sample_id, index)
                        with Image.open(record["screenshot"]) as handle:
                            image = handle.convert("RGB")
                        states = trunk_states(processor, model, image,
                                              record["synthetic_axtree"],
                                              layer=args.layer, device=device)
                    soft, _x, valid_o = joint(*collate([states]))
                    held_out.append(float(observation_distill_kl(
                        model, processor, soft, valid_o,
                        PROBES[args.held_out_probe], continuation, ti, tl,
                    )))
                    # The same teacher scored against the *previous*
                    # observation's latent. The matched KL alone cannot say
                    # whether the state carries this screen: most of it is the
                    # generic cost of a 32-slot prefix standing in for ~1100
                    # real tokens, and that part is identical either way.
                    # Subtracting gives the observation-specific part, which is
                    # the quantity this whole line is about -- and which, being
                    # the conditional rather than the marginal, is exactly what
                    # appears late in training. Without the trajectory we would
                    # only see it at step 9000.
                    if mismatch_soft is not None:
                        held_out_mismatched.append(float(observation_distill_kl(
                            model, processor, mismatch_soft, mismatch_valid,
                            PROBES[args.held_out_probe], continuation, ti, tl,
                        )))
                    mismatch_soft, mismatch_valid = soft, valid_o
        validation = float(np.mean(losses))
        gap = (float(np.mean(held_out_mismatched)) - float(np.mean(held_out[:-1]))
               if held_out_mismatched else None)
        # Same observations, both representations -- the comparison is paired.
        monitors = {"learned": spread(np.stack(learned)),
                    "pooled": spread(np.stack(pooled))}
        regressed = (
            monitors["learned"]["pairwise_cosine"] > monitors["pooled"]["pairwise_cosine"]
            or monitors["learned"]["effective_rank"] < monitors["pooled"]["effective_rank"]
        )
        history.append({
            "step": step, "validation_answer_ce": validation,
            "train_ce": totals["ce"] / args.accumulate,
            "train_kl": totals["kl"] / args.accumulate,
            # One draw per step, not one per micro-batch: L_sem has its own
            # sampler and its own batch, so dividing by accumulate would be wrong.
            "train_sem": totals["sem"],
            "train_obs": totals["obs"],
            "held_out_probe_kl": (float(np.mean(held_out)) if held_out else None),
            "held_out_probe_kl_mismatched": (float(np.mean(held_out_mismatched))
                                             if held_out_mismatched else None),
            "held_out_probe_gap": gap,
            "monitors": monitors, "collapse_regressed": regressed,
        })
        flag = "  COLLAPSE-REGRESSED" if regressed else ""
        print(
            f"[qformer] step {step:5d}  val CE {validation:.4f}  "
            f"(train CE {totals['ce']/args.accumulate:.4f}, "
            f"KL {totals['kl']/args.accumulate:.4f}, "
            f"sem {totals['sem']:.4f}, obs {totals['obs']:.4f})  "
            + (f"headmin {head_norm['min']:.4f}  " if "min" in head_norm else "")
            + (f"P4 {np.mean(held_out):.4f}" if held_out else "")
            + (f"/{np.mean(held_out_mismatched):.4f} gap "
               f"{np.mean(held_out_mismatched) - np.mean(held_out[:-1]):+.4f}  "
               if held_out_mismatched else "  " if held_out else "")
            + f"cos {monitors['learned']['pairwise_cosine']:+.4f}"
            f"/{monitors['pooled']['pairwise_cosine']:+.4f}  "
            f"rank {monitors['learned']['effective_rank']:.1f}"
            f"/{monitors['pooled']['effective_rank']:.1f}  "
            f"mean/dev {monitors['learned']['mean_to_deviation']:.1f}"
            f"/{monitors['pooled']['mean_to_deviation']:.1f}{flag}",
            flush=True,
        )
        def keep(path: str, why: str) -> None:
            save_bridge(
                path, joint, protocol=QFORMER_PROTOCOL,
                queries=args.queries, layer=args.layer,
                best_step=step, validation_answer_ce=validation,
                held_out_probe_gap=gap, selected_by=why,
                distill_weight=args.distill_weight, obs_weight=args.obs_weight,
                objective="answer_ce+distill_kl",
                qformer_sha256=qformer_hash(joint.qformer),
                monitors=monitors,
            )

        # Two checkpoints, because the two numbers peak at different steps and
        # only one of them is what this experiment is about. On the w=0.5 arm
        # val CE bottomed at 1250 while the held-out gap peaked at 1500, and
        # keeping only the CE-best threw away the better representation. Early
        # stopping still watches CE -- that keeps the stopping rule comparable
        # with every run before this one -- but the artifact selected on the gap
        # is now preserved alongside it.
        if validation < best:
            best, best_step, stale = validation, step, 0
            keep(args.output, "validation_answer_ce")
        else:
            stale += 1
        if gap is not None and gap > best_gap:
            best_gap, best_gap_step = gap, step
            keep(str(Path(args.output).with_suffix(".gapbest.pt")), "held_out_probe_gap")
        if stale >= args.patience_evals:
            break

    report = {
        "protocol": QFORMER_PROTOCOL,
        # Spelled out rather than a bare "answer_ce+distill_kl": the previous
        # run's report recorded neither the sem weight nor its form, so which
        # L_sem it used could not be recovered from the artifact afterwards.
        "objective": "+".join(filter(None, [
            "answer_ce", f"distill_kl[{args.distill_teacher}]",
            f"sem[{args.sem_mode}]" if args.sem_weight > 0 else None,
            "observation_kl" if args.obs_weight > 0 else None,
        ])),
        "queries": args.queries, "distill_weight": args.distill_weight,
        "accumulate": args.accumulate, "best_step": best_step,
        "obs_weight": args.obs_weight, "distill_teacher": args.distill_teacher,
        "held_out_probe": args.held_out_probe,
        "teacher_cache": (str(Path(args.teacher_cache).resolve())
                          if args.teacher_cache else None),
        "skipped_steps": skipped,
        "dropped_microbatches": dropped,
        "dropped_sem_steps": dropped_sem,
        "drop_microbatches": bool(args.drop_microbatches),
        "microbatch_max_norm": args.microbatch_max_norm,
        # Which rows never contributed. 2.1% of them is a small number and a
        # real selection bias -- they cluster on the longer observations -- so
        # the paper has to disclose it rather than call the training set whole.
        "dropped_rows": [
            {"step": s, "row": r, "norm": n} for s, r, n in drop_log[:500]
        ],
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "min_lr_fraction": args.min_lr_fraction,
        "best_gap": (best_gap if best_gap > -math.inf else None),
        "best_gap_step": best_gap_step,
        "sem_mode": args.sem_mode,
        "sem_weight": args.sem_weight, "sem_batch": args.sem_batch,
        "sem_extra_negatives": args.sem_extra_negatives,
        "sem_temperature": args.sem_temperature,
        "sem_sessions": len(sem_sessions) if args.sem_weight > 0 else 0,
        "trunk_dtype": "float32" if args.trunk_fp32 else "bfloat16",
        "learning_rate": args.learning_rate, "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "best_validation_answer_ce": best, "pairs_metadata": metadata,
        "history": history,
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
