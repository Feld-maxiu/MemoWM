"""Offline, rescue-only second-pass calibration of the frozen WMA utility gate.

This experiment does not modify the released WMA adapter or gate. Under one
shared posterior and lambda, two sequential keep tests are exactly equivalent
to keep_mask(max(U_original, U_conditional), entropy, lambda). The resulting
artifact therefore uses the existing decoder interface without transmitting a
per-observation mask. Missing second-pass measurements fall back to the original
policy and are explicitly reported, not mistaken for measured zero utility.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.utility_gate.export_mask import MASK_PROTOCOL, keep_mask, per_state_rate

PROTOCOL = 'residualmem_two_stage_utility_v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def read_mask(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        if meta.get('protocol') != MASK_PROTOCOL:
            raise ValueError('initial mask protocol mismatch')
        utility = np.asarray(data['utility'], np.float64)
        lam = float(data['lam'])
    if utility.shape != (1024,) or not np.isfinite(utility).all() or (utility < 0).any():
        raise ValueError('invalid initial utility')
    return utility, lam, meta


def restore_variants(truth, fill, keep, positions):
    """Whole-mask reference followed by one restoration for each dropped code."""
    truth = np.asarray(truth)
    fill = np.asarray(fill)
    keep = np.asarray(keep, bool)
    if truth.shape != fill.shape or keep.shape != truth.shape:
        raise ValueError('unaligned codes/fill/keep')
    positions = np.asarray(positions, np.int64)
    if len(np.unique(positions)) != len(positions):
        raise ValueError('duplicate restoration positions')
    if ((positions < 0) | (positions >= truth.size)).any() or keep.reshape(-1)[positions].any():
        raise ValueError('restore only first-pass dropped positions')
    baseline = np.where(keep, truth, fill)
    variants = np.repeat(baseline[None], len(positions) + 1, axis=0)
    for index, position in enumerate(positions, 1):
        variants[index].reshape(-1)[position] = truth.reshape(-1)[position]
    return variants


def combine_utilities(original, totals, counts, *, minimum_labels=1):
    original = np.asarray(original, np.float64)
    totals = np.asarray(totals, np.float64)
    counts = np.asarray(counts, np.int64)
    if original.shape != totals.shape or original.shape != counts.shape:
        raise ValueError('unaligned utility statistics')
    if minimum_labels < 1 or (counts < 0).any() or (totals < 0).any() or not np.isfinite(totals).all():
        raise ValueError('invalid utility statistics')
    measured = counts >= minimum_labels
    conditional = original.copy()
    conditional[measured] = totals[measured] / counts[measured]
    return np.maximum(original, conditional), conditional, measured


def select_states(state_rows, drop, sample_names, max_states, seed):
    """Cover rare candidate positions before filling a diverse random pilot.

    Selection reads only question availability, sample ids and first-pass masks,
    never answer content or measured second-pass effects.
    """
    rng = np.random.default_rng(seed)
    state_rows = np.asarray(state_rows, np.int64)
    drop = np.asarray(drop, bool)
    if not len(state_rows) or drop.shape != (len(state_rows), 1024) or max_states < 1:
        raise ValueError('invalid pilot candidate states')
    target = np.minimum(drop.sum(0), 4)
    selected = []
    covered = np.zeros(1024, np.int64)
    remaining = set(range(len(state_rows)))
    while remaining and len(selected) < max_states and (covered < target).any():
        weights = (covered < target) / np.maximum(drop.sum(0), 1)
        scores = drop @ weights
        best = min(remaining, key=lambda i: (-scores[i], int(state_rows[i])))
        if scores[best] == 0:
            break
        selected.append(best)
        remaining.remove(best)
        covered += drop[best]
    sample_counts = Counter(str(sample_names[i]) for i in selected)
    random_order = {int(i): n for n, i in enumerate(rng.permutation(len(state_rows)))}
    while remaining and len(selected) < max_states:
        best = min(remaining, key=lambda i: (sample_counts[str(sample_names[i])], random_order[i]))
        selected.append(best)
        remaining.remove(best)
        sample_counts[str(sample_names[best])] += 1
    return np.sort(state_rows[selected])


def load_inputs(args):
    from experiments.utility_gate.label_counterfactual import cache_order
    original, lam, mask_meta = read_mask(args.mask)
    with np.load(args.gate / 'codes.npz', allow_pickle=True) as data:
        codes = np.asarray(data['codes/train'], np.uint8)
        state_ids = data['state_ids/train'].astype(str)
    with np.load(args.posteriors, allow_pickle=False) as data:
        posterior = {key: np.asarray(data[key]) for key in data.files}
    order = cache_order(args.gate / 'records.jsonl', 'train')
    posterior_lookup = {int(value): i for i, value in enumerate(posterior['target_indices'])}
    posterior_rows = np.asarray([posterior_lookup.get(order.get(sid, -1), -1) for sid in state_ids])
    present = np.flatnonzero(posterior_rows >= 0)
    if not np.array_equal(codes[present], posterior['target_codes'][posterior_rows[present]]):
        raise ValueError('posterior true codes and gate codes disagree')
    with np.load(args.pairs, allow_pickle=False) as data:
        pairs = {key: np.asarray(data[key]) for key in data.files if key != 'metadata'}
    return original, lam, mask_meta, codes, state_ids, posterior, posterior_rows, pairs


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    original, lam, _, codes, state_ids, posterior, posterior_rows, pairs = load_inputs(args)
    lookup = {sid: index for index, sid in enumerate(state_ids)}
    pair_state = np.asarray([lookup.get(f'{sample}-{int(record):04d}', -1)
                            for sample, record in zip(pairs['sample_id'], pairs['record_index'])])
    eligible = [i for i, state in enumerate(pair_state)
                if state >= 0 and posterior_rows[state] >= 0 and pairs['split'][i] == 'train'
                and str(pairs['question'][i]).strip() and str(pairs['answer'][i]).strip()]
    states = np.unique(pair_state[eligible])
    drop = ~keep_mask(original, posterior['entropy_bits'][posterior_rows[states]], lam)
    chosen = select_states(states, drop, [state_ids[i].rsplit('-', 1)[0] for i in states], args.max_states, args.seed)
    chosen_set = set(chosen.tolist())
    rows = [int(i) for i in eligible if int(pair_state[i]) in chosen_set]
    chosen_drop = ~keep_mask(original, posterior['entropy_bits'][posterior_rows[chosen]], lam)
    files = {key: str(Path(value).resolve()) for key, value in {
        'mask': args.mask, 'codes': args.gate / 'codes.npz', 'records': args.gate / 'records.jsonl',
        'posteriors': args.posteriors, 'pairs': args.pairs, 'codebook': args.codebook, 'checkpoint': args.checkpoint,
    }.items()}
    output = {
        'protocol': PROTOCOL, 'stage': 'pilot', 'files': files,
        'sha256': {key: sha256(path) for key, path in files.items()},
        'lambda': lam, 'seed': args.seed, 'pair_split': 'train',
        'pair_rows': rows, 'state_rows': chosen.tolist(),
        'pair_state': {str(i): int(pair_state[i]) for i in rows},
        'state_ids': {str(i): str(state_ids[i]) for i in chosen},
        'posterior_rows': {str(i): int(posterior_rows[i]) for i in chosen},
        'observations': len(chosen), 'questions': len(rows),
        'samples': len({state_ids[i].rsplit('-', 1)[0] for i in chosen}),
        'eligible_candidate_positions': int(drop.any(0).sum()),
        'selected_candidate_positions': int(chosen_drop.any(0).sum()),
        'expected_labels': int(sum((~keep_mask(original, posterior['entropy_bits'][posterior_rows[pair_state[i]]].reshape(-1), lam)).sum() for i in rows)),
        'background': 'original gate with frozen all-send-history WM posterior; not final-policy closed loop',
        'decision': 'rescue only; original lambda fixed; effective utility=max(original, conditional mean)',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key not in {'pair_rows', 'pair_state', 'state_rows', 'state_ids', 'posterior_rows'}}, indent=2), flush=True)


def label(args):
    import torch
    from experiments.state_tokenizer.reader_losses import FULL_ANSWER_TOKENS, answer_variant_scores
    from experiments.utility_gate.verify_scorer import load_codebook, load_reader, rebuild
    plan = json.loads(args.plan.read_text())
    if plan['protocol'] != PROTOCOL or not 0 <= args.rank < args.world_size or args.chunk < 1:
        raise ValueError('invalid plan, shard or chunk')
    for key, path in plan['files'].items():
        if sha256(path) != plan['sha256'][key]:
            raise ValueError(f'input changed: {key}')
    original, lam, _ = read_mask(plan['files']['mask'])
    with np.load(plan['files']['codes'], allow_pickle=False) as data:
        codes = np.asarray(data['codes/train'], np.uint8)
    with np.load(plan['files']['posteriors'], allow_pickle=False) as data:
        entropy = np.asarray(data['entropy_bits'])
        fill = np.asarray(data['wm_argmax'])
    with np.load(plan['files']['pairs'], allow_pickle=False) as data:
        questions, answers = data['question'].astype(str), data['answer'].astype(str)
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    processor, model, connector = load_reader(args.model, plan['files']['checkpoint'], device, torch.float32, 32)
    book = load_codebook(Path(plan['files']['codebook']))
    rows = plan['pair_rows'][args.rank::args.world_size]
    if args.limit is not None:
        rows = rows[:args.limit]
    output = args.output / f'rank-{args.rank}'
    output.mkdir(parents=True, exist_ok=True)
    config = {'protocol': PROTOCOL, 'plan_sha256': sha256(args.plan), 'model': str(Path(args.model).resolve()),
              'rank': args.rank, 'world_size': args.world_size, 'chunk': args.chunk,
              'batch_size': args.chunk + 3, 'dtype': 'float32', 'answer_tokens': FULL_ANSWER_TOKENS,
              'pair_rows': rows}
    config_path = output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('refusing to mix labeling configurations')
    write_json(config_path, config)
    start_time = time.monotonic()
    scored_rows = 0
    for done, pair_row in enumerate(rows, 1):
        target = output / f'pair-{pair_row:06d}.npz'
        if target.exists():
            with np.load(target, allow_pickle=False) as cached:
                if str(cached['plan_sha256']) != config['plan_sha256'] or int(cached['pair_row']) != pair_row:
                    raise ValueError('invalid resumed label row')
            continue
        state = plan['pair_state'][str(pair_row)]
        posterior = plan['posterior_rows'][str(state)]
        truth = codes[state]
        keep = keep_mask(original, entropy[posterior].reshape(-1), lam).reshape(truth.shape)
        positions = np.flatnonzero(~keep.reshape(-1))
        effective = truth.reshape(-1)[positions] != fill[posterior].reshape(-1)[positions]
        active = positions[effective]
        gains = np.zeros(len(positions), np.float64)
        position_index = {int(j): i for i, j in enumerate(positions)}
        base_nll = full_nll = None
        row_null_max = 0.0
        row_started = time.monotonic()
        for offset in range(0, max(len(active), 1), args.chunk):
            block = active[offset:offset + args.chunk]
            restored = restore_variants(truth, fill[posterior], keep, block)
            batch_codes = np.repeat(restored[:1], args.chunk + 3, axis=0)
            batch_codes[1] = truth
            # Row 2 is an identical reference/null control in every forward.
            batch_codes[3:3 + len(block)] = restored[1:]
            latents = torch.as_tensor(rebuild(batch_codes, book), device=device, dtype=torch.float32)
            with torch.inference_mode():
                soft = connector(latents, torch.ones(latents.shape[:2], dtype=torch.bool, device=device))
                scored = answer_variant_scores(model, processor, soft, questions[pair_row], answers[pair_row], max_answer_tokens=FULL_ANSWER_TOKENS)
            nll = scored['nll_bits'].numpy().astype(np.float64)
            if not np.isfinite(nll).all():
                raise ValueError('non-finite reader NLL')
            null_error = abs(nll[2] - nll[0])
            row_null_max = max(row_null_max, null_error)
            if null_error > 1e-4:
                raise ValueError(f'null-control error too large: {null_error}')
            if base_nll is not None and max(abs(base_nll - nll[0]), abs(full_nll - nll[1])) > 1e-3:
                raise ValueError('reference drift across equal-geometry chunks')
            base_nll, full_nll = float(nll[0]), float(nll[1])
            for index, position in enumerate(block, 3):
                gains[position_index[int(position)]] = nll[0] - nll[index]
            del latents, soft, scored
        temporary = target.with_suffix('.partial.npz')
        np.savez_compressed(temporary, pair_row=pair_row, state_row=state, position=positions,
                            restore_gain_bits=gains, conditional_utility=np.abs(gains),
                            original_utility=original[positions], entropy_bits=entropy[posterior].reshape(-1)[positions],
                            effective=effective, baseline_nll_bits=base_nll, full_nll_bits=full_nll,
                            null_max_bits=row_null_max, plan_sha256=np.asarray(config['plan_sha256']))
        temporary.replace(target)
        scored_rows += 1
        print(json.dumps({'rank': args.rank, 'done': done, 'total': len(rows), 'pair_row': pair_row,
                          'positions': len(positions), 'effective_positions': len(active),
                          'row_seconds': round(time.monotonic() - row_started, 2),
                          'average_seconds': round((time.monotonic() - start_time) / scored_rows, 2),
                          'null_max_bits': row_null_max, 'cuda_peak_gib': torch.cuda.max_memory_allocated(device) / 2**30}), flush=True)
    write_json(output / 'complete.json', {'protocol': PROTOCOL, 'rows': len(rows), 'config': config})


def export(args):
    plan = json.loads(args.plan.read_text())
    plan_hash = sha256(args.plan)
    if args.output.exists():
        raise FileExistsError(args.output)
    original, lam, old_meta = read_mask(plan['files']['mask'])
    if sha256(plan['files']['mask']) != plan['sha256']['mask']:
        raise ValueError('initial mask changed')
    totals = np.zeros(1024, np.float64)
    counts = np.zeros(1024, np.int64)
    state_sets = [set() for _ in range(1024)]
    seen = set()
    protocols = set()
    for config_path in sorted(args.labels.glob('rank-*/config.json')):
        config = json.loads(config_path.read_text())
        protocols.add((config['plan_sha256'], config['model'], config['chunk'], config['dtype'], config['answer_tokens']))
    if len(protocols) != 1 or next(iter(protocols))[0] != plan_hash:
        raise ValueError('missing or mixed labeling configurations')
    for path in sorted(args.labels.glob('rank-*/pair-*.npz')):
        if path.name.endswith('.partial.npz'):
            continue
        with np.load(path, allow_pickle=False) as data:
            row = int(data['pair_row'])
            if row in seen or row not in plan['pair_rows'] or str(data['plan_sha256']) != plan_hash:
                raise ValueError(f'invalid or duplicated label row: {path}')
            state = int(data['state_row'])
            if state != plan['pair_state'][str(row)]:
                raise ValueError('label state mismatch')
            positions = np.asarray(data['position'], np.int64)
            values = np.asarray(data['conditional_utility'], np.float64)
            if len(positions) != len(np.unique(positions)) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('invalid labels')
            np.add.at(totals, positions, values)
            np.add.at(counts, positions, 1)
            for j in positions:
                state_sets[int(j)].add(state)
            seen.add(row)
    if seen != set(plan['pair_rows']) or int(counts.sum()) != plan['expected_labels']:
        raise ValueError(f'incomplete labels: {len(seen)}/{len(plan["pair_rows"])} rows, {counts.sum()}/{plan["expected_labels"]} labels')
    effective, conditional, measured = combine_utilities(original, totals, counts, minimum_labels=args.minimum_labels)
    state_counts = np.asarray([len(s) for s in state_sets], np.int64)
    reports = {}
    for name, path in [('fit_corpus', Path(plan['files']['posteriors'])), ('wma_web', args.test_posteriors)]:
        with np.load(path, allow_pickle=False) as data:
            entropy, bits = data['entropy_bits'], data['code_bits']
        before = keep_mask(original, entropy, lam)
        after = keep_mask(effective, entropy, lam)
        assert np.all(after | ~before)
        rescued = after & ~before
        both_drop = (~before & ~after).sum()
        reports[name] = {
            'observations': len(before), 'posterior_sha256': sha256(path),
            'comparison': 'same frozen all-send-history posteriors, not a closed-loop rate',
            'initial_rate_bits': per_state_rate(bits, before), 'upgraded_rate_bits': per_state_rate(bits, after),
            'initial_keep_fraction': float(before.mean()), 'upgraded_keep_fraction': float(after.mean()),
            'mask_agreement': float((before == after).mean()),
            'changed_positions_mean': float(rescued.sum(1).mean()),
            'rescued_fraction_of_initial_drops': float(rescued.sum() / max((~before).sum(), 1)),
            'dropped_set_jaccard': float(both_drop / max((~before | ~after).sum(), 1)),
            'unmeasured_candidate_positions': np.flatnonzero((~before).any(0) & ~measured).tolist(),
            'low_support_candidate_positions': np.flatnonzero((~before).any(0) & ((counts < 16) | (state_counts < 4))).tolist(),
            'newly_dropped_positions': int((before & ~after).sum()),
        }
    meta = {
        'protocol': MASK_PROTOCOL, 'estimator_protocol': PROTOCOL, 'stage': 'pilot',
        'lambda': lam, 'mask_bits': 0, 'plan': str(args.plan.resolve()), 'plan_sha256': plan_hash,
        'background_mask_sha256': plan['sha256']['mask'], 'background_lambda': lam,
        'rule': 'keep iff max(original_utility, conditional_utility) >= lambda * entropy',
        'utility_is': 'rescue-only envelope of original and mean absolute single-restoration NLL effect',
        'fitted_on_question_distribution': old_meta.get('fitted_on_question_distribution'),
        'pair_split': 'train', 'questions': len(seen), 'observations': plan['observations'],
        'labels': int(counts.sum()), 'measured_positions': int(measured.sum()),
        'increased_utility_positions': int((effective > original).sum()),
        'minimum_labels': args.minimum_labels, 'insufficient_support_policy': 'unchanged original utility, explicitly unverified',
        'gate_lambda_must_match_background': True, 'same_posterior_comparison': reports,
        'not_validated': ['closed-loop upgraded-policy rate', 'upgraded-policy QA quality'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, utility=effective, lam=lam, original_utility=original,
                        conditional_utility=conditional, label_counts=counts, state_counts=state_counts,
                        measured=measured, metadata=np.asarray(json.dumps(meta, ensure_ascii=False)))
    write_json(args.output.with_suffix('.json'), meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--gate', type=Path, default=Path('gate'))
    p.add_argument('--mask', type=Path, default=Path('gate/mask-lambda0.0010.npz'))
    p.add_argument('--posteriors', type=Path, default=Path('gate/posteriors.npz'))
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--codebook', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--max-states', type=int, default=96)
    p.add_argument('--seed', type=int, default=35)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('label')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:2')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=1)
    p.add_argument('--limit', type=int)
    p.add_argument('--chunk', type=int, default=6)
    p.add_argument('--memory-fraction', type=float, default=.55)
    p = sub.add_parser('export')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--test-posteriors', type=Path, default=Path('gate/posteriors-testset.npz'))
    p.add_argument('--minimum-labels', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'label': label, 'export': export}[args.command](args)


if __name__ == '__main__':
    main()
