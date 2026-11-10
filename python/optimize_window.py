"""One global (W, tau) per feedback architecture on six training scenarios, then frozen validation."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import numpy as np
from coding import Config, DEGREES, PMF
from channel import metadata
from feedback_options import ARCHITECTURES, add_feedback_arguments, config_from_arguments, feedback_metadata
from experiments import (ROOT, COLUMNS, sources, runtime, compatible_manifest,
                         run_jobs, load_samples, write_json, sha)
from statistics_tools import (quantile, quantile_interval, paired_gain,
                              binomial_interval, paired_outage_reduction)

ERASURES = (.014, .3, .5, .59, .7, .85)
# Wide enough that no architecture's choice sits at a grid edge: half duplex pays real
# time per report and prefers few, large reports; dedicated feedback prefers small ones.
WINDOWS = (4, 8, 12, 16, 32, 64, 128, 256, 512)
THRESHOLDS = (.25, .4, .5, .6)
KIND = 'global_window_state_threshold_correlation_aware'
REFERENCE = 'pooling'


def candidate_grid(windows=WINDOWS, thresholds=THRESHOLDS):
    windows, thresholds = sorted(set(windows)), sorted(set(thresholds))
    if not windows or not set(windows) <= set(WINDOWS):
        raise ValueError(f'W must be drawn from {set(WINDOWS)}')
    if not thresholds or not set(thresholds) <= set(THRESHOLDS):
        raise ValueError('tau must be drawn from {0.25,0.4,0.5,0.6}; tau=0 is ablation only')
    return {f'w{w}_tau{t:g}': {'window': w, 'state_threshold': float(t)}
            for w in windows for t in thresholds}


def policy_config(cfg, policy):
    return replace(cfg, window=policy['window'], state_threshold=policy['state_threshold'])


def select_global(training, candidates):
    """Equal-scenario mean of T99 normalized by same-scenario pooling T99."""
    expected = {f'{e:g}' for e in ERASURES}
    if set(training) != expected:
        raise ValueError('Global selection needs all six scenarios')
    for point in training.values():
        if set(point) != set(candidates) | {REFERENCE}:
            raise ValueError('Different candidates across scenarios')
        ref = point[REFERENCE]['t99']
        if ref is None or not math.isfinite(ref) or ref <= 0:
            raise ValueError('Reference pooling training T99 must be finite and positive')
    scores = {}
    for key in candidates:
        ratios = [math.inf if training[f'{e:g}'][key]['t99'] is None
                  else training[f'{e:g}'][key]['t99']/training[f'{e:g}'][REFERENCE]['t99']
                  for e in ERASURES]
        if any(math.isnan(r) or r <= 0 for r in ratios):
            raise ValueError('Invalid candidate T99')
        scores[key] = {'objective': float(np.mean(ratios)),
                       'worst_normalized_t99': float(max(ratios))}
    # Exact ties: larger W (fewer reports), then smaller tau.
    key = min(candidates, key=lambda k: (scores[k]['objective'], -candidates[k]['window'],
                                         candidates[k]['state_threshold']))
    if not math.isfinite(scores[key]['objective']):
        raise ValueError('No finite global candidate')
    return dict(candidates[key]), scores


def read_optimization(path, architecture=None):
    """Reject stale, edited or wrong-architecture optimization contracts."""
    path = Path(path)
    result = json.loads(path.read_text())
    design = result['manifest']
    if design['kind'] != KIND or design['source_hashes'] != sources():
        raise ValueError('Need a current correlation-aware optimization from these sources')
    if architecture is not None and design['architecture'] != architecture:
        raise ValueError(f'Optimization belongs to {design["architecture"]}, not {architecture}')
    if sha(path.parent/'manifest.json') != result['manifest_sha256']:
        raise ValueError('Optimization manifest hash mismatch')
    if json.loads((path.parent/'manifest.json').read_text()) != design:
        raise ValueError('Optimization and saved manifest disagree')
    if design['erasure_points'] != list(ERASURES):
        raise ValueError('All six predeclared conditions are required')
    expected = candidate_grid(design['windows'], design['thresholds'])
    if design['candidates'] != expected:
        raise ValueError('Candidate grid does not match the declared design')
    selected, _ = select_global(result['training'], expected)
    if result['selected_policy'] != selected:
        raise ValueError('Stored policy does not match global training selection')
    return result


def comparison_statistics(arrays, deadline, reps):
    selected = arrays['selected'][:,2]
    report = {'selected_t99': quantile(selected), 'selected_t99_ci95': quantile_interval(selected),
              'selected_failures': int(np.sum(~np.isfinite(selected))),
              'frozen_outage_deadline_slots': deadline, 'methods': {}, 'comparisons': {}}
    for key, rows in arrays.items():
        times = rows[:,2]
        misses = int(np.sum(times > deadline))
        report['methods'][key] = {
            't99': quantile(times), 't99_ci95': quantile_interval(times),
            'mean_completion_slots': float(times.mean()),
            'failures': int(np.sum(~np.isfinite(times))),
            'outage_at_frozen_deadline': misses/len(times),
            'outage_ci95': binomial_interval(misses, len(times)),
            'mean_reports': float(rows[:,10].mean()),
            **{f'mean_{COLUMNS[i]}': float(rows[:,i].mean()) for i in range(13,len(COLUMNS))},
            'received_repairs_without_recovery_percent': 100*float(rows[:,15].sum()/rows[:,14].sum())
                if rows[:,14].sum() else None}
        if key != 'selected':
            gain, ci = paired_gain(times, selected, reps, seed=401)
            outage_gain, outage_ci = paired_outage_reduction(times, selected, deadline)
            report['comparisons'][key] = {
                'reference_t99': quantile(times), 'gain_percent': gain, 'gain_ci95': ci,
                'outage_reduction_percentage_points': outage_gain,
                'outage_reduction_ci95': outage_ci}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--architecture', choices=sorted(ARCHITECTURES), required=True)
    p.add_argument('--n-train', type=int, default=2000)
    p.add_argument('--n-validation', type=int, default=4000)
    p.add_argument('--windows', type=int, nargs='+', default=WINDOWS)
    p.add_argument('--thresholds', type=float, nargs='+', default=THRESHOLDS)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--chunk-size', type=int, default=50)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=1210000000)
    p.add_argument('--output', type=Path, required=True)
    add_feedback_arguments(p)
    args = p.parse_args()
    if min(args.n_train, args.n_validation, args.workers, args.chunk_size, args.bootstrap) < 1 or args.seed < 0:
        p.error('Sizes must be positive and seed nonnegative')
    try:
        candidates = candidate_grid(args.windows, args.thresholds)
        cfg = config_from_arguments(args, args.architecture)
        for policy in candidates.values():
            policy_config(cfg, policy).validate()
    except ValueError as exc:
        p.error(str(exc))
    # The same training/validation blocks are used for every architecture.
    variants = [(REFERENCE, 'pooling', {})] + [
        (key, 'bitmap', dict(c)) for key, c in candidates.items()]
    windows = {key: (c.get('window', cfg.window)) for key, _, c in variants}
    spacing = max(args.n_train, args.n_validation) + 1000
    val_seed = args.seed + len(ERASURES)*spacing
    manifest = {
        'kind': KIND, 'architecture': args.architecture, 'columns': COLUMNS, 'config': asdict(cfg),
        'candidates': candidates, 'windows': sorted(set(args.windows)),
        'thresholds': sorted(set(args.thresholds)), 'erasure_points': ERASURES,
        'n_train': args.n_train, 'n_validation': args.n_validation, 'seed': args.seed,
        'spacing': spacing, 'validation_seed': val_seed, 'bootstrap': args.bootstrap,
        'interval_methods': {'t99_gain':'paired block bootstrap, seed 401',
                             'outage_probability':'exact 95% binomial',
                             'paired_outage_reduction':'exact 97.5% discordance intervals plus union bound'},
        'objective': 'equal-scenario mean of training T99 / same-scenario pooling training T99',
        'tie_break': 'larger W, then smaller tau, only on exact objective ties',
        'selection': 'one global training-only (W, tau) per architecture; validation/test never retune it',
        'controller': ('V1 unreported repairs and RF launches stay in prediction; V3 unreported FSO '
                       'launches predicted only if latest q_hat_F >= tau; M bad reported state makes '
                       'every RF opportunity a targeted repair; stagnation repair unchanged'),
        'source_hashes': sources(), 'runtime': runtime(),
        'pmf': {'degrees': DEGREES.tolist(), 'probabilities': PMF.tolist()},
        'channel': [metadata(replace(cfg, eps_fso=e)) for e in ERASURES],
        'feedback': feedback_metadata(cfg),
    }
    out = args.output.resolve()
    compatible_manifest(out/'manifest.json', manifest)
    train = [(f'train_{e:g}', replace(cfg, eps_fso=e), args.seed+i*spacing,
              args.n_train, variants) for i, e in enumerate(ERASURES)]
    run_jobs(train, out, args.workers, args.chunk_size)
    training = {}
    for cell, c, first, n, spec in train:
        a = load_samples(out, cell, first, n, [v[0] for v in spec], windows)
        training[f'{c.eps_fso:g}'] = {
            key: {'t99': quantile(x[:,2]), 'ci95': quantile_interval(x[:,2]),
                  'mean': float(x[:,2].mean()), 'failures': int(np.sum(~np.isfinite(x[:,2])))}
            for key,x in a.items()}
    selected, scores = select_global(training, candidates)
    w, tau = selected['window'], selected['state_threshold']
    validation_variants = [
        ('selected', 'bitmap', {'window': w, 'state_threshold': tau}),
        ('no_state_repair', 'bitmap', {'window': w, 'state_threshold': tau, 'state_repair': False}),
        ('no_state', 'bitmap', {'window': w, 'state_threshold': 0.}),
        ('pooling', 'pooling', {})]
    vwindows = {key: o.get('window', cfg.window) for key, _, o in validation_variants}
    validation = [(f'validation_{e:g}', replace(cfg, eps_fso=e), val_seed+i*spacing,
                   args.n_validation, validation_variants) for i,e in enumerate(ERASURES)]
    run_jobs(validation, out, args.workers, args.chunk_size)
    validated = {}
    for cell, c, first, n, spec in validation:
        a = load_samples(out, cell, first, n, [v[0] for v in spec], vwindows)
        deadline = training[f'{c.eps_fso:g}'][REFERENCE]['t99']
        validated[f'{c.eps_fso:g}'] = comparison_statistics(a, deadline, args.bootstrap)
    result = {
        'manifest': manifest, 'selected_policy': selected,
        'global_training_scores': scores, 'training': training, 'validation': validated,
        'manifest_sha256': sha(out/'manifest.json'),
        'raw_samples_sha256': {p.relative_to(out).as_posix(): sha(p)
                              for p in sorted((out/'samples').rglob('*.npz'))},
        'claim_scope': 'empirical global grid minimum, not per-channel or population optimal; negative validation retained'}
    write_json(out/'optimization.json', result)
    print(f'[{args.architecture}] single frozen policy:', selected, flush=True)
    print('Optimization:', out/'optimization.json', flush=True)


if __name__ == '__main__':
    main()
