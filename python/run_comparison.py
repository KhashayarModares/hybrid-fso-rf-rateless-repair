"""Six curves on identical fresh blocks: two baselines and four frozen feedback architectures."""
import argparse
from dataclasses import asdict, fields, replace
from pathlib import Path
from channel import metadata
from coding import Config, DEGREES, PMF
from model import simulate
from feedback_options import ARCHITECTURES, feedback_metadata
from experiments import (ROOT, COLUMNS, CONTROLLER_FIELDS, sources, runtime, compatible_manifest,
                         run_jobs, write_json, sha, variant_config)
from optimize_window import ERASURES, read_optimization

CURVES = ('pooling', 'open_loop', 'hd_real', 'hd_ideal', 'ded_real', 'ded_ideal')
KIND = 'six_curve_correlation_aware_evaluation'


def curve_variants(selections):
    variants = [('pooling', 'pooling', {}), ('open_loop', 'open_loop', {})]
    for arch in CURVES[2:]:
        policy = selections[arch]['selected_policy']
        variants.append((arch, 'bitmap', {**ARCHITECTURES[arch], 'window': policy['window'],
                                          'state_threshold': policy['state_threshold']}))
    return variants


def common_config(selections):
    """All four trainings must share every non-controller (channel/code/physics) field."""
    configs = [selections[a]['manifest']['config'] for a in CURVES[2:]]
    shared = [f.name for f in fields(Config) if f.name not in CONTROLLER_FIELDS]
    for c in configs[1:]:
        if any(c[name] != configs[0][name] for name in shared):
            raise ValueError('The four optimizations used different channel/code/physics settings')
    base = dict(configs[CURVES[2:].index('ded_real')])
    return Config(**base)


def evaluation_cells(selections, nmc, nmc_outage, seed):
    """Only channel mean and seed vary; each architecture keeps its frozen (W, tau)."""
    cfg = common_config(selections)
    variants = curve_variants(selections)
    spacing = max(nmc, nmc_outage) + 1000
    requests = [('outage', .59, nmc_outage), *[(f'eps_{e:g}', e, nmc) for e in ERASURES]]
    last_test = seed + len(requests)*spacing
    for arch in CURVES[2:]:
        d = selections[arch]['manifest']
        first = d['seed']
        last = d['validation_seed'] + (len(ERASURES)-1)*d['spacing'] + d['n_validation']
        if not (last_test <= first or seed >= last):
            raise ValueError(f'Test seeds overlap {arch} training/validation seeds')
    return [(name, replace(cfg, eps_fso=e), seed+i*spacing, n, variants)
            for i, (name, e, n) in enumerate(requests)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arch in CURVES[2:]:
        p.add_argument(f'--opt-{arch.replace("_", "-")}', type=Path, required=True,
                       help=f'optimization.json of the {arch} training')
    p.add_argument('--nmc', type=int, default=10000)
    p.add_argument('--nmc-outage', type=int, default=1000000)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--chunk-size', type=int, default=500)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=1300000000)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-pilot', action='store_true',
                   help='Allow smaller, explicitly preliminary figure-evaluation samples')
    args = p.parse_args()
    if min(args.nmc, args.nmc_outage, args.workers, args.chunk_size, args.bootstrap) < 1 or args.seed < 0:
        p.error('Sizes must be positive')
    if (args.nmc < 10000 or args.nmc_outage < 1000000) and not args.allow_pilot:
        p.error('Use --allow-pilot for smaller preliminary evaluation samples')
    try:
        selections = {arch: read_optimization(getattr(args, f'opt_{arch}'), arch) for arch in CURVES[2:]}
        cells = evaluation_cells(selections, args.nmc, args.nmc_outage, args.seed)
    except ValueError as exc:
        p.error(str(exc))
    out = args.output.resolve()
    variants = cells[0][4]
    manifest = {'kind': KIND, 'curves': CURVES, 'columns': COLUMNS,
        'variants': [list(v) for v in variants],
        'cells': {name: {'config': asdict(c), 'seed0': seed, 'n': n, 'channel': metadata(c)}
                  for name, c, seed, n, _ in cells},
        'source_hashes': sources(), 'runtime': runtime(),
        'optimization_sha256': {a: sha(getattr(args, f'opt_{a}')) for a in CURVES[2:]},
        'policies': {a: selections[a]['selected_policy'] for a in CURVES[2:]},
        'validation': {a: selections[a]['validation'] for a in CURVES[2:]},
        'feedback': {a: feedback_metadata(variant_config(cells[0][1], v))
                     for a, v in zip(CURVES, variants) if a in ARCHITECTURES},
        'bootstrap': args.bootstrap, 'bootstrap_seed': 701,
        'pmf': {'degrees': DEGREES.tolist(), 'probabilities': PMF.tolist()},
        'sampling': 'same immutable Markov path on physical time; independent per-link ordinal noise; paired public graphs',
        'endpoint': 'receiver decoding completion',
        'publication_minimums': {'outage': 1000000, 'sweep': 10000},
        'paper_mode': 'preliminary' if args.allow_pilot else 'publication'}
    compatible_manifest(out/'manifest.json', manifest)
    run_jobs(cells, out, args.workers, args.chunk_size)
    for v in variants:
        write_json(out/'traces'/f'{v[0]}.json',
                   simulate(variant_config(cells[0][1], v), v[1], args.seed, record_trace=True))
    from plot_results import generate
    generate(out, allow_pilot=args.allow_pilot)
    write_json(out/'completed.json', {'status': 'complete', 'manifest_sha256': sha(out/'manifest.json'),
        'raw_samples_sha256': {p.relative_to(out).as_posix(): sha(p) for p in sorted((out/'samples').rglob('*.npz'))}})
    print('Finished:', out, flush=True)


if __name__ == '__main__':
    main()
