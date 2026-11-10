"""Replace the real feedback curves of a finished six-curve campaign on identical blocks.

The new real half-duplex and real dedicated curves (report loss; half-duplex turnaround) are
retrained separately and simulated on exactly the campaign's blocks: same cells, seeds,
channel/code configuration and sample sizes. The kept curves (pooling, open loop, ideal
feedback) are read from the campaign after checking every sample file against the SHA-256 the
campaign recorded, AND after the current code has re-simulated a set of campaign blocks and
reproduced every stored row exactly. The ideal half-duplex curve is dropped: with free
feedback it equals the ideal dedicated curve.
"""
import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import numpy as np
from channel import trial_inputs
from coding import Config
from experiments import (COLUMNS, CONTROLLER_FIELDS, block_row, compatible_manifest, load_samples,
                         run_jobs, runtime, sha, sources, variant_config, write_json)
from feedback_options import feedback_metadata
from model import simulate
from optimize_window import read_optimization
from plot_results import curve_statistics, draw_figures, figure_data

KIND = 'replacement_curve_evaluation'
KEPT = ('pooling', 'open_loop', 'ded_ideal')
NEW = {'hd_lossy': 'hd_real', 'ded_lossy': 'ded_real'}   # new curve -> trained architecture
CURVES = ('pooling', 'open_loop', 'hd_lossy', 'ded_lossy', 'ded_ideal')


def verified_campaign(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'manifest.json').read_text())
    completed = json.loads((folder/'completed.json').read_text())
    if manifest['kind'] != 'six_curve_correlation_aware_evaluation':
        raise ValueError('Campaign folder is not a finished six-curve evaluation')
    if sha(folder/'manifest.json') != completed['manifest_sha256']:
        raise ValueError('Campaign manifest changed after completion')
    files = {p.relative_to(folder).as_posix(): p for p in sorted((folder/'samples').rglob('*.npz'))}
    if set(files) != set(completed['raw_samples_sha256']):
        raise ValueError('Campaign sample files differ from completed.json')
    for key, path in files.items():
        if sha(path) != completed['raw_samples_sha256'][key]:
            raise ValueError(f'Campaign sample {key} does not match completed.json')
    return folder, manifest


def reproduce(folder, manifest, arrays, per_cell):
    """Re-simulate stored campaign blocks with the current code; every row must match exactly."""
    variants = {v[0]: v for v in manifest['variants']}
    checked = 0
    for name, spec in manifest['cells'].items():
        cfg = Config(**spec['config'])
        for seed in range(spec['seed0'], spec['seed0'] + min(per_cell, spec['n'])):
            bank = trial_inputs(cfg, seed)
            for key in variants:
                row = np.asarray(block_row(simulate(variant_config(cfg, variants[key]),
                                                    variants[key][1], seed, bank=bank)), dtype=float)
                stored = arrays[name][key][seed - spec['seed0']]
                if not np.array_equal(row, stored, equal_nan=True):
                    raise ValueError(f'Current code does not reproduce campaign block {name}/{key}/{seed}')
                checked += 1
    return checked


def feedback_resource(head, cfg, curves):
    """Per-block feedback resource use at the outage point, from the recorded rows only."""
    col = {name: i for i, name in enumerate(COLUMNS)}
    source_bits = cfg.k * cfg.payload_bits
    resource = {}
    for key in curves:
        rows = head[key]
        mean = lambda name: float(rows[:, col[name]].mean())
        resource[key] = {
            'reports_started_per_block': mean('reports_launched'),
            'reports_delivered_per_block': mean('reports_completed'),
            'frame_bits_per_block': mean('feedback_frame_bits'),
            'feedback_bits_per_source_bit': mean('feedback_frame_bits') / source_bits,
            'feedback_airtime_per_block_us': 1e6*mean('bitmap_airtime_observed_s'),
            'feedback_link_utilization': float(np.mean(rows[:, col['bitmap_airtime_observed_s']]
                                                       / rows[:, col['observation_s']])),
            'rf_data_silenced_per_block_us': 1e6*mean('rf_forward_silenced_s'),
            'mean_information_age_us': 1e6*mean('feedback_mean_state_age_s')}
    return resource


def new_variant(key, selection):
    """Controller and feedback fields come only from the frozen training configuration."""
    trained = selection['manifest']['config']
    overrides = {name: trained[name] for name in sorted(CONTROLLER_FIELDS)}
    overrides.update(window=selection['selected_policy']['window'],
                     state_threshold=selection['selected_policy']['state_threshold'])
    return (key, 'bitmap', overrides)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True, help='finished six-curve publication folder')
    p.add_argument('--opt-hd-lossy', type=Path, required=True, help='optimization.json, real half duplex')
    p.add_argument('--opt-ded-lossy', type=Path, required=True, help='optimization.json, real dedicated')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--chunk-size', type=int, default=500)
    p.add_argument('--reproduce-blocks', type=int, default=20,
                   help='campaign blocks per cell re-simulated for every campaign curve')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if min(args.workers, args.chunk_size, args.reproduce_blocks) < 1:
        p.error('Sizes must be positive')
    try:
        campaign, cm = verified_campaign(args.campaign)
        paths = {'hd_lossy': args.opt_hd_lossy, 'ded_lossy': args.opt_ded_lossy}
        selections = {k: read_optimization(paths[k], arch) for k, arch in NEW.items()}
    except ValueError as exc:
        p.error(str(exc))
    shared = [f.name for f in fields(Config) if f.name not in CONTROLLER_FIELDS and f.name != 'eps_fso']
    base = cm['cells']['outage']['config']
    for key, selection in selections.items():
        trained = selection['manifest']['config']
        if not trained['feedback_loss_probability'] > 0:
            p.error(f'{key}: the replacement training must use a positive report loss probability')
        if any(trained[k] != base[k] for k in shared):
            p.error(f'{key}: training used different channel/code/physics settings than the campaign')
        d = selection['manifest']
        first, last = d['seed'], d['validation_seed'] + 5*d['spacing'] + d['n_validation']
        for spec in cm['cells'].values():
            if not (spec['seed0'] + spec['n'] <= first or spec['seed0'] >= last):
                p.error(f'{key}: campaign test seeds overlap its training/validation seeds')
    variants = [new_variant(key, selections[key]) for key in NEW]
    windows = {v[0]: v[2].get('window', 0) for v in cm['variants']}
    old = {name: load_samples(campaign, name, spec['seed0'], spec['n'], tuple(windows), windows)
           for name, spec in cm['cells'].items()}
    checked = reproduce(campaign, cm, old, args.reproduce_blocks)
    print(f'Current code reproduced {checked} stored campaign rows exactly', flush=True)
    out = args.output.resolve()
    cells = [(name, Config(**spec['config']), spec['seed0'], spec['n'], variants)
             for name, spec in cm['cells'].items()]
    feedback = {v[0]: feedback_metadata(variant_config(cells[0][1], v)) for v in variants}
    manifest = {'kind': KIND, 'curves': CURVES, 'kept_from_campaign': KEPT, 'new_curves': list(NEW),
                'columns': COLUMNS, 'variants': [list(v) for v in variants],
                'cells': {name: {'config': asdict(c), 'seed0': s, 'n': n, 'channel': cm['cells'][name]['channel']}
                          for name, c, s, n, _ in cells},
                'campaign_manifest_sha256': sha(campaign/'manifest.json'),
                'campaign_completed_sha256': sha(campaign/'completed.json'),
                'campaign_source_hashes': cm['source_hashes'],
                'reproduced_campaign_rows': checked,
                'source_hashes': sources(), 'runtime': runtime(),
                'optimization_sha256': {k: sha(paths[k]) for k in NEW},
                'policies': {k: selections[k]['selected_policy'] for k in NEW},
                'validation': {k: selections[k]['validation'] for k in NEW},
                'feedback': feedback, 'bootstrap': cm['bootstrap'], 'bootstrap_seed': 701,
                'endpoint': 'receiver decoding completion'}
    compatible_manifest(out/'manifest.json', manifest)
    run_jobs(cells, out, args.workers, args.chunk_size)
    for v in variants:
        write_json(out/'traces'/f'{v[0]}.json',
                   simulate(variant_config(cells[0][1], v), 'bitmap', cells[0][2], record_trace=True))
    new_windows = {v[0]: v[2]['window'] for v in variants}
    arrays = {}
    for name, c, s, n, _ in cells:
        new = load_samples(out, name, s, n, tuple(new_windows), new_windows)
        arrays[name] = {**{k: old[name][k] for k in KEPT}, **new}
        if not all(np.array_equal(arrays[name][k][:, 0], arrays[name]['pooling'][:, 0]) for k in CURVES):
            raise RuntimeError('Replacement and campaign blocks are not paired')
    sweep = sorted((c for c in arrays if c.startswith('eps_')), key=lambda c: cm['cells'][c]['config']['eps_fso'])
    eps = [cm['cells'][c]['config']['eps_fso'] for c in sweep]
    n = cm['cells']['outage']['n']
    times, shown, q = figure_data(out, arrays['outage'], [arrays[c] for c in sweep], eps, n, CURVES)
    draw_figures(out, times, shown, eps, q, n, CURVES)
    stats, sweep_stats = curve_statistics(arrays['outage'], [(c, arrays[c], cm['cells'][c]['channel']) for c in sweep],
                                          CURVES, cm['bootstrap'])
    campaign_summary = json.loads((campaign/'summary.json').read_text())
    summary = {'endpoint': 'receiver decoding completion',
               'policies': {'ded_ideal': campaign_summary['policies']['ded_ideal'], **manifest['policies']},
               'slot_s': campaign_summary['slot_s'], 'n_outage': n,
               'n_sweep': min(cm['cells'][c]['n'] for c in sweep), 'preliminary': False,
               'feedback': {'ded_ideal': cm['feedback']['ded_ideal'], **feedback},
               'tail_display_minimum_misses': 10, 'tail_display_floor': 10/n,
               'curves': stats, 'sweep': sweep_stats, 'replacement_validation': manifest['validation'],
               'feedback_resource': feedback_resource(arrays['outage'], Config(**base),
                                                      ('hd_lossy', 'ded_lossy', 'ded_ideal')),
               'reproduced_campaign_rows': checked}
    write_json(out/'summary.json', summary)
    write_json(out/'completed.json', {'status': 'complete', 'manifest_sha256': sha(out/'manifest.json'),
        'raw_samples_sha256': {q.relative_to(out).as_posix(): sha(q) for q in sorted((out/'samples').rglob('*.npz'))}})
    print('Finished:', out, flush=True)


if __name__ == '__main__':
    main()
