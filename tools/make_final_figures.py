"""Draw the final five-curve publication figures from the finished runs, with proof of origin.

usage: python -B tools/make_final_figures.py <outputs folder>
           [--result publication_final] [--campaign publication] [--dest figures_final]

1. Every sample file of the replacement run and of the reused campaign is checked against the
   SHA-256 that run recorded in its completed.json; any mismatch stops the script.
2. The plotted data (outage staircase and T99 sweep) is recomputed from those verified samples
   and must equal the replacement run's own figure_data exactly.
3. Only then are the figures drawn (paper style) into <outputs>/<dest>; the run folders are only read.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
sys.path.insert(0, str(ROOT/'tools'))
from experiments import load_samples          # noqa: E402
from export_csv import verify                 # noqa: E402
from plot_results import draw_figures, figure_data  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('outputs', type=Path)
    p.add_argument('--result', default='publication_final')
    p.add_argument('--campaign', default='publication')
    p.add_argument('--dest', default='figures_final')
    p.add_argument('--size', nargs=2, type=float, default=(4.2, 2.8), metavar=('WIDTH_IN', 'HEIGHT_IN'),
                   help='figure size in inches; 3.5 in is one IEEE column, drawn without scaling')
    args = p.parse_args()
    outputs = args.outputs.resolve()
    result, campaign, dest = outputs/args.result, outputs/args.campaign, outputs/args.dest
    rm = json.loads((result/'manifest.json').read_text())
    cm = json.loads((campaign/'manifest.json').read_text())
    if rm.get('kind') != 'replacement_curve_evaluation':
        raise SystemExit(f'{result} is not a finished replacement run')
    for folder in (result, campaign):
        files, record = verify(folder)
        print(f'{folder.name}: {len(files)} sample files verified against {record}', flush=True)
    if rm['campaign_manifest_sha256'] != __import__('hashlib').sha256(
            (campaign/'manifest.json').read_bytes()).hexdigest():
        raise SystemExit('The replacement run was not made from this campaign')
    curves, kept = tuple(rm['curves']), tuple(rm['kept_from_campaign'])
    new_windows = {v[0]: v[2]['window'] for v in rm['variants']}
    old_windows = {v[0]: v[2].get('window', 0) for v in cm['variants']}
    arrays = {}
    for name, spec in rm['cells'].items():
        new = load_samples(result, name, spec['seed0'], spec['n'], tuple(new_windows), new_windows)
        old = load_samples(campaign, name, spec['seed0'], spec['n'], tuple(old_windows), old_windows)
        arrays[name] = {**{k: old[k] for k in kept}, **new}
    sweep = sorted((c for c in arrays if c.startswith('eps_')), key=lambda c: rm['cells'][c]['config']['eps_fso'])
    eps = [rm['cells'][c]['config']['eps_fso'] for c in sweep]
    n = rm['cells']['outage']['n']
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    times, shown, q = figure_data(dest, arrays['outage'], [arrays[c] for c in sweep], eps, n, curves)
    for name in ('fig2_outage.dat', 'fig3_deadline.dat'):
        mine = np.loadtxt(dest/'figure_data'/name, skiprows=1)
        theirs = np.loadtxt(result/'figure_data'/name, skiprows=1)
        if mine.shape != theirs.shape or not np.array_equal(mine, theirs, equal_nan=True):
            raise SystemExit(f'Recomputed {name} differs from the run\'s own figure data')
        print(f'{name}: recomputed from verified samples, identical to the run ({len(mine)} rows)', flush=True)
    draw_figures(dest, times, shown, eps, q, n, curves, size=tuple(args.size))
    shutil.copy2(result/'summary.json', dest/'summary.json')
    print('Final figures:', dest, flush=True)


if __name__ == '__main__':
    main()
