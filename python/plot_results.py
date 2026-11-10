"""Two six-curve figures and honest statistics for the frozen per-architecture policies."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coding import Config
from experiments import COLUMNS, sources, load_samples, write_json
from statistics_tools import quantile, quantile_interval, paired_gain, binomial_interval

CURVES = ('pooling', 'open_loop', 'hd_real', 'hd_ideal', 'ded_real', 'ded_ideal')
# Paper convention: black/red baselines, blue proposed dedicated, green dashed ideal.
LABELS = ('Soft switching', 'Open-loop assignment', 'Proposed (half-duplex)',
          'Proposed (half-duplex, ideal)', 'Proposed (dedicated)',
          'Proposed (dedicated, ideal)')
COLORS = ('black', '#cc0000', '#e08000', '#8a2be2', '#0000cc', '#008000')
MARKERS = ('o', 's', 'v', 'p', '^', 'd')
STYLES = ('-', '-', '-', '--', '-', '--')


def paper_axes(ax, log_y=False):
    ax.tick_params(direction='in', which='both', top=True, right=True)
    ax.grid(True, which='major', linestyle=':', linewidth=.6, color='0.7')
    if log_y:
        ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=.35, color='0.85')
    ax.set_axisbelow(True)


def legend_overlap(ax, legend):
    """Display-space length of drawn curve segments that pass under the legend box."""
    ax.figure.canvas.draw()
    box = legend.get_window_extent()
    hidden = 0.
    for line in ax.get_lines():
        xy = line.get_xydata()
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if line.get_drawstyle().startswith('steps') and len(xy) > 1:
            # Expand post-steps: horizontal then vertical moves between data points.
            corner = np.column_stack([xy[1:, 0], xy[:-1, 1]])
            xy = np.insert(xy, np.arange(1, len(xy)), corner, axis=0)
        if len(xy) < 2:
            continue
        pts = ax.transData.transform(xy)
        for a, b in zip(pts[:-1], pts[1:]):
            # Sample each drawn segment every display pixel.
            k = max(2, int(np.hypot(*(b-a))))
            seg = a + np.linspace(0, 1, k)[:, None]*(b-a)
            inside = ((seg[:, 0] >= box.x0) & (seg[:, 0] <= box.x1) &
                      (seg[:, 1] >= box.y0) & (seg[:, 1] <= box.y1))
            hidden += inside.sum()
    return hidden


LEGEND_LAYOUTS = (
    {},  # matplotlib defaults, as in the paper figures
    {'labelspacing': .25, 'borderpad': .3, 'handlelength': 1.6, 'handletextpad': .4,
     'borderaxespad': .3},  # same text, tighter box
)


def place_legend(ax, candidates):
    """Use the first position/layout whose box covers no drawn curve."""
    best = None
    for layout in LEGEND_LAYOUTS:
        for loc in candidates:
            legend = ax.legend(loc=loc, framealpha=1, fancybox=False, edgecolor='black', **layout)
            hidden = legend_overlap(ax, legend)
            if hidden == 0:
                return loc
            if best is None or hidden < best[1]:
                best = (loc, hidden, layout)
    ax.legend(loc=best[0], framealpha=1, fancybox=False, edgecolor='black', **best[2])
    print(f'Warning: legend covers {best[1]} curve pixels at {best[0]}', flush=True)
    return best[0]


STYLE = {key: (label, color, marker, style) for key, label, color, marker, style in
         zip(CURVES, LABELS, COLORS, MARKERS, STYLES)}
# Real dedicated feedback with report loss and processing replaces the lossless curve.
STYLE['ded_lossy'] = STYLE['ded_real']
STYLE['hd_lossy'] = STYLE['hd_real']


def curve_labels(curves):
    labels = {m: STYLE[m][0] for m in curves}
    if 'ded_ideal' in curves and 'hd_ideal' not in curves:
        # Free feedback is architecture-independent: one ideal reference curve.
        labels['ded_ideal'] = 'Proposed (ideal)'
    return labels


def drop_to_floor(shown, floor):
    """Drawing only: show the step at which the outage falls below the ten-miss floor.

    Values with fewer than ten misses stay hidden. The first hidden point is the deadline at
    which the empirical outage drops below the floor; plotting it below the axis limit draws
    that final vertical step down to the floor instead of ending the staircase in mid-air.
    """
    y = np.array(shown, dtype=float)
    finite = np.flatnonzero(np.isfinite(y))
    if len(finite) and finite[-1] + 1 < len(y):
        y[finite[-1] + 1] = floor / 10
    return y


def draw_figures(out, times, shown, eps, q, n, curves=CURVES, size=(4.2, 2.8)):
    """size: figure width and height in inches (3.5 in is one IEEE column, drawn without scaling)."""
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
                         'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7})
    labels = curve_labels(curves)
    fig, ax = plt.subplots(figsize=size, layout='constrained')
    for m in curves:
        _, color, mark, style = STYLE[m]
        ax.step(times, drop_to_floor(shown[m], 10/n), where='post', color=color, linestyle=style,
                marker=mark, markevery=max(1, len(times)//12), markersize=3, markerfacecolor='none',
                label=labels[m])
    ax.set(xlabel=r'Deadline $\tau$ (slots)', ylabel=r'$P_{\mathrm{out}}(\tau)$',
           yscale='log', ylim=(10/n, 1.05))
    paper_axes(ax, log_y=True)
    loc = place_legend(ax, ('upper right', 'center right', 'lower right', 'lower left', 'center'))
    for extension in ('png', 'pdf'):
        fig.savefig(out/f'figure2_outage.{extension}', dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=size, layout='constrained')
    for m in curves:
        _, color, mark, style = STYLE[m]
        ax.plot(eps, q[m], color=color, linestyle=style, marker=mark, markersize=4,
                markerfacecolor='none', label=labels[m])
    ax.set(xlabel=r'Stationary FSO erasure probability $\bar\varepsilon_{\mathrm{F}}$',
           ylabel=r'$T_{99}$ (slots)', xlim=(0, .9))
    paper_axes(ax)
    loc3 = place_legend(ax, ('upper left', 'center left', 'lower right'))
    for extension in ('png', 'pdf'):
        fig.savefig(out/f'figure3_deadline.{extension}', dpi=300)
    plt.close(fig)
    print(f'Legends: outage {loc}, deadline {loc3}', flush=True)


def redraw(out):
    """Figures only, from the run's own plotted data; statistics are not recomputed."""
    out = Path(out).resolve()
    manifest = json.loads((out/'manifest.json').read_text())
    with open(out/'figure_data'/'fig2_outage.dat') as stream:
        header = stream.readline().split()
    curves = tuple(header[1:])
    if header[0] != 'tau' or not curves or set(curves) - set(STYLE):
        raise ValueError('Unexpected figure data columns')
    fig2 = np.loadtxt(out/'figure_data'/'fig2_outage.dat', skiprows=1)
    fig3 = np.loadtxt(out/'figure_data'/'fig3_deadline.dat', skiprows=1)
    shown = {m: fig2[:, i+1] for i, m in enumerate(curves)}
    q = {m: list(fig3[:, i+1]) for i, m in enumerate(curves)}
    draw_figures(out, fig2[:, 0], shown, list(fig3[:, 0]), q, manifest['cells']['outage']['n'], curves)


def figure_data(out, head, sweep, eps, n, curves):
    """Write the exact plotted data; the ten-miss rule only affects what is drawn."""
    data = out/'figure_data'
    data.mkdir(exist_ok=True)
    series = {m: head[m][:,2] for m in curves}
    times = np.unique(np.r_[0., *[t[np.isfinite(t)] for t in series.values()]])
    counts = {m: len(t)-np.searchsorted(np.sort(t), times, side='right') for m, t in series.items()}
    # Display threshold only: all raw outcomes, including failures, remain in statistics.
    shown = {m: np.where(counts[m] >= 10, counts[m]/n, np.nan) for m in curves}
    np.savetxt(data/'fig2_outage.dat', np.column_stack([times]+[shown[m] for m in curves]),
               header='tau '+' '.join(curves), comments='', fmt='%.15g')
    q = {m: [quantile(a[m][:,2]) for a in sweep] for m in curves}
    np.savetxt(data/'fig3_deadline.dat', np.column_stack([eps]+[q[m] for m in curves]),
               header='eps '+' '.join(curves), comments='', fmt='%.15g')
    return times, shown, q


def curve_statistics(head, sweep_cells, curves, bootstrap):
    """Pointwise statistics of every curve, paired against pooling on the same blocks."""
    n = len(head['pooling'])
    series = {m: head[m][:,2] for m in curves}
    reference = quantile(series['pooling'])
    stats, sweep = {}, {}
    for i, m in enumerate(curves):
        t, row = series[m], head[m]
        gain, ci = paired_gain(series['pooling'], t, bootstrap, seed=701+i)
        stats[m] = {
            't99_slots': quantile(t), 't99_ci95': quantile_interval(t),
            't9999_slots': quantile(t, .9999), 't99999_slots': quantile(t, .99999),
            'mean_completion_slots': float(t.mean()), 'failures': int(np.sum(~np.isfinite(t))),
            'gain_percent': gain, 'gain_ci95': ci,
            'outage_at_pooling_t99': int(np.sum(t > reference))/n,
            'outage_ci95': binomial_interval(int(np.sum(t > reference)), n),
            'mean_reports': float(row[:,10].mean()),
            'received_repairs_without_recovery_percent': 100*float(row[:,15].sum()/row[:,14].sum())
                if row[:,14].sum() else 0.,
            **{f'mean_{COLUMNS[k]}': float(row[:,k].mean()) for k in range(12, len(COLUMNS))}}
    for i, (cell, a, channel) in enumerate(sweep_cells):
        sweep[cell] = {
            'channel': channel,
            'curves': {m: {'t99': quantile(a[m][:,2]), 't99_ci95': quantile_interval(a[m][:,2]),
                           'failures': int(np.sum(~np.isfinite(a[m][:,2]))),
                           'gain_vs_pooling': paired_gain(a['pooling'][:,2], a[m][:,2],
                                                          bootstrap, seed=801+10*i+j)}
                       for j, m in enumerate(curves)}}
    return stats, sweep


def generate(out, allow_pilot=False):
    out = Path(out).resolve()
    manifest = json.loads((out/'manifest.json').read_text())
    if manifest['source_hashes'] != sources():
        raise ValueError('Saved samples belong to different source files')
    if manifest['kind'] != 'six_curve_correlation_aware_evaluation' or tuple(manifest['curves']) != CURVES:
        raise ValueError('Only the frozen six-curve campaign belongs in these figures')
    windows = {v[0]: v[2].get('window', 0) for v in manifest['variants']}
    arrays = {name: load_samples(out, name, spec['seed0'], spec['n'], CURVES, windows)
              for name, spec in manifest['cells'].items()}
    head = arrays['outage']
    n = manifest['cells']['outage']['n']
    cells = sorted((c for c in arrays if c.startswith('eps_')),
                   key=lambda c: manifest['cells'][c]['config']['eps_fso'])
    n_sweep = min(manifest['cells'][c]['n'] for c in cells)
    preliminary = n < 1000000 or n_sweep < 10000
    if preliminary and not allow_pilot:
        raise ValueError('Use --allow-pilot to label smaller samples as preliminary')
    if n < 20:
        raise ValueError('At least 20 outage blocks are needed for the ten-miss display rule')
    eps = [manifest['cells'][c]['config']['eps_fso'] for c in cells]
    times, shown, q = figure_data(out, head, [arrays[c] for c in cells], eps, n, CURVES)
    draw_figures(out, times, shown, eps, q, n)
    stats, sweep = curve_statistics(
        head, [(c, arrays[c], manifest['cells'][c]['channel']) for c in cells], CURVES, manifest['bootstrap'])
    summary = {'endpoint': 'receiver decoding completion', 'policies': manifest['policies'],
               'slot_s': Config(**manifest['cells']['outage']['config']).slot_s,
               'n_outage': n, 'n_sweep': n_sweep, 'preliminary': preliminary,
               'feedback': manifest['feedback'], 'tail_display_minimum_misses': 10,
               'tail_display_floor': 10/n, 'curves': stats, 'sweep': sweep,
               'optimization_validation': manifest['validation']}
    write_json(out/'summary.json', summary)
    print('Created two six-curve figures; preliminary:', preliminary, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--allow-pilot', action='store_true')
    p.add_argument('--figures-only', action='store_true',
                   help='Redraw both figures from figure_data/*.dat without recomputing statistics')
    a = p.parse_args()
    if a.figures_only:
        redraw(a.output)
    else:
        generate(a.output, a.allow_pilot)
