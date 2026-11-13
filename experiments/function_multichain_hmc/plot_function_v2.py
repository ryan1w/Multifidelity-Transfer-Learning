"""Figs. func (main text, Sec. 3.1) and func2 (Supplementary Fig. S2) redrawn for print, from the saved results only.

Same content as function_0{1,2}_4longchains.jpeg of hmc_multichain.py (= images/Function/function_0{1,2}_4chains.jpeg
of the paper): top row deterministic MFNN (prep/<scenario>.npz: mfnn_mean), bottom row HMC pooled over the four long
chains L0-L3 (chains/<scenario>_L<i>.npz: 4 x 14000 predictions at the 500 test points; posterior mean and 2.5/97.5
percentiles, as hmc_multichain.band_metrics). No network is trained and no chain is run.

Changes for legibility: the figure is drawn at its printed size (0.9 \\linewidth = 5.52 in; letterpaper, 3 cm
margins) so that font sizes are the printed sizes (all text >= 7 pt; a subscript is 0.7 of its label, so the y label
is 10.5 pt); one legend per row above the panels instead of legends inside them (they covered data); the y axis is
shared within each row; axis labels x and y_HF added; data points drawn above the lines; row label "HMC" instead of
"HMC (4 long chains)". Colors, line styles and the 2 x 3 layout (rows deterministic/HMC, columns (a)-(c)) are kept.

It also recomputes the numbers quoted in main.tex Sec. 3.1 (RMSE, 95% coverage, mean width on the 500-point grid)
from the same arrays and writes them to figure_v2_metrics.csv.

Usage: python plot_function_v2.py [--outdir DIR]   (default: figures/ next to this script)
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(HERE, 'prep')
CHAINS = os.path.join(HERE, 'chains')
LONG_TAGS = ['L0', 'L1', 'L2', 'L3']
# Composite figures as example_mf_bnn.FIGURES (columns (a)-(c) in caption order), with the _v2 output names
FIGURES = {'function_01_4chains_v2.jpeg': ['fig1a', 'fig1b', 'fig1c'],
           'function_02_4chains_v2.jpeg': ['fig2a', 'fig2b', 'fig2c']}
# Values quoted in main.tex Sec. 3.1 (\label{sec:nonlinear}); widths are quoted for (a) and (c) only
PAPER_VALUES = {
    'fig1a': {'det_rmse': 0.095, 'hmc_rmse': 0.024, 'coverage_pct': 99.8, 'mean_width': 0.25},
    'fig1b': {'det_rmse': 0.085, 'hmc_rmse': 0.042, 'coverage_pct': 97.4},
    'fig1c': {'det_rmse': 0.126, 'hmc_rmse': 0.063, 'coverage_pct': 99.6, 'mean_width': 0.45},
    'fig2a': {'det_rmse': 0.142, 'hmc_rmse': 0.058, 'coverage_pct': 93.6},
    'fig2b': {'det_rmse': 0.254, 'hmc_rmse': 0.099, 'coverage_pct': 99.2},
    'fig2c': {'det_rmse': 0.251, 'hmc_rmse': 0.145, 'coverage_pct': 99.0},
}

# Printed size: \linewidth = 443.58 pt (8.5 in - 2 x 3 cm) = 6.138 in; Figs. func/func2 use 0.9 \linewidth
TEXT_WIDTH_IN = (8.5 * 72.27 - 2 * 3 / 2.54 * 72.27) / 72.27
FIG_W = 0.9 * TEXT_WIDTH_IN
FS_TICK, FS_LEGEND, FS_XLABEL, FS_YLABEL, FS_ROW, FS_PANEL = 8, 7.5, 9, 10.5, 9, 9.5
DPI = 600

ROW_LABELS = ['Deterministic', 'HMC']
LEGEND_LABELS = {'data': 'High-fidelity data', 'true': 'True function', 'pred': 'Prediction',
                 'mean': 'Posterior mean', 'band': '95% credible interval'}


def load_scenario(scn):
    with np.load(os.path.join(PREP, f'{scn}.npz')) as d:
        d = {k: d[k] for k in d.files}
    preds = np.stack([np.load(os.path.join(CHAINS, f'{scn}_{t}.npz'))['preds'] for t in LONG_TAGS], axis=1)
    pooled = preds.reshape(-1, preds.shape[-1])            # [4 x 14000, 500], as hmc_multichain.pool_summary
    lo, hi = np.percentile(pooled, [2.5, 97.5], axis=0)
    d.update(hmc_mean=pooled.mean(axis=0), lo=lo, hi=hi, n_draws=pooled.shape[0], n_chains=preds.shape[1])
    return d


def metrics(d):
    y = d['y_true'].ravel()
    return {'det_rmse': float(np.sqrt(np.mean((d['mfnn_mean'] - y) ** 2))),
            'hmc_rmse': float(np.sqrt(np.mean((d['hmc_mean'] - y) ** 2))),
            'coverage_pct': 100 * float(np.mean((d['lo'] <= y) & (y <= d['hi']))),
            'mean_width': float(np.mean(d['hi'] - d['lo']))}


def draw_panel(ax, d, row):
    x = d['x_test'].ravel()
    if row == 1:
        ax.fill_between(x, d['lo'], d['hi'], color='red', alpha=0.3, lw=0, zorder=1)
    ax.plot(x, d['y_true'].ravel(), ls='--', color='green', lw=1.0, dashes=(3.2, 1.8), zorder=2)
    ax.plot(x, d['mfnn_mean'] if row == 0 else d['hmc_mean'], color='red', lw=1.1, zorder=3)
    ax.scatter(d['x_hf'].ravel(), d['y_hf'].ravel(), color='black', s=9, lw=0, zorder=4)


def legend_handles(row):
    h = [Line2D([], [], ls='', marker='o', ms=3.4, color='black', label=LEGEND_LABELS['data']),
         Line2D([], [], ls='--', dashes=(3.2, 1.8), lw=1.0, color='green', label=LEGEND_LABELS['true'])]
    if row == 0:
        h.append(Line2D([], [], lw=1.1, color='red', label=LEGEND_LABELS['pred']))
    else:
        h += [Line2D([], [], lw=1.1, color='red', label=LEGEND_LABELS['mean']),
              Patch(facecolor='red', alpha=0.3, lw=0, label=LEGEND_LABELS['band'])]
    return h


def make_figure(columns, data, path):
    # Layout in inches (top to bottom): legend, row 1, x ticks, legend, row 2, x ticks, x label, panel labels;
    # left to right: row label, y label, y tick labels, three panels with a shared y axis per row
    left, right, gap_x = 0.72, 0.04, 0.13
    leg_h, row_h, tick_h, xlabel_h, panel_h = 0.21, 1.08, 0.21, 0.15, 0.22
    fig_h = 2 * (leg_h + row_h + tick_h) + xlabel_h + panel_h + 0.02
    pw = (FIG_W - left - right - 2 * gap_x) / 3
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=100)
    axes = np.empty((2, 3), dtype=object)
    for i in range(2):
        top = fig_h - 0.02 - leg_h - i * (leg_h + row_h + tick_h)
        for j in range(3):
            x0 = left + j * (pw + gap_x)
            axes[i, j] = fig.add_axes([x0 / FIG_W, (top - row_h) / fig_h, pw / FIG_W, row_h / fig_h],
                                      sharey=axes[i, 0] if j else None)
    for j, scn in enumerate(columns):
        for i in range(2):
            draw_panel(axes[i, j], data[scn], i)
    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            ax.set_xlim(-0.03, 1.03)
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            # "0", "0.2", ..., "1": narrow end labels, so that "1" and "0" of neighboring panels stay apart
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
            ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.6, pad=1.5)
            for s in ax.spines.values():
                s.set_linewidth(0.6)
            if j:
                ax.tick_params(labelleft=False)
        axes[i, 0].set_ylabel('$y_{HF}$', fontsize=FS_YLABEL)
        axes[i, 0].yaxis.set_label_coords(-0.37 / pw, 0.5)     # same position in both rows
        # Row label left of the y label, centered on the row
        pos = axes[i, 0].get_position()
        fig.text(0.03 / FIG_W, 0.5 * (pos.y0 + pos.y1), ROW_LABELS[i], fontsize=FS_ROW, rotation=90,
                 ha='left', va='center')
        # Row legend above the row, centered on the three panels, outside the axes
        handles = legend_handles(i)
        fig.legend(handles=handles, loc='center', ncol=len(handles), frameon=False, fontsize=FS_LEGEND,
                   bbox_to_anchor=(0.5 * (axes[i, 0].get_position().x0 + axes[i, 2].get_position().x1),
                                   pos.y1 + 0.5 * leg_h / fig_h),
                   handlelength=1.5, handletextpad=0.5, columnspacing=1.0, borderaxespad=0, borderpad=0)
    for j in range(3):
        axes[1, j].set_xlabel('$x$', fontsize=FS_XLABEL, labelpad=1)
        pos = axes[1, j].get_position()
        fig.text(0.5 * (pos.x0 + pos.x1), (0.5 * panel_h + 0.01) / fig_h, f'({"abc"[j]})', fontsize=FS_PANEL,
                 ha='center', va='center')
    fig.savefig(path, dpi=DPI, pil_kwargs={'quality': 95})
    check_layout(fig, path)
    plt.close(fig)


def drawn_texts(fig):
    """The text objects that are drawn: figure texts, legend entries, axis labels and the tick labels of the ticks
    inside the view limits (the tick locator also labels ticks beyond them, which are not drawn)."""
    out = list(fig.texts) + [t for leg in fig.legends for t in leg.get_texts()]
    for ax in fig.axes:
        out += [ax.xaxis.label, ax.yaxis.label, ax.title]
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            for tick in axis.get_major_ticks():
                if min(lim) - 1e-9 <= tick.get_loc() <= max(lim) + 1e-9:
                    out += [tick.label1, tick.label2]
    return [t for t in out if isinstance(t, Text) and t.get_visible() and t.get_text().strip()]


def check_layout(fig, path):
    """Smallest font at the printed size (the figure is drawn at that size; mathtext subscripts are 0.7 of their
    label), no legend overlapping an axes, no text outside the figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    sizes, boxes = [], []
    for t in drawn_texts(fig):
        sub = 0.7 if '_{' in t.get_text() else 1.0
        sizes.append(t.get_fontsize() * sub)
        bb = t.get_window_extent(r)
        assert bb.x0 >= -0.5 and bb.y0 >= -0.5 and bb.x1 <= fig.bbox.width + 0.5 and \
            bb.y1 <= fig.bbox.height + 0.5, f'text {t.get_text()!r} outside the figure'
        boxes.append((t.get_text(), bb))
    for k, (s1, b1) in enumerate(boxes):
        for s2, b2 in boxes[k + 1:]:
            assert not b1.overlaps(b2), f'texts {s1!r} and {s2!r} overlap in {path}'
    for leg in fig.legends:
        lb = leg.get_window_extent(r)
        for ax in fig.axes:
            assert not lb.overlaps(ax.get_window_extent(r)), f'legend overlaps an axes in {path}'
    print(f'{os.path.basename(path)}: {fig.get_figwidth():.3f} x {fig.get_figheight():.3f} in at {DPI} dpi, '
          f'smallest text {min(sizes):.2f} pt at the printed width {FIG_W:.3f} in')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default=os.path.join(HERE, 'figures'))
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rows = []
    for filename, columns in FIGURES.items():
        data = {scn: load_scenario(scn) for scn in columns}
        for scn in columns:
            m = metrics(data[scn])
            row = {'scenario': scn, 'hmc_draws': data[scn]['n_draws'], **m}
            for k, v in PAPER_VALUES[scn].items():
                decimals = 1 if k == 'coverage_pct' else (2 if k == 'mean_width' else 3)
                row[f'paper_{k}'] = v
                row[f'{k}_reproduced'] = round(m[k], decimals) == v
            rows.append(row)
            print(scn, {k: round(v, 4) for k, v in m.items()},
                  'reproduced' if all(row[f'{k}_reproduced'] for k in PAPER_VALUES[scn]) else 'NOT reproduced')
        out = os.path.join(a.outdir, filename)
        assert not os.path.exists(out) or out.endswith('_v2.jpeg'), out
        make_figure(columns, data, out)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(HERE, 'figure_v2_metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
