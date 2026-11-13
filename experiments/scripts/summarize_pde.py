"""Summaries of the inverse-PDE reviewer runs (MC-dropout rate sweep, deep ensemble, single-network MC dropout).

Reads the merged sweep folders written by example_mf_bpinn_noise_analysis.py --mode merge, the deep-ensemble runs and
the published study (results_bpinn/noise_study_ode_v2), and writes median-over-seeds tables (CSV + Markdown) and a
metric-vs-noise figure into the experiment folders.
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.dirname(R)
EXP2 = os.path.join(R, 'pde_dropout_rate_ensemble')
EXP3 = os.path.join(R, 'dropout_single_net')
PUB = os.path.join(B, 'results_bpinn', 'noise_study_ode_v2', 'noise_study_runs.csv')
P_VALUES = ['0.01', '0.02', '0.05', '0.1', '0.2']
NOISE = [0.01, 0.05, 0.1, 0.15, 0.2]
METRICS = ['u_mse', 'f_mse', 'k_abs_error', 'u_coverage_95', 'f_coverage_95', 'u_width_95', 'f_width_95']


def read_csv(path):
    with open(path, encoding='utf-8-sig') as f:
        return [r for r in csv.DictReader(f) if r.get('noise_level')]


def norm(rows, label):
    # Common field names: the sweep CSV calls |k - 0.7| nu_abs_error and k nu_mean
    out = []
    for r in rows:
        o = {'model': label, 'noise_level': float(r['noise_level']), 'seed': int(float(r['seed']))}
        for m in METRICS:
            src = {'k_abs_error': 'nu_abs_error'}.get(m, m) if m not in r else m
            o[m] = float(r[src])
        o['k_mean'] = float(r.get('k_mean', r.get('nu_mean', 'nan')))
        o['k_sd'] = float(r.get('k_sd', r.get('nu_std', 'nan')) or 'nan')
        o['train_seconds'] = float(r.get('train_seconds', r.get('train_seconds_total', 'nan')))
        out.append(o)
    return out


def medians(rows):
    table = []
    for label in dict.fromkeys(r['model'] for r in rows):
        for noise in NOISE:
            rs = [r for r in rows if r['model'] == label and abs(r['noise_level'] - noise) < 1e-9]
            if rs:
                table.append({'model': label, 'noise_level': noise, 'runs': len(rs),
                              **{m: float(np.median([r[m] for r in rs])) for m in METRICS + ['train_seconds']}})
    return table


def write(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('Saved', path)


def markdown(rows, labels):
    lines = ['| model | sigma | u MSE | f MSE | abs(k-0.7) | u cov95 | f cov95 | u width | f width |',
             '|---|---|---|---|---|---|---|---|---|']
    for label in labels:
        for r in rows:
            if r['model'] == label:
                lines.append(f"| {label} | {r['noise_level']:g} | {r['u_mse']:.2e} | {r['f_mse']:.2e} | "
                             f"{r['k_abs_error']:.3f} | {r['u_coverage_95']:.2f} | {r['f_coverage_95']:.2f} | "
                             f"{r['u_width_95']:.3f} | {r['f_width_95']:.3f} |")
    return '\n'.join(lines)


def plot(table, labels, styles, path, title):
    panels = [('u_mse', 'u MSE', True), ('f_mse', 'f MSE', True), ('k_abs_error', r'$|k - k_{\mathrm{true}}|$', True),
              ('u_coverage_95', 'u 95% coverage', False), ('f_coverage_95', 'f 95% coverage', False),
              ('f_width_95', 'f mean 95% width', True)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    for ax, (key, name, log) in zip(axes.ravel(), panels):
        for label in labels:
            rs = sorted((r for r in table if r['model'] == label), key=lambda r: r['noise_level'])
            if not rs or (label.startswith('Deterministic') and not log):
                continue   # the deterministic model has no interval
            if label.startswith('Deterministic') and key == 'f_width_95':
                continue
            ax.plot([r['noise_level'] for r in rs], [r[key] for r in rs], label=label, **styles[label])
        ax.set_title(name, fontsize=16)
        ax.set_xlabel('Noise level', fontsize=14)
        ax.set_xticks([0.0, 0.1, 0.2])
        ax.set_xlim(0.0, 0.21)
        ax.grid(alpha=0.3)
        if log:
            ax.set_yscale('log')
        else:
            ax.axhline(0.95, ls='--', color='black', lw=1.2)
            ax.set_ylim(0.0, 1.05)
    axes[0, 0].legend(fontsize=11)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('Saved', path)


def main():
    pub = read_csv(PUB)
    published = {m: norm([r for r in pub if r['model'] == m], lab) for m, lab in
                 [('mfbpinn_hmc', 'HMC (published)'), ('mfbpinn_vi', 'VI (published)'),
                  ('mfpinn', 'MC dropout p=0.05 (published)'), ('mfpinn_det', 'Deterministic (published)')]}

    # Experiment 2a: dropout-rate sweep
    sweep = []
    for p in P_VALUES:
        path = os.path.join(EXP2, f'p{p}', 'noise_study_runs.csv')
        if os.path.exists(path):
            sweep += norm(read_csv(path), f'MC dropout p={p}')
    # Reproduction check: p=0.05 against the published MC-dropout runs
    rerun = {(r['noise_level'], r['seed']): r for r in sweep if r['model'] == 'MC dropout p=0.05'}
    diffs = [max(abs(rerun[(r['noise_level'], r['seed'])][m] - r[m]) for m in METRICS)
             for r in published['mfpinn'] if (r['noise_level'], r['seed']) in rerun]
    print(f'p=0.05 re-run vs published MC dropout: {len(diffs)} runs, max metric difference '
          f'{max(diffs) if diffs else float("nan"):.3g}')

    ens_path = os.path.join(EXP2, 'deep_ensemble', 'ensemble_runs.csv')
    ens = norm(read_csv(ens_path), 'Deep ensemble (10 det.)') if os.path.exists(ens_path) else []
    ref = published['mfbpinn_hmc'] + published['mfbpinn_vi'] + published['mfpinn_det']
    table2 = medians(sweep + ens + ref)
    have_exp2 = os.path.isdir(EXP2)   # dropout-rate sweep / deep ensemble (not included in this repository)
    if have_exp2:
        write(os.path.join(EXP2, 'median_over_seeds.csv'), table2)
    labels2 = [f'MC dropout p={p}' for p in P_VALUES] + ['Deep ensemble (10 det.)', 'Deterministic (published)',
                                                        'HMC (published)', 'VI (published)']
    if have_exp2:
        with open(os.path.join(EXP2, 'median_over_seeds.md'), 'w') as f:
            f.write(markdown(table2, labels2) + '\n')
    blues = plt.cm.Blues(np.linspace(0.35, 1.0, len(P_VALUES)))
    styles = {f'MC dropout p={p}': {'color': c, 'marker': 'o', 'lw': 2} for p, c in zip(P_VALUES, blues)}
    styles.update({'Deep ensemble (10 det.)': {'color': 'black', 'marker': 's', 'lw': 2, 'ls': '--'},
                   'Deterministic (published)': {'color': '#9467bd', 'marker': 'v', 'lw': 1.5, 'ls': ':'},
                   'HMC (published)': {'color': '#d62728', 'marker': 'o', 'lw': 2.5},
                   'VI (published)': {'color': '#2ca02c', 'marker': 'o', 'lw': 2.5},
                   'MC dropout single net p=0.05': {'color': '#ff7f0e', 'marker': 'D', 'lw': 2.5},
                   'MC dropout p=0.05 (published)': {'color': '#1f77b4', 'marker': 'o', 'lw': 2.5}})
    if have_exp2:
        plot(table2, labels2, styles, os.path.join(EXP2, 'metric_vs_noise_dropout_rate.png'),
             'Inverse PDE: MC-dropout rate sweep and deep ensemble (median over 3 noise realizations)')

    # Experiment 3: single-network MC dropout
    single_path = os.path.join(EXP3, 'merged', 'noise_study_runs.csv')
    if os.path.exists(single_path):
        single = norm(read_csv(single_path), 'MC dropout single net p=0.05')
        table3 = medians(single + published['mfpinn'] + published['mfbpinn_hmc'] + published['mfbpinn_vi'])
        write(os.path.join(EXP3, 'median_over_seeds.csv'), table3)
        labels3 = ['MC dropout p=0.05 (published)', 'MC dropout single net p=0.05', 'HMC (published)',
                   'VI (published)']
        with open(os.path.join(EXP3, 'median_over_seeds.md'), 'w') as f:
            f.write(markdown(table3, labels3) + '\n')
        plot(table3, labels3, styles, os.path.join(EXP3, 'metric_vs_noise_single_net.png'),
             'Inverse PDE: MC dropout with the Eq. 3 split vs a single [2,50,50,1] network')
        # Per-run paired comparison (same data, same LF network, same nonlinear-head initialization)
        pairs = []
        for r in single:
            q = next(x for x in published['mfpinn'] if x['noise_level'] == r['noise_level'] and x['seed'] == r['seed'])
            pairs.append({'noise_level': r['noise_level'], 'seed': r['seed'],
                          **{f'{m}_split': q[m] for m in METRICS[:5]}, **{f'{m}_single': r[m] for m in METRICS[:5]},
                          'k_split': q['k_mean'], 'k_single': r['k_mean']})
        write(os.path.join(EXP3, 'paired_runs.csv'), pairs)


if __name__ == '__main__':
    main()
