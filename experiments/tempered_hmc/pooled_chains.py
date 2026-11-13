"""Two independent chains of HMC on the tempered posterior per run (80 and 250 leapfrog steps, same start): between-
chain convergence diagnostics and pooled estimates (comment C5).

For every (sigma, seed) with both chains: Gelman-Rubin R-hat over the 4 half-chains (split-R-hat of the two chains)
for k and, as the maximum over the 101 thinned test points, for u and f; pooled k (4000 draws: mean, sd, 95% interval,
standardized error); pooled u and f on the thinned grid (every 20th of the 2001 test points): MSE of the mean, coverage
of the truth by the 2.5/97.5 percentiles, mean interval width. The published HMC and VI runs are evaluated on the same
thinned grid from their saved means and percentile bands. Writes tables/pooled_runs.csv and prints summaries."""
import glob
import os
import re

import numpy as np
import pandas as pd

BENCH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(BENCH, 'results_bpinn')
K_TRUE = 0.7
STRIDE = 20
C_K = 1.0 / np.sqrt(np.sum(np.tanh(np.sin(6 * np.linspace(-0.7, 0.7, 32)) ** 3) ** 2))


def rhat(chains):
    # Gelman-Rubin over split halves: chains is a list of (n,) or (n, m) arrays of equal length
    halves = []
    for c in chains:
        n = c.shape[0] // 2
        halves += [c[:n], c[n:2 * n]]
    x = np.stack(halves)                       # (M, n, ...)
    n = x.shape[1]
    w = x.var(axis=1, ddof=1).mean(axis=0)
    b = n * x.mean(axis=1).var(axis=0, ddof=1)
    return np.sqrt(((n - 1) / n * w + b / n) / w)


def published(folder, model, noise, seed):
    path = os.path.join(PUB, folder, 'runs', f'{model}_sigma{noise:.3f}_seed{seed}.npz')
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def band_metrics(mean, lo, hi, true):
    return {'mse': float(np.mean((mean - true) ** 2)), 'cov': float(np.mean((true >= lo) & (true <= hi))),
            'width': float(np.mean(hi - lo))}


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, 'hmc_tempered_L250*', 'parts', '*', 'diag', '*.npz'))):
        m = re.search(r'sigma([0-9.]+)_seed(\d+)\.npz$', path)
        noise, seed = float(m.group(1)), int(m.group(2))
        other = path.replace('hmc_tempered_L250', 'hmc_tempered')
        if not os.path.exists(other):
            continue
        a, b = np.load(other), np.load(path)                       # L80 chain, L250 chain
        run = np.load(os.path.join(os.path.dirname(os.path.dirname(other)), 'runs', os.path.basename(other)))
        u_true, f_true = run['u_true'][::STRIDE], run['f_true'][::STRIDE]
        k = np.concatenate([a['k'], b['k']])
        lo, hi = np.percentile(k, [2.5, 97.5])
        row = {'noise_level': noise, 'seed': seed, 'k_mean': k.mean(), 'k_sd': k.std(),
               'k_mean_L80': a['k'].mean(), 'k_mean_L250': b['k'].mean(), 'k_sd_L80': a['k'].std(),
               'k_sd_L250': b['k'].std(), 'k_in95': int(lo <= K_TRUE <= hi), 'z_k': (k.mean() - K_TRUE) / k.std(),
               'rhat_k': float(rhat([a['k'], b['k']])),
               'rhat_u_max': float(np.max(rhat([a['u_thin'].astype(float), b['u_thin'].astype(float)]))),
               'rhat_f_max': float(np.max(rhat([a['f_thin'].astype(float), b['f_thin'].astype(float)]))),
               'rhat_chi2': float(rhat([a['chi2_u'] + a['chi2_f'], b['chi2_u'] + b['chi2_f']]))}
        for name, true in (('u', u_true), ('f', f_true)):
            draws = np.concatenate([a[name + '_thin'], b[name + '_thin']]).astype(float)
            for key, val in band_metrics(draws.mean(0), *np.percentile(draws, [2.5, 97.5], axis=0), true).items():
                row[f'{name}_{key}_pooled'] = val
            for tag, ch in (('L80', a), ('L250', b)):
                dr = ch[name + '_thin'].astype(float)
                row[f'{name}_mse_{tag}'] = float(np.mean((dr.mean(0) - true) ** 2))
        folder = 'noise_study_ode_v2' if seed <= 3 else 'calibration_sigma0.2_v2'
        for label, model in (('HMC', 'mfbpinn_hmc'), ('VI', 'mfbpinn_vi')):
            d = published(folder, model, noise, seed)
            for name in ('u', 'f'):
                met = band_metrics(d[name + '_mean'][::STRIDE], d[name + '_lo'][::STRIDE], d[name + '_hi'][::STRIDE],
                                   d[name + '_true'][::STRIDE])
                for key, val in met.items():
                    row[f'{name}_{key}_{label}'] = val
            row[f'k_mean_{label}'] = float(d['k_mean'])
            row[f'k_sd_{label}'] = float(d['k_sd'])
        rows.append(row)
    r = pd.DataFrame(rows).sort_values(['noise_level', 'seed'])
    os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
    r.to_csv(os.path.join(HERE, 'tables', 'pooled_runs.csv'), index=False)
    if not len(r):
        print('no run with both chains yet')
        return
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 60)
    pd.set_option('display.float_format', lambda v: f'{v:.3g}')
    print(r[['noise_level', 'seed', 'k_mean_L80', 'k_mean_L250', 'k_sd_L80', 'k_sd_L250', 'k_mean', 'k_sd', 'k_in95',
             'rhat_k', 'rhat_u_max', 'rhat_f_max', 'rhat_chi2', 'f_mse_L80', 'f_mse_L250', 'f_mse_pooled', 'f_mse_HMC',
             'u_cov_pooled', 'f_cov_pooled']].to_string(index=False))
    for study, x in (('noise study (seeds 1-3)', r[r.seed <= 3]), ('sigma = 0.2, seeds 4-23', r[r.seed >= 4])):
        if not len(x):
            continue
        print(f'\n== {study}: n = {len(x)} ==')
        print(f"k sd ratio pooled HMC-T / HMC: median {np.median(x.k_sd / x.k_sd_HMC):.2f} "
              f"[{np.min(x.k_sd / x.k_sd_HMC):.2f}, {np.max(x.k_sd / x.k_sd_HMC):.2f}]; VI / pooled HMC-T: "
              f"{np.median(x.k_sd_VI / x.k_sd):.2f} [{np.min(x.k_sd_VI / x.k_sd):.2f}, {np.max(x.k_sd_VI / x.k_sd):.2f}]")
        print(f"k sd pooled / (0.41 sigma sqrt(0.1)): {np.median(x.k_sd / (C_K * x.noise_level * np.sqrt(0.1))):.2f} "
              f"[{np.min(x.k_sd / (C_K * x.noise_level * np.sqrt(0.1))):.2f}, "
              f"{np.max(x.k_sd / (C_K * x.noise_level * np.sqrt(0.1))):.2f}]")
        print(f"k mean: pooled HMC-T - HMC median {np.median(x.k_mean - x.k_mean_HMC):+.3f}; VI - pooled HMC-T median "
              f"{np.median(x.k_mean_VI - x.k_mean):+.3f}; mean k error pooled HMC-T {np.mean(x.k_mean - K_TRUE):+.3f}")
        print(f"k 95% interval contains 0.7: pooled HMC-T {int(x.k_in95.sum())}/{len(x)}; z mean {x.z_k.mean():+.2f}, "
              f"sd {x.z_k.std(ddof=0):.2f}")
        print(f"R-hat (2 chains, split): k median {x.rhat_k.median():.2f} max {x.rhat_k.max():.2f}; "
              f"u max-over-grid median {x.rhat_u_max.median():.2f}; f median {x.rhat_f_max.median():.2f}; "
              f"chi2 median {x.rhat_chi2.median():.2f}")
        for name in ('u', 'f'):
            print(f"{name}: MSE pooled / HMC median {np.median(x[f'{name}_mse_pooled'] / x[f'{name}_mse_HMC']):.2f}; "
                  f"single chains L80 {np.median(x[f'{name}_mse_L80'] / x[f'{name}_mse_HMC']):.2f}, "
                  f"L250 {np.median(x[f'{name}_mse_L250'] / x[f'{name}_mse_HMC']):.2f}; coverage pooled "
                  f"{x[f'{name}_cov_pooled'].mean():.2f} (HMC {x[f'{name}_cov_HMC'].mean():.2f}, VI "
                  f"{x[f'{name}_cov_VI'].mean():.2f}); width pooled / HMC "
                  f"{np.median(x[f'{name}_width_pooled'] / x[f'{name}_width_HMC']):.2f}, VI / pooled "
                  f"{np.median(x[f'{name}_width_VI'] / x[f'{name}_width_pooled']):.2f}; VI MSE / pooled "
                  f"{np.median(x[f'{name}_mse_VI'] / x[f'{name}_mse_pooled']):.2f}")
    x = r[r.seed <= 3]
    if len(x):
        print('\nnoise study, medians per level (pooled HMC-T):')
        print(x.groupby('noise_level')[['k_sd', 'k_sd_HMC', 'k_sd_VI', 'f_mse_pooled', 'f_mse_HMC', 'f_mse_VI',
                                        'u_mse_pooled', 'u_mse_HMC', 'u_mse_VI', 'u_cov_pooled', 'f_cov_pooled']]
              .median().to_string())


if __name__ == '__main__':
    main()
