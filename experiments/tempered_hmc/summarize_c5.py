"""Comment C5: HMC (untempered), HMC on the tempered posterior, VI (beta = 0.1, published) and VI (beta = 1).

Reads the published runs (results_bpinn/noise_study_ode_v2, calibration_sigma0.2_v2, ablations_v2/vi_beta1) and the
runs of this folder (hmc_tempered/, hmc_tempered_cal/, vi_beta1_cal/, validation_beta1/), and writes
  tables/runs_all.csv            one row per (method, noise, seed) with the metrics and the k interval
  tables/noise_study_medians.csv medians over seeds 1-3 per (method, noise level)
  tables/calibration_sigma0.2.csv  sigma = 0.2, seeds 4-23: means of coverage, k statistics, standardized errors
  tables/decomposition.csv       paired ratios: tempering (HMC-T / HMC), VI error on the same target (VI / HMC-T),
                                 VI(beta = 1) / HMC
and prints the paper numbers it reproduces (main.tex L303, L307, Table 3).
"""
import glob
import os

import numpy as np
import pandas as pd

BENCH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(BENCH, 'results_bpinn')
K_TRUE = 0.7
BETA = 0.1
NOISE = [0.01, 0.05, 0.1, 0.15, 0.2]
X_F = np.linspace(-0.7, 0.7, 32)
C_K = 1.0 / np.sqrt(np.sum(np.tanh(np.sin(6 * X_F) ** 3) ** 2))     # 0.4149: sd(k) = C_K sigma if u is known


def k_draws(folder, model, noise, seed):
    path = os.path.join(folder, 'runs', f'{model}_sigma{noise:.3f}_seed{seed}.npz')
    with np.load(path) as d:
        return np.asarray(d['k_samples'], np.float64)


def load(folder, model, label, csv_glob=None):
    # Rows of one method from a (merged) sweep folder, or from the per-run part folders of this study
    files = sorted(glob.glob(csv_glob)) if csv_glob else [os.path.join(folder, 'noise_study_runs.csv')]
    rows = []
    for f in files:
        d = pd.read_csv(f)
        d = d[d.model == model].copy()
        d['run_folder'] = os.path.dirname(f)
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    d['method'] = label
    d['seed'] = d.seed.astype(int)
    lo, hi = [], []
    for _, r in d.iterrows():
        k = k_draws(r.run_folder, model, float(r.noise_level), int(r.seed))
        a, b = np.percentile(k, [2.5, 97.5])
        lo.append(a)
        hi.append(b)
    d['k_lo95'], d['k_hi95'] = lo, hi
    d['k_in95'] = ((d.k_lo95 <= K_TRUE) & (K_TRUE <= d.k_hi95)).astype(int)
    d['z_k'] = d.nu_error / d.nu_std
    d['k_sd_over_C_sigma'] = d.nu_std / (C_K * d.noise_level)
    d['k_sd_over_C_sigma_sqrtbeta'] = d.nu_std / (C_K * d.noise_level * np.sqrt(BETA))
    return d


def all_runs():
    parts = []
    for folder in ('noise_study_ode_v2', 'calibration_sigma0.2_v2'):
        parts.append(load(os.path.join(PUB, folder), 'mfbpinn_hmc', 'HMC'))
        parts.append(load(os.path.join(PUB, folder), 'mfbpinn_vi', 'VI b=0.1'))
    parts.append(load(os.path.join(PUB, 'ablations_v2', 'vi_beta1'), 'mfbpinn_vi', 'VI b=1'))
    parts.append(load(None, 'mfbpinn_vi', 'VI b=1', os.path.join(HERE, 'vi_beta1_cal', 'parts', '*', 'noise_study_runs.csv')))
    for sub, label in (('hmc_tempered', 'HMC-T'), ('hmc_tempered_cal', 'HMC-T'),
                       ('hmc_tempered_L250', 'HMC-T L250'), ('hmc_tempered_L250_cal', 'HMC-T L250')):
        parts.append(load(None, 'mfbpinn_hmc_tempered', label,
                          os.path.join(HERE, sub, 'parts', '*', 'noise_study_runs.csv')))
    d = pd.concat([p for p in parts if len(p)], ignore_index=True)
    d['study'] = np.where(d.seed <= 3, 'noise', 'cal')
    return d


METRICS = ['u_mse', 'f_mse', 'u_coverage_95', 'f_coverage_95', 'u_width_95', 'f_width_95', 'nu_mean', 'nu_abs_error',
           'nu_std', 'k_sd_over_C_sigma', 'k_sd_over_C_sigma_sqrtbeta', 'k_ess', 'hmc_acceptance', 'hmc_step_size']
ORDER = ['HMC', 'HMC-T', 'HMC-T L250', 'VI b=0.1', 'VI b=1']


def medians(d):
    g = d[d.study == 'noise'].groupby(['method', 'noise_level'])
    m = g[METRICS].median()
    m['k_in95'] = g.k_in95.sum()
    m['n'] = g.size()
    return m.reset_index()


def calibration(d):
    c = d[(d.study == 'cal') & np.isclose(d.noise_level, 0.2)]
    rows = []
    for meth in ORDER:
        x = c[c.method == meth]
        if not len(x):
            continue
        rows.append({'method': meth, 'n': len(x), 'u_cov_mean': x.u_coverage_95.mean(), 'f_cov_mean': x.f_coverage_95.mean(),
                     'u_width_mean': x.u_width_95.mean(), 'f_width_mean': x.f_width_95.mean(),
                     'u_mse_median': x.u_mse.median(), 'f_mse_median': x.f_mse.median(),
                     'k_in95': int(x.k_in95.sum()), 'k_err_mean': x.nu_error.mean(),
                     'k_abs_err_median': x.nu_abs_error.median(), 'k_sd_mean': x.nu_std.mean(),
                     'k_sd_median': x.nu_std.median(), 'z_mean': x.z_k.mean(), 'z_sd': x.z_k.std(ddof=0),
                     'k_sd_over_C_sigma_median': x.k_sd_over_C_sigma.median(),
                     'k_ess_median': x.k_ess.median(), 'k_ess_min': x.k_ess.min()})
    return pd.DataFrame(rows)


def paired(d, num, den, cols=('u_mse', 'f_mse', 'u_width_95', 'f_width_95', 'nu_std', 'nu_abs_error')):
    a = d[d.method == num].set_index(['noise_level', 'seed'])
    b = d[d.method == den].set_index(['noise_level', 'seed'])
    idx = a.index.intersection(b.index)
    out = pd.DataFrame(index=idx)
    for c in cols:
        out[c + '_ratio'] = a.loc[idx, c] / b.loc[idx, c]
    for c in ('u_coverage_95', 'f_coverage_95', 'k_in95'):
        out[c + '_diff'] = a.loc[idx, c] - b.loc[idx, c]
    out['nu_mean_diff'] = a.loc[idx, 'nu_mean'] - b.loc[idx, 'nu_mean']
    out['pair'] = f'{num} / {den}'
    return out.reset_index()


def main():
    d = all_runs()
    os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
    d.to_csv(os.path.join(HERE, 'tables', 'runs_all.csv'), index=False)
    m = medians(d)
    m.to_csv(os.path.join(HERE, 'tables', 'noise_study_medians.csv'), index=False)
    c = calibration(d)
    c.to_csv(os.path.join(HERE, 'tables', 'calibration_sigma0.2.csv'), index=False)
    pairs = pd.concat([paired(d, 'HMC-T', 'HMC'), paired(d, 'HMC-T L250', 'HMC'), paired(d, 'HMC-T L250', 'HMC-T'),
                       paired(d, 'VI b=0.1', 'HMC-T'), paired(d, 'VI b=0.1', 'HMC-T L250'),
                       paired(d, 'VI b=1', 'HMC'), paired(d, 'VI b=0.1', 'HMC')], ignore_index=True)
    pairs.to_csv(os.path.join(HERE, 'tables', 'decomposition.csv'), index=False)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.float_format', lambda v: f'{v:.4g}')
    print(f'C_K = {C_K:.4f} (0.41 in L307); C_K sqrt(beta) = {C_K * np.sqrt(BETA):.4f}')
    print('\nRuns per method/study:\n', d.groupby(['method', 'study']).size())
    for meth in ORDER:
        x = m[m.method == meth]
        if len(x):
            print(f'\n== {meth}: medians over seeds 1-3 (k_in95 = count of 3) ==')
            print(x.drop(columns=['method']).to_string(index=False))
    print('\n== sigma = 0.2, seeds 4-23 ==\n', c.to_string(index=False))
    g = pairs[pairs.seed <= 3].groupby(['pair', 'noise_level'])
    print('\n== paired ratios, medians over seeds 1-3 ==')
    print(g[[col for col in pairs.columns if col.endswith('_ratio')] + ['nu_mean_diff']].median().to_string())
    g = pairs[pairs.seed >= 4].groupby('pair')
    print('\n== paired ratios, sigma = 0.2 seeds 4-23 (median) ==')
    print(g[[col for col in pairs.columns if col.endswith('_ratio')] + ['nu_mean_diff']].median().to_string())
    print('\n== paired ratios, all 35 runs (15 noise study + 20 at sigma = 0.2): median [min, max] ==')
    for pair, x in pairs.groupby('pair'):
        sel = x[x.noise_level >= 0.05]
        print(f'{pair:22s} n={len(x):2d}: k sd ratio {x.nu_std_ratio.median():.2f} [{x.nu_std_ratio.min():.2f}, '
              f'{x.nu_std_ratio.max():.2f}] (sigma>=0.05: {sel.nu_std_ratio.median():.2f} [{sel.nu_std_ratio.min():.2f}, '
              f'{sel.nu_std_ratio.max():.2f}]); k mean diff median {x.nu_mean_diff.median():+.3f}; '
              f'u width ratio {x.u_width_95_ratio.median():.2f}; f width ratio {x.f_width_95_ratio.median():.2f}; '
              f'u MSE ratio {x.u_mse_ratio.median():.2f}; f MSE ratio {x.f_mse_ratio.median():.2f}')

    # Sampler diagnostics of HMC on the tempered posterior (and of the beta = 1 validation runs)
    diag_cols = ['hmc_step_size', 'step_ratio_to_HMC', 'burnin_acceptance', 'hmc_acceptance', 'k_ess', 'ess_u_min',
                 'ess_f_min', 'ess_chi2_total', 'k_split_rhat', 'rhat_u_max', 'rhat_f_max']
    hmc = d[d.method == 'HMC'].set_index(['noise_level', 'seed']).hmc_step_size
    rows = []
    for meth in ('HMC-T', 'HMC-T L250'):
        x = d[d.method == meth].copy()
        if not len(x):
            continue
        x['step_ratio_to_HMC'] = [hmc.loc[(n, s)] / v for n, s, v in zip(x.noise_level, x.seed, x.hmc_step_size)]
        for (study, noise), y in x.groupby(['study', 'noise_level']):
            row = {'method': meth, 'study': study, 'noise_level': noise, 'n': len(y)}
            for col in diag_cols:
                row[col + '_median'] = y[col].median()
                row[col + '_min'] = y[col].min()
                row[col + '_max'] = y[col].max()
            rows.append(row)
    if rows:
        diag = pd.DataFrame(rows)
        diag.to_csv(os.path.join(HERE, 'tables', 'hmc_tempered_diagnostics.csv'), index=False)
        print('\n== sampler diagnostics (median [min-max]) ==')
        for _, r in diag.iterrows():
            print(f"{r.method:10s} {r.study:5s} sigma={r.noise_level:<5} n={r.n:2d}: step {r.hmc_step_size_median:.1e} "
                  f"(HMC/this {r.step_ratio_to_HMC_median:.1f} [{r.step_ratio_to_HMC_min:.1f}-{r.step_ratio_to_HMC_max:.1f}]), "
                  f"acc burn-in {r.burnin_acceptance_median:.2f} sampling {r.hmc_acceptance_median:.2f}; "
                  f"ESS k {r.k_ess_median:.0f} [{r.k_ess_min:.0f}-{r.k_ess_max:.0f}], u min {r.ess_u_min_median:.0f} "
                  f"[{r.ess_u_min_min:.0f}-{r.ess_u_min_max:.0f}], f min {r.ess_f_min_median:.0f} "
                  f"[{r.ess_f_min_min:.0f}-{r.ess_f_min_max:.0f}], chi2 {r.ess_chi2_total_median:.0f}; split-Rhat k max "
                  f"{r.k_split_rhat_max:.2f}, u max {r.rhat_u_max_max:.2f}, f max {r.rhat_f_max_max:.2f}")


if __name__ == '__main__':
    main()
