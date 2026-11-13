"""Validation of hmc_tempered.py: with --beta 1 it must reproduce the published HMC runs (same data, LF network, MAP
warm start, chain seed and settings). Compares every metric of the CSV and every array of the run npz with the
published run, and checks that the tempered runs started from the published MAP estimate (map_k)."""
import glob
import os

import numpy as np
import pandas as pd

BENCH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
PUB = {seed: os.path.join(BENCH, 'results_bpinn', 'noise_study_ode_v2' if seed <= 3 else 'calibration_sigma0.2_v2')
       for seed in range(1, 24)}
SKIP = {'train_seconds', 'predict_seconds', 'model', 'run_folder'}


def published_row(noise, seed, model):
    d = pd.read_csv(os.path.join(PUB[seed], 'noise_study_runs.csv'))
    r = d[(d.model == model) & np.isclose(d.noise_level, noise) & (d.seed == seed)]
    assert len(r) == 1, (noise, seed, model)
    return r.iloc[0]


def main():
    print('== beta = 1 runs vs the published HMC runs ==')
    for f in sorted(glob.glob(os.path.join(HERE, 'validation_beta1', 'parts', '*', 'noise_study_runs.csv'))):
        mine = pd.read_csv(f).iloc[0]
        noise, seed = float(mine.noise_level), int(mine.seed)
        pub = published_row(noise, seed, 'mfbpinn_hmc')
        worst = 0.0
        for c in pub.index:
            if c in SKIP or c not in mine.index or not np.isfinite(float(pub[c])):
                continue
            worst = max(worst, abs(float(mine[c]) - float(pub[c])) / max(abs(float(pub[c])), 1e-12))
        name = f'mfbpinn_hmc_sigma{noise:.3f}_seed{seed}.npz'
        a = np.load(os.path.join(os.path.dirname(f), 'runs', name))
        b = np.load(os.path.join(PUB[seed], 'runs', name))
        arr = max(float(np.max(np.abs(np.asarray(a[k], float) - np.asarray(b[k], float)))) if np.asarray(a[k]).size else 0.0
                  for k in b.files)
        print(f'sigma={noise} seed={seed}: k {mine.nu_mean:.6f}+-{mine.nu_std:.6f} (published {pub.nu_mean:.6f}+-'
              f'{pub.nu_std:.6f}); u_mse {mine.u_mse:.6e} ({pub.u_mse:.6e}); f_mse {mine.f_mse:.6e} ({pub.f_mse:.6e}); '
              f'ESS(k) {mine.k_ess:.1f} ({pub.k_ess:.1f}); acc {mine.hmc_acceptance:.4f} ({pub.hmc_acceptance:.4f}); '
              f'max rel. diff of all CSV metrics {worst:.2e}; max abs diff of all npz arrays {arr:.2e}')
    print('\n== tempered runs from the published start: MAP warm start = published MAP (map_k) ==')
    n, worst = 0, 0.0
    folders = ('hmc_tempered', 'hmc_tempered_cal', 'hmc_tempered_L250', 'hmc_tempered_L250_cal')
    for f in sorted(sum((glob.glob(os.path.join(HERE, sub, 'parts', '*', 'noise_study_runs.csv')) for sub in folders), [])):
        mine = pd.read_csv(f).iloc[0]
        pub = published_row(float(mine.noise_level), int(mine.seed), 'mfbpinn_hmc')
        worst = max(worst, abs(mine.map_k - pub.map_k))
        n += 1
    print(f'{n} tempered runs (2 chains x 35); max |map_k - published map_k| = {worst:.2e} '
          '(the 5 chains in hmc_tempered_tmap/ start from the tempered MAP on purpose)')
    print('\n== VI beta = 1 runs at sigma = 0.2, seeds 4-23: same data and MAP warm start as the published runs ==')
    n, worst = 0, 0.0
    for f in sorted(glob.glob(os.path.join(HERE, 'vi_beta1_cal', 'parts', '*', 'noise_study_runs.csv'))):
        mine = pd.read_csv(f).iloc[0]
        pub = published_row(float(mine.noise_level), int(mine.seed), 'mfbpinn_vi')
        worst = max(worst, abs(mine.map_k - pub.map_k))
        n += 1
    print(f'{n} VI(beta = 1) runs; max |map_k - published map_k| = {worst:.2e}')


if __name__ == '__main__':
    main()
