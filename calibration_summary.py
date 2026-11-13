import argparse
import os

import numpy as np

from example_mf_bpinn_noise_analysis import (K_TRUE, MODEL_RUNNERS, build_dataset, high_fidelity_data, load_run_npz,
                                             read_csv)

# Sec. 4.2 calibration at one noise level over many noise realizations (results_bpinn/calibration_sigma0.2_v2,
# seeds 4-23): coverage of the true u and f by the 95% bands, coverage of the true k by the 95% interval of the k
# draws, standardized k errors, and the noise of the u observations of every realization (incl. the Fig. pde_inv seeds)


def main():
    parser = argparse.ArgumentParser(description='Calibration summary of a Sec. 4.2 sweep at one noise level')
    parser.add_argument('--sweep', default='./results_bpinn/calibration_sigma0.2_v2')
    parser.add_argument('--noise', type=float, default=0.2)
    parser.add_argument('--noise-seeds', nargs='+', type=int, default=list(range(1, 24)),
                        help='realizations whose u-observation noise is ranked')
    args = parser.parse_args()

    rows = [r for r in read_csv(os.path.join(args.sweep, 'noise_study_runs.csv'))
            if abs(float(r['noise_level']) - args.noise) < 1e-9]
    for model in MODEL_RUNNERS:
        runs = [r for r in rows if r['model'] == model]
        if not runs:
            continue
        cov_u = np.array([float(r['u_coverage_95']) for r in runs])
        cov_f = np.array([float(r['f_coverage_95']) for r in runs])
        k_err = np.array([float(r['nu_error']) for r in runs])
        line = (f'{model:12s} {len(runs)} runs: 95% coverage of true u mean {cov_u.mean():.3f} (median '
                f'{np.median(cov_u):.3f}), of true f mean {cov_f.mean():.3f} (median {np.median(cov_f):.3f}); '
                f'k error mean {k_err.mean():+.3f}, median |k error| {np.median(np.abs(k_err)):.3f}')
        k_sd = np.array([float(r['nu_std']) for r in runs])
        if np.all(np.isfinite(k_sd)):
            z = k_err / k_sd
            inside = 0
            for r in runs:
                k = load_run_npz(args.sweep, model, args.noise, int(float(r['seed'])))['k_samples']
                lo, hi = np.percentile(k, [2.5, 97.5])
                inside += lo <= K_TRUE <= hi
            line += f'; 95% k interval contains k_true in {inside}/{len(runs)}; z_k mean {z.mean():+.2f}, sd {z.std():.2f}'
        print(line)

    # Noise of the u observations, in units of sigma
    noise = []
    for seed in args.noise_seeds:
        d = build_dataset(args.noise, seed)
        e = (np.asarray(d['u_train_hf']).ravel() - high_fidelity_data(np.asarray(d['xt_train_hf'])).ravel()) / args.noise
        noise.append((float(np.sqrt(np.mean(e ** 2))), seed, int(np.sum(np.abs(e) > 1.9))))
    noise.sort(reverse=True)
    print(f'u-observation noise RMS/sigma over seeds {args.noise_seeds[0]}-{args.noise_seeds[-1]}: median '
          f'{np.median([n[0] for n in noise]):.2f}; largest: ' +
          ', '.join(f'seed {s} {rms:.2f} ({n} of 8 beyond 1.9 sigma)' for rms, s, n in noise[:3]))


if __name__ == '__main__':
    main()
