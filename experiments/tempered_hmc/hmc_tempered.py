"""HMC on the tempered posterior of the Sec. 4.2 inverse Poisson problem (review comment C5).

The VI model of Sec. 4.2 minimizes the KL-tempered objective of Eq. 13b with beta = 0.1, which approximates the
distribution proportional to p(D | theta)^(1/beta) p(theta) (main.tex L144). For the Gaussian likelihoods of u and f
this is, up to a constant, the posterior with the likelihood sd sigma * sqrt(beta) and the same priors (N(0, 1) on all
network parameters, log k ~ N(0, 1)). This driver runs the published HMC runner of
example_mf_bpinn_noise_analysis.py (run_bpinn_hmc) unchanged, except that the likelihood sd used by the HMC chain is
multiplied by sqrt(beta):

  * data, frozen LF network, seeds, MAP warm start (5 restarts x 10000 Adam steps on the UNTEMPERED posterior, i.e.
    exactly the published start shared by HMC and VI), 2000 burn-in + 2000 samples, 80 leapfrog steps, initial step
    0.01 adapted over the first 80% of burn-in to a median acceptance of 0.9: as published (script defaults);
  * HMC chain: likelihood sd = likelihood_sigma(noise) * sqrt(beta).

With --beta 1 the driver reproduces the published HMC runs (validation). With --map-target tempered the MAP warm start
also uses the tempered sd (sensitivity check of the start). --chain-seed-offset changes only the seed of the HMC
chain (momentum draws), for independent chains from the same start. No existing file is modified: the runner is
imported and the model class is replaced at run time by a subclass that only switches sigma_obs.

Outputs per (noise, seed) in --outdir: runs/<model>_sigma<s>_seed<k>.npz (layout of the published runs),
noise_study_runs.csv (metrics of compute_metrics), diag/<model>_sigma<s>_seed<k>.npz (thinned u/f draws, k draws,
chi^2 traces) and diag/*.json (sampler diagnostics).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

BENCH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, BENCH)
import example_mf_bpinn_noise_analysis as E   # noqa: E402  (changes the cwd to BENCH)
import model_bpinn                            # noqa: E402
import tensorflow as tf                       # noqa: E402
import tensorflow_probability as tfp          # noqa: E402


class TemperedHMCBayesianPINN(model_bpinn.BayesianPINN):
    """BayesianPINN whose HMC chain (train) uses the likelihood sd sigma_obs * sqrt(beta); the MAP warm start
    (map_estimate) uses the untempered sd unless map_tempered. Everything else is inherited unchanged."""
    beta = 1.0
    map_tempered = False
    last = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_untempered = self.sigma_obs
        self.sigma_tempered = tf.convert_to_tensor(float(self.sigma_obs.numpy()) * np.sqrt(self.beta), tf.float32)
        TemperedHMCBayesianPINN.last = self

    def map_estimate(self, *args, **kwargs):
        self.sigma_obs = self.sigma_tempered if self.map_tempered else self.sigma_untempered
        return super().map_estimate(*args, **kwargs)

    def train(self, *args, **kwargs):
        self.sigma_obs = self.sigma_tempered
        return super().train(*args, **kwargs)


def ess(x):
    # Geyer initial positive sequence, as for k in model_bpinn (x: samples along axis 0)
    return np.asarray(tfp.mcmc.effective_sample_size(tf.convert_to_tensor(np.asarray(x, np.float64)),
                                                     filter_beyond_positive_pairs=True))


def split_rhat(x):
    # Split-R-hat of one chain (two halves); x: (n,) or (n, m)
    x = np.asarray(x, np.float64)
    n = x.shape[0] // 2
    halves = np.stack([x[:n], x[n:2 * n]])               # (2, n, ...)
    w = halves.var(axis=1, ddof=1).mean(axis=0)
    b = n * halves.mean(axis=1).var(axis=0, ddof=1)
    var_plus = (n - 1) / n * w + b / n
    return np.sqrt(var_plus / w)


def chi2_traces(model, dataset):
    # Untempered chi^2 of the u and f observations for every posterior draw (mixing diagnostic of the network)
    x_u, u, x_f, f = (dataset['xt_train_hf'], dataset['u_train_hf'], dataset['xt_train_f'], dataset['f_train'])
    s = float(model.sigma_untempered.numpy())
    cu, cf = [], []
    for state in model.samples:
        pu = model.model(x_u, state[0])
        pf = model.physical_residual(x_f, state[0], model.k_from_state(state[1]))
        cu.append(float(tf.reduce_sum(tf.square((pu - u) / s))))
        cf.append(float(tf.reduce_sum(tf.square((pf - f) / s))))
    return np.asarray(cu), np.asarray(cf)


def make_cfg(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--noise-levels', nargs='+', type=float, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--beta', type=float, default=0.1, help='tempering: HMC likelihood sd x sqrt(beta)')
    parser.add_argument('--map-target', choices=['untempered', 'tempered'], default='untempered')
    parser.add_argument('--chain-seed-offset', type=int, default=0)
    parser.add_argument('--model-name', default=None, help='default mfbpinn_hmc (beta 1) or mfbpinn_hmc_tempered')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--thin', type=int, default=20, help='test-grid stride of the saved u/f draws')
    E.add_common_args(parser)        # the published defaults (hmc_samples, hmc_burnin, map_steps, ...)
    cfg = parser.parse_args(argv)
    if cfg.model_name is None:
        cfg.model_name = 'mfbpinn_hmc' if cfg.beta == 1.0 else 'mfbpinn_hmc_tempered'
    if not os.path.isabs(cfg.outdir):
        raise SystemExit('--outdir must be absolute (the imported script changes the working directory)')
    return cfg


def main():
    cfg = make_cfg()
    TemperedHMCBayesianPINN.beta = cfg.beta
    TemperedHMCBayesianPINN.map_tempered = cfg.map_target == 'tempered'
    E.BayesianPINN = TemperedHMCBayesianPINN       # run_bpinn_hmc builds this class; nothing else changes
    if cfg.chain_seed_offset:
        published_set_model_seed = E.set_model_seed

        def set_model_seed(dataset, model_name):
            # called before the model is built (no random op) and after the MAP: shifts only the chain seed
            tf.keras.utils.set_random_seed(E.run_seed(dataset['noise_level'], dataset['seed']) + 1
                                           + E.MODEL_INDEX[model_name] + cfg.chain_seed_offset)
        E.set_model_seed = set_model_seed
        assert published_set_model_seed is not set_model_seed

    os.makedirs(os.path.join(cfg.outdir, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(cfg.outdir, 'diag'), exist_ok=True)
    config = dict(vars(cfg))
    config.update({'driver': os.path.abspath(__file__), 'runner': 'example_mf_bpinn_noise_analysis.run_bpinn_hmc',
                   'hmc_likelihood_sd': 'likelihood_sigma(noise) * sqrt(beta)',
                   'map_likelihood_sd': ('likelihood_sigma(noise) * sqrt(beta)' if cfg.map_target == 'tempered'
                                         else 'likelihood_sigma(noise) (published MAP warm start)'),
                   'tf_threads': (os.environ.get('TF_NUM_INTRAOP_THREADS'), os.environ.get('TF_NUM_INTEROP_THREADS'))})
    with open(os.path.join(cfg.outdir, 'config.json'), 'w') as fh:
        json.dump(config, fh, indent=2, default=str)

    records = []
    for noise_level in cfg.noise_levels:
        for seed in cfg.seeds:
            dataset = E.build_dataset(noise_level, seed, n_lf=cfg.n_lf, lf_noise=cfg.lf_noise,
                                      grid_points=cfg.grid_points, common_noise=cfg.common_noise)
            print(f'Running {cfg.model_name}: beta={cfg.beta}, noise={noise_level:.3f}, seed={seed}', flush=True)
            t0 = time.perf_counter()
            result = E.run_bpinn_hmc(dataset, cfg)
            wall = time.perf_counter() - t0
            model = TemperedHMCBayesianPINN.last
            E.save_run_npz(cfg.outdir, cfg.model_name, noise_level, seed, dataset, result)
            metrics = E.compute_metrics(cfg.model_name, noise_level, seed, result, dataset)
            metrics['sigma_hmc'] = float(model.sigma_tempered.numpy())
            metrics['beta'] = cfg.beta
            metrics['burnin_acceptance'] = float(model.burnin_acceptance)
            k = np.asarray(result['k_samples'], np.float64)
            lo, hi = np.percentile(k, [2.5, 97.5])
            metrics['k_lo95'], metrics['k_hi95'] = float(lo), float(hi)
            metrics['k_in95'] = int(lo <= E.K_TRUE <= hi)
            metrics['k_split_rhat'] = float(split_rhat(k))
            thin = slice(None, None, cfg.thin)
            u_thin = result['u_samples'][:, thin]
            f_thin = result['f_samples'][:, thin]
            ess_u, ess_f = ess(u_thin), ess(f_thin)
            chi2_u, chi2_f = chi2_traces(model, dataset)
            ess_chi2 = ess(np.stack([chi2_u, chi2_f, chi2_u + chi2_f], axis=1))
            metrics.update({'ess_u_min': float(ess_u.min()), 'ess_u_median': float(np.median(ess_u)),
                            'ess_f_min': float(ess_f.min()), 'ess_f_median': float(np.median(ess_f)),
                            'rhat_u_max': float(np.max(split_rhat(u_thin))),
                            'rhat_f_max': float(np.max(split_rhat(f_thin))),
                            'ess_chi2_total': float(ess_chi2[2]), 'chi2_u_mean': float(chi2_u.mean()),
                            'chi2_f_mean': float(chi2_f.mean()), 'wall_seconds': wall})
            records.append(metrics)
            E.write_csv(os.path.join(cfg.outdir, 'noise_study_runs.csv'), records)
            stem = f'{cfg.model_name}_sigma{noise_level:.3f}_seed{seed}'
            np.savez(os.path.join(cfg.outdir, 'diag', stem + '.npz'), x_thin=dataset['x_test'][thin],
                     u_thin=u_thin.astype(np.float32), f_thin=f_thin.astype(np.float32), k=k,
                     chi2_u=chi2_u, chi2_f=chi2_f, ess_u=ess_u, ess_f=ess_f)
            with open(os.path.join(cfg.outdir, 'diag', stem + '.json'), 'w') as fh:
                json.dump(metrics, fh, indent=2)
            print(f"  k={metrics['nu_mean']:.4f}+-{metrics['nu_std']:.4f} in95={metrics['k_in95']} "
                  f"u_mse={metrics['u_mse']:.3e} f_mse={metrics['f_mse']:.3e} cov_u={metrics['u_coverage_95']:.3f} "
                  f"cov_f={metrics['f_coverage_95']:.3f} acc={metrics['hmc_acceptance']:.3f} "
                  f"step={metrics['hmc_step_size']:.2e} ess_k={metrics['k_ess']:.0f} "
                  f"ess_u_min={metrics['ess_u_min']:.0f} ess_f_min={metrics['ess_f_min']:.0f} "
                  f"map_k={metrics['map_k']:.6f} wall={wall:.0f}s", flush=True)


if __name__ == '__main__':
    main()
