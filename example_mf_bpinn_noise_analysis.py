import argparse
import csv
import shutil
import json
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model_bpinn import BayesianPINN, DeepNeuralNetwork, MultiFidelityNN, LAMBDA_PDE
from model_bnn_vi import BayesianPINNVI, GaussianVIBayesianPINN


plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'legend.loc': 'best',
    'figure.figsize': (7, 5),
    'figure.titlesize': 24,
    'lines.linewidth': 2.5,
    'lines.markersize': 7,
    'image.cmap': 'turbo'
})

print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))


OUTPUT_DIR = './results_bpinn/noise_study_ode_v2'
MODEL_COLORS = {
    'mfpinn': '#1f77b4',
    'mfpinn_det': '#9467bd',
    'mfbpinn_hmc': '#d62728',
    'mfbpinn_vi': '#2ca02c',
}
DISPLAY_NAMES = {
    'mfpinn': 'MC dropout',
    'mfpinn_det': 'Deterministic',
    'mfbpinn_hmc': 'HMC',
    'mfbpinn_vi': 'VI',
}

# Section 4.2 inverse Poisson problem, shared with example_mf_bpinn.py (Fig. pde_inv).
# Domain and u_LF = sin(6x) follow the code and the published figures (main.tex:249/253 state [-1, 1] and sin(8 pi x)).
XMIN, XMAX = -0.7, 0.7
K_TRUE = 0.7            # k in LAMBDA_PDE * u_xx + k * tanh(u) = f
K_INIT = 0.05           # initial k of every model

N_LF = 1000             # low-fidelity points (main.tex:253)
LF_NOISE = 0.01         # low-fidelity noise sd (main.tex:253)
N_HF = 8                # u sensors
N_F = 32                # f sensors (main.tex:253)
GRID_POINTS = 2001      # test grid

LEARNING_RATE = 0.001   # Adam, constant (main.tex:213)
LF_LAYERS = [1, 40, 40, 1]
LF_EPOCHS = 10000

MF_WIDTH = 50           # nonlinear HF head [2, 50, 50, 1]: two hidden layers (main.tex:213)
MF_DEPTH = 2
MF_LINEAR_LAYERS = [2, 1]   # affine linear HF head
L2_LIN = 0.1            # lambda_reg on the linear-head weights
ALPHA = 0.5             # Eq. 3: alpha * F_l + (1 - alpha) * F_nl
MF_EPOCHS = 10000
MC_DROPOUT = 0.05       # dropout on the HF-head hidden layers (main.tex:255)
MF_PREDICT_SAMPLES = 100    # stochastic forward passes (main.tex:255)

BNN_LAYERS = [2, 50, 50, 1]
HMC_SAMPLES = 2000      # main.tex:213
HMC_BURNIN = 2000
HMC_LEAPFROG = 80
HMC_STEP_SIZE = 0.01    # initial step, adapted during the first 80% of burn-in (--no-adapt-step-size: fixed)

VI_STEPS = 10000
VI_TRAIN_SAMPLES = 10   # K in Eq. 13
VI_PREDICT_SAMPLES = 2000
VI_OBJECTIVE = 'elbo'
MAP_STEPS = 10000       # Adam steps of the MAP warm start of HMC and VI (0: legacy initialization)
MAP_RESTARTS = 5        # random initializations of the MAP warm start; the highest posterior is kept
VI_BETA = 0.1          # KL-tempered ELBO; with beta = 1 (exact Eq. 12) q under-fits (sigma >= 0.15: |k err| 0.4-0.5)
VI_FAMILY = 'gaussian'  # Gaussian q over all unknowns with diag + rank-VI_RANK covariance ('legacy_mf': per-layer mean field)
VI_RANK = 10            # rank of the low-rank part of the VI covariance (0: mean-field)

# Bayesian model of the inverse problem (HMC and VI)
POSITIVE_K = True       # log-normal prior on k: log k ~ N(LOGK_PRIOR), so k > 0 (all models: k = exp(log k))
LOGK_PRIOR = (0.0, 1.0)
LEARN_NOISE = False     # True: infer sigma_u, sigma_f (HMC recovers them; VI attributes part of the signal to noise)
NOISE_PRIOR_MEDIAN = 0.1    # log sigma ~ N(log NOISE_PRIOR_MEDIAN, NOISE_PRIOR_STD^2) for sigma_u and sigma_f
NOISE_PRIOR_STD = 1.5

NOISE_LEVELS = [0.01, 0.05, 0.1, 0.15, 0.2]
SEEDS = [1, 2, 3]
PDE_INV_NOISE = [0.01, 0.1, 0.2]    # Fig. pde_inv scenarios (main.tex:255)
PLOT_MODELS = ['mfpinn', 'mfbpinn_hmc', 'mfbpinn_vi']   # Fig. metric_noise (the deterministic model, without dropout, is in metric_vs_noise_all.png)
MODEL_INDEX = {'mfpinn': 0, 'mfpinn_det': 1, 'mfbpinn_hmc': 2, 'mfbpinn_vi': 3}


def ensure_output_dirs(outdir):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'runs'), exist_ok=True)


def low_fidelity_data(x):
    y = np.sin(6 * x)
    return y.astype(np.float32)


def high_fidelity_data(x):
    y = np.sin(6 * x)**3
    return y.astype(np.float32)


def f_function(x, k=K_TRUE):
    f = LAMBDA_PDE * (-108 * np.sin(6 * x) * ((np.sin(6 * x) ** 2) - 2 * np.cos(6 * x) ** 2))
    f += k * np.tanh(np.sin(6 * x) ** 3)
    return f.astype(np.float32)


def likelihood_sigma(noise_level):
    # Gaussian likelihood for u and f with the known noise sd, floored for noise-free data
    return max(float(noise_level), 1e-3)


def run_seed(noise_level, seed):
    # Seed of one (sigma, seed) run; independent of the order in which runs are made
    return int(np.random.SeedSequence([seed, int(round(noise_level * 1e6))]).generate_state(1)[0] % 2**31)


def build_dataset(noise_level, seed, n_lf=N_LF, lf_noise=LF_NOISE, grid_points=GRID_POINTS, common_noise=False):
    # Separate LF, u and f noise streams; independent draws per (sigma, seed) unless common_noise
    entropy = [seed] if common_noise else [seed, int(round(noise_level * 1e6))]
    rng_lf, rng_u, rng_f = [np.random.default_rng(s) for s in np.random.SeedSequence(entropy).spawn(3)]

    x_lf = np.linspace(XMIN, XMAX, n_lf)[:, None].astype(np.float32)
    x_hf = np.linspace(XMIN, XMAX, N_HF)[:, None].astype(np.float32)
    x_f = np.linspace(XMIN, XMAX, N_F)[:, None].astype(np.float32)
    u_lf_noisy = low_fidelity_data(x_lf) + (lf_noise * rng_lf.standard_normal(x_lf.shape)).astype(np.float32)
    u_hf_noisy = high_fidelity_data(x_hf) + (noise_level * rng_u.standard_normal(x_hf.shape)).astype(np.float32)
    f_noisy = f_function(x_f) + (noise_level * rng_f.standard_normal(x_f.shape)).astype(np.float32)

    x_test = np.linspace(XMIN, XMAX, grid_points)[:, None].astype(np.float32)
    u_hf_clean = high_fidelity_data(x_test)
    f_clean = f_function(x_test)

    dataset = {
        'xt_train_lf': tf.convert_to_tensor(x_lf, dtype=tf.float32),
        'u_train_lf': tf.convert_to_tensor(u_lf_noisy, dtype=tf.float32),
        'xt_train_hf': tf.convert_to_tensor(x_hf, dtype=tf.float32),
        'u_train_hf': tf.convert_to_tensor(u_hf_noisy, dtype=tf.float32),
        'xt_train_f': tf.convert_to_tensor(x_f, dtype=tf.float32),
        'f_train': tf.convert_to_tensor(f_noisy, dtype=tf.float32),
        'xt_test': tf.convert_to_tensor(x_test, dtype=tf.float32),
        'u_true': u_hf_clean.reshape(-1),
        'f_true': f_clean.reshape(-1),
        'x_test': x_test.reshape(-1),
        'noise_level': float(noise_level),
        'seed': int(seed),
        'lf_noise': float(lf_noise),
    }
    return dataset


def build_dataset_test(noise_level, seed):
    return build_dataset(noise_level, seed)


def train_low_fidelity_model(dataset, cfg):
    # Same seed in every runner, so each model gets an identical (reproducible) frozen LF network
    tf.keras.utils.set_random_seed(run_seed(dataset['noise_level'], dataset['seed']))
    model_lf = DeepNeuralNetwork(LF_LAYERS)
    model_lf.train(dataset['xt_train_lf'], dataset['u_train_lf'], epochs=cfg.lf_epochs, learning_rate=LEARNING_RATE)
    return model_lf


def set_model_seed(dataset, model_name):
    tf.keras.utils.set_random_seed(run_seed(dataset['noise_level'], dataset['seed']) + 1 + MODEL_INDEX[model_name])


def set_map_seed(dataset):
    # Same seed for the MAP restarts of HMC and VI, so both schemes start from the same MAP estimate
    tf.keras.utils.set_random_seed(run_seed(dataset['noise_level'], dataset['seed']) + 100)


def safe_r2(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator <= 1e-12:
        return np.nan
    return 1.0 - numerator / denominator


def empirical_coverage(y_true, pred_mean, pred_std, scale=2.0):
    lower = pred_mean - scale * pred_std
    upper = pred_mean + scale * pred_std
    return np.mean((y_true >= lower) & (y_true <= upper))


def interval_95(samples):
    # 95% interval from the 2.5/97.5 percentiles of the samples (the band drawn in the figures)
    return np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


def compute_metrics(model_name, noise_level, seed, result, dataset):
    u_true = dataset['u_true']
    f_true = dataset['f_true']
    pred_u_samples = result['u_samples']
    pred_f_samples = result['f_samples']
    nu_mean = result['k_mean']

    pred_u_mean = pred_u_samples.mean(axis=0).reshape(u_true.shape)
    pred_u_std = pred_u_samples.std(axis=0).reshape(u_true.shape)
    pred_f_mean = pred_f_samples.mean(axis=0).reshape(f_true.shape)
    pred_f_std = pred_f_samples.std(axis=0).reshape(f_true.shape)
    u_lo, u_hi = interval_95(pred_u_samples)
    f_lo, f_hi = interval_95(pred_f_samples)

    metrics = {
        'model': model_name,
        'noise_level': float(noise_level),
        'seed': int(seed),
        'u_mse': float(np.mean((pred_u_mean - u_true) ** 2)),
        'u_r2': float(safe_r2(u_true, pred_u_mean)),
        'u_pred_std': float(np.mean(pred_u_std)),
        'u_coverage_2std': float(empirical_coverage(u_true, pred_u_mean, pred_u_std)),
        'u_coverage_95': float(np.mean((u_true >= u_lo) & (u_true <= u_hi))),
        'u_width_95': float(np.mean(u_hi - u_lo)),
        'f_mse': float(np.mean((pred_f_mean - f_true) ** 2)),
        'f_r2': float(safe_r2(f_true, pred_f_mean)),
        'f_pred_std': float(np.mean(pred_f_std)),
        'f_coverage_2std': float(empirical_coverage(f_true, pred_f_mean, pred_f_std)),
        'f_coverage_95': float(np.mean((f_true >= f_lo) & (f_true <= f_hi))),
        'f_width_95': float(np.mean(f_hi - f_lo)),
        'nu_mean': float(nu_mean),
        'nu_std': float(result['k_sd']),
        'nu_error': float(nu_mean - K_TRUE),
        'nu_abs_error': float(abs(nu_mean - K_TRUE)),
        'sigma_obs': float(result['sigma_obs']),
        'train_seconds': float(result['train_seconds']),
        'predict_seconds': float(result['predict_seconds']),
        'hmc_acceptance': float(result.get('hmc_acceptance', np.nan)),
        'hmc_step_size': float(result.get('hmc_step_size', np.nan)),
        'k_ess': float(result.get('k_ess', np.nan)),
        'vi_final_loss': float(result.get('vi_final_loss', np.nan)),
        'vi_kl_drift': float(result.get('vi_kl_drift', np.nan)),   # relative KL change over the last tenth of VI
        'map_k': float(result.get('map_k', np.nan)),   # k of the selected MAP warm start (HMC/VI)
        'sigma_u_mean': float(result.get('sigma_u_mean', np.nan)),   # learned noise levels (HMC/VI with learn_noise)
        'sigma_f_mean': float(result.get('sigma_f_mean', np.nan)),
    }
    return metrics


def mf_options(cfg):
    # Eq. 3 blend (--combine sum: legacy F_nl + F_l) and the affine (or legacy width//2) linear head
    alpha = None if cfg.combine == 'sum' else cfg.alpha
    linear_layers = MF_LINEAR_LAYERS if cfg.linear_head == 'affine' else None
    return {'alpha': alpha, 'linear_layers': linear_layers, 'l2_lambda': cfg.l2_lambda}


def run_pinn_dropout(dataset, cfg, model_name='mfpinn', dropout_rate=None):
    dropout_rate = cfg.mc_dropout if dropout_rate is None else dropout_rate
    model_lf = train_low_fidelity_model(dataset, cfg)
    set_model_seed(dataset, model_name)
    model = MultiFidelityNN(1, MF_WIDTH, MF_DEPTH, 1, nu=tf.Variable(K_INIT, dtype=tf.float32), lf_model=model_lf,
                            dropout_rate=dropout_rate, positive_k=cfg.positive_k, **mf_options(cfg))
    start = time.perf_counter()
    model.train(dataset['xt_train_hf'], dataset['u_train_hf'], dataset['xt_train_f'], dataset['f_train'],
                epochs=cfg.mf_epochs, learning_rate=LEARNING_RATE)
    k_hat = float(model.k_value().numpy())
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    if dropout_rate > 0.0:
        pred_u_samples = model.mc_predict(dataset['xt_test'], num_samples=cfg.mc_passes)
        pred_f_samples = model.mc_predict_function(dataset['xt_test'], num_samples=cfg.mc_passes)
    else:
        pred_u_samples = model.predict(dataset['xt_test']).numpy().reshape(1, -1)
        pred_f_samples = model.predict_function(dataset['xt_test']).numpy().reshape(1, -1)
    predict_seconds = time.perf_counter() - start
    # k is a point estimate (no MC-dropout spread)
    return {'u_samples': pred_u_samples, 'f_samples': pred_f_samples, 'k_samples': np.array([k_hat]),
            'k_mean': k_hat, 'k_sd': np.nan, 'sigma_obs': np.nan,
            'train_seconds': train_seconds, 'predict_seconds': predict_seconds}


def bayes_options(cfg):
    # Prior on k and noise model shared by HMC and VI
    return {'positive_k': cfg.positive_k, 'logk_prior': (cfg.logk_prior_mean, cfg.logk_prior_std),
            'learn_noise': cfg.learn_noise, 'noise_prior': (float(np.log(cfg.noise_prior_median)), cfg.noise_prior_std)}


def noise_summary(model):
    # Posterior means of the learned noise levels (nan with the known noise level)
    samples = getattr(model, 'sigma_samples', None)
    if samples is None:
        return {'sigma_u_mean': np.nan, 'sigma_f_mean': np.nan}
    return {'sigma_u_mean': float(np.mean(samples[:, 0])), 'sigma_f_mean': float(np.mean(samples[:, 1]))}


def run_pinn_deterministic(dataset, cfg):
    # Dropout-free MFPINN with the same heads and trainable k
    return run_pinn_dropout(dataset, cfg, model_name='mfpinn_det', dropout_rate=0.0)


def run_bpinn_hmc(dataset, cfg):
    model_lf = train_low_fidelity_model(dataset, cfg)
    set_model_seed(dataset, 'mfbpinn_hmc')
    sigma_obs = likelihood_sigma(dataset['noise_level'])
    model = BayesianPINN(BNN_LAYERS, nu=tf.Variable(K_INIT, dtype=tf.float32), num_samples=cfg.hmc_samples,
                         num_burnin_steps=cfg.hmc_burnin, lf_model=model_lf,
                         sigma_obs=sigma_obs, residual='ode', **bayes_options(cfg))
    start = time.perf_counter()
    if cfg.map_steps > 0:
        set_map_seed(dataset)
        model.map_estimate(dataset['xt_train_hf'], dataset['u_train_hf'], dataset['xt_train_f'], dataset['f_train'],
                           steps=cfg.map_steps, learning_rate=LEARNING_RATE, restarts=cfg.map_restarts)
        set_model_seed(dataset, 'mfbpinn_hmc')
    model.train(dataset['xt_train_hf'], dataset['u_train_hf'], dataset['xt_train_f'], dataset['f_train'],
                step_size=cfg.hmc_step_size, leapfrog_steps=cfg.hmc_leapfrog, adapt_step_size=cfg.adapt_step_size)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    pred_u_samples, pred_nu_samples = model.samples_predict(dataset['xt_test'])
    pred_f_samples = model.samples_predict_residual(dataset['xt_test'])
    predict_seconds = time.perf_counter() - start
    n_test = len(dataset['x_test'])
    pred_nu_samples = np.asarray(pred_nu_samples, dtype=np.float64).reshape(-1)
    return {'u_samples': np.reshape(pred_u_samples, (-1, n_test)), 'f_samples': np.reshape(pred_f_samples, (-1, n_test)),
            'k_samples': pred_nu_samples, 'k_mean': float(np.mean(pred_nu_samples)),
            'k_sd': float(np.std(pred_nu_samples)), 'sigma_obs': sigma_obs,
            'train_seconds': train_seconds, 'predict_seconds': predict_seconds,
            'hmc_acceptance': model.acceptance_rate, 'hmc_step_size': model.final_step_size, 'k_ess': model.k_ess,
            **noise_summary(model), 'sigma_samples': getattr(model, 'sigma_samples', None),
            'map_k': getattr(model, 'map_nu', np.nan),
            'map_history': np.asarray(getattr(model, 'map_history', []), dtype=np.float64).reshape(-1, 2)}


def run_bpinn_vi(dataset, cfg):
    model_lf = train_low_fidelity_model(dataset, cfg)
    set_model_seed(dataset, 'mfbpinn_vi')
    sigma_obs = likelihood_sigma(dataset['noise_level'])
    if cfg.vi_family == 'legacy_mf':
        # Per-layer mean-field VI with k ~ N(0, 1) and the known noise level (earlier version of the benchmark)
        model = BayesianPINNVI(BNN_LAYERS, nu=tf.Variable(K_INIT, dtype=tf.float32), prior_mean=0.0, prior_std=1.0,
                               lowfidelity_model=model_lf, sigma_obs=sigma_obs, residual='ode')
    else:
        model = GaussianVIBayesianPINN(BNN_LAYERS, nu=tf.Variable(K_INIT, dtype=tf.float32), lf_model=model_lf,
                                       sigma_obs=sigma_obs, residual='ode', rank=cfg.vi_rank, **bayes_options(cfg))
    start = time.perf_counter()
    if cfg.map_steps > 0:
        set_map_seed(dataset)
        model.map_estimate(dataset['xt_train_hf'], dataset['u_train_hf'], dataset['xt_train_f'], dataset['f_train'],
                           steps=cfg.map_steps, learning_rate=LEARNING_RATE, restarts=cfg.map_restarts)
        set_model_seed(dataset, 'mfbpinn_vi')
    model.train(dataset['xt_train_hf'], dataset['u_train_hf'], dataset['xt_train_f'], dataset['f_train'],
                num_steps=cfg.vi_steps, learning_rate=LEARNING_RATE, num_samples=cfg.vi_train_samples,
                objective=cfg.vi_objective, beta=cfg.vi_beta, lr_decay_rate=cfg.vi_lr_decay)
    k_mean, k_sd = model.nu_posterior()
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    pred_u_samples, pred_f_samples, pred_nu_samples = model.samples_predict(dataset['xt_test'], num_samples=cfg.vi_predict_samples)
    predict_seconds = time.perf_counter() - start
    n_test = len(dataset['x_test'])
    # E_q[k] and sd(k) analytically from q(k); the predictive draws of k are kept for the histogram
    return {'u_samples': np.reshape(pred_u_samples, (-1, n_test)), 'f_samples': np.reshape(pred_f_samples, (-1, n_test)),
            'k_samples': np.asarray(pred_nu_samples, dtype=np.float64).reshape(-1), 'k_mean': k_mean, 'k_sd': k_sd,
            'sigma_obs': sigma_obs, 'train_seconds': train_seconds, 'predict_seconds': predict_seconds,
            'vi_final_loss': float(np.mean(model.vi_losses[-100:])), 'vi_kl_drift': getattr(model, 'vi_kl_drift', np.nan),
            **noise_summary(model),
            'sigma_samples': getattr(model, 'sigma_samples', None), 'map_k': getattr(model, 'map_nu', np.nan),
            'map_history': np.asarray(getattr(model, 'map_history', []), dtype=np.float64).reshape(-1, 2)}


MODEL_RUNNERS = {
    'mfpinn': run_pinn_dropout,
    'mfpinn_det': run_pinn_deterministic,
    'mfbpinn_hmc': run_bpinn_hmc,
    'mfbpinn_vi': run_bpinn_vi,
}


def run_npz_path(outdir, model_name, noise_level, seed):
    return os.path.join(outdir, 'runs', f'{model_name}_sigma{noise_level:.3f}_seed{seed}.npz')


def save_run_npz(outdir, model_name, noise_level, seed, dataset, result):
    # Data, predictive summaries, k and HMC diagnostics of one run, so Figs. pde_inv and metric_noise can be
    # redrawn without training
    os.makedirs(os.path.join(outdir, 'runs'), exist_ok=True)
    u_lo, u_hi = interval_95(result['u_samples'])
    f_lo, f_hi = interval_95(result['f_samples'])
    sigma_samples = result.get('sigma_samples')
    sigma_samples = np.zeros((0, 2)) if sigma_samples is None else np.asarray(sigma_samples, dtype=np.float64)
    np.savez(run_npz_path(outdir, model_name, noise_level, seed),
             x_test=dataset['x_test'], u_true=dataset['u_true'], f_true=dataset['f_true'],
             x_u=np.asarray(dataset['xt_train_hf']).reshape(-1), u_obs=np.asarray(dataset['u_train_hf']).reshape(-1),
             x_f=np.asarray(dataset['xt_train_f']).reshape(-1), f_obs=np.asarray(dataset['f_train']).reshape(-1),
             x_lf=np.asarray(dataset['xt_train_lf']).reshape(-1), u_lf=np.asarray(dataset['u_train_lf']).reshape(-1),
             u_mean=result['u_samples'].mean(axis=0), u_std=result['u_samples'].std(axis=0), u_lo=u_lo, u_hi=u_hi,
             f_mean=result['f_samples'].mean(axis=0), f_std=result['f_samples'].std(axis=0), f_lo=f_lo, f_hi=f_hi,
             k_samples=result['k_samples'], k_mean=result['k_mean'], k_sd=result['k_sd'],
             sigma_obs=result['sigma_obs'], noise_level=noise_level, seed=seed,
             hmc_acceptance=result.get('hmc_acceptance', np.nan), hmc_step_size=result.get('hmc_step_size', np.nan),
             k_ess=result.get('k_ess', np.nan), map_k=result.get('map_k', np.nan),
             map_history=result.get('map_history', np.zeros((0, 2))),
             sigma_samples=sigma_samples)   # posterior draws of (sigma_u, sigma_f) with learned noise levels


def load_run_npz(outdir, model_name, noise_level, seed):
    path = run_npz_path(outdir, model_name, noise_level, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found (run the sweep or example_mf_bpinn.py for this model, noise and seed)')
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def nan_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values))


def nan_median(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    return float(np.median(values)) if values.size else np.nan


def summarize_records(records):
    grouped = {}
    for record in records:
        key = (record['model'], record['noise_level'])
        grouped.setdefault(key, []).append(record)

    summary = []
    metric_names = [
        'u_mse', 'u_r2', 'u_pred_std', 'u_coverage_2std', 'u_coverage_95', 'u_width_95',
        'f_mse', 'f_r2', 'f_pred_std', 'f_coverage_2std', 'f_coverage_95', 'f_width_95',
        'nu_mean', 'nu_std', 'nu_error', 'nu_abs_error',
        'train_seconds', 'predict_seconds', 'hmc_acceptance', 'hmc_step_size', 'k_ess', 'vi_final_loss', 'vi_kl_drift',
        'sigma_u_mean', 'sigma_f_mean'
    ]
    for (model, noise_level), items in sorted(grouped.items()):
        row = {'model': model, 'noise_level': noise_level, 'num_runs': len(items)}
        for metric in metric_names:
            values = [item.get(metric, np.nan) for item in items]   # nan for sweeps written before the metric existed
            row[f'{metric}_mean'], row[f'{metric}_std'] = nan_mean_std(values)
            # The median is robust to a seed trapped in a spurious mode (e.g. a negative k)
            row[f'{metric}_median'] = nan_median(values)
        summary.append(row)
    return summary


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    # utf-8-sig also reads the hand-made legacy CSV (BOM); empty rows are skipped
    with open(path, 'r', encoding='utf-8-sig') as csv_file:
        return [row for row in csv.DictReader(csv_file) if row.get('model')]


def write_run_config(path, cfg, extra=None):
    config = dict(vars(cfg))
    legacy = cfg.vi_family == 'legacy_mf'   # legacy VI: k ~ N(0, 1) on k, known noise level, its own MAP
    config.update({
        'domain': [XMIN, XMAX], 'k_true': K_TRUE, 'k_init': K_INIT, 'lambda_pde': LAMBDA_PDE,
        'n_u': N_HF, 'n_f': N_F, 'learning_rate': LEARNING_RATE,
        'optimizer': 'Adam, constant learning rate' if cfg.vi_lr_decay is None else
                     f'Adam, constant learning rate (VI: ExponentialDecay, rate {cfg.vi_lr_decay} per vi_steps/10 steps)',
        'lf_layers': LF_LAYERS, 'mf_nonlinear_layers': [2] + [MF_WIDTH] * MF_DEPTH + [1],
        'mf_linear_layers': MF_LINEAR_LAYERS if cfg.linear_head == 'affine' else [2, MF_WIDTH // 2, 1],
        'bnn_layers': BNN_LAYERS,
        'prior': 'N(0, 1) on all BNN weights and biases; ' + (
            f'log k ~ N({cfg.logk_prior_mean}, {cfg.logk_prior_std}^2)' if cfg.positive_k else 'k ~ N(0, 1)')
            + (' (HMC; legacy_mf VI: k ~ N(0, 1) on k itself)' if legacy else ' (HMC and VI)'),
        'bayesian_init': ((f'MAP warm start (best of {cfg.map_restarts} restarts of {cfg.map_steps} Adam steps, lr '
                           f'{LEARNING_RATE}, on -log posterior from Glorot-uniform weights, zero biases and k = {K_INIT}'
                           + ('; learned noise levels held at the prior median' if cfg.learn_noise else '') + ')'
                           + ('; HMC starts at the MAP; legacy_mf VI: its own MAP over the variational means from '
                              'weight means N(0, 0.1^2), then VI' if legacy else
                              ', the same restarts for HMC and VI; HMC starts at the MAP, VI mean = MAP'))
                          if cfg.map_steps > 0 else
                          f'no warm start: HMC from all-zero network parameters, VI from '
                          + ('weight means N(0, 0.1^2)' if legacy else 'zero network means') + f' (k = {K_INIT})'),
        'k_prior': ((f'log k ~ N({cfg.logk_prior_mean}, {cfg.logk_prior_std}^2) (k = exp(log k) in the MFPINNs and HMC'
                     + ('; legacy_mf VI: k ~ N(0, 1) on k itself)' if legacy else ' and VI)'))
                    if cfg.positive_k else 'k ~ N(0, 1)'),
        'noise_model': ((f'learned: log sigma_u, log sigma_f ~ N(log {cfg.noise_prior_median}, {cfg.noise_prior_std}^2)'
                         + (' (HMC only; legacy_mf VI: known noise level)' if legacy else ''))
                        if cfg.learn_noise else 'known: sigma_u = sigma_f = noise level'),
        'vi': (f'{cfg.vi_family}' + (f', rank {cfg.vi_rank}, analytic KL' if cfg.vi_family == 'gaussian' else '')),
        'likelihood_sigma': ((f'learned sigma_u, sigma_f (noise_model; held at the prior median {cfg.noise_prior_median} '
                              'in the MAP warm start); the CSV/npz sigma_obs is the nominal noise level'
                              + ('; legacy_mf VI: max(noise_level, 1e-3)' if legacy else ''))
                             if cfg.learn_noise else 'max(noise_level, 1e-3) for u and f'),
        'hmc_adaptation': 'TFP SimpleStepSizeAdaptation over the first 80% of burn-in: the step size is multiplied or '
                          'divided by 1.01 per step so that the median per-step acceptance probability tends to 0.9 '
                          '(the mean acceptance realised in each run is hmc_acceptance in the CSV/npz)'
                          if cfg.adapt_step_size else 'none (fixed step size)',
    })
    config.update(extra or {})
    with open(path, 'w') as json_file:
        json.dump(config, json_file, indent=2, default=str)


def model_rows(summary_rows, model_name):
    rows = [row for row in summary_rows if row['model'] == model_name]
    return sorted(rows, key=lambda row: float(row['noise_level']))


def present_models(summary_rows, models):
    return [model_name for model_name in models if any(row['model'] == model_name for row in summary_rows)]


def plot_curves(ax, summary_rows, models, metric_key, run_rows=None):
    # With the per-run records: median over the seeds (line) and every seed (markers), so a run trapped in a spurious
    # mode stays visible without dominating the curve. Summary CSV only (e.g. the legacy one): mean +/- std band.
    std_key = metric_key.replace('_mean', '_std')
    run_key = metric_key[:-len('_mean')] if metric_key.endswith('_mean') else metric_key
    for model_name in present_models(summary_rows, models):
        if run_rows:
            runs = [row for row in run_rows if row['model'] == model_name]
            noise = sorted({float(row['noise_level']) for row in runs})
            values = [np.abs(np.array([float(row[run_key]) for row in runs if float(row['noise_level']) == level]))
                      for level in noise]
            ax.plot(noise, [nan_median(v) for v in values], marker='o', color=MODEL_COLORS.get(model_name),
                    label=DISPLAY_NAMES.get(model_name, model_name))
            for level, v in zip(noise, values):
                ax.scatter(np.full(v.size, level), v, marker='x', s=40, alpha=0.7, zorder=3,
                           color=MODEL_COLORS.get(model_name))
            continue
        rows = model_rows(summary_rows, model_name)
        noise = np.array([float(row['noise_level']) for row in rows])
        # abs() also turns the signed k error of legacy CSVs into an absolute error
        metric = np.array([np.abs(float(row[metric_key])) for row in rows])
        metric_std = np.array([float(row[std_key]) for row in rows])
        num_runs = np.array([int(float(row['num_runs'])) for row in rows])
        ax.plot(noise, metric, marker='o', color=MODEL_COLORS.get(model_name),
                label=DISPLAY_NAMES.get(model_name, model_name))
        if np.any(num_runs > 1):
            ax.fill_between(noise, metric - metric_std, metric + metric_std,
                            color=MODEL_COLORS.get(model_name), alpha=0.2)


def format_noise_axis(ax):
    ax.set_xlabel('Noise level')
    ax.set_xticks([0.0, 0.1, 0.2])
    ax.set_xlim(0.0, 0.21)
    ax.grid(alpha=0.3)


def seed_legend_handles(ax, run_rows):
    handles, labels = ax.get_legend_handles_labels()
    if run_rows:
        handles.append(plt.Line2D([0], [0], marker='x', color='gray', linestyle='None', markersize=8))
        labels.append('Individual seeds')
    return handles, labels


def plot_metric_curves(summary_rows, outdir, models, output_name='metric_vs_noise.png', run_rows=None):
    # Fig. metric_noise. With the per-run records: accuracy (u, f, k; log scale) in the top row and uncertainty quality
    # (95% coverage of the clean truth) and cost (training time of the high-fidelity model) in the bottom row.
    # Summary CSV only (e.g. the legacy one): the three error panels with mean +/- std bands.
    k_key = 'nu_abs_error_mean' if 'nu_abs_error_mean' in summary_rows[0] else 'nu_error_mean'
    metrics = [
        ('u_mse_mean', 'u MSE', 'MSE'),
        ('f_mse_mean', 'f MSE', 'MSE'),
        (k_key, r'$|k - k_{\mathrm{true}}|$', 'Absolute error'),
    ]
    if run_rows and all(key in run_rows[0] for key in ('u_coverage_95', 'f_coverage_95', 'train_seconds')):
        metrics += [
            ('u_coverage_95_mean', 'u 95% coverage', 'Coverage'),
            ('f_coverage_95_mean', 'f 95% coverage', 'Coverage'),
            ('train_seconds_mean', 'Training time', 'Time (s)'),
        ]
    n_rows = 2 if len(metrics) > 3 else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (metric_key, title, ylabel) in zip(axes, metrics):
        plot_curves(ax, summary_rows, models, metric_key, run_rows)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        format_noise_axis(ax)
        if 'coverage' in metric_key:
            ax.axhline(0.95, linestyle='--', color='black', linewidth=1.5)
            ax.set_ylim(0.0, 1.05)
        elif run_rows:
            # Errors and times span several decades (and single seeds can be far off): log scale
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0.0)
    axes[0].legend(*seed_legend_handles(axes[0], run_rows), fontsize=14)
    fig.tight_layout()
    plt.locator_params(axis='x', nbins=3)
    fig.savefig(os.path.join(outdir, 'figures', output_name), dpi=200)
    plt.close(fig)


def plot_uq_curves(summary_rows, outdir, models, output_name='metric_vs_noise_uq.png', run_rows=None):
    # Uncertainty quality: coverage of the clean truth by the 95% band and mean predictive sd
    metrics = [
        ('u_coverage_95_mean', 'u 95% coverage', 'Coverage'),
        ('f_coverage_95_mean', 'f 95% coverage', 'Coverage'),
        ('u_pred_std_mean', 'u predictive std', 'Mean std'),
        ('f_pred_std_mean', 'f predictive std', 'Mean std'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    for ax, (metric_key, title, ylabel) in zip(axes, metrics):
        plot_curves(ax, summary_rows, models, metric_key, run_rows)
        if 'coverage' in metric_key:
            ax.axhline(0.95, linestyle='--', color='black', linewidth=1.5)
            ax.set_ylim(0.0, 1.05)
        else:
            ax.set_ylim(bottom=0.0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        format_noise_axis(ax)
    axes[0].legend(*seed_legend_handles(axes[0], run_rows), fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'figures', output_name), dpi=200)
    plt.close(fig)


def compute_winner_rows(summary_rows):
    noise_levels = sorted({float(row['noise_level']) for row in summary_rows})
    winner_rows = []
    metric_specs = {
        'u_mse_mean': 'min',
        'nu_abs_error_mean': 'min',
        'u_coverage_95_mean': 'target',
    }

    for noise_level in noise_levels:
        rows = [row for row in summary_rows if float(row['noise_level']) == noise_level]
        scores = {row['model']: 0.0 for row in rows}
        for metric, direction in metric_specs.items():
            values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
            if direction == 'target':
                values = np.abs(values - 0.95)
            order = np.argsort(values)
            for rank, idx in enumerate(order):
                scores[rows[idx]['model']] += rank
        winner_model = min(scores, key=scores.get)
        winner_rows.append({
            'noise_level': noise_level,
            'winner_model': winner_model,
            'winner_score': scores[winner_model],
        })
    return winner_rows


def plot_winner_strip(winner_rows, outdir, model_names):
    cmap = plt.matplotlib.colors.ListedColormap([MODEL_COLORS[name] for name in model_names])

    fig, ax = plt.subplots(figsize=(12, 2.5))
    winner_indices = np.array([[model_names.index(row['winner_model']) for row in winner_rows]])
    ax.imshow(winner_indices, aspect='auto', cmap=cmap, vmin=0, vmax=max(len(model_names) - 1, 1))
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(winner_rows)))
    ax.set_xticklabels([f"{row['noise_level']:.2f}" for row in winner_rows], rotation=45)
    ax.set_xlabel('Noise level')
    ax.set_title('Best Model By Noise Level')

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS[name], lw=6, label=DISPLAY_NAMES[name]) for name in model_names]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'figures', 'winner_by_noise.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_error_vs_uncertainty(summary_rows, outdir, models):
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name in present_models(summary_rows, models):
        rows = model_rows(summary_rows, model_name)
        x_values = np.array([row['u_pred_std_mean'] for row in rows])
        y_values = np.array([row['u_mse_mean'] for row in rows])
        labels = [row['noise_level'] for row in rows]
        ax.plot(x_values, y_values, marker='o', color=MODEL_COLORS[model_name], label=DISPLAY_NAMES[model_name])
        for x_value, y_value, label in zip(x_values, y_values, labels):
            ax.annotate(f'{label:.2f}', (x_value, y_value), textcoords='offset points', xytext=(4, 4), fontsize=9)
    ax.set_xlabel('Mean predictive std')
    ax.set_ylabel('U MSE')
    ax.set_title('Error vs Uncertainty')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'figures', 'error_vs_uncertainty.png'), dpi=200)
    plt.close(fig)


def run_noise_study(cfg):
    ensure_output_dirs(cfg.outdir)
    write_run_config(os.path.join(cfg.outdir, 'config.json'), cfg)
    records = []

    for noise_level in cfg.noise_levels:
        for seed in cfg.seeds:
            dataset = build_dataset(noise_level, seed, n_lf=cfg.n_lf, lf_noise=cfg.lf_noise,
                                    grid_points=cfg.grid_points, common_noise=cfg.common_noise)
            for model_name in cfg.models:
                print(f'Running model={model_name}, noise={noise_level:.3f}, seed={seed}')
                result = MODEL_RUNNERS[model_name](dataset, cfg)
                save_run_npz(cfg.outdir, model_name, noise_level, seed, dataset, result)
                metrics = compute_metrics(model_name, noise_level, seed, result, dataset)
                records.append(metrics)
                # Written after every run, so an interrupted sweep keeps its finished runs
                write_csv(os.path.join(cfg.outdir, 'noise_study_runs.csv'), records)
                print(
                    f"  u_mse={metrics['u_mse']:.4e}, "
                    f"f_mse={metrics['f_mse']:.4e}, "
                    f"u_std={metrics['u_pred_std']:.4e}, "
                    f"u_cov95={metrics['u_coverage_95']:.3f}, "
                    f"k={metrics['nu_mean']:.4f}, "
                    f"|k_err|={metrics['nu_abs_error']:.4e}, "
                    f"train={metrics['train_seconds']:.1f}s"
                )

    summary_rows = summarize_records(records)
    winner_rows = compute_winner_rows(summary_rows)

    write_csv(os.path.join(cfg.outdir, 'noise_study_summary.csv'), summary_rows)
    write_csv(os.path.join(cfg.outdir, 'noise_study_winners.csv'), winner_rows)

    # Fig. metric_noise shows the caption models (--plot-models); the other figures show every model that was run,
    # including the dropout-free MFPINN
    all_models = present_models(summary_rows, list(MODEL_RUNNERS))
    plot_metric_curves(summary_rows, cfg.outdir, cfg.plot_models, run_rows=records)
    plot_metric_curves(summary_rows, cfg.outdir, all_models, output_name='metric_vs_noise_all.png', run_rows=records)
    plot_uq_curves(summary_rows, cfg.outdir, all_models, run_rows=records)
    plot_winner_strip(winner_rows, cfg.outdir, all_models)

    print(f'Saved outputs to {cfg.outdir}')


def merge_studies(cfg):
    # Combine sweeps run in parallel (e.g. one process per seed, each with its own --outdir) into cfg.outdir:
    # concatenated runs CSV, npz files, summary, winners and figures, as if written by one sweep
    if not cfg.merge_dirs:
        raise SystemExit('--mode merge needs --merge-dirs')
    ensure_output_dirs(cfg.outdir)
    os.makedirs(os.path.join(cfg.outdir, 'runs'), exist_ok=True)
    records, run_files, reference = [], [], None
    for directory in cfg.merge_dirs:
        with open(os.path.join(directory, 'config.json')) as json_file:
            settings = {k: v for k, v in json.load(json_file).items()
                        if k not in ('noise_levels', 'seeds', 'models', 'outdir', 'plot_models', 'mode', 'merge_dirs',
                                     'merged_from')}
        if reference is None:
            reference = settings
        elif settings != reference:
            raise SystemExit(f'{directory} was run with different settings than {cfg.merge_dirs[0]}')
        records += read_csv(os.path.join(directory, 'noise_study_runs.csv'))
        run_files += [os.path.join(directory, 'runs', n) for n in os.listdir(os.path.join(directory, 'runs'))]
    keys = [(row['model'], float(row['noise_level']), int(float(row['seed']))) for row in records]
    if len(set(keys)) != len(keys):
        raise SystemExit('The merged sweeps contain the same (model, noise level, seed) run more than once')
    for row in records:
        for key, value in row.items():
            if key == 'seed':
                row[key] = int(float(value))
            elif key != 'model':
                row[key] = float(value)
    records.sort(key=lambda row: (row['noise_level'], row['seed'], list(MODEL_RUNNERS).index(row['model'])))
    # Summaries first, so a failure leaves the output folder untouched
    summary_rows = summarize_records(records)
    winner_rows = compute_winner_rows(summary_rows)
    for path in run_files:
        shutil.copy2(path, os.path.join(cfg.outdir, 'runs', os.path.basename(path)))
    # Run settings of the first sweep (all merged sweeps must share them), with the merged grid
    with open(os.path.join(cfg.merge_dirs[0], 'config.json')) as json_file:
        config = json.load(json_file)
    config.update({'noise_levels': sorted({row['noise_level'] for row in records}),
                   'seeds': sorted({int(row['seed']) for row in records}),
                   'models': [m for m in MODEL_RUNNERS if any(row['model'] == m for row in records)],
                   'merged_from': [os.path.abspath(d) for d in cfg.merge_dirs], 'outdir': cfg.outdir, 'mode': 'merge',
                   'merge_dirs': [os.path.abspath(d) for d in cfg.merge_dirs], 'plot_models': cfg.plot_models})
    with open(os.path.join(cfg.outdir, 'config.json'), 'w') as json_file:
        json.dump(config, json_file, indent=2, default=str)
    write_csv(os.path.join(cfg.outdir, 'noise_study_runs.csv'), records)
    write_csv(os.path.join(cfg.outdir, 'noise_study_summary.csv'), summary_rows)
    write_csv(os.path.join(cfg.outdir, 'noise_study_winners.csv'), winner_rows)
    all_models = present_models(summary_rows, list(MODEL_RUNNERS))
    plot_metric_curves(summary_rows, cfg.outdir, cfg.plot_models, run_rows=records)
    plot_metric_curves(summary_rows, cfg.outdir, all_models, output_name='metric_vs_noise_all.png', run_rows=records)
    plot_uq_curves(summary_rows, cfg.outdir, all_models, run_rows=records)
    plot_winner_strip(winner_rows, cfg.outdir, all_models)
    print(f'Merged {len(records)} runs from {len(cfg.merge_dirs)} sweeps into {cfg.outdir}')


def plot_from_csv(cfg):
    # Re-plot Fig. metric_noise from a summary CSV (generated, or the legacy hand-made one); the per-run CSV next to it,
    # when present, gives the median + per-seed version
    csv_path = cfg.csv or os.path.join(cfg.outdir, 'noise_study_summary.csv')
    summary_rows = read_csv(csv_path)
    runs_path = os.path.join(os.path.dirname(csv_path), 'noise_study_runs.csv')
    run_rows = read_csv(runs_path) if os.path.exists(runs_path) else None
    all_models = present_models(summary_rows, list(MODEL_RUNNERS))
    os.makedirs(os.path.join(cfg.outdir, 'figures'), exist_ok=True)
    plot_metric_curves(summary_rows, cfg.outdir, cfg.plot_models, run_rows=run_rows)
    plot_metric_curves(summary_rows, cfg.outdir, all_models, output_name='metric_vs_noise_all.png', run_rows=run_rows)
    if 'u_coverage_95_mean' in summary_rows[0]:
        plot_uq_curves(summary_rows, cfg.outdir, all_models, run_rows=run_rows)
    else:
        print('No 95%-coverage columns in this CSV; metric_vs_noise_uq.png skipped')
    print(f'Saved figures from {csv_path} to {os.path.join(cfg.outdir, "figures")}')


def add_common_args(parser):
    # Section 4.2 settings shared with example_mf_bpinn.py; the defaults are the manuscript values
    parser.add_argument('--lf-epochs', type=int, default=LF_EPOCHS)
    parser.add_argument('--mf-epochs', type=int, default=MF_EPOCHS)
    parser.add_argument('--mc-passes', type=int, default=MF_PREDICT_SAMPLES, help='MC-dropout forward passes')
    parser.add_argument('--mc-dropout', type=float, default=MC_DROPOUT, help='dropout on the HF-head hidden layers')
    parser.add_argument('--hmc-samples', type=int, default=HMC_SAMPLES)
    parser.add_argument('--hmc-burnin', type=int, default=HMC_BURNIN)
    parser.add_argument('--hmc-leapfrog', type=int, default=HMC_LEAPFROG)
    parser.add_argument('--hmc-step-size', type=float, default=HMC_STEP_SIZE)
    parser.add_argument('--no-adapt-step-size', dest='adapt_step_size', action='store_false',
                        help='plain HMC at a fixed step size (default: adapted during burn-in)')
    parser.add_argument('--map-steps', type=int, default=MAP_STEPS,
                        help='Adam steps of the MAP warm start of the HMC chain and the VI means (0: legacy start)')
    parser.add_argument('--positive-k', action=argparse.BooleanOptionalAction, default=POSITIVE_K,
                        help='log-normal prior on k (k = exp(log k) > 0 in all models); --no-positive-k: k ~ N(0, 1)')
    parser.add_argument('--logk-prior-mean', type=float, default=LOGK_PRIOR[0])
    parser.add_argument('--logk-prior-std', type=float, default=LOGK_PRIOR[1])
    parser.add_argument('--learn-noise', action=argparse.BooleanOptionalAction, default=LEARN_NOISE,
                        help='HMC/VI infer sigma_u and sigma_f (log-normal priors); --no-learn-noise: the true noise level')
    parser.add_argument('--noise-prior-median', type=float, default=NOISE_PRIOR_MEDIAN)
    parser.add_argument('--noise-prior-std', type=float, default=NOISE_PRIOR_STD, help='std of log sigma')
    parser.add_argument('--vi-family', choices=['gaussian', 'legacy_mf'], default=VI_FAMILY,
                        help="gaussian: q over all unknowns with diag + low-rank covariance; legacy_mf: per-layer mean field")
    parser.add_argument('--vi-rank', type=int, default=VI_RANK, help='rank of the low-rank covariance part (0: mean field)')
    parser.add_argument('--map-restarts', type=int, default=MAP_RESTARTS,
                        help='random initializations of the MAP warm start (the highest posterior is kept)')
    parser.add_argument('--vi-steps', type=int, default=VI_STEPS)
    parser.add_argument('--vi-train-samples', type=int, default=VI_TRAIN_SAMPLES, help='K in Eq. 13')
    parser.add_argument('--vi-predict-samples', type=int, default=VI_PREDICT_SAMPLES)
    parser.add_argument('--vi-objective', choices=['elbo', 'tempered_mse'], default=VI_OBJECTIVE,
                        help="'tempered_mse' (only with --vi-family legacy_mf): legacy-form MSE_u + MSE_f + 0.9 KL/N loss (N = variational parameters, "
                             "LF frozen, so not an exact reproduction of the old runs)")
    parser.add_argument('--vi-beta', type=float, default=VI_BETA, help='KL weight in the ELBO (1 = exact Eq. 13)')
    parser.add_argument('--vi-lr-decay', type=float, default=None,
                        help='VI learning-rate decay per num_steps/10 steps (legacy: 0.95; default constant)')
    parser.add_argument('--n-lf', type=int, default=N_LF)
    parser.add_argument('--lf-noise', type=float, default=LF_NOISE)
    parser.add_argument('--grid-points', type=int, default=GRID_POINTS)
    parser.add_argument('--common-noise', action='store_true', help='reuse the same noise draws at every noise level')
    parser.add_argument('--alpha', type=float, default=ALPHA, help='Eq. 3 weight of the linear head')
    parser.add_argument('--combine', choices=['convex', 'sum'], default='convex',
                        help="'sum' is the legacy F_nl + F_l")
    parser.add_argument('--linear-head', choices=['affine', 'legacy'], default='affine',
                        help="'legacy' is the [2, width//2, 1] identity stack")
    parser.add_argument('--l2-lambda', type=float, default=L2_LIN, help='lambda_reg on the linear-head weights')
    parser.add_argument('--deterministic', action='store_true', help='enable TensorFlow op determinism')
    return parser


def check_args(parser, args):
    if not 0.0 <= args.alpha <= 1.0:
        parser.error('--alpha must be in [0, 1]')
    if args.map_steps > 0 and args.map_restarts < 1:
        parser.error('--map-restarts must be >= 1 when --map-steps > 0')
    if args.vi_objective != 'elbo' and args.vi_family != 'legacy_mf':
        parser.error("--vi-objective tempered_mse needs --vi-family legacy_mf (the Gaussian VI supports the ELBO only)")
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Noise sweep of the inverse Poisson problem (Fig. metric_noise). '
                    'Relative paths are resolved against the mfbnn folder.')
    parser.add_argument('--mode', choices=['sweep', 'plot', 'merge'], default='sweep',
                        help='merge: combine the runs of sweeps run in parallel (--merge-dirs) into --outdir')
    parser.add_argument('--merge-dirs', nargs='+', default=None, help='sweep output folders to merge (--mode merge)')
    parser.add_argument('--models', nargs='+', choices=list(MODEL_RUNNERS), default=list(MODEL_RUNNERS))
    parser.add_argument('--plot-models', nargs='+', choices=list(MODEL_RUNNERS), default=PLOT_MODELS,
                        help='models of Fig. metric_noise (metric_vs_noise.png); metric_vs_noise_all.png and '
                             'metric_vs_noise_uq.png show every model in the summary')
    parser.add_argument('--noise-levels', nargs='+', type=float, default=NOISE_LEVELS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--outdir', default=None,
                        help=f'default {OUTPUT_DIR}; --mode plot with --csv: {OUTPUT_DIR}/replot_<CSV name>')
    parser.add_argument('--csv', default=None, help='summary CSV for --mode plot (default: OUTDIR/noise_study_summary.csv)')
    add_common_args(parser)
    args = parser.parse_args()
    check_args(parser, args)
    if args.outdir is None:
        args.outdir = OUTPUT_DIR
        if args.mode == 'plot' and args.csv is not None:
            # A re-plot of another CSV gets its own folder, so it never overwrites the sweep's Fig. metric_noise
            args.outdir = os.path.join(OUTPUT_DIR, 'replot_' + os.path.splitext(os.path.basename(args.csv))[0])
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'sweep':
        run_noise_study(args)
    elif args.mode == 'merge':
        merge_studies(args)
    else:
        plot_from_csv(args)
