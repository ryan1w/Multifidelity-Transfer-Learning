import argparse
import csv
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_probability as tfp

from model_bnn import BayesianNeuralNetwork, DeepNeuralNetwork, MultiFidelityNN

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

np.random.seed(42)
tf.random.set_seed(42)
print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'legend.loc': 'lower right',
    'figure.figsize': (10, 6),
    'figure.titlesize': 24,
    'lines.linewidth': 4,
    'lines.markersize': 14
})

# 14 unevenly distributed high-fidelity points of Fig. 2(c)
x_data = np.array([0.0803396, 0.1645980, 0.2442850, 0.2645330, 0.3487920,
0.3873290, 0.5088180, 0.5290660, 0.6100590, 0.6525150, 0.7537560, 0.7942520,
0.9163950, 0.9784450]).reshape(-1, 1).astype(np.float32)

# Data settings of Fig. 1 (main.tex:223, 230) and Fig. 2 (main.tex:239):
# (sigma_L, sigma_H, figure, HF layout). The number of LF points is set per figure (--n-lf-fig1, --n-lf-fig2).
SCENARIOS = {
    'fig1a': (0.01, 0.01, 'fig1', 'uniform'),
    'fig1b': (0.05, 0.01, 'fig1', 'uniform'),
    'fig1c': (0.05, 0.05, 'fig1', 'uniform'),
    'fig2a': (0.3, 0.01, 'fig2', 'uniform'),
    'fig2b': (0.3, 0.1, 'fig2', 'uniform'),
    'fig2c': (0.3, 0.1, 'fig2', 'uneven'),
}
# Composite figures: columns (a)-(c) in caption order, rows deterministic (top) and HMC (bottom)
FIGURES = {
    'fig1': ('function_01.jpeg', ['fig1a', 'fig1b', 'fig1c']),
    'fig2': ('function_02.jpeg', ['fig2a', 'fig2b', 'fig2c']),
}
MODELS = ['mfnn', 'mfbnn_hmc']
ROW_LABELS = {'mfnn': 'Deterministic', 'mfbnn_hmc': 'HMC'}
# Each model is seeded with --seed + offset (the data and the LF network use --seed), so results do not
# depend on the --model order and no two networks share initializer seeds
MODEL_SEED_OFFSET = {'mfnn': 1, 'mfbnn_hmc': 2}

LF_LAYERS = [1] + [50] * 2 + [1]   # low-fidelity DNN
HF_LAYERS = [2] + [20] * 2 + [1]   # MFNN nonlinear head and Bayesian HF network, input [x, y_LF(x)]
N_HF = 14
N_TEST = 500
SCENARIO_KEYS = ['scenario', 'sigma_l', 'sigma_h', 'n_lf', 'layout', 'seed']
# CLI options that only select runs or name outputs; the runs shown in one row of a composite must agree on all others
RUN_SELECTION_ARGS = ['scenario', 'model', 'outdir', 'figures_only', 'show', 'deterministic']

def low_fidelity_data(x):
    y = np.sin(8 * np.pi * x)
    # y = 0.5*(6*x - 2)**2*np.sin(12*x - 4) + 10*(x - 0.5) - 5
    return y

def high_fidelity_data(x):
    y = (x - np.sqrt(2.0)) * low_fidelity_data(x)**2
    # y = (6*x - 2)**2*np.sin(12*x - 4)
    return y

def add_noise(x, noise_level=0.01):
    y = x + noise_level * np.random.normal(size=x.shape).astype(np.float32)
    return y

def generate_data(scn, n_lf, seed=42):
    # Legacy draw order: seed, then the (n_lf, 1) LF noise, then the (14, 1) HF noise; set_random_seed also calls
    # np.random.seed(seed). With n_lf=51 this regenerates the published Fig. 1 HF points. The published Fig. 2 used
    # n_lf=10001, so with the default n_lf=2000 its HF noise realisation differs from the published one.
    sigma_l, sigma_h, _, layout = SCENARIOS[scn]
    tf.keras.utils.set_random_seed(seed)
    x_lf = np.linspace(0, 1, n_lf)[:, None].astype(np.float32)
    y_lf = add_noise(low_fidelity_data(x_lf), noise_level=sigma_l).astype(np.float32)
    x_hf = x_data if layout == 'uneven' else np.linspace(0, 1, N_HF)[:, None].astype(np.float32)
    y_hf = add_noise(high_fidelity_data(x_hf), noise_level=sigma_h).astype(np.float32)
    return x_lf, y_lf, x_hf, y_hf

def training_low_fidelity(x_lf, y_lf, epochs=20000, learning_rate=0.001):
    model_lf = DeepNeuralNetwork(LF_LAYERS)
    model_lf.train(x_lf, y_lf, epochs=epochs, learning_rate=learning_rate)
    return model_lf

def build_mfnn(model_lf, args):
    return MultiFidelityNN(1, 20, 2, 1, lf_model=model_lf,
                           alpha=None if args.combine == 'sum' else args.alpha,
                           nonlinear_layers=HF_LAYERS,
                           linear_layers=[2, 1] if args.linear_head == 'affine' else None,
                           l2_lambda=args.l2_lambda)

def build_mfbnn_hmc(sigma_h, args):
    sigma_obs = sigma_h if args.likelihood_sigma == 'noise' else args.sigma_fixed
    return BayesianNeuralNetwork(HF_LAYERS, num_samples=args.hmc_samples, num_burnin_steps=args.hmc_burnin,
                                 sigma_obs=sigma_obs, prior_std=args.prior_std)

def run_mfnn(model_lf, data, args):
    model_mf = build_mfnn(model_lf, args)
    model_mf.train(data['x_lf'], data['y_lf'], data['x_hf'], data['y_hf'], epochs=args.mf_epochs, learning_rate=args.lr)
    arrays = {'mean': model_mf.predict(data['x_test']).numpy().ravel()}
    info = {'nonlinear_layers': list(model_mf.nonlinear_model.layer_sizes), 'linear_layers': list(model_mf.linear_model.layer_sizes),
            'alpha': model_mf.alpha, 'l2_lambda': model_mf.l2_lambda}
    return arrays, info

def mixing_diagnostics(sampled_predictions):
    # Mixing of the single HMC chain, measured on f(x) at the test points: effective sample size (ESS) and
    # split-R-hat (the two halves of the chain compared as two chains; needs at least 4 samples)
    preds = tf.constant(sampled_predictions)
    level = tf.get_logger().level
    tf.get_logger().setLevel('ERROR')   # tfp's FFT autocorrelation logs a harmless complex-to-real cast warning
    # Geyer's initial positive sequence (TFP's default returns exactly N at points with a negative lag-1 autocorrelation)
    ess = tfp.mcmc.effective_sample_size(preds, filter_beyond_positive_pairs=True).numpy()
    tf.get_logger().setLevel(level)
    rhat = tfp.mcmc.potential_scale_reduction(preds[:, None], split_chains=True).numpy() if len(preds) >= 4 else np.nan
    return {'ess_min': float(np.min(ess)), 'ess_median': float(np.median(ess)), 'split_rhat_max': float(np.max(rhat))}

def run_mfbnn_hmc(model_lf, data, sigma_h, args):
    model_bnn = build_mfbnn_hmc(sigma_h, args)
    x_train = np.hstack((data['x_hf'], model_lf.predict(data['x_hf']).numpy()))
    model_bnn.train(x_train, data['y_hf'], step_size=args.hmc_step_size, leapfrog_steps=args.hmc_leapfrog,
                    adapt_step_size=not args.no_adapt_step_size, target_accept_prob=args.target_accept_prob)

    x_test_bnn = np.hstack((data['x_test'], data['lf_pred_test']))
    sampled_predictions = model_bnn.samples_predict(x_test_bnn).reshape(len(model_bnn.samples), -1)
    lo, hi = np.percentile(sampled_predictions, [2.5, 97.5], axis=0)   # 95% credible interval
    arrays = {'mean': sampled_predictions.mean(axis=0), 'lo': lo, 'hi': hi, 'samples': model_bnn.samples}
    info = {'bnn_layers': HF_LAYERS, 'sigma_obs': model_bnn.sigma_obs, 'prior_std': model_bnn.prior_std,
            'num_samples_kept': int(len(model_bnn.samples)), 'acceptance_rate': model_bnn.acceptance_rate,
            'burnin_acceptance': model_bnn.burnin_acceptance, 'final_step_size': model_bnn.final_step_size,
            **mixing_diagnostics(sampled_predictions)}
    # A well-mixed chain has ESS >= 100 and split-R-hat <= 1.1 at every test point (NaN fails both tests)
    if not (info['ess_min'] >= 100 and info['split_rhat_max'] <= 1.1):
        print(f'Warning: the HMC chain mixes poorly (ESS of f(x) min {info["ess_min"]:.0f} of {len(sampled_predictions)} '
              f'samples, split-R-hat max {info["split_rhat_max"]:.2f}); the 95% band and its coverage may be unreliable. '
              'Longer trajectories (--hmc-leapfrog) or more samples (--hmc-samples) may improve mixing.')
    return arrays, info

def compute_metrics(y_true, mean, lo=None, hi=None):
    y_true, mean = y_true.ravel(), mean.ravel()
    metrics = {'rmse': float(np.sqrt(np.mean((mean - y_true)**2))), 'max_abs_error': float(np.max(np.abs(mean - y_true)))}
    if lo is not None:
        metrics['coverage_95'] = float(np.mean((lo <= y_true) & (y_true <= hi)))
        metrics['mean_width_95'] = float(np.mean(hi - lo))
    return metrics

def plot_panel(ax, d, model):
    ax.scatter(d['x_hf'], d['y_hf'], color='black', label='High fidelity data')
    ax.plot(d['x_test'], d['y_true'], linestyle='--', color='green',  label='True function')
    if model == 'mfnn':
        ax.plot(d['x_test'], d['mean'], color='red', label='Prediction')
    else:
        ax.plot(d['x_test'], d['mean'], color='red', label='Posterior mean')
        ax.fill_between(d['x_test'].ravel(), d['lo'], d['hi'], color='red', alpha=0.3, label='95% credible interval')
    ax.legend()

def run_scenario(scn, models, args):
    sigma_l, sigma_h, figure, layout = SCENARIOS[scn]
    n_lf = args.n_lf_fig1 if figure == 'fig1' else args.n_lf_fig2
    settings = {'scenario': scn, 'figure': figure, 'sigma_l': sigma_l, 'sigma_h': sigma_h, 'n_lf': n_lf,
                'layout': layout, 'seed': args.seed}
    print(f'=== {scn}: sigma_L={sigma_l}, sigma_H={sigma_h}, N_LF={n_lf}, HF layout {layout}, seed {args.seed}')

    x_lf, y_lf, x_hf, y_hf = generate_data(scn, n_lf, args.seed)
    x_test = np.linspace(0., 1., N_TEST)[:, None].astype(np.float32)
    y_true = high_fidelity_data(x_test).astype(np.float32)

    # One low-fidelity network per scenario, shared by both models
    t0 = time.time()
    model_lf = training_low_fidelity(x_lf, y_lf, epochs=args.lf_epochs, learning_rate=args.lr)
    lf_time = time.time() - t0
    lf_pred_test = model_lf.predict(x_test).numpy()
    lf_rmse = float(np.sqrt(np.mean((lf_pred_test - low_fidelity_data(x_test))**2)))
    data = {'x_test': x_test, 'y_true': y_true, 'x_lf': x_lf, 'y_lf': y_lf, 'x_hf': x_hf, 'y_hf': y_hf,
            'lf_pred_test': lf_pred_test}

    for model in models:
        model_seed = args.seed + MODEL_SEED_OFFSET[model]
        tf.keras.utils.set_random_seed(model_seed)
        t0 = time.time()
        if model == 'mfnn':
            arrays, info = run_mfnn(model_lf, data, args)
        else:
            arrays, info = run_mfbnn_hmc(model_lf, data, sigma_h, args)
        train_time = time.time() - t0
        metrics = compute_metrics(y_true, arrays['mean'], arrays.get('lo'), arrays.get('hi'))
        metrics['lf_rmse'] = lf_rmse

        out = {k: np.asarray(v, dtype=np.float32) for k, v in {**data, **arrays}.items()}
        out.update(settings)
        stem = os.path.join(args.outdir, f'{scn}_{model}')
        np.savez(stem + '.npz', **out)
        config = {**settings, 'model': model, 'model_seed': model_seed, 'lf_layers': LF_LAYERS, **info,
                  'metrics': metrics, 'lf_train_time_s': lf_time, 'train_time_s': train_time, 'args': vars(args)}
        with open(stem + '.json', 'w') as f:
            json.dump(config, f, indent=2)

        fig = plt.figure(figsize=(10, 6))
        plot_panel(fig.gca(), out, model)
        fig.savefig(stem + '.png')
        if args.show:
            plt.show()
        plt.close(fig)
        print(f'{scn} {model}: RMSE {metrics["rmse"]:.4f}, LF RMSE {lf_rmse:.4f}'
              + (f', 95% coverage {metrics["coverage_95"]:.3f}, mean width {metrics["mean_width_95"]:.4f}, '
                 f'acceptance {info["acceptance_rate"]:.3f}, step size {info["final_step_size"]:.4g}, '
                 f'ESS min {info["ess_min"]:.0f}, split-R-hat max {info["split_rhat_max"]:.2f}' if model != 'mfnn' else '')
              + f' -> {stem}.png')

def settings_diff(configs):
    # Saved CLI options (except RUN_SELECTION_ARGS) that differ between the given runs
    settings = [{k: v for k, v in c['args'].items() if k not in RUN_SELECTION_ARGS} for c in configs]
    return sorted({k for s in settings for k in s if any(t.get(k) != s[k] for t in settings)})

def write_metrics_csv(outdir):
    rows, configs = [], []
    for scn in SCENARIOS:
        for model in MODELS:
            path = os.path.join(outdir, f'{scn}_{model}.json')
            if not os.path.exists(path):
                continue
            with open(path) as f:
                c = json.load(f)
            configs.append(c)
            m = c['metrics']
            rows.append({'scenario': scn, 'figure': c['figure'], 'model': model, 'sigma_L': c['sigma_l'],
                         'sigma_H': c['sigma_h'], 'n_lf': c['n_lf'], 'hf_layout': c['layout'], 'seed': c['seed'],
                         'rmse': m['rmse'], 'max_abs_error': m['max_abs_error'],
                         'coverage_95': m.get('coverage_95', ''), 'mean_width_95': m.get('mean_width_95', ''),
                         'lf_rmse': m['lf_rmse'], 'sigma_obs': c.get('sigma_obs', ''),
                         'hmc_acceptance': c.get('acceptance_rate', ''), 'hmc_burnin_acceptance': c.get('burnin_acceptance', ''),
                         'hmc_final_step_size': c.get('final_step_size', ''), 'hmc_ess_min': c.get('ess_min', ''),
                         'hmc_ess_median': c.get('ess_median', ''), 'hmc_split_rhat_max': c.get('split_rhat_max', ''),
                         'train_time_s': c['train_time_s']})
    if not rows:
        print(f'No runs found in {outdir}; function_metrics.csv not written')
        return
    # Partial re-runs into the same outdir can mix configurations (make_function_figures refuses such composites)
    for figure in FIGURES:
        for model in MODELS:
            diff = settings_diff([c for c in configs if c['figure'] == figure and c['model'] == model])
            if diff:
                print(f'Warning: the {figure} {model} runs in {outdir} used different options: {", ".join(diff)}')
    path = os.path.join(outdir, 'function_metrics.csv')
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {path} ({len(rows)} runs)')

def make_function_figures(outdir, show=False):
    # Builds function_01.jpeg (Fig. 1) and function_02.jpeg (Fig. 2) from the saved runs, 3200x1500 px like the published composites
    refused = []
    for filename, columns in FIGURES.values():
        stems = {(scn, model): os.path.join(outdir, f'{scn}_{model}') for scn in columns for model in MODELS}
        missing = [os.path.basename(s + ext) for s in stems.values() for ext in ('.npz', '.json') if not os.path.exists(s + ext)]
        if missing:
            print(f'Skipping {filename}: missing {", ".join(missing)}')
            continue
        # An inconsistent figure is skipped like a missing one, so the other figure is still built
        try:
            configs = {}
            for key, stem in stems.items():
                with open(stem + '.json') as f:
                    configs[key] = json.load(f)
            # The panels of a row must come from runs with the same options (a partial re-run could have replaced some)
            diffs = {model: settings_diff([configs[(scn, model)] for scn in columns]) for model in MODELS}
            assert not any(diffs.values()), f'{filename}: the panels of a row were run with different options {diffs}'

            runs_by_scn = {}
            for scn in columns:
                runs = {}
                for model in MODELS:
                    with np.load(stems[(scn, model)] + '.npz') as npz:
                        runs[model] = dict(npz)
                # Both rows of a column must show the same data setting and LF network (the hand-made Fig. 1 had swapped panels)
                ref = runs[MODELS[0]]
                for model in MODELS[1:]:
                    assert all(np.array_equal(ref[k], runs[model][k]) for k in ('x_hf', 'y_hf')), \
                        f'{scn}: {MODELS[0]} and {model} runs used different HF data'
                    assert all(ref[k] == runs[model][k] for k in SCENARIO_KEYS), \
                        f'{scn}: {MODELS[0]} and {model} runs have different scenario settings'
                    assert np.array_equal(ref['lf_pred_test'], runs[model]['lf_pred_test']), \
                        f'{scn}: {MODELS[0]} and {model} runs used different LF networks; re-run {scn} with --model both'
                assert str(ref['scenario']) == scn, f'{stems[(scn, MODELS[0])]}.npz holds scenario {ref["scenario"]}'
                runs_by_scn[scn] = runs
        except AssertionError as err:
            print(f'Skipping {filename}: {err}')
            refused.append(filename)
            continue

        fig, axes = plt.subplots(2, 3, figsize=(32, 15), dpi=100)
        fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.15, wspace=0.15, hspace=0.2)
        for j, scn in enumerate(columns):
            for i, model in enumerate(MODELS):
                plot_panel(axes[i, j], runs_by_scn[scn][model], model)

        for i, model in enumerate(MODELS):
            pos = axes[i, 0].get_position()
            fig.text(0.045, 0.5 * (pos.y0 + pos.y1), ROW_LABELS[model], fontsize=48, ha='center', va='center',
                     rotation=90)
        for j in range(len(columns)):
            pos = axes[1, j].get_position()
            fig.text(0.5 * (pos.x0 + pos.x1), 0.04, f'({"abc"[j]})', fontsize=60, ha='center', va='center')
        path = os.path.join(outdir, filename)
        fig.savefig(path, dpi=100, pil_kwargs={'quality': 95})
        if show:
            plt.show()
        plt.close(fig)
        print(f'Saved {path}')
    if refused:
        raise SystemExit(f'Not built (inconsistent runs): {", ".join(refused)}')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Section 4.1 nonlinear-function benchmark: deterministic and HMC models (Figs. func and func2). '
                    'Defaults are the manuscript settings.')
    parser.add_argument('--scenario', nargs='+', default=['all'], choices=list(SCENARIOS) + ['fig1', 'fig2', 'all'],
                        help='scenarios to run; fig1/fig2 select the three columns of a figure')
    parser.add_argument('--model', default='both', choices=MODELS + ['both'])
    parser.add_argument('--seed', type=int, default=42, help='data seed (models use seed + 1 and seed + 2)')
    parser.add_argument('--outdir', default='./results/function_v2',
                        help='output folder; relative paths are relative to this script folder')
    parser.add_argument('--n-lf-fig1', type=int, default=51, help='LF points for fig1a-c (main.tex:223)')
    parser.add_argument('--n-lf-fig2', type=int, default=2000, help='LF points for fig2a-c (main.tex:239)')
    parser.add_argument('--lf-epochs', type=int, default=20000)
    parser.add_argument('--mf-epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.001, help='Adam learning rate (LF and MFNN)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Eq. 3 weight: y = alpha*F_l + (1-alpha)*F_nl')
    parser.add_argument('--combine', default='convex', choices=['convex', 'sum'],
                        help='convex: Eq. 3 with --alpha; sum: legacy F_nl + F_l')
    parser.add_argument('--linear-head', default='affine', choices=['affine', 'legacy'],
                        help='affine: [2,1]; legacy: [2,10,1] identity-activation stack')
    parser.add_argument('--l2-lambda', type=float, default=1.0, help='weight of the L2 term on the linear-head weights')
    parser.add_argument('--hmc-samples', type=int, default=2000)
    parser.add_argument('--hmc-burnin', type=int, default=2000)
    parser.add_argument('--hmc-leapfrog', type=int, default=80)
    parser.add_argument('--hmc-step-size', type=float, default=0.01, help='initial (or, without adaptation, fixed) leapfrog step')
    parser.add_argument('--no-adapt-step-size', action='store_true',
                        help='plain HMC at the fixed step size (default: adapt during the first 80%% of burn-in)')
    parser.add_argument('--target-accept-prob', type=float, default=0.9, help='target acceptance of the step-size adaptation')
    parser.add_argument('--likelihood-sigma', default='noise', choices=['noise', 'fixed'],
                        help='noise: sigma_obs = sigma_H of the scenario; fixed: --sigma-fixed')
    parser.add_argument('--sigma-fixed', type=float, default=0.01)
    parser.add_argument('--prior-std', type=float, default=1.0, help='std of the N(0, s^2) prior on all BNN weights and biases')
    parser.add_argument('--figures-only', action='store_true', help='rebuild the composites and the metrics CSV from saved runs')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--deterministic', action='store_true', help='call tf.config.experimental.enable_op_determinism()')
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error('--alpha must lie in [0, 1]')
    return args

def main():
    args = parse_args()
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()
    os.makedirs(args.outdir, exist_ok=True)

    if not args.figures_only:
        groups = {'all': list(SCENARIOS), 'fig1': FIGURES['fig1'][1], 'fig2': FIGURES['fig2'][1]}
        requested = [s for name in args.scenario for s in groups.get(name, [name])]
        models = MODELS if args.model == 'both' else [args.model]
        for scn in [s for s in SCENARIOS if s in requested]:
            run_scenario(scn, models, args)

    write_metrics_csv(args.outdir)
    make_function_figures(args.outdir, show=args.show)

if __name__ == '__main__':
    main()
