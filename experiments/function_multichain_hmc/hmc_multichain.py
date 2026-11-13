"""Reviewer experiment: multi-chain HMC for the Section 4.1 nonlinear-function benchmark (Figs. func and func2).

Stages (each scenario/chain is a separate process, so they can run in parallel):
  prep     data (seed 42), low-fidelity network and deterministic MFNN exactly as example_mf_bnn.run_scenario;
           saves the LF predictions that every HMC chain of the scenario shares
  chain    one HMC chain with the paper's sampler settings (2000 burn-in, 80 leapfrog steps, initial step 0.01
           adapted over the first 80% of burn-in to a target acceptance of 0.9); --init zeros (paper) or glorot
  analyze  pooled multi-chain metrics, diagnostics and the redrawn composite figures

Nothing in the benchmark folder is modified; outputs go to this script's folder.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, BENCH)

import numpy as np
import example_mf_bnn as ex          # note: chdirs to BENCH and sets the plotting rcParams of the paper figures
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from model_bnn import BayesianNeuralNetwork

PUBLISHED = os.path.join(BENCH, 'results', 'function_v2')
PREP = os.path.join(HERE, 'prep')
CHAINS = os.path.join(HERE, 'chains')
DATA_SEED = 42
MFNN_SEED = DATA_SEED + ex.MODEL_SEED_OFFSET['mfnn']      # 43, as in run_scenario
PAPER_HMC_SEED = DATA_SEED + ex.MODEL_SEED_OFFSET['mfbnn_hmc']   # 44, as in run_scenario


def mf_args():
    # The manuscript defaults of example_mf_bnn.parse_args that build_mfnn / run_mfnn read
    return argparse.Namespace(combine='convex', alpha=0.5, linear_head='affine', l2_lambda=1.0, mf_epochs=10000,
                              lr=0.001)


def prep(scn):
    sigma_l, sigma_h, figure, layout = ex.SCENARIOS[scn]
    n_lf = 51 if figure == 'fig1' else 2000
    x_lf, y_lf, x_hf, y_hf = ex.generate_data(scn, n_lf, DATA_SEED)
    x_test = np.linspace(0., 1., ex.N_TEST)[:, None].astype(np.float32)
    y_true = ex.high_fidelity_data(x_test).astype(np.float32)
    t0 = time.time()
    model_lf = ex.training_low_fidelity(x_lf, y_lf, epochs=20000, learning_rate=0.001)
    lf_time = time.time() - t0
    lf_pred_test = model_lf.predict(x_test).numpy()
    lf_pred_hf = model_lf.predict(x_hf).numpy()
    data = {'x_test': x_test, 'y_true': y_true, 'x_lf': x_lf, 'y_lf': y_lf, 'x_hf': x_hf, 'y_hf': y_hf,
            'lf_pred_test': lf_pred_test}
    tf.keras.utils.set_random_seed(MFNN_SEED)
    t0 = time.time()
    arrays, _ = ex.run_mfnn(model_lf, data, mf_args())
    mf_time = time.time() - t0
    out = dict(data, lf_pred_hf=lf_pred_hf, mfnn_mean=arrays['mean'], sigma_l=sigma_l, sigma_h=sigma_h, n_lf=n_lf,
               layout=layout, scenario=scn)
    np.savez(os.path.join(PREP, f'{scn}.npz'), **out)

    # Agreement with the published run (results/function_v2)
    check = {'lf_train_s': lf_time, 'mfnn_train_s': mf_time}
    with np.load(os.path.join(PUBLISHED, f'{scn}_mfnn.npz')) as pub:
        check['y_hf_equal'] = bool(np.array_equal(pub['y_hf'], y_hf))
        check['lf_pred_test_maxdiff'] = float(np.max(np.abs(pub['lf_pred_test'] - lf_pred_test)))
        check['mfnn_mean_maxdiff'] = float(np.max(np.abs(pub['mean'] - arrays['mean'])))
    check['mfnn_rmse'] = float(np.sqrt(np.mean((arrays['mean'] - y_true.ravel()) ** 2)))
    with open(os.path.join(PREP, f'{scn}_check.json'), 'w') as f:
        json.dump(check, f, indent=2)
    print(scn, check)


def glorot_init(rng):
    # Glorot-uniform weights and zero biases in the flat parameter order of BayesianNeuralNetwork.reshape_params
    parts = []
    for i in range(len(ex.HF_LAYERS) - 1):
        fan_in, fan_out = ex.HF_LAYERS[i], ex.HF_LAYERS[i + 1]
        lim = np.sqrt(6.0 / (fan_in + fan_out))
        parts.append(rng.uniform(-lim, lim, size=fan_in * fan_out))
        parts.append(np.zeros(fan_out))
    return np.concatenate(parts).astype(np.float32)


def chain_path(scn, tag):
    return os.path.join(CHAINS, f'{scn}_{tag}.npz')


def run_chain(scn, tag, seed, init, burnin, samples):
    with np.load(os.path.join(PREP, f'{scn}.npz')) as d:
        d = {k: d[k] for k in d.files}
    sigma_h = float(d['sigma_h'])
    x_train = np.hstack((d['x_hf'], d['lf_pred_hf']))
    x_test_bnn = np.hstack((d['x_test'], d['lf_pred_test']))
    tf.keras.utils.set_random_seed(seed)
    model = BayesianNeuralNetwork(ex.HF_LAYERS, num_samples=samples, num_burnin_steps=burnin, sigma_obs=sigma_h,
                                  prior_std=1.0)
    if init == 'glorot':
        model.params.assign(glorot_init(np.random.default_rng(seed)))
    init_params = model.params.numpy().copy()
    t0 = time.time()
    model.train(x_train, d['y_hf'], step_size=0.01, leapfrog_steps=80, adapt_step_size=True, target_accept_prob=0.9)
    train_s = time.time() - t0

    preds = model.samples_predict(x_test_bnn).reshape(len(model.samples), -1).astype(np.float32)
    lp_fn = tf.function(lambda p: model.joint_log_prob(x_train, d['y_hf'], p))
    logpost = np.array([float(lp_fn(tf.constant(p))) for p in model.samples])
    np.savez_compressed(chain_path(scn, tag), preds=preds, logpost=logpost, init_params=init_params,
                        final_params=model.samples[-1], seed=seed, init=init, burnin=burnin, samples=samples,
                        acceptance=model.acceptance_rate, burnin_acceptance=model.burnin_acceptance,
                        step_size=model.final_step_size, train_s=train_s)
    print(f'{scn} {tag}: train {train_s:.0f} s, acceptance {model.acceptance_rate:.3f}, step {model.final_step_size:.3g}')


# ---------------------------------------------------------------- analysis

def ess_rhat(preds):
    """preds [n, chains, points] -> (ESS per point, split-R-hat per point); ESS by Geyer's initial positive sequence
    (filter_beyond_positive_pairs=True), across chains when chains > 1 (TFP cross_chain_dims)."""
    level = tf.get_logger().level
    tf.get_logger().setLevel('ERROR')
    p = tf.constant(preds, dtype=tf.float64)
    if preds.shape[1] > 1:
        ess = tfp.mcmc.effective_sample_size(p, filter_beyond_positive_pairs=True, cross_chain_dims=1).numpy()
    else:
        ess = tfp.mcmc.effective_sample_size(p[:, 0], filter_beyond_positive_pairs=True).numpy()
    rhat = tfp.mcmc.potential_scale_reduction(p, independent_chain_ndims=1, split_chains=True).numpy()
    tf.get_logger().setLevel(level)
    return ess, rhat


def band_metrics(y_true, preds):
    mean = preds.mean(axis=0)
    lo, hi = np.percentile(preds, [2.5, 97.5], axis=0)
    return {'rmse': float(np.sqrt(np.mean((mean - y_true) ** 2))),
            'coverage_95': float(np.mean((lo <= y_true) & (y_true <= hi))),
            'mean_width_95': float(np.mean(hi - lo))}, mean, lo, hi


def load_published(scn):
    with open(os.path.join(PUBLISHED, f'{scn}_mfbnn_hmc.json')) as f:
        c = json.load(f)
    m = c['metrics']
    return {'rmse': m['rmse'], 'coverage_95': m['coverage_95'], 'mean_width_95': m['mean_width_95'],
            'ess_min': c['ess_min'], 'split_rhat_max': c['split_rhat_max'], 'step_size': c['final_step_size']}


def load_chain(scn, tag):
    with np.load(chain_path(scn, tag)) as c:
        return {k: c[k] for k in c.files}


def pool_summary(preds, y_true):
    """preds [n, C, points]: metrics of the pooled draws, multi-chain diagnostics and chain agreement."""
    n, C, _ = preds.shape
    pooled = preds.reshape(n * C, -1)
    m, mean, lo, hi = band_metrics(y_true, pooled)
    ess, rhat = ess_rhat(preds)
    out = {'chains': C, 'samples_per_chain': n, **m, 'max_abs_error': float(np.max(np.abs(mean - y_true))),
           'ess_min': float(ess.min()), 'ess_median': float(np.median(ess)),
           'split_rhat_max': float(rhat.max()), 'split_rhat_median': float(np.median(rhat)),
           'frac_rhat_gt_1.1': float(np.mean(rhat > 1.1))}
    if C > 1:
        sd = pooled.std(axis=0)
        chain_means = preds.mean(axis=0)
        # Function-space agreement: deviation of each chain mean from the pooled mean in units of the pooled sd,
        # and the RMS (over x) difference between the means of two chains
        z = np.abs(chain_means - mean[None]) / sd[None]
        pair = [float(np.sqrt(np.mean((chain_means[i] - chain_means[j]) ** 2))) for i in range(C) for j in range(i + 1, C)]
        chain_rmse = np.sqrt(np.mean((chain_means - y_true[None]) ** 2, axis=1))
        out.update({'chain_mean_dev_max_sd': float(z.max()), 'chain_mean_dev_rms_sd': float(np.sqrt(np.mean(z ** 2))),
                    'chain_mean_pair_rms_max': float(max(pair)), 'pooled_sd_mean': float(sd.mean()),
                    'chain_rmse_min': float(chain_rmse.min()), 'chain_rmse_max': float(chain_rmse.max())})
    return out, mean, lo, hi


def analyze(chain_tags, long_tags):
    rows, per_chain, long_rows = [], [], []
    store = {}
    for scn in ex.SCENARIOS:
        with np.load(os.path.join(PREP, f'{scn}.npz')) as d:
            d = {k: d[k] for k in d.files}
        y_true = d['y_true'].ravel()
        chains = [load_chain(scn, tag) for tag in chain_tags]
        preds = np.stack([c['preds'] for c in chains], axis=1)          # [n, C, 500]
        summ, mean, lo, hi = pool_summary(preds, y_true)
        row = {'scenario': scn, 'run': f'{len(chain_tags)} chains x {preds.shape[0]}', **summ}
        row.update({f'published_{k}': v for k, v in load_published(scn).items()})
        rows.append(row)
        store[scn] = dict(d, mean=mean, lo=lo, hi=hi, chain_means=preds.mean(axis=0))
        for tag, c, cp in zip(chain_tags, chains, np.moveaxis(preds, 1, 0)):
            cs, *_ = pool_summary(cp[:, None], y_true)
            per_chain.append({'scenario': scn, 'chain': tag, 'init': str(c['init']), 'seed': int(c['seed']),
                              'samples': len(cp), 'rmse': cs['rmse'], 'coverage_95': cs['coverage_95'],
                              'mean_width_95': cs['mean_width_95'], 'ess_min': cs['ess_min'],
                              'split_rhat_max': cs['split_rhat_max'], 'acceptance': float(c['acceptance']),
                              'step_size': float(c['step_size']), 'logpost_mean': float(c['logpost'].mean()),
                              'train_s': float(c['train_s'])})
        longs = [(tag, load_chain(scn, tag)) for tag in long_tags if os.path.exists(chain_path(scn, tag))]
        for tag, c in longs:
            lp = c['preds']
            short = chain_tags[long_tags.index(tag)] if long_tags.index(tag) < len(chain_tags) else None
            prefix_equal = None
            if short is not None:
                sp = chains[chain_tags.index(short)]['preds']
                prefix_equal = bool(np.array_equal(lp[:len(sp)], sp))
            ls, lmean, llo, lhi = pool_summary(lp[:, None], y_true)
            # The long chain cut into 4 consecutive segments, diagnosed like 4 chains
            q = (len(lp) // 4) * 4
            qs, *_ = pool_summary(lp[:q].reshape(4, -1, lp.shape[1]).transpose(1, 0, 2), y_true)
            long_rows.append({'scenario': scn, 'run': f'single chain {tag} x {len(lp)}', 'seed': int(c['seed']),
                              'init': str(c['init']), 'extends': short, 'prefix_identical': prefix_equal,
                              **{k: ls[k] for k in ('rmse', 'coverage_95', 'mean_width_95', 'max_abs_error', 'ess_min',
                                                    'ess_median', 'split_rhat_max')},
                              'quarter_rhat_max': qs['split_rhat_max'], 'quarter_dev_max_sd': qs['chain_mean_dev_max_sd'],
                              'acceptance': float(c['acceptance']), 'step_size': float(c['step_size']),
                              'logpost_mean': float(c['logpost'].mean()), 'train_s': float(c['train_s'])})
            store[scn][f'long_{tag}'] = (lmean, llo, lhi)
        if len(longs) == len(long_tags) and len(longs) > 1:
            lpreds = np.stack([c['preds'] for _, c in longs], axis=1)
            ls, lmean, llo, lhi = pool_summary(lpreds, y_true)
            rows.append({'scenario': scn, 'run': f'{len(longs)} chains x {lpreds.shape[0]}', **ls,
                         **{f'published_{k}': v for k, v in load_published(scn).items()}})
            store[scn]['long_pooled'] = (lmean, llo, lhi)
    write_csv(os.path.join(HERE, 'multichain_summary.csv'), rows)
    write_csv(os.path.join(HERE, 'per_chain.csv'), per_chain)
    if long_rows:
        write_csv(os.path.join(HERE, 'long_chain.csv'), long_rows)
    make_figures(store)
    make_chain_figure(store, chain_tags)


def write_csv(path, rows):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('Saved', path)


def make_figures(store):
    # Paper layout of function_01/02.jpeg (example_mf_bnn.make_function_figures): top row deterministic, bottom
    # row HMC pooled over 4 chains: the paper-length chains (2000 samples each, *_4chains.jpeg) and, when present,
    # the 4 long chains (14000 samples each, *_4longchains.jpeg)
    variants = [('_4chains.jpeg', None, 'HMC (4 chains)')]
    if all('long_pooled' in store[scn] for scn in ex.SCENARIOS):
        variants.append(('_4longchains.jpeg', 'long_pooled', 'HMC (4 long chains)'))
    for suffix, key, label in variants:
        for filename, columns in ex.FIGURES.values():
            fig, axes = plt.subplots(2, 3, figsize=(32, 15), dpi=100)
            fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.15, wspace=0.15, hspace=0.2)
            for j, scn in enumerate(columns):
                s = store[scn]
                ex.plot_panel(axes[0, j], dict(s, mean=s['mfnn_mean']), 'mfnn')
                band = s if key is None else dict(s, mean=s[key][0], lo=s[key][1], hi=s[key][2])
                ex.plot_panel(axes[1, j], band, 'mfbnn_hmc')
            for i, text in enumerate(['Deterministic', label]):
                pos = axes[i, 0].get_position()
                fig.text(0.045, 0.5 * (pos.y0 + pos.y1), text, fontsize=48 if i == 0 else 36, ha='center',
                         va='center', rotation=90)
            for j in range(len(columns)):
                pos = axes[1, j].get_position()
                fig.text(0.5 * (pos.x0 + pos.x1), 0.04, f'({"abc"[j]})', fontsize=60, ha='center', va='center')
            path = os.path.join(HERE, filename.replace('.jpeg', suffix))
            fig.savefig(path, dpi=100, pil_kwargs={'quality': 95})
            plt.close(fig)
            print('Saved', path)


def make_chain_figure(store, chain_tags):
    # Diagnostic: posterior mean of each paper-length chain and of each long chain against the pooled 95% band
    fig, axes = plt.subplots(2, 3, figsize=(30, 13))
    colors = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:brown']
    for ax, scn in zip(axes.ravel(), ex.SCENARIOS):
        s = store[scn]
        x = s['x_test'].ravel()
        ax.fill_between(x, s['lo'], s['hi'], color='red', alpha=0.15, label='95% band, 4 chains x 2000')
        ax.plot(x, s['y_true'].ravel(), '--', color='green', lw=3, label='true')
        for c, (tag, m) in enumerate(zip(chain_tags, s['chain_means'])):
            ax.plot(x, m, color=colors[c % 4], lw=2, label=f'{tag} mean (2000)')
        for c, key in enumerate(sorted(k for k in s if k.startswith('long_L'))):
            ax.plot(x, s[key][0], color=colors[c % 4], lw=1.5, ls=':', label=key.replace('long_', '') + ' mean (14000)')
        ax.scatter(s['x_hf'], s['y_hf'], color='black', s=60, zorder=4)
        ax.set_title(scn, fontsize=20)
    axes[0, 0].legend(fontsize=11, loc='lower left', ncol=2)
    fig.tight_layout()
    path = os.path.join(HERE, 'chain_means.png')
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print('Saved', path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prep', 'chain', 'analyze'])
    p.add_argument('--scenario', nargs='+', default=list(ex.SCENARIOS))
    p.add_argument('--tag', default='c0')
    p.add_argument('--seed', type=int, default=PAPER_HMC_SEED)
    p.add_argument('--init', choices=['zeros', 'glorot'], default='zeros')
    p.add_argument('--burnin', type=int, default=2000)
    p.add_argument('--samples', type=int, default=2000)
    p.add_argument('--chain-tags', nargs='+', default=['c0', 'c1', 'c2', 'c3'])
    p.add_argument('--long-tags', nargs='+', default=[])
    a = p.parse_args()
    os.makedirs(PREP, exist_ok=True)
    os.makedirs(CHAINS, exist_ok=True)
    if a.stage == 'prep':
        for scn in a.scenario:
            prep(scn)
    elif a.stage == 'chain':
        for scn in a.scenario:
            run_chain(scn, a.tag, a.seed, a.init, a.burnin, a.samples)
    else:
        analyze(a.chain_tags, a.long_tags)


if __name__ == '__main__':
    main()
