import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf

# Section 4.2 setup (data, LF network, settings) of the noise sweep
from example_mf_bpinn_noise_analysis import (
    BNN_LAYERS, LEARNING_RATE, LF_EPOCHS, MAP_STEPS, NOISE_LEVELS, SEEDS, build_dataset, likelihood_sigma,
    train_low_fidelity_model, write_csv)
from model_bpinn import BayesianPINN


OUTPUT = './results_bpinn/ablations_v2/negative_k_optimum.csv'
K_STARTS = [0.05, -1.0]     # initial k of the MAP optimization (0.05: the sweep's initial value)


def main():
    # Local optima of the posterior with the standard normal prior on k (positive_k=False): one MAP optimization per
    # initial k from the same Glorot initialization; a start at k < 0 shows whether a negative-k optimum exists
    parser = argparse.ArgumentParser(description='MAP optima of the section 4.2 posterior with an N(0, 1) prior on k')
    parser.add_argument('--noise-levels', nargs='+', type=float, default=NOISE_LEVELS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--steps', type=int, default=MAP_STEPS)
    parser.add_argument('--output', default=OUTPUT)
    args = parser.parse_args()

    rows = []
    for noise_level in args.noise_levels:
        for seed in args.seeds:
            dataset = build_dataset(noise_level, seed)
            model_lf = train_low_fidelity_model(dataset, argparse.Namespace(lf_epochs=LF_EPOCHS))
            for k_start in K_STARTS:
                tf.keras.utils.set_random_seed(1234)
                model = BayesianPINN(BNN_LAYERS, nu=tf.Variable(k_start, dtype=tf.float32), lf_model=model_lf,
                                     sigma_obs=likelihood_sigma(noise_level), residual='ode', positive_k=False)
                k_map = model.map_estimate(dataset['xt_train_hf'], dataset['u_train_hf'], dataset['xt_train_f'],
                                           dataset['f_train'], steps=args.steps, learning_rate=LEARNING_RATE, restarts=1)
                u_pred = model.model(dataset['xt_test'], model.params).numpy().reshape(-1)
                f_pred = model.physical_residual(dataset['xt_test'], model.params, model.nu).numpy().reshape(-1)
                rows.append({'noise_level': noise_level, 'seed': seed, 'k_start': k_start, 'k_map': k_map,
                             'neg_log_posterior': model.map_history[0][0],
                             'u_mse': float(np.mean((u_pred - dataset['u_true']) ** 2)),
                             'f_mse': float(np.mean((f_pred - dataset['f_true']) ** 2))})
                print(rows[-1])
                write_csv(args.output, rows)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
