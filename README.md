# Multi-fidelity Bayesian neural networks: benchmarks

Code for the two benchmarks (Section 3) of the paper *Bayesian multi-fidelity transfer learning for carbon fiber
composites*: a nonlinear function and an inverse problem for a nonlinear Poisson equation, both with noisy
multi-fidelity data. They compare a deterministic multi-fidelity neural network (MFNN) with Bayesian versions trained
by Hamiltonian Monte Carlo (HMC), variational inference (VI), and Monte Carlo (MC) dropout.

Results are not included; every number and figure of Section 3 is produced by the commands below.

## Installation

Python 3.12, CPU only:

```bash
pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES= TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
```

Run all commands from the repository root. The published runs used the CPU with one TensorFlow thread per run (the
`export` above); results are then bit-identical when rerun on the same machine and agree statistically on other
machines. The runs of each loop below are independent and can be run in parallel.

## Files

| file | content |
|---|---|
| `model_bnn.py` | MFNN and HMC model of the nonlinear-function benchmark |
| `model_bpinn.py` | MFNN with a PDE residual (deterministic and MC dropout) and HMC model of the inverse problem |
| `model_bnn_vi.py` | VI with a low-rank Gaussian family and the tempered or untempered ELBO |
| `example_mf_bnn.py` | nonlinear function: data, low-fidelity network, deterministic MFNN, single-chain HMC |
| `example_mf_bpinn_noise_analysis.py` | inverse problem: MC dropout, deterministic MFNN, HMC, and VI at five noise levels (sweep, merge, figures) |
| `experiments/function_multichain_hmc/` | four short and four long HMC chains per scenario (`hmc_multichain.py`); numbers of Section 3 (`plot_function_v2.py`) |
| `experiments/tempered_hmc/` | HMC on the tempered posterior that VI approximates (`hmc_tempered.py`) and its analysis |
| `experiments/scripts/summarize_pde.py` | MC dropout with a single high-fidelity network, paired with the published runs |
| `calibration_summary.py` | calibration at noise level 0.2 over 20 noise realizations |
| `negative_k_optimum.py` | the spurious optimum with negative k that motivates the log-normal prior on k |

## Nonlinear function

```bash
python example_mf_bnn.py          # six scenarios, deterministic MFNN and single-chain HMC -> results/function_v2
M=experiments/function_multichain_hmc/hmc_multichain.py
for s in fig1a fig1b fig1c fig2a fig2b fig2c; do
    python $M prep --scenario $s
    for i in 0 1 2 3; do          # chain i: seed 44 + 1000 i; chain 0 starts at zero, the others at Glorot weights
        if [ $i = 0 ]; then init=zeros; else init=glorot; fi
        python $M chain --scenario $s --tag c$i --seed $((44 + 1000 * i)) --init $init
        python $M chain --scenario $s --tag L$i --seed $((44 + 1000 * i)) --init $init --samples 14000
    done
done
python $M analyze --chain-tags c0 c1 c2 c3 --long-tags L0 L1 L2 L3
python experiments/function_multichain_hmc/plot_function_v2.py
```

The `prep` stage checks that it reproduces the deterministic MFNN of `results/function_v2` exactly. The paper uses
scenarios `fig1a`-`fig1c`: 51 low- and 14 high-fidelity points, with noise standard deviations of 0.01/0.01,
0.05/0.01, and 0.05/0.05 (low/high fidelity). `plot_function_v2.py` pools the four long chains (4 x 14000 draws) and
writes the RMSE of the deterministic model and HMC and the coverage of the 95% credible intervals to
`figure_v2_metrics.csv`.

## Inverse PDE problem

Noise study (Fig. 3): 5 noise levels x 3 noise realizations, all four models.

```bash
P=example_mf_bpinn_noise_analysis.py
for s in 1 2 3; do for n in 0.01 0.05 0.1 0.15 0.2; do
    python $P --noise-levels $n --seeds $s --outdir results_bpinn/noise_study_ode_v2_parts/s${s}_n$n
done; done
python $P --mode merge --merge-dirs results_bpinn/noise_study_ode_v2_parts/* --outdir results_bpinn/noise_study_ode_v2
```

This writes `results_bpinn/noise_study_ode_v2/figures/metric_vs_noise.png` (Fig. 3) and `noise_study_runs.csv`. The
CSV also contains the deterministic MFNN, shown in `metric_vs_noise_all.png`.

VI with the untempered ELBO (beta = 1):

```bash
for s in 1 2 3; do for n in 0.01 0.05 0.1 0.15 0.2; do
    python $P --models mfbpinn_vi --vi-beta 1.0 --noise-levels $n --seeds $s \
        --outdir results_bpinn/ablations_v2/vi_beta1_parts/s${s}_n$n
done; done
python $P --mode merge --models mfbpinn_vi --plot-models mfbpinn_vi --vi-beta 1.0 \
    --merge-dirs results_bpinn/ablations_v2/vi_beta1_parts/* --outdir results_bpinn/ablations_v2/vi_beta1
```

Calibration at noise level 0.2 over 20 further noise realizations (seeds 4-23):

```bash
for s in $(seq 4 23); do
    python $P --noise-levels 0.2 --seeds $s --outdir results_bpinn/calibration_sigma0.2_v2_parts/s${s}_n0.2
done
python $P --mode merge --merge-dirs results_bpinn/calibration_sigma0.2_v2_parts/* \
    --outdir results_bpinn/calibration_sigma0.2_v2
python calibration_summary.py
```

### HMC on the tempered posterior

VI with beta = 0.1 approximates the tempered posterior p(D|theta)^(1/beta) p(theta). `hmc_tempered.py` samples it
with the HMC settings of the noise study; only the likelihood standard deviation changes, to sigma * sqrt(beta). With
`--beta 1` it reproduces the published HMC runs exactly.

Run the noise study, the calibration runs, and the beta = 1 runs above first; then:

```bash
T=experiments/tempered_hmc
# validation: with beta = 1 the published HMC runs are reproduced
python $T/hmc_tempered.py --beta 1 --noise-levels 0.1 --seeds 1 --outdir $T/validation_beta1/parts/s1_n0.1
python $T/hmc_tempered.py --beta 1 --noise-levels 0.2 --seeds 4 --outdir $T/validation_beta1/parts/s4_n0.2
# one tempered chain per run, and a second chain with 250 leapfrog steps
for s in 1 2 3; do for n in 0.01 0.05 0.1 0.15 0.2; do
    python $T/hmc_tempered.py --beta 0.1 --noise-levels $n --seeds $s --outdir $T/hmc_tempered/parts/s${s}_n$n
    python $T/hmc_tempered.py --beta 0.1 --hmc-leapfrog 250 --noise-levels $n --seeds $s \
        --outdir $T/hmc_tempered_L250/parts/s${s}_n$n
done; done
for s in $(seq 4 23); do
    python $T/hmc_tempered.py --beta 0.1 --noise-levels 0.2 --seeds $s --outdir $T/hmc_tempered_cal/parts/s${s}_n0.2
    python $T/hmc_tempered.py --beta 0.1 --hmc-leapfrog 250 --noise-levels 0.2 --seeds $s \
        --outdir $T/hmc_tempered_L250_cal/parts/s${s}_n0.2
    python $P --models mfbpinn_vi --vi-beta 1.0 --noise-levels 0.2 --seeds $s --outdir $T/vi_beta1_cal/parts/s${s}_n0.2
done
python $T/check_validation.py
python $T/summarize_c5.py
python $T/pooled_chains.py
```

The chains of the tempered posterior mix slowly, so their per-run numbers are indicative.

### MC dropout with a single high-fidelity network

```bash
for s in 1 2 3; do for n in 0.01 0.05 0.1 0.15 0.2; do
    python $P --models mfpinn --mc-dropout 0.05 --alpha 0 --noise-levels $n --seeds $s \
        --outdir experiments/dropout_single_net/parts/s${s}_n$n
done; done
python $P --mode merge --models mfpinn --plot-models mfpinn --alpha 0 \
    --merge-dirs experiments/dropout_single_net/parts/* --outdir experiments/dropout_single_net/merged
python experiments/scripts/summarize_pde.py
```

`--alpha 0` replaces the linear and nonlinear high-fidelity networks by the nonlinear one alone, as in the HMC and VI
models. `summarize_pde.py` pairs each run with the published MC-dropout run of the same noise level and seed and
writes `experiments/dropout_single_net/paired_runs.csv`.

## Citation

[citation of the paper]

## License

MIT License (see `LICENSE`): free to use, modify, and redistribute, with the copyright notice kept.
