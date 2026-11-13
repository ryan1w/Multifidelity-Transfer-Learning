import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow.keras.activations import tanh, linear
from tqdm import tqdm


# np.random.seed(42)
# tf.random.set_seed(42)

# Coefficient lambda of the nonlinear Poisson equation lambda * u_xx + k * tanh(u) = f (main.tex Eq. 21)
LAMBDA_PDE = 0.01


def select_residual(residual, input_width):
    # 'ode' (1-D nonlinear Poisson) or 'heat'; None keeps the legacy rule (input width 2 -> 'ode')
    if residual is None:
        residual = 'ode' if input_width == 2 else 'heat'
    if residual not in ('ode', 'heat'):
        raise ValueError(f"residual must be 'ode', 'heat' or None, got {residual!r}")
    return residual


class BayesianPINN(tf.Module):
    def __init__(self, layer_sizes, nu=0.7, num_samples=2000, num_burnin_steps=2000, lf_model=None,  sigma_obs=0.1,
                 residual=None, positive_k=False, logk_prior=(0.0, 1.0), learn_noise=False,
                 noise_prior=(float(np.log(0.1)), 1.5)):
        super(BayesianPINN, self).__init__()
        self.num_samples = num_samples
        self.num_burnin_steps = num_burnin_steps
        self.layer_sizes = layer_sizes
        self.lf_model = lf_model
        self.sigma_obs = tf.convert_to_tensor(sigma_obs, dtype=tf.float32)
        # Inverse parameter k: sampled directly with an N(0, 1) prior (legacy), or with positive_k as log k with an
        # N(logk_prior) prior, i.e. a log-normal prior that enforces k > 0
        self.positive_k = positive_k
        self.logk_prior = (float(logk_prior[0]), float(logk_prior[1]))
        k_init = float(tf.convert_to_tensor(nu, dtype=tf.float32))
        self.nu = tf.convert_to_tensor(np.log(k_init), dtype=tf.float32) if positive_k else nu   # k-state
        # Noise levels: known (sigma_obs for u and f), or with learn_noise sampled as log sigma_u, log sigma_f with
        # N(noise_prior) priors; the warm start holds them at the prior median
        self.learn_noise = learn_noise
        self.noise_prior = (float(noise_prior[0]), float(noise_prior[1]))
        self.log_sigma = (tf.fill([2], tf.constant(self.noise_prior[0], tf.float32)) if learn_noise else None)

        self.shapes = [((layer_sizes[i], layer_sizes[i + 1]), (layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]
        num_params = sum((layer_sizes[i] + 1) * layer_sizes[i+1] for i in range(len(layer_sizes) - 1))
        # self.params = tf.Variable(tf.random.normal([num_params], stddev=0.1), dtype=tf.float32)
        self.params = tf.Variable(tf.zeros([num_params]))

        if select_residual(residual, layer_sizes[0]) == 'ode':
            self.physical_residual = self.ode_residual
        else:
            self.physical_residual = self.pde_residual


    def reshape_params(self, flat_params):
        weights = []
        biases = []
        idx = 0
        for w_shape, b_shape in self.shapes:
            w_size = w_shape[0] * w_shape[1]
            b_size = b_shape

            weights.append(tf.reshape(flat_params[idx: idx + w_size], w_shape))
            idx += w_size
            biases.append(tf.reshape(flat_params[idx: idx + b_size], (b_shape,)))
            idx += b_size
        return weights, biases

    def model(self, x, flat_params):
        weights, biases = self.reshape_params(flat_params)
        if self.lf_model is not None:
            outputs = tf.concat([x, self.lf_model(x)], axis=-1)
        else:
            outputs = x
            
        for w, b in zip(weights[:-1], biases[:-1]):
            outputs = tanh(tf.matmul(outputs, w) + b)
        return tf.matmul(outputs, weights[-1]) + biases[-1]

    def ode_residual(self, x_in, flat_params, nu):
        x = tf.convert_to_tensor(x_in, dtype=tf.float32)

        with tf.GradientTape() as tape2:
            tape2.watch(x)
            with tf.GradientTape() as tape1:
                tape1.watch(x)
                u = self.model(x, flat_params)
            u_x = tape1.gradient(u, x)
        u_xx = tape2.gradient(u_x, x)  # Second derivative of u with respect to x
        residual = LAMBDA_PDE * u_xx + nu * tanh(u)
        pi = tf.constant(np.pi, dtype=tf.float32)
        # residual = 1 / (192*pi**2) * u_xx - nu / (24*pi) * u * u_x
        return residual
    
    def pde_residual(self, xt, flat_params, nu):
        x = tf.convert_to_tensor(xt[:, 0:1], dtype=tf.float32)
        t = tf.convert_to_tensor(xt[:, 1:2], dtype=tf.float32)
        
        with tf.GradientTape() as tape2:
            tape2.watch([x, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, t])
                u = self.model(tf.concat([x, t], axis=1), flat_params)
            u_x = tape1.gradient(u, x)
            u_t = tape1.gradient(u, t)
        u_xx = tape2.gradient(u_x, x)
        del tape1
    
        # Compute the PDE residual: u_t - nu * u_xx = 0
        residual = u_t - nu * u_xx
        return residual

    def k_from_state(self, nu_state):
        # k-state -> k (k = exp(log k) with positive_k)
        return tf.exp(nu_state) if self.positive_k else nu_state

    def noise_levels(self, log_sigma=None):
        # (sigma_u, sigma_f): known sigma_obs, or exp of the learned log noise levels
        if not self.learn_noise:
            return self.sigma_obs, self.sigma_obs
        log_sigma = self.log_sigma if log_sigma is None else log_sigma
        return tf.exp(log_sigma[0]), tf.exp(log_sigma[1])

    def log_likelihood(self, x_u, u, x_f, f, flat_params, k, sigma_u, sigma_f):
        pred_u = self.model(x_u, flat_params)
        likelihood_u = tf.reduce_sum(tfd.Normal(loc=pred_u, scale=sigma_u).log_prob(u))
        pde_residual = self.physical_residual(x_f, flat_params, k)
        likelihood_f = tf.reduce_sum(tfd.Normal(loc=pde_residual, scale=sigma_f).log_prob(f))
        return likelihood_u + likelihood_f

    def log_prior(self, flat_params, nu_state, log_sigma=None):
        log_prob = tf.reduce_sum(tfd.Normal(loc=0., scale=1.0).log_prob(flat_params))
        if self.positive_k:
            log_prob += tfd.Normal(loc=self.logk_prior[0], scale=self.logk_prior[1]).log_prob(nu_state)
        else:
            log_prob += tfd.Normal(loc=0., scale=1.0).log_prob(nu_state)
        if self.learn_noise and log_sigma is not None:
            log_prob += tf.reduce_sum(tfd.Normal(loc=self.noise_prior[0], scale=self.noise_prior[1]).log_prob(log_sigma))
        return log_prob

    def joint_log_prob(self, x_u, u, x_f, f, flat_params, nu, log_sigma=None):
        # nu is the k-state (k, or log k with positive_k); log_sigma = (log sigma_u, log sigma_f) with learn_noise
        sigma_u, sigma_f = self.noise_levels(log_sigma)
        return (self.log_prior(flat_params, nu, log_sigma) +
                self.log_likelihood(x_u, u, x_f, f, flat_params, self.k_from_state(nu), sigma_u, sigma_f))

    def map_estimate(self, x_u, u, x_f, f, steps=10000, learning_rate=1e-3, restarts=1):
        # MAP warm start: Adam on -log p(theta, nu | D) from Glorot initializations (all-zero weights never break the
        # hidden-unit symmetry), keeping the restart with the highest posterior; the chain then starts at the MAP. With
        # the N(0, 1) prior on k this usually places it in the positive-k mode, but every restart can end in the
        # spurious negative-k mode (f reproduced, u nearly flat), which can also have the higher density (6 u sensors,
        # sigma=0.2, seed 1); positive_k excludes that mode. Learned noise levels are held at the prior median here
        # (a joint MAP over a flexible network and its noise level drives the noise level towards zero)
        nu_init = tf.convert_to_tensor(self.nu, dtype=tf.float32)
        best = None
        self.map_history = []   # (-log posterior, k) of every restart
        for restart in range(restarts):
            init = []
            for w_shape, b_shape in self.shapes:
                init.append(tf.reshape(tf.keras.initializers.GlorotUniform()(w_shape), [-1]))
                init.append(tf.zeros([b_shape]))
            params = tf.Variable(tf.concat(init, axis=0))
            nu = tf.Variable(nu_init)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            @tf.function
            def map_step():
                with tf.GradientTape() as tape:
                    loss = -self.joint_log_prob(x_u, u, x_f, f, params, nu, self.log_sigma)
                grads = tape.gradient(loss, [params, nu])
                optimizer.apply_gradients(zip(grads, [params, nu]))
                return loss

            for step in tqdm(range(steps), desc=f'MAP warm start {restart + 1}/{restarts}'):
                map_step()
            loss = float(-self.joint_log_prob(x_u, u, x_f, f, params, nu, self.log_sigma))
            k_value = float(self.k_from_state(nu).numpy())
            tqdm.write(f'MAP restart {restart + 1}: -log posterior {loss:.4e}, k {k_value:.4f}')
            self.map_history.append((loss, k_value))
            if np.isfinite(loss) and (best is None or loss < best[0]):
                best = (loss, params.numpy(), float(nu.numpy()), k_value)
        if best is None:
            raise RuntimeError('no MAP restart gave a finite -log posterior')
        self.params.assign(best[1])
        self.nu = tf.convert_to_tensor(best[2], dtype=tf.float32)
        self.map_nu = best[3]
        return self.map_nu

    def train(self, x_u, u, x_f, f, step_size=0.01, leapfrog_steps=80, adapt_step_size=True, target_accept_prob=0.9):
        # step_size_var = tf.Variable(step_size, dtype=tf.float32)
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=lambda *args: self.joint_log_prob(x_u, u, x_f, f, *args),
            step_size=step_size,
            num_leapfrog_steps=leapfrog_steps
        )

        if adapt_step_size:
            # Initial step size adapted during the first 80% of burn-in, then frozen for sampling. TFP multiplies or
            # divides it by 1.01 per step, so the median per-step acceptance probability (not the mean acceptance
            # rate stored below) tends to target_accept_prob
            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                hmc_kernel,
                num_adaptation_steps=int(self.num_burnin_steps*0.8),
                target_accept_prob=target_accept_prob
            )
        else:
            # Plain HMC with a fixed leapfrog step size
            kernel = hmc_kernel

        # Process sampling step by step
        @tf.function
        def one_step(state, kernel_results):
            return kernel.one_step(state, kernel_results)

        samples = []
        accepted = []
        state = (tf.convert_to_tensor(self.params), tf.convert_to_tensor(self.nu, dtype=tf.float32))
        if self.learn_noise:
            # The chain also samples (log sigma_u, log sigma_f), starting at the prior median
            state = state + (tf.convert_to_tensor(self.log_sigma),)
        kernel_results = kernel.bootstrap_results(state)
        total_steps = self.num_samples + self.num_burnin_steps
        for step in tqdm(range(total_steps), desc='Sampling'):
            state, kernel_results = one_step(state, kernel_results)
            hmc_results = kernel_results.inner_results if adapt_step_size else kernel_results
            accepted.append(bool(hmc_results.is_accepted.numpy()))
            # Keep exactly num_samples post-burn-in states
            if step >= self.num_burnin_steps:
                samples.append(state)

            if (step + 1) % 100 == 0:
                tqdm.write(f'Step {step + 1}, Acceptance (last 100 steps): {np.mean(accepted[-100:]):.3f}')
        self.samples = samples

        # Sampler diagnostics: acceptance rates, step size used for sampling and effective sample size of k
        self.burnin_acceptance = float(np.mean(accepted[:self.num_burnin_steps])) if self.num_burnin_steps > 0 else float('nan')
        self.acceptance_rate = float(np.mean(accepted[self.num_burnin_steps:])) if samples else float('nan')
        self.final_step_size = float(kernel_results.new_step_size) if adapt_step_size else float(step_size)
        if len(samples) > 1:
            # Geyer's initial positive sequence: TFP's default (filter_threshold=0) stops at the first negative
            # autocorrelation and so returns exactly N whenever the lag-1 autocorrelation of k is negative
            self.k_ess = float(tfp.mcmc.effective_sample_size(tf.stack([self.k_from_state(s[1]) for s in samples]),
                                                              filter_beyond_positive_pairs=True))
        else:
            self.k_ess = float('nan')
        # Posterior samples of the learned noise levels (sigma_u, sigma_f)
        self.sigma_samples = (np.exp(np.stack([s[2].numpy() for s in samples])) if self.learn_noise and samples
                              else None)
        tqdm.write(f'HMC acceptance (sampling): {self.acceptance_rate:.3f}, burn-in: {self.burnin_acceptance:.3f}, '
                   f'step size: {self.final_step_size:.3e}, ESS(k): {self.k_ess:.1f}')

        # Define the MCMC sample chain
        # def trace_fn(state, results):
        #     step = results.step + 1
        #     tf.cond(
        #         tf.equal(step % 100, 0),
        #         lambda: tf.print("Step:", step, "Step size:", results.new_step_size, "Acceptance:", results.inner_results.is_accepted),
        #         lambda: tf.no_op()
        #     )
        #     return results

        # @tf.function
        # def run_mcmc():
        #     return tfp.mcmc.sample_chain(
        #         num_results=self.num_samples,
        #         num_burnin_steps=self.num_burnin_steps,
        #         current_state=self.params,  # Pass weights and biases as separate lists
        #         kernel=hmc_adaptive_kernel,
        #         trace_fn=trace_fn,
        #     )

        # self.samples, self.kernel_results = run_mcmc()


    def samples_predict(self, x):
        sampled_preds = [self.model(x, p[0]) for p in self.samples]
        sampled_preds = np.array(sampled_preds).squeeze()
        sampled_nu = np.array([self.k_from_state(p[1]) for p in self.samples]).squeeze()
        return sampled_preds, sampled_nu

    def samples_predict_residual(self, x):
        sampled_residuals = [self.physical_residual(x, p[0], self.k_from_state(p[1])) for p in self.samples]
        sampled_residuals = np.array(sampled_residuals).squeeze()
        return sampled_residuals


class DeepNeuralNetwork(tf.Module):
    def __init__(self, layer_sizes, activation=tanh, dropout_rate=0.0):
        super(DeepNeuralNetwork, self).__init__()
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate

        # One initializer instance per layer: a shared Keras 3 initializer gives identical equal-shaped layers
        self.weights = [tf.Variable(self.param_init_fn()([layer_sizes[i], layer_sizes[i + 1]]), dtype=tf.float32)
                        for i in range(len(layer_sizes) - 1)]
        self.biases = [tf.Variable(tf.zeros([layer_sizes[i + 1]]), dtype=tf.float32)
                       for i in range(len(layer_sizes) - 1)]

    def param_init_fn(self, inittializer='glorot_uniform'):
        if inittializer == 'glorot_uniform':
            init_fn = tf.initializers.GlorotUniform()
        elif inittializer == 'glorot_normal':
            init_fn = tf.initializers.GlorotNormal()
        elif inittializer == 'he_uniform':
            init_fn = tf.initializers.HeUniform()
        else:
            init_fn = tf.initializers.HeNormal()

        return init_fn

    def __call__(self, x, dropout=True):
        outputs = x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            outputs = self.activation(tf.matmul(outputs, w) + b)
            if dropout and self.dropout_rate > 0.0:
                outputs = tf.nn.dropout(outputs, rate=self.dropout_rate)
        return tf.matmul(outputs, self.weights[-1]) + self.biases[-1]
    
    @tf.function
    def train_step(self, x, y, optimizer):
        with tf.GradientTape() as tape:
            pred = self(x)
            loss = tf.reduce_mean(tf.square(y - pred))
        gradients = tape.gradient(loss, self.weights + self.biases)
        optimizer.apply_gradients(zip(gradients, self.weights + self.biases))
        return loss

    def train(self, x, y, epochs=1000, learning_rate=0.001, weight_decay=0.0, lr_decay_rate=None):
        # Adam at a constant learning rate (main.tex:213); weight decay and LR decay are opt-in
        optimizer = make_optimizer(learning_rate, epochs, weight_decay, lr_decay_rate)

        for epoch in tqdm(range(epochs), desc='Single Fidelity Training'):
            loss = self.train_step(x, y, optimizer)

            if (epoch + 1) % 1000 == 0:
                current_lr = float(np.asarray(optimizer.learning_rate))
                tqdm.write(f'Epoch {epoch+1}, LR: {current_lr}, Loss: {loss.numpy()}')

    def predict(self, x):
        pred = self(x, dropout=False)
        return pred


def make_optimizer(learning_rate, num_steps, weight_decay=0.0, lr_decay_rate=None):
    lr = learning_rate
    if lr_decay_rate is not None:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=max(1, num_steps // 10),
            decay_rate=lr_decay_rate,
            staircase=True
        )
    if weight_decay > 0.0:
        return tf.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    return tf.optimizers.Adam(learning_rate=lr)


class MultiFidelityNN(tf.Module):
    def __init__(self, input, width, depth, output, nu=0.0, lf_model=None, dropout_rate=0.0, alpha=None,
                 nonlinear_layers=None, linear_layers=None, l2_lambda=0.1, positive_k=False):
        super(MultiFidelityNN, self).__init__()
        # Inverse parameter k: one deterministic trainable scalar (a tf.Variable passed in is trained in place), or with
        # positive_k the trainable log k (k = exp(log k) > 0, initialized at the log of nu)
        self.positive_k = positive_k
        if positive_k:
            self.k_param = tf.Variable(float(np.log(float(tf.convert_to_tensor(nu, dtype=tf.float32)))),
                                       dtype=tf.float32, name='log_k')
        else:
            self.k_param = nu if isinstance(nu, tf.Variable) else tf.Variable(float(nu), dtype=tf.float32, name='k')
        self.k = self.k_param   # legacy alias: k itself unless positive_k
        self.nu = self.k
        self.dropout_rate = dropout_rate
        # alpha=None keeps the legacy sum F_nl + F_l; a float gives Eq. 3: alpha * F_l + (1 - alpha) * F_nl
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError(f'alpha must be in [0, 1], got {alpha}')
        self.alpha = None if alpha is None else float(alpha)
        self.l2_lambda = l2_lambda

        self.lf_model = lf_model
        if lf_model is None:
            layer_sizes = [input] + [width] * depth + [output]
            self.lf_model = DeepNeuralNetwork(layer_sizes, tanh, dropout_rate)

        # Explicit layer lists override the legacy width/depth rule
        if nonlinear_layers is None:
            nonlinear_layers = [input + output] + [width] * depth + [output]
        self.nonlinear_model = DeepNeuralNetwork(nonlinear_layers, tanh, dropout_rate)
        if linear_layers is None:
            linear_layers = [input + output] +  [width // 2] * (depth // 2) + [output]
        self.linear_model = DeepNeuralNetwork(linear_layers, tf.identity, dropout_rate)

        if input == 1:
            self.physical_residual = self.ode_residual
        else:
            self.physical_residual = self.pde_residual

    def k_value(self):
        return tf.exp(self.k_param) if self.positive_k else self.k_param

    def sample_nu(self):
        return self.k_value()

    def ode_residual(self, x, mc=True):
        x = tf.convert_to_tensor(x, dtype=tf.float32)

        with tf.GradientTape() as tape2:
            tape2.watch(x)
            with tf.GradientTape() as tape1:
                tape1.watch(x)
                u = self(x, mc=mc)
            u_x = tape1.gradient(u, x)
        u_xx = tape2.gradient(u_x, x)  # Second derivative of u with respect to x
        force_func = LAMBDA_PDE * u_xx + self.k_value() * tanh(u)
        return force_func

    def pde_residual(self, xt, mc=True):
        x = tf.convert_to_tensor(xt[:, 0:1], dtype=tf.float32)
        t = tf.convert_to_tensor(xt[:, 1:2], dtype=tf.float32)

        with tf.GradientTape() as tape2:
            tape2.watch([x, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, t])
                u = self(tf.concat([x, t], axis=1), mc=mc)
            u_x = tape1.gradient(u, x)
            u_t = tape1.gradient(u, t)
        u_xx = tape2.gradient(u_x, x)
        del tape1

        # Compute the PDE residual: u_t - nu * u_xx = 0
        residual = u_t - self.k_value() * u_xx
        return residual

    @tf.function
    def train_step(self, x_hf, u_hf, x_f, f_data, optimizer):
        with tf.GradientTape() as tape:
            pred_hf = self(x_hf)
            loss_hf = tf.reduce_mean(tf.square(u_hf - pred_hf))
            loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.linear_model.weights])
            pred_f = self.physical_residual(x_f)
            loss_residual = tf.reduce_mean(tf.square(pred_f - f_data))
            loss = loss_hf + loss_residual + loss_l2 * self.l2_lambda
        # Both HF heads and k are trained; the pretrained LF network stays frozen
        variables = (self.nonlinear_model.weights + self.nonlinear_model.biases +
                     self.linear_model.weights + self.linear_model.biases + [self.k_param])
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss, loss_hf, loss_residual, loss_l2

    def train(self, x_hf, u_hf, f_sensor, f_data, epochs=1000, learning_rate=0.001, weight_decay=0.0, lr_decay_rate=None):
        # Adam at a constant learning rate (main.tex:213); weight decay and LR decay are opt-in, never on k
        optimizer = make_optimizer(learning_rate, epochs, weight_decay, lr_decay_rate)
        if weight_decay > 0.0:
            optimizer.exclude_from_weight_decay(var_list=[self.k_param])
        for epoch in tqdm(range(epochs), desc='Multi-Fidelity Training'):
            loss, loss_hf, loss_f, loss_l2 = self.train_step(x_hf, u_hf, f_sensor, f_data, optimizer)

            if epoch % 1000 == 0:
                tqdm.write(f'Epoch {epoch}, Loss: {loss.numpy()}, loss_hf: {loss_hf.numpy()}, loss_f: {loss_f.numpy()}, loss_l2: {loss_l2.numpy()}, k: {self.k_value().numpy()}')

    def __call__(self, x_hf, mc=True):
        # The frozen LF network has no dropout; MC dropout acts on the hidden layers of the HF heads only
        pred_lf = self.lf_model(x_hf, dropout=False)
        x_combined = tf.concat([x_hf, pred_lf], axis=-1)
        y_nl = self.nonlinear_model(x_combined, dropout=mc)
        y_l = self.linear_model(x_combined, dropout=mc)
        if self.alpha is None:
            return y_nl + y_l
        return self.alpha * y_l + (1.0 - self.alpha) * y_nl

    def predict(self, x):
        pred = self(x, mc=False)
        return pred

    def predict_function(self, x):
        residual = self.physical_residual(x, mc=False)
        return residual

    def mc_predict(self, x, num_samples=100):
        predictions = [self(x, mc=True).numpy().flatten() for _ in range(num_samples)]
        return np.stack(predictions)

    def mc_predict_function(self, x, num_samples=100):
        residuals = [self.physical_residual(x, mc=True).numpy().flatten() for _ in range(num_samples)]
        return np.stack(residuals)

    def mc_predict_nu(self, x, num_samples=100):
        # k is a point estimate: its MC-dropout sd is reported as nan (3-tuple kept for existing callers)
        return self.mc_predict(x, num_samples), float(self.k_value().numpy()), float('nan')
