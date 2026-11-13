import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow_probability import distributions as tfd
from tensorflow.keras.activations import tanh, linear
from tqdm import tqdm

from model_bpinn import LAMBDA_PDE, select_residual, BayesianPINN


class GaussianVIBayesianPINN(BayesianPINN):
    """Variational inference with a Gaussian q over all unknowns z = [network parameters, k-state, (log sigma_u,
    log sigma_f)] whose covariance is diag(d^2) + V V^T (V: dim x rank; rank 0 is mean-field). Model, likelihood and
    prior are those of BayesianPINN. Loss: -E_q[log p(D | z)] + beta * KL(q||p) (Eq. 12, tempered with beta < 1), the
    expectation estimated from num_samples reparameterized draws and the KL to the Gaussian prior in closed form. With
    beta = 1, VI and HMC target the same posterior; beta < 1 targets the tempered posterior proportional to
    p(D | z)^(1/beta) p(z). A low-rank covariance can represent the posterior correlation between k and the network
    parameters that a mean-field q ignores."""

    def __init__(self, layer_sizes, nu, lf_model=None, sigma_obs=0.01, residual=None, positive_k=True,
                 logk_prior=(0.0, 1.0), learn_noise=False, noise_prior=(float(np.log(0.1)), 1.5), rank=10,
                 rho_init=-5.0, scalar_std_init=0.1, v_init_std=1e-3):
        super().__init__(layer_sizes, nu=nu, num_samples=0, num_burnin_steps=0, lf_model=lf_model,
                         sigma_obs=sigma_obs, residual=residual, positive_k=positive_k, logk_prior=logk_prior,
                         learn_noise=learn_noise, noise_prior=noise_prior)
        self.rank = int(rank)
        self.n_net = int(self.params.shape[0])
        self.n_noise = 2 if learn_noise else 0
        self.dim = self.n_net + 1 + self.n_noise
        # Prior N(m0, s0^2) in the unconstrained coordinates (as in BayesianPINN.log_prior)
        k_prior = self.logk_prior if positive_k else (0.0, 1.0)
        self.prior_mean = tf.constant([0.0] * self.n_net + [k_prior[0]] + [self.noise_prior[0]] * self.n_noise,
                                      tf.float32)
        self.prior_std = tf.constant([1.0] * self.n_net + [k_prior[1]] + [self.noise_prior[1]] * self.n_noise,
                                     tf.float32)
        # Variational parameters: mean, diagonal std softplus(rho) and low-rank factor V (small random, so the rank
        # columns differ from the start)
        rho_scalar = float(np.log(np.expm1(scalar_std_init)))
        self.q_mu = tf.Variable(tf.zeros([self.dim]), name='q_mu')
        self.q_rho = tf.Variable(tf.constant([rho_init] * self.n_net + [rho_scalar] * (1 + self.n_noise), tf.float32),
                                 name='q_rho')
        self.q_V = tf.Variable(tf.random.normal([self.dim, self.rank], stddev=v_init_std), name='q_V') \
            if self.rank > 0 else None
        self._vi_vars = [self.q_mu, self.q_rho] + ([self.q_V] if self.rank > 0 else [])
        self.set_mean_from_state()

    def variational_variables(self):
        return self._vi_vars

    def set_mean_from_state(self):
        # q mean <- current point (network parameters, k-state, log noise levels), e.g. the MAP of map_estimate
        parts = [tf.convert_to_tensor(self.params), tf.reshape(tf.convert_to_tensor(self.nu, tf.float32), [1])]
        if self.learn_noise:
            parts.append(tf.convert_to_tensor(self.log_sigma))
        self.q_mu.assign(tf.concat(parts, axis=0))

    def map_estimate(self, *args, **kwargs):
        k_map = super().map_estimate(*args, **kwargs)
        self.set_mean_from_state()
        return k_map

    def split(self, z):
        log_sigma = z[self.n_net + 1:] if self.learn_noise else None
        return z[:self.n_net], z[self.n_net], log_sigma

    def sample_z(self, n):
        z = self.q_mu[None, :] + tf.nn.softplus(self.q_rho)[None, :] * tf.random.normal([n, self.dim])
        if self.rank > 0:
            z += tf.matmul(tf.random.normal([n, self.rank]), self.q_V, transpose_b=True)
        return z

    def marginal_var(self, index):
        v = tf.square(tf.nn.softplus(self.q_rho[index]))
        if self.rank > 0:
            v += tf.reduce_sum(tf.square(self.q_V[index]))
        return v

    def kl_divergence(self):
        # KL(N(mu, D^2 + V V^T) || N(m0, S0^2)); log det via the matrix determinant lemma
        d2 = tf.square(tf.nn.softplus(self.q_rho))
        s02 = tf.square(self.prior_std)
        trace = tf.reduce_sum(d2 / s02)
        logdet_q = tf.reduce_sum(tf.math.log(d2))
        if self.rank > 0:
            trace += tf.reduce_sum(tf.square(self.q_V) / s02[:, None])
            inner = tf.eye(self.rank) + tf.matmul(self.q_V / d2[:, None], self.q_V, transpose_a=True)
            logdet_q += 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(inner))))
        quad = tf.reduce_sum(tf.square(self.q_mu - self.prior_mean) / s02)
        logdet_p = tf.reduce_sum(tf.math.log(s02))
        return 0.5 * (trace + quad - float(self.dim) + logdet_p - logdet_q)

    def log_likelihood_z(self, x_u, u, x_f, f, z):
        flat, nu_state, log_sigma = self.split(z)
        sigma_u, sigma_f = self.noise_levels(log_sigma)
        return self.log_likelihood(x_u, u, x_f, f, flat, self.k_from_state(nu_state), sigma_u, sigma_f)

    def neg_elbo(self, x_u, u, x_f, f, num_samples=10, beta=1.0):
        z = self.sample_z(num_samples)
        expected_ll = tf.add_n([self.log_likelihood_z(x_u, u, x_f, f, z[i]) for i in range(num_samples)]) / num_samples
        kl = self.kl_divergence()
        return -expected_ll + beta * kl, expected_ll, kl

    def train(self, x_u, u, x_f, f, num_steps=10000, learning_rate=0.001, num_samples=10, objective='elbo', beta=1.0,
              lr_decay_rate=None):
        if objective != 'elbo':
            raise ValueError("GaussianVIBayesianPINN supports objective='elbo' only")
        lr = learning_rate
        if lr_decay_rate is not None:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate, decay_steps=max(1, num_steps // 10), decay_rate=lr_decay_rate,
                staircase=False)
        optimizer = tf.optimizers.Adam(learning_rate=lr)

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                loss, expected_ll, kl = self.neg_elbo(x_u, u, x_f, f, num_samples, beta)
            grads = tape.gradient(loss, self._vi_vars)
            optimizer.apply_gradients(zip(grads, self._vi_vars))
            return loss, expected_ll, kl

        self.vi_losses = []
        self.vi_kl_history = []   # (step, closed-form KL) every 1000 steps
        for step in tqdm(range(num_steps), desc='VI Training'):
            loss, expected_ll, kl = train_step()
            if (step + 1) % 1000 == 0 or step + 1 == num_steps:
                k_mean, k_sd = self.nu_posterior()
                self.vi_kl_history.append((step + 1, float(kl)))
                tqdm.write(f'Step {step + 1}, loss: {loss.numpy():.4e}, E log p(D|z): {expected_ll.numpy():.4e}, '
                           f'KL: {kl.numpy():.4e}, k: {k_mean:.4f} +- {k_sd:.4f}')
            self.vi_losses.append(float(loss))
        # q settles slowly (the KL keeps falling while the variances grow): flag a run whose KL still drifts, as in
        # the composite-application VI (change of the mean KL between the last two tenths of the logged values,
        # relative to the final one)
        self.vi_kl_drift = self.kl_drift()
        if not np.isfinite(self.vi_kl_drift):
            print('Note: KL drift not computed (fewer than 2 logged KL values; the check needs >= 2000 VI steps)')
        elif abs(self.vi_kl_drift) > 0.02:
            print(f'Warning: the VI has not converged (KL changed by {100 * self.vi_kl_drift:.1f}% of its final value '
                  f'between the last two tenths of the run); use more steps')

    def kl_drift(self):
        # (mean KL of the last tenth - mean KL of the tenth before) / mean KL of the last tenth of the KL values logged
        # every 1000 steps (one value each below 20000 steps); nan with fewer than 2 logged values
        kl_logged = np.asarray([kl for _, kl in self.vi_kl_history])
        if len(kl_logged) < 2:
            return float('nan')
        n = max(1, len(kl_logged) // 10)
        last, before = kl_logged[-n:].mean(), kl_logged[-2 * n:-n].mean()
        return float((last - before) / last)

    def samples_predict(self, x, num_samples=2000):
        z = self.sample_z(num_samples)
        u_samples, f_samples, k_samples, sigma_samples = [], [], [], []
        for i in range(num_samples):
            flat, nu_state, log_sigma = self.split(z[i])
            k = self.k_from_state(nu_state)
            u_samples.append(self.model(x, flat).numpy())
            f_samples.append(self.physical_residual(x, flat, k).numpy())
            k_samples.append(float(k.numpy()))
            if self.learn_noise:
                sigma_samples.append(np.exp(log_sigma.numpy()))
        self.sigma_samples = np.stack(sigma_samples) if sigma_samples else None
        return np.stack(u_samples).squeeze(), np.stack(f_samples).squeeze(), np.asarray(k_samples)

    def nu_posterior(self):
        # Mean and sd of q(k) (log-normal moments with positive_k), including the low-rank part of the variance
        m = float(self.q_mu[self.n_net].numpy())
        v = float(self.marginal_var(self.n_net).numpy())
        if self.positive_k:
            mean = float(np.exp(m + 0.5 * v))
            return mean, float(mean * np.sqrt(np.expm1(v)))
        return m, float(np.sqrt(v))

class BayesianLayer(tf.Module):
    def __init__(self, in_dim, out_dim, prior_mean=0.0, prior_std=1.0, activation=tf.nn.tanh):
        super(BayesianLayer, self).__init__()
        
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.prior = tfd.Normal(self.prior_mean, self.prior_std)

        # Variational parameters for weights and bias
        self.weight_mu = tf.Variable(tf.random.normal([in_dim, out_dim], mean=0.0, stddev=0.1), name='weight_mu')
        self.weight_rho = tf.Variable(tf.random.normal([in_dim, out_dim], mean=-5.0, stddev=0.1), name='weight_rho')
        self.bias_mu = tf.Variable(tf.zeros([out_dim]), name='bias_mu')
        self.bias_rho = tf.Variable(tf.random.normal([out_dim], mean=-5.0, stddev=0.1), name='bias_rho')

        self.activation = activation

    def sample_weights(self):
        weight_sigma = tf.nn.softplus(self.weight_rho)
        bias_sigma = tf.nn.softplus(self.bias_rho)

        weight = self.weight_mu + weight_sigma * tf.random.normal(tf.shape(weight_sigma))
        bias = self.bias_mu + bias_sigma * tf.random.normal(tf.shape(bias_sigma))

        return weight, bias

    def model(self, x, weight, bias):
        out = self.activation(tf.matmul(x, weight) + bias)
        return out
    
    def log_prior(self, weight, bias):
        lg_pr = tf.reduce_sum(self.prior.log_prob(weight))
        lg_pr += tf.reduce_sum(self.prior.log_prob(bias))
        return lg_pr
    
    def log_variational_posterior(self, weight, bias):
        weight_sigma = tf.nn.softplus(self.weight_rho)
        bias_sigma = tf.nn.softplus(self.bias_rho)
        lg_post = tf.reduce_sum(tfd.Normal(self.weight_mu, weight_sigma).log_prob(weight))
        lg_post += tf.reduce_sum(tfd.Normal(self.bias_mu, bias_sigma).log_prob(bias))
        return lg_post
    
    def kl_loss(self):
        weight_sigma = tf.nn.softplus(self.weight_rho)
        bias_sigma = tf.nn.softplus(self.bias_rho)
        weight_variational_posterior = tfd.Normal(self.weight_mu, weight_sigma)
        bias_variational_posterior = tfd.Normal(self.bias_mu, bias_sigma)
        
        kl_weight = tfd.kl_divergence(weight_variational_posterior, self.prior)
        kl_bias = tfd.kl_divergence(bias_variational_posterior, self.prior)

        kl_layer = tf.reduce_sum(kl_weight) + tf.reduce_sum(kl_bias)
        return kl_layer
    
    # def kl_loss(self):
    #     weight_sigma = tf.nn.softplus(self.weight_rho)
    #     bias_sigma = tf.nn.softplus(self.bias_rho)
        
    #     kl_weight = tf.math.log(self.prior_std / weight_sigma) + \
    #     (weight_sigma**2 + (self.weight_mu - self.prior_mean)**2)/(2 * self.prior_std**2) - 0.5
        
    #     kl_bias = tf.math.log(self.prior_std / bias_sigma) + \
    #     (bias_sigma**2 + (self.bias_mu - self.prior_mean)**2)/(2 * self.prior_std**2) - 0.5
        
    #     kl_layer = tf.reduce_sum(kl_weight) + tf.reduce_sum(kl_bias)
    #     return kl_layer

class BayesianPINNVI(tf.Module):
    def __init__(self, layer_sizes, nu, prior_mean=0.0, prior_std=1.0, lowfidelity_model=None, sigma_obs=0.01,
                 residual=None):
        super(BayesianPINNVI, self).__init__()

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.loss_func = tf.keras.losses.MeanSquaredError()

        self.lf_model = lowfidelity_model
        self.sigma_obs = tf.convert_to_tensor(sigma_obs, dtype=tf.float32)

        self.layers = []
        for i in range(self.num_layers - 1):
            self.layers.append(BayesianLayer(layer_sizes[i], layer_sizes[i+1], prior_mean, prior_std, tf.nn.tanh))
        self.layers.append(BayesianLayer(layer_sizes[-2], layer_sizes[-1], prior_mean, prior_std, linear))
        
        if select_residual(residual, layer_sizes[0]) == 'ode':
            self.physical_residual = self.ode_function
        else:
            self.physical_residual = self.pde_function

        self.is_forward = True
        if isinstance(nu, tf.Variable):
            # Variational parameters for the PDE parameter nu.
            self.nu_mu = tf.Variable(tf.cast(nu, tf.float32), name='nu_mu')
            self.nu_rho = tf.Variable(tf.ones([], dtype=tf.float32), name='nu_rho')
            self.is_forward = False
        else:    
            self.nu = tf.convert_to_tensor(nu, dtype=tf.float32)

        # Only the variational parameters are optimised: tf.Module would also collect the attached LF network,
        # which must stay frozen (no stop_gradient on its output, since u_x and u_xx need du_LF/dx)
        self._vi_vars = [var for layer in self.layers
                         for var in (layer.weight_mu, layer.weight_rho, layer.bias_mu, layer.bias_rho)]
        if not self.is_forward:
            self._vi_vars += [self.nu_mu, self.nu_rho]

    def variational_variables(self):
        return self._vi_vars

    def map_estimate(self, x_u, u, x_f, f, steps=10000, learning_rate=1e-3, restarts=1):
        # MAP warm start of the variational means: Adam on -[log p(D | mu) + log p(mu)] with the network evaluated at
        # its means (the standard deviations are not trained here), from `restarts` random initializations of the
        # means; the restart with the highest posterior is kept, so the ELBO optimisation starts in the MAP mode
        means = [var for layer in self.layers for var in (layer.weight_mu, layer.bias_mu)]
        if not self.is_forward:
            means.append(self.nu_mu)
        nu_init = None if self.is_forward else self.nu_mu.numpy()

        def neg_log_posterior():
            weights = [layer.weight_mu for layer in self.layers]
            biases = [layer.bias_mu for layer in self.layers]
            nu = self.nu if self.is_forward else self.nu_mu
            return -(self.log_likelihood(x_u, u, x_f, f, weights, biases, nu) + self.log_prior(weights, biases, nu))

        best = None
        self.map_history = []   # (-log posterior, nu) of every restart
        for restart in range(restarts):
            if restart > 0:
                # Same initialization as BayesianLayer: weight means N(0, 0.1^2), bias means 0
                for layer in self.layers:
                    layer.weight_mu.assign(tf.random.normal(layer.weight_mu.shape, mean=0.0, stddev=0.1))
                    layer.bias_mu.assign(tf.zeros(layer.bias_mu.shape))
                if not self.is_forward:
                    self.nu_mu.assign(nu_init)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            @tf.function
            def map_step():
                with tf.GradientTape() as tape:
                    loss = neg_log_posterior()
                grads = tape.gradient(loss, means)
                optimizer.apply_gradients(zip(grads, means))
                return loss

            for step in tqdm(range(steps), desc=f'MAP warm start {restart + 1}/{restarts}'):
                map_step()
            loss = float(neg_log_posterior())
            k_value = float(self.nu.numpy()) if self.is_forward else float(self.nu_mu.numpy())
            tqdm.write(f'MAP restart {restart + 1}: -log posterior {loss:.4e}, k {k_value:.4f}')
            self.map_history.append((loss, k_value))
            if np.isfinite(loss) and (best is None or loss < best[0]):
                best = (loss, [var.numpy() for var in means])
        if best is None:
            raise RuntimeError('no MAP restart gave a finite -log posterior')
        for var, value in zip(means, best[1]):
            var.assign(value)
        self.map_nu = float(self.nu.numpy()) if self.is_forward else float(self.nu_mu.numpy())
        return self.map_nu

    def nu_posterior(self):
        # Analytic mean and sd of q(k) (no Monte Carlo error); a fixed nu has sd 0
        if self.is_forward:
            return float(self.nu.numpy()), 0.0
        return float(self.nu_mu.numpy()), float(tf.nn.softplus(self.nu_rho).numpy())

    def sample_weights(self):
        weights_sample = []
        biases_sample = []
        for layer in self.layers:
            weight, bias = layer.sample_weights()
            weights_sample.append(weight)
            biases_sample.append(bias)

        if self.is_forward:
            nu_sample = self.nu
        else:
            nu_sigma = tf.nn.softplus(self.nu_rho)
            nu_sample = self.nu_mu + nu_sigma * tf.random.normal([])

        return weights_sample, biases_sample, nu_sample
    
    def model(self, x, weights, biases):
        if self.lf_model is not None:
            out = tf.concat([x, self.lf_model(x)], axis=-1)
        else:
            out = x

        for i in range(self.num_layers):
            out = self.layers[i].model(out, weights[i], biases[i])
        return out
        
    def kl_loss(self):
        kl_div = 0.0
        for layer in self.layers:
            kl_div += layer.kl_loss()

        num_parameters = tf.reduce_sum([tf.reduce_prod(var.shape) for var in self._vi_vars])
        kl_div = kl_div / tf.cast(num_parameters, tf.float32)
        return kl_div
     
    def ode_function(self, x_in, weights, biases, nu):
        x = tf.convert_to_tensor(x_in, dtype=tf.float32)
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            with tf.GradientTape() as tape1:
                tape1.watch(x)
                u = self.model(x, weights, biases)
            u_x = tape1.gradient(u, x)
        u_xx = tape2.gradient(u_x, x)
        f = LAMBDA_PDE * u_xx + nu * tf.tanh(u)
        # f = -1/400 * u_xx + u
        return f
    
    def pde_function(self, x_in, weights, biases, nu):
        x = tf.convert_to_tensor(x_in[:, 0:1], dtype=tf.float32)
        t = tf.convert_to_tensor(x_in[:, 1:2], dtype=tf.float32)
        with tf.GradientTape() as tape2:
            tape2.watch([x, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, t])
                u = self.model(tf.concat([x, t], axis=-1), weights, biases)
            u_x = tape1.gradient(u, x)
            u_t = tape1.gradient(u, t)
        u_xx = tape2.gradient(u_x, x)
        f = u_t - nu * u_xx

        del tape1
        return f
    
    def log_prior(self, weights, biases, nu):
        lg_pr = 0.0
        for i in range(self.num_layers):
            lg_pr += self.layers[i].log_prior(weights[i], biases[i])
        if self.is_forward == False:
            lg_pr += tfd.Normal(0.0, 1.0).log_prob(nu)
        return lg_pr

    def log_variational_posterior(self, weights, biases, nu):
        lg_post = 0.0
        for i in range(self.num_layers):
            lg_post += self.layers[i].log_variational_posterior(weights[i], biases[i])

        if self.is_forward == False:
            nu_sigma = tf.nn.softplus(self.nu_rho)
            lg_post += tfd.Normal(self.nu_mu, nu_sigma).log_prob(nu)
        return lg_post
    
    def log_likelihood(self, x_u, u, x_f, f, weights, biases, nu):
        pred_u = self.model(x_u, weights, biases)
        ll_u = tf.reduce_sum(tfd.Normal(pred_u, self.sigma_obs).log_prob(u))
        res = self.physical_residual(x_f, weights, biases, nu)
        ll_f = tf.reduce_sum(tfd.Normal(res, self.sigma_obs).log_prob(f))
        return ll_u + ll_f
    
    # Negative evidence low
    def elbo(self, x_u, u, x_f, f, num_samples=10, beta=1.0):
        elbo_sum = 0.0
        log_likelihood = 0.0
        log_prior = 0.0
        log_variational_posterior = 0.0
        for _ in range(num_samples):
            weights_sample, biases_sample, nu_sample = self.sample_weights()
            log_likelihood += self.log_likelihood(x_u, u, x_f, f, weights_sample, biases_sample, nu_sample)
            log_prior += self.log_prior(weights_sample, biases_sample, nu_sample)
            log_variational_posterior += self.log_variational_posterior(weights_sample, biases_sample, nu_sample)
        elbo_sum = -(log_likelihood + beta * (log_prior - log_variational_posterior))
            
        return elbo_sum / num_samples, log_likelihood / num_samples, log_prior / num_samples, log_variational_posterior / num_samples
    
    @tf.function
    def train_step_sample(self, x_u, u, x_f, f, optimizer, num_samples, beta=1.0):
        with tf.GradientTape() as tape:
            loss, llh, log_pior, log_post = self.elbo(x_u, u, x_f, f, num_samples, beta)
        loss_grads = tape.gradient(loss, self._vi_vars)
        optimizer.apply_gradients(zip(loss_grads, self._vi_vars))
        return loss, llh, log_pior, log_post

    # Legacy-form tempered surrogate MSE_u + MSE_f + 0.9 * KL / N (objective='tempered_mse'). Not an exact reproduction
    # of the old runs: N counts the variational parameters only and the LF stays frozen (the old code fine-tuned the
    # LF and counted its weights in N)
    @tf.function
    def train_step(self, x_u, u, x_f, f, optimizer, num_samples):
        loss_u, loss_res, loss_kl = 0.0, 0.0, 0.0
        with tf.GradientTape() as tape:
            for _ in range(num_samples):
                weights_sample, biases_sample, nu_sample = self.sample_weights()
                pred_u = self.model(x_u, weights_sample, biases_sample)
                loss_u += self.loss_func(pred_u, u)
                residual = self.physical_residual(x_f, weights_sample, biases_sample, nu_sample)
                loss_res += self.loss_func(residual, f)
                loss_kl += self.kl_loss()
            loss = loss_u  + loss_res + 0.9 * loss_kl
            loss = loss / num_samples
        loss_grads = tape.gradient(loss, self._vi_vars)
        optimizer.apply_gradients(zip(loss_grads, self._vi_vars))
        return loss, loss_u, loss_res, loss_kl

    def train(self, x_u, u, x_f, f, num_steps=2000, learning_rate=0.001, num_samples=10, objective='elbo', beta=1.0,
              lr_decay_rate=None):
        # Adam at a constant learning rate; lr_decay_rate=0.95 gives the legacy continuous decay
        lr = learning_rate
        if lr_decay_rate is not None:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=max(1, num_steps // 10),
                decay_rate=lr_decay_rate,
                staircase=False
            )
        optimizer = tf.optimizers.Adam(learning_rate=lr)
        self.vi_losses = []
        for step in tqdm(range(num_steps), desc="VI Training"):
            if objective == 'elbo':
                # Eq. 13: -ELBO = 1/K sum_i [log q(theta_i) - log p(theta_i) - log p(D | theta_i)]
                loss, llh, log_prior, log_post = self.train_step_sample(x_u, u, x_f, f, optimizer, num_samples, beta)
                if (step + 1) % 1000 == 0:
                    tqdm.write(f"Step {step+1}, -ELBO: {loss.numpy()}, E log p(D|theta): {llh.numpy()}, E log p(theta): {log_prior.numpy()}, E log q: {log_post.numpy()}")
            elif objective == 'tempered_mse':
                loss, loss_u, loss_res, loss_kl = self.train_step(x_u, u, x_f, f, optimizer, num_samples)
                if (step + 1) % 1000 == 0:
                    tqdm.write(f"Step {step+1}, loss: {loss.numpy()}, loss_u: {loss_u.numpy()}, loss_residual: {loss_res.numpy()}, loss_kl: {loss_kl.numpy()}")
            else:
                raise ValueError(f"objective must be 'elbo' or 'tempered_mse', got {objective!r}")
            self.vi_losses.append(float(loss))
                
    def samples_predict(self, x, num_samples=50):
        pred_samples = []
        nu_samples = []
        f_samples = []
        for _ in range(num_samples):
            weights_sample, biases_sample, nu_sample = self.sample_weights()
            pred_samples.append(self.model(x, weights_sample, biases_sample))
            f_samples.append(self.physical_residual(x, weights_sample, biases_sample, nu_sample))
            nu_samples.append(nu_sample)
        pred_samples = tf.stack(pred_samples, axis=0).numpy().squeeze()
        f_samples = tf.stack(f_samples, axis=0).numpy().squeeze()
        nu_samples = tf.stack(nu_samples, axis=0).numpy().squeeze()
        return pred_samples, f_samples, nu_samples

class BayesianPINN_VI(tf.Module):
    def __init__(self, layer_sizes, nu=0.7):
        super(BayesianPINN_VI, self).__init__()
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        # Variational parameters for weights and biases for each layer.
        self.weight_mu = []
        self.weight_rho = []
        self.bias_mu = []
        self.bias_rho = []
        for i in range(self.num_layers):
            # Initialize means and rhos for weights
            w_mu = tf.Variable(tf.random.normal([layer_sizes[i], layer_sizes[i+1]], stddev=0.1),
                               name=f'w_mu_{i}')
            w_rho = tf.Variable(tf.random.normal([layer_sizes[i], layer_sizes[i+1]], stddev=0.1),
                                name=f'w_rho_{i}')
            # Initialize means and rhos for biases
            b_mu = tf.Variable(tf.zeros([layer_sizes[i+1]]), name=f'b_mu_{i}')
            b_rho = tf.Variable(tf.random.normal([layer_sizes[i+1]], stddev=0.1),
                                name=f'b_rho_{i}')
            self.weight_mu.append(w_mu)
            self.weight_rho.append(w_rho)
            self.bias_mu.append(b_mu)
            self.bias_rho.append(b_rho)

        self.is_forward = True
        if isinstance(nu, tf.Variable):
            # Variational parameters for the PDE parameter nu.
            self.nu_mu = tf.Variable(tf.cast(nu, tf.float32), name='nu_mu')
            self.nu_rho = tf.Variable(tf.ones([], dtype=tf.float32), name='nu_rho')
            self.is_forward = False
        else:    
            self.nu = tf.convert_to_tensor(nu, dtype=tf.float32)

    def sample_weights(self):
        """Sample network weights, biases, and nu from the variational posterior."""
        weights_sample = []
        biases_sample = []
        for i in range(self.num_layers):
            # Use softplus to ensure positivity of the standard deviation.
            w_std = tf.nn.softplus(self.weight_rho[i])
            b_std = tf.nn.softplus(self.bias_rho[i])
            w_sample = self.weight_mu[i] + w_std * tf.random.normal(tf.shape(self.weight_mu[i]))
            b_sample = self.bias_mu[i] + b_std * tf.random.normal(tf.shape(self.bias_mu[i]))
            weights_sample.append(w_sample)
            biases_sample.append(b_sample)
        
        if self.is_forward:
            nu_sample = self.nu
        else:
            nu_std = tf.nn.softplus(self.nu_rho)
            nu_sample = self.nu_mu + nu_std * tf.random.normal([])

        return weights_sample, biases_sample, nu_sample

    def model(self, x, weights, biases):
        """Feedforward network with tanh activation in hidden layers and a linear output."""
        out = x
        for i in range(self.num_layers - 1):
            out = tf.tanh(tf.matmul(out, weights[i]) + biases[i])
        out = tf.matmul(out, weights[-1]) + biases[-1]
        return out

    def ode_residual(self, x, weights, biases, nu):
        """Compute the PDE residual at points x."""
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            with tf.GradientTape() as tape1:
                tape1.watch(x)
                u = self.model(x, weights, biases)
            u_x = tape1.gradient(u, x)[:, :1]
        u_xx = tape2.gradient(u_x, x)[:, :1]
        return 0.01 * u_xx + nu * tf.tanh(u)

    def log_prior(self, weights, biases, nu):
        """Log prior probability assuming independent standard Normal distributions."""
        lp = 0.0
        for w in weights:
            lp += tf.reduce_sum(tfd.Normal(0., 1.).log_prob(w))
        for b in biases:
            lp += tf.reduce_sum(tfd.Normal(0., 1.).log_prob(b))
        lp += tfd.Normal(0., 1.).log_prob(nu)
        return lp

    def log_likelihood(self, x_u, u, x_f, f, weights, biases, nu):
        """Compute the log likelihood for the data and PDE residual."""
        pred_u = self.model(x_u, weights, biases)
        ll_u = tf.reduce_sum(tfd.Normal(pred_u, 0.01).log_prob(u))
        res = self.ode_residual(x_f, weights, biases, nu)
        ll_f = tf.reduce_sum(tfd.Normal(res, 0.01).log_prob(f))
        return ll_u + ll_f

    def elbo(self, x_u, u, x_f, f, num_samples=10):
        """Monte Carlo estimate of the Evidence Lower Bound (ELBO)."""
        elbo_sum = 0.0
        for _ in range(num_samples):
            weights_sample, biases_sample, nu_sample = self.sample_weights()
            log_lik = self.log_likelihood(x_u, u, x_f, f, weights_sample, biases_sample, nu_sample)
            log_pr = self.log_prior(weights_sample, biases_sample, nu_sample)
            # Compute log q (variational posterior) for each parameter.
            log_q = 0.0
            for i in range(self.num_layers):
                w_std = tf.nn.softplus(self.weight_rho[i])
                b_std = tf.nn.softplus(self.bias_rho[i])
                log_q += tf.reduce_sum(tfd.Normal(self.weight_mu[i], w_std).log_prob(weights_sample[i]))
                log_q += tf.reduce_sum(tfd.Normal(self.bias_mu[i], b_std).log_prob(biases_sample[i]))

            if self.is_forward == False:
                nu_std = tf.nn.softplus(self.nu_rho)
                log_q += tfd.Normal(self.nu_mu, nu_std).log_prob(nu_sample)
            
            elbo_sum += (log_lik + log_pr - log_q)
        return elbo_sum / tf.cast(num_samples, tf.float32)

    @tf.function
    def train_step(self, x_u, u, x_f, f, optimizer, num_samples):
        """A single training step: compute the loss, gradients, and update parameters."""
        with tf.GradientTape() as tape:
            loss = -self.elbo(x_u, u, x_f, f, num_samples=num_samples)
        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, x_u, u, x_f, f, num_steps=1000, learning_rate=0.01, num_samples=10):
        """Train the network by maximizing the ELBO."""
        optimizer = tf.optimizers.Adam(learning_rate)
        losses = []
        for step in tqdm(range(num_steps), desc="Bayesian PINN Training"):
            loss = self.train_step(x_u, u, x_f, f, optimizer, num_samples)
            losses.append(loss.numpy())
            if (step + 1) % 1000 == 0:
                tqdm.write(f"Step {step+1}, Loss: {loss.numpy()}")
        self.vi_losses = losses

    def samples_predict(self, x, num_samples=100):
        """Generate predictions by averaging over samples from the variational posterior."""
        preds = []
        nu_samples = []
        for _ in range(num_samples):
            weights_sample, biases_sample, nu_sample = self.sample_weights()
            preds.append(self.model(x, weights_sample, biases_sample))
            nu_samples.append(nu_sample)
        preds = tf.stack(preds, axis=0).numpy().squeeze()
        nu_samples = tf.stack(nu_samples, axis=0).numpy().squeeze()
        return preds, nu_samples
