import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow.keras.activations import tanh, linear
from tqdm import tqdm


np.random.seed(42)
tf.random.set_seed(42)

class BayesianNeuralNetwork(tf.Module):
    def __init__(self, layer_sizes, num_samples=2000, num_burnin_steps=2000, sigma_obs=0.01, prior_std=1.0):
        super(BayesianNeuralNetwork, self).__init__()
        self.num_samples = num_samples
        self.num_burnin_steps = num_burnin_steps
        self.layer_sizes = layer_sizes
        # Likelihood y ~ N(f(x; theta), sigma_obs^2) and i.i.d. prior N(0, prior_std^2) on all weights and biases
        self.sigma_obs = float(sigma_obs)
        self.prior_std = float(prior_std)

        # self.weights = [tf.Variable(tf.random.normal([layer_sizes[i], layer_sizes[i + 1]]), dtype=tf.float32)
        #                 for i in range(len(layer_sizes) - 1)]
        # self.biases = [tf.Variable(tf.random.normal([layer_sizes[i + 1]]), dtype=tf.float32)
        #                for i in range(len(layer_sizes) - 1)]
        self.shapes = [((layer_sizes[i], layer_sizes[i + 1]), (layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]
        num_params = sum((layer_sizes[i] + 1) * layer_sizes[i+1] for i in range(len(layer_sizes) - 1))
        # self.params = tf.Variable(tf.random.normal([num_params], stddev=1.0), dtype=tf.float32)
        self.params = tf.Variable(tf.zeros([num_params]))

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
        outputs = x
        for w, b in zip(weights[:-1], biases[:-1]):
            outputs = tanh(tf.matmul(outputs, w) + b)
        return tf.matmul(outputs, weights[-1]) + biases[-1]

    def joint_log_prob(self, x, y, flat_params):
        log_prob = tf.reduce_sum(tfd.Normal(loc=0., scale=self.prior_std).log_prob(flat_params))

        pred = self.model(x, flat_params)
        likelihood = 0.0  # Initialize likelihood before using it
        likelihood += tf.reduce_sum(tfd.Normal(loc=pred, scale=self.sigma_obs).log_prob(y))
        return log_prob + likelihood

    def train(self, x, y, step_size=0.01, leapfrog_steps=80, adapt_step_size=True, target_accept_prob=0.9):
        # step_size_var = tf.Variable(step_size, dtype=tf.float32)
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=lambda *args: self.joint_log_prob(x, y, *args),
            step_size=step_size,
            num_leapfrog_steps=leapfrog_steps
        )

        # The step size starts at step_size and is adapted during the first 80% of burn-in;
        # adapt_step_size=False runs plain HMC at the fixed step_size
        if adapt_step_size:
            hmc_adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                hmc_kernel,
                num_adaptation_steps=int(self.num_burnin_steps*0.8),
                target_accept_prob=target_accept_prob
            )
        else:
            hmc_adaptive_kernel = hmc_kernel

        # Process sampling step by step
        @tf.function
        def one_step(state, kernel_results):
            return hmc_adaptive_kernel.one_step(state, kernel_results)

        samples = []
        accepted = []
        state = tf.convert_to_tensor(self.params)
        kernel_results = hmc_adaptive_kernel.bootstrap_results(state)
        total_steps = self.num_samples + self.num_burnin_steps
        for step in tqdm(range(total_steps), desc='Sampling'):
            state, kernel_results = one_step(state, kernel_results)
            accepted.append(kernel_results.inner_results.is_accepted if adapt_step_size else kernel_results.is_accepted)
            # Keep exactly num_samples states after num_burnin_steps transitions
            if step >= self.num_burnin_steps:
                samples.append(state)

            if (step + 1) % 100 == 0:
                rate = np.mean([a.numpy() for a in accepted[-100:]])
                tqdm.write(f'Step {step + 1}, Acceptance (last 100 steps): {rate:.3f}'
                           + (f', Step size: {float(kernel_results.new_step_size):.4g}' if adapt_step_size else ''))
        self.params.assign(state)
        self.samples = np.array(samples)

        # Sampler diagnostics: mean acceptance during burn-in and over the kept samples, and the step size used for sampling
        accepted = np.array([a.numpy() for a in accepted], dtype=np.float32)
        self.burnin_acceptance = float(accepted[:self.num_burnin_steps].mean()) if self.num_burnin_steps > 0 else float('nan')
        self.acceptance_rate = float(accepted[self.num_burnin_steps:].mean()) if self.num_samples > 0 else float('nan')
        self.final_step_size = float(kernel_results.new_step_size) if adapt_step_size else float(step_size)
        tqdm.write(f'HMC: kept {len(samples)} samples, acceptance {self.acceptance_rate:.3f} '
                   f'(burn-in {self.burnin_acceptance:.3f}), step size {self.final_step_size:.4g}')
        if not 0.2 <= self.acceptance_rate <= 0.99:
            tqdm.write(f'Warning: HMC acceptance rate {self.acceptance_rate:.3f} is outside [0.2, 0.99]')

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
        sampled_preds = [self.model(x, p) for p in self.samples]
        sampled_preds = np.array(sampled_preds).squeeze()
        return sampled_preds


class DeepNeuralNetwork(tf.Module):
    def __init__(self, layer_sizes, activation=tanh):
        super(DeepNeuralNetwork, self).__init__()
        self.layer_sizes = layer_sizes
        self.activation = activation

        # One initializer per layer: a Keras-3 initializer fixes its seed (drawn from Python's random module) when
        # it is created, so a shared instance gives equal-shaped layers identical weights.
        # tf.keras.utils.set_random_seed makes the weights reproducible.
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

    def __call__(self, x):
        outputs = x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            outputs = self.activation(tf.matmul(outputs, w) + b)
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
        # Adam at a constant learning rate; the staircase decay and AdamW weight decay are opt-in
        lr_schedule = learning_rate
        if lr_decay_rate is not None:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=max(1, epochs // 10),
                decay_rate=lr_decay_rate,
                staircase=True
            )
        if weight_decay > 0:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        for epoch in tqdm(range(epochs), desc='Single Fidelity Training'):
            loss = self.train_step(x, y, optimizer)

            if (epoch + 1) % 1000 == 0:
                current_lr = float(optimizer.learning_rate)
                tqdm.write(f'Epoch {epoch+1}, LR: {current_lr}, Loss: {loss.numpy()}')

    def predict(self, x):
        pred = self(x)
        return pred
    
class MultiFidelityNN(tf.Module):
    def __init__(self, input, width, depth, output, lf_model=None, alpha=None,
                 nonlinear_layers=None, linear_layers=None, l2_lambda=1.0):
        super(MultiFidelityNN, self).__init__()

        # Eq. 3: y_HF = alpha * F_l + (1 - alpha) * F_nl; alpha=None keeps the legacy unweighted sum F_nl + F_l
        if alpha is not None:
            alpha = float(alpha)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f'alpha must lie in [0, 1], got {alpha}')
        self.alpha = alpha
        self.l2_lambda = l2_lambda

        self.lf_model = lf_model
        if lf_model is None:
            layer_sizes = [input] + [width] * depth + [output]
            self.lf_model = DeepNeuralNetwork(layer_sizes)

        # Explicit layer lists override the legacy width/depth rule
        if nonlinear_layers is None:
            nonlinear_layers = [input + output] + [width] * depth + [output]
        self.nonlinear_model = DeepNeuralNetwork(nonlinear_layers)
        if linear_layers is None:
            linear_layers = [input + output] +  [width // 2] * (depth // 2) + [output]
        self.linear_model = DeepNeuralNetwork(linear_layers, linear)

    @tf.function
    def train_step(self, x_hf, y_hf, optimizer):
        with tf.GradientTape() as tape:
            pred_hf = self(x_hf)
            loss_hf = tf.reduce_mean(tf.square(y_hf - pred_hf))
            # lambda * L_reg, with L_reg = sum of 0.5*||W||^2 over the linear-head weights
            loss_l2 = self.l2_lambda * tf.add_n([tf.nn.l2_loss(v) for v in self.linear_model.weights])
            loss = loss_hf + loss_l2
        variables = self.nonlinear_model.weights + self.nonlinear_model.biases + self.linear_model.weights + self.linear_model.biases #+ self.lf_model.weights + self.lf_model.biases
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss, loss_hf, loss_l2

    def train(self, x_lf, y_lf, x_hf, y_hf, epochs=1000, learning_rate=0.001, weight_decay=0.0):
        # Second stage of the two-stage scheme: the pretrained LF network stays frozen and x_lf/y_lf only report its fit
        if x_lf is not None and y_lf is not None:
            loss_lf = tf.reduce_mean(tf.square(self.lf_model(x_lf) - y_lf))
            tqdm.write(f'Frozen low-fidelity model, loss_lf: {loss_lf.numpy()}')
        if weight_decay > 0:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        for epoch in tqdm(range(epochs), desc='Multi-Fidelity Training'):
            loss, loss_hf, loss_l2 = self.train_step(x_hf, y_hf, optimizer)

            if epoch % 100 == 0:
                tqdm.write(f'Epoch {epoch}, Loss: {loss.numpy()}, loss_hf: {loss_hf.numpy()}, loss_l2: {loss_l2.numpy()}')

    def __call__(self, x_hf):
        pred_lf = self.lf_model(x_hf)
        x_combined = tf.concat([x_hf, pred_lf], axis=-1)
        if self.alpha is None:
            y = self.nonlinear_model(x_combined) + self.linear_model(x_combined)
        else:
            y = self.alpha * self.linear_model(x_combined) + (1.0 - self.alpha) * self.nonlinear_model(x_combined)
        return y
    
    def predict(self, x):
        pred = self(x)
        return pred