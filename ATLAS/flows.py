import jax
import jax.numpy as jnp
import jax.random as jr
import haiku as hk
import optax
import distrax
import numpy as np
from typing import Sequence
from tqdm.auto import trange
from functools import partial


class Flow:
    """
    Rational Quadratic Spline normalizing flow with
    custom power-law preprocessing.

    This class wraps a full training + inference pipeline including:
    - preprocessing
    - model definition
    - optimization
    - sampling and evaluation
    """

    def __init__(
        self,
        data,
        flow_num_layers=4,
        hidden_size=128,
        mlp_num_layers=2,
        num_bins=8,
        learning_rate=1e-4,
        B=6.0,
        p=0.3,
        seed=0,
    ):

        """
        Flow: Rational Quadratic Spline Normalizing Flow
        ================================================

        This module defines the `Flow` class, a full implementation of a
        normalizing flow using JAX, Haiku, Distrax, and Optax.

        The implementation is adapted from:
        https://cocalc.com/github/probml/pyprobml/blob/master/deprecated/notebooks/flow_spline_mnist_jax.ipynb

        Original authors:
        Murphy, K., Soliman, M., Duran-Martin, G., Kara, A., Liang Ang, M., Reddy, S., & Patel, D. (2021)
        PyProbML library for Probabilistic Machine Learning

        -----------------------------------------------------------------------
        OVERVIEW
        -----------------------------------------------------------------------

        This model implements a density estimator using a sequence of invertible
        transformations (a normalizing flow). The transformations map data between:

            physical space (original data)
                ↔
            normalized bounded space
                ↔
            latent Gaussian space

        The pipeline consists of:

        1. Power-law preprocessing
        2. Affine normalization to a bounded domain [-B, B]
        3. A stack of rational quadratic spline coupling layers
        4. A standard Gaussian base distribution

        -----------------------------------------------------------------------
        PREPROCESSING TRANSFORM
        -----------------------------------------------------------------------

        Before training, each input x is transformed as follows:

        Step 1: Power transform
            x' = sign(x) * |x|^gamma

            where gamma = exp(alpha)

        Step 2: Compute statistics from transformed data
            mean = (max(x') + min(x')) / 2
            half_range = (max(x') - min(x')) / 2

        Step 3: Normalize to bounded space
            x_norm = (x' - mean) / half_range * B

        Full forward transform:
            x_norm = (sign(x) * |x|^gamma - mean) / half_range * B

        Inverse transform:
            z = (half_range / B) * x_norm + mean
            x = sign(z) * |z|^(1/gamma)

        Jacobian corrections are applied during likelihood evaluation.

        -----------------------------------------------------------------------
        FEATURES
        -----------------------------------------------------------------------

        - Exact log-likelihood computation
        - Invertible transformations
        - Efficient sampling
        - JIT-compiled operations (via JAX)
        - Automatic differentiation

        -----------------------------------------------------------------------

        Initialize the flow model.

        Parameters
        ----------
        data : array (N, D)
            Training dataset
        flow_num_layers : int
            Number of coupling layers
        hidden_size : int
            Width of MLP hidden layers
        mlp_num_layers : int
            Number of hidden layers in conditioner networks
        num_bins : int
            Number of bins in spline transform
        learning_rate : float
            Adam optimizer learning rate
        B : float
            Bound for normalized space
        p : float
            Power transform parameter
        seed : int
            Random seed
        """

        self.N, self.D = data.shape
        self.B = float(B) - 1 #avoiding boundry problems

        # ------------------------------------------------------------
        # Power transform parameters
        # ------------------------------------------------------------

        self.alpha = jnp.log(p)
        dummy = jnp.sign(data) * jnp.abs(data) ** (jnp.exp(self.alpha))
        min_x = dummy.min(axis =0)
        max_x = dummy.max(axis = 0)
        self.mean_x = (max_x + min_x) / 2
        self.half_range = (max_x - min_x) / 2

        # ------------------------------------------------------------
        # Normalize data
        # ------------------------------------------------------------

        self.x_norm = self.to_unit_interval(data)
        assert self.x_norm.min() > -B
        assert self.x_norm.max() < B
        # ------------------------------------------------------------
        # Flow config
        # ------------------------------------------------------------

        self.event_shape = (self.D,)
        self.flow_num_layers = flow_num_layers
        self.hidden_sizes = [hidden_size] * mlp_num_layers
        self.num_bins = num_bins
        self.learning_rate = learning_rate

        # ------------------------------------------------------------
        # RNG
        # ------------------------------------------------------------

        self.key = jr.PRNGKey(seed)

        # ------------------------------------------------------------
        # Build/init
        # ------------------------------------------------------------

        self._build_model()
        self._init_params()

    # ================================================================
    # Utilities
    # ================================================================

    # ================================================================
    # Data transforms
    # ================================================================
    @partial(jax.jit, static_argnums=0)
    def from_unit_interval(self, arr):
        """
        Transform normalized values back to physical space.

        Implements the inverse of the preprocessing transform.
        """
        dummy = (self.half_range[None, :] / self.B * arr + self.mean_x[None, :])
        return jnp.sign(dummy) * jnp.abs(dummy) ** (1.0 / jnp.exp(self.alpha))

    @partial(jax.jit, static_argnums=0)
    def to_unit_interval(self, arr):
        """
        Transform physical data into normalized bounded space [-B, B].
        """
        dummy = jnp.sign(arr) * jnp.abs(arr) ** (jnp.exp(self.alpha))
        return (dummy - self.mean_x[None, :]) / self.half_range[None, :] * self.B


    # ================================================================
    # Jacobians
    # ================================================================


    # ================================================================
    # Jacobian: physical → normalized
    # ================================================================
    @partial(jax.jit, static_argnums=0)
    def logdet_to_unit_interval(self, arr):
        """
        Log absolute determinant of the Jacobian of the transformation
        from physical space to normalized space.
        """

        gamma = jnp.exp(self.alpha)

        logdet = jnp.sum(
            jnp.log(gamma)
            + (gamma - 1.0) * jnp.log(jnp.abs(arr))
            + jnp.log(self.B / self.half_range),
            axis=-1,
        )

        return logdet


    # ================================================================
    # Jacobian: normalized → physical
    # ================================================================
    @partial(jax.jit, static_argnums=0)
    def logdet_from_unit_interval(self, arr):
        """
        Log absolute determinant of the Jacobian of the transformation
        from normalized space back to physical space.
        """

        beta = 1.0 / jnp.exp(self.alpha)

        dummy = (
            self.half_range[None, :] / self.B * arr
            + self.mean_x[None, :]
        )

        logdet = jnp.sum(
            jnp.log(self.half_range / self.B)
            + jnp.log(beta)
            + (beta - 1.0) * jnp.log(jnp.abs(dummy)),
            axis=-1,
        )

        return logdet


    # ================================================================
    # Batching
    # ================================================================

    @staticmethod
    def get_batches(data, batch_size, key):
        """
        Generate shuffled mini-batches of data.
        """

        N = data.shape[0]

        idx = jax.random.permutation(key, N)

        for i in range(0, N, batch_size):
            yield data[idx[i:i + batch_size]]

    # ================================================================
    # Conditioner
    # ================================================================

    @staticmethod
    def make_conditioner(
        event_shape,
        hidden_sizes,
        num_bijector_params,
    ):
        """
        Construct the neural network used to parameterize spline bijectors.
        """

        return hk.Sequential([
            hk.Flatten(
                preserve_dims=-len(event_shape)
            ),

            hk.nets.MLP(
                hidden_sizes,
                activate_final=True,
            ),

            hk.Linear(
                np.prod(event_shape)
                * num_bijector_params,

                w_init=jnp.zeros,
                b_init=jnp.zeros,
            ),

            hk.Reshape(
                event_shape
                + (num_bijector_params,),
                preserve_dims=-1,
            ),
        ])

    # ================================================================
    # Flow definition
    # ================================================================

    @staticmethod
    def make_flow_model(
        event_shape,
        num_layers,
        hidden_sizes,
        num_bins,
        B,
    ):
        """
        Build the complete normalizing flow model.

        The flow consists of multiple masked coupling layers using
        rational quadratic spline transformations.
        """

        mask = (
            jnp.arange(np.prod(event_shape))
            % 2
        )

        mask = mask.reshape(event_shape).astype(bool)

        def bijector_fn(params):
            return distrax.RationalQuadraticSpline(
                params,
                range_min=-B - 1,
                range_max=B + 1,
            )

        num_bijector_params = 3 * num_bins + 1

        layers = []

        for _ in range(num_layers):

            layers.append(
                distrax.MaskedCoupling(
                    mask=mask,

                    bijector=bijector_fn,

                    conditioner=Flow.make_conditioner(
                        event_shape,
                        hidden_sizes,
                        num_bijector_params,
                    ),
                )
            )

            mask = jnp.logical_not(mask)

        flow = distrax.Inverse(
            distrax.Chain(layers)
        )

        base_dist = distrax.Independent(
            distrax.Normal(
                loc=jnp.zeros(event_shape),
                scale=jnp.ones(event_shape),
            ),
            reinterpreted_batch_ndims=len(event_shape),
        )

        return distrax.Transformed(
            base_dist,
            flow,
        )

    # ================================================================
    # Haiku transforms
    # ================================================================

    def _build_model(self):
        """
        Define Haiku-transformed functions for log-probability,
        sampling, and forward transformations.
        """

        @hk.without_apply_rng
        @hk.transform
        def log_prob_fn(data):

            model = Flow.make_flow_model(
                self.event_shape,
                self.flow_num_layers,
                self.hidden_sizes,
                self.num_bins,
                self.B,
            )

            return model.log_prob(data)

        @hk.without_apply_rng
        @hk.transform
        def sample_fn(key, num_samples):

            model = Flow.make_flow_model(
                self.event_shape,
                self.flow_num_layers,
                self.hidden_sizes,
                self.num_bins,
                self.B,
            )

            return model.sample(
                seed=key,
                sample_shape=[num_samples],
            )

        @hk.without_apply_rng
        @hk.transform
        def transform_fn(z):

            model = Flow.make_flow_model(
                self.event_shape,
                self.flow_num_layers,
                self.hidden_sizes,
                self.num_bins,
                self.B,
            )

            return model.bijector.forward(z)

        @hk.without_apply_rng
        @hk.transform
        def transform_and_jac_fn(z):

            model = Flow.make_flow_model(
                self.event_shape,
                self.flow_num_layers,
                self.hidden_sizes,
                self.num_bins,
                self.B,
            )

            return model.bijector.forward_and_log_det(z)

        self.log_prob_fn = log_prob_fn
        self.sample_fn = sample_fn
        self.transform_fn = transform_fn
        self.transform_and_jac_fn = transform_and_jac_fn

    # ================================================================
    # Initialization
    # ================================================================

    def _init_params(self):
        """
        Initialize model parameters and optimizer state.
        """

        self.key, init_key = jr.split(self.key)

        self.params = self.log_prob_fn.init(
            init_key,
            self.x_norm[:1],
        )

        self.optimizer = optax.adam(
            self.learning_rate
        )

        self.opt_state = self.optimizer.init(
            self.params
        )

    # ================================================================
    # Training
    # ================================================================

    def loss_fn(self, params, batch):
        """
        Negative log-likelihood loss function.
        The same thing as KL-divergence!
        """

        logp = self.log_prob_fn.apply(
            params,
            batch,
        )

        return -jnp.mean(logp)

    @partial(jax.jit, static_argnums=0)
    def update(
        self,
        params,
        opt_state,
        batch,
    ):

        """
        Perform one gradient update step using Adam.
        """

        grads = jax.grad(
            self.loss_fn
        )(params, batch)

        updates, opt_state = self.optimizer.update(
            grads,
            opt_state,
        )

        params = optax.apply_updates(
            params,
            updates,
        )

        return params, opt_state

    def fit(
        self,
        num_epochs=100,
        batch_size=256,
    ):
        """
        Train the model using mini-batch gradient descent.
        """

        for _ in trange(num_epochs):

            self.key, subkey = jr.split(self.key)

            for batch in self.get_batches(
                self.x_norm,
                batch_size,
                subkey,
            ):

                self.params, self.opt_state = (
                    self.update(
                        self.params,
                        self.opt_state,
                        batch,
                    )
                )

    # ================================================================
    # Public API
    # ================================================================

    @partial(jax.jit, static_argnums=0)
    def log_prob(self, x):
        """
        Compute log-probability in physical space.
        """

        x_norm = self.to_unit_interval(x)

        logp_norm = self.log_prob_fn.apply(
            self.params,
            x_norm,
        )

        logdet = self.logdet_to_unit_interval(x)

        return logp_norm + logdet

    def sample(self, num_samples):
        """
        Draw samples in physical space.
        """

        self.key, subkey = jr.split(self.key)

        samples_norm = self.sample_fn.apply(
            self.params,
            subkey,
            num_samples,
        )

        return self.from_unit_interval(
            samples_norm
        )

    @partial(jax.jit, static_argnums=0)
    def forward_pass(self, z):
        """
        Map latent samples to physical space.
        """

        physics_norm = self.transform_fn.apply(
            self.params,
            z,
        )

        return self.from_unit_interval(
            physics_norm
        )

    @partial(jax.jit, static_argnums=0)
    def forward_pass_with_logdet(self, z):
        """
        Forward transform with full Jacobian correction.
        """

        physics_norm, logdet_flow = self.transform_and_jac_fn.apply(
            self.params,
            z,
        )

        physics = self.from_unit_interval(physics_norm)

        logdet_phys = self.logdet_from_unit_interval(physics_norm)

        total_logdet = logdet_flow + logdet_phys

        return physics, total_logdet