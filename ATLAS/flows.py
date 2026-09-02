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
import random
jax.config.update("jax_enable_x64", True)
import pickle
import os

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers (pure numpy, used at init and in public API)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_normalizer(min_x, max_x, p, B):
    """
    Fit the power-affine normalizer from user-supplied range statistics.

    The forward direction (physical → normalized) is:
        dummy = sign(x) * |x|^exp(alpha)          # power stretch
        norm  = (dummy - mean) / half_range * B   # affine to [-B, B]

    The inverse (normalized → physical) is:
        dummy = half_range/B * norm + mean
        x     = sign(dummy) * |dummy|^(1/exp(alpha))

    Parameters
    ----------
    min_x, max_x : 1-D array-like, shape (D,)
        Per-feature physical-space range boundaries.
    p : float
        Power parameter.  alpha = log(p).
    B : float
        Target half-width of the normalized space.

    Returns
    -------
    dict with keys: alpha, mean, half_range, B
    """
    min_x = np.asarray(min_x, dtype=np.float64)
    max_x = np.asarray(max_x, dtype=np.float64)
    alpha = np.log(float(p))
    gamma = np.exp(alpha)
    dummy_min = np.sign(min_x) * np.abs(min_x) ** gamma
    dummy_max = np.sign(max_x) * np.abs(max_x) ** gamma
    mean      = (dummy_max + dummy_min) / 2.0
    half_range = (dummy_max - dummy_min) / 2.0
    return dict(alpha=alpha, mean=mean, half_range=half_range, B=float(B))


def _from_coefficients_np(arr, norm):
    """Physical → normalized  (numpy, used during batched training reads)."""
    alpha, mean, half_range, B = norm['alpha'], norm['mean'], norm['half_range'], norm['B']
    dummy = np.sign(arr) * np.abs(arr) ** np.exp(alpha)
    return (dummy - mean) / half_range * B


def _to_coefficients_np(arr, norm):
    """Normalized → physical  (numpy, used at inference)."""
    alpha, mean, half_range, B = norm['alpha'], norm['mean'], norm['half_range'], norm['B']
    dummy = half_range / B * arr + mean
    return np.sign(dummy) * np.abs(dummy) ** (1.0 / np.exp(alpha))


def _from_coefficients_jnp(arr, alpha, mean, half_range, B):
    """Physical → normalized  (jax, used inside jit-compiled methods)."""
    dummy = jnp.sign(arr) * jnp.abs(arr) ** jnp.exp(alpha)
    return (dummy - mean) / half_range * B


def _to_coefficients_jnp(arr, alpha, mean, half_range, B):
    """Normalized → physical  (jax, used inside jit-compiled methods)."""
    dummy = half_range / B * arr + mean
    return jnp.sign(dummy) * jnp.abs(dummy) ** (1.0 / jnp.exp(alpha))


def _logdet_from_coefficients_jnp(arr, alpha, mean, half_range, B):
    """
    Log |det J| of the physical → normalized map.
    J is diagonal so log|det J| = sum_i log|df_i/dx_i|.

    df/dx = exp(alpha) * |x|^(exp(alpha)-1) * (B / half_range)
    """
    gamma = jnp.exp(alpha)
    return jnp.sum(
        jnp.log(gamma)
        + (gamma - 1.0) * jnp.log(jnp.abs(arr))
        + jnp.log(B / half_range),
        axis=-1,
    )


def _logdet_to_coefficients_jnp(arr, alpha, mean, half_range, B):
    """
    Log |det J| of the normalized → physical map.
    """
    beta = 1.0 / jnp.exp(alpha)
    dummy = half_range / B * arr + mean
    return jnp.sum(
        jnp.log(half_range / B)
        + jnp.log(beta)
        + (beta - 1.0) * jnp.log(jnp.abs(dummy)),
        axis=-1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Memmapped-safe batch iterator
# ─────────────────────────────────────────────────────────────────────────────

def _iter_batches(data, context, x_norm, c_norm, batch_size, key):
    """
    Yield (x_batch, c_batch) pairs in random order without loading the
    full dataset into memory.

    Parameters
    ----------
    data, context : array-like with shape[0] == N
        Raw memmapped (or in-memory) arrays.  Only small index-selected
        slices are materialised at a time.
    x_norm, c_norm : dict
        Normalizer dicts returned by _fit_normalizer for data and context.
    batch_size : int
    key : jax PRNGKey

    Notes
    -----
    *Shuffling* is done by drawing a random permutation of row indices
    (cheap — just integers).  Each batch then reads only `batch_size`
    rows from the mmap file, normalises them on the fly, and converts
    to jax arrays.  Peak memory is O(batch_size * (D + C)), regardless
    of N.
    """
    N = data.shape[0]
    perm = np.array(jr.permutation(key, N))          # numpy for fancy indexing
    for start in range(0, N, batch_size):
        idx = perm[start: start + batch_size]
        if len(idx) < batch_size:
            continue  
        # Fancy-index into mmap → materialises only this slice
        x_raw = np.asarray(data[idx], dtype=np.float64)
        c_raw = np.asarray(context[idx], dtype=np.float64)
        x_batch = jnp.asarray(_from_coefficients_np(x_raw, x_norm))
        c_batch = jnp.asarray(_from_coefficients_np(c_raw, c_norm))
        yield x_batch, c_batch

def _iter_batches_no_context(data, x_norm, batch_size, key):
    """
    Yield (x_batch, c_batch) pairs in random order without loading the
    full dataset into memory.

    Parameters
    ----------
    data, context : array-like with shape[0] == N
        Raw memmapped (or in-memory) arrays.  Only small index-selected
        slices are materialised at a time.
    x_norm, c_norm : dict
        Normalizer dicts returned by _fit_normalizer for data and context.
    batch_size : int
    key : jax PRNGKey

    Notes
    -----
    *Shuffling* is done by drawing a random permutation of row indices
    (cheap — just integers).  Each batch then reads only `batch_size`
    rows from the mmap file, normalises them on the fly, and converts
    to jax arrays.  Peak memory is O(batch_size * (D + C)), regardless
    of N.
    """
    N = data.shape[0]
    perm = np.array(jr.permutation(key, N))          # numpy for fancy indexing
    for start in range(0, N, batch_size):
        idx = perm[start: start + batch_size]
        if len(idx) < batch_size:
            continue  
        # Fancy-index into mmap → materialises only this slice
        x_raw = np.asarray(data[idx], dtype=np.float64)
        x_batch = jnp.asarray(_from_coefficients_np(x_raw, x_norm))
        return x_batch



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
        data_min = jnp.array([False]),
        data_max = jnp.array([False]),
        tset_paths = jnp.array([False]),
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
        self.tset_paths = tset_paths
        
        self.N, self.D = data.shape
        self.B = float(B) #avoiding boundry problems
        self.alpha = jnp.log(p)
        self.gamma = jnp.exp(self.alpha)

        # ------------------------------------------------------------
        # Power transform parameters
        # ------------------------------------------------------------
        if data_min.any() and data_max.any():
            min_x = data_min
            max_x = data_max
            dummy_min = np.sign(min_x) * np.abs(min_x) ** self.gamma
            dummy_max = np.sign(max_x) * np.abs(max_x) ** self.gamma
            self.mean_x      = (dummy_max + dummy_min) / 2.0
            self.half_range  = (dummy_max - dummy_min) / 2.0
        else:
            dummy = jnp.sign(data) * jnp.abs(data) ** self.gamma
            min_x = dummy.min(axis =0)
            max_x = dummy.max(axis = 0)
            self.mean_x = (max_x + min_x) / 2.0
            self.half_range = (max_x - min_x) / 2.0

        # ------------------------------------------------------------
        # Normalize data
        # ------------------------------------------------------------

        self.x_norm = self.to_unit_interval(data)
        # assert self.x_norm.min() > -B
        # assert self.x_norm.max() < B
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
        def inverse_transform_fn(z_inv):

            model = Flow.make_flow_model(
                self.event_shape,
                self.flow_num_layers,
                self.hidden_sizes,
                self.num_bins,
                self.B,
            )

            return model.bijector.inverse(z_inv)
            
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
        self.inv_transform_fn = inverse_transform_fn
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
        mode = 'pre-loaded',
        num_epochs=100,
        batch_size=256,
        checkpoint=None,
    ):
        """
        Train the model using mini-batch gradient descent.

        mode : str
        num_epochs : int
        batch_size : int
        checkpoint : path or None
            Write training state here after every epoch if not None,
            and resume from last epoch recorded here if it already exists. 
        """


        if mode == 'pre-loaded':
            # load from last checkpointed epoch
            start_epoch = 0           
            if checkpoint is not None and os.path.exists(checkpoint):
                start_epoch = self._load_checkpoint(checkpoint, batch_size)
                print(f"[flow] resuming from {checkpoint} at epoch "
                        f"{start_epoch}/{num_epochs}", flush=True)
            self.resumed_from = start_epoch # needed for interpreting wall-clock timing on a resumed run
                 
            for ep in trange(start_epoch, num_epochs, 
                            initial=start_epoch, total=num_epochs,
                            colour = 'blue', desc = "Training the flow from pre-loaded data"):

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

                if checkpoint is not None:
                    self._save_checkpoint(checkpoint, ep + 1, batch_size)

        if mode == 'from_disk':
            if checkpoint is not None:
                err = "Checkpointing is only supported for mode='pre-loaded'."
                raise ValueError(err)
            for _ in trange(num_epochs, colour = 'blue', desc = "Training the flow from disk"):
                rand_idx = random.randint(0, len(self.tset_paths) - 1)
                training_set = jnp.load(self.tset_paths[rand_idx])
                idxs = jnp.array(random.sample(range(training_set.shape[0]), 
                                k = min(batch_size, training_set.shape[0])))
                batch =  self.to_unit_interval(training_set[idxs])
                self.params, self.opt_state = self.update(self.params, 
                                                            self.opt_state, 
                                                            batch)

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
    def backward_pass(self, target_samp):
        """
        Map latent samples to physical space.
        """
        x = self.to_unit_interval(target_samp)
        
        physics_norm = self.inv_transform_fn.apply(
            self.params,
            x,
        )

        return physics_norm

    
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

    def save_params(self, flow_save_path):
        leaves, treedef = jax.tree_util.tree_flatten(self.params)
        save_dict = {f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)}
        save_dict["treedef"] = np.array(str(treedef))
        np.savez(flow_save_path, **save_dict)
    
    def load_params(self, flow_load_path):
        data = np.load(flow_load_path, allow_pickle=True)
        _, treedef = jax.tree_util.tree_flatten(self.params)
        leaves = [jnp.asarray(data[f"leaf_{i}"]) for i in range(treedef.num_leaves)]
        self.params = jax.tree_util.tree_unflatten(treedef, leaves)


    # ------------- Checkpointing -----------
    # To resume flow training if interrupted partway through

    def _checkpoint_signature(self, batch_size):
        # Configuration must match for a checkpoint to belong to this run.
        # Prevents resuming into a different state
        return dict(
            N=int(self.N), D=int(self.D),
            flow_num_layers=int(self.flow_num_layers),
            hidden_sizes=[int(h) for h in self.hidden_sizes],
            num_bins=int(self.num_bins),
            learning_rate=float(self.learning_rate),
            B=float(self.B), gamma=float(self.gamma),
            batch_size=int(batch_size),
        )

    def _save_checkpoint(self, path, epoch, batch_size):
        """Params, optimizer state, RNG key and epoch counter."""
        # `opt_state` is saved, not just `params`.  `save_params` is deliberately
        # not reused here: it writes the parameter leaves through `np.savez` plus
        # a stringified treedef, which cannot carry an optax NamedTuple tree or a
        # PRNG key.  Resuming without Adam's first and second moments would
        # restart them at zero and put a visible transient in the loss.
        blob = dict(
            params=jax.tree_util.tree_map(np.asarray, self.params),
            opt_state=jax.tree_util.tree_map(np.asarray, self.opt_state),
            key=np.asarray(self.key),
            epoch=int(epoch),
            signature=self._checkpoint_signature(batch_size),
        )
        # Write to a temporary file and rename.  The atomic rename is the whole
        # reason this is safe to interrupt: a kill during the write leaves the
        # PREVIOUS checkpoint intact rather than a half-written file that cannot
        # be unpickled -- which would turn one lost epoch into all of them.
        tmp = f'{path}.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)       

    def _load_checkpoint(self, path, batch_size):
        """Restore training state; return the number of epochs already done."""
        with open(path, 'rb') as f:
            blob = pickle.load(f)

        want, got = self._checkpoint_signature(batch_size), blob['signature']
        if got != want:
            differ = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
            raise ValueError(
                f"{path} was written by a different configuration: "
                f"(checkpoint, this run) differ on {differ}.  Train into a "
                f"different directory, or delete the checkpoint to start over.")

        self.params = jax.tree_util.tree_map(jnp.asarray, blob['params'])
        self.opt_state = jax.tree_util.tree_map(jnp.asarray, blob['opt_state'])
        # Restoring the key is what makes a resume CONTINUE the RNG stream.
        # Without it the resumed epochs would replay the batch order of the
        # epochs already trained on.
        self.key = jnp.asarray(blob['key'])
        return int(blob['epoch'])

    

class ConditionalFlow:
    """
    Conditional Rational Quadratic Spline normalizing flow.

    Models p(x | c) where c is a conditioning context vector.

    Both data and context are independently normalised through a
    power-affine map:

        physical → normalized:
            dummy = sign(x) * |x|^exp(alpha)
            norm  = (dummy - mean) / half_range * B

        normalized → physical:
            dummy = half_range/B * norm + mean
            x     = sign(dummy) * |dummy|^(1/exp(alpha))

    The user supplies the physical-space range [min_x, max_x] for both
    arrays, plus p and B, so no data statistics need to be computed from
    the full dataset — making the class fully compatible with memmapped
    inputs.

    Distrax does NOT natively support conditional flows.
    Conditioning is implemented by closing over `c` inside each
    conditioner MLP: the MLP receives [x_masked ‖ c_norm] as input.
    """

    def __init__(
        self,
        # ── data / context (memmapped or in-memory) ──────────────────────
        data,
        context,
        # ── normalisation ranges (per-feature 1-D arrays) ────────────────
        data_min = None,
        data_max = None,
        context_min = None,
        context_max = None,
        normalizer_dict = jnp.array([False]),
        tset_paths = None,
        last_feature_index = None,
        last_feature_index_for_context = None,
        flow_save_path = None,
        flow_load_path = None,
        # ── shared normalisation hyperparameters ─────────────────────────
        p=0.1,
        B=4.0,
        # ── flow architecture ─────────────────────────────────────────────
        flow_num_layers=4,
        hidden_size=128,
        mlp_num_layers=2,
        num_bins=8,
        # ── optimisation ─────────────────────────────────────────────────
        learning_rate=1e-4,
        seed=0,
    ):
        """
        Parameters
        ----------
        data : array-like (N, D)
            Target variables.  May be a numpy memmap — only batch-sized
            slices are ever read.
        context : array-like (N, C)
            Conditioning variables.  Same memory constraints as data.
        data_min, data_max : array-like (D,)
            Per-feature physical-space bounds of `data`.
        context_min, context_max : array-like (C,)
            Per-feature physical-space bounds of `context`.
        p : float
            Power parameter shared by both normalizers.  alpha = log(p).
        B : float
            Target half-width of the normalized space for both arrays.
        flow_num_layers : int
            Number of RQS masked-coupling layers.
        hidden_size : int
            Width of each conditioner MLP hidden layer.
        mlp_num_layers : int
            Number of hidden layers in each conditioner MLP.
        num_bins : int
            Number of spline bins per RQS layer.
        learning_rate : float
            Adam learning rate.
        seed : int
            PRNG seed.
        """
        self.N, self.D = data.shape[0], data.shape[1]
        self.C = context.shape[1]
        self.tset_paths = tset_paths
        self.last_feature_idx = last_feature_index
        self.last_feature_index_for_context = last_feature_index_for_context

        self.flow_save_path = flow_save_path
        self.flow_load_path = flow_load_path
        
        # Keep references to the raw arrays (may be memmapped)
        self._data    = data
        self._context = context

        # ── Normalizers ───────────────────────────────────────────────────
        # Fit from user-supplied ranges; never touches the full data array.
        if any(normalizer_dict):
            self.x_norm = normalizer_dict['data']
            self.c_norm = normalizer_dict['context']
        else:
            self.x_norm = _fit_normalizer(data_min,    data_max,    p, B)
            self.c_norm = _fit_normalizer(context_min, context_max, p, B)

        # Jax-side constants for jit-compiled methods
        self._x_alpha      = jnp.float64(self.x_norm['alpha'])
        self._x_mean       = jnp.asarray(self.x_norm['mean'])
        self._x_half_range = jnp.asarray(self.x_norm['half_range'])
        self._x_B          = jnp.float64(self.x_norm['B'])

        self._c_alpha      = jnp.float64(self.c_norm['alpha'])
        self._c_mean       = jnp.asarray(self.c_norm['mean'])
        self._c_half_range = jnp.asarray(self.c_norm['half_range'])
        self._c_B          = jnp.float64(self.c_norm['B'])

        # ── Flow config ───────────────────────────────────────────────────
        self.event_shape    = (self.D,)
        self.flow_num_layers = flow_num_layers
        self.hidden_sizes    = [hidden_size] * mlp_num_layers
        self.num_bins        = num_bins
        self.learning_rate   = learning_rate

        self.key = jr.PRNGKey(seed)

        self._build_model()
        self._init_params()

    # ====================================================================
    # Normalisation (jax, used in public API and loss)
    # ====================================================================

    @partial(jax.jit, static_argnums=0)
    def normalize_data(self, x):
        """Physical data → normalized [-B, B]."""
        return _from_coefficients_jnp(
            x, self._x_alpha, self._x_mean, self._x_half_range, self._x_B
        )

    @partial(jax.jit, static_argnums=0)
    def denormalize_data(self, x_norm):
        """Normalized → physical data."""
        return _to_coefficients_jnp(
            x_norm, self._x_alpha, self._x_mean, self._x_half_range, self._x_B
        )

    @partial(jax.jit, static_argnums=0)
    def normalize_context(self, c):
        """Physical context → normalized [-B, B]."""
        return _from_coefficients_jnp(
            c, self._c_alpha, self._c_mean, self._c_half_range, self._c_B
        )

    @partial(jax.jit, static_argnums=0)
    def denormalize_context(self, c_norm):
        """Normalized → physical context."""
        return _to_coefficients_jnp(
            c_norm, self._c_alpha, self._c_mean, self._c_half_range, self._c_B
        )

    @partial(jax.jit, static_argnums=0)
    def _logdet_normalize_data(self, x):
        return _logdet_from_coefficients_jnp(
            x, self._x_alpha, self._x_mean, self._x_half_range, self._x_B
        )

    @partial(jax.jit, static_argnums=0)
    def _logdet_denormalize_data(self, x_norm):
        return _logdet_to_coefficients_jnp(
            x_norm, self._x_alpha, self._x_mean, self._x_half_range, self._x_B
        )

    # ====================================================================
    # Flow internals
    # ====================================================================

    @staticmethod
    def make_conditioner(event_shape, hidden_sizes, num_bijector_params, context_size):
        def factory(c):
            net = hk.Sequential([
                hk.Flatten(preserve_dims=-len(event_shape)),
                hk.nets.MLP(hidden_sizes, activate_final=True),
                hk.Linear(
                    np.prod(event_shape) * num_bijector_params,
                    w_init=jnp.zeros,
                    b_init=jnp.zeros,
                ),
                hk.Reshape(
                    event_shape + (num_bijector_params,),
                    preserve_dims=-1,
                ),
            ])

            def conditioner(x_masked):
                if x_masked.ndim == 1:
                    inp = jnp.concatenate([x_masked, c[0]], axis=-1)
                else:
                    inp = jnp.concatenate([x_masked, c], axis=-1)
                return net(inp)

            return conditioner
        return factory

    @staticmethod
    def make_flow_model(event_shape, num_layers, hidden_sizes, num_bins, B, c):
        mask = (jnp.arange(np.prod(event_shape)) % 2).reshape(event_shape).astype(bool)
        context_size       = c.shape[-1]
        num_bijector_params = 3 * num_bins + 1

        def bijector_fn(params):
            return distrax.RationalQuadraticSpline(
                params, range_min=-B - 1, range_max=B + 1
            )

        layers = []
        for _ in range(num_layers):
            conditioner = ConditionalFlow.make_conditioner(
                event_shape, hidden_sizes, num_bijector_params, context_size
            )(c)
            layers.append(
                distrax.MaskedCoupling(
                    mask=mask, bijector=bijector_fn, conditioner=conditioner
                )
            )
            mask = jnp.logical_not(mask)

        flow = distrax.Inverse(distrax.Chain(layers))
        base_dist = distrax.Independent(
            distrax.Normal(jnp.zeros(event_shape), jnp.ones(event_shape)),
            reinterpreted_batch_ndims=len(event_shape),
        )
        return distrax.Transformed(base_dist, flow)

    # ====================================================================
    # Haiku transforms
    # ====================================================================

    def _build_model(self):
        # Must be a plain Python float — RationalQuadraticSpline does
        # `range_min >= range_max` at init time, which cannot be a traced value.
        B = float(self.x_norm['B'])

        @hk.without_apply_rng
        @hk.transform
        def log_prob_fn(x_norm, c_norm):
            model = ConditionalFlow.make_flow_model(
                self.event_shape, self.flow_num_layers,
                self.hidden_sizes, self.num_bins, B, c_norm,
            )
            return model.log_prob(x_norm)

        @hk.without_apply_rng
        @hk.transform
        def sample_fn(key, c_norm, num_samples):
            model = ConditionalFlow.make_flow_model(
                self.event_shape, self.flow_num_layers,
                self.hidden_sizes, self.num_bins, B, c_norm,
            )
            return model.sample(seed=key, sample_shape=[num_samples])

        @hk.without_apply_rng
        @hk.transform
        def transform_fn(z, c_norm):
            model = ConditionalFlow.make_flow_model(
                self.event_shape, self.flow_num_layers,
                self.hidden_sizes, self.num_bins, B, c_norm,
            )
            return model.bijector.forward(z)

        @hk.without_apply_rng
        @hk.transform
        def transform_and_jac_fn(z, c_norm):
            model = ConditionalFlow.make_flow_model(
                self.event_shape, self.flow_num_layers,
                self.hidden_sizes, self.num_bins, B, c_norm,
            )
            return model.bijector.forward_and_log_det(z)

        self.log_prob_fn        = log_prob_fn
        self.sample_fn          = sample_fn
        self.transform_fn       = transform_fn
        self.transform_and_jac_fn = transform_and_jac_fn

    # ====================================================================
    # Initialisation
    # ====================================================================

    def _init_params(self):
        self.key, init_key = jr.split(self.key)

        # Read two rows from the raw arrays (minimal mmap touch)
        x0 = jnp.asarray(
            _from_coefficients_np(
                np.asarray(self._data[:2],    dtype=np.float64), self.x_norm
            )
        )
        c0 = jnp.asarray(
            _from_coefficients_np(
                np.asarray(self._context[:2], dtype=np.float64), self.c_norm
            )
        )

        self.params    = self.log_prob_fn.init(init_key, x0, c0)
        self.optimizer = optax.adam(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    # ====================================================================
    # Training
    # ====================================================================

    def loss_fn(self, params, x_batch, c_batch):
        """Negative mean log-likelihood in normalized space."""
        return -jnp.mean(self.log_prob_fn.apply(params, x_batch, c_batch))

    @partial(jax.jit, static_argnums=0, donate_argnums=(1, 2))
    def _update(self, params, opt_state, x_batch, c_batch):
        grads = jax.grad(self.loss_fn)(params, x_batch, c_batch)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def fit(self, mode = 'pre-loaded', num_epochs=100, batch_size=256, context_array = None):
        """
        Train the flow.

        Batches are read lazily from `self._data` and `self._context`,
        normalised on the fly, and immediately discarded.  Only
        `batch_size` rows are ever resident in memory at once, so this
        is safe with arbitrarily large memmapped arrays.
        """
        if mode == 'pre-loaded':
            for _ in trange(num_epochs, colour = 'green', desc = "Training the flow the pre-loaded way"):
                self.key, subkey = jr.split(self.key)
                for x_batch, c_batch in _iter_batches(
                    self._data, self._context,
                    self.x_norm, self.c_norm,
                    batch_size, subkey,
                ):
                    self.params, self.opt_state = self._update(
                        self.params, self.opt_state, x_batch, c_batch
                    )
        elif mode == 'simple':
            for _ in trange(num_epochs, colour = 'blue', desc = "Training the flow the simple way"):
                rand_idx = random.randint(0, len(self.tset_paths) - 1)
                training_set = jnp.load(self.tset_paths[rand_idx])
                idxs = jnp.array(random.sample(range(training_set.shape[0]), k = min(batch_size, training_set.shape[0])))
                training_set = training_set[idxs]
                x_batch =  self.normalize_data(training_set[:, :self.last_feature_idx])
                c_batch =  self.normalize_context(training_set[:, self.last_feature_idx:self.last_feature_index_for_context])
                self.params, self.opt_state = self._update(self.params, 
                                                           self.opt_state, 
                                                           x_batch, c_batch)
        
        elif mode == 'broadcastable_context':
            for _ in trange(num_epochs, colour = 'blue', desc = "Training the flow the broadcastable_context way"):
                rand_idx = random.randint(0, len(context_array) - 1)
                training_set = jnp.load(self.tset_paths[rand_idx])
                idxs = jnp.array(random.sample(range(training_set.shape[0]), k = min(batch_size, training_set.shape[0])))
                training_set = training_set[idxs]
                x_batch =  self.normalize_data(training_set[:, :self.last_feature_idx])
                c_batch =  self.normalize_context(context_array[rand_idx:rand_idx+1])
                c_batch = jnp.broadcast_to(c_batch, (x_batch.shape[0], c_batch.shape[-1]))
                self.params, self.opt_state = self._update(self.params, 
                                                           self.opt_state, 
                                                           x_batch, c_batch)            
    
    def live_fit(self, x_batch, c_batch):
        self.params, self.opt_state = self._update(self.params, 
                                                    self.opt_state, 
                                                    x_batch, c_batch)

    # ====================================================================
    # Public API  (inputs/outputs always in physical space)
    # ====================================================================

    @partial(jax.jit, static_argnums=0)
    def log_prob(self, x, c):
        """
        log p(x | c) in physical space.

        Parameters
        ----------
        x : (batch, D)  — physical-space data
        c : (batch, C)  — physical-space context
        """
        x_n = self.normalize_data(x)
        c_n = self.normalize_context(c)
        logp_norm = self.log_prob_fn.apply(self.params, x_n, c_n)
        logdet    = self._logdet_normalize_data(x)
        return logp_norm + logdet

    def sample(self, c, num_samples):
        """
        Draw samples from p(x | c).

        Parameters
        ----------
        c : (num_samples, C) or (1, C) or (C,) — physical-space context.
            A single context row is broadcast to all samples.
        num_samples : int

        Returns
        -------
        samples : (num_samples, D)  in physical space.
        """
        # c = jnp.atleast_2d(jnp.asarray(c, dtype=jnp.float64))
        if c.shape[0] == 1:
            c = jnp.tile(c, (num_samples, 1))

        c_n = self.normalize_context(c)

        self.key, subkey = jr.split(self.key)
        samples_norm = self.sample_fn.apply(self.params, subkey, c_n, num_samples)
        return self.denormalize_data(samples_norm)

    @partial(jax.jit, static_argnums=0)
    def forward_pass(self, z, c):
        """
        Map latent z ~ N(0, I) to physical space given context c.

        Parameters
        ----------
        z : (batch, D)
        c : (batch, C)  — physical-space context
        """
        c_n = self.normalize_context(c)
        x_norm = self.transform_fn.apply(self.params, z, c_n)
        return self.denormalize_data(x_norm)

    @partial(jax.jit, static_argnums=0)
    def forward_pass_with_logdet(self, z, c):
        """
        Forward transform with full Jacobian log-determinant.

        Parameters
        ----------
        z : (batch, D)
        c : (batch, C)  — physical-space context

        Returns
        -------
        x           : (batch, D)   physical-space samples
        total_logdet: (batch,)     log |det J_total|
        """
        c_n = self.normalize_context(c)
        x_norm, logdet_flow = self.transform_and_jac_fn.apply(self.params, z, c_n)
        x = self.denormalize_data(x_norm)
        logdet_denorm = self._logdet_denormalize_data(x_norm)
        return x, logdet_flow + logdet_denorm

    def save_params(self, flow_save_path):
        leaves, treedef = jax.tree_util.tree_flatten(self.params)
        save_dict = {f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)}
        save_dict["treedef"] = np.array(str(treedef))
        np.savez(flow_save_path, **save_dict)
    
    def load_params(self, flow_load_path):
        data = np.load(flow_load_path, allow_pickle=True)
        _, treedef = jax.tree_util.tree_flatten(self.params)
        leaves = [jnp.asarray(data[f"leaf_{i}"]) for i in range(treedef.num_leaves)]
        self.params = jax.tree_util.tree_unflatten(treedef, leaves)


class FlowMatching:
    """
    Unconditional Flow Matching  (optimal-transport straight paths).

    Replaces the RQS bijector stack with a single vector-field MLP
    v_theta(x_t, t) trained to push N(0,I) → p(x) along straight paths.

    Training objective (CFM / OT-CFM):
        x_t    = (1 - t) * x0 + t * x1          straight path
        target = x1 - (1 - sigma_min) * x0       constant vector field
        loss   = E[ || v(x_t, t) - target ||^2 ]

    Sampling:
        Integrate dx/dt = v_theta(x_t, t) from t=0 to t=1
        using Euler (fast) or RK4 (accurate).

    Pre/post-processing (power-affine normalisation) is identical to
    the Flow class so the two are interchangeable.

    Parameters
    ----------
    data : array (N, D)
        Training dataset.
    hidden_size : int
        Width of vector-field MLP hidden layers.
    num_layers : int
        Number of hidden layers in the vector-field MLP.
    learning_rate : float
        Adam learning rate.
    B : float
        Bound for normalized space (same role as in Flow).
    p : float
        Power transform parameter.
    sigma_min : float
        Minimum path noise; keeps paths slightly stochastic.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        data,
        hidden_size=256,
        num_layers=4,
        learning_rate=1e-4,
        B=6.0,
        p=0.3,
        sigma_min=1e-4,
        seed=0,
    ):
        self.N, self.D = data.shape
        self.B         = float(B) - 1
        self.sigma_min = sigma_min

        # ── Power transform (same as Flow) ───────────────────────────────
        self.alpha      = jnp.log(p)
        dummy           = jnp.sign(data) * jnp.abs(data) ** jnp.exp(self.alpha)
        min_x           = dummy.min(axis=0)
        max_x           = dummy.max(axis=0)
        self.mean_x     = (max_x + min_x) / 2
        self.half_range = (max_x - min_x) / 2

        self.x_norm_data = self.to_unit_interval(data)
        assert self.x_norm_data.min() > -B
        assert self.x_norm_data.max() <  B

        # ── Architecture ─────────────────────────────────────────────────
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.learning_rate = learning_rate
        self.key           = jr.PRNGKey(seed)

        self._build_model()
        self._init_params()

    # ====================================================================
    # Data transforms  (identical to Flow)
    # ====================================================================

    @partial(jax.jit, static_argnums=0)
    def to_unit_interval(self, arr):
        dummy = jnp.sign(arr) * jnp.abs(arr) ** jnp.exp(self.alpha)
        return (dummy - self.mean_x[None, :]) / self.half_range[None, :] * self.B

    @partial(jax.jit, static_argnums=0)
    def from_unit_interval(self, arr):
        dummy = self.half_range[None, :] / self.B * arr + self.mean_x[None, :]
        return jnp.sign(dummy) * jnp.abs(dummy) ** (1.0 / jnp.exp(self.alpha))

    @partial(jax.jit, static_argnums=0)
    def logdet_to_unit_interval(self, arr):
        gamma = jnp.exp(self.alpha)
        return jnp.sum(
            jnp.log(gamma)
            + (gamma - 1.0) * jnp.log(jnp.abs(arr))
            + jnp.log(self.B / self.half_range),
            axis=-1,
        )

    @partial(jax.jit, static_argnums=0)
    def logdet_from_unit_interval(self, arr):
        beta  = 1.0 / jnp.exp(self.alpha)
        dummy = self.half_range[None, :] / self.B * arr + self.mean_x[None, :]
        return jnp.sum(
            jnp.log(self.half_range / self.B)
            + jnp.log(beta)
            + (beta - 1.0) * jnp.log(jnp.abs(dummy)),
            axis=-1,
        )

    # ====================================================================
    # Batching  (identical to Flow)
    # ====================================================================

    @staticmethod
    def get_batches(data, batch_size, key):
        N   = data.shape[0]
        idx = jax.random.permutation(key, N)
        for i in range(0, N, batch_size):
            yield data[idx[i: i + batch_size]]

    # ====================================================================
    # Vector field MLP
    # ====================================================================

    # @staticmethod
    # def make_vector_field(D, hidden_size, num_layers):
    #     """
    #     Build the Haiku-transformed vector field v_theta(x_t, t) → (D,).

    #     Input: [x_t (D) | t (1)]  →  MLP  →  D

    #     Uses SiLU activations (smoother than ReLU for ODE flows).
    #     Output layer is zero-initialised so the field starts near zero
    #     — the same trick as the zero-init conditioner in the RQS flow.
    #     """
    #     @hk.without_apply_rng
    #     @hk.transform
    #     def vector_field_fn(x_t, t):
    #         # t: (batch,) → (batch, 1) for concatenation
    #         t_feat = jnp.atleast_1d(t)
    #         if t_feat.ndim == 1:
    #             t_feat = t_feat[:, None]

    #         inp = jnp.concatenate([x_t, t_feat], axis=-1)   # (batch, D+1)

    #         # Build MLP: [hidden × num_layers] → D
    #         layers = []
    #         for _ in range(num_layers):
    #             layers += [hk.Linear(hidden_size), jax.nn.silu]
    #         layers.append(
    #             hk.Linear(D, w_init=jnp.zeros, b_init=jnp.zeros)
    #         )
    #         return hk.Sequential(layers)(inp)

    #     return vector_field_fn

    @staticmethod
    def make_vector_field(D, hidden_size, num_layers):

        @hk.without_apply_rng
        @hk.transform
        def vector_field_fn(x_t, t):
            # Sinusoidal time embedding  (replaces raw scalar t)
            half = hidden_size // 2
            freqs = jnp.exp(
                -jnp.log(10000.0) * jnp.arange(half) / (half - 1)
            )                                               # (half,)
            t_feat = jnp.atleast_1d(t)[:, None] * freqs[None, :]  # (batch, half)
            t_emb  = jnp.concatenate(
                [jnp.sin(t_feat), jnp.cos(t_feat)], axis=-1
            )                                               # (batch, hidden_size)

            inp = jnp.concatenate([x_t, t_emb], axis=-1)   # (batch, D + hidden_size)

            layers = []
            for _ in range(num_layers):
                layers += [hk.Linear(hidden_size), jax.nn.silu]
            layers.append(
                hk.Linear(D, w_init=jnp.zeros, b_init=jnp.zeros)
            )
            return hk.Sequential(layers)(inp)

        return vector_field_fn
        
    # ====================================================================
    # Build / init
    # ====================================================================

    def _build_model(self):
        self.vf_fn = self.make_vector_field(
            self.D, self.hidden_size, self.num_layers
        )

    def _init_params(self):
        self.key, k = jr.split(self.key)
        x0 = jnp.zeros((2, self.D))
        t0 = jnp.zeros((2,))
        self.params    = self.vf_fn.init(k, x0, t0)
        self.optimizer = optax.adam(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    # ====================================================================
    # Training
    # ====================================================================

    def loss_fn(self, params, x1, key):
        """
        CFM loss  E[ || v(x_t, t) - (x1 - (1-sigma_min)*x0) ||^2 ]
        """
        k1, k2  = jr.split(key)
        x0      = jr.normal(k1, x1.shape)
        t       = jr.uniform(k2, (x1.shape[0],))
        t_b     = t[:, None]
        x_t     = (1.0 - (1.0 - self.sigma_min) * t_b) * x0 + t_b * x1
        target  = x1 - (1.0 - self.sigma_min) * x0
        v_pred  = self.vf_fn.apply(params, x_t, t)
        return jnp.mean((v_pred - target) ** 2)

    @partial(jax.jit, static_argnums=0)
    def update(self, params, opt_state, batch, key):
        loss, grads = jax.value_and_grad(self.loss_fn)(params, batch, key)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def fit(self, num_epochs=100, batch_size=256):
        """
        Train via mini-batch gradient descent.
        """
        for _ in trange(num_epochs, colour='green', desc='Training FlowMatching'):
            self.key, data_key, step_key = jr.split(self.key, 3)
            for batch in self.get_batches(self.x_norm_data, batch_size, data_key):
                self.key, step_key = jr.split(self.key)
                self.params, self.opt_state, _ = self.update(
                    self.params, self.opt_state, batch, step_key
                )

    # ====================================================================
    # ODE integrators  (jax.lax.scan → single XLA program per call)
    # ====================================================================

    # @partial(jax.jit, static_argnums=(0, 2))
    # def _euler(self, x0, num_steps):
    #     """Euler integrator.  1 MLP call per step."""
    #     dt = 1.0 / num_steps

    #     def step(x, t_scalar):
    #         t = jnp.full((x.shape[0],), t_scalar)
    #         return x + dt * self.vf_fn.apply(self.params, x, t), None

    #     ts      = jnp.linspace(0.0, 1.0 - dt, num_steps)
    #     x1, _   = jax.lax.scan(step, x0, ts)
    #     return x1

    @partial(jax.jit, static_argnums=(0, 2))
    def _euler(self, x0, num_steps):
        """
        More steps near t=1 where the tail vector field is largest.
        Uses a cosine schedule instead of uniform spacing.
        """
        # cosine spacing: dense near t=1
        i    = jnp.arange(num_steps)
        ts   = 1.0 - jnp.cos(i / num_steps * jnp.pi / 2)   # (num_steps,)
        dts  = jnp.diff(ts, append=jnp.array([1.0]))         # step sizes

        def step(x, t_and_dt):
            t_scalar, dt = t_and_dt
            t = jnp.full((x.shape[0],), t_scalar)
            return x + dt * self.vf_fn.apply(self.params, x, t), None

        x1, _ = jax.lax.scan(step, x0, (ts, dts))
        return x1

    @partial(jax.jit, static_argnums=(0, 2))
    def _rk4(self, x0, num_steps):
        """RK4 integrator.  4 MLP calls per step, much better quality."""
        dt = 1.0 / num_steps

        def step(x, t_scalar):
            t  = jnp.full((x.shape[0],), t_scalar)
            k1 = self.vf_fn.apply(self.params, x,              t         )
            k2 = self.vf_fn.apply(self.params, x + dt/2 * k1, t + dt/2  )
            k3 = self.vf_fn.apply(self.params, x + dt/2 * k2, t + dt/2  )
            k4 = self.vf_fn.apply(self.params, x + dt   * k3, t + dt    )
            return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4), None

        ts      = jnp.linspace(0.0, 1.0 - dt, num_steps)
        x1, _   = jax.lax.scan(step, x0, ts)
        return x1

    # ====================================================================
    # Public API  (mirrors Flow exactly)
    # ====================================================================

    def sample(self, num_samples, num_steps=10, method='euler'):
        """
        Draw samples in physical space.

        Parameters
        ----------
        num_samples : int
        num_steps   : int    ODE steps.  1–4 for speed, 10–20 for quality.
        method      : 'euler' | 'rk4'
        """
        self.key, subkey = jr.split(self.key)
        x0       = jr.normal(subkey, (num_samples, self.D))
        integrate = self._euler if method == 'euler' else self._rk4
        x1_norm  = integrate(x0, num_steps)
        return self.from_unit_interval(x1_norm)

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def forward_pass(self, z, num_steps=10, method='euler'):
        """
        Map latent samples z ~ N(0,I) to physical space.

        Parameters
        ----------
        z         : (batch, D)
        num_steps : int
        method    : 'euler' | 'rk4'
        """
        integrate = self._euler if method == 'euler' else self._rk4
        x1_norm   = integrate(z, num_steps)
        return self.from_unit_interval(x1_norm)

    def backward_pass(self, x, num_steps=10, method='euler'):
        """
        Map physical-space samples back to latent space  (reverse ODE).

        Runs the ODE backwards from t=1 to t=0.

        Parameters
        ----------
        x         : (batch, D)  physical-space points
        num_steps : int
        method    : 'euler' | 'rk4'
        """
        x_norm = self.to_unit_interval(x)

        dt = 1.0 / num_steps

        def euler_step_back(z, t_scalar):
            t = jnp.full((z.shape[0],), t_scalar)
            return z - dt * self.vf_fn.apply(self.params, z, t), None

        def rk4_step_back(z, t_scalar):
            t  = jnp.full((z.shape[0],), t_scalar)
            k1 = self.vf_fn.apply(self.params, z,              t         )
            k2 = self.vf_fn.apply(self.params, z - dt/2 * k1, t - dt/2  )
            k3 = self.vf_fn.apply(self.params, z - dt/2 * k2, t - dt/2  )
            k4 = self.vf_fn.apply(self.params, z - dt   * k3, t - dt    )
            return z - dt / 6 * (k1 + 2*k2 + 2*k3 + k4), None

        step_fn = euler_step_back if method == 'euler' else rk4_step_back
        ts      = jnp.linspace(1.0, dt, num_steps)          # t=1 → t=0
        z0, _   = jax.lax.scan(step_fn, x_norm, ts)
        return z0

    def save_params(self, path):
        leaves, _  = jax.tree_util.tree_flatten(self.params)
        save_dict  = {f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)}
        np.savez(path, **save_dict)

    def load_params(self, path):
        data     = np.load(path, allow_pickle=True)
        _, treedef = jax.tree_util.tree_flatten(self.params)
        leaves   = [jnp.asarray(data[f"leaf_{i}"]) for i in range(treedef.num_leaves)]
        self.params = jax.tree_util.tree_unflatten(treedef, leaves)


# ═════════════════════════════════════════════════════════════════════════════
#  ConditionalFlowMatching  —  conditional
# ═════════════════════════════════════════════════════════════════════════════

class ConditionalFlowMatching:
    """
    Conditional Flow Matching  —  models p(x | c).

    Architecture mirrors ConditionalFlow exactly:
    - Same power-affine normalisation for both data and context.
    - Same memmapped-safe batch iterator.
    - Same fit() modes ('pre-loaded' and tset_paths).
    - Same save/load interface.

    The RQS bijector stack is replaced by a single vector-field MLP:
        v_theta(x_t, t, c_norm) → (D,)
    which receives [x_t ‖ t ‖ c_norm] as input.

    Sampling uses Euler or RK4 ODE integration.

    Parameters
    ----------
    data, context : array-like (N, D) / (N, C)
        May be numpy memmaps.
    data_min / max, context_min / max : (D,) / (C,)
        Per-feature physical-space bounds.
    normalizer_dict : dict or jnp.array([False])
        Pre-fitted normalizer dict with keys 'data' and 'context'.
        If supplied, data_min/max and context_min/max are ignored.
    tset_paths : list of str, optional
        Paths to .npy training-set files (used in fit mode 'on-the-fly').
    last_feature_index : int, optional
        Column split: data = arr[:, :last_feature_index],
                      context = arr[:, last_feature_index:]
    flow_save_path / flow_load_path : str, optional
        Paths for save_params / load_params.
    p, B : float
        Normalisation hyperparameters.
    hidden_size : int
        Width of vector-field MLP hidden layers.
    num_layers : int
        Number of hidden layers in the vector-field MLP.
    learning_rate : float
    sigma_min : float
        Minimum path noise.
    seed : int
    """

    def __init__(
        self,
        data,
        context,
        data_min=None,
        data_max=None,
        context_min=None,
        context_max=None,
        normalizer_dict=jnp.array([False]),
        tset_paths=None,
        last_feature_index=None,
        last_feature_index_for_context=None,
        flow_save_path=None,
        flow_load_path=None,
        p=0.1,
        B=4.0,
        hidden_size=256,
        num_layers=4,
        learning_rate=1e-4,
        sigma_min=1e-4,
        seed=0,
    ):
        self.N, self.D      = data.shape[0], data.shape[1]
        self.C              = context.shape[1]
        self.sigma_min      = sigma_min
        self.tset_paths     = tset_paths
        self.last_feature_idx = last_feature_index
        self.last_feature_index_for_context = last_feature_index_for_context
        self.flow_save_path = flow_save_path
        self.flow_load_path = flow_load_path

        self._data    = data
        self._context = context

        # ── Normalizers ───────────────────────────────────────────────────
        if any(normalizer_dict):
            self.x_norm = normalizer_dict['data']
            self.c_norm = normalizer_dict['context']
        else:
            self.x_norm = _fit_normalizer(data_min,    data_max,    p, B)
            self.c_norm = _fit_normalizer(context_min, context_max, p, B)

        self._x_alpha      = jnp.float64(self.x_norm['alpha'])
        self._x_mean       = jnp.asarray(self.x_norm['mean'])
        self._x_half_range = jnp.asarray(self.x_norm['half_range'])
        self._x_B          = jnp.float64(self.x_norm['B'])

        self._c_alpha      = jnp.float64(self.c_norm['alpha'])
        self._c_mean       = jnp.asarray(self.c_norm['mean'])
        self._c_half_range = jnp.asarray(self.c_norm['half_range'])
        self._c_B          = jnp.float64(self.c_norm['B'])

        # ── Architecture ─────────────────────────────────────────────────
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.learning_rate = learning_rate
        self.key           = jr.PRNGKey(seed)

        self._build_model()
        self._init_params()

    # ====================================================================
    # Normalisation  (identical to ConditionalFlow)
    # ====================================================================

    @partial(jax.jit, static_argnums=0)
    def normalize_data(self, x):
        return _from_coefficients_jnp(
            x, self._x_alpha, self._x_mean, self._x_half_range, self._x_B
        )

    @partial(jax.jit, static_argnums=0)
    def denormalize_data(self, x_norm):
        return _to_coefficients_jnp(
            x_norm, self._x_alpha, self._x_mean, self._x_half_range, self._x_B
        )

    @partial(jax.jit, static_argnums=0)
    def normalize_context(self, c):
        return _from_coefficients_jnp(
            c, self._c_alpha, self._c_mean, self._c_half_range, self._c_B
        )

    @partial(jax.jit, static_argnums=0)
    def denormalize_context(self, c_norm):
        return _to_coefficients_jnp(
            c_norm, self._c_alpha, self._c_mean, self._c_half_range, self._c_B
        )

    @partial(jax.jit, static_argnums=0)
    def _logdet_normalize_data(self, x):
        return _logdet_from_coefficients_jnp(
            x, self._x_alpha, self._x_mean, self._x_half_range, self._x_B
        )

    @partial(jax.jit, static_argnums=0)
    def _logdet_denormalize_data(self, x_norm):
        return _logdet_to_coefficients_jnp(
            x_norm, self._x_alpha, self._x_mean, self._x_half_range, self._x_B
        )

    # ====================================================================
    # Vector field MLP
    # ====================================================================

    @staticmethod
    def make_vector_field(D, C, hidden_size, num_layers):

        @hk.without_apply_rng
        @hk.transform
        def vector_field_fn(x_t, t, c_norm):
            # ── Sinusoidal time embedding ───────────────────────────────
            # Gives the network hidden_size features for time instead of 1.
            # Critical for high-dimensional data where the field varies
            # rapidly with t.
            half  = hidden_size // 2
            freqs = jnp.exp(
                -jnp.log(10000.0) * jnp.arange(half) / (half - 1)
            )                                               # (half,)
            t_feat = jnp.atleast_1d(t)[:, None] * freqs[None, :]
            t_emb  = jnp.concatenate(
                [jnp.sin(t_feat), jnp.cos(t_feat)], axis=-1
            )                                               # (batch, hidden_size)

            # ── Input projection ────────────────────────────────────────
            # Project x_t and c separately before combining — avoids the
            # 450-dim input immediately bottlenecking through a 1024-wide
            # layer that also has to process t and c simultaneously.
            x_proj = hk.Linear(hidden_size, name='x_proj')(x_t)
            c_proj = hk.Linear(hidden_size, name='c_proj')(c_norm)

            h = x_proj + t_emb + c_proj    # additive fusion, shape (batch, H)

            # ── Residual MLP ─────────────────────────────────────────────
            # Residual connections help gradients flow in deep networks
            # and are essential for D=450 where the output must closely
            # track the input (the field is often small).
            for _ in range(num_layers):
                h_new = hk.Linear(hidden_size)(h)
                h_new = jax.nn.silu(h_new)
                h_new = hk.Linear(hidden_size)(h_new)
                h     = h + h_new           # residual skip connection

            return hk.Linear(D, w_init=jnp.zeros, b_init=jnp.zeros)(h)

        return vector_field_fn

    # ====================================================================
    # Build / init
    # ====================================================================

    def _build_model(self):
        self.vf_fn = self.make_vector_field(
            self.D, self.C, self.hidden_size, self.num_layers
        )

    def _init_params(self):
        self.key, k = jr.split(self.key)
        x0 = jnp.zeros((2, self.D))
        t0 = jnp.zeros((2,))
        c0 = jnp.zeros((2, self.C))
        self.params    = self.vf_fn.init(k, x0, t0, c0)
        self.optimizer = optax.adam(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    # ====================================================================
    # Training
    # ====================================================================

    def loss_fn(self, params, x1, c_norm, key):
        """
        Conditional CFM loss  E[ || v(x_t, t, c) - target ||^2 ]
        """
        k1, k2 = jr.split(key)
        x0     = jr.normal(k1, x1.shape)
        t      = jr.uniform(k2, (x1.shape[0],))
        t_b    = t[:, None]
        x_t    = (1.0 - (1.0 - self.sigma_min) * t_b) * x0 + t_b * x1
        target = x1 - (1.0 - self.sigma_min) * x0
        v_pred = self.vf_fn.apply(params, x_t, t, c_norm)
        return jnp.mean((v_pred - target) ** 2)

    @partial(jax.jit, static_argnums=0, donate_argnums=(1, 2))
    def _update(self, params, opt_state, x_batch, c_batch, key):
        loss, grads = jax.value_and_grad(self.loss_fn)(
            params, x_batch, c_batch, key
        )
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def fit(self, mode='pre-loaded', num_epochs=100, batch_size=256):
        """
        Train the model.

        mode='pre-loaded'  : uses self._data / self._context via _iter_batches
                             (memmapped-safe, lazy reads).
        mode='on-the-fly'  : randomly picks a file from self.tset_paths each
                             epoch, identical to ConditionalFlow's second mode.
        """
        if mode == 'pre-loaded':
            for _ in trange(num_epochs, colour='green',
                            desc='Training ConditionalFlowMatching (pre-loaded)'):
                self.key, data_key, step_key = jr.split(self.key, 3)
                for x_batch, c_batch in _iter_batches(
                    self._data, self._context,
                    self.x_norm, self.c_norm,
                    batch_size, data_key,
                ):
                    self.key, step_key = jr.split(self.key)
                    self.params, self.opt_state, _ = self._update(
                        self.params, self.opt_state, x_batch, c_batch, step_key
                    )
        else:
            for _ in trange(num_epochs, colour='blue',
                            desc='Training ConditionalFlowMatching (on-the-fly)'):
                rand_idx     = random.randint(0, len(self.tset_paths) - 1)
                training_set = jnp.load(self.tset_paths[rand_idx])
                idxs         = jnp.array(random.sample(
                    range(training_set.shape[0]),
                    k=min(batch_size, training_set.shape[0]),
                ))
                training_set = training_set[idxs]
                x_batch = self.normalize_data(training_set[:, :self.last_feature_idx])
                c_batch = self.normalize_context(training_set[:, self.last_feature_idx:self.last_feature_index_for_context])
                self.key, step_key = jr.split(self.key)
                self.params, self.opt_state, _ = self._update(
                    self.params, self.opt_state, x_batch, c_batch, step_key
                )

    # ====================================================================
    # ODE integrators
    # ====================================================================

    @partial(jax.jit, static_argnums=(0, 3))
    def _euler(self, x0, c_norm, num_steps):
        dt = 1.0 / num_steps

        def step(x, t_scalar):
            t = jnp.full((x.shape[0],), t_scalar)
            return x + dt * self.vf_fn.apply(self.params, x, t, c_norm), None

        ts    = jnp.linspace(0.0, 1.0 - dt, num_steps)
        x1, _ = jax.lax.scan(step, x0, ts)
        return x1

    @partial(jax.jit, static_argnums=(0, 3))
    def _rk4(self, x0, c_norm, num_steps):
        dt = 1.0 / num_steps

        def step(x, t_scalar):
            t  = jnp.full((x.shape[0],), t_scalar)
            k1 = self.vf_fn.apply(self.params, x,              t,        c_norm)
            k2 = self.vf_fn.apply(self.params, x + dt/2 * k1, t + dt/2, c_norm)
            k3 = self.vf_fn.apply(self.params, x + dt/2 * k2, t + dt/2, c_norm)
            k4 = self.vf_fn.apply(self.params, x + dt   * k3, t + dt,   c_norm)
            return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4), None

        ts    = jnp.linspace(0.0, 1.0 - dt, num_steps)
        x1, _ = jax.lax.scan(step, x0, ts)
        return x1

    # ====================================================================
    # Public API  (mirrors ConditionalFlow exactly)
    # ====================================================================

    def sample(self, c, num_samples, num_steps=10, method='euler'):
        """
        Draw samples from p(x | c).

        Parameters
        ----------
        c           : (C,) | (1,C) | (num_samples, C)  physical-space context
        num_samples : int
        num_steps   : int    ODE steps
        method      : 'euler' | 'rk4'
        """
        c = jnp.atleast_2d(jnp.asarray(c, dtype=jnp.float64))
        if c.shape[0] == 1:
            c = jnp.tile(c, (num_samples, 1))

        c_n      = self.normalize_context(c)
        integrate = self._euler if method == 'euler' else self._rk4

        self.key, subkey = jr.split(self.key)
        x0      = jr.normal(subkey, (num_samples, self.D))
        x1_norm = integrate(x0, c_n, num_steps)
        return self.denormalize_data(x1_norm)

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def forward_pass(self, z, c, num_steps=10, method='euler'):
        """
        Map latent z ~ N(0,I) to physical space given context c.

        Parameters
        ----------
        z         : (batch, D)
        c         : (batch, C)  physical-space context
        num_steps : int
        method    : 'euler' | 'rk4'
        """
        c_n      = self.normalize_context(c)
        integrate = self._euler if method == 'euler' else self._rk4
        x1_norm  = integrate(z, c_n, num_steps)
        return self.denormalize_data(x1_norm)

    def backward_pass(self, x, c, num_steps=10, method='euler'):
        """
        Map physical-space samples back to latent space (reverse ODE).

        Parameters
        ----------
        x         : (batch, D)  physical-space points
        c         : (batch, C)  physical-space context
        num_steps : int
        method    : 'euler' | 'rk4'
        """
        x_norm = self.normalize_data(x)
        c_n    = self.normalize_context(c)
        dt     = 1.0 / num_steps

        def euler_back(z, t_scalar):
            t = jnp.full((z.shape[0],), t_scalar)
            return z - dt * self.vf_fn.apply(self.params, z, t, c_n), None

        def rk4_back(z, t_scalar):
            t  = jnp.full((z.shape[0],), t_scalar)
            k1 = self.vf_fn.apply(self.params, z,              t,        c_n)
            k2 = self.vf_fn.apply(self.params, z - dt/2 * k1, t - dt/2, c_n)
            k3 = self.vf_fn.apply(self.params, z - dt/2 * k2, t - dt/2, c_n)
            k4 = self.vf_fn.apply(self.params, z - dt   * k3, t - dt,   c_n)
            return z - dt / 6 * (k1 + 2*k2 + 2*k3 + k4), None

        step_fn  = euler_back if method == 'euler' else rk4_back
        ts       = jnp.linspace(1.0, dt, num_steps)
        z0, _    = jax.lax.scan(step_fn, x_norm, ts)
        return z0

    # ====================================================================
    # Save / load  (identical interface to ConditionalFlow)
    # ====================================================================

    def save_params(self):
        leaves, _  = jax.tree_util.tree_flatten(self.params)
        save_dict  = {f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)}
        np.savez(self.flow_save_path, **save_dict)

    def load_params(self):
        data       = np.load(self.flow_load_path, allow_pickle=True)
        _, treedef = jax.tree_util.tree_flatten(self.params)
        leaves     = [jnp.asarray(data[f"leaf_{i}"]) for i in range(treedef.num_leaves)]
        self.params = jax.tree_util.tree_unflatten(treedef, leaves)

"""
Masked Autoregressive Flow (MAF) with MADE conditioners.

Architecture
------------
The RQS masked-coupling layers in ConditionalFlow are replaced by
Masked Autoregressive Flow (MAF) layers, each built from a MADE block.

MADE vs MLP conditioner
-----------------------
MaskedCoupling (what ConditionalFlow uses):
    - Splits x into two halves by index parity (mask).
    - The *unchanged* half is fed through an unconstrained MLP to produce
      spline params for the *transformed* half.
    - Parallelisable: all transformed dims updated simultaneously.
    - Requires num_layers coupling layers to mix all dimensions.

MADE (what this class uses):
    - A single network with masked weights enforces the autoregressive
      property: output_i depends only on inputs x_{<i}.
    - Every dimension gets its own dedicated conditioner output in one
      forward pass — no split needed.
    - More expressive per layer: D autoregressive conditionals vs D/2.
    - Forward (z→x) is sequential O(D) — slow for sampling.
    - Inverse (x→z) is parallel O(1) — fast for density evaluation.
    - This is MAF: fast log_prob, slow sample.
    - Contrast with IAF (fast sample, slow log_prob).

For PTA / GW inference you usually call log_prob far more often than
sample, so MAF is the natural choice.

Mask construction (Germain et al. 2015)
----------------------------------------
Assign each unit an integer "degree" m:
    Input  units: m(x_i) = i          (1-indexed, 0..D-1 here)
    Hidden units: m(h_k) ~ Uniform{min_degree, D-1}
    Output units: m(out_i) = i

Weight mask W_{kj} = 1  iff  m(k) >= m(j)   (hidden→hidden, hidden→input)
           W_{kj} = 1  iff  m(k) >  m(j)   (output→hidden)

This guarantees output_i depends only on x_{0..i-1}.

Context conditioning
--------------------
Context c is concatenated to the *input* of every MADE hidden layer,
after the masked input projection. Because c is not autoregressive
(it conditions everything), it bypasses the masks entirely.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import haiku as hk
import optax
import distrax
import numpy as np
from tqdm.auto import trange
from functools import partial
import random

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers  (identical to conditional_flows.py)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_normalizer(min_x, max_x, p, B):
    min_x      = np.asarray(min_x, dtype=np.float64)
    max_x      = np.asarray(max_x, dtype=np.float64)
    alpha      = np.log(float(p))
    gamma      = np.exp(alpha)
    dummy_min  = np.sign(min_x) * np.abs(min_x) ** gamma
    dummy_max  = np.sign(max_x) * np.abs(max_x) ** gamma
    mean       = (dummy_max + dummy_min) / 2.0
    half_range = (dummy_max - dummy_min) / 2.0
    return dict(alpha=alpha, mean=mean, half_range=half_range, B=float(B))

def _from_coefficients_np(arr, norm):
    alpha, mean, half_range, B = (
        norm['alpha'], norm['mean'], norm['half_range'], norm['B'])
    dummy = np.sign(arr) * np.abs(arr) ** np.exp(alpha)
    return (dummy - mean) / half_range * B

def _from_coefficients_jnp(arr, alpha, mean, half_range, B):
    dummy = jnp.sign(arr) * jnp.abs(arr) ** jnp.exp(alpha)
    return (dummy - mean) / half_range * B

def _to_coefficients_jnp(arr, alpha, mean, half_range, B):
    dummy = half_range / B * arr + mean
    return jnp.sign(dummy) * jnp.abs(dummy) ** (1.0 / jnp.exp(alpha))

def _logdet_from_coefficients_jnp(arr, alpha, mean, half_range, B):
    gamma = jnp.exp(alpha)
    return jnp.sum(
        jnp.log(gamma)
        + (gamma - 1.0) * jnp.log(jnp.abs(arr))
        + jnp.log(B / half_range),
        axis=-1,
    )

def _logdet_to_coefficients_jnp(arr, alpha, mean, half_range, B):
    beta  = 1.0 / jnp.exp(alpha)
    dummy = half_range / B * arr + mean
    return jnp.sum(
        jnp.log(half_range / B)
        + jnp.log(beta)
        + (beta - 1.0) * jnp.log(jnp.abs(dummy)),
        axis=-1,
    )

def _iter_batches(data, context, x_norm, c_norm, batch_size, key):
    N    = data.shape[0]
    perm = np.array(jr.permutation(key, N))
    for start in range(0, N, batch_size):
        idx = perm[start: start + batch_size]
        if len(idx) < batch_size:
            continue
        x_raw   = np.asarray(data[idx],    dtype=np.float64)
        c_raw   = np.asarray(context[idx], dtype=np.float64)
        x_batch = jnp.asarray(_from_coefficients_np(x_raw, x_norm))
        c_batch = jnp.asarray(_from_coefficients_np(c_raw, c_norm))
        yield x_batch, c_batch


# ─────────────────────────────────────────────────────────────────────────────
# MADE layer  (the core building block)
# ─────────────────────────────────────────────────────────────────────────────

class MaskedLinear(hk.Module):
    """
    A linear layer whose weight matrix is elementwise-multiplied by a
    fixed binary mask, enforcing the autoregressive ordering.

    mask : (out_features, in_features) bool array
    """

    def __init__(self, out_features, mask, with_bias=True, name=None):
        super().__init__(name=name)
        self.out_features = out_features
        self.mask         = mask          # (out, in) — fixed at construction
        self.with_bias    = with_bias

    def __call__(self, x):
        # x : (..., in_features)
        w = hk.get_parameter(
            'w',
            shape=(self.mask.shape[1], self.out_features),
            dtype=x.dtype,
            init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        )
        # Apply mask: zero out forbidden connections
        w_masked = w * self.mask.T          # (in, out)
        out = x @ w_masked
        if self.with_bias:
            b = hk.get_parameter(
                'b',
                shape=(self.out_features,),
                dtype=x.dtype,
                init=jnp.zeros,
            )
            out = out + b
        return out


class MADEBlock(hk.Module):
    """
    One MADE block: masked autoencoder for distribution estimation.

    Produces, for each input dimension i, the parameters of a
    RationalQuadraticSpline conditioned on x_{<i} and context c.

    Architecture
    ------------
    Input layer:  MaskedLinear  (D → hidden)   mask enforces autoregression
    Hidden layers: MaskedLinear (hidden → hidden) + activation
    Output layer: MaskedLinear  (hidden → D * num_params)  strict mask

    Context injection
    -----------------
    c is concatenated to the hidden representation *after* the first
    masked projection, then at every subsequent hidden layer.
    This means c sees all of x but x_i never sees x_{>=i}.

    Parameters
    ----------
    D             : int   input / output dimension
    C             : int   context dimension (0 for unconditional)
    hidden_size   : int   width of hidden layers
    num_hidden    : int   number of hidden layers
    num_params    : int   spline params per dimension  (3*num_bins + 1)
    """

    def __init__(self, D, C, hidden_size, num_hidden, num_params, name=None):
        super().__init__(name=name)
        self.D          = D
        self.C          = C
        self.hidden_size = hidden_size
        self.num_hidden  = num_hidden
        self.num_params  = num_params

        # ── Degree assignment (Germain et al. 2015) ───────────────────────
        # Input degrees:  0, 1, ..., D-1
        # Hidden degrees: cycle through 0..D-2  (never D-1 so at least
        #                 one valid path reaches every output)
        # Output degrees: 0, 1, ..., D-1  (same as input)
        self.m_input  = np.arange(D)                         # (D,)
        self.m_hidden = np.arange(hidden_size) % max(1, D-1) # (H,)
        self.m_output = np.arange(D)                         # (D,)

        # ── Masks ─────────────────────────────────────────────────────────
        # input → first hidden:  h_k receives x_j  iff  m_h[k] >= m_in[j]
        self.mask_in = jnp.asarray(
            self.m_hidden[:, None] >= self.m_input[None, :],  # (H, D)
            dtype=jnp.float32,
        )

        # hidden → hidden:  m_h[k] >= m_h[j]
        self.mask_hh = jnp.asarray(
            self.m_hidden[:, None] >= self.m_hidden[None, :],  # (H, H)
            dtype=jnp.float32,
        )

        # last hidden → output:  out_i depends on x_{<i}
        # strict inequality: output_i must NOT see x_i
        # We tile to cover all num_params outputs per dimension
        mask_out_base = jnp.asarray(
            self.m_output[:, None] > self.m_hidden[None, :],  # (D, H)
            dtype=jnp.float32,
        )
        # Repeat for num_params outputs per dimension: (D*P, H)
        self.mask_out = jnp.repeat(mask_out_base, num_params, axis=0)

    def __call__(self, x, c=None):
        """
        x : (batch, D)
        c : (batch, C) or None

        Returns
        -------
        params : (batch, D, num_params)
        """
        # ── Input projection (masked) ─────────────────────────────────────
        h = MaskedLinear(self.hidden_size, self.mask_in, name='lin_in')(x)

        # ── Hidden layers: inject context after each masked projection ────
        for i in range(self.num_hidden):
            h = jax.nn.tanh(h)
            # Append context if provided (no mask needed — c is global)
            if c is not None:
                h_cat = jnp.concatenate([h, c], axis=-1)
                # Project back to hidden_size with a plain (unmasked) linear
                # so the context information is mixed in before the next
                # masked step
                h = hk.Linear(
                    self.hidden_size,
                    name=f'ctx_proj_{i}',
                    w_init=hk.initializers.VarianceScaling(
                        1.0, 'fan_avg', 'uniform'),
                    b_init=jnp.zeros,
                )(h_cat)
            if i < self.num_hidden - 1:
                h = MaskedLinear(
                    self.hidden_size, self.mask_hh, name=f'lin_h{i}'
                )(h)

        h = jax.nn.tanh(h)

        # ── Output projection (strictly masked, zero-init) ────────────────
        out = MaskedLinear(
            self.D * self.num_params,
            self.mask_out,
            name='lin_out',
        )(h)
        # Reshape: (batch, D*P) → (batch, D, P)
        return out.reshape(x.shape[0], self.D, self.num_params)


# ─────────────────────────────────────────────────────────────────────────────
# MAF layer: one autoregressive RQS transformation
# ─────────────────────────────────────────────────────────────────────────────

class MAFLayer(hk.Module):
    """
    One MAF layer wrapping a MADEBlock with RQS bijectors.

    Inverse pass (x → z, used for log_prob): parallel O(1)
        For each i: z_i = RQS_i^{-1}(x_i | params_i(x_{<i}, c))
        All i computed simultaneously because params depend on x (known).

    Forward pass (z → x, used for sample): sequential O(D)
        For each i: x_i = RQS_i(z_i | params_i(x_{<i}, c))
        x_i needed to compute params_{i+1} — inherently sequential.
    """

    def __init__(self, D, C, hidden_size, num_hidden, num_bins, B, name=None):
        super().__init__(name=name)
        self.D          = D
        self.num_bins   = num_bins
        self.B          = B
        self.made       = MADEBlock(
            D, C, hidden_size, num_hidden,
            num_params=3 * num_bins + 1,
            name=f'{name}_made' if name else 'made',
        )

    def inverse_and_log_det(self, x, c=None):
        """
        x → z  (fast: all params computable in parallel)

        Returns z, log|det J_{x→z}|
        """
        # MADE forward pass: (batch, D, num_params)
        params = self.made(x, c)

        # Apply RQS inverse to each dimension independently
        def invert_dim(x_i, params_i):
            spline = distrax.RationalQuadraticSpline(
                params_i[None, :],          # (1, num_params)
                range_min=-(self.B + 1),
                range_max=  self.B + 1,
            )
            z_i, ld_i = spline.inverse_and_log_det(x_i[None])
            return z_i[0], ld_i[0]

        # vmap over the D dimensions, then over the batch
        def invert_sample(x_row, params_row):
            # x_row: (D,), params_row: (D, P)
            z_row, ld_row = jax.vmap(invert_dim)(x_row, params_row)
            return z_row, ld_row.sum()

        z, log_det = jax.vmap(invert_sample)(x, params)
        return z, log_det

    def forward_single(self, z_row, c_row=None):
        """
        z → x for a SINGLE sample — sequential loop over dimensions.
        Used inside jax.lax.scan during sampling.

        z_row   : (D,)
        c_row   : (C,) or None

        Returns x_row : (D,)
        """
        c_2d = c_row[None, :] if c_row is not None else None

        num_params = 3 * self.num_bins + 1

        def fori_fn(i, x_so_far):
            # All MADE params from current x_so_far — (1, D, P)
            all_params = self.made(x_so_far[None, :], c_2d)

            # Extract params for dimension i via dynamic_slice (JAX-safe)
            # all_params[0] shape: (D, P) → slice row i → (1, P)
            params_i = jax.lax.dynamic_slice(
                all_params[0], (i, 0), (1, num_params)
            )  # (1, P)

            z_i = jax.lax.dynamic_slice(z_row, (i,), (1,))  # (1,)

            spline = distrax.RationalQuadraticSpline(
                params_i,               # (1, P)
                range_min=-(self.B + 1),
                range_max=  self.B + 1,
            )
            x_i, _ = spline.forward_and_log_det(z_i)        # (1,)
            return x_so_far.at[i].set(x_i[0])

        x_row = jax.lax.fori_loop(0, self.D, fori_fn, jnp.zeros_like(z_row))
        return x_row


# ─────────────────────────────────────────────────────────────────────────────
# ConditionalMADEFlow
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalMADEFlow:
    """
    Conditional Masked Autoregressive Flow (MAF) with RQS bijectors.

    Replaces the MaskedCoupling + MLP conditioner of ConditionalFlow
    with stacked MAF layers, each backed by a MADE block.

    Key properties vs ConditionalFlow
    ----------------------------------
    log_prob  : O(num_layers) parallel MADE passes  — same speed
    sample    : O(num_layers × D) sequential steps  — SLOWER
    forward_pass : slow (sequential per dimension)
    backward_pass / inverse_pass : fast (parallel)

    Use this class when:
    - You evaluate log_prob much more often than you sample.
    - You want strictly autoregressive conditionals (no coupling approximation).
    - D is moderate (≤ ~50); for large D the sequential sampling is painful.

    All normalisation, save/load, fit modes, and public API are identical
    to ConditionalFlow so the two are drop-in replacements.
    """

    def __init__(
        self,
        data,
        context,
        data_min=None,
        data_max=None,
        context_min=None,
        context_max=None,
        normalizer_dict=jnp.array([False]),
        tset_paths=None,
        last_feature_index=None,
        flow_save_path=None,
        flow_load_path=None,
        p=0.1,
        B=4.0,
        flow_num_layers=4,
        hidden_size=128,
        mlp_num_layers=2,       # number of MADE hidden layers per MAF layer
        num_bins=8,
        learning_rate=1e-4,
        seed=0,
    ):
        self.N, self.D      = data.shape[0], data.shape[1]
        self.C              = context.shape[1]
        self.tset_paths     = tset_paths
        self.last_feature_idx = last_feature_index
        self.flow_save_path = flow_save_path
        self.flow_load_path = flow_load_path

        self._data    = data
        self._context = context

        # ── Normalizers ───────────────────────────────────────────────────
        if any(normalizer_dict):
            self.x_norm = normalizer_dict['data']
            self.c_norm = normalizer_dict['context']
        else:
            self.x_norm = _fit_normalizer(data_min, data_max, p, B)
            self.c_norm = _fit_normalizer(context_min, context_max, p, B)

        self._x_alpha      = jnp.float64(self.x_norm['alpha'])
        self._x_mean       = jnp.asarray(self.x_norm['mean'])
        self._x_half_range = jnp.asarray(self.x_norm['half_range'])
        self._x_B          = jnp.float64(self.x_norm['B'])

        self._c_alpha      = jnp.float64(self.c_norm['alpha'])
        self._c_mean       = jnp.asarray(self.c_norm['mean'])
        self._c_half_range = jnp.asarray(self.c_norm['half_range'])
        self._c_B          = jnp.float64(self.c_norm['B'])

        # ── Flow config ───────────────────────────────────────────────────
        self.flow_num_layers = flow_num_layers
        self.hidden_size     = hidden_size
        self.num_hidden      = mlp_num_layers   # MADE hidden layers
        self.num_bins        = num_bins
        self.learning_rate   = learning_rate
        self.B_float         = float(self.x_norm['B'])

        self.key = jr.PRNGKey(seed)
        self._build_model()
        self._init_params()

    # ====================================================================
    # Normalisation  (identical to ConditionalFlow)
    # ====================================================================

    @partial(jax.jit, static_argnums=0)
    def normalize_data(self, x):
        return _from_coefficients_jnp(
            x, self._x_alpha, self._x_mean, self._x_half_range, self._x_B)

    @partial(jax.jit, static_argnums=0)
    def denormalize_data(self, x_norm):
        return _to_coefficients_jnp(
            x_norm, self._x_alpha, self._x_mean, self._x_half_range, self._x_B)

    @partial(jax.jit, static_argnums=0)
    def normalize_context(self, c):
        return _from_coefficients_jnp(
            c, self._c_alpha, self._c_mean, self._c_half_range, self._c_B)

    @partial(jax.jit, static_argnums=0)
    def denormalize_context(self, c_norm):
        return _to_coefficients_jnp(
            c_norm, self._c_alpha, self._c_mean, self._c_half_range, self._c_B)

    @partial(jax.jit, static_argnums=0)
    def _logdet_normalize_data(self, x):
        return _logdet_from_coefficients_jnp(
            x, self._x_alpha, self._x_mean, self._x_half_range, self._x_B)

    @partial(jax.jit, static_argnums=0)
    def _logdet_denormalize_data(self, x_norm):
        return _logdet_to_coefficients_jnp(
            x_norm, self._x_alpha, self._x_mean, self._x_half_range, self._x_B)

    # ====================================================================
    # Model construction
    # ====================================================================

    def _make_maf_stack(self, x, c, inverse=True):
        """
        Build and apply all MAF layers.

        inverse=True  : x → z  (fast, for log_prob)
        inverse=False : z → x  (slow, for sample)
        """
        D, C         = self.D, self.C
        hidden_size  = self.hidden_size
        num_hidden   = self.num_hidden
        num_bins     = self.num_bins
        B            = self.B_float
        num_layers   = self.flow_num_layers

        # Between MAF layers we permute dimensions to improve mixing.
        # Use a fixed reversal permutation (simple and effective).
        perm    = jnp.arange(D - 1, -1, -1)   # reverse order
        inv_perm = jnp.argsort(perm)

        total_log_det = jnp.zeros(x.shape[0])

        if inverse:
            # x → z: apply layers in forward order, permute between them
            h = x
            for i in range(num_layers):
                layer = MAFLayer(
                    D, C, hidden_size, num_hidden, num_bins, B,
                    name=f'maf_{i}',
                )
                h, ld = layer.inverse_and_log_det(h, c)
                total_log_det = total_log_det + ld
                if i < num_layers - 1:
                    h = h[:, perm]          # permute between layers
            return h, total_log_det

        else:
            # z → x: apply layers in reverse order, un-permute between them
            # Build list of permutations applied during inverse
            perms = [perm] * (num_layers - 1)

            h = x
            for i in range(num_layers - 1, -1, -1):
                # Un-permute before applying inverse layer
                if i < num_layers - 1:
                    h = h[:, inv_perm]

                layer = MAFLayer(
                    D, C, hidden_size, num_hidden, num_bins, B,
                    name=f'maf_{i}',
                )
                # forward_single must be vmapped over the batch
                if c is not None:
                    h = jax.vmap(layer.forward_single)(h, c)
                else:
                    h = jax.vmap(lambda z_row: layer.forward_single(z_row))(h)

            return h

    def _build_model(self):
        D, C = self.D, self.C

        @hk.without_apply_rng
        @hk.transform
        def log_prob_fn(x_norm, c_norm):
            """x_norm → log p(x_norm | c_norm) in normalised space."""
            z, log_det = self._make_maf_stack(x_norm, c_norm, inverse=True)
            # Base distribution: N(0, I)
            log_p_z = jnp.sum(
                -0.5 * z ** 2 - 0.5 * jnp.log(2 * jnp.pi),
                axis=-1,
            )
            # log p(x) = log p(z) + log|det J_{x→z}|
            return log_p_z + log_det

        @hk.without_apply_rng
        @hk.transform
        def inverse_fn(x_norm, c_norm):
            """x_norm → z  (the fast direction)."""
            z, _ = self._make_maf_stack(x_norm, c_norm, inverse=True)
            return z

        @hk.without_apply_rng
        @hk.transform
        def inverse_and_logdet_fn(x_norm, c_norm):
            return self._make_maf_stack(x_norm, c_norm, inverse=True)

        @hk.without_apply_rng
        @hk.transform
        def forward_fn(z, c_norm):
            """z → x_norm  (the slow, sequential direction)."""
            return self._make_maf_stack(z, c_norm, inverse=False)

        self.log_prob_fn          = log_prob_fn
        self.inverse_fn           = inverse_fn
        self.inverse_and_logdet_fn = inverse_and_logdet_fn
        self.forward_fn           = forward_fn

    # ====================================================================
    # Initialisation
    # ====================================================================

    def _init_params(self):
        self.key, init_key = jr.split(self.key)

        x0 = jnp.asarray(
            _from_coefficients_np(
                np.asarray(self._data[:2], dtype=np.float64), self.x_norm))
        c0 = jnp.asarray(
            _from_coefficients_np(
                np.asarray(self._context[:2], dtype=np.float64), self.c_norm))

        self.params    = self.log_prob_fn.init(init_key, x0, c0)
        self.optimizer = optax.adam(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    # ====================================================================
    # Training  (identical to ConditionalFlow)
    # ====================================================================

    def loss_fn(self, params, x_batch, c_batch):
        return -jnp.mean(self.log_prob_fn.apply(params, x_batch, c_batch))

    @partial(jax.jit, static_argnums=0, donate_argnums=(1, 2))
    def _update(self, params, opt_state, x_batch, c_batch):
        grads = jax.grad(self.loss_fn)(params, x_batch, c_batch)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def fit(self, mode='pre-loaded', num_epochs=100, batch_size=256):
        if mode == 'pre-loaded':
            for _ in trange(num_epochs, colour='green',
                            desc='Training ConditionalMADEFlow (pre-loaded)'):
                self.key, subkey = jr.split(self.key)
                for x_batch, c_batch in _iter_batches(
                    self._data, self._context,
                    self.x_norm, self.c_norm,
                    batch_size, subkey,
                ):
                    self.params, self.opt_state = self._update(
                        self.params, self.opt_state, x_batch, c_batch)
        else:
            for _ in trange(num_epochs, colour='blue',
                            desc='Training ConditionalMADEFlow (on-the-fly)'):
                rand_idx     = random.randint(0, len(self.tset_paths) - 1)
                training_set = jnp.load(self.tset_paths[rand_idx])
                idxs         = jnp.array(random.sample(
                    range(training_set.shape[0]),
                    k=min(batch_size, training_set.shape[0]),
                ))
                training_set = training_set[idxs]
                x_batch = self.normalize_data(
                    training_set[:, :self.last_feature_idx])
                c_batch = self.normalize_context(
                    training_set[:, self.last_feature_idx:])
                self.params, self.opt_state = self._update(
                    self.params, self.opt_state, x_batch, c_batch)

    # ====================================================================
    # Public API  (same interface as ConditionalFlow)
    # ====================================================================

    @partial(jax.jit, static_argnums=0)
    def log_prob(self, x, c):
        """
        log p(x | c) in physical space.

        Fast: O(num_layers) parallel MADE evaluations.
        """
        x_n = self.normalize_data(x)
        c_n = self.normalize_context(c)
        logp_norm = self.log_prob_fn.apply(self.params, x_n, c_n)
        logdet    = self._logdet_normalize_data(x)
        return logp_norm + logdet

    def sample(self, c, num_samples, chunk_size=None):
        """
        Draw samples from p(x | c).

        SLOW: O(num_layers × D) sequential MADE calls per sample.
        Use chunk_size to bound memory for large num_samples.

        Parameters
        ----------
        c           : (C,) | (1,C) | (num_samples, C)  physical context
        num_samples : int
        chunk_size  : int or None
        """
        c = jnp.atleast_2d(jnp.asarray(c))
        if c.shape[0] == 1:
            c = jnp.tile(c, (num_samples, 1))
        c_n = self.normalize_context(c)

        if chunk_size is None:
            self.key, subkey = jr.split(self.key)
            z        = jr.normal(subkey, (num_samples, self.D))
            x_norm   = self.forward_fn.apply(self.params, z, c_n)
        else:
            chunks = []
            for start in range(0, num_samples, chunk_size):
                end = min(start + chunk_size, num_samples)
                self.key, subkey = jr.split(self.key)
                z       = jr.normal(subkey, (end - start, self.D))
                x_chunk = self.forward_fn.apply(
                    self.params, z, c_n[start:end])
                chunks.append(x_chunk)
            x_norm = jnp.concatenate(chunks, axis=0)

        return self.denormalize_data(x_norm)

    @partial(jax.jit, static_argnums=0)
    def forward_pass(self, z, c):
        """
        Latent z → physical x  (slow, sequential).

        Parameters
        ----------
        z : (batch, D)
        c : (batch, C)  physical-space context
        """
        c_n    = self.normalize_context(c)
        x_norm = self.forward_fn.apply(self.params, z, c_n)
        return self.denormalize_data(x_norm)

    @partial(jax.jit, static_argnums=0)
    def backward_pass(self, x, c):
        """
        Physical x → latent z  (fast, parallel).

        Parameters
        ----------
        x : (batch, D)  physical-space points
        c : (batch, C)  physical-space context
        """
        x_n = self.normalize_data(x)
        c_n = self.normalize_context(c)
        return self.inverse_fn.apply(self.params, x_n, c_n)

    @partial(jax.jit, static_argnums=0)
    def backward_pass_with_logdet(self, x, c):
        """
        Physical x → (z, log|det J|)  (fast, parallel).

        log|det J| here is log|det J_{x→z}| including the normalisation
        Jacobian, so:
            log p(x|c) = log p_z(z) + log|det J_{x→z}|
        """
        x_n = self.normalize_data(x)
        c_n = self.normalize_context(c)
        z, logdet_flow = self.inverse_and_logdet_fn.apply(
            self.params, x_n, c_n)
        logdet_norm = self._logdet_normalize_data(x)
        return z, logdet_flow + logdet_norm

    # ====================================================================
    # Save / load  (identical to ConditionalFlow)
    # ====================================================================

    def save_params(self):
        leaves, _ = jax.tree_util.tree_flatten(self.params)
        save_dict = {f"leaf_{i}": np.asarray(l)
                     for i, l in enumerate(leaves)}
        np.savez(self.flow_save_path, **save_dict)

    def load_params(self):
        data       = np.load(self.flow_load_path, allow_pickle=True)
        _, treedef = jax.tree_util.tree_flatten(self.params)
        leaves     = [jnp.asarray(data[f"leaf_{i}"])
                      for i in range(treedef.num_leaves)]
        self.params = jax.tree_util.tree_unflatten(treedef, leaves)