from ATLAS.utils import jagged2padded, jit_method
from ATLAS.utils import jit
from ATLAS.signals.signals_utils import _timing_model_svd, stabelize_TNT, stabelize_TDNTD

import numpy as np
import itertools
from tqdm import tqdm
from tqdm.auto import trange

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import jax.scipy.linalg as jsl
from functools import partial

import numpyro
import numpyro.distributions as dist

EPOCH_THRESHOLD = 1.0 # seconds

# Helper functions for finding TOA epochs --------------------------------------
def _get_psr_WN_helpers(psr, dt=1.0):
    """Get the helper arrays/matrices for computing the white noise covariance matrix

    This function computes several helper quantities for computing the N matrix
    and its inverse solutions for a given pulsar. These include:
    - backends: (N_backends) an ordered list of unique backend names for this pulsar
    - B: (N_toa) an array of backend indices for each toa (i.e. which backend each toa belongs to)
    - U_pad: (N_epoch, max_epoch_size) a padded array of toa indices for each epoch [padded with -1]
    - U_mask: (N_epoch, max_epoch_size) a boolean mask indicating which elements of U_pad are valid
    - V: (N_epoch) an array of backend indices for each epoch (i.e. which backend each epoch belongs to)

    The user supplies a pulsar object which must contain the following attributes:
    - toas: (N_toa) array of time of arrivals for this pulsar
    - backend_flags: (N_toa) array of backend names for each toa

    Finally, the user also supplies a time interval `dt` for grouping toas into epochs. 
    Toas are grouped into epochs such that all toas in an epoch are within a time 
    interval `dt` of the first toa in that epoch. `dt` is given in seconds, and the 
    default value is 1.0 second.


    Parameters
    ----------
    psr : object
        The pulsar object to compute the helper arrays/matrices for [check description]
    dt : float
        The time interval (in seconds) for grouping toas into epochs, by default 1.0

    Returns
    -------
    out : tuple
        A tuple containing the following helper arrays/matrices for computing the white noise covariance matrix:
        - backends: (N_backends) an ordered tuple of unique backend names for this pulsar
        - B: (N_toa) an array of backend indices for each toa (i.e. which backend each toa belongs to)
        - U_pad: (N_epoch, max_epoch_size) a padded array of toa indices for each epoch [padded with -1]
        - U_mask: (N_epoch, max_epoch_size) a boolean mask indicating which elements of U_pad are valid
        - V: (N_epoch) an array of backend indices for each epoch (i.e. which backend each epoch belongs to)
    """
    # The array of backends and an array which indicates which backend each toa belongs to
    backends, B = np.unique(psr.backend_flags, return_inverse=True)
    backends = tuple(backends) # Convert to tuple
    # The indices of toas for each backend (i.e. which toas belong to each backend)
    backend_idx = [np.where(B == i)[0] for i in range(len(backends))]

    U = [] # We don't know how many toas will be in each epoch!
    V = [] # We don't know how many epochs there will be!
    for i in range(len(backends)):
        # Get this backend's toas
        toas = psr.toas[backend_idx[i]]

        # Get the epochs for this backend (indices of toas in this backend)
        backend_epochs = _get_epochs(toas, dt)

        # Map backend indices to global indices through B[i]
        epochs = [backend_idx[i][e] for e in backend_epochs]

        U.extend(epochs) # Add these epochs to U
        V.extend([i]*len(epochs)) # Add the backend index for these epochs to V

    # U is a list of arrays of toa indices for each epoch
    # we need to convert this to a padded array for fast einsum computations
    U_pad, U_mask = jagged2padded(U)
    U_pad, U_mask = jnp.array(U_pad, dtype=int), jnp.array(U_mask, dtype=bool)

    # V is a list of backend indices for each epoch (i.e. which backend each epoch belongs to)
    V = jnp.array(V, dtype=int)
    # B is an array of backend indices for each toa (i.e. which backend each toa belongs to)
    B = jnp.array(B, dtype=int)

    out = (backends, B, U_pad, U_mask, V)
    return out


def _get_epochs(toas, dt):
    """Create a list of toa epochs, where each epoch is a list of toa indices

    This method groups toas indices into epochs such that all toas in an epoch 
    are within a time interval `dt` of adjacent TOAs. i.e. if the difference
    between adjacent toas is greater than or equal to `dt`, then they belong to 
    different epochs. This method does NOT take into account timing back ends.

    Parameters
    ----------
    toas : np.ndarray
        Array of time of arrivals (toas)
    dt : float
        Time interval for grouping toas into epochs

    Returns
    -------
    epoch_idx : list
        A list of numpy arrays, each containing the indices of toas in that epoch
    """
    isort = np.argsort(toas)
    sort_toas = toas[isort]

    # Find indices where adjacent toas are separated by >= dt
    breaks = np.where(np.diff(sort_toas) >= dt)[0] + 1 # (plus 1 since diff)
    epochs = np.split(isort, breaks)

    # Convert to sorted jax arrays and drop single-toa epochs.
    epoch_idx = [jnp.sort(jnp.array(epoch, dtype=int))
                 for epoch in epochs if len(epoch) > 1]

    return epoch_idx

class DiagSinglePulsarWhiteCov:
    """A simple TOA covariance matrix with only TOA errors. (no extra noise factors)

    This class implements a simplified white noise covariance matrix for a pulsar
    with a fixed pulsar timing model. This model only includes TOA errors and does 
    not include any EFAC, EQUAD, or ECORR parameters. This class is best used for 
    simulated or whitened data.
    
    The covariance matrix can be written as:
        N_ij = delta_ij * sigma_i^2
        where sigma_i is the TOA error for the i-th TOA.

    This class uses JAX pytrees which enable JIT compilation of functions with
    objects of this class as an input.

    Attributes
    ----------
    psr_name : str
        The name of the pulsar this covariance matrix corresponds to.
    ntoas : int
        The number of TOAs for this pulsar.
    toaerrs : array
        The TOA errors for this pulsar. [ntoas]
    nvec : array
        An array of the diagonal elements of the covariance matrix (the white noise variances). [ntoas]
    """

    def __init__(self, psr, marg = False):
        """The constructor for the simplified white noise covariance matrix.

        This constructor initializes the simplified white noise covariance matrix. It
        takes a pulsar object and extracts the TOA errors to construct the diagonal
        covariance matrix.

        Parameters
        ----------
        psr : object
            The pulsar object to construct the covariance matrix for.
        marg: bool:
            Do you want to marginalize over linear timing model errors?
        """
        # All attributes are static in the simple case
        # Static attributes ----------------------------------------------------
        self.psr_name = psr.name
        self.ntoas = len(psr.toas)
        self.toaerrs = psr.toaerrs
        self.marg = marg
        self.Mmat = _timing_model_svd(psr.Mmat)
        self.Mprior = self.Mmat.shape[1] * jnp.log(1e40)

        #jitted function that returns (left.T N right)
        self.solve = self._solve_func_maker(return_logdet = False)
        #unmarg solver to be used within marg solver function later
        self.solve_unmarg = partial(self._solve_unmarg, return_logdet = False) 
        #jitted function that returns (left.T N right, logDetN)
        self.solve_with_logdet = self._solve_func_maker(return_logdet = True)

    @jit_method
    def get_nvec_jvec(self):
        """Compute nvec.

        Returns
        -------
        tuple
            Tuple ``(nvec, jvec)`` where

            - ``nvec`` contains the diagonal white-noise variances
              for each TOA.
            - ``jvec`` is None.
        """
        nvec = self.toaerrs**2
        jvec = None
        return nvec, jvec

    def _solve_func_maker(self, return_logdet):
        """Solve the linear system left^T N^{-1} right.

        This method implements the solution to the linear system left^T N^{-1} right
        for the simplified covariance matrix, which is just a diagonal matrix with the
        TOA errors squared on the diagonal. This method can be JIT compiled in other functions!

        NOTE: the left matrix will be transposed for you and should be [N_toa, N]

        Parameters
        ----------
        white_noise_helpers: tuple
            This contaains (nvec, jvec). jvec is set to None
        left : array-like
            The left-hand side of the linear system. [N_toa, N]
        right : array-like
            The right-hand side of the linear system. [N_toa, M]
        
        Returns
        -------
        array-like
            The solution to the linear system. [N, M]
        """
        if self.marg:
            new_func = partial(self._solve_marg, return_logdet = return_logdet)
            return jit(new_func)
        else:
            new_func = partial(self._solve_unmarg, return_logdet = return_logdet)
            return jit(new_func)


    def _solve_unmarg(self, white_noise_helpers, left, right, return_logdet):
        """Solve the linear system left^T N^{-1} right.

        This method implements the solution to the linear system left^T N^{-1} right
        for the simplified covariance matrix, which is just a diagonal matrix with the
        TOA errors squared on the diagonal. This method can be JIT compiled in other functions!

        NOTE: the left matrix will be transposed for you and should be [N_toa, N]

        Parameters
        ----------
        white_noise_helpers: tuple
            This contaains (nvec, jvec). jvec is set to None
        left : array-like
            The left-hand side of the linear system. [N_toa, N]
        right : array-like
            The right-hand side of the linear system. [N_toa, M]
        
        Returns
        -------
        array-like
            The solution to the linear system. [N, M]
        """
        nvec = white_noise_helpers[0]
        # Only the diagonal! (L^T Ainv R)
        LNR = left.T @ (1/nvec[:,None] * right) # (N, M)

        if return_logdet:
            return LNR, self.logdet_unmarg(white_noise_helpers)
        else:
            return LNR


    def _solve_marg(self, white_noise_helpers, left, right, return_logdet):
        """Solve the linear system left^T D^{-1} right. 

        This method solves the linear equation (left).T @ D^{-1} @ right, where
        D is the timing-model marginalized covariance matrix.

        NOTE: the left matrix will be transposed for you and should be [N_toa, N]

        Parameters
        ----------
        white_noise_helpers: tuple
            (nvec, jvec) where jvec = None

        left : array-like
            The left-hand side of the equation. [Ntoa, n]
        right : array-like
            The right-hand side of the equation. [Ntoa, m]

        Returns
        -------
        array-like
            The result of the linear equation. [n, m] or [m]
        """
        # Solve L^T N^{-1} R - L^T N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1} R
        # Term1 = L^T N^{-1} R
        term1 = self.solve_unmarg(white_noise_helpers, left, right)

        # Term2 = L^T N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1} R
        MNM = self.solve_unmarg(white_noise_helpers, self.Mmat, self.Mmat)
        LNM = self.solve_unmarg(white_noise_helpers, left, self.Mmat)
        MNR = self.solve_unmarg(white_noise_helpers, self.Mmat, right)
            
        cf = jsl.cho_factor(MNM)
        term2 = LNM @ jsl.cho_solve(cf, MNR)

        if not return_logdet:
            return term1 - term2
        else:
            return term1 - term2, self.logdet_unmarg(white_noise_helpers)\
                                 + 2 * jnp.sum(jnp.log(cf[0].diagonal())) + self.Mprior

    @jit_method
    def logdet_unmarg(self, white_noise_helpers):
        """Compute the log-determinant of the white-noise covariance matrix.

        Parameters
        ----------
        helpers : tuple
            Tuple ``(nvec, jvec)`` containing the diagonal and ECORR
            covariance components.

        Returns
        -------
        float
            The log-determinant of the covariance matrix.
        """
        nvec, jvec = white_noise_helpers
        return jnp.sum(jnp.log(nvec))

class SinglePulsarWhiteCov:
    """A complete TOA covariance matrix with EFAC, EQUAD, and ECORR.

    This class implements the full white noise covariance matrix for a pulsar 
    with a fixed pulsar timing model. This model includes contributions from EFAC, 
    EQUAD, and ECORR parameters. 
    The covariance matrix can be written as: 
        N_ij = delta_ij EFAC_i^2 * (sigma_i^2 + EQUAD_i^2) + ECORR_ij^2
        - if i and j are within the same epoch, otherwise N_ij = 0

    This class uses the Sherman-Morrison formula to calculate the linear system
    left^T N^{-1} right efficiently without explicitly inverting the covariance matrix.

    This class uses JAX pytrees which enable JIT compilation of functions with
    objects of this class as an input.

    Attributes
    ----------
    psr_name : str
        The name of the pulsar this covariance matrix corresponds to.
    ntoas : int
        The number of TOAs for this pulsar.
    nepochs : int
        The number of epochs for this pulsar.
    toaerrs : array
        The TOA errors for this pulsar. [ntoas]
    backends : list
        A list of unique backend names for this pulsar.
    B : array
        An array of backend indices for each TOA. [ntoas]
    U_pad : array
        A padded array of TOA indices for each epoch. [nepochs, max_epoch_size]
    U_mask : array
        A boolean mask indicating which elements of U_pad are valid. [nepochs, max_epoch_size]
    V : array
        An array of epoch indices for each TOA. [ntoas]
    lower_prior_bounds: array
        lower prior bounds
    upper_prior_bounds: array
        upper prior bounds
    """

    def __init__(self, psr,
                marg = False, 
                efac_prior_bounds = (0.01, 10), # (low, high)
                efac_prior_normal = (1., 0.25), # (mean, std)
                log10equad_prior_bounds = (-9, -5), # (low, high)
                log10ecorr_prior_bounds = (-9, -5) # (low, high)
                ):
        """A constructor for the full white noise covariance matrix.

        Parameters
        ----------
        psr : object
            the pulsar object

        marg: bool
            Do you want to marginalize over linear timing model errors?

        *prior_bounds : tuple, optional
            the lower and upper prior bounds for white noise params
        """
        backends, B, U_pad, U_mask, V = _get_psr_WN_helpers(psr, dt=EPOCH_THRESHOLD)

        self.marg = marg

        self.Mmat = _timing_model_svd(psr.Mmat)
        self.Mprior = self.Mmat.shape[1] * jnp.log(1e40)

        # Static attributes ----------------------------------------------------
        self.psr_name = psr.name
        self.ntoas = len(psr.toas)
        self.nepochs = len(U_pad)
        self.toaerrs = psr.toaerrs

        self.backends = backends
        self.n_backends = len(backends)
        self.B = B # To construct nvec
        self.U_pad = U_pad
        self.U_mask = U_mask
        self.V = V # To construct jvec

        # Prior Ranges--------------------------------------
        self.lower_efac = efac_prior_bounds[0] 
        self.upper_efac = efac_prior_bounds[1]
        self.center_efac = efac_prior_normal[0]
        self.sigma_efac = efac_prior_normal[1]
        self.lower_log10equad = log10equad_prior_bounds[0]
        self.upper_log10equad = log10equad_prior_bounds[1]
        self.lower_log10ecorr = log10ecorr_prior_bounds[0]
        self.upper_log10ecorr = log10ecorr_prior_bounds[1]

        #jitted function that returns (left.T N right)
        self.solve = self._solve_func_maker(return_logdet = False)
        #unmarg solver to be used within marg solver function later
        self.solve_unmarg = partial(self._solve_unmarg, return_logdet = False) 
        #jitted function that returns (left.T N right, logDetN)
        self.solve_with_logdet = self._solve_func_maker(return_logdet = True)

    def params_dict_to_vector(self, params):
        """Extract white noise params from a dict into a flat JAX array.

        Call this ONCE outside the sampler to pay the Python dict-lookup
        and host→device transfer cost.

        Layout: [ef_0, ..., ef_nb, eq_0, ..., eq_nb, ec_0, ..., ec_nb]
        where nb = len(self.backends).

        Parameters
        ----------
        params : dict

        Returns
        -------
        wn_vec : jnp.ndarray [3 * nbackends]
        """
        names = self.get_param_names()
        v = jnp.array([params[n] for n in names])
        return v
    
    def make_numpyro_prior(self, uniform_efac = False):
        """Sample white-noise parameters using NumPyro priors.

        Generates one EFAC, log10_t2equad, and log10_ecorr parameter for
        each backend and returns them as a flat vector with layout

            [ef_0, ..., ef_nb | eq_0, ..., eq_nb | ec_0, ..., ec_nb]

        Parameters
        ----------
        uniform_efac : bool, optional
            If True, sample EFAC values from a uniform distribution
            between ``lower_efac`` and ``upper_efac``. Otherwise,
            sample from a truncated normal distribution centered on
            ``center_efac`` with standard deviation ``sigma_efac``.

        Returns
        -------
        jnp.ndarray
            White-noise parameter vector of shape
            ``[3 * n_backends]``.
        """
        psr  = self.psr_name
        ef, eq, ec = [], [], []

        if uniform_efac:
            efac_base_dist = dist.Uniform(self.lower_efac, self.upper_efac)
        else:
            efac_base_dist = dist.TruncatedNormal(loc=self.center_efac, 
                                scale=self.sigma_efac, low=self.lower_efac)
            
        for b in self.backends:   # same iteration order as params_dict_to_vector
            ef.append(numpyro.sample(f'{psr}_{b}_efac', efac_base_dist))
            eq.append(numpyro.sample(f'{psr}_{b}_log10_t2equad',
                                    dist.Uniform(self.lower_log10equad, self.upper_log10equad)))
            ec.append(numpyro.sample(f'{psr}_{b}_log10_ecorr',
                                    dist.Uniform(self.lower_log10ecorr, self.upper_log10ecorr)))
        # Concatenate in the same [ef | eq | ec] order that params_dict_to_vector uses
        return jnp.concatenate([jnp.stack(ef), jnp.stack(eq), jnp.stack(ec)])

    def get_prior_bounds(self):
        """Return lower and upper bounds for all white-noise parameters.

        Returns
        -------
        tuple[list, list]
            Two lists containing the lower and upper bounds,
            respectively, ordered as

                [EFACs | log10_t2equads | log10_ecorrs].
        """
        low  = (
            [self.lower_efac]       * len(self.backends) +
            [self.lower_log10equad] * len(self.backends) +
            [self.lower_log10ecorr] * len(self.backends)
        )
        high = (
            [self.upper_efac]       * len(self.backends) +
            [self.upper_log10equad] * len(self.backends) +
            [self.upper_log10ecorr] * len(self.backends)
        )
        return low, high

    def prior_draw(self, uniform_efac = True):
        """Draw white-noise parameters from the prior distribution.

        Generates one EFAC, log10_t2equad, and log10_ecorr parameter for
        each backend and returns them as a flat vector with layout

            [ef_0, ..., ef_nb | eq_0, ..., eq_nb | ec_0, ..., ec_nb]

        Parameters
        ----------
        uniform_efac : bool, optional
            If True, draw EFAC values from a uniform distribution
            between ``lower_efac`` and ``upper_efac``. Otherwise,
            draw from a normal distribution centered on
            ``center_efac`` with width ``sigma_efac`` and reflected
            about ``lower_efac`` to enforce positivity.

        Returns
        -------
        jnp.ndarray
            White-noise parameter vector of shape
            ``[3 * n_backends]``.
        """
        ef, eq, ec = [], [], []
        for b in self.backends:   # same iteration order as params_dict_to_vector
            if uniform_efac:
                ef.append(np.random.uniform(self.lower_efac, self.upper_efac))
            else:
                ef.append(self.lower_efac + 
                        np.abs(np.random.normal(loc = self.center_efac, scale = self.sigma_efac)))
            eq.append(np.random.uniform(self.lower_log10equad, self.upper_log10equad))
            ec.append(np.random.uniform(self.lower_log10ecorr, self.upper_log10ecorr))
        # Concatenate in the same [ef | eq | ec] order that params_dict_to_vector uses
        return jnp.concatenate([jnp.stack(ef), jnp.stack(eq), jnp.stack(ec)])

    def prior_draw_dictionary(self, 
                            param_values = np.array([False]), 
                            uniform_efac = False):
        """Return white-noise parameters as a dictionary.

        Creates a parameter dictionary whose keys match
        ``get_param_names()``. Values are taken from
        ``param_values`` if provided; otherwise they are drawn
        from the prior distribution.

        Parameters
        ----------
        param_values : array-like, optional
            Parameter values ordered as

                [EFACs | log10_t2equads | log10_ecorrs].

        uniform_efac : bool, optional
            Passed to ``prior_draw()`` when generating new
            samples.

        Returns
        -------
        dict
            Mapping from parameter names to parameter values.
        """
        names = self.get_param_names()
        if np.any(param_values):
            xs = [float(x) for x in param_values]
        else:
            xs = [float(x) for x in self.prior_draw(uniform_efac = uniform_efac)]
        return dict(zip(names, xs))

    def get_param_names(self):
        """Return the white-noise parameter names for this pulsar.

        Names are returned in the order

            [EFACs | log10_t2equads | log10_ecorrs]

        and correspond directly to the parameter-vector layout
        used by ``params_dict_to_vector()``.

        Returns
        -------
        list[str]
            White-noise parameter names.
        """
        # Params is a dictionary of parameter values for each backends
        ef = [f'{self.psr_name}_{b}_efac'           for b in self.backends]
        eq = [f'{self.psr_name}_{b}_log10_t2equad'   for b in self.backends]
        ec = [f'{self.psr_name}_{b}_log10_ecorr'     for b in self.backends]
        return ef + eq + ec

    @jit(static_argnums = 0)
    def get_nvec_jvec(self, wn_vec):
        """Compute nvec and jvec from a parameter vector.

        This is the sampler-hot-path replacement for dictionary-based
        parameter access. The computation uses only JAX operations and
        is fully JIT compatible.

        Parameters
        ----------
        wn_vec : jnp.ndarray
            White-noise parameter vector of shape
            ``[3 * n_backends]`` with layout

                [EFACs | log10_t2equads | log10_ecorrs].

        Returns
        -------
        tuple
            Tuple ``(nvec, jvec)`` where

            - ``nvec`` contains the diagonal white-noise variances
              for each TOA.
            - ``jvec`` contains the ECORR variance associated with
              each epoch.
        """
        nb = len(self.backends)
        ef = wn_vec[:nb]
        eq = wn_vec[nb:2*nb]
        ec = wn_vec[2*nb:]

        ef2 = ef ** 2
        eq2 = 10.0 ** (2.0 * eq)
        ec2 = 10.0 ** (2.0 * ec)

        nvec = ef2[self.B] * (self.toaerrs ** 2 + eq2[self.B])
        jvec = ec2[self.V]

        return nvec, jvec

    def _solve_func_maker(self, return_logdet):
        """Solve the linear system left^T N^{-1} right.

        This method implements the solution to the linear system left^T N^{-1} right
        for the simplified covariance matrix, which is just a diagonal matrix with the
        TOA errors squared on the diagonal. This method can be JIT compiled in other functions!

        NOTE: the left matrix will be transposed for you and should be [N_toa, N]

        Parameters
        ----------
        white_noise_helpers: tuple
            This contaains (nvec, jvec). jvec is set to None
        left : array-like
            The left-hand side of the linear system. [N_toa, N]
        right : array-like
            The right-hand side of the linear system. [N_toa, M]
        
        Returns
        -------
        array-like
            The solution to the linear system. [N, M]
        """
        if self.marg:
            new_func = partial(self._solve_marg, return_logdet = return_logdet)
            return jit(new_func)
        else:
            new_func = partial(self._solve_unmarg, return_logdet = return_logdet)
            return jit(new_func)

    def _solve_unmarg(self, white_noise_helpers, left, right, return_logdet):
        """Solve the linear system ``left^T N^{-1} right``.

        This method evaluates

            left^T N^{-1} right

        using the Sherman-Morrison formula and also returns the
        log-determinant of the covariance matrix.

        Parameters
        ----------
        white_noise_helpers : tuple
            Tuple ``(nvec, jvec)`` containing the diagonal and
            ECORR covariance components.
        left : array-like
            Left-hand matrix of shape ``[N_toa, N]``.
        right : array-like
            Right-hand matrix of shape ``[N_toa, M]``.

        Returns
        -------
        tuple
            A tuple ``(solve_result, logdet_N)`` containing the
            covariance-weighted matrix product and the
            log-determinant of the covariance matrix.
        """
        nvec, jvec = white_noise_helpers
        # Get the diagonal bit.
        Ainv = 1/nvec
        # The low-rank bit (L^T Ainv R)
        term1 = left.T @ (Ainv[:,None] * right) # (N, M)

        # b is the number of blocks (epochs)
        # LAinv is (b, N), RAinv is (b, M) (Mask is used to zero out the padded values)
        LAinv = jnp.einsum('biN,bi -> bN', left[self.U_pad,:], Ainv[self.U_pad]*self.U_mask) 
        RAinv = jnp.einsum('biM,bi -> bM', right[self.U_pad,:], Ainv[self.U_pad]*self.U_mask)

        # Numerator of sherman morrison update (b, N, M)
        num = jnp.einsum('ni,nj -> nij', LAinv, RAinv*jvec[:,None]) # (b, N, M)
        denom = 1.0 + jnp.einsum('n,nj -> n', jvec, Ainv[self.U_pad]*self.U_mask)  # (b)
        term2 = jnp.sum(num / denom[:,None,None], axis=0) # Sum over blocks (N, M)
        
        solve_result = term1 - term2

        if return_logdet:
            return solve_result, self.logdet_unmarg(white_noise_helpers)
        else:
            return solve_result
   
    def _solve_marg(self, white_noise_helpers, left, right, return_logdet):
        """Solve a linear equation (left).T @ D^{-1} @ right.

        This method solves the linear equation (left).T @ D^{-1} @ right, where
        D is the timing-model marginalized covariance matrix.

        NOTE: the left matrix will be transposed for you and should be [N_toa, N]

        Parameters
        ----------
        helpers : tuple
            Tuple ``(nvec, jvec)`` containing the diagonal and ECORR covariance components.
        Fmat : array [total_ntoas, N_basis]
        left : array-like
            The left-hand side of the equation. [Ntoa, n]
        right : array-like
            The right-hand side of the equation. [Ntoa, m]

        Returns
        -------
        array-like
            The result of the linear equation. [n, m] or [m]
        """
        # Solve L^T N^{-1} R - L^T N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1} R
        # Term1 = L^T N^{-1} R
        term1 = self.solve_unmarg(white_noise_helpers, left, right)

        # Term2 = L^T N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1} R
        MNM = self.solve_unmarg(white_noise_helpers, self.Mmat, self.Mmat)
        LNM = self.solve_unmarg(white_noise_helpers, left, self.Mmat)
        MNR = self.solve_unmarg(white_noise_helpers, self.Mmat, right)

        # The covariance matrix MNM can be ill-conditioned, we should stabilize it
        cf = jsl.cho_factor(MNM)
        term2 = LNM @ jsl.cho_solve(cf, MNR)

        if return_logdet:
            logdet = self.logdet_unmarg(white_noise_helpers)\
                 + 2 * jnp.sum(jnp.log(cf[0].diagonal())) + self.Mprior
            return term1 - term2, logdet
        else:
            return term1 - term2

    @jit_method
    def logdet_unmarg(self, white_noise_helpers):
        """Compute the log-determinant of the white-noise covariance matrix.

        Parameters
        ----------
        helpers : tuple
            Tuple ``(nvec, jvec)`` containing the diagonal and ECORR
            covariance components.

        Returns
        -------
        float
            The log-determinant of the covariance matrix.
        """
        # Diagonal contribution
        nvec, jvec = white_noise_helpers
        logdet_D = jnp.sum(jnp.log(nvec))

        # A^{-1}
        Ainv = 1.0 / nvec

        # For each epoch: sum_{i in epoch} Ainv_i
        epoch_sums = jnp.einsum(
            'bi,bi->b',
            Ainv[self.U_pad],
            self.U_mask
        )

        # Low-rank correction
        logdet_corr = jnp.sum(jnp.log(1.0 + jvec * epoch_sums))

        return logdet_D + logdet_corr

class WhiteCov:
    """The Multi-pulsar white noise covariance matrix handler (also works with one pulsar!).

    Exploits the block-diagonal structure of N across pulsars to efficiently
    compute T^T N^{-1} T and T^T N^{-1} r for all pulsars simultaneously.

    Because N is block-diagonal across pulsars, we have:
        T^T N^{-1} T = sum_p  T_p^T N_p^{-1} T_p
        T^T N^{-1} r = sum_p  T_p^T N_p^{-1} r_p
    where the _p subscript selects the TOA rows belonging to pulsar p.

    Within each per-pulsar block, the Sherman-Morrison formula from
    Fix_TM_TOA_cov_full is re-used, so ECORR is handled exactly.

    This class is a JAX pytree; all methods decorated with @jit can be
    JIT-compiled even when an instance of this class is an argument.

    Attributes
    ----------
    npulsars : int
        Number of pulsars.
    ntoas_per_psr : tuple of int
        Number of TOAs for each pulsar, in order.
    total_ntoas : int
        Total number of TOAs across all pulsars.
    toa_starts : tuple of int
        Start index in the global TOA array for each pulsar.
    toa_ends : tuple of int
        End index (exclusive) in the global TOA array for each pulsar.
    cov_matrices : list of white noise cov objects
        Per-pulsar covariance matrix objects (each itself a pytree).
    """

    def __init__(self, 
                data,
                stabelize_TNT = False,
                efac_prior_bounds = (0.01, 10), # (low, high)
                efac_prior_normal = (1., 0.25), # (mean, std)
                log10equad_prior_bounds = (-9, -5), # (low, high)
                log10ecorr_prior_bounds = (-9, -5) # (low, high)
                ):
        """Construct a multi-pulsar white noise covariance handler.

        Parameters
        ----------
        psrs : object
            the pulsar object

        data : object
            ATLAS data object

        diag_white_cov : bool
            do you want simple no backend diagonal white noise?
        """
        # Extracting the data analysis settings
        self.data = data
        self.diag_white_cov = self.data.diag_white_cov
        self.marg = self.data.marg
        self.npulsars = self.data.npsrs
        self.stabelize = stabelize_TNT

        self.cov_matrices = []
        pbar = trange(self.npulsars)
        for pidx in pbar:
            psr = self.data.psrs[pidx]
            pbar.set_description(f"Construncting the white noise cov matrix for {psr.name}")
            if not self.diag_white_cov:
                self.cov_matrices.append(SinglePulsarWhiteCov(
                                                            psr, 
                                                            marg = self.marg, 
                                                            efac_prior_bounds = efac_prior_bounds,
                                                            efac_prior_normal = efac_prior_normal,
                                                            log10equad_prior_bounds = log10equad_prior_bounds,
                                                            log10ecorr_prior_bounds = log10ecorr_prior_bounds
                                                            ))
            else:
                self.cov_matrices.append(DiagSinglePulsarWhiteCov(psr, 
                                                            marg = self.marg
                                                            ))

        self.ntoas_per_psr = tuple(len(psr.toas) for psr in self.data.psrs)
        self.total_ntoas = sum(self.ntoas_per_psr)
        # Precompute static slice boundaries for each pulsar in the global array.
        cumulative = np.cumsum([0] + list(self.ntoas_per_psr))
        self.toa_starts = tuple(int(c) for c in cumulative[:-1])
        self.toa_ends   = tuple(int(c) for c in cumulative[1:])
        self.pulsar_idxs = jnp.arange(self.npulsars)

        self.data.add_white_noise_cov(self)

    # ------------------------------------------------------------------
    # Parameter interface
    # ------------------------------------------------------------------

    def get_prior_bounds(self):
        """Concatenate per-pulsar prior bounds.

        Returns
        -------
        concatenated prior bounds across pulsars and backends.

        """
        return jnp.concat([jnp.array(cov.get_prior_bounds()) for cov in self.cov_matrices], axis = -1)

    def params_dict_to_vector(self, params):
        """Concatenate per-pulsar wn_vecs into one flat array.

        Returns
        -------
        wn_vec : jnp.ndarray [npulsars * 3 * max_nbackends]
            Concatenation of each per-pulsar params_dict_to_vector output.
        """
        return jnp.concatenate([cov.params_dict_to_vector(params)
                                for cov in self.cov_matrices])

    def sample_numpyro_multi_psr_wn_vec(self, uniform_efac = False):
        """Sample white noise for all pulsars and return a concatenated wn_vec.

        The layout mirrors MultiPsr_WN_Cov.params_dict_to_vector:
            [wn_vec_psr0 | wn_vec_psr1 | ...]
        where each per-pulsar block is [ef | eq | ec] for that pulsar's backends.

        Parameters
        ----------
        multi_cov : MultiPsr_WN_Cov

        Returns
        -------
        wn_vec : jnp.ndarray [sum_p(3 * nbackends_p)]
        """
        return jnp.concatenate([cov.make_numpyro_prior(uniform_efac = uniform_efac) for cov in self.cov_matrices])

    def prior_draw(self, uniform_efac = False):
        """Sample white noise for all pulsars and return a concatenated wn_vec.

        The layout mirrors MultiPsr_WN_Cov.params_dict_to_vector:
            [wn_vec_psr0 | wn_vec_psr1 | ...]
        where each per-pulsar block is [ef | eq | ec] for that pulsar's backends.

        Parameters
        ----------
        multi_cov : MultiPsr_WN_Cov

        Returns
        -------
        wn_vec : jnp.ndarray [sum_p(3 * nbackends_p)]
        """
        return jnp.concatenate([cov.prior_draw(uniform_efac = uniform_efac) for cov in self.cov_matrices])

    def prior_draw_dictionary(self, param_values = None, uniform_efac = False):
        """Sample white noise for all pulsars and return a concatenated wn_vec.

        The layout mirrors MultiPsr_WN_Cov.params_dict_to_vector:
            [wn_vec_psr0 | wn_vec_psr1 | ...]
        where each per-pulsar block is [ef | eq | ec] for that pulsar's backends.

        Parameters
        ----------
        multi_cov : MultiPsr_WN_Cov

        Returns
        -------
        wn_vec : jnp.ndarray [sum_p(3 * nbackends_p)]
        """
        names = self.get_param_names()
        if np.any(param_values):
            xs = [float(x) for x in param_values]
        else:
            xs = [float(x) for x in self.prior_draw(uniform_efac = uniform_efac)]
        return dict(zip(names, xs))

    def get_param_names(self):
        """Sample white noise for all pulsars and return a concatenated wn_vec.

        The layout mirrors MultiPsr_WN_Cov.params_dict_to_vector:
            [wn_vec_psr0 | wn_vec_psr1 | ...]
        where each per-pulsar block is [ef | eq | ec] for that pulsar's backends.

        Parameters
        ----------
        multi_cov : MultiPsr_WN_Cov

        Returns
        -------
        wn_vec : jnp.ndarray [sum_p(3 * nbackends_p)]
        """
        return list(itertools.chain.from_iterable([cov.get_param_names() for cov in self.cov_matrices]))

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def get_red_helpers(self, red_noise_basis, residuals, white_noise_params):
        """Compute both F_p^T N_p^{-1} F_p and F_p^T N_p^{-1} r_p per pulsar.
        NOTE: The F-matrix can be replaced with a T-matrix. F and T are interchangable.

        Appends r_p as an extra column of the right-hand side so only one
        call to solve() is needed per pulsar:
            solve(F_p, [F_p | r_p]) = [F_p^T N_p^{-1} F_p | F_p^T N_p^{-1} r_p]

        Parameters
        ----------
        red_noise_basis : array [total_ntoas, N_basis]
            Global Fourier design matrix (F-matrix).
        residuals : array [total_ntoas]
            Global residuals vector.
        white_noise_params: array
            The Global (concatenated) white noise values
        Returns
        -------
        FNF : array [npulsars, N_basis, N_basis]
        FNr : array [npulsars, N_basis]
        rNr: array [1]
        log_det_N: array [1]
        """
        N_basis = red_noise_basis.shape[1]
        FNF = jnp.zeros((self.npulsars, N_basis, N_basis))
        FNr = jnp.zeros((self.npulsars, N_basis))

        log_det_N = 0
        rNr = 0
        wn_params_start_idx = 0

        if self.diag_white_cov:
            for pidx, cov, start, end in zip(self.pulsar_idxs, 
                                            self.cov_matrices, 
                                            self.toa_starts, 
                                            self.toa_ends):

                white_noise_helper = cov.get_nvec_jvec()
                F_p  = red_noise_basis[start:end, :]                    # [n_p, N_basis]
                r_p  = residuals[start:end]                             # [n_p]
                # One solve per pulsar: right = [T_p | r_p]
                Fr_p = jnp.concatenate([F_p, r_p[:, None]], axis=1)     # [n_p, N_basis+1]
                res  = cov.solve(white_noise_helper, F_p, Fr_p)         # [N_basis, N_basis+1]

                FNF = FNF.at[pidx].set(res[:, :N_basis])
                FNr = FNr.at[pidx].set(res[:, N_basis])

                x, y = cov.solve_with_logdet(white_noise_helper, r_p[:, None], r_p[:, None])
                rNr += x
                log_det_N += y

            return FNF, FNr, rNr[0, 0], log_det_N

        else:
            for pidx, cov, start, end in zip(self.pulsar_idxs, 
                                            self.cov_matrices, 
                                            self.toa_starts, 
                                            self.toa_ends):

                wn_params_end_idx = wn_params_start_idx + 3 * cov.n_backends
                wn_params = white_noise_params[wn_params_start_idx: wn_params_end_idx]
                wn_params_start_idx = wn_params_end_idx

                white_noise_helper = cov.get_nvec_jvec(wn_params)
                F_p  = red_noise_basis[start:end, :]                    # [n_p, N_basis]
                r_p  = residuals[start:end]                             # [n_p]
                # One solve per pulsar: right = [T_p | r_p]
                Fr_p = jnp.concatenate([F_p, r_p[:, None]], axis=1)     # [n_p, N_basis+1]
                res  = cov.solve(white_noise_helper, F_p, Fr_p)         # [N_basis, N_basis+1]

                FNF = FNF.at[pidx].set(res[:, :N_basis])
                FNr = FNr.at[pidx].set(res[:, N_basis])

                x, y = cov.solve_with_logdet(white_noise_helper, r_p[:, None], r_p[:, None])
                rNr += x
                log_det_N += y

            if self.stabelize:
                return stabelize_TNT(FNF, FNF.shape[-1]), FNr, rNr[0, 0], log_det_N
            else:
                return FNF, FNr, rNr[0, 0], log_det_N

    def get_red_det_helpers(self, red_noise_basis, det_signal_basis,
                            residuals, white_noise_params):
        """Compute the red-noise *and* deterministic-signal N^{-1} helpers per pulsar.

        This extends :meth:`get_red_helpers` with a second (independently sized)
        design matrix ``det_signal_basis`` (denoted "TD") for a deterministic
        signal. In addition to the usual red-noise helpers it returns every
        N^{-1}-weighted cross product between the red-noise basis ``T``, the
        deterministic basis ``TD``, and the residuals ``r``.

        Parameters
        ----------
        red_noise_basis : array [total_ntoas, N_basis]
            Global red-noise (Fourier / T-matrix) design matrix.
        det_signal_basis : array [total_ntoas, N_det_basis]
            Global deterministic-signal design matrix. May have a different
            number of columns than ``red_noise_basis``.
        residuals : array [total_ntoas]
            Global residuals vector.
        white_noise_params : array
            The global (concatenated) white noise values.

        Returns
        -------
        TNT : array [npulsars, N_basis, N_basis]
            T^T N^{-1} T for each pulsar.
        TNr : array [npulsars, N_basis]
            T^T N^{-1} r for each pulsar.
        rNr : array [1]
            r^T N^{-1} r summed over pulsars.
        logdet_N : array [1]
            log|N| summed over pulsars.
        TDNTD : array [npulsars, N_det_basis, N_det_basis]
            TD^T N^{-1} TD for each pulsar.
        TDNr : array [npulsars, N_det_basis]
            TD^T N^{-1} r for each pulsar.
        TNTD : array [npulsars, N_basis, N_det_basis]
            T^T N^{-1} TD for each pulsar.
        """
        N_basis = red_noise_basis.shape[1]
        N_det_basis = det_signal_basis.shape[1]
        Ntot = N_basis + N_det_basis

        TNT   = jnp.zeros((self.npulsars, N_basis, N_basis))
        TNr   = jnp.zeros((self.npulsars, N_basis))
        TDNTD = jnp.zeros((self.npulsars, N_det_basis, N_det_basis))
        TDNr  = jnp.zeros((self.npulsars, N_det_basis))
        TNTD  = jnp.zeros((self.npulsars, N_basis, N_det_basis))

        logdet_N = 0
        rNr = 0
        wn_params_start_idx = 0

        for pidx, cov, start, end in zip(self.pulsar_idxs,
                                         self.cov_matrices,
                                         self.toa_starts,
                                         self.toa_ends):

            if self.diag_white_cov:
                white_noise_helper = cov.get_nvec_jvec()
            else:
                wn_params_end_idx = wn_params_start_idx + 3 * cov.n_backends
                wn_params = white_noise_params[wn_params_start_idx: wn_params_end_idx]
                wn_params_start_idx = wn_params_end_idx
                white_noise_helper = cov.get_nvec_jvec(wn_params)

            T_p  = red_noise_basis[start:end, :]                   # [n_p, N_basis]
            TD_p = det_signal_basis[start:end, :]                  # [n_p, N_det_basis]
            r_p  = residuals[start:end]                            # [n_p]

            # Stack once: M_p = [T_p | TD_p | r_p]  -> [n_p, Ntot + 1]
            M_p = jnp.concatenate([T_p, TD_p, r_p[:, None]], axis=1)

            # One solve per pulsar gives the full symmetric Gram matrix + logdet.
            # G = M_p^T N_p^{-1} M_p  -> [Ntot + 1, Ntot + 1]
            G, y = cov.solve_with_logdet(white_noise_helper, M_p, M_p)

            TNT   = TNT.at[pidx].set(G[:N_basis, :N_basis])
            TNTD  = TNTD.at[pidx].set(G[:N_basis, N_basis:Ntot])
            TDNTD = TDNTD.at[pidx].set(G[N_basis:Ntot, N_basis:Ntot])
            TNr   = TNr.at[pidx].set(G[:N_basis, Ntot])
            TDNr  = TDNr.at[pidx].set(G[N_basis:Ntot, Ntot])

            rNr += G[Ntot, Ntot]
            logdet_N += y
        if self.stabelize:
            return stabelize_TNT(TNT, TNT.shape[-1]), 
            TNr, rNr, logdet_N, stabelize_TDNTD(TDNTD, TDNTD.shape[-1]), 
            TDNr, TNTD
        else:
            return TNT, TNr, rNr, logdet_N, TDNTD, TDNr, TNTD