import numpy as np
from functools import cached_property, partial
import warnings, inspect, jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from ATLAS.utils import jit_method

fref = 1 / (1 * 365.25 * 24 * 60 * 60)

# ---------------------------------------------------------------------------
# Helper to parse a PSD function's helper-dictionary the same way it is done
# for the GWB PSD.  Returns (param_value_container, varied_param_indxs).
# ---------------------------------------------------------------------------
def _parse_psd_func(psd_func, helper_dict, n_bins, free_spec_sentinel="halflog10_rho"):
    """
    Inspect `psd_func` and build the parameter-container / varied-index arrays
    used to evaluate the PSD inside JAX-JIT code.

    The function signature is assumed to start with (f, df, *params).  The
    sentinel `free_spec_sentinel` marks the free-spectral (bin-by-bin) model
    whose parameter count equals `n_bins` rather than the number of named
    keyword arguments.

    Parameters
    ----------
    psd_func : callable
        The PSD function, e.g. from a GWBFunctions-like class.
    helper_dict : dict
        A dictionary that mirrors the structure of `gwb_helper_dictionary` but
        for this particular PSD.  Expected keys (when applicable):
          * "psd_param_upper_lim"   – upper prior bounds for varied params
          * "psd_param_lower_lim"   – lower prior bounds for varied params
          * "ordered_psd_model_params" – param name array for ordering check
          * "fixed_psd_param_indices"  – indices of fixed params (optional)
          * "fixed_psd_param_values"   – values of those fixed params (optional)
    n_bins : int
        Number of frequency bins (used only for the free-spectral model).
    free_spec_sentinel : str
        Name in the function signature that marks the free-spectral model.

    Returns
    -------
    param_value_container : jnp.ndarray
        Zero-filled container; fixed slots are pre-filled.
    varied_param_indxs : jnp.ndarray[int]
        Indices of the parameters that are *not* fixed (i.e. sampled).
    """
    sigs = np.array(
        [str(p) for p in inspect.signature(psd_func).parameters
         if "args" not in str(p)][2:]  # skip leading `f` and `df`
    )

    if free_spec_sentinel in sigs:
        container = jnp.zeros(n_bins)
        varied = jnp.arange(n_bins, dtype=int)
        return container, varied

    container = jnp.zeros(len(sigs))
    if "fixed_psd_param_indices" in helper_dict:
        fixed_idx = helper_dict["fixed_psd_param_indices"]
        fixed_val = helper_dict["fixed_psd_param_values"]
        container = container.at[fixed_idx].set(fixed_val)
        varied = jnp.array(
            [i for i in range(len(sigs)) if i not in fixed_idx], dtype=int
        )
    else:
        varied = jnp.arange(len(sigs), dtype=int)

    # Ordering safety-check
    if "ordered_psd_model_params" in helper_dict:
        supplied = helper_dict["ordered_psd_model_params"]
        assert np.all(supplied[varied] == sigs[varied]), (
            f"PSD param ordering mismatch for {psd_func.__name__}. "
            f"Signature expects {sigs[varied]}, got {supplied[varied]}."
        )

    return container, varied


def _parse_orf_func(orf_func, helper_dict):
    """
    Inspect `orf_func` and decide whether it is fixed (no free parameters
    beyond the angular separation) or has free parameters to be sampled.

    The function signature is assumed to start with (angle, *params).

    Returns
    -------
    orf_fixed : bool
    orf_signs : np.ndarray of str   (parameter names after `angle`)
    """
    orf_signs = np.array(
        [str(p) for p in inspect.signature(orf_func).parameters
         if "args" not in str(p)][1:]  # skip leading `angle`
    )

    if "ordered_orf_model_params" in helper_dict:
        supplied = helper_dict["ordered_orf_model_params"]
        assert np.all(supplied == orf_signs), (
            f"ORF param ordering mismatch.  Signature expects {orf_signs}, "
            f"got {supplied}."
        )
        return False, orf_signs

    return True, orf_signs


# ---------------------------------------------------------------------------
# Unified red-noise covariance-matrix class
# ---------------------------------------------------------------------------

class PerPulsarRedNoise:
    """
    A unified class for constructing the red-noise covariance (phi) matrix
    for pulsar timing array (PTA) analyses.

    This class generalises the original ``PowerLawRedNoise``,
    ``FreeSpecRedNoise``, ``DMRedNoise``, and ``ParameterizedGwb`` classes
    into a single entry-point.  The non-GWB (intrinsic red noise, DM, …)
    spectral components are described by callable PSD functions in exactly
    the same way as the GWB PSD, together with companion helper dictionaries
    that carry prior bounds and optional fixed-parameter information.

    Parameter vector layout
    -----------------------
    The flat ``xs`` vector passed to ``get_phi_mat`` / ``get_phi_diag`` is
    ordered as::

        xs = [ irn_psd_params (num_IR_params),
               dm_psd_params  (num_DM_params),   ← only if irn_psd_func is not None
               gwb_psd_params (n_gwb_varied),
               orf_params     (n_orf_varied) ]    ← only if ORF has free params

    where:

    * ``irn_psd_params`` is a flat array of shape
      ``(n_irn_varied_per_pulsar * Npulsars,)`` with pulsar index varying
      *slowest* (i.e. all params for pulsar 0 first, then pulsar 1, etc.).
    * ``dm_psd_params``  follows the same layout for the DM component.
    * ``gwb_psd_params`` are the varied GWB PSD parameters.
    * ``orf_params``     are the varied ORF parameters (absent when the ORF is
      fixed, i.e. no free parameters).

    IRN / DM PSD function convention
    ---------------------------------
    Both ``irn_psd_func`` and ``dm_psd_func`` must have the signature::

        psd_func(f, df, *params) -> jnp.ndarray of shape (n_bins,)

    where ``params`` is the parameter *tuple for a single pulsar*.  The
    function is called once per pulsar inside a ``vmap``.

    If ``irn_psd_func`` is ``None`` (the default) the class reduces to a
    GWB-only model (equivalent to the original ``ParameterizedGwb``).

    GWB / ORF function convention
    ------------------------------
    ``gwb_psd_func(f, df, *params) -> jnp.ndarray of shape (crn_bins,)``
    ``orf_func(angle, *params)     -> jnp.ndarray of shape (n_pairs,)``

    Helper dictionary keys
    ----------------------
    ``gwb_helper_dictionary`` (same as before):
        * ``"gwb_psd_param_upper_lim"``
        * ``"gwb_psd_param_lower_lim"``
        * ``"ordered_gwb_psd_model_params"``  (optional, for ordering check)
        * ``"fixed_gwb_psd_param_indices"``   (optional)
        * ``"fixed_gwb_psd_param_values"``    (optional)
        * ``"ordered_orf_model_params"``       (optional, marks free ORF)

    ``irn_helper_dictionary`` / ``dm_helper_dictionary`` (same structure,
    but keys prefixed with ``"psd_"`` instead of ``"gwb_psd_"`` / ``"gwb_"``):
        * ``"psd_param_upper_lim"``
        * ``"psd_param_lower_lim"``
        * ``"ordered_psd_model_params"``  (optional)
        * ``"fixed_psd_param_indices"``   (optional)
        * ``"fixed_psd_param_values"``    (optional)

    Authors
    -------
    Nima Laal (original pandora classes, 02/12/2025)
    Unified refactor: Ge (06/2025)
    """

    def __init__(
        self,
        signal_indices,          # from build_basis
        linear_timing_model_size,
        irn_psd_func=None,
        irn_helper_dictionary=None,
        irn_bins=None,
        f_irn=None,
        # ---- DM noise (optional) ----
        dm_psd_func=None,
        dm_helper_dictionary=None,
        dm_bins=None,
        f_dm=None,
        # ---- shared ----
        renorm_const=1.0,
        Npulsars=1,
    ):
        ...
        # ------------------------------------------------------------------ #
        #  Frequency index arrays from signal_indices                          #
        # ------------------------------------------------------------------ #
        tm_offset = linear_timing_model_size

        def _fourier_slice(name):
            sl = signal_indices[name]
            return slice((sl.start - tm_offset) // 2,
                        (sl.stop  - tm_offset) // 2)

        # ------------------------------------------------------------------ #
        #  Basic bookkeeping                                                   #
        # ------------------------------------------------------------------ #
        self.irn_psd_func = irn_psd_func
        self.dm_psd_func = dm_psd_func
        self.renorm_const = renorm_const
        self.logrenorm_offset = 0.5 * jnp.log10(renorm_const)
        
        self.Npulsars = Npulsars

        has_irn = irn_psd_func is not None
        self.has_irn = has_irn
        has_dm = dm_psd_func is not None
        self.has_dm = has_dm
        if self.has_dm is None and self.has_irn is None:
            raise ValueError("Either `irn` or 'dm' needs to be supplied." )

        if has_irn:
            self.IRN_slice = _fourier_slice('unc')

        if has_dm:
            self.DM_slice = _fourier_slice('dm')

        # n_total_bins = stop of the last occupied slice
        all_stops = []
        if has_irn: all_stops.append(self.IRN_slice.stop)
        if has_dm:  all_stops.append(self.DM_slice.stop)
        self.n_total_bins = max(all_stops)

        # ------------------------------------------------------------------ #
        #  IRN bookkeeping                                                     #
        # ------------------------------------------------------------------ #
        if has_irn:
            assert irn_bins is not None and f_irn is not None, (
                "irn_bins and f_irn must be supplied when irn_psd_func is given."
            )
            assert irn_helper_dictionary is not None, (
                "irn_helper_dictionary must be supplied when irn_psd_func is given."
            )
            self.irn_bins = irn_bins
            self.f_irn = f_irn if f_irn.ndim == 2 else jnp.broadcast_to(f_irn, (self.Npulsars, self.irn_bins)) 
            self.df_irn = jnp.diff(jnp.concatenate((jnp.zeros((self.Npulsars, 1)), self.f_irn), axis = 1))

        # ------------------------------------------------------------------ #
        #  DM bookkeeping                                                      #
        # ------------------------------------------------------------------ #
        has_dm = dm_psd_func is not None
        self.has_dm = has_dm
        if has_dm:
            assert dm_bins is not None and f_dm is not None, (
                "dm_bins and f_dm must be supplied when dm_psd_func is given."
            )
            assert dm_helper_dictionary is not None, (
                "dm_helper_dictionary must be supplied when dm_psd_func is given."
            )
            self.dm_bins = dm_bins
            self.f_dm = f_dm if f_dm.ndim == 2 else jnp.broadcast_to(f_dm, (self.Npulsars, self.dm_bins))
            self.df_dm = jnp.diff(jnp.concatenate((jnp.zeros((self.Npulsars, 1)), self.f_dm), axis = 1))
            
        # ------------------------------------------------------------------ #
        #  Parse IRN PSD                                                       #
        # ------------------------------------------------------------------ #
        if has_irn:
            self.irn_param_container, self.irn_varied_indxs = _parse_psd_func(
                irn_psd_func, irn_helper_dictionary, irn_bins
            )
            # Number of varied params per pulsar
            self.n_irn_varied = int(len(self.irn_varied_indxs))
            self.num_IR_params = self.n_irn_varied * self.Npulsars
        else:
            self.num_IR_params = 0

        # ------------------------------------------------------------------ #
        #  Parse DM PSD                                                        #
        # ------------------------------------------------------------------ #
        if has_dm:
            self.dm_param_container, self.dm_varied_indxs = _parse_psd_func(
                dm_psd_func, dm_helper_dictionary, dm_bins
            )
            self.n_dm_varied = int(len(self.dm_varied_indxs))
            self.num_DM_params = self.n_dm_varied * self.Npulsars
        else:
            self.num_DM_params = 0

        # ------------------------------------------------------------------ #
        #  Parameter-vector slice indices                                      #
        # ------------------------------------------------------------------ #
        # Layout: [ IRN(0..num_IR_params) | DM(..+num_DM_params) |
        self.irn_end_idx = self.num_IR_params
        self.dm_end_idx = self.irn_end_idx + self.num_DM_params

        # ------------------------------------------------------------------ #
        #  Prior bounds                                                        #
        # ------------------------------------------------------------------ #
        upper, lower = jnp.array([]), jnp.array([])

        if has_irn:
            irn_upper = jnp.tile(
                irn_helper_dictionary["psd_param_upper_lim"] + self.logrenorm_offset,
                self.Npulsars
            )
            irn_lower = jnp.tile(
                irn_helper_dictionary["psd_param_lower_lim"] + self.logrenorm_offset,
                self.Npulsars
            )
            upper = jnp.concatenate([upper, irn_upper])
            lower = jnp.concatenate([lower, irn_lower])

        if has_dm:
            dm_upper = jnp.tile(
                dm_helper_dictionary["psd_param_upper_lim"] + self.logrenorm_offset,
                self.Npulsars
            )
            dm_lower = jnp.tile(
                dm_helper_dictionary["psd_param_lower_lim"] + self.logrenorm_offset,
                self.Npulsars
            )
            upper = jnp.concatenate([upper, dm_upper])
            lower = jnp.concatenate([lower, dm_lower])

        self.upper_prior_lim_all = upper
        self.lower_prior_lim_all = lower

        if renorm_const != 1:
            warnings.warn(
                "You have chosen to change units.  Make sure your amplitude "
                "priors reflect that!"
            )

    # ---------------------------------------------------------------------- #
    #  Internal PSD helpers                                                   #
    # ---------------------------------------------------------------------- #

    @jit_method
    def _eval_irn_psd_all(self, irn_params_flat):
        """
        Evaluate the IRN PSD for *all* pulsars.

        Parameters
        ----------
        irn_params_flat : jnp.ndarray, shape (num_IR_params,)
            Flat array; reshaped to (Npulsars, n_irn_varied) before vmapping.

        Returns
        -------
        jnp.ndarray, shape (irn_bins, Npulsars)
        """
        per_psr = irn_params_flat.reshape(self.Npulsars, self.n_irn_varied)

        def _single(params, freqs_, df_):
            filled = self.irn_param_container.at[self.irn_varied_indxs].set(params)
            return self.irn_psd_func(freqs_, df_, *filled)

        return jax.vmap(_single)(per_psr, self.f_irn, self.df_irn).T  # → (irn_bins, Npulsars)

    @jit_method
    def _eval_dm_psd_all(self, dm_params_flat):
        """
        Evaluate the DM PSD for *all* pulsars.

        Returns
        -------
        jnp.ndarray, shape (dm_bins, Npulsars)
        """
        per_psr = dm_params_flat.reshape(self.Npulsars, self.n_dm_varied)

        def _single(params, freqs_, df_):
            filled = self.dm_param_container.at[self.dm_varied_indxs].set(params)
            return self.dm_psd_func(freqs_, df_, *filled)

        return jax.vmap(_single)(per_psr, self.f_dm, self.df_dm).T  # → (dm_bins, Npulsars)

    # ---------------------------------------------------------------------- #
    #  Parameter unpacking                                                    #
    # ---------------------------------------------------------------------- #

    @jit_method
    def _unpack(self, xs):
        """Return (irn_flat, dm_flat) from ``xs``."""
        irn = xs[:self.irn_end_idx]
        dm  = xs[self.irn_end_idx:self.dm_end_idx]
        return irn, dm

    # ---------------------------------------------------------------------- #
    #  Core phi builders                                                      #
    # ---------------------------------------------------------------------- #

    @jit_method
    def get_phi_diag(self, xs):
        irn_flat, dm_flat = self._unpack(xs)

        phi_diag = jnp.zeros((self.n_total_bins, self.Npulsars))

        if self.has_irn:
            phi_diag = phi_diag.at[self.IRN_slice].add(
                self._eval_irn_psd_all(irn_flat)
            )

        if self.has_dm:
            phi_diag = phi_diag.at[self.DM_slice].add(
                self._eval_dm_psd_all(dm_flat)
            )

        return phi_diag

    @jit_method
    def get_phi_mat(self, xs):
        """
        Build the full phi-matrix  (n_total_bins, Npulsars, Npulsars).

        Off-diagonal (cross-pulsar) elements are filled only in the GWB bins,
        weighted by the ORF.

        Parameters
        ----------
        xs : jnp.ndarray
            Flat parameter vector (see class docstring for layout).

        Returns
        -------
        phi : jnp.ndarray  (n_total_bins, Npulsars, Npulsars)
        """
        return self.get_phi_diag(xs)

    @jit_method
    def get_phi_mat_full(self, xs):
        """
        Like ``get_phi_mat`` but also fills the *upper* triangle so the matrix
        is explicitly symmetric.
        """
        return self.get_phi_diag(xs)

    @jit_method
    def get_phi_mat_CURN(self, xs):
        """
        Return the phi diagonal (CURN = Common Uncorrelated Red Noise):
        cross-pulsar correlations are ignored even if an ORF was supplied.

        Returns
        -------
        phi_diag : jnp.ndarray  (n_total_bins, Npulsars)
        psd_common : jnp.ndarray  (crn_bins,)
        """
        return self.get_phi_diag(xs)

    # ---------------------------------------------------------------------- #
    #  Inversion                                                              #
    # ---------------------------------------------------------------------- #

    @jit_method
    def get_phi_mat_inv(self, phi):
        """
        Invert the phi-matrix using mixed Cholesky + diagonal strategies.

        GWB-containing bins use Cholesky factorisation; purely-IRN bins and
        DM bins (which are diagonal) use direct reciprocal inversion.

        Parameters
        ----------
        phi : jnp.ndarray  (n_total_bins, Npulsars)

        Returns
        -------
        phiinv : jnp.ndarray  (2*n_total_bins, Npulsars)
            Repeated twice along axis-0 (one for each quadrature component).
        logdet_phi : float
        """
        phiinv = jnp.repeat(1/phi, 2, axis=0)
        log_det_phi = 2.0 * jnp.sum(jnp.log(phi))
        return phiinv[..., None] * jnp.eye(phiinv.shape[-1]) , log_det_phi 

    # ---------------------------------------------------------------------- #
    #  Prior                                                                  #
    # ---------------------------------------------------------------------- #

    @jit_method
    def get_lnprior(self, xs):
        """Uniform log-prior: returns a small constant if in bounds, -inf otherwise."""
        in_bounds = jnp.logical_and(
            xs > self.lower_prior_lim_all, xs < self.upper_prior_lim_all
        ).all()
        return jax.lax.cond(in_bounds, self._spit_neg_number, self._spit_neg_infinity)

    def get_lnprior_numpy(self, xs):
        return self.get_lnprior(xs).__array__()

    @jit_method
    def make_initial_guess(self, key):
        """Draw a uniform initial sample within the prior bounds."""
        return jr.uniform(
            key,
            shape=(self.upper_prior_lim_all.shape[0],),
            minval=self.lower_prior_lim_all,
            maxval=self.upper_prior_lim_all,
        )

    def test(self, key):
        """Draw a uniform initial sample within the prior bounds."""
        return jr.uniform(
            key,
            shape=(self.upper_prior_lim_all.shape[0],),
            minval=self.lower_prior_lim_all,
            maxval=self.upper_prior_lim_all,
        )

    # ---------------------------------------------------------------------- #
    #  Utilities                                                              #
    # ---------------------------------------------------------------------- #

    def jax_to_numpy_CPU(self, jax_CPU_array):
        return np.from_dlpack(jax_CPU_array)

    def _spit_neg_infinity(self):
        return -jnp.inf

    def _spit_neg_number(self):
        return -8.01

class CorrelatedPulsarRedNoise:
    """
    A unified class for constructing the red-noise covariance (phi) matrix
    for pulsar timing array (PTA) analyses.

    This class generalises the original ``PowerLawRedNoise``,
    ``FreeSpecRedNoise``, ``DMRedNoise``, and ``ParameterizedGwb`` classes
    into a single entry-point.  The non-GWB (intrinsic red noise, DM, …)
    spectral components are described by callable PSD functions in exactly
    the same way as the GWB PSD, together with companion helper dictionaries
    that carry prior bounds and optional fixed-parameter information.

    Parameter vector layout
    -----------------------
    The flat ``xs`` vector passed to ``get_phi_mat`` / ``get_phi_diag`` is
    ordered as::

        xs = [ irn_psd_params (num_IR_params),
               dm_psd_params  (num_DM_params),   ← only if irn_psd_func is not None
               gwb_psd_params (n_gwb_varied),
               orf_params     (n_orf_varied) ]    ← only if ORF has free params

    where:

    * ``irn_psd_params`` is a flat array of shape
      ``(n_irn_varied_per_pulsar * Npulsars,)`` with pulsar index varying
      *slowest* (i.e. all params for pulsar 0 first, then pulsar 1, etc.).
    * ``dm_psd_params``  follows the same layout for the DM component.
    * ``gwb_psd_params`` are the varied GWB PSD parameters.
    * ``orf_params``     are the varied ORF parameters (absent when the ORF is
      fixed, i.e. no free parameters).

    IRN / DM PSD function convention
    ---------------------------------
    Both ``irn_psd_func`` and ``dm_psd_func`` must have the signature::

        psd_func(f, df, *params) -> jnp.ndarray of shape (n_bins,)

    where ``params`` is the parameter *tuple for a single pulsar*.  The
    function is called once per pulsar inside a ``vmap``.

    If ``irn_psd_func`` is ``None`` (the default) the class reduces to a
    GWB-only model (equivalent to the original ``ParameterizedGwb``).

    GWB / ORF function convention
    ------------------------------
    ``gwb_psd_func(f, df, *params) -> jnp.ndarray of shape (crn_bins,)``
    ``orf_func(angle, *params)     -> jnp.ndarray of shape (n_pairs,)``

    Helper dictionary keys
    ----------------------
    ``gwb_helper_dictionary`` (same as before):
        * ``"gwb_psd_param_upper_lim"``
        * ``"gwb_psd_param_lower_lim"``
        * ``"ordered_gwb_psd_model_params"``  (optional, for ordering check)
        * ``"fixed_gwb_psd_param_indices"``   (optional)
        * ``"fixed_gwb_psd_param_values"``    (optional)
        * ``"ordered_orf_model_params"``       (optional, marks free ORF)

    ``irn_helper_dictionary`` / ``dm_helper_dictionary`` (same structure,
    but keys prefixed with ``"psd_"`` instead of ``"gwb_psd_"`` / ``"gwb_"``):
        * ``"psd_param_upper_lim"``
        * ``"psd_param_lower_lim"``
        * ``"ordered_psd_model_params"``  (optional)
        * ``"fixed_psd_param_indices"``   (optional)
        * ``"fixed_psd_param_values"``    (optional)

    Authors
    -------
    Nima Laal (original pandora classes, 02/12/2025)
    Unified refactor: Ge (06/2025)
    """
    def __init__(
        self,
        psr_pos,
        Npulsars,
        signal_indices,
        linear_timing_model_size,
        # ---- GWB ----
        gwb_psd_func,
        orf_func,
        gwb_helper_dictionary,
        crn_bins,
        f_common,
        # ---- non-GWB intrinsic red noise (optional) ----
        irn_psd_func=None,
        irn_helper_dictionary=None,
        irn_bins=None,
        f_irn=None,
        # ---- DM noise (optional) ----
        dm_psd_func=None,
        dm_helper_dictionary=None,
        dm_bins=None,
        f_dm=None,
        # ---- shared ----
        renorm_const=1.0,
    ):
        ...
        # Build basis and extract signal indices
        # signal_indices maps signal name -> slice into Fmat columns (offset by tm_size)
        # We need the Fourier-only row indices for phi_diag, so subtract tm_offset
        self.signal_indices = signal_indices
        tm_offset = linear_timing_model_size  # 0 if no T| prefix

        def _fourier_slice(name):
            """Convert Fmat column slice -> phi_diag frequency-row slice."""
            sl = self.signal_indices[name]
            start_col = sl.start - tm_offset
            stop_col  = sl.stop  - tm_offset
            # Each freq bin = 2 columns; integer-divide to get freq indices
            return slice(start_col // 2, stop_col // 2)

        # Replace first/last_*_bin_index with slices from signal_indices
        self.GWB_slice     = _fourier_slice('cor')   # or whatever name used in basis_string
        self.GWB_fidxs     = jnp.arange(self.GWB_slice.start, self.GWB_slice.stop)
        self.crn_bins      = crn_bins

        # ------------------------------------------------------------------ #
        #  Basic bookkeeping                                                   #
        # ------------------------------------------------------------------ #
        self.gwb_helper_dictionary = gwb_helper_dictionary
        self.Npulsars = Npulsars
        self.psr_pos = psr_pos
        self.gwb_psd_func = gwb_psd_func
        self.orf_func = orf_func
        self.irn_psd_func = irn_psd_func
        self.dm_psd_func = dm_psd_func
        self.diag_idx = jnp.arange(Npulsars)
        self.ppair_number = int(Npulsars * (Npulsars - 1) * 0.5)
        self.renorm_const = renorm_const
        self.logrenorm_offset = 0.5 * jnp.log10(renorm_const)

        # Frequency scalars / arrays
        self.crn_bins = crn_bins
        self.f_common = f_common[..., None]
        self.df_gwb = jnp.diff(jnp.concatenate((jnp.zeros((1)), 
                                                f_common), 
                                                axis = 0))[..., None]

        # ------------------------------------------------------------------ #
        #  IRN bookkeeping                                                     #
        # ------------------------------------------------------------------ #
        has_irn = irn_psd_func is not None
        self.has_irn = has_irn
        if has_irn:
            assert irn_bins is not None and f_irn is not None, (
                "irn_bins and f_irn must be supplied when irn_psd_func is given."
            )
            assert irn_helper_dictionary is not None, (
                "irn_helper_dictionary must be supplied when irn_psd_func is given."
            )
            self.irn_bins = irn_bins if has_irn else crn_bins
            self.f_irn = f_irn if f_irn.ndim == 2 else jnp.broadcast_to(f_irn, (self.Npulsars, self.irn_bins)) 
            self.df_irn = jnp.diff(jnp.concatenate((jnp.zeros((self.Npulsars, 1)), self.f_irn), axis = 1))

        # ------------------------------------------------------------------ #
        #  DM bookkeeping                                                      #
        # ------------------------------------------------------------------ #
        has_dm = dm_psd_func is not None
        self.has_dm = has_dm
        if has_dm:
            assert dm_bins is not None and f_dm is not None, (
                "dm_bins and f_dm must be supplied when dm_psd_func is given."
            )
            assert dm_helper_dictionary is not None, (
                "dm_helper_dictionary must be supplied when dm_psd_func is given."
            )
            self.dm_bins = dm_bins
            self.f_dm = f_dm if f_dm.ndim == 2 else jnp.broadcast_to(f_dm, (self.Npulsars, self.dm_bins))
            self.df_dm = jnp.diff(jnp.concatenate((jnp.zeros((self.Npulsars, 1)), self.f_dm), axis = 1))
            
        # ------------------------------------------------------------------ #
        #  Frequency index arrays                                              #
        # ------------------------------------------------------------------ #
        if has_irn:
            self.IRN_slice = _fourier_slice('unc')   # or 'irn', match basis_string name
            self.nonGWB_fidxs = jnp.array(
                [i for i in range(int(self.IRN_slice.start), int(self.IRN_slice.stop))
                if i not in self.GWB_fidxs]
            )
            self.separate_inversion_strat = bool(self.nonGWB_fidxs.any())

        if has_dm:
            self.DM_slice  = _fourier_slice('dm')
            self.DM_fidxs  = jnp.arange(self.DM_slice.start, self.DM_slice.stop)

        # n_total_bins is now just the stop of the last signal slice
        all_stops = [self.GWB_slice.stop]
        if has_irn: all_stops.append(self.IRN_slice.stop)
        if has_dm:  all_stops.append(self.DM_slice.stop)
        self.n_total_bins = max(all_stops)

        # ------------------------------------------------------------------ #
        #  Pulsar-pair angular separations & multi-dim index arrays           #
        # ------------------------------------------------------------------ #
        self._eye = jnp.repeat(np.eye(Npulsars)[None], crn_bins, axis=0)

        I_tri, J_tri = np.tril_indices(Npulsars)
        a, b = [], []
        for i, j in zip(I_tri, J_tri):
            if i != j:
                a.append(i)
                b.append(j)
        a = np.array(a, dtype=int)
        b = np.array(b, dtype=int)

        self.xi = jnp.array(
            [np.arccos(np.dot(psr_pos[i], psr_pos[j])) for i, j in zip(a, b)]
        )

        self.I = jnp.repeat(a[None, :], crn_bins, axis=0)
        self.J = jnp.repeat(b[None, :], crn_bins, axis=0)
        self.KGW = jnp.repeat(self.GWB_fidxs[:, None], len(a), axis=1)

        if has_irn and self.nonGWB_fidxs.any():
            self.DIR = jnp.repeat(self.diag_idx[None, :],
                                  len(self.nonGWB_fidxs), axis=0)
            self.KIR = jnp.repeat(self.nonGWB_fidxs[:, None], Npulsars, axis=1)

        if has_dm:
            self.DIRDM = jnp.repeat(self.diag_idx[None, :], dm_bins, axis=0)
            self.KDM = jnp.repeat(self.DM_fidxs[:, None], Npulsars, axis=1)

        # ------------------------------------------------------------------ #
        #  Parse GWB PSD + ORF                                                #
        # ------------------------------------------------------------------ #
        # Re-use the GWB key-naming convention but route through the generic
        # helper by renaming keys temporarily.
        _gwb_hd_adapted = {
            k.replace("gwb_psd_", "psd_").replace("gwb_", "psd_"): v
            for k, v in gwb_helper_dictionary.items()
            if k not in ("ordered_orf_model_params",)
        }
        # keep ordered_gwb_psd_model_params accessible as ordered_psd_model_params
        if "ordered_gwb_psd_model_params" in gwb_helper_dictionary:
            _gwb_hd_adapted["ordered_psd_model_params"] = (
                gwb_helper_dictionary["ordered_gwb_psd_model_params"]
            )

        self.gwb_param_container, self.gwb_varied_indxs = _parse_psd_func(
            gwb_psd_func, _gwb_hd_adapted, crn_bins
        )

        self.orf_fixed, _ = _parse_orf_func(orf_func, gwb_helper_dictionary)
        if self.orf_fixed:
            self.orf_val = orf_func(self.xi)

        # ------------------------------------------------------------------ #
        #  Parse IRN PSD                                                       #
        # ------------------------------------------------------------------ #
        if has_irn:
            self.irn_param_container, self.irn_varied_indxs = _parse_psd_func(
                irn_psd_func, irn_helper_dictionary, irn_bins
            )
            # Number of varied params per pulsar
            self.n_irn_varied = int(len(self.irn_varied_indxs))
            self.num_IR_params = self.n_irn_varied * Npulsars
        else:
            self.num_IR_params = 0

        # ------------------------------------------------------------------ #
        #  Parse DM PSD                                                        #
        # ------------------------------------------------------------------ #
        if has_dm:
            self.dm_param_container, self.dm_varied_indxs = _parse_psd_func(
                dm_psd_func, dm_helper_dictionary, dm_bins
            )
            self.n_dm_varied = int(len(self.dm_varied_indxs))
            self.num_DM_params = self.n_dm_varied * Npulsars
        else:
            self.num_DM_params = 0

        # ------------------------------------------------------------------ #
        #  Parameter-vector slice indices                                      #
        # ------------------------------------------------------------------ #
        # Layout: [ IRN(0..num_IR_params) | DM(..+num_DM_params) |
        #           GWB_PSD(..+n_gwb) | ORF(..+n_orf) ]
        self.irn_end_idx = self.num_IR_params
        self.dm_end_idx = self.irn_end_idx + self.num_DM_params
        self.gwb_psd_end_idx = self.dm_end_idx + int(len(self.gwb_varied_indxs))

        # ------------------------------------------------------------------ #
        #  Prior bounds                                                        #
        # ------------------------------------------------------------------ #
        upper, lower = jnp.array([]), jnp.array([])

        if has_irn:
            irn_upper = jnp.tile(
                irn_helper_dictionary["psd_param_upper_lim"] + self.logrenorm_offset,
                Npulsars
            )
            irn_lower = jnp.tile(
                irn_helper_dictionary["psd_param_lower_lim"] + self.logrenorm_offset,
                Npulsars
            )
            upper = jnp.concatenate([upper, irn_upper])
            lower = jnp.concatenate([lower, irn_lower])

        if has_dm:
            dm_upper = jnp.tile(
                dm_helper_dictionary["psd_param_upper_lim"] + self.logrenorm_offset,
                Npulsars
            )
            dm_lower = jnp.tile(
                dm_helper_dictionary["psd_param_lower_lim"] + self.logrenorm_offset,
                Npulsars
            )
            upper = jnp.concatenate([upper, dm_upper])
            lower = jnp.concatenate([lower, dm_lower])

        upper = jnp.concatenate(
            [upper, jnp.array(gwb_helper_dictionary["gwb_psd_param_upper_lim"])]
        )
        lower = jnp.concatenate(
            [lower, jnp.array(gwb_helper_dictionary["gwb_psd_param_lower_lim"])]
        )

        if not self.orf_fixed:
            upper = jnp.concatenate(
                [upper, jnp.array(gwb_helper_dictionary["orf_param_upper_lim"])]
            )
            lower = jnp.concatenate(
                [lower, jnp.array(gwb_helper_dictionary["orf_param_lower_lim"])]
            )

        self.upper_prior_lim_all = upper
        self.lower_prior_lim_all = lower

        if renorm_const != 1:
            warnings.warn(
                "You have chosen to change units.  Make sure your amplitude "
                "priors reflect that!"
            )

    # ---------------------------------------------------------------------- #
    #  Internal PSD helpers                                                   #
    # ---------------------------------------------------------------------- #

    @jit_method
    def _eval_gwb_psd(self, gwb_psd_params):
        """Evaluate the GWB PSD on ``self.f_common``."""
        filled = self.gwb_param_container.at[self.gwb_varied_indxs].set(
            gwb_psd_params
        )
        return self.gwb_psd_func(self.f_common, self.df_gwb, *filled)

    @jit_method
    def _eval_irn_psd_all(self, irn_params_flat):
        """
        Evaluate the IRN PSD for *all* pulsars.

        Parameters
        ----------
        irn_params_flat : jnp.ndarray, shape (num_IR_params,)
            Flat array; reshaped to (Npulsars, n_irn_varied) before vmapping.

        Returns
        -------
        jnp.ndarray, shape (irn_bins, Npulsars)
        """
        per_psr = irn_params_flat.reshape(self.Npulsars, self.n_irn_varied)

        def _single(params, freqs_, df_):
            filled = self.irn_param_container.at[self.irn_varied_indxs].set(params)
            return self.irn_psd_func(freqs_, df_, *filled)

        return jax.vmap(_single)(per_psr, self.f_irn, self.df_irn).T  # → (irn_bins, Npulsars)

    @jit_method
    def _eval_dm_psd_all(self, dm_params_flat):
        """
        Evaluate the DM PSD for *all* pulsars.

        Returns
        -------
        jnp.ndarray, shape (dm_bins, Npulsars)
        """
        per_psr = dm_params_flat.reshape(self.Npulsars, self.n_dm_varied)

        def _single(params, freqs_, df_):
            filled = self.dm_param_container.at[self.dm_varied_indxs].set(params)
            return self.dm_psd_func(freqs_, df_, *filled)

        return jax.vmap(_single)(per_psr, self.f_dm, self.df_dm).T  # → (dm_bins, Npulsars)

    # ---------------------------------------------------------------------- #
    #  Parameter unpacking                                                    #
    # ---------------------------------------------------------------------- #

    @jit_method
    def _unpack(self, xs):
        """Return (irn_flat, dm_flat, gwb_psd_params, orf_params) from ``xs``."""
        irn = xs[:self.irn_end_idx]
        dm  = xs[self.irn_end_idx:self.dm_end_idx]
        gwb = xs[self.dm_end_idx:self.gwb_psd_end_idx]
        orf = xs[self.gwb_psd_end_idx:]
        return irn, dm, gwb, orf

    # ---------------------------------------------------------------------- #
    #  Core phi builders                                                      #
    # ---------------------------------------------------------------------- #

    @jit_method
    def get_phi_diag(self, xs):
        irn_flat, dm_flat, gwb_params, _ = self._unpack(xs)
        psd_common = self._eval_gwb_psd(gwb_params)

        phi_diag = jnp.zeros((self.n_total_bins, self.Npulsars))

        if self.has_irn:
            irn_psd = self._eval_irn_psd_all(irn_flat)          # (irn_bins, Npulsars)
            phi_diag = phi_diag.at[self.IRN_slice].add(irn_psd)

        if self.has_dm:
            dm_psd = self._eval_dm_psd_all(dm_flat)             # (dm_bins, Npulsars)
            phi_diag = phi_diag.at[self.DM_slice].add(dm_psd)

        phi_diag = phi_diag.at[self.GWB_slice].add(psd_common)

        return phi_diag, psd_common

    @jit_method
    def get_phi_mat(self, xs):
        """
        Build the full phi-matrix  (n_total_bins, Npulsars, Npulsars).

        Off-diagonal (cross-pulsar) elements are filled only in the GWB bins,
        weighted by the ORF.

        Parameters
        ----------
        xs : jnp.ndarray
            Flat parameter vector (see class docstring for layout).

        Returns
        -------
        phi : jnp.ndarray  (n_total_bins, Npulsars, Npulsars)
        """
        phi_diag, psd_common = self.get_phi_diag(xs)
        n_total = phi_diag.shape[0]

        phi = jnp.zeros((n_total, self.Npulsars, self.Npulsars))
        phi = phi.at[:, self.diag_idx, self.diag_idx].set(phi_diag)

        *_, orf_params = self._unpack(xs)
        orf_val = self.orf_val if self.orf_fixed else self.orf_func(self.xi, *orf_params)
        return phi.at[self.KGW, self.I, self.J].set(orf_val * psd_common)

    @jit_method
    def get_phi_mat_full(self, xs):
        """
        Like ``get_phi_mat`` but also fills the *upper* triangle so the matrix
        is explicitly symmetric.
        """
        phi_diag, psd_common = self.get_phi_diag(xs)
        n_total = phi_diag.shape[0]

        phi = jnp.zeros((n_total, self.Npulsars, self.Npulsars))
        phi = phi.at[:, self.diag_idx, self.diag_idx].set(phi_diag)

        *_, orf_params = self._unpack(xs)
        orf_val = self.orf_val if self.orf_fixed else self.orf_func(self.xi, *orf_params)
        phi = phi.at[self.KGW, self.I, self.J].set(orf_val * psd_common)
        return phi.at[self.KGW, self.J, self.I].set(orf_val * psd_common)

    @jit_method
    def get_phi_mat_CURN(self, xs):
        """
        Return the phi diagonal (CURN = Common Uncorrelated Red Noise):
        cross-pulsar correlations are ignored even if an ORF was supplied.

        Returns
        -------
        phi_diag : jnp.ndarray  (n_total_bins, Npulsars)
        psd_common : jnp.ndarray  (crn_bins,)
        """
        return self.get_phi_diag(xs)

    @jit_method
    def get_phi_mat_from_diag(self, phi_diag, psd_common, orf_params=None):
        """
        Build the full phi-matrix from a pre-computed diagonal and GWB PSD.

        Parameters
        ----------
        phi_diag : jnp.ndarray  (n_total_bins, Npulsars)
        psd_common : jnp.ndarray  (crn_bins,)
        orf_params : jnp.ndarray or None
            Required when the ORF is not fixed.
        """
        n_total = phi_diag.shape[0]
        phi = jnp.zeros((n_total, self.Npulsars, self.Npulsars))
        phi = phi.at[:, self.diag_idx, self.diag_idx].set(phi_diag)

        if self.orf_fixed:
            orf_val = self.orf_val
        else:
            assert orf_params is not None, (
                "orf_params must be provided when the ORF has free parameters."
            )
            orf_val = self.orf_func(self.xi, *orf_params)

        return phi.at[self.KGW, self.I, self.J].set(orf_val * psd_common)

    # ---------------------------------------------------------------------- #
    #  Inversion                                                              #
    # ---------------------------------------------------------------------- #

    @jit_method
    def get_phi_mat_inv(self, phi):
        """
        Invert the phi-matrix using mixed Cholesky + diagonal strategies.

        GWB-containing bins use Cholesky factorisation; purely-IRN bins and
        DM bins (which are diagonal) use direct reciprocal inversion.

        Parameters
        ----------
        phi : jnp.ndarray  (n_total_bins, Npulsars, Npulsars)

        Returns
        -------
        phiinv : jnp.ndarray  (2*n_total_bins, Npulsars, Npulsars)
            Repeated twice along axis-0 (one for each quadrature component).
        logdet_phi : float
        """
        n_total = phi.shape[0]
        phiinv = jnp.zeros_like(phi)

        # --- GWB bins: Cholesky ---
        cp = jsp.linalg.cho_factor(phi[self.GWB_fidxs], lower=True)
        phiinv = phiinv.at[self.GWB_fidxs].set(
            jsp.linalg.cho_solve(cp, self._eye)
        )
        logdet_phi = 2.0 * jnp.sum(jnp.log(cp[0].diagonal(axis1=-2, axis2=-1)))

        # --- IRN-only bins: diagonal inversion ---
        if self.has_irn and self.separate_inversion_strat:
            diags_irn = phi[self.nonGWB_fidxs].diagonal(axis1=-2, axis2=-1)
            phiinv = phiinv.at[self.KIR, self.DIR, self.DIR].set(1.0 / diags_irn)
            logdet_phi = logdet_phi + jnp.sum(jnp.log(diags_irn))

        # --- DM bins: diagonal inversion ---
        if self.has_dm:
            diags_dm = phi[self.DM_fidxs].diagonal(axis1=-2, axis2=-1)
            phiinv = phiinv.at[self.KDM, self.DIRDM, self.DIRDM].set(1.0 / diags_dm)
            logdet_phi = logdet_phi + jnp.sum(jnp.log(diags_dm))

        return jnp.repeat(phiinv, 2, axis=0), 2.0 * logdet_phi

    @jit_method
    def partial_reparm_helper(self, xs, pad_mask):
        irn_flat, dm_flat, gwb_params, _ = self._unpack(xs)
        psd_common = self._eval_gwb_psd(gwb_params)
        *_, orf_params = self._unpack(xs)
        orf_val = self.orf_val if self.orf_fixed else self.orf_func(self.xi, *orf_params)

        # non-GWB diagonal (IRN + DM rows only, no GWB rows)
        n_non_gwb = (self.IRN_slice.stop - self.IRN_slice.start if self.has_irn else 0) + \
                    (self.DM_slice.stop  - self.DM_slice.start  if self.has_dm  else 0)
        phi_diag_non_gwb = jnp.zeros((n_non_gwb, self.Npulsars))

        # Offset slices relative to phi_diag_non_gwb (which starts at 0)
        irn_local = slice(0, self.IRN_slice.stop - self.IRN_slice.start) if self.has_irn else None
        dm_local  = slice(irn_local.stop if irn_local else 0,
                        (irn_local.stop if irn_local else 0) + 
                        (self.DM_slice.stop - self.DM_slice.start)) if self.has_dm else None

        if self.has_irn:
            phi_diag_non_gwb = phi_diag_non_gwb.at[irn_local].add(
                self._eval_irn_psd_all(irn_flat)
            )
        if self.has_dm:
            phi_diag_non_gwb = phi_diag_non_gwb.at[dm_local].add(
                self._eval_dm_psd_all(dm_flat)
            )

        phi_gwb = jnp.zeros((self.crn_bins, self.Npulsars, self.Npulsars))
        phi_gwb = phi_gwb.at[:, self.diag_idx, self.diag_idx].set(psd_common)
        phi_gwb = phi_gwb.at[self.KGW, self.I, self.J].set(orf_val * psd_common)
        phi_gwb = phi_gwb.at[self.KGW, self.J, self.I].set(orf_val * psd_common)

        # per-pulsar phi
        phiinv_non_gwb = 1 / phi_diag_non_gwb
        logdet_phi_non_gwb = 2.0 * jnp.sum(jnp.log(phi_diag_non_gwb))
        ltm_size = pad_mask.shape[-1]
        concat_phiinv_non_gwb = jnp.zeros((2 * phi_diag_non_gwb.shape[0] + ltm_size, self.Npulsars))
        concat_phiinv_non_gwb = concat_phiinv_non_gwb.at[ltm_size:].set(jnp.repeat(phiinv_non_gwb, repeats=2, axis=0))
        concat_phiinv_non_gwb = concat_phiinv_non_gwb.at[:ltm_size].set(pad_mask.mT)
        concat_phiinv_non_gwb += 1e-40

        # gwb phi
        cp = jsp.linalg.cho_factor(phi_gwb, lower=True)
        phiinv_gwb = jsp.linalg.cho_solve(cp, self._eye)
        logdet_phi_gwb = 4.0 * jnp.sum(jnp.log(cp[0].diagonal(axis1=-2, axis2=-1)))     # x4 b/c Cholesky + nfreq size
        
        result = (jnp.repeat(phiinv_gwb, repeats=2, axis=0),
                  logdet_phi_gwb,
                  concat_phiinv_non_gwb.mT,
                  logdet_phi_non_gwb)
        
        return result



    # ---------------------------------------------------------------------- #
    #  Prior                                                                  #
    # ---------------------------------------------------------------------- #

    @jit_method
    def get_lnprior(self, xs):
        """Uniform log-prior: returns a small constant if in bounds, -inf otherwise."""
        in_bounds = jnp.logical_and(
            xs > self.lower_prior_lim_all, xs < self.upper_prior_lim_all
        ).all()
        return jax.lax.cond(in_bounds, self._spit_neg_number, self._spit_neg_infinity)

    def get_lnprior_numpy(self, xs):
        return self.get_lnprior(xs).__array__()

    @jit_method
    def make_initial_guess(self, key):
        """Draw a uniform initial sample within the prior bounds."""
        return jr.uniform(
            key,
            shape=(self.upper_prior_lim_all.shape[0],),
            minval=self.lower_prior_lim_all,
            maxval=self.upper_prior_lim_all,
        )

    # ---------------------------------------------------------------------- #
    #  Utilities                                                              #
    # ---------------------------------------------------------------------- #

    def jax_to_numpy_CPU(self, jax_CPU_array):
        return np.from_dlpack(jax_CPU_array)

    def _spit_neg_infinity(self):
        return -jnp.inf

    def _spit_neg_number(self):
        return -8.01