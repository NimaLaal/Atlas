"""
Holds the class describing all deterministic signals.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

import numpy as np
from scipy.signal.windows import tukey
from typing import Optional, Callable
from functools import partial

from ATLAS.utils import jit_method, jit
from ATLAS.data import PTA_Data
from ATLAS.signals.factorized.base import SuperSignal
from ATLAS.signals import signals_utils as sutils
from ATLAS.signals.deterministic import utils as dutils


class Deterministic(SuperSignal):

    """
    Class for determinist signals. Contains methods to get
    the induced residuals from a deterministic signal over
    the observed TOAs using interpolation from a Fourier
    representation. Used for resiual subtraction.

    Required Attributes
    -------------------
    name : str
        The name of the deterministic signal.
    data : Data
        An instance of the Data class from the `data` module.
    signal_helper : dict
        The stochastic signal specification passed straight to
        ``SuperSignal.__init__`` (see that class for the expected schema).
    get_delays_func : Callable
        A JAX friendly function which takes in parameters of the
        deterministic model and outputs the induced timing delays
        across all pulsars. See det_signals.py for example functions.
    det_parameter_bounds : array
        The minima and maxima allowed values of the deterministic parameters,
        with shape (nparams, 2) where nparams is the length of the parameters
        supplied to 'get_delays_func'. The parameter minima are at [:, 0] and
        the maxima at [:, 1].
    nfreqs_det : int
        Number of frequency bins used to reprsentent deterministic signal.
        Defaults to 60.
    with_psr_params : bool
        Deterministic signals from individual binaries depend on a set of (npsr) pulsar
        phase parameters and (npsr) pulsar distance parameters, where npsr is the number
        of pulsars in the array. If False, the pipeline will wrap 'get_delays_func'
        to accept these parameters anyway and feed it NoneType in those parameter slots.
    window_ext_factor : float
        Factor by which to extend Tspan for frequency bins representing deterministic signal,
        aids with Gibbs phenomena. Defaults to 2.
    get_coeffs_func : Callable
        A JAX friendly function which maps the parameters of the deterministic
        model to the frequency representation of the signal. If None, defaults
        to the FFT method below.
    additional_ln_factor : callable
        Additional log-pdf to add to posterior call. Useful to add non-uniform priors
        or reparameterizations.
    """

    def __init__(self,
                 name,
                 data,
                 get_delays_func,
                 det_parameter_bounds,
                 nfreqs_det = 60,
                 with_psr_params = True,
                 window_ext_factor = 2.,
                 get_coeffs_func = None,
                 ):

        self.name = name
        self.data = data
        self.get_delays_func = get_delays_func
        self.det_parameter_bounds = jnp.array(det_parameter_bounds)
        self.det_param_mins = self.det_parameter_bounds[:, 0]
        self.det_param_maxs = self.det_parameter_bounds[:, 1]
        self.nparams_det = self.det_param_mins.shape[0]
        self.nfreqs_det = nfreqs_det
        self.num_coeff_det = 2 * self.nfreqs_det
        self.with_psr_params = with_psr_params

        # number of TOAs per pulsar
        self.num_toas_per_psr = jnp.array([psr.toas.shape[0] for psr in self.data.psrs])

        # if pulsar parameters not needed for model, wrap input function
        self.get_delays_func = jit(self._wrap_delays_func(self.get_delays_func,
                                                              self.with_psr_params))

        # if no get_coeffs_func specified, use FFT method
        self._get_coeffs_func = get_coeffs_func or self.get_coeffs_via_FFT
        self.get_coeffs_func = jit(self._get_coeffs_func)

        # sparse TOAs for CW FFT
        window_ext = self.data.pta_tspan * window_ext_factor
        Tspan_ext = self.data.pta_tspan + 2. * window_ext
        first_toa = np.min([np.min(psr.toas) for psr in self.data.psrs])
        last_toa = np.max([np.max(psr.toas) for psr in self.data.psrs])
        sparse_toas_det = np.array([np.linspace(first_toa - window_ext, last_toa + window_ext,
                                                self.num_coeff_det + 2, endpoint=False)
                                    for _ in range(self.data.npsrs)])
        self.sparse_toas_det_jax = jnp.array(sparse_toas_det)
        sparse_toas_scaled_shifted_np = [(sparse_toas - dutils.tref) * dutils.cw_renorm
                                         for sparse_toas in sparse_toas_det]
        self.sparse_toas_shifted_scaled = jnp.array(sparse_toas_scaled_shifted_np)
        self.Nsparse = sparse_toas_det.shape[1]
        self.freqs_forFFT = jnp.array([np.fft.fftfreq(self.Nsparse, Tspan_ext / self.Nsparse)
                                       for _ in range(self.data.npsrs)])
        self.Tukey_det = jnp.array(tukey(self.Nsparse, alpha=(Tspan_ext - self.data.pta_tspan) / Tspan_ext))

        freqs_for_Fmat = jnp.array([self.freqs_forFFT[0, j + 1] for j in range(self.nfreqs_det)])
        Fs_det = [sutils.get_fourier_design_matrix(psr.toas, freqs_for_Fmat)
                       for psr in self.data.psrs]
        self.Fs_det_concat = jnp.concat(Fs_det, axis=0)


    def get_coeffs_via_FFT(self, det_params, psr_phases, psr_dists):
        """
        Mapping from deterministic signal parameters to Fourier space.
        This is simply a FFT.

        Note the frequency bins used here are generally different than those used
        by stochastic models. To avoid Gibbs phenomena from non-periodic deterministic
        signals over Tspan, we use an "extended" basis where we FFT the signal over a period
        extended either side of Tspan after applying a Tukey window. The Fourier design matrix,
        however, maps the Fourier coefficients to TOAs **within** the PTA Tspan. This introduces
        cross terms in the posterior. See Gundersen & Cornish 2025.

        Parameters
        ----------
        det_params : array
            Parmeters values of the deterministic model.
        psr_phases : array or None
            The phase of the gravitational wave at each pulsar. This is a model parameter
            for continuous gravitational waves from individual SMBHBs. For other deterministic
            models, this can be None.
        psr_dists : array or None
            The distance to each pulsar [kpc]. This is a model parameter for continuous
            gravitational waves from individual SMBHBs. For other deterministic models,
            this can be None.

        Returns
        -------
        coeff : array
            A (npsrs, 2*nfreq) array where npsrs is the number of pulsars in the array and
            nfreq is the number of frequency bins used to represent to the deterministic
            model in Fourier space.
        """

        # get timing delays induced by the deterministic signal over "sparse" (evenly-spaced) TOAs
        det_residuals = self.get_delays_func(self.sparse_toas_shifted_scaled, self.data.psr_pos,
                                                    det_params, psr_phases, psr_dists)
        # window residuals over extended observation
        det_residuals_windowed = self.Tukey_det * det_residuals
        # do FFT
        det_fft = jnp.fft.fft(det_residuals_windowed, n=None, axis=-1, norm=None)  # dim (Np, 2 * Nf + 2)

        # apply time shift to set initial time
        det_fft *= jnp.exp(-1.j * 2 * jnp.pi * self.freqs_forFFT * self.sparse_toas_det_jax[:, 0:1])

        # extract sine and cosine coefficients
        a_n = jnp.imag(det_fft[:, :self.Nsparse // 2]) * (-2 / self.Nsparse)  # (Np, Nf + 1)
        b_n = jnp.real(det_fft[:, :self.Nsparse // 2]) * (2 / self.Nsparse)  # (Np, Nf + 1)

        # interweave sine/cosine coefficients and reshape to (Np, 2 * Nf)
        coeff = jnp.concatenate((a_n, b_n), axis=1).reshape((self.data.npsrs, 2, self.nfreqs_det + 1))\
                        .transpose((0, 2, 1)).reshape((self.data.npsrs, self.num_coeff_det + 2))

        return coeff[:, 2:]  # remove DC


    def _wrap_delays_func(self, func, with_psr_params):
        """
        If a deterministic model is supplied that does **not**
        take pulsar distance and phase as model parameters,
        then the 'get_delay_func' is wrapped to accept these
        parameters anyway and the sampler will supply NoneTypes
        for these inputs.
        """
        if with_psr_params:
            return func
        else:
            def wrapped(toas, psrpos, params, psr_phases, psr_dists):
                return func(toas, psrpos, params)
            return wrapped
    
    def get_det_residuals(self, det_params, psr_phases, psr_dists):
        """
        Get the residuals induced over the observed TOAs via
        interpolation with the Fourier basis.

        Parameters
        ----------
        det_params : array
            Parameters of deterministic signal.
        psr_phases : array
            Phase at each pulsar [npsr].
        psr_dists : array
            Distance to each pulsar [kpc], [npsr].

        Returns
        -------
        det_residuals : array
            Residuals induced by deterministic signal
            over the observed TOAs, [ntoas_all].
        """

        # get Fourier representation
        a_det = self.get_coeffs_func(det_params, psr_phases, psr_dists)  # [npsr, 2 * nfreqs_det]
        a_per_toa = jnp.repeat(a_det, self.num_toas_per_psr, axis=0)

        # use Fourier design matrix to interpolate onto TOAs
        det_residuals = jnp.sum(self.Fs_det_concat * a_per_toa, axis=1)
        return det_residuals


class JointDeterministic(SuperSignal):

    """
    Class for joint stochastic + deterministic analyses.

    This class *is a* stochastic :class:`~ATLAS.signals.factorized.base.SuperSignal`
    (it inherits the full factorized-likelihood machinery: basis construction, white
    noise helper products, the reparameterized stochastic posterior, etc.) and layers
    the deterministic signal on top. It overrides the posterior evaluation to include the
    deterministic contribution.

    Required Attributes
    -------------------
    name : str
        The name of the deterministic signal.
    data : Data
        An instance of the Data class from the `data` module.
    signal_helper : dict
        The stochastic signal specification passed straight to
        ``SuperSignal.__init__`` (see that class for the expected schema).
    get_delays_func : Callable
        A JAX friendly function which takes in parameters of the
        deterministic model and outputs the induced timing delays
        across all pulsars. See det_signals.py for example functions.
    det_parameter_bounds : array
        The minima and maxima allowed values of the deterministic parameters,
        with shape (nparams, 2) where nparams is the length of the parameters
        supplied to 'get_delays_func'. The parameter minima are at [:, 0] and
        the maxima at [:, 1].
    nfreqs_det : int
        Number of frequency bins used to reprsentent deterministic signal.
        Defaults to 60.
    with_psr_params : bool
        Deterministic signals from individual binaries depend on a set of (npsr) pulsar
        phase parameters and (npsr) pulsar distance parameters, where npsr is the number
        of pulsars in the array. If False, the pipeline will wrap 'get_delays_func'
        to accept these parameters anyway and feed it NoneType in those parameter slots.
    window_ext_factor : float
        Factor by which to extend Tspan for frequency bins representing deterministic signal,
        aids with Gibbs phenomena. Defaults to 2.
    get_coeffs_func : Callable
        A JAX friendly function which maps the parameters of the deterministic
        model to the frequency representation of the signal. If None, defaults
        to the FFT method below.
    additional_ln_factor : callable
        Additional log-pdf to add to posterior call. Useful to add non-uniform priors
        or reparameterizations.
    """

    def __init__(self,
                 name,
                 data,
                 signal_helper,
                 get_delays_func,
                 det_parameter_bounds,
                 nfreqs_det = 60,
                 with_psr_params = True,
                 window_ext_factor = 2.,
                 get_coeffs_func = None,
                 additional_ln_factor = None,
                 ):

        # build the stochastic (SuperSignal) machinery this class extends
        super().__init__(signal_helper = signal_helper, data = data)

        self.name = name
        self.get_delays_func = get_delays_func
        self.det_parameter_bounds = jnp.array(det_parameter_bounds)
        self.det_param_mins = self.det_parameter_bounds[:, 0]
        self.det_param_maxs = self.det_parameter_bounds[:, 1]
        self.nparams_det = self.det_param_mins.shape[0]
        self.nfreqs_det = nfreqs_det
        self.num_coeff_det = 2 * self.nfreqs_det
        self.with_psr_params = with_psr_params
        self.additional_ln_factor = additional_ln_factor

        # if pulsar parameters not needed for model, wrap input function
        self.get_delays_func = jit(self._wrap_delays_func(self.get_delays_func,
                                                              self.with_psr_params))

        # if no get_coeffs_func specified, use FFT method
        self._get_coeffs_func = get_coeffs_func or self.get_coeffs_via_FFT
        self.get_coeffs_func = jit(self._get_coeffs_func)

        # sparse TOAs for CW FFT
        window_ext = self.data.pta_tspan * window_ext_factor
        Tspan_ext = self.data.pta_tspan + 2. * window_ext
        first_toa = np.min([np.min(psr.toas) for psr in self.data.psrs])
        last_toa = np.max([np.max(psr.toas) for psr in self.data.psrs])
        sparse_toas_det = np.array([np.linspace(first_toa - window_ext, last_toa + window_ext,
                                                self.num_coeff_det + 2, endpoint=False)
                                    for _ in range(self.data.npsrs)])
        self.sparse_toas_det_jax = jnp.array(sparse_toas_det)
        sparse_toas_scaled_shifted_np = [(sparse_toas - dutils.tref) * dutils.cw_renorm
                                         for sparse_toas in sparse_toas_det]
        self.sparse_toas_shifted_scaled = jnp.array(sparse_toas_scaled_shifted_np)
        self.Nsparse = sparse_toas_det.shape[1]
        self.freqs_forFFT = jnp.array([np.fft.fftfreq(self.Nsparse, Tspan_ext / self.Nsparse)
                                       for _ in range(self.data.npsrs)])
        self.Tukey_det = jnp.array(tukey(self.Nsparse, alpha=(Tspan_ext - self.data.pta_tspan) / Tspan_ext))

        freqs_for_Fmat = jnp.array([self.freqs_forFFT[0, j + 1] for j in range(self.nfreqs_det)])
        Fs_det = [sutils.get_fourier_design_matrix(psr.toas, freqs_for_Fmat)
                       for psr in self.data.psrs]
        self.Fs_det_concat = jnp.concat(Fs_det, axis=0)


    def update_white_matrix_products_unjitted(self, N_list, white_noise_params, reff):
        """Get the helper objects for likelihood evaluation.

        This method computes the helper objects TNT, TNr, rNr, and logdet_N for each pulsar, which are
        needed for the likelihood evaluation and posterior drawing. The TNT, TNr, rNr, and logdet_N
        are computed as:
        - TNT = T^T N^{-1} T
        - TNr = T^T N^{-1} r
        where T is the Fourier design matrix for each pulsar, N is the white noise
        covariance matrix for each pulsar, and r is the effective residuals.

        This overrides :meth:`SuperSignal.update_white_matrix_products_unjitted` to also
        return the deterministic-basis products (FDNFD, FDNr, FNFD) required by the joint
        posterior.

        Parameters
        ----------
        N_list : list of Atlas.nMatrix.base.Base_TOA_cov
            The white noise covariance matrices for each pulsar.
        reff : list of arrays
            The effective residuals for each pulsar. [npsr, npsr_toas]

        Returns
        -------
        tuple
            The helper objects (TNT, TNr, rNr, and logdet_N) for each pulsar.
            [npsr, nmode, nmode], [npsr, nmode]
        """
        return N_list.get_red_det_helpers(red_noise_basis = self.get_Fmat_concat,
                                          det_signal_basis = self.Fs_det_concat,
                                          residuals = reff,
                                          white_noise_params = white_noise_params) # [FNF, FNr, rNr, logdetN, FDNFD, FDNr, FNFD]

    def get_coeffs_via_FFT(self, det_params, psr_phases, psr_dists):
        """
        Mapping from deterministic signal parameters to Fourier space.
        This is simply a FFT.

        Note the frequency bins used here are generally different than those used
        by stochastic models. To avoid Gibbs phenomena from non-periodic deterministic
        signals over Tspan, we use an "extended" basis where we FFT the signal over a period
        extended either side of Tspan after applying a Tukey window. The Fourier design matrix,
        however, maps the Fourier coefficients to TOAs **within** the PTA Tspan. This introduces
        cross terms in the posterior. See Gundersen & Cornish 2025.

        Parameters
        ----------
        det_params : array
            Parmeters values of the deterministic model.
        psr_phases : array or None
            The phase of the gravitational wave at each pulsar. This is a model parameter
            for continuous gravitational waves from individual SMBHBs. For other deterministic
            models, this can be None.
        psr_dists : array or None
            The distance to each pulsar [kpc]. This is a model parameter for continuous
            gravitational waves from individual SMBHBs. For other deterministic models,
            this can be None.

        Returns
        -------
        coeff : array
            A (npsrs, 2*nfreq) array where npsrs is the number of pulsars in the array and
            nfreq is the number of frequency bins used to represent to the deterministic
            model in Fourier space.
        """

        # get timing delays induced by the deterministic signal over "sparse" (evenly-spaced) TOAs
        det_residuals = self.get_delays_func(self.sparse_toas_shifted_scaled, self.data.psr_pos,
                                                    det_params, psr_phases, psr_dists)
        # window residuals over extended observation
        det_residuals_windowed = self.Tukey_det * det_residuals
        # do FFT
        det_fft = jnp.fft.fft(det_residuals_windowed, n=None, axis=-1, norm=None)  # dim (Np, 2 * Nf + 2)

        # apply time shift to set initial time
        det_fft *= jnp.exp(-1.j * 2 * jnp.pi * self.freqs_forFFT * self.sparse_toas_det_jax[:, 0:1])

        # extract sine and cosine coefficients
        a_n = jnp.imag(det_fft[:, :self.Nsparse // 2]) * (-2 / self.Nsparse)  # (Np, Nf + 1)
        b_n = jnp.real(det_fft[:, :self.Nsparse // 2]) * (2 / self.Nsparse)  # (Np, Nf + 1)

        # interweave sine/cosine coefficients and reshape to (Np, 2 * Nf)
        coeff = jnp.concatenate((a_n, b_n), axis=1).reshape((self.data.npsrs, 2, self.nfreqs_det + 1))\
                        .transpose((0, 2, 1)).reshape((self.data.npsrs, self.num_coeff_det + 2))

        return coeff[:, 2:]  # remove DC


    def _wrap_delays_func(self, func, with_psr_params):
        """
        If a deterministic model is supplied that does **not**
        take pulsar distance and phase as model parameters,
        then the 'get_delay_func' is wrapped to accept these
        parameters anyway and the sampler will supply NoneTypes
        for these inputs.
        """
        if with_psr_params:
            return func
        else:
            def wrapped(toas, psrpos, params, psr_phases, psr_dists):
                return func(toas, psrpos, params)
            return wrapped


    @jit_method
    def lnposterior_reparam(self, helpers, red_params, det_params, psr_phases, psr_dists, z):
        """
        This method evaluates the posterior under a reparameterization of the Fourier
        coefficients. The coefficients represent Gaussian processes described by phi_cube.
        Call this method within gradient-based samplers.

        This overrides :meth:`SuperSignal.lnposterior_reparam` to jointly evaluate the
        stochastic and deterministic contributions (it consumes the extended helper tuple
        and folds in the deterministic Fourier coefficients).
        NOTE: the reparameterization is based on the posterior of the coeffcients.
        Parameters
        ----------
        helpers : tuple
            The helper objects (TNT, TNr, rNr, logdet_N) for each pulsar.
            [npsr, nmode, nmode], [npsr, nmode]
        red_noise_cov : array
            The red noise covaraince matrix
            over FREQUENCY! [nfreqs, npsr, npsr]
        z : array
            "Whitened coefficients", [npsr, nmode]
        Returns
        -------
        tuple
            Fourier coefficients with variance imposed by spectral model [npsr, nmodes] and
            the log-determinant of the Jacobian of the coordinate transformation [float].
        """
        TNT, TNr, rNr, logdet_N, TDNTD, TDNr, TNTD = helpers
        red_noise_cov = self.model.get_phi_mat_full(red_params)
        if self.npsrs == 1:
            phiinvs_diags = jnp.repeat(1/red_noise_cov, 2, axis = 0) #[nmodes, npsrs]
            logdet_phimat = 2 * jnp.sum(jnp.log(red_noise_cov)) #2 is to account for 2*nfreq=nmodes
        else:
            phiinvs, logdet_phimat = self.model.get_phi_mat_inv(red_noise_cov)
            phiinvs_diags = phiinvs.diagonal(axis1 = -2, axis2 = -1) #[nmodes, npsrs]

        if self.linear_timing and not self.marg_tm:
            phiinvs_diags_ltm = jnp.full(shape = (self.nmodes, self.npsrs),
                                         fill_value = self.lowest_value_eq_to_zero)
            phiinvs_diags = phiinvs_diags_ltm.at[self.linear_timing_model_size:, :].add(phiinvs_diags)
            # set prior variance of padded parameters to one for stable transformation
            phiinvs_diags = phiinvs_diags.at[:self.linear_timing_model_size, :].add(self._pad_mask.T)

        # get Fourier coefficients of deterministic signal
        a_det = self.get_coeffs_func(det_params, psr_phases, psr_dists)

        # inner product needed for likelihood and standardizing transformation
        TNTDas = jax.vmap(lambda x, y: jnp.dot(x, y))(TNTD, a_det)

        # Posterior precision Cholesky (cho_factor equivalent), batched over pulsars
        Sigma_inv = TNT.at[:, self._diag_idx , self._diag_idx ].add(phiinvs_diags.T)     # [npsr, nmodes, nmodes]
        Sigma_inv_L = jsl.cho_factor(Sigma_inv, lower = True)  # [npsr, nmodes, nmodes]

        # MAP coefficients via cho_solve pattern: forward then back substitution
        a_hat = jsl.cho_solve(Sigma_inv_L, TNr[..., None] - TNTDas[..., None])

        # Standardizing transform via back substitution
        Lz = jax.lax.linalg.triangular_solve(
            Sigma_inv_L[0], z[..., None], left_side=True, lower=True, transpose_a=True,
        )  # L^T Lz = z

        coeff = a_hat + Lz  # [npsr, nmodes, 1]

        lndet_Jac = -jnp.sum(jnp.log(Sigma_inv_L[0].diagonal(axis1=-2, axis2=-1)))

        # Log-likelihood
        aFNr      = jnp.sum(coeff[..., 0] * TNr)
        aFNFa     = jnp.sum(coeff.mT @ TNT @ coeff)
        lnlike_value = aFNr - 0.5 * aFNFa

        if self.npsrs == 1:
            lnprior_value = -0.5 * ((coeff[:, self.linear_timing_model_size:, 0]**2 * phiinvs_diags[self.linear_timing_model_size:, :].T).sum() + logdet_phimat)
        else:
            aG = coeff[:, self.linear_timing_model_size:] #[npsr, 2 * nfreq, 1]
            lnprior_value = -0.5 * ((aG.transpose(1, 2, 0) @ phiinvs @ aG.transpose(1, 0, 2)).sum() + logdet_phimat)

        if self.linear_timing and not self.marg_tm:
            # add probability density for padded (i.e. zero-ed) timing model parameters for HMC sampler
            # these parameters do not impact the likelihood, prior, and are uncorrelated with all other parameters
            # so this should not effect parameter estimation, but merely provides some curvature for HMC to latch
            # onto when sampling
            padded_logpdf = -0.5 * jnp.sum((self._pad_mask * coeff[:, :self.linear_timing_model_size, 0])**2)
        else:
            padded_logpdf = 0.

        # continuous wave contribution
        lnlike_det_add = jnp.sum(jax.vmap(lambda x, y: jnp.dot(x, y))(a_det, TDNr)) \
        + -jnp.sum(jax.vmap(lambda x, y: jnp.dot(x, y))(coeff[..., 0], TNTDas)) \
        + -0.5 * jnp.sum(jax.vmap(lambda x, y: jnp.dot(x, jnp.dot(y, x)))(a_det, TDNTD))

        # additional density
        if self.additional_ln_factor is not None:
            add_lnpdf = self.additional_ln_factor(red_params, det_params, psr_phases, psr_dists, z)
        else:
            add_lnpdf = 0.

        
        log_density = lnlike_value + lnprior_value + lndet_Jac - 0.5 * (rNr + logdet_N) \
            + padded_logpdf + lnlike_det_add + add_lnpdf
        
        return log_density, coeff[..., 0]
