
from ATLAS.utils import jit_method, jit
from ATLAS.signals import signals_utils as sutils
from ATLAS.signals.factorized.utils import make_irn_model
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.random as jrandom
from ATLAS import parameterized
from ATLAS.psd_functions import free_spectrum, make_free_spectrum
from tqdm import trange
from functools import partial
import numpy as np
import random
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from tqdm_joblib import ParallelPbar
from joblib import delayed
from functools import cached_property

class Red:
    """A signal class for a factorized likelihood (not prior).

    The frequencies are the first `nfreqs` harmonics of 1/Tspan. Tspan could either
    be the PTA timespan or the individual pulsar timespans. Supplying `user_freqs`
    overwrites this.

    Attributes
    ----------
    name : str
        The name of the signal.
    init_params : dict
        The parameters used for initialization, which can be useful for re-initialization.
    parameter_names : list of str
        A list of parameter names corresponding to the parameters of the signal.
    n_parameters : int
        The number of parameters in the signal.
    parameter_range : array
        An array of the lower and upper bounds for each parameter [n_parameters, 2].
    allow_posterior_draw : bool
        Whether to allow posterior draws for this signal.
    sampling_method : str
        The sampling method to use for the signal.
    initialized : bool
        Whether the signal has been fully initialized with data.
    psr_toas : list of arrays
        A reference to the list of TOA arrays for each pulsar [npsr, npsr_toas].
    nfreqs : int
        The number of frequency bins in the free spectrum.
    nmodes : int
        The number of modes in the Fourier design matrix (2*nfreqs for sine and cosine).
    npsrs : int
        The number of pulsars in the dataset.
    use_pulsar_tspan : bool
        Whether to use individual pulsar timespans for frequency calculation, or the PTA timespan.
    tspans : array or float
        The timespans used for frequency calculation. Either scalar or [npsrs]
    freqs : array
        The frequencies used in the Fourier design matrix. Either [nfreqs] or [npsrs, nfreqs]
    log_prior_volume : float
        The log of the prior volume for the parameters, used for uniform priors.
    fixed_wn : bool
        Whether the white noise is fixed, which can allow for optimization in computations.
    _psd_range : array
        The range of the power spectral density in linear space, used for computations.
    _diag_idx : array
        An array of diagonal indices for the Sigma matrix, used for efficient updates.
    """
    def __init__(self,
                 name,
                 data,
                 psd_function,
                 lower_bound_psd,
                 upper_bound_psd, 
                 nfreqs=10, 
                 halflog10_rho_range=(-9,-2), 
                 use_pulsar_tspan=False,
                 posterior_draw_ndraws = 1,
                 user_freqs = jnp.array([False]),
                ):
        """The constructor for the IRN_Freespectrum signal class.

        This signal models the intrinsic red noise as a pulsar-independent signal 
        in all pulsars. The power spectrum is modeled as a free spectrum with nfreqs 
        frequencies. The parameters are given as halflog10_rho, which is defined as
        - <a^T a> = rho^2 -> halflog10_rho = 0.5 * log10(<a^T a>)
        This means that the units are log(seconds). 

        The frequencies are the first `nfreqs` harmonics of 1/Tspan. Tspan is either
        the PTA timespan or the individual pulsar timespans, depending on the
        `use_pulsar_tspan` flag.

        If data is not provided, the signal use a simple initialization (see 
        Atlas.signals.base.Signal_Base).

        Parameters
        ----------
        name : str
            The name of the signal.
        data : Atlas.data.Data.PTA_Data
            Atlas data object.
        nfreqs : int, optional
            The number of frequency bins in the free spectrum, by default 10.
        halflog10_rho_range : tuple, optional
            The range of the halflog10_rho parameters, by default (-9,-2).
        use_pulsar_tspan : bool, optional
            Whether to use individual pulsar timespans for frequency calculation, 
            or the PTA timespan, by default False.
        user_freqs: arr, optional
            Frequency bins provided by user to use instead of i/T harmonics.
        """
        # PSD reparameterization
        self.psd_function, self.psd_reparam_helper = make_irn_model(psd_function, 
                                        lower_bound_array = lower_bound_psd, 
                                        upper_bound_array = upper_bound_psd)
        
        self.name = name
        self.data = data
        self.ndraws = posterior_draw_ndraws # Number of posterior draws to run in parallel when sampling_method is 'posterior_draws'

        # Helper attributes needed for the rest of the class--------------------
        self.psr_toas = self.data.toas # Reference to a list of TOA arrays for each pulsar [npsr, npsr_toas]
        self.nfreqs = nfreqs
        self.nmodes = 2*nfreqs # Sine and cosine modes per frequency
        self.npsrs = self.data.npsrs 

        self.use_pulsar_tspan = use_pulsar_tspan # Use pulsar tspan, or PTA tspan?
        if user_freqs.any():
            self.freqs = user_freqs
            self.nfreqs = len(self.freqs)
            self.nmodes = 2 * self.nfreqs
        else:    
            if use_pulsar_tspan:
                self.tspans = data.psr_tspans # Array of individual pulsar timespans [npsrs]
                # Frequencies for this signal (Different frequencies for each pulsar)
                self.freqs = []
                for i in range(self.npsrs):
                    f = sutils.get_harmonic_frequencies(self.nfreqs, self.tspans[i]) # [nfreqs]
                    self.freqs.append(f)
                self.freqs = jnp.array(self.freqs) # [npsrs, nfreqs]
            else:
                self.tspans = data.pta_tspan
                # Frequencies for this signal (Same frequencies for all pulsars)
                self.freqs = sutils.get_harmonic_frequencies(self.nfreqs, self.tspans) # [nfreqs]  

        # length of parameters
        self.n_parameters = len(self.freqs) # Number of parameters
        par_range = jnp.ones((self.n_parameters, 2)) * jnp.array(halflog10_rho_range)[None,:]
        self.parameter_range = par_range # Range for each parameter, shape (n_parameters, 2)

        # Uniform prior volume
        diff = par_range[:,1] - par_range[:,0] # high - low
        self.log_prior_volume = jnp.sum( jnp.log(diff) ) 
        
        # extract data analysis settings from data object
        self.fixed_wn = data.fixed_wn
        self.fixed_res = data.fixed_res
        self.linear_timing = data.linear_timing

        # function to get helper objects for likelihood
        self.get_helpers = self._get_helpers()

        # Hidden attributes if needed-------------------------------------------
        # PSD range in linear space for computations.
        self._psd_range = 10**(2*jnp.array(halflog10_rho_range, dtype=float)) 
        self._diag_idx = jnp.arange(self.nmodes)
        
        self.get_red_basis = jnp.concat(self.get_basis()) # [n_toas, nmodes]

    # Helper methods------------------------------------------------------------
    @jit_method
    def get_basis(self):
        """Get the list Fourier design matrix for all pulsars. [npsr, npsr_toas, nmodes]

        This helper method computes the Fourier design matrix for each pulsar based 
        on the TOAs and the frequencies. Each Fourier design matrix has dimensions
        [npsr_toas, nmodes]. Ordered as sine and cosine for each frequency:
        i.e. columns are ordered as [sin(2pi f1 t), cos(2pi f1 t), sin(2pi f2 t), cos(2pi f2 t), ...]

        Returns
        -------
        List of arrays
             The list of Fourier design matrices for each pulsar. [npsr, npsr_toas, nmodes]
        """
        if self.use_pulsar_tspan:
            # self.freqs is [npsr, nfreqs]
            T = [sutils.get_fourier_design_matrix(t, f, microseconds=False)
                 for t,f in zip(self.psr_toas, self.freqs)] # List of [npsr_toas, nmodes]
        else:
            # self.freqs is [nfreqs], same for all pulsars
            T = [sutils.get_fourier_design_matrix(t, self.freqs, microseconds=False)
                 for t in self.psr_toas] # List of [npsr_toas, nmodes]
        return T # [npsr, npsr_toas, nmodes]
    

    def _get_helpers(self):
        """This helper method returns a jit-ed function to calculate TNT, TNr, rNr, logdet_N objects
        needed for likelihood evaluation. Data analysis settings are extracted from the
        data object.

        Returns
        -------
        new_func: callable
            Function which return TNT, TNr, rNr, logdet_N, etc. helper arrays.
        """

        if self.fixed_wn and not self.fixed_res:
            new_func = partial(self.update_white_matrix_products_unjitted,
                               N_list = self.data.Nmat,
                               white_noise_params = self.data.fixed_white_noise_params,
                               )
            return jit(new_func)

        elif not self.fixed_wn and not self.fixed_res:
            new_func = partial(self.update_white_matrix_products_unjitted,
                               N_list = self.data.Nmat,
                               )
            return jit(new_func)

        elif not self.fixed_wn and self.fixed_res:
            new_func = partial(self.update_white_matrix_products_unjitted,
                               N_list = self.data.Nmat,
                               reff = jnp.concat(self.data.raw_residuals)[:, None],
                               )
            return jit(new_func)

        elif self.fixed_wn and self.fixed_res:
            new_func = partial(self.update_white_matrix_products_unjitted,
                               N_list = self.data.Nmat,
                               white_noise_params = self.data.fixed_white_noise_params,
                               reff = jnp.concat(self.data.raw_residuals)[:, None],)
            return jit(new_func)

    @jit_method
    def get_phi_diag(self, params):
        """Get the diagonal of the phi matrix from the parameters. [npsr, nmodes]

        This helper method transforms the input parameters (npsr*halflog10_rhos) 
        into the diagonal of the phi matrix (fourier coefficients covariance) for 
        each pulsar. Note that this only returns the diagonal, meaning assuming 
        a stationary process.

        Parameters
        ----------
        params : array
            The input parameters for the signal [npsr*nfreqs]

        Returns
        -------
        array
            The diagonal of the phi matrix for each pulsar. [npsr, nmodes]
        """
        # params is [nfreqs*npsrs], convert to [npsrs, nfreqs]
        halflog10_rho = sutils.sigmaVec2blockVec(params, self.npsrs) # [npsrs, nfreqs]
        # Convert to PSD values and diagonal matrices
        phi_diag = jnp.repeat( 10**(2*halflog10_rho), 2, axis=1 ) # [npsrs, 2*nfreqs]
        return phi_diag # [npsrs, 2*nfreqs]

    @jit_method
    def get_sigma(self, TNT, phi_diag):
        """Get the per-pulsar fourier coefficient covariance matrix Sigma. [npsr, nmodes, nmodes]

        This helper method computes the per-pulsar covariance matrix Sigma for the
        fourier coefficients. This is given by Sigma = TNT + phi^{-1}, where
        TNT is the contribution from the white noise and phi^{-1} is the contribution
        from the IRN signal.

        Note that this implicitly assumes that the each pulsar is independent.

        Parameters
        ----------
        TNT : array
            The contribution from the white noise for each pulsar. [npsr, nmodes, nmodes]
        phi_diag : array
            The diagonal of the phi matrix for each pulsar. [npsrs, nmodes]

        Returns
        -------
        array
            The covariance matrix Sigma for each pulsar. [npsr, nmodes, nmodes]
        """
        # Since each pulsar is independent, full Sigma is block diagonal, so we can
        # just make each individual pulsar's Sigma matrix instead
        phiinv = 1/phi_diag # [npsrs, 2*nfreqs]
        Sigma = TNT.at[:, self._diag_idx, self._diag_idx].add(phiinv) # [npsr, nmodes, nmodes]
        return Sigma # List of arrays [npsr, nmodes, nmodes]

    @jit_method
    def get_sigma_from_phiinv(self, TNT, phiinv):
        """Get the per-pulsar fourier coefficient covariance matrix Sigma. [npsr, nmodes, nmodes]

        This helper method computes the per-pulsar covariance matrix Sigma for the
        fourier coefficients. This is given by Sigma = TNT + phi^{-1}, where
        TNT is the contribution from the white noise and phi^{-1} is the contribution
        from the IRN signal. phi^{-1} is supplied by the user in this function.

        Note that this implicitly assumes that the each pulsar is independent.

        Parameters
        ----------
        TNT : array
            The contribution from the white noise for each pulsar. [npsr, nmodes, nmodes]
        phiinv : array
            The inverse of the red noise covaraince matrix. [npsrs, nmodes]

        Returns
        -------
        array
            The covariance matrix Sigma for each pulsar. [npsr, nmodes, nmodes]
        """
        # Since each pulsar is independent, full Sigma is block diagonal, so we can
        # just make each individual pulsar's Sigma matrix instead
        Sigma = TNT.at[:, self._diag_idx, self._diag_idx].add(phiinv) # [npsr, nmodes, nmodes]
        return Sigma # List of arrays [npsr, nmodes, nmodes]

    @jit_method
    def _get_coefficient_realization(self, helpers, params, key):
        """Get a realization of all pulsar fourier coefficients. [npsr, nmodes]

        This helper method generates a random realization of the pulsar fourier 
        coefficients given the helpers (TNT, TNr, rNr, and logdet_N), the parameters (halflog10_rhos),
        and a random key. The realization is drawn from a multivariate normal
        distribution such that:
        - covariance -> Sigma = TNT + phi^{-1} 
        - mean = Sigma^{-1} TNr

        Parameters
        ----------
        helpers : tuple
            The helpers (TNT, TNr, rNr, and logdet_N). [npsr, nmodes, nmodes], [npsr, nmodes]
        params : array
            The input parameters for the signal [npsr*nfreqs]
        key : jax.random.PRNGKey
            The random key for generating the realization.

        Returns
        -------
        array
            A realization of the fourier coefficients for each pulsar. [npsr, nmodes]
        """
        TNT, TNr, rNr, logdet_N = helpers # Unpack helpers [npsr, nmode, nmode], [npsr, nmode]
        phi_diag = self.get_phi_diag(params) # Get phi diagonal from params [npsr, nfreqs]
        
        Sigma = self.get_sigma(TNT, phi_diag) # [npsr, nmode, nmode]
        cfs = jsl.cho_factor(Sigma) # Cholesky for each psr [npsr, nmodes, nmodes]

        # Get the mean (covariance is Sigma)
        mean = jsl.cho_solve(cfs, TNr[:,:,None]) # [npsr, nmodes, 1] 

        # Transform a unit mean Gaussian random variable to desired distribution
        U = jrandom.normal(key, shape=(self.npsrs, self.nmodes, self.ndraws)) # [npsr, nmodes, ndraws]
        # Project U into the desired distribution
        coef = (mean + jsl.solve_triangular(cfs[0], U)) # [npsr, nmodes, ndraws]
        return coef # [npsr, nmodes]
    
    # Required methods----------------------------------------------------------
    def update_white_matrix_products_unjitted(self, N_list, white_noise_params, reff):
        """Get the helper objects for likelihood evaluation.

        This method computes the helper objects TNT, TNr, rNr, and logdet_N for each pulsar, which are
        needed for the likelihood evaluation and posterior drawing. The TNT, TNr, rNr, and logdet_N
        are computed as:
        - TNT = T^T N^{-1} T
        - TNr = T^T N^{-1} r
        where T is the Fourier design matrix for each pulsar, N is the white noise
        covariance matrix for each pulsar, and r is the effective residuals.

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
        return N_list.get_red_helpers(red_noise_basis = self.get_red_basis, 
                                      residuals = reff, 
                                      white_noise_params = white_noise_params) # [FNF, FNr, rNr, logdetN]

    # Required methods----------------------------------------------------------
    @jit_method
    def ln_prior_freespec(self, params):
        """Compute the log-prior for the IRN signal parameters.

        This method computes the log of the uniform prior on the IRN parameters.
        All parameters are uniform priors with the range specified by 
        self.parameter_range.

        Parameters
        ----------
        params : array
             The input parameters, which are halflog10_rhos for each frequency
             for each pulsar. [npsr*nfreqs]

        Returns
        -------
        float
            The log-prior for the  signal parameters. [1]
        """
        state = jnp.logical_and(params >= self.parameter_range[:,0],
                                params <= self.parameter_range[:,1]).all()
        lnprior = jnp.where(state, -self.log_prior_volume, -jnp.inf)
        return lnprior # scalar


    @jit_method
    def prior_draw_freespec(self, key):
        """Draw a set of parameters from the prior distribution for the IRN signal.

        This method generates a random draw from the uniform prior distribution on the
        IRN parameters. The parameters are drawn uniformly from the range specified by
        self.parameter_range.

        Parameters
        ----------
        key : jax.random.PRNGKey
             The random key for generating the draw.

        Returns
        -------
        array
            The drawn parameters. [n_parameters]
        """
        params = jrandom.uniform(key, shape=(self.n_parameters,), 
                                 minval=self.parameter_range[:,0], 
                                 maxval=self.parameter_range[:,1]) # [n_parameters]
        return params # [n_parameters]


    @jit_method
    def posterior_draw_freespec(self, helpers, params, key):
        """Draw a set of parameters from the posterior distribution for the IRN signal.

        This method generates a random draw from the posterior distribution on the 
        IRN parameters. This method is often called "gibbs sampling" in the literature,
        but since we are using blocked-gibbs sampling, we opt to label it as "drawing
        from the posterior" instead. Since the fourier coefficients distribution for
        a set of PSDs is analytically known and the distribution of PSDs is known
        for a set of fourier coefficients, we can directly draw from the posterior
        using this method.

        Parameters
        ----------
        helpers : tuple
            The helper objects (TNT, TNr, rNr, and logdet_N) for each pulsar. 
            [npsr, nmode, nmode], [npsr, nmode]
        params : array
             The current parameters, which are halflog10_rhos for each frequency
             for each pulsar. [npsr*nfreqs]
        key : jax.random.PRNGKey
            The random key for generating the draw.

        Returns
        -------
        array            
            The drawn parameters from the posterior distribution. [npsr*nfreqs]
        """
        key1, key2 = jrandom.split(key)
        # Get a realization of the coefficients
        coef = self._get_coefficient_realization(helpers, params, key1) # [npsr, nmodes, ndraws]
        
        # Sum sine and cosine elements
        beta = 0.5*(coef[:,::2]**2 + coef[:,1::2]**2) # [npsr, nfreqs, ndraws]

        # Get exponential terms
        low = jnp.exp(-beta/self._psd_range[0]) # [npsr, nfreqs, ndraws]
        high = jnp.exp(-beta/self._psd_range[1]) # [npsr, nfreqs, ndraws]

        # Uniform values
        U = jrandom.uniform(key2, shape=low.shape, 
                            minval=low, 
                            maxval=high) # [npsr, nfreqs, ndraws]
        
        # Compute equation B3 from Laal et al. 
        new_params = -beta / jnp.log(U) # [npsr, nfreqs, ndraws]

        # Convert back to halflog10_rho parameters
        halflog10_rho = 0.5*jnp.log10(new_params).reshape(-1, self.ndraws) # [npsr*nfreqs, ndraws]
        return halflog10_rho.T # [ndraws, nfreqs*npsrs]

    @jit_method
    def posterior_draw_from_coeff(self, coef, key):
        """Draw a set of parameters from the posterior distribution for the IRN signal.

        This method generates a random draw from the posterior distribution on the 
        IRN parameters. This method is often called "gibbs sampling" in the literature,
        but since we are using blocked-gibbs sampling, we opt to label it as "drawing
        from the posterior" instead. Since the fourier coefficients distribution for
        a set of PSDs is analytically known and the distribution of PSDs is known
        for a set of fourier coefficients, we can directly draw from the posterior
        using this method.

        Parameters
        ----------
        helpers : tuple
            The helper objects (TNT, TNr, rNr, and logdet_N) for each pulsar. 
            [npsr, nmode, nmode], [npsr, nmode]
        params : array
             The current parameters, which are halflog10_rhos for each frequency
             for each pulsar. [npsr*nfreqs]
        key : jax.random.PRNGKey
            The random key for generating the draw.

        Returns
        -------
        array            
            The drawn parameters from the posterior distribution. [npsr*nfreqs]
        """
        key1, key2 = jrandom.split(key)
        # Get a realization of the coefficients
        
        # Sum sine and cosine elements
        beta = 0.5*(coef[:,::2]**2 + coef[:,1::2]**2) # [npsr, nfreqs, ndraws]

        # Get exponential terms
        low = jnp.exp(-beta/self._psd_range[0]) # [npsr, nfreqs, ndraws]
        high = jnp.exp(-beta/self._psd_range[1]) # [npsr, nfreqs, ndraws]

        # Uniform values
        U = jrandom.uniform(key2, shape=low.shape, 
                            minval=low, 
                            maxval=high) # [npsr, nfreqs, ndraws]
        
        # Compute equation B3 from Laal et al. 
        new_params = -beta / jnp.log(U) # [npsr, nfreqs]

        # Convert back to halflog10_rho parameters
        halflog10_rho = 0.5*jnp.log10(new_params).reshape(-1, self.ndraws) # [npsr*nfreqs, nfreqs]
        return halflog10_rho.T # [ndraws, nfreqs*npsrs]

class GaussianTiming:
    """A signal class for a factorized likelihood (not prior) for the timing model.

    Attributes
    ----------
    name : str
        The name of the signal.
    init_params : dict
        The parameters used for initialization, which can be useful for re-initialization.
    parameter_names : list of str
        A list of parameter names corresponding to the parameters of the signal.
    n_parameters : int
        The number of parameters in the signal.
    parameter_range : array
        An array of the lower and upper bounds for each parameter [n_parameters, 2].
    allow_posterior_draw : bool
        Whether to allow posterior draws for this signal.
    sampling_method : str
        The sampling method to use for the signal.
    initialized : bool
        Whether the signal has been fully initialized with data.
    psr_toas : list of arrays
        A reference to the list of TOA arrays for each pulsar [npsr, npsr_toas].
    nfreqs : int
        The number of frequency bins in the free spectrum.
    nmodes : int
        The number of modes in the Fourier design matrix (2*nfreqs for sine and cosine).
    npsrs : int
        The number of pulsars in the dataset.
    use_pulsar_tspan : bool
        Whether to use individual pulsar timespans for frequency calculation, or the PTA timespan.
    tspans : array or float
        The timespans used for frequency calculation. Either scalar or [npsrs]
    freqs : array
        The frequencies used in the Fourier design matrix. Either [nfreqs] or [npsrs, nfreqs]
    log_prior_volume : float
        The log of the prior volume for the parameters, used for uniform priors.
    fixed_wn : bool
        Whether the white noise is fixed, which can allow for optimization in computations.
    _psd_range : array
        The range of the power spectral density in linear space, used for computations.
    _diag_idx : array
        An array of diagonal indices for the Sigma matrix, used for efficient updates.
    """
    def __init__(self,
                 name,
                 data,
                 nmodes,
                 gtm_psd = None,
                 lower_bound_psd = None,
                 upper_bound_psd = None,
                 timing_model = None,
                 basis = None,
                 z_scale = 100, 
                 num_trials_for_pca = 10,
                 pca_seed_per_pulsar = None
                ):
        """
        Parameters
        ----------
        name : str
            The name of the signal.
        data : Atlas.data.Data.PTA_Data
            Atlas data object.
        """
        if nmodes % 2 != 0:
            raise ValueError(f"Expected an even integer for `nmodes`, got {nmodes}.")
        self.nmodes = nmodes
        self.nfreqs = int(nmodes/2)
        self.gtm_psd = gtm_psd
        
        # PSD reparameterization
        if lower_bound_psd is None and upper_bound_psd is None:
            psd_function = make_free_spectrum(self.nfreqs)
            param_names = [f"halflog10_rho_{i}" for i in range(self.nfreqs)]
            fixed_kwargs = {param_names[i]: v for i, v in zip(jnp.arange(self.nfreqs), jnp.zeros(self.nfreqs))}
            self.psd_function, self.psd_reparam_helper = make_irn_model(partial(psd_function, **fixed_kwargs), 
                                            lower_bound_array = jnp.array([]), 
                                            upper_bound_array = jnp.array([]))
        else:
        # PSD reparameterization
            self.psd_function, self.psd_reparam_helper = make_irn_model(free_spectrum, 
                                            lower_bound_array = lower_bound_psd, 
                                            upper_bound_array = upper_bound_psd)
                                            
        self.name = name
        self.data = data
        self.z_scale = z_scale
        self.num_trials_for_pca = num_trials_for_pca

        if pca_seed_per_pulsar is None:
            self.pca_seeds = jrandom.split(jrandom.key(random.randint(0, 81982)), data.npsrs)
        else:
            self.pca_seeds = pca_seed_per_pulsar

        if basis is None:
            self.tm_model = timing_model
            self.U, self.explained_variance_ratio, self.timing_residual_training_set = self._get_basis()

        else:
            self.U = basis

        self.nfreqs = int(nmodes/2)
        self.freqs = jnp.ones(self.nfreqs)

    
    def get_basis(self):
        return self.U

    def _fit_pulsar(self, tm_residuals_prior):

        mask = (
            np.isfinite(tm_residuals_prior).all(axis=-1)
            # & ((tm_residuals_prior > -10) & (tm_residuals_prior < 10)).all(axis=-1)
        )
        tm_residuals_prior = tm_residuals_prior[mask]

        if tm_residuals_prior.shape[0] < self.nmodes:
            return None
        else:
            svd = PCA(
                n_components=self.nmodes,
                whiten=True,
                random_state=random.randint(0, 18971),
            )
            svd.fit(tm_residuals_prior)
            U = svd.components_.T * np.sqrt(svd.explained_variance_)[None, :]
            
            return (
                U,
                svd.explained_variance_ratio_.sum(),
            )

    def _get_basis(self):

        timing_bases = []
        explained_variance_ratio = []

        pbar = trange(self.data.npsrs)
        for pidx in pbar:
            pbar.set_description(f"Generating prior samples and performing PCA for {self.data.psr_names[pidx]}")
            
            for num_iters in range(self.num_trials_for_pca):
                timing_residual_training_set = self.tm_model.sample_training_residuals(key = self.pca_seeds[pidx], 
                                                                                        num_samples = int(1e4),
                                                                                        z_scale = self.z_scale,
                                                                                        pulsar_index = pidx)
                ans = self._fit_pulsar(timing_residual_training_set)
                if ans is None:
                    print('Not enough samples in the prior. Shrinking the prior...')
                    self.z_scale = self.z_scale/10
                    if num_iters == self.num_trials_for_pca - 1:
                        raise ValueError(f"Cannot find prior samples for {self.data.psr_names[pidx]}")

                    continue
                    
                else:
                    timing_bases.append(ans[0])
                    explained_variance_ratio.append(ans[1])
                    break

        return timing_bases, explained_variance_ratio, timing_residual_training_set

class SuperSignal:
    """A signal class for a pulsar-independent free spectrum red noise (IRN) signal.

    This signal models the intrinsic red noise as a pulsar-independent signal
    in each pulsar. The power spectrum is modeled as a free spectrum with nfreqs 
    frequency bins. The parameters are given as halflog10_rho, which is defined as
    - <a^T a> = rho^2 -> halflog10_rho = 0.5 * log10(<a^T a>)
    This means that the units are log(seconds).

    The frequencies are the first `nfreqs` harmonics of 1/Tspan. Tspan could either
    be the PTA timespan or the individual pulsar timespans.

    Attributes
    ----------
    name : str
        The name of the signal.
    init_params : dict
        The parameters used for initialization, which can be useful for re-initialization.
    parameter_names : list of str
        A list of parameter names corresponding to the parameters of the signal.
    n_parameters : int
        The number of parameters in the signal.
    parameter_range : array
        An array of the lower and upper bounds for each parameter [n_parameters, 2].
    allow_posterior_draw : bool
        Whether to allow posterior draws for this signal.
    sampling_method : str
        The sampling method to use for the signal.
    initialized : bool
        Whether the signal has been fully initialized with data.
    psr_toas : list of arrays
        A reference to the list of TOA arrays for each pulsar [npsr, npsr_toas].
    nfreqs : int
        The number of frequency bins in the free spectrum.
    nmodes : int
        The number of modes in the Fourier design matrix (2*nfreqs for sine and cosine).
    npsrs : int
        The number of pulsars in the dataset.
    use_pulsar_tspan : bool
        Whether to use individual pulsar timespans for frequency calculation, or the PTA timespan.
    tspans : array or float
        The timespans used for frequency calculation. Either scalar or [npsrs]
    freqs : array
        The frequencies used in the Fourier design matrix. Either [nfreqs] or [npsrs, nfreqs]
    log_prior_volume : float
        The log of the prior volume for the parameters, used for uniform priors.
    fixed_wn : bool
        Whether the white noise is fixed, which can allow for optimization in computations.
    _psd_range : array
        The range of the power spectral density in linear space, used for computations.
    _diag_idx : array
        An array of diagonal indices for the Sigma matrix, used for efficient updates.
    """
    def __init__(self,
                signal_list,
                signal_combination_string,
                data,
                ):
        """The constructor for the IRN_Freespectrum signal class.

        This signal models the intrinsic red noise as a pulsar-independent signal 
        in all pulsars. The power spectrum is modeled as a free spectrum with nfreqs 
        frequencies. The parameters are given as halflog10_rho, which is defined as
        - <a^T a> = rho^2 -> halflog10_rho = 0.5 * log10(<a^T a>)
        This means that the units are log(seconds). 

        The frequencies are the first `nfreqs` harmonics of 1/Tspan. Tspan is either
        the PTA timespan or the individual pulsar timespans, depending on the
        `use_pulsar_tspan` flag.

        If data is not provided, the signal use a simple initialization (see 
        Atlas.signals.base.Signal_Base).

        Parameters
        ----------
        signal_list : list,
            List of signal components.
        data : Atlas.data
            Atlas data object.
        """
        self.signal_combination_string = signal_combination_string
        self.data = data
        # extract data anlaysis settings from data object
        self.fixed_wn = self.data.fixed_wn
        self.fixed_res = self.data.fixed_res
        self.linear_timing = self.data.linear_timing
        self.marg_tm = self.data.marg
        self.Mmat = self.data.Mmat
        
        # linear timing model attributes
        if self.marg_tm:
            self.linear_timing_model_size = 0
        else:
            self.linear_timing_model_size = max([x.shape[-1] for x in self.Mmat]) if self.linear_timing else 0
        self.linear_timing = data.linear_timing
        self.lowest_value_eq_to_zero = 1e-40

        self.signal_map = {s.name: s for s in signal_list}
        self.has_unc = False; self.has_cor = False 
        self.has_dm = False; self.has_gtm = False
        self.has_det = False
        if 'cor' in self.signal_map.keys():
            self.has_cor = True
        if 'unc' in self.signal_map.keys():
            self.has_unc = True
        if 'dm' in self.signal_map.keys():
            self.has_dm = True
        if 'gtm' in self.signal_map.keys():
            self.has_gtm = True
        if 'det' in self.signal_map.keys():
            self.has_det = True
            self.det_signal = self.signal_map['det']

        self.get_Fmat_concat, self.signal_comb_idxs = self.build_basis(self.signal_combination_string, self.signal_map)
        self.chrom_idxs = self.signal_comb_idxs['dm'] if 'dm' in self.signal_comb_idxs.keys() else None

        self.nmodes = self.get_Fmat_concat.shape[-1]
        self.npsrs = self.data.npsrs

        # get helper arrays for likelihood
        self.get_helpers = self._get_helpers()

        # number of frequency bins for NumPyro interface
        self.nfreqs = int((self.nmodes - self.linear_timing_model_size)/2) # Sine and cosine modes per frequency

        # Hidden attributes if needed-------------------------------------------
        self._diag_idx = jnp.arange(self.nmodes)
        
        # (npsr, linear_timing_model_size) with ones where padded, zero else
        self._pad_mask_list = []
        for M in self.Mmat:
            mask_per_psr = jnp.ones((self.linear_timing_model_size,))
            mask_per_psr = mask_per_psr.at[:M.shape[1]].set(0.)
            self._pad_mask_list.append(mask_per_psr)
        self._pad_mask = jnp.array(self._pad_mask_list)
        
        self.eps_diag_idx = jnp.arange(self.linear_timing_model_size)

        self.model_maker()
        
    @jit_method     
    def update_red_basis(self, chrom_index):
        """        
        Update the red noise basis using a
        chromatic index per pulsar. 

        Args:
            chrom_index: The chromatic index per pulsar.
            The shape of the array must be (npsrs)

        Returns:
            the updated red noise basis
        """
        index = chrom_index[self.data.dm_exploder_idxs]
        DM = self.data.ref_over_radio_freqs ** index
        return self.get_Fmat_concat.at[:, self.chrom_idxs].multiply(DM[:, None])

    def padd_tm_design_matrix(self):
        """        
        Padd zeros to the timing model design matrix in places where 
        there is no parameter.

        Returns:
            list: padded timing model design matrix in the shape of
            (n_psr, linear_timing_model_size)
        """        
        padded_list = []
        for M in self.Mmat:
            padded = jnp.zeros((M.shape[0], self.linear_timing_model_size))
            padded = padded.at[:, :M.shape[1]].set(M)
            padded_list.append(padded)
        return padded_list
    
    def Tmaker(self, 
                Fmats, 
                padd_tm_design_matrix = True):
        """
        Build the T-matrix as the concatenated basis of 
        timing and red noise (in that order!). There is an
        option to pad to the timing basis of each pulsar with
        zeros so that all timing design matricies have the same size.
        This is more GPU friendly!

        Args:
            Fmats array: the red noise basis matrix (F-mat) for all pulsars
            padd_tm_design_matrix (bool, optional): Do you want to pad 
            the timing design matrix with zeros to extend its size to a 
            common size across pulsar? Defaults to True.

        Returns:
            aray: The T-matrix as[M, F]
        """                
        if padd_tm_design_matrix:
            Mmats  = self.padd_tm_design_matrix()
        else:
            Mmats = self.Mmat
        T = []
        for F, M in zip(Fmats, Mmats):
            T.append(jnp.concat((M, F), axis = -1))
        return T

    def build_basis(self, basis_string, signal_map):
        """
        Build the combined basis matrix and index slices for each signal.

        The timing model columns (if [T] prefix used) are prepended to the
        shared block only. Signal indices are offset accordingly so they
        correctly index into the full T-matrix columns.

        Parameters
        ----------
        basis_string : str
            e.g. "[T]:unc+cor->unc | cw"
        signal_map : dict
            Maps signal names to signal objects,
            e.g. {'unc': sig_unc, 'cor': sig_cor, 'cw': sig_cw}

        Returns
        -------
        Fmat : jnp.ndarray, shape (n_toas, total_cols)
            Full basis matrix [M | F_shared | F_sep1 | ...]
        signal_indices : dict[str, slice]
            Maps each signal name to its column slice in Fmat.
            For shared signals, slices index into the Fourier columns only
            (i.e. offset by linear_timing_model_size, width = sig.nmodes).
            For separate signals, same convention.
        """
        cfg = sutils.parse_basis_string(basis_string)

        include_timing = cfg['include_timing']
        shared_names   = cfg['shared_names']
        rep            = cfg['representative']
        separate_names = cfg['separate_names']
        order          = cfg['order']

        # Validate all names are in signal_map
        missing = set(order) - set(signal_map)
        if missing:
            raise ValueError(f"Signals {missing} not found in signal_map")

        tm_offset = self.linear_timing_model_size if include_timing else 0

        # --- Shared block ---
        if rep is not None:
            raw_basis = signal_map[rep].get_basis()
            shared_Fmat = (
                jnp.concat(self.Tmaker(raw_basis, padd_tm_design_matrix=True))
                if include_timing
                else jnp.concat(raw_basis)
            )
        else:
            shared_Fmat = None

        # --- Separate blocks (never include timing model) ---
        separate_Fmats = {
            name: jnp.concat(signal_map[name].get_basis())
            for name in separate_names
        }

        # --- Assemble in order ---
        Fmats          = []
        signal_indices = {}
        col            = tm_offset   # start after M columns

        # Track timing model indices
        if include_timing:
            signal_indices['timing'] = slice(0, tm_offset)

        shared_block_placed = False
        shared_start        = None

        for name in order:
            if name in shared_names:
                if not shared_block_placed:
                    Fmats.append(shared_Fmat)
                    shared_start        = col
                    shared_block_placed = True
                    col += shared_Fmat.shape[1] - tm_offset  # advance by F cols only
                # Each shared signal gets a slice of width nmodes within the shared block
                sig_nmodes = signal_map[name].nmodes
                signal_indices[name] = slice(shared_start, shared_start + sig_nmodes)
            else:
                F      = separate_Fmats[name]
                n_cols = F.shape[1]
                Fmats.append(F)
                signal_indices[name] = slice(col, col + n_cols)
                col += n_cols

        Fmat = jnp.concat(Fmats, axis=1)
        return Fmat, signal_indices

    def _get_helpers(self):
        """This helper method returns a jit-ed function to calculate TNT, TNr, rNr, logdet_N objects
        needed for likelihood evaluation. Data analysis settings are extracted from the
        data object.

        The T-matrix is passed to the jitted function as a *traced argument*, never
        bound into the partial. Binding it makes it a compile-time constant, and XLA
        then writes it into the executable as a literal -- twice over, since it also
        keeps a pre-tiled copy for the GEMM, and again for every epoch-gather that
        cannot be constant-folded once the white noise is a runtime value. On the
        NANOGrav 15-year set that inflated the compiled helper build to 9.5 GB, past
        what a 24 GB card will load. As an argument it is the buffer we already own.

        Returns
        -------
        callable
            Function which returns TNT, TNr, rNr, logdet_N, etc. helper arrays. It
            accepts an optional ``red_noise_basis`` to override the default T-matrix
            (used by the chromatic-index path, which rescales the DM columns).
        """
        bound = dict(N_list = self.data.Nmat)
        if self.fixed_wn:
            bound['white_noise_params'] = self.data.fixed_white_noise_params
        if self.fixed_res:
            bound['reff'] = jnp.concat(self.data.raw_residuals)[:, None]

        core = jit(partial(self.update_white_matrix_products_unjitted, **bound))

        def get_helpers(red_noise_basis = None, **kwargs):
            if red_noise_basis is None:
                red_noise_basis = self.get_Fmat_concat
            return core(red_noise_basis = red_noise_basis, **kwargs)

        # The jitted core, so callers (and the profiler) can lower/compile it with the
        # T-matrix as a genuine argument. NOTE: wrapping ``get_helpers`` in a further
        # jit re-captures the default T-matrix as a compile-time constant, which is
        # exactly what this change exists to avoid -- pass ``red_noise_basis``
        # explicitly from the outermost jitted function instead.
        get_helpers.core = core

        return get_helpers

    def model_maker(self):
        """Add a specific parameterization of the power spectral density based
        on Atlas' `parameterized.py`.

        Parameters
        ----------
        model: Atlas.parameterized object.
            An instantiation of a parameterized object.
        """
        model_kwargs = {'pulsar_names':self.data.psr_names}

        if self.has_unc:
            model_kwargs.update(
                irn_psd_func          = self.signal_map['unc'].psd_function,
                irn_helper_dictionary = self.signal_map['unc'].psd_reparam_helper,
                irn_bins              = self.signal_map['unc'].nfreqs,
                f_irn                 = self.signal_map['unc'].freqs,
            )

        if self.has_gtm:
            model_kwargs.update(
                gtm_psd_func          = self.signal_map['gtm'].psd_function,
                gtm_helper_dictionary = self.signal_map['gtm'].psd_reparam_helper,
                gtm_bins              = self.signal_map['gtm'].nfreqs,
                f_gtm                 = self.signal_map['gtm'].freqs,
                gtm_psd               = self.signal_map['gtm'].gtm_psd 
            )

        if self.has_dm:
            model_kwargs.update(
                dm_psd_func           = self.signal_map['dm'].psd_function,
                dm_helper_dictionary  = self.signal_map['dm'].psd_reparam_helper,
                dm_bins               = self.signal_map['dm'].nfreqs,
                f_dm                  = self.signal_map['dm'].freqs,
            )

        if not self.has_cor:
            model_kwargs.update(dict(
                Npulsars                 = self.data.npsrs,
                signal_indices           = self.signal_comb_idxs,
                linear_timing_model_size = self.linear_timing_model_size,
            )
            )
            self.model = partial(parameterized.PerPulsarRedNoise, **model_kwargs)()
            
        else:
            model_kwargs.update(dict(
                psr_pos                  = self.data.psr_pos,
                Npulsars                 = self.data.npsrs,
                signal_indices           = self.signal_comb_idxs,
                linear_timing_model_size = self.linear_timing_model_size,
                gwb_psd_func             = self.signal_map['cor'].psd_function,
                orf_func                 = self.signal_map['cor'].orf_function,
                gwb_helper_dictionary    = self.signal_map['cor'].psd_reparam_helper,
                crn_bins                 = self.signal_map['cor'].nfreqs,
                f_common                 = self.signal_map['cor'].freqs,
            )
            )
            self.model = partial(parameterized.CorrelatedPulsarRedNoise, **model_kwargs)()

    def update_white_matrix_products_unjitted(self, red_noise_basis, N_list, white_noise_params, reff):
        """Get the helper objects for likelihood evaluation.

        This method computes the helper objects TNT, TNr, rNr, and logdet_N for each pulsar, which are
        needed for the likelihood evaluation and posterior drawing. The TNT, TNr, rNr, and logdet_N
        are computed as:
        - TNT = T^T N^{-1} T
        - TNr = T^T N^{-1} r
        where T is the Fourier design matrix for each pulsar, N is the white noise
        covariance matrix for each pulsar, and r is the effective residuals.

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
        return N_list.get_red_helpers(red_noise_basis = red_noise_basis, 
                                      residuals = reff, 
                                      white_noise_params = white_noise_params) # [FNF, FNr, rNr, logdetN]
                                     
    @jit_method
    def get_sigma(self, TNT, phi_diag):
        """Get the per-pulsar fourier coefficient covariance matrix Sigma. [npsr, nmodes, nmodes]

        This helper method computes the per-pulsar covariance matrix Sigma for the
        fourier coefficients. This is given by Sigma = TNT + phi^{-1}, where
        TNT is the contribution from the white noise and phi^{-1} is the contribution
        from the IRN signal.

        Note that this implicitly assumes that the each pulsar is independent.

        Parameters
        ----------
        TNT : array
            The contribution from the white noise for each pulsar. [npsr, nmodes, nmodes]
        phi_diag : array
            The diagonal of the phi matrix for each pulsar. [npsrs, nmodes]

        Returns
        -------
        array
            The covariance matrix Sigma for each pulsar. [npsr, nmodes, nmodes]
        """
        # Since each pulsar is independent, full Sigma is block diagonal, so we can
        # just make each individual pulsar's Sigma matrix instead
        phiinv = 1/phi_diag # [npsrs, 2*nfreqs]
        Sigma = TNT.at[:, self._diag_idx, self._diag_idx].add(phiinv) # [npsr, nmodes, nmodes]
        return Sigma # List of arrays [npsr, nmodes, nmodes]

    @jit_method
    def get_sigma_from_phiinv(self, TNT, phiinv):
        """Get the per-pulsar fourier coefficient covariance matrix Sigma. [npsr, nmodes, nmodes]

        This helper method computes the per-pulsar covariance matrix Sigma for the
        fourier coefficients. This is given by Sigma = TNT + phi^{-1}, where
        TNT is the contribution from the white noise and phi^{-1} is the contribution
        from the IRN signal. phi^{-1} is supplied by the user in this function.

        Note that this implicitly assumes that the each pulsar is independent.

        Parameters
        ----------
        TNT : array
            The contribution from the white noise for each pulsar. [npsr, nmodes, nmodes]
        phiinv : array
            The inverse of the red noise covaraince matrix. [npsrs, nmodes]

        Returns
        -------
        array
            The covariance matrix Sigma for each pulsar. [npsr, nmodes, nmodes]
        """
        # Since each pulsar is independent, full Sigma is block diagonal, so we can
        # just make each individual pulsar's Sigma matrix instead
        Sigma = TNT.at[:, self._diag_idx, self._diag_idx].add(phiinv) # [npsr, nmodes, nmodes]
        return Sigma # List of arrays [npsr, nmodes, nmodes]

    def ln_likelihood_curn(self, helpers, params):
        """Get the Fourier coefficient marginalized likelihood function for 
        CURN + IRN (i.e., joint common and uncommon) signal.

        This likelihood is marginalized over the fourier coefficients
        and is given by:
        lnlike = (rN^{-1}T)Sigma^{-1}(T^TN^{-1}r) - logdet(Sigma) - logdet(phi))
        
        Parameters
        ----------
        helpers : tuple
            The helper objects (TNT, TNr, rNr, logdet_N) for each pulsar. 
            [npsr, nmode, nmode], [npsr, nmode], [0], [0]
        params : array
            The input parameters, which describe both IRN and GWB

        Returns
        -------
        float
            The log-likelihood contribution from the IRN signal. [1]
        """
        TNT, TNr, rNr, logdet_N = helpers # Unpack helpers [npsr, nmode, nmode], [npsr, nmode], [0], [0]

        phi, psd_common = self.model.get_phi_mat_CURN(params) #[nfreq,npsrs]
        phiinv = jnp.repeat(1/phi.T, 2, axis = 1)
        logdet_phis = 2 * jnp.sum(jnp.log(phi), axis = 0)

        Sigma = self.get_sigma_from_phiinv(TNT, phiinv) # [npsr, nmode, nmode]
        cfs = jsl.cho_factor(Sigma) # Cholesky for each psr [npsr, nmodes, nmodes]

        # Log determinant of phi
        diags = jnp.diagonal(cfs[0], axis1=1, axis2=2) # [npsr, nmodes]
        logdet_Sigma = 2*jnp.sum(jnp.log(diags)) # scalar

        # rNT(Sigma^{-1})TNr
        expvals = jnp.sum(TNr[:,:,None] * jsl.cho_solve(cfs, TNr[:,:,None])) # [npsr, nmodes, 1] -> scalar
        lnlike = 0.5 * (expvals - logdet_Sigma - logdet_phis.sum()) # scalar
        return lnlike - 0.5 * (rNr + logdet_N)

    @cached_property
    def partial_marg_lnposterior_helper(self):
        # 'cor' and 'det' are already single slices, so they stay contiguous
        # by construction — no conversion needed.
        cor_idx = self.signal_comb_idxs['cor']
        # 'P' is every non-GWB stochastic block: the linear timing model, the
        # intrinsic red noise, DM noise and the Adaptus basis, in column order.
        # `dm` was missing here, so with a `dm` block in the model string the
        # reparameterised index set omitted its columns entirely and the phi
        # diagonal no longer matched the sliced TNT -- DM noise could be built
        # but never evaluated.
        # Every key is optional: a model string need not carry an 'ltm' prefix,
        # and need not include every stochastic block. Indexing these
        # unconditionally is what made both reparameterised likelihoods raise
        # KeyError('timing') for any string without 'ltm'.
        P_parts = [self.signal_comb_idxs[k]
                   for k in ('timing', 'unc', 'dm', 'gtm')
                   if k in self.signal_comb_idxs]
        if not P_parts:
            raise ValueError(
                f"model string {self.signal_combination_string!r} has no "
                "non-GWB stochastic block, so there is nothing to marginalise "
                "or reparameterise over"
            )
        P_idx = sutils.merge_slices(*P_parts)
        det_idx = None
        if self.has_det:
            det_idx = self.signal_comb_idxs['det']

        return cor_idx, P_idx, det_idx

    @cached_property
    def _jitted_partial_marg_lnposterior(self):
        """Built once per instance and reused — avoids re-tracing on every call."""
        if self.has_det:
            return jit(self.__partial_marg_lnposterior)
        else:
            return jit(partial(self.__partial_marg_lnposterior, D_params=None))

    @cached_property
    def lnposterior_reparam_helper(self):
        """Index slices for lnposterior_reparam.

        Unlike partial_marg_lnposterior, lnposterior_reparam jointly
        reparameterizes ALL non-det modes (timing [+ gtm] + unc + cor) via z —
        there's no analytic marginalization here, so 'cor' and 'P' don't need
        to stay separate. Merge them into one contiguous 'reparam' block,
        with the merge order matching how partial_marg_lnposterior_helper
        itself is built (P = timing+unc+[gtm] first, cor appended after) so
        that `linear_timing_model_size`-based slicing of the timing-model
        prefix still lines up correctly.

        ASSUMPTION: this assumes the underlying T matrix column order is
        [timing (+unc+gtm) | cor | det] i.e. matches partial_marg's P/cor/det
        layout. If your T matrix actually orders columns differently, adjust
        the merge_slices call below accordingly.
        """
        cor_idx, P_idx, det_idx = self.partial_marg_lnposterior_helper
        reparam_idx = sutils.merge_slices_unique(P_idx, cor_idx)
        return reparam_idx, det_idx

    @cached_property
    def _jitted_lnposterior_reparam(self):
        """Built once per instance and reused — avoids re-tracing on every call."""
        if self.has_det:
            return jit(self.__lnposterior_reparam)
        else:
            return jit(partial(self.__lnposterior_reparam, D_params=None))

    def partial_marg_lnposterior(self, helpers, red_params, z, D_params=None):
        """Public entry point — dispatches to the cached jitted implementation."""
        if self.has_det:
            return self._jitted_partial_marg_lnposterior(helpers, red_params, z, D_params)
        else:
            return self._jitted_partial_marg_lnposterior(helpers, red_params, z)
            
    def lnposterior_reparam(self, helpers, red_params, z, D_params=None):
        """Public entry point — dispatches to the cached jitted implementation.

        Mirrors partial_marg_lnposterior: det signal is only touched when
        self.has_det is True, in which case D_params = (det_params,
        psr_phases, psr_dists) must be supplied.
        """
        if self.has_det:
            return self._jitted_lnposterior_reparam(helpers, red_params, z, D_params)
        else:
            return self._jitted_lnposterior_reparam(helpers, red_params, z)

    def __lnposterior_reparam(self, helpers, red_params, z, D_params=None):
        """
        Evaluates the posterior under a reparameterization of the Fourier
        coefficients (Gaussian processes described by phi_cube), jointly with
        an optional deterministic (e.g. CW) signal. Call within gradient-based
        samplers.

        Parameters
        ----------
        helpers : tuple
            (TNT, TNr, rNr, logdet_N) for each pulsar, over the FULL unified
            design matrix (timing [+ gtm] + unc + cor [+ det], whichever are
            present) — the same TNT/TNr used by partial_marg_lnposterior.
            [npsr, nmode_total, nmode_total], [npsr, nmode_total]
        red_params : array
            Spectral model parameters for the red noise / GWB covariance.
        z : array
            "Whitened coefficients" for the reparameterized (non-det) modes,
            [npsr, n_reparam].
        D_params : tuple or None
            (det_params, psr_phases, psr_dists). Required iff self.has_det.

        Returns
        -------
        tuple
            (log_density, coeff) — coeff are the reparameterized (non-det)
            Fourier coefficients, [npsr, n_reparam].
        """
        TNT, TNr, rNr, logdet_N = helpers
        reparam_idx, det_idx = self.lnposterior_reparam_helper

        # slice out the jointly-reparameterized block (timing [+gtm] + unc + cor)
        RR = TNT[:, *sutils.block_slice(reparam_idx)]        # [npsr, nreparam, nreparam]
        Rr = TNr[:, sutils.vec_slice(reparam_idx), None]      # [npsr, nreparam, 1]

        red_noise_cov = self.model.get_phi_mat_full(red_params)
        if self.npsrs == 1:
            if self.has_gtm:
                phiinvs_diags = 1 / red_noise_cov  # [nmodes, npsrs]
                logdet_phimat = jnp.sum(jnp.log(red_noise_cov))
            else:
                phiinvs_diags = jnp.repeat(1 / red_noise_cov, 2, axis=0)  # [nmodes, npsrs]
                logdet_phimat = 2 * jnp.sum(jnp.log(red_noise_cov))  # 2 accounts for 2*nfreq=nmodes
        else:
            phiinvs, logdet_phimat = self.model.get_phi_mat_inv(red_noise_cov)
            phiinvs_diags = phiinvs.diagonal(axis1=-2, axis2=-1)  # [nmodes, npsrs]

        if self.linear_timing and not self.marg_tm:
            phiinvs_diags_ltm = jnp.full(shape=(self.nmodes, self.npsrs),
                                        fill_value=self.lowest_value_eq_to_zero)
            phiinvs_diags = phiinvs_diags_ltm.at[self.linear_timing_model_size:, :].add(phiinvs_diags)
            # set prior variance of padded parameters to one for stable transformation
            phiinvs_diags = phiinvs_diags.at[:self.linear_timing_model_size, :].add(self._pad_mask.T)

        # deterministic signal, sliced directly out of the same unified TNT/TNr
        # (no separately-built design matrix / helper tensors)
        if self.has_det:
            det_params, psr_phases, psr_dists = D_params
            a_det = self.det_signal.get_coeffs_func(det_params, psr_phases, psr_dists)[..., None]  # [npsr, ndet, 1]

            DD = TNT[:, *sutils.block_slice(det_idx)]                    # [npsr, ndet, ndet]
            RD = TNT[:, *sutils.block_slice(reparam_idx, det_idx)]       # [npsr, nreparam, ndet]
            Dr = TNr[:, sutils.vec_slice(det_idx), None]                 # [npsr, ndet, 1]

            RDas = RD @ a_det  # [npsr, nreparam, 1]
        else:
            RDas = 0.

        # Posterior precision Cholesky (cho_factor equivalent), batched over pulsars
        diag_idx = jnp.arange(RR.shape[-1])
        Sigma_inv = RR.at[:, diag_idx, diag_idx].add(phiinvs_diags.T)  # [npsr, nreparam, nreparam]
        Sigma_inv_L = jsl.cho_factor(Sigma_inv, lower=True)

        # MAP coefficients — shifted by the deterministic signal's contribution when present
        a_hat = jsl.cho_solve(Sigma_inv_L, Rr - RDas)

        # Standardizing transform via back substitution
        Lz = jax.lax.linalg.triangular_solve(
            Sigma_inv_L[0], z[..., None], left_side=True, lower=True, transpose_a=True,
        )  # L^T Lz = z

        coeff = a_hat + Lz  # [npsr, nreparam, 1]

        lndet_Jac = -jnp.sum(jnp.log(Sigma_inv_L[0].diagonal(axis1=-2, axis2=-1)))

        # Log-likelihood (non-det part)
        aFNr = jnp.sum(coeff[..., 0] * Rr[..., 0])
        aFNFa = jnp.sum(coeff.mT @ RR @ coeff)
        lnlike_value = aFNr - 0.5 * aFNFa

        if self.npsrs == 1:
            lnprior_value = -0.5 * ((coeff[:, self.linear_timing_model_size:, 0]**2 * phiinvs_diags[self.linear_timing_model_size:, :].T).sum() + logdet_phimat)
        else:
            aG = coeff[:, self.linear_timing_model_size:]  # [npsr, 2*nfreq, 1]
            lnprior_value = -0.5 * ((aG.transpose(1, 2, 0) @ phiinvs @ aG.transpose(1, 0, 2)).sum() + logdet_phimat)

        if self.linear_timing and not self.marg_tm:
            # add probability density for padded (i.e. zero-ed) timing model parameters for HMC sampler
            # these parameters do not impact the likelihood, prior, and are uncorrelated with all other
            # parameters so this should not affect parameter estimation, but merely provides some
            # curvature for HMC to latch onto when sampling
            padded_logpdf = -0.5 * jnp.sum((self._pad_mask * coeff[:, :self.linear_timing_model_size, 0])**2)
        else:
            padded_logpdf = 0.

        # deterministic (e.g. CW) contribution
        if self.has_det:
            lnlike_det_add = jnp.sum(a_det.mT @ Dr) \
                - jnp.sum(coeff.mT @ RDas) \
                - 0.5 * jnp.sum(a_det.mT @ DD @ a_det)
        else:
            lnlike_det_add = 0.

        log_density = lnlike_value + lnprior_value + lndet_Jac - 0.5 * (rNr + logdet_N) \
            + padded_logpdf + lnlike_det_add

        return log_density, coeff[..., 0]

    def __partial_marg_lnposterior(self, helpers, red_params, z, D_params=None):

        # unpack helper objects
        TNT, TNr, rNr, logdet_N = helpers
        cor_idx, P_idx, det_idx = self.partial_marg_lnposterior_helper
        GG = TNT[:, *sutils.block_slice(cor_idx)]
        PP = TNT[:, *sutils.block_slice(P_idx)]
        GP = TNT[:, *sutils.block_slice(cor_idx, P_idx)]
        Gr = TNr[:, sutils.vec_slice(cor_idx), None]
        Pr = TNr[:, sutils.vec_slice(P_idx), None]

        # only touch deterministic-signal terms if the parent class says to
        if self.has_det:
            DD = TNT[:, *sutils.block_slice(det_idx)]
            GD = TNT[:, *sutils.block_slice(cor_idx, det_idx)]
            PD = TNT[:, *sutils.block_slice(P_idx, det_idx)]
            Dr = TNr[:, sutils.vec_slice(det_idx), None]

            det_params, psr_phases, psr_dists = D_params
            d = self.det_signal.get_coeffs_func(det_params, psr_phases, psr_dists)[..., None]

        # get covariance matrices
        phiinv_G, logdet_phi_G, phiinv_P, logdet_phi_P = self.model.partial_reparm_helper(red_params, self._pad_mask)

        diag_idxs_P = jnp.arange(PP.shape[-1])
        Sigma_inv_P = PP.at[:, diag_idxs_P, diag_idxs_P].add(phiinv_P)

        # natively batched cholesky factor/solve — no vmap needed
        Sigma_inv_P_chol = jsl.cho_factor(Sigma_inv_P, lower=True)
        I_P = jnp.broadcast_to(jnp.identity(Sigma_inv_P.shape[-1]), Sigma_inv_P.shape)
        Sigma_P = jsl.cho_solve(Sigma_inv_P_chol, I_P)
        logdet_Sigma_inv_P = 2 * jnp.sum(jnp.log(jnp.diagonal(Sigma_inv_P_chol[0], axis1=-2, axis2=-1)))

        # Build the quadratic form of the log-posterior.  Every term here stays
        # PER-PULSAR, shape [npsr, 1, 1], and must stay that way until the single
        # `jnp.sum` at the end.  Two traps, both of which were live:
        #   * `rNr` is the ARRAY-WIDE total, accumulated across pulsars inside
        #     `get_red_helpers`.  Folding it into this per-pulsar array subtracts
        #     it once per pulsar.  It is applied exactly once, outside the sum.
        #   * collapsing `U` to a scalar here lets it broadcast back across the
        #     per-pulsar axis when it is added to `g.mT @ V` below, so the final
        #     `jnp.sum` counts it `npsr` times over.
        # Together those scaled the P-block evidence term by `npsr` and `rNr` by
        # `npsr**2` -- and because `Sigma_P` depends on the red-noise parameters,
        # that is a parameter-dependent bias, not a constant offset.
        U = 0.5 * Pr.mT @ Sigma_P @ Pr
        V = -GP @ Sigma_P @ Pr + Gr
        if self.has_det:
            U = U + 0.5 * d.mT @ PD.mT @ Sigma_P @ PD @ d \
                - Pr.mT @ Sigma_P @ PD @ d - 0.5 * d.mT @ DD @ d + d.mT @ Dr
            V = V + GP @ Sigma_P @ PD @ d - GD @ d

        W_inv_without_prior = -GP @ Sigma_P @ GP.mT + GG

        # normalization
        norm = -0.5 * (logdet_N + logdet_phi_G + logdet_phi_P + logdet_Sigma_inv_P)

        # standardizing transformation
        diag_idxs_G = jnp.arange(W_inv_without_prior.shape[-1])
        phiinv_G_diag = jnp.diagonal(phiinv_G, axis1=1, axis2=2).T
        W_inv_curn = W_inv_without_prior.at[:, diag_idxs_G, diag_idxs_G].add(phiinv_G_diag)
        W_inv_curn_L = jsl.cho_factor(W_inv_curn, lower=True)
        g_hat = jsl.cho_solve(W_inv_curn_L, V)
        Lz = jax.lax.linalg.triangular_solve(W_inv_curn_L[0], z[..., None],
                                            left_side=True, lower=True, transpose_a=True)
        g = g_hat + Lz  # [npsr, nmodes, 1]
        lndet_Jac = -jnp.sum(jnp.log(W_inv_curn_L[0].diagonal(axis1=-2, axis2=-1)))

        ln_likelihood = U + g.mT @ V - 0.5 * g.mT @ W_inv_without_prior @ g
        lnprior = -0.5 * (g.transpose(1, 2, 0) @ phiinv_G @ g.transpose(1, 0, 2)).sum()

        # `rNr` enters exactly once, here, outside the per-pulsar sum.
        result = norm + lnprior + lndet_Jac + jnp.sum(ln_likelihood) - 0.5 * rNr
        return result, g