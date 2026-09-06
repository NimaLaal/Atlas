import numpy as np
from tqdm import tqdm
from functools import cached_property, partial
import os, time, glob, warnings, random
from enterprise_extensions import model_utils, blocks
from enterprise.signals import signal_base, gp_signals
from itertools import combinations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr

class FailException(Exception):
    pass

#####################################Some Custom ORFs###################################
bins = jnp.array([1e-3, 30.0, 50.0, 80.0, 100.0,
                         120.0, 150.0, 180.0]) * jnp.pi/180.0
def HD_ORF(angle):
    return 3/2*( (1/3 + ((1-jnp.cos(angle))/2) * (jnp.log((1-jnp.cos(angle))/2) - 1/6)))

def bin_orf(angle, params):
    '''
    Agnostic binned spatial correlation function. Bin edges are
    placed at edges and across angular separation space. Changing bin
    edges will require manual intervention to create new function.

    :param: params
        inter-pulsar correlation bin amplitudes.

    Author: S. R. Taylor (2020)

    '''
    idx = jnp.digitize(angle, bins)
    return params[idx-1]

def gt_orf(angle, tau):
    """
    General Transverse (GT) Correlations. This ORF is used to detect the relative
    significance of all possible correlation patterns induced by the most general
    family of transverse gravitational waves.

    :param: tau
        tau = 1 results in ST correlations while tau = -1 results in HD correlations.

    Author: N. Laal (2020)

    """
    k = 1/2*(1-jnp.cos(angle))
    return 1/8 * (3+jnp.cos(angle)) + (1-tau)*3/4*k*jnp.log(k)
##################################################################################

 
class BayesPowerMulti(object):
    '''
    A class to perform a multi-pulsar Gibbs sampling

    param: `psrs`: a list of enterprise pulsar objects (Npulsars)
    param: `crn_bins`: the number of frequency bins
    param: `df`: the degrees of freedom of the Inverse-Wishart distribution 
    param: `Tspan`: the baseline of the PTA in secodns
    param: `noise_dict`: noise-dictionary containing the white noise params.
    param: `gamma_spectrum`: the spectral index of the common red noise used for chi-squared fitting
    param: `backend`: the backend to use
    param: `inc_ecorr`: whether to include ecorr
    param: 'int_rn': whether to include non-gwb red noise
    param: `half_logphi_ii_lower`:the lowest value of 0.5log_10 of the diagonals of the phi-matrix in units of seconds
    param: `half_logphi_ii_upper`:the highest value of 0.5log_10 of the diagonals of the phi-matrix in units of seconds
    param: `renorm_const`: the constant to change the units of the matricies from seconds to something else (e.g., nano seconds)
    param: `empirical_nd_orf`: the orf used in the empirical noise distribution run
    param: `fail_trial_count`: the number of times you allow linalg operations to fail without force-kicking the Gibbs sampler

    Author:
    Nima Laal (04/11/2024)
    '''
    def __init__(self, 
                 psrs,
                 run_type_object,
                half_logphi_ii_lower = -9,
                half_logphi_ii_upper = -2,
                df = None, 
                TNr=jnp.array([False]),
                TNT=jnp.array([False]),
                noise_dict = None, 
                backend = 'none', 
                tnequad = False, 
                inc_ecorr = False, 
                int_rn = True,
                gamma_spectrum = 13/3,
                renorm_const = 1,
                fail_trial_count = 1000):
        
        self.psr = psrs
        self.renorm_const = renorm_const
        self.run_type_object = run_type_object
        self.Tspan = Tspan=self.run_type_object.Tspan
        self.crn_bins = self.run_type_object.crn_bins
        self.Npulsars = self.run_type_object.Npulsars
        self.kmax = 2 * self.crn_bins
        self.Npulsars = self.run_type_object.Npulsars
        self.num_a_draws = self.Npulsars * 2

        if not df:
            self.df = self.Npulsars + 1  ##degrees of freedom of the inverse-wishart distribution
        else:
            self.df = df
        self.noise_dict = noise_dict
        self.gamma = gamma_spectrum
        self.fail_trial_count = fail_trial_count
        self.diag_idx = jnp.arange(0, self.run_type_object.Npulsars, 1, int)
        self.k_idx = jnp.arange(0, self.kmax, 1, int)
        self.c_idx = jnp.arange(0, self.crn_bins, 1, int)
        self.diag_offset = jnp.log10(renorm_const)/2
        self.lower_auto = half_logphi_ii_lower + self.diag_offset
        self.upper_auto = half_logphi_ii_upper + self.diag_offset
        self.eye = jnp.eye(self.run_type_object.Npulsars * self.kmax)
        self.eye_batched = jnp.repeat(jnp.eye(self.Npulsars)[None], self.crn_bins, axis=0)
        self.wishart_helper

        if not TNr.any() and not TNT.any():
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            wn = blocks.white_noise_block(
                vary=False,
                inc_ecorr=inc_ecorr,
                gp_ecorr=False,
                select=backend,
                tnequad=tnequad,
            )
            if not self.num_IR_params:
                rn = blocks.red_noise_block(
                    psd="powerlaw",
                    prior="log-uniform",
                    Tspan=self.Tspan,
                    components=self.int_bins,
                    gamma_val=None,
                )
            
            gwb = blocks.common_red_noise_block(
                psd="powerlaw",
                prior="log-uniform",
                Tspan=self.Tspan,
                components=self.crn_bins,
                gamma_val=13 / 3,
                name="gw",
                orf="hd",
            )
            if not self.num_IR_params:
                s = tm + wn + rn + gwb
            else:
                s = tm + wn + gwb

            self.pta = signal_base.PTA(
                [s(p) for p in psrs], signal_base.LogLikelihoodDenseCholesky
            )
            self.pta.set_default_params(self.noise_dict)

            self._TNr = jnp.concatenate(self.pta.get_TNr(params={})) / jnp.sqrt(
                self.renorm_const
            )
            self._TNT = jnp.array(
                sl.block_diag(*self.pta.get_TNT(params={})) / self.renorm_const
            )
            if del_pta_after_init:
                del self.pta
        else:
            self._TNr = TNr / jnp.sqrt(self.renorm_const)
            self._TNT = TNT / self.renorm_const

        ##############Make TNT More Stable:
        print(f'Condition number of the TNT matrix before stabilizing is: {np.format_float_scientific(np.linalg.cond(self._TNT))}')
        D = jnp.outer(jnp.sqrt(self._TNT.diagonal()), jnp.sqrt(self._TNT.diagonal()))
        corr = self._TNT/D
        corr = (corr + 1e-5 * jnp.eye(self._TNT.shape[0]))/(1 + 1e-5)
        self._TNT = D * corr
        print(f'Condition number of the TNT matrix after stabilizing is: {np.format_float_scientific(np.linalg.cond(self._TNT))}')
    ################Utility Functions################

    @partial(jax.jit, static_argnums=(0,))
    def pl_to_rho(self, log10amp, gamma):
        '''
        Function to convert a powerlaw model to a free-spectral model

        param:`log10amp`: the log10 of the amplitude of the red noise
        param: `gamma`: the spectral index of the red noise 
        '''
        return 10**(2*log10amp)/(12 * jnp.pi**2 * self.freqs[:, None]**3 * self.Tspan) * (self.freqs[:, None]/self.fref)**(3-gamma)

    @partial(jax.jit, static_argnums=(0,))
    def get_ln_pos(self, dens_phi, expval, cf):
        '''
        Function to calculate the marginalized posterior distribution
        '''
        LP = jnp.linalg.cholesky(dens_phi)
        logdet_phi = 2 * jnp.sum(jnp.log(LP[:, self.diag_idx, self.diag_idx]))
        logdet_sigma = 2 * jnp.sum(jnp.log(cf.diagonal()))
        loglike = 0.5 * (jnp.dot(self._TNr, expval) - logdet_sigma - 2 * logdet_phi)
        logprior = -self.Npulsars * logdet_phi           
        return loglike + logprior
    
    def kick(self, phi, phi_new, mean, mean_new, chol_var, chol_var_new):
        '''
        Metrapolis Hastings algorithm
        '''
        lnhastings = self.get_ln_pos(phi_new, mean_new, chol_var_new) - self.get_ln_pos(phi, mean, chol_var)
        pred = jnp.log(random.random()) < lnhastings
        decision = jax.lax.cond(
            lnhastings,
            self.spit_True,
            self.spit_False)
        return decision, lnhastings
                       
    def spit_True(self):
        return 1.0

    def spit_False(self):
        return 0.0

    @partial(jax.jit, static_argnums=(0,))
    def chi_sq_fit(self, phi_ij):
        '''
        chi-squared fit to the cross-corrlation values with fixed spectrum
        ''' 
        return 0.5 * jnp.log10(jnp.einsum('qkn,k,n->q',phi_ij, self.spectrum, self.orf_val)/self.bot)
    
    @partial(jax.jit, static_argnums=(0,))
    def to_ent_phiinv(self, phiinv):
        """
        Changes the format of the phiinv matrix from (2*n_freq, n_pulsar, n_pulsar) to (2*n_freq * n_pulsar by 2*n_freq * n_pulsar)
        by adding zeros to the cross-frequency terms.

        :param: `phiinv`: the phiinv matrix with the shape (2*n_freq, n_pulsar, n_pulsar).

        :return: the phiinv matrix with the shape (2*n_freq * n_pulsar by 2*n_freq * n_pulsar).
        """
        phiinv_ent = jnp.zeros((self.Npulsars, self.kmax, self.Npulsars, self.kmax))
        phiinv_ent = phiinv_ent.at[:, self.k_idx, :, self.k_idx].add(phiinv)
        return phiinv_ent.reshape(
            (self.Npulsars * self.kmax, self.Npulsars * self.kmax)
        )
        
    @cached_property
    def wishart_helper(self):
        '''
        This function caches the indicies needed to sample from a standard-wishart distribution
        '''
        self.I, self.J = jnp.tril_indices(self.Npulsars)
        i_cross = []
        j_cross = []
        for i, j in zip(self.I, self.J):
            if not i == j:
                i_cross.append(i)
                j_cross.append(j)
        self.i_cross = jnp.array(i_cross)
        self.j_cross = jnp.array(j_cross)
        self.dfs = jnp.array([self.Npulsars + 1 - i for i in range(self.Npulsars)])

        self.freqs = jnp.arange(1/self.Tspan, (self.crn_bins + .01)/self.Tspan, 1/self.Tspan)
        self.fref = 1/(60 * 60 * 24 * 365.25)
        self.spectrum = 1/(12 * jnp.pi**2 * self.freqs**3 * self.Tspan) * (self.freqs/self.fref)**(3-self.gamma)
        self.xi = self.run_type_object.xi
        self.orf_val = HD_ORF(self.xi)
        self.bot = jnp.einsum('k,n->', self.spectrum**2, self.orf_val**2) * self.renorm_const

    @partial(jax.jit, static_argnums=(0,))
    def standard_wishart(self, rng_key1, rng_key2):
        '''
        Samples from the cholesky decomposition of the standard-wishart distribution.
        `wishart_helper` function needs to have been called once prior to calling this
        function.

        param: `all_freqs_diff = True`: whether to use different set of random numbers
        for each frequency bin
        '''
        A = jnp.zeros((self.crn_bins, self.Npulsars, self.Npulsars))
        A = A.at[:, self.i_cross, self.j_cross].add(jr.random.normal(rng_key1, shape = (self.crn_bins, len(self.I) - self.Npulsars)))
        return A.at[:, self.diag_idx, self.diag_idx].add(jnp.sqrt(jr.random.chisquare(rng_key2, self.dfs, shape = (self.crn_bins, self.Npulsars))))
        
    ################Conditional Distribution Functions################
    @partial(jax.jit, static_argnums=(0,))
    def get_scale_mat(self, a):
        scale = jnp.mean(jnp.transpose(a, (0, 1, 3, 2)) @ a, axis = 0)
        return jnp.linalg.cholesky(scale[0::2] + scale[1::2])  
    
    @partial(jax.jit, static_argnums=(0,))
    def get_mean(self, phiinv):
        """
        Estimates the mean of the Fourier coefficients as well as the log-determinant of the `Sigma` matrix.

        :param: `phiinv`: the phiinv matrix of the shape (2*n_freq, n_pulsar, n_pulsar).
        package does this automatically.

        :return: the mean of the Fourier coefficients as well as the log-determinant of the `Sigma` matrix.
        """
        cf = jsp.linalg.cho_factor(self._TNT + self.to_ent_phiinv(phiinv), lower=False)
        return jsp.linalg.cho_solve(cf, self._TNr), cf[0]

    @partial(jax.jit, static_argnums=(0,))
    def a_given_phiinv(self, mean, chol_var):
        '''
        Performs the `a given phiinv` step of the Gibbs sampling.

        param: `phiinv`: the phiinv matrix. The dimensions are (2*n_freq, n_pulsar, n_pulsar).
        param: `return_mean`: whether to only return the mean of the `a` distribution
        '''
        rand_vec = jr.random.normal(shape=(self.kmax * self.Npulsars, self.num_a_draws))
        a = mean[:, None] + jsp.linalg.solve_triangular(chol_var, rand_vec, trans=0, lower=False)
        a = a.reshape(self.Npulsars, self.kmax, self.num_a_draws).transpose((2, 1, 0))
        return a[:, :, None, :]
    
    @partial(jax.jit, static_argnums=(0,))
    def empiricalphi_to_iwphi(self, xs, chi_fit = True):
        '''
        Converts an empirical estimate of PSD to an IW phi matrix.

        param: `x0`: the parameterized PSD estimates 
        '''
        phi = self.run_type_object.get_phi_mat(xs)
        cp = jsp.linalg.cho_factor(phi, lower=True)
        phiinv_dense = jnp.repeat(jsp.linalg.cho_solve(cp,  self.eye_batched), 2, axis=0)
        mean, chol_var = self.get_mean(self.to_ent_phiinv(phiinv_dense))
        a = self.a_given_phiinv(mean, chol_var)
        return self.phi_given_a(a, chi_sq = chi_fit)

    @partial(jax.jit, static_argnums=(0,))
    def _cond_fun(self, state_count):
        diags, count = state_count
        cond1 = jnp.logical_and(diags > self.lower_auto, diags < self.upper_auto).all()
        cond2 = count < self.fail_trial_count
        return cond1 and cond2

    @partial(jax.jit, static_argnums=(0,))
    def _body_fun(self, state_count):
        diags, count = state_count
        return diags, count + 1

    @partial(jax.jit, static_argnums=(0,))
    def phi_given_a(self, rng_key1, rng_key2, sp,
                    lower_GWB_amp = -16, upper_GWB_amp = -13):
        '''
        Performs the `phiinv given a` step of the Gibbs sampling.
        This function samples from a non-truncated wishart distribution.

        param: `a`: the set of Fourier coefficients. The size must be (n_freq, 1, n_pulsar)
        param: `tol`: the amount added to the diagonals of the scale-matrtix to make the cholesky factorization stable.
        param: `check_diags`: whether to apply rejection sampling to the Inverse-wishart distribution
        param: `lower`: the lower limit of 0.5log10rho used in the rejection sampling
        param: `upper`: the upper limit of 0.5log10rho used in the rejection sampling

        Notes:
        L is an upper-triangular matrix
        SP is a lower-triangular matrix
        L @ L.T = S^-1
        SP @ SP.T = S
        A = Standard-Wishart
        phiinv = (LA) @ (LA).T
        phi = ((LA)^-1).T @ (LA)^-1
        LA = x = solve(SP.T, A)
        ((LA)^-1) = y.T = solve(A, SP.T)
        phiinv = x @ x.T
        phi = y @ y.T
        '''

        A = self.standard_wishart(rng_key1, rng_key2)
        spT = sp.transpose((0, 2, 1))

        yT = jsp.linalg.solve_triangular(A, spT, trans=0, lower=True)
        phi = yT.transpose((0, 2, 1)) @ yT
        diags = 0.5 * jnp.log10(phi.diagonal(axis1 = 1, axis2 = 2))

        ## Check 1: Is phi bounded?
        init_state_count = (diags, 0)
        final_sate = jax.lax.while_loop(self._cond_fun, self._body_fun, init_state_count)    
        # # Check 2: Are correlations ok?
        # amp_fit = self.chi_sq_fit(phi[None, :, self.i_cross, self.j_cross])
        # pred = jnp.any(amp_fit < lower_GWB_amp) or jnp.any(amp_fit > upper_GWB_amp) or jnp.any(~np.isfinite(amp_fit))
        # bounds_flag = jax.lax.cond(
        #     pred,
        #     self.spit_True,
        #     self.spit_False)

        x = jsp.linalg.solve_triangular(spT, A, trans=0, lower=False)
        phiinv = x @ x.transpose((0, 2, 1))
        mean, chol_var = self.get_mean(phiinv)
        
        return phi, diags, mean, chol_var


    def phi_given_a_little(self, sp, lower_GWB_amp = -16, upper_GWB_amp = -13):
        '''
        Performs the `phiinv given a` step of the Gibbs sampling.
        This function samples from a non-truncated wishart distribution.

        param: `a`: the set of Fourier coefficients. The size must be (n_freq, 1, n_pulsar)
        param: `tol`: the amount added to the diagonals of the scale-matrtix to make the cholesky factorization stable.
        param: `check_diags`: whether to apply rejection sampling to the Inverse-wishart distribution
        param: `lower`: the lower limit of 0.5log10rho used in the rejection sampling
        param: `upper`: the upper limit of 0.5log10rho used in the rejection sampling

        Notes:
        L is an upper-triangular matrix
        SP is a lower-triangular matrix
        L @ L.T = S^-1
        SP @ SP.T = S
        A = Standard-Wishart
        phiinv = (LA) @ (LA).T
        phi = ((LA)^-1).T @ (LA)^-1
        LA = x = solve(SP.T, A)
        ((LA)^-1) = y.T = solve(A, SP.T)
        phiinv = x @ x.T
        phi = y @ y.T
        '''

        for _ in range(self.fail_trial_count):

            A = self.standard_wishart()
            if A.ndim == 3:
                y = np.array([st_solve(A[fidx], sp[fidx].T, trans=0, lower=True, unit_diagonal=False, overwrite_b=False, check_finite=True) 
                        for fidx in self.c_idx])
            else:
                y = st_solve(A, sp.transpose((2, 1, 0)), trans=0, lower=True, unit_diagonal=False, overwrite_b=False, check_finite=True).transpose((2, 0, 1))
            yT = np.transpose(y, (0, 2, 1))
            
            phi = yT @ y
            diags = 0.5 * np.log10(phi.diagonal(axis1 = 1, axis2 = 2)) 
            
            ## Check 1: Is phi bounded?
            if np.any(diags < self.lower_auto) or np.any(diags > self.upper_auto):
                continue

            # Check 2: Are correlations ok?
            amp_fit = self.chi_sq_fit(phi[None, :, self.i_cross, self.j_cross])
            if np.any(amp_fit < lower_GWB_amp) or np.any(amp_fit > upper_GWB_amp) or np.any(~np.isfinite(amp_fit)):
                continue

            return diags - self.diag_offset, [amp_fit[0], self.chi_sq_fit_st(phi[None, :, self.i_cross, self.j_cross])[0]], phi[:, self.i_cross, self.j_cross]/self.renorm_const 

    def do_multi_gibbs(self, 
                       savedir,
                       emp_chain_path,
                       niter,
                       gibbs_weight = int(1e4),
                       empir_weight = 10,
                       gibbs_average_weight = 10,
                       resume = False,
                       save_diag_only = False, 
                       pbar_freq = int(5e3),
                       torch_device = 'cuda',
                       num_saved_samples = int(2e5),
                       a_is_scaled = False,
                       save_chol_phi = False, 
                       progress_bar = True):
        '''
        Performs a multi-pulsar gibbs sampling routine

        param: `niter`: The number of iterations
        param: `savedir`: the directory to save the outpur of the sampling
        param: `save_scale`: whether to save the sacale-matrix
        param: `save_phi`: whether to save the phi-matrix
        param: `save_diag_only`: whether to save the diagonals of the phi-matrix instead
        '''
        self.wishart_helper
        # ndraws = int(np.ceil(self.Npulsars/2))
        ndraws = int(1e4)
        pos = list(range(3))
        ew = empir_weight + gibbs_weight
        gw = empir_weight + gibbs_weight + gibbs_average_weight
        no_save_flag_thrshold = 2500

        os.makedirs(savedir, exist_ok=True)

        ###Empirical Noise Distribution
        try:
            phi_saved = np.load(emp_chain_path + '/chain_1.npy', mmap_mode='r')[:, :-4]
        except FileNotFoundError:
            phi_saved = np.load(emp_chain_path + '/chain.npy', mmap_mode='r')[:, :-4]
        print(phi_saved.shape)
        phi_saved_last_idx = phi_saved.shape[0] - 1
        phi, diags, mean, chol_var = self.empiricalphi_to_iwphi(phi_saved[random.randint(0, phi_saved_last_idx)])
        # a_saved = torch.tensor(np.load(emp_chain_path + '/amat.npy', mmap_mode='r')[..., None, :], dtype = torch.float32, device = torch_device)
        # burn_a_saved = int(0.25 * a_saved.shape[0])
        # a_last_idx = a_saved.shape[0]
        # phi, diags, mean, chol_var = self.empiricalcoeff_to_iwphi(a_saved, burn_a_saved, a_last_idx, a_is_scaled = a_is_scaled)
        a = self.a_given_phiinv(mean, chol_var)

        bools = np.ones(niter, dtype = bool)        
        bools[:int(0.25 * niter)] = False
        save_idxs = np.sort(random.sample(np.where(bools)[0].tolist(), k = num_saved_samples))

        if resume and os.path.isfile(savedir + '/phimat_diag.npy'):
            mode = 'r+'
            start_idx = np.nonzero(np.load(savedir + '/phimat_diag.npy', mmap_mode='r')[:, 0, 0])[0].max()
            print(f'Resumed with start index at {start_idx}.')
        else:
            mode = 'w+'
            start_idx = 0

        chain_diag = np.lib.format.open_memmap(savedir + '/phimat_diag.npy', 
                            mode=mode, 
                            dtype='float64', 
                            shape=(num_saved_samples, self.crn_bins, self.Npulsars), 
                            fortran_order=False)
        if not save_diag_only:                
            chain_phi = np.lib.format.open_memmap(savedir + '/phimat.npy', 
                            mode=mode, 
                            dtype='float64', 
                            shape=(num_saved_samples, self.crn_bins, len(self.i_cross)), 
                            fortran_order=False)
        if save_chol_phi:
            chain_chol_phi = np.lib.format.open_memmap(savedir + '/chol_phi.npy', 
                            mode=mode, 
                            dtype='float64', 
                            shape=(num_saved_samples, self.crn_bins, self.Npulsars, self.Npulsars), 
                            fortran_order=False)
            
        chain_a = np.lib.format.open_memmap(savedir + '/amat.npy', 
                    mode=mode, 
                    dtype='float64',
                    shape=(num_saved_samples, self.kmax, 1, self.Npulsars), 
                    fortran_order=False)

        # chain_a[0] = a
        
        ###Step 1 to n:
        if progress_bar:
            pbar = tqdm(range(start_idx, niter), colour="GREEN")
        else:
            pbar = range(start_idx, niter)
            st = time.time()

        no_save_flag = 0
        save_idx = 0
        stamp = 0
        for ii in pbar:
            try:

                if not ii%pbar_freq and ii and not progress_bar:
                    print(f'{round(ii/niter * 100, 2)} Percent Done in {round((time.time() - st)/60, 2)} Minutes.', end='\r')

                phi, diags, mean, chol_var = self.phi_given_a(a, cache_chol_phi = save_chol_phi)

                jump = random.choices(pos, cum_weights=[gibbs_weight, ew, gw])[0]

                if jump == 1:
                    no_save_flag = 0
                    phi_new, diags_new, mean_new, chol_var_new = self.empiricalphi_to_iwphi(phi_saved[random.randint(0, phi_saved_last_idx)])
                    # phi_new, diags_new, mean_new, chol_var_new = self.empiricalcoeff_to_iwphi(a_saved, burn_a_saved, a_last_idx, a_is_scaled = a_is_scaled)
                    if self.kick(phi, phi_new, mean, mean_new, chol_var, chol_var_new):
                        phi = phi_new
                        mean = mean_new
                        diags = diags_new
                        chol_var = chol_var_new  
                    else:
                        print('No empirical jump...')
                elif jump == 2 and save_idx > ndraws:
                    phi_new, diags_new, mean_new, chol_var_new = self.empiricalcoeff_to_iwphi(torch.tensor(chain_a[0:save_idx], dtype = torch.float32, device = 'cuda'), 0, save_idx, a_is_scaled = True)
                    if self.kick(phi, phi_new, mean, mean_new, chol_var, chol_var_new):
                        phi = phi_new
                        mean = mean_new
                        diags = diags_new
                        chol_var = chol_var_new
                        stamp = save_idx    
                    else:
                        print('No Gibbs average jump...')

            except FailException:
                print('Gibbs sampling failed! Force-kicking Gibbs...')
                # phi, diags, mean, chol_var = self.empiricalcoeff_to_iwphi(a_saved, burn_a_saved, a_last_idx, a_is_scaled = a_is_scaled)
                phi, diags, mean, chol_var = self.empiricalphi_to_iwphi(phi_saved[random.randint(0, phi_saved_last_idx)], chi_fit = False)

            a = self.a_given_phiinv(mean, chol_var)
            no_save_flag+=1
            
            ##################Saving###########################
            if save_idx < num_saved_samples:
                if ii == save_idxs[save_idx]:
                    chain_diag[save_idx] = diags - self.diag_offset
                    chain_a[save_idx] = a/np.sqrt(self.renorm_const)
                    if not save_diag_only:
                        chain_phi[save_idx] = phi[:, self.i_cross, self.j_cross]/self.renorm_const
                    if save_chol_phi:
                        chain_chol_phi[save_idx] = self.chol_phi/np.sqrt(self.renorm_const)
                    save_idx+=1
            else:
                break