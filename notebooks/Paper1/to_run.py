
import sys
sys.path.append('../')
sys.path.append('../../')

import ATLAS

import json, pickle
from glob import glob
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt

from functools import partial
import jax.numpy as jnp
import jax.random as jrandom

from ATLAS.utils import jit, jit_method, get_pulsar_timespan
from ATLAS.utils import jagged2padded
from ATLAS.signals.signals_utils import _timing_model_svd, get_fourier_design_matrix

import numpyro
import numpyro.distributions as dist

sim_path = '../../datasets/sim_NG15_01'
with open(sim_path+'/injected_params.json', 'r') as f:
    injected_params = json.load(f)


from ATLAS.pulsar import load_pulsars
from ATLAS.data import PTA_Data



par = sorted(glob(sim_path+'/par/*.par'))
tim = sorted(glob(sim_path+'/tim/*.tim'))
psrs = load_pulsars(par, tim, use_enterprise=True)

data = PTA_Data(psrs, fixed_res = False, marg_timing = False, 
                diag_white_cov = False, linear_timing = True)


binary_params = ['PB', 'PBDOT', 'A1', 'A1DOT', 'T0', 'TASC', 'ECC', 'EDOT', 'OM', 'OMDOT', 
                 'EPS1', 'EPS2', 'SINI', 'M2', 'KIN', 'KOM', 'H3', 
                 'FB0', 'FB1', 'FB2', 'FB3', 'FB4', 'FB5']


M_full = [p.Mmat for p in psrs]
M_full_labels = [np.array(p.Mmat_labels) for p in psrs]

is_binary = []
for M in M_full_labels:
    b = jnp.array([(l in binary_params) for l in M])
    is_binary.append(b)

M_other = [M[:,~b] for M, b in zip(M_full, is_binary)]
U_other = [_timing_model_svd(M) for M in M_other]
M_other_labels = [M[~b] for M, b in zip(M_full_labels, is_binary)]

M_binary = [M[:,b] for M, b in zip(M_full, is_binary)]
M_binary_labels = [M[b] for M, b in zip(M_full_labels, is_binary)]

# Concatenate the un-normed binary timing model with the normed other timing model
M_comb = [jnp.concatenate([b, o], axis=1) for b, o in zip(M_binary, U_other)]
M_comb_labels = [np.concatenate([b, o]) for b, o in zip(M_binary_labels, M_other_labels)]

# Add the concatenated timing model to data
data.add_timing_design_matrix(M_comb)



from ATLAS.psd_functions import powerlaw, free_spectrum, hd_orf
from ATLAS.signals.factorized.base import Red, SuperSignal
from ATLAS.signals.correlated.base import Correlated
from ATLAS.nMatrix.base import WhiteCov
from ATLAS.signals import signals_utils as sutils
import ATLAS.utils as atlas_utils
import ATLAS.signals.factorized.utils as unc_utils
import ATLAS.signals.correlated.utils as cor_utils

## White Noise Model
wn_model = WhiteCov(data = data, stabilize_TNT = True)
wn_lower_bound, wn_upper_bound = wn_model.get_prior_bounds() #white noise prior bounds

## Red Noise Model
non_gwb_nfreqs = 30
gwb_nfreqs = 14
sig_unc = Red(name='unc',
            psd_function = partial(powerlaw),
            nfreqs=non_gwb_nfreqs,
            lower_bound_psd = jnp.array([-18, 0.]),
            upper_bound_psd = jnp.array([-11, 7.0]),
            data = data,
            use_pulsar_tspan = False)

# GWB model
sig_cor = Correlated(name='cor',
            psd_function = partial(powerlaw),
            orf_function = hd_orf,
            nfreqs=gwb_nfreqs,
            lower_bound_psd = jnp.array([-18., 0.]),
            upper_bound_psd = jnp.array([-11., 1.]),
            data = data)

# Combined model
sig = SuperSignal(signal_list = [sig_unc],#, sig_cor],
                signal_combination_string = "ltm|unc->unc",
                data=data)


import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value, init_to_uniform

from ATLAS.samplers.canetoadracing import MultiHMCGibbs

theta_wn_dist = dist.Uniform(wn_lower_bound, wn_upper_bound)
xs_dist = dist.Uniform(sig.model.lower_prior_lim_all, sig.model.upper_prior_lim_all)
z_a_dist = dist.Normal(0, 1).expand((sig.npsrs, sig.nmodes))
r_all = jnp.concatenate(data.raw_residuals)

def model():

    ######################################## White Noise ########################################
    # Declare the variables in the model
    theta_wn = numpyro.sample('theta_wn', theta_wn_dist)
    helpers_now = sig.get_helpers(reff = r_all, white_noise_params=theta_wn)
    
    ######################################## Red Noise ########################################
    # Declare the variables in the model
    xs = numpyro.sample('xs', xs_dist) #PSD params
    z_a = numpyro.sample('z_a', z_a_dist) #the reparam coefficients
    # evaluate the posterior
    lprob, coeff = sig.lnposterior_reparam(helpers = helpers_now, red_params = xs, z = z_a)
    numpyro.factor('lnpost', lprob + 0.5 * jnp.sum(z_a**2))
    
    numpyro.deterministic('tm_z', z_a[:,:sig.linear_timing_model_size])
    numpyro.deterministic('tm_coeff', coeff[:,:sig.linear_timing_model_size])


kernel = NUTS(model=model, max_tree_depth=12)

mcmc = MCMC(sampler=kernel, num_warmup=1000, num_samples=1000)

mcmc.run(jrandom.PRNGKey(170817))
samples = mcmc.get_samples()

with open('output_samples.pkl', 'wb') as f:
    import pickle
    pickle.dump(samples, f)
    

