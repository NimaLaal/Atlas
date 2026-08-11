import sys
sys.path.append('/data/taylor_group/Nima/ATLAS')
sys.path.append('../')
import glob
import numpy as np
import pickle
import json
import os
from ATLAS.data import PTA_Data
from ATLAS.model_builder import ModelBuilder
from ATLAS.psd_functions import powerlaw, free_spectrum, hd_orf, gt_orf, bin_orf
from functools import partial
import jax.numpy as jnp
import jax.random as jrandom
from ATLAS.samplers.canetoadracing import model_maker
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

pnames = ['B1855+09', 'B1937+21', 'B1953+29', 'J0023+0923', 'J0030+0451', 'J0340+4130', 'J0406+3039', 
    'J0437-4715', 'J0509+0856', 'J0557+1551', 'J0605+3757', 'J0610-2100', 'J0613-0200', 'J0636+5128', 
    'J0645+5158', 'J0709+0458', 'J0740+6620', 'J0931-1902', 'J1012+5307', 'J1012-4235', 'J1022+1001', 
    'J1024-0719', 'J1125+7819', 'J1312+0051', 'J1453+1902', 'J1455-3330', 'J1600-3053', 'J1614-2230', 
    'J1630+3734', 'J1640+2224', 'J1643-1224', 'J1705-1903', 
    'J1713+0747', 
    'J1719-1438', 'J1730-2304', 'J1738+0333', 'J1741+1351', 'J1744-1134', 'J1745+1017', 'J1747-4036', 
    'J1751-2857', 'J1802-2124', 'J1811-2405', 'J1832-0836', 'J1843-1113', 'J1853+1303', 'J1903+0327', 
    'J1909-3744', 'J1910+1256', 'J1911+1347', 'J1918-0642', 'J1923+2515', 'J1944+0907', 'J1946+3417', 
    'J2010-1323', 'J2017+0603', 'J2033+1734', 'J2043+1711', 'J2124-3358', 'J2145-0750', 'J2214+3000', 
    'J2229+2643', 'J2234+0611', 'J2234+0944', 'J2302+4442', 'J2317+1439', 'J2322+2057']

parfiles_ref = sorted(glob.glob('/*.par'))
timfiles_ref = sorted(glob.glob('/*.tim'))
parfiles = []
for pname in pnames:
    for p in parfiles_ref:
        if pname in p and not 'ao' in p and not 'gbt' in p:
            parfiles.append(p)
timfiles = []
for pname in pnames:
    for p in timfiles_ref:
        if pname in p and not 'ao' in p and not 'gbt' in p:
            timfiles.append(p)


sdir = ...
os.makedirs(sdir, exist_ok = True)

with open(path_to_noise_psrs_object, 'rb') as fin:
    psrs = pickle.load(fin)
psrs = [psr for psr in psrs if psr.name != 'J1713+0747']

with open(path_to_noise_dictionary, 'r') as fin:
    noise_dict = json.load(fin)


data = PTA_Data(psrs, 
                adaptus_basis = [np.load(path_to_Adaptus_bases)[0]
                                  for psr in psrs],
                num_gwb_bins = 14,
                num_irn_bins = 30,
                num_dm_bins = None,
                adaptus_size = 500, 
                fixed_white_noise_params = None,
                linear_timing = True,
                marg_timing = False,
                diag_white_cov = False,
                fixed_res = False,
                timfiles = timfiles,
                parfiles = parfiles,
                noise_dict = noise_dict,
                dm_ref_freq = 1400
                )
Mmats = [np.load(path_to_Mmat)[0] for psr in psrs] #1-column Mmat (phase offset)
data.add_timing_design_matrix(Mmats)

m = ModelBuilder(data = data)
# tm = m.make_timing_model(enterprise_data = True); data.add_timing_design_matrix(tm.Mmats)
wn = m.make_white_noise(stabilize_TNT = True)
rn = m.make_red_noise("ltm|unc+cor->unc;gtm",
                use_pulsar_tspan = False,
                irn_psd_function = partial(powerlaw),
                gwb_psd_function = partial(powerlaw, gamma = 13/3),
                orf_function = bin_orf, #hd_orf,
                dm_psd_function = None,
                irn_lower_bound_psd = jnp.array([-18, 0.]),
                irn_upper_bound_psd = jnp.array([-11, 7.]),
                dm_lower_bound_psd = None,
                dm_upper_bound_psd = None,
                gwb_lower_bound_psd = jnp.array([-18]),
                gwb_upper_bound_psd = jnp.array([-11]),
                upper_bound_orf = jnp.ones(7) * -.95,
                lower_bound_orf = jnp.ones(7) * .95,
                 )

raw_res = jnp.concat(data.raw_residuals)
helpers = rn.get_helpers(reff = raw_res, 
                white_noise_params = wn.params_dict_to_vector(data.noise_dict))
wn_lower_bound, wn_upper_bound = wn.get_prior_bounds()

nuts_kernel = NUTS(
            model = model_maker, #############NOTE: check the model_maker to see what it does.
            target_accept_prob = 0.8,
            # init_strategy = init_to_value(values={"red_noise": x0, 
            #                                      "z_a": jnp.zeros((data.npsrs, rn.nmodes))}),
            max_tree_depth = 10)
mcmc = MCMC(
    sampler=nuts_kernel,
    num_warmup=500,
    num_samples=3000,
    num_chains=1,
)
mcmc.run(jrandom.key(170817), 
                extra_fields=("~z.z_a",),
                raw_residuals = raw_res, 
                super_sig = rn,
                vary_white = False,
                wn_lower_bound = wn_lower_bound,
                wn_upper_bound = wn_upper_bound,
                tm_model = None,
                helpers = helpers,
                save_red_coeff = False,
                marg_over_non_gwb = False)

os.makedirs(sdir, exist_ok=True)
np.savez_compressed(sdir + '/TARGET.npz', **mcmc.get_samples())