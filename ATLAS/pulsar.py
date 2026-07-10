
from ATLAS.utils import jit, jit_method

import jax.numpy as jnp

from glob import glob

from tqdm_joblib import ParallelPbar
from joblib import delayed

from jug.engine.session import TimingSession
from enterprise.pulsar import Pulsar as E_Pulsar
import pint.logging
import logging

MAX_JOBS = 8
LINEAR_PARAMS = ['offset','f','dm','fd','jump'] # Matt also said NE_SW, but I dunno what that is

def load_pulsars(par, tim, use_enterprise=True):
    # par could be a list of par files, a directory string, or a single par file string
    if isinstance(par, list): # list of par file strings
        par_files = par
    elif isinstance(par, str): # Could be a directory string or a single par file string
        if par.endswith('.par'):
            par_files = [par] # single par file string
        else: # directory string
            par_files = sorted(glob(f'{par}/*.par'))
    else:
        raise ValueError('Unrecognized par file input')
    
    # tim could be a list of tim files, a directory string, or a single tim file string
    if isinstance(tim, list): # list of tim file strings
        tim_files = tim
    elif isinstance(tim, str): # Could be a directory string or a single tim file string
        if tim.endswith('.tim'):
            tim_files = [tim] # single tim file string
        else: # directory string
            tim_files = sorted(glob(f'{tim}/*.tim'))
    else:
        raise ValueError('Unrecognized tim file input')
    
    # Ensure that the pars have corresponding tim files
    assert len(par_files) == len(tim_files), 'Number of par files must match number of tim files'

    def foo(i):
        pname = par_files[i].split('/')[-1].split('_')[0]
        tname = tim_files[i].split('/')[-1].split('_')[0]
        assert pname == tname, f'Par file {par_files[i]} and tim file {tim_files[i]} do not match'

        psr = Pulsar(par_files[i], tim_files[i], use_enterprise=use_enterprise)
        return psr
    
    psrs = ParallelPbar("Loading pulsars...")(n_jobs=MAX_JOBS)(
        delayed(foo)(i) for i in range(len(par_files))
    )

    return psrs

class Pulsar:
    def __init__(self, par, tim, use_enterprise=True):
        self.par_file = par
        self.tim_file = tim

        # Construct the pulsar using enterprise
        # when we can construct Mmat with JUG, we can use that entirely 
        if use_enterprise:
            pint.logging.setup(level="ERROR")
            logging.getLogger('enterprise').setLevel(logging.ERROR)

            psr = E_Pulsar(par, tim, sort=False, drop_pintpsr=False) # keeps psr.model

            self.name = psr.name

            # Actual TOA stuffs
            self.toas = jnp.array(psr.toas)
            self.residuals = jnp.array(psr.residuals)
            self.toaerrs = jnp.array(psr.toaerrs)
            self.freqs = jnp.array(psr.freqs)
            self.backend_flags = list(psr.backend_flags) # Can't store as jnp array

            # Position stuffs
            self.raj = jnp.double(psr._raj)
            self.decj = jnp.double(psr._decj)
            self.pos = jnp.array(psr.pos)

            # Timing model parameters
            self.fit_param_names = list(psr.fitpars)[1:] # Ignore the offset
            self.fit_param_values = jnp.array([psr.model[k].value for k in self.fit_param_names], dtype=jnp.float64)
            self.fit_param_uncertainties = jnp.array([psr.model[k].uncertainty_value for k in self.fit_param_names], dtype=jnp.float64)

            # Linearized design matrix
            self.Mmat = jnp.array(psr.Mmat)
            self.Mmat_labels = list(psr.fitpars)

            # Label which columns are exactly linear
            linear = []
            for l in self.Mmat_labels:
                state = any(l.lower().startswith(lp) for lp in LINEAR_PARAMS)
                linear.append(state)
            self.Mmat_is_linear = jnp.array(linear)
            
        else:
            s = TimingSession(par, tim)
            result = s.fit_parameters()

            self.name = s.params['PSR']

            # Actual TOA stuffs
            self.toas = jnp.array(result['bat_sec'])
            self.residuals = jnp.array(result['residuals_us']) * 1e-6
            self.toaerrs = jnp.array([t.error_us for t in s.toas_data]) * 1e-6
            self.freqs = jnp.array(result['freq_bary_mhz'])
            self.backend_flags = list([t.flags['f'] for t in s.toas_data]) # Can't store as jnp array

            # Position stuffs
            self.raj = float(s.params['_raj_rad'])
            self.decj = float(s.params['_decj_rad'])
            # Convert RAJ and DECJ to Cartesian coordinates
            pos = [jnp.cos(self.raj) * jnp.cos(self.decj),
                   jnp.sin(self.raj) * jnp.cos(self.decj),
                   jnp.sin(self.decj)]
            self.pos = jnp.array(pos)

            # Timing model parameters
            self.fit_param_names = list(result['final_params'].keys())
            self.fit_param_values = jnp.array(list(result['final_params'].values()), dtype=jnp.float64)
            self.fit_param_uncertainties = jnp.array(list(result['uncertainties'].values()), dtype=jnp.float64)
                                                                 
            # Linearized design matrix
            self.Mmat = jnp.array(result['design_matrix'])
            self.Mmat_labels = result['design_matrix_labels']

            # Label which columns are exactly linear
            linear = []
            for l in self.Mmat_labels:
                state = any(l.startswith(lp) for lp in LINEAR_PARAMS)
                linear.append(state)

            self.Mmat_is_linear = jnp.array(linear)

    @property
    def Mmat_linear(self):
        return self.Mmat[:, self.Mmat_is_linear]
    
    @property
    def Mmat_linear_labels(self):
        return [l for l, is_lin in zip(self.Mmat_labels, self.Mmat_is_linear) if is_lin]







