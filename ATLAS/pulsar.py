
from Atlas.utils import jit, jit_method

import jax.numpy as jnp

from glob import glob
from tqdm import tqdm

from jug.engine.session import TimingSession
from enterprise.pulsar import Pulsar as E_Pulsar
import pint.logging
import logging


LINEAR_PARAMS = ['offset','f','dm','fd','jump'] # Matt also said NE_SW, but I dunno what that is


class Pulsar:
    def __init__(self, par, tim, use_enterprise=True):
        self.par_file = par
        self.tim_file = tim

        # Construct the pulsar using enterprise
        # when we can construct Mmat with JUG, we can use that entirely 
        if use_enterprise:
            pint.logging.setup(level="ERROR")
            logging.getLogger('enterprise').setLevel(logging.ERROR)

            ent_psr = E_Pulsar(par, tim, sort=False)

            self.name = ent_psr.name

            self.toas = jnp.array(ent_psr.toas)
            self.residuals = jnp.array(ent_psr.residuals)
            self.toaerrs = jnp.array(ent_psr.toaerrs)
            self.freqs = jnp.array(ent_psr.freqs)
            self.backend_flags = list(ent_psr.backend_flags) # Can't store as jnp array

            self._raj = jnp.double(ent_psr._raj)
            self._decj = jnp.double(ent_psr._decj)

            self.pos = jnp.array(ent_psr.pos)

            self.Mmat = jnp.array(ent_psr.Mmat)
            self.Mmat_labels = list(ent_psr.fitpars)

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

            self.toas = jnp.array(result['bat_sec'])
            self.residuals = jnp.array(result['residuals_us']) * 1e-6
            self.toaerrs = jnp.array([t.error_us for t in s.toas_data]) * 1e-6
            self.freqs = jnp.array(result['freq_bary_mhz'])
            self.backend_flags = list([t.flags['f'] for t in s.toas_data]) # Can't store as jnp array

            self._raj = float(s.params['_raj_rad'])
            self._decj = float(s.params['_decj_rad'])
            
            # Convert RAJ and DECJ to Cartesian coordinates
            pos = [jnp.cos(self._raj) * jnp.cos(self._decj),
                   jnp.sin(self._raj) * jnp.cos(self._decj),
                   jnp.sin(self._decj)]
            self.pos = jnp.array(pos)

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

    psrs = []
    for p,t in tqdm(zip(par_files, tim_files), total=len(par_files), desc='Loading pulsars'):
        pname = p.split('/')[-1].split('_')[0]
        tname = t.split('/')[-1].split('_')[0]
        assert pname == tname, f'Par file {p} and tim file {t} do not match'

        psr = Pulsar(p, t, use_enterprise=use_enterprise)
        psrs.append(psr)
    
    return psrs





