
from ATLAS.utils import jit, jit_method

import jax.numpy as jnp
import numpy as np

from glob import glob

from tqdm_joblib import ParallelPbar
from joblib import delayed

from jug.engine.session import TimingSession
from enterprise.pulsar import Pulsar as E_Pulsar
import pint.logging
import logging
from dataclasses import dataclass

MAX_JOBS = 8
LINEAR_PARAMS = ['offset','f','dm','fd','jump'] # Matt also said NE_SW, but I dunno what that is
TEMPO2_ALIASES = ('tempo2', 't2', 'libstempo')

def _backend_flags(psr):
    """Backend flags for an enterprise pulsar, tolerant of flagless TOAs.

    enterprise's ``backend_flags`` raises when the TOAs carry no flags at all,
    which is the case for the IPTA mock data challenge sets read through
    tempo2. Fall back to a single unnamed backend -- what the PINT reader
    yields for the very same files.
    """
    try:
        return list(psr.backend_flags)
    except (ValueError, AttributeError):
        return [''] * len(psr.toas)

def load_pulsars(par, tim, use_enterprise=True, timing_package='pint'):
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
        # strip the extension before splitting, so that datasets named plainly
        # ("J0030+0451.par") pair up as well as NANOGrav's "{PSR}_PINT_DMX.par"
        pname = par_files[i].split('/')[-1].rsplit('.', 1)[0].split('_')[0]
        tname = tim_files[i].split('/')[-1].rsplit('.', 1)[0].split('_')[0]
        assert pname == tname, f'Par file {par_files[i]} and tim file {tim_files[i]} do not match'

        psr = Pulsar(par_files[i], tim_files[i], use_enterprise=use_enterprise,
                     timing_package=timing_package)
        return psr
    
    psrs = ParallelPbar("Loading pulsars...")(n_jobs=MAX_JOBS)(
        delayed(foo)(i) for i in range(len(par_files))
    )

    return psrs

class Pulsar:
    def __init__(self, par, tim, use_enterprise=True, timing_package='pint'):
        self.par_file = par
        self.tim_file = tim
        self.timing_package = timing_package

        # Construct the pulsar using enterprise
        # when we can construct Mmat with JUG, we can use that entirely
        if use_enterprise:
            pint.logging.setup(level="ERROR")
            logging.getLogger('enterprise').setLevel(logging.ERROR)

            use_tempo2 = str(timing_package).lower() in TEMPO2_ALIASES
            if use_tempo2:
                # tempo2 reads TCB par files and every BINARY model (including
                # tempo2's own auto-dispatching "BINARY T2") natively, so it
                # loads datasets PINT refuses or silently mis-scales.
                psr = E_Pulsar(par, tim, timing_package='tempo2',
                               sort=False, drop_t2pulsar=False) # keeps psr.t2pulsar
            else:
                # Be explicit: enterprise picks tempo2 when libstempo is
                # importable, so omitting this silently ignores
                # timing_package='pint' and then fails on psr.model.
                psr = E_Pulsar(par, tim, timing_package='pint',
                               sort=False, drop_pintpsr=False) # keeps psr.model

            self.name = psr.name

            # Actual TOA stuffs
            self.toas = jnp.array(psr.toas)
            self.residuals = jnp.array(psr.residuals)
            self.toaerrs = jnp.array(psr.toaerrs)
            self.freqs = jnp.array(psr.freqs)
            self.backend_flags = _backend_flags(psr) # Can't store as jnp array

            # Position stuffs
            self.raj = jnp.double(psr._raj)
            self.decj = jnp.double(psr._decj)
            self.pos = jnp.array(psr.pos)

            # Timing model parameters
            self.fit_param_names = list(psr.fitpars)[1:] # Ignore the offset
            if use_tempo2:
                self.fit_param_values = jnp.array([psr.t2pulsar[k].val for k in self.fit_param_names], dtype=jnp.float64)
                self.fit_param_uncertainties = jnp.array([psr.t2pulsar[k].err for k in self.fit_param_names], dtype=jnp.float64)
            else:
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
            result = s.fit_parameters(verbose=False)

            self.name = s.params['PSR']

            # Actual TOA stuffs
            self.toas = jnp.array(result['bat_sec'])
            self.residuals = jnp.array(result['residuals_us']) * 1e-6
            self.toaerrs = jnp.array([t.error_us for t in s.toas_data]) * 1e-6
            self.freqs = jnp.array(result['freq_bary_mhz'])
            self.backend_flags = list([t.flags['f'] for t in s.toas_data]) # Can't store as jnp array

            # Pulsar Distances
            pdist = np.load('../datasets/NG15/pulsar_distances_15yr.npz')[self.name][:2]
            self.pdist = jnp.array([(float(x), float(y)) for x, y in [pdist]])[0]

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
            self.Mmat_linear = self.Mmat[:, self.Mmat_is_linear]
            self.Mmat_linear_labels = [l for l, is_lin in zip(self.Mmat_labels, self.Mmat_is_linear) if is_lin]

    # @property
    # def Mmat_linear(self):
    #     return self.Mmat[:, self.Mmat_is_linear]
    
    # @property
    # def Mmat_linear_labels(self):
    #     return [l for l, is_lin in zip(self.Mmat_labels, self.Mmat_is_linear) if is_lin]



@dataclass
class PulsarDataLoader:
    par_file: str
    tim_file: str
    timing_package: str

    # Basic info
    name: str

    # TOA data
    toas: jnp.ndarray
    residuals: jnp.ndarray
    toaerrs: jnp.ndarray
    freqs: jnp.ndarray
    backend_flags: list

    # Pulsar distance
    pdist: jnp.ndarray

    # Position
    raj: float
    decj: float
    pos: jnp.ndarray

    # Timing-model parameters
    fit_param_names: list
    fit_param_values: jnp.ndarray
    fit_param_uncertainties: jnp.ndarray

    # Linearized design matrix
    Mmat: jnp.ndarray
    Mmat_labels: list
    Mmat_is_linear: jnp.ndarray
    Mmat_linear: jnp.ndarray
    Mmat_linear_labels: list



