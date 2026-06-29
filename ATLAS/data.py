
from Atlas.utils import get_pulsar_timespan
from Atlas.utils import jit, jit_method

import jax.numpy as jnp


class PTA_Data:
    """A class to hold static data attributes of the PTA dataset.

    This class is intended to hold all the static data attributes of the PTA
    dataset, such as the pulsar objects, their TOAs, positions, and timespans.
    See the following attributes for details.

    Attributes
    ----------
    psrs : list
        A list of enterprise-like pulsar objects.
    npsrs : int
        The number of pulsars in the dataset.
    psr_names : list
        A list of pulsar names corresponding to the pulsar objects.
    fixed_wn : bool
        A flag indicating whether the white noise matrices are fixed.
    toas : list of arrays
        A list where each element is an array of TOAs for a pulsar. [npsrs, ntoas]
    psr_pos : array
        An array containing the positions of the pulsars in unit-Cartesian 
        coordinates. [npsrs, 3]
    pta_tspan : float
        The total timespan covered by the PTA dataset, calculated as the
        difference between the maximum and minimum TOAs across all pulsars.
    psr_tspans : array
        An array containing the individual timespans for each pulsar, calculated
        as the difference between the maximum and minimum TOAs for each pulsar. [npsrs]
    """

    def __init__(self, psrs, fixed_wn=False, 
                fixed_white_noise_params = None,
                linear_timing = False, # Linearized M matrix
                fixed_res = False): 
        """The constructor for the PTA_Data class

        This class is intended to hold all the static data attributes of the PTA 
        dataset, such as the pulsar objects, their TOAs, positions, and timespans.

        Parameters
        ----------
        psrs : list
            A list of enterprise-like pulsar objects.
        fixed_wn : bool
            Whether the white noise matrices are fixed, by default False.
        """
        self.psrs = psrs # List of pulsar objects
        self.npsrs = len(psrs) # Number of pulsars
        self.npairs = self.npsrs * (self.npsrs - 1) // 2 # Number of unique pulsar pairs
        self.psr_names = [p.name for p in psrs] # List of pulsar names
        self.fixed_wn = fixed_wn # Whether the white noise matrices are fixed (bool)
        self.fixed_white_noise_params = fixed_white_noise_params
        # (jagged) List of each pulsar's TOAs (npsrs, ntoas)
        self.toas = [jnp.array(p.toas) for p in psrs]
        # Array of pulsar positions in unit-Cartesian coordinates (npsrs, 3)
        self.psr_pos = jnp.array([p.pos for p in psrs])

        # Total timespan for the whole PTA (float) (maxtoa - mintoa)
        self.pta_tspan = get_pulsar_timespan(psrs)
        # Array of each pulsar's individual timespans (npsrs) 
        self.psr_tspans = jnp.array([get_pulsar_timespan(p) for p in psrs])

        # Raw residuals - List of each pulsar's residuals (npsrs, npsr_toas)
        self.raw_residuals = [jnp.array(p.residuals) for p in psrs]
        self.fixed_res = fixed_res
        
        self.linear_timing = linear_timing

    def add_white_noise_cov(self, white_noise_cov):
        self.Nmat = white_noise_cov

    def add_timing_design_matrix(self, Mmat):
        self.Mmat = Mmat
        