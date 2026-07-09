
from ATLAS.utils import get_pulsar_timespan
from ATLAS.utils import jit, jit_method
from ATLAS.signals.signals_utils import _timing_model_svd

import jax.numpy as jnp
from tqdm import tqdm


class PTA_Data:
    """A class to hold static data attributes of the PTA dataset.

    This class is intended to hold all the static data attributes of the PTA
    dataset, such as the pulsar objects, their TOAs, positions, and timespans.
    """

    def __init__(self, 
                psrs, 
                fixed_white_noise_params = jnp.array([False]),
                linear_timing = False,
                marg_timing = False,
                diag_white_cov = False,
                fixed_res = False,
                dm_ref_freq = 1400): 
        """The constructor for the PTA_Data class

        This class is intended to hold all the static data attributes of the PTA 
        dataset, such as the pulsar objects, their TOAs, positions, and timespans.

        Parameters
        ----------
        psrs : list
            A list of pulsar objects.
        fixed_white_noise_params : jnp.ndarray, optional
            An array indicating which white noise parameters are fixed, by default jnp.array([False])
        linear_timing : bool, optional
            A flag indicating whether to use linear timing, by default False
        marg_timing : bool, optional
            A flag indicating whether to use marginalized timing, by default False
        diag_white_cov : bool, optional
            A flag indicating whether to use diagonal white noise covariance, by default False
        fixed_res : bool, optional
            A flag indicating whether to fix the residuals, by default False
        dm_ref_freq : int, optional
            The reference frequency for dispersion measure calculations, by default 1400
        """
        self.psrs = psrs # List of pulsar objects
        self.npsrs = len(psrs) # Number of pulsars
        self.npairs = self.npsrs * (self.npsrs - 1) // 2 # Number of unique pulsar pairs
        self.psr_names = [p.name for p in psrs] # List of pulsar names

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

        # Linear Timing Design Matrix
        self.Mmat = [_timing_model_svd(psr.Mmat) for psr in psrs]

        ######################PTA Data Analysis General Settings######################
        # Whether the white noise matrices are fixed (bool)
        self.fixed_wn = True if fixed_white_noise_params.any() else False 
        self.fixed_white_noise_params = fixed_white_noise_params
        # No residual subtraction?
        self.fixed_res = fixed_res
        # Linear (M \epsilon) appraoch to modeling the timing model errors
        self.linear_timing = linear_timing
        self.diag_white_cov = diag_white_cov
        self.marg = marg_timing

        # Radio frequencies
        radio_freqs = jnp.concat([psr.freqs for psr in psrs])
        self.dm_ref_freq = dm_ref_freq
        self.ref_over_radio_freqs = self.dm_ref_freq / radio_freqs

        # A celever way to broadcast DM index over all concatenated toas
        ct = 0
        self.dm_exploder_idxs = []
        for pidx in range(self.npsrs):
            self.dm_exploder_idxs.append(ct * jnp.ones(len(self.toas[pidx])))
            ct+=1
        self.dm_exploder_idxs = jnp.concat(self.dm_exploder_idxs).astype(int)

    def add_white_noise_cov(self, white_noise_cov):
        self.Nmat = white_noise_cov

    def add_timing_design_matrix(self, Mmat):
        self.Mmat = Mmat
        