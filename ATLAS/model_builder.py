from ATLAS.signals import signals_utils as sutils
import jax
jax.config.update("jax_enable_x64", True)
from ATLAS.signals.factorized.base import Red, SuperSignal, GaussianTiming
from ATLAS.signals.correlated.base import Correlated
from ATLAS.signals.timing.base import build_multi_psr_timing_model
from ATLAS.nMatrix.base import WhiteCov
from ATLAS.signals.deterministic.base import Deterministic
import jax.numpy as jnp

class ModelBuilder:
    def __init__(self,
                data,
                explicit_timing_model_params_to_sample = None):
    
        self.data = data
        self.SAMPLE = explicit_timing_model_params_to_sample
        
    def make_timing_model(self, enterprise_data = True):
        
        if self.SAMPLE is None:
            if enterprise_data:    
                self.SAMPLE = [self.data.psrs[pidx].fitpars[1:] for pidx in range(self.data.npsrs)]
                for i, sublist in enumerate(self.SAMPLE):
                    for j, s in enumerate(sublist):
                        self.SAMPLE[i][j] = s.replace('A1DOT', 'XDOT')
            else:
                self.SAMPLE = [self.data.psrs[pidx].fit_param_names for pidx in range(self.data.npsrs)]
                
        load_how_many_in_parallel = min(self.data.npsrs, 10)
        return build_multi_psr_timing_model(self.data.parfiles, 
                                                self.data.timfiles, 
                                                self.SAMPLE, 
                                                load_how_many_in_parallel = load_how_many_in_parallel,
                                                data = self.data)

    def make_white_noise(self, stabilize_TNT = True):

        wn_model = WhiteCov(data = self.data, stabilize_TNT = stabilize_TNT)
        return wn_model

    def make_red_noise(self,
                red_noise_combination_string, 
                use_pulsar_tspan = False,
                irn_psd_function = None,
                gwb_psd_function = None,
                det_delay_function = None,
                orf_function = None,
                dm_psd_function = None,
                upper_bound_orf = None,
                lower_bound_orf = None,
                irn_lower_bound_psd = None,
                irn_upper_bound_psd = None,
                dm_lower_bound_psd = None,
                dm_upper_bound_psd = None,
                gwb_lower_bound_psd = None,
                gwb_upper_bound_psd = None,
                det_parameter_bounds = None):
        
        red_signal_names = sutils.parse_basis_string(red_noise_combination_string)['shared_names'] + \
                           sutils.parse_basis_string(red_noise_combination_string)['separate_names']

        signals = []
        if 'unc' in red_signal_names:
            signals.append(Red(name='unc',
                        psd_function = irn_psd_function,
                        nfreqs=self.data.num_irn_bins,
                        lower_bound_psd = irn_lower_bound_psd,
                        upper_bound_psd = irn_upper_bound_psd,
                        data = self.data,
                        use_pulsar_tspan = use_pulsar_tspan))

        if 'gtm' in red_signal_names:
            signals.append(GaussianTiming(name='gtm',
                        nmodes=self.data.adaptus_size,
                        timing_model = None,
                        lower_bound_psd = None, #jnp.ones(int(self.data.adaptus_size/2))*-15,
                        upper_bound_psd = None, #jnp.ones(int(self.data.adaptus_size/2))*2,
                        data = self.data,
                        basis = self.data.adaptus_basis))

        if 'cor' in red_signal_names:
            signals.append(Correlated(name='cor',
                psd_function = gwb_psd_function,
                orf_function = orf_function,
                nfreqs=self.data.num_gwb_bins,
                lower_bound_psd = gwb_lower_bound_psd,
                upper_bound_psd = gwb_upper_bound_psd,
                upper_bound_orf = upper_bound_orf,
                lower_bound_orf = lower_bound_orf,
                data = self.data))

        if 'dm' in red_signal_names:
            signals.append(Red(name='dm',
                        psd_function = dm_psd_function,
                        nfreqs=self.data.num_dm_bins,
                        lower_bound_psd = dm_lower_bound_psd,
                        upper_bound_psd = dm_upper_bound_psd,
                        data = self.data,
                        use_pulsar_tspan = use_pulsar_tspan))

        if 'det' in red_signal_names:
            signals.append(Deterministic(name='det',
                         data=self.data,
                         get_delays_func=det_delay_function,
                         det_parameter_bounds=det_parameter_bounds)
            )

        return SuperSignal(signal_list = signals,
                        signal_combination_string = red_noise_combination_string,
                        data=self.data)
