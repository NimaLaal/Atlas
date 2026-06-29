import numpy as np
import jax.numpy as jnp
import inspect
import Atlas.psd_functions as psd_functions

# non-GWB Model Definition Utilities--------------------------------------------------------------

def param_order_help(
    lower_bound_array,
    upper_bound_array,
    list_of_psd_params=["log10_A", "gamma"],
    fixed_psd_params=[None],
    fixed_psd_param_values=[None],
):
    """
    A utility function that helps organize and structure parameters
    related to gravitational wave background (non-GWB) PSD.

    :param lower_bound_array: the lower bound on the model params (PSD) as a JAX array
    :param upper_bound_array: the upper bound on the model params (PSD) as a JAX array
    :list_of_psd_params: a list containing the name of the PSD model parameters. The ordering
      ***MUST*** match those in `psd_functions.py`
    :fixed_psd_params: a list containing the non-GWB PSD parameters that you want to be fixed
    :fixed_psd_param_values: a JAX array containing the values of fixed non-GWB PSD params.

    """
    x = {}
    x.update({"orderedpsd_model_params": list_of_psd_params})
    if any(fixed_psd_params):
        fixed_psd_param_indxs = [
            list(list_of_psd_params).index(_) for _ in fixed_psd_params
        ]
        x.update({"fixed_psd_params": fixed_psd_params})

    x.update(
        {
            "varied_psd_params": [
                *[_ for _ in list_of_psd_params if _ not in fixed_psd_params],
            ]
        }
    )
    x.update({"psd_param_lower_lim": lower_bound_array})
    x.update({"psd_param_upper_lim": upper_bound_array})
    if any(fixed_psd_params):
        x.update({"fixed_psd_params": fixed_psd_params})
        x.update({"fixed_psd_param_indices": jnp.array(fixed_psd_param_indxs)})
        x.update({"fixed_psd_param_values": fixed_psd_param_values})
    return x


def broken_pl(
    renorm_const,
    lower_amp=-18.0,
    upper_amp=-11.0,
    lower_gamma=0.0,
    upper_gamma=7.0,
    lower_log10_fb=-8.7,
    upper_log10_fb=-7.0,
):
    """
    A lazy way to get the right `param_order_help` dictionary for a fixed gamma HD model
    """
    logamp_offset = logamp_offset = 0.5 * jnp.log10(renorm_const)
    chosen_psd_model = psd_functions.broken_powerlaw
    chosen_psd_model_params = np.array(
        [
            str(_)
            for _ in inspect.signature(chosen_psd_model).parameters
            if not "args" in str(_)
        ][2:]
    )
    return (
        chosen_psd_model,
        param_order_help(
            list_of_psd_params=chosen_psd_model_params,
            lower_bound_array=jnp.array(
                [lower_amp + logamp_offset, lower_gamma, lower_log10_fb]
            ),
            upper_bound_array=jnp.array(
                [upper_amp + logamp_offset, upper_gamma, upper_log10_fb]
            ),
            fixed_psd_params=["delta", "kappa"],
            fixed_psd_param_values=jnp.array([0.0, 0.1]),
        ),
    )

def varied_gamma_pl(
    renorm_const, 
    lower_amp=-18.0, 
    upper_amp=-11.0,
    lower_gamma=0.0, 
    upper_gamma=7.0
):
    """
    A lazy way to get the right `param_order_help` dictionary for a varied gamma HD model
    """
    logamp_offset = 0.5 * jnp.log10(renorm_const)
    chosen_psd_model = psd_functions.powerlaw
    chosen_psd_model_params = np.array(
        [
            str(_)
            for _ in inspect.signature(chosen_psd_model).parameters
            if not "args" in str(_)
        ][2:]
    )
    return (
        chosen_psd_model,
        param_order_help(
            list_of_psd_params=chosen_psd_model_params,
            lower_bound_array=jnp.array([lower_amp + logamp_offset, lower_gamma]),
            upper_bound_array=jnp.array([upper_amp + logamp_offset, upper_gamma]),
            fixed_psd_param_values=[],
        ),
    )

def spectrum(
    renorm_const, 
    crn_bins, 
    lower_halflog10_rho=-9, 
    upper_halflog10_rho=-1
    ):
    """
    A lazy way to get the right `param_order_help` dictionary for a free-spectral HD model
    """
    logamp_offset = 0.5 * jnp.log10(renorm_const)
    chosen_psd_model = psd_functions.free_spectrum
    chosen_psd_model_params = np.array(
        [
            str(_)
            for _ in inspect.signature(chosen_psd_model).parameters
            if not "args" in str(_)
        ][2:]
    )
    return (
        chosen_psd_model,
        param_order_help(
            list_of_psd_params=chosen_psd_model_params,
            lower_bound_array=jnp.ones(crn_bins)
            * (lower_halflog10_rho + logamp_offset),
            upper_bound_array=jnp.ones(crn_bins)
            * (upper_halflog10_rho + logamp_offset),
            fixed_psd_param_values=[],
        ),
    )