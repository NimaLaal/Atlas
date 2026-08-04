"""
gwb_model_builder.py
====================
Utilities for constructing the ``(gwb_psd_func, orf_func, gwb_helper_dictionary)``
triple consumed by ``RedNoise``.

Design
------
All factory functions are built on a single generic ``make_gwb_model`` function.
The named shortcuts (``fixed_gamma_hd_pl``, ``varied_gamma_hd_pl``, …) are thin
``functools.partial`` wrappers that pre-fill the choices that define each model
variant, leaving only the user-facing knobs (prior bounds, ``renorm_const``) as
call-time arguments.

Adding a new model variant therefore requires only one ``partial`` line, with no
boilerplate duplication.
"""

import numpy as np
import inspect
from functools import partial

import jax.numpy as jnp

import ATLAS.psd_functions as psd_functions
from ATLAS.signals.signals_utils import _extract_fixed_from_partial, _sig_params


# ---------------------------------------------------------------------------
# Low-level helper: build the gwb_helper_dictionary
# ---------------------------------------------------------------------------

def _param_order_help(
    lower_bound_array,
    upper_bound_array,
    list_of_psd_params,
    lower_bound_orf = None,
    upper_bound_orf = None,
    list_of_orf_params=(),
    fixed_psd_params=(),
    fixed_psd_param_values=(),
    fixed_orf_params=(),          
    fixed_orf_param_values=(),    
):
    d = {}
    d["ordered_gwb_psd_model_params"] = np.array(list_of_psd_params)

    if list_of_orf_params:
        d["ordered_orf_model_params"] = np.array(list_of_orf_params)

    if fixed_psd_params:
        fixed_idx = [list(list_of_psd_params).index(p) for p in fixed_psd_params]
        d["fixed_gwb_psd_param_indices"] = jnp.array(fixed_idx)
        d["fixed_gwb_psd_param_values"]  = jnp.array(fixed_psd_param_values)

    if fixed_orf_params:
        fixed_idx = [list(list_of_orf_params).index(p) for p in fixed_orf_params]
        d["fixed_orf_param_indices"] = jnp.array(fixed_idx)
        d["fixed_orf_param_values"]  = jnp.array(fixed_orf_param_values)

    d["gwb_psd_param_lower_lim"] = lower_bound_array
    d["gwb_psd_param_upper_lim"] = upper_bound_array

    d["orf_param_lower_lim"] = lower_bound_orf
    d["orf_param_upper_lim"] = upper_bound_orf
    
    return d


def _sig_params(func, skip=2):
    """Return the non-``*args`` parameter names of ``func``, skipping the first ``skip``."""
    return np.array(
        [str(p) for p in inspect.signature(func).parameters if "args" not in str(p)][skip:]
    )


# ---------------------------------------------------------------------------
# Generic factory
# ---------------------------------------------------------------------------

def make_gwb_model(
    psd_func,
    orf_func,
    lower_bound_array,
    upper_bound_array,
    lower_bound_orf = None,
    upper_bound_orf = None,
    renorm_const=1.0,
    fixed_psd_params=(),
    fixed_psd_param_values=(),
    fixed_orf_params=(),
    fixed_orf_param_values=()
):
    """
    Build the ``(psd_func, orf_func, gwb_helper_dictionary)`` triple for ``RedNoise``.

    This is the single generic entry-point.  All named shortcuts below are
    ``partial`` applications of this function.

    Parameters
    ----------
    psd_func : callable
        GWB PSD function with signature ``(f, df, *params)``.
    orf_func : callable
        ORF function with signature ``(angle, *free_params)``.
        A fixed ORF (e.g. HD) takes only ``angle``; a free ORF (e.g. GT)
        also takes additional parameters — these are detected automatically
        from the signature.
    lower_bound_array, upper_bound_array : array-like
        Prior bounds for the *varied* parameters in the order::

            [ varied_psd_params]

        A ``renorm_const``-derived log-amplitude offset is applied
        automatically to every element (same convention as pandora).
    renorm_const : float
        Unit renormalisation constant.  Enters as
        ``0.5 * log10(renorm_const)`` added to the bounds.
    fixed_psd_params : sequence[str]
        Names of PSD parameters to hold fixed (must match the PSD function
        signature).
    fixed_psd_param_values : array-like
        Values for ``fixed_psd_params``, in the same order.

    Returns
    -------
    psd_func : callable
    orf_func : callable
    gwb_helper_dictionary : dict
    """
    psd_func, fixed_psd_params, fixed_psd_param_values = _extract_fixed_from_partial(
        psd_func, fixed_psd_params, fixed_psd_param_values, skip=2
    )
    orf_func, fixed_orf_params, fixed_orf_param_values = _extract_fixed_from_partial(
        orf_func, fixed_orf_params, fixed_orf_param_values, skip=1
    )

    logamp_offset = 0.5 * jnp.log10(renorm_const)
    lower = jnp.asarray(lower_bound_array) + logamp_offset
    upper = jnp.asarray(upper_bound_array) + logamp_offset

    psd_params = _sig_params(psd_func, skip=2)   # skip f, df
    orf_params = _sig_params(orf_func,  skip=1)   # skip angle

    helper = _param_order_help(
        lower_bound_array    = lower,
        upper_bound_array    = upper,
        lower_bound_orf      = lower_bound_orf,
        upper_bound_orf      = upper_bound_orf,
        list_of_psd_params   = psd_params,
        list_of_orf_params   = list(orf_params) if len(orf_params) else (),
        fixed_psd_params     = fixed_psd_params,
        fixed_psd_param_values = fixed_psd_param_values,
    )
    return psd_func, orf_func, helper


# ---------------------------------------------------------------------------
# Named shortcuts via partial
# ---------------------------------------------------------------------------
# Each partial pre-fills ``psd_func``, ``orf_func``, and any fixed-parameter
# bookkeeping.  The user still supplies the prior bounds (and optionally
# ``renorm_const``).
#
# Calling convention for all shortcuts:
#
#   model_func(lower_bound_array, upper_bound_array, renorm_const=1.0)
#     → (psd_func, orf_func, gwb_helper_dictionary)
#
# ---------------------------------------------------------------------------

# Power-law + HD ORF, gamma fixed at 13/3
fixed_gamma_hd_pl = partial(
    make_gwb_model,
    psd_func              = psd_functions.powerlaw,
    orf_func              = psd_functions.hd_orf,
    fixed_psd_params      = ("gamma",),
    fixed_psd_param_values= (13 / 3,),
)

# Power-law + HD ORF, gamma free
varied_gamma_hd_pl = partial(
    make_gwb_model,
    psd_func = psd_functions.powerlaw,
    orf_func = psd_functions.hd_orf,
)

# Broken power-law + HD ORF, delta and kappa fixed
broken_pl_hd = partial(
    make_gwb_model,
    psd_func              = psd_functions.broken_powerlaw,
    orf_func              = psd_functions.hd_orf,
    fixed_psd_params      = ("delta", "kappa"),
    fixed_psd_param_values= (0.0, 0.1),
)

# Free-spectral + HD ORF
hd_spectrum = partial(
    make_gwb_model,
    psd_func = psd_functions.free_spectrum,
    orf_func = psd_functions.hd_orf,
)

# Power-law + generalised-transverse ORF (free ORF parameter)
varied_gamma_gt_pl = partial(
    make_gwb_model,
    psd_func = psd_functions.powerlaw,
    orf_func = psd_functions.gt_orf,
)

# Power-law + CURN (monopole ORF), gamma free
varied_gamma_curn_pl = partial(
    make_gwb_model,
    psd_func = psd_functions.powerlaw,
    orf_func = psd_functions.monopole_orf,
)

# Power-law + dipole ORF, gamma free
varied_gamma_dipole_pl = partial(
    make_gwb_model,
    psd_func = psd_functions.powerlaw,
    orf_func = psd_functions.dipole_orf,
)