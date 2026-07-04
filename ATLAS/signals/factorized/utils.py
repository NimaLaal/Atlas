"""
irn_model_builder.py
====================
Utilities for constructing the ``(psd_func, irn_helper_dictionary)`` or
``(psd_func, dm_helper_dictionary)`` pairs consumed by ``RedNoise`` for
non-GWB (intrinsic red noise, DM, …) spectral components.

Design
------
Mirrors ``gwb_model_builder.py`` exactly.  All shortcuts are thin
``functools.partial`` wrappers around a single generic ``make_irn_model``
function.  Adding a new variant requires one ``partial`` line.
"""

import numpy as np
import inspect
from functools import partial

import jax.numpy as jnp

import ATLAS.psd_functions as psd_functions
from ATLAS.signals.signals_utils import _extract_fixed_from_partial, _sig_params

def _param_order_help(
    lower_bound_array,
    upper_bound_array,
    list_of_psd_params,
    fixed_psd_params=(),
    fixed_psd_param_values=(),
):
    """
    Build the ``irn_helper_dictionary`` / ``dm_helper_dictionary`` consumed
    by ``RedNoise``.

    Parameters
    ----------
    lower_bound_array, upper_bound_array : jnp.ndarray
        Prior bounds for the *varied* PSD parameters.
    list_of_psd_params : sequence[str]
        Names of all PSD parameters, in the order they appear in the PSD
        function signature (after ``f`` and ``df``).
    fixed_psd_params : sequence[str]
        Subset of ``list_of_psd_params`` to hold constant.
    fixed_psd_param_values : array-like
        Values for ``fixed_psd_params``, in the same order.
    """
    d = {}
    d["ordered_psd_model_params"] = np.array(list_of_psd_params)

    if fixed_psd_params:
        fixed_idx = [list(list_of_psd_params).index(p) for p in fixed_psd_params]
        d["fixed_psd_param_indices"] = jnp.array(fixed_idx)
        d["fixed_psd_param_values"]  = jnp.array(fixed_psd_param_values)

    d["psd_param_lower_lim"] = lower_bound_array
    d["psd_param_upper_lim"] = upper_bound_array
    return d


# ---------------------------------------------------------------------------
# Generic factory
# ---------------------------------------------------------------------------

def make_irn_model(
    psd_func,
    lower_bound_array,
    upper_bound_array,
    renorm_const=1.0,
    fixed_psd_params=(),
    fixed_psd_param_values=(),
):
    """
    Build the ``(psd_func, irn_helper_dictionary)`` pair for ``RedNoise``.

    This is the single generic entry-point.  All named shortcuts below are
    ``partial`` applications of this function.

    Parameters
    ----------
    psd_func : callable
        PSD function with signature ``(f, df, *params)``.
    lower_bound_array, upper_bound_array : array-like
        Prior bounds for the *varied* parameters.  A ``renorm_const``-derived
        log-amplitude offset is applied automatically to every element.
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
    irn_helper_dictionary : dict
    """
    psd_func, fixed_psd_params, fixed_psd_param_values = _extract_fixed_from_partial(
        psd_func, fixed_psd_params, fixed_psd_param_values, skip = 2,
    )

    logamp_offset = 0.5 * jnp.log10(renorm_const)
    lower = jnp.asarray(lower_bound_array) + logamp_offset
    upper = jnp.asarray(upper_bound_array) + logamp_offset

    psd_params = _sig_params(psd_func, skip=2)   # skip f, df

    helper = _param_order_help(
        lower_bound_array      = lower,
        upper_bound_array      = upper,
        list_of_psd_params     = psd_params,
        fixed_psd_params       = fixed_psd_params,
        fixed_psd_param_values = fixed_psd_param_values,
    )
    return psd_func, helper


# ---------------------------------------------------------------------------
# Named shortcuts via partial
# ---------------------------------------------------------------------------
# Each partial pre-fills ``psd_func`` and any fixed-parameter bookkeeping.
# The user still supplies the prior bounds (and optionally ``renorm_const``).
#
# Calling convention for all shortcuts:
#
#   model_func(lower_bound_array, upper_bound_array, renorm_const=1.0)
#     → (psd_func, irn_helper_dictionary)
#
# ---------------------------------------------------------------------------

# Power-law, both amplitude and gamma free
varied_gamma_pl = partial(
    make_irn_model,
    psd_func = psd_functions.powerlaw,
)

# Power-law, gamma fixed at 13/3 (IRN analogue of the HD GWB spectral index)
fixed_gamma_pl = partial(
    make_irn_model,
    psd_func               = psd_functions.powerlaw,
    fixed_psd_params       = ("gamma",),
    fixed_psd_param_values = (13 / 3,),
)

# Broken power-law, delta and kappa fixed
broken_pl = partial(
    make_irn_model,
    psd_func               = psd_functions.broken_powerlaw,
    fixed_psd_params       = ("delta", "kappa"),
    fixed_psd_param_values = (0.0, 0.1),
)

# Free-spectral (one log-PSD value per frequency bin)
spectrum = partial(
    make_irn_model,
    psd_func = psd_functions.free_spectrum,
)