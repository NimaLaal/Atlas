from __future__ import annotations
import math
import re
import warnings
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

from jug.engine.session import TimingSession
from jug.utils.constants import K_DM_SEC, SECS_PER_DAY, C_KM_S, T_SUN_SEC
from jug.delays.combined import combined_delays
from jug.delays.barycentric_jax import (
    pulsar_direction_jax, roemer_delay_jax, shapiro_delay_jax,
)
from jug.model.dmx import parse_dmx_ranges, build_dmx_design_matrix
from jug.fitting.derivatives_jump import (
    parse_jump_from_par_line,
    create_jump_mask_from_flags,
    create_jump_mask_from_mjd_range,
)
jax.config.update('jax_enable_x64', True)
# tqdm_joblib / joblib are only needed for the optional parallel par/tim loader
# in build_multi_psr_timing_model; import lazily there so the module (and the
# single-pulsar path) imports even when those optional deps are absent.
from ATLAS.utils import jit, jit_method
from functools import partial
import jax.scipy.linalg as jsl

def _timing_model_svd(M):
    """Create an more stable basis for the timing model design matrix using SVD.

    This function is used to create a basis U which represents the timing model
    design matrix M and normalizes each of the basis vectors. This can be used
    as a more stable alternative to M in timing model marginalization. 

    This takes an SVD of the design matrix M, and returns only the left singular 
    vectors U as the new basis. This works since during the marginalization process,
    the singular values aren't important when integrating over the whole range of
    timing model coefficients. Likewise, the right singular vectors are only
    used to project into the original basis, which we do not need.

    Parameters
    ----------
    M : array
        The design matrix for the timing model. [ntoas, nparams]

    Returns
    -------
    array
        The left singular vectors of the design matrix M, which can be used as a more
        stable basis for timing model marginalization. [ntoas, nparams]
    """
    U,C,V = jnp.linalg.svd(M, full_matrices=False)
    # Return just the left singular vector.
    # The singular values are a weighting factor that isn't important when marginalizing.
    # the right singular vectors are used to project into the original basis, which isn't 
    # important for marginalization either. 
    return U

_MAS_YR_TO_RAD_DAY = (np.pi / 180.0 / 3.6e6) / 365.25

_BINARY_KIND = {
    '': 'none', 'NONE': 'none',
    'ELL1': 'ell1', 'ELL1H': 'ell1',
    'DD': 'dd', 'DDH': 'dd', 'DDGR': 'dd', 'DDK': 'dd',
    'BT': 'bt', 'BTX': 'bt',
}

# jug.delays.combined.combined_delays binary_model_id switch (jax.lax.switch order):
#   0=None, 1=ELL1, 2=DD, 3=T2, 4=BT, 5=DDK.  We route ELL1/DD/BT here; the
#   hand-rolled binary/astrometry kernels are retired in favour of
#   combined_delays (binary/DM/FD/SW) and barycentric_jax (astrometry).
_BINARY_MODEL_ID = {'none': 0, 'ell1': 1, 'dd': 2, 'bt': 4}

# Parameters wired into delay kernels below.
# Anything else is accepted (warned) but contributes zero delay.
_KERNEL_PARAMS = frozenset({
    'F0', 'F1', 'DM',
    'RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX',
    # Ecliptic coordinate aliases — remapped to equatorial internally by JUG.
    # They are accepted here and transparently forwarded to the same astrometry
    # kernel via _ecliptic_to_equatorial().  Sampling ELONG samples RAJ/DECJ.
    'ELONG', 'ELAT', 'PMELONG', 'PMELAT',
    'PB', 'A1',
    'TASC', 'EPS1', 'EPS2',
    'T0', 'ECC', 'OM',
    'M2', 'SINI',
    # Secondary binary / DM-Taylor params, wired through combined_delays as traced
    # args (DD/BT only for XDOT/OMDOT; DM1/DM2 extend the DM Taylor for all models).
    'XDOT', 'OMDOT', 'DM1', 'DM2',
})

# Patterns for parameter families handled generically.
_RE_FN    = re.compile(r'^F(\d+)$')         # F2, F3, …
_RE_DMX   = re.compile(r'^DMX_\d+$', re.IGNORECASE)
_RE_JUMP  = re.compile(r'^JUMP(\d+)$', re.IGNORECASE)
_RE_FD    = re.compile(r'^FD(\d+)$', re.IGNORECASE)  # FD1, FD2, FD3, …

# Ecliptic public names → JUG internal equatorial working keys
_ECL_TO_EQ: dict[str, str] = {
    'ELONG':   '_ecliptic_lon_deg',   # degrees; converted to _raj_rad by JUG
    'ELAT':    '_ecliptic_lat_deg',   # degrees; converted to _decj_rad by JUG
    'PMELONG': '_ecliptic_pm_lon',    # mas/yr along ecliptic lon
    'PMELAT':  '_ecliptic_pm_lat',    # mas/yr along ecliptic lat
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_param(params: dict, name: str) -> float:
    """Read any par-file parameter as float, handling coordinate aliases.

    RAJ/DECJ  → JUG internal radian keys (_raj_rad/_decj_rad).
    ELONG/ELAT/PMELONG/PMELAT → JUG internal ecliptic keys (_ecliptic_*).
      The ecliptic values are in degrees (lon/lat) or mas/yr (proper motion).
      The astrometry kernel receives the equatorial equivalents computed by
      _ecliptic_to_equatorial(), not these raw values, so theta_0_full stores
      the ecliptic values purely for computing deltas.
    """
    _aliases = {
        'RAJ':  '_raj_rad',
        'DECJ': '_decj_rad',
        **_ECL_TO_EQ,   # ELONG → _ecliptic_lon_deg, etc.
    }
    key = _aliases.get(name, name)
    val = params.get(key, 0.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _classify_param(name: str) -> str:
    """Return the class of a parameter: 'kernel', 'fn', 'dmx', 'jump', 'fd', 'unknown'."""
    if name in _KERNEL_PARAMS:
        return 'kernel'
    if _RE_FN.match(name):
        return 'fn'
    if _RE_DMX.match(name):
        return 'dmx'
    if _RE_JUMP.match(name):
        return 'jump'
    if _RE_FD.match(name):
        return 'fd'
    return 'unknown'


def _make_ecl_to_eq_fn(params_base: dict):
    """Return a pure-JAX function that rotates ecliptic → equatorial.

    The obliquity rotation matrix is computed once from the par file (a Python
    float) and baked in as a compile-time constant.  The returned function
    accepts JAX tracers for lon_deg, lat_deg, pm_lon, pm_lat and is safe to
    call inside @jax.jit.

    Parameters
    ----------
    params_base : dict
        The JUG params dict; used only to read the ECL frame string at setup time.

    Returns
    -------
    ecl_to_eq : callable(lon_deg, lat_deg, pm_lon, pm_lat)
                → (ra_rad, dec_rad, pmra_masyr, pmdec_masyr)
        All arguments and return values are JAX scalars (traceable).
    """
    from jug.io.par_reader import OBLIQUITY_ARCSEC
    ecl_frame  = str(params_base.get('_ecliptic_frame',
                                     params_base.get('ECL', 'IERS2010'))).upper()
    obl_arcsec = OBLIQUITY_ARCSEC.get(ecl_frame, OBLIQUITY_ARCSEC['IERS2010'])
    obl_rad    = obl_arcsec * np.pi / (180.0 * 3600.0)
    # Frozen Python floats — become compile-time constants inside jit.
    _cos_obl   = float(np.cos(obl_rad))
    _sin_obl   = float(np.sin(obl_rad))
    _deg2rad   = float(np.pi / 180.0)
    _twopi     = float(2.0 * np.pi)

    def ecl_to_eq(lon_deg, lat_deg, pm_lon, pm_lat):
        """Pure-JAX ecliptic → equatorial rotation.  Safe inside @jax.jit."""
        lon_rad = lon_deg * _deg2rad
        lat_rad = lat_deg * _deg2rad

        cos_lon = jnp.cos(lon_rad);  sin_lon = jnp.sin(lon_rad)
        cos_lat = jnp.cos(lat_rad);  sin_lat = jnp.sin(lat_rad)

        # Ecliptic → ICRS Cartesian rotation (obliquity about x-axis)
        x =  cos_lon * cos_lat
        y =  sin_lon * cos_lat * _cos_obl - sin_lat * _sin_obl
        z =  sin_lon * cos_lat * _sin_obl + sin_lat * _cos_obl

        ra_rad  = jnp.arctan2(y, x) % _twopi
        dec_rad = jnp.arctan2(z, jnp.sqrt(x**2 + y**2))

        # Proper-motion rotation (same obliquity matrix, Jacobian)
        cos_ra  = jnp.cos(ra_rad);   sin_ra  = jnp.sin(ra_rad)
        cos_dec = jnp.cos(dec_rad);  sin_dec = jnp.sin(dec_rad)

        dx    = -sin_lon * pm_lon - cos_lon * sin_lat * pm_lat
        dy    =  cos_lon * pm_lon - sin_lon * sin_lat * pm_lat
        dz    =  cos_lat * pm_lat
        dx_eq =  dx
        dy_eq =  dy * _cos_obl - dz * _sin_obl
        dz_eq =  dy * _sin_obl + dz * _cos_obl

        pmra_masyr  = -sin_ra * dx_eq + cos_ra * dy_eq
        pmdec_masyr = (-cos_ra * sin_dec * dx_eq
                       - sin_ra * sin_dec * dy_eq
                       + cos_dec * dz_eq)

        return ra_rad, dec_rad, pmra_masyr, pmdec_masyr

    return ecl_to_eq


def _build_dmx_masks(params: dict, tdb_mjd_np: np.ndarray,
                     freq_mhz_np: np.ndarray, mjd_utc_np: np.ndarray
                     ) -> dict[str, jnp.ndarray]:
    """
    Precompute one boolean JAX mask per DMX window from the par file.

    The key matches the canonical label returned by parse_dmx_ranges
    (e.g. 'DMX_0001').  Mask is True for every TOA in the window.
    """
    dmx_ranges = parse_dmx_ranges(params)
    masks = {}
    for rng in dmx_ranges:
        mask_np = (mjd_utc_np >= rng.r1_mjd) & (mjd_utc_np <= rng.r2_mjd)
        masks[rng.label] = jnp.array(mask_np, dtype=jnp.bool_)
    return masks


def _build_jump_masks(params: dict, toas_data: list,
                      tdb_mjd_np: np.ndarray) -> dict[str, jnp.ndarray]:
    """
    Precompute one boolean JAX mask per JUMP line from the par file.

    Keys are 'JUMP1', 'JUMP2', … matching the numbered convention used
    throughout the JUG engine (session.py writes JUMP{n} into params).
    """
    jump_lines = params.get('_jump_lines', [])
    masks = {}
    toa_flags = [getattr(t, 'flags', {}) for t in toas_data]

    for idx, jline in enumerate(jump_lines, start=1):
        key = f'JUMP{idx}'
        jinfo = parse_jump_from_par_line(jline)
        if jinfo['type'] == 'flag':
            mask_np = create_jump_mask_from_flags(
                toa_flags, jinfo['flag_name'], jinfo['flag_value'])
        elif jinfo['type'] == 'mjd':
            mask_np = np.array(
                create_jump_mask_from_mjd_range(
                    jnp.array(tdb_mjd_np),
                    jinfo['mjd_start'], jinfo['mjd_end']),
                dtype=bool)
        else:
            mask_np = np.zeros(len(tdb_mjd_np), dtype=bool)
        masks[key] = jnp.array(mask_np, dtype=jnp.bool_)

    return masks


# ---------------------------------------------------------------------------
# Physical reparametrisations (NaN-gradient guard for the JUG kernels)
# ---------------------------------------------------------------------------
# combined_delays evaluates  log(1 - SINI·sinΦ)  and  sqrt(1 - ECC²).  Because
# jnp.where evaluates BOTH branches, a SINI>1 or ECC≥1 proposal produces a NaN
# that poisons the gradient even when the runtime guard would have selected 0 —
# fatal for NUTS.  We therefore map the SAMPLED (unconstrained, real) coordinate
# u = z·σ to a PHYSICAL value that is *always* in range, for every u ∈ ℝ, with a
# finite derivative everywhere.  This is stronger than a bounded prior: even an
# out-of-support proposal can never reach the kernel.
#
#   SINI : sample cos i.  cosi = tanh(u + atanh(cosi₀)) ∈ (-1,1)
#          → SINI = sqrt(1 - cosi²) = sech(u+c) ∈ (0,1].  Uses cos i (not
#          sin(angle)) so the sampler sits away from the SINI=1 edge where
#          log(1 - SINI·sinΦ) → log(0).
#   ECC  : ecc = ½(1 + tanh(u + atanh(2ecc₀-1))) ∈ (0,1).  Never reaches 1.
#   M2,PX: positive via log-space, M2 = M2₀·exp(u) (or softplus if par value ≤0).
#   else : plain affine  x₀ + u   (F0, F1, DM, OM, EPS1/2, … unchanged).
#
# u = z·σ is centred so that u=0 reproduces the par-file value x₀.
_REPARAM_PARAMS = frozenset({'SINI', 'ECC', 'M2', 'PX'})

_EPS_BND = 1e-9  # keep atanh arguments strictly inside (-1, 1)


def reparametrise(name: str, x0: float, u):
    """Map a sampled unconstrained coordinate ``u`` → physical parameter value.

    ``u`` is the scaled, mean-zero sampler coordinate (u = z·σ); ``x0`` is the
    par-file value reproduced at u=0.  Returns a JAX-traceable value guaranteed
    to lie in the parameter's physical domain for every real ``u``, with finite
    gradient everywhere — so the downstream JUG delay kernels never see a
    log/sqrt of a negative.
    """
    if name == 'SINI':
        # Monotonic inclination map: sample cos i = tanh(u+c) ∈ (-1,1), one-to-one
        # in the sampler coordinate, then SINI = sqrt(1 - cosi²).  Do NOT use the
        # algebraic shortcut sech(u+c): cosh is even, so ±(u+c) alias to the same
        # SINI — a two-to-one map that plants a mirror mode in the posterior.
        # cosi is clipped to ±(1-ε) so 1-cosi² ≥ ε(2-ε) > 0 and the sqrt gradient
        # never reaches the singular endpoint (tanh only hits ±1 at infinite u;
        # clip guards float64 saturation anyway).
        cosi0 = jnp.sqrt(jnp.clip(1.0 - x0**2, 0.0, 1.0))
        c     = jnp.arctanh(jnp.clip(cosi0, -1.0 + _EPS_BND, 1.0 - _EPS_BND))
        cosi  = jnp.clip(jnp.tanh(u + c), -1.0 + _EPS_BND, 1.0 - _EPS_BND)
        return jnp.sqrt(1.0 - cosi**2)                # ∈ (0, 1)
    if name == 'ECC':
        c   = jnp.arctanh(jnp.clip(2.0 * x0 - 1.0, -1.0 + _EPS_BND, 1.0 - _EPS_BND))
        return 0.5 * (1.0 + jnp.tanh(u + c))         # ∈ (0, 1)
    if name in ('M2', 'PX'):
        # log-space positivity.  If the par value is ≤0 (param absent), fall back
        # to softplus so u=0 still gives a small positive number with finite grad.
        return jnp.where(
            x0 > 0.0,
            x0 * jnp.exp(u),
            jnp.logaddexp(0.0, u),                   # softplus(u) > 0
        )
    return x0 + u                                     # plain affine (unchanged)


# ---------------------------------------------------------------------------
# Canonical bounded-prior sampling — THE single prior mechanism (production+tests)
# ---------------------------------------------------------------------------
# Per-parameter PROPER bounded priors on the PHYSICAL coordinate.  Established as
# the robust path in the weak-data prereq: improper-z + log|dθ/dz| Jacobian blew
# up (NUTS escaping to |z|→∞ at the SINI→0 boundary); proper bounded priors give
# nominal coverage.  No Jacobian factor is used anywhere — the prior lives
# entirely in the bounded distribution on the physical coordinate.
#
#   SINI : cos i ~ Uniform(-1, 1)  →  SINI = sqrt(1 - cos²i)   (isotropic incl.)
#   ECC  : Uniform(0, 1-1e-6)
#   M2   : log-uniform over [M2_LOG_LO, M2_LOG_HI]   (positivity; solar masses)
#   PX   : Uniform(0, PX_MAX) mas        (flat, inert — no scale preference)
#   else : affine  theta0 + z·σ_mle,  z ~ Normal(0, AFFINE_Z_SD)   (well-constrained)
#
# M2 and PX use FLAT (uniform) physical-range priors, deliberately inert for
# recovery: log-uniform pulls toward low mass / small parallax (scale-informative)
# and would bias an informative-data param.  These are VALIDATION priors — the
# production science-run prior choice is a separate, later decision.
#
# reparametrise() above is NOT used here — that is the in-domain transform for the
# unconstrained z_concat path only, and carries no prior.
_M2_BOUNDS = (0.0, 3.0)        # M2 ∈ [0, 3] Msun  (J1909 truth ≈0.21 → 7% of range)
_PX_BOUNDS = (0.0, 10.0)       # PX ∈ [0, 10] mas  (J1909 truth ≈1.0 → 10% of range)
_ECC_HI    = 1.0 - 1e-6
_AFFINE_Z_SD = 8.0


def sample_timing_theta(sample_list, theta_0, sigma, prefix=''):
    """numpyro: draw each sampled timing param from its proper bounded prior.

    THE single prior mechanism — called by both the production MultiPsrTimingModel
    block and the validation harnesses, so every timing param is sampled the same
    way.  Returns ``{param_name: value}`` (physical units), and registers each as
    a numpyro.deterministic for post-processing.

    Parameters
    ----------
    sample_list : list[str]     params to sample (physical names)
    theta_0     : dict          par-file values (affine centre / init)
    sigma       : dict          linearised-MLE σ per param (affine scale)
    prefix      : str           name prefix (e.g. per-pulsar 'p0_') to avoid clashes
    """
    import numpyro
    import numpyro.distributions as dist
    theta = {}
    for k in sample_list:
        if k == 'SINI':
            cosi = numpyro.sample(f'{prefix}cosi_{k}', dist.Uniform(-1.0, 1.0))
            theta[k] = jnp.sqrt(jnp.clip(1.0 - cosi ** 2, 1e-12, 1.0))
            numpyro.deterministic(f'{prefix}{k}', theta[k])   # transformed → emit
        elif k == 'ECC':
            theta[k] = numpyro.sample(f'{prefix}{k}', dist.Uniform(0.0, _ECC_HI))   # site == name
        elif k == 'M2':
            theta[k] = numpyro.sample(f'{prefix}{k}', dist.Uniform(*_M2_BOUNDS))    # flat, site == name
        elif k == 'PX':
            theta[k] = numpyro.sample(f'{prefix}{k}', dist.Uniform(*_PX_BOUNDS))    # flat, site == name
        else:                                          # affine, wide Normal in z
            s = max(float(sigma[k]), 1e-30)
            z = numpyro.sample(f'{prefix}z_{k}', dist.Normal(0.0, _AFFINE_Z_SD))
            theta[k] = theta_0[k] + z * s
            numpyro.deterministic(f'{prefix}{k}', theta[k])   # transformed → emit
    return theta


def timing_init_values(sample_list, theta_0, prefix=''):
    """Matching init dict (init_to_value) for ``sample_timing_theta`` priors."""
    iv = {}
    for k in sample_list:
        if k == 'SINI':
            iv[f'{prefix}cosi_{k}'] = float(np.sqrt(max(1.0 - theta_0[k] ** 2, 0.0)))
        elif k == 'ECC':
            iv[f'{prefix}{k}'] = float(min(max(theta_0[k], 0.0), _ECC_HI))
        elif k == 'M2':
            iv[f'{prefix}{k}'] = float(min(max(theta_0[k], _M2_BOUNDS[0] + 1e-6), _M2_BOUNDS[1] - 1e-6))
        elif k == 'PX':
            iv[f'{prefix}{k}'] = float(min(max(theta_0[k], _PX_BOUNDS[0] + 1e-6), _PX_BOUNDS[1] - 1e-6))
        else:
            iv[f'{prefix}z_{k}'] = 0.0
    return iv


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_timing_model(par_path: str, tim_path: str,
                       sample_list: list[str] | None = None,
                       marginalise_list: list[str] | None = None,
                       *,
                       marginalise_all: bool = True,
                       modes: dict[str, str] | None = None):
    """Build a JIT'd JAX delta-m function with an explicit per-parameter split.

    Each fittable JUG parameter is placed in exactly one of three modes:
      • 'sample'      — varied nonlinearly through ``delta_m_us`` (sampled z).
      • 'marginalise' — carried as a column of the analytic linear basis ``M``.
      • 'fix'         — held at its par-file value (no column, not sampled).

    Parameters
    ----------
    par_path, tim_path : str
    sample_list : list[str], optional
        Params to sample nonlinearly.
    marginalise_list : list[str], optional
        Params to marginalise linearly.  If given, ONLY these (plus OFFSET) are
        marginalised and every other non-sampled label is fixed.
    marginalise_all : bool, default True
        When ``marginalise_list`` is None: marginalise every fittable JUG label
        not in ``sample_list`` (the historical behaviour).  When False, nothing
        is marginalised except OFFSET (all non-sampled params fixed).
    modes : dict[str, str], optional
        Direct per-param mapping {param: 'sample'|'marginalise'|'fix'}.  When
        given it is authoritative and ``sample_list`` / ``marginalise_list`` /
        ``marginalise_all`` are ignored.  Params absent from the dict are fixed.

    OFFSET is always marginalised (kept in M); it is never sampled or fixed.

    Returns
    -------
    delta_m_us : jit-compiled callable(theta_dict) → jnp.ndarray  [µs]
    aux : dict
        theta_0, errors_us, r_obs_us, tdb_mjd, mle, sample_list,
        marginalise_list, modes, binary_model, n_toa, dmx_masks, jump_masks,
        Mmat, Mmat_param_names
    """
    # ---- 1. Run JUG --------------------------------------------------------
    session = TimingSession(par_path, tim_path, verbose=False)
    result  = session.compute_residuals(subtract_tzr=False)
    params  = session.params

    result_dummy = session.fit_parameters(max_iter=5)
    # Raw design matrix — PINT-matching sign/units; columns == design_matrix_labels
    # (OFFSET first, then every free fit param).
    _design_matrix_raw = np.asarray(result_dummy['design_matrix'])    # (n_toa, n_col)
    _design_labels_raw = list(result_dummy['design_matrix_labels'])   # incl. 'OFFSET'
    _fittable = [l for l in _design_labels_raw if l != 'OFFSET']       # real JUG labels
    _fittable_set = set(_fittable)

    # ---- Resolve the three-way mode assignment ----------------------------
    if modes is not None:
        if sample_list or marginalise_list:
            raise ValueError("Pass either `modes` OR (`sample_list`/"
                             "`marginalise_list`), not both — they would conflict.")
        bad_mode = {k: v for k, v in modes.items()
                    if v not in ('sample', 'marginalise', 'fix')}
        if bad_mode:
            raise ValueError(f"modes: invalid mode(s) {bad_mode}; "
                             "use 'sample' | 'marginalise' | 'fix'.")
        sample_set = {k for k, v in modes.items() if v == 'sample'}
        marg_set   = {k for k, v in modes.items() if v == 'marginalise'}
    else:
        sample_set = set(sample_list or [])
        if marginalise_list is not None:
            marg_set = set(marginalise_list)
        elif marginalise_all:
            marg_set = _fittable_set - sample_set          # historical default
        else:
            marg_set = set()

    # ---- Validation: real labels, exactly one mode ------------------------
    # Every requested param must match a real JUG design label — this catches
    # the ECC-vs-E / XDOT-vs-A1DOT silent-mismatch case.
    requested = sample_set | marg_set
    unmatched = sorted(requested - _fittable_set)
    if unmatched:
        import difflib
        hints = {p: difflib.get_close_matches(p, _fittable, n=3) for p in unmatched}
        raise ValueError(
            f"Requested timing params not found among JUG design labels: {unmatched}. "
            f"Closest real labels: {hints}.  Fittable labels: {sorted(_fittable)}")
    overlap = sorted(sample_set & marg_set)
    if overlap:
        raise ValueError(f"Params {overlap} are in BOTH sample and marginalise "
                         "modes; each param must be in exactly one mode.")

    sample_list = [k for k in _fittable if k in sample_set]   # canonical order
    fixed_set   = _fittable_set - sample_set - marg_set
    mode_map = ({k: 'sample' for k in sample_set}
                | {k: 'marginalise' for k in marg_set}
                | {k: 'fix' for k in fixed_set})

    unknown = [k for k in sample_list if _classify_param(k) == 'unknown']
    if unknown:
        warnings.warn(
            f"Sampled params {unknown} have no delay kernel; they will appear in "
            "the Jacobian/MLE with zero columns and cannot be recovered.  Add a "
            "kernel branch to m_total_sec() to give them physical meaning.",
            stacklevel=2,
        )

    # ---- Build M from EXACTLY the marginalised set (strip BEFORE the SVD) --
    # A sampled param must never appear in M (double-counting); a fixed param is
    # dropped from M too (held at par).  OFFSET is always retained.  The strip
    # acts on the RAW design matrix — its columns ARE the parameters; after the
    # SVD, U's columns mix every parameter and can no longer be split per-param.
    _linear_keep_idx = [i for i, lab in enumerate(_design_labels_raw)
                        if lab == 'OFFSET' or lab in marg_set]
    Mmat_param_names = [_design_labels_raw[i] for i in _linear_keep_idx]
    Mmat             = jnp.asarray(_design_matrix_raw[:, _linear_keep_idx])  # restricted; (n,0) if none
    if not _linear_keep_idx and marg_set:
        raise ValueError(f"marginalise_list {sorted(marg_set)} matched no design "
                         "columns — internal error.")
    # Empty M (fully-nonlinear fit, no OFFSET column) is allowed: the constant
    # phase offset is absorbed by delta_m_us's weighted-mean subtraction, so no
    # linear basis is required.  _timing_model_svd handles the (n_toa, 0) case.

    binary_model = params.get('BINARY', '').upper()
    binary_kind  = _BINARY_KIND.get(binary_model)
    if binary_kind is None:
        raise NotImplementedError(
            f"Binary model {binary_model!r} not supported.  "
            f"Supported: {sorted(set(_BINARY_KIND) - {''})} or no binary."
        )
    has_binary = binary_kind != 'none'

    # ---- 2. Cache fixed arrays --------------------------------------------
    n_toa       = result['n_toas']
    tdb_mjd_ld  = np.asarray(result['tdb_mjd'], dtype=np.longdouble)
    mjd_utc_np  = np.asarray(result.get('mjd_utc', result['tdb_mjd']), dtype=np.float64)
    PEPOCH      = np.longdouble(params['PEPOCH'])
    dt_sec      = jnp.asarray(np.asarray(
        (tdb_mjd_ld - PEPOCH) * np.longdouble(SECS_PER_DAY), dtype=np.float64))
    tdb_mjd     = jnp.asarray(np.asarray(tdb_mjd_ld, dtype=np.float64))
    freq_mhz    = jnp.asarray(np.asarray(result['freq_bary_mhz'],  dtype=np.float64))
    errors_us   = jnp.asarray(np.asarray(result['errors_us'],      dtype=np.float64))
    weights     = 1.0 / errors_us**2
    ssb_obs_km  = jnp.asarray(np.asarray(result['ssb_obs_pos_ls'], dtype=np.float64) * C_KM_S)
    obs_sun_km  = jnp.asarray(np.asarray(result['obs_sun_pos_ls'], dtype=np.float64) * C_KM_S)
    sw_delay    = jnp.asarray(np.asarray(result.get('sw_delay_sec',    np.zeros(n_toa)), dtype=np.float64))
    tropo_delay = jnp.asarray(np.asarray(result.get('tropo_delay_sec', np.zeros(n_toa)), dtype=np.float64))
    r_obs_us    = jnp.asarray(np.asarray(result['residuals_us'],   dtype=np.float64))

    POSEPOCH_0 = float(params.get('POSEPOCH', params['PEPOCH']))
    # Proper-motion baseline (t - POSEPOCH) reduced in longdouble on the host,
    # then float64 — fed to pulsar_direction_jax so the JAX trace never forms the
    # years-long subtraction itself (matches the NumPy barycentric precision path).
    dt_pos_days = jnp.asarray(np.asarray(
        tdb_mjd_ld - np.longdouble(POSEPOCH_0), dtype=np.float64))

    # Frozen secondary binary params
    GAMMA_0 = float(params.get('GAMMA', 0.0))
    PBDOT_0 = float(params.get('PBDOT', 0.0))
    OMDOT_0 = float(params.get('OMDOT', 0.0))
    XDOT_0  = float(params.get('XDOT', params.get('A1DOT', 0.0)))
    EDOT_0  = float(params.get('EDOT', 0.0))

    # ---- combined_delays wiring (binary + DM + FD + solar wind) ------------
    # combined_delays carries the binary epoch (TASC for ELL1, T0 for DD/BT)
    # baked into the precomputed tt_binary_sec array, NOT as a traced argument.
    # We rebuild tt_binary_sec inside the JAX trace from the (possibly sampled)
    # epoch param so its physics is unchanged from the hand-rolled path.
    binary_model_id = _BINARY_MODEL_ID[binary_kind]
    BINARY_EPOCH_NAME = {'ell1': 'TASC', 'dd': 'T0', 'bt': 'T0'}.get(binary_kind)
    DMEPOCH_0 = float(params.get('DMEPOCH', params['PEPOCH']))
    # DM Taylor: dm_eff = DM + DM1·dt + DM2·dt²/2 with dt in years from DMEPOCH
    # (combined_delays forms dt_years = (tdb − dm_epoch)/365.25 internally).  The
    # factorials [0!,1!,2!] = [1,1,2] match PINT's DM(t) convention.  DM1/DM2 are
    # frozen at par values unless sampled, so when frozen they contribute the SAME
    # per-TOA term to m(θ) and m(θ₀) and cancel in the θ₀-referenced delta — the
    # pass-1 regression (DM1=DM2=0 there) stays bit-identical.
    _DM_FACTORIALS = jnp.asarray([1.0, 1.0, 2.0], dtype=jnp.float64)
    # FD orbital-frequency (FB) path is unused here — pass empty arrays so the
    # PB/T0 branch is selected (use_fb=False).
    _FB_EMPTY      = jnp.asarray([], dtype=jnp.float64)

    # ---- 3. Higher-order spindown: collect all Fn from par -----------------
    fn_orders_full: list[int] = [0, 1]  # F0, F1 always present
    k = 2
    while f'F{k}' in params:
        fn_orders_full.append(k)
        k += 1
    fn_0_full = {n: float(params.get(f'F{n}', 0.0)) for n in fn_orders_full}

    # ---- 3b. FD profile-evolution coefficients ----------------------------
    # FD delay: Σ_n FDn × log(ν_GHz)^n  [seconds]
    fd_orders_full: list[int] = []
    k = 1
    while f'FD{k}' in params:
        fd_orders_full.append(k)
        k += 1
    fd_0_full = {n: float(params.get(f'FD{n}', 0.0)) for n in fd_orders_full}

    # ---- 3c. Ecliptic coordinate mode -------------------------------------
    is_ecliptic = bool(params.get('_ecliptic_coords', False))
    _ECL_SAMPLE_KEYS = frozenset({'ELONG', 'ELAT', 'PMELONG', 'PMELAT'})
    has_ecliptic_sample = is_ecliptic and bool(_ECL_SAMPLE_KEYS & set(sample_list))
    # Build the pure-JAX rotation function once (obliquity baked in as constant).
    ecl_to_eq_jax = _make_ecl_to_eq_fn(params) if is_ecliptic else None

    # ---- 4. DMX masks (frozen, precomputed outside JAX) --------------------
    freq_mhz_np = np.asarray(result['freq_bary_mhz'], dtype=np.float64)
    tdb_mjd_np  = np.asarray(tdb_mjd_ld, dtype=np.float64)
    dmx_masks   = _build_dmx_masks(params, tdb_mjd_np, freq_mhz_np, mjd_utc_np)

    # DM derivative: K_DM / ν² — same for every DMX window, frozen
    dm_deriv    = jnp.asarray(K_DM_SEC / freq_mhz_np**2, dtype=jnp.float64)

    # Frozen DMX baseline values from par
    dmx_0 = {label: float(params.get(label, 0.0)) for label in dmx_masks}

    # ---- 5. JUMP masks (frozen, precomputed from _jump_lines + TOA flags) --
    jump_masks = _build_jump_masks(params, session.toas_data, tdb_mjd_np)

    # Frozen JUMP baseline values from params (JUG stores them as JUMP1, JUMP2, …)
    jump_0 = {key: float(params.get(key, 0.0)) for key in jump_masks}

    # ---- 6. Build θ₀ dynamically for every requested parameter -------------
    theta_0_full: dict[str, float] = {}
    for k_name in sample_list:
        cls = _classify_param(k_name)
        if cls == 'dmx':
            theta_0_full[k_name] = dmx_0.get(k_name, 0.0)
        elif cls == 'jump':
            theta_0_full[k_name] = jump_0.get(k_name, 0.0)
        elif cls == 'fn':
            order = int(_RE_FN.match(k_name).group(1))
            theta_0_full[k_name] = fn_0_full.get(order, 0.0)
        elif cls == 'fd':
            order = int(_RE_FD.match(k_name).group(1))
            theta_0_full[k_name] = fd_0_full.get(order, 0.0)
        else:
            theta_0_full[k_name] = _read_param(params, k_name)
    # Fallback pool for _g() — every kernel param and every Fn/FDn in par
    for k_name in _KERNEL_PARAMS:
        if k_name not in theta_0_full:
            theta_0_full[k_name] = _read_param(params, k_name)
    for order in fn_orders_full:
        fname = f'F{order}'
        if fname not in theta_0_full:
            theta_0_full[fname] = fn_0_full[order]
    for order in fd_orders_full:
        fname = f'FD{order}'
        if fname not in theta_0_full:
            theta_0_full[fname] = fd_0_full[order]
    # Frozen baselines for the secondary binary / DM-Taylor params.  XDOT is the
    # JUG design label for A1DOT, so read it via the XDOT_0 fallback (par stores
    # A1DOT); OMDOT/DM1/DM2 are par-native.  These override any _read_param zero.
    theta_0_full['XDOT']  = XDOT_0
    theta_0_full['OMDOT'] = OMDOT_0
    theta_0_full['DM1']   = float(params.get('DM1', 0.0))
    theta_0_full['DM2']   = float(params.get('DM2', 0.0))

    F0_0   = theta_0_full['F0']
    sample_set = set(sample_list)

    def _g(theta: dict, k_name: str):
        """Return sampled value (traced) or frozen Python float constant."""
        return theta[k_name] if k_name in sample_set else theta_0_full[k_name]

    # ---- 7. Total timing model in seconds ----------------------------------
    @jax.jit
    def m_total_sec(theta: dict) -> jnp.ndarray:
        # --- Astrometry ---
        # When ecliptic params are sampled we rotate the traced ecliptic values
        # to equatorial inside JAX.  ecl_to_eq_jax is a closure whose obliquity
        # constants were baked in at setup time, so it is fully traceable.
        if has_ecliptic_sample:
            lon_deg = _g(theta, 'ELONG')
            lat_deg = _g(theta, 'ELAT')
            pm_lon  = _g(theta, 'PMELONG')
            pm_lat  = _g(theta, 'PMELAT')
            RA, DEC, PMRA, PMDEC = ecl_to_eq_jax(lon_deg, lat_deg, pm_lon, pm_lat)
        else:
            RA    = _g(theta, 'RAJ');   DEC   = _g(theta, 'DECJ')
            PMRA  = _g(theta, 'PMRA'); PMDEC  = _g(theta, 'PMDEC')

        F0    = _g(theta, 'F0');   F1    = _g(theta, 'F1')
        DM    = _g(theta, 'DM')
        PX    = _g(theta, 'PX')
        PB    = _g(theta, 'PB');   A1    = _g(theta, 'A1')
        TASC  = _g(theta, 'TASC')
        EPS1  = _g(theta, 'EPS1'); EPS2  = _g(theta, 'EPS2')
        T0    = _g(theta, 'T0');   ECC   = _g(theta, 'ECC'); OM = _g(theta, 'OM')
        M2    = _g(theta, 'M2');   SINI  = _g(theta, 'SINI')
        DM1   = _g(theta, 'DM1');  DM2   = _g(theta, 'DM2')
        XDOT  = _g(theta, 'XDOT'); OMDOT = _g(theta, 'OMDOT')

        # Astrometry now sourced from the JUG JAX twins (jug.delays.barycentric_jax),
        # mirroring JUG's NumPy barycentric path.  dt_pos_days is the host-side
        # longdouble-reduced (t - POSEPOCH).  NOTE: the twin uses RIGOROUS
        # great-circle proper-motion propagation (matching PINT/ERFA), where the
        # retired hand-rolled _pulsar_direction used the linear tangent-plane
        # update — they differ at O((PM·dt)²).  For frozen astrometry (the recovery
        # configs) L_hat is identical in m(θ) and m(θ₀), so roemer/shapiro cancel
        # in the θ₀-referenced delta and the change is correctness-neutral there.
        L_hat   = pulsar_direction_jax(dt_pos_days, RA, DEC,
                                       PMRA  * _MAS_YR_TO_RAD_DAY,
                                       PMDEC * _MAS_YR_TO_RAD_DAY)
        roemer  = roemer_delay_jax(ssb_obs_km, L_hat, PX)
        shapiro = shapiro_delay_jax(obs_sun_km, L_hat, T_SUN_SEC)

        # --- Binary + DM + FD + solar wind via JUG combined_delays ----------
        # combined_delays returns (dm_sec + sw_sec + fd_sec + binary_sec); it does
        # NOT include roemer_shapiro in its output (that is added in the return).
        # roemer_shapiro is consumed only inside the kernel, to form the pre-binary
        # delay that shifts the binary time.  The binary epoch (TASC/T0) is carried
        # by the precomputed tt_binary_sec, rebuilt here from the (possibly sampled)
        # epoch so the physics matches the retired hand-rolled kernels exactly.
        #
        # Neutrality wiring (reproduces the hand-rolled path):
        #   • DM      : single constant coeff [DM] → dm_sec = K_DM·DM/ν²  (DM1/DM2
        #               were not modelled by the hand-rolled path).
        #   • FD      : fd_coeffs = [FD1, FD2, …] → Σ FDn·log(ν/GHz)ⁿ.  Absolute
        #               vs the old (FDn−FDn₀) form, but identical in delta_m_us
        #               (the θ₀ subtraction removes the constant per-TOA offset).
        #   • sol.W   : ne_sw = 0; the frozen precomputed sw_delay is folded into
        #               the tropo (prebinary-only) slot — it shifts the binary time
        #               but is NOT returned, matching the hand-rolled return.
        #   • DMX     : dmx_sec = None — the hand-rolled binary pre-delay excluded
        #               DMX; the DMX offsets are added separately below.
        #   • ELL1    : secondaries (PBDOT/XDOT/GAMMA/EPS*DOT) ZEROED — _ell1_delay
        #               ignored them.
        #   • DD/BT   : frozen par secondaries (GAMMA/PBDOT/OMDOT/XDOT/EDOT) passed
        #               through, matching dd_binary_delay_vectorized.
        fd_coeffs = (jnp.asarray([_g(theta, f'FD{order}') for order in fd_orders_full],
                                 dtype=jnp.float64)
                     if fd_orders_full else _FB_EMPTY)
        has_fd    = len(fd_orders_full) > 0
        # DM Taylor [DM, DM1, DM2] (matched by _DM_FACTORIALS=[1,1,2]).  DM1/DM2 are
        # traced when sampled, else frozen (cancel in the θ₀-referenced delta).
        dm_coeffs = jnp.asarray([DM, DM1, DM2], dtype=jnp.float64)
        tropo_prebin = tropo_delay + sw_delay      # frozen; prebinary-only slot

        if has_binary:
            epoch = _g(theta, BINARY_EPOCH_NAME)
            tt_binary_sec = (tdb_mjd - epoch) * SECS_PER_DAY
        else:
            tt_binary_sec = jnp.zeros_like(tdb_mjd)

        if binary_kind == 'ell1':
            r_shap = T_SUN_SEC * M2
            combined = combined_delays(
                tdb_mjd, freq_mhz, obs_sun_km, L_hat,
                dm_coeffs, _DM_FACTORIALS, DMEPOCH_0,
                0.0, fd_coeffs, has_fd,
                roemer + shapiro, has_binary, binary_model_id,
                PB, A1, TASC, EPS1, EPS2, 0.0, 0.0, 0.0, 0.0, 0.0, r_shap, SINI,
                0.0, 0.0, 0.0, 0.0, 0.0, M2, SINI, 0.0, 0.0, 0.0, 0.0, 0.0,
                _FB_EMPTY, _FB_EMPTY, 0.0, False,
                tropo_sec=tropo_prebin, tt_binary_sec=tt_binary_sec,
            )
        else:   # dd / bt — frozen par secondaries forwarded (matches hand-rolled)
            combined = combined_delays(
                tdb_mjd, freq_mhz, obs_sun_km, L_hat,
                dm_coeffs, _DM_FACTORIALS, DMEPOCH_0,
                0.0, fd_coeffs, has_fd,
                roemer + shapiro, has_binary, binary_model_id,
                PB, A1, 0.0, EPS1, EPS2, 0.0, 0.0, PBDOT_0, XDOT, GAMMA_0, 0.0, 0.0,
                ECC, OM, T0, OMDOT, EDOT_0, M2, SINI, 0.0, 0.0, 0.0, 0.0, 0.0,
                _FB_EMPTY, _FB_EMPTY, 0.0, False,
                tropo_sec=tropo_prebin, tt_binary_sec=tt_binary_sec,
            )

        # --- Spindown Taylor series ---
        spin = jnp.zeros_like(dt_sec)
        for order in fn_orders_full:
            fname   = f'F{order}'
            Fn      = _g(theta, fname)
            Fn_0    = theta_0_full[fname]
            coeff   = dt_sec**(order + 1) / math.factorial(order + 1)
            spin    = spin - (Fn - Fn_0) * coeff / F0_0

        # --- DMX chromatic offsets ---
        dmx_delay = jnp.zeros_like(tdb_mjd)
        for label, mask in dmx_masks.items():
            if label in sample_set:
                ΔDMX      = _g(theta, label) - theta_0_full[label]
                dmx_delay = dmx_delay + jnp.where(mask, ΔDMX * dm_deriv, 0.0)

        # --- JUMP phase offsets ---
        jump_delay = jnp.zeros_like(tdb_mjd)
        for jkey, mask in jump_masks.items():
            if jkey in sample_set:
                ΔJUMP      = _g(theta, jkey) - theta_0_full[jkey]
                jump_delay = jump_delay + jnp.where(mask, ΔJUMP, 0.0)

        # ----------------------------------------------------------------
        # Extension point: add new delay kernels here.
        # ----------------------------------------------------------------

        return roemer + shapiro + combined + spin + dmx_delay + jump_delay

    # ---- 8. delta_m_us at θ₀ ----------------------------------------------
    theta_0_sampled = {k_name: theta_0_full[k_name] for k_name in sample_list}
    m0_sec = m_total_sec(theta_0_sampled)

    @jax.jit
    def delta_m_us(theta: dict) -> jnp.ndarray:
        """Mean-subtracted Δm in µs.  Autodiff-able."""
        dm_us = (m_total_sec(theta) - m0_sec) * 1.0e6
        wmean = jnp.sum(weights * dm_us) / jnp.sum(weights)
        return dm_us - wmean

    # ---- 9. Linearised MLE (Jacobian via JVP at θ₀) -----------------------
    X_cols  = []
    col_norms = []
    for k_name in sample_list:
        v    = {kk: jnp.asarray(0.0, dtype=jnp.float64) for kk in sample_list}
        v[k_name] = jnp.asarray(1.0, dtype=jnp.float64)
        col  = np.asarray(jax.jvp(delta_m_us, (theta_0_sampled,), (v,))[1])
        X_cols.append(col)
        col_norms.append(float(np.sqrt(np.dot(col, col))))

    # Guard: any parameter whose Jacobian column is numerically zero (e.g. an
    # unimplemented param that still reached sample_list) would make XtWX
    # singular.  Identify these, warn, and exclude them from the solve.
    _ZERO_THRESH = 1e-30
    active_mask  = np.array(col_norms) > _ZERO_THRESH
    zero_params  = [k for k, active in zip(sample_list, active_mask) if not active]
    if zero_params:
        warnings.warn(
            f"MLE: parameters {zero_params} have zero Jacobian columns and "
            "will be excluded from the linear solve (their MLE delta/sigma "
            "will be NaN).  This usually means a kernel branch is missing.",
            stacklevel=2,
        )

    active_idx  = np.where(active_mask)[0]
    X           = np.column_stack([X_cols[i] for i in active_idx]) if active_idx.size else np.empty((n_toa, 0))
    W           = np.asarray(weights)
    r_obs_np    = np.asarray(r_obs_us)

    if active_idx.size > 0:
        XtWX      = X.T @ (W[:, None] * X)
        XtWr      = X.T @ (W * r_obs_np)
        # Use SVD-based pseudoinverse for robustness against near-singular cases
        # (e.g. highly covariant DMX bins or JUMP/astrometry degeneracies).
        U, s, Vt = np.linalg.svd(XtWX)
        rcond     = np.finfo(float).eps * max(XtWX.shape) * s[0]
        s_inv     = np.where(s > rcond, 1.0 / s, 0.0)
        Sigma_active = (Vt.T * s_inv) @ U.T
        theta_hat_active = Sigma_active @ XtWr
        sigmas_active    = np.sqrt(np.maximum(np.diag(Sigma_active), 0.0))
    else:
        Sigma_active = np.empty((0, 0))
        theta_hat_active = np.empty(0)
        sigmas_active    = np.empty(0)

    # Reconstruct full-length arrays (NaN for zero/inactive params)
    theta_hat_full = np.full(len(sample_list), np.nan)
    sigmas_full    = np.full(len(sample_list), np.nan)
    cov_full       = np.full((len(sample_list), len(sample_list)), np.nan)
    for out_i, in_i in enumerate(active_idx):
        theta_hat_full[in_i] = theta_hat_active[out_i]
        sigmas_full[in_i]    = sigmas_active[out_i]
    for out_i, in_i in enumerate(active_idx):
        for out_j, in_j in enumerate(active_idx):
            cov_full[in_i, in_j] = Sigma_active[out_i, out_j]

    mle = {
        'delta':  dict(zip(sample_list, theta_hat_full)),
        'sigma':  dict(zip(sample_list, sigmas_full)),
        'array':  theta_hat_full,
        'sigmas': sigmas_full,
        'cov':    cov_full,
        'zero_params': zero_params,
    }

    aux = {
        'theta_0':      theta_0_sampled,
        'errors_us':    errors_us,
        'r_obs_us':     r_obs_us,
        'tdb_mjd':      tdb_mjd,
        'mle':          mle,
        'sample_list':  list(sample_list),
        'marginalise_list': sorted(marg_set),       # params carried in M
        'fixed_list':   sorted(fixed_set),          # params held at par value
        'modes':        mode_map,                   # {param: 'sample'|'marginalise'|'fix'}
        'binary_model': binary_model or 'NONE',
        'n_toa':        n_toa,
        'dmx_masks':    dmx_masks,   # {label: bool array}  — for diagnostics
        'jump_masks':   jump_masks,  # {JUMP{n}: bool array} — for diagnostics
        'Mmat': _timing_model_svd(Mmat),           # SVD basis of sample-list-stripped M
        'Mmat_param_names': Mmat_param_names,       # retained linear labels (pre-SVD, auditable)
    }
    return delta_m_us, aux


class MultiPsrTimingModel:
    """Multi-pulsar timing model residual calculator.

    Wraps a list of per-pulsar ``delta_m_us`` callables from
    ``setup_timing_model`` and maps a concatenated
    parameter vector → concatenated timing residuals.

    The parameter layout is:
        z_concat = [z_0_F0, z_0_F1, ...,   # pulsar 0 params in SAMPLE order
                    z_1_F0, z_1_F1, ...,   # pulsar 1 params in SAMPLE order
                    ...]
    i.e. each pulsar contributes ``len(SAMPLE)`` entries in SAMPLE order.
    This is consistent with the numpyro usage where z_{k};{pidx} are sampled
    independently per pulsar.

    Attributes
    ----------
    delta_m_list : list of callable
        Per-pulsar JIT'd delta_m_us(theta_dict) -> jnp.ndarray [n_toa_p]
    aux_list : list of dict
        Per-pulsar aux dicts from setup_timing_model.
    sample_list : list of str
        Parameter names (same for every pulsar).
    scales : list of dict
        Per-pulsar {param: sigma} dicts used to un-normalise z → theta.
    nparams : int
        Number of sampled parameters per pulsar (= len(sample_list)).
    npulsars : int
        Number of pulsars.
    ntoas_per_psr : tuple of int
        Number of TOAs per pulsar.
    toa_starts : tuple of int
        Start index in concatenated residual vector for each pulsar.
    toa_ends : tuple of int
        End index (exclusive) for each pulsar.
    """

    def __init__(self, 
                delta_m_list, 
                aux_list, 
                sample_list, 
                scales,
                data):
        """
        Parameters
        ----------
        delta_m_list : list of callable
            Output of setup_timing_model(...)[0] for each pulsar.
        aux_list : list of dict
            Output of setup_timing_model(...)[1] for each pulsar.
        sample_list : list of str
            Shared parameter names to sample (same order for every pulsar).
        scales : list of dict
            Per-pulsar {param: scale} dicts.  The z-vector is un-normalised as
                theta[k] = theta_0[k] + z[k] * scales[p][k]
            matching the numpyro convention in the example.
        """
        self.raw_residuals = data.raw_residuals

        self.delta_m_list  = delta_m_list
        self.aux_list      = aux_list
        self.sample_list   = list(sample_list)
        self.scales        = scales
        self.nparams       = len(sample_list)
        self.npulsars      = len(delta_m_list)

        ntoas = tuple(int(aux['n_toa']) for aux in aux_list)
        self.ntoas_per_psr = ntoas
        self.total_ntoas = sum(self.ntoas_per_psr)
        cumulative = np.cumsum([0] + list(ntoas))
        self.toa_starts = tuple(int(c) for c in cumulative[:-1])
        self.toa_ends   = tuple(int(c) for c in cumulative[1:])
        
        self.Mmats = [x['Mmat'] for x in aux_list]
        self.neps_per_psr = tuple(x.shape[1] for x in self.Mmats)
        self.total_neps = sum(self.neps_per_psr)
        # Precompute static slice boundaries for each pulsar in the global array.
        cumulative_eps = np.cumsum([0] + list(self.neps_per_psr))
        self.eps_starts = tuple(int(c) for c in cumulative_eps[:-1])
        self.eps_ends   = tuple(int(c) for c in cumulative_eps[1:])

        
    # ------------------------------------------------------------------
    # Parameter layout helpers
    # ------------------------------------------------------------------

    @jit_method
    def linear_residuals(self, epsilons):
        """Get a realization of all pulsar fourier coefficients. [npsr, nmodes]

        This helper method generates a random realization of the pulsar fourier 
        coefficients given the helpers (TNT and TNr), the parameters (halflog10_rhos),
        and a random key. The realization is drawn from a multivariate normal
        distribution such that:
        - covariance -> Sigma = TNT + phi^{-1} 
        - mean = Sigma^{-1} TNr

        Parameters
        ----------
        helpers : tuple
            The helpers (TNT and TNr). [npsr, nmodes, nmodes], [npsr, nmodes]
        params : array
            The input parameters for the signal [npsr*nfreqs]
        key : jax.random.PRNGKey
            The random key for generating the realization.

        Returns
        -------
        array
            A realization of the fourier coefficients for each pulsar. [npsr, nmodes]
        """
        timing_residuals = jnp.zeros(self.total_ntoas)
        for Mmat, start_eps, end_eps, start_toa, end_toa in zip(self.Mmats, 
                                                                self.eps_starts, self.eps_ends, 
                                                                self.toa_starts, self.toa_ends):
            timing_residuals = timing_residuals.at[start_toa:end_toa].set(Mmat @ epsilons[start_eps:end_eps])
        return timing_residuals

    @jit_method
    def get_epsilon(self, helpers, key):
        """Get a realization of all pulsar fourier coefficients. [npsr, nmodes]

        This helper method generates a random realization of the pulsar fourier 
        coefficients given the helpers (TNT and TNr), the parameters (halflog10_rhos),
        and a random key. The realization is drawn from a multivariate normal
        distribution such that:
        - covariance -> Sigma = TNT + phi^{-1} 
        - mean = Sigma^{-1} TNr

        Parameters
        ----------
        helpers : tuple
            The helpers (TNT and TNr). [npsr, nmodes, nmodes], [npsr, nmodes]
        params : array
            The input parameters for the signal [npsr*nfreqs]
        key : jax.random.PRNGKey
            The random key for generating the realization.

        Returns
        -------
        array
            A realization of the fourier coefficients for each pulsar. [npsr, nmodes]
        """
        MNMs, MNrs = helpers # Unpack helpers [npsr, nmode, nmode], [npsr, nmode]
        epsilons = jnp.zeros(self.total_neps)
        for MNM, MNr, start, end in zip(MNMs, MNrs, self.eps_starts, self.eps_ends):

            cf = jsl.cho_factor(MNM) # Cholesky for each psr [npsr, nmodes, nmodes]
            # Get the mean (covariance is Sigma)
            mean = jsl.cho_solve(cf, MNr[..., None])[..., 0] # [npsr, nmodes, 1] 
            # Transform a unit mean Gaussian random variable to desired distribution
            U = jrandom.normal(key, shape=(mean.shape[0],)) # [npsr, nmodes, ndraws]
            # Project U into the desired distribution
            epsil = mean + jsl.solve_triangular(cf[0], U)
            epsilons = epsilons.at[start:end].set(epsil)
        return epsilons

    @jit_method
    def get_mean(self, helpers):
        """Get a realization of all pulsar fourier coefficients. [npsr, nmodes]

        This helper method generates a random realization of the pulsar fourier 
        coefficients given the helpers (TNT and TNr), the parameters (halflog10_rhos),
        and a random key. The realization is drawn from a multivariate normal
        distribution such that:
        - covariance -> Sigma = TNT + phi^{-1} 
        - mean = Sigma^{-1} TNr

        Parameters
        ----------
        helpers : tuple
            The helpers (TNT and TNr). [npsr, nmodes, nmodes], [npsr, nmodes]
        params : array
            The input parameters for the signal [npsr*nfreqs]
        key : jax.random.PRNGKey
            The random key for generating the realization.

        Returns
        -------
        array
            A realization of the fourier coefficients for each pulsar. [npsr, nmodes]
        """
        MNMs, MNrs = helpers # Unpack helpers [npsr, nmode, nmode], [npsr, nmode]
        mean = []
        epsilons = jnp.zeros(self.total_neps)
        for MNM, MNr, start, end in zip(MNMs, MNrs, self.eps_starts, self.eps_ends):

            cf = jsl.cho_factor(MNM) # Cholesky for each psr [npsr, nmodes, nmodes]
            # Get the mean (covariance is Sigma)
            mean.append(jsl.cho_solve(cf, MNr[..., None])[..., 0]) # [npsr, nmodes, 1] 
        return mean
        
    @jit_method
    def get_epsilon_from_z(self, helpers, z):
        """Get a realization of all pulsar fourier coefficients. [npsr, nmodes]

        This helper method generates a random realization of the pulsar fourier 
        coefficients given the helpers (TNT and TNr), the parameters (halflog10_rhos),
        and a random key. The realization is drawn from a multivariate normal
        distribution such that:
        - covariance -> Sigma = TNT + phi^{-1} 
        - mean = Sigma^{-1} TNr

        Parameters
        ----------
        helpers : tuple
            The helpers (TNT and TNr). [npsr, nmodes, nmodes], [npsr, nmodes]
        params : array
            The input parameters for the signal [npsr*nfreqs]
        key : jax.random.PRNGKey
            The random key for generating the realization.

        Returns
        -------
        array
            A realization of the fourier coefficients for each pulsar. [npsr, nmodes]
        """
        MNMs, MNrs = helpers # Unpack helpers [npsr, nmode, nmode], [npsr, nmode]
        epsilons = jnp.zeros(self.total_neps)
        for MNM, MNr, start, end in zip(MNMs, MNrs, self.eps_starts, self.eps_ends):

            cf = jsl.cho_factor(MNM) # Cholesky for each psr [npsr, nmodes, nmodes]
            # Get the mean (covariance is Sigma)
            mean = jsl.cho_solve(cf, MNr[..., None])[..., 0] # [npsr, nmodes, 1] 
            # Transform a unit mean Gaussian random variable to desired distribution
            U = z[start:end]
            # Project U into the desired distribution
            epsil = mean + jsl.solve_triangular(cf[0], U)
            epsilons = epsilons.at[start:end].set(epsil)
        return epsilons

    def z_to_theta(self, pidx, z_p):
        """Un-normalise a per-pulsar z-vector → theta dict.

        Parameters
        ----------
        pidx : int
            Pulsar index.
        z_p : array [nparams]
            Normalised parameter vector for this pulsar.

        Returns
        -------
        theta : dict  {param_name: scalar}
        """
        theta_0 = self.aux_list[pidx]['theta_0']
        sc      = self.scales[pidx]
        # u = z·σ is the scaled, mean-zero sampler coordinate; reparametrise()
        # maps it to a physical value that stays in-domain for every real z,
        # so SINI≤1 / 0≤ECC<1 / M2,PX>0 are guaranteed before the JUG kernels.
        return {k: reparametrise(k, theta_0[k], z_p[i] * sc[k])
                for i, k in enumerate(self.sample_list)}

    # ------------------------------------------------------------------
    # Main callable
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def residuals(self, z_concat):
        """Compute concatenated timing residuals for all pulsars.

        Parameters
        ----------
        z_concat : array [npulsars * nparams]
            Flat, normalised parameter vector.  Layout:
                [z_0[0], z_0[1], ..., z_0[nparams-1],
                 z_1[0], z_1[1], ..., z_1[nparams-1], ...]

        Returns
        -------
        res : array [total_ntoas]
            Concatenated timing residuals in seconds:
                stochastic_res_p = r_obs_p - delta_m_p(theta_p)   (seconds)
            ordered by pulsar, matching toa_starts/toa_ends.
        """
        parts = []
        for pidx in range(self.npulsars):
            # Slice this pulsar's normalised params — static indices, no dynamic_slice needed
            z_p     = z_concat[pidx * self.nparams : (pidx + 1) * self.nparams]
            theta_p = self.z_to_theta(pidx, z_p)
            tm_res  = self.delta_m_list[pidx](theta_p) * 1e-6          # µs → s
            r_obs_p = self.raw_residuals[pidx]
            parts.append(r_obs_p - tm_res)
        return jnp.concatenate(parts)                                    # [total_ntoas]

    # ------------------------------------------------------------------
    # Production numpyro timing block — bounded proper priors (the single mechanism)
    # ------------------------------------------------------------------
    def sample_residuals(self):
        """numpyro block: sample every pulsar's timing params from the canonical
        bounded proper priors and return the concatenated stochastic residuals
        (seconds).  This is the production timing Gibbs group; it imposes the
        SAME bounded priors as the validation harnesses via ``sample_timing_theta``
        and uses NO Jacobian factor.  Call inside a numpyro model.

        Returns
        -------
        res : array [total_ntoas]   r_obs_p - delta_m_p(theta_p), per pulsar.
        """
        parts = []
        for pidx in range(self.npulsars):
            theta_p = sample_timing_theta(
                self.sample_list,
                self.aux_list[pidx]['theta_0'],
                self.aux_list[pidx]['mle']['sigma'],
                prefix=f'p{pidx}_',
            )
            tm_res = self.delta_m_list[pidx](theta_p) * 1e-6              # µs → s
            parts.append(self.raw_residuals[pidx] - tm_res)
        return jnp.concatenate(parts)

    def init_values(self, prefix_each=True):
        """init_to_value dict matching ``sample_residuals`` priors (all pulsars)."""
        iv = {}
        for pidx in range(self.npulsars):
            iv.update(timing_init_values(self.sample_list,
                                         self.aux_list[pidx]['theta_0'],
                                         prefix=f'p{pidx}_'))
        return iv

    # ------------------------------------------------------------------
    # Convenience: numpyro-compatible sampled-z → residuals (legacy z path)
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def residuals_from_z_dict(self, z_dict):
        """Same as residuals() but accepts a numpyro-style z_dict.

        Parameters
        ----------
        z_dict : dict  {f'z_{k};{pidx}': scalar}
            Keyed exactly as the numpyro example produces.

        Returns
        -------
        res : array [total_ntoas]
        """
        parts = []
        for pidx in range(self.npulsars):
            theta_0 = self.aux_list[pidx]['theta_0']
            sc      = self.scales[pidx]
            theta_p = {k: reparametrise(k, theta_0[k], z_dict[f'z_{k};{pidx}'] * sc[k])
                       for k in self.sample_list}
            tm_res  = self.delta_m_list[pidx](theta_p) * 1e-6
            # Same observed-residual source as residuals()/sample_residuals():
            # data.raw_residuals, NOT aux['r_obs_us'] (JUG's own fit residuals),
            # so every path subtracts the model from the identical data vector.
            r_obs_p = self.raw_residuals[pidx]
            parts.append(r_obs_p - tm_res)
        return jnp.concatenate(parts)

# ---------------------------------------------------------------------------
# Constructor helper — mirrors the unoptimised setup loop exactly
# ---------------------------------------------------------------------------

def build_multi_psr_timing_model(parfiles, 
                                 timfiles, 
                                 sample_list,
                                 data, 
                                 load_how_many_in_parallel = 1):
    """Set up a MultiPsrTimingModel from par/tim file lists.

    This is a drop-in replacement for the unoptimised setup loop:

        delta_m_us, aux = [], []
        for parfile, timfile in zip(parfiles, timfiles):
            dm, a = setup_timing_model(parfile, timfile, sample_list)
            delta_m_us.append(dm); aux.append(a)

    Parameters
    ----------
    parfiles : list of str
    timfiles : list of str
    sample_list : list of str
        e.g. ['F0', 'F1', 'DM']

    Returns
    -------
    model : MultiPsrTimingModel
    """
    njobs = int(load_how_many_in_parallel)

    if njobs > 1:
        from tqdm_joblib import ParallelPbar
        from joblib import delayed
        ans = np.array(
            ParallelPbar("Loading the par and tim files...")(n_jobs=njobs)(
            delayed(setup_timing_model)(par, tim, sample_list) for par, tim in zip(parfiles, timfiles)
        ), dtype = object
        )
        delta_m_list = ans[:, 0].tolist()
        aux_list = ans[:, 1].tolist()

    else:
        delta_m_list, aux_list = [], []
        pidx = 0
        for par, tim in zip(parfiles, timfiles):
            dm, a = setup_timing_model(par, tim, sample_list)
            delta_m_list.append(dm)
            aux_list.append(a)
            pidx+=1
    # Per-pulsar scales: linearised-MLE σ for affine params (physical step per
    # unit z).  Reparametrised params (SINI/ECC/M2/PX) are sampled in an
    # UNCONSTRAINED coordinate u = z·scale that reparametrise() maps to the
    # physical value; their physical σ (often ~1e-3) would make u·scale tiny and
    # freeze the param at θ₀, so use a unit scale → O(1) exploration of the full
    # physical domain.
    scales = [{k: (1.0 if k in _REPARAM_PARAMS else float(aux['mle']['sigma'][k]))
               for k in sample_list}
              for aux in aux_list]

    return MultiPsrTimingModel(delta_m_list, 
                                aux_list,
                                sample_list, 
                                scales,
                                data = data)