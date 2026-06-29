from __future__ import annotations
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial

import numpy as np
import jax
import jax.scipy.linalg as jsl
import jax.random as jrandom
from Atlas.utils import jit, jit_method

jax.config.update('jax_enable_x64', True)

from jug.engine.session import TimingSession
from jug.utils.constants import K_DM_SEC, SECS_PER_DAY, C_KM_S, KPC_TO_KM, T_SUN_SEC
from jug.delays.binary_dd import dd_binary_delay_vectorized
from jug.delays.binary_bt import bt_binary_delay_vectorized

from tqdm_joblib import ParallelPbar
from joblib import Parallel, delayed

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

# mas/yr → rad/day (proper motion convention used by JUG/Tempo2/PINT)
_MAS_YR_TO_RAD_DAY = (np.pi / 180.0 / 3.6e6) / 365.25
_AU_KM = 1.495978707e8

# Currently supported sampled parameters.  ELL1-style: TASC/EPS1/EPS2.
# Keplerian (DD/BT): T0/ECC/OM.  Don't mix them within one sample_list.
_SUPPORTED = frozenset({
    'F0', 'F1', 'DM',
    'RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX',
    'PB', 'A1',
    'TASC', 'EPS1', 'EPS2',          # ELL1
    'T0', 'ECC', 'OM',                # DD / BT / DDK
    'M2', 'SINI',
})

# Map BINARY tag → internal kind.  None means "no binary".
_BINARY_KIND = {
    '':        'none',
    'NONE':    'none',
    'ELL1':    'ell1',
    'ELL1H':   'ell1',
    'DD':      'dd',
    'DDH':     'dd',
    'DDGR':    'dd',
    'DDK':     'dd',                  # DDK without Kopeikin corrections
    'BT':      'bt',
    'BTX':     'bt',
}


# ---------------------------------------------------------------------------
# JAX ports of jug.delays.barycentric forward functions (pure float64).
# Identical math to JUG's NumPy versions, just np→jnp.
# ---------------------------------------------------------------------------

@jax.jit
def _pulsar_direction(ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, t_mjd):
    """Pulsar unit vector L̂(t) in equatorial cartesian, with proper motion."""
    dt = t_mjd - posepoch
    cos_dec0 = jnp.cos(dec_rad)
    ra  = ra_rad  + pmra_rad_day  * dt / cos_dec0
    dec = dec_rad + pmdec_rad_day * dt
    cos_dec, sin_dec = jnp.cos(dec), jnp.sin(dec)
    cos_ra,  sin_ra  = jnp.cos(ra),  jnp.sin(ra)
    return jnp.stack([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec], axis=1)


@jax.jit
def _roemer_delay(ssb_obs_pos_km, L_hat, parallax_mas):
    """Roemer delay (s):  -r·L̂/c  +  parallax correction."""
    re_dot_L = jnp.sum(ssb_obs_pos_km * L_hat, axis=1)
    roemer   = -re_dot_L / C_KM_S
    re_sqr   = jnp.sum(ssb_obs_pos_km**2, axis=1)
    px_safe  = jnp.where(parallax_mas != 0.0, parallax_mas, 1.0)
    L_km     = KPC_TO_KM / px_safe
    parallax = jnp.where(
        (parallax_mas != 0.0) & (re_sqr > 0),
        0.5 * (re_sqr / L_km) * (1.0 - re_dot_L**2 / jnp.where(re_sqr > 0, re_sqr, 1.0)) / C_KM_S,
        0.0,
    )
    return roemer + parallax


@jax.jit
def _shapiro_delay(obs_body_pos_km, L_hat, T_body):
    """Shapiro delay (s) for a massive body: -2·T_body·log((r - r·cosθ)/AU)."""
    r         = jnp.sqrt(jnp.sum(obs_body_pos_km**2, axis=1))
    rcostheta = jnp.sum(obs_body_pos_km * L_hat, axis=1)
    return -2.0 * T_body * jnp.log((r - rcostheta) / _AU_KM)


# ---------------------------------------------------------------------------
# ELL1 binary kernel — port of jug.delays.combined.branch_ell1.
# Higher-order EPS1/EPS2 expansion (NOT a T2-Keplerian conversion — that
# drops cubic terms and is wrong for ELL1).
# ---------------------------------------------------------------------------

@jax.jit
def _ell1_delay(t_mjd, pb, a1, tasc, eps1, eps2, m2, sini):
    """ELL1 binary delay (s).  Frozen: pbdot=xdot=eps1dot=eps2dot=gamma=H3=H4=STIG=0."""
    dt_days    = t_mjd - tasc
    dt_sec_bin = dt_days * SECS_PER_DAY
    n0         = 2.0 * jnp.pi / (pb * SECS_PER_DAY)
    Phi        = n0 * dt_sec_bin

    sin_Phi, cos_Phi = jnp.sin(Phi), jnp.cos(Phi)
    sin_2Phi, cos_2Phi = jnp.sin(2*Phi), jnp.cos(2*Phi)
    sin_3Phi, cos_3Phi = jnp.sin(3*Phi), jnp.cos(3*Phi)
    sin_4Phi, cos_4Phi = jnp.sin(4*Phi), jnp.cos(4*Phi)

    e1, e2 = eps1, eps2
    e1s, e2s = e1**2, e2**2
    e1c, e2c = e1**3, e2**3

    Dre_a1 = (
        sin_Phi + 0.5*(e2*sin_2Phi - e1*cos_2Phi)
        - (1.0/8.0)*(5*e2s*sin_Phi - 3*e2s*sin_3Phi - 2*e2*e1*cos_Phi
                     + 6*e2*e1*cos_3Phi + 3*e1s*sin_Phi + 3*e1s*sin_3Phi)
        - (1.0/12.0)*(5*e2c*sin_2Phi + 3*e1s*e2*sin_2Phi - 6*e1*e2s*cos_2Phi
                      - 4*e1c*cos_2Phi - 4*e2c*sin_4Phi + 12*e1s*e2*sin_4Phi
                      + 12*e1*e2s*cos_4Phi - 4*e1c*cos_4Phi)
    )
    Drep_a1 = (
        cos_Phi + e1*sin_2Phi + e2*cos_2Phi
        - (1.0/8.0)*(5*e2s*cos_Phi - 9*e2s*cos_3Phi + 2*e1*e2*sin_Phi
                     - 18*e1*e2*sin_3Phi + 3*e1s*cos_Phi + 9*e1s*cos_3Phi)
        - (1.0/12.0)*(10*e2c*cos_2Phi + 6*e1s*e2*cos_2Phi + 12*e1*e2s*sin_2Phi
                      + 8*e1c*sin_2Phi - 16*e2c*cos_4Phi + 48*e1s*e2*cos_4Phi
                      - 48*e1*e2s*sin_4Phi + 16*e1c*sin_4Phi)
    )
    Drepp_a1 = (
        -sin_Phi + 2*e1*cos_2Phi - 2*e2*sin_2Phi
        - (1.0/8.0)*(-5*e2s*sin_Phi + 27*e2s*sin_3Phi + 2*e1*e2*cos_Phi
                     - 54*e1*e2*cos_3Phi - 3*e1s*sin_Phi - 27*e1s*sin_3Phi)
        - (1.0/12.0)*(-20*e2c*sin_2Phi - 12*e1s*e2*sin_2Phi + 24*e1*e2s*cos_2Phi
                      + 16*e1c*cos_2Phi + 64*e2c*sin_4Phi - 192*e1s*e2*sin_4Phi
                      - 192*e1*e2s*cos_4Phi + 64*e1c*cos_4Phi)
    )
    Dre   = a1 * Dre_a1
    Drep  = a1 * Drep_a1
    Drepp = a1 * Drepp_a1
    binary_roemer = Dre * (1.0 - n0*Drep + (n0*Drep)**2 + 0.5*n0**2*Dre*Drepp)
    r_shap = T_SUN_SEC * m2
    s_shap = sini
    shapiro = jnp.where(
        (r_shap > 0.0) & (s_shap > 0.0),
        -2.0 * r_shap * jnp.log(1.0 - s_shap * sin_Phi),
        0.0,
    )
    return binary_roemer + shapiro


def setup_timing_model(par_path, tim_path, sample_list):
    """
    One public function: ``setup_timing_model(par_path, tim_path, sample_list)`` →
    returns ``(delta_m_us, aux)`` where:

    delta_m_us(theta_dict) -> jnp.ndarray
        Pure JAX, JIT-compiled, autodiff-able.  Returns ``m(θ) - m(θ₀)`` in µs,
        mean-subtracted with TOA weights to match JUG's residual convention.
        ``theta_dict`` keys are exactly the names in ``sample_list``; everything
        else is frozen at the par-file value θ₀.

    aux : dict with cached arrays needed by the surrounding sampler:
        'theta_0'    – par-file values for the sampled parameters
        'errors_us'  – per-TOA uncertainties (µs)
        'r_obs_us'   – JUG-anchor residuals at θ₀ (µs)
        'tdb_mjd'    – TOA times in TDB MJD
        'mle'        – linearised MLE Δθ + 1σ (good NUTS warm-start + scales)
        'sample_list', 'binary_model', 'n_toa'

    Supports ELL1 / ELL1H, DD / DDH / DDGR / DDK (Kopeikin off), BT / BTX, and
    no-binary pulsars.  Sample-list keys (only those listed):
    F0, F1, DM, RAJ, DECJ, PMRA, PMDEC, PX, PB, A1,
    TASC, EPS1, EPS2          (ELL1)
    T0, ECC, OM               (DD / BT / DDK)
    M2, SINI                   (Shapiro)

    DDK Kopeikin geometry corrections are not yet wired in (DDK uses the plain DD
    kernel here).  Higher-order spin/DM polynomials and JUMPs are extensions left
    for future iterations.


    Build a JIT'd JAX delta-m function for a given par/tim and sampled-param list.

    Parameters
    ----------
    par_path, tim_path : str
        Paths to par and tim files.
    sample_list : list[str]
        Names of parameters to be sampled.  Each name must be in:
        F0, F1, DM, RAJ, DECJ, PMRA, PMDEC, PX, PB, A1, TASC, EPS1, EPS2, M2, SINI.

    Returns
    -------
    delta_m_us : jit-compiled callable(theta_dict) -> jnp.ndarray
        Per-TOA delta (µs), mean-subtracted.  ``theta_dict`` must contain exactly
        the keys in ``sample_list``.  Frozen parameters use the par-file value θ₀.
    aux : dict

    """
    bad = [k for k in sample_list if k not in _SUPPORTED]
    if bad:
        raise ValueError(
            f"Unsupported sampled params: {bad}.  Supported: {sorted(_SUPPORTED)}"
        )

    # 1. Run JUG once
    session = TimingSession(par_path, tim_path, verbose=False)
    result  = session.compute_residuals(subtract_tzr=False)
    params  = session.params

    result_dummy = session.fit_parameters(max_iter=5)

    # Design matrix — PINT-matching sign/units
    Mmat   = jnp.array(result_dummy['design_matrix'])          # shape (n_toa, n_param)
    Mmat_param_names = result_dummy['design_matrix_labels']   # list of param names


    binary_model = params.get('BINARY', '').upper()
    binary_kind  = _BINARY_KIND.get(binary_model)
    if binary_kind is None:
        raise NotImplementedError(
            f"Binary model {binary_model!r} not yet supported in jug_jax_timing.  "
            f"Supported: {sorted(set(_BINARY_KIND) - {''})} or no binary."
        )
    has_binary = binary_kind != 'none'

    # Validate sample_list against the binary parametrisation
    if binary_kind == 'ell1':
        bad = [k for k in sample_list if k in ('T0', 'ECC', 'OM')]
        if bad:
            raise ValueError(
                f"ELL1 binary uses TASC/EPS1/EPS2 — don't sample {bad}. "
                "Use TASC/EPS1/EPS2 instead."
            )
    elif binary_kind in ('dd', 'bt'):
        bad = [k for k in sample_list if k in ('TASC', 'EPS1', 'EPS2')]
        if bad:
            raise ValueError(
                f"{binary_model} binary uses T0/ECC/OM — don't sample {bad}. "
                "Use T0/ECC/OM instead."
            )

    # 2. Cache arrays as JAX
    n_toa       = result['n_toas']
    tdb_mjd_ld  = np.asarray(result['tdb_mjd'], dtype=np.longdouble)
    PEPOCH      = np.longdouble(params['PEPOCH'])
    dt_sec      = jnp.asarray(np.asarray(
        (tdb_mjd_ld - PEPOCH) * np.longdouble(SECS_PER_DAY), dtype=np.float64))
    tdb_mjd     = jnp.asarray(np.asarray(tdb_mjd_ld, dtype=np.float64))
    freq_mhz    = jnp.asarray(np.asarray(result['freq_bary_mhz'], dtype=np.float64))
    errors_us   = jnp.asarray(np.asarray(result['errors_us'],     dtype=np.float64))
    weights     = 1.0 / errors_us**2
    ssb_obs_km  = jnp.asarray(np.asarray(result['ssb_obs_pos_ls'], dtype=np.float64) * C_KM_S)
    obs_sun_km  = jnp.asarray(np.asarray(result['obs_sun_pos_ls'], dtype=np.float64) * C_KM_S)
    sw_delay    = jnp.asarray(np.asarray(result.get('sw_delay_sec',  np.zeros(n_toa)), dtype=np.float64))
    tropo_delay = jnp.asarray(np.asarray(result.get('tropo_delay_sec',np.zeros(n_toa)), dtype=np.float64))
    r_obs_us    = jnp.asarray(np.asarray(result['residuals_us'], dtype=np.float64))

    # 3. θ₀ — one entry per *supported* parameter (not just sampled ones)
    theta_0_full = {
        'F0':   float(params['F0']),
        'F1':   float(params.get('F1', 0.0)),
        'DM':   float(params['DM']),
        'RAJ':  float(params['_raj_rad']),
        'DECJ': float(params['_decj_rad']),
        'PMRA': float(params.get('PMRA',  0.0)),
        'PMDEC':float(params.get('PMDEC', 0.0)),
        'PX':   float(params.get('PX',    0.0)),
        'PB':   float(params.get('PB',    0.0)),
        'A1':   float(params.get('A1',    0.0)),
        # ELL1 parametrisation
        'TASC': float(params.get('TASC',  0.0)),
        'EPS1': float(params.get('EPS1',  0.0)),
        'EPS2': float(params.get('EPS2',  0.0)),
        # DD / BT parametrisation (and ELL1→Keplerian conversion below)
        'T0':   float(params.get('T0',    0.0)),
        'ECC':  float(params.get('ECC',   0.0)),
        'OM':   float(params.get('OM',    0.0)),
        'M2':   float(params.get('M2',    0.0)),
        'SINI': float(params.get('SINI',  0.0)),
    }

    # If the par is ELL1 but we want DD-kernel access (and vice versa), JUG's
    # _extract_binary_params does the conversion.  We don't currently sample
    # across kinds, so just leave the missing keys at 0.

    POSEPOCH_0 = float(params.get('POSEPOCH', params['PEPOCH']))
    F0_0 = theta_0_full['F0']

    # Frozen "secondary" binary params — taken from par, never sampled.
    GAMMA_0 = float(params.get('GAMMA', 0.0))
    PBDOT_0 = float(params.get('PBDOT', 0.0))
    OMDOT_0 = float(params.get('OMDOT', 0.0))
    XDOT_0  = float(params.get('XDOT', params.get('A1DOT', 0.0)))
    EDOT_0  = float(params.get('EDOT', 0.0))

    # 4. Build the JAX m(θ) closure.  Sampled keys come from ``theta``;
    # frozen ones are constants captured by the closure.
    sample_set = set(sample_list)
    def _g(theta, k):
        # Pure-Python branch decided at trace time → JAX sees either a traced
        # array (sampled) or a Python float constant (frozen).
        return theta[k] if k in sample_set else theta_0_full[k]

    @jax.jit
    def m_total_sec(theta):
        F0   = _g(theta, 'F0');   F1   = _g(theta, 'F1')
        DM   = _g(theta, 'DM')
        RA   = _g(theta, 'RAJ');  DEC  = _g(theta, 'DECJ')
        PMRA = _g(theta, 'PMRA'); PMDEC= _g(theta, 'PMDEC')
        PX   = _g(theta, 'PX')
        PB   = _g(theta, 'PB');   A1   = _g(theta, 'A1')
        TASC = _g(theta, 'TASC')
        EPS1 = _g(theta, 'EPS1'); EPS2 = _g(theta, 'EPS2')
        T0   = _g(theta, 'T0');   ECC  = _g(theta, 'ECC');   OM = _g(theta, 'OM')
        M2   = _g(theta, 'M2');   SINI = _g(theta, 'SINI')

        L_hat   = _pulsar_direction(RA, DEC,
                                    PMRA  * _MAS_YR_TO_RAD_DAY,
                                    PMDEC * _MAS_YR_TO_RAD_DAY,
                                    POSEPOCH_0, tdb_mjd)
        roemer  = _roemer_delay(ssb_obs_km, L_hat, PX)
        shapiro = _shapiro_delay(obs_sun_km, L_hat, T_SUN_SEC)
        dm_d    = K_DM_SEC * DM / freq_mhz**2

        if has_binary:
            pre_delay = roemer + shapiro + tropo_delay + dm_d + sw_delay
            t_prebin  = tdb_mjd - pre_delay / SECS_PER_DAY
            if binary_kind == 'ell1':
                binary = _ell1_delay(t_prebin, PB, A1, TASC, EPS1, EPS2, M2, SINI)
            elif binary_kind == 'dd':
                # JUG's DD kernel — full Damour-Deruelle with all secondary
                # params frozen at par values (sampled binary params override).
                binary = dd_binary_delay_vectorized(
                    t_prebin, PB, A1, ECC, OM, T0,
                    GAMMA_0, PBDOT_0, OMDOT_0, XDOT_0, EDOT_0,
                    SINI, M2, None, None, None,
                )
            elif binary_kind == 'bt':
                binary = bt_binary_delay_vectorized(
                    t_prebin, PB, A1, ECC, OM, T0,
                    GAMMA_0, PBDOT_0, M2, SINI, OMDOT_0, XDOT_0,
                )
            else:
                binary = jnp.zeros_like(tdb_mjd)
        else:
            binary = jnp.zeros_like(tdb_mjd)

        # Spin: -Δφ/F0 = -[(F0-F0_0)·dt + ½·(F1-F1_0)·dt²] / F0_0
        spin = -(F0 - theta_0_full['F0']) * dt_sec / F0_0 \
               - 0.5 * (F1 - theta_0_full['F1']) * dt_sec**2 / F0_0
        return roemer + shapiro + dm_d + binary + spin

    # 5. Cache m(θ₀), define delta_m_us
    theta_0_sampled = {k: theta_0_full[k] for k in sample_list}
    m0_sec = m_total_sec(theta_0_sampled)

    @jax.jit
    def delta_m_us(theta):
        """Mean-subtracted Δm in µs.  Pure JAX, autodiff-able."""
        m_th = m_total_sec(theta)
        dm_us = (m_th - m0_sec) * 1.0e6
        wmean = jnp.sum(weights * dm_us) / jnp.sum(weights)
        return dm_us - wmean

    # 6. Linearised MLE — Jacobian via JVP at θ₀, then weighted least squares.
    # Useful for warm-starting NUTS and providing per-param scales.
    X_cols = []
    for k in sample_list:
        v = {kk: jnp.asarray(0.0, dtype=jnp.float64) for kk in sample_list}
        v[k] = jnp.asarray(1.0, dtype=jnp.float64)
        col = jax.jvp(delta_m_us, (theta_0_sampled,), (v,))[1]
        X_cols.append(np.asarray(col))
    X        = np.column_stack(X_cols)
    W        = np.asarray(weights)
    r_obs_np = np.asarray(r_obs_us)
    XtWX     = X.T @ (W[:, None] * X)
    XtWr     = X.T @ (W * r_obs_np)
    try:
        Sigma    = np.linalg.pinv(XtWX)
        theta_hat= Sigma @ XtWr
        sigmas   = np.sqrt(np.diag(Sigma))
        mle = {
            'delta':  dict(zip(sample_list, theta_hat)),
            'sigma':  dict(zip(sample_list, sigmas)),
            'array':  theta_hat,
            'sigmas': sigmas,
            'cov':    Sigma,
        }
    except np.linalg.LinAlgError:
        print('******MLE Failed******')
        mle = {}

    aux = {
        'theta_0':      theta_0_sampled,
        'errors_us':    errors_us,
        'r_obs_us':     r_obs_us,
        'tdb_mjd':      tdb_mjd,
        'mle':          mle,
        'sample_list':  list(sample_list),
        'binary_model': binary_model or 'NONE',
        'n_toa':        n_toa,
        'Mmat':         Mmat,
        'Mmat_param_names': Mmat_param_names
    }
    return delta_m_us, aux


class MultiPsrTimingModel:
    """Multi-pulsar timing model residual calculator.

    Wraps a list of per-pulsar ``delta_m_us`` callables from
    ``jug_sampling.setup_timing_model`` and maps a concatenated
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
                data = None, 
                full_linear = False):
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
        self.delta_m_list  = delta_m_list
        self.aux_list      = aux_list
        self.sample_list   = list(sample_list)
        self.scales        = scales
        self.nparams       = len(sample_list)
        self.npulsars      = len(delta_m_list)

        ntoas = tuple(int(aux['n_toa']) for aux in aux_list)
        self.ntoas_per_psr = ntoas
        cumulative = np.cumsum([0] + list(ntoas))
        self.toa_starts = tuple(int(c) for c in cumulative[:-1])
        self.toa_ends   = tuple(int(c) for c in cumulative[1:])
        self.Mmats = [x['Mmat'] for x in aux_list]
    # ------------------------------------------------------------------
    # Parameter layout helpers
    # ------------------------------------------------------------------
    
    @jit_method # Get epsilon
    def _get_coefficient_realization(self, helpers, key):
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
        key : jax.random.PRNGKey
            The random key for generating the realization.

        Returns
        -------
        array
            A realization of the fourier coefficients for each pulsar. [npsr, nmodes]
        """
        MNMs, MNrs = helpers # Unpack helpers [npsr, nmode, nmode], [npsr, nmode]
        coeff = []
        for MNM, MNr in zip(MNMs, MNrs):
            cf = jsl.cho_factor(MNM) # Cholesky for each psr [npsr, nmodes, nmodes]

            # Get the mean (covariance is Sigma)
            mean = jsl.cho_solve(cf, MNr[..., None]) # [npsr, nmodes, 1] 

            # Transform a unit mean Gaussian random variable to desired distribution
            U = jrandom.normal(key, shape=(mean.shape[0], 1)) # [npsr, nmodes, ndraws]
            # Project U into the desired distribution
            coeff.append(mean + jsl.solve_triangular(cf[0], U)[..., 0]) # [npsr, nmodes, ndraws]

        return jnp.concat(coeff) # [npsr, nmodes]

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
        return {k: theta_0[k] + z_p[i] * sc[k]
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
            r_obs_p = self.aux_list[pidx]['r_obs_us'] * 1e-6            # µs → s
            parts.append(r_obs_p - tm_res)
        return jnp.concatenate(parts)                                    # [total_ntoas]

    # ------------------------------------------------------------------
    # Convenience: numpyro-compatible sampled-z → residuals
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
            theta_p = {k: theta_0[k] + z_dict[f'z_{k};{pidx}'] * sc[k]
                       for k in self.sample_list}
            tm_res  = self.delta_m_list[pidx](theta_p) * 1e-6
            r_obs_p = self.aux_list[pidx]['r_obs_us'] * 1e-6
            parts.append(r_obs_p - tm_res)
        return jnp.concatenate(parts)

# ---------------------------------------------------------------------------
# Constructor helper — mirrors the unoptimised setup loop exactly
# ---------------------------------------------------------------------------

def build_multi_psr_timing_model(parfiles, 
                                 timfiles, 
                                 sample_list, 
                                 load_how_many_in_parallel = 1,
                                 data = None):
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
    # Per-pulsar scales: sigma of the linearised MLE, matching the example
    scales = [{k: float(aux['mle']['sigma'][k]) for k in sample_list}
              for aux in aux_list]

    return MultiPsrTimingModel(delta_m_list, 
                                aux_list,
                                sample_list, 
                                scales, 
                                full_linear = data.linear_timing, 
                                data = data)