"""Single-pulsar joint timing + noise sampler — the JUG timing model as a block
in ATLAS's block-Gibbs sampler, sampled jointly with ATLAS's noise.

This is the repository entrypoint for the validated JUG-sourced timing-model
sampler.  The timing machinery (JUG-sourced forward model, precision routing,
offset handling, Hessian preconditioning) is kept as validated; the red noise
uses ATLAS's own free-spectrum representation (``get_phi_diag``) with the
Fourier coefficients **analytically marginalised** (``get_sigma_from_phiinv``),
and white noise is ATLAS's :class:`WhiteCov`.

Timing parameter handling — four modes (``JointConfig.timing_mode``)
-------------------------------------------------------------------
* ``"sample_all"``       — sample every fitted timing param nonlinearly; marginalise none.
* ``"fit_flags"``        — sample the par's fit-flagged params; marginalise none.  JUG
                           fits exactly the flagged params, so this is identical to
                           ``sample_all`` (provided as an explicit name).
* ``"marginalise_linear"`` — sample the nonlinear binary params (M2/SINI/ECC), and
                           **marginalise all the linear params analytically** (F0/F1/DM/FD/
                           JUMP/position/…) by carrying their design matrix in the
                           likelihood.  Fast; the standard marginalised-timing PTA mode.
* ``"specified"``        — sample ``JointConfig.sample_list``; marginalise the rest.

Marginalisation reuses the same machinery as the red noise: the marginalised
params enter the likelihood through their design matrix ``M`` (SVD basis from
``setup_timing_model``), stacked into ``T = [M | F_red]`` with a flat prior on the
``M`` columns.  Only the *sampled* params are routed / sampled through the JUG
forward model; the marginalised ones never enter the sampler.

Architecture
------------
One NumPyro model holds every site.  The timing block produces
``stochastic_res = data − timing_model(theta_sampled)`` and hands it to
``WhiteCov.get_red_helpers`` on the basis ``T=[M|F_red]``; the likelihood
marginalises the linear-timing (M) columns and the red
coefficients.  Two Gibbs groups under :class:`MultiHMCGibbs`:
``[ sampled timing params | xs (red amplitudes) + theta_wn (white) ]``.  The
timing block carries a frozen timing-only ``jug_metric``.

Scope
-----
Validated: single pulsar, timing + free-spectrum red + white.  DM noise,
multi-pulsar / GWB are not included.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import jax.scipy.linalg as jsl

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.diagnostics import summary

from jug.engine.session import TimingSession

from ATLAS.pulsar import Pulsar
from ATLAS.data import PTA_Data
from ATLAS.nMatrix.base import WhiteCov
from ATLAS.signals.factorized.base import Red, SuperSignal
from ATLAS import parameterized
import ATLAS.signals.factorized.utils as unc_utils
from ATLAS.signals.signals_utils import get_power_law_psd
from ATLAS.signals.timing.base import setup_timing_model
from ATLAS.signals.timing.routing import route
from ATLAS.signals.timing import preconditioning as P
from ATLAS.samplers.canetoadracing import MultiHMCGibbs

DEFAULT_BOUND = ("M2", "SINI", "ECC")
_BOUND_SPECS = {"M2": ("uniform", 0.0, 3.0), "ECC": ("uniform", 0.0, 1.0)}
TIMING_MODES = ("sample_all", "fit_flags", "marginalise_linear", "specified")


@dataclass
class JointConfig:
    """Configuration for a single-pulsar joint timing + red + white run."""

    par: str
    tim: str
    nf_red: int = 30
    bound: Sequence[str] = DEFAULT_BOUND
    timing_mode: str = "sample_all"                 # see module docstring
    sample_list: Optional[Sequence[str]] = None     # required for timing_mode="specified"
    inject: Optional[dict] = None                   # keys: efac, log10A_red, gamma_red
    inject_seed: int = 20260627
    z_prior_sd: float = 8.0
    fit_max_iter: int = 5

    def __post_init__(self):
        if self.timing_mode not in TIMING_MODES:
            raise ValueError(f"timing_mode must be one of {TIMING_MODES}, got {self.timing_mode!r}")
        if self.timing_mode == "specified" and not self.sample_list:
            raise ValueError('timing_mode="specified" requires sample_list.')


class JointProblem:
    """Forward model + routing + ATLAS noise objects, all on one TOA order.

    The sample/marginalise split (``timing_mode``) is resolved here: ``sampled``
    params go through the JUG forward model and the sampler; ``marginalised``
    params are carried analytically via their design matrix ``Mmat``.
    """

    def __init__(self, cfg: JointConfig):
        self.cfg = cfg
        s = TimingSession(cfg.par, cfg.tim, verbose=False)
        fj = s.fit_parameters(max_iter=cfg.fit_max_iter)
        freq_mhz = np.asarray(s.compute_residuals(subtract_tzr=False)["freq_bary_mhz"], float)
        self.labs = list(fj["design_matrix_labels"])
        self.free_params = [l for l in self.labs if l != "OFFSET"]     # JUG-fitted (flagged) params
        self.sigJUG = {k: float(fj["uncertainties"][k]) for k in self.free_params}
        self._Cjug = np.asarray(fj["covariance"], float)

        # resolve which params are sampled vs marginalised
        self.sampled, self.marginalised = self._resolve_split()

        # JUG timing forward model over the SAMPLED params; marginalised params
        # are carried in Mmat (SVD design-matrix basis).  delta_m: theta -> Delta m [us]
        # ("delta_m" = delta of the timing model m; NOT the dispersion measure.)
        delta_m, aux = setup_timing_model(cfg.par, cfg.tim, sample_list=self.sampled,
                                          marginalise_list=(self.marginalised or None))
        self.delta_m = delta_m
        self.th0 = aux["theta_0"]
        self.errs = np.asarray(aux["errors_us"]) * 1e-6
        self.r_obs = np.asarray(aux["r_obs_us"]) * 1e-6
        tdb = np.asarray(aux["tdb_mjd"])
        tsec = (tdb - tdb.min()) * 86400.0
        # design matrix of the marginalised params (None if nothing marginalised)
        self.Mmat = np.asarray(aux["Mmat"]) if self.marginalised else None
        self.base_sampled = {k: jnp.asarray(float(self.th0[k])) for k in self.sampled}

        # precision routing — only over the SAMPLED params
        self.BOUND = [k for k in cfg.bound if k in self.sampled]
        self.OFFSET = [k for k in self.sampled
                       if k not in self.BOUND and
                       route(k, float(self.th0[k]), self.sigJUG[k], delta_m, self.base_sampled)["bucket"] == "offset"]
        self.AFFINE = [k for k in self.sampled if k not in self.OFFSET + self.BOUND]
        self.cols = P.offset_columns(delta_m, self.base_sampled, self.OFFSET) if self.OFFSET else {}
        self.SINI_0 = float(self.th0["SINI"]) if "SINI" in self.th0 else None
        self.cosi_0 = (float(np.sqrt(max(1 - self.SINI_0 ** 2, 0.0)))
                       if self.SINI_0 is not None else None)

        # ATLAS noise objects on the setup TOA order
        psr = Pulsar(cfg.par, cfg.tim, use_enterprise=False)
        psr.toas = jnp.asarray(tsec)
        psr.residuals = jnp.asarray(self.r_obs)
        psr.toaerrs = jnp.asarray(self.errs)
        psr.freqs = jnp.asarray(freq_mhz)
        self.data = PTA_Data([psr], fixed_res=False, marg_timing=False,
                             diag_white_cov=False, linear_timing=False)
        self.wn = WhiteCov(data=self.data)
        psd_func, helper = unc_utils.spectrum(renorm_const=1, crn_bins=cfg.nf_red)
        self.sig_unc = Red(name='cor', nfreqs=cfg.nf_red, halflog10_rho_range=(-9, -2),
                           data=self.data, use_pulsar_tspan=False)
        sh = {'shared_basis': {'signal_list': [self.sig_unc], 'index_of_signal_used_for_basis': 0},
              'separate': {'signal_list': []}, 'order': 'cor'}
        self.sig = SuperSignal(signal_helper=sh, data=self.data)
        Tspan = self.sig_unc.tspans
        self.o = parameterized.SinglePulsarRedNoise(
            int_bins=cfg.nf_red, f_intrin=self.sig_unc.freqs, df=1 / Tspan, Tspan=Tspan,
            renorm_const=1, irn_psd_func=psd_func, irn_helper_dictionary=helper)
        self.sig.add_parameterization(self.o)
        self.nmodes = self.sig.nmodes
        self.F_red = np.asarray(self.sig_unc.get_basis()[0])
        # combined marginalisation basis T = [Mmat | F_red]
        if self.Mmat is not None:
            self.Tmat = jnp.asarray(np.concatenate([self.Mmat, self.F_red], axis=1))
            self.n_marg = self.Mmat.shape[1]
        else:
            self.Tmat = jnp.asarray(self.F_red)
            self.n_marg = 0
        self.ll_wn, self.ul_wn = self.wn.get_prior_bounds()

        if cfg.inject is not None:
            self.data_s = self._inject(cfg.inject, cfg.inject_seed)
        else:
            self.data_s = jnp.asarray(self.r_obs)

        self._build_timing_metric()

    def _resolve_split(self):
        cfg = self.cfg
        free = self.free_params
        bound = [b for b in cfg.bound if b in free]
        mode = cfg.timing_mode
        if mode == "sample_all":
            return list(free), []
        if mode == "fit_flags":
            # JUG's fitted set (free_params) IS the par's fit-flagged set — it fits
            # exactly the flagged params, with JUG's canonical labels (e.g. A1DOT->XDOT).
            # So this equals sample_all; kept as a distinct, explicit name.
            return list(free), []
        if mode == "marginalise_linear":
            sampled = [k for k in free if k in bound]      # nonlinear binary geometry
            marg = [k for k in free if k not in sampled]
            return sampled, marg
        if mode == "specified":
            want = set(cfg.sample_list)
            unknown = want - set(free)
            if unknown:
                raise ValueError(f"sample_list params not in the fitted set: {sorted(unknown)}")
            sampled = [k for k in free if k in want]
            marg = [k for k in free if k not in want]
            return sampled, marg
        raise ValueError(mode)

    def _inject(self, inj, seed):
        rng = np.random.default_rng(seed)
        Pr_bin = np.asarray(get_power_law_psd(self.sig_unc.freqs, inj["log10A_red"], inj["gamma_red"], modes=1))
        c_red = np.repeat(np.sqrt(Pr_bin), 2) * rng.normal(size=self.nmodes)
        self._red_true_halflog10 = 0.5 * np.log10(Pr_bin)
        red_s = self.F_red @ c_red
        white_s = rng.normal(0, self.errs * inj["efac"])
        return jnp.asarray(self.r_obs + red_s + white_s)

    def _build_timing_metric(self):
        """Frozen timing-only metric (JUG covariance -> unconstrained coords) over
        the SAMPLED params.  Empty when nothing is sampled."""
        order = [k for k in (self.OFFSET + self.AFFINE + ["M2", "SINI", "ECC"]) if k in self.sampled]
        self.metric_order = order
        self.TIMING_SITES = [self._site_of(k) for k in order]
        if not order:
            self.timing_metric = None
            return
        idx = [self.labs.index(k) for k in order]
        Cphys = self._Cjug[np.ix_(idx, idx)]
        specs = {k: _BOUND_SPECS[k] for k in ("M2", "ECC") if k in order}
        if "SINI" in order:
            specs["SINI"] = ("sini", self.cosi_0, self.SINI_0)
        D = P.coordinate_jacobian(order, {k: float(self.th0[k]) for k in order}, self.sigJUG, specs)
        self.timing_metric = P.jug_metric(Cphys, D)

    def _site_of(self, k):
        if k in self.OFFSET: return f"u_{k}"
        if k == "SINI": return "cosi"
        if k in ("M2", "ECC"): return k
        return f"z_{k}"


# ---------------------------------------------------------------------------
def build_model(prob: JointProblem, sample_noise: bool = True,
                fixed_noise: Optional[dict] = None) -> Callable:
    """One NumPyro model: sampled timing params -> residual -> marginalised
    (M-columns + red coeffs) likelihood + white."""
    cfg = prob.cfg
    OFFSET, AFFINE, BOUND = prob.OFFSET, prob.AFFINE, prob.BOUND
    sigJUG, th0 = prob.sigJUG, prob.th0
    cols, base_sampled = prob.cols, prob.base_sampled
    delta_m, data_s, sig, o = prob.delta_m, prob.data_s, prob.sig, prob.o
    Tmat, n_marg, Nmat = prob.Tmat, prob.n_marg, prob.data.Nmat
    ll_wn, ul_wn = prob.ll_wn, prob.ul_wn
    sd = cfg.z_prior_sd
    has_M2, has_SINI, has_ECC = ("M2" in BOUND), ("SINI" in BOUND), ("ECC" in BOUND)
    if not sample_noise and fixed_noise is None:
        raise ValueError("sample_noise=False requires fixed_noise.")
    xs_fixed = None if sample_noise else jnp.asarray(_fixed_red_xs(prob, fixed_noise))

    def timing_residual(theta):
        off = 0.0
        for k in OFFSET:
            off = off + theta[f"d{k}"] * cols[k]
        th = dict(base_sampled)
        for k in AFFINE:
            th[k] = theta[k]
        if has_SINI: th["SINI"] = theta["SINI"]
        if has_M2:   th["M2"] = theta["M2"]
        if has_ECC:  th["ECC"] = theta["ECC"]
        return data_s - (delta_m(th) + off) * 1e-6

    def marg_loglik(TNT, TNr, rNr, logdet_N, xs):
        phi_red = jnp.repeat(o.get_phi_diag(xs)[:, 0], 2)                 # [nmodes]
        # flat prior on the marginalised timing (M) columns -> zero precision there.
        # phiinv length = n_marg + nmodes = TNT column count (T = [M | F_red]).
        phiinv = jnp.concatenate([jnp.zeros(n_marg), 1.0 / phi_red])     # [ncol]
        logdet_phi = jnp.sum(jnp.log(phi_red))                           # M columns: flat, no det term
        # Sigma = TNT + diag(phiinv), built directly (sig.get_sigma_from_phiinv is
        # hardwired to sig's nmodes and cannot take the augmented basis).
        Sigma = TNT[0] + jnp.diag(phiinv)                               # [ncol, ncol]
        cf = jsl.cho_factor(Sigma)
        ld = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(cf[0]))))
        expv = jnp.sum(TNr[0] * jsl.cho_solve(cf, TNr[0]))
        return 0.5 * (expv - ld - logdet_phi) - 0.5 * (rNr + logdet_N)

    def model():
        theta = {}
        for k in OFFSET:
            u = numpyro.sample(f"u_{k}", dist.Normal(0., sd)); theta[f"d{k}"] = u * sigJUG[k]
            numpyro.deterministic(f"d{k}", theta[f"d{k}"])
        for k in AFFINE:
            z = numpyro.sample(f"z_{k}", dist.Normal(0., sd)); theta[k] = float(th0[k]) + z * sigJUG[k]
            numpyro.deterministic(k, theta[k])
        if has_SINI:
            cosi = numpyro.sample("cosi", dist.Uniform(-1., 1.))
            theta["SINI"] = jnp.sqrt(jnp.clip(1 - cosi ** 2, 1e-12, 1.)); numpyro.deterministic("SINI", theta["SINI"])
        if has_M2:
            theta["M2"] = numpyro.sample("M2", dist.Uniform(0., 3.))
        if has_ECC:
            theta["ECC"] = numpyro.sample("ECC", dist.Uniform(0., 1.))
        stochastic_res = timing_residual(theta)

        if sample_noise:
            theta_wn = numpyro.sample("theta_wn", dist.Uniform(ll_wn, ul_wn))
            xs = numpyro.sample("xs", dist.Uniform(o.lower_prior_lim_all, o.upper_prior_lim_all))
        else:
            theta_wn = jnp.asarray(_wn_vector_for_efac(prob, fixed_noise["efac"]))
            xs = xs_fixed
        # helpers built on the FULL basis T=[M|F_red] (marginalised timing + red);
        # WhiteCov.get_red_helpers handles the white-noise solve for our basis.
        TNT, TNr, rNr, logdet_N = Nmat.get_red_helpers(
            red_noise_basis=Tmat, residuals=stochastic_res, white_noise_params=theta_wn)
        numpyro.factor("lnpost", marg_loglik(TNT, TNr, rNr, logdet_N, xs))

    return model


def _fixed_red_xs(prob, noise):
    Pr_bin = np.asarray(get_power_law_psd(prob.sig_unc.freqs, noise["log10A_red"], noise["gamma_red"], modes=1))
    return 0.5 * np.log10(Pr_bin)


def _wn_vector_for_efac(prob, efac):
    lo, hi = np.asarray(prob.ll_wn), np.asarray(prob.ul_wn)
    v = 0.5 * (lo + hi)
    names = prob.wn.get_param_names() if hasattr(prob.wn, "get_param_names") else []
    for i, nm in enumerate(names):
        if "efac" in nm:
            v[i] = efac
    return v


# ---------------------------------------------------------------------------
def run_gibbs(prob: JointProblem, model, sample_noise: bool = True, key: int = 1,
              num_warmup: int = 700, num_samples: int = 1500, num_chains: int = 2,
              target_accept_prob: float = 0.9, progress_bar: bool = False):
    """2-block MultiHMCGibbs: sampled-timing (frozen jug_metric) | [xs, white] adaptive."""
    th0 = prob.th0
    iv = {f"u_{k}": 0.0 for k in prob.OFFSET}
    iv.update({f"z_{k}": 0.0 for k in prob.AFFINE})
    if "SINI" in prob.BOUND: iv["cosi"] = prob.cosi_0
    if "M2" in prob.BOUND: iv["M2"] = float(th0["M2"])
    if "ECC" in prob.BOUND: iv["ECC"] = float(th0["ECC"])
    if sample_noise:
        iv.update({"theta_wn": 0.5 * (np.asarray(prob.ll_wn) + np.asarray(prob.ul_wn)),
                   "xs": prob.o.make_initial_guess(jax.random.PRNGKey(1))})

    kernels, groups = [], []
    if prob.TIMING_SITES:
        tsites = tuple(prob.TIMING_SITES)
        kernels.append(NUTS(model, target_accept_prob=target_accept_prob, max_tree_depth=9,
                            dense_mass=[tsites], inverse_mass_matrix={tsites: jnp.asarray(prob.timing_metric)},
                            adapt_mass_matrix=False, init_strategy=init_to_value(values=iv)))
        groups.append(list(tsites))
    if sample_noise:
        kernels.append(NUTS(model, target_accept_prob=target_accept_prob, max_tree_depth=8,
                            dense_mass=False, init_strategy=init_to_value(values=iv)))
        groups.append(["theta_wn", "xs"])

    mc = MCMC(MultiHMCGibbs(kernels, groups), num_warmup=num_warmup, num_samples=num_samples,
              num_chains=num_chains, chain_method="sequential", progress_bar=progress_bar)
    mc.run(jax.random.PRNGKey(key))
    return mc


# ---------------------------------------------------------------------------
def extract_results(prob: JointProblem, mc, sample_noise: bool, truth_json: Optional[str] = None):
    """Per-parameter posteriors for the SAMPLED params + named white-noise params.
    Marginalised timing params are not sampled and are absent (they are integrated
    out analytically).  ``cosi`` and ``SINI`` are the same DOF — plot only one."""
    OFFSET, AFFINE, th0 = prob.OFFSET, prob.AFFINE, prob.th0
    fl = mc.get_samples(); g = mc.get_samples(group_by_chain=True)
    TK = OFFSET + AFFINE + [k for k in ("M2", "ECC", "cosi", "SINI") if k in prob.BOUND or k == "cosi" and "SINI" in prob.BOUND]
    inj = prob.cfg.inject or {}
    tvv = {}
    if truth_json is not None:
        tvv = {r["param"]: r["true"] for r in json.load(open(truth_json))["rows"]}
    out = {"median": {}, "sigma": {}, "rhat": {}, "ess": {}, "truth": {},
           "samples": {}, "is_offset": {}, "white_names": [], "marginalised": list(prob.marginalised)}

    def _record(key, x, gg, truth, is_offset=False):
        x = np.asarray(x); gg = np.asarray(gg)
        out["samples"][key] = x
        out["median"][key] = float(np.median(x)); out["sigma"][key] = float(np.std(x))
        st = summary({key: gg})[key]
        out["rhat"][key] = float(np.max(st["r_hat"])); out["ess"][key] = float(np.min(st["n_eff"]))
        out["truth"][key] = truth; out["is_offset"][key] = is_offset

    order = []
    for k in TK:
        nm = (f"d{k}" if k in OFFSET else k)
        if nm not in fl:
            continue
        truth = (0.0 if k in OFFSET else
                 (prob.cosi_0 if k == "cosi" else
                  (prob.SINI_0 if k == "SINI" else tvv.get(k, float(th0[k]) if k in prob.free_params else np.nan))))
        _record(k, fl[nm], g[nm], truth, is_offset=(k in OFFSET))
        order.append(k)

    if sample_noise and "theta_wn" in fl:
        names = prob.wn.get_param_names() if hasattr(prob.wn, "get_param_names") else \
                [f"wn_{i}" for i in range(np.asarray(fl["theta_wn"]).shape[-1])]
        x = np.asarray(fl["theta_wn"]); gg = np.asarray(g["theta_wn"])
        for i, wn_name in enumerate(names):
            truth = inj.get("efac", np.nan) if wn_name.endswith("efac") else np.nan
            _record(wn_name, x[..., i], gg[..., i], truth)
            order.append(wn_name); out["white_names"].append(wn_name)

    out["order"] = order
    if "xs" in fl:
        out["red_halflog10_rho"] = np.asarray(fl["xs"])
        if prob.cfg.inject is not None:
            out["red_halflog10_rho_truth"] = prob._red_true_halflog10
    return out


def run_joint_timing_noise(cfg: JointConfig, sample_noise: bool = True, key: int = 1,
                           num_warmup: int = 700, num_samples: int = 1500, num_chains: int = 2,
                           truth_json: Optional[str] = None, verbose: bool = True):
    """Configure -> build model -> 2-block Gibbs -> summarise."""
    prob = JointProblem(cfg)
    if verbose:
        print(f"[{cfg.timing_mode}] sampled ({len(prob.sampled)}): {prob.sampled}")
        print(f"           marginalised ({len(prob.marginalised)}): {prob.marginalised}")
    fixed_noise = cfg.inject if not sample_noise else None
    model = build_model(prob, sample_noise=sample_noise, fixed_noise=fixed_noise)
    mc = run_gibbs(prob, model, sample_noise=sample_noise, key=key, num_warmup=num_warmup,
                   num_samples=num_samples, num_chains=num_chains, progress_bar=verbose)
    res = extract_results(prob, mc, sample_noise=sample_noise, truth_json=truth_json)
    res["problem"] = prob; res["mcmc"] = mc
    return res
