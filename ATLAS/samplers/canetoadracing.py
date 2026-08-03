'''
The credit goes to
@software{coleman_krawczyk_2024_12167630,
  author       = {Coleman Krawczyk},
  title        = {CKrawczyk/MultiHMCGibbs: v1.0.0},
  month        = jun,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.12167630},
  url          = {https://doi.org/10.5281/zenodo.12167630}
}
'''
# Copyright 2024 Coleman Krawczyk
# SPDX-License-Identifier: Apache-2.0

import copy
from collections import namedtuple, Counter
from functools import partial
from itertools import chain

import jax.numpy as jnp
from jax import device_put, jacfwd, random, value_and_grad, vmap
from numpyro.distributions.transforms import biject_to
from numpyro.handlers import condition, seed, substitute, trace
from numpyro.infer.initialization import init_to_sample, init_to_uniform
from numpyro.infer.mcmc import MCMCKernel
from numpyro.util import is_prng_key
import numpyro.distributions as dist

import numpyro
import numpyro.distributions as dist
# ---------------------------------------------------------------------------
# Public state type
# ---------------------------------------------------------------------------

MultiHMCGibbsState = namedtuple(
    "MultiHMCGibbsState",
    "z, hmc_states, diverging, rng_key, potential_energy",
)
"""
 - **z**                - dict of current latent values (all sites, unconstrained)
 - **hmc_states**       - list of HMCState, one per inner kernel
 - **diverging**        - stacked bool array, one entry per inner kernel
 - **rng_key**          - stacked rng keys, one per inner kernel
 - **potential_energy** - potential energy after the last kernel step
"""

# ---------------------------------------------------------------------------
# AnalyticState — used by AnalyticRhoTransition (and subclasses)
# ---------------------------------------------------------------------------

AnalyticState = namedtuple(
    "AnalyticState",
    ["z", "rng_key", "diverging", "potential_energy"],
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wrap_model(model, *args, **kwargs):
    cond_values = kwargs.pop("_cond_sites", {})
    with condition(data=cond_values), substitute(data=cond_values):
        return model(*args, **kwargs)


# ---------------------------------------------------------------------------
# MultiHMCGibbs
# ---------------------------------------------------------------------------

class MultiHMCGibbs(MCMCKernel):
    """Multi-HMC-within-Gibbs.

    Combines multiple gradient-based kernels (HMC / NUTS), each conditioned
    on a different subset of sample sites, as steps in a Gibbs sampler.

    Parameters
    ----------
    inner_kernels : list of HMC / NUTS kernels
        All kernels **must use the same model**.
    gibbs_sites_list : list of lists of str
        Each inner list is the set of *free* sites for that kernel step.
        Every sample site in the model must appear in exactly one list.
    num_steps : list of int, optional
        How many times each kernel runs per Gibbs sweep.  Defaults to 1 for
        every kernel.  A value of ``n > 1`` for kernel ``k`` means that
        kernel runs ``n`` consecutive times, each time conditioning on the
        freshest state of all other sites.

    Example
    -------
    >>> kernel = MultiHMCGibbs(
    ...     [NUTS(model), NUTS(model)],
    ...     [['y'], ['x']],
    ...     num_steps=[1, 3],   # x kernel runs 3 times per sweep
    ... )
    """

    sample_field = "z"

    def __init__(self, inner_kernels, gibbs_sites_list, num_steps=None):
        # ── num_steps ──────────────────────────────────────────────────────
        if num_steps is None:
            self.num_steps = [1] * len(inner_kernels)
        else:
            if len(num_steps) != len(inner_kernels):
                raise ValueError(
                    f"num_steps length ({len(num_steps)}) must match "
                    f"inner_kernels length ({len(inner_kernels)})"
                )
            if any(n < 1 for n in num_steps):
                raise ValueError("All num_steps values must be >= 1")
            self.num_steps = list(num_steps)

        # ── kernels ────────────────────────────────────────────────────────
        self.inner_kernels = []
        self.gibbs_sites_list = gibbs_sites_list
        for kdx, kernel in enumerate(inner_kernels):
            if kernel._model is not inner_kernels[0]._model:
                raise ValueError(
                    f"inner kernel {kdx} does not have the same NumPyro model "
                    f"as kernel 0."
                )
            k = copy.copy(kernel)
            k._model = partial(_wrap_model, k.model)
            # All sites except this kernel's own sites are conditioning sites.
            # frozenset gives O(1) membership test vs O(n) for a list.
            k._cond_sites = frozenset(chain.from_iterable(
                self.gibbs_sites_list[:kdx] + self.gibbs_sites_list[kdx + 1:]
            ))
            self.inner_kernels.append(k)

        self._prototype_trace = None
        self._sample_fn = None

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self.inner_kernels[0]._model

    @property
    def default_fields(self):
        return ("z", "diverging", "potential_energy")

    def get_diagnostics_str(self, state):
        num_steps      = "/".join(str(s.num_steps)               for s in state.hmc_states)
        step_size      = "/".join(f"{s.adapt_state.step_size:.2e}" for s in state.hmc_states)
        mean_accept    = "/".join(f"{s.mean_accept_prob:.2f}"     for s in state.hmc_states)
        return f"{num_steps} steps of size {step_size}. acc. prob={mean_accept}"

    def postprocess_fn(self, args, kwargs):
        _ = kwargs.pop("_cond_sites", {})

        def combined(z):
            constrained = {
                name: self._site_bijectors[name](val)
                for name, val in z.items()
                if name in self._site_bijectors
            }
            model_trace = trace(substitute(self.model, data=constrained)).get_trace(*args, **kwargs)
            out = dict(constrained)
            for name, site in model_trace.items():
                if site["type"] == "deterministic":
                    out[name] = site["value"]
            return out

        return combined

    def check_gibbs_sites(self, model_args, model_kwargs):
        """Verify every sample site appears in exactly one Gibbs group."""
        t = trace(
            substitute(
                seed(self.model, random.PRNGKey(0)),
                substitute_fn=init_to_uniform,
            )
        ).get_trace(*model_args, **model_kwargs)

        all_sites = Counter(
            key for key, value in t.items()
            if value["type"] == "sample" and not value["is_observed"]
        )
        # Deduplicate listed sites — num_steps > 1 does not mean a site is
        # listed twice; it just runs the same kernel multiple times.
        listed_sites = Counter(set(chain.from_iterable(self.gibbs_sites_list)))

        if listed_sites != all_sites:
            msg = "Expected each site to be listed **exactly once**. "
            missing = list((all_sites - listed_sites).keys())
            if missing:
                msg += f"Sites in model but not listed: {missing}. "
            extra = list((listed_sites - all_sites).keys())
            if extra:
                msg += f"Sites listed but not in model: {extra}."
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
        self.check_gibbs_sites(model_args, model_kwargs)

        def init_fn(init_params, key_zs):
            if (init_params is not None) and (len(init_params) == 0):
                init_params = None
            diverging = jnp.zeros(len(self.inner_kernels), dtype=bool)

            # Build prototype trace once
            if self._prototype_trace is None:
                self._prototype_trace = trace(
                    substitute(seed(self.model, key_zs[0]), substitute_fn=init_to_sample)
                ).get_trace(*model_args, **model_kwargs)

            # Pre-bake bijectors: unconstrained → constrained for every sample site.
            # Stored on self so _sample_one can use them without re-running the model.
            self._site_bijectors = {
                name: biject_to(site["fn"].support)
                for name, site in self._prototype_trace.items()
                if site["type"] == "sample" and not site["is_observed"]
            }

            z = {}
            hmc_states = []
            rng_keys = []
            for key_z, kernel in zip(key_zs[1:], self.inner_kernels):
                cond_sites_kdx  = {}
                init_params_kdx = {} if (init_params is not None) else None
                for name, site in self._prototype_trace.items():
                    if init_params is not None:
                        if name in kernel._cond_sites:
                            cond_sites_kdx[name] = init_params[name]
                        elif name in init_params:
                            init_params_kdx[name] = init_params[name]
                    elif name in kernel._cond_sites:
                        cond_sites_kdx[name] = site["value"]

                model_kwargs_kdx = model_kwargs | {"_cond_sites": cond_sites_kdx}
                hmc_state_kdx = kernel.init(
                    key_z,
                    num_warmup,
                    init_params=init_params_kdx,
                    model_args=model_args,
                    model_kwargs=model_kwargs_kdx,
                )
                hmc_states.append(hmc_state_kdx)
                rng_keys.append(hmc_state_kdx.rng_key)
                z = z | hmc_state_kdx.z

            return MultiHMCGibbsState(
                z, hmc_states, diverging, jnp.stack(rng_keys), 0.0
            )

        if is_prng_key(rng_key):
            key_zs = random.split(rng_key, len(self.inner_kernels) + 1)
            init_state = init_fn(init_params, key_zs)
            self._sample_fn = self._sample_one
        else:
            init_params = {} if init_params is None else init_params
            key_zs = vmap(partial(random.split, num=len(self.inner_kernels) + 1))(rng_key)
            init_state = vmap(init_fn)(init_params, key_zs)
            self._sample_fn = vmap(self._sample_one, in_axes=(0, None, None))

        return device_put(init_state)

    # ------------------------------------------------------------------
    # _sample_one
    # ------------------------------------------------------------------

    def _sample_one(self, state, model_args, model_kwargs):
        """One full Gibbs sweep (all kernels, respecting num_steps)."""
        model_kwargs = {} if model_kwargs is None else model_kwargs
        z = state.z

        # Build constrained-space dict via bijectors — zero model evaluations.
        # Only sample sites are present; deterministic sites are absent by design.
        z_constrained = {
            name: self._site_bijectors[name](val)
            for name, val in z.items()
            if name in self._site_bijectors
        }

        hmc_states, diverging, rng_keys = [], [], []

        for hmc_state, kernel, n_steps in zip(
            state.hmc_states, self.inner_kernels, self.num_steps
        ):
            for _ in range(n_steps):
                # Conditioning sites for this kernel (small set → fast dict build)
                z_cond_constrained = {k: z_constrained[k] for k in kernel._cond_sites}

                # Capture z_cond_constrained in default arg to avoid closure bug
                def potential_fn(z_hmc, _cond=z_cond_constrained):
                    return kernel._potential_fn_gen(
                        *model_args, _cond_sites=_cond, **model_kwargs
                    )(z_hmc)

                if kernel._forward_mode_differentiation:
                    pe     = potential_fn(hmc_state.z)
                    z_grad = jacfwd(potential_fn)(hmc_state.z)
                else:
                    pe, z_grad = value_and_grad(potential_fn)(hmc_state.z)

                hmc_state = hmc_state._replace(z_grad=z_grad, potential_energy=pe)
                hmc_state = kernel.sample(
                    hmc_state,
                    model_args,
                    model_kwargs | {"_cond_sites": z_cond_constrained},
                )

                # Incremental constrained-space update — only changed sites
                for name, val in hmc_state.z.items():
                    if name in self._site_bijectors:
                        z_constrained[name] = self._site_bijectors[name](val)

                # Update z after every sub-step so later sub-steps of the same
                # kernel see the freshest values
                z = z | hmc_state.z

            hmc_states.append(hmc_state)
            diverging.append(hmc_state.diverging)
            rng_keys.append(hmc_state.rng_key)

        return MultiHMCGibbsState(
            z,
            hmc_states,
            jnp.stack(diverging),
            jnp.stack(rng_keys),
            hmc_state.potential_energy,
        )

    def sample(self, state, model_args, model_kwargs):
        return self._sample_fn(state, model_args, model_kwargs)


# ---------------------------------------------------------------------------
# AnalyticRhoTransition — analytic Gibbs draw for a named rho site
# ---------------------------------------------------------------------------

class AnalyticRhoTransition(MCMCKernel):
    """Analytic Gibbs draw for a half-log10-rho site.

    Must be used inside ``MultiHMCGibbsWithAnalytic``, not plain
    ``MultiHMCGibbs``, because it bypasses the gradient / potential-energy
    bookkeeping that the vanilla ``_sample_one`` loop performs unconditionally.

    Parameters
    ----------
    model : callable
        NumPyro model (same object passed to the NUTS kernels).
    sig : object
        Signal object with ``prior_draw(rng_key)`` and
        ``posterior_draw_from_coeff(coeff, rng_key)`` methods.
    rho_site : str
        Name of the half-log10-rho sample site.  Default ``"psr_half_log10_rho"``.
    coeff_site : str
        Name of the Fourier-coefficient deterministic site.  Default ``"coeff"``.
    rho_low : float
        Lower bound of the uniform prior on rho.  Default ``-9.0``.
    rho_high : float
        Upper bound of the uniform prior on rho.  Default ``-2.0``.
    """

    sample_field = "z"

    def __init__(
        self,
        model,
        sig,
        rho_site: str  = "psr_half_log10_rho",
        coeff_site: str = "coeff",
        rho_low: float  = -9.0,
        rho_high: float = -2.0,
    ):
        # Both _model and model are required by MultiHMCGibbs internals
        self._model = model
        self.model  = model
        self.sig        = sig
        self.rho_site   = rho_site
        self.coeff_site = coeff_site
        self.rho_low    = rho_low
        self.rho_high   = rho_high
        # Bijector: unconstrained ℝ ↔ constrained (rho_low, rho_high)
        self._rho_transform = biject_to(dist.Uniform(rho_low, rho_high).support)

    @property
    def default_fields(self):
        return ()

    def postprocess_fn(self, args, kwargs):
        return lambda x: x

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        rho_init = self.sig.prior_draw(rng_key)
        if init_params is not None and self.rho_site in init_params:
            rho_init = init_params[self.rho_site]
        return AnalyticState(
            z={self.rho_site: rho_init},
            rng_key=rng_key,
            diverging=jnp.array(False),
            potential_energy=jnp.array(0.0),
        )

    def sample(self, state, model_args, model_kwargs, *, z_constrained):
        """One analytic Gibbs draw.

        Parameters
        ----------
        z_constrained : dict
            Full constrained-space dict from ``postprocess_fn(z)``.
            Must contain ``self.coeff_site`` (a deterministic site — it is
            absent from the plain Gibbs ``z`` dict, which is why it must be
            passed explicitly by ``MultiHMCGibbsWithAnalytic._sample_one``).
        """
        rng_key, subkey = __import__("jax").random.split(state.rng_key)

        current_coeff = z_constrained[self.coeff_site]

        # Draw is in constrained space (rho_low, rho_high)
        rho_constrained = self.sig.posterior_draw_from_coeff(
            current_coeff[..., None], subkey
        )[0]

        # Store the unconstrained counterpart so that
        #   postprocess_fn(z)[self.rho_site] == rho_constrained
        # and the NUTS kernels condition on the correct value.
        rho_unconstrained = self._rho_transform.inv(rho_constrained)

        return AnalyticState(
            z={self.rho_site: rho_unconstrained},
            rng_key=rng_key,
            diverging=jnp.array(False),
            potential_energy=jnp.array(0.0),
        )


# ---------------------------------------------------------------------------
# MultiHMCGibbsWithAnalytic
# ---------------------------------------------------------------------------

class MultiHMCGibbsWithAnalytic(MultiHMCGibbs):
    """``MultiHMCGibbs`` extended to support analytic Gibbs kernels.

    Any kernel that is an instance of ``AnalyticTransition`` is handled
    specially in ``_sample_one``:

    * Gradient / potential-energy bookkeeping is skipped (the analytic kernel
      does not need it and does not expose the required HMC-state fields).
    * ``postprocess_fn(z)`` is called once per sweep, just before the analytic
      kernel runs, so that deterministic sites (e.g. ``coeff``) are available
      to it with the freshest values from the preceding NUTS steps.

    Parameters
    ----------
    inner_kernels : list
        Mix of NUTS / HMC kernels and ``AnalyticTransition`` instances.
    gibbs_sites_list : list of lists of str
        Same contract as ``MultiHMCGibbs``.
    num_steps : list of int, optional
        Same contract as ``MultiHMCGibbs``.
    AnalyticTransition : type
        The class (or base class) used to identify analytic kernels.
        Defaults to ``AnalyticRhoTransition``.

    Example
    -------
    >>> analytic = AnalyticRhoTransition(model=model, sig=sig)
    >>> kernel = MultiHMCGibbsWithAnalytic(
    ...     inner_kernels=[NUTS(model), NUTS(model), NUTS(model), analytic],
    ...     gibbs_sites_list=[
    ...         list(wn_noise_dict.keys()),
    ...         [f"z_{k}" for k in SAMPLE],
    ...         ["z_a"],
    ...         ["psr_half_log10_rho"],
    ...     ],
    ...     num_steps=[1, 1, 2, 1],          # z_a kernel runs twice per sweep
    ...     AnalyticTransition=AnalyticRhoTransition,
    ... )
    """

    def __init__(
        self,
        inner_kernels,
        gibbs_sites_list,
        num_steps=None,
        AnalyticTransition=None,
    ):
        # Analytic kernels do not have ._model set the same way as NUTS kernels;
        # skip the model-identity check for them by temporarily patching.
        self.AnalyticTransition = AnalyticTransition or AnalyticRhoTransition
        super().__init__(inner_kernels, gibbs_sites_list, num_steps=num_steps)

    # Override __init__ loop to skip model-identity check for analytic kernels
    def __init__(
        self,
        inner_kernels,
        gibbs_sites_list,
        num_steps=None,
        AnalyticTransition=None,
    ):
        self.AnalyticTransition = AnalyticTransition or AnalyticRhoTransition

        # ── num_steps ──────────────────────────────────────────────────────
        if num_steps is None:
            self.num_steps = [1] * len(inner_kernels)
        else:
            if len(num_steps) != len(inner_kernels):
                raise ValueError(
                    f"num_steps length ({len(num_steps)}) must match "
                    f"inner_kernels length ({len(inner_kernels)})"
                )
            if any(n < 1 for n in num_steps):
                raise ValueError("All num_steps values must be >= 1")
            self.num_steps = list(num_steps)

        # ── kernels ────────────────────────────────────────────────────────
        self.inner_kernels = []
        self.gibbs_sites_list = gibbs_sites_list

        # Find the reference model from the first non-analytic kernel
        ref_model = next(
            k._model for k in inner_kernels
            if not isinstance(k, self.AnalyticTransition)
        )

        for kdx, kernel in enumerate(inner_kernels):
            if isinstance(kernel, self.AnalyticTransition):
                # Analytic kernels are stored as-is; they manage their own state
                k = kernel
            else:
                if kernel._model is not ref_model:
                    raise ValueError(
                        f"inner kernel {kdx} does not have the same NumPyro "
                        f"model as the reference kernel."
                    )
                k = copy.copy(kernel)
                k._model = partial(_wrap_model, k.model)

            k._cond_sites = frozenset(chain.from_iterable(
                self.gibbs_sites_list[:kdx] + self.gibbs_sites_list[kdx + 1:]
            ))
            self.inner_kernels.append(k)

        self._prototype_trace = None
        self._sample_fn = None

    # ------------------------------------------------------------------
    # init — same as parent but guarantees _site_bijectors is always set
    # ------------------------------------------------------------------

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        # Use the first non-analytic kernel for the prototype trace
        state = MultiHMCGibbs.init(
            self, rng_key, num_warmup, init_params, model_args, model_kwargs
        )
        # _site_bijectors is set inside MultiHMCGibbs.init already; re-assert
        # here to be explicit and future-proof.
        self._site_bijectors = {
            name: biject_to(site["fn"].support)
            for name, site in self._prototype_trace.items()
            if site["type"] == "sample" and not site["is_observed"]
        }
        return state

    # ------------------------------------------------------------------
    # Diagnostics — gracefully handles analytic kernel states
    # ------------------------------------------------------------------

    def get_diagnostics_str(self, state):
        steps, sizes, probs = [], [], []
        for kernel, s in zip(self.inner_kernels, state.hmc_states):
            if isinstance(kernel, self.AnalyticTransition):
                steps.append("analytic")
                sizes.append("N/A")
                probs.append("1.00")
            else:
                steps.append(str(s.num_steps))
                sizes.append(f"{s.adapt_state.step_size:.2e}")
                probs.append(f"{s.mean_accept_prob:.2f}")
        return "{} steps of size {}. acc. prob={}".format(
            "/".join(steps), "/".join(sizes), "/".join(probs)
        )

    # ------------------------------------------------------------------
    # _sample_one
    # ------------------------------------------------------------------

    def _sample_one(self, state, model_args, model_kwargs):
        """One full Gibbs sweep supporting both NUTS and analytic kernels."""
        model_kwargs = {} if model_kwargs is None else model_kwargs
        postprocess_fn = self.postprocess_fn(model_args, model_kwargs)

        z = state.z

        # Constrained-space dict built from bijectors — zero model evaluations.
        # Deterministic sites (e.g. coeff) are absent; computed on-demand below.
        z_constrained = {
            name: self._site_bijectors[name](val)
            for name, val in z.items()
            if name in self._site_bijectors
        }

        hmc_states, diverging, rng_keys = [], [], []

        for hmc_state, kernel, n_steps in zip(
            state.hmc_states, self.inner_kernels, self.num_steps
        ):
            for _ in range(n_steps):

                if isinstance(kernel, self.AnalyticTransition):
                    # ── Analytic path ──────────────────────────────────────
                    # One full forward pass to obtain deterministic sites
                    # (e.g. coeff) with values from the freshest z.
                    z_constrained_full = postprocess_fn(z)
                    hmc_state = kernel.sample(
                        hmc_state,
                        model_args,
                        model_kwargs,
                        z_constrained=z_constrained_full,
                    )
                    # The analytic draw lives in unconstrained space already
                    # (AnalyticRhoTransition.sample applies .inv before storing).

                else:
                    # ── NUTS / HMC path ────────────────────────────────────
                    z_cond_constrained = {
                        k: z_constrained[k] for k in kernel._cond_sites
                    }

                    # Default-arg capture avoids closure bug across loop iterations
                    def potential_fn(z_hmc, _cond=z_cond_constrained):
                        return kernel._potential_fn_gen(
                            *model_args, _cond_sites=_cond, **model_kwargs
                        )(z_hmc)

                    if kernel._forward_mode_differentiation:
                        pe     = potential_fn(hmc_state.z)
                        z_grad = jacfwd(potential_fn)(hmc_state.z)
                    else:
                        pe, z_grad = value_and_grad(potential_fn)(hmc_state.z)

                    hmc_state = hmc_state._replace(z_grad=z_grad, potential_energy=pe)
                    hmc_state = kernel.sample(
                        hmc_state,
                        model_args,
                        model_kwargs | {"_cond_sites": z_cond_constrained},
                    )

                    # Incremental update — only sites owned by this kernel
                    for name, val in hmc_state.z.items():
                        if name in self._site_bijectors:
                            z_constrained[name] = self._site_bijectors[name](val)

                # Update z after every sub-step so later sub-steps (and later
                # kernels) condition on the freshest state
                z = z | hmc_state.z

            hmc_states.append(hmc_state)
            diverging.append(hmc_state.diverging)
            rng_keys.append(hmc_state.rng_key)

        return MultiHMCGibbsState(
            z,
            hmc_states,
            jnp.stack(diverging),
            jnp.stack(rng_keys),
            hmc_state.potential_energy,
        )


def model_maker(raw_residuals, 
                super_sig,
                vary_white = False,
                wn_lower_bound = None,
                wn_upper_bound = None,
                tm_model = None,
                helpers = None,
                save_red_coeff = False,
                fixed_white_noise_params = None,
                ):
                
    ######################################## Timing Model ########################################
    if tm_model:
        lam = numpyro.sample("timing_lam", dist.HalfNormal(10.0))
        k = numpyro.sample("timing_k", dist.Normal(0, 50).expand([tm_model.nparams_total]))
        stochastic_res = tm_model.residuals(k * lam)
    else:
        stochastic_res = raw_residuals

    ######################################## White Noise ########################################
    if vary_white:
        theta_wn = numpyro.sample('white_noise', dist.Uniform(wn_lower_bound, wn_upper_bound))
        helpers_now = super_sig.get_helpers(reff = stochastic_res,
                                  white_noise_params = theta_wn)

    elif not vary_white and tm_model:
        helpers_now = super_sig.get_helpers(reff = stochastic_res,
                                  white_noise_params = fixed_white_noise_params)
    else:
        helpers_now = helpers

    ######################################## Red Noise ########################################
    xs = numpyro.sample('red_noise', dist.Uniform(super_sig.model.lower_prior_lim_all, 
                                                  super_sig.model.upper_prior_lim_all))
    # evaluate the posterior
    # if super_sig.has_cor:
    #     z_a = numpyro.sample('z_a', dist.Normal(0, 1).expand((super_sig.npsrs, 2*super_sig.data.num_gwb_bins)))
    #     lprob, coeff = super_sig.partial_marg_lnposterior(helpers = helpers_now, red_params = xs, z = z_a)
    # else:
    z_a = numpyro.sample('z_a', dist.Normal(0, 1).expand((super_sig.npsrs, super_sig.nmodes)))
    lprob, coeff = super_sig.lnposterior_reparam(helpers = helpers_now, red_params = xs, z = z_a)

    numpyro.factor('lnpost', lprob + 0.5 * jnp.sum(z_a**2))
    if save_red_coeff:
        numpyro.deterministic('coeff', coeff)