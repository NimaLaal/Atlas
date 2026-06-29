import copy
from itertools import chain

from collections import namedtuple, Counter
from functools import partial
from numpyro.distributions.transforms import biject_to

from jax import device_put, jacfwd, random, value_and_grad, numpy as jnp, vmap
from numpyro.handlers import condition, seed, substitute, trace
from numpyro.infer.initialization import init_to_sample, init_to_uniform
from numpyro.infer.mcmc import MCMCKernel
from numpyro.util import is_prng_key
from types import SimpleNamespace
import numpyro.distributions as dist

class AnalyticRhoTransition(MCMCKernel):
    """Analytic Gibbs draw for a named half-log10-rho site.

    Parameters
    ----------
    model : callable
        NumPyro model.
    sig : object
        Signal object with ``prior_draw`` and ``posterior_draw_from_coeff``.
    rho_site : str
        Name of the half-log10-rho sample site. Default ``"psr_half_log10_rho"``.
    coeff_site : str
        Name of the coefficient deterministic site. Default ``"coeff"``.
    rho_low : float
        Lower bound of the uniform prior on rho. Default ``-9.0``.
    rho_high : float
        Upper bound of the uniform prior on rho. Default ``-2.0``.
    """

    sample_field = "z"

    def __init__(
        self,
        model,
        sig,
        rho_site: str = "psr_half_log10_rho",
        coeff_site: str = "coeff",
        rho_low: float = -9.0,
        rho_high: float = -2.0,
    ):
        self._model = model
        self.model = model
        self.sig = sig
        self.rho_site = rho_site
        self.coeff_site = coeff_site
        self.rho_low = rho_low
        self.rho_high = rho_high
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
        rng_key, subkey = jrandom.split(state.rng_key)

        current_coeff = z_constrained[self.coeff_site]

        # draw in constrained space (rho_low, rho_high)
        rho_constrained = self.sig.posterior_draw_from_coeff(
            current_coeff[..., None], subkey
        )[0]

        # store unconstrained so postprocess_fn(z)[rho_site] == rho_constrained
        rho_unconstrained = self._rho_transform.inv(rho_constrained)

        return AnalyticState(
            z={self.rho_site: rho_unconstrained},
            rng_key=rng_key,
            diverging=jnp.array(False),
            potential_energy=jnp.array(0.0),
        )