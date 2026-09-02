"""The NumPyro model at the top of an ATLAS run.

``model_maker`` is the actual entry point of a fit: it samples the timing model
(optionally), the white noise (optionally), the red-noise spectrum and the
non-centred coefficients, then hands the log-density to NumPyro. It lived in
``samplers/canetoadracing.py`` next to the vendored ``MultiHMCGibbs`` kernels,
where it both hid the top of the model inside a sampler module and collided by
name with ``SuperSignal.model_maker`` -- an unrelated method that builds the phi
parameterisation. Grep for one and you found the other.

``from ATLAS.samplers.canetoadracing import model_maker`` still works.

The ``numpyro.factor('lnpost', lprob + 0.5 * sum(z_a**2))`` line at the end is
load-bearing and not obviously so. Both likelihoods return a density in the
*reparameterised* coordinate, and NumPyro has already added the ``N(0, 1)``
log-prior for ``z_a`` from the ``sample`` statement above; the ``+0.5 sum(z^2)``
cancels that double-count. Removing it silently corrupts the posterior rather
than raising, which is why ``tests/test_identities.py`` pins the identity
directly.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

__all__ = ["model_maker"]


def model_maker(raw_residuals,
                super_sig,
                marg_over_non_gwb,
                vary_white = False,
                wn_lower_bound = None,
                wn_upper_bound = None,
                tm_model = None,
                helpers = None,
                save_red_coeff = False,
                fixed_white_noise_params = None,
                red_noise_basis = None,
                tm_direct_sampling_type = 'klam',
                ):
    """NumPyro model for the ATLAS global fit.

    Parameters
    ----------
    red_noise_basis : array, optional
        The T-matrix. Only consulted when the helpers are rebuilt inside the model
        (``vary_white=True``, or a sampled timing model). Defaults to the signal's
        own ``get_Fmat_concat``.

        Pass it explicitly -- as a model argument through ``MCMC.run``, with
        ``MCMC(..., jit_model_args=True)`` -- when the helpers are rebuilt every
        leapfrog step. Left to default it is captured from the signal object and
        XLA writes it into the potential-energy executable as a literal; on the
        NANOGrav 15-year set that is several GB of generated code. As a model
        argument it is traced, and the executable holds a pointer instead.

    tm_direct_sampling_type : str, optional
        How a sampled timing model is parameterised. ``'klam'`` (the default)
        draws a global scale ``timing_lam`` and unit-scale coefficients
        ``timing_k`` and forms the residuals from their product. Anything else
        calls ``tm_model.sample_residuals()``, which samples each pulsar's
        physical timing parameters directly from their bounded priors.
    """

    ######################################## Timing Model ########################################
    if tm_model:
        if tm_direct_sampling_type == 'klam':
            lam = numpyro.sample("timing_lam", dist.HalfNormal(10.0))
            k = numpyro.sample("timing_k", dist.Normal(0, 50).expand([tm_model.nparams_total]))
            stochastic_res = tm_model.residuals(k * lam)
        else:
            stochastic_res = tm_model.sample_residuals()
    else:
        stochastic_res = raw_residuals

    ######################################## White Noise ########################################
    if vary_white:
        theta_wn = numpyro.sample('white_noise', dist.Uniform(wn_lower_bound, wn_upper_bound))
        helpers_now = super_sig.get_helpers(reff = stochastic_res,
                                  white_noise_params = theta_wn,
                                  red_noise_basis = red_noise_basis)

    elif not vary_white and tm_model:
        helpers_now = super_sig.get_helpers(reff = stochastic_res,
                                  white_noise_params = fixed_white_noise_params,
                                  red_noise_basis = red_noise_basis)
    else:
        helpers_now = helpers

    ######################################## Red Noise ########################################
    xs = numpyro.sample('red_noise', dist.Uniform(super_sig.model.lower_prior_lim_all, 
                                                  super_sig.model.upper_prior_lim_all))
    # evaluate the posterior
    if marg_over_non_gwb:
        z_a = numpyro.sample('z_a', dist.Normal(0, 1).expand((super_sig.npsrs, 2*super_sig.data.num_gwb_bins)))
        lprob, coeff = super_sig.partial_marg_lnposterior(helpers = helpers_now, red_params = xs, z = z_a)
    else:
        z_a = numpyro.sample('z_a', dist.Normal(0, 1).expand((super_sig.npsrs, super_sig.nmodes)))
        lprob, coeff = super_sig.lnposterior_reparam(helpers = helpers_now, red_params = xs, z = z_a)

    numpyro.factor('lnpost', lprob + 0.5 * jnp.sum(z_a**2))
    if save_red_coeff:
        numpyro.deterministic('coeff', coeff)
