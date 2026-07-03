"""Hessian preconditioning + offset sampling for the JUG-backed timing model.

These utilities turn JUG's validated full-covariance Hessian into a NUTS metric and
provide the float64-safe offset coordinate for near-wall linear params.  They were
validated on the full 21-parameter J1600-3053 timing model: unbiased recovery from a
+-3sigma start (all params within 0.16 sigma of truth, sigma_post/sigma_JUG in
[0.97, 1.01]), 18/21 with a single frozen full metric and 21/21 once the tight
chromatic/jump sub-block (FD1/FD2/JUMP1, 84-94% correlated) is Gibbs-blocked with its
own conditional metric.  See routing.py for the precision-keyed offset/absolute policy.

Design notes
------------
* OFFSET sampling (offset_columns): for params routed 'offset' (near the float64 wall,
  e.g. F0, ELONG), sample the offset dF directly and add dF * (d resid / d param), where
  the Jacobian column is taken once at theta0.  This NEVER forms theta0 + dF, so it
  avoids the (278 + 1e-13) - 278 cancellation that freezes physical-F0 sampling.  Exact
  for linear params; valid for near-wall params generally (locally linear over their
  tiny exploration range).
* METRIC seeding (jug_metric): the JUG covariance C = (X^T N^-1 X)^-1 equals sigma_post
  (validated), so its transform into the sampler's unconstrained coordinates is the
  optimal NUTS mass matrix.  FREEZE it (adapt_mass_matrix=False): numpyro's window
  adaptation degrades a near-singular high-dim seed from few warmup samples, and a mass
  matrix never biases the target, so freezing the validated metric is both safe and
  necessary at high dimension.
* GIBBS blocking (gibbs_block_metrics): when a tight sub-block remains stiff under the
  full (ill-conditioned) metric, sample it as its own Gibbs group with its CONDITIONAL
  covariance inv(H[block, block]) (H = metric^-1), which is far better conditioned.
"""
import numpy as np
import jax, jax.numpy as jnp


def offset_columns(delta_m_us, base, names):
    """Jacobian columns d(delta_m_us)/d(param) at ``base`` for the offset params.

    Parameters
    ----------
    delta_m_us : callable(theta_dict) -> jnp.ndarray
        The timing-model delta function (microseconds).
    base : dict
        theta0 dict (all sampled params), the point at which the columns are taken.
    names : list[str]
        Params to build columns for (the 'offset'-routed ones).

    Returns
    -------
    dict[str, jnp.ndarray]
        {name: column}.  Use as  residual_contribution = dF * column  with dF the
        sampled offset (so theta0 + dF is never formed -> no float64 cancellation).
    """
    cols = {}
    for name in names:
        tang = {k: jnp.array(0.0) for k in base}
        tang[name] = jnp.array(1.0)
        cols[name] = jax.jvp(delta_m_us, (base,), (tang,))[1]
    return cols


def sigmoid_jacobian_uniform(value, lo, hi):
    """d(physical)/d(unconstrained) at ``value`` for a numpyro Uniform(lo,hi) site,
    whose unconstrained transform is  physical = lo + (hi-lo)*sigmoid(x).  Used for the
    bounded reparametrised params (M2 ~ U(0,3); cos i ~ U(-1,1))."""
    s = (float(value) - lo) / (hi - lo)
    return (hi - lo) * s * (1.0 - s)


def coordinate_jacobian(order, theta0, sigJUG, bound_specs):
    """Diagonal coordinate->physical Jacobian D for the metric transform.

    Offset and affine params scale by sigma_JUG (the sampler coord is the sigma_JUG-
    scaled offset / z, exact).  Bounded params use their sigmoid-transform Jacobian at
    theta0 (approximate, fine for a preconditioner).  For an inclination param sampled
    as cos i with the physical entry being SINI, pass bound_specs[name] = ('sini',
    cosi0, sini0) so D folds in dSINI/dcosi = -cosi/SINI as well.

    Parameters
    ----------
    order : list[str]                 param order matching the covariance rows
    theta0 : dict                     theta0 values
    sigJUG : dict                     per-param JUG formal sigma (affine/offset scale)
    bound_specs : dict[str, tuple]    {name: ('uniform', lo, hi)} or
                                      {name: ('sini', cosi0, sini0)}

    Returns
    -------
    np.ndarray  (len(order),)   the diagonal D (signed)
    """
    D = np.empty(len(order))
    for i, k in enumerate(order):
        if k in bound_specs:
            spec = bound_specs[k]
            if spec[0] == "uniform":
                D[i] = sigmoid_jacobian_uniform(theta0[k], spec[1], spec[2])
            elif spec[0] == "sini":
                _, cosi0, sini0 = spec
                dcosi_dx = sigmoid_jacobian_uniform(cosi0, -1.0, 1.0)
                D[i] = (-cosi0 / sini0) * dcosi_dx          # dSINI/dx
            else:
                raise ValueError(f"unknown bound spec {spec!r}")
        else:
            D[i] = float(sigJUG[k])                          # offset / affine
    return D


def jug_metric(C_phys, D, jitter_rel=1e-12):
    """Transform JUG's physical covariance sub-block into the sampler's unconstrained
    coordinates:  Sigma_unc = D^-1 C_phys D^-1  (D diagonal).  Adds a tiny PD jitter.

    Pass Sigma_unc as a frozen dense inverse_mass_matrix:
        NUTS(model, dense_mass=[SITES], inverse_mass_matrix={SITES: Sigma_unc},
             adapt_mass_matrix=False)
    """
    C = np.asarray(C_phys, float)
    S = C / np.outer(D, D)
    w = np.linalg.eigvalsh(S)
    jit = max(0.0, -w.min()) + jitter_rel * np.trace(S) / S.shape[0]
    return S + jit * np.eye(S.shape[0])


def gibbs_block_metrics(Sigma_unc, block_index_lists):
    """Conditional covariances for Gibbs blocking.  For block A (given the rest),
    the correct per-block metric is inv(H[A,A]) where H = Sigma_unc^-1 (precision) —
    the CONDITIONAL covariance, far better conditioned than the marginal sub-block when
    a tight degenerate sub-block (e.g. FD1/FD2/JUMP1) sits inside an ill-conditioned
    full metric.

    Parameters
    ----------
    Sigma_unc : np.ndarray            full unconstrained metric (from jug_metric)
    block_index_lists : list[list[int]]   row indices of each Gibbs block

    Returns
    -------
    list[np.ndarray]    per-block conditional covariance (the frozen inverse_mass for
                        each block's inner NUTS kernel)
    """
    H = np.linalg.inv(np.asarray(Sigma_unc, float))
    return [np.linalg.inv(H[np.ix_(ix, ix)]) for ix in block_index_lists]
