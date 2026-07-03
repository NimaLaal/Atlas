"""Per-parameter sampling routing keyed on NUMERICAL PRECISION, not parameter name.

Principle
---------
A param sampled as theta = theta0 + delta and then differenced inside the model
(theta - theta0) loses precision when |delta|/|theta0| approaches float64 epsilon
(eps = 2.22e-16).  The round-trip recovers `delta` with relative error ~ eps/ratio,
where ratio = |delta_scale| / |theta0|.  So:

  ratio ~ eps (1e-16)  -> delta fully lost (param frozen): F0 (278 Hz, sig~1e-13).
  ratio ~ 1e-10        -> rel err ~ eps/1e-10 = 2e-6  -> ~10 of 16 sig figs kept.
  ratio ~ 1            -> rel err ~ eps              -> exact.

Policy
------
PREC_THRESH = 1e-10.  Route a param to OFFSET sampling (sample `delta` directly,
never forming theta0+delta) when ratio < PREC_THRESH; else ABSOLUTE reconstruction.

Margin justification: the catastrophic wall is at ratio ~ eps ~ 1e-16.  1e-10 sits
6 orders of magnitude ABOVE the wall.  A param routed to ABSOLUTE recovers its offset
with relative error ~ eps/ratio <= eps/1e-10 = 2.2e-6, i.e. ALWAYS >= ~6 sig figs of
the offset retained (the offset itself is ~sigma, so 6 sig figs of sigma is far finer
than any recovery needs).  The 6-decade margin covers (a) the sampler exploring
several sigma, (b) round-off over many gradient evals per leapfrog, (c) borderline
params.  Tighter (1e-13) would let a param losing ~10 digits go absolute; looser
(1e-6) would needlessly offset-route safe params.  Near-wall params, by contrast,
would lose >6 digits (down to total freeze at F0's 3.6e-16) and MUST go offset.

Validity constraint
-------------------
OFFSET sampling uses the EXACT linear form delta * (d resid / d theta) with a CONSTANT
Jacobian column -> valid ONLY for params linear in the residual (F0,F1,DM,JUMP,FD,FB,
position-to-1st-order).  Nonlinear params (M2,SINI,ECC,OM,PB,...) have no constant
column and MUST stay absolute.  They never need offset anyway: their offset/value
ratio is ~0.1, nowhere near eps.  If a param is BOTH near the wall AND nonlinear, the
router returns 'conflict' (needs longdouble / special handling) rather than silently
offset-routing a nonlinear param.
"""
import numpy as np
import jax, jax.numpy as jnp

EPS = float(np.finfo(np.float64).eps)        # 2.220446e-16
PREC_THRESH = 1e-10                          # offset-route below this ratio
LIN_TOL = 1e-6                               # Jacobian-column constancy tol for 'linear'


def precision_ratio(theta0, sigma):
    """ratio = |sigma| / |theta0|.  theta0 ~ 0 -> inf (no cancellation possible)."""
    t = abs(float(theta0))
    if t == 0.0 or not np.isfinite(t):
        return np.inf
    return abs(float(sigma)) / t


def exploration_probe(theta0, sigma):
    """Float64-resolvable probe step at the SAMPLER'S EXPLORATION SCALE (a few sigma),
    floored at a few ULP of theta0 so it is representable even when sigma is sub-ULP
    (F0).  Testing linearity here — not at some arbitrary larger step — is essential:
    a globally-curved param (ELONG: position->L_hat->Roemer is trigonometric) is
    LOCALLY LINEAR over its tiny sigma-range, which is exactly the range offset
    sampling uses.  Probing at |theta0|*1e-6 would falsely flag such a param nonlinear."""
    ulp = float(np.spacing(abs(float(theta0)))) if float(theta0) != 0 else 0.0
    return max(3.0 * abs(float(sigma)), 4.0 * ulp, 1e-300)


def is_linear(dm, base, name, probe):
    """True if d(resid)/d(name) is constant over the probe step (locally linear).
    `probe` should be the exploration-scale step (see exploration_probe): linearity is
    judged where the sampler operates, so near-wall params correctly read as locally
    linear.  Returns (linear: bool, rel_col_change: float)."""
    def col_at(shift):
        b = dict(base)
        b[name] = jnp.asarray(float(base[name]) + shift)
        tang = {k: jnp.array(0.0) for k in base}; tang[name] = jnp.array(1.0)
        return np.asarray(jax.jvp(dm, (b,), (tang,))[1])
    c0 = col_at(0.0)
    c1 = col_at(probe)
    denom = np.linalg.norm(c0)
    if denom == 0.0:
        return False, np.inf                 # zero column -> not a usable linear param
    rel = np.linalg.norm(c1 - c0) / denom
    return rel < LIN_TOL, float(rel)


class RoutingConflictError(RuntimeError):
    """Raised when a param is near the float64 wall (offset MUST be used to avoid
    cancellation) but is NOT locally linear over its exploration range (offset CANNOT
    be used — no constant Jacobian column).  Such a param cannot be sampled safely in
    float64 by either path; it needs longdouble handling (or a non-affine reparam).
    This is the bucket that protects an arbitrary future pulsar: it fails loudly rather
    than silently mis-routing a nonlinear param down the constant-column offset path."""


def route(name, theta0, sigma, dm, base, probe=None):
    """Return dict: {bucket, ratio, linear, rel_col_change, near_wall}.
    bucket in {'offset','absolute','conflict'}.  Pure classifier (no raise) — use
    route_checked() in production paths to fail loudly on conflict."""
    ratio = precision_ratio(theta0, sigma)
    if probe is None:
        probe = exploration_probe(theta0, sigma)     # few-sigma exploration scale
    linear, rel = is_linear(dm, base, name, probe)
    near_wall = ratio < PREC_THRESH
    if near_wall and linear:
        bucket = "offset"
    elif near_wall and not linear:
        bucket = "conflict"                  # near wall but no constant column
    else:
        bucket = "absolute"
    return dict(bucket=bucket, ratio=ratio, linear=linear, rel_col_change=rel,
                near_wall=near_wall)


def route_checked(name, theta0, sigma, dm, base, probe=None):
    """route(), but RAISE on conflict so a hard case errors visibly instead of
    silently mis-routing.  This is the live guard for arbitrary pulsars."""
    r = route(name, theta0, sigma, dm, base, probe)
    if r["bucket"] == "conflict":
        raise RoutingConflictError(
            f"param {name!r}: ratio={r['ratio']:.2e} < {PREC_THRESH:.0e} (near float64 "
            f"wall, eps={EPS:.1e}) BUT not locally linear (rel_col_change="
            f"{r['rel_col_change']:.2e} >= {LIN_TOL:.0e}). Offset sampling needs a "
            f"constant Jacobian column; this param has neither a safe absolute path "
            f"(cancellation) nor a valid offset path. Needs longdouble / custom reparam.")
    return r
