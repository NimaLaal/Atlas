"""Deterministic synthetic PTA fixtures.

Generated from a seed rather than committed as a binary: ``numpy``'s PCG64
stream is guaranteed stable across releases, so this is reproducible without
putting a blob in git, and CI can build it with nothing but NumPy.

The generator deliberately produces the three structural features that the
identity tests need in order to exercise the code paths that actually break:

* **multi-TOA epochs** -- observing sessions carry three TOAs a few hundred
  milliseconds apart at different radio frequencies, all on one backend, so
  ``_get_epochs`` (threshold 1 s, grouped per backend) finds genuine ECORR
  blocks rather than the degenerate one-TOA-per-epoch case;
* **two backends per pulsar**, so ``nvec``/``jvec`` indexing through ``B``
  and ``V`` is non-trivial;
* **unequal timing-model widths** across pulsars, so ``SuperSignal`` pads to
  a common ``linear_timing_model_size`` and ``_pad_mask`` is non-empty --
  the padded-column machinery is otherwise never touched.

TOAs are absolute seconds of order 4.6e9, matching real data, because the
conditioning of the Fourier basis depends on that scale.
"""

from __future__ import annotations

import numpy as np

from .pulsar import FixturePulsar

__all__ = ["make_synth_pta", "SYNTH2_SEED"]

SYNTH2_SEED = 20260831

_YEAR = 31557600.0          # seconds in a Julian year
_T0 = 4.6e9                 # absolute epoch, ~MJD 53000 in seconds

# Well-separated unit vectors: pairwise angles ~50-110 deg, so the
# Hellings-Downs ORF takes distinct, non-degenerate values.
_POSITIONS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.30, 0.85, 0.43],
        [-0.55, 0.42, 0.72],
        [0.10, -0.75, 0.65],
        [-0.80, -0.35, 0.49],
    ]
)


def _design_matrix(toas, radio_freqs, ncol, tspan):
    """A plausible timing design matrix with ``ncol`` columns.

    Columns are ordered the way a real one is: offset, spin frequency and its
    derivative, then astrometry (annual), then a dispersion column.  Scaled to
    O(1) before ATLAS's SVD sees them, which is what a real design matrix
    looks like after ``fit`` normalisation.
    """
    tbar = (toas - toas.mean()) / tspan
    cols = [
        np.ones_like(tbar),                        # phase offset
        tbar,                                      # F0
        tbar**2,                                   # F1
        np.sin(2 * np.pi * (toas - _T0) / _YEAR),  # RA-like
        np.cos(2 * np.pi * (toas - _T0) / _YEAR),  # DEC-like
        (1400.0 / radio_freqs) ** 2,               # DM
    ]
    if ncol > len(cols):
        raise ValueError(f"ncol={ncol} exceeds the {len(cols)} columns defined here")
    # Keep the DM column whenever we truncate -- it is the one that is
    # correlated with the radio frequencies and so with any chromatic model.
    keep = list(range(ncol - 1)) + [len(cols) - 1] if ncol > 1 else [0]
    return np.column_stack([cols[i] for i in keep[:ncol]])


def _powerlaw_realisation(rng, toas, tspan, nfreq, log10_A, gamma):
    """A red-noise realisation drawn on the first ``nfreq`` harmonics of 1/T."""
    f = np.arange(1, nfreq + 1) / tspan
    df = 1.0 / tspan
    fref = 1.0 / _YEAR
    psd = df * 10 ** (2 * log10_A) / (12 * np.pi**2 * f**3) * (f / fref) ** (3 - gamma)
    amps = rng.normal(scale=np.sqrt(psd)[:, None], size=(nfreq, 2))
    phase = 2 * np.pi * f[None, :] * toas[:, None]
    return (np.sin(phase) @ amps[:, 0]) + (np.cos(phase) @ amps[:, 1])


def make_synth_pta(
    npsr: int = 2,
    seed: int = SYNTH2_SEED,
    tspan_yr: float = 12.0,
    sessions: tuple[int, ...] = (84, 68, 76, 60, 72),
    ncols: tuple[int, ...] = (6, 4, 5, 3, 6),
    toas_per_session: int = 3,
    log10_A_irn: float = -14.5,
    gamma_irn: float = 3.5,
    log10_A_gwb: float = -14.8,
    gamma_gwb: float = 13 / 3,
):
    """Build ``npsr`` synthetic pulsars.

    Returns a list of :class:`FixturePulsar`.  Residuals contain a white-noise
    draw (EFAC/EQUAD/ECORR structure), a per-pulsar red-noise realisation and
    a *common* (uncorrelated-realisation) red process, so that a fit to this
    data is not pathological.  The injected values are not the point -- these
    fixtures exist to test algebra, not recovery.
    """
    if npsr > len(_POSITIONS):
        raise ValueError(f"only {len(_POSITIONS)} positions defined")

    rng = np.random.default_rng(seed)
    tspan = tspan_yr * _YEAR
    psrs = []

    for i in range(npsr):
        nsess = sessions[i]
        ncol = ncols[i]

        # Sessions spread over the span, jittered, with a small per-pulsar
        # offset so the pulsars do not share an observing grid.
        start = _T0 + i * 0.11 * _YEAR
        session_t = np.sort(
            start
            + np.linspace(0.0, tspan, nsess)
            + rng.uniform(-2.0, 2.0, size=nsess) * 86400.0
        )

        # Three TOAs per session, a few hundred ms apart -> one ECORR epoch.
        within = np.linspace(0.0, 0.3, toas_per_session)
        toas = (session_t[:, None] + within[None, :]).ravel()

        radio = np.tile(np.array([430.0, 820.0, 1400.0])[:toas_per_session], nsess)

        # Two backends, switching partway through -- each session sits wholly
        # inside one backend, so epochs never straddle the boundary.
        sess_backend = np.where(np.arange(nsess) < nsess // 2, "bk_A", "bk_B")
        backend_flags = np.repeat(sess_backend, toas_per_session)

        # Backend-dependent TOA uncertainties, lognormal spread.
        base = np.where(backend_flags == "bk_A", 0.5e-6, 1.2e-6)
        toaerrs = base * np.exp(rng.normal(0.0, 0.25, size=toas.size))

        # White-noise realisation: EFAC=1.1, EQUAD=3e-7 s, ECORR=2e-7 s.
        nvec = (1.1**2) * (toaerrs**2 + (3e-7) ** 2)
        white = rng.normal(scale=np.sqrt(nvec))
        ecorr = np.repeat(rng.normal(scale=2e-7, size=nsess), toas_per_session)

        red = _powerlaw_realisation(rng, toas, tspan, 30, log10_A_irn, gamma_irn)
        gwb = _powerlaw_realisation(rng, toas, tspan, 14, log10_A_gwb, gamma_gwb)

        residuals = white + ecorr + red + gwb
        Mmat = _design_matrix(toas, radio, ncol, tspan)

        psrs.append(
            FixturePulsar(
                name=f"J{i:04d}+{i:04d}",
                toas=toas,
                residuals=residuals,
                toaerrs=toaerrs,
                freqs=radio,
                pos=_POSITIONS[i] / np.linalg.norm(_POSITIONS[i]),
                Mmat=Mmat,
                backend_flags=backend_flags,
                pdist=(1.0 + 0.3 * i, 0.2),
            )
        )

    return psrs


def make_adaptus_basis(psrs, nmodes=8, seed=SYNTH2_SEED + 1):
    """A synthetic stand-in for an Adaptus (``gtm``) timing basis.

    The real thing PCAs prior-predictive timing residuals and keeps the leading
    components, scaled by ``sqrt(explained_variance)`` -- so columns are
    orthogonal with geometrically decreasing norm.  This mimics that shape
    using smooth Chebyshev-like functions of normalised time, skipping the
    constant and linear terms the timing model already carries, then
    orthonormalising.

    Returns ``(basis, gtm_psd)`` where ``basis`` is a list of ``[ntoa, nmodes]``
    arrays and ``gtm_psd`` is ``[nmodes, npsr]`` -- the shape
    ``parameterized`` asserts for a directly supplied ``gtm_psd``.
    """
    if nmodes % 2 != 0:
        raise ValueError(f"nmodes must be even, got {nmodes}")
    rng = np.random.default_rng(seed)
    basis = []
    for p in psrs:
        t = np.asarray(p.toas, dtype=np.float64)
        x = 2 * (t - t.min()) / (t.max() - t.min()) - 1.0
        cols = [np.cos(k * np.arccos(np.clip(x, -1, 1))) for k in range(2, nmodes + 2)]
        M = np.column_stack(cols)
        # Nudge off exact degeneracy so the QR is not artificially perfect.
        M += rng.normal(scale=1e-6, size=M.shape)
        Q, _ = np.linalg.qr(M)
        scale = 1e-6 * 10.0 ** (-0.15 * np.arange(nmodes))
        basis.append(Q[:, :nmodes] * scale[None, :])
    # Non-constant across modes AND pulsars, so a mis-sliced gtm block shows up.
    gtm_psd = np.array([[10.0 ** (-0.1 * m - 0.05 * i) for i in range(len(psrs))]
                        for m in range(nmodes)])
    return basis, gtm_psd
