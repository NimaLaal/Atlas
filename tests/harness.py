"""Shared construction for the identity tests.

Builds an ATLAS model and the matching independent reference ingredients from
the same fixture, so a test body is a one-line comparison.

Imports only ``ATLAS.data``, ``ATLAS.model_builder``, ``ATLAS.nMatrix`` and
``ATLAS.psd_functions`` -- none of which need ``jug``, ``enterprise``,
``libstempo`` or ``$TEMPO2``.  ``ATLAS.pulsar`` is deliberately never touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from scipy.linalg import block_diag

from . import reference as ref
from .fixtures.pulsar import load_fixture
from .fixtures.synthetic import make_synth_pta

DATA_DIR = Path(__file__).resolve().parent / "fixtures" / "data"

from ATLAS.data import PTA_Data
from ATLAS.model_builder import ModelBuilder
from ATLAS.nMatrix.base import WhiteCov
from ATLAS.psd_functions import powerlaw, hd_orf

# One white-noise point, used everywhere so the reference and ATLAS agree by
# construction on the parameters rather than by accident.
EFAC = 1.1
LOG10_EQUAD = float(np.log10(3e-7))
LOG10_ECORR = float(np.log10(2e-7))
TOAS_PER_SESSION = 3

ZERO_ORF = lambda angle: jnp.zeros_like(jnp.asarray(angle))
ORFS = {"hd": hd_orf, "zero": ZERO_ORF}


@dataclass
class Model:
    psrs: list
    data: object
    wn: object
    rn: object
    helpers: tuple
    wn_vec: jnp.ndarray
    residuals: np.ndarray
    n_gwb: int
    n_irn: int
    orf_name: str
    include_ecorr: bool

    @property
    def npsr(self):
        return len(self.psrs)

    @property
    def n_tm(self):
        return self.rn.linear_timing_model_size

    @property
    def tm_widths(self):
        return [p.Mmat.shape[1] for p in self.psrs]

    def red_params(self, irn=(-15.0, 3.5), gwb=(-14.8, 13 / 3), irn_overrides=None):
        """Flat red-noise vector: [irn_log10A, irn_gamma] per pulsar, then GWB."""
        vals = []
        for i in range(self.npsr):
            a, g = irn
            if irn_overrides and i in irn_overrides:
                a, g = irn_overrides[i]
            vals += [a, g]
        return jnp.array(vals + list(gwb))


def fixture_available(name):
    return name == "synth" or (DATA_DIR / f"{name}.npz").exists()


def load_psrs(fixture, npsr):
    if fixture == "synth":
        return make_synth_pta(npsr)
    psrs, _ = load_fixture(DATA_DIR / f"{fixture}.npz")
    return psrs[:npsr] if npsr else psrs


def white_noise_vector(psrs, include_ecorr=True):
    """Flat WN vector: per pulsar, [EFACs | log10_equads | log10_ecorrs].

    Backend counts differ per pulsar in real data, so this cannot be a tile.
    """
    out = []
    for p in psrs:
        nb = np.unique(np.asarray(p.backend_flags)).size
        out += [EFAC] * nb + [LOG10_EQUAD] * nb
        if include_ecorr:
            out += [LOG10_ECORR] * nb
    return jnp.array(out)


@lru_cache(maxsize=32)
def build(npsr=2, model_string="ltm|unc+cor->unc", linear_timing=True,
          marg_timing=False, orf_name="hd", n_gwb=4, n_irn=6, include_ecorr=True,
          fixture="synth"):
    psrs = load_psrs(fixture, npsr)
    npsr = len(psrs)
    data = PTA_Data(
        psrs, num_gwb_bins=n_gwb, num_irn_bins=n_irn, num_dm_bins=None,
        adaptus_basis=None, adaptus_size=None, fixed_white_noise_params=None,
        linear_timing=linear_timing, marg_timing=marg_timing,
        diag_white_cov=False, fixed_res=False,
        timfiles=None, parfiles=None, noise_dict=None, dm_ref_freq=1400,
    )
    wn = WhiteCov(data=data, stabilize_TNT=False, include_ecorr=include_ecorr)
    rn = ModelBuilder(data=data).make_red_noise(
        model_string, use_pulsar_tspan=False,
        irn_psd_function=powerlaw, gwb_psd_function=powerlaw,
        orf_function=ORFS[orf_name], dm_psd_function=None,
        irn_lower_bound_psd=jnp.array([-20.0, 0.0]),
        irn_upper_bound_psd=jnp.array([-11.0, 7.0]),
        gwb_lower_bound_psd=jnp.array([-18.0, 0.0]),
        gwb_upper_bound_psd=jnp.array([-11.0, 7.0]),
        dm_lower_bound_psd=None, dm_upper_bound_psd=None,
        upper_bound_orf=None, lower_bound_orf=None,
    )
    wn_vec = white_noise_vector(psrs, include_ecorr)
    residuals = np.concatenate([np.asarray(p.residuals) for p in psrs])
    helpers = rn.get_helpers(reff=jnp.concat(data.raw_residuals),
                             white_noise_params=wn_vec)
    return Model(psrs, data, wn, rn, helpers, wn_vec, residuals,
                 n_gwb, n_irn, orf_name, include_ecorr)


# --------------------------------------------------------------------------- #
#  Independent reference ingredients for the same model
# --------------------------------------------------------------------------- #

def svd_basis(M):
    """``signals_utils._timing_model_svd``, reimplemented from its docstring."""
    U, _, _ = np.linalg.svd(np.asarray(M, dtype=np.float64), full_matrices=False)
    norm = np.sqrt((U ** 2).sum(axis=0))
    out = U / norm
    out[:, norm == 0] = 0.0
    return out


def freq_grid(model):
    tspan = model.data.pta_tspan
    return (np.arange(1, model.n_irn + 1) / tspan,
            np.arange(1, model.n_gwb + 1) / tspan,
            1.0 / tspan)


def design_blocks(model, include_timing=None):
    """Per-pulsar ``T = [M_padded | F]``, built independently."""
    if include_timing is None:
        include_timing = model.n_tm > 0
    f_irn, _, _ = freq_grid(model)
    blocks = []
    for p in model.psrs:
        F = ref.fourier_basis(p.toas, f_irn)
        if include_timing:
            Ms = svd_basis(p.Mmat)
            Mp = np.zeros((p.ntoa, model.n_tm))
            Mp[:, :Ms.shape[1]] = Ms
            F = np.hstack([Mp, F])
        blocks.append(F)
    return blocks


def noise_blocks(model, include_ecorr=None):
    if include_ecorr is None:
        include_ecorr = model.include_ecorr
    blocks = []
    for p in model.psrs:
        backends = np.unique(np.asarray(p.backend_flags))
        bidx = np.searchsorted(backends, np.asarray(p.backend_flags))
        nvec = ref.white_nvec(p.toaerrs, bidx,
                              np.full(backends.size, EFAC),
                              np.full(backends.size, LOG10_EQUAD))
        if include_ecorr:
            epochs, _ = ref.epochs_from_toas(p.toas, p.backend_flags)
            jvec = np.full(len(epochs), 10.0 ** (2 * LOG10_ECORR))
            blocks.append(ref.dense_N(nvec, epochs, jvec))
        else:
            blocks.append(ref.dense_N(nvec))
    return blocks


def psd_pieces(model, red_params):
    """IRN (per pulsar, bin resolution) and GWB PSDs, from the flat vector."""
    f_irn, f_gwb, df = freq_grid(model)
    xs = np.asarray(red_params, dtype=np.float64)
    irn = np.column_stack([ref.powerlaw_psd(f_irn, df, xs[2 * i], xs[2 * i + 1])
                           for i in range(model.npsr)])
    gwb = ref.powerlaw_psd(f_gwb, df, xs[-2], xs[-1])
    return irn, gwb


def orf_matrix(model):
    n = model.npsr
    pos = np.array([np.asarray(p.pos) for p in model.psrs])
    out = np.eye(n)
    if model.orf_name == "zero":
        return out
    for i in range(n):
        for j in range(i + 1, n):
            zeta = np.arccos(np.clip(pos[i] @ pos[j], -1.0, 1.0))
            out[i, j] = out[j, i] = ref.hd_orf(zeta)
    return out


def global_phi(model, red_params):
    irn, gwb = psd_pieces(model, red_params)
    ncol = model.n_tm + 2 * model.n_irn
    return ref.build_phi(model.npsr, ncol, model.n_tm, model.tm_widths,
                         irn, gwb, orf_matrix(model))


def phiinv_diag(model, red_params):
    """The diagonal precision ATLAS uses to build its standardising transform."""
    irn, gwb = psd_pieces(model, red_params)
    ncol = model.n_tm + 2 * model.n_irn
    pad = np.asarray(model.rn._pad_mask) if model.n_tm else None
    out = []
    for i in range(model.npsr):
        d = np.empty(ncol)
        if model.n_tm:
            d[:model.n_tm] = np.where(pad[i] > 0, 1.0, 1e-40)
        tot = irn[:, i].copy()
        tot[:model.n_gwb] += gwb
        d[model.n_tm:] = 1.0 / np.repeat(tot, 2)
        out.append(d)
    return out


def dense_bundle(model, include_ecorr=None):
    T = design_blocks(model)
    N = noise_blocks(model, include_ecorr)
    return T, N, block_diag(*N), model.residuals
