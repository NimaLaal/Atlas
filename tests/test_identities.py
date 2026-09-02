"""Exact identities ATLAS must satisfy.

Every tolerance here is measured, not aspirational.  Where a path cannot reach
machine precision the reason is stated in the test.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.stats import multivariate_normal

from . import harness as H
from . import reference as ref
from ATLAS.psd_functions import hd_orf as atlas_hd
from ATLAS.signals.signals_utils import get_harmonic_frequencies, get_fourier_design_matrix

TIGHT = 1e-12          # machine-precision identities
LOOSE = 1e-9           # identities through a Cholesky of a correlated phi
GRAD_TOL = 1e-6        # central differences


# --------------------------------------------------------------------------- #
#  0. The reference is itself correct
# --------------------------------------------------------------------------- #

def test_reference_matches_scipy():
    rng = np.random.default_rng(0)
    n, k = 60, 5
    T = rng.normal(size=(n, k))
    N = np.diag(rng.uniform(1.0, 2.0, n))
    phi = np.diag(rng.uniform(0.5, 2.0, k))
    r = rng.normal(size=n)
    mine = ref.marginal_logL(r, [T], phi, N)
    theirs = multivariate_normal.logpdf(r, mean=np.zeros(n), cov=N + T @ phi @ T.T)
    assert abs(mine - theirs) / abs(theirs) < TIGHT


def test_reference_hd_orf_matches_atlas():
    zeta = np.linspace(1e-3, np.pi, 25)
    assert np.max(np.abs(ref.hd_orf(zeta) - np.asarray(atlas_hd(jnp.asarray(zeta))))) < 1e-14


def test_reference_fourier_basis_matches_atlas():
    """Columns must interleave sin/cos per frequency -- the jnp.repeat(phi, 2)
    used throughout ATLAS is only correct for that ordering."""
    toas = np.linspace(4.6e9, 4.6e9 + 3e8, 40)
    freqs = np.array([1e-9, 2e-9, 3e-9])
    assert np.max(np.abs(ref.fourier_basis(toas, freqs)
                         - np.asarray(get_fourier_design_matrix(jnp.asarray(toas),
                                                                jnp.asarray(freqs))))) < 1e-15


def test_frequency_grid_is_harmonic():
    m = H.build()
    f_irn, _, _ = H.freq_grid(m)
    assert np.allclose(f_irn, np.asarray(get_harmonic_frequencies(m.n_irn, m.data.pta_tspan)))
    assert np.allclose(f_irn, np.asarray(m.rn.signal_map["unc"].freqs))


def test_reference_epochs_match_atlas():
    """The independent epoch grouping must reproduce ATLAS's U_pad/U_mask."""
    m = H.build()
    for i, p in enumerate(m.psrs):
        epochs, _ = ref.epochs_from_toas(p.toas, p.backend_flags)
        cov = m.wn.cov_matrices[i]
        atlas = [np.asarray(pad)[np.asarray(msk)]
                 for pad, msk in zip(cov.U_pad, cov.U_mask)]
        assert len(epochs) == len(atlas)
        got = sorted(tuple(a.tolist()) for a in atlas)
        want = sorted(tuple(e.tolist()) for e in epochs)
        assert got == want


# --------------------------------------------------------------------------- #
#  1. ln_likelihood_curn is the free oracle
# --------------------------------------------------------------------------- #

def test_curn_matches_dense_reference():
    """The Woodbury CURN marginal must equal a dense float64 solve exactly.

    Exercises the ECORR Sherman-Morrison solve, logdet_N, the phi assembly and
    the interleaved basis ordering in one comparison.
    """
    m = H.build(model_string="unc+cor->unc", linear_timing=False, orf_name="zero")
    red = m.red_params(irn_overrides={1: (-14.7, 4.0)})
    atlas = float(m.rn.ln_likelihood_curn(m.helpers, red))

    T, N, Nfull, r = H.dense_bundle(m)
    irn, gwb = H.psd_pieces(m, red)
    phi = ref.build_phi(m.npsr, 2 * m.n_irn, 0, [0] * m.npsr, irn, gwb, orf=None)
    expected = ref.marginal_logL(r, T, phi, Nfull) + ref.atlas_offset(r.size)
    assert abs(atlas - expected) / abs(expected) < TIGHT


def test_ecorr_solve_matches_dense():
    """left^T N^-1 right and logdet N, against a dense diag + U J U^T inverse."""
    m = H.build()
    N = H.noise_blocks(m)
    rng = np.random.default_rng(4)
    for i, p in enumerate(m.psrs):
        cov = m.wn.cov_matrices[i]
        wn = m.wn_vec[i * cov.n_params:(i + 1) * cov.n_params]
        helpers = cov.get_nvec_jvec(wn)
        left = rng.normal(size=(p.ntoa, 3))
        right = rng.normal(size=(p.ntoa, 2))
        got, logdet = cov.solve_with_logdet(helpers, jnp.asarray(left), jnp.asarray(right))
        want = left.T @ np.linalg.solve(N[i], right)
        assert np.max(np.abs(np.asarray(got) - want)) / np.max(np.abs(want)) < 1e-10
        assert abs(float(logdet) - np.linalg.slogdet(N[i])[1]) / abs(
            np.linalg.slogdet(N[i])[1]) < TIGHT


# --------------------------------------------------------------------------- #
#  2. The reparameterised densities
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("orf_name", ["zero", "hd"])
def test_reparam_density_matches_dense_conditional(orf_name):
    """`lnposterior_reparam(z)` must differ from the dense joint by a constant.

    A difference-in-z identity: the Jacobian is independent of z, so this is
    exact whether or not the transform whitens the target -- unlike a
    z-invariance test, which only holds in the ORF-free limit (see
    `test_z_invariance_requires_zero_orf`).
    """
    m = H.build(orf_name=orf_name)
    red = m.red_params(irn_overrides={1: (-14.7, 4.0)})
    T, _, Nfull, r = H.dense_bundle(m)
    phi = H.global_phi(m, red)

    rng = np.random.default_rng(7)
    deltas = []
    for z in rng.normal(size=(6, m.npsr, m.rn.nmodes)):
        lp, coeff = m.rn.lnposterior_reparam(m.helpers, red, jnp.asarray(z))
        deltas.append(float(lp) - ref.conditional_logL(
            r, T, np.asarray(coeff).ravel(), phi, Nfull))
    deltas = np.array(deltas)
    assert np.ptp(deltas) / abs(deltas.mean()) < TIGHT


def test_reparam_absolute_constant():
    """That constant is the Jacobian plus exactly the terms ATLAS drops."""
    m = H.build(orf_name="zero")
    red = m.red_params(irn_overrides={1: (-14.7, 4.0)})
    T, N, Nfull, r = H.dense_bundle(m)
    phi = H.global_phi(m, red)

    z = np.random.default_rng(3).normal(size=(m.npsr, m.rn.nmodes))
    lp, coeff = m.rn.lnposterior_reparam(m.helpers, red, jnp.asarray(z))
    half_logdet = ref.precision_logdet(T, N, H.phiinv_diag(m, red))
    joint = ref.conditional_logL(r, T, np.asarray(coeff).ravel(), phi, Nfull)

    n_a = m.npsr * m.rn.nmodes
    predicted = (0.5 * r.size * np.log(2 * np.pi)
                 + 0.5 * n_a * np.log(2 * np.pi)
                 + 0.5 * sum(m.tm_widths) * np.log(1e40))
    measured = float(lp) - (joint - half_logdet)
    assert abs(measured - predicted) / predicted < 1e-9


def test_z_invariance_requires_zero_orf():
    """Exact whitening holds only when the ORF vanishes.

    `Sigma_inv` is built from the diagonal of `phiinvs` while the density's
    prior uses the full correlated matrix -- a deliberate CURN preconditioner
    (`W_inv_curn` in `partial_marg`).  Documented here so nobody "fixes" the
    correlated case by asserting invariance.
    """
    red_kw = dict(irn_overrides={1: (-14.7, 4.0)})
    rng = np.random.default_rng(11)

    m0 = H.build(orf_name="zero")
    v0 = np.array([float(m0.rn.lnposterior_reparam(
        m0.helpers, m0.red_params(**red_kw), jnp.asarray(z))[0]) + 0.5 * np.sum(z ** 2)
        for z in rng.normal(size=(12, m0.npsr, m0.rn.nmodes))])
    assert np.ptp(v0) / abs(v0.mean()) < TIGHT

    m1 = H.build(orf_name="hd")
    v1 = np.array([float(m1.rn.lnposterior_reparam(
        m1.helpers, m1.red_params(**red_kw), jnp.asarray(z))[0]) + 0.5 * np.sum(z ** 2)
        for z in rng.normal(size=(12, m1.npsr, m1.rn.nmodes))])
    assert np.ptp(v1) / abs(v1.mean()) > 1e-6


# --------------------------------------------------------------------------- #
#  3. The two likelihoods must agree -- the regression that caught the npsr bug
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("npsr", [2, 3, 4])
def test_partial_marg_agrees_with_reparam(npsr):
    """With the ORF zero both transforms whiten exactly, so both

        log_density(z) + 0.5 * sum(z**2)

    equal ln p(r) plus the constants each drops -- and those constants are the
    same.  Regression for the `npsr`-scaling of the P-block evidence term and
    the `npsr**2`-scaling of `rNr` in `__partial_marg_lnposterior`.
    """
    m = H.build(npsr=npsr, orf_name="zero")
    red = m.red_params()
    rng = np.random.default_rng(2)
    zr = rng.normal(size=(npsr, m.rn.nmodes))
    zp = rng.normal(size=(npsr, 2 * m.n_gwb))
    v_r = float(m.rn.lnposterior_reparam(m.helpers, red, jnp.asarray(zr))[0]) \
        + 0.5 * np.sum(zr ** 2)
    v_p = float(m.rn.partial_marg_lnposterior(m.helpers, red, jnp.asarray(zp))[0]) \
        + 0.5 * np.sum(zp ** 2)
    assert abs(v_r - v_p) / abs(v_r) < LOOSE


@pytest.mark.parametrize("irn_log10_A", [-16.0, -15.0, -14.0])
def test_partial_marg_agreement_is_parameter_independent(irn_log10_A):
    """The old bug's signature was a gap that drifted with the IRN amplitude,
    because `Sigma_P` depends on the red-noise parameters."""
    m = H.build(orf_name="zero")
    red = m.red_params(irn=(irn_log10_A, 3.5))
    rng = np.random.default_rng(5)
    zr = rng.normal(size=(m.npsr, m.rn.nmodes))
    zp = rng.normal(size=(m.npsr, 2 * m.n_gwb))
    v_r = float(m.rn.lnposterior_reparam(m.helpers, red, jnp.asarray(zr))[0]) \
        + 0.5 * np.sum(zr ** 2)
    v_p = float(m.rn.partial_marg_lnposterior(m.helpers, red, jnp.asarray(zp))[0]) \
        + 0.5 * np.sum(zp ** 2)
    assert abs(v_r - v_p) / abs(v_r) < LOOSE


def test_partial_marg_rNr_enters_once():
    """`rNr` is the array-wide total; scaling it must not scale the answer by npsr."""
    m = H.build(npsr=3, orf_name="zero")
    red = m.red_params()
    TNT, TNr, rNr, logdet_N = m.helpers
    z = np.zeros((m.npsr, 2 * m.n_gwb))
    base = float(m.rn.partial_marg_lnposterior(m.helpers, red, jnp.asarray(z))[0])
    bumped = float(m.rn.partial_marg_lnposterior(
        (TNT, TNr, rNr + 2.0, logdet_N), red, jnp.asarray(z))[0])
    assert abs((base - bumped) - 1.0) < 1e-8


# --------------------------------------------------------------------------- #
#  4. Gradients
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fn_name", ["lnposterior_reparam", "partial_marg_lnposterior"])
def test_grad_matches_finite_difference(fn_name):
    m = H.build(orf_name="hd")
    red = m.red_params(irn_overrides={1: (-14.7, 4.0)})
    nz = m.rn.nmodes if fn_name == "lnposterior_reparam" else 2 * m.n_gwb
    z = jnp.asarray(np.random.default_rng(8).normal(size=(m.npsr, nz)) * 0.1)
    fn = getattr(m.rn, fn_name)

    def f(rp):
        return fn(m.helpers, rp, z)[0]

    g = np.asarray(jax.grad(f)(red))
    # eps=1e-6 puts central-difference roundoff (~eps_machine * |f| / eps, and
    # |f| ~ 5e3 here) at ~5e-7, i.e. at the tolerance itself. 1e-4 drops it two
    # orders while truncation error stays far below.
    eps = 1e-4
    for i in range(red.size):
        step = jnp.zeros_like(red).at[i].set(eps)
        fd = (float(f(red + step)) - float(f(red - step))) / (2 * eps)
        scale = max(abs(fd), abs(g[i]), 1.0)
        assert abs(fd - g[i]) / scale < GRAD_TOL, f"param {i}: grad {g[i]} vs fd {fd}"


# --------------------------------------------------------------------------- #
#  5. Known defects, pinned as xfail so a later stage sees them go green
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(reason="B3 (partial): signal_comb_idxs['timing'] and ['unc'] are "
                          "indexed unguarded, so both reparameterised likelihoods "
                          "require an 'ltm' prefix", raises=KeyError, strict=True)
def test_reparam_without_timing_block():
    m = H.build(model_string="unc+cor->unc", linear_timing=False)
    z = jnp.zeros((m.npsr, m.rn.nmodes))
    m.rn.lnposterior_reparam(m.helpers, m.red_params(), z)


@pytest.mark.xfail(reason="build_basis emits cor=slice(18,26) into a 20-column basis "
                          "for a separate (non-overlapping) cor block",
                   raises=ValueError, strict=True)
def test_separate_cor_block():
    m = H.build(model_string="ltm|unc;cor")
    z = jnp.zeros((m.npsr, m.rn.nmodes))
    m.rn.lnposterior_reparam(m.helpers, m.red_params(), z)


@pytest.mark.xfail(reason="B1: PTA_Data stores self.dm_bins, make_red_noise reads "
                          "self.data.num_dm_bins", raises=AttributeError, strict=True)
def test_dm_model_string_builds():
    H.build(model_string="ltm|unc+cor->unc;dm")


# --------------------------------------------------------------------------- #
#  6. The marginalised-timing path and the column layout
# --------------------------------------------------------------------------- #

def test_marg_timing_solve_matches_dense():
    """`_solve_marg` must equal a dense projection orthogonal to the timing
    subspace, and its logdet must carry the `Mprior` flat-prior constant.

    This is the other half of the funnel: with `marg_timing=True` the timing
    model is absorbed into N rather than appearing as columns of T.
    """
    m = H.build(model_string="unc+cor->unc", linear_timing=False, marg_timing=True)
    N = H.noise_blocks(m)
    rng = np.random.default_rng(6)
    for i, p in enumerate(m.psrs):
        cov = m.wn.cov_matrices[i]
        wn = m.wn_vec[i * cov.n_params:(i + 1) * cov.n_params]
        helpers = cov.get_nvec_jvec(wn)
        left = rng.normal(size=(p.ntoa, 3))
        right = rng.normal(size=(p.ntoa, 2))
        got, logdet = cov.solve_with_logdet(helpers, jnp.asarray(left), jnp.asarray(right))

        M = H.svd_basis(p.Mmat)
        Ninv_M = np.linalg.solve(N[i], M)
        MNM = M.T @ Ninv_M
        want = left.T @ np.linalg.solve(N[i], right) \
            - (left.T @ Ninv_M) @ np.linalg.solve(MNM, M.T @ np.linalg.solve(N[i], right))
        assert np.max(np.abs(np.asarray(got) - want)) / np.max(np.abs(want)) < 1e-8

        want_logdet = (np.linalg.slogdet(N[i])[1] + np.linalg.slogdet(MNM)[1]
                       + M.shape[1] * np.log(1e40))
        assert abs(float(logdet) - want_logdet) / abs(want_logdet) < 1e-10


def test_basis_column_layout():
    """`cor` must be nested at the head of `unc`, not adjacent to it.

    The GWB is modelled in the lowest bins only and shares the IRN's columns,
    so `partial_marg`'s P and G blocks overlap by construction.  Anything that
    assumes they are disjoint is wrong.
    """
    m = H.build()
    idx = m.rn.signal_comb_idxs
    assert idx["timing"] == slice(0, m.n_tm)
    assert idx["unc"] == slice(m.n_tm, m.n_tm + 2 * m.n_irn)
    assert idx["cor"] == slice(m.n_tm, m.n_tm + 2 * m.n_gwb)
    assert idx["cor"].start == idx["unc"].start
    assert idx["cor"].stop <= idx["unc"].stop
    assert m.rn.nmodes == m.n_tm + 2 * m.n_irn

    # Padding: every pulsar's real timing columns come first, padded ones after,
    # and the padded columns of T are identically zero.
    T = np.asarray(m.rn.get_Fmat_concat)
    start = 0
    for i, p in enumerate(m.psrs):
        block = T[start:start + p.ntoa]
        w = m.tm_widths[i]
        assert np.all(block[:, w:m.n_tm] == 0.0)
        assert np.asarray(m.rn._pad_mask)[i].sum() == m.n_tm - w
        start += p.ntoa


# --------------------------------------------------------------------------- #
#  7. The same identities on real data
# --------------------------------------------------------------------------- #

# MDC1 has one TOA per observing epoch, so ECORR is exactly degenerate with
# EQUAD there and ATLAS refuses it by design -- see the include_ecorr guard in
# SinglePulsarWhiteCov. NG15 is multi-frequency and does have real epochs.
REAL = [pytest.param(f, ec, id=f, marks=pytest.mark.skipif(
    not H.fixture_available(f), reason=f"{f}.npz not generated"))
    for f, ec in (("mdc1_5", False), ("ng15_3", True))]


@pytest.mark.parametrize("fixture,ecorr", REAL)
def test_curn_matches_dense_reference_real_data(fixture, ecorr):
    """Real backend structure, real ECORR epochs, real design matrices.

    ng15_3 in particular has timing blocks of 55/40/42 columns, so the padding
    machinery is genuinely exercised rather than nominally.
    """
    m = H.build(npsr=3, model_string="unc+cor->unc", linear_timing=False,
                orf_name="zero", fixture=fixture, include_ecorr=ecorr)
    red = m.red_params()
    atlas = float(m.rn.ln_likelihood_curn(m.helpers, red))
    T, N, Nfull, r = H.dense_bundle(m)
    irn, gwb = H.psd_pieces(m, red)
    phi = ref.build_phi(m.npsr, 2 * m.n_irn, 0, [0] * m.npsr, irn, gwb, orf=None)
    expected = ref.marginal_logL(r, T, phi, Nfull) + ref.atlas_offset(r.size)
    assert abs(atlas - expected) / abs(expected) < LOOSE


@pytest.mark.parametrize("fixture,ecorr", REAL)
def test_partial_marg_agrees_with_reparam_real_data(fixture, ecorr):
    m = H.build(npsr=3, orf_name="zero", fixture=fixture, include_ecorr=ecorr)
    red = m.red_params()
    rng = np.random.default_rng(21)
    zr = rng.normal(size=(m.npsr, m.rn.nmodes))
    zp = rng.normal(size=(m.npsr, 2 * m.n_gwb))
    v_r = float(m.rn.lnposterior_reparam(m.helpers, red, jnp.asarray(zr))[0]) \
        + 0.5 * np.sum(zr ** 2)
    v_p = float(m.rn.partial_marg_lnposterior(m.helpers, red, jnp.asarray(zp))[0]) \
        + 0.5 * np.sum(zp ** 2)
    assert abs(v_r - v_p) / abs(v_r) < LOOSE


@pytest.mark.parametrize("fixture,ecorr", REAL)
def test_reparam_density_matches_dense_conditional_real_data(fixture, ecorr):
    m = H.build(npsr=3, orf_name="hd", fixture=fixture, include_ecorr=ecorr)
    red = m.red_params()
    T, _, Nfull, r = H.dense_bundle(m)
    phi = H.global_phi(m, red)
    rng = np.random.default_rng(22)
    deltas = []
    for z in rng.normal(size=(4, m.npsr, m.rn.nmodes)):
        lp, coeff = m.rn.lnposterior_reparam(m.helpers, red, jnp.asarray(z))
        deltas.append(float(lp) - ref.conditional_logL(
            r, T, np.asarray(coeff).ravel(), phi, Nfull))
    deltas = np.array(deltas)
    assert np.ptp(deltas) / abs(deltas.mean()) < LOOSE


@pytest.mark.parametrize("fixture,ecorr", REAL)
def test_fixture_provenance_recorded(fixture, ecorr):
    """A fixture without provenance is not reproducible."""
    from .fixtures.pulsar import load_fixture
    _, prov = load_fixture(H.DATA_DIR / f"{fixture}.npz")
    assert prov.get("source") and prov.get("atlas_sha")
