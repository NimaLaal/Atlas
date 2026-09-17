"""ECORR identifiability is detected from the data, not assumed.

Every MDC1 epoch holds a single TOA, which makes ECORR exactly degenerate with
EQUAD; SinglePulsarWhiteCov refuses to sample it.  PTA_Data detects that at load
time and WhiteCov honours the verdict, so make_white_noise() works on such a
dataset without the caller knowing about epochs at all.
"""

from __future__ import annotations

import pytest

from . import harness as H
from . import reference as ref
from ATLAS.data import PTA_Data
from ATLAS.model_builder import ModelBuilder
from ATLAS.nMatrix.base import WhiteCov


def _data(fixture, npsr=None):
    """A minimal PTA_Data, built directly.

    Not through H.build: that is lru_cached and always passes include_ecorr
    explicitly, so it cannot exercise the auto-detection path.
    """
    return PTA_Data(H.load_psrs(fixture, npsr),
                    num_gwb_bins=4,
                    num_irn_bins=6,
                    linear_timing=True,
                    marg_timing=False,
                    diag_white_cov=False)


@pytest.mark.parametrize("fixture,expected", [("mdc1_5", False), ("ng15_3", True)])
def test_pta_data_detects_ecorr_identifiability(fixture, expected):
    if not H.fixture_available(fixture):
        pytest.skip(f"no {fixture} fixture")
    data = _data(fixture)
    assert data.include_ecorr is expected
    # Cross-check against the independent epoch reimplementation.
    assert data.include_ecorr == any(
        len(ref.epochs_from_toas(p.toas, p.backend_flags)[0]) > 0
        for p in data.psrs)


def test_make_white_noise_on_mdc1_does_not_raise():
    """The regression: this used to raise ValueError from the ECORR guard."""
    if not H.fixture_available("mdc1_5"):
        pytest.skip("no mdc1_5 fixture")
    wn = ModelBuilder(data=_data("mdc1_5")).make_white_noise(stabilize_TNT=False)
    assert wn.include_ecorr is False
    assert wn.cov_matrices[0].n_params_per_backend == 2
    assert not any('ecorr' in n for n in wn.get_param_names())


def test_ng15_keeps_ecorr():
    """Auto-detection must not quietly drop ECORR from data that supports it."""
    if not H.fixture_available("ng15_3"):
        pytest.skip("no ng15_3 fixture")
    wn = ModelBuilder(data=_data("ng15_3", 2)).make_white_noise(stabilize_TNT=False)
    assert wn.include_ecorr is True
    assert wn.cov_matrices[0].n_params_per_backend == 3
    assert any('ecorr' in n for n in wn.get_param_names())


def test_explicit_include_ecorr_overrides_detection():
    if not H.fixture_available("ng15_3"):
        pytest.skip("no ng15_3 fixture")
    wn = WhiteCov(data=_data("ng15_3", 2), include_ecorr=False)
    assert wn.include_ecorr is False
    assert wn.cov_matrices[0].n_params_per_backend == 2
