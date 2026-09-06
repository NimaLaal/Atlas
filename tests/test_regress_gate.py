"""The regression gate must fail closed.

`tools/regress.py` decides whether two revisions agree. A gate that reports
"pass" when it cannot actually compare two arrays is worse than no gate, so the
comparison and the pass/fail predicate are tested directly here rather than
only exercised through a full two-worktree run.

The specific hole this pins: `d.max()` and `rel.max()` are non-finite when
either array contains NaN or Inf, and a non-finite value compares False against
any tolerance -- so `rel.max() > tol` reads as a pass and a broken likelihood
slips through.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("_regress", ROOT / "tools" / "regress.py")
regress = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(regress)

TOL = 1e-9


def _pair(tmp_path, a, b):
    pa, pb = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(pa, **a)
    np.savez(pb, **b)
    return regress.compare(pa, pb)


def _verdict(rows):
    return all(regress._row_ok(r, TOL) for r in rows)


def test_identical_arrays_pass(tmp_path):
    x = np.linspace(-3, 3, 20)
    rows = _pair(tmp_path, {"g": x}, {"g": x.copy()})
    assert _verdict(rows)
    assert rows[0][1] == 0.0 and rows[0][2] == 0.0


def test_small_difference_passes(tmp_path):
    x = np.linspace(1, 2, 20)
    rows = _pair(tmp_path, {"g": x}, {"g": x * (1 + 1e-14)})
    assert _verdict(rows)


def test_real_difference_fails(tmp_path):
    x = np.linspace(1, 2, 20)
    rows = _pair(tmp_path, {"g": x}, {"g": x * 1.01})
    assert not _verdict(rows)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["a", "b", "both"])
def test_non_finite_fails(tmp_path, bad, side):
    """The hole. Without an explicit check these all reported a pass."""
    x = np.linspace(1, 2, 20)
    a, b = x.copy(), x.copy()
    if side in ("a", "both"):
        a[3] = bad
    if side in ("b", "both"):
        b[3] = bad
    rows = _pair(tmp_path, {"g": a}, {"g": b})
    assert not _verdict(rows), f"{bad} on {side} was treated as a pass"
    assert "NON-FINITE" in rows[0][3]


def test_shape_change_fails(tmp_path):
    rows = _pair(tmp_path, {"g": np.zeros(5)}, {"g": np.zeros(6)})
    assert not _verdict(rows)


def test_missing_array_fails(tmp_path):
    rows = _pair(tmp_path, {"g": np.zeros(5)}, {"h": np.zeros(5)})
    assert not _verdict(rows)


def test_zero_arrays_do_not_divide_by_zero(tmp_path):
    """Both sides exactly zero is agreement, not 0/0."""
    rows = _pair(tmp_path, {"g": np.zeros(5)}, {"g": np.zeros(5)})
    assert _verdict(rows)
    assert rows[0][2] == 0.0


def test_descriptive_arrays_must_match_exactly(tmp_path):
    """`param_names` and friends are compared for equality, not tolerance."""
    rows = _pair(tmp_path,
                 {"param_names": np.array(["a", "b"])},
                 {"param_names": np.array(["a", "c"])})
    assert not _verdict(rows)
    rows = _pair(tmp_path,
                 {"param_names": np.array(["a", "b"])},
                 {"param_names": np.array(["a", "b"])})
    assert _verdict(rows)


def test_unknown_note_fails_closed(tmp_path):
    """A note the predicate has never seen must not read as a pass."""
    assert not regress._row_ok(("g", 0.0, 0.0, "something new"), TOL)
