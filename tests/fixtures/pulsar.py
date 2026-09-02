"""A minimal, dependency-free stand-in for an ``enterprise`` ``Pulsar``.

ATLAS's likelihood core reads a small, fixed set of attributes off a pulsar
object.  This module makes that contract explicit so the test suite can run
without ``enterprise``, ``libstempo``, ``PINT``, ``jug`` or a ``$TEMPO2``
runtime -- none of which are importable on a plain CI runner, and all of
which ``ATLAS.pulsar`` requires at import time.

The attributes actually read, verified against the tree at ``99d5997``:

=================  ==========================================  =================
attribute          read by                                     shape / units
=================  ==========================================  =================
``name``           ``PTA_Data``, ``SinglePulsarWhiteCov``      str
``toas``           ``PTA_Data``, ``_get_psr_WN_helpers``       (ntoa,) seconds
``residuals``      ``PTA_Data.raw_residuals``                  (ntoa,) seconds
``toaerrs``        ``SinglePulsarWhiteCov``                    (ntoa,) seconds
``freqs``          ``PTA_Data.ref_over_radio_freqs``           (ntoa,) MHz
``pos``            ``PTA_Data.psr_pos``                        (3,) unit vector
``Mmat``           ``PTA_Data.Mmat``, ``_solve_marg``          (ntoa, npar)
``backend_flags``  ``_get_psr_WN_helpers``                     (ntoa,) str
=================  ==========================================  =================

``pdist`` is carried through because the deterministic/CW path (Stage 4)
wants it; nothing on the stochastic path reads it today.

Everything is plain NumPy.  ``_get_psr_WN_helpers`` fancy-indexes ``toas``
with an integer array and calls ``np.unique`` on ``backend_flags``, so those
two in particular must be real NumPy arrays rather than JAX ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["FixturePulsar", "save_fixture", "load_fixture"]


@dataclass
class FixturePulsar:
    """Duck-typed pulsar carrying exactly what ATLAS reads."""

    name: str
    toas: np.ndarray          # (ntoa,) seconds
    residuals: np.ndarray     # (ntoa,) seconds
    toaerrs: np.ndarray       # (ntoa,) seconds
    freqs: np.ndarray         # (ntoa,) MHz (radio frequency)
    pos: np.ndarray           # (3,) unit vector
    Mmat: np.ndarray          # (ntoa, npar) timing design matrix
    backend_flags: np.ndarray # (ntoa,) str
    pdist: tuple = (1.0, 0.2) # kpc, (mean, std) -- unused on the stochastic path

    def __post_init__(self):
        self.toas = np.asarray(self.toas, dtype=np.float64)
        self.residuals = np.asarray(self.residuals, dtype=np.float64)
        self.toaerrs = np.asarray(self.toaerrs, dtype=np.float64)
        self.freqs = np.asarray(self.freqs, dtype=np.float64)
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.Mmat = np.asarray(self.Mmat, dtype=np.float64)
        self.backend_flags = np.asarray(self.backend_flags, dtype="<U32")

        n = self.toas.size
        for attr in ("residuals", "toaerrs", "freqs", "backend_flags"):
            got = getattr(self, attr).shape
            if got != (n,):
                raise ValueError(f"{self.name}: {attr} has shape {got}, expected ({n},)")
        if self.Mmat.shape[0] != n:
            raise ValueError(
                f"{self.name}: Mmat has {self.Mmat.shape[0]} rows, expected {n}"
            )
        if self.pos.shape != (3,):
            raise ValueError(f"{self.name}: pos has shape {self.pos.shape}, expected (3,)")

    # ``ATLAS.sim`` writes through this alias; enterprise exposes both.
    @property
    def _residuals(self):
        return self.residuals

    @property
    def ntoa(self) -> int:
        return self.toas.size

    def __repr__(self) -> str:
        return (
            f"FixturePulsar({self.name!r}, ntoa={self.ntoa}, "
            f"ncol(M)={self.Mmat.shape[1]}, "
            f"backends={sorted(set(self.backend_flags.tolist()))}, "
            f"Tspan={(self.toas.max() - self.toas.min()) / 31557600:.2f} yr)"
        )


_ARRAYS = ("toas", "residuals", "toaerrs", "freqs", "pos", "Mmat", "backend_flags")


def save_fixture(psrs, path, provenance: dict | None = None) -> Path:
    """Write a list of ``FixturePulsar`` to a single ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "__names__": np.array([p.name for p in psrs], dtype="<U64"),
        "__provenance__": np.array(json.dumps(provenance or {})),
    }
    for i, p in enumerate(psrs):
        for a in _ARRAYS:
            payload[f"psr{i}_{a}"] = getattr(p, a)
        payload[f"psr{i}_pdist"] = np.asarray(p.pdist, dtype=np.float64)
    np.savez_compressed(path, **payload)
    return path


def load_fixture(path):
    """Read a ``.npz`` written by :func:`save_fixture`.

    Returns ``(psrs, provenance)``.
    """
    with np.load(Path(path), allow_pickle=False) as z:
        names = [str(n) for n in z["__names__"]]
        provenance = json.loads(str(z["__provenance__"]))
        psrs = [
            FixturePulsar(
                name=names[i],
                **{a: z[f"psr{i}_{a}"] for a in _ARRAYS},
                pdist=tuple(z[f"psr{i}_pdist"].tolist()),
            )
            for i in range(len(names))
        ]
    return psrs, provenance
