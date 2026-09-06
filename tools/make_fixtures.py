#!/usr/bin/env python
"""Regenerate the committed test fixtures from real data.

Run this in an environment that can actually load pulsars -- it needs PINT or
tempo2 and, for the MDC1 pickle, ``enterprise``.  The *test suite* needs none
of that: it reads the ``.npz`` files this writes.

    TEMPO2=... python tools/make_fixtures.py

The synthetic fixture (``tests/fixtures/synthetic.py``) is generated from a
seed at test time and is deliberately not committed as a binary.
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tests.fixtures.pulsar import FixturePulsar, save_fixture

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "data"

# The three smallest ideal_NG15 pulsars, kept whole: decimating would break the
# ECORR epoch structure, and whole-array dense reference solves at ~1900 TOAs
# are still sub-second.
NG15_PULSARS = ["J0557+1551", "J0605+3757", "J1012-4235"]
MDC1_N = 5
# The full MDC1 array, for the golden run. Small enough to commit (~4,700 TOAs
# across 36 pulsars over 4.94 yr), which makes the golden fit reproducible
# without a tempo2 runtime.
MDC1_ALL = 36


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _backend_flags(psr):
    try:
        return np.asarray(list(psr.backend_flags), dtype="<U32")
    except (ValueError, AttributeError):
        return np.array([""] * len(psr.toas), dtype="<U32")


def from_enterprise(psr):
    return FixturePulsar(
        name=str(psr.name),
        toas=np.asarray(psr.toas, dtype=np.float64),
        residuals=np.asarray(psr.residuals, dtype=np.float64),
        toaerrs=np.asarray(psr.toaerrs, dtype=np.float64),
        freqs=np.asarray(psr.freqs, dtype=np.float64),
        pos=np.asarray(psr.pos, dtype=np.float64),
        Mmat=np.asarray(psr.Mmat, dtype=np.float64),
        backend_flags=_backend_flags(psr),
        pdist=tuple(np.asarray(getattr(psr, "pdist", (1.0, 0.2)), dtype=np.float64).tolist()),
    )


def make_mdc1():
    src = ROOT / "notebooks" / "psrs_mdc1_t2_36psr.pkl"
    if not src.exists():
        print(f"  skip mdc1: {src} not present")
        return
    with open(src, "rb") as fin:
        allpsrs = pickle.load(fin)

    small = sorted(allpsrs, key=lambda p: len(p.toas))[:MDC1_N]
    fx = [from_enterprise(p) for p in small]
    path = save_fixture(fx, OUT / "mdc1_5.npz", provenance=dict(
        source=str(src.relative_to(ROOT)),
        description="IPTA MDC1 Open-1, loaded via tempo2 (TCB-correct); 5 smallest pulsars",
        atlas_sha=_git_sha(),
    ))
    print(f"  mdc1_5.npz   {path.stat().st_size/1e3:7.1f} kB  "
          + ", ".join(f"{p.name}({p.ntoa})" for p in fx))

    fx = [from_enterprise(p) for p in sorted(allpsrs, key=lambda p: p.name)]
    path = save_fixture(fx, OUT / "mdc1_36.npz", provenance=dict(
        source=str(src.relative_to(ROOT)),
        description=("IPTA MDC1 Open-1, all 36 pulsars, loaded via tempo2. "
                     "Must be loaded with tempo2, not PINT: the par files carry "
                     "EPHVER 5 and no UNITS line, so PINT reads TCB as TDB and "
                     "the residuals become a uniform hash over one pulse period. "
                     "This is the fixture the golden run uses."),
        injected_gwb_log10_A=-13.301, injected_gwb_gamma=13 / 3,
        atlas_sha=_git_sha(),
    ))
    print(f"  mdc1_36.npz  {path.stat().st_size/1e3:7.1f} kB  "
          f"{len(fx)} pulsars, {sum(p.ntoa for p in fx)} TOAs")


def make_ng15():
    par = ROOT / "datasets" / "ideal_NG15" / "par"
    tim = ROOT / "datasets" / "ideal_NG15" / "tim"
    pars, tims = [], []
    for name in NG15_PULSARS:
        p = sorted(par.glob(f"{name}*.par"))
        t = sorted(tim.glob(f"{name}*.tim"))
        if not p or not t:
            print(f"  skip ng15_3: no par/tim for {name}")
            return
        pars.append(str(p[0]))
        tims.append(str(t[0]))
    from ATLAS.pulsar import load_pulsars          # needs PINT / tempo2
    psrs = load_pulsars(pars, tims, timing_package="pint")
    fx = [from_enterprise(p) for p in psrs]
    path = save_fixture(fx, OUT / "ng15_3.npz", provenance=dict(
        source="datasets/ideal_NG15",
        description="NANOGrav 15 yr 'ideal' set via PINT; 3 smallest pulsars, undecimated",
        atlas_sha=_git_sha(),
    ))
    print(f"  ng15_3.npz   {path.stat().st_size/1e3:7.1f} kB  "
          + ", ".join(f"{p.name}({p.ntoa}, {p.Mmat.shape[1]} tm cols)" for p in fx))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["mdc1", "ng15"], default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("Writing fixtures:")
    if args.only in (None, "mdc1"):
        make_mdc1()
    if args.only in (None, "ng15"):
        make_ng15()
