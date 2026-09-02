"""Committed artifacts must not carry machine identifiers.

The benchmark and noise-floor snapshots are generated on a developer's machine
and committed, so anything the generator writes into them ends up in the
repository. A fully qualified internal hostname is an infrastructure detail
that does not belong there, and it also makes the files churn on every
regeneration elsewhere.

`bench.harness.host_label` hashes the hostname instead, so results still group
by machine without naming it. This test is the guard on that.

Scope: only files this repository generates. `datasets/` is excluded -- those
par/tim files arrive with a `# Host:` provenance line written by whatever
produced them, and rewriting upstream data to satisfy a lint is the wrong
trade.
"""
from __future__ import annotations

import getpass
import platform
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

GENERATED = [
    "bench/results/*.json",
    "tools/*.json",
    "tests/golden/*.json",
]
FIXTURES = "tests/fixtures/data/*.npz"

# Deliberately narrow, to avoid matching version strings like "0.10.1".
FQDN = re.compile(r"\b[\w-]+(?:\.[\w-]+)+\.(?:edu|com|org|net|gov|mil|local|internal|lan)\b",
                  re.IGNORECASE)
HOME = re.compile(r"(?:/home/|/Users/|C:\\\\Users\\\\)[\w.-]+")


def _offenders(text):
    found = []
    node = platform.node()
    if "." in node and node in text:
        found.append(f"hostname {node!r}")
    found += [f"FQDN {m!r}" for m in set(FQDN.findall(text))]
    found += [f"home path {m!r}" for m in set(HOME.findall(text))]
    try:
        user = getpass.getuser()
    except Exception:
        user = None
    if user and len(user) > 3 and re.search(rf"\b{re.escape(user)}\b", text):
        found.append(f"username {user!r}")
    return found


def _generated_files():
    return sorted(f for pat in GENERATED for f in ROOT.glob(pat))


def test_there_are_artifacts_to_check():
    """A guard that silently checks nothing is not a guard."""
    assert _generated_files(), "no generated artifacts found; the globs are stale"


@pytest.mark.parametrize("path", _generated_files(), ids=lambda p: p.name)
def test_generated_artifact_has_no_machine_identifier(path):
    found = _offenders(path.read_text(errors="replace"))
    assert not found, f"{path.relative_to(ROOT)} contains: {'; '.join(found)}"


@pytest.mark.parametrize("path", _generated_files(), ids=lambda p: p.name)
def test_generated_filename_has_no_machine_identifier(path):
    found = _offenders(path.name)
    assert not found, f"filename {path.name!r} contains: {'; '.join(found)}"


@pytest.mark.parametrize(
    "path", sorted(ROOT.glob(FIXTURES)), ids=lambda p: p.name)
def test_fixture_provenance_has_no_machine_identifier(path):
    """Fixture provenance records where the data came from; it must do that with
    repository-relative paths, not absolute ones from someone's home directory."""
    with np.load(path, allow_pickle=False) as z:
        text = str(z["__provenance__"])
    found = _offenders(text)
    assert not found, f"{path.name} provenance contains: {'; '.join(found)}"
