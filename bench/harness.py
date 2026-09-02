"""Measurement protocol for ATLAS benchmarks.

Three rules, each of which cost real time to learn:

1. **Repeat the fast side, not the slow one.** Contention distorts a 2.5 s
   measurement roughly six-fold and a 90 s one by under 1%. So the API takes a
   *target total duration* and works out the repeat count itself, rather than
   taking a repeat count and letting you accidentally measure the slow variant
   once. :func:`timeit` does this.

2. **Record compiled memory, not just wall clock.** ``memory_analysis()`` gives
   generated-code / argument / temp bytes, which are load-independent and
   therefore quotable without a caveat, unlike wall clock. They are also what
   actually decides whether a configuration will load on a given card.

3. **Record loadavg alongside every timing.** A number without it cannot be
   compared against a number taken on a different day.

Never quote a compile-time host-RAM high-water mark as a sampling footprint:
after the helper build, consecutive gradient calls add 0 MB. The two scale
differently and conflating them overstates by roughly 2x.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax

__all__ = ["Timing", "timeit", "compiled_memory", "machine", "write_results"]


@dataclass
class Timing:
    label: str
    median_ms: float
    min_ms: float
    max_ms: float
    repeats: int
    rounds: int
    loadavg: float
    note: str = ""

    def __str__(self):
        spread = (self.max_ms - self.min_ms) / self.median_ms if self.median_ms else 0.0
        return (f"{self.label:44s} {self.median_ms:10.3f} ms  "
                f"(+-{spread * 100:4.1f}%, n={self.repeats}x{self.rounds}, "
                f"load {self.loadavg:.2f}){'  ' + self.note if self.note else ''}")


def timeit(fn, *args, label="", target_s=2.0, rounds=3, note="") -> Timing:
    """Time ``fn(*args)``, choosing the repeat count from a target duration.

    One warm-up call is made and blocked on, so compilation is excluded. The
    repeat count is then set so that each of ``rounds`` batches takes about
    ``target_s`` -- which is the mechanical form of "repeat the fast side".
    """
    jax.block_until_ready(fn(*args))

    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    single = time.perf_counter() - t0
    repeats = max(1, int(target_s / max(single, 1e-6)))

    per_round = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(repeats):
            r = fn(*args)
        jax.block_until_ready(r)
        per_round.append((time.perf_counter() - t0) / repeats * 1e3)

    return Timing(label=label, median_ms=statistics.median(per_round),
                  min_ms=min(per_round), max_ms=max(per_round),
                  repeats=repeats, rounds=rounds,
                  loadavg=os.getloadavg()[0], note=note)


def compiled_memory(fn, *args) -> dict:
    """Generated-code / argument / temp / output bytes for ``fn(*args)``.

    Load-independent, so quotable without a caveat. A large
    ``generated_code`` means constants have been baked into the executable --
    which is what a closed-over array on ``self`` does under ``jit_method``.
    """
    try:
        compiled = jax.jit(fn).lower(*args).compile()
        m = compiled.memory_analysis()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    out = {}
    for name in ("generated_code_size_in_bytes", "argument_size_in_bytes",
                 "temp_size_in_bytes", "output_size_in_bytes",
                 "alias_size_in_bytes"):
        val = getattr(m, name, None)
        if val is not None:
            out[name.replace("_size_in_bytes", "")] = int(val)
    return out


def fmt_bytes(n):
    if n is None:
        return "-"
    for unit in ("B", "kB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024


def machine() -> dict:
    dev = jax.devices()[0]
    return dict(
        host=platform.node(),
        platform=jax.default_backend(),
        device=f"{dev.platform}:{dev.device_kind}",
        jax=jax.__version__,
        x64=bool(jax.config.read("jax_enable_x64")),
        loadavg=os.getloadavg()[0],
        recorded=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def write_results(name, payload, outdir=None):
    outdir = Path(outdir or Path(__file__).resolve().parent / "results")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}-{machine()['host']}-{time.strftime('%Y%m%d')}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
