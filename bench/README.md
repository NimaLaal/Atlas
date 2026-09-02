# bench/

Benchmarks with a protocol, so that two numbers taken on different days are
comparable.

```bash
python bench/bench_gradient.py                                  # 2 synthetic pulsars
python bench/bench_gradient.py --fixture ng15_3 --npsr 3
python bench/bench_gradient.py --case ltm-gtm --target 5
```

Results land in `bench/results/<name>-<host>-<date>.json` with the machine,
backend, JAX version, loadavg and model configuration recorded alongside every
number.

## The protocol

**Repeat the fast side, not the slow one.** `harness.timeit` takes a *target
duration* and derives the repeat count, rather than taking a repeat count and
letting you accidentally measure the slow variant once. This is not pedantry:
contention distorts a 2.5 s measurement roughly six-fold and a 90 s one by under
1%, so a before/after where the "after" is fast and measured once is worthless.

**Record compiled memory, not just wall clock.** `memory_analysis()` reports
generated-code, argument and temp bytes. Those are load-independent and so
quotable without a caveat, unlike wall clock, and they are what actually decides
whether a configuration will load on a given card. A large *generated-code*
figure means constants have been baked into the executable — which is exactly
what a closed-over array on `self` does under `jit_method`, and how the NG15
helper build once reached 9.5 GB of generated code.

**Record loadavg with every timing.** Included in every row and in the JSON.

**Never quote a compile-time host-RAM high-water mark as a sampling footprint.**
Every host-RAM figure in the older profiling notes is a high-water mark during
XLA compilation. After the helper build, 30 consecutive gradient calls add 0 MB.
The two scale differently and conflating them overstates by roughly 2x.

## What the gradient benchmark measures

| row | meaning |
|---|---|
| `helper build (forward)` | one `Tᵀ N⁻¹ T` / `Tᵀ N⁻¹ r` assembly |
| `grad, helpers FROZEN` | a leapfrog step with `vary_white=False` |
| `grad, helpers REBUILT` | a leapfrog step with `vary_white=True` |
| `grad, partial_marg` | the same, through the marginalised-P-block likelihood |

The headline is the **ratio** of the two gradient rows: the price of letting
white noise vary, which is the dominant cost of a genuine global fit.

Backend caveat: the CPU backend attributes closed-over constants differently
from the GPU backend, so `generated_code` can read 0 B on CPU for a computation
that shows megabytes on GPU. Compare memory figures only within one backend —
which is why the backend is recorded in every result file.
