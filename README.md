# ATLAS

A JAX/NumPyro **pulsar-timing-array global fit**.

A PTA measures nanohertz gravitational waves by looking for a spatially
correlated red-noise process with the Hellings–Downs signature in the timing
residuals of tens of millisecond pulsars.

**The thesis.** Conventional PTA analysis *stages* the fit: freeze the timing
solution, then freeze the white noise, then linearise and marginalise the timing
model, then sample the red noise and GWB. Timing-model and white-noise
uncertainty therefore never propagate into the GWB posterior. ATLAS refuses that
staging and puts every parameter in **one** posterior simultaneously —
non-linear timing parameters, EFAC/EQUAD/ECORR white noise, per-pulsar intrinsic
red noise, DM noise, deterministic sources and the correlated background —
sampled with gradient methods in JAX.

Nearly every design decision follows from that. Differentiable timing kernels,
non-centred reparameterisations, frozen dense mass matrices and analytic block
marginalisation all exist to make a posterior of this dimension actually
samplable.

---

## The one thing to internalise: the funnel

Every piece of data compresses, exactly once per likelihood evaluation, into
four arrays — collectively `helpers`:

| array | shape | meaning |
|---|---|---|
| `TNT` | `[npsr, ncol, ncol]` | `Tᵀ N⁻¹ T` |
| `TNr` | `[npsr, ncol]` | `Tᵀ N⁻¹ r` |
| `rNr` | scalar | `rᵀ N⁻¹ r`, **summed over the whole array** |
| `logdet_N` | scalar | `ln det N`, summed over the whole array |

**Nothing downstream of `helpers` ever touches a TOA again.** Understand what
builds them and what consumes them and you understand ATLAS.

```
                        PTA_Data  (data.py)
                       /                   \
       WhiteCov (nMatrix/base.py)       SuperSignal (signals/factorized/base.py)
       N = EFAC²(σ² + EQUAD²)           T = [ M | F_unc | F_dm | U_gtm ]
         + ECORR, Sherman–Morrison      column-slice map per signal
                       \                   /
                        \                 /
                     helpers = (TNT, TNr, rNr, logdet_N)
                                  |
             SuperSignal.lnposterior_reparam   (or partial_marg_lnposterior)
             Σ⁻¹ = TNT + φ⁻¹  ->  non-centred draw  ->  log density
                                  |
                     model_maker  (ATLAS/model.py)
                                  |
                  NUTS / MultiHMCGibbs  (samplers/canetoadracing.py)
```

Two traps in that picture, both real:

* **Construction order is load-bearing and unenforced.** `WhiteCov.__init__`
  ends by mutating the data object (`data.add_white_noise_cov(self)`), and
  `SuperSignal` reads `data.Nmat` during *its* construction. **Build white noise
  before red noise**, or you get an `AttributeError` from inside a
  `functools.partial`.
* With `vary_white=False` the helpers are built **once, outside** the sampler and
  passed in as a constant. With `vary_white=True` they are rebuilt on **every
  leapfrog step**, which is the dominant cost of a global fit.

---

## The model string

The single most important abstraction is the string passed to `make_red_noise`.
It is a small einsum-flavoured mini-language:

```
"ltm|unc+cor->unc;gtm"
 └┬┘ └───┬───┘└┬┘ └┬┘
  │      │     │   └── separate blocks, each gets its own columns
  │      │     └────── representative: whose basis the shared group uses
  │      └──────────── shared group: these signals SHARE columns
  └─────────────────── prepend the linear timing-model design matrix M
```

Five recognised signal names: `unc` (per-pulsar intrinsic red noise), `cor`
(correlated GWB), `dm` (dispersion-measure noise), `gtm` (the Gaussian/Adaptus
timing basis) and `det` (deterministic signal).

**"Shared" means overlapping, not adjacent.** When `unc` and `cor` share a
basis, `cor`'s slice is *nested inside* `unc`'s — the GWB is modelled only in
the lowest frequency bins, so it occupies the first `2 × n_gwb` columns of the
IRN block. Two independent Gaussian processes ride the same basis columns with
distinct coefficient vectors. In φ the overlap becomes an addition; only the
GWB bins acquire off-diagonal ORF-weighted cross-pulsar terms, and the rest get
a cheap reciprocal inverse instead of a Cholesky.

Worked layout for `"ltm|unc+cor->unc;dm,gtm"` at 4 GWB bins, 6 IRN bins, 3 DM
bins, 8 Adaptus modes and a 6-column timing block:

```
col   0 ..  5   timing   M, zero-padded to the array-wide maximum width
col   6 .. 17   unc      F_irn         <- cor occupies cols 6..13, nested
col  18 .. 23   dm       F_dm
col  24 .. 31   gtm      U_adaptus
```

**Padding.** Every pulsar's timing design matrix is padded to a common width so
the whole thing is one batched `[npsr, ntoa, ncol]` tensor for the GPU. Padded
columns are identically zero in `T` and are given prior precision 1 (not the
`1e-40` the real ones get), so HMC sees unit curvature instead of a flat,
infinitely wide direction.

> **Shelf life.** This section describes the pre-registry code. Stage 1 of the
> development plan replaces the five hardcoded names with a component registry
> and makes the column map self-describing via `pta.summary()`, at which point
> the table above should be generated rather than written down.

---

## Four timing-model philosophies

Four mutually exclusive treatments of the timing model coexist in the tree,
selected by different flags in different places. This is the largest single
source of confusion in the repository.

| philosophy | how to select | what it does | status |
|---|---|---|---|
| **Marginalise in N** | `marg_timing=True` | absorbs `M` into `N` via a projection; no timing columns in `T` | fastest; `cond(Σ) ≈ 5×10⁵` |
| **Linear, sampled** | `linear_timing=True` | `M` is prepended to `T`; coefficients sampled with a near-flat `φ⁻¹ = 1e-40` prior | **the current production path**; `cond(Σ)` up to `10¹⁸` |
| **Non-linear (JUG)** | pass a `tm_model` to `model_maker` | samples physical timing parameters through differentiable JUG kernels | works; needs a frozen dense mass matrix |
| **Adaptus** | `;gtm` in the model string | PCA of prior-predictive timing residuals, used as a GP design basis | works; combines with either linear option |

Exactly one of the first two should be active. Nothing currently enforces that.

---

## Installation

```bash
pip install -e .            # the sampling path
pip install -e .[test]      # + pytest
pip install -e .[notebooks] # + matplotlib, corner
```

Two dependencies are not installable from PyPI and are therefore not declared:

* **JUG** — the timing back end, needed only for the non-linear timing path and
  for Adaptus basis generation:
  `pip install git+https://github.com/MattTMiles/jug.git@dev`.
  Note the name collision: PyPI's `jug` is an unrelated task-parallelism
  package, so `import jug` succeeding tells you nothing. Check
  `import jug.delays.barycentric_jax`.
* **tempo2** — `ATLAS.pulsar` imports `enterprise`, which imports `libstempo`
  when it is available, which requires `$TEMPO2` to point at a tempo2 runtime
  directory. Loading with `timing_package='tempo2'` reads TCB par files and
  tempo2's auto-dispatching `BINARY T2` model, both of which PINT cannot.

`MultiHMCGibbs` is vendored in `samplers/canetoadracing.py` and is not a
dependency.

**The likelihood core needs none of that.** `ATLAS.data`,
`ATLAS.model_builder`, `ATLAS.nMatrix` and `ATLAS.signals.factorized` import
cleanly without JUG, libstempo, PINT or `$TEMPO2`; only `ATLAS.pulsar` and
`ATLAS.signals.timing` need them. That is what lets the test suite run on a
plain CI runner.

---

## Normalisation: what the likelihoods drop

All three likelihoods omit constants, and they omit the **same** ones — which is
why they are comparable with each other but are **not** normalised
log-densities:

1. `-(N_toa/2) ln 2π` from the likelihood. ATLAS's value is therefore *larger*
   than a normalised one. At NG15 scale that is ~5.5×10⁵ nats.
2. On `linear_timing=True` only: `-0.5 · n_tm · ln(1e40)` per pulsar over real
   (unpadded) timing columns. The flat timing prior enters `Σ` as
   `φ⁻¹ = 1e-40`, but its log-determinant is never added.

Both are constant in every sampled parameter, so neither affects a posterior, an
MCMC acceptance ratio, or a Bayes factor between models fitted to the same data.
They **do** affect any absolute evidence, any comparison across different
`N_toa` (including "with and without J1713"), and any comparison against a code
that normalises properly.

`tests/reference.py`'s module docstring is the authoritative statement of this,
and `tests/test_identities.py` asserts the offset rather than assuming it.

---

## Tests

```bash
pip install -e .[test]
pytest                                  # ~11 s on CPU, no GPU or data needed
python tools/measure_margins.py         # achieved margins -> tools/noise_floors.json
python tools/regress.py --null          # harness self-test: must be exactly 0.0
python tools/regress.py --ref-a HEAD~1 --ref-b HEAD
```

The suite anchors on `tests/reference.py` — a dense float64 NumPy implementation
of `log N(r; 0, N + TφTᵀ)` that shares no code with ATLAS. Everything else hangs
off comparisons against it, so "the refactor is correct" never means "correct
relative to a baseline that may itself be wrong".

Fixtures are duck-typed: `tests/fixtures/pulsar.py` defines the eight attributes
ATLAS actually reads off a pulsar object. `tests/fixtures/data/*.npz` carry real
MDC1 and NG15 data (268 kB, with provenance); regenerate with
`tools/make_fixtures.py`, which is the only part that needs PINT or tempo2.

`tests/harness.py:CORPUS` is the single definition of "every currently-working
model string", shared with the regression harness:

| case | model string | columns |
|---|---|---|
| `curn` | `unc+cor->unc` | 12 |
| `curn-margtm` | `unc+cor->unc`, `marg_timing=True` | 12 |
| `ltm` | `ltm\|unc+cor->unc` | 18 |
| `ltm-dm` | `ltm\|unc+cor->unc;dm` | 24 |
| `ltm-gtm` | `ltm\|unc+cor->unc;gtm` | 26 |
| `ltm-dm-gtm` | `ltm\|unc+cor->unc;dm,gtm` | 32 |

---

## Layout

| path | lines | what |
|---|---|---|
| `data.py` | 141 | `PTA_Data` — immutable-ish data plus run flags |
| `model_builder.py` | 114 | `ModelBuilder` — the public construction API |
| `model.py` | 84 | `model_maker` — the NumPyro model, top of a run |
| `nMatrix/base.py` | 1329 | white-noise covariance and its solves |
| `signals/factorized/base.py` | 1494 | `Red`, `GaussianTiming`, `SuperSignal`, the likelihoods |
| `parameterized.py` | 1623 | φ assembly and its inverse, per model class |
| `signals/signals_utils.py` | 480 | basis-string parser, column bookkeeping, SVD |
| `signals/correlated/base.py` | 608 | `Correlated` — the GWB signal and its ORF |
| `signals/timing/` | 2266 | non-linear timing model via JUG |
| `signals/deterministic/` | 713 | CW / deterministic signals |
| `samplers/canetoadracing.py` | 678 | vendored `MultiHMCGibbs` kernels |
| `psd_functions.py` | 501 | PSD and ORF library |
| `sim.py` | 357 | simulation |
| `experimental/` | 4720 | reachable from nothing — see its `__init__` docstring |

---

## Known broken

Kept honest rather than hidden. The identity suite pins these with
`xfail(strict=True)` where it can, so a later stage sees them go green.

* **There is no working end-to-end CW path.** `JointDeterministic` cannot be
  constructed (it calls `super().__init__(signal_helper=…)` against a signature
  that takes `(signal_list, signal_combination_string, data)`), the CW delay
  function and its caller disagree about whether `tref` has already been
  subtracted, and `model_maker` never samples deterministic parameters.
* **`"ltm|unc;cor"`** — a *separate*, non-overlapping `cor` block — emits column
  slices past the end of the basis.
* **`ln_likelihood_curn` cannot be used with `gtm`**: it calls
  `jnp.repeat(φ, 2)` on a `get_phi_mat_CURN` result that a directly supplied
  `gtm_psd` has already promoted to mode resolution. The two `parameterized`
  classes also disagree on whether that method returns an array or a tuple.
* **The chromatic index is plumbed but connected to nothing.**
  `SuperSignal.update_red_basis` has no callers.
* `stabilize_TNT(eps=1e-6)` is a silent no-op for any pulsar with padded timing
  columns, because padded columns of `T` are exactly zero so `min(diag(TNT))` is
  exactly 0. Its docstring describes an eigenvalue algorithm that is commented
  out.
