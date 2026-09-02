# ATLAS/signals

See the repository [README](../../README.md) for the funnel, the model-string
grammar and the column layout this package produces.

* `factorized/` — `Red`, `GaussianTiming` and `SuperSignal`: the basis, the
  helper assembly and the three likelihoods.
* `correlated/` — `Correlated`: the GWB signal and its overlap reduction
  function.
* `deterministic/` — CW and other deterministic signals. **Currently has no
  working end-to-end path**; see "Known broken" in the top-level README.
* `timing/` — the non-linear timing model, built on JUG. The most carefully
  engineered code in the tree: it raises `RoutingConflictError` rather than
  guessing, and freezes a dense NUTS metric from JUG's own covariance.
* `signals_utils.py` — the basis-string parser, the column-slice bookkeeping
  (`merge_slices`, `block_slice`, `vec_slice`) and the timing-model SVD.
