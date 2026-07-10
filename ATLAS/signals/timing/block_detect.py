"""Auto-detect Gibbs blocks from a parameter covariance — no parameter names.

Two methods (cross-checked):

PRIMARY — eigenstructure of the correlation matrix R = D^-1 C D^-1 (D=sqrt(diag C)).
  A small eigenvalue of R is a near-degenerate parameter COMBINATION (a direction much
  more tightly determined than any single param, lambda<<1).  Its eigenvector's
  high-participation params are the degenerate cluster.  This catches MULTI-param joint
  degeneracies (e.g. FD1-FD2-JUMP1 trading off 3-ways) that no single pairwise
  correlation reveals.  Threshold: eigenvalue < EIG_THRESH; participation v_p^2 > PARTIC.

CROSS-CHECK — correlation graph: edge where |R_ij| > CORR_THRESH; connected components
  (size>=2) are blocks.  Pairwise; agrees with the eigen method when degeneracies are
  pairwise-visible, can miss purely-joint ones.

Self-containment: a block is Gibbs-useful only if its params are MORE correlated with
each other than with the outside.  If a candidate block's max cross-correlation is
comparable to its internal correlation, OR everything collapses into one giant
component, the geometry is 'hard' — Gibbs won't separate it; flag for global
whitening/reparam (the analogue of the routing CONFLICT bucket).

Thresholds (justified):
  EIG_THRESH = 0.10 : eigenvalue 0.1 = a combination 10x stiffer than an uncorrelated
                      param -> a clear degeneracy.  (J1600 FD block smallest eval 0.018.)
  PARTIC     = 0.10 : param contributes >10% of the degenerate direction's variance.
  CORR_THRESH= 0.70 : |corr|>0.7 shares ~half the variance -> a strong pairwise link.
  SELF_RATIO = 0.80 : block kept only if max cross-corr < SELF_RATIO * mean internal
                      corr; else flagged 'not self-contained' (Gibbs won't fully help).
"""
import numpy as np

EIG_THRESH = 0.10
PARTIC = 0.10
CORR_THRESH = 0.70
SELF_RATIO = 0.80
BOUNDARY_THRESH = 0.50   # affine<->offset corr for a boundary degeneracy
BOUNDARY_MIN_OFFSET = 2  # boundary block only when an affine param is entangled with
                         # >= this many near-wall offset params (the case the frozen
                         # metric can't whiten: multi-coupling x offset numerical extremity)


def _merge_overlapping(clusters):
    clusters = [set(c) for c in clusters]
    merged = True
    while merged:
        merged = False
        out = []
        while clusters:
            a = clusters.pop()
            hit = [b for b in clusters if a & b]
            if hit:
                for b in hit:
                    a |= b; clusters.remove(b)
                merged = True
            out.append(a)
        clusters = out
    return clusters


def _connected_components(adj):
    n = adj.shape[0]; seen = set(); comps = []
    for i in range(n):
        if i in seen:
            continue
        stack = [i]; comp = set()
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j); comp.add(j)
            stack += [k for k in range(n) if adj[j, k] and k not in seen]
        comps.append(comp)
    return comps


def detect_gibbs_blocks(C, names, eig_thresh=EIG_THRESH, partic=PARTIC,
                        corr_thresh=CORR_THRESH, self_ratio=SELF_RATIO, verbose=True,
                        blockable=None, offset=None):
    """Return dict: blocks_eig, blocks_graph, agree, blocks (final, self-contained),
    hard (flag if geometry is not block-separable).

    ``C`` / ``names`` should be the FULL covariance over ALL sampled params, so the
    self-containment test sees every cross-correlation (a candidate affine block that is
    tangled with the bounded binary params M2/SINI would otherwise be wrongly kept and
    HURT Gibbs mixing).  ``blockable`` (optional set of names) restricts the FINAL blocks
    to params that may be Gibbs-grouped (the affine ones); offset/bounded params are
    sampled in their own coords and never grouped.
    """
    C = np.asarray(C, float); n = len(names)
    blockset = set(names if blockable is None else blockable)
    D = np.sqrt(np.diag(C)); R = C / np.outer(D, D)
    R = 0.5 * (R + R.T)
    evals, evecs = np.linalg.eigh(R)

    # ---- PRIMARY: eigenstructure ----
    clustersE = []
    deg_dirs = []
    for i, ev in enumerate(evals):
        if ev < eig_thresh:
            w = evecs[:, i] ** 2
            members = [j for j in range(n) if w[j] > partic]
            if len(members) >= 2:
                clustersE.append(members); deg_dirs.append((ev, members))
    blocksE_idx = _merge_overlapping(clustersE)
    blocksE = [sorted(b) for b in blocksE_idx]

    # ---- CROSS-CHECK: correlation graph ----
    A = (np.abs(R) > corr_thresh) & ~np.eye(n, dtype=bool)
    blocksG = [sorted(c) for c in _connected_components(A) if len(c) >= 2]

    # ---- BOUNDARY blocks: an affine param entangled with >= BOUNDARY_MIN_OFFSET
    # near-wall OFFSET params.  Offset params have extreme dynamic range; an affine
    # param coupled to several of them sits in a degeneracy the frozen full metric
    # whitens poorly (seen: J1909 PBDOT vs PB+TASC).  Group them so a Gibbs conditional
    # metric handles the joint block.  (offset_idx passed in via ``offset_names``.) ----
    offset_names = set(offset) if offset is not None else set()   # near-wall offset params ONLY
    off_i = [i for i in range(n) if names[i] in offset_names]
    aff_i = [i for i in range(n) if names[i] in blockset]
    boundary = []
    for a in aff_i:
        partners = [o for o in off_i if abs(R[a, o]) > BOUNDARY_THRESH]
        if len(partners) >= BOUNDARY_MIN_OFFSET:
            boundary.append([a] + partners)

    # ---- self-containment (vs FULL R) + hard flag ----
    # Eigen tight blocks must be entirely blockable (affine).  Boundary blocks may
    # additionally contain the offset params the affine glue is coupled to.
    final = []; hard = []

    def _consider(b, allowed):
        b = sorted(b); bset = set(b); names_b = set(names[i] for i in b)
        if not names_b <= allowed:
            hard.append((b, 0.0, 1.0)); return
        internal = [abs(R[i, j]) for i in b for j in b if i < j]
        meanint = np.mean(internal) if internal else 0.0
        cross = [abs(R[i, j]) for i in b for j in range(n) if j not in bset]
        maxcross = max(cross) if cross else 0.0
        if meanint > 0 and maxcross < self_ratio * meanint and len(b) < n:
            final.append(b)
        else:
            hard.append((b, meanint, maxcross))

    for b in _merge_overlapping([list(x) for x in blocksE]):
        _consider(b, blockset)                              # affine-only
    for b in _merge_overlapping(boundary):
        _consider(b, blockset | offset_names)               # affine + coupled offsets
    final = _merge_overlapping(final)                       # merge any eigen/boundary overlap
    final = [sorted(b) for b in final]

    # giant-component / everything-correlated check
    giant = any(len(c) > 0.6 * n for c in _connected_components(A))

    def nm(idxs):
        return [names[i] for i in idxs]

    agree = sorted([frozenset(nm(b)) for b in blocksE]) == sorted([frozenset(nm(b)) for b in blocksG])
    if verbose:
        print(f"  R cond={np.linalg.cond(R):.2e}  smallest evals={np.round(evals[:4],4)}")
        print(f"  EIGEN blocks : {[nm(b) for b in blocksE] or 'none'}")
        for ev, mem in deg_dirs:
            print(f"     deg-dir eval={ev:.4f}  params={nm(mem)}")
        print(f"  GRAPH blocks : {[nm(b) for b in blocksG] or 'none'}")
        print(f"  methods agree: {agree}")
        if hard:
            print(f"  NOT self-contained (flagged hard): {[(nm(b),round(mi,2),round(mc,2)) for b,mi,mc in hard]}")
        if giant:
            print(f"  ** GIANT correlated component (> 60% of params) -> geometry hard, "
                  f"Gibbs won't separate; needs global whitening/reparam")
    return dict(blocks_eig=[nm(b) for b in blocksE], blocks_graph=[nm(b) for b in blocksG],
                blocks=[nm(b) for b in final], agree=agree, hard=[nm(b) for b, _, _ in hard],
                giant=giant, R=R, evals=evals)


def blockable_affine(par, tim):
    """Params that are Gibbs-blockable: the AFFINE-routed ones (not offset near-wall,
    not bounded-reparametrised).  Offset and bounded params are sampled in their own
    coords, so Gibbs blocks only ever group affine params.  Uses the routing policy
    (precision + linearity), not names."""
    import warnings; warnings.filterwarnings("ignore")
    from jug.engine.session import TimingSession
    from ATLAS.signals.timing.base import setup_timing_model
    import jax.numpy as jnp
    from ATLAS.signals.timing.routing import route, PREC_THRESH
    s = TimingSession(par, tim, verbose=False); fj = s.fit_parameters(max_iter=5)
    LBL = [l for l in fj["design_matrix_labels"] if l != "OFFSET"]
    sigJUG = {k: float(fj["uncertainties"][k]) for k in LBL}
    dm, aux = setup_timing_model(par, tim, sample_list=LBL, marginalise_list=[])
    th0 = aux["theta_0"]; base = {k: jnp.asarray(float(th0[k])) for k in LBL}
    BOUND = {"M2", "SINI", "ECC"}                          # bounded-reparametrised
    offset = [k for k in LBL if route(k, float(th0[k]), sigJUG[k], dm, base)["bucket"] == "offset"]
    affine = [k for k in LBL if k not in offset and k not in BOUND]
    labs = list(fj["design_matrix_labels"]); C = np.asarray(fj["covariance"])
    return LBL, offset, sorted(BOUND & set(LBL)), affine, labs, C


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("usage: python -m ATLAS.signals.timing.block_detect <par> <tim>")
    par, tim = sys.argv[1], sys.argv[2]
    LBL, offset, bound, affine, labs, C = blockable_affine(par, tim)
    # Index C (covariance, OFFSET-excluded) by LBL, not labs: labs INCLUDES
    # 'OFFSET' so labs.index would be off-by-one whenever JUG emits it.
    idx = [LBL.index(k) for k in affine]; Caff = C[np.ix_(idx, idx)]
    print(f"\n===== {len(LBL)} params | offset={offset} bound={bound} affine={len(affine)} =====")
    print("Detecting Gibbs blocks among AFFINE params only:")
    out = detect_gibbs_blocks(Caff, affine)
    print(f"  FINAL self-contained blocks (used for Gibbs): {out['blocks'] or 'none'}")
    if out['giant']:
        print("  -> geometry HARD: dominant correlated web, Gibbs limited")
