"""Feasibility projections applied to every proposed vector before evaluation.

:class:`ReferenceCountGuard` is the ``near2_ref_*`` guardrail: ``angsep`` and
``anglemax`` jointly decide how many reference frames each target frame has, and an
unconstrained search discovers degenerate corners with almost no references that
produce meaningless (self-subtraction-free) reductions which nonetheless score
well.  The proposed pair is projected to the nearest feasible pair that leaves at
least ``n_min_ref`` references for at least ``ref_frac`` of the target frames, using
a census of the (frame-selected, binned) parallactic angles of each partition.

Projections are pure functions ``x -> x'`` and the *projected* vector is what gets
evaluated, logged and used to train the model.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from .klip import arcdist_deg, bin_angles, frame_selection_mask
from .space import SearchSpace

__all__ = ["ref_fraction", "max_feasible_angsep", "min_feasible_anglemax", "ReferenceCountGuard",
           "FrameSelectionGuard", "MinBinGuard", "compose", "ReferenceLibraryGuard"]


def ref_fraction(bangles: np.ndarray, angsep: float, anglemax: float, arcdist: float, nminref: int) -> float:
    """Fraction of (binned) target frames with >= ``nminref`` references in the
    window ``angsep*arcdist <= |dPA| <= anglemax``."""
    b = np.asarray(bangles, float)
    n = b.size
    if n <= 1:
        return 0.0
    lo = angsep * abs(arcdist)
    dpa = np.abs(b[:, None] - b[None, :])
    ok = (dpa >= lo) & (dpa <= anglemax)
    np.fill_diagonal(ok, False)
    return float((ok.sum(axis=1) >= nminref).mean())


def max_feasible_angsep(bangles, anglemax, arcdist, nminref, ref_frac, aslo, ashi, iters: int = 41) -> float:
    if ref_fraction(bangles, aslo, anglemax, arcdist, nminref) < ref_frac:
        return aslo
    if ref_fraction(bangles, ashi, anglemax, arcdist, nminref) >= ref_frac:
        return ashi
    lo, hi = aslo, ashi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ref_fraction(bangles, mid, anglemax, arcdist, nminref) >= ref_frac:
            lo = mid
        else:
            hi = mid
    return lo


def min_feasible_anglemax(bangles, angsep, arcdist, nminref, ref_frac, amlo, amhi, iters: int = 41) -> float:
    if ref_fraction(bangles, angsep, amhi, arcdist, nminref) < ref_frac:
        return amhi
    if ref_fraction(bangles, angsep, amlo, arcdist, nminref) >= ref_frac:
        return amlo
    lo, hi = amlo, amhi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ref_fraction(bangles, angsep, mid, arcdist, nminref) >= ref_frac:
            hi = mid
        else:
            lo = mid
    return hi


class ReferenceCountGuard:
    """Project ``(angsep, anglemax)`` of every partition block to a feasible pair.

    Parameters
    ----------
    angles_fn
        ``f(partition_id) -> per-frame parallactic angles`` (deg).
    tags_fn
        ``f(partition_id) -> {'corrs','noises','coronoise'}`` or None.
    fwhm_px, lam_over_d_px
        for the binning PA-span rule and the ``angsep`` unit.
    n_min_ref, ref_frac
        guardrail thresholds (10 references for 95 % of targets in production).
    random_redraw
        for random (warm-up / explore) proposals, draw ``angsep`` uniformly inside
        the feasible range instead of snapping to its boundary (keeps the random
        phase unbiased; guided proposals are always snapped).
    """

    def __init__(self, angles_fn: Callable[[Any], np.ndarray], tags_fn: Optional[Callable] = None,
                 fwhm_px: float = 6.38, lam_over_d_px: float = 6.206, n_min_ref: int = 10,
                 ref_frac: float = 0.95, random_redraw: bool = True, k_max: int = 100):
        self.angles_fn, self.tags_fn = angles_fn, tags_fn
        self.fwhm_px, self.lam_over_d_px = float(fwhm_px), float(lam_over_d_px)
        self.n_min_ref, self.ref_frac = int(n_min_ref), float(ref_frac)
        self.random_redraw = bool(random_redraw)
        self.k_max = int(k_max)
        self._cache: Dict[Any, np.ndarray] = {}

    def binned_angles(self, pid, params: Dict[str, Any], outrad: float) -> np.ndarray:
        """Replicate the reducer's frame selection + angle-aware binning."""
        if pid not in self._cache:
            self._cache[pid] = np.asarray(self.angles_fn(pid), float)
        ang = self._cache[pid]
        tags = self.tags_fn(pid) if self.tags_fn else None
        k = params.get("k_klip", None)
        k = self.k_max if (k is None or k < 1) else k
        keep = frame_selection_mask(ang.size, tags, params.get("corr_thresh"), params.get("noise_max"),
                                    params.get("coronoise_max"), int(round(params.get("bin", 1))), int(round(k)))
        a = ang[keep]
        dth = np.rad2deg(0.5 * self.fwhm_px / max(float(outrad), 1.0))
        return bin_angles(a, int(round(params.get("bin", 1))), dth)

    def feasible_pair(self, bang, angsep, anglemax, arcdist, am_hi, rng=None):
        """One partition: returns projected ``(angsep, anglemax)``."""
        if bang.size <= 1:
            return angsep, anglemax
        if ref_fraction(bang, 0.0, anglemax, arcdist, self.n_min_ref) < self.ref_frac:
            am = min_feasible_anglemax(bang, 0.0, arcdist, self.n_min_ref, self.ref_frac, anglemax, am_hi)
            return 0.0, am
        asf = max_feasible_angsep(bang, anglemax, arcdist, self.n_min_ref, self.ref_frac, 0.0, max(angsep, 0.0))
        if angsep > asf:
            if rng is not None and self.random_redraw:
                return float(rng.random() * asf), anglemax
            return asf, anglemax
        return angsep, anglemax

    def __call__(self, x: np.ndarray, space: SearchSpace, rng=None, is_random: bool = False,
                 zone: Optional[Sequence[float]] = None, **_) -> np.ndarray:
        if "angsep" not in space.bases or "anglemax" not in space.bases:
            return x
        x = np.asarray(x, float).copy()
        cfg = space.decode(x)
        inrad = float(cfg.params.get("inrad", 0.0)) if zone is None else float(zone[0])
        outrad = float(cfg.params.get("outrad", 70.0)) if zone is None else float(zone[1])
        arc = arcdist_deg(inrad, outrad, self.lam_over_d_px)
        am_hi = float(space[[space.params[i].name for i in space.dims_of_base("anglemax")][0]].hi)
        r = rng if is_random else None
        parts = space.partitions or [None]
        as_all, am_all = [], []
        for pid in parts:
            params = cfg.params_for(pid) if pid is not None else cfg.params
            bang = self.binned_angles(pid, params, outrad)
            a_s, a_m = self.feasible_pair(bang, float(params["angsep"]), float(params["anglemax"]), arc, am_hi, r)
            if pid is None or pid in cfg.selected:
                as_all.append(a_s)
                am_all.append(a_m)
            for i in space.dims_of_partition(pid) if pid is not None else []:
                b = space.params[i].base
                if b == "angsep":
                    x[i] = a_s
                elif b == "anglemax":
                    x[i] = a_m
        # global (non-replicated) angsep/anglemax dims: most restrictive over included partitions
        for i in space.dims_of_base("angsep"):
            if space.params[i].partition is None:
                x[i] = min(as_all)
        for i in space.dims_of_base("anglemax"):
            if space.params[i].partition is None:
                x[i] = max(am_all)
        return x

    def describe(self) -> Dict[str, Any]:
        return {"name": "reference_count_guard", "n_min_ref": self.n_min_ref, "ref_frac": self.ref_frac,
                "random_redraw": self.random_redraw}


class FrameSelectionGuard:
    """Snap frame-selection triplets that would be silently ignored to the canonical
    no-cut corner (addendum 2 §1b).

    The reducer refuses a frame cut that would leave fewer than ``max(k_klip*bin, 3)``
    frames (or that culls nothing) and uses the full cube instead.  Because ``k_klip``
    and ``bin`` are searched too, feasibility is a joint constraint over five
    dimensions that no static bound can express.  Without this projection the
    history credits scores to threshold triplets that never acted (61 % of IDL
    evaluations before the fix), and the density model learns from coordinates with
    no causal role.  Here an infeasible triplet is rewritten as ``corr_thresh -> lo``,
    ``noise_max -> hi``, ``coronoise_max -> hi`` **before** it is recorded; the
    reduction is unchanged (it was going to use the full cube anyway).

    The keep rule is the reducer's own :func:`klip_tpe.klip.frame_selection_mask` --
    one implementation shared by the reducer, this guard and the reference-count
    census, so the three can never drift apart.
    """

    def __init__(self, tags_fn: Callable[[Any], Optional[Dict[str, np.ndarray]]],
                 nframes_fn: Callable[[Any], int]):
        self.tags_fn, self.nframes_fn = tags_fn, nframes_fn

    def acted(self, pid, params: Dict[str, Any]) -> bool:
        tags = self.tags_fn(pid)
        nf = int(self.nframes_fn(pid))
        keep = frame_selection_mask(nf, tags, params.get("corr_thresh"), params.get("noise_max"),
                                    params.get("coronoise_max"), int(round(params.get("bin", 1))),
                                    int(round(params.get("k_klip", 1))))
        return not bool(keep.all())

    def __call__(self, x: np.ndarray, space: SearchSpace, **_) -> np.ndarray:
        bases = space.bases
        if not any(b in bases for b in ("corr_thresh", "noise_max", "coronoise_max")):
            return x
        x = np.asarray(x, float).copy()
        cfg = space.decode(x)
        parts = space.partitions or [None]
        n_snapped = 0
        for pid in parts:
            params = cfg.params_for(pid) if pid is not None else cfg.params
            if self.acted(pid, params):
                continue
            dims = space.dims_of_partition(pid) if pid is not None else \
                [i for i in space.reduction_dims if space.params[i].partition is None]
            for i in dims:
                b = space.params[i].base
                if b == "corr_thresh":
                    x[i] = space.params[i].lo
                elif b in ("noise_max", "coronoise_max"):
                    x[i] = space.params[i].hi
            n_snapped += 1
        self.last_snapped = n_snapped
        return x

    def describe(self) -> Dict[str, Any]:
        return {"name": "frame_selection_guard"}


class MinBinGuard:
    """Keep the temporal bin at or above ``bin_min`` frames.

    Cost, not correctness.  ``bin`` sets how many consecutive frames are mean-combined
    before KLIP, so the reference library -- and the covariance and its decomposition --
    scale with ``nframes / bin``.  On a sequence of a few thousand frames the bottom of the
    range is not a slightly slower configuration, it is a different order of magnitude: on
    the four-night LMIRCam RX J0534 search, evaluations at ``bin >= 10`` took a median 25 s
    and the three that drew ``bin = 1`` took a median 8300 s (worst 12371 s, between
    neighbours of 32 s and 22 s).  Three evaluations out of 484 had consumed two thirds of
    the run's wall time, and 5000 of them would have taken a fortnight.

    The generic search space spans ``bin`` from 1 to ``nframes / 8``, which is right for the
    61-frame beta Pic cube it was written against and wrong for a 1500-frame night.  The
    NEAR space has always used the IDL production range (5-30 frames per bin) and shows
    none of this.  So this is a floor for the generic path, not a new idea.

    It exists as a *projection* rather than only as a narrower range because a running
    search cannot change its space: a checkpoint restores the recorded parameter bounds, by
    design, since the history was produced under them.  The projection hook, though, comes
    from the driver on every resume -- so flooring here rescues a run in flight with its
    evaluations intact, where editing the range would mean starting over.
    """

    def __init__(self, bin_min: int = 5):
        self.bin_min = int(max(bin_min, 1))
        self.last_snapped = 0

    def __call__(self, x: np.ndarray, space: "SearchSpace", **_) -> np.ndarray:
        if "bin" not in space.bases or self.bin_min <= 1:
            return x
        x = np.asarray(x, float).copy()
        n = 0
        for i, p in enumerate(space.params):
            if p.base != "bin":
                continue
            floor = min(float(self.bin_min), float(p.hi))
            if x[i] < floor:
                x[i] = floor
                n += 1
        self.last_snapped = n
        return x

    def describe(self) -> Dict[str, Any]:
        return {"name": "min_bin_guard", "bin_min": self.bin_min}


class ReferenceLibraryGuard:
    """Keep the searched reference counts buildable, and ``k_klip`` no larger than them.

    The counts (``nkeep_<pool>``, :mod:`klip_tpe.reflib`) say how many frames each
    reference pool contributes to every target's basis, so their sum IS the basis size.
    Asking for 30 KL modes from a 12-frame library is not a worse configuration, it is not
    a configuration: ``klip_basis`` silently clamps, several distinct draws collapse onto
    the same reduction, and the search spends evaluations learning that they tie.  This is
    the ``nref_eff = ((nkalt < 25) + (nkpsf < 25)) > 2`` clamp of ``optimize_mwc_tpe.pro``,
    applied before the reduction rather than inside it.

    Order matters against :class:`ReferenceCountGuard`, which caps ``k_klip`` by the
    ANGULAR reference census.  A searched library does not use that census -- its basis
    comes from the pools -- so this runs last and its cap wins.
    """

    def __init__(self, n_min_ref: int = 2, k_max: int = 100):
        self.n_min_ref = int(max(n_min_ref, 1))
        self.k_max = int(k_max)
        self.last_clamped = 0

    def __call__(self, x: np.ndarray, space: "SearchSpace", **_) -> np.ndarray:
        keep_i = [i for i, p in enumerate(space.params) if p.name.startswith("nkeep_")]
        if not keep_i:
            return x
        x = np.asarray(x, float).copy()
        hi = {i: float(space.params[i].hi) for i in keep_i}
        for i in keep_i:                                  # integers, inside their bounds
            x[i] = float(np.clip(round(x[i]), space.params[i].lo, hi[i]))
        total = int(sum(x[i] for i in keep_i))
        if total < self.n_min_ref:                        # raise from the largest pool first
            for i in sorted(keep_i, key=lambda j: -hi[j]):
                add = min(hi[i] - x[i], self.n_min_ref - total)
                x[i] += add
                total += int(add)
                if total >= self.n_min_ref:
                    break
        n = 0
        for i, p in enumerate(space.params):
            if p.base != "k_klip" and p.name != "k_klip":
                continue
            cap = float(min(total, self.k_max, p.hi))
            if x[i] > cap:
                x[i] = max(cap, p.lo)
                n += 1
        self.last_clamped = n
        return x

    def describe(self) -> Dict[str, Any]:
        return {"name": "reference_library_guard", "n_min_ref": self.n_min_ref, "k_max": self.k_max}


class PyKLIPLibraryGuard:
    """Keep pyKLIP's own library dimensions live: ``k_klip <= maxnumbasis <= pool``.

    pyKLIP keeps at most ``maxnumbasis`` references per target and sector, then clips the
    KL truncation to the references it kept (``klip_math``: ``np.clip(numbasis - 1, 0,
    tot_basis - 1)``).  Outside ``k_klip <= maxnumbasis <= pool`` one of the two therefore
    does nothing, and draws that differ only there reduce identically -- the dead-dimension
    failure the searched ``nkeep_*`` counts had on this backend, in miniature.  ``pool`` is
    what ``mode`` makes available at this ``bin``: the other roll's binned frames (ADI), the
    reference star's frames (RDI; the reference cube is not binned) or both (ADI+RDI).
    On a gridded ``maxnumbasis`` the upper bound is the first grid point at or above ``pool``
    (every such value means "all of them"), so the grid snap that follows cannot pull "take
    everything" below the pool.

    Runs last, after :class:`ReferenceCountGuard`, so its cap on ``k_klip`` wins.
    """

    def __init__(self, n_alt: int, n_ref: int, default_mode: str = "ADI+RDI", k_max: int = 100):
        self.n_alt, self.n_ref = int(n_alt), int(n_ref)
        self.default_mode = str(default_mode)
        self.k_max = int(k_max)

    def pool(self, mode: str, bin_: float) -> int:
        alt = int(np.ceil(self.n_alt / max(float(bin_), 1.0))) if self.n_alt else 0
        m = str(mode).upper()
        return {"ADI": alt, "RDI": self.n_ref}.get(m, alt + self.n_ref)

    def __call__(self, x: np.ndarray, space: "SearchSpace", **_) -> np.ndarray:
        names = [p.name for p in space.params]
        if "maxnumbasis" not in names:
            return x
        x = np.asarray(x, float).copy()
        if "mode" in names:
            pm = space.params[names.index("mode")]
            mode = pm.choices[int(np.clip(round(x[names.index("mode")]), 0, len(pm.choices) - 1))]
        else:
            mode = self.default_mode
        bins = [x[i] for i, p in enumerate(space.params) if p.base == "bin" or p.name == "bin"]
        pool = max(self.pool(mode, max(bins) if bins else 1.0), 1)
        k = 1.0
        for i, p in enumerate(space.params):
            if p.base == "k_klip" or p.name == "k_klip":
                x[i] = float(np.clip(x[i], p.lo, max(min(pool, self.k_max, p.hi), p.lo)))
                k = max(k, x[i])
        im = names.index("maxnumbasis")
        pmn = space.params[im]
        top = min(pool, pmn.hi)
        # pyKLIP keeps min(maxnumbasis, pool) frames, so every value >= pool means "all of
        # them".  When the grid does not hold the pool itself -- a binned pool, 95 at bin 9 --
        # clip to the first grid point AT OR ABOVE it: clipped to the pool, the sanitize that
        # follows snapped it to a grid neighbour below (90), and "take everything" was out of
        # reach at every bin but 1.
        grid = getattr(pmn, "grid", None)
        if grid is not None and len(grid):
            above = [float(g) for g in grid if float(g) >= top]
            if above:
                top = min(above)
        x[im] = float(np.clip(x[im], max(k, pmn.lo), max(top, pmn.lo)))
        return x

    def describe(self) -> Dict[str, Any]:
        return {"name": "pyklip_library_guard", "n_alt": self.n_alt, "n_ref": self.n_ref,
                "k_max": self.k_max}


def compose(*projections):
    """Chain several projections into one (applied in order; each sees the previous
    output, so e.g. the reference-count census runs on the *executed* frame set)."""
    ps = [p for p in projections if p is not None]

    def _f(x, space, **kw):
        for p in ps:
            x = p(x, space, **kw)
        return x
    _f.describe = lambda: {"name": "compose", "steps": [getattr(p, "describe", lambda: str(p))() for p in ps]}
    _f.steps = ps
    return _f
