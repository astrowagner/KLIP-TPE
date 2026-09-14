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
           "FrameSelectionGuard", "MinBinGuard", "compose"]


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
