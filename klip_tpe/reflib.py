"""Reference libraries whose composition the optimizer searches.

Port of the reference-frame selection in the IDL ``optimize_mwc_tpe.pro`` /
``reduce_nircam_coro.pro`` pair, generalized from two hard-coded groups to any number.

The IDL scheme, for the record
------------------------------
MWC 758 F430M (JWST/NIRCam GO 4014) is reduced in ARDI: the reference library for a given
science frame is drawn from the *other* science roll plus the dedicated reference star
HD 36575.  ``reduce_nircam_coro`` restores a precomputed similarity matrix and then, **for
each science frame independently**, ranks each of those two pools by similarity to that
frame and keeps the best ``nkalt`` alternate-roll frames and the best ``nkpsf`` HD 36575
frames.  Both counts are searched dimensions (``nkalt`` in [0, 25], ``nkpsf`` in [1, 25]),
and ``k_klip`` is clamped to their sum.  Frames from a target's *own* roll are never
eligible, because a science frame contains the companion and would subtract it.

What is general here
--------------------
* Any number of named :class:`ReferenceGroup` s, each with its own searched count.
* A group may be science data (then it carries a ``partition`` label per frame and the
  same-partition exclusion applies) or an external star (no exclusion, never a target).
* Two similarity metrics, both measured where it matters -- inside the optimized annulus,
  after the same high-pass filter the reduction uses.  ``cc`` is a peak normalized
  cross-correlation (higher is better); ``ssr`` is the residual of a per-pair scale +
  offset fit (lower is better).  Which one ranks references better is itself a question
  worth searching, so the metric is selectable.

The expensive part -- the similarity matrix -- depends only on the data and the filter
width, never on the searched counts, so it is computed once per (annulus, filter) and
cached.  Selecting a library for one frame is then an ``argsort`` over a row.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .klip import highpass

__all__ = ["ReferenceGroup", "ReferenceLibrary", "similarity_matrix"]

#: similarity metrics: name -> (better is larger?)
METRICS = {"cc": True, "ssr": False}


def _annulus_mask(shape: Tuple[int, int], inrad: float, outrad: float,
                  center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    ny, nx = shape
    cx, cy = ((nx - 1) / 2.0, (ny - 1) / 2.0) if center is None else center
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy)
    return (rr >= inrad) & (rr <= outrad)


def _prep(cube: np.ndarray, mask: np.ndarray, filt: float) -> np.ndarray:
    """High-pass each frame, take the annulus pixels, mean-subtract, unit-normalise.

    Normalising here means the correlation of two prepared rows is their dot product and
    the whole matrix is one ``@``; it also makes ``cc`` independent of the frames' overall
    brightness, which for a coronagraphic sequence varies with pointing and is not a
    similarity of the speckle field.
    """
    out = np.empty((cube.shape[0], int(mask.sum())), float)
    for i, im in enumerate(cube):
        f = highpass(np.asarray(im, float), int(round(filt)), nan_aware=True) if filt and filt > 0 \
            else np.asarray(im, float)
        v = f[mask]
        v = np.where(np.isfinite(v), v, 0.0)
        v -= v.mean()
        nrm = np.sqrt((v * v).sum())
        out[i] = v / nrm if nrm > 0 else v
    return out


def similarity_matrix(target_cube: np.ndarray, library_cube: np.ndarray, *,
                      inrad: float, outrad: float, filt: float = 0.0,
                      metric: str = "cc", center: Optional[Tuple[float, float]] = None,
                      shift_px: int = 1) -> np.ndarray:
    """``(n_library, n_target)`` similarity, measured in the annulus after high-pass.

    ``cc`` is the normalized cross-correlation maximised over integer shifts within
    ``+/-shift_px`` -- the "peak" of the IDL's peak cross-correlation.  JWST coronagraphic
    frames are co-registered to a fraction of a pixel, so the search is a robustness
    measure rather than a registration: at ``shift_px=0`` it is the plain zero-lag
    correlation and costs one matrix product.

    ``ssr`` fits ``target ~ a * reference + b`` per pair by least squares and returns the
    mean squared residual.  On unit-normalised, mean-subtracted rows that reduces to
    ``1 - cc**2``, which is a monotone function of ``cc`` -- so the two metrics rank
    identically at zero lag and differ only through the shift search.  Kept distinct
    because the IDL writes both and a caller may want the residual's scale.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown similarity metric {metric!r}; have {sorted(METRICS)}")
    mask = _annulus_mask(target_cube.shape[-2:], inrad, outrad, center)
    if not mask.any():
        raise ValueError(f"empty annulus {inrad}-{outrad} px for shape {target_cube.shape[-2:]}")
    T = _prep(target_cube, mask, filt)
    best = _prep(library_cube, mask, filt) @ T.T                 # zero lag, (nlib, ntar)
    for dy in range(-shift_px, shift_px + 1):
        for dx in range(-shift_px, shift_px + 1):
            if dy == 0 and dx == 0:
                continue
            rolled = np.roll(np.roll(library_cube, dy, axis=-2), dx, axis=-1)
            np.maximum(best, _prep(rolled, mask, filt) @ T.T, out=best)
    return best if metric == "cc" else np.clip(1.0 - best ** 2, 0.0, None)


@dataclass
class ReferenceGroup:
    """One pool of candidate reference frames with its own searched count.

    ``partition`` is per-frame and only meaningful for a group taken from the science data
    itself: a target frame never draws references carrying its own label, which is what
    keeps a roll from subtracting its own companion.  An external reference star leaves it
    ``None`` and every one of its frames is always eligible.
    """
    name: str
    rows: np.ndarray                              # indices into the combined library cube
    partition: Optional[np.ndarray] = None        # per-frame label, same length as rows
    min_keep: int = 0
    max_keep: Optional[int] = None                # default: all of them

    def __post_init__(self):
        self.rows = np.asarray(self.rows, int)
        if self.partition is not None:
            self.partition = np.asarray(self.partition)
            if self.partition.shape != self.rows.shape:
                raise ValueError(f"group {self.name!r}: {self.partition.size} labels for "
                                 f"{self.rows.size} frames")
        # The reachable size of a partitioned pool is not its row count: a target only ever
        # sees the rows carrying a DIFFERENT label, so for two 25-frame rolls the ceiling is
        # 25, not 50.  Bounding at 50 would leave half the searched range describing
        # configurations identical to "keep everything eligible", which the optimizer would
        # have to spend evaluations discovering are ties.
        reach = int(self.rows.size)
        if self.partition is not None and self.partition.size:
            vals, counts = np.unique(self.partition, return_counts=True)
            reach = int(counts.sum() - counts.min()) if vals.size > 1 else 0
        if self.max_keep is None:
            self.max_keep = reach
        self.max_keep = int(min(self.max_keep, reach))
        self.min_keep = int(np.clip(self.min_keep, 0, self.max_keep))

    @property
    def param_name(self) -> str:
        return f"nkeep_{self.name}"


@dataclass
class ReferenceLibrary:
    """Named groups over one library cube, plus the similarity that ranks them.

    ``similarity`` is ``(n_library, n_target)``; ``better_is_larger`` says which end of it
    a good reference sits at.  :meth:`select` is called once per target frame per
    evaluation, so it does no arithmetic beyond an ``argsort`` of the eligible rows.
    """
    groups: List[ReferenceGroup]
    similarity: np.ndarray
    target_partition: Optional[np.ndarray] = None   # label per TARGET frame
    metric: str = "cc"
    n_min_ref: int = 2

    def __post_init__(self):
        self.similarity = np.asarray(self.similarity, float)
        names = [g.name for g in self.groups]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate reference-group names: {names}")
        if self.target_partition is not None:
            self.target_partition = np.asarray(self.target_partition)

    @property
    def better_is_larger(self) -> bool:
        return METRICS[self.metric]

    @property
    def n_target(self) -> int:
        return int(self.similarity.shape[1])

    @property
    def max_total(self) -> int:
        return int(sum(g.max_keep for g in self.groups))

    def params(self):
        """One integer :class:`~klip_tpe.space.Param` per group, for the search space."""
        from .space import Param
        return [Param(name=g.param_name, lo=float(g.min_keep), hi=float(g.max_keep),
                      kind="int", default=float(g.max_keep), role="reduction",
                      doc=f"reference frames kept from the {g.name!r} pool, "
                          f"ranked per target frame by {self.metric}")
                for g in self.groups]

    def keep_from_config(self, cfg: Dict) -> Dict[str, int]:
        """Pull ``nkeep_<group>`` out of a decoded config, defaulting to the full pool."""
        return {g.name: int(round(float(cfg.get(g.param_name, g.max_keep)))) for g in self.groups}

    def eligible(self, target: int, group: ReferenceGroup) -> np.ndarray:
        """Rows of ``group`` this target may use (the same-partition exclusion)."""
        if group.partition is None or self.target_partition is None:
            return group.rows
        return group.rows[group.partition != self.target_partition[target]]

    def select(self, target: int, keep: Dict[str, int]) -> np.ndarray:
        """Library rows for one target frame: the best ``keep[name]`` of each group.

        Ranking is per target, which is the whole point -- a frame taken in poor pointing
        is best matched by different references than the frame after it, and averaging
        that choice over the sequence is what a fixed library does.
        """
        out: List[np.ndarray] = []
        for g in self.groups:
            n = int(keep.get(g.name, g.max_keep))
            if n <= 0:
                continue
            rows = self.eligible(target, g)
            if rows.size == 0:
                continue
            s = self.similarity[rows, target]
            order = np.argsort(s)[::-1] if self.better_is_larger else np.argsort(s)
            good = order[np.isfinite(s[order])]
            out.append(rows[good[:n]])
        if not out:
            return np.zeros(0, int)
        return np.unique(np.concatenate(out))

    def mask(self, target: int, keep: Dict[str, int], n_library: int) -> np.ndarray:
        m = np.zeros(int(n_library), bool)
        m[self.select(target, keep)] = True
        return m

    def project(self, cfg: Dict) -> Dict:
        """Feasibility projection: enough references to build a basis, and ``k_klip``
        no larger than the number retained.

        This is the ``nref_eff`` clamp of ``optimize_mwc_tpe.pro`` -- asking for 30 KL
        modes from a 12-frame library is not a worse configuration, it is not a
        configuration, and letting the search spend evaluations discovering that is the
        kind of waste the guard exists to prevent.  Raising counts rather than lowering
        ``k_klip`` would change what the user asked for, so the total is only raised when
        it is below ``n_min_ref``, and then from the largest pool first.
        """
        cfg = dict(cfg)
        keep = self.keep_from_config(cfg)
        total = sum(keep.values())
        if total < self.n_min_ref:
            for g in sorted(self.groups, key=lambda g: -g.max_keep):
                room = g.max_keep - keep[g.name]
                add = int(min(room, self.n_min_ref - total))
                keep[g.name] += add
                total += add
                if total >= self.n_min_ref:
                    break
        for g in self.groups:
            cfg[g.param_name] = int(keep[g.name])
        if "k_klip" in cfg and total > 0:
            cfg["k_klip"] = int(max(1, min(int(round(float(cfg["k_klip"]))), total)))
        return cfg


def science_and_reference_library(science: np.ndarray, partition: Sequence,
                                  external: Optional[Dict[str, np.ndarray]] = None, *,
                                  inrad: float, outrad: float, filt: float = 0.0,
                                  metric: str = "cc", shift_px: int = 1,
                                  n_min_ref: int = 2,
                                  science_group: str = "altroll",
                                  min_keep: Optional[Dict[str, int]] = None):
    """Build the MWC 758 arrangement for any data set.

    ``science`` is the target cube with a ``partition`` label per frame (roll, night,
    channel); it becomes one group whose same-label frames are excluded per target.  Each
    entry of ``external`` is a cube of reference-star frames and becomes a group of its
    own, always eligible.  Returns ``(library_cube, ReferenceLibrary)`` where the library
    cube is the science frames followed by each external cube in turn, so the group row
    indices address it directly.
    """
    science = np.asarray(science, float)
    cubes = [science]
    groups = [ReferenceGroup(science_group, np.arange(science.shape[0]),
                             partition=np.asarray(list(partition)))]
    off = science.shape[0]
    for name, cube in (external or {}).items():
        cube = np.asarray(cube, float)
        if cube.shape[-2:] != science.shape[-2:]:
            raise ValueError(f"reference group {name!r} is {cube.shape[-2:]}, "
                             f"science is {science.shape[-2:]}")
        cubes.append(cube)
        groups.append(ReferenceGroup(name, np.arange(off, off + cube.shape[0])))
        off += cube.shape[0]
    for g in groups:
        g.min_keep = int((min_keep or {}).get(g.name, g.min_keep))
        g.__post_init__()
    library = np.concatenate(cubes, axis=0)
    sim = similarity_matrix(science, library, inrad=inrad, outrad=outrad, filt=filt,
                            metric=metric, shift_px=shift_px)
    return library, ReferenceLibrary(groups, sim, target_partition=np.asarray(list(partition)),
                                     metric=metric, n_min_ref=n_min_ref)
