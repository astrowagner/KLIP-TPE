"""Reference libraries whose composition the optimizer searches.

It ports the reference-frame selection of the IDL predecessor's ``optimize_mwc_tpe.pro`` and
``reduce_nircam_coro.pro``, generalized from two hard-coded groups to any number.

The IDL predecessor's scheme
----------------------------
MWC 758 F430M (JWST/NIRCam GO 4014) is reduced in ARDI.  The reference library of a science frame is
drawn from the *other* science roll and the dedicated reference star HD 36575.
``reduce_nircam_coro`` restores a precomputed similarity matrix and then, **for each science frame on
its own**, ranks each of those two pools by similarity to that frame and keeps the best ``nkalt``
frames of the other roll and the best ``nkpsf`` frames of HD 36575.  Both counts are searched
dimensions (``nkalt`` in [0, 25], ``nkpsf`` in [1, 25]), and ``k_klip`` is clamped to their sum.
Frames from a target's *own* roll are never eligible, because a science frame contains the companion
and would subtract it.

What is general here
--------------------
* Any number of named :class:`ReferenceGroup` objects, each with its own searched count.
* A group may be science data, with a ``partition`` label per frame and the same-partition exclusion,
  or an external star, never excluded and never a target.
* A science group of one continuously rotating sequence has no label to exclude by.  With
  ``rotation=True`` the ADI rule decides, per target and per annulus, with the label on top of it when
  there is more than one.  The references must have turned the planet by ``angsep`` and by no more
  than ``anglemax`` (:func:`klip_tpe.klip.reference_mask`).  With every eligible frame kept and no
  other pool, that is plain ADI to the bit.
* Two similarity metrics, both measured where it matters, inside the optimized annulus and after the
  high-pass filter the reduction uses.  ``cc`` is a peak normalized cross-correlation, higher better.
  ``ssr`` is the residual of a scale and offset fit per pair, lower better.  Which one ranks references
  better is itself a question worth searching, so the metric is selectable.

The expensive part, the similarity matrix, depends only on the data and the filter width, never on the
searched counts, so it is computed once per annulus and filter and cached.  Selecting a library for one
frame is then an ``argsort`` over a row.
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
    """Return the mask of the annulus ``inrad <= r <= outrad`` about the frame's center or a given one.

    Parameters
    ----------
    shape : tuple of 2 int
        The frame's shape, ``(ny, nx)``.
    inrad : float
        Inner radius, in pixels.
    outrad : float
        Outer radius, in pixels.
    center : tuple of 2 float or None, default None
        The center, ``(x, y)`` in pixels, None for ``((nx - 1)/2, (ny - 1)/2)``.

    Returns
    -------
    mask : ndarray of bool, shape (ny, nx)
        True inside the annulus.
    """
    ny, nx = shape
    cx, cy = ((nx - 1) / 2.0, (ny - 1) / 2.0) if center is None else center
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy)
    return (rr >= inrad) & (rr <= outrad)


def _prep(cube: np.ndarray, mask: np.ndarray, filt: float) -> np.ndarray:
    """Return each frame high-passed, cut to the annulus, mean-subtracted and normalized to unit length.

    Normalized so, the correlation of two rows is their dot product and the whole matrix is one product.
    It also makes ``cc`` independent of the frames' overall brightness, which in a coronagraphic sequence
    varies with pointing and is no similarity of the speckle field.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The frames.
    mask : ndarray of bool, shape (ny, nx)
        The annulus.
    filt : float
        Width of the high-pass filter, in pixels, 0 for none.

    Returns
    -------
    rows : ndarray, shape (n, npix)
        Each frame's annulus pixels with NaN set to 0, a frame of zero norm left unnormalized.
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
    """Return each library frame's similarity to each target frame, in the annulus after the high-pass.

    ``cc`` is the normalized cross-correlation maximized over integer shifts within ``+/-shift_px``, the
    "peak" of the IDL predecessor's peak cross-correlation.  JWST coronagraphic frames are co-registered
    to a fraction of a pixel, so the shifts make the measure robust rather than register the frames, and
    at ``shift_px=0`` it is the plain correlation at zero lag, one matrix product.  ``ssr`` fits
    ``target ~ a reference + b`` per pair by least squares and returns the mean squared residual.

    Parameters
    ----------
    target_cube : ndarray, shape (n_target, ny, nx)
        The target frames.
    library_cube : ndarray, shape (n_library, ny, nx)
        The library frames.
    inrad : float
        The annulus' inner radius, in pixels.
    outrad : float
        Its outer radius, in pixels.
    filt : float, default 0.0
        Width of the high-pass filter, in pixels, 0 for none.
    metric : {"cc", "ssr"}, default "cc"
        The similarity.
    center : tuple of 2 float or None, default None
        The star's position, ``(x, y)`` in pixels, None for the frame's center.
    shift_px : int, default 1
        Largest shift searched, in pixels.

    Returns
    -------
    similarity : ndarray, shape (n_library, n_target)
        The similarity, larger better for ``cc`` and smaller better for ``ssr``.

    Raises
    ------
    ValueError
        If the metric is unknown or the annulus holds no pixel.

    Notes
    -----
    On unit-normalized, mean-subtracted rows ``ssr`` reduces to ``1 - cc**2``, which falls as a positive
    ``cc`` grows, so the two rank alike at zero lag and differ only through the shift search.  They are
    kept apart because the IDL predecessor writes both and a caller may want the residual's scale.
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
    """One pool of candidate reference frames, with its own searched count.

    A group from the science data carries a ``partition`` label per frame, and a target frame never draws
    references with its own label, which keeps a roll from subtracting its own companion.  An external
    reference star leaves the labels None, and every one of its frames is always eligible.

    Attributes
    ----------
    name : str
        The group's name, which names its parameter ``nkeep_<name>``.
    rows : ndarray of int, shape (n,)
        Its frames' rows in the combined library cube.
    partition : ndarray, shape (n,), default None
        Each frame's label, None for an external star.
    min_keep : int, default 0
        Fewest frames kept.
    max_keep : int, default None
        Most frames kept, None for every frame a target can reach.
    rotation : bool, default False
        Whether the frames are science frames of a rotating sequence.  A target's eligible rows are then
        those the ADI rule admits (``angsep`` and ``anglemax`` over the library's angles), never itself.
    scales : ndarray, shape (n,), default None
        Each row's scale factor ``lambda_target / lambda_row``, for frames of other wavelength groups
        magnified about the star, None otherwise.  A target's eligible rows are then those whose
        displacement of the planet, by rotation and scaling together, reaches the arc ``angsep`` asks for
        (:func:`klip_tpe.rescale.displacement_px`), with ``|dPA| <= anglemax`` as for ADI.
    n_labels : int
        Number of distinct labels, 0 without them, set at construction.

    Notes
    -----
    A partitioned pool's reachable size is not its row count, since a target only sees the rows with
    another label.  For two rolls of 25 frames the ceiling is 25, not 50, and a bound of 50 would leave
    half the searched range describing ties with keeping every eligible frame.  With one label and the
    rotation rule every other frame can be eligible, so the ceiling is the count less the target.
    """
    name: str
    rows: np.ndarray                              # indices into the combined library cube
    partition: Optional[np.ndarray] = None        # per-frame label, same length as rows
    min_keep: int = 0
    max_keep: Optional[int] = None                # default: all of them
    #: science frames of a rotating sequence: a target's eligible rows are the ones the ADI
    #: rule admits (``angsep``, ``anglemax`` over the library's angles), never itself
    rotation: bool = False
    #: frames of other wavelength groups magnified about the star, one scale factor per row
    #: (``lambda_target / lambda_row``): the rows the displacement rule admits, the planet
    #: having moved by rotation and scaling together at least as far as ``angsep`` asks
    #: (:func:`klip_tpe.rescale.displacement_px`), with ``|dPA| <= anglemax`` as for ADI
    scales: Optional[np.ndarray] = None

    def __post_init__(self):
        """Check the group and bound its counts by what a target can reach.

        Raises
        ------
        ValueError
            If the labels or the scales are not one per frame, or the group has both the rotation rule and
            scales.
        """
        self.rows = np.asarray(self.rows, int)
        self.n_labels = 0
        if self.scales is not None:
            self.scales = np.asarray(self.scales, float).ravel()
            if self.scales.shape != self.rows.shape:
                raise ValueError(f"group {self.name!r}: {self.scales.size} scales for {self.rows.size} frames")
            if self.rotation:
                raise ValueError(f"group {self.name!r}: a group has the rotation rule or scales, not both")
        if self.partition is not None:
            self.partition = np.asarray(self.partition)
            if self.partition.shape != self.rows.shape:
                raise ValueError(f"group {self.name!r}: {self.partition.size} labels for "
                                 f"{self.rows.size} frames")
            self.n_labels = int(np.unique(self.partition).size) if self.partition.size else 0
        # The reachable size of a partitioned pool is not its row count: a target only ever
        # sees the rows carrying a DIFFERENT label, so for two 25-frame rolls the ceiling is
        # 25, not 50.  Bounding at 50 would leave half the searched range describing
        # configurations identical to "keep everything eligible", which the optimizer would
        # have to spend evaluations discovering are ties.  One label and the rotation rule:
        # every other frame can be eligible, so the ceiling is the count less the target.
        reach = int(self.rows.size)
        if self.partition is not None and self.partition.size:
            vals, counts = np.unique(self.partition, return_counts=True)
            if vals.size > 1:
                reach = int(counts.sum() - counts.min())
            else:
                reach = int(max(self.rows.size - 1, 0)) if self.rotation else 0
        elif self.rotation:
            reach = int(max(self.rows.size - 1, 0))
        if self.max_keep is None:
            self.max_keep = reach
        self.max_keep = int(min(self.max_keep, reach))
        self.min_keep = int(np.clip(self.min_keep, 0, self.max_keep))

    @property
    def param_name(self) -> str:
        """The name of the group's searched count, ``nkeep_<name>``.
        """
        return f"nkeep_{self.name}"


@dataclass
class ReferenceLibrary:
    """Named groups of reference frames over one library cube, and the similarity that ranks them.

    :meth:`select` is called once per target frame per evaluation, so it does no arithmetic beyond an
    ``argsort`` of the eligible rows.

    Attributes
    ----------
    groups : list of ReferenceGroup
        The groups, their names distinct.
    similarity : ndarray, shape (n_library, n_target)
        Each library frame's similarity to each target frame (:func:`similarity_matrix`).
    target_partition : ndarray, shape (n_target,), default None
        Each target frame's label, None without labels.
    metric : {"cc", "ssr"}, default "cc"
        The similarity's metric, which says which end of it a good reference sits at.
    n_min_ref : int, default 2
        Fewest references a configuration may keep (:meth:`project`).
    angles : ndarray, shape (n_library,), default None
        The binned parallactic angle in degrees of each library row that is a science frame, of the
        target's group or rescaled from another, which a group with ``rotation`` or ``scales`` needs.
    mid_px : float, default None
        The annulus' mid radius in pixels, where a scaled group's displacement rule is measured, as the ADI
        arc rule is (:func:`klip_tpe.klip.arcdist_deg`).
    """
    groups: List[ReferenceGroup]
    similarity: np.ndarray
    target_partition: Optional[np.ndarray] = None   # label per TARGET frame
    metric: str = "cc"
    n_min_ref: int = 2
    #: the (binned) parallactic angle of each library row that is a science frame (of the
    #: target's group or, rescaled, of another), indexed by row; a group with ``rotation``
    #: or ``scales`` is ranked only among the rows its rule admits
    angles: Optional[np.ndarray] = None
    #: the annulus' mid radius in pixels, where the displacement rule of a scaled group is
    #: measured (the radius the ADI arc rule uses too, :func:`klip_tpe.klip.arcdist_deg`)
    mid_px: Optional[float] = None

    def __post_init__(self):
        """Check the library against its groups.

        Raises
        ------
        ValueError
            If two groups share a name, a ruled group lacks the library's angles or an angle for one of its
            rows, or a scaled group lacks ``mid_px``.
        """
        self.similarity = np.asarray(self.similarity, float)
        names = [g.name for g in self.groups]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate reference-group names: {names}")
        if self.target_partition is not None:
            self.target_partition = np.asarray(self.target_partition)
        if self.angles is not None:
            self.angles = np.asarray(self.angles, float).ravel()
        for g in self.groups:
            ruled = g.rotation or g.scales is not None
            if ruled and self.angles is None:
                raise ValueError(f"group {g.name!r} applies the rotation rule but the library has no angles")
            if ruled and g.rows.size and g.rows.max() >= (self.angles.size if self.angles is not None else 0):
                raise ValueError(f"group {g.name!r}: {self.angles.size} angles for rows up to {g.rows.max()}")
            if g.scales is not None and self.mid_px is None:
                raise ValueError(f"group {g.name!r} applies the displacement rule but the library has no mid_px")

    @property
    def better_is_larger(self) -> bool:
        """Whether a larger similarity is a better reference, True for ``cc``.
        """
        return METRICS[self.metric]

    @property
    def n_target(self) -> int:
        """The number of target frames.
        """
        return int(self.similarity.shape[1])

    @property
    def max_total(self) -> int:
        """The most references the groups can keep together.
        """
        return int(sum(g.max_keep for g in self.groups))

    def params(self):
        """Return one integer search parameter per group, its count of frames kept.

        Returns
        -------
        params : list of Param
            The parameters ``nkeep_<group>``, each from the group's ``min_keep`` to its ``max_keep``, the
            latter their default.
        """
        from .space import Param
        return [Param(name=g.param_name, lo=float(g.min_keep), hi=float(g.max_keep),
                      kind="int", default=float(g.max_keep), role="reduction",
                      doc=f"reference frames kept from the {g.name!r} pool, "
                          f"ranked per target frame by {self.metric}")
                for g in self.groups]

    def keep_from_config(self, cfg: Dict) -> Dict[str, int]:
        """Return each group's count from a decoded configuration, the full pool where it has none.

        Parameters
        ----------
        cfg : dict
            The decoded configuration.

        Returns
        -------
        keep : dict
            Each group's name and count.
        """
        return {g.name: int(round(float(cfg.get(g.param_name, g.max_keep)))) for g in self.groups}

    def eligible(self, target: int, group: ReferenceGroup, angsep_deg: float = 0.0,
                 anglemax: float = 360.0, census: bool = False) -> np.ndarray:
        """Return the rows of a group a target frame may use.

        A target never uses its own label, unless the group rotates under one label.  A ``rotation`` group
        admits only the rows the ADI rule admits, ``angsep_deg <= |dPA| <= anglemax``, never the target, and
        when that leaves fewer than 4 rows it falls back to every row but the target, as ``klip_annular``
        does without a library (``multiklip``'s fallback).  A group with ``scales`` admits the rows whose
        displacement of the planet at the annulus' mid radius, by rotation and scaling together, is at least
        the arc ``angsep_deg`` stands for, within ``anglemax`` of rotation, with no fallback.

        Parameters
        ----------
        target : int
            The target frame's index.
        group : ReferenceGroup
            The group.
        angsep_deg : float, default 0.0
            The least rotation in degrees, the arc of the ADI rule at the annulus' mid radius.
        anglemax : float, default 360.0
            The most rotation, in degrees.
        census : bool, default False
            Whether to count before the fallback, as the starved-target rule reads it.

        Returns
        -------
        rows : ndarray of int
            The rows, in the group's order.  A target the displacement rule leaves no SDI reference has none.
        """
        rows = group.rows
        if group.partition is not None and self.target_partition is not None \
                and not (group.rotation and group.n_labels < 2):
            # the label rule (one roll never serves itself); a rotating sequence under one
            # label is the case the rotation rule exists for, so the label stands down there
            rows = rows[group.partition != self.target_partition[target]]
        if group.scales is not None:
            # other groups' frames, magnified: the planet must have moved by rotation and
            # scaling together at least the arc the ADI rule asks for at the mid radius
            # (angsep_deg is that arc as an angle), and the rotation stays within anglemax
            from .rescale import displacement_px
            dpa = self.angles[rows] - self.angles[target]
            sep_px = np.deg2rad(float(angsep_deg)) * float(self.mid_px)
            disp = displacement_px(self.mid_px, group.scales[np.searchsorted(group.rows, rows)], dpa)
            ok = (disp >= sep_px - 1e-9) & (np.abs(dpa) <= float(anglemax))
            return rows[ok]
        if not group.rotation:
            return rows
        dpa = np.abs(self.angles[rows] - self.angles[target])
        ok = (dpa >= float(angsep_deg)) & (dpa <= float(anglemax)) & (rows != target)
        got = rows[ok]
        if got.size < 4 and not census:
            got = group.rows[group.rows != target]
        return got

    def select(self, target: int, keep: Dict[str, int], angsep_deg: float = 0.0,
               anglemax: float = 360.0, census: bool = False) -> np.ndarray:
        """Return the library rows of one target frame, the best ``keep[name]`` of each group.

        The ranking is per target, which is the point of it.  A frame taken in poor pointing is best matched
        by other references than the frame after it, and a fixed library averages that choice over the
        sequence.  A group kept whole keeps every eligible row, which makes a rotating sequence's full pool
        plain ADI, and any other group drops a row whose similarity is not finite.

        Parameters
        ----------
        target : int
            The target frame's index.
        keep : dict
            Each group's count, the full pool for a group missing.
        angsep_deg : float, default 0.0
            The least rotation for the groups with the rotation rule, in degrees.
        anglemax : float, default 360.0
            The most rotation, in degrees.
        census : bool, default False
            Whether the rotation groups count before their fallback (:meth:`eligible`).

        Returns
        -------
        rows : ndarray of int
            The rows, sorted and unique, empty when no group gives one.
        """
        out: List[np.ndarray] = []
        for g in self.groups:
            n = int(keep.get(g.name, g.max_keep))
            if n <= 0:
                continue
            rows = self.eligible(target, g, angsep_deg, anglemax, census=census)
            if rows.size == 0:
                continue
            if n >= rows.size:
                # every eligible row, ranked or not.  A rotating sequence keeps them all (that is
                # what makes the full pool plain ADI); another pool drops a row whose similarity
                # is not finite, as the ranking below always did
                if not g.rotation:
                    rows = rows[np.isfinite(self.similarity[rows, target])]
                out.append(rows)
                continue
            s = self.similarity[rows, target]
            order = np.argsort(s)[::-1] if self.better_is_larger else np.argsort(s)
            good = order[np.isfinite(s[order])]
            out.append(rows[good[:n]])
        if not out:
            return np.zeros(0, int)
        return np.unique(np.concatenate(out))

    def is_plain_adi(self, keep: Dict[str, int]) -> bool:
        """Tell whether these counts make the library plain ADI.

        They do with one rotation group under one label kept whole and every other pool at zero.
        ``klip_annular`` then reduces without the library at all, so that point of the space *is* the ADI
        reduction, shortcuts and all (its shared basis when nothing is excluded, its cap on ``k``), rather
        than a per-target rendering of it.

        Parameters
        ----------
        keep : dict
            Each group's count.

        Returns
        -------
        plain : bool
            True for plain ADI.
        """
        rot = [g for g in self.groups if g.rotation]
        if len(rot) != 1 or rot[0].n_labels > 1:
            return False
        # every other pool -- a reference star, rescaled frames of other groups -- at zero
        for g in self.groups:
            n = int(keep.get(g.name, g.max_keep))
            if g.rotation:
                if n < g.max_keep:
                    return False
            elif n > 0:
                return False
        return True

    def mask(self, target: int, keep: Dict[str, int], n_library: int, angsep_deg: float = 0.0,
             anglemax: float = 360.0) -> np.ndarray:
        """Return a target frame's library rows (:meth:`select`) as a mask.

        Parameters
        ----------
        target : int
            The target frame's index.
        keep : dict
            Each group's count.
        n_library : int
            Number of library frames.
        angsep_deg : float, default 0.0
            The least rotation, in degrees.
        anglemax : float, default 360.0
            The most rotation, in degrees.

        Returns
        -------
        mask : ndarray of bool, shape (n_library,)
            True on the rows kept.
        """
        m = np.zeros(int(n_library), bool)
        m[self.select(target, keep, angsep_deg, anglemax)] = True
        return m

    def project(self, cfg: Dict) -> Dict:
        """Return a configuration made feasible, with enough references and ``k_klip`` no larger than their number.

        The total is raised only when it is below ``n_min_ref``, and then from the largest pool first, since
        raising the counts to fit ``k_klip`` would change what the search asked for.

        Parameters
        ----------
        cfg : dict
            The decoded configuration.

        Returns
        -------
        cfg : dict
            A copy, each group's count and ``k_klip`` set.

        Notes
        -----
        It is the ``nref_eff`` clamp of the IDL predecessor's ``optimize_mwc_tpe.pro``.  Asking for 30 KL
        modes from a 12-frame library is not a worse configuration, it is no configuration, and a search left
        to discover that spends evaluations the guard exists to save.
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
                                  min_keep: Optional[Dict[str, int]] = None,
                                  rotation: bool = False, angles: Optional[Sequence[float]] = None):
    """Build the MWC 758 arrangement for any data set.

    The science cube becomes one group whose same-label frames are excluded per target, and each external
    cube of reference-star frames a group of its own, always eligible.

    Parameters
    ----------
    science : ndarray, shape (n, ny, nx)
        The target frames.
    partition : sequence
        Each target frame's label, its roll, night or channel.
    external : dict or None, default None
        Each reference star's name and frames, shape ``(m, ny, nx)``.
    inrad : float
        The annulus' inner radius, in pixels.
    outrad : float
        Its outer radius, in pixels.
    filt : float, default 0.0
        Width of the high-pass filter the similarity is measured after, in pixels.
    metric : {"cc", "ssr"}, default "cc"
        The similarity.
    shift_px : int, default 1
        Largest shift the similarity searches, in pixels.
    n_min_ref : int, default 2
        Fewest references a configuration keeps.
    science_group : str, default "altroll"
        The science group's name.
    min_keep : dict or None, default None
        Fewest frames kept, by group name.
    rotation : bool, default False
        Whether the science group selects by the ADI rule, as for a rotating sequence with no label to
        exclude by.
    angles : sequence of float or None, default None
        The science frames' parallactic angles in degrees, which ``rotation`` needs.

    Returns
    -------
    library : ndarray, shape (n + sum(m), ny, nx)
        The science frames followed by each external cube in turn, which the groups' rows address.
    reflib : ReferenceLibrary
        The library.

    Raises
    ------
    ValueError
        If an external cube's frames do not have the science frames' shape.
    """
    science = np.asarray(science, float)
    cubes = [science]
    groups = [ReferenceGroup(science_group, np.arange(science.shape[0]),
                             partition=np.asarray(list(partition)), rotation=bool(rotation))]
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
                                     metric=metric, n_min_ref=n_min_ref,
                                     angles=None if angles is None else np.asarray(angles, float))
