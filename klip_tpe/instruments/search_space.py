"""The production search space, shared by every instrument.

:func:`make_space` builds the per-partition KLIP parameter block, 9 parameters per night, and the
night selection that NEAR, the generic adapter, NOMIC and ALES search.

Notes
-----
It lived in :mod:`klip_tpe.instruments.near` until 2026-11-01.  ``near.make_space`` and
``near.production_defaults`` still name these same objects, so an instrument's change to its
space no longer edits another instrument's module.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..reducer import PartitionedReducer
from ..space import Param, SearchSpace, kgrid

__all__ = ["make_space", "pa_span_max", "production_defaults"]

production_defaults: Dict[str, Any] = {
    # "eval 29 of run_20260620_082516" seed configuration
    "bin": 29, "n_ang": 2, "filter": 12, "angsep": 1.923, "anglemax": 26.0,
    "corr_thresh": 0.907, "noise_max": 1.451, "coronoise_max": 1.471, "k_klip": 6,
}


def pa_span_max(reducer: PartitionedReducer) -> float:
    """Return the largest parallactic-angle span of any partition.

    Parameters
    ----------
    reducer : PartitionedReducer
        The reducer.

    Returns
    -------
    span : float
        The span, in degrees.
    """
    return max(float(np.ptp(reducer.frame_angles(p))) for p in reducer.partitions())


def make_space(reducer: PartitionedReducer, per_night: bool = True, k_klip_max: Optional[int] = None,
               opt_framesel: bool = True, search_angles: bool = True, selection: Optional[str] = "two_slot",
               anglemax_hi: Optional[float] = None, defaults: Optional[Dict[str, Any]] = None,
               width_range: Optional[Tuple[float, float]] = None, search_k: bool = True,
               max_drop: int = 2, bin_range: Optional[Tuple[int, int]] = None,
               zone_range: Optional[Tuple[float, float]] = None, search_post_hp: bool = False,
               pin: Optional[Dict[str, Any]] = None,
               angsep_range: Optional[Tuple[float, float]] = None,
               corr_range: Optional[Tuple[float, float]] = None) -> SearchSpace:
    """Build the production search space, a KLIP parameter block per night and the night selection.

    The block holds ``bin``, ``n_ang``, ``filter``, ``angsep``, ``anglemax``, the frame selection
    (``corr_thresh``, ``noise_max``, ``coronoise_max``) and ``k_klip``.  A parameter whose range
    collapses to a point is pinned in ``SearchSpace.fixed`` rather than searched.

    Parameters
    ----------
    reducer : PartitionedReducer
        The reducer, whose partitions and frames set the ranges.
    per_night : bool, default True
        Whether each partition gets its own block.  False shares one block.
    k_klip_max : int or None, default None
        Maximum ``k_klip``.  None takes ``nframes / 5``, from 10 to 100.
    opt_framesel : bool, default True
        Whether to search the frame selection.
    search_angles : bool, default True
        Whether to search ``angsep`` and ``anglemax``.
    selection : str or None, default "two_slot"
        Night-selection scheme (:meth:`klip_tpe.space.SearchSpace.with_selection`).  None searches no
        selection.
    anglemax_hi : float or None, default None
        Upper bound of ``anglemax``, in degrees.  None takes the largest PA span, or 360 for a span of
        5 degrees or less, and a bound of 20 or less pins ``anglemax`` at 360.
    defaults : dict or None, default None
        Defaults that replace those of :data:`production_defaults`.
    width_range : tuple of 2 floats or None, default None
        Range of the global ``width`` of the adaptive-annulus protocol, in pixels
        (``RunConfig.opt_width``), its default at the midpoint.  None leaves it out.
    search_k : bool, default True
        Whether to search ``k_klip``.  False leaves it to the per-evaluation k-scan.
    max_drop : int, default 2
        Number of drop slots of the two-slot selection, the IDL predecessor's ``drop1`` and
        ``drop2``.  Raise it when nights times image groups give many partitions.
    bin_range : tuple of 2 ints or None, default None
        Range of frames per temporal bin.  None takes the NEAR production 5 to 30, and short
        sequences need a smaller one (:func:`klip_tpe.instruments.generic.make_space`).
    zone_range : tuple of 2 floats or None, default None
        Range of ``zone_in`` and ``zone_out``, how far the KLIP zone reaches inside and outside the
        scored annulus, in pixels.  None leaves them out, and their defaults come from ``defaults``,
        else 0.
    search_post_hp : bool, default False
        Whether to search ``post_hp`` (0 or 1), the residuals' second high-pass, its default from
        ``defaults``, else 1.
    pin : dict or None, default None
        Parameters held at a value for every evaluation.  A block parameter leaves the search, and any
        other name goes to the reducer as given (``SearchSpace.fixed``).
    angsep_range : tuple of 2 floats or None, default None
        Range of ``angsep``, in lambda/D.  None takes 0 to 3.
    corr_range : tuple of 2 floats or None, default None
        Range of ``corr_thresh``.  None takes 0 to 1, and a floor just below the data's lowest
        correlation spends the draws where the cut acts.

    Returns
    -------
    space : SearchSpace
        The search space.

    Raises
    ------
    ValueError
        If ``angsep_range`` or ``corr_range`` is out of order or out of bounds.

    Notes
    -----
    A floor of ``angsep`` above 0 keeps the search off ``angsep = 0``, which with ``anglemax`` at the
    PA span is another algorithm.  :func:`klip_tpe.klip.klip_annular` then builds one basis per zone
    from all the frames, each target inside its own basis, as the IDL predecessor's auto-fast rule
    does.  The bound is not of measure zero, since a proposal past it is clipped onto it and the
    reference-count guard projects onto 0 too, so a search lands there often (MWC 758 with ALES,
    every winner, at the cost of most of the spiral arms' flux).

    Frame correlations sit just below 1 (MWC 758's ALES frames, 0.944-0.999), so most of 0 to 1 cuts
    no frame, and the frame-selection guard records every such proposal at the no-cut floor.  Keep
    ``corr_range``'s floor at or below every partition's lowest correlation, since it is then the
    no-cut value.  The defaults of ``angsep`` and ``corr_thresh`` are clipped into their ranges.
    """
    d = dict(production_defaults)
    d.update(defaults or {})
    if k_klip_max is None:
        nfmax = max(r.data.nframes for r in reducer.reducers.values())
        k_klip_max = int(min(max(nfmax // 5, 10), 100))
    if anglemax_hi is None:
        span = pa_span_max(reducer)
        anglemax_hi = float(np.ceil(span)) if span > 5 else 360.0
    b_lo, b_hi = (5, 30) if bin_range is None else (int(bin_range[0]), int(bin_range[1]))
    b_lo, b_hi = max(b_lo, 1), max(b_hi, max(b_lo, 1))
    extra_fixed: Dict[str, Any] = {}      # parameters pinned open rather than searched
    block = [Param("bin", b_lo, b_hi, "int", default=int(min(max(d["bin"], b_lo), b_hi)),
                   doc="temporal binning (frames)"),
             Param("n_ang", 1, 8, "int", default=d["n_ang"], doc="azimuthal subdivisions"),
             Param("filter", 0, 25, "int", default=d["filter"], doc="high-pass width (px)")]
    if search_angles:
        as_lo, as_hi = (0.0, 3.0) if angsep_range is None else (float(angsep_range[0]), float(angsep_range[1]))
        if not 0.0 <= as_lo <= as_hi:
            raise ValueError(f"angsep_range {tuple(angsep_range)}: want 0 <= lo <= hi (lambda/D)")
        block += [Param("angsep", as_lo, as_hi, "float", default=float(np.clip(float(d["angsep"]), as_lo, as_hi)),
                        doc="min reference separation (lambda/D)")]
        # anglemax caps |dPA| between a target and a reference, and its floor is 20 deg.  A
        # sequence whose own PA span cannot reach 20 therefore has nothing to constrain: every
        # pair already satisfies any admissible value.  Searching it anyway used to raise
        # `anglemax: hi < lo` outright for a span between 5 and 20 deg, because anglemax_hi
        # was set from the span while the floor stayed at 20 -- a crash in make_space for any
        # such data set.  Pin it open instead of searching a window that cannot close.
        if float(anglemax_hi) <= 20.0:
            extra_fixed["anglemax"] = 360.0
        else:
            block += [Param("anglemax", 20.0, anglemax_hi, "int",
                            default=min(d["anglemax"], anglemax_hi),
                            doc="max reference field-rotation window (deg)")]
    if opt_framesel:
        c_lo, c_hi = (0.0, 1.0) if corr_range is None else (float(corr_range[0]), float(corr_range[1]))
        if not 0.0 <= c_lo <= c_hi <= 1.0:
            raise ValueError(f"corr_range {tuple(corr_range)}: want 0 <= lo <= hi <= 1")
        block += [Param("corr_thresh", c_lo, c_hi, "float", default=float(np.clip(float(d["corr_thresh"]), c_lo, c_hi))),
                  Param("noise_max", 0.3, 3.0, "float", default=d["noise_max"]),
                  Param("coronoise_max", 0.3, 3.0, "float", default=d["coronoise_max"])]
    if search_k:
        block += [Param("k_klip", 1, k_klip_max, "int", grid=kgrid(k_klip_max), default=min(d["k_klip"], k_klip_max),
                        doc="KL modes (non-uniform sampling grid)")]
    if zone_range is not None:
        z0, z1 = float(zone_range[0]), float(zone_range[1])
        clip = lambda v: float(np.clip(float(v), z0, z1))
        block += [Param("zone_in", z0, z1, "float", default=clip(d.get("zone_in", 0.0)),
                        doc="KLIP zone inside the annulus (px)"),
                  Param("zone_out", z0, z1, "float", default=clip(d.get("zone_out", 0.0)),
                        doc="KLIP zone outside the annulus (px)")]
    if search_post_hp:
        block += [Param("post_hp", 0, 1, "int", default=int(bool(d.get("post_hp", 1))),
                        doc="high-pass the residuals again before derotating (0/1)")]
    parts = reducer.partitions()
    # A parameter whose range has collapsed to a point is not a search dimension, it is a
    # constant wearing one.  It still costs the sampler a coordinate, the TPE a density to
    # model and the reader a line in the setup file, and it buys nothing.  NIRCam hit this
    # for real: its cubes are short enough that bin_range came out (1, 1), so every
    # HIP 65426 run has been searching `bin : [1.000, 1.000]`.  Pin it instead -- `fixed`
    # reaches the reducer through `decode`, so the value is unchanged and only the dimension
    # goes away.
    fixed: Dict[str, Any] = {"spat_mean": False, "temp_mean": False, **extra_fixed}
    pin_ids = set()
    for b in block:
        if float(b.hi) <= float(b.lo):
            fixed[b.name] = b.default if b.default is not None else b.lo
            pin_ids.add(id(b))
    for name, v in (pin or {}).items():
        for b in block:
            if b.name == name:
                pin_ids.add(id(b))
        fixed[name] = v
    block = [b for b in block if id(b) not in pin_ids]
    sp = SearchSpace(fixed=fixed)
    if per_night and len(parts) > 1:
        sp.replicate(block, parts, name_fmt="{base}_{pid}")
    else:
        for b in block:
            sp.add(b)
        sp.partitions = list(parts)
    if width_range is not None:
        w0, w1 = float(width_range[0]), float(width_range[1])
        sp.add(Param("width", w0, w1, "int", default=int(round(0.5 * (w0 + w1))),
                     doc="annulus width (px); outrad = min(inrad + width, r_cap)"))
    if selection and len(parts) > 1:
        sp.with_selection(selection, partitions=parts, max_drop=int(max_drop))
    return sp
