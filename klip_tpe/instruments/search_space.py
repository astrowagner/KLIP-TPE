"""The production search space, shared by every instrument.

:func:`make_space` builds the per-partition KLIP parameter block (9 parameters per night, plus
the selection) that NEAR, the generic adapter, NOMIC and ALES search.  It lived in
:mod:`klip_tpe.instruments.near` until 2026-11-01, and ``near.make_space`` and
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
    """The largest parallactic-angle span of any partition (deg)."""
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
    """The production search space: 9 parameters per night (+ drop1/drop2).

    ``width_range=(w0, w1)`` adds the searched global integer ``width`` dim of the
    adaptive-annulus protocol (A §3.1, ``RunConfig.opt_width``; default = the range
    midpoint); ``search_k=False`` leaves the ``k_klip`` dims out (scan k-modes, where k
    comes from the per-eval k-scan).  ``max_drop`` = number of drop slots of the two-slot
    selection (2 = IDL drop1/drop2; raise it when nights x image groups give many
    partitions).  ``bin_range`` = (lo, hi) frames per temporal bin (default the NEAR
    production 5-30; short sequences need a smaller range -- see
    :func:`klip_tpe.instruments.generic.make_space`).

    Two dimensions of the built-in KLIP that the production space leaves at their reference
    values can be searched too (both off by default, so the space is the one it was):
    ``zone_range=(lo, hi)`` adds ``zone_in`` and ``zone_out``, px by which the KLIP zone
    reaches inside and outside the annulus the run scores (0 = the annulus; the reducer's
    ``outrad_cap`` still bounds the outer edge), and ``search_post_hp=True`` adds the 0/1
    ``post_hp``, whether the residuals are high-passed again before derotation (1, the
    reference).  Their defaults come from ``defaults`` (else 0, 0 and 1).

    ``pin={name: value}`` holds reduction parameters at a value for every evaluation: a
    parameter of the block is taken out of the search, and any other name (``zone_in`` without
    ``zone_range``, say) is passed to the reducer as given (``SearchSpace.fixed``).

    ``angsep_range=(lo, hi)`` (lambda/D) replaces the 0-3 that ``angsep`` is searched over.  A
    floor above 0 keeps the search off ``angsep = 0``, which with ``anglemax`` at the PA span is
    another algorithm: :func:`klip_tpe.klip.klip_annular` then builds one basis per zone from all
    the frames, each target frame inside its own basis (the IDL's auto-fast rule), where any
    positive ``angsep`` leaves the target out.  The bound is not a point of measure zero: a
    proposal past it is clipped onto it, and the reference-count guard projects onto 0 too, so
    a search lands there often and keeps it when it scores (MWC 758 with ALES: every winner, at
    the cost of most of the spiral arms' flux).  The default is clipped into the range.

    ``corr_range=(lo, hi)`` replaces the 0-1 that ``corr_thresh`` is searched over.  Frame
    correlations sit just below 1 (MWC 758's ALES frames: 0.944-0.999), so most of 0-1 cuts no
    frame, and the frame-selection guard records every such proposal at the floor, the no-cut
    corner: of the random draws only the few that land in the thin band where the cut acts try
    a cut at all.  A range from just below the data's lowest correlation spends the draws there.
    Its floor is then the no-cut value, so keep it at or below every partition's lowest
    correlation.  The default is clipped into the range."""
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
