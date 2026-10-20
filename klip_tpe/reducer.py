"""The black box being optimized.

A :class:`Reducer` turns a parameter dictionary (plus optional synthetic
injections) into a derotated, combined image.  :class:`KLIPReducer` is the
reference implementation on top of :mod:`klip_tpe.klip`; anything else (pyKLIP,
VIP, a shell script) can be wrapped by subclassing :class:`Reducer` and
implementing :meth:`Reducer.reduce`.

:class:`PartitionedReducer` handles datasets made of several independent
partitions (nights, epochs, detector sides, rolls ...): each selected partition is
reduced with its own parameter block and the results are combined with
``equal`` or ``sqrt_texp`` weights.  This is the multi-night machinery of the IDL
code with the night-specific plumbing removed.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .injection import (InjectionModel, _check_float32_headroom, add_crop, cube_peak, inject_frames,
                        place_source, prepare_sources)
from .klip import (BLOCK_BYTES, KLIPParams, _highpass_frames, arcdist_deg, bin_by_groups, bin_frames,
                   bin_groups, derotate, destripe, frame_blocks, frame_selection_mask, group_bounds, group_mean,
                   highpass_stack, klip_annular, nw_ang_comb, run_blocks)
from .metrics import Source
from .space import Config

__all__ = ["reducer_class", "set_klip_gram", "KLIP_GRAM_MODES", "ReductionRequest", "ReductionResult", "Reducer", "Dataset", "KLIPReducer",
           "PartitionedReducer", "EvalImages"]


@dataclass
class ReductionRequest:
    params: Dict[str, Any]
    injections: Optional[Sequence[Source]] = None
    k_scan: bool = False
    tag: str = ""
    fm_sources: Optional[Sequence[Source]] = None   # KLIP-FM test sources (model cube, never added to the data)
    extras: bool = False                            # also produce the nosub (clean pass) / cadi (injected pass) reference image
    threads: int = 1                                # this partition job's threads: the per-target KLIP loop and the per-frame steps (set by PartitionedReducer)
    #: ``"reduce"`` (the default) or ``"prepare"``: the latter returns the prepared frames
    #: (selected, injected, high-passed, binned) as :class:`SiblingFrames`, for the SDI pool
    #: of the other wavelength groups, and reduces nothing
    stage: str = "reduce"
    #: the other groups' prepared frames for the SDI pool: ``{pid: SiblingFrames}``, or the
    #: shared-memory descriptor :class:`klip_tpe.parallel.SharedArrays` of them
    siblings: Any = None


@dataclass
class SiblingFrames:
    """One wavelength group's frames as another group's reduction needs them for its SDI
    pool: prepared exactly as the group would prepare its own (the same selection,
    injections, high-pass and binning), not yet rescaled.  ``mcube`` is the KLIP-FM model
    cube through the same chain, when the request asked for KLIP-FM."""
    pid: Any
    lam_m: float
    bcube: np.ndarray                 # (n_binned, ny, nx)
    bang: np.ndarray                  # the binned angles
    mcube: Optional[np.ndarray] = None


@dataclass
class PreparedFrames:
    """The first half of :meth:`KLIPReducer.reduce`: everything up to the binned frames,
    before any PSF subtraction."""
    p: Dict[str, Any]
    k: int
    bin_: int
    filt: int
    inrad: float
    outrad: float
    dth: float
    keep: np.ndarray
    bcube: np.ndarray
    bang: np.ndarray
    grp: np.ndarray
    bkeep: np.ndarray
    mcube: Optional[np.ndarray]
    hp_nan: bool
    hp_binned: bool
    nth: int
    meta: Dict[str, Any]
    #: the annulus the zone was made from, ``(inner, outer)`` px with the runner's edge
    #: padding: ``inrad``/``outrad`` are the KLIP zone, which ``zone_in``/``zone_out`` widen
    #: beyond it.  ``angsep``'s arc and the bin's PA span are measured on the annulus.  None
    #: (a caller that builds one itself): the zone is the annulus.
    ann: Optional[Tuple[float, float]] = None

    def siblings_payload(self, pid: Any, lam_m: float) -> SiblingFrames:
        return SiblingFrames(pid, float(lam_m), np.ascontiguousarray(self.bcube, np.float32),
                             np.asarray(self.bang, float),
                             None if self.mcube is None else np.ascontiguousarray(self.mcube, np.float32))


@dataclass
class ReductionResult:
    image: np.ndarray                # (ny, nx) or (nk, ny, nx) when k_scan
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)
    fm_image: Optional[np.ndarray] = None    # KLIP-FM forward-modelled planet response (ny, nx)
    nosub: Optional[np.ndarray] = None       # derotate + median stack, no PSF subtraction (clean pass, extras=True)
    cadi: Optional[np.ndarray] = None        # classical ADI reference (injected pass, extras=True)


class Reducer:
    """Interface every backend implements.

    Class attributes a backend should set:

    ``pxscale`` (arcsec/px), ``fwhm`` (px) -- needed by the metric;
    ``angle_convention`` ``'pa'`` (PA East of North, +90 in the image) or ``'math'``;
    ``supports_kscan`` -- can return a cube over k = 1..k_klip in one call;
    ``lam_over_d_px`` -- used for the ``angsep`` unit and the FWHM.
    """

    pxscale: float = 1.0
    fwhm: float = 4.0
    lam_over_d_px: float = 4.0
    angle_convention: str = "pa"
    supports_kscan: bool = False
    supports_fm: bool = False
    name: str = "reducer"

    def reduce(self, req: ReductionRequest) -> ReductionResult:  # pragma: no cover
        raise NotImplementedError

    def partitions(self) -> List[Any]:
        return []

    def partition_weight(self, pid) -> float:
        return 1.0

    def frame_angles(self, pid=None) -> Optional[np.ndarray]:
        """Per-frame parallactic angles (used by the reference-count guardrail)."""
        return None

    def frame_tags(self, pid=None) -> Optional[Dict[str, np.ndarray]]:
        return None

    def matched_filter_kernel(self, rho_as: float):
        return None

    def cost_estimate(self, params: Dict[str, Any], k_scan: bool = False, n_inject: int = 0,
                      fm: bool = False) -> Optional[float]:
        """Rough single-thread seconds one :meth:`reduce` with ``params`` takes, used only to
        schedule partition jobs (the longest first, and threads by cost).  None: unknown,
        and the jobs keep a flat thread count in submission order."""
        return None

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "pxscale": self.pxscale, "fwhm": self.fwhm,
                "angle_convention": self.angle_convention, "supports_kscan": self.supports_kscan,
                "supports_fm": self.supports_fm}


# ----------------------------------------------------------------------------
# reference KLIP reducer
# ----------------------------------------------------------------------------
@dataclass
class Dataset:
    """One pupil-tracking sequence, already centred on the star."""

    cube: np.ndarray                       # (n, ny, nx) float32
    angles: np.ndarray                     # parallactic angle per frame (deg)
    tags: Optional[Dict[str, np.ndarray]] = None   # corrs / noises / coronoise per frame
    texp: float = 0.0                      # total on-source exposure (s), for weights
    name: str = "data"
    meta: Dict[str, Any] = field(default_factory=dict)
    ref_cube: Optional[np.ndarray] = None  # (nref, ny, nx) pre-registered PSF-reference-star frames (RDI/ARDI)

    @property
    def nframes(self) -> int:
        return int(self.cube.shape[0])


class _FrameCache:
    """One set of sources injected into a cube frame by frame, as reductions ask for frames.

    ``done`` marks the frames injected so far.  The lock covers filling frames and copying
    out a selection, so two reductions of the same sources on threads never read a frame
    another is still writing."""

    __slots__ = ("cube", "done", "lock")

    def __init__(self, cube: np.ndarray):
        self.cube = cube
        self.done = np.zeros(cube.shape[0], bool)
        self.lock = threading.Lock()


class _CropCache:
    """What one set of sources adds to each frame (:func:`klip_tpe.injection.stamp_crop`,
    per source), kept once the set comes back.

    A set seen for the first time is placed without keeping anything: the search draws new
    sources for every reduction, and keeping their stamps would only write them out to
    memory.  A set that comes back -- param-verify reduces one fixed set under many
    configurations, a fixed-source search one set per annulus -- keeps each frame's stamps
    from its second reduction on, so a later selection places only the frames it adds.  One
    set is kept at a time, the latest that came back."""

    __slots__ = ("seen", "key", "frames", "lock")

    def __init__(self):
        self.seen = None
        self.key = None
        self.frames: Dict[int, list] = {}
        self.lock = threading.Lock()

    def lookup(self, key) -> Tuple[bool, Optional[Dict[int, list]]]:
        """``(first_sighting, frames)``: whether ``key`` is new, and the dict its frames'
        stamps go in, or None when they are not kept."""
        with self.lock:
            if key == self.key:
                return False, self.frames
            if key == self.seen:
                self.key, self.frames = key, {}
                return False, self.frames
            self.seen = key
            return True, None


def _flux_taken(orig: np.ndarray, edited: np.ndarray, angles: np.ndarray, sources: Sequence[Source], model,
                pxscale: float, *, truenorth: float, fallback, angle_convention: str,
                n_frames: int = 32) -> Tuple[float, int]:
    """The fraction of the ``sources``' flux that left the frames when they were subtracted
    (1 = all of it), measured on up to ``n_frames`` frames spread over the cube:
    ``sum(sign(m) (orig - edited)) / sum(|m|)``, with ``m`` the sources injected into zeros.
    For sources added, pass the frames the other way round.  float32 rounding loses the part
    of a source that sits below half a unit in the last place of the pixel it lands on.
    Returns ``(fraction, frames measured)``."""
    n = int(orig.shape[0])
    idx = np.unique(np.linspace(0, n - 1, min(n, max(int(n_frames), 1))).round().astype(np.intp))
    m = np.zeros((idx.size,) + tuple(orig.shape[-2:]), np.float32)
    inject_frames(m, m, np.asarray(angles, float)[idx], list(sources), model, pxscale, None, truenorth=truenorth,
                  fallback=fallback, angle_convention=angle_convention)
    took = (np.asarray(orig[idx], np.float32).astype(np.float64)
            - np.asarray(edited[idx], np.float32).astype(np.float64))
    m64 = m.astype(np.float64)
    ok = np.isfinite(took) & (m64 != 0.0)
    den = float(np.sum(np.abs(m64[ok])))
    if not den > 0.0:
        return float("nan"), int(idx.size)
    return float(np.sum(np.sign(m64[ok]) * took[ok]) / den), int(idx.size)


def _taken_text(red, attr: str = "subtracted_kept") -> str:
    """``n4 0.9999998, n6 0.9999997`` (or one number): what :func:`_flux_taken` measured for
    the sources a reducer subtracted (``attr="injected_kept"``: added), '' when it measured
    nothing."""
    kept = getattr(red, attr, None)
    items = list(kept.items()) if isinstance(kept, dict) else [(None, kept)]
    out = [(k, v) for k, v in items if v is not None and np.isfinite(v)]
    return ", ".join(f"{k} {v:.7f}" if k is not None else f"{v:.7f}" for k, v in out)


def _flag(v) -> bool:
    """A yes/no request parameter: True for True, 1, "1", "yes", "true"."""
    if isinstance(v, str):
        return v.strip().lower() in ("1", "yes", "true", "y", "on")
    return bool(v)


def _zone_ext(p: Dict[str, Any]) -> Tuple[float, float]:
    """``(zone_in, zone_out)`` of a request's parameters, px >= 0: how far the KLIP zone
    reaches inside and outside the annulus (0, 0 when the request does not say)."""
    return (max(float(p.get("zone_in", 0.0) or 0.0), 0.0), max(float(p.get("zone_out", 0.0) or 0.0), 0.0))


class KLIPReducer(Reducer):
    """Reference annular KLIP/ADI reducer.

    Parameters (all optional, defaults = ``reduce_near_2`` behaviour)::

        k_klip, bin, filter, n_ang, inrad, outrad, angsep, anglemax,
        corr_thresh, noise_max, coronoise_max, fast, spat_mean, temp_mean, n_min_ref,
        comb_type ('nwadi' | 'mean' | 'median'),
        use_rdi (bool), rdi_mode ('rdi' | 'ardi'), do_destripe (bool), destripe_clip (float),
        fm_selfsub (bool, KLIP-FM self-subtraction terms; default True),
        hp_nan_aware (bool, default False: the high-pass of the science frames is IDL's
            ``smooth`` without ``/nan``, and a frame with a NaN pixel anywhere comes out
            NaN all over, since the boxcar's running sum carries the NaN along its line.
            True is ``smooth(/nan)``: each NaN pixel stays NaN and the rest of the frame is
            filtered.  For frames with NaN at their registration edges, say ALES's, where
            False drops every bin with such a frame.)
        prep_fused (bool, default True: the frames before KLIP are made one bin group at a
            time, see below.  False makes them a whole cube per step, the reference the
            fused path is tested against, to the bit.)
        klip_gram ('target' | 'zone', default 'target': how the per-target KLIP forms each
            target's reference covariance, :attr:`klip_tpe.klip.KLIPParams.gram`.  'zone'
            reads it off one Gram matrix per zone and projects without forming the modes,
            the same KLIP to float64 rounding and about twice as fast per target at small
            bins, not the same to the bit.  :func:`set_klip_gram` sets it on every reducer
            of a partitioned one.)
        zone_in, zone_out (px, default 0: the KLIP zone is the annulus.  The zone's inner
            edge moves in by zone_in and its outer edge out by zone_out, so the basis is
            fitted over a wider ring than the one the run scores, as the ALES paper reduction
            of MWC 758 fits one 5-27 px annulus.  ``angsep``'s lambda/D of arc and the bin's
            PA span stay those of the annulus, where the sources are.)
        post_hp (bool, default True: the KLIP residuals are high-passed again, with the same
            box, before they are derotated, as ``reduce_near_2`` does.  False derotates them
            as they come out of KLIP, as the ALES reduction's ADI images of MWC 758 do.)

    The chain: frame selection -> inject (the kept frames) -> high-pass -> [destripe 90, 0]
    -> angle-aware binning -> annular KLIP [RDI/ARDI basis, KLIP-FM] -> [high-pass
    (NaN-aware)] -> derotate -> combine [-> mask_fn].  The selection reads only the frame
    tags, so injecting the kept frames alone changes no number.

    Everything before KLIP is done one bin group at a time (:meth:`_prep_fused`): a group's
    kept frames are copied out of the cube, given their sources, high-passed, destriped and
    averaged into the one binned frame, while they are still in the processor's cache.  The
    frames are read from memory once per reduction, where a whole-cube step per stage read
    and wrote them seven times, and no injected cube is ever held.  Each step is the
    whole-cube step's own function on the same frames in the same order, so the numbers are
    the same to the bit.

    ``mask_fn(image) -> image`` is the optional legacy post-mask hook
    (``block_burn`` / ``block_airy``), applied to the final science image only.
    """

    name = "klip"
    supports_kscan = True
    supports_fm = True
    #: A backend that overrides :meth:`_subtract` builds its own basis and does NOT apply a
    #: searched reference library unless it declares it here.  pyKLIP, VIP and the custom
    #: ``FunctionReducer`` all override it; pyKLIP in particular hands the WHOLE reference
    #: cube to ``klip_parallelized`` and lets it pick.
    _honours_reference_library = False

    @property
    def supports_reference_library(self) -> bool:
        """True only where ``nkeep_<group>`` actually changes the reduction: the built-in
        annular KLIP (whose ``_subtract`` passes ``kp.ref_lib`` / ``kp.ref_keep`` to
        :func:`klip_tpe.klip.klip_annular`), or a backend that says it applies them."""
        return (type(self)._subtract is KLIPReducer._subtract
                or bool(getattr(type(self), "_honours_reference_library", False)))

    def __init__(self, data: Dataset, pxscale: float, lam_m: float, diam_m: float,
                 injection_model: Optional[InjectionModel] = None,
                 fallback_model: Optional[InjectionModel] = None, truenorth: float = 0.0,
                 fwhm_px: Optional[float] = None, zone_pad: float = 2.0, outrad_cap: float = 70.0,
                 defaults: Optional[Dict[str, Any]] = None, angle_convention: str = "pa",
                 mask_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.data = data
        self.pxscale = float(pxscale)
        self.lam_m, self.diam_m = float(lam_m), float(diam_m)
        self.lam_over_d_px = (lam_m / diam_m) * 206265.0 / pxscale
        self.fwhm = float(fwhm_px) if fwhm_px else 1.028 * self.lam_over_d_px
        self.model = injection_model
        self.fallback = fallback_model
        self.truenorth = float(truenorth)
        self.zone_pad = float(zone_pad)
        self.outrad_cap = float(outrad_cap)
        self.angle_convention = angle_convention
        self.mask_fn = mask_fn
        self.defaults = {"k_klip": 10, "bin": 1, "filter": 0, "n_ang": 1, "inrad": 0.0,
                         "outrad": outrad_cap, "angsep": 0.0, "anglemax": 360.0,
                         "corr_thresh": 0.95, "noise_max": 2.0, "coronoise_max": 2.0,   # reduce_near_2 defaults; 2.0 = historical no-op
                         "fast": False, "spat_mean": False, "temp_mean": False,
                         "n_min_ref": 10, "comb_type": "nwadi",
                         "use_rdi": False, "rdi_mode": "ardi", "do_destripe": False, "destripe_clip": 0.0,
                         "fm_selfsub": True, "hp_nan_aware": False, "prep_fused": True,
                         "klip_gram": "target"}
        self.defaults.update(defaults or {})
        self._inj_cache: Dict[Tuple, _FrameCache] = {}
        self._fm_cache: Dict[Tuple, _FrameCache] = {}
        #: the fused path's stamps, per kind: the injected sources and the KLIP-FM model's
        self._crops: Dict[str, _CropCache] = {"inj": _CropCache(), "fm": _CropCache()}
        #: the level of the cube's brightest pixels for the float32 headroom check, computed once
        self._peak: Optional[float] = None
        self._inj_lock = threading.Lock()
        self._cost_cache: Dict[Tuple, float] = {}
        #: optional searched reference library (:mod:`klip_tpe.reflib`).  Set through
        #: :meth:`set_reference_library`; when it is None every RDI/ARDI reduction uses
        #: the whole reference cube as one shared basis, which is the previous behaviour.
        self._reflib_spec: Optional[Dict[str, Any]] = None
        self._reflib_cache: Dict[Tuple, Any] = {}

    # -- helpers -------------------------------------------------------------
    def partitions(self) -> List[Any]:
        return []

    def frame_angles(self, pid=None):
        return self.data.angles

    def frame_tags(self, pid=None):
        return self.data.tags

    def partition_weight(self, pid=None) -> float:
        return float(np.sqrt(self.data.texp)) if self.data.texp > 0 else 1.0

    def matched_filter_kernel(self, rho_as: float):
        if self.model is None:
            return None
        return self.model.matched_filter_kernel(rho_as, self.fwhm)

    def dth_max_deg(self, outrad: float) -> float:
        """Max PA span inside one temporal bin: half a FWHM (``1.028 lambda/D``) of
        motion at ``outrad``."""
        return float(np.rad2deg(0.5 * 1.028 * self.lam_over_d_px / max(outrad, 1.0)))

    def cost_estimate(self, params: Dict[str, Any], k_scan: bool = False, n_inject: int = 0,
                      fm: bool = False) -> Optional[float]:
        """Rough single-thread seconds of one :meth:`reduce`, for scheduling only.

        From the frames kept, the frames after binning and the KLIP zone: the injection and
        the high-pass of every kept frame, the derotation and combination of every binned
        frame, and the per-target KL bases (binned frames x references squared x zone
        pixels).  The
        coefficients come from 2024-frame, 150 x 150 timings on one 2.8 GHz core, and only
        the ratios between jobs are used."""
        try:
            p = dict(self.defaults)
            p.update({k: v for k, v in params.items() if v is not None})
            k = int(max(round(p["k_klip"]), 1))
            bin_ = int(max(round(p["bin"]), 1))
            filt = int(round(p["filter"]))
            n_ang = int(max(round(p["n_ang"]), 1))
            inrad = max(float(p["inrad"]), 0.0)
            outrad = min(float(p["outrad"]), self.outrad_cap)
            z_in, z_out = _zone_ext(p)                  # the KLIP zone beyond the annulus
            zin, zout = max(inrad - z_in, 0.0), min(outrad + z_out, self.outrad_cap)
            # a searched library's counts: more rows in every target's basis, and the SDI
            # pool's rescaling of every sibling frame
            spec = self._reflib_spec or {}
            extra = 0
            n_sdi = 0
            if spec:
                for name in ["sdi"] + list((spec.get("groups") or {spec.get("ref_group", "psfref"): None}).keys()):
                    v = p.get(f"nkeep_{name}")
                    if v is not None:
                        extra += int(max(round(float(v)), 0))
                        if name == "sdi":
                            n_sdi = int(max(round(float(v)), 0))
            key = (k, bin_, filt > 1, n_ang, round(inrad, 2), round(outrad, 2), p["corr_thresh"], p["noise_max"],
                   p["coronoise_max"], round(float(p["angsep"]), 3), round(float(p["anglemax"]), 2),
                   bool(p["fast"]), bool(k_scan), int(n_inject), bool(fm), int(extra), int(n_sdi),
                   round(zin, 2), round(zout, 2))
            hit = self._cost_cache.get(key)
            if hit is not None:
                return hit
            ds = self.data
            keep = frame_selection_mask(ds.nframes, ds.tags, p["corr_thresh"], p["noise_max"],
                                        p["coronoise_max"], bin_, k)
            ang = np.asarray(ds.angles, float)[keep]
            n_kept = int(ang.size)
            m = int(bin_groups(ang, bin_, self.dth_max_deg(outrad)).max()) + 1 if n_kept else 0
            npix = int(np.prod(ds.cube.shape[1:]))
            zone = np.pi * max(zout ** 2 - zin ** 2, 1.0)
            span = float(ang.max() - ang.min()) if n_kept > 1 else 0.0
            sep = float(p["angsep"]) * abs(arcdist_deg(inrad, outrad, self.lam_over_d_px))
            amax = float(p["anglemax"])
            if (bool(p["fast"]) or (sep <= 0 and amax >= span)) and extra == 0:
                klip = 2.2e-10 * m * m * zone                        # one basis per sector
            else:
                frac = 1.0 if span <= 0 else min(1.0, max(2.0 * (amax - sep), 0.0) / span)
                mref = max(min(float(m), m * frac), 4.0) + extra
                klip = 2.2e-10 * m * mref * mref * zone + 2e-4 * m * n_ang
            if n_sdi > 0 and spec.get("sdi"):
                lenders = max(len(spec["sdi"].get("lenders") or []) - 1, 1)
                klip += 1.3e-7 * lenders * m * npix * (2 if fm else 1)   # the rescaling of every sibling frame
            raw = (1.5e-8 if filt > 1 else 2e-9) * n_kept * npix     # high-pass, copies, binning
            post = 3.6e-7 * m * npix                                 # post high-pass, derotate, combine
            cost = raw + post * (k if k_scan else 1) + klip
            if fm:
                cost += raw + post + klip                            # the model cube follows the data
            if n_inject:
                cost += 1.2e-4 * n_inject * n_kept                    # each source into each kept frame
            if len(self._cost_cache) > 4096:
                self._cost_cache.clear()
            self._cost_cache[key] = float(cost)
            return float(cost)
        except Exception:
            return None

    def _inject(self, sources: Sequence[Source], keep: Optional[np.ndarray] = None,
                threads: int = 1) -> np.ndarray:
        """The cube with ``sources`` injected, every frame; with ``keep``, the kept frames
        only (``cube[keep]``, in order).

        Only the frames asked for are injected.  The frame selection depends on the tags,
        not on the injection, so a frame it drops never needs its companions.  The latest
        set of sources stays cached with the frames injected so far, so a reduction with
        the same sources and another selection injects only the frames it adds.  Each frame
        gets the numbers :func:`klip_tpe.injection.inject_sources` gives it."""
        key = tuple(s.as_tuple() for s in sources)
        with self._inj_lock:
            ent = self._inj_cache.get(key)
            if ent is None:
                if self.model is None:
                    raise ValueError("KLIPReducer needs an injection_model to inject sources")
                # frames are written as they are asked for: a frame never asked for costs no
                # memory, and no frame is read before it is written
                ent = _FrameCache(np.empty(self.data.cube.shape, np.float32))
                self._inj_cache = {key: ent}       # keep only the latest set of sources
                _check_float32_headroom(self.data.cube, sources, self.model, self.fallback,
                                        peak=self.cube_peak())
        return self._fill(ent, self.data.cube, sources, keep, threads)

    def model_cube(self, sources: Sequence[Source], keep: Optional[np.ndarray] = None,
                   threads: int = 1) -> np.ndarray:
        """KLIP-FM planet-model cube: the sources injected into a ZERO cube with the
        same injection code / conventions as the data (reduce_near_2 L1441-1494).
        ``keep``: the kept frames only, as :meth:`_inject`."""
        key = tuple(s.as_tuple() for s in sources)
        with self._inj_lock:
            ent = self._fm_cache.get(key)
            if ent is None:
                if self.model is None:
                    raise ValueError("KLIPReducer needs an injection_model to build the FM model cube")
                ent = _FrameCache(np.zeros(self.data.cube.shape, np.float32))
                self._fm_cache = {key: ent}
        return self._fill(ent, ent.cube, sources, keep, threads)

    def _fill(self, ent: _FrameCache, base: np.ndarray, sources: Sequence[Source],
              keep: Optional[np.ndarray], threads: int) -> np.ndarray:
        """Inject the frames of ``keep`` (all by default) that ``ent`` does not hold yet,
        from ``base``, and return the frames asked for."""
        want = np.ones(ent.done.size, bool) if keep is None else np.asarray(keep, bool)
        with ent.lock:
            todo = np.flatnonzero(want & ~ent.done)
            if todo.size:
                inject_frames(ent.cube, base, self.data.angles, sources, self.model, self.pxscale, todo,
                              truenorth=self.truenorth, fallback=self.fallback,
                              angle_convention=self.angle_convention, threads=threads)
                ent.done[todo] = True
            return ent.cube if want.all() else ent.cube[want]

    def __getstate__(self):
        # the injection caches are scratch, and a lock does not pickle
        d = self.__dict__.copy()
        d.update(_inj_cache={}, _fm_cache={})
        d.pop("_inj_lock", None)
        d.pop("_crops", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._inj_lock = threading.Lock()
        self._crops = {"inj": _CropCache(), "fm": _CropCache()}
        self.__dict__.setdefault("_peak", None)

    def cube_peak(self) -> float:
        """The level of the cube's brightest pixels as the float32 headroom check samples it
        (:func:`~klip_tpe.injection.cube_peak`), computed
        once.  The check runs for every new set of sources, and its strided sample touches
        most of the cube's cache lines: once per draw it was most of a read of the cube."""
        if getattr(self, "_peak", None) is None:
            self._peak = cube_peak(self.data.cube)
        return self._peak

    def subtract_sources(self, sources: Sequence[Any], threads: int = 1) -> int:
        """Take point sources out of the data once, before any reduction.

        Each source goes into every frame of ``data.cube`` with the negative of its contrast,
        through :func:`~klip_tpe.injection.inject_frames` with this reducer's model, fallback,
        true north, centre and angle convention.  Those are the numbers a draw's injection of
        the same source adds, with the sign flipped.  Every reduction then works on the edited
        frames: the clean pass, each draw, the reference ranking (:meth:`_clean_bins`),
        validation and param-verify.  The KLIP-FM model cube does not, since it is built from
        zeros (:meth:`model_cube`).

        ``sources``: :class:`~klip_tpe.metrics.Source` or ``(rho_arcsec, pa_deg, contrast)``,
        e.g. :func:`~klip_tpe.injection.read_sources`.  A source of contrast 0 is skipped, so a
        list of zeros leaves the frames as they were, to the bit.  The edited frames are a copy:
        a dataset the caller still holds is not changed.  Call it before the worker pool starts
        (:meth:`PartitionedReducer.subtract_sources` checks), since the workers keep the
        reducers they were forked with.  The edit is kept in :attr:`data_edits` and in
        :meth:`describe`.

        The flux that left the frames is measured on a sample of frames against the sources
        injected into zeros, and kept as :attr:`subtracted_kept` (1 = all of it).  It warns when
        float32 rounding kept more than 0.1 % of it, in place of the per-source headroom check a
        draw's injection gets: in a chain whose contrasts pass through 0, the faintest sources
        lose part of their own flux and carry almost none of the chain's.  Returns the number of
        sources subtracted."""
        src = [s if isinstance(s, Source) else Source(*(float(v) for v in s)) for s in sources]
        n, kept = self._edit_frames(src, -1.0, threads, "subtract_sources")
        if n:
            self.subtracted_kept = kept
        rec = dict(getattr(self, "data_edits", None) or {})
        rec["subtracted_sources"] = list(rec.get("subtracted_sources") or []) + [list(s.as_tuple()) for s in src]
        self.data_edits = rec
        return n

    def inject_sources(self, sources: Sequence[Any], threads: int = 1) -> int:
        """Add point sources to the data once, before any reduction: planets put into the
        frames on purpose, to see what a reduction does to them.

        :meth:`subtract_sources` with the opposite sign.  Each source goes into every frame of
        ``data.cube`` with its own contrast, through the code a draw's injection uses, and from
        then on it is part of the data: the clean pass, each draw, the reference ranking,
        validation and param-verify see it, and the KLIP-FM model cube does not.  The flux that
        reached the frames is measured as for a subtraction (:attr:`injected_kept`).  The edit
        is kept in :attr:`data_edits` as ``injected_sources``.  After :meth:`subtract_sources`
        when both are asked for.  Returns the number of sources added."""
        src = [s if isinstance(s, Source) else Source(*(float(v) for v in s)) for s in sources]
        n, kept = self._edit_frames(src, 1.0, threads, "inject_sources")
        if n:
            self.injected_kept = kept
        rec = dict(getattr(self, "data_edits", None) or {})
        rec["injected_sources"] = list(rec.get("injected_sources") or []) + [list(s.as_tuple()) for s in src]
        self.data_edits = rec
        return n

    def _edit_frames(self, src: Sequence[Source], sign: float, threads: int, what: str) -> Tuple[int, float]:
        """``src`` into every frame of ``data.cube`` at ``sign`` times its contrast (-1 takes
        the sources out, +1 puts them in), on a float32 copy, and the fraction of their flux
        that moved (:func:`_flux_taken`).  Returns ``(sources applied, fraction)``."""
        ny, nx = self.data.cube.shape[-2:]
        far = [s for s in src if s.rho / self.pxscale > max(ny, nx)]
        if far:
            raise ValueError(f"source at rho={far[0].rho:g} arcsec is {far[0].rho / self.pxscale:.0f} px from the "
                             f"star, beyond the {nx} x {ny} px frame: rho is in arcsec (pixel scale "
                             f"{self.pxscale:g} arcsec/px)")
        own = [s for s in src if float(s.contrast) != 0.0]
        todo = [Source(s.rho, s.theta, sign * float(s.contrast)) for s in own]
        if not todo:
            return 0, float("nan")
        if self.model is None:
            raise ValueError(f"{what} needs an injection_model")
        from dataclasses import replace
        orig = self.data.cube
        cube = np.array(orig, dtype=np.float32, order="C", copy=True)
        inject_frames(cube, cube, self.data.angles, todo, self.model, self.pxscale, None,
                      truenorth=self.truenorth, fallback=self.fallback,
                      angle_convention=self.angle_convention, threads=max(int(threads), 1))
        before, after = (orig, cube) if sign < 0 else (cube, orig)
        kept, k = _flux_taken(before, after, self.data.angles, own, self.model, self.pxscale,
                              truenorth=self.truenorth, fallback=self.fallback, angle_convention=self.angle_convention)
        if np.isfinite(kept) and abs(kept - 1.0) > 1e-3:
            warnings.warn(f"{what}: {kept:.5f} of the sources' flux {'left' if sign < 0 else 'reached'} the frames "
                          f"(measured on {k} of {orig.shape[0]} frames), float32 rounding "
                          f"{'kept' if sign < 0 else 'lost'} the rest.  flux_unit is {self.model.flux_unit:.4g} -- if "
                          f"that is a normalised template's own sum rather than the star's flux in the science "
                          f"frames' units, set star_flux (see generic.star_flux_from_aperture_photometry).",
                          RuntimeWarning, stacklevel=3)
        self.data = replace(self.data, cube=cube)
        # everything worked out from the old frames goes
        self._peak = None
        self._inj_cache, self._fm_cache = {}, {}
        self._crops = {"inj": _CropCache(), "fm": _CropCache()}
        self._reflib_cache = {}
        return len(todo), kept

    def set_reference_library(self, *, partition, groups=None, metric: str = "cc",
                             shift_px: int = 1, n_min_ref: int = 2,
                             min_keep: Optional[Dict[str, int]] = None,
                             ref_group: str = "psfref",
                             similarity: Optional[np.ndarray] = None,
                             rotation: bool = False,
                             searched: Optional[Sequence[str]] = None,
                             sdi: Optional[Dict[str, Any]] = None) -> None:
        """Make the reference library a searched object rather than a fixed basis.

        ``partition`` is one label per *unbinned* science frame -- the roll, night or
        channel a frame belongs to.  A target never draws references sharing its label, so
        this is what stops a roll subtracting its own companion.  ``groups`` optionally
        splits ``data.ref_cube`` into several named external pools (name -> row indices);
        the default makes it one pool called ``psfref``.

        ``rotation`` makes the science pool one the ADI rule selects from, per target and
        per annulus (``angsep``, ``anglemax`` over the binned angles, the label on top when
        there is more than one): the form for one continuously rotating sequence, which has
        no label to exclude by.  With every eligible science frame kept and
        ``nkeep_<ref_group> = 0`` the reduction is plain ADI to the bit, so "no reference
        frames" is a point of the search space.

        ``searched`` names the pools whose counts become search dimensions (default: every
        pool).  A pool left out is not a dimension at all: every one of its eligible frames
        is in the basis, as it is without a library.  ``searched=("psfref",)`` with
        ``rotation`` is ADI plus a searched number of reference-star frames, the ALES form.

        ``sdi`` adds the pool of the other wavelength groups' frames, rescaled about the star
        to this group's wavelength (:mod:`klip_tpe.rescale`), ranked by correlation with the
        target and counted by ``nkeep_sdi``: ``{"n_frames": total frames the other groups
        hold (for the count's range), "cap": most-correlated rows the count may reach,
        "lenders": their ids}``.  The frames themselves arrive with each request
        (``ReductionRequest.siblings``, prepared by the groups' own :meth:`prepare`, which
        :class:`PartitionedReducer` arranges), so this only declares the pool.  A target's
        eligible SDI rows are the ones whose planet has moved, by rotation and scaling
        together, the arc ``angsep`` asks for (the displacement rule,
        :func:`klip_tpe.rescale.displacement_px`), within ``anglemax`` of rotation.

        The similarity matrix depends on the binned cube, the high-pass width and the
        annulus, all of which the search can move, so it is built lazily and cached per
        distinct combination.  With ``bin`` fixed -- which is how the MWC 758 run was
        configured -- that is exactly one matrix for the whole search.

        ``similarity`` supplies a ``(n_library, n_target)`` matrix instead of computing
        one.  That is how a port is checked against the thing it was ported from: hand it
        the IDL's own ``F430M_refmetrics_cc.fits`` and any difference in the result is a
        difference in the SELECTION, not in how two languages measure correlation.
        """
        if not self.supports_reference_library:
            # This used to succeed on every backend: the counts were added to the search
            # space from here, and a backend with its own basis then ignored them.  GO 1386
            # MIRI run v6 searched nkeep_altroll / nkeep_psfref for 600 evaluations on
            # pyKLIP, and four libraries from pure ADI to pure RDI gave bit-identical scores
            # on 40 common draws -- the "elected" counts were where TPE left two dead
            # dimensions.  A searched library that is not applied must not be searchable.
            raise NotImplementedError(
                f"{type(self).__name__} ({self.name!r}) does not apply a searched reference "
                f"library: its _subtract builds its own basis from the whole reference cube, "
                f"so nkeep_* would be searched and change nothing.  Use the built-in engine "
                f"(backend 'klip') for a searched library, or search this backend's own "
                f"selection instead (pyKLIP: its ADI / RDI / ADI+RDI mode).")
        pools = ["altroll"] + (["sdi"] if sdi else []) + list((groups or {ref_group: None}).keys())
        if sdi:
            sdi = {"n_frames": int(sdi.get("n_frames", 0)), "cap": int(sdi.get("cap", 100)),
                   "lenders": list(sdi.get("lenders", []))}
            if sdi["n_frames"] < 1 or sdi["cap"] < 1:
                raise ValueError(f"an SDI pool needs frames and a cap above 0, got {sdi}")
        if searched is not None:
            searched = tuple(str(s) for s in searched)
            bad = [s for s in searched if s not in pools]
            if bad:
                raise ValueError(f"searched pools {bad} are not among {pools}")
        self._reflib_spec = {"partition": np.asarray(list(partition)), "groups": groups,
                             "metric": metric, "shift_px": int(shift_px),
                             "n_min_ref": int(n_min_ref), "min_keep": dict(min_keep or {}),
                             "ref_group": ref_group, "rotation": bool(rotation), "searched": searched,
                             "sdi": sdi or None,
                             "similarity": None if similarity is None else np.asarray(similarity, float)}
        self._reflib_cache.clear()

    def reference_params(self):
        """The searched counts this reducer's library contributes to the space (the
        ``searched`` pools of :meth:`set_reference_library`; a pool left out keeps every
        eligible frame and is no dimension)."""
        if self._reflib_spec is None or not self.supports_reference_library:
            return []
        searched = self._reflib_spec.get("searched")
        return [p for p in self._reflib_stub().params()
                if searched is None or p.name[len("nkeep_"):] in searched]

    def _reflib_stub(self):
        """A library over the UNBINNED geometry, for building the search space before any
        reduction has run.  Only the group sizes matter here, so the similarity is a
        placeholder of the right shape."""
        from .reflib import ReferenceGroup, ReferenceLibrary
        spec = self._reflib_spec
        n_sci = int(self.data.cube.shape[0])
        ref = self.data.ref_cube
        n_ref = 0 if ref is None else int(np.asarray(ref).shape[0])
        groups = [ReferenceGroup("altroll", np.arange(n_sci), partition=spec["partition"],
                                 min_keep=spec["min_keep"].get("altroll", 0),
                                 rotation=bool(spec.get("rotation", False)))]
        n_sdi = 0
        if spec.get("sdi"):
            n_sdi = int(spec["sdi"]["n_frames"])
            groups.append(ReferenceGroup("sdi", n_sci + np.arange(n_sdi), min_keep=spec["min_keep"].get("sdi", 0),
                                         max_keep=min(int(spec["sdi"]["cap"]), n_sdi), scales=np.ones(n_sdi)))
        for name, rows in (spec["groups"] or {spec["ref_group"]: np.arange(n_ref)}).items():
            rows = np.asarray(rows, int)
            if rows.size:
                groups.append(ReferenceGroup(name, n_sci + n_sdi + rows,
                                             min_keep=spec["min_keep"].get(name, 0)))
        ruled = bool(spec.get("rotation")) or n_sdi > 0
        return ReferenceLibrary(groups, np.zeros((n_sci + n_sdi + n_ref, n_sci)),
                                target_partition=spec["partition"], metric=spec["metric"],
                                n_min_ref=spec["n_min_ref"],
                                angles=np.concatenate([np.asarray(self.data.angles, float), np.zeros(n_sdi)])
                                if ruled else None,
                                mid_px=1.0 if n_sdi else None)

    def _reflib(self, bcube, ref, bkeep_partition, filt, inrad, outrad, key_extra=(), sim_cube=None,
                bang=None, sdi=None, mid_px=None):
        """The library for this evaluation: ``(selection, basis)``.  ``bang``: the binned
        angles, which a ``rotation`` science pool selects by (they follow from the key).
        ``mid_px``: the radius the SDI pool's displacement rule measures at (default the
        mid radius of ``inrad``-``outrad``; the annulus' when the zone reaches beyond it).

        The SELECTION -- the groups and the similarity matrix that ranks each target's
        references -- is built once per distinct (binning, frame selection, filter, annulus)
        and reused; ``key_extra`` carries the binning and the frame-selection mask, and
        ``sim_cube`` is the same binned cube WITHOUT injections, so that the ranking does not
        depend on what was injected or on which evaluation came first.  The BASIS rows are
        always this evaluation's own cube, ``[bcube | SDI rows | ref]``.

        ``sdi`` (from :meth:`_sdi_rows`) is this evaluation's SDI pool: its rows go between
        the science frames and the reference star's, and its block of the similarity is
        computed here and now, against this evaluation's own target frames at zero lag,
        never cached.  The rows are rescaled frames of the same registered sequence, so the
        +/- 1 px search the other pools use has nothing to find there, and they carry this
        evaluation's injections: a cached ranking would make the selection depend on which
        evaluation came first, the very thing the cache rule below exists to prevent, and a
        second, clean prep of every sibling would double the rescaling.  The injections'
        footprint in the ranking is a few pixels of the thousands in the annulus.

        Until 2026-09-27 the cache held the basis rows too, keyed without the injections:
        every later reduction with the same key took its science (other-roll) references from
        whichever reduction had filled the cache first -- another evaluation's injections, or
        none -- so an injected source was not self-subtracted by the other roll, a clean
        reduction carried another evaluation's sources, and a result depended on the order of
        the reductions.  It inflated every configuration that used the other roll on the
        built-in engine's searched library (NIRCam DK, D2K, H2K; MIRI on the built-in engine);
        pyKLIP, which builds its own basis, and the libraries without an other-roll pool were
        untouched."""
        from .reflib import ReferenceGroup, ReferenceLibrary, similarity_matrix
        spec = self._reflib_spec
        n_sci = int(bcube.shape[0])
        n_ref = 0 if ref is None else int(ref.shape[0])
        library = bcube if ref is None else np.concatenate([bcube, ref], axis=0)
        key = (n_sci, int(filt), round(float(inrad), 3), round(float(outrad), 3),
               spec["metric"], spec["shift_px"]) + tuple(key_extra)
        lib = self._reflib_cache.get(key)
        if lib is None:
            rot = bool(spec.get("rotation", False))
            groups = [ReferenceGroup("altroll", np.arange(n_sci), partition=bkeep_partition,
                                     min_keep=spec["min_keep"].get("altroll", 0), rotation=rot)]
            for name, rows in (spec["groups"] or {spec["ref_group"]: np.arange(n_ref)}).items():
                rows = np.asarray(rows, int)
                if rows.size:
                    groups.append(ReferenceGroup(name, n_sci + rows,
                                                 min_keep=spec["min_keep"].get(name, 0)))
            if rot and (bang is None or np.asarray(bang).size != n_sci):
                raise ValueError("a rotation library needs one binned angle per science frame")
            if spec.get("similarity") is not None:
                sim = spec["similarity"]
                if sim.shape != (library.shape[0], n_sci):
                    raise ValueError(f"supplied similarity is {sim.shape}, expected "
                                     f"{(library.shape[0], n_sci)} (library x target)")
            else:
                sc = bcube if sim_cube is None else sim_cube
                if sc.shape != bcube.shape:
                    raise ValueError(f"similarity cube {sc.shape} does not match the binned cube {bcube.shape}")
                sim = similarity_matrix(sc, sc if ref is None else np.concatenate([sc, ref], axis=0),
                                        inrad=inrad, outrad=outrad, filt=0.0,
                                        metric=spec["metric"], shift_px=spec["shift_px"])
            lib = ReferenceLibrary(groups, sim, target_partition=bkeep_partition,
                                   metric=spec["metric"], n_min_ref=spec["n_min_ref"],
                                   angles=np.asarray(bang, float) if rot else None)
            self._reflib_cache[key] = lib
        if sdi is None:
            return lib, library
        # this evaluation's SDI rows between the science frames and the reference star's
        n_sdi = int(sdi["rows"].shape[0])
        cap = int(spec["sdi"]["cap"]) if spec.get("sdi") else n_sdi
        groups = []
        for g in lib.groups:
            if g.name == "altroll":
                groups.append(ReferenceGroup(g.name, g.rows, partition=g.partition, min_keep=g.min_keep,
                                             max_keep=g.max_keep, rotation=g.rotation))
            else:
                groups.append(ReferenceGroup(g.name, g.rows + n_sdi, partition=g.partition, min_keep=g.min_keep,
                                             max_keep=g.max_keep))
        groups.insert(1, ReferenceGroup("sdi", n_sci + np.arange(n_sdi), min_keep=spec["min_keep"].get("sdi", 0),
                                        max_keep=min(cap, n_sdi), scales=sdi["scales"]))
        sim_sdi = similarity_matrix(bcube, sdi["rows"], inrad=inrad, outrad=outrad, filt=0.0,
                                    metric=spec["metric"], shift_px=0)
        sim = np.concatenate([lib.similarity[:n_sci], sim_sdi, lib.similarity[n_sci:]], axis=0)
        ang = np.asarray(bang, float) if lib.angles is None else np.asarray(lib.angles, float)[:n_sci]
        full = ReferenceLibrary(groups, sim, target_partition=bkeep_partition, metric=spec["metric"],
                                n_min_ref=spec["n_min_ref"],
                                angles=np.concatenate([ang, np.asarray(sdi["angles"], float)]),
                                mid_px=max(float(inrad) + abs(float(outrad) - float(inrad)) / 2.0, 1.0)
                                if mid_px is None else max(float(mid_px), 1.0))
        library = np.concatenate([bcube, np.asarray(sdi["rows"], bcube.dtype)] + ([ref] if ref is not None else []),
                                 axis=0)
        return full, library

    def _binned_partition(self, keep, grp, bkeep):
        """Partition label per surviving BINNED frame.

        Three maskings compose here and getting any of them wrong silently mislabels a
        roll -- which would let a target draw references from its own roll, exactly the
        failure the exclusion exists to prevent.  ``keep`` is the frame-selection mask over
        the raw frames; ``grp`` is the bin index of each SELECTED frame; ``bkeep`` drops
        all-zero bins.  A bin takes the label of its first frame, and binning never spans
        partitions because the frames of a roll are contiguous.
        """
        lab = np.asarray(self._reflib_spec["partition"])
        if lab.size != int(np.asarray(keep).size):
            raise ValueError(f"reference library has {lab.size} partition labels for "
                             f"{np.asarray(keep).size} science frames")
        lab = lab[np.asarray(keep, bool)]
        grp = np.asarray(grp, int)
        ng = int(grp.max()) + 1 if grp.size else 0
        first = np.full(ng, -1, int)
        for i in range(grp.size - 1, -1, -1):          # last write wins -> the first frame
            first[grp[i]] = i
        out = lab[first]
        mixed = [b for b in range(ng) if len(set(lab[grp == b].tolist())) > 1]
        if mixed:
            raise ValueError(f"bins {mixed[:4]} mix partitions; bin within a partition or "
                             f"fix bin so the exclusion stays meaningful")
        return out[np.asarray(bkeep, bool)]

    def _reference_cube(self, filt: int, meta: Dict[str, Any], threads: int = 1) -> Optional[np.ndarray]:
        """RDI reference frames: centre-cropped to the science FOV and given the
        first high-pass (``smooth(/nan)``, reduce_near_2 L1864-1867).  Not frame
        selected, not binned (the file is required to be pre-registered/binned)."""
        ref = self.data.ref_cube
        ny, nx = self.data.cube.shape[1:]
        if ref is None:
            meta["rdi_note"] = "no reference cube -> ADI"
            return None
        ref = np.asarray(ref, np.float32)
        if ref.ndim != 3 or ref.shape[1] < ny or ref.shape[2] < nx:
            meta["rdi_note"] = "reference cube shape mismatch -> ADI"
            return None
        if ref.shape[1] > ny or ref.shape[2] > nx:
            oy, ox = (ref.shape[1] - ny) // 2, (ref.shape[2] - nx) // 2
            ref = ref[:, oy:oy + ny, ox:ox + nx]
        if filt > 1:
            ref = highpass_stack(ref, filt, nan_aware=True, threads=threads)
        return ref

    # -- main entry ----------------------------------------------------------
    def prepare(self, req: ReductionRequest) -> PreparedFrames:
        """The first half of :meth:`reduce`: the frame selection, the injections into the
        kept frames, the KLIP-FM model cube, the high-pass, the destriping and the
        angle-aware binning.  What another wavelength group's reduction needs from this one
        for its SDI pool is exactly this, so the SDI mode runs it once per group and hands
        the result round (``ReductionRequest.stage = "prepare"``)."""
        p = dict(self.defaults)
        p.update({k: v for k, v in req.params.items() if v is not None})
        k = int(max(round(p["k_klip"]), 1))
        bin_ = int(max(round(p["bin"]), 1))
        filt = int(round(p["filter"]))
        # the KLIP zone is the annulus padded by zone_pad on each edge the runner marks: by
        # default only an edge shared with another annulus (RunConfig.edge_padding).  A
        # request without the marks pads both, as the IDL does.
        pad_in = self.zone_pad if _flag(p.get("pad_in", True)) else 0.0
        pad_out = self.zone_pad if _flag(p.get("pad_out", True)) else 0.0
        ann_in = max(float(p["inrad"]) - pad_in, 0.0)
        ann_out = min(float(p["outrad"]) + pad_out, self.outrad_cap)
        ann_in = min(ann_in, ann_out - 1.0)
        # zone_in / zone_out widen the zone beyond the annulus (0, 0: the zone is the annulus).
        # The bin's PA span and angsep's arc stay the annulus', where the sources are.
        z_in, z_out = _zone_ext(p)
        inrad = max(ann_in - z_in, 0.0)
        outrad = min(ann_out + z_out, self.outrad_cap)
        inrad = min(inrad, outrad - 1.0)
        ds = self.data
        meta: Dict[str, Any] = {}
        # the job's threads: the per-target KLIP loop, and the per-frame steps around it
        nth = int(max(getattr(req, "threads", 1) or 1, 1))
        do_fm = bool(req.fm_sources)
        if do_fm and req.k_scan:
            meta["fm_note"] = "KLIP-FM skipped: not supported with k_scan"
            do_fm = False
        keep = frame_selection_mask(ds.nframes, ds.tags, p["corr_thresh"], p["noise_max"],
                                    p["coronoise_max"], bin_, k)
        dth = self.dth_max_deg(ann_out)
        # ``hp_order="binned"`` (under test, off by default): bin first, then high-pass the
        # binned frames.  Both are linear, so the images agree to rounding, and the high-pass
        # costs 1/bin as much.  The means are taken in double precision: binned before the
        # high-pass, the frames still carry their background.  The IDL high-passes every
        # frame before binning, and a destriped cube keeps that order (not linear).
        hp_binned = str(p.get("hp_order", "raw")).lower() == "binned" and filt > 1 and not p["do_destripe"]
        hp_nan = _flag(p.get("hp_nan_aware", False))
        if _flag(p.get("prep_fused", True)):
            # one bin group at a time, the frames read once (_prep_fused)
            bcube, bang, grp, bkeep, mcube = self._prep_fused(
                p, keep, bin_, filt, dth, hp_nan, hp_binned, nth, req.injections or None,
                req.fm_sources if do_fm else None)
        else:
            # the reference: a whole cube per step.  The selection reads the tags, not the
            # pixels, so it comes first.  Only the kept frames are injected, and the injected
            # cubes come back as cube[keep].
            cube = self._inject(req.injections, keep, threads=nth) if req.injections else ds.cube
            mcube = self.model_cube(req.fm_sources, keep, threads=nth) if do_fm else None
            if mcube is not None and filt > 1 and not hp_binned:
                mcube = highpass_stack(mcube, filt, nan_aware=hp_nan, threads=nth)
            bcube, bang, grp, bkeep = self._prep_cube(cube, p, keep, bin_, filt, dth, hp_nan, hp_binned, nth,
                                                      selected=bool(req.injections))
            if mcube is not None and bcube.shape[0]:
                if hp_binned:
                    mb = bin_by_groups(mcube, grp, threads=nth, dtype=np.float64) if bin_ > 1 else mcube
                    mb = highpass_stack(mb, filt, nan_aware=hp_nan, threads=nth)
                else:
                    mb = bin_by_groups(mcube, grp, threads=nth) if bin_ > 1 else np.asarray(mcube, np.float32)
                mcube = mb[bkeep]
        if bcube.shape[0] == 0:
            # bin_frames drops bins whose nansum is zero.  np.nansum of an ALL-NaN frame is
            # 0.0, so a cube that is mostly NaN -- a subarray padded into a larger grid, say
            # -- is wiped out entirely by the high-pass, which at nan_aware=False spreads NaN
            # over the whole frame.  Every downstream shape then goes to zero and the failure
            # surfaces somewhere far away and unrecognisable.
            nan_frac = float(np.mean(~np.isfinite(np.asarray(ds.cube, float))))
            raise ValueError(
                f"every frame was dropped as empty after filtering and binning "
                f"(filter={filt}, bin={bin_}); {100 * nan_frac:.0f}% of the input pixels are "
                f"NaN. A high-pass filter spreads NaN across the whole frame, so a padded or "
                f"masked cube must be cropped to its finite region before it reaches the "
                f"reducer.")
        return PreparedFrames(p, k, bin_, filt, inrad, outrad, dth, keep, bcube, bang, grp, bkeep, mcube,
                              hp_nan, hp_binned, nth, meta, ann=(ann_in, ann_out))

    # ------------------------------------------------------------------ the fused path
    def _prep_fused(self, p, keep, bin_, filt, dth, hp_nan, hp_binned, nth, injections=None, fm_sources=None):
        """The frames before KLIP, one bin group at a time: ``(bcube, bang, grp, bkeep,
        mcube)`` as the whole-cube steps make them (:meth:`_prep_cube`, :meth:`_inject`,
        :meth:`model_cube`, and the model's binning in :meth:`prepare`), to the bit.

        A group's kept frames are copied out of the cube (``float32`` when they take
        sources, as the injected cube held them), given each source's stamp in the order of
        the sources (:func:`klip_tpe.injection.place_source`, the arithmetic of
        :func:`~klip_tpe.injection.inject_frames`), high-passed a block of frames at a time
        (``_highpass_frames``, which filters every frame on its own), destriped frame by
        frame when asked, and averaged by :func:`~klip_tpe.klip.group_mean`, the function the
        whole-cube binning uses.  With ``hp_order="binned"`` the group is averaged in double
        precision first and the mean high-passed.  The KLIP-FM model cube is made the same
        way from zero frames.

        So the cube is read once and nothing the size of the cube is written: the stages
        that used to read and write a whole cube each -- the injected copy, the compact copy
        of the kept frames, the high-pass, the binning -- run on a group while it is in the
        processor's cache.  The groups go to the job's threads."""
        ds = self.data
        kidx = np.flatnonzero(keep)
        a_ = np.asarray(ds.angles, float)[kidx]
        grp = bin_groups(a_, bin_, dth)
        bounds = group_bounds(grp)
        ng = bounds.size - 1
        if bin_ <= 1:
            bang = a_.copy()
        else:
            bang = np.bincount(grp, weights=a_, minlength=ng) / np.bincount(grp, minlength=ng)
        clip = float(p["destripe_clip"]) if (p["do_destripe"] and not hp_binned) else None
        place = self._placer(injections, "inj") if injections else None
        bc = self._fused_bins(ds.cube, kidx, bounds, bin_, filt, hp_nan, hp_binned, nth, place, clip)
        tot = np.array([np.nansum(f) for f in bc])
        bkeep = tot != 0
        mcube = None
        if fm_sources:
            mb = self._fused_bins(None, kidx, bounds, bin_, filt, hp_nan, hp_binned, nth,
                                  self._placer(fm_sources, "fm"), None)
            mcube = mb[bkeep]
        return bc[bkeep], bang[bkeep], grp, bkeep, mcube

    def _placer(self, sources: Sequence[Source], kind: str):
        """``place(frames, J)``: every source's stamp added to ``frames[i]``, frame ``J[i]``
        of the cube, in the order of the sources, as :func:`inject_frames` adds them.  The
        stamps of a set that comes back are kept (:class:`_CropCache`)."""
        if self.model is None:
            raise ValueError("KLIPReducer needs an injection_model to inject sources" if kind == "inj" else
                             "KLIPReducer needs an injection_model to build the FM model cube")
        key = tuple(s.as_tuple() for s in sources)
        first, cache = self._crops[kind].lookup(key)
        if first and kind == "inj":
            _check_float32_headroom(self.data.cube, sources, self.model, self.fallback, peak=self.cube_peak())
        prepared = prepare_sources(sources, self.model, self.fallback, self.pxscale)
        ny, nx = self.data.cube.shape[1:]
        center = ((nx - 1) / 2.0, (ny - 1) / 2.0)
        angles = np.asarray(self.data.angles, float)
        model, truenorth, conv = self.model, self.truenorth, self.angle_convention

        def place(frames: np.ndarray, J: np.ndarray) -> None:
            todo = []
            for i, j in enumerate(J):
                got = None if cache is None else cache.get(int(j))
                if got is None:
                    todo.append(i)
                else:
                    for c in got:
                        add_crop(frames[i], c)
            if not todo:
                return
            Jt = [int(J[i]) for i in todo]
            kept = None if cache is None else [[] for _ in todo]
            for pp in prepared:
                def sink(k, c):
                    add_crop(frames[todo[k]], c)
                    if kept is not None:
                        kept[k].append((c[0], c[1], c[2], c[3], c[4].copy()))
                place_source(Jt, pp, model, angles, truenorth, center, conv, (ny, nx), sink)
            if kept is not None:
                for k, j in enumerate(Jt):
                    cache[j] = kept[k]
        return place

    def _fused_bins(self, base, kidx, bounds, bin_, filt, hp_nan, hp_binned, nth, place, clip) -> np.ndarray:
        """Every bin group's frame, before the all-zero test: ``base[kidx]`` (zero frames
        when ``base`` is None) through ``place``, the high-pass, ``clip``'s destriping and
        the mean, a batch of whole groups at a time (see :meth:`_prep_fused`)."""
        ny, nx = self.data.cube.shape[1:]
        npix = int(ny) * int(nx)
        ng = int(bounds.size) - 1
        out = np.empty((ng, ny, nx), np.float32)
        # a batch holds whole groups and, past its first group, no more frames than one block
        # of highpass_stack (its float64 copy within BLOCK_BYTES): bin 1 goes five frames of
        # 150 x 150 at a time, a bin of 29 one group at a time
        cap = max(1, BLOCK_BYTES // (8 * npix))

        def hp(fr):
            if filt <= 1:
                return fr
            if fr.dtype != np.float32:
                return highpass_stack(fr, filt, nan_aware=hp_nan, threads=1)
            for lo, hi in frame_blocks(fr.shape[0], 1, npix):    # in place, a block at a time
                fr[lo:hi] = _highpass_frames(fr[lo:hi], filt, hp_nan)
            return fr

        def block(g0, g1):
            g = g0
            while g < g1:
                h = g + 1
                while h < g1 and bounds[h + 1] - bounds[g] <= cap:
                    h += 1
                s0, s1 = int(bounds[g]), int(bounds[h])
                J = kidx[s0:s1]
                if base is None:
                    fr = np.zeros((s1 - s0, ny, nx), np.float32)
                else:
                    fr = base[J]                     # a copy, in the cube's own precision
                    if place is not None and fr.dtype != np.float32:
                        fr = fr.astype(np.float32)   # as the injected cube held the frames
                if place is not None:
                    place(fr, J)
                if not hp_binned:
                    fr = hp(fr)
                    if clip is not None:
                        # reduce_near_2 L1672-1677: 90 deg (columns) then 0 deg (rows); science only
                        fr = np.stack([destripe(destripe(f, 90.0, clip), 0.0, clip) for f in fr])
                    for q in range(g, h):
                        a, b = int(bounds[q]) - s0, int(bounds[q + 1]) - s0
                        out[q] = fr[a] if bin_ <= 1 else group_mean(fr[a:b], np.float32)
                else:
                    for q in range(g, h):
                        a, b = int(bounds[q]) - s0, int(bounds[q + 1]) - s0
                        m = np.asarray(fr[a:b], np.float32) if bin_ <= 1 else group_mean(fr[a:b], np.float64)[None]
                        out[q] = highpass_stack(m, filt, nan_aware=hp_nan, threads=1)[0]
                g = h
        run_blocks(block, frame_blocks(ng, nth), nth)
        return out

    def _clean_bins(self, p, keep, bin_, filt, dth, hp_nan, hp_binned, nth) -> np.ndarray:
        """The binned frames of the cube without injections (the reference library's
        ranking cube), by the path :meth:`prepare` takes."""
        if _flag(p.get("prep_fused", True)):
            return self._prep_fused(p, keep, bin_, filt, dth, hp_nan, hp_binned, nth)[0]
        return self._prep_cube(self.data.cube, p, keep, bin_, filt, dth, hp_nan, hp_binned, nth)[0]

    def _prep_cube(self, c, p, keep, bin_, filt, dth, hp_nan, hp_binned, nth, selected=False):
        """frame selection, high-pass, destripe and binning of a science cube
        (``selected``: ``c`` holds only the kept frames already)"""
        a_ = self.data.angles
        if not keep.all():
            a_ = a_[keep]
            if not selected:
                c = c[keep]
        if hp_binned:
            return bin_frames(c, a_, bin_, dth, return_groups=True, threads=nth, dtype=np.float64,
                              then=lambda bc: highpass_stack(bc, filt, nan_aware=hp_nan, threads=nth))
        if filt > 1:
            c = highpass_stack(c, filt, nan_aware=hp_nan, threads=nth)
        if p["do_destripe"]:
            # reduce_near_2 L1672-1677: 90 deg (columns) then 0 deg (rows); science only
            clip = float(p["destripe_clip"])
            c = np.stack([destripe(destripe(f, 90.0, clip), 0.0, clip) for f in c])
        return bin_frames(c, a_, bin_, dth, return_groups=True, threads=nth)

    def reduce(self, req: ReductionRequest) -> ReductionResult:
        if str(getattr(req, "stage", "reduce")) == "prepare":
            pr = self.prepare(req)
            return pr.siblings_payload(getattr(self, "partition_id", None), self.lam_m)
        return self._reduce_prepared(self.prepare(req), req)

    def _sdi_rows(self, siblings, prep: PreparedFrames):
        """The SDI pool's rows for this group: every sibling's prepared frames magnified
        about the star by ``lambda_this / lambda_sibling`` (:mod:`klip_tpe.rescale`), with
        their angles and scales, and the model cubes the same way when KLIP-FM is on, so a
        planet sits at ``r * s`` in the FM rows as in the data rows.  Returns ``None``
        without siblings."""
        from .parallel import SharedArrays
        from .rescale import rescale_frames
        if siblings is None:
            return None
        closer = None
        if isinstance(siblings, SharedArrays):
            arrays, closer = siblings.attach()
            siblings = {pid: SiblingFrames(pid, float(ex["lam_m"]), arrays[f"{pid}/bcube"], np.asarray(ex["bang"], float),
                                           arrays.get(f"{pid}/mcube"))
                        for pid, ex in siblings.extra.items()}
        try:
            rows, angs, scales, fms, lenders, counts = [], [], [], [], [], []
            for pid, sib in siblings.items():
                if getattr(sib, "pid", pid) == getattr(self, "partition_id", object()) or sib.bcube.shape[0] == 0:
                    continue
                s = float(self.lam_m) / float(sib.lam_m)
                rows.append(rescale_frames(sib.bcube, s, threads=prep.nth))
                angs.append(np.asarray(sib.bang, float))
                scales.append(np.full(int(sib.bcube.shape[0]), s))
                if prep.mcube is not None:
                    if sib.mcube is None:
                        raise ValueError(f"KLIP-FM with an SDI pool needs the siblings' model cubes ({pid} has none)")
                    fms.append(rescale_frames(sib.mcube, s, threads=prep.nth))
                lenders.append(pid)
                counts.append(int(sib.bcube.shape[0]))
        finally:
            if closer is not None:
                closer()
        if not rows:
            return None
        return {"rows": np.concatenate(rows, axis=0), "angles": np.concatenate(angs),
                "scales": np.concatenate(scales), "fm_rows": np.concatenate(fms, axis=0) if fms else None,
                "lenders": lenders, "counts": counts}

    def _reduce_prepared(self, prep: PreparedFrames, req: ReductionRequest) -> ReductionResult:
        p, k, bin_, filt = prep.p, prep.k, prep.bin_, prep.filt
        inrad, outrad, dth, keep = prep.inrad, prep.outrad, prep.dth, prep.keep
        bcube, bang, grp, bkeep, mcube = prep.bcube, prep.bang, prep.grp, prep.bkeep, prep.mcube
        meta, nth = prep.meta, prep.nth
        ds = self.data
        spec = self._reflib_spec
        # a zone reaching beyond its annulus (zone_in / zone_out) keeps angsep's arc, and the
        # SDI displacement rule's radius, on the annulus
        ann = None if prep.ann is None else (float(prep.ann[0]), float(prep.ann[1]))
        widened = ann is not None and (ann[0] != float(inrad) or ann[1] != float(outrad))
        # the SDI pool: the other groups' frames, rescaled, when the count asks for any
        sdi = None
        if spec is not None and spec.get("sdi") and int(round(float(p.get("nkeep_sdi", 0) or 0))) > 0:
            if req.siblings is None:
                # a bare KLIPReducer.reduce: nobody prepared the other groups' frames
                meta["sdi_note"] = "nkeep_sdi > 0 but no sibling frames came with the request: no SDI rows"
            else:
                sdi = self._sdi_rows(req.siblings, prep)
        ref_basis = fm_ref = None
        ref_lib = ref_keep = None
        rdi_mode = str(p["rdi_mode"]).lower()
        adi_point = self._library_at_adi_point(p, int(bcube.shape[0]))
        if adi_point is not None:
            # the ADI point of a searched library -- every count the space searches at 0 and the
            # rotating science pool whole -- IS the ADI reduction, and klip_annular would run it
            # as such (library_as_adi) after the reference cube's high-pass and the similarity
            # matrix had been paid for.  Neither is needed to reduce nothing but the sequence.
            meta.update(rdi_mode="searched", ref_keep=dict(adi_point),
                        n_ref_frames=0 if ds.ref_cube is None else int(np.asarray(ds.ref_cube).shape[0]),
                        ref_metric=spec["metric"])
        elif p["use_rdi"] or sdi is not None:
            ref = self._reference_cube(filt, meta, threads=nth) if p["use_rdi"] else None
            if spec is not None:
                # Searched library: the basis cube is [science | SDI rows | references] in that
                # order, because the group row indices the library hands back address it that
                # way, and every target picks its own rows out of it.  ARDI and RDI stop being
                # separate modes here -- an all-zero altroll count IS pure RDI, and the
                # optimizer can elect it rather than being told.
                bpart = self._binned_partition(keep, grp, bkeep)
                kx = (bin_, round(float(dth), 6), hash(np.asarray(keep, bool).tobytes()))
                sim_cube = None
                if req.injections:
                    # the ranking is computed once per key, on the frames WITHOUT injections
                    # -- rebuilt here only if the key is new
                    k0 = (int(bcube.shape[0]), int(filt), round(float(inrad), 3), round(float(outrad), 3),
                          spec["metric"], spec["shift_px"]) + kx
                    if k0 not in self._reflib_cache:
                        sim_cube = self._clean_bins(p, keep, bin_, filt, dth, prep.hp_nan, prep.hp_binned, nth)
                ref_lib, ref_basis = self._reflib(bcube, ref, bpart, filt, inrad, outrad, key_extra=kx,
                                                  sim_cube=sim_cube, bang=bang, sdi=sdi,
                                                  mid_px=0.5 * (ann[0] + ann[1]) if widened else None)
                ref_keep = ref_lib.keep_from_config(p)
                if mcube is not None:
                    fm_ref = np.concatenate(
                        [mcube] + ([sdi["fm_rows"]] if sdi is not None else [])
                        + ([np.zeros_like(ref)] if ref is not None else []), axis=0)
                meta.update(rdi_mode="searched", n_ref_frames=0 if ref is None else int(ref.shape[0]),
                            ref_keep=dict(ref_keep), ref_metric=ref_lib.metric)
                if sdi is not None:
                    meta.update(n_sdi_rows=int(sdi["rows"].shape[0]), sdi_lenders=[str(x) for x in sdi["lenders"]])
            elif ref is not None:
                if rdi_mode == "rdi":
                    ref_basis = ref
                    fm_ref = np.zeros_like(ref) if mcube is not None else None
                else:                                   # ARDI: reference + science
                    ref_basis = np.concatenate([ref, bcube], axis=0)
                    fm_ref = np.concatenate([np.zeros_like(ref), mcube], axis=0) if mcube is not None else None
                meta.update(rdi_mode=rdi_mode, n_ref_frames=int(ref.shape[0]))

        kp = KLIPParams(k_klip=k, inrad=inrad, outrad=outrad, n_ang=int(max(round(p["n_ang"]), 1)),
                        fast=bool(p["fast"]), angsep=float(p["angsep"]), anglemax=float(p["anglemax"]),
                        n_min_ref=int(p["n_min_ref"]), spat_mean=bool(p["spat_mean"]),
                        temp_mean=bool(p["temp_mean"]), k_scan=bool(req.k_scan),
                        threads=int(max(getattr(req, "threads", 1), 1)),
                        ref_lib=ref_lib, ref_keep=ref_keep, gram=str(p.get("klip_gram", "target") or "target"),
                        arc_radii=ann if widened else None)
        if widened:
            meta["annulus"] = [ann[0], ann[1]]
        fm_img = nosub = cadi = None
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # nanmedian of all-NaN corners
            img, fm_img, info = self._subtract(bcube, bang, kp, p, filt, req, mcube, ref_basis, fm_ref, meta)
            if req.extras:
                if req.injections:
                    # classical ADI (L2072-2086): 2nd high-pass, median PSF in the detector frame, derotate, median
                    c2 = highpass_stack(bcube, filt, nan_aware=True, threads=nth) if filt > 1 else bcube
                    c2 = c2 - np.nanmedian(c2, axis=0)
                    cadi = np.asarray(np.nanmedian(derotate(c2, bang, self.truenorth, threads=nth), axis=0),
                                      np.float32)
                else:
                    # no subtraction (L2087-2096): derotate + median stack of the selected/filtered/binned cube
                    nosub = np.asarray(np.nanmedian(derotate(bcube, bang, self.truenorth, threads=nth), axis=0),
                                       np.float32)
        if self.mask_fn is not None:
            img = np.stack([self.mask_fn(im) for im in img]) if req.k_scan else self.mask_fn(img)
        meta.update(info)
        if adi_point is not None:
            meta["library_as_adi"] = True
        meta.update(n_kept=int(keep.sum()), n_binned=int(bcube.shape[0]), inrad=inrad, outrad=outrad,
                    params=p, tag=req.tag)
        return ReductionResult(np.asarray(img, np.float32), self.partition_weight(), meta,
                               fm_image=fm_img, nosub=nosub, cadi=cadi)

    def _library_at_adi_point(self, p: Dict[str, Any], n_binned: int) -> Optional[Dict[str, int]]:
        """The counts of a searched library when they leave nothing but the rotating sequence
        itself (every pool the space searches at 0, the one-label science pool whole), else
        None.  A pool the space does not search is at its maximum, so it counts as present."""
        spec = self._reflib_spec
        if spec is None or not spec.get("rotation") or not self.supports_reference_library:
            return None
        if int(np.unique(np.asarray(spec["partition"])).size) > 1:
            return None
        searched = spec.get("searched")
        stub = self._reflib_stub()
        keep: Dict[str, int] = {}
        for g in stub.groups:
            is_searched = searched is None or g.name in searched
            v = p.get(g.param_name)
            if g.name == "altroll":
                n = int(round(float(v))) if (is_searched and v is not None) else max(n_binned - 1, 0)
                if n < max(n_binned - 1, 0):
                    return None
                keep[g.name] = n                      # the count as the record has it (the config's own when searched)
                continue
            if not is_searched or v is None or int(round(float(v))) != 0:
                return None
            keep[g.name] = 0
        return keep

    # ------------------------------------------------------------------ backend hook
    def _subtract(self, bcube: np.ndarray, bang: np.ndarray, kp: KLIPParams, p: Dict[str, Any], filt: int,
                  req: ReductionRequest, mcube: Optional[np.ndarray], ref_basis: Optional[np.ndarray],
                  fm_ref: Optional[np.ndarray], meta: Dict[str, Any]):
        """PSF subtraction + derotation + combination of the *pre-processed* (injected,
        frame-selected, high-passed, binned) cube.  Returns ``(image, fm_image, info)``
        where ``image`` is ``(ny, nx)`` or, for ``kp.k_scan``, ``(k, ny, nx)`` for
        ``k = 1..kp.k_klip``.  Backends (pyKLIP, VIP, a user pipeline) override this and
        inherit the rest of the chain unchanged; the reference implementation is the
        annular KLIP of :mod:`klip_tpe.klip`."""
        out = klip_annular(bcube, bang, kp, self.lam_over_d_px, fm_cube=mcube, ref_cube=ref_basis,
                           fm_ref_cube=fm_ref, fm_selfsub=bool(p["fm_selfsub"]))
        res, info = out[0], out[1]
        fm_res = out[2] if len(out) > 2 else None
        ct = p["comb_type"]
        nth = int(max(getattr(req, "threads", 1) or 1, 1))
        # the residuals' second high-pass (reduce_near_2's); post_hp = False leaves them as KLIP
        # made them, as the ALES reduction's ADI images of MWC 758 are
        post_hp = filt > 1 and _flag(p.get("post_hp", True))

        def _post(rc: np.ndarray) -> np.ndarray:
            if post_hp:
                rc = highpass_stack(rc, filt, nan_aware=True, threads=nth)
            return derotate(rc, bang, self.truenorth, threads=nth)

        def _combine(dr: np.ndarray, ref: Optional[np.ndarray] = None) -> np.ndarray:
            if ct == "median":
                return np.nanmedian(dr, axis=0)
            if ct == "mean":
                return np.nanmean(dr, axis=0)
            return nw_ang_comb(dr, bang, ref_cube=ref, threads=nth)

        fm_img = None
        if req.k_scan:
            img = np.stack([_combine(_post(res[kk])) for kk in range(res.shape[0])])
        else:
            dr = _post(res)
            img = _combine(dr)
            if fm_res is not None:
                # FM frames post-processed like the data, combined with the SCIENCE weights (nw_ang_comb_ref)
                fm_img = np.asarray(_combine(_post(fm_res), ref=dr), np.float32)
        return img, fm_img, info

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(lam_m=self.lam_m, diam_m=self.diam_m, truenorth=self.truenorth, zone_pad=self.zone_pad,
                 outrad_cap=self.outrad_cap, defaults=dict(self.defaults), nframes=self.data.nframes,
                 has_ref_cube=self.data.ref_cube is not None, mask_fn=self.mask_fn is not None,
                 injection_model=None if self.model is None else self.model.describe(),
                 # which frames these are, when the loader said (load_calints does): the run's
                 # record of its input, beside the record of what it did with it
                 frames=(getattr(self.data, "meta", None) or {}).get("provenance"))
        if getattr(self, "data_edits", None):
            d["data_edits"] = dict(self.data_edits)
        spec = getattr(self, "_reflib_spec", None)
        if spec is not None:
            n_ref = 0 if self.data.ref_cube is None else int(np.asarray(self.data.ref_cube).shape[0])
            searched = spec.get("searched")
            d["reference_library"] = {
                "metric": spec["metric"], "shift_px": spec["shift_px"], "n_min_ref": spec["n_min_ref"],
                "ref_group": spec["ref_group"], "n_ref_frames": n_ref, "rotation": bool(spec.get("rotation", False)),
                "n_labels": int(np.unique(np.asarray(spec["partition"])).size),
                "pools": [g.param_name + (f" {g.min_keep}-{g.max_keep} searched"
                                          if searched is None or g.name in searched
                                          else " all eligible frames (not searched)")
                          for g in self._reflib_stub().groups]}
        return d


# ----------------------------------------------------------------------------
# multi-partition wrapper
# ----------------------------------------------------------------------------
@dataclass
class EvalImages:
    """Everything one evaluation produced."""

    image: np.ndarray                              # combined injected (or clean) image
    stack: Optional[np.ndarray] = None             # (npart, ny, nx) per-partition images
    weights: Optional[np.ndarray] = None
    partitions: Optional[List[Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    fm_image: Optional[np.ndarray] = None          # KLIP-FM response, combined with the SAME partition weights
    fm_stack: Optional[np.ndarray] = None          # (npart, ny, nx) per-partition FM images
    nosub: Optional[np.ndarray] = None             # extras: combined no-subtraction reference (clean pass)
    cadi: Optional[np.ndarray] = None              # extras: combined classical-ADI reference (injected pass)


def combine_stack(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """NaN-aware weighted mean over the first axis (``near2m_avg`` / ``near2m_pncombine``)."""
    stack = np.asarray(stack, float)
    w = np.asarray(weights, float).reshape((-1,) + (1,) * (stack.ndim - 1))
    fin = np.isfinite(stack)
    num = np.nansum(np.where(fin, stack, 0.0) * w, axis=0)
    den = np.sum(fin * w, axis=0)
    out = num / np.where(den > 0, den, np.nan)
    return out


def _run_one(args):
    reducer, req = args
    return reducer.reduce(req)


class PartitionedReducer(Reducer):
    """Reduce every selected partition with its own parameter block and combine.

    Parameters
    ----------
    reducers
        ``{partition_id: Reducer}``.
    weighting
        ``'equal'`` (uniform mean; the production per-night combine -- inverse
        variance is pathological here because an over-subtracted night has low
        scatter *and* low throughput) or ``'sqrt_texp'`` (``Reducer.partition_weight``).
    max_workers
        the thread budget.  >1 dispatches partition jobs to forked worker processes (or
        a thread pool, when forking is unavailable).
    threads_per_job
        threads each partition job uses for its per-target work.  The process pool gets
        ``max_workers // threads_per_job`` processes, and never fewer than the
        ``min(max_workers, 2 x partitions)`` it had before, so the runner can keep as many
        reductions in flight as the budget can serve (a trial's clean and all its draws,
        several validation trials).  Per-job scaling is flat past about 4 threads.
    representative
        the partition whose pixel scale, FWHM, lambda/D and matched-filter kernel stand for
        the combined image (the metric and the sampler read them from here).  Default: the
        first partition, which is right when the partitions share a wavelength.  Wavelength
        groups of an IFS cube do not, and the first group is the bluest: an ALES adapter
        names the group at the mean wavelength instead.
    """

    name = "partitioned"
    # class defaults, so an instance made without __init__ still works; __init__ sets its own
    threads_per_job = 4
    _threads_fallback = False
    _start_lock = threading.Lock()
    _recent_costs = None
    #: ``{pid: factor}``: partition ``pid`` gets every injected source at its contrast times
    #: ``factor`` (:meth:`set_injection_spectrum`); None, the default, injects the contrast
    #: given in every partition, a source with the star's spectrum
    injection_scale: Optional[Dict[Any, float]] = None
    injection_spectrum_info: Optional[Dict[str, Any]] = None

    def __init__(self, reducers: Dict[Any, Reducer], weighting: str = "equal", max_workers=1,
                 pool: str = "auto", log=None, threads_per_job: int = 4, representative: Any = None):
        if not reducers:
            raise ValueError("need at least one partition")
        if representative is not None and representative not in reducers:
            raise KeyError(f"representative partition {representative!r} is not one of {list(reducers)}")
        from .parallel import resolve_workers
        self.reducers = dict(reducers)
        self.weighting = weighting
        self.max_workers = resolve_workers(max_workers)
        self.threads_per_job = max(int(threads_per_job or 4), 1)
        self.pool_kind = pool            # auto | processes | threads
        self._log = log or (lambda s: None)
        self._procs = None
        self._threads_fallback = False   # worker processes could not be started
        self._start_lock = threading.Lock()    # the runner keeps several reductions in flight
        self._recent_costs = None        # costs of the latest partition jobs (see _plan)
        if self.max_workers > 1:
            from .parallel import pin_blas
            pin_blas()                      # one BLAS thread per worker thread (no oversubscription)
        self.representative = representative if representative is not None else next(iter(self.reducers))
        for pid, r in self.reducers.items():
            if getattr(r, "partition_id", None) is None:
                r.partition_id = pid
        #: which partitions lend their frames to the others' SDI pools (``"all"``, or
        #: ``"selected"``: only the partitions the configuration includes); an IFS adapter sets it
        self.sdi_lenders: str = "all"
        first = self.reducers[self.representative]
        self.pxscale, self.fwhm = first.pxscale, first.fwhm
        self.lam_over_d_px = first.lam_over_d_px
        self.angle_convention = first.angle_convention
        #: a record of what the partitions are (an IFS adapter's wavelength groups, say),
        #: written to the run's setup file by :meth:`describe`
        self.provenance: Optional[Dict[str, Any]] = None
        self.supports_kscan = all(r.supports_kscan for r in self.reducers.values())
        self.supports_fm = all(getattr(r, "supports_fm", False) for r in self.reducers.values())
        #: what the partitions ARE -- only the wording of the display's partition panels
        #: ("night" for a multi-epoch campaign, "channel", "roll", "group", ...)
        self.partition_label = "night"
        self._pool: Optional[cf.ThreadPoolExecutor] = None

    def partitions(self) -> List[Any]:
        return list(self.reducers)

    def frame_angles(self, pid=None):
        return self.reducers[pid].frame_angles()

    def frame_tags(self, pid=None):
        return self.reducers[pid].frame_tags()

    def partition_weight(self, pid) -> float:
        return 1.0 if self.weighting == "equal" else self.reducers[pid].partition_weight()

    def matched_filter_kernel(self, rho_as: float):
        return self.reducers[getattr(self, "representative", next(iter(self.reducers)))].matched_filter_kernel(rho_as)

    # -- the injected sources' spectrum ----------------------------------------------
    def set_injection_spectrum(self, spectrum: Any, norm: Any = None, label: Optional[str] = None
                               ) -> Optional[Dict[Any, float]]:
        """Give every injected source a spectrum across the partitions.

        A source of contrast ``c`` goes into partition ``pid`` at ``c * factor[pid]``, in every
        reduction this reducer makes for the run: the search's, calibration's and validation's
        injections, the KLIP-FM sources and the frames an SDI pool borrows.  ``c`` is then the
        contrast where the factor is 1, which is ``norm``: the representative partition by
        default (the one whose template is the matched filter), another partition id, or a
        wavelength in metres.  A source of negative contrast, a known companion taken out of
        the frames (``RunConfig.subtract_known``), keeps the contrast given.

        ``spectrum``: None (flat: the contrast given in every partition, a source with the
        star's colours, the default); ``{pid: factor}`` (normalised at ``norm`` too); a callable
        ``f(lambda_um) -> relative contrast``, read at each partition's wavelength (``lam_m``);
        or ``(lambda_um, values)``, interpolated linearly in wavelength and held at its end
        values outside.  Factors are finite and >= 0 (0: no source in that partition).
        ``label`` says what the spectrum is in :meth:`describe` and the run's setup file.

        Set it before the run starts: the run's calibrated contrast is in its units.  Returns
        the factors (None for flat)."""
        if spectrum is None:
            self.injection_scale = None
            self.injection_spectrum_info = None
            return None
        lam_um = {pid: (float(getattr(r, "lam_m", np.nan)) * 1e6) for pid, r in self.reducers.items()}
        if isinstance(spectrum, dict):
            raw = {pid: float(spectrum[pid]) for pid in self.reducers if pid in spectrum}
            missing = [pid for pid in self.reducers if pid not in raw]
            if missing:
                raise KeyError(f"the spectrum has no factor for partitions {missing[:4]}")
            fn = None
            kind = "per partition"
        else:
            if callable(spectrum):
                fn = spectrum
                kind = "function of wavelength"
            else:
                wl, val = (np.asarray(v, float).ravel() for v in spectrum)
                if wl.size != val.size or wl.size < 1:
                    raise ValueError(f"spectrum table: {wl.size} wavelengths and {val.size} values")
                o = np.argsort(wl)
                wl, val = wl[o], val[o]
                fn = lambda x, wl=wl, val=val: float(np.interp(float(x), wl, val))
                kind = f"table of {wl.size} wavelengths, {wl[0]:.3f}-{wl[-1]:.3f} um, linear"
            bad = [pid for pid, v in lam_um.items() if not np.isfinite(v)]
            if bad:
                raise ValueError(f"partitions {bad[:4]} have no wavelength (lam_m) to read the spectrum at")
            raw = {pid: float(fn(lam_um[pid])) for pid in self.reducers}
        rep = getattr(self, "representative", next(iter(self.reducers)))
        if norm is None or norm in self.reducers:
            at = rep if norm is None else norm
            ref = raw[at]
            at_txt = f"partition {at} ({lam_um[at]:.3f} um)" if np.isfinite(lam_um[at]) else f"partition {at}"
        else:
            if fn is None:
                raise ValueError(f"norm={norm!r} is not a partition, and a per-partition spectrum has no "
                                 f"value at a wavelength")
            ref = float(fn(float(norm) * 1e6))
            at_txt = f"{float(norm) * 1e6:.3f} um"
        if not (np.isfinite(ref) and ref > 0):
            raise ValueError(f"the spectrum is {ref!r} at its normalisation ({at_txt}); it must be > 0 there")
        fac = {pid: v / ref for pid, v in raw.items()}
        bad = {pid: v for pid, v in fac.items() if not (np.isfinite(v) and v >= 0)}
        if bad:
            raise ValueError(f"spectrum factors must be finite and >= 0: {dict(list(bad.items())[:4])}")
        self.injection_scale = fac
        self.injection_spectrum_info = {
            "label": label or kind, "kind": kind, "normalised_at": at_txt,
            "factors": {str(pid): round(float(v), 6) for pid, v in fac.items()},
            "lam_um": {str(pid): round(float(v), 5) for pid, v in lam_um.items() if np.isfinite(v)}}
        vals = np.array(list(fac.values()))
        getattr(self, "_log", lambda s: None)(
            f"  injected spectrum ({label or kind}): contrast x {vals.min():.3g}-{vals.max():.3g} across "
            f"{len(fac)} partitions, 1 at {at_txt}")
        return dict(fac)

    def scaled_sources(self, sources: Optional[Sequence[Source]], pid: Any) -> Optional[Sequence[Source]]:
        """``sources`` as partition ``pid`` injects them: each positive contrast times the
        partition's spectrum factor (:meth:`set_injection_spectrum`); unchanged without one."""
        scale = getattr(self, "injection_scale", None)
        if not sources or not scale:
            return sources
        f = float(scale.get(pid, 1.0))
        if f == 1.0:
            return sources
        return [Source(s.rho, s.theta, float(s.contrast) * f) if float(s.contrast) > 0 else s for s in sources]

    def subtract_sources(self, sources: Sequence[Any], threads: int = 1) -> int:
        """:meth:`KLIPReducer.subtract_sources` on every partition: the same sources, on the
        sky, out of every night's frames.  It has to run before the worker processes start,
        since they keep the reducers they were started with, and it refuses afterwards.
        Returns the number of sources subtracted (per partition)."""
        return self._edit_partitions(sources, threads, "subtract_sources", "subtracted_kept", "subtracted_sources")

    def inject_sources(self, sources: Sequence[Any], threads: int = 1) -> int:
        """:meth:`KLIPReducer.inject_sources` on every partition: the same sources, on the sky,
        added to every night's frames, before the worker processes start.  Returns the number
        of sources added (per partition)."""
        return self._edit_partitions(sources, threads, "inject_sources", "injected_kept", "injected_sources")

    def _edit_partitions(self, sources: Sequence[Any], threads: int, method: str, attr: str, key: str) -> int:
        if self._procs is not None:
            raise RuntimeError(f"{method} has to run before the worker pool starts: the worker "
                               "processes keep the frames they were started with")
        src = [s if isinstance(s, Source) else Source(*(float(v) for v in s)) for s in sources]
        n = 0
        kept: Dict[Any, Optional[float]] = {}
        for pid, r in self.reducers.items():
            f = getattr(r, method, None)
            if f is None:
                raise TypeError(f"partition {pid}: a {type(r).__name__} cannot {method.replace('_', ' ')}")
            n = f(src, threads=threads)
            kept[pid] = getattr(r, attr, None)
        setattr(self, attr, kept)
        rec = dict(getattr(self, "data_edits", None) or {})
        rec[key] = list(rec.get(key) or []) + [list(s.as_tuple()) for s in src]
        self.data_edits = rec
        return n

    _pool_logged = False

    def start_workers(self) -> str:
        """Start the worker processes (IDL bridges) once the data are loaded.  Returns the
        pool kind in use: ``processes`` (forked workers, :meth:`n_procs` of them),
        ``threads`` (fallback / requested) or ``serial``.

        Which one a run got is written to its log, here rather than at the call site: the
        command line used to report it and a script that built its own runner reported
        nothing, so the only record of the mode a run was in was whether a *failure* message
        happened to be there.  Working out after the fact that a search had spent 350
        evaluations in thread mode should not require inferring it from an absence.
        """
        kind = self._pool_kind_now()
        if not self._pool_logged:
            self._pool_logged = True
            from .parallel import describe
            procs = (getattr(self._procs, "n", None) or self.n_procs()) if kind == "processes" else None
            per = self.job_threads()
            costed = bool(procs) and all(getattr(type(r), "cost_estimate", Reducer.cost_estimate)
                                         is not Reducer.cost_estimate for r in self.reducers.values())
            cap = min(4 * per, int(self.max_workers)) if costed else None
            self._log(f"  parallel: {describe(self.max_workers, len(self.reducers), procs, per, cap)}"
                      f"; pool = {kind}")
            if kind == "threads" and self.pool_kind in ("auto", "processes"):
                self._log("    threads share the GIL, so expect several times less throughput than "
                          "worker processes: worth fixing rather than living with.")
        return kind

    def _pool_kind_now(self) -> str:
        if self.max_workers <= 1:
            return "serial"
        with self._start_lock:
            return self._pool_kind_locked()

    def _pool_kind_locked(self) -> str:
        if self._procs is not None:
            return "processes"
        if self.pool_kind in ("auto", "processes") and not self._threads_fallback:
            try:
                from .parallel import ProcessPool
                self._procs = ProcessPool(self.reducers, self.n_procs(), log=self._log)
                return "processes"
            except Exception as exc:
                self._log(f"  parallel: worker processes unavailable ({exc!r}); using threads")
                self._threads_fallback = True
        return "threads"

    def n_procs(self) -> int:
        """Worker processes for the budget: one per ``threads_per_job`` threads, and never
        fewer than ``min(max_workers, 2 x partitions)`` (the count before 2026-10-11)."""
        b = int(self.max_workers)
        return max(b // self.threads_per_job, min(b, 2 * len(self.reducers)), 1)

    def job_threads(self, n_sel: Optional[int] = None) -> int:
        """Per-target threads for one partition job.  With worker processes the budget is
        shared out by process, so every job gets ``max_workers // processes`` whatever else
        is in flight.  The thread-pool fallback keeps the old rule: the injected and the
        clean reduction of ``n_sel`` partitions share the budget."""
        if self._procs is not None or (self.max_workers > 1 and self.pool_kind != "threads"
                                       and not self._threads_fallback):
            return max(int(self.max_workers) // self.n_procs(), 1)
        n = len(self.reducers) if n_sel is None else int(n_sel)
        conc = 2 if self.max_workers >= 2 else 1
        return max(int(self.max_workers) // max(conc * n, 1), 1)

    def job_slots(self) -> int:
        """Partition jobs that can run at the same time: the worker processes, the
        thread-pool size, or 1 when serial."""
        if self.max_workers <= 1:
            return 1
        if self._procs is not None:
            return int(getattr(self._procs, "n", None) or self.n_procs())
        if self.pool_kind != "threads" and not self._threads_fallback:
            return self.n_procs()
        return int(self.max_workers)

    def memory_note(self) -> Optional[str]:
        """A log line on the worker processes' memory (Linux), for sizing the pool against
        the machine: their peak resident sets, and what they hold together now.  None
        without worker processes or off Linux."""
        mem = getattr(self._procs, "memory", None)
        m = mem() if callable(mem) else None
        if not m:
            return None
        from .parallel import mem_available
        gb = 1024.0 ** 3
        pk = sorted(m["peak"])
        avail = mem_available()
        return (f"worker memory: {len(pk)} processes peaked at {pk[len(pk) // 2] / gb:.2f} GB each (largest "
                f"{pk[-1] / gb:.2f} GB, each counting the data it shares), {m['pss'] / gb:.1f} GB together now "
                f"(shared data counted once)" + (f", {avail / gb:.0f} GB available" if avail else ""))

    def _map(self, jobs, keys=None):
        if self.max_workers > 1 and (len(jobs) > 1 or self._procs is not None):
            if self.pool_kind != "threads" and self._procs is None:
                self.start_workers()
            if self._procs is not None:
                # a lone job goes to the workers too: the runner keeps several reductions in
                # flight, and run here they would share this process's GIL
                pid_of = {id(r): pid for pid, r in self.reducers.items()}
                return self._procs.map([(pid_of[id(r)], req) for r, req in jobs], keys)
            with self._start_lock:
                if self._pool is None:
                    self._pool = cf.ThreadPoolExecutor(max_workers=self.max_workers)
            # pump_wait, not map(): the main thread must keep servicing the GUI event loop
            # while the partitions reduce, or the live window is marked unresponsive
            from .parallel import pump_wait
            return pump_wait([self._pool.submit(_run_one, j) for j in jobs])
        return [_run_one(j) for j in jobs]

    def close(self):
        if self._procs is not None:
            self._procs.close()
            self._procs = None
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def sdi_lender_ids(self, selected: Sequence[Any]) -> Optional[List[Any]]:
        """The partitions whose frames feed the others' SDI pools, or None when no partition
        has an SDI pool (:meth:`KLIPReducer.set_reference_library` with ``sdi``)."""
        if not any((getattr(r, "_reflib_spec", None) or {}).get("sdi") for r in self.reducers.values()):
            return None
        return list(self.reducers) if str(self.sdi_lenders) == "all" else list(selected)

    def _prepare_siblings(self, config: Config, sel, injections, k_scan, tag, fm_sources, per, kind):
        """The SDI mode's first round: every lending partition's frames prepared as it would
        prepare its own for this configuration (``ReductionRequest.stage = "prepare"``), in
        parallel, then handed to the partition jobs -- in one shared-memory block with worker
        processes (:class:`klip_tpe.parallel.SharedArrays`), as they are otherwise.  Nothing
        is done when the configuration's ``nkeep_sdi`` is 0: that point is the reduction
        without an SDI pool, and costs exactly that.  Returns ``(siblings, shared)``."""
        lenders = self.sdi_lender_ids(sel)
        if lenders is None or int(round(float(config.params.get("nkeep_sdi", 0) or 0))) <= 0:
            return None, None
        jobs = [(self.reducers[pid], ReductionRequest(config.params_for(pid), self.scaled_sources(injections, pid),
                                                      k_scan, f"{tag}_{pid}_prep",
                                                      fm_sources=self.scaled_sources(fm_sources, pid),
                                                      threads=per, stage="prepare"))
                for pid in lenders]
        payload = {pid: sf for pid, sf in zip(lenders, self._map(jobs))}
        if kind != "processes":
            return payload, None
        from .parallel import SharedArrays
        arrays, extra = {}, {}
        for pid, sf in payload.items():
            arrays[f"{pid}/bcube"] = sf.bcube
            if sf.mcube is not None:
                arrays[f"{pid}/mcube"] = sf.mcube
            extra[pid] = {"lam_m": float(sf.lam_m), "bang": np.asarray(sf.bang, float)}
        if getattr(self, "_no_shared_memory", False):
            return payload, None
        try:
            shared = SharedArrays.create(arrays, extra)
        except Exception as exc:          # no shared memory here: the arrays travel with the jobs
            self._no_shared_memory = True
            self._log(f"  (shared memory unavailable for the SDI frames, {exc!r}; passing them with the jobs, "
                      f"which is slower)")
            return payload, None
        return shared, shared

    def reduce_config(self, config: Config, injections: Optional[Sequence[Source]] = None,
                      k_scan: bool = False, tag: str = "", selected: Optional[Sequence[Any]] = None,
                      fm_sources: Optional[Sequence[Source]] = None, extras: bool = False
                      ) -> EvalImages:
        """``fm_sources`` requests the KLIP-FM pass (``EvalImages.fm_image``, combined
        with the same partition weights as the science image); ``extras`` the
        nosub / cadi reference images."""
        sel = list(config.selected if selected is None else selected)
        kind = self.start_workers() if self.max_workers > 1 else "serial"
        per = self.job_threads(len(sel))          # threads inside each partition job
        siblings, shared = self._prepare_siblings(config, sel, injections, k_scan, tag, fm_sources, per, kind)
        # each partition injects the sources at its own point of their spectrum, when they have one
        jobs = [(self.reducers[pid], ReductionRequest(config.params_for(pid), self.scaled_sources(injections, pid),
                                                      k_scan, f"{tag}_{pid}",
                                                      fm_sources=self.scaled_sources(fm_sources, pid), extras=extras,
                                                      threads=per, siblings=siblings))
                for pid in sel]
        keys = None
        plan = self._plan(jobs, per) if kind == "processes" else None
        if plan is not None:
            threads, walls = plan
            for (_, req), t in zip(jobs, threads):
                req.threads = t
            now = time.monotonic()
            # the job expected to run longest goes first among those submitted together;
            # one submitted later goes ahead of an earlier one only when it is longer by
            # more than the time between them, so a trial's jobs are not held back for long
            keys = [now - w for w in walls]
        try:
            results = self._map(jobs, keys)
        finally:
            if shared is not None:
                shared.release()
        stack = np.stack([r.image for r in results])
        w = np.array([self.partition_weight(pid) for pid in sel], float)
        img = combine_stack(stack, w)
        meta = {str(pid): r.meta for pid, r in zip(sel, results)}
        ev = EvalImages(img, stack, w, sel, meta)

        def _comb(attr):
            ims = [getattr(r, attr) for r in results]
            if any(im is None for im in ims):
                return None, None
            st = np.stack(ims)
            return combine_stack(st, w), st
        ev.fm_image, ev.fm_stack = _comb("fm_image")
        ev.nosub, _ = _comb("nosub")
        ev.cadi, _ = _comb("cadi")
        return ev

    def reduce(self, req: ReductionRequest) -> ReductionResult:
        """Single-block interface: the same params for every partition."""
        cfg = Config(params=dict(req.params), per_partition={p: dict(req.params) for p in self.reducers},
                     selected=list(self.reducers), x=np.zeros(0))
        ev = self.reduce_config(cfg, req.injections, req.k_scan, req.tag, fm_sources=req.fm_sources,
                                extras=req.extras)
        return ReductionResult(ev.image, 1.0, {"partitions": ev.partitions}, fm_image=ev.fm_image,
                               nosub=ev.nosub, cadi=ev.cadi)

    def _plan(self, jobs, base: int):
        """Threads and expected wall time of each partition job, from the partitions'
        :meth:`Reducer.cost_estimate`, or None when one has no estimate.

        Threads scale with the job's expected cost against the mean of the latest jobs,
        rounded to a power of two between 1 and ``4 x base``: a big night at a small bin gets
        many, a light job one, and over many jobs the mean stays near ``base``
        (``--threads-per-job``).  The images do not depend on the thread count."""
        costs = []
        for r, req in jobs:
            f = getattr(r, "cost_estimate", None)
            c = f(req.params, k_scan=req.k_scan, n_inject=len(req.injections or ()),
                  fm=bool(req.fm_sources)) if callable(f) else None
            if c is None or not np.isfinite(c) or c <= 0:
                return None
            costs.append(float(c))
        with self._start_lock:
            if self._recent_costs is None:
                self._recent_costs = collections.deque(maxlen=max(64, 4 * self.n_procs()))
            self._recent_costs.extend(costs)
            mean = float(np.mean(self._recent_costs))
        cap = max(1, min(4 * int(base), int(self.max_workers)))
        threads = []
        for c in costs:
            t = base * c / max(mean, 1e-12)
            t = 1 if t <= 1.0 else 2 ** int(round(np.log2(t)))
            threads.append(int(min(max(t, 1), cap)))
        walls = [c / t ** 0.85 for c, t in zip(costs, threads)]
        return threads, walls

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(weighting=self.weighting, max_workers=self.max_workers, threads_per_job=self.threads_per_job,
                 representative=str(getattr(self, "representative", next(iter(self.reducers)))),
                 partition_label=getattr(self, "partition_label", "night"),
                 provenance=getattr(self, "provenance", None),
                 partitions={str(k): r.describe() for k, r in self.reducers.items()})
        if getattr(self, "injection_spectrum_info", None):
            d["injection_spectrum"] = dict(self.injection_spectrum_info)
        if getattr(self, "data_edits", None):
            d["data_edits"] = dict(self.data_edits)
        if self.sdi_lender_ids(list(self.reducers)) is not None:
            d["sdi_lenders"] = str(self.sdi_lenders)
        return d


KLIP_GRAM_MODES = ("target", "zone")


def set_klip_gram(reducer: Reducer, mode: str) -> List[Any]:
    """Set ``klip_gram`` (:attr:`klip_tpe.klip.KLIPParams.gram`) as the default of every
    KLIP reducer in ``reducer``, a :class:`PartitionedReducer`'s partitions or a single one.
    Returns the partitions (or ``[None]``) it was set on.  Set it before the worker pool
    starts: the workers keep the reducers they were forked with."""
    mode = str(mode or "target").lower()
    if mode not in KLIP_GRAM_MODES:
        raise ValueError(f"klip_gram is one of {KLIP_GRAM_MODES}, not {mode!r}")
    subs = getattr(reducer, "reducers", None)
    items = list(subs.items()) if isinstance(subs, dict) and subs else [(None, reducer)]
    done = []
    for pid, r in items:
        d = getattr(r, "defaults", None)
        if isinstance(d, dict) and "klip_gram" in d:
            d["klip_gram"] = mode
            done.append(pid)
    return done


def reducer_class(backend: str = "klip"):
    """Reducer class for a backend name: ``'klip'`` (built-in annular KLIP), ``'pyklip'``,
    ``'vip'`` (imported lazily; see :mod:`klip_tpe.backends`)."""
    b = (backend or "klip").lower()
    if b in ("klip", "builtin", "default"):
        return KLIPReducer
    if b == "pyklip":
        from .backends.pyklip import PyKLIPReducer
        return PyKLIPReducer
    if b == "vip":
        from .backends.vip import VIPReducer
        return VIPReducer
    raise ValueError(f"unknown backend {backend!r} (klip | pyklip | vip)")
