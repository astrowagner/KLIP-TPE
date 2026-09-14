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

import concurrent.futures as cf
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .injection import InjectionModel, inject_sources
from .klip import (KLIPParams, bin_by_groups, bin_frames, derotate, destripe, frame_selection_mask,
                   highpass, klip_annular, nw_ang_comb)
from .metrics import Source
from .space import Config

__all__ = ["reducer_class", "ReductionRequest", "ReductionResult", "Reducer", "Dataset", "KLIPReducer",
           "PartitionedReducer", "EvalImages"]


@dataclass
class ReductionRequest:
    params: Dict[str, Any]
    injections: Optional[Sequence[Source]] = None
    k_scan: bool = False
    tag: str = ""
    fm_sources: Optional[Sequence[Source]] = None   # KLIP-FM test sources (model cube, never added to the data)
    extras: bool = False                            # also produce the nosub (clean pass) / cadi (injected pass) reference image
    threads: int = 1                                # per-target KLIP threads for this partition (set by PartitionedReducer)


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


class KLIPReducer(Reducer):
    """Reference annular KLIP/ADI reducer.

    Parameters (all optional, defaults = ``reduce_near_2`` behaviour)::

        k_klip, bin, filter, n_ang, inrad, outrad, angsep, anglemax,
        corr_thresh, noise_max, coronoise_max, fast, spat_mean, temp_mean, n_min_ref,
        comb_type ('nwadi' | 'mean' | 'median'),
        use_rdi (bool), rdi_mode ('rdi' | 'ardi'), do_destripe (bool), destripe_clip (float),
        fm_selfsub (bool, KLIP-FM self-subtraction terms; default True)

    The chain: inject -> frame selection -> high-pass -> [destripe 90, 0] ->
    angle-aware binning -> annular KLIP [RDI/ARDI basis, KLIP-FM] -> high-pass
    (NaN-aware) -> derotate -> combine [-> mask_fn].

    ``mask_fn(image) -> image`` is the optional legacy post-mask hook
    (``block_burn`` / ``block_airy``), applied to the final science image only.
    """

    name = "klip"
    supports_kscan = True
    supports_fm = True

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
                         "fm_selfsub": True}
        self.defaults.update(defaults or {})
        self._inj_cache: Dict[Tuple, np.ndarray] = {}
        self._fm_cache: Dict[Tuple, np.ndarray] = {}

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

    def _inject(self, sources: Sequence[Source]) -> np.ndarray:
        key = tuple(s.as_tuple() for s in sources)
        if key in self._inj_cache:
            return self._inj_cache[key]
        if self.model is None:
            raise ValueError("KLIPReducer needs an injection_model to inject sources")
        cube = inject_sources(self.data.cube, self.data.angles, sources, self.model, self.pxscale,
                              truenorth=self.truenorth, fallback=self.fallback,
                              angle_convention=self.angle_convention)
        self._inj_cache = {key: cube}      # keep only the latest injected cube
        return cube

    def model_cube(self, sources: Sequence[Source]) -> np.ndarray:
        """KLIP-FM planet-model cube: the sources injected into a ZERO cube with the
        same injection code / conventions as the data (reduce_near_2 L1441-1494)."""
        key = tuple(s.as_tuple() for s in sources)
        if key in self._fm_cache:
            return self._fm_cache[key]
        if self.model is None:
            raise ValueError("KLIPReducer needs an injection_model to build the FM model cube")
        zeros = np.zeros(self.data.cube.shape, np.float32)
        cube = inject_sources(zeros, self.data.angles, sources, self.model, self.pxscale,
                              truenorth=self.truenorth, fallback=self.fallback,
                              angle_convention=self.angle_convention, copy=False)
        self._fm_cache = {key: cube}
        return cube

    def _reference_cube(self, filt: int, meta: Dict[str, Any]) -> Optional[np.ndarray]:
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
            ref = np.stack([highpass(f, filt, nan_aware=True) for f in ref])
        return ref

    # -- main entry ----------------------------------------------------------
    def reduce(self, req: ReductionRequest) -> ReductionResult:
        p = dict(self.defaults)
        p.update({k: v for k, v in req.params.items() if v is not None})
        k = int(max(round(p["k_klip"]), 1))
        bin_ = int(max(round(p["bin"]), 1))
        filt = int(round(p["filter"]))
        inrad = max(float(p["inrad"]) - self.zone_pad, 0.0)
        outrad = min(float(p["outrad"]) + self.zone_pad, self.outrad_cap)
        inrad = min(inrad, outrad - 1.0)
        ds = self.data
        cube = self._inject(req.injections) if req.injections else ds.cube
        angles = ds.angles
        meta: Dict[str, Any] = {}
        do_fm = bool(req.fm_sources)
        if do_fm and req.k_scan:
            meta["fm_note"] = "KLIP-FM skipped: not supported with k_scan"
            do_fm = False
        mcube = self.model_cube(req.fm_sources) if do_fm else None

        keep = frame_selection_mask(ds.nframes, ds.tags, p["corr_thresh"], p["noise_max"],
                                    p["coronoise_max"], bin_, k)
        if not keep.all():
            cube, angles = cube[keep], angles[keep]
            if mcube is not None:
                mcube = mcube[keep]

        if filt > 1:
            cube = np.stack([highpass(f, filt, nan_aware=False) for f in cube])
            if mcube is not None:
                mcube = np.stack([highpass(f, filt, nan_aware=False) for f in mcube])
        if p["do_destripe"]:
            # reduce_near_2 L1672-1677: 90 deg (columns) then 0 deg (rows); science only
            clip = float(p["destripe_clip"])
            cube = np.stack([destripe(destripe(f, 90.0, clip), 0.0, clip) for f in cube])
        dth = self.dth_max_deg(outrad)
        bcube, bang, grp, bkeep = bin_frames(cube, angles, bin_, dth, return_groups=True)
        if mcube is not None:
            mb = bin_by_groups(mcube, grp) if bin_ > 1 else np.asarray(mcube, np.float32)
            mcube = mb[bkeep]

        ref_basis = fm_ref = None
        rdi_mode = str(p["rdi_mode"]).lower()
        if p["use_rdi"]:
            ref = self._reference_cube(filt, meta)
            if ref is not None:
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
                        threads=int(max(getattr(req, "threads", 1), 1)))
        fm_img = nosub = cadi = None
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # nanmedian of all-NaN corners
            img, fm_img, info = self._subtract(bcube, bang, kp, p, filt, req, mcube, ref_basis, fm_ref, meta)
            if req.extras:
                if req.injections:
                    # classical ADI (L2072-2086): 2nd high-pass, median PSF in the detector frame, derotate, median
                    c2 = np.stack([highpass(f, filt, nan_aware=True) for f in bcube]) if filt > 1 else bcube
                    c2 = c2 - np.nanmedian(c2, axis=0)
                    cadi = np.asarray(np.nanmedian(derotate(c2, bang, self.truenorth), axis=0), np.float32)
                else:
                    # no subtraction (L2087-2096): derotate + median stack of the selected/filtered/binned cube
                    nosub = np.asarray(np.nanmedian(derotate(bcube, bang, self.truenorth), axis=0), np.float32)
        if self.mask_fn is not None:
            img = np.stack([self.mask_fn(im) for im in img]) if req.k_scan else self.mask_fn(img)
        meta.update(info)
        meta.update(n_kept=int(keep.sum()), n_binned=int(bcube.shape[0]), inrad=inrad, outrad=outrad,
                    params=p, tag=req.tag)
        return ReductionResult(np.asarray(img, np.float32), self.partition_weight(), meta,
                               fm_image=fm_img, nosub=nosub, cadi=cadi)

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

        def _post(rc: np.ndarray) -> np.ndarray:
            if filt > 1:
                rc = np.stack([highpass(f, filt, nan_aware=True) for f in rc])
            return derotate(rc, bang, self.truenorth)

        def _combine(dr: np.ndarray, ref: Optional[np.ndarray] = None) -> np.ndarray:
            if ct == "median":
                return np.nanmedian(dr, axis=0)
            if ct == "mean":
                return np.nanmean(dr, axis=0)
            return nw_ang_comb(dr, bang, ref_cube=ref)

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
                 injection_model=None if self.model is None else self.model.describe())
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
        >1 dispatches partitions (and the clean pass) concurrently with a thread
        pool; numpy releases the GIL in the heavy linear algebra.
    """

    name = "partitioned"

    def __init__(self, reducers: Dict[Any, Reducer], weighting: str = "equal", max_workers=1,
                 pool: str = "auto", log=None):
        if not reducers:
            raise ValueError("need at least one partition")
        from .parallel import resolve_workers
        self.reducers = dict(reducers)
        self.weighting = weighting
        self.max_workers = resolve_workers(max_workers)
        self.pool_kind = pool            # auto | processes | threads
        self._log = log or (lambda s: None)
        self._procs = None
        if self.max_workers > 1:
            from .parallel import pin_blas
            pin_blas()                      # one BLAS thread per worker thread (no oversubscription)
        first = next(iter(self.reducers.values()))
        self.pxscale, self.fwhm = first.pxscale, first.fwhm
        self.lam_over_d_px = first.lam_over_d_px
        self.angle_convention = first.angle_convention
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
        return next(iter(self.reducers.values())).matched_filter_kernel(rho_as)

    _pool_logged = False

    def start_workers(self) -> str:
        """Start the worker processes (IDL bridges) once the data are loaded.  Returns the
        pool kind in use: ``processes`` (forked workers, one per thread of the budget, up
        to 2 x partitions), ``threads`` (fallback / requested) or ``serial``.

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
            self._log(f"  parallel: {describe(self.max_workers, len(self.reducers))}; pool = {kind}")
            if kind == "threads" and self.pool_kind in ("auto", "processes"):
                self._log("    threads share the GIL, so expect several times less throughput than "
                          "worker processes: worth fixing rather than living with.")
        return kind

    def _pool_kind_now(self) -> str:
        if self.max_workers <= 1:
            return "serial"
        if self._procs is not None:
            return "processes"
        if self.pool_kind in ("auto", "processes"):
            try:
                from .parallel import ProcessPool
                n = min(self.max_workers, 2 * len(self.reducers))
                self._procs = ProcessPool(self.reducers, n, log=self._log)
                return "processes"
            except Exception as exc:
                self._log(f"  parallel: worker processes unavailable ({exc!r}); using threads")
        return "threads"

    def _map(self, jobs):
        if self.max_workers > 1 and len(jobs) > 1:
            if self.pool_kind != "threads" and self._procs is None:
                self.start_workers()
            if self._procs is not None:
                pid_of = {id(r): pid for pid, r in self.reducers.items()}
                return self._procs.map([(pid_of[id(r)], req) for r, req in jobs])
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

    def reduce_config(self, config: Config, injections: Optional[Sequence[Source]] = None,
                      k_scan: bool = False, tag: str = "", selected: Optional[Sequence[Any]] = None,
                      fm_sources: Optional[Sequence[Source]] = None, extras: bool = False
                      ) -> EvalImages:
        """``fm_sources`` requests the KLIP-FM pass (``EvalImages.fm_image``, combined
        with the same partition weights as the science image); ``extras`` the
        nosub / cadi reference images."""
        sel = list(config.selected if selected is None else selected)
        # thread budget: partitions in parallel, the remainder as per-target threads inside each
        # (the runner reduces the injected and clean cubes concurrently -> halve for that)
        conc = 2 if self.max_workers >= 2 else 1
        per = max(self.max_workers // max(conc * len(sel), 1), 1)
        jobs = [(self.reducers[pid], ReductionRequest(config.params_for(pid), injections, k_scan,
                                                      f"{tag}_{pid}", fm_sources=fm_sources, extras=extras,
                                                      threads=per))
                for pid in sel]
        results = self._map(jobs)
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

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(weighting=self.weighting, max_workers=self.max_workers,
                 partitions={str(k): r.describe() for k, r in self.reducers.items()})
        return d


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
