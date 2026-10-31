"""GPUKLIPReducer: klip-tpe's KLIPReducer with ``reduce()`` on the card for the requests it supports.

    from gpu_reducer import GPUKLIPReducer
    g = GPUKLIPReducer.from_cpu(reducer)          # reducer: a KLIPReducer with its data loaded and edited
    res = g.reduce(req)                           # ReductionResult, as KLIPReducer.reduce returns it

For a request :meth:`supports` accepts (a plain reduction: no KLIP-FM pass, no k-scan, no extras,
no reference library or SDI pool, no destriping, no centring, no explicit fast path) the chain of
:mod:`klip_gpu` runs on the device and returns the night's image, with the meta keys
``KLIPReducer._reduce_prepared`` writes.  Any other request goes to ``KLIPReducer.reduce`` on the
CPU, so the reducer is complete either way; the pool should route such jobs to CPU workers.

The CUDA context is made at the first reduction, never at construction, so the reducers can be
built in the parent and used in a forked worker.  The cube goes to the card once (float32) and
stays; a request moves only its parameters in and one image out.

Settings (constructor): ``eigh`` ("xbatched64", the validated default; "jacobi32" with ``rr``;
later a float32 solver of our own), ``compact`` (the size-class step of the per-target
eigenproblems, 0 = padded), ``frames_dtype`` (float32: the frames' precision on the card).

Results agree with KLIPReducer.reduce to ~1e-5 of the image noise on a run's frames (NEAR2 s5,
2026-10-31), not to the bit: the Gram's dot products and the solver differ in rounding.  The same
job on the card twice gives the same image to the bit (deterministic kernels).  Tests:
``tests/test_gpu_reducer.py`` (synthetic nights; the NumPy backend always, the card when there is
one), ``scripts/check_gpu_reducer.py`` (a finished run's frames and configurations).

Written in the NEAR2 GPU session (bench/gpu_reducer/ on patrick), 2026-10-31.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from klip_tpe.reducer import KLIPReducer, ReductionRequest, ReductionResult, _flag, _zone_center, _zone_ext

try:
    from . import klip_gpu as kg          # as klip_tpe.gpu_reducer, next to klip_tpe/klip_gpu.py
except ImportError:
    import klip_gpu as kg                 # standalone, in bench/gpu_reducer/

VERSION = "0.6 (2026-10-31, name in warm(), for klip_tpe/)"


class GPUKLIPReducer(KLIPReducer):
    name = "klip-gpu"

    #: the settings, as class attributes so an instance built with __new__ has them
    eigh_mode: str = "xbatched64"
    compact: int = 0
    rr: bool = False
    batch_sectors: bool = True
    frames_dtype = np.float32
    cpu_fallback: bool = True
    backend_name: str = "cupy"       # "numpy" runs the same chain on the host (tests)

    @classmethod
    def from_cpu(cls, cpu: KLIPReducer, eigh: str = "xbatched64", compact: int = 0, rr: bool = False,
                 frames_dtype=np.float32, cpu_fallback: bool = True, batch_sectors: bool = True) -> "GPUKLIPReducer":
        """The same reducer (its data with the run's edits, its model, its defaults) with
        ``reduce`` on the card."""
        g = cls.__new__(cls)
        g.__dict__.update(cpu.__dict__)
        g.__dict__.update(_inj_cache={}, _fm_cache={}, _cost_cache=dict(getattr(cpu, "_cost_cache", {})))
        g.__setstate__(g.__dict__.copy())          # fresh lock and crop caches
        g.eigh_mode, g.compact, g.rr = str(eigh), int(compact), bool(rr)
        g.frames_dtype, g.cpu_fallback = frames_dtype, bool(cpu_fallback)
        g.batch_sectors = bool(batch_sectors)
        g._gpu = None
        g._zones = {}
        g.stats = {"gpu_jobs": 0, "gpu_s": 0.0, "cpu_jobs": 0, "cpu_s": 0.0}
        return g

    # -- pickling: the device state never travels --------------------------------
    def __getstate__(self):
        d = super().__getstate__()
        d["_gpu"] = None
        d["_zones"] = {}
        return d

    def __setstate__(self, d):
        super().__setstate__(d)
        self.__dict__.setdefault("_gpu", None)
        self.__dict__.setdefault("_zones", {})
        self.__dict__.setdefault("stats", {"gpu_jobs": 0, "gpu_s": 0.0, "cpu_jobs": 0, "cpu_s": 0.0})

    # -- the device -----------------------------------------------------------------
    def _device(self) -> Dict[str, Any]:
        """The backend and the cube on the card, made at the first call (after any fork)."""
        if self._gpu is None:
            be = kg.Backend(self.backend_name)
            cube = be.to_dev(self.data.cube, np.float32)
            self._gpu = {"be": be, "cube": cube, "name": be.device_name()}
        return self._gpu

    def warm(self) -> Dict[str, Any]:
        """Make the CUDA context and put the cube on the card, before the first job (the card
        worker calls it once after the fork, on each partition).  Returns what a log line needs:
        ``name`` (the card), ``driver`` (the kernel driver, from /proc/driver/nvidia/version; None
        where that file is missing), the CUDA and CuPy versions, the memory in use on the card and
        the cube's size, and the reducer's version and settings.  Raises when CuPy or the card is
        missing, so the caller can go on CPU-only.  With the NumPy backend it returns the same keys
        for the host (tests)."""
        import os
        dev = self._device()
        be = dev["be"]
        out: Dict[str, Any] = {"name": dev["name"], "device": dev["name"], "driver": None, "backend": be.name,
                               "cube_gb": round(self.data.cube.nbytes / 1e9, 3), "pid": os.getpid(),
                               "reducer": VERSION, "eigh": self.eigh_mode, "compact": self.compact,
                               "batch_sectors": self.batch_sectors}
        if be.name != "cupy":
            return out
        cp = be.cp
        free, total = cp.cuda.runtime.memGetInfo()
        try:
            import re
            with open("/proc/driver/nvidia/version") as f:
                m = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", f.readline())
                out["driver"] = m.group(1) if m else None
        except Exception:  # noqa: BLE001
            pass
        out.update(cuda_driver_api=int(cp.cuda.runtime.driverGetVersion()),
                   cuda_runtime=int(cp.cuda.runtime.runtimeGetVersion()), cupy=cp.__version__,
                   memory_used_gb=round((total - free) / 1e9, 2), memory_total_gb=round(total / 1e9, 1))
        return out

    def release(self) -> None:
        self._gpu = None
        self._zones = {}

    # -- what goes to the card ---------------------------------------------------
    def supports(self, req: ReductionRequest) -> bool:
        """Whether ``reduce`` runs this request on the card (otherwise on the CPU)."""
        if str(getattr(req, "stage", "reduce")) != "reduce":
            return False
        if req.fm_sources or req.k_scan or req.extras or getattr(req, "siblings", None) is not None:
            return False
        p = dict(self.defaults)
        p.update({k: v for k, v in req.params.items() if v is not None})
        for key in ("fast", "spat_mean", "temp_mean", "use_rdi", "do_destripe"):
            if _flag(p.get(key, False)):
                return False
        if str(p.get("comb_type", "nwadi")) not in ("nwadi", "median", "mean"):
            return False
        if self._reflib_spec is not None or int(round(float(p.get("nkeep_sdi", 0) or 0))) > 0:
            return False
        return True

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(device="gpu", gpu_reducer=VERSION, eigh=self.eigh_mode, compact=self.compact, rr=self.rr,
                 batch_sectors=self.batch_sectors, frames_dtype=np.dtype(self.frames_dtype).name)
        return d

    # -- the reduction ---------------------------------------------------------------
    def reduce(self, req: ReductionRequest) -> ReductionResult:
        if not self.supports(req):
            if not self.cpu_fallback:
                raise ValueError(f"GPUKLIPReducer does not support this request (tag {req.tag!r})")
            t0 = time.perf_counter()
            res = KLIPReducer.reduce(self, req)
            res.meta["device"] = "cpu"
            self.stats["cpu_jobs"] += 1
            self.stats["cpu_s"] += time.perf_counter() - t0
            return res
        t0 = time.perf_counter()
        res = self._reduce_gpu(req)
        self.stats["gpu_jobs"] += 1
        self.stats["gpu_s"] += time.perf_counter() - t0
        return res

    #: per-step times of the last reduction (seconds), kept when ``profile`` is True (each step
    #: then waits for the card, which costs a little)
    profile: bool = False
    last_steps: Optional[Dict[str, float]] = None

    def _reduce_gpu(self, req: ReductionRequest) -> ReductionResult:
        from klip_tpe.klip import bin_groups, frame_selection_mask
        dev = self._device()
        be, xp = dev["be"], dev["be"].xp
        ds = self.data
        steps: Dict[str, float] = {}
        t_last = [time.perf_counter()]

        def lap(name):
            if self.profile:
                be.sync()
                now = time.perf_counter()
                steps[name] = steps.get(name, 0.0) + now - t_last[0]
                t_last[0] = now
        # ---- the geometry and the integer settings, as KLIPReducer.prepare derives them
        p = dict(self.defaults)
        p.update({k: v for k, v in req.params.items() if v is not None})
        k = int(max(round(p["k_klip"]), 1))
        bin_ = int(max(round(p["bin"]), 1))
        filt = int(round(p["filter"]))
        pad_in = self.zone_pad if _flag(p.get("pad_in", True)) else 0.0
        pad_out = self.zone_pad if _flag(p.get("pad_out", True)) else 0.0
        ann_in = max(float(p["inrad"]) - pad_in, 0.0)
        ann_out = min(float(p["outrad"]) + pad_out, self.outrad_cap)
        ann_in = min(ann_in, ann_out - 1.0)
        z_in, z_out = _zone_ext(p)
        inrad = max(ann_in - z_in, 0.0)
        outrad = min(ann_out + z_out, self.outrad_cap)
        inrad = min(inrad, outrad - 1.0)
        widened = (inrad != ann_in) or (outrad != ann_out)
        dth = self.dth_max_deg(ann_out)
        hp_binned = str(p.get("hp_order", "raw")).lower() == "binned" and filt > 1
        hp_nan = _flag(p.get("hp_nan_aware", False))
        post_hp = filt > 1 and _flag(p.get("post_hp", True))
        keep = frame_selection_mask(ds.nframes, ds.tags, p["corr_thresh"], p["noise_max"], p["coronoise_max"], bin_, k)
        meta: Dict[str, Any] = {}
        lap("setup (host)")
        # ---- the frames: inject, select, high-pass, bin (or bin, then high-pass)
        # the selection reads only the frame tags, so the kept frames alone are injected (as
        # KLIPReducer injects only the kept frames): the same numbers, fewer frames to touch
        cube = dev["cube"]
        a_ = np.asarray(ds.angles, float)[keep]
        if not keep.all():
            cube = cube[xp.asarray(np.flatnonzero(keep))]
        lap("select")
        if req.injections:
            cube = kg.inject(be, cube, a_, req.injections, self.model, self.pxscale, self.truenorth,
                             self.angle_convention, fallback=self.fallback)
        lap("inject")
        grp = bin_groups(a_, bin_, dth) if bin_ > 1 else np.arange(a_.size)
        if bin_ > 1:
            ng = int(grp.max()) + 1
            bang = np.bincount(grp, weights=a_, minlength=ng) / np.bincount(grp, minlength=ng)
        else:
            bang = a_.copy()
        if filt > 1 and not hp_binned:
            cube = kg.highpass(be, cube, filt, nan_aware=hp_nan, dtype=self.frames_dtype)
        lap("highpass")
        if bin_ > 1:
            Bm = xp.asarray(kg.bin_matrix(grp).astype(np.float64 if hp_binned else self.frames_dtype))
            bc = (Bm @ cube.reshape(cube.shape[0], -1).astype(Bm.dtype)).reshape(Bm.shape[0], *cube.shape[1:])
        else:
            bc = cube
        if filt > 1 and hp_binned:
            bc = kg.highpass(be, bc, filt, nan_aware=hp_nan, dtype=np.float64)
        bc = bc.astype(xp.float32)
        del cube
        # bins whose frames are all NaN (or sum to 0) are dropped, as bin_frames drops them
        tot = xp.nansum(bc.reshape(bc.shape[0], -1), axis=1)
        bkeep = be.asnumpy(tot != 0)
        if not bkeep.all():
            bc = bc[xp.asarray(np.flatnonzero(bkeep))]
            bang = bang[bkeep]
        if bc.shape[0] == 0:
            raise ValueError(f"every frame was dropped as empty after filtering and binning (filter={filt}, bin={bin_})")
        lap("bin")
        # ---- KLIP
        res, info = kg.klip_annular(be, bc, bang, k=k, inrad=inrad, outrad=outrad, n_ang=int(max(round(p["n_ang"]), 1)),
                                    angsep=float(p["angsep"]), anglemax=float(p["anglemax"]),
                                    n_min_ref=int(p["n_min_ref"]), lam_over_d_px=self.lam_over_d_px,
                                    arc_radii=(ann_in, ann_out) if widened else None, zone_center=_zone_center(p),
                                    zone_angles=str(p.get("zone_angles") or "ccw"), eigh_mode=self.eigh_mode,
                                    compact=self.compact, rr=self.rr, batch_sectors=self.batch_sectors,
                                    zone_cache=self._zones)
        del bc
        lap("klip")
        # ---- the residuals' high-pass, the derotation, the combination
        if post_hp:
            res = kg.highpass(be, res, filt, nan_aware=True, dtype=self.frames_dtype)
        lap("post highpass")
        dr = kg.rotate(be, res, bang + self.truenorth, cval=np.nan).astype(xp.float32)
        del res
        lap("derotate")
        ct = str(p.get("comb_type", "nwadi"))
        if ct in ("median", "mean"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns outside the zone
                img = xp.nanmedian(dr, axis=0) if ct == "median" else xp.nanmean(dr, axis=0)
        else:
            img = kg.nw_ang_comb(be, dr, bang)
        del dr
        lap("combine")
        out = be.asnumpy(img).astype(np.float32)
        lap("download")
        if self.mask_fn is not None:
            out = self.mask_fn(out)
        meta.update({kk: v for kk, v in info.items()})
        if widened:
            meta["annulus"] = [ann_in, ann_out]
        meta.update(n_kept=int(keep.sum()), n_binned=int(bang.size), inrad=inrad, outrad=outrad, params=p, tag=req.tag,
                    device="gpu", eigh=self.eigh_mode, compact=self.compact, batch_sectors=self.batch_sectors)
        if self.profile:
            lap("meta")
            self.last_steps = steps
        return ReductionResult(np.asarray(out, np.float32), self.partition_weight(), meta)


def wrap_partitions(pr, **kw) -> Dict[Any, GPUKLIPReducer]:
    """``{pid: GPUKLIPReducer}`` for every KLIPReducer of a PartitionedReducer (the others are
    left out)."""
    out = {}
    for pid, r in pr.reducers.items():
        if isinstance(r, KLIPReducer) and not isinstance(r, GPUKLIPReducer):
            out[pid] = GPUKLIPReducer.from_cpu(r, **kw)
    return out
