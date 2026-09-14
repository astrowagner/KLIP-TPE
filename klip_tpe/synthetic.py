"""A synthetic reducer for unit tests and demos: no telescope data, but the same
API, so TPE + calibration + validation + checkpointing can be exercised in seconds.

The "reduction" produces a noise image whose injected-source throughput and
residual noise depend smoothly (and multimodally) on the parameters, with an
integer ``k_klip`` optimum that differs per partition and a decoy: small ``k``
gives high throughput but high noise, like a real over-/under-subtraction
trade-off.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .injection import GaussianPSF, inject_sources
from .metrics import Source
from .reducer import Dataset, KLIPReducer, ReductionRequest, ReductionResult, Reducer

__all__ = ["SyntheticReducer", "make_synthetic_partitions", "synthetic_klip_dataset"]


class SyntheticReducer(Reducer):
    name = "synthetic"
    supports_kscan = True

    def __init__(self, size: int = 100, pxscale: float = 0.05, fwhm: float = 4.0, k_opt: int = 8,
                 bin_opt: float = 12.0, filter_opt: float = 10.0, noise: float = 1.0, star_flux: float = 1e6,
                 seed: int = 0, partition: Any = None, texp: float = 1000.0, fm: bool = False):
        """``fm=True`` advertises ``supports_fm``: a request with ``fm_sources`` also returns
        the noiseless throughput-scaled model response (the KLIP-FM analogue)."""
        self.size, self.pxscale, self.fwhm = int(size), float(pxscale), float(fwhm)
        self.supports_fm = bool(fm)
        self.lam_over_d_px = self.fwhm / 1.028
        self.k_opt, self.bin_opt, self.filter_opt = k_opt, bin_opt, filter_opt
        self.noise0, self.star_flux, self.seed = noise, star_flux, seed
        self.partition = partition
        self.texp = texp
        self.model = GaussianPSF(self.fwhm, star_flux=star_flux)
        self.angles = np.linspace(-20, 20, 200)

    def frame_angles(self, pid=None):
        return self.angles

    def partition_weight(self, pid=None) -> float:
        return float(np.sqrt(self.texp))

    # smooth "physics"
    def throughput(self, k: float, bin_: float, filt: float) -> float:
        t_k = 0.25 + 0.75 * np.exp(-0.5 * ((k - self.k_opt) / 4.0) ** 2) * (1 - 0.5 * np.exp(-k / 3.0))
        t_b = 0.7 + 0.3 * np.exp(-0.5 * ((bin_ - self.bin_opt) / 6.0) ** 2)
        t_f = 0.8 + 0.2 * np.cos(np.pi * (filt - self.filter_opt) / 12.0) ** 2
        return float(t_k * t_b * t_f)

    def noise_level(self, k: float, bin_: float, filt: float, angsep: float = 0.5) -> float:
        n_k = 1.0 + 3.0 * np.exp(-k / 4.0) + 0.02 * k       # under-subtraction at small k, mild growth at large k
        n_b = 1.0 + 0.3 * np.abs(bin_ - self.bin_opt) / 10.0
        n_a = 1.0 + 0.5 * (angsep - 1.0) ** 2
        return float(self.noise0 * n_k * n_b * n_a)

    def _rng(self, req: ReductionRequest) -> np.random.Generator:
        key = repr((sorted((k, round(float(v), 6) if isinstance(v, (int, float)) else str(v))
                           for k, v in req.params.items()),
                    None if not req.injections else [s.as_tuple() for s in req.injections],
                    req.k_scan, self.seed, str(self.partition)))
        h = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        return np.random.default_rng(h)

    def matched_filter_kernel(self, rho_as: float):
        return self.model.matched_filter_kernel(rho_as, self.fwhm)

    def _model_image(self, p: Dict[str, Any], k: int, sources) -> np.ndarray:
        """Noiseless model response: the sources at ``throughput * contrast`` on zeros,
        NaN outside the zone like the science image."""
        bin_ = float(p.get("bin", 10)); filt = float(p.get("filter", 10))
        thr = self.throughput(k, bin_, filt)
        frame = np.zeros((1, self.size, self.size), np.float32)
        src = [Source(s.rho, s.theta, s.contrast * thr) for s in sources]
        frame = inject_sources(frame, np.zeros(1), src, self.model, self.pxscale, truenorth=0.0, copy=False)
        img = np.asarray(frame[0], np.float32).copy()
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        c = (self.size - 1) / 2.0
        rr = np.hypot(xx - c, yy - c)
        img[(rr < float(p.get("inrad", 0)) - 2) | (rr > float(p.get("outrad", 45)) + 2)] = np.nan
        return img

    def _image(self, p: Dict[str, Any], k: int, injections, rng) -> np.ndarray:
        bin_ = float(p.get("bin", 10)); filt = float(p.get("filter", 10)); asep = float(p.get("angsep", 0.5))
        img = rng.standard_normal((self.size, self.size)) * self.noise_level(k, bin_, filt, asep)
        # a few static speckles so clean-subtraction has something to do
        yy, xx = np.mgrid[0:self.size, 0:self.size]
        c = (self.size - 1) / 2.0
        for (dx, dy, amp) in ((9, 4, 6.0), (-7, 8, 4.0), (3, -11, 5.0)):
            img += amp * self.noise0 * np.exp(-0.5 * ((xx - c - dx) ** 2 + (yy - c - dy) ** 2) / (self.fwhm / 2.3548) ** 2)
        rr = np.hypot(xx - c, yy - c)
        img[(rr < float(p.get("inrad", 0)) - 2) | (rr > float(p.get("outrad", 45)) + 2)] = np.nan
        if injections:
            thr = self.throughput(k, bin_, filt)
            frame = np.zeros((1, self.size, self.size), np.float32)
            src = [Source(s.rho, s.theta, s.contrast * thr) for s in injections]
            frame = inject_sources(frame, np.zeros(1), src, self.model, self.pxscale, truenorth=0.0, copy=False)
            img = img + frame[0]
        return img.astype(np.float32)

    def reduce(self, req: ReductionRequest) -> ReductionResult:
        rng = self._rng(req)
        p = req.params
        k = int(max(round(float(p.get("k_klip", 5))), 1))
        fm_img = None
        if req.k_scan:
            img = np.stack([self._image(p, kk, req.injections, rng) for kk in range(1, k + 1)])
        else:
            img = self._image(p, k, req.injections, rng)
            if self.supports_fm and req.fm_sources:
                fm_img = self._model_image(p, k, req.fm_sources)
        return ReductionResult(img, self.partition_weight(), {"k": k, "partition": self.partition}, fm_image=fm_img)


def make_synthetic_partitions(ids: Sequence[Any], k_opts: Optional[Sequence[int]] = None, **kw) -> Dict[Any, SyntheticReducer]:
    k_opts = k_opts or [6 + 3 * i for i in range(len(ids))]
    return {pid: SyntheticReducer(k_opt=int(k), seed=i, partition=pid, **kw)
            for i, (pid, k) in enumerate(zip(ids, k_opts))}


def synthetic_klip_dataset(nframes: int = 60, size: int = 64, pa_span: float = 40.0, seed: int = 1,
                           speckle_amp: float = 50.0, noise: float = 1.0) -> Dataset:
    """A tiny pupil-tracking cube with a quasi-static speckle field for testing the
    real :class:`~klip_tpe.reducer.KLIPReducer` end to end."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    rr = np.hypot(xx - c, yy - c)
    static = np.zeros((size, size))
    for _ in range(25):
        x0, y0 = rng.uniform(c - 20, c + 20, 2)
        static += rng.uniform(0.3, 1.0) * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / 1.5 ** 2)
    static *= speckle_amp * np.exp(-rr / 15.0)
    cube = np.empty((nframes, size, size), np.float32)
    for i in range(nframes):
        drift = 1.0 + 0.1 * np.sin(i / 7.0)
        cube[i] = static * drift + rng.standard_normal((size, size)) * noise
    angles = np.linspace(-pa_span / 2, pa_span / 2, nframes)
    tags = {"corrs": np.clip(rng.normal(0.98, 0.01, nframes), 0, 1),
            "noises": np.abs(rng.normal(1.0, 0.1, nframes)), "coronoise": np.abs(rng.normal(1.0, 0.1, nframes))}
    return Dataset(cube, angles, tags, texp=float(nframes), name=f"synthetic{seed}")
