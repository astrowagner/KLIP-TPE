"""Custom-pipeline backends.

Two levels, both documented in ``docs/CUSTOM_PIPELINE.md``:

* :class:`FunctionReducer` -- you supply **one function** that PSF-subtracts, derotates
  and combines a pre-processed cube; everything else (injection, frame selection,
  high-pass, binning, k-scan bookkeeping, products) is inherited from
  :class:`~klip_tpe.reducer.KLIPReducer`.  This is the quickest way to optimize the
  parameters of an existing pipeline.
* :class:`ExternalReducer` -- you own the whole reduction (``reduce(request)``), the
  optimizer only needs images back.  Use it when your pipeline cannot take a cube in
  memory (e.g. it works from files) or does its own injection.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from ..injection import InjectionModel
from ..metrics import Source
from ..reducer import Dataset, KLIPParams, KLIPReducer, Reducer, ReductionRequest, ReductionResult

__all__ = ["FunctionReducer", "ExternalReducer", "SubtractFn"]

#: signature of the user function for :class:`FunctionReducer`::
#:
#:     fn(cube, angles, params, k_scan) -> image            (ny, nx)
#:                                    or  -> cube of images (k, ny, nx) when k_scan
#:
#: ``cube`` (n, ny, nx) float32 is already injected / frame-selected / high-passed /
#: binned; ``angles`` (n,) are the matching parallactic angles (deg; derotate by rotating
#: each frame counter-clockwise by ``angles[i] + truenorth``); ``params`` is the resolved
#: parameter dict (``k_klip, inrad, outrad, n_ang, angsep, anglemax, ...`` plus anything
#: you add to ``extra_params``); ``k_scan`` asks for one image per k in ``1..k_klip``.
SubtractFn = Callable[[np.ndarray, np.ndarray, Dict[str, Any], bool], np.ndarray]


class FunctionReducer(KLIPReducer):
    """Run *your* PSF subtraction on the optimizer's pre-processed cube.

    Parameters
    ----------
    data, pxscale, lam_m, diam_m, injection_model, truenorth, fwhm_px, ...
        as for :class:`~klip_tpe.reducer.KLIPReducer`.
    subtract
        the user function (:data:`SubtractFn`).  It receives the cube after injection,
        frame selection, high-pass and binning and must return the **derotated,
        combined** image (north up when rotated CCW by ``angles + truenorth``).
    supports_kscan
        whether ``subtract`` honours ``k_scan=True`` (returns ``(k, ny, nx)``); if
        False the Runner's scan k-modes are unavailable for this reducer.
    extra_params
        defaults for parameters your function reads that the built-in reducer does not
        know (they become searchable by adding a ``Param`` of the same name to the space).
    name
        label used in logs / setup files.
    """

    supports_fm = False

    def __init__(self, data: Dataset, subtract: SubtractFn, pxscale: float, lam_m: float, diam_m: float,
                 injection_model: Optional[InjectionModel] = None, supports_kscan: bool = False,
                 extra_params: Optional[Dict[str, Any]] = None, name: str = "custom", **kw):
        super().__init__(data, pxscale=pxscale, lam_m=lam_m, diam_m=diam_m, injection_model=injection_model, **kw)
        self._fn = subtract
        self.supports_kscan = bool(supports_kscan)
        self.name = name
        if extra_params:
            self.defaults.update(extra_params)

    def _subtract(self, bcube, bang, kp: KLIPParams, p, filt, req, mcube, ref_basis, fm_ref, meta):
        params = dict(p)
        params.update(k_klip=kp.k_klip, inrad=kp.inrad, outrad=kp.outrad, n_ang=kp.n_ang, angsep=kp.angsep,
                      anglemax=kp.anglemax, truenorth=self.truenorth, lam_over_d_px=self.lam_over_d_px,
                      fwhm_px=self.fwhm, ref_cube=ref_basis)
        img = self._fn(np.asarray(bcube, np.float32), np.asarray(bang, float), params, bool(kp.k_scan))
        img = np.asarray(img, np.float32)
        if kp.k_scan and img.ndim != 3:
            raise ValueError("subtract() must return a (k, ny, nx) cube when k_scan=True")
        if not kp.k_scan and img.ndim != 2:
            raise ValueError("subtract() must return a 2-D image")
        return img, None, {"backend": self.name}

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(backend=self.name, subtract=getattr(self._fn, "__name__", repr(self._fn)))
        return d


class ExternalReducer(Reducer):
    """Wrap a pipeline that does everything itself.

    Subclass (or pass ``reduce_fn``) and implement ``reduce(request) -> ReductionResult``:
    ``request.params`` is the parameter dict for this evaluation, ``request.injections``
    the list of :class:`~klip_tpe.metrics.Source` to add (``rho`` arcsec, ``theta`` deg
    E of N, ``contrast``) or ``None`` for the clean pass, ``request.k_scan`` asks for a
    ``(k, ny, nx)`` cube.  Return the combined, north-up image centred on the star at
    ``((nx-1)/2, (ny-1)/2)`` in ``ReductionResult(image, weight, meta)``.
    ``frame_angles()`` must return the parallactic angles (for the reference-count
    guardrail and the binning-span rule) and ``frame_tags()`` the per-frame quality tags
    or None (then the frame-selection dims should be left out of the space).
    """

    name = "external"
    supports_kscan = False
    supports_fm = False

    def __init__(self, pxscale: float, fwhm_px: float, lam_over_d_px: Optional[float] = None,
                 angles: Optional[Sequence[float]] = None, tags: Optional[Dict[str, np.ndarray]] = None,
                 texp: float = 1.0, reduce_fn: Optional[Callable[[ReductionRequest], ReductionResult]] = None,
                 matched_filter_kernel_fn: Optional[Callable[[float], Optional[np.ndarray]]] = None,
                 name: str = "external"):
        self.pxscale = float(pxscale)
        self.fwhm = float(fwhm_px)
        self.lam_over_d_px = float(lam_over_d_px) if lam_over_d_px else self.fwhm / 1.028
        self._angles = None if angles is None else np.asarray(angles, float)
        self._tags = tags
        self._texp = float(texp)
        self._fn = reduce_fn
        self._mf = matched_filter_kernel_fn
        self.name = name

    def reduce(self, req: ReductionRequest) -> ReductionResult:
        if self._fn is None:
            raise NotImplementedError("subclass ExternalReducer.reduce or pass reduce_fn")
        return self._fn(req)

    def frame_angles(self, pid=None):
        return self._angles

    def frame_tags(self, pid=None):
        return self._tags

    def partition_weight(self, pid=None) -> float:
        return float(np.sqrt(self._texp))

    def matched_filter_kernel(self, rho_as: float):
        return None if self._mf is None else self._mf(rho_as)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(backend=self.name)
        return d
