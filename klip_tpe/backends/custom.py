"""Backends for a pipeline of your own, at two levels, both in ``docs/CUSTOM_PIPELINE.md``.

* :class:`FunctionReducer` takes one function of yours that PSF-subtracts, derotates and combines a
  prepared cube.  Everything else, the injection, frame selection, high-pass filter, binning, k-scan
  bookkeeping and products, comes from :class:`~klip_tpe.reducer.KLIPReducer`.  It is the quickest way
  to optimize the parameters of a pipeline you have.
* :class:`ExternalReducer` leaves the whole reduction to you (``reduce(request)``), and the search only
  needs images back.  It suits a pipeline that cannot take a cube in memory, such as one that works
  from files, or one that injects sources itself.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from ..injection import InjectionModel
from ..metrics import Source
from ..reducer import Dataset, KLIPParams, KLIPReducer, Reducer, ReductionRequest, ReductionResult

__all__ = ["FunctionReducer", "ExternalReducer", "SubtractFn"]

#: signature of the function :class:`FunctionReducer` takes::
#:
#:     fn(cube, angles, params, k_scan) -> image (ny, nx), or a cube of images (k, ny, nx) with k_scan
#:
#: ``cube`` (n, ny, nx), float32, is already injected, frame-selected, high-pass filtered and binned.
#: ``angles`` (n,) are its parallactic angles in degrees, and a frame is derotated by turning it
#: counter-clockwise by ``angles[i] + truenorth``.  ``params`` is the resolved parameter dict
#: (``k_klip``, ``inrad``, ``outrad``, ``n_ang``, ``angsep``, ``anglemax`` and the rest, with anything
#: added to ``extra_params``).  ``k_scan`` asks for one image per k in ``1..k_klip``.
SubtractFn = Callable[[np.ndarray, np.ndarray, Dict[str, Any], bool], np.ndarray]


class FunctionReducer(KLIPReducer):
    """Run your own PSF subtraction on the search's prepared cube.

    Parameters
    ----------
    data : Dataset
        The pre-processed frames.
    subtract : callable
        Your function (:data:`SubtractFn`).  It receives the cube after the injection, frame selection,
        high-pass filter and binning, and returns the derotated, combined image, north up when the frames
        are turned counter-clockwise by ``angles + truenorth``.
    pxscale : float
        Plate scale, in arcsec per pixel.
    lam_m : float
        Wavelength, in meters.
    diam_m : float
        Telescope diameter, in meters.
    injection_model : InjectionModel or None, default None
        The model fake sources are injected with, as :class:`~klip_tpe.reducer.KLIPReducer` takes it.
    supports_kscan : bool, default False
        Whether ``subtract`` honors ``k_scan=True`` and returns ``(k, ny, nx)``.  Without it the run's
        scanned k modes are not available.
    extra_params : dict or None, default None
        Defaults of parameters your function reads that the built-in reducer does not know.  A ``Param``
        of the same name in the search space makes one searched.
    name : str, default "custom"
        Label of the backend, in logs and setup files.
    kw : dict
        Other arguments of :class:`~klip_tpe.reducer.KLIPReducer`, such as ``truenorth`` and ``fwhm_px``.

    Attributes
    ----------
    supports_fm : bool
        False, since your function gives no forward model.
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
        """Run your function on the prepared cube, in place of the built-in KLIP step.

        Parameters
        ----------
        bcube : ndarray, shape (nb, ny, nx)
            The prepared frames, injected, selected, high-pass filtered and binned.
        bang : ndarray, shape (nb,)
            Parallactic angle of each frame, in degrees.
        kp : KLIPParams
            The KLIP settings.
        p : dict
            The request's parameters.
        filt : int
            Width of the high-pass box, in pixels, not used here.
        req : ReductionRequest
            The request, not used here.
        mcube : ndarray or None
            The KLIP-FM model cube, not used here.
        ref_basis : ndarray or None
            Reference frames for the basis, handed to your function as ``params["ref_cube"]``.
        fm_ref : ndarray or None
            The planet model in ``ref_basis``, not used here.
        meta : dict
            The reduction's notes, not used here.

        Returns
        -------
        image : ndarray, shape (ny, nx) or (k, ny, nx)
            Your function's image, or one per number of KL modes with ``kp.k_scan``.
        fm_image : None
            No forward model.
        info : dict
            ``backend``, the backend's name.

        Raises
        ------
        ValueError
            If your function's result has the wrong number of dimensions.
        """
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
        """Return the reducer's setup, with the backend's name and your function's.

        Returns
        -------
        setup : dict
            :meth:`klip_tpe.reducer.KLIPReducer.describe`'s, with ``backend`` and ``subtract``.
        """
        d = super().describe()
        d.update(backend=self.name, subtract=getattr(self._fn, "__name__", repr(self._fn)))
        return d


class ExternalReducer(Reducer):
    """Wrap a pipeline that does the whole reduction itself.

    Subclass it and override :meth:`reduce`, or pass ``reduce_fn``.  ``request.params`` holds the
    evaluation's parameters, ``request.injections`` the sources to add (:class:`~klip_tpe.metrics.Source`,
    ``rho`` in arcsec, ``theta`` in degrees East of North, and ``contrast``) or None for the clean pass, and
    ``request.k_scan`` asks for a ``(k, ny, nx)`` cube.  The result is the combined, north-up image with
    the star at ``((nx - 1) / 2, (ny - 1) / 2)``, in ``ReductionResult(image, weight, meta)``.

    Parameters
    ----------
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm_px : float
        FWHM, in pixels.
    lam_over_d_px : float or None, default None
        lambda/D, in pixels, None for ``fwhm_px / 1.028``.
    angles : sequence of float or None, default None
        The frames' parallactic angles, for the reference-count guard and the binning's span rule.
    tags : dict of str to ndarray or None, default None
        Each frame's quality tags, None when there are none.  The frame-selection dimensions then stay out
        of the search space.
    texp : float, default 1.0
        Exposure time, whose square root is the reducer's weight among partitions.
    reduce_fn : callable or None, default None
        Function ``reduce_fn(request) -> ReductionResult``, None for a subclass.
    matched_filter_kernel_fn : callable or None, default None
        Function ``f(rho_as) -> kernel`` of the matched filter at a separation, None for the metric's own.
    name : str, default "external"
        Label of the backend.

    Attributes
    ----------
    supports_kscan : bool
        False unless a subclass sets it.
    supports_fm : bool
        False, since the pipeline gives no forward model.
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
        """Reduce one request with your pipeline.

        Parameters
        ----------
        req : ReductionRequest
            The request.

        Returns
        -------
        result : ReductionResult
            ``reduce_fn``'s result.

        Raises
        ------
        NotImplementedError
            If there is no ``reduce_fn`` and no subclass overrides this method.
        """
        if self._fn is None:
            raise NotImplementedError("subclass ExternalReducer.reduce or pass reduce_fn")
        return self._fn(req)

    def frame_angles(self, pid=None):
        """Return the frames' parallactic angles.

        Parameters
        ----------
        pid : object, default None
            The partition, not used.

        Returns
        -------
        angles : ndarray or None
            The angles, in degrees, None when none were given.
        """
        return self._angles

    def frame_tags(self, pid=None):
        """Return the frames' quality tags.

        Parameters
        ----------
        pid : object, default None
            The partition, not used.

        Returns
        -------
        tags : dict or None
            The tags, None when none were given.
        """
        return self._tags

    def partition_weight(self, pid=None) -> float:
        """Return the reducer's weight among partitions, the square root of its exposure time.

        Parameters
        ----------
        pid : object, default None
            The partition, not used.

        Returns
        -------
        weight : float
            The weight.
        """
        return float(np.sqrt(self._texp))

    def matched_filter_kernel(self, rho_as: float):
        """Return the matched filter's kernel at a separation.

        Parameters
        ----------
        rho_as : float
            The separation, in arcsec.

        Returns
        -------
        kernel : ndarray or None
            ``matched_filter_kernel_fn``'s kernel, None without it, for the metric's own.
        """
        return None if self._mf is None else self._mf(rho_as)

    def describe(self) -> Dict[str, Any]:
        """Return the reducer's setup, with the backend's name.

        Returns
        -------
        setup : dict
            :meth:`klip_tpe.reducer.Reducer.describe`'s, with ``backend``.
        """
        d = super().describe()
        d.update(backend=self.name)
        return d
