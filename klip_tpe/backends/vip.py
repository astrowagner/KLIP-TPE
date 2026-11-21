"""VIP backend (https://vip.readthedocs.io).

:class:`VIPReducer` runs ``vip_hci.psfsub.pca_annular`` (the default), ``pca`` (full frame) or
``median_sub`` (classical ADI) on the optimizer's pre-processed cube.  VIP derotates, counterclockwise
by ``angle_list`` as in this package, and combines the frames itself.  The searched parameters map to
VIP's as follows:

==============  ==============================================================
``k_klip``       ``ncomp`` (k-scan: one call per k, since VIP's tuple ``ncomp`` means
                 values per annulus, not a scan)
``n_ang``        ``n_segments`` (``pca_annular``, while ``median_sub`` has no segments and
                 uses ``nframes = 2 k`` references closest in angle)
``inrad/outrad`` ``radius_int`` and ``asize = outrad - inrad`` (one annulus, which
                 ``n_annuli`` splits)
``angsep``       ``delta_rot = angsep * (lambda/D) / FWHM`` (VIP counts in FWHM)
``anglemax``     not available in VIP (ignored, so best left out of the space)
``comb_type``    ``collapse`` (``'median'`` or ``'mean'``, ``'nwadi'`` -> ``'median'``)
==============  ==============================================================

The extra defaults are ``algo`` (``'pca_annular'``, ``'pca'`` or ``'median_sub'``), ``n_annuli``,
``svd_mode``, ``imlib`` and ``interpolation`` (VIP's rotation library), ``min_frames_lib``,
``max_frames_lib`` and ``nproc``.  For RDI and ARDI ``Dataset.ref_cube`` goes to VIP as ``cube_ref``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..reducer import Dataset, KLIPParams, KLIPReducer

__all__ = ["VIPReducer"]


def _require_vip():
    """Check that VIP imports.

    Raises
    ------
    ImportError
        If ``vip_hci`` is not installed.
    """
    try:
        import vip_hci  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("VIPReducer needs vip_hci: pip install vip_hci") from exc


class VIPReducer(KLIPReducer):
    """Reducer that runs VIP's PSF subtraction on the optimizer's pre-processed cube.

    It takes the searched parameters as the module maps them and VIP's own as extra defaults, added under
    the caller's, and emulates the k-scan with one VIP call per k.  Making one checks that VIP imports.

    Parameters
    ----------
    data : Dataset
        The data set.
    pxscale : float
        Pixel scale, in arcseconds per pixel.
    lam_m : float
        Wavelength, in meters.
    diam_m : float
        Telescope diameter, in meters.
    kw : dict
        The other keywords of :class:`~klip_tpe.reducer.KLIPReducer`.

    Attributes
    ----------
    name : str
        ``"vip"``.
    supports_kscan : bool
        True, the scan emulated.
    supports_fm : bool
        False, since VIP gives this package no forward model.

    Raises
    ------
    ImportError
        If ``vip_hci`` is not installed.
    """
    name = "vip"
    supports_kscan = True          # emulated: one VIP call per k
    supports_fm = False

    def __init__(self, data: Dataset, pxscale: float, lam_m: float, diam_m: float, **kw):
        _require_vip()
        super().__init__(data, pxscale=pxscale, lam_m=lam_m, diam_m=diam_m, **kw)
        for k, v in dict(algo="pca_annular", n_annuli=1, svd_mode="lapack", imlib="opencv",
                         interpolation="bilinear", min_frames_lib=2, max_frames_lib=200, nproc=1).items():
            self.defaults.setdefault(k, v)

    def _one(self, bcube, bang, kp: KLIPParams, p, ncomp: int, cube_ref):
        """Return one VIP reduction of a binned cube at ``ncomp`` components.

        The angles take the true-north offset, the annulus is ``radius_int`` with ``asize`` per annulus of
        ``n_annuli``, and ``delta_rot`` is ``angsep`` in FWHM, as VIP counts.

        Parameters
        ----------
        bcube : ndarray, shape (nb, ny, nx)
            The binned cube.
        bang : ndarray, shape (nb,)
            Parallactic angle of each frame, in degrees.
        kp : KLIPParams
            The KLIP settings.
        p : dict
            The request's parameters with VIP's defaults.
        ncomp : int
            Number of principal components, or for ``median_sub`` half the references.
        cube_ref : ndarray or None
            The reference cube for RDI or ARDI, None for ADI.

        Returns
        -------
        image : ndarray, shape (ny, nx)
            The derotated, combined image.

        Raises
        ------
        ValueError
            If ``algo`` is unknown.
        """
        import warnings
        from vip_hci.psfsub import median_sub, pca, pca_annular
        ang = np.asarray(bang, float) + self.truenorth
        collapse = "mean" if p["comb_type"] == "mean" else "median"
        width = max(kp.outrad - kp.inrad, 1.0)
        n_ann = max(int(p["n_annuli"]), 1)
        asize = width / n_ann
        delta_rot = float(kp.angsep) * self.lam_over_d_px / self.fwhm
        algo = str(p["algo"])
        common = dict(imlib=str(p["imlib"]), interpolation=str(p["interpolation"]), collapse=collapse,
                      nproc=int(p["nproc"]), verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if algo == "pca_annular":
                return pca_annular(np.asarray(bcube, np.float32), ang, cube_ref=cube_ref, radius_int=int(round(kp.inrad)),
                                   fwhm=float(self.fwhm), asize=float(asize), n_segments=int(kp.n_ang),
                                   delta_rot=delta_rot, ncomp=int(ncomp), svd_mode=str(p["svd_mode"]),
                                   min_frames_lib=int(p["min_frames_lib"]), max_frames_lib=int(p["max_frames_lib"]),
                                   **common)
            if algo == "pca":
                return pca(np.asarray(bcube, np.float32), ang, cube_ref=cube_ref, ncomp=int(ncomp),
                           svd_mode=str(p["svd_mode"]), mask_center_px=int(round(kp.inrad)) or None,
                           delta_rot=delta_rot if delta_rot > 0 else None, fwhm=float(self.fwhm), **common)
            if algo == "median_sub":
                return median_sub(np.asarray(bcube, np.float32), ang, fwhm=float(self.fwhm),
                                  radius_int=int(round(kp.inrad)), asize=int(max(round(asize), 1)),
                                  delta_rot=delta_rot, mode="annular", nframes=int(max(ncomp, 1)) * 2,
                                  cube_ref=cube_ref, **common)
        raise ValueError(f"VIPReducer: unknown algo {algo!r}")

    def _subtract(self, bcube, bang, kp: KLIPParams, p, filt, req, mcube, ref_basis, fm_ref, meta):
        """Subtract the PSF with VIP, one image per number of KL modes in a k-scan.

        With ``use_rdi`` the reference cube goes to VIP as ``cube_ref``, the science frames added to it for
        ARDI.  The image is cut to the zone, as the base class cuts it.

        Parameters
        ----------
        bcube : ndarray, shape (nb, ny, nx)
            The prepared frames, injected, selected, high-passed and binned.
        bang : ndarray, shape (nb,)
            Parallactic angle of each frame, in degrees.
        kp : KLIPParams
            The KLIP settings.
        p : dict
            The request's parameters.
        filt : int
            Width of the high-pass box in pixels, which picks the reference cube.
        req : ReductionRequest
            The request, unused.
        mcube : ndarray or None
            The KLIP-FM model cube, unused.
        ref_basis : ndarray or None
            Reference frames for the basis, unused.
        fm_ref : ndarray or None
            The planet model in ``ref_basis``, unused.
        meta : dict
            The reduction's notes.

        Returns
        -------
        image : ndarray, shape (ny, nx) or (k_klip, ny, nx)
            The image, or one per number of KL modes with ``kp.k_scan``, NaN outside the zone.
        fm_image : None
            No KLIP-FM image.
        info : dict
            ``backend`` and ``algo``.
        """
        cube_ref = None
        if p["use_rdi"]:
            ref = self._reference_cube(filt, meta)
            if ref is not None:
                cube_ref = np.asarray(ref, np.float32)
                if str(p["rdi_mode"]).lower() == "ardi":
                    cube_ref = np.concatenate([cube_ref, np.asarray(bcube, np.float32)], axis=0)
        ks = range(1, kp.k_klip + 1) if kp.k_scan else [kp.k_klip]
        imgs = []
        for k in ks:
            im = self._one(bcube, bang, kp, p, k, cube_ref)
            im = np.asarray(im, np.float32)
            ny, nx = bcube.shape[1:]
            yy, xx = np.mgrid[0:ny, 0:nx]
            rr = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
            im = np.where((rr >= kp.inrad) & (rr <= kp.outrad), im, np.nan)     # zone only, like the base
            imgs.append(im)
        img = np.stack(imgs) if kp.k_scan else imgs[0]
        return img, None, {"backend": "vip", "algo": p["algo"]}

    def describe(self) -> Dict[str, Any]:
        """Return the reducer's settings with ``backend``, for a run's record.

        Returns
        -------
        info : dict
            The base settings and ``backend``, ``"vip"``.
        """
        d = super().describe()
        d.update(backend="vip")
        return d
