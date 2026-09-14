"""VIP backend (https://vip.readthedocs.io).

:class:`VIPReducer` runs ``vip_hci.psfsub.pca_annular`` (default), ``pca`` (full-frame)
or ``median_sub`` (classical ADI) on the optimizer's pre-processed cube.  VIP derotates
(CCW by ``angle_list``, same convention as this package) and collapses internally.
Parameter mapping (searched -> VIP):

==============  ==============================================================
``k_klip``       ``ncomp``  (k-scan: one call per k -- VIP's tuple ``ncomp`` means
                 per-annulus values, not a scan)
``n_ang``        ``n_segments``            (``pca_annular``; ``median_sub`` has no segments and
                 uses ``nframes = 2 k`` closest-in-angle references)
``inrad/outrad`` ``radius_int`` / ``asize = outrad - inrad`` (one annulus; ``n_annuli``
                 splits it)
``angsep``       ``delta_rot = angsep * (lambda/D) / FWHM``  (VIP counts in FWHM)
``anglemax``     not available in VIP (ignored; leave it out of the space)
``comb_type``    ``collapse`` (``'median'`` | ``'mean'``; ``'nwadi'`` -> ``'median'``)
==============  ==============================================================

Extra defaults: ``algo`` (``'pca_annular'`` | ``'pca'`` | ``'median_sub'``), ``n_annuli``,
``svd_mode``, ``imlib`` / ``interpolation`` (VIP rotation library), ``min_frames_lib``,
``max_frames_lib``, ``nproc``.  RDI/ARDI: ``Dataset.ref_cube`` is passed as ``cube_ref``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..reducer import Dataset, KLIPParams, KLIPReducer

__all__ = ["VIPReducer"]


def _require_vip():
    try:
        import vip_hci  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("VIPReducer needs vip_hci: pip install vip_hci") from exc


class VIPReducer(KLIPReducer):
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
        d = super().describe()
        d.update(backend="vip")
        return d
