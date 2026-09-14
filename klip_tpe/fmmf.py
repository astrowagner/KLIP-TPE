"""Forward-modelled matched filter (FMMF).

A matched filter is only optimal if the template is what the planet actually looks like
*in the reduced image*.  KLIP is not flux-conserving: it eats a separation- and
configuration-dependent fraction of the planet and surrounds what is left with negative
self-subtraction lobes, so filtering the residual frame with the instrumental PSF -- or
worse, a Gaussian -- throws away signal and mismatches the wings.  The forward-modelled
matched filter (Pueyo 2016; Ruffio et al. 2017) instead propagates the PSF model through
the *same* KLIP projection and filters with the result.

This module plugs that idea into the optimizer.  :class:`FMMFSNR` is a drop-in
replacement for :class:`~klip_tpe.metrics.MawetPeakSNR` that asks the runner for the
KLIP-FM pass of every configuration it evaluates and builds one matched filter **per
source**, from the local forward-modelled response.  Everything downstream -- the
small-sample-corrected ring statistics, the clean-subtraction rule, the validation
protocol -- is unchanged, so an FMMF run and a PSF-matched-filter run are directly
comparable; only the filter differs.

On JWST the PSF model to forward-model comes from :mod:`klip_tpe.stpsf_psf` (STPSF /
WebbPSF off-axis coronagraphic PSFs), which is the combination
``optimize_jwst(..., psf='stpsf', metric='fmmf')`` sets up.

Typical use::

    from klip_tpe import fmmf, stpsf_psf
    model  = stpsf_psf.model_for_datasets(dsets, star_flux=fstar)
    metric = fmmf.FMMFSNR(pxscale=px, fwhm=fw, kernel_fn=model.matched_filter_kernel)
    runner = Runner(..., objective=Objective(metric))

The forward model is unavailable in one place, the per-evaluation k-scan (KLIP-FM is not
defined for a stack of k values at once); there the metric falls back to ``kernel_fn``,
i.e. to the ordinary PSF matched filter, and says so in :meth:`FMMFSNR.describe`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import Metric, gaussian_kernel, mawet_peak_snr, radprof, source_xy, star_center

__all__ = ["fm_kernel", "fm_kernels", "FMMFSNR", "fmmf_map"]


def fm_kernel(fm_image: np.ndarray, x: float, y: float, size: int, *, zero_mean: bool = True,
              lsq_norm: bool = True, min_snr_px: float = 0.0) -> Optional[np.ndarray]:
    """The forward-modelled matched filter at pixel ``(x, y)``.

    Cuts a ``size`` x ``size`` stamp of the KLIP-FM response around the source, optionally
    removes its mean (``zero_mean``, so the filter ignores a flat pedestal) and normalises
    it by ``sum(m^2)`` (``lsq_norm``), which makes the filtered image the least-squares
    amplitude of the template -- the FMMF estimate of the source's flux in the units the
    forward model was computed in.

    Returns None when the stamp holds no usable signal (outside the KLIP zone, all-NaN,
    or a peak below ``min_snr_px`` times the stamp's own robust scatter), so the caller
    can fall back to an analytic filter.
    """
    fm = np.asarray(fm_image, float)
    ny, nx = fm.shape
    size = int(size) | 1
    h = size // 2
    ix, iy = int(round(float(x))), int(round(float(y)))
    x0, y0 = ix - h, iy - h
    st = np.zeros((size, size), float)
    xs0, ys0 = max(x0, 0), max(y0, 0)
    xs1, ys1 = min(x0 + size, nx), min(y0 + size, ny)
    if xs1 <= xs0 or ys1 <= ys0:
        return None
    sub = fm[ys0:ys1, xs0:xs1]
    st[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = np.where(np.isfinite(sub), sub, 0.0)
    pk = float(np.max(np.abs(st)))
    if not np.isfinite(pk) or pk <= 0:
        return None
    if min_snr_px > 0:
        sd = float(np.median(np.abs(st - np.median(st)))) * 1.4826
        if sd > 0 and pk < min_snr_px * sd:
            return None
    if zero_mean:
        st = st - st.mean()
    den = float((st * st).sum())
    if den <= 0:
        return None
    return st / den if lsq_norm else st / pk


def fm_kernels(fm_image: Optional[np.ndarray], rho, theta, pxscale: float, fwhm: float, *,
               size_fwhm: float = 2.5, angle_convention: str = "pa", zero_mean: bool = True,
               lsq_norm: bool = True, min_snr_px: float = 0.0,
               shape: Optional[Tuple[int, int]] = None) -> List[Optional[np.ndarray]]:
    """One :func:`fm_kernel` per source, ``None`` where the forward model gives nothing."""
    rho = np.atleast_1d(np.asarray(rho, float))
    theta = np.atleast_1d(np.asarray(theta, float))
    if fm_image is None:
        return [None] * rho.size
    fm = np.asarray(fm_image, float)
    if fm.ndim != 2 or not np.isfinite(fm).any():
        return [None] * rho.size
    if shape is not None and tuple(shape) != fm.shape:      # e.g. a k-scan cube, or a crop
        return [None] * rho.size
    size = int(2 * np.ceil(size_fwhm * float(fwhm)) + 1)
    cx, cy = star_center(fm.shape)
    xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
    return [fm_kernel(fm, x, y, size, zero_mean=zero_mean, lsq_norm=lsq_norm,
                      min_snr_px=min_snr_px) for x, y in zip(xs, ys)]


@dataclass
class FMMFSNR(Metric):
    """Mawet small-sample S/N with a **forward-modelled** matched filter per source.

    Identical to :class:`~klip_tpe.metrics.MawetPeakSNR` except for the filter: the
    kernel at each source is the KLIP-FM response of that source under the configuration
    being scored (``fm=`` supplied by the runner), normalised so the filtered image is the
    least-squares amplitude of the template.  Where no forward model is available the
    metric uses ``kernel_fn`` (the instrument PSF, e.g. the STPSF off-axis library) and
    finally a Gaussian, so a run never fails for want of an FM image.

    ``size_fwhm`` sets the filter support (default 2.5 FWHM, wider than the PSF matched
    filter because the self-subtraction lobes carry real information); ``zero_mean``
    removes a flat pedestal from the template; ``min_snr_px`` rejects an FM stamp whose
    peak is below that many times its own robust scatter (0 = accept any finite stamp).

    ``fm_from_difference`` (read by :meth:`klip_tpe.runner.Runner._fm_for`) lets a reducer
    with no analytic KLIP-FM -- pyKLIP, VIP, spaceKLIP -- supply the *numerical* forward
    model ``injected - clean`` instead, which is the same thing to first order and costs
    nothing extra when ``clean_subtract=True`` (the production default, and what this
    metric expects: without a clean reduction there is no forward model to fall back on).
    """

    pxscale: float
    fwhm: float
    kernel_fn: Optional[Callable[[float], np.ndarray]] = None
    mf_fwhm: Optional[float] = None
    size_fwhm: float = 2.5
    zero_mean: bool = True
    min_snr_px: float = 0.0
    fm_from_difference: bool = True
    search_px: float = 1.5
    excl_fwhm: float = 1.5
    known: Sequence[Tuple[float, float]] = ()
    declip_nsig: Optional[float] = 6.0
    band_fwhm: float = 0.6
    min_ring: int = 6
    penalty: str = "geometric"
    pixel_mask: Optional[np.ndarray] = None
    flatten: bool = True
    angle_convention: str = "pa"
    name: str = "fmmf"
    needs_fm: bool = True
    stats: Dict[str, int] = field(default_factory=lambda: {"fm": 0, "fallback": 0})

    def _fallback(self, rho_one: float) -> np.ndarray:
        if self.kernel_fn is not None:
            k = self.kernel_fn(float(rho_one))
            if k is not None:
                return np.asarray(k, float)
        return gaussian_kernel(self.mf_fwhm or self.fwhm)

    def kernels(self, fm, rho, theta, shape=None) -> List[np.ndarray]:
        """The per-source filters actually used (FM template where available)."""
        rho = np.atleast_1d(np.asarray(rho, float))
        ks = fm_kernels(fm, rho, theta, self.pxscale, self.fwhm, size_fwhm=self.size_fwhm,
                        angle_convention=self.angle_convention, zero_mean=self.zero_mean,
                        min_snr_px=self.min_snr_px, shape=shape)
        out: List[np.ndarray] = []
        for k, r in zip(ks, rho):
            if k is None:
                self.stats["fallback"] += 1
                out.append(self._fallback(float(r)))
            else:
                self.stats["fm"] += 1
                out.append(k)
        return out

    def per_source(self, img_inj, img_clean, rho, theta, fm=None) -> np.ndarray:
        img = radprof(img_inj) if self.flatten else np.asarray(img_inj, float)
        ks = self.kernels(fm, rho, theta, shape=np.shape(img))
        return mawet_peak_snr(img, rho, theta, self.pxscale, self.fwhm, kernel=ks,
                              search_px=self.search_px, excl_fwhm=self.excl_fwhm, known=self.known,
                              declip_nsig=self.declip_nsig, band_fwhm=self.band_fwhm,
                              min_ring=self.min_ring, penalty=self.penalty,
                              pixel_mask=self.pixel_mask, angle_convention=self.angle_convention)

    def describe(self) -> Dict:
        n = max(self.stats["fm"] + self.stats["fallback"], 1)
        return {"name": self.name, "fwhm": self.fwhm, "pxscale": self.pxscale,
                "mf_fwhm": self.mf_fwhm, "measured_kernel": self.kernel_fn is not None,
                "size_fwhm": self.size_fwhm, "zero_mean": self.zero_mean,
                "min_snr_px": self.min_snr_px, "search_px": self.search_px,
                "excl_fwhm": self.excl_fwhm, "known": [list(k) for k in self.known],
                "declip_nsig": self.declip_nsig, "band_fwhm": self.band_fwhm,
                "min_ring": self.min_ring, "penalty": self.penalty, "flatten": self.flatten,
                "fm_from_difference": self.fm_from_difference,
                "fm_fraction": round(self.stats["fm"] / n, 4),
                "n_filtered": self.stats["fm"] + self.stats["fallback"]}


# --------------------------------------------------------------------------- full map
def fmmf_map(image: np.ndarray, fm_image: np.ndarray, sources: Sequence[Any], pxscale: float,
             fwhm: float, *, contrast: float = 1.0, size_fwhm: float = 2.5,
             angle_convention: str = "pa", zero_mean: bool = True, flatten: bool = True,
             n_theta: Optional[int] = None) -> Dict[str, np.ndarray]:
    """A full-frame FMMF amplitude and S/N map from a ring of forward-modelled templates.

    ``sources`` are the FM test sources whose response ``fm_image`` holds (a golden-angle
    spiral from :func:`klip_tpe.products.fm_test_sources` works well): each supplies the
    template for the separations around its own, the templates are interpolated in radius,
    and every pixel is filtered with the template of its radius.  ``contrast`` is the
    contrast those sources were forward-modelled at, so ``amplitude`` comes out in
    contrast units.

    Returns ``{'amplitude', 'snr', 'seps_px', 'templates'}``: the LSQ amplitude map, the
    map divided by the azimuthal robust scatter at the same radius (a detection map that
    is directly comparable across separations), the template radii in pixels, and the
    templates themselves.
    """
    img = radprof(image) if flatten else np.asarray(image, float)
    img = np.where(np.isfinite(img), img, 0.0)
    fm = np.asarray(fm_image, float)
    ny, nx = img.shape
    cx, cy = star_center(img.shape)
    size = int(2 * np.ceil(size_fwhm * float(fwhm)) + 1)

    rho = np.array([s.rho for s in sources], float)
    theta = np.array([s.theta for s in sources], float)
    xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
    # one template per test source, rotated so its star-ward axis points at -x
    tpl: Dict[float, List[np.ndarray]] = {}
    from scipy import ndimage
    for r, th, x, y in zip(rho, theta, xs, ys):
        k = fm_kernel(fm, x, y, size, zero_mean=zero_mean, lsq_norm=False)
        if k is None:
            continue
        az = np.degrees(np.arctan2(y - cy, x - cx))
        tpl.setdefault(round(float(r), 4), []).append(
            ndimage.rotate(k, -az, reshape=False, order=1, mode="constant", cval=0.0))
    if not tpl:
        raise ValueError("no usable forward-modelled templates in fm_image")
    radii = np.array(sorted(tpl), float)
    stack = np.array([np.mean(tpl[round(float(r), 4)], axis=0) for r in radii])

    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy) * pxscale
    az = np.degrees(np.arctan2(yy - cy, xx - cx))
    # radial bins: each pixel is filtered with the template of the nearest modelled radius
    edges = np.concatenate([[-np.inf], (radii[1:] + radii[:-1]) / 2.0, [np.inf]])
    amp = np.full((ny, nx), np.nan)
    nth = int(n_theta or max(8, int(round(2 * np.pi * np.max(rr) / (pxscale * fwhm)))))
    azbins = np.linspace(-180.0, 180.0, nth + 1)
    for i, r in enumerate(radii):
        band = (rr >= edges[i]) & (rr < edges[i + 1])
        if not band.any():
            continue
        for j in range(nth):
            sl = band & (az >= azbins[j]) & (az < azbins[j + 1])
            if not sl.any():
                continue
            k = ndimage.rotate(stack[i], (azbins[j] + azbins[j + 1]) / 2.0, reshape=False,
                               order=1, mode="constant", cval=0.0)
            den = float((k * k).sum())
            if den <= 0:
                continue
            amp[sl] = ndimage.convolve(img, k / den, mode="nearest")[sl]
    amp = amp / float(contrast)

    snr = np.full_like(amp, np.nan)
    rpx = np.hypot(xx - cx, yy - cy)
    for r0 in range(int(np.nanmax(rpx)) + 1):
        sl = (rpx >= r0 - 0.5) & (rpx < r0 + 0.5) & np.isfinite(amp)
        if sl.sum() < 5:
            continue
        v = amp[sl]
        sd = float(np.median(np.abs(v - np.median(v)))) * 1.4826
        if sd > 0:
            snr[sl] = (v - np.median(v)) / sd
    return {"amplitude": amp, "snr": snr, "seps_px": radii / pxscale, "templates": stack}
