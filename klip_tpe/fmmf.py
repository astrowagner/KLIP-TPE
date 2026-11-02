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
from scipy import ndimage

from .metrics import (Metric, Source, gaussian_kernel, mawet_peak_snr, radprof, source_xy, star_center,
                      turn_kernel)

__all__ = ["fm_kernel", "fm_kernels", "fm_stamp", "template_from_stamp", "FMMFSNR", "fmmf_map", "turn_template",
           "FMTemplates", "ring_positions", "aperture_templates"]


def turn_template(k: np.ndarray, dphi_deg: float, order: int = 1) -> np.ndarray:
    """``k`` turned by ``dphi_deg`` degrees about its center, counterclockwise in the image's
    ``(x, y) = (column, row)`` frame, the sense of ``arctan2(y - cy, x - cx)``.  An ADI forward
    model's template turned by the azimuth from its source to another position is the template
    a source there would have.  ``scipy.ndimage.rotate`` turns the other way for a positive
    angle, hence the sign."""
    return ndimage.rotate(np.asarray(k, float), -float(dphi_deg), reshape=False, order=int(order),
                          mode="constant", cval=0.0)


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
    st = fm_stamp(fm_image, x, y, size)
    if st is None:
        return None
    return template_from_stamp(st, zero_mean=zero_mean, lsq_norm=lsq_norm, min_snr_px=min_snr_px)


def fm_stamp(fm_image: np.ndarray, x: float, y: float, size: int) -> Optional[np.ndarray]:
    """The ``size`` x ``size`` stamp (``size`` made odd) of ``fm_image`` around the pixel of
    ``(x, y)``, NaN and whatever lies off the frame as 0, or None when it lies off the frame."""
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
    return st


def template_from_stamp(st: np.ndarray, *, zero_mean: bool = True, lsq_norm: bool = True,
                        min_snr_px: float = 0.0) -> Optional[np.ndarray]:
    """:func:`fm_kernel`'s filter from a stamp :func:`fm_stamp` cut: None without usable signal,
    else zero-meaned (``zero_mean``) and normalized by its sum of squares (``lsq_norm``) or by
    its peak."""
    st = np.asarray(st, float)
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


# --------------------------------------------------------------------------- a forward model at each aperture
class FMTemplates:
    """The forward models of one configuration at positions of the image, each from a
    forward-model pass with a source alone there: the template a source at that position has,
    its PSF, its throughput and its self-subtraction included, where turning the source's own
    template (:func:`klip_tpe.metrics.turn_kernel`) is exact only for a PSF and a coronagraph
    round about the star (NEAR2 note, 2026-11-01, asked 2).

    ``make(sources) -> [image, ...]``: one forward-model image per model source, in order, each
    from a pass of its own (:meth:`klip_tpe.runner.Runner._fm_passes` yields them as they come,
    made in parallel, by KLIP-FM or, on a backend without it, as ``injected - clean``), None
    for a pass that gave nothing.  The model sources are at ``contrast``, in ``angle_convention``.  Each position is
    forward-modelled once and its stamp kept, so the injected and the clean terms of a score
    share the passes.  ``n_passes`` counts the positions asked for, ``n_empty`` those that came
    back with no stamp (outside the zone, a pass that failed, or no clean image to subtract)."""

    def __init__(self, make: Callable[[List[Source]], Sequence[Optional[np.ndarray]]], shape: Tuple[int, ...],
                 pxscale: float, *, contrast: float = 1.0, angle_convention: str = "pa"):
        self.make = make
        self.shape = (int(shape[-2]), int(shape[-1]))
        self.pxscale = float(pxscale)
        self.contrast = float(contrast)
        self.angle_convention = angle_convention
        self._stamps: Dict[Tuple[float, float, int], Optional[np.ndarray]] = {}
        self.n_passes = 0
        self.n_empty = 0

    def source_at(self, x: float, y: float) -> Source:
        """The model source at pixel ``(x, y)``."""
        cx, cy = star_center(self.shape)
        rho = float(np.hypot(x - cx, y - cy)) * self.pxscale
        phi = float(np.degrees(np.arctan2(y - cy, x - cx)))
        th = phi - 90.0 if self.angle_convention == "pa" else phi
        return Source(rho, th % 360.0, self.contrast)

    def stamps(self, xy: Sequence[Tuple[float, float]], size: int) -> List[Optional[np.ndarray]]:
        """The forward model of a source alone at each of ``xy`` (px), as the ``size`` x ``size``
        stamp :func:`fm_stamp` cuts around its pixel, None where its pass gave nothing.  The
        positions not seen before are forward-modelled now, in one call of ``make``."""
        size = int(size) | 1
        keys = [(round(float(x), 3), round(float(y), 3), size) for x, y in xy]
        at: Dict[Tuple[float, float, int], Tuple[float, float]] = {}
        for k, (x, y) in zip(keys, xy):
            at.setdefault(k, (float(x), float(y)))
        todo = [k for k in at if k not in self._stamps]
        if todo:
            # each image is cut as it comes, so a lazy make never holds them all
            got = 0
            for k, im in zip(todo, self.make([self.source_at(*at[k]) for k in todo])):
                st = None
                if im is not None and np.ndim(im) == 2 and tuple(np.shape(im)) == self.shape:
                    st = fm_stamp(im, at[k][0], at[k][1], k[2])
                    if st is not None and not np.any(st):
                        st = None
                self.n_passes += 1
                self.n_empty += st is None
                self._stamps[k] = st
                got += 1
            if got != len(todo):
                raise ValueError(f"FMTemplates: {got} forward models for {len(todo)} sources")
        return [self._stamps[k] for k in keys]


def ring_positions(rho: float, theta: float, pxscale: float, fwhm: float, shape,
                   angle_convention: str = "pa") -> Tuple[float, List[Tuple[float, float]]]:
    """The noise ring :func:`~klip_tpe.metrics.mawet_peak_snr` lays out for a source at
    ``(rho, theta)``, through the position given: ``nap = floor(2 pi r / fwhm)`` apertures
    ``2 pi / nap`` apart, the source's own first.  Returns ``(az0, [(x, y), ...])``, ``az0`` the
    source's azimuth in radians (``arctan2(y - cy, x - cx)``) and the positions in pixels."""
    cx, cy = star_center(shape)
    xs, ys = source_xy([rho], [theta], pxscale, cx, cy, angle_convention)
    r = float(np.hypot(xs[0] - cx, ys[0] - cy))
    az0 = float(np.arctan2(ys[0] - cy, xs[0] - cx))
    nap = int(np.floor(2 * np.pi * r / float(fwhm)))
    if nap < 1:
        return az0, [(float(xs[0]), float(ys[0]))]
    a = az0 + np.arange(nap) * 2 * np.pi / nap
    return az0, [(float(cx + r * np.cos(t)), float(cy + r * np.sin(t))) for t in a]


def _wrap(a: float) -> float:
    return float(np.angle(np.exp(1j * a)))


class _ApertureTemplates:
    """``(x, y) -> template`` for one source's noise samples (``templates_at`` of
    :func:`~klip_tpe.metrics.mawet_peak_snr`): at an aperture of the source's ring, the forward
    model of a source alone there.  Between two apertures, their two models weighted by how near
    in azimuth each is, then zero-meaned and normalized again.  Turning a model by the azimuth
    between them would turn a PSF that is not round with it, and keeping one would keep the self
    subtraction's lobes along another azimuth, each wrong to first order in the apertures'
    spacing.  The weighted pair is wrong to second order either way.  An aperture whose pass gave
    no template leaves its neighbor alone, and between two such apertures the sample takes the
    source's own template turned to it, as ``turn`` does (``n_turned`` counts them).  ``azs``:
    each template's azimuth, the source's first, then each aperture's pixel's."""

    def __init__(self, cx: float, cy: float, az0: float, kernels: Sequence[Optional[np.ndarray]],
                 azs: Optional[Sequence[float]] = None):
        self.cx, self.cy, self.az0 = float(cx), float(cy), float(az0)
        self.k = [None if k is None else np.asarray(k, float) for k in kernels]
        self.n = len(self.k)
        self.step = 2 * np.pi / self.n
        self.azs = ([self.az0 + q * self.step for q in range(self.n)] if azs is None
                    else [float(a) for a in azs])
        # each template's shape, from its least-squares filter k = m / sum(m^2)
        self.m = [None if k is None else k / float((k * k).sum()) for k in self.k]
        self.n_turned = sum(k is None for k in self.k[1:])
        self._made: Dict[Tuple[int, ...], np.ndarray] = {}

    def __call__(self, x: float, y: float) -> np.ndarray:
        phi = float(np.arctan2(float(y) - self.cy, float(x) - self.cx))
        u = _wrap(phi - self.az0) / self.step
        qa = int(np.floor(u)) % self.n
        qb = (qa + 1) % self.n
        span = _wrap(self.azs[qb] - self.azs[qa]) if self.n > 1 else 0.0
        w = float(np.clip(_wrap(phi - self.azs[qa]) / span, 0.0, 1.0)) if span > 0 else 0.0
        ka, kb = self.k[qa], self.k[qb]
        if ka is None and kb is None:
            key: Tuple[int, ...] = (-1, int(round(_wrap(phi - self.az0) * 1e6)))
            t = self._made.get(key)
            if t is None:
                t = self._made[key] = turn_kernel(self.k[0], key[1] * 1e-6)
            return t
        if kb is None or (ka is not None and w <= 0.0):
            return ka
        if ka is None or w >= 1.0:
            return kb
        if ka.shape != kb.shape:
            return ka if w < 0.5 else kb
        key = (qa, qb, int(round(w * 1e4)))
        t = self._made.get(key)
        if t is None:
            wq = key[2] * 1e-4
            m = (1.0 - wq) * self.m[qa] + wq * self.m[qb]
            if all(abs(float(k.sum())) <= 1e-9 * float(np.abs(k).sum()) for k in (ka, kb)):
                m = m - m.mean()
            d = float((m * m).sum())
            t = self._made[key] = m / d if d > 0 else ka
        return t


def aperture_templates(fm_at: FMTemplates, rho, theta, own: Sequence[Optional[np.ndarray]], pxscale: float,
                       fwhm: float, size: int, shape, *, zero_mean: bool = True, min_snr_px: float = 0.0,
                       angle_convention: str = "pa") -> List[Optional[_ApertureTemplates]]:
    """For each source with a template of its own (``own[s]``, its least-squares filter), the
    forward model of a source alone at the pixel of each aperture of its ring
    (:func:`ring_positions`), from ``fm_at``'s passes, all asked for at once, and cut and
    normalized as :func:`fm_kernel` cuts the source's.  A source without a template of its own
    takes None.  The model sources sit on the pixels' centers, so each template is the filter for
    a source on the pixel it is applied at, wherever on the pixel the ring crossed it."""
    rho = np.atleast_1d(np.asarray(rho, float))
    theta = np.atleast_1d(np.asarray(theta, float))
    cx, cy = star_center(shape)
    rings: List[Optional[Tuple[float, List[Tuple[float, float]]]]] = []
    allxy: List[Tuple[float, float]] = []
    for s in range(rho.size):
        if own[s] is None:
            rings.append(None)
            continue
        az0, pos = ring_positions(float(rho[s]), float(theta[s]), pxscale, fwhm, shape, angle_convention)
        pix = [(float(round(x)), float(round(y))) for x, y in pos[1:]]
        rings.append((az0, pix))
        allxy += pix
    stamps = fm_at.stamps(allxy, size) if allxy else []
    out: List[Optional[_ApertureTemplates]] = []
    i = 0
    for s, ring in enumerate(rings):
        if ring is None:
            out.append(None)
            continue
        ks: List[Optional[np.ndarray]] = [np.asarray(own[s], float)]
        for _ in ring[1]:
            st = stamps[i]
            i += 1
            ks.append(None if st is None else template_from_stamp(st, zero_mean=zero_mean, lsq_norm=True,
                                                                  min_snr_px=min_snr_px))
        azs = [ring[0]] + [float(np.arctan2(y - cy, x - cx)) for x, y in ring[1]]
        out.append(_ApertureTemplates(cx, cy, ring[0], ks, azs))
    return out


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

    ``turn`` (default True): each noise aperture of the ring and each pixel of the radial band
    is filtered with the source's template turned by its azimuth from the source
    (:func:`klip_tpe.metrics.turn_kernel`), the template a source there would have.  The
    self-subtraction lobes of an ADI forward model lie along the azimuth through the source,
    so read off the one map of the unturned template, an aperture 90 deg away saw them along
    the radius, and against anisotropic residuals the ring's scatter was not the noise of the
    source's own filter (NEAR2 note, 2026-11-01).  ``RunConfig.fmmf_turn`` sets it for a run,
    and a run saved before keeps the one map (``False``).

    ``fm_apertures`` (default False): each noise aperture takes the forward model of a source
    alone on its pixel, made by a forward-model pass of its own (:class:`FMTemplates`, handed
    over as ``fm_at=`` by the runner), and a sample between two apertures, as each bin of the
    radial band, their two models weighted by azimuth.  Each sample is then filtered as a source
    there would be, where the PSF or the coronagraph is not round about the star too, and the
    ring's scatter is that of the forward-modelled amplitude map at the source's separation
    (NEAR2 note, 2026-11-01, asked 2).  For the noise of the source's own filter, turning stays
    exact where the noise is round about the star, and each aperture's own model is exact where
    the noise changes as the PSF does, a PSF and a noise pattern both fixed on the detector, say
    (``tests/test_fmmf_apertures.py``).  It costs a pass per aperture, about ``2 pi r / FWHM``
    per source.  Without ``fm_at`` the samples are turned (``turn``) or read off the one map.
    ``RunConfig.fmmf_fm_apertures`` sets it for a run.  ``stats['fm_passes']`` counts the
    passes, ``stats['fm_empty']`` those that gave no template.

    ``fm_from_difference`` (read by :meth:`klip_tpe.runner.Runner._fm_for`) lets a reducer
    with no analytic KLIP-FM -- this package's pyKLIP, VIP and spaceKLIP backends -- supply
    the *numerical* forward model ``injected - clean`` instead, which is the same thing to
    first order and costs nothing extra when ``clean_subtract=True`` (the production
    default, and what this metric expects: without a clean reduction there is no forward
    model to fall back on).
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
    flatten: bool = False
    angle_convention: str = "pa"
    name: str = "fmmf"
    needs_fm: bool = True
    stats: Dict[str, int] = field(default_factory=lambda: {"fm": 0, "fallback": 0})
    #: no S/N for a source placed inside this many FWHM of the star (as
    #: :class:`~klip_tpe.metrics.MawetPeakSNR`, which the Runner fits to the injection floor)
    inner_fwhm: float = 1.0
    #: the noise samples take the source's template turned to their azimuth (see above)
    turn: bool = True
    #: each noise aperture takes a forward model of its own, from the runner's ``fm_at`` (see above)
    fm_apertures: bool = False

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

    def per_source(self, img_inj, img_clean, rho, theta, fm=None, why: Optional[list] = None,
                   fm_at: Optional[FMTemplates] = None) -> np.ndarray:
        img = radprof(img_inj) if self.flatten else np.asarray(img_inj, float)
        ks = self.kernels(fm, rho, theta, shape=np.shape(img))
        # the forward-modelled templates are turned to each noise sample's azimuth, or each
        # sample takes the forward model of a source there (fm_apertures).  The fallbacks, round
        # PSF or Gaussian filters, are neither
        turn, tat = None, None
        at = bool(getattr(self, "fm_apertures", False)) and fm_at is not None
        if self.turn or at:
            fmk = fm_kernels(fm, rho, theta, self.pxscale, self.fwhm, size_fwhm=self.size_fwhm,
                             angle_convention=self.angle_convention, zero_mean=self.zero_mean,
                             min_snr_px=self.min_snr_px, shape=np.shape(img))
            if len(fmk) == len(ks):
                if self.turn:
                    turn = [k is not None for k in fmk]
                if at:
                    n0, e0 = fm_at.n_passes, fm_at.n_empty
                    own = [k if f is not None else None for k, f in zip(ks, fmk)]
                    size = int(2 * np.ceil(self.size_fwhm * float(self.fwhm)) + 1)
                    tat = aperture_templates(fm_at, rho, theta, own, self.pxscale, self.fwhm, size, np.shape(img),
                                             zero_mean=self.zero_mean, min_snr_px=self.min_snr_px,
                                             angle_convention=self.angle_convention)
                    self.stats["fm_passes"] = self.stats.get("fm_passes", 0) + fm_at.n_passes - n0
                    self.stats["fm_empty"] = self.stats.get("fm_empty", 0) + fm_at.n_empty - e0
        snr, det = mawet_peak_snr(img, rho, theta, self.pxscale, self.fwhm, kernel=ks,
                                  search_px=self.search_px, excl_fwhm=self.excl_fwhm, known=self.known,
                                  declip_nsig=self.declip_nsig, band_fwhm=self.band_fwhm,
                                  min_ring=self.min_ring, penalty=self.penalty,
                                  pixel_mask=self.pixel_mask, inner_fwhm=self.inner_fwhm,
                                  angle_convention=self.angle_convention, return_details=True, turn=turn,
                                  templates_at=tat)
        if why is not None:
            why[:] = [d.get("why") for d in det]
        return snr

    def describe(self) -> Dict:
        n = max(self.stats["fm"] + self.stats["fallback"], 1)
        return {"name": self.name, "fwhm": self.fwhm, "pxscale": self.pxscale,
                "mf_fwhm": self.mf_fwhm, "measured_kernel": self.kernel_fn is not None,
                "size_fwhm": self.size_fwhm, "zero_mean": self.zero_mean,
                "min_snr_px": self.min_snr_px, "search_px": self.search_px,
                "excl_fwhm": self.excl_fwhm, "known": [list(k) for k in self.known],
                "declip_nsig": self.declip_nsig, "band_fwhm": self.band_fwhm,
                "min_ring": self.min_ring, "penalty": self.penalty, "flatten": self.flatten,
                "inner_fwhm": self.inner_fwhm, "fm_from_difference": self.fm_from_difference,
                "turn": bool(self.turn), "fm_apertures": bool(getattr(self, "fm_apertures", False)),
                "fm_aperture_passes": int(self.stats.get("fm_passes", 0)),
                "fm_aperture_empty": int(self.stats.get("fm_empty", 0)),
                "fm_fraction": round(self.stats["fm"] / n, 4),
                "n_filtered": self.stats["fm"] + self.stats["fallback"]}


# --------------------------------------------------------------------------- full map
def fmmf_map(image: np.ndarray, fm_image: np.ndarray, sources: Sequence[Any], pxscale: float,
             fwhm: float, *, contrast: float = 1.0, size_fwhm: float = 2.5,
             angle_convention: str = "pa", zero_mean: bool = True, flatten: bool = False,
             n_theta: Optional[int] = None, fm_at: Optional[FMTemplates] = None) -> Dict[str, np.ndarray]:
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

    ``fm_at`` (:class:`FMTemplates`, this configuration's forward-model passes): each bin of
    azimuth at each template radius takes the forward model of a source alone on the pixel
    nearest its center in place of the radius' template turned there, a pass per bin and radius.
    A bin whose pass gave nothing keeps the turned template.  ``n_fm_bins``, returned with
    ``fm_at``, counts the bins that took their own.
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
    # One template per test source, turned so its star-ward axis points at -x.  Until
    # 2026-11-01 both turns of this function went the other way (scipy's rotate turns
    # clockwise in this frame), which mirrored every template but at its own azimuth: a planet
    # 30 to 60 deg from the test source kept about 0.7 of its amplitude.
    tpl: Dict[float, List[np.ndarray]] = {}
    for r, th, x, y in zip(rho, theta, xs, ys):
        k = fm_kernel(fm, x, y, size, zero_mean=zero_mean, lsq_norm=False)
        if k is None:
            continue
        az = np.degrees(np.arctan2(y - cy, x - cx))
        tpl.setdefault(round(float(r), 4), []).append(turn_template(k, -az))
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
    # each pixel's azimuth bin, the last one closed (a pixel at +180 deg fell in none before)
    jbin = np.clip(np.floor((az + 180.0) / (360.0 / nth)).astype(int), 0, nth - 1)
    azc = (azbins[:-1] + azbins[1:]) / 2.0
    own: Dict[Tuple[int, int], np.ndarray] = {}
    if fm_at is not None:
        # a forward model of its own at the center of every bin of every radius that has pixels
        cells = [(i, j) for i in range(radii.size) for j in range(nth)
                 if ((rr >= edges[i]) & (rr < edges[i + 1]) & (jbin == j)).any()]
        # each on the center of the pixel nearest the bin's, the filter for a source on a pixel
        xy = [(float(round(cx + radii[i] / pxscale * np.cos(np.radians(azc[j])))),
               float(round(cy + radii[i] / pxscale * np.sin(np.radians(azc[j]))))) for i, j in cells]
        for c, st in zip(cells, fm_at.stamps(xy, size)):
            t = None if st is None else template_from_stamp(st, zero_mean=zero_mean, lsq_norm=False)
            if t is not None:
                own[c] = t
    for i, r in enumerate(radii):
        band = (rr >= edges[i]) & (rr < edges[i + 1])
        if not band.any():
            continue
        for j in range(nth):
            sl = band & (jbin == j)
            if not sl.any():
                continue
            k = own.get((i, j))
            if k is None:
                k = turn_template(stack[i], azc[j])
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
    out = {"amplitude": amp, "snr": snr, "seps_px": radii / pxscale, "templates": stack}
    if fm_at is not None:
        out["n_fm_bins"] = len(own)
    return out
