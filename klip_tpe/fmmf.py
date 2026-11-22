"""Forward-modeled matched filter (FMMF, Pueyo 2016, Ruffio et al. 2017).

A matched filter is optimal only when the template is what the planet looks like in the reduced
image.  KLIP removes a separation- and configuration-dependent fraction of the planet and leaves
negative self-subtraction lobes around the rest, so filtering with the instrumental PSF, or a
Gaussian, loses signal and mismatches the wings.  The FMMF propagates the PSF model through the
same KLIP projection and filters with the result.

:class:`FMMFSNR` replaces :class:`~klip_tpe.metrics.MawetPeakSNR`.  It asks the runner for the
KLIP-FM pass of every configuration it evaluates and builds one matched filter per source from
the local forward-modeled response.  The ring statistics, the clean-subtraction rule and the
validation are unchanged, so an FMMF run and a PSF-matched-filter run differ only in the filter.
On JWST the PSF model comes from :mod:`klip_tpe.stpsf_psf`, the combination
``optimize_jwst(..., psf="stpsf", metric="fmmf")`` sets up.

The forward model is not defined for the per-evaluation k-scan, where the metric falls back to
``kernel_fn``, the ordinary PSF matched filter, and says so in :meth:`FMMFSNR.describe`.

Examples
--------
::

    from klip_tpe import fmmf, stpsf_psf
    model = stpsf_psf.model_for_datasets(dsets, star_flux=fstar)
    metric = fmmf.FMMFSNR(pxscale=px, fwhm=fw, kernel_fn=model.matched_filter_kernel)
    runner = Runner(..., objective=Objective(metric))
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
    """Turn a template about its center, counter-clockwise in the image's ``(x, y)`` frame.

    The sense is that of ``arctan2(y - cy, x - cx)``.  An ADI forward model's template turned by the
    azimuth from its source to another position is the template a source there would have.

    Parameters
    ----------
    k : ndarray, shape (K, K)
        The template.
    dphi_deg : float
        Turn, in degrees, counter-clockwise.
    order : int, default 1
        Spline order of the rotation.

    Returns
    -------
    turned : ndarray, shape (K, K)
        The turned template, 0 where it rotated in from outside.

    Notes
    -----
    ``scipy.ndimage.rotate`` turns the other way for a positive angle, hence the sign in the code.
    """
    return ndimage.rotate(np.asarray(k, float), -float(dphi_deg), reshape=False, order=int(order),
                          mode="constant", cval=0.0)


def fm_kernel(fm_image: np.ndarray, x: float, y: float, size: int, *, zero_mean: bool = True,
              lsq_norm: bool = True, min_snr_px: float = 0.0) -> Optional[np.ndarray]:
    """Return the forward-modeled matched filter at a pixel.

    It cuts a stamp of the KLIP-FM response around the source (:func:`fm_stamp`), removes its mean so
    the filter ignores a flat pedestal, and normalizes it by ``sum(m^2)``, which makes the filtered
    image the least-squares amplitude of the template, the FMMF estimate of the source's flux.

    Parameters
    ----------
    fm_image : ndarray, shape (ny, nx)
        The KLIP-FM response image.
    x : float
        Column of the source, in pixels.
    y : float
        Row of the source, in pixels.
    size : int
        Width of the stamp in pixels, made odd.
    zero_mean : bool, default True
        Whether to remove the stamp's mean.
    lsq_norm : bool, default True
        Whether to normalize by the sum of squares.  False normalizes by the peak.
    min_snr_px : float, default 0.0
        Minimum peak of the stamp, in units of its own robust scatter.  0 sets no minimum.

    Returns
    -------
    kernel : ndarray or None
        The filter, or None when the stamp holds no usable signal (off the frame, outside the KLIP
        zone, all NaN, or below ``min_snr_px``), so the caller can fall back to an analytic filter.
    """
    st = fm_stamp(fm_image, x, y, size)
    if st is None:
        return None
    return template_from_stamp(st, zero_mean=zero_mean, lsq_norm=lsq_norm, min_snr_px=min_snr_px)


def fm_stamp(fm_image: np.ndarray, x: float, y: float, size: int) -> Optional[np.ndarray]:
    """Cut a stamp of an image around the pixel of a position.

    Parameters
    ----------
    fm_image : ndarray, shape (ny, nx)
        The image.
    x : float
        Column of the position, in pixels.
    y : float
        Row of the position, in pixels.
    size : int
        Width of the stamp in pixels, made odd.

    Returns
    -------
    stamp : ndarray or None
        The stamp, with NaN and whatever lies off the frame as 0, or None when it lies wholly off the
        frame.
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
    return st


def template_from_stamp(st: np.ndarray, *, zero_mean: bool = True, lsq_norm: bool = True,
                        min_snr_px: float = 0.0) -> Optional[np.ndarray]:
    """Turn a stamp into :func:`fm_kernel`'s filter.

    Parameters
    ----------
    st : ndarray
        The stamp (:func:`fm_stamp`).
    zero_mean : bool, default True
        Whether to remove the stamp's mean.
    lsq_norm : bool, default True
        Whether to normalize by the sum of squares.  False normalizes by the peak.
    min_snr_px : float, default 0.0
        Minimum peak of the stamp, in units of its own robust scatter.  0 sets no minimum.

    Returns
    -------
    kernel : ndarray or None
        The filter, or None without usable signal.
    """
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
    """Return one :func:`fm_kernel` per source.

    Parameters
    ----------
    fm_image : ndarray or None
        The KLIP-FM response image, shape (ny, nx), with the star at ``((nx-1)/2, (ny-1)/2)``.
    rho : float or array_like
        Separation of each source, in arcsec.
    theta : float or array_like
        Angle of each source, in degrees.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    size_fwhm : float, default 2.5
        Half width of the stamps, in FWHM.
    angle_convention : {"pa", "math"}, default "pa"
        How ``theta`` is read (:func:`~klip_tpe.metrics.source_xy`).
    zero_mean : bool, default True
        Whether to remove each stamp's mean.
    lsq_norm : bool, default True
        Whether to normalize by the sum of squares.
    min_snr_px : float, default 0.0
        Minimum peak of a stamp, in units of its robust scatter.
    shape : tuple of int or None, default None
        Shape the image must have, such as the science image's.  None does not check.

    Returns
    -------
    kernels : list of ndarray or None
        One filter per source, None where the forward model gives nothing, and all None for a missing,
        empty or mismatched image.
    """
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
    """Forward models of one configuration at positions of the image, each from a pass with a source alone there.

    Each template includes the source's PSF, throughput and self-subtraction at its position, where
    turning the source's own template (:func:`klip_tpe.metrics.turn_kernel`) is exact only for a PSF
    and coronagraph round about the star (NEAR2 note, 2026-11-01, asked 2).  Each position is
    forward-modeled once and its stamp kept, so the injected and the clean terms of a score share the
    passes.

    Parameters
    ----------
    make : callable
        Function ``make(sources) -> [image, ...]``, one forward-model image per model source, in order,
        from a pass of its own, None for a pass that gave nothing.  :meth:`klip_tpe.runner.Runner._fm_passes`
        yields them as they come, by KLIP-FM or, on a backend without it, as ``injected - clean``.
    shape : tuple of int
        Shape of the images, whose last two entries are ``(ny, nx)``.
    pxscale : float
        Plate scale, in arcsec per pixel.
    contrast : float, default 1.0
        Contrast of the model sources.
    angle_convention : {"pa", "math"}, default "pa"
        Angle convention of the model sources.

    Attributes
    ----------
    n_passes : int
        Number of positions asked for.
    n_empty : int
        Number of positions that came back without a stamp, outside the zone, from a failed pass or
        without a clean image to subtract.
    """

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
        """Return the model source at a pixel.

        Parameters
        ----------
        x : float
            Column of the pixel.
        y : float
            Row of the pixel.

        Returns
        -------
        source : Source
            The source, at the model contrast.
        """
        cx, cy = star_center(self.shape)
        rho = float(np.hypot(x - cx, y - cy)) * self.pxscale
        phi = float(np.degrees(np.arctan2(y - cy, x - cx)))
        th = phi - 90.0 if self.angle_convention == "pa" else phi
        return Source(rho, th % 360.0, self.contrast)

    def stamps(self, xy: Sequence[Tuple[float, float]], size: int) -> List[Optional[np.ndarray]]:
        """Return the forward model of a source alone at each position, as a stamp around its pixel.

        The positions not seen before are forward-modeled now, in one call of ``make``, and each image is
        cut as it comes, so a lazy ``make`` never holds them all.

        Parameters
        ----------
        xy : sequence of tuple of 2 floats
            The ``(x, y)`` positions, in pixels.
        size : int
            Width of the stamps in pixels, made odd (:func:`fm_stamp`).

        Returns
        -------
        stamps : list of ndarray or None
            One stamp per position, None where the pass gave nothing.

        Raises
        ------
        ValueError
            If ``make`` returns another number of images than sources.
        """
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
    """Lay out the noise ring :func:`~klip_tpe.metrics.mawet_peak_snr` uses for a source.

    The ring runs through the source's position, with ``nap = floor(2 pi r / fwhm)`` apertures
    ``2 pi / nap`` apart, the source's own first.

    Parameters
    ----------
    rho : float
        Separation of the source, in arcsec.
    theta : float
        Angle of the source, in degrees.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    shape : tuple of int
        Shape of the image, whose last two entries are ``(ny, nx)``.
    angle_convention : {"pa", "math"}, default "pa"
        How ``theta`` is read.

    Returns
    -------
    az0 : float
        Azimuth of the source, ``arctan2(y - cy, x - cx)``, in radians.
    positions : list of tuple of 2 floats
        The ``(x, y)`` aperture positions, in pixels.
    """
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
    """Wrap an angle onto (-pi, pi].

    Parameters
    ----------
    a : float
        The angle, in radians.

    Returns
    -------
    wrapped : float
        The angle, in radians.
    """
    return float(np.angle(np.exp(1j * a)))


class _ApertureTemplates:
    """Templates for one source's noise samples, the forward model of a source alone at each aperture of its ring.

    It is ``templates_at`` of :func:`~klip_tpe.metrics.mawet_peak_snr`.  Between two apertures it takes
    their two models weighted by how near in azimuth each is, zero-meaned and normalized again, which is
    wrong only to second order in the apertures' spacing.  An aperture without a template leaves its
    neighbor alone, and between two such apertures the sample takes the source's own template turned
    to it.

    Parameters
    ----------
    cx : float
        Column of the star, in pixels.
    cy : float
        Row of the star, in pixels.
    az0 : float
        Azimuth of the source, in radians.
    kernels : sequence of ndarray or None
        The least-squares filter of each aperture, the source's first, None where a pass gave none.
    azs : sequence of float or None, default None
        Azimuth of each filter, in radians.  None spaces them evenly from ``az0``.

    Attributes
    ----------
    n_turned : int
        Number of apertures without a template of their own.
    """

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
        """Return the template for a noise sample at a pixel.

        Parameters
        ----------
        x : float
            Column of the sample.
        y : float
            Row of the sample.

        Returns
        -------
        template : ndarray
            The least-squares filter for a source there.
        """
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
    """Build the per-aperture templates of each source that has a template of its own.

    For each such source it asks ``fm_at`` for the forward model of a source alone at the pixel of
    each aperture of its ring (:func:`ring_positions`), all at once, and cuts and normalizes them as
    :func:`fm_kernel` does.  The model sources sit on the pixels' centers, so each template is the filter
    for a source on the pixel it is applied at.

    Parameters
    ----------
    fm_at : FMTemplates
        The forward-model passes.
    rho : float or array_like
        Separation of each source, in arcsec.
    theta : float or array_like
        Angle of each source, in degrees.
    own : sequence of ndarray or None
        Each source's own least-squares filter, None where it has none.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    size : int
        Width of the stamps, in pixels.
    shape : tuple of int
        Shape of the image.
    zero_mean : bool, default True
        Whether to remove each stamp's mean.
    min_snr_px : float, default 0.0
        Minimum peak of a stamp, in units of its robust scatter.
    angle_convention : {"pa", "math"}, default "pa"
        How ``theta`` is read.

    Returns
    -------
    templates : list of _ApertureTemplates or None
        One per source, None for a source without a template of its own.
    """
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
    """Mawet small-sample S/N with a forward-modeled matched filter per source.

    It is :class:`~klip_tpe.metrics.MawetPeakSNR` but for the filter.  The kernel at each source is its
    KLIP-FM response under the configuration being scored (``fm=`` from the runner), normalized so the
    filtered image is the least-squares amplitude of the template.  Where no forward model is
    available it uses ``kernel_fn``, then a Gaussian, so a run never fails for want of an FM image.

    Attributes
    ----------
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    kernel_fn : callable or None, default None
        Fallback filter ``rho_arcsec -> kernel``, such as the STPSF off-axis library's.  None, or a None
        kernel, falls back to the Gaussian.
    mf_fwhm : float or None, default None
        FWHM of the Gaussian fallback, in pixels.  None takes ``fwhm``.
    size_fwhm : float, default 2.5
        Half width of the filter's support, in FWHM, wider than a PSF filter's since the
        self-subtraction lobes carry information.
    zero_mean : bool, default True
        Whether to remove a flat pedestal from each template.
    min_snr_px : float, default 0.0
        Minimum peak of an FM stamp, in units of its own robust scatter.  0 accepts any finite stamp.
    fm_from_difference : bool, default True
        Whether a reducer without analytic KLIP-FM may supply ``injected - clean`` as the forward
        model (:meth:`klip_tpe.runner.Runner._fm_for`), the same to first order and free with
        ``clean_subtract``.
    search_px : float, default 1.5
        Radius of the peak search, in pixels.
    excl_fwhm : float, default 1.5
        Radius of the keep-outs, in FWHM.
    known : sequence of tuple of 2 floats, default ()
        Known companions, as ``(rho_arcsec, theta_deg)``.
    declip_nsig : float or None, default 6.0
        Threshold of the despike in robust sigma.  None does not despike.
    declip_isolated : bool, default False
        Whether the despike spares a pixel whose neighbors stand out with it, such as a point
        source's core (:func:`~klip_tpe.metrics.declip`).  False despikes as the IDL predecessor does.
    band_fwhm : float, default 0.6
        Half width of the radial band, in FWHM.
    min_ring : int, default 6
        Minimum number of clean ring samples for the ring to give the noise.
    penalty : {"geometric", "retained"}, default "geometric"
        Count in the small-sample penalty.
    pixel_mask : ndarray of bool or None, default None
        Pixels left out of the noise samples, True to exclude.
    flatten : bool, default False
        Whether to subtract the azimuthal mean profile before scoring.
    angle_convention : {"pa", "math"}, default "pa"
        How the sources' angles are read.
    name : str, default "fmmf"
        Name of the metric, for a run's record.
    needs_fm : bool, default True
        Whether the metric asks for the KLIP-FM pass, always True here.
    stats : dict of str to int
        Counts of the filters used, ``fm`` and ``fallback``, and with ``fm_apertures`` the passes
        asked for (``fm_passes``) and those that came back empty (``fm_empty``).
    inner_fwhm : float, default 1.0
        Inner guard, in FWHM, which the runner fits to the injection floor.
    turn : bool, default True
        Whether each noise sample takes the source's template turned to its azimuth
        (:func:`klip_tpe.metrics.turn_kernel`), the template a source there would have.  A run saved
        before ``RunConfig.fmmf_turn`` keeps the one map (False).
    fm_apertures : bool, default False
        Whether each noise aperture takes the forward model of a source alone on its pixel, from the
        runner's ``fm_at`` (:class:`FMTemplates`), at the cost of about ``2 pi r / FWHM`` passes per
        source.  Without ``fm_at`` the samples are turned or read off the one map.
    correlate : bool, default True
        Whether the image is correlated with each template, as a matched filter is.  False convolves,
        as runs saved before 2026-11-04 did (``RunConfig.fmmf_correlate``).

    Notes
    -----
    The self-subtraction lobes of an ADI forward model lie along the azimuth through the source.  Read
    off the one map of the unturned template, an aperture 90 degrees away saw them along the radius,
    and against anisotropic residuals the ring's scatter was not the noise of the source's own filter
    (NEAR2 note, 2026-11-01).  Turning is exact where the noise is round about the star, and each
    aperture's own model is exact where the noise changes as the PSF does, such as a PSF and a noise
    pattern both fixed on the detector (``tests/test_fmmf_apertures.py``).
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
    declip_isolated: bool = False
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
    #: True correlates the image with each template, as a matched filter does, and False convolves (see above)
    correlate: bool = True

    def _fallback(self, rho_one: float) -> np.ndarray:
        """Return the fallback filter at a separation, ``kernel_fn``'s or the Gaussian.

        Parameters
        ----------
        rho_one : float
            Separation, in arcsec.

        Returns
        -------
        kernel : ndarray
            The filter.
        """
        if self.kernel_fn is not None:
            k = self.kernel_fn(float(rho_one))
            if k is not None:
                return np.asarray(k, float)
        return gaussian_kernel(self.mf_fwhm or self.fwhm)

    def kernels(self, fm, rho, theta, shape=None) -> List[np.ndarray]:
        """Return the filter used at each source, the FM template where available.

        Each call counts the templates and the fallbacks in ``stats``.

        Parameters
        ----------
        fm : ndarray or None
            The KLIP-FM response image.
        rho : float or array_like
            Separation of each source, in arcsec.
        theta : float or array_like
            Angle of each source, in degrees.
        shape : tuple of int or None, default None
            Shape the FM image must have.  None does not check.

        Returns
        -------
        kernels : list of ndarray
            One filter per source.
        """
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
        """Return each source's S/N, with forward-modeled filters.

        Parameters
        ----------
        img_inj : ndarray, shape (ny, nx)
            The image scored.
        img_clean : ndarray or None
            The clean reduction, unused here.
        rho : sequence of float
            Separation of each source, in arcsec.
        theta : sequence of float
            Angle of each source, in degrees.
        fm : ndarray or None, default None
            The KLIP-FM response image of the configuration.
        why : list or None, default None
            List filled with each source's reason for a NaN.
        fm_at : FMTemplates or None, default None
            Forward-model passes at the noise apertures, used with ``fm_apertures``.

        Returns
        -------
        snr : ndarray
            S/N of each source.
        """
        img = radprof(img_inj) if self.flatten else np.asarray(img_inj, float)
        ks = self.kernels(fm, rho, theta, shape=np.shape(img))
        # the forward-modeled templates are turned to each noise sample's azimuth, or each
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
                                  declip_nsig=self.declip_nsig,
                                  declip_isolated=bool(getattr(self, "declip_isolated", False)),
                                  band_fwhm=self.band_fwhm, min_ring=self.min_ring, penalty=self.penalty,
                                  pixel_mask=self.pixel_mask, inner_fwhm=self.inner_fwhm,
                                  angle_convention=self.angle_convention, return_details=True, turn=turn,
                                  templates_at=tat, correlate=bool(getattr(self, "correlate", True)))
        if why is not None:
            why[:] = [d.get("why") for d in det]
        return snr

    def describe(self) -> Dict:
        """Return the metric's settings and its filter counts, for a run's record.

        Returns
        -------
        info : dict
            The settings, ``fm_fraction`` (the share of filters that were FM templates), ``n_filtered`` and
            the aperture passes.
        """
        n = max(self.stats["fm"] + self.stats["fallback"], 1)
        return {"name": self.name, "fwhm": self.fwhm, "pxscale": self.pxscale,
                "mf_fwhm": self.mf_fwhm, "measured_kernel": self.kernel_fn is not None,
                "size_fwhm": self.size_fwhm, "zero_mean": self.zero_mean,
                "min_snr_px": self.min_snr_px, "search_px": self.search_px,
                "excl_fwhm": self.excl_fwhm, "known": [list(k) for k in self.known],
                "declip_nsig": self.declip_nsig,
                **({"declip_isolated": True} if getattr(self, "declip_isolated", False) else {}),
                "band_fwhm": self.band_fwhm,
                "min_ring": self.min_ring, "penalty": self.penalty, "flatten": self.flatten,
                "inner_fwhm": self.inner_fwhm, "fm_from_difference": self.fm_from_difference,
                "turn": bool(self.turn), "fm_apertures": bool(getattr(self, "fm_apertures", False)),
                "correlate": bool(getattr(self, "correlate", True)),
                "fm_aperture_passes": int(self.stats.get("fm_passes", 0)),
                "fm_aperture_empty": int(self.stats.get("fm_empty", 0)),
                "fm_fraction": round(self.stats["fm"] / n, 4),
                "n_filtered": self.stats["fm"] + self.stats["fallback"]}


# --------------------------------------------------------------------------- full map
def fmmf_map(image: np.ndarray, fm_image: np.ndarray, sources: Sequence[Any], pxscale: float,
             fwhm: float, *, contrast: float = 1.0, size_fwhm: float = 2.5,
             angle_convention: str = "pa", zero_mean: bool = True, flatten: bool = False,
             n_theta: Optional[int] = None, fm_at: Optional[FMTemplates] = None,
             correlate: bool = True) -> Dict[str, np.ndarray]:
    """Make a full-frame FMMF amplitude and S/N map from a ring of forward-modeled templates.

    Each FM test source supplies the template for the separations nearest its own, turned so its
    star-ward axis points at -x, and every pixel is filtered with the template of its radius turned to
    its bin of azimuth.

    Parameters
    ----------
    image : ndarray, shape (ny, nx)
        The reduced image.
    fm_image : ndarray, shape (ny, nx)
        The KLIP-FM response of the test sources.
    sources : sequence of Source
        The FM test sources, such as the golden-angle spiral of
        :func:`klip_tpe.products.fm_test_sources`.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    contrast : float, default 1.0
        Contrast the test sources were forward-modeled at, so ``amplitude`` comes out in contrast units.
    size_fwhm : float, default 2.5
        Half width of the templates, in FWHM.
    angle_convention : {"pa", "math"}, default "pa"
        How the sources' angles are read.
    zero_mean : bool, default True
        Whether to remove each template's mean.
    flatten : bool, default False
        Whether to subtract the image's azimuthal mean profile first.
    n_theta : int or None, default None
        Number of azimuth bins.  None takes the circumference at the largest radius in FWHM, at least 8.
    fm_at : FMTemplates or None, default None
        This configuration's forward-model passes.  Each bin of azimuth at each template radius then
        takes the forward model of a source alone on the pixel nearest its center, and a bin whose
        pass gave nothing keeps the turned template.
    correlate : bool, default True
        Whether each pixel is the image correlated with its template.  False convolves, as before
        2026-11-04 (GitHub agent's finding 13).

    Returns
    -------
    maps : dict
        ``amplitude``, the least-squares amplitude map, ``snr``, the map over its azimuthal robust
        scatter at each radius, ``seps_px``, the template radii in pixels, and ``templates``.  With
        ``fm_at``, ``n_fm_bins`` counts the bins that took their own model.

    Raises
    ------
    ValueError
        If ``fm_image`` holds no usable template.

    Notes
    -----
    Until 2026-11-01 both turns went the other way, which mirrored every template about its own
    azimuth, and a planet 30 to 60 degrees from its test source kept about 0.7 of its amplitude.
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
    # radial bins: each pixel is filtered with the template of the nearest modeled radius
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
            amp[sl] = (ndimage.correlate if correlate else ndimage.convolve)(img, k / den, mode="nearest")[sl]
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
