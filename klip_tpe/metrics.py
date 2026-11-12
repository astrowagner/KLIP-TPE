"""Detection metrics and the search objective.

Images are 2-D arrays indexed ``img[y, x]``, with the star at ``cx = (nx-1)/2, cy = (ny-1)/2``
for frames of ny by nx pixels, between pixels for even sizes.  A source sits at ``(rho, theta)``,
``rho`` in arcsec and ``theta`` a position angle East of North, at pixel
``x = cx + r cos(theta + 90), y = cy + r sin(theta + 90)`` for ``r = rho / pxscale``, North up
and East left.  A reducer with another convention declares it
(:attr:`klip_tpe.reducer.Reducer.angle_convention`) and the runner converts.

Notes
-----
These routines succeed those of ``optimize_near_2_tpe.pro``, the IDL predecessor: ``radprof``,
``near2m_declip``, ``near2m_mfkern``, ``near2_snrpk`` (:func:`mawet_peak_snr`), ``near2_snr_inj``
(:func:`injection_difference_snr`), ``near2m_clip0`` and its clean-subtraction rule
(:class:`Objective`).  Its instrument variants (MWC, LMIRCam, HPBoo) differ from NEAR only by
option flags of :class:`MawetPeakSNR`.
"""
from __future__ import annotations

import collections
import functools
import hashlib
import inspect
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

__all__ = ["radprof", "declip", "gaussian_kernel", "source_xy", "mawet_peak_snr",
           "injection_difference_snr", "aperture_mawet_snr", "clip0", "MawetPeakSNR",
           "InjectionDifferenceSNR", "Objective", "Source", "nanmedian_even", "pa_wedge_mask", "line_mask",
           "disk_mask_from_image", "why_failed", "WHY_GUARD", "WHY_RING", "WHY_NOISE", "Sidelobes",
           "sidelobe_snr", "RingRecorder", "ring_role", "ring_rules", "rule_kwargs", "matched_filter",
           "VERIFY_IDL", "verify_idl"]


# ----------------------------------------------------------------------------
# a record of the noise rings, for diagrams (klip_tpe.ringplot, scripts/plot_noise_rings.py)
# ----------------------------------------------------------------------------
#: the active :class:`RingRecorder`, or None (the default: nothing is recorded)
_RING_RECORDER: Optional["RingRecorder"] = None
#: per thread: the image a ring is measured on (:func:`ring_role`) and the rings
#: :func:`sidelobe_snr` holds back until it knows which attempt it kept
_RING_LOCAL = threading.local()


class RingRecorder:
    """Recorder of the noise rings the S/N measurements use, for diagrams.

    While active (``with RingRecorder() as rec:``), every noise ring an S/N is measured on is kept in
    ``rec.records``, one dict per source.  A record holds the ring's apertures with what became of
    each (``"used"``, ``"source"`` within an injected source's keep-out, ``"known"`` a known
    companion's, ``"arc"`` in a sidelobe arc, ``"masked"`` on the pixel mask, ``"off"`` off the data)
    and whether the ring or the radial band gave the noise.  Recording changes no S/N.

    Parameters
    ----------
    phase : callable or None, default None
        Function that names the current phase, asked when a record is made
        (:func:`klip_tpe.ringplot.phase_from_stack`).  None records no phase.
    limit : int, default 20000
        Maximum number of records kept.

    Attributes
    ----------
    records : list of dict
        The records, in the order they were made.

    Notes
    -----
    :func:`mawet_peak_snr`, :func:`aperture_mawet_snr`, :func:`sidelobe_snr` (the attempt it kept),
    :func:`klip_tpe.verify.snr_fap`, :func:`klip_tpe.verify.stim_fap` and
    :func:`klip_tpe.products.snr_map` record.
    """

    def __init__(self, phase: Optional[Callable[[], Optional[str]]] = None, limit: int = 20000):
        self.records: List[Dict[str, Any]] = []
        self.phase = phase
        self.limit = int(limit)
        self._lock = threading.Lock()
        self._prev: Optional["RingRecorder"] = None

    def add(self, rec: Dict[str, Any]) -> None:
        """Add a record, labeled with the phase and the image's role.

        Parameters
        ----------
        rec : dict
            The record, given ``phase`` from :attr:`phase` and ``role`` from :class:`ring_role` when it
            has neither.  It is dropped once :attr:`limit` records are kept.
        """
        if "phase" not in rec and self.phase is not None:
            try:
                rec["phase"] = self.phase()
            except Exception:                 # noqa: BLE001 - a label, never a failure
                rec["phase"] = None
        rec.setdefault("role", getattr(_RING_LOCAL, "role", None))
        with self._lock:
            if len(self.records) < self.limit:
                self.records.append(rec)

    def __enter__(self) -> "RingRecorder":
        """Make this recorder the active one.

        Returns
        -------
        rec : RingRecorder
            This recorder.
        """
        global _RING_RECORDER
        self._prev, _RING_RECORDER = _RING_RECORDER, self
        return self

    def __exit__(self, *exc) -> bool:
        """Restore the recorder that was active before.

        Parameters
        ----------
        exc : tuple
            Type, value and traceback of an exception raised inside, ignored.

        Returns
        -------
        suppress : bool
            False, so an exception propagates.
        """
        global _RING_RECORDER
        _RING_RECORDER = self._prev
        return False


def _ring_emit(rec: Dict[str, Any]) -> None:
    """Send a ring record to the innermost :func:`sidelobe_snr` holding rings, or else to the active recorder.

    Parameters
    ----------
    rec : dict
        The record, as :func:`_ring_rec` builds it.
    """
    held = getattr(_RING_LOCAL, "held", None)
    if held:
        held[-1].append(rec)
        return
    r = _RING_RECORDER
    if r is not None:
        r.add(rec)


class ring_role:
    """Context manager that marks the rings measured inside it as measured on one image.

    ``with ring_role("clean"):`` marks them as on the objective's clean image.  Without it a ring's
    ``role`` is None, the injected image.

    Parameters
    ----------
    role : str or None
        Name of the image, such as ``"clean"``.
    """

    def __init__(self, role: Optional[str]):
        self.role = role
        self._prev = None

    def __enter__(self):
        """Set the role for the rings measured inside.

        Returns
        -------
        ctx : ring_role
            This context manager.
        """
        self._prev = getattr(_RING_LOCAL, "role", None)
        _RING_LOCAL.role = self.role
        return self

    def __exit__(self, *exc) -> bool:
        """Restore the role that was set before.

        Parameters
        ----------
        exc : tuple
            Type, value and traceback of an exception raised inside, ignored.

        Returns
        -------
        suppress : bool
            False, so an exception propagates.
        """
        _RING_LOCAL.role = self._prev
        return False


def line_mask(shape: Tuple[int, int], lines: Sequence[Tuple[float, float]], angle_convention: str = "pa",
              r_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Mask straight strips through the star, for ``pixel_mask``.

    A strip holds the pixels within ``half_width_px`` of the line through the star at position angle
    ``pa_deg``, on both sides of the star.  It suits a feature fixed on the sky along one direction
    and about one beam wide, such as the line through alpha Cen A along the A-B direction in the
    NEAR data, which a wedge (:func:`pa_wedge_mask`) masks too widely far out and too narrowly close
    in.  Choose the strips before looking at the image, as for :func:`pa_wedge_mask`, and keep the
    injections off them too (:class:`~klip_tpe.positions.PositionSampler`'s ``forbidden_pa``).

    Parameters
    ----------
    shape : tuple of int
        Shape of the image, ``(ny, nx)``.
    lines : sequence of tuple of 2 floats
        The strips, as ``(pa_deg, half_width_px)``, the angle in degrees and the half width in
        pixels.
    angle_convention : {"pa", "math"}, default "pa"
        "pa" reads ``pa_deg`` East of North, and "math" counter-clockwise from +x.
    r_range : tuple of 2 floats or None, default None
        The ``(inner, outer)`` radii in pixels the strips are limited to.  None leaves them unlimited.

    Returns
    -------
    mask : ndarray of bool, shape (ny, nx)
        True for each pixel to exclude.
    """
    ny, nx = int(shape[0]), int(shape[1])
    cx, cy = star_center((ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx, dy = xx - cx, yy - cy
    out = np.zeros((ny, nx), bool)
    for pa, hw in lines:
        phi = np.deg2rad(float(pa) + (90.0 if angle_convention == "pa" else 0.0))   # as source_xy
        out |= np.abs(dx * np.sin(phi) - dy * np.cos(phi)) <= float(hw)
    if r_range is not None:
        rr = np.hypot(dx, dy)
        out &= (rr >= float(r_range[0])) & (rr <= float(r_range[1]))
    return out


def pa_wedge_mask(shape: Tuple[int, int], bands: Sequence[Tuple[float, float]],
                  angle_convention: str = "pa",
                  r_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Mask azimuthal wedges, for ``pixel_mask``.

    It suits resolved emission known beforehand to sit at a given position angle, such as a debris
    disk seen close to edge-on, so its flux stays out of the noise ring and the radial-band sigma.
    Pair it with :class:`~klip_tpe.positions.PositionSampler`'s ``forbidden_pa`` on the same bands,
    so the injections avoid what the noise estimate no longer measures.

    Parameters
    ----------
    shape : tuple of int
        Shape of the image, ``(ny, nx)``.
    bands : sequence of tuple of 2 floats
        The wedges, as ``(center_pa_deg, half_width_deg)``, in degrees.
    angle_convention : {"pa", "math"}, default "pa"
        "pa" reads the angles East of North, and "math" counter-clockwise from +x.
    r_range : tuple of 2 floats or None, default None
        The ``(inner, outer)`` radii in pixels the wedges are limited to, for a disk of known
        extent.  None leaves them unlimited.

    Returns
    -------
    mask : ndarray of bool, shape (ny, nx)
        True for each pixel to exclude.

    Notes
    -----
    Removing ring apertures because they are bright biases the noise estimate down and inflates
    every S/N, so the exclusion has to be justified before the image is looked at.  A disk at a
    catalogued position angle qualifies, the brightest speckle group does not.  On beta Pic the two
    are comparable in the annulus.  The disk bands carry about +0.2 sigma of median excess over the
    rest of the ring, while removing about 22% of the apertures.
    """
    ny, nx = int(shape[0]), int(shape[1])
    cx, cy = star_center((ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx, dy = xx - cx, yy - cy
    phi = np.degrees(np.arctan2(dy, dx))
    pa = phi - (90.0 if angle_convention == "pa" else 0.0)
    out = np.zeros((ny, nx), bool)
    for c, hw in bands:
        out |= np.abs((pa - float(c) + 180.0) % 360.0 - 180.0) <= float(hw)
    if r_range is not None:
        rr = np.hypot(dx, dy)
        out &= (rr >= float(r_range[0])) & (rr <= float(r_range[1]))
    return out


def disk_mask_from_image(img: np.ndarray, fwhm_px: float, *, nsig: float = 2.0,
                         r_range: Optional[Tuple[float, float]] = None, min_beams: float = 2.0,
                         grow_fwhm: float = 1.0, smooth_fwhm: float = 1.0,
                         protect: Sequence[Tuple[float, float]] = (), protect_fwhm: float = 1.5,
                         pxscale: Optional[float] = None, angle_convention: str = "pa",
                         nsig_neg: Optional[float] = None, neg_reach_fwhm: float = 2.0) -> np.ndarray:
    """Mask the bright extended structure of an image, such as a disk that no wedge or strip follows.

    The mask serves the metric's ``pixel_mask`` and the sampler's ``forbidden_mask`` together
    (:func:`klip_tpe.instruments.generic.default_config`'s ``disk_mask``).  The image is smoothed,
    referred to its median and robust sigma (1.4826 times the MAD) over ``r_range`` and cut at
    ``nsig``.  Pieces under ``min_beams`` beams are dropped and the rest grown by ``grow_fwhm``, since
    the metric samples the matched-filtered image, which carries the disk about a FWHM beyond its
    edge.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Image of the disk, ny by nx pixels, with the star at ``((nx-1)/2, (ny-1)/2)``.
    fwhm_px : float
        FWHM of the PSF, in pixels.
    nsig : float, default 2.0
        Threshold in robust sigma above the median.
    r_range : tuple of 2 floats or None, default None
        The ``(inner, outer)`` radii in pixels the statistics and the mask are limited to.  None
        takes all finite pixels.
    min_beams : float, default 2.0
        Minimum size of a piece, in beams of ``pi (fwhm_px/2)^2`` pixels.  An isolated speckle is
        not a disk.
    grow_fwhm : float, default 1.0
        Growth of the mask, in FWHM.
    smooth_fwhm : float, default 1.0
        FWHM of the NaN-aware Gaussian smoothing, in units of ``fwhm_px``.  0 does not smooth.
    protect : sequence of tuple of 2 floats, default ()
        Positions cleared from the mask, as ``(rho_arcsec, pa_deg)``, so a companion on the disk
        stays measurable.  Its own ``known`` exclusion keeps it out of the noise anyway.
    protect_fwhm : float, default 1.5
        Radius cleared around each protected position, in FWHM.
    pxscale : float or None, default None
        Plate scale in arcsec per pixel, needed with ``protect``.
    angle_convention : {"pa", "math"}, default "pa"
        How the protected positions' angles are read (:func:`source_xy`).
    nsig_neg : float or None, default None
        Threshold in robust sigma below the median for the disk's ADI negatives, the troughs
        self-subtraction digs beside extended emission.  Troughs in pieces of at least ``min_beams``
        beams within ``neg_reach_fwhm`` of the bright structure are kept, and None leaves them out.
    neg_reach_fwhm : float, default 2.0
        Maximum distance of a kept trough from the bright structure, in FWHM.

    Returns
    -------
    mask : ndarray of bool, shape (ny, nx)
        True for each pixel to exclude.  All False when fewer than 10 pixels are usable or the
        sigma is 0.

    Raises
    ------
    ValueError
        If ``protect`` is given without ``pxscale``.

    Notes
    -----
    The caution of :func:`pa_wedge_mask` applies with more force.  This masks whatever is bright,
    and a bright speckle group masked with the disk biases the noise estimate down.  Build it from
    an image made before the search, such as a disk-preserving reduction or another epoch, never
    from a search's own winner, and look at it.
    """
    img = np.asarray(img, float)
    ny, nx = img.shape
    cx, cy = star_center(img.shape)
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy)
    fin = np.isfinite(img)
    fwhm_px = float(fwhm_px)
    sg = max(float(smooth_fwhm), 0.0) * fwhm_px / 2.3548
    if sg > 0:
        num = ndimage.gaussian_filter(np.where(fin, img, 0.0), sg)
        den = ndimage.gaussian_filter(fin.astype(float), sg)
        sm = np.full(img.shape, np.nan)
        good = fin & (den > 0.2)
        sm[good] = num[good] / den[good]
    else:
        sm = np.where(fin, img, np.nan)
    inr = np.ones(img.shape, bool) if r_range is None else (rr >= float(r_range[0])) & (rr <= float(r_range[1]))
    ok = inr & np.isfinite(sm)
    out = np.zeros(img.shape, bool)
    if ok.sum() < 10:
        return out
    med = float(np.median(sm[ok]))
    sig = 1.4826 * float(np.median(np.abs(sm[ok] - med)))
    if not sig > 0:
        return out
    beam = np.pi * (fwhm_px / 2.0) ** 2

    def big(m: np.ndarray) -> np.ndarray:
        """``m`` without its pieces under ``min_beams`` beams"""
        if not m.any() or min_beams <= 0:
            return m
        lab, n = ndimage.label(m)
        sizes = np.asarray(ndimage.sum(m, lab, index=np.arange(1, n + 1)), float)
        small = 1 + np.flatnonzero(sizes < float(min_beams) * beam)
        return m & ~np.isin(lab, small) if small.size else m

    out = big(ok & (sm - med > float(nsig) * sig))
    if nsig_neg is not None and out.any():
        neg = big(ok & (sm - med < -float(nsig_neg) * sig))
        if neg.any():
            near = ndimage.distance_transform_edt(~out) <= float(neg_reach_fwhm) * fwhm_px
            lab, _ = ndimage.label(neg)
            touch = np.unique(lab[neg & near])
            touch = touch[touch > 0]
            if touch.size:
                out = out | np.isin(lab, touch)
    if out.any() and grow_fwhm > 0:
        out = ndimage.distance_transform_edt(~out) <= float(grow_fwhm) * fwhm_px
    out &= inr
    if len(protect):
        if pxscale is None:
            raise ValueError("protect needs pxscale")
        px, py = source_xy([p[0] for p in protect], [p[1] for p in protect], float(pxscale), cx, cy, angle_convention)
        for x0, y0 in zip(px, py):
            out &= np.hypot(xx - x0, yy - y0) > float(protect_fwhm) * fwhm_px
    return out


@dataclass(frozen=True)
class Source:
    """A point source, by separation, position angle and contrast.

    Attributes
    ----------
    rho : float
        Separation from the star, in arcsec.
    theta : float
        Position angle East of North, in degrees.
    contrast : float, default 0.0
        Contrast to the star, in the injection model's flux unit.
    """
    rho: float        # arcsec
    theta: float      # deg, PA east of north (metric convention)
    contrast: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return the source as ``(rho, theta, contrast)``.

        Returns
        -------
        values : tuple of 3 floats
            Separation in arcsec, position angle in degrees and contrast.
        """
        return (float(self.rho), float(self.theta), float(self.contrast))


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def star_center(shape) -> Tuple[float, float]:
    """Return the star's pixel, ``((nx-1)/2, (ny-1)/2)``, for an image's shape.

    Parameters
    ----------
    shape : tuple of int
        Shape of the image or cube, whose last two entries are ``(ny, nx)``.

    Returns
    -------
    cx : float
        Column of the star, in pixels.
    cy : float
        Row of the star, in pixels.
    """
    ny, nx = shape[-2], shape[-1]
    return (nx - 1) / 2.0, (ny - 1) / 2.0


def source_xy(rho_as, theta_deg, pxscale: float, cx: float, cy: float, angle_convention: str = "pa"):
    """Convert source positions to pixel coordinates.

    Parameters
    ----------
    rho_as : float or array_like
        Separations, in arcsec.
    theta_deg : float or array_like
        Angles, in degrees.
    pxscale : float
        Plate scale, in arcsec per pixel.
    cx : float
        Column of the star, in pixels.
    cy : float
        Row of the star, in pixels.
    angle_convention : {"pa", "math"}, default "pa"
        "pa" reads ``theta_deg`` East of North, ``x = cx + r cos(theta + 90)``, and "math"
        counter-clockwise from +x.

    Returns
    -------
    x : ndarray
        Column of each source, in pixels.
    y : ndarray
        Row of each source, in pixels.
    """
    rho = np.atleast_1d(np.asarray(rho_as, float))
    th = np.atleast_1d(np.asarray(theta_deg, float))
    phi = np.deg2rad(th + (90.0 if angle_convention == "pa" else 0.0))
    return cx + (rho / pxscale) * np.cos(phi), cy + (rho / pxscale) * np.sin(phi)


def nanmedian_even(v) -> float:
    """Return the median of the finite values, the mean of the two middle ones for an even count.

    Parameters
    ----------
    v : array_like
        The values.

    Returns
    -------
    median : float
        The median, or NaN when no value is finite.

    Notes
    -----
    It matches IDL's ``median(/even)``, which the IDL predecessor uses.
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def radprof(image: np.ndarray) -> np.ndarray:
    """Subtract an image's azimuthal mean profile, in bins of integer radius.

    Parameters
    ----------
    image : ndarray, shape (ny, nx)
        The image.  NaN pixels are left out of the means and stay NaN.

    Returns
    -------
    flat : ndarray, shape (ny, nx)
        The image minus its profile, in float64.

    Notes
    -----
    The bins are centered on ``(nx/2, ny/2)``, half a pixel from the star, as in the IDL
    predecessor's ``radprof``.  Only the flattening is affected, never the source positions.
    """
    img = np.asarray(image, float)
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ri = np.floor(np.hypot(xx - nx / 2.0, yy - ny / 2.0)).astype(int)
    nmu = ri.max() + 1
    mu = np.full(nmu, np.nan)
    flat = img.ravel()
    rf = ri.ravel()
    fin = np.isfinite(flat)
    cnt = np.bincount(rf[fin], minlength=nmu)
    sm = np.bincount(rf[fin], weights=flat[fin], minlength=nmu)
    ok = cnt > 0
    mu[ok] = sm[ok] / cnt[ok]
    return img - mu[ri]


def declip(img: np.ndarray, nsig: float = 6.0) -> np.ndarray:
    """Replace isolated single-pixel spikes by their 3x3 median.

    A spike is a finite pixel with a finite 3x3 neighborhood that lies more than ``nsig`` robust
    sigma (1.4826 times the median absolute residual) from its 3x3 median.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.  An array of another number of dimensions comes back unchanged.
    nsig : float, default 6.0
        Threshold in robust sigma.

    Returns
    -------
    out : ndarray
        The despiked image, in float64.  The input comes back unchanged when fewer than 20 pixels
        qualify or the sigma is 0.

    Notes
    -----
    It does what ``near2m_declip`` does in the IDL predecessor.
    """
    img = np.asarray(img, float)
    if img.ndim != 2:
        return img
    fin = np.isfinite(img)
    imz = np.where(fin, img, 0.0)
    m = ndimage.median_filter(imz, size=3, mode="nearest")
    nfin = ndimage.convolve(fin.astype(float), np.ones((3, 3)), mode="nearest")
    res = imz - m
    g = fin & (nfin >= 8.5)
    if g.sum() < 20:
        return img
    rob = float(np.median(np.abs(res[g]))) * 1.4826
    if rob <= 0:
        return img
    out = img.copy()
    bad = g & (np.abs(res) > nsig * rob)
    out[bad] = m[bad]
    return out


def gaussian_kernel(fwhm: float, size: Optional[int] = None) -> np.ndarray:
    """Return a unit-sum Gaussian matched-filter kernel.

    Parameters
    ----------
    fwhm : float
        FWHM of the Gaussian, in pixels.
    size : int or None, default None
        Width of the kernel, in pixels.  None takes ``2 * ceil(1.5 * fwhm) + 1``.

    Returns
    -------
    kernel : ndarray, shape (size, size)
        The kernel, summing to 1.

    Notes
    -----
    It is the default branch of ``near2m_mfkern`` in the IDL predecessor.
    """
    kd = int(2 * np.ceil(1.5 * fwhm) + 1) if size is None else int(size)
    kc = (kd - 1) / 2.0
    sg = fwhm / 2.3548
    yy, xx = np.mgrid[0:kd, 0:kd]
    k = np.exp(-0.5 * ((xx - kc) ** 2 + (yy - kc) ** 2) / sg ** 2)
    return k / k.sum()


def kernel_from_profile(stamp: np.ndarray, fwhm: float) -> np.ndarray:
    """Build a unit-sum matched-filter kernel from a PSF stamp's azimuthal profile.

    The profile is the mean in rings 1 px wide about the stamp's middle, interpolated linearly onto
    a kernel ``2 * ceil(1.5 * fwhm) + 1`` pixels wide and clipped at 0.

    Parameters
    ----------
    stamp : ndarray, shape (sy, sx)
        PSF stamp, centered on its middle pixel ``((sx-1)/2, (sy-1)/2)``.
    fwhm : float
        FWHM of the PSF, in pixels, which sets the kernel's width.

    Returns
    -------
    kernel : ndarray
        The kernel, summing to 1, or :func:`gaussian_kernel` when the profile sums to 0 or less.

    Notes
    -----
    It is the measured matched filter of ``near2m_mfkern`` in the IDL predecessor, used there when
    the injection library is active.
    """
    st = np.asarray(stamp, float)
    kd = int(2 * np.ceil(1.5 * fwhm) + 1)
    kc = (kd - 1) / 2.0
    scy, scx = (st.shape[0] - 1) / 2.0, (st.shape[1] - 1) / 2.0
    yy, xx = np.mgrid[0:st.shape[0], 0:st.shape[1]]
    rr = np.hypot(xx - scx, yy - scy)
    nrp = int(np.ceil(kc)) + 2
    prof = np.zeros(nrp)
    for ir in range(nrp):
        w = (rr >= ir - 0.5) & (rr < ir + 0.5)
        if w.any():
            prof[ir] = np.nanmean(st[w])
    ky, kx = np.mgrid[0:kd, 0:kd]
    d = np.hypot(kx - kc, ky - kc)
    i0 = np.minimum(np.floor(d).astype(int), nrp - 2)
    fr = d - i0
    k = (1 - fr) * prof[i0] + fr * prof[i0 + 1]
    k = np.clip(k, 0, None)
    if k.sum() <= 0:
        return gaussian_kernel(fwhm)
    return k / k.sum()


# ----------------------------------------------------------------------------
# one declip and one matched filter per image and kernel
# ----------------------------------------------------------------------------
#: images :func:`mawet_peak_snr` has prepared (declipped, zero-filled and filtered), kept by
#: their bytes.  A trial's clean image is scored once for each of its draws, and
#: :func:`sidelobe_snr` measures each source, and each fallback, on the same image again, so
#: the same declip and the same filtering came back several times an evaluation.  Keyed by
#: the bytes, so an image changed in place is another image.  0 turns the cache off.
PREPARED_CACHE_SIZE = 8
_prepared_cache: "collections.OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = collections.OrderedDict()
_prepared_lock = threading.Lock()


def _digest(a: np.ndarray) -> Tuple[Any, ...]:
    """Return a key for an array's contents, its shape, dtype and a 128-bit hash of its bytes.

    Parameters
    ----------
    a : ndarray
        The array.

    Returns
    -------
    key : tuple
        ``(shape, dtype string, digest)``.
    """
    a = np.ascontiguousarray(a)
    return (a.shape, a.dtype.str, hashlib.blake2b(a.data, digest_size=16).digest())


def _frozen(*arrays: np.ndarray) -> None:
    """Make arrays read-only.

    Parameters
    ----------
    arrays : ndarray
        The arrays, changed in place.
    """
    for a in arrays:
        a.flags.writeable = False


def _prepare(img: np.ndarray, declip_nsig: Optional[float]) -> Dict[str, Any]:
    """Prepare an image for the matched filter, from the cache when these bytes were prepared before.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, in float.
    declip_nsig : float or None
        Threshold of the despike in robust sigma (:func:`declip`).  None does not despike.

    Returns
    -------
    entry : dict
        ``img`` the despiked image, ``imz`` it zero-filled, ``finite`` its finite pixels and
        ``maps`` its correlations so far, by kernel.  Cached arrays are read-only copies, never the
        caller's.
    """
    def build() -> Dict[str, Any]:
        d = declip(img, declip_nsig) if declip_nsig is not None else img
        if d is img:
            d = img.copy()
        return {"img": d, "imz": np.where(np.isfinite(d), d, 0.0), "finite": np.isfinite(d), "maps": {}}
    if PREPARED_CACHE_SIZE <= 0:
        return build()
    key = _digest(img) + (None if declip_nsig is None else float(declip_nsig),)
    with _prepared_lock:
        hit = _prepared_cache.get(key)
        if hit is not None:
            _prepared_cache.move_to_end(key)
            return hit
    entry = build()
    _frozen(entry["img"], entry["imz"], entry["finite"])
    with _prepared_lock:
        _prepared_cache[key] = entry
        while len(_prepared_cache) > PREPARED_CACHE_SIZE:
            _prepared_cache.popitem(last=False)
    return entry


def clear_prepared_cache() -> None:
    """Forget every prepared image (:data:`PREPARED_CACHE_SIZE`)."""
    with _prepared_lock:
        _prepared_cache.clear()


@functools.lru_cache(maxsize=16)
def _grids(ny: int, nx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the pixel indices and the distances from the star for an image shape, read-only.

    Parameters
    ----------
    ny : int
        Number of rows.
    nx : int
        Number of columns.

    Returns
    -------
    yy : ndarray, shape (ny, nx)
        Row of each pixel.
    xx : ndarray, shape (ny, nx)
        Column of each pixel.
    rr : ndarray, shape (ny, nx)
        Distance of each pixel from the star (:func:`star_center`), in pixels.
    """
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = star_center((ny, nx))
    rr = np.hypot(xx - cx, yy - cy)
    _frozen(yy, xx, rr)
    return yy, xx, rr


def _per_source_kernels(kernel, nsrc: int, fwhm: float) -> List[np.ndarray]:
    """Expand :func:`mawet_peak_snr`'s ``kernel`` argument to one kernel per source.

    Sources with the same kernel share the same object, so its correlation is done once.

    Parameters
    ----------
    kernel : ndarray, sequence or None
        One 2-D kernel, a sequence of one per source or of one, or None.
    nsrc : int
        Number of sources.
    fwhm : float
        FWHM of the Gaussian that None, empty or all-NaN entries take, in pixels.

    Returns
    -------
    kernels : list of ndarray
        One kernel per source.

    Raises
    ------
    ValueError
        If the sequence has neither one nor ``nsrc`` entries.
    """
    if kernel is None:
        return [gaussian_kernel(fwhm)] * nsrc
    if isinstance(kernel, np.ndarray) and kernel.ndim == 2:
        return [kernel] * nsrc
    ks = list(kernel)
    if len(ks) == 1:
        ks = ks * nsrc
    if len(ks) != nsrc:
        raise ValueError(f"got {len(ks)} kernels for {nsrc} sources")
    fallback = None
    out = []
    for k in ks:
        if k is None or np.size(k) == 0 or not np.isfinite(np.asarray(k, float)).any():
            if fallback is None:
                fallback = gaussian_kernel(fwhm)
            out.append(fallback)
        else:
            out.append(np.asarray(k, float))
    return out


def clip0(x):
    """Clamp finite negative values to 0, leaving NaN as it is.

    Parameters
    ----------
    x : array_like
        The values.

    Returns
    -------
    clipped : ndarray
        A float64 copy with the negatives set to 0.

    Notes
    -----
    It does what ``near2m_clip0`` does in the IDL predecessor.
    """
    x = np.asarray(x, float).copy()
    b = np.isfinite(x) & (x < 0)
    x[b] = 0.0
    return x


def _aperture_sum(img: np.ndarray, x: float, y: float, r: float) -> float:
    """Sum the finite pixels whose centers lie within a radius of a position.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.
    x : float
        Column of the aperture's center, in pixels.
    y : float
        Row of the aperture's center, in pixels.
    r : float
        Radius of the aperture, in pixels.

    Returns
    -------
    total : float
        The sum, or NaN when the aperture holds no finite pixel.  Pixels outside the image are
        ignored.
    """
    ny, nx = img.shape
    x0, x1 = int(max(np.floor(x - r), 0)), int(min(np.ceil(x + r), nx - 1))
    y0, y1 = int(max(np.floor(y - r), 0)), int(min(np.ceil(y + r), ny - 1))
    if x1 < x0 or y1 < y0:
        return np.nan
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    vals = img[y0:y1 + 1, x0:x1 + 1][m]
    vals = vals[np.isfinite(vals)]
    return float(vals.sum()) if vals.size else np.nan


def _arc_regions(arcs, pxscale: float, fwhm: float, cx: float, cy: float, angle_convention: str = "pa"):
    """Convert ``excl_arcs`` entries to the image's polar coordinates.

    Parameters
    ----------
    arcs : sequence of tuple of 4 floats or None
        Arcs as ``(rho_arcsec, theta_deg, half_width_deg, radial_fwhm)``.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    cx : float
        Column of the star, in pixels.
    cy : float
        Row of the star, in pixels.
    angle_convention : {"pa", "math"}, default "pa"
        How the arcs' angles are read (:func:`source_xy`).

    Returns
    -------
    regions : list of tuple of 4 floats
        Each arc as ``(r_px, phi_rad, half_width_rad, radial_px)``.
    """
    out = []
    for a in arcs or ():
        x, y = source_xy([float(a[0])], [float(a[1])], pxscale, cx, cy, angle_convention)
        out.append((float(np.hypot(x[0] - cx, y[0] - cy)), float(np.arctan2(y[0] - cy, x[0] - cx)),
                    float(np.deg2rad(a[2])), float(a[3]) * float(fwhm)))
    return out


def _in_arcs(x, y, regions, cx: float, cy: float):
    """Tell which positions lie in any arc region.

    A position is in an arc when it lies within ``radial_px`` of the arc's radius and within
    ``half_width`` of its azimuth.

    Parameters
    ----------
    x : float or ndarray
        Columns, in pixels.
    y : float or ndarray
        Rows, in pixels.
    regions : sequence of tuple of 4 floats
        The arcs, as :func:`_arc_regions` gives them.
    cx : float
        Column of the star, in pixels.
    cy : float
        Row of the star, in pixels.

    Returns
    -------
    inside : ndarray of bool
        True for each position in an arc, shaped as ``x``.
    """
    r = np.hypot(np.asarray(x, float) - cx, np.asarray(y, float) - cy)
    ph = np.arctan2(np.asarray(y, float) - cy, np.asarray(x, float) - cx)
    m = np.zeros(np.shape(r), bool)
    for rc, pc, hw, dr in regions:
        m |= (np.abs(r - rc) <= dr) & (np.abs((ph - pc + np.pi) % (2 * np.pi) - np.pi) <= hw)
    return m


# ----------------------------------------------------------------------------
# per-source detection S/N  (near2_snrpk)
# ----------------------------------------------------------------------------
#: Why :func:`mawet_peak_snr` leaves a source's S/N NaN (``details[s]['why']``).  The
#: objective puts them in a failed draw's record and log line.
WHY_GUARD = "inside the inner guard"
WHY_RING = "fewer than 5 apertures on its ring"
WHY_NOISE = "no noise samples on its ring or band"


def turn_kernel(k: np.ndarray, dphi: float, *, order: int = 3) -> np.ndarray:
    """Turn a least-squares template about its center and normalize it again as it was.

    The template ``m = k / sum(k^2)`` is turned counter-clockwise in the image's ``(x, y)`` frame,
    the sense of ``arctan2(y - cy, x - cx)``.  It comes back zero-mean when ``k`` was, as
    ``m' / sum(m'^2)`` for the turned template ``m'``.  An ADI forward model's template, turned by
    the azimuth from its source to another position on the ring, is the template a source there
    would have (NEAR2 note, 2026-11-01).

    Parameters
    ----------
    k : ndarray, shape (K, K)
        The least-squares filter, K pixels wide.
    dphi : float
        Turn, in radians, counter-clockwise.
    order : int, default 3
        Spline order of the rotation.

    Returns
    -------
    turned : ndarray, shape (K, K)
        The turned filter, or ``k`` itself when ``dphi`` is 0 or not finite or ``k`` is all 0.

    Notes
    -----
    ``scipy.ndimage.rotate`` turns the other way for a positive angle, hence the sign in the code.
    """
    k = np.asarray(k, float)
    s2 = float((k * k).sum())
    if s2 <= 0 or not np.isfinite(dphi) or dphi == 0.0:
        return k
    m = k / s2                                    # the template, from its least-squares filter
    t = ndimage.rotate(m, -np.degrees(float(dphi)), reshape=False, order=int(order), mode="constant", cval=0.0)
    if abs(float(k.sum())) <= 1e-9 * float(np.abs(k).sum()):
        t = t - t.mean()
    d = float((t * t).sum())
    return t / d if d > 0 else k


class _TurnedFilter:
    """Matched-filter values of one source's template turned to other positions on the image.

    Each value is what ``ndimage.correlate(imz, turned, mode="nearest")`` gives at that pixel, or
    with ``correlate`` False what ``ndimage.convolve`` gives, from one small dot product instead of
    a whole map.  Turned templates are kept by angle, to 1e-6 rad, since the ring and the band ask
    the same angles of several pixels.

    Parameters
    ----------
    imz : ndarray, shape (ny, nx)
        The zero-filled image.
    kernel : ndarray, shape (K, K)
        The source's template, K pixels wide with K odd.
    az : float
        Azimuth of the source from +x, in radians.
    pads : dict of int to ndarray
        Copies of ``imz`` padded at their edges, by pad width, shared between filters and filled
        as needed.
    correlate : bool, default True
        Whether to correlate, or else convolve, as :func:`mawet_peak_snr` does.
    """

    def __init__(self, imz: np.ndarray, kernel: np.ndarray, az: float, pads: Dict[int, np.ndarray],
                 correlate: bool = True):
        self.k = np.asarray(kernel, float)
        self.c = self.k.shape[0] // 2
        if self.c not in pads:                    # the frame padded as mode="nearest" extends it
            pads[self.c] = np.pad(imz, self.c, mode="edge")
        self.pad = pads[self.c]
        self.az = float(az)
        self.correlate = bool(correlate)
        self._flip: Dict[int, np.ndarray] = {}

    def _flipped(self, dphi: float) -> np.ndarray:
        """Return the template turned by ``dphi`` as the dot product takes it, flipped 180 deg to convolve.

        Parameters
        ----------
        dphi : float
            Turn, in radians, counter-clockwise.

        Returns
        -------
        template : ndarray, shape (K, K)
            The turned template.
        """
        key = int(round(float(dphi) * 1e6))
        if key not in self._flip:
            t = turn_kernel(self.k, key * 1e-6)
            self._flip[key] = t if self.correlate else t[::-1, ::-1]
        return self._flip[key]

    def at(self, x: int, y: int, dphi: float) -> float:
        """Return the filtered value at a pixel, with the template turned by ``dphi``.

        Parameters
        ----------
        x : int
            Column of the pixel.
        y : int
            Row of the pixel.
        dphi : float
            Turn of the template, in radians.

        Returns
        -------
        value : float
            The matched-filter value.
        """
        K = self.k.shape[0]
        return float((self.pad[y:y + K, x:x + K] * self._flipped(dphi)).sum())

    def band(self, x: np.ndarray, y: np.ndarray, cx: float, cy: float, width: float) -> np.ndarray:
        """Return the values at pixels, each with the template turned to its bin of azimuth.

        The bins are those :func:`klip_tpe.fmmf.fmmf_map` filters by.

        Parameters
        ----------
        x : ndarray of int
            Columns of the pixels.
        y : ndarray of int
            Rows of the pixels.
        cx : float
            Column of the star, in pixels.
        cy : float
            Row of the star, in pixels.
        width : float
            Width of a bin of azimuth, in radians.

        Returns
        -------
        values : ndarray
            The matched-filter value at each pixel.
        """
        x, y = np.asarray(x, int), np.asarray(y, int)
        dphi = np.angle(np.exp(1j * (np.arctan2(y - cy, x - cx) - self.az)))
        b = np.round(dphi / width).astype(int)
        out = np.empty(x.size, float)
        K = self.k.shape[0]
        win = np.lib.stride_tricks.sliding_window_view(self.pad, (K, K))
        for bi in np.unique(b):
            sel = np.where(b == bi)[0]
            fk = self._flipped(bi * width)
            out[sel] = np.einsum("nij,ij->n", win[y[sel], x[sel]], fk)
        return out


class _PlacedFilter:
    """Matched-filter values at pixels of the image, each with the template of a source there.

    Each value is what ``ndimage.correlate(imz, k, mode="nearest")`` gives at that pixel with the
    template ``k = kernel_at(x, y)``, or with ``correlate`` False what ``ndimage.convolve`` gives,
    from one small dot product per sample.  It gives the forward model of a source at each noise
    aperture (:class:`klip_tpe.fmmf.FMTemplates`).

    Parameters
    ----------
    imz : ndarray, shape (ny, nx)
        The zero-filled image.
    kernel_at : callable
        Function ``(x, y) -> template or None`` that gives the template of a source at a pixel,
        square and of an odd size.
    pads : dict of int to ndarray
        Copies of ``imz`` padded at their edges, by pad width, shared and filled as needed.
    correlate : bool, default True
        Whether to correlate, or else convolve.
    """

    def __init__(self, imz: np.ndarray, kernel_at: Callable[[float, float], Optional[np.ndarray]],
                 pads: Dict[int, np.ndarray], correlate: bool = True):
        self.imz, self.kernel_at, self.pads = imz, kernel_at, pads
        self.correlate = bool(correlate)
        self._flip: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _flipped(self, k: np.ndarray) -> np.ndarray:
        """Return a template as the dot product takes it, flipped 180 deg to convolve.

        Parameters
        ----------
        k : ndarray
            The template.

        Returns
        -------
        template : ndarray
            The template as used, kept by the template's identity.
        """
        e = self._flip.get(id(k))
        if e is None or e[0] is not k:            # the kernel is kept, so its id stays its own
            kk = np.asarray(k, float)
            e = self._flip[id(k)] = (k, kk if self.correlate else kk[::-1, ::-1])
        return e[1]

    def _pad(self, c: int) -> np.ndarray:
        """Return the image padded by ``c`` pixels at its edges, as ``mode="nearest"`` extends it.

        Parameters
        ----------
        c : int
            Pad width, in pixels.

        Returns
        -------
        padded : ndarray
            The padded image.
        """
        if c not in self.pads:                    # the frame padded as mode="nearest" extends it
            self.pads[c] = np.pad(self.imz, c, mode="edge")
        return self.pads[c]

    def at(self, x: int, y: int) -> Optional[float]:
        """Return the filtered value at a pixel, with the template of a source there.

        Parameters
        ----------
        x : int
            Column of the pixel.
        y : int
            Row of the pixel.

        Returns
        -------
        value : float or None
            The matched-filter value, or None when ``kernel_at`` gives no template there.
        """
        k = self.kernel_at(x, y)
        if k is None:
            return None
        K = k.shape[0]
        return float((self._pad(K // 2)[y:y + K, x:x + K] * self._flipped(k)).sum())

    def band(self, x: np.ndarray, y: np.ndarray, cx: float, cy: float, r: float, az: float,
             width: float) -> np.ndarray:
        """Return the values at pixels, in bins of azimuth each filtered with the template at its center.

        Parameters
        ----------
        x : ndarray of int
            Columns of the pixels.
        y : ndarray of int
            Rows of the pixels.
        cx : float
            Column of the star, in pixels.
        cy : float
            Row of the star, in pixels.
        r : float
            Radius of the circle the bins' centers lie on, in pixels.
        az : float
            Azimuth the bins are counted from, in radians.
        width : float
            Width of a bin, in radians.

        Returns
        -------
        values : ndarray
            The matched-filter value at each pixel, NaN in a bin ``kernel_at`` gives no template for.
        """
        x, y = np.asarray(x, int), np.asarray(y, int)
        dphi = np.angle(np.exp(1j * (np.arctan2(y - cy, x - cx) - az)))
        b = np.round(dphi / width).astype(int)
        out = np.full(x.size, np.nan)
        for bi in np.unique(b):
            a = az + bi * width
            k = self.kernel_at(cx + r * np.cos(a), cy + r * np.sin(a))
            if k is None:
                continue
            sel = np.where(b == bi)[0]
            K = k.shape[0]
            win = np.lib.stride_tricks.sliding_window_view(self._pad(K // 2), (K, K))
            out[sel] = np.einsum("nij,ij->n", win[y[sel], x[sel]], self._flipped(k))
        return out


def mawet_peak_snr(img: np.ndarray, rho, theta, pxscale: float, fwhm: float, *,
                   kernel=None, search_px: float = 1.5,
                   excl_fwhm: float = 1.5, known: Sequence[Tuple[float, float]] = (),
                   declip_nsig: Optional[float] = 6.0, band_fwhm: float = 0.6,
                   min_ring: int = 6, penalty: str = "geometric",
                   pixel_mask: Optional[np.ndarray] = None, inner_fwhm: float = 1.0,
                   angle_convention: str = "pa", return_details: bool = False,
                   excl_arcs: Sequence[Tuple[float, float, float, float]] = (),
                   peak_xy: Optional[Sequence[Tuple[float, float]]] = None,
                   exclude_xy: Sequence[Tuple[float, float]] = (), mfmap: Optional[np.ndarray] = None,
                   turn: Any = None,
                   templates_at: Optional[Sequence[Optional[Callable[[float, float], Optional[np.ndarray]]]]] = None,
                   correlate: bool = True):
    """Measure the matched-filter detection S/N of each source (Mawet et al. 2014).

    The image is despiked and correlated with ``kernel``, the template as a source looks.  Each
    source is located as the matched-filter peak within ``search_px`` of its position, and its noise
    comes from the ``nap = floor(2 pi r / fwhm)`` apertures on the ring through that peak, less those
    within ``excl_fwhm`` FWHM of an injected, known or excluded source.  A radial band
    ``|r' - r| <= band_fwhm * fwhm`` sets a floor on the sigma and replaces the ring when the ring
    keeps fewer than ``min_ring`` samples.  The small-sample penalty is ``sqrt(1 + 1/n)``.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, ny by nx pixels, with the star at ``((nx-1)/2, (ny-1)/2)``.
    rho : float or array_like
        Separation of each source, in arcsec.
    theta : float or array_like
        Angle of each source, in degrees, read as ``angle_convention`` says.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    kernel : ndarray, sequence or None, default None
        The matched filter, one 2-D kernel for every source or one per source, such as the
        forward-modeled templates of :mod:`klip_tpe.fmmf`.  None, or a None entry, takes the FWHM
        Gaussian, and each distinct kernel costs one correlation of the image.
    search_px : float, default 1.5
        Radius of the peak search around each source, in pixels.
    excl_fwhm : float, default 1.5
        Radius of the keep-out around each injected, known or excluded source, in FWHM.
    known : sequence of tuple of 2 floats, default ()
        Known companions, as ``(rho_arcsec, theta_deg)``, kept out of the noise.
    declip_nsig : float or None, default 6.0
        Threshold of the single-pixel despike in robust sigma (:func:`declip`).  None does not
        despike.
    band_fwhm : float, default 0.6
        Half width of the radial band, in FWHM.
    min_ring : int, default 6
        Minimum number of clean ring samples for the ring to give the noise.
    penalty : {"geometric", "retained"}, default "geometric"
        Count ``n`` in the small-sample penalty.  "geometric" takes ``max(nap, 3)``, as for NEAR,
        and "retained" the clean apertures kept, as for the NIRCam variant.
    pixel_mask : ndarray of bool or None, default None
        Pixels left out of the noise samples, True to exclude, such as a disk's.
    inner_fwhm : float, default 1.0
        Inner guard, in FWHM.  A source placed inside it gets no S/N.
    angle_convention : {"pa", "math"}, default "pa"
        How ``theta`` is read (:func:`source_xy`).
    return_details : bool, default False
        Whether to also return each source's geometry and noise statistics.
    excl_arcs : sequence of tuple of 4 floats, default ()
        Arcs left out of the ring and the band, as ``(rho_arcsec, theta_deg, half_width_deg,
        radial_fwhm)``.  These are a source's sidelobes, the arc an ADI reduction over-subtracts
        along at its radius (:mod:`klip_tpe.contrast_grid`).
    peak_xy : sequence of tuple of 2 floats or None, default None
        One ``(x, y)`` pixel per source where the peak is read instead of searched for, such as the
        injected image's peak for the clean term (:func:`sidelobe_snr`).  None searches.
    exclude_xy : sequence of tuple of 2 floats, default ()
        More ``(x, y)`` pixels whose keep-out the ring and the band skip, such as the verify's
        other sources (:func:`klip_tpe.verify.snr_fap`).
    mfmap : ndarray or None, default None
        Matched-filter map already made, used for every source instead of correlating ``img``
        (:func:`matched_filter`).  It does not go with ``turn`` or ``templates_at``.
    turn : bool, sequence of bool or None, default None
        Whether each source's kernel is a template in the source's own orientation, an ADI forward
        model's (:class:`klip_tpe.fmmf.FMMFSNR`).  Each noise sample then takes the template turned
        by its azimuth from the source (:func:`turn_kernel`), and None reads every sample off one
        map, which suits a round kernel.
    templates_at : sequence of callable or None, default None
        One entry per source, None or a function ``(x, y) -> template`` that gives the forward model
        of a source at that pixel (:class:`klip_tpe.fmmf.FMTemplates`).  Each noise sample then takes
        the template there, an entry takes the place of ``turn`` for its source, and a sample the
        function gives None for is read off the source's map.
    correlate : bool, default True
        Whether the matched filter correlates the image with the kernel, so a source shaped like it
        gives its own amplitude.  False convolves, which turns the template 180 deg first, as FMMF
        runs saved before 2026-11-04 measured.

    Returns
    -------
    snr : ndarray
        S/N of each source.  It may be negative, a peak below the ring mean, or NaN, for a source
        inside the inner guard, on a ring of fewer than 5 apertures or without noise samples.
    details : list of dict
        [return_details=True] Each source's ``x``, ``y``, ``r``, ``nap``, ``peak``, ``sigma``,
        ``bg``, ``nclean``, ``n_left``, ``min_ring``, ``ring_used`` and ``sd_band``, and for a NaN
        ``why`` (:data:`WHY_GUARD`, :data:`WHY_RING`, :data:`WHY_NOISE`).

    Raises
    ------
    ValueError
        If ``mfmap`` is not the image's shape, or ``kernel``, ``turn`` or ``templates_at`` has an
        entry count other than the number of sources.

    Notes
    -----
    The guard and the ring count are asked of the position the source was given, ``rho``, so a
    clean and an injected image are judged by the same geometry.  The IDL predecessor's
    ``near2_snrpk`` asked the guard of the located peak instead.  On a clean image that peak is
    whichever noise pixel is highest within ``search_px``, and for a source at 1.1 FWHM it lay inside
    1 FWHM about half the time, so the clean term, and with it the draw, failed while the injected
    image scored (NEAR2 note, 2026-10-13).  Where both guards pass, the S/N is the same.

    Under ``turn`` and ``templates_at`` the band's pixels are filtered in bins half a FWHM of arc
    wide, each with the template at its bin's center (NEAR2 note, 2026-11-01).  The source's own
    peak keeps the map of its unturned kernel.  Correlating and convolving give the same map, to
    the bit, for a kernel that is the same turned 180 deg, such as the Gaussian
    (``RunConfig.fmmf_correlate``, GitHub agent's finding 13).
    """
    img = np.asarray(img, float)
    ny, nx = img.shape
    cx, cy = star_center(img.shape)
    rho = np.atleast_1d(np.asarray(rho, float))
    theta = np.atleast_1d(np.asarray(theta, float))
    nsrc = rho.size
    exr = excl_fwhm * fwhm
    if mfmap is not None:
        img = np.asarray(img, float)
        if np.shape(mfmap) != img.shape:
            raise ValueError(f"mawet_peak_snr: mfmap {np.shape(mfmap)} is not the image's {img.shape}")
        # the map is made already.  The despike leaves every pixel as finite as it was, so the
        # image's own finite pixels are the prepared image's, and nothing is hashed or convolved
        prep = {"img": img, "finite": np.isfinite(img)}
        kerns = [None] * nsrc
    else:
        # the declip, the zero fill and each kernel's convolution, once per image and kernel
        prep = _prepare(img, declip_nsig)
        kerns = _per_source_kernels(kernel, nsrc, fwhm)
    img, imz = prep["img"], prep.get("imz")
    _amaps: Dict[int, np.ndarray] = {}
    if turn is None or turn is False or mfmap is not None:
        turns = [False] * nsrc
    elif turn is True:
        turns = [True] * nsrc
    else:
        turns = [bool(t) for t in turn]
        if len(turns) != nsrc:
            raise ValueError(f"mawet_peak_snr: {len(turns)} turn flags for {nsrc} sources")
    placed: List[Any] = [None] * nsrc
    if templates_at is not None and mfmap is None:
        placed = list(templates_at)
        if len(placed) != nsrc:
            raise ValueError(f"mawet_peak_snr: {len(placed)} templates_at entries for {nsrc} sources")
    _pads: Dict[int, np.ndarray] = {}

    def amap(i: int) -> np.ndarray:
        if mfmap is not None:
            return mfmap
        k = kerns[i]
        if id(k) not in _amaps:
            kk = _digest(np.asarray(k)) + (bool(correlate),)
            m = prep["maps"].get(kk)
            if m is None:
                m = (ndimage.correlate if correlate else ndimage.convolve)(imz, k, mode="nearest")
                _frozen(m)
                prep["maps"][kk] = m
            _amaps[id(k)] = m
        return _amaps[id(k)]

    xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
    ex = np.array([float(e[0]) for e in exclude_xy], float)
    ey = np.array([float(e[1]) for e in exclude_xy], float)
    kx = ky = np.zeros(0)
    if len(known):
        kr = np.array([k[0] for k in known], float)
        kt = np.array([k[1] for k in known], float)
        kx, ky = source_xy(kr, kt, pxscale, cx, cy, angle_convention)
    yy, xx, rr = _grids(ny, nx)
    finite = prep["finite"]
    if pixel_mask is not None:
        finite = finite & ~np.asarray(pixel_mask, bool)
    arcs = _arc_regions(excl_arcs, pxscale, fwhm, cx, cy, angle_convention)
    arc_px = _in_arcs(xx, yy, arcs, cx, cy) if arcs else None
    isr = int(np.ceil(search_px))
    recording = _RING_RECORDER is not None

    snr = np.full(nsrc, np.nan)
    details: List[Dict[str, float]] = []
    for s in range(nsrc):
        A = amap(s)
        px, py = int(round(xs[s])), int(round(ys[s]))
        pkv = -np.inf
        bx, by = px, py
        if peak_xy is not None:
            bx, by = int(round(float(peak_xy[s][0]))), int(round(float(peak_xy[s][1])))
            if 0 <= bx < nx and 0 <= by < ny and np.isfinite(img[by, bx]):
                pkv = A[by, bx]
        else:
            for dy in range(-isr, isr + 1):
                for dx in range(-isr, isr + 1):
                    if dx * dx + dy * dy > search_px * search_px:
                        continue
                    qx, qy = int(round(xs[s])) + dx, int(round(ys[s])) + dy
                    if 0 <= qx < nx and 0 <= qy < ny and np.isfinite(img[qy, qx]) and A[qy, qx] > pkv:
                        pkv, bx, by = A[qy, qx], qx, qy
        px, py = bx, by
        r = float(np.hypot(px - cx, py - cy))
        nap = int(np.floor(2 * np.pi * r / fwhm))
        det = {"x": px, "y": py, "r": r, "nap": nap, "peak": float(A[py, px]) if np.isfinite(pkv) else np.nan}
        r_req = float(rho[s]) / pxscale                # where the source was put (px)
        # (a source placed exactly on the guard passes: rho = floor * fwhm_as comes back
        # through pxscale a rounding error either side of it)
        if not r_req >= float(inner_fwhm) * fwhm * (1.0 - 1e-9):
            det["why"] = WHY_GUARD
            details.append(det)
            if recording:
                _ring_emit(_ring_rec("mawet_peak_snr", (ny, nx), cx, cy, det, fwhm, exr, [], xs, ys, kx, ky,
                                     excl_arcs, band_fwhm=band_fwhm))
            continue
        if int(np.floor(2 * np.pi * r_req / fwhm)) < 5:
            det["why"] = WHY_RING
            details.append(det)
            if recording:
                _ring_emit(_ring_rec("mawet_peak_snr", (ny, nx), cx, cy, det, fwhm, exr, [], xs, ys, kx, ky,
                                     excl_arcs, band_fwhm=band_fwhm))
            continue
        th0 = np.arctan2(py - cy, px - cx)
        vals, valok = [], []
        ring: Optional[List[List[Any]]] = [] if recording else None
        # each sample with the forward model of a source there (templates_at), or with the
        # source's template turned to the azimuth it is taken at (turn)
        pf = (_PlacedFilter(imz, placed[s], _pads, correlate=correlate)
              if placed[s] is not None and imz is not None else None)
        tf = (_TurnedFilter(imz, kerns[s], np.arctan2(ys[s] - cy, xs[s] - cx), _pads, correlate=correlate)
              if pf is None and turns[s] and imz is not None else None)
        for q in range(1, nap):
            th = th0 + q * 2 * np.pi / nap
            xk, yk = int(round(cx + r * np.cos(th))), int(round(cy + r * np.sin(th)))
            if np.any(np.hypot(xk - xs, yk - ys) < exr):
                if ring is not None:
                    ring.append([xk, yk, "source"])
                continue
            if ex.size and np.any(np.hypot(xk - ex, yk - ey) < exr):
                if ring is not None:
                    ring.append([xk, yk, "excluded"])
                continue
            if not (0 <= xk < nx and 0 <= yk < ny) or not finite[yk, xk]:
                if ring is not None:
                    ring.append([xk, yk, "masked" if (pixel_mask is not None and 0 <= xk < nx and 0 <= yk < ny
                                                      and bool(np.asarray(pixel_mask)[yk, xk])
                                                      and bool(prep["finite"][yk, xk])) else "off"])
                continue
            if arcs and bool(_in_arcs(xk, yk, arcs, cx, cy)):
                if ring is not None:
                    ring.append([xk, yk, "arc"])
                continue
            mok = not (kx.size and np.any(np.hypot(xk - kx, yk - ky) < exr))
            if pf is not None:
                v = pf.at(xk, yk)
                vals.append(A[yk, xk] if v is None else v)
            else:
                vals.append(A[yk, xk] if tf is None else tf.at(xk, yk, th - tf.az))
            valok.append(mok)
            if ring is not None:
                ring.append([xk, yk, "used" if mok else "known"])
        vals, valok = np.array(vals), np.array(valok, bool)
        # radial band statistics (sigma floor / fallback)
        sd_band, mn_band = -1.0, 0.0
        bnd = (np.abs(rr - r) <= band_fwhm * fwhm) & finite
        if bnd.sum() >= 5:
            keep = bnd.copy()
            for t in range(nsrc):
                keep &= (xx - xs[t]) ** 2 + (yy - ys[t]) ** 2 >= exr * exr
            for t in range(kx.size):
                keep &= (xx - kx[t]) ** 2 + (yy - ky[t]) ** 2 >= exr * exr
            for t in range(ex.size):
                keep &= (xx - ex[t]) ** 2 + (yy - ey[t]) ** 2 >= exr * exr
            if arc_px is not None:
                keep &= ~arc_px
            sel = keep if keep.sum() >= 5 else bnd
            if pf is not None:
                byy, bxx = np.nonzero(sel)
                bvals = pf.band(bxx, byy, cx, cy, r, float(np.arctan2(ys[s] - cy, xs[s] - cx)),
                                0.5 * fwhm / max(r, 1.0))
                off = ~np.isfinite(bvals)
                if off.any():
                    bvals[off] = A[byy[off], bxx[off]]
            elif tf is None:
                bvals = A[sel]
            else:
                byy, bxx = np.nonzero(sel)
                bvals = tf.band(bxx, byy, cx, cy, 0.5 * fwhm / max(r, 1.0))
            if bvals.size >= 5:
                sd_band, mn_band = float(np.std(bvals, ddof=1)), float(np.mean(bvals))
        nclean, sd_ring, mn_ring = 0, -1.0, 0.0
        if vals.size >= 3:
            use = vals
            if valok.sum() >= 4:
                use = vals[valok]
            nclean = use.size
            mn_ring = float(np.mean(use))
            sd_ring = float(np.std(use, ddof=1))
            if sd_ring <= 0:
                sd_ring = float(np.median(np.abs(use - np.median(use)))) * 1.4826
        sd, bg = -1.0, 0.0
        if nclean >= min_ring and sd_ring > 0:
            sd, bg = sd_ring, mn_ring
        elif sd_band > 0:
            sd, bg = sd_band, mn_band
        elif sd_ring > 0:
            sd, bg = sd_ring, mn_ring
        # nclean is the number of noise realizations this separation actually got, after
        # injected sources, known companions, the frame edge and the pixel mask have eaten
        # into the ring.  It decides whether the ring or the radial band is used, so it is
        # the quantity a source-count rule has to respect -- recorded so it can be checked.
        det["nclean"], det["min_ring"], det["n_left"] = int(nclean), int(min_ring), int(vals.size)
        det["ring_used"] = bool(nclean >= min_ring and sd_ring > 0)
        if sd > 0:
            if sd_band > 0:
                sd = max(sd, sd_band)
            n_pen = max(nap, 3) if penalty == "geometric" else max(nclean, 1)
            snr[s] = (A[py, px] - bg) / (sd * np.sqrt(1.0 + 1.0 / n_pen))
        else:
            det["why"] = WHY_NOISE
        det.update(sigma=sd, bg=bg, nclean=nclean, sd_band=sd_band)
        details.append(det)
        if ring is not None:
            if vals.size >= 3 and valok.sum() < 4:
                # fewer than 4 apertures off the known companions: the ring keeps theirs too
                ring = [[xk, yk, "used, known" if why == "known" else why] for xk, yk, why in ring]
            used = bool(det.get("ring_used"))
            noise = ("ring" if used else "band" if sd_band > 0 else
                     f"ring, fewer than {min_ring}" if sd_ring > 0 else None)
            if noise in ("band", None):
                ring = [[xk, yk, "unused" if why.startswith("used") else why] for xk, yk, why in ring]
            _ring_emit(_ring_rec("mawet_peak_snr", (ny, nx), cx, cy, det, fwhm, exr, ring, xs, ys, kx, ky,
                                 excl_arcs, band_fwhm=band_fwhm, snr=float(snr[s]), noise=noise,
                                 min_ring=int(min_ring), band_floor=bool(used and sd_band > sd_ring > 0),
                                 excluded_xy=[(float(a), float(b)) for a, b in zip(ex, ey)]))
    if return_details:
        return snr, details
    return snr


def _ring_rec(fn: str, shape, cx: float, cy: float, det: Dict[str, Any], fwhm: float, exr: float, ring, xs, ys,
              kx=(), ky=(), excl_arcs=(), **extra) -> Dict[str, Any]:
    """Build one source's noise ring as :class:`RingRecorder` keeps it.

    Parameters
    ----------
    fn : str
        Name of the S/N function that measured it.
    shape : tuple of int
        Shape of the image, ``(ny, nx)``.
    cx : float
        Column of the star, in pixels.
    cy : float
        Row of the star, in pixels.
    det : dict
        The source's details, as :func:`mawet_peak_snr` gives them.
    fwhm : float
        FWHM of the PSF, in pixels.
    exr : float
        Radius of the keep-outs, in pixels.
    ring : sequence of list
        Each aperture as ``[x, y, what]``.
    xs : sequence of float
        Columns of the sources, in pixels.
    ys : sequence of float
        Rows of the sources, in pixels.
    kx : sequence of float, default ()
        Columns of the known companions, in pixels.
    ky : sequence of float, default ()
        Rows of the known companions, in pixels.
    excl_arcs : sequence of tuple, default ()
        The arcs left out, as :func:`mawet_peak_snr` takes them.
    extra : dict
        More keys for the record.

    Returns
    -------
    rec : dict
        The record.
    """
    rec = {"fn": fn, "shape": tuple(int(v) for v in shape), "center": (float(cx), float(cy)),
           "x": float(det.get("x", np.nan)), "y": float(det.get("y", np.nan)), "r": float(det.get("r", np.nan)),
           "nap": int(det.get("nap", 0) or 0), "fwhm": float(fwhm), "exr": float(exr),
           "ring": [list(a) for a in ring], "sources_xy": [(float(a), float(b)) for a, b in zip(xs, ys)],
           "known_xy": [(float(a), float(b)) for a, b in zip(kx, ky)],
           "arcs": [tuple(float(v) for v in a) for a in excl_arcs], "why": det.get("why"),
           "nclean": int(det.get("nclean", 0) or 0)}
    rec.update(extra)
    return rec


#: the keywords of :func:`mawet_peak_snr` a run's metric sets, which every ring S/N of the run
#: takes (:func:`ring_rules`): the search's, the verify's and the candidate search's
RING_RULE_KEYS = ("search_px", "excl_fwhm", "declip_nsig", "band_fwhm", "min_ring", "penalty", "pixel_mask",
                  "inner_fwhm")

#: the rules of a run that measures as before 2026-11-01 (``RunConfig.verify_snr = "idl"``): the
#: verify and the candidate search take the S/N of the IDL predecessor's verify
#: (``near2v_snrfap``, a Gaussian kernel, no despike or band), param_verify the metric's
#: defaults, and no ring or map outside the search the pixel mask
VERIFY_IDL: Dict[str, Any] = {"verify": "idl"}


def verify_idl(rules: Optional[Dict[str, Any]]) -> bool:
    """Tell whether ring rules keep the measures of before 2026-11-01 (:data:`VERIFY_IDL`).

    Parameters
    ----------
    rules : dict or None
        Ring rules, as :func:`ring_rules` gives them.

    Returns
    -------
    old : bool
        True when ``rules["verify"]`` is "idl".
    """
    return bool(rules) and str(rules.get("verify", "")).lower() == "idl"


def ring_rules(metric) -> Dict[str, Any]:
    """Return the noise-ring rules of a run's metric, for the verify's and the candidate search's S/N.

    The rules are the keywords of :data:`RING_RULE_KEYS` the metric sets, ``kernel_at`` (its kernel
    at a separation) and ``flatten``.  A forward-model metric (:class:`~klip_tpe.fmmf.FMMFSNR`) gives
    the kernel it falls back to, its measured one or the Gaussian, since the verify has no forward
    model.

    Parameters
    ----------
    metric : MawetPeakSNR or None
        The run's metric.

    Returns
    -------
    rules : dict
        The rules, empty without such a metric, which leaves :func:`mawet_peak_snr`'s own defaults.
    """
    if metric is None or not hasattr(metric, "excl_fwhm"):
        return {}
    out: Dict[str, Any] = {k: getattr(metric, k) for k in RING_RULE_KEYS if hasattr(metric, k)}
    if callable(getattr(metric, "kernel", None)):
        out["kernel_at"] = metric.kernel
    elif callable(getattr(metric, "kernel_fn", None)) or getattr(metric, "mf_fwhm", None):
        kf = getattr(metric, "kernel_fn", None)
        fw = float(getattr(metric, "mf_fwhm", None) or getattr(metric, "fwhm"))

        def _at(rho, kf=kf, fw=fw):
            k = kf(float(rho)) if callable(kf) else None
            return gaussian_kernel(fw) if k is None else k
        out["kernel_at"] = _at
    if getattr(metric, "flatten", False):
        out["flatten"] = True
    return out


def rule_kwargs(rules: Optional[Dict[str, Any]], rho: float, shape) -> Dict[str, Any]:
    """Turn ring rules into :func:`mawet_peak_snr`'s keywords for a source at one separation.

    Parameters
    ----------
    rules : dict or None
        Ring rules, as :func:`ring_rules` gives them.
    rho : float
        Separation of the source, in arcsec, where the kernel is taken.
    shape : tuple of int
        Shape of the image, ``(ny, nx)``.  A pixel mask of another shape is dropped.

    Returns
    -------
    kwargs : dict
        The keywords, with ``kernel`` the kernel at ``rho``.
    """
    kw = {k: v for k, v in (rules or {}).items() if k in RING_RULE_KEYS}
    m = kw.get("pixel_mask")
    if m is not None and np.shape(m) != tuple(shape):
        kw.pop("pixel_mask")
    ka = (rules or {}).get("kernel_at")
    if callable(ka):
        kw["kernel"] = ka(float(rho))
    return kw


def matched_filter(img: np.ndarray, fwhm: float, rules: Optional[Dict[str, Any]] = None,
                   rho: Optional[float] = None) -> np.ndarray:
    """Return the matched-filter map :func:`mawet_peak_snr` measures an image on.

    The image is despiked, zero-filled and correlated with the kernel the rules give at ``rho``.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.
    fwhm : float
        FWHM of the PSF, in pixels, for the Gaussian kernel used when the rules give none.
    rules : dict or None, default None
        Ring rules, as :func:`ring_rules` gives them.  None despikes at 6 sigma and takes the
        Gaussian.
    rho : float or None, default None
        Separation where the kernel is taken, in arcsec.  None takes 0.

    Returns
    -------
    mfmap : ndarray, shape (ny, nx)
        The matched-filter map.
    """
    kw = rule_kwargs(rules, rho if rho is not None else 0.0, np.shape(img)) if rules else {}
    prep = _prepare(np.asarray(img, float), kw.get("declip_nsig", 6.0))
    k = kw.get("kernel")
    k = gaussian_kernel(fwhm) if k is None else np.asarray(k, float)
    return ndimage.correlate(prep["imz"], k, mode="nearest")


def _ring_fluxes(img, cx, cy, sep, phi0, nap, aprad, excl_xy, exr, known_xy, skip=None, trace=None,
                 n_src=None, skip_k=(), pixel_mask=None):
    """Return the aperture sums on a source's ring, the source's own position left out.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.
    cx : float
        Column of the star, in pixels.
    cy : float
        Row of the star, in pixels.
    sep : float
        Radius of the ring, in pixels.
    phi0 : float
        Azimuth of the source from +x, in radians.
    nap : int
        Number of positions on the ring, the source's included.
    aprad : float
        Radius of the apertures, in pixels.
    excl_xy : sequence of tuple of 2 floats
        The ``(x, y)`` pixels whose keep-out the ring skips, the sources first.
    exr : float
        Radius of the keep-outs, in pixels.
    known_xy : sequence of tuple of 2 floats
        The ``(x, y)`` pixels of known companions, whose apertures are kept but flagged False in
        ``ok``.
    skip : callable or None, default None
        Function ``(x, y) -> bool`` that leaves an aperture out, such as one in a sidelobe arc.
    trace : list or None, default None
        List that gets ``[x, y, what]`` for every aperture (:class:`RingRecorder`).
    n_src : int or None, default None
        Number of leading entries of ``excl_xy`` that are sources, the rest known companions.
    skip_k : sequence of int, default ()
        Ring positions left out by their index, the source's neighbors in the IDL predecessor's
        contrast generators.
    pixel_mask : ndarray of bool or None, default None
        Pixels that leave out an aperture centered on them, as :func:`mawet_peak_snr` leaves out a
        sample there.

    Returns
    -------
    ref : ndarray
        Sum of each aperture kept.
    ok : ndarray of bool
        True for each aperture kept clear of the known companions.
    """
    ref, ok = [], []
    ny, nx = np.shape(img)
    mask = None if pixel_mask is None or np.shape(pixel_mask) != (ny, nx) else np.asarray(pixel_mask, bool)
    for k in range(1, nap):
        phik = phi0 + k * 2 * np.pi / nap
        xk, yk = cx + sep * np.cos(phik), cy + sep * np.sin(phik)
        if k in skip_k:
            if trace is not None:
                trace.append([xk, yk, "source"])
            continue
        if mask is not None:
            ix, iy = int(round(xk)), int(round(yk))
            if 0 <= ix < nx and 0 <= iy < ny and mask[iy, ix]:
                if trace is not None:
                    trace.append([xk, yk, "masked"])
                continue
        hit = [j for j, (ex, ey) in enumerate(excl_xy) if np.hypot(xk - ex, yk - ey) < exr]
        if hit:
            if trace is not None:
                trace.append([xk, yk, "source" if n_src is None or min(hit) < n_src else "known"])
            continue
        if skip is not None and skip(xk, yk):
            if trace is not None:
                trace.append([xk, yk, "arc"])
            continue
        f = _aperture_sum(img, xk, yk, aprad)
        if np.isfinite(f):
            ref.append(f)
            ok.append(not any(np.hypot(xk - kx, yk - ky) < exr for kx, ky in known_xy))
            if trace is not None:
                trace.append([xk, yk, "used" if ok[-1] else "known"])
        elif trace is not None:
            trace.append([xk, yk, "off"])
    return np.array(ref), np.array(ok, bool)


def injection_difference_snr(img_inj: np.ndarray, img_clean: np.ndarray, rho, theta, pxscale: float,
                             fwhm: float, *, excl_fwhm: float = 1.5,
                             known: Sequence[Tuple[float, float]] = (),
                             angle_convention: str = "pa") -> np.ndarray:
    """Measure the injection-recovery S/N of each source on the difference of two images.

    The signal is the aperture flux of ``inj - clean`` at the source, and the noise the scatter of
    the apertures on the clean image's ring at its radius, times ``sqrt(1 + 1/nap)``.  A ring with
    fewer than 6 usable apertures is resampled at half-FWHM spacing, while the penalty keeps the
    independent count.

    Parameters
    ----------
    img_inj : ndarray, shape (ny, nx)
        The image with the sources injected.
    img_clean : ndarray, shape (ny, nx)
        The same reduction without them.
    rho : float or array_like
        Separation of each source, in arcsec.
    theta : float or array_like
        Angle of each source, in degrees, read as ``angle_convention`` says.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels, twice the aperture radius.
    excl_fwhm : float, default 1.5
        Radius of the keep-out around each source and known companion, in FWHM.
    known : sequence of tuple of 2 floats, default ()
        Known companions, as ``(rho_arcsec, theta_deg)``.
    angle_convention : {"pa", "math"}, default "pa"
        How ``theta`` is read (:func:`source_xy`).

    Returns
    -------
    snr : ndarray
        S/N of each source, NaN when its ring keeps fewer than 2 apertures or its own aperture holds
        no finite pixel.

    Notes
    -----
    It does what ``near2_snr_inj`` does in the IDL predecessor.
    """
    img_inj, img_clean = np.asarray(img_inj, float), np.asarray(img_clean, float)
    cx, cy = star_center(img_inj.shape)
    rho = np.atleast_1d(np.asarray(rho, float))
    theta = np.atleast_1d(np.asarray(theta, float))
    exr = excl_fwhm * fwhm
    aprad = fwhm / 2.0
    diff = np.where(np.isfinite(img_inj - img_clean), img_inj - img_clean, 0.0)
    xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
    excl = list(zip(xs, ys))
    known_xy = []
    if len(known):
        kx, ky = source_xy([k[0] for k in known], [k[1] for k in known], pxscale, cx, cy, angle_convention)
        known_xy = list(zip(kx, ky))
    phi = np.deg2rad(theta + (90.0 if angle_convention == "pa" else 0.0))
    snr = np.full(rho.size, np.nan)
    for s in range(rho.size):
        sep = rho[s] / pxscale
        nap = max(int(np.floor(2 * np.pi * sep / fwhm)), 4)
        f_src = _aperture_sum(diff, xs[s], ys[s], aprad)
        ref, ok = _ring_fluxes(img_clean, cx, cy, sep, phi[s], nap, aprad, excl, exr, known_xy)
        use = ref[ok] if ok.sum() >= 4 else ref
        if use.size < 6:
            nap2 = max(int(np.floor(2 * np.pi * sep / (0.5 * fwhm))), nap)
            ref2, ok2 = _ring_fluxes(img_clean, cx, cy, sep, phi[s], nap2, aprad, excl + known_xy, exr, [])
            if ref2.size > use.size:
                use = ref2
        if use.size < 2 or not np.isfinite(f_src):
            continue
        noise = np.std(use, ddof=1) * np.sqrt(1.0 + 1.0 / max(nap, 3))
        snr[s] = f_src / noise if noise > 0 else np.nan
    return snr


def aperture_mawet_snr(img: np.ndarray, rho, theta, pxscale: float, fwhm: float, *,
                       excl_fwhm: float = 1.5, angle_convention: str = "pa",
                       known: Sequence[Tuple[float, float]] = (),
                       excl_arcs: Sequence[Tuple[float, float, float, float]] = (),
                       return_details: bool = False, pixel_mask: Optional[np.ndarray] = None):
    """Measure each source's plain aperture S/N, as the IDL predecessor's contrast generators do.

    The S/N is ``(f_src - mean(ring)) / (std(ring) sqrt(1 + 1/n))`` with apertures of radius FWHM/2,
    the source's at its given position without a peak search, and ``n`` the apertures kept.  The
    ring holds ``ntheta = round(2 pi r / FWHM) - 1`` positions counted from the source's and skips
    the source and the position each side of it, or the source alone when ``ntheta < 6``.  The other
    sources, the known companions, ``excl_arcs`` and ``pixel_mask`` leave apertures out.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, with the star at ``((nx-1)/2, (ny-1)/2)``.
    rho : float or array_like
        Separation of each source, in arcsec.
    theta : float or array_like
        Angle of each source, in degrees, read as ``angle_convention`` says.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    excl_fwhm : float, default 1.5
        Radius of the keep-out around the other sources and the known companions, in FWHM.
    angle_convention : {"pa", "math"}, default "pa"
        How ``theta`` is read (:func:`source_xy`).
    known : sequence of tuple of 2 floats, default ()
        Known companions, as ``(rho_arcsec, theta_deg)``.
    excl_arcs : sequence of tuple of 4 floats, default ()
        Arcs left out of the ring, as for :func:`mawet_peak_snr`.
    return_details : bool, default False
        Whether to also return each source's ring statistics.
    pixel_mask : ndarray of bool or None, default None
        Pixels that leave out an aperture centered on them, as the run's disk mask leaves out the
        search's samples.

    Returns
    -------
    snr : ndarray
        S/N of each source, NaN when its ring keeps fewer than 2 apertures.
    details : list of dict
        [return_details=True] Each source's ``nclean`` and ``n_left``, the apertures used, ``nap``,
        ``peak`` the source's flux, ``bg`` and ``sigma`` the ring's mean and scatter, and ``why``
        for a NaN.

    Notes
    -----
    The ring is that of the IDL predecessor's ``contrastcurve_*.pro``.  Its optimizer's
    ``near2_snr`` placed ``floor(2 pi r / FWHM)`` positions and skipped 1.5 FWHM around the source,
    which left the same ring but for a position now and then, as this function did to 2026-11-01.
    """
    img = np.asarray(img, float)
    cx, cy = star_center(img.shape)
    rho = np.atleast_1d(np.asarray(rho, float))
    theta = np.atleast_1d(np.asarray(theta, float))
    exr = excl_fwhm * fwhm
    aprad = fwhm / 2.0
    imz = np.where(np.isfinite(img), img, 0.0)
    xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
    excl = list(zip(xs, ys))
    if len(known):
        kx, ky = source_xy([k[0] for k in known], [k[1] for k in known], pxscale, cx, cy, angle_convention)
        excl += list(zip(kx, ky))
    arcs = _arc_regions(excl_arcs, pxscale, fwhm, cx, cy, angle_convention)
    skip = (lambda x_, y_: bool(_in_arcs(x_, y_, arcs, cx, cy))) if arcs else None
    phi = np.deg2rad(theta + (90.0 if angle_convention == "pa" else 0.0))
    snr = np.full(rho.size, np.nan)
    details: List[Dict[str, Any]] = []
    recording = _RING_RECORDER is not None
    for s in range(rho.size):
        sep = rho[s] / pxscale
        nap = max(int(round(2 * np.pi * sep / fwhm)) - 1, 2)
        near = {1, nap - 1} if nap >= 6 else set()
        f_src = _aperture_sum(imz, xs[s], ys[s], aprad)
        trace: Optional[List[List[Any]]] = [] if recording else None
        others = excl[:s] + excl[s + 1:]               # the other sources, then the known companions
        ref, _ = _ring_fluxes(img, cx, cy, sep, phi[s], nap, aprad, others, exr, [], skip=skip, trace=trace,
                              n_src=int(rho.size) - 1, skip_k=near, pixel_mask=pixel_mask)
        details.append({"nclean": int(ref.size), "n_left": int(ref.size), "nap": nap, "peak": float(f_src)})
        if trace is not None:
            kxy = excl[int(rho.size):]
            _ring_emit(_ring_rec("aperture_mawet_snr", img.shape, cx, cy,
                                 dict(details[-1], x=xs[s], y=ys[s], r=sep), fwhm, exr, trace, xs, ys,
                                 [k[0] for k in kxy], [k[1] for k in kxy], excl_arcs,
                                 noise="ring" if ref.size >= 2 else None, aperture_px=aprad))
        if ref.size < 2 or not np.isfinite(f_src):
            details[-1]["why"] = WHY_NOISE
            continue
        sd = float(np.std(ref, ddof=1))
        details[-1].update(bg=float(ref.mean()), sigma=sd)
        noise = sd * np.sqrt(1.0 + 1.0 / ref.size)
        snr[s] = (f_src - ref.mean()) / noise if noise > 0 else np.nan
    if return_details:
        return snr, details
    return snr


# ----------------------------------------------------------------------------
# noise rings without the sources' ADI sidelobes
# ----------------------------------------------------------------------------
@dataclass
class Sidelobes:
    """The noise rings' sidelobe rule, for :func:`sidelobe_snr`.

    ADI over-subtracts a source along the arc the field rotation sweeps at its radius, within
    ``radial_fwhm`` FWHM of its separation.  Every injected source's arc and every known companion's
    are left out of every source's noise ring.  Where fewer than ``min_apertures`` apertures remain,
    ``mode`` decides where the noise comes from.

    Attributes
    ----------
    rotation_deg : float or None, default None
        Half width of the arcs, in degrees, the field rotation.  None takes the floor alone.
    mode : {"farthest", "clean"}, default "farthest"
        Fallback when too few apertures remain.  "farthest" takes the apertures farthest from the
        source, its own arc shrinking to let them in, and "clean" measures the noise on the clean
        reduction's ring through the source, which holds no injected source.
    min_apertures : int, default 4
        Minimum number of apertures a ring must keep.
    floor_fwhm : float, default 2.0
        Minimum half width of an arc, in FWHM of arc at its radius.
    radial_fwhm : float, default 1.5
        Radial half width of an arc, in FWHM.
    """

    rotation_deg: Optional[float] = None
    mode: str = "farthest"
    min_apertures: int = 4
    floor_fwhm: float = 2.0
    radial_fwhm: float = 1.5

    def half_width(self, r_px: float, fwhm: float) -> float:
        """Return the half width of an arc at a separation.

        Parameters
        ----------
        r_px : float
            Separation, in pixels.
        fwhm : float
            FWHM of the PSF, in pixels.

        Returns
        -------
        hw : float
            Half width in degrees, the larger of ``rotation_deg`` and the floor, at most 180.
        """
        floor = float(np.rad2deg(self.floor_fwhm * fwhm / max(float(r_px), 1e-6)))
        rot = self.rotation_deg
        if rot is None or not np.isfinite(rot):
            return min(floor, 180.0)
        return min(max(float(rot), floor), 180.0)

    def describe(self) -> Dict[str, Any]:
        """Return the rule's settings, for a run's record.

        Returns
        -------
        info : dict
            The five fields.
        """
        return {"rotation_deg": None if self.rotation_deg is None else float(self.rotation_deg),
                "mode": str(self.mode), "min_apertures": int(self.min_apertures),
                "floor_fwhm": float(self.floor_fwhm), "radial_fwhm": float(self.radial_fwhm)}


def _farther_widths(nap: int, start_deg: float) -> List[float]:
    """Return the half widths below ``start_deg`` that let in the ring's apertures group by group.

    Each half width admits the next group of apertures in order of their distance from the source,
    widest first.

    Parameters
    ----------
    nap : int
        Number of positions on the ring, the source's own at 0.
    start_deg : float
        Half width of the source's own arc, in degrees.

    Returns
    -------
    widths : list of float
        The half widths, in degrees, widest first.
    """
    step = 360.0 / max(int(nap), 1)
    dist = sorted({min(q, nap - q) * step for q in range(1, int(nap))}, reverse=True)
    return [d - 0.25 * step for d in dist if d - 0.25 * step < start_deg]


def sidelobe_snr(img: np.ndarray, rho, theta, pxscale: float, fwhm: float, sidelobes: Sidelobes, *,
                 img_clean: Optional[np.ndarray] = None, known: Sequence[Tuple[float, float]] = (),
                 measure: Optional[Callable] = None, angle_convention: str = "pa",
                 only: Optional[int] = None, drop: Sequence[str] = ("known",), **kw):
    """Measure each source's S/N with :class:`Sidelobes` noise rings, one source at a time.

    Every source's arc and the known companions' arcs are left out of the source's ring, then the
    rule of at least ``min_apertures`` apertures applies.  :func:`mawet_peak_snr`'s ``min_ring``
    defaults to ``min_apertures``, so a ring that keeps them is used rather than the radial band.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, with the star at ``((nx-1)/2, (ny-1)/2)``.
    rho : float or array_like
        Separation of each source, in arcsec.
    theta : float or array_like
        Angle of each source, in degrees, read as ``angle_convention`` says.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    sidelobes : Sidelobes
        The rule.
    img_clean : ndarray or None, default None
        The clean reduction, for ``mode="clean"``.  Without it that mode takes the farthest
        apertures.
    known : sequence of tuple of 2 floats, default ()
        Known companions, as ``(rho_arcsec, theta_deg)``.
    measure : callable or None, default None
        S/N function with :func:`mawet_peak_snr`'s signature, which it defaults to.  It may take
        ``peak_xy`` (``measure.takes_peak_xy``), for the clean ring through the injected source.
    angle_convention : {"pa", "math"}, default "pa"
        How ``theta`` is read (:func:`source_xy`).
    only : int or None, default None
        Index of the one source to measure, the others giving their arcs and a NaN S/N.  None
        measures all.
    drop : sequence of {"known", "others"}, default ("known",)
        What the farthest fallback lets in, in order, once the source's own arc is gone.  "known"
        lets in the known companions' arcs, as the run's metric does, and "others" the other
        sources' arcs, which ``("others", "known")`` lets in first, as the verify does.
    kw : dict
        More keywords for ``measure``.

    Returns
    -------
    snr : ndarray
        S/N of each source.
    details : list of dict
        Each source's details from ``measure``, plus ``arcs`` (the arcs left out), ``arc_deg`` (its
        own arc's half width), ``n_arcs`` (the apertures outside all the arcs) and ``fallback``.
        Where the fallback too leaves fewer than ``min_apertures``, ``fallback_tried`` names it and
        ``n_fallback`` is the most apertures it found.

    Raises
    ------
    ValueError
        If ``drop`` holds something other than "known" and "others".

    Notes
    -----
    ``fallback`` is None, ``"farthest"``, ``"farthest, known arcs too"``,
    ``"farthest, the other sources' arcs too"``, ``"farthest, every arc let in"`` (the ring without
    the rule) or ``"clean"``.  The clean ring runs through the injected source where the injected
    image located it (``peak_xy``), so the two rings have one radius.
    """
    measure = measure or mawet_peak_snr
    rho = np.atleast_1d(np.asarray(rho, float))
    theta = np.atleast_1d(np.asarray(theta, float))
    n = rho.size
    min_ap = max(int(sidelobes.min_apertures), 1)
    rad = float(sidelobes.radial_fwhm)
    known = [(float(k[0]), float(k[1])) for k in known]
    known_arcs = [(kr, kt, sidelobes.half_width(kr / pxscale, fwhm), rad) for kr, kt in known]
    hw = [sidelobes.half_width(r / pxscale, fwhm) for r in rho]
    src_arcs = [(float(rho[i]), float(theta[i]), hw[i], rad) for i in range(n)]
    if measure is mawet_peak_snr:
        kw.setdefault("min_ring", min_ap)
    retained = measure is not mawet_peak_snr or str(kw.get("penalty", "geometric")) != "geometric"
    snr = np.full(n, np.nan)
    dets: List[Dict[str, Any]] = []
    # RingRecorder: each attempt's ring is held back, and the one the source kept is recorded
    recording = _RING_RECORDER is not None
    held: List[Dict[str, Any]] = []
    emits: List[Dict[str, Any]] = []
    all_xy: List[Tuple[float, float]] = []
    if recording:
        if getattr(_RING_LOCAL, "held", None) is None:
            _RING_LOCAL.held = []
        _RING_LOCAL.held.append(held)
        cx0, cy0 = star_center(np.shape(img))
        axs, ays = source_xy(rho, theta, pxscale, cx0, cy0, angle_convention)
        all_xy = [(float(a), float(b)) for a, b in zip(axs, ays)]

    def n_left(d_: Dict[str, Any]) -> int:
        """The ring's apertures left after the exclusions (``nclean`` is 0 below 3)."""
        return int(d_.get("n_left", d_.get("nclean", 0)))
    try:
        _sidelobe_sources(n, only, src_arcs, rho, theta, hw, rad, known, known_arcs, img, img_clean, pxscale,
                          fwhm, sidelobes, measure, angle_convention, kw, min_ap, retained, drop, snr, dets,
                          held, emits, all_xy, n_left)
    finally:
        if recording:
            _RING_LOCAL.held.pop()
    for e in emits:
        _ring_emit(e)
    return snr, dets


def _sidelobe_sources(n, only, src_arcs, rho, theta, hw, rad, known, known_arcs, img, img_clean, pxscale, fwhm,
                      sidelobes, measure, angle_convention, kw, min_ap, retained, drop, snr, dets, held, emits,
                      all_xy, n_left) -> None:
    """Run :func:`sidelobe_snr`'s loop over the sources, filling ``snr`` and ``dets``.

    Parameters
    ----------
    n : int
        Number of sources.
    only : int or None
        Index of the one source to measure, or None for all.
    src_arcs : list of tuple of 4 floats
        Each source's arc, ``(rho, theta, half_width_deg, radial_fwhm)``.
    rho : ndarray
        Separation of each source, in arcsec.
    theta : ndarray
        Angle of each source, in degrees.
    hw : list of float
        Half width of each source's arc, in degrees.
    rad : float
        Radial half width of the arcs, in FWHM.
    known : list of tuple of 2 floats
        Known companions, as ``(rho_arcsec, theta_deg)``.
    known_arcs : list of tuple of 4 floats
        The known companions' arcs.
    img : ndarray
        The image.
    img_clean : ndarray or None
        The clean reduction, for ``mode="clean"``.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    sidelobes : Sidelobes
        The rule.
    measure : callable
        The S/N function.
    angle_convention : {"pa", "math"}
        How ``theta`` is read.
    kw : dict
        More keywords for ``measure``.
    min_ap : int
        Minimum number of apertures a ring must keep.
    retained : bool
        Whether the clean fallback's penalty counts the apertures kept rather than the ring's
        positions.
    drop : sequence of str
        What the farthest fallback lets in, as for :func:`sidelobe_snr`.
    snr : ndarray
        S/N of each source, filled in place.
    dets : list
        Details of each source, appended to.
    held : list
        Rings the measure recorded while a :class:`RingRecorder` is active, emptied as they are used.
    emits : list
        Ring records of the attempt each source kept, appended to.
    all_xy : list of tuple of 2 floats
        The ``(x, y)`` pixel of every source, for the records.
    n_left : callable
        Function that gives the apertures a ring kept after the exclusions, from its details.
    """
    for i in range(n):
        others = src_arcs[:i] + src_arcs[i + 1:]

        if only is not None and i != int(only):
            dets.append({})
            continue

        def run(own: float, with_known: bool = True, image=img, arcs=None, **over):
            if arcs is None:
                arcs = ([(src_arcs[i][0], src_arcs[i][1], own, rad)] if own > 0 else []) + others \
                    + (known_arcs if with_known else [])
            n0 = len(held)
            s_, d_ = measure(image, [rho[i]], [theta[i]], pxscale, fwhm, known=known, excl_arcs=arcs,
                             angle_convention=angle_convention, return_details=True, **dict(kw, **over))
            out = dict(d_[0], arcs=list(arcs))
            if len(held) > n0:
                out["_ring"] = held[-1]
                del held[n0:]
            return float(s_[0]), out
        noise_ring = None
        s, d = run(hw[i])
        d.update(arc_deg=hw[i], n_arcs=n_left(d), fallback=None)
        if not d.get("why") and int(d.get("nclean", 0)) < min_ap:
            if sidelobes.mode == "clean" and img_clean is not None:
                # the clean ring through the injected source as located: a peak search of its
                # own would settle on a noise peak up to search_px away, at another radius
                over = ({"peak_xy": [(d["x"], d["y"])]}
                        if (measure is mawet_peak_snr or getattr(measure, "takes_peak_xy", False))
                        and "x" in d and "y" in d else {})
                _, dc = run(0.0, image=img_clean, arcs=known_arcs, **over)
                cring = dc.pop("_ring", None)
                peak, sd = d.get("peak", np.nan), dc.get("sigma", -1.0)
                if not dc.get("why") and int(dc.get("nclean", 0)) >= min_ap and sd > 0 and np.isfinite(peak):
                    n_pen = max(int(dc.get("nclean", 1)), 1) if retained else max(int(dc.get("nap", 3)), 3)
                    s = (float(peak) - float(dc.get("bg", 0.0))) / (float(sd) * np.sqrt(1.0 + 1.0 / n_pen))
                    d.update(fallback="clean", nclean=int(dc["nclean"]), sigma=float(sd), bg=float(dc.get("bg", 0.0)))
                    if cring is not None:
                        noise_ring = dict(cring, noise_image="clean")
                else:
                    d.update(fallback_tried="clean", n_fallback=n_left(dc))
            else:
                found, most = False, d["n_arcs"]
                for own in _farther_widths(int(d.get("nap", 0) or 0), hw[i]):
                    s2, d2 = run(own)
                    if d2.get("why"):
                        break
                    most = max(most, n_left(d2))
                    if int(d2.get("nclean", 0)) >= min_ap:
                        s, d, found = s2, dict(d2, arc_deg=own, n_arcs=d["n_arcs"], fallback="farthest"), True
                        break
                gone: set = set()
                for what in drop:
                    if found:
                        break
                    if what not in ("known", "others"):
                        raise ValueError(f"sidelobe_snr: drop={tuple(drop)!r}: 'known' and 'others'")
                    if not (known_arcs if what == "known" else others):
                        continue
                    gone.add(what)
                    arcs_ = ([] if "others" in gone else others) + ([] if "known" in gone else known_arcs)
                    s2, d2 = run(0.0, arcs=arcs_)
                    if d2.get("why"):
                        continue
                    most = max(most, n_left(d2))
                    if int(d2.get("nclean", 0)) >= min_ap:
                        how = ("farthest, known arcs too" if gone == {"known"} else
                               "farthest, the other sources' arcs too" if gone == {"others"} else
                               "farthest, every arc let in")
                        s, d, found = s2, dict(d2, arc_deg=0.0, n_arcs=d["n_arcs"], fallback=how), True
                if not found:
                    d.update(fallback_tried="farthest", n_fallback=most)
        ring = d.pop("_ring", None)
        if noise_ring is not None:
            ring = noise_ring
        if ring is not None:
            emits.append(dict(ring, rule=str(sidelobes.mode), fallback=d.get("fallback"),
                              fallback_tried=d.get("fallback_tried"), arc_deg=d.get("arc_deg"),
                              n_arcs=d.get("n_arcs"), all_sources_xy=list(all_xy), measured=int(i)))
        snr[i] = s
        dets.append(d)


# ----------------------------------------------------------------------------
# metric objects
# ----------------------------------------------------------------------------
class Metric:
    """Base class of the metrics, callables ``(img_inj, img_clean, sources) -> per-source S/N``.

    A metric that sets ``needs_fm`` True (:class:`~klip_tpe.fmmf.FMMFSNR`) asks the runner for the
    KLIP-FM pass of the same configuration and is called with the keyword ``fm``, the forward-modeled
    response of the injected sources, from which it builds its position-dependent matched filter.
    ``fm=None`` must stay valid, since the k-scan path has no forward model and the metric falls back
    there.

    Attributes
    ----------
    name : str
        Name of the metric, for a run's record.
    needs_clean : bool
        Whether the metric uses the clean image on every call.
    needs_fm : bool
        Whether the metric asks for the KLIP-FM pass.
    """

    name = "metric"
    needs_clean = False
    needs_fm = False

    def per_source(self, img_inj, img_clean, rho, theta) -> np.ndarray:  # pragma: no cover
        """Return each source's S/N.

        Parameters
        ----------
        img_inj : ndarray, shape (ny, nx)
            The image scored.
        img_clean : ndarray or None
            The clean reduction, for a metric that uses it.
        rho : sequence of float
            Separation of each source, in arcsec.
        theta : sequence of float
            Angle of each source, in degrees.

        Returns
        -------
        snr : ndarray
            S/N of each source.
        """
        raise NotImplementedError

    def describe(self) -> Dict:
        """Return the metric's settings, for a run's record.

        Returns
        -------
        info : dict
            ``name``, with more keys in subclasses.
        """
        return {"name": self.name}


@dataclass
class MawetPeakSNR(Metric):
    """Default search and validation metric, the matched-filter S/N of :func:`mawet_peak_snr`.

    Attributes
    ----------
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    kernel_fn : callable or None, default None
        Function ``rho_arcsec -> kernel`` with which an injection model supplies its measured
        matched filter at the sources' median separation.  None, or a None kernel, takes the
        Gaussian.
    mf_fwhm : float or None, default None
        FWHM of the Gaussian matched filter, in pixels.  None takes ``fwhm``.
    search_px : float, default 1.5
        Radius of the peak search, in pixels.
    excl_fwhm : float, default 1.5
        Radius of the keep-outs, in FWHM.
    known : sequence of tuple of 2 floats, default ()
        Known companions, as ``(rho_arcsec, theta_deg)``.
    declip_nsig : float or None, default 6.0
        Threshold of the despike in robust sigma.  None does not despike.
    band_fwhm : float, default 0.6
        Half width of the radial band, in FWHM.
    min_ring : int, default 6
        Minimum number of clean ring samples for the ring to give the noise.
    penalty : {"geometric", "retained"}, default "geometric"
        Count in the small-sample penalty (:func:`mawet_peak_snr`).
    pixel_mask : ndarray of bool or None, default None
        Pixels left out of the noise samples, True to exclude.
    flatten : bool, default False
        Whether to subtract the azimuthal mean profile (:func:`radprof`) before scoring.
    angle_convention : {"pa", "math"}, default "pa"
        How the sources' angles are read (:func:`source_xy`).
    name : str, default "mawet_peak"
        Name of the metric, for a run's record.
    inner_fwhm : float, default 1.0
        Inner guard, in FWHM.  The :class:`~klip_tpe.runner.Runner` lowers it to the injection floor
        when the floor is closer in (``RunConfig.min_r_fwhm``), so a run that injects there is
        scored there.

    Notes
    -----
    ``flatten`` False departs from the IDL predecessor, every variant of which flattens.
    :func:`radprof` subtracts the mean of each integer-radius bin, so anything not azimuthally uniform
    at a radius leaks into that mean and is subtracted off the whole ring, such as a bright
    companion, the MIRI 4QPM dead zones or a glow-stick residual.  On a ring the mask already eats,
    the mean comes from the surviving pixels and is removed from all of them, which biases the
    sources the metric measures.
    """

    pxscale: float
    fwhm: float
    kernel_fn: Optional[Callable[[float], np.ndarray]] = None
    mf_fwhm: Optional[float] = None
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
    name: str = "mawet_peak"
    inner_fwhm: float = 1.0

    def kernel(self, rho) -> np.ndarray:
        """Return the matched filter for sources at the given separations.

        Parameters
        ----------
        rho : float or array_like
            Separations of the sources, in arcsec, whose median is used.

        Returns
        -------
        kernel : ndarray
            The measured filter from ``kernel_fn``, or the Gaussian of ``mf_fwhm`` or ``fwhm``.
        """
        if self.kernel_fn is not None:
            k = self.kernel_fn(float(np.nanmedian(np.atleast_1d(rho))))
            if k is not None:
                return k
        return gaussian_kernel(self.mf_fwhm or self.fwhm)

    def per_source(self, img_inj, img_clean, rho, theta, why: Optional[list] = None,
                   sidelobes: Optional["Sidelobes"] = None) -> np.ndarray:
        """Return each source's S/N on an image.

        Parameters
        ----------
        img_inj : ndarray, shape (ny, nx)
            The image scored.
        img_clean : ndarray or None
            The clean reduction, used only by ``sidelobes`` with ``mode="clean"``.
        rho : sequence of float
            Separation of each source, in arcsec.
        theta : sequence of float
            Angle of each source, in degrees.
        why : list or None, default None
            List filled with each source's reason for a NaN, None for a finite S/N.
        sidelobes : Sidelobes or None, default None
            Rule that leaves the sources' and the known companions' ADI sidelobes out of the noise
            rings (:func:`sidelobe_snr`).  None uses plain rings.

        Returns
        -------
        snr : ndarray
            S/N of each source.
        """
        img = radprof(img_inj) if self.flatten else np.asarray(img_inj, float)
        kw = dict(kernel=self.kernel(rho), search_px=self.search_px, excl_fwhm=self.excl_fwhm,
                  declip_nsig=self.declip_nsig, band_fwhm=self.band_fwhm, penalty=self.penalty,
                  pixel_mask=self.pixel_mask, inner_fwhm=self.inner_fwhm)
        if sidelobes is not None:
            cln = None if img_clean is None else (radprof(img_clean) if self.flatten else np.asarray(img_clean, float))
            snr, det = sidelobe_snr(img, rho, theta, self.pxscale, self.fwhm, sidelobes, img_clean=cln,
                                    known=self.known, angle_convention=self.angle_convention, **kw)
        else:
            snr, det = mawet_peak_snr(img, rho, theta, self.pxscale, self.fwhm, known=self.known,
                                      min_ring=self.min_ring, angle_convention=self.angle_convention,
                                      return_details=True, **kw)
        if why is not None:
            why[:] = [d.get("why") for d in det]
        return snr

    def describe(self) -> Dict:
        """Return the metric's settings, for a run's record.

        Returns
        -------
        info : dict
            The settings, with ``pixel_mask_px`` the number of masked pixels, since a masked noise
            estimate is not comparable with an unmasked one.
        """
        return {"name": self.name, "fwhm": self.fwhm, "pxscale": self.pxscale,
                "mf_fwhm": self.mf_fwhm, "measured_kernel": self.kernel_fn is not None,
                "search_px": self.search_px, "excl_fwhm": self.excl_fwhm,
                "known": [list(k) for k in self.known], "declip_nsig": self.declip_nsig,
                "band_fwhm": self.band_fwhm, "min_ring": self.min_ring, "penalty": self.penalty,
                "inner_fwhm": self.inner_fwhm,
                # a masked noise estimate is not comparable with an unmasked one, so the
                # run's own setup record has to say whether one was in force
                "pixel_mask_px": None if self.pixel_mask is None else int(np.count_nonzero(self.pixel_mask)),
                "flatten": self.flatten}


@dataclass
class InjectionDifferenceSNR(Metric):
    """Metric on the difference of the injected and the clean images (:func:`injection_difference_snr`).

    It needs the clean image on every call.

    Attributes
    ----------
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    excl_fwhm : float, default 1.5
        Radius of the keep-outs, in FWHM.
    known : sequence of tuple of 2 floats, default ()
        Known companions, as ``(rho_arcsec, theta_deg)``.
    flatten : bool, default False
        Whether to subtract the azimuthal mean profile (:func:`radprof`) from both images first.
    angle_convention : {"pa", "math"}, default "pa"
        How the sources' angles are read (:func:`source_xy`).
    name : str, default "injection_difference"
        Name of the metric, for a run's record.
    needs_clean : bool, default True
        Whether the metric uses the clean image, always True here.

    Notes
    -----
    It is the ``use_noinj_noise`` option of the IDL predecessor.
    """

    pxscale: float
    fwhm: float
    excl_fwhm: float = 1.5
    known: Sequence[Tuple[float, float]] = ()
    flatten: bool = False
    angle_convention: str = "pa"
    name: str = "injection_difference"
    needs_clean: bool = True

    def per_source(self, img_inj, img_clean, rho, theta) -> np.ndarray:
        """Return each source's S/N on the difference image.

        Parameters
        ----------
        img_inj : ndarray, shape (ny, nx)
            The reduction with the sources injected.
        img_clean : ndarray
            The clean reduction.
        rho : sequence of float
            Separation of each source, in arcsec.
        theta : sequence of float
            Angle of each source, in degrees.

        Returns
        -------
        snr : ndarray
            S/N of each source.

        Raises
        ------
        ValueError
            If ``img_clean`` is None.
        """
        if img_clean is None:
            raise ValueError("InjectionDifferenceSNR needs the clean image")
        a = radprof(img_inj) if self.flatten else np.asarray(img_inj, float)
        b = radprof(img_clean) if self.flatten else np.asarray(img_clean, float)
        return injection_difference_snr(a, b, rho, theta, self.pxscale, self.fwhm,
                                        excl_fwhm=self.excl_fwhm, known=self.known,
                                        angle_convention=self.angle_convention)

    def describe(self) -> Dict:
        """Return the metric's settings, for a run's record.

        Returns
        -------
        info : dict
            ``name``, ``fwhm``, ``pxscale``, ``excl_fwhm`` and ``known``.
        """
        return {"name": self.name, "fwhm": self.fwhm, "pxscale": self.pxscale,
                "excl_fwhm": self.excl_fwhm, "known": [list(k) for k in self.known]}


@dataclass
class ScoreResult:
    """The objective's score of one draw.

    Attributes
    ----------
    score : float
        Aggregate of ``per_source``, NaN for a failed draw.
    per_source : ndarray
        Per-source values after the clean correction.
    raw_per_source : ndarray
        Per-source values on the injected image, uncorrected.
    clean_per_source : ndarray or None, default None
        Per-source values on the clean image, or None without a clean term.
    why : str or None, default None
        Why ``score`` is NaN, such as ``"clean S/N NaN (inside the inner guard)"``, or None when it
        is not.
    """
    score: float                         # aggregate (nan = failed)
    per_source: np.ndarray               # corrected per-source values
    raw_per_source: np.ndarray           # uncorrected injected-image values
    clean_per_source: Optional[np.ndarray] = None
    #: why ``score`` is NaN (None when it is not), e.g. ``"clean S/N NaN (inside the inner
    #: guard)"``: which image failed at each source, and the metric's reason when it gives one
    why: Optional[str] = None

    @property
    def raw_score(self) -> float:
        """Median of the uncorrected per-source values.
        """
        return nanmedian_even(self.raw_per_source)


_TAKES_WHY: Dict[Any, bool] = {}


def _takes(metric, name: str) -> bool:
    """Tell whether a metric's ``per_source`` takes a keyword.

    Parameters
    ----------
    metric : Metric
        The metric.
    name : str
        Name of the keyword.

    Returns
    -------
    takes : bool
        True when the signature names it.  The answer is cached per function.
    """
    ps = getattr(metric, "per_source", None)
    key = (getattr(ps, "__func__", ps), name)
    try:
        return _TAKES_WHY[key]
    except (KeyError, TypeError):
        pass
    try:
        ok = name in inspect.signature(ps).parameters
    except (TypeError, ValueError):
        ok = False
    try:
        _TAKES_WHY[key] = ok
    except TypeError:
        pass
    return ok


def _takes_why(metric) -> bool:
    """Tell whether a metric's ``per_source`` takes ``why`` (see :meth:`MawetPeakSNR.per_source`).

    Parameters
    ----------
    metric : Metric
        The metric.

    Returns
    -------
    takes : bool
        True when the signature names ``why``.
    """
    return _takes(metric, "why")


def why_failed(raw, why_raw=None, clean=None, why_clean=None) -> str:
    """Explain why an aggregate of per-source values is NaN.

    At each source it names the image whose S/N was NaN, the injected one first, and the metric's
    reason when it gave one, counted over the sources.

    Parameters
    ----------
    raw : array_like
        Per-source values on the injected image.
    why_raw : list of str or None, default None
        The metric's reason for each NaN in ``raw``.
    clean : array_like or None, default None
        Per-source values on the clean image.
    why_clean : list of str or None, default None
        The metric's reason for each NaN in ``clean``.

    Returns
    -------
    why : str
        The explanation, such as "clean S/N NaN (inside the inner guard) at 2 of 3 sources", or
        "no sources".
    """
    raw = np.atleast_1d(np.asarray(raw, float))
    if raw.size == 0:
        return "no sources"
    items: List[str] = []
    for s in range(raw.size):
        if not np.isfinite(raw[s]):
            r = why_raw[s] if why_raw else None
            items.append("injected S/N NaN" + (f" ({r})" if r else ""))
        elif clean is not None and not np.isfinite(np.asarray(clean, float)[s]):
            r = why_clean[s] if why_clean else None
            items.append("clean S/N NaN" + (f" ({r})" if r else ""))
        else:
            items.append("S/N not finite")
    counts: Dict[str, int] = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    if raw.size == 1:
        return items[0]
    return ", ".join(f"{k} at {v} of {raw.size} sources" for k, v in counts.items())


class Objective:
    """Objective that turns per-source metric values into the scalar the optimizer maximizes.

    With ``clean_subtract`` each source scores ``s_inj - max(s_clean, 0)``, a one-sided penalty that
    removes speckle contamination at the injection site but never makes signal from a dark hole.  A
    NaN clean term drops the source from the aggregate, and validation always uses
    :meth:`score_raw`.

    Parameters
    ----------
    metric : Metric
        The per-source metric.
    clean_subtract : bool, default True
        Whether to subtract the clean image's score, as production runs do.  A metric that already
        uses the clean image (``metric.needs_clean``) ignores it.
    aggregate : {"median", "mean"}, default "median"
        How the per-source values combine.

    Attributes
    ----------
    sidelobes : Sidelobes or None
        Noise rings' sidelobe rule, handed to a metric that takes it.  The
        :class:`~klip_tpe.runner.Runner` sets it from ``RunConfig.sidelobes``.

    Notes
    -----
    A metric that already uses the clean image, such as :class:`InjectionDifferenceSNR` on
    ``inj - clean``, removes the speckle term by construction.  For it :meth:`score_search` returns
    the raw values with ``clean_per_source`` None, instead of subtracting a second, ill-defined
    clean term.
    """

    def __init__(self, metric: Metric, clean_subtract: bool = True, aggregate: str = "median"):
        self.metric = metric
        self.clean_subtract = bool(clean_subtract)
        self.aggregate = aggregate
        #: the noise rings' sidelobe rule (:class:`Sidelobes`), handed to a metric that takes
        #: it; the :class:`~klip_tpe.runner.Runner` sets it from ``RunConfig.sidelobes``
        self.sidelobes: Optional[Sidelobes] = None

    @property
    def needs_clean(self) -> bool:
        """Whether a search score needs the clean image.
        """
        return self.clean_subtract or self.metric.needs_clean

    @property
    def effective_clean_subtract(self) -> bool:
        """Whether the clean term is applied, False for a metric that already uses the clean image.
        """
        return self.clean_subtract and not self.metric.needs_clean

    def _agg(self, v) -> float:
        """Combine per-source values into one, over the finite ones.

        Parameters
        ----------
        v : array_like
            The per-source values.

        Returns
        -------
        agg : float
            Their median, or their mean with ``aggregate="mean"``, NaN when no value is finite.
        """
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan")
        if self.aggregate == "mean":
            return float(np.mean(v))
        return float(np.median(v))

    def _fmkw(self, fm, fm_at=None) -> Dict:
        """Return the forward-model keywords for the metric, only those it asked for.

        Every :class:`Metric` without ``needs_fm`` keeps its 4-argument signature.

        Parameters
        ----------
        fm : object or None
            The KLIP-FM pass of the injected sources.
        fm_at : object or None, default None
            Forward-model passes at the noise apertures (:class:`klip_tpe.fmmf.FMTemplates`), passed
            only to a metric that takes them.

        Returns
        -------
        kwargs : dict
            Empty, ``{"fm": fm}`` or ``{"fm": fm, "fm_at": fm_at}``.
        """
        if not self.metric.needs_fm:
            return {}
        if fm_at is not None and _takes(self.metric, "fm_at"):
            return {"fm": fm, "fm_at": fm_at}
        return {"fm": fm}

    def _values(self, img, img_clean, rho, th, kw) -> Tuple[np.ndarray, Optional[List[Optional[str]]]]:
        """Return the metric's per-source values, and each NaN's reason when the metric can give one.

        Parameters
        ----------
        img : ndarray
            The image scored.
        img_clean : ndarray or None
            The clean image passed to the metric.
        rho : sequence of float
            Separation of each source, in arcsec.
        th : sequence of float
            Angle of each source, in degrees.
        kw : dict
            More keywords for the metric, joined by ``sidelobes`` when the metric takes it.

        Returns
        -------
        values : ndarray
            The per-source values.
        why : list of str or None
            Each source's reason for a NaN, or None when the metric gives none.
        """
        if self.sidelobes is not None and _takes(self.metric, "sidelobes"):
            kw = dict(kw, sidelobes=self.sidelobes)
        if not _takes_why(self.metric):
            return self.metric.per_source(img, img_clean, rho, th, **kw), None
        why: List[Optional[str]] = []
        v = self.metric.per_source(img, img_clean, rho, th, why=why, **kw)
        return v, (why if len(why) == len(rho) else None)

    def score_raw(self, img_inj, sources: Sequence[Source], img_clean=None, fm=None, fm_at=None) -> ScoreResult:
        """Score a draw on the injected image alone, as validation does.

        Parameters
        ----------
        img_inj : ndarray, shape (ny, nx)
            The reduction with the sources injected.
        sources : sequence of Source
            The injected sources.
        img_clean : ndarray or None, default None
            The clean reduction, for a metric that uses it.
        fm : object or None, default None
            The KLIP-FM pass, for a metric with ``needs_fm``.
        fm_at : object or None, default None
            Forward-model passes at the noise apertures, for a metric that takes them.

        Returns
        -------
        result : ScoreResult
            The score, with ``per_source`` equal to ``raw_per_source``.
        """
        rho = [s.rho for s in sources]
        th = [s.theta for s in sources]
        raw, wr = self._values(img_inj, img_clean, rho, th, self._fmkw(fm, fm_at))
        agg = self._agg(raw)
        return ScoreResult(agg, raw, raw, None, why=None if np.isfinite(agg) else why_failed(raw, wr))

    def score_search(self, img_inj, sources: Sequence[Source], img_clean=None, fm=None, fm_at=None) -> ScoreResult:
        """Score a draw as the search does, with the clean term subtracted.

        The clean term is filtered with the same kernels as the injected image, since the forward-model
        template belongs to the configuration, not to the injection.

        Parameters
        ----------
        img_inj : ndarray, shape (ny, nx)
            The reduction with the sources injected.
        sources : sequence of Source
            The injected sources.
        img_clean : ndarray or None, default None
            The clean reduction, needed when the clean term applies.
        fm : object or None, default None
            The KLIP-FM pass, for a metric with ``needs_fm``.
        fm_at : object or None, default None
            Forward-model passes at the noise apertures, for a metric that takes them.

        Returns
        -------
        result : ScoreResult
            The score.

        Raises
        ------
        ValueError
            If the clean term applies and ``img_clean`` is None.
        """
        rho = [s.rho for s in sources]
        th = [s.theta for s in sources]
        kw = self._fmkw(fm, fm_at)
        raw, wr = self._values(img_inj, img_clean, rho, th, kw)
        if not self.effective_clean_subtract:
            # plain metric, or a difference metric that already used the clean image:
            # the search score IS the raw score (no separate clean term)
            agg = self._agg(raw)
            return ScoreResult(agg, raw, raw, None, why=None if np.isfinite(agg) else why_failed(raw, wr))
        if img_clean is None:
            raise ValueError("clean_subtract objective needs the clean image")
        # the FM template is a property of the configuration, not of the injection, so the
        # clean term is filtered with exactly the same kernels
        with ring_role("clean"):                   # RingRecorder: these rings are the clean image's
            cl, wc = self._values(img_clean, None, rho, th, kw)
        corr = raw - clip0(cl)
        agg = self._agg(corr)
        return ScoreResult(agg, corr, raw, cl, why=None if np.isfinite(agg) else why_failed(raw, wr, cl, wc))

    def describe(self) -> Dict:
        """Return the objective's settings, for a run's record.

        Returns
        -------
        info : dict
            The metric's settings, ``clean_subtract``, ``effective_clean_subtract``, ``aggregate`` and
            the sidelobe rule when the metric takes it.
        """
        out = {"metric": self.metric.describe(), "clean_subtract": self.clean_subtract,
               "effective_clean_subtract": self.effective_clean_subtract, "aggregate": self.aggregate}
        if self.sidelobes is not None and _takes(self.metric, "sidelobes"):
            out["sidelobes"] = self.sidelobes.describe()
        return out
