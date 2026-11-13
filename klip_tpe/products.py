"""Products of an annulus and of a run: noise profiles, contrast curves, the stitch across annuli and S/N maps.

The noise profiles, the injection-calibrated 5-sigma contrast curves, the sensitivity-weighted
stitch and the matched-filter S/N maps succeed the IDL predecessor's ``near2m_noiseprof``,
``near2m_kfit``, contrast-curve block, final stitch and ``near2m_snrmap``.  The KLIP-FM cross-check
curve succeeds its ``near2_fmresp`` and FM curve.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from . import metrics as _metrics
from .metrics import Source, gaussian_kernel, radprof, source_xy, star_center

__all__ = ["noise_profile", "throughput_fit", "contrast_curve", "stitch_annuli", "snr_map", "fm_responses",
           "fm_response", "fm_test_sources", "fm_contrast_curve"]


def noise_profile(img: np.ndarray, fwhm: float, rmin: float, rmax: float,
                  kernel: Optional[np.ndarray] = None, known: Sequence[Tuple[float, float]] = (),
                  pxscale: Optional[float] = None, excl_fwhm: float = 1.5,
                  pixel_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return the small-sample-corrected 1-sigma noise at each integer radius.

    Each ring holds ``nap = floor(2 pi r / fwhm)`` apertures, and its noise is
    ``std(ddof=1) sqrt(1 + 1/n)`` over the ``n`` apertures used.  The default aperture is a top-hat sum
    of radius ``fwhm / 2``, as the IDL predecessor's ``near2m_noiseprof`` takes, and ``kernel``
    measures the matched-filter noise instead.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.
    fwhm : float
        FWHM of the PSF, in pixels.
    rmin : float
        Smallest radius, in pixels, rounded up.
    rmax : float
        Largest radius, in pixels, rounded down.
    kernel : ndarray or None, default None
        The aperture's kernel.  None takes the top-hat.
    known : sequence of tuple of 2 floats, default ()
        Known companions, as ``(rho_arcsec, pa_deg)``, whose apertures within ``excl_fwhm`` FWHM are left
        out, so a detected planet does not raise its own contrast limit.  They need ``pxscale``.
    pxscale : float or None, default None
        Plate scale, in arcsec per pixel.
    excl_fwhm : float, default 1.5
        Radius of the keep-out around each companion, in FWHM.
    pixel_mask : ndarray of bool or None, default None
        Pixels no aperture is placed on, True to exclude, such as the run's disk mask.

    Returns
    -------
    sigma : ndarray
        The noise at each radius, NaN where the ring holds fewer than 4 apertures or fewer than 3 are
        used.
    radii : ndarray
        The radii, in pixels.
    """
    img = np.asarray(img, float)
    cx, cy = star_center(img.shape)
    if kernel is None:
        rad = fwhm / 2.0
        kd = int(2 * np.ceil(rad) + 1)
        kc = (kd - 1) / 2.0
        yy, xx = np.mgrid[0:kd, 0:kd]
        kernel = ((xx - kc) ** 2 + (yy - kc) ** 2 <= rad * rad).astype(float)
    A = ndimage.correlate(np.where(np.isfinite(img), img, 0.0), kernel, mode="nearest")
    fin = np.isfinite(img)
    if pixel_mask is not None and np.shape(pixel_mask) == img.shape:
        fin = fin & ~np.asarray(pixel_mask, bool)
    kn = []
    if len(known) and pxscale:
        for rho, pa in known:                       # (arcsec, deg east of north) -> pixels
            rr = float(rho) / float(pxscale)
            th = np.deg2rad(float(pa))
            kn.append((cx - rr * np.sin(th), cy + rr * np.cos(th)))
    radii = np.arange(int(np.ceil(rmin)), int(np.floor(rmax)) + 1)
    sig = np.full(radii.size, np.nan)
    for i, r in enumerate(radii):
        nap = int(np.floor(2 * np.pi * r / fwhm))
        if nap < 4:
            continue
        vals = []
        for p in range(nap):
            th = p * 2 * np.pi / nap
            x, y = int(round(cx + r * np.cos(th))), int(round(cy + r * np.sin(th)))
            if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0] and fin[y, x]):
                continue
            if any(np.hypot(x - kx, y - ky) < excl_fwhm * fwhm for kx, ky in kn):
                continue                            # aperture on a known companion
            vals.append(A[y, x])
        if len(vals) >= 3:
            sig[i] = np.std(vals, ddof=1) * np.sqrt(1.0 + 1.0 / len(vals))
    return sig, radii.astype(float)


def throughput_fit(r_s: np.ndarray, k_s: np.ndarray, r_eval: np.ndarray, order: int = 2) -> np.ndarray:
    """Fit ``log K`` against separation with a robust polynomial, and evaluate it.

    The order drops until there are enough points, and never exceeds the number of distinct radii less
    one, so samples all at one radius (``--inject-rho``) take order 0, the mean of ``log K``, rather
    than a fit numpy warns is rank-deficient.  The residuals are clipped at 3 sigma once, and the
    evaluation is clamped to the sampled range of radii.

    Parameters
    ----------
    r_s : ndarray
        Separation of each sample.
    k_s : ndarray
        Throughput of each sample, positive.
    r_eval : ndarray
        Separations to evaluate at, in the units of ``r_s``.
    order : int, default 2
        Order of the polynomial at most.

    Returns
    -------
    k : ndarray
        The fitted throughput at ``r_eval``, the samples' median for fewer than 3 samples, NaN for none.

    Notes
    -----
    It succeeds the IDL predecessor's ``near2m_kfit``.  The cap on the order dates from NEAR2,
    2026-11-02.
    """
    r_s, k_s = np.asarray(r_s, float), np.asarray(k_s, float)
    g = np.isfinite(r_s) & np.isfinite(k_s) & (k_s > 0)
    r_s, k_s = r_s[g], k_s[g]
    r_eval = np.asarray(r_eval, float)
    if r_s.size == 0:
        return np.full(r_eval.shape, np.nan)
    if r_s.size < 3:
        return np.full(r_eval.shape, float(np.median(k_s)))

    def _distinct(r):
        return int(np.unique(np.round(r, 9)).size)

    lk = np.log(k_s)
    o = int(max(min(order, r_s.size - 1, _distinct(r_s) - 1), 0))
    coef = np.polyfit(r_s, lk, o)
    res = lk - np.polyval(coef, r_s)
    sd = np.std(res)
    if sd > 0 and r_s.size > o + 2:
        keep = np.abs(res) <= 3 * sd
        if keep.sum() > o + 1:
            o2 = int(max(min(o, _distinct(r_s[keep]) - 1), 0))
            coef = np.polyfit(r_s[keep], lk[keep], o2)
    re = np.clip(r_eval, r_s.min(), r_s.max())
    return np.exp(np.polyval(coef, re))


def contrast_curve(clean_img: np.ndarray, sample_r_as: Sequence[float], sample_snr: Sequence[float],
                   contrast: float, fwhm: float, pxscale: float, rin_px: float, rout_px: float,
                   throughput_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                   kfit_order: int = 2, nsigma: float = 5.0,
                   known: Sequence[Tuple[float, float]] = (),
                   pixel_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Return the injection-calibrated ``nsigma`` contrast curve of an annulus.

    Each validation sample measures the recovered throughput ``K = SNR_inj sigma(r) / contrast``, with
    ``sigma`` the :func:`noise_profile` of the clean image.  The coronagraph's transmission ``T(r)`` is
    factored out before a smooth fit (:func:`throughput_fit`) and put back, and the curve is
    ``nsigma sigma(r) / K_fit(r)``.

    Parameters
    ----------
    clean_img : ndarray, shape (ny, nx)
        The clean reduction.
    sample_r_as : sequence of float
        Separation of each sample, in arcsec.
    sample_snr : sequence of float
        S/N of each sample.
    contrast : float
        Contrast the samples were injected at.
    fwhm : float
        FWHM of the PSF, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    rin_px : float
        Inner edge of the annulus, in pixels, raised to one FWHM.
    rout_px : float
        Outer edge of the annulus, in pixels.
    throughput_fn : callable or None, default None
        The coronagraph's transmission, ``T(r_arcsec)``, floored at 1e-3 at the samples.  None takes 1.
    kfit_order : int, default 2
        Order of the throughput fit at most.
    nsigma : float, default 5.0
        Number of sigma of the curve.
    known : sequence of tuple of 2 floats, default ()
        Known companions, as ``(rho_arcsec, pa_deg)``, left out of the noise rings.
    pixel_mask : ndarray of bool or None, default None
        Pixels left out of the noise rings, True to exclude.

    Returns
    -------
    curve : dict of str to ndarray
        ``r_as`` (the radii, in arcsec), ``sigma`` (the noise profile), ``curve`` (NaN for fewer than 3
        samples), ``sample_r`` and ``sample_c`` (each sample's ``nsigma contrast / SNR``) and, with 3
        samples or more, ``K_samples``.
    """
    r_s = np.asarray(sample_r_as, float)
    s_s = np.asarray(sample_snr, float)
    g = np.isfinite(r_s) & np.isfinite(s_s) & (s_s > 0)
    r_s, s_s = r_s[g], s_s[g]
    sig, rprof = noise_profile(clean_img, fwhm, max(rin_px, fwhm), rout_px, known=known, pxscale=pxscale,
                               pixel_mask=pixel_mask)
    r_as = rprof * pxscale
    out = {"r_as": r_as, "sigma": sig, "sample_r": r_s, "sample_c": nsigma * contrast / np.where(s_s > 0, s_s, np.nan)}
    if r_s.size < 3:
        out["curve"] = np.full(r_as.shape, np.nan)
        return out
    ok = np.isfinite(sig)
    sig_at = np.interp(r_s / pxscale, rprof[ok], sig[ok]) if ok.sum() >= 2 else np.full(r_s.shape, np.nan)
    Ks = s_s * sig_at / contrast
    T_s = np.ones_like(r_s) if throughput_fn is None else np.maximum(np.asarray(throughput_fn(r_s), float), 1e-3)
    Kalg = Ks / T_s
    T_e = np.ones_like(r_as) if throughput_fn is None else np.asarray(throughput_fn(r_as), float)
    Kfit = throughput_fit(r_s, Kalg, r_as, kfit_order) * T_e
    out["curve"] = nsigma * sig / np.maximum(Kfit, 1e-12)
    out["K_samples"] = Ks
    return out


def stitch_annuli(images: Sequence[np.ndarray], edges: Sequence[float], fwhm: float,
                  legacy: bool = False, pad: float = 2.0, pad_ends: bool = False,
                  pixel_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Stitch per-annulus images, each weighted by its sensitivity.

    Annulus ``ia``'s weight is ``1 / median(sigma(r))^2`` over its interior ``[in + fwhm, out - fwhm]``,
    normalized to the largest, and its tile is ``edge_ia - pad <= r <= edge_ia+1 + pad``.  Where tiles
    overlap the pixels are weighted means.

    Parameters
    ----------
    images : sequence of ndarray
        One image per annulus, ``(ny, nx)``.
    edges : sequence of float
        The annuli's edges, ``nann + 1`` of them, in pixels.
    fwhm : float
        FWHM of the PSF, in pixels.
    legacy : bool, default False
        Whether to weigh the annuli equally.
    pad : float, default 2.0
        Pad of the edges, in pixels.
    pad_ends : bool, default False
        Whether to pad the first inner edge and the last outer edge too, as the IDL predecessor does.
    pixel_mask : ndarray of bool or None, default None
        Pixels left out of the noise behind the weights, True to exclude.

    Returns
    -------
    stitch : ndarray, shape (ny, nx)
        The stitched image.
    weights : ndarray, shape (nann,)
        Each annulus' weight.

    Raises
    ------
    ValueError
        If there are no images.
    """
    n = len(images)
    if n == 0:
        raise ValueError("no images")
    ny, nx = images[0].shape
    cx, cy = star_center((ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy)
    w = np.ones(n)
    if not legacy and n > 1:
        for ia, img in enumerate(images):
            sig, rp = noise_profile(img, fwhm, edges[ia], edges[ia + 1], pixel_mask=pixel_mask)
            gi = (rp >= edges[ia] + fwhm) & (rp <= edges[ia + 1] - fwhm) & np.isfinite(sig) & (sig > 0)
            if gi.sum() == 0:
                gi = np.isfinite(sig) & (sig > 0)
            sg = float(np.median(sig[gi])) if gi.any() else 1.0
            w[ia] = 1.0 / sg ** 2 if sg > 0 else 1.0
        w /= w.max()
    num = np.zeros((ny, nx))
    den = np.zeros((ny, nx))
    for ia, img in enumerate(images):
        pin = pad if (ia > 0 or pad_ends) else 0.0
        pout = pad if (ia < n - 1 or pad_ends) else 0.0
        tile = (rr >= edges[ia] - pin) & (rr <= edges[ia + 1] + pout) & np.isfinite(img)
        num[tile] += w[ia] * img[tile]
        den[tile] += w[ia]
    out = num / np.where(den > 0, den, np.nan)
    return out, w


#: the pixels a radius' noise band takes before the S/N map widens it past the pixel mask
SNR_BAND_MIN = 20


def snr_map(img: np.ndarray, fwhm: float, kernel: Optional[np.ndarray] = None,
            exclude_xy: Sequence[Tuple[float, float]] = (), excl_fwhm: float = 1.5,
            rmin: float = 1.0, excl_arcs: Sequence[Tuple[float, float, float, float]] = (),
            pxscale: Optional[float] = None, angle_convention: str = "pa",
            pixel_mask: Optional[np.ndarray] = None, band_min: int = SNR_BAND_MIN,
            info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Make a matched-filter S/N map, with a robust noise per radius and the Mawet small-sample penalty.

    At each radius the S/N is ``(A - mean) / (sigma sqrt(1 + 1/nap))``, with ``A`` the matched-filter
    map, ``nap = max(floor(2 pi r / fwhm), 3)`` and ``sigma`` the larger of the standard deviation and
    the MAD-based sigma of the radius' 1 px band.  It is a quantity for display and diagnosis, not the
    search's score, whose noise is measured otherwise.

    Where the pixel mask leaves a radius' band fewer than ``band_min`` pixels, the band widens a pixel
    each side at a time until it holds that many, up to 3 FWHM each side, and the pixels at that radius
    take their S/N from the wider band.  A band the mask does not touch is never widened.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.
    fwhm : float
        FWHM of the PSF, in pixels.
    kernel : ndarray or None, default None
        The matched filter.  None takes the FWHM Gaussian.
    exclude_xy : sequence of tuple of 2 floats, default ()
        The ``(x, y)`` pixels of sources left out of every band's noise.
    excl_fwhm : float, default 1.5
        Radius of their keep-outs, in FWHM.
    rmin : float, default 1.0
        Smallest radius mapped, in pixels.
    excl_arcs : sequence of tuple of 4 floats, default ()
        Arcs ``(rho_arcsec, theta_deg, half_width_deg, radial_fwhm)`` whose pixels are left out of every
        band's noise, such as the known companions' ADI sidelobes under the run's sidelobe rule.
    pxscale : float or None, default None
        Plate scale, in arcsec per pixel, without which ``excl_arcs`` is ignored.
    angle_convention : {"pa", "math"}, default "pa"
        How the arcs' angles are read.
    pixel_mask : ndarray of bool or None, default None
        Pixels left out of every band's noise, True to exclude, as the search's rings leave them out.
        The map still has a value there.
    band_min : int, default SNR_BAND_MIN
        Fewest pixels of a band before it widens past the mask.
    info : dict or None, default None
        Dict that gets ``widened``, each widened radius in pixels with its band's half-width.

    Returns
    -------
    snr : ndarray, shape (ny, nx)
        The S/N map, NaN where a band holds fewer than 5 pixels or has no scatter.

    Notes
    -----
    It succeeds the IDL predecessor's ``near2m_snrmap``.  The widened band dates from 2026-11-01 (Kevin
    Wagner), after a mask over whole rings left the map blank there.  A
    :class:`~klip_tpe.metrics.RingRecorder` that is on records the bands' pixels.
    """
    img = np.asarray(img, float)
    ny, nx = img.shape
    cx, cy = star_center(img.shape)
    if kernel is None:
        kernel = gaussian_kernel(fwhm)
    fin = np.isfinite(img)
    A = ndimage.correlate(np.where(fin, img, 0.0), kernel, mode="nearest")
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy)
    keep = fin.copy()
    for ex, ey in exclude_xy:
        keep &= np.hypot(xx - ex, yy - ey) >= excl_fwhm * fwhm
    if len(excl_arcs) and pxscale is not None:
        from .metrics import _arc_regions, _in_arcs
        keep &= ~_in_arcs(xx, yy, _arc_regions(excl_arcs, pxscale, fwhm, cx, cy, angle_convention), cx, cy)
    keep_open = keep.copy()                       # the noise pixels without the mask
    masked = pixel_mask is not None and np.shape(pixel_mask) == img.shape
    if masked:
        keep &= ~np.asarray(pixel_mask, bool)
    if _metrics._RING_RECORDER is not None:
        _metrics._ring_emit({"fn": "snr_map", "shape": tuple(img.shape), "center": (float(cx), float(cy)),
                             "fwhm": float(fwhm), "exr": float(excl_fwhm * fwhm), "keep": keep.copy(),
                             "finite": fin.copy(), "band_px": 0.5, "ring": [],
                             "sources_xy": [(float(a), float(b)) for a, b in exclude_xy], "known_xy": [],
                             "arcs": [tuple(float(v) for v in a) for a in excl_arcs], "noise": "band",
                             "why": None, "x": float("nan"), "y": float("nan"), "r": float("nan"), "nclean": 0})
    out = np.full(img.shape, np.nan)
    rmax = int(np.floor(rr[fin].max())) if fin.any() else 0
    hw_max = max(3.0 * float(fwhm), 1.5)
    widened: Dict[int, float] = {}
    for r in range(max(int(rmin), 1), rmax + 1):
        dist = np.abs(rr - r)
        band = (dist <= 0.5) & keep
        hw = 0.5
        nb = int(band.sum())
        if masked and nb < band_min and int(((dist <= 0.5) & keep_open).sum()) > nb:
            # the mask starves this radius: its noise from a band widened past the mask
            while nb < band_min and hw + 1.0 <= hw_max:
                hw += 1.0
                band = (dist <= hw) & keep
                nb = int(band.sum())
        vals = A[band]
        if vals.size < 5:
            continue
        sd = np.std(vals, ddof=1)
        mad = np.median(np.abs(vals - np.median(vals))) * 1.4826
        sd = max(sd, mad)
        if sd <= 0:
            continue
        if hw > 0.5:
            widened[r] = hw
        nap = max(int(np.floor(2 * np.pi * r / fwhm)), 3)
        ring = (dist <= 0.5) & fin
        out[ring] = (A[ring] - np.mean(vals)) / (sd * np.sqrt(1.0 + 1.0 / nap))
    if info is not None:
        info["widened"] = widened
    return out


def snr_widened_note(info: Optional[Dict[str, Any]], pxscale: Optional[float] = None, short: bool = False) -> str:
    """Describe the radii whose S/N came from a band widened past the pixel mask, as one line or a few words.

    Parameters
    ----------
    info : dict or None
        The ``info`` of :func:`snr_map`.
    pxscale : float or None, default None
        Plate scale, in arcsec per pixel, to give the radii in arcsec.  None gives them in pixels.
    short : bool, default False
        Whether to give a few words, for a panel.

    Returns
    -------
    note : str
        The note, empty when no band was widened.
    """
    w = (info or {}).get("widened") or {}
    if not w:
        return ""
    rs = sorted(int(r) for r in w)
    spans, a = [], rs[0]
    for b0, b1 in zip(rs, rs[1:] + [None]):
        if b1 is None or b1 != b0 + 1:
            spans.append((a, b0))
            a = b1
    if pxscale:
        txt = ", ".join(f'{x * pxscale:.2f}"' if x == y else f'{x * pxscale:.2f}-{y * pxscale:.2f}"' for x, y in spans)
    else:
        txt = ", ".join(f"{x} px" if x == y else f"{x}-{y} px" for x, y in spans)
    if short:
        return f"wider noise band at {txt} (mask)"
    return f"noise band widened past the mask at {txt} (to +/-{max(w.values()):.1f} px)"


def snr_widened_header(info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return FITS keywords for the radii :func:`snr_map` widened.

    Parameters
    ----------
    info : dict or None
        The ``info`` of :func:`snr_map`.

    Returns
    -------
    keywords : dict
        ``SNRWIDEN`` (their count), ``SNRWIDR`` (their spans, in pixels) and ``SNRWIDHW`` (the widest
        half-width, in pixels), empty when no band was widened.
    """
    w = (info or {}).get("widened") or {}
    if not w:
        return {}
    rs = sorted(int(r) for r in w)
    spans, a = [], rs[0]
    for b0, b1 in zip(rs, rs[1:] + [None]):
        if b1 is None or b1 != b0 + 1:
            spans.append(f"{a}" if a == b0 else f"{a}-{b0}")
            a = b1
    return {"SNRWIDEN": len(rs), "SNRWIDR": ",".join(spans)[:68], "SNRWIDHW": float(max(w.values()))}


# ----------------------------------------------------------------------------
# KLIP-FM cross-check curve (the IDL predecessor's optimize_near_2_tpe.pro: near2_fmresp L398-441,
# curve L7566-7677)
# ----------------------------------------------------------------------------
def fm_response(img: np.ndarray, rho, theta, pxscale: float, fwhm: float,
                kernel_fn: Optional[Callable[[float], Optional[np.ndarray]]] = None,
                search_px: float = 1.5, angle_convention: str = "pa") -> np.ndarray:
    """Return the matched-filter peak of a noiseless KLIP-FM image at each source.

    The peak is the maximum of the matched-filter map within ``search_px`` of the source's position,
    with no ring or background subtracted.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The KLIP-FM image.
    rho : float or array_like
        Separation of each source, in arcsec.
    theta : float or array_like
        Angle of each source, in degrees.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    kernel_fn : callable or None, default None
        Function ``rho_arcsec -> kernel`` giving each source the filter at its own separation, since the
        measured library PSF broadens toward small separations.  None, or a None kernel, takes the FWHM
        Gaussian.
    search_px : float, default 1.5
        Radius of the peak search, in pixels.
    angle_convention : {"pa", "math"}, default "pa"
        How ``theta`` is read.

    Returns
    -------
    resp : ndarray
        The peak at each source, NaN where no pixel of the search is finite.

    Notes
    -----
    It succeeds the IDL predecessor's ``near2_fmresp``.
    """
    img = np.asarray(img, float)
    ny, nx = img.shape
    cx, cy = star_center(img.shape)
    rho = np.atleast_1d(np.asarray(rho, float))
    theta = np.atleast_1d(np.asarray(theta, float))
    imz = np.where(np.isfinite(img), img, 0.0)
    xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
    isr = int(np.ceil(search_px))
    resp = np.full(rho.size, np.nan)
    gk = gaussian_kernel(fwhm)
    conv_cache: Dict[bytes, np.ndarray] = {}
    for s in range(rho.size):
        kern = None
        if kernel_fn is not None and np.isfinite(rho[s]):
            kern = kernel_fn(float(rho[s]))
        if kern is None:
            kern = gk
        kern = np.asarray(kern, float)
        key = kern.tobytes()
        if key not in conv_cache:
            conv_cache[key] = ndimage.correlate(imz, np.asarray(kern, float), mode="nearest")
        A = conv_cache[key]
        px0, py0 = int(round(xs[s])), int(round(ys[s]))
        pkv, bx, by = -np.inf, px0, py0
        for dy in range(-isr, isr + 1):
            for dx in range(-isr, isr + 1):
                if dx * dx + dy * dy > search_px * search_px:
                    continue
                qx, qy = px0 + dx, py0 + dy
                if 0 <= qx < nx and 0 <= qy < ny and np.isfinite(img[qy, qx]) and A[qy, qx] > pkv:
                    pkv, bx, by = A[qy, qx], qx, qy
        if np.isfinite(pkv):
            resp[s] = A[by, bx]
    return resp


def fm_responses(fm_image: np.ndarray, sources: Sequence[Source], pxscale: float, fwhm: float,
                 kernel_fn: Optional[Callable[[float], Optional[np.ndarray]]] = None,
                 search_px: float = 1.5, angle_convention: str = "pa", flatten: bool = False) -> np.ndarray:
    """Return :func:`fm_response` of a KLIP-FM image at each source, as :func:`fm_contrast_curve` measures them.

    Parameters
    ----------
    fm_image : ndarray, shape (ny, nx)
        The KLIP-FM image.
    sources : sequence of Source
        The sources.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    kernel_fn : callable or None, default None
        Function ``rho_arcsec -> kernel``.
    search_px : float, default 1.5
        Radius of the peak search, in pixels.
    angle_convention : {"pa", "math"}, default "pa"
        How the sources' angles are read.
    flatten : bool, default False
        Whether to subtract the image's azimuthal mean profile first (:func:`~klip_tpe.metrics.radprof`).

    Returns
    -------
    resp : ndarray
        The peak at each source.
    """
    rho = np.array([s.rho for s in sources], float)
    th = np.array([s.theta for s in sources], float)
    img = radprof(fm_image) if flatten else np.asarray(fm_image, float)
    return fm_response(img, rho, th, pxscale, fwhm, kernel_fn=kernel_fn, search_px=search_px,
                       angle_convention=angle_convention)


def fm_test_sources(rin_px: float, rout_px: float, pxscale: float, contrast: float) -> List[Source]:
    """Return the KLIP-FM cross-check's test sources, on a golden-angle spiral through the annulus.

    There are ``nfm = clip(floor((rout - 1 - rin) / 1.5) + 1, 5, 24)`` radii from ``rin`` to
    ``rout - 1`` px, at angles ``theta_i = (137.508 i) mod 360`` degrees, all at ``contrast``.

    Parameters
    ----------
    rin_px : float
        Inner radius, in pixels, already ``max(inner edge, fwhm)``.
    rout_px : float
        Outer edge of the annulus, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    contrast : float
        Contrast of the sources.

    Returns
    -------
    sources : list of Source
        The test sources.

    Notes
    -----
    They are the IDL predecessor's (L7574-7576).
    """
    rin, rout = float(rin_px), float(rout_px)
    nfm = int(min(24, max(5, int(np.floor((rout - 1.0 - rin) / 1.5)) + 1)))
    i = np.arange(nfm, dtype=float)
    rad = rin + (rout - 1.0 - rin) * i / max(nfm - 1, 1)
    th = (i * 137.508) % 360.0
    return [Source(float(r * pxscale), float(t), float(contrast)) for r, t in zip(rad, th)]


def fm_contrast_curve(fm_image: np.ndarray, clean_image: np.ndarray, contrast: float, fwhm: float,
                      pxscale: float, rin_px: float, rout_px: float, sources: Sequence[Source],
                      kernel_fn: Optional[Callable[[float], Optional[np.ndarray]]] = None,
                      nsigma: float = 5.0, search_px: float = 1.5,
                      angle_convention: str = "pa", flatten: bool = False,
                      known: Sequence[Tuple[float, float]] = (),
                      resp: Optional[np.ndarray] = None,
                      pixel_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Return the KLIP-FM ``nsigma`` contrast curve of an annulus.

    The response of each test source gives ``K_fm = resp / contrast``, and the curve is
    ``nsigma sigma_mf(r) / K_fm`` at the test radii with ``K_fm > 0``, at least 3 of them.  ``sigma_mf``
    is the :func:`noise_profile` of the clean image measured with the unit-sum matched filter,
    ``kernel_fn`` at the median valid test separation, else the FWHM Gaussian.

    Parameters
    ----------
    fm_image : ndarray, shape (ny, nx)
        The KLIP-FM image of the test sources.
    clean_image : ndarray, shape (ny, nx)
        The clean reduction.
    contrast : float
        Contrast of the test sources.
    fwhm : float
        FWHM of the PSF, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    rin_px : float
        Inner radius of the noise profile, in pixels.
    rout_px : float
        Outer radius of the noise profile, in pixels.
    sources : sequence of Source
        The test sources (:func:`fm_test_sources`).
    kernel_fn : callable or None, default None
        Function ``rho_arcsec -> kernel``.
    nsigma : float, default 5.0
        Number of sigma of the curve.
    search_px : float, default 1.5
        Radius of the peak search, in pixels.
    angle_convention : {"pa", "math"}, default "pa"
        How the sources' angles are read.
    flatten : bool, default False
        Whether to subtract the azimuthal mean profile (:func:`~klip_tpe.metrics.radprof`) from both
        images first, as the IDL predecessor does.  It has to be both or neither, since ``K_fm`` is
        measured on the model and divided into a noise measured on the data.
    known : sequence of tuple of 2 floats, default ()
        Known companions, as ``(rho_arcsec, pa_deg)``, left out of the noise rings.
    resp : ndarray or None, default None
        The responses already measured, one per source (:func:`fm_responses`), such as each on a
        forward-model pass of its own (``Runner._fm_pass(one_at_a_time=True)``), so that no source is
        read through another's self-subtraction lobes.  None measures them on ``fm_image``.
    pixel_mask : ndarray of bool or None, default None
        Pixels left out of the noise rings, True to exclude.

    Returns
    -------
    curve : dict of str to ndarray
        ``r_as`` (the valid test radii, in arcsec), ``curve``, ``K_fm`` and ``resp`` (all the sources,
        NaN where not valid), ``r_test_as`` (all the test radii) and, with 3 valid sources or more,
        ``sigma_mf`` and ``r_sigma_px`` (the noise profile).  The curve is NaN for fewer than 3 valid
        sources or fewer than 2 finite radii of the profile.

    Raises
    ------
    ValueError
        If ``resp`` does not hold one response per source.

    Notes
    -----
    It follows A_main_procedure.md section 9.1 and the IDL predecessor's L7655-7677.
    """
    rho = np.array([s.rho for s in sources], float)
    _flat = radprof if flatten else (lambda a: np.asarray(a, float))
    if resp is None:
        resp = fm_responses(fm_image, sources, pxscale, fwhm, kernel_fn=kernel_fn, search_px=search_px,
                            angle_convention=angle_convention, flatten=flatten)
    else:
        resp = np.asarray(resp, float).ravel()
        if resp.size != rho.size:
            raise ValueError(f"{resp.size} responses for {rho.size} sources")
    K = resp / float(contrast)
    g = np.isfinite(K) & (K > 0)
    out: Dict[str, np.ndarray] = {"r_test_as": rho, "K_fm": K, "resp": resp,
                                  "r_as": rho[g], "curve": np.full(int(g.sum()), np.nan)}
    if g.sum() < 3:
        return out
    kern = None
    if kernel_fn is not None:
        kern = kernel_fn(float(np.median(rho[g])))
    if kern is None:
        kern = gaussian_kernel(fwhm)
    sig, rprof = noise_profile(_flat(clean_image), fwhm, rin_px, rout_px, kernel=np.asarray(kern, float),
                               known=known, pxscale=pxscale, pixel_mask=pixel_mask)
    out["sigma_mf"], out["r_sigma_px"] = sig, rprof
    ok = np.isfinite(sig)
    if ok.sum() < 2:
        return out
    sig_at = np.interp(rho[g] / pxscale, rprof[ok], sig[ok])
    out["curve"] = nsigma * sig_at / K[g]
    return out
