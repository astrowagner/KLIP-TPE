"""Per-annulus and run-level products: noise profiles, injection-calibrated
5-sigma contrast curves, the sensitivity-weighted stitch across annuli and
matched-filter S/N maps (ports of ``near2m_noiseprof``, ``near2m_kfit``, the
contrast-curve block, the final stitch and ``near2m_snrmap``)."""
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
    """Small-sample-corrected 1-sigma noise vs integer radius (px) from ``nap =
    floor(2 pi r / fwhm)`` apertures per ring.  Default aperture = tophat sum of
    radius ``fwhm/2`` (``near2m_noiseprof``); pass ``kernel`` to measure the
    matched-filter noise instead.  ``known`` = [(rho_arcsec, pa_deg), ...] real companions
    whose apertures (within ``excl_fwhm * fwhm``) are left out of the ring statistics, so a
    detected planet does not inflate its own contrast limit (needs ``pxscale``).
    ``pixel_mask`` (True = left out, the run's disk mask): no sample on it, as no ring of the
    run takes one there.  Returns ``(sigma, radii)``."""
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
    """Robust polynomial fit of ``log K`` vs separation (``near2m_kfit``): the order is
    reduced until there are enough points, 3-sigma clipped once, and the evaluation
    is clamped to the sampled radius range.  Fewer than 3 points -> median.  The order
    never exceeds the number of distinct radii less one, so samples all at one radius
    (``--inject-rho``) take order 0, the mean of ``log K``, rather than a fit numpy warns is
    rank-deficient (NEAR2, 2026-11-02)."""
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
    """Injection-calibrated ``nsigma`` contrast curve.

    ``K(r) = SNR_inj * sigma(r) / contrast`` measures the recovered throughput at
    each validation sample; the coronagraph transmission ``T(r)`` is factored out
    before a smooth fit and re-applied, and ``c5 = nsigma * sigma(r) / K_fit(r)``.
    ``known`` = real companions [(rho_arcsec, pa_deg), ...] excluded from the noise ring.
    Returns radii (arcsec), curve, the per-sample ``nsigma*contrast/SNR`` points and
    the noise profile.
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
    """Sensitivity-weighted stitch of per-annulus images.  Weight of annulus ``ia`` =
    ``1 / median(sigma(r))^2`` over its interior ``[in+fwhm, out-fwhm]`` (normalised
    to the max), tile mask ``edge_ia - pad <= r <= edge_ia+1 + pad``; overlaps are
    weighted means.  The first inner edge and the last outer edge are not padded unless
    ``pad_ends`` (the IDL pads them).  ``legacy=True`` uses equal weights.  ``pixel_mask``
    (True = left out, the run's disk mask) stays out of the noise behind the weights."""
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
    """Matched-filter S/N map with a per-radius robust (MAD-floored) sigma and the
    Mawet small-sample penalty (``near2m_snrmap``).  A display/diagnostic quantity
    -- NOT the search score (different noise convention).

    ``excl_arcs`` (with ``pxscale``): ``(rho_arcsec, theta_deg, half_width_deg,
    radial_fwhm)`` arcs whose pixels are left out of every ring's noise, as the known
    companions' ADI sidelobes under the run's sidelobe rule.  ``pixel_mask`` (True = left
    out, the run's disk mask): its pixels are left out of every ring's noise, as the search's
    rings leave them out.  The map still has a value there.

    Where the mask leaves a radius' 1 px band fewer than ``band_min`` pixels, the band widens
    a pixel each side at a time until it holds that many, up to 3 FWHM each side (Kevin
    Wagner, 2026-11-01: a mask over whole rings left the map blank there).  The pixels at that
    radius take their S/N from the wider band, and ``info`` (a dict), when given, gets
    ``widened``, each such radius (px) with the half-width its band took.  A band the mask
    does not touch is never widened."""
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
    """The radii whose S/N came from a band widened past the pixel mask (:func:`snr_map`'s
    ``info``), as one line, or a few words for a panel (``short``): ``""`` when there are none."""
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
    """FITS keywords for :func:`snr_map`'s widened radii: ``SNRWIDEN``, their count, and
    ``SNRWIDR``, the radii (px), ``SNRWIDHW`` the widest half-width (px).  Empty when none."""
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
# KLIP-FM cross-check curve (optimize_near_2_tpe.pro near2_fmresp L398-441, curve L7566-7677)
# ----------------------------------------------------------------------------
def fm_response(img: np.ndarray, rho, theta, pxscale: float, fwhm: float,
                kernel_fn: Optional[Callable[[float], Optional[np.ndarray]]] = None,
                search_px: float = 1.5, angle_convention: str = "pa") -> np.ndarray:
    """``near2_fmresp``: matched-filter PEAK of the (noiseless) KLIP-FM image at each
    source, located as the maximum within ``search_px`` of the nominal position.
    No ring / background subtraction.  ``kernel_fn(rho_as) -> kernel`` supplies a
    per-source kernel at each source's OWN separation (the measured library PSF
    broadens toward small separation); None / a None return -> FWHM Gaussian."""
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
    """:func:`fm_response` of ``fm_image`` at each of ``sources``, after :func:`radprof` with
    ``flatten`` (as :func:`fm_contrast_curve` measures them)."""
    rho = np.array([s.rho for s in sources], float)
    th = np.array([s.theta for s in sources], float)
    img = radprof(fm_image) if flatten else np.asarray(fm_image, float)
    return fm_response(img, rho, th, pxscale, fwhm, kernel_fn=kernel_fn, search_px=search_px,
                       angle_convention=angle_convention)


def fm_test_sources(rin_px: float, rout_px: float, pxscale: float, contrast: float) -> List[Source]:
    """The KLIP-FM cross-check test sources (L7574-7576): ``nfm = clip(floor((rout-1-rin)
    /1.5)+1, 5, 24)`` radii from ``rin`` to ``rout-1`` px on a golden-angle spiral
    ``theta_i = (i*137.508) mod 360`` deg, all at ``contrast``.  ``rin_px`` should
    already be ``max(annulus inner edge, fwhm)``."""
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
    """KLIP-FM ``nsigma`` contrast curve (A_main_procedure.md section 9.1, L7655-7677).

    ``resp = fm_response(fm_image, sources)``; ``K_fm = resp / contrast``;
    ``sigma_mf`` = :func:`noise_profile` of ``clean_image`` measured with the
    unit-sum matched-filter kernel (``kernel_fn`` at the median valid test
    separation, else the FWHM Gaussian); ``curve = nsigma * interp(sigma_mf)(r) /
    K_fm`` at the test radii with ``K_fm > 0`` (at least 3 needed, else NaN).

    ``flatten`` applies :func:`radprof` to BOTH images first -- the IDL behaviour, and
    off by default here.  It has to be both or neither: ``K_fm`` is a response measured
    on the model and divided into a noise measured on the data, so flattening one and
    not the other puts a bias straight into the contrast axis.

    ``known`` = real companions [(rho_arcsec, pa_deg), ...] left out of the noise rings.

    ``resp`` = the responses already measured, one per source (:func:`fm_responses`), in
    place of measuring them on ``fm_image``: each on a forward-model pass of its own, say
    (``Runner._fm_pass(one_at_a_time=True)``), so that no source is read through another's
    self-subtraction lobes.

    Returns ``r_as`` (valid test radii, arcsec), ``curve``, ``K_fm`` and ``resp`` (all
    sources, NaN where invalid), plus ``sigma_mf`` / ``r_sigma_px`` (the profile) and
    ``r_test_as`` (all test radii).
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
