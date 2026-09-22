"""Detection metrics and the search objective.

Image convention: 2-D ``numpy`` arrays indexed ``img[y, x]``; the star sits at
``cx = (nx-1)/2, cy = (ny-1)/2`` (between pixels for even sizes).  Source
positions are given as ``(rho_arcsec, theta_deg)`` with ``theta`` a position angle
East of North; the pixel position is ``x = cx + r cos(theta+90), y = cy + r sin(theta+90)``
(North up, East left).  A reducer with a different convention declares it (see
:class:`~klip_tpe.reducer.Reducer.angle_convention`) and the runner converts.

Ported from ``optimize_near_2_tpe.pro``: ``radprof``, ``near2m_declip``,
``near2m_mfkern``, ``near2_snrpk`` (:func:`mawet_peak_snr`), ``near2_snr_inj``
(:func:`injection_difference_snr`), ``near2m_clip0`` and the clean-subtraction rule
(:class:`Objective`).  The instrument variants (MWC/LMIRCam/HPBoo) differ from NEAR
only by option flags of :class:`MawetPeakSNR`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

__all__ = ["radprof", "declip", "gaussian_kernel", "source_xy", "mawet_peak_snr",
           "injection_difference_snr", "aperture_mawet_snr", "clip0", "MawetPeakSNR",
           "InjectionDifferenceSNR", "Objective", "Source", "nanmedian_even", "pa_wedge_mask"]


def pa_wedge_mask(shape: Tuple[int, int], bands: Sequence[Tuple[float, float]],
                  angle_convention: str = "pa",
                  r_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Boolean mask (True = exclude) of azimuthal wedges, for ``pixel_mask``.

    ``bands`` = ``[(centre_pa_deg, half_width_deg), ...]``.  Built for resolved emission
    that is known *a priori* to sit at a given position angle -- a debris disk seen close to
    edge-on -- so that its flux is kept out of the noise ring and the radial-band sigma.
    Pair it with :class:`~klip_tpe.positions.PositionSampler`'s ``forbidden_pa`` on the same
    bands, so the injections avoid what the noise estimate is no longer measuring.

    A word on when this is honest.  Removing ring apertures *because they are bright* biases
    the noise estimate downwards and inflates every S/N; the exclusion has to be justified
    before you look at the image.  A disk at a catalogued position angle qualifies, the
    brightest speckle group does not -- and on beta Pic the two are comparable in this
    annulus, so the cut is worth less than it looks (measured: the disk bands carry ~+0.2
    sigma of median excess over the rest of the ring, while removing ~22% of the apertures).

    ``r_range`` (pixels) limits the wedges radially, for a disk with a known extent.
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


@dataclass(frozen=True)
class Source:
    rho: float        # arcsec
    theta: float      # deg, PA east of north (metric convention)
    contrast: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float]:
        return (float(self.rho), float(self.theta), float(self.contrast))


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def star_center(shape) -> Tuple[float, float]:
    ny, nx = shape[-2], shape[-1]
    return (nx - 1) / 2.0, (ny - 1) / 2.0


def source_xy(rho_as, theta_deg, pxscale: float, cx: float, cy: float, angle_convention: str = "pa"):
    """Pixel coordinates of sources.  ``angle_convention='pa'``: PA East of North
    (x = cx + r cos(theta+90)); ``'math'``: angle measured from +x counter-clockwise."""
    rho = np.atleast_1d(np.asarray(rho_as, float))
    th = np.atleast_1d(np.asarray(theta_deg, float))
    phi = np.deg2rad(th + (90.0 if angle_convention == "pa" else 0.0))
    return cx + (rho / pxscale) * np.cos(phi), cy + (rho / pxscale) * np.sin(phi)


def nanmedian_even(v) -> float:
    """Median over finite values, mean of the two middle values for even counts
    (IDL ``median(/even)``); ``nan`` if nothing is finite."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def radprof(image: np.ndarray) -> np.ndarray:
    """Subtract the azimuthal (integer-radius-bin) NaN-mean profile.  Bins are
    computed about ``(nx/2, ny/2)`` -- deliberately the IDL ``radprof`` convention,
    which differs by half a pixel from the star centre.  Only the flattening is
    affected, never source positions."""
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
    """Replace isolated single-pixel spikes by their 3x3 median (``near2m_declip``).
    NaN-safe: only fully interior finite pixels are touched."""
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
    """Unit-sum Gaussian matched-filter kernel of the given FWHM (``near2m_mfkern``
    default branch): ``kd = 2*ceil(1.5*fwhm)+1`` pixels wide."""
    kd = int(2 * np.ceil(1.5 * fwhm) + 1) if size is None else int(size)
    kc = (kd - 1) / 2.0
    sg = fwhm / 2.3548
    yy, xx = np.mgrid[0:kd, 0:kd]
    k = np.exp(-0.5 * ((xx - kc) ** 2 + (yy - kc) ** 2) / sg ** 2)
    return k / k.sum()


def kernel_from_profile(stamp: np.ndarray, fwhm: float) -> np.ndarray:
    """Azimuthally averaged, unit-sum kernel built from a PSF stamp (the measured
    matched filter of ``near2m_mfkern`` when the injection library is active)."""
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


def _per_source_kernels(kernel, nsrc: int, fwhm: float) -> List[np.ndarray]:
    """Normalise the ``kernel`` argument of :func:`mawet_peak_snr` to one kernel per
    source, sharing the *same object* when they are all the same (so the convolution is
    done once)."""
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
    """Clamp finite negatives to 0, leave NaN untouched (``near2m_clip0``)."""
    x = np.asarray(x, float).copy()
    b = np.isfinite(x) & (x < 0)
    x[b] = 0.0
    return x


def _aperture_sum(img: np.ndarray, x: float, y: float, r: float) -> float:
    """Tophat aperture sum (pixel centres inside radius r); NaN if the aperture holds
    no finite pixel.  Pixels outside the image are ignored."""
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


# ----------------------------------------------------------------------------
# per-source detection S/N  (near2_snrpk)
# ----------------------------------------------------------------------------
def mawet_peak_snr(img: np.ndarray, rho, theta, pxscale: float, fwhm: float, *,
                   kernel=None, search_px: float = 1.5,
                   excl_fwhm: float = 1.5, known: Sequence[Tuple[float, float]] = (),
                   declip_nsig: Optional[float] = 6.0, band_fwhm: float = 0.6,
                   min_ring: int = 6, penalty: str = "geometric",
                   pixel_mask: Optional[np.ndarray] = None,
                   angle_convention: str = "pa", return_details: bool = False):
    """Matched-filter Mawet et al. (2014) detection S/N of each source.

    * optional single-pixel de-spiking, then convolution with ``kernel`` (default
      FWHM Gaussian);
    * the source is *located* as the matched-filter peak within ``search_px`` of
      the nominal position (absorbs sub-pixel recovery offsets);
    * noise from ``nap = floor(2 pi r / fwhm)`` apertures on the same ring, skipping
      apertures within ``excl_fwhm*fwhm`` of any injected or ``known`` source;
    * a radial band ``|r' - r| <= band_fwhm*fwhm`` gives a sigma FLOOR (and replaces
      the ring when it holds fewer than ``min_ring`` clean samples);
    * small-sample penalty ``sqrt(1 + 1/n)`` with ``n = max(nap, 3)`` (``penalty=
      'geometric'``, NEAR) or the retained clean-aperture count (``'retained'``,
      NIRCam variant).

    ``kernel`` is one 2-D matched filter for every source, or a **sequence of one
    kernel per source** (``None`` entries fall back to the FWHM Gaussian) when the
    filter is position-dependent -- the forward-modelled templates of
    :mod:`klip_tpe.fmmf`.  Each distinct kernel costs one convolution of the frame.

    Returns per-source S/N, **which may be negative** (peak below the ring mean) or
    NaN (inside 1 FWHM, starved ring).  ``pixel_mask`` (True = exclude) removes
    e.g. disk pixels from the noise samples.
    """
    img = np.asarray(img, float)
    ny, nx = img.shape
    cx, cy = star_center(img.shape)
    rho = np.atleast_1d(np.asarray(rho, float))
    theta = np.atleast_1d(np.asarray(theta, float))
    nsrc = rho.size
    exr = excl_fwhm * fwhm
    if declip_nsig is not None:
        img = declip(img, declip_nsig)
    imz = np.where(np.isfinite(img), img, 0.0)
    kerns = _per_source_kernels(kernel, nsrc, fwhm)
    _amaps: Dict[int, np.ndarray] = {}

    def amap(i: int) -> np.ndarray:
        k = kerns[i]
        if id(k) not in _amaps:
            _amaps[id(k)] = ndimage.convolve(imz, k, mode="nearest")
        return _amaps[id(k)]

    xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
    kx = ky = np.zeros(0)
    if len(known):
        kr = np.array([k[0] for k in known], float)
        kt = np.array([k[1] for k in known], float)
        kx, ky = source_xy(kr, kt, pxscale, cx, cy, angle_convention)
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy)
    finite = np.isfinite(img)
    if pixel_mask is not None:
        finite = finite & ~np.asarray(pixel_mask, bool)
    isr = int(np.ceil(search_px))

    snr = np.full(nsrc, np.nan)
    details: List[Dict[str, float]] = []
    for s in range(nsrc):
        A = amap(s)
        px, py = int(round(xs[s])), int(round(ys[s]))
        pkv = -np.inf
        bx, by = px, py
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
        if r < fwhm or nap < 5:
            details.append(det)
            continue
        th0 = np.arctan2(py - cy, px - cx)
        vals, valok = [], []
        for q in range(1, nap):
            th = th0 + q * 2 * np.pi / nap
            xk, yk = int(round(cx + r * np.cos(th))), int(round(cy + r * np.sin(th)))
            if np.any(np.hypot(xk - xs, yk - ys) < exr):
                continue
            if not (0 <= xk < nx and 0 <= yk < ny) or not finite[yk, xk]:
                continue
            mok = not (kx.size and np.any(np.hypot(xk - kx, yk - ky) < exr))
            vals.append(A[yk, xk])
            valok.append(mok)
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
            bvals = A[keep] if keep.sum() >= 5 else A[bnd]
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
        if sd > 0:
            if sd_band > 0:
                sd = max(sd, sd_band)
            n_pen = max(nap, 3) if penalty == "geometric" else max(nclean, 1)
            snr[s] = (A[py, px] - bg) / (sd * np.sqrt(1.0 + 1.0 / n_pen))
        det.update(sigma=sd, bg=bg, nclean=nclean, sd_band=sd_band)
        details.append(det)
    if return_details:
        return snr, details
    return snr


def _ring_fluxes(img, cx, cy, sep, phi0, nap, aprad, excl_xy, exr, known_xy):
    ref, ok = [], []
    for k in range(1, nap):
        phik = phi0 + k * 2 * np.pi / nap
        xk, yk = cx + sep * np.cos(phik), cy + sep * np.sin(phik)
        if any(np.hypot(xk - ex, yk - ey) < exr for ex, ey in excl_xy):
            continue
        f = _aperture_sum(img, xk, yk, aprad)
        if np.isfinite(f):
            ref.append(f)
            ok.append(not any(np.hypot(xk - kx, yk - ky) < exr for kx, ky in known_xy))
    return np.array(ref), np.array(ok, bool)


def injection_difference_snr(img_inj: np.ndarray, img_clean: np.ndarray, rho, theta, pxscale: float,
                             fwhm: float, *, excl_fwhm: float = 1.5,
                             known: Sequence[Tuple[float, float]] = (),
                             angle_convention: str = "pa") -> np.ndarray:
    """Injection-recovery S/N (``near2_snr_inj``): signal = aperture flux of
    ``inj - clean`` at the source, noise = scatter of same-radius apertures on the
    clean image times ``sqrt(1+1/nap)`` (starved rings are resampled at half-FWHM
    spacing but the penalty keeps the independent count)."""
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
                       excl_fwhm: float = 1.5, angle_convention: str = "pa") -> np.ndarray:
    """Plain aperture Mawet S/N (``near2_snr``): ``(f_src - mean(ring)) / (std(ring)
    sqrt(1+1/n))`` with FWHM/2 apertures; kept for completeness / legacy comparison."""
    img = np.asarray(img, float)
    cx, cy = star_center(img.shape)
    rho = np.atleast_1d(np.asarray(rho, float))
    theta = np.atleast_1d(np.asarray(theta, float))
    exr = excl_fwhm * fwhm
    aprad = fwhm / 2.0
    imz = np.where(np.isfinite(img), img, 0.0)
    xs, ys = source_xy(rho, theta, pxscale, cx, cy, angle_convention)
    excl = list(zip(xs, ys))
    phi = np.deg2rad(theta + (90.0 if angle_convention == "pa" else 0.0))
    snr = np.full(rho.size, np.nan)
    for s in range(rho.size):
        sep = rho[s] / pxscale
        nap = max(int(np.floor(2 * np.pi * sep / fwhm)), 4)
        f_src = _aperture_sum(imz, xs[s], ys[s], aprad)
        ref, _ = _ring_fluxes(img, cx, cy, sep, phi[s], nap, aprad, excl, exr, [])
        if ref.size < 2 or not np.isfinite(f_src):
            continue
        noise = np.std(ref, ddof=1) * np.sqrt(1.0 + 1.0 / ref.size)
        snr[s] = (f_src - ref.mean()) / noise if noise > 0 else np.nan
    return snr


# ----------------------------------------------------------------------------
# metric objects
# ----------------------------------------------------------------------------
class Metric:
    """Callable ``(img_inj, img_clean, sources) -> per-source S/N``.

    A metric that sets ``needs_fm = True`` (:class:`~klip_tpe.fmmf.FMMFSNR`) asks the
    runner for the KLIP-FM pass of the same configuration and is called with the extra
    keyword ``fm=`` -- the forward-modelled response of the injected sources, from which
    it builds its position-dependent matched filter.  ``fm=None`` must stay valid: the
    k-scan path has no forward model, and the metric falls back there.
    """

    name = "metric"
    needs_clean = False
    needs_fm = False

    def per_source(self, img_inj, img_clean, rho, theta) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> Dict:
        return {"name": self.name}


@dataclass
class MawetPeakSNR(Metric):
    """Default search & validation metric (see :func:`mawet_peak_snr`).

    ``kernel_fn(rho_arcsec) -> kernel`` lets an injection model supply the measured
    matched filter at the sources' median separation; ``flatten`` applies
    :func:`radprof` before scoring.

    ``flatten=False`` is the default, which is a deliberate divergence from the IDL --
    every IDL variant flattens.  :func:`radprof` subtracts the azimuthal mean of each
    integer-radius bin, so anything that is not azimuthally uniform at a given radius
    leaks into that mean and is subtracted off the whole ring: a bright companion, the
    MIRI 4QPM dead zones, a glow-stick residual.  On a ring the mask already eats, the
    mean is taken over the surviving pixels and then removed from all of them, which
    biases the very sources the metric is trying to measure.  Pass ``flatten=True`` to
    restore the IDL behaviour."""

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

    def kernel(self, rho) -> np.ndarray:
        if self.kernel_fn is not None:
            k = self.kernel_fn(float(np.nanmedian(np.atleast_1d(rho))))
            if k is not None:
                return k
        return gaussian_kernel(self.mf_fwhm or self.fwhm)

    def per_source(self, img_inj, img_clean, rho, theta) -> np.ndarray:
        img = radprof(img_inj) if self.flatten else np.asarray(img_inj, float)
        return mawet_peak_snr(img, rho, theta, self.pxscale, self.fwhm, kernel=self.kernel(rho),
                              search_px=self.search_px, excl_fwhm=self.excl_fwhm, known=self.known,
                              declip_nsig=self.declip_nsig, band_fwhm=self.band_fwhm,
                              min_ring=self.min_ring, penalty=self.penalty,
                              pixel_mask=self.pixel_mask, angle_convention=self.angle_convention)

    def describe(self) -> Dict:
        return {"name": self.name, "fwhm": self.fwhm, "pxscale": self.pxscale,
                "mf_fwhm": self.mf_fwhm, "measured_kernel": self.kernel_fn is not None,
                "search_px": self.search_px, "excl_fwhm": self.excl_fwhm,
                "known": [list(k) for k in self.known], "declip_nsig": self.declip_nsig,
                "band_fwhm": self.band_fwhm, "min_ring": self.min_ring, "penalty": self.penalty,
                # a masked noise estimate is not comparable with an unmasked one, so the
                # run's own setup record has to say whether one was in force
                "pixel_mask_px": None if self.pixel_mask is None else int(np.count_nonzero(self.pixel_mask)),
                "flatten": self.flatten}


@dataclass
class InjectionDifferenceSNR(Metric):
    """Alternative metric on ``inj - clean`` (``use_noinj_noise`` in IDL).  Needs the
    clean image on every call."""

    pxscale: float
    fwhm: float
    excl_fwhm: float = 1.5
    known: Sequence[Tuple[float, float]] = ()
    flatten: bool = False
    angle_convention: str = "pa"
    name: str = "injection_difference"
    needs_clean: bool = True

    def per_source(self, img_inj, img_clean, rho, theta) -> np.ndarray:
        if img_clean is None:
            raise ValueError("InjectionDifferenceSNR needs the clean image")
        a = radprof(img_inj) if self.flatten else np.asarray(img_inj, float)
        b = radprof(img_clean) if self.flatten else np.asarray(img_clean, float)
        return injection_difference_snr(a, b, rho, theta, self.pxscale, self.fwhm,
                                        excl_fwhm=self.excl_fwhm, known=self.known,
                                        angle_convention=self.angle_convention)

    def describe(self) -> Dict:
        return {"name": self.name, "fwhm": self.fwhm, "pxscale": self.pxscale,
                "excl_fwhm": self.excl_fwhm, "known": [list(k) for k in self.known]}


@dataclass
class ScoreResult:
    score: float                         # aggregate (nan = failed)
    per_source: np.ndarray               # corrected per-source values
    raw_per_source: np.ndarray           # uncorrected injected-image values
    clean_per_source: Optional[np.ndarray] = None

    @property
    def raw_score(self) -> float:
        return nanmedian_even(self.raw_per_source)


class Objective:
    """Turns per-source metric values into the scalar the optimizer maximises.

    ``clean_subtract=True`` (production): ``s = s_inj - max(s_clean, 0)`` per source,
    a strictly one-sided penalty that removes speckle contamination at the injection
    site but can never manufacture signal from a dark hole; NaN clean terms drop the
    source from the aggregate.  Validation always uses :meth:`score_raw`.

    A metric that already consumes the clean image (``metric.needs_clean``, e.g.
    :class:`InjectionDifferenceSNR` on ``inj - clean``) has the speckle term removed by
    construction, so for it ``clean_subtract`` is a documented no-op: :meth:`score_search`
    returns the raw ``metric(inj, clean)`` values with ``clean_per_source=None`` instead of
    subtracting a second, ill-defined clean term (``metric(clean, None)`` would raise).
    """

    def __init__(self, metric: Metric, clean_subtract: bool = True, aggregate: str = "median"):
        self.metric = metric
        self.clean_subtract = bool(clean_subtract)
        self.aggregate = aggregate

    @property
    def needs_clean(self) -> bool:
        return self.clean_subtract or self.metric.needs_clean

    @property
    def effective_clean_subtract(self) -> bool:
        """``clean_subtract`` as actually applied: False for a metric that already uses
        the clean image (the difference metric), where the clean term is a no-op."""
        return self.clean_subtract and not self.metric.needs_clean

    def _agg(self, v) -> float:
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan")
        if self.aggregate == "mean":
            return float(np.mean(v))
        return float(np.median(v))

    def _fmkw(self, fm) -> Dict:
        """``fm`` reaches the metric only when it asked for it, so every existing
        :class:`Metric` keeps its 4-argument signature."""
        return {"fm": fm} if self.metric.needs_fm else {}

    def score_raw(self, img_inj, sources: Sequence[Source], img_clean=None, fm=None) -> ScoreResult:
        rho = [s.rho for s in sources]
        th = [s.theta for s in sources]
        raw = self.metric.per_source(img_inj, img_clean, rho, th, **self._fmkw(fm))
        return ScoreResult(self._agg(raw), raw, raw, None)

    def score_search(self, img_inj, sources: Sequence[Source], img_clean=None, fm=None) -> ScoreResult:
        rho = [s.rho for s in sources]
        th = [s.theta for s in sources]
        kw = self._fmkw(fm)
        raw = self.metric.per_source(img_inj, img_clean, rho, th, **kw)
        if not self.effective_clean_subtract:
            # plain metric, or a difference metric that already used the clean image:
            # the search score IS the raw score (no separate clean term)
            return ScoreResult(self._agg(raw), raw, raw, None)
        if img_clean is None:
            raise ValueError("clean_subtract objective needs the clean image")
        # the FM template is a property of the configuration, not of the injection, so the
        # clean term is filtered with exactly the same kernels
        cl = self.metric.per_source(img_clean, None, rho, th, **kw)
        corr = raw - clip0(cl)
        return ScoreResult(self._agg(corr), corr, raw, cl)

    def describe(self) -> Dict:
        return {"metric": self.metric.describe(), "clean_subtract": self.clean_subtract,
                "effective_clean_subtract": self.effective_clean_subtract, "aggregate": self.aggregate}
