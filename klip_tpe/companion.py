"""Known companions: fit one, take it out of the frames, and measure its S/N honestly.

Two jobs, both done with the negative-fake-companion method: a companion injected with the
NEGATIVE of its contrast at its position cancels it in every frame, before PSF subtraction,
so nothing of it reaches the KL basis, the noise rings or the score.

:func:`fit_negative_companion`
    the contrast (and optionally a sub-pixel position correction) that minimises the
    residual energy in a stamp around the companion.  Fit it in a configuration whose
    reference library does not contain the companion (RDI): there the reduction is linear
    in the companion's flux and the fit is unbiased.  A basis built from frames that hold
    the companion (ADI) subtracts a bright source non-linearly, and the fitted contrast
    then absorbs that configuration's self-subtraction instead of measuring the source.

:func:`companion_snr`
    the companion's S/N in one configuration, from its own reduced image (clean minus
    companion-removed) over the ring scatter of the companion-removed image.  The search
    metric applied to the clean image does worse on a bright companion in three ways,
    each by a different amount per configuration: its de-spiking clips the companion's
    core; the companion's own light beyond the 1.5-FWHM exclusion -- PSF lobes, the
    negative wings ADI leaves, the imprint KLIP spreads over the companion's sector --
    lands in the noise ring (on JWST/NIRCam it set the ring scatter at 5-14 times the
    companion-free value across the whole annulus); and at small separations its few
    apertures make the scatter depend on where they fall.  Here the companion's light
    goes into the signal model, the noise image holds none of it, nothing is de-spiked,
    and the ring scatter is averaged over phases of the aperture ring.

Both take a ``reduce_fn(sources) -> 2-D image`` that reduces ONE fixed configuration with
the given extra injections (``None`` for none), so they work with any engine, partition
layout or zone: wrap ``Runner._reduce`` or a reducer's ``reduce`` in a lambda.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .metrics import Source, source_xy, star_center

__all__ = ["fit_negative_companion", "companion_snr", "ring_sigma"]


def _xy(shape, rho, pa, pxscale, angle_convention):
    cx, cy = star_center(shape)
    xs, ys = source_xy([rho], [pa], pxscale, cx, cy, angle_convention)
    return float(xs[0]), float(ys[0])


def _rho_pa(shape, x, y, pxscale, angle_convention):
    """Inverse of :func:`source_xy` for one position."""
    cx, cy = star_center(shape)
    rho = float(np.hypot(x - cx, y - cy)) * pxscale
    phi = float(np.degrees(np.arctan2(y - cy, x - cx)))
    pa = phi - 90.0 if angle_convention == "pa" else phi
    return rho, pa % 360.0


def fit_negative_companion(reduce_fn: Callable[[Optional[Sequence[Source]]], np.ndarray],
                           rho: float, pa: float, contrast0: float, fwhm_px: float, pxscale: float, *,
                           angle_convention: str = "pa", fit_position: bool = True,
                           radius_fwhm: float = 2.0,
                           rel_grid: Sequence[float] = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0),
                           max_evals: int = 40, clean: Optional[np.ndarray] = None,
                           log: Optional[Callable[[str], None]] = None) -> Dict[str, float]:
    """Fit the companion's contrast (and position) by the negative-fake-companion method.

    The figure of merit is the sum of squared residuals within ``radius_fwhm`` FWHM of the
    starting position in ``reduce_fn([Source(rho, pa, -c)])``.  A contrast grid at the
    starting position (``rel_grid`` x ``contrast0``, re-centred once if the minimum falls on
    an edge) gives the start; with ``fit_position`` a Nelder-Mead over (contrast, dx, dy)
    refines it, for at most ``max_evals`` further reductions.

    Returns ``{'rho', 'pa', 'contrast', 'chi2', 'chi2_clean', 'removed_fraction',
    'n_reductions'}``; ``removed_fraction = 1 - chi2 / chi2_clean`` is how much of the
    stamp's energy the fitted companion accounts for.
    """
    clean = np.asarray(reduce_fn(None), float) if clean is None else np.asarray(clean, float)
    shape = clean.shape
    x0, y0 = _xy(shape, rho, pa, pxscale, angle_convention)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    stamp = ((xx - x0) ** 2 + (yy - y0) ** 2 <= (radius_fwhm * fwhm_px) ** 2) & np.isfinite(clean)
    if stamp.sum() < 5:
        raise ValueError(f"no finite pixels within {radius_fwhm} FWHM of ({rho}\", {pa} deg)")
    chi2_clean = float(np.sum(clean[stamp] ** 2))
    n_red = 1
    cache: Dict[Tuple[float, float, float], float] = {}

    def chi2(c, dx=0.0, dy=0.0):
        nonlocal n_red
        key = (round(float(c), 12), round(float(dx), 4), round(float(dy), 4))
        if key not in cache:
            r, p = (rho, pa) if dx == 0.0 and dy == 0.0 else _rho_pa(shape, x0 + dx, y0 + dy, pxscale,
                                                                       angle_convention)
            img = np.asarray(reduce_fn([Source(r, p, -float(c))]), float)
            n_red += 1
            v = img[stamp]
            cache[key] = float(np.sum(np.where(np.isfinite(v), v, 0.0) ** 2))
        return cache[key]

    def grid_min(c_ref):
        gs = float(c_ref) * np.asarray(rel_grid, float)
        ch = np.array([chi2(g) for g in gs])
        i = int(np.argmin(ch))
        best = float(gs[i])
        if 0 < i < len(gs) - 1:
            cf = np.polyfit(gs[i - 1:i + 2], ch[i - 1:i + 2], 2)
            if cf[0] > 0:
                best = float(-cf[1] / (2 * cf[0]))
        return best, i in (0, len(gs) - 1)

    c1, edge = grid_min(contrast0)
    if edge:
        c1, _ = grid_min(c1)
    best = (c1, 0.0, 0.0)
    if fit_position:
        from scipy.optimize import minimize
        scale = np.array([max(abs(c1), 1e-12), 1.0, 1.0])

        def f(z):
            c, dx, dy = z * scale
            if c <= 0:
                return 1e300
            return chi2(c, dx, dy)

        z0 = np.array([c1, 0.0, 0.0]) / scale
        simplex = np.array([z0, z0 + [0.1, 0, 0], z0 + [0, 0.3, 0], z0 + [0, 0, 0.3]])
        res = minimize(f, z0, method="Nelder-Mead",
                       options=dict(initial_simplex=simplex, maxfev=int(max_evals), xatol=1e-3, fatol=1e-6 * chi2_clean))
        cb, dxb, dyb = res.x * scale
        if f(res.x) <= chi2(c1):
            best = (float(cb), float(dxb), float(dyb))
    c, dx, dy = best
    r_b, p_b = (rho, pa) if dx == 0.0 and dy == 0.0 else _rho_pa(shape, x0 + dx, y0 + dy, pxscale, angle_convention)
    ch = chi2(c, dx, dy)
    out = dict(rho=float(r_b), pa=float(p_b), contrast=float(c), chi2=float(ch), chi2_clean=chi2_clean,
               removed_fraction=float(1.0 - ch / chi2_clean) if chi2_clean > 0 else float("nan"),
               n_reductions=int(n_red), dx_px=float(dx), dy_px=float(dy))
    if log is not None:
        log(f"  negative companion: contrast {c:.4e} at {r_b:.4f}\" PA {p_b:.2f} "
            f"(moved {np.hypot(dx, dy):.2f} px), removes {100 * out['removed_fraction']:.1f}% of the stamp energy "
            f"({n_red} reductions)")
    return out


def ring_sigma(img: np.ndarray, rho: float, pa: float, fwhm_px: float, pxscale: float, kernel: np.ndarray, *,
               pixel_mask: Optional[np.ndarray] = None, angle_convention: str = "pa",
               exclude: Sequence[Tuple[float, float]] = (), excl_fwhm: float = 1.5,
               n_phase: int = 8) -> Dict[str, float]:
    """Scatter of the matched-filtered ``img`` at ``rho``: FWHM-spaced apertures on the ring
    through (``rho``, ``pa``), skipping masked pixels and apertures within ``excl_fwhm`` FWHM
    of (``rho``, ``pa``) and of every ``exclude`` position, with Mawet's small-sample factor;
    the RMS over ``n_phase`` phase offsets of the aperture ring, so the value does not depend
    on where the apertures happen to fall.  Also the scatter of all matched-filtered pixels in
    a 1-px band at that radius (same exclusions), for comparison."""
    img = np.asarray(img, float)
    A = ndimage.convolve(np.where(np.isfinite(img), img, 0.0), kernel, mode="nearest")
    ny, nx = img.shape
    cx, cy = star_center(img.shape)
    x0, y0 = _xy(img.shape, rho, pa, pxscale, angle_convention)
    ex = [(x0, y0)] + [_xy(img.shape, r, p, pxscale, angle_convention) for r, p in exclude]
    exr = excl_fwhm * fwhm_px
    r = float(np.hypot(x0 - cx, y0 - cy))
    th0 = float(np.arctan2(y0 - cy, x0 - cx))
    nap = int(np.floor(2 * np.pi * r / fwhm_px))
    fin = np.isfinite(img)
    if pixel_mask is not None:
        fin &= ~np.asarray(pixel_mask, bool)
    sig, ns = [], []
    for ph in np.arange(max(int(n_phase), 1)) / max(int(n_phase), 1):
        v = []
        for q in range(nap):
            th = th0 + (q + ph) * 2 * np.pi / nap
            x, y = cx + r * np.cos(th), cy + r * np.sin(th)
            if any(np.hypot(x - a, y - b) < exr for a, b in ex):
                continue
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < nx and 0 <= yi < ny and fin[yi, xi]:
                v.append(A[yi, xi])
        if len(v) >= 3:
            sig.append(float(np.std(v, ddof=1) * np.sqrt(1.0 + 1.0 / len(v))))
            ns.append(len(v))
    yy, xx = np.mgrid[0:ny, 0:nx]
    band = (np.abs(np.hypot(xx - cx, yy - cy) - r) <= 0.5) & fin
    for a, b in ex:
        band &= (xx - a) ** 2 + (yy - b) ** 2 >= exr * exr
    sig_band = float(np.std(A[band], ddof=1) * np.sqrt(1.0 + 1.0 / max(nap, 1))) if band.sum() >= 3 else float("nan")
    return dict(sigma=float(np.sqrt(np.mean(np.square(sig)))) if sig else float("nan"),
                sigma_min=float(min(sig)) if sig else float("nan"), sigma_max=float(max(sig)) if sig else float("nan"),
                sigma_band=sig_band, n_ap=int(np.median(ns)) if ns else 0, nap=nap)


def companion_snr(reduce_fn: Callable[[Optional[Sequence[Source]]], np.ndarray], rho: float, pa: float,
                  contrast: float, fwhm_px: float, pxscale: float, kernel: np.ndarray, *,
                  pixel_mask: Optional[np.ndarray] = None, angle_convention: str = "pa",
                  clean: Optional[np.ndarray] = None, removed: Optional[np.ndarray] = None,
                  search_px: float = 1.5, n_phase: int = 8) -> Dict[str, object]:
    """Forward-model S/N of a known companion in one configuration.

    ``signal`` is the matched-filter peak (within ``search_px`` of the position) of the
    companion's own reduced image, ``clean - removed``, where ``removed`` is the reduction with
    the companion cancelled by a negative injection at (``rho``, ``pa``, ``contrast``) -- use
    the values :func:`fit_negative_companion` returns.  ``sigma`` is :func:`ring_sigma` of
    ``removed`` at the companion's separation.  Returns the S/N (``snr``; ``snr_band`` with the
    band scatter), both images, and the pieces.
    """
    clean = np.asarray(reduce_fn(None), float) if clean is None else np.asarray(clean, float)
    removed = (np.asarray(reduce_fn([Source(rho, pa, -float(contrast))]), float) if removed is None
               else np.asarray(removed, float))
    model = clean - removed
    A = ndimage.convolve(np.where(np.isfinite(model), model, 0.0), kernel, mode="nearest")
    x0, y0 = _xy(clean.shape, rho, pa, pxscale, angle_convention)
    isr = int(np.ceil(search_px))
    best, bx, by = -np.inf, int(round(x0)), int(round(y0))
    for dy in range(-isr, isr + 1):
        for dx in range(-isr, isr + 1):
            if dx * dx + dy * dy > search_px * search_px:
                continue
            qx, qy = int(round(x0)) + dx, int(round(y0)) + dy
            if 0 <= qy < A.shape[0] and 0 <= qx < A.shape[1] and A[qy, qx] > best:
                best, bx, by = float(A[qy, qx]), qx, qy
    rs = ring_sigma(removed, rho, pa, fwhm_px, pxscale, kernel, pixel_mask=pixel_mask,
                    angle_convention=angle_convention, n_phase=n_phase)
    snr = best / rs["sigma"] if rs["sigma"] > 0 else float("nan")
    snr_band = best / rs["sigma_band"] if rs["sigma_band"] > 0 else float("nan")
    return dict(snr=float(snr), snr_band=float(snr_band), signal=float(best), peak_xy=(bx, by),
                sigma=rs["sigma"], sigma_band=rs["sigma_band"], sigma_range=(rs["sigma_min"], rs["sigma_max"]),
                n_ap=rs["n_ap"], nap=rs["nap"], contrast=float(contrast), clean=clean, removed=removed)
