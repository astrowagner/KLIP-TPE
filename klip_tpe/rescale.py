"""Spectral rescaling of frames about the star, for SDI references.

A speckle sits at a radius proportional to the wavelength, a planet does not.  Magnifying a
frame of wavelength group g' by ``s = lambda_g / lambda_g'`` about the star puts its speckles
where group g has them and its planet at ``r * s`` instead of ``r``, which is what makes it a
reference for g that the planet's own light does not fully reach.  The interpolation is the
cubic B-spline of :func:`scipy.ndimage.map_coordinates` (order 3), NaN-aware: NaN pixels are
set to zero for the spline and every output pixel whose stencil touched one is NaN again,
and so is every output pixel that falls outside the frame.  The same choices as the ALES
reduction's own ``asdi.rescale`` (high-pass before the rescaling, cubic, NaN propagated).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["rescale_frame", "rescale_frames", "scale_coordinates", "displacement_px"]


def scale_coordinates(shape: Tuple[int, int], s: float,
                      center: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Sample coordinates ``(ys, xs)`` that magnify a frame of ``shape`` by ``s`` about
    ``center`` (``(x, y)``, the array centre ``((nx - 1) / 2, (ny - 1) / 2)`` by default):
    output pixel ``(y, x)`` reads the input at ``center + (p - center) / s``."""
    ny, nx = int(shape[0]), int(shape[1])
    cx, cy = ((nx - 1) / 2.0, (ny - 1) / 2.0) if center is None else (float(center[0]), float(center[1]))
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    s = float(s)
    return cy + (y - cy) / s, cx + (x - cx) / s


def rescale_frame(img: np.ndarray, s: float, center: Optional[Tuple[float, float]] = None,
                  order: int = 3, coords: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """One frame magnified by ``s`` about ``center`` (``s > 1`` enlarges), NaN-aware.
    ``s == 1`` returns a float copy unchanged.  ``coords`` are :func:`scale_coordinates` of
    the frame, when the caller has them already."""
    from scipy.ndimage import map_coordinates
    img = np.asarray(img, float)
    if abs(float(s) - 1.0) < 1e-12:
        return img.copy()
    ys, xs = scale_coordinates(img.shape, s, center) if coords is None else coords
    bad = ~np.isfinite(img)
    out = map_coordinates(np.where(bad, 0.0, img), [ys, xs], order=int(order), mode="constant", cval=np.nan)
    if bad.any():
        # a NaN's reach: bilinear weight of the bad pixels at the sample point; any at all
        # (above round-off) and the output pixel took a value from one of them
        bm = map_coordinates(bad.astype(float), [ys, xs], order=1, mode="constant", cval=1.0)
        out[bm > 0.01] = np.nan
    return out


def rescale_frames(cube: np.ndarray, s: float, center: Optional[Tuple[float, float]] = None,
                   order: int = 3, threads: int = 1, dtype=np.float32) -> np.ndarray:
    """Every frame of a ``(n, ny, nx)`` cube through :func:`rescale_frame`, on ``threads``
    (the spline evaluation releases the GIL for part of its work)."""
    cube = np.asarray(cube)
    n = int(cube.shape[0])
    out = np.empty(cube.shape, dtype)
    if n == 0:
        return out
    if abs(float(s) - 1.0) < 1e-12:
        out[...] = cube
        return out
    coords = scale_coordinates(cube.shape[1:], s, center)

    def run(i):
        out[i] = rescale_frame(cube[i], s, center, order=order, coords=coords)

    nth = int(max(threads or 1, 1))
    if nth <= 1 or n < 4:
        for i in range(n):
            run(i)
        return out
    from .parallel import target_pool
    list(target_pool(min(nth, n)).map(run, range(n)))
    return out


def displacement_px(mid_px: float, scales: Sequence[float], dpa_deg: Sequence[float]) -> np.ndarray:
    """How far a planet at the annulus' mid radius moves between a target frame and a
    reference frame that is rotated by ``dpa_deg`` and magnified by ``scales`` (1 for the
    target's own group): ``mid * sqrt(1 + s^2 - 2 s cos dPA)`` pixels.  At ``s = 1`` this is
    the chord ``2 mid sin(dPA / 2)``, the ADI arc rule to first order; at ``dPA = 0`` it is
    the spectral displacement ``mid |s - 1|``."""
    s = np.asarray(scales, float)
    d = np.deg2rad(np.asarray(dpa_deg, float))
    return float(mid_px) * np.sqrt(np.maximum(1.0 + s * s - 2.0 * s * np.cos(d), 0.0))
