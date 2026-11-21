"""Spectral rescaling of frames about the star, for SDI references.

A speckle sits at a radius proportional to the wavelength, and a planet does not.  Magnifying a frame
of wavelength group g' by ``s = lambda_g / lambda_g'`` about the star puts its speckles where group g
has them and its planet at ``r * s`` instead of ``r``, which makes it a reference for g that the
planet's own light does not fully reach.

The interpolation is the cubic B-spline of :func:`scipy.ndimage.map_coordinates` (order 3), aware of
NaN.  A NaN pixel is set to zero for the spline, and every output pixel whose stencil touched one is
NaN again, as is every output pixel that falls outside the frame.  These are the ALES reduction's own
choices in ``asdi.rescale``: the high-pass filter before the rescaling, the cubic spline, and NaN
carried through.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["rescale_frame", "rescale_frames", "scale_coordinates", "displacement_px"]


def scale_coordinates(shape: Tuple[int, int], s: float,
                      center: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return the sample coordinates that magnify a frame about a center.

    Output pixel ``p`` reads the input at ``center + (p - center) / s``.

    Parameters
    ----------
    shape : tuple of 2 ints
        The frame's shape, ``(ny, nx)``.
    s : float
        The magnification.
    center : tuple of 2 floats or None, default None
        The center, ``(x, y)``, None for the array's center ``((nx - 1) / 2, (ny - 1) / 2)``.

    Returns
    -------
    ys : ndarray, shape (ny, nx)
        The rows to read.
    xs : ndarray, shape (ny, nx)
        The columns to read.
    """
    ny, nx = int(shape[0]), int(shape[1])
    cx, cy = ((nx - 1) / 2.0, (ny - 1) / 2.0) if center is None else (float(center[0]), float(center[1]))
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    s = float(s)
    return cy + (y - cy) / s, cx + (x - cx) / s


def rescale_frame(img: np.ndarray, s: float, center: Optional[Tuple[float, float]] = None,
                  order: int = 3, coords: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """Return a frame magnified about a center, with NaN carried through.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The frame.
    s : float
        The magnification, above 1 to enlarge.  1 returns a float copy unchanged.
    center : tuple of 2 floats or None, default None
        The center, ``(x, y)``, None for the array's center.
    order : int, default 3
        Order of the spline.
    coords : tuple of 2 ndarrays or None, default None
        :func:`scale_coordinates` of the frame, when the caller has them already.

    Returns
    -------
    out : ndarray, shape (ny, nx)
        The magnified frame, NaN where a NaN pixel reached the stencil or the sample fell outside.
    """
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
    """Return every frame of a cube magnified about a center (:func:`rescale_frame`).

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The frames.
    s : float
        The magnification.
    center : tuple of 2 floats or None, default None
        The center, ``(x, y)``, None for the array's center.
    order : int, default 3
        Order of the spline.
    threads : int, default 1
        Number of threads.  The spline's evaluation releases the GIL for part of its work.
    dtype : numpy dtype, default numpy.float32
        The result's type.

    Returns
    -------
    out : ndarray, shape (n, ny, nx)
        The magnified frames.
    """
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
    """Return how far a planet at an annulus' mid radius moves between a target and its references.

    A reference frame is turned by ``dpa_deg`` and magnified by ``scales``, 1 for the target's own group,
    which moves the planet by ``mid * sqrt(1 + s^2 - 2 s cos dPA)`` pixels.  At ``s = 1`` that is the chord
    ``2 mid sin(dPA / 2)``, the ADI arc rule to first order, and at ``dPA = 0`` the spectral displacement
    ``mid |s - 1|``.

    Parameters
    ----------
    mid_px : float
        The annulus' mid radius, in pixels.
    scales : sequence of float
        Each reference's magnification.
    dpa_deg : sequence of float
        Each reference's turn, in degrees.

    Returns
    -------
    displacement : ndarray
        The planet's displacement per reference, in pixels.
    """
    s = np.asarray(scales, float)
    d = np.deg2rad(np.asarray(dpa_deg, float))
    return float(mid_px) * np.sqrt(np.maximum(1.0 + s * s - 2.0 * s * np.cos(d), 0.0))
