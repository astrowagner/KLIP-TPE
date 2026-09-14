"""Generic KLIP / ADI engine (port of ``multiklip.pro`` + ``get_klip_basis_new.pro``
+ the processing chain of ``reduce_near_2.pro``: frame selection, angle-aware
temporal binning, boxcar high-pass, annular KLIP with per-target reference
selection, derotation and noise-weighted combination ``nw_ang_comb``).

Array convention: cubes are ``(nframes, ny, nx)``; images ``(ny, nx)``; the star is
at ``((nx-1)/2, (ny-1)/2)``.  Angles are degrees.  Rotations are counter-clockwise
in the ``(x, y)`` frame (IDL ``rot(img, a)`` is clockwise by ``a``, so IDL
``rot(img, -a)`` == :func:`rotate_ccw` ``(img, a)``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

__all__ = ["frame_selection_mask", "bin_frames", "bin_groups", "bin_by_groups", "bin_angles",
           "highpass", "destripe", "rotate_ccw", "derotate", "nw_ang_comb", "nw_ang_comb_ref",
           "klip_basis", "zone_indices", "klip_annular", "klip_fm_zone", "arcdist_deg",
           "reference_mask", "KLIPParams"]


# ----------------------------------------------------------------------------
# frame selection (near2_framesel)
# ----------------------------------------------------------------------------
def frame_selection_mask(nf: int, tags: Optional[Dict[str, np.ndarray]], corr_thresh: Optional[float],
                         noise_max: Optional[float], coronoise_max: Optional[float],
                         bin_: int, k_klip: int) -> np.ndarray:
    """Boolean keep-mask.  ``keep = corr >= corr_thresh & noise <= noise_max*mean(noise)
    & coronoise <= coronoise_max*mean(coronoise)`` (NaN tags pass).  The cut is
    skipped entirely (all True) when it would keep fewer than
    ``max(round(bin)*round(k_klip), 3)`` frames or would cull nothing."""
    keep = np.ones(nf, bool)
    if not tags or corr_thresh is None:
        return keep
    corrs = np.asarray(tags.get("corrs", []), float)
    if corrs.size != nf:
        return keep
    keep &= corrs >= float(corr_thresh)
    for key, thr in (("noises", noise_max), ("coronoise", coronoise_max)):
        v = tags.get(key)
        if v is None or thr is None:
            continue
        v = np.asarray(v, float)
        if v.size == nf:
            keep &= (v <= float(thr) * np.nanmean(v)) | ~np.isfinite(v)
    nw = int(keep.sum())
    minkeep = max(int(round(max(bin_, 1))) * int(round(max(k_klip, 1))), 3)
    if nw < minkeep or nw == nf or nw == 0:
        return np.ones(nf, bool)
    return keep


# ----------------------------------------------------------------------------
# angle-aware temporal binning
# ----------------------------------------------------------------------------
def _bin_groups(angles: np.ndarray, bin_: int, dth_max: float) -> np.ndarray:
    grp = np.zeros(angles.size, int)
    g, cnt = 0, 0
    amn = amx = angles[0] if angles.size else 0.0
    for ii in range(angles.size):
        if cnt == 0:
            amn = amx = angles[ii]
        else:
            tmn, tmx = min(amn, angles[ii]), max(amx, angles[ii])
            if cnt >= bin_ or (tmx - tmn) > dth_max:
                g += 1
                cnt = 0
                amn = amx = angles[ii]
            else:
                amn, amx = tmn, tmx
        grp[ii] = g
        cnt += 1
    return grp


def bin_angles(angles: np.ndarray, bin_: int, dth_max_deg: float) -> np.ndarray:
    """Binned parallactic angles only (used by the reference-count guardrail)."""
    angles = np.asarray(angles, float)
    b = int(round(max(bin_, 1)))
    if b <= 1 or angles.size == 0:
        return angles.copy()
    grp = _bin_groups(angles, b, dth_max_deg)
    ng = grp.max() + 1
    return np.bincount(grp, weights=angles, minlength=ng) / np.bincount(grp, minlength=ng)


def bin_groups(angles: np.ndarray, bin_: int, dth_max_deg: float) -> np.ndarray:
    """Group index per frame of the angle-aware binning (``bin_grp`` in
    ``reduce_near_2``, reused verbatim for the KLIP-FM model cube).  ``bin_ <= 1``
    -> every frame its own group."""
    angles = np.asarray(angles, float)
    b = int(round(max(bin_, 1)))
    if b <= 1:
        return np.arange(angles.size)
    return _bin_groups(angles, b, dth_max_deg)


def bin_by_groups(cube: np.ndarray, grp: np.ndarray) -> np.ndarray:
    """Mean-combine the frames of each group (``bin_type='mean'``; the model cube is
    always mean-binned, reduce_near_2 L1792-1806)."""
    ng = int(grp.max()) + 1 if grp.size else 0
    bc = np.empty((ng,) + cube.shape[1:], np.float32)
    for g in range(ng):
        w = np.flatnonzero(grp == g)
        bc[g] = cube[w].mean(axis=0) if w.size > 1 else cube[w[0]]
    return bc


def bin_frames(cube: np.ndarray, angles: np.ndarray, bin_: int, dth_max_deg: float,
               return_groups: bool = False):
    """Mean-combine consecutive frames into bins of at most ``bin_`` frames, closing
    a bin early when its PA span would exceed ``dth_max_deg``.  Drops all-zero bins
    (``bads``, reduce_near_2 L1747-1761).  With ``return_groups`` also returns
    ``(grp, keep)`` so a companion cube can be binned identically
    (:func:`bin_by_groups` followed by ``[keep]``)."""
    angles = np.asarray(angles, float)
    grp = bin_groups(angles, bin_, dth_max_deg)
    if int(round(max(bin_, 1))) <= 1:
        bc, ba = np.asarray(cube, np.float32), angles.copy()
    else:
        bc = bin_by_groups(cube, grp)
        ng = grp.max() + 1
        ba = np.bincount(grp, weights=angles, minlength=ng) / np.bincount(grp, minlength=ng)
    tot = np.array([np.nansum(f) for f in bc])
    keep = tot != 0
    if return_groups:
        return bc[keep], ba[keep], grp, keep
    return bc[keep], ba[keep]


# ----------------------------------------------------------------------------
# spatial high-pass (IDL smooth() unsharp mask)
# ----------------------------------------------------------------------------
def highpass(img: np.ndarray, width: int, nan_aware: bool = False) -> np.ndarray:
    """``img - smooth(img, width)`` with IDL semantics: boxcar of odd width (even
    widths are bumped by one); pixels within ``width//2`` of the edge are left
    unchanged by ``smooth`` so their high-passed value is 0.  ``nan_aware``
    mirrors ``smooth(/nan)``.  ``width <= 1`` returns the input."""
    w = int(width)
    if w <= 1:
        return np.asarray(img, np.float32)
    if w % 2 == 0:
        w += 1
    img = np.asarray(img, np.float64)
    if nan_aware:
        fin = np.isfinite(img)
        num = ndimage.uniform_filter(np.where(fin, img, 0.0), w, mode="constant")
        den = ndimage.uniform_filter(fin.astype(float), w, mode="constant")
        with np.errstate(invalid="ignore", divide="ignore"):
            sm = np.where(den > 0, num / den, np.nan)
    else:
        sm = ndimage.uniform_filter(img, w, mode="constant")
    h = w // 2
    out = img - sm
    out[:h, :] = 0.0
    out[-h:, :] = 0.0
    out[:, :h] = 0.0
    out[:, -h:] = 0.0
    # NaN pixels stay NaN (smooth without /nan propagates them anyway)
    out[~np.isfinite(img)] = np.nan
    return out.astype(np.float32)


# ----------------------------------------------------------------------------
# row / column destriping (reduce_near_2 L1672-1677: destripe(img, 90) then destripe(img, 0))
# ----------------------------------------------------------------------------
def destripe(img: np.ndarray, angle_deg: float, clip_level: float = 0.0, max_iter: int = 5) -> np.ndarray:
    """Subtract a robust offset from every stripe of ``img`` along ``angle_deg``.

    ``destripe.pro`` is not available; this is the standard robust median
    destriper it is assumed to be (**assumption**, documented here): ``angle=90``
    subtracts from each *column* the median of its finite pixels, ``angle=0`` does
    the same for each *row*.  ``clip_level > 0`` sigma-clips the stripe first
    (iterated ``max_iter`` times, sigma = 1.4826 MAD about the running median) so
    bright sources do not bias the offset; ``clip_level = 0`` (the reduce_near_2
    call) is the plain NaN-median.  Only multiples of 90 degrees are supported.
    NaNs are preserved.
    """
    img = np.asarray(img, np.float64)
    a = float(angle_deg) % 180.0
    if abs(a - 90.0) < 1e-6:
        axis = 0            # per column: median over rows
    elif a < 1e-6:
        axis = 1            # per row: median over columns
    else:
        raise ValueError("destripe: only angles of 0 or 90 degrees are supported")
    work = img.copy()
    with np.errstate(all="ignore"):
        med = np.nanmedian(work, axis=axis, keepdims=True)
        if clip_level > 0:
            for _ in range(int(max_iter)):
                dev = np.abs(work - med)
                mad = np.nanmedian(dev, axis=axis, keepdims=True) * 1.4826
                bad = dev > clip_level * np.where(mad > 0, mad, np.inf)
                if not bad.any():
                    break
                work = np.where(bad, np.nan, work)
                new = np.nanmedian(work, axis=axis, keepdims=True)
                new = np.where(np.isfinite(new), new, med)
                if np.allclose(new, med, equal_nan=True):
                    med = new
                    break
                med = new
    med = np.where(np.isfinite(med), med, 0.0)
    return (img - med).astype(np.float32)


# ----------------------------------------------------------------------------
# rotation
# ----------------------------------------------------------------------------
#: How NaN input pixels enter the bilinear stencil of :func:`rotate_ccw`.
#: ``"zero"`` (default) reproduces IDL ``rot(img, -a, /interp)`` as measured on the NEAR
#: production images: a NaN neighbour contributes 0, so a pixel whose stencil is partly
#: outside the KLIP zone is damped toward 0 instead of dropped (the zone rim stays
#: smooth, ~1 px of "coverage" is gained at the rim); the output is NaN only where no
#: finite input pixel contributes.  ``"propagate"`` NaNs any pixel whose stencil touches
#: a NaN (strict; gives a noisy 1-2 px rim of few-frame coverage after derotation).
ROT_NAN_MODE = "zero"


def rotate_ccw(img: np.ndarray, angle_deg: float, cval: float = np.nan, order: int = 1,
               nan_mode: Optional[str] = None) -> np.ndarray:
    """Rotate an image counter-clockwise by ``angle_deg`` about ``((nx-1)/2, (ny-1)/2)``
    with bilinear interpolation (IDL ``rot(img, -angle, /interp)``).  NaN handling per
    ``nan_mode`` (default :data:`ROT_NAN_MODE`); pixels mapping outside the input are
    ``cval`` (NaN by default; IDL leaves them 0 and the callers mask them by coverage)."""
    img = np.asarray(img, float)
    if angle_deg == 0.0:
        return img.copy()
    mode = nan_mode or ROT_NAN_MODE
    ny, nx = img.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx, dy = xx - cx, yy - cy
    xs = cx + ca * dx + sa * dy
    ys = cy - sa * dx + ca * dy
    fin = np.isfinite(img)
    out = ndimage.map_coordinates(np.where(fin, img, 0.0), [ys, xs], order=order, mode="constant", cval=0.0)
    # stencil validity: 1 where every contributing input pixel is finite and inside
    w = ndimage.map_coordinates(fin.astype(float), [ys, xs], order=order, mode="constant", cval=0.0)
    inside = (xs >= 0) & (xs <= nx - 1) & (ys >= 0) & (ys <= ny - 1)
    if mode == "propagate":
        out[(w < 1.0 - 1e-6) & inside] = np.nan
    else:                                  # "zero": NaN neighbours count as 0, no-coverage -> NaN
        out[(w <= 1e-6) & inside] = np.nan
    out[~inside] = cval
    return out


def derotate(cube: np.ndarray, angles: np.ndarray, truenorth: float = 0.0) -> np.ndarray:
    """Rotate every frame CCW by ``angle + truenorth`` (sky North up after this)."""
    out = np.empty_like(cube, dtype=np.float32)
    for i in range(cube.shape[0]):
        out[i] = rotate_ccw(cube[i], float(angles[i]) + truenorth)
    return out


# ----------------------------------------------------------------------------
# noise-weighted ADI combination (nw_ang_comb.pro)
# ----------------------------------------------------------------------------
def nw_ang_comb(derot_cube: np.ndarray, angles: np.ndarray, ref_cube: Optional[np.ndarray] = None) -> np.ndarray:
    """Coverage-weighted inverse-variance mean of an already-derotated cube.  The
    per-pixel temporal variance is measured in the pupil frame (frames rotated back
    by ``angles``), floored at ``median(var>0)*1e-6``, rotated forward again and
    used as the weight; frames that are NaN at a pixel get zero weight there.

    ``ref_cube`` (``nw_ang_comb_ref.pro``): measure the variance -- i.e. the
    weights -- on this derotated cube (the SCIENCE residuals) instead of on the
    input, so a noiseless model cube is averaged with exactly the weights the
    science frames received (the linear forward model of the nwadi combine)."""
    n = derot_cube.shape[0]
    if n == 1:
        return np.asarray(derot_cube[0], float)
    src = derot_cube if ref_cube is None else ref_cube
    A = np.empty(derot_cube.shape, np.float64)
    for i in range(n):
        A[i] = rotate_ccw(np.where(np.isfinite(derot_cube[i]), derot_cube[i], 0.0), -float(angles[i]), cval=0.0)
    A[~np.isfinite(A)] = 0.0
    if ref_cube is None:
        V = A
    else:
        V = np.empty(derot_cube.shape, np.float64)
        for i in range(n):
            V[i] = rotate_ccw(np.where(np.isfinite(src[i]), src[i], 0.0), -float(angles[i]), cval=0.0)
        V[~np.isfinite(V)] = 0.0
    var = np.var(V, axis=0, ddof=1)
    pos = var[var > 0]
    vfloor = float(np.median(pos)) * 1e-6 if pos.size else 1e-30
    var = np.maximum(var, vfloor)
    num = np.zeros(A.shape[1:])
    den = np.zeros(A.shape[1:])
    cov = np.isfinite(derot_cube)
    for i in range(n):
        Ai = rotate_ccw(A[i], float(angles[i]), cval=0.0)
        Vi = np.maximum(rotate_ccw(var, float(angles[i]), cval=0.0), vfloor)
        Ai[~np.isfinite(Ai)] = 0.0
        Vi[~np.isfinite(Vi)] = vfloor
        inv = cov[i] / Vi
        num += Ai * inv
        den += inv
    out = num / np.maximum(den, 1e-30)
    out[den <= 0] = np.nan
    return out


def nw_ang_comb_ref(in_cube: np.ndarray, angles: np.ndarray, ref_cube: np.ndarray) -> np.ndarray:
    """``nw_ang_comb_ref.pro``: combine ``in_cube`` (e.g. derotated KLIP-FM
    residuals) with the per-pixel inverse-variance weights of ``ref_cube`` (the
    derotated science residuals).  Coverage comes from ``in_cube``."""
    return nw_ang_comb(in_cube, angles, ref_cube=ref_cube)


# ----------------------------------------------------------------------------
# KLIP core
# ----------------------------------------------------------------------------
def klip_basis(R: np.ndarray, k: int, spat_mean: bool = False, temp_mean: bool = False,
               tol: float = 1e-6) -> np.ndarray:
    """KL basis of the reference matrix ``R`` (M frames x N pixels), Soummer et al.
    2012 as in ``get_klip_basis_new``: eigen-decompose ``R R^T`` (float64), drop
    eigenvalues below ``tol``, ``Z_m = R^T v_m / sqrt(lambda_m)``.  Returns the first
    ``min(k, M)`` modes as rows (modes with zero eigenvalue are zero rows)."""
    R = np.asarray(R, np.float64)
    if R.ndim == 1:
        R = R[None, :]
    if spat_mean:
        R = R - R.mean(axis=1, keepdims=True)
    if temp_mean:
        R = R - R.mean(axis=0, keepdims=True)
    M = R.shape[0]
    k = int(min(max(k, 1), M))
    C = R @ R.T
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    lam = np.where(lam < tol, 0.0, lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(lam > 0, 1.0 / np.sqrt(np.where(lam > 0, lam, 1.0)), 0.0)
    Z = (V * scale[None, :]).T @ R           # (M, N)
    return Z[:k]


def zone_indices(shape: Tuple[int, int], inrad: float, outrad: float, a0_deg: float, a1_deg: float,
                 center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Flat indices of the pixels with ``inrad <= r <= outrad`` and ``a0 <= ang <= a1``
    (inclusive, degrees CCW from +x).  ``center`` defaults to ``(nx/2, ny/2)`` -- the
    IDL ``getzone`` convention (half a pixel off the star for even sizes; kept for
    fidelity, only the zone mask is affected)."""
    ny, nx = shape
    if center is None:
        center = (nx / 2.0, ny / 2.0)
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - center[0], yy - center[1])
    ang = np.rad2deg(np.arctan2(yy - center[1], xx - center[0])) % 360.0
    m = (r >= inrad) & (r <= outrad) & (ang >= a0_deg) & (ang <= a1_deg)
    return np.flatnonzero(m.ravel())


def arcdist_deg(inrad: float, outrad: float, lam_over_d_px: float) -> float:
    """Degrees of parallactic rotation per unit ``angsep`` (one lambda/D of arc at the
    annulus mid-radius)."""
    mid = inrad + abs(outrad - inrad) / 2.0
    mid = max(mid, 1.0)
    return 360.0 * lam_over_d_px / (2.0 * np.pi * mid)


def reference_mask(angles: np.ndarray, target: int, angsep_deg: float, anglemax: float) -> np.ndarray:
    dpa = np.abs(angles - angles[target])
    m = (dpa >= angsep_deg) & (dpa <= anglemax)
    m[target] = False
    return m


@dataclass
class KLIPParams:
    k_klip: int = 10
    inrad: float = 0.0
    outrad: float = 70.0
    n_ang: int = 1
    fast: bool = False
    angsep: float = 0.0          # in lambda/D of arc at the annulus mid radius
    anglemax: float = 360.0      # deg
    n_min_ref: int = 10
    spat_mean: bool = False
    temp_mean: bool = False
    k_scan: bool = False
    zone_center: Optional[Tuple[float, float]] = None
    threads: int = 1             # per-target work of the slow path spread over this many threads


_FM_TOL = 1e-6                   # eigenvalue tolerance of get_klip_basis_new / the FM block


def klip_fm_zone(R: np.ndarray, Mref: Optional[np.ndarray], T: np.ndarray, Mtar: np.ndarray, k: int,
                 fm_selfsub: bool = True) -> np.ndarray:
    """KLIP-FM (Pueyo 2016, first order) of one zone -- ``multiklip.pro`` L365-403
    (fast path) / L494-527 (per-target path), same math.

    ``R``    (nref, npx)  raw basis-building frames (NaN already zeroed);
    ``Mref`` (nref, npx)  the planet model as it appears in those frames, or None
                          (planet-free reference library -> no basis perturbation);
    ``T``    (ntar, npx)  the target (stellar) frames as projected by KLIP;
    ``Mtar`` (ntar, npx)  the model in the target frames.

    Returns ``FM = M - sum_k [<M,Z_k>Z_k + <S,dZ_k>Z_k + <S,Z_k>dZ_k]`` (the two
    perturbation terms only when ``fm_selfsub``), with ``Z_k = R^T u_k / sqrt(l_k)``
    and the first-order eigen-perturbation from ``dC = R M^T + M R^T``::

        dl_k  = u_k . dC u_k
        du_k  = sum_{j != k, |l_k - l_j| > 1e-6 max(|l_k|, 1)} (u_j . dC u_k) / (l_k - l_j) u_j
        dZ_k  = -dl_k / (2 l_k) Z_k + (R^T du_k + M^T u_k) / sqrt(l_k)
    """
    R = np.asarray(R, np.float64)
    T = np.asarray(T, np.float64)
    Mtar = np.asarray(Mtar, np.float64)
    nref = R.shape[0]
    C = R @ R.T
    lam, U = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    nmod = int(min(max(k, 0), nref))
    if nmod == 0:
        return Mtar.copy()
    valid = lam > _FM_TOL
    scale = np.where(valid, 1.0 / np.sqrt(np.where(valid, lam, 1.0)), 0.0)
    Zb = ((U * scale[None, :]).T @ R)[:nmod]                    # (nmod, npx)
    dZb = np.zeros_like(Zb)
    if Mref is not None and fm_selfsub:
        Mref = np.asarray(Mref, np.float64)
        G = R @ Mref.T
        dA = G + G.T                                            # symmetric dC
        Tk = dA @ U                                             # column k2 = dC u_k
        proj = U.T @ Tk                                         # proj[j, k2] = u_j . dC u_k
        idx = np.arange(nref)
        for k2 in range(nmod):
            if lam[k2] <= _FM_TOL:
                continue
            u_k = U[:, k2]
            dlam = proj[k2, k2]
            ddl = lam[k2] - lam
            ok = (idx != k2) & (np.abs(ddl) > _FM_TOL * max(abs(lam[k2]), 1.0))
            coef = np.where(ok, proj[:, k2] / np.where(ok, ddl, 1.0), 0.0)
            du = U @ coef
            term = du @ R + u_k @ Mref
            dZb[k2] = -dlam / (2.0 * lam[k2]) * Zb[k2] + term / np.sqrt(lam[k2])
    sM = Mtar @ Zb.T                                            # <M_i, Z_k>
    FM = Mtar - sM @ Zb                                         # over-subtraction
    if fm_selfsub and Mref is not None:
        sS = T @ Zb.T                                           # <S_i, Z_k>
        aS = T @ dZb.T                                          # <S_i, dZ_k>
        FM = FM - aS @ Zb - sS @ dZb                            # self-subtraction (perturbation)
    return FM


def klip_annular(cube: np.ndarray, angles: np.ndarray, p: KLIPParams, lam_over_d_px: float,
                 log=None, fm_cube: Optional[np.ndarray] = None, ref_cube: Optional[np.ndarray] = None,
                 fm_ref_cube: Optional[np.ndarray] = None, fm_selfsub: bool = True):
    """Annular KLIP of a (binned, high-passed) cube.

    fast=True   : one basis per zone from all frames, all frames projected;
    fast=False  : per-target basis from the frames with
                  ``angsep*arcdist <= |dPA| <= anglemax`` (fewer than 4 -> all
                  other frames, the ``multiklip`` fallback).  Targets with fewer
                  than ``n_min_ref`` references are dropped (NaN frame) unless that
                  would leave fewer than ``max(n_min_ref, ceil(0.25 n))`` targets, in
                  which case dropping is disabled (``reduce_near_2`` safety floor).
    k_scan=True : residuals for k = 1..k_klip; result shape ``(k, n, ny, nx)``.
    k_klip=0    : no subtraction (residual = input inside the zones).

    ref_cube    : (RDI / ARDI, ``multiklip style='rdi'``) build the basis from these
                  frames instead of the science cube -- for ARDI pass the
                  concatenation ``[ref, science]``.  Forces the single-basis path.
    fm_cube     : KLIP-FM planet-model cube, frame-aligned with ``cube``; the
                  forward-modelled residuals are returned as a third element.
    fm_ref_cube : the model as it appears in ``ref_cube`` (zeros for planet-free
                  reference frames); when ``ref_cube`` is None the science model
                  is reused (ADI).  ``fm_selfsub=False`` keeps only the
                  over-subtraction term (the ``fm_selfsub_g`` diagnostic toggle).
                  FM is not supported together with ``k_scan``.

    Pixels outside the zones are NaN.  Returns ``(residuals, info)``, or
    ``(residuals, info, fm_residuals)`` when ``fm_cube`` is given.  ``info['nref']``
    is the per-target reference count (census window, before the <4 fallback),
    ``info['nref_used']`` the count actually used, ``info['dropped']`` the mask of
    NaN'd targets.
    """
    n, ny, nx = cube.shape
    angles = np.asarray(angles, float)
    k = int(max(p.k_klip, 0))
    fast = bool(p.fast)
    do_fm = fm_cube is not None
    if do_fm and p.k_scan:
        raise ValueError("KLIP-FM is fixed-k: not supported with k_scan")
    if do_fm and fm_cube.shape != cube.shape:
        raise ValueError("fm_cube must be frame-aligned with cube")
    rdi = ref_cube is not None
    if rdi:
        if ref_cube.shape[1:] != (ny, nx):
            raise ValueError("ref_cube must have the science frame size")
        fast = True
    paspan = float(angles.max() - angles.min()) if n >= 2 else 0.0
    auto_fast = False
    if not fast and p.angsep <= 0 and p.anglemax >= paspan:
        fast, auto_fast = True, True
    arc = arcdist_deg(p.inrad, p.outrad, lam_over_d_px)
    angsep_deg = p.angsep * abs(arc)
    info = {"fast": fast, "auto_fast": auto_fast, "arcdist": arc, "n_frames": n, "n_dropped": 0,
            "rdi": rdi, "n_basis_frames": int(ref_cube.shape[0]) if rdi else n, "fm": do_fm,
            "fm_selfsub": bool(fm_selfsub) if do_fm else None}

    shape_out = (k, n, ny, nx) if p.k_scan else (n, ny, nx)
    out = np.full(shape_out, np.nan, np.float32)
    flat = cube.reshape(n, ny * nx)
    fm_flat = fm_cube.reshape(n, ny * nx) if do_fm else None
    fm_out = np.full((n, ny, nx), np.nan, np.float32) if do_fm else None
    ref_flat = ref_cube.reshape(ref_cube.shape[0], ny * nx) if rdi else None
    fmref_flat = fm_ref_cube.reshape(fm_ref_cube.shape[0], ny * nx) if (rdi and fm_ref_cube is not None) else None

    zones = []
    for jj in range(int(max(p.n_ang, 1))):
        a0 = jj * 360.0 / p.n_ang
        a1 = min((jj + 1) * 360.0 / p.n_ang, 360.0)
        idx = zone_indices((ny, nx), p.inrad, p.outrad, a0, a1, p.zone_center)
        if idx.size:
            zones.append(idx)

    def _basis(R):
        if k == 0:
            return np.zeros((0, R.shape[1]))
        return klip_basis(R, k, p.spat_mean, p.temp_mean)

    def _project(T, Z, kk):
        # T (t, N), Z (m, N) orthonormal rows -> residuals for modes 0..kk-1
        if kk == 0:
            return T.copy()
        S = T @ Z[:kk].T
        return T - S @ Z[:kk]

    def _zero_nan(A):
        return np.where(np.isfinite(A), A, 0.0)

    if fast:
        nbasis = ref_flat.shape[0] if rdi else n
        info["nref"] = np.full(n, nbasis - (0 if rdi else 1))
        info["nref_used"] = info["nref"].copy()
        info["dropped"] = np.zeros(n, bool)
        for idx in zones:
            D = _zero_nan(flat[:, idx].astype(np.float64))          # science frames in the zone
            R = _zero_nan(ref_flat[:, idx].astype(np.float64)) if rdi else D
            Z = _basis(R)
            T = D.copy()
            if p.spat_mean:
                T -= T.mean(axis=1, keepdims=True)
            if p.temp_mean:
                T -= R.mean(axis=0, keepdims=True)
            kk_max = min(Z.shape[0], k)
            if p.k_scan:
                S = T @ Z.T
                for kk in range(kk_max):
                    res = T - S[:, :kk + 1] @ Z[:kk + 1]
                    o = out[kk].reshape(n, ny * nx)
                    o[:, idx] = res
            else:
                res = _project(T, Z, kk_max)
                o = out.reshape(n, ny * nx)
                o[:, idx] = res
            if do_fm:
                Mtar = _zero_nan(fm_flat[:, idx].astype(np.float64))
                if not rdi:
                    Mref = Mtar                                   # ADI: refs == science frames
                elif fmref_flat is not None:
                    Mref = _zero_nan(fmref_flat[:, idx].astype(np.float64))
                else:
                    Mref = None                                   # RDI without a ref model: no self-sub
                fo = fm_out.reshape(n, ny * nx)
                fo[:, idx] = klip_fm_zone(R, Mref, T, Mtar, k, fm_selfsub)
        return (out, info, fm_out) if do_fm else (out, info)

    # ---- per-target (slow) path -------------------------------------------
    nref = np.array([reference_mask(angles, t, angsep_deg, p.anglemax).sum() for t in range(n)])
    nstarv = int((nref < p.n_min_ref).sum())
    do_drop = (n - nstarv) >= max(p.n_min_ref, int(np.ceil(0.25 * n)))
    info["n_starved"] = nstarv
    info["drop_enabled"] = bool(do_drop)
    info["nref"] = nref
    nref_used = nref.copy()
    dropped = np.zeros(n, bool)
    zone_data = [flat[:, idx].astype(np.float64) for idx in zones]

    def _target(t):
        if do_drop and nref[t] < p.n_min_ref:
            dropped[t] = True
            return                        # frame stays NaN
        refs = reference_mask(angles, t, angsep_deg, p.anglemax)
        if refs.sum() < 4:
            refs = np.ones(n, bool)
            refs[t] = False
        nref_used[t] = int(refs.sum())
        for idx, D in zip(zones, zone_data):
            R = _zero_nan(D[refs])
            Z = _basis(R)
            T = _zero_nan(D[t:t + 1].copy())
            Star = T.copy()               # raw target (the FM uses the un-adjusted stellar frame, L493)
            if p.spat_mean:
                T -= T.mean()
            if p.temp_mean:
                T -= R.mean(axis=0, keepdims=True)
            kk_max = min(Z.shape[0], k)
            if p.k_scan:
                S = T @ Z.T
                for kk in range(kk_max):
                    out[kk, t].reshape(-1)[idx] = (T - S[:, :kk + 1] @ Z[:kk + 1])[0]
            else:
                out[t].reshape(-1)[idx] = _project(T, Z, kk_max)[0]
            if do_fm:
                Mref = _zero_nan(fm_flat[refs][:, idx].astype(np.float64))
                Mtar = _zero_nan(fm_flat[t:t + 1, idx].astype(np.float64))
                fm_out[t].reshape(-1)[idx] = klip_fm_zone(R, Mref, Star, Mtar, k, fm_selfsub)[0]

    nthr = int(max(getattr(p, "threads", 1), 1))
    if nthr > 1 and n > 1:
        # targets write disjoint frames of ``out`` -> thread safe; BLAS pinned to 1 thread each
        from .parallel import single_blas_thread, target_pool
        with single_blas_thread():
            list(target_pool(nthr).map(_target, range(n)))
    else:
        for t in range(n):
            _target(t)
    info["n_dropped"] = int(dropped.sum())
    info["nref_used"] = nref_used
    info["dropped"] = dropped
    return (out, info, fm_out) if do_fm else (out, info)
