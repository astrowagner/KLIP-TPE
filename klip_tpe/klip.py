"""KLIP and ADI engine: frame selection, angle-aware binning, high-pass filtering, annular KLIP,
derotation and noise-weighted combination.

Cubes are ``(n, ny, nx)`` for n frames of ny by nx pixels, and images ``(ny, nx)``.  The star is
at ``((nx-1)/2, (ny-1)/2)``, angles are in degrees, and rotations are counter-clockwise with x to
the right and y up.

Notes
-----
These routines succeed the IDL predecessor's ``multiklip.pro``, ``get_klip_basis_new.pro`` and
the processing chain of ``reduce_near_2.pro``.  That code rotates with IDL's ``rot(img, a)``,
which turns clockwise by ``a``, so its ``rot(img, -a)`` is :func:`rotate_ccw` ``(img, a)``.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

__all__ = ["frame_selection_mask", "bin_frames", "bin_groups", "bin_by_groups", "bin_angles", "bin_values",
           "highpass", "highpass_stack", "destripe", "rotate_ccw", "rotate_ccw_many", "derotate",
           "group_mean", "group_bounds", "bins_with_data", "rotation_backend", "set_rotation_backend",
           "rotation_info",
           "nw_ang_comb", "nw_ang_comb_ref", "klip_basis", "zone_indices", "klip_annular", "klip_fm_zone",
           "arcdist_deg", "reference_mask", "KLIPParams"]


# ----------------------------------------------------------------------------
# frame blocks on the job's threads
# ----------------------------------------------------------------------------
#: float64 bytes one block of frames may take per temporary array: about 5 frames of
#: 150 x 150, which stays in cache (larger blocks ran slower than frame by frame)
BLOCK_BYTES = 1_000_000


def frame_blocks(n: int, threads: int = 1, frame_px: int = 0) -> List[Tuple[int, int]]:
    """Split n frames into contiguous blocks for worker threads.

    Use it with :func:`run_blocks`.  There are two blocks per thread, and no block's float64 copy
    exceeds :data:`BLOCK_BYTES`, so a block stays in the CPU cache.

    Parameters
    ----------
    n : int
        Number of frames.  0 or less gives no blocks.
    threads : int, default 1
        Number of threads the blocks will run on.  1 gives one block unless ``frame_px`` caps its
        size.
    frame_px : int, default 0
        Number of pixels in one frame, which caps a block's size.  0 sets no cap.

    Returns
    -------
    blocks : list of tuple of int
        The ``(lo, hi)`` frame ranges, ``lo`` included and ``hi`` excluded, in order and covering
        all n frames.
    """
    n = int(n)
    if n <= 0:
        return []
    k = 1 if threads <= 1 else min(n, 2 * int(threads))
    if frame_px > 0:
        k = max(k, -(-n * int(frame_px) * 8 // BLOCK_BYTES))
    k = min(k, n)
    edges = np.linspace(0, n, k + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def run_blocks(fn, blocks: Sequence[Tuple[int, int]], threads: int = 1) -> None:
    """Call ``fn(lo, hi)`` for every block of frames, on the given number of threads.

    The threads come from the process-wide pool (:func:`klip_tpe.parallel.target_pool`).  Each
    block must write only its own frames.

    Parameters
    ----------
    fn : callable
        Function called as ``fn(lo, hi)`` for frames ``lo`` to ``hi - 1``.  Its return value is
        ignored.
    blocks : sequence of tuple of int
        The ``(lo, hi)`` frame ranges, as :func:`frame_blocks` gives them.
    threads : int, default 1
        Number of threads.  1 runs the blocks in turn in the calling thread.
    """
    blocks = list(blocks)
    if int(threads) <= 1 or len(blocks) <= 1:
        for lo, hi in blocks:
            fn(lo, hi)
        return
    from .parallel import target_pool
    list(target_pool(int(threads)).map(lambda b: fn(*b), blocks))


# ----------------------------------------------------------------------------
# frame selection (near2_framesel)
# ----------------------------------------------------------------------------
def frame_selection_mask(nf: int, tags: Optional[Dict[str, np.ndarray]], corr_thresh: Optional[float],
                         noise_max: Optional[float], coronoise_max: Optional[float],
                         bin_: int, k_klip: int) -> np.ndarray:
    """Choose the frames to keep from per-frame quality tags.

    A frame is kept when its correlation is at least ``corr_thresh``, its background noise at most
    ``noise_max`` times the mean, and its residual noise at most ``coronoise_max`` times the mean.
    A NaN tag does not reject a frame.  The cut is skipped, keeping every frame, when it would keep
    fewer than ``max(bin_ * k_klip, 3)`` frames or reject none.

    Parameters
    ----------
    nf : int
        Number of frames.
    tags : dict of str to ndarray, or None
        Per-frame tags of length ``nf``: ``"corrs"`` (correlation with the reference PSF),
        ``"noises"`` (background noise) and ``"coronoise"`` (coronagraphic or PSF-fit residual
        noise).  None, or ``"corrs"`` missing or of another length, keeps every frame.
    corr_thresh : float or None
        Minimum correlation a frame needs.  None keeps every frame.
    noise_max : float or None
        Maximum background noise, as a multiple of the mean of ``tags["noises"]``.  None applies no
        background-noise cut.
    coronoise_max : float or None
        Maximum residual noise, as a multiple of the mean of ``tags["coronoise"]``.  None applies no
        residual-noise cut.
    bin_ : int
        Temporal binning factor that follows, used only in the floor on the frames kept.
    k_klip : int
        Number of KL modes the reduction uses, used only in the floor on the frames kept.

    Returns
    -------
    keep : ndarray of bool, shape (nf,)
        True for each frame to keep.
    """
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
    """Number the frames into bins of consecutive frames, the work of :func:`bin_groups`.

    A bin closes when it holds ``bin_`` frames or when the next frame would widen its span of
    parallactic angle past ``dth_max``.

    Parameters
    ----------
    angles : ndarray, shape (n,)
        Parallactic angles of the n frames in time order, in degrees.
    bin_ : int
        Maximum number of frames in a bin.
    dth_max : float
        Maximum span of parallactic angle in a bin, in degrees.

    Returns
    -------
    grp : ndarray of int, shape (n,)
        Bin number of each frame, counting from 0.
    """
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
    """Compute the mean parallactic angle of each bin of the angle-aware binning.

    Use it to count the references a binning leaves without binning any frames.  The angles are
    those :func:`bin_frames` computes before it drops the bins without data.

    Parameters
    ----------
    angles : ndarray
        Parallactic angles of the frames in time order, in degrees.
    bin_ : int
        Maximum number of frames per bin.  1 or less returns a copy of ``angles``.
    dth_max_deg : float
        Maximum span of parallactic angle in a bin, in degrees.

    Returns
    -------
    bin_angles : ndarray
        Mean parallactic angle of each bin, in degrees.
    """
    angles = np.asarray(angles, float)
    b = int(round(max(bin_, 1)))
    if b <= 1 or angles.size == 0:
        return angles.copy()
    grp = _bin_groups(angles, b, dth_max_deg)
    ng = grp.max() + 1
    return np.bincount(grp, weights=angles, minlength=ng) / np.bincount(grp, minlength=ng)


def bin_groups(angles: np.ndarray, bin_: int, dth_max_deg: float) -> np.ndarray:
    """Assign each frame its bin number in the angle-aware binning.

    A bin holds at most ``bin_`` consecutive frames and closes early when the next frame would widen
    its span of parallactic angle past ``dth_max_deg``.  Use it with :func:`bin_by_groups` to bin a
    second cube of the same frames as :func:`bin_frames` binned the first.

    Parameters
    ----------
    angles : ndarray, shape (n,)
        Parallactic angles of the n frames in time order, in degrees.
    bin_ : int
        Maximum number of frames per bin.  1 or less puts every frame in a bin of its own.
    dth_max_deg : float
        Maximum span of parallactic angle in a bin, in degrees.

    Returns
    -------
    grp : ndarray of int, shape (n,)
        Bin number of each frame, counting from 0.
    """
    angles = np.asarray(angles, float)
    b = int(round(max(bin_, 1)))
    if b <= 1:
        return np.arange(angles.size)
    return _bin_groups(angles, b, dth_max_deg)


def group_mean(frames: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Average the frames of one bin, as :func:`bin_by_groups` does.

    Parameters
    ----------
    frames : ndarray, shape (m, ny, nx)
        The bin's m frames, ny by nx pixels each.
    dtype : numpy dtype, default numpy.float32
        Precision of the sum and the mean.

    Returns
    -------
    mean : ndarray, shape (ny, nx)
        Mean frame.  A bin of one frame gives that frame itself, not a copy.
    """
    if frames.shape[0] > 1:
        return frames.mean(axis=0, dtype=None if np.dtype(dtype) == np.float32 else dtype)
    return frames[0]


def group_bounds(grp: np.ndarray) -> np.ndarray:
    """Find where each bin starts, from the frames' bin numbers.

    Parameters
    ----------
    grp : ndarray of int, shape (n,)
        Bin number of each of the n frames, as :func:`bin_groups` gives it.  The numbers must not
        decrease along the frames.

    Returns
    -------
    starts : ndarray of int
        ``[s_0, s_1, ..., n]``, the first frame of each bin and then n, so bin ``g`` holds frames
        ``s_g`` to ``s_(g+1) - 1``.  ``[0]`` when there are no frames.

    Raises
    ------
    ValueError
        If the bin numbers decrease somewhere.
    """
    grp = np.asarray(grp)
    if grp.size == 0:
        return np.zeros(1, int)
    if np.any(np.diff(grp) < 0):
        raise ValueError("bin groups must run along the frames")
    starts = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]])
    return np.r_[starts, grp.size].astype(int)


def bin_by_groups(cube: np.ndarray, grp: np.ndarray, threads: int = 1, dtype=np.float32) -> np.ndarray:
    """Average the frames of each bin into one frame.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The n frames, ny by nx pixels each.
    grp : ndarray of int, shape (n,)
        Bin number of each frame, as :func:`bin_groups` gives it.
    threads : int, default 1
        Number of threads the bins are spread over.
    dtype : numpy dtype, default numpy.float32
        Precision of the means.  float64 keeps frames that still carry a large background accurate.

    Returns
    -------
    binned : ndarray, shape (nbin, ny, nx)
        Mean frame of each of the nbin bins, in ``dtype``.
    """
    ng = int(grp.max()) + 1 if grp.size else 0
    bc = np.empty((ng,) + cube.shape[1:], dtype)

    def block(lo, hi):
        for g in range(lo, hi):
            w = np.flatnonzero(grp == g)
            bc[g] = group_mean(cube[w], dtype)
    run_blocks(block, frame_blocks(ng, threads), threads)
    return bc


def bins_with_data(bc: np.ndarray) -> np.ndarray:
    """Find the frames that hold data, those with a pixel that is neither NaN nor 0.

    The binning drops the frames that hold none.  The test looks at the pixels rather than the
    frame's sum, since a high-passed frame can sum to exactly 0 in float32.

    Parameters
    ----------
    bc : ndarray, shape (n, ny, nx)
        The n binned frames, ny by nx pixels each.

    Returns
    -------
    has_data : ndarray of bool, shape (n,)
        True for each frame with data.

    Notes
    -----
    Until 2026-11-10 the test was the frame's ``nansum``, as the IDL predecessor
    (``reduce_near_2.pro`` L1747-1761, ``bads``) tests its ``total``.  On the WISPIT 2 ALES search
    a bin's sum landed on 0 in 1 draw of 3000, and the injected cube lost a bin the clean cube kept.
    """
    out = np.zeros(len(bc), bool)
    for i, f in enumerate(bc):
        f = np.asarray(f)
        out[i] = bool(np.any(f[~np.isnan(f)] != 0))
    return out


def bin_frames(cube: np.ndarray, angles: np.ndarray, bin_: int, dth_max_deg: float,
               return_groups: bool = False, threads: int = 1, then=None, dtype=np.float32, bkeep=None):
    """Average consecutive frames into bins, then drop the bins without data.

    A bin holds at most ``bin_`` frames and closes early when the next frame would widen its span of
    parallactic angle past ``dth_max_deg``.  Use it to cut the number of frames before KLIP.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The n frames in time order, ny by nx pixels each.
    angles : ndarray, shape (n,)
        Parallactic angle of each frame, in degrees.
    bin_ : int
        Maximum number of frames per bin.  1 or less keeps every frame, as float32.
    dth_max_deg : float
        Maximum span of parallactic angle in a bin, in degrees.
    return_groups : bool, default False
        Whether to also return each frame's bin number and the mask of the bins kept, to bin a
        second cube of the same frames the same way.
    threads : int, default 1
        Number of threads the bins are spread over.
    then : callable or None, default None
        Function applied to the binned cube before the bins without data are found, such as a
        high-pass filter.
    dtype : numpy dtype, default numpy.float32
        Precision of the bin means.
    bkeep : ndarray of bool or None, default None
        Bins to keep in place of the test for data, such as those another cube of the same frames
        kept.  None keeps the bins with data (:func:`bins_with_data`).

    Returns
    -------
    binned : ndarray, shape (nkeep, ny, nx)
        The nkeep bins kept.
    binned_angles : ndarray, shape (nkeep,)
        Mean parallactic angle of each bin kept, in degrees.
    grp : ndarray of int, shape (n,)
        [return_groups=True] Bin number of each input frame.
    keep : ndarray of bool
        [return_groups=True] True for each bin kept, one value per bin before the drop.

    Raises
    ------
    ValueError
        If ``bkeep`` does not have one value per bin.
    """
    angles = np.asarray(angles, float)
    grp = bin_groups(angles, bin_, dth_max_deg)
    if int(round(max(bin_, 1))) <= 1:
        bc, ba = np.asarray(cube, np.float32), angles.copy()
    else:
        bc = bin_by_groups(cube, grp, threads=threads, dtype=dtype)
        ng = grp.max() + 1
        ba = np.bincount(grp, weights=angles, minlength=ng) / np.bincount(grp, minlength=ng)
    if then is not None:
        bc = then(bc)
    if bkeep is None:
        keep = bins_with_data(bc)
    else:
        keep = np.asarray(bkeep, bool)
        if keep.shape != (bc.shape[0],):
            raise ValueError(f"bkeep has {keep.size} bins, the binning {bc.shape[0]}")
    if return_groups:
        return bc[keep], ba[keep], grp, keep
    return bc[keep], ba[keep]


def bin_values(values: np.ndarray, grp: np.ndarray, bin_: int) -> np.ndarray:
    """Average a per-frame quantity over the bins, as :func:`bin_frames` averages the angles.

    Use it for each frame's true north, so a bin is derotated by its mean angle plus its mean north.
    The result has one value per bin before the bins without data are dropped, so index it with the
    ``keep`` that :func:`bin_frames` returns.

    Parameters
    ----------
    values : ndarray, shape (n,)
        The quantity's value for each of the n frames.
    grp : ndarray of int, shape (n,)
        Bin number of each frame.
    bin_ : int
        Binning factor.  1 or less returns a copy of ``values``.

    Returns
    -------
    means : ndarray
        Mean value of each bin.
    """
    v = np.asarray(values, float)
    if int(round(max(bin_, 1))) <= 1:
        return v.copy()
    ng = int(np.asarray(grp).max()) + 1
    return np.bincount(grp, weights=v, minlength=ng) / np.bincount(grp, minlength=ng)


# ----------------------------------------------------------------------------
# spatial high-pass (the unsharp mask the IDL predecessor makes with smooth())
# ----------------------------------------------------------------------------
def highpass(img: np.ndarray, width: int, nan_aware: bool = False) -> np.ndarray:
    """Subtract a boxcar-smoothed copy of an image from it, an unsharp mask.

    It removes structure wider than about ``width`` pixels, such as the stellar halo, before KLIP.
    The boxcar width is made odd, and the pixels within ``width // 2`` of the edges come out 0.

    Parameters
    ----------
    img : ndarray, shape (ny, nx) or (n, ny, nx)
        One image, or a stack of n images filtered one by one.
    width : int
        Width of the boxcar in pixels, raised by one when even.  1 or less returns the input
        unfiltered, as float32.
    nan_aware : bool, default False
        Whether to leave NaN pixels out of the boxcar average.  False lets a NaN spread through the
        average around it, and either way a NaN pixel stays NaN.

    Returns
    -------
    filtered : ndarray
        Filtered image, the shape of ``img``, in float32.
    """
    w = int(width)
    if w <= 1:
        return np.asarray(img, np.float32)
    return _highpass_frames(img, w, nan_aware)


def _highpass_frames(img: np.ndarray, width: int, nan_aware: bool) -> np.ndarray:
    """Filter one frame or a block of frames as :func:`highpass` does.

    Parameters
    ----------
    img : ndarray, shape (ny, nx) or (n, ny, nx)
        One frame, or a block of n frames filtered one by one.
    width : int
        Width of the boxcar in pixels, at least 2, raised by one when even.
    nan_aware : bool
        Whether to leave NaN pixels out of the boxcar average.

    Returns
    -------
    filtered : ndarray
        Filtered frame or frames, the shape of ``img``, in float32.
    """
    w = int(width)
    if w % 2 == 0:
        w += 1
    img = np.asarray(img, np.float64)
    size = (1,) * (img.ndim - 2) + (w, w)
    if nan_aware:
        fin = np.isfinite(img)
        num = ndimage.uniform_filter(np.where(fin, img, 0.0), size, mode="constant")
        den = ndimage.uniform_filter(fin.astype(float), size, mode="constant")
        with np.errstate(invalid="ignore", divide="ignore"):
            sm = np.where(den > 0, num / den, np.nan)
    else:
        sm = ndimage.uniform_filter(img, size, mode="constant")
    h = w // 2
    out = img - sm
    out[..., :h, :] = 0.0
    out[..., -h:, :] = 0.0
    out[..., :, :h] = 0.0
    out[..., :, -h:] = 0.0
    # NaN pixels stay NaN (smooth without /nan propagates them anyway)
    out[~np.isfinite(img)] = np.nan
    return out.astype(np.float32)


def highpass_stack(cube: np.ndarray, width: int, nan_aware: bool = False, threads: int = 1) -> np.ndarray:
    """Apply :func:`highpass` to every frame of a cube, spread over threads.

    The frames are filtered a block at a time so the threads can run at once.  Each frame comes out
    as :func:`highpass` gives it.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The n frames, ny by nx pixels each.
    width : int
        Width of the boxcar in pixels.  1 or less returns the cube unfiltered, as float32.
    nan_aware : bool, default False
        Whether to leave NaN pixels out of the boxcar average.
    threads : int, default 1
        Number of threads.

    Returns
    -------
    filtered : ndarray, shape (n, ny, nx)
        Filtered cube, in float32.
    """
    cube = np.asarray(cube)
    if int(width) <= 1:
        return np.asarray(cube, np.float32)
    out = np.empty(cube.shape, np.float32)
    if cube.shape[0] == 0:
        return out

    def block(lo, hi):
        out[lo:hi] = _highpass_frames(cube[lo:hi], width, nan_aware)
    run_blocks(block, frame_blocks(cube.shape[0], threads, int(np.prod(cube.shape[1:]))), threads)
    return out


# ----------------------------------------------------------------------------
# row / column destriping (reduce_near_2 L1672-1677: destripe(img, 90) then destripe(img, 0))
# ----------------------------------------------------------------------------
def destripe(img: np.ndarray, angle_deg: float, clip_level: float = 0.0, max_iter: int = 5) -> np.ndarray:
    """Subtract a robust offset from every column or row of an image.

    Use it to remove detector stripes.  With ``angle_deg`` 90 each column loses the median of its
    finite pixels, with 0 each row.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image.  NaN pixels stay NaN.
    angle_deg : float
        Direction of the stripes in degrees, 90 for columns and 0 for rows.  Other multiples of 90
        map onto these, and any other angle raises ValueError.
    clip_level : float, default 0.0
        Clipping threshold in sigma, 1.4826 times the median absolute deviation, for leaving bright
        pixels out of a stripe's median.  0 takes the plain median.
    max_iter : int, default 5
        Maximum number of clipping rounds when ``clip_level`` is positive.

    Returns
    -------
    destriped : ndarray, shape (ny, nx)
        Destriped image, in float32.

    Notes
    -----
    The IDL predecessor (``reduce_near_2.pro`` L1672-1677) calls a ``destripe.pro`` that was not
    available.  This is the standard robust median destriper it is assumed to be.
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
#: ``"zero"`` (default) reproduces the IDL predecessor's ``rot(img, -a, /interp)`` as measured
#: on its NEAR production images: a NaN neighbor contributes 0, so a pixel whose stencil is partly
#: outside the KLIP zone is damped toward 0 instead of dropped (the zone rim stays
#: smooth, ~1 px of "coverage" is gained at the rim); the output is NaN only where no
#: finite input pixel contributes.  ``"propagate"`` NaNs any pixel whose stencil touches
#: a NaN (strict; gives a noisy 1-2 px rim of few-frame coverage after derotation).
ROT_NAN_MODE = "zero"

#: the libraries :func:`rotate_ccw` can rotate with, bilinearly either way
ROTATION_BACKENDS = ("auto", "scipy", "opencv")
_ROT: Dict[str, Any] = {"backend": None, "cv2": None, "why": "", "checked": False}


def _cv_warp(cv2, img: np.ndarray, ca: float, sa: float) -> np.ndarray:
    """Rotate an image with OpenCV's bilinear ``warpAffine``, sampling where :func:`_rotation_coords` does.

    Parameters
    ----------
    cv2 : module
        The imported OpenCV module.
    img : ndarray, shape (ny, nx)
        Image without NaN pixels, which the callers replace.
    ca : float
        Cosine of the rotation angle.
    sa : float
        Sine of the rotation angle.

    Returns
    -------
    rotated : ndarray, shape (ny, nx)
        Rotated image, in float64.  Pixels whose sample falls outside the image are 0.
    """
    ny, nx = img.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    M = np.array([[ca, sa, cx - ca * cx - sa * cy], [-sa, ca, cy + sa * cx - ca * cy]], np.float64)
    out = cv2.warpAffine(np.ascontiguousarray(img, np.float32), M, (nx, ny),
                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0.0)
    return np.asarray(out, np.float64)


def _opencv_check():
    """Check whether OpenCV is installed and rotates as scipy's bilinear interpolation does.

    It rotates a smooth test image with both at three angles and accepts OpenCV when they agree to
    1e-4 of the image's peak.  OpenCV 5 agrees to float32 rounding, and OpenCV 4, which rounds its
    sample positions to 1/32 px, does not.  The answer is worked out once and kept.

    Returns
    -------
    cv2 : module or None
        The OpenCV module when it passes, else None.
    why : str
        Version of OpenCV that passed, or the reason it did not.
    """
    if _ROT["checked"]:
        return _ROT["cv2"], _ROT["why"]
    _ROT["checked"] = True
    try:
        import cv2
    except Exception as exc:              # noqa: BLE001 - absent, or a broken install
        _ROT["cv2"], _ROT["why"] = None, f"OpenCV is not installed ({exc.__class__.__name__})"
        return _ROT["cv2"], _ROT["why"]
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    img = (np.exp(-0.5 * ((xx - 30.3) ** 2 + (yy - 35.7) ** 2) / 9.0)
           + 0.5 * np.exp(-0.5 * ((xx - 20.0) ** 2 + (yy - 41.0) ** 2) / 4.0))
    worst = 0.0
    try:
        for ang in (13.7, -41.2, 97.3):
            a = np.deg2rad(ang)
            ca, sa = float(np.cos(a)), float(np.sin(a))
            xs, ys = _rotation_coords(n, n, ca, sa)
            inside = (xs >= 0) & (xs <= n - 1) & (ys >= 0) & (ys <= n - 1)
            ref = ndimage.map_coordinates(img, [ys, xs], order=1, mode="constant", cval=0.0)
            got = _cv_warp(cv2, img, ca, sa)
            worst = max(worst, float(np.max(np.abs(got - ref)[inside])) / float(img.max()))
    except Exception as exc:              # noqa: BLE001
        _ROT["cv2"], _ROT["why"] = None, f"OpenCV {getattr(cv2, '__version__', '?')} failed the check ({exc!r})"
        return _ROT["cv2"], _ROT["why"]
    if worst > 1e-4:
        _ROT["cv2"], _ROT["why"] = None, (f"OpenCV {cv2.__version__} rotates to {worst:.0e} of the peak: it rounds "
                                          f"the sample positions, as OpenCV 5 does not")
    else:
        _ROT["cv2"], _ROT["why"] = cv2, f"OpenCV {cv2.__version__}"
    return _ROT["cv2"], _ROT["why"]


def set_rotation_backend(name: str = "auto") -> str:
    """Choose the library :func:`rotate_ccw` rotates frames with.

    Both interpolate bilinearly.  OpenCV's ``warpAffine`` is about 5 times faster than scipy's
    ``map_coordinates`` on a 150 x 150 frame and gives the same numbers to float32 rounding.  The
    choice also goes into the environment variable ``KLIP_TPE_ROTATE``, so worker processes started
    later rotate alike, and OpenCV is set to one thread.

    Parameters
    ----------
    name : {"auto", "scipy", "opencv"}, default "auto"
        Library to use.  "opencv" falls back to scipy with a RuntimeWarning when OpenCV is not
        usable, and "auto" takes OpenCV when it passes :func:`_opencv_check`, else scipy.

    Returns
    -------
    backend : str
        Library now in use, "scipy" or "opencv".

    Raises
    ------
    ValueError
        If ``name`` is not one of the three.
    """
    key = str(name or "auto").strip().lower()
    if key not in ROTATION_BACKENDS:
        raise ValueError(f"rotation backend {name!r}: one of {', '.join(ROTATION_BACKENDS)}")
    use = "scipy"
    if key in ("auto", "opencv"):
        cv2, why = _opencv_check()
        if cv2 is not None:
            use = "opencv"
            try:
                cv2.setNumThreads(1)
            except Exception:             # noqa: BLE001
                pass
        elif key == "opencv":
            warnings.warn(f"rotation: {why}; rotating with scipy", RuntimeWarning, stacklevel=2)
    _ROT["backend"] = use
    os.environ["KLIP_TPE_ROTATE"] = use
    return use


def rotation_backend() -> str:
    """Return the library :func:`rotate_ccw` rotates with.

    The first call reads the environment variable ``KLIP_TPE_ROTATE`` ("auto", "scipy" or "opencv",
    default "auto") and passes it to :func:`set_rotation_backend`.

    Returns
    -------
    backend : str
        "scipy" or "opencv".
    """
    if _ROT["backend"] is None:
        set_rotation_backend(os.environ.get("KLIP_TPE_ROTATE", "auto"))
    return _ROT["backend"]


def rotation_info() -> str:
    """Return the rotation library and its version, for a run's record.

    Returns
    -------
    info : str
        Library and version, such as "opencv 5.0.0" or "scipy 1.15.3".
    """
    if rotation_backend() == "opencv" and _ROT["cv2"] is not None:
        return f"opencv {_ROT['cv2'].__version__}"
    import scipy
    return f"scipy {scipy.__version__}"


def rotate_ccw(img: np.ndarray, angle_deg: float, cval: float = np.nan, order: int = 1,
               nan_mode: Optional[str] = None) -> np.ndarray:
    """Rotate an image counter-clockwise about the star with bilinear interpolation.

    The center of rotation is ``((nx-1)/2, (ny-1)/2)``, where klip-tpe keeps the star, and
    counter-clockwise is with x to the right and y up.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, ny by nx pixels.  NaN pixels are handled as ``nan_mode`` says.
    angle_deg : float
        Rotation angle, counter-clockwise, in degrees.  0 returns a copy.
    cval : float, default numpy.nan
        Value of output pixels whose sample falls outside the image.
    order : int, default 1
        Spline order of the interpolation, 1 for bilinear.  Higher orders always go through scipy.
    nan_mode : {"zero", "propagate"} or None, default None
        "zero" counts a NaN neighbor as 0, giving NaN only where no finite pixel contributes.
        "propagate" gives NaN wherever a NaN pixel contributes.  None takes :data:`ROT_NAN_MODE`,
        which is "zero".

    Returns
    -------
    rotated : ndarray, shape (ny, nx)
        Rotated image, in float64.
    """
    img = np.asarray(img, float)
    if angle_deg == 0.0:
        return img.copy()
    mode = nan_mode or ROT_NAN_MODE
    ny, nx = img.shape
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    xs, ys = _rotation_coords(ny, nx, ca, sa)
    fin = np.isfinite(img)
    inside = (xs >= 0) & (xs <= nx - 1) & (ys >= 0) & (ys <= ny - 1)
    cv = _ROT["cv2"] if order == 1 and rotation_backend() == "opencv" else None
    if order == 1 and fin.all():
        # Every bilinear stencil of a point inside the frame lies on finite input pixels,
        # so the validity weight below would be 1 there and mark nothing: skip it.  A
        # finite input is every rotation nw_ang_comb makes, three per frame.
        out = _cv_warp(cv, img, ca, sa) if cv is not None else \
            ndimage.map_coordinates(img, [ys, xs], order=1, mode="constant", cval=0.0)
    elif cv is not None:
        out = _cv_warp(cv, np.where(fin, img, 0.0), ca, sa)
        w = _cv_warp(cv, fin.astype(np.float32), ca, sa)
    else:
        out = ndimage.map_coordinates(np.where(fin, img, 0.0), [ys, xs], order=order, mode="constant", cval=0.0)
        # stencil validity: 1 where every contributing input pixel is finite and inside
        w = ndimage.map_coordinates(fin.astype(float), [ys, xs], order=order, mode="constant", cval=0.0)
    if not (order == 1 and fin.all()):
        if mode == "propagate":
            out[(w < 1.0 - 1e-6) & inside] = np.nan
        else:                                  # "zero": NaN neighbors count as 0, no-coverage -> NaN
            out[(w <= 1e-6) & inside] = np.nan
    out[~inside] = cval
    return out


def rotate_ccw_many(img: np.ndarray, angles_deg: Sequence[float], cval: float = np.nan) -> np.ndarray:
    """Rotate one image by each of several angles, as :func:`rotate_ccw` does one angle at a time.

    A finite image is sampled for all the angles in one interpolation call, and an image with a NaN
    pixel is rotated angle by angle.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, ny by nx pixels.
    angles_deg : sequence of float
        The n rotation angles, counter-clockwise, in degrees.
    cval : float, default numpy.nan
        Value of output pixels whose sample falls outside the image.

    Returns
    -------
    rotated : ndarray, shape (n, ny, nx)
        Rotated images, plane ``j`` equal to ``rotate_ccw(img, angles_deg[j], cval=cval)``.
    """
    img = np.asarray(img, float)
    n = len(angles_deg)
    ny, nx = img.shape
    if n == 0:
        return np.empty((0, ny, nx))
    if not np.isfinite(img).all():
        return np.stack([rotate_ccw(img, angles_deg[j], cval=cval) for j in range(n)])
    ca, sa = np.ones(n), np.zeros(n)
    zero = np.zeros(n, bool)
    for j in range(n):
        ang = angles_deg[j]
        if ang == 0.0:
            zero[j] = True                     # rotate_ccw returns a copy, no interpolation
            continue
        a = np.deg2rad(ang)
        ca[j], sa[j] = np.cos(a), np.sin(a)
    xs, ys = _rotation_coords(ny, nx, ca[:, None, None], sa[:, None, None])
    inside = (xs >= 0) & (xs <= nx - 1) & (ys >= 0) & (ys <= ny - 1)
    cv = _ROT["cv2"] if rotation_backend() == "opencv" else None
    if cv is not None:
        out = np.stack([img if zero[j] else _cv_warp(cv, img, float(ca[j]), float(sa[j])) for j in range(n)])
    else:
        out = ndimage.map_coordinates(img, [ys, xs], order=1, mode="constant", cval=0.0)
    out[~inside] = cval
    if zero.any():
        out[zero] = img
    return out


def _rotation_coords(ny: int, nx: int, ca, sa) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the input position a counter-clockwise rotation samples for each output pixel.

    Parameters
    ----------
    ny : int
        Number of rows in the image.
    nx : int
        Number of columns in the image.
    ca : float or ndarray
        Cosine of the angle, a number for one rotation or an (n, 1, 1) array for n of them.
    sa : float or ndarray
        Sine of the angle, shaped as ``ca``.

    Returns
    -------
    xs : ndarray, shape (ny, nx) or (n, ny, nx)
        The x position sampled for each output pixel.
    ys : ndarray
        The y position sampled for each output pixel, shaped as ``xs``.
    """
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    dx, dy = _centred_grid(ny, nx)
    xs = cx + ca * dx + sa * dy
    ys = cy - sa * dx + ca * dy
    return xs, ys


_GRIDS: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}


def _centred_grid(ny: int, nx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return each pixel's offset from the frame center ``((nx-1)/2, (ny-1)/2)``, cached per shape.

    Parameters
    ----------
    ny : int
        Number of rows in the frame.
    nx : int
        Number of columns in the frame.

    Returns
    -------
    dx : ndarray, shape (ny, nx)
        The x offset of each pixel, read-only.
    dy : ndarray, shape (ny, nx)
        The y offset of each pixel, read-only.
    """
    g = _GRIDS.get((ny, nx))
    if g is None:
        yy, xx = np.mgrid[0:ny, 0:nx]
        g = (xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
        for arr in g:
            arr.setflags(write=False)
        if len(_GRIDS) >= 16:
            _GRIDS.clear()
        _GRIDS[(ny, nx)] = g
    return g


def derotate(cube: np.ndarray, angles: np.ndarray, truenorth=0.0, threads: int = 1) -> np.ndarray:
    """Rotate each frame to north up.

    Frame ``i`` turns counter-clockwise by ``angles[i] + truenorth`` about the star.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The n frames to rotate, ny by nx pixels each.
    angles : ndarray, shape (n,)
        Parallactic angle of each frame, in degrees.
    truenorth : float or ndarray, default 0.0
        True-north offset added to each frame's angle, in degrees.  One value, or one per frame.
    threads : int, default 1
        Number of threads the frames are spread over.

    Returns
    -------
    derot : ndarray, shape (n, ny, nx)
        Derotated frames, in float32.  Pixels rotated in from outside the frame are NaN.

    Raises
    ------
    ValueError
        If ``truenorth`` has one value per frame but not n of them.
    """
    out = np.empty_like(cube, dtype=np.float32)
    per = np.ndim(truenorth) > 0
    if per:
        tn = np.asarray(truenorth, float).ravel()
        if tn.shape[0] != cube.shape[0]:
            raise ValueError(f"derotate: {tn.shape[0]} true-north values for {cube.shape[0]} frames")

    def block(lo, hi):
        for i in range(lo, hi):
            out[i] = rotate_ccw(cube[i], float(angles[i]) + (float(tn[i]) if per else truenorth))
    run_blocks(block, frame_blocks(cube.shape[0], threads), threads)
    return out


# ----------------------------------------------------------------------------
# noise-weighted ADI combination (nw_ang_comb.pro)
# ----------------------------------------------------------------------------
def nw_ang_comb(derot_cube: np.ndarray, angles: np.ndarray, ref_cube: Optional[np.ndarray] = None,
                threads: int = 1) -> np.ndarray:
    """Combine derotated frames into one image, each pixel weighted by its inverse variance.

    This is the noise-weighted ADI combination.  Each pixel's variance over time is measured in the
    pupil frame, floored at 1e-6 of the median variance, and rotated with each frame.  A frame that
    is NaN at a pixel gives no weight there.

    Parameters
    ----------
    derot_cube : ndarray, shape (n, ny, nx)
        The n derotated frames, ny by nx pixels each.
    angles : ndarray, shape (n,)
        Parallactic angle of each frame, in degrees, used to turn the frames back to the pupil
        frame.  A true-north offset common to all frames can be left out.
    ref_cube : ndarray or None, default None
        Derotated cube of the same shape whose variance sets the weights, such as the science
        residuals when combining KLIP-FM model frames.  None weighs ``derot_cube`` by its own
        variance.
    threads : int, default 1
        Number of threads for the rotations.

    Returns
    -------
    image : ndarray, shape (ny, nx)
        Combined image, NaN where no frame has data.  A single frame comes back as it is.
    """
    n = derot_cube.shape[0]
    if n == 1:
        return np.asarray(derot_cube[0], float)
    src = derot_cube if ref_cube is None else ref_cube
    A = np.empty(derot_cube.shape, np.float64)

    def back(cube, dst):
        def block(lo, hi):
            for i in range(lo, hi):
                dst[i] = rotate_ccw(np.where(np.isfinite(cube[i]), cube[i], 0.0), -float(angles[i]), cval=0.0)
        return block
    run_blocks(back(derot_cube, A), frame_blocks(n, threads), threads)
    A[~np.isfinite(A)] = 0.0
    if ref_cube is None:
        V = A
    else:
        V = np.empty(derot_cube.shape, np.float64)
        run_blocks(back(src, V), frame_blocks(n, threads), threads)
        V[~np.isfinite(V)] = 0.0
    var = np.var(V, axis=0, ddof=1)
    pos = var[var > 0]
    vfloor = float(np.median(pos)) * 1e-6 if pos.size else 1e-30
    var = np.maximum(var, vfloor)
    num = np.zeros(A.shape[1:])
    den = np.zeros(A.shape[1:])
    cov = np.isfinite(derot_cube)

    def weigh(i):
        """frame i's weighted image and weight, back in the sky frame"""
        Ai = rotate_ccw(A[i], float(angles[i]), cval=0.0)
        Vi = np.maximum(rotate_ccw(var, float(angles[i]), cval=0.0), vfloor)
        Ai[~np.isfinite(Ai)] = 0.0
        Vi[~np.isfinite(Vi)] = vfloor
        inv = cov[i] / Vi
        return Ai * inv, inv
    if int(threads) <= 1:
        for i in range(n):
            wi, inv = weigh(i)
            num += wi
            den += inv
    else:
        # a span of frames at a time on the threads, then added up in frame order
        span = max(2 * int(threads), int(BLOCK_BYTES // max(A[0].nbytes, 1)))
        for s0 in range(0, n, span):
            s1 = min(s0 + span, n)
            wimg = np.empty((s1 - s0,) + A.shape[1:])
            winv = np.empty((s1 - s0,) + A.shape[1:])

            def fwd(lo, hi, s0=s0, wimg=wimg, winv=winv):
                for i in range(s0 + lo, s0 + hi):
                    wimg[i - s0], winv[i - s0] = weigh(i)
            run_blocks(fwd, frame_blocks(s1 - s0, threads), threads)
            for j in range(s1 - s0):
                num += wimg[j]
                den += winv[j]
    out = num / np.maximum(den, 1e-30)
    out[den <= 0] = np.nan
    return out


def nw_ang_comb_ref(in_cube: np.ndarray, angles: np.ndarray, ref_cube: np.ndarray) -> np.ndarray:
    """Combine one derotated cube with the noise weights of another, as :func:`nw_ang_comb` does.

    Use it to combine derotated KLIP-FM model frames with the weights of the derotated science
    residuals.

    Parameters
    ----------
    in_cube : ndarray, shape (n, ny, nx)
        The n derotated frames to combine.  Its NaN pixels set the coverage.
    angles : ndarray, shape (n,)
        Parallactic angle of each frame, in degrees.
    ref_cube : ndarray, shape (n, ny, nx)
        Derotated frames whose variance sets the weights.

    Returns
    -------
    image : ndarray, shape (ny, nx)
        Combined image.
    """
    return nw_ang_comb(in_cube, angles, ref_cube=ref_cube)


# ----------------------------------------------------------------------------
# KLIP core
# ----------------------------------------------------------------------------
def _eig_desc(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose a symmetric matrix, the largest eigenvalue first.

    Parameters
    ----------
    C : ndarray, shape (m, m)
        Symmetric matrix.

    Returns
    -------
    lam : ndarray, shape (m,)
        Eigenvalues, from the largest to the smallest.
    V : ndarray, shape (m, m)
        Eigenvectors as columns, in the order of ``lam``.
    """
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    return lam[order], V[:, order]


def _basis_scale(lam: np.ndarray, tol: float) -> np.ndarray:
    """Compute each KL mode's normalization ``1 / sqrt(lam)``, 0 for the modes dropped.

    Parameters
    ----------
    lam : ndarray
        Eigenvalues of the references' covariance matrix, the largest first.
    tol : float
        Minimum eigenvalue a mode keeps.  Modes below it get a scale of 0.

    Returns
    -------
    scale : ndarray
        Scale of each mode, one per eigenvalue.
    """
    lam = np.where(lam < tol, 0.0, lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(lam > 0, 1.0 / np.sqrt(np.where(lam > 0, lam, 1.0)), 0.0)


def _klip_basis_eig(R: np.ndarray, k: int, spat_mean: bool = False, temp_mean: bool = False,
                    tol: float = 1e-6, only_k: bool = False):
    """Compute the KL modes of a set of reference frames, with the eigen-decomposition behind them.

    Parameters
    ----------
    R : ndarray, shape (m, p) or (p,)
        Pixels of the m reference frames, p pixels each, or of one frame.
    k : int
        Number of modes to return, clamped to 1 to m.
    spat_mean : bool, default False
        Whether to subtract each frame's mean over its pixels first.
    temp_mean : bool, default False
        Whether to subtract the frames' mean frame first.
    tol : float, default 1e-6
        Minimum eigenvalue a mode keeps.  Modes below it are rows of zeros.
    only_k : bool, default False
        Whether to form only the first ``k`` modes.  False forms all m and keeps the first ``k``,
        the same modes though not always to the bit.

    Returns
    -------
    Z : ndarray, shape (k, p)
        KL modes as rows.
    lam : ndarray
        Eigenvalues of ``R R^T``, the largest first, before the ``tol`` cut.
    V : ndarray
        Eigenvectors of ``R R^T`` as columns, in the order of ``lam``.
    R : ndarray
        Reference frames as decomposed, in float64, with the means subtracted when asked.
    """
    R = np.asarray(R, np.float64)
    if R.ndim == 1:
        R = R[None, :]
    if spat_mean:
        R = R - R.mean(axis=1, keepdims=True)
    if temp_mean:
        R = R - R.mean(axis=0, keepdims=True)
    M = R.shape[0]
    k = int(min(max(k, 1), M))
    lam, V = _eig_desc(R @ R.T)
    scale = _basis_scale(lam, tol)
    if only_k:
        Z = (V[:, :k] * scale[None, :k]).T @ R
    else:
        Z = ((V * scale[None, :]).T @ R)[:k]          # all M modes, the first k kept
    return Z, lam, V, R


def klip_basis(R: np.ndarray, k: int, spat_mean: bool = False, temp_mean: bool = False,
               tol: float = 1e-6) -> np.ndarray:
    """Compute the KL basis of a set of reference frames (Soummer et al. 2012).

    It eigen-decomposes ``R R^T`` in float64, drops the eigenvalues below ``tol`` and forms the
    modes ``Z_j = R^T v_j / sqrt(lam_j)``, where ``v_j`` and ``lam_j`` are the j-th eigenvector
    and eigenvalue.

    Parameters
    ----------
    R : ndarray, shape (m, p)
        Pixels of the m reference frames, p pixels each.
    k : int
        Number of modes to return, clamped to 1 to m.
    spat_mean : bool, default False
        Whether to subtract each frame's mean over its pixels first.
    temp_mean : bool, default False
        Whether to subtract the frames' mean frame first.
    tol : float, default 1e-6
        Minimum eigenvalue a mode keeps.  Modes below it are rows of zeros.

    Returns
    -------
    Z : ndarray, shape (min(k, m), p)
        The first ``min(k, m)`` modes, as rows of unit norm.
    """
    return _klip_basis_eig(R, k, spat_mean, temp_mean, tol)[0]


ZONE_ANGLES = ("ccw", "getzone2015")


def zone_indices(shape: Tuple[int, int], inrad: float, outrad: float, a0_deg: float, a1_deg: float,
                 center: Optional[Tuple[float, float]] = None, angles: str = "ccw") -> np.ndarray:
    """Return the pixels of an annular sector, as flat indices into an image.

    A pixel belongs when its distance ``r`` from ``center`` has ``inrad <= r <= outrad`` and its
    azimuth lies from ``a0_deg`` to ``a1_deg``, both ends included.

    Parameters
    ----------
    shape : tuple of int
        Shape of the image, ``(ny, nx)``.
    inrad : float
        Inner radius of the sector, in pixels.
    outrad : float
        Outer radius of the sector, in pixels.
    a0_deg : float
        First azimuth of the sector, in degrees.
    a1_deg : float
        Last azimuth of the sector, in degrees.
    center : tuple of 2 floats or None, default None
        The ``(x, y)`` position the radii and azimuths are measured from.  None takes
        ``(nx/2, ny/2)``, half a pixel from the star at ``((nx-1)/2, (ny-1)/2)``.
    angles : {"ccw", "getzone2015"}, default "ccw"
        Azimuth convention.  "ccw" counts counter-clockwise from +x over [0, 360), and
        "getzone2015" counter-clockwise from -x over (0, 360].

    Returns
    -------
    idx : ndarray of int
        Flat indices ``row * nx + column`` of the sector's pixels, in increasing order.

    Raises
    ------
    ValueError
        If ``angles`` is not one of the two.

    Notes
    -----
    "getzone2015" reproduces the sectors of the IDL predecessor's 2015 ``getzone`` routine, the
    same pixels in the same order.
    """
    ny, nx = shape
    if center is None:
        center = (nx / 2.0, ny / 2.0)
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - center[0], yy - center[1])
    if angles == "getzone2015":
        ang = np.rad2deg(np.arctan2(yy - center[1], xx - center[0])) + 180.0
    elif angles == "ccw":
        ang = np.rad2deg(np.arctan2(yy - center[1], xx - center[0])) % 360.0
    else:
        raise ValueError(f"zone angles are one of {ZONE_ANGLES}, not {angles!r}")
    m = (r >= inrad) & (r <= outrad) & (ang >= a0_deg) & (ang <= a1_deg)
    return np.flatnonzero(m.ravel())


def arcdist_deg(inrad: float, outrad: float, lam_over_d_px: float) -> float:
    """Compute the field rotation that moves a source by lambda/D at an annulus' middle radius.

    Multiply ``angsep``, in lambda/D, by it to get degrees.

    Parameters
    ----------
    inrad : float
        Inner radius of the annulus, in pixels.
    outrad : float
        Outer radius of the annulus, in pixels.
    lam_over_d_px : float
        lambda/D in pixels.

    Returns
    -------
    arcdist : float
        Field rotation in degrees, ``360 * lam_over_d_px / (2 pi r_mid)`` for the middle radius
        ``r_mid = (inrad + outrad) / 2``, at least 1 px.
    """
    mid = inrad + abs(outrad - inrad) / 2.0
    mid = max(mid, 1.0)
    return 360.0 * lam_over_d_px / (2.0 * np.pi * mid)


def reference_mask(angles: np.ndarray, target: int, angsep_deg: float, anglemax: float) -> np.ndarray:
    """Select the frames that may serve as references for a target frame, by the ADI rule.

    A frame qualifies when its parallactic angle differs from the target's by at least
    ``angsep_deg`` and at most ``anglemax`` degrees.  The target itself never does.

    Parameters
    ----------
    angles : ndarray, shape (n,)
        Parallactic angles of the n frames, in degrees.
    target : int
        Index of the target frame.
    angsep_deg : float
        Minimum difference of angle, in degrees.
    anglemax : float
        Maximum difference of angle, in degrees.

    Returns
    -------
    mask : ndarray of bool, shape (n,)
        True for each frame that qualifies.
    """
    dpa = np.abs(angles - angles[target])
    m = (dpa >= angsep_deg) & (dpa <= anglemax)
    m[target] = False
    return m


@dataclass
class KLIPParams:
    """Settings for one annular KLIP PSF subtraction (Soummer et al. 2012) by :func:`klip_annular`.

    The subtraction runs in an annulus around the star, the zone, optionally cut into azimuthal
    sectors with a KL basis each.  The cube holds n frames of ny by nx pixels.

    Attributes
    ----------
    k_klip : int, default 10
        Number of KL modes in the PSF model.  0 subtracts nothing.
    inrad : float, default 0.0
        Inner radius of the zone, in pixels.  Pixels inside it come out NaN.
    outrad : float, default 70.0
        Outer radius of the zone, in pixels.  Pixels beyond it come out NaN.
    n_ang : int, default 1
        Number of equal azimuthal sectors in the zone.
    fast : bool, default False
        Whether to build one KL basis per sector from all frames, the target included, instead of
        one per target frame from the references ``angsep`` and ``anglemax`` allow.  It is faster
        and self-subtracts more.
    angsep : float, default 0.0
        Minimum field rotation between a target frame and its references, in lambda/D at the
        zone's middle radius.
    anglemax : float, default 360.0
        Maximum field rotation between a target frame and its references, in degrees.
    n_min_ref : int, default 10
        Minimum number of references for a target frame when ``fast`` is False.  A frame with
        fewer comes out NaN, unless that would leave fewer than ``max(n_min_ref, ceil(n/4))``
        frames, and then none is dropped.
    spat_mean : bool, default False
        Whether to subtract each frame's mean over the sector before the projection.
    temp_mean : bool, default False
        Whether to subtract the references' mean frame from the target and the references before
        the projection.
    k_scan : bool, default False
        Whether to return the residuals for every number of KL modes from 1 to ``k_klip``, as a
        first axis of length ``k_klip``.
    zone_center : tuple of 2 floats or None, default None
        The ``(x, y)`` pixel the zone's radii and sectors are measured from.  None takes
        ``(nx/2, ny/2)``, half a pixel from the star at ``((nx-1)/2, (ny-1)/2)``.
    threads : int, default 1
        Number of CPU threads that share the target frames when ``fast`` is False.
    ref_lib : ReferenceLibrary or None, default None
        Library of RDI reference frames ranked by similarity to each target frame
        (:class:`klip_tpe.reflib.ReferenceLibrary`).  With one, each target takes its best matches
        from the ``ref_cube`` given to :func:`klip_annular`.
    ref_keep : dict of str to int, or None, default None
        Number of frames each group of ``ref_lib`` gives each target, as ``{group: count}``.  A
        group left out gives its most.
    gram : {"target", "zone"}, default "target"
        How the per-target projection is computed when ``fast`` is False.  "target" builds each
        target's KL modes from its references, and "zone" projects through the sector's Gram
        matrix, computed once, which is faster and equal to rounding.
    arc_radii : tuple of 2 floats or None, default None
        The ``(inner, outer)`` radii in pixels whose middle sets where ``angsep`` converts to
        degrees.  None takes ``inrad`` and ``outrad``.
    zone_angles : {"ccw", "getzone2015"}, default "ccw"
        Axis the sector azimuths are counted from.  "ccw" counts counter-clockwise from +x, and
        "getzone2015" counter-clockwise from -x.

    Notes
    -----
    "getzone2015" reproduces the sectors of the IDL predecessor's 2015 ``getzone`` routine.
    """
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
    #: searched reference library (:mod:`klip_tpe.reflib`).  With one, ``ref_cube`` is a
    #: POOL rather than a fixed basis: every target frame gets its own best-matched subset
    #: of it, so RDI runs per target instead of from one shared basis.
    ref_lib: Optional[Any] = None
    ref_keep: Optional[Dict[str, int]] = None
    #: how the per-target path forms a target's reference covariance.  ``"target"``: from
    #: the references' pixels, ``R R^T`` for every target, all M modes formed and the first k
    #: kept, as always.  ``"zone"``: one Gram matrix ``G = D D^T`` per zone (the pool's under
    #: a searched library), each target's covariance read off it as ``G[refs, refs]``, and
    #: the target projected on its first k modes through ``G[t, refs]`` without forming them
    #: (:func:`klip_annular`).  The same KLIP to float64 rounding but not to the bit, since the
    #: Gram's dot products are summed in another order.  The fast path forms only its k modes.
    gram: str = "target"
    #: ``(inner, outer)`` px of the annulus whose mid radius turns ``angsep`` into degrees of
    #: rotation, when the zone reaches beyond it (``zone_in`` / ``zone_out`` of the reducer);
    #: None: the zone's own ``inrad`` and ``outrad``
    arc_radii: Optional[Tuple[float, float]] = None
    #: how the sectors' angles are measured about ``zone_center`` (:func:`zone_indices`)
    zone_angles: str = "ccw"


_FM_TOL = 1e-6                   # eigenvalue tolerance of get_klip_basis_new / the FM block


def klip_fm_zone(R: np.ndarray, Mref: Optional[np.ndarray], T: np.ndarray, Mtar: np.ndarray, k: int,
                 fm_selfsub: bool = True, eig: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = None
                 ) -> np.ndarray:
    """Forward-model a planet through KLIP in one zone (KLIP-FM, Pueyo 2016, to first order).

    The forward model includes the planet's over-subtraction and its self-subtraction, the change of
    the KL modes the planet in the references causes.  With the references' KL modes ``Z_k`` and
    their first-order change ``dZ_k``::

        FM = Mtar - sum_k [ <Mtar, Z_k> Z_k + <T, dZ_k> Z_k + <T, Z_k> dZ_k ]

    where ``<a, b>`` is a dot product over the zone's pixels.

    Parameters
    ----------
    R : ndarray, shape (nref, npx)
        Pixels of the nref reference frames in the zone, npx of them, NaN replaced by 0.
    Mref : ndarray or None
        Planet model in the reference frames, shape (nref, npx).  None, for a planet-free RDI
        library, leaves the KL modes unperturbed.
    T : ndarray, shape (ntar, npx)
        Pixels of the ntar target frames in the zone, as KLIP projects them.
    Mtar : ndarray, shape (ntar, npx)
        Planet model in the target frames.
    k : int
        Number of KL modes.  0 returns ``Mtar`` unchanged.
    fm_selfsub : bool, default True
        Whether to include the self-subtraction terms, the two with ``dZ_k``.  False keeps only the
        over-subtraction, as a diagnostic.
    eig : tuple or None, default None
        Decomposition of ``R R^T`` already made for this ``R``, as its eigenvalues (the largest
        first), its eigenvectors, and its first ``k`` modes or None.  None decomposes again.

    Returns
    -------
    fm : ndarray, shape (ntar, npx)
        Forward-modeled planet.

    Notes
    -----
    The first-order change comes from ``dC = R Mref^T + Mref R^T``, with ``l_k`` and ``u_k`` the
    eigenvalues and eigenvectors of ``R R^T`` and ``Z_k = R^T u_k / sqrt(l_k)``:
    ``dl_k = u_k . dC u_k``, ``du_k = sum_{j != k} (u_j . dC u_k) / (l_k - l_j) u_j`` over
    eigenvalues that differ by more than 1e-6 of ``max(|l_k|, 1)``, and
    ``dZ_k = -dl_k / (2 l_k) Z_k + (R^T du_k + Mref^T u_k) / sqrt(l_k)``.  The IDL predecessor
    does the same in ``multiklip.pro`` L365-403 (one basis) and L494-527 (per target).
    """
    R = np.asarray(R, np.float64)
    T = np.asarray(T, np.float64)
    Mtar = np.asarray(Mtar, np.float64)
    nref = R.shape[0]
    nmod = int(min(max(k, 0), nref))
    if nmod == 0:
        return Mtar.copy()
    if eig is None:
        lam, U = _eig_desc(R @ R.T)
    else:
        lam, U = eig[0], eig[1]
    valid = lam > _FM_TOL
    scale = np.where(valid, 1.0 / np.sqrt(np.where(valid, lam, 1.0)), 0.0)
    if eig is not None and eig[2] is not None:
        Zb = eig[2]
    else:
        Zb = ((U * scale[None, :]).T @ R)[:nmod]                # (nmod, npx)
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
    """Run annular KLIP PSF subtraction on a cube and return each frame's residuals.

    For every target frame and every sector of the zone, it projects the frame onto the first
    ``p.k_klip`` KL modes of its references and subtracts that PSF model.  The modes come from one
    basis per sector (``p.fast``) or from each target's own references, chosen by the ADI rule
    (:func:`reference_mask`).  The residuals are neither derotated nor combined.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The n science frames, ny by nx pixels each, not derotated, with the star at
        ``((nx-1)/2, (ny-1)/2)``.  NaN pixels enter the PSF model as 0, so their residuals come
        out finite.
    angles : ndarray, shape (n,)
        Parallactic angle of each frame, in degrees.  Only the differences matter, so a constant
        offset such as true north can be left out.
    p : KLIPParams
        Settings of the subtraction.
    lam_over_d_px : float
        lambda/D in pixels, which converts ``p.angsep`` to degrees (:func:`arcdist_deg`).
    log : callable or None, default None
        Function called with a message when ``p.k_klip`` is capped (see Notes).  None stays silent.
    fm_cube : ndarray or None, default None
        Planet model cube aligned with ``cube``, for KLIP-FM.  Its forward-modeled residuals come
        back as a third output, and it does not go with ``p.k_scan``.
    ref_cube : ndarray or None, default None
        Reference frames for RDI, shape (m, ny, nx), or for ARDI the concatenation
        ``[ref, science]``.  The KL basis then comes from these frames, one basis for all targets
        unless ``p.ref_lib`` selects from them per target.  None takes the references from ``cube``.
    fm_ref_cube : ndarray or None, default None
        Planet model in ``ref_cube``, zeros for planet-free reference frames.  None leaves the modes
        unperturbed when ``ref_cube`` is given and takes ``fm_cube`` otherwise.
    fm_selfsub : bool, default True
        Whether to include KLIP-FM's self-subtraction terms.  False keeps only the
        over-subtraction, as a diagnostic.

    Returns
    -------
    residuals : ndarray, shape (n, ny, nx) or (k_klip, n, ny, nx)
        Residual frames in float32, with a first axis over the number of KL modes when
        ``p.k_scan``.  Pixels outside the zone, and frames dropped for too few references, are NaN.
    info : dict
        How the subtraction ran.  Its keys are listed under Notes.
    fm_residuals : ndarray, shape (n, ny, nx)
        [fm_cube given] Forward-modeled planet frames, in float32.

    Raises
    ------
    ValueError
        If ``p.gram`` is unknown, if ``fm_cube`` comes with ``p.k_scan`` or does not match ``cube``,
        or if ``ref_cube``'s frames are not the science frames' size.

    Notes
    -----
    ``info`` holds the path taken (``fast``, ``auto_fast``, ``rdi``, ``klip_gram``,
    ``library_as_adi``), ``arcdist`` (degrees of field rotation per lambda/D), the frame counts
    (``n_frames``, ``n_basis_frames``), the modes used (``k_requested``, ``k_effective``,
    ``k_capped``), each target's reference count before and after the fallback to all frames
    (``nref``, ``nref_used``) and the frames dropped (``dropped``, ``n_dropped``).  The per-target
    path also reports ``n_starved`` and ``drop_enabled``, and a searched library ``ref_keep`` and
    ``ref_metric``.

    The per-target path (``p.fast`` False) drops targets with fewer than ``p.n_min_ref``
    references, unless that would leave fewer than ``max(n_min_ref, ceil(n/4))`` targets, and falls
    back to all other frames when fewer than 4 qualify.  With ``p.angsep`` 0 and ``p.anglemax`` past
    the sequence's PA span every target's references are all the other frames, so the fast path
    runs (``info['auto_fast']``).  A basis built from the science frames themselves (fast, no RDI)
    caps ``k_klip`` at n - 1, since n modes would subtract every frame from itself.

    The reducer then high-passes the residuals again, derotates them (:func:`derotate`) and combines
    them, by default with :func:`nw_ang_comb`::

        res, info = klip_annular(cube, angles, KLIPParams(k_klip=10, inrad=8, outrad=30), lam_over_d_px)
        image = nw_ang_comb(derotate(res, angles, truenorth), angles)
    """
    n, ny, nx = cube.shape
    angles = np.asarray(angles, float)
    k = int(max(p.k_klip, 0))
    fast = bool(p.fast)
    do_fm = fm_cube is not None
    gram = str(getattr(p, "gram", "target") or "target").lower()
    if gram not in ("target", "zone"):
        raise ValueError(f"KLIPParams.gram is 'target' or 'zone', not {getattr(p, 'gram', None)!r}")
    if do_fm and p.k_scan:
        raise ValueError("KLIP-FM is fixed-k: not supported with k_scan")
    if do_fm and fm_cube.shape != cube.shape:
        raise ValueError("fm_cube must be frame-aligned with cube")
    rdi = ref_cube is not None
    use_lib = rdi and p.ref_lib is not None
    lib_as_adi = False
    if use_lib and p.ref_lib.is_plain_adi(dict(p.ref_keep or {})):
        # the counts leave nothing but the rotating sequence itself: that is the ADI
        # reduction, taken as the ADI code takes it, so no reference frames means exactly
        # no reference frames
        use_lib, rdi, lib_as_adi = False, False, True
    if rdi:
        if ref_cube.shape[1:] != (ny, nx):
            raise ValueError("ref_cube must have the science frame size")
        # A fixed reference cube is one basis for every target (refsel=0 + /fast in the
        # IDL predecessor).  A searched library is not: each target keeps its own
        # best-matched subset, so the basis changes frame to frame and the per-target path
        # has to run.
        fast = not use_lib
    paspan = float(angles.max() - angles.min()) if n >= 2 else 0.0
    auto_fast = False
    # The auto-fast shortcut says: with no angular exclusion asked for, a per-target basis
    # would be the same frames for every target, so build it once.  That reasoning fails
    # the moment a searched library is in play -- there the per-target basis differs by
    # construction -- so the shortcut has to stand down.
    if not fast and not use_lib and p.angsep <= 0 and p.anglemax >= paspan:
        fast, auto_fast = True, True
    a_in, a_out = (p.inrad, p.outrad) if p.arc_radii is None else (float(p.arc_radii[0]), float(p.arc_radii[1]))
    arc = arcdist_deg(a_in, a_out, lam_over_d_px)
    angsep_deg = p.angsep * abs(arc)
    # A shared basis built from the frames it is about to subtract (fast/auto_fast without
    # RDI: R = D) spans those frames exactly, so mode n annihilates every one of them and
    # the residual is float64 round-off -- an image of ~1e-15 whose S/N is the ratio of two
    # round-off numbers, i.e. an O(1) RANDOM draw that an optimizer will happily chase.
    # klip_basis silently clamps k to the row count, so nothing errors and nothing warns.
    # The per-target path is safe (reference_mask drops the target) and so is RDI (the basis
    # frames are not the target frames); only this combination has to be capped.
    k_req, k_capped = k, False
    if fast and not rdi and n >= 2 and k >= n:
        k, k_capped = n - 1, True
        if log is not None:
            log(f"  k_klip={k_req} >= {n} basis frames and the basis is built from the science "
                f"frames themselves: capped to {k} (k={k_req} would subtract every frame from "
                f"itself and leave only round-off)")
    info = {"fast": fast, "auto_fast": auto_fast, "arcdist": arc, "n_frames": n, "n_dropped": 0,
            "rdi": rdi, "n_basis_frames": int(ref_cube.shape[0]) if rdi else n, "fm": do_fm,
            "library_as_adi": lib_as_adi,
            "k_requested": int(k_req), "k_effective": int(k), "k_capped": bool(k_capped),
            "fm_selfsub": bool(fm_selfsub) if do_fm else None, "klip_gram": gram}

    # A k-scan returns one image per requested k, and that count must NOT depend on the cap:
    # a PartitionedReducer stacks the per-partition scans, and two channels whose binned
    # frame counts differ by one would cap differently and produce stacks of different
    # length ("all input arrays must have the same shape").  Slices past the cap repeat the
    # capped result, which is what the non-scan path returns for those k anyway.
    shape_out = (k_req, n, ny, nx) if p.k_scan else (n, ny, nx)
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
        idx = zone_indices((ny, nx), p.inrad, p.outrad, a0, a1, p.zone_center, p.zone_angles)
        if idx.size:
            zones.append(idx)

    def _basis(R):
        # the modes, eigenvalues and eigenvectors.  The zone mode forms only the k modes it uses.
        if k == 0:
            return np.zeros((0, R.shape[1])), None, None
        Z, lam, V, _ = _klip_basis_eig(R, k, p.spat_mean, p.temp_mean, only_k=(gram == "zone"))
        return Z, lam, V

    def _fm_eig(R, Z, lam, V):
        # The basis' decomposition, handed to klip_fm_zone so it is not made twice.  The FM
        # decomposes the raw frames, so only without centring, and it takes the basis' modes
        # only where its eigenvalue cut keeps the same ones (they differ at lambda == tol).
        if lam is None or p.spat_mean or p.temp_mean:
            return None
        nmod = min(k, R.shape[0])
        valid = lam[:nmod] > _FM_TOL
        sf = np.where(valid, 1.0 / np.sqrt(np.where(valid, lam[:nmod], 1.0)), 0.0)
        return lam, V, (Z[:nmod] if np.array_equal(_basis_scale(lam, 1e-6)[:nmod], sf) else None)

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
            Z, lam, V = _basis(R)
            T = D.copy()
            if p.spat_mean:
                T -= T.mean(axis=1, keepdims=True)
            if p.temp_mean:
                T -= R.mean(axis=0, keepdims=True)
            kk_max = min(Z.shape[0], k)
            if p.k_scan:
                S = T @ Z.T
                for kk in range(k_req):
                    m = min(kk + 1, kk_max)                 # past the cap: repeat the cap
                    res = T - S[:, :m] @ Z[:m] if m else T.copy()
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
                fo[:, idx] = klip_fm_zone(R, Mref, T, Mtar, k, fm_selfsub, eig=_fm_eig(R, Z, lam, V))
        return (out, info, fm_out) if do_fm else (out, info)

    # ---- per-target (slow) path -------------------------------------------
    keep = dict(p.ref_keep or {}) if use_lib else {}
    if use_lib:
        if p.ref_lib.n_target != n:
            raise ValueError(f"reference library was built for {p.ref_lib.n_target} target "
                             f"frames, this cube has {n}")
        # the census before any fallback, as reference_mask's is: a rotation group counts
        # the rows the ADI rule admits, so the starved-target rule reads the same number
        # with and without a library
        nref = np.array([p.ref_lib.select(t, keep, angsep_deg, p.anglemax, census=True).size for t in range(n)])
        info["ref_keep"] = {g.name: int(keep.get(g.name, g.max_keep)) for g in p.ref_lib.groups}
        info["ref_metric"] = p.ref_lib.metric
    else:
        nref = np.array([reference_mask(angles, t, angsep_deg, p.anglemax).sum() for t in range(n)])
    nstarv = int((nref < p.n_min_ref).sum())
    do_drop = (n - nstarv) >= max(p.n_min_ref, int(np.ceil(0.25 * n)))
    info["n_starved"] = nstarv
    info["drop_enabled"] = bool(do_drop)
    info["nref"] = nref
    nref_used = nref.copy()
    dropped = np.zeros(n, bool)
    zone_data = [flat[:, idx].astype(np.float64) for idx in zones]
    #: where the BASIS rows come from -- the library pool under a searched library, the
    #: science cube itself otherwise.  ``refs`` indexes whichever of the two it is.
    basis_flat = ref_flat if use_lib else flat
    basis_zone = [basis_flat[:, idx].astype(np.float64) for idx in zones] if use_lib else zone_data
    n_basis = basis_flat.shape[0]
    # gram="zone": one Gram matrix per zone, made before the targets.  A target's covariance
    # is the submatrix of it at its references, its products with them are its row, and its
    # projection on the first k modes is a weighted sum of the reference frames, so no target
    # forms its covariance or its modes from the pixels.  temp_mean centers on each target's
    # own reference set, which no single Gram holds, so it keeps the per-target path.
    zone_gram = gram == "zone" and k > 0 and not p.temp_mean
    info["klip_gram"] = "zone" if zone_gram else "target"
    zgram = []
    if zone_gram:
        for D, BD in zip(zone_data, basis_zone):
            zd = _zero_nan(D)
            if p.spat_mean:
                zd = zd - zd.mean(axis=1, keepdims=True)     # each frame's own mean, as klip_basis takes it
            if use_lib:
                bzd = _zero_nan(BD)
                if p.spat_mean:
                    bzd = bzd - bzd.mean(axis=1, keepdims=True)
                zgram.append((zd, bzd, bzd @ bzd.T, zd @ bzd.T))
            else:
                G = zd @ zd.T
                zgram.append((zd, zd, G, G))

    def _fm_models(t, refs, idx):
        # the planet model in the references and in the target, as the FM takes them
        mflat = fmref_flat if (use_lib and fmref_flat is not None) else fm_flat
        if use_lib and fmref_flat is None:
            Mref = None            # library RDI with no model in the references
        else:
            Mref = _zero_nan(mflat[refs][:, idx].astype(np.float64))
        return Mref, _zero_nan(fm_flat[t:t + 1, idx].astype(np.float64))

    def _zone_target(t, r, zi, idx, D, BD, refs):
        zd, bzd, G, X = zgram[zi]
        T = zd[t]                                       # zero-filled (and centered), as T below
        kk_max = min(k, r.size)
        if kk_max == 0:
            res = T[None, :]
            lam = V = None
        else:
            lam, V = _eig_desc(G[np.ix_(r, r)])
            Vk = V[:, :kk_max]
            # T's coefficient on mode m is (G[t, r] . v_m) / sqrt(lambda_m), and mode m is
            # R^T v_m / sqrt(lambda_m): the projection is w . R with w = V_k c
            c = (X[t, r] @ Vk) * _basis_scale(lam, 1e-6)[:kk_max] ** 2
            if p.k_scan:
                W = np.zeros((kk_max, bzd.shape[0]))
                W[:, r] = np.cumsum(Vk * c[None, :], axis=1).T     # row m: the weights of modes 0..m
                res = T[None, :] - W @ bzd
            else:
                w = np.zeros(bzd.shape[0])
                w[r] = Vk @ c
                res = (T - w @ bzd)[None, :]
        if p.k_scan:
            for kk in range(k_req):
                out[kk, t].reshape(-1)[idx] = res[min(kk, res.shape[0] - 1)]
        else:
            out[t].reshape(-1)[idx] = res[0]
        if do_fm:
            Mref, Mtar = _fm_models(t, refs, idx)
            if p.spat_mean or lam is None:
                # the FM decomposes the raw frames, which the centered Gram does not hold
                R = _zero_nan(BD[refs])
                fm = klip_fm_zone(R, Mref, _zero_nan(D[t:t + 1]), Mtar, k, fm_selfsub)
            else:
                R = bzd[r]
                valid = lam > _FM_TOL
                sf = np.where(valid, 1.0 / np.sqrt(np.where(valid, lam, 1.0)), 0.0)
                Zb = (V[:, :kk_max] * sf[None, :kk_max]).T @ R
                fm = klip_fm_zone(R, Mref, zd[t:t + 1], Mtar, k, fm_selfsub, eig=(lam, V, Zb))
            fm_out[t].reshape(-1)[idx] = fm[0]

    def _target(t):
        if do_drop and nref[t] < p.n_min_ref:
            dropped[t] = True
            return                        # frame stays NaN
        if use_lib:
            refs = p.ref_lib.mask(t, keep, n_basis, angsep_deg, p.anglemax)
            if refs.sum() < 2:
                # every eligible frame, rather than a basis of one: the counts came from
                # the optimizer and a starved draw should score badly, not crash
                refs = np.zeros(n_basis, bool)
                for g in p.ref_lib.groups:
                    refs[p.ref_lib.eligible(t, g)] = True
        else:
            refs = reference_mask(angles, t, angsep_deg, p.anglemax)
            if refs.sum() < 4:
                refs = np.ones(n, bool)
                refs[t] = False
        nref_used[t] = int(refs.sum())
        r = np.flatnonzero(refs) if zone_gram else None
        for zi, (idx, D, BD) in enumerate(zip(zones, zone_data, basis_zone)):
            if zone_gram:
                _zone_target(t, r, zi, idx, D, BD, refs)
                continue
            R = _zero_nan(BD[refs])
            Z, lam, V = _basis(R)
            T = _zero_nan(D[t:t + 1].copy())
            Star = T.copy()               # raw target (the FM uses the un-adjusted stellar frame, L493)
            if p.spat_mean:
                T -= T.mean()
            if p.temp_mean:
                T -= R.mean(axis=0, keepdims=True)
            kk_max = min(Z.shape[0], k)
            if p.k_scan:
                S = T @ Z.T
                for kk in range(k_req):
                    m = min(kk + 1, kk_max)                 # past the cap: repeat the cap
                    out[kk, t].reshape(-1)[idx] = (T - S[:, :m] @ Z[:m] if m else T)[0]
            else:
                out[t].reshape(-1)[idx] = _project(T, Z, kk_max)[0]
            if do_fm:
                Mref, Mtar = _fm_models(t, refs, idx)
                fm_out[t].reshape(-1)[idx] = klip_fm_zone(R, Mref, Star, Mtar, k, fm_selfsub,
                                                          eig=_fm_eig(R, Z, lam, V))[0]

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
