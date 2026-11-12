"""LBTI/NOMIC adapter for data pre-processed with pyNOMIC (https://github.com/vivekv7323/pyNOMIC).

pyNOMIC ends with registered, background-subtracted per-frame FITS files and a few ``.npz`` side
files written by its notebooks.  This module turns them into one
:class:`~klip_tpe.reducer.Dataset` per partition, a chop state and night, and builds the reducer,
search space, guardrail and objective as the NEAR adapter does, so the runner and the command
line work unchanged.

The positional arrays read, in the order the notebooks ``np.savez`` them:

==============================  ==================================================
``<obj>_NOMIC_chop_correction``  ``files, chops, header_info``  (``header_info[1]`` =
                                 parallactic angle per frame, ``header_info[2]`` = JD)
``<obj>_NOMIC_aligned``          ``aligned_files, original_psf_locs, imgfits,
                                 array_shape, file_size``
``<obj>_NOMIC_evaluated``        ``fwhms, eccentricities, psfmaxima, background_dev,
                                 correlations, residual_dev, lbtfits, reffits``
``<obj>_NOMIC_binned_evaluated`` the same eight for the binned frames followed by
                                 ``binned_files, binned_chops, binned_angles``
==============================  ==================================================

The frame-quality tags map onto the optimizer's three frame-selection dimensions:

* ``corrs`` from ``correlations``, the cross-correlation with the reference PSF, which the
  searched ``corr_thresh`` keeps at or above.
* ``noises`` from ``background_dev`` over its median, for ``noise_max``.
* ``coronoise`` from ``residual_dev`` over its median, for ``coronoise_max``, the PSF-fit residual
  being the closest NOMIC analog of NEAR's coronagraphic-residual tag.

The injection model is pyNOMIC's own, the obstructed-Airy fit of each frame (``reffits`` row ``j``
for frame ``j``), so a companion of contrast ``c`` in frame ``j`` is exactly pyNOMIC's
``inject_source``, ``airy_disk(c * amp_j, sigmax_j, sigmay_j, p_j)``.  ``psf="frame"``, each
frame's own star as an empirical template, is an option.

Image groups are contiguous time segments separated by jumps of the star's position on the
detector, from nod or pointing offsets, which pyNOMIC's notebooks split
(``hf.image_groups(header_info[2], maxima[:, 0])``) and drop by hand.  Here a group can be a
partition of its own.  ``groups="auto"`` runs the same split (:func:`image_groups`, with the star
positions from ``<obj>_NOMIC_psf_subtraction.npz`` ``maxima`` or ``<obj>_NOMIC_aligned.npz``
``original_psf_locs``), and per-frame labels or JD split times give your own.  Every night, chop
state and group then becomes one partition ``<name><A|B>g<k>`` with its own KLIP basis and
parameter block, which the optimizer treats as it treats nights.  Groups with fewer than
``group_min_frames`` frames are left out.

Notes
-----
Confirm the angle convention on a known binary before trusting position angles.  The loader takes
pyNOMIC's ``header_info[1]`` as the parallactic angle such that a counter-clockwise rotation by
``parang_sign * parang + truenorth`` puts north up, the convention of pyNOMIC's ``inject_source``
(detector azimuth ``PA - parang``).  ``parang_sign=-1`` flips it, and ``truenorth`` adds a fixed
offset.
"""
from __future__ import annotations

import glob
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..feasibility import FrameSelectionGuard, ReferenceCountGuard, compose
from ..injection import AiryPSF, FramePSF, GaussianPSF, InjectionModel
from ..metrics import MawetPeakSNR, Objective
from ..positions import PositionSampler
from ..reducer import Dataset, KLIPReducer, PartitionedReducer, reducer_class
from ..space import SearchSpace
from . import search_space as _space

__all__ = ["NOMIC", "read_frames", "datasets_from_arrays", "load_pynomic", "image_groups", "group_labels",
           "make_reducer", "make_space",
           "make_guard", "default_config", "production_defaults"]


class NOMIC:
    """NOMIC's instrument constants, for a single LBT aperture in the N' band.

    Attributes
    ----------
    pxscale : float
        Plate scale, in arcsec per pixel, pyNOMIC's default.
    lam : float
        Center of the N' band, in meters.
    diam : float
        Diameter of one LBT aperture, in meters.  Override it in :func:`make_reducer` for Fizeau data.
    truenorth : float
        True-north offset added to the parallactic angle at derotation, in degrees.
    crop_half : int
        Half size of the KLIP crop, in pixels, as pyNOMIC's ``crop_size=150``.
    lam_over_d_px : float
        lambda/D in pixels, 15.23.
    fwhm_px : float
        ``1.028`` lambda/D in pixels, 15.66, where pyNOMIC's notebooks use about 14.
    outrad_cap : float
        Maximum outer radius of a zone, in pixels.
    """
    pxscale = 0.0179                 # arcsec / px (pyNOMIC default)
    lam = 11.1e-6                    # m, N' band center
    diam = 8.4                       # m (one LBT aperture; Fizeau data: override in make_reducer)
    truenorth = 0.0                  # deg, added to the parallactic angle at derotation
    crop_half = 75                   # KLIP crop half size (150 px, like pyNOMIC's crop_size=150)
    lam_over_d_px = (lam / diam) * 206265.0 / pxscale     # 15.23
    fwhm_px = 1.028 * lam_over_d_px                       # 15.66 (pyNOMIC notebooks use ~14)
    outrad_cap = 70.0


production_defaults = dict(_space.production_defaults)


# ----------------------------------------------------------------------------
# reading pyNOMIC products
# ----------------------------------------------------------------------------
def _fits_data(path: str) -> np.ndarray:
    """Read a FITS file's data, as float32.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    data : ndarray
        The data of extension 0, or of extension 1 when 0 is empty.
    """
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        d = h[0].data
        if d is None:
            d = h[1].data
    return np.asarray(d, np.float32)


def read_frames(files: Sequence[Union[str, os.PathLike]], crop_half: int = NOMIC.crop_half,
                center: Optional[Tuple[float, float]] = None, nan_policy: str = "zero",
                progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Read per-frame FITS files into a cropped float32 cube.

    After the crop the star sits at ``(h - 0.5, h - 0.5)`` for ``h = crop_half``, as in the NEAR
    cubes.

    Parameters
    ----------
    files : sequence of str or path
        The frame files, one frame each, the first plane of a 3-d file.
    crop_half : int, default NOMIC.crop_half
        Half size of the crop, in pixels.
    center : tuple of 2 floats or None, default None
        The star's ``(x, y)`` pixel.  None takes ``((nx-1)/2, (ny-1)/2)``, where pyNOMIC's
        ``frame_registration`` puts it.
    nan_policy : {"zero", "keep"}, default "zero"
        "zero" replaces NaN by 0, such as the over-subtracted PSF ghosts pyNOMIC's ``mask_files``
        masks, and "keep" leaves them.
    progress : callable or None, default None
        Function called as ``progress(done, total)`` every 500 frames and at the end.

    Returns
    -------
    cube : ndarray, shape (n, 2 crop_half, 2 crop_half)
        The frames, in float32.

    Raises
    ------
    ValueError
        If there are no files or the crop does not fit in the frames.
    """
    files = [str(f) for f in files]
    if not files:
        raise ValueError("no frames")
    first = _fits_data(files[0])
    ny, nx = first.shape[-2:]
    cx, cy = ((nx - 1) / 2.0, (ny - 1) / 2.0) if center is None else (float(center[0]), float(center[1]))
    h = int(crop_half)
    x0, y0 = int(round(cx + 0.5)) - h, int(round(cy + 0.5)) - h
    if x0 < 0 or y0 < 0 or x0 + 2 * h > nx or y0 + 2 * h > ny:
        raise ValueError(f"crop_half={h} does not fit in {nx}x{ny} frames around ({cx}, {cy})")
    cube = np.empty((len(files), 2 * h, 2 * h), np.float32)
    for i, f in enumerate(files):
        d = first if i == 0 else _fits_data(f)
        if d.ndim == 3:
            d = d[0]
        cube[i] = d[y0:y0 + 2 * h, x0:x0 + 2 * h]
        if progress is not None and (i % 500 == 0 or i == len(files) - 1):
            progress(i + 1, len(files))
    if nan_policy == "zero":
        cube[~np.isfinite(cube)] = 0.0
    return cube


def _pre_bin(cube: np.ndarray, angles: np.ndarray, tags: Optional[Dict[str, np.ndarray]], m: int):
    """Mean-bin consecutive frames by ``m``, to relieve memory before the search.

    The searched ``bin`` then acts on the pre-binned frames, and frames past the last full bin are
    dropped.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The frames.
    angles : ndarray, shape (n,)
        Parallactic angle of each frame, in degrees.
    tags : dict of str to ndarray or None
        Per-frame quality tags.
    m : int
        Frames per bin.  1 or less leaves everything as it is.

    Returns
    -------
    cube : ndarray
        The binned frames, in float32.
    angles : ndarray
        Mean angle of each bin, in degrees.
    tags : dict or None
        Mean tags of each bin.
    """
    n = cube.shape[0] // m * m
    if m <= 1 or n == 0:
        return cube, angles, tags
    c = cube[:n].reshape(n // m, m, *cube.shape[1:]).mean(axis=1).astype(np.float32)
    a = angles[:n].reshape(n // m, m).mean(axis=1)
    t = None if tags is None else {k: v[:n].reshape(n // m, m).mean(axis=1) for k, v in tags.items()}
    return c, a, t


def datasets_from_arrays(files: Sequence, chops: Sequence[str], angles: Sequence[float],
                         correlations: Optional[Sequence[float]] = None,
                         background_dev: Optional[Sequence[float]] = None,
                         residual_dev: Optional[Sequence[float]] = None,
                         reffits: Optional[np.ndarray] = None, frame_bool: Optional[Sequence[bool]] = None,
                         chop_states: Sequence[str] = ("CHOP_A", "CHOP_B"), name: str = "nomic",
                         crop_half: int = NOMIC.crop_half, center: Optional[Tuple[float, float]] = None,
                         parang_sign: float = 1.0, texp_per_frame: Optional[float] = None,
                         pre_bin: int = 1, nan_policy: str = "zero", max_frames: Optional[int] = None,
                         groups: Optional[np.ndarray] = None, group_min_frames: int = 20,
                         log: Callable[[str], None] = print) -> Dict[str, Dataset]:
    """Build one :class:`~klip_tpe.reducer.Dataset` per chop state, or per chop state and image group, from pyNOMIC's arrays.

    Without ``groups`` the partitions are ``<name>A`` and ``<name>B``.  With them, each chop state and
    group with at least ``group_min_frames`` frames becomes its own partition ``<name><A|B>g<k>``.

    Parameters
    ----------
    files : sequence
        Path of each frame's FITS file.
    chops : sequence of str
        Chop state of each frame.
    angles : sequence of float
        Parallactic angle of each frame, in degrees.
    correlations : sequence of float or None, default None
        Each frame's cross-correlation with the reference PSF, the ``corrs`` tag.
    background_dev : sequence of float or None, default None
        Each frame's background scatter, whose ratio to its median is the ``noises`` tag.
    residual_dev : sequence of float or None, default None
        Each frame's PSF-fit residual scatter, whose ratio to its median is the ``coronoise`` tag.
    reffits : ndarray or None, default None
        pyNOMIC's Airy-fit parameters, shape (n, 7).  Their median goes to ``meta["airy"]`` and,
        without pre-binning, each frame's own to ``meta["airy_frames"]``, for the injection model.
    frame_bool : sequence of bool or None, default None
        pyNOMIC's ``frame_rejection`` output, the frames kept beforehand.  The searched frame
        selection then acts on the survivors through the tags.
    chop_states : sequence of str, default ("CHOP_A", "CHOP_B")
        The chop states, each a partition.
    name : str, default "nomic"
        Prefix of the partition identifiers.
    crop_half : int, default NOMIC.crop_half
        Half size of the crop, in pixels.
    center : tuple of 2 floats or None, default None
        The star's ``(x, y)`` pixel in the frames (:func:`read_frames`).
    parang_sign : float, default 1.0
        Factor on the angles, -1 for the opposite convention.
    texp_per_frame : float or None, default None
        Exposure time of a frame, in seconds.  None counts frames.
    pre_bin : int, default 1
        Consecutive frames averaged at load (:func:`_pre_bin`).
    nan_policy : {"zero", "keep"}, default "zero"
        What to do with NaN pixels (:func:`read_frames`).
    max_frames : int or None, default None
        Maximum number of frames per partition.  None takes all.
    groups : ndarray of int or None, default None
        Image-group label of each frame (:func:`group_labels`).
    group_min_frames : int, default 20
        Minimum number of frames of a group's partition.
    log : callable, default print
        Function called with each log line.

    Returns
    -------
    datasets : dict of str to Dataset
        The partitions, by identifier.

    Raises
    ------
    ValueError
        If the files, chops, angles or group labels differ in length.
    """
    files = np.asarray([str(f) for f in files])
    chops = np.asarray([str(c) for c in chops])
    angles = np.asarray(angles, float) * float(parang_sign)
    n = len(files)
    if not (len(chops) == len(angles) == n):
        raise ValueError(f"files ({n}), chops ({len(chops)}) and angles ({len(angles)}) differ in length")
    keep = np.ones(n, bool) if frame_bool is None else np.asarray(frame_bool, bool)
    glab = None
    if groups is not None:
        glab = np.asarray(groups, int)
        if len(glab) != n:
            raise ValueError(f"groups has {len(glab)} labels for {n} frames")

    def _tag(v):
        if v is None:
            return None
        v = np.asarray(v, float)
        if v.ndim > 1:
            v = v.reshape(v.shape[0], -1)[:, 0]
        return v
    corr, bdev, rdev = _tag(correlations), _tag(background_dev), _tag(residual_dev)
    out: Dict[str, Dataset] = {}
    units = [(st, None) for st in chop_states] if glab is None else \
        [(st, g) for st in chop_states for g in np.unique(glab)]
    for st, g in units:
        sel = np.flatnonzero(keep & (chops == st) & (True if g is None else (glab == g)))
        if max_frames:
            sel = sel[:int(max_frames)]
        suffix = st.replace("CHOP_", "").replace("CHOP", "") or st
        pid = f"{name}{suffix}" + ("" if g is None else f"g{int(g) + 1}")
        if sel.size == 0 or (g is not None and sel.size < int(group_min_frames)):
            log(f"  {pid}: {sel.size} frames -- skipped" + ("" if g is None else f" (< group_min_frames {group_min_frames})"))
            continue
        log(f"  reading {pid}: {sel.size} frames ...")
        cube = read_frames(files[sel], crop_half=crop_half, center=center, nan_policy=nan_policy)
        ang = angles[sel]
        tags = None
        if corr is not None or bdev is not None or rdev is not None:
            tags = {}
            tags["corrs"] = corr[sel] if corr is not None else np.ones(sel.size)
            tags["noises"] = (bdev[sel] / np.nanmedian(bdev[sel])) if bdev is not None else np.ones(sel.size)
            tags["coronoise"] = (rdev[sel] / np.nanmedian(rdev[sel])) if rdev is not None else np.ones(sel.size)
            for k, v in tags.items():
                v = np.asarray(v, float)
                v[~np.isfinite(v)] = np.nanmedian(v) if np.isfinite(v).any() else 1.0
                tags[k] = v
        cube, ang, tags = _pre_bin(cube, ang, tags, int(pre_bin))
        meta: Dict[str, Any] = {"chop": st, "pre_bin": int(pre_bin), "n_raw": int(sel.size),
                                "crop_half": int(crop_half), "parang_sign": float(parang_sign)}
        if g is not None:
            meta["group"] = int(g) + 1
        if reffits is not None:
            rf = np.asarray(reffits, float)
            rf = rf[sel] if rf.ndim == 2 and rf.shape[0] == n else rf
            rf = rf[np.all(np.isfinite(rf), axis=1)] if rf.ndim == 2 else rf
            if rf.size:
                meta["airy"] = np.median(rf, axis=0) if rf.ndim == 2 else rf
                if rf.ndim == 2 and int(pre_bin) <= 1:
                    rf_all = np.asarray(reffits, float)[sel] if np.asarray(reffits).ndim == 2 and \
                        np.asarray(reffits).shape[0] == n else None
                    if rf_all is not None:
                        meta["airy_frames"] = rf_all          # frame j -> its own fit (pyNOMIC inject_source)
        texp = float(sel.size * texp_per_frame) if texp_per_frame else float(sel.size)
        out[pid] = Dataset(cube, ang, tags, texp=texp, name=pid, meta=meta)
        log(f"  {pid}: {cube.shape[0]} frames (pre_bin {pre_bin}), PA span {ang.min():.1f}..{ang.max():.1f} deg, "
            f"tags {'yes' if tags else 'no'}")
    return out


def image_groups(times: np.ndarray, positions: np.ndarray, smooth: float = 100.0) -> np.ndarray:
    """Split a sequence where the smoothed star position jumps, as pyNOMIC's ``helper_functions.image_groups`` does.

    The positions are Gaussian-smoothed over ``smooth`` frames and cubic-splined onto a fine time
    grid.  ``|d position / d t|`` is smoothed again, and every peak above five standard deviations of
    the quiet part (``derivative < 2 * mean``) is a split.

    Parameters
    ----------
    times : ndarray, shape (n,)
        Time of each frame, such as its JD.
    positions : ndarray, shape (n,)
        The star's position on the detector in each frame, in pixels.
    smooth : float, default 100.0
        Width of the Gaussian smoothing, in frames.

    Returns
    -------
    labels : ndarray of int, shape (n,)
        Group of each frame, from 0 and contiguous in time.  Times out of order are sorted first, and
        fewer than 3 frames or repeated times give one group.
    """
    from scipy.interpolate import CubicSpline
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    t = np.asarray(times, float)
    x = np.asarray(positions, float)
    n = len(t)
    if n < 3 or not np.all(np.diff(t) > 0):
        order = np.argsort(t, kind="stable")
        if n >= 3 and np.all(np.diff(t[order]) > 0):
            lab = np.empty(n, int)
            lab[order] = image_groups(t[order], x[order], smooth)
            return lab
        return np.zeros(n, int)
    xs = gaussian_filter1d(x, float(smooth), mode="nearest")
    spl = CubicSpline(t, xs)
    grid = np.linspace(t[0], t[-1], 10 ** 5)
    deriv = np.abs(spl(grid, 1))
    deriv = gaussian_filter1d(deriv, float(smooth) * (len(grid) / max(n, 1)), mode="nearest")
    quiet = deriv[deriv < 2 * np.mean(deriv)]
    cut = 5 * np.std(quiet) if quiet.size else np.inf
    peaks, _ = find_peaks(deriv, prominence=cut)
    lab = np.zeros(n, int)
    for pk in peaks:
        lab[t > grid[pk]] += 1
    # renumber contiguously (a peak between two frames that leaves no frame behind it)
    _, lab = np.unique(lab, return_inverse=True)
    return lab.astype(int)


def group_labels(spec, times: Optional[np.ndarray], positions: Optional[np.ndarray], n: int,
                 smooth: float = 100.0, log: Callable[[str], None] = print) -> Optional[np.ndarray]:
    """Resolve a ``groups`` specification into per-frame labels.

    Parameters
    ----------
    spec : str, sequence or None
        None, "none" or "off" for no grouping, "auto" for :func:`image_groups` on the times and
        positions, ``n`` whole numbers for labels as they are, or a shorter sequence of JD split
        times.
    times : ndarray or None
        Time of each frame.
    positions : ndarray or None
        The star's position in each frame, in pixels, for "auto".
    n : int
        Number of frames.
    smooth : float, default 100.0
        Width of :func:`image_groups`' smoothing, in frames.
    log : callable, default print
        Function called with the groups' sizes.

    Returns
    -------
    labels : ndarray of int or None
        Group of each frame, renumbered from 0, or None without grouping.

    Raises
    ------
    ValueError
        If ``spec`` is an unknown string, or the times or positions it needs are missing.
    """
    if spec is None or (isinstance(spec, str) and spec.lower() in ("", "none", "off")):
        return None
    if isinstance(spec, str):
        if spec.lower() != "auto":
            raise ValueError(f"groups={spec!r}: expected 'auto', 'none', per-frame labels or split times")
        if times is None or positions is None:
            raise ValueError("groups='auto' needs the frame times and star positions "
                             "(<obj>_NOMIC_psf_subtraction.npz maxima or <obj>_NOMIC_aligned.npz original_psf_locs)")
        lab = image_groups(times, positions, smooth)
    else:
        arr = np.asarray(spec)
        if arr.ndim == 1 and len(arr) == n and np.issubdtype(arr.dtype, np.integer):
            lab = arr.astype(int)
        elif arr.ndim == 1 and len(arr) == n and np.all(np.mod(arr, 1) == 0):
            lab = arr.astype(int)
        else:
            if times is None:
                raise ValueError("split-time groups need the frame times")
            lab = np.searchsorted(np.sort(np.asarray(arr, float)), np.asarray(times, float), side="right")
    _, lab = np.unique(lab, return_inverse=True)
    counts = np.bincount(lab)
    log("  image groups: " + ", ".join(f"g{k + 1}: {c} frames ({100.0 * c / max(n, 1):.0f}%)" for k, c in enumerate(counts)))
    return lab.astype(int)


def _npz_arrays(path: str) -> List[Any]:
    """Read an ``.npz`` file's positional arrays, ``arr_0`` onward, in order.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    arrays : list
        The arrays.
    """
    z = np.load(path, allow_pickle=True)
    return [z[k] for k in sorted(z.files, key=lambda s: int(s.split("_")[1]) if s.startswith("arr_") else 0)]


def load_pynomic(workdir: str, obj: str, frames: Union[str, Sequence] = "masked", binned: bool = False,
                 name: Optional[str] = None, use_frame_rejection: Optional[Sequence[bool]] = None,
                 groups=None, group_smooth: float = 100.0, **kw) -> Dict[str, Dataset]:
    """Load a pyNOMIC reduction directory, where the notebook ran.

    Parameters
    ----------
    workdir : str
        The directory.
    obj : str
        Name of the object, the prefix of the ``.npz`` files.
    frames : str or sequence, default "masked"
        A directory under ``workdir`` whose sorted ``*.fits`` are the frames, such as "masked" or
        "aligned", or a list of paths.
    binned : bool, default False
        Whether to use ``<obj>_NOMIC_binned_evaluated.npz``, pyNOMIC's own temporal bins with their
        angles, chops and tags, instead of the unbinned frames.
    name : str or None, default None
        Prefix of the partition identifiers.  None takes ``obj``.
    use_frame_rejection : sequence of bool or None, default None
        pyNOMIC's ``frame_rejection`` output, the frames kept beforehand.
    groups : str, sequence or None, default None
        Image-group specification (:func:`group_labels`), one partition per chop state and group.
        None makes the chop states the partitions.
    group_smooth : float, default 100.0
        Width of the "auto" split's smoothing, in frames.
    kw : dict
        More keywords for :func:`datasets_from_arrays`, such as ``crop_half``, ``parang_sign``,
        ``pre_bin``, ``max_frames`` or ``group_min_frames``.

    Returns
    -------
    datasets : dict of str to Dataset
        The partitions, by identifier.

    Raises
    ------
    ValueError
        If the frame files do not match the chop entries.
    """
    name = name or obj
    if binned:
        arr = _npz_arrays(os.path.join(workdir, f"{obj}_NOMIC_binned_evaluated.npz"))
        (fwhms, eccs, psfmax, backdev, corrs, resdev, lbtfits, reffits, bfiles, bchops, bangles) = arr[:11]
        bfiles = [os.path.join(workdir, f) if not os.path.isabs(str(f)) else str(f) for f in bfiles]
        return datasets_from_arrays(bfiles, bchops, bangles, corrs, backdev, resdev, reffits,
                                    frame_bool=use_frame_rejection, name=name, **kw)
    files_c, chops, header_info = _npz_arrays(os.path.join(workdir, f"{obj}_NOMIC_chop_correction.npz"))[:3]
    angles = np.asarray(header_info[1], float)
    times = np.asarray(header_info[2], float) if len(header_info) > 2 else None
    glab = None
    if groups is not None and not (isinstance(groups, str) and groups.lower() in ("", "none", "off")):
        pos = None
        for fn, idx, col in ((f"{obj}_NOMIC_psf_subtraction.npz", 1, 0), (f"{obj}_NOMIC_aligned.npz", 1, 0)):
            fp = os.path.join(workdir, fn)
            if os.path.exists(fp):
                try:
                    arr = np.asarray(_npz_arrays(fp)[idx], float)
                    if arr.ndim == 2 and arr.shape[0] == len(chops):
                        pos = arr[:, col]           # pyNOMIC: hf.image_groups(header_info[2], maxima[:, 0])
                        break
                except Exception:
                    continue
        glab = group_labels(groups, times, pos, len(chops), smooth=group_smooth, log=kw.get("log", print))
    if isinstance(frames, str):
        flist = sorted(glob.glob(os.path.join(workdir, frames, "*.fits")))
        flist = [f for f in flist if not os.path.basename(f).startswith(".")]
    else:
        flist = [str(f) for f in frames]
    if len(flist) != len(chops):
        raise ValueError(f"{len(flist)} frame files but {len(chops)} chop entries -- pass frames= explicitly")
    ev = os.path.join(workdir, f"{obj}_NOMIC_evaluated.npz")
    corrs = backdev = resdev = reffits = None
    if os.path.exists(ev):
        arr = _npz_arrays(ev)
        fwhms, eccs, psfmax, backdev, corrs, resdev, lbtfits, reffits = arr[:8]
    return datasets_from_arrays(flist, chops, angles, corrs, backdev, resdev, reffits,
                                frame_bool=use_frame_rejection, name=name, groups=glab, **kw)


# ----------------------------------------------------------------------------
# reducer / space / guard / objective
# ----------------------------------------------------------------------------
def injection_model_for(ds: Dataset, fwhm_px: float, psf: str = "airy", lam_over_d_px: float = NOMIC.lam_over_d_px,
                        r_ee_px: Optional[float] = None) -> InjectionModel:
    """Build the injection PSF of one partition.

    Parameters
    ----------
    ds : Dataset
        The partition.
    fwhm_px : float
        FWHM of the Gaussian fallback, in pixels.
    psf : {"airy", "frame", "gaussian"}, default "airy"
        "airy" injects frame ``j`` with pyNOMIC's Airy fit of frame ``j`` (``meta["airy_frames"]``), as
        pyNOMIC's ``inject_source`` does, with the median fit for the matched filter, the FWHM and the
        flux unit.  "frame" takes each frame's own star as an empirical template
        (:class:`~klip_tpe.injection.FramePSF`), and "gaussian" a Gaussian of flux unit 1, for smoke
        tests only.
    lam_over_d_px : float, default NOMIC.lam_over_d_px
        lambda/D in pixels, which sets the default radius of "frame".
    r_ee_px : float or None, default None
        Radius of the unit-flux normalization of "frame", in pixels.  None takes ``1.5`` lambda/D.

    Returns
    -------
    model : InjectionModel
        The model.  "airy" without a usable median fit falls back to the Gaussian.
    """
    a = ds.meta.get("airy")
    if psf == "frame":
        r_ee = float(r_ee_px) if r_ee_px else 1.5 * lam_over_d_px
        return FramePSF(ds.cube, r_ee=r_ee, r_stamp=min(1.7 * r_ee, ds.cube.shape[-1] / 2.0 - 1))
    if psf == "airy" and a is not None and len(a) >= 5 and np.isfinite(a[:5]).all() and a[1] > 0 and a[2] > 0:
        fp = ds.meta.get("airy_frames")
        if fp is not None and len(fp) != ds.cube.shape[0]:
            fp = None                                   # pre-binned: per-frame fits no longer line up
        return AiryPSF(sigmax=float(a[1]), sigmay=float(a[2]), p=float(a[4]), amp=float(a[0]), frame_params=fp)
    return GaussianPSF(fwhm_px, star_flux=1.0)


def fitted_fwhm(ds: Dataset) -> Optional[float]:
    """Return the core FWHM from pyNOMIC's median Airy fit, ``1.028 * mean(sigmax, sigmay)``.

    Parameters
    ----------
    ds : Dataset
        The partition.

    Returns
    -------
    fwhm : float or None
        FWHM in pixels, or None without a usable fit.
    """
    a = ds.meta.get("airy")
    if a is not None and len(a) >= 3 and np.isfinite(a[1:3]).all() and a[1] > 0 and a[2] > 0:
        return float(1.028 * 0.5 * (a[1] + a[2]))
    return None


def make_reducer(datasets: Dict[str, Dataset], weighting: str = "equal", max_workers="auto", pool: str = "auto",
                 fast: bool = False, n_min_ref: int = 10, pxscale: float = NOMIC.pxscale,
                 lam_m: float = NOMIC.lam, diam_m: float = NOMIC.diam, truenorth: float = NOMIC.truenorth,
                 fwhm_px: Optional[float] = None, outrad_cap: float = NOMIC.outrad_cap,
                 do_destripe: bool = False, psf: str = "airy", r_ee_px: Optional[float] = None,
                 backend: str = "klip", log: Callable[[str], None] = print) -> PartitionedReducer:
    """Build one reducer per partition, combined by a :class:`~klip_tpe.reducer.PartitionedReducer`.

    A partition is a chop state, night and image group.

    Parameters
    ----------
    datasets : dict of str to Dataset
        The partitions.
    weighting : {"equal", "sqrt_texp"}, default "equal"
        Weights of the combination.
    max_workers : int or str, default "auto"
        The thread budget.
    pool : {"auto", "processes", "threads"}, default "auto"
        Kind of worker pool.
    fast : bool, default False
        Default of the reducers' ``fast``.
    n_min_ref : int, default 10
        Default minimum number of references per target.
    pxscale : float, default NOMIC.pxscale
        Plate scale, in arcsec per pixel.
    lam_m : float, default NOMIC.lam
        Wavelength, in meters.
    diam_m : float, default NOMIC.diam
        Telescope diameter, in meters.
    truenorth : float, default NOMIC.truenorth
        True-north offset, in degrees.
    fwhm_px : float or None, default None
        FWHM of the PSF, in pixels.  None takes the median fitted core FWHM (:func:`fitted_fwhm`),
        else ``1.028`` lambda/D.
    outrad_cap : float, default NOMIC.outrad_cap
        Maximum outer radius of a zone, in pixels.
    do_destripe : bool, default False
        Default of the reducers' destriping.
    psf : {"airy", "frame", "gaussian"}, default "airy"
        The injection model (:func:`injection_model_for`).
    r_ee_px : float or None, default None
        Radius of the unit-flux normalization of "frame", in pixels.
    backend : {"klip", "pyklip", "vip"}, default "klip"
        The reduction backend.
    log : callable, default print
        Function called with each log line.

    Returns
    -------
    reducer : PartitionedReducer
        The combined reducer.
    """
    reducers = {}
    lod = (lam_m / diam_m) * 206265.0 / pxscale
    for pid, ds in datasets.items():
        fw = fwhm_px or fitted_fwhm(ds) or 1.028 * lod
        model = injection_model_for(ds, fw, psf=psf, lam_over_d_px=lod, r_ee_px=r_ee_px)
        red = reducer_class(backend)(ds, pxscale=pxscale, lam_m=lam_m, diam_m=diam_m, injection_model=model,
                          fallback_model=None, truenorth=truenorth, fwhm_px=fw, zone_pad=2.0,
                          outrad_cap=outrad_cap,
                          defaults={"fast": fast, "n_min_ref": n_min_ref, "comb_type": "nwadi",
                                    "do_destripe": bool(do_destripe)})
        red.name = f"nomic_{pid}"
        reducers[pid] = red
        extra = f", {model.n_bad} frames without a usable core" if getattr(model, "n_bad", 0) else ""
        log(f"  {pid}: injection model {model.name} (star flux unit {model.flux_unit:.3g}{extra}), fwhm {fw:.2f} px")
    pr = PartitionedReducer(reducers, weighting=weighting, max_workers=max_workers, pool=pool, log=log)
    pr.name = "nomic"
    return pr


def make_space(reducer: PartitionedReducer, **kw) -> SearchSpace:
    """Build the NEAR production parameterization, 9 parameters per partition and the drop dimensions.

    Parameters
    ----------
    reducer : PartitionedReducer
        The reducer.
    kw : dict
        Keywords of :func:`klip_tpe.instruments.search_space.make_space`.

    Returns
    -------
    space : SearchSpace
        The search space.
    """
    return _space.make_space(reducer, **kw)


def make_guard(reducer: PartitionedReducer, n_min_ref: int = 10, ref_frac: float = 0.95, k_max: int = 100,
               frame_selection: bool = True):
    """Build the frame-selection snapping and the reference-count guardrail, from the reducer's own lambda/D.

    The binning-span rule and the unit of ``angsep`` come from the reducer.

    Parameters
    ----------
    reducer : PartitionedReducer
        The reducer.
    n_min_ref : int, default 10
        Minimum number of references per target.
    ref_frac : float, default 0.95
        Fraction of the target frames that must keep ``n_min_ref`` references.
    k_max : int, default 100
        Maximum number of KL modes.
    frame_selection : bool, default True
        Whether to snap the frame-selection dimensions to the no-cut value.

    Returns
    -------
    guard : callable
        The guard.
    """
    r0 = next(iter(reducer.reducers.values()))
    ref = ReferenceCountGuard(angles_fn=reducer.frame_angles, tags_fn=reducer.frame_tags,
                              fwhm_px=1.028 * r0.lam_over_d_px, lam_over_d_px=r0.lam_over_d_px,
                              n_min_ref=n_min_ref, ref_frac=ref_frac, k_max=k_max)
    if not frame_selection:
        return ref
    fs = FrameSelectionGuard(tags_fn=reducer.frame_tags, nframes_fn=lambda pid: reducer.reducers[pid].data.nframes)
    return compose(fs, ref)


def default_config(reducer: PartitionedReducer, clean_subtract: bool = True, known=(),
                   metric: str = "mawet"):
    """Build the objective, the matched-filter Mawet S/N with the Airy kernel, and the position sampler.

    Parameters
    ----------
    reducer : PartitionedReducer
        The reducer, whose plate scale, FWHM and matched filter the metric takes.
    clean_subtract : bool, default True
        Whether the objective subtracts the clean image's score.
    known : sequence of tuple of 2 floats, default ()
        Real companions, as ``(rho_arcsec, pa_deg)``.
    metric : {"mawet", "fmmf"}, default "mawet"
        The metric.  "fmmf" forward-models the kernel through each subtraction.

    Returns
    -------
    objective : Objective
        The objective.
    sampler : PositionSampler
        The sampler of injection positions.
    """
    kw = dict(pxscale=reducer.pxscale, fwhm=reducer.fwhm, kernel_fn=reducer.matched_filter_kernel,
              known=list(known))
    if str(metric).lower() == "fmmf":
        from ..fmmf import FMMFSNR
        metric = FMMFSNR(**kw)
    else:
        metric = MawetPeakSNR(**kw)
    objective = Objective(metric, clean_subtract=clean_subtract)
    sampler = PositionSampler(fwhm_as=reducer.fwhm * reducer.pxscale, known=list(known))
    return objective, sampler
