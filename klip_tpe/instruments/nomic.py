"""LBTI/NOMIC adapter for data pre-processed with **pyNOMIC**
(https://github.com/vivekv7323/pyNOMIC).

pyNOMIC ends with registered (star-centred), background-subtracted, per-frame FITS
files plus a handful of ``.npz`` side files written by its notebooks.  This module
turns those into :class:`~klip_tpe.reducer.Dataset` objects -- one per *partition*
(chop state, and night if you pass several) -- and builds the reducer / search space /
guardrail / objective exactly like the NEAR adapter does, so ``Runner`` and the CLI
work unchanged.

What is read (positional arrays, in the order the pyNOMIC notebooks ``np.savez`` them):

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

Frame-quality tags map onto the optimizer's three frame-selection dimensions as

* ``corrs``     <- ``correlations``  (cross-correlation with the reference PSF; the
  searched ``corr_thresh`` keeps frames with ``corrs >= corr_thresh``),
* ``noises``    <- ``background_dev / median``  (``noise_max`` multiplier),
* ``coronoise`` <- ``residual_dev / median``    (``coronoise_max`` multiplier; PSF-fit
  residual is the closest NOMIC analogue of NEAR's coronagraphic-residual tag).

The injection model is pyNOMIC's own: the obstructed-Airy fit of **each frame**
(``reffits`` row ``j`` for frame ``j``), so a companion of contrast ``c`` in frame ``j``
is exactly pyNOMIC's ``inject_source`` -> ``airy_disk(c * amp_j, sigmax_j, sigmay_j, p_j)``.
``psf='frame'`` (each frame's own star as an empirical template) is available as an option.

**Groups.**  pyNOMIC's notebooks split a sequence into *image groups* -- contiguous
time segments separated by jumps of the star position on the detector (nod / pointing
offsets), ``hf.image_groups(header_info[2], maxima[:, 0])`` -- and drop the shifted ones
by hand.  Here a group can be a *partition of its own*: ``groups="auto"`` runs the same
split (ported below as :func:`image_groups`, the star positions read from
``<obj>_NOMIC_psf_subtraction.npz`` ``maxima`` or ``<obj>_NOMIC_aligned.npz``
``original_psf_locs``), ``groups=<per-frame labels>`` or ``groups=[JD split times]`` use
your own, and every (night x chop state x group) then becomes one partition
``<name><A|B>g<k>`` with its own KLIP basis and parameter block.  The optimizer treats
nights and groups identically -- they are all partitions: the drop1/drop2 selection dims
(or the binary scheme) decide which ones enter the combination, the per-partition blocks
tune each one, and the products / display panels ("night inclusion", "night effect") show
them side by side.  Groups with fewer than ``group_min_frames`` frames are left out.

Conventions to confirm on a known binary before trusting position angles: the
loader assumes pyNOMIC's ``header_info[1]`` is the parallactic angle such that a
counter-clockwise rotation by ``parang_sign * parang + truenorth`` puts north up
(this is the convention pyNOMIC's ``inject_source`` uses: detector azimuth
``PA - parang``).  ``parang_sign=-1`` flips it; ``truenorth`` adds a fixed offset.
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
from . import near as _near

__all__ = ["NOMIC", "read_frames", "datasets_from_arrays", "load_pynomic", "image_groups", "group_labels",
           "make_reducer", "make_space",
           "make_guard", "default_config", "production_defaults"]


class NOMIC:
    """Instrument constants (single-aperture LBT, N' band)."""
    pxscale = 0.0179                 # arcsec / px (pyNOMIC default)
    lam = 11.1e-6                    # m, N' band centre
    diam = 8.4                       # m (one LBT aperture; Fizeau data: override in make_reducer)
    truenorth = 0.0                  # deg, added to the parallactic angle at derotation
    crop_half = 75                   # KLIP crop half size (150 px, like pyNOMIC's crop_size=150)
    lam_over_d_px = (lam / diam) * 206265.0 / pxscale     # 15.23
    fwhm_px = 1.028 * lam_over_d_px                       # 15.66 (pyNOMIC notebooks use ~14)
    outrad_cap = 70.0


production_defaults = dict(_near.production_defaults)


# ----------------------------------------------------------------------------
# reading pyNOMIC products
# ----------------------------------------------------------------------------
def _fits_data(path: str) -> np.ndarray:
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        d = h[0].data
        if d is None:
            d = h[1].data
    return np.asarray(d, np.float32)


def read_frames(files: Sequence[Union[str, os.PathLike]], crop_half: int = NOMIC.crop_half,
                center: Optional[Tuple[float, float]] = None, nan_policy: str = "zero",
                progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Read per-frame FITS files into a ``(n, 2h, 2h)`` float32 cube cropped around
    ``center`` (default: the frame centre ``((nx-1)/2, (ny-1)/2)``, which is where pyNOMIC's
    ``frame_registration`` puts the star; after cropping the star sits at ``(h-0.5, h-0.5)``
    exactly like the NEAR cubes).  ``nan_policy``: ``'zero'`` replaces NaN (pyNOMIC's
    ``mask_files`` NaN-masks the over-subtracted PSF ghosts) by 0, ``'keep'`` leaves them."""
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
    """Plain consecutive mean-binning by ``m`` frames (memory relief before the search;
    the searched ``bin`` then acts on these pre-binned frames)."""
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
    """Build one :class:`Dataset` per chop state (x image group) from pyNOMIC's arrays.

    ``groups`` = per-frame integer labels (see :func:`group_labels`): each (chop state,
    group) with at least ``group_min_frames`` frames becomes its own partition
    ``<name><A|B>g<k>``; without ``groups`` the partitions are ``<name>A`` / ``<name>B``.

    ``frame_bool`` (pyNOMIC ``frame_rejection`` output) pre-drops frames; the searched
    ``corr_thresh / noise_max / coronoise_max`` then act on the survivors through the tags.
    ``reffits`` (n, 7) Airy-fit parameters go to ``Dataset.meta['airy']`` (median) for the
    injection model.  Partition ids are ``f"{name}{suffix}"`` with suffix ``A``/``B``.
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
    """Port of pyNOMIC ``helper_functions.image_groups``: split a sequence where the
    (smoothed) star position jumps.  Returns an integer group label per frame
    (0 .. G-1, contiguous in time).  Same recipe as pyNOMIC: Gaussian-smooth the
    positions (kernel ``smooth`` frames), cubic-spline them onto a fine time grid,
    take |d position / d t|, smooth again, and call every peak above five standard
    deviations of the quiet part (``derivative < 2 * mean``) a split."""
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
    """Resolve a ``groups`` specification into per-frame labels: ``None``/``'none'`` -> no
    grouping; ``'auto'`` -> :func:`image_groups` on ``(times, positions)``; a sequence of
    ``n`` integers -> used as is; a shorter sequence of floats -> JD split times."""
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
    z = np.load(path, allow_pickle=True)
    return [z[k] for k in sorted(z.files, key=lambda s: int(s.split("_")[1]) if s.startswith("arr_") else 0)]


def load_pynomic(workdir: str, obj: str, frames: Union[str, Sequence] = "masked", binned: bool = False,
                 name: Optional[str] = None, use_frame_rejection: Optional[Sequence[bool]] = None,
                 groups=None, group_smooth: float = 100.0, **kw) -> Dict[str, Dataset]:
    """Load a pyNOMIC reduction directory (where the notebook ran).

    ``groups``: ``None`` (partitions = chop states), ``'auto'`` (pyNOMIC's image-group split
    of the sequence on the star position, :func:`image_groups`; ``group_smooth`` = its
    kernel), per-frame integer labels, or a list of JD split times -> one partition per
    (chop state, group).  ``group_min_frames`` (keyword, default 20) drops tiny groups.

    ``frames``: ``'masked'`` / ``'aligned'`` (a directory under ``workdir`` whose sorted
    ``*.fits`` are the frames, as in the notebooks), or an explicit list of paths.
    ``binned=True`` uses ``<obj>_NOMIC_binned_evaluated.npz`` (pyNOMIC's own temporal bins,
    with their angles/chops/tags) instead of the unbinned frames.  Other keywords go to
    :func:`datasets_from_arrays` (``crop_half, parang_sign, pre_bin, max_frames, ...``).
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
    """Injection PSF for one partition.

    ``psf='airy'`` (default, = pyNOMIC): frame ``j`` is injected with pyNOMIC's Airy fit of
    frame ``j`` (``ds.meta['airy_frames']``, the ``reffits`` rows); the median fit gives the
    matched filter, the FWHM and the reported flux unit.  This is what pyNOMIC's own
    ``inject_source`` does, so contrasts are directly comparable between the two tools.
    ``psf='frame'``: :class:`~klip_tpe.injection.FramePSF`, each frame's own star as an
    empirical template (needs the unsaturated core in the frames).
    ``psf='gaussian'``: instrument-FWHM Gaussian with flux unit 1 (smoke tests only)."""
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
    """Core FWHM (px) from pyNOMIC's median Airy fit, ``1.028 * mean(sigmax, sigmay)``."""
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
    """One :class:`KLIPReducer` per partition (chop state x night x image group).  ``psf`` selects the
    injection model (see :func:`injection_model_for`; default pyNOMIC's per-frame Airy fit).  ``fwhm_px`` defaults to the median fitted core FWHM when pyNOMIC's Airy fits are
    available, else ``1.028 lambda/D``."""
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
    """Same parameterisation as the NEAR production space (9 per partition + drop dims);
    see :func:`klip_tpe.instruments.near.make_space` for the keywords."""
    return _near.make_space(reducer, **kw)


def make_guard(reducer: PartitionedReducer, n_min_ref: int = 10, ref_frac: float = 0.95, k_max: int = 100,
               frame_selection: bool = True):
    """Frame-selection no-cut snapping followed by the reference-count guardrail, with the
    binning-span rule and ``angsep`` unit taken from the reducer itself."""
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
    """Objective (matched-filter Mawet S/N with the Airy kernel) + position sampler;
    ``metric='fmmf'`` forward-models that kernel through each subtraction instead."""
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
