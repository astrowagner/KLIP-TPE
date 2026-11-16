"""spaceKLIP and JWST backend, from spaceKLIP's products or the stage-2 files to the optimizer.

spaceKLIP's own PSF-subtraction step hands its stage-2 and ImageTools products to pyKLIP
(``pyklip.instruments.JWST.JWSTData``, then ``klip_dataset``).  This module makes the same hand-over
to the optimizer.  The science rolls become partitions, the reference-star exposures become the RDI
library, and :class:`~klip_tpe.backends.pyklip.PyKLIPReducer` subtracts, in mode ``"ADI+RDI"`` by
default as in spaceKLIP.  So the parameters the optimizer searches (``numbasis``, ``subsections``,
``movement`` or ``angsep``, the annulus, the frame selection, the binning and the high-pass) are the
knobs of a spaceKLIP reduction.

:func:`load_spaceklip` reads a ``spaceKLIP.Database``, or its ``obs`` table dict, with a
concatenation key, or lists of science and reference FITS files.  It uses pyKLIP's ``JWSTData`` when
that imports (it needs pyKLIP and astroquery), else :func:`read_jwst_files`, which does the same
header arithmetic (``STARCENX``/``STARCENY`` or ``CRPIX``, and ``PA = ROLL_REF - V3I_YANG * VPARITY``).

:func:`load_calints` reads stage-2 ``*_calints.fits`` files without spaceKLIP, with the background
subtraction, static hot pixels, destriping, bad-pixel repair, registration and crop a coronagraphic
JWST data set needs.

:func:`make_reducer` builds a :class:`~klip_tpe.backends.pyklip.PyKLIPReducer` per roll, or the
built-in annular KLIP, with a JWST PSF for injection.  The PSF is an STPSF model of the mask, a
webbpsf or ``webbpsf_ext`` offset PSF given as ``psf_template`` (a 2-D array or a FITS path, such as
spaceKLIP's ``analysistools.get_offsetpsf`` output or a ``JWST_PSF`` model), else a Gaussian of the
filter's ``1.028 lambda/D``, whose contrasts then need calibrating.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..injection import GaussianPSF, InjectionModel, TemplatePSF
from ..reducer import Dataset, KLIPReducer, PartitionedReducer
from .pyklip import PyKLIPReducer, dataset_from_pyklip

__all__ = ["load_spaceklip", "load_calints", "make_reducer", "read_jwst_files", "JWST_DIAM",
           "fill_dq_neighbours", "sigma_clip_repair", "shift_keeping_gaps", "blank_sky",
           "static_hot_pixels", "destripe_detector", "frames_provenance", "frames_signature"]

JWST_DIAM = 6.5   # m (effective; lambda/D for the angsep unit and the default FWHM)


def _background_done(*headers) -> bool:
    """Tell whether Image2's background step has run on an exposure (``S_BKDSUB = 'COMPLETE'``).

    Parameters
    ----------
    headers : FITS header or dict
        The exposure's headers, any of which may carry the keyword.

    Returns
    -------
    done : bool
        True when one of them says ``COMPLETE``.
    """
    for h in headers:
        v = h.get("S_BKDSUB") if hasattr(h, "get") else None
        if v is not None and str(v).strip().upper() == "COMPLETE":
            return True
    return False


def blank_sky(files: Sequence[str]) -> np.ndarray:
    """Return the median blank-sky frame of dedicated background pointings, for subtraction.

    ``calints`` are in MJy/sr, a surface brightness already divided by the integration time, so
    exposures of different lengths combine and subtract as they are.  The median runs over every
    integration of every file, so the background's cosmic rays do not print onto the science frames.
    DQ ``DO_NOT_USE`` pixels are left out.

    Parameters
    ----------
    files : sequence of str
        Paths of the background ``calints`` files.

    Returns
    -------
    sky : ndarray, shape (ny, nx)
        The median frame.  Pixels no background frame has, such as the unilluminated border of a MIRI
        coronagraphic subarray, are NaN, as they already are in the frames it is subtracted from.
    """
    from astropy.io import fits
    ims = []
    for f in files:
        with fits.open(f) as h:
            d = np.asarray(h["SCI"].data, float)
            dq = np.asarray(h["DQ"].data, int)
            ims.append(np.where((dq & 1).astype(bool), np.nan, d))
    with np.errstate(all="ignore"):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(np.concatenate(ims, axis=0), axis=0)


def static_hot_pixels(sky: np.ndarray, nsig: float = 10.0, min_excess: float = 5.0,
                      window: int = 7) -> np.ndarray:
    """Find the pixels a blank-sky frame shows as hot or dead that the DQ array did not flag.

    ``sky`` holds no star, so anything sharp in it is the detector.  A pixel is hot when it stands above
    the median of its finite ``window`` row neighbors and above that of its finite ``window`` column
    neighbors, itself left out, by more than ``nsig`` robust sigmas of that excess and by more than
    ``min_excess``.  Asking for both lines keeps the 4QPM glow sticks out, since a pixel on a stick is
    level with its neighbors along it.  A pixel the same amount below both is dead, and a pixel with
    fewer than 4 finite neighbors on either line, on the unilluminated border, is never flagged.

    Parameters
    ----------
    sky : ndarray, shape (ny, nx)
        The blank-sky median (:func:`blank_sky`).
    nsig : float, default 10.0
        Least excess, in robust sigmas of the excesses.
    min_excess : float, default 5.0
        Least excess, in image units (MJy/sr on JWST, where 5 is a quarter of the F1140C sky).
    window : int, default 7
        Length of each line of neighbors, in pixels, the pixel at its middle.

    Returns
    -------
    hot : ndarray of bool, shape (ny, nx)
        True at each hot or dead pixel.

    Notes
    -----
    ERS 1386 F1140C has an unflagged hot pixel 5 pixels from the star (detector x, y = 125, 111), 52
    MJy/sr above its neighbors on the blank sky and 80 in the reference frames.  The pipeline's
    background subtraction removes it only in part, leaving +12 to 15 MJy/sr in the science frames and
    +25 to 30 in the references.  KLIP cannot fit a fixed detector feature that the library carries at
    another level, so it survived every reduction as a bright spot 0.6" north of the star, where Carter
    et al. (2023), who replaced "~30 static hot pixels" by hand, show none.
    """
    b = np.asarray(sky, float)
    fin = np.isfinite(b)
    if not fin.any():
        return np.zeros(b.shape, bool)
    h = max(int(window) // 2, 1)
    ny, nx = b.shape

    def line_median(axis):
        """Return the median of the finite neighbors within ``h`` along ``axis``, the pixel left out,
        and their count."""
        stack, count = [], np.zeros(b.shape, int)
        for k in range(-h, h + 1):
            if k == 0:
                continue
            sh = np.full(b.shape, np.nan)
            if axis == 1:
                src, dst = (slice(None), slice(max(k, 0), nx + min(k, 0))), (slice(None), slice(max(-k, 0), nx + min(-k, 0)))
            else:
                src, dst = (slice(max(k, 0), ny + min(k, 0)), slice(None)), (slice(max(-k, 0), ny + min(-k, 0)), slice(None))
            sh[dst] = b[src]
            stack.append(sh)
            count += np.isfinite(sh)
        with np.errstate(all="ignore"):
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                return np.nanmedian(np.stack(stack), axis=0), count

    row_med, n_row = line_median(1)
    col_med, n_col = line_median(0)
    ok = fin & (n_row >= 4) & (n_col >= 4) & np.isfinite(row_med) & np.isfinite(col_med)
    up = b - np.maximum(row_med, col_med)          # excess over BOTH lines through the pixel
    down = np.minimum(row_med, col_med) - b        # deficit below both
    dev = np.where(up > 0, up, np.where(down > 0, -down, 0.0))
    dev = np.where(ok, dev, 0.0)
    v = dev[ok]
    sig = 1.4826 * float(np.median(np.abs(v - np.median(v)))) if v.size else 0.0
    thr = max(float(nsig) * sig, float(min_excess))
    return ok & (np.abs(dev) > thr)


def frames_provenance(reducer) -> Optional[Dict[str, Any]]:
    """Return the loader's record of the frames a reducer holds, from its first partition with one.

    Parameters
    ----------
    reducer : Reducer
        The reducer, a :class:`~klip_tpe.reducer.PartitionedReducer` or a single partition's.

    Returns
    -------
    provenance : dict or None
        :func:`load_calints`'s provenance from a ``Dataset.meta``, None without one.
    """
    subs = list(getattr(reducer, "reducers", {}).values()) or [reducer]
    for r in subs:
        rec = (getattr(getattr(r, "data", None), "meta", None) or {}).get("provenance")
        if rec:
            return rec
    return None


def frames_signature(provenance: Optional[Dict[str, Any]]) -> str:
    """Return a short, stable name for a frame treatment.

    The name covers the products (file names, pipeline version, CRDS context, pixel scale) and what the
    loader did to them (hot pixels, background, destriping, repair).  Two loads with the same name hand
    the reducer the same frames, so anything cached from one, such as a companion fit or a default
    image, holds for the other.

    Parameters
    ----------
    provenance : dict or None
        The loader's record (:func:`frames_provenance`).

    Returns
    -------
    signature : str
        The first 10 hexadecimal digits of SHA-1 of the record's key fields, ``"none"`` without a record.
    """
    if not provenance:
        return "none"
    import hashlib
    import json as _json
    hp = provenance.get("hot_pixels") or {}
    key = {"files": provenance.get("files"), "cal_ver": provenance.get("cal_ver"),
           "crds_ctx": provenance.get("crds_ctx"),
           "pxscale": round(float(provenance.get("pxscale") or 0.0), 9),
           "hot_pixels": sorted(map(tuple, hp.get("pixels") or [])) if hp.get("applied") else None,
           "background_subtracted": provenance.get("background_subtracted"),
           "destripe": provenance.get("destripe"), "repair": provenance.get("repair")}
    return hashlib.sha1(_json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:10]


def shift_keeping_gaps(im: np.ndarray, shift, order: int = 3) -> np.ndarray:
    """Shift an image as ``scipy.ndimage.shift`` does, without one NaN taking the whole image.

    At ``order`` 2 or more ``ndimage.shift`` prefilters with a recursive spline filter, so a single
    non-finite pixel reaches every pixel of the output.  Here the gaps are zero-filled for the
    interpolation and the same interpolator runs over the mask of finite pixels, a partition of unity.
    It comes back at 1 wherever the spline's support lay in good data, and the pixels where it does not,
    which would carry invented values, come back NaN.  The mask is extended with ``mode="nearest"``, so
    the frame's border, a boundary condition and not a gap, is not flagged.

    Parameters
    ----------
    im : ndarray, shape (ny, nx)
        The image.
    shift : float or sequence of 2 floats
        The shift, ``(dy, dx)`` in pixels as ``ndimage.shift`` takes it.
    order : int, default 3
        Order of the spline interpolation.

    Returns
    -------
    shifted : ndarray, shape (ny, nx)
        The shifted image, NaN where the interpolation reached a gap.  An image without gaps comes back
        as ``ndimage.shift`` gives it.

    Notes
    -----
    On a MIRI MASK1140 ``calints`` frame, whose unilluminated border is 13 % NaN after the bad-pixel
    repair, ``order=3`` returned 99.2 % NaN and ``order=1`` 13.3 %.  That took the whole HIP 65426
    F1140C cube to NaN inside :func:`load_calints`, and the failure surfaced two steps downstream as
    "every frame was dropped as empty" in the reducer.
    """
    from scipy import ndimage
    im = np.asarray(im, float)
    bad = ~np.isfinite(im)
    if not bad.any():
        return ndimage.shift(im, shift, order=order)
    out = ndimage.shift(np.where(bad, 0.0, im), shift, order=order)
    w = ndimage.shift((~bad).astype(float), shift, order=order, mode="nearest")
    out[np.abs(w - 1.0) > 1e-3] = np.nan
    return out


def _clip_sky(m: np.ndarray, clip: float) -> np.ndarray:
    """Sigma-clip an image about its global median, so a bright feature cannot drag a line's offset with it.

    Parameters
    ----------
    m : ndarray, shape (ny, nx)
        The image, masked pixels NaN.
    clip : float
        Clip, in robust sigmas.  0 or less clips nothing.

    Returns
    -------
    clipped : ndarray, shape (ny, nx)
        The image, NaN beyond the clip.

    Notes
    -----
    A line whose pixels are all rejected gets no offset and is left alone.  Its unclipped median would
    be worse.  On a real F1140C integration a global median of 21 and a MAD of 3.1 reject 31 rows and 91
    columns outright, since those lines run through the PSF wings and not the sky, and their unclipped
    medians are 240 to 470 MJy/sr.  Subtracting those took the stellar peak from 627 to 256.
    """
    if clip <= 0:
        return m
    with np.errstate(all="ignore"):
        med = np.nanmedian(m)
        mad = np.nanmedian(np.abs(m - med)) * 1.4826
    if not np.isfinite(mad) or mad <= 0:
        return m
    return np.where(np.abs(m - med) < clip * mad, m, np.nan)


def destripe_detector(cube: np.ndarray, center: Optional[Tuple[float, float]] = None,
                      star_radius_px: float = 45.0, boundary_px: float = 12.0,
                      clip: float = 4.0, columns: bool = True,
                      min_line_px: int = 30) -> Tuple[np.ndarray, Dict[str, float]]:
    """Remove each frame's offset per row, then per column, measured on masked sky.

    It runs in the detector frame on the full subarray, before any crop.  The offset has to be measured
    where the astrophysical signal is not, and on MIRI's 288 x 224 MASK1140 subarray a row is mostly
    sky, while on the 81 x 81 crop the reducer takes every row passes through the coronagraphic PSF.
    Destriping the crop would subtract the target.  The mask takes out ``star_radius_px`` around
    ``center`` (the PSF and its halo) and ``boundary_px`` about the 4QPM boundaries and the glow sticks
    along them, which the row offsets would otherwise take in, and ``clip`` sigma-clips the rest.  Each
    offset is subtracted from the whole line, masked pixels included.  A line that the mask leaves
    thinner than ``min_line_px`` pixels falls back to the star's mask alone, and a line with no unmasked
    finite pixel gets no offset.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        The frames, in the detector frame.
    center : tuple of 2 floats or None, default None
        The star, ``(x, y)`` in 0-based pixels.  None masks nothing but the clip.
    star_radius_px : float, default 45.0
        Radius of the star's mask, in pixels.
    boundary_px : float, default 12.0
        Half-width of the mask along each axis through the star, the 4QPM boundaries, in pixels.
    clip : float, default 4.0
        Clip of the sky, in robust sigmas (:func:`_clip_sky`).
    columns : bool, default True
        Whether to remove the columns' offsets after the rows'.
    min_line_px : int, default 30
        Fewest unmasked pixels a line keeps before it falls back to the star's mask alone.

    Returns
    -------
    cube : ndarray, shape (n, ny, nx)
        The destriped frames.
    stats : dict
        ``row_sigma`` and ``col_sigma``, the mean over frames of the offsets' scatter.

    Notes
    -----
    On a raw F1140C integration the per-row offset scatters by 3.1 to 3.7 MJy/sr against a
    pixel-to-pixel scatter of 3.0 to 3.7, and destriping lowers the sky's scatter by 1.63 to 1.77
    times.  In the pipeline it runs after the blank-sky subtraction, where the gain is 1.28 times
    (per-row sigma 1.05, per-column 0.327, frame scatter 1.27 to 0.991, on HIP 65426 F1140C).  The row
    pattern correlates at +0.997 between the integrations of an exposure, so it is a static detector
    pattern and not random 1/f noise.  Being static, it is largely common to the science and background
    pointings, which is why the blank-sky median takes most of it, and why the rest survives KLIP as a
    fixed shape that derotation smears around the field.

    Before the fallback, the star's disk and the boundary band crossed at the star and emptied rows
    101 to 124 and columns 108 to 131 of MASK1140, straight through the target.  Those lines kept their
    offsets while every other line lost its own, which made the line-to-line steps inside the 81 x 81
    crop go from 1.87 raw to 4.05 destriped along rows.  With the fallback they go to 1.46.
    """
    cube = np.asarray(cube, float)
    if cube.ndim != 3 or cube.shape[0] == 0:
        return cube, {"row_sigma": 0.0, "col_sigma": 0.0}
    ny, nx = cube.shape[1:]
    yy, xx = np.mgrid[0:ny, 0:nx]
    blocked = np.zeros((ny, nx), bool)
    disc = np.zeros((ny, nx), bool)
    if center is not None:
        cx, cy = float(center[0]), float(center[1])
        if star_radius_px > 0:
            disc = np.hypot(xx - cx, yy - cy) < star_radius_px
        blocked |= disc
        if boundary_px > 0:
            blocked |= (np.abs(yy - cy) < boundary_px) | (np.abs(xx - cx) < boundary_px)
    # A line the mask empties gets NO offset, and that is worse than not destriping at all:
    # every other line loses its offset while this one keeps it, which turns a smooth
    # gradient into a step.  The star disc and the boundary band cross at the star, so on
    # MASK1140 the two together covered rows 101-124 and columns 108-131 outright -- 24 of
    # each, straight through the target, and the crop is centered on exactly that.  Measured
    # on a real F1140C integration, line-to-line steps inside the 81x81 crop went from 1.87
    # raw to 4.05 destriped along rows; with the fallback below they go to 1.46.  The global
    # sky scatter never noticed (1.826 vs 1.827), because the damage was confined to the one
    # region the science comes from.
    #
    # So where the full mask leaves a line too thin to measure, that line falls back to the
    # star disc alone: the ~200 rows away from the boundary keep their glow-stick protection,
    # and the ones crossing it are still corrected.
    per_axis = {}
    for axis in (1, 0):
        m = blocked.copy()
        thin = (~m).sum(axis=axis) < int(min_line_px)
        if thin.any() and disc.any():
            if axis == 1:
                m[thin, :] = disc[thin, :]
            else:
                m[:, thin] = disc[:, thin]
        per_axis[axis] = m
    import warnings as _w
    out = cube.copy()
    rs, cs = [], []
    ctx = _w.catch_warnings()
    ctx.__enter__()
    _w.simplefilter("ignore", RuntimeWarning)   # all-NaN rows are expected and handled below
    for i in range(out.shape[0]):
        m = _clip_sky(np.where(per_axis[1], np.nan, out[i]), clip)
        with np.errstate(all="ignore"):
            row = np.nanmedian(m, axis=1, keepdims=True)
        rs.append(float(np.nanstd(row)))
        out[i] = out[i] - np.where(np.isfinite(row), row, 0.0)
        if columns:
            m2 = _clip_sky(np.where(per_axis[0], np.nan, out[i]), clip)
            with np.errstate(all="ignore"):
                col = np.nanmedian(m2, axis=0, keepdims=True)
            cs.append(float(np.nanstd(col)))
            out[i] = out[i] - np.where(np.isfinite(col), col, 0.0)
    stats = {"row_sigma": float(np.nanmean(rs) if rs else 0.0),
             "col_sigma": float(np.nanmean(cs) if cs else 0.0)}
    ctx.__exit__(None, None, None)
    return out, stats


def fill_dq_neighbours(im: np.ndarray, maxit: int = 20) -> Tuple[np.ndarray, int]:
    """Replace each non-finite pixel by the median of its finite orthogonal and diagonal neighbors.

    It is spaceKLIP's bad-pixel treatment (Carter et al. 2023, section 2.2), and it touches only the
    pixels the pipeline flagged, DQ ``DO_NOT_USE``, which :func:`load_calints` has turned into NaN.
    Clusters close from their rims inward.  A coronagraphic PSF is full of sharp, isolated blobs, the
    six-lobed Lyot-stop pattern and, for a companion behind MASK335R, a core of three bars, and a filter
    that decides from the values what an outlier is eats those (:func:`sigma_clip_repair`).

    Parameters
    ----------
    im : ndarray, shape (ny, nx)
        The image, flagged pixels NaN.
    maxit : int, default 20
        Number of passes at most.

    Returns
    -------
    filled : ndarray, shape (ny, nx)
        The image, NaN where a gap was too wide to close.
    n_flagged : int
        Number of non-finite pixels in ``im``, those it set out to fill.
    """
    out = np.array(im, float)
    n0 = int((~np.isfinite(out)).sum())
    ny, nx = out.shape
    for _ in range(int(maxit)):
        bad = ~np.isfinite(out)
        if not bad.any():
            break
        p = np.pad(out, 1, constant_values=np.nan)
        st = np.stack([p[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(st, axis=0)
        out[bad] = med[bad]
    return out, n0


def sigma_clip_repair(im: np.ndarray, size: int = 5, nsig: float = 7.0) -> Tuple[np.ndarray, int, int]:
    """Repair an image by a median filter, the outlier repair of ``load_calints`` before version 1.1.

    Each pixel that differs from the ``size x size`` median filter by more than ``nsig`` times the
    frame-wide robust scatter of that difference is replaced by the median.  Do not use it on
    coronagraphic frames.  The scatter is set by the empty sky, so the threshold is a few counts, and
    every pixel of the structured PSF, star and companion alike, exceeds it.  It is kept so that runs
    made before the fix can be reproduced (``repair="sigma"``).

    Parameters
    ----------
    im : ndarray, shape (ny, nx)
        The image, flagged pixels NaN.
    size : int, default 5
        Width of the median filter, in pixels.
    nsig : float, default 7.0
        Threshold, in robust sigmas of the difference from the filter.

    Returns
    -------
    repaired : ndarray, shape (ny, nx)
        The image.
    n_rewritten : int
        Number of pixels replaced.
    n_flagged : int
        Number of those that were non-finite.

    Notes
    -----
    On ERS 1386 F444W it rewrote about 4,500 to 5,500 pixels of each 320 x 320 frame, of which only
    about 1,560 were DQ-flagged.  The rest was the PSF, median-filtered.  That blurred HIP 65426 b from
    the three-bar core of Carter et al. (2023) into one blob at 39 % of its peak, and since the pixels
    rewritten differ from frame to frame it left a residual that depends on the roll, which was taken
    for a speckle.
    """
    from scipy import ndimage
    im = np.asarray(im, float)
    bad = ~np.isfinite(im)
    med = ndimage.median_filter(np.where(bad, np.nanmedian(im), im), size=int(size))
    r = np.abs(im - med)
    s = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
    hit = bad | (r > float(nsig) * s)
    return np.where(hit, med, im), int(hit.sum()), int(bad.sum())


def _files_from_database(database, key: str) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Return a spaceKLIP concatenation's science and reference files and its observation details.

    Parameters
    ----------
    database : spaceKLIP.Database or dict
        The database, or its ``obs`` dict of tables.
    key : str
        The concatenation.

    Returns
    -------
    sci : list of str
        The science files (``TYPE`` "SCI").
    ref : list of str
        The reference files (``TYPE`` "REF").
    info : dict
        The first row's ``FILTER``, ``CWAVEL``, ``PIXSCALE``, ``TELESCOP``, ``INSTRUME`` and
        ``APERNAME``, where the table has them.
    """
    obs = database.obs if hasattr(database, "obs") else database
    tab = obs[key]
    sci = [str(f) for t, f in zip(tab["TYPE"], tab["FITSFILE"]) if t == "SCI"]
    ref = [str(f) for t, f in zip(tab["TYPE"], tab["FITSFILE"]) if t == "REF"]
    info = {}
    for col in ("FILTER", "CWAVEL", "PIXSCALE", "TELESCOP", "INSTRUME", "APERNAME"):
        if col in tab.colnames:
            info[col] = tab[col][0]
    return sci, ref, info


def read_jwst_files(filepaths: Sequence[str], center_keywords: Tuple[str, str] = ("STARCENX", "STARCENY")
                    ) -> Dict[str, np.ndarray]:
    """Read spaceKLIP's stage-2 or ImageTools ``*_calints.fits`` products into arrays.

    It does what ``pyklip.instruments.JWST.JWSTData.readdata`` does, without the SVO lookup.  The star is
    at ``center_keywords``, else at ``CRPIX``, moved by the dither offset ``XOFFSET``/``YOFFSET``, and
    the position angle is ``ROLL_REF - V3I_YANG * VPARITY``.

    Parameters
    ----------
    filepaths : sequence of str
        Paths of the files.
    center_keywords : tuple of 2 str, default ("STARCENX", "STARCENY")
        The ``SCI`` header's keywords of the star's x and y, in 1-based pixels.

    Returns
    -------
    data : dict
        ``input`` (the integrations, shape ``(n, ny, nx)`` for ``n`` integrations), ``centers``
        (shape ``(n, 2)``, 0-based), ``PAs``, ``filenums``, ``filenames``, ``wvs`` (the wavelength, in
        m) and ``pxscale`` (the median plate scale, in arcsec per pixel).
    """
    from astropy.io import fits
    imgs, centers, pas, nums, names, wvs, pxs = [], [], [], [], [], [], []
    for i, f in enumerate(filepaths):
        with fits.open(f) as h:
            ph, sh = h[0].header, h["SCI"].header
            data = np.asarray(h["SCI"].data, np.float32)
            if data.ndim == 2:
                data = data[None]
            nints = data.shape[0]
            pxscale = float(np.sqrt(sh["PIXAR_A2"])) if "PIXAR_A2" in sh else float(ph.get("PIXSCALE", np.nan))
            cx = sh.get(center_keywords[0], sh.get("CRPIX1")) - 1
            cy = sh.get(center_keywords[1], sh.get("CRPIX2")) - 1
            if "XOFFSET" in ph and np.isfinite(pxscale):
                cx += ph["XOFFSET"] / pxscale
                cy += ph["YOFFSET"] / pxscale
            pa = float(sh["ROLL_REF"]) - float(sh.get("V3I_YANG", 0.0)) * float(sh.get("VPARITY", 1))
            imgs.append(data)
            centers.append(np.tile([cx, cy], (nints, 1)))
            pas.append(np.full(nints, pa))
            nums.append(np.full(nints, i))
            names.append(np.array([os.path.basename(f)] * nints))
            wvs.append(np.full(nints, _wavelength_m(ph)))
            pxs.append(pxscale)
    return {"input": np.concatenate(imgs), "centers": np.concatenate(centers), "PAs": np.concatenate(pas),
            "filenums": np.concatenate(nums), "filenames": np.concatenate(names), "wvs": np.concatenate(wvs),
            "pxscale": float(np.nanmedian(pxs))}


def _wavelength_m(ph) -> float:
    """Return a filter's central wavelength from a primary header.

    Parameters
    ----------
    ph : FITS header
        The primary header.

    Returns
    -------
    wavelength : float
        The wavelength, in m, from ``CWAVEL`` (in microns) when present, else from pyKLIP's JWST
        filter tables (NIRCam ``PUPIL`` or ``FILTER``, NIRISS, MIRI).  NaN without either.
    """
    if "CWAVEL" in ph:
        return 1e-6 * float(ph["CWAVEL"])
    try:
        from pyklip.instruments.JWST import JWSTData
        inst = str(ph.get("INSTRUME", "")).upper()
        tabs = {"NIRCAM": "wave_nircam", "NIRISS": "wave_niriss", "MIRI": "wave_miri"}
        import ast, inspect, re                    # the offline tables live as literals in __init__
        src = inspect.getsource(JWSTData.__init__)
        ms = re.findall(r"self\." + tabs.get(inst, "x") + r" = (\{[^\n]*\})", src)
        lit = [m for m in ms if len(m) > 2]              # skip the empty '{}' initializers
        if lit:
            table = ast.literal_eval(lit[0])
            key = ph.get("PUPIL") if inst == "NIRCAM" and ph.get("PUPIL") in table else ph.get("FILTER")
            if key in table:
                return 1e-6 * float(table[key])
    except Exception:
        pass
    return float("nan")


class _ArrData:
    """Stand-in for a pyKLIP data object, what :func:`~klip_tpe.backends.pyklip.dataset_from_pyklip` reads.

    Parameters
    ----------
    d : dict
        The arrays (:func:`read_jwst_files`), ``input``, ``centers``, ``PAs``, ``filenums``,
        ``filenames`` and ``wvs``.
    psflib : _ArrLib or None, default None
        The reference library.
    """

    def __init__(self, d: Dict[str, np.ndarray], psflib=None):
        self.input, self.centers, self.PAs = d["input"], d["centers"], d["PAs"]
        self.filenums, self.filenames, self.wvs = d["filenums"], d["filenames"], d["wvs"]
        self.psflib = psflib


class _ArrLib:
    """Stand-in for pyKLIP's reference library, its frames shifted to one center.

    Parameters
    ----------
    master_library : ndarray, shape (n, ny, nx)
        The ``n`` reference frames.
    aligned_center : tuple of 2 floats
        The center they share, ``(x, y)`` in 0-based pixels.
    """
    def __init__(self, master_library: np.ndarray, aligned_center: Tuple[float, float]):
        self.master_library, self.aligned_center = master_library, aligned_center


def load_spaceklip(database=None, key: Optional[str] = None, sci_files: Optional[Sequence[str]] = None,
                   ref_files: Optional[Sequence[str]] = None, crop_half: Optional[int] = None,
                   partition_by: Optional[str] = "roll", name: Optional[str] = None,
                   use_pyklip_reader: bool = True, log: Callable[[str], None] = print) -> Dict[str, Dataset]:
    """Load JWST frames into datasets, one per science roll by default, with the RDI reference cube.

    The frames come from a spaceKLIP database's concatenation or from lists of files.  pyKLIP's
    ``JWSTData`` reads them where it imports, else :func:`read_jwst_files`, the reference frames shifted
    to one center.

    Parameters
    ----------
    database : spaceKLIP.Database or dict or None, default None
        The database, or its ``obs`` dict of tables.  None takes ``sci_files`` and ``ref_files``.
    key : str or None, default None
        The concatenation.  None takes the database's one concatenation.
    sci_files : sequence of str or None, default None
        The science files, without a database.
    ref_files : sequence of str or None, default None
        The reference files, without a database.
    crop_half : int or None, default None
        Half-width of the crop about the star, in pixels.  None keeps the frames whole.
    partition_by : {"roll", "filenums"} or None, default "roll"
        How the integrations split into datasets, by position angle, by exposure, or all in one.  Each
        exposure of a JWST roll is a ``filenum`` of its own, so "filenums" gives one dataset per
        exposure, often of a single frame.
    name : str or None, default None
        Prefix of the datasets' names.  None takes the concatenation, else ``"jwst"``.
    use_pyklip_reader : bool, default True
        Whether to read with pyKLIP's ``JWSTData`` where it imports.
    log : callable, default print
        Function that takes each line of the log.

    Returns
    -------
    datasets : dict of str to Dataset
        The datasets by name, each ``meta`` with ``source``, ``pxscale`` (arcsec per pixel),
        ``wavelength_m`` and ``filter`` where the headers give them.

    Raises
    ------
    ValueError
        If a database holds several concatenations and no ``key`` is given, or there are no science
        files.
    """
    if database is not None:
        if key is None:
            keys = list((database.obs if hasattr(database, "obs") else database).keys())
            if len(keys) != 1:
                raise ValueError(f"pass key= (available: {keys})")
            key = keys[0]
        sci_files, ref_files, info = _files_from_database(database, key)
        if name is None:
            name = str(key)
    else:
        info = {}
    if not sci_files:
        raise ValueError("no science files")
    name = name or "jwst"
    data = None
    if use_pyklip_reader:
        try:
            from pyklip.instruments.JWST import JWSTData
            data = JWSTData(list(sci_files), list(ref_files) if ref_files else None)
        except Exception as exc:
            log(f"  pyklip JWSTData unavailable ({exc!r}); using the built-in reader")
    if data is None:
        d = read_jwst_files(sci_files)
        psflib = None
        if ref_files:
            r = read_jwst_files(ref_files)
            # each reference frame is first shifted to one common center; dataset_from_pyklip
            # then treats the library like pyKLIP's (single aligned_center)
            target = tuple(np.median(r["centers"], axis=0))
            psflib = _ArrLib(_recentre_stack(r["input"], r["centers"], target), target)
        data = _ArrData(d, psflib)
        pxscale = d["pxscale"]
        wave = float(np.nanmedian(d["wvs"])) if np.isfinite(d["wvs"]).any() else None
    else:
        pxscale = float(getattr(data, "pxscale", np.nan)) if hasattr(data, "pxscale") else float(info.get("PIXSCALE", np.nan))
        if not np.isfinite(pxscale):                  # pyKLIP's JWSTData does not carry it -> read a header
            try:
                from astropy.io import fits
                with fits.open(sci_files[0]) as h:
                    sh = h["SCI"].header
                    pxscale = float(np.sqrt(sh["PIXAR_A2"])) if "PIXAR_A2" in sh else float(h[0].header.get("PIXSCALE", np.nan))
            except Exception:
                pxscale = float("nan")
        wave = float(np.nanmedian(np.asarray(data.wvs, float))) if len(data.wvs) else None
    out = dataset_from_pyklip(data, crop_half=crop_half, partition_by=partition_by, name=name)
    for ds in out.values():
        ds.meta.update(source="spaceklip", pxscale=pxscale, wavelength_m=wave, filter=info.get("FILTER"))
        log(f"  {ds.name}: {ds.nframes} frames, PA {ds.angles.min():.1f}..{ds.angles.max():.1f} deg, "
            f"refs {0 if ds.ref_cube is None else ds.ref_cube.shape[0]}")
    return out


def load_calints(files: Sequence[str], science_target: Optional[str] = None, half_px: int = 55,
                 align: bool = True, repair: Union[bool, str] = True,
                 star_center: Optional[Tuple[float, float]] = None, keep_frames: bool = False,
                 partition: str = "roll", filter: Optional[str] = None,
                 background: Optional[bool] = None, destripe: Optional[bool] = None,
                 ref_targets: Optional[Union[str, Sequence[str]]] = None,
                 hot_pixels: Optional[bool] = None,
                 log: Callable[[str], None] = print) -> Tuple[Dict[str, Dataset], Dict[str, Any]]:
    """Load stage-2 ``*_calints.fits`` files into datasets, without spaceKLIP.

    The files split into science (``TARGPROP`` starting with ``science_target``), dedicated background
    pointings (``BKG`` or ``BACKGROUND``) and the PSF references, everything else or the stars of
    ``ref_targets``.  A file named as a PSF reference stays a reference even when its name starts with
    the target's.  The frames then go through these steps:

    * The background pointings' blank-sky median (:func:`blank_sky`) is subtracted from the frames the
      pipeline did not subtract it from, told apart by ``S_BKDSUB = 'COMPLETE'`` (``background``).
    * Static hot and dead pixels the DQ array misses, found on that median (:func:`static_hot_pixels`),
      are flagged in every frame (``hot_pixels``).
    * Each frame's row and column offsets are removed on the full subarray (:func:`destripe_detector`,
      ``destripe``).
    * DQ ``DO_NOT_USE`` pixels are filled from their neighbors (:func:`fill_dq_neighbours`, ``repair``),
      and nothing else is touched.
    * The frames are registered to the median science frame by cross-correlation (``align``).  A frame
      whose offset cannot be measured, with its peak outside a +/-6 px box or not a maximum, stays
      where it is.
    * The frames are cropped about the star to ``2 * half_px + 1`` pixels, an odd width, so the star
      lies at ``((nx - 1) / 2, (ny - 1) / 2)`` (:func:`klip_tpe.metrics.star_center`).

    The position angle is ``ROLL_REF - V3I_YANG * VPARITY``.

    Parameters
    ----------
    files : sequence of str
        Paths of the files, science, references and background pointings together.
    science_target : str or None, default None
        Prefix of the science target's ``TARGPROP``, dashes and underscores ignored.  None takes the
        first file's.
    half_px : int, default 55
        Half-width of the crop, in pixels.
    align : bool, default True
        Whether to register the frames by cross-correlation.
    repair : bool or {"dq", "sigma", "none"}, default True
        How flagged pixels are repaired.  True or "dq" fills them from their neighbors, False or "none"
        leaves them NaN, which pyKLIP's RDI library cannot take, and "sigma" is the median filter before
        version 1.1 (:func:`sigma_clip_repair`), kept to reproduce old runs.
    star_center : tuple of 2 floats or None, default None
        The star, ``(x, y)`` in 0-based detector pixels, the point the crop is centered on.  None takes
        ``CRPIX``, the aperture's reference point, which is no measured star position.
    keep_frames : bool, default False
        Whether ``info`` keeps the image stacks too, about 18 MB for ERS 1386.
    partition : {"roll", "all"}, default "roll"
        One dataset per roll angle, or one holding every science frame.
    filter : str or None, default None
        The filter to load.  None asks for the files to hold one filter.
    background : bool or None, default None
        Whether to subtract the blank sky.  None subtracts it where needed, and refuses a mixture of
        frames with and without it when there is no background pointing to fix it with.  False stacks
        them as they are.
    destripe : bool or None, default None
        Whether to destripe.  None destripes MIRI frames.
    ref_targets : str or sequence of str or None, default None
        Prefixes of the ``TARGPROP`` of the stars the RDI library may hold.  None takes every file that
        is neither science nor background.
    hot_pixels : bool or None, default None
        Whether to flag the static hot pixels.  None flags them where there are background pointings to
        find them in, and True insists.
    log : callable, default print
        Function that takes each line of the log.

    Returns
    -------
    datasets : dict of str to Dataset
        ``roll1``, ``roll2`` ... (or ``sci`` with ``partition="all"``), each with the RDI library as its
        reference cube and ``meta`` with ``pxscale``, ``wavelength_m``, ``header``, ``pixar_sr``,
        ``bunit`` and ``provenance``.
    info : dict
        The ``meta`` keys with ``n_sci``, ``n_ref``, ``crop_px``, ``frames`` (one record per integration),
        ``repair``, ``partition``, ``filter``, ``star_center``, ``crpix``, ``rolls``, ``hot_pixels`` (as
        ``(x, y)`` detector pixels) and the fractions of pixels repaired and left non-finite.  With
        ``keep_frames`` also ``raw_sci``, ``raw_ref``, ``aligned_sci``, ``aligned_ref``, ``crop_sci`` and
        ``crop_ref``.

    Raises
    ------
    ValueError
        If there are no files, an exposure occurs twice, the files span several filters and none is
        asked for, the filter asked for is not among them, ``ref_targets`` matches no reference, there are
        no science files, ``repair`` or ``partition`` is not one of their values, or the frames mix
        background-subtracted and untreated ones with no background pointing among the files.  Also if
        ``background`` or ``hot_pixels`` is True without a background pointing.

    Notes
    -----
    ``partition`` decides what pyKLIP's ``mode`` can mean.  A partition is reduced on its own, so with
    one per roll every frame in it has the same position angle.  ADI then has no reference frames
    (pyKLIP returns NaN), and ADI+RDI is RDI, the other roll never in the basis.  Only with "all" does
    ADI subtract the other roll, and only then is a searched ``mode`` a real choice (on ERS 1386 F444W at
    k = 10, an injected S/N of 6.2 with ADI, 6.9 with RDI and 7.2 with ADI+RDI).  The price of "all" is
    one ``k_klip`` block for both rolls.

    An archive download of one program routinely mixes background-subtracted and untreated frames,
    since a program's science targets carry ``S_BKDSUB = 'COMPLETE'`` and its PSF reference stars usually
    do not.  On ERS 1386 F1140C, 14 of 16 reference frames arrived with a sky pedestal of about 19 MJy/sr
    and the 4QPM glow sticks, beside science frames without them.  The library's dominant common mode was
    then the background, and the reducer's high-pass, which hides a smooth pedestal, gave the optimizer a
    reason to prefer a hard high-pass.

    ``CRPIX`` is the same in every file of a program, dithers included, and on ERS 1386 it misses HIP
    65426 by about 0.8 px, which moves the companion inside its own separation and mismatches its KLIP
    throughput against the fakes.  spaceKLIP's star-centering step (``STARCENX``/``STARCENY``) is the
    right source, and ``scripts/check_hip65426_contrast.py`` solves for it from the companion in each
    roll.  An even crop would leave the star half a pixel off in each axis, 0.7 px radially, a 3 degree
    error in position angle at HIP 65426 b's 13 px.

    Without ``ref_targets``, the MIRI directory of HIP 65426, which also holds HD 141569A and its
    reference HD 140986, gave F1140C a library of 90 frames of phi Cen, 20 of HD 140986 and 12 of HD
    141569A, a resolved disk in the KL basis of a bare star.  The library's stars are now logged by name
    and count.
    """
    from astropy.io import fits

    files = sorted(files)
    if not files:
        raise ValueError("load_calints got no files")

    # One exposure, once.  A JWST product is named after its exposure, so a name that occurs
    # twice is the same exposure twice -- typically two processings of it side by side, the
    # archive calints and a re-reduction from the raw ramps in a subfolder, which a recursive
    # search finds together.  Stacked, every frame would enter twice, from two pipelines.
    _seen: Dict[str, str] = {}
    _dups: List[Tuple[str, str]] = []
    for f in files:
        b = os.path.basename(str(f))
        if b in _seen:
            _dups.append((_seen[b], str(f)))
        else:
            _seen[b] = str(f)
    if _dups:
        ex = "; ".join(f"{a} and {b}" for a, b in _dups[:3])
        raise ValueError(
            f"{len(_dups)} exposure(s) occur more than once among the {len(files)} files -- the "
            f"same product name in two places, e.g. {ex}. That is usually two processings of one "
            f"data set in one directory tree (MAST's mastDownload/ beside a re-reduction from "
            f"uncal). Point the loader at one of them.")

    # One filter, always.  The wavelength and the pixel scale were taken from the FIRST
    # file and applied to every frame, so a directory holding more than one filter was
    # stacked into a single dataset at a single lambda -- different masks, different
    # lambda/D, no error.  A whole program downloaded at once is exactly that: the ERS
    # (GO 1386) has HIP 65426 in F1140C and F1550C and HD 141569A in all three, together.
    def _filt(f):
        return str(fits.getheader(f).get("FILTER", "")).strip().upper()

    present = sorted({_filt(f) for f in files} - {""})
    if filter:
        want_f = str(filter).strip().upper()
        if want_f not in present:
            raise ValueError(f"no files with FILTER={want_f!r}; this set has {present}")
        files = [f for f in files if _filt(f) == want_f]
        log(f"  calints: {len(files)} file(s) in {want_f} (of {present})")
    elif len(present) > 1:
        raise ValueError(
            f"these {len(files)} files span {len(present)} filters {present}, and a "
            f"dataset has one wavelength and one mask. Pass filter= to choose; loading "
            f"them together would stack every filter at the first file's lambda.")

    def targ(f):
        return str(fits.getheader(f).get("TARGPROP", "")).replace("-", "").replace("_", "").upper()

    def is_background(f):
        """Tell whether a file is a dedicated background pointing, by its ``TARGPROP``.

        These offset exposures of blank sky are essential for MIRI, whose thermal background is
        strong and structured, but they are no PSF references.  The split below takes the
        science and puts everything else in the library, so without this they would end up in
        the RDI basis, empty sky modeling the star's diffraction.  They give nothing to subtract
        with and dilute the frames that do, and on MIRI program 11225 they outnumber the real
        references.
        """
        t = str(fits.getheader(f).get("TARGPROP", "")).upper()
        return "BKG" in t or "BACKGROUND" in t

    def is_reference(f):
        """Tell whether the program named a file's pointing as a PSF reference.

        The science match is ``startswith``, so that "HIP65426" finds "HIP-65426".  A program
        that names its reference after its target defeats it.  AU Mic's is
        ``AU_Mic_psf_reference``, which normalizes to AUMICPSFREFERENCE and starts with AUMIC,
        so the reference was taken for science, its frames derotated and stacked with the
        target's, and the RDI library came out empty.  Nothing said so, since more science
        frames than the program has is not an obvious enough number to catch.  An explicit name
        wins over a prefix match.
        """
        t = str(fits.getheader(f).get("TARGPROP", "")).upper()
        return ("PSF_REFERENCE" in t or "PSFREF" in t or "PSF-REF" in t
                or "REFERENCE" in t or t.endswith("-REF") or t.endswith("_REF"))

    want = (science_target or targ(files[0])).replace("-", "").replace("_", "").upper()
    bkg = [f for f in files if is_background(f)]
    rest = [f for f in files if f not in bkg]
    # ...unless the caller asked for the reference itself as the science target, which is
    # a fair thing to want (reducing the reference star to check it).
    want_is_ref = any(k in want for k in ("PSFREFERENCE", "PSFREF", "REFERENCE"))
    named_ref = [] if want_is_ref else [f for f in rest if is_reference(f)]
    sci = [f for f in rest if f not in named_ref and targ(f).startswith(want)]
    ref = [f for f in rest if f not in sci]
    # "Science, and everything else is the library" is right for a download of one
    # program and wrong for a directory holding several.  HIP 65426's MIRI directory also
    # holds HD 141569A and its reference HD 140986, so the F1140C library came out at 122
    # frames of which only 90 were phi Cen -- and 12 were HD 141569A, a resolved DISK,
    # putting disk structure into the KL basis of a bare star.  Nothing in the numbers said
    # so; the loader only ever reported "16 reference files".
    if ref_targets:
        keys = [str(t).replace("-", "").replace("_", "").upper() for t in
                ([ref_targets] if isinstance(ref_targets, str) else list(ref_targets))]
        kept = [f for f in ref if any(targ(f).startswith(k) for k in keys)]
        dropped = [f for f in ref if f not in kept]
        if not kept:
            raise ValueError(
                f"ref_targets={list(keys)} matched none of the {len(ref)} reference file(s); "
                f"their TARGPROPs are {sorted({targ(f) for f in ref})}")
        if dropped:
            log(f"  calints: reference library restricted to {', '.join(keys)} -- "
                f"{len(dropped)} file(s) dropped ("
                + ", ".join(f"{t} x{sum(1 for f in dropped if targ(f) == t)}"
                            for t in sorted({targ(f) for f in dropped})) + ")")
        ref = kept
    if ref:
        # Always say WHO is in the library, by name and count.  This is the line whose
        # absence hid HD 141569A in there.
        log("  calints: RDI library from " + ", ".join(
            f"{t} ({sum(1 for f in ref if targ(f) == t)} file(s))"
            for t in sorted({targ(f) for f in ref})))
    stolen = [f for f in named_ref if targ(f).startswith(want)]
    if stolen:
        log(f"  calints: {len(stolen)} file(s) whose TARGPROP starts with {want!r} are named "
            f"as PSF references and were kept in the library, not the science set "
            f"(e.g. {os.path.basename(stolen[0])})")
    if bkg:
        log(f"  calints: {len(bkg)} background pointing(s) held out of the RDI library "
            f"(blank sky is not a PSF reference); they are used for subtraction below")
    if not sci:
        raise ValueError(f"no science files matching TARGPROP {want!r} among {len(files)} files")
    if not ref:
        log(f"  calints: no PSF-reference files among the {len(files)} given -- the "
            f"datasets will carry no RDI library and the reduction will be ADI only")

    versions: Dict[str, set] = {"CAL_VER": set(), "CRDS_CTX": set()}

    def _note_versions(ph):
        for k in versions:
            v = ph.get(k) if hasattr(ph, "get") else None
            if v not in (None, ""):
                versions[k].add(str(v).strip())

    def read(fs, role):
        ims, pas, prov = [], [], []
        for f in fs:
            with fits.open(f) as h:
                d = np.asarray(h["SCI"].data, float)
                dq = np.asarray(h["DQ"].data, int)
                s = h["SCI"].header
                ph = h[0].header
                _note_versions(ph)
                pa = float(s["ROLL_REF"]) - float(s.get("V3I_YANG", 0)) * float(s.get("VPARITY", 1))
                bad = (dq & 1).astype(bool)
                d = np.where(bad, np.nan, d)
                # One plane of a calints cube is ONE integration -- a single up-the-ramp
                # charge accumulation, already fitted to a slope.  EFFINTTM is its
                # exposure time, the JWST analogue of a DIT; the NFRAMES frames that
                # build each group were averaged on the detector and no longer exist
                # separately.  int_mid_mjd comes from INT_TIMES when the extension is
                # there, so each frame carries its own clock rather than the exposure's.
                mids = None
                if "INT_TIMES" in h and h["INT_TIMES"].data is not None:
                    t = h["INT_TIMES"].data
                    if "int_mid_MJD_UTC" in t.columns.names:
                        mids = np.asarray(t["int_mid_MJD_UTC"], float)
                for i in range(d.shape[0]):
                    ims.append(d[i])
                    pas.append(pa)
                    prov.append({"file": os.path.basename(f), "integration": i, "role": role,
                                 "target": str(ph.get("TARGPROP", "")), "pa": pa,
                                 "bkgsub": _background_done(ph, s),
                                 "n_dq": int(bad[i].sum()),
                                 "effinttm": float(ph.get("EFFINTTM", np.nan)),
                                 "readpatt": str(ph.get("READPATT", "")),
                                 "ngroups": int(ph.get("NGROUPS", 0)),
                                 "nframes": int(ph.get("NFRAMES", 0)),
                                 "groupgap": int(ph.get("GROUPGAP", 0)),
                                 "tframe": float(ph.get("TFRAME", np.nan)),
                                 "int_mid_mjd": (float(mids[i]) if mids is not None
                                                 and i < mids.size else float("nan")),
                                 # the small-grid dither offsets live in the PRIMARY
                                 # header, not SCI -- read_jwst_files gets this right too
                                 "xoffset": float(ph.get("XOFFSET", 0.0)),
                                 "yoffset": float(ph.get("YOFFSET", 0.0))})
        return np.array(ims), np.array(pas), prov

    rmode = {True: "dq", False: "none", None: "none"}.get(repair, repair)
    rmode = str(rmode).lower()
    if rmode not in ("dq", "sigma", "none"):
        raise ValueError(f"repair must be True/'dq', False or 'sigma', got {repair!r}")

    unflagged_rewritten: List[int] = []

    def _repair(cube, prov=None):
        out = np.array(cube, float)
        for i, im in enumerate(out):
            if rmode == "sigma":
                out[i], n_hit, n_bad = sigma_clip_repair(im)
                unflagged_rewritten.append(n_hit - n_bad)
            else:
                out[i], n_hit = fill_dq_neighbours(im)
            if prov is not None:
                prov[i]["n_repaired"] = int(n_hit)
                prov[i]["repair"] = rmode
        return out

    def _xs(im, ref_, search=6):
        """Return the sub-pixel offset of ``im`` from ``ref_`` as ``(dx, dy, ok)``.

        ``ok`` False means no offset was measured, and the caller must not shift.  It used to
        return a confident wrong number in two ways:

        * ``np.fft.rfft2`` of an array holding one NaN is all NaN, and ``np.argmax`` of an
          all-NaN array returns 0, so every frame of a MIRI cube came back as
          ``(-search, -search)``, a six-pixel shift measured from nothing.  The median is now
          removed and the gaps zero-filled, so they add no correlation.
        * Where the correlation has no peak inside the box, ``argmax`` still returns its highest
          corner.  On HIP 65426 F1140C the peak sits more than 100 px away and zero lag is a
          minimum, since the 41 integrations of one exposure have nothing to register against
          each other, and the answer was six pixels in a corner.  A real peak is interior and
          positive, so anything else is now reported as unmeasured.
        """
        a = np.nan_to_num(np.asarray(im, float) - np.nanmedian(im))
        b = np.nan_to_num(np.asarray(ref_, float) - np.nanmedian(ref_))
        cc = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape))
        c = np.array(im.shape) // 2
        sub = cc[c[0] - search:c[0] + search + 1, c[1] - search:c[1] + search + 1]
        k = np.unravel_index(np.argmax(sub), sub.shape)
        edge = k[0] in (0, sub.shape[0] - 1) or k[1] in (0, sub.shape[1] - 1)
        if edge or not np.isfinite(sub[k]) or sub[k] <= 0:
            return 0.0, 0.0, False
        dy, dx = k[0] - search, k[1] - search
        p = lambda a, b, c_: 0.0 if (a - 2 * b + c_) == 0 else 0.5 * (a - c_) / (a - 2 * b + c_)
        dy += p(sub[k[0] - 1, k[1]], sub[k[0], k[1]], sub[k[0] + 1, k[1]])
        dx += p(sub[k[0], k[1] - 1], sub[k[0], k[1]], sub[k[0], k[1] + 1])
        return dx, dy, True

    S, pas, prov_s = read(sci, "SCI")
    if ref:
        R, _, prov_r = read(ref, "REF")
    else:
        R, prov_r = np.zeros((0,) + S.shape[1:]), []
    raw_s = np.array(S, np.float32) if keep_frames else None
    raw_r = np.array(R, np.float32) if keep_frames else None
    # Where the gaps were BEFORE any repair, so the log can say how many of them are in the
    # region actually returned.  A MIRI coronagraphic subarray is a quarter unilluminated and
    # flagged; counting that against the 81x81 stamp around the star reports 27% where the
    # stamp's own figure is under one per cent, which is alarm about the wrong pixels.
    # ---- the dedicated background, and the mixture an archive download hides -------------
    # A program's SCIENCE targets get dedicated background pointings and Image2 subtracts
    # them (S_BKDSUB='COMPLETE'); its pure PSF REFERENCE stars usually do not get one, so the
    # step never runs on them.  ERS 1386 at F1140C: both HIP-65426 rolls and both HD-141569A
    # exposures are subtracted, while HIP-68245 (9 files) and HD-140986 (5) are not -- so 14
    # of 16 reference frames arrive carrying a ~19 MJy/sr sky pedestal and the 4QPM glow
    # sticks (+4.6 MJy/sr along the horizontal mask boundary, measured on a blank-sky frame
    # with no star in it), and the loader used to stack all of that into ONE RDI library
    # beside science frames that had none of it.  The library's dominant common mode is then
    # the background rather than the stellar PSF, and because the reducer's high-pass hides a
    # smooth pedestal, the optimizer is handed a reason to prefer a hard high-pass and to
    # report that as the best reduction parameter.
    done = [p["bkgsub"] for p in prov_s + prov_r]
    need_s = np.array([not p["bkgsub"] for p in prov_s], bool)
    need_r = np.array([not p["bkgsub"] for p in prov_r], bool) if prov_r else np.zeros(0, bool)
    mixed, n_need = len(set(done)) > 1, int(sum(1 for d in done if not d))
    if background is not False and n_need:
        if bkg:
            BG = blank_sky(bkg)
            if need_s.any():
                S[need_s] -= BG
            if R.size and need_r.any():
                R[need_r] -= BG
            log(f"  calints: subtracted the blank-sky median of {len(bkg)} background "
                f"pointing(s) from {n_need} of {len(done)} frames -- the ones whose header "
                f"carries no S_BKDSUB=COMPLETE"
                + (f"; the other {len(done) - n_need} the pipeline had already done, and "
                   f"stacking the two untreated is the thing this avoids" if mixed else ""))
        elif mixed:
            raise ValueError(
                f"{n_need} of {len(done)} science+reference frames have not had the "
                f"background subtracted (no S_BKDSUB=COMPLETE) and the other "
                f"{len(done) - n_need} have, and there is no background pointing among the "
                f"files to fix it with. Stacking them into one KLIP library mixes two sky "
                f"levels -- on MIRI that is a ~19 MJy/sr pedestal plus the 4QPM glow sticks "
                f"in some frames and not others. Fetch the programme's *-BACKGROUND "
                f"exposures (scripts/fetch_jwst_ar.py gets them), or pass background=False "
                f"to stack them anyway and say why.")
        else:
            log("  calints: no frame carries S_BKDSUB=COMPLETE and no background pointing "
                "was given -- the set is at least self-consistent, but the sky is still in "
                "it, in the KLIP basis and in the noise the objective is measured against")
    if background is True and not bkg:
        raise ValueError("background=True but no dedicated background pointing is among the "
                         "files given")

    # ---- static hot pixels the DQ array misses, from the blank sky -----------------------
    hot_xy: List[Tuple[int, int]] = []
    if hot_pixels is True and not bkg:
        raise ValueError("hot_pixels=True but no dedicated background pointing is among the "
                         "files given to find them in")
    if hot_pixels is not False and bkg:
        hot = static_hot_pixels(blank_sky(bkg))
        ys_, xs_ = np.nonzero(hot)
        hot_xy = [(int(x), int(y)) for x, y in zip(xs_, ys_)]
        if hot_xy:
            S[:, hot] = np.nan
            if R.size:
                R[:, hot] = np.nan
        log(f"  calints: {len(hot_xy)} static hot/dead pixel(s) the DQ array does not flag, "
            f"found on the blank-sky median of {len(bkg)} background pointing(s), flagged in "
            f"every frame and repaired like DQ pixels"
            + (": " + ", ".join(f"({x},{y})" for x, y in hot_xy) if 0 < len(hot_xy) <= 40 else ""))

    # ---- detector-frame destriping, on the FULL subarray and before the crop ----------
    # After the background subtraction (so the sky pedestal is not folded into the row
    # offsets) and before the crop (where a row would be mostly PSF).  On by default for
    # MIRI, whose per-row offset is as large as its pixel-to-pixel noise.
    _ph = fits.getheader(sci[0]) if sci else {}
    _inst = str(_ph.get("INSTRUME", "")).strip().upper()
    if destripe is None:
        destripe = (_inst == "MIRI")
    if destripe:
        _sh = fits.getheader(sci[0], "SCI")
        _ctr = (tuple(float(v) for v in star_center) if star_center is not None
                else (float(_sh["CRPIX1"]) - 1.0, float(_sh["CRPIX2"]) - 1.0))
        _n0 = float(np.nanmedian(np.abs(S - np.nanmedian(S)))) * 1.4826
        S, st_s = destripe_detector(S, _ctr)
        if R.size:
            R, _ = destripe_detector(R, _ctr)
        _n1 = float(np.nanmedian(np.abs(S - np.nanmedian(S)))) * 1.4826
        log(f"  calints: destriped in the detector frame on the full {S.shape[2]}x{S.shape[1]} "
            f"subarray -- per-row offset sigma {st_s['row_sigma']:.3g}, per-column "
            f"{st_s['col_sigma']:.3g}; science frame scatter {_n0:.3g} -> {_n1:.3g}"
            + (f" ({_n0 / _n1:.2f}x)" if _n1 > 0 else "")
            + f".  Star masked to r=45 px and the 4QPM boundaries to 12 px, so the PSF and "
              f"the glow sticks do not set the offsets.")
    elif _inst == "MIRI":
        log("  calints: destriping OFF for a MIRI set -- its per-row offset is typically as "
            "large as the pixel noise, and it survives KLIP as a fixed detector pattern")

    gaps0_s = ~np.isfinite(S)
    gaps0_r = ~np.isfinite(R) if R.size else None
    if rmode != "none":
        S = _repair(S, prov_s)
        R = _repair(R, prov_r) if R.size else R
    if unflagged_rewritten:
        log(f"  calints: repair='sigma' rewrote a median of {int(np.median(unflagged_rewritten))} UNFLAGGED "
            f"pixels per frame on top of the DQ ones -- that is the PSF being median-filtered, star and "
            f"companion alike (HIP 65426 b peaks at 7 instead of 19 MJy/sr).  Use repair='dq'.")
    n_unmeasured = [0]
    if align:
        with np.errstate(all="ignore"):
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                a0 = np.nanmedian(S, axis=0)   # np.median over a cube with gaps is all-NaN

        def _shift_all(cube, prov):
            out = []
            for i, im in enumerate(cube):
                dx, dy, ok = _xs(im, a0)
                if prov is not None:
                    prov[i]["dx"], prov[i]["dy"], prov[i]["registered"] = \
                        float(dx), float(dy), bool(ok)
                if not ok:
                    n_unmeasured[0] += 1
                    out.append(np.asarray(im, float))      # unshifted, not shifted by junk
                else:
                    out.append(shift_keeping_gaps(im, (-dy, -dx), order=3))
            return np.array(out)

        S = _shift_all(S, prov_s)
        if R.size:
            R = _shift_all(R, prov_r)
        nf = int(S.shape[0] + (R.shape[0] if R.size else 0))
        if n_unmeasured[0]:
            log(f"  calints: no cross-correlation offset could be measured for "
                f"{n_unmeasured[0]} of {nf} frame(s) -- the peak was outside the +/-6 px "
                f"search box or not a maximum -- and those frames were left UNSHIFTED rather "
                f"than moved by the highest corner of the box. Integrations of a single "
                f"exposure at a fixed pointing have no offset to measure, so this is expected "
                f"for them; if you expected a dither, pass align=False and centre on "
                f"star_center= instead of trusting this.")

    hdr = fits.getheader(sci[0], "SCI")
    px = float(np.sqrt(hdr["PIXAR_A2"]))
    if star_center is not None:
        cx, cy = float(star_center[0]), float(star_center[1])
    else:
        cx, cy = hdr["CRPIX1"] - 1, hdr["CRPIX2"] - 1
        log(f"  calints: centring on CRPIX ({cx:.2f}, {cy:.2f}) -- the APERTURE reference "
            f"point, not a measured star position; pass star_center= if you have one")
    H = int(half_px)
    n = 2 * H + 1                                        # ODD: see the docstring

    def crop(cube):
        if not cube.size:
            return cube
        fx, fy = cx - round(cx), cy - round(cy)
        x0, y0 = int(round(cx)) - H, int(round(cy)) - H
        out = np.array([shift_keeping_gaps(im, (-fy, -fx), order=3)[y0:y0 + n, x0:x0 + n]
                        for im in cube], np.float32)
        # Close what the interpolation invalidated.  Two sub-pixel splines have widened each
        # gap by their own support, and the cube is the reducer's input: its high-pass runs
        # at nan_aware=False over a running sum, so a NaN anywhere costs everything after it.
        # The gaps that survive here are inside the crop and isolated, which is what
        # fill_dq_neighbours exists for -- unlike the unilluminated border it could not close.
        if rmode != "none":
            out = np.array([fill_dq_neighbours(im)[0] for im in out], np.float32)
        return out

    Sx, Rx = crop(S), crop(R)
    for f in bkg:
        _note_versions(fits.getheader(f))
    # What these frames are, for the run's record (run_setup.json, through the reducer's
    # describe()): which products, made by which pipeline, and what this loader did to them.
    # The calints are an input a reduction depends on, so a run has to be able to say which.
    provenance = {
        "loader": "load_calints",
        "files": {"sci": [os.path.basename(f) for f in sci], "ref": [os.path.basename(f) for f in ref],
                  "bkg": [os.path.basename(f) for f in bkg]},
        "dirs": sorted({os.path.dirname(os.path.abspath(str(f))) for f in list(sci) + list(ref) + list(bkg)}),
        "cal_ver": sorted(versions["CAL_VER"]), "crds_ctx": sorted(versions["CRDS_CTX"]),
        "pxscale": px,
        "hot_pixels": {"requested": hot_pixels, "applied": bool(hot_pixels is not False and bkg),
                       "n": len(hot_xy), "pixels": [list(p) for p in hot_xy]},
        "background_subtracted": (int(n_need) if (background is not False and n_need and bkg) else 0),
        "destripe": bool(destripe), "repair": rmode, "align": bool(align),
        "n_unshifted": int(n_unmeasured[0]) if align else None,
    }
    meta = {"pxscale": px, "wavelength_m": _wavelength_m(fits.getheader(sci[0])),
            "header": dict(fits.getheader(sci[0])), "pixar_sr": float(hdr.get("PIXAR_SR", np.nan)),
            "bunit": str(hdr.get("BUNIT", "")).strip(), "provenance": provenance}
    if str(partition).lower() not in ("roll", "all"):
        raise ValueError(f"partition must be 'roll' or 'all', got {partition!r}")
    dsets: Dict[str, Dataset] = {}
    if str(partition).lower() == "all":
        dsets["sci"] = Dataset(Sx, pas, name="sci", ref_cube=Rx if Rx.size else None, meta=dict(meta))
    else:
        for k, pa in enumerate(np.unique(np.round(pas, 1))):
            m = np.round(pas, 1) == pa
            dsets[f"roll{k + 1}"] = Dataset(Sx[m], pas[m], name=f"roll{k + 1}",
                                            ref_cube=Rx if Rx.size else None, meta=dict(meta))
    info = dict(meta, n_sci=int(S.shape[0]), n_ref=int(R.shape[0]), crop_px=n,
                frames=prov_s + prov_r, repair=rmode, partition=str(partition).lower(),
                filter=str(fits.getheader(sci[0]).get("FILTER", "")),
                star_center=(float(cx), float(cy)),
                crpix=(float(hdr["CRPIX1"] - 1), float(hdr["CRPIX2"] - 1)),
                rolls=[float(v) for v in np.unique(np.round(pas, 1))],
                hot_pixels=hot_xy)
    if keep_frames:
        info.update(raw_sci=raw_s, raw_ref=raw_r,
                    aligned_sci=np.asarray(S, np.float32), aligned_ref=np.asarray(R, np.float32),
                    crop_sci=Sx, crop_ref=Rx)
    nrep = [p.get("n_repaired", 0) for p in prov_s + prov_r]
    npix = int(np.prod(S.shape[-2:])) if S.size else 0
    frac = (float(np.median(nrep)) / npix) if (nrep and npix) else 0.0
    # The same census over the region that is actually returned.  This is the number that
    # bears on the reduction: everything outside the stamp is thrown away a line later.
    x0, y0 = int(round(cx)) - H, int(round(cy)) - H
    gwin = [g[:, y0:y0 + n, x0:x0 + n] for g in (gaps0_s, gaps0_r) if g is not None and g.size]
    ncrop = np.concatenate([g.sum(axis=(1, 2)) for g in gwin]) if gwin else np.zeros(0)
    fcrop = float(np.median(ncrop)) / float(n * n) if ncrop.size else 0.0
    left = float(np.mean(~np.isfinite(np.concatenate([c for c in (Sx, Rx) if c.size]))) ) \
        if (Sx.size or Rx.size) else 0.0
    log(f"  calints: {len(sci)} science / {len(ref)} reference files -> {S.shape[0]} + "
        f"{R.shape[0]} frames, {n}x{n} px at {px*1e3:.2f} mas, rolls {info['rolls']}, "
        f"{info['bunit']!r}; repair={rmode!r}"
        + (f" ({int(np.median(ncrop))} px/frame flagged inside the {n}x{n} stamp, "
           f"{100 * fcrop:.1f}%; {int(np.median(nrep))} px/frame, {100 * frac:.0f}% over the "
           f"whole subarray)" if nrep and rmode != "none" else ""))
    # Two different numbers, and only the first bears on the reduction.  A quarter of a MIRI
    # 4QPM subarray comes back DQ-flagged -- HIP 65426 in MASK1140 is 17,790 of 64,512 pixels
    # per frame -- but that is the unilluminated border outside the coronagraph field, and 44
    # of the 6,561 pixels of the returned stamp were flagged.  Warning on the subarray figure
    # points at pixels nothing downstream ever sees.
    if fcrop > 0.05 and rmode != "none":
        log(f"  calints: that is {100 * fcrop:.0f}% of every returned frame replaced by an "
            f"interpolation of its neighbours. Those pixels carry no independent "
            f"information but do enter the KLIP basis; consider repair=False (the built-in "
            f"reducers treat NaN as missing) and check the result against this one.")
    if left > 0:
        why = ("you asked for repair='none'" if rmode == "none" else
               "gaps too wide for fill_dq_neighbours to close -- the stamp may be running "
               "off the illuminated area, so try a smaller half_px or pass star_center=")
        log(f"  calints: {100 * left:.1f}% of the returned cube is non-finite ({why}). The "
            f"reducers' high-pass runs over a running sum, so one gap costs every pixel "
            f"after it along both axes and whole frames are then dropped as empty.")
    info["repaired_fraction"] = frac
    info["repaired_fraction_crop"] = fcrop
    info["nonfinite_fraction"] = left
    return dsets, info


def _recentre_stack(imgs: np.ndarray, centers: np.ndarray, target: Tuple[float, float]) -> np.ndarray:
    """Shift each frame of a stack so its star lies at one center, its gaps zero-filled.

    Parameters
    ----------
    imgs : ndarray, shape (n, ny, nx)
        The frames.
    centers : ndarray, shape (n, 2)
        Each frame's star, ``(x, y)`` in 0-based pixels.
    target : tuple of 2 floats
        The center, ``(x, y)`` in 0-based pixels.

    Returns
    -------
    shifted : ndarray of float32, shape (n, ny, nx)
        The frames, shifted by linear interpolation, 0 outside.
    """
    from scipy import ndimage
    out = np.empty_like(imgs, dtype=np.float32)
    for i, (cx, cy) in enumerate(centers):
        f = np.where(np.isfinite(imgs[i]), imgs[i], 0.0)
        out[i] = ndimage.shift(f, (target[1] - cy, target[0] - cx), order=1, mode="constant", cval=0.0)
    return out


def make_reducer(datasets: Dict[str, Dataset], pxscale: Optional[float] = None, wavelength_m: Optional[float] = None,
                 diam_m: float = JWST_DIAM, psf: str = "auto",
                 psf_template: Union[None, str, np.ndarray] = None,
                 psf_center: Optional[Tuple[float, float]] = None, star_flux: Optional[float] = None,
                 injection_model: Optional[InjectionModel] = None,
                 mode: str = "ADI+RDI", weighting: str = "equal", max_workers=1, pool: str = "threads",
                 n_min_ref: int = 10,
                 fwhm_px: Optional[float] = None, outrad_cap: Optional[float] = None,
                 stpsf_kw: Optional[Dict[str, Any]] = None, backend: str = "pyklip",
                 log: Callable[[str], None] = print, **kw) -> PartitionedReducer:
    """Build a reducer of one partition per dataset, pyKLIP's or the built-in KLIP, with a JWST injection PSF.

    The reference library is searched in each engine's own terms.  The built-in engine (``backend="klip"``)
    takes ``set_reference_library`` (``nkeep_altroll`` and ``nkeep_psfref``, one count per pool), which
    pyKLIP refuses, and pyKLIP searches its own selection through ``set_native_library`` (``mode`` and
    ``maxnumbasis``).  ``mode`` maps onto the built-in engine as ADI to no reference cube, RDI to
    ``rdi_mode="rdi"`` and ADI+RDI to ``"ardi"``, and a dataset without a reference cube is reduced ADI.

    The injection model is, in order, ``injection_model``, then with ``psf="stpsf"`` the off-axis
    coronagraphic PSF of the actual mask from STPSF (:func:`klip_tpe.stpsf_psf.model_for_datasets`),
    which depends on separation and carries the mask's throughput, then ``psf_template``, else a
    Gaussian of ``1.028 lambda/D``.

    Parameters
    ----------
    datasets : dict of str to Dataset
        The datasets, by partition.
    pxscale : float or None, default None
        Plate scale, in arcsec per pixel.  None takes each dataset's ``meta``.
    wavelength_m : float or None, default None
        Wavelength, in m.  None takes each dataset's ``meta``.
    diam_m : float, default JWST_DIAM
        Telescope diameter for lambda/D, in m.
    psf : {"auto", "stpsf"}, default "auto"
        Whether to model the injection PSF with STPSF.  "auto" takes ``psf_template`` or the Gaussian.
    psf_template : str or ndarray or None, default None
        An offset PSF at the data's plate scale, a 2-D array or a FITS path, such as webbpsf's or
        ``webbpsf_ext``'s.
    psf_center : tuple of 2 floats or None, default None
        The template's center, ``(x, y)`` in pixels.  None takes its middle.
    star_flux : float or None, default None
        The star's flux, which puts the template on the data's flux scale.  None takes the template's
        own flux as the unit, so contrasts are relative to it.
    injection_model : InjectionModel or None, default None
        A model built already, which takes precedence.
    mode : {"ADI", "RDI", "ADI+RDI"}, default "ADI+RDI"
        The reduction's mode.
    weighting : {"equal", "sqrt_texp"}, default "equal"
        Weights of the partitions' combination (:class:`~klip_tpe.reducer.PartitionedReducer`).
    max_workers : int or str, default 1
        Number of partitions reduced at once.
    pool : {"threads", "processes", "auto"}, default "threads"
        Workers of the partitions.  pyKLIP starts its own process pool inside ``klip_parallelized``, and
        a forked worker, being daemonic, may not have children.
    n_min_ref : int, default 10
        Fewest reference frames a reduction takes.
    fwhm_px : float or None, default None
        FWHM of the PSF, in pixels.  None takes ``1.028 lambda/D``.
    outrad_cap : float or None, default None
        Largest outer radius of a zone, in pixels.  None takes half the frame less 2.
    stpsf_kw : dict or None, default None
        More keywords for :func:`klip_tpe.stpsf_psf.model_for_datasets`.
    backend : {"pyklip", "klip"}, default "pyklip"
        The engine, pyKLIP's or the built-in annular KLIP.
    log : callable, default print
        Function that takes each line of the log.
    kw : dict
        More defaults of every partition's reducer.

    Returns
    -------
    reducer : PartitionedReducer
        The reducer, named ``"spaceklip"``.

    Raises
    ------
    ValueError
        If a dataset's plate scale or wavelength is unknown, or ``backend`` is neither "pyklip" nor
        "klip".
    """
    if injection_model is None and str(psf).lower() == "stpsf":
        from ..stpsf_psf import model_for_datasets
        injection_model = model_for_datasets(list(datasets.values()), star_flux=star_flux or 1.0,
                                             pxscale=pxscale, log=log, **(stpsf_kw or {}))
    reducers = {}
    for pid, ds in datasets.items():
        px = pxscale or ds.meta.get("pxscale")
        lam = wavelength_m or ds.meta.get("wavelength_m")
        if not px or not np.isfinite(px):
            raise ValueError(f"{pid}: pixel scale unknown -- pass pxscale=")
        if not lam or not np.isfinite(lam):
            raise ValueError(f"{pid}: wavelength unknown -- pass wavelength_m=")
        lod = (lam / diam_m) * 206265.0 / px
        fw = fwhm_px or 1.028 * lod
        if injection_model is not None:
            model: InjectionModel = injection_model
        elif psf_template is not None:
            t = psf_template
            if isinstance(t, str):
                from astropy.io import fits
                t = np.asarray(fits.getdata(t), float)
            model = TemplatePSF(t, center=psf_center, ee_radius_px=2.0 * lod, star_flux=star_flux)
        else:
            model = GaussianPSF(fw, star_flux=star_flux or 1.0)
        ny = ds.cube.shape[-1]
        cap = outrad_cap or (ny / 2.0 - 2.0)
        eff_mode = mode if ds.ref_cube is not None else "ADI"
        if str(backend).lower() == "klip":
            m_up = str(eff_mode).upper()
            red = KLIPReducer(ds, pxscale=px, lam_m=lam, diam_m=diam_m, injection_model=model, fwhm_px=fw,
                              outrad_cap=cap, defaults={"use_rdi": "RDI" in m_up,
                                                        "rdi_mode": "rdi" if m_up == "RDI" else "ardi",
                                                        "n_min_ref": n_min_ref, **kw})
        elif str(backend).lower() == "pyklip":
            red = PyKLIPReducer(ds, pxscale=px, lam_m=lam, diam_m=diam_m, injection_model=model, fwhm_px=fw,
                                outrad_cap=cap, defaults={"mode": eff_mode, "n_min_ref": n_min_ref, **kw})
        else:
            raise ValueError(f"backend must be 'pyklip' or 'klip', not {backend!r}")
        red.name = f"spaceklip_{pid}"
        reducers[pid] = red
        log(f"  {pid}: {str(backend).lower()} {eff_mode}, lambda/D {lod:.2f} px, fwhm {fw:.2f} px, "
            f"injection {model.name}")
    pr = PartitionedReducer(reducers, weighting=weighting, max_workers=max_workers, pool=pool)
    pr.name = "spaceklip"
    pr.partition_label = "roll"                              # display wording only
    return pr
