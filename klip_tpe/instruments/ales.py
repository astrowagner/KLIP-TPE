"""LBTI/ALES adapter, an integral-field cube as wavelength groups, one partition each.

ALES, the LMIRCam lenslet spectrograph in the L band, delivers a ``(wavelength, frame, y, x)`` cube
of 99 channels at R ~ 40-55, so neighboring channels share a resolution element and a channel is
not the unit to optimize over.  The adapter cuts the usable channels into ``ngrp`` groups of equal
width in ``ln lambda``, about one resolution element each, averages each group's channels into
one ``(frame, y, x)`` cube, and hands the groups to :class:`~klip_tpe.reducer.PartitionedReducer`
as partitions, the way nights or rolls are.  Each group carries its own wavelength, so ``angsep``
in lambda/D and the Mawet apertures are right for it, its own PSF template and, when asked, its
own reference-star library.  One parameter block is shared by every group, and a 0/1 flag per group
(``sel_gNN``) lets the optimizer leave out at most ``max_excluded`` groups
(:class:`GroupCountGuard`).

A reference star (``load_ales(..., ref="HIP91748")``) is a fixed ARDI library by default.  With
``make_reducer(..., reflib=True)`` (``--ales-reflib``) it is a searched one, each target frame's
basis being the science frames the ADI rule admits plus the ``nkeep_psfref`` reference means most
correlated with it in the annulus.  ``nkeep_psfref`` is a search dimension from 0, plain ADI to
the bit, to the whole pool.  The reference frames are taken in consecutive means of ``ref_bin``
(``--ales-ref-bin``, 5 by default), so the count means one thing whatever the science ``bin``.

Conventions, from the ALES reduction's README, section 4:

* The products are north up and east left once every frame is rotated counter-clockwise by
  ``LBT_PARA + 180`` degrees.  The adapter adds the 180 to the angles once (``angle_offset``) and
  keeps ``truenorth = 0``.
* The star sits at the science cube's center ``((nx - 1) / 2, (ny - 1) / 2)`` and on the integer
  pixel ``(STARX, STARY)`` of the template's larger grid.
* The 0.0345"/spaxel and the 180 degrees are assumed, not calibrated on the night, and WISPIT 2b's
  position must not calibrate either, since its orbital motion since the published epoch is
  unknown.
* The template is the median of the registered science frames themselves, in the frames' units,
  so a contrast is a plain flux ratio to the star in that group.

Examples
--------
::

    from klip_tpe.instruments import ales
    datasets, lam_m, fwhm_px, table = ales.load_ales(PRODUCTS, TARGET)
    red = ales.make_reducer(datasets, lam_m, fwhm_px, max_workers=4)
    space = ales.make_space(red)                       # shared block + sel_g00 .. sel_g19
    space.project = ales.make_guard(red, max_excluded=10)
    objective, sampler = ales.default_config(red, known=[(RHO, PA)], forbidden_pa=[(PA, 25.0)])

On the command line it is ``klip-tpe near --instrument ales --root PRODUCTS ...``.  The products
are those the ALES reduction's handoff of 2026-10-13 describes.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..feasibility import FrameSelectionGuard, MinBinGuard, ReferenceCountGuard, ReferenceLibraryGuard, compose
from ..reducer import Dataset, PartitionedReducer
from ..space import SearchSpace
from . import generic

__all__ = ["ALES", "BASELINE", "group_table", "group_mean", "reference_means", "frame_blocks", "load_groups",
           "load_ales", "load_release", "release_nights", "make_reducer", "make_space", "make_guard",
           "default_config", "GroupCountGuard", "product_paths", "representative_group"]


class ALES:
    """ALES's instrument constants, the assumptions of the ALES reduction's README.

    Attributes
    ----------
    pxscale : float
        Plate scale, in arcsec per spaxel, assumed from MWC 758 and not calibrated on the night.
    diam_m : float
        Diameter of one LBT aperture, in meters (DX only, no Fizeau).
    fwhm_lod : float
        Measured broadband FWHM at 3.5 um, in lambda/D.
    angle_offset : float
        Offset added to the angles, in degrees, so derotation is counter-clockwise by
        ``LBT_PARA + 180``.
    texp_frame : float
        Exposure time of a frame, in seconds, five ramp intervals of 0.9835 s.
    c0 : int
        First usable channel.
    c1 : int
        Last usable channel.
    ngrp : int
        Number of groups over the usable channels, for R = 54 per group (15 gives one R = 40 element
        each).
    tag_r_in : float
        Inner radius of the quality tags, in spaxels, since r >= 6 is background-dominated there.
    """

    pxscale = 0.0345            # arcsec / spaxel, ASSUMED (MWC 758 value; not calibrated on the night)
    diam_m = 8.4                # one LBT aperture (DX only, no Fizeau)
    fwhm_lod = 1.03             # measured broadband FWHM / (lambda/D) at 3.5 um
    angle_offset = 180.0        # derotation = CCW by LBT_PARA + 180 deg
    texp_frame = 4.9176         # s per frame (5 x 0.9835 s ramp intervals)
    c0, c1 = 6, 88              # usable channels
    ngrp = 20                   # groups over them (R_group = 54; 15 = one R = 40 element each)
    tag_r_in = 2.0              # quality tags from r >= 2 spaxels (r >= 6 is background-dominated there)


#: the ALES reduction's own ADI-KLIP parameters, the seed of the search: 5-frame bins, a
#: 13-spaxel boxcar high-pass, k = 5, references at >= 0.75 FWHM = 0.77 lambda/D of rotation, no
#: frame selection (the no-cut corner of the three thresholds)
BASELINE: Dict[str, Any] = {"bin": 5, "filter": 13, "k_klip": 5, "angsep": 0.77, "anglemax": 45, "n_ang": 1,
                            "corr_thresh": 0.0, "noise_max": 3.0, "coronoise_max": 3.0}


def product_paths(products_dir: str, target: str = "WISPIT2") -> Dict[str, str]:
    """Return the ALES reduction's product files for a target.

    Parameters
    ----------
    products_dir : str
        Directory of the products.
    target : str, default "WISPIT2"
        Name of the target, the prefix of the files.

    Returns
    -------
    paths : dict of str to str
        ``cube``, ``angles``, ``waves``, ``psf``, ``frames`` and ``frametags``, such as
        ``<target>_cube_cen_clean.fits``.
    """
    p = lambda s: os.path.join(products_dir, f"{target}_{s}")
    return {"cube": p("cube_cen_clean.fits"), "angles": p("primary_angles_clean.fits"),
            "waves": p("waves.fits"), "psf": p("star_psf_cube.fits"),
            "frames": p("frame_numbers_clean.fits"), "frametags": p("frametags.csv")}


def group_table(waves_um: Sequence[float], c0: int = ALES.c0, c1: int = ALES.c1,
                ngrp: int = ALES.ngrp, name_fmt: str = "g{:02d}") -> List[Dict[str, Any]]:
    """Build constant-R wavelength groups over channels ``c0`` to ``c1``, inclusive.

    A channel's edges are the midpoints to its neighbors, and the group edges are equally spaced in
    ``ln lambda`` from the lower edge of ``c0`` to the upper edge of ``c1``.  A channel belongs to the
    group whose edges contain its wavelength, the handoff's rule (section 6.2).  A slot that receives no
    channel is left out and the names close up.

    Parameters
    ----------
    waves_um : sequence of float
        Wavelength of each channel, in micrometers.
    c0 : int, default ALES.c0
        First channel.
    c1 : int, default ALES.c1
        Last channel.
    ngrp : int, default ALES.ngrp
        Number of group slots.
    name_fmt : str, default "g{:02d}"
        Format of the group names.

    Returns
    -------
    table : list of dict
        One dict per group, with ``name``, ``channels``, ``lam_lo`` and ``lam_hi`` (its edges, um),
        ``lam_mean`` (the mean wavelength of its channels, um), ``n`` and ``slot``.

    Raises
    ------
    ValueError
        If the channels lie outside the cube or ``ngrp`` does not fit them.
    """
    w = np.asarray(waves_um, float).ravel()
    nch = w.size
    c0, c1, ngrp = int(c0), int(c1), int(ngrp)
    if not (0 <= c0 <= c1 < nch):
        raise ValueError(f"channels {c0}..{c1} outside 0..{nch - 1}")
    if ngrp < 1 or ngrp > c1 - c0 + 1:
        raise ValueError(f"ngrp = {ngrp} for {c1 - c0 + 1} channels")
    lo = 0.5 * (w[c0 - 1] + w[c0]) if c0 > 0 else w[c0] - 0.5 * (w[c0 + 1] - w[c0])
    hi = 0.5 * (w[c1] + w[c1 + 1]) if c1 + 1 < nch else w[c1] + 0.5 * (w[c1] - w[c1 - 1])
    edges = np.exp(np.linspace(np.log(lo), np.log(hi), ngrp + 1))
    ch = np.arange(c0, c1 + 1)
    g = np.clip(np.searchsorted(edges, w[ch], side="right") - 1, 0, ngrp - 1)
    out: List[Dict[str, Any]] = []
    for i in range(ngrp):
        members = ch[g == i]
        if members.size == 0:
            continue
        out.append({"name": name_fmt.format(len(out)), "channels": [int(c) for c in members],
                    "lam_lo": float(edges[i]), "lam_hi": float(edges[i + 1]),
                    "lam_mean": float(w[members].mean()), "n": int(members.size), "slot": i})
    return out


def group_mean(cube: np.ndarray, channels: Sequence[int], dtype=np.float32) -> np.ndarray:
    """Average a ``(wavelength, frame, y, x)`` cube over channels, ignoring NaN.

    The frames are read in blocks of about 200 MB of float64, so the full cube is never expanded and a
    memory-mapped FITS stays on disk.

    Parameters
    ----------
    cube : ndarray, shape (nwave, nframe, ny, nx)
        The IFS cube.
    channels : sequence of int
        The channels to average.
    dtype : numpy dtype, default numpy.float32
        Precision of the result.

    Returns
    -------
    mean : ndarray, shape (nframe, ny, nx)
        The mean over the channels.
    """
    ch = [int(c) for c in channels]
    nfr = cube.shape[1]
    per_frame = int(np.prod(cube.shape[2:])) * len(ch) * 8
    step = max(1, int(2.0e8 // max(per_frame, 1)))          # ~200 MB of float64 at a time
    out = None
    for f0 in range(0, nfr, step):
        block = np.stack([np.asarray(cube[c, f0:f0 + step], np.float64) for c in ch])
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN pixels: "Mean of empty slice"
            m = np.nanmean(block, axis=0)
        if out is None:
            out = np.empty((nfr,) + m.shape[1:], dtype)
        out[f0:f0 + step] = m
    return out


def _fwhm_px(lam_m: float, pxscale: float, diam_m: float, fwhm_lod: float) -> float:
    """Return the FWHM in pixels for a wavelength.

    Parameters
    ----------
    lam_m : float
        Wavelength, in meters.
    pxscale : float
        Plate scale, in arcsec per pixel.
    diam_m : float
        Telescope diameter, in meters.
    fwhm_lod : float
        FWHM in lambda/D.

    Returns
    -------
    fwhm : float
        The FWHM, in pixels.
    """
    return float(fwhm_lod * (lam_m / diam_m) * 206265.0 / pxscale)


def reference_means(cube: np.ndarray, n: int, blocks: Optional[Sequence[Any]] = None,
                    dtype=np.float32) -> np.ndarray:
    """Average consecutive runs of ``n`` frames of a cube, within each observing block.

    The frames a block has left over at its end are dropped, so blocks of 23, 41 and 37 frames in
    fives make 4 + 8 + 7 = 19 means and no mean straddles a gap.  The reference star's frames are not
    derotated, so no angle closes a run early, and a fixed run gives the searched ``nkeep_psfref`` one
    meaning for the whole search.

    Parameters
    ----------
    cube : ndarray, shape (nf, ny, nx)
        The reference frames.
    n : int
        Frames per mean.  1 or less returns the cube itself.
    blocks : sequence or None, default None
        Observing-block label of each frame.  None makes the whole sequence one block.
    dtype : numpy dtype, default numpy.float32
        Precision of the means.

    Returns
    -------
    means : ndarray
        The means, in order.

    Raises
    ------
    ValueError
        If the labels do not match the frames or no block holds ``n`` frames.
    """
    n = int(n)
    nf = int(cube.shape[0])
    if n <= 1:
        return np.asarray(cube, dtype)
    if blocks is None:
        starts = [0]
        ends = [nf]
    else:
        lab = np.asarray(list(blocks))
        if lab.size != nf:
            raise ValueError(f"{lab.size} block labels for {nf} reference frames")
        edges = [0] + [int(i) for i in np.flatnonzero(lab[1:] != lab[:-1]) + 1] + [nf]
        starts, ends = edges[:-1], edges[1:]
    runs = [(a + b * n, a + (b + 1) * n) for a, e in zip(starts, ends) for b in range((e - a) // n)]
    if not runs:
        raise ValueError(f"{nf} reference frames in blocks of {[e - a for a, e in zip(starts, ends)]} "
                         f"make no mean of {n}")
    out = np.empty((len(runs),) + tuple(cube.shape[1:]), dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN pixels
        for i, (a, b) in enumerate(runs):
            out[i] = np.nanmean(np.asarray(cube[a:b], np.float64), axis=0)
    return out


def frame_blocks(frametags_csv: str) -> Optional[np.ndarray]:
    """Read the ``block`` column of a reduction's ``<target>_frametags.csv``.

    Parameters
    ----------
    frametags_csv : str
        Path of the file, written by the reduction's ``post08_frametags.py``.

    Returns
    -------
    blocks : ndarray of str or None
        The observing-block label of each clean frame, or None when the file or the column is
        missing.
    """
    import csv
    if not frametags_csv or not os.path.exists(frametags_csv):
        return None
    with open(frametags_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "block" not in rows[0]:
        return None
    return np.array([str(r["block"]) for r in rows])


def _group_cube(cube, cached: Optional[Dict[str, Any]], name: str, channels: Sequence[int]) -> np.ndarray:
    """Return a group's ``(frame, y, x)`` cube, from the cache or as the mean of the cube's channels.

    Parameters
    ----------
    cube : ndarray or None
        The IFS cube, or None when the cache holds the group.
    cached : dict or None
        Group cubes by name, arrays or ``.npy`` paths.
    name : str
        Name of the group.
    channels : sequence of int
        The group's channels.

    Returns
    -------
    cube : ndarray
        The group's cube.

    Raises
    ------
    ValueError
        If the group is not cached and there is no cube.
    """
    if cached and name in cached:
        v = cached[name]
        return np.load(v) if isinstance(v, str) else np.asarray(v, np.float32)
    if cube is None:
        raise ValueError(f"group {name} is not in the cache and no cube was given")
    return group_mean(cube, channels)


def load_groups(cube: Optional[np.ndarray], angles: Sequence[float], waves_um: Sequence[float],
                psf: Optional[np.ndarray] = None, *, c0: int = ALES.c0, c1: int = ALES.c1,
                ngrp: int = ALES.ngrp, groups: Optional[Sequence[Dict[str, Any]]] = None,
                only: Optional[Sequence[str]] = None,
                pxscale: float = ALES.pxscale, diam_m: float = ALES.diam_m, fwhm_lod: float = ALES.fwhm_lod,
                angle_offset: float = ALES.angle_offset, angle_sign: float = 1.0,
                texp: Optional[float] = None, tags: Any = "auto", tag_r_in: float = ALES.tag_r_in,
                nan_policy: str = "keep", ref_cube: Optional[np.ndarray] = None,
                psf_center: Optional[Tuple[float, float]] = None,
                cached: Optional[Dict[str, Any]] = None, ref_cached: Optional[Dict[str, Any]] = None,
                source: Optional[str] = None, ref_bin: int = 1, ref_source: Optional[str] = None,
                ref_blocks: Optional[Sequence[Any]] = None, log: Callable[[str], None] = print
                ) -> Tuple[Dict[str, Dataset], Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    """Turn the wavelength groups of an IFS cube into :class:`~klip_tpe.reducer.Dataset` partitions.

    Parameters
    ----------
    cube : ndarray or None
        The IFS cube, shape (nwave, nframe, ny, nx), as astropy reads the products (a memory map is
        fine), or None when ``cached`` holds every group.
    angles : sequence of float
        LBT parallactic angle of each frame, in degrees, before ``angle_sign`` and ``angle_offset``.
    waves_um : sequence of float
        Wavelength of each channel, in micrometers.
    psf : ndarray or None, default None
        Template on its own grid, shape (nwave, ny', nx'), averaged over each group's channels.
    c0 : int, default ALES.c0
        First channel of the groups.
    c1 : int, default ALES.c1
        Last channel of the groups.
    ngrp : int, default ALES.ngrp
        Number of group slots.
    groups : sequence of dict or None, default None
        Group table to use instead of :func:`group_table`'s.
    only : sequence of str or None, default None
        Names of the groups to keep.  None keeps all.
    pxscale : float, default ALES.pxscale
        Plate scale, in arcsec per spaxel.
    diam_m : float, default ALES.diam_m
        Telescope diameter, in meters.
    fwhm_lod : float, default ALES.fwhm_lod
        FWHM in lambda/D.
    angle_offset : float, default ALES.angle_offset
        Offset added to the angles, in degrees.
    angle_sign : float, default 1.0
        Factor on the angles, applied before the offset.
    texp : float or None, default None
        Total exposure time, in seconds.  None takes the frames times :attr:`ALES.texp_frame`.
    tags : {"auto"}, dict or None, default "auto"
        Frame-quality tags, as for :func:`~klip_tpe.instruments.generic.load_cube`.
    tag_r_in : float, default ALES.tag_r_in
        Inner radius of the "auto" tags, in spaxels.
    nan_policy : {"keep", "zero"}, default "keep"
        What to do with NaN pixels.
    ref_cube : ndarray or None, default None
        A reference star's cube, shape (nwave, nref, ny, nx), averaged the same way and attached per
        group, which turns RDI and ARDI on by default in :func:`make_reducer`.
    psf_center : tuple of 2 floats or None, default None
        The template's ``(x, y)`` center.  None takes the 5 x 5 centroid of its peak.
    cached : dict or None, default None
        Ready group cubes by name, arrays or ``.npy`` paths.
    ref_cached : dict or None, default None
        Ready reference group cubes by name, unbinned.
    source : str or None, default None
        Where the cube came from, for the provenance.
    ref_bin : int, default 1
        Reference frames per mean (:func:`reference_means`).  1 takes the frames as they are.
    ref_source : str or None, default None
        Where the reference cube came from, for the provenance.
    ref_blocks : sequence or None, default None
        Observing-block label of each reference frame.
    log : callable, default print
        Function called with each log line.

    Returns
    -------
    datasets : dict of str to Dataset
        The partitions, each with ``meta["provenance"]`` recording the group and any reference star.
    lam_m : dict of str to float
        Mean wavelength of each group, in meters.
    fwhm_px : dict of str to float
        FWHM of each group, in pixels.
    table : list of dict
        The group table, with ``fwhm_px`` and ``n_nan``, the NaN pixels left in the group's cube.

    Raises
    ------
    ValueError
        If the cube is not 4-d, the channels, frames, angles or template do not match, no group is
        kept, or a group is neither cached nor in the cube.
    """
    w = np.asarray(waves_um, float).ravel()
    if cube is not None:
        if np.ndim(cube) != 4:
            raise ValueError(f"an IFS cube is (nwave, nframe, ny, nx), got {np.shape(cube)}")
        if w.size != cube.shape[0]:
            raise ValueError(f"{cube.shape[0]} channels but {w.size} wavelengths")
    a = np.asarray(angles, float).ravel()
    if cube is not None and a.size != cube.shape[1]:
        raise ValueError(f"{cube.shape[1]} frames but {a.size} angles")
    table = [dict(g) for g in (groups if groups is not None else group_table(w, c0, c1, ngrp))]
    if only is not None:
        keep = {str(n) for n in only}
        table = [g for g in table if g["name"] in keep]
        if not table:
            raise ValueError(f"none of {sorted(keep)} is a group")
    if psf is not None and np.ndim(psf) == 3 and psf.shape[0] != w.size:
        raise ValueError(f"template has {psf.shape[0]} channels, the cube {w.size}")
    datasets: Dict[str, Dataset] = {}
    lam_m: Dict[str, float] = {}
    fwhm_px: Dict[str, float] = {}
    for g in table:
        name, ch = g["name"], g["channels"]
        cg = _group_cube(cube, cached, name, ch)
        if cg.shape[0] != a.size:
            raise ValueError(f"group {name}: {cg.shape[0]} frames but {a.size} angles")
        tpl = None
        if psf is not None:
            t = np.asarray(psf, float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                tpl = np.nanmean(t[ch], axis=0) if t.ndim == 3 else t
            tpl = np.where(np.isfinite(tpl), tpl, 0.0)          # the template's all-NaN pixels
        rg = None
        meta = {"provenance": {"instrument": "ALES", "group": name, "channels": list(ch),
                               "lam_um": [float(w[c]) for c in ch], "lam_mean_um": g["lam_mean"],
                               "lam_edges_um": [g["lam_lo"], g["lam_hi"]], "nframes": int(cg.shape[0]),
                               "source": source, "angle_offset_deg": float(angle_offset),
                               "angle_sign": float(angle_sign)}}
        if ref_cube is not None or (ref_cached and name in ref_cached):
            raw = _group_cube(ref_cube, ref_cached, name, ch)
            rg = reference_means(raw, ref_bin, blocks=ref_blocks)
            sizes = None
            if ref_blocks is not None:
                lab = np.asarray(list(ref_blocks))
                cuts = np.flatnonzero(lab[1:] != lab[:-1]) + 1
                sizes = [int(v) for v in np.diff(np.r_[0, cuts, lab.size])]
            meta["provenance"].update(ref_source=ref_source, ref_frames=int(raw.shape[0]), ref_blocks=sizes,
                                      ref_bin=int(max(ref_bin, 1)), n_ref=int(rg.shape[0]))
        if psf_center is not None:
            meta["psf_center"] = (float(psf_center[0]), float(psf_center[1]))
        ds = generic.load_cube(cg, a, psf=tpl, name=name, angle_sign=angle_sign, angle_offset=angle_offset,
                               texp=texp if texp else float(cg.shape[0]) * ALES.texp_frame,
                               tags=tags, tag_r_in=tag_r_in, nan_policy=nan_policy, meta=meta,
                               ref_cube=rg)
        datasets[name] = ds
        lam_m[name] = g["lam_mean"] * 1e-6
        fwhm_px[name] = _fwhm_px(lam_m[name], pxscale, diam_m, fwhm_lod)
        g["n_nan"] = int(np.count_nonzero(~np.isfinite(ds.cube)))
        g["fwhm_px"] = fwhm_px[name]
        rtxt = ""
        if rg is not None:
            pv = meta["provenance"]
            blk = f" in blocks of {'+'.join(str(v) for v in pv['ref_blocks'])}" if pv["ref_blocks"] else ""
            rtxt = (f", {pv['ref_frames']} reference frames" if pv["ref_bin"] <= 1 else
                    f", {pv['ref_frames']} reference frames{blk} in means of {pv['ref_bin']}: {pv['n_ref']}")
        log(f"  {name}: channels {ch[0]}-{ch[-1]} ({len(ch)}), {g['lam_lo']:.4f}-{g['lam_hi']:.4f} um, "
            f"mean {g['lam_mean']:.3f} um, fwhm {fwhm_px[name]:.2f} px, {ds.cube.shape[0]} frames, "
            f"{g['n_nan']} NaN px" + rtxt)
    return datasets, lam_m, fwhm_px, table


def _n_ref_frames(rpaths: Dict[str, str], rcached: Dict[str, Any], names: Sequence[str]) -> int:
    """Return the reference cube's frame count, from a cached group mean or the FITS header.

    No data are read.

    Parameters
    ----------
    rpaths : dict of str to str
        The reference star's product paths.
    rcached : dict of str to str
        Cached reference group cubes, ``.npy`` paths by name.
    names : sequence of str
        Names of the groups.

    Returns
    -------
    n : int
        The number of reference frames.
    """
    for n in names:
        f = rcached.get(n)
        if f:
            return int(np.load(f, mmap_mode="r").shape[0])
    from astropy.io import fits
    with fits.open(rpaths["cube"], memmap=True) as h:
        return int(h[0].header.get("NAXIS3", h[0].data.shape[1]))


def _fits(path: str, memmap: bool = True):
    """Open a FITS file and return its primary data and header.

    Parameters
    ----------
    path : str
        Path of the file.
    memmap : bool, default True
        Whether to memory-map the data.

    Returns
    -------
    data : ndarray
        The primary HDU's data.
    header : Header
        The primary HDU's header.
    """
    from astropy.io import fits
    h = fits.open(path, memmap=memmap)
    return h[0].data, h[0].header


def load_ales(products_dir: str, target: str = "WISPIT2", ref: Optional[str] = None, *,
              c0: int = ALES.c0, c1: int = ALES.c1, ngrp: int = ALES.ngrp, only: Optional[Sequence[str]] = None,
              pxscale: float = ALES.pxscale, diam_m: float = ALES.diam_m, fwhm_lod: float = ALES.fwhm_lod,
              angle_offset: float = ALES.angle_offset, angle_sign: float = 1.0, tags: Any = "auto",
              tag_r_in: float = ALES.tag_r_in, nan_policy: str = "keep", cache: Optional[str] = None,
              ref_bin: int = 1, log: Callable[[str], None] = print
              ) -> Tuple[Dict[str, Dataset], Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    """Run :func:`load_groups` on the reduction's product files (:func:`product_paths`).

    The template's star is at the header's ``STARX`` and ``STARY`` when it has them, else at the
    centroid of its peak.  The reference means never straddle an observing block, the blocks coming
    from the ``block`` column of ``<ref>_frametags.csv`` beside the products (:func:`frame_blocks`).

    Parameters
    ----------
    products_dir : str
        Directory of the products.
    target : str, default "WISPIT2"
        Name of the target.
    ref : str or None, default None
        Name of a reference star whose products sit beside the target's, such as "HIP91748".  Its group
        means become each group's RDI library.
    c0 : int, default ALES.c0
        First channel of the groups.
    c1 : int, default ALES.c1
        Last channel of the groups.
    ngrp : int, default ALES.ngrp
        Number of group slots.
    only : sequence of str or None, default None
        Names of the groups to keep.
    pxscale : float, default ALES.pxscale
        Plate scale, in arcsec per spaxel.
    diam_m : float, default ALES.diam_m
        Telescope diameter, in meters.
    fwhm_lod : float, default ALES.fwhm_lod
        FWHM in lambda/D.
    angle_offset : float, default ALES.angle_offset
        Offset added to the angles, in degrees.
    angle_sign : float, default 1.0
        Factor on the angles.
    tags : {"auto"}, dict or None, default "auto"
        Frame-quality tags.
    tag_r_in : float, default ALES.tag_r_in
        Inner radius of the "auto" tags, in spaxels.
    nan_policy : {"keep", "zero"}, default "keep"
        What to do with NaN pixels.
    cache : str or None, default None
        Directory where the group cubes are written as ``.npy`` the first time and read back after,
        which matters on a slow volume.  The reference star's group means are cached unbinned, so
        ``ref_bin`` may change between runs.
    ref_bin : int, default 1
        Reference frames per mean (:func:`reference_means`).  1 takes every frame.
    log : callable, default print
        Function called with each log line.

    Returns
    -------
    datasets : dict of str to Dataset
        The partitions.
    lam_m : dict of str to float
        Mean wavelength of each group, in meters.
    fwhm_px : dict of str to float
        FWHM of each group, in pixels.
    table : list of dict
        The group table (:func:`load_groups`).
    """
    paths = product_paths(products_dir, target)
    w = np.asarray(_fits(paths["waves"], memmap=False)[0], float).ravel()
    a = np.asarray(_fits(paths["angles"], memmap=False)[0], float).ravel()
    psf, ph = _fits(paths["psf"], memmap=False)
    psf = np.asarray(psf, float)
    center = None
    if ph.get("STARX") is not None and ph.get("STARY") is not None:
        center = (float(ph["STARX"]), float(ph["STARY"]))
    table = group_table(w, c0, c1, ngrp)
    names = [g["name"] for g in table] if only is None else [str(n) for n in only]
    key = f"{target}_c{c0}-{c1}_n{ngrp}"
    rkey = None if ref is None else f"{ref}_c{c0}-{c1}_n{ngrp}"

    def from_cache(k):
        if not (cache and k):
            return {}
        return {n: f for n in names for f in [os.path.join(cache, f"{k}_{n}.npy")] if os.path.exists(f)}

    cached, rcached = from_cache(key), from_cache(rkey)
    cube = None if all(n in cached for n in names) else _fits(paths["cube"])[0]
    rcube = rpath = rblocks = None
    if ref is not None:
        rpaths = product_paths(products_dir, ref)
        rpath = rpaths["cube"]
        rblocks = frame_blocks(rpaths["frametags"])
        if rblocks is not None and rblocks.size != _n_ref_frames(rpaths, rcached, names):
            log(f"  ({ref}_frametags.csv has {rblocks.size} rows for a cube of another length; the blocks are ignored)")
            rblocks = None
        if not all(n in rcached for n in names):
            rcube = _fits(rpath)[0]
            if cache:
                # the cache holds the reference group means as they are; the means of
                # ref_bin are taken from it on every load, so ref_bin is free to change
                os.makedirs(cache, exist_ok=True)
                for g in table:
                    if g["name"] in names and g["name"] not in rcached:
                        f = os.path.join(cache, f"{rkey}_{g['name']}.npy")
                        np.save(f, group_mean(rcube, g["channels"]))
                        rcached[g["name"]] = f
                rcube = None
    log(f"ALES {target}: {w.size} channels, {a.size} frames; {len(table)} groups over channels {c0}-{c1} "
        f"({w[c0]:.4f}-{w[c1]:.4f} um)" + (f"; reference star {ref}" if ref else "")
        + (f"; {len(cached)} group cubes from the cache" if cached else ""))
    datasets, lam_m, fwhm_px, table = load_groups(
        cube, a, w, psf, c0=c0, c1=c1, ngrp=ngrp, groups=table, only=only, pxscale=pxscale, diam_m=diam_m,
        fwhm_lod=fwhm_lod, angle_offset=angle_offset, angle_sign=angle_sign, tags=tags, tag_r_in=tag_r_in,
        nan_policy=nan_policy, ref_cube=rcube, psf_center=center, cached=cached, ref_cached=rcached,
        source=paths["cube"], ref_bin=ref_bin, ref_source=rpath, ref_blocks=rblocks, log=log)
    if cache:
        os.makedirs(cache, exist_ok=True)
        for g in table:
            f = os.path.join(cache, f"{key}_{g['name']}.npy")
            if not os.path.exists(f):
                np.save(f, datasets[g["name"]].cube)
    return datasets, lam_m, fwhm_px, table


def release_nights(files: Dict[str, str]) -> List[str]:
    """Return the nights of a data set in the release layout, the dates of its ``meta_<date>`` roles.

    Parameters
    ----------
    files : dict of str to str
        The data set's files by role.

    Returns
    -------
    nights : list of str
        The dates, in order.
    """
    return sorted(str(k)[5:] for k in files if str(k).startswith("meta_"))


def load_release(files: Dict[str, str], nights: Optional[Sequence[str]] = None, only: Optional[Sequence[str]] = None,
                 labels: Optional[Dict[str, str]] = None, *, pxscale: Optional[float] = None,
                 diam_m: float = ALES.diam_m, fwhm_lod: float = ALES.fwhm_lod, angle_offset: Optional[float] = None,
                 angle_sign: float = 1.0, tags: Any = "auto", tag_r_in: float = ALES.tag_r_in,
                 nan_policy: str = "keep", log: Callable[[str], None] = print
                 ) -> Tuple[Dict[str, Dataset], Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    """Load an ALES data set in the release layout as partitions, one per night and wavelength group.

    The layout is the one :func:`klip_tpe.datasets.fetch` returns for "lbti_ales_mwc758".  For each
    night a ``meta_<date>`` file holds one template per group in its primary HDU, then the ``ANGLES``,
    ``FRAMES``, ``GROUPS`` and ``WAVES`` extensions, and one ``gNN_<date>`` cube per group holds the mean
    of that group's channels.  Every night shares one group grid (``GRIDLO``, ``GRIDHI``, ``NGROUP``), so
    ``g07`` covers the same wavelengths each night while its mean wavelength follows that night's
    channels.

    Parameters
    ----------
    files : dict of str to str
        The data set's files by role.
    nights : sequence of str or None, default None
        Nights to keep, as dates or labels.  None keeps all.
    only : sequence of str or None, default None
        Groups to keep, the same on every night.
    labels : dict of str to str or None, default None
        Label of each date.  None labels the nights n1, n2 and on in the date order of the whole data
        set, so a night loaded alone keeps its label.
    pxscale : float or None, default None
        Plate scale, in arcsec per spaxel.  None reads ``PIXSCALE`` from the meta header.
    diam_m : float, default ALES.diam_m
        Telescope diameter, in meters.
    fwhm_lod : float, default ALES.fwhm_lod
        FWHM in lambda/D.
    angle_offset : float or None, default None
        Offset added to the angles, in degrees.  None reads it from ``ANGCONV`` (``LBT_PARA + 180``).
    angle_sign : float, default 1.0
        Factor on the angles.
    tags : {"auto"}, dict or None, default "auto"
        Frame-quality tags.
    tag_r_in : float, default ALES.tag_r_in
        Inner radius of the "auto" tags, in spaxels.
    nan_policy : {"keep", "zero"}, default "keep"
        What to do with NaN pixels.
    log : callable, default print
        Function called with each log line.

    Returns
    -------
    datasets : dict of str to Dataset
        The partitions, named ``<label>_<group>`` such as ``n1_g07``.  Each template is the night's
        own, so a contrast is a flux ratio to the star in that group on that night.
    lam_m : dict of str to float
        Mean wavelength of each partition, in meters.
    fwhm_px : dict of str to float
        FWHM of each partition, in pixels.
    table : list of dict
        One row per partition, with ``name``, ``night``, ``date``, ``group``, ``channels``, ``lam_lo``,
        ``lam_hi``, ``lam_mean``, ``n``, ``slot``, ``fwhm_px`` and ``n_nan``.

    Raises
    ------
    ValueError
        If there is no meta file, a night is unknown, the nights' grids differ, a cube does not match
        its angles, or no group is kept.
    """
    from astropy.io import fits
    have = release_nights(files)
    if not have:
        raise ValueError("no meta_<date> file in this data set (the release layout of datasets.fetch)")
    # n1, n2, ... by position in the whole data set, so a night keeps its label when it is loaded alone
    lab = {d: f"n{i + 1}" for i, d in enumerate(have)}
    lab.update({str(k): str(v) for k, v in (labels or {}).items()})
    by_label = {v: k for k, v in lab.items()}
    keep = have if nights is None else [by_label.get(str(d), str(d)) for d in nights]
    unknown = [d for d in keep if d not in have]
    if unknown:
        raise ValueError(f"nights {unknown} are not in this data set ({have}, labelled {[lab[d] for d in have]})")
    datasets: Dict[str, Dataset] = {}
    lam_m: Dict[str, float] = {}
    fwhm_px: Dict[str, float] = {}
    table: List[Dict[str, Any]] = []
    grid = None
    for date in keep:
        with fits.open(files[f"meta_{date}"], memmap=False) as hd:
            h = hd[0].header
            tpl_all = np.asarray(hd[0].data, float)
            ang = np.asarray(hd["ANGLES"].data, float).ravel()
            gt = hd["GROUPS"].data
            rows = [{"group": str(gt["NAME"][i]).strip(), "lam_lo": float(gt["LAM_LO"][i]),
                     "lam_hi": float(gt["LAM_HI"][i]), "lam_mean": float(gt["LAM_MEAN"][i]),
                     "n": int(gt["NCHAN"][i]), "c0": int(gt["CHAN0"][i]), "c1": int(gt["CHAN1"][i]), "slot": i}
                    for i in range(len(gt))]
        g = (h.get("GRIDLO"), h.get("GRIDHI"), h.get("NGROUP"))
        if grid is None:
            grid = g
        elif g != grid:
            raise ValueError(f"night {date} has the group grid {g}, the first night {grid}: the groups would not match")
        px = float(pxscale if pxscale is not None else h.get("PIXSCALE", ALES.pxscale))
        off = angle_offset
        if off is None:
            conv = str(h.get("ANGCONV", ""))
            off = float(conv.split("+")[1]) if "+" in conv else ALES.angle_offset
        texp = float(h.get("EXPTIME", ALES.texp_frame))
        ny, nx = tpl_all.shape[1:]
        centre = ((nx - 1) / 2.0, (ny - 1) / 2.0)
        log(f"ALES night {lab[date]} ({date[:4]}-{date[4:6]}-{date[6:]}): {ang.size} frames, LBT_PARA "
            f"{ang.min():.2f} to {ang.max():.2f} deg, {len(rows)} groups over {grid[0]}-{grid[1]} um")
        for r in rows:
            if only is not None and r["group"] not in {str(o) for o in only}:
                continue
            pid = f"{lab[date]}_{r['group']}"
            cube = np.asarray(fits.getdata(files[f"{r['group']}_{date}"]), np.float32)
            if cube.shape[0] != ang.size:
                raise ValueError(f"{pid}: {cube.shape[0]} frames but {ang.size} angles")
            tpl = np.where(np.isfinite(tpl_all[r["slot"]]), tpl_all[r["slot"]], 0.0)
            ch = list(range(r["c0"], r["c1"] + 1))
            meta = {"psf_center": centre,
                    "provenance": {"instrument": "ALES", "night": lab[date], "date": date, "group": r["group"],
                                   "channels": ch, "lam_mean_um": r["lam_mean"],
                                   "lam_edges_um": [r["lam_lo"], r["lam_hi"]], "nframes": int(cube.shape[0]),
                                   "source": files[f"{r['group']}_{date}"], "angle_offset_deg": float(off),
                                   "angle_sign": float(angle_sign), "pxscale": px}}
            ds = generic.load_cube(cube, ang, psf=tpl, name=pid, angle_sign=angle_sign, angle_offset=off,
                                   texp=texp * cube.shape[0], tags=tags, tag_r_in=tag_r_in,
                                   nan_policy=nan_policy, meta=meta)
            datasets[pid] = ds
            lam_m[pid] = r["lam_mean"] * 1e-6
            fwhm_px[pid] = _fwhm_px(lam_m[pid], px, diam_m, fwhm_lod)
            table.append({"name": pid, "night": lab[date], "date": date, "group": r["group"], "channels": ch,
                          "lam_lo": r["lam_lo"], "lam_hi": r["lam_hi"], "lam_mean": r["lam_mean"], "n": r["n"],
                          "slot": r["slot"], "fwhm_px": fwhm_px[pid],
                          "n_nan": int(np.count_nonzero(~np.isfinite(ds.cube)))})
    if not datasets:
        raise ValueError(f"none of the groups {list(only or [])} is in this data set")
    log(f"  {len(datasets)} partitions ({len(keep)} night{'s' if len(keep) > 1 else ''} x "
        f"{len(datasets) // max(len(keep), 1)} groups), FWHM {min(fwhm_px.values()):.2f}-{max(fwhm_px.values()):.2f} px")
    return datasets, lam_m, fwhm_px, table


def representative_group(lam_m: Dict[str, float], at_m: Optional[float] = None) -> str:
    """Return the group whose wavelength is nearest a wavelength, to stand for the combined image.

    Its FWHM and matched filter stand for the combination's.

    Parameters
    ----------
    lam_m : dict of str to float
        Mean wavelength of each group, in meters.
    at_m : float or None, default None
        The wavelength, in meters.  None takes the mean of the groups'.

    Returns
    -------
    name : str
        Name of the group.
    """
    target = float(np.mean(list(lam_m.values()))) if at_m is None else float(at_m)
    return min(lam_m, key=lambda n: abs(lam_m[n] - target))


def make_reducer(datasets: Dict[str, Dataset], lam_m: Dict[str, float], fwhm_px: Dict[str, float], *,
                 pxscale: float = ALES.pxscale, diam_m: float = ALES.diam_m, ee_radius_px: float = 3.0,
                 representative: Optional[str] = None, weighting: str = "equal", max_workers="auto",
                 pool: str = "auto", fast: bool = False, n_min_ref: int = 10, outrad_cap: Optional[float] = None,
                 use_rdi: Optional[bool] = None, rdi_mode: str = "ardi", backend: str = "klip",
                 table: Optional[Sequence[Dict[str, Any]]] = None, defaults: Optional[Dict[str, Any]] = None,
                 reflib: bool = False, reflib_science: str = "all", reflib_metric: str = "cc",
                 reflib_n_min_ref: int = 2, sdi: bool = False, sdi_cap: int = 100, sdi_lenders: str = "all",
                 spectrum: Any = None, spectrum_norm: Any = None, spectrum_label: Optional[str] = None,
                 log: Callable[[str], None] = print) -> PartitionedReducer:
    """Build the groups' reducer, :func:`~klip_tpe.instruments.generic.make_reducer` with the group at the mean wavelength standing for the combination.

    The metric's FWHM and matched filter and the sampler's floor come from the representative group
    (:func:`representative_group`), not from the bluest.  The science frames' high-pass is NaN-aware
    here (``hp_nan_aware``, which ``defaults`` may override), since the registered frames carry NaN at
    their edges and the boxcar of the IDL predecessor spreads a NaN over the whole frame.  ``table``
    becomes the reducer's provenance, which the run's setup file carries.

    Parameters
    ----------
    datasets : dict of str to Dataset
        The groups.
    lam_m : dict of str to float
        Mean wavelength of each group, in meters.
    fwhm_px : dict of str to float
        FWHM of each group, in pixels.
    pxscale : float, default ALES.pxscale
        Plate scale, in arcsec per spaxel.
    diam_m : float, default ALES.diam_m
        Telescope diameter, in meters.
    ee_radius_px : float, default 3.0
        Radius of the template's normalization, in spaxels, the reduction's aperture, so a contrast is
        a flux ratio there.
    representative : str or None, default None
        The group standing for the combination.  None takes :func:`representative_group`.
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
    outrad_cap : float or None, default None
        Maximum outer radius of a zone, in pixels.
    use_rdi : bool or None, default None
        Whether to build the basis from the reference cube.  None does so when a group carries one.
    rdi_mode : {"ardi", "rdi"}, default "ardi"
        How a fixed reference cube enters the basis.
    backend : {"klip", "pyklip", "vip"}, default "klip"
        The reduction backend.
    table : sequence of dict or None, default None
        The group table, from :func:`load_groups`.
    defaults : dict or None, default None
        More default request parameters.
    reflib : bool, default False
        Whether the reference star is a searched library, one per group in the ``rotation`` form
        (:meth:`~klip_tpe.reducer.KLIPReducer.set_reference_library`).  Each target's basis is then
        the science frames the ADI rule admits plus the ``nkeep_psfref`` reference means most
        correlated with it, searched from 0, plain ADI to the bit, to the whole pool.
    reflib_science : {"all", "searched"}, default "all"
        "all" keeps every admitted science frame, and "searched" searches the science count too
        (``nkeep_altroll``, whose zero is pure RDI).
    reflib_metric : str, default "cc"
        Similarity measure of the library's ranking.
    reflib_n_min_ref : int, default 2
        The library's own floor on a target's references.
    sdi : bool, default False
        Whether to add the other groups' frames as an SDI pool, up to ``nkeep_sdi`` per target,
        searched from 0 to ``sdi_cap``, magnified about the star to the group's wavelength.  The
        rows are the most correlated among those whose planet moved at least ``angsep`` by rotation
        and rescaling together, within ``anglemax``.
    sdi_cap : int, default 100
        Maximum ``nkeep_sdi``.
    sdi_lenders : {"all", "selected"}, default "all"
        Which groups lend frames.  Under "all" an excluded group still does.
    spectrum : callable, tuple, dict or None, default None
        Spectrum of the injected sources, the relative contrast to the star
        (:meth:`~klip_tpe.reducer.PartitionedReducer.set_injection_spectrum`).  None injects the same
        contrast in every group, a source with the star's colors.
    spectrum_norm : object or None, default None
        Where the spectrum's factor is 1, a partition or a wavelength in meters.  None takes the
        representative group.
    spectrum_label : str or None, default None
        Name of the spectrum, for the run's setup file.
    log : callable, default print
        Function called with each log line.

    Returns
    -------
    reducer : PartitionedReducer
        The reducer of the groups.

    Raises
    ------
    ValueError
        If an SDI pool spans several nights, ``reflib_science`` or ``sdi_lenders`` is unknown, or a
        searched library lacks a reference star in some group.

    Notes
    -----
    With ``sdi``, a reference star given (``load_ales(ref=...)``) is the library's third pool,
    searched, with or without ``reflib``.  ``nkeep_sdi = nkeep_psfref = 0`` is plain ADI to the bit and
    costs plain ADI, since the other groups are prepared only when ``nkeep_sdi`` asks.
    """
    rep = representative or representative_group(lam_m)
    d = {"hp_nan_aware": True}
    d.update(defaults or {})
    red = generic.make_reducer(datasets, pxscale=pxscale, lam_m=lam_m, diam_m=diam_m, fwhm_px=fwhm_px,
                               ee_radius_px=ee_radius_px, weighting=weighting, max_workers=max_workers,
                               pool=pool, fast=fast, n_min_ref=n_min_ref, outrad_cap=outrad_cap,
                               use_rdi=use_rdi, rdi_mode=rdi_mode, backend=backend, defaults=d,
                               partition_label="group", log=log)
    r = red.reducers[rep]
    red.representative = rep
    red.pxscale, red.fwhm, red.lam_over_d_px = r.pxscale, r.fwhm, r.lam_over_d_px
    red.name = "ales"
    for pid, rr in red.reducers.items():
        rr.name = f"ales_{pid}"
    red.provenance = {"instrument": "ALES", "representative": rep, "ee_radius_px": float(ee_radius_px),
                      "groups": [dict(g) for g in table] if table is not None else
                      [{"name": n, "lam_mean": lam_m[n] * 1e6, "fwhm_px": fwhm_px[n]} for n in datasets]}
    log(f"  representative group {rep}: {lam_m[rep] * 1e6:.3f} um, fwhm {red.fwhm:.2f} px, "
        f"lambda/D {red.lam_over_d_px:.2f} px (the metric's and the sampler's)")
    if spectrum is not None:
        red.set_injection_spectrum(spectrum, norm=spectrum_norm, label=spectrum_label)
    if reflib or sdi:
        nights = {((ds.meta or {}).get("provenance") or {}).get("night") for ds in datasets.values()} - {None}
        if sdi and len(nights) > 1:
            raise ValueError(f"an SDI pool takes frames from every other partition, and these come from "
                             f"{len(nights)} nights ({sorted(nights)}).  Load one night for SDI "
                             f"(load_release(nights=[...]))")
        if reflib_science not in ("all", "searched"):
            raise ValueError(f"reflib_science is 'all' or 'searched', not {reflib_science!r}")
        if str(sdi_lenders) not in ("all", "selected"):
            raise ValueError(f"sdi_lenders is 'all' or 'selected', not {sdi_lenders!r}")
        without = [pid for pid, ds in datasets.items() if ds.ref_cube is None]
        if reflib and without:
            raise ValueError(f"a searched reference library needs a reference star in every group; "
                             f"{without[:4]} have none (load_ales(ref=...))")
        has_ref = not without
        pools = (["sdi"] if sdi else []) + (["psfref"] if has_ref else [])
        searched = None if reflib_science == "searched" else tuple(pools)
        nfr = {pid: int(rr.data.nframes) for pid, rr in red.reducers.items()}
        for pid, rr in red.reducers.items():
            sdi_spec = None
            if sdi:
                sdi_spec = {"n_frames": int(sum(v for q, v in nfr.items() if q != pid)), "cap": int(sdi_cap),
                            "lenders": list(red.reducers)}
            rr.set_reference_library(partition=np.zeros(nfr[pid], int), ref_group="psfref",
                                     n_min_ref=reflib_n_min_ref, metric=reflib_metric, rotation=True,
                                     searched=searched, sdi=sdi_spec)
        red.sdi_lenders = str(sdi_lenders)
        pv = (next(iter(datasets.values())).meta or {}).get("provenance", {})
        rec: Dict[str, Any] = {"metric": reflib_metric, "n_min_ref": int(reflib_n_min_ref), "rotation": True,
                               "pools": ["altroll"] + pools,
                               "science_pool": ("every frame the ADI rule admits (angsep, anglemax per annulus), "
                                                "not searched" if searched else
                                                "searched, nkeep_altroll (0 = pure RDI)")}
        if has_ref:
            n_ref = {pid: int(ds.ref_cube.shape[0]) for pid, ds in datasets.items()}
            rec.update(pool="psfref", source=pv.get("ref_source"), ref_frames=pv.get("ref_frames"),
                       ref_blocks=pv.get("ref_blocks"), ref_bin=pv.get("ref_bin"),
                       n_ref=n_ref[rep] if len(set(n_ref.values())) == 1 else n_ref,
                       nkeep_psfref=f"searched 0-{n_ref[rep]}" + ("" if sdi else " (0 = plain ADI)"))
        if sdi:
            rec["sdi"] = {"cap": int(sdi_cap), "lenders": str(sdi_lenders), "n_groups": len(red.reducers),
                          "nkeep_sdi": f"searched 0-{int(sdi_cap)}, the rows most correlated with the target "
                                       f"frame among the eligible ones (0 = no SDI)",
                          "eligibility": "the planet's displacement at the annulus mid radius, by rotation and "
                                         "rescaling together, at least angsep (lambda/D), and |dPA| <= anglemax",
                          "rescaling": "every other group's prepared frames (selected, injected, high-passed, "
                                       "binned with the searched bin) magnified about the star by "
                                       "lambda_target / lambda_group, cubic B-spline, NaN-aware",
                          "ranking": "zero-lag correlation with the target frame in the annulus, on this "
                                     "evaluation's frames"}
            rec["adi_point"] = "nkeep_sdi = nkeep_psfref = 0 is plain ADI" if has_ref else "nkeep_sdi = 0 is plain ADI"
        red.provenance["reference_library"] = rec
        log("  searched reference library, pools: " + ", ".join(rec["pools"])
            + (f"; {rec['n_ref']} reference means per group" if has_ref else "")
            + (f" ({pv['ref_frames']} frames" + (f" in blocks of {'+'.join(str(v) for v in pv['ref_blocks'])}"
                                                   if pv.get("ref_blocks") else "")
               + f" in means of {pv['ref_bin']})" if has_ref and pv.get("ref_bin", 1) > 1 else "")
            + (f"; SDI rows from the other groups, nkeep_sdi 0-{int(sdi_cap)}, lenders {sdi_lenders}" if sdi else "")
            + "; science frames: " + ("the ADI rule, all of them" if searched else "the ADI rule, nkeep_altroll searched"))
    return red


def make_space(reducer: PartitionedReducer, defaults: Optional[Dict[str, Any]] = None,
               bin_range: Tuple[int, int] = (1, 30), selection: Optional[str] = "binary",
               max_drop: int = 10, **kw) -> SearchSpace:
    """Build one parameter block shared by every group, plus the group flags.

    ``bin``, ``n_ang``, ``filter``, ``angsep``, ``anglemax``, the frame selection and ``k_klip`` appear
    once each, since the wavelength dependence that matters is carried by each group's own lambda.  A
    searched reference library adds its counts once (``nkeep_psfref``, ``nkeep_altroll`` when the
    science count is searched, and ``nkeep_sdi``), seeded at 0 unless ``defaults`` says otherwise, so
    the first evaluation is the plain ADI baseline.  The defaults are the ALES reduction's
    (:data:`BASELINE`), since the NEAR production seed is wrong for these data.

    Parameters
    ----------
    reducer : PartitionedReducer
        The groups' reducer.
    defaults : dict or None, default None
        Defaults over :data:`BASELINE`.
    bin_range : tuple of 2 ints, default (1, 30)
        Range of frames per temporal bin.
    selection : str or None, default "binary"
        Group selection, "binary" for one 0/1 flag per group (``sel_gNN``) or "two_slot" with
        ``max_drop`` slots.
    max_drop : int, default 10
        Number of drop slots of the "two_slot" selection.
    kw : dict
        More keywords for :func:`klip_tpe.instruments.search_space.make_space`, such as
        ``zone_range``, ``search_post_hp`` or ``pin``.

    Returns
    -------
    space : SearchSpace
        The search space.

    Notes
    -----
    The ALES paper reduction of MWC 758 fits one 5-27 spaxel zone and derotates the residuals without
    a second high-pass, in a 0.4-0.8" annulus ``zone_in`` 6.6, ``zone_out`` 3.8 and ``post_hp`` 0.
    ``pin={"zone_in": ..., "zone_out": ..., "n_ang": 6}`` holds that zone and its six sectors fixed,
    and :func:`~klip_tpe.reducer.set_zone_center` ``(red, (30, 30), "getzone2015")`` draws them as the
    paper's 2015 ``getzone`` did.
    """
    d = dict(BASELINE)
    d.update(defaults or {})
    kw.setdefault("per_night", False)
    space = generic.make_space(reducer, bin_range=bin_range, selection=selection, max_drop=max_drop,
                               defaults=d, **kw)
    for p in space.params:
        if p.name in ("nkeep_psfref", "nkeep_sdi"):
            p.default = float(np.clip(float(d.get(p.name, 0)), p.lo, p.hi))
    return space


class GroupCountGuard:
    """Guard that keeps at most ``max_excluded`` groups switched off by the ``sel_*`` flags.

    The binary selection has no bound of its own, and an all-off draw decodes to all on.  A draw with
    more than ``max_excluded`` zeros is projected back by switching groups on again in ``prefer`` order
    until exactly ``max_excluded`` are off, and a random warm-up draw switches on groups drawn at
    random among the excluded, so the random phase stays unbiased.  The projected vector is what is
    evaluated and recorded.

    Parameters
    ----------
    max_excluded : int, default 10
        Maximum number of groups switched off.
    prefer : sequence or None, default None
        Partition identifiers in the order they are switched back on, the most wanted first.  None
        takes the groups nearest ``prefer_m``.
    lam_m : dict or None, default None
        Mean wavelength of each group, in meters, for the default order.
    prefer_m : float, default 3.7e-6
        Wavelength where the companion's S/N per group is highest, in meters.
    random_redraw : bool, default True
        Whether a random draw switches on random groups rather than the preferred ones.

    Attributes
    ----------
    last_restored : int
        Number of groups the last call switched back on.
    """

    def __init__(self, max_excluded: int = 10, prefer: Optional[Sequence[Any]] = None,
                 lam_m: Optional[Dict[Any, float]] = None, prefer_m: float = 3.7e-6, random_redraw: bool = True):
        self.max_excluded = int(max(max_excluded, 0))
        self.prefer_m = float(prefer_m)
        if prefer is None and lam_m:
            prefer = sorted(lam_m, key=lambda n: abs(float(lam_m[n]) - self.prefer_m))
        self.prefer = list(prefer) if prefer is not None else None
        self.random_redraw = bool(random_redraw)
        self.last_restored = 0

    def __call__(self, x: np.ndarray, space: SearchSpace, rng=None, is_random: bool = False, **_) -> np.ndarray:
        """Project a draw onto the bound on excluded groups.

        Parameters
        ----------
        x : ndarray
            The draw, in the search space's coordinates.
        space : SearchSpace
            The search space, whose binary selection dimensions are read.
        rng : numpy.random.Generator or None, default None
            Random generator, for a random draw.
        is_random : bool, default False
            Whether the draw is from the random warm-up.
        _ : dict
            Other keywords of the guard chain, ignored.

        Returns
        -------
        x : ndarray
            The projected draw, a copy when changed.
        """
        self.last_restored = 0
        if getattr(space, "selection", None) != "binary" or not space.partitions:
            return x
        x = np.asarray(x, float).copy()
        dims = {space.params[i].partition: i for i in space.selection_dims}
        off = [pid for pid, i in dims.items() if int(np.clip(round(float(x[i])), 0, 1)) == 0]
        if len(off) == len(dims):
            off = []                     # the all-off draw already decodes to all on
        excess = len(off) - self.max_excluded
        if excess <= 0:
            return x
        if is_random and rng is not None and self.random_redraw:
            order = [off[j] for j in rng.permutation(len(off))]
        else:
            rank = {pid: k for k, pid in enumerate(self.prefer or [])}
            order = sorted(off, key=lambda pid: (rank.get(pid, len(rank)), str(pid)))
        for pid in order[:excess]:
            x[dims[pid]] = 1.0
        self.last_restored = excess
        return x

    def describe(self) -> Dict[str, Any]:
        """Return the guard's settings, for a run's record.

        Returns
        -------
        info : dict
            ``name``, ``max_excluded``, ``prefer``, ``prefer_um`` and ``random_redraw``.
        """
        return {"name": "group_count_guard", "max_excluded": self.max_excluded,
                "prefer": [str(p) for p in (self.prefer or [])], "prefer_um": self.prefer_m * 1e6,
                "random_redraw": self.random_redraw}


def make_guard(reducer: PartitionedReducer, max_excluded: Optional[int] = 10,
               prefer: Optional[Sequence[Any]] = None, prefer_m: float = 3.7e-6,
               n_min_ref: int = 10, ref_frac: float = 0.95, k_max: int = 100,
               frame_selection: Optional[bool] = None, bin_min: int = 1):
    """Build the group-count bound followed by the generic guards, with the representative group's lambda/D and FWHM.

    The group bound goes first, so the reference census sees the groups included.  With a searched
    library whose every pool is searched, the library guard comes last, as in
    :func:`~klip_tpe.instruments.generic.make_guard`.  With the science pool kept whole the census
    guard governs the basis, and ``nkeep_psfref`` runs free from 0, the ADI point, to the pool.

    Parameters
    ----------
    reducer : PartitionedReducer
        The groups' reducer.
    max_excluded : int or None, default 10
        Maximum number of groups switched off (:class:`GroupCountGuard`).  None sets no bound.
    prefer : sequence or None, default None
        Order in which groups are switched back on.
    prefer_m : float, default 3.7e-6
        Wavelength of the preferred groups, in meters.
    n_min_ref : int, default 10
        Minimum number of references per target.
    ref_frac : float, default 0.95
        Fraction of the target frames that must keep ``n_min_ref`` references.
    k_max : int, default 100
        Maximum number of KL modes.
    frame_selection : bool or None, default None
        Whether to snap the frame-selection dimensions.  None does so when every group has quality
        tags.
    bin_min : int, default 1
        Minimum temporal bin.

    Returns
    -------
    guard : callable
        The composed guard.
    """
    lam = {pid: float(r.lam_m) for pid, r in reducer.reducers.items()}
    gc = GroupCountGuard(max_excluded, prefer=prefer, lam_m=lam, prefer_m=prefer_m) if max_excluded is not None else None
    ref = ReferenceCountGuard(angles_fn=reducer.frame_angles, tags_fn=reducer.frame_tags,
                              fwhm_px=reducer.fwhm, lam_over_d_px=reducer.lam_over_d_px,
                              n_min_ref=n_min_ref, ref_frac=ref_frac, k_max=k_max)
    mb = MinBinGuard(bin_min) if int(bin_min) > 1 else None
    if frame_selection is None:
        frame_selection = all(r.frame_tags() for r in reducer.reducers.values())
    fs = FrameSelectionGuard(tags_fn=reducer.frame_tags,
                             nframes_fn=lambda pid: reducer.reducers[pid].data.nframes) if frame_selection else None
    lib = None
    for r in reducer.reducers.values():
        spec = getattr(r, "_reflib_spec", None)
        if spec is not None and getattr(r, "supports_reference_library", True):
            if spec.get("searched") is None:
                lib = ReferenceLibraryGuard(n_min_ref=int(spec.get("n_min_ref", 2)), k_max=k_max)
            break
    return compose(*[p for p in (gc, mb, fs, ref, lib) if p is not None])


def default_config(reducer: PartitionedReducer, clean_subtract: bool = True, known=(),
                   forbidden_pa=(), metric: str = "mawet", **metric_kw):
    """Build the Mawet objective with the representative group's FWHM and matched filter, and the sampler.

    It is :func:`~klip_tpe.instruments.generic.default_config` with the metric's despiking off
    (``declip_nsig=None``, which ``metric_kw`` may set back).  ALES samples the PSF at 2.1 to 3.0
    spaxels per FWHM, and a point source's peak stands far enough above its neighbors that the
    despiking would cut a companion's peak and its S/N.

    Parameters
    ----------
    reducer : PartitionedReducer
        The groups' reducer.
    clean_subtract : bool, default True
        Whether the objective subtracts the clean image's score.
    known : sequence of tuple of 2 floats, default ()
        Companions, as ``(rho_arcsec, pa_deg)``, kept out of the noise rings and away from the
        injections.
    forbidden_pa : sequence of tuple of 2 floats, default ()
        Wedges the injections keep out of, as ``(center_pa_deg, half_width_deg)``, such as
        ``[(PA, 25.0)]`` around a companion at ``PA``.
    metric : {"mawet", "fmmf"}, default "mawet"
        The metric.
    metric_kw : dict
        More keywords for the metric, such as ``disk_mask`` or ``pixel_mask``.

    Returns
    -------
    objective : Objective
        The objective.
    sampler : PositionSampler
        The sampler of injection positions.
    """
    kw = dict(declip_nsig=None)
    kw.update(metric_kw)
    return generic.default_config(reducer, clean_subtract=clean_subtract, known=known,
                                  forbidden_pa=forbidden_pa, metric=metric, **kw)
