"""LBTI/ALES adapter: an integral-field cube as wavelength groups, one partition each.

ALES (the LMIRCam lenslet spectrograph, L band) delivers a ``(wavelength, frame, y, x)``
cube of 99 channels at R ~ 40-55, so neighbouring channels share a resolution element and a
channel is not the unit to optimize over.  The adapter cuts the usable channels into
``ngrp`` groups of equal width in ``ln lambda`` (about one resolution element each), means
each group's channels into one ``(frame, y, x)`` cube, and hands the groups to
:class:`~klip_tpe.reducer.PartitionedReducer` as partitions, the way nights or rolls are.
Each group carries its own wavelength (so ``angsep`` in lambda/D and the Mawet apertures
are right for it), its own PSF template (the group mean of the per-channel template) and,
when asked, its own reference-star library.  One parameter block is shared by every group,
and a 0/1 flag per group (``sel_gNN``) lets the optimizer leave groups out, at most
``max_excluded`` of them (:class:`GroupCountGuard`).  The products are those the ALES
reduction's handoff of 2026-10-13 describes.  From Python::

    from klip_tpe.instruments import ales
    datasets, lam_m, fwhm_px, table = ales.load_ales(PRODUCTS, TARGET)
    red = ales.make_reducer(datasets, lam_m, fwhm_px, max_workers=4)
    space = ales.make_space(red)                       # shared block + sel_g00 .. sel_g19
    space.project = ales.make_guard(red, max_excluded=10)
    objective, sampler = ales.default_config(red, known=[(RHO, PA)], forbidden_pa=[(PA, 25.0)])

or on the command line, ``klip-tpe near --instrument ales --root PRODUCTS ...``.

A reference star (``load_ales(..., ref="HIP91748")``) is a fixed ARDI library by default.
With ``make_reducer(..., reflib=True)`` (``--ales-reflib``) it is a searched one: each target
frame's basis is the science frames the ADI rule admits for it plus the ``nkeep_psfref``
reference means most correlated with it in the annulus, with ``nkeep_psfref`` a dimension of
the search from 0, plain ADI to the bit, to the whole pool.  The reference frames are taken
in consecutive means of ``ref_bin`` (``--ales-ref-bin``, 5 by default) so the count means one
thing whatever the science ``bin`` is.

Conventions (ALES reduction README, section 4):

* the products are north up, east left once every frame is rotated counter-clockwise by
  ``LBT_PARA + 180`` deg; the adapter adds the 180 to the angle array once
  (``angle_offset``) and keeps ``truenorth = 0``;
* the star sits at the array centre ``((nx - 1) / 2, (ny - 1) / 2)`` of the science cube and
  on the integer pixel ``(STARX, STARY)`` of the template's larger grid;
* 0.0345"/spaxel and the 180 deg are assumed, not calibrated on the night, and WISPIT 2b's
  position must not be used to calibrate either (its orbital motion since the published
  epoch is unknown);
* the template is the median of the registered science frames themselves, in the frames'
  units, so a contrast is a plain flux ratio to the star in that group.
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
    """Instrument constants (the ALES reduction README's assumptions)."""

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
    """The reduction's product files for ``target`` (``<target>_cube_cen_clean.fits`` ...)."""
    p = lambda s: os.path.join(products_dir, f"{target}_{s}")
    return {"cube": p("cube_cen_clean.fits"), "angles": p("primary_angles_clean.fits"),
            "waves": p("waves.fits"), "psf": p("star_psf_cube.fits"),
            "frames": p("frame_numbers_clean.fits"), "frametags": p("frametags.csv")}


def group_table(waves_um: Sequence[float], c0: int = ALES.c0, c1: int = ALES.c1,
                ngrp: int = ALES.ngrp, name_fmt: str = "g{:02d}") -> List[Dict[str, Any]]:
    """Constant-R wavelength groups over channels ``c0..c1`` (inclusive).

    A channel's edges are the midpoints to its neighbours; the group edges are equally
    spaced in ``ln lambda`` between the lower edge of ``c0`` and the upper edge of ``c1``,
    and a channel belongs to the group whose edges contain its wavelength (the handoff's
    rule, section 6.2).  Returns one dict per group: ``name``, ``channels`` (channel
    indices), ``lam_lo``, ``lam_hi`` (the group's edges, um), ``lam_mean`` (mean wavelength
    of its channels, um), ``n`` and ``slot`` (its index among the ``ngrp`` slots).  A slot
    that receives no channel is left out and the names close up.
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
    """``nanmean`` of a ``(wavelength, frame, y, x)`` cube over ``channels``, in blocks of
    frames so the full cube is never expanded whole (a memory-mapped FITS stays on disk)."""
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
    return float(fwhm_lod * (lam_m / diam_m) * 206265.0 / pxscale)


def reference_means(cube: np.ndarray, n: int, blocks: Optional[Sequence[Any]] = None,
                    dtype=np.float32) -> np.ndarray:
    """``nanmean`` of consecutive runs of ``n`` frames of a ``(frame, y, x)`` cube, within
    each observing block when ``blocks`` (one label per frame) is given, the frames a block
    has left over at its end dropped.  Blocks of 23, 41 and 37 frames in fives make
    4 + 8 + 7 = 19 means, and no mean straddles a gap.  Without ``blocks`` the whole sequence
    is one block.

    The reference star's frames are not derotated, so there is no angle to close a run
    early on, and a fixed run gives the searched count ``nkeep_psfref`` one meaning for the
    whole search: the ``bin`` of the science frames may move, the reference means do not.
    ``n <= 1`` returns the cube itself."""
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
    """The ``block`` column of a reduction's ``<target>_frametags.csv`` (one observing block
    label per clean frame, written by its ``post08_frametags.py``), or None when the file or
    the column is missing."""
    import csv
    if not frametags_csv or not os.path.exists(frametags_csv):
        return None
    with open(frametags_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "block" not in rows[0]:
        return None
    return np.array([str(r["block"]) for r in rows])


def _group_cube(cube, cached: Optional[Dict[str, Any]], name: str, channels: Sequence[int]) -> np.ndarray:
    """A group's ``(frame, y, x)`` cube: from ``cached`` (an array or a ``.npy`` path per
    group) when it holds the group, else the mean of the cube's channels."""
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
    """Wavelength groups of an IFS cube as :class:`~klip_tpe.reducer.Dataset` partitions.

    ``cube``: ``(nwave, nframe, ny, nx)`` (astropy's reading of the IDL-order products; a
    memory map is fine), or None when ``cached`` holds every group; ``angles``: one per
    frame, the LBT parallactic angle, to which ``angle_sign`` and ``angle_offset`` are
    applied once; ``psf``: ``(nwave, ny', nx')`` template on its own grid, meaned over each
    group's channels and centred at ``psf_center`` (the 5 x 5 centroid of its peak when
    None); ``ref_cube``: a reference star's ``(nwave, nref, ny, nx)`` cube, meaned the same
    way and attached per group (RDI/ARDI are then on by default in :func:`make_reducer`),
    its frames taken in consecutive means of ``ref_bin`` first (:func:`reference_means`,
    within the observing blocks ``ref_blocks`` labels, one per reference frame; 1 = the
    frames as they are).  ``groups`` overrides the table :func:`group_table` would
    build; ``only`` keeps the named groups; ``cached`` / ``ref_cached`` give ready group
    cubes (arrays or ``.npy`` paths) by group name, the reference ones unbinned.

    Returns ``(datasets, lam_m, fwhm_px, table)``: the partitions, their mean wavelengths
    (m) and FWHM (px, ``fwhm_lod`` lambda/D), and the group table with ``fwhm_px`` and
    ``n_nan`` (NaN pixels left in the group's cube) added to each group.  Each dataset's
    ``meta['provenance']`` records the group, and with a reference star ``ref_source``,
    ``ref_frames`` (its frames), ``ref_blocks`` (frames per observing block), ``ref_bin``
    and ``n_ref`` (the means attached).
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
    """The reference cube's frame count: from a cached group mean when there is one, else
    from the FITS header (no data read)."""
    for n in names:
        f = rcached.get(n)
        if f:
            return int(np.load(f, mmap_mode="r").shape[0])
    from astropy.io import fits
    with fits.open(rpaths["cube"], memmap=True) as h:
        return int(h[0].header.get("NAXIS3", h[0].data.shape[1]))


def _fits(path: str, memmap: bool = True):
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
    """:func:`load_groups` on the reduction's product files (:func:`product_paths`).

    ``ref`` names a reference star whose products sit beside the target's (``"HIP91748"``)
    and attaches its group means as each group's RDI library, in consecutive means of
    ``ref_bin`` frames (:func:`reference_means`; 1 = every frame).  ``cache`` is a
    directory: the group cubes are written there as ``.npy`` the first time and read back
    after, which matters when the products live on a slow volume (a group mean reads the
    whole cube); the reference star's group means are cached
    unbinned, so ``ref_bin`` may change between runs.  The means never straddle an
    observing block: the blocks come from the ``block`` column of ``<ref>_frametags.csv``
    beside the products (:func:`frame_blocks`), the whole sequence is one block without it.
    The template's star is at the header's ``STARX``/``STARY`` when it has them, else at
    the centroid of its peak.
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
    """The nights of a data set in the release layout, as the dates of its ``meta_<date>``
    roles, in date order."""
    return sorted(str(k)[5:] for k in files if str(k).startswith("meta_"))


def load_release(files: Dict[str, str], nights: Optional[Sequence[str]] = None, only: Optional[Sequence[str]] = None,
                 labels: Optional[Dict[str, str]] = None, *, pxscale: Optional[float] = None,
                 diam_m: float = ALES.diam_m, fwhm_lod: float = ALES.fwhm_lod, angle_offset: Optional[float] = None,
                 angle_sign: float = 1.0, tags: Any = "auto", tag_r_in: float = ALES.tag_r_in,
                 nan_policy: str = "keep", log: Callable[[str], None] = print
                 ) -> Tuple[Dict[str, Dataset], Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    """An ALES data set in the release layout as partitions, one per night and wavelength group.

    The layout is the one :func:`klip_tpe.datasets.fetch` returns for ``"lbti_ales_mwc758"``:
    for each night a ``meta_<date>`` file (its primary HDU one template per group, then the
    ``ANGLES``, ``FRAMES``, ``GROUPS`` and ``WAVES`` extensions) and one ``gNN_<date>`` cube per
    group, ``(frame, y, x)``, the mean of that group's channels.  Every night shares one group
    grid (the meta header's ``GRIDLO``, ``GRIDHI``, ``NGROUP``), so ``g07`` covers the same
    wavelengths on each night while its mean wavelength follows that night's channels.

    ``nights``: the nights to keep, as dates or labels (default all of them).  They are
    labelled ``n1``, ``n2``, ... in the date order of the whole data set, so a night loaded alone
    keeps its label, unless ``labels`` maps a date to a label.  ``only``: the groups to keep, the
    same on every night.  The partitions are ``<label>_<group>`` (``n1_g07``).  The plate
    scale and the angle convention come from the meta header (``PIXSCALE``, ``ANGCONV`` =
    ``LBT_PARA + 180``) unless ``pxscale`` or ``angle_offset`` is given.  Each template is the
    night's own (the star is at the header's ``STARCEN``, the array centre), so a contrast is a
    flux ratio to the star in that group on that night.

    Returns ``(datasets, lam_m, fwhm_px, table)`` as :func:`load_groups` does, the table with
    one row per partition (``name``, ``night``, ``date``, ``group``, ``channels``, ``lam_lo``,
    ``lam_hi``, ``lam_mean``, ``n``, ``fwhm_px``, ``n_nan``)."""
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
    """The group whose wavelength is nearest ``at_m`` (default: the mean of the groups'
    wavelengths): it stands for the combined image's FWHM and matched filter."""
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
    """:func:`generic.make_reducer` over the groups, with the group at the mean wavelength
    standing for the combination (:func:`representative_group`): the metric's FWHM and
    matched filter and the sampler's floor come from it, not from the bluest group.  The
    template is normalised in ``ee_radius_px`` (3 spaxels, the reduction's aperture), so
    ``flux_unit`` is the star's flux there in that group and a contrast is a flux ratio.
    ``table`` (from :func:`load_groups`) is recorded as the reducer's provenance, which the
    run's setup file carries.

    The high-pass of the science frames is NaN-aware here (``hp_nan_aware``, which
    ``defaults`` may override).  The registered ALES frames carry NaN along their edges
    (rows and columns the registration shifted in), and the IDL boxcar carries a NaN along
    the whole frame, so every bin holding such a frame would come out NaN and be dropped.
    The ALES reduction's own high-pass is NaN-aware too.

    ``reflib`` makes the reference star a searched library
    (:meth:`~klip_tpe.reducer.KLIPReducer.set_reference_library`, one per group, the
    ``rotation`` form): every target frame's basis is the science frames the ADI rule
    admits for it (``angsep``, ``anglemax``, per annulus, as without a library) plus the
    ``nkeep_psfref`` reference means most correlated with it in the annulus
    (``reflib_metric``), ``nkeep_psfref`` searched from 0, which is plain ADI to the bit,
    to the whole pool.  ``reflib_science='searched'`` searches the science count too
    (``nkeep_altroll``, the NIRCam form, whose zero is pure RDI); ``'all'`` keeps every
    admitted science frame.  ``reflib_n_min_ref`` is the library's own floor on a target's
    references.  Without ``reflib`` a reference star is the fixed ARDI library it was.

    ``sdi`` adds the other groups' frames to every group's library as an SDI pool: each
    target frame's basis may take up to ``nkeep_sdi`` (searched from 0, which is no SDI, to
    ``sdi_cap``) of the other groups' frames of the same evaluation, prepared as the group
    prepares its own and magnified about the star to its wavelength, the ones most correlated
    with it among those the displacement rule admits (the planet moved by rotation and
    rescaling together at least ``angsep``, within ``anglemax``).  ``sdi_lenders`` says which
    groups lend frames: ``'all'`` (an excluded group still does) or ``'selected'``.  A
    reference star given (``load_ales(ref=...)``) is the library's third pool then, searched,
    whether or not ``reflib`` is set.  ``nkeep_sdi = nkeep_psfref = 0`` is plain ADI to the
    bit, and costs plain ADI: the other groups are prepared only when ``nkeep_sdi`` asks.

    ``spectrum`` gives the injected sources a spectrum
    (:meth:`~klip_tpe.reducer.PartitionedReducer.set_injection_spectrum`): a callable
    ``f(lambda_um)``, a ``(lambda_um, contrast)`` table or ``{partition: factor}``, the relative
    contrast to the star, so a source of contrast ``c`` goes into each group at ``c`` times
    the spectrum there over the spectrum at ``spectrum_norm`` (default the representative
    group; a partition id or a wavelength in metres).  None, the default, injects the same
    contrast in every group: a source with the star's colours.  ``spectrum_label`` names it in
    the run's setup file."""
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
    """One parameter block shared by every group plus the group flags.

    ``bin``, ``n_ang``, ``filter``, ``angsep``, ``anglemax``, the frame-selection dims and
    ``k_klip`` once each (``per_night=False``: the wavelength dependence that matters is
    carried by each group's own lambda), and ``selection='binary'`` adds one 0/1 flag per
    group, ``sel_gNN``.  ``'two_slot'`` with ``max_drop`` slots is the other bounded form.
    A reducer with a searched reference library (``reflib`` of :func:`make_reducer`) adds
    its counts once, ``nkeep_psfref`` (and ``nkeep_altroll`` when the science count is
    searched), and ``nkeep_sdi`` with an SDI pool, shared by every group like the rest of the
    block.  ``nkeep_psfref`` and ``nkeep_sdi`` seed at 0 unless ``defaults`` says otherwise:
    the search's first evaluation is then the plain ADI baseline itself, and its score the one
    every library draw is read against.
    The defaults are the ALES reduction's baseline (:data:`BASELINE`); the NEAR production
    seed (bin 29, corr_thresh 0.907) is wrong for these data.

    ``zone_range=(lo, hi)`` and ``search_post_hp=True`` (:func:`klip_tpe.instruments.near.make_space`)
    search the KLIP zone's reach beyond the annulus (``zone_in``, ``zone_out``, px) and the
    residuals' second high-pass (``post_hp``).  The ALES paper reduction of MWC 758 fits
    one 5-27 spaxel zone and derotates the residuals without a second high-pass: in a
    0.4-0.8" annulus that is ``zone_in`` 6.6, ``zone_out`` 3.8 (the 62-spaxel frames cap the
    zone at 27 anyway) and ``post_hp`` 0.  ``pin={"zone_in": ..., "zone_out": ..., "n_ang": 6}``
    holds that zone and its six sectors fixed instead, and
    :func:`~klip_tpe.reducer.set_zone_center` ``(red, (30, 30), "getzone2015")`` draws them as the
    paper's 2015 ``getzone`` did, the same pixels in the same sectors."""
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
    """At most ``max_excluded`` groups excluded by the ``sel_*`` flags.

    The binary selection has no bound of its own: a proposal may switch off any number of
    groups, and an all-off draw silently decodes to all on.  A draw with more than
    ``max_excluded`` zeros is projected back by switching groups on again, in ``prefer``
    order (a list of partition ids, the most wanted first; by default the groups nearest
    ``prefer_m``, where the companion's S/N per group is highest), until exactly
    ``max_excluded`` are off.  A random (warm-up) draw instead switches on groups drawn at
    random among the excluded ones, so the random phase stays unbiased, as the
    reference-count guard does for ``angsep``.  The projected vector is what gets
    evaluated and recorded, so ``results.jsonl`` never holds a selection past the bound.
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
        return {"name": "group_count_guard", "max_excluded": self.max_excluded,
                "prefer": [str(p) for p in (self.prefer or [])], "prefer_um": self.prefer_m * 1e6,
                "random_redraw": self.random_redraw}


def make_guard(reducer: PartitionedReducer, max_excluded: Optional[int] = 10,
               prefer: Optional[Sequence[Any]] = None, prefer_m: float = 3.7e-6,
               n_min_ref: int = 10, ref_frac: float = 0.95, k_max: int = 100,
               frame_selection: Optional[bool] = None, bin_min: int = 1):
    """The group-count bound, then the generic guards (bin floor, frame-selection snapping,
    reference count), the last with the representative group's lambda/D and FWHM.  The
    group bound goes first so the reference census sees the groups that are included.

    With a searched reference library whose every pool is searched (``reflib_science=
    'searched'`` of :func:`make_reducer`) the library guard comes last, as in
    :func:`generic.make_guard`: the searched counts are then the basis size, so their sum
    is floored and ``k_klip`` capped by it.  With the science pool kept whole (``'all'``)
    the basis is the angular census plus ``nkeep_psfref``, the census guard already
    governs it, and ``nkeep_psfref`` runs free from 0, the ADI point, to the pool."""
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
    """:func:`generic.default_config`: the Mawet objective with the representative group's
    FWHM and matched filter, and the one sampler.  ``known`` keeps the companions out of the
    noise rings and the injections away from them; ``forbidden_pa`` keeps the injections out
    of PA wedges (a companion at ``(RHO, PA)``: ``known=[(RHO, PA)], forbidden_pa=[(PA, 25.0)]``,
    so no source lands within 25 deg of it in azimuth).  ``disk_mask`` (True = disk, the
    reduced frame's shape; :func:`klip_tpe.metrics.disk_mask_from_image`) keeps a disk of any
    shape out of the noise and the injections off it (see :func:`generic.default_config`).

    The metric's de-spiking is off here (``declip_nsig=None``, which ``metric_kw`` may set
    back).  It replaces a pixel that stands more than 6 robust sigma above its 3 x 3 median
    by that median, which on NEAR's 6.2 px FWHM never touches a source.  ALES samples the
    PSF at 2.1-3.0 spaxels per FWHM, and a point source's peak stands that far above its
    neighbours, so the de-spiking would cut a companion's peak and its S/N with it.
    The reduction has already repaired its bad spaxels and 5 sigma temporal outliers.
    """
    kw = dict(declip_nsig=None)
    kw.update(metric_kw)
    return generic.default_config(reducer, clean_subtract=clean_subtract, known=known,
                                  forbidden_pa=forbidden_pa, metric=metric, **kw)
