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
``max_excluded`` of them (:class:`GroupCountGuard`).

The first data set was WISPIT 2 (UT 2026-06-05, DX aperture, 535 frames x 99 channels,
62 x 62 spaxels), described in the ALES reduction's handoff of 2026-10-13::

    from klip_tpe.instruments import ales
    datasets, lam_m, fwhm_px, table = ales.load_ales(PRODUCTS, "WISPIT2")
    red = ales.make_reducer(datasets, lam_m, fwhm_px, max_workers=4)
    space = ales.make_space(red)                       # shared block + sel_g00 .. sel_g19
    space.project = ales.make_guard(red, max_excluded=10)
    objective, sampler = ales.default_config(red, known=[ales.WISPIT2B],
                                             forbidden_pa=[(ales.WISPIT2B[1], 25.0)])

or on the command line, ``klip-tpe near --instrument ales --root PRODUCTS ...``.

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

from ..feasibility import FrameSelectionGuard, MinBinGuard, ReferenceCountGuard, compose
from ..reducer import Dataset, PartitionedReducer
from ..space import SearchSpace
from . import generic

__all__ = ["ALES", "WISPIT2B", "BASELINE", "group_table", "group_mean", "load_groups", "load_ales",
           "make_reducer", "make_space", "make_guard", "default_config", "GroupCountGuard",
           "product_paths", "representative_group"]


class ALES:
    """Instrument constants (the ALES reduction README's assumptions)."""

    pxscale = 0.0345            # arcsec / spaxel, ASSUMED (MWC 758 value; not calibrated on 2026-06-05)
    diam_m = 8.4                # one LBT aperture (DX only, no Fizeau)
    fwhm_lod = 1.03             # measured broadband FWHM / (lambda/D) at 3.5 um
    angle_offset = 180.0        # derotation = CCW by LBT_PARA + 180 deg
    texp_frame = 4.9176         # s per frame (5 x 0.9835 s ramp intervals)
    c0, c1 = 6, 88              # usable channels (2.899-4.175 um)
    ngrp = 20                   # groups over them (R_group = 54; 15 = one R = 40 element each)
    tag_r_in = 2.0              # quality tags from r >= 2 spaxels (r >= 6 is background-dominated there)


#: WISPIT 2b as the ALES reduction places it in these frames: ``(rho_arcsec, pa_deg)`` at the
#: assumed plate scale, 8.663 spaxels.  A sign/offset check of the derotation only, and the
#: ``known`` source its noise rings leave out.
WISPIT2B = (8.663344937255616 * ALES.pxscale, 241.46154214127014)

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
    frames so an 814 MB cube is never expanded whole (a memory-mapped FITS stays on disk)."""
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
                source: Optional[str] = None, log: Callable[[str], None] = print
                ) -> Tuple[Dict[str, Dataset], Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    """Wavelength groups of an IFS cube as :class:`~klip_tpe.reducer.Dataset` partitions.

    ``cube``: ``(nwave, nframe, ny, nx)`` (astropy's reading of the IDL-order products; a
    memory map is fine), or None when ``cached`` holds every group; ``angles``: one per
    frame, the LBT parallactic angle, to which ``angle_sign`` and ``angle_offset`` are
    applied once; ``psf``: ``(nwave, ny', nx')`` template on its own grid, meaned over each
    group's channels and centred at ``psf_center`` (the 5 x 5 centroid of its peak when
    None); ``ref_cube``: a reference star's ``(nwave, nref, ny, nx)`` cube, meaned the same
    way and attached per group (RDI/ARDI are then on by default in :func:`make_reducer`).
    ``groups`` overrides the table :func:`group_table` would build; ``only`` keeps the named
    groups; ``cached`` / ``ref_cached`` give ready group cubes (arrays or ``.npy`` paths) by
    group name.

    Returns ``(datasets, lam_m, fwhm_px, table)``: the partitions, their mean wavelengths
    (m) and FWHM (px, ``fwhm_lod`` lambda/D), and the group table with ``fwhm_px`` and
    ``n_nan`` (NaN pixels left in the group's cube) added to each group.
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
            tpl = np.where(np.isfinite(tpl), tpl, 0.0)          # 50 all-NaN pixels in WISPIT 2's template
        rg = None
        if ref_cube is not None or (ref_cached and name in ref_cached):
            rg = _group_cube(ref_cube, ref_cached, name, ch)
        meta = {"provenance": {"instrument": "ALES", "group": name, "channels": list(ch),
                               "lam_um": [float(w[c]) for c in ch], "lam_mean_um": g["lam_mean"],
                               "lam_edges_um": [g["lam_lo"], g["lam_hi"]], "nframes": int(cg.shape[0]),
                               "source": source, "angle_offset_deg": float(angle_offset),
                               "angle_sign": float(angle_sign)}}
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
        log(f"  {name}: channels {ch[0]}-{ch[-1]} ({len(ch)}), {g['lam_lo']:.4f}-{g['lam_hi']:.4f} um, "
            f"mean {g['lam_mean']:.3f} um, fwhm {fwhm_px[name]:.2f} px, {ds.cube.shape[0]} frames, "
            f"{g['n_nan']} NaN px" + (f", {rg.shape[0]} reference frames" if rg is not None else ""))
    return datasets, lam_m, fwhm_px, table


def _fits(path: str, memmap: bool = True):
    from astropy.io import fits
    h = fits.open(path, memmap=memmap)
    return h[0].data, h[0].header


def load_ales(products_dir: str, target: str = "WISPIT2", ref: Optional[str] = None, *,
              c0: int = ALES.c0, c1: int = ALES.c1, ngrp: int = ALES.ngrp, only: Optional[Sequence[str]] = None,
              pxscale: float = ALES.pxscale, diam_m: float = ALES.diam_m, fwhm_lod: float = ALES.fwhm_lod,
              angle_offset: float = ALES.angle_offset, angle_sign: float = 1.0, tags: Any = "auto",
              tag_r_in: float = ALES.tag_r_in, nan_policy: str = "keep", cache: Optional[str] = None,
              log: Callable[[str], None] = print
              ) -> Tuple[Dict[str, Dataset], Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    """:func:`load_groups` on the reduction's product files (:func:`product_paths`).

    ``ref`` names a reference star whose products sit beside the target's (``"HIP91748"``)
    and attaches its group means as each group's RDI library.  ``cache`` is a directory:
    the group cubes are written there as ``.npy`` the first time and read back after,
    which matters when the products live on a slow volume (the full cube is 814 MB and a
    group mean reads all of it).  The template's star is at the header's ``STARX``/``STARY``
    when it has them, else at the centroid of its peak.
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
    rcube = None
    if ref is not None and not all(n in rcached for n in names):
        rcube = _fits(product_paths(products_dir, ref)["cube"])[0]
    log(f"ALES {target}: {w.size} channels, {a.size} frames; {len(table)} groups over channels {c0}-{c1} "
        f"({w[c0]:.4f}-{w[c1]:.4f} um)" + (f"; reference star {ref}" if ref else "")
        + (f"; {len(cached)} group cubes from the cache" if cached else ""))
    datasets, lam_m, fwhm_px, table = load_groups(
        cube, a, w, psf, c0=c0, c1=c1, ngrp=ngrp, groups=table, only=only, pxscale=pxscale, diam_m=diam_m,
        fwhm_lod=fwhm_lod, angle_offset=angle_offset, angle_sign=angle_sign, tags=tags, tag_r_in=tag_r_in,
        nan_policy=nan_policy, ref_cube=rcube, psf_center=center, cached=cached, ref_cached=rcached,
        source=paths["cube"], log=log)
    if cache:
        os.makedirs(cache, exist_ok=True)
        for g in table:
            f = os.path.join(cache, f"{key}_{g['name']}.npy")
            if not os.path.exists(f):
                np.save(f, datasets[g["name"]].cube)
            if ref is not None and datasets[g["name"]].ref_cube is not None:
                rf = os.path.join(cache, f"{rkey}_{g['name']}.npy")
                if not os.path.exists(rf):
                    np.save(rf, datasets[g["name"]].ref_cube)
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
    (rows and columns the registration shifted in, all beyond 27 spaxels), and the IDL
    boxcar carries a NaN along the whole frame: at bin 5 it emptied 40 of WISPIT 2's 108
    bins, and b's matched-filter S/N on the reduction's own parameters fell from 6.2 to 5.3
    (to 3.5 with the de-spiking below as well).  The ALES reduction's own high-pass is
    NaN-aware too."""
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
    return red


def make_space(reducer: PartitionedReducer, defaults: Optional[Dict[str, Any]] = None,
               bin_range: Tuple[int, int] = (1, 30), selection: Optional[str] = "binary",
               max_drop: int = 10, **kw) -> SearchSpace:
    """One parameter block shared by every group plus the group flags.

    ``bin``, ``n_ang``, ``filter``, ``angsep``, ``anglemax``, the frame-selection dims and
    ``k_klip`` once each (``per_night=False``: the wavelength dependence that matters is
    carried by each group's own lambda), and ``selection='binary'`` adds one 0/1 flag per
    group, ``sel_gNN``.  ``'two_slot'`` with ``max_drop`` slots is the other bounded form.
    The defaults are the ALES reduction's baseline (:data:`BASELINE`); the NEAR production
    seed (bin 29, corr_thresh 0.907) is wrong for these data."""
    d = dict(BASELINE)
    d.update(defaults or {})
    kw.setdefault("per_night", False)
    return generic.make_space(reducer, bin_range=bin_range, selection=selection, max_drop=max_drop,
                              defaults=d, **kw)


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
    group bound goes first so the reference census sees the groups that are included."""
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
    return compose(*[p for p in (gc, mb, fs, ref) if p is not None])


def default_config(reducer: PartitionedReducer, clean_subtract: bool = True, known=(),
                   forbidden_pa=(), metric: str = "mawet", **metric_kw):
    """:func:`generic.default_config`: the Mawet objective with the representative group's
    FWHM and matched filter, and the one sampler.  ``known`` keeps the companions out of the
    noise rings and the injections away from them; ``forbidden_pa`` keeps the injections out
    of PA wedges (WISPIT 2b: ``known=[WISPIT2B], forbidden_pa=[(WISPIT2B[1], 25.0)]``, so no
    source lands within 25 deg of it in azimuth).

    The metric's de-spiking is off here (``declip_nsig=None``, which ``metric_kw`` may set
    back).  It replaces a pixel that stands more than 6 robust sigma above its 3 x 3 median
    by that median, which on NEAR's 6.2 px FWHM never touches a source.  ALES samples the
    PSF at 2.1-3.0 spaxels per FWHM, and a point source's peak stands that far above its
    neighbours: on the baseline image it cut WISPIT 2b's matched-filter S/N from 6.2 to 3.9.
    The reduction has already repaired its bad spaxels and 5 sigma temporal outliers.
    """
    kw = dict(declip_nsig=None)
    kw.update(metric_kw)
    return generic.default_config(reducer, clean_subtract=clean_subtract, known=known,
                                  forbidden_pa=forbidden_pa, metric=metric, **kw)
