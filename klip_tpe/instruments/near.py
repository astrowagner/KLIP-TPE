"""VLT/VISIR NEAR (AGPM N-band) adapter -- the data layout and injection model of
``reduce_near_2.pro`` / ``near2_n4psf.pro``, wired onto the generic
:class:`~klip_tpe.reducer.KLIPReducer`.

Per night ``n<seq>/`` under ``root``::

    AB_cube_skysub_cen_clean.fits   [nf, 432, 432]  cleaned, centred frames (star at 215.5, 215.5)
    AB_parang_clean.sav             angles (deg)
    AB_frametags.sav                corrs, noises, coronoise      (optional)
    texp.sav                        texp (s)                      (optional)
    PSF_ACenB.fits                  16x16 B-star template, centre (7.5, 7.5)
    AB_cube_refstar_clean.fits      pre-registered PSF-reference-star frames (optional, RDI/ARDI)
root/psflib/n4_psf_cube_EEnorm.fits    measured off-axis AGPM PSF library [14, 41, 41]
root/psflib/throughput_merged.csv      EE(1 lambda/D) throughput vs separation

Frames are cropped to 150x150 (``+-75`` px about the star) at load, as the IDL
reduction does before KLIP; all separations here are in the cropped frame.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..feasibility import FrameSelectionGuard, ReferenceCountGuard, compose
from ..injection import LibraryPSF, TemplatePSF, shift_bilinear
from ..metrics import MawetPeakSNR, Objective
from ..positions import PositionSampler
from ..klip import rotate_ccw
from ..reducer import Dataset, KLIPReducer, PartitionedReducer, reducer_class
from ..space import Param, SearchSpace, kgrid

__all__ = ["NEAR", "load_night", "N4Library", "make_reducer", "make_space", "make_guard",
           "default_config", "production_defaults", "legacy_masks"]


class NEAR:
    """Instrument constants."""
    pxscale = 0.0456                 # arcsec / px
    lam_klip = 11.25e-6              # m, used for lambda/D (angsep unit, binning rule)
    lam_metric = 11.0e-6             # m, the optimizer's FWHM (6.24 px)
    diam = 8.2                       # m
    truenorth = 36.5                 # deg, added to the parallactic angle at derotation
    hsize = 216                      # full-frame half size (432 px)
    shsize = 75                      # KLIP crop half size (150 px)
    n4_refpa = -93.9                 # deg, distortion axis of the PSF library
    b_to_a = 129.0 / 57.0            # contrast is relative to alpha Cen A; template is B
    lam_over_d_px = (lam_klip / diam) * 206265.0 / pxscale          # 6.206
    fwhm_px = 1.028 * (lam_metric / diam) * 206265.0 / pxscale       # 6.238
    fwhm_bin_px = 1.028 * lam_over_d_px                              # 6.380


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------
def _readsav(path: str) -> Dict[str, Any]:
    from scipy.io import readsav
    return readsav(path)


def load_night(root: str, seq: int, crop_half: int = NEAR.shsize, frames: Optional[slice] = None,
               cube_name: Optional[str] = None) -> Dataset:
    """Load one night as a cropped :class:`Dataset` (float32).  ``frames`` may
    subsample the sequence for quick tests.  ``cube_name`` defaults to the full
    cleaned cube, falling back to ``AB_cube_crop150.fits`` (a pre-cropped copy)."""
    from astropy.io import fits
    d = os.path.join(root, f"n{seq}")
    if cube_name is None:
        cube_name = "AB_cube_skysub_cen_clean.fits"
        if not os.path.exists(os.path.join(d, cube_name)) and os.path.exists(os.path.join(d, "AB_cube_crop150.fits")):
            cube_name = "AB_cube_crop150.fits"      # pre-cropped copy (crop150.py)
    with fits.open(os.path.join(d, cube_name), memmap=True) as h:
        data = h[0].data
        n = data.shape[0]
        sl = slice(0, n) if frames is None else frames
        ny, nx = data.shape[1], data.shape[2]
        cy0, cx0 = ny // 2, nx // 2
        y0, y1 = cy0 - crop_half, cy0 + crop_half
        x0, x1 = cx0 - crop_half, cx0 + crop_half
        cube = np.ascontiguousarray(data[sl, y0:y1, x0:x1]).astype(np.float32)
    angles = np.asarray(_readsav(os.path.join(d, "AB_parang_clean.sav"))["angles"], float)[sl]
    tags = None
    tf = os.path.join(d, "AB_frametags.sav")
    if os.path.exists(tf):
        t = _readsav(tf)
        tags = {k: np.asarray(t[k], float)[sl] for k in ("corrs", "noises", "coronoise") if k in t}
        if tags.get("corrs") is None or tags["corrs"].size != cube.shape[0]:
            tags = None
    texp = 0.0
    xf = os.path.join(d, "texp.sav")
    if os.path.exists(xf):
        texp = float(np.asarray(_readsav(xf)["texp"]).ravel()[0])
        if frames is not None:
            texp *= cube.shape[0] / max(n, 1)
    psf = None
    pf = os.path.join(d, "PSF_ACenB.fits")
    if os.path.exists(pf):
        psf = np.asarray(fits.getdata(pf), float)
    ref_cube = None
    rf = os.path.join(d, "AB_cube_refstar_clean.fits")
    if os.path.exists(rf):
        # RDI / ARDI reference star (reduce_near_2 L1860-1866): must be at least the science
        # size; centre-cropped like the science cube.  Never frame-subsampled.
        with fits.open(rf, memmap=True) as h:
            rd = h[0].data
            if rd is not None and rd.ndim == 3 and rd.shape[1] >= 2 * crop_half and rd.shape[2] >= 2 * crop_half:
                ry0, rx0 = rd.shape[1] // 2, rd.shape[2] // 2
                ref_cube = np.ascontiguousarray(rd[:, ry0 - crop_half:ry0 + crop_half,
                                                   rx0 - crop_half:rx0 + crop_half]).astype(np.float32)
    return Dataset(cube, angles, tags, texp=texp, name=f"n{seq}",
                   meta={"psf_template": psf, "crop_half": crop_half, "nframes_total": n, "root": root,
                         "ref_file": rf if ref_cube is not None else None},
                   ref_cube=ref_cube)


# ----------------------------------------------------------------------------
# legacy NaN masks (reduce_near_2 L2147-2172; block_burn / block_airy, default off)
# ----------------------------------------------------------------------------
def legacy_masks(aa: int = 0, bb: int = 0, ab: int = 0, ba: int = 0, block_airy: bool = False,
                 truenorth: float = NEAR.truenorth, shsize: int = NEAR.shsize,
                 hsize: int = NEAR.hsize) -> Callable[[np.ndarray], np.ndarray]:
    """``mask_fn`` for :class:`~klip_tpe.reducer.KLIPReducer` reproducing the two
    legacy masks of the final image.

    ``block_burn`` (any of ``aa, bb, ab, ba > 0``, L2153-2159): rotate the image so the
    persistence streak is vertical (``fang = 54 - truenorth`` deg, IDL
    ``rot(img, -fang)`` = CCW by ``fang``), NaN the column band
    ``x in [shsize-aa, shsize+ab-1]`` for ``y <= shsize`` and ``x in [shsize-ba,
    shsize+bb-1]`` for ``y >= shsize``, rotate back (the double bilinear rotation
    of the whole image is part of the legacy behaviour).
    ``block_airy`` (L2162-2172): NaN two 85-px disks centred at the full-frame
    positions (57, 86) and (244, 215) expressed in the cropped frame.
    """
    fang = 54.0 - float(truenorth)
    burn = any(v > 0 for v in (aa, bb, ab, ba))

    def fn(img: np.ndarray) -> np.ndarray:
        out = np.asarray(img, float).copy()
        ny, nx = out.shape
        if burn:
            out = rotate_ccw(out, fang)
            if aa > 0 or ab > 0:
                out[0:min(shsize + 1, ny), max(shsize - aa, 0):min(shsize + ab, nx)] = np.nan
            if bb > 0 or ba > 0:
                out[shsize:min(2 * shsize, ny), max(shsize - ba, 0):min(shsize + bb, nx)] = np.nan
            out = rotate_ccw(out, -fang)
        if block_airy:
            yy, xx = np.mgrid[0:ny, 0:nx]
            for (xc_full, yc_full) in ((57.0, 86.0), (244.0, 215.0)):
                xc, yc = shsize + xc_full - hsize, shsize + yc_full - hsize
                out[np.hypot(xx - xc, yy - yc) <= 85.0] = np.nan
        return out.astype(np.float32)
    return fn


# ----------------------------------------------------------------------------
# injection model
# ----------------------------------------------------------------------------
def _throughput_table(psflib: str) -> Optional[Callable]:
    p = os.path.join(psflib, "throughput_merged.csv")
    if not os.path.exists(p):
        return None
    seps, thr = [], []
    with open(p) as f:
        for row in csv.reader(f):
            try:
                seps.append(float(row[0]))
                thr.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    seps, thr = np.array(seps), np.array(thr)
    o = np.argsort(seps)
    seps, thr = seps[o], thr[o]

    def fn(r):
        r = np.asarray(r, float)
        out = np.interp(r, seps, thr, right=1.0)
        return np.clip(out, 0.0, 1.0)
    return fn


def analytic_throughput(r):
    """Fallback ``T(r) = 1 - exp(-(r/0.742)^1.73)`` (AGPM-N4 EE within 1 lambda/D)."""
    r = np.asarray(r, float)
    return np.clip(1.0 - np.exp(-(np.maximum(r, 0) / 0.742) ** 1.73), 0.0, 1.0)


def maire_agpm_transmission(r):
    """Digitised Maire et al. (2020) AGPM-N4 curve (legacy ``near2_agpm_trans``)."""
    rr = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 6.0]
    tt = [0.00, 0.05, 0.18, 0.337, 0.438, 0.513, 0.599, 0.734, 0.810, 0.86, 0.898, 0.94, 0.97, 1.00]
    return np.clip(np.interp(np.asarray(r, float), rr, tt), 0, 1)


class N4Library:
    """Measured off-axis AGPM-N4 PSF library + throughput (``near2_n4psf.pro``)."""

    def __init__(self, psflib: str):
        from astropy.io import fits
        self.dir = psflib
        with fits.open(os.path.join(psflib, "n4_psf_cube_EEnorm.fits")) as h:
            cube = np.asarray(h[0].data, float)          # (14, 41, 41)
            hdr = h[0].header
        self.platescale = float(hdr.get("PLATESCL", 0.0453))
        self.lambdad_mas = float(hdr.get("LAMBDAD", 283.0))
        self.rap = (self.lambdad_mas / 1000.0) / self.platescale    # 6.25 library px
        # slices 12 and 13 share the final offset -> average; 13 slices at 0.150..1.050"
        sl = list(cube[:12]) + [0.5 * (cube[12] + cube[13])] if cube.shape[0] >= 14 else list(cube)
        stamps = []
        for s in sl:
            st = shift_bilinear(np.where(np.isfinite(s), s, 0.0), -0.5, -0.5)[:40, :40]   # star (20,20)->(19.5,19.5)
            stamps.append(st)
        self.seps = (np.arange(len(stamps)) + 2) * 0.075
        self.stamps = np.array(stamps)
        self.throughput = _throughput_table(psflib) or analytic_throughput

    def model(self, flux_unit: float = 1.0) -> LibraryPSF:
        return LibraryPSF(self.stamps, self.seps, center=(19.5, 19.5), ee_radius_px=self.rap,
                          throughput_fn=self.throughput, refpa_deg=NEAR.n4_refpa, flux_unit=flux_unit)


def template_flux_unit(psf_template: np.ndarray) -> float:
    """Flux unit for a contrast relative to alpha Cen A: ``(129/57) * EE_B(1 lambda/D)`` of
    the B-star template (16x16, centre 7.5, 7.5)."""
    t = np.where(np.isfinite(psf_template), psf_template, 0.0)
    yy, xx = np.mgrid[0:t.shape[0], 0:t.shape[1]]
    c = ((t.shape[1] - 1) / 2.0, (t.shape[0] - 1) / 2.0)
    m = np.hypot(xx - c[0], yy - c[1]) <= NEAR.lam_over_d_px
    return NEAR.b_to_a * float(t[m].sum())


# ----------------------------------------------------------------------------
# factory
# ----------------------------------------------------------------------------
production_defaults: Dict[str, Any] = {
    # "eval 29 of run_20260620_082516" seed configuration
    "bin": 29, "n_ang": 2, "filter": 12, "angsep": 1.923, "anglemax": 26.0,
    "corr_thresh": 0.907, "noise_max": 1.451, "coronoise_max": 1.471, "k_klip": 6,
}


def make_reducer(root: str, nights: Sequence[int] = (1, 2, 3, 4, 5, 6), frames: Optional[slice] = None,
                 use_library: bool = True, weighting: str = "equal", max_workers="auto", pool: str = "auto",
                 fast: bool = False, n_min_ref: int = 10, datasets: Optional[Dict[int, Dataset]] = None,
                 log: Callable[[str], None] = print, use_rdi: bool = False, rdi_mode: str = "ardi",
                 do_destripe: bool = False, mask_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 backend: str = "klip") -> PartitionedReducer:
    """Build the per-night :class:`PartitionedReducer` for the NEAR campaign.
    ``backend``: ``'klip'`` (built-in) | ``'pyklip'`` | ``'vip'`` PSF subtraction.
    ``use_rdi``/``rdi_mode`` use ``n<seq>/AB_cube_refstar_clean.fits`` when present
    (nights without one fall back to ADI); ``mask_fn`` e.g. :func:`legacy_masks`."""
    lib = None
    if use_library:
        try:
            lib = N4Library(os.path.join(root, "psflib"))
        except Exception as exc:
            log(f"  psflib unavailable ({exc!r}); using the B template + analytic throughput")
    reducers = {}
    for seq in nights:
        ds = datasets[seq] if datasets and seq in datasets else load_night(root, seq, frames=frames)
        psf = ds.meta.get("psf_template")
        fu = template_flux_unit(psf) if psf is not None else 1.0
        thru = lib.throughput if lib is not None else analytic_throughput
        fallback = TemplatePSF(psf, center=(7.5, 7.5), ee_radius_px=NEAR.lam_over_d_px,
                               star_flux=fu, throughput_fn=thru) if psf is not None else None
        model = lib.model(flux_unit=fu) if lib is not None else fallback
        red = reducer_class(backend)(ds, pxscale=NEAR.pxscale, lam_m=NEAR.lam_klip, diam_m=NEAR.diam,
                          injection_model=model, fallback_model=fallback, truenorth=NEAR.truenorth,
                          fwhm_px=NEAR.fwhm_px, zone_pad=2.0, outrad_cap=70.0,
                          defaults={"fast": fast, "n_min_ref": n_min_ref, "comb_type": "nwadi",
                                    "use_rdi": bool(use_rdi), "rdi_mode": rdi_mode,
                                    "do_destripe": bool(do_destripe)},
                          mask_fn=mask_fn)
        red.name = f"near_n{seq}"
        reducers[f"n{seq}"] = red
        nref = 0 if ds.ref_cube is None else int(ds.ref_cube.shape[0])
        log(f"  loaded n{seq}: {ds.nframes} frames, PA span {ds.angles.min():.1f}..{ds.angles.max():.1f} deg, "
            f"texp {ds.texp:.0f} s, tags {'yes' if ds.tags else 'no'}, ref-star frames {nref}")
    pr = PartitionedReducer(reducers, weighting=weighting, max_workers=max_workers, pool=pool, log=log)
    pr.name = "near"
    return pr


def pa_span_max(reducer: PartitionedReducer) -> float:
    return max(float(np.ptp(reducer.frame_angles(p))) for p in reducer.partitions())


def make_space(reducer: PartitionedReducer, per_night: bool = True, k_klip_max: Optional[int] = None,
               opt_framesel: bool = True, search_angles: bool = True, selection: Optional[str] = "two_slot",
               anglemax_hi: Optional[float] = None, defaults: Optional[Dict[str, Any]] = None,
               width_range: Optional[Tuple[float, float]] = None, search_k: bool = True,
               max_drop: int = 2, bin_range: Optional[Tuple[int, int]] = None) -> SearchSpace:
    """The production search space: 9 parameters per night (+ drop1/drop2).

    ``width_range=(w0, w1)`` adds the searched global integer ``width`` dim of the
    adaptive-annulus protocol (A §3.1, ``RunConfig.opt_width``; default = the range
    midpoint); ``search_k=False`` leaves the ``k_klip`` dims out (scan k-modes, where k
    comes from the per-eval k-scan).  ``max_drop`` = number of drop slots of the two-slot
    selection (2 = IDL drop1/drop2; raise it when nights x image groups give many
    partitions).  ``bin_range`` = (lo, hi) frames per temporal bin (default the NEAR
    production 5-30; short sequences need a smaller range -- see
    :func:`klip_tpe.instruments.generic.make_space`)."""
    d = dict(production_defaults)
    d.update(defaults or {})
    if k_klip_max is None:
        nfmax = max(r.data.nframes for r in reducer.reducers.values())
        k_klip_max = int(min(max(nfmax // 5, 10), 100))
    if anglemax_hi is None:
        span = pa_span_max(reducer)
        anglemax_hi = float(np.ceil(span)) if span > 5 else 360.0
    b_lo, b_hi = (5, 30) if bin_range is None else (int(bin_range[0]), int(bin_range[1]))
    b_lo, b_hi = max(b_lo, 1), max(b_hi, max(b_lo, 1))
    extra_fixed: Dict[str, Any] = {}      # parameters pinned open rather than searched
    block = [Param("bin", b_lo, b_hi, "int", default=int(min(max(d["bin"], b_lo), b_hi)),
                   doc="temporal binning (frames)"),
             Param("n_ang", 1, 8, "int", default=d["n_ang"], doc="azimuthal subdivisions"),
             Param("filter", 0, 25, "int", default=d["filter"], doc="high-pass width (px)")]
    if search_angles:
        block += [Param("angsep", 0.0, 3.0, "float", default=d["angsep"],
                        doc="min reference separation (lambda/D)")]
        # anglemax caps |dPA| between a target and a reference, and its floor is 20 deg.  A
        # sequence whose own PA span cannot reach 20 therefore has nothing to constrain: every
        # pair already satisfies any admissible value.  Searching it anyway used to raise
        # `anglemax: hi < lo` outright for a span between 5 and 20 deg, because anglemax_hi
        # was set from the span while the floor stayed at 20 -- a crash in make_space for any
        # such data set.  Pin it open instead of searching a window that cannot close.
        if float(anglemax_hi) <= 20.0:
            extra_fixed["anglemax"] = 360.0
        else:
            block += [Param("anglemax", 20.0, anglemax_hi, "int",
                            default=min(d["anglemax"], anglemax_hi),
                            doc="max reference field-rotation window (deg)")]
    if opt_framesel:
        block += [Param("corr_thresh", 0.0, 1.0, "float", default=d["corr_thresh"]),
                  Param("noise_max", 0.3, 3.0, "float", default=d["noise_max"]),
                  Param("coronoise_max", 0.3, 3.0, "float", default=d["coronoise_max"])]
    if search_k:
        block += [Param("k_klip", 1, k_klip_max, "int", grid=kgrid(k_klip_max), default=min(d["k_klip"], k_klip_max),
                        doc="KL modes (non-uniform sampling grid)")]
    parts = reducer.partitions()
    # A parameter whose range has collapsed to a point is not a search dimension, it is a
    # constant wearing one.  It still costs the sampler a coordinate, the TPE a density to
    # model and the reader a line in the setup file, and it buys nothing.  NIRCam hit this
    # for real: its cubes are short enough that bin_range came out (1, 1), so every
    # HIP 65426 run has been searching `bin : [1.000, 1.000]`.  Pin it instead -- `fixed`
    # reaches the reducer through `decode`, so the value is unchanged and only the dimension
    # goes away.
    fixed: Dict[str, Any] = {"spat_mean": False, "temp_mean": False, **extra_fixed}
    pin_ids = set()
    for b in block:
        if float(b.hi) <= float(b.lo):
            fixed[b.name] = b.default if b.default is not None else b.lo
            pin_ids.add(id(b))
    block = [b for b in block if id(b) not in pin_ids]
    sp = SearchSpace(fixed=fixed)
    if per_night and len(parts) > 1:
        sp.replicate(block, parts, name_fmt="{base}_{pid}")
    else:
        for b in block:
            sp.add(b)
        sp.partitions = list(parts)
    if width_range is not None:
        w0, w1 = float(width_range[0]), float(width_range[1])
        sp.add(Param("width", w0, w1, "int", default=int(round(0.5 * (w0 + w1))),
                     doc="annulus width (px); outrad = min(inrad + width, r_cap)"))
    if selection and len(parts) > 1:
        sp.with_selection(selection, partitions=parts, max_drop=int(max_drop))
    return sp


def make_guard(reducer: PartitionedReducer, n_min_ref: int = 10, ref_frac: float = 0.95,
               k_max: int = 100, frame_selection: bool = True):
    """Feasibility projection: frame-selection no-cut snapping (addendum 2) followed
    by the reference-count guardrail evaluated on the executed frame set."""
    ref = ReferenceCountGuard(angles_fn=reducer.frame_angles, tags_fn=reducer.frame_tags,
                              fwhm_px=NEAR.fwhm_bin_px, lam_over_d_px=NEAR.lam_over_d_px,
                              n_min_ref=n_min_ref, ref_frac=ref_frac, k_max=k_max)
    if not frame_selection:
        return ref
    fs = FrameSelectionGuard(tags_fn=reducer.frame_tags,
                             nframes_fn=lambda pid: reducer.reducers[pid].data.nframes)
    return compose(fs, ref)


def default_config(reducer: PartitionedReducer, clean_subtract: bool = True, known=(),
                   measured_kernel: bool = True, metric: str = "mawet"):
    """Objective + position sampler with the production metric settings.  (The two-source
    area-midpoint rule lives in ``RunConfig.pair_area_midpoint``, applied by the Runner.)
    ``metric='fmmf'`` swaps the matched filter for the forward-modelled one."""
    kfn = reducer.matched_filter_kernel if measured_kernel else None
    kw = dict(pxscale=NEAR.pxscale, fwhm=NEAR.fwhm_px, kernel_fn=kfn, known=list(known))
    if str(metric).lower() == "fmmf":
        from ..fmmf import FMMFSNR
        metric = FMMFSNR(**kw)
    else:
        metric = MawetPeakSNR(**kw)
    objective = Objective(metric, clean_subtract=clean_subtract)
    sampler = PositionSampler(fwhm_as=NEAR.fwhm_px * NEAR.pxscale, known=list(known))
    return objective, sampler
