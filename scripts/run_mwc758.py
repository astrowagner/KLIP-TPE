#!/usr/bin/env python3
"""MWC 758 F430M with a SEARCHED reference library -- the Python port of the IDL
``optimize_mwc_tpe.pro`` run, on the same data.

Why this one first: it is the only pilot where the answer is already known.  The IDL
searched ``nkalt`` (alternate-roll frames to keep) and ``nkpsf`` (HD 36575 frames to keep)
along with ``k_klip``, the high-pass width and the combine type, and its outputs are still
on disk as ``reduced/F430M_ARDIKLIP_scan_inj_*_refsel*.fits``.  So this run is a test of
the port, not just a demonstration that it runs.

    python3 run_mwc758.py --check              # load, verify geometry, no search
    python3 run_mwc758.py --idl-cc --n-iter 200
    WORKERS=2 python3 run_mwc758.py --n-iter 400 --out M2_mwc758

``--idl-cc`` hands the search the IDL's own similarity matrix
(``F430M_refmetrics_cc.fits``, 75 library x 50 target) instead of recomputing one.  With
it, any difference from the IDL is a difference in the SELECTION or the KLIP, not in how
two languages measure a cross-correlation -- which is the comparison worth making first.

Data, all already on disk:
    MWC758_Cycle2/reduced/F430M_cube_cen_clean.fits   50 science  (rollA 0-24, rollB 25-49)
    HD36575/reduced/F430M_cube_cen_clean.fits         25 reference-star frames
    MWC758_Cycle2/reduced/F430M_angles_clean.sav      the position angles
    MWC758_Cycle2/reduced/F430M_refmetrics_cc.fits    library x target similarity
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from astropy.io import fits

from klip_tpe import (CalibrationConfig, MawetPeakSNR, RunConfig, Runner, Source,
                      ValidationConfig)
from klip_tpe.instruments import generic
from klip_tpe.reducer import Dataset, PartitionedReducer
from klip_tpe.space import SearchSpace

# ---------------------------------------------------------------- the data set
DATA = os.path.expanduser("~/Data/JWST")
SCI = f"{DATA}/MWC758_Cycle2/reduced/F430M_cube_cen_clean.fits"
REF = f"{DATA}/HD36575/reduced/F430M_cube_cen_clean.fits"
ANG = f"{DATA}/MWC758_Cycle2/reduced/F430M_angles_clean.sav"
CCM = f"{DATA}/MWC758_Cycle2/reduced/F430M_refmetrics_cc.fits"
PSF = f"{DATA}/MWC758_Cycle2/MWC758_PSFs_MASK335R_F430M_2024-03-02.fits"

NROLL = 25                 # frames per science roll
LAM, DIAM = 4.30e-6, 6.5   # F430M, JWST
MWC758C = (0.607, 213.8)   # known companion: rho (arcsec), PA (deg), from the IDL setup
#: Detector sampling, as the IDL run_setup records (0.0630 arcsec, FWHM 2.23 px; this
#: filter and aperture give 1.025 lam/D = 2.22 px at that scale, which is the check).
#:
#: The trap: the 640-pixel ``_cen`` cubes are NOT oversampled.  They are the 320-pixel
#: SUB320A335R subarray padded with NaN so the star lands on the frame centre -- only 25%
#: of each frame is finite, in a 318 x 319 box.  The STPSF template beside them IS
#: oversampled (642 px, PIXELSCL 0.0312, OSAMP 2), and taking ITS scale for the science
#: data doubles every radius: the companion moves from 9.6 px to 19.4 and the annulus that
#: should contain it is drawn somewhere else.  The PSF's scale describes the PSF.
PXSCALE = 0.0630
#: The IDL searched inrad/outrad 2-15 px at that scale.
ANNULUS_PX = (2, 15)


def load_angles(path):
    """The IDL angle save file, or a .txt/.npy fallback beside it."""
    for alt, rd in ((path, None),
                    (path.replace(".sav", ".txt"), np.loadtxt),
                    (path.replace(".sav", ".npy"), np.load)):
        if not os.path.exists(alt):
            continue
        if rd is not None:
            return np.asarray(rd(alt), float).ravel()
        from scipy.io import readsav
        d = readsav(alt)
        for v in d.values():
            v = np.asarray(v).ravel()
            if v.size and np.issubdtype(v.dtype, np.number):
                return v.astype(float)
    raise FileNotFoundError(f"no angles beside {path}")


def load_psf(path, pxscale):
    """The off-axis template, block-averaged onto the science pixel grid.

    STPSF writes an oversampled template and records its own scale in ``PIXELSCL``.  That
    is the PSF's grid, not the data's -- see the note on ``PXSCALE`` above -- so the only
    thing to do with it here is work out the integer factor by which the template has to be
    binned down to match the science pixels."""
    if not os.path.exists(path):
        raise SystemExit(
            f"no PSF template at {path}.\n"
            f"The package refuses to inject a Gaussian of unit flux, because the resulting\n"
            f"'contrast' is in raw detector units and nothing downstream can tell it from a\n"
            f"real one.  Point --psf at the STPSF file (MWC758_PSFs_MASK335R_F430M_*.fits).")
    with fits.open(path) as hl:
        hdu = hl[0] if hl[0].data is not None else hl[1]
        psf = np.asarray(hdu.data, float)
        px_psf = float(hdu.header.get("PIXELSCL", pxscale))
    n = max(int(round(pxscale / px_psf)), 1)
    if n > 1:
        cut = (psf.shape[-1] // n) * n
        sl = psf[..., :cut, :cut]
        sh = sl.shape[:-2] + (cut // n, n, cut // n, n)
        psf = sl.reshape(sh).mean(axis=(-3, -1)) * (n * n)      # sum-preserving
        print(f"  PSF template binned {n}x{n}: {px_psf:.6f}\"/px -> {px_psf * n:.6f}\"/px, "
              f"{psf.shape[-1]} px (science grid is {pxscale:.4f}\"/px)")
    else:
        print(f"  PSF template {psf.shape} already on the science grid")
    return psf


def crop_to_finite(*cubes, center=None):
    """Largest square crop, centred on the star, in which every frame of every cube is
    finite.

    The ``_cen`` cubes are a subarray padded into a bigger grid, so three quarters of each
    frame is NaN.  A high-pass filter at ``nan_aware=False`` spreads that NaN over the whole
    frame; ``np.nansum`` of an all-NaN frame is 0.0; ``bin_frames`` drops zero-sum bins; and
    the science cube arrives at the reducer with no frames at all.  Cropping first is the
    fix, and it has to be SYMMETRIC about the star, because the package puts the star at the
    geometric centre of whatever array it is given."""
    fin = np.ones(cubes[0].shape[-2:], bool)
    for c in cubes:
        fin &= np.isfinite(np.asarray(c)).all(axis=0)
    ny, nx = fin.shape
    cy, cx = ((ny - 1) / 2.0, (nx - 1) / 2.0) if center is None else center
    ys, xs = np.where(fin)
    if ys.size == 0:
        raise SystemExit("no pixel is finite in every frame")
    half = int(min(cx - xs.min(), xs.max() - cx, cy - ys.min(), ys.max() - cy))
    if half < 20:
        raise SystemExit(f"the finite region is only {2 * half + 1} px across; "
                         f"is the star really at the frame centre?")
    y0, y1 = int(round(cy - half)), int(round(cy + half)) + 1
    x0, x1 = int(round(cx - half)), int(round(cx + half)) + 1
    out = [np.ascontiguousarray(np.asarray(c)[..., y0:y1, x0:x1]) for c in cubes]
    print(f"  cropped {ny}x{nx} -> {out[0].shape[-2]}x{out[0].shape[-1]} about the star "
          f"(the padding is NaN and a high-pass would spread it over everything)")
    for c in out:
        if not np.isfinite(c).all():
            raise SystemExit("the crop still contains NaN")
    return out


def build(args):
    sci = np.asarray(fits.getdata(args.sci), np.float32)
    ref = np.asarray(fits.getdata(args.ref), np.float32)
    ang = load_angles(args.angles)
    if sci.shape[0] != ang.size:
        raise SystemExit(f"{sci.shape[0]} science frames but {ang.size} angles")
    print(f"  science {sci.shape}  reference {ref.shape}  PA span {np.ptp(ang):.2f} deg")

    sci, ref = crop_to_finite(sci, ref)
    px = args.pxscale or PXSCALE
    psf = load_psf(args.psf, px)
    fwhm_px = 1.025 * LAM / DIAM * 206265.0 / px
    print(f"  pxscale {px:.6f}\"/px   FWHM {fwhm_px:.2f} px "
          f"(the IDL run recorded 0.0630 and 2.23)")

    ds = Dataset(cube=sci, angles=ang, name="mwc758")
    ds.ref_cube = ref
    red = generic.make_reducer({"mwc758": ds}, star_flux=args.star_flux,
                               max_workers=os.environ.get("WORKERS", "auto"),
                               pxscale=px, lam_m=LAM, diam_m=DIAM, fwhm_px=fwhm_px,
                               psf=psf, use_rdi=True,
                               log=lambda s: None, partition_label="dataset")
    r0 = next(iter(red.reducers.values()))
    args._px, args._fwhm = px, fwhm_px

    sim = None
    if args.idl_cc:
        sim = np.asarray(fits.getdata(args.ccm), float)
        # the IDL writes (library, target) with library = [rollA | rollB | HD], which is
        # this package's order too (science first, then each external pool)
        if sim.shape != (sci.shape[0] + ref.shape[0], sci.shape[0]):
            sim = sim.T
        print(f"  using the IDL similarity matrix {sim.shape} "
              f"(median cross-roll {np.median(sim[:NROLL, NROLL:]):.4f}, "
              f"HD {np.median(sim[sci.shape[0]:]):.4f})")

    part = np.array(["rollA"] * NROLL + ["rollB"] * (sci.shape[0] - NROLL))
    r0.set_reference_library(partition=part, ref_group="hd36575",
                             min_keep={"hd36575": 1},     # nkpsf in [1, 25], as the IDL has it
                             n_min_ref=2, metric="cc", similarity=sim)
    return red, r0, sci, ref, ang


def annulus_px(px):
    """The IDL's inrad/outrad, rescaled if the science grid ever changes."""
    lo, hi = ANNULUS_PX
    return round(lo * PXSCALE / px), round(hi * PXSCALE / px)


def check_geometry(red, sci, px, fwhm_px):
    r0 = next(iter(red.reducers.values()))
    rho_px = MWC758C[0] / px
    lo, hi = annulus_px(px)
    ny, nx = sci.shape[-2:]
    print(f"  companion at {MWC758C[0]}\" = {rho_px:.1f} px  (frame {ny}x{nx}, "
          f"half-width {nx // 2} px, {rho_px / fwhm_px:.1f} FWHM out)")
    if rho_px > nx / 2:
        raise SystemExit("the companion falls outside the frame -- pixel scale is wrong")
    inside = lo <= rho_px <= hi
    print(f"  searched annulus {lo}-{hi} px = {lo * px:.3f}-{hi * px:.3f}\" "
          f"{'CONTAINS' if inside else 'DOES NOT CONTAIN'} the companion")
    if not inside:
        raise SystemExit("the searched annulus misses the companion -- check --pxscale")
    print(f"  reference pools: {[g.name for g in r0._reflib_stub().groups]}")
    for p in r0.reference_params():
        print(f"    {p.name:20s} [{p.lo:.0f}, {p.hi:.0f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sci", default=SCI)
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--angles", default=ANG)
    ap.add_argument("--ccm", default=CCM)
    ap.add_argument("--psf", default=PSF)
    ap.add_argument("--pxscale", type=float, default=None,
                    help="override the scale read from the PSF header")
    ap.add_argument("--forced-contrast", type=float, default=None,
                    help="skip calibration and inject at this contrast (the IDL used 1e-4)")
    ap.add_argument("--idl-cc", action="store_true", help="use the IDL similarity matrix")
    ap.add_argument("--check", action="store_true", help="load and verify, run nothing")
    ap.add_argument("--match-idl", action="store_true",
                    help="search exactly the IDL's vector (adds comb_type, pins n_ang=1)")
    ap.add_argument("--n-iter", type=int, default=400)
    ap.add_argument("--n-init", type=int, default=80)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--star-flux", type=float, default=None,
                    help="stellar flux in the cube's units; required for a calibrated contrast")
    ap.add_argument("--out", default="M2_mwc758")
    a = ap.parse_args()

    print("MWC 758 F430M, searched reference library")
    red, r0, sci, ref, ang = build(a)
    check_geometry(red, sci, a._px, a._fwhm)
    # Match the IDL's searched vector, so a difference in the answer is a difference in
    # the reference library rather than in what was searched.  optimize_mwc_tpe.pro put
    # combtype [0,2], klipfilter [5,9], nkalt [0,25], nkpsf [1,25] and k_klip [1,50] in the
    # search and held n_ang at 1; without --match-idl the package's own wider ranges stand.
    from klip_tpe import Param
    space = generic.make_space(red, k_klip_max=50, search_angles=False, bin_range=(1, 1))
    if a.match_idl:
        keep = [q for q in space.params if q.name not in ("n_ang", "filter")]
        space = SearchSpace(keep)
        space.add(Param(name="filter", lo=5.0, hi=9.0, kind="int", default=5.0,
                        role="reduction", doc="klipfilter, the IDL's [5, 9]"))
        space.add(Param(name="comb_type", lo=0.0, hi=2.0, kind="categorical",
                        choices=["nwadi", "mean", "median"], default=0.0, role="reduction",
                        doc="combtype, the IDL's [0, 2]"))
    space.project = generic.make_guard(red, n_min_ref=2, k_max=50)
    print(f"  searched dimensions ({space.ndim}): {[p.name for p in space.params]}")
    if a.match_idl:
        print("  (--match-idl: n_ang pinned to 1 as the IDL had it, comb_type searched)")
    if a.check:
        # one reduction at the default, so a signature or geometry error surfaces here
        # rather than forty minutes into a search
        from klip_tpe.reducer import ReductionRequest
        x0 = np.array([q.default if q.default is not None else q.lo
                       for q in space.params], float)
        dec = space.decode(space.project(x0, space) if space.project else x0)
        p0 = dict(dec.per_partition.get("mwc758", dec.params))
        p0.update(inrad=annulus_px(a._px)[0], outrad=annulus_px(a._px)[1], use_rdi=True)
        res = red.reducers["mwc758"].reduce(ReductionRequest(p0, None))
        print(f"  default reduction ok: image {res.image.shape}, "
              f"ref_keep {res.meta.get('ref_keep')}, nref_used "
              f"{np.unique(res.meta.get('nref_used', [0]))[:4]}")
        print("\n--check: nothing searched.  Drop it to start the search.")
        return

    obj, samp = generic.default_config(red, known=[MWC758C])
    lo, hi = annulus_px(a._px)
    cal = CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=3)
    if a.forced_contrast:
        cal = CalibrationConfig(forced=[a.forced_contrast])
        print(f"  injection contrast forced to {a.forced_contrast:g} (no calibration), as the IDL run had it")
    cfg = RunConfig(ann_edges=[lo, hi], n_iter=a.n_iter, n_init=a.n_init, seed=a.seed,
                    n_sources=3, search_mode="tpe",
                    validation=ValidationConfig(n_top=3, n_valid=5),
                    calibration=cal,
                    defaults={"k_klip": 10, "bin": 1, "n_ang": 1, "filter": 5,
                              "comb_type": "nwadi", "use_rdi": True,
                              "nkeep_altroll": 25, "nkeep_hd36575": 25},
                    fm_curve=False, save_eval_images=False)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper_runs", a.out)
    out = os.path.normpath(out)
    cbs = []
    if os.environ.get("SHOW", "window") not in ("0", "off", "no", "none", ""):
        from klip_tpe.display import LiveDisplay
        cbs = [LiveDisplay(out, pdf_every=0, movie=False, dpi=100, every=10,
                           show=os.environ.get("SHOW", "window"))]
    print(f"  -> {out}")
    Runner(red, space, obj, samp, cfg, out, log=print, callbacks=cbs).run()


if __name__ == "__main__":
    main()
