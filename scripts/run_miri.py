#!/usr/bin/env python3
"""Optimize a JWST/MIRI coronagraphic sequence from its stage-2 ``calints``.

    python3 scripts/run_miri.py --data ~/Data/JWST/<target> --target <NAME> --check
    python3 scripts/run_miri.py --data ~/Data/JWST/<target> --target <NAME> \
        --ann 6 30 --n-iter 1000 --out M_<target>

Three things here are MIRI's and not NIRCam's, and each of them is a way to get a wrong
answer quietly rather than a crash (see :mod:`klip_tpe.instruments.miri`):

1. *Throughput is a function of position, not radius.*  The four-quadrant phase masks
   suppress along two lines through the star, so at one separation the throughput varies
   by a factor of two to four with azimuth.  ``psf='stpsf'`` routes MIRI to the 2-D map
   rather than the radial grid; the run log prints which model was built, and it should
   say ``miri_library``.

2. *The dead zones must not reach the noise estimate.*  Where the phase mask has taken the
   starlight it has taken the speckles too, so a dead-zone pixel is quieter than the ring it
   sits in: left in, it depresses sigma, inflates every S/N, and gives the optimizer an
   incentive to choose parameters that preserve the dead zones.  They are excluded from the
   statistic (:func:`~klip_tpe.instruments.miri.dead_zone_pixel_mask` as ``pixel_mask``) and
   kept in the cube.  Not the other way round: NaN-ing them into the cube, which is what an
   earlier version of this script did, is destroyed by the high-pass filter -- it runs at
   ``nan_aware=False`` and spreads each NaN over a box its own width, so 5% of the frame
   becomes 100% of it and every frame is dropped as empty.

3. *Injections must not land in a dead zone.*  The boundaries are fixed to the detector
   and the sampler works in sky position angle, so the forbidden sectors depend on the
   rolls (:func:`~klip_tpe.instruments.miri.forbidden_pa`).  An injection into a dead zone
   recovers nothing, and the search reads that as a property of the parameters it was
   trying.  It is a property of the mask.

``--check`` exercises the whole path -- load, mask, model, one reduction at the seeded
default -- and prints what it built, without starting a search.  Run it first.  It goes
through the same ``reduce_config`` call the search uses, because reducing one partition
directly skips partition selection and a space that decodes to nothing passes that check
and fails the run.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from klip_tpe.backends import spaceklip as sk                      # noqa: E402
from klip_tpe.instruments import generic, miri                     # noqa: E402
from klip_tpe.runner import (CalibrationConfig, RunConfig, Runner,  # noqa: E402
                             ValidationConfig)


def default_annuli(fwhm_px, crop_half, first_fwhm=6.0, inner_fwhm=2.0, max_px=40.0):
    """Default MIRI annulus edges: a short inner annulus, then two wider ones.

    The MIRI coronagraphic field spans 2 to ~11 FWHM, over which the contrast and the
    best KLIP parameters both change a great deal, so one annulus over the whole range
    averages the inner working distance together with the background-limited outside.

    The inner edge is the usual 2 FWHM.  The first annulus ENDS at ``first_fwhm`` FWHM
    (~20 px on F1140C), and that number is measured rather than chosen: with the 4QPM dead
    zones eating ~10% of the ring and one known companion on it, ending the first annulus
    at 6 FWHM leaves 8 clean Mawet apertures for 3 injected sources and 7 for 4, while
    ending it at 5 FWHM leaves room for only 2.  Since the count comes from
    ``n_sources_rule`` as 4 for the innermost annulus, 6 FWHM is where a shrunk inner
    annulus and a usable noise ring meet.

    The remaining span is split at its geometric mean, which keeps the two outer annuli
    comparable in log-radius rather than handing the outermost most of the field.  Falls
    back to fewer annuli when the crop leaves no room to split.
    """
    lo = max(inner_fwhm * float(fwhm_px), 3.0)
    hi = min(0.45 * float(crop_half) * 2.0, float(max_px))
    e1 = first_fwhm * float(fwhm_px)
    if hi <= 1.5 * lo:
        return (lo, hi)
    if hi <= 1.3 * e1 or e1 <= 1.2 * lo:
        return (lo, hi)
    e2 = float(np.sqrt(e1 * hi))
    edges = [lo, e1, e2, hi]
    # Drop interior edges that would leave an annulus too thin to mean anything radially.
    # A zone narrower than ~1.5 FWHM cannot hold a distinct set of KLIP parameters, and its
    # injection band collapses to a single radius anyway; on F1550C the geometric split does
    # exactly this.  Dropping the edge merges the pair instead of keeping a slice that only
    # looks like a separate annulus in the setup file.
    wmin = 1.5 * float(fwhm_px)
    changed = True
    while changed and len(edges) > 2:
        changed = False
        for i in range(len(edges) - 1):
            if edges[i + 1] - edges[i] < wmin:
                edges.pop(i + 1 if i + 1 < len(edges) - 1 else i)
                changed = True
                break
    return tuple(edges)


def resolve_star_flux(a, info, m, log):
    """``flux_unit`` for the injection model, or refuse to run.

    What one unit of *contrast* is worth in the frames' own units.  The default everywhere
    below this is ``star_flux or 1.0``, and 1.0 is not a neutral choice -- it is a source of
    one count against a MIRI cube whose pixels reach several hundred MJy/sr.  For HIP 65426
    in F1140C the right value is 1.12e5, so every injection was 1.1e5 times too faint: the
    calibration walked its entire ladder from 3e-5 to the 1e-1 cap with the median S/N flat
    at zero (-0.11, 0.00, -0.02, -0.31, -0.04), reported that it could not calibrate, and
    the search ran for two hours ranking noise.  Nothing else in the run said anything.

    So: derive it, or stop.  In order -- ``--star-flux`` (the raw number),
    ``--flux-density-jy`` (the star's flux density in this filter, converted with the data's
    own ``PIXAR_SR`` and the injection library's own EE radius), or
    ``datasets.PHOTOMETRY['<target>_<filter>']`` if the pair is recorded there.  ``--star-flux
    1`` is still accepted, because a deliberate raw-units run is a legitimate thing to ask
    for -- but it has to be asked for.
    """
    from klip_tpe import datasets
    if a.star_flux is not None:
        log(f"  flux: star_flux = {a.star_flux:.4e} (given)")
        return float(a.star_flux)
    S, src = a.flux_density_jy, "--flux-density-jy"
    if S is None:
        key = f"{a.target or ''}_{m['filter']}".lower().replace("-", "").replace(" ", "")
        phot = datasets.PHOTOMETRY.get(key)
        if phot and phot.get("flux_density_jy"):
            S, src = float(phot["flux_density_jy"]), f"datasets.PHOTOMETRY[{key!r}]"
            log(f"  flux: {src} -> S = {S:.5f} Jy ({phot.get('ref', '')})")
    if S is None:
        raise SystemExit(
            f"no stellar flux for {a.target!r} in {m['filter']}, and the fallback is 1.0 -- "
            f"which is not a neutral default but a source of one count against a cube whose "
            f"pixels reach hundreds of MJy/sr. Every injection would be ~1e5 times too faint, "
            f"the calibration would fail at its contrast cap, and the search would spend "
            f"hours ranking noise (this happened). Give it one of:\n"
            f"  --flux-density-jy S   the star's flux density in this filter [Jy]\n"
            f"  --star-flux F         the flux unit directly, if you have computed it\n"
            f"  --star-flux 1         deliberately raw units, no contrast axis\n"
            f"or add '{a.target or '<target>'}_{m['filter']}'.lower() to "
            f"klip_tpe.datasets.PHOTOMETRY with its provenance.")
    pixar_sr = float(info.get("pixar_sr") or float("nan"))
    if not np.isfinite(pixar_sr):
        raise SystemExit("the frames carry no PIXAR_SR, so a flux density cannot be put on "
                         "their scale; pass --star-flux instead")
    return miri.star_flux_from_flux_density(m["filter"], S, pixar_sr,
                                            bunit=info.get("bunit") or "MJy/sr", log=log)


def build(a, log):
    """Everything up to the Runner: datasets, reducer, space, objective, sampler."""
    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.data), "**", "jw*_calints.fits"),
                             recursive=True))
    if not files:
        raise SystemExit(f"no jw*_calints.fits under {a.data} -- "
                         f"fetch them with scripts/fetch_jwst_ar.py")
    log(f"{len(files)} calints under {a.data}")
    # --filter SELECTS files, it does not just relabel them.  A programme downloaded whole
    # holds several filters in one directory and a dataset has one wavelength and one mask.
    sc = tuple(a.star_center) if getattr(a, "star_center", None) else None
    dsets, info = sk.load_calints(files, science_target=a.target, half_px=a.crop,
                                  partition=a.partition, filter=a.filter, star_center=sc,
                                  destripe=getattr(a, "destripe", None),
                                  ref_targets=getattr(a, "ref_target", None), log=log)
    filt = a.filter or info.get("FILTER") or info.get("filter")
    if not filt or str(filt).upper() not in miri.MODES:
        raise SystemExit(
            f"filter {filt!r} is not a MIRI coronagraphic filter {sorted(miri.MODES)}. "
            f"This driver is for MIRI; NIRCam sequences go through their own path.")
    m = miri.mode_for_filter(filt)
    px = float(info.get("pxscale") or miri.pixelscale(m["filter"]))
    log(f"{m['filter']} / {m['image_mask']} ({m['kind']}), {px:.6f}\"/px, "
        f"lambda = {m['lam_m'] * 1e6:.3f} um")

    if a.nan_dead_zones:
        # Only honest with no high-pass at all.  Kept because a no-filter reduction is a
        # legitimate thing to ask for and there the NaNs do stay where they are put.
        log("  --nan-dead-zones: setting the suppressed pixels to NaN IN THE CUBE. This is "
            "destroyed by any high-pass filter (nan_aware=False spreads NaN by the filter "
            "width); only use it with the filter pinned to 1")
        dsets = miri.apply_quadrant_mask(dsets, min_throughput=a.min_throughput, log=log)
    if not a.dead_zones:
        log("  NOT excluding the quadrant dead zones (--no-dead-zones): suppressed pixels "
            "stay in the noise statistics, where they depress sigma and inflate S/N, and "
            "injections may land in them")

    star_flux = resolve_star_flux(a, info, m, log)
    backend = getattr(a, "backend", "pyklip") or "pyklip"
    red = sk.make_reducer(dsets, pxscale=px, wavelength_m=m["lam_m"], diam_m=miri.DIAMETER_M,
                          psf="stpsf", star_flux=star_flux, max_workers=a.workers,
                          mode=a.mode, backend=backend, log=log)
    lod = (m["lam_m"] / miri.DIAMETER_M) * 206265.0 / px
    log(f"lambda/D = {lod:.2f} px, FWHM = {red.fwhm:.2f} px")

    # A searched reference library instead of the fixed all-or-nothing one: how many
    # correlation-ranked frames to keep from the OTHER roll (nkeep_altroll) and from the
    # reference star (nkeep_psfref).  Those two subsume the ADI / RDI / ADI+RDI categorical --
    # nkeep_psfref = 0 is ADI, nkeep_altroll = 0 is RDI -- and are strictly more expressive,
    # because they can take PART of a pool rather than all of it.  ReferenceLibraryGuard
    # already floors their sum at n_min_ref, so the empty basis is unreachable while both
    # pure cases stay reachable.
    if getattr(a, "searched_library", True):
        if len(red.reducers) != 1:
            log("  library: NOT searched -- --partition roll puts each roll in its own "
                "reducer, where every frame shares the target's partition label, so the "
                "alternate-roll pool is empty by construction.  Use --partition all.")
        else:
            r0 = next(iter(red.reducers.values()))
            nref = 0 if r0.data.ref_cube is None else int(np.shape(r0.data.ref_cube)[0])
            ang = np.asarray(r0.data.angles, float)
            part = np.round(ang, 1).astype(str)            # the roll is the label
            rolls, counts = np.unique(part, return_counts=True)
            if nref < 2 or rolls.size < 2:
                log(f"  library: NOT searched -- {nref} reference frame(s) and {rolls.size} "
                    f"roll(s); a ranked library needs a pool to rank")
            else:
                try:
                    r0.set_reference_library(partition=part, ref_group="psfref",
                                             n_min_ref=2, metric="cc")
                except NotImplementedError as exc:
                    # Refuse before any compute.  This used to be accepted on pyKLIP, which
                    # then ignored the counts: run v6 searched them for 4 h and every library
                    # from pure ADI to pure RDI reduced bit-identically.
                    raise SystemExit(
                        f"a searched reference library needs an engine that applies it, and "
                        f"--backend {backend} does not.\n  {exc}\n"
                        f"Either --backend klip (the built-in annular KLIP, which does), or "
                        f"--no-searched-library to reduce with {backend}'s own {a.mode} library "
                        f"-- which is what every {backend} run before this check did anyway.")
                log(f"  library: searched, ranked per target frame by cross-correlation -- "
                    f"nkeep_altroll over {int(ang.size)} science frames in {rolls.size} rolls "
                    f"({', '.join(f'{c}@{r}' for r, c in zip(rolls, counts))}), "
                    f"nkeep_psfref over {nref} reference frames; sum floored at 2")

    ann = tuple(a.ann) if a.ann else default_annuli(red.fwhm, a.crop)
    if len(ann) < 2 or any(b <= c for c, b in zip(ann, ann[1:])):
        raise SystemExit(f"--ann needs 2 or more increasing edges in pixels; got {list(ann)}")
    # The forbidden sectors are computed at ONE radius but the injections now span several
    # annuli, and a dead zone of fixed physical width subtends a LARGER angle the closer in
    # you look.  So use the innermost annulus' mid-radius: that over-masks the outer rings
    # slightly, where there is aperture budget to spare, rather than under-masking the inner
    # one, where there is not.
    mid_as = 0.5 * (ann[0] + ann[1]) * px
    log(f"{len(ann) - 1} annulus/annuli at "
        + ", ".join(f"{lo:.1f}-{hi:.1f} px ({lo * px:.2f}-{hi * px:.2f}\")"
                    for lo, hi in zip(ann, ann[1:]))
        + f"; forbidden sectors computed at the innermost mid-radius {mid_as:.2f}\"")

    known = [tuple(k) for k in (a.known or [])]
    # The forbidden sectors are computed on the ring the injections actually land on and
    # from THESE rolls: the dead zones are fixed to the detector, so which sky angles they
    # eat depends on how the telescope was pointed.
    angles = np.concatenate([np.asarray(d.angles, float).ravel() for d in dsets.values()])
    tn = float(getattr(red, "truenorth", 0.0))
    fpa, pmask = [], None
    if a.dead_zones:
        fpa = miri.forbidden_pa(angles, rho_as=mid_as, filter=m["filter"], truenorth=tn,
                                min_throughput=a.min_throughput, log=log)
        # The same geometry as fpa, per pixel rather than per PA at one separation, in the
        # de-rotated frame the metric works in.  The two go together: injections avoiding
        # sectors that still inflate the ring sigma are scored against a noise level nothing
        # is measuring them at.
        shp = np.asarray(next(iter(dsets.values())).cube).shape[-2:]
        pmask = miri.dead_zone_pixel_mask(shp, px, angles, filter=m["filter"], truenorth=tn,
                                          min_throughput=a.min_throughput, log=log)
    obj, samp = generic.default_config(red, known=known, forbidden_pa=fpa, pixel_mask=pmask)
    if fpa:
        blocked = sum(2 * w for _, w in fpa)
        log(f"  injections barred from {blocked:.0f} deg of the ring ({100 * blocked / 360:.0f}%)")
        if blocked > 180:
            log("  WARNING: more than half the ring is a dead zone at this separation -- "
                "the annulus is probably too close in for this mask")
    return dsets, info, red, ann, obj, samp, m, px


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="directory holding jw*_calints.fits")
    ap.add_argument("--target", default=None, help="TARGPROP of the science target; the "
                                                   "other targets become the RDI library")
    ap.add_argument("--filter", default=None,
                    help="SELECT the files of this filter (not just relabel them). A "
                         "programme downloaded whole holds several; a dataset has one "
                         "wavelength and one mask, so loading them together is wrong")
    # 'all', not 'roll'.  A searched reference library needs both rolls in ONE reducer: under
    # 'roll' each partition holds only its own roll's frames and ref_cube is the reference
    # star alone, so the "altroll" pool is empty (ReferenceLibrary.eligible drops every frame
    # sharing the target's partition label, and under 'roll' they all share it).  The price,
    # as the loader's own docstring says, is one bin/n_ang/filter/k_klip block for both rolls
    # instead of one per roll -- which also takes the dimension count down, not up.
    ap.add_argument("--partition", default="all", choices=["roll", "all"])
    ap.add_argument("--ref-target", nargs="+", default=None, metavar="TARGPROP",
                    help="restrict the RDI library to these star(s) by TARGPROP prefix. "
                         "The default takes every non-background pointing that is not the "
                         "science target, which in a multi-programme directory pulls in "
                         "other people's targets: HIP 65426's MIRI download also holds "
                         "HD 141569A, a resolved disk")
    ap.add_argument("--no-searched-library", dest="searched_library", action="store_false",
                    help="keep the whole RDI library fixed instead of searching how many "
                         "correlation-ranked frames to keep from each pool")
    ap.add_argument("--crop", type=int, default=80, metavar="HALF",
                    help="crop to 2*HALF+1 px about the star (default 80 = 17.5 arcsec)")
    ap.add_argument("--star-center", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="measured star position in the SUBARRAY, 0-based, instead of CRPIX "
                         "(which is the aperture reference point). Measure it by maximising "
                         "the point symmetry of the stacked frame -- a flux centroid is "
                         "biased by the four-quadrant residual and comes out on the wrong "
                         "side. On GO 1386 F1140C, CRPIX is good to 0.39 px (45 mas), both "
                         "rolls agreeing to 0.05 px, so this is not usually needed")
    ap.add_argument("--ann", type=float, nargs="+", default=None, metavar="EDGE",
                    help="annulus edges in pixels, 2 or more, increasing (2 edges = one "
                         "annulus). Default: three annuli, see default_annuli()")
    ap.add_argument("--known", type=float, nargs=2, action="append", metavar=("RHO", "PA"),
                    help="a real companion (arcsec, deg) injections keep clear of; repeatable")
    ap.add_argument("--star-flux", type=float, default=None,
                    help="the injection model's flux unit: what one unit of CONTRAST is "
                         "worth in the frames' own units. Without it (or --flux-density-jy, "
                         "or a PHOTOMETRY entry) the run refuses to start, because the "
                         "fallback of 1.0 makes every injection ~1e5x too faint and the "
                         "search then ranks noise for hours without complaining")
    ap.add_argument("--flux-density-jy", type=float, default=None, metavar="S",
                    help="the star's flux density in THIS filter [Jy]; converted to a flux "
                         "unit with the frames' PIXAR_SR and the injection library's own EE "
                         "radius. Preferred over --star-flux: it is a number you can cite")
    ap.add_argument("--mode", default="ADI+RDI")
    ap.add_argument("--backend", default="pyklip", choices=["pyklip", "klip"],
                    help="PSF-subtraction engine. Only 'klip' (the built-in annular KLIP) applies "
                         "the searched reference library; pyKLIP builds its own basis from the "
                         "whole reference cube, so with it you must pass --no-searched-library")
    ap.add_argument("--min-throughput", type=float, default=0.30,
                    help="pixels transmitting less than this are dead zones (default 0.30)")
    ap.add_argument("--no-dead-zones", "--no-mask-quadrants", dest="dead_zones",
                    action="store_false",
                    help="stop excluding the dead zones from the noise estimate and from the "
                         "injection positions. Only to reproduce a reduction that did not")
    ap.add_argument("--nan-dead-zones", action="store_true",
                    help="ALSO set the suppressed pixels to NaN in the cube. Incompatible "
                         "with a high-pass filter of any width (it spreads NaN over the "
                         "whole frame and every frame is then dropped as empty)")
    ap.add_argument("--n-iter", type=int, default=1000)
    ap.add_argument("--n-remeasure", type=int, default=3, metavar="N",
                    help="fresh source draws per search trial, averaged into its score "
                         "(default 3). The objective's only random input is the injected "
                         "sources' azimuth, and it is worth sd 0.84 of S/N on a single draw; "
                         "averaging 3 measured 0.48. Costs N reductions per trial. 1 = the "
                         "IDL's single draw")
    ap.add_argument("--n-init", type=int, default=None)
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--max-drop", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--workers", default="auto")
    ap.add_argument("--check", action="store_true",
                    help="build everything and reduce once at the default, then stop")
    ap.add_argument("--default-only", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no-destripe", dest="destripe", action="store_false", default=None,
                    help="skip the detector-frame destriping, which is ON by default for "
                         "MIRI: the per-row offset measures as large as the pixel-to-pixel "
                         "noise and removing it drops the sky scatter by ~1.7x")
    ap.add_argument("--show", nargs="?", const="window", default=False, metavar="MODE",
                    help="also open a live window in THIS process ('window' or 'inline'). "
                         "Panels are written either way -- prefer watching them with "
                         "'klip-tpe view --run-dir <out>' from a second shell, which is a "
                         "separate process and so cannot be starved by the optimizer")
    ap.add_argument("--no-display", dest="display", action="store_false",
                    help="do not write panel PNGs at all (nothing to watch, live or later)")
    ap.add_argument("--display-every", type=int, default=10, metavar="N",
                    help="render a panel every N evaluations (default 10; a new best and the "
                         "last evaluation of an annulus always render)")
    ap.add_argument("--pdf-every", type=int, default=0, metavar="N",
                    help="also write a PDF panel every N evaluations (default 0 = never)")
    a = ap.parse_args(argv)

    out = a.out or os.path.join(os.getcwd(), f"miri_{a.target or 'run'}")
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "run.log"), "a", buffering=1)

    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True)
        logf.write(line + "\n")

    dsets, info, red, ann, obj, samp, m, px = build(a, log)

    # search_angles=False: angsep and anglemax are not weak on two-roll JWST data, they are
    # inert.  Each partition is one roll, so every science frame in it shares a position
    # angle and reference_mask's dpa = |angle - angle[target]| is identically 0.  Then
    # dpa <= anglemax holds for any anglemax >= 20 (the parameter's own floor), and
    # dpa >= angsep_deg fails for every frame once angsep > 0, which empties the mask and
    # sends it through `if refs.sum() < 4: refs = all but the target` -- exactly the
    # angsep = 0 set.  Same basis by two routes, at the cost of 4 of 12 dimensions.
    space = generic.make_space(red, k_klip_max=a.k_max, max_drop=a.max_drop, search_angles=False)
    space.project = generic.make_guard(red, k_max=a.k_max)
    log(f"search space: {space.ndim} dimensions over {len(dsets)} partition(s)")

    n_init = a.n_init or min(max(int(0.15 * a.n_iter), 40), 400)
    n_iter, n_init = (1, 1) if (a.default_only or a.check) else (a.n_iter, n_init)
    cfg = RunConfig(ann_edges=[float(v) for v in ann], n_iter=n_iter, n_init=n_init, seed=a.seed,
                    validation=ValidationConfig(n_top=1 if n_iter == 1 else 6, n_valid=10),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=3),
                    n_remeasure=1 if a.check else max(int(a.n_remeasure), 1),
                    verify=n_iter > 1, save_eval_images=False)

    if a.check:
        # The same call path the search takes.  Reducing one partition directly skips
        # partition selection, so a space that decodes to no selected partitions passes
        # that check and then fails the run.
        dec = space.decode(space.default_vector())
        log(f"decoded selection: {list(dec.selected)}")
        if not list(dec.selected):
            raise SystemExit("the space decodes to NO selected partitions; reduce_config "
                             "would map over an empty list")
        # Everything that describes the INPUT goes before the reduction, because the
        # reduction is what fails, and a diagnostic printed after it never prints.  That is
        # how a 100%-NaN cube read as a masking bug for two runs.
        filt = dec.params.get("filter", "?")
        nan_frac = float(np.mean(~np.isfinite(np.asarray(list(dsets.values())[0].cube, float))))
        log(f"default high-pass filter width: {filt} px")
        log(f"non-finite pixels in the first partition's cube: {100 * nan_frac:.2f}%")
        pm = getattr(obj.metric, "pixel_mask", None)
        log("dead zones excluded from the noise estimate: "
            + (f"{int(np.count_nonzero(pm))} px ({100 * np.mean(pm):.1f}% of the frame)"
               if pm is not None else "NONE"))
        if nan_frac > 0 and isinstance(filt, (int, float)) and filt > 1:
            raise SystemExit(
                f"{100 * nan_frac:.2f}% of the cube is non-finite and the default high-pass "
                f"width is {filt} px. That filter runs over a running sum, so ONE gap costs "
                f"every pixel after it along both axes -- one gap near a corner of an 81x81 "
                f"frame takes three quarters of it -- and every frame is then dropped as "
                f"empty. Fix the cube, not the filter: the loader's own log says how many "
                f"pixels it could not close and why.")
        ev = red.reduce_config(dec, None, tag="check")
        img = np.asarray(ev.image if hasattr(ev, "image") else ev, float)
        model = red.reducers[list(dec.selected)[0]].model
        # Finite WHERE IT MATTERS.  Most of the frame is outside the reduced zones and comes
        # back NaN by design, so the whole-frame fraction says little; the objective is
        # measured in the annulus, and a reduction that returns nothing there returns nothing.
        ny, nx = img.shape[-2:]
        yy, xx = np.mgrid[0:ny, 0:nx]
        rr = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
        inann = (rr >= ann[0]) & (rr <= ann[1])
        fin_ann = float(np.isfinite(img[inann]).mean())
        log(f"default reduction: image {img.shape}, finite {100 * np.isfinite(img).mean():.1f}% "
            f"of the frame, {100 * fin_ann:.1f}% inside the search annulus")
        if fin_ann < 0.5:
            raise SystemExit(
                f"only {100 * fin_ann:.1f}% of the search annulus is finite after the default "
                f"reduction, so the objective has almost nothing to measure. The usual cause "
                f"is a partition with no usable references: KLIP needs frames separated by "
                f"more than angsep at the outer radius, and a single roll reduced ADI-only "
                f"has none. Check the '{len(dsets)} partition(s)' and mode lines above -- "
                f"'pyklip ADI' where you expected 'ADI+RDI' means the reference files were "
                f"not recognised (--target selects the science TARGPROP; everything else "
                f"becomes the library).")
        log(f"injection model: {model.name}  azimuth_dependent="
            f"{getattr(model, 'azimuth_dependent', False)}")
        if m["kind"] == "4qpm" and not getattr(model, "azimuth_dependent", False):
            raise SystemExit(
                f"the injection model is {model.name!r} with a radial throughput, on a "
                f"four-quadrant mask. Every contrast this run reports would be an azimuthal "
                f"average of values differing by a factor of two to four.")
        log("check passed")
        return 0

    # Writing the panels and putting them on screen are two different things, as they are in
    # klip_tpe.cli: --display writes, --show opens a window.  Tying them together (which this
    # script used to do) means a run started without --show writes no panels, and then the
    # separate viewer this very line advertises has nothing to watch -- for the whole run,
    # with no way to attach later.
    callbacks, disp = [], None
    if a.display:
        from klip_tpe.display import LiveDisplay
        disp = LiveDisplay(out, every=a.display_every, pdf_every=a.pdf_every, movie=False,
                           dpi=100, show=a.show)
        callbacks = [disp]
        log(f"panels every {a.display_every} eval(s) -> {out}/  (live window: "
            f"{a.show or 'off'}; from another shell: klip-tpe view --run-dir {out})")

    runner = Runner(red, space, obj, samp, cfg, out, log=log, callbacks=callbacks,
                    resume="never" if a.fresh else "auto")
    runner.run()
    log(f"done -- {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
