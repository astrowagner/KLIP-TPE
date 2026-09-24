#!/usr/bin/env python3
"""RX J0534.0-0221 (LBTI/LMIRCam L'): optimize the four-night KLIP reduction.

Why this data set is worth a section of its own
-----------------------------------------------
The other three targets in the paper compare an optimized reduction against a *convention*
default.  Here the thing to beat is a published reduction of the same data: the discovery
paper detects RX J0534 b at **S/N ~ 5.2** in the four-night stack, using five principal
components in a local optimization zone at the companion's position.  The planet itself is
independently established -- JWST/NIRCam F444W at S/N ~ 17.5, and common proper motion
between the two epochs at the 6-7 sigma level -- so it is ground truth rather than a
candidate, and it sits at 0.39", the separation where this is hardest.

It is also the paper's largest partition problem.  Four nights of visibly different quality,
each observed in LBTI's dual-aperture mode (two 8.4-m primaries imaging the target
independently, so two uncorrelated speckle fields), each side written as two cubes:
sixteen partitions that can be included, excluded and tuned separately.

Ground truth (discovery paper, Table 2, epoch 2026.159)
------------------------------------------------------
    dRA = -336 +/- 12 mas, dDec = +198 +/- 17 mas  ->  rho = 0.390", PA = 300.5"
    L' = 16.82 +/- 0.15 mag, f = 48 +/- 7 uJy
The companion is passed as ``known=``: it is kept out of the noise annulus and out of the
objective, so the optimizer never sees it.  Its recovered S/N is the held-out test.

Running it
----------
    python scripts/run_rxj0534.py                  # 16 partitions (night x side x cube)
    python scripts/run_rxj0534.py --partitions 4   # one per night, comparable to run B2
    python scripts/run_rxj0534.py --default-only   # just the baseline, no search (~minutes)

``--partitions`` is the interesting knob: 4 nights reproduces the 38-D regime the paper
already benchmarks, 16 pushes it to ~150-D.  Running both gives the dimension-scaling
section a third, real point rather than a synthetic one.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from astropy.io import fits

from klip_tpe import (CalibrationConfig, RunConfig, Runner, ValidationConfig)
from klip_tpe.instruments import generic

# ----------------------------------------------------------------- instrument / target
ROOT = os.environ.get("LMIRCAM_ROOT", "/Volumes/RAID36TB/LMIRCam")
NIGHTS = ["RXJ0534", "RXJ0534_2", "RXJ0534_3", "RXJ0534_4"]      # UT 2026 Feb 26, 27, 28, Mar 2
SIDES = ["left", "right"]
CUBES = [1, 2]

PXSCALE = 0.0107          # arcsec/px, reduce_lmircam.pro line 225
LAM_M = 3.8e-6            # standard L'
DIAM_M = 8.4              # one LBT primary (dual-aperture, incoherent)

PLANET = (0.390, 300.5)   # arcsec, deg E of N -- from the Table 2 offsets above


class CompanionTrace:
    """Measure the real companion at every evaluation, and never let it into the score.

    The objective is fed injected sources only, so the optimizer cannot tune itself onto
    RX J0534 b -- that is what makes its recovered S/N an honest test against the published
    5.2.  But the number is worth watching *while* the search runs: it says whether the
    configurations the injections like are the ones that actually bring the planet out, and
    it is the trace the paper needs beside the injected gain.

    So it is recorded here, on the side: one line per evaluation in ``companion_snr.jsonl``,
    and the current value shown on the live panel through the display's note line.
    """

    def __init__(self, reducer, planet, out_dir, log, display=None):
        from klip_tpe import MawetPeakSNR
        self.m = MawetPeakSNR(pxscale=reducer.pxscale, fwhm=reducer.fwhm,
                              kernel_fn=reducer.matched_filter_kernel)
        self.rho, self.pa = float(planet[0]), float(planet[1])
        self.path = os.path.join(out_dir, "companion_snr.jsonl")
        self.log, self.display, self.best = log, display, float("-inf")

    def on_eval(self, runner, record, inj, clean, is_best):
        img = getattr(clean, "image", clean)
        if img is None or np.ndim(img) != 2:
            return
        snr = float(self.m.per_source(img, None, [self.rho], [self.pa])[0])
        if not np.isfinite(snr):
            return
        with open(self.path, "a") as f:
            f.write(json.dumps({"annulus": int(runner.ia), "index": int(record.index),
                                "score": float(record.score), "companion_snr": snr}) + "\n")
        note = f"RX J0534 b (held out): S/N {snr:.2f}"
        if snr > self.best:
            self.best = snr
            note += "  <- best so far"
            self.log(f"  companion S/N {snr:.2f} at evaluation {record.index + 1} "
                     f"(published four-night stack: ~5.2)")
        if self.display is not None:
            # the panel's note line; set on the display rather than the record so the
            # objective and everything derived from it stay untouched
            self.display._live_reason = note


def _label(night, side, cube, how):
    """Partition name at the requested granularity."""
    n = NIGHTS.index(night) + 1
    if how >= 16:
        return f"n{n}{side[0]}{cube}"
    if how >= 8:
        return f"n{n}{side[0]}"
    return f"n{n}"


def _psf_for(night, side, cube):
    """Unsaturated PSF for this night/side, and the exposure time it was taken at.

    The saturated science frames carry no stellar flux, so the injection template and the
    contrast normalization both come from the unsaturated sequence taken alongside them.
    Median-combining the aligned unsaturated cube beats any single frame: LMIRCam's bad
    pixels are bright enough to outshine the core in one frame (the ``_adi`` products have
    a hot pixel above the peak), and the median removes them.
    """
    d = os.path.join(ROOT, night, "reduced-unsat")
    pat = os.path.join(d, f"{night}_{side}_cube{cube}_align_cen_bin.fits")
    if not os.path.exists(pat):                      # some nights only have one unsat cube
        alt = sorted(glob.glob(os.path.join(d, f"{night}_{side}_cube*_align_cen_bin.fits")))
        if not alt:
            return None, None
        pat = alt[0]
    cube_arr = np.asarray(fits.getdata(pat), float)
    if cube_arr.ndim == 2:
        cube_arr = cube_arr[None]
    psf = np.nanmedian(cube_arr, axis=0)
    sav = pat.replace("_cube", "_angles").replace("_align_cen_bin.fits", "_bin.sav")
    texp = _texp(sav)
    return psf, texp


def _texp(sav_path):
    """Mean per-frame exposure time from the IDL angles save file (``exptimesN``)."""
    from scipy.io import readsav
    try:
        d = readsav(sav_path, verbose=False)
    except Exception:
        return None
    for k, v in d.items():
        if k.lower().startswith("exptime"):
            a = np.asarray(v, float)
            return float(np.nanmedian(a[np.isfinite(a) & (a > 0)])) if a.size else None
    return None


def _angles(sav_path):
    from scipy.io import readsav
    d = readsav(sav_path, verbose=False)
    for k, v in d.items():
        if k.lower().startswith("angle"):
            return np.asarray(v, float).ravel()
    raise KeyError(f"no angles variable in {sav_path}")


def load(how=16, clean=False, crop_half=80, log=print):
    """The sixteen (night, side, cube) sequences, merged to the requested granularity.

    ``clean=False`` loads the *pre-selection* cubes so the optimizer's own frame-quality
    parameters have something to choose; ``clean=True`` uses the pipeline's already-cleaned
    frames, which is what the published reduction ran on.

    ``crop_half`` trims each 400x400 frame to ``2*crop_half+1`` about the star.  The search
    annulus reaches 55 px, so 80 keeps a wide margin while cutting 5,500 frames from ~7 GB
    to ~1 GB -- and the KLIP basis is built on the kept pixels, so it is the difference
    between an evaluation taking seconds and taking half a minute.  Pass 0 to keep the
    full frame.
    """
    tag = "clean_bin" if clean else "bin"
    dsets, star_flux, groups = {}, {}, {}
    for night in NIGHTS:
        for side in SIDES:
            for c in CUBES:
                base = os.path.join(ROOT, night, "reduced", f"{night}_{side}")
                cube = f"{base}_cube{c}_align_cen_{tag}.fits"
                ang = f"{base}_angles{c}_{'clean_bin' if clean else 'bin'}.sav"
                if not (os.path.exists(cube) and os.path.exists(ang)):
                    log(f"  missing: {os.path.basename(cube)}")
                    continue
                groups.setdefault(_label(night, side, c, how), []).append((cube, ang, night, side, c))

    for pid, members in sorted(groups.items()):
        cubes, angs, psfs, texps = [], [], [], []
        for cube, ang, night, side, c in members:
            arr = np.asarray(fits.getdata(cube), float)
            a = _angles(ang)
            if a.size != arr.shape[0]:
                raise ValueError(f"{os.path.basename(cube)}: {arr.shape[0]} frames, {a.size} angles")
            cubes.append(arr)
            angs.append(a)
            p, t_un = _psf_for(night, side, c)
            t_sci = _texp(ang)
            if p is not None and t_un and t_sci:
                psfs.append(p)
                # star flux in science-frame units: the unsaturated PSF scaled by the
                # exposure-time ratio, which is what the saturated core would have held
                texps.append(float(p[np.isfinite(p) & (p > 0)].sum()) * (t_sci / t_un))
        arr = np.concatenate(cubes, axis=0)
        a = np.concatenate(angs)
        psf = None if not psfs else np.nanmedian(np.stack([q / np.nansum(q) for q in psfs]), axis=0)
        # clamp rather than refuse: the crop is a convenience, and a frame smaller than the
        # requested half-width should simply be kept whole
        h = min(int(crop_half or 0), (min(arr.shape[1:]) - 1) // 2)
        ds = generic.load_cube(arr, a, psf=psf, name=pid, crop_half=h or None)
        dsets[pid] = ds
        if texps:
            star_flux[pid] = float(np.mean(texps))
        log(f"  {pid}: {arr.shape[0]} frames, {a.max() - a.min():.1f} deg rotation"
            f"{'' if pid not in star_flux else f', star flux {star_flux[pid]:.3g}'}")
    if not dsets:
        raise SystemExit(f"no RX J0534 cubes found under {ROOT} -- set LMIRCAM_ROOT")
    return dsets, (star_flux or None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--partitions", type=int, default=16, choices=[4, 8, 16],
                    help="4 = one per night, 8 = night x side, 16 = night x side x cube (default)")
    ap.add_argument("--n-iter", type=int, default=5000,
                    help="evaluations per annulus (default: 5000)")
    ap.add_argument("--n-init", type=int, default=None,
                    help="default: 15%% of --n-iter, capped at 400 -- a long run does not need a "
                         "proportionally longer random warm-up")
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--max-drop", type=int, default=3, help="partition-exclusion slots")
    ap.add_argument("--bin-min", type=int, default=5,
                    help="fewest frames per temporal bin (NEAR's production minimum). The "
                         "generic search space allows bin=1, which on a 1500-frame night is "
                         "no binning at all: measured here, bin>=10 evaluated in ~25 s and "
                         "bin=1 in ~8300 s. 1 removes the floor.")
    ap.add_argument("--ann", type=float, nargs=2, default=None, metavar=("IN", "OUT"),
                    help="annulus in pixels; the default is the companion's separation "
                         "+/- 2 FWHM, which puts it exactly at the mid-radius the injections use")
    ap.add_argument("--ann-fwhm", type=float, default=2.0, metavar="N",
                    help="half-width of the default annulus in FWHM (default 2)")
    # 3, not 2: this is LMIRCam.  The NIRCam limit of two (a small coronagraphic field, where
    # simultaneous sources perturb the basis each other sees) was applied here by mistake on
    # 2026-09-22 (471c017) and is a NIRCam setting only -- run_demos D / H2 carry it.
    ap.add_argument("--n-sources", type=int, default=3, metavar="N",
                    help="injected sources per evaluation (default 3). They share the ring with "
                         "the real planet, and every source on it blanks part of the noise "
                         "annulus the S/N is measured against")
    ap.add_argument("--allow-offset-ann", action="store_true",
                    help="proceed even when the injections do not land on the companion's "
                         "separation. Only for a deliberately wide band (a contrast curve over "
                         "a range); for a detection run it means optimizing the wrong radius")
    ap.add_argument("--clean", action="store_true", help="use the pipeline's cleaned frames")
    ap.add_argument("--crop", type=int, default=80, metavar="HALF",
                    help="crop each frame to 2*HALF+1 about the star (default 80; 0 = full frame). "
                         "The annulus reaches 55 px, so 80 is a wide margin and cuts ~7 GB to ~1 GB")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--workers", default="auto")
    ap.add_argument("--default-only", action="store_true",
                    help="reduce once at the published settings and stop (no search)")
    ap.add_argument("--resume", action="store_true",
                    help="require that there is a run in --out to continue (a check-pointed "
                         "run is continued automatically either way). Its search configuration "
                         "comes from the checkpoint, so options changed since it started are "
                         "reported and not applied")
    ap.add_argument("--fresh", action="store_true",
                    help="start a new search even if --out holds a check-pointed run")
    ap.add_argument("--show", nargs="?", const="window", default=False, metavar="MODE",
                    help="live window: --show for a matplotlib window, --show inline to update "
                         "a Jupyter output cell in place, --show auto to pick by context")
    a = ap.parse_args(argv)

    out = a.out or os.path.join(os.getcwd(), f"rxj0534_p{a.partitions}")
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "run.log"), "a", buffering=1)

    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True)
        logf.write(line + "\n")

    log(f"RX J0534 b at rho={PLANET[0]}\", PA={PLANET[1]} deg "
        f"({PLANET[0] / PXSCALE:.1f} px); published four-night stack: S/N ~ 5.2")
    log(f"loading {a.partitions} partitions from {ROOT} ({'cleaned' if a.clean else 'pre-selection'} frames"
        f"{'' if not a.crop else f', cropped to {2 * a.crop + 1} px'})")
    dsets, star_flux = load(a.partitions, clean=a.clean, crop_half=a.crop, log=log)

    # what the panels should call a partition: at 4 they really are the four nights, at 8
    # and 16 they are sub-sequences within them, and the landscape's title says so
    label = "night" if a.partitions == 4 else "sequence"
    red = generic.make_reducer(dsets, pxscale=PXSCALE, lam_m=LAM_M, diam_m=DIAM_M,
                               star_flux=star_flux, max_workers=a.workers, log=log,
                               partition_label=label)
    log(f"lambda/D = {(LAM_M / DIAM_M) * 206265.0 / PXSCALE:.2f} px, FWHM = {red.fwhm:.2f} px")

    # The annulus is built around the companion rather than chosen: +/- N FWHM puts its
    # separation exactly at the band's mid-radius, which is where 'fixed_pa' places every
    # injected source.  So the objective is measured on the companion's own ring -- the
    # separation the result is about -- instead of being diluted across a wide band.
    r_px = PLANET[0] / PXSCALE
    half = a.ann_fwhm * red.fwhm
    ann = tuple(a.ann) if a.ann else (max(r_px - half, 1.0), r_px + half)
    mid = 0.5 * (ann[0] + ann[1])
    log(f"annulus {ann[0]:.1f}-{ann[1]:.1f} px ({ann[0] * PXSCALE:.3f}-{ann[1] * PXSCALE:.3f}\"), "
        f"mid-radius {mid:.1f} px vs companion at {r_px:.1f} px")
    if abs(mid - r_px) > 0.5 * red.fwhm:
        log(f"  note: injections will sit at {mid * PXSCALE:.3f}\", not the companion's "
            f"{PLANET[0]:.3f}\" -- pass --ann symmetric about {r_px:.1f} px to match it")

    # The companion is 'known': injections keep clear of it, it is masked out of the noise
    # ring, and it never enters the score -- that is what makes its recovered S/N a real
    # test against the published 5.2.  'fixed_pa' then puts every injected source on its
    # separation ring at PAs spread around it, as optimize_lmircam_tpe.pro's annmode does.
    # The RADIUS is what is fixed for the annulus; the PAs are re-drawn every evaluation, as
    # near2m_randpos does ("so sources still rotate eval-to-eval (anti-gaming)").
    obj, samp = generic.default_config(red, known=[PLANET])
    samp.strategy = "fixed_pa"

    n_init = a.n_init or min(max(int(0.15 * a.n_iter), 40), 400)
    # bin_range floors the space for a NEW run; the guard floors it for a RESUMED one, whose
    # parameter bounds come from its checkpoint and cannot change.  Both, so a run started
    # today and a run rescued mid-flight search the same thing.
    b_hi = max(int(min(d.nframes for d in dsets.values()) // 8), a.bin_min + 1)
    space = generic.make_space(red, k_klip_max=a.k_max, max_drop=a.max_drop,
                               bin_range=(a.bin_min, b_hi))
    space.project = generic.make_guard(red, k_max=a.k_max, n_min_ref=5, bin_min=a.bin_min)
    log(f"temporal bin searched over {a.bin_min}-{b_hi} frames "
        f"(--bin-min {a.bin_min}; below ~5 a single evaluation costs hours, not seconds)")
    log(f"search space: {space.ndim} dimensions "
        f"({len(dsets)} partitions x per-partition block + {a.max_drop} exclusion slots)")

    # --default-only walks the identical path with a budget of one, so the first (seeded
    # default) evaluation is the whole run: same calibration, same metric, same validation.
    # That makes 'default -> optimized' one comparison rather than two pipelines.
    n_iter, n_init = (1, 1) if a.default_only else (a.n_iter, n_init)
    cfg = RunConfig(ann_edges=[ann[0], ann[1]], n_iter=n_iter, n_init=n_init, seed=a.seed,
                    n_sources=a.n_sources,
                    # NEAR's two-source convention collapses the band to the annulus'
                    # AREA-weighted mid radius, which for a band centred on the companion
                    # sits 4 px outside it -- the injections would leave its ring exactly
                    # when --n-sources 2 is chosen.  Here the band is already collapsed onto
                    # that ring on purpose, so the convention has nothing to add.
                    pair_area_midpoint=False,
                    validation=ValidationConfig(n_top=1 if a.default_only else 6,
                                                n_valid=10),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=3),
                    defaults={"k_klip": 5},          # what the published reduction used
                    fm_curve=True, verify=not a.default_only, save_eval_images=False)

    runner_probe = Runner(red, space, obj, samp, cfg, out, log=lambda s: None)
    runner_probe.ia = 0
    placed = runner_probe.search_sources(0, cfg.contrast0)      # one representative draw
    log(f"injected sources ({len(placed)}; these separations hold for every evaluation of "
        f"this annulus, the position angles are re-drawn each time):")
    for s in placed:
        log(f"    rho = {s.rho:.4f}\" ({s.rho / PXSCALE:5.2f} px)   PA = {s.theta:6.1f} deg"
            f"   {abs((s.theta - PLANET[1] + 180) % 360 - 180):5.1f} deg from the companion")
    # Everything on the ring costs noise apertures: the S/N of each source is measured
    # against the others' ring, and each source blanks ~3 FWHM of arc around itself.  With
    # the real planet there too, one injection too many quietly thins the statistic the
    # whole objective rests on.
    nap = int(2 * np.pi * (PLANET[0] / PXSCALE) / red.fwhm)
    n_on_ring = len(placed) + 1                      # + RX J0534 b
    left = max(nap - 3 * n_on_ring, 1)
    log(f"  ring holds {nap} noise apertures; {n_on_ring} sources on it ({len(placed)} injected "
        f"+ the planet) leave ~{left} for the noise estimate "
        f"(small-sample penalty x{np.sqrt(1 + 1 / left):.3f})")
    if left < 8:
        log(f"  WARNING: only ~{left} independent noise apertures -- consider fewer injections "
            f"(--n-sources) or a wider annulus")
    off = max(abs(s.rho - PLANET[0]) for s in placed) / PXSCALE
    if off > 0.25 and not a.allow_offset_ann:
        raise SystemExit(
            f"the injections are {off:.1f} px off the companion's separation "
            f"({PLANET[0] / PXSCALE:.2f} px) -- with --ann given, pass a band symmetric about it, "
            f"drop --ann to let the companion's separation set it, or pass --allow-offset-ann "
            f"if a wide band is what you want")
    if off > 0.25:
        log(f"  WARNING: {off:.1f} px off the companion's separation, allowed by "
            f"--allow-offset-ann -- this run optimizes {placed[0].rho:.3f}\", not "
            f"{PLANET[0]:.3f}\"")
    else:
        log(f"  all within {off:.2f} px of the companion at {PLANET[0] / PXSCALE:.2f} px -- "
            f"the objective is measured on its own ring, at whatever angles each "
            f"evaluation draws")

    callbacks, disp = [], None
    if a.show:
        from klip_tpe.display import LiveDisplay
        # show= is what opens the window; without it the display writes panels to steps/
        # and nothing appears on screen, which looks exactly like a broken display
        disp = LiveDisplay(out, every=10, pdf_every=0, movie=False, dpi=100, show=a.show)
        callbacks = [disp]
        log(f"live window: {a.show}  (a second, freeze-proof view: klip-tpe view --run-dir {out})")
    # watch the real companion alongside the search without ever scoring it
    callbacks.append(CompanionTrace(red, PLANET, out, log, display=disp))

    # A check-pointed run in --out is continued automatically (Runner.resume_mode); --fresh
    # starts over, --resume insists there is something to continue.
    from klip_tpe.runner import checkpoint_candidates
    if a.resume and not checkpoint_candidates(out):
        raise SystemExit(f"nothing to resume in {out} (no checkpoint.json) -- "
                         f"drop --resume to start it")
    runner = Runner(red, space, obj, samp, cfg, out, log=log, callbacks=callbacks,
                    resume="never" if a.fresh else "auto")
    results = runner.run()
    if runner._resumed:
        # the configuration came from the checkpoint, by design -- so anything changed on
        # this command line since the run started is NOT in force, and must not look as if
        # it were
        rc = runner.cfg
        log(f"  continued with the recorded settings: n_iter={rc.n_iter}  "
            f"n_sources={rc.n_sources}  ann_edges={rc.ann_edges}  "
            f"fixed_sources={rc.fixed_sources}")
        for name, now, then in (("--n-iter", a.n_iter, rc.n_iter),
                                ("--n-sources", a.n_sources, rc.n_sources)):
            if then is not None and now != then and not (isinstance(then, list) and now in then):
                log(f"  note: {name}={now} was given, but the run continues with {then} -- "
                    f"use --fresh, or a new --out, to change it")

    # the held-out test: the companion's own S/N, which the objective never scored
    try:
        from klip_tpe import MawetPeakSNR
        m = MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel)
        for r in results:
            img = getattr(r, "clean_image", None)
            if img is not None:
                snr = float(m.per_source(img, None, [PLANET[0]], [PLANET[1]])[0])
                log(f"annulus {r.annulus + 1}: RX J0534 b recovered at S/N = {snr:.2f} "
                    f"(published four-night stack: ~5.2)")
    except Exception as exc:
        log(f"companion S/N not measured here ({exc!r}); collect.py computes it from the products")
    log(f"done -- products in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
