#!/usr/bin/env python
"""Paper-quality runs of klip-tpe on the three public data sets + the search benchmark.

    python run_demos.py A      # beta Pic, 3 annuli, single partition
    python run_demos.py B      # beta Pic split into 4 time groups (partition selection)
    python run_demos.py C      # HD 95086 IRDIS K1+K2, 2 annuli
    python run_demos.py D      # HIP 65426 NIRCam F444W (RDI, searched mode)
    python run_demos.py E      # benchmark: TPE / random / grid at matched budget (beta Pic)
    python run_demos.py G2     # the same benchmark on HD 95086  (SPHERE K1+K2, 20-D)
    python run_demos.py H2     # the same benchmark on HIP 65426 (JWST 2 rolls,  11-D)

``WORKERS=6 python run_demos.py G2`` caps the core budget; the default is every core.

Everything lands in ``/home/claude/paper_runs/<name>/``.
"""
import os
import sys
import time

import numpy as np
from astropy.io import fits

from klip_tpe import (CalibrationConfig, Param, RunConfig, Runner, ValidationConfig, datasets)
from klip_tpe.instruments import generic
from klip_tpe.reducer import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
BP = (0.452, 211.9)          # beta Pic b

#: beta Pic's debris disk, as azimuthal wedges (centre PA, half-width) in degrees.
#: Measured on the default reduction of this cube rather than taken from the literature: the
#: azimuthal excess across the 8-22 px annulus peaks at PA 28-38 and 208-222, which is the
#: catalogued near-edge-on axis.  +/-20 deg covers both lobes.
#:
#: What it is worth, measured on the same reduction, so the paper can say so: the disk bands
#: carry about +0.2 sigma of median excess over the rest of the ring, and cutting them costs
#: ~22% of the noise apertures.  At the innermost injection radius (12.7 px) the disk does
#: supply 26% of the ring scatter -- but comparable excesses sit at PA 244, 354 and 92, which
#: are speckles and stay.  The cut is defensible because the disk is known in advance and a
#: speckle is not; it is not a cure for the azimuthal structure in this field.
BP_DISK = [(29.0, 20.0), (209.0, 20.0)]


def bp_disk(reducer=None, shape=None):
    """``(forbidden_pa, pixel_mask)`` for beta Pic, or ``((), None)`` when DISK_CUT=0.

    Both halves or neither: injections that avoid a disk which still inflates the ring sigma
    are measured against a noise level nothing is scoring them at.
    """
    if os.environ.get("DISK_CUT", "1").strip() in ("0", "off", "no", ""):
        return (), None
    from klip_tpe.metrics import pa_wedge_mask
    if shape is None and reducer is not None:
        r0 = next(iter(reducer.reducers.values()))
        shape = r0.data.cube.shape[-2:]
    return BP_DISK, (None if shape is None else pa_wedge_mask(tuple(shape), BP_DISK))
HD = (0.62, 145.0)           # HD 95086 b
HIP = (0.826, 150.2)         # HIP 65426 b


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def show_mode():
    """The live-window mode for this stage, from ``$SHOW`` (default ``window``; ``0`` off).

    Every stage here already built a ``LiveDisplay`` -- but without ``show=``, which writes
    the panels to ``steps/`` and opens nothing.  That is indistinguishable from a broken
    display, and it is the same mistake the RX J0534 driver made.  The benchmarks had no
    display at all; they get one too, and because the window lives on the LiveDisplay class
    it is a single window that follows whichever slot is running rather than one per slot.
    """
    v = os.environ.get("SHOW", "window").strip() or "window"
    return None if v in ("0", "off", "no", "none", "") else v


def _display(d, **kw):
    from klip_tpe.display import LiveDisplay
    kw.setdefault("every", 10)
    return LiveDisplay(d, pdf_every=0, movie=False, dpi=100, show=show_mode(), **kw)


def workers():
    """The core budget for this stage, from ``$WORKERS`` (``rerun_paper.sh`` exports it).

    Every reducer here used to say ``max_workers=workers()``, which means *every core on the
    machine*.  ``rerun_paper.sh`` has always exported WORKERS and nothing has ever read it,
    so ``WORKERS=4 ./rerun_paper.sh bench`` ran on all 32 -- which is only correct when the
    stage is the only thing on the machine, and it never is: four searches at 32 workers
    each took single reductions from 6 s to 729 s and killed one run outright.
    """
    w = os.environ.get("WORKERS", "auto").strip() or "auto"
    return w if w in ("auto", "max", "all") else int(w)


# ---------------------------------------------------------------- beta Pic
def betapic_dataset(groups=1):
    f = datasets.fetch("naco_betapic", quiet=True)
    inst = datasets.INSTRUMENT["naco_betapic"]
    ds = generic.load_cube(f["cube"], f["angles"], psf=f["psf"], name="betapic")
    # The distributed template is NORMALISED (flux 1.000000000 inside r = 2.000 px), so its
    # own sum -- 4.349, what star_flux=None would use -- is not beta Pic's brightness and
    # leaves the axis 9.4e5 from a contrast.  datasets.PHOTOMETRY carries VIP's published
    # starphot for this very cube, in that same aperture; converting it to the template's
    # normalisation gives 3.327e6, and beta Pic b then measures dL' = 7.79 against Absil et
    # al. (2013)'s 8.01 +/- 0.16 from these same data (scripts/check_betapic_contrast.py).
    # The halo fit that used to stand here was invalid on coronagraphic data and returned
    # fluxes 5.28x and 8.04x apart on runs A2 and B2.
    p = datasets.PHOTOMETRY["naco_betapic"]
    sf = generic.star_flux_from_aperture_photometry(ds.meta["psf"], p["starphot"], p["aperture_px"])
    if groups <= 1:
        return {"betapic": ds}, sf, inst
    n = ds.cube.shape[0]
    edges = np.linspace(0, n, groups + 1).astype(int)
    out = {}
    for k in range(groups):
        sl = slice(edges[k], edges[k + 1])
        out[f"g{k + 1}"] = Dataset(ds.cube[sl], ds.angles[sl],
                                   {kk: v[sl] for kk, v in (ds.tags or {}).items()} or None,
                                   texp=float(edges[k + 1] - edges[k]), name=f"g{k + 1}",
                                   meta=dict(ds.meta))
    return out, sf, inst


def run_A():
    dsets, sf, inst = betapic_dataset(1)
    red = generic.make_reducer(dsets, star_flux=sf, max_workers=workers(), log=log, **inst)
    space = generic.make_space(red, k_klip_max=30)
    space.project = generic.make_guard(red, k_max=30)
    fpa, pmask = bp_disk(red)
    obj, samp = generic.default_config(red, known=[BP], forbidden_pa=fpa, pixel_mask=pmask)
    cfg = RunConfig(ann_edges=[8, 16, 26, 40], n_iter=[400, 300, 300], n_init=[80, 60, 60], seed=11,
                    validation=ValidationConfig(n_top=3, n_valid=5), n_sources=3,
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=3),
                    defaults={"k_klip": 10}, fm_curve=True, verify=True, param_verify=True,
                    save_eval_images=False)
    d = os.path.join(OUT, "A_betapic")
    Runner(red, space, obj, samp, cfg, d, log=log,
           callbacks=[_display(d)]).run()


def run_A2():
    """Run A with annulus edges that keep beta Pic b (16.6 px) well inside one annulus
    rather than on a seam, so the real companion is an honest independent check."""
    dsets, sf, inst = betapic_dataset(1)
    red = generic.make_reducer(dsets, star_flux=sf, max_workers=workers(), log=log, **inst)
    space = generic.make_space(red, k_klip_max=30)
    space.project = generic.make_guard(red, k_max=30)
    fpa, pmask = bp_disk(red)
    obj, samp = generic.default_config(red, known=[BP], forbidden_pa=fpa, pixel_mask=pmask)
    cfg = RunConfig(ann_edges=[6, 12, 24, 40], n_iter=[400, 300, 300], n_init=[80, 60, 60], seed=11,
                    validation=ValidationConfig(n_top=3, n_valid=5), n_sources=3,
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=3),
                    defaults={"k_klip": 10}, fm_curve=True, verify=True, param_verify=True,
                    save_eval_images=False)
    d = os.path.join(OUT, "A2_betapic")
    Runner(red, space, obj, samp, cfg, d, log=log,
           callbacks=[_display(d)]).run()


def run_B():
    dsets, sf, inst = betapic_dataset(4)
    red = generic.make_reducer(dsets, star_flux=sf, max_workers=workers(), log=log,
                               partition_label="group", **inst)
    space = generic.make_space(red, k_klip_max=12, max_drop=2)
    space.project = generic.make_guard(red, k_max=12, n_min_ref=5)
    fpa, pmask = bp_disk(red)
    obj, samp = generic.default_config(red, known=[BP], forbidden_pa=fpa, pixel_mask=pmask)
    cfg = RunConfig(ann_edges=[8, 22], n_iter=400, n_init=80, seed=12, n_sources=3,
                    validation=ValidationConfig(n_top=3, n_valid=5),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=3),
                    defaults={"k_klip": 5}, fm_curve=True, save_eval_images=False)
    d = os.path.join(OUT, os.environ.get("B_DIR", "B_betapic_groups"))
    Runner(red, space, obj, samp, cfg, d, log=log,
           callbacks=[_display(d)]).run()


# ---------------------------------------------------------------- HD 95086
def hd95086_objects():
    files = datasets.fetch("sphere_hd95086", quiet=True)
    inst = datasets.INSTRUMENT["sphere_hd95086"]
    ang = fits.getdata(files["angles"])
    dsets, sf, lam = {}, {}, {}
    # The flux frames are ALREADY on the science frames' scale in this distribution: the DIT
    # ratio and the ND transmission went in when the products were made.  Applying
    # dit_science/dit_flux/nd_transmission here as well over-counted the star by 1347x and
    # moved the whole contrast axis of runs C and G2 with it.  See tutorial 02 for the
    # arithmetic against HD 95086 b's published contrast.
    for b in ("K1", "K2"):
        ds = generic.load_cube(files[f"cube_{b}"], ang, psf=files[f"psf_{b}"], name=b)
        dsets[b] = ds
        p = ds.meta["psf"]
        sf[b] = float(p[p > 0].sum())
        lam[b] = inst["lam_m"] if b == "K1" else inst["lam_m_K2"]
    red = generic.make_reducer(dsets, pxscale=inst["pxscale"], lam_m=lam, diam_m=inst["diam_m"],
                               star_flux=sf, max_workers=workers(), log=log, partition_label="channel")
    return red


def run_C():
    red = hd95086_objects()
    space = generic.make_space(red, k_klip_max=30)
    space.project = generic.make_guard(red, k_max=30)
    obj, samp = generic.default_config(red, known=[HD])
    cfg = RunConfig(ann_edges=[20, 45, 75], n_iter=[350, 300], n_init=[70, 60], seed=13, n_sources=3,
                    validation=ValidationConfig(n_top=3, n_valid=5),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=3),
                    defaults={"k_klip": 10}, fm_curve=True, verify=True, save_eval_images=False)
    d = os.path.join(OUT, "C_hd95086")
    Runner(red, space, obj, samp, cfg, d, log=log,
           callbacks=[_display(d)]).run()


# ---------------------------------------------------------------- HIP 65426 (JWST)
def hip65426_objects():
    """RDI reducer for the ERS 1386 F444W rolls, on an absolute contrast axis.

    The star cannot be measured off these frames -- HIP 65426 and the reference star phi Cen
    are both behind MASK335R in every exposure -- so both the flux scale and the star's
    position are imported.  ``datasets.PHOTOMETRY['hip65426_f444w']`` carries them with their
    provenance and the check that pins them; ``docs/FLUX_CALIBRATION.md`` has the reasoning,
    including why the occulter's T(rho) is deliberately NOT part of the star flux.
    """
    import glob
    from klip_tpe import stpsf_psf
    from klip_tpe.backends import spaceklip as sk
    D = os.path.expanduser("~/.klip_tpe/data/jwst_hip65426")
    files = sorted(glob.glob(os.path.join(D, "jw*calints.fits")))
    if not stpsf_psf.have_stpsf():
        raise RuntimeError(
            "runs D and H2 need STPSF and its data files (pip install stpsf; export "
            "STPSF_PATH=<stpsf-data>).  There is no fallback on purpose: without the off-axis "
            "grid this path used GaussianPSF(star_flux=1.0) and called raw detector units a "
            "contrast, which is how run D shipped an axis 2.3e5 off.  See "
            "docs/FLUX_CALIBRATION.md.")
    phot = datasets.PHOTOMETRY["hip65426_f444w"]
    dsets, info = sk.load_calints(files, science_target="HIP65426",
                                  star_center=tuple(phot["star_center"]), log=log)
    # Off-axis PSF of the actual mask on a ladder of separations, with the mask throughput
    # measured from the same grid.  A Gaussian is the wrong shape AND the wrong scale here:
    # it needs contrast 40 to reach the peak the real PSF reaches at 320.
    grid = stpsf_psf.offaxis_grid("NIRCam", info["filter"] or "F444W", image_mask="MASK335R",
                                 seps_as=np.arange(0.2, 3.01, 0.2), stamp_px=41, nlambda=3,
                                 log=log)
    sf = stpsf_psf.star_flux_from_flux_density(
        grid, phot["flux_density_jy"], info["pixar_sr"], bunit=info["bunit"] or "MJy/sr",
        optics_transmission=phot["optics_transmission"], log=log)
    model = stpsf_psf.library(grid, star_flux=sf)
    red = sk.make_reducer(dsets, injection_model=model, mode="RDI", max_workers=workers(), log=log)
    return red


def run_D():
    red = hip65426_objects()
    space = generic.make_space(red, k_klip_max=18, search_angles=False)
    space.add(Param("mode", 0, 2, "categorical", choices=["ADI", "RDI", "ADI+RDI"], default="RDI",
                    doc="pyKLIP PSF-subtraction mode"))
    space.project = generic.make_guard(red, k_max=18, n_min_ref=4)
    obj, samp = generic.default_config(red, known=[HIP])
    cfg = RunConfig(ann_edges=[6, 20, 45], n_iter=[200, 150], n_init=[40, 30], seed=14, n_sources=4,
                    validation=ValidationConfig(n_top=3, n_valid=5),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=3),
                    defaults={"k_klip": 10}, fm_curve=False, save_eval_images=False)
    d = os.path.join(OUT, "D_hip65426")
    Runner(red, space, obj, samp, cfg, d, log=log,
           callbacks=[_display(d)]).run()


# ---------------------------------------------------------------- benchmark
def run_E():
    from klip_tpe.bench import run_benchmark
    dsets, sf, inst = betapic_dataset(1)
    red = generic.make_reducer(dsets, star_flux=sf, max_workers=workers(), log=lambda s: None, **inst)
    fpa, pmask = bp_disk(red)
    obj, samp = generic.default_config(red, known=[BP], forbidden_pa=fpa, pixel_mask=pmask)

    def make_runner(mode, seed, run_dir):
        space = generic.make_space(red, k_klip_max=30)
        space.project = generic.make_guard(red, k_max=30)
        cfg = RunConfig(ann_edges=[8, 22], n_iter=300, n_init=60, seed=seed, search_mode=mode, n_sources=3,
                        validation=ValidationConfig(n_top=3, n_valid=5),
                        calibration=CalibrationConfig(forced=[1.31e-3]),      # the contrast run A calibrated
                        defaults={"k_klip": 10}, fm_curve=False, save_fits=False, save_eval_images=False,
                        write_setup_files=False)
        return Runner(red, space, obj, samp, cfg, run_dir, log=lambda s: None)

    run_benchmark(make_runner, modes=("tpe", "random", "grid"), seeds=(0, 1, 2, 3, 4),
                  n_iter=300, n_init=60, out_dir=os.path.join(OUT, "E_bench"), log=log)


def run_F():
    """The same benchmark at HIGH dimension: the 4-group beta Pic split (38 searched
    dimensions with the two partition-exclusion slots), TPE vs random only -- a coarse
    grid is not a meaningful strategy in 38 dimensions."""
    from klip_tpe.bench import run_benchmark
    dsets, sf, inst = betapic_dataset(4)
    red = generic.make_reducer(dsets, star_flux=sf, max_workers=workers(), log=lambda s: None,
                               partition_label="group", **inst)
    fpa, pmask = bp_disk(red)
    obj, samp = generic.default_config(red, known=[BP], forbidden_pa=fpa, pixel_mask=pmask)

    def make_runner(mode, seed, run_dir):
        space = generic.make_space(red, k_klip_max=12, max_drop=2)
        space.project = generic.make_guard(red, k_max=12, n_min_ref=5)
        cfg = RunConfig(ann_edges=[8, 22], n_iter=300, n_init=60, seed=seed, search_mode=mode,
                        n_sources=3, validation=ValidationConfig(n_top=3, n_valid=5),
                        calibration=CalibrationConfig(forced=[2.087e-3]),   # run B's calibration
                        defaults={"k_klip": 5}, fm_curve=False, save_fits=False,
                        save_eval_images=False, write_setup_files=False)
        return Runner(red, space, obj, samp, cfg, run_dir, log=lambda s: None)

    run_benchmark(make_runner, modes=("tpe", "random"), seeds=(0, 1, 2, 3, 4),
                  n_iter=300, n_init=60, out_dir=os.path.join(OUT, "F_bench_highdim"), log=log)


def _bench_hi(tag, groups, modes, k_max, max_drop, defaults, forced, ann_edges, out, seeds=range(8),
              n_iter=800, n_init=80, n_top=8, n_valid=15, make_red=None, known=None, n_sources=3,
              add_params=None, search_angles=True, n_min_ref=5):
    """The benchmark again, with enough statistical power to settle it.

    Run F reported TPE behind random after validation (9.34 +/- 0.57 vs 9.64 +/- 0.53,
    paired -0.30 +/- 0.14) and the first reading was that it had not searched long enough.
    The traces say otherwise: TPE's running best had plateaued by ~eval 180 (+0.04 over the
    last 100) while *random* was still climbing (+0.44, one seed's best at eval 290), so the
    budget was short for random, not for TPE.

    What was actually short was the validation.  A single validated score carries a
    trial-to-trial sd of 1.39; with 5 trials that is +/-0.62 per candidate, twice the
    difference being reported, and across the 3 candidates per seed the rank correlation
    between search score and validated score was +0.00 (TPE) and -0.50 (random) -- the
    search score barely predicted the validated one at all.  So this run gives random a
    budget it can finish (800), keeps the same for TPE, triples the validated candidates
    and triples the trials each, and adds three more seeds to the paired test.
    """
    from klip_tpe.bench import run_benchmark
    if make_red is None:                      # the default subject: beta Pic, as runs E2/F2
        def make_red():
            dsets, sf, inst = betapic_dataset(groups)
            return generic.make_reducer(dsets, star_flux=sf, max_workers=workers(), log=lambda s: None,
                                        partition_label="group" if groups > 1 else "dataset", **inst)
    red = make_red()
    kn = list(known if known is not None else [BP])
    fpa, pmask = bp_disk(red) if kn == [BP] else ((), None)     # only beta Pic has the disk
    obj, samp = generic.default_config(red, known=kn, forbidden_pa=fpa, pixel_mask=pmask)

    def make_runner(mode, seed, run_dir):
        # max_drop=None: leave make_space's own default, which is what the matching science
        # run used -- the benchmark should search the problem that run actually solved
        kw = {} if max_drop is None else {"max_drop": max_drop}
        space = generic.make_space(red, k_klip_max=k_max, search_angles=search_angles, **kw)
        for pr in (add_params or ()):
            space.add(pr())
        space.project = generic.make_guard(red, k_max=k_max, n_min_ref=n_min_ref)
        cfg = RunConfig(ann_edges=ann_edges, n_iter=n_iter, n_init=n_init, seed=seed, search_mode=mode,
                        n_sources=n_sources, validation=ValidationConfig(n_top=n_top, n_valid=n_valid),
                        calibration=CalibrationConfig(forced=[forced]), defaults=defaults,
                        fm_curve=False, save_fits=False, save_eval_images=False, write_setup_files=False)
        # a display per slot, all blitting into the one shared window (see LiveDisplay._win):
        # a benchmark is the run you most want to watch and the one that had no window at all
        cbs = [_display(run_dir, every=25)] if show_mode() else []
        return Runner(red, space, obj, samp, cfg, run_dir, log=lambda s: None, callbacks=cbs)

    # reuse the batch already in this directory, so relaunching after an interruption
    # resumes the check-pointed slots instead of starting a fresh batch beside them
    d = os.path.join(OUT, out)
    tag_file = os.path.join(d, "bench_tag.txt")
    tag = open(tag_file).read().strip() if os.path.exists(tag_file) else None
    if tag:
        log(f"resuming batch {tag} in {out}")
    run_benchmark(make_runner, modes=modes, seeds=tuple(seeds), n_iter=n_iter, n_init=n_init,
                  bench_tag=tag, out_dir=d, log=log)


def run_B2():
    """The beta Pic 4-group run, into its own directory.

    This was only ever reachable as ``B_DIR=B2_betapic_groups python3 run_demos.py B``,
    which is how a rerun driver asking for "B2" got a KeyError.  The paper's table names it
    B2, so it is a stage like the rest.
    """
    os.environ.setdefault("B_DIR", "B2_betapic_groups")
    run_B()


def run_F2():
    """Run F at 800 evaluations, 8 validated candidates x 15 trials, 8 seeds (38-D)."""
    _bench_hi("F2", 4, ("tpe", "random"), 12, 2, {"k_klip": 5}, 2.087e-3, [8, 22], "F2_bench_highdim")


def run_E2():
    """Run E at the same power (9-D), so the two dimensionalities stay comparable."""
    _bench_hi("E2", 1, ("tpe", "random", "grid"), 30, 0, {"k_klip": 10}, 1.31e-3, [8, 22], "E2_bench")


# -- the same benchmark on the other two data sets --------------------------------------
#
# E2 and F2 are both beta Pic, and beta Pic has a bright, near-edge-on debris disk running
# straight through the [8, 22] px annulus they search.  The injected sources sit on a ring
# that contains disk flux, so the noise term of every S/N in those two runs is measured
# against a field that is not empty.
#
# That is a caveat on the *absolute* numbers, not on the comparison.  A seed fixes the
# injection RADII for the annulus and seeds the whole run, and every mode runs the same
# seeds, so tpe seed 0 and random seed 0 start from an identical seeded default -- visible in
# the bench summary, where those scores agree to four decimals.  The position ANGLES rotate
# per evaluation in both arms, as the reference does, so the disk's contribution is a
# common-mode term in both and largely cancels in the paired difference.
#
# Still, one target is one target.  G2 and H2 put the same protocol on two fields with no
# scattered-light disk in them, and in doing so give the benchmark a dimensionality ladder
# (9, 11, 20, 38) across three instruments and both ground and space, so "does TPE beat
# random?" can be answered as a trend rather than as an anecdote.
def run_G2():
    """HD 95086, SPHERE/IRDIS K1+K2 (20-D): two channels, no disk in scattered light.

    Configured as run C's first annulus -- same space, same k cap, and C's own calibrated
    contrast -- so the benchmark searches the problem the science run actually solved.
    """
    _bench_hi("G2", 1, ("tpe", "random"), 30, None, {"k_klip": 10}, 5.899e-9, [20, 45], "G2_bench_sphere",
              make_red=hd95086_objects, known=[HD], n_sources=3, n_min_ref=10)


def run_H2():
    """HIP 65426, JWST/NIRCam F444W (11-D): two rolls, RDI against the reference library.

    The other end of the range from beta Pic -- space, two frames per roll, a searched
    ADI/RDI/ADI+RDI mode, and a field with no disk at all.  Angles are not searched (run D
    does not search them either: with two frames per roll there is nothing to select on).
    """
    _bench_hi("H2", 1, ("tpe", "random"), 18, None, {"k_klip": 10}, 5.270e1, [6, 20], "H2_bench_jwst",
              make_red=hip65426_objects, known=[HIP], n_sources=4, search_angles=False, n_min_ref=4,
              add_params=[lambda: Param("mode", 0, 2, "categorical", choices=["ADI", "RDI", "ADI+RDI"],
                                        default="RDI", doc="pyKLIP PSF-subtraction mode")])


if __name__ == "__main__":
    which = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
    t0 = time.time()
    {"A": run_A, "A2": run_A2, "B": run_B, "B2": run_B2, "C": run_C, "D": run_D,
     "E": run_E, "F": run_F, "E2": run_E2, "F2": run_F2,
     "G2": run_G2, "H2": run_H2}[which]()
    log(f"{which} done in {(time.time() - t0) / 60:.1f} min")
