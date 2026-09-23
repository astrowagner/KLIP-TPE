#!/usr/bin/env python
"""Paper-quality runs of klip-tpe on the three public data sets + the search benchmark.

    python run_demos.py A      # beta Pic, 3 annuli, single partition
    python run_demos.py B      # beta Pic split into 4 time groups (partition selection)
    python run_demos.py C      # HD 95086 IRDIS K1+K2, 2 annuli
    python run_demos.py D      # HIP 65426 NIRCam F444W (RDI, searched mode)
    python run_demos.py E      # benchmark: TPE / random / grid at matched budget (beta Pic)
    python run_demos.py G2     # the same benchmark on HD 95086  (SPHERE K1+K2, 20-D)
    python run_demos.py H2     # the same benchmark on HIP 65426 (JWST 2 rolls,   5-D)

``WORKERS=6 python run_demos.py G2`` caps the core budget; the default is every core.
``NITER=1000 python run_demos.py A2`` raises every annulus to at least that many
evaluations (``BENCH_NITER`` for the benchmark stages, ``ITER_SCALE`` to multiply each
stage's own budget); see :func:`budget`.  ``long_run.sh`` drives all of them at once.

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


def budget(n_iter, n_init, bench=False):
    """Evaluation budget for a stage, with the environment allowed to raise it.

    The budgets written into the stages below are the ones the 2026-09 paper draft used,
    and on the convergence traces (Figures~5 and~12) the running best is still climbing at
    the end of several of them -- a search reported at a budget it has not converged at
    understates every strategy, and understates the guided one most, since that is the one
    still finding things.  Rather than edit the numbers in place and lose the record of
    what the draft ran, the stages ask here and the environment decides:

      ``NITER``        per-annulus floor for the science stages (``BENCH_NITER`` for the
                       four benchmark stages); the stage runs at least this many.
      ``ITER_SCALE``   multiply each stage's own budget (so the relative weighting of a
                       wide outer annulus against a narrow inner one is preserved).

    Both may be set; the larger wins per annulus.  ``n_init`` follows at each annulus'
    own warm-up fraction -- a longer search with the original 80 random draws would spend
    a smaller fraction of its budget on the warm-up and change what is being compared.
    Unset, everything returns exactly what the stage asked for, so a plain
    ``python3 run_demos.py A2`` still reproduces the draft.
    """
    scale = float(os.environ.get("ITER_SCALE", "1") or 1)
    floor = int(os.environ.get("BENCH_NITER" if bench else "NITER", "0") or 0)
    scalar = not isinstance(n_iter, (list, tuple))
    its = [n_iter] if scalar else list(n_iter)
    ins = [n_init] * len(its) if not isinstance(n_init, (list, tuple)) else list(n_init)
    out_it, out_in = [], []
    for it, ini in zip(its, ins):
        new = max(floor, int(round(it * scale)))
        out_it.append(new)
        out_in.append(max(20, int(round(ini * new / float(it)))))
    if (out_it != its or out_in != ins):
        log(f"  budget: n_iter {its} -> {out_it}, n_init {ins} -> {out_in}"
            f"  (ITER_SCALE={scale}, {'BENCH_NITER' if bench else 'NITER'}={floor})")
    return (out_it[0], out_in[0]) if scalar else (out_it, out_in)


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
    n_iter, n_init = budget([400, 300, 300], [80, 60, 60])
    cfg = RunConfig(ann_edges=[8, 16, 26, 40], n_iter=n_iter, n_init=n_init, seed=11,
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
    n_iter, n_init = budget([400, 300, 300], [80, 60, 60])
    cfg = RunConfig(ann_edges=[6, 12, 24, 40], n_iter=n_iter, n_init=n_init, seed=11,
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
    n_iter, n_init = budget(400, 80)
    cfg = RunConfig(ann_edges=[8, 22], n_iter=n_iter, n_init=n_init, seed=12, n_sources=3,
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
    n_iter, n_init = budget([350, 300], [70, 60])
    cfg = RunConfig(ann_edges=[20, 45, 75], n_iter=n_iter, n_init=n_init, seed=13, n_sources=3,
                    validation=ValidationConfig(n_top=3, n_valid=5),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=3),
                    defaults={"k_klip": 10}, fm_curve=True, verify=True, save_eval_images=False)
    d = os.path.join(OUT, "C_hd95086")
    Runner(red, space, obj, samp, cfg, d, log=log,
           callbacks=[_display(d)]).run()


# ---------------------------------------------------------------- HIP 65426 (JWST)
def hip65426_objects(partition="all"):
    """Reducer for the ERS 1386 F444W rolls, on an absolute contrast axis.

    The star cannot be measured off these frames -- HIP 65426 and the reference star phi Cen
    are both behind MASK335R in every exposure -- so both the flux scale and the star's
    position are imported.  ``datasets.PHOTOMETRY['hip65426_f444w']`` carries them with their
    provenance and the check that pins them; ``docs/FLUX_CALIBRATION.md`` has the reasoning,
    including why the occulter's T(rho) is deliberately NOT part of the star flux.

    ``partition='all'`` (default since 2026-09-17): both rolls in ONE partition, so that
    pyKLIP's searched ``mode`` means something.  A partition is reduced on its own; with one
    per roll every frame in it shared a PA, so ADI had no reference frames and ADI+RDI was
    RDI -- the other roll was never in the basis, and on top of that pyKLIP at movement 0
    put each frame in its OWN basis (fixed in the backend).  Runs D and H2 before this date
    "elected RDI" against two modes that could not work.  With both rolls together, at the
    seeded default k = 10: injected S/N 6.2 ADI, 6.9 RDI, 7.2 ADI+RDI -- a real choice.
    ``partition='roll'`` keeps the old per-roll layout (one k block per roll, RDI only in
    effect).
    """
    import glob
    from klip_tpe import stpsf_psf
    from klip_tpe.backends import spaceklip as sk
    D = os.path.expanduser("~/.klip_tpe/data/jwst_hip65426")
    files = sorted(glob.glob(os.path.join(D, "jw*calints.fits")))
    phot = datasets.PHOTOMETRY["hip65426_f444w"]
    dsets, info = sk.load_calints(files, science_target="HIP65426", partition=partition,
                                  star_center=tuple(phot["star_center"]), log=log)
    # Off-axis PSF of the actual mask on a ladder of separations, with the mask throughput
    # measured from the same grid.  A Gaussian is the wrong shape AND the wrong scale here:
    # it needs contrast 40 to reach the peak the real PSF reaches at 320.
    #
    # STPSF itself is only needed when the grid is NOT cached -- offaxis_grid and
    # unocculted_ee both check the cache before importing it -- so a machine whose Python is
    # too old for STPSF (it wants >= 3.10) can still run this from a copied cache.  Point
    # KLIP_TPE_DATA at the directory that holds `stpsf_cache/`.  There is deliberately no
    # fallback: without the grid this path used GaussianPSF(star_flux=1.0) and called raw
    # detector units a contrast, which is how run D shipped an axis 2.3e5 off.
    try:
        grid = stpsf_psf.offaxis_grid("NIRCam", info["filter"] or "F444W", image_mask="MASK335R",
                                      seps_as=np.arange(0.2, 3.01, 0.2), stamp_px=41, nlambda=3,
                                      log=log)
        sf = stpsf_psf.star_flux_from_flux_density(
            grid, phot["flux_density_jy"], info["pixar_sr"], bunit=info["bunit"] or "MJy/sr",
            optics_transmission=phot["optics_transmission"], log=log)
    except ImportError as exc:
        raise RuntimeError(
            f"runs D and H2 need the STPSF off-axis grid and it is not in the cache "
            f"({stpsf_psf.cache_dir()}).  Either install STPSF and its data files "
            f"(pip install stpsf, Python >= 3.10, STPSF_PATH=<stpsf-data>) or copy a cached "
            f"grid there and set KLIP_TPE_DATA.  See docs/FLUX_CALIBRATION.md.  ({exc})"
        ) from exc
    model = stpsf_psf.library(grid, star_flux=sf)
    red = sk.make_reducer(dsets, injection_model=model, mode="RDI", max_workers=workers(), log=log)
    # A searched reference library rather than the fixed one, and it replaces the ADI / RDI /
    # ADI+RDI categorical: nkeep_psfref = 0 is ADI, nkeep_altroll = 0 is RDI, and unlike the
    # categorical either pool can contribute PART of itself.  This sequence has 2 science
    # frames per roll and 18 phi Cen frames, so the ranges are [0, 4] and [0, 18] with their
    # sum floored at 2 by ReferenceLibraryGuard.
    if partition == "all" and len(red.reducers) == 1:
        r0 = next(iter(red.reducers.values()))
        nref = 0 if r0.data.ref_cube is None else int(np.shape(r0.data.ref_cube)[0])
        part = np.round(np.asarray(r0.data.angles, float), 1).astype(str)
        if nref >= 2 and np.unique(part).size >= 2:
            r0.set_reference_library(partition=part, ref_group="psfref",
                                     n_min_ref=2, metric="cc")
            log(f"  library: searched -- nkeep_altroll over {part.size} science frames in "
                f"{np.unique(part).size} rolls, nkeep_psfref over {nref} reference frames")
    return red


def run_D():
    red = hip65426_objects()
    # No `mode` dimension: the library's nkeep_altroll / nkeep_psfref subsume it (0 in one
    # pool IS the corresponding pure mode) and can also take part of a pool, which the
    # categorical could not.
    space = generic.make_space(red, k_klip_max=18, search_angles=False)
    space.project = generic.make_guard(red, k_max=18, n_min_ref=4)
    obj, samp = generic.default_config(red, known=[HIP])
    n_iter, n_init = budget([200, 150], [40, 30])
    # 2, not 4: NIRCam's coronagraphic field is small enough that four sources at one
    # contrast perturb the KLIP basis each other sees.  The Mawet ring still holds 9 clean
    # apertures at 4 (measured on this run's own stitch), so it is mutual contamination
    # that sets this, not ring starvation -- n_sources_rule's packing cap would allow 4.
    cfg = RunConfig(ann_edges=[6, 20, 45], n_iter=n_iter, n_init=n_init, seed=14, n_sources=2,
                    validation=ValidationConfig(n_top=3, n_valid=5),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=3),
                    # each trial scored as the mean of 3 fresh source draws, as MIRI does.
                    # NIRCam reductions are ~2.2 s (MIRI is 22.6), so this is the cheap end of
                    # the trade that on MIRI took the objective's sd from 0.84 to 0.48.
                    n_remeasure=3,
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
                        calibration=CalibrationConfig(forced=[3.0e-4]),    # measured on the current axis, see run_E2
                        defaults={"k_klip": 10}, fm_curve=False, save_fits=False, save_eval_images=False,
                        write_setup_files=False)
        return Runner(red, space, obj, samp, cfg, run_dir, log=lambda s: None)

    run_benchmark(make_runner, modes=bench_modes(("tpe", "random", "grid")), seeds=(0, 1, 2, 3, 4),
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
                        calibration=CalibrationConfig(forced=[3.0e-4]),    # measured on the current axis, see run_F2
                        defaults={"k_klip": 5}, fm_curve=False, save_fits=False,
                        save_eval_images=False, write_setup_files=False)
        return Runner(red, space, obj, samp, cfg, run_dir, log=lambda s: None)

    run_benchmark(make_runner, modes=bench_modes(("tpe", "random")), seeds=(0, 1, 2, 3, 4),
                  n_iter=300, n_init=60, out_dir=os.path.join(OUT, "F_bench_highdim"), log=log)


def _forced_list(forced, ann_edges):
    """``CalibrationConfig.forced`` is per annulus, so a multi-annulus bench needs one
    contrast per annulus.  A scalar means "this one everywhere", which is right for the
    single-annulus benches and wrong the moment a bench grows a second annulus: the
    calibrated contrast is a property of the zone, not of the target."""
    nann = max(len(ann_edges) - 1, 1)
    vals = list(forced) if isinstance(forced, (list, tuple)) else [forced] * nann
    if len(vals) != nann:
        raise ValueError(f"{len(vals)} forced contrast(s) for {nann} annuli: give one each")
    return vals


def bench_modes(default):
    """The benchmark arms to run, overridable with ``$BENCH_MODES``.

    One arm at a time is what a re-run needs.  ``run_benchmark`` skips finished slots and
    resumes check-pointed ones -- right for relaunching after an interruption, and exactly
    wrong when one arm has to be recomputed because its sampler changed.  Retire that arm
    first (``scripts/supersede_bench_mode.py``), then name it here, and the other arms keep
    their tag and stay comparable instead of being recomputed for nothing.
    """
    env = os.environ.get("BENCH_MODES", "").replace(",", " ").split()
    if not env:
        return tuple(default)
    bad = [m for m in env if m not in default]
    if bad:
        raise SystemExit(f"BENCH_MODES={bad} not in this stage's arms {tuple(default)}")
    print(f"  (BENCH_MODES={' '.join(env)}: running {len(env)} of {len(default)} arms)")
    return tuple(env)


def _bench_hi(tag, groups, modes, k_max, max_drop, defaults, forced, ann_edges, out, seeds=range(8),
              n_iter=800, n_init=80, n_top=8, n_valid=15, make_red=None, known=None, n_sources=3,
              add_params=None, search_angles=True, n_min_ref=5, n_remeasure=1):
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
    n_iter, n_init = budget(n_iter, n_init, bench=True)
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
                        calibration=CalibrationConfig(forced=_forced_list(forced, ann_edges)),
                        defaults=defaults, n_remeasure=n_remeasure,
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
    run_benchmark(make_runner, modes=bench_modes(modes), seeds=tuple(seeds), n_iter=n_iter, n_init=n_init,
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
    # 3.0e-4 is MEASURED on the current beta Pic flux axis (scripts/calibrate_bench_contrast.py
    # F2: median S/N 4.47 at the default config, k_default 2).  It replaces 2.087e-3, "run B's
    # calibration" -- a number from before ff20c4b moved star_flux from the template's own sum
    # (4.3491) to VIP's published 3.3268e6.  On today's axis 2.087e-3 injects a 168-count peak,
    # a source nearly three times brighter than beta Pic b, and every configuration detects it.
    _bench_hi("F2", 4, ("tpe", "random"), 12, 2, {"k_klip": 5}, 3.0e-4, [8, 22], "F2_bench_highdim")


def run_E2():
    """Run E at the same power (9-D), so the two dimensionalities stay comparable."""
    # 3.0e-4, measured the same way (E2: median S/N 4.21, k_default 3); it replaces 1.31e-3,
    # "the contrast run A calibrated" on the pre-ff20c4b axis, which today injects a 105-count
    # peak -- about twice beta Pic b.  The archived E2_bench / F2_bench_highdim results were
    # produced on the old axis: their optimisation is valid (every configuration saw the same
    # injections) but their contrast labels are not comparable with anything measured now.
    _bench_hi("E2", 1, ("tpe", "random", "grid"), 30, 0, {"k_klip": 10}, 3.0e-4, [8, 22], "E2_bench")


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
    # 7.946e-6 is run C's annulus-1 calibration ON THE ABSOLUTE AXIS.  It replaces a forced
    # 5.899e-9, which was that same calibration read out of C_hd95086/annulus01 (written
    # 2026-09-13) two days BEFORE the star-flux fix of ff20c4b (2026-09-15) -- the one whose
    # own message says it "moved the contrast axis of tutorial 02 and paper runs C and G2".
    # hd95086_objects was fixed there and this constant was not, so the two disagreed by the
    # full 1347.0246 = dit_science/dit_flux/nd_transmission over-count.  A contrast of
    # 5.899e-9 puts the injection peak at 2.4e-4 counts in a cube whose pixels reach 5.8e+02:
    # below the float32 quantum, so the fakes were being rounded away before KLIP saw them
    # (reducer.py raises the RuntimeWarning that says so).  Run C itself still has to be
    # redone for the same reason -- its calibration.json predates the fix as well.
    # Two annuli, not one.  G2 used to take only C's FIRST annulus, [20, 45] px = 0.245-0.551",
    # and HD 95086 b is at 0.620" = 50.6 px -- 5.6 px (1.26 FWHM) beyond the outer edge.  So
    # the one benchmark on a field with a real companion never looked at it, while E2/F2
    # (beta Pic b at 16.6 px in [8, 22]) and H2 (HIP 65426 b at 13.2 px in [6, 20]) both do.
    # C's own split, [20, 45, 75], puts it inside annulus 2 but only 1.26 FWHM in, still on
    # the boundary.  [20, 36, 66] centres it instead -- 3.3 FWHM from the inner edge and 3.5
    # from the outer -- and keeps annulus 2 at C's 30 px width so one forced contrast stays
    # appropriate across the zone.
    #
    # Moving an edge invalidates C's calibrated contrasts, because the contrast that puts the
    # default configuration at median S/N 5 depends on the radii the sources are injected at.
    # These two were MEASURED for these zones by scripts/calibrate_bench_contrast.py, which runs
    # Runner.calibrate -- the same code path the science runs use: 1.112e-5 -> S/N 5.70
    # (k_default 2) and 4.773e-6 -> S/N 5.21 (k_default 4), both inside the (4, 6) target.
    # That script's control reproduces run C on C's own zones, 6.58 and 6.95 against C's
    # recorded 5.88 and 5.40.
    _bench_hi("G2", 1, ("tpe", "random"), 30, None, {"k_klip": 10}, (1.112e-5, 4.773e-6),
              [20, 36, 66], "G2_bench_sphere",
              make_red=hd95086_objects, known=[HD], n_sources=3, n_min_ref=10)


def run_H2():
    """HIP 65426, JWST/NIRCam F444W: both rolls in one partition, a searched ADI/RDI/ADI+RDI.

    The other end of the range from beta Pic -- space, two frames per roll, a searched
    ADI/RDI/ADI+RDI mode, and a field with no disk at all.  Angles are not searched (run D
    does not search them either: with two frames per roll there is nothing to select on).
    """
    # 2.022e-04 is the calibrated contrast for this annulus on the ABSOLUTE axis, measured by
    # scripts/calibrate_bench_contrast.py H2 on 2026-09-17 (median S/N 4.74 at the default
    # configuration, k-scan optimum 4) -- both rolls in one partition, DQ-filled frames,
    # optics_transmission = 1, star centre (149.65, 172.96).  It replaces 1.637e-04 (the
    # same measurement with one partition per roll, where mode was degenerate) and, before
    # that, 2.324e-04, which was measured on frames the old sigma-clip repair had median-filtered
    # (the whole PSF, companion included) against a 0.561 "optics transmission" that only
    # existed to hide that; and before it a forced 5.270e1 -- a "contrast" of 52.7, the
    # raw-detector-units axis of flux_unit = 1.0.  See docs/FLUX_CALIBRATION.md.
    _bench_hi("H2", 1, ("tpe", "random"), 18, None, {"k_klip": 10}, 2.022e-04, [6, 20], "H2_bench_jwst",
              make_red=hip65426_objects, known=[HIP], n_sources=2, search_angles=False, n_min_ref=2,
              # 3 draws per trial, averaged.  A benchmark exists to separate TPE from random,
              # and on MIRI a single draw scatters with sd 0.84 against a ~6 range -- most of
              # what such a benchmark measures is that noise.  NIRCam reductions are ~2.2 s, so
              # 16 slots x 800 evals goes from ~8 h to ~24 h.  This batch is already
              # incomparable to E2/F2/G2 (space, radprof, library, source count), so nothing
              # further is lost by also fixing what it measures.
              n_remeasure=3,
              # no `mode`: hip65426_objects now installs a searched reference library, and
              # nkeep_altroll / nkeep_psfref subsume the categorical.  n_min_ref drops to 2
              # because it is now the floor on the two counts' SUM (the basis size), not on
              # an angular reference census.
              )


if __name__ == "__main__":
    which = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
    t0 = time.time()
    {"A": run_A, "A2": run_A2, "B": run_B, "B2": run_B2, "C": run_C, "D": run_D,
     "E": run_E, "F": run_F, "E2": run_E2, "F2": run_F2,
     "G2": run_G2, "H2": run_H2}[which]()
    log(f"{which} done in {(time.time() - t0) / 60:.1f} min")
