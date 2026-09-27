#!/usr/bin/env python
"""Collect the numbers the paper quotes from the runs in this directory.

    python collect.py                  # everything the paper reads (PAPER, at the bottom)
    python collect.py D DK H HK        # or any subset; each updates its entry in summary.json

Every "default -> optimized" number comes from one comparison made the same way.  For
each annulus the *seeded default* -- the very vector the run started from: the space's
defaults with the run's ``RunConfig.defaults`` on top, at the k its calibration k-scan
chose, pushed through the same reference-count guard the Runner projects every seed
through -- and the run's validated winner are both re-scored on the SAME ``N_TRIALS``
sets of fresh randomized injections at the annulus' own calibrated contrast, with the
raw metric validation uses.  Because the 5-sigma contrast is ``c5 = 5 x contrast /
SNR_inj``, the ratio of the two medians *is* the contrast gain, throughput included, and
pairing the trials takes the draw-to-draw scatter (a factor ~1.4 between two draws of the
same configuration) out of it.  The run's own validated score is kept beside the paired
re-measurement; the configured default (``k_klip`` from RunConfig.defaults, before the
k-scan) is measured too, so the paper can say what the one-dimensional k-scan bought and
what the full search bought on top of it.

Until 2026-09-17 the "default" here was ``space.default_vector()`` unprojected -- angsep
1.923 lambda/D and anglemax 26 deg, which the guard rewrites to 0 / the PA span before any
run sees them -- and the winner was the validated score from a different set of draws.

Independently we measure the real companion in both configurations, a check on a real source
that no injection can fake.  Both are reduced WITH the companion, through the path the run's
own reductions take (Runner._reduce: the annulus zone, each partition's own parameter block).
Until 2026-09-26 the default went through ``red.reduce(ReductionRequest(params))``, which gives
every partition the global -- channel-AVERAGED -- parameters: HD 95086's K1 and K2 are seeded
with slightly different ``angsep``, enough to change which frames serve as references, and its
companion measured 17.5 in a configuration the run never used (16.7 in its own).  The winner
is reduced afresh the same way rather than read from ``best_clean.fits``, which a run that
subtracts the companion (RunConfig.subtract_known) writes without it.

Two companion S/N are recorded.  ``planet_snr_*`` is the run's search metric on the clean
image, kept for continuity.  ``companion_fm_*`` is the one the paper quotes: the companion is
fitted and cancelled by a negative injection (klip_tpe.companion), its S/N is its own reduced
image over the ring scatter of the companion-free image.  The metric does worse on a bright
companion three ways, each configuration-dependent: its de-spiking clips the companion's core,
the companion's own light beyond the 1.5-FWHM exclusion enters the noise ring (on NIRCam it set
the ring scatter at 5-14x the companion-free value), and at small separations a handful of
apertures make the scatter depend on where they fall.  On 2026-09-26 the metric gave NIRCam's
winners 12.1 -> 16.7 (pyKLIP) for a companion whose forward-model S/N falls from 164 to 107.
"""
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
from astropy.io import fits

from klip_tpe import RunConfig, ValidationConfig
from klip_tpe.instruments import generic
from klip_tpe.metrics import MawetPeakSNR, Source
from klip_tpe.products import noise_profile
from klip_tpe.runner import Runner

import run_demos as R

OUT = R.OUT
TARGETS = {"A": ("A_betapic", "betapic", R.BP), "A2": ("A2_betapic", "betapic", R.BP),
           "B": ("B_betapic_groups", "betapic", R.BP), "B2": ("B2_betapic_groups", "betapic", R.BP),
           "C": ("C_hd95086", "hd95086", R.HD),
           "D": (R.ENGINE_DIRS["D"]["pyklip"], "hip65426", R.HIP),
           "DK": (R.ENGINE_DIRS["D"]["klip"], "hip65426", R.HIP),
           # D / DK with HIP 65426 b taken out of the frames first (run_demos.hip65426b_negfc)
           "D2": (R.ENGINE_DIRS["D2"]["pyklip"], "hip65426", R.HIP),
           "D2K": (R.ENGINE_DIRS["D2"]["klip"], "hip65426", R.HIP),
           # the mode-only pyKLIP run of 2026-09-20, before the library was searched at all
           "D0": ("D_hip65426", "hip65426", R.HIP)}
N_TRIALS = 8          # paired re-measurement sets per annulus (default, flat default, winner)

#: Published contrast of the companion in each data set's own band -- the CHECK on each
#: contrast axis.  Every axis is now calibrated in its own right (beta Pic from VIP's
#: published starphot, HD 95086 from the flux frames, HIP 65426 from a stellar flux density
#: through the MJy/sr calibration and the STPSF off-axis PSF; docs/FLUX_CALIBRATION.md), and
#: the companion is what the calibration is tested against: ``flux_scale`` is the ratio of
#: the companion's contrast as this axis measures it to the published value, and it should
#: be 1.  It is recorded, not applied, unless ANCHOR_APPLY=1 is set -- rescaling a
#: calibrated axis onto the companion would turn the check into a tautology.
#:
#: HOW it is measured matters.  Until 2026-09-17 the companion's S/N in the clean image was
#: compared with the S/N of three sources injected on its ring, and S/N is not a flux: an
#: injected source's noise ring holds the OTHER sources' PSF structure -- for the NIRCam
#: coronagraphic PSF, six lobes at 2-3 FWHM that the 1.5-FWHM exclusion does not remove --
#: so the fakes' S/N was depressed relative to the lone companion's (run D: 7.27 against
#: 11.68 at a contrast the peaks put 16% apart) and ``flux_scale`` came out 1.87 on an axis
#: that scripts/check_hip65426_contrast.py puts at 0.97.  The comparison is now a ratio of
#: matched-filter PEAKS -- each fake in (injected - clean), the companion in the
#: radial-profile-subtracted clean image, same reduction, same separation, same kernel -- so
#: the throughput cancels and only the flux ratio is left.  The earlier beta Pic (1.21) and
#: HD 95086 values were made with the S/N method and have to be re-collected.
ANCHOR = {
    "betapic":  (6.25e-4, "Absil et al. 2013, dL' = 8.01 +/- 0.16 (this very data set)"),
    "hd95086":  (1.32e-5, "Chauvin et al. 2018, dK1 = 12.2 +/- 0.1 (2015-02-03)"),
    "hip65426": (3.30e-4, "Carter et al. 2023, dF444W = 8.703 +/- 0.055"),
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run_flatten(which):
    """Whether the run's metric subtracted the radial profile.  The default flipped to False
    on 2026-09-22; the runs made before it (A2, B2, C and the E2/F2/G2 benchmarks) optimized
    and validated with it on, and their run_setup.json records it -- re-scoring them with the
    new default measures a different objective from the one they were searched on."""
    d = TARGETS[which][0]
    try:
        with open(os.path.join(OUT, d, "run_setup.json")) as f:
            m = json.load(f).get("objective", {}).get("metric", {})
    except (OSError, ValueError):
        return False
    return bool(m.get("flatten", True))      # absent: a run from before the key was recorded


def build(which):
    """(reducer, space, objective, sampler) exactly as the run built them, so 'default'
    means the very vector the run was seeded with."""
    from klip_tpe import Param
    if which in ("A", "A2", "B", "B2"):
        single = which in ("A", "A2")
        dsets, sf, inst = R.betapic_dataset(1 if single else 4)
        red = generic.make_reducer(dsets, star_flux=sf, max_workers="auto", log=lambda s: None,
                                   partition_label="dataset" if single else "group", **inst)
        if single:
            space = generic.make_space(red, k_klip_max=30)
            space.project = generic.make_guard(red, k_max=30)
        else:
            space = generic.make_space(red, k_klip_max=12, max_drop=2)
            space.project = generic.make_guard(red, k_max=12, n_min_ref=5)
        known = R.BP
    elif which == "C":
        red = R.hd95086_objects()
        space = generic.make_space(red, k_klip_max=30)
        space.project = generic.make_guard(red, k_max=30)
        known = R.HD
    elif which in ("D", "DK", "D2", "D2K"):
        red = R.hip65426_objects(engine="pyklip" if which in ("D", "D2") else "klip")
        space = generic.make_space(red, k_klip_max=18, search_angles=False)
        space.project = generic.make_guard(red, k_max=18, n_min_ref=4)
        known = R.HIP
    elif which == "D0":
        red = R.hip65426_objects(engine="pyklip", library=False)
        space = generic.make_space(red, k_klip_max=18, search_angles=False)
        space.add(Param("mode", 0, 2, "categorical", choices=["ADI", "RDI", "ADI+RDI"], default="RDI",
                        doc="pyKLIP PSF-subtraction mode"))
        space.project = generic.make_guard(red, k_max=18, n_min_ref=4)
        known = R.HIP
    else:
        raise ValueError(which)
    # The run's own objective.  The beta Pic runs forbid the disk's position angles to the
    # injections and mask its pixels out of the noise rings (run_demos.bp_disk); until
    # 2026-09-26 this built the objective without them, so A2/B2's paired re-measurement put
    # sources on the disk and counted the disk as noise -- a different objective from the one
    # the search optimized (A2's first annulus re-measured at 8.2 a winner that validated 14.3).
    fpa, pmask = R.bp_disk(red) if which in ("A", "A2", "B", "B2") else ((), None)
    obj, samp = generic.default_config(red, known=[known], forbidden_pa=fpa, pixel_mask=pmask,
                                       flatten=run_flatten(which))
    return red, space, obj, samp


_BUILT = {}


def built(which):
    """``build(which)``, once per process (figs.py asks for several images of one run)."""
    if which not in _BUILT:
        _BUILT[which] = build(which)
    return _BUILT[which]


def clean_image(which, a, x):
    """The clean reduction of search vector ``x`` over the annulus of record ``a`` (a
    summary.json annulus), with the companion PRESENT, through the path the run's own
    reductions take (Runner._reduce: the run's zone, each partition's own block) -- what the
    figures show beside the S/N collect measured on the same images."""
    d = TARGETS[which][0]
    with open(os.path.join(OUT, d, "run_setup.json")) as f:
        setup = json.load(f)["config"]
    red, space, obj, samp = built(which)
    cfg = RunConfig(ann_edges=list(setup.get("ann_edges") or [a["inrad_px"], a["outrad_px"]]), n_iter=1, n_init=1,
                    seed=0, n_sources=int(setup.get("n_sources", 3) or 3), defaults=dict(setup.get("defaults") or {}),
                    validation=ValidationConfig(n_top=1, n_valid=1), save_fits=False, save_eval_images=False,
                    write_setup_files=False)
    runner = Runner(red, space, obj, samp, cfg, os.path.join(OUT, f"_default_{which}"), log=lambda s: None)
    runner.ia, runner.contrast = int(a["annulus"]), float(a["contrast"])
    return np.asarray(runner._reduce(space.decode(np.asarray(x, float)), None, tag="figure").image, float)


def run_vector(x, run_params, space):
    """A vector the run recorded, in THIS space's layout, matched by parameter name.

    ``run_params`` is the ``space.params`` list of the run's run_setup.json.  The layout can
    differ from the space ``build`` makes today: make_space now pins a dimension whose range
    has collapsed to a point instead of searching it, so B2 (made before that) carries
    ``bin_g1..bin_g4 : [1, 1]`` -- 38 entries against today's 34 -- and decoding its winner
    position by position read n_ang as filter, filter as angsep, ..., and the last group's
    k_klip as a drop slot (it came out as g1, g2, g4 with n_ang 1, filter 6, k 3 in g1, a
    configuration the run never evaluated).  A run dimension this space lacks has to be
    pinned here at the value the run held it at; a dimension this space has and the run
    lacked cannot be reconstructed at all."""
    names = [p["name"] for p in run_params]
    x = [float(v) for v in x]
    if len(names) != len(x):
        raise ValueError(f"run vector has {len(x)} entries for {len(names)} recorded parameters")
    got = dict(zip(names, x))
    missing = [n for n in space.names if n not in got]
    if missing:
        raise ValueError(f"the run did not search {missing}; its vectors cannot be rebuilt here")
    fixed = dict(getattr(space, "fixed", {}) or {})
    for p, v in zip(run_params, x):
        if p["name"] in space.names:
            continue
        base = p.get("base") or p["name"]
        if base not in fixed or float(fixed[base]) != v:
            raise ValueError(f"run dimension {p['name']} = {v} is neither searched nor pinned "
                             f"at that value here (fixed: {fixed.get(base)})")
    return np.asarray([got[n] for n in space.names], float)


def planet_snr(img, red, planet, metric=None):
    """The companion's S/N, measured with the run's own metric when given -- its noise ring
    then has the same exclusions as every score the search ranked on (the disk, for beta Pic).
    Kept for continuity; the paper quotes :func:`companion_fm` (see the module docstring)."""
    m = metric or MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel)
    return float(m.per_source(img, None, [planet[0]], [planet[1]])[0])


def companion_fm(runner0, cfg, red, obj, rho, pa, c_guess, fit_position=False, log=None):
    """The companion's forward-model S/N in configuration ``cfg`` (klip_tpe.companion).

    Each selected partition is reduced on its own, with the companion present and cancelled
    by a negative injection at the contrast fitted in that partition -- a channel may see a
    different contrast (HD 95086 b is brighter in K2 than in K1) -- and the partitions are
    combined as the partitioned reducer combines them.  ``runner0`` must not subtract the
    companion itself (RunConfig.subtract_known empty).  With ``fit_position`` the first
    partition's fit also refines the position, which the rest then use."""
    from klip_tpe.companion import companion_snr, fit_negative_companion
    from klip_tpe.reducer import combine_stack
    from klip_tpe.space import Config
    sel = list(cfg.selected)
    m = obj.metric
    ac = getattr(red, "angle_convention", "pa")
    per, pos = {}, (float(rho), float(pa))
    for i, pid in enumerate(sel):
        one = Config(dict(cfg.params), dict(cfg.per_partition), [pid], cfg.x) if len(sel) > 1 else cfg

        def fn(s, one=one):
            return np.asarray(runner0._reduce(one, s, tag="companion_fm").image, float)
        clean = fn(None)
        f = fit_negative_companion(fn, pos[0], pos[1], float(c_guess), red.fwhm, red.pxscale, angle_convention=ac,
                                   fit_position=bool(fit_position and i == 0), clean=clean, log=log)
        pos = (f["rho"], f["pa"]) if (fit_position and i == 0) else pos
        per[pid] = dict(fit=f, clean=clean, removed=fn([Source(pos[0], pos[1], -f["contrast"])]))
    wfn = getattr(red, "partition_weight", lambda p: 1.0)
    w = np.array([wfn(p) for p in sel], float)
    clean = combine_stack(np.stack([per[p]["clean"] for p in sel]), w)
    removed = combine_stack(np.stack([per[p]["removed"] for p in sel]), w)
    r = companion_snr(None, pos[0], pos[1], 0.0, red.fwhm, red.pxscale, m.kernel([pos[0]]),
                      pixel_mask=m.pixel_mask, angle_convention=ac, clean=clean, removed=removed)
    return {"snr": r["snr"], "snr_band": r["snr_band"], "signal": r["signal"], "sigma": r["sigma"],
            "sigma_range": list(r["sigma_range"]), "n_ap": r["n_ap"], "rho_as": pos[0], "pa_deg": pos[1],
            "contrast": {str(p): per[p]["fit"]["contrast"] for p in sel},
            "removed_fraction": {str(p): per[p]["fit"]["removed_fraction"] for p in sel},
            "metric_on_clean": float(m.per_source(clean, None, [pos[0]], [pos[1]])[0])}


def sigma_curve(img, red, rin_px, rout_px, known):
    sig, rad = noise_profile(img, red.fwhm, max(rin_px, red.fwhm), rout_px,
                             known=[known], pxscale=red.pxscale)
    ok = np.isfinite(sig) & (sig > 0)
    return (rad[ok] * red.pxscale).tolist(), sig[ok].tolist()


def collect(which, n_trials=N_TRIALS):
    d, tag, planet = TARGETS[which]
    run = os.path.join(OUT, d)
    fr = json.load(open(os.path.join(run, "final_results.json")))
    log(f"{which}: {len(fr['annuli'])} annuli from {d}")
    red, space, obj, samp = build(which)
    setup_all = json.load(open(os.path.join(run, "run_setup.json")))
    setup = setup_all["config"]
    run_params = (setup_all.get("space") or {}).get("params")
    out = {"run": d, "target": tag, "planet": list(planet), "pxscale": red.pxscale,
           "fwhm_px": red.fwhm, "partitions": [str(p) for p in red.partitions()],
           "n_frames": {str(p): int(r.data.nframes) for p, r in red.reducers.items()},
           "search_space": {"ndim": space.ndim, "names": list(space.names)},
           "setup": {k: setup.get(k) for k in ("n_iter", "n_init", "seed", "n_sources",
                                               "ann_edges", "search_mode") if k in setup},
           "annuli": []}
    # the vector the run was SEEDED with: the space's per-parameter defaults with the run's
    # RunConfig.defaults on top (D, C and the beta Pic runs all seed k_klip = 10 that way).
    # Until 2026-09-17 this took space.default_vector() alone -- k = 6 for D -- so the
    # "default" column measured a configuration the run never started from.
    seeded = dict(setup.get("defaults") or {})
    out["seeded_defaults"] = seeded
    k_flat = int(seeded.get("k_klip", 10) or 10)
    tmp = os.path.join(OUT, f"_default_{which}")
    x0_by_ann = {}
    # the companion the run took out of its frames, if any: the paired re-measurement below
    # scores the run's own objective, so it subtracts it too; the companion's own
    # measurement and the figures reduce without it
    sub = [tuple(float(v) for v in t) for t in (setup.get("subtract_known") or [])]
    out["subtract_known"] = [list(t) for t in sub]
    rho_c = planet[0]
    c_guess = float(ANCHOR.get(tag, (1e-4,))[0])
    fm_pos = None
    plain_by_ann = {}
    for a in fr["annuli"]:
        ia, rin, rout = a["annulus"], a["inrad"], a["outrad"]
        contrast = a["contrast"]
        k_seed = int(a.get("k_default") or k_flat)
        rec = {"annulus": ia, "inrad_px": rin, "outrad_px": rout,
               "inrad_as": rin * red.pxscale, "outrad_as": rout * red.pxscale,
               "contrast": contrast, "search_best": a["search_best_score"],
               "validated": a["validated"], "winner_score": a["winner_score"],
               "winner_index": a["winner_index"], "winner_params": a["winner_config"]["params"],
               "winner_per_partition": a["winner_config"]["per_partition"],
               "selected": [str(s) for s in a["winner_config"]["selected"]],
               "validation_table": [{k: t.get(k) for k in ("eval_index", "search_score",
                                                           "validated_score", "trials")}
                                    for t in a["validation_table"]]}
        # --- the seeded default and the winner, paired, through the validation metric
        cfg = RunConfig(ann_edges=list(setup.get("ann_edges") or [rin, rout]),
                        n_iter=1, n_init=1, seed=99 + ia,
                        n_sources=int(setup.get("n_sources", 3)), defaults=dict(seeded),
                        validation=ValidationConfig(n_top=1, n_valid=n_trials),
                        save_fits=False, save_eval_images=False, write_setup_files=False,
                        subtract_known=list(sub))
        runner = Runner(red, space, obj, samp, cfg, tmp, log=lambda s: None)
        runner.ia = ia
        runner.contrast = contrast
        # the same, keeping the companion in the frames: its own measurement and the images
        runner0 = Runner(red, space, obj, samp, replace(cfg, subtract_known=[]), tmp, log=lambda s: None)
        runner0.ia = ia
        runner0.contrast = contrast
        # the run's own source count: the ring-survival check that sets it probes placements
        # seeded by the run's seed, so under this runner's seed it could decide differently
        # (A2's innermost annulus came out 2 sources at 0.258" against the run's 3 at 0.245")
        n_run = len(a.get("winner_sources") or [])
        if n_run:
            runner._nsrc_cache[ia] = n_run
        # the Runner seeds the PROJECTED default: reference-count guard applied (angsep,
        # anglemax, k cap), exactly as _reset_history does
        x0 = runner._project(runner._default_vector(k_seed), is_random=False)
        x_flat = runner._project(runner._default_vector(k_flat), is_random=False)
        x_win = None
        if a.get("winner_x") is not None:
            x_win = (run_vector(a["winner_x"], run_params, space) if run_params
                     else np.asarray(a["winner_x"], float))
        x0_by_ann[ia] = x0
        cands = {"default": x0, "winner": x_win}
        if k_flat != k_seed:
            cands["default_flat"] = x_flat
        trials = {k: [] for k in cands if cands[k] is not None}
        rlo, rhi = runner._band(ia, space.decode(x0))
        for _ in range(n_trials):
            src = samp.sample(runner._nsrc(ia), rlo, rhi, runner.rng, contrast)
            for name, x in cands.items():
                if x is None:
                    continue
                r, _, _ = runner.evaluate(x, "default", contrast=contrast, sources=src, raw_only=True)
                ok = r.raw_score is not None and np.isfinite(r.raw_score)
                trials[name].append(float(r.raw_score) if ok else float("nan"))
        d_tr = np.array(trials["default"], float)
        rec["default_k"] = k_seed
        rec["default_trials"] = [None if not np.isfinite(v) else float(v) for v in d_tr]
        rec["default_score"] = float(np.nanmedian(d_tr)) if np.isfinite(d_tr).any() else float("nan")
        if "default_flat" in trials:
            f_tr = np.array(trials["default_flat"], float)
            rec["default_flat_k"] = k_flat
            rec["default_flat_trials"] = [None if not np.isfinite(v) else float(v) for v in f_tr]
            rec["default_flat_score"] = float(np.nanmedian(f_tr)) if np.isfinite(f_tr).any() else float("nan")
        if "winner" in trials:
            w_tr = np.array(trials["winner"], float)
            rec["winner_trials"] = [None if not np.isfinite(v) else float(v) for v in w_tr]
            rec["winner_remeasured"] = float(np.nanmedian(w_tr)) if np.isfinite(w_tr).any() else float("nan")
            both = np.isfinite(w_tr) & np.isfinite(d_tr)
            rec["paired_wins"] = int(np.sum(w_tr[both] > d_tr[both]))
            rec["paired_n"] = int(both.sum())
            rec["gain"] = (rec["winner_remeasured"] / rec["default_score"]
                           if both.any() and rec["default_score"] > 0 else None)
            rec["gain_per_trial_median"] = float(np.nanmedian(w_tr[both] / d_tr[both])) if both.any() else None
        else:
            rec["gain"] = None
        rec["gain_vs_validated"] = ((rec["winner_score"] / rec["default_score"])
                                    if np.isfinite(rec["default_score"]) and rec["default_score"] > 0 else None)
        # --- the real companion, and the noise profile, in both clean images: reduced with the
        # companion present, through the run's own path (zone, per-partition blocks)
        cfg0 = space.decode(x0)
        cfgw = None if x_win is None else space.decode(x_win)
        plain_by_ann[ia] = (runner0, cfg0)
        p0 = dict(cfg0.params, inrad=rin, outrad=rout)
        img0 = np.asarray(runner0._reduce(cfg0, None, tag="collect_default").image, float)
        img1 = None if cfgw is None else np.asarray(runner0._reduce(cfgw, None, tag="collect_winner").image, float)
        best = os.path.join(run, f"annulus{ia + 1:02d}", "best_clean.fits")
        if img1 is not None and not sub and os.path.exists(best):
            f_img = np.asarray(fits.getdata(best), float)
            both = np.isfinite(f_img) & np.isfinite(img1)
            dev = float(np.nanmax(np.abs(f_img[both] - img1[both])) / np.nanstd(img1[both])) if both.any() else np.nan
            rec["winner_vs_best_clean_sigma"] = dev
            if not np.isfinite(dev) or dev > 0.05:
                log(f"  ** ann {ia + 1}: the winner re-reduced differs from the run's best_clean.fits by "
                    f"{dev:.3g} sigma -- the package changed since the run?")
        rec["default_params"] = {k: (v if isinstance(v, (str, bool)) else float(v))
                                 for k, v in p0.items()}
        rec["default_per_partition"] = {str(p): {k: (v if isinstance(v, (str, bool)) else float(v))
                                                 for k, v in d.items()} for p, d in cfg0.per_partition.items()}
        rec["default_x"] = [float(v) for v in x0]
        rec["winner_x_space"] = None if x_win is None else [float(v) for v in x_win]
        # an annulus whose calibration never reached the S/N window is not on the usual
        # scale -- neither its injected S/N nor its 5-sigma curve -- so say so here rather
        # than letting a number into the table that looks like the others
        cal_p = os.path.join(run, f"annulus{ia + 1:02d}", "calibration.json")
        if os.path.exists(cal_p):
            try:
                cal = json.load(open(cal_p))
            except Exception:
                cal = {}
            rec["calibration_snr"] = cal.get("snr")
            rec["uncalibrated"] = bool(cal.get("uncalibrated", False))
            rec["forced_contrast"] = float(cal.get("forced") or 0.0)
            if rec["uncalibrated"]:
                log(f"  ** ann {ia + 1} was NOT calibrated (default S/N {cal.get('snr')} at "
                    f"{contrast:.3e}); its S/N and c5 are not comparable with the other annuli")
        rec["planet_snr_default"] = planet_snr(img0, red, planet, obj.metric)
        rec["planet_snr_optimized"] = None if img1 is None else planet_snr(img1, red, planet, obj.metric)
        # the forward-model S/N, in the annulus that holds the companion: the default's fit also
        # refines the companion's position, which the winner's then uses
        if rin * red.pxscale <= rho_c <= rout * red.pxscale:
            try:
                fm0 = companion_fm(runner0, cfg0, red, obj, planet[0], planet[1], c_guess, fit_position=True, log=log)
                fm_pos = (fm0["rho_as"], fm0["pa_deg"])
                rec["companion_fm_default"] = fm0
                if cfgw is not None:
                    rec["companion_fm_optimized"] = companion_fm(runner0, cfgw, red, obj, fm_pos[0], fm_pos[1],
                                                                 float(np.median(list(fm0["contrast"].values()))),
                                                                 log=log)
                log(f"  ann {ia + 1}: companion forward-model S/N {fm0['snr']:.1f} -> "
                    f"{rec.get('companion_fm_optimized', {}).get('snr', float('nan')):.1f} "
                    f"(metric {rec['planet_snr_default']:.1f} -> {rec['planet_snr_optimized'] or float('nan'):.1f}); "
                    f"fitted contrast {fm0['contrast']}")
            except Exception as exc:
                log(f"  ann {ia + 1}: companion forward-model S/N failed: {exc!r}")
        r0, s0 = sigma_curve(img0, red, rin, rout, planet)
        rec["sigma_default"] = {"r_as": r0, "sigma": s0}
        if img1 is not None:
            r1, s1 = sigma_curve(img1, red, rin, rout, planet)
            rec["sigma_optimized"] = {"r_as": r1, "sigma": s1}
            rec["sigma_ratio_median"] = float(np.nanmedian(
                np.asarray(s0) / np.interp(r0, r1, s1)))
        flat = ("" if "default_flat_score" not in rec
                else f" [k={k_flat} flat default {rec['default_flat_score']:.2f}]")
        log(f"  ann {ia + 1} [{rin:.0f}-{rout:.0f} px]: injected S/N default(k={k_seed}) {rec['default_score']:.2f} -> "
            f"winner {rec.get('winner_remeasured', float('nan')):.2f} paired on {rec.get('paired_n', 0)} sets "
            f"(x{rec['gain'] or float('nan'):.2f}, winner better in {rec.get('paired_wins', 0)}/{rec.get('paired_n', 0)}); "
            f"run validated {rec['winner_score']:.2f}{flat}; "
            f"planet {rec['planet_snr_default']:.1f} -> "
            f"{-1 if rec['planet_snr_optimized'] is None else rec['planet_snr_optimized']:.1f}")
        out["annuli"].append(rec)

    ia_c = next((a["annulus"] for a in out["annuli"] if a["inrad_as"] <= rho_c <= a["outrad_as"]), None)
    r0 = plain_by_ann.get(ia_c)
    anchor(out, red, planet, None if not isinstance(r0, tuple) else
           (lambda s, r0=r0: np.asarray(r0[0]._reduce(r0[1], s, tag="anchor").image, float)))
    st = os.path.join(run, "klip_stitched.fits")
    if os.path.exists(st):
        out["stitched_planet_snr"] = planet_snr(np.asarray(fits.getdata(st), float), red, planet, obj.metric)
    out.update(read_curves(run))
    return out


def read_curves(run):
    """contrast_curve.txt holds several blocks (injection-calibrated, per-annulus
    samples, KLIP-FM cross-check) with different column counts; split on the comments."""
    p = os.path.join(run, "contrast_curve.txt")
    if not os.path.exists(p):
        return {}
    blocks, name, rows = {}, "main", []
    def flush():
        if rows:
            blocks[name] = {"r_as": [r[0] for r in rows], "c5": [r[1] for r in rows]}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            low = line.lower()
            if "klip-fm" in low:
                flush(); rows = []; name = "fm"
            elif "sample" in low:
                flush(); rows = []; name = "samples"
            continue
        try:
            v = [float(x) for x in line.split()]
        except ValueError:
            continue
        if len(v) >= 2:
            rows.append(v)
    flush()
    return {"curves": blocks}


def anchor(out, red=None, planet=None, reduce0=None):
    """Check the contrast axis against the companion's published contrast (see ANCHOR), and
    rescale onto it only when ANCHOR_APPLY=1.

    Three sources are injected at the companion's own separation, 90/180/270 deg away, at
    the annulus' calibrated contrast, and reduced with the same default configuration.  The
    companion's contrast in this axis' units is ``c x peak_companion / median(peak_fake)``,
    with each fake's matched-filter peak read in (injected - clean) and the companion's in
    the radial-profile-subtracted clean image: same reduction, same separation, same kernel,
    so the KLIP throughput cancels.  The S/N values are recorded alongside for the record;
    they are NOT used for the scale (see the note above ANCHOR).  ``reduce0(sources)`` reduces
    the companion annulus' seeded default as :func:`collect` measured it, companion present,
    through the run's own path (before 2026-09-26: ``red.reduce(params)``, the channel-averaged
    parameters on a two-channel run)."""
    from klip_tpe.metrics import mawet_peak_snr, radprof
    pub, ref = ANCHOR.get(out["target"], (None, None))
    out["anchor_reference"] = ref
    out["flux_scale_applied"] = 1.0
    if pub is None or red is None or reduce0 is None:
        return
    rho, pa = planet
    a = next((a for a in out["annuli"] if a["inrad_as"] <= rho <= a["outrad_as"]), None)
    if a is None or a.get("planet_snr_default") is None or not np.isfinite(a["planet_snr_default"]):
        out["flux_scale"] = None
        return
    c = float(a["contrast"])
    clean = reduce0(None)
    srcs = [Source(rho, (pa + d) % 360.0, c) for d in (90.0, 180.0, 270.0)]
    img = reduce0(srcs)
    ker = red.matched_filter_kernel(rho)
    ac = red.angle_convention if hasattr(red, "angle_convention") else "pa"
    _, dc = mawet_peak_snr(radprof(clean), [rho], [pa], red.pxscale, red.fwhm, kernel=ker,
                           return_details=True, angle_convention=ac)
    sn_f, df = mawet_peak_snr(radprof(img - clean), [s.rho for s in srcs], [s.theta for s in srcs],
                              red.pxscale, red.fwhm, kernel=ker, known=[(rho, pa)],
                              return_details=True, angle_convention=ac)
    sn_img, _ = mawet_peak_snr(radprof(img), [s.rho for s in srcs], [s.theta for s in srcs],
                               red.pxscale, red.fwhm, kernel=ker, known=[(rho, pa)],
                               return_details=True, angle_convention=ac)
    pk_c = float(dc[0]["peak"])
    pk_f = np.array([d["peak"] for d in df], float)
    pk_f = pk_f[np.isfinite(pk_f) & (pk_f > 0)]
    if not np.isfinite(pk_c) or pk_c <= 0 or pk_f.size == 0:
        out["flux_scale"] = None
        return
    implied = c * pk_c / float(np.median(pk_f))       # companion contrast in this axis' units
    scale = implied / pub
    apply = os.environ.get("ANCHOR_APPLY", "") == "1"
    out.update(implied_companion_contrast=implied, published_companion_contrast=pub,
               anchor_inj_contrast=c, anchor_companion_peak=pk_c,
               anchor_inj_peaks=[float(v) for v in pk_f],
               anchor_inj_snr=float(np.nanmedian(sn_img)), anchor_companion_snr=a["planet_snr_default"],
               flux_scale=scale, flux_scale_applied=(scale if apply else 1.0),
               anchor_method="matched-filter peak ratio, fakes in (inj - clean)")
    for aa in out["annuli"]:
        aa["contrast_anchored"] = aa["contrast"] / out["flux_scale_applied"]
    dmag = -2.5 * np.log10(implied) - (-2.5 * np.log10(pub))
    log(f"  check: companion peak {pk_c:.4g} vs fakes {np.median(pk_f):.4g} at {c:.3e} on its ring "
        f"-> {implied:.3e} in this axis' units, published {pub:.3e}: ratio {scale:.3f} "
        f"({dmag:+.2f} mag); S/N for the record {np.nanmedian(sn_img):.2f} (fakes) / "
        f"{a['planet_snr_default']:.2f} (companion).  "
        + ("axis RESCALED by it (ANCHOR_APPLY=1)" if apply else "axis left on its calibrated scale"))


#: Which directory each benchmark reads, newest acceptable first, so a half-migrated tree
#: never silently mixes two objectives into one figure.
#:
#: Naming warning: the ``*_unpaired_20260913/`` directories were archived under that name on
#: the belief that the per-evaluation injection redraw was a porting bug.  It is not -- it is
#: what ``near2m_randpos`` does -- so those runs use the REFERENCE objective and the
#: "paired" (fixed_sources=True) reruns of 2026-09-13/14 are the wrong ones.  Any bench
#: directory produced in that window must be redone before it is collected.
BENCH_DIRS = {"E": ("E2_bench", "E_bench"), "F": ("F2_bench_highdim", "F_bench_highdim"),
              "G": ("G2_bench_sphere",),
              # H3 on each engine: H2 with HIP 65426 b subtracted (2026-09-27).  Not falling back
              # to H2: b's light set its noise ring, and H2K's scores came from the searched-
              # library cache that handed one reduction another's frames.  (Nor to the old
              # H2_bench_jwst, whose newest slots searched the nkeep counts pyKLIP ignored.)
              "H": (R.ENGINE_DIRS["H3"]["pyklip"],), "HK": (R.ENGINE_DIRS["H3"]["klip"],)}


def bench_dir(which):
    """First of this benchmark's candidate directories that actually has rows."""
    for sub in BENCH_DIRS[which]:
        if os.path.exists(os.path.join(OUT, sub, "bench_summary.txt")):
            return sub
    return BENCH_DIRS[which][0]


def collect_bench(sub="E_bench"):
    from klip_tpe.bench import summarize_bench
    p = os.path.join(OUT, sub)
    if not os.path.exists(os.path.join(p, "bench_summary.txt")):
        log(f"{sub}: no rows yet")
        return None
    out = summarize_bench(p, log=log)
    out["out_dir"] = p
    return out


#: What ``python collect.py`` collects when no target is named: everything the paper's
#: numbers and figures read -- the searched runs (A2, B2, C; NIRCam as D2 / D2K, with the
#: companion taken out of the frames, and as D / DK, the runs that kept it), and the
#: benchmarks, H2 on both engines.  Until 2026-09-25 the default was A C D B: A and B are the
#: runs from before 2026-09-14, whose directories are gone, so it refreshed C and D, printed
#: two tracebacks and left the A2 and B2 the figures read as they were.  A target whose run
#: has not finished is skipped with a note.
PAPER = ("A2", "B2", "C", "D", "DK", "D2", "D2K", "E", "F", "G", "H", "HK")


if __name__ == "__main__":
    which = [w.upper() for w in sys.argv[1:]] or list(PAPER)
    out = os.path.join(OUT, "summary.json")
    old = json.load(open(out)) if os.path.exists(out) else {}
    done = []
    for w in which:
        if w not in TARGETS and w not in BENCH_DIRS:
            log(f"{w}: not a target ({' '.join(TARGETS)}) or a benchmark ({' '.join(BENCH_DIRS)}) -- skipped")
            continue
        if w in TARGETS and not os.path.exists(os.path.join(OUT, TARGETS[w][0], "final_results.json")):
            log(f"{w}: {TARGETS[w][0]} has no final_results.json (not run, or not finished) -- "
                f"skipped, its entry in summary.json left as it was")
            continue
        try:
            old[w] = (collect_bench(bench_dir(w)) if w in BENCH_DIRS else collect(w))
            done.append(w)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            log(f"{w}: FAILED {exc!r}")
        json.dump(old, open(out, "w"), indent=1)
    log(f"wrote {out}: {' '.join(done)}" if done else "nothing collected; summary.json unchanged")
