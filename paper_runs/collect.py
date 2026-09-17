#!/usr/bin/env python
"""Collect the numbers the paper quotes from the runs in this directory.

    python collect.py A C D B E        # any subset; updates summary.json

Every "default -> optimized" number comes from one comparison made the same way.  For
each annulus we take the run's validated winner and re-run the *seeded default*
configuration through the identical validation protocol: ``n_valid`` trials of fresh
randomized injections at the annulus' own calibrated contrast, scored with the same
metric.  Because the 5-sigma contrast is ``c5 = 5 x contrast / SNR_inj``, the ratio of
those two median S/N values *is* the contrast gain, throughput included.

Independently we measure the real companion in both clean images (Mawet small-sample
matched-filter S/N), which is a check on a real source that no injection can fake.
"""
import json
import os
import sys
import time

import numpy as np
from astropy.io import fits

from klip_tpe import RunConfig, ValidationConfig
from klip_tpe.instruments import generic
from klip_tpe.metrics import MawetPeakSNR, Source
from klip_tpe.products import noise_profile
from klip_tpe.reducer import ReductionRequest
from klip_tpe.runner import Runner

import run_demos as R

OUT = R.OUT
TARGETS = {"A": ("A_betapic", "betapic", R.BP), "A2": ("A2_betapic", "betapic", R.BP),
           "B": ("B_betapic_groups", "betapic", R.BP), "B2": ("B2_betapic_groups", "betapic", R.BP),
           "C": ("C_hd95086", "hd95086", R.HD), "D": ("D_hip65426", "hip65426", R.HIP)}
N_DEFAULT_TRIALS = 5

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
    elif which == "D":
        red = R.hip65426_objects()
        space = generic.make_space(red, k_klip_max=18, search_angles=False)
        space.add(Param("mode", 0, 2, "categorical", choices=["ADI", "RDI", "ADI+RDI"], default="RDI",
                        doc="pyKLIP PSF-subtraction mode"))
        space.project = generic.make_guard(red, k_max=18, n_min_ref=4)
        known = R.HIP
    else:
        raise ValueError(which)
    obj, samp = generic.default_config(red, known=[known])
    return red, space, obj, samp


def planet_snr(img, red, planet):
    m = MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel)
    return float(m.per_source(img, None, [planet[0]], [planet[1]])[0])


def sigma_curve(img, red, rin_px, rout_px, known):
    sig, rad = noise_profile(img, red.fwhm, max(rin_px, red.fwhm), rout_px,
                             known=[known], pxscale=red.pxscale)
    ok = np.isfinite(sig) & (sig > 0)
    return (rad[ok] * red.pxscale).tolist(), sig[ok].tolist()


def collect(which, n_trials=N_DEFAULT_TRIALS):
    d, tag, planet = TARGETS[which]
    run = os.path.join(OUT, d)
    fr = json.load(open(os.path.join(run, "final_results.json")))
    log(f"{which}: {len(fr['annuli'])} annuli from {d}")
    red, space, obj, samp = build(which)
    setup = json.load(open(os.path.join(run, "run_setup.json")))["config"]
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
    x0 = space.default_vector(seeded)
    out["seeded_defaults"] = seeded
    tmp = os.path.join(OUT, f"_default_{which}")
    for a in fr["annuli"]:
        ia, rin, rout = a["annulus"], a["inrad"], a["outrad"]
        contrast = a["contrast"]
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
        # --- the seeded default through the identical validation protocol
        cfg = RunConfig(ann_edges=list(setup.get("ann_edges") or [rin, rout]),
                        n_iter=1, n_init=1, seed=99 + ia,
                        n_sources=int(setup.get("n_sources", 3)),
                        validation=ValidationConfig(n_top=1, n_valid=n_trials),
                        save_fits=False, save_eval_images=False, write_setup_files=False)
        runner = Runner(red, space, obj, samp, cfg, tmp, log=lambda s: None)
        runner.ia = ia
        runner.contrast = contrast
        trials = []
        for _ in range(n_trials):
            r, _, _ = runner.evaluate(x0, "default", contrast=contrast, raw_only=True)
            if r.raw_score is not None and np.isfinite(r.raw_score):
                trials.append(float(r.raw_score))
        rec["default_trials"] = trials
        rec["default_score"] = float(np.median(trials)) if trials else float("nan")
        rec["gain"] = (rec["winner_score"] / rec["default_score"]) if trials and rec["default_score"] > 0 else None
        # --- the real companion, and the noise profile, in both clean images
        p0 = dict(space.decode(x0).params, inrad=rin, outrad=rout)
        img0 = red.reduce(ReductionRequest(params=p0)).image
        best = os.path.join(run, f"annulus{ia + 1:02d}", "best_clean.fits")
        img1 = np.asarray(fits.getdata(best), float) if os.path.exists(best) else None
        rec["default_params"] = {k: (v if isinstance(v, (str, bool)) else float(v))
                                 for k, v in p0.items()}
        rec["planet_snr_default"] = planet_snr(img0, red, planet)
        rec["planet_snr_optimized"] = None if img1 is None else planet_snr(img1, red, planet)
        r0, s0 = sigma_curve(img0, red, rin, rout, planet)
        rec["sigma_default"] = {"r_as": r0, "sigma": s0}
        if img1 is not None:
            r1, s1 = sigma_curve(img1, red, rin, rout, planet)
            rec["sigma_optimized"] = {"r_as": r1, "sigma": s1}
            rec["sigma_ratio_median"] = float(np.nanmedian(
                np.asarray(s0) / np.interp(r0, r1, s1)))
        log(f"  ann {ia + 1} [{rin:.0f}-{rout:.0f} px]: injected S/N {rec['default_score']:.2f} -> "
            f"{rec['winner_score']:.2f} (x{rec['gain'] or float('nan'):.2f}); "
            f"planet {rec['planet_snr_default']:.1f} -> "
            f"{-1 if rec['planet_snr_optimized'] is None else rec['planet_snr_optimized']:.1f}")
        out["annuli"].append(rec)

    anchor(out, red, space, planet, x0=x0)
    st = os.path.join(run, "klip_stitched.fits")
    if os.path.exists(st):
        out["stitched_planet_snr"] = planet_snr(np.asarray(fits.getdata(st), float), red, planet)
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


def anchor(out, red=None, space=None, planet=None, x0=None):
    """Check the contrast axis against the companion's published contrast (see ANCHOR), and
    rescale onto it only when ANCHOR_APPLY=1.

    Three sources are injected at the companion's own separation, 90/180/270 deg away, at
    the annulus' calibrated contrast, and reduced with the same default configuration.  The
    companion's contrast in this axis' units is ``c x peak_companion / median(peak_fake)``,
    with each fake's matched-filter peak read in (injected - clean) and the companion's in
    the radial-profile-subtracted clean image: same reduction, same separation, same kernel,
    so the KLIP throughput cancels.  The S/N values are recorded alongside for the record;
    they are NOT used for the scale (see the note above ANCHOR)."""
    from klip_tpe.metrics import mawet_peak_snr, radprof
    pub, ref = ANCHOR.get(out["target"], (None, None))
    out["anchor_reference"] = ref
    out["flux_scale_applied"] = 1.0
    if pub is None or red is None:
        return
    rho, pa = planet
    a = next((a for a in out["annuli"] if a["inrad_as"] <= rho <= a["outrad_as"]), None)
    if a is None or a.get("planet_snr_default") is None or not np.isfinite(a["planet_snr_default"]):
        out["flux_scale"] = None
        return
    c = float(a["contrast"])
    x0 = space.default_vector() if x0 is None else x0
    p0 = dict(space.decode(x0).params, inrad=a["inrad_px"], outrad=a["outrad_px"])
    clean = red.reduce(ReductionRequest(params=p0)).image
    srcs = [Source(rho, (pa + d) % 360.0, c) for d in (90.0, 180.0, 270.0)]
    img = red.reduce(ReductionRequest(params=p0, injections=srcs)).image
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
              "G": ("G2_bench_sphere",), "H": ("H2_bench_jwst",)}


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


if __name__ == "__main__":
    which = [w.upper() for w in sys.argv[1:]] or ["A", "C", "D", "B"]
    out = os.path.join(OUT, "summary.json")
    old = json.load(open(out)) if os.path.exists(out) else {}
    for w in which:
        try:
            old[w] = (collect_bench(bench_dir(w)) if w in BENCH_DIRS else collect(w))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            log(f"{w}: FAILED {exc!r}")
        json.dump(old, open(out, "w"), indent=1)
    log(f"wrote {out}")
