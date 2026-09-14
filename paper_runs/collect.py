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

#: Published contrast of the companion in each data set's own band.  The injection
#: templates carry no absolute photometry (the beta Pic template comes from a halo ratio,
#: the SPHERE one from a flux frame with its own DIT and neutral density, and the JWST one
#: has none at all), so every contrast axis is anchored on the companion itself: we measure
#: the companion's S/N and the injected S/N at a known contrast in the same annulus, which
#: gives the companion's contrast in template units, and rescale so that it equals the
#: published value.  Ratios -- gains, curve shapes -- are untouched by this.
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
    x0 = space.default_vector()
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

    anchor(out, red, space, planet)
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


def anchor(out, red=None, space=None, planet=None):
    """Put the contrast axis on the companion's published scale (see ANCHOR).

    The companion's contrast in template units is measured against injections *at its own
    separation* -- three sources on the same ring, 90/180/270 deg away -- reduced with the
    same (default) configuration, so the throughput of the comparison matches the
    companion's exactly and only the flux ratio is left."""
    pub, ref = ANCHOR.get(out["target"], (None, None))
    out["anchor_reference"] = ref
    if pub is None or red is None:
        return
    rho, pa = planet
    a = next((a for a in out["annuli"] if a["inrad_as"] <= rho <= a["outrad_as"]), None)
    if a is None or a.get("planet_snr_default") is None or not np.isfinite(a["planet_snr_default"]):
        out["flux_scale"] = None
        return
    c = float(a["contrast"])
    p0 = dict(space.decode(space.default_vector()).params,
              inrad=a["inrad_px"], outrad=a["outrad_px"])
    srcs = [Source(rho, (pa + d) % 360.0, c) for d in (90.0, 180.0, 270.0)]
    img = red.reduce(ReductionRequest(params=p0, injections=srcs)).image
    m = MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel,
                     known=[(rho, pa)])
    sn = m.per_source(img, None, [s.rho for s in srcs], [s.theta for s in srcs])
    sn = float(np.nanmedian(sn))
    if not np.isfinite(sn) or sn <= 0:
        out["flux_scale"] = None
        return
    implied = c * a["planet_snr_default"] / sn     # companion contrast in template units
    out.update(implied_companion_contrast=implied, published_companion_contrast=pub,
               anchor_inj_contrast=c, anchor_inj_snr=sn,
               anchor_companion_snr=a["planet_snr_default"], flux_scale=implied / pub)
    for aa in out["annuli"]:
        aa["contrast_anchored"] = aa["contrast"] / out["flux_scale"]
    log(f"  anchor: {sn:.2f} S/N for injections at {c:.3e} on the companion's ring vs "
        f"{a['planet_snr_default']:.2f} for the companion -> {implied:.3e} in template units, "
        f"published {pub:.3e} (scale / {out['flux_scale']:.4g})")


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
