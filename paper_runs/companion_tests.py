#!/usr/bin/env python
"""The companion tests of the paper's Section 3: what the searched optimum does to a real source.

    python3 companion_tests.py                    # all four; each is skipped if its inputs are missing
    python3 companion_tests.py betapic hd95086    # or any subset

betapic   beta Pic b loses S/N in A2's middle-annulus winner while injected sensitivity there
          rises x1.44.  Inject one source at a time at b's separation, clear of b and the disk,
          at (i) the annulus' calibrated contrast and (ii) b's own contrast (collect.py's
          negative-companion fit), and measure default vs winner: metric, forward-model S/N and
          recovered peak (throughput).  Needs collect.py A2 first.
hd95086   Section "Known sources distort ...": on C's outer-annulus winner, (a) the noise at the
          companion's separation with the companion left in the ring vs excluded, (b) one source
          at the calibrated contrast placed 1 FWHM from the companion, on either side, vs the
          same source 90/180/270 deg away.
nircam    runs D / DK kept HIP 65426 b in the frames: the metric's ring scatter with it and
          without it, and the paired default-vs-winner comparison with it in the data and taken
          out (run_demos.hip65426b_negfc), at each annulus' calibrated contrast and, inside
          1.25", at the contrast D2 calibrated without it.  40 common draws.
miri      HIP 65426 b at F1140C sits at 0.82", inside the injection band of the inner annulus
          (sources at 1.2-1.75").  Inject at 0.823" at each run's calibrated contrast and at the
          companion's (Carter et al. 2023: dF1140C = 8.264 -> 4.95e-4), default vs winner, on
          both engines.  Needs MIRI_DATA and MIRI_RUNS as for miri_fig.py.

Every test rebuilds the run's own objective (collect.build / library_ablation._rebuild) and
re-scores with its own metric.  Results go to companion_tests.json beside this file.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import collect as C                                                   # noqa: E402
import run_demos as R                                                 # noqa: E402
from scipy import ndimage                                             # noqa: E402
from klip_tpe import CalibrationConfig, RunConfig, ValidationConfig   # noqa: E402
from klip_tpe.metrics import Source, mawet_peak_snr, radprof, source_xy   # noqa: E402
from klip_tpe.runner import Runner                                    # noqa: E402

OUTFILE = os.path.join(HERE, "companion_tests.json")


def _run(which):
    d = os.path.join(C.OUT, C.TARGETS[which][0])
    with open(os.path.join(d, "final_results.json")) as f:
        fr = json.load(f)
    with open(os.path.join(d, "run_setup.json")) as f:
        st = json.load(f)
    return d, fr, st


def _runner(red, space, obj, samp, st, ia, contrast, seed, n_sources, tag):
    cfg = RunConfig(ann_edges=list(st["config"]["ann_edges"]), n_iter=1, n_init=1, seed=seed,
                    n_sources=n_sources, defaults=dict(st["config"].get("defaults") or {}),
                    validation=ValidationConfig(n_top=1, n_valid=1),
                    save_fits=False, save_eval_images=False, write_setup_files=False)
    r = Runner(red, space, obj, samp, cfg, os.path.join(C.OUT, tag), log=lambda s: None)
    r.ia, r.contrast = ia, contrast
    return r


def test_betapic(n_pa=8):
    """beta Pic b (A2, middle annulus): faint vs bright sources at the companion's separation.

    ONE source per reduction, at ``n_pa`` position angles clear of b and of the disk wedges, at
    (i) the annulus' calibrated contrast and (ii) b's own contrast as the negative-companion fit
    in the seeded default measures it (collect.py's ``companion_fm_default``).  Each source is
    measured two ways: the run's metric (what validation scores), and the forward-model S/N of
    the paper's companion values -- its matched-filter peak in (injected - clean) over the ring
    scatter of the clean image, b excluded -- with the recovered peak itself as the throughput.
    Until 2026-09-26 this injected THREE sources per reduction at flux_scale x the published
    contrast (5.5e-4), which put three bright sources in one ring and one basis and set their
    brightness from a peak ratio that b's own extra self-subtraction biases low."""
    from klip_tpe.companion import ring_sigma
    which = "A2"
    _, fr, st = _run(which)
    red, space, obj, samp = C.build(which)
    m = obj.metric
    a = fr["annuli"][1]
    runner = _runner(red, space, obj, samp, st, a["annulus"], a["contrast"], 7, 1, "_bright")
    x0 = runner._project(runner._default_vector(int(a["k_default"])), is_random=False)
    xw = C.run_vector(a["winner_x"], st["space"]["params"], space)
    cfgs = {"default": space.decode(x0), "winner": space.decode(xw)}
    rho_b, pa_b = C.TARGETS[which][2]
    with open(os.path.join(C.OUT, "summary.json")) as f:
        s = json.load(f)[which]
    ann = next(x for x in s["annuli"] if x["annulus"] == a["annulus"])
    fm = ann.get("companion_fm_default") or {}
    if not fm:
        raise KeyError("summary.json has no companion_fm_default for A2 -- run collect.py A2 first")
    c_b = float(np.median(list(fm["contrast"].values())))
    rho_b, pa_b = float(fm["rho_as"]), float(fm["pa_deg"])
    fpa = list(R.bp_disk(red)[0])
    pas = []
    for k in range(72):
        p = (pa_b + 40.0 + k * 5.0) % 360.0
        if abs((p - pa_b + 180.0) % 360.0 - 180.0) < 40.0:
            continue
        if any(abs((p - c0 + 180.0) % 360.0 - 180.0) <= hw for c0, hw in fpa):
            continue
        pas.append(p)
    pas = [pas[i] for i in np.linspace(0, len(pas) - 1, n_pa).round().astype(int)]
    clean = {k: np.asarray(runner._reduce(c, None, tag=f"bp_{k}_clean").image, float) for k, c in cfgs.items()}
    out = {"pas": pas, "companion_contrast": c_b, "rho_as": rho_b}
    for lab, cc in (("calibrated", float(a["contrast"])), ("companion", c_b)):
        res = {k: {"metric": [], "fm": [], "peak": []} for k in cfgs}
        for pa in pas:
            for k, c in cfgs.items():
                img = np.asarray(runner._reduce(c, [Source(rho_b, pa, cc)], tag=f"bp_{k}_{lab}").image, float)
                res[k]["metric"].append(float(m.per_source(img, None, [rho_b], [pa])[0]))
                d = img - clean[k]
                A = ndimage.convolve(np.where(np.isfinite(d), d, 0.0), m.kernel([rho_b]), mode="nearest")
                cx, cy = (d.shape[1] - 1) / 2.0, (d.shape[0] - 1) / 2.0
                xs, ys = source_xy([rho_b], [pa], red.pxscale, cx, cy, red.angle_convention)
                xi, yi = int(round(xs[0])), int(round(ys[0]))
                pk = float(np.max(A[yi - 1:yi + 2, xi - 1:xi + 2]))
                sg = ring_sigma(clean[k], rho_b, pa, red.fwhm, red.pxscale, m.kernel([rho_b]), pixel_mask=m.pixel_mask,
                                angle_convention=red.angle_convention, exclude=[(rho_b, pa_b)])["sigma"]
                res[k]["peak"].append(pk)
                res[k]["fm"].append(pk / sg)
        row = {"contrast": cc}
        for q in ("metric", "fm", "peak"):
            dd, ww = np.array(res["default"][q]), np.array(res["winner"][q])
            row[q] = {"default": dd.tolist(), "winner": ww.tolist(), "ratio_of_medians": float(np.median(ww) / np.median(dd)),
                      "winner_better": int((ww > dd).sum())}
        out[lab] = row
        print(f"betapic  {lab:10s} c={cc:.3e} at {rho_b:.3f} arcsec, {len(pas)} PAs: metric S/N "
              f"{np.median(res['default']['metric']):.2f} -> {np.median(res['winner']['metric']):.2f} "
              f"(x{row['metric']['ratio_of_medians']:.2f}, {row['metric']['winner_better']}/{len(pas)}); forward-model S/N "
              f"{np.median(res['default']['fm']):.2f} -> {np.median(res['winner']['fm']):.2f} "
              f"(x{row['fm']['ratio_of_medians']:.2f}, {row['fm']['winner_better']}/{len(pas)}); recovered peak "
              f"x{row['peak']['ratio_of_medians']:.2f}", flush=True)
    for k in cfgs:
        lin = (np.median(out["companion"]["peak"][k]) / np.median(out["calibrated"]["peak"][k])
               / (out["companion"]["contrast"] / out["calibrated"]["contrast"]))
        out[f"linearity_{k}"] = float(lin)
        print(f"betapic  {k}: recovered peak at b's contrast / at the calibrated one, per unit contrast = {lin:.2f}",
              flush=True)
    return out


def test_hd95086():
    """HD 95086 b (C, outer annulus): noise inflation and on-companion injections."""
    from astropy.io import fits
    which = "C"
    d, fr, st = _run(which)
    red, space, obj, samp = C.build(which)
    m = obj.metric
    rho_c, pa_c = C.TARGETS[which][2]
    a = fr["annuli"][1]
    ia = a["annulus"]
    img = np.asarray(fits.getdata(os.path.join(d, f"annulus{ia + 1:02d}", "best_clean.fits")), float)
    im = radprof(img) if m.flatten else img
    probes = [pa_c + dp for dp in (60, 120, 180, 240, 300)]
    sig = {}
    for lab, known in (("excluded", [(rho_c, pa_c)]), ("included", [])):
        _, det = mawet_peak_snr(im, [rho_c] * len(probes), probes, m.pxscale, m.fwhm, kernel=m.kernel([rho_c]),
                                search_px=m.search_px, excl_fwhm=m.excl_fwhm, known=known,
                                declip_nsig=m.declip_nsig, band_fwhm=m.band_fwhm, min_ring=m.min_ring,
                                penalty=m.penalty, pixel_mask=m.pixel_mask, angle_convention=m.angle_convention,
                                return_details=True)
        sig[lab] = [float(x["sigma"]) for x in det]
    ratio = float(np.median(sig["included"]) / np.median(sig["excluded"]))
    print(f"hd95086  sigma at {rho_c}\": companion in the ring / excluded = {ratio:.2f}", flush=True)
    runner = _runner(red, space, obj, samp, st, ia, a["contrast"], 5, 1, "_known")
    xw = C.run_vector(a["winner_x"], st["space"]["params"], space)
    dpa = float(np.degrees(red.fwhm * red.pxscale / rho_c))        # 1 FWHM in azimuth at rho_c
    inj = {}
    for lab, pa in (("near_minus", pa_c - dpa), ("near_plus", pa_c + dpa), ("p90", pa_c + 90),
                    ("p180", pa_c + 180), ("p270", pa_c + 270)):
        r, _, _ = runner.evaluate(xw, "default", contrast=a["contrast"],
                                  sources=[Source(rho_c, pa, a["contrast"])], raw_only=True)
        inj[lab] = float(r.raw_score)
    print(f"hd95086  injection 1 FWHM from the companion: S/N {inj['near_minus']:.2f} / {inj['near_plus']:.2f};"
          f" at 90/180/270 deg: {inj['p90']:.2f} / {inj['p180']:.2f} / {inj['p270']:.2f}", flush=True)
    return {"sigma": sig, "sigma_ratio": ratio, "injections": inj, "contrast": a["contrast"], "dpa_deg": dpa}


def test_miri(n_draws=10):
    """HIP 65426 b in F1140C: default vs winner at the companion's separation, both engines."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
    import library_ablation as LA
    data = os.path.expanduser(os.environ.get("MIRI_DATA", "~/Data/JWST/hip65426_miri"))
    runs = os.path.expanduser(os.environ.get("MIRI_RUNS", os.path.dirname(HERE)))
    out = {}
    for engine in ("pyklip", "klip"):
        run_dir = os.path.join(runs, f"miri_HIP-65426_F1140C_v7_{engine}")
        with open(os.path.join(run_dir, "run_setup.json")) as f:
            setup = json.load(f)["config"]
        with open(os.path.join(run_dir, "final_results.json")) as f:
            fr = json.load(f)["annuli"][0]
        a = type("A", (), dict(data=data, workers="auto", backend=engine))()
        rb = LA._rebuild("miri", setup, a, lambda s: None)
        red, ann, obj, samp, space = (rb[k] for k in ("red", "ann", "obj", "samp", "space"))
        cfg = RunConfig(ann_edges=[float(v) for v in ann], n_iter=1, n_init=1, seed=3,
                        validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[1e-4]),
                        n_remeasure=1, n_sources=2, save_fits=False, save_eval_images=False, fm_curve=False, verify=False)
        runner = Runner(red, space, obj, samp, cfg, os.path.join(run_dir, "_bright"), log=lambda s: None)
        runner.ia, runner.contrast = 0, float(fr["contrast"])
        x_def = space.default_vector()
        cand = {"default": LA._vec(runner, x_def),
                "winner": LA._vec(runner, LA._x_from_params(space, fr["winner_config"]["params"], x_def))}
        rho = LA.PLANET[0]
        for lab, c in (("calibrated", float(fr["contrast"])), ("companion", 4.95e-4)):
            rng = np.random.default_rng(17)
            sc = {k: [] for k in cand}
            for _ in range(n_draws):
                src = samp.sample(2, rho, rho, rng, c)
                for k, x in cand.items():
                    r, _, _ = runner.evaluate(x, "default", contrast=c, sources=src, raw_only=True)
                    sc[k].append(float(r.raw_score))
            d, w = np.array(sc["default"]), np.array(sc["winner"])
            out[f"{engine}_{lab}"] = {"contrast": c, "rho_as": rho, "default": d.tolist(), "winner": w.tolist(),
                                      "ratio_of_medians": float(np.median(w) / np.median(d)),
                                      "winner_better": int((w > d).sum())}
            print(f"miri     {engine:6s} {lab:10s} c={c:.2e} at {rho}\": default {np.median(d):.2f}  winner "
                  f"{np.median(w):.2f}  x{np.median(w) / np.median(d):.2f}  winner better {int((w > d).sum())}/{n_draws}",
                  flush=True)
    return out


def test_nircam(n_draws=40):
    """HIP 65426 b's light in runs D / DK (NIRCam, companion kept in the frames).

    (a) the metric's ring scatter across D's annuli with the companion in the data and taken
    out (negative injection at run_demos.hip65426b_negfc), seeded default and winner; (b) the
    paired default-vs-winner comparison of collect.py, on ``n_draws`` common draws at each
    annulus' calibrated contrast, with the companion in the data (what the searches optimized)
    and taken out -- only the positive injections are scored -- and, inside 1.25", also at the
    contrast D2 calibrated once the companion was gone."""
    from klip_tpe.companion import ring_sigma
    f = R.hip65426b_negfc()
    neg = [Source(float(f["rho"]), float(f["pa"]), -float(f["contrast"]))]
    out = {"negfc": {k: f[k] for k in ("rho", "pa", "contrast")}}
    c_d2 = None
    try:
        with open(os.path.join(C.OUT, "summary.json")) as fh:
            c_d2 = float(json.load(fh)["D2"]["annuli"][0]["contrast"])
    except (OSError, KeyError, IndexError):
        pass
    for which in ("D", "DK"):
        _, fr, st = _run(which)
        red, space, obj, samp = C.build(which)
        m = obj.metric
        rho_b, pa_b = C.TARGETS[which][2]
        res = {}
        for a in fr["annuli"]:
            ia = a["annulus"]
            runner = _runner(red, space, obj, samp, st, ia, a["contrast"], 1234 + ia, int(st["config"].get("n_sources") or 2),
                             "_nircam")
            x0 = runner._project(runner._default_vector(int(a.get("k_default") or 10)), is_random=False)
            xw = C.run_vector(a["winner_x"], st["space"]["params"], space)
            cfgs = {"default": space.decode(x0), "winner": space.decode(xw)}
            # (a) ring scatter at three separations across the annulus
            seps = np.linspace(a["inrad"] * red.pxscale + 0.1, a["outrad"] * red.pxscale - 0.1, 3)
            sig = {}
            for k, c in cfgs.items():
                cl = np.asarray(runner._reduce(c, None, tag="nc_cl").image, float)
                rm = np.asarray(runner._reduce(c, neg, tag="nc_rm").image, float)
                sig[k] = [float(ring_sigma(cl, r, pa_b + 90.0, red.fwhm, red.pxscale, m.kernel([r]), exclude=[(rho_b, pa_b)],
                                           angle_convention=red.angle_convention)["sigma"]
                                / ring_sigma(rm, r, pa_b + 90.0, red.fwhm, red.pxscale, m.kernel([r]), exclude=[(rho_b, pa_b)],
                                             angle_convention=red.angle_convention)["sigma"]) for r in seps]
            # (b) paired comparison, companion in / out
            n = runner._nsrc(ia)
            rlo, rhi = runner._band(ia, cfgs["default"])
            levels = [("calibrated", float(a["contrast"]))]
            if ia == 0 and c_d2:
                levels.append(("d2_calibrated", c_d2))
            pair = {}
            for lab, cc in levels:
                rng = np.random.default_rng(777 + ia)
                draws = [samp.sample(n, rlo, rhi, rng, cc) for _ in range(n_draws)]
                sc = {(k, rem): [] for k in cfgs for rem in (False, True)}
                for src in draws:
                    for k, c in cfgs.items():
                        for rem in (False, True):
                            img = runner._reduce(c, list(src) + (neg if rem else []), tag="nc_pair").image
                            sc[(k, rem)].append(float(obj.score_raw(np.asarray(img, float), src).score))
                for rem in (False, True):
                    d, w = np.array(sc[("default", rem)]), np.array(sc[("winner", rem)])
                    pair[f"{lab}_{'removed' if rem else 'in'}"] = {
                        "contrast": cc, "default": float(np.median(d)), "winner": float(np.median(w)),
                        "ratio_of_medians": float(np.median(w) / np.median(d)), "winner_better": int((w > d).sum()),
                        "n": int(len(d))}
            res[f"annulus{ia + 1}"] = {"sigma_ratio_clean_over_removed": sig, "seps_as": seps.tolist(), "paired": pair}
            for key, v in pair.items():
                print(f"nircam   {which:2s} ann {ia + 1} {key:22s} c={v['contrast']:.2e}: default {v['default']:.2f} winner "
                      f"{v['winner']:.2f}  x{v['ratio_of_medians']:.2f}  winner better {v['winner_better']}/{v['n']}", flush=True)
            print(f"nircam   {which:2s} ann {ia + 1} ring scatter, companion in / removed, at {np.round(seps, 2).tolist()} arcsec: "
                  f"default {np.round(sig['default'], 1).tolist()}  winner {np.round(sig['winner'], 1).tolist()}", flush=True)
        out[which] = res
    return out


TESTS = {"betapic": test_betapic, "hd95086": test_hd95086, "miri": test_miri, "nircam": test_nircam}

if __name__ == "__main__":
    which = [w.lower() for w in sys.argv[1:]] or list(TESTS)
    res = {}
    if os.path.exists(OUTFILE):
        with open(OUTFILE) as f:
            res = json.load(f)
    for w in which:
        try:
            res[w] = TESTS[w]()
        except (FileNotFoundError, KeyError, SystemExit) as exc:
            print(f"{w}: skipped -- {exc!r}")
    with open(OUTFILE, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {OUTFILE}")
