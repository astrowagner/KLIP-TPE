#!/usr/bin/env python
"""The companion tests of the paper's Section 3: what the searched optimum does to a real source.

    python3 companion_tests.py                    # all three; each is skipped if its inputs are missing
    python3 companion_tests.py betapic hd95086    # or any subset

betapic   beta Pic b drops from S/N 15.3 (seeded default) to 6.4 (A2's middle-annulus winner)
          while injected sensitivity there rises x1.44.  Inject three sources at b's separation
          at (i) the annulus' calibrated contrast and (ii) b's contrast as this axis measures it
          (collect.py's companion check), and score default vs winner on the same draws.
hd95086   Section "Known sources distort ...": on C's outer-annulus winner, (a) the noise at the
          companion's separation with the companion left in the ring vs excluded, (b) one source
          at the calibrated contrast placed 1 FWHM from the companion, on either side, vs the
          same source 90/180/270 deg away.
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
from klip_tpe import CalibrationConfig, RunConfig, ValidationConfig   # noqa: E402
from klip_tpe.metrics import Source, mawet_peak_snr, radprof          # noqa: E402
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


def test_betapic(n_draws=8):
    """beta Pic b (A2, middle annulus): faint vs bright sources at the companion's separation."""
    which = "A2"
    _, fr, st = _run(which)
    red, space, obj, samp = C.build(which)
    a = fr["annuli"][1]
    runner = _runner(red, space, obj, samp, st, a["annulus"], a["contrast"], 7, 3, "_bright")
    runner._nsrc_cache[a["annulus"]] = 3
    x0 = runner._project(runner._default_vector(int(a["k_default"])), is_random=False)
    xw = C.run_vector(a["winner_x"], st["space"]["params"], space)
    rho_b = C.TARGETS[which][2][0]
    # b's contrast as this axis measures it: the companion check of collect.py (flux_scale x published)
    with open(os.path.join(C.OUT, "summary.json")) as f:
        s = json.load(f)[which]
    c_b = float(s["flux_scale"]) * C.ANCHOR["betapic"][0]
    out = {}
    for lab, c in (("calibrated", float(a["contrast"])), ("companion", c_b)):
        rng = np.random.default_rng(11)
        res = {"default": [], "winner": []}
        for _ in range(n_draws):
            src = samp.sample(3, rho_b, rho_b, rng, c)
            for nm, x in (("default", x0), ("winner", xw)):
                r, _, _ = runner.evaluate(x, "default", contrast=c, sources=src, raw_only=True)
                res[nm].append(float(r.raw_score))
        d, w = np.array(res["default"]), np.array(res["winner"])
        out[lab] = {"contrast": c, "rho_as": rho_b, "default": d.tolist(), "winner": w.tolist(),
                    "ratio_of_medians": float(np.median(w) / np.median(d)), "winner_better": int((w > d).sum())}
        print(f"betapic  {lab:10s} c={c:.3e} at {rho_b}\": default {np.median(d):.2f}  winner {np.median(w):.2f}  "
              f"x{np.median(w) / np.median(d):.2f}  winner better {int((w > d).sum())}/{n_draws}", flush=True)
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


TESTS = {"betapic": test_betapic, "hd95086": test_hd95086, "miri": test_miri}

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
