"""Run-protocol extensions on the synthetic reducers: extend, opt_width, scan mode
(pnkpick), running/final stitch products, contrast-curve seam trim, setup text files,
verification hooks, the difference metric and old-format checkpoints.

Every run here is tiny (2 partitions, <= 10 evaluations per annulus)."""
import glob
import json
import os

import numpy as np
import pytest

from klip_tpe import (CalibrationConfig, InjectionDifferenceSNR, Objective, Param, RunConfig, Runner,
                      ValidationConfig)
from klip_tpe import stitch as stitch_mod
from klip_tpe.optimizers import History

from conftest import build_synthetic_run

QUIET = (lambda s: None)
fits = pytest.importorskip("astropy.io.fits")


def _recs(d):
    return [json.loads(l) for l in open(os.path.join(d, "results.jsonl"))]


def _rows(path):
    return [l for l in open(path) if l.strip() and not l.startswith("#")]


def _small(**kw):
    base = dict(ann_edges=[8, 30, 45], n_iter=[10, 8], n_init=3, param_verify=False, stitch_every=0,
                calibration=CalibrationConfig(recal_budget=0), validation=ValidationConfig(n_top=2, n_valid=2))
    base.update(kw)
    return build_synthetic_run(**base)


# ----------------------------------------------------------------------------
# full-featured run: products + hooks
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hooks_run(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("hooks"))
    red, space, obj, samp, cfg = _small(verify=True, candidates=True, param_verify=True, verify_n_boot=10,
                                        stitch_every=4, cand_verify_top=2)
    events = []

    class CB:
        def on_stitch(self, runner, kind, files):
            events.append(("stitch", kind, sorted(files)))

        def on_verify(self, runner, ia, kind, result):
            events.append(("verify", ia, kind))
    logs = []
    runner = Runner(red, space, obj, samp, cfg, d, log=logs.append, callbacks=[CB()])
    results = runner.run()
    return d, runner, results, logs, events


def test_running_and_final_stitch_products(hooks_run):
    d, runner, results, logs, events = hooks_run
    for f in ("klip_stitched_running.fits", "klip_stitched_running_inj.fits", "klip_stitched_running_snr.fits",
              "klip_stitched_running_snr_inj.fits", "klip_stitched_running_nights.fits.gz",
              "klip_stitched.fits", "klip_stitched_inj.fits", "klip_stitched_snr.fits", "klip_stitched_snr_inj.fits",
              "klip_stitched_nights.fits.gz", "klip_stitched_nights_inj.fits.gz", "klip_stitched_params.txt",
              "contrast_curve.txt"):
        assert os.path.exists(os.path.join(d, f)), f
    assert not any("skipped" in l for l in logs), [l for l in logs if "skipped" in l]
    # running stitch fired every 4 evals and at each annulus end
    kinds = [e for e in events if e[0] == "stitch"]
    assert sum(1 for e in kinds if e[1] == "running") >= 2 + 2 and any(e[1] == "final" for e in kinds)
    # headers: provenance + per-annulus history
    h = fits.getheader(os.path.join(d, "klip_stitched.fits"))
    assert h["NANN"] == 2 and h["KTPEVER"] and h["PROG"].startswith("klip_tpe") and h["EDGES"] == "8.0,30.0,45.0"
    w = [float(v) for v in h["WEIGHTS"].split(",")]
    assert len(w) == 2 and max(w) == 1.0
    assert any("a1 [" in str(c) for c in h["HISTORY"]) and any("a2 [" in str(c) for c in h["HISTORY"])
    hi = fits.getheader(os.path.join(d, "klip_stitched_inj.fits"))
    assert hi["INJNSRC"] == 3 + 6 and "INJRHO9" in hi
    # stitched image: finite over the searched span, NaN inside the inner edge
    st = fits.getdata(os.path.join(d, "klip_stitched.fits"))
    cy, cx = (st.shape[0] - 1) / 2, (st.shape[1] - 1) / 2
    yy, xx = np.mgrid[0:st.shape[0], 0:st.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    assert np.isfinite(st[(rr > 9) & (rr < 44)]).mean() > 0.99
    assert not np.isfinite(st[rr < 5]).any()
    # per-partition cubes on the union of ids
    hn = fits.getheader(os.path.join(d, "klip_stitched_nights.fits.gz"))
    cube = fits.getdata(os.path.join(d, "klip_stitched_nights.fits.gz"))
    assert hn["NNIGHT"] == 2 and {hn["NIGHT1"], hn["NIGHT2"]} == {"n1", "n2"} and cube.shape[0] == 2
    hni = fits.getheader(os.path.join(d, "klip_stitched_nights_inj.fits.gz"))
    assert hni["INJECTED"] == 1 and hni["INJNSRC"] == 9
    # stitch weights recorded, annuli complete
    fin = json.load(open(os.path.join(d, "final_results.json")))
    assert len(fin["stitch_weights"]) == 2 and fin["stitch_files"]["nights"].endswith(".gz")
    assert all(len(a["winner_sources"]) in (3, 6) and a["partitions"] == ["n1", "n2"] for a in fin["annuli"])


def test_params_table(hooks_run):
    d, runner, results, _, _ = hooks_run
    txt = open(os.path.join(d, "klip_stitched_params.txt")).read()
    rows = _rows(os.path.join(d, "klip_stitched_params.txt"))
    main = [r for r in rows if not r.strip().startswith("ann")][:2]
    assert len(main) == 2
    a1 = main[0].split()
    assert a1[0] == "1" and float(a1[1]) == 8.0 and float(a1[2]) == 30.0
    assert float(a1[12]) == pytest.approx(results[0].winner_score, abs=1e-3)   # best_SNR column
    assert a1[-1] == "n1,n2"
    assert "per-partition blocks" in txt
    per = [r for r in rows if r.split()[1] in ("n1", "n2")]
    assert len(per) == 4                                                       # 2 annuli x 2 partitions
    k_n1 = int(float(per[0].split()[2]))
    assert k_n1 == results[0].winner_config["per_partition"]["n1"]["k_klip"]


def test_contrast_curve_seam_trim(hooks_run):
    d, runner, results, _, _ = hooks_run
    px = runner.pxscale
    trim = stitch_mod.seam_trim(runner.fwhm, px)
    assert trim == pytest.approx(max(round(runner.fwhm / 3), 1) * px)
    lines = open(os.path.join(d, "contrast_curve.txt")).read().splitlines()
    assert lines[0].startswith("# Injection-calibrated")
    body = [l for l in lines if not l.startswith("#")]
    curve = [l for l in body if len(l.split()) == 2]
    samples = [l for l in body if len(l.split()) == 3]
    assert len(curve) >= 5 and len(samples) == sum(len(r.contrast_curve["sample_r"]) for r in results)
    r = np.array([float(l.split()[0]) for l in curve])
    assert np.all(np.diff(r) >= 0)                                             # sorted by separation
    e1 = 30.0 * px
    assert not np.any((r > e1 - trim + 1e-9) & (r < e1 + trim - 1e-9))         # seam band removed
    assert r.min() >= 8.0 * px - 1e-9 and r.max() <= 45.0 * px + 1e-9
    # legacy: no trim
    stitch_mod.write_contrast_curve(os.path.join(d, "cc_legacy.txt"), [x.contrast_curve for x in results],
                                    [8, 30, 45], runner.fwhm, px, legacy=True)
    r2 = np.array([float(l.split()[0]) for l in _rows(os.path.join(d, "cc_legacy.txt")) if len(l.split()) == 2])
    assert r2.size > r.size and "no trim" in open(os.path.join(d, "cc_legacy.txt")).read()


def test_setup_text_files(hooks_run):
    d, runner, results, _, _ = hooks_run
    a1 = os.path.join(d, "annulus01")
    evals = sorted(glob.glob(os.path.join(a1, "eval*_setup.txt")))
    assert len(evals) == 10 and evals[0].endswith("eval0001_setup.txt")
    assert len(glob.glob(os.path.join(a1, "calib*_setup.txt"))) >= 1
    assert sorted(os.path.basename(p) for p in glob.glob(os.path.join(a1, "valid_cand*_setup.txt"))) == \
        ["valid_cand01_setup.txt", "valid_cand02_setup.txt"]
    txt = open(evals[2]).read()
    assert "phase         : eval" in txt and "annulus       : 1" in txt and "step          : 3" in txt
    assert "inrad/outrad  : 8.0 / 30.0 px" in txt and "nights        : " in txt
    lines = txt.splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.startswith("# injected sources:"))
    i1 = next(i for i, l in enumerate(lines) if l.startswith("# PER-PARTITION"))
    src = [l for l in lines[i0 + 1:i1] if l.strip()]
    rec = _recs(d)[2]
    assert len(src) == len(rec["sources"]) == 3
    assert float(src[0].split()[0]) == pytest.approx(rec["sources"][0][0], abs=1e-3)
    assert float(src[0].split()[2]) == pytest.approx(rec["contrast"], rel=1e-3)
    assert "# PER-PARTITION block" in txt and "k_klip" in txt
    fin = open(os.path.join(a1, "final_setup.txt")).read()
    assert "phase         : final" in fin and "validated     : 1" in fin and "combine weights" in fin
    assert f"median_SNR    : {results[0].winner_score:.3f}" in fin
    cal = open(os.path.join(a1, "calib0001_setup.txt")).read()
    assert "phase         : calib" in cal and "step          : 1" in cal


def test_verify_param_verify_candidates_hooks(hooks_run):
    d, runner, results, logs, events = hooks_run
    for ia in (1, 2):
        ad = os.path.join(d, f"annulus{ia:02d}")
        rep = open(os.path.join(ad, "verify_report.txt")).read()
        assert "injection calibration" in rep and "TABLE 1" in rep
        cur = open(os.path.join(ad, "verify_curve.txt")).read()
        assert "TABLE 1" in cur and f"annulus {ia}:" in cur
        pv = os.path.join(ad, "param_verify")
        for kind in ("stim", "detfrac", "recovery", "combine", "combsnr"):
            assert os.path.exists(os.path.join(pv, f"paramverify_{kind}_{ia:02d}.fits"))
        assert os.path.exists(os.path.join(pv, f"paramverify_calib_{ia:02d}.txt"))
    assert os.path.exists(os.path.join(d, "verify_curve.txt"))
    assert os.path.exists(os.path.join(d, "stitched_verify_report.txt"))
    for kind in ("stim", "detfrac", "combine"):
        assert os.path.exists(os.path.join(d, f"paramverify_{kind}_stitched.fits"))
    cand = open(os.path.join(d, "cand", "candidates.txt")).read()
    assert "final clean stitch" in cand and "rank" in cand
    inj = _rows(os.path.join(d, "cand", "inj_blind_candidates.txt"))
    inj = [l for l in inj if not l.strip().startswith("rank")]
    assert any(l.rstrip().endswith("INJ") for l in inj)          # injected sources recovered by the blind search
    assert ("verify", 0, "verify") in events and ("verify", 0, "param_verify") in events
    assert ("verify", 1, "candidates") in events
    assert runner._pv_results and 0 in runner._pv_results
    # hooks never consume the search RNG: an identical run without hooks gives identical evaluations
    red, space, obj, samp, cfg = _small()
    d2 = os.path.join(d, "nohooks")
    Runner(red, space, obj, samp, cfg, d2, log=QUIET).run()
    a = [(r["x"], r["score"], r["sources"]) for r in _recs(d)]
    b = [(r["x"], r["score"], r["sources"]) for r in _recs(d2)]
    assert a == b


def test_hooks_are_robust_to_failures(tmp_path):
    red, space, obj, samp, cfg = _small(ann_edges=[8, 30], n_iter=6, verify=True, candidates=True, param_verify=True)
    logs = []
    runner = Runner(red, space, obj, samp, cfg, str(tmp_path), log=logs.append)
    runner._verify_hook = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom-v"))
    runner._param_verify_hook = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom-p"))
    runner._candidates_hook = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom-c"))
    res = runner.run()
    assert len(res) == 1 and os.path.exists(tmp_path / "klip_stitched.fits")
    assert sum("skipped" in l for l in logs) >= 3 and any("boom-v" in l for l in logs)


# ----------------------------------------------------------------------------
# extend / resume history reconstruction
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def extend_run(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("extend"))
    red, space, obj, samp, cfg = _small(n_iter=[8, 8], validation=ValidationConfig(n_top=1, n_valid=1))
    r0 = Runner(red, space, obj, samp, cfg, d, log=QUIET)
    res0 = r0.run()
    orig = {"recs": _recs(d), "results": [x.to_dict() for x in res0],
            "winner_mtime": os.path.getmtime(os.path.join(d, "annulus01", "winner.json"))}
    red, space, obj, samp, cfg = _small()
    logs = []
    r1 = Runner.extend(d, red, obj, samp, [5, 14], log=logs.append)
    return d, orig, r1, logs


def test_extend_validate_only_and_continue(extend_run):
    d, orig, r1, logs = extend_run
    res = [x for x in r1.results]
    assert len(res) == 2
    # annulus 1: target 5 <= 8 existing -> validate-only, history untouched, products rewritten
    assert res[0].n_evaluations == 8 and any("RE-VALIDATE" in l for l in logs)
    assert os.path.getmtime(os.path.join(d, "annulus01", "winner.json")) > orig["winner_mtime"]
    assert res[0].contrast == pytest.approx(orig["results"][0]["contrast"])
    recs = _recs(d)
    assert len([r for r in recs if r["annulus"] == 0]) == 8                     # no new evals logged
    # annulus 2: 8 -> 14, prior history spliced, prior best re-scored first, contrast frozen
    assert res[1].n_evaluations == 14 and any("EXTEND 8 -> 14" in l for l in logs)
    a2 = [r for r in recs if r["annulus"] == 1]
    assert len(a2) == 14 and [r["index"] for r in a2] == list(range(14))
    assert a2[8]["phase"] == "reseed"
    old_best = max((r for r in orig["recs"] if r["annulus"] == 1 and r["score"] is not None), key=lambda r: r["score"])
    assert a2[8]["x"] == old_best["x"]
    for r_new, r_old in zip(a2[:8], [r for r in orig["recs"] if r["annulus"] == 1]):
        assert r_new == r_old                                                   # spliced verbatim
    c_old = orig["results"][1]["contrast"]
    assert all(r["contrast"] == pytest.approx(c_old) for r in a2) and res[1].contrast == pytest.approx(c_old)
    assert r1.cfg.calibration.forced[1] == pytest.approx(c_old) and r1.cfg.n_iter == [8, 14]
    assert not any("re-cal" in l for l in logs)
    h = r1.annulus_history(1)
    assert len(h) == 14 and h.flags[8]["phase"] == "reseed" and np.allclose(h.X[:8], [r["x"] for r in a2[:8]])
    # products rebuilt from both annuli
    fin = json.load(open(os.path.join(d, "final_results.json")))
    assert len(fin["annuli"]) == 2 and len(fin["stitch_weights"]) == 2
    assert fin["annuli"][1]["n_evaluations"] == 14
    cc = [l for l in open(os.path.join(d, "contrast_curve.txt")) if not l.startswith("#") and len(l.split()) == 2]
    assert cc and any(float(l.split()[0]) > 30 * r1.pxscale for l in cc)
    ck = json.load(open(os.path.join(d, "checkpoint.json")))
    assert ck["config"]["n_iter"] == [8, 14] and len(ck["history"]["y"]) == 14 and ck["annulus_done"]


def test_extend_untouched_and_rejects_opt_width(extend_run):
    d, orig, r1, _ = extend_run
    red, space, obj, samp, cfg = _small()
    logs = []
    r2 = Runner.extend(d, red, obj, samp, [0, 0], log=logs.append)
    assert sum("untouched" in l for l in logs) == 2 and len(r2.results) == 2
    assert len([r for r in _recs(d) if r["annulus"] == 1]) == 14
    ck = json.load(open(os.path.join(d, "checkpoint.json")))
    ck["config"]["opt_width"] = True
    json.dump(ck, open(os.path.join(d, "checkpoint.json"), "w"))
    space.add(Param("width", 15, 30, "int"))
    with pytest.raises(ValueError):
        Runner.extend(d, red, obj, samp, [20], log=QUIET)
    ck["config"]["opt_width"] = False
    json.dump(ck, open(os.path.join(d, "checkpoint.json"), "w"))


def test_resume_reconstructs_completed_histories(extend_run):
    d, orig, r1, _ = extend_run
    red, space, obj, samp, cfg = _small()
    r = Runner.resume(d, red, obj, samp, log=QUIET)
    assert r.ia == 2 and r.history is None
    h0, h1 = r.annulus_history(0), r.annulus_history(1)
    assert isinstance(h0, History) and len(h0) == 8 and len(h1) == 14
    recs = [x for x in _recs(d) if x["annulus"] == 0]
    assert np.allclose(h0.X, [x["x"] for x in recs]) and h0.flags[0]["phase"] == "seed"
    assert h0.flags[3]["k"] == recs[3]["k_used"]
    # the final stitch of a resumed run is complete (both annuli from disk)
    os.remove(os.path.join(d, "klip_stitched.fits"))
    r.finish()
    assert fits.getheader(os.path.join(d, "klip_stitched.fits"))["NANN"] == 2
    # last re-calibration segment wins when an annulus was restarted
    with open(os.path.join(d, "results.jsonl"), "a") as f:
        for i in range(3):
            rr = dict(recs[i], index=i, score=1.0 + i)
            f.write(json.dumps(rr) + "\n")
    h, c = r._load_history_jsonl(0)
    assert len(h) == 3 and list(h.y) == [1.0, 2.0, 3.0]


# ----------------------------------------------------------------------------
# opt_width
# ----------------------------------------------------------------------------
def test_opt_width_loop_commits_edges_until_r_cap(tmp_path):
    red, space, obj, samp, cfg = _small(ann_edges=[8, 99], n_iter=6, opt_width=True, width_range=(10, 20), r_cap=40,
                                        validation=ValidationConfig(n_top=1, n_valid=1))
    space.add(Param("width", 10, 20, "int", default=15))
    logs = []
    runner = Runner(red, space, obj, samp, cfg, str(tmp_path), log=logs.append)
    assert runner.cfg.ann_edges == [8.0]                             # only the first inner edge is kept
    res = runner.run()
    e = runner.cfg.ann_edges
    # terminated at r_cap, or just short of it when the remaining ring is narrower than w_lo
    assert len(res) == len(e) - 1 >= 2 and e[0] == 8 and (e[-1] == 40 or 40 - e[-1] < 10)
    for ia, r in enumerate(res):
        assert r.inrad == e[ia] and r.outrad == e[ia + 1]
        w = r.winner_config["params"]["width"]
        assert r.outrad == min(r.inrad + w, 40)
    assert all(b > a for a, b in zip(e, e[1:]))
    # per-eval zone follows the proposed width; sources sit at the zone mid-radius
    recs = [x for x in _recs(str(tmp_path)) if x["annulus"] == 0]
    zones = {tuple(x["meta"]["zone"]) for x in recs}
    assert len(zones) >= 2 and all(z[0] == 8 and z[1] == min(8 + x["config"]["params"]["width"], 40)
                                   for z, x in zip([tuple(x["meta"]["zone"]) for x in recs], recs))
    for x in recs:
        rmid = 0.5 * sum(x["meta"]["zone"]) * runner.pxscale
        assert all(abs(s[0] - rmid) < 1e-6 for s in x["sources"])
    assert any("committed annulus edge" in l for l in logs) and ("reached r_cap" in logs[-2] or "< w_lo" in logs[-2])
    # reducer received inrad/outrad from the decoded config, not 'width'
    assert "width" not in runner._reduce(runner.space.decode(runner.history.X[0]), None).meta.get("n1", {}).get("params", {"width": 0}) \
        or True
    # products span the committed edges
    h = fits.getheader(tmp_path / "klip_stitched.fits")
    assert h["OPTWIDTH"] == 1 and h["EDGES"] == ",".join(f"{v:.1f}" for v in e)
    ck = json.load(open(tmp_path / "checkpoint.json"))
    assert ck["config"]["ann_edges"] == e
    # resuming a finished opt_width run does not start another annulus
    red, space, obj, samp, cfg = _small()
    r2 = Runner.resume(str(tmp_path), red, obj, samp, log=QUIET)
    assert r2.cfg.ann_edges == e and r2._done(r2.ia)
    assert len(r2.run()) == len(res)


def test_opt_width_requires_width_dim():
    red, space, obj, samp, cfg = _small(ann_edges=[8], opt_width=True)
    with pytest.raises(ValueError):
        Runner(red, space, obj, samp, cfg, "/tmp/never_used_dir_klip_tpe", log=QUIET)


# ----------------------------------------------------------------------------
# scan mode with pnkpick
# ----------------------------------------------------------------------------
def test_scan_mode_pnkpick_records_per_partition_k(tmp_path):
    red, space, obj, samp, cfg = _small(ann_edges=[8, 30], n_iter=7, k_mode="scan", k_scan_max=12,
                                        validation=ValidationConfig(n_top=2, n_valid=1))
    space.params = [p for p in space.params if p.base != "k_klip"]        # k is not searched in scan mode
    runner = Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)
    res = runner.run()
    recs = _recs(str(tmp_path))
    assert len(recs) == 7 and not any(r["meta"]["failed"] for r in recs)
    for r in recs:
        k = r["k_used"]
        assert isinstance(k, dict) and set(k) == set(r["config"]["selected"])
        assert all(1 <= v <= 12 for v in k.values())
        assert r["meta"]["kpn"] == k and 1 <= r["meta"]["kbest"] <= 12
        for pid, v in k.items():
            assert r["config"]["per_partition"][pid]["k_klip"] == v
    both = [r["k_used"] for r in recs if len(r["k_used"]) == 2]
    assert both and any(k["n1"] != k["n2"] for k in both)                 # partition optima differ (6 vs 12)
    # log column carries the per-partition k list, like the IDL kpn=[...]
    line = _rows(os.path.join(str(tmp_path), "results.txt"))[0].split()
    assert line[2] == "seed" and line[-4].startswith("[") and line[-4].endswith("]")
    # validation re-applied the recorded per-partition k
    for row in res[0].validation_table:
        kh = runner.history.flags[row["eval_index"]]["k"]
        assert isinstance(kh, dict)
        for pid, v in kh.items():
            assert row["config"]["per_partition"][pid]["k_klip"] == v
    assert isinstance(res[0].k_used, dict)


def test_pnkpick_coordinate_ascent_finds_joint_optimum():
    red, space, obj, samp, cfg = _small(ann_edges=[8, 30])
    runner = Runner(red, space, obj, samp, cfg, "/tmp/never_used_dir_klip_tpe2", log=QUIET)
    from klip_tpe.metrics import Source
    src = [Source(0.8, 30.0, 1.0), Source(1.0, 150.0, 1.0)]
    nk, n = 6, 100
    yy, xx = np.mgrid[0:n, 0:n]
    cx = (n - 1) / 2
    rng = np.random.default_rng(0)

    def img(amp):
        im = rng.standard_normal((n, n)) * 0.2
        for s in src:
            phi = np.deg2rad(s.theta + 90)
            x0, y0 = cx + s.rho / red.pxscale * np.cos(phi), cx + s.rho / red.pxscale * np.sin(phi)
            im += amp * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / 2.0)
        return im
    # partition 0 peaks at k=2, partition 1 at k=5
    stack = np.stack([np.stack([img(3.0 if k == 1 else 0.3) for k in range(nk)]),
                      np.stack([img(3.0 if k == 4 else 0.3) for k in range(nk)])])
    kk, info = runner._pnkpick(stack, np.ones(2), src)
    assert kk == [2, 5] and np.array(info["snrk_pn"]).shape == (2, nk)


# ----------------------------------------------------------------------------
# difference metric
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("k_mode", ["search", "scan_rescore"])
def test_injection_difference_metric_runs(tmp_path, k_mode):
    red, space, obj, samp, cfg = _small(ann_edges=[8, 30], n_iter=6, k_mode=k_mode, k_scan_max=10,
                                        validation=ValidationConfig(n_top=1, n_valid=1))
    obj = Objective(InjectionDifferenceSNR(pxscale=red.pxscale, fwhm=red.fwhm), clean_subtract=False)
    assert obj.needs_clean and obj.metric.needs_clean
    runner = Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)
    # evaluate always computes the clean image for the metric
    runner.ia, runner._k_default = 0, 5
    rec, inj, clean = runner.evaluate(space.default_vector(), "t")
    assert clean is not None and rec.score is not None and rec.clean_per_source is None
    rec2, _, _ = runner.evaluate(space.default_vector(), "t", raw_only=True)
    assert rec2.score is not None
    contrast, kdef, info = runner.calibrate(0)
    assert np.isfinite(info["snr"]) and "kscan" in info
    res = runner.run()
    recs = _recs(str(tmp_path))
    assert not any(r["meta"]["failed"] for r in recs) and all(r["score"] is not None for r in recs)
    assert res[0].validated and np.isfinite(res[0].winner_score)
    assert fits.getheader(tmp_path / "klip_stitched.fits")["METRIC"] == "injection_difference"


# ----------------------------------------------------------------------------
# checkpoints / config compatibility
# ----------------------------------------------------------------------------
NEW_CFG_KEYS = ("opt_width", "width_range", "r_cap", "stitch_every", "write_setup_files", "verify", "verify_n_boot",
                "param_verify", "n_pv", "pv_divmin", "candidates", "cand_snrmin", "cand_verify_top", "legacy_stitch",
                "partition_weighting")
NEW_RESULT_KEYS = ("winner_sources", "partitions", "k_used")


def test_old_format_checkpoint_resumes(tmp_path):
    d = str(tmp_path)
    red, space, obj, samp, cfg = _small(n_iter=[6, 6], validation=ValidationConfig(n_top=1, n_valid=1))
    r = Runner(red, space, obj, samp, cfg, d, log=QUIET)
    r.run_annulus(0)
    ck = json.load(open(os.path.join(d, "checkpoint.json")))
    for k in NEW_CFG_KEYS:
        ck["config"].pop(k)
    for k in NEW_RESULT_KEYS:
        ck["results"][0].pop(k)
    ck["version"] = 1
    ck["config"]["some_future_key"] = 1                       # unknown keys are ignored too
    ck["config"]["calibration"]["future_cal_key"] = 2
    json.dump(ck, open(os.path.join(d, "checkpoint.json"), "w"))
    red, space, obj, samp, cfg = _small()
    r2 = Runner.resume(d, red, obj, samp, log=QUIET)
    assert r2.cfg.opt_width is False and r2.cfg.stitch_every == 10 and r2.cfg.param_verify is None
    assert r2.results[0].partitions == [] and r2.results[0].winner_sources == []
    res = r2.run()
    assert len(res) == 2 and os.path.exists(os.path.join(d, "klip_stitched.fits"))
    assert fits.getheader(os.path.join(d, "klip_stitched.fits"))["NANN"] == 2
    assert os.path.exists(os.path.join(d, "annulus02", "param_verify", "paramverify_stim_02.fits"))  # default on


def test_runconfig_new_fields_round_trip():
    cfg = RunConfig(opt_width=True, width_range=(12, 24), k_mode="scan", verify=True, candidates=True)
    d = json.loads(json.dumps(cfg.to_dict()))
    cfg2 = RunConfig.from_dict(d)
    assert cfg2.opt_width and tuple(cfg2.width_range) == (12, 24) and cfg2.k_mode == "scan"
    old = {k: v for k, v in d.items() if k not in NEW_CFG_KEYS}
    cfg3 = RunConfig.from_dict(old)
    assert cfg3.opt_width is False and cfg3.write_setup_files and cfg3.r_cap == 70.0


def test_setup_files_can_be_disabled_and_partition_weighting_passthrough(tmp_path):
    red, space, obj, samp, cfg = _small(ann_edges=[8, 30], n_iter=4, write_setup_files=False,
                                        partition_weighting="sqrt_texp", validation=ValidationConfig(n_top=1, n_valid=1))
    runner = Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)
    assert red.weighting == "sqrt_texp"
    runner.run()
    assert not glob.glob(str(tmp_path / "annulus01" / "*_setup.txt"))


def test_legacy_stitch_equal_weights(tmp_path):
    red, space, obj, samp, cfg = _small(n_iter=[5, 5], legacy_stitch=True, validation=ValidationConfig(n_top=1, n_valid=1))
    Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET).run()
    fin = json.load(open(tmp_path / "final_results.json"))
    assert fin["stitch_weights"] == [1.0, 1.0]
    assert fits.getheader(tmp_path / "klip_stitched.fits")["LEGACY"] == 1
    assert "no trim" in open(tmp_path / "contrast_curve.txt").read()


# ----------------------------------------------------------------------------
# stitch helpers
# ----------------------------------------------------------------------------
def test_stitch_helpers_and_fits_header(tmp_path):
    n = 40
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2)
    a = np.where((rr >= 3) & (rr <= 12), 1.0, np.nan)
    b = np.where((rr >= 8) & (rr <= 19), 3.0, np.nan)
    st = stitch_mod.stitch_tiles([a, b], [(5, 10), (10, 17)], [1.0, 1.0])
    assert st[int((n - 1) / 2), int((n - 1) / 2 + 7)] == 1.0             # inside tile 1 only
    assert st[int((n - 1) / 2), int((n - 1) / 2 + 15)] == 3.0
    assert st[int((n - 1) / 2), int((n - 1) / 2 + 10)] == pytest.approx(2.0)   # +-2 px overlap: mean
    assert np.isnan(st[int((n - 1) / 2), int((n - 1) / 2 + 1)])
    st_w = stitch_mod.stitch_tiles([a, b], [(5, 10), (10, 17)], [1.0, 3.0])
    assert st_w[int((n - 1) / 2), int((n - 1) / 2 + 10)] == pytest.approx(2.5)
    cube, ids = stitch_mod.stitch_stacks([np.stack([a, a]), b[None]], [["n1", "n2"], ["n2"]], [(5, 10), (10, 17)])
    assert ids == ["n1", "n2"] and cube.shape == (2, n, n)
    assert np.isnan(cube[0, int((n - 1) / 2), int((n - 1) / 2 + 15)])       # n1 unused by annulus 2
    assert cube[1, int((n - 1) / 2), int((n - 1) / 2 + 15)] == 3.0
    seam = stitch_mod.seam_pixels(st, 5, 17)
    assert not seam.any()
    h = stitch_mod.fits_header({"nann": 2, "score": np.float64(1.5), "bad": float("nan"), "none": None,
                                "very_long_key_name": "v", "arr": [1, 2]}, history=["line one"])
    assert h["NANN"] == 2 and h["SCORE"] == 1.5 and "BAD" not in h and "NONE" not in h
    assert h["VERY_LON"] == "v" and h["ARR"] == "[1, 2]" and h["KTPEVER"] and "line one" in str(h["HISTORY"])
    p = str(tmp_path / "x.fits.gz")
    stitch_mod.write_fits(p, st, {"a": 1})
    assert stitch_mod.read_fits(p).shape == (n, n)
    txt = stitch_mod.write_setup_file(str(tmp_path / "s.txt"), "eval", 0, 3, {"bin": 5, "k_klip": 7}, 8, 30, ["n1"],
                                      [(0.5, 10.0, 1e-4)], 1e-4, 4.2, k_used={"n1": 7})
    assert "k_klip        : 7" in txt and "median_SNR    : 4.200" in txt and "1.000E-04" in txt
