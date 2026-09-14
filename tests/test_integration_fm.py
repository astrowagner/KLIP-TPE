"""Integration pass: KLIP-FM cross-check curve / preview in the Runner (A 9.1 / 9.3),
the display state additions (calibration images, per-source labels, FM panel), the
difference-metric objective with clean_subtract, and near.make_space options."""
import json
import os

import numpy as np
import pytest

from klip_tpe import (CalibrationConfig, InjectionDifferenceSNR, MawetPeakSNR, Objective, PartitionedReducer,
                      PositionSampler, RunConfig, Runner, ValidationConfig)
from klip_tpe.display import LiveDisplay
from klip_tpe.runner import AnnulusResult, RunCallback
from klip_tpe.synthetic import SyntheticReducer, make_synthetic_partitions

from conftest import PARTS, build_synthetic_run

QUIET = (lambda s: None)


def _fm_setup(**overrides):
    kw = dict(ann_edges=[8, 30], n_iter=9, n_init=4, validation=ValidationConfig(n_top=2, n_valid=2),
              calibration=CalibrationConfig(recal_budget=0, n_remeasure=1))
    kw.update(overrides)
    red, space, obj, samp, cfg = build_synthetic_run(**kw)
    red = PartitionedReducer(make_synthetic_partitions(PARTS, k_opts=[6, 12], fm=True))
    assert red.supports_fm
    return red, space, obj, samp, cfg


class Capture(RunCallback):
    def __init__(self):
        self.calib = []
        self.evals = []
        self.validation = []
        self.done = []

    def on_calibration(self, runner, info, images):
        self.calib.append((dict(info), images))

    def on_eval(self, runner, record, inj, clean, is_best):
        self.evals.append((record, inj, clean, is_best, dict(runner.best_images)))

    def on_validation(self, runner, table, winner):
        self.validation.append((list(table), dict(winner)))

    def on_annulus_done(self, runner, result, winner):
        self.done.append((result, dict(winner)))


@pytest.fixture(scope="module")
def fm_run(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("fm"))
    red, space, obj, samp, cfg = _fm_setup(fm_preview=True)
    cap = Capture()
    logs = []
    disp = LiveDisplay(d, every=1, pdf_every=0, dpi=60, movie=False)
    runner = Runner(red, space, obj, samp, cfg, d, log=logs.append, callbacks=[cap, disp])
    results = runner.run()
    return d, runner, results, logs, cap, disp


# ----------------------------------------------------------------------------
# A 9.1: KLIP-FM cross-check curve after validation
# ----------------------------------------------------------------------------
def test_fm_curve_products(fm_run):
    d, runner, results, logs, cap, disp = fm_run
    assert runner.cfg.fm_curve is True and runner.do_fm
    assert not [l for l in logs if "KLIP-FM" in l and ("failed" in l or "skipped" in l)], logs
    r = results[0]
    assert r.fm_curve is not None
    c = np.array([np.nan if v is None else v for v in r.fm_curve["curve"]], float)
    rr = np.array([np.nan if v is None else v for v in r.fm_curve["r_as"]], float)
    assert np.isfinite(c).sum() >= 3 and (c[np.isfinite(c)] > 0).all()
    assert rr.size == c.size and np.all(np.diff(rr) > 0)
    # the test spiral spans [max(inrad, fwhm), outrad - 1] px
    assert rr.min() >= max(r.inrad, runner.fwhm) * runner.pxscale - 1e-9
    assert rr.max() <= (r.outrad - 1) * runner.pxscale + 1e-9
    assert set(r.fm_curve) >= {"r_as", "curve", "K_fm", "resp", "r_test_as"}
    # JSON-safe and round-trips through winner.json / AnnulusResult.from_dict
    w = json.load(open(os.path.join(d, "annulus01", "winner.json")))
    assert w["fm_curve"]["curve"] == r.fm_curve["curve"]
    assert AnnulusResult.from_dict(w).fm_curve == r.fm_curve
    # best_fm.fits
    from klip_tpe.stitch import read_fits
    fm = read_fits(os.path.join(d, "annulus01", "best_fm.fits"))
    assert fm is not None and fm.shape == (100, 100) and np.nanmax(fm) > 0
    # per-annulus and run-level contrast_curve.txt carry the KLIP-FM section
    for path in (os.path.join(d, "annulus01", "contrast_curve.txt"), os.path.join(d, "contrast_curve.txt")):
        lines = open(path).read().splitlines()
        idx = [i for i, l in enumerate(lines) if l.startswith("# KLIP-FM")]
        assert len(idx) == 1, path
        i = idx[0]
        assert lines[i + 1].startswith("#") and "sep_arcsec" in lines[i + 1] and "contrast_5sig" in lines[i + 1]
        rows = [l.split() for l in lines[i + 2:] if l.strip() and not l.startswith("#")]
        assert len(rows) == int(np.isfinite(c).sum()) and all(len(x) == 2 for x in rows)
        assert np.allclose([float(x[0]) for x in rows], rr[np.isfinite(c)], atol=1e-3)
    # the winner passed to on_annulus_done carries the FM pass
    res, winner = cap.done[0]
    assert winner["fm"]["fm_image"] is not None and len(winner["fm"]["sources"]) == len(r.fm_curve["r_test_as"])


def test_fm_preview_and_display(fm_run):
    d, runner, results, logs, cap, disp = fm_run
    assert not [l for l in logs if "preview failed" in l]
    assert not [l for l in logs if "[display]" in l and "failed" in l], [l for l in logs if "failed" in l]
    # every new best carries an FM preview: fm_image attached to the eval's images and
    # best_images['fm'] holds the image + curve for the display
    bests = [(rec, inj, b) for rec, inj, clean, is_best, b in cap.evals if is_best]
    assert bests
    for rec, inj, b in bests:
        assert inj.fm_image is not None and inj.fm_image.shape == inj.image.shape
        assert b["fm"]["fm_image"] is inj.fm_image
        assert len(b["fm"]["sources"]) == len(rec.sources)
        assert "curve" in b["fm"] and "r_as" in b["fm"]["curve"]
    assert runner.best_images is runner._best_images
    # non-best evals do not run the preview
    assert all(inj.fm_image is None for rec, inj, clean, is_best, b in cap.evals if not is_best)
    # the display rendered every step with the FM thumbnail and the annulus-done frame
    steps = sorted(f for f in os.listdir(os.path.join(d, "steps")) if f.startswith("step"))
    vc = runner.cfg.validation
    assert len(steps) == runner.cfg.n_iter + vc.n_top * vc.n_valid + 1   # evals + validation trials + annulus end
    assert os.path.getsize(os.path.join(d, "steps", steps[-1])) > 30_000


# ----------------------------------------------------------------------------
# display state: calibration images, per_source / k_default, angle_convention
# ----------------------------------------------------------------------------
def test_calibration_images_and_panel(fm_run):
    d, runner, results, logs, cap, disp = fm_run
    assert len(cap.calib) == 1
    info, images = cap.calib[0]
    assert isinstance(images, dict)
    assert images["inj"].image.ndim == 2 and images["clean"].image.ndim == 2
    assert len(images["sources"]) == runner._nsrc(0) == len(images["per_source"])
    assert images["clean_per_source"] is not None and len(images["clean_per_source"]) == len(images["sources"])
    assert info["k_default"] == results[0].k_default
    # the panel was drawn from the calibration images (before any seed evaluation)
    p = os.path.join(d, "annulus01", "calibration_panel.png")
    assert os.path.getsize(p) > 20_000
    assert not disp._calib_pending


def test_calibration_panel_uses_images(tmp_path, monkeypatch):
    """Monkeypatch render_calibration to check it receives the real calibration images."""
    import klip_tpe.display as disp_mod
    calls = []

    def fake_render(ad, info, out, clean=None, inj=None, sources=None, per_source=None, clean_per_source=None, **kw):
        calls.append(dict(clean=clean, inj=inj, sources=sources, per_source=per_source, cps=clean_per_source))
        open(out, "w").write("x")
        return out
    monkeypatch.setattr(disp_mod, "render_calibration", fake_render)
    red, space, obj, samp, cfg = _fm_setup(n_iter=5, n_init=3, fm_curve=False, save_fits=False,
                                           validation=ValidationConfig(n_top=1, n_valid=1))
    disp = LiveDisplay(str(tmp_path), every=100, pdf_every=0, save_png=False, movie=False)
    Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET, callbacks=[disp]).run()
    assert len(calls) == 1
    c = calls[0]
    assert c["inj"] is not None and c["inj"].ndim == 2 and c["clean"] is not None
    assert len(c["sources"]) == len(c["per_source"]) == len(c["cps"]) >= 2


def test_validation_winner_fields(fm_run):
    d, runner, results, logs, cap, disp = fm_run
    table, winner = cap.validation[0]
    r = results[0]
    assert r.validated
    assert winner["k_default"] == r.k_default == runner._k_default
    assert len(winner["per_source"]) == len(winner["sources"]) == len(r.winner_sources) > 0
    assert r.per_source == winner["per_source"]
    # committed trial: the per-source median equals the committed trial's raw score
    row = [t for t in table if t["eval_index"] == winner["eval_index"]][0]
    commit = row["committed_trial"]
    assert np.isclose(np.median(winner["per_source"]), row["trials"][commit])
    w = json.load(open(os.path.join(d, "annulus01", "winner.json")))
    assert w["per_source"] == r.per_source and w["k_default"] == r.k_default
    # EvalRecord.meta carries the reducer's angle convention
    recs = [json.loads(l) for l in open(os.path.join(d, "results.jsonl"))]
    assert all(rec["meta"]["angle_convention"] == runner.reducer.angle_convention == "pa" for rec in recs)
    from klip_tpe.display import annulus_from_run
    assert annulus_from_run(d, 0).angle_convention == "pa"


def test_fm_curve_off_and_scan_mode(tmp_path):
    red, space, obj, samp, cfg = _fm_setup(n_iter=5, n_init=3, fm_curve=False,
                                           validation=ValidationConfig(n_top=1, n_valid=1))
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "off"), log=QUIET)
    res = r.run()
    assert res[0].fm_curve is None and not os.path.exists(tmp_path / "off" / "annulus01" / "best_fm.fits")
    assert "# KLIP-FM" not in open(tmp_path / "off" / "contrast_curve.txt").read()
    # scan k-mode: the FM cross-check runs at the k the winner actually used (a plain
    # reduction at that k, never a k-scan cube), so the curve is still produced
    red, space, obj, samp, cfg = _fm_setup(n_iter=4, n_init=3, k_mode="scan_rescore", k_scan_max=8,
                                           validation=ValidationConfig(n_top=1, n_valid=1))
    space.params = [p for p in space.params if p.base != "k_klip"]
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "scan"), log=QUIET)
    assert r.do_fm
    res = r.run()
    assert res[0].fm_curve is not None and res[0].k_used is not None
    assert os.path.exists(tmp_path / "scan" / "annulus01" / "best_fm.fits")


# ----------------------------------------------------------------------------
# plain (non-partitioned) reducer path of Runner._reduce forwards fm_sources
# ----------------------------------------------------------------------------
def test_plain_reducer_fm_forwarding(tmp_path):
    from klip_tpe import Param, SearchSpace
    from klip_tpe.space import kgrid
    red = SyntheticReducer(k_opt=8, seed=3, fm=True)
    space = SearchSpace([Param("bin", 5, 30, "int", default=12), Param("k_klip", 1, 30, "int", grid=kgrid(30), default=5)])
    obj = Objective(MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm), clean_subtract=True)
    samp = PositionSampler(fwhm_as=red.fwhm * red.pxscale)
    cfg = RunConfig(ann_edges=[8, 30], n_iter=5, n_init=3, seed=2, validation=ValidationConfig(n_top=1, n_valid=1),
                    calibration=CalibrationConfig(recal_budget=0, n_remeasure=1), param_verify=False)
    r = Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)
    res = r.run()
    assert res[0].fm_curve is not None and sum(v is not None for v in res[0].fm_curve["curve"]) >= 3
    assert os.path.exists(tmp_path / "annulus01" / "best_fm.fits")


# ----------------------------------------------------------------------------
# metrics: difference metric + clean_subtract is a documented no-op
# ----------------------------------------------------------------------------
def test_difference_metric_clean_subtract_noop(rng):
    fwhm, px = 4.0, 0.05
    from klip_tpe.metrics import Source
    img_c = rng.standard_normal((80, 80))
    img_i = img_c.copy()
    src = [Source(0.8, 30.0, 1.0), Source(1.0, 200.0, 1.0)]
    from klip_tpe.metrics import source_xy, star_center
    cx, cy = star_center(img_i.shape)
    xs, ys = source_xy([s.rho for s in src], [s.theta for s in src], px, cx, cy)
    yy, xx = np.mgrid[0:80, 0:80]
    for x, y in zip(xs, ys):
        img_i += 8.0 * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / (fwhm / 2.3548) ** 2)
    m = InjectionDifferenceSNR(pxscale=px, fwhm=fwhm)
    o_on = Objective(m, clean_subtract=True)
    o_off = Objective(m, clean_subtract=False)
    assert o_on.needs_clean and o_on.clean_subtract and not o_on.effective_clean_subtract
    assert o_on.describe()["effective_clean_subtract"] is False
    a = o_on.score_search(img_i, src, img_c)
    b = o_off.score_search(img_i, src, img_c)
    raw = o_on.score_raw(img_i, src, img_c)
    assert a.clean_per_source is None
    assert np.allclose(a.per_source, b.per_source) and np.allclose(a.per_source, raw.per_source)
    assert np.isfinite(a.score) and a.score == pytest.approx(raw.score)
    with pytest.raises(ValueError):
        o_on.score_search(img_i, src, None)      # the metric itself still needs the clean image


def test_difference_metric_clean_subtract_run(tmp_path):
    red, space, _, samp, cfg = _fm_setup(n_iter=5, n_init=3, fm_curve=False, save_fits=False,
                                         validation=ValidationConfig(n_top=1, n_valid=1))
    obj = Objective(InjectionDifferenceSNR(pxscale=red.pxscale, fwhm=red.fwhm), clean_subtract=True)
    logs = []
    r = Runner(red, space, obj, samp, cfg, str(tmp_path), log=logs.append)
    res = r.run()
    assert not [l for l in logs if "evaluation failed" in l]
    assert res[0].winner_score > 0 and np.isfinite(res[0].search_best_score)
    recs = [json.loads(l) for l in open(tmp_path / "results.jsonl")]
    assert all(rec["clean_per_source"] is None for rec in recs)
    assert all(rec["score"] == rec["raw_score"] for rec in recs if rec["score"] is not None)


# ----------------------------------------------------------------------------
# near.make_space: width_range / search_k
# ----------------------------------------------------------------------------
class _FakeNight:
    def __init__(self, n):
        class D:  # noqa
            nframes = n
        self.data = D()


class _FakeNear:
    def __init__(self, nights=("n1", "n2", "n3"), nframes=400):
        self.reducers = {p: _FakeNight(nframes) for p in nights}

    def partitions(self):
        return list(self.reducers)

    def frame_angles(self, pid):
        return np.linspace(-40, 40, 50)


def test_make_space_width_and_search_k():
    from klip_tpe.instruments.near import make_space
    red = _FakeNear()
    sp = make_space(red)
    assert "k_klip" in sp.bases and "width" not in sp.bases
    assert [p.name for p in sp.params][-2:] == ["drop1", "drop2"]
    sp2 = make_space(red, width_range=(15, 30), search_k=False)
    assert "k_klip" not in sp2.bases and "width" in sp2.bases
    w = sp2.params[[p.name for p in sp2.params].index("width")]
    assert (w.lo, w.hi, w.kind, w.partition) == (15.0, 30.0, "int", None) and w.default == 22
    assert sp2.ndim == sp.ndim - len(red.partitions()) + 1
    # a single global block also gets the width dim, and no k
    sp3 = make_space(red, per_night=False, width_range=(10.0, 20.0), search_k=False)
    assert "width" in sp3.names and "k_klip" not in sp3.names and sp3.partitions == red.partitions()


def test_build_near_space_uses_make_space_options(monkeypatch):
    import argparse
    from klip_tpe import cli
    from klip_tpe.instruments import near
    seen = {}

    def fake_make_space(red, **kw):
        seen.update(kw)
        return "space"
    monkeypatch.setattr(near, "make_space", fake_make_space)
    a = argparse.Namespace(global_block=False, k_max=None, no_framesel=False, fast=False, no_selection=False,
                           opt_width=True, width_range=[12.0, 24.0], k_mode="scan")
    assert cli.build_near_space(object(), a) == "space"
    assert seen["width_range"] == (12.0, 24.0) and seen["search_k"] is False
    a.opt_width, a.k_mode = False, "search"
    cli.build_near_space(object(), a)
    assert seen["width_range"] is None and seen["search_k"] is True
    # the CLI exposes the new RunConfig fields
    ns = cli.main.__globals__  # noqa: F841  (import check only)
    p = argparse.ArgumentParser()
    cli._protocol_args(p)
    args = p.parse_args(["--no-fm-curve", "--no-fm-preview"])
    assert args.fm_curve is False and args.fm_preview is False
    assert p.parse_args([]).fm_curve is True and p.parse_args([]).fm_preview is True
