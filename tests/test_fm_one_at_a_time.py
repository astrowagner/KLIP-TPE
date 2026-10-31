"""KLIP-FM one source at a time, and the finished annulus's KLIP-FM panel on the winner's own
injections.

The cross-check models its test spiral in one pass and reads each source off the joint model.
The model is linear in the sources, so one pass per source gives the same model in sum, and
each source's own response.  The finished annulus's KLIP-FM square used to show that spiral;
it now models the injections of the picture in the best cells, as the injected-PSF square
beside it does.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

from klip_tpe import CalibrationConfig, ContrastGridConfig, PartitionedReducer, Runner, ValidationConfig
from klip_tpe.metrics import Source
from klip_tpe.products import fm_contrast_curve, fm_responses
from klip_tpe.runner import RunCallback
from klip_tpe.synthetic import make_synthetic_partitions

from conftest import PARTS, build_synthetic_run

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
fm1 = pytest.importorskip("fm_one_at_a_time")

QUIET = (lambda s: None)


def _reducer():
    return PartitionedReducer(make_synthetic_partitions(PARTS, k_opts=[6, 12], fm=True))


def _setup(**over):
    kw = dict(ann_edges=[8, 30], n_iter=6, n_init=3, validation=ValidationConfig(n_top=2, n_valid=2),
              calibration=CalibrationConfig(recal_budget=0, n_remeasure=1))
    kw.update(over)
    _, space, obj, samp, cfg = build_synthetic_run(**kw)
    return _reducer(), space, obj, samp, cfg


class _Done(RunCallback):
    def __init__(self):
        self.done = []

    def on_annulus_done(self, runner, result, winner):
        self.done.append((result, dict(winner)))


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """A finished synthetic run with the KLIP-FM cross-check and the injection grid, its last
    frame's images and the sources its injected-PSF square was drawn from."""
    import klip_tpe.display as D
    d = str(tmp_path_factory.mktemp("fm1"))
    red, space, obj, samp, cfg = _setup(contrast_grid=ContrastGridConfig(mode="grid", n_pa=4))
    cap = _Done()
    frames, models = [], []
    real_render, real_model = D.render_step, D.injected_model_image

    def spy_render(ad, i, images, *a, **k):
        frames.append(images)
        return real_render(ad, i, images, *a, **k)

    def spy_model(runner, sources, shape=None):
        img = real_model(runner, sources, shape)
        models.append(([tuple(s)[:2] for s in sources], img))
        return img
    D.render_step, D.injected_model_image = spy_render, spy_model
    try:
        disp = D.LiveDisplay(d, every=100, pdf_every=0, dpi=50, movie=False)
        runner = Runner(red, space, obj, samp, cfg, d, log=QUIET, callbacks=[cap, disp])
        results = runner.run()
    finally:
        D.render_step, D.injected_model_image = real_render, real_model
    return d, obj, samp, results, cap, frames, models


@pytest.mark.slow
def test_the_finished_fm_panel_models_the_winners_injections(finished):
    from klip_tpe.stitch import read_fits
    d, obj, samp, results, cap, frames, models = finished
    r = results[0]
    result, winner = cap.done[0]
    committed = [tuple(s)[:2] for s in r.winner_sources]
    assert winner.get("fm_inj") is not None, "no forward model of the winner's injections"
    assert np.allclose([(s.rho, s.theta) for s in winner["fm_inj"]["sources"]], committed)
    fmi = read_fits(os.path.join(d, "annulus01", "best_fm_inj.fits"))
    inj = read_fits(os.path.join(d, "annulus01", "best_inj.fits"))
    assert fmi is not None and fmi.shape == inj.shape and np.nanmax(fmi) > 0
    spiral = read_fits(os.path.join(d, "annulus01", "best_fm.fits"))
    assert spiral is not None and not np.allclose(np.nan_to_num(spiral), np.nan_to_num(fmi)), \
        "best_fm.fits is still the test spiral's model"
    last = [f for f in frames if getattr(f, "note", None) == "annulus complete"][-1]
    assert np.array_equal(np.nan_to_num(last.fm_image), np.nan_to_num(winner["fm_inj"]["fm_image"]))
    assert "winner's injections" in last.fm_label
    drawn = [src for src, img in models if img is not None and img is last.inj_model]
    assert drawn and np.allclose(drawn[-1], committed), "the injected-PSF square is not the best cells' injections"


@pytest.mark.slow
def test_one_at_a_time_sums_to_the_joint_model(finished, tmp_path):
    """Built the way scripts/fm_one_at_a_time.py builds it: the run resumed on a fresh reducer."""
    d, obj, samp, results, cap, frames, models = finished
    r = results[0]
    runner = Runner.resume(d, _reducer(), obj, samp, log=QUIET)
    runner._start_workers()
    runner.ia = 0
    cfg = runner._config_from_dict(r.winner_config)
    zone = runner._zone(0, cfg)
    clean = runner._annulus_images(0)["clean"]
    sp = runner._fm_pass(cfg, zone, r.contrast, "t_sp", clean_img=clean)
    one = runner._fm_pass(cfg, zone, r.contrast, "t_one", clean_img=clean, one_at_a_time=True, keep_singles=True)
    assert not sp["one_at_a_time"] and one["one_at_a_time"]
    assert len(one["fm_singles"]) == len(sp["sources"]) == len(one["sources"]) > 3
    a, b = np.nan_to_num(sp["fm_image"]), np.nan_to_num(one["fm_image"])
    assert np.allclose(a, b, rtol=1e-6, atol=1e-9 * np.abs(a).max()), "the forward model is not linear in the sources"
    kfn, conv = runner.reducer.matched_filter_kernel, runner.reducer.angle_convention
    for s, img, resp in zip(one["sources"], one["fm_singles"], one["curve"]["resp"]):
        mine = fm_responses(img, [s], runner.pxscale, runner.fwhm, kernel_fn=kfn, angle_convention=conv)[0]
        assert resp == pytest.approx(mine)


@pytest.mark.slow
def test_the_diagnostic_on_a_finished_run(finished, tmp_path):
    d, obj, samp, results, cap, frames, models = finished
    runner = Runner.resume(d, _reducer(), obj, samp, log=QUIET)
    out = str(tmp_path / "diag")
    rep = fm1.diagnose(runner, out, log=QUIET)
    assert os.path.getsize(os.path.join(out, "comparison.png")) > 10_000
    assert json.load(open(os.path.join(out, "report.json"))) == json.loads(json.dumps(rep))
    ann = rep["annuli"][0]
    assert {"spiral", "one_at_a_time", "throughput_one_over_spiral", "grid"} <= set(ann)
    assert ann["throughput_one_over_spiral"]["median"] is not None
    rows = ann["grid"]["by_separation"]
    assert rows and all(set(w) >= {"sep_as", "grid", "fm_spiral", "fm_one_at_a_time", "fm_at_grid_fm_noise",
                                   "fm_at_grid_grid_snr"} for w in rows)
    pos = ann["grid"]["positions"]
    assert pos and sum(p["grid_snr"] is not None for p in pos) >= len(pos) // 2


@pytest.mark.slow
def test_the_run_can_measure_its_test_sources_one_at_a_time(tmp_path):
    logs = []
    red, space, obj, samp, cfg = _setup(n_iter=4, n_init=3, fm_one_at_a_time=True,
                                        validation=ValidationConfig(n_top=1, n_valid=1))
    res = Runner(red, space, obj, samp, cfg, str(tmp_path), log=logs.append).run()
    assert any("KLIP-FM cross-check" in l and "one per pass" in l for l in logs), logs
    assert res[0].fm_curve is not None and sum(v is not None for v in res[0].fm_curve["curve"]) >= 3


def test_responses_can_be_given():
    """fm_contrast_curve with the responses measured elsewhere uses them as they are."""
    rng = np.random.default_rng(1)
    clean = rng.normal(0, 1, (80, 80))
    srcs = [Source(0.5 + 0.1 * i, 40.0 * i, 1e-3) for i in range(6)]
    resp = np.linspace(1.0, 2.0, 6) * 1e-3
    cc = fm_contrast_curve(np.zeros_like(clean), clean, 1e-3, 4.0, 0.05, 8.0, 38.0, srcs, resp=resp)
    assert cc["K_fm"] == pytest.approx(resp / 1e-3)
    assert np.isfinite(cc["curve"]).sum() >= 3
    with pytest.raises(ValueError):
        fm_contrast_curve(np.zeros_like(clean), clean, 1e-3, 4.0, 0.05, 8.0, 38.0, srcs, resp=resp[:3])


def test_solve_contrast_finds_the_target_of_an_affine_snr():
    c = fm1.solve_contrast(lambda c: 3e5 * c - 0.4, 1e-5, 5.0)
    assert c == pytest.approx(5.4 / 3e5, rel=1e-6)
    assert np.isnan(fm1.solve_contrast(lambda c: 2.0, 1e-5, 5.0)), "an S/N that does not respond has no answer"
