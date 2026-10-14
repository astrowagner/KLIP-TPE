"""The metric's inner guard, asked of where a source was put, and failed draws that say why.

From the NEAR2 note of 2026-10-13 (``run_20261013_170907``): with ``--min-r-fwhm 1.0`` the one
source of annulus 1 sat at 1.10 FWHM, and half its draws failed.  The metric located the
source as the filtered image's peak within 1.5 px, then gave no S/N to a peak inside 1 FWHM.
On the injected image that peak is the planet.  On the clean image it is the highest noise
pixel nearby, which lay inside 1 FWHM about half the time, so the clean term was NaN and the
draw failed, with ``score=failed`` and nothing else in the log.  What must hold:

* a source at 1.0 FWHM on a noise image scores, with a clean image and without one;
* the guard asks where the source was put: a peak inside 1 FWHM does not fail a source put
  at 1.1 FWHM, and a peak outside it does not pass one put at 0.95 FWHM;
* the guard follows the injection floor down, never up, the run records it, a resume puts
  it back, and param_verify scores with it;
* a draw without a score says why, on its log line and in results.jsonl, and the log counts
  an annulus' failed draws;
* the starved-ring line counts the apertures a source keeps where it was put.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import pytest

from klip_tpe import (CalibrationConfig, GaussianPSF, KLIPReducer, MawetPeakSNR, Objective, Param,
                      PartitionedReducer, RunConfig, Runner, SearchSpace, ValidationConfig)
from klip_tpe.fmmf import FMMFSNR
from klip_tpe.metrics import (WHY_GUARD, WHY_RING, Source, mawet_peak_snr, source_xy, star_center,
                              why_failed)
from klip_tpe.param_verify import param_verify_cubes
from klip_tpe.positions import PositionSampler
from klip_tpe.runner import EvalRecord, _error_why
from klip_tpe.synthetic import synthetic_klip_dataset

from conftest import build_synthetic_run


def _blob(n, x, y, fwhm, amp):
    yy, xx = np.mgrid[0:n, 0:n]
    sg = fwhm / 2.3548
    return amp * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / sg ** 2)


# ----------------------------------------------------------------------------- the metric
@pytest.mark.parametrize("px,fwhm", [(0.0456, 6.2378), (0.05, 4.0)])      # NEAR, and these tests'
def test_a_source_at_one_fwhm_scores_with_and_without_a_clean_image(px, fwhm):
    n = 121
    cx, cy = star_center((n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    bright = 1 + 4 * np.exp(-np.hypot(xx - cx, yy - cy) / (2 * fwhm))       # residuals rise inward
    m = MawetPeakSNR(pxscale=px, fwhm=fwhm)
    sub, plain = Objective(m, clean_subtract=True), Objective(m, clean_subtract=False)
    rng = np.random.default_rng(5)
    rho = 1.0 * fwhm * px                    # exactly on the guard, as the sampler puts it
    peak_inside = 0
    for k in range(60):
        pa = 6.0 * k
        clean = rng.standard_normal((n, n)) * bright
        x, y = source_xy(rho, pa, px, cx, cy)
        inj = clean + _blob(n, x[0], y[0], fwhm, 12.0)
        r = sub.score_search(inj, [Source(rho, pa)], clean)
        assert np.isfinite(r.score) and np.isfinite(r.clean_per_source[0]), (pa, r.why)
        assert r.why is None
        assert np.isfinite(plain.score_search(inj, [Source(rho, pa)]).score), pa
        _, det = mawet_peak_snr(clean, rho, pa, px, fwhm, return_details=True)
        peak_inside += det[0]["r"] < fwhm
    # the case the note describes: the clean image's peak inside 1 FWHM, scored all the same
    assert peak_inside >= 15, peak_inside


def test_the_guard_asks_where_the_source_was_put():
    px, fwhm, n = 0.05, 4.0, 101
    cx, cy = star_center((n, n))
    rng = np.random.default_rng(2)
    base = 0.01 * rng.standard_normal((n, n))
    # put at 1.1 FWHM (4.4 px north of the star), the filtered image's peak at 3 px: inside 1 FWHM
    img = base + _blob(n, cx, cy + 3, fwhm, 5.0)
    s, det = mawet_peak_snr(img, 1.1 * fwhm * px, 0.0, px, fwhm, return_details=True)
    assert det[0]["r"] == pytest.approx(3.0) and det[0]["r"] < fwhm
    assert np.isfinite(s[0]) and "why" not in det[0]
    # put at 0.95 FWHM (3.8 px), the peak at 5 px: outside 1 FWHM, no S/N all the same
    img = base + _blob(n, cx, cy + 5, fwhm, 5.0)
    s, det = mawet_peak_snr(img, 0.95 * fwhm * px, 0.0, px, fwhm, return_details=True)
    assert det[0]["r"] == pytest.approx(5.0)
    assert np.isnan(s[0]) and det[0]["why"] == WHY_GUARD
    # a lower guard lets it through, the metric's own included, and inside 5/(2 pi) FWHM the
    # ring is too short anyway
    assert np.isfinite(mawet_peak_snr(img, 0.95 * fwhm * px, 0.0, px, fwhm, inner_fwhm=0.9)[0])
    put = ([0.95 * fwhm * px], [0.0])
    for cls in (MawetPeakSNR, FMMFSNR):
        assert np.isnan(cls(pxscale=px, fwhm=fwhm).per_source(img, None, *put)[0]), cls
        why = []
        assert np.isfinite(cls(pxscale=px, fwhm=fwhm, inner_fwhm=0.9).per_source(img, None, *put, why=why)[0])
        assert why == [None]
        cls(pxscale=px, fwhm=fwhm).per_source(img, None, *put, why=why)
        assert why == [WHY_GUARD], cls
    s, det = mawet_peak_snr(img, 0.75 * fwhm * px, 0.0, px, fwhm, inner_fwhm=0.5, return_details=True)
    assert np.isnan(s[0]) and det[0]["why"] == WHY_RING


def test_the_objective_says_why_a_score_is_nan():
    px, fwhm, n = 0.05, 4.0, 101
    noise = np.random.default_rng(3).standard_normal((n, n))
    obj = Objective(MawetPeakSNR(pxscale=px, fwhm=fwhm), clean_subtract=True)
    inside = [Source(0.8 * fwhm * px, 10.0)]
    r = obj.score_search(noise, inside, noise)
    assert np.isnan(r.score) and r.why == f"injected S/N NaN ({WHY_GUARD})"
    two = [Source(0.8 * fwhm * px, 10.0), Source(0.8 * fwhm * px, 190.0)]
    assert obj.score_raw(noise, two).why == f"injected S/N NaN ({WHY_GUARD}) at 2 of 2 sources"
    assert obj.score_search(noise, [Source(1.0, 10.0)], noise).why is None
    # a metric that cannot say why still has the image named
    assert why_failed([3.0], None, [np.nan], None) == "clean S/N NaN"
    assert why_failed([np.nan, 2.0], [WHY_RING, None], [1.0, np.nan], [None, "x"]) == \
        f"injected S/N NaN ({WHY_RING}) at 1 of 2 sources, clean S/N NaN (x) at 1 of 2 sources"


# ------------------------------------------------------------------- the run and the floor
def test_the_guard_follows_the_floor_down_and_the_run_records_it(tmp_path):
    for floor, want, said in ((0.9, 0.9, True), (1.0, 1.0, False), (1.5, 1.0, False), (None, 1.0, False)):
        red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], min_r_fwhm=floor)
        logs = []
        Runner(red, space, obj, samp, cfg, str(tmp_path / f"r{floor}"), log=logs.append)
        assert obj.metric.inner_fwhm == pytest.approx(want), floor
        assert bool([s for s in logs if "inner guard" in s]) == said, (floor, logs)
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], min_r_fwhm=0.5)
    logs = []
    Runner(red, space, obj, samp, cfg, str(tmp_path / "half"), log=logs.append)
    assert obj.metric.inner_fwhm == pytest.approx(0.5)
    assert [s for s in logs if "fewer than 5 apertures" in s], logs

    red, space, obj, samp, cfg = build_synthetic_run(
        ann_edges=[8, 30], n_iter=3, n_init=2, min_r_fwhm=0.9, param_verify=False,
        calibration=CalibrationConfig(recal_budget=0), validation=ValidationConfig(n_top=1, n_valid=1))
    d = tmp_path / "run"
    Runner(red, space, obj, samp, cfg, str(d), log=lambda s: None).run()
    setup = json.load(open(d / "run_setup.json"))
    assert setup["objective"]["metric"]["inner_fwhm"] == pytest.approx(0.9)
    fresh = Objective(MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm), clean_subtract=True)
    Runner.resume(str(d), red, fresh, PositionSampler(fwhm_as=samp.fwhm_as), log=lambda s: None)
    assert fresh.metric.inner_fwhm == pytest.approx(0.9), "a resume scores with the guard the run had"
    # and so does a script's own Runner over the run's directory, which adopts the checkpoint
    mine = Objective(MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm), clean_subtract=True)
    scripted = Runner(red, space, mine, PositionSampler(fwhm_as=samp.fwhm_as), RunConfig(ann_edges=[8, 30]),
                      str(d), log=lambda s: None)
    assert mine.metric.inner_fwhm == 1.0
    scripted._auto_resume()
    assert mine.metric.inner_fwhm == pytest.approx(0.9)


def test_param_verify_scores_with_the_runs_guard(tmp_path, monkeypatch):
    px, fwhm, n = 0.05, 4.0, 61
    cx, cy = star_center((n, n))
    rng = np.random.default_rng(4)
    src = [Source(0.95 * fwhm * px, 30.0)]
    x, y = source_xy(src[0].rho, src[0].theta, px, cx, cy)
    cclean = rng.standard_normal((3, n, n))
    cinj = cclean + _blob(n, x[0], y[0], fwhm, 8.0)[None]
    k = {g: param_verify_cubes(cclean, cinj, [1, 1, 1], src, 1e-3, px, fwhm, 2.0, 12.0, inner_fwhm=g)["Kcal"][0]
         for g in (1.0, 0.9)}
    assert np.isnan(k[1.0]) and np.isfinite(k[0.9])
    # and the Runner hands param_verify its own guard
    import klip_tpe.param_verify as pv
    seen = {}
    monkeypatch.setattr(pv, "param_verify", lambda *a, **kw: seen.update(kw))
    red, space, obj, samp, cfg = build_synthetic_run(
        ann_edges=[8, 30], n_iter=4, n_init=2, min_r_fwhm=0.9, param_verify=True,
        calibration=CalibrationConfig(recal_budget=0), validation=ValidationConfig(n_top=1, n_valid=1))
    Runner(red, space, obj, samp, cfg, str(tmp_path / "pv"), log=lambda s: None).run()
    assert seen.get("inner_fwhm") == pytest.approx(0.9)


# --------------------------------------------------------------------------- failed draws
@dataclass
class _CleanFailsNorth(MawetPeakSNR):
    """The clean term has no S/N for a source north of the star (PA 0-180), with a reason:
    a failure that depends on position angle, like the one in the note."""

    def per_source(self, img_inj, img_clean, rho, theta, why=None):
        v = super().per_source(img_inj, img_clean, rho, theta, why=why)
        if img_clean is None and np.all(np.asarray(theta, float) % 360.0 < 180.0):
            v = np.full(np.shape(v), np.nan)
            if why is not None:
                why[:] = ["test reason"] * len(v)
        return v


def test_a_draw_without_a_score_says_why(tmp_path):
    red, space, _, samp, cfg = build_synthetic_run(
        ann_edges=[8, 30], n_iter=8, n_init=4, n_sources=1, n_remeasure=3, param_verify=False,
        calibration=CalibrationConfig(recal_budget=0, n_remeasure=4),
        validation=ValidationConfig(n_top=1, n_valid=2))
    obj = Objective(_CleanFailsNorth(pxscale=red.pxscale, fwhm=red.fwhm), clean_subtract=True)
    logs = []
    Runner(red, space, obj, samp, cfg, str(tmp_path), log=logs.append).run()
    why = "clean S/N NaN (test reason)"
    assert [s for s in logs if re.search(r"draw \d/3 score=failed \(" + re.escape(why) + r"\)", s)], logs
    assert [s for s in logs if re.search(r"\(mean of 3(, sd \S+)?, \d failed\)", s)], logs
    recs = [json.loads(s) for s in open(tmp_path / "results.jsonl")]
    nfail = 0
    for r in recs:
        m = r["meta"]
        if not m.get("draw_failed"):
            assert "draw_why" not in m and "why" not in m
            continue
        nfail += m["draw_failed"]
        assert [w is not None for w in m["draw_why"]] == [s is None for s in m["draw_scores"]]
        assert {w for w in m["draw_why"] if w} == {why}
        assert (r["score"] is None) == (m["draw_failed"] == m["draw_n"]) == (m.get("why") == why)
    assert nfail > 0
    warned = [s for s in logs if "draws failed" in s and "**" in s]
    assert len(warned) == 1 and "clean S/N NaN (test reason))" in warned[0], warned
    note = [s for s in logs if s.strip().startswith("note:") and "draws failed" in s]
    ndraws = 3 * len(recs)
    assert note == [f"  note: {nfail} of {ndraws} draws failed ({nfail} {why})"], (note, logs)
    cal = [s for s in logs if s.strip().startswith("calibration trial") and "draws failed" in s]
    assert cal and all(why in s for s in cal), logs
    cj = json.load(open(tmp_path / "annulus01" / "calibration.json"))
    assert any(why in (t.get("why") or []) for t in cj["trials"])


def test_a_trial_carries_each_draws_reason(tmp_path, monkeypatch):
    def rec(score, why=None):
        meta = {"failed": bool(why and why.startswith("error")), **({"why": why} if why else {})}
        return EvalRecord(annulus=0, index=-1, phase="tpe", x=[], config={}, sources=[], score=score,
                          raw_score=score, per_source=[], raw_per_source=[], clean_per_source=None,
                          partition_snr={}, k_used=None, contrast=1e-3, wall_s=1.0, meta=meta)
    boom = _error_why(RuntimeError("boom"))
    t = Runner._combine_draws([rec(2.0), rec(None, boom), rec(4.0)], rec(4.0))
    assert t.score == 3.0 and t.meta["draw_failed"] == 1 and "why" not in t.meta
    assert t.meta["draw_why"] == [None, "error: RuntimeError('boom')", None]
    t = Runner._combine_draws([rec(None, "a"), rec(None, "b"), rec(None, "a")], rec(None, "a"))
    assert t.score is None and t.meta["why"] == "a"
    t = Runner._combine_draws([rec(1.0), rec(None, "b")], rec(None, "b"))
    assert "why" not in t.meta, "the last draw's reason is not the trial's"
    assert len(_error_why(RuntimeError("x" * 1000))) == 300
    # a reduction that raises: the record says so
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30])
    r = Runner(red, space, obj, samp, cfg, str(tmp_path), log=lambda s: None)

    def fail(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(r, "_reduce", fail)
    got, _, _ = r.evaluate(r.space.sanitize(np.full(r.space.ndim, 0.5)), "tpe")
    assert got.score is None and got.meta["why"] == "error: RuntimeError('boom')"


# ------------------------------------------------------------------------ the starved ring
def _klip_runner(tmp_path, edges, rmin):
    px = 0.05
    ds = synthetic_klip_dataset(nframes=20, size=64, pa_span=60, seed=1)
    kr = KLIPReducer(ds, px, 3.89 * px / 206265 * 8.2, 8.2, outrad_cap=28,
                     injection_model=GaussianPSF(4.0, star_flux=1e4))
    red = PartitionedReducer({"n1": kr})
    space = SearchSpace()
    space.add(Param("k_klip", 1, 10, "int", default=5))
    obj = Objective(MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm))
    logs = []
    r = Runner(red, space, obj, PositionSampler(fwhm_as=red.fwhm * red.pxscale),
               RunConfig(ann_edges=edges, n_sources=1, min_r_fwhm=rmin, seed=1), str(tmp_path), log=logs.append)
    return r, logs


def test_the_starved_ring_line_counts_where_the_source_was_put(tmp_path):
    r, logs = _klip_runner(tmp_path, [2, 9], 0.3)
    r._nsrc(0)
    said = [s for s in logs if "Mawet ring" in s]
    assert len(said) == 1, logs
    m = re.search(r'one source at ([0-9.]+)" keeps (\d+)(?: to (\d+))? clean', said[0])
    assert m, said
    rho, few = float(m.group(1)), int(m.group(2))
    most = int(m.group(3) or few)
    # what a source there keeps, found where it was put, over the azimuths
    shape = r._probe_shape()
    cx, cy = star_center(shape)
    kept = []
    for pa in np.arange(0.0, 360.0, 5.0):
        img = np.zeros(shape)
        x, y = source_xy(rho, pa, r.pxscale, cx, cy)
        img[int(round(y[0])), int(round(x[0]))] = 1.0
        _, det = mawet_peak_snr(img, rho, pa, r.pxscale, r.fwhm, declip_nsig=None, return_details=True)
        kept.append(det[0]["nclean"])
    assert min(kept) <= few <= most <= max(kept), (few, most, sorted(set(kept)))
    assert few > 0 or min(kept) == 0
