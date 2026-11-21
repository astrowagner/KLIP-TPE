"""Tests of the companion extraction step (klip_tpe.extract)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from klip_tpe import (CalibrationConfig, MawetPeakSNR, Objective, Param, PositionSampler, RunConfig,
                      SearchSpace, ValidationConfig)
from klip_tpe.extract import (ExtractConfig, extract_companion, fit_companion, fit_contrast, parabola_sigma,
                              patch_mask, patch_residual, quadratic_sigma, recovery_check, write_extract)
from klip_tpe.injection import GaussianPSF, inject_sources, read_sources
from klip_tpe.metrics import Source, source_xy, star_center
from klip_tpe.reducer import Dataset, KLIPReducer, PartitionedReducer, ReductionRequest
from klip_tpe.runner import AnnulusResult, Runner
from klip_tpe.synthetic import synthetic_klip_dataset

PX = 0.05
LAM = 3.89 * PX / 206265 * 8.2                      # -> fwhm ~ 4 px on an 8.2 m aperture
TRUE = Source(0.9, 60.0, 4e-3)                      # the companion put into the data
PARAMS = dict(k_klip=5, inrad=8, outrad=26, n_ang=1, angsep=0.0, anglemax=360.0, fast=True, bin=1, filter=0)
QUICK = ExtractConfig(n_iter=60, n_init=20, nref=3, n_fakes=0, seed=7)


@pytest.fixture(scope="module")
def red():
    ds = synthetic_klip_dataset(nframes=40, size=64, pa_span=60, seed=3)
    model = GaussianPSF(4.0, star_flux=1e4)
    base = KLIPReducer(ds, PX, LAM, 8.2, injection_model=model, outrad_cap=30)
    cube = inject_sources(ds.cube, ds.angles, [TRUE], model, PX, angle_convention=base.angle_convention)
    ds2 = Dataset(cube, ds.angles, ds.tags, texp=ds.texp, name="with_companion", meta=dict(ds.meta or {}))
    return KLIPReducer(ds2, PX, LAM, 8.2, injection_model=model, outrad_cap=30)


def _reduce_fn(red, params=PARAMS):
    return lambda srcs: red.reduce(ReductionRequest(dict(params), list(srcs) if srcs else None)).image


def _px_offset(red, a, b):
    cx, cy = star_center((64, 64))
    xa, ya = source_xy([a[0]], [a[1]], PX, cx, cy, red.angle_convention)
    xb, yb = source_xy([b[0]], [b[1]], PX, cx, cy, red.angle_convention)
    return float(np.hypot(xa[0] - xb[0], ya[0] - yb[0]))


def test_quadratic_sigma_reads_the_curvature_at_the_floor():
    rng = np.random.default_rng(0)
    H = np.array([[8.0, 1.0, 0.5], [1.0, 4.0, 0.0], [0.5, 0.0, 2.0]])     # in scaled units
    scale = np.array([0.1, 10.0, 2e-3])
    x0 = np.array([0.9, 60.0, 4e-3])
    u = rng.uniform(-0.3, 0.3, (600, 3))
    noise = 0.002
    R = 0.1 + 0.5 * np.einsum("ni,ij,nj->n", u, H, u) + rng.normal(0, noise, 600)
    f = quadratic_sigma(x0 + u * scale, R, x0, scale)
    assert f["ok"] and f["floor"] == pytest.approx(noise, rel=0.1)
    want = np.sqrt(2.0 * noise * np.diag(np.linalg.inv(H))) * scale
    assert np.allclose(f["sigma"], want, rtol=0.1)
    # too few trials, or a saddle, give no sigma
    assert not quadratic_sigma(x0 + u[:8] * scale, R[:8], x0, scale)["ok"]
    assert not quadratic_sigma(x0 + u * scale, -R, x0, scale)["ok"]


def test_parabola_sigma_finds_the_vertex_and_the_rise_of_the_floor():
    rng = np.random.default_rng(1)
    x = np.linspace(-1.0, 1.0, 21)
    y = 1.0 + 3.0 * (x / 2.0 - 0.1) ** 2 + rng.normal(0, 0.01, x.size)
    f = parabola_sigma(x, y, 0.0, 2.0)
    assert f["ok"] and f["minimum"] == pytest.approx(0.2, abs=0.02)
    assert f["sigma"] == pytest.approx(np.sqrt(f["floor"] / f["coef"][2]) * 2.0)
    assert not parabola_sigma(x, -y, 0.0, 2.0)["ok"], "a parabola that opens downward has no sigma"
    assert not parabola_sigma(x[:4], y[:4], 0.0, 2.0)["ok"]


def test_patch_residual_is_the_patch_norm_less_its_mean():
    img = np.full((32, 32), 3.0)
    m = patch_mask(img.shape, 0.4, 0.0, 3.0, 0.05)
    assert patch_residual(img, m) == 0.0
    img[m] += np.where(np.arange(m.sum()) % 2, 1.0, -1.0)
    assert patch_residual(img, m) == pytest.approx(np.sqrt(m.sum()), rel=0.05)
    img[16, 16] = np.nan
    assert np.isfinite(patch_residual(img, m))
    with pytest.raises(ValueError, match="holds"):
        patch_mask(img.shape, 5.0, 0.0, 1.0, 0.05)
    with pytest.raises(ValueError):
        ExtractConfig(box_contrast=(1.2, 2.0)).check()


def test_fit_companion_recovers_the_injected_companion(red):
    f = fit_companion(_reduce_fn(red), TRUE.rho + 0.02, TRUE.theta + 1.5, 3e-3, fwhm_px=red.fwhm, pxscale=PX,
                      angle_convention=red.angle_convention, config=QUICK)
    b = f["best"]
    assert _px_offset(red, (b["rho"], b["pa"]), (TRUE.rho, TRUE.theta)) < 0.5
    assert b["contrast"] == pytest.approx(TRUE.contrast, rel=0.1)
    assert b["residual"] < 0.5, "the companion is cancelled well below the un-subtracted patch"
    assert f["n_reductions"] == 1 + QUICK.n_iter + 3 * (4 * QUICK.nref + 1)
    assert len(f["trials"]["residual"]) == QUICK.n_iter
    sig = [p["sigma"] for p in f["profile_fits"]]
    assert all(np.isfinite(sig)) and all(s > 0 for s in sig)
    # the trials stay in the box
    X = np.asarray(f["trials"]["x"])
    assert np.all(X >= np.asarray(f["lo"]) - 1e-12) and np.all(X <= np.asarray(f["hi"]) + 1e-12)


def test_fit_contrast_with_the_position_held(red):
    g = fit_contrast(_reduce_fn(red), TRUE.rho, TRUE.theta, 2e-3, fwhm_px=red.fwhm, pxscale=PX,
                     angle_convention=red.angle_convention, nref=3)
    assert g["ok"] and g["contrast"] == pytest.approx(TRUE.contrast, rel=0.08)
    assert 0 < g["sigma"] < 0.5 * TRUE.contrast
    assert g["n_reductions"] == len(g["points"]["contrast"]) + 1


def test_recovery_check_fits_fakes_like_the_companion(red):
    f = fit_companion(_reduce_fn(red), TRUE.rho, TRUE.theta, TRUE.contrast, fwhm_px=red.fwhm, pxscale=PX,
                      angle_convention=red.angle_convention, config=QUICK)
    cfg = ExtractConfig(n_iter=60, n_init=20, n_fakes=2, fake_iter=40, seed=3)
    rc = recovery_check(_reduce_fn(red), f, fwhm_px=red.fwhm, pxscale=PX, angle_convention=red.angle_convention,
                        config=cfg)
    assert len(rc["fakes"]) == 2
    assert abs(rc["bias"]["contrast"]) < 0.15
    for fk in rc["fakes"]:
        assert _px_offset(red, (fk["fit"]["rho"], fk["fit"]["pa"]), (fk["true"]["rho"], fk["true"]["pa"])) < 1.0
    # the fakes sit at the companion's separation, away from it
    assert all(abs(fk["true"]["rho"] - f["best"]["rho"]) < 1e-12 for fk in rc["fakes"])
    assert {round(fk["true"]["pa"] - f["best"]["pa"]) % 360 for fk in rc["fakes"]} == {120, 240}


def _finished_runner(red, tmp_path, subtract):
    space = SearchSpace([Param("k_klip", 1, 10, "int", default=5)])
    obj = Objective(MawetPeakSNR(pxscale=PX, fwhm=red.fwhm, known=[(TRUE.rho, TRUE.theta)]), clean_subtract=True)
    samp = PositionSampler(fwhm_as=red.fwhm * PX, known=[(TRUE.rho, TRUE.theta)])
    cfg = RunConfig(ann_edges=[8, 26], n_iter=2, n_init=2, seed=1, n_sources=1,
                    validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[1e-3]),
                    save_fits=False, save_eval_images=False, write_setup_files=False, fm_curve=False,
                    subtract_known=[(TRUE.rho, TRUE.theta, TRUE.contrast)] if subtract else [])
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "run"), log=lambda s: None)
    win = space.decode(space.default_vector())
    r.results = [AnnulusResult(annulus=0, inrad=8.0, outrad=26.0, contrast=1e-3, search_best_index=0,
                               search_best_score=5.0, validated=True, winner_index=0, winner_score=5.0,
                               winner_x=list(win.x), winner_config=win.to_dict(), validation_table=[],
                               validation_samples={})]
    return r


def test_extract_companion_on_a_finished_run_writes_its_products(red, tmp_path):
    r = _finished_runner(red, tmp_path, subtract=True)
    cfg = ExtractConfig(n_iter=40, n_init=15, nref=3, n_fakes=1, fake_iter=30, seed=5)
    found = extract_companion(r, TRUE.rho + 0.01, TRUE.theta - 1.0, 3e-3, config=cfg, log=lambda s: None)
    # the companion in subtract_known was left out while it was fitted, and is back afterwards
    assert found["left_out"] and r.cfg.subtract_known == [(TRUE.rho, TRUE.theta, TRUE.contrast)]
    b = found["fit"]["best"]
    assert _px_offset(red, (b["rho"], b["pa"]), (TRUE.rho, TRUE.theta)) < 0.7
    assert found["configuration"]["what"].startswith("the elected winner")
    files = write_extract(found, str(tmp_path / "out"), label="test companion")
    for kind in ("summary_txt", "summary_json", "trials", "profiles", "fakes", "subtract_sources", "fit_pdf",
                 "images_png", "clean_fits", "residual_fits"):
        assert os.path.exists(files[kind]), kind
    s = json.load(open(files["summary_json"]))
    assert s["sigma_profile"]["rho"] > 0 and s["best"]["rho_px"] == pytest.approx(b["rho"] / PX)
    (src,) = read_sources(files["subtract_sources"])
    assert src.contrast == pytest.approx(b["contrast"], rel=1e-5)
    text = open(files["summary_txt"]).read()
    assert "#1 quadratic" in text and "#2 profiles" in text and "--subtract-sources line" in text
    # one fake has no scatter, so the sigma to quote is the profile's
    assert s["sigma_quoted"]["rho"] == pytest.approx(s["sigma_profile"]["rho"])


@pytest.fixture(scope="module")
def two_partitions():
    ds = synthetic_klip_dataset(nframes=40, size=64, pa_span=60, seed=3)
    model = GaussianPSF(4.0, star_flux=1e4)
    base = KLIPReducer(ds, PX, LAM, 8.2, injection_model=model, outrad_cap=30)
    reds = {}
    for pid, c in (("a", 4e-3), ("b", 2e-3)):        # the companion is brighter in "a"
        cube = inject_sources(ds.cube, ds.angles, [Source(TRUE.rho, TRUE.theta, c)], model, PX,
                              angle_convention=base.angle_convention)
        d = Dataset(cube, ds.angles, ds.tags, texp=ds.texp, name=pid, meta=dict(ds.meta or {}))
        reds[pid] = KLIPReducer(d, PX, LAM, 8.2, injection_model=model, outrad_cap=30)
    return PartitionedReducer(reds)


def test_each_partition_gets_its_own_contrast_with_the_position_held(two_partitions, tmp_path):
    red = two_partitions
    space = SearchSpace().replicate([Param("k_klip", 1, 10, "int", default=5)], ["a", "b"])
    obj = Objective(MawetPeakSNR(pxscale=PX, fwhm=red.fwhm), clean_subtract=True)
    samp = PositionSampler(fwhm_as=red.fwhm * PX)
    cfg = RunConfig(ann_edges=[8, 26], n_iter=2, n_init=2, seed=1, save_fits=False, save_eval_images=False,
                    write_setup_files=False, fm_curve=False)
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "run2"), log=lambda s: None)
    win = space.decode(space.default_vector())
    assert len(win.selected) == 2
    r.results = [AnnulusResult(annulus=0, inrad=8.0, outrad=26.0, contrast=1e-3, search_best_index=0,
                               search_best_score=5.0, validated=True, winner_index=0, winner_score=5.0,
                               winner_x=list(win.x), winner_config=win.to_dict(), validation_table=[],
                               validation_samples={})]
    ec = ExtractConfig(n_iter=40, n_init=15, nref=3, n_fakes=2, fake_iter=25, seed=2)
    found = extract_companion(r, TRUE.rho, TRUE.theta + 1.0, 3e-3, config=ec, log=lambda s: None)
    g = {x["partition"]: x for x in found["groups"]}
    assert g["a"]["contrast"] == pytest.approx(4e-3, rel=0.15) and g["b"]["contrast"] == pytest.approx(2e-3, rel=0.15)
    assert all(len(x["recovery"]["fakes"]) == 2 for x in found["groups"])
    files = write_extract(found, str(tmp_path / "out2"))
    s = json.load(open(files["summary_json"]))
    assert all(x["sigma_quoted"] >= x["sigma"] for x in s["groups"])
    assert os.path.exists(files["spectrum_pdf"]) and os.path.exists(files["partitions"])


def test_the_cli_parses_extract():
    from klip_tpe.cli import _build_parser
    a = _build_parser().parse_args(["extract", "--run-dir", "r", "--rho", "0.286", "--pa", "25.65", "--contrast",
                                   "5.5e-3", "--n-iter", "300", "--fakes", "0", "--box-contrast", "0.6", "1.8"])
    assert a.rho == 0.286 and a.n_iter == 300 and a.fakes == 0 and a.box_contrast == [0.6, 1.8]
    assert a.patch == 2.0 and a.nref == 5 and a.annulus is None and a.config is None
