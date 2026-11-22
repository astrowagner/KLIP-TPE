"""Tests of the companion extraction step (klip_tpe.extract)."""
from __future__ import annotations

import dataclasses
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


def _held(c_true, b=0.27, n=41):
    """Return a reduction with the position held: smooth noise, and the companion less the injection."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(1)
    cx, cy = star_center((n, n))
    (x0,), (y0,) = source_xy([TRUE.rho], [TRUE.theta], PX, cx, cy, "pa")
    yy, xx = np.mgrid[0:n, 0:n]
    psf = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * (4.0 / 2.355) ** 2))
    noise = gaussian_filter(rng.standard_normal((n, n)), 4.0 / 2.355)
    mask = patch_mask((n, n), TRUE.rho, TRUE.theta, 8.0, PX, "pa")
    noise *= b * patch_residual(c_true * psf, mask) / patch_residual(noise, mask)
    return (lambda srcs: noise + c_true * psf + sum(s.contrast * psf for s in (srcs or []))), mask


def test_fit_contrast_refines_before_its_profile_and_takes_the_given_width():
    rf, mask = _held(4e-3)
    cs = np.linspace(2e-3, 6e-3, 4001)
    c_min = cs[int(np.argmin([patch_residual(rf([Source(TRUE.rho, TRUE.theta, -c)]), mask) for c in cs]))]
    fits = [fit_contrast(rf, TRUE.rho, TRUE.theta, c0, fwhm_px=4.0, pxscale=PX, half_frac=0.05)
            for c0 in (0.853 * c_min, c_min, 1.11 * c_min)]
    for g in fits:
        # wherever the coarse scan's points fall, the fit lands on the minimum
        assert g["ok"] and g["contrast"] == pytest.approx(c_min, rel=2e-3)
        assert g["half"] == pytest.approx(0.05 * g["refined"])
        lo, hi = min(g["profile"]["contrast"]), max(g["profile"]["contrast"])
        assert 0.5 * (hi - lo) == pytest.approx(g["half"], rel=1e-6)
    # the sigma where the parabola has risen by its floor grows with the profile's width: a profile as
    # wide as the coarse scan's steps reads the scan, not the data
    wide = fit_contrast(rf, TRUE.rho, TRUE.theta, c_min, fwhm_px=4.0, pxscale=PX, half_frac=0.4)
    assert wide["sigma"] > 10 * fits[1]["sigma"] and wide["floor"] > 100 * fits[1]["floor"]
    with pytest.raises(ValueError, match="half_frac"):
        fit_contrast(rf, TRUE.rho, TRUE.theta, c_min, fwhm_px=4.0, pxscale=PX, half_frac=0.0)


def test_an_exactly_placed_companion_fits_back_at_its_contrast():
    """A companion drawn at its own sub-pixel position in every frame, not interpolated, is fitted back at
    its contrast to better than 1 % with the cubic placement.  The bilinear one fits it high."""
    fw, c_true = 2.5, 1e-2
    ds = synthetic_klip_dataset(nframes=40, size=64, pa_span=60, seed=3, speckle_amp=5.0, noise=0.1)
    model = GaussianPSF(fw, star_flux=1e4)
    norm = 1.0 / float(model._stamp.max())              # the unit stamp's divisor: its center pixel was 1
    n = ds.cube.shape[-1]
    yy, xx = np.mgrid[0:n, 0:n]
    sg = fw / 2.3548
    cube = np.array(ds.cube, np.float64)
    for j, ang in enumerate(ds.angles):                  # the injection's own geometry, pa convention, north 0
        az = np.deg2rad(TRUE.theta - 270.0 - ang)
        x0, y0 = (n - 1) / 2.0 + TRUE.rho / PX * np.cos(az), (n - 1) / 2.0 + TRUE.rho / PX * np.sin(az)
        cube[j] += c_true * model.flux_unit * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sg ** 2) / norm
    red = KLIPReducer(Dataset(cube.astype(np.float32), ds.angles, ds.tags, texp=ds.texp, name="exact", meta={}),
                      PX, LAM, 8.2, injection_model=model, outrad_cap=30)
    got = {}
    for interp in ("cubic", "linear"):
        rf = (lambda srcs, interp=interp: red.reduce(ReductionRequest(dict(PARAMS, inject_interp=interp),
                                                                      list(srcs) if srcs else None)).image)
        g = fit_contrast(rf, TRUE.rho, TRUE.theta, 0.8 * c_true, fwhm_px=red.fwhm, pxscale=PX,
                         angle_convention=red.angle_convention, nref=3, half_frac=0.05)
        got[interp] = g["contrast"] / c_true
    assert abs(got["cubic"] - 1.0) < 0.01, got
    assert got["linear"] > 1.03, got


def test_debias_tests_the_fakes_bias_with_students_t():
    from klip_tpe.extract import _debias
    r = _debias(60.0, 0.15, 0.03, 5, "pa")
    assert r["corrected"] == pytest.approx(59.85) and r["t"] == pytest.approx(5.0) and r["se"] == 0.03
    assert r["t95"] == pytest.approx(2.776, abs=1e-3) and r["significant"]
    assert _debias(0.1, 0.2, 0.01, 5, "pa")["corrected"] == pytest.approx(359.9)
    assert not _debias(0.9, 2e-4, 5e-4, 5, "rho")["significant"]
    c = _debias(4e-3, -0.02, 0.009, 5, "contrast")
    assert c["corrected"] == pytest.approx(4e-3 / 0.98) and c["se"] == pytest.approx(0.009 * c["corrected"])
    assert not c["significant"], "t -2.2 is short of 2.78 for 4 degrees of freedom"
    one = _debias(0.9, 0.5, float("nan"), 1, "rho")
    assert not one["significant"] and np.isnan(one["corrected"])


def test_the_summary_quotes_the_fit_less_a_significant_bias(red):
    from klip_tpe.extract import _summary, _txt
    f = fit_companion(_reduce_fn(red), TRUE.rho, TRUE.theta, TRUE.contrast, fwhm_px=red.fwhm, pxscale=PX,
                      angle_convention=red.angle_convention, config=QUICK)
    b = f["best"]
    rec = {"fakes": [{}] * 5, "bias": {"rho": 1e-4, "pa": 0.15, "contrast": -0.02},
           "bias_se": {"rho": 5e-4, "pa": 0.01, "contrast": 0.009},
           "scatter": {"rho": 1.1e-3, "pa": 0.022, "contrast": 0.02}, "n_reductions": 50}

    def group(pid, bias, se):
        return {"partition": pid, "lam_um": 3.5, "contrast": 4e-3, "sigma": 1e-5, "ok": True, "floor": 1e-6,
                "best": 4e-3, "n_reductions": 40, "recovery": {"fakes": [{}] * 5, "bias": bias, "bias_se": se,
                                                               "scatter": se * 5 ** 0.5, "n_reductions": 100}}

    found = {"fit": f, "groups": [group("a", -0.03, 0.005), group("b", 0.005, 0.006)], "recovery": rec,
             "config": dataclasses.asdict(QUICK), "run_dir": "run", "fwhm_px": red.fwhm, "pxscale": PX,
             "configuration": {"what": "a test", "annulus": 0, "inrad": 8.0, "outrad": 26.0},
             "angle_convention": red.angle_convention, "left_out": []}
    s = _summary(found)
    assert s["bias_test"]["significant"] == {"rho": False, "pa": True, "contrast": False}
    assert s["quoted"]["pa"] == pytest.approx((b["pa"] - 0.15) % 360)
    assert s["quoted"]["rho"] == b["rho"] and s["quoted"]["contrast"] == b["contrast"]
    assert s["corrected"]["contrast"] == pytest.approx(b["contrast"] / 0.98)
    assert s["sigma_quoted"]["pa"] == pytest.approx(np.hypot(max(s["sigma_profile"]["pa"], 0.022), 0.01))
    assert s["sigma_quoted"]["rho"] == pytest.approx(max(s["sigma_profile"]["rho"], 1.1e-3))
    a, bb = s["groups"]
    assert a["bias_significant"] and a["contrast_quoted"] == pytest.approx(4e-3 / 0.97)
    assert a["sigma_quoted"] == pytest.approx(np.hypot(0.005 * 5 ** 0.5 * 4e-3, 0.005 * 4e-3 / 0.97))
    assert not bb["bias_significant"] and bb["contrast_quoted"] == 4e-3
    text = _txt(s, "a test")
    assert "best less bias" in text and "pa is the best less the fakes' significant bias" in text
    assert "less its significant bias" in text


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


def test_extract_takes_overrides_places_by_cubic_and_skips_the_fm_pass(red, tmp_path, monkeypatch):
    r = _finished_runner(red, tmp_path, subtract=False)
    seen = []
    real = r._reduce

    def spy(cfg, sources, *args, **kw):
        seen.append((dict(cfg.params), kw.get("fm_sources", "absent")))
        return real(cfg, sources, *args, **kw)

    monkeypatch.setattr(r, "_reduce", spy)
    events = []
    cfg = ExtractConfig(n_iter=24, n_init=10, nref=0, n_fakes=0, seed=5)
    found = extract_companion(r, TRUE.rho, TRUE.theta, TRUE.contrast, config=cfg, params={"k_klip": 3},
                              log=lambda s: None, progress=events.append)
    assert all(p["k_klip"] == 3 and p["inject_interp"] == "cubic" for p, _ in seen)
    assert all(fm == () for _, fm in seen), "the step reads the image alone, so no KLIP-FM pass"
    c = found["configuration"]
    assert c["overrides"] == {"k_klip": 3} and c["what"].endswith("with k_klip=3") and c["inject_interp"] == "cubic"
    assert events[-1]["kind"] == "result" and set(events[-1]["quoted"]) == {"rho", "pa", "contrast"}
    # params can name the placement too, and the summary says which it was
    seen.clear()
    found = extract_companion(r, TRUE.rho, TRUE.theta, TRUE.contrast, config=cfg,
                              params={"inject_interp": "linear"}, log=lambda s: None)
    assert all(p["inject_interp"] == "linear" for p, _ in seen)
    from klip_tpe.extract import _summary, _txt
    assert "placed by linear interpolation" in _txt(_summary(found), "t")
    with pytest.raises(ValueError, match="inject_interp"):
        ExtractConfig(inject_interp="nearest").check()


def test_a_run_carries_its_placement_in_every_reduction(red, tmp_path, monkeypatch):
    r = _finished_runner(red, tmp_path, subtract=False)
    seen = []
    real = red.reduce
    monkeypatch.setattr(red, "reduce", lambda req: (seen.append(dict(req.params)), real(req))[1])
    win = r.space.decode(r.space.default_vector())
    r._reduce(win, None)
    assert "inject_interp" not in seen[-1], "None leaves it to the reducer's default"
    r.cfg.inject_interp = "cubic"
    r._reduce(win, [TRUE])
    assert seen[-1]["inject_interp"] == "cubic"
    from klip_tpe.runner import RunConfig
    assert RunConfig.from_dict(dict(r.cfg.to_dict())).inject_interp == "cubic"
    assert RunConfig.from_dict({k: v for k, v in r.cfg.to_dict().items() if k != "inject_interp"}).inject_interp is None


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
    ec = ExtractConfig(n_iter=40, n_init=15, nref=3, n_fakes=2, fake_iter=25, seed=2, group_positions=True,
                       group_iter=30, group_init=10)
    found = extract_companion(r, TRUE.rho, TRUE.theta + 1.0, 3e-3, config=ec, log=lambda s: None)
    g = {x["partition"]: x for x in found["groups"]}
    assert g["a"]["contrast"] == pytest.approx(4e-3, rel=0.15) and g["b"]["contrast"] == pytest.approx(2e-3, rel=0.15)
    # the partitions' profiles take the combined fit's relative width
    hf = min(max(found["fit"]["profile_half"][2] / found["fit"]["best"]["contrast"], 0.01), 0.5)
    assert all(x["half"] == pytest.approx(hf * x["refined"]) for x in found["groups"])
    assert all(len(x["recovery"]["fakes"]) == 2 for x in found["groups"])
    files = write_extract(found, str(tmp_path / "out2"))
    s = json.load(open(files["summary_json"]))
    assert all(x["sigma_quoted"] >= x["sigma"] for x in s["groups"])
    assert os.path.exists(files["spectrum_pdf"]) and os.path.exists(files["partitions"])
    # each partition's own position, the same in both: their mean sits on the combined fit
    gp = s["group_positions"]
    assert [r["partition"] for r in gp["rows"]] == ["a", "b"]
    for row in gp["rows"]:
        assert _px_offset(red, (row["rho"], row["pa"]), (TRUE.rho, TRUE.theta)) < 1.0
        assert row["sigma_dx_px"] > 0 and row["sigma_dy_px"] > 0
    assert abs(gp["mean"]["ddx_px"]) < 1.0 and abs(gp["mean"]["ddy_px"]) < 1.0
    assert gp["chromatic"]["dx"]["dof"] == 1 and gp["wavelength"] in ("um", "index")
    assert os.path.exists(files["positions_pdf"]) and os.path.exists(files["partition_positions"])
    assert "chromatic:" in open(files["summary_txt"]).read()


def test_the_cli_parses_extract():
    from klip_tpe.cli import _build_parser, _set_params
    a = _build_parser().parse_args(["extract", "--run-dir", "r", "--rho", "0.452", "--pa", "211.2", "--contrast",
                                   "7e-4", "--n-iter", "300", "--fakes", "0", "--box-contrast", "0.6", "1.8"])
    assert a.rho == 0.452 and a.n_iter == 300 and a.fakes == 0 and a.box_contrast == [0.6, 1.8]
    assert a.patch == 2.0 and a.nref == 5 and a.annulus is None and a.config is None
    assert not a.group_positions and a.group_iter == 300 and a.group_box == 0.5
    assert a.extract_interp == "cubic" and a.set_params == [] and getattr(a, "inject_interp", None) is None
    a = _build_parser().parse_args(["extract", "--rho", "0.3", "--pa", "20", "--contrast", "1e-3", "--set",
                                   "nkeep_altroll=0", "--set", "rdi_mode=rdi", "--inject-interp", "linear"])
    assert a.extract_interp == "linear"
    assert _set_params(a.set_params) == {"nkeep_altroll": 0, "rdi_mode": "rdi"}
    assert _set_params(["a=1.5", "b=true", "c=None", "d=x=y"]) == {"a": 1.5, "b": True, "c": None, "d": "x=y"}
    with pytest.raises(ValueError, match="KEY=VALUE"):
        _set_params(["nokey"])
