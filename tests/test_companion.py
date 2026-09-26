"""Known companions: the negative-fake-companion fit, the forward-model S/N, and
RunConfig.subtract_known (klip_tpe.companion, Runner._reduce)."""
from __future__ import annotations

import numpy as np
import pytest

from klip_tpe import (CalibrationConfig, MawetPeakSNR, Objective, Param, PositionSampler, RunConfig,
                      SearchSpace, ValidationConfig)
from klip_tpe.companion import companion_snr, fit_negative_companion, ring_sigma
from klip_tpe.injection import GaussianPSF, inject_sources
from klip_tpe.metrics import Source, source_xy, star_center
from klip_tpe.reducer import Dataset, KLIPReducer, ReductionRequest
from klip_tpe.runner import Runner
from klip_tpe.synthetic import synthetic_klip_dataset

PX = 0.05
LAM = 3.89 * PX / 206265 * 8.2                      # -> fwhm ~ 4 px on an 8.2 m aperture
TRUE = Source(0.9, 60.0, 4e-3)                      # the companion put into the data
PARAMS = dict(k_klip=5, inrad=8, outrad=26, n_ang=1, angsep=0.0, anglemax=360.0, fast=True, bin=1, filter=0)


@pytest.fixture(scope="module")
def red_with_companion():
    ds = synthetic_klip_dataset(nframes=40, size=64, pa_span=60, seed=3)
    model = GaussianPSF(4.0, star_flux=1e4)
    base = KLIPReducer(ds, PX, LAM, 8.2, injection_model=model, outrad_cap=30)
    cube = inject_sources(ds.cube, ds.angles, [TRUE], model, PX, angle_convention=base.angle_convention)
    ds2 = Dataset(cube, ds.angles, ds.tags, texp=ds.texp, name="with_companion", meta=dict(ds.meta or {}))
    red = KLIPReducer(ds2, PX, LAM, 8.2, injection_model=model, outrad_cap=30)
    red.companion_free = base            # the same data without the companion, for reference
    return red


def _reduce_fn(red, params=PARAMS):
    return lambda srcs: red.reduce(ReductionRequest(dict(params), list(srcs) if srcs else None)).image


def test_fit_negative_companion_recovers_the_injected_companion(red_with_companion):
    red = red_with_companion
    f = fit_negative_companion(_reduce_fn(red), TRUE.rho + 0.02, TRUE.theta + 1.5, 2e-3, red.fwhm, PX,
                               angle_convention=red.angle_convention, max_evals=40)
    assert f["contrast"] == pytest.approx(TRUE.contrast, rel=0.08)
    # what is left in the stamp is what was there without the companion: the speckle residual
    free = _reduce_fn(red.companion_free)(None)
    cx0, cy0 = star_center(free.shape)
    x0, y0 = source_xy([TRUE.rho + 0.02], [TRUE.theta + 1.5], PX, cx0, cy0, red.angle_convention)
    yy, xx = np.mgrid[0:64, 0:64]
    st = (xx - x0[0]) ** 2 + (yy - y0[0]) ** 2 <= (2.0 * red.fwhm) ** 2
    assert f["chi2"] < 1.2 * np.nansum(free[st] ** 2)
    # the position correction found its way back toward the truth (started 0.4 px and 1.5 deg off)
    img = np.zeros((64, 64))
    cx, cy = star_center(img.shape)
    xt, yt = source_xy([TRUE.rho], [TRUE.theta], PX, cx, cy, red.angle_convention)
    xf, yf = source_xy([f["rho"]], [f["pa"]], PX, cx, cy, red.angle_convention)
    assert np.hypot(xf[0] - xt[0], yf[0] - yt[0]) < 0.5


def test_companion_snr_keeps_the_companion_out_of_its_own_noise(red_with_companion):
    red = red_with_companion
    m = MawetPeakSNR(pxscale=PX, fwhm=red.fwhm)
    r = companion_snr(_reduce_fn(red), TRUE.rho, TRUE.theta, TRUE.contrast, red.fwhm, PX, m.kernel([TRUE.rho]),
                      angle_convention=red.angle_convention)
    assert np.isfinite(r["snr"]) and r["snr"] > 5
    # the noise image does not hold the companion: its ring scatter is the empty-field value,
    # the same as the clean image's ring well away from the companion
    far = ring_sigma(r["clean"], TRUE.rho, TRUE.theta + 180.0, red.fwhm, PX, m.kernel([TRUE.rho]),
                     angle_convention=red.angle_convention, exclude=[(TRUE.rho, TRUE.theta)], excl_fwhm=4.0)
    assert r["sigma"] == pytest.approx(far["sigma"], rel=0.5)
    # and the companion is gone from the removed image
    cx, cy = star_center(r["clean"].shape)
    xs, ys = source_xy([TRUE.rho], [TRUE.theta], PX, cx, cy, red.angle_convention)
    yy, xx = np.mgrid[0:64, 0:64]
    st = (xx - xs[0]) ** 2 + (yy - ys[0]) ** 2 <= (1.5 * red.fwhm) ** 2
    assert np.nansum(r["removed"][st] ** 2) < 0.1 * np.nansum(r["clean"][st] ** 2)


def _runner(red, tmp_path, subtract):
    space = SearchSpace([Param("k_klip", 1, 10, "int", default=5)])
    obj = Objective(MawetPeakSNR(pxscale=PX, fwhm=red.fwhm, known=[(TRUE.rho, TRUE.theta)]), clean_subtract=True)
    samp = PositionSampler(fwhm_as=red.fwhm * PX, known=[(TRUE.rho, TRUE.theta)])
    cfg = RunConfig(ann_edges=[8, 26], n_iter=2, n_init=2, seed=1, n_sources=1,
                    validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[1e-3]),
                    save_fits=False, save_eval_images=False, write_setup_files=False, fm_curve=False,
                    subtract_known=[(TRUE.rho, TRUE.theta, TRUE.contrast)] if subtract else [])
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / ("sub" if subtract else "plain")), log=lambda s: None)
    r.ia, r.contrast = 0, 1e-3
    return r, space


def test_subtract_known_takes_the_companion_out_of_every_reduction(red_with_companion, tmp_path):
    red = red_with_companion
    plain, space = _runner(red, tmp_path, False)
    sub, _ = _runner(red, tmp_path, True)
    cfg = space.decode(space.default_vector())
    cx, cy = star_center((64, 64))
    xs, ys = source_xy([TRUE.rho], [TRUE.theta], PX, cx, cy, red.angle_convention)
    yy, xx = np.mgrid[0:64, 0:64]
    st = (xx - xs[0]) ** 2 + (yy - ys[0]) ** 2 <= (1.5 * red.fwhm) ** 2
    e_plain = np.nansum(plain._reduce(cfg, None).image[st] ** 2)
    e_sub = np.nansum(sub._reduce(cfg, None).image[st] ** 2)
    assert e_sub < 0.1 * e_plain
    # an injected reduction keeps its own source and loses the companion
    src = [Source(0.9, 240.0, 2e-3)]
    img = sub._reduce(cfg, src).image
    assert np.nansum(img[st] ** 2) < 0.1 * e_plain
    xi, yi = source_xy([0.9], [240.0], PX, cx, cy, red.angle_convention)
    assert img[int(round(yi[0])), int(round(xi[0]))] > 0
    # and nothing reaches the score but the caller's own sources
    r, _, _ = sub.evaluate(sub.space.default_vector(), "default", contrast=2e-3, sources=src, raw_only=True)
    assert len(r.raw_per_source) == 1


def test_subtract_known_round_trips_through_the_run_setup():
    cfg = RunConfig(subtract_known=[(0.8176, 150.12, 3.59e-4)])
    back = RunConfig.from_dict(cfg.to_dict())
    assert [tuple(map(float, t)) for t in back.subtract_known] == [(0.8176, 150.12, 3.59e-4)]
    assert RunConfig().subtract_known == []
