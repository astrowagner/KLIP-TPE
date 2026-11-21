"""Frames that hold NaN beyond a detector's edge: the outer regions only some frames cover (NOMIC note, 2026-11-21).

The synthetic cubes here start 13.5 px left of the star in every frame, NaN beyond, so on the sky the edge
turns with the frames and a pixel of the outer circles is covered by some frames and not by others, as
NOMIC's beyond 3".  The edge lies on the left, as NOMIC's does, where the predecessor's high-pass carries
each NaN along its row and column over the whole frame.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from klip_tpe import CalibrationConfig, MawetPeakSNR, Objective, Param, PositionSampler, RunConfig, SearchSpace
from klip_tpe.injection import GaussianPSF
from klip_tpe.klip import KLIPParams, coverage_map, derotate, klip_annular, rotate_ccw
from klip_tpe.metrics import WHY_COVERAGE, RingRecorder, mawet_peak_snr, source_xy
from klip_tpe.reducer import Dataset, KLIPReducer, ReductionRequest
from klip_tpe.runner import Runner
from klip_tpe.synthetic import synthetic_klip_dataset

PX = 0.05
LAM = 3.89 * PX / 206265 * 8.2                      # FWHM about 4 px
FIRST = 18                                          # the first column with data, 13.5 px left of the star
PARAMS = dict(k_klip=4, bin=2, filter=7, n_ang=2, angsep=0.3, anglemax=360.0, inrad=6, outrad=28, fast=False,
              n_min_ref=4, corr_thresh=0.0, noise_max=10.0, coronoise_max=10.0)


def _edge(ds: Dataset) -> Dataset:
    cube = np.array(ds.cube, np.float32)
    cube[:, :, :FIRST] = np.nan
    return Dataset(cube, ds.angles, ds.tags, texp=ds.texp, name="edge", meta=dict(ds.meta or {}))


def _nanmean(cube: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # the pixels no frame covers
        return np.nanmean(cube, axis=0)


def _reducer(ds: Dataset, **defaults) -> KLIPReducer:
    return KLIPReducer(ds, PX, LAM, 8.2, injection_model=GaussianPSF(4.0, star_flux=1e4), outrad_cap=30,
                       defaults=defaults or None)


@pytest.fixture(scope="module")
def edge_data() -> Dataset:
    return _edge(synthetic_klip_dataset(nframes=24, size=64, pa_span=90, seed=4))


def test_frames_with_nan_beyond_the_edge_reduce_without_dropping_any(edge_data):
    red = _reducer(edge_data)
    assert red.has_nan and red.defaults["hp_nan_aware"], "the NaN-aware high-pass comes on by itself"
    res = red.reduce(ReductionRequest(dict(PARAMS), None))
    clean = _reducer(synthetic_klip_dataset(nframes=24, size=64, pa_span=90, seed=4))
    assert not clean.has_nan and not clean.defaults["hp_nan_aware"], "frames without NaN change nothing"
    ref = clean.reduce(ReductionRequest(dict(PARAMS), None))
    assert res.meta["n_binned"] == ref.meta["n_binned"], "no bin dropped as empty"
    assert np.isfinite(res.image).sum() > 1000
    # the predecessor's high-pass, asked for, spreads each NaN over its frame, as before
    with pytest.raises(ValueError, match="every frame was dropped as empty"):
        _reducer(edge_data, hp_nan_aware=False).reduce(ReductionRequest(dict(PARAMS), None))


def test_each_sky_pixel_is_the_mean_over_the_frames_that_cover_it():
    n, size = 8, 64
    angles = np.linspace(-45.0, 45.0, n)
    v = np.arange(1.0, n + 1.0)
    cube = np.full((n, size, size), np.nan, np.float32)
    cube[:, :, FIRST:] = v[:, None, None]               # frame i holds v_i where it has data
    p = KLIPParams(k_klip=0, inrad=0.0, outrad=31.0, fast=True)
    res, info = klip_annular(cube, angles, p, 4.0)
    img = _nanmean(derotate(res, angles))
    yy, xx = np.mgrid[0:size, 0:size]
    zone = np.hypot(xx - 31.5, yy - 31.5) < 25.0                    # inside the KLIP zone at every angle
    # each frame's share of each sky pixel, its bilinear weight on the frame's data once turned
    w = np.array([np.nan_to_num(rotate_ccw(np.isfinite(cube[i]).astype(float), angles[i])) for i in range(n)])
    cov = w > 1e-6
    want = (v[:, None, None] * w * cov).sum(axis=0) / np.where(cov.any(axis=0), cov.sum(axis=0), np.nan)
    some = zone & cov.any(axis=0) & ~cov.all(axis=0)
    assert some.sum() > 300, "pixels covered by some frames and not others"
    np.testing.assert_allclose(img[some], want[some], rtol=1e-5)
    # away from every frame's edge, the plain mean of the frames that cover the pixel
    full = w > 1.0 - 1e-6
    clear = some & (full | (w < 1e-6)).all(axis=0)
    assert clear.sum() > 100
    np.testing.assert_allclose(img[clear], (v[:, None, None] * full).sum(axis=0)[clear] / full.sum(axis=0)[clear],
                               rtol=1e-5)
    assert np.isnan(img[zone & ~cov.any(axis=0)]).all(), "no frame there, no data"
    assert info["nan_target_px"] > 0
    # the map counts the same frames
    np.testing.assert_allclose(coverage_map(cube, angles, max_frames=None), cov.mean(axis=0))
    # zero-filled, as before, the missing frames pull each pixel toward 0
    old, _ = klip_annular(cube, angles, KLIPParams(k_klip=0, inrad=0.0, outrad=31.0, fast=True, keep_nan=False), 4.0)
    old_img = _nanmean(derotate(old, angles))
    assert np.all(old_img[clear] < img[clear] - 1e-3)


@pytest.mark.parametrize("kw", [dict(fast=True), dict(fast=False, angsep=0.5, n_ang=2),
                                dict(fast=False, angsep=0.5, gram="zone"), dict(fast=False, k_scan=True)])
def test_keep_nan_changes_only_the_targets_nan_pixels(edge_data, kw):
    cube = np.array(edge_data.cube, np.float32)
    ang = np.asarray(edge_data.angles, float)
    base = dict(k_klip=4, inrad=4.0, outrad=30.0, n_min_ref=4)
    new, info = klip_annular(cube, ang, KLIPParams(**base, **kw), 4.0)
    old, _ = klip_annular(cube, ang, KLIPParams(**base, keep_nan=False, **kw), 4.0)
    fin = np.isfinite(cube)
    if new.ndim == 4:
        fin = np.broadcast_to(fin, new.shape)
    np.testing.assert_array_equal(new[fin], old[fin])
    assert np.isnan(new[~fin]).all()
    assert np.isfinite(old[~fin]).any(), "zero-filled, they came out as minus the model"
    assert info["nan_target_px"] > 0


def test_the_gpu_formulation_keeps_the_same_nan(edge_data):
    from klip_tpe.gpu_reducer import GPUKLIPReducer
    red = _reducer(edge_data)
    g = GPUKLIPReducer.from_cpu(red)
    g.backend_name = "numpy"
    req = dict(PARAMS)
    a = red.reduce(ReductionRequest(req, None)).image
    b = g.reduce(ReductionRequest(req, None)).image
    assert g.stats["gpu_jobs"] == 1, "the device path ran, not the CPU fallback"
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    noise = float(np.nanstd(a))
    assert np.nanmax(np.abs(a - b)) <= 1e-3 * noise


def _runner(red, tmp_path, name, **cfg_kw):
    space = SearchSpace([Param("k_klip", 1, 10, "int", default=5)])
    metric = MawetPeakSNR(pxscale=PX, fwhm=red.fwhm)
    obj = Objective(metric, clean_subtract=True)
    samp = PositionSampler(fwhm_as=red.fwhm * PX)
    cfg = RunConfig(ann_edges=[6, 28], n_iter=2, n_init=2, seed=1, calibration=CalibrationConfig(forced=[1e-3]),
                    save_fits=False, save_eval_images=False, write_setup_files=False, fm_curve=False, **cfg_kw)
    lines = []
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / name), log=lines.append)
    return r, metric, samp, lines


def test_a_run_masks_where_too_few_frames_hold_data(edge_data, tmp_path):
    red = _reducer(edge_data)
    r, metric, samp, lines = _runner(red, tmp_path, "run")
    assert os.path.exists(tmp_path / "run" / "coverage.fits")
    cov = r.coverage
    assert cov.shape == (64, 64) and cov.max() == pytest.approx(1.0) and cov.min() == 0.0
    low = cov < 0.5
    np.testing.assert_array_equal(metric.coverage_mask, low)
    assert (metric.pixel_mask >= low).all() and (samp.forbidden_mask >= low).all()
    assert samp.mask_pxscale == PX
    assert any("coverage: every frame holds data within" in s for s in lines)
    # a source where fewer than half the frames hold data is refused, one where all do is not
    rho = 20 * PX
    pas = np.arange(0.0, 360.0, 5.0)
    xs, ys = source_xy(np.full(pas.size, rho), pas, PX, 31.5, 31.5, red.angle_convention)
    c = cov[np.round(ys).astype(int), np.round(xs).astype(int)]
    assert samp._on_mask([rho], [float(pas[np.argmin(c)])]) and not samp._on_mask([rho], [float(pas[np.argmax(c)])])
    # read back by a run started again on the directory, not computed again
    r2, _, _, lines2 = _runner(red, tmp_path, "run")
    np.testing.assert_array_equal(r2.coverage, cov)
    assert not any("in 0 s" in s or "pixel, in" in s for s in lines2)
    # a run that began without the masks resumes as it began, its directory only read
    (tmp_path / "older").mkdir()
    (tmp_path / "older" / "results.jsonl").write_text("")
    r4, metric4, samp4, lines4 = _runner(red, tmp_path, "older")
    assert r4.coverage is None and metric4.pixel_mask is None and samp4.forbidden_mask is None
    assert not os.path.exists(tmp_path / "older" / "coverage.fits")
    assert any("began without the coverage masks" in s for s in lines4)
    # frames without NaN: no map and no masks
    r3, metric3, samp3, _ = _runner(_reducer(synthetic_klip_dataset(nframes=24, size=64, pa_span=90, seed=4)),
                                    tmp_path, "clean")
    assert r3.coverage is None and metric3.pixel_mask is None and samp3.forbidden_mask is None
    assert not os.path.exists(tmp_path / "clean" / "coverage.fits")


def test_no_noise_aperture_falls_where_too_few_frames_hold_data():
    rng = np.random.default_rng(3)
    n = 96
    img = rng.normal(0.0, 1.0, (n, n))
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    fwhm = 4.0
    img += 12.0 * np.exp(-0.5 * ((xx - c) ** 2 + (yy - (c + 30)) ** 2) / (fwhm / 2.3548) ** 2)   # PA 0, 30 px
    cov = xx > c + 12                                     # the right side has too few frames
    with RingRecorder() as rec:
        snr, det = mawet_peak_snr(img, [30 * PX], [0.0], PX, fwhm, coverage_mask=cov, return_details=True)
    d = det[0]
    ring = rec.records[-1]["ring"]
    used = [(x, y) for x, y, why in ring if str(why).startswith("used")]
    assert used and not any(cov[y, x] for x, y in used)
    assert d["n_uncovered"] == sum(why == "uncovered" for _, _, why in ring) > 0
    # the geometric penalty counts the apertures on data only
    want = (d["peak"] - d["bg"]) / (d["sigma"] * np.sqrt(1.0 + 1.0 / (d["nap"] - d["n_uncovered"])))
    assert snr[0] == pytest.approx(want)
    # a ring the coverage leaves short gives no S/N, where the pixel mask alone falls back to the band
    starve = ~((xx < c - 20) & (np.abs(yy - c) < 8)) & ~((xx - c) ** 2 + (yy - c - 30) ** 2 < 49)
    s1, d1 = mawet_peak_snr(img, [30 * PX], [0.0], PX, fwhm, coverage_mask=starve, return_details=True)
    s2, d2 = mawet_peak_snr(img, [30 * PX], [0.0], PX, fwhm, pixel_mask=starve, return_details=True)
    assert np.isnan(s1[0]) and d1[0]["why"] == WHY_COVERAGE
    assert np.isfinite(s2[0]) and not d2[0]["ring_used"]
    # and without a coverage mask the rings are as they were
    s3, d3 = mawet_peak_snr(img, [30 * PX], [0.0], PX, fwhm, return_details=True)
    assert d3[0]["n_uncovered"] == 0


def test_partitions_weigh_their_maps_by_their_frames(edge_data):
    from klip_tpe.reducer import PartitionedReducer
    a = _reducer(edge_data)
    other = _edge(synthetic_klip_dataset(nframes=12, size=64, pa_span=40, seed=5))
    b = _reducer(other)
    both = PartitionedReducer({"a": a, "b": b})
    assert both.has_nan and not PartitionedReducer({"c": _reducer(synthetic_klip_dataset(nframes=8, size=64))}).has_nan
    want = (24 * a.coverage_map() + 12 * b.coverage_map()) / 36.0
    np.testing.assert_allclose(both.coverage_map(), want, rtol=1e-6)


def test_the_curve_file_lists_the_position_angles_of_each_separation(tmp_path):
    from klip_tpe.stitch import write_contrast_curve
    fin = {"r_as": [3.5, 4.0], "curve": [2e-4, 3e-4], "n": [18, 12], "n_pos": [20, 13], "n_all": [20, 20],
           "annulus": [1, 1], "target_snr": 5.0, "read_with": "grid", "seams": [], "seam_steps": None}
    path = str(tmp_path / "contrast_curve.txt")
    write_contrast_curve(path, [None], [0.0, 300.0], 15.65, 0.0179, final_curve=fin)
    text = open(path).read()
    assert "# sep_arcsec   n_settled   n_tested   n_laid_out" in text
    rows = [ln.split() for ln in text.splitlines() if ln.startswith("#    4.000")]
    assert rows == [["#", "4.000", "12", "13", "20"]], "12 of 20 at 4.0\", as the IDL predecessor's curve"
    data = np.loadtxt(path)
    assert data.shape == (2, 2), "the rows of numbers are the curve's alone"
