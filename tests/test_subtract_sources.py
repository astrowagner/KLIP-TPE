"""Point sources taken out of the data once, at load (``KLIPReducer.subtract_sources``).

What must hold:

* A list of zero contrasts changes nothing, to the bit.
* The frames are the ones an outside subtraction with the same injection code gives, to the
  bit, so a run on such a cube (``--cube-name``) and a run with ``--subtract-sources`` of the
  same list agree.
* Every path sees the edited frames: the clean pass, a draw's injected pass, the reference
  ranking (``_clean_bins``).  The KLIP-FM model cube does not.
* The caller's dataset is not changed, the edit is recorded, a partitioned reducer refuses
  once its workers run, and a separation given in pixels is caught.
* The float32 headroom check judges a negative contrast as it judges the positive one.
* The flux that left the frames is measured, a chain through 0 does not warn, and a list lost
  to float32 rounding does.
* Sources added at load (``inject_sources``) are an outside injection with the same code, to
  the bit, the same list subtracted takes them out again, and the FM model cube sees neither.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from klip_tpe.injection import GaussianPSF, _check_float32_headroom, inject_sources, read_sources
from klip_tpe.metrics import Source
from klip_tpe.reducer import Dataset, KLIPReducer, PartitionedReducer, ReductionRequest, _taken_text
from klip_tpe.synthetic import synthetic_klip_dataset

PX = 0.05
LAM = 3.89 * PX / 206265 * 8.2                       # FWHM about 4 px on an 8.2 m aperture
PARAMS = dict(k_klip=5, inrad=6, outrad=26, n_ang=1, angsep=0.0, anglemax=360.0, fast=True, bin=1, filter=0)
LINE = [Source(0.40 + 0.08 * i, 13.0 if i % 2 else 193.0, 2e-3 * (1 + 0.1 * i)) for i in range(8)]


def _ds(seed=3):
    return synthetic_klip_dataset(nframes=30, size=64, pa_span=60, seed=seed)


def _red(ds):
    return KLIPReducer(ds, PX, LAM, 8.2, injection_model=GaussianPSF(4.0, star_flux=1e4), outrad_cap=30)


def _clean(red, params=PARAMS, inj=None):
    return red.reduce(ReductionRequest(dict(params), list(inj) if inj else None)).image


def test_zero_contrasts_change_nothing_to_the_bit():
    ds = _ds()
    red = _red(ds)
    before = _clean(red)
    cube0 = red.data.cube
    n = red.subtract_sources([Source(s.rho, s.theta, 0.0) for s in LINE])
    assert n == 0 and red.data.cube is cube0
    assert np.array_equal(_clean(red).view(np.uint32), before.view(np.uint32))
    assert len(red.data_edits["subtracted_sources"]) == len(LINE)


def test_the_frames_are_an_outside_subtraction_with_the_same_code():
    ds = _ds()
    red = _red(ds)
    ext = inject_sources(ds.cube, ds.angles, [Source(s.rho, s.theta, -s.contrast) for s in LINE], red.model,
                         PX, truenorth=red.truenorth, angle_convention=red.angle_convention)
    red_ext = _red(Dataset(ext, ds.angles, ds.tags, texp=ds.texp, name="ext", meta=dict(ds.meta or {})))
    red.subtract_sources(LINE)
    assert red.data.cube.dtype == np.float32
    assert np.array_equal(red.data.cube.view(np.uint32), ext.view(np.uint32))
    # the clean pass and a draw's injected pass, to the bit
    assert np.array_equal(_clean(red).view(np.uint32), _clean(red_ext).view(np.uint32))
    draw = [Source(0.9, 100.0, 3e-3), Source(0.9, 280.0, 3e-3)]
    assert np.array_equal(_clean(red, inj=draw).view(np.uint32), _clean(red_ext, inj=draw).view(np.uint32))


def test_every_path_sees_it_and_the_fm_model_does_not():
    ds = _ds()
    free = _red(ds)
    # the data with the line in it, then the line taken out again
    with_line = inject_sources(ds.cube, ds.angles, LINE, free.model, PX, truenorth=free.truenorth,
                               angle_convention=free.angle_convention)
    red = _red(Dataset(with_line, ds.angles, ds.tags, texp=ds.texp, name="line", meta=dict(ds.meta or {})))
    raw = _clean(red)
    red.subtract_sources(LINE)
    # back to the line-free frames to float32 rounding, and far from the frames with the line
    scale = float(np.nanmax(np.abs(ds.cube)))
    assert np.nanmax(np.abs(red.data.cube - ds.cube)) < 4 * np.finfo(np.float32).eps * scale
    out = _clean(red)
    ref = _clean(free)
    assert np.nanmax(np.abs(out - ref)) < 1e-3 * np.nanmax(np.abs(raw - ref))
    # the reference ranking reads the edited frames
    keep = np.ones(red.data.nframes, bool)
    p = dict(red.defaults, **PARAMS)
    cb = red._clean_bins(p, keep, 1, 0, 1.0, False, False, 1)
    cb0 = free._clean_bins(p, keep, 1, 0, 1.0, False, False, 1)
    assert np.nanmax(np.abs(cb - cb0)) < 4 * np.finfo(np.float32).eps * scale
    # the KLIP-FM model cube is the sources in zeros, without the subtraction
    fm = [Source(0.9, 45.0, 3e-3)]
    assert np.array_equal(red.model_cube(fm), free.model_cube(fm))


def test_the_callers_dataset_is_not_changed_and_the_edit_is_described():
    ds = _ds()
    keep = ds.cube.copy()
    red = _red(ds)
    red.subtract_sources(LINE[:2])
    assert np.array_equal(ds.cube, keep) and red.data.cube is not ds.cube
    d = red.describe()
    assert d["data_edits"]["subtracted_sources"] == [list(s.as_tuple()) for s in LINE[:2]]


def test_a_partitioned_reducer_subtracts_from_every_night_and_refuses_once_its_workers_run():
    a, b = _ds(3), _ds(4)
    pr = PartitionedReducer({"n1": _red(a), "n2": _red(b)}, max_workers=2, pool="processes", log=lambda s: None)
    try:
        pr.subtract_sources(LINE[:3])
        for r in pr.reducers.values():
            assert len(r.data_edits["subtracted_sources"]) == 3
        assert pr.describe()["data_edits"]["subtracted_sources"] == [list(s.as_tuple()) for s in LINE[:3]]
        if pr.start_workers() != "processes":
            pytest.skip("no worker processes here")
        with pytest.raises(RuntimeError, match="before the worker pool starts"):
            pr.subtract_sources(LINE[:1])
    finally:
        pr.close()


def test_a_separation_in_pixels_is_caught():
    red = _red(_ds())
    with pytest.raises(ValueError, match="rho is in arcsec"):
        red.subtract_sources([(80.0, 13.0, 1e-3)])


def test_the_headroom_check_judges_a_negative_contrast_as_the_positive_one():
    ds = _ds()
    model = GaussianPSF(4.0, star_flux=1e4)

    def warns(c):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_float32_headroom(ds.cube, [Source(0.8, 13.0, c)], model)
        return any("float32" in str(x.message) for x in w)
    assert not warns(3e-3) and not warns(-3e-3)
    assert warns(1e-12) and warns(-1e-12)


def test_one_hot_pixel_does_not_set_the_headroom_level():
    """The level is the 99.9th percentile of the bright pixels, not the maximum: a single hot
    pixel far from a source no longer makes the check call a well-deposited source lost."""
    rng = np.random.default_rng(0)
    cube = (10.0 + rng.normal(0, 1, (8, 61, 61))).astype(np.float32)
    ang = np.linspace(0, 40, 8)
    model = GaussianPSF(3.5, star_flux=1.0)
    src = [Source(0.3, 70.0, -0.05)]
    cube[3, 5, 5] = 5e4
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = inject_sources(cube, ang, src, model, 0.027)
    dep = float((out.astype(np.float64) - cube.astype(np.float64)).sum()) / cube.shape[0]
    assert dep / (-0.05 * model.flux_unit) > 0.999


def test_the_flux_that_left_the_frames_is_measured_and_a_loss_warns():
    red = _red(_ds())
    chain = LINE + [Source(0.9, 13.0, 1e-9), Source(1.0, 193.0, -2e-3)]     # through 0, both signs
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        red.subtract_sources(chain)
    assert red.subtracted_kept == pytest.approx(1.0, abs=1e-5)
    faint = _red(_ds())
    with pytest.warns(RuntimeWarning, match="left the frames"):
        faint.subtract_sources([Source(0.8, 13.0, 1e-12)])
    assert faint.subtracted_kept < 0.5
    pr = PartitionedReducer({"n1": _red(_ds(3)), "n2": _red(_ds(4))}, max_workers=1, log=lambda s: None)
    pr.subtract_sources(LINE[:3])
    assert sorted(pr.subtracted_kept) == ["n1", "n2"]
    assert all(v == pytest.approx(1.0, abs=1e-5) for v in pr.subtracted_kept.values())
    assert re.fullmatch(r"n1 [01]\.\d{7}, n2 [01]\.\d{7}", _taken_text(pr))
    assert _taken_text(_red(_ds())) == ""                                    # nothing subtracted yet


def test_sources_added_at_load_are_an_outside_injection_and_a_subtraction_takes_them_out():
    ds = _ds()
    keep = ds.cube.copy()
    red = _red(ds)
    planets = [Source(0.9, 13.0, 3e-3), Source(1.1, 193.0, 2e-3)]
    ext = inject_sources(ds.cube, ds.angles, planets, red.model, PX, truenorth=red.truenorth,
                         angle_convention=red.angle_convention)
    assert red.inject_sources(planets + [Source(0.7, 50.0, 0.0)]) == 2
    assert np.array_equal(red.data.cube.view(np.uint32), np.asarray(ext, np.float32).view(np.uint32))
    assert np.array_equal(ds.cube, keep) and red.injected_kept == pytest.approx(1.0, abs=1e-5)
    assert red.data_edits["injected_sources"][:2] == [list(s.as_tuple()) for s in planets]
    assert np.array_equal(red.model_cube([Source(0.9, 45.0, 3e-3)]), _red(ds).model_cube([Source(0.9, 45.0, 3e-3)]))
    # the same list subtracted takes them out again, to float32 rounding
    red.subtract_sources(planets)
    scale = float(np.nanmax(np.abs(ds.cube)))
    assert np.nanmax(np.abs(red.data.cube - ds.cube)) < 4 * np.finfo(np.float32).eps * scale
    # every night of a partitioned reducer
    pr = PartitionedReducer({"n1": _red(_ds(3)), "n2": _red(_ds(4))}, max_workers=1, log=lambda s: None)
    assert pr.inject_sources(planets) == 2
    assert sorted(pr.injected_kept) == ["n1", "n2"] and pr.data_edits["injected_sources"] == [
        list(s.as_tuple()) for s in planets]
    assert re.fullmatch(r"n1 [01]\.\d{7}, n2 [01]\.\d{7}", _taken_text(pr, "injected_kept"))


def test_read_sources(tmp_path):
    p = tmp_path / "chain.txt"
    p.write_text("# rho pa contrast\n0.40 13 1.0e-6\n\n0.48, 193, 0.9e-6  # far side\n0.56 13 -2e-7 3\n")
    got = read_sources(str(p))
    assert [s.as_tuple() for s in got] == [(0.40, 13.0, 1.0e-6), (0.48, 193.0, 0.9e-6), (0.56, 13.0, -2e-7)]
    p.write_text("0.4 13\n")
    with pytest.raises(ValueError, match="need rho, pa and contrast"):
        read_sources(str(p))
