"""Frames rotate with OpenCV when it is precise enough, with scipy otherwise.

From Vivek Vijayakumar's question of 2026-11-10 and Kevin Wagner's answer to it.  OpenCV's
bilinear ``warpAffine`` is about 5 times faster than ``scipy.ndimage.map_coordinates`` on a
150 x 150 frame.  OpenCV 5 samples where scipy does, to float32 rounding.  OpenCV 4 rounds the
sample positions to 1/32 px.  What must hold:

* ``KLIP_TPE_ROTATE`` / :func:`set_rotation_backend` choose ``auto`` (OpenCV when it passes the
  precision check, else scipy), ``scipy`` or ``opencv``.  OpenCV asked for and not usable falls
  back to scipy with a warning.  An imprecise OpenCV fails the check.
* With OpenCV, a rotation gives scipy's numbers to float32 rounding, with or without NaN
  pixels, for one angle or many, and a whole reduction agrees.
* A new run records the library it rotated with, and a resume keeps it.  A run recorded before
  the field resumes with scipy.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from klip_tpe import klip
from klip_tpe.klip import rotate_ccw, rotate_ccw_many, rotation_backend, set_rotation_backend

opencv_ok = klip._opencv_check()[0] is not None
needs_opencv = pytest.mark.skipif(not opencv_ok, reason="no precise OpenCV (OpenCV 5) here")


@pytest.fixture
def backend():
    """Switch the rotation library in a test, back to scipy (the suite's) after it."""
    yield set_rotation_backend
    set_rotation_backend("scipy")


def _scene(n=81, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.zeros((n, n))
    for _ in range(25):
        x0, y0 = rng.uniform(10, n - 10, 2)
        img += rng.uniform(0.5, 1.0) * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / 2.0 ** 2)
    return img


def test_the_choice_and_its_fallbacks(backend, monkeypatch):
    assert rotation_backend() == "scipy", "the suite rotates with scipy"
    assert backend("scipy") == "scipy"
    with pytest.raises(ValueError, match="one of auto, scipy, opencv"):
        backend("fast")
    # OpenCV asked for and absent: scipy, with a warning
    monkeypatch.setitem(klip._ROT, "checked", True)
    monkeypatch.setitem(klip._ROT, "cv2", None)
    monkeypatch.setitem(klip._ROT, "why", "OpenCV is not installed (ImportError)")
    with pytest.warns(RuntimeWarning, match="not installed"):
        assert backend("opencv") == "scipy"
    assert backend("auto") == "scipy"
    import os
    assert os.environ["KLIP_TPE_ROTATE"] == "scipy"


@needs_opencv
def test_an_imprecise_opencv_fails_the_check(monkeypatch):
    real = klip._cv_warp

    def rounded(cv2, img, ca, sa):        # samples a little off where scipy samples, as OpenCV 4's
        return real(cv2, img, ca + 1.0 / 2048, sa)
    monkeypatch.setitem(klip._ROT, "checked", False)
    monkeypatch.setattr(klip, "_cv_warp", rounded)
    cv2, why = klip._opencv_check()
    assert cv2 is None and "rounds the sample positions" in why
    monkeypatch.setitem(klip._ROT, "checked", False)
    monkeypatch.setattr(klip, "_cv_warp", real)
    assert klip._opencv_check()[0] is not None


@needs_opencv
def test_opencv_rotates_as_scipy_does(backend):
    img = _scene()
    holes = img.copy()
    holes[30:33, 40:44] = np.nan                       # a masked ghost
    holes[:, :3] = np.nan                              # a registration edge
    want = {}
    backend("scipy")
    for ang in (7.3, -33.0, 151.2):
        want[ang] = (rotate_ccw(img, ang), rotate_ccw(holes, ang), rotate_ccw(holes, ang, nan_mode="propagate"),
                     rotate_ccw(img, ang, cval=0.0))
    many_s = rotate_ccw_many(img, [0.0, 7.3, -33.0])
    assert backend("opencv") == "opencv"
    peak = float(np.nanmax(img))
    for ang, (a, b, c, d) in want.items():
        for ref, got in ((a, rotate_ccw(img, ang)), (b, rotate_ccw(holes, ang)),
                         (c, rotate_ccw(holes, ang, nan_mode="propagate")), (d, rotate_ccw(img, ang, cval=0.0))):
            assert got.dtype == np.float64 and got.shape == ref.shape
            assert np.array_equal(np.isnan(got), np.isnan(ref)), ang
            assert np.nanmax(np.abs(got - ref)) < 1e-4 * peak, ang
    many_c = rotate_ccw_many(img, [0.0, 7.3, -33.0])
    assert np.array_equal(many_c[0], img) and np.array_equal(np.isnan(many_c), np.isnan(many_s))
    assert np.nanmax(np.abs(many_c - many_s)) < 1e-4 * peak
    assert np.array_equal(many_c[1], rotate_ccw(img, 7.3), equal_nan=True)


@needs_opencv
def test_a_reduction_agrees_between_the_libraries(backend):
    from klip_tpe.injection import GaussianPSF
    from klip_tpe.metrics import Source
    from klip_tpe.reducer import KLIPReducer, ReductionRequest
    from klip_tpe.synthetic import synthetic_klip_dataset
    ds = synthetic_klip_dataset(nframes=40, size=61, pa_span=40, seed=3)
    red = KLIPReducer(ds, pxscale=0.05, lam_m=3.8e-6, diam_m=8.2, injection_model=GaussianPSF(3.0, star_flux=1e4),
                      fwhm_px=3.0, outrad_cap=28)
    req = ReductionRequest(dict(bin=2, filter=5, k_klip=5, n_ang=2, inrad=6, outrad=20, angsep=0.5, anglemax=90.0,
                                corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0), [Source(0.6, 40.0, 3e-3)])
    imgs = {}
    for name in ("scipy", "opencv"):
        assert backend(name) == name
        imgs[name] = red.reduce(req).image
    a, b = imgs["scipy"], imgs["opencv"]
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    m = np.isfinite(a)
    assert np.sqrt(np.mean((a[m] - b[m]) ** 2)) < 1e-4 * np.sqrt(np.mean(a[m] ** 2))


def test_a_run_records_its_library_and_a_resume_keeps_it(backend):
    from klip_tpe.runner import RunConfig, Runner
    logs = []
    new = types.SimpleNamespace(cfg=RunConfig(ann_edges=[5, 10]), log=logs.append)
    Runner._install_rotation(new)
    assert new.cfg.rotation == rotation_backend() == "scipy"
    assert RunConfig.from_dict(new.cfg.to_dict()).rotation == "scipy"
    # a run recorded before the field: scipy, whatever this process would choose
    old = types.SimpleNamespace(cfg=RunConfig(ann_edges=[5, 10]), log=logs.append)
    if opencv_ok:
        backend("opencv")
    Runner._install_rotation(old, from_checkpoint=True)
    assert old.cfg.rotation == "scipy" and rotation_backend() == "scipy"
    if opencv_ok:
        # a run made with OpenCV resumes with it
        rec = types.SimpleNamespace(cfg=RunConfig(ann_edges=[5, 10], rotation="opencv"), log=logs.append)
        Runner._install_rotation(rec, from_checkpoint=True)
        assert rotation_backend() == "opencv"
    # one recorded with a library this machine lacks: logged, and scipy
    rec = types.SimpleNamespace(cfg=RunConfig(ann_edges=[5, 10], rotation="opencv"), log=logs.append)
    import pytest as _pt
    with _pt.MonkeyPatch.context() as mp:
        mp.setitem(klip._ROT, "checked", True)
        mp.setitem(klip._ROT, "cv2", None)
        mp.setitem(klip._ROT, "why", "OpenCV is not installed (ImportError)")
        backend("scipy")
        with _pt.warns(RuntimeWarning):
            Runner._install_rotation(rec, from_checkpoint=True)
    assert rotation_backend() == "scipy" and rec.cfg.rotation == "opencv"
    assert any("not usable here" in m for m in logs)
