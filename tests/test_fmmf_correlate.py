"""The matched filter correlates (the GitHub agent's finding 13, Kevin Wagner 2026-11-03).

A matched filter correlates the image with the template, so a source shaped like the template
gives its own amplitude.  ``ndimage.convolve`` turns the template 180 deg first.  What must hold:

* A lopsided template, an ADI forward model with its self-subtraction lobe on one side, gives
  its source's amplitude when correlated and loses part of it when convolved: in
  ``mawet_peak_snr``'s map, the turned and placed filters, ``fmmf_map`` and the FMMF's S/N.
* Round kernels (the Gaussian, an azimuthally averaged PSF) give the same map to the bit either
  way, so every Mawet run, the verify, the S/N maps and the KLIP-FM curve are as they were.
* A new run's FMMF correlates, and a run saved before 2026-11-04 resumes convolving
  (``RunConfig.fmmf_correlate``).
"""
from __future__ import annotations

import types

import numpy as np
import pytest
from scipy import ndimage

from klip_tpe.fmmf import FMMFSNR, fm_kernel, fmmf_map
from klip_tpe.metrics import (Source, _PlacedFilter, _TurnedFilter, gaussian_kernel, kernel_from_profile,
                              mawet_peak_snr, source_xy)
from klip_tpe.runner import RunConfig, Runner

PX, FWHM, N = 0.05, 4.0, 81
C = (N - 1) / 2.0
SIZE = int(2 * np.ceil(2.5 * FWHM) + 1)


def _lopsided(x, y, amp=1.0, n=N, lobe=0.6):
    """A core and one negative lobe 1.6 FWHM along the azimuth through (x, y), the other side
    empty."""
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    az = np.arctan2(y - c, x - c)
    tx, ty = -np.sin(az), np.cos(az)
    s, d = FWHM / 2.3548, 1.6 * FWHM

    def g(x0, y0, a):
        return a * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s))
    return amp * (g(x, y, 1.0) - g(x + d * tx, y + d * ty, lobe))


def _src(pa=30.0, rho=1.2):
    x, y = source_xy([rho], [pa], PX, C, C)
    return int(round(x[0])), int(round(y[0]))


def test_round_kernels_give_the_same_map_to_the_bit():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((N, N))
    stamp = _lopsided(20, 20, n=41)                 # a lopsided PSF, averaged round
    for k in (gaussian_kernel(FWHM), gaussian_kernel(3.3), kernel_from_profile(stamp, FWHM)):
        assert np.array_equal(k, k[::-1, ::-1])
        assert np.array_equal(ndimage.correlate(img, k, mode="nearest"), ndimage.convolve(img, k, mode="nearest"))
        a = mawet_peak_snr(img, [0.8, 1.3], [40.0, 220.0], PX, FWHM, kernel=k, return_details=True)
        b = mawet_peak_snr(img, [0.8, 1.3], [40.0, 220.0], PX, FWHM, kernel=k, return_details=True, correlate=False)
        assert np.array_equal(a[0], b[0], equal_nan=True) and a[1] == b[1]


def test_a_lopsided_template_gives_its_amplitude_only_correlated():
    x0, y0 = _src()
    planet = _lopsided(x0, y0)
    m = fm_kernel(planet, x0, y0, SIZE)
    img = planet + 1e-3 * np.random.default_rng(1).standard_normal((N, N))
    rho = float(np.hypot(x0 - C, y0 - C)) * PX
    pa = float(np.degrees(np.arctan2(y0 - C, x0 - C))) - 90.0
    s1, d1 = mawet_peak_snr(img, [rho], [pa], PX, FWHM, kernel=[m], declip_nsig=None, return_details=True)
    s0, d0 = mawet_peak_snr(img, [rho], [pa], PX, FWHM, kernel=[m], declip_nsig=None, return_details=True,
                            correlate=False)
    assert d1[0]["peak"] == pytest.approx(1.0, abs=0.01) and (d1[0]["x"], d1[0]["y"]) == (x0, y0)
    assert d0[0]["peak"] < 0.95, "convolving turns the lobe to the other side"
    assert s1[0] > s0[0]
    # the turned and the placed filters follow the same convention
    az = np.arctan2(y0 - C, x0 - C)
    assert _TurnedFilter(img, m, az, {}).at(x0, y0, 0.0) == pytest.approx(1.0, abs=0.01)
    assert _TurnedFilter(img, m, az, {}, correlate=False).at(x0, y0, 0.0) < 0.95
    assert _PlacedFilter(img, lambda x, y: m, {}).at(x0, y0) == pytest.approx(1.0, abs=0.01)
    assert _PlacedFilter(img, lambda x, y: m, {}, correlate=False).at(x0, y0) < 0.95


def test_fmmf_map_gives_a_lopsided_source_its_amplitude():
    x0, y0 = _src(pa=0.0, rho=1.5)
    fm = _lopsided(x0, y0)
    src = [Source(1.5, 0.0, 1.0)]
    x1, y1 = _src(pa=120.0, rho=1.5)
    im = _lopsided(x1, y1)
    a1 = fmmf_map(im, fm, src, PX, FWHM, n_theta=180)["amplitude"][y1, x1]
    a0 = fmmf_map(im, fm, src, PX, FWHM, n_theta=180, correlate=False)["amplitude"][y1, x1]
    assert a1 == pytest.approx(1.0, rel=0.05) and a0 < 0.95 * a1     # the turn interpolates the template


def test_the_fmmf_scores_more_of_a_lopsided_source_correlated():
    x0, y0 = _src(pa=60.0)
    fm = _lopsided(x0, y0)
    rng = np.random.default_rng(2)
    img = 8.0 * fm + rng.standard_normal((N, N))
    rho = float(np.hypot(x0 - C, y0 - C)) * PX
    pa = float(np.degrees(np.arctan2(y0 - C, x0 - C))) - 90.0
    new, old = FMMFSNR(pxscale=PX, fwhm=FWHM), FMMFSNR(pxscale=PX, fwhm=FWHM, correlate=False)
    s1 = new.per_source(img, None, [rho], [pa], fm=fm)
    s0 = old.per_source(img, None, [rho], [pa], fm=fm)
    assert s1[0] > 1.1 * s0[0] > 0
    assert new.describe()["correlate"] is True and old.describe()["correlate"] is False


def test_a_new_run_correlates_and_a_run_saved_before_convolves():
    cfg = RunConfig(ann_edges=[8, 30])
    assert cfg.fmmf_correlate is True and RunConfig.from_dict(cfg.to_dict()).fmmf_correlate is True
    d = cfg.to_dict()
    d.pop("fmmf_correlate")
    old = RunConfig.from_dict(d)
    assert old.fmmf_correlate is False, "a run saved before 2026-11-04 keeps its filter"
    for c, want in ((cfg, True), (old, False)):
        m = FMMFSNR(pxscale=PX, fwhm=FWHM, correlate=not want)
        Runner._install_fmmf_turn(types.SimpleNamespace(objective=types.SimpleNamespace(metric=m), cfg=c))
        assert m.correlate is want
