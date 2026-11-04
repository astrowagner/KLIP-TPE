"""One declip and one matched filter per image and kernel (metrics.PREPARED_CACHE_SIZE).

A trial's clean image is scored once for each of its draws, and the sidelobe rule measures
each source, and each fallback, on the same image again.  The cache keeps each image's declip
and convolutions by its bytes, so the repeats cost nothing and every number stays the same.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from klip_tpe import metrics
from klip_tpe.metrics import (MawetPeakSNR, Objective, Sidelobes, Source, mawet_peak_snr, source_xy,
                              star_center)

PX, FWHM = 0.05, 4.0
KD = int(2 * np.ceil(1.5 * FWHM) + 1)                # the matched filter's width (13 px)


def _frame(seed, n=101, srcs=()):
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, 1.0, (n, n))
    cx, cy = star_center(img.shape)
    yy, xx = np.mgrid[0:n, 0:n]
    s = FWHM / 2.3548
    for rho, th, amp in srcs:
        x, y = source_xy([rho], [th], PX, cx, cy, "pa")
        img += amp * np.exp(-((xx - x[0]) ** 2 + (yy - y[0]) ** 2) / (2 * s * s))
    img[:3] = np.nan                                  # a NaN edge, as reductions have
    return img


@pytest.fixture(autouse=True)
def _fresh():
    metrics.clear_prepared_cache()
    yield
    metrics.clear_prepared_cache()


def _trial():
    draws = [[Source(0.6, 40.0 + 70 * j, 1.0), Source(0.9, 220.0 + 70 * j, 1.0)] for j in range(4)]
    clean = _frame(1)
    injs = [_frame(1, srcs=[(s.rho, s.theta, 25.0) for s in d]) for d in draws]
    obj = Objective(MawetPeakSNR(pxscale=PX, fwhm=FWHM))
    obj.sidelobes = Sidelobes(rotation_deg=150.0)
    return obj, clean, injs, draws


def _score(obj, clean, injs, draws):
    return [obj.score_search(inj, d, clean) for inj, d in zip(injs, draws)]


def test_the_cache_changes_no_number(monkeypatch):
    obj, clean, injs, draws = _trial()
    monkeypatch.setattr(metrics, "PREPARED_CACHE_SIZE", 0)
    off = _score(obj, clean, injs, draws)
    monkeypatch.setattr(metrics, "PREPARED_CACHE_SIZE", 8)
    on = _score(obj, clean, injs, draws)
    again = _score(obj, clean, injs, draws)           # every image prepared already
    assert any(np.isfinite(r.score) for r in off)
    for a, b, c in zip(off, on, again):
        for r in (b, c):
            np.testing.assert_equal(r.score, a.score)
            np.testing.assert_array_equal(r.per_source, a.per_source)
            np.testing.assert_array_equal(r.raw_per_source, a.raw_per_source)
            np.testing.assert_array_equal(r.clean_per_source, a.clean_per_source)
    # and the details of a direct call, a kernel per source included
    img = injs[0]
    ks = [metrics.gaussian_kernel(FWHM), metrics.gaussian_kernel(1.3 * FWHM)]
    monkeypatch.setattr(metrics, "PREPARED_CACHE_SIZE", 0)
    s0, d0 = mawet_peak_snr(img, [0.6, 0.9], [40.0, 220.0], PX, FWHM, kernel=ks, return_details=True)
    monkeypatch.setattr(metrics, "PREPARED_CACHE_SIZE", 8)
    s1, d1 = mawet_peak_snr(img, [0.6, 0.9], [40.0, 220.0], PX, FWHM, kernel=ks, return_details=True)
    np.testing.assert_array_equal(s1, s0)
    assert d1 == d0


def test_a_trial_filters_each_image_once(monkeypatch):
    """Four draws of two sources under the sidelobe rule: the clean image and each injected
    image go through the matched filter once each."""
    obj, clean, injs, draws = _trial()
    calls = []
    real = ndimage.correlate                           # the matched filter correlates

    def counting(a, k, *args, **kw):
        if np.shape(k) == (KD, KD):                    # not the declip's 3x3 neighbour count
            calls.append(1)
        return real(a, k, *args, **kw)
    monkeypatch.setattr(ndimage, "correlate", counting)
    monkeypatch.setattr(metrics, "PREPARED_CACHE_SIZE", 0)
    _score(obj, clean, injs, draws)
    n_off = len(calls)
    calls.clear()
    monkeypatch.setattr(metrics, "PREPARED_CACHE_SIZE", 8)
    _score(obj, clean, injs, draws)
    assert len(calls) == 1 + len(injs), "the clean image once, each injected image once"
    assert n_off >= 4 * len(calls), f"{n_off} convolutions without the cache"


def test_an_image_changed_in_place_is_prepared_again():
    img = _frame(2, srcs=[(0.8, 30.0, 20.0)])
    s1 = mawet_peak_snr(img, [0.8], [30.0], PX, FWHM)
    img[60:70, 40:60] += 5.0                            # the same array, other bytes
    s2 = mawet_peak_snr(img, [0.8], [30.0], PX, FWHM)
    metrics.clear_prepared_cache()
    s3 = mawet_peak_snr(img, [0.8], [30.0], PX, FWHM)
    assert s2[0] == s3[0] != s1[0]


def test_the_cache_keeps_its_own_read_only_copies():
    img = _frame(3)
    prep = metrics._prepare(img, None)                 # no declip: the image as given
    assert prep["img"] is not img and np.array_equal(prep["img"], img, equal_nan=True)
    with pytest.raises(ValueError):
        prep["imz"][5, 5] = 1.0
    img[5, 5] = 99.0                                   # the caller's array stays writable
    assert metrics._prepare(img, None) is not prep
