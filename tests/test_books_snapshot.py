"""The small things of NEAR2's s7 note (2026-11-02), section 4, beside the viewer's box
(``tests/test_winsize.py``).

* The books' snapshot keeps the matched filter as its kernels, without the metric and the
  reducer behind it: a run with a measured kernel had drawn its books on the main thread since
  the ring rules joined the snapshot, the pickle failing on the PSF library's throughput.
* The PSF library's throughput pickles.
* ``throughput_fit`` fits order 0 when every sample sits at one radius, with no RankWarning.
"""
from __future__ import annotations

import pickle
import types
import warnings

import numpy as np
import pytest

from klip_tpe import books
from klip_tpe.instruments import near
from klip_tpe.metrics import MawetPeakSNR, gaussian_kernel, ring_rules
from klip_tpe.products import throughput_fit


class _Holder:
    """A stand-in for the reducer: a kernel, beside something that does not pickle."""

    def __init__(self):
        self.frames = np.zeros((50, 64, 64), np.float32)
        self.lock = lambda: None                    # a closure: what stopped the pickle

    def matched_filter_kernel(self, rho_as):
        return gaussian_kernel(3.0 + float(rho_as))


def test_the_books_snapshot_leaves_the_reducer_behind():
    holder = _Holder()
    metric = MawetPeakSNR(pxscale=0.05, fwhm=4.0, kernel_fn=holder.matched_filter_kernel)
    rules = ring_rules(metric)
    ad = types.SimpleNamespace(known=[(0.8, 30.0), (1.1, 200.0)], pxscale=0.05, fwhm=4.0, inrad=10.0, outrad=30.0,
                               extra={"rules": rules, "draws": [1, 2]})
    with pytest.raises(Exception):
        pickle.dumps(ad)
    snap = books._snapshot(ad)
    blob = pickle.dumps(snap)
    assert len(blob) < 200_000 and holder.frames.nbytes > 600_000
    back = pickle.loads(blob)
    ka = back.extra["rules"]["kernel_at"]
    # exact where the drawings ask (the known companions and their median), and in the annulus
    for rho in (0.8, 1.1, 0.95):
        assert np.array_equal(ka(rho), metric.kernel(rho))
    assert np.array_equal(ka(0.5), metric.kernel(0.5)) and np.array_equal(ka(1.5), metric.kernel(1.5))
    # the rest as it was, and the live snapshot untouched
    assert back.extra["draws"] == [1, 2] and ad.extra["rules"]["kernel_at"] is rules["kernel_at"]
    assert books._snapshot(types.SimpleNamespace(extra={})) is not None


def test_the_psf_library_throughput_pickles(tmp_path):
    (tmp_path / "throughput_merged.csv").write_text("sep,thr\n0.2,0.19\n0.5,0.51\n1.0,0.81\n2.0,0.95\n")
    fn = near._throughput_table(str(tmp_path))
    r = np.linspace(0.0, 3.0, 31)
    back = pickle.loads(pickle.dumps(fn))
    assert np.array_equal(back(r), fn(r))
    assert fn(1.0) == pytest.approx(0.81) and fn(3.0) == 1.0 and fn(0.0) == pytest.approx(0.19)
    assert near._throughput_table(str(tmp_path / "none")) is None


def test_throughput_fit_at_one_radius_is_order_zero():
    k = np.array([2.0, 2.2, 1.8, 2.1, 1.9])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = throughput_fit(np.full(5, 0.75), k, np.array([0.5, 0.75, 1.0]))
    assert np.allclose(out, np.exp(np.mean(np.log(k))))
    # two radii: a line at most, still no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out2 = throughput_fit(np.array([0.5, 0.5, 1.0, 1.0, 1.0]), np.array([1.0, 1.0, 2.0, 2.0, 2.0]),
                              np.array([0.5, 1.0]))
    assert np.allclose(out2, [1.0, 2.0])
