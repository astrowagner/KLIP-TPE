"""The despike that spares a source's core, and the noise read on the clean reduction.

After the FLIP note of 2026-11-21, "MawetPeakSNR's de-spiking and noise floor distort the search
objective".  The IDL predecessor's despike replaces every pixel more than 6 robust sigma from its
3x3 median, which at 2.3 px per FWHM takes a source's core whenever the pixel-to-pixel scatter is
far below the noise that sets the S/N.  ``declip_isolated`` replaces a pixel only when its neighbors
do not stand out with it.  A PSF whose matched-filtered wings reach past the keep-outs puts each
source into its own noise, so its S/N grows ever more slowly.  ``noise_image="clean"`` reads the
injected image's noise on the clean reduction, which holds no injected source.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from klip_tpe.metrics import (RING_RULE_KEYS, MawetPeakSNR, Objective, declip, kernel_from_profile,
                              matched_filter, mawet_peak_snr, ring_rules, rule_kwargs)
from klip_tpe.runner import Runner

N, C, PX, FW = 121, 60.0, 0.05, 2.3
YY, XX = np.mgrid[0:N, 0:N]


def _gauss(x, y, fwhm=FW):
    """Return a Gaussian of unit peak at ``(x, y)`` on the test grid."""
    return np.exp(-0.5 * ((XX - x) ** 2 + (YY - y) ** 2) / (fwhm / 2.3548) ** 2)


def _field(seed, white=0.05):
    """Return smooth speckles of unit sigma at the PSF's scale, with a little white noise on top."""
    rng = np.random.default_rng(seed)
    g = np.exp(-0.5 * (np.hypot(*np.mgrid[-5:6, -5:6]) / (FW / 2.3548)) ** 2)
    sp = ndimage.convolve(rng.normal(size=(N, N)), g, mode="wrap")
    return sp / sp.std() + white * rng.normal(size=(N, N))


def _xy(rho_px, pa):
    """Return the pixel of a source at ``rho_px`` and position angle ``pa``."""
    th = np.deg2rad(pa + 90.0)
    return C + rho_px * np.cos(th), C + rho_px * np.sin(th)


def test_the_isolated_despike_spares_a_sampled_core_and_still_takes_spikes():
    img = _field(1)
    rob = 1.4826 * np.median(np.abs(img - ndimage.median_filter(img, 3)))
    img[20, 90] += 20 * rob                               # a hot pixel
    img[100, 30] += 15 * rob
    img[100, 31] += 15 * rob                              # a hit on two pixels
    img[30, 25] -= 20 * rob                               # a cold pixel
    near = np.zeros((N, N), bool)
    for x, y in ((80.0, 60.0), (40.4, 70.6)):             # on a pixel and between pixels
        img = img + 12.0 * _gauss(x, y)
        near |= np.hypot(XX - x, YY - y) <= FW
    old = declip(img, 6.0) != img
    new = declip(img, 6.0, isolated=True) != img
    assert old[near].any()                                # the IDL predecessor's rule takes the cores
    assert not new[near].any()
    for y, x in ((20, 90), (100, 30), (100, 31), (30, 25)):
        assert old[y, x] and new[y, x]


def test_the_metric_keeps_a_source_s_snr_with_the_isolated_despike():
    K = kernel_from_profile(_gauss(C, C)[45:76, 45:76], FW)
    base = _field(2)
    s_off, s_iso, s_idl = [], [], []
    for pa in np.arange(8) * 45.0 + 10.0:
        img = base + 12.0 * _gauss(*_xy(16.0, pa))
        for out, kw in ((s_off, dict(declip_nsig=None)), (s_iso, dict(declip_isolated=True)), (s_idl, {})):
            m = MawetPeakSNR(pxscale=PX, fwhm=FW, kernel_fn=lambda rho: K, **kw)
            out.append(float(m.per_source(img, None, [16.0 * PX], [pa])[0]))
    # a few lone pixels of the noise still go, and the sources' S/N stays within 1 %
    np.testing.assert_allclose(s_iso, s_off, rtol=0.01)
    assert np.median(s_idl) < 0.9 * np.median(s_off)


def _wings(x, y):
    """Return a PSF whose matched-filtered wings reach past the keep-outs: a core and a broad halo."""
    return _gauss(x, y) + 0.15 * _gauss(x, y, 7.0)


def test_noise_read_on_the_clean_image_keeps_a_bright_source_s_snr_linear():
    K = kernel_from_profile(_wings(C, C)[45:76, 45:76], FW)
    clean = _field(3)
    own, cln = {}, {}
    for a in (16.0, 64.0):
        o, c = [], []
        for pa in np.arange(8) * 45.0 + 10.0:
            inj = clean + a * _wings(*_xy(14.0, pa))
            o.append(mawet_peak_snr(inj, [14.0 * PX], [pa], PX, FW, kernel=K, declip_nsig=None)[0])
            c.append(mawet_peak_snr(inj, [14.0 * PX], [pa], PX, FW, kernel=K, declip_nsig=None, noise_img=clean)[0])
        own[a], cln[a] = float(np.median(o)), float(np.median(c))
    assert cln[64.0] / cln[16.0] == pytest.approx(4.0, rel=0.05)
    assert own[64.0] / own[16.0] < 2.0                    # the source in its own noise


def test_the_metric_reads_its_noise_on_the_clean_image_and_the_clean_term_on_its_own():
    clean = _field(4)
    inj = clean + 10.0 * _wings(*_xy(14.0, 40.0)) + 10.0 * _wings(*_xy(24.0, 200.0))
    rho, pa = [14.0 * PX, 24.0 * PX], [40.0, 200.0]
    m = MawetPeakSNR(pxscale=PX, fwhm=FW, noise_image="clean")
    assert m.wants_clean and not m.needs_clean and m.describe()["noise_image"] == "clean"
    want = mawet_peak_snr(inj, rho, pa, PX, FW, noise_img=clean)
    np.testing.assert_allclose(m.per_source(inj, clean, rho, pa), want)
    own = MawetPeakSNR(pxscale=PX, fwhm=FW)
    assert "noise_image" not in own.describe() and not own.wants_clean
    np.testing.assert_allclose(m.per_source(inj, None, rho, pa), own.per_source(inj, None, rho, pa))
    # the search's clean term reads the clean image's own noise, and is still subtracted
    src = [SimpleNamespace(rho=r, theta=t) for r, t in zip(rho, pa)]
    res = Objective(m, clean_subtract=True).score_search(inj, src, clean)
    assert Objective(m, clean_subtract=True).effective_clean_subtract
    np.testing.assert_allclose(res.raw_per_source, want)
    np.testing.assert_allclose(res.clean_per_source, own.per_source(clean, None, rho, pa))
    with pytest.raises(ValueError):
        MawetPeakSNR(pxscale=PX, fwhm=FW, noise_image="ring")
    with pytest.raises(ValueError):
        mawet_peak_snr(inj, rho, pa, PX, FW, noise_img=clean[:-1])
    with pytest.raises(ValueError):
        mawet_peak_snr(inj, rho, pa, PX, FW, noise_img=clean, mfmap=np.zeros_like(inj))


def test_the_runner_reduces_the_clean_image_for_a_metric_that_reads_its_noise_there():
    for metric, want in ((MawetPeakSNR(pxscale=PX, fwhm=FW, noise_image="clean"), True),
                         (MawetPeakSNR(pxscale=PX, fwhm=FW), False)):
        stub = SimpleNamespace(objective=SimpleNamespace(metric=metric, needs_clean=False))
        stub._metric_reads_clean = lambda s=stub: Runner._metric_reads_clean(s)
        assert Runner._needs_clean(stub, raw_only=True) is want


def test_the_rules_and_records_of_other_runs_stay_as_they_were():
    assert "declip_isolated" in RING_RULE_KEYS
    plain = MawetPeakSNR(pxscale=PX, fwhm=FW)
    assert "declip_isolated" not in ring_rules(plain) and "declip_isolated" not in plain.describe()
    iso = MawetPeakSNR(pxscale=PX, fwhm=FW, declip_isolated=True)
    rules = ring_rules(iso)
    assert rules["declip_isolated"] is True and iso.describe()["declip_isolated"] is True
    assert rule_kwargs(rules, 0.5, (N, N))["declip_isolated"] is True
    img = _field(5) + 12.0 * _gauss(80.0, 60.0)
    k = rules["kernel_at"](0.5)
    np.testing.assert_allclose(matched_filter(img, FW, rules, 0.5),
                               ndimage.correlate(declip(img, 6.0, isolated=True), k, mode="nearest"))
    assert not np.allclose(matched_filter(img, FW, ring_rules(plain), 0.5),
                           matched_filter(img, FW, rules, 0.5))


def test_a_run_reduces_the_clean_image_for_the_noise_without_a_clean_term(tmp_path):
    import json
    import os
    from conftest import build_synthetic_run
    from klip_tpe import ValidationConfig

    red, space, _, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=[8], n_init=4,
                                                   validation=ValidationConfig(n_top=1, n_valid=1))
    seen = []

    class Counting(MawetPeakSNR):
        """The metric, noting whether each injected image came with the clean one."""

        def per_source(self, img_inj, img_clean, rho, theta, why=None, sidelobes=None):
            """Note whether the clean image came, and measure."""
            seen.append(img_clean is not None)
            return super().per_source(img_inj, img_clean, rho, theta, why=why, sidelobes=sidelobes)

    obj = Objective(Counting(pxscale=red.pxscale, fwhm=red.fwhm, noise_image="clean", declip_isolated=True),
                    clean_subtract=False)
    d = str(tmp_path / "run")
    Runner(red, space, obj, samp, cfg, d, log=lambda *a: None).run()
    assert sum(seen) >= 8                                 # every search evaluation, and the validation
    md = json.load(open(os.path.join(d, "run_setup.json")))["objective"]["metric"]
    assert md["noise_image"] == "clean" and md["declip_isolated"] is True
