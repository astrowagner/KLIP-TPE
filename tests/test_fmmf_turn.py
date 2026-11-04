"""The forward-modelled templates turned to each noise sample's azimuth (NEAR2 note,
2026-11-01).  An ADI forward model's self-subtraction lobes lie along the azimuth through the
source, so a noise aperture elsewhere on the ring is filtered with the template turned by its
azimuth from the source, the template a source there would have.  And fmmf_map, which turned
its templates the other way until then.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from klip_tpe.fmmf import FMMFSNR, fm_kernel, fmmf_map
from klip_tpe.metrics import Source, _TurnedFilter, mawet_peak_snr, source_xy, star_center, turn_kernel
from klip_tpe.runner import RunConfig, Runner

from conftest import build_synthetic_run

N, FWHM, PX = 101, 4.0, 0.05
CX, CY = star_center((N, N))
YY, XX = np.mgrid[0:N, 0:N]


def _planet(x, y, amp=1.0, n=N):
    """An ADI-like planet at (x, y): a core and two negative lobes along the azimuth, one deeper."""
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    az = np.arctan2(y - c, x - c)
    tx, ty = -np.sin(az), np.cos(az)
    s, d = FWHM / 2.3548, 1.6 * FWHM

    def g(x0, y0, a):
        return a * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s))

    return amp * (g(x, y, 1.0) - g(x + d * tx, y + d * ty, 0.6) - g(x - d * tx, y - d * ty, 0.3))


def _arcs(seed, smear_deg):
    """Unit noise smeared along the azimuth by ``smear_deg`` (KLIP residuals are), barely along
    the radius."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((80, 720))
    if smear_deg > 0:
        w = ndimage.gaussian_filter1d(w, smear_deg / 0.5, axis=1, mode="wrap")
    w = ndimage.gaussian_filter1d(w, 0.7, axis=0)
    w /= w.std()
    th = (np.degrees(np.arctan2(YY - CY, XX - CX)) % 360.0) / 0.5
    return ndimage.map_coordinates(w, [np.hypot(XX - CX, YY - CY), th], order=1, mode="wrap")


def _source(pa, rho=1.5):
    x, y = source_xy([rho], [pa], PX, CX, CY)
    return float(x[0]), float(y[0])


def test_the_unturned_template_reads_the_map():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((61, 61))
    k = rng.standard_normal((11, 11))
    k -= k.mean()
    k /= (k * k).sum()
    corr = ndimage.correlate(img, k, mode="nearest")
    conv = ndimage.convolve(img, k, mode="nearest")
    tf = _TurnedFilter(img, k, 0.3, {})
    old = _TurnedFilter(img, k, 0.3, {}, correlate=False)       # runs saved before 2026-11-04
    for x in (0, 1, 30, 60):
        for y in (0, 4, 30, 60):
            assert tf.at(x, y, 0.0) == pytest.approx(corr[y, x], abs=1e-12)
            assert old.at(x, y, 0.0) == pytest.approx(conv[y, x], abs=1e-12)
    assert np.allclose(turn_kernel(k, 2 * np.pi), k, atol=1e-12)


def test_a_turned_template_recovers_a_planet_at_any_azimuth():
    x0, y0 = round(CX + 30 * np.cos(0.35)), round(CY + 30 * np.sin(0.35))
    az0 = np.arctan2(y0 - CY, x0 - CX)
    km = fm_kernel(_planet(x0, y0), x0, y0, 21)
    ref = None
    for dphi in (0.0, 30.0, 45.0, 90.0, 150.0, -60.0):
        a = az0 + np.radians(dphi)
        ix, iy = int(round(CX + 30 * np.cos(a))), int(round(CY + 30 * np.sin(a)))
        im = _planet(ix, iy)
        got = _TurnedFilter(im, km, az0, {}).at(ix, iy, np.arctan2(iy - CY, ix - CX) - az0)
        ref = got if ref is None else ref
        assert got == pytest.approx(ref, rel=0.01), dphi
        if 45 <= abs(dphi) <= 90:                 # the unturned template misses the lobes
            assert ndimage.correlate(im, km, mode="nearest")[iy, ix] < 0.85 * ref


@pytest.mark.parametrize("smear", [0.0, 15.0])
def test_the_ring_scatter_is_the_noise_of_the_source_s_own_filter(smear):
    """The true scatter of the source's filtered value, over noise draws, against the ring's
    estimate: with the template turned they agree (the band floor lifts both by a tenth or
    so).  Unturned, against residuals smeared along the azimuth, the ring saw the lobes across
    the arcs, and its scatter came out far above the source's."""
    pa = 20.0
    xs, ys = _source(pa)
    km = fm_kernel(_planet(xs, ys), xs, ys, 21)
    pk = [(int(round(xs)), int(round(ys)))]
    true, turned, flat = [], [], []
    for seed in range(24):
        im = _arcs(seed, smear)
        true.append(ndimage.correlate(im, km, mode="nearest")[pk[0][1], pk[0][0]])
        for t, out in ((True, turned), (False, flat)):
            _, d = mawet_peak_snr(im, [1.5], [pa], PX, FWHM, kernel=[km], declip_nsig=None, turn=[t], peak_xy=pk,
                                  return_details=True)
            out.append(d[0]["sigma"])
    st = float(np.std(true, ddof=1))
    assert 0.8 < np.mean(turned) / st < 1.35
    if smear:
        assert np.mean(flat) / st > 1.6
    else:
        assert np.mean(flat) == pytest.approx(np.mean(turned), rel=0.1)


def test_the_fmmf_metric_turns_its_templates_and_not_its_fallbacks():
    xs, ys = _source(20.0)
    fm = _planet(xs, ys)
    img = 2.0 * fm + 0.3 * _arcs(1, 15.0)
    on = FMMFSNR(pxscale=PX, fwhm=FWHM).per_source(img, None, [1.5], [20.0], fm=fm)[0]
    off = FMMFSNR(pxscale=PX, fwhm=FWHM, turn=False).per_source(img, None, [1.5], [20.0], fm=fm)[0]
    km = fm_kernel(fm, xs, ys, int(2 * np.ceil(2.5 * FWHM) + 1))
    assert off == pytest.approx(float(mawet_peak_snr(img, [1.5], [20.0], PX, FWHM, kernel=[km])[0]), rel=1e-12)
    assert on > 1.3 * off, "the arcs no longer swell the ring's scatter"
    # without a forward model the fallback kernel is round, and nothing is turned
    a = FMMFSNR(pxscale=PX, fwhm=FWHM).per_source(img, None, [1.5], [20.0], fm=None)[0]
    b = FMMFSNR(pxscale=PX, fwhm=FWHM, turn=False).per_source(img, None, [1.5], [20.0], fm=None)[0]
    assert a == b
    assert FMMFSNR(pxscale=PX, fwhm=FWHM).describe()["turn"] is True


def test_a_run_turns_them_and_a_saved_run_keeps_the_one_map(tmp_path):
    assert RunConfig().fmmf_turn is True
    d = RunConfig().to_dict()
    d.pop("fmmf_turn")
    assert RunConfig.from_dict(d).fmmf_turn is False
    red, space, obj, samp, cfg = build_synthetic_run(fmmf_turn=False)
    from klip_tpe.metrics import Objective
    obj = Objective(FMMFSNR(pxscale=red.pxscale, fwhm=red.fwhm), clean_subtract=True)
    Runner(red, space, obj, samp, cfg, str(tmp_path / "off"), log=lambda s: None)
    assert obj.metric.turn is False
    red, space, _, samp, cfg = build_synthetic_run()
    obj = Objective(FMMFSNR(pxscale=red.pxscale, fwhm=red.fwhm, turn=False), clean_subtract=True)
    Runner(red, space, obj, samp, cfg, str(tmp_path / "on"), log=lambda s: None)
    assert obj.metric.turn is True


def test_fmmf_map_turns_its_template_with_the_azimuth():
    """Until 2026-11-01 both turns went the other way, which mirrored the template but at its
    own azimuth: a planet 30 to 60 deg from the test source kept about 0.7 of its amplitude."""
    n = 121
    c = (n - 1) / 2.0
    x0, y0 = source_xy([1.5], [0.0], PX, c, c)
    fm = _planet(x0[0], y0[0], n=n)
    got = []
    for pa in (0.0, 30.0, 45.0, 90.0, 135.0, 200.0, 300.0):
        x, y = source_xy([1.5], [pa], PX, c, c)
        amp = fmmf_map(_planet(x[0], y[0], n=n), fm, [Source(1.5, 0.0, 1.0)], PX, FWHM, n_theta=180)["amplitude"]
        got.append(amp[int(round(y[0])), int(round(x[0]))])
    assert np.all(np.isfinite(got)), "PA 90 lies at +180 deg of azimuth, in the last bin"
    assert min(got) > 0.9 and max(got) - min(got) < 0.05, got
