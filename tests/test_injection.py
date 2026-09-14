import numpy as np
import pytest

from klip_tpe.injection import (GaussianPSF, LibraryPSF, TemplatePSF, add_stamp, inject_sources,
                                shift_bilinear)
from klip_tpe.klip import derotate
from klip_tpe.metrics import Source, mawet_peak_snr, source_xy, star_center

PX = 0.05
FWHM = 4.0


def _centroid(img):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    s = img.sum()
    return (img * xx).sum() / s, (img * yy).sum() / s


# ----------------------------------------------------------------------------
# stamps
# ----------------------------------------------------------------------------
def test_shift_bilinear_moves_centroid():
    st = GaussianPSF(3.0, size=15)._stamp
    sh = shift_bilinear(st, 0.4, -0.3)
    cx0, cy0 = _centroid(st)
    cx1, cy1 = _centroid(sh)
    assert cx1 - cx0 == pytest.approx(0.4, abs=0.01) and cy1 - cy0 == pytest.approx(-0.3, abs=0.01)
    assert sh.sum() == pytest.approx(st.sum(), rel=1e-3)


def test_add_stamp_subpixel_centroid_and_clipping():
    frame = np.zeros((31, 31))
    st = GaussianPSF(3.0, size=11)._stamp
    add_stamp(frame, st, (5.0, 5.0), 14.3, 9.6)
    cx, cy = _centroid(frame)
    assert cx == pytest.approx(14.3, abs=0.01) and cy == pytest.approx(9.6, abs=0.01)
    assert frame.sum() == pytest.approx(1.0, rel=1e-3)
    # adds (does not overwrite)
    add_stamp(frame, st, (5.0, 5.0), 14.3, 9.6)
    assert frame.sum() == pytest.approx(2.0, rel=1e-3)
    # partially outside: clipped, no exception, flux reduced
    f2 = np.zeros((31, 31))
    add_stamp(f2, st, (5.0, 5.0), 1.0, 29.0)
    assert 0.2 < f2.sum() < 0.9
    # fully outside: no-op
    f3 = np.zeros((31, 31))
    add_stamp(f3, st, (5.0, 5.0), -20.0, 5.0)
    assert f3.sum() == 0.0
    # non-integer stamp centre (even-sized stamp)
    f4 = np.zeros((31, 31))
    even = np.ones((4, 4)) / 16.0
    add_stamp(f4, even, (1.5, 1.5), 10.0, 20.0)
    assert _centroid(f4) == pytest.approx((10.0, 20.0), abs=1e-9)


def test_gaussian_psf_unit_flux_and_size():
    g = GaussianPSF(FWHM, star_flux=1e5, size=40)
    st, (cx, cy), ok = g.stamp(1.0)
    assert ok and st.shape == (41, 41) and (cx, cy) == (20.0, 20.0)   # size forced odd
    assert st.sum() == pytest.approx(1.0)
    assert g.flux_unit == 1e5 and g.throughput(0.5) == 1.0
    assert st[20, 22] / st[20, 20] == pytest.approx(0.5, abs=1e-3)
    assert g.describe()["name"] == "gaussian"


def test_template_psf_normalisation():
    yy, xx = np.mgrid[0:31, 0:31]
    t = 7.0 * np.exp(-0.5 * ((xx - 15) ** 2 + (yy - 15) ** 2) / 2.0 ** 2)
    t[0, 0] = np.nan
    m = TemplatePSF(t)
    st, c, ok = m.stamp(0.7)
    assert ok and c == (15.0, 15.0)
    assert st.sum() == pytest.approx(1.0)
    assert m.flux_unit == pytest.approx(np.nansum(t)) and m.template_flux == m.flux_unit
    # encircled-energy normalisation: unit flux inside the aperture, > 1 in total
    m2 = TemplatePSF(t, ee_radius_px=3.0, star_flux=123.0, throughput_fn=lambda r: 0.5)
    st2, _, _ = m2.stamp(0.7)
    inside = np.hypot(xx - 15, yy - 15) <= 3.0
    assert st2[inside].sum() == pytest.approx(1.0)
    assert st2.sum() > 1.0
    assert m2.flux_unit == 123.0 and m2.throughput(0.7) == 0.5
    # zero template does not divide by zero
    z = TemplatePSF(np.zeros((5, 5)))
    assert np.all(z.stamp(1.0)[0] == 0)


def test_library_psf_interpolation_and_ok_flag():
    yy, xx = np.mgrid[0:21, 0:21]
    s0 = np.exp(-0.5 * ((xx - 10) ** 2 + (yy - 10) ** 2) / 1.5 ** 2)
    s1 = np.exp(-0.5 * ((xx - 10) ** 2 + (yy - 10) ** 2) / 3.0 ** 2)
    lib = LibraryPSF(np.stack([s0, s1, s0]), [0.5, 1.0, 1.5], (10, 10), ee_radius_px=8.0,
                     throughput_fn=lambda r: 0.8, refpa_deg=None, flux_unit=5.0)
    ee = np.hypot(xx - 10, yy - 10) <= 8.0
    for r in (0.5, 0.75, 1.0, 1.5):
        st, c, ok = lib.stamp(r)
        assert ok and c == (10, 10)
        assert st[ee].sum() == pytest.approx(1.0)
    st_lo, _, _ = lib.stamp(0.5)
    st_mid, _, _ = lib.stamp(1.0)
    st_q, _, _ = lib.stamp(0.75)
    # interpolated stamp lies between its neighbours (peak height monotone)
    assert st_mid[10, 10] < st_q[10, 10] < st_lo[10, 10]
    # midpoint = renormalised average of the two neighbours
    avg = 0.5 * (lib.slices[0] + lib.slices[1])
    np.testing.assert_allclose(st_q, avg / avg[ee].sum(), atol=1e-12)
    # outside the tabulated range -> ok False, stamp None
    for r in (0.49, 1.51, np.nan):
        st, c, ok = lib.stamp(r)
        assert not ok and st is None
    assert lib.throughput(0.9) == 0.8 and lib.flux_unit == 5.0
    k = lib.matched_filter_kernel(0.75, FWHM)
    assert k is not None and k.sum() == pytest.approx(1.0)
    assert lib.matched_filter_kernel(3.0, FWHM) is None


def test_inject_sources_fallback_model_and_missing_template():
    lib = LibraryPSF(np.ones((2, 11, 11)), [1.0, 2.0], (5, 5), 4.0)
    cube = np.zeros((2, 41, 41), np.float32)
    with pytest.raises(ValueError):
        inject_sources(cube, np.zeros(2), [Source(0.5, 0.0, 1.0)], lib, PX)
    out = inject_sources(cube, np.zeros(2), [Source(0.5, 0.0, 1.0)], lib, PX, fallback=GaussianPSF(3.0, size=11))
    assert out.sum() > 0 and cube.sum() == 0             # copy by default
    out2 = inject_sources(cube, np.zeros(2), [Source(0.5, 0.0, 1.0)], lib, PX, fallback=GaussianPSF(3.0, size=11), copy=False)
    assert cube.sum() > 0 and out2 is cube


# ----------------------------------------------------------------------------
# detector geometry
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("truenorth", [0.0, 5.0, -12.5])
def test_inject_sources_detector_azimuth_and_derotated_pa(truenorth):
    n = 64
    angles = np.array([0.0, 30.0, -45.0, 100.0])
    cube = np.zeros((angles.size, n, n), np.float32)
    theta, rho = 60.0, 0.8
    out = inject_sources(cube, angles, [Source(rho, theta, 1.0)], GaussianPSF(FWHM, star_flux=1.0), PX,
                         truenorth=truenorth)
    cx, cy = star_center((n, n))
    for j, pa in enumerate(angles):
        py, px = np.unravel_index(np.nanargmax(out[j]), (n, n))
        az = np.rad2deg(np.arctan2(py - cy, px - cx))
        expected = theta - truenorth - 270.0 - pa
        d = (az - expected + 180) % 360 - 180
        assert abs(d) < 3.0                              # integer-pixel peak, ~1 deg at 16 px
        assert np.hypot(px - cx, py - cy) == pytest.approx(rho / PX, abs=1.0)
        assert out[j].sum() == pytest.approx(1.0, rel=1e-3)   # contrast * flux_unit
    # after derotation every frame puts the source at PA theta (metrics convention)
    der = derotate(out, angles, truenorth)
    xs, ys = source_xy(rho, theta, PX, cx, cy)
    for j in range(angles.size):
        py, px = np.unravel_index(np.nanargmax(der[j]), (n, n))
        assert abs(px - xs[0]) <= 1.0 and abs(py - ys[0]) <= 1.0
    comb = np.nan_to_num(np.nanmean(der, axis=0))
    cxm, cym = _centroid(np.where(comb > 0.2 * comb.max(), comb, 0))
    assert cxm == pytest.approx(xs[0], abs=0.3) and cym == pytest.approx(ys[0], abs=0.3)
    # the metric finds it where it expects it
    rng = np.random.default_rng(0)
    assert mawet_peak_snr(comb + 1e-4 * rng.standard_normal((n, n)), rho, theta, PX, FWHM)[0] > 20


def test_inject_sources_math_convention_and_amplitude():
    n = 48
    cube = np.zeros((1, n, n), np.float32)
    out = inject_sources(cube, np.zeros(1), [Source(0.5, 0.0, 2e-3)], GaussianPSF(FWHM, star_flux=1e4), PX,
                         angle_convention="math")
    cx, cy = star_center((n, n))
    py, px = np.unravel_index(np.nanargmax(out[0]), (n, n))
    assert px - cx == pytest.approx(10.0, abs=0.5) and py == pytest.approx(cy, abs=0.5)   # along +x
    assert out.sum() == pytest.approx(2e-3 * 1e4, rel=1e-3)


def test_inject_sources_rotates_anisotropic_stamp():
    n = 41
    st = np.zeros((11, 11))
    st[5, 3:8] = 1.0                                     # horizontal bar, reference axis along +x
    lib = LibraryPSF(np.stack([st, st]), [0.1, 2.0], (5, 5), 6.0, refpa_deg=0.0)
    cube = np.zeros((1, n, n), np.float32)
    # source at detector azimuth 90 deg (math convention, parang 0): bar should become vertical
    out = inject_sources(cube, np.zeros(1), [Source(0.5, 90.0, 1.0)], lib, PX, angle_convention="math")
    img = out[0]
    py, px = np.unravel_index(np.nanargmax(img), img.shape)
    thr = 0.5 * img.max()                                # stamp is unit-normalised (0.2 per pixel)
    col = img[:, px]
    row = img[py, :]
    assert (col > thr).sum() >= 4 and (row > thr).sum() <= 2
