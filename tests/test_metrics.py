import numpy as np
import pytest

from klip_tpe.metrics import (InjectionDifferenceSNR, MawetPeakSNR, Metric, Objective, Source,
                              aperture_mawet_snr, clip0, declip, gaussian_kernel,
                              injection_difference_snr, kernel_from_profile, mawet_peak_snr,
                              nanmedian_even, radprof, source_xy, star_center)

FWHM = 4.0
PX = 0.05
N = 101


def _grid(n=N):
    yy, xx = np.mgrid[0:n, 0:n]
    return yy, xx


def _gauss(xs, ys, amp, fwhm=FWHM, n=N):
    yy, xx = _grid(n)
    sg = fwhm / 2.3548
    return amp * np.exp(-0.5 * ((xx - xs) ** 2 + (yy - ys) ** 2) / sg ** 2)


@pytest.fixture
def noise():
    return np.random.default_rng(7).standard_normal((N, N))


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def test_star_center_even_and_odd():
    assert star_center((101, 101)) == (50.0, 50.0)
    assert star_center((100, 100)) == (49.5, 49.5)
    assert star_center((3, 10, 20)) == (9.5, 4.5)


def test_source_xy_pa_conventions():
    cx = cy = 50.0
    x, y = source_xy(1.0, 0.0, PX, cx, cy)           # PA 0 -> north, +y
    assert x[0] == pytest.approx(50.0, abs=1e-9) and y[0] == pytest.approx(70.0)
    x, y = source_xy(1.0, 90.0, PX, cx, cy)          # PA 90 -> east, -x
    assert x[0] == pytest.approx(30.0) and y[0] == pytest.approx(50.0, abs=1e-9)
    x, y = source_xy(1.0, 180.0, PX, cx, cy)
    assert y[0] == pytest.approx(30.0)
    x, y = source_xy(1.0, 0.0, PX, cx, cy, angle_convention="math")   # from +x
    assert x[0] == pytest.approx(70.0) and y[0] == pytest.approx(50.0, abs=1e-9)
    xs, ys = source_xy([0.5, 1.0], [0.0, 90.0], PX, cx, cy)
    assert xs.shape == (2,) and ys[0] == pytest.approx(60.0)


def test_nanmedian_even():
    assert nanmedian_even([1, 2, 3, 4]) == 2.5
    assert nanmedian_even([3, np.nan, 1]) == 2.0
    assert nanmedian_even([np.nan, 7]) == 7.0
    assert np.isnan(nanmedian_even([np.nan, np.nan]))
    assert np.isnan(nanmedian_even([]))


# ----------------------------------------------------------------------------
# image helpers
# ----------------------------------------------------------------------------
def test_radprof_removes_azimuthal_profile():
    yy, xx = _grid()
    rr = np.hypot(xx - N / 2.0, yy - N / 2.0)      # IDL centre convention (nx/2, ny/2)
    prof = 100.0 * np.exp(-rr / 10.0)
    out = radprof(prof)
    assert np.abs(out).max() < 0.1 * prof.max()
    assert np.abs(out[rr > 3]).max() < 3.0           # residual from within-bin gradient only
    # profile-free image with a point source is essentially unchanged near the source
    src = _gauss(70, 50, 10.0)
    o2 = radprof(src)
    assert o2[50, 70] == pytest.approx(10.0, abs=0.5)
    # NaN preserved
    img = prof.copy()
    img[5, 5] = np.nan
    assert np.isnan(radprof(img)[5, 5]) and np.isfinite(radprof(img)[50, 50])


def test_declip_removes_hot_pixel_keeps_gaussian(noise):
    img = 0.1 * noise.copy()
    img[30, 30] = 50.0
    d = declip(img)
    assert abs(d[30, 30]) < 1.0
    assert np.array_equal(np.delete(d.ravel(), 30 * N + 30), np.delete(img.ravel(), 30 * N + 30))
    # a FWHM-wide Gaussian (peak-to-neighbour residual below the 6-sigma cut) survives
    img2 = noise.copy() + _gauss(60, 40, 20.0)
    d2 = declip(img2)
    assert d2[40, 60] == pytest.approx(img2[40, 60])
    # NaN safe, nothing touched near the NaN
    img3 = noise.copy()
    img3[10, 10] = np.nan
    img3[11, 10] = 50.0                              # adjacent to NaN -> not fully interior, untouched
    d3 = declip(img3)
    assert np.isnan(d3[10, 10]) and d3[11, 10] == 50.0
    # too few interior pixels -> unchanged
    tiny = np.zeros((3, 3))
    tiny[1, 1] = 100
    assert np.array_equal(declip(tiny), tiny)


def test_declip_clips_peak_of_very_bright_gaussian(noise):
    """Documented behaviour, not a requirement: a FWHM=4 px Gaussian whose peak
    exceeds ~40x the noise has a 3x3-median residual above 6 sigma and its peak
    pixel is replaced.  The matched filter afterwards smooths this, but it is a
    flux bias for bright sources worth knowing about."""
    img = noise.copy() + _gauss(60, 40, 200.0)
    d = declip(img)
    assert d[40, 60] < img[40, 60] - 5


def test_gaussian_kernel_unit_sum_and_size():
    k = gaussian_kernel(FWHM)
    assert k.shape == (2 * int(np.ceil(1.5 * FWHM)) + 1,) * 2 == (13, 13)
    assert k.sum() == pytest.approx(1.0)
    assert k[6, 6] == k.max()
    assert np.allclose(k, k.T) and np.allclose(k, k[::-1, ::-1])
    # half maximum at FWHM/2 = 2 px from the centre
    assert k[6, 8] / k[6, 6] == pytest.approx(0.5, abs=1e-3)
    assert k[8, 6] / k[6, 6] == pytest.approx(0.5, abs=1e-3)
    assert gaussian_kernel(FWHM, size=7).shape == (7, 7)
    assert gaussian_kernel(6.5).shape == (21, 21)


def test_kernel_from_profile_matches_gaussian():
    st = _gauss(20, 20, 3.0, n=41)
    k = kernel_from_profile(st, FWHM)
    g = gaussian_kernel(FWHM)
    assert k.shape == g.shape and k.sum() == pytest.approx(1.0)
    assert np.abs(k - g).max() < 0.02
    assert np.array_equal(kernel_from_profile(np.zeros((41, 41)), FWHM), g)  # empty -> Gaussian fallback


def test_clip0_keeps_nan():
    v = clip0([-1.0, 0.0, 2.5, np.nan, -np.inf])
    assert v[0] == 0.0 and v[1] == 0.0 and v[2] == 2.5 and np.isnan(v[3])
    assert v[4] == -np.inf                      # only *finite* negatives are clamped
    assert clip0(np.float64(-2.0)) == 0.0


# ----------------------------------------------------------------------------
# Mawet peak S/N
# ----------------------------------------------------------------------------
def test_mawet_peak_snr_recovers_planted_source(noise):
    cx, cy = star_center(noise.shape)
    xs, ys = source_xy(1.0, 30.0, PX, cx, cy)
    img = noise + _gauss(xs[0], ys[0], 20.0)
    s = mawet_peak_snr(img, 1.0, 30.0, PX, FWHM)
    assert s.shape == (1,) and s[0] > 25
    s2, det = mawet_peak_snr(img, [1.0], [30.0], PX, FWHM, return_details=True)
    assert s2[0] == s[0]
    assert abs(det[0]["x"] - xs[0]) <= 1.5 and abs(det[0]["y"] - ys[0]) <= 1.5
    assert det[0]["nap"] == int(np.floor(2 * np.pi * det[0]["r"] / FWHM))
    # sub-pixel offset of the nominal position is absorbed by the search radius
    s3 = mawet_peak_snr(img, 1.0 + 0.6 * PX, 31.0, PX, FWHM)
    assert s3[0] > 0.8 * s[0]


def test_mawet_peak_snr_empty_positions_near_zero(noise):
    thetas = np.arange(0, 360, 40.0)
    s = mawet_peak_snr(noise, np.full(thetas.size, 1.0), thetas, PX, FWHM)
    assert np.all(np.isfinite(s))
    assert np.median(np.abs(s)) < 2.0           # peak-search bias is small
    assert np.abs(s).max() < 4.0


def test_mawet_peak_snr_can_be_negative(noise):
    cx, cy = star_center(noise.shape)
    xs, ys = source_xy(1.0, 30.0, PX, cx, cy)
    img = noise - _gauss(xs[0], ys[0], 20.0)
    assert mawet_peak_snr(img, 1.0, 30.0, PX, FWHM)[0] < -5


def test_mawet_peak_snr_nan_inside_fwhm_and_details(noise):
    s = mawet_peak_snr(noise, [0.5 * FWHM * PX, 1.0], [0.0, 0.0], PX, FWHM)
    assert np.isnan(s[0]) and np.isfinite(s[1])
    assert np.isnan(mawet_peak_snr(noise, 0.99 * FWHM * PX, 0.0, PX, FWHM)[0])
    assert np.isfinite(mawet_peak_snr(noise, 1.3 * FWHM * PX, 0.0, PX, FWHM)[0])


def test_mawet_peak_snr_excludes_other_sources_and_known(noise):
    cx, cy = star_center(noise.shape)
    # a bright contaminant on the same ring inflates the noise unless excluded
    xk, yk = source_xy(1.0, 200.0, PX, cx, cy)
    img = noise + _gauss(xk[0], yk[0], 60.0)
    xs, ys = source_xy(1.0, 30.0, PX, cx, cy)
    img = img + _gauss(xs[0], ys[0], 20.0)
    s_plain = mawet_peak_snr(img, 1.0, 30.0, PX, FWHM)[0]
    s_known = mawet_peak_snr(img, 1.0, 30.0, PX, FWHM, known=[(1.0, 200.0)])[0]
    s_both = mawet_peak_snr(img, [1.0, 1.0], [30.0, 200.0], PX, FWHM)[0]
    assert s_known > s_plain
    assert s_both > s_plain
    # pixel mask: masking everything except a thin band leaves the estimate finite
    mask = np.zeros(noise.shape, bool)
    mask[:, :10] = True
    assert np.isfinite(mawet_peak_snr(img, 1.0, 30.0, PX, FWHM, pixel_mask=mask)[0])


def test_mawet_peak_snr_penalty_modes_and_nan_pixels(noise):
    cx, cy = star_center(noise.shape)
    xs, ys = source_xy(1.0, 30.0, PX, cx, cy)
    img = noise + _gauss(xs[0], ys[0], 20.0)
    g = mawet_peak_snr(img, 1.0, 30.0, PX, FWHM, penalty="geometric")[0]
    r = mawet_peak_snr(img, 1.0, 30.0, PX, FWHM, penalty="retained")[0]
    assert g > 0 and r > 0 and g != r
    img2 = img.copy()
    img2[:20, :] = np.nan
    assert np.isfinite(mawet_peak_snr(img2, 1.0, 30.0, PX, FWHM)[0])
    # source itself inside a NaN hole: no finite peak pixel is found, the nominal
    # position is scored on the zero-filled matched-filter image -> ~0, not NaN
    # (documented behaviour; a NaN would drop the source from the aggregate instead)
    img3 = img.copy()
    img3[int(ys[0]) - 3:int(ys[0]) + 4, int(xs[0]) - 3:int(xs[0]) + 4] = np.nan
    s3, det3 = mawet_peak_snr(img3, 1.0, 30.0, PX, FWHM, declip_nsig=None, return_details=True)
    assert np.isnan(det3[0]["peak"]) and abs(s3[0]) < 1.0


def test_other_snr_estimators(noise):
    cx, cy = star_center(noise.shape)
    xs, ys = source_xy(1.0, 30.0, PX, cx, cy)
    inj = noise + _gauss(xs[0], ys[0], 20.0)
    d = injection_difference_snr(inj, noise, 1.0, 30.0, PX, FWHM)
    assert d.shape == (1,) and d[0] > 10
    a = aperture_mawet_snr(inj, 1.0, 30.0, PX, FWHM)
    assert a[0] > 5
    assert abs(aperture_mawet_snr(noise, 1.0, 100.0, PX, FWHM)[0]) < 4


# ----------------------------------------------------------------------------
# metric objects and the objective
# ----------------------------------------------------------------------------
def test_mawet_metric_object_flatten_and_kernel(noise):
    cx, cy = star_center(noise.shape)
    xs, ys = source_xy(1.0, 30.0, PX, cx, cy)
    yy, xx = _grid()
    halo = 50 * np.exp(-np.hypot(xx - N / 2, yy - N / 2) / 8.0)
    img = halo + noise + _gauss(xs[0], ys[0], 20.0)

    # flatten is OFF by default, and that is not a free choice: this halo falls off as
    # exp(-r/8), so it has a steep gradient ACROSS each reference aperture.  radprof
    # removes it (it is azimuthally symmetric); without radprof the gradient inflates
    # the ring scatter and the same source measures ~10x lower.  Both numbers are
    # pinned so a silent flip of the default cannot pass.
    assert MawetPeakSNR(pxscale=PX, fwhm=FWHM).flatten is False
    s_flat = MawetPeakSNR(pxscale=PX, fwhm=FWHM, flatten=True).per_source(img, None, [1.0], [30.0])
    s_raw = MawetPeakSNR(pxscale=PX, fwhm=FWHM).per_source(img, None, [1.0], [30.0])
    assert s_flat[0] > 20
    assert 2.0 < s_raw[0] < 0.25 * s_flat[0]

    m = MawetPeakSNR(pxscale=PX, fwhm=FWHM, flatten=True)
    s = m.per_source(img, None, [1.0], [30.0])
    assert s[0] > 20
    calls = []

    def kfn(r):
        calls.append(r)
        return gaussian_kernel(FWHM, size=9)

    m2 = MawetPeakSNR(pxscale=PX, fwhm=FWHM, kernel_fn=kfn)
    m2.per_source(img, None, [0.8, 1.0], [30.0, 90.0])
    assert calls == [0.9]
    assert m2.describe()["measured_kernel"] is True
    m3 = MawetPeakSNR(pxscale=PX, fwhm=FWHM, kernel_fn=lambda r: None)
    assert m3.kernel(1.0).shape == gaussian_kernel(FWHM).shape


def test_radial_profile_subtraction_is_off_by_default_everywhere():
    """One assertion per place radprof used to be applied unconditionally.

    The point of the change is that a run is scored, saved and displayed on the SAME
    image.  Each of these was an independent default, so a single one flipping back is
    exactly the kind of inconsistency this pins down.
    """
    import inspect
    from klip_tpe import candidates, display, fmmf, param_verify, products, verify

    assert MawetPeakSNR(pxscale=PX, fwhm=FWHM).flatten is False
    assert InjectionDifferenceSNR(pxscale=PX, fwhm=FWHM).flatten is False
    assert fmmf.FMMFSNR(pxscale=PX, fwhm=FWHM).flatten is False

    for fn in (fmmf.fmmf_map, display.draw_image, display._snr_map, products.fm_contrast_curve,
               verify.verify_maps, param_verify.param_verify, param_verify.param_verify_cubes,
               candidates.find_candidates, candidates.candidate_metrics):
        p = inspect.signature(fn).parameters.get("flatten")
        assert p is not None, f"{fn.__name__} lost its flatten switch"
        assert p.default is False, f"{fn.__name__} flattens by default"


def test_runner_products_follow_the_metric_not_a_separate_knob():
    """The saved stitches used to flatten whatever the metric did, so the FITS on disk
    could differ from the image that was optimised.  They now follow the metric."""
    from klip_tpe.runner import Runner

    class _R:
        flatten_products = Runner.flatten_products
        _flat_tag = Runner._flat_tag
        objective = Objective(MawetPeakSNR(pxscale=PX, fwhm=FWHM))

    class _RF(_R):
        objective = Objective(MawetPeakSNR(pxscale=PX, fwhm=FWHM, flatten=True))

    assert _R().flatten_products is False
    assert _R()._flat_tag == ""
    assert _RF().flatten_products is True
    assert "radprof" in _RF()._flat_tag


def test_injection_difference_metric_needs_clean(noise):
    m = InjectionDifferenceSNR(pxscale=PX, fwhm=FWHM)
    assert m.needs_clean
    with pytest.raises(ValueError):
        m.per_source(noise, None, [1.0], [0.0])
    obj = Objective(m, clean_subtract=False)
    assert obj.needs_clean


class _FakeMetric(Metric):
    """Returns a preset per-source vector for the injected image and another for the clean one."""

    name = "fake"

    def __init__(self, inj, clean):
        self.inj, self.clean = np.asarray(inj, float), np.asarray(clean, float)

    def per_source(self, img_inj, img_clean, rho, theta):
        return self.inj.copy() if img_inj is not None and img_inj.sum() > 0 else self.clean.copy()


def test_objective_score_search_clean_subtraction_rule():
    m = _FakeMetric(inj=[6.0, 5.0, 8.0, 7.0], clean=[2.0, -3.0, np.nan, 0.0])
    obj = Objective(m, clean_subtract=True)
    img_inj, img_clean = np.ones((4, 4)), np.zeros((4, 4))
    src = [Source(1.0, t) for t in (0, 90, 180, 270)]
    r = obj.score_search(img_inj, src, img_clean)
    # raw - max(clean, 0); negative clean never inflates; NaN clean drops the source
    np.testing.assert_array_equal(r.per_source[[0, 1, 3]], [4.0, 5.0, 7.0])
    assert np.isnan(r.per_source[2])
    assert r.score == 5.0                        # median of [4, 5, 7]
    assert r.raw_score == 6.5                    # median of the raw four
    np.testing.assert_array_equal(r.raw_per_source, [6, 5, 8, 7])
    np.testing.assert_array_equal(r.clean_per_source[:2], [2.0, -3.0])
    # raw scoring ignores the clean image
    rr = obj.score_raw(img_inj, src, img_clean)
    assert rr.score == 6.5 and rr.clean_per_source is None
    with pytest.raises(ValueError):
        obj.score_search(img_inj, src, None)


def test_objective_without_clean_subtract_and_aggregates():
    m = _FakeMetric(inj=[1.0, np.nan, 4.0], clean=[0, 0, 0])
    obj = Objective(m, clean_subtract=False)
    r = obj.score_search(np.ones((2, 2)), [Source(1, 0)] * 3)
    assert r.score == 2.5 and not obj.needs_clean
    obj_mean = Objective(m, clean_subtract=False, aggregate="mean")
    assert obj_mean.score_raw(np.ones((2, 2)), [Source(1, 0)] * 3).score == 2.5
    m2 = _FakeMetric(inj=[np.nan, np.nan], clean=[0, 0])
    assert np.isnan(Objective(m2, clean_subtract=False).score_raw(np.ones((2, 2)), [Source(1, 0)] * 2).score)
    assert obj.describe()["metric"]["name"] == "fake"


def test_known_sources_excluded_from_noise_rings():
    """A real companion must not inflate sigma (and thus the contrast limit) at its own
    separation: noise_profile / contrast_curve / fm_contrast_curve take known=."""
    from klip_tpe.products import contrast_curve, noise_profile
    rng = np.random.default_rng(4)
    n = 121
    img = rng.normal(0, 1.0, (n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    rho_as, pa, pxscale, fwhm = 0.5, 40.0, 0.01, 4.0
    x0 = c - rho_as / pxscale * np.sin(np.deg2rad(pa))
    y0 = c + rho_as / pxscale * np.cos(np.deg2rad(pa))
    img += 60.0 * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / (fwhm / 2.355) ** 2)   # the companion
    s_raw, r = noise_profile(img, fwhm, 10, 55)
    s_ex, _ = noise_profile(img, fwhm, 10, 55, known=[(rho_as, pa)], pxscale=pxscale)
    i = int(np.argmin(np.abs(r * pxscale - rho_as)))
    assert s_raw[i] > 3 * s_ex[i]                      # the planet dominated its own ring
    far = np.abs(r * pxscale - rho_as) > 3 * fwhm * pxscale
    assert np.allclose(s_raw[far], s_ex[far], equal_nan=True)      # nothing else changes
    cc_raw = contrast_curve(img, [0.3, 0.5, 0.7], [5.0, 5.0, 5.0], 1e-4, fwhm, pxscale, 10, 55)
    cc_ex = contrast_curve(img, [0.3, 0.5, 0.7], [5.0, 5.0, 5.0], 1e-4, fwhm, pxscale, 10, 55,
                           known=[(rho_as, pa)])
    j = int(np.argmin(np.abs(cc_raw["r_as"] - rho_as)))
    assert cc_ex["curve"][j] < cc_raw["curve"][j]                  # the limit is no longer inflated
