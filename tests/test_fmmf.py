"""Forward-modelled matched filter (klip_tpe.fmmf) and the STPSF PSF grid.

The STPSF tests that need the package/data files are skipped when it is not installed;
everything that only needs the *conventions* (stamp centring, cache round-trip, header
parsing) runs everywhere against synthetic arrays.
"""
import json

import numpy as np
import pytest

from klip_tpe.fmmf import FMMFSNR, fm_kernel, fm_kernels, fmmf_map
from klip_tpe.metrics import MawetPeakSNR, Objective, Source, gaussian_kernel, source_xy, star_center

from conftest import build_synthetic_run

QUIET = (lambda s: None)


PX, FWHM = 0.045, 4.0


def _gauss(ny, nx, x, y, fwhm, amp=1.0):
    yy, xx = np.mgrid[0:ny, 0:nx]
    s = fwhm / 2.3548
    return amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))


def _fm_like(ny, nx, x, y, fwhm, amp=1.0, lobe=0.45, dy=None):
    """A planet as KLIP leaves it: an attenuated core with negative lobes above and
    below -- the shape a Gaussian filter is *wrong* for."""
    dy = dy if dy is not None else 1.3 * fwhm
    return (_gauss(ny, nx, x, y, fwhm, amp)
            - _gauss(ny, nx, x, y - dy, fwhm, lobe * amp)
            - _gauss(ny, nx, x, y + dy, fwhm, lobe * amp))


def _scene(n=101, rho_as=1.0, theta=0.0, amp=1.0, noise=0.0, seed=3):
    """(image, fm_image, sources) with one planet of the FM shape at (rho, theta)."""
    cx, cy = star_center((n, n))
    xs, ys = source_xy([rho_as], [theta], PX, cx, cy)
    fm = _fm_like(n, n, xs[0], ys[0], FWHM, 1.0)
    img = amp * fm.copy()
    if noise:
        rng = np.random.default_rng(seed)
        from scipy import ndimage
        img = img + ndimage.gaussian_filter(rng.normal(0, noise, (n, n)), FWHM / 2.3548) * 2.3548
    return img, fm, [Source(rho_as, theta, 1e-4)]


# ------------------------------------------------------------------ kernels
def test_fm_kernel_is_zero_mean_and_lsq_normalised():
    img, fm, src = _scene(amp=3.0)
    cx, cy = star_center(fm.shape)
    x, y = (v[0] for v in source_xy([src[0].rho], [src[0].theta], PX, cx, cy))
    k = fm_kernel(fm, x, y, size=int(2 * np.ceil(2.5 * FWHM) + 1))
    assert k is not None
    assert abs(float(k.sum())) < 1e-9                      # zero-mean: blind to a pedestal
    # filtering the scene with it returns the amplitude of the template
    from scipy import ndimage
    amp = ndimage.convolve(img, k, mode="nearest")[int(round(y)), int(round(x))]
    assert amp == pytest.approx(3.0, rel=0.05)


def test_fm_kernel_declines_empty_or_flat_stamps():
    n = 41
    assert fm_kernel(np.zeros((n, n)), 20, 20, 9) is None
    assert fm_kernel(np.full((n, n), np.nan), 20, 20, 9) is None
    assert fm_kernel(np.zeros((n, n)), 200, 200, 9) is None          # off the frame
    noise = np.random.default_rng(0).normal(0, 1, (n, n))
    assert fm_kernel(noise, 20, 20, 9, min_snr_px=100.0) is None     # nothing above the scatter


def test_fm_kernels_one_per_source_and_none_without_a_model():
    _, fm, _ = _scene()
    rho, th = [0.6, 1.0, 1.4], [0.0, 90.0, 200.0]
    ks = fm_kernels(fm, rho, th, PX, FWHM)
    assert len(ks) == 3 and ks[1] is not None
    assert fm_kernels(None, rho, th, PX, FWHM) == [None, None, None]
    # a k-scan cube (wrong shape for this image) is declined rather than mis-sliced
    assert fm_kernels(fm, rho, th, PX, FWHM, shape=(7, 7)) == [None, None, None]


# ------------------------------------------------------------------ the metric
def test_fmmf_beats_the_psf_matched_filter_on_a_self_subtracted_planet():
    """Same image, same statistics: only the filter differs.  The FM template knows
    about the negative lobes, so it collects more of the planet."""
    img, fm, src = _scene(amp=1.0, noise=0.05)
    rho, th = [s.rho for s in src], [s.theta for s in src]
    gauss = MawetPeakSNR(pxscale=PX, fwhm=FWHM).per_source(img, None, rho, th)[0]
    fmmf = FMMFSNR(pxscale=PX, fwhm=FWHM).per_source(img, None, rho, th, fm=fm)[0]
    assert np.isfinite(gauss) and np.isfinite(fmmf)
    assert fmmf > gauss


def test_fmmf_falls_back_to_the_psf_kernel_without_a_forward_model():
    img, fm, src = _scene(amp=1.0, noise=0.05)
    rho, th = [s.rho for s in src], [s.theta for s in src]
    kfn = lambda r: gaussian_kernel(FWHM)                             # noqa: E731
    m = FMMFSNR(pxscale=PX, fwhm=FWHM, kernel_fn=kfn)
    got = m.per_source(img, None, rho, th, fm=None)[0]
    ref = MawetPeakSNR(pxscale=PX, fwhm=FWHM, kernel_fn=kfn).per_source(img, None, rho, th)[0]
    assert got == pytest.approx(ref, rel=1e-9, nan_ok=True)
    assert m.stats == {"fm": 0, "fallback": 1}
    assert m.describe()["fm_fraction"] == 0.0


def test_fmmf_mixes_fm_and_fallback_per_source():
    """A source the forward model covers uses the FM template; one it does not falls
    back, and describe() reports the mix."""
    img, fm, src = _scene()
    cx, cy = star_center(fm.shape)
    x0, y0 = (v[0] for v in source_xy([src[0].rho], [src[0].theta], PX, cx, cy))
    yy, xx = np.mgrid[0:fm.shape[0], 0:fm.shape[1]]
    fm = np.where(np.hypot(xx - x0, yy - y0) < 12, fm, 0.0)   # only this source is modelled
    m = FMMFSNR(pxscale=PX, fwhm=FWHM)
    ks = m.kernels(fm, [1.0, 1.9], [0.0, 137.0])
    assert len(ks) == 2
    assert m.stats["fm"] == 1 and m.stats["fallback"] == 1
    assert m.describe()["fm_fraction"] == pytest.approx(0.5)


def test_objective_only_passes_fm_to_a_metric_that_asked_for_it():
    img, fm, src = _scene(amp=1.0, noise=0.05)
    plain = Objective(MawetPeakSNR(pxscale=PX, fwhm=FWHM), clean_subtract=False)
    plain.score_raw(img, src, None, fm=fm)                  # accepted and ignored
    assert plain._fmkw(fm) == {}
    assert Objective(FMMFSNR(pxscale=PX, fwhm=FWHM))._fmkw(fm) == {"fm": fm}


def test_clean_term_uses_the_same_templates_as_the_injected_term():
    """The FM template belongs to the configuration, not to the injection, so the
    clean-subtraction term must be filtered with the identical kernels."""
    img, fm, src = _scene(amp=1.0, noise=0.05)
    m = FMMFSNR(pxscale=PX, fwhm=FWHM)
    seen = []
    orig = m.kernels

    def spy(fm_, rho, theta, shape=None):
        ks = orig(fm_, rho, theta, shape=shape)
        seen.append([None if k is None else float(np.sum(k * k)) for k in ks])
        return ks
    m.kernels = spy
    Objective(m, clean_subtract=True).score_search(img, src, img * 0.1, fm=fm)
    assert len(seen) == 2 and seen[0] == seen[1]


# ------------------------------------------------------------------ full-frame map
def test_fmmf_map_recovers_the_amplitude_and_flags_the_planet():
    img, fm, src = _scene(n=81, rho_as=0.8, amp=2.0)
    out = fmmf_map(img, fm, src, PX, FWHM, contrast=1.0, flatten=False, n_theta=16)
    assert set(out) == {"amplitude", "snr", "seps_px", "templates"}
    cx, cy = star_center(img.shape)
    x, y = (v[0] for v in source_xy([src[0].rho], [src[0].theta], PX, cx, cy))
    assert np.isfinite(out["amplitude"][int(round(y)), int(round(x))])
    assert out["snr"][int(round(y)), int(round(x))] > 3.0


def test_fmmf_map_needs_a_usable_forward_model():
    img, _, src = _scene()
    with pytest.raises(ValueError):
        fmmf_map(img, np.zeros_like(img), src, PX, FWHM)


# ------------------------------------------------------------------ runner wiring
def test_runner_requests_the_fm_pass_for_an_fmmf_objective(tmp_path):
    """``needs_fm`` must make the runner ask its reducer for the KLIP-FM response of the
    very sources being scored -- otherwise the metric silently runs on fallbacks."""
    from klip_tpe import PartitionedReducer
    from klip_tpe.runner import Runner
    from klip_tpe.synthetic import make_synthetic_partitions

    _, space, _, samp, cfg = build_synthetic_run(n_iter=[3, 3])
    red = PartitionedReducer(make_synthetic_partitions(["n1", "n2"], k_opts=[6, 12], fm=True))
    metric = FMMFSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel)
    runner = Runner(red, space, Objective(metric, clean_subtract=True), samp, cfg,
                    str(tmp_path), log=QUIET)
    runner.ia = 0
    asked = []
    orig = red.reduce_config

    def spy(cfg_, sources, k_scan=False, tag="", fm_sources=None, extras=False):
        asked.append((tag, sources is not None, fm_sources is not None))
        return orig(cfg_, sources, k_scan, tag, fm_sources=fm_sources, extras=extras)
    red.reduce_config = spy
    rec, inj_ev, _ = runner.evaluate(space.default_vector(), "seed", contrast=3e-4)

    inj = [a for a in asked if a[0].endswith("_inj")]
    cln = [a for a in asked if a[0].endswith("_clean")]
    assert inj and all(a[2] for a in inj), "injected reduction ran without the FM pass"
    assert cln and not any(a[2] for a in cln), "clean reduction should not forward-model"
    assert rec.score is None or np.isfinite(rec.score)
    assert inj_ev.fm_image is not None and np.isfinite(inj_ev.fm_image).any()
    assert metric.stats["fm"] > 0, "the forward-modelled templates were never used"
    assert rec.partition_snr and any(v is not None for v in rec.partition_snr.values())


def test_numerical_forward_model_when_the_backend_has_no_klip_fm(tmp_path):
    """pyKLIP / VIP / spaceKLIP have no analytic KLIP-FM.  The runner must then hand the
    metric ``injected - clean``, which is the same response to first order -- so FMMF
    still filters with forward-modelled templates, not with fallbacks."""
    from klip_tpe import PartitionedReducer
    from klip_tpe.runner import Runner
    from klip_tpe.synthetic import make_synthetic_partitions

    _, space, _, samp, cfg = build_synthetic_run(n_iter=[3, 3])
    red = PartitionedReducer(make_synthetic_partitions(["n1", "n2"], k_opts=[6, 12]))   # fm=False
    assert not red.supports_fm
    metric = FMMFSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel)
    runner = Runner(red, space, Objective(metric, clean_subtract=True), samp, cfg,
                    str(tmp_path), log=QUIET)
    runner.ia = 0
    asked = []
    orig = red.reduce_config

    def spy(cfg_, sources, k_scan=False, tag="", fm_sources=None, extras=False):
        asked.append(fm_sources is not None)
        return orig(cfg_, sources, k_scan, tag, fm_sources=fm_sources, extras=extras)
    red.reduce_config = spy
    rec, inj_ev, clean_ev = runner.evaluate(space.default_vector(), "seed", contrast=3e-4)

    assert asked and not any(asked), "no point asking a backend without KLIP-FM for it"
    assert inj_ev.fm_image is None and clean_ev is not None
    fm = runner._fm_for(inj_ev, clean_ev)["fm"]
    assert fm is not None and np.allclose(fm, inj_ev.image - clean_ev.image, equal_nan=True)
    assert metric.stats["fm"] > 0, "the numerical forward model was not used"
    # and without a clean reduction there is simply no forward model
    assert runner._fm_for(inj_ev, None)["fm"] is None


def test_fm_is_not_requested_for_an_ordinary_metric(tmp_path):
    from klip_tpe.runner import Runner

    red, space, obj, samp, cfg = build_synthetic_run(n_iter=[3, 3])
    runner = Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)
    runner.ia = 0
    asked = []
    orig = red.reduce_config

    def spy(cfg_, sources, k_scan=False, tag="", fm_sources=None, extras=False):
        asked.append(fm_sources is not None)
        return orig(cfg_, sources, k_scan, tag, fm_sources=fm_sources, extras=extras)
    red.reduce_config = spy
    runner.evaluate(space.default_vector(), "seed", contrast=3e-4)
    assert asked and not any(asked)


# ------------------------------------------------------------------ STPSF conventions
def test_stamp_centring_and_source_measurement():
    from klip_tpe import stpsf_psf as S

    big = np.zeros((61, 61))
    big += _gauss(61, 61, 42.3, 30.0, 3.0)              # a source 12.3 px to +x of centre
    sx, sy = S._source_center(big, (42.0, 30.0))
    assert sx == pytest.approx(42.3, abs=0.15) and sy == pytest.approx(30.0, abs=0.15)
    st = S._stamp_at(big, (sx, sy), 21)
    assert st.shape == (21, 21)
    px, py = S._source_center(st, (10, 10))
    assert px == pytest.approx(10.0, abs=0.1) and py == pytest.approx(10.0, abs=0.1)


def test_offaxis_cache_round_trip_and_library(tmp_path, monkeypatch):
    from klip_tpe import stpsf_psf as S

    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path))
    n, ns = 21, 4
    seps = np.array([0.3, 0.6, 1.2, 2.4])
    g = {"slices": np.array([_gauss(n, n, 10, 10, 3.0, 1.0) for _ in range(ns)]),
         "seps": seps, "transmission": np.array([0.05, 0.4, 0.95, 1.0]),
         "unocculted": _gauss(n, n, 10, 10, 3.0), "pxscale": 0.063, "fwhm_px": 3.0,
         "ee_radius_px": 4.5, "center": (10.0, 10.0), "meta": {"instrument": "NIRCam"}}
    p = str(tmp_path / "grid.fits")
    S._write_cache(p, g)
    back = S._read_cache(p)
    assert back["center"] == (10.0, 10.0)
    assert back["pxscale"] == pytest.approx(0.063)
    assert back["ee_radius_px"] == pytest.approx(4.5)
    assert back["meta"] == {"instrument": "NIRCam"}
    assert np.allclose(back["transmission"], g["transmission"])
    assert np.allclose(back["slices"], g["slices"], atol=1e-6)

    m = S.library(back, star_flux=7.0)
    st, c, ok = m.stamp(0.9)
    assert ok and c == (10.0, 10.0) and m.flux_unit == 7.0
    assert m.refpa_deg == 0.0
    assert m.throughput(0.6) == pytest.approx(0.4)
    assert m.throughput(0.1) == pytest.approx(0.05)       # clamped below the first knot
    assert m.throughput(9.9) == pytest.approx(1.0)        # and above the last


def test_mode_from_header_reads_the_coronagraph():
    from klip_tpe.stpsf_psf import mode_from_header

    m = mode_from_header({"INSTRUME": "NIRCAM", "FILTER": "F444W", "CORONMSK": "MASKA335R",
                          "PUPIL": "MASKRND", "APERNAME": "NRCA5_MASK335R"})
    assert m == {"instrument": "NIRCam", "filter": "F444W", "image_mask": "MASK335R",
                 "pupil_mask": "MASKRND", "aperture": "NRCA5_MASK335R"}
    # the Lyot stop is filled in from the mask when PUPIL carries something else
    m2 = mode_from_header({"INSTRUME": "NIRCAM", "FILTER": "F356W", "CORONMSK": "MASKA335R",
                           "PUPIL": "CLEAR"})
    assert m2["pupil_mask"] == "MASKRND"
    m3 = mode_from_header({"INSTRUME": "MIRI", "FILTER": "F1140C", "CORONMSK": "FQPM1140"})
    assert m3["instrument"] == "MIRI" and m3["pupil_mask"] == "MASKFQPM"


@pytest.mark.skipif(not __import__("klip_tpe.stpsf_psf", fromlist=["x"]).have_stpsf(),
                    reason="stpsf and its data files are not installed")
def test_offaxis_grid_against_the_real_coronagraph(tmp_path, monkeypatch):
    """One real STPSF grid: the source must land on the stamp centre and the throughput
    must rise monotonically to 1, with MASK335R's half-transmission near 0.63 arcsec."""
    from klip_tpe import stpsf_psf as S

    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path))
    seps = [0.3, 0.45, 0.6, 0.9, 1.2]
    g = S.offaxis_grid("NIRCam", "F444W", image_mask="MASK335R", seps_as=seps,
                       stamp_px=21, nlambda=1, oversample=2, log=lambda s: None)
    assert g["slices"].shape == (len(seps), 21, 21)
    for sl in g["slices"]:
        x, y = S._source_center(sl, (10, 10))
        assert x == pytest.approx(10.0, abs=0.25) and y == pytest.approx(10.0, abs=0.25)
    t = g["transmission"]
    assert np.all(np.diff(t) > 0) and t[0] < 0.2 and t[-1] > 0.9
    assert float(np.interp(0.5, t, g["seps"])) == pytest.approx(0.63, abs=0.08)
    # and the second call comes straight out of the cache
    g2 = S.offaxis_grid("NIRCam", "F444W", image_mask="MASK335R", seps_as=seps,
                        stamp_px=21, nlambda=1, oversample=2, log=lambda s: None)
    assert np.allclose(g2["transmission"], t)
    assert json.loads(json.dumps(g2["meta"]))["version"] == S._CACHE_VERSION
