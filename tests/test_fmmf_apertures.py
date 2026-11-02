"""A forward model at each noise aperture (NEAR2 note, 2026-11-01, asked 2; Kevin Wagner,
2026-11-02: off unless asked, and when asked for every FM S/N of the run).

Turning the source's template to each aperture is exact only for a PSF and a coronagraph round
about the star.  Here each aperture of a source's ring takes the forward model of a source alone
on its pixel, from a forward-model pass of its own (:class:`klip_tpe.fmmf.FMTemplates`), and a
sample between two apertures, as each bin of the radial band, their two models weighted by
azimuth.  The scene for the physics is a core elongated along x and fixed in the image, as a
detector's PSF under little rotation is, with two self-subtraction lobes along the azimuth.

The templates are then right anywhere.  The noise is another matter.  The ring's job is the
noise of the source's own filter.  Turning carries that filter along the ring by rotation, exact
where the noise is round about the star.  Each aperture's own model carries it by the PSF's own
changes, exact where the noise changes as the PSF does.  The tests measure both.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from klip_tpe import cli
from klip_tpe.fmmf import (FMMFSNR, FMTemplates, _ApertureTemplates, aperture_templates, fm_kernel, fmmf_map,
                           ring_positions)
from klip_tpe.metrics import (Objective, Source, _PlacedFilter, _TurnedFilter, mawet_peak_snr, source_xy, star_center,
                              turn_kernel)
from klip_tpe.runner import RunConfig, Runner

from conftest import build_synthetic_run

N, FWHM, PX = 101, 4.0, 0.05
CX, CY = star_center((N, N))
SIZE = int(2 * np.ceil(2.5 * FWHM) + 1)
QUIET = lambda s: None  # noqa: E731


def _planet(x, y, amp=1.0, n=N, ax=2.2, lobe=0.4, lobe2=None, dl=1.6):
    """A core elongated along x by ``ax`` and two lobes along the azimuth through (x, y),
    ``dl`` FWHM away, ``lobe`` deep (and ``lobe2`` the second, by default as deep)."""
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    az = np.arctan2(y - c, x - c)
    tx, ty = -np.sin(az), np.cos(az)
    s, d = FWHM / 2.3548, dl * FWHM

    def g(x0, y0, a):
        return a * np.exp(-(xx - x0) ** 2 / (2 * (ax * s) ** 2) - (yy - y0) ** 2 / (2 * s * s))

    return amp * (g(x, y, 1.0) - g(x + d * tx, y + d * ty, lobe)
                  - g(x - d * tx, y - d * ty, lobe if lobe2 is None else lobe2))


def _fm_for(srcs, n=N, **shape):
    """The forward-model passes: one image per source, the source alone."""
    c = (n - 1) / 2.0
    out = []
    for s in srcs:
        x, y = source_xy([s.rho], [s.theta], PX, c, c)
        out.append(_planet(float(x[0]), float(y[0]), n=n, **shape))
    return out


def _stripes(seed):
    rng = np.random.default_rng(seed)
    w = ndimage.gaussian_filter1d(rng.standard_normal((N, N)), 6.0, axis=1)
    return w / w.std()


def _source(pa, rho=1.5):
    x, y = source_xy([rho], [pa], PX, CX, CY)
    return float(x[0]), float(y[0])


def test_a_placed_template_reads_the_convolution_with_it():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((61, 61))
    bank = []
    for _ in range(4):
        k = rng.standard_normal((9, 9))
        bank.append(k - k.mean())

    def at(x, y):
        return bank[int(round(x)) % 2 + 2 * (int(round(y)) % 2)]

    pf = _PlacedFilter(img, at, {})
    for x in (0, 1, 30, 60):
        for y in (0, 3, 30, 60):
            want = ndimage.convolve(img, at(x, y), mode="nearest")[y, x]
            assert pf.at(x, y) == pytest.approx(want, abs=1e-12)
    # the band: each bin with the template at its center on the circle
    c = 30.0
    yy, xx = np.nonzero(np.abs(np.hypot(*np.mgrid[0:61, 0:61] - c) - 20) <= 1.0)
    got = pf.band(xx, yy, c, c, 20.0, 0.3, 0.1)
    b = np.round(np.angle(np.exp(1j * (np.arctan2(yy - c, xx - c) - 0.3))) / 0.1).astype(int)
    for i in (0, len(xx) // 3, len(xx) - 1):
        a = 0.3 + b[i] * 0.1
        k = at(c + 20 * np.cos(a), c + 20 * np.sin(a))
        assert got[i] == pytest.approx(ndimage.convolve(img, k, mode="nearest")[yy[i], xx[i]], abs=1e-12)
    # no template there: None, and NaN in the band
    assert _PlacedFilter(img, lambda x, y: None, {}).at(5, 5) is None
    assert np.isnan(_PlacedFilter(img, lambda x, y: None, {}).band(xx[:3], yy[:3], c, c, 20.0, 0.0, 0.1)).all()


def test_templates_that_turn_give_the_turned_s_n():
    """A callable that turns the source's template gives what ``turn`` gives: the band to the
    bit, the ring to the angle of each aperture's pixel against that of its exact place."""
    xs, ys = _source(20.0)
    km = fm_kernel(_planet(xs, ys), xs, ys, SIZE)
    az = np.arctan2(ys - CY, xs - CX)
    im = _stripes(3)
    pk = [(int(round(xs)), int(round(ys)))]
    s1, d1 = mawet_peak_snr(im, [1.5], [20.0], PX, FWHM, kernel=[km], declip_nsig=None, turn=[True], peak_xy=pk,
                            return_details=True)
    at = [lambda x, y: turn_kernel(km, int(round((np.arctan2(y - CY, x - CX) - az) * 1e6)) * 1e-6)]
    s2, d2 = mawet_peak_snr(im, [1.5], [20.0], PX, FWHM, kernel=[km], declip_nsig=None, templates_at=at,
                            peak_xy=pk, return_details=True)
    assert d2[0]["sd_band"] == pytest.approx(d1[0]["sd_band"], rel=1e-6)
    assert s2[0] == pytest.approx(s1[0], rel=0.03)
    assert d2[0]["peak"] == d1[0]["peak"]                 # the source keeps its own map
    with pytest.raises(ValueError):
        mawet_peak_snr(im, [1.5, 1.5], [20.0, 200.0], PX, FWHM, templates_at=at)


def test_ring_positions_lay_out_the_metric_s_ring():
    az0, pos = ring_positions(1.5, 20.0, PX, FWHM, (N, N))
    xs, ys = _source(20.0)
    nap = int(np.floor(2 * np.pi * 30.0 / FWHM))
    assert len(pos) == nap and pos[0] == pytest.approx((xs, ys))
    assert az0 == pytest.approx(np.arctan2(ys - CY, xs - CX))
    r = [np.hypot(x - CX, y - CY) for x, y in pos]
    assert np.allclose(r, 30.0)
    a = np.unwrap([np.arctan2(y - CY, x - CX) for x, y in pos])
    assert np.allclose(np.diff(a), 2 * np.pi / nap)


def test_each_position_is_forward_modelled_once():
    asked = []

    def make(srcs):
        asked.append(list(srcs))
        return [None if s.rho > 2.4 else _planet(*_source(s.theta, s.rho)) for s in srcs]

    fa = FMTemplates(make, (N, N), PX, contrast=2e-4)
    xy = [_source(10.0), _source(50.0), _source(10.0), (CX + 49.0, CY)]
    st = fa.stamps(xy, SIZE)
    assert len(asked) == 1 and len(asked[0]) == 3, "a position asked twice is forward-modelled once"
    assert all(s.contrast == 2e-4 for s in asked[0])
    assert st[0] is st[2] and st[0].shape == (SIZE, SIZE) and st[3] is None
    assert (fa.n_passes, fa.n_empty) == (3, 1)
    fa.stamps(xy[:2], SIZE)
    assert len(asked) == 1, "positions seen before are not forward-modelled again"
    # the model source sits where it was asked, in the run's angle convention
    s = fa.source_at(*_source(77.0, 1.2))
    assert (s.rho, s.theta) == (pytest.approx(1.2), pytest.approx(77.0))
    s = FMTemplates(make, (N, N), PX, angle_convention="math").source_at(CX, CY + 10.0)
    assert s.theta == pytest.approx(90.0)


def _mid(p, q):
    """The point of the ring halfway in azimuth between pixels ``p`` and ``q``."""
    a = np.arctan2(p[1] - CY, p[0] - CX)
    m = a + np.angle(np.exp(1j * (np.arctan2(q[1] - CY, q[0] - CX) - a))) / 2
    return CX + 30 * np.cos(m), CY + 30 * np.sin(m)


def test_between_apertures_and_where_a_pass_gave_nothing():
    xs, ys = _source(20.0)
    km = fm_kernel(_planet(xs, ys), xs, ys, SIZE)
    az0, pos = ring_positions(1.5, 20.0, PX, FWHM, (N, N))
    pix = [(float(round(x)), float(round(y))) for x, y in pos]
    probe = FMTemplates(None, (N, N), PX)
    gone = {round(probe.source_at(*pix[q]).theta, 6) for q in (3, 4)}

    def make(srcs):
        return [None if round(float(s.theta), 6) in gone else _planet(*_source(s.theta, s.rho)) for s in srcs]

    fa = FMTemplates(make, (N, N), PX)
    at = aperture_templates(fa, [1.5], [20.0], [km], PX, FWHM, SIZE, (N, N))[0]
    assert isinstance(at, _ApertureTemplates) and at.n_turned == 2 and fa.n_empty == 2
    # on an aperture's pixel, the forward model of a source alone on it
    x, y = pix[6]
    own = fm_kernel(_planet(x, y), x, y, SIZE)
    assert np.allclose(at(x, y), own, atol=1e-9 * np.abs(own).max())
    # halfway between two, their two models alike, zero-meaned and normalized again
    m = [at.k[q] / float((at.k[q] ** 2).sum()) for q in (6, 7)]
    want = (m[0] + m[1]) / 2
    want = want - want.mean()
    want = want / float((want ** 2).sum())
    assert np.allclose(at(*_mid(pix[6], pix[7])), want, atol=2e-3 * np.abs(want).max())
    # between two without a model, the source's template turned there; beside one, its neighbor
    xm, ym = _mid(pix[3], pix[4])
    t = turn_kernel(km, np.angle(np.exp(1j * (np.arctan2(ym - CY, xm - CX) - az0))))
    assert np.allclose(at(xm, ym), t, atol=1e-6 * np.abs(t).max())
    assert at(*_mid(pix[4], pix[5])) is at.k[5] and at(*_mid(pix[2], pix[3])) is at.k[2]
    assert at(xs + 0.4, ys) is not None                    # next to the source, with its own template
    # a source without a template of its own takes none
    assert aperture_templates(fa, [1.5], [20.0], [None], PX, FWHM, SIZE, (N, N)) == [None]


def test_each_aperture_s_own_model_recovers_a_planet_anywhere():
    """The core keeps its orientation across the field, the lobes follow the azimuth.  The
    turned template recovers a planet 90 to 135 deg away at 0.70 to 0.84, the forward models at
    the apertures, weighted between two, whole to within 1 %."""
    xs, ys = _source(20.0)
    km = fm_kernel(_planet(xs, ys), xs, ys, SIZE)
    az0 = np.arctan2(ys - CY, xs - CX)
    at = aperture_templates(FMTemplates(_fm_for, (N, N), PX), [1.5], [20.0], [km], PX, FWHM, SIZE, (N, N))[0]
    for dphi in (30.0, 60.0, 90.0, 135.0, 180.0, -100.0):
        a = az0 + np.radians(dphi)
        ix, iy = int(round(CX + 30 * np.cos(a))), int(round(CY + 30 * np.sin(a)))
        im = _planet(ix, iy)
        assert _PlacedFilter(im, at, {}).at(ix, iy) == pytest.approx(1.0, abs=0.01), dphi
        if dphi in (90.0, 135.0):
            turned = _TurnedFilter(im, km, az0, {}).at(ix, iy, np.arctan2(iy - CY, ix - CX) - az0)
            assert turned < (0.9 if dphi == 90.0 else 0.8), dphi


_G = np.exp(-0.5 * (np.arange(-24, 25) / 6.0) ** 2)
_G /= _G.sum()


def _var_stripes(k):
    """The exact variance of filter ``k``'s value in :func:`_stripes`' noise."""
    rows = np.array([np.convolve(r, _G) for r in np.asarray(k, float)])
    return float((rows ** 2).sum() / (_G ** 2).sum())


def _var_white(k):
    return float((np.asarray(k, float) ** 2).sum())


@pytest.mark.parametrize("pa", [20.0, 65.0, 110.0])
def test_which_noise_each_choice_measures(pa):
    """The ring's job is the noise of the source's own filter.  Turning carries that filter
    along the ring by rotation, exact where the noise is round about the star.  Each aperture's
    own model carries it by the PSF's own changes, exact where the noise changes as the PSF does.
    Exact variances, as the RMS over the ring against the source's own: in noise striped along x,
    fixed in the image as the PSF is, the apertures' own models 0.98 to 0.99, turned 0.65 to
    0.83.  In white noise turned 1.00 to 1.01, the apertures' own 0.89 to 1.16, as their norm
    changes along the ring."""
    xs, ys = _source(pa)
    km = fm_kernel(_planet(xs, ys), xs, ys, SIZE)
    az0, pos = ring_positions(1.5, pa, PX, FWHM, (N, N))
    at = aperture_templates(FMTemplates(_fm_for, (N, N), PX), [1.5], [pa], [km], PX, FWHM, SIZE, (N, N))[0]
    turned = [turn_kernel(km, np.angle(np.exp(1j * (np.arctan2(y - CY, x - CX) - az0)))) for x, y in pos[1:]]
    own = at.k[1:]
    ratio = {}
    for name, var in (("stripes", _var_stripes), ("white", _var_white)):
        v0 = var(km)
        ratio[name] = [float(np.sqrt(np.mean([var(k) for k in ks]) / v0)) for ks in (turned, own)]
    assert abs(ratio["stripes"][1] - 1) < 0.03 < abs(ratio["stripes"][0] - 1) - 0.1
    assert ratio["white"][0] == pytest.approx(1.0, abs=0.02)
    assert abs(ratio["white"][1] - 1) > 0.08


def _arcs(seed, smear_deg=15.0):
    """Unit noise smeared along the azimuth, round about the star as ADI residuals are."""
    rng = np.random.default_rng(seed)
    w = ndimage.gaussian_filter1d(rng.standard_normal((80, 720)), smear_deg / 0.5, axis=1, mode="wrap")
    w = ndimage.gaussian_filter1d(w, 0.7, axis=0)
    w /= w.std()
    yy, xx = np.mgrid[0:N, 0:N]
    th = (np.degrees(np.arctan2(yy - CY, xx - CX)) % 360.0) / 0.5
    return ndimage.map_coordinates(w, [np.hypot(xx - CX, yy - CY), th], order=1, mode="wrap")


def test_a_round_psf_measures_the_same_either_way():
    """With a round core every aperture's own model is the source's turned: the ring's scatter
    comes out the same both ways, against residuals smeared along the azimuth, and far from the
    one map's."""
    xs, ys = _source(20.0)
    fm_for = lambda srcs: [_planet(*_source(s.theta, s.rho), ax=1.0, lobe=0.6, lobe2=0.3) for s in srcs]  # noqa: E731
    km = fm_kernel(_planet(xs, ys, ax=1.0, lobe=0.6, lobe2=0.3), xs, ys, SIZE)
    at = aperture_templates(FMTemplates(fm_for, (N, N), PX), [1.5], [20.0], [km], PX, FWHM, SIZE, (N, N))[0]
    pk = [(int(round(xs)), int(round(ys)))]
    sig = {"turn": [], "own": [], "map": []}
    for seed in range(12):
        im = _arcs(seed)
        for name, kw in (("turn", dict(turn=[True])), ("own", dict(templates_at=[at])), ("map", {})):
            _, d = mawet_peak_snr(im, [1.5], [20.0], PX, FWHM, kernel=[km], declip_nsig=None, peak_xy=pk,
                                  return_details=True, **kw)
            sig[name].append(d[0]["sigma"])
    assert np.mean(sig["own"]) == pytest.approx(np.mean(sig["turn"]), rel=0.02)
    assert np.mean(sig["map"]) > 1.5 * np.mean(sig["own"])


def test_the_metric_takes_the_passes_only_when_asked():
    xs, ys = _source(20.0)
    fm = _planet(xs, ys)
    img = 3.0 * fm + 0.3 * _stripes(1)
    fa = FMTemplates(_fm_for, (N, N), PX)
    off = FMMFSNR(pxscale=PX, fwhm=FWHM)
    a = off.per_source(img, None, [1.5], [20.0], fm=fm, fm_at=fa)[0]
    assert fa.n_passes == 0 and a == off.per_source(img, None, [1.5], [20.0], fm=fm)[0]
    on = FMMFSNR(pxscale=PX, fwhm=FWHM, fm_apertures=True)
    b = on.per_source(img, None, [1.5], [20.0], fm=fm, fm_at=fa)[0]
    nap = int(np.floor(2 * np.pi * 30.0 / FWHM))
    assert np.isfinite(b) and b != a
    assert fa.n_passes == nap - 1 and on.stats["fm_passes"] == nap - 1 and on.stats["fm_empty"] == 0
    # the clean term of the same score asks for no pass of its own
    on.per_source(0.3 * _stripes(2), None, [1.5], [20.0], fm=fm, fm_at=fa)
    assert fa.n_passes == nap - 1 and on.stats["fm_passes"] == nap - 1
    d = on.describe()
    assert d["fm_apertures"] is True and d["fm_aperture_passes"] == nap - 1
    assert FMMFSNR(pxscale=PX, fwhm=FWHM).describe()["fm_apertures"] is False
    # without a forward model of its own the source's fallback filter is round, and no pass is made
    fb = FMTemplates(_fm_for, (N, N), PX)
    on.per_source(img, None, [1.5], [20.0], fm=None, fm_at=fb)
    assert fb.n_passes == 0
    # the objective hands the passes to a metric that takes them, and to no other
    assert Objective(on)._fmkw(fm, fa) == {"fm": fm, "fm_at": fa}
    assert Objective(on)._fmkw(fm) == {"fm": fm}


def test_fmmf_map_takes_a_forward_model_per_bin():
    """The map's amplitude is in units of a template's peak, so a planet comes back at the
    value its own template gives it.  Each bin's own forward model gives that to 1 % at PA 45 and
    135; the turned template, 0.73 and 0.76 of it."""
    n = 121
    c = (n - 1) / 2.0
    shape = dict(ax=2.5, lobe=0.3, dl=1.0)
    x0, y0 = source_xy([1.5], [0.0], PX, c, c)
    fm = _planet(x0[0], y0[0], n=n, **shape)
    src = [Source(1.5, 0.0, 1.0)]
    for pa in (45.0, 135.0):
        x, y = source_xy([1.5], [pa], PX, c, c)
        ix, iy = int(round(x[0])), int(round(y[0]))
        im = _planet(ix, iy, n=n, **shape)
        k = fm_kernel(im, ix, iy, SIZE, lsq_norm=False)
        truth = ndimage.convolve(im, k / float((k * k).sum()), mode="nearest")[iy, ix]
        fa = FMTemplates(lambda srcs: _fm_for(srcs, n=n, **shape), (n, n), PX)
        out = fmmf_map(im, fm, src, PX, FWHM, n_theta=72, fm_at=fa)
        assert out["n_fm_bins"] == fa.n_passes > 0
        assert out["amplitude"][iy, ix] == pytest.approx(truth, rel=0.02), pa
        turned = fmmf_map(im, fm, src, PX, FWHM, n_theta=72)
        assert "n_fm_bins" not in turned
        assert turned["amplitude"][iy, ix] < 0.85 * truth, pa


# ------------------------------------------------------------------ the run
def _run(tmp_path, fm=True, **over):
    from klip_tpe import PartitionedReducer
    from klip_tpe.synthetic import make_synthetic_partitions

    _, space, _, samp, cfg = build_synthetic_run(n_iter=[3, 3], **over)
    red = PartitionedReducer(make_synthetic_partitions(["n1", "n2"], k_opts=[6, 12], fm=fm))
    metric = FMMFSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel)
    runner = Runner(red, space, Objective(metric, clean_subtract=True), samp, cfg, str(tmp_path), log=QUIET)
    runner.ia = 0
    asked = []
    orig = red.reduce_config

    def spy(cfg_, sources, k_scan=False, tag="", fm_sources=None, extras=False):
        asked.append((tag, None if sources is None else list(sources),
                      None if fm_sources is None else list(fm_sources)))
        return orig(cfg_, sources, k_scan, tag, fm_sources=fm_sources, extras=extras)
    red.reduce_config = spy
    return runner, space, metric, asked


def test_a_run_makes_a_pass_per_aperture_and_its_two_terms_share_them(tmp_path):
    runner, space, metric, asked = _run(tmp_path, fmmf_fm_apertures=True)
    assert metric.fm_apertures is True
    rec, inj, clean = runner.evaluate(space.default_vector(), "seed", contrast=3e-4)
    passes = [a for a in asked if a[0].startswith("fm_ap")]
    assert passes and all(a[1] is None and len(a[2]) == 1 for a in passes), "one model source per pass, no injection"
    want = []
    for s in rec.sources:
        _, pos = ring_positions(s[0], s[1], runner.pxscale, runner.fwhm, np.shape(inj.image))
        want += [(float(round(x)), float(round(y))) for x, y in pos[1:]]
    got = []
    for a in passes:
        s = a[2][0]
        x, y = source_xy([s.rho], [s.theta], runner.pxscale, *star_center(np.shape(inj.image)))
        got.append((round(float(x[0]), 6), round(float(y[0]), 6)))
    assert len(got) == len(want), "the injected and the clean terms share the passes"
    assert np.allclose(sorted(got), sorted(want), atol=1e-6)
    assert metric.stats["fm_passes"] == len(want)
    assert rec.score is None or np.isfinite(rec.score)


def test_a_run_without_klip_fm_takes_numerical_passes(tmp_path):
    runner, space, metric, asked = _run(tmp_path, fm=False, fmmf_fm_apertures=True)
    rec, inj, clean = runner.evaluate(space.default_vector(), "seed", contrast=3e-4)
    passes = [a for a in asked if a[0].startswith("fm_ap")]
    assert passes and all(a[0].endswith("_inj") and len(a[1]) == 1 and a[2] is None for a in passes)
    assert metric.stats["fm_passes"] == len(passes) and metric.stats["fm_empty"] == 0
    # without a clean image there is nothing to subtract: every aperture takes the turned template
    fa = runner._fm_for(inj, clean, cfg=space.decode(space.default_vector()), sources=[Source(*s) for s in rec.sources])
    assert isinstance(fa["fm_at"], FMTemplates)
    assert list(runner._fm_passes(space.decode(space.default_vector()), [Source(1.0, 30.0, 3e-4)], None)) == [None]


def test_off_by_default_and_for_runs_saved_before(tmp_path):
    assert RunConfig().fmmf_fm_apertures is False
    d = RunConfig(fmmf_fm_apertures=True).to_dict()
    assert RunConfig.from_dict(d).fmmf_fm_apertures is True
    d.pop("fmmf_fm_apertures")
    assert RunConfig.from_dict(d).fmmf_fm_apertures is False
    runner, space, metric, asked = _run(tmp_path)
    assert metric.fm_apertures is False
    rec, inj, clean = runner.evaluate(space.default_vector(), "seed", contrast=3e-4)
    assert not [a for a in asked if a[0].startswith("fm_ap")]
    assert "fm_at" not in runner._fm_for(inj, clean, cfg=space.decode(space.default_vector()))


def test_the_command_line_turns_it_on():
    base = ["near", "--root", "data"]
    assert "fmmf_fm_apertures" not in cli._protocol_config(cli._build_parser().parse_args(base))
    a = cli._build_parser().parse_args(base + ["--metric", "fmmf", "--fmmf-fm-apertures"])
    assert cli._protocol_config(a)["fmmf_fm_apertures"] is True
