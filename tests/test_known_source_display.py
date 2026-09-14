"""A known companion has to be visible on the panel, because it is nowhere else.

The objective never scores it -- that is the whole point: its recovered S/N is the held-out
check on a search that could not have tuned itself to it.  So it appears in no source list,
no per-source label and no score.  IDL circles it on the images and prints its S/N, and
without that the one number the run exists to produce is invisible while the run is going.
"""
import numpy as np
import pytest

from klip_tpe import display as D


def _ad(known=(), fwhm=4.0, pxscale=0.045, n=12):
    params = [D.ParamInfo("bin", "bin", 1.0, 20.0, "int", None, "reduction")]
    return D.AnnulusData(
        run_name="t", annulus=0, nann=1, inrad=8.0, outrad=30.0, pxscale=pxscale, fwhm=fwhm,
        params=params, partitions=["n1"], X=np.zeros((n, 1)), y=np.linspace(1, 5, n),
        phases=["tpe"] * n, k_used=[5] * n, selected=[["n1"]] * n,
        part_snr=[{"n1": 1.0}] * n, sources=[[(0.5, 30.0, 1e-4)]] * n,
        per_source=[[3.0]] * n, raw_per_source=[[3.0]] * n, clean_per_source=[[0.2]] * n,
        raw=np.linspace(1, 5, n), wall=np.ones(n), contrast=np.full(n, 1e-4),
        configs=[{}] * n, n_init=4, n_iter=n, gamma=0.25, metric_name="mawet",
        search_mode="tpe", known=list(known))


def _planted(ad, rho, pa, amp=8.0, size=120):
    """An image with a point source at the given (rho, PA)."""
    rng = np.random.default_rng(0)
    img = rng.normal(0, 1, (size, size))
    c = size / 2
    x = c + (rho / ad.pxscale) * np.cos(np.deg2rad(pa + 90.0))
    y = c + (rho / ad.pxscale) * np.sin(np.deg2rad(pa + 90.0))
    yy, xx = np.mgrid[:size, :size]
    return img + amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (ad.fwhm / 2.355) ** 2))


# ------------------------------------------------------------------ the measurement
def test_a_known_companion_is_measured_where_it_is():
    ad = _ad(known=[(0.5, 120.0)])
    snr = D.known_snr(ad, _planted(ad, 0.5, 120.0))
    assert snr and len(snr) == 1
    assert snr[0] > 5, f"a planted source came back at S/N {snr[0]:.1f}"


def test_a_source_scores_well_above_the_same_field_without_one():
    """The number has to respond to the thing it is measuring.

    An absolute threshold on synthetic noise would be testing this test's own field rather
    than the measurement, so the claim is relative: planting a source at the known position
    must raise its S/N well clear of the identical field without it.
    """
    ad = _ad(known=[(0.5, 120.0)])
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, (120, 120))
    c = 60.0
    x = c + (0.5 / ad.pxscale) * np.cos(np.deg2rad(120.0 + 90.0))
    y = c + (0.5 / ad.pxscale) * np.sin(np.deg2rad(120.0 + 90.0))
    yy, xx = np.mgrid[:120, :120]
    with_src = base + 12.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (ad.fwhm / 2.355) ** 2))
    empty = D.known_snr(ad, base)[0]
    full = D.known_snr(ad, with_src)[0]
    assert full > empty + 5.0, f"planting a source moved the S/N only {empty:.1f} -> {full:.1f}"


def test_nothing_known_measures_nothing():
    assert D.known_snr(_ad(), np.zeros((40, 40))) is None
    assert D.known_snr(_ad(known=[(0.5, 10.0)]), None) is None
    assert D.known_snr(_ad(known=[(0.5, 10.0)]), np.zeros(7)) is None


def test_the_measurement_is_cached_per_image():
    """Several cells of one panel must not each pay for it."""
    ad = _ad(known=[(0.5, 120.0)])
    img = _planted(ad, 0.5, 120.0)
    a = D.known_snr(ad, img)
    b = D.known_snr(ad, img)
    assert a == b
    assert D.known_snr(ad, _planted(ad, 0.5, 120.0, amp=40.0))[0] > a[0], "the cache is stale"


def test_a_freed_image_cannot_hand_its_address_to_the_next_one():
    """``id(img)`` is only a sound key while the array is alive.

    An array that is collected frees its address for the next allocation, and the cache then
    answered for the wrong image -- reproducible here by dropping every reference between
    measurements.  Each entry holds the array it measured, so this cannot happen.
    """
    import gc
    ad = _ad(known=[(0.5, 120.0)])
    seen = []
    for amp in (10.0, 40.0, 90.0):
        img = _planted(ad, 0.5, 120.0, amp=amp)
        seen.append(D.known_snr(ad, img)[0])
        del img
        gc.collect()                       # the next array may land at the same address
    assert seen[0] < seen[1] < seen[2], f"the cache answered for a freed image: {seen}"


def test_the_cache_is_bounded_and_releases_what_it_holds():
    """It keeps arrays alive by design, so it must not keep many."""
    ad = _ad(known=[(0.5, 120.0)])
    D._KNOWN_SNR_CACHE.clear()
    for i in range(D._KNOWN_SNR_CACHE_MAX * 3):
        D.known_snr(ad, _planted(ad, 0.5, 120.0, amp=10.0 + i))
    assert len(D._KNOWN_SNR_CACHE) <= D._KNOWN_SNR_CACHE_MAX


# ------------------------------------------------------------------ what gets drawn
def _circles(ad, img, known_labels=None, sources=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    fig = Figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    D.draw_image(ax, img, ad, "t", sources=sources, known_labels=known_labels)
    import matplotlib.colors as mc
    out = []
    for p in ax.patches:
        if isinstance(p, matplotlib.patches.Circle):
            out.append((p.get_edgecolor(), p.get_radius(), p.get_linestyle(), p.center))
    return out, [t.get_text() for t in ax.texts], mc


def test_a_known_companion_is_circled_on_every_image():
    ad = _ad(known=[(0.5, 120.0)])
    circles, _, mc = _circles(ad, _planted(ad, 0.5, 120.0))
    want = mc.to_rgba(D.KNOWN_COLOR)
    assert any(np.allclose(c[0], want) for c in circles), "the known companion was not circled"


def test_it_cannot_be_mistaken_for_an_injection():
    """Distinct colour, so a reader never reads a real companion as a test source."""
    ad = _ad(known=[(0.5, 120.0)])
    circles, _, mc = _circles(ad, _planted(ad, 0.5, 120.0), sources=[(0.5, 300.0, 1e-4)])
    known = [c for c in circles if np.allclose(c[0], mc.to_rgba(D.KNOWN_COLOR))]
    inj = [c for c in circles if np.allclose(c[0], mc.to_rgba(D.SRC_COLOR))]
    assert known and inj, (len(known), len(inj))
    assert known[0][2] != inj[0][2], "known and injected circles share a line style"
    assert known[0][1] > inj[0][1], "the known circle is not distinguishable by size either"
    r, g, b, _ = mc.to_rgba(D.KNOWN_COLOR)
    r2, g2, b2, _ = mc.to_rgba(D.SRC_COLOR)
    assert abs(r - r2) + abs(g - g2) + abs(b - b2) > 0.8, "the two colours are too close"


def test_its_snr_is_printed_when_the_caller_has_one():
    ad = _ad(known=[(0.5, 120.0)])
    img = _planted(ad, 0.5, 120.0)
    _, texts, _ = _circles(ad, img, known_labels=D.known_snr(ad, img))
    assert texts, "no label was drawn"
    assert any(t.replace("-", "").replace(".", "").isdigit() for t in texts), texts


def test_no_label_without_a_measurement():
    ad = _ad(known=[(0.5, 120.0)])
    circles, texts, _ = _circles(ad, _planted(ad, 0.5, 120.0), known_labels=None)
    assert circles, "still circle it even with no number"
    assert not texts, texts


# ------------------------------------------------------------------ where it comes from
def test_the_known_list_reaches_the_panel_from_the_objective():
    import types
    runner = types.SimpleNamespace(
        objective=types.SimpleNamespace(metric=types.SimpleNamespace(known=[(0.39, 300.5)])),
        sampler=types.SimpleNamespace(known=[]))
    assert D._runner_known(runner) == [(0.39, 300.5)]


def test_it_falls_back_to_the_sampler():
    import types
    runner = types.SimpleNamespace(
        objective=types.SimpleNamespace(metric=types.SimpleNamespace(known=[])),
        sampler=types.SimpleNamespace(known=[(0.8, 12.0)]))
    assert D._runner_known(runner) == [(0.8, 12.0)]


def test_none_declared_is_an_empty_list_not_a_failure():
    import types
    runner = types.SimpleNamespace(objective=types.SimpleNamespace(metric=object()),
                                   sampler=object())
    assert D._runner_known(runner) == []
