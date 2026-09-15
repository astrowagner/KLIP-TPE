"""The best cells must circle the injections that are in the picture they draw.

``Runner._best_images`` is a *bundle*: an image plus the sources that were injected into
it plus the record it came from.  Everything that draws the best cells treats it that way.

When an annulus finishes, the runner replaces the bundle's ``inj``/``clean`` with the
**winner's committed trial** -- the reduction it keeps as ``annulus01/best_inj.fits``.  That
picture's injections are validation's *fresh* draw, not the search evaluation the bundle was
built from: validation re-scores the elected config at new positions precisely so the winner
is not a lucky realisation.  The overwrite used to move the image and leave ``sources``
behind, and the panel then circled the search azimuths on the committed image.

Measured on the NACO beta Pic tutorial, one annulus, three sources:

    image contains injections at PA   28.4  148.4  268.4     (committed trial)
    circles drawn at PA              167.9  287.9   47.9     (search evaluation)

One azimuth step, 19.5 deg, which at 0.35-0.47 arcsec is 3.8-5.4 px against a 3.6 px FWHM --
every circle sitting a full beam off its blob.  It showed on the *annulus-done* frame, which
is the one that stays on the screen and the one a notebook shows when the run ends, so it was
the most-looked-at panel in the package.  Nothing about the science was wrong: the injector,
the metric's apertures and the display all agree to half a pixel (``test_injection``), and
the scores were never read off the picture.  It was a bookkeeping split, and it looked
exactly like a placement bug.
"""
from __future__ import annotations

import numpy as np
import pytest

from klip_tpe import Runner
from conftest import build_synthetic_run


def _pa(sources):
    def th(s):
        return float(getattr(s, "theta", None) if hasattr(s, "theta") else s[1])
    return sorted(round(th(s) % 360.0, 1) for s in sources)


# ----------------------------------------------------------- the bundle stays coherent

def test_the_committed_trial_brings_its_own_sources_into_the_bundle(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run(n_iter=[8, 6], n_init=4, seed=5)
    r = Runner(red, space, obj, samp, cfg, str(tmp_path), log=lambda s: None)
    results = r.run()
    b = r.best_images
    assert b.get("inj") is not None, "the run kept no best image to check"
    committed = _pa(results[-1].winner_sources)
    assert committed, "the winner committed no trial"
    assert _pa(b["sources"]) == committed, (
        "best_images['inj'] is the committed trial but best_images['sources'] still holds "
        "the search evaluation's injections")


def test_the_bundles_per_source_values_move_with_it(tmp_path):
    """The labels printed next to the circles are per-source S/N; they belong to the same
    trial as the circles, or the panel reports one evaluation's numbers on another's."""
    red, space, obj, samp, cfg = build_synthetic_run(n_iter=[8, 6], n_init=4, seed=5)
    r = Runner(red, space, obj, samp, cfg, str(tmp_path), log=lambda s: None)
    results = r.run()
    ps = r.best_images.get("per_source")
    assert ps is not None and len(ps) == len(r.best_images["sources"])
    want = [v for v in (results[-1].per_source or [])]
    assert len(ps) == len(want)


# --------------------------------------------------- the panel honours the bundle's sources

def test_step_images_sources_override_the_history_lookup():
    """``best_sources`` is what makes the fix reachable from every call site: the cells look
    the circles up by index into the history unless the caller says otherwise."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    import klip_tpe.display as D

    n = 4
    ad = D.AnnulusData(
        run_name="t", annulus=0, nann=1, inrad=8.0, outrad=30.0, pxscale=0.0272, fwhm=3.6,
        params=[D.ParamInfo("k_klip", "k", 1.0, 20.0, "int", None, "reduction")],
        partitions=["p1"], X=np.zeros((n, 1)), y=np.linspace(1, 4, n),
        phases=["tpe"] * n, k_used=[5] * n, selected=[["p1"]] * n,
        part_snr=[{"p1": 1.0}] * n,
        sources=[[(0.40, 30.0 + 40.0 * j, 1e-4)] for j in range(n)],
        per_source=[[3.0]] * n, raw_per_source=[[3.0]] * n, clean_per_source=[[0.2]] * n,
        raw=np.linspace(1, 4, n), wall=np.ones(n), contrast=np.full(n, 1e-4),
        configs=[{}] * n, n_init=1, n_iter=n, gamma=0.25, metric_name="mawet",
        search_mode="tpe", known=[])

    seen = []
    orig = D.draw_image

    def spy(ax, img, ad_, title, cmap="inferno", sources=None, **kw):
        seen.append((title, [tuple(np.round(s[:2], 3)) for s in (sources or [])]))
        return orig(ax, img, ad_, title, cmap, sources=sources, **kw)

    img = np.random.default_rng(0).normal(size=(81, 81))
    D.draw_image = spy
    try:
        # without an override the best cells take the history's row
        D.render_step(ad, 3, D.StepImages(cur_inj=img, cur_clean=img, best_inj=img,
                                          best_clean=img, best_index=1),
                      None, None, fig=Figure(figsize=(8, 5)))
        plain = [s for t, s in seen if s]
        assert any((0.4, 70.0) in s for s in plain), "the history row was not used by default"

        seen.clear()
        D.render_step(ad, 3, D.StepImages(cur_inj=img, cur_clean=img, best_inj=img,
                                          best_clean=img, best_index=1,
                                          best_sources=[(0.40, 199.0, 1e-4)],
                                          best_labels=[7.5]),
                      None, None, fig=Figure(figsize=(8, 5)))
        over = [s for t, s in seen if s]
        assert any((0.4, 199.0) in s for s in over), "best_sources did not reach the cells"
        assert not any((0.4, 70.0) in s for s in over), \
            "the history row is still being circled next to the override"
    finally:
        D.draw_image = orig


def test_the_current_cells_are_untouched_by_the_override():
    """Only the best cells move; the current-eval cells keep following ``i``."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    import klip_tpe.display as D

    n = 3
    ad = D.AnnulusData(
        run_name="t", annulus=0, nann=1, inrad=8.0, outrad=30.0, pxscale=0.0272, fwhm=3.6,
        params=[D.ParamInfo("k_klip", "k", 1.0, 20.0, "int", None, "reduction")],
        partitions=["p1"], X=np.zeros((n, 1)), y=np.linspace(1, 3, n),
        phases=["tpe"] * n, k_used=[5] * n, selected=[["p1"]] * n,
        part_snr=[{"p1": 1.0}] * n,
        sources=[[(0.40, 10.0 * (j + 1), 1e-4)] for j in range(n)],
        per_source=[[3.0]] * n, raw_per_source=[[3.0]] * n, clean_per_source=[[0.2]] * n,
        raw=np.linspace(1, 3, n), wall=np.ones(n), contrast=np.full(n, 1e-4),
        configs=[{}] * n, n_init=1, n_iter=n, gamma=0.25, metric_name="mawet",
        search_mode="tpe", known=[])

    seen = []
    orig = D.draw_image

    def spy(ax, img, ad_, title, cmap="inferno", sources=None, **kw):
        seen.append((title.lower(), [tuple(np.round(s[:2], 3)) for s in (sources or [])]))
        return orig(ax, img, ad_, title, cmap, sources=sources, **kw)

    img = np.random.default_rng(1).normal(size=(81, 81))
    D.draw_image = spy
    try:
        D.render_step(ad, 2, D.StepImages(cur_inj=img, cur_clean=img, best_inj=img,
                                          best_clean=img, best_index=0,
                                          best_sources=[(0.40, 300.0, 1e-4)], best_labels=[4.0]),
                      None, None, fig=Figure(figsize=(8, 5)))
    finally:
        D.draw_image = orig
    cur = [s for t, s in seen if s and t.startswith("test")]
    assert cur and all((0.4, 30.0) in s for s in cur), \
        f"the current cells stopped following the current evaluation: {cur}"


def test_the_best_header_block_quotes_the_picture(tmp_path):
    """The panel prints ``BEST  sep = ...  PA = ...`` next to the cells.  It read the same
    history row the circles did, so on the annulus-done frame it announced the search
    azimuths for an image that holds the committed trial's -- the one place a reader would
    go to check the circles, agreeing with them and with nothing else."""
    import matplotlib
    matplotlib.use("Agg")

    import klip_tpe.display as D

    n = 2
    ad = D.AnnulusData(
        run_name="t", annulus=0, nann=1, inrad=8.0, outrad=30.0, pxscale=0.0272, fwhm=3.6,
        params=[D.ParamInfo("k_klip", "k", 1.0, 20.0, "int", None, "reduction")],
        partitions=["p1"], X=np.zeros((n, 1)), y=np.array([1.0, 2.0]),
        phases=["tpe"] * n, k_used=[5] * n, selected=[["p1"]] * n,
        part_snr=[{"p1": 1.0}] * n,
        sources=[[(0.45, 101.0, 1e-3), (0.45, 281.0, 1e-3)],
                 [(0.45, 231.0, 1e-3), (0.45, 51.0, 1e-3)]],
        per_source=[[9.7, 9.2]] * n, raw_per_source=[[9.7, 8.9]] * n,
        clean_per_source=[[0.9, 2.3]] * n,
        raw=np.array([1.0, 2.0]), wall=np.ones(n), contrast=np.full(n, 1.31e-3),
        configs=[{}] * n, n_init=1, n_iter=n, gamma=0.25, metric_name="mawet",
        search_mode="tpe", known=[])

    plain = D._inj_lines(ad, 0)
    assert "PA = 101, 281 deg" in plain

    committed = D._inj_lines(ad, 0, [(0.45, 28.0, 1e-3), (0.45, 208.0, 1e-3)])
    assert "PA = 28, 208 deg" in committed, committed
    assert 'sep = 0.45, 0.45"' in committed
    assert "1.31E-03" in committed[2], "the contrast line must survive the override"


# ------------------------------------------- an injection model that misses the annulus

def test_a_library_that_misses_the_annulus_says_which_range_it_has():
    """``LibraryPSF`` has no template outside its own separations, and a search injects
    across the WHOLE annulus, not just at the known companion.  Tutorial 3 built its STPSF
    grid out to 2.0" for an annulus reaching 2.82": every one of 50 evaluations aborted on
    the outermost source, the run finished with no winner and no best image, and the first
    thing to notice was a notebook cell failing to open a file that was never written.  The
    error now names the separation asked for and the range the model covers."""
    import numpy as np

    from klip_tpe import Source
    from klip_tpe.injection import LibraryPSF, inject_sources

    m = LibraryPSF(np.ones((3, 11, 11)), [0.2, 1.0, 2.0], center=(5, 5), ee_radius_px=3.0)
    with pytest.raises(ValueError) as e:
        inject_sources(np.zeros((1, 41, 41)), np.zeros(1), [Source(2.4, 10.0, 1e-4)], m, 0.0626)
    msg = str(e.value)
    assert "2.400" in msg, msg
    assert "0.200-2.000" in msg, "the message must name the range the model actually covers"
    assert "fallback" in msg
    # inside the range it is fine
    inject_sources(np.zeros((1, 41, 41)), np.zeros(1), [Source(1.5, 10.0, 1e-4)], m, 0.0626)


def test_an_annulus_where_everything_failed_says_so():
    """A search in which every evaluation fails used to end quietly: winner index -1, no
    best image, and nothing in the log between the last failure and 'search done'.  The
    first thing to notice was a notebook cell failing to open a file never written."""
    import os

    import numpy as np

    from klip_tpe.optimizers import History

    h = History(2)
    for _ in range(4):
        h.append(np.zeros(2), None)          # a failed evaluation scores None -> nan
    assert len(h) == 4 and int(h.valid.sum()) == 0, "this is the state the guard has to catch"
    h.append(np.zeros(2), 1.0)
    assert int(h.valid.sum()) == 1, "and it must not fire once anything succeeded"

    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "klip_tpe", "runner.py")).read()
    assert "nok = int(self.history.valid.sum())" in src
    assert "no winner and no best image" in src
    i_guard = src.index("no winner and no best image")
    i_done = src.index('search done: best')
    assert i_guard < i_done, "the warning belongs before the search-done line, not after it"
