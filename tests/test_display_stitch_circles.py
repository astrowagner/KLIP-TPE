"""The best S/N cell's title gives the values circled in its picture, and the README frame's
stitch circles stay with the stitch.

RX J0534's three-annulus README run (2026-11-10) ended on a frame whose best S/N cell read
"Best mean S/N  orig 12.12 / corr 11.41" -- the search trial's scores -- over two sources
circled 10.0 and 10.5 from the winner's committed trial, validation's fresh draw.  The same
frame had two empty circles in "Stitched SNR (with inj)", at 1.16" and PA 131 / 311 deg: the
search trial's sources, beside the winners' committed trials that the stitch holds.  0e9c172
stopped the annulus-done frame adding the search trial's; the tests here hold that for the
README frame's case.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

import klip_tpe.display as D
from klip_tpe.display import LiveDisplay, _same_sources
from klip_tpe.runner import _write_fits

#: the README run's annuli: two sources each, the winners' committed trials
WINNERS = [[(0.390, 224.3, 1.6e-4), (0.390, 44.3, 1.6e-4)],
           [(0.774, 193.8, 1.1e-4), (0.774, 13.8, 1.1e-4)],
           [(1.157, 203.9, 3.8e-5), (1.157, 23.9, 3.8e-5)]]
#: annulus 3's search trial of the same configuration
SEARCH = [(1.157, 130.6, 3.8e-5), (1.157, 310.6, 3.8e-5)]


def _circles(run_dir, done, best):
    """The stitch circles of a display on ``run_dir`` whose runner has finished annuli
    ``done`` and the running best ``best`` = (annulus, sources)."""
    img = np.random.default_rng(0).normal(size=(64, 64))
    for name in ("klip_stitched_running.fits", "klip_stitched_running_inj.fits"):
        _write_fits(str(run_dir / name), img, {"RUNNING": 1})
    res = [types.SimpleNamespace(annulus=a, winner_sources=list(WINNERS[a])) for a in done]
    rec = types.SimpleNamespace(annulus=best[0], sources=list(best[1]))
    runner = types.SimpleNamespace(ia=best[0], results=res, best_images={"record": rec})
    out = LiveDisplay(str(run_dir), save_png=False, movie=False, pdf_every=0)._running_stitch(runner)
    assert out.get("stitch_inj") is not None and out.get("stitch_clean") is not None
    return out.get("stitch_sources") or []


@pytest.mark.parametrize("done, extra", [([0, 1, 2], False), ([0, 1], True)])
def test_the_running_best_is_circled_only_while_its_annulus_runs(tmp_path, done, extra):
    """Done (the README frame): the three winners' committed trials and not the search trial."""
    got = _circles(tmp_path, done, (2, SEARCH))
    want = [s for a in done for s in WINNERS[a]] + (list(SEARCH) if extra else [])
    assert _same_sources(got, want), got


@pytest.fixture(scope="module")
def two_annuli(tmp_path_factory):
    """A finished two-annulus synthetic run: its directory, runner and results."""
    from klip_tpe import Runner
    from conftest import build_synthetic_run
    d = tmp_path_factory.mktemp("circles_run")
    red, space, obj, samp, cfg = build_synthetic_run(n_iter=[8, 6], n_init=4, seed=5)
    r = Runner(red, space, obj, samp, cfg, str(d), log=lambda s: None)
    return d, r, r.run()


def test_a_real_run_circles_its_stitchs_own_injections(two_annuli):
    """End to end: after the run the stitch's circles are the two winners' committed trials,
    which is what the stitch's header lists."""
    d, r, results = two_annuli
    held = D._header_sources(str(d / "klip_stitched_running_inj.fits"))
    assert held, "the run's running stitch lists no injections"
    winners = [tuple(s) for res in results for s in res.winner_sources]
    assert _same_sources(held, winners), (held, winners)
    runner = types.SimpleNamespace(ia=max(int(getattr(r, "ia", 1)), 1), results=r.results, best_images=r.best_images)
    out = LiveDisplay(str(d), save_png=False, movie=False, pdf_every=0)._running_stitch(runner)
    assert _same_sources(out.get("stitch_sources") or [], winners), out.get("stitch_sources")


def test_the_redrawn_done_frame_names_the_validation_trial(two_annuli, tmp_path):
    """``klip-tpe render`` draws an annulus' last frame as the live annulus-done frame did,
    the committed trial in the best cells: its S/N title gives that trial's circled values."""
    import matplotlib
    matplotlib.use("Agg")
    d, r, results = two_annuli
    assert results[-1].validated, "the synthetic run's last annulus was not validated"
    seen = []
    orig = D.draw_image

    def spy(ax, img, ad_, title, cmap="inferno", **k):
        seen.append(title)
        return orig(ax, img, ad_, title, cmap, **k)

    D.draw_image = spy
    try:
        D.render_steps(str(d), annuli=[1], every=1000, out_dir=str(tmp_path / "rebuilt"), movie=False)
    finally:
        D.draw_image = orig
    best = [t for t in seen if t.startswith("Best") and ("S/N" in t or "SNR" in t)]
    assert best, seen
    assert best[-1].startswith("Best (validation trial) "), best[-1]
    assert "orig" not in best[-1], best[-1]


# ------------------------------------------------------------------ the best S/N cell's title

#: the winner's search trial and its committed (validation) trial, scaled into a small frame
T_SEARCH = [(0.45, 130.6, 1e-4), (0.45, 310.6, 1e-4)]
T_VALID = [(0.45, 203.9, 1e-4), (0.45, 23.9, 1e-4)]


def _ad():
    n = 3
    return D.AnnulusData(
        run_name="t", annulus=0, nann=1, inrad=8.0, outrad=30.0, pxscale=0.0272, fwhm=3.6,
        params=[D.ParamInfo("k_klip", "k", 1.0, 20.0, "int", None, "reduction")],
        partitions=["p1"], X=np.zeros((n, 1)), y=np.array([9.0, 11.41, 10.0]),
        phases=["tpe"] * n, k_used=[5] * n, selected=[["p1"]] * n,
        part_snr=[{"p1": 1.0}] * n,
        sources=[list(T_SEARCH)] * n,
        per_source=[[12.0, 12.24]] * n, raw_per_source=[[12.0, 12.24]] * n, clean_per_source=[[0.2, 0.3]] * n,
        raw=np.array([9.5, 12.12, 10.5]), wall=np.ones(n), contrast=np.full(n, 1e-4),
        configs=[{}] * n, n_init=1, n_iter=n, gamma=0.25, metric_name="mawet",
        search_mode="tpe", known=[])


def _best_snr_title(**kw):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    seen = []
    orig = D.draw_image

    def spy(ax, img, ad_, title, cmap="inferno", **k):
        seen.append(title)
        return orig(ax, img, ad_, title, cmap, **k)

    img = np.random.default_rng(2).normal(size=(81, 81))
    D.draw_image = spy
    try:
        D.render_step(_ad(), 2, D.StepImages(cur_inj=img, cur_clean=img, best_inj=img, best_clean=img,
                                             best_index=1, **kw),
                      None, None, fig=Figure(figsize=(8, 5)))
    finally:
        D.draw_image = orig
    t = [s for s in seen if s.startswith("Best") and "S/N" in s]
    assert len(t) == 1, seen
    return t[0]


def test_the_validated_frames_best_title_gives_the_circled_trial():
    t = _best_snr_title(best_sources=T_VALID, best_labels=[10.0, 10.5], best_trial="validation trial")
    assert t == "Best (validation trial) mean S/N = 10.25", t


def test_a_picture_of_the_search_trial_keeps_its_scores():
    t = _best_snr_title(best_sources=list(reversed(T_SEARCH)), best_labels=[12.24, 12.0],
                        best_trial="validation trial")
    assert t == "Best mean S/N  orig 12.12 / corr 11.41", t


def test_another_trial_without_a_name_still_gives_its_circled_values():
    t = _best_snr_title(best_sources=T_VALID, best_labels=[10.0, 10.5])
    assert t == "Best mean S/N = 10.25", t
