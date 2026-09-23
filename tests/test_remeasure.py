"""``RunConfig.n_remeasure``: a trial scored as the mean of several fresh draws.

The objective's only stochastic input is the injected sources' azimuth anchor, and on
HIP 65426 F1140C repeated evaluations of identical configurations scatter with sd 0.84
against a useful range of 0.86-7.11.  Averaging draws is how the search stops ranking that
noise.  What has to hold: one trial is one history entry however many draws it took, the
score is the MEAN, and everything positional comes from the last draw so the panel shows a
real measurement rather than an average of images.
"""
from __future__ import annotations

import numpy as np
import pytest

from klip_tpe.runner import EvalRecord, RunCallback, RunConfig, Runner


def _rec(score, raw=None, wall=1.0, idx=0, srcs=((1.0, 0.0, 1e-4),), per=(5.0,)):
    return EvalRecord(annulus=0, index=idx, phase="tpe", x=[0.0], config={"params": {}},
                      sources=[tuple(s) for s in srcs], score=score,
                      raw_score=score if raw is None else raw,
                      per_source=list(per), raw_per_source=list(per), clean_per_source=None,
                      partition_snr={}, k_used=7, contrast=1e-4, wall_s=wall, meta={"keep": "me"})


def test_draws_average_and_the_last_one_supplies_the_positions():
    recs = [_rec(4.0, srcs=((1.0, 10.0, 1e-4),), per=(4.0,)),
            _rec(6.0, srcs=((1.0, 20.0, 1e-4),), per=(6.0,)),
            _rec(8.0, srcs=((1.0, 30.0, 1e-4),), per=(8.0,))]
    out = Runner._combine_draws(recs, recs[-1])
    assert out.score == pytest.approx(6.0), "score is the mean of the draws"
    assert out.raw_score == pytest.approx(6.0)
    assert out.sources == [(1.0, 30.0, 1e-4)], "positions come from the LAST draw"
    assert out.per_source == [8.0], "so do the per-source values the panel labels with"
    assert out.wall_s == pytest.approx(3.0), "the trial cost every draw"
    assert out.meta["keep"] == "me", "the last draw's own meta survives"
    assert out.meta["draw_scores"] == [4.0, 6.0, 8.0]
    assert out.meta["draw_n"] == 3
    assert out.meta["draw_sd"] == pytest.approx(2.0)
    assert out.meta["draw_spread"] == pytest.approx(4.0)
    assert out.meta["draw_failed"] == 0


def test_a_failed_draw_is_dropped_from_the_mean_not_counted_as_zero():
    """A reduction that failed is missing information, not a score of zero -- counting it as
    one would make any configuration that fails intermittently look catastrophic and teach
    the optimizer to avoid a neighbourhood for the wrong reason."""
    out = Runner._combine_draws([_rec(4.0), _rec(None), _rec(8.0)], _rec(8.0))
    assert out.score == pytest.approx(6.0)
    assert out.meta["draw_failed"] == 1
    assert out.meta["draw_scores"] == [4.0, None, 8.0]
    allfail = Runner._combine_draws([_rec(None), _rec(None)], _rec(None))
    assert allfail.score is None and allfail.meta["draw_failed"] == 2
    assert allfail.meta["draw_sd"] is None


def test_one_draw_is_the_old_behaviour_exactly():
    r = _rec(5.0)
    out = Runner._combine_draws([r], r)
    assert out.score == pytest.approx(5.0)
    assert out.meta["draw_n"] == 1 and out.meta["draw_sd"] is None


def test_averaging_shrinks_the_noise_by_root_n():
    """The claim the feature rests on, on synthetic draws with the measured sd."""
    rng = np.random.default_rng(4)
    sd1 = 0.84
    single, mean3 = [], []
    for _ in range(20000):
        d = rng.normal(0.0, sd1, 3)
        single.append(d[0])
        mean3.append(d.mean())
    got = np.std(single) / np.std(mean3)
    assert abs(got - np.sqrt(3)) < 0.05, f"mean of 3 should cut sd by sqrt(3), got {got:.3f}"


def test_evaluate_mean_makes_one_history_entry_and_reports_every_draw():
    """One trial, one entry.  Appending draws individually would let the optimizer read one
    configuration as n and would triple the apparent budget."""
    seen = []

    class _R:
        cfg = RunConfig(ann_edges=[6, 20], n_remeasure=3)
        evaluate_mean = Runner.evaluate_mean
        _combine_draws = staticmethod(Runner._combine_draws)

        def __init__(self):
            self.n = 0

        def evaluate(self, x, phase, tag="eval"):
            self.n += 1
            return _rec(float(self.n) * 2.0), object(), None

    r = _R()
    out, inj, clean = r.evaluate_mean(np.zeros(1), "tpe", tag="t",
                                      on_draw=lambda j, nd, rec, i_, c_: seen.append((j, nd, rec.score)))
    assert r.n == 3, "three draws were reduced"
    assert out.score == pytest.approx(4.0), "mean of 2, 4, 6"
    assert [s[0] for s in seen] == [0, 1, 2], "every draw was reported live, in order"
    assert all(s[1] == 3 for s in seen)
    assert [s[2] for s in seen] == [2.0, 4.0, 6.0]


def test_n_remeasure_1_does_not_wrap_or_retag():
    """With one draw the path must be the plain one -- no draw suffix on the tag, so eval
    image filenames and the saved products keep the names every earlier run used."""
    tags = []

    class _R:
        cfg = RunConfig(ann_edges=[6, 20], n_remeasure=1)
        evaluate_mean = Runner.evaluate_mean
        _combine_draws = staticmethod(Runner._combine_draws)

        def evaluate(self, x, phase, tag="eval"):
            tags.append(tag)
            return _rec(5.0), None, None

    out, _, _ = _R().evaluate_mean(np.zeros(1), "tpe", tag="a1_e7")
    assert tags == ["a1_e7"], f"tag must not gain a draw suffix, got {tags}"
    assert "draw_n" not in out.meta


def test_on_draw_is_optional_for_a_callback():
    """A callback written before this feature must keep working untouched."""
    class Old(RunCallback):
        pass

    assert hasattr(Old(), "on_draw")
    Old().on_draw(None, 0, 3, _rec(5.0), None, None)   # must not raise


# ----------------------------------------------------------------------------
# a collapsed range is a constant, not a dimension
# ----------------------------------------------------------------------------
def _fake_partitioned(parts, angles, nframes=80, pxscale=0.11, fwhm=3.34):
    """Minimum surface make_space reads: per-partition frames, angles and tags."""
    ang = np.asarray(angles, float)

    class _D:
        pass

    class _R:
        pxscale, fwhm, lam_over_d_px, truenorth = 0.0, 0.0, 0.0, 0.0
        angle_convention = "pa"

        def __init__(self):
            self.data = _D()
            self.data.nframes = int(nframes)
            self.data.cube = np.zeros((int(nframes), 40, 40), np.float32)
            self.data.angles = ang
            self.data.tags = []
            self.data.texp = 1.0

        def frame_tags(self, pid=None):
            return []

        def frame_angles(self, pid=None):
            return ang

    class _P:
        def __init__(self):
            self.reducers = {p: _R() for p in parts}
            self.pxscale, self.fwhm = pxscale, fwhm

        def partitions(self):
            return list(parts)

        def frame_tags(self, pid=None):
            return []

        def frame_angles(self, pid=None):
            return ang

    for r in (_R,):
        r.pxscale, r.fwhm, r.lam_over_d_px = pxscale, fwhm, 3.25
    return _P()


def test_a_point_range_parameter_is_pinned_not_searched():
    """NIRCam's cubes are short enough that make_space computed bin_range = (1, 1), so every
    HIP 65426 run has carried `bin : [1.000, 1.000]` as a searched dimension.  It must become
    a fixed value: the same number reaches the reducer, the sampler loses a coordinate."""
    from klip_tpe.instruments import near

    red = _fake_partitioned(["sci"], np.linspace(0, 1, 8), nframes=8, pxscale=0.06, fwhm=2.3)
    sp = near.make_space(red, bin_range=(1, 1), opt_framesel=False, search_angles=False,
                         selection=None, k_klip_max=18)
    names = {p.base or p.name for p in sp.params}
    assert "bin" not in names, f"bin must not be searched when its range is a point: {sorted(names)}"
    assert sp.fixed.get("bin") == 1, f"bin must still reach the reducer as 1, got {sp.fixed}"

    sp2 = near.make_space(red, bin_range=(1, 5), opt_framesel=False, search_angles=False,
                          selection=None, k_klip_max=18)
    assert "bin" in {p.base or p.name for p in sp2.params}
    assert "bin" not in sp2.fixed


def test_no_space_ships_a_constant_dimension():
    """The general form of the bug: any parameter whose lo equals its hi."""
    from klip_tpe.instruments import near

    red = _fake_partitioned(["roll1", "roll2"], np.linspace(0, 9, 80))
    for kw in ({}, {"search_angles": False}, {"bin_range": (1, 1)}, {"bin_range": (3, 3)}):
        sp = near.make_space(red, opt_framesel=False, k_klip_max=20, **kw)
        bad = [(p.name, p.lo, p.hi) for p in sp.params if float(p.hi) <= float(p.lo)]
        assert not bad, f"constant dimensions shipped with {kw}: {bad}"


def test_search_angles_false_removes_both_inert_dimensions():
    """angsep and anglemax are inert on two-roll data: within a roll every science frame
    shares a position angle, so reference_mask's dpa is identically zero, anglemax always
    passes, and any angsep > 0 empties the mask and falls back to the angsep = 0 set.  On a
    two-partition MIRI run that is 4 of 12 dimensions."""
    from klip_tpe.instruments import near

    red = _fake_partitioned(["roll1", "roll2"], np.full(80, 108.0))   # one roll: no rotation
    on = {p.base or p.name for p in near.make_space(red, opt_framesel=False, k_klip_max=20).params}
    off = {p.base or p.name for p in near.make_space(red, opt_framesel=False, k_klip_max=20,
                                                     search_angles=False).params}
    assert {"angsep", "anglemax"} <= on
    assert not ({"angsep", "anglemax"} & off)
    assert {"k_klip", "filter", "n_ang"} <= off, "only the inert pair should go"


def test_a_pa_span_between_5_and_20_degrees_no_longer_crashes_make_space():
    """Regression: anglemax_hi came from the sequence's PA span while the parameter's floor
    stayed at 20, so any span in (5, 20] built Param(20, span) and raised `anglemax: hi < lo`
    before a single evaluation ran.  Such a span cannot constrain anything, so the parameter
    is pinned open."""
    from klip_tpe.instruments import near

    for span in (6.0, 9.4, 12.0, 19.9):
        red = _fake_partitioned(["p"], np.linspace(0.0, span, 40), nframes=40)
        sp = near.make_space(red, opt_framesel=False, k_klip_max=20, selection=None)
        assert "anglemax" not in {p.base or p.name for p in sp.params}, f"span {span}"
        assert sp.fixed.get("anglemax") == 360.0, f"span {span}: must be pinned open, got {sp.fixed}"
        assert "angsep" in {p.base or p.name for p in sp.params}, "angsep is a separate question"
    # a span with real room keeps searching it
    red = _fake_partitioned(["p"], np.linspace(0.0, 90.0, 40), nframes=40)
    sp = near.make_space(red, opt_framesel=False, k_klip_max=20, selection=None)
    assert "anglemax" in {p.base or p.name for p in sp.params}


# ----------------------------------------------------------------------------
# the draws as an error bar
# ----------------------------------------------------------------------------
def _axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots()[1]


def test_draw_scores_of_reads_the_meta_and_ignores_single_draws():
    from klip_tpe.plots import draw_scores_of

    got = draw_scores_of([{"meta": {"draw_scores": [4.0, 6.0, 5.0]}},
                          {"meta": {"draw_scores": [5.0]}},          # one draw: no range to show
                          {"meta": {"draw_scores": [3.0, None, 7.0]}},  # a failed draw drops out
                          {"meta": {}},
                          {}])
    assert got == [[4.0, 6.0, 5.0], None, [3.0, 7.0], None, None]


def test_draw_ranges_spans_min_to_max_and_no_ops_without_draws():
    from klip_tpe.plots import draw_ranges

    ax = _axes()
    n = draw_ranges(ax, [1, 2, 3], [[4.0, 6.0, 5.0], None, [1.0, 2.0]])
    assert n == 2, "one bar per trial that had more than one draw"
    seg = [c for c in ax.collections if hasattr(c, "get_segments")][-1].get_segments()
    spans = sorted((float(s[0][1]), float(s[1][1])) for s in seg)
    assert spans == [(1.0, 2.0), (4.0, 6.0)], f"bars must span min..max, got {spans}"
    # a single-draw run must add nothing at all -- no bar, no legend entry
    ax2 = _axes()
    assert draw_ranges(ax2, [1, 2], [None, None]) == 0
    assert not ax2.collections
    assert ax2.get_legend_handles_labels()[1] == []


def test_draw_ranges_labels_the_draw_count_and_sits_under_the_points():
    from klip_tpe.plots import draw_ranges

    ax = _axes()
    draw_ranges(ax, [1, 2], [[4.0, 6.0, 5.0], [4.5, 5.5, 5.0]])
    labels = ax.get_legend_handles_labels()[1]
    assert any("min-max of 3" in s for s in labels), labels
    coll = [c for c in ax.collections if hasattr(c, "get_segments")][-1]
    assert coll.get_zorder() < 3, "the bar must sit under the score points, not over them"


def test_each_bar_takes_its_own_point_colour():
    """The bars belong to their points, so they carry the phase colour rather than one grey
    band across every phase."""
    from klip_tpe.plots import draw_ranges

    ax = _axes()
    n = draw_ranges(ax, [1, 2, 3], [[4.0, 6.0], [5.0, 7.0], [1.0, 3.0]],
                    colors=["#111111", "#222222", "#111111"])
    assert n == 3
    colls = [c for c in ax.collections if hasattr(c, "get_segments")]
    # grouped by colour: two calls for two distinct colours, not one per trial
    assert len(colls) == 2, f"expected one vlines per distinct colour, got {len(colls)}"
    from matplotlib.colors import to_hex
    got = {}
    for c in colls:
        col = to_hex(c.get_colors()[0])
        got[col] = len(c.get_segments())
    assert got == {"#111111": 2, "#222222": 1}, got


def test_the_legend_gets_one_neutral_entry_not_one_per_phase():
    """Per-colour entries would duplicate the phase legend the points already carry."""
    from klip_tpe.plots import draw_ranges

    ax = _axes()
    draw_ranges(ax, [1, 2, 3, 4], [[4.0, 6.0]] * 4,
                colors=["#111111", "#222222", "#333333", "#444444"])
    labels = ax.get_legend_handles_labels()[1]
    assert sum(1 for s in labels if "min-max" in s) == 1, labels


def test_a_short_colour_list_does_not_misalign_the_bars():
    """Defensive: the caller builds colours from phases and draws from meta independently."""
    from klip_tpe.plots import draw_ranges

    ax = _axes()
    assert draw_ranges(ax, [1, 2, 3], [[4.0, 6.0], [5.0, 7.0], [1.0, 3.0]],
                       colors=["#111111"]) == 3


def test_draw_ranges_tolerates_a_wholly_failed_trial():
    from klip_tpe.plots import draw_ranges

    ax = _axes()
    assert draw_ranges(ax, [1, 2], [[None, None], [2.0, 4.0]]) == 1


def test_the_panel_trace_draws_the_range_when_the_run_remeasured():
    """Wiring check for the live panel: the bars come from AnnulusData.extra['draws'],
    which annulus_from_run fills from each record's meta."""
    from klip_tpe import display as dm

    calls = []
    orig = dm.__dict__.get("draw_ranges")

    import klip_tpe.plots as pl
    real = pl.draw_ranges

    def spy(ax, x, draws, **kw):
        calls.append([list(d) if d else None for d in draws])
        return real(ax, x, draws, **kw)

    pl.draw_ranges = spy
    try:
        ad = _min_annulus_data(dm, y=[4.0, 6.0], draws=[{"draw_scores": [3.0, 5.0]},
                                                        {"draw_scores": [5.0, 7.0]}])
        dm.panel_trace(_axes(), ad)
    finally:
        pl.draw_ranges = real
        if orig is not None:
            dm.__dict__["draw_ranges"] = orig
    assert calls, "panel_trace did not consult draw_ranges"
    assert calls[0] == [[3.0, 5.0], [5.0, 7.0]]


def _min_annulus_data(dm, y, draws):
    """Smallest AnnulusData panel_trace will accept."""
    n = len(y)
    return dm.AnnulusData(
        run_name="t", annulus=0, nann=1, inrad=6.0, outrad=20.0, pxscale=0.11, fwhm=3.3,
        params=[dm.ParamInfo("k_klip", "k_klip", 1, 10, "int", None, "reduction")],
        partitions=["p"], X=np.zeros((n, 1)), y=np.array(y, float), phases=["tpe"] * n,
        k_used=[5] * n, selected=[["p"]] * n, part_snr=[{}] * n, sources=[[]] * n,
        per_source=[[]] * n, raw_per_source=[[]] * n, clean_per_source=[None] * n,
        raw=np.array(y, float), wall=np.ones(n), contrast=np.full(n, 1e-4),
        configs=[{}] * n, n_init=1, n_iter=n, gamma=0.25, metric_name="m", search_mode="tpe",
        seed_default=False, extra={"draws": draws})


def test_the_panel_render_step_actually_draws_gets_the_bars_too():
    """The bug this guards: the bars went into ``panel_trace``, which only
    ``render_step_classic`` calls.  The live window and the step PDFs go through
    ``render_step`` -> ``panel_convergence_idl``, so neither showed anything.  Both panels
    are checked here, by counting the vline segments each one adds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from klip_tpe import display as dm

    draws = [{"draw_scores": [4.0, 6.0, 5.0], "draw_n": 3},
             {"draw_scores": [5.0, 7.0, 6.0], "draw_n": 3},
             {"draw_scores": [6.5, 6.9, 6.7], "draw_n": 3}]
    ad = _min_annulus_data(dm, y=[5.0, 6.0, 6.7], draws=draws)
    for fn in (dm.panel_convergence_idl, dm.panel_trace):
        ax = plt.subplots()[1]
        fn(ax, ad)
        segs = sum(len(c.get_segments()) for c in ax.collections if hasattr(c, "get_segments"))
        assert segs >= 3, f"{fn.__name__} drew {segs} range bars for 3 remeasured trials"
        plt.close(ax.figure)


def test_the_convergence_panel_widens_its_y_limits_for_the_bars():
    """A clipped error bar is worse than none: the panel sets its own y-range from the means,
    and the draws reach past them by construction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from klip_tpe import display as dm

    # one draw far above every mean
    ad = _min_annulus_data(dm, y=[5.0, 5.0], draws=[{"draw_scores": [4.0, 6.0], "draw_n": 2},
                                                    {"draw_scores": [1.0, 12.0], "draw_n": 2}])
    ax = plt.subplots()[1]
    dm.panel_convergence_idl(ax, ad)
    lo, hi = ax.get_ylim()
    assert hi >= 12.0, f"upper bound {hi:.2f} clips a draw at 12"
    assert lo <= 1.0, f"lower bound {lo:.2f} clips a draw at 1"
    plt.close(ax.figure)


def test_a_single_draw_run_leaves_the_convergence_panel_untouched():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from klip_tpe import display as dm

    ad = _min_annulus_data(dm, y=[5.0, 6.0], draws=[{}, {}])
    ax = plt.subplots()[1]
    dm.panel_convergence_idl(ax, ad)
    segs = sum(len(c.get_segments()) for c in ax.collections if hasattr(c, "get_segments"))
    assert segs == 0, f"drew {segs} bars for a run with one draw per trial"
    plt.close(ax.figure)
