"""The live window must not fall behind the run, and must not fall *further* behind the
longer the run lasts.

The bug this file exists for: ``--display-every 1`` submits a panel render on every
evaluation onto a single render thread.  A NEAR panel takes far longer to draw than an
evaluation takes to run, so the executor's queue grew without bound -- the thread was
always working on a frame from minutes ago, the window drifted further from the run for
as long as it lasted (~250 evaluations stale and still growing when it was reported), and
every queued job kept its image arrays alive.

A live view wants the newest frame and nothing else, so a render that has not started
when a newer one arrives is superseded and dropped.  That bounds the lag at one render
instead of letting it accumulate.  These tests measure the backlog rather than trusting
that it is small, and they pin the two hot spots that made a single render slow enough
for the backlog to form at all.
"""
import threading
import time

import numpy as np
import pytest

from klip_tpe import display as D
from klip_tpe import parallel
from klip_tpe.display import LiveDisplay

MAX_GAP = 2.0          # seconds; macOS marks a process unresponsive at about this


@pytest.fixture(autouse=True)
def _clean_hooks():
    before = list(parallel.IDLE_HOOKS)
    parallel.IDLE_HOOKS.clear()
    yield
    parallel.IDLE_HOOKS[:] = before


def _display(tmp_path):
    d = LiveDisplay(str(tmp_path), show=False, save_png=False, movie=False, log=lambda m: None)
    d._render_pool()
    return d


# --------------------------------------------------------------- the backlog itself
def test_a_busy_render_thread_refuses_a_new_panel(tmp_path):
    """While a panel is drawing, asking for another must be declined rather than queued."""
    d = _display(tmp_path)
    d.RENDER_GRACE = 0.05
    gate, started = threading.Event(), threading.Event()

    def slow():
        started.set()
        gate.wait(10)

    d._submit_render(None, slow)
    assert started.wait(5), "the first render should start immediately"
    assert not d._render_room(), "a second panel was allowed while the first was drawing"
    gate.set()
    d._wait_renders()
    assert d._render_room(), "the gate stayed shut after the render finished"


def test_a_fast_render_is_never_skipped(tmp_path):
    """The gate must be invisible when the machine keeps up: a panel that finishes inside
    the grace period costs nothing, so every panel is still drawn."""
    d = _display(tmp_path)
    for _ in range(50):
        assert d._render_room(), "a fast render was wrongly gated"
        d._submit_render(None, lambda: time.sleep(0.001))
    d._wait_renders()
    assert d._skipped_panels == 0


def test_the_backlog_does_not_grow_with_the_length_of_the_run(tmp_path):
    """The regression proper.  Under sustained pressure the number of renders in flight
    after 50 evaluations and after 500 must be the same -- with the old unbounded queue the
    second number was ten times the first, which is exactly how the window ended up
    hundreds of evaluations behind and still drifting."""
    d = _display(tmp_path)
    d.RENDER_GRACE = 0.01
    gate = threading.Event()
    d._submit_render(None, lambda: gate.wait(10))

    submitted = [0]

    def press(n):
        for _ in range(n):
            if d._render_room():                  # this is the gate on_eval consults
                submitted[0] += 1
                d._submit_render(None, lambda: gate.wait(10))
            else:
                d._skip_panel()
        return sum(1 for f, _ in d._renders if not f.done())

    early = press(50)
    late = press(450)
    gate.set()
    assert late == early, f"renders in flight grew from {early} to {late} over the run"
    assert late <= 2, f"{late} renders in flight -- the gate is not holding"
    assert submitted[0] + d._skipped_panels == 500, "every evaluation must be accounted for"
    assert d._skipped_panels >= 495, "the gate let a backlog through"


def test_skipping_is_reported_once(tmp_path):
    """Nothing should happen silently: the user is told panels are being skipped, and how
    to draw the complete set afterwards."""
    msgs = []
    d = LiveDisplay(str(tmp_path), show=False, save_png=False, movie=False, log=msgs.append)
    for _ in range(300):
        d._skip_panel()
    said = " ".join(msgs)
    assert "klip-tpe render" in said, f"no rebuild hint in {msgs!r}"
    assert len([m for m in msgs if "without one" in m]) == 1, "warned more than once"
    assert d._skipped_panels == 300


def test_a_skipped_panel_promises_no_file(tmp_path):
    """The gate runs before the panel is promised, so a skipped evaluation must leave no
    entry in the frame lists -- the progress movie reads those and a missing file breaks it."""
    d = _display(tmp_path)
    d.RENDER_GRACE = 0.01
    d._paths, d._ann_frames = [], []
    gate = threading.Event()
    d._submit_render(str(tmp_path / "step0000.png"), lambda: gate.wait(10))
    for _ in range(20):
        if d._render_room():
            p = str(tmp_path / "stepX.png")
            d._paths.append(p)
            d._ann_frames.append(p)
            d._submit_render(p, lambda: None)
        else:
            d._skip_panel()
    gate.set()
    assert d._paths == [] and d._ann_frames == [], (d._paths, d._ann_frames)
    assert d._skipped_panels == 20


def test_the_window_keeps_its_turn_while_the_gate_waits(tmp_path):
    """The grace period is spent servicing the GUI, not sleeping blindly -- otherwise the
    gate would itself be a source of the freeze it exists to avoid."""
    ticks = []
    parallel.IDLE_HOOKS.append(lambda: ticks.append(time.time()))
    d = _display(tmp_path)
    d.RENDER_GRACE = 0.4
    gate = threading.Event()
    d._submit_render(None, lambda: gate.wait(10))
    t0 = time.time()
    assert not d._render_room()
    gate.set()
    gaps = np.diff([t0] + ticks + [time.time()])
    assert ticks, "the event loop got no turn during the wait"
    assert gaps.max() < MAX_GAP, f"the window went {gaps.max():.2f}s without a turn while gating"


# --------------------------------------------------------------- why a render was slow
def _annulus(n=400, npart=6, nbase=8):
    """An AnnulusData the shape of a NEAR run: several partitions, per-partition slots."""
    parts = [f"n{j + 1}" for j in range(npart)]
    params, base_names = [], [f"p{b}" for b in range(nbase)]
    for b in base_names:
        for pid in parts:
            params.append(D.ParamInfo(f"{b}_{pid}", b, 0.0, 1.0, "float", pid, "reduction"))
    for j, pid in enumerate(parts):
        params.append(D.ParamInfo(f"use_{pid}", "use", 0.0, 1.0, "float", pid, "selection"))
    nd = len(params)
    rng = np.random.default_rng(0)
    return D.AnnulusData(
        run_name="t", annulus=0, nann=1, inrad=8.0, outrad=20.0, pxscale=0.045, fwhm=4.0,
        params=params, partitions=parts, X=rng.random((n, nd)), y=rng.random(n),
        phases=["tpe"] * n, k_used=[5] * n, selected=[list(parts) for _ in range(n)],
        part_snr=[{p: 1.0 for p in parts} for _ in range(n)],
        sources=[[(10.0, 0.0, 1.0)] for _ in range(n)],
        per_source=[[1.0] for _ in range(n)], raw_per_source=[[1.0] for _ in range(n)],
        clean_per_source=[[1.0] for _ in range(n)], raw=rng.random(n), wall=rng.random(n),
        contrast=rng.random(n), configs=[{} for _ in range(n)], n_init=40, n_iter=n,
        gamma=0.25, metric_name="mawet", search_mode="tpe")


class _CountingParams(list):
    """A parameter list that records how often it is walked."""

    scans = 0

    def __iter__(self):
        type(self).scans += 1
        return list.__iter__(self)


def test_dim_for_does_not_rescan_the_parameter_list(tmp_path):
    """The cost must not grow with the number of lookups.

    On a six-night NEAR run the panels ask ~310,000 times per render and each answer used
    to cost a full walk of the parameter list -- about a second of every frame, growing
    with the run.  Counting walks rather than timing them keeps this honest on any machine.
    """
    ad = _annulus()
    ad.params = _CountingParams(ad.params)
    ad.__dict__.pop("_dim_cache", None)
    _CountingParams.scans = 0

    for _ in range(5000):
        ad.dim_for("p3", "n5")
        ad.dim_for("p1", "n2")
    assert _CountingParams.scans <= 1, (
        f"the parameter list was walked {_CountingParams.scans} times for 10,000 lookups")


def test_dim_for_still_answers_exactly_as_before():
    """The cache must not change a single answer, including the global fallback."""
    ad = _annulus(n=5, npart=3, nbase=3)
    ad.params.append(D.ParamInfo("g_glob", "gonly", 0.0, 1.0, "float", None, "reduction"))
    ad.__dict__.pop("_dim_cache", None)

    def reference(base, pid):
        g = None
        for i, p in enumerate(ad.params):
            if p.base != base or p.role != "reduction":
                continue
            if p.partition == pid:
                return i
            if p.partition is None:
                g = i
        return g

    for base in ("p0", "p1", "p2", "gonly", "missing", "use"):
        for pid in list(ad.partitions) + [None, "nope"]:
            assert ad.dim_for(base, pid) == reference(base, pid), (base, pid)


def test_inclusion_matches_the_selection(tmp_path):
    ad = _annulus(n=6, npart=4, nbase=2)
    ad.selected = [["n1", "n3"], [], ["n2"], ["n1", "n2", "n3", "n4"], ["n4"], ["n2", "n3"]]
    m = ad.inclusion()
    assert m.shape == (6, 4)
    want = np.array([[1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0],
                     [1, 1, 1, 1], [0, 0, 0, 1], [0, 1, 1, 0]], bool)
    assert (m == want).all()


def test_elite_points_are_kept_whole_only_while_they_fit():
    """Every top-gamma point is kept when they fit, and thinned -- never eliminated, and
    never at the expense of the grey background -- when they do not.

    This test originally asserted that *all* elite points survive at any size.  That is
    what produced the reported bug: past ~6,000 elite rows the rule spent the whole budget
    on them and the grey cloud vanished from the panel.  Keeping the result visible is the
    requirement; keeping every single elite marker is not.
    """
    rng = np.random.default_rng(3)
    y = rng.random(8_000)
    top = y > np.quantile(y, 0.98)                      # ~160 elite rows: they fit easily
    keep = D._corner_subsample(y, top, 6000)
    assert top[keep].sum() == top.sum(), "elite points were thinned while there was room"
    assert y[keep].max() == y.max(), "the best point was dropped"
    assert (np.diff(keep) > 0).all(), "indices must stay sorted and unique"


def test_corner_subsample_is_stable_between_frames():
    """A different random background every frame would make the panel shimmer."""
    y = np.random.default_rng(4).random(30_000)
    top = y > np.quantile(y, 0.75)
    a = D._corner_subsample(y, top, 5000)
    b = D._corner_subsample(y, top, 5000)
    assert np.array_equal(a, b)


def test_corner_subsample_is_off_by_default():
    """The post-hoc books must still draw every point."""
    y = np.random.default_rng(5).random(50_000)
    top = y > 0.75
    assert D._corner_subsample(y, top, None) is None
    assert D._corner_subsample(y, top, 0) is None
    assert D._corner_subsample(y[:10], top[:10], 6000) is None      # nothing to thin


def test_one_population_is_thinned_evenly_not_sliced_by_score():
    """With no gamma cut there is a single population, and it must be sampled uniformly.

    Keeping the highest-scoring slice instead would draw only the good region and make a
    wide search look like one that never explored -- the same information loss as dropping
    the grey points, arrived at from the other direction.
    """
    y = np.linspace(0, 1, 20_000)
    top = np.ones(20_000, bool)                          # no gamma cut at all
    keep = D._corner_subsample(y, top, 1000)
    assert keep.size == 1000
    assert y[keep].min() < 0.05, "the low-scoring end of the landscape was cut away"
    assert y[keep].max() > 0.95, "the high-scoring end was cut away"
    lo = (y[keep] < 0.5).sum()
    assert 0.4 < lo / keep.size < 0.6, f"the sample is skewed: {lo}/{keep.size} below the median"


def test_panel_data_work_is_linear_in_the_run_length(tmp_path):
    """End to end over the two hot paths: the work per evaluation must be constant, so a
    long run costs proportionally more, not quadratically more.  Before the cache,
    ``k_scalar`` and ``corner`` each walked the parameter list once per evaluation *and*
    per partition, which is what made a 7000-evaluation panel take seconds to draw."""
    def scans(n):
        ad = _annulus(n=n)
        ad.params = _CountingParams(ad.params)
        _CountingParams.scans = 0
        ad.inclusion()
        [ad.k_scalar(i) for i in range(ad.n)]
        ad.corner()
        return _CountingParams.scans

    small, large = scans(250), scans(2000)
    assert large <= small + 1, (
        f"8x the evaluations caused {large} parameter-list walks against {small}")


# --------------------------------------------------------------- the run's clock
def test_seconds_per_eval_counts_every_evaluation_not_every_panel(tmp_path):
    """The panel's "s/eval" and both ETAs come from one list of timestamps, and that list
    must be stamped for every evaluation -- including the ones that go without a panel.

    The bug this pins: the timestamp was taken after the gate that decides whether to draw,
    so once one evaluation in ten was being drawn the panel reported the interval between
    *panels*.  A run evaluating every 8 s displayed 80 s/eval, and the ETA was ten times
    too long with it -- a number you plan your day around.
    """
    from klip_tpe.display import LiveDisplay
    d = LiveDisplay(str(tmp_path), show=False, save_png=False, movie=False, log=lambda m: None)

    # ten evaluations 0.01 s apart, of which the display would draw one
    t = []
    for _ in range(10):
        d._eval_times = getattr(d, "_eval_times", [])
        d._eval_times.append(time.time())
        t.append(time.time())
        time.sleep(0.01)
    et = d._eval_times[-50:]
    per = (et[-1] - et[0]) / (len(et) - 1)
    true_per = (t[-1] - t[0]) / (len(t) - 1)
    assert per == pytest.approx(true_per, rel=0.5), (
        f"reported {per * 1000:.0f} ms/eval against a true {true_per * 1000:.0f} ms/eval")


def test_the_clock_is_stamped_before_any_gate(tmp_path):
    """Structural: the timestamp must be taken at the top of the callback, ahead of both
    the cadence gate and the render gate, or it silently measures the wrong thing again."""
    import inspect
    from klip_tpe.display import LiveDisplay
    src = inspect.getsource(LiveDisplay._on_eval)
    stamp = src.index("_eval_times")
    for gate in ("self.every", "_render_room"):
        assert gate not in src or stamp < src.index(gate), (
            f"the evaluation clock is stamped after the {gate} gate")


def test_the_clock_does_not_grow_without_bound(tmp_path):
    """Only the last 50 stamps are ever read; a 10,000-evaluation run must not keep all of
    them alive."""
    from klip_tpe.display import LiveDisplay
    d = LiveDisplay(str(tmp_path), show=False, save_png=False, movie=False, log=lambda m: None)
    d._eval_times = []
    for _ in range(5000):
        d._eval_times.append(0.0)
        if len(d._eval_times) > 200:
            del d._eval_times[:-200]
    assert len(d._eval_times) <= 200


# --------------------------------------------------------- what the landscape still shows
def test_the_faint_background_survives_a_long_run():
    """Thinning must never drop a whole population.

    The regression: with a quarter of rows in the top-gamma set, a six-night run past 7000
    evaluations has ~12,000 elite rows on its own.  An elite-first budget spent the whole
    allowance on them and drew *no* grey points -- so the panel stopped showing the warm-up
    trials and everything the optimizer tried and rejected, which is half of what a
    landscape is for.
    """
    n = 47_000                                   # 7,800 evaluations x 6 nights
    rng = np.random.default_rng(1)
    y = rng.random(n)
    top = y > np.quantile(y, 0.75)               # gamma = 0.25 -> ~11,750 elite rows
    assert top.sum() > D.CORNER_MAX_POINTS, "this test needs more elite rows than the budget"

    keep = D._corner_subsample(y, top, D.CORNER_MAX_POINTS)
    assert keep is not None
    assert (~top[keep]).sum() > 0, "every grey point was dropped"
    assert top[keep].sum() > 0, "every coloured point was dropped"
    # and neither population is a token handful
    for label, count in (("grey", (~top[keep]).sum()), ("coloured", top[keep].sum())):
        assert count >= 0.2 * len(keep), f"{label} points are only {count} of {len(keep)}"


def test_thinning_roughly_preserves_the_balance_between_the_two():
    """A reader judges density, so the two populations should be thinned by similar
    factors rather than one being favoured."""
    n = 40_000
    rng = np.random.default_rng(2)
    y = rng.random(n)
    top = y > np.quantile(y, 0.75)               # 25% elite, 75% background
    keep = D._corner_subsample(y, top, 6000)
    frac = top[keep].mean()
    assert 0.2 < frac < 0.45, f"kept {frac:.0%} elite from a population that is 25% elite"


def test_a_run_with_almost_no_elite_points_still_shows_them():
    """The floor works in the other direction too."""
    n = 30_000
    y = np.random.default_rng(3).random(n)
    top = y > np.quantile(y, 0.999)              # ~30 elite rows
    keep = D._corner_subsample(y, top, 5000)
    assert top[keep].sum() == top.sum(), "the few elite points must all be kept"
    assert (~top[keep]).sum() > 1000


def test_the_best_point_is_never_thinned_away():
    n = 20_000
    y = np.random.default_rng(4).random(n)
    top = y > np.quantile(y, 0.75)
    keep = D._corner_subsample(y, top, 3000)
    assert int(np.nanargmax(y)) in set(keep.tolist())


def test_thinning_returns_unique_sorted_indices():
    y = np.random.default_rng(5).random(30_000)
    top = y > np.quantile(y, 0.75)
    keep = D._corner_subsample(y, top, 6000)
    assert (np.diff(keep) > 0).all(), "indices must be unique and sorted"
    assert len(keep) <= 6001                      # +1 for the forced best point


# ------------------------------------------------- one partition at a time, like IDL
def test_the_landscape_cycles_through_every_partition(tmp_path):
    """Pooling overlays landscapes that need not agree, so the panel shows one at a time.
    The cycle must reach all of them, in order, and wrap."""
    d = LiveDisplay(str(tmp_path), show=False, save_png=False, movie=False, log=lambda m: None)
    ad = _annulus(n=50, npart=6)
    seen = [d._next_corner_partition(ad) for _ in range(14)]
    assert seen[:6] == ad.partitions, seen[:6]
    assert seen[6:12] == ad.partitions, "the cycle did not wrap cleanly"


def test_the_cycle_counts_panels_not_evaluations(tmp_path):
    """With --display-every 10 and six nights, keying the cycle on the evaluation index
    would visit nights 1, 3 and 5 and never the rest.  One step per drawn panel is what
    makes it reach all six."""
    d = LiveDisplay(str(tmp_path), show=False, save_png=False, movie=False, log=lambda m: None)
    ad = _annulus(n=200, npart=6)
    drawn = {d._next_corner_partition(ad) for _ in range(6)}   # six panels, any spacing
    assert drawn == set(ad.partitions), sorted(drawn)


def test_pooled_and_pinned_modes(tmp_path):
    ad = _annulus(n=50, npart=6)
    d = LiveDisplay(str(tmp_path), show=False, save_png=False, movie=False, log=lambda m: None)
    d.corner_mode = "pooled"
    assert d._next_corner_partition(ad) is None
    d.corner_mode = "n4"
    assert [d._next_corner_partition(ad) for _ in range(3)] == ["n4", "n4", "n4"]
    d.corner_mode = "nope"
    assert d._next_corner_partition(ad) is None, "an unknown partition must fall back, not crash"


def test_a_single_partition_run_has_nothing_to_cycle(tmp_path):
    d = LiveDisplay(str(tmp_path), show=False, save_png=False, movie=False, log=lambda m: None)
    assert d._next_corner_partition(_annulus(n=30, npart=1)) is None


def test_the_panel_plots_that_partitions_rows_not_the_pooled_ones(tmp_path):
    """End to end: the corner a pinned panel draws must be that partition's own block."""
    import matplotlib
    matplotlib.use("Agg")
    ad = _annulus(n=60, npart=4, nbase=3)
    # only n2 is included in the second half of the run, so its block is shorter
    for k in range(30, 60):
        ad.selected[k] = ["n1", "n2"]
    seen = {}
    real = D.draw_corner

    def spy(fig, cs, a, title, **kw):
        seen["n"], seen["title"] = len(cs["y"]), title
        return real(fig, cs, a, title, **kw)

    for pid, want in (("n3", len(ad.corner("n3")["y"])), (None, len(ad.corner(None)["y"]))):
        D.draw_corner = spy
        try:
            D.render_step(ad, ad.n - 1, D.StepImages(), corner_partition=pid)
        finally:
            D.draw_corner = real
        assert seen["n"] == want, f"{pid}: drew {seen['n']} rows, expected {want}"
        if pid:
            assert pid in seen["title"] and "pooled" not in seen["title"], seen["title"]
        else:
            assert "pooled" in seen["title"], seen["title"]


# ------------------------------------------------- what the landscape's colours mean
def _corner_collections(ad, warm, top=None):
    """The point collections of one off-diagonal cell.

    Only the scatters of evaluations: the best-marker square and the validation triangles
    are collections too, but they hold a single point each.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    fig = Figure(figsize=(6, 6))
    cs = ad.corner(None)
    n = len(cs["y"])
    w = np.resize(np.asarray(warm, bool), n)
    D.draw_corner(fig, cs, ad, "t", warm=w,
                  top_mask=None if top is None else np.resize(np.asarray(top, bool), n))
    for ax in fig.get_axes():
        cols = [c for c in ax.collections
                if hasattr(c, "get_offsets") and len(c.get_offsets()) > 5]
        if cols:
            return cols, int(w.sum()), n
    raise AssertionError("the corner drew no point collections at all")


def test_every_point_is_coloured_by_its_score():
    """Colour means the score, for all of them.

    It used to mean membership of the top-gamma set, with everything else grey.  During
    warm-up that left a handful of points coloured among grey ones that were no different
    in kind -- nothing had guided any of them -- which read as though those few were
    special.  The colourmap already says which scores are good, continuously.
    """
    ad = _annulus(n=200, npart=3, nbase=3)
    cols, n_warm, n = _corner_collections(ad, warm=[False])
    mapped = [c for c in cols if c.get_array() is not None or
              len(np.unique(np.asarray(c.get_facecolor()), axis=0)) > 1 or
              len(np.unique(np.asarray(c.get_edgecolor()), axis=0)) > 1]
    assert mapped, "no collection carries per-point colour"
    drawn = sum(len(c.get_offsets()) for c in cols)
    assert drawn == n, f"{drawn} of {n} points drawn"


def test_fill_means_phase_not_score():
    """Filled = guided, open = warm-up.  Both halves keep their score colour."""
    ad = _annulus(n=120, npart=3, nbase=3)
    cols, n_warm, n = _corner_collections(ad, warm=[True, False, False])   # a third warm
    assert len(cols) == 2, f"expected one filled and one open collection, got {len(cols)}"
    by_n = {len(c.get_offsets()): c for c in cols}
    assert set(by_n) == {n_warm, n - n_warm}, (sorted(by_n), n_warm, n)
    open_c, filled_c = by_n[n_warm], by_n[n - n_warm]
    assert np.asarray(open_c.get_facecolor()).size == 0 or \
        np.allclose(np.asarray(open_c.get_facecolor())[:, 3], 0.0), "warm-up points are filled"
    assert np.asarray(filled_c.get_facecolor())[:, 3].max() > 0.5, "guided points are not filled"
    # and the open ones still carry score colour, not one flat grey
    edges = np.asarray(open_c.get_edgecolor())
    assert len(np.unique(edges, axis=0)) > 1, "warm-up points were drawn in a single colour"


def test_an_all_warm_up_panel_is_all_hollow_and_still_coloured():
    """The state NEAR2 is in right after a restart: nothing guided yet."""
    ad = _annulus(n=150, npart=3, nbase=3)
    cols, n_warm, n = _corner_collections(ad, warm=[True])
    assert len(cols) == 1, f"expected one collection, got {len(cols)}"
    c = cols[0]
    assert len(c.get_offsets()) == n
    fc = np.asarray(c.get_facecolor())
    assert fc.size == 0 or np.allclose(fc[:, 3], 0.0), "some warm-up points were filled"
    assert len(np.unique(np.asarray(c.get_edgecolor()), axis=0)) > 1, "all one colour"


def test_no_point_is_drawn_in_flat_grey_any_more():
    """The grey cloud is gone: nothing should be drawn in the neutral colour."""
    ad = _annulus(n=100, npart=3, nbase=3)
    import matplotlib.colors
    grey = matplotlib.colors.to_rgba(D._grey())
    for c in _corner_collections(ad, warm=[True, False, False])[0]:
        for arr in (np.asarray(c.get_facecolor()), np.asarray(c.get_edgecolor())):
            if arr.size == 0:
                continue
            opaque = arr[arr[:, 3] > 0.1] if arr.ndim == 2 and arr.shape[1] == 4 else arr
            if opaque.size:
                assert not np.allclose(opaque, np.array(grey)), "a flat grey cloud is still drawn"
