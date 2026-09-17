"""Live display + post-hoc books on a tiny synthetic run (2 partitions, 14 evals)."""
import os

import numpy as np
import pytest

# end-to-end on synthetic data (~5 min: a real run rendering every panel and PDF book on two cores); `-m \"not slow\"` skips it
pytestmark = pytest.mark.slow

from klip_tpe import CalibrationConfig, Runner, ValidationConfig
from klip_tpe import plots
from klip_tpe.display import (AnnulusData, LiveDisplay, StepImages, annulus_from_run, plot_annulus_books,
                              render_step, render_steps)

from conftest import build_synthetic_run

QUIET = (lambda s: None)


def _size(path):
    assert os.path.exists(path), path
    return os.path.getsize(path)


@pytest.fixture(scope="module")
def live_run(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("live"))
    red, space, obj, samp, cfg = build_synthetic_run(
        ann_edges=[8, 30], n_iter=14, n_init=5,
        validation=ValidationConfig(n_top=2, n_valid=2),
        calibration=CalibrationConfig(recal_budget=0, n_remeasure=1))
    logs = []
    disp = LiveDisplay(d, every=1, pdf_every=5, dpi=70)
    runner = Runner(red, space, obj, samp, cfg, d, log=logs.append, callbacks=[disp])
    results = runner.run()
    return d, runner, results, logs, disp


def test_live_display_products(live_run):
    d, runner, results, logs, disp = live_run
    # never a swallowed callback failure
    assert not [l for l in logs if "failed" in l and "display" in l], [l for l in logs if "failed" in l]
    assert not [l for l in logs if l.strip().startswith("callback")]
    steps = sorted(f for f in os.listdir(os.path.join(d, "steps")) if f.startswith("step"))
    assert len(steps) == 19                       # 14 evaluations + 2x2 validation trials + the annulus-done frame
    assert all(_size(os.path.join(d, "steps", f)) > 30_000 for f in steps)
    assert _size(os.path.join(d, "steps", "intro.png")) > 5_000
    ad = os.path.join(d, "annulus01")
    for name, lo in (("calibration_panel.png", 20_000), ("validation_panel.png", 20_000), ("corner.pdf", 8_000),
                     ("landscapes.pdf", 8_000), ("parhist.pdf", 6_000), ("edf.png", 8_000),
                     ("partition_map.png", 10_000), ("kbook.pdf", 4_000),
                     ("eval_0005_panel.pdf", 20_000), ("eval_0010_panel.pdf", 20_000), ("eval_0014_panel.pdf", 20_000)):
        assert _size(os.path.join(ad, name)) > lo, name
    assert _size(os.path.join(d, "opt_steps.gif")) > 100_000
    if os.path.exists(os.path.join(d, "opt_steps.mp4")):      # only with an ffmpeg backend
        assert _size(os.path.join(d, "opt_steps.mp4")) > 10_000


def test_annulus_data_shape(live_run):
    d, runner, results, logs, disp = live_run
    ad = annulus_from_run(d, 0)
    assert isinstance(ad, AnnulusData) and ad.n == 14 and ad.ndim == runner.space.ndim
    assert ad.partitions == ["n1", "n2"] and ad.replicated and ad.multi
    assert ad.best()[0] == results[0].search_best_index
    assert ad.winner is not None and ad.winner["validated"]
    inc = ad.inclusion()
    assert inc.shape == (14, 2) and inc.any(axis=1).all()
    cs = ad.corner()
    assert cs["names"] == ["bin", "k_klip"] and cs["X"].shape[0] == int(inc.sum())
    cs1 = ad.corner("n1")
    assert cs1["X"].shape[0] == int(inc[:, 0].sum())
    assert ad.top_mask().sum() == max(int(round(ad.gamma * 14)), 1)
    lines = ad.config_lines(ad.n - 1, "cfg")
    assert any("n1" in l for l in lines) and any(" k " in l or l.rstrip().endswith(" k") for l in lines)   # k_klip -> 'k'


def test_post_hoc_books_and_steps(live_run):
    d, runner, results, logs, disp = live_run
    out = plot_annulus_books(d, 0)
    base = {"corner", "landscapes", "parhist", "edf", "nightmap", "kbook",
            "importance", "paracoord", "rank", "slice", "products"}
    assert base <= set(out), out
    assert "verify_error" not in out, out.get("verify_error")
    for k in ("importance", "paracoord", "rank", "slice", "products"):
        assert _size(out[k]) > 5_000, k
    if "verify_limits" in out:                  # needs the winner's partition stacks + sources
        assert _size(out["verify_limits"]) > 5_000 and _size(out["verify_subsets_inj"]) > 5_000
    from klip_tpe.display import eta_squared
    x = np.repeat(np.arange(4.0), 10)
    assert eta_squared(x, x) > 0.99                              # y fully explained by x
    assert eta_squared(x, np.tile(np.arange(10.0), 4)) < 0.05    # independent
    figs = plots.plot_all(d)
    for k in ("corner_ann01", "landscapes_ann01", "parhist_ann01", "edf_ann01", "nightmap_ann01", "kbook_ann01",
              "calibration_ann01", "trace_ann01", "validation"):
        assert k in figs, k
    assert _size(figs["calibration_ann01"]) > 20_000
    sd = os.path.join(d, "steps_rebuilt")
    paths = render_steps(d, every=4, dpi=60, out_dir=sd, movie=False)
    assert len(paths) == 5 and all(_size(p) > 20_000 for p in paths)     # evals 1, 5, 9, 13 and the last (14)
    assert paths[-1].endswith("step0004.png")


def test_render_step_edge_cases(tmp_path):
    """First eval (no history), failed eval (NaN), single partition, no k_klip."""
    space = {"params": [{"name": "bin", "lo": 1, "hi": 20, "kind": "int"},
                        {"name": "filt", "lo": 0.0, "hi": 1.0, "kind": "float"}], "partitions": []}
    cfg = {"ann_edges": [5, 20], "n_iter": 6, "n_init": 3, "gamma": 0.3, "search_mode": "tpe", "seed_default": True}
    from klip_tpe.display import annulus_from_records, render_corner, render_landscapes, render_parhist_book, render_kbook
    img = np.random.default_rng(0).standard_normal((48, 48))
    recs = []
    for i, (ph, sc) in enumerate([("seed", 3.0), ("warmup", None), ("warmup", 2.0), ("tpe", 4.0), ("local", 3.5), ("tpe", 4.4)]):
        recs.append({"annulus": 0, "index": i, "phase": ph, "x": [1 + 3 * i, 0.1 * i], "config": {"params": {"bin": 1 + 3 * i, "filt": 0.1 * i},
                     "per_partition": {}, "selected": [], "x": [1 + 3 * i, 0.1 * i]},
                     "sources": [(0.5, 30.0, 1e-4), (0.6, 150.0, 1e-4), (0.55, 270.0, 1e-4)], "score": sc, "raw_score": sc,
                     "per_source": [] if sc is None else [sc, sc + 0.5, sc - 0.5], "raw_per_source": [], "clean_per_source": None,
                     "partition_snr": {}, "k_used": None, "contrast": 1e-4, "wall_s": 0.1, "meta": {"failed": sc is None, "selected": []}})
    # first eval only
    ad1 = annulus_from_records(recs[:1], space, cfg, 0, 0.05, 4.0, "m", "edge")
    assert not ad1.multi and not ad1.has_k()
    fig = render_step(ad1, 0, StepImages(cur_inj=img, cur_clean=img, best_inj=img, best_clean=img, best_index=0),
                      str(tmp_path / "s0.png"), dpi=50)
    assert _size(str(tmp_path / "s0.png")) > 10_000
    # failed eval + full history, no images at all
    ad = annulus_from_records(recs, space, cfg, 0, 0.05, 4.0, "m", "edge")
    assert np.isnan(ad.y[1]) and ad.best()[0] == 5
    render_step(ad, 1, StepImages(note="no images"), str(tmp_path / "s1.png"), dpi=50, fig=fig)
    assert _size(str(tmp_path / "s1.png")) > 10_000
    render_corner(ad, str(tmp_path / "c.pdf"))
    render_landscapes(ad, str(tmp_path / "l.pdf"))
    render_parhist_book(ad, str(tmp_path / "p.pdf"))
    render_kbook(ad, str(tmp_path / "k.pdf"))
    for f in ("c.pdf", "l.pdf", "p.pdf", "k.pdf"):
        assert _size(str(tmp_path / f)) > 2_000


def _wide_annulus(nparts=6, n=12):
    """6 partitions x 9 replicated params, 2 injected sources (the near_full annulus-1 case)."""
    from klip_tpe.display import annulus_from_records
    parts = [f"n{k + 1}" for k in range(nparts)]
    bases = [("bin", 5, 30, "int"), ("n_ang", 1, 9, "int"), ("filter", 0, 25, "int"), ("angsep", 0, 3, "float"),
             ("anglemax", 20, 100, "int"), ("corr_thresh", 0, 1, "float"), ("noise_max", 0.3, 3, "float"),
             ("coronoise_max", 0.3, 3, "float"), ("k_klip", 1, 100, "int")]
    params = [{"name": f"{b}_{p}", "base": b, "lo": lo, "hi": hi, "kind": k, "partition": p}
              for b, lo, hi, k in bases for p in parts]
    space = {"params": params, "partitions": parts}
    cfg = {"ann_edges": [0, 20], "n_iter": 20, "n_init": 5, "gamma": 0.25, "search_mode": "tpe", "seed_default": True}
    rng = np.random.default_rng(1)
    recs = []
    for i in range(n):
        x, per = [], {}
        for b, lo, hi, k in bases:
            for p in parts:
                v = rng.uniform(lo, hi)
                v = int(v) if k == "int" else float(v)
                x.append(v)
                per.setdefault(p, {})[b] = v
        sel = parts[:3 + i % 4]
        recs.append({"annulus": 0, "index": i, "phase": "seed" if i == 0 else ("warmup" if i < 5 else "tpe"), "x": x,
                     "config": {"params": {}, "per_partition": per, "selected": sel, "x": x},
                     "sources": [(0.5, 30.0, 6e-5), (0.6, 200.0, 6e-5)], "score": float(rng.uniform(0, 2)),
                     "raw_score": 1.0, "per_source": [2.3, 1.0], "raw_per_source": [], "clean_per_source": [0.1, 0.2],
                     "partition_snr": {p: 1.0 for p in sel}, "k_used": {p: int(per[p]["k_klip"]) for p in sel},
                     "contrast": 6e-5, "wall_s": 1.0, "meta": {"selected": sel}})
    return annulus_from_records(recs, space, cfg, 0, 0.0455, 4.0, "mawet_peak", "synth6x9")


def test_config_table_fits_and_abbreviates(tmp_path):
    from klip_tpe.display import TEXT_WIDTH
    ad = _wide_annulus()
    lines = ad.config_lines(3, "-- current config --")
    assert all(len(l) <= TEXT_WIDTH for l in lines), max(len(l) for l in lines)
    head = lines[1]
    for h in ("bin", "nang", "filt", "asep", "amax", "corr", "noise", "coro", " k ", " in"):
        assert h in head, (h, head)
    assert "corr_thresh" not in head and len(lines) == 2 + 6      # one column block: header + 6 partitions
    # a narrower budget wraps the columns into blocks (partition column repeated, 'in' only once)
    wrapped = ad.config_lines(3, "cfg", max_width=40)
    assert len(wrapped) > len(lines) and sum(l.rstrip().endswith("in") for l in wrapped) == 1
    img = np.random.default_rng(0).standard_normal((80, 80))
    p = str(tmp_path / "wide.png")
    render_step(ad, 11, StepImages(cur_clean=img, cur_inj=img, best_inj=img, best_clean=img, best_index=3,
                                   fm_image=np.abs(img)), p, dpi=60, contrast_note="eval 12: test reason")
    assert _size(p) > 20_000


def test_live_preview_two_sources():
    """The per-eval contrast preview needs only 2 positive per-source values (median-K
    throughput); with fewer it reports why."""
    from klip_tpe.display import _live_curve, _live_curve_info
    ad = _wide_annulus()
    rng = np.random.default_rng(2)
    clean = rng.standard_normal((80, 80))
    cc, why = _live_curve_info(clean, ad, 3)
    assert why is None and cc is not None and cc.get("fit", "").startswith("median K")
    c = np.asarray(cc["curve"], float)
    assert np.isfinite(c).any() and (c[np.isfinite(c)] > 0).all() and len(cc["sample_c"]) == 2
    assert _live_curve(clean, ad, 3) is not None
    ad.per_source[4] = [2.3, -0.4]
    cc, why = _live_curve_info(clean, ad, 4)
    assert cc is None and "1 of 2" in why and ">= 2" in why
    assert _live_curve_info(None, ad, 3)[1] == "no clean image for this eval"
    ad.sources[5] = ad.sources[5][:1]
    ad.per_source[5] = ad.per_source[5][:1]
    assert "need >= 2" in _live_curve_info(clean, ad, 5)[1]


def test_annulus_done_frame_keeps_last_images(live_run):
    """The annulus-done frame shows the LAST evaluated images titled 'last eval N'."""
    d, runner, results, logs, disp = live_run
    assert disp._last_images.get("index") == 13 and disp._last_images.get("inj") is not None
    r = results[0]
    for f in ("fm_curve", "per_source", "k_default"):
        assert hasattr(r, f)
    # the live curve list copes with AnnulusResult objects carrying the new optional fields
    class FakeRunner:
        results = [r]
        best_images = {}
    ad = annulus_from_run(d, 0)
    curves = disp._curves(FakeRunner(), ad, None, done=True)
    labels = [c[0] for c in curves]
    assert any("validated winner" in l for l in labels)
    if r.fm_curve:
        assert any("KLIP-FM cross-check" in l for l in labels)


def test_live_display_never_raises(tmp_path):
    """A broken renderer is logged, not raised."""
    from klip_tpe import display as dm
    d = str(tmp_path / "boom")
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=3, n_init=2,
                                                     validation=ValidationConfig(n_top=1, n_valid=1),
                                                     calibration=CalibrationConfig(recal_budget=0, n_remeasure=1))
    logs = []
    orig = dm.render_step
    dm.render_step = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        runner = Runner(red, space, obj, samp, cfg, d, log=logs.append, callbacks=[LiveDisplay(d, movie=False)])
        runner.run()
    finally:
        dm.render_step = orig
    # the step panel renders on a worker thread now: the failure surfaces as a
    # "[display] render failed" log line (or, for the inline final frame, a callback failure)
    assert any("boom" in l and ("render failed" in l or "failed" in l) for l in logs)
    assert os.path.exists(os.path.join(d, "final_results.json"))


def test_intro_and_calshow(tmp_path, live_run):
    """The launch movie renders every act and the frame set + GIF are written; the
    calibration panel renders with and without images."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from klip_tpe.intro import N_RELEASE, draw_calshow, draw_intro_frame, render_intro_frames
    fig = Figure(figsize=(8, 4), dpi=40); FigureCanvasAgg(fig)
    for f in (0, 60, 100, 150, 200, N_RELEASE - 1):
        draw_intro_frame(fig, f)
        assert len(fig.axes) == 1
    paths = render_intro_frames(str(tmp_path), every=30, dpi=30, movie=True)   # explicit call (aliens=False by default)
    assert len(paths) == 10 and all(_size(p) > 1000 for p in paths)
    assert os.path.exists(os.path.join(str(tmp_path), "intro.gif"))
    d, runner, results, logs, disp = live_run
    ad = annulus_from_run(d, 0)
    draw_calshow(fig, ad, None, None, [], 1, 3e-5, float("nan"))
    img = np.random.default_rng(0).normal(size=(100, 100))
    draw_calshow(fig, ad, img, img, [(0.8, 30.0, 3e-5), (0.8, 210.0, 3e-5)], 2, 3e-5, 4.1,
                 per_source=[4.0, 4.2], clean_per_source=[0.3, -0.2])
    fig.savefig(str(tmp_path / "cal.png"), dpi=40)
    assert _size(str(tmp_path / "cal.png")) > 2000
    # the live run wrote calibration-trial frames
    assert any(f.startswith("calib_ann01_trial") for f in os.listdir(os.path.join(d, "steps")))


def test_validation_trials_reach_step_display(tmp_path):
    """Every validation trial produces a step frame (IDL's ``valid`` frames) with the
    trial's images in the Test cells and a VALIDATION step label."""
    from klip_tpe import display as dm
    d = str(tmp_path / "val")
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=3, n_init=2,
                                                     validation=ValidationConfig(n_top=2, n_valid=2),
                                                     calibration=CalibrationConfig(recal_budget=0, n_remeasure=1))
    seen = []
    orig = dm.render_step

    def spy(ad, i, images, png, *a, **k):
        seen.append((images.cur_override, k.get("step_label")))
        return orig(ad, i, images, png, *a, **k)
    dm.render_step = spy
    logs = []
    try:
        disp = LiveDisplay(d, movie=False, dpi=40)
        runner = Runner(red, space, obj, samp, cfg, d, log=logs.append, callbacks=[disp])
        runner.run()
    finally:
        dm.render_step = orig
    val = [s for s in seen if s[0]]
    assert len(val) == 4, (len(val), logs[-5:])
    assert all("VALIDATION" in (s[1] or "") for s in val)
    assert all(s[0]["sources"] and len(s[0]["per_source"]) == len(s[0]["sources"]) for s in val)
    assert not any("render failed" in l for l in logs), [l for l in logs if "failed" in l]
    # the frames are numbered on after the search frames
    assert len([p for p in os.listdir(os.path.join(d, "steps")) if p.startswith("step")]) >= 3 + 4


def test_resume_mid_annulus_restores_display_history(tmp_path):
    """After a restart in the middle of an annulus the live panel shows the whole annulus
    (records before the restart come back from results.jsonl), not just the new evals."""
    from klip_tpe import display as dm
    d = str(tmp_path / "res")
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=6, n_init=2,
                                                     validation=ValidationConfig(n_top=1, n_valid=1),
                                                     calibration=CalibrationConfig(recal_budget=0, n_remeasure=1))
    r = Runner(red, space, obj, samp, cfg, d, log=lambda s: None, callbacks=[LiveDisplay(d, movie=False, dpi=40)])
    orig = r.checkpoint

    class Boom(Exception):
        pass

    def cp(final_annulus=False):
        orig(final_annulus)
        if r.history is not None and len(r.history) == 3:
            raise Boom
    r.checkpoint = cp
    with pytest.raises(Boom):
        r.run()
    seen = []
    orig_rs = dm.render_step

    def spy(ad, i, images, *a, **k):
        seen.append((i, ad.n))
        return orig_rs(ad, i, images, *a, **k)
    dm.render_step = spy
    logs = []
    try:
        red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=6, n_init=2,
                                                         validation=ValidationConfig(n_top=1, n_valid=1),
                                                         calibration=CalibrationConfig(recal_budget=0, n_remeasure=1))
        r2 = Runner.resume(d, red, obj, samp, log=logs.append)
        r2.callbacks = [LiveDisplay(d, movie=False, dpi=40)]
        r2.run()
    finally:
        dm.render_step = orig_rs
    assert any("restored 3 earlier evaluations" in l for l in logs), logs
    search = [(i, n) for i, n in seen if i >= 3][:3]
    assert search == [(3, 4), (4, 5), (5, 6)], seen


def test_resume_restores_best_images(tmp_path):
    """On a resume the incumbent is re-reduced once so the Best cells / products have images."""
    from klip_tpe import display as dm
    d = str(tmp_path / "rb")
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=6, n_init=2,
                                                     validation=ValidationConfig(n_top=1, n_valid=1),
                                                     calibration=CalibrationConfig(recal_budget=0, n_remeasure=1))
    r = Runner(red, space, obj, samp, cfg, d, log=lambda s: None)
    orig = r.checkpoint

    class Boom(Exception):
        pass

    def cp(final_annulus=False):
        orig(final_annulus)
        if r.history is not None and len(r.history) == 4:
            raise Boom
    r.checkpoint = cp
    with pytest.raises(Boom):
        r.run()
    bi0, _ = r.history.best()
    seen = []
    orig_rs = dm.render_step

    def spy(ad, i, images, *a, **k):
        seen.append((i, images.best_index, images.best_inj is not None))
        return orig_rs(ad, i, images, *a, **k)
    dm.render_step = spy
    logs = []
    try:
        red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=6, n_init=2,
                                                         validation=ValidationConfig(n_top=1, n_valid=1),
                                                         calibration=CalibrationConfig(recal_budget=0, n_remeasure=1))
        r2 = Runner.resume(d, red, obj, samp, log=logs.append)
        assert not r2._best_images
        r2.callbacks = [LiveDisplay(d, movie=False, dpi=40)]
        r2.run()
    finally:
        dm.render_step = orig_rs
    assert any("best images restored" in l for l in logs), logs
    assert r2._best_images.get("record").index == bi0
    first = [s for s in seen if s[0] == 4][0]
    assert first[2], seen                      # the Best cell had an image on the first post-restart frame


def test_progress_movie_written_periodically(tmp_path):
    """annulusNN/progress.gif is rebuilt every movie_every evaluations and at annulus end."""
    d = str(tmp_path / "mv")
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=5, n_init=2,
                                                     validation=ValidationConfig(n_top=1, n_valid=1),
                                                     calibration=CalibrationConfig(recal_budget=0, n_remeasure=1))
    from klip_tpe import animate
    calls = []
    orig = animate.make_movie

    def spy(paths, out_gif, *a, **k):
        calls.append((os.path.basename(out_gif), len(paths)))
        return orig(paths, out_gif, *a, **k)
    animate.make_movie = spy
    try:
        Runner(red, space, obj, samp, cfg, d, log=lambda s: None,
               callbacks=[LiveDisplay(d, movie=True, dpi=40, movie_every=2)]).run()
    finally:
        animate.make_movie = orig
    prog = [c for c in calls if c[0] == "progress.gif"]
    assert len(prog) >= 3 and prog[-1][1] > prog[0][1], calls        # evals 2, 4, 5(last) + annulus end
    assert os.path.getsize(os.path.join(d, "annulus01", "progress.gif")) > 20_000
    assert os.path.exists(os.path.join(d, "opt_steps.gif"))
    assert animate.thin(list(range(1000)), 400)[-1] == 999 and len(animate.thin(list(range(1000)), 400)) == 400


def test_inline_display_updates_one_output(tmp_path):
    """show='inline' pushes the panel into one IPython display handle, updated in place."""
    ipd = pytest.importorskip("IPython.display")
    calls = []

    class H:
        def update(self, img):
            calls.append("update")
    orig = ipd.display
    ipd.display = lambda img, display_id=False: (calls.append("display"), H())[1]
    try:
        d = str(tmp_path / "inl")
        red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=3, n_init=2,
                                                         validation=ValidationConfig(n_top=1, n_valid=1),
                                                         calibration=CalibrationConfig(recal_budget=0, n_remeasure=1))
        disp = LiveDisplay(d, movie=False, dpi=40, show="inline")
        logs = []
        Runner(red, space, obj, samp, cfg, d, log=logs.append, callbacks=[disp]).run()
    finally:
        ipd.display = orig
    assert disp.inline and calls[0] == "display" and calls.count("display") == 1 and calls.count("update") >= 3
    assert not [l for l in logs if "display update failed" in l]


def test_image_panels_are_north_up_east_left():
    """The sky orientation of every image panel, pinned.

    The codebase places a source at ``x = cx + r cos(PA+90)``, ``y = cy + r sin(PA+90)``,
    so North is ``+y`` and East is ``-x``.  An image drawn with ``origin='lower'`` and an
    x axis that increases to the right is therefore ALREADY North up / East left, and must
    not be flipped again -- a mirrored panel still looks self-consistent (the source and
    its circle move together), so only a test catches it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from klip_tpe.display import AnnulusData, ParamInfo, draw_image
    from klip_tpe.metrics import source_xy, star_center

    # the convention itself
    x_n, y_n = source_xy([1.0], [0.0], 1.0, 0.0, 0.0)        # North
    x_e, y_e = source_xy([1.0], [90.0], 1.0, 0.0, 0.0)       # East
    assert y_n[0] == pytest.approx(1.0) and x_n[0] == pytest.approx(0.0, abs=1e-9)
    assert x_e[0] == pytest.approx(-1.0) and y_e[0] == pytest.approx(0.0, abs=1e-9)

    # and the panel that draws it
    n = 41
    px = 0.05
    cx, cy = star_center((n, n))
    img = np.zeros((n, n))
    xs, ys = source_xy([0.5], [90.0], px, cx, cy)            # a source due East
    img[int(round(ys[0])), int(round(xs[0]))] = 1.0
    ad = AnnulusData(run_name="t", annulus=0, nann=1, inrad=0.0, outrad=18.0, pxscale=px,
                     fwhm=3.0,
                     params=[ParamInfo("k_klip", "k_klip", 1, 10, "int", None, "reduction")],
                     partitions=["p"], X=np.zeros((1, 1)), y=np.array([1.0]), phases=["seed"],
                     k_used=[5], selected=[["p"]], part_snr=[{}], sources=[[]], per_source=[[]],
                     raw_per_source=[[]], clean_per_source=[None], raw=np.array([1.0]),
                     wall=np.array([1.0]), contrast=np.array([1e-4]), configs=[{}], n_init=1,
                     n_iter=1, gamma=0.25, metric_name="m", search_mode="tpe")
    fig, ax = plt.subplots()
    try:
        draw_image(ax, img, ad, "t", flatten=False)
        x0, x1 = ax.get_xlim()
        assert x1 > x0, "the x axis must increase to the right (East is already -x)"
        # the bright pixel must land left of centre in data coordinates
        im = ax.images[0]
        ext = im.get_extent()
        col = int(np.argmax(img.sum(axis=0)))
        x_data = ext[0] + (col + 0.5) / img.shape[1] * (ext[1] - ext[0])
        assert x_data < 0, "a source due East must be drawn to the LEFT of the star"
    finally:
        plt.close(fig)


def test_walk_and_parhist_pages_do_not_warn_on_a_pinned_dimension():
    """A dimension whose range collapsed (lo == hi: a partition the guard fixed, a
    parameter pinned for this annulus) made draw_walk and _parhist_page call set_xlim /
    set_ylim with identical limits -- one matplotlib UserWarning per cell per page, which
    is what the tutorial notebooks were full of.  Both pages now widen flat ranges the way
    the KDE corner already did."""
    import warnings
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from klip_tpe.display import AnnulusData, ParamInfo, draw_walk, _parhist_page
    n = 6
    X = np.column_stack([np.linspace(2, 9, n), np.full(n, 12.0), np.linspace(1, 5, n)])   # middle one pinned
    ad = AnnulusData(run_name="t", annulus=0, nann=1, inrad=6.0, outrad=20.0, pxscale=0.05, fwhm=3.0,
                     params=[ParamInfo("k_klip", "k_klip", 1, 10, "int", None, "reduction"),
                             ParamInfo("bin", "bin", 12, 12, "int", None, "reduction"),
                             ParamInfo("angsep", "angsep", 0, 5, "float", None, "reduction")],
                     partitions=["p"], X=X, y=np.linspace(1.0, 3.0, n), phases=["seed"] * 3 + ["tpe"] * 3,
                     k_used=[5] * n, selected=[["p"]] * n, part_snr=[{}] * n, sources=[[]] * n,
                     per_source=[[]] * n, raw_per_source=[[]] * n, clean_per_source=[None] * n,
                     raw=np.linspace(1.0, 3.0, n), wall=np.ones(n), contrast=np.full(n, 1e-4),
                     configs=[{}] * n, n_init=3, n_iter=n, gamma=0.25, metric_name="m", search_mode="tpe")
    cs = ad.corner()
    assert any(l == h for l, h in zip(cs["lo"], cs["hi"])), "the fixture must contain a flat dimension"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)          # any 'identical low and high' warning fails
        fig = Figure(figsize=(6, 6)); FigureCanvasAgg(fig)
        draw_walk(fig, cs, ad, "walk")
        fig = Figure(figsize=(6, 8)); FigureCanvasAgg(fig)
        _parhist_page(fig, ad, cs, "parhist", current=n - 1)


def test_display_pdfs_never_ask_freetype_for_u_fffe():
    """Type 3 embedding (matplotlib <= 3.10) builds a cp1252 width table whose five undefined
    slots decode to U+FFFE, and matplotlib hides the resulting 'Glyph 65534 missing' warning
    behind a warnings.catch_warnings() that a concurrent catch_warnings() on another thread
    (the reducer's, every evaluation) wipes -- so it leaked out of live runs at random.  With
    pdf.fonttype 42 the table is never built.  The race is simulated by taking matplotlib's
    filter away: any Glyph warning then fails the test.

    matplotlib 3.11 builds the Type 3 widths from the font's own charmap, never asks FreeType
    for U+FFFE, and no longer imports `warnings` in backend_pdf: there the leak cannot be
    reproduced, and the test only checks that the display rc still selects Type 42 (the Macs
    the runs happen on are on older matplotlibs) and that its PDFs are warning-free."""
    import io
    import warnings
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends import backend_pdf
    from klip_tpe.display import _rc

    import types

    class _NoFilter:                      # what the other thread's __exit__ does to the filter
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    mpl_version = tuple(int(p) for p in matplotlib.__version__.split(".")[:2])
    leak_exists = mpl_version < (3, 11)
    # backend_pdf's `warnings` name is swapped for a stand-in whose filter does nothing; the
    # real module (and this test's own recording) is untouched.  3.11 has no such name (and no
    # filter to take away) -- nothing to swap.
    real = getattr(backend_pdf, "warnings", None)
    if real is not None:
        backend_pdf.warnings = types.SimpleNamespace(catch_warnings=lambda *a, **k: _NoFilter(),
                                                     filterwarnings=lambda *a, **k: None,
                                                     simplefilter=lambda *a, **k: None, warn=warnings.warn)
    try:
        def render(fonttype):
            with matplotlib.rc_context({"pdf.fonttype": fonttype}):
                fig = Figure(figsize=(2, 1))
                fig.text(0.1, 0.5, "final", family="sans-serif")
                fig.text(0.1, 0.2, "k = 10", family="monospace")
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    fig.savefig(io.BytesIO(), format="pdf")
            return [str(x.message) for x in w if "Glyph" in str(x.message)]
        if leak_exists:
            assert real is not None, "matplotlib < 3.11 hides the U+FFFE lookups behind backend_pdf.warnings"
            assert render(3), "the fixture must reproduce the leak with Type 3 (else the test proves nothing)"
        assert render(42) == []
        with _rc():
            assert matplotlib.rcParams["pdf.fonttype"] == 42, "the display rc must select Type 42"
            fig = Figure(figsize=(2, 1)); fig.text(0.1, 0.5, "final")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fig.savefig(io.BytesIO(), format="pdf")
            assert not [x for x in w if "Glyph" in str(x.message)]
    finally:
        if real is not None:
            backend_pdf.warnings = real
