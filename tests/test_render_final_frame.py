"""``klip-tpe render`` draws an annulus' last frame as the live annulus-done frame drew it.

From the WISPIT 2 ALES session, 2026-11-10.  On ``wispit2_ales2`` the redrawn last frame of
annulus 4 showed the search's best (eval 994) where the live frame showed the validated winner
(eval 854), annulus 4's image alone in the stitched cells where the live frame showed the stitch
of annuli 1 to 4, no injected PSF, and annulus 4's own 2 h 19 m as "Elapsed" where the live
frame showed the run's 14 h 42 m.  What must hold, on a synthetic run of three annuli:

* The redrawn last frame of each annulus gets the live frame's cells: the validated winner, its
  committed image circled at its own sources with their values, the running stitch of the
  finished annuli and its sources, the KLIP-FM model, the injected PSF, the labels, and the
  run's elapsed time at the annulus' end.
* A run made before these were kept still gets them where it can: the stitch from the run's own
  running stitch (the last annulus) or made again from the winners' products, the elapsed time
  from the heartbeat, and the injected PSF from the frames when the caller can load them.
* The live annulus-done frame circles only the winners' sources in the stitch.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import klip_tpe.display as D
from klip_tpe import Runner
from klip_tpe.display import LiveDisplay, render_steps

from conftest import build_synthetic_run

QUIET = (lambda s: None)


class _Captured:
    def __init__(self):
        self.frames = []

    def __call__(self, ad, i, images, out_png=None, out_pdf=None, curves=(), elapsed_s=np.nan, eta_s=np.nan,
                 step_label="", white=False, **kw):
        if not white:
            self.frames.append({"annulus": ad.annulus, "i": i, "images": images, "elapsed": elapsed_s,
                                "label": step_label, "curves": [c[0] for c in curves or []]})
        return object()


def _same(a, b, float32=False):
    """Equal images; ``float32``: to the rounding of a float32 product on disk."""
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a, float), np.asarray(b, float)
    atol = 1e-6 * float(np.nanmax(np.abs(b))) if float32 and np.isfinite(b).any() else 1e-12
    return a.shape == b.shape and np.allclose(a, b, rtol=2e-6, atol=atol, equal_nan=True)


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("final") / "run")
    mp = pytest.MonkeyPatch()
    cap = _Captured()
    mp.setattr(D, "render_step", cap)
    mp.setattr(D, "render_verify_books", lambda *a, **k: {})
    mp.setattr(D, "render_products_sheet", lambda *a, **k: None)
    mp.setattr(LiveDisplay, "_books_start", lambda self, *a, **k: None)
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 20, 32, 44], n_iter=[4, 4, 4], n_init=2,
                                                     stitch_every=0, param_verify=False, verify=False)
    disp = LiveDisplay(d, every=1, pdf_every=0, movie=False, show=False, dpi=60)
    results = Runner(red, space, obj, samp, cfg, d, log=QUIET, callbacks=[disp]).run()
    mp.undo()
    done = {f["annulus"]: f for f in cap.frames if str(f["label"]).startswith("annulus done")}
    assert sorted(done) == [0, 1, 2]
    return d, results, done


def _redraw(d, tmp_path, **kw):
    cap = _Captured()
    mp = pytest.MonkeyPatch()
    mp.setattr(D, "render_step", cap)
    try:
        render_steps(d, every=1000, out_dir=str(tmp_path / "rebuilt"), movie=False, **kw)
    finally:
        mp.undo()
    n = {}
    for f in cap.frames:
        n[f["annulus"]] = max(n.get(f["annulus"], -1), f["i"])
    return {f["annulus"]: f for f in cap.frames if f["i"] == n[f["annulus"]]}


def _check(redrawn, live_done, results, keys=("stitch", "inj_model", "elapsed"), float32=False):
    for ia, lf in live_done.items():
        rf = redrawn[ia]
        a, b = rf["images"], lf["images"]
        assert a.best_index == b.best_index == results[ia].winner_index
        assert a.best_sources == b.best_sources and a.best_labels == b.best_labels
        assert _same(a.best_inj, b.best_inj) and _same(a.best_clean, b.best_clean)
        assert _same(a.cur_inj, b.cur_inj) and a.cur_label == b.cur_label and a.note == b.note
        assert _same(a.fm_image, b.fm_image) and a.fm_label == b.fm_label
        if "stitch" in keys:
            assert _same(a.stitch_clean, b.stitch_clean, float32) and _same(a.stitch_inj, b.stitch_inj, float32), ia
            assert a.stitch_sources == b.stitch_sources, ia
        if "inj_model" in keys:
            assert _same(a.inj_model, b.inj_model), ia
        if "elapsed" in keys:
            assert rf["elapsed"] == pytest.approx(lf["elapsed"], abs=1e-6), ia
        assert rf["label"].startswith(lf["label"]) and rf["label"].endswith("(rebuilt from log)")
        assert rf["curves"] == lf["curves"]


def test_the_redrawn_last_frame_is_the_live_one(live, tmp_path):
    d, results, done = live
    for ia in range(3):
        a = os.path.join(d, f"annulus{ia + 1:02d}")
        assert os.path.exists(os.path.join(a, "annulus_done.json")) and os.path.exists(os.path.join(a, "best_inj_model.fits"))
        assert os.path.exists(os.path.join(a, "klip_stitched_running.fits"))
    assert done[0]["images"].stitch_clean is None, "annulus 1 shows its best images, as the IDL does"
    assert done[2]["images"].stitch_clean is not None and done[2]["images"].inj_model is not None
    # the stitch circles the winners' sources, not the search best's
    want = [tuple(t) for r in results for t in r.winner_sources]
    assert done[2]["images"].stitch_sources == want
    _check(_redraw(d, tmp_path), done, results)
    # one annulus drawn alone still has the finished annuli' curves and stitch
    alone = _redraw(d, tmp_path, annuli=[2])
    assert list(alone) == [2] and alone[2]["curves"] == done[2]["curves"]
    _check(alone, {2: done[2]}, results)


def test_a_run_made_before_gets_the_frame_where_it_can(live, tmp_path):
    import shutil
    d0, results, done = live
    d = str(tmp_path / "old")
    shutil.copytree(d0, d)
    for ia in range(3):
        a = os.path.join(d, f"annulus{ia + 1:02d}")
        for f in ("annulus_done.json", "best_inj_model.fits", "klip_stitched_running.fits",
                  "klip_stitched_running_inj.fits"):
            os.remove(os.path.join(a, f))
    # the stitch: annulus 3's from the run's running stitch, annulus 2's made again from the products
    st = Runner.annulus_end_stitch(d, 2)
    assert st["how"].startswith("the run's running stitch")
    assert Runner.annulus_end_stitch(d, 1)["how"].startswith("made again")
    red = _redraw(d, tmp_path)
    _check(red, done, results, keys=("stitch",), float32=True)       # made again from float32 products
    # no injected PSF without the frames, and the cell says why
    assert red[2]["images"].inj_model is None and "--with-data" in red[2]["images"].inj_model_note
    # the elapsed time from the heartbeat, within the time the hooks took
    for ia in range(3):
        assert red[ia]["elapsed"] == pytest.approx(done[ia]["elapsed"], abs=5.0)
    # with the frames, the injected PSF of the winner's sources, as the live frame drew it
    build = build_synthetic_run()[0]
    from klip_tpe.injection import injected_model
    red2 = _redraw(d, tmp_path, inj_model_fn=lambda src, shape: injected_model(build, src, shape))
    for ia in range(3):
        assert _same(red2[ia]["images"].inj_model, done[ia]["images"].inj_model), ia


def test_a_reducer_given_from_python_draws_the_injected_psf(live, tmp_path):
    """A run started from a script recorded no data arguments, so ``render --with-data`` cannot
    rebuild its reducer: the script's reducer goes to ``render_steps`` or ``Runner.render``."""
    import shutil
    d0, results, done = live
    d = str(tmp_path / "old")
    shutil.copytree(d0, d)
    for ia in range(3):
        os.remove(os.path.join(d, f"annulus{ia + 1:02d}", "best_inj_model.fits"))
    red = build_synthetic_run()[0]
    got = _redraw(d, tmp_path, reducer=red)
    for ia in range(3):
        assert _same(got[ia]["images"].inj_model, done[ia]["images"].inj_model), ia
    # a reducer at another plate scale is refused
    import types
    with pytest.raises(ValueError, match="another scale"):
        render_steps(d, every=1000, out_dir=str(tmp_path / "x"), movie=False,
                     reducer=types.SimpleNamespace(pxscale=float(red.pxscale) * 1.5))
    # Runner.render: the resumed runner's reducer, into steps_rebuilt
    cap = _Captured()
    mp = pytest.MonkeyPatch()
    mp.setattr(D, "render_step", cap)
    try:
        r2, space, obj, samp, _ = build_synthetic_run()
        runner = Runner.resume(d, r2, obj, samp, log=QUIET, same_frames=False)
        runner.render(every=1000, movie=False)
    finally:
        mp.undo()
    last = [f for f in cap.frames if str(f["label"]).startswith("annulus done")]
    assert len(last) == 3 and all(f["images"].inj_model is not None for f in last)
    assert os.path.isdir(os.path.join(d, "steps_rebuilt"))


def test_render_with_data_takes_the_runs_instrument_or_refuses(tmp_path, capsys):
    """``render --with-data`` on a run another instrument made: the instrument and scale come
    from run_setup.json, never NEAR's defaults, and a run without recorded data arguments is
    refused with the Python entry to use instead."""
    import json
    from klip_tpe import cli
    d = tmp_path / "wispit2_like"
    d.mkdir()
    (d / "run_setup.json").write_text(json.dumps({"reducer": {"name": "ales", "pxscale": 0.0345,
                                                             "provenance": {"instrument": "ALES"}}}))
    assert cli._run_instrument(str(d)) == "ales"
    with pytest.raises(SystemExit) as ei:
        cli.main(["render", "--run-dir", str(d), "--with-data", "--no-movie"])
    msg = str(ei.value)
    assert "--root" in msg and "render_steps(run_dir, reducer=red" in msg
    out = capsys.readouterr().out
    assert "--instrument ales, as run_setup.json records" in out
    assert "0.0456" not in out and "--north constant" not in out, "NEAR's defaults on an ALES run"
    # an instrument typed that is not the run's is refused
    with pytest.raises(SystemExit, match="made with --instrument ales"):
        cli.main(["render", "--run-dir", str(d), "--with-data", "--instrument", "nomic", "--no-movie"])
    # a run whose record does not say is refused
    (d / "run_setup.json").write_text(json.dumps({"reducer": {"name": "synthetic"}}))
    with pytest.raises(SystemExit, match="does not say which instrument"):
        cli.main(["render", "--run-dir", str(d), "--with-data", "--no-movie"])
    # and the NEAR constants never land on another instrument's run
    (d / "run_setup.json").write_text(json.dumps({"reducer": {"name": "ales", "pxscale": 0.0345}}))
    import types
    a = types.SimpleNamespace(instrument="near", run_dir=str(d), _argv=["resume", "--run-dir", str(d)])
    with pytest.raises(SystemExit, match="--instrument ales"):
        cli._constants_for_run(a)
