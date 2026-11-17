"""The step panels drawn by a process of their own (:mod:`klip_tpe.panels`).

From the NEAR2 note of 2026-11-16 on s7e.  The panels were drawn on a thread of the run's own
process, and matplotlib holds the interpreter while it draws, so the main thread ran at a
fraction of its speed whenever a panel was being drawn.  What must hold:

* a run draws its step panels, their PDFs, the validation trials' panels and the progress movie
  in another process, the same files the render thread writes, and closes that process at its
  end;
* the run's own process draws no search panel;
* a resumed run gives the process the annulus' records again;
* a panel process that stops hands the panels back to the render thread, with a line in the log;
* a panel for a window comes back with a copy drawn at the window's pixels.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from conftest import build_synthetic_run
from klip_tpe import ValidationConfig
from klip_tpe import display as D
from klip_tpe.display import LiveDisplay
from klip_tpe.runner import Runner, RunCallback


def _objs(**over):
    kw = dict(ann_edges=[8, 20, 30], n_iter=[6, 5], n_init=3, seed=5, save_fits=False, candidates=False,
              verify=False, param_verify=False, batch=2, batch_mode="pipeline",
              validation=ValidationConfig(n_top=1, n_valid=2))
    kw.update(over)
    return build_synthetic_run(**kw)


def _run(tmp_path, name, engine, monkeypatch, callbacks=(), **over):
    monkeypatch.setenv("KLIP_TPE_PANELS", engine)
    d = str(tmp_path / name)
    disp = LiveDisplay(d, every=2, pdf_every=4, dpi=40)
    logs = []
    r = Runner(*_objs(**over), d, log=logs.append, callbacks=[disp, *callbacks])
    r.run()
    return d, disp, logs


def _files(d):
    names = [os.path.relpath(p, d) for p in glob.glob(os.path.join(d, "steps", "step*.png"))]
    names += [os.path.relpath(p, d) for p in glob.glob(os.path.join(d, "annulus*", "eval_*_panel.pdf"))]
    names += [os.path.relpath(p, d) for p in glob.glob(os.path.join(d, "annulus*", "progress.gif"))]
    return sorted(names)


class _Watch(RunCallback):
    """The display's panel process as each evaluation finds it."""

    def __init__(self, disp):
        self.disp, self.pids = disp, set()

    def on_eval(self, runner, rec, inj, clean, is_best):
        pp = self.disp._pp
        if pp is not None:
            self.pids.add(pp.proc.pid)


def test_the_panels_are_drawn_by_another_process(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    labels = []
    orig = D.render_step

    def spy(*a, **kw):
        labels.append(kw.get("step_label"))
        return orig(*a, **kw)
    monkeypatch.setattr(D, "render_step", spy)
    d_thr, _, _ = _run(tmp_path, "thread", "thread", monkeypatch)
    drawn_here = len(labels)
    labels.clear()
    disp = LiveDisplay(str(tmp_path / "process"), every=2, pdf_every=4, dpi=40)
    watch = _Watch(disp)
    monkeypatch.setenv("KLIP_TPE_PANELS", "process")
    logs = []
    Runner(*_objs(), str(tmp_path / "process"), log=logs.append, callbacks=[disp, watch]).run()
    d_proc = str(tmp_path / "process")
    assert len(watch.pids) == 1 and os.getpid() not in watch.pids
    assert _files(d_proc) == _files(d_thr), "the process writes the files the render thread writes"
    assert any(f.endswith(".pdf") for f in _files(d_proc)) and any(f.endswith(".gif") for f in _files(d_proc))
    assert drawn_here > len(labels), "the search's and validation's panels left this process"
    assert all(str(lab).startswith(("annulus done", "run complete")) for lab in labels), labels
    assert not [s for s in logs if "panel process" in s], [s for s in logs if "panel process" in s]
    assert disp._pp is None, "the run's end closes the process"
    (pid,) = watch.pids
    with pytest.raises(OSError):
        os.kill(pid, 0)


def test_a_resume_gives_the_process_the_records_again(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    monkeypatch.setenv("KLIP_TPE_PANELS", "process")
    d = str(tmp_path / "run")

    class Stop(RunCallback):
        def on_eval(self, runner, rec, inj, clean, is_best):
            if rec.index >= 4:
                raise KeyboardInterrupt

    red, space, obj, samp, cfg = _objs(ann_edges=[8, 30], n_iter=[9])
    disp = LiveDisplay(d, every=1, pdf_every=0, dpi=40, movie=False)
    with pytest.raises(KeyboardInterrupt):
        Runner(red, space, obj, samp, cfg, d, log=lambda s: None, callbacks=[disp, Stop()]).run()
    disp._close_panels()
    before = len(glob.glob(os.path.join(d, "steps", "step*.png")))
    disp2 = LiveDisplay(d, every=1, pdf_every=0, dpi=40, movie=False)
    sent = []
    orig = disp2._remote_sync

    def watch(pp, runner, model=True):
        orig(pp, runner, model)
        sent.append(disp2._pp_sent)
    disp2._remote_sync = watch
    logs = []
    Runner.resume(d, *_objs(ann_edges=[8, 30], n_iter=[9])[:1], obj, samp, log=logs.append, callbacks=[disp2]).run()
    assert sent and sent[0] >= 5, "the first panel after the resume carries the evaluations before it"
    assert len(glob.glob(os.path.join(d, "steps", "step*.png"))) > before
    assert not [s for s in logs if "failed in the panel process" in s]


def test_a_panel_process_that_stops_hands_the_panels_back(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_PANELS", "process")
    logs = []
    disp = LiveDisplay(str(tmp_path), save_png=False, movie=False, log=logs.append)
    pp = disp._panel_proc()
    assert pp is not None and pp.broken is None
    pp.proc.kill()
    pp.proc.wait()
    pp.wait(timeout=5)
    for _ in range(100):
        if pp.broken is not None:
            break
        pp._r.join(0.05)
    assert pp.broken is not None
    assert disp._panel_proc() is None and disp._pp_off
    assert any("the panel process stopped" in s for s in logs), logs
    assert disp._render_room(), "the render thread's gate takes over"


def test_the_thread_engine_starts_no_process(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_PANELS", "thread")
    disp = LiveDisplay(str(tmp_path), save_png=False, movie=False, log=lambda s: None)
    assert disp._panel_proc() is None and disp._pp is None


def test_a_window_panel_comes_back_drawn_at_its_pixels(tmp_path, monkeypatch):
    """The process draws the window's copy at the window's own pixels, as the render thread does."""
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    from klip_tpe import panels
    d = str(tmp_path / "run")
    r = Runner(*_objs(ann_edges=[8, 30], n_iter=[5]), d, log=lambda s: None)
    r.run()
    from klip_tpe.plots import _annulus_records, load_run
    recs = _annulus_records(load_run(d), 0)
    r.ia = 0
    recipe = D.runner_recipe(r)
    pp = panels.PanelProcess({"run_dir": d, "dpi": 40, "corner_mode": "cycle"})
    try:
        img = np.random.default_rng(0).normal(size=(60, 60))
        pp.send({"op": "recipe", "kw": recipe, "reset": True})
        pp.send({"op": "records", "start": 0, "recs": recs})
        png = str(tmp_path / "p.png")
        mid = pp.send({"op": "panel", "i": len(recs) - 1, "live": True, "corner": True, "stitch": True,
                       "cur": {"cur_clean": img, "cur_inj": img}, "img": {"best_index": None, "fm_label": None},
                       "kw": {"step_label": "test"}, "png": png, "pdf": None, "box": (925, 495)}, reply=True)
        assert pp.wait(timeout=120)
        (rep,) = [x for x in pp.take() if x["id"] == mid]
        assert rep["ok"], rep.get("error")
        assert os.path.exists(png)
        scr = rep["screen"]
        assert scr.dtype == np.uint8 and scr.shape[2] == 4
        assert max(scr.shape[0] / 495, scr.shape[1] / 925) == pytest.approx(1.0, abs=0.01), scr.shape
    finally:
        pp.close()
    assert not os.path.isdir(pp.dir)
