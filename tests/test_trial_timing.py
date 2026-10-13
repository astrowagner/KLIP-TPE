"""A trial's own times, on the live panel and in run.log.

From the NEAR2 note of 2026-10-12 on per-trial timing.  With four trials in flight the
panel read "avg 0.6 s/eval (2.1 s reduce)".  The interval between two evaluations is the
throughput, not the time any of them takes, and ``wall_s`` runs on to the trial's scoring,
so it also holds the wait for workers and for the main thread.  What must hold:

* every search record carries ``reduce_s``, from its dispatch until the last of its
  reductions was back, and its dispatch time, and ``reduce_s <= wall_s``;
* a trial's eval time, from its dispatch until it is processed, is never below its reduce
  time, with trials in flight as with one at a time;
* run.log says ``reduce Xs / eval Ys``;
* the panel says ``avg eval .. (reduce ..)``, ``last eval .. (reduce ..)`` and
  ``1 done every .. (N in flight)``, and nowhere ``s/eval``.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pytest

from conftest import build_synthetic_run
from klip_tpe import PartitionedReducer, ValidationConfig
from klip_tpe.runner import Runner


def _objs(workers=1, **over):
    kw = dict(ann_edges=[8, 30], n_iter=[8], n_init=3, seed=5, n_remeasure=2, save_fits=False,
              candidates=False, verify=False, param_verify=False,
              validation=ValidationConfig(n_top=1, n_valid=1))
    kw.update(over)
    red, space, obj, samp, cfg = build_synthetic_run(**kw)
    if workers > 1:
        red = PartitionedReducer(red.reducers, max_workers=workers, pool="processes")
    return red, space, obj, samp, cfg


class _Clock:
    """Each search trial's record and its eval time as the display counts it."""

    def __init__(self):
        self.rows = []

    def on_eval(self, runner, rec, inj, clean, is_best):
        self.rows.append((rec.reduce_s, rec.wall_s, time.time() - rec.meta["t_dispatch"], rec.phase))

    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda *a, **k: None
        raise AttributeError(name)


@pytest.mark.parametrize("workers,batch,mode", [(1, 1, "pipeline"), (4, 1, "pipeline"), (4, 3, "pipeline"),
                                                (4, 3, "sync")])
def test_reduce_never_exceeds_eval_and_the_log_says_both(tmp_path, monkeypatch, workers, batch, mode):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    clock, logs = _Clock(), []
    r = Runner(*_objs(workers, batch=batch, batch_mode=mode), str(tmp_path / "run"), log=logs.append,
               callbacks=[clock])
    orig_reduce, orig_score = r._reduce, r._score_images

    def slow_reduce(cfg_, sources, k_scan=False, tag="", **kw):
        time.sleep(0.05)
        return orig_reduce(cfg_, sources, k_scan=k_scan, tag=tag, **kw)

    def slow_score(*a, **kw):
        time.sleep(0.03)                       # scoring comes after the reductions are back
        return orig_score(*a, **kw)
    r._reduce, r._score_images = slow_reduce, slow_score
    r.run()
    search = [row for row in clock.rows if row[3] not in ("seed", "calib")]
    assert len(search) >= 8
    for red_s, wall_s, eval_s, _ in search:
        assert red_s is not None and red_s >= 0.045, "a trial's reductions took at least 50 ms"
        assert red_s <= wall_s + 1e-6 <= eval_s + 2e-6
        assert wall_s - red_s >= 0.025, "at least the last draw's scoring (30 ms) lies between the two"
    lines = [s for s in logs if re.search(r"\] eval \d+/\d+ ", s)]
    assert lines and all(re.search(r"reduce [\d.]+s / eval [\d.]+s", s) for s in lines), lines[:2]
    assert not any("/ loop" in s for s in lines)


def test_a_record_keeps_its_reduction_time_and_dispatch(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    d = tmp_path / "run"
    Runner(*_objs(4, batch=2), str(d), log=lambda s: None).run()
    recs = [json.loads(line) for line in open(d / "results.jsonl")]
    search = [x for x in recs if x["phase"] not in ("seed",)]
    assert search and all(x["reduce_s"] is not None and 0 <= x["reduce_s"] <= x["wall_s"] for x in search)
    assert all(isinstance(x["meta"]["t_dispatch"], float) for x in search)


def test_the_panel_names_the_trial_times_and_the_rate(tmp_path):
    """The live panel's timing block, and the classic panel's current-evaluation line."""
    import matplotlib
    matplotlib.use("Agg")
    from klip_tpe.display import StepImages, _step_text, _trial_times, annulus_from_records, render_step
    recs = []
    for k in range(6):
        recs.append({"annulus": 0, "index": k, "phase": "tpe", "x": [10.0, 5.0, 10.0, 5.0, 0.0, 1.0],
                     "score": 3.0 + k, "raw_score": 3.0, "wall_s": 2.0, "reduce_s": 1.0 + 0.1 * k,
                     "sources": [[0.6, 10.0 * k, 1e-3]], "per_source": [3.0], "contrast": 1e-3,
                     "config": {"selected": ["n1", "n2"]}, "meta": {"t_dispatch": 0.0}})
    red, space, obj, samp, cfg = build_synthetic_run()
    ad = annulus_from_records(recs, space.to_dict(), cfg.to_dict(), 0, 0.05, 4.0)
    assert np.allclose(ad.extra["reduce"], [1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    # after the run, or for a validation trial: the records' reduction times only
    tt = _trial_times(ad, 5, None)
    assert np.isnan(tt["avg"][0]) and tt["avg"][1] == pytest.approx(1.25) and tt["last"][1] == pytest.approx(1.5)
    timing = {"avg": (2.4, 1.2), "last": (2.6, 1.5), "every": 0.6, "flight": 4}
    fig = render_step(ad, 5, StepImages(), timing=timing, loop_s=0.6, elapsed_s=100.0, eta_s=50.0,
                      eta_full_s=500.0)
    text = "\n".join(t.get_text() for t in fig.texts)
    assert "avg  eval 2.4 s  (reduce 1.2 s)" in text and "last eval 2.6 s  (reduce 1.5 s)" in text
    assert "1 done every 0.6 s  (4 in flight)" in text
    assert "s/eval" not in text
    lines = _step_text(ad, 5, 5, 100.0, 50.0, timing)
    assert any(line.startswith("eval       2.6 s  (reduce 1.5 s)") for line in lines), lines
    assert not any("s/eval" in line for line in lines)
