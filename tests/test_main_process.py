"""The main process' share of a pipelined search, a line per annulus.

From the NEAR2 note of 2026-11-16 on s7e.  Once the workers keep up, the main process sets a
pipelined search's pace: every trial waits for it to be scored, recorded and followed by the
next proposal.  The note asked for a line per annulus, beside the cards' lines, with the main
process' busy fraction and the trials' mean wait for it.  What must hold:

* each annulus of a pipelined search logs two lines, the busy share and the trials' wait, then
  the main thread's time per trial and its steps;
* ``annulus_done.json`` and ``final_results.json`` keep the numbers, and ``bench_summary.txt``
  has a ``# main`` line per annulus;
* a main thread that sets the pace reads busy and makes trials wait, and one that waits for the
  reductions reads idle with trials that do not wait;
* a search without trials in flight gets no line.
"""
from __future__ import annotations

import json
import os
import re
import time

import pytest

from conftest import build_synthetic_run
from klip_tpe import PartitionedReducer, ValidationConfig
from klip_tpe.runner import Runner


def _runner(tmp_path, logs, batch=3, n_iter=(9, 8), **over):
    kw = dict(ann_edges=[8, 20, 30], n_iter=list(n_iter), n_init=3, seed=5, save_fits=False, candidates=False,
              verify=False, param_verify=False, batch=batch, batch_mode="pipeline",
              validation=ValidationConfig(n_top=1, n_valid=1))
    kw.update(over)
    red, space, obj, samp, cfg = build_synthetic_run(**kw)
    red = PartitionedReducer(red.reducers, max_workers=4, pool="threads")
    return Runner(red, space, obj, samp, cfg, str(tmp_path / "run"), log=logs.append)


def _slow(r, reduce_s=0.0, main_s=0.0):
    """Make the reductions or the main thread's bookkeeping take at least so long."""
    if reduce_s:
        orig = r._reduce

        def slow_reduce(cfg_, sources, k_scan=False, tag="", **kw):
            time.sleep(reduce_s)
            return orig(cfg_, sources, k_scan=k_scan, tag=tag, **kw)
        r._reduce = slow_reduce
    if main_s:
        orig_b = r._eval_bookkeeping

        def slow_books(*a, **kw):
            time.sleep(main_s)
            return orig_b(*a, **kw)
        r._eval_bookkeeping = slow_books


def _numbers(r, ia):
    with open(os.path.join(r.run_dir, f"annulus{ia + 1:02d}", "annulus_done.json")) as f:
        return json.load(f).get("main")


def test_each_annulus_says_how_busy_the_main_process_was(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    logs = []
    r = _runner(tmp_path, logs)
    _slow(r, reduce_s=0.02)
    r.run()
    busy = [s for s in logs if "the main process was busy" in s]
    steps = [s for s in logs if "the main process took" in s]
    assert len(busy) == 2 and len(steps) == 2, busy + steps
    for ia, (b, st) in enumerate(zip(busy, steps)):
        assert b.strip().startswith(f"annulus {ia + 1}:") and st.strip().startswith(f"annulus {ia + 1}:")
        assert re.search(r"busy \d+ % of the search's .+, and a trial waited [\d.]+ s for it on average", b), b
        assert re.search(r"took [\d.]+ s a trial, [\d.]+ s scoring, [\d.]+ s recording, [\d.]+ s proposing and "
                         r"[\d.]+ s on the files", st), st
        m = _numbers(r, ia)
        assert m is not None and m["annulus"] == ia + 1
        # every search trial but the seed, and those a re-calibration threw away before they counted
        assert m["trials"] >= r.cfg.per_annulus(r.cfg.n_iter, ia) - 1
        assert 0.0 <= m["busy_fraction"] <= 1.0 and m["trial_wait_s"] >= 0.0 and m["search_s"] > 0
        assert set(m["step_s"]) == {"score", "record", "propose", "rest"}
        assert all(v >= -1e-6 for v in m["step_s"].values()) and not m["since_resume"]
    with open(os.path.join(r.run_dir, "final_results.json")) as f:
        fin = json.load(f)
    assert [m["annulus"] for m in fin["main"]] == [1, 2]
    with open(os.path.join(r.run_dir, "bench_summary.txt")) as f:
        summ = f.read()
    for ia in (1, 2):
        assert re.search(rf"# main annulus {ia}: busy \d+ % of the search's \d+ s, a trial waited [\d.]+ s for it, "
                         r"[\d.]+ s a trial \(score [\d.]+, record [\d.]+, propose [\d.]+, rest [\d.]+\)", summ), summ


@pytest.mark.parametrize("slow,busy_side", [("main", True), ("reduce", False)])
def test_who_sets_the_pace(tmp_path, monkeypatch, slow, busy_side):
    """A slow main thread with fast reductions is busy and makes the trials wait.  A fast one
    behind slow reductions waits for them, and a trial finds it free."""
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    logs = []
    r = _runner(tmp_path, logs, n_iter=(10,), ann_edges=[8, 30])
    if slow == "main":
        _slow(r, main_s=0.15)
    else:
        _slow(r, reduce_s=0.25)
    r.run()
    m = _numbers(r, 0)
    if busy_side:
        assert m["busy_fraction"] > 0.8, m
        assert m["trial_wait_s"] > 0.1, "three trials in flight behind a 0.15 s main thread wait for it"
        assert m["step_s"]["rest"] >= 0.15
    else:
        assert m["busy_fraction"] < 0.6, m
        assert m["trial_wait_s"] < 0.05, m


def test_one_trial_at_a_time_gets_no_line(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    logs = []
    r = _runner(tmp_path, logs, batch=1, n_iter=(6,), ann_edges=[8, 30])
    r.run()
    assert not [s for s in logs if "main process" in s]
    assert _numbers(r, 0) is None
    with open(os.path.join(r.run_dir, "final_results.json")) as f:
        assert "main" not in json.load(f)
