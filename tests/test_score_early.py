"""A trial's draws scored where they were reduced (Runner.SCORE_EARLY).

On a parallel reducer each draw's search score is made in the thread that ran its reduction,
as soon as its images and the trial's clean ones are back, so the main thread finds the trial
scored when its turn comes.  What must hold: the run is the run the main thread's scoring made,
record for record, and the search's scoring happens off the main thread.
"""
from __future__ import annotations

import json
import os
import threading

import pytest

from conftest import build_synthetic_run
from klip_tpe import PartitionedReducer, RunConfig, ValidationConfig
from klip_tpe.runner import Runner


def _run(tmp_path, name, early):
    red, space, obj, samp, _ = build_synthetic_run()
    pr = PartitionedReducer(red.reducers, max_workers=3, pool="threads")
    cfg = RunConfig(ann_edges=[8, 30], n_iter=[10], n_init=4, seed=5, defaults={"k_klip": 5}, n_remeasure=3,
                    batch=2, validation=ValidationConfig(n_top=1, n_valid=1))
    run_dir = str(tmp_path / name)
    r = Runner(pr, space, obj, samp, cfg, run_dir, log=lambda s: None)
    r.SCORE_EARLY = early
    where = []
    orig = r._score_images

    def spy(*a, **kw):
        where.append(threading.current_thread() is threading.main_thread())
        return orig(*a, **kw)
    r._score_images = spy
    try:
        r.run()
    finally:
        pr.close()
    recs = [json.loads(ln) for ln in open(os.path.join(run_dir, "results.jsonl")) if ln.strip()]
    return recs, where


def _fields(recs):
    keep = ("draw_scores", "draw_raw_scores", "draw_sources", "draw_failed")
    return [(r["annulus"], r["index"], r["phase"], r["score"], r["raw_score"], r["per_source"],
             r["raw_per_source"], r["clean_per_source"], r["x"], {k: r["meta"].get(k) for k in keep})
            for r in recs]


@pytest.mark.slow
def test_scoring_where_reduced_changes_no_record(tmp_path):
    early, where_early = _run(tmp_path, "early", True)
    main, where_main = _run(tmp_path, "main", False)
    assert _fields(early) == _fields(main)
    assert all(where_main), "with the switch off every draw is scored on the main thread"
    off = len(where_early) - sum(where_early)
    assert off >= 3 * 10, f"the search's draws scored off the main thread: {off} of {len(where_early)}"


def test_no_early_scoring_on_a_serial_reducer_or_for_a_forward_model_metric(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run()
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "serial"), log=lambda s: None)
    assert not r._score_early()
    calls = [lambda: 1, lambda: 2]
    assert r._scored_calls(calls, [[]]) == calls
    pr = PartitionedReducer(red.reducers, max_workers=3, pool="threads")
    try:
        r2 = Runner(pr, space, obj, samp, cfg, str(tmp_path / "par"), log=lambda s: None)
        assert r2._score_early()
        r2.objective.metric.needs_fm = True
        assert not r2._score_early()
    finally:
        pr.close()
