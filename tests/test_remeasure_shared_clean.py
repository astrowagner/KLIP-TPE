"""An averaged trial reduces its clean image once.

``Runner.evaluate_mean`` used to call ``evaluate`` once per draw, and every call reduced the
injected AND the clean image -- but the clean image depends on the configuration alone, so
at ``n_remeasure = 3`` that was 6 reductions where 4 do (found by the NEAR2 / IDL session,
2026-10-07).  What must hold now:

* the trial is the trial the per-draw loop made -- same positions, same RNG stream after it,
  same per-draw scores -- so nothing a run records changes, and a run started on the old
  code can be resumed on the new one;
* one clean reduction per trial, and ``n`` injected ones;
* on a parallel reducer at most two reductions at a time (the pair the thread budget was
  sized for), and the later draws do overlap, which is where the wall-time saving is;
* a first draw that failed leaves nothing to share, and the later draws reduce their own;
* the k-scan modes keep the per-draw loop (each draw picks its own k, hence its own clean).
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from conftest import build_synthetic_run
from klip_tpe import PartitionedReducer, ValidationConfig
from klip_tpe.runner import Runner

QUIET = lambda s: None  # noqa: E731


def _runner(tmp_path, name, share=True, workers=1, pool="threads", **over):
    kw = dict(seed=7, n_remeasure=3)
    kw.update(over)
    red, space, obj, samp, cfg = build_synthetic_run(**kw)
    if workers > 1:
        red = PartitionedReducer(red.reducers, max_workers=workers, pool=pool)
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / name), log=QUIET)
    r.SHARE_CLEAN = share
    r.ia, r.contrast, r._k_default = 0, 3e-5, 5
    return r, space


def _instrument(r, delay=0.0):
    """Count clean / injected reductions and the most that ran at once."""
    st = {"clean": 0, "inj": 0, "now": 0, "max": 0}
    lock = threading.Lock()
    orig = r._reduce

    def counted(cfg_, sources, k_scan=False, tag="", **kw):
        with lock:
            st["clean" if not sources else "inj"] += 1
            st["now"] += 1
            st["max"] = max(st["max"], st["now"])
        try:
            if delay:
                time.sleep(delay)
            return orig(cfg_, sources, k_scan, tag, **kw)
        finally:
            with lock:
                st["now"] -= 1
    r._reduce = counted
    return st


def _vectors(space, n=3):
    rng = np.random.default_rng(99)
    return [space.default_vector()] + [np.asarray(space.random(rng), float) for _ in range(n - 1)]


def _same_trial(a, b):
    assert a.meta["draw_sources"] == b.meta["draw_sources"], "the draws went to different positions"
    assert a.meta["draw_scores"] == pytest.approx(b.meta["draw_scores"], abs=1e-9)
    assert a.meta["draw_raw_scores"] == pytest.approx(b.meta["draw_raw_scores"], abs=1e-9)
    assert a.score == pytest.approx(b.score, abs=1e-9)
    assert a.sources == b.sources and a.per_source == pytest.approx(b.per_source, abs=1e-9)


def test_one_clean_reduction_per_trial_and_the_same_trial(tmp_path):
    each, space = _runner(tmp_path, "each", share=False)
    shared, _ = _runner(tmp_path, "shared", share=True)
    ce, cs = _instrument(each), _instrument(shared)
    xs = [each._project(x, is_random=False) for x in _vectors(space)]
    for x in xs:
        ra, ia, cla = each.evaluate_mean(x, "tpe", tag="t")
        rb, ib, clb = shared.evaluate_mean(x, "tpe", tag="t")
        _same_trial(ra, rb)
        np.testing.assert_allclose(ia.image, ib.image, equal_nan=True)
        np.testing.assert_allclose(cla.image, clb.image, equal_nan=True)
    assert each.rng.bit_generator.state == shared.rng.bit_generator.state, \
        "the search RNG must stand where the per-draw loop left it"
    n = len(xs)
    assert (ce["clean"], ce["inj"]) == (3 * n, 3 * n)
    assert (cs["clean"], cs["inj"]) == (n, 3 * n), "one clean reduction per trial, three injected"


def test_parallel_reducer_pairs_the_later_draws(tmp_path):
    each, space = _runner(tmp_path, "each", share=False)
    par, _ = _runner(tmp_path, "par", share=True, workers=2)
    st = _instrument(par, delay=0.05)
    x = each._project(space.default_vector(), is_random=False)
    ra, _, _ = each.evaluate_mean(x, "tpe", tag="t")
    rb, _, _ = par.evaluate_mean(x, "tpe", tag="t")
    _same_trial(ra, rb)
    assert (st["clean"], st["inj"]) == (1, 3)
    assert st["max"] == 2, f"two reductions at a time, never more (saw {st['max']})"
    par.reducer.close()


def test_an_overlapping_trial_reports_its_elapsed_time():
    """Draws that ran side by side each took the pair's time; their sum is not the trial's."""
    from klip_tpe.runner import EvalRecord

    def rec(score, wall):
        return EvalRecord(annulus=0, index=0, phase="tpe", x=[0.0], config={"params": {}},
                          sources=[(1.0, 0.0, 1e-4)], score=score, raw_score=score, per_source=[score],
                          raw_per_source=[score], clean_per_source=None, partition_snr={}, k_used=5,
                          contrast=1e-4, wall_s=wall, meta={})
    recs = [rec(4.0, 2.0), rec(5.0, 2.0), rec(6.0, 2.0)]
    assert Runner._combine_draws(recs, recs[-1]).wall_s == pytest.approx(6.0)
    assert Runner._combine_draws(recs, recs[-1], wall=4.1).wall_s == pytest.approx(4.1)


def test_worker_processes_give_the_same_trial(tmp_path):
    """The production pool: forked workers, two maps in flight at once (ProcessPool sorts
    results by job id, so the pair cannot swap images)."""
    each, space = _runner(tmp_path, "each", share=False)
    par, _ = _runner(tmp_path, "proc", share=True, workers=4, pool="processes")
    xs = [each._project(x, is_random=False) for x in _vectors(space, 2)]
    try:
        for x in xs:
            ra, ia, _ = each.evaluate_mean(x, "tpe", tag="t")
            rb, ib, _ = par.evaluate_mean(x, "tpe", tag="t")
            _same_trial(ra, rb)
            np.testing.assert_allclose(ia.image, ib.image, equal_nan=True)
    finally:
        par.reducer.close()


def test_draws_are_reported_in_order(tmp_path):
    par, space = _runner(tmp_path, "par", share=True, workers=2)
    _instrument(par, delay=0.02)
    seen = []
    x = par._project(space.default_vector(), is_random=False)
    out, _, _ = par.evaluate_mean(x, "tpe", tag="t", on_draw=lambda j, n, rec, i_, c_: seen.append((j, n)))
    assert seen == [(0, 3), (1, 3), (2, 3)]
    assert out.meta["draw_n"] == 3
    par.reducer.close()


def test_a_failed_first_draw_leaves_the_later_draws_their_own_clean(tmp_path):
    r, space = _runner(tmp_path, "flaky", share=True)
    st = _instrument(r)
    counted = r._reduce
    calls = {"n": 0}

    def flaky(cfg_, sources, k_scan=False, tag="", **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated reducer crash")
        return counted(cfg_, sources, k_scan, tag, **kw)
    r._reduce = flaky
    out, _, clean = r.evaluate_mean(r._project(space.default_vector(), is_random=False), "tpe", tag="t")
    assert out.meta["draw_failed"] == 1 and out.score is not None
    assert st["clean"] == 2, "draws 2 and 3 had nothing to share and reduced their own"
    assert clean is not None


def test_kscan_modes_keep_the_per_draw_loop(tmp_path):
    r, space = _runner(tmp_path, "scan", share=True)
    called = {}
    r._kscan_active = lambda: True
    r._evaluate_draws_each = lambda x, phase, tag, n, on_draw=None: called.setdefault("each", (tag, n)) or (None,) * 3
    r.evaluate_mean(space.default_vector(), "tpe", tag="a1_e3")
    assert called == {"each": ("a1_e3", 3)}


def test_a_whole_search_is_unchanged(tmp_path):
    """Calibration, seed, search and validation, end to end: the shared clean must not move
    a single recorded number."""
    out = {}
    for share in (False, True):
        red, space, obj, samp, cfg = build_synthetic_run(
            seed=3, n_remeasure=3, ann_edges=[8, 30], n_iter=7, n_init=3,
            validation=ValidationConfig(n_top=2, n_valid=2), save_eval_images=False, fm_curve=False,
            fm_preview=False, write_setup_files=False)
        r = Runner(red, space, obj, samp, cfg, str(tmp_path / f"run_{share}"), log=QUIET)
        r.SHARE_CLEAN = share
        res = r.run()
        out[share] = (r, res)
    (ra, resa), (rb, resb) = out[False], out[True]
    ya, yb = np.asarray(ra.history.y, float), np.asarray(rb.history.y, float)
    assert len(ya) == 7 and np.isfinite(ya).all()
    np.testing.assert_allclose(ya, yb, atol=1e-9, equal_nan=True)
    recs = {s: [json.loads(l) for l in open(tmp_path / f"run_{s}" / "results.jsonl")] for s in (False, True)}
    searched = [r for r in recs[True] if r.get("meta", {}).get("draw_n")]
    assert len(searched) >= 6, "the search trials were averaged over draws"
    for a, b in zip(recs[False], recs[True]):
        assert a.get("sources") == b.get("sources")
        assert a.get("meta", {}).get("draw_sources") == b.get("meta", {}).get("draw_sources")
    assert resa[0].winner_index == resb[0].winner_index
    assert resa[0].winner_score == pytest.approx(resb[0].winner_score, abs=1e-9)
    va = [row["validated_score"] for row in resa[0].validation_table]
    vb = [row["validated_score"] for row in resb[0].validation_table]
    assert len(va) == 2 and va == pytest.approx(vb, abs=1e-9)
