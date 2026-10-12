"""Several reductions in flight: the schedule changes, never the results.

The NEAR2 session's note of 2026-10-11 asked for four changes, in order of payoff against
risk:

1. a worker pool sized by the thread budget, ``max_workers // threads_per_job`` processes,
   where it had been two per partition whatever the budget;
2. a trial's clean image and all its draws reduced in one round;
3. validation trials, param_verify configurations, calibration remeasurements and the
   k-scan pair reduced several at a time and taken in order;
4. ``RunConfig.batch`` (``--batch``): that many trials in flight, opt-in because TPE then
   proposes from an older posterior.

and the second note, the same day: a batch keeps its trials in flight (a new proposal as
soon as the oldest trial is in, ``batch_mode="pipeline"``), and the pool sends the longest
job first.

What must hold:

* changes 1 to 3 give the run a serial reducer gives, record for record: positions, scores,
  per-draw scores, calibration, validation trials, param_verify and the generator state;
* draws, validation trials and trials of a batch are reported in order, from the main
  thread, however their reductions finish;
* a run stopped part way through a trial or a validation chunk resumes to those records;
* a batch, pipelined or whole, gives the same history on one process and on many, a
  re-calibration and a failed clean reduction inside it included;
* a pipelined run stopped with trials in flight resumes to the run that never stopped, and
  a run stopped part way through a whole batch resumes at the batch's start;
* the worker pool hands each result to the caller that asked for it, at once, and takes
  the waiting job with the smallest key first.
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import pytest

from conftest import build_synthetic_run
from klip_tpe import CalibrationConfig, PartitionedReducer, ValidationConfig
from klip_tpe.parallel import ProcessPool
from klip_tpe.runner import RunCallback, Runner

QUIET = lambda s: None  # noqa: E731


class Interrupt(BaseException):      # like Ctrl-C: past every per-trial guard
    pass


def _objs(workers=1, **over):
    kw = dict(ann_edges=[8, 30, 45], n_iter=[9, 7], n_init=4, seed=5, n_remeasure=3, save_fits=False,
              candidates=False, verify=False, validation=ValidationConfig(n_top=2, n_valid=4))
    kw.update(over)
    red, space, obj, samp, cfg = build_synthetic_run(**kw)
    if workers > 1:
        red = PartitionedReducer(red.reducers, max_workers=workers, pool="processes")
    return red, space, obj, samp, cfg


def _wrap(r, fn):
    """Route every reduction of runner ``r`` through ``fn(orig, cfg, sources, k_scan, tag, **kw)``."""
    orig = r._reduce

    def wrapped(cfg_, sources, k_scan=False, tag="", **kw):
        return fn(orig, cfg_, sources, k_scan, tag, **kw)
    r._reduce = wrapped


def _cut_at(at):
    def fn(orig, cfg_, sources, k_scan, tag, **kw):
        if tag == at:
            raise Interrupt
        return orig(cfg_, sources, k_scan=k_scan, tag=tag, **kw)
    return fn


def _records(d):
    with open(os.path.join(str(d), "results.jsonl")) as f:
        out = [json.loads(line) for line in f]
    for r in out:
        r.pop("wall_s", None)
    return out


def _first_of_each(recs):
    """As the readers of results.jsonl take it: the first record of each index (runs here
    have one segment per annulus)."""
    seen, out = set(), []
    for r in recs:
        if (r["annulus"], r["index"]) not in seen:
            seen.add((r["annulus"], r["index"]))
            out.append(r)
    return out


def _annulus_files(d, n_ann=2):
    out = {}
    for ia in range(1, n_ann + 1):
        for name in ("calibration.json", "validation.json", "winner.json"):
            p = os.path.join(str(d), f"annulus{ia:02d}", name)
            out[(ia, name)] = json.load(open(p)) if os.path.exists(p) else None
    return out


def _rng(d):
    return json.load(open(os.path.join(str(d), "checkpoint.json")))["rng_state"]


def _same_pv(a, b):
    """Two runners' param_verify results, annulus by annulus (paths aside)."""
    assert sorted(a) == sorted(b)
    for ia in a:
        pa, pb = a[ia], b[ia]
        assert sorted(pa) == sorted(pb)
        for k in pa:
            if k == "files":
                continue
            va, vb = pa[k], pb[k]
            if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
                assert np.allclose(np.asarray(va, float), np.asarray(vb, float), equal_nan=True), (ia, k)
            elif isinstance(va, dict):
                assert va.keys() == vb.keys(), (ia, k)
                for kk in va:
                    assert np.allclose(np.asarray(va[kk], float), np.asarray(vb[kk], float),
                                       equal_nan=True), (ia, k, kk)
            else:
                assert va == vb, (ia, k)


def _same_run(da, db, ra=None, rb=None, dedup=False):
    a, b = _records(da), _records(db)
    if dedup:
        a, b = _first_of_each(a), _first_of_each(b)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x == y, f"record {x['annulus']}/{x['index']} differs"
    assert _annulus_files(da) == _annulus_files(db)
    assert _rng(da) == _rng(db), "the generator ends in a different state"
    if ra is not None and rb is not None:
        _same_pv(ra._pv_results, rb._pv_results)


class _Order(RunCallback):
    """What the callbacks saw, in the order they saw it, and from which thread."""

    def __init__(self):
        self.draws, self.trials, self.evals, self.threads = [], [], [], set()

    def _here(self):
        self.threads.add(threading.current_thread() is threading.main_thread())

    def on_draw(self, runner, draw, n_draws, record, inj, clean):
        self._here()
        self.draws.append((len(runner.history), draw, n_draws))

    def on_validation_trial(self, runner, ia, ci, n_cand, eval_index, trial, n_valid, *rest):
        self._here()
        self.trials.append((ia, ci, trial))

    def on_eval(self, runner, record, inj, clean, is_best):
        self._here()
        self.evals.append((record.annulus, record.index))


@pytest.fixture(scope="module")
def serial(tmp_path_factory):
    """The reference: the same run on a serial reducer."""
    d = tmp_path_factory.mktemp("serial")
    red, space, obj, samp, cfg = _objs()
    r = Runner(red, space, obj, samp, cfg, str(d), log=QUIET)
    r.run()
    return d, r


# ------------------------------------------------------------------ 1. the pool's size
class _Part:
    """A partition with nothing to reduce: the pool's size is all these tests read."""
    partition, pxscale, fwhm, lam_over_d_px, angle_convention = "n", 0.01, 4.0, 4.0, "pa"
    supports_kscan = supports_fm = False

    def partition_weight(self):
        return 1.0

    def describe(self):
        return {"name": "part"}


def _pool(n_part, workers, **kw):
    return PartitionedReducer({f"n{i}": _Part() for i in range(n_part)}, max_workers=workers, **kw)


def test_the_pool_is_sized_by_the_thread_budget():
    """NEAR2's six nights got 12 processes whatever --workers said: 96 or 192 only raised
    the threads per job, and per-job scaling is flat past about 4."""
    got = {w: (r.n_procs(), r.job_threads(), r.job_slots()) for w in (50, 96, 192)
           for r in [_pool(6, w)]}
    assert got == {50: (12, 4, 12), 96: (24, 4, 24), 192: (48, 4, 48)}
    r = _pool(6, 192, threads_per_job=8)
    assert (r.n_procs(), r.job_threads()) == (24, 8)
    r = _pool(6, 8)                                 # never fewer than the 2 x partitions it had
    assert (r.n_procs(), r.job_threads()) == (8, 1)
    assert _pool(6, 1).job_slots() == 1
    assert _pool(6, 96).describe()["threads_per_job"] == 4


def test_the_thread_fallback_keeps_its_budget_rule():
    r = _pool(6, 48, pool="threads")
    assert r.job_threads(6) == 4                    # 48 // (2 x 6), as before
    assert r.job_slots() == 48


# --------------------------------------------------------------- the pool's results
class _Sleepy:
    partition = "a"

    def reduce(self, req):
        time.sleep(req[0])
        return req[1]


def test_the_pool_hands_each_result_to_its_caller_at_once():
    """Each caller used to read the shared queue itself.  A result another caller read
    first waited for its owner's next read to time out, a second without a live window:
    longer than a synthetic reduction, and on every concurrent reduction."""
    pool = ProcessPool({"a": _Sleepy()}, 3)
    try:
        slow = {}
        t = threading.Thread(target=lambda: slow.setdefault("out", pool.map([("a", (2.0, "slow"))])))
        t.start()
        time.sleep(0.2)
        t0 = time.time()
        for i in range(5):
            assert pool.map([("a", (0.0, (i, 1))), ("a", (0.0, (i, 2)))]) == [(i, 1), (i, 2)]
        assert time.time() - t0 < 1.0, "five quick maps waited on the slow one"
        t.join(timeout=10)
        assert slow["out"] == ["slow"]
    finally:
        pool.close()


def test_a_dead_worker_fails_what_it_held():
    pool = ProcessPool({"a": _Sleepy()}, 1)
    try:
        err = {}

        def wait():
            try:
                pool.map([("a", (30.0, "never"))])
            except Exception as exc:            # noqa: BLE001
                err["exc"] = exc
        t = threading.Thread(target=wait)
        t.start()
        time.sleep(0.3)
        pool.procs[0].kill()
        t.join(timeout=10)
        assert not t.is_alive() and "died" in str(err.get("exc"))
    finally:
        pool.close()


# ------------------------------------------------------- 2 and 3. the same run, sooner
@pytest.mark.slow
def test_a_run_on_worker_processes_is_the_serial_run(serial, tmp_path):
    """Calibration (its k-scan and remeasurements), the search trials, validation and
    param_verify, with reductions finishing out of order on purpose: the earlier draws and
    every other validation trial are slowed down."""
    d_ref, r_ref = serial
    red, space, obj, samp, cfg = _objs(workers=4)
    seen = _Order()
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "par"), log=QUIET, callbacks=[seen])
    lock = threading.Lock()
    st = {"now": 0, "max": 0, "val_max": 0, "val_now": 0}

    def slow_early(orig, cfg_, sources, k_scan, tag, **kw):
        val = "_val" in tag
        with lock:
            st["now"] += 1
            st["max"] = max(st["max"], st["now"])
            if val:
                st["val_now"] += 1
                st["val_max"] = max(st["val_max"], st["val_now"])
        try:
            if tag.endswith(("_clean", "_d1_inj")) or (val and tag[-1:] in "02468"):
                time.sleep(0.03)
            return orig(cfg_, sources, k_scan=k_scan, tag=tag, **kw)
        finally:
            with lock:
                st["now"] -= 1
                if val:
                    st["val_now"] -= 1
    _wrap(r, slow_early)
    r.run()
    _same_run(d_ref, tmp_path / "par", r_ref, r)
    assert st["max"] >= 4, "the clean and the three draws of a trial were never in flight together"
    assert st["val_max"] >= 2, "validation trials never overlapped"
    # reported in order, from the main thread
    assert seen.threads == {True}
    for i in range(0, len(seen.draws), 3):
        assert [d[1] for d in seen.draws[i:i + 3]] == [0, 1, 2]
    by_cand = {}
    for ia, ci, t in seen.trials:
        by_cand.setdefault((ia, ci), []).append(t)
    assert by_cand and all(ts == list(range(4)) for ts in by_cand.values())


@pytest.mark.slow
def test_a_run_stopped_mid_round_resumes_to_the_serial_run(serial, tmp_path):
    """Stopped while the second draw of a trial is reducing, the other draws in flight with
    it: the trial is not recorded, and the resumed run makes it again."""
    d_ref, r_ref = serial
    objs = _objs(workers=4)
    red, space, obj, samp, cfg = objs
    d = tmp_path / "cut"
    r = Runner(red, space, obj, samp, cfg, str(d), log=QUIET)
    _wrap(r, _cut_at("a1_e5_d2_inj"))
    with pytest.raises(Interrupt):
        r.run()
    assert len(_records(d)) == 4
    r2 = Runner.resume(str(d), red, obj, samp, log=QUIET)
    r2.run()
    _same_run(d_ref, d, r_ref, r2)


@pytest.mark.slow
def test_a_validation_chunk_stopped_part_way_resumes_to_the_serial_run(serial, tmp_path):
    """Stopped at a candidate's third trial with its fourth in flight: the trials before it
    are on disk, in order, and nothing after it."""
    import pickle
    d_ref, r_ref = serial
    red, space, obj, samp, cfg = _objs(workers=4)
    d = tmp_path / "cut"
    r = Runner(red, space, obj, samp, cfg, str(d), log=QUIET)
    _wrap(r, _cut_at("a1_val1_t2"))
    with pytest.raises(Interrupt):
        r.run()
    recs = []
    with open(os.path.join(str(d), "annulus01", "val_cand02_trials.pkl"), "rb") as f:
        while True:
            try:
                recs.append(pickle.load(f))
            except EOFError:
                break
    assert [len(x["trials"]) for x in recs[1:]] == [1, 1], "header, trial 0, trial 1"
    logs, tags = [], []
    r2 = Runner.resume(str(d), red, obj, samp, log=logs.append)

    def spy(orig, cfg_, sources, k_scan, tag, **kw):
        tags.append(tag)
        return orig(cfg_, sources, k_scan=k_scan, tag=tag, **kw)
    _wrap(r2, spy)
    r2.run()
    assert any("resuming after trial 2" in s for s in logs)
    assert "a1_val1_t1" not in tags and "a1_val1_t2" in tags
    _same_run(d_ref, d, r_ref, r2)


@pytest.mark.slow
def test_the_k_scan_modes_give_the_serial_run_on_worker_processes(tmp_path):
    """Each draw scans k there: the scan and its clean scan reduce together."""
    over = dict(k_mode="scan_rescore", k_scan_max=12, n_remeasure=2, n_iter=[6, 5],
                validation=ValidationConfig(n_top=2, n_valid=2))
    runs = []
    for w in (1, 4):
        red, space, obj, samp, cfg = _objs(workers=w, **over)
        r = Runner(red, space, obj, samp, cfg, str(tmp_path / f"w{w}"), log=QUIET)
        r.run()
        runs.append(r)
    _same_run(tmp_path / "w1", tmp_path / "w4", *runs)


# ------------------------------------------------------------ 4. concurrent batches
@pytest.mark.slow
@pytest.mark.parametrize("mode,n_remeasure", [("sync", 1), ("sync", 3), ("pipeline", 1), ("pipeline", 3)])
def test_a_batch_is_the_same_on_one_process_and_on_many(tmp_path, mode, n_remeasure):
    """A serial reducer reduces a batch's trials one at a time; worker processes reduce them
    together.  A re-calibration lands inside a batch here (at eval 6 of 1 + 3 + 3, or with
    two trials still in flight), so the trials after it are dropped and the generator goes
    back to where it was after the trial that triggered it."""
    over = dict(batch=3, n_remeasure=n_remeasure, batch_mode=mode)
    logs, runs = {}, []
    for w in (1, 4):
        red, space, obj, samp, cfg = _objs(workers=w, **over)
        logs[w] = []
        r = Runner(red, space, obj, samp, cfg, str(tmp_path / f"w{w}"), log=logs[w].append)
        if w > 1:
            lock = threading.Lock()
            st = {"now": 0, "max": 0}

            def count(orig, cfg_, sources, k_scan, tag, **kw):
                with lock:
                    st["now"] += 1
                    st["max"] = max(st["max"], st["now"])
                try:
                    time.sleep(0.01)
                    return orig(cfg_, sources, k_scan=k_scan, tag=tag, **kw)
                finally:
                    with lock:
                        st["now"] -= 1
            _wrap(r, count)
        r.run()
        runs.append(r)
    _same_run(tmp_path / "w1", tmp_path / "w4", *runs)
    assert any("re-cal" in s and "restarting" in s for s in logs[4])
    assert st["max"] > n_remeasure + 1, "the proposals of a batch never reduced together"


@pytest.mark.slow
def test_a_whole_batch_stopped_part_way_resumes_at_its_start(tmp_path):
    """``batch_mode="sync"`` checkpoints a batch whole: stopped after its first proposal was
    recorded, the resumed run asks for the same batch again and is the run that never
    stopped (the first proposal is in results.jsonl twice, and the readers keep the first)."""
    over = dict(batch=3, batch_mode="sync", calibration=CalibrationConfig(recal_budget=0))
    red, space, obj, samp, cfg = _objs(workers=4, **over)
    Runner(red, space, obj, samp, cfg, str(tmp_path / "ref"), log=QUIET).run()
    red, space, obj, samp, cfg = _objs(workers=4, **over)
    d = tmp_path / "cut"
    r = Runner(red, space, obj, samp, cfg, str(d), log=QUIET)
    _wrap(r, _cut_at("a1_e6_d1_inj"))               # seed, then e2-e4, then e5-e7
    with pytest.raises(Interrupt):
        r.run()
    assert [x["index"] for x in _records(d)] == [0, 1, 2, 3, 4]
    tags = []
    r2 = Runner.resume(str(d), red, obj, samp, log=QUIET)

    def spy(orig, cfg_, sources, k_scan, tag, **kw):
        tags.append(tag)
        return orig(cfg_, sources, k_scan=k_scan, tag=tag, **kw)
    _wrap(r2, spy)
    r2.run()
    assert "a1_e5_d1_inj" in tags, "the batch starts again"
    assert [x["index"] for x in _records(d)][:6] == [0, 1, 2, 3, 4, 4]
    _same_run(tmp_path / "ref", d, dedup=True)


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["sync", "pipeline"])
def test_a_failed_clean_inside_a_batch_fails_only_its_trial(tmp_path, mode):
    """The first proposal's clean reduction fails: its first draw fails with it, its later
    draws reduce their own clean images, as one at a time, and the next proposal still gets
    its own reductions, not the leftovers of this one."""
    over = dict(batch=3, batch_mode=mode, calibration=CalibrationConfig(recal_budget=0))

    def crash(orig, cfg_, sources, k_scan, tag, **kw):
        if tag == "a1_e2_d1_clean":
            raise RuntimeError("simulated reducer crash")
        return orig(cfg_, sources, k_scan=k_scan, tag=tag, **kw)
    runs = []
    for w in (1, 4):
        red, space, obj, samp, cfg = _objs(workers=w, **over)
        r = Runner(red, space, obj, samp, cfg, str(tmp_path / f"w{w}"), log=QUIET)
        _wrap(r, crash)
        r.run()
        runs.append(r)
    e2 = _records(tmp_path / "w4")[1]
    assert e2["index"] == 1 and e2["meta"]["draw_failed"] == 1 and e2["score"] is not None
    _same_run(tmp_path / "w1", tmp_path / "w4", *runs)


# ------------------------------------------------------------ pipelined batches
def test_runs_saved_before_batch_mode_resume_with_whole_batches():
    from klip_tpe import RunConfig
    cfg = RunConfig(ann_edges=[8, 30], batch=4)
    assert cfg.batch_mode == "pipeline"
    d = json.loads(json.dumps(cfg.to_dict()))
    assert RunConfig.from_dict(d).batch_mode == "pipeline"
    d.pop("batch_mode")
    assert RunConfig.from_dict(d).batch_mode == "sync"


@pytest.mark.slow
def test_a_pipelined_run_stopped_in_flight_resumes_exactly(tmp_path):
    """Stopped while three trials are in flight, the checkpoint holds them, and the resume
    reduces them again before it asks for anything new: the same records as the run that
    never stopped, none twice."""
    over = dict(batch=3, calibration=CalibrationConfig(recal_budget=0))
    red, space, obj, samp, cfg = _objs(workers=4, **over)
    Runner(red, space, obj, samp, cfg, str(tmp_path / "ref"), log=QUIET).run()
    red, space, obj, samp, cfg = _objs(workers=4, **over)
    d = tmp_path / "cut"
    r = Runner(red, space, obj, samp, cfg, str(d), log=QUIET)
    _wrap(r, _cut_at("a1_e7_d2_inj"))
    with pytest.raises(Interrupt):
        r.run()
    st = json.load(open(os.path.join(str(d), "checkpoint.json")))
    assert [x["index"] for x in _records(d)] == list(range(len(st["history"]["y"])))
    assert [p["tag"] for p in st["pending"]] == [f"a1_e{k}" for k in range(len(st["history"]["y"]) + 1,
                                                                         len(st["history"]["y"]) + 4)]
    tags = []
    r2 = Runner.resume(str(d), red, obj, samp, log=QUIET)

    def spy(orig, cfg_, sources, k_scan, tag, **kw):
        tags.append(tag)
        return orig(cfg_, sources, k_scan=k_scan, tag=tag, **kw)
    _wrap(r2, spy)
    r2.run()
    assert "a1_e7_d2_inj" in tags
    idx = [x["index"] for x in _records(d) if x["annulus"] == 0]
    assert idx == sorted(set(idx)), "an evaluation recorded twice"
    _same_run(tmp_path / "ref", d)


@pytest.mark.slow
def test_the_pipeline_counts_its_pending_trials(tmp_path):
    """Trials in flight count toward the warm-up and the grid: guided proposals start at
    eval n_init, and a grid search never proposes a cell twice."""
    over = dict(batch=3, n_remeasure=1, n_iter=[12, 6], calibration=CalibrationConfig(recal_budget=0),
                validation=ValidationConfig(n_top=1, n_valid=1), param_verify=False)
    red, space, obj, samp, cfg = _objs(workers=4, **over)
    Runner(red, space, obj, samp, cfg, str(tmp_path / "tpe"), log=QUIET).run()
    ph = [x["phase"] for x in _records(tmp_path / "tpe") if x["annulus"] == 0]
    assert ph[:4] == ["seed", "warmup", "warmup", "warmup"] and "warmup" not in ph[4:]
    cells = {}
    for b in (1, 3):
        red, space, obj, samp, cfg = _objs(workers=4, search_mode="grid", **dict(over, batch=b))
        Runner(red, space, obj, samp, cfg, str(tmp_path / f"grid{b}"), log=QUIET).run()
        cells[b] = [x["x"] for x in _records(tmp_path / f"grid{b}") if x["phase"] == "grid" and x["annulus"] == 0]
    assert len(cells[3]) == 11 and cells[3] == cells[1], "the grid's cells, in the grid's order"


@pytest.mark.slow
def test_the_pipeline_keeps_its_trials_in_flight(tmp_path):
    """Reductions of three trials at once (two each: the clean and one draw), and the next
    trial's go out before the finished trial's files are written."""
    over = dict(batch=3, n_remeasure=1, n_iter=[10, 6], calibration=CalibrationConfig(recal_budget=0),
                validation=ValidationConfig(n_top=1, n_valid=1), param_verify=False)
    red, space, obj, samp, cfg = _objs(workers=4, **over)
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "run"), log=QUIET)
    lock = threading.Lock()
    st = {"now": 0, "max": 0, "order": []}

    def count(orig, cfg_, sources, k_scan, tag, **kw):
        with lock:
            st["now"] += 1
            st["max"] = max(st["max"], st["now"])
            st["order"].append(("start", tag))
        try:
            time.sleep(0.02)
            return orig(cfg_, sources, k_scan=k_scan, tag=tag, **kw)
        finally:
            with lock:
                st["now"] -= 1
    _wrap(r, count)
    order = []
    calls, saved = r._trial_calls, r._save_eval_images

    def sent(cfg_, draws, tag):
        order.append(("sent", tag))
        return calls(cfg_, draws, tag)

    def written(rec, *a, **kw):
        order.append(("written", f"a{rec.annulus + 1}_e{rec.index + 1}"))
        return saved(rec, *a, **kw)
    r._trial_calls, r._save_eval_images = sent, written
    r.run()
    assert st["max"] >= 6, f"three trials' reductions in flight together (saw {st['max']})"
    # e2 in, e5 asked and sent (2 + 3), and only then e2's files
    assert order.index(("sent", "a1_e5")) < order.index(("written", "a1_e2"))
    assert order.index(("sent", "a1_e4")) < order.index(("written", "a1_e2"))


@pytest.mark.slow
def test_a_recalibration_in_the_pipeline_rewinds_the_generator(tmp_path):
    """The re-calibration after eval 6 drops evals 7-8 in flight and restarts from the
    generator as it stood right after eval 6 drew its positions, as if 7 and 8 had never
    been asked."""
    red, space, obj, samp, cfg = _objs(workers=4, batch=3)
    logs = []
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "run"), log=logs.append)
    after_draws, at_reset = {}, []
    calls, apply_ = r._trial_calls, r._recal_apply

    def sent(cfg_, draws, tag):
        after_draws.setdefault(tag, r.rng.bit_generator.state)
        return calls(cfg_, draws, tag)

    def reset(ia, rc):
        if rc[0] == "reset" and not at_reset:
            at_reset.append(r.rng.bit_generator.state)
        return apply_(ia, rc)
    r._trial_calls, r._recal_apply = sent, reset
    r.run()
    assert any("restarting annulus" in s for s in logs) and at_reset
    assert at_reset[0] == after_draws["a1_e6"]
    assert after_draws["a1_e8"] != after_draws["a1_e6"], "evals 7 and 8 had drawn by then"
