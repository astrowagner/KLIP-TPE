"""Which pool a run got, and not losing worker processes to an import race.

Both of these come from one afternoon's evidence: a 147-D RX J0534 search silently fell
back to threads on a resume (``ImportError: cannot import name 'SEM_VALUE_MAX' from
partially initialized module multiprocessing.synchronize``) and spent the next 350
evaluations GIL-bound, while its 4-partition sibling left no record of its pool mode at all
-- so the mode of a run that had just died had to be inferred from the absence of a failure
message.
"""
from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys
import threading

import pytest

from klip_tpe.parallel import ProcessPool


# --------------------------------------------------------------- the import race itself

def test_importing_klip_tpe_is_enough_to_settle_synchronize():
    """``multiprocessing.Queue.__init__`` ends in a lazy ``from .synchronize import
    SEM_VALUE_MAX`` (CPython's own comment there: "Can raise ImportError").  It has to be
    resolved at program start, in the main thread, or a render thread's concurrent import can
    hand the pool a half-built module -- and ``klip_tpe.parallel`` itself is imported lazily
    from inside functions, so doing it there alone would be too late.  Importing the package
    must be sufficient."""
    out = subprocess.run([sys.executable, "-c",
                          "import klip_tpe, sys; print('multiprocessing.synchronize' in sys.modules)"],
                         capture_output=True, text=True)
    assert out.stdout.strip() == "True", out.stderr[-2000:]
    assert "multiprocessing.synchronize" in sys.modules


def test_queues_retry_a_partially_initialized_synchronize(monkeypatch):
    """The belt to that braces: one ImportError must not cost the run its worker pool."""
    calls = []

    class FlakyCtx:
        def Queue(self):
            calls.append(1)
            if len(calls) <= 2:            # both queues of the first attempt fail
                raise ImportError("cannot import name 'SEM_VALUE_MAX' from partially "
                                  "initialized module 'multiprocessing.synchronize'")
            return f"q{len(calls)}"

    a, b = ProcessPool._queues(FlakyCtx())
    assert (a, b) == ("q3", "q4")


def test_queues_give_up_with_the_real_error_after_the_retries(monkeypatch):
    class BrokenCtx:
        def Queue(self):
            raise ImportError("no working sem_open on this platform")

    with pytest.raises(ImportError, match="sem_open"):
        ProcessPool._queues(BrokenCtx(), tries=2)


def test_queues_are_built_for_real_from_a_fork_context():
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("no fork start method")
    a, b = ProcessPool._queues(mp.get_context("fork"))
    try:
        a.put(("hello", 1))
        assert a.get(timeout=5) == ("hello", 1)
    finally:
        for q in (a, b):
            q.close()


def test_a_concurrent_importer_cannot_break_queue_creation():
    """The shape of the original failure: another thread importing multiprocessing while
    the pool builds its queues.  With the eager import in place this must be a non-event."""
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("no fork start method")
    stop = threading.Event()

    def churn():
        while not stop.is_set():
            import importlib
            importlib.import_module("multiprocessing.synchronize")
            importlib.import_module("multiprocessing.queues")

    t = threading.Thread(target=churn, daemon=True)
    t.start()
    try:
        ctx = mp.get_context("fork")
        for _ in range(8):
            a, b = ProcessPool._queues(ctx)
            a.close(); b.close()
    finally:
        stop.set()
        t.join(timeout=2)


# ------------------------------------------------------- every run records its pool kind

class _FakeReducer:
    partition = "n1"

    def partition_weight(self):
        return 1.0


def _partitioned(n_part=2, workers=4, pool="threads", log=None):
    from klip_tpe.reducer import PartitionedReducer
    reds = {f"n{i+1}": _FakeReducer() for i in range(n_part)}
    r = PartitionedReducer.__new__(PartitionedReducer)
    r.reducers = reds
    r.max_workers = workers
    r.pool_kind = pool
    r._procs = None
    r._pool = None
    r._log = log or (lambda s: None)
    return r


def test_start_workers_logs_the_budget_and_the_pool_kind():
    said = []
    r = _partitioned(n_part=4, workers=8, pool="threads", log=said.append)
    assert r.start_workers() == "threads"
    line = "\n".join(said)
    assert "parallel:" in line and "pool = threads" in line
    assert "8 workers" in line and "partition jobs in parallel" in line


def test_a_thread_fallback_says_what_it_costs():
    """A line you skim past is how a run spends 350 evaluations in the slow mode."""
    said = []
    r = _partitioned(pool="auto", log=said.append)
    # force the fallback the way the real failure did
    import klip_tpe.reducer as rmod
    orig = rmod.PartitionedReducer._pool_kind_now

    def broken(self):
        self._log("  parallel: worker processes unavailable (ImportError('boom')); using threads")
        return "threads"

    rmod.PartitionedReducer._pool_kind_now = broken
    try:
        assert r.start_workers() == "threads"
    finally:
        rmod.PartitionedReducer._pool_kind_now = orig
    line = "\n".join(said)
    assert "GIL" in line and "worth fixing" in line


def test_the_pool_line_is_logged_once_not_per_evaluation():
    said = []
    r = _partitioned(pool="threads", log=said.append)
    for _ in range(5):
        r.start_workers()
    assert sum("pool = " in s for s in said) == 1


def test_serial_and_process_modes_report_themselves():
    said = []
    r = _partitioned(workers=1, log=said.append)
    assert r.start_workers() == "serial"
    assert "pool = serial" in "\n".join(said)

    said2 = []
    r2 = _partitioned(pool="threads", log=said2.append)
    r2._procs = object()                       # as if a pool were already up
    assert r2.start_workers() == "processes"
    assert "pool = processes" in "\n".join(said2)
