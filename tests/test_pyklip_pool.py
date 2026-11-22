"""The pool context the pyKLIP backend gives pyKLIP (klip_tpe.backends.pyklip._NoMainContext), without pyKLIP."""
from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="no forkserver and no SIGINT to a pid on Windows")


def test_the_pools_workers_ignore_a_ctrl_c_and_run_pyklips_initializer():
    from klip_tpe.backends.pyklip import _NoMainContext
    ctx = _NoMainContext(multiprocessing.get_context("forkserver"))
    # pyKLIP's call: processes, initializer and initargs by keyword, and maxtasksperchild, which is dropped
    pool = ctx.Pool(processes=2, initializer=signal.signal, initargs=(signal.SIGUSR1, signal.SIG_IGN),
                    maxtasksperchild=50)
    try:
        assert pool.apply(signal.getsignal, (signal.SIGINT,)) == signal.SIG_IGN
        assert pool.apply(signal.getsignal, (signal.SIGUSR1,)) == signal.SIG_IGN, "pyKLIP's own initializer ran"
        res = pool.apply_async(time.sleep, (1.0,))
        time.sleep(0.3)                                   # the task is running
        workers = list(pool._pool)
        for w in workers:                                 # what a Ctrl-C to the process group does to them
            os.kill(w.pid, signal.SIGINT)
        assert res.get(timeout=30) is None, "the task came back, so pyKLIP's wait for it returns"
        assert all(w.is_alive() for w in workers) and list(pool._pool) == workers
    finally:
        pool.terminate()
        pool.join()


def test_positional_arguments_reach_the_pool_as_pyklip_once_passed_them():
    from klip_tpe.backends.pyklip import _NoMainContext
    pool = _NoMainContext(multiprocessing.get_context("forkserver")).Pool(1, signal.signal,
                                                                          (signal.SIGUSR2, signal.SIG_IGN), 10)
    try:
        assert pool.apply(signal.getsignal, (signal.SIGUSR2,)) == signal.SIG_IGN
        assert pool.apply(signal.getsignal, (signal.SIGINT,)) == signal.SIG_IGN
    finally:
        pool.terminate()
        pool.join()
