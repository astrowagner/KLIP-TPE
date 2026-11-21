"""The main thread's rest step split, the checkpoint's interval, and the CPU workers' idle time.

From NEAR2's note of 2026-11-17 on s7e, s7f and s7g.  The rest step of the main process grew
across a run, and with 8 trials in flight the CPU workers stayed 14 to 24 % idle.  What must hold:

* a routine checkpoint is written at most every ``CHECKPOINT_EVERY_S``, and less often when its
  last write took more than ``CHECKPOINT_SHARE`` of that, up to ``CHECKPOINT_MAX_S``.  0 still
  writes every save;
* the worker pool splits its workers' time into reducing, moving jobs and results, idle with no
  job waiting and idle with jobs waiting, and a worker that comes free takes a waiting job at once;
* the runner's line per annulus gives those shares since the search started.
"""
from __future__ import annotations

import sys
import time
import types

import numpy as np
import pytest

from klip_tpe.runner import Runner


def _stub(every=10.0, last_s=0.0):
    return types.SimpleNamespace(CHECKPOINT_EVERY_S=every, CHECKPOINT_SHARE=Runner.CHECKPOINT_SHARE,
                                 CHECKPOINT_MAX_S=Runner.CHECKPOINT_MAX_S, _ckpt_stats={"last_s": last_s})


def test_the_checkpoint_interval_grows_with_a_slow_write():
    assert Runner._checkpoint_interval(_stub(10.0, 0.05)) == 10.0          # a quick write: every 10 s
    assert Runner._checkpoint_interval(_stub(10.0, 1.0)) == pytest.approx(1.0 / Runner.CHECKPOINT_SHARE)
    assert Runner._checkpoint_interval(_stub(10.0, 30.0)) == Runner.CHECKPOINT_MAX_S
    assert Runner._checkpoint_interval(_stub(0.0, 30.0)) == 0.0, "0 still writes every routine save"
    assert Runner._checkpoint_interval(_stub(1e9, 0.0)) == 1e9


def test_the_cpu_line_splits_the_workers_time():
    logs = []
    counts = {"jobs": 100, "busy_s": 300.0, "work_s": 250.0, "idle_dry_s": 50.0, "idle_queued_s": 0.0,
              "workers": 4, "t": 1000.0}
    stub = types.SimpleNamespace(log=logs.append, _cpu_counts=lambda: dict(counts))
    stub._cpu_mark = stub._cpu_counts()
    counts.update(jobs=500, busy_s=1100.0, work_s=900.0, idle_dry_s=210.0, idle_queued_s=10.0, t=1250.0)
    Runner._cpu_note(stub, 0)
    assert logs == ["  annulus 1: the 4 CPU workers were busy 80 % of the search's 4m, 65 % reducing and "
                    "15 % moving jobs and results, idle 16 % with no job waiting and 1 % with jobs waiting, "
                    "over 400 jobs"], logs
    assert stub._cpu_annuli[0] == {"annulus": 1, "jobs": 400, "search_s": 250.0, "workers": 4,
                                   "busy_fraction": 0.8, "work_fraction": 0.65, "move_fraction": 0.15,
                                   "idle_dry_fraction": 0.16, "idle_queued_fraction": 0.01}
    assert stub._cpu_mark is None
    Runner._cpu_note(stub, 1)
    assert len(logs) == 1, "no search since the last line, no line"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="forked workers")
def test_the_pool_counts_its_idle_and_reducing_time():
    from klip_tpe import GaussianPSF, KLIPReducer, PartitionedReducer
    from klip_tpe.space import Config
    from klip_tpe.synthetic import synthetic_klip_dataset
    px = 0.05
    lod = 3.89 * px / 206265
    reds = {f"n{i}": KLIPReducer(synthetic_klip_dataset(nframes=30, size=48, pa_span=50, seed=i), px, lod * 8.2,
                                 8.2, injection_model=GaussianPSF(4.0, star_flux=1e4), outrad_cap=22) for i in (1, 2)}
    par = PartitionedReducer(reds, max_workers=2, pool="processes")
    try:
        assert par.start_workers() == "processes"
        params = {"k_klip": 4, "inrad": 6, "outrad": 18, "bin": 2, "filter": 0, "angsep": 0.5, "anglemax": 35}
        c = Config(params=params, per_partition={k: dict(params) for k in reds}, selected=list(reds),
                   x=np.zeros(0))
        s0 = par.cpu_summary()
        for _ in range(3):
            par.reduce_config(c, tag="t")
        time.sleep(0.3)                                  # both workers idle, nothing waiting
        s1 = par.cpu_summary()
    finally:
        par.close()
    assert s1["jobs"] - s0["jobs"] == 6 and s1["workers"] == 2
    work, busy = s1["work_s"] - s0["work_s"], s1["busy_s"] - s0["busy_s"]
    assert work > 0 and busy >= 0.9 * work, (work, busy)
    assert s1["idle_dry_s"] - s0["idle_dry_s"] >= 0.5, "two workers idle for 0.3 s with no job waiting"
    assert s1["idle_queued_s"] - s0["idle_queued_s"] < 0.05, "a worker that comes free takes a waiting job"
