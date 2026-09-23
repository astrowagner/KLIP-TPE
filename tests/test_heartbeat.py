"""heartbeat.json: the file that says whether a quiet run is slow or gone.

The failure this exists for: a 4-partition RX J0534 search stopped after evaluation 13 and
nothing in the run directory could say whether it was wedged, killed, or inside one slow
reduction -- its neighbour was legitimately taking 479 s for single evaluations at the time.
"""
from __future__ import annotations

import json
import os
import time

from klip_tpe import heartbeat as hbmod
from klip_tpe.heartbeat import FILENAME, Heartbeat, describe, read


def test_start_writes_a_file_a_reader_can_parse(tmp_path):
    hb = Heartbeat(str(tmp_path), period=0.05).start()
    try:
        got = read(str(tmp_path))
    finally:
        hb.stop()
    assert got is not None
    assert got["pid"] == os.getpid()
    assert got["fresh"] is True
    assert got["age"] < hbmod.FRESH_FOR


def test_the_thread_keeps_stamping_while_the_main_thread_is_busy(tmp_path):
    """The point of the daemon thread: a main thread stuck in a long reduction is exactly
    when we need to know the process is alive, so the clock cannot live in the main loop."""
    hb = Heartbeat(str(tmp_path), period=0.05).start()
    try:
        first = read(str(tmp_path))["t"]
        time.sleep(0.3)                       # stands in for a reduction: no hooks, no pumps
        second = read(str(tmp_path))["t"]
    finally:
        hb.stop()
    assert second > first


def test_stage_names_the_work_and_restarts_its_clock(tmp_path):
    hb = Heartbeat(str(tmp_path), period=5.0)
    hb.stage("annulus 1 eval 14/5000 (warmup)", annulus=0, eval=14)
    a = read(str(tmp_path))
    time.sleep(0.05)
    hb.stage("annulus 1 validation", annulus=0)
    b = read(str(tmp_path))
    assert a["stage"].startswith("annulus 1 eval 14")
    assert a["eval"] == 14
    assert b["stage"] == "annulus 1 validation"
    assert b["stage_t0"] > a["stage_t0"]
    assert b["stage_s"] < a["stage_s"] + 1.0


def test_per_step_stages_wait_for_the_next_stamp(tmp_path):
    """A stage that changes every evaluation must not rewrite the file itself: at ~1 s per
    evaluation that outran Dropbox, which filed 115 conflicted copies of heartbeat.json in
    20 minutes on H2K.  ``write=False`` makes it live at once -- the stall clock restarts --
    and leaves it to the thread's next stamp; a change of phase is still written at once."""
    hb = Heartbeat(str(tmp_path), period=5.0)              # no thread: stamps only when told
    hb.stage("annulus 1 calibration", annulus=0)           # a phase: on disk at once
    first = read(str(tmp_path))
    for ev in range(1, 50):
        hb.stage(f"annulus 1 eval {ev}/800 (tpe)", write=False, eval=ev)
    assert read(str(tmp_path))["stage"] == "annulus 1 calibration"     # 49 steps, 0 writes
    live = hb.snapshot()
    assert live["stage"] == "annulus 1 eval 49/800 (tpe)" and live["eval"] == 49
    assert live["stage_t0"] > first["stage_t0"]                       # the stall clock moved
    hb.write()                                                         # the thread's stamp
    assert read(str(tmp_path))["stage"] == "annulus 1 eval 49/800 (tpe)"


def test_update_revises_details_without_restarting_the_stage_clock(tmp_path):
    hb = Heartbeat(str(tmp_path), period=5.0)
    hb.stage("annulus 1 eval 3/100 (tpe)")
    t0 = read(str(tmp_path))["stage_t0"]
    hb.update(n_eval=77)
    hb.write()
    got = read(str(tmp_path))
    assert got["n_eval"] == 77
    assert got["stage_t0"] == t0


def test_stall_warning_fires_once_then_at_each_doubling(tmp_path):
    said = []
    hb = Heartbeat(str(tmp_path), log=said.append, period=5.0)
    hb.stage("annulus 1 eval 14/5000 (warmup)")
    hb.stall_after(0.05)
    time.sleep(0.06)
    hb._check_stall()
    hb._check_stall()                      # not yet doubled: still one warning
    assert len(said) == 1
    assert "alive" in said[0] and "resume" in said[0]
    time.sleep(0.08)                       # past 2x
    hb._check_stall()
    assert len(said) == 2


def test_a_new_stage_clears_the_warning(tmp_path):
    said = []
    hb = Heartbeat(str(tmp_path), log=said.append, period=5.0)
    hb.stall_after(0.05)
    hb.stage("eval 1")
    time.sleep(0.06)
    hb._check_stall()
    hb.stage("eval 2")                     # fresh stage, fresh clock
    hb._check_stall()
    assert len(said) == 1


def test_stall_warning_off_by_default_and_when_cleared(tmp_path):
    said = []
    hb = Heartbeat(str(tmp_path), log=said.append, period=5.0)
    hb.stage("final products")
    time.sleep(0.05)
    hb._check_stall()                      # no stall_after set
    hb.stall_after(0.01)
    hb.stall_after(None)
    time.sleep(0.02)
    hb._check_stall()
    assert said == []


def test_stop_keeps_the_stage_it_died_in(tmp_path):
    """A run killed mid-evaluation should leave the evaluation named: that is the diagnosis."""
    hb = Heartbeat(str(tmp_path), period=0.05).start()
    hb.stage("annulus 1 eval 14/5000 (warmup)")
    hb.stop()
    got = read(str(tmp_path))
    assert got["stage"].startswith("annulus 1 eval 14")


def test_stop_can_relabel_a_clean_finish(tmp_path):
    hb = Heartbeat(str(tmp_path), period=0.05).start()
    hb.stage("final products")
    hb.stop("finished")
    assert read(str(tmp_path))["stage"] == "finished"


def test_writes_are_atomic_and_leave_no_temporary_files(tmp_path):
    hb = Heartbeat(str(tmp_path), period=0.05).start()
    time.sleep(0.2)
    hb.stop()
    names = sorted(os.listdir(tmp_path))
    assert names == [FILENAME], names


def test_nothing_raises_when_the_directory_is_gone(tmp_path):
    d = tmp_path / "vanishes"
    d.mkdir()
    hb = Heartbeat(str(d), period=0.05).start()
    for f in d.iterdir():
        f.unlink()
    d.rmdir()
    hb.write()                     # must not raise: bookkeeping never stops a search
    hb.stop()
    assert read(str(d)) is None


def test_read_is_none_without_a_heartbeat_and_for_junk(tmp_path):
    assert read(str(tmp_path)) is None
    (tmp_path / FILENAME).write_text("{not json")
    assert read(str(tmp_path)) is None
    (tmp_path / FILENAME).write_text("[1, 2]")
    assert read(str(tmp_path)) is None


def test_read_falls_back_to_the_file_mtime_when_t_is_missing(tmp_path):
    (tmp_path / FILENAME).write_text(json.dumps({"pid": 1, "stage": "eval 1"}))
    got = read(str(tmp_path))
    assert got is not None and got["age"] < 30 and got["fresh"] is True


def test_a_stale_heartbeat_is_not_fresh(tmp_path):
    (tmp_path / FILENAME).write_text(json.dumps(
        {"pid": 1, "stage": "annulus 1 eval 14/5000 (warmup)",
         "t": time.time() - 3600, "stage_t0": time.time() - 3700, "stage_s": 100.0}))
    got = read(str(tmp_path))
    assert got["fresh"] is False
    assert got["age"] > 3000
    # the stage clock is read forward to now (stage_s at the last stamp + the age since),
    # so "how long has it been in this?" is honest even for a run that died in it
    assert describe(got).endswith("(62m)"), describe(got)


def test_describe_is_empty_for_no_heartbeat():
    assert describe(None) == ""
    assert describe({}) == ""


# ----------------------------------------------------------------- registry integration

def test_run_state_calls_a_run_with_a_fresh_heartbeat_running(tmp_path, monkeypatch):
    from klip_tpe import registry
    (tmp_path / "results.jsonl").write_text('{"annulus": 0}\n')
    # an evaluation that has run for an hour: the old rule called this "stalled"
    old = time.time() - 3600
    os.utime(tmp_path / "results.jsonl", (old, old))
    hb = Heartbeat(str(tmp_path), period=0.05).start()
    hb.stage("annulus 1 eval 14/5000 (warmup)")
    monkeypatch.setattr(registry, "processes_for", lambda d, table=None: [4242])
    try:
        st = registry.run_state(str(tmp_path))
    finally:
        hb.stop()
    assert st["state"] == "running"
    assert st["age"] > 1000                      # it really has not written for an hour
    assert st["stage"].startswith("annulus 1 eval 14")


def test_run_state_calls_a_quiet_heartbeat_with_live_processes_stalled(tmp_path, monkeypatch):
    from klip_tpe import registry
    (tmp_path / "results.jsonl").write_text('{"annulus": 0}\n')
    (tmp_path / FILENAME).write_text(json.dumps(
        {"pid": 4242, "stage": "annulus 1 eval 14/5000 (warmup)", "t": time.time() - 600,
         "stage_t0": time.time() - 900, "stage_s": 300.0}))
    monkeypatch.setattr(registry, "processes_for", lambda d, table=None: [4242])
    st = registry.run_state(str(tmp_path))
    assert st["state"] == "stalled"
    assert "eval 14" in registry.format_table([dict(st, argv=None, started=None, registered=False)])


def test_run_state_without_a_heartbeat_keeps_the_old_rules(tmp_path, monkeypatch):
    from klip_tpe import registry
    (tmp_path / "results.jsonl").write_text('{"annulus": 0}\n')
    monkeypatch.setattr(registry, "processes_for", lambda d, table=None: [4242])
    assert registry.run_state(str(tmp_path))["state"] == "running"
    old = time.time() - 3600
    os.utime(tmp_path / "results.jsonl", (old, old))
    assert registry.run_state(str(tmp_path))["state"] == "orphaned"
    monkeypatch.setattr(registry, "processes_for", lambda d, table=None: [])
    assert registry.run_state(str(tmp_path))["state"] == "stopped"


def test_a_dead_run_reads_as_stopped_even_with_a_stale_heartbeat(tmp_path, monkeypatch):
    """What p4 looked like: the heartbeat names the evaluation it died in, no process left."""
    from klip_tpe import registry
    (tmp_path / "results.jsonl").write_text('{"annulus": 0}\n')
    (tmp_path / FILENAME).write_text(json.dumps(
        {"pid": 4242, "stage": "annulus 1 eval 14/5000 (warmup)", "t": time.time() - 5000,
         "stage_t0": time.time() - 5100, "stage_s": 100.0}))
    monkeypatch.setattr(registry, "processes_for", lambda d, table=None: [])
    st = registry.run_state(str(tmp_path))
    assert st["state"] == "stopped"
    assert st["stage"].startswith("annulus 1 eval 14")
