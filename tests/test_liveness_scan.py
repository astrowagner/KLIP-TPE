"""Which processes run a run, and one writer per run (a note from the NOMIC archive session, 2026-11-25).

The scan counted any process whose command line held the run directory or its name and held
``klip_tpe``, so a ``tee -a .../klip_tpe/run_a.log`` in the resume's own pipeline blocked the resume,
an analysis script given the run counted, and a run of the same name in another folder claimed the
other's processes, which ``runs --stop`` then signaled.  A process now counts when it runs klip-tpe and
an argument resolves to the run, or when the registry recorded its pid, and a run holds a lock.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

from klip_tpe import registry
from klip_tpe.registry import Proc, RunLock, lock_holder, processes_for, runs_named

RUN = "/x/klip_tpe/run_a"


@pytest.fixture(autouse=True)
def _own_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))


def _p(pid, argv, cwd="/home/k", ppid=1):
    return Proc(pid, " ".join(argv), tuple(argv), cwd, ppid)


def test_what_names_the_run_without_running_it_does_not_count():
    table = [
        _p(11, ["tee", "-a", "/x/klip_tpe/run_a.log"]),
        _p(12, ["tee", "-a", RUN + "/resume.log"]),
        _p(13, ["python3", "plot.py", RUN]),
        _p(14, ["python3", "plot_procyon_curves.py", "out.png", RUN]),
        _p(15, ["/usr/bin/python3", "/usr/local/bin/klip-tpe", "resume", "--run-dir", "/y/klip_tpe/run_a"]),
        _p(16, ["klip-tpe", "resume", "--run-dir", "run_a"], cwd="/y/klip_tpe"),
        _p(17, ["watch", "-n", "60", "ls", RUN]),
        _p(18, ["bash", "-c", f"klip-tpe resume --run-dir {RUN} | tee -a {RUN}.log"]),
        _p(19, ["awk", "/eval/", RUN + "/results.jsonl"]),
    ]
    assert processes_for(RUN, table) == []


def test_a_klip_tpe_process_of_the_run_and_its_workers_count():
    table = [
        _p(21, ["/usr/bin/python3", "/usr/local/bin/klip-tpe", "resume", "--run-dir", RUN]),
        _p(22, ["/usr/bin/python3", "/usr/local/bin/klip-tpe", "resume", "--run-dir", RUN], ppid=21),   # a worker
        _p(23, ["klip-tpe", "resume", RUN]),
        _p(24, ["klip-tpe", "near", "--root", "/d", f"--run-dir={RUN}"]),
        _p(25, ["python", "-m", "klip_tpe.cli", "resume", "--run-dir", "run_a"], cwd="/x/klip_tpe"),
        _p(26, ["python3", "-m", "klip_tpe", "resume", "--run-dir", "./klip_tpe/run_a/"], cwd="/x"),
        _p(27, ["sh", "-c", "ffmpeg ..."], ppid=21),                   # anything the run starts
        _p(28, ["/usr/bin/python3", "/usr/local/bin/klip-tpe", "resume", "--run-dir", RUN], ppid=1),   # an orphan
        _p(29, ["tee", "-a", RUN + ".log"], ppid=30),                  # the pipeline's tee, beside it
        _p(30, ["bash"], ppid=1),
    ]
    assert processes_for(RUN, table) == [21, 22, 23, 24, 25, 26, 27, 28]


def test_a_run_a_script_started_counts_by_its_recorded_pid():
    d = "/x/klip_tpe/run_b"
    registry.register(d, argv=["run_demos.py", "A2"])
    me = os.getpid()
    rec = registry.records_for(d)[0]
    pid = int(rec["pid"])
    assert pid == me
    other = me + 100000
    table = [Proc(other, "python3 run_demos.py A2", ("python3", "run_demos.py", "A2"), "/x", 1),
             Proc(other + 1, "python3 run_demos.py A2", ("python3", "run_demos.py", "A2"), "/x", other)]
    # the recorded pid is this process, left out as the scanner itself: record the other one
    with open(registry.registry_path(), "a") as f:
        import json
        f.write(json.dumps({"run_dir": d, "pid": other, "argv": ["run_demos.py", "A2"], "started": time.time()}) + "\n")
    assert processes_for(d, table) == [other, other + 1]
    # the same pid running something else now, a pid reused, does not count
    reused = [Proc(other, "python3 other.py", ("python3", "other.py"), "/x", 1)]
    assert processes_for(d, reused) == []


def test_a_hand_built_table_of_command_lines_still_reads():
    assert processes_for(RUN, [(41, f"/usr/bin/python3 /usr/local/bin/klip-tpe near --run-dir {RUN}"),
                               (42, f"tee -a {RUN}.log")]) == [41]


def test_stop_takes_a_path_and_a_shared_name_lists_its_runs(tmp_path, monkeypatch, capsys):
    from klip_tpe import cli
    a_dir, b_dir = tmp_path / "P4" / "klip_tpe" / "run_x", tmp_path / "P34" / "klip_tpe" / "run_x"
    c_dir = tmp_path / "P4" / "klip_tpe" / "run_y"
    for d in (a_dir, b_dir, c_dir):
        d.mkdir(parents=True)
    runs = [{"run_dir": str(d), "state": "running", "pids": [1]} for d in (a_dir, b_dir, c_dir)]
    assert runs_named(str(a_dir), runs) == [runs[0]]
    assert runs_named("run_x", runs) == runs[:2] and runs_named("run_y", runs) == [runs[2]]
    assert runs_named("run_z", runs) == []
    stopped = []
    monkeypatch.setattr(registry, "known_runs", lambda root=None: [dict(r, n_eval=None, age=None, annulus=None)
                                                                     for r in runs])
    monkeypatch.setattr(registry, "stop_run", lambda d, **kw: stopped.append(d) or 1)
    monkeypatch.setattr(registry, "format_table", lambda runs: "")
    cli.cmd_runs(SimpleNamespace(root=None, stop_all=False, stop_stale=False, stop=["run_x"]))
    assert stopped == [] and "2 runs have that name" in capsys.readouterr().out
    cli.cmd_runs(SimpleNamespace(root=None, stop_all=False, stop_stale=False, stop=[str(a_dir)]))
    assert stopped == [str(a_dir)]
    cli.cmd_runs(SimpleNamespace(root=None, stop_all=False, stop_stale=False, stop=["run_y"]))
    assert stopped == [str(a_dir), str(c_dir)]


_HOLD = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {code!r})
    from klip_tpe.registry import RunLock
    lk = RunLock({d!r})
    print("held" if lk.acquire() else "refused", flush=True)
    time.sleep(120)
""")


def _holder(d):
    code = os.path.dirname(os.path.dirname(os.path.abspath(registry.__file__)))
    pr = subprocess.Popen([sys.executable, "-c", _HOLD.format(code=code, d=str(d))], stdout=subprocess.PIPE,
                          text=True)
    assert pr.stdout.readline().strip() == "held"
    return pr


def test_one_writer_holds_the_lock_and_a_crash_leaves_none(tmp_path):
    d = tmp_path / "run_l"
    d.mkdir()
    assert lock_holder(d) is None                                   # no lock file yet
    mine = RunLock(d)
    assert mine.acquire() and mine.held and lock_holder(d) is None   # this process is no other
    mine.release()
    pr = _holder(d)
    try:
        assert lock_holder(d) == pr.pid
        assert not RunLock(d).acquire()
    finally:
        pr.kill()                                                    # a crash: no release
        pr.wait(timeout=30)
    assert lock_holder(d) is None and RunLock(d).acquire()


def _state(f):
    return f.stat().st_mtime_ns, f.read_bytes()


def test_the_probe_only_reads(tmp_path):
    """``rereduce``, ``render`` and the other commands that read a run probe its lock, and change nothing."""
    d = tmp_path / "run_p"
    d.mkdir()
    mine = RunLock(d)
    assert mine.acquire()
    mine.release()
    f = d / "run.lock"
    before = _state(f)
    time.sleep(0.01)
    assert lock_holder(d) is None and _state(f) == before
    pr = _holder(d)
    try:
        held = _state(f)
        assert lock_holder(d) == pr.pid and _state(f) == held
    finally:
        pr.kill()
        pr.wait(timeout=30)
    assert lock_holder(d) is None and _state(f) == held


def test_a_probe_in_flight_refuses_no_writer(tmp_path):
    import fcntl
    import threading
    d = tmp_path / "run_s"
    d.mkdir()
    (d / "run.lock").write_text("")
    fd = os.open(d / "run.lock", os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_SH)                                 # a probe, caught holding its shared lock
    t = threading.Timer(0.1, lambda: (fcntl.flock(fd, fcntl.LOCK_UN), os.close(fd)))
    t.start()
    lk = RunLock(d)
    try:
        assert lk.acquire() and lk.held
    finally:
        t.join()
        lk.release()


def test_a_run_refuses_to_write_where_another_holds_the_lock(tmp_path):
    from conftest import build_synthetic_run
    from klip_tpe import CalibrationConfig, Runner, ValidationConfig
    d = tmp_path / "run_r"
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 20], n_iter=[4], n_init=2, param_verify=False,
                                                     calibration=CalibrationConfig(recal_budget=0),
                                                     validation=ValidationConfig(n_top=1, n_valid=1))
    r = Runner(red, space, obj, samp, cfg, str(d), log=lambda s: None)
    pr = _holder(d)
    try:
        with pytest.raises(RuntimeError, match="being written by another process"):
            r.run()
        assert not (d / "results.jsonl").exists()
    finally:
        pr.kill()
        pr.wait(timeout=30)


def test_a_resume_is_refused_while_the_lock_is_held(tmp_path):
    from klip_tpe import cli
    d = tmp_path / "root" / "run_q"
    d.mkdir(parents=True)
    registry.register(str(d), argv=["klip-tpe", "near", "--root", "/data", "--run-dir", str(d)])
    pr = _holder(d)
    try:
        with pytest.raises(SystemExit, match="its lock is held by pid"):
            cli._fill_from_registry(SimpleNamespace(root=None, run_dir=str(d)))
    finally:
        pr.kill()
        pr.wait(timeout=30)
    cli._fill_from_registry(SimpleNamespace(root=["/data"], run_dir=str(d)))       # free: no refusal
