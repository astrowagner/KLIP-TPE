"""Run bookkeeping: nothing runs unseen, and a stopped run resumes with no arguments.

Two failures motivated this module and both are tested here:

* worker processes outliving a parent that was killed, invisibly, for days;
* a resume that needed the data root, the night list and a wrapper script to be
  remembered and retyped -- so in practice it was not used.
"""
import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

from klip_tpe import registry


@pytest.fixture
def reg(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    return tmp_path


def _make_run(tmp_path, name="run_20260101_010101", n_eval=50, annulus=1, done=False):
    d = tmp_path / "root" / "comb" / "opt" / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "results.jsonl", "w") as f:
        for i in range(n_eval):
            f.write(json.dumps({"annulus": annulus, "index": i}) + "\n")
    if done:
        (d / "final_results.json").write_text("{}")
    return str(d)


def test_register_and_list(reg):
    d = _make_run(reg)
    registry.register(d, argv=["klip-tpe", "near", "--root", "/data", "--run-dir", d])
    runs = registry.known_runs()
    assert [r["run_dir"] for r in runs] == [os.path.abspath(d)]
    r = runs[0]
    assert r["n_eval"] == 50 and r["annulus"] == 1 and r["registered"]
    assert r["argv"][1] == "near"
    assert "run_20260101_010101" in registry.format_table(runs)


def test_state_reflects_the_filesystem_not_the_registry(reg):
    """A registry entry must never be able to claim a run is alive."""
    d = _make_run(reg, n_eval=3)
    registry.register(d, argv=["klip-tpe", "near"])
    st = registry.run_state(d)
    assert st["pids"] == [] and st["state"] == "stopped"
    _make_run(reg, name="run_done", done=True)
    assert registry.run_state(str(reg / "root" / "comb" / "opt" / "run_done"))["state"] == "finished"


def test_a_deleted_run_directory_does_not_masquerade_as_a_run(reg):
    d = _make_run(reg, name="run_gone")
    registry.register(d, argv=["klip-tpe", "near"])
    import shutil
    shutil.rmtree(d)
    r = next(r for r in registry.known_runs() if r["run_dir"] == os.path.abspath(d))
    assert not r["exists"] and r["state"] == "stopped" and r["n_eval"] is None


@pytest.mark.parametrize("args,is_run", [
    ("/usr/bin/python /usr/local/bin/klip-tpe near --run-dir {d}", True),
    ("python -m klip_tpe.cli resume --run-dir {d}", True),
    ("bash -c echo {d}", False),
    ("grep -r klip_tpe {d}", False),
    ("/usr/bin/vim {d}/notes.txt", False),
    ("tail -f {d}/run.log", False),
    ("/usr/bin/python /usr/local/bin/other-tool --dir {d}", False),
])
def test_only_real_runs_are_counted_as_processes(args, is_run):
    """A shell, an editor or a grep that mentions a run directory is not a run -- calling
    one a run would wrongly block a resume."""
    assert registry._looks_like_a_run(args.format(d="/data/run_x")) is is_run


def test_unregistered_runs_under_a_root_are_still_found(reg):
    d = _make_run(reg, name="run_from_before")
    runs = registry.known_runs(root=str(reg / "root"))
    assert os.path.abspath(d) in [r["run_dir"] for r in runs]
    assert not runs[0]["registered"]


def test_the_newest_entry_per_run_wins(reg):
    d = _make_run(reg)
    registry.register(d, argv=["klip-tpe", "near", "--old"])
    time.sleep(0.01)
    registry.register(d, argv=["klip-tpe", "resume", "--new"])
    runs = registry.known_runs()
    assert len(runs) == 1 and runs[0]["argv"][-1] == "--new"


def test_a_broken_registry_file_degrades_quietly(reg):
    d = _make_run(reg)
    registry.register(d, argv=["klip-tpe", "near"])
    with open(registry.registry_path(), "a") as f:
        f.write("not json at all\n{partial\n")
    assert len(registry.known_runs()) == 1        # the good line survives, no exception


def test_registry_is_never_load_bearing(monkeypatch, tmp_path):
    """Registration failing (read-only home, no disk) must not affect anything."""
    monkeypatch.setenv("KLIP_TPE_DATA", "/proc/nonexistent-and-unwritable")
    registry.register(str(tmp_path))              # must not raise
    assert registry.known_runs() == []


# --------------------------------------------------------------------- orphaned workers
def test_workers_reap_themselves_when_the_parent_is_killed(tmp_path):
    """The bug this exists for: a SIGKILLed parent never runs multiprocessing's atexit
    hook, so daemon=True does not help and the workers block on the queue for ever."""
    code = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {os.getcwd()!r})
        from klip_tpe.parallel import ProcessPool
        class R:
            def reduce(self, req): return req
        p = ProcessPool({{"a": R()}}, 2, log=lambda s: None)
        print(" ".join(str(x.pid) for x in p.procs), flush=True)
        time.sleep(120)
    """)
    pr = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        line = pr.stdout.readline()
        pids = [int(x) for x in line.split()]
        assert len(pids) == 2 and all(_alive(p) for p in pids)
        pr.kill()                                  # SIGKILL: no atexit, no handler
        pr.wait(timeout=30)
        deadline = time.time() + 30
        while time.time() < deadline and any(_alive(p) for p in pids):
            time.sleep(0.5)
        assert not any(_alive(p) for p in pids), f"orphaned workers survived: {pids}"
    finally:
        if pr.poll() is None:
            pr.kill()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
