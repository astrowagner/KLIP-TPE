"""The run's main thread, measured from inside the run.

From the NEAR2 note of 2026-10-12: with four trials in flight (552d9cc) the search ran
0.50 s per evaluation over evaluations 501-2500 and 0.67 s over 4501-5040, at a constant
bin mix, so something on the main thread grew with the history.  The routine checkpoint
is now serialised only when it is written (tests/test_checkpoint_sync.py) and a TPE
proposal computes all its candidates' densities at once (tests/test_optimizers.py).  What
must hold here:

* ``KLIP_TPE_PROFILE_MAIN`` makes a run write ``main_thread_profile.txt``, and without it
  nothing starts;
* the profile names the functions that held the main thread, with its waits counted as
  ``(waiting)``.
"""
from __future__ import annotations

import re
import threading
import time

from conftest import build_synthetic_run
from klip_tpe.profiling import ENV, FILENAME, MainThreadSampler, sampler_from_env
from klip_tpe.runner import Runner


def _share(text, section, name):
    """The 'last' share of ``name`` in the given table of a profile report."""
    part = text.split("# inclusive")[1] if section == "inclusive" else text.split("# where the stack ended")[1]
    if section == "inclusive":
        part = part.split("# where the stack ended")[0]
    for line in part.splitlines():
        if line.rstrip().endswith(name):
            return float(re.findall(r"([\d.]+)%", line)[0])
    return 0.0


def _busy(seconds):
    end = time.time() + seconds
    while time.time() < end:
        sum(range(2000))


def test_the_profile_names_where_the_time_goes(tmp_path):
    path = tmp_path / FILENAME
    s = MainThreadSampler(str(path), every_s=1000.0, period_s=0.002, status=lambda: "here").start()
    gate = threading.Event()
    for _ in range(6):
        _busy(0.08)
        gate.wait(0.08)                    # blocked on a lock: idle, not working
    s.stop()
    text = path.read_text()
    busy = _share(text, "inclusive", "test_main_thread.py:_busy")
    idle = _share(text, "leaf", "(waiting)")
    assert 25.0 < busy < 75.0 and 25.0 < idle < 75.0, text
    assert "(here -> here)" in text and "samples" in text


def test_a_run_writes_its_profile_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    monkeypatch.delenv(ENV, raising=False)
    said = []
    assert sampler_from_env(str(tmp_path), log=said.append) is None and not said
    monkeypatch.setenv(ENV, "soon")
    assert sampler_from_env(str(tmp_path), log=said.append) is None
    assert any("not a number" in s for s in said)

    d = tmp_path / "quiet"
    monkeypatch.delenv(ENV)
    Runner(*build_synthetic_run(n_iter=[6], ann_edges=[8, 30], n_init=3), str(d), log=lambda s: None).run()
    assert not (d / FILENAME).exists()

    d = tmp_path / "asked"
    monkeypatch.setenv(ENV, "1")
    said = []
    Runner(*build_synthetic_run(n_iter=[6], ann_edges=[8, 30], n_init=3), str(d), log=said.append).run()
    assert any("profiling the main thread" in s for s in said)
    text = (d / FILENAME).read_text()
    assert _share(text, "inclusive", "klip_tpe/runner.py:_run") > 0 or "klip_tpe/runner.py:_run" in text
    assert "evaluation" in text.splitlines()[1] or "evaluation" in text
