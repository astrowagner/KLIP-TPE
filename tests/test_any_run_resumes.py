"""Every run is resumable, however it was built.

``Runner.resume`` covered the command line.  A script that constructs a Runner directly --
which is how the instrument demos, the paper runs and the RX J0534 driver all work -- would
instead start a fresh search over an interrupted one: same directory, same log, history
thrown away.  That is worse than refusing, because it looks like a resume.

So resuming belongs to the Runner rather than to the CLI, and so does putting the run on the
record that ``klip-tpe runs`` and a bare ``klip-tpe resume`` read.
"""
import json
import os

import numpy as np
import pytest

# end-to-end on synthetic data (~35 s on two cores); `-m \"not slow\"` skips it
pytestmark = pytest.mark.slow

from klip_tpe import Runner
from klip_tpe import registry
from conftest import build_synthetic_run


def _run(d, **kw):
    red, space, obj, samp, cfg = build_synthetic_run(**kw)
    r = Runner(red, space, obj, samp, cfg, str(d), log=lambda s: None)
    r.run()
    return r


def _n_evals(d):
    p = os.path.join(str(d), "results.jsonl")
    return sum(1 for line in open(p) if line.strip()) if os.path.exists(p) else 0


def test_a_directly_built_runner_resumes_instead_of_starting_over(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    d = tmp_path / "run"
    _run(d, n_iter=10, n_init=4, seed=3)
    first = _n_evals(d)
    assert first >= 10 and os.path.exists(os.path.join(str(d), "checkpoint.json"))

    logs = []
    red, space, obj, samp, cfg = build_synthetic_run(n_iter=10, n_init=4, seed=3)
    r2 = Runner(red, space, obj, samp, cfg, str(d), log=logs.append)
    r2.run()
    assert r2._resumed, "a second run over the same directory started from scratch"
    assert any("resumed" in m for m in logs), logs[:5]
    assert _n_evals(d) >= first, "the history shrank"


def test_resume_never_starts_a_fresh_search(tmp_path, monkeypatch):
    """The escape hatch, for someone who really does want to redo it."""
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    d = tmp_path / "run"
    _run(d, n_iter=10, n_init=4, seed=3)
    red, space, obj, samp, cfg = build_synthetic_run(n_iter=10, n_init=4, seed=3)
    r2 = Runner(red, space, obj, samp, cfg, str(d), log=lambda s: None, resume="never")
    r2.run()
    assert not r2._resumed


def test_nothing_to_resume_is_simply_a_new_run(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    d = tmp_path / "fresh"
    r = _run(d, n_iter=8, n_init=3, seed=5)
    assert not r._resumed


def test_a_broken_checkpoint_does_not_stop_the_run(tmp_path, monkeypatch):
    """Bookkeeping must never be load-bearing: a truncated checkpoint starts a new search
    with a word of explanation, rather than raising on the way in."""
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    d = tmp_path / "run"
    _run(d, n_iter=8, n_init=3, seed=3)
    with open(os.path.join(str(d), "checkpoint.json"), "w") as f:
        f.write('{"config": {"truncated"')
    logs = []
    red, space, obj, samp, cfg = build_synthetic_run(n_iter=8, n_init=3, seed=3)
    r2 = Runner(red, space, obj, samp, cfg, str(d), log=logs.append)
    r2.run()
    assert not r2._resumed
    assert any("unreadable" in m for m in logs), logs[:5]


def test_the_resumed_configuration_comes_from_the_checkpoint(tmp_path, monkeypatch):
    """Not from whatever the caller happened to build this time: the recorded history was
    produced under the checkpoint's settings and is not comparable with any others."""
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    d = tmp_path / "run"
    _run(d, n_iter=10, n_init=4, seed=3, n_sources=3)
    red, space, obj, samp, cfg = build_synthetic_run(n_iter=10, n_init=4, seed=3, n_sources=2)
    r2 = Runner(red, space, obj, samp, cfg, str(d), log=lambda s: None)
    r2.run()
    assert r2.cfg.n_sources == 3, "the caller's settings overrode the recorded ones"


def test_any_run_appears_on_the_record(tmp_path, monkeypatch):
    """So `klip-tpe runs` can see it and a bare `klip-tpe resume` can restart it, even
    though no command line built it."""
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    d = tmp_path / "run"
    _run(d, n_iter=8, n_init=3, seed=3)
    known = [r["run_dir"] for r in registry.known_runs()]
    assert os.path.abspath(str(d)) in known, known


def test_registration_failing_never_stops_a_run(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", "/proc/nowhere-unwritable")
    r = _run(tmp_path / "run", n_iter=8, n_init=3, seed=3)
    assert r.results, "the run did not complete"
