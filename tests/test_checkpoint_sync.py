"""Checkpoints that survive a synced folder.

What happened (2026-09-23, paper_runs/ inside Dropbox): H2K evaluates in ~1 s and rewrote a
200 kB checkpoint.json after every evaluation.  Dropbox could not keep up -- 49 "conflicted
copies" of it in 20 minutes -- and once put its own older, online-only version back under
the real name while the newest save (800 evaluations, annulus done, param_verify recorded)
sat in ``checkpoint (Kevin Wagner's conflicted copy 2026-09-23 48).json``.  A resume trusting
the name would have gone back in time; one that could not read the placeholder would have
started over.

So: routine (per-evaluation) saves are written at most every ``CHECKPOINT_EVERY_S`` and the
one held back is flushed on any exit; and a resume reads every copy and takes the one
furthest along.
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

from klip_tpe import Runner
from klip_tpe import runner as runner_mod
from klip_tpe.runner import checkpoint_candidates, load_checkpoint
from conftest import build_synthetic_run

DROPBOX = "checkpoint (Kevin Wagner's conflicted copy 2026-09-23 {}).json"


def _ck(records, ia=0, done=False, hooks=None, wall=1.0):
    return {"config": {}, "space": {}, "rng_state": {}, "ia": ia, "annulus_done": done,
            "records_count": records, "hooks_done": hooks or {}, "wall_s": wall}


def _put(d, name, st):
    with open(os.path.join(d, name), "w") as f:
        json.dump(st, f)


# ------------------------------------------------------------------ choosing the copy

def test_resume_takes_the_copy_furthest_along(tmp_path):
    d = str(tmp_path)
    assert load_checkpoint(d) == (None, None) and checkpoint_candidates(d) == []
    # the H2K case: the real name holds the older save, the copy the newest
    _put(d, "checkpoint.json", _ck(800, wall=990.0))
    _put(d, DROPBOX.format(47), _ck(750, wall=900.0))
    _put(d, DROPBOX.format(48), _ck(800, done=True, hooks={"0": ["param_verify"]}, wall=1007.0))
    _put(d, "checkpoint 2.json", {"not": "a checkpoint"})                  # junk is skipped
    said = []
    st, path = load_checkpoint(d, log=said.append)
    assert os.path.basename(path) == DROPBOX.format(48) and st["annulus_done"] is True
    assert any("resuming from" in s and "conflicted copy 2026-09-23 48" in s for s in said), said
    assert any("checkpoint 2.json unreadable" in s for s in said), said


def test_a_tie_goes_to_the_real_name_and_an_unreadable_one_is_skipped(tmp_path):
    d = str(tmp_path)
    _put(d, "checkpoint.json", _ck(10))
    _put(d, DROPBOX.format(1), _ck(10))
    assert load_checkpoint(d)[1] == os.path.join(d, "checkpoint.json")
    with open(os.path.join(d, "checkpoint.json"), "w") as f:              # an online-only
        f.write('{"config": {"trunc')                                     # placeholder, say
    said = []
    st, path = load_checkpoint(d, log=said.append)
    assert os.path.basename(path) == DROPBOX.format(1)
    assert any("checkpoint.json is unreadable" in s for s in said), said


def test_a_checkpoint_ahead_of_results_jsonl_is_reported(tmp_path):
    d = str(tmp_path)
    _put(d, "checkpoint.json", _ck(12))
    with open(os.path.join(d, "results.jsonl"), "w") as f:
        f.write("{}\n" * 9)
    said = []
    load_checkpoint(d, log=said.append)
    assert any("results.jsonl holds 9 lines" in s for s in said), said


def test_a_slot_with_only_a_synced_copy_is_resumable(tmp_path):
    from klip_tpe import bench
    d = bench.slot_dir(str(tmp_path), "T0", "tpe", 0)
    os.makedirs(d)
    _put(d, DROPBOX.format(3), _ck(40))
    st = bench.bench_status(str(tmp_path), bench_tag="T0", modes=("tpe",), seeds=(0,))
    assert st[0]["status"] == "resumable", st


# ------------------------------------------------------------------ holding routine saves

def test_routine_saves_are_held_and_flushed(tmp_path, monkeypatch):
    red, space, obj, samp, cfg = build_synthetic_run()
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "run"), log=lambda s: None)
    written = []
    real = runner_mod._atomic_write
    monkeypatch.setattr(runner_mod, "_atomic_write",
                        lambda p, t: (written.append(os.path.basename(p)), real(p, t)))
    ck = tmp_path / "run" / "checkpoint.json"
    for i in range(20):                        # twenty evaluations inside CHECKPOINT_EVERY_S
        r.records_count = i + 1
        r.checkpoint(routine=True)
    assert written.count("checkpoint.json") == 1                  # the first; the rest held
    assert json.loads(ck.read_text())["records_count"] == 1
    r.flush_checkpoint()
    assert written.count("checkpoint.json") == 2
    assert json.loads(ck.read_text())["records_count"] == 20       # the LAST one taken
    r.flush_checkpoint()                                           # nothing held: no write
    assert written.count("checkpoint.json") == 2
    r.records_count = 21
    r.checkpoint(routine=True)                                     # still inside the gap: held
    r.checkpoint(final_annulus=True)                               # a phase save: at once,
    assert written.count("checkpoint.json") == 3                   # and it supersedes the held
    assert r._ckpt_pending is None
    st = json.loads(ck.read_text())
    assert st["records_count"] == 21 and st["annulus_done"] is True


# ------------------------------------------------------------------ end to end

def _n_evals(d):
    p = os.path.join(str(d), "results.jsonl")
    return sum(1 for line in open(p) if line.strip()) if os.path.exists(p) else 0


def _records(d):
    """(annulus, index, x, score) of every line, in order."""
    out = []
    for line in open(os.path.join(str(d), "results.jsonl")):
        r = json.loads(line)
        out.append((r["annulus"], r["index"], tuple(round(v, 9) for v in r["x"]), r.get("score")))
    return out


def _interrupting(monkeypatch):
    """Raise KeyboardInterrupt as search evaluation ``stop[0]`` of annulus ``stop[1]`` is
    about to start -- between two finished evaluations, as a Ctrl-C lands."""
    stop = {"at": None}
    real = Runner._hb_eval

    def hb_eval(self, ia, ev, n_iter, phase):
        if stop["at"] == (ia, ev):
            raise KeyboardInterrupt
        return real(self, ia, ev, n_iter, phase)
    monkeypatch.setattr(Runner, "_hb_eval", hb_eval)
    return stop


BUDGET = dict(n_iter=[10, 8], n_init=4, seed=3)


@pytest.mark.slow
def test_an_interrupted_run_keeps_its_last_evaluation_and_resumes_exactly(tmp_path, monkeypatch):
    """With every routine save held back, a Ctrl-C still leaves the checkpoint at the last
    finished evaluation (the flush on the way out), and the resumed run ends exactly where
    an uninterrupted one does: same evaluations, same scores, none logged twice."""
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    ref = tmp_path / "uninterrupted"
    Runner(*build_synthetic_run(**BUDGET), str(ref), log=lambda s: None).run()

    monkeypatch.setattr(Runner, "CHECKPOINT_EVERY_S", 1e9)
    stop = _interrupting(monkeypatch)
    d = tmp_path / "interrupted"
    stop["at"] = (0, 7)
    with pytest.raises(KeyboardInterrupt):
        Runner(*build_synthetic_run(**BUDGET), str(d), log=lambda s: None).run()
    st = json.loads((d / "checkpoint.json").read_text())
    assert st["records_count"] == _n_evals(d), "the held save was not flushed on the way out"

    stop["at"] = None
    r = Runner(*build_synthetic_run(**BUDGET), str(d), log=lambda s: None)
    r.run()
    assert r._resumed
    assert _records(d) == _records(ref)


@pytest.mark.slow
def test_a_resume_after_dropbox_swapped_the_checkpoint_continues_from_the_newest(tmp_path, monkeypatch):
    """The H2K case end to end: the newest save moved to a conflicted copy, an older one put
    back under the real name.  Resuming from the older would replay evaluations and log
    them twice; resuming from the copy continues where the run was."""
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    monkeypatch.setattr(Runner, "CHECKPOINT_EVERY_S", 0.0)       # every save written
    stop = _interrupting(monkeypatch)
    d = tmp_path / "run"
    stop["at"] = (0, 5)
    with pytest.raises(KeyboardInterrupt):
        Runner(*build_synthetic_run(**BUDGET), str(d), log=lambda s: None).run()
    older = (d / "checkpoint.json").read_text()
    stop["at"] = (0, 9)
    with pytest.raises(KeyboardInterrupt):
        Runner(*build_synthetic_run(**BUDGET), str(d), log=lambda s: None).run()
    newest = json.loads((d / "checkpoint.json").read_text())
    assert newest["records_count"] > json.loads(older)["records_count"]

    shutil.move(str(d / "checkpoint.json"), str(d / DROPBOX.format(48)))
    (d / "checkpoint.json").write_text(older)

    stop["at"] = None
    said = []
    r = Runner(*build_synthetic_run(**BUDGET), str(d), log=said.append)
    r.run()
    assert any("resuming from" in s and "conflicted copy" in s for s in said), said[:8]
    # a repeat within a segment is a replay (calibration segments legitimately restart at 0)
    from klip_tpe.bench import read_records
    assert len(read_records(str(d), last_segment=False)) == _n_evals(d), \
        "evaluations were replayed: it resumed from the older save"
