"""A checkpoint that survives a crash of the machine, not only of the process.

patrick crashed hard during run_20261011_145111, and every file written in its last ~30 s
came back with 0 bytes, checkpoint.json among them.  The run had nothing to resume from.
Temp-and-rename made the writes atomic, not durable.  What must hold now:

* a durable write fsyncs the data before the rename and the directory after it;
* the save before the newest stays on disk as checkpoint.prev.json, never displaced by a
  broken checkpoint.json, and a resume takes it when checkpoint.json is empty or missing;
* no checkpoint counts more lines of results.jsonl or results.txt than had been fsynced
  when it was written, and the products of a finished annulus or hook reach the disk
  before the checkpoint that records them;
* a resume after a crash, from either save, ends where an uninterrupted run ends;
* half a line left at the end of a log is cut before anything appends to it;
* an empty heartbeat.json reads as "unknown".
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import stat

import pytest

from conftest import build_synthetic_run
from klip_tpe import Runner, ValidationConfig, durable, heartbeat, registry
from klip_tpe import runner as runner_mod
from klip_tpe.bench import read_records
from klip_tpe.param_verify import PV_KINDS
from klip_tpe.runner import load_checkpoint

QUIET = lambda s: None  # noqa: E731


class Crash(BaseException):
    """The machine stopping.  Not an Exception, so no guard swallows it."""


def _record_fsyncs(mp):
    """Stand in for os.fsync: record, in order, what each call would have made durable.
    ``("file", inode, contents)`` or ``("dir", inode, None)``."""
    events = []

    def fake(fd):
        st = os.fstat(fd)
        if stat.S_ISDIR(st.st_mode):
            events.append(("dir", st.st_ino, None))
        else:
            events.append(("file", st.st_ino, os.pread(fd, st.st_size, 0)))
    mp.setattr(os, "fsync", fake)
    return events


def _as_checkpoint(data):
    if not data or not data.startswith(b"{"):
        return None
    try:
        st = json.loads(data)
    except ValueError:
        return None
    return st if isinstance(st, dict) and "rng_state" in st and "records_count" in st else None


def _ck(records, ia=0, done=False, hooks=None, wall=1.0, extends=None):
    st = {"config": {}, "space": {}, "rng_state": {}, "ia": ia, "annulus_done": done,
          "records_count": records, "hooks_done": hooks or {}, "wall_s": wall}
    if extends is not None:
        st["extends"] = extends
    return st


def _put(d, name, st):
    with open(os.path.join(d, name), "w") as f:
        json.dump(st, f)


# ------------------------------------------------------------------ the write itself

def test_a_durable_write_fsyncs_the_file_then_its_directory(tmp_path, monkeypatch):
    events = _record_fsyncs(monkeypatch)
    p = tmp_path / "checkpoint.json"
    runner_mod._atomic_write(str(p), '{"a": 1}')
    # the data, under the inode the name now points to, then the directory holding the name
    assert events == [("file", os.stat(p).st_ino, b'{"a": 1}'), ("dir", os.stat(tmp_path).st_ino, None)]
    events.clear()
    runner_mod._atomic_write(str(p), b"two", keep=str(tmp_path / "checkpoint.prev.json"))
    assert (tmp_path / "checkpoint.prev.json").read_bytes() == b'{"a": 1}' and p.read_bytes() == b"two"
    assert events == [("file", os.stat(p).st_ino, b"two"), ("dir", os.stat(tmp_path).st_ino, None)]
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".tmp_")]


def test_a_file_system_without_fsync_still_gets_the_write_and_a_failing_disk_keeps_the_old(tmp_path, monkeypatch):
    p = tmp_path / "x.json"
    p.write_text("old")

    def refuse(fd):
        raise OSError(errno.EINVAL, "fsync not supported here")
    monkeypatch.setattr(os, "fsync", refuse)
    durable.write(str(p), "new")
    assert p.read_text() == "new"

    def fail(fd):
        raise OSError(errno.EIO, "I/O error")
    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError):
        durable.write(str(p), "newer", keep=str(tmp_path / "x.prev.json"))
    assert p.read_text() == "new" and not (tmp_path / "x.prev.json").exists()
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".tmp_")]


# ------------------------------------------------------------------ the save before the newest

def test_the_checkpoint_keeps_the_save_before_it_but_never_a_broken_one(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run()
    d = tmp_path / "run"
    r = Runner(red, space, obj, samp, cfg, str(d), log=QUIET)
    main, prev = d / "checkpoint.json", d / "checkpoint.prev.json"
    for n in (1, 2, 3):
        r.records_count = n
        r.checkpoint()
        assert json.loads(main.read_text())["records_count"] == n
        assert json.loads(prev.read_text())["records_count"] == n - 1 if n > 1 else not prev.exists()
    # the next process finds checkpoint.json emptied by a crash: the good copy stays put
    main.write_text("")
    r2 = Runner(red, space, obj, samp, cfg, str(d), log=QUIET)
    r2.records_count = 9
    r2.checkpoint()
    assert json.loads(main.read_text())["records_count"] == 9
    assert json.loads(prev.read_text())["records_count"] == 2


def test_an_empty_or_missing_checkpoint_json_resumes_from_the_save_before_it(tmp_path):
    d = str(tmp_path)
    _put(d, "checkpoint.prev.json", _ck(40))
    open(os.path.join(d, "checkpoint.json"), "w").close()               # 0 bytes, as on patrick
    said = []
    st, path = load_checkpoint(d, log=said.append)
    assert os.path.basename(path) == "checkpoint.prev.json" and st["records_count"] == 40
    assert any("resuming from checkpoint.prev.json" in s and "checkpoint.json is unreadable" in s
               for s in said), said
    os.remove(os.path.join(d, "checkpoint.json"))                       # between the two renames
    said = []
    assert os.path.basename(load_checkpoint(d, log=said.append)[1]) == "checkpoint.prev.json"
    assert any("checkpoint.json is missing" in s for s in said), said
    # a readable checkpoint.json is the newer save and wins, over a backup copy as well
    _put(d, "checkpoint.json", _ck(41, wall=2.0))
    _put(d, "checkpoint.bak.json", _ck(39))
    assert load_checkpoint(d)[1] == os.path.join(d, "checkpoint.json")


def test_an_extends_saves_outrank_the_finished_runs_copies(tmp_path):
    """Re-validating annulus 1 of a finished two-annulus run adds no evaluation and goes
    back an annulus, so by the other counts the finished run's checkpoint.prev.json looks
    further along than the extend's newest save."""
    d = str(tmp_path)
    done = {"0": ["stitch", "param_verify"], "1": ["stitch", "param_verify"]}
    _put(d, "checkpoint.prev.json", _ck(30, ia=1, done=True, hooks=done, wall=500.0))
    _put(d, "checkpoint.json", _ck(30, ia=0, done=False, hooks={"1": done["1"]}, wall=520.0, extends=1))
    assert load_checkpoint(d)[1] == os.path.join(d, "checkpoint.json")


# ------------------------------------------------------------------ logs

def test_half_a_line_is_cut_off_a_log_and_a_shared_log_starts_a_new_line(tmp_path):
    p = tmp_path / "results.jsonl"
    p.write_bytes(b'{"index": 0}\n{"index": 1}\n{"ind')
    assert "cut 5 byte(s)" in durable.repair_tail(str(p))
    assert p.read_bytes() == b'{"index": 0}\n{"index": 1}\n'
    assert durable.repair_tail(str(p)) == ""                            # a clean end is left alone
    p.write_bytes(b'{"index": 0}\n' + b"\0" * 5000)                     # zeros where data never landed
    durable.repair_tail(str(p))
    assert p.read_bytes() == b'{"index": 0}\n'
    p.write_bytes(b'{"index": 0}\n' + b"x" * 200_000)                   # longer than one read
    durable.repair_tail(str(p))
    assert p.read_bytes() == b'{"index": 0}\n'
    p.write_bytes(b"\0" * 100)                                          # not one complete line
    durable.repair_tail(str(p))
    assert p.read_bytes() == b""
    # the run registry is appended to by every run, so it is not cut: a new line starts instead
    q = tmp_path / "runs.jsonl"
    q.write_bytes(b'{"run_dir": "a"}\n{"run_d')
    durable.append_line(str(q), '{"run_dir": "b"}')
    durable.append_line(str(q), '{"run_dir": "c"}\n')
    assert q.read_bytes().splitlines()[-2:] == [b'{"run_dir": "b"}', b'{"run_dir": "c"}']


def test_an_empty_heartbeat_reads_as_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    d = tmp_path / "run_x"
    d.mkdir()
    (d / "heartbeat.json").write_text("")
    (d / "results.jsonl").write_text('{"annulus": 0, "index": 0}\n{"annu')
    assert heartbeat.read(str(d)) is None
    st = registry.run_state(str(d), table=[])
    assert st["heartbeat"] is None and st["stage"] is None and st["state"] == "stopped"
    assert st["n_eval"] == 1, "half a line is not an evaluation"
    registry.register(str(d), ["klip-tpe", "run", "--out", str(d)])
    assert "run_x" in registry.format_table(registry.known_runs())


# ------------------------------------------------------------------ a whole run

def _digest(data):
    return hashlib.sha1(data).hexdigest()


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory):
    """A whole two-annulus run (param_verify on: two partitions) with every save written,
    and every fsync recorded instead of made."""
    d = str(tmp_path_factory.mktemp("recorded") / "run")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KLIP_TPE_DATA", os.path.join(os.path.dirname(d), "home"))
        mp.setattr(Runner, "CHECKPOINT_EVERY_S", 0.0)
        events = _record_fsyncs(mp)
        r = Runner(*build_synthetic_run(n_iter=[8, 6], n_init=3, seed=4), d, log=QUIET)
        r.run()
    return r, d, events


def test_no_checkpoint_counts_more_lines_than_were_on_disk(recorded_run):
    r, d, events = recorded_run
    jl = os.stat(os.path.join(d, "results.jsonl")).st_ino
    txt = os.stat(os.path.join(d, "results.txt")).st_ino
    durable_now, n_ckpt = {}, 0
    for kind, ino, data in events:
        st = _as_checkpoint(data) if kind == "file" else None
        if st is None:
            if kind == "file":
                durable_now[ino] = data
            continue
        n_ckpt += 1
        n = st["records_count"]
        # a crash right after this checkpoint reached the disk truncates the logs to their last fsync
        assert durable_now.get(jl, b"").count(b"\n") >= n, f"checkpoint {n_ckpt} counts {n}"
        assert max(durable_now.get(txt, b"").count(b"\n") - 1, 0) >= n          # less the header row
    assert n_ckpt > 20 and r.records_count == open(os.path.join(d, "results.jsonl")).read().count("\n")


def test_what_a_checkpoint_records_is_on_disk_before_it(recorded_run):
    """The products a resume reads back for a finished annulus, and the param_verify maps
    of a finished hook, are fsynced before the first checkpoint that records them."""
    r, d, events = recorded_run
    synced, first = set(), {}
    for kind, ino, data in events:
        st = _as_checkpoint(data) if kind == "file" else None
        if st is None:
            if kind == "file":
                synced.add((ino, _digest(data)))
            continue
        for ia in range(int(st["ia"]) + (1 if st["annulus_done"] else 0)):
            first.setdefault((ia, "done"), set(synced))
        for k, hooks in (st.get("hooks_done") or {}).items():
            for h in hooks:
                first.setdefault((int(k), h), set(synced))

    def on_disk(p):
        with open(p, "rb") as f:
            return os.stat(p).st_ino, _digest(f.read())
    n = 0
    for ia in (0, 1):
        ad = os.path.join(d, f"annulus{ia + 1:02d}")
        for name in ("best_clean.fits", "best_inj.fits", "best_clean_partitions.fits", "best_inj_partitions.fits",
                     "winner.json", "validation.json", "contrast_curve.txt"):
            p = os.path.join(ad, name)
            if os.path.exists(p):
                assert on_disk(p) in first[(ia, "done")], (ia, name)
                n += 1
        for kind in PV_KINDS:
            p = os.path.join(ad, "param_verify", f"paramverify_{kind}_{ia + 1:02d}.fits")
            assert on_disk(p) in first[(ia, "param_verify")], (ia, kind)
            n += 1
    assert n >= 2 * (3 + len(PV_KINDS))


def test_a_validation_candidate_is_on_disk_before_the_checkpoint_that_pairs_with_it(recorded_run):
    """val_candNN.pkl and validation.json pair with the RNG state of the checkpoint written
    after each candidate: a resume from that checkpoint reloads them instead of validating
    the candidate again, and draws the next candidate's positions from that state."""
    import pickle
    r, d, events = recorded_run
    synced, n = [], 0
    for kind, ino, data in events:
        st = _as_checkpoint(data) if kind == "file" else None
        if st is None:
            if kind == "file":
                synced.append((ino, data))
            continue
        for ia in (0, 1):
            ad = os.path.join(d, f"annulus{ia + 1:02d}")
            rows = json.load(open(os.path.join(ad, "validation.json")))
            for ci in range(len(rows)):
                pk = os.path.join(ad, f"val_cand{ci + 1:02d}.pkl")
                with open(pk, "rb") as f:
                    blob = f.read()
                if pickle.loads(blob)["rng"] != st["rng_state"] or st["ia"] != ia:
                    continue
                # the checkpoint written after candidate ci of annulus ia
                assert (os.stat(pk).st_ino, blob) in synced, (ia, ci)
                ok = False
                for _, data2 in synced:
                    try:
                        got = json.loads(data2)
                    except ValueError:
                        continue
                    ok = ok or (isinstance(got, list) and got[:ci + 1] == rows[:ci + 1])
                assert ok, f"validation.json with candidate {ci + 1} of annulus {ia + 1} not on disk first"
                n += 1
    assert n >= 3


def test_a_param_verify_cache_entry_cut_short_is_reduced_again(recorded_run):
    r, d, _ = recorded_run
    cdir = os.path.join(d, "annulus01", "param_verify")
    caches = sorted(f for f in os.listdir(cdir) if f.startswith("cache_") and f.endswith(".npz"))
    assert caches
    open(os.path.join(cdir, caches[0]), "w").close()                    # 0 bytes, as a crash leaves it
    tags = []
    orig = r._reduce

    def spy(cfg, sources, tag="", **kw):
        tags.append(tag)
        return orig(cfg, sources, tag=tag, **kw)
    r._reduce = spy
    try:
        res = r._param_verify_hook(0, r._winner_from_disk(0))
    finally:
        r._reduce = orig
    assert res is not None
    assert sorted(tags) == ["a1_pv_clean", "a1_pv_inj"], "only the damaged entry is reduced again"
    import numpy as np
    with np.load(os.path.join(cdir, caches[0])) as z:
        assert z["clean"].ndim == 2 and z["inj"].ndim == 2


# ------------------------------------------------------------------ crash and resume, end to end

def _budget():
    # three validation candidates: a crash after the second leaves one more to draw for
    return dict(n_iter=[10, 8], n_init=4, seed=3, validation=ValidationConfig(n_top=3, n_valid=2))


def _dedup(d):
    return [(x["annulus"], x["index"], tuple(round(v, 9) for v in x["x"]), x.get("score"))
            for x in read_records(str(d), last_segment=False)]


def _validation(d, ia=0):
    return [(row["trials"], row["trial_sources"])
            for row in json.load(open(os.path.join(str(d), f"annulus{ia + 1:02d}", "validation.json")))]


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("reference") / "run"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KLIP_TPE_DATA", str(d.parent / "home"))
        Runner(*build_synthetic_run(**_budget()), str(d), log=QUIET).run()
    return d


@pytest.mark.slow
def test_a_crash_that_empties_checkpoint_json_resumes_from_the_save_before_it(tmp_path, monkeypatch, reference_run):
    """patrick's crash replayed.  The machine stops in annulus 2 as the save after its fifth
    evaluation is about to be written.  Every log loses what was written after its last
    fsync, that fifth evaluation's line among them, and keeps half a line instead.
    checkpoint.json comes back with 0 bytes.  The resume takes checkpoint.prev.json and ends
    where the uninterrupted run does."""
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    d = tmp_path / "run"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Runner, "CHECKPOINT_EVERY_S", 0.0)
        mp.setattr(Runner, "flush_checkpoint", lambda self: None)       # nothing runs on the way out
        real = Runner._write_checkpoint

        def write_checkpoint(self, text):
            st = json.loads(text)
            if st["ia"] == 1 and not st["annulus_done"] and len((st["history"] or {}).get("y", [])) == 5:
                raise Crash
            return real(self, text)
        mp.setattr(Runner, "_write_checkpoint", write_checkpoint)
        events = _record_fsyncs(mp)
        with pytest.raises(Crash):
            Runner(*build_synthetic_run(**_budget()), str(d), log=QUIET).run()
    last = {ino: data for kind, ino, data in events if kind == "file"}
    n_lines = {}
    for name, torn in (("results.jsonl", b'{"annulus": 1, "ind'), ("results.txt", b"  2     5 tpe ")):
        p = d / name
        lost = p.read_bytes()[len(last[os.stat(p).st_ino]):]
        assert lost.count(b"\n") == 1, f"{name}: the fifth evaluation's line was not fsynced yet"
        p.write_bytes(last[os.stat(p).st_ino] + torn)
        n_lines[name] = last[os.stat(p).st_ino].count(b"\n")
    ck = json.loads((d / "checkpoint.json").read_text())
    prev = json.loads((d / "checkpoint.prev.json").read_text())
    assert prev["records_count"] < ck["records_count"] <= n_lines["results.jsonl"]
    assert ck["records_count"] <= n_lines["results.txt"] - 1
    (d / "checkpoint.json").write_bytes(b"")

    said = []
    r = Runner(*build_synthetic_run(**_budget()), str(d), log=said.append)
    r.run()
    assert r._resumed
    assert any("resuming from checkpoint.prev.json" in s for s in said), said[:6]
    assert any("results.jsonl: cut" in s for s in said) and any("results.txt: cut" in s for s in said)
    for line in open(d / "results.jsonl"):
        json.loads(line)
    assert _dedup(d) == _dedup(reference_run)
    assert _validation(d, 1) == _validation(reference_run, 1)


@pytest.mark.slow
def test_a_crash_between_the_two_renames_resumes_validation_exactly(tmp_path, monkeypatch, reference_run):
    """The machine stops while the save after validation candidate 2 replaces
    checkpoint.json, after the old save moved to checkpoint.prev.json and before the new one
    took its name.  The resume starts from the save after candidate 1, whose RNG would draw
    candidate 2's positions again for candidate 3.  The pickles carry the RNG state, and
    validation goes on as if nothing happened."""
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    d = tmp_path / "run"
    real_write = durable.write
    fired = []

    def write(path, data, keep=None):
        if (os.path.basename(path) == "checkpoint.json" and keep and not fired
                and os.path.exists(os.path.join(str(d), "annulus01", "val_cand02.pkl"))):
            fired.append(path)
            with open(os.path.join(os.path.dirname(path), ".tmp_crash.json"), "w") as f:
                f.write(data)
            os.replace(path, keep)
            raise Crash
        return real_write(path, data, keep=keep)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(durable, "write", write)
        mp.setattr(Runner, "flush_checkpoint", lambda self: None)
        with pytest.raises(Crash):
            Runner(*build_synthetic_run(**_budget()), str(d), log=QUIET).run()
    assert fired and not (d / "checkpoint.json").exists()
    assert len(json.load(open(d / "annulus01" / "validation.json"))) == 2

    said = []
    r = Runner(*build_synthetic_run(**_budget()), str(d), log=said.append)
    r.run()
    assert any("resuming from checkpoint.prev.json" in s and "checkpoint.json is missing" in s for s in said)
    assert _validation(d, 0) == _validation(reference_run, 0)
    assert _validation(d, 1) == _validation(reference_run, 1)
    assert _dedup(d) == _dedup(reference_run)
