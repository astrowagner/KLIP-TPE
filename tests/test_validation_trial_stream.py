"""The intra-candidate validation checkpoint is linear in the number of trials.

``val_candNN_trials.pkl`` used to be rewritten whole after every trial -- the clean image and
every trial image so far -- and ``checkpoint.json`` with it, so a 25 x 50 validation wrote
~1,300 image sets and 50 copies of the search history per candidate (found by the NEAR2 /
IDL session, 2026-10-07).  Now it is a header plus one appended record per trial, each with
the RNG state after its draw.  What must hold:

* one header and one record per trial; checkpoint.json is not rewritten per trial;
* a resume mid-candidate is exact (the same trials as an uninterrupted run), from the new
  stream and from the single pickle the old code wrote;
* a record cut short by the interruption is dropped and cut off the file;
* a failed write rewrites the stream whole at the next trial instead of leaving a gap.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pytest

from conftest import build_synthetic_run
from klip_tpe import ValidationConfig
from klip_tpe.runner import Runner

QUIET = lambda s: None  # noqa: E731


def _records(path):
    out = []
    with open(path, "rb") as f:
        while True:
            try:
                out.append(pickle.load(f))
            except EOFError:
                return out


def _cfg(**over):
    kw = dict(ann_edges=[8, 30], n_iter=8, n_init=3, seed=3, save_fits=False, verify=False,
              candidates=False, validation=ValidationConfig(n_top=2, n_valid=4))
    kw.update(over)
    return build_synthetic_run(**kw)


class Interrupt(BaseException):      # like Ctrl-C: not swallowed by the per-trial guards
    pass


def _interrupted(tmp_path, at="a1_val1_t2"):
    red, space, obj, samp, cfg = _cfg()
    d = str(tmp_path / "cut")
    r = Runner(red, space, obj, samp, cfg, d, log=QUIET)
    orig = r._reduce

    def cut(cfg_, src, tag="", **kw):
        if tag == at:
            raise Interrupt
        return orig(cfg_, src, tag=tag, **kw)
    r._reduce = cut
    with pytest.raises(Interrupt):
        r.run()
    return d, (red, space, obj, samp, cfg)


def _reference(tmp_path):
    red, space, obj, samp, cfg = _cfg()
    d = str(tmp_path / "ref")
    Runner(red, space, obj, samp, cfg, d, log=QUIET).run()
    return json.load(open(os.path.join(d, "annulus01", "validation.json")))


def _resume_and_compare(d, objs, ref_val, logs=None):
    red, space, obj, samp, cfg = objs
    logs = [] if logs is None else logs
    r2 = Runner.resume(d, red, obj, samp, log=logs.append)
    tags = []
    orig = r2._reduce

    def spy(cfg_, src, tag="", **kw):
        tags.append(tag)
        return orig(cfg_, src, tag=tag, **kw)
    r2._reduce = spy
    r2.run()
    cut_val = json.load(open(os.path.join(d, "annulus01", "validation.json")))
    assert [a["trials"] for a in cut_val] == [b["trials"] for b in ref_val]
    assert [a["trial_sources"] for a in cut_val] == [b["trial_sources"] for b in ref_val]
    return tags


def test_one_header_and_one_record_per_trial_and_no_checkpoint_per_trial(tmp_path):
    red, space, obj, samp, cfg = _cfg()
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "run"), log=QUIET)
    seen, ckpts = [], {"n": 0}
    save = r._save_validation_trials
    ckpt = r.checkpoint

    def spy_save(ia, ci, *a):
        save(ia, ci, *a)
        seen.append((ci, [sorted(k for k in rec) for rec in _records(r._val_trials_pkl(ia, ci))]))

    def spy_ckpt(*a, **kw):
        ckpts["n"] += 1
        return ckpt(*a, **kw)
    r._save_validation_trials = spy_save
    r.checkpoint = spy_ckpt
    n_before = {"n": 0}
    validate = r.validate

    def spy_validate(ia):
        n_before["n"] = ckpts["n"]
        return validate(ia)
    r.validate = spy_validate
    r.run()
    for ci in (0, 1):
        per = [recs for c, recs in seen if c == ci]
        assert len(per) == 4
        for t, recs in enumerate(per):
            assert len(recs) == 1 + (t + 1), "a header, then one record per finished trial"
            assert "clean" in recs[0] and "format" in recs[0]
            assert all("rng" in rec and "trials" in rec and "clean" not in rec for rec in recs[1:])
    # from validation on: its start, one per finished candidate, and the three of the
    # annulus end -- 6.  One per trial as well would be 14.
    during = ckpts["n"] - n_before["n"]
    assert during == 1 + 2 + 3, f"checkpoint.json written {during} times from a 2 x 4 validation on"


def test_resume_mid_candidate_from_the_stream(tmp_path):
    ref_val = _reference(tmp_path)
    d, objs = _interrupted(tmp_path)
    pk = os.path.join(d, "annulus01", "val_cand02_trials.pkl")
    recs = _records(pk)
    assert len(recs) == 3 and [len(x["trials"]) for x in recs[1:]] == [1, 1]
    logs = []
    tags = _resume_and_compare(d, objs, ref_val, logs)
    assert any("resuming after trial 2" in l for l in logs)
    assert "a1_val1_t0" not in tags and "a1_val1_t1" not in tags and "a1_val1_clean" not in tags
    assert not os.path.exists(pk)


def test_resume_from_the_single_pickle_the_old_code_wrote(tmp_path):
    """A run interrupted under the old code: one whole-state pickle, and checkpoint.json
    rewritten after that trial (so the RNG is there, not in the pickle)."""
    ref_val = _reference(tmp_path)
    d, objs = _interrupted(tmp_path)
    pk = os.path.join(d, "annulus01", "val_cand02_trials.pkl")
    recs = _records(pk)
    old = {"cands": recs[0]["cands"], "clean": recs[0]["clean"]}
    for k in Runner._VT_LISTS:
        old[k] = [v for rec in recs[1:] for v in rec[k]]
    with open(pk, "wb") as f:
        pickle.dump(old, f, protocol=pickle.HIGHEST_PROTOCOL)
    cp = os.path.join(d, "checkpoint.json")
    st = json.load(open(cp))
    st["rng_state"] = recs[-1]["rng"]          # what the old per-trial checkpoint held
    json.dump(st, open(cp, "w"))
    logs = []
    tags = _resume_and_compare(d, objs, ref_val, logs)
    assert any("resuming after trial 2" in l for l in logs)
    assert "a1_val1_t1" not in tags


def test_a_record_cut_short_is_dropped_and_cut_off(tmp_path):
    ref_val = _reference(tmp_path)
    d, objs = _interrupted(tmp_path, at="a1_val1_t3")
    pk = os.path.join(d, "annulus01", "val_cand02_trials.pkl")
    ends = []
    with open(pk, "rb") as f:
        while True:
            try:
                pickle.load(f)
            except EOFError:
                break
            ends.append(f.tell())
    assert len(ends) == 4                        # header + trials 0, 1, 2
    # the interruption landed inside the write of trial 2's record
    cut_at = (ends[2] + ends[3]) // 2
    blob = open(pk, "rb").read()
    with open(pk, "wb") as f:
        f.write(blob[:cut_at])
    logs = []
    tags = _resume_and_compare(d, objs, ref_val, logs)
    assert any("resuming after trial 2" in l for l in logs), logs[-12:]
    assert "a1_val1_t2" in tags, "the trial whose record was cut short is reduced again"


def test_the_cut_record_is_cut_off_the_file(tmp_path):
    d, objs = _interrupted(tmp_path, at="a1_val1_t3")
    red, space, obj, samp, cfg = objs
    pk = os.path.join(d, "annulus01", "val_cand02_trials.pkl")
    good = os.path.getsize(pk)
    with open(pk, "ab") as f:
        f.write(b"\x80\x05\x95 half a record")
    r = Runner.resume(d, red, obj, samp, log=QUIET)
    r.ia = 0
    got = r._load_validation_trials(0, 1, _records_cands(pk))
    assert got is not None and len(got["trials"]) == 3
    assert os.path.getsize(pk) == good, "the partial record is gone, the next append follows trial 2"


def _records_cands(pk):
    with open(pk, "rb") as f:
        return pickle.load(f)["cands"]


def test_a_failed_write_rewrites_the_stream_whole(tmp_path):
    red, space, obj, samp, cfg = _cfg()
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "unit"), log=QUIET)
    os.makedirs(r._ann_dir(0), exist_ok=True)
    cands, clean = [3, 5], {"clean": "image"}
    lists = {k: [] for k in Runner._VT_LISTS}

    def trial(t):
        lists["trials"].append(float(t))
        lists["trial_imgs"].append(f"img{t}")
        lists["trial_src"].append([(0.5, 10.0 * t, 1e-4)])
        lists["trial_ps"].append([float(t)])
        lists["samples_r"].append(0.5)
        lists["samples_s"].append(float(t))
        r.rng.random()                           # the draw moves the generator
        r._save_validation_trials(0, 1, cands, clean, *[lists[k] for k in Runner._VT_LISTS])

    trial(0)
    trial(1)
    r._vt_on_disk[(0, 1)] = None                 # as if the last append had failed
    trial(2)
    recs = _records(r._val_trials_pkl(0, 1))
    assert len(recs) == 2 and recs[1]["trials"] == [0.0, 1.0, 2.0], "header + everything, once"
    trial(3)
    recs = _records(r._val_trials_pkl(0, 1))
    assert len(recs) == 3 and recs[2]["trials"] == [3.0]
    state = r.rng.bit_generator.state
    r.rng.random()
    r._resumed = True
    d = r._load_validation_trials(0, 1, cands)
    assert d["trials"] == [0.0, 1.0, 2.0, 3.0] and d["trial_imgs"] == ["img0", "img1", "img2", "img3"]
    assert d["samples_s"] == [0.0, 1.0, 2.0, 3.0] and d["clean"] == clean
    assert r.rng.bit_generator.state == state, "the RNG is back where the last trial left it"
    assert r._load_validation_trials(0, 1, [3, 6]) is None, "another candidate list is not this one"
