"""One evaluation must enter the optimizer's history once.

``results.jsonl`` is append-only and nothing guarantees an index appears once: two processes
on the same run directory both append (the registry guard is best-effort and a stale entry
lets a second ``klip-tpe resume`` through), and a resume can re-log what it replayed.  On
rxj0534/p4 that produced 114 duplicated evaluations -- same ``x``, same ``score``, only
``wall_s`` differing, because both processes evaluated the same configuration and the later
one was competing for cores.

A duplicated point is not harmless: TPE fits a density to these points, so a repeat doubles
that configuration's weight.  What must NOT be collapsed is the calibration loop, which
legitimately re-runs indices 0..n at each trial contrast, leaving several segments that each
start at ``index == 0``.
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "dedup_results", os.path.join(HERE, "scripts", "dedup_results.py"))
dd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dd)


def _rec(ia, idx, score, contrast=1e-4, wall=10.0, x=(1.0, 2.0)):
    return json.dumps({"annulus": ia, "index": idx, "x": list(x), "score": score,
                       "phase": "tpe", "contrast": contrast, "wall_s": wall})


def test_calibration_segments_are_not_duplicates():
    """Four seed segments at four contrasts, each replaying indices 0..2: 12 records, all kept."""
    lines = []
    for c in (2.8e-4, 1.8e-4, 1.4e-4, 1.7e-4):
        for i in range(3):
            lines.append(_rec(0, i, 3.0 + i, contrast=c))
    out, st = dd.dedup(lines)
    assert st["dropped"] == 0, "the calibration loop must survive de-duplication"
    assert len(out) == 12
    assert len(dd.segments([json.loads(l) for l in lines])) == 4


def test_a_second_writer_is_collapsed_keeping_the_first():
    """The real shape: one segment, and a second process re-logging the same indices with a
    longer wall time."""
    lines = [_rec(0, 0, 5.0)]
    lines += [_rec(0, i, 10.0 + i, wall=20.0) for i in range(1, 6)]
    lines += [_rec(0, i, 10.0 + i, wall=99.0) for i in (3, 4, 5)]      # the second writer
    out, st = dd.dedup(lines)
    assert st["dropped"] == 3 and st["kept"] == 6
    walls = [json.loads(l)["wall_s"] for l in out if json.loads(l)["index"] in (3, 4, 5)]
    assert walls == [20.0, 20.0, 20.0], "keep the uncontended timing, i.e. the first record"
    assert [json.loads(l)["index"] for l in out] == [0, 1, 2, 3, 4, 5]


def test_annuli_do_not_collide():
    lines = [_rec(0, 0, 1.0), _rec(0, 1, 2.0), _rec(1, 0, 3.0), _rec(1, 1, 4.0)]
    out, st = dd.dedup(lines)
    assert st["dropped"] == 0, "annulus 1 index 0 is not a repeat of annulus 0 index 0"


def test_unparsable_lines_are_counted_and_kept():
    lines = [_rec(0, 0, 1.0), "{truncated", _rec(0, 1, 2.0), _rec(0, 1, 2.0)]
    out, st = dd.dedup(lines)
    assert st["unparsable"] == 1 and st["dropped"] == 1
    assert "{truncated" in out, "a half-written line is evidence, not garbage to discard"


def test_the_loader_drops_duplicates_at_read_time(tmp_path):
    """Belt and braces: even on a file nobody cleaned, the history must not double-count."""
    from klip_tpe.runner import Runner

    src = open(os.path.join(HERE, "klip_tpe", "runner.py")).read()
    i = src.index("def _load_history_jsonl")
    body = src[i:i + 2200]
    assert "seen, uniq, ndup" in body, "the read-time guard has to stay"
    assert "uniq.append(r)" in body and "ndup += 1" in body

    class _Fake:
        space = type("S", (), {"ndim": 2})()
        cfg = type("C", (), {"contrast0": 1e-4})()
        def __init__(self, d): self.run_dir = d
        def log(self, m): self.logged = m
    f = _Fake(str(tmp_path))
    with open(os.path.join(str(tmp_path), "results.jsonl"), "w") as fh:
        fh.write(_rec(0, 0, 1.0) + "\n")
        for i in range(1, 4):
            fh.write(_rec(0, i, float(i)) + "\n")
        fh.write(_rec(0, 3, 3.0, wall=99.0) + "\n")          # the duplicate
    h, c = Runner._load_history_jsonl(f, 0)
    assert h is not None and len(h) == 4, f"expected 4 unique points, got {len(h)}"
    assert getattr(f, "logged", "").find("duplicated") > 0


def test_read_records_drops_repeats_but_keeps_calibration_segments(tmp_path):
    """bench.read_records feeds the paper's evaluation counts and running-best curves, so it
    has to make the same distinction the loader does."""
    from klip_tpe import bench

    p = tmp_path / "results.jsonl"
    lines = []
    for c in (2.8e-4, 1.7e-4):                      # two calibration segments
        for i in range(3):
            lines.append(_rec(0, i, 3.0 + i, contrast=c))
    lines += [_rec(0, i, 10.0 + i) for i in range(3, 8)]
    lines += [_rec(0, i, 10.0 + i, wall=99.0) for i in (5, 6, 7)]     # a second writer
    p.write_text("\n".join(lines) + "\n")

    last = bench.read_records(str(tmp_path))
    assert [r["index"] for r in last] == [0, 1, 2, 3, 4, 5, 6, 7], "repeats gone, order kept"
    assert [r["wall_s"] for r in last if r["index"] in (5, 6, 7)] == [10.0] * 3

    every = bench.read_records(str(tmp_path), last_segment=False)
    assert len(every) == 11, "both calibration segments survive when the caller wants them all"
    assert sum(1 for r in every if r["index"] == 0) == 2


def test_read_records_is_idempotent_on_a_clean_file(tmp_path):
    from klip_tpe import bench
    lines = [_rec(0, i, float(i)) for i in range(6)]
    (tmp_path / "results.jsonl").write_text("\n".join(lines) + "\n")
    assert len(bench.read_records(str(tmp_path))) == 6
