#!/usr/bin/env python
"""Remove duplicated evaluations from a run's ``results.jsonl``.

    python scripts/dedup_results.py RUN_DIR              # report only
    python scripts/dedup_results.py RUN_DIR --write      # write RUN_DIR/results.dedup.jsonl
    python scripts/dedup_results.py RUN_DIR --in-place   # replace it (backup kept), run must be idle

``results.jsonl`` is append-only and nothing guarantees an evaluation appears in it once.
Two processes on one run directory both append -- the registry guard that refuses a second
``klip-tpe resume`` is best-effort, and a stale entry lets one through -- and a resume can
re-log what it replayed.  The duplicates are real duplicates: same ``x``, same ``score``,
only ``wall_s`` differs, because both processes evaluated the same configuration and the
later one was fighting the first for cores.

That matters downstream.  ``Runner.annulus_history`` rebuilds a completed annulus from this
file, and TPE fits a density to those points, so a repeated point doubles that
configuration's weight.  ``collect.py`` and ``klip_tpe.bench`` count the rows.

WHAT IS NOT A DUPLICATE: the calibration loop re-runs indices 0..n at each trial contrast,
so a run legitimately holds several segments that all start at ``index == 0`` with different
``contrast``.  Those are kept.  This only collapses repeats *within* a segment, keeping the
first record of each index -- the one whose ``wall_s`` was measured uncontended.

Both writing modes refuse while ``heartbeat.json`` is fresh.  ``--in-place`` would replace
the file under a writer holding it open in append mode, sending every later evaluation to an
orphaned inode.  ``--write`` is no safer in practice: its copy is a snapshot that the live
run has already outgrown by the time you read the message, so moving it over the original
silently discards everything evaluated since.  On a live run this reports and stops --
which costs nothing, because both readers drop duplicates at read time.  Clean the file
itself with ``--in-place`` once the run is finished.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Tuple

LIVE_S = 300.0          # a heartbeat younger than this means someone is still writing


def segments(recs: List[dict]) -> List[Tuple[int, int]]:
    """``(start, stop)`` of each calibration segment, per annulus, in file order."""
    out, start = [], 0
    for i, r in enumerate(recs):
        if r.get("index") == 0 and i > start:
            out.append((start, i))
            start = i
    out.append((start, len(recs)))
    return out


def dedup(lines: List[str]) -> Tuple[List[str], Dict[str, int]]:
    recs, keep_idx, bad = [], [], 0
    for i, ln in enumerate(lines):
        try:
            recs.append(json.loads(ln))
            keep_idx.append(i)
        except Exception:
            bad += 1
    drop = set()
    for a, b in segments(recs):
        seen = set()
        for j in range(a, b):
            key = (recs[j].get("annulus"), recs[j].get("index"))
            if key in seen:
                drop.add(keep_idx[j])
            else:
                seen.add(key)
    out = [ln for i, ln in enumerate(lines) if i not in drop]
    return out, {"lines": len(lines), "unparsable": bad, "dropped": len(drop), "kept": len(out)}


def live_age(run_dir: str):
    p = os.path.join(run_dir, "heartbeat.json")
    if not os.path.exists(p):
        return None
    try:
        return time.time() - float(json.load(open(p)).get("t", 0))
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--write", action="store_true", help="write RUN_DIR/results.dedup.jsonl")
    ap.add_argument("--in-place", action="store_true", help="replace results.jsonl (backup kept)")
    ap.add_argument("--force", action="store_true", help="allow --in-place on a live run (don't)")
    a = ap.parse_args(argv)

    src = os.path.join(a.run_dir, "results.jsonl")
    if not os.path.exists(src):
        raise SystemExit(f"no results.jsonl in {a.run_dir}")
    lines = open(src).read().splitlines()
    out, st = dedup(lines)

    age = live_age(a.run_dir)
    live = age is not None and age < LIVE_S
    print(f"{src}")
    print(f"  {st['lines']} lines, {st['unparsable']} unparsable")
    print(f"  {st['dropped']} duplicate(s) -> {st['kept']} kept")
    print(f"  heartbeat: " + ("none" if age is None else f"{age/60:.1f} min old"
                              + ("   *** RUN IS LIVE ***" if live else "")))
    if not st["dropped"]:
        print("  nothing to do")
        return 0

    if live and (a.in_place or a.write) and not a.force:
        raise SystemExit(
            "  refusing to write while the run is live.\n"
            "  --in-place would replace the file under a writer holding it open in append mode,\n"
            "  sending every later evaluation to an orphaned inode.  --write is no safer: its\n"
            "  copy is a snapshot the run has already outgrown, so moving it over the original\n"
            "  would discard everything evaluated since.\n"
            "  Nothing needs doing now -- Runner._load_history_jsonl and bench.read_records both\n"
            "  drop the repeats at read time.  Re-run with --in-place once the run is finished\n"
            "  (or --force if you are certain the writer is gone).")

    if a.in_place:
        bak = src + ".dup." + time.strftime("%Y%m%d_%H%M%S")
        shutil.copy2(src, bak)
        tmp = src + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(out) + "\n")
        os.replace(tmp, src)
        print(f"  backup  {bak}")
        print(f"  rewrote {src}")
    elif a.write:
        dst = os.path.join(a.run_dir, "results.dedup.jsonl")
        with open(dst, "w") as f:
            f.write("\n".join(out) + "\n")
        print(f"  wrote   {dst}")
        print("  this is a COPY; the original is untouched. Prefer --in-place on a finished run.")
    else:
        print("  (report only; --write for a copy, --in-place to replace)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
