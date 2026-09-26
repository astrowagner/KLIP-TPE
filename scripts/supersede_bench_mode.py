#!/usr/bin/env python3
"""Retire one arm of a finished benchmark batch so it can be re-run cleanly.

    python3 scripts/supersede_bench_mode.py paper_runs/E2_bench grid
    BENCH_MODES=grid python3 paper_runs/run_demos.py E2

``run_benchmark`` skips finished slots and resumes check-pointed ones, which is what makes
relaunching after an interruption safe -- and exactly what stops a re-run.  Point it at a
batch whose grid arm is already recorded and it does nothing at all.

Resuming is worse than doing nothing here.  ``GridSearch.ask`` counts the grid evaluations
already in the history and calls ``cell(ngrid + k)`` with the *current* geometry, so a slot
resumed across a change in that geometry appends new cells onto a history of old ones: one
run, two designs, no record of the join.

So the arm is retired rather than deleted -- its slot directories are renamed and its rows
are moved out of ``bench_summary.txt`` into ``bench_summary_superseded.txt``, with the
reason. The other arms are untouched, keep their tag, and stay comparable, so only the
affected mode is recomputed.

Nothing is removed: the renamed directories and the moved rows are still there to compare
against.  Pass ``--dry-run`` to see what would move.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time


def current_tag(out_dir: str) -> str:
    p = os.path.join(out_dir, "bench_tag.txt")
    if not os.path.exists(p):
        raise SystemExit(f"no bench_tag.txt in {out_dir} -- is that a benchmark directory?")
    return open(p).read().strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="the benchmark directory, e.g. paper_runs/E2_bench")
    ap.add_argument("mode", help="the arm to retire, e.g. grid")
    ap.add_argument("--tag", default=None, help="batch tag (default: the one in bench_tag.txt)")
    ap.add_argument("--reason", default="superseded", help="recorded beside the moved rows")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    d = os.path.abspath(os.path.expanduser(a.out_dir))
    tag = a.tag or current_tag(d)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    # the arm's live slots only: a prefix match also took the slots an earlier retirement had
    # renamed (..._s0_superseded_20260920_150534) and stamped them a second time
    live = re.compile(rf"{re.escape(tag)}_{re.escape(a.mode)}_s\d+")

    slots = sorted(n for n in os.listdir(d)
                   if live.fullmatch(n) and os.path.isdir(os.path.join(d, n)))
    if not slots:
        raise SystemExit(f"no {a.mode!r} slots for batch {tag} in {d} -- nothing to do")

    summary = os.path.join(d, "bench_summary.txt")
    keep, moved, header = [], [], None
    if os.path.exists(summary):
        for line in open(summary):
            fields = line.split()
            if line.startswith("#"):
                header = line
                keep.append(line)
            elif fields[:2] == [tag, a.mode]:      # this batch's rows for this arm only
                moved.append(line)
            else:
                keep.append(line)

    print(f"batch {tag} in {d}")
    print(f"  {len(slots)} slot director{'y' if len(slots) == 1 else 'ies'} -> "
          f"*_{a.mode}_sN_{a.reason}_{stamp}")
    for n in slots:
        print(f"    {n}")
    print(f"  {len(moved)} summary row(s) -> bench_summary_superseded.txt")
    if a.dry_run:
        print("  (--dry-run: nothing moved)")
        return 0

    for n in slots:
        shutil.move(os.path.join(d, n), os.path.join(d, f"{n}_{a.reason}_{stamp}"))
    if moved:
        out = os.path.join(d, "bench_summary_superseded.txt")
        new = not os.path.exists(out)
        with open(out, "a") as f:
            if new and header:
                f.write(header)
            f.write(f"# {a.reason} {stamp}: {a.mode} arm of {tag} retired for re-run\n")
            f.writelines(moved)
        with open(summary, "w") as f:
            f.writelines(keep)

    stage = {"E2_bench": "E2", "F2_bench_highdim": "F2", "G2_bench_sphere": "G2",
             "H2_bench_jwst_pyklip": "H2", "H2_bench_jwst_klip": "H2K"}.get(os.path.basename(d), "<stage>")
    # ... and at the budget of the arms it will be compared with.  Printed without it, E2's
    # grid arm was re-run at the stage's default 800 evaluations beside tpe and random at the
    # 1000 long_run.sh had set.
    def budgets(lines):
        return sorted({int(f[4]) for f in (ln.split() for ln in lines)
                       if f and not f[0].startswith("#") and f[0] == tag and len(f) > 4})
    others, own = budgets(keep), budgets(moved)
    target = others or own
    env = f"BENCH_NITER={target[-1]} " if len(target) == 1 else ""
    print(f"\ndone. Re-run just that arm, keeping the batch tag and its budget"
          f"{f' (n_iter {target[0]})' if len(target) == 1 else ''}:\n"
          f"  {env}BENCH_MODES={a.mode} python3 paper_runs/run_demos.py {stage}")
    if own and others and own != others:
        print(f"  (the retired arm ran n_iter {'/'.join(map(str, own))}; the rest of the batch "
              f"{'/'.join(map(str, others))})")
    if len(others) > 1:
        print(f"  (the arms left in this batch disagree on n_iter -- {'/'.join(map(str, others))} -- "
              f"so the re-run will be refused until they agree; see bench_summary.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
