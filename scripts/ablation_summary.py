#!/usr/bin/env python
"""The numbers the paper quotes from library_ablation.py's output, in the paper's convention.

    python scripts/ablation_summary.py RUN/library_ablation.json
    python scripts/ablation_summary.py A/library_ablation_at_B_contrast.json --versus B/library_ablation.json

library_ablation.py logs the MEAN over its draws; the paper's tables and text use the MEDIAN
over the same draws, and count the draws on which one configuration is ahead of another
(Table 3 and Sections 3.4-3.5).  This prints, per annulus, every configuration's median, the
winner's ratio of medians to each, and the number of draws on which the winner is ahead; with
``--versus``, the head to head of the two runs' winners on common draws (same seed, draw count
and contrast -- one of the two made with ``--contrast-from`` the other).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import numpy as np


def _med(v) -> float:
    return float(np.nanmedian(np.asarray(v, float)))


def summarize(ab: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per annulus: {config: {"median", "ratio" (winner / config), "winner_ahead", "n"}}."""
    out = []
    for A in ab["annuli"]:
        cf = A["configs"]
        w = np.asarray(cf["winner"]["raw"], float)
        row = {"annulus": A["annulus"], "contrast": A["contrast"], "configs": {}}
        for name, rec in cf.items():
            x = np.asarray(rec["raw"], float)
            row["configs"][name] = {"median": _med(x), "ratio": _med(w) / _med(x),
                                    "winner_ahead": int(np.sum(w > x)), "n": int(x.size),
                                    "params": rec.get("params"), "planet_snr": rec.get("planet_snr")}
        out.append(row)
    return out


def head_to_head(mine: Dict[str, Any], theirs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Winner against winner on common draws: ratio of medians and draws the first is ahead on.
    Refuses annuli whose draws or contrasts differ."""
    by = {B["annulus"]: B for B in theirs["annuli"]}
    rows = []
    for A in mine["annuli"]:
        B = by.get(A["annulus"])
        if B is None:
            continue
        da, db = np.asarray(A["draws"], float), np.asarray(B["draws"], float)
        if da.shape != db.shape or not np.allclose(da, db) or not np.isclose(float(A["contrast"]), float(B["contrast"])):
            raise ValueError(f"annulus {A['annulus']}: the two ablations were not made on the same draws and "
                             f"contrast (use --contrast-from the other's run directory)")
        a = np.asarray(A["configs"]["winner"]["raw"], float)
        b = np.asarray(B["configs"]["winner"]["raw"], float)
        rows.append({"annulus": A["annulus"], "contrast": A["contrast"], "median_first": _med(a),
                     "median_second": _med(b), "ratio": _med(a) / _med(b), "first_ahead": int(np.sum(a > b)),
                     "n": int(a.size)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ablation", help="a library_ablation.py --out JSON")
    ap.add_argument("--versus", default=None, help="the other run's ablation JSON, on the same draws and contrast")
    a = ap.parse_args(argv)
    ab = json.load(open(a.ablation))
    print(f"{a.ablation}  ({ab.get('backend')}, {ab.get('n_draws')} draws)")
    for row in summarize(ab):
        print(f"annulus {row['annulus']}  contrast {row['contrast']:.3e}")
        for name, r in row["configs"].items():
            pl = r["planet_snr"]
            print(f"   {name:12s} median {r['median']:6.2f}   winner/this {r['ratio']:5.3f}   winner ahead "
                  f"{r['winner_ahead']:2d}/{r['n']}   planet {'--' if pl is None else format(pl, '.2f')}")
    if a.versus:
        th = json.load(open(a.versus))
        print(f"head to head: {ab.get('backend')} winner / {th.get('backend')} winner")
        for r in head_to_head(ab, th):
            print(f"   annulus {r['annulus']}: {r['median_first']:.2f} vs {r['median_second']:.2f}  x{r['ratio']:.3f}  "
                  f"first ahead on {r['first_ahead']}/{r['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
