#!/usr/bin/env python
"""The benchmark numbers the paper quotes (Section 4, Table 4), from the batch directories.

    python scripts/bench_numbers.py paper_runs/E2_bench paper_runs/F2_bench_highdim \\
        paper_runs/G2_bench_sphere paper_runs/H3_bench_jwst_pyklip_sub paper_runs/H3_bench_jwst_klip_sub

Per batch: the validated mean +/- sd of each arm, the paired TPE - other differences with their
t statistic and wins (klip_tpe.bench.summarize_bench), the search-to-validated inflation of each
arm, the speed factor -- the first evaluation at which the TPE's mean running-best SEARCH score
reaches the other arm's final mean, as a fraction of the budget -- and the raw-scale inflation
(the raw median injected S/N at each run's best search-score evaluation, minus the run's
validated score).  Pooled over every run given: the median raw - search excess at each run's
best evaluation.
"""
from __future__ import annotations

import os
import statistics
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klip_tpe.bench import bench_convergence, read_records, read_summary, summarize_bench, summary_run_dirs  # noqa: E402


def speed_factor(tpe_mean: np.ndarray, other_final: float) -> Optional[Dict[str, float]]:
    """First evaluation (1-based) at which the TPE's mean running-best reaches ``other_final``,
    and the budget over it; ``None`` if it never does."""
    hit = np.where(np.asarray(tpe_mean, float) >= float(other_final))[0]
    if not hit.size:
        return None
    e = int(hit[0]) + 1
    return {"eval": e, "n": int(len(tpe_mean)), "factor": len(tpe_mean) / e}


def raw_scale(out_dir: str) -> Dict[str, List[Dict[str, float]]]:
    """Per arm, per run: raw - search at the best search-score evaluation, and raw - validated."""
    per: Dict[str, List[Dict[str, float]]] = {}
    for r in read_summary(out_dir):
        d = os.path.join(out_dir, f"{r['bench_tag']}_{r['mode']}_s{r['seed']}")
        recs = [x for x in read_records(d, r["annulus"]) if x.get("score") is not None and x.get("raw_score") is not None]
        if not recs:
            continue
        b = max(recs, key=lambda x: x["score"])
        per.setdefault(r["mode"], []).append({"raw_minus_search": float(b["raw_score"]) - float(b["score"]),
                                              "raw_inflation": float(b["raw_score"]) - float(r["validated_best"])})
    return per


def main(argv=None) -> int:
    dirs = list(argv if argv is not None else sys.argv[1:])
    pooled: List[float] = []
    for d in dirs:
        print(f"\n===== {d}")
        s = summarize_bench(d, log=print)
        for m, st in s["modes"].items():
            print(f"   inflation (search - validated) {m}: {st['optimism']:.2f}")
        runs = summary_run_dirs(d)
        anns = sorted({int(r["annulus"]) for r in read_summary(d)})
        for ia in anns:
            conv = bench_convergence(runs, annulus=ia)
            if "tpe" not in conv:
                continue
            for other in [m for m in conv if m != "tpe"]:
                sf = speed_factor(conv["tpe"]["mean"], conv[other]["mean"][-1])
                txt = ("never" if sf is None else f"at eval {sf['eval']} of {sf['n']} -> factor {sf['factor']:.1f}")
                print(f"   annulus {ia + 1}: TPE mean running-best reaches {other}'s final "
                      f"{conv[other]['mean'][-1]:.3f} {txt}")
        rs = raw_scale(d)
        for m, v in rs.items():
            pooled.extend(x["raw_minus_search"] for x in v)
            print(f"   raw scale {m}: raw - search at best {statistics.mean(x['raw_minus_search'] for x in v):+.2f}, "
                  f"raw inflation {statistics.mean(x['raw_inflation'] for x in v):.2f}")
    if pooled:
        print(f"\npooled median raw - search at each run's best evaluation: {statistics.median(pooled):+.2f} "
              f"over {len(pooled)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
