#!/usr/bin/env python
"""The main thread's own work per evaluation, with the reductions taken out of the way.

Every partition returns its image at once (a fixed noise frame plus the injected sources), so
the search's pace is set by what the main thread does with each trial: scoring the draws,
asking the TPE for the next proposal, and the files, the checkpoint and the log.  The script
runs the same search several ways and prints the median interval between two evaluations'
dispatches, in milliseconds:

    no early scoring, no cache    the draws scored on the main thread, each image filtered
                                  for every call (the code before 8cfc1fe)
    no early scoring, cache       each image filtered once (8cfc1fe)
    early scoring, cache          the draws scored in the threads that ran their reductions
    scoring stubbed               the floor: the main thread with no scoring at all

    python scripts/bench_main_thread.py [--size 150] [--fwhm 6.5] [--nights 6] [--n-iter 120]
                                        [--n-remeasure 5] [--batch 4] [--workers 8]

On a machine with spare cores the early scoring runs beside the main thread.  On two cores it
competes with it, so this sandbox's numbers understate the gain.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import types

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_root, "klip_tpe")) and _root not in sys.path:
    sys.path.insert(0, _root)

from klip_tpe import metrics  # noqa: E402
from klip_tpe.metrics import MawetPeakSNR, Objective, ScoreResult, source_xy  # noqa: E402
from klip_tpe.positions import PositionSampler  # noqa: E402
from klip_tpe.reducer import PartitionedReducer, Reducer, ReductionResult  # noqa: E402
from klip_tpe.runner import CalibrationConfig, RunConfig, Runner, ValidationConfig  # noqa: E402
from klip_tpe.space import Param, SearchSpace, kgrid  # noqa: E402


class InstantReducer(Reducer):
    """A partition that reduces at once: a fixed noise frame, and each injected source as a
    Gaussian of the FWHM at its position and contrast."""

    def __init__(self, size: int, pxscale: float, fwhm: float, seed: int, rotation: float = 150.0):
        self.pxscale, self.fwhm = float(pxscale), float(fwhm)
        self.lam_over_d_px = self.fwhm / 1.028
        self.base = np.random.default_rng(seed).normal(size=(size, size)).astype(np.float32)
        self.c = (size - 1) / 2.0
        self.yy, self.xx = np.mgrid[0:size, 0:size]
        self.rotation = float(rotation)
        # the frame shape the source count's probe reads (Runner._probe_shape): with NEAR's
        # 150 deg of rotation the count goes to one source a draw, as on the real nights
        self.data = types.SimpleNamespace(cube=np.zeros((2, size, size), np.float32))

    def frame_angles(self, pid=None):
        return np.linspace(-self.rotation / 2.0, self.rotation / 2.0, 60)

    def reduce(self, req):
        img = self.base.copy()
        s2 = 2.0 * (self.fwhm / 2.3548) ** 2
        for s in req.injections or ():
            x, y = source_xy([s.rho], [s.theta], self.pxscale, self.c, self.c)
            img += (3.0e5 * s.contrast * np.exp(-((self.xx - x[0]) ** 2 + (self.yy - y[0]) ** 2) / s2)).astype(np.float32)
        return ReductionResult(img, 1.0, {"tag": req.tag})


def _pace_ms(run_dir: str) -> float:
    recs = [json.loads(ln) for ln in open(os.path.join(run_dir, "results.jsonl")) if ln.strip()]
    t = sorted(r["meta"]["t_dispatch"] for r in recs if r["phase"] not in ("seed", "reseed", "calib"))
    d = np.diff(t)
    return float(np.median(d[len(d) // 5:]) * 1e3) if d.size > 5 else float("nan")


def run_once(a, label: str, early: bool, cache: int, stub: bool) -> float:
    parts = [f"n{i + 1}" for i in range(a.nights)]
    reds = {p: InstantReducer(a.size, a.pxscale, a.fwhm, seed=i) for i, p in enumerate(parts)}
    pr = PartitionedReducer(reds, max_workers=a.workers, pool=a.pool)
    block = [Param("bin", 5, 30, "int", default=12), Param("k_klip", 1, 30, "int", grid=kgrid(30), default=5),
             Param("angsep", 0.5, 2.0, default=1.0), Param("corr_thresh", 0.8, 1.0, default=0.9)]
    space = SearchSpace().replicate(block, parts).with_selection("two_slot", partitions=parts)
    obj = Objective(MawetPeakSNR(pxscale=a.pxscale, fwhm=a.fwhm), clean_subtract=True)
    if stub:
        obj.score_search = lambda img, srcs, clean=None, fm=None: ScoreResult(
            5.0, np.full(len(srcs), 5.0), np.full(len(srcs), 5.0), np.zeros(len(srcs)))
    samp = PositionSampler(fwhm_as=a.fwhm * a.pxscale)
    cfg = RunConfig(ann_edges=[a.inner, a.outer], n_iter=[a.n_iter], n_init=min(20, a.n_iter // 3), seed=3,
                    n_remeasure=a.n_remeasure, batch=a.batch, validation=ValidationConfig(n_top=1, n_valid=1),
                    calibration=CalibrationConfig(forced=[1e-4]), fm_curve=False, param_verify=False)
    metrics.PREPARED_CACHE_SIZE = cache
    metrics.clear_prepared_cache()
    run_dir = tempfile.mkdtemp(prefix="bench_main_")
    try:
        r = Runner(pr, space, obj, samp, cfg, run_dir, log=lambda s: None)
        r.SCORE_EARLY = early
        t0 = time.time()
        r.run()
        pace = _pace_ms(run_dir)
        print(f"  {label:<32s} {pace:8.1f} ms per evaluation   (run {time.time() - t0:5.1f} s)", flush=True)
        return pace
    finally:
        pr.close()
        shutil.rmtree(run_dir, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", type=int, default=150, help="image side (px)")
    ap.add_argument("--fwhm", type=float, default=6.5, help="FWHM (px), NEAR's by default")
    ap.add_argument("--pxscale", type=float, default=0.0453)
    ap.add_argument("--inner", type=float, default=10.0, help="the annulus' inner edge (px)")
    ap.add_argument("--outer", type=float, default=40.0, help="the annulus' outer edge (px)")
    ap.add_argument("--nights", type=int, default=6)
    ap.add_argument("--n-iter", type=int, default=120)
    ap.add_argument("--n-remeasure", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pool", default="threads", choices=["threads", "processes"])
    a = ap.parse_args(argv)
    print(f"{a.nights} nights of {a.size} px at FWHM {a.fwhm:g} px, {a.n_iter} evaluations of "
          f"{a.n_remeasure} draws, {a.batch} trials in flight, {a.workers} workers ({a.pool})")
    run_once(a, "no early scoring, no cache", False, 0, False)
    run_once(a, "no early scoring, cache", False, 8, False)
    run_once(a, "early scoring, cache", True, 8, False)
    run_once(a, "scoring stubbed", True, 8, True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
