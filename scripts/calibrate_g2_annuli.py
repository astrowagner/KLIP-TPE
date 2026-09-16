#!/usr/bin/env python
"""Calibrate G2's injection contrast for a given set of annulus edges.

    TQDM_DISABLE=1 python scripts/calibrate_g2_annuli.py --edges 20 36 66

``_bench_hi`` forces one contrast per annulus so that every slot of the benchmark sees an
identical problem -- if each slot calibrated its own, the paired TPE-vs-random comparison
would be between different problems.  Those forced numbers were run C's, which means they
are only right for run C's zones.  Move an annulus edge and they are no longer a calibration
of anything: the contrast that puts the default configuration at median S/N 5 depends on the
radii the sources are injected at.

So this runs the real thing rather than interpolating.  It builds the G2 reducer, space,
guard, objective and sampler exactly as ``_bench_hi`` does, then calls ``Runner.calibrate``
-- the same code path the science runs use -- with ``forced`` cleared, and prints the
contrast each annulus settles on.  Paste those into ``run_G2``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
for _p in (os.path.join(_HERE, "paper_runs"), os.path.join(os.path.dirname(_HERE), "paper_runs")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from klip_tpe import CalibrationConfig, RunConfig, Runner, ValidationConfig, datasets
from klip_tpe.instruments import generic


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=float, nargs="+", default=[20, 36, 66])
    ap.add_argument("--seed", type=int, default=0, help="the seed _bench_hi's first slot uses")
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--n-sources", type=int, default=3)
    ap.add_argument("--n-min-ref", type=int, default=10)
    ap.add_argument("--keep", default=None, help="keep the scratch run directory here")
    ap.add_argument("--forced", type=float, nargs="+", default=None,
                    help="measure the median S/N AT these contrasts instead of solving for one "
                         "(one per annulus) -- the control: [20 45 75] with run C's numbers has "
                         "to reproduce run C's S/N, or the harness is wrong, not the zone")
    a = ap.parse_args(argv)

    def log(m):
        print(m, flush=True)

    from run_demos import HD, hd95086_objects
    red = hd95086_objects()
    px = datasets.INSTRUMENT["sphere_hd95086"]["pxscale"]
    p = HD[0] / px
    nann = len(a.edges) - 1
    log(f"\nedges {a.edges}  ->  {nann} annuli; HD 95086 b at {p:.2f} px")
    for i in range(nann):
        lo, hi = a.edges[i], a.edges[i + 1]
        log(f"   annulus {i+1}: [{lo:g}, {hi:g}] px = [{lo*px:.3f}, {hi*px:.3f}] arcsec"
            + ("   <- contains HD 95086 b" if lo <= p <= hi else ""))

    space = generic.make_space(red, k_klip_max=a.k_max, search_angles=True)
    space.project = generic.make_guard(red, k_max=a.k_max, n_min_ref=a.n_min_ref)
    obj, samp = generic.default_config(red, known=[HD])
    cfg = RunConfig(ann_edges=list(a.edges), n_iter=1, n_init=1, seed=a.seed, search_mode="tpe",
                    n_sources=a.n_sources, validation=ValidationConfig(n_top=1, n_valid=1),
                    # forced cleared -> the loop actually runs; everything else is _bench_hi's
                    calibration=CalibrationConfig(
                        forced=(list(a.forced) if a.forced else [0.0] * nann)),
                    defaults={"k_klip": 10}, fm_curve=False, save_fits=False,
                    save_eval_images=False, write_setup_files=False)
    d = a.keep or tempfile.mkdtemp(prefix="g2cal_")
    try:
        r = Runner(red, space, obj, samp, cfg, d, log=log)
        out = []
        for ia in range(nann):
            log(f"\n=== annulus {ia+1} ===")
            # Runner._reduce takes the zone from self.ia, which _search_annulus normally sets.
            # Calling calibrate() straight out of the loop leaves it at 0, so annulus 2 gets
            # reduced in annulus 1's zone and every source lands in the NaN outside it --
            # a silent all-NaN score that looks exactly like a zone that cannot be calibrated.
            r.ia = ia
            contrast, kdef, info = r.calibrate(ia)
            snr = info.get("snr")
            out.append((ia, contrast, kdef, snr, len(info.get("trials", []))))
        log("\n" + "=" * 62)
        log(f"{'annulus':<10}{'contrast':>14}{'median S/N':>13}{'k_default':>12}{'trials':>9}")
        for ia, c, kdef, snr, nt in out:
            log(f"{ia+1:<10}{c:>14.4e}{(snr if snr is not None else float('nan')):>13.2f}"
                f"{kdef:>12}{nt:>9}")
        log("\npaste into run_G2:")
        log("    (" + ", ".join(f"{c:.4g}" for _, c, _, _, _ in out) + "), "
            + str([int(e) for e in a.edges]) + ",")
    finally:
        if a.keep is None:
            shutil.rmtree(d, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
