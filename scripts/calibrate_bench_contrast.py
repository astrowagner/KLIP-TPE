#!/usr/bin/env python
"""Measure the forced injection contrast a benchmark should use, on the CURRENT flux axis.

    TQDM_DISABLE=1 python scripts/calibrate_bench_contrast.py E2
    TQDM_DISABLE=1 python scripts/calibrate_bench_contrast.py F2
    TQDM_DISABLE=1 python scripts/calibrate_bench_contrast.py G2 [--edges 20 36 66]
    TQDM_DISABLE=1 python scripts/calibrate_bench_contrast.py G2 --forced 7.946e-6 6.201e-6 --edges 20 45 75   # control

``_bench_hi`` forces one contrast per annulus so that every slot of a benchmark sees an
identical problem.  Each of those constants was once a science run's calibration -- and a
calibration is a number on a flux axis.  When the axis moved (the beta Pic star flux went
from the template's own sum, 4.3491, to VIP's published 3.3268e6 in ff20c4b; HD 95086's by
1347.0246), every forced constant that was not re-measured started injecting something
else entirely.  E2's ``1.31e-3`` and F2's ``2.087e-3`` were "the contrast run A/B
calibrated" on the OLD axis; on the current one they inject a 105-168 count peak, a source
two to three times brighter than beta Pic b, and the benchmark stops being a benchmark.

So this measures the number instead of converting it.  It builds the bench's reducer, space,
guard, objective and sampler exactly as ``_bench_hi`` does and calls ``Runner.calibrate`` --
the same code path the science runs use -- with ``forced`` cleared.  ``--forced`` measures the
median S/N AT given contrasts instead, which is the control: run C's own zones with run C's
own numbers have to reproduce run C's S/N, or the harness is wrong rather than the constant.

Add a bench here when you add one to run_demos.py; the test in tests/test_benchmark_zones.py
pins the constants that come out.
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

# the benches of run_demos.py: (groups, k_max, max_drop, defaults, default edges, n_sources,
# n_min_ref, search_angles) -- keep in step with the _bench_hi calls
BENCH = {
    "E2": dict(groups=1, k_max=30, max_drop=0,    defaults={"k_klip": 10}, edges=[8, 22],
               n_sources=3, n_min_ref=5,  search_angles=True,  target="betapic"),
    "F2": dict(groups=4, k_max=12, max_drop=2,    defaults={"k_klip": 5},  edges=[8, 22],
               n_sources=3, n_min_ref=5,  search_angles=True,  target="betapic"),
    "G2": dict(groups=1, k_max=30, max_drop=None, defaults={"k_klip": 10}, edges=[20, 36, 66],
               n_sources=3, n_min_ref=10, search_angles=True,  target="hd95086"),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=sorted(BENCH))
    ap.add_argument("--edges", type=float, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0, help="_bench_hi's first slot uses seed 0")
    ap.add_argument("--forced", type=float, nargs="+", default=None,
                    help="measure median S/N at these contrasts (one per annulus) instead of solving")
    ap.add_argument("--keep", default=None, help="keep the scratch run directory here")
    a = ap.parse_args(argv)
    B = BENCH[a.bench]
    edges = list(a.edges) if a.edges else list(B["edges"])
    nann = len(edges) - 1

    def log(m):
        print(m, flush=True)

    import run_demos as R
    if B["target"] == "betapic":
        dsets, sf, inst = R.betapic_dataset(B["groups"])
        red = generic.make_reducer(dsets, star_flux=sf, max_workers=1, log=lambda s: None,
                                   partition_label="group" if B["groups"] > 1 else "dataset", **inst)
        known = [R.BP]
        fpa, pmask = R.bp_disk(red)
        px = inst["pxscale"]
    else:
        red = R.hd95086_objects()
        known = [R.HD]
        fpa, pmask = (), None
        px = datasets.INSTRUMENT["sphere_hd95086"]["pxscale"]
    r0 = next(iter(red.reducers.values()))
    log(f"\n{a.bench}: {B['target']} groups={B['groups']}  star_flux/flux_unit = "
        f"{getattr(r0.model, 'flux_unit', float('nan')):.4e}  edges {edges}")
    comp = known[0][0] / px
    for i in range(nann):
        lo, hi = edges[i], edges[i + 1]
        log(f"   annulus {i+1}: [{lo:g}, {hi:g}] px = [{lo*px:.3f}, {hi*px:.3f}]\""
            + (f"   <- contains the companion ({comp:.1f} px)" if lo <= comp <= hi else ""))

    kw = {} if B["max_drop"] is None else {"max_drop": B["max_drop"]}
    space = generic.make_space(red, k_klip_max=B["k_max"], search_angles=B["search_angles"], **kw)
    space.project = generic.make_guard(red, k_max=B["k_max"], n_min_ref=B["n_min_ref"])
    obj, samp = generic.default_config(red, known=known, forbidden_pa=fpa, pixel_mask=pmask)
    cfg = RunConfig(ann_edges=edges, n_iter=1, n_init=1, seed=a.seed, search_mode="tpe",
                    n_sources=B["n_sources"], validation=ValidationConfig(n_top=1, n_valid=1),
                    calibration=CalibrationConfig(forced=(list(a.forced) if a.forced else [0.0] * nann)),
                    defaults=B["defaults"], fm_curve=False, save_fits=False,
                    save_eval_images=False, write_setup_files=False)
    d = a.keep or tempfile.mkdtemp(prefix=f"{a.bench.lower()}cal_")
    try:
        r = Runner(red, space, obj, samp, cfg, d, log=log)
        out = []
        for ia in range(nann):
            log(f"\n=== annulus {ia+1} ===")
            r.ia = ia            # Runner._reduce takes the zone from self.ia (set by _search_annulus in a run)
            contrast, kdef, info = r.calibrate(ia)
            out.append((ia, contrast, kdef, info.get("snr"), len(info.get("trials", []))))
        log("\n" + "=" * 62)
        log(f"{'annulus':<10}{'contrast':>14}{'median S/N':>13}{'k_default':>12}{'trials':>9}")
        for ia, c, kdef, snr, nt in out:
            log(f"{ia+1:<10}{c:>14.4e}{(snr if snr is not None else float('nan')):>13.2f}{kdef:>12}{nt:>9}")
        log(f"\npaste into run_{a.bench} in run_demos.py:")
        vals = ", ".join(f"{c:.4g}" for _, c, _, _, _ in out)
        log(f"    {'(' + vals + ')' if nann > 1 else vals}, {[int(e) for e in edges]},")
    finally:
        if a.keep is None:
            shutil.rmtree(d, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
