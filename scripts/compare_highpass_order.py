"""Does high-passing after binning give the evaluations high-passing every frame gives?

``KLIPReducer`` high-passes every kept frame and then bins them, the IDL's order.  Both steps
are linear, so binning first and high-passing the binned frames gives the same images to
float32 rounding, at 1/bin of the high-pass cost (``hp_order="binned"``, off by default).
This script measures the difference on evaluations a run has made.  For each one it reduces
the injected and the clean image both ways, and reports the images' largest and rms
difference against their scatter, the search score and the per-source S/N both ways, and the
time of one partition job both ways.  The yardstick for a score difference is the spread
between the run's own draws of that trial (``draw_sd``).

usage::

    python scripts/compare_highpass_order.py --run-dir <run> [--n 40] [--workers 96] [--time-jobs 3]
    python scripts/compare_highpass_order.py --synthetic [--n 12] [--pedestal 1e4]

Without ``--root`` the data arguments come from the run's record, as ``klip-tpe resume`` takes
them; the ones given here win.  The run directory is only read.  The report goes to stdout and
to ``--out`` (default ``hp_order_<run>.json`` beside the run directory).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from klip_tpe import cli                                    # noqa: E402
from klip_tpe.metrics import Source                         # noqa: E402
from klip_tpe.space import Config                           # noqa: E402

DATA_KEYS = ("root", "nights", "instrument", "obj", "name", "binned", "frames", "crop_half", "parang_sign",
             "pre_bin", "cube", "angles", "psf", "pxscale", "lam", "diam", "ref_cube", "names", "star_flux",
             "known", "no_library", "fast", "n_min_ref", "truenorth", "fwhm_px", "metric", "backend",
             "max_frames", "weighting")


def _args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    cli._add_near_args(p)
    p.add_argument("--run-dir", default=None, help="the run whose evaluations to re-reduce")
    p.add_argument("--n", type=int, default=40, help="evaluations: the best half and a random half (default 40)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-jobs", type=int, default=3,
                   help="evaluations whose partition jobs are also timed one by one, both ways (default 3)")
    p.add_argument("--out", default=None, help="JSON report (default hp_order_<run>.json beside the run)")
    p.add_argument("--synthetic", action="store_true", help="two synthetic nights instead of a run")
    p.add_argument("--pedestal", type=float, default=0.0, help="--synthetic: a constant added to every frame")
    a, _ = p.parse_known_args()
    a._given = {k for k, v in vars(a).items() if v != p.get_default(k)}
    return a


def _from_record(a):
    """The data arguments the run recorded when it started (the newest record with a root)."""
    from klip_tpe.registry import records_for
    for rec in records_for(a.run_dir):
        argv = rec.get("argv") or []
        if len(argv) < 2:
            continue
        rest, skip = [], False
        for tok in argv[2:]:
            if skip:
                skip = False
                continue
            if tok == "--run-dir":
                skip = True
                continue
            if not tok.startswith("--run-dir="):
                rest.append(tok)
        try:
            prev, _ = cli._build_parser().parse_known_args([argv[1], "--run-dir", a.run_dir] + rest)
        except SystemExit:
            continue
        if getattr(prev, "root", None) or getattr(prev, "cube", None):
            for k in DATA_KEYS + ("workers", "pool"):
                if k not in a._given and hasattr(prev, k):
                    setattr(a, k, getattr(prev, k))
            return True
    return False


def _run_objects(a):
    from klip_tpe.runner import RunConfig, Runner, load_checkpoint
    from klip_tpe.space import SearchSpace
    if not getattr(a, "root", None) and not getattr(a, "cube", None) and not _from_record(a):
        raise SystemExit("no data arguments: give them (as for klip-tpe near) or run on the run's own machine")
    red, objective, sampler, guard, _ = cli._near_objects(a, log=lambda s: None)
    st, _ = load_checkpoint(a.run_dir)
    if st is None:
        raise SystemExit(f"no checkpoint in {a.run_dir}")
    cfg = RunConfig.from_dict(st["config"])
    if cfg.opt_width:
        raise SystemExit("opt_width runs are not supported here")
    space = SearchSpace.from_dict(st["space"], project=guard)
    r = Runner(red, space, objective, sampler, cfg, tempfile.mkdtemp(prefix="hp_order_"), log=lambda s: None,
               resume="never")
    recs = []
    with open(os.path.join(a.run_dir, "results.jsonl")) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("score") is not None and not (rec.get("meta") or {}).get("failed"):
                recs.append(rec)
    seen, uniq = set(), []
    for rec in recs:
        key = (rec["annulus"], tuple(np.round(rec["x"], 9)), tuple(map(tuple, rec["sources"])))
        if key not in seen:
            seen.add(key)
            uniq.append(rec)
    rng = np.random.default_rng(a.seed)
    order = sorted(range(len(uniq)), key=lambda i: -uniq[i]["score"])
    nb = min(a.n // 2, len(order))
    best, rest = order[:nb], order[nb:]
    pick = best + (list(rng.choice(rest, size=min(a.n - nb, len(rest)), replace=False)) if rest else [])
    picks = []
    for i in pick:
        rec = uniq[int(i)]
        picks.append({"annulus": int(rec["annulus"]), "index": int(rec["index"]), "x": np.asarray(rec["x"], float),
                      "sources": [Source(*s) for s in rec["sources"]], "contrast": float(rec["contrast"]),
                      "k": rec.get("k_used"), "draw_sd": (rec.get("meta") or {}).get("draw_sd"),
                      "logged": float(rec["score"])})
    return r, red, picks, os.path.basename(os.path.abspath(a.run_dir))


def _synthetic_objects(a):
    from klip_tpe import (GaussianPSF, KLIPReducer, MawetPeakSNR, Objective, PartitionedReducer, PositionSampler,
                          RunConfig, Runner)
    from klip_tpe.space import Param, SearchSpace, kgrid
    from klip_tpe.synthetic import synthetic_klip_dataset
    px = 0.05
    reds = {}
    for i, (pid, nf) in enumerate((("n1", 400), ("n2", 900))):
        ds = synthetic_klip_dataset(nframes=nf, size=100, pa_span=80, seed=i + 1)
        ds.cube += np.float32(a.pedestal)
        reds[pid] = KLIPReducer(ds, px, 3.89 * px / 206265 * 8.2, 8.2, outrad_cap=44,
                                injection_model=GaussianPSF(4.0, star_flux=1e4))
    red = PartitionedReducer(reds, max_workers=int(a.workers) if str(a.workers).isdigit() else 1)
    block = [Param("bin", 2, 30, "int", default=12), Param("k_klip", 2, 20, "int", grid=kgrid(20), default=8),
             Param("filter", 5, 15, "int", default=9)]
    space = SearchSpace([Param("angsep", 0.0, 2.0, default=1.0), Param("anglemax", 15.0, 60.0, default=30.0)]
                        ).replicate(block, list(reds))
    obj = Objective(MawetPeakSNR(pxscale=px, fwhm=4.0), clean_subtract=True)
    samp = PositionSampler(fwhm_as=4.0 * px)
    cfg = RunConfig(ann_edges=[8, 26, 40], n_iter=10, seed=1)
    r = Runner(red, space, obj, samp, cfg, tempfile.mkdtemp(prefix="hp_order_"), log=lambda s: None, resume="never")
    rng = np.random.default_rng(a.seed)
    picks = []
    for j in range(a.n):
        ia = j % 2
        r.ia = ia
        x = np.asarray(space.random(rng), float)
        lo, hi = r._band(ia, space.decode(x))
        picks.append({"annulus": ia, "index": j, "x": x, "sources": samp.sample(4, lo, hi, rng, 2e-3),
                      "contrast": 2e-3, "k": None, "draw_sd": None, "logged": None})
    return r, red, picks, "synthetic"


def _with_order(cfg: Config, order: str) -> Config:
    per = {pid: dict(d, hp_order=order) for pid, d in cfg.per_partition.items()}
    return Config(dict(cfg.params, hp_order=order), per, cfg.selected, cfg.x)


def _diff(a, b):
    """Largest and rms difference over the scatter of ``a``, and whether the NaNs agree."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    fin = np.isfinite(a) & np.isfinite(b)
    scale = float(np.nanstd(a[fin])) if fin.any() else 0.0
    d = (b - a)[fin]
    if not d.size or scale <= 0:
        return 0.0, 0.0, bool((np.isfinite(a) == np.isfinite(b)).all())
    return (float(np.max(np.abs(d)) / scale), float(np.sqrt(np.mean(d * d)) / scale),
            bool((np.isfinite(a) == np.isfinite(b)).all()))


def main():
    a = _args()
    if a.synthetic:
        r, red, picks, name = _synthetic_objects(a)
    elif a.run_dir:
        r, red, picks, name = _run_objects(a)
    else:
        raise SystemExit("give --run-dir or --synthetic")
    out = os.path.abspath(a.out or os.path.join(os.path.dirname(os.path.abspath(a.run_dir or ".")),
                                               f"hp_order_{name}.json"))
    rows, timings = [], []
    print(f"{len(picks)} evaluations of {name}: injected and clean reductions, high-pass before binning "
          f"(raw, the default) and after (binned)")
    print(f"{'ann':>3} {'eval':>5} {'bins':>14} {'score raw':>9} {'binned':>8} {'diff':>8} {'draw sd':>7} "
          f"{'max|d|/sd inj':>13} {'clean':>7} {'nans':>5}")
    for n_done, pk in enumerate(picks):
        r.ia, r.contrast = pk["annulus"], pk["contrast"]
        cfg = r.space.decode(pk["x"])
        if pk["k"] is not None:
            cfg = r._apply_k_flag(cfg, pk["k"])
        res = {}
        captured = []
        for order in ("raw", "binned"):
            c2 = _with_order(cfg, order)
            orig_map = getattr(red, "_map", None)
            if order == "raw" and n_done < a.time_jobs and orig_map is not None:
                def cap(jobs, keys=None, _m=orig_map):
                    captured.extend(jobs)
                    return _m(jobs, keys)
                red._map = cap
            try:
                t0 = time.time()
                inj = r._reduce(c2, pk["sources"], tag=f"hp_{order}_inj")
                clean = r._reduce(c2, None, tag=f"hp_{order}_clean") if r._needs_clean() else None
            finally:
                if orig_map is not None:
                    red._map = orig_map
            sr = r._score_images(inj, clean, pk["sources"])
            res[order] = (inj, clean, sr, time.time() - t0)
        (ia_, ca, sa, _), (ib, cb, sb, _) = res["raw"], res["binned"]
        d_inj = _diff(ia_.image, ib.image)
        d_cln = _diff(ca.image, cb.image) if ca is not None else (0.0, 0.0, True)
        d_parts = [_diff(x, y) for x, y in zip(ia_.stack, ib.stack)] if ia_.stack is not None else []
        ps = [float(v) for v in np.nan_to_num(np.asarray(sb.per_source, float) - np.asarray(sa.per_source, float))]
        bins = [int(round(cfg.params_for(pid).get("bin", 1))) for pid in cfg.selected]
        row = {"annulus": pk["annulus"], "index": pk["index"], "bins": bins, "selected": [str(s) for s in cfg.selected],
               "score_raw": float(sa.score), "score_binned": float(sb.score), "score_diff": float(sb.score - sa.score),
               "logged": pk["logged"], "draw_sd": pk["draw_sd"], "per_source_diff_max": float(np.max(np.abs(ps))),
               "inj_max_rel": d_inj[0], "inj_rms_rel": d_inj[1], "clean_max_rel": d_cln[0], "clean_rms_rel": d_cln[1],
               "partition_max_rel": max([d[0] for d in d_parts], default=0.0),
               "nan_masks_agree": bool(d_inj[2] and d_cln[2] and all(d[2] for d in d_parts))}
        rows.append(row)
        sd = "" if pk["draw_sd"] is None else f"{pk['draw_sd']:.3f}"
        print(f"{pk['annulus'] + 1:>3} {pk['index'] + 1:>5} {','.join(map(str, bins)):>14} {sa.score:>9.4f} "
              f"{sb.score:>8.4f} {sb.score - sa.score:>8.1e} {sd:>7} {d_inj[0]:>13.1e} {d_cln[0]:>7.1e} "
              f"{'same' if row['nan_masks_agree'] else 'DIFF':>5}", flush=True)
        for sub, req in captured:                   # the partition jobs, one by one, on one thread each
            t = {}
            for order in ("raw", "binned"):
                rq = dataclasses.replace(req, params=dict(req.params, hp_order=order), threads=1)
                t0 = time.perf_counter()
                sub.reduce(rq)
                t[order] = time.perf_counter() - t0
            nb = int(round(req.params.get("bin", 1)))
            timings.append({"tag": req.tag, "bin": nb, "raw_s": t["raw"], "binned_s": t["binned"]})
            print(f"      job {req.tag}: bin {nb}, {t['raw']:.2f} s -> {t['binned']:.2f} s on one thread")
    sdiff = np.array([abs(rw["score_diff"]) for rw in rows])
    sds = np.array([rw["draw_sd"] for rw in rows if rw["draw_sd"]], float)
    summary = {
        "evaluations": len(rows),
        "score_abs_diff_max": float(sdiff.max()) if sdiff.size else None,
        "score_abs_diff_median": float(np.median(sdiff)) if sdiff.size else None,
        "draw_sd_median": float(np.median(sds)) if sds.size else None,
        "per_source_abs_diff_max": max([rw["per_source_diff_max"] for rw in rows], default=None),
        "inj_max_rel_max": max([rw["inj_max_rel"] for rw in rows], default=None),
        "inj_rms_rel_median": float(np.median([rw["inj_rms_rel"] for rw in rows])) if rows else None,
        "partition_max_rel_max": max([rw["partition_max_rel"] for rw in rows], default=None),
        "nan_masks_agree": all(rw["nan_masks_agree"] for rw in rows),
        "job_seconds_raw_median": float(np.median([t["raw_s"] for t in timings])) if timings else None,
        "job_seconds_binned_median": float(np.median([t["binned_s"] for t in timings])) if timings else None,
    }
    print("\nsummary")
    for k, v in summary.items():
        print(f"  {k:26s} {v:.3g}" if isinstance(v, float) else f"  {k:26s} {v}")
    with open(out, "w") as f:
        json.dump({"run": name, "summary": summary, "evaluations": rows, "jobs": timings}, f, indent=1)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
