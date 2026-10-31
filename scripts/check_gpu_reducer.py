#!/usr/bin/env python
"""check_gpu_reducer.py -- GPUKLIPReducer against KLIPReducer on a finished NEAR run's frames and configurations.

For each of N configurations of one annulus (the validation's best, as klip-tpe ranked them; the
last N records when the annulus has no validation yet) and every night they select, the clean and
one injected reduction are made by klip-tpe's own KLIPReducer.reduce (--threads) and by
GPUKLIPReducer.reduce on the card, through the same ReductionRequest.  Per night: the images'
agreement (max |difference| in units of the annulus noise, and the NaN footprint pixel for pixel),
whether the same job twice on the card gives the same bits, and the two times.  Per configuration,
when every night it selects is loaded: the combined images' agreement and both scores through
klip-tpe's objective, beside the run's recorded raw score.  Then, on the heaviest night loaded,
the sectors batched against one by one and the compact size classes against the padded stack.

    python scripts/check_gpu_reducer.py --run run_20261028_202846 --annulus 2 --n 3 \\
        [--nights 1 2 3 4 5 6] [--root /mnt/RAID64TB/NEAR2/data] [--profile] [--compact 16]

The run is found as a path, or by name under <root>/comb/opt/ and its completed/.  The frames are
loaded as the run loaded them (near.make_reducer with the run's data edits: cube_name, dropped
frames, subtracted and injected sources) and checked against the run's record, as a resume does.
Reads the run and the data only; writes nothing but its log (tee it).  Exit status 0 on PASS
(every job within 1e-3 of the noise, every card job bit-identical with itself).

Written in the NEAR2 GPU session, 2026-10-31 (bench/gpu_reducer/ on patrick).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isfile(os.path.join(HERE, "gpu_reducer.py")):       # the copy beside this file (bench/gpu_reducer/)
    sys.path.insert(0, HERE)
    from gpu_reducer import GPUKLIPReducer  # noqa: E402
else:                                                           # the package (scripts/ in the repo)
    _root = os.path.dirname(HERE)
    if os.path.isdir(os.path.join(_root, "klip_tpe")) and _root not in sys.path:
        sys.path.insert(0, _root)                               # a checkout that is not installed
    from klip_tpe.gpu_reducer import GPUKLIPReducer  # noqa: E402


# ----------------------------------------------------------------------------- the run
def find_run(spec: str, root: str) -> str:
    opt = os.path.join(root, "comb", "opt")
    for c in (spec, os.path.join(opt, os.path.basename(spec)), os.path.join(opt, "completed", os.path.basename(spec))):
        if os.path.isfile(os.path.join(c, "run_setup.json")):
            return os.path.abspath(c)
    raise SystemExit(f"no run_setup.json for {spec} (looked in {spec}, {opt} and its completed/)")


def read_setup(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "run_setup.json")) as f:
        return json.load(f)


def load_records(run_dir: str, ia: int) -> List[Dict[str, Any]]:
    """The scored records of annulus ``ia`` (0-based) from results.jsonl, in file order, at the
    annulus' final contrast only (a re-calibration restarts an annulus with a new contrast, and
    klip-tpe's validation ranks the records of the final segment)."""
    recs: List[Dict[str, Any]] = []
    contrast = None
    with open(os.path.join(run_dir, "results.jsonl")) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("annulus") != ia:
                continue
            if contrast is not None and r.get("contrast") != contrast:
                recs = []
            contrast = r.get("contrast")
            if not (r.get("meta") or {}).get("failed"):
                recs.append(r)
    return recs


def validated_records(run_dir: str, ia: int, recs: Sequence[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """The annulus' ``n`` best configurations by klip-tpe's own validation
    (annulusNN/validation.json, ``validated_score``), as their search records; [] without one."""
    path = os.path.join(run_dir, f"annulus{ia + 1:02d}", "validation.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = [r for r in json.load(f) if r.get("validated_score") is not None]
    rows.sort(key=lambda r: -float(r["validated_score"]))
    by_index = {int(r["index"]): r for r in recs}
    out = []
    for row in rows[:n]:
        e = int(row["eval_index"])
        if e in by_index:
            out.append(dict(by_index[e], validated_score=float(row["validated_score"])))
    return out


def pad_edges(cfg: Dict[str, Any], a_in: float, a_out: float) -> Tuple[bool, bool]:
    """Runner._pad_edges on the run's RunConfig dict: whether the zone's inner and outer edges
    are padded (``edge_padding`` "all", or "between" for an edge another annulus shares)."""
    if str(cfg.get("edge_padding", "all")).lower() == "all":
        return True, True
    e = [float(v) for v in (cfg.get("ann_edges") or [])]
    if cfg.get("opt_width"):
        lo = e[0] if e else 0.0
        w_lo = float(cfg["width_range"][0]) if cfg.get("width_range") else 0.0
        r_cap = float(cfg.get("r_cap", 70.0))
        outer_end = a_out >= r_cap - 0.5 or r_cap - a_out < w_lo
    else:
        lo, hi = (e[0], e[-1]) if len(e) >= 2 else (0.0, e[0] if e else 70.0)
        outer_end = a_out >= hi - 1e-6
    return a_in > lo + 1e-6, not outer_end


def build_reducer(root: str, nights: Sequence[int], cfg: Dict[str, Any], threads: int, log=print):
    """klip-tpe's NEAR reducer on the frames the run was made on: near.make_reducer with the
    run's data edits, then the check a resume makes (runner._core_edits) that the reducer's frames
    are the run's."""
    from klip_tpe.instruments import near
    from klip_tpe.metrics import Source
    from klip_tpe.runner import _core_edits, _describe_edits
    edits = dict(cfg.get("data_edits") or {})
    known = {"cube_name", "dropped_frames", "subtracted_sources", "injected_sources", "subtract_file", "subtract_line",
             "inject_file"}
    unknown = sorted(k for k in edits if k not in known)
    if unknown:
        raise SystemExit(f"the run's data_edits carry {unknown}, which this script does not know: refusing to reduce "
                         f"frames that may not be the run's")
    drop = {str(k): [int(i) for i in v] for k, v in (edits.get("dropped_frames") or {}).items() if v}
    missing = [k for k in drop if int(k[1:]) not in [int(s) for s in nights]]
    if missing:
        log(f"  note: the run dropped frames of {' '.join(missing)}, which --nights leaves out")
        drop = {k: v for k, v in drop.items() if k not in missing}
    kw: Dict[str, Any] = {}
    if edits.get("cube_name"):
        kw["cube_name"] = str(edits["cube_name"])
    if drop:
        kw["drop_frames"] = drop
    if edits.get("subtracted_sources"):
        kw["subtract_sources"] = [Source(*[float(v) for v in s[:3]]) for s in edits["subtracted_sources"]]
    if edits.get("injected_sources"):
        kw["inject_sources"] = [Source(*[float(v) for v in s[:3]]) for s in edits["injected_sources"]]
    pr = near.make_reducer(root, [int(s) for s in nights], max_workers=1, pool="threads", log=log, **kw)
    want = _core_edits(edits)
    if missing:                                     # the nights not loaded cannot carry their drops
        want["dropped_frames"] = {k: v for k, v in want.get("dropped_frames", {}).items() if k not in missing}
        if not want["dropped_frames"]:
            del want["dropped_frames"]
    have = _core_edits(getattr(pr, "data_edits", None))
    if want != have:
        raise SystemExit(f"the reducer's frames are not the run's: the run recorded {_describe_edits(want)}, the "
                         f"reducer holds {_describe_edits(have)}")
    log(f"  frames: {_describe_edits(have)}")
    return pr


def objective_for(pr, setup: Dict[str, Any], log=print):
    """klip-tpe's objective as the run had it (near.default_config, the recorded inner guard); a
    run with a pixel mask or known sources is refused (the scores would not be the run's)."""
    from klip_tpe.instruments import near
    objective, sampler = near.default_config(pr)
    rec_obj = (setup.get("objective") or {}).get("metric") or {}
    rec_smp = setup.get("sampler") or {}
    if rec_obj.get("inner_fwhm") is not None:
        objective.metric.inner_fwhm = float(rec_obj["inner_fwhm"])
    if rec_obj.get("pixel_mask_px") or rec_smp.get("known"):
        raise SystemExit("the run's metric has a pixel mask or known sources, which this script does not reproduce")
    try:
        mine = objective.metric.describe() if hasattr(objective.metric, "describe") else {}
    except Exception:  # noqa: BLE001
        mine = {}
    diff = {k: (rec_obj.get(k), mine.get(k)) for k in ("name", "excl_fwhm", "min_ring", "inner_fwhm", "search_px",
                                                       "band_fwhm", "declip_nsig", "penalty", "measured_kernel")
            if k in rec_obj and k in mine and rec_obj.get(k) != mine.get(k)}
    if diff:
        log(f"  WARNING: the metric differs from the run's record: {diff}")
    if (setup.get("reducer") or {}).get("weighting", "equal") != "equal":
        raise SystemExit("the run combined its nights with weights other than equal")
    return objective


def score(objective, inj: np.ndarray, srcs, clean) -> float:
    r = objective.score_raw(np.asarray(inj, float), srcs, None if clean is None else np.asarray(clean, float))
    return float(r.score) if np.isfinite(r.score) else float("nan")


# ----------------------------------------------------------------------------- the comparison
def annulus_stats(ref: np.ndarray, test: np.ndarray, a_in: float, a_out: float):
    ny, nx = ref.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - nx / 2.0, yy - ny / 2.0)
    m = (r >= a_in) & (r <= a_out)
    fr, ft = np.isfinite(ref), np.isfinite(test)
    sig = float(np.std(ref[m & fr]))
    both = m & fr & ft
    d = np.abs(test - ref)[both]
    return (d.max() / sig if d.size else float("nan")), (np.sqrt(np.mean(d ** 2)) / sig if d.size else float("nan")), \
        bool(np.array_equal(fr, ft)), sig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="a run directory, or a run name under <root>/comb/opt/")
    ap.add_argument("--root", default="/mnt/RAID64TB/NEAR2/data")
    ap.add_argument("--annulus", type=int, default=1, help="1-based")
    ap.add_argument("--n", type=int, default=3, help="configurations: the validation's best n")
    ap.add_argument("--nights", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--compact", type=int, default=0, help="the size-class step (0: the padded stack, the default)")
    ap.add_argument("--eigh", default="xbatched64")
    ap.add_argument("--rr", action="store_true")
    ap.add_argument("--threads", type=int, default=8, help="the CPU reduction's threads")
    ap.add_argument("--backend", default="cupy", help="cupy, or numpy to run the same chain on the host")
    ap.add_argument("--profile", action="store_true", help="the card's time per step of each job")
    a = ap.parse_args()
    from klip_tpe.metrics import Source
    from klip_tpe.reducer import ReductionRequest, combine_stack
    from klip_tpe.version import describe as kt_describe
    run_dir = find_run(a.run, a.root)
    setup = read_setup(run_dir)
    cfg = setup.get("config") if isinstance(setup.get("config"), dict) else setup
    print(f"== run {os.path.basename(run_dir)}, klip-tpe {kt_describe()}, GPUKLIPReducer from "
          f"{sys.modules[GPUKLIPReducer.__module__].__file__}")
    pr = build_reducer(a.root, a.nights, cfg, a.threads, log=lambda s: print("  " + s))
    objective = objective_for(pr, setup, log=print)
    ia = a.annulus - 1
    recs = load_records(run_dir, ia)
    if not recs:
        raise SystemExit(f"no scored records of annulus {a.annulus} in {run_dir}/results.jsonl")
    picks = validated_records(run_dir, ia, recs, a.n) or recs[-a.n:]
    gpus = {pid: GPUKLIPReducer.from_cpu(r, eigh=a.eigh, compact=a.compact, rr=a.rr) for pid, r in pr.reducers.items()}
    for g in gpus.values():
        g.backend_name = a.backend
        g.profile = bool(a.profile)
    first = next(iter(gpus.values()))
    print(f"== warm(): {first.warm()}")
    worst, worst_comb, nonbit = 0.0, 0.0, 0
    t_cpu, t_gpu, n_jobs = 0.0, 0.0, 0
    for rec in picks:
        c = rec["config"]
        zone = rec["meta"]["zone"]
        a_in, a_out = float(zone[0]), float(zone[1])
        pad_in, pad_out = pad_edges(cfg, a_in, a_out)
        sel = [str(s) for s in c["selected"]]
        loaded = [s for s in sel if s in pr.reducers]
        srcs = [Source(float(s[0]), float(s[1]), float(s[2])) for s in rec["sources"]]
        print(f"\n== eval {rec['index']} (annulus {a.annulus}): bin {c['params'].get('bin')} n_ang "
              f"{[c['per_partition'][p].get('n_ang') for p in sel]} nights {sel}, recorded raw score {rec.get('raw_score')}"
              + (f", validated {rec['validated_score']:.4f}" if "validated_score" in rec else ""))
        if len(loaded) < len(sel):
            print(f"   nights {[s for s in sel if s not in loaded]} not loaded (--nights): per-night checks on {loaded}, "
                  f"no combined score")
        stacks: Dict[str, List[Tuple[str, np.ndarray]]] = {"cpu": [], "gpu": []}
        for pid in loaded:
            params = dict(c["per_partition"][pid], inrad=a_in, outrad=a_out, pad_in=pad_in, pad_out=pad_out)
            red, g = pr.reducers[pid], gpus[pid]
            for what, s in (("clean", None), ("inj", srcs)):
                req = ReductionRequest(params, s, False, f"chk_{pid}_{what}", threads=a.threads)
                if not g.supports(req):
                    print(f"   {pid} {what}: the GPU reducer does not support this request; skipped")
                    continue
                red._inj_cache = {}
                t0 = time.perf_counter(); rc = red.reduce(req); tc = time.perf_counter() - t0
                g.reduce(req)                               # the night's cube goes to the card once
                t0 = time.perf_counter(); rg = g.reduce(req); tg = time.perf_counter() - t0
                rg2 = g.reduce(req)
                dmax, drms, same_nan, _ = annulus_stats(rc.image, rg.image, a_in, a_out)
                bit = bool(np.array_equal(rg.image, rg2.image, equal_nan=True))
                nonbit += int(not bit)
                worst = max(worst, dmax)
                t_cpu += tc; t_gpu += tg; n_jobs += 1
                print(f"   {pid} {what:5s}: max|d|/sigma {dmax:.1e} rms {drms:.1e}  nan same {same_nan}  twice-identical {bit}  "
                      f"cpu {tc:.3f} s  card {tg:.3f} s  (n_binned {rc.meta.get('n_binned')}/{rg.meta.get('n_binned')}, "
                      f"k_eff {rc.meta.get('k_effective')}/{rg.meta.get('k_effective')}, "
                      f"dropped {rc.meta.get('n_dropped')}/{rg.meta.get('n_dropped')})")
                if a.profile and g.last_steps:
                    print("        card steps: " + ", ".join(f"{k} {v * 1000:.1f}" for k, v in g.last_steps.items()) + " ms")
                stacks["cpu"].append((what, rc.image)); stacks["gpu"].append((what, rg.image))
        if len(loaded) < len(sel):
            continue
        w = np.array([pr.partition_weight(pid) for pid in sel], float)
        comb = {}
        for dev in ("cpu", "gpu"):
            for what in ("clean", "inj"):
                ims = [im for wh, im in stacks[dev] if wh == what]
                if len(ims) == len(sel):
                    comb[(dev, what)] = combine_stack(np.stack(ims), w)
        if ("cpu", "inj") in comb and ("gpu", "inj") in comb:
            dmax, drms, same_nan, _ = annulus_stats(comb[("cpu", "inj")], comb[("gpu", "inj")], a_in, a_out)
            worst_comb = max(worst_comb, dmax)
            sc = score(objective, comb[("cpu", "inj")], srcs, comb.get(("cpu", "clean")))
            sg = score(objective, comb[("gpu", "inj")], srcs, comb.get(("gpu", "clean")))
            print(f"   combined: max|d|/sigma {dmax:.1e} rms {drms:.1e}; score cpu {sc:.4f} card {sg:.4f} (diff {sg - sc:+.1e}); "
                  f"recorded raw {rec.get('raw_score')}")
    print(f"\n== {n_jobs} jobs: worst per-night max|d|/sigma {worst:.1e}, worst combined {worst_comb:.1e}, "
          f"not bit-identical on the card twice: {nonbit}; CPU {t_cpu:.1f} s ({a.threads} threads), card {t_gpu:.1f} s "
          f"(card / CPU {t_gpu / max(t_cpu, 1e-9):.2f})")
    ok = n_jobs > 0 and worst < 1e-3 and nonbit == 0
    # sectors batched against one by one, and compact against padded, on the last configuration's heaviest night loaded
    if picks:
        rec = picks[-1]; c = rec["config"]; zone = rec["meta"]["zone"]
        a_in, a_out = float(zone[0]), float(zone[1]); pad_in, pad_out = pad_edges(cfg, a_in, a_out)
        loaded = [s for s in [str(x) for x in c["selected"]] if s in pr.reducers]
        if loaded:
            pid = max(loaded, key=lambda p: pr.reducers[p].data.nframes)
            params = dict(c["per_partition"][pid], inrad=a_in, outrad=a_out, pad_in=pad_in, pad_out=pad_out)
            srcs = [Source(float(s[0]), float(s[1]), float(s[2])) for s in rec["sources"]]
            req = ReductionRequest(params, srcs, False, "cmp", threads=a.threads)
            g = gpus[pid]
            out = {}
            for name, cmp_, bs in (("padded, sectors batched", 0, True), ("padded, sectors one by one", 0, False),
                                   ("compact 16", 16, True), ("compact 32", 32, True)):
                g.compact, g.batch_sectors = cmp_, bs
                g.reduce(req)
                t0 = time.perf_counter(); r = g.reduce(req); t1 = time.perf_counter() - t0
                t0 = time.perf_counter(); r = g.reduce(req); out[name] = (r.image, min(t1, time.perf_counter() - t0))
            ref = out["padded, sectors batched"][0]
            print(f"== {pid}, eval {rec['index']} (bin {params.get('bin')}, {params.get('n_ang')} sectors): "
                  + "; ".join(f"{name} {t:.3f} s" + ("" if name.startswith("padded, sectors b")
                                                       else f" (max|d|/sigma {annulus_stats(ref, im, a_in, a_out)[0]:.1e})")
                              for name, (im, t) in out.items()))
            g.compact, g.batch_sectors = a.compact, True
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
