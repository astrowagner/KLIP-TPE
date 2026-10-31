#!/usr/bin/env python
"""KLIP-FM one source at a time, against the test spiral and the injection grid, on a finished run.

    python3 scripts/fm_one_at_a_time.py                                    # the README run on RX J0534
    python3 scripts/fm_one_at_a_time.py --run-dir ~/klip_tpe_runs/rxj0534/p4
    python3 scripts/fm_one_at_a_time.py --target hd95086                   # ~/klip_tpe_runs/hd95086

The run's KLIP-FM cross-check models its test sources (up to 24 on the golden-angle spiral) in
one pass and reads each source's response off that joint model.  The model is linear in the
sources (first order, Pueyo 2016), so the joint model is the sum of the single ones, and a
source's response there differs from its own model's only where another source's core or
self-subtraction lobes reach it.  For the winner of each annulus, at its configuration, this
measures:

1. the spiral, one pass (the run's cross-check);
2. the spiral, one source per pass;
3. the injection grid's settled positions, one source per pass, with the cross-check's noise
   (the matched-filter ring at that radius);
4. the same positions with the grid's own S/N: each source's model added to the clean image
   at the contrast where the grid's estimator gives the grid's target S/N (the ring without
   the source's and the companions' sidelobe arcs, the radial band as a floor, the peak
   search, the small-sample penalty).

1 against 2 is the crowding of the spiral.  3 against 4 is the noise estimate.  4 against the
grid is what is left: the forward model's throughput against the injected sources' own.

The data and the reducer are rebuilt as the run's script built them (``--target``), and the run
directory is only read, apart from the report: ``<run-dir>/fm_one_at_a_time/report.json`` and
``comparison.png``.  It costs a reduction for each test source and each grid position, about
70 on RX J0534's annulus.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_HERE, os.path.join(_HERE, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _arr(v) -> np.ndarray:
    return np.array([np.nan if x is None else x for x in (v or [])], float)


def solve_contrast(measure: Callable[[float], float], c0: float, target: float = 5.0, iters: int = 6) -> float:
    """The contrast at which ``measure(c)``, an S/N close to affine in ``c``, reaches ``target``:
    secant steps from ``c0`` and ``2 c0``.  NaN when the S/N does not respond."""
    c1, c2 = float(c0), 2.0 * float(c0)
    s1, s2 = measure(c1), measure(c2)
    for _ in range(iters):
        if not (np.isfinite(s1) and np.isfinite(s2)) or s2 == s1:
            return float("nan")
        c3 = c2 + (target - s2) * (c2 - c1) / (s2 - s1)
        if not np.isfinite(c3) or c3 <= 0:
            c3 = 0.5 * min(c1, c2)
        s3 = measure(c3)
        c1, s1, c2, s2 = c2, s2, c3, s3
        if np.isfinite(s3) and abs(s3 / target - 1.0) < 1e-3:
            break
    return float(c2) if np.isfinite(s2) else float("nan")


def _num(v) -> Optional[float]:
    """A JSON-safe number: None for NaN, inf or None."""
    return None if v is None or not np.isfinite(v) else float(v)


def _interp_curve(curve: Optional[Dict[str, Any]], r: np.ndarray) -> np.ndarray:
    """A run-style curve (``r_as``, ``curve``) interpolated in log at ``r`` (NaN outside it)."""
    if not curve:
        return np.full(np.size(r), np.nan)
    rr, cc = _arr(curve.get("r_as")), _arr(curve.get("curve"))
    ok = np.isfinite(rr) & np.isfinite(cc) & (cc > 0)
    if ok.sum() < 2:
        return np.full(np.size(r), np.nan)
    o = np.argsort(rr[ok])
    x, y = rr[ok][o], np.log(cc[ok][o])
    out = np.exp(np.interp(r, x, y))
    return np.where((r >= x[0]) & (r <= x[-1]), out, np.nan)


def diagnose(runner, out_dir: str, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """The four measurements of the module docstring for every finished annulus of
    ``runner`` (built with :meth:`klip_tpe.runner.Runner.resume`).  Writes ``report.json``
    and ``comparison.png`` to ``out_dir`` and returns the report."""
    from klip_tpe import contrast_grid as CG
    from klip_tpe.metrics import Source

    os.makedirs(out_dir, exist_ok=True)
    runner._start_workers()
    report: Dict[str, Any] = {"run_dir": os.path.abspath(runner.run_dir), "annuli": []}
    ia_keep = runner.ia
    try:
        for r in [x for x in (runner.results or []) if x is not None]:
            ia = int(r.annulus)
            runner.ia = ia
            cfg = runner._config_from_dict(r.winner_config)
            zone = tuple(float(v) for v in runner._zone(ia, cfg))
            ims = runner._annulus_images(ia) or {}
            clean = ims.get("clean")
            if clean is None:
                clean = runner._reduce(cfg, None, tag=f"fm1_a{ia + 1}_clean").image
            clean = np.asarray(clean, float)
            c0 = float(r.contrast or runner.contrast)
            px = float(runner.pxscale)
            log(f"annulus {ia + 1}: winner eval {r.winner_index + 1}, zone {zone[0]:.1f}-{zone[1]:.1f} px, "
                f"test contrast {c0:.3e}")
            t = time.time()
            sp = runner._fm_pass(cfg, zone, c0, f"fm1_a{ia + 1}_spiral", clean_img=clean)
            if sp is None:
                log("  the reducer gives no KLIP-FM image; nothing to compare")
                continue
            log(f"  spiral, one pass: {len(sp['sources'])} test sources ({time.time() - t:.0f} s)")
            t = time.time()
            one = runner._fm_pass(cfg, zone, c0, f"fm1_a{ia + 1}_one", clean_img=clean, one_at_a_time=True)
            log(f"  spiral, one source per pass ({time.time() - t:.0f} s)")
            k_sp, k_one = _arr(sp["curve"].get("K_fm")), _arr(one["curve"].get("K_fm"))
            ratio = k_one / k_sp
            ok = np.isfinite(ratio)
            ann: Dict[str, Any] = {
                "annulus": ia + 1, "winner_eval": int(r.winner_index) + 1, "zone_px": list(zone), "contrast": c0,
                "spiral": sp["curve"], "one_at_a_time": one["curve"],
                "throughput_one_over_spiral": {"r_as": [float(v) for v in _arr(sp["curve"].get("r_test_as"))],
                                               "ratio": [_num(v) for v in ratio],
                                               "median": float(np.median(ratio[ok])) if ok.any() else None}}
            if ok.any():
                log(f"  each source's own model against the joint one: throughput x{np.median(ratio[ok]):.3f} "
                    f"(median; {ratio[ok].min():.3f}-{ratio[ok].max():.3f})")
            gpath = os.path.join(runner.run_dir, f"annulus{ia + 1:02d}", "contrast_grid.json")
            grid = None
            if os.path.exists(gpath):
                with open(gpath) as f:
                    grid = json.load(f)
            if grid:
                pos = [p for p in grid["positions"] if p.get("status") in CG.STATUS_DONE and p.get("contrast")]
                srcs = [Source(float(p["rho_as"]), float(p["theta"]), c0) for p in pos]
                t = time.time()
                g = runner._fm_pass(cfg, zone, c0, f"fm1_a{ia + 1}_grid", sources=srcs, clean_img=clean,
                                    one_at_a_time=True, keep_singles=True)
                log(f"  the grid's {len(pos)} settled positions, one source per pass ({time.time() - t:.0f} s)")
                geo = runner._grid_geometry(cfg)
                target = float(grid.get("target_snr", 5.0))
                kg = _arr(g["curve"].get("K_fm"))
                sig, rsig = _arr(g["curve"].get("sigma_mf")), _arr(g["curve"].get("r_sigma_px"))
                oks = np.isfinite(sig) & np.isfinite(rsig)
                rho = np.array([float(p["rho_as"]) for p in pos])
                c_fmnoise = np.full(len(pos), np.nan)
                if oks.sum() >= 2:
                    c_fmnoise = np.where(kg > 0, target * np.interp(rho / px, rsig[oks], sig[oks]) / kg, np.nan)
                c_gridsnr = np.full(len(pos), np.nan)
                for j, (p, fmj) in enumerate(zip(pos, g.get("fm_singles") or [])):
                    r_px, th = float(p["rho_as"]) / px, float(p["theta"])

                    def meas(c: float, fmj=fmj, r_px=r_px, th=th) -> float:
                        snr, d = runner._grid_snr(geo, clean + (c / c0) * fmj, r_px, th)
                        if d.get("why") or int(d.get("nclean", 0)) < geo["min_ring"]:
                            return float("nan")
                        return snr
                    c_gridsnr[j] = solve_contrast(meas, float(p["contrast"]), target)
                rows = []
                seps, gcur = _arr(grid.get("r_as")), _arr(grid.get("curve"))
                cg_ = np.array([float(p["contrast"]) for p in pos])
                for k in range(seps.size):
                    sel = np.array([int(p["sep_index"]) == k for p in pos], bool)

                    def med(v, sel=sel):
                        w = v[sel][np.isfinite(v[sel])]
                        return float(np.median(w)) if w.size else None
                    rows.append({"sep_as": float(seps[k]), "n": int(sel.sum()), "grid": _num(gcur[k]),
                                 "grid_positions_median": med(cg_),
                                 "fm_spiral": _num(_interp_curve(sp["curve"], np.array([seps[k]]))[0]),
                                 "fm_one_at_a_time": _num(_interp_curve(one["curve"], np.array([seps[k]]))[0]),
                                 "fm_at_grid_fm_noise": med(c_fmnoise),
                                 "fm_at_grid_grid_snr": med(c_gridsnr)})
                ann["grid"] = {"curve": {k_: grid.get(k_) for k_ in ("r_as", "curve", "p16", "p84", "n")},
                               "positions": [{"rho_as": float(p["rho_as"]), "theta": float(p["theta"]),
                                              "sep_index": int(p["sep_index"]), "grid_contrast": float(p["contrast"]),
                                              "K_fm": _num(a), "fm_noise": _num(b), "grid_snr": _num(c)}
                                             for p, a, b, c in zip(pos, kg, c_fmnoise, c_gridsnr)],
                               "by_separation": rows}
                log("  sep(\")   grid      FM spiral  FM one   FM@grid  FM@grid,grid S/N   (5-sigma contrast)")
                for w in rows:
                    f = [w["grid"], w["fm_spiral"], w["fm_one_at_a_time"], w["fm_at_grid_fm_noise"],
                         w["fm_at_grid_grid_snr"]]
                    log(f"  {w['sep_as']:6.3f}  " + "  ".join("   n/a   " if v is None else f"{v:9.3e}" for v in f))
            report["annuli"].append(ann)
    finally:
        runner.ia = ia_keep
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    try:
        plot(report, os.path.join(out_dir, "comparison.png"))
    except Exception as exc:                                  # noqa: BLE001 - the numbers are written
        log(f"  no plot ({exc!r})")
    log(f"report: {os.path.join(out_dir, 'report.json')}")
    return report


def plot(report: Dict[str, Any], path: str) -> None:
    """One panel per annulus: the grid's median and band, the cross-check's spiral curve one pass
    and one source per pass, and the forward model at the grid's positions with each noise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    anns = report["annuli"]
    fig, axes = plt.subplots(1, max(len(anns), 1), figsize=(7.2 * max(len(anns), 1), 4.6), squeeze=False)
    for ax, ann in zip(axes[0], anns):
        g = (ann.get("grid") or {}).get("curve") or {}
        if g.get("r_as"):
            r, c, lo, hi = _arr(g["r_as"]), _arr(g["curve"]), _arr(g.get("p16")), _arr(g.get("p84"))
            ax.fill_between(r, lo, hi, color="0.85", lw=0, label="injection grid, 16-84 % over PA")
            ax.plot(r, c, "o-", color="0.1", lw=2, ms=5, label="injection grid (median)")
        for key, lab, sty in (("spiral", "KLIP-FM, spiral in one pass (the run's)", dict(color="#2a9d8f", ls="--", lw=1.6)),
                              ("one_at_a_time", "KLIP-FM, spiral one source per pass", dict(color="#2a9d8f", ls="-", lw=1.6))):
            cc = ann.get(key) or {}
            if cc.get("r_as"):
                ax.plot(_arr(cc["r_as"]), _arr(cc["curve"]), label=lab, **sty)
        rows = (ann.get("grid") or {}).get("by_separation") or []
        if rows:
            s = np.array([w["sep_as"] for w in rows])
            for key, lab, sty in (("fm_at_grid_fm_noise", "KLIP-FM at the grid's positions, its own noise",
                                   dict(color="#e76f51", marker="s", ls=":", lw=1.4, ms=5)),
                                  ("fm_at_grid_grid_snr", "KLIP-FM at the grid's positions, the grid's S/N",
                                   dict(color="#264653", marker="D", ls="-.", lw=1.4, ms=5))):
                v = np.array([np.nan if w[key] is None else w[key] for w in rows], float)
                ax.plot(s, v, label=lab, **sty)
        ax.set_yscale("log")
        ax.set_xlabel("separation (\")")
        ax.set_ylabel("5-sigma contrast")
        ax.set_title(f"annulus {ann['annulus']}: winner eval {ann['winner_eval']}", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(os.path.basename(report["run_dir"].rstrip(os.sep)) + ": KLIP-FM one source at a time", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _partitions_of(run_dir: str) -> int:
    with open(os.path.join(run_dir, "checkpoint.json")) as f:
        st = json.load(f)
    return len((st.get("space") or {}).get("partitions") or []) or 4


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="rxj0534", choices=["rxj0534", "hd95086"],
                   help="how to rebuild the run's data and reducer (default rxj0534)")
    p.add_argument("--run-dir", default=None,
                   help="the finished run (default ~/klip_tpe_runs/rxj0534/readme or ~/klip_tpe_runs/hd95086)")
    p.add_argument("--out", default=None, help="report directory (default <run-dir>/fm_one_at_a_time)")
    p.add_argument("--partitions", type=int, default=None, choices=[4, 8, 16],
                   help="RX J0534: the run's partitions (default: read from its checkpoint)")
    p.add_argument("--crop", type=int, default=80, help="RX J0534: the run's --crop (default 80)")
    p.add_argument("--clean", action="store_true", help="RX J0534: the run used --clean")
    p.add_argument("--workers", default="auto", help="reduction workers (default auto)")
    a = p.parse_args(argv)
    run_dir = os.path.expanduser(a.run_dir or ("~/klip_tpe_runs/rxj0534/readme" if a.target == "rxj0534"
                                               else "~/klip_tpe_runs/hd95086"))
    out = os.path.expanduser(a.out or os.path.join(run_dir, "fm_one_at_a_time"))

    from klip_tpe import Runner
    from klip_tpe.instruments import generic

    if a.target == "rxj0534":
        import run_rxj0534 as rx
        parts = a.partitions or _partitions_of(run_dir)
        red, _ = rx.make_reducer(parts, clean=a.clean, crop=a.crop, workers=a.workers)
        known = [rx.PLANET]
    else:
        import readme_panel as rp
        red = rp.hd95086_reducer(a.workers)
        known = [rp.HD95086_B]
    obj, samp = generic.default_config(red, known=known)
    runner = Runner.resume(run_dir, red, obj, samp)
    diagnose(runner, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
