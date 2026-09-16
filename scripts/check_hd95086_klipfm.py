#!/usr/bin/env python
"""Is the HD 95086 KLIP-FM model right?

    TQDM_DISABLE=1 python scripts/check_hd95086_klipfm.py [--png out.png]

The dashboard's "KLIP-FM model (best)" panel for run G2 shows the three injected sources
sitting on a field of concentric arcs that fills the whole annulus.  Arcs at the sources'
own radii are expected -- a source at radius r is over-subtracted along the arc the field
rotation sweeps at that radius -- so eyeballing the panel cannot say whether the model is
right.  This measures it against the one reference that cannot be argued with:

    numerical FM  =  reduce(data + sources)  -  reduce(data)

Both reductions see the same data and the same configuration, so the speckle field cancels
exactly and what is left is the pipeline's true response to the sources, self-subtraction
and all.  The analytic KLIP-FM (Pueyo 2016 first order, klip_tpe.klip.klip_fm_zone) has to
reproduce it.  Where the two disagree, the analytic one is wrong.

Reported per source and over the whole annulus: peak ratio, the throughput each implies,
and the fraction of the model's total power that lands away from any source.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
for _p in (os.path.join(_HERE, "paper_runs"), os.path.join(os.path.dirname(_HERE), "paper_runs")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from klip_tpe.metrics import radprof, source_xy, star_center
from klip_tpe.reducer import ReductionRequest, Source

# the winner of bench_20260915185229_tpe_s2, annulus 1
CONTRAST = 5.899e-09
SOURCES = [(0.3324160801788618, 21.619248293209097),
           (0.39812500000000006, 141.6192482932091),
           (0.4638339198211383, 261.6192482932091)]
PARAMS = {"spat_mean": False, "temp_mean": False, "bin": 6, "n_ang": 2, "filter": 9,
          "angsep": 0.0, "anglemax": 33, "corr_thresh": 0.0, "noise_max": 3.0,
          "coronoise_max": 3.0, "k_klip": 15, "inrad": 20.0, "outrad": 45.0}


def stamp(img, xy, half=6):
    x, y = int(round(xy[0])), int(round(xy[1]))
    return np.nan_to_num(img[y - half:y + half + 1, x - half:x + half + 1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None)
    ap.add_argument("--contrast", type=float, default=CONTRAST,
                    help="injection contrast; raise it to test the first-order regime")
    a = ap.parse_args(argv)

    def log(m):
        print(m, flush=True)

    from run_demos import hd95086_objects
    red = hd95086_objects()
    # the winner selected K1 only
    if hasattr(red, "select"):
        try:
            red.select(["K1"])
        except Exception:
            pass

    srcs = [Source(r, t, a.contrast) for r, t in SOURCES]
    px = red.pxscale

    log(f"contrast {a.contrast:.3e}   sources {[(round(r,3), round(t,1)) for r, t in SOURCES]}")
    log("reducing: clean, injected, analytic FM ...")
    clean = red.reduce(ReductionRequest(params=PARAMS, tag="clean"))
    inj = red.reduce(ReductionRequest(params=PARAMS, injections=srcs, tag="inj"))
    fm = red.reduce(ReductionRequest(params=PARAMS, fm_sources=srcs, tag="fm"))

    ci, cc, cf = (np.asarray(x.image, float) for x in (inj, clean, fm))
    fa = np.asarray(fm.fm_image, float)                       # analytic KLIP-FM
    fn = ci - cc                                              # numerical FM (ground truth)

    cx, cy = star_center(fa.shape)
    log(f"\n{'':<10}{'analytic peak':>15}{'numerical peak':>16}{'ratio':>9}"
        f"{'T_analytic':>12}{'T_numerical':>13}")
    rows = []
    for i, (rho, th) in enumerate(SOURCES):
        ux, uy = source_xy(np.array([rho]), np.array([th]), px, cx, cy, red.angle_convention)
        xy = (float(ux[0]), float(uy[0]))
        pa_, pn_ = float(np.nanmax(stamp(radprof(fa), xy))), float(np.nanmax(stamp(radprof(fn), xy)))
        rows.append((rho, pa_, pn_))
        r = pa_ / pn_ if pn_ else np.nan
        log(f"src {i+1:<6}{pa_:>15.3e}{pn_:>16.3e}{r:>9.3f}"
            f"{pa_/a.contrast:>12.3e}{pn_/a.contrast:>13.3e}")

    # where does the model's power actually sit?
    yy, xx = np.mgrid[0:fa.shape[0], 0:fa.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    zone = (rr >= PARAMS["inrad"]) & (rr <= PARAMS["outrad"]) & np.isfinite(fa)
    near = np.zeros_like(zone)
    for rho, th in SOURCES:
        ux, uy = source_xy(np.array([rho]), np.array([th]), px, cx, cy, red.angle_convention)
        near |= np.hypot(xx - float(ux[0]), yy - float(uy[0])) <= 2.0 * red.fwhm
    for name, arr in (("analytic", fa), ("numerical", fn)):
        p = np.nan_to_num(arr) ** 2
        frac = p[zone & ~near].sum() / max(p[zone].sum(), 1e-300)
        log(f"\n{name:>10}: {frac*100:6.2f}% of the in-annulus power is MORE THAN 2 FWHM "
            f"from any source")
    good = np.isfinite(fa) & np.isfinite(fn) & zone
    num, den = np.nan_to_num(fa[good]), np.nan_to_num(fn[good])
    corr = float(np.corrcoef(num, den)[0, 1]) if num.size > 2 else np.nan
    log(f"{'':>10}  analytic vs numerical, over the annulus: r = {corr:.3f}")
    log(f"{'':>10}  rms(analytic - numerical) / rms(numerical) = "
        f"{np.std(num - den) / max(np.std(den), 1e-300):.3f}")

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mc
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6))
        pk = float(np.nanmax(np.abs(fn[zone])))
        for axi, (t, arr) in zip(ax, (("analytic KLIP-FM", fa), ("numerical  inj - clean", fn),
                                      ("analytic - numerical", fa - fn))):
            n = mc.SymLogNorm(linthresh=pk * 0.1, vmin=-pk, vmax=pk, base=10)
            axi.imshow(np.where(zone, arr, np.nan), origin="lower", cmap="inferno", norm=n)
            axi.set_title(t, fontsize=9); axi.set_xticks([]); axi.set_yticks([])
        plt.tight_layout(); plt.savefig(a.png, dpi=130)
        log(f"\nwrote {a.png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
