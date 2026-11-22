#!/usr/bin/env python3
"""Draw Fig. 8 of the KLIP-TPE paper (f6_contrast.pdf) from the final-image grid's curves.

The layout follows ``paper_runs/figs.py:fig_contrast``: one panel per data set (beta Pic A2,
HD 95086 C, HIP 65426 NIRCam D2 on pyKLIP), the optimized curve, the KLIP-FM cross-check where the
run computed it, the seeded default scaled by the paired gain of each annulus, the annulus edges,
and the companion's separation and published contrast.  The optimized curve is the final-image
grid's (``fig8_curve.py``): the median contrast at S/N = 5 over position angle at each separation,
with the 16th to 84th percentiles as a band.

Usage::

    cd paper_runs && python3 fig8_plot.py

reads ``fig8/fig8_<stage>.json`` and ``summary.json`` where ``run_demos.py`` puts its runs (this
folder, or ``$RUNS_DIR``) and writes ``figs/f6_contrast.pdf`` there, over the throughput-fit
curve that ``figs.py``'s ``fig_contrast`` still draws, so run it after ``figs.py``.
``--curves``, ``--summary`` and ``--out`` override the three, and ``--data DIR`` takes both
inputs from one folder.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#: Panel order and the stage that stands for each target, as in figs.py.
PANELS = [("betapic", "A2"), ("hd95086", "C"), ("hip65426", "D2")]
NICE = {"betapic": r"$\beta$ Pic b (VLT/NACO $L'$)",
        "hd95086": "HD 95086 b (VLT/SPHERE $K1K2$)",
        "hip65426": "HIP 65426 b (JWST/NIRCam F444W)"}


def main():
    """Draw the figure from the curves and the paper's summary.json, as the command line asks."""
    # where run_demos.py puts its runs, worked out as it does without importing the package
    runs = os.path.abspath(os.path.expanduser(os.environ.get("RUNS_DIR", "").strip()
                                              or os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--curves", default=os.path.join(runs, "fig8"),
                    help="folder with fig8_<stage>.json (default: fig8/ in the runs' folder)")
    ap.add_argument("--summary", default=os.path.join(runs, "summary.json"),
                    help="collect.py's summary.json (default: the runs' folder)")
    ap.add_argument("--data", default=None, help="one folder holding both, in place of --curves and --summary")
    ap.add_argument("--out", default=os.path.join(runs, "figs", "f6_contrast.pdf"),
                    help="output file, PDF or PNG (default: figs/f6_contrast.pdf in the runs' folder)")
    a = ap.parse_args()
    if a.data:
        a.curves, a.summary = a.data, os.path.join(a.data, "summary.json")
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    s = json.load(open(a.summary))
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.1, 2.5))
    for ax, (t, st) in zip(np.atleast_1d(axes), PANELS):
        f = json.load(open(os.path.join(a.curves, f"fig8_{st}.json")))
        cv = f["curve"]
        r = np.asarray(cv["r_as"], float)
        c = np.asarray([np.nan if v is None else v for v in cv["curve"]], float)
        lo = np.asarray([np.nan if v is None else v for v in cv["p16"]], float)
        hi = np.asarray([np.nan if v is None else v for v in cv["p84"]], float)
        ann = np.asarray(cv["annulus"], int)
        rec = s[st]
        fs = float(rec.get("flux_scale_applied") or 1.0)       # 1.0: no axis is scaled onto a companion
        ok = np.isfinite(c)
        ax.fill_between(r[ok], lo[ok] / fs, hi[ok] / fs, color="0.82", lw=0, zorder=1,
                        label="16th to 84th percentile")
        ax.semilogy(r[ok], c[ok] / fs, "k-", lw=1.3, marker="o", ms=2.2, zorder=3,
                    label="optimized (one source per reduction)")
        cur = rec.get("curves") or {}
        if "fm" in cur:
            ax.semilogy(cur["fm"]["r_as"], np.asarray(cur["fm"]["c5"]) / fs, "--", color="seagreen", lw=1.0,
                        zorder=2, label="KLIP-FM cross-check")
        # the seeded default: the optimized curve of each annulus times that annulus' paired gain
        for k, an in enumerate(rec["annuli"]):
            m = ok & (ann == k + 1)
            fac = an.get("gain") or (an["winner_score"] / max(an["default_score"], 1e-9))
            if m.any():
                ax.semilogy(r[m], c[m] / fs * fac, ":", color="crimson", lw=1.1, zorder=2,
                            label="seeded default" if k == 0 else None)
            ax.axvline(an["inrad_as"], color="0.85", lw=0.6, zorder=0)
        ax.axvline(rec["planet"][0], color="c", ls=":", lw=0.9)
        pub = rec.get("published_companion_contrast")
        if pub:
            ax.plot([rec["planet"][0]], [pub], "*", ms=9, color="c", mec="k", mew=0.4, zorder=5)
        ax.set_title(NICE[t], fontsize=7.5)
        ax.set_xlabel("separation (arcsec)")
        ax.grid(alpha=.25, which="both")
    np.atleast_1d(axes)[0].set_ylabel(r"5$\sigma$ contrast")
    fig.tight_layout()
    # one key under the row, gathered from every panel (NIRCam has no KLIP-FM curve)
    seen = {}
    for ax in np.atleast_1d(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    order = ["optimized (one source per reduction)", "16th to 84th percentile", "KLIP-FM cross-check",
             "seeded default"]
    labs = [k for k in order if k in seen]
    fig.legend([seen[k] for k in labs], labs, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(labs),
               frameon=False, fontsize=6.5)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out)
    plt.close(fig)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
