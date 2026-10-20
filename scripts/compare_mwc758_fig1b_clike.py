#!/usr/bin/env python3
"""compare_mwc758_fig1b_clike.py -- a c-tuned klip-tpe search of 2019 Nov 14 next to the source of the
published Fig. 1b of Wagner et al. (2023, Nature Astronomy 7, 1208): the paper's ASDI image of 3.8-4.14 um
(the 2022-02-11 run), klip-tpe at the paper's ADI settings on the same band, and the search's validated
winner, on all its groups and on those at 3.8 um and beyond.

  python3 scripts/compare_mwc758_fig1b_clike.py --run RUN_DIR [--reduced DIR] [--published PNG]
          [--out DIR] [--workers N]

RUN_DIR is a finished scripts/run_mwc758_ales_clike.py run (``--nights nov14``; of a two-night run the
November 14 groups are taken).  For every image: MWC 758c's S/N as tutorial 7 measures it (the Mawet S/N
with the n2_g10 template as the matched filter), and where c stands among the 31 empty positions of its
ring (every 10 deg, out of the 50 deg wedge round c): z = (c - their mean) / their sd, and how many of them
reach c.  The panels are shown as Fig. 1b is (a unit-peak Gaussian of FWHM lambda/D at 3.97 um, zero within
6 and beyond 22 px, the published colour map), each with its own stretch, -1 to +5 robust sigma of its ring.
Writes OUT/fig1b_clike.png and OUT/fig1b_clike.json (OUT defaults to RUN_DIR/fig1b_comparison).  Only reads
RUN_DIR, the paper's products and the release."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
import warnings

import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import mwc758_idl2022 as M                                       # noqa: E402
from klip_tpe import datasets                                    # noqa: E402
from klip_tpe.instruments import ales                            # noqa: E402
from klip_tpe.metrics import MawetPeakSNR, source_xy             # noqa: E402
from klip_tpe.reducer import set_klip_gram, set_zone_center      # noqa: E402

RHO, PA = 0.603, 228.5
RED_GROUPS = ["g%02d" % g for g in range(15, 20)]                # 3.79-4.15 um, Fig. 1b's band
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--reduced", default="/Volumes/RAID36TB/LMIRCam/MWC758_6/reduced")
ap.add_argument("--published", default=os.path.join(REPO, "tutorials", "runs", "mwc758_paper_setup", "paper_figure",
                                                    "published_fig1_capture.png"))
ap.add_argument("--out", default=None)
ap.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 2)))
a = ap.parse_args()
out_dir = a.out or os.path.join(a.run, "fig1b_comparison")
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------- the run's winner
res = json.load(open(os.path.join(a.run, "final_results.json")))["annuli"][0]
setup = json.load(open(os.path.join(a.run, "run_setup.json")))
spec = (setup.get("reducer") or {}).get("injection_spectrum") or {}
kept = list(res["partitions"])
blk = res["winner_config"]["per_partition"][kept[0]]
parts = fits.getdata(os.path.join(a.run, "annulus01", "best_clean_partitions.fits"))
parts = parts[None] if parts.ndim == 2 else parts
n2 = [i for i, p in enumerate(kept) if p.startswith("n2_")]
n2_red = [i for i in n2 if kept[i].split("_")[1] in RED_GROUPS]
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    win_all = np.nanmean(parts[n2], axis=0)                     # one night: equal weights within it
    win_red = np.nanmean(parts[n2_red], axis=0) if n2_red else None

# ---------------------------------------------------------------- klip-tpe at the paper's settings
files = datasets.fetch("lbti_ales_mwc758", quiet=True,
                       roles=[k for k in datasets.DATASETS["lbti_ales_mwc758"] if k.endswith("_20191114")])
dsets, lam, fwhm, table = ales.load_release(files, nights=["20191114"], labels={"20191114": "n2"},
                                           only=sorted(set(RED_GROUPS) | {"g10"}), log=lambda s: None)
red = ales.make_reducer(dsets, lam, fwhm, table=table, max_workers=a.workers, representative="n2_g10", log=lambda s: None)
set_klip_gram(red, "zone")
set_zone_center(red, (30.0, 30.0), angles="getzone2015")         # the paper's zones: the 2015 getzone about (sc/2-1, sc/2-1)
E0, E1 = 0.4 / red.pxscale, 0.8 / red.pxscale
space = ales.make_space(red, bin_range=(5, 30), zone_range=(0.0, 8.0), search_post_hp=True)
PAPER = dict(bin=15, n_ang=6, filter=11, angsep=0.01, anglemax=127.0, corr_thresh=0.0, noise_max=3.0,
             coronoise_max=3.0, k_klip=15, zone_in=E0 - 5.0, zone_out=27.0 - E1, post_hp=0)
cfg = space.decode(space.default_vector())
for d in [cfg.params, *cfg.per_partition.values()]:
    d.update(PAPER)
    d.update(inrad=E0, outrad=E1, pad_in=False, pad_out=False)
paper_kt = red.reduce_config(cfg, selected=["n2_" + g for g in RED_GROUPS]).image
red.close()

# ---------------------------------------------------------------- c on its ring
metric = MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel, declip_nsig=None)
ring = MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel, declip_nsig=None,
                    known=[(RHO, PA)])
TH = np.array([t for t in np.arange(0.0, 360.0, 10.0) if abs(((t - PA + 180) % 360) - 180) > 25.0])


def c_on_ring(img):
    c = float(metric.per_source(img, None, [RHO], [PA])[0])
    e = np.asarray(ring.per_source(img, None, [RHO] * TH.size, list(TH)), float)
    e = e[np.isfinite(e)]
    return {"c_snr": c, "ring_mean": float(e.mean()), "ring_sd": float(e.std()), "ring_max": float(e.max()),
            "z": float((c - e.mean()) / e.std()), "n_ring_at_or_above_c": int((e >= c).sum()), "n_ring": int(e.size)}


ng = len(n2)
panels = [("Fig. 1b source: the paper's ASDI,\n3.8-4.14 um (2022-02-11 run)",
           fits.getdata(os.path.join(a.reduced, "MWC758_sdiklip_3.8-4.14.fits")).astype(float)),
          ("klip-tpe at the paper's ADI settings,\n3.8-4.15 um (g15-g19)", paper_kt),
          (f"the c-tuned winner, its {ng} Nov 14\ngroups ({kept[n2[0]].split('_')[1]}-{kept[n2[-1]].split('_')[1]})", win_all)]
if win_red is not None:
    panels.append((f"the c-tuned winner, its {len(n2_red)} groups\nat 3.8-4.15 um", win_red))
stats = {name.replace("\n", " "): c_on_ring(im) for name, im in panels}
for name, s in stats.items():
    print(f"{name:70s} c S/N {s['c_snr']:5.2f}; ring {s['ring_mean']:+.2f} +- {s['ring_sd']:.2f} (max {s['ring_max']:.2f}); "
          f"z {s['z']:+.2f}; {s['n_ring_at_or_above_c']} of {s['n_ring']} ring positions at or above c")

# ---------------------------------------------------------------- the figure
import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import Circle                            # noqa: E402

try:
    sys.path.insert(0, HERE)
    from compare_mwc758_paper_figure import Published            # noqa: E402
    cmap = Published(a.published).cmap()
except Exception as exc:                                         # no capture at hand: a stand-in map
    print(f"(published colour map unavailable: {exc!r}; using inferno)")
    cmap = "inferno"
xc, yc = (v[0] for v in source_xy(RHO, PA, red.pxscale, 30.5, 30.5))
yy, xx = np.mgrid[0:62, 0:62]
rr = np.hypot(xx - 31.0, yy - 31.0)
fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.3))
for ax, (name, im) in zip(np.atleast_1d(axes), panels):
    sm = M.mask_annulus(M.lod_smooth(np.nan_to_num(im), 3.97), 6, 22)
    v = sm[(rr >= 7) & (rr <= 21)]
    sig = 1.4826 * np.median(np.abs(v - np.median(v)))
    ax.imshow(sm, origin="lower", cmap=cmap, vmin=-1.0 * sig, vmax=5.0 * sig, interpolation="nearest")
    ax.add_patch(Circle((xc, yc), 3.6, fill=False, ec="w", lw=1.0, ls="--"))
    s = stats[name.replace("\n", " ")]
    ax.set_title(f"{name}\nc S/N {s['c_snr']:.1f}, z {s['z']:+.1f} on its ring", fontsize=8.5)
    ax.set_xlim(4, 57)
    ax.set_ylim(4, 57)
    ax.set_xticks([])
    ax.set_yticks([])
keep = ("bin", "filter", "k_klip", "n_ang", "angsep", "anglemax", "corr_thresh", "noise_max", "coronoise_max",
        "zone_in", "zone_out", "post_hp", "nkeep_sdi")
# the winner's processing geometry: its KLIP zone (the annulus widened by zone_in / zone_out, capped at 27 px)
zc = (((setup.get("reducer") or {}).get("partitions") or {}).get(str(kept[0])) or {}).get("defaults", {}).get("zone_center")
z_in = max(E0 - float(blk.get("zone_in") or 0.0), 0.0)
z_out = min(E1 + float(blk.get("zone_out") or 0.0), 27.0)
geom = (f"winner's KLIP zone {z_in:.1f}-{z_out:.1f} px in {blk.get('n_ang')} sector(s) about "
        + (f"({zc[0]:g}, {zc[1]:g})" if zc else "(31, 31)") + "; the paper's 5-27 px in 6 about (30, 30)")
print(geom)
wtxt = "  ".join(f"{k} {blk[k]:.3g}" if isinstance(blk[k], float) else f"{k} {blk[k]}" for k in keep if blk.get(k) is not None)
fig.suptitle("MWC 758, LBTI/ALES 2019-11-14: Fig. 1b's source next to klip-tpe tuned on c-coloured injections "
             f"(validated S/N {res['winner_score']:.1f} at contrast {res['contrast']:.2g})", fontsize=10)
fig.text(0.01, 0.015, "winner: " + wtxt + "\n" + geom + ("\ninjections: " + spec["label"] if spec.get("label") else ""),
         fontsize=7)
fig.text(0.99, 0.015, "north up, east left; lambda/D-smoothed as Fig. 1b; each panel -1 to +5 robust sigma",
         fontsize=7, ha="right")
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
png = os.path.join(out_dir, "fig1b_clike.png")
fig.savefig(png, dpi=150, facecolor="white")
json.dump({"run": os.path.abspath(a.run), "winner": {k: blk.get(k) for k in keep}, "winner_score": res["winner_score"],
           "contrast": res["contrast"], "partitions": kept, "injection_spectrum": spec.get("label"),
           "paper_settings": PAPER, "winner_geometry": geom, "panels": stats}, open(os.path.join(out_dir, "fig1b_clike.json"), "w"), indent=1,
          default=float)
print("wrote", png)
