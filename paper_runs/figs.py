#!/usr/bin/env python
"""Build the paper figures from the runs in this directory.

    python figs.py            # everything it can find, into ./figs/

Writes (and caches the default-configuration reductions it needs as FITS):

  f1_display.png     the live diagnostic display at the end of an annulus
  f2_trace.pdf       honest-score convergence for the three data sets
  f3_landscape.pdf   sampled parameter landscape (one annulus)
  f4_partition.pdf   partition (time-group) selection on beta Pic
  f5_gallery.pdf     default vs optimized reductions, companion circled
  f6_contrast.pdf    injection-calibrated 5-sigma contrast + KLIP-FM cross-check
  f7_bench.pdf       search-strategy benchmark
  f8_paramverify.pdf parameter-ensemble reliability maps
"""
import json
import os
import shutil
import sys
import warnings

import numpy as np
from astropy.io import fits

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import collect as C
import run_demos as R
from klip_tpe.bench import read_records
from klip_tpe.metrics import source_xy, star_center
from klip_tpe.reducer import ReductionRequest

OUT = R.OUT
FIG = os.path.join(OUT, "figs")
CACHE = os.path.join(OUT, "_default_images")
os.makedirs(FIG, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

# which run is the primary demonstration of each target
PRIMARY = {"betapic": "A2", "hd95086": "C", "hip65426": "D"}
NICE = {"betapic": r"$\beta$ Pic b (VLT/NACO $L'$)",
        "hd95086": "HD 95086 b (VLT/SPHERE $K1K2$)",
        "hip65426": "HIP 65426 b (JWST/NIRCam F444W)"}


def summary():
    return json.load(open(os.path.join(OUT, "summary.json")))


def default_image(which, ia, rin, rout):
    """The seeded-default reduction over one annulus, cached as FITS."""
    p = os.path.join(CACHE, f"{which}_a{ia + 1}.fits")
    if os.path.exists(p):
        return np.asarray(fits.getdata(p), float)
    red, space, _, _ = C.build(which)
    img = red.reduce(ReductionRequest(params=dict(space.decode(space.default_vector()).params,
                                                  inrad=rin, outrad=rout))).image
    fits.writeto(p, np.asarray(img, np.float32), overwrite=True)
    return np.asarray(img, float)


def snr_map(img, fwhm, known=None, px=None, excl_fwhm=1.5):
    """Per-pixel matched-filter S/N (radial-profile flattened, azimuthal sigma).

    ``known`` = [(rho_as, pa_deg), ...] is kept out of the per-ring median and scatter, as
    the per-source metric keeps it out of its noise apertures: a companion left in its own
    ring inflates sigma there and suppresses the map exactly where the reader is looking
    (Section "Known sources distort ..." measures 73% on HD 95086)."""
    from scipy import ndimage
    from klip_tpe.metrics import gaussian_kernel, radprof
    a = ndimage.convolve(np.where(np.isfinite(radprof(img)), radprof(img), 0.0),
                         gaussian_kernel(fwhm), mode="nearest")
    ny, nx = img.shape
    cx, cy = star_center(img.shape)
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - cx, yy - cy)
    stat = np.isfinite(a)
    for k in (known or ()):
        if px:
            kx, ky = source_xy([k[0]], [k[1]], px, cx, cy)
            stat &= np.hypot(xx - kx[0], yy - ky[0]) > excl_fwhm * fwhm
    out = np.full_like(a, np.nan)
    for r0 in range(int(rr.max()) + 1):
        ring = (rr >= r0 - 0.5) & (rr < r0 + 0.5)
        m = ring & np.isfinite(a)
        ref = ring & stat
        if m.sum() < 6 or ref.sum() < 6:
            continue
        v = a[ref]
        med = np.median(v)
        sg = 1.4826 * np.median(np.abs(v - med))
        if sg > 0:
            out[m] = (a[m] - med) / sg
    return out


#: the companion marker, dark enough to read on a white background
PLANET_EC = "#0b5394"


def _img_kw(lo, hi, snr=False):
    """Colour mapping for an image panel, on a WHITE background.

    A printed page should not be a field of black ink, and a reader should be able to see
    where the data are zero.  Intensity panels therefore use a sequential white-to-black
    map; signed panels (matched-filter S/N, the parameter-verification maps) use a
    diverging map centred on zero, so that zero is white, sources are red and
    over-subtraction is blue.  ``NaN`` outside the optimized annulus stays the axes'
    white."""
    from matplotlib.colors import Normalize, TwoSlopeNorm
    if snr:
        if lo < 0 < hi:
            return dict(cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi))
        if hi <= 0:
            return dict(cmap="Blues_r", norm=Normalize(vmin=lo, vmax=hi))
        return dict(cmap="Reds", norm=Normalize(vmin=max(lo, 0.0), vmax=hi))
    return dict(cmap="Greys", norm=Normalize(vmin=lo, vmax=hi))


def _show(ax, img, px, planet, title, vlim=None, box_as=None, snr=False):
    ny, nx = img.shape
    cx, cy = star_center(img.shape)
    ext = [(-0.5 - cx) * px, (nx - 0.5 - cx) * px, (-0.5 - cy) * px, (ny - 0.5 - cy) * px]
    v = img[np.isfinite(img)]
    lo, hi = vlim or np.nanpercentile(v, [1.0, 99.6])
    ax.set_facecolor("white")
    ax.imshow(img, origin="lower", extent=ext, **_img_kw(lo, hi, snr))
    xs, ys = source_xy([planet[0]], [planet[1]], 1.0, 0.0, 0.0)
    ax.add_patch(Circle((xs[0], ys[0]), 0.12, fill=False, ec=PLANET_EC, lw=1.0))
    # x = cx + r cos(PA+90) in this codebase, so East is -x: plotted with x increasing
    # rightward the frame is ALREADY North up, East left.  Do not flip it again.
    if box_as:
        ax.set_xlim(-box_as, box_as)
        ax.set_ylim(-box_as, box_as)
    ax.set_title(title, fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([])
    return lo, hi


def compass(ax, frac=0.16, color="0.15", lw=1.0):
    """N/E arrows in the corner -- North is +y, East is -x (see _show).  Drawn in axes
    fractions with a white stroke so they read on both bright and dark backgrounds."""
    import matplotlib.patheffects as pe
    stroke = [pe.withStroke(linewidth=1.8, foreground="w")]
    ox, oy = 0.30, 0.10
    for dx, dy, lab in ((0.0, frac, "N"), (-frac, 0.0, "E")):
        ar = ax.annotate("", xy=(ox + dx, oy + dy), xytext=(ox, oy), xycoords="axes fraction",
                         textcoords="axes fraction",
                         arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=7,
                                         shrinkA=0, shrinkB=0))
        ar.arrow_patch.set_path_effects(stroke)
        ax.text(ox + 1.45 * dx, oy + 1.45 * dy, lab, color=color, fontsize=6.5,
                ha="center", va="center", transform=ax.transAxes).set_path_effects(stroke)


# --------------------------------------------------------------------- f5 gallery
def fig_gallery(s):
    tg = [t for t in PRIMARY if PRIMARY[t] in s]
    fig, axes = plt.subplots(len(tg), 3, figsize=(7.1, 2.45 * len(tg)))
    axes = np.atleast_2d(axes)
    for row, t in enumerate(tg):
        w = PRIMARY[t]
        r = s[w]
        px, planet = r["pxscale"], r["planet"]
        # the annulus that contains the companion
        a = min(r["annuli"], key=lambda a: abs(0.5 * (a["inrad_as"] + a["outrad_as"]) - planet[0])
                if not (a["inrad_as"] <= planet[0] <= a["outrad_as"]) else -1)
        img0 = default_image(w, a["annulus"], a["inrad_px"], a["outrad_px"])
        img1 = np.asarray(fits.getdata(os.path.join(
            OUT, r["run"], f"annulus{a['annulus'] + 1:02d}", "best_clean.fits")), float)
        box = 1.25 * a["outrad_as"]
        _show(axes[row, 0], img0, px, planet,
              f"default   S/N {a['planet_snr_default']:.1f}", box_as=box)
        _show(axes[row, 1], img1, px, planet,
              f"optimized   S/N {a['planet_snr_optimized']:.1f}", box_as=box)
        m = snr_map(img1, r["fwhm_px"], known=[planet], px=px)
        _show(axes[row, 2], m, px, planet, "optimized S/N map", vlim=(-4, 6), box_as=box, snr=True)
        axes[row, 0].set_ylabel(NICE[t], fontsize=7.5)
        axes[row, 0].text(0.03, 0.92, f"{a['inrad_as']:.2f}--{a['outrad_as']:.2f}\"",
                          transform=axes[row, 0].transAxes, color="0.15", fontsize=6.5)
        for c in range(3):
            compass(axes[row, c])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f5_gallery.pdf"))
    plt.close(fig)
    print("  f5_gallery.pdf")


# --------------------------------------------------------------------- f2 traces
def fig_trace(s):
    tg = [t for t in PRIMARY if PRIMARY[t] in s]
    fig, axes = plt.subplots(1, len(tg), figsize=(7.1, 2.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, tg):
        w = PRIMARY[t]
        run = os.path.join(OUT, s[w]["run"])
        ia = s[w]["annuli"][0]["annulus"]
        # every calibration segment, as before, but each evaluation once: a resume replays
        # the evaluations after its checkpoint and re-logs them (bench.read_records)
        rec = read_records(run, annulus=ia, last_segment=False)
        y = np.array([np.nan if r.get("score") is None else r["score"] for r in rec])
        warm = np.array([str(r.get("phase", "")).startswith(("warm", "seed", "explore")) for r in rec])
        n = np.arange(1, y.size + 1)
        run_best = np.fmax.accumulate(np.where(np.isfinite(y), y, -np.inf))
        ax.plot(n[warm], y[warm], "o", ms=2.0, mfc="none", mec="0.6", mew=0.5, label="warm-up")
        ax.plot(n[~warm], y[~warm], "o", ms=2.0, color="#1f77b4", label="TPE")
        ax.plot(n, np.where(np.isfinite(run_best), run_best, np.nan), "k-", lw=1.2, label="running best")
        a0 = s[w]["annuli"][0]
        ax.axhline(a0["default_score"], color="crimson", ls="--", lw=0.9, label="seeded default")
        ax.plot([n[-1]], [a0["winner_score"]], "*", ms=9, color="darkorange", label="validated")
        ax.set_title(NICE[t], fontsize=7.5)
        ax.set_xlabel("evaluation")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("median injected S/N (clean-subtracted)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f2_trace.pdf"))
    plt.close(fig)
    print("  f2_trace.pdf")


# --------------------------------------------------------------------- f6 contrast
def fig_contrast(s):
    tg = [t for t in PRIMARY if PRIMARY[t] in s]
    fig, axes = plt.subplots(1, len(tg), figsize=(7.1, 2.5))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, tg):
        w = PRIMARY[t]
        r = s[w]
        cur = r.get("curves") or {}
        fs = float(r.get("flux_scale_applied") or 1.0)  # 1.0 unless collect ran with ANCHOR_APPLY=1
        if "main" in cur:
            ax.semilogy(cur["main"]["r_as"], np.asarray(cur["main"]["c5"]) / fs, "k-", lw=1.3,
                        label="optimized (injection-calibrated)")
        if "fm" in cur:
            ax.semilogy(cur["fm"]["r_as"], np.asarray(cur["fm"]["c5"]) / fs, "--",
                        color="seagreen", lw=1.0, label="KLIP-FM cross-check")
        # the default configuration, scaled from the same annulus calibration
        for a in r["annuli"]:
            if not a.get("sigma_default") or not a.get("sigma_optimized"):
                continue
            rr = np.asarray(a["sigma_default"]["r_as"])
            if "main" in cur:
                base = np.interp(rr, cur["main"]["r_as"], np.asarray(cur["main"]["c5"]) / fs)
                # paired gain (winner and default re-measured on the same injections) when
                # collect recorded one; the validated/default ratio otherwise
                fac = a.get("gain") or (a["winner_score"] / max(a["default_score"], 1e-9))
                ax.semilogy(rr, base * fac, ":", color="crimson", lw=1.1,
                            label="seeded default" if a is r["annuli"][0] else None)
            for e in (a["inrad_as"],):
                ax.axvline(e, color="0.85", lw=0.6, zorder=0)
        ax.axvline(r["planet"][0], color="c", ls=":", lw=0.9)
        pub = r.get("published_companion_contrast")
        if pub:
            ax.plot([r["planet"][0]], [pub], "*", ms=9, color="c", mec="k", mew=0.4, zorder=5)
        ax.set_title(NICE[t], fontsize=7.5)
        ax.set_xlabel("separation (arcsec)")
        ax.grid(alpha=.25, which="both")
    axes[0].set_ylabel(r"5$\sigma$ contrast")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f6_contrast.pdf"))
    plt.close(fig)
    print("  f6_contrast.pdf")


# --------------------------------------------------------------------- f4 partitions
def fig_partition(s, key=None):
    key = key or ("B2" if "B2" in s else "B")
    if key not in s:
        return
    r = s[key]
    run = os.path.join(OUT, r["run"])
    recs = read_records(run, last_segment=False)          # each evaluation once (see fig_trace)
    parts = r["partitions"]
    sel = [set(str(x) for x in (rc.get("meta", {}).get("selected") or parts)) for rc in recs]
    y = np.array([np.nan if rc.get("score") is None else rc["score"] for rc in recs])
    M = np.array([[1.0 if p in sset else np.nan for sset in sel] for p in parts])
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.3), gridspec_kw={"width_ratios": [2.1, 1]})
    im = ax[0].imshow(M * y[None, :], aspect="auto", origin="lower", cmap="viridis",
                      extent=[1, len(y), -0.5, len(parts) - 0.5], interpolation="nearest")
    ax[0].set_yticks(range(len(parts)))
    ax[0].set_yticklabels(parts)
    ax[0].set_xlabel("evaluation")
    ax[0].set_title("group inclusion (colour = injected S/N)", fontsize=7.5)
    fig.colorbar(im, ax=ax[0], pad=0.01).set_label("median S/N", fontsize=6.5)
    win = set(r["annuli"][0]["selected"])
    for j, p in enumerate(parts):
        inn = np.array([p in sset for sset in sel])
        med = {}
        for m, dx, c, lab in ((inn, -0.16, "#1f77b4", "included"),
                              (~inn, 0.16, "#d62728", "excluded")):
            v = y[m & np.isfinite(y)]
            if not v.size:
                continue
            ax[1].plot(np.full(v.size, j) + dx + np.random.uniform(-0.05, 0.05, v.size),
                       v, ".", ms=1.5, color=c, alpha=.35, label=lab if j == 0 else None)
            med[lab] = float(np.median(v))
            ax[1].plot([j + dx - 0.09, j + dx + 0.09], [med[lab]] * 2, "-", color=c, lw=2.2)
        if len(med) == 2:
            d = med["included"] - med["excluded"]
            ax[1].annotate(f"{d:+.1f}", (j, max(med.values())), textcoords="offset points",
                           xytext=(0, 7), ha="center", fontsize=6.5,
                           color="k" if d > 0 else "#d62728")
    ax[1].set_xticks(range(len(parts)))
    ax[1].set_xticklabels([f"{p}\n{'kept' if p in win else 'dropped'}" for p in parts])
    ax[1].set_xlim(-0.5, len(parts) - 0.5)
    ax[1].set_ylabel("median injected S/N")
    ax[1].set_title("marginal effect of each group", fontsize=7.5)
    ax[1].legend(frameon=False, fontsize=6, loc="lower left", markerscale=4)
    ax[1].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f4_partition.pdf"))
    plt.close(fig)
    print("  f4_partition.pdf")


# --------------------------------------------------------------------- f8 param-verify
def fig_paramverify(s, key="A2"):
    if key not in s:
        return
    run = os.path.join(OUT, s[key]["run"])
    names = [("paramverify_stim_stitched.fits", "parameter-STIM"),
             ("paramverify_detfrac_stitched.fits", "detection fraction"),
             ("paramverify_recovery_stitched.fits", "recovery consistency"),
             ("paramverify_combsnr_stitched.fits", "meta-combined S/N")]
    have = [(f, t) for f, t in names if os.path.exists(os.path.join(run, f))]
    if not have:
        return
    px, planet = s[key]["pxscale"], s[key]["planet"]
    fig, axes = plt.subplots(1, len(have), figsize=(1.85 * len(have), 2.1))
    for ax, (f, t) in zip(np.atleast_1d(axes), have):
        img = np.asarray(fits.getdata(os.path.join(run, f)), float)
        _show(ax, img, px, planet, t, snr=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f8_paramverify.pdf"))
    plt.close(fig)
    print("  f8_paramverify.pdf")


# --------------------------------------------------------------------- copies
def rebuild_books(run, ia0):
    """Re-render the corner and parameter-history books from the run's records.

    Both are written by the live display as the annulus finishes, so a copy in
    ``figs/`` is only as current as the last run -- and books written while the live
    panel was drawing inherited its dark theme (fixed in klip_tpe.display, but the
    products on disk keep whatever they were written with).  ``annulus_from_run``
    rebuilds them from ``results.jsonl`` in a few seconds and needs no reduction, so
    the paper figures are always rendered by the installed display code."""
    from klip_tpe.display import annulus_from_run, render_corner, render_parhist_book
    d = os.path.join(run, f"annulus{ia0 + 1:02d}")
    try:
        ad = annulus_from_run(run, ia0)
    except Exception as e:                      # a partial run still gets the old copy
        print(f"  (books not rebuilt: {type(e).__name__}: {e})")
        return
    render_corner(ad, os.path.join(d, "corner.pdf"))
    render_parhist_book(ad, os.path.join(d, "parhist.pdf"))


def copies(s):
    def grab(src, dst):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(FIG, dst))
            print(f"  {dst}")
    prim = "B2" if "B2" in s else PRIMARY["betapic"]
    if prim in s:
        run = os.path.join(OUT, s[prim]["run"])
        ia0 = s[prim]["annuli"][0]["annulus"]
        ia = ia0 + 1
        rebuild_books(run, ia0)
        grab(os.path.join(run, f"annulus{ia:02d}", "step_display_white.png"), "f1_display.png")
        grab(os.path.join(run, f"annulus{ia:02d}", "corner.pdf"), "f3_landscape.pdf")
        grab(os.path.join(run, f"annulus{ia:02d}", "parhist.pdf"), "f10_parhist.pdf")


# --------------------------------------------------------------------- f9 stitched
def fig_stitch(s, key="A2"):
    if key not in s:
        return
    r = s[key]
    run = os.path.join(OUT, r["run"])
    img = np.asarray(fits.getdata(os.path.join(run, "klip_stitched.fits")), float)
    px, planet = r["pxscale"], r["planet"]
    # Blank the region inside the first annulus: nothing there was optimized, and the raw
    # core residual is orders of magnitude brighter than the searched field, so left in it
    # saturates the stretch and hides everything the figure is about.
    ny0, nx0 = img.shape
    cx0, cy0 = star_center(img.shape)
    yy0, xx0 = np.mgrid[0:ny0, 0:nx0]
    img = np.where(np.hypot(xx0 - cx0, yy0 - cy0) * px < r["annuli"][0]["inrad_as"], np.nan, img)
    m = snr_map(img, r["fwhm_px"], known=[planet], px=px)
    fig, ax = plt.subplots(2, 1, figsize=(3.4, 6.4))
    box = 1.05 * r["annuli"][-1]["outrad_as"]
    # the inner annulus carries far more residual power than the outer ones, so the
    # stitched frame is shown on a symmetric-log stretch set by the outer field
    from matplotlib.colors import SymLogNorm
    ny, nx = img.shape
    cx, cy = star_center(img.shape)
    yy, xx = np.mgrid[0:ny, 0:nx]
    out = np.hypot(xx - cx, yy - cy) * px > 0.55 * box
    sg = 1.4826 * np.nanmedian(np.abs(img[out & np.isfinite(img)] -
                                      np.nanmedian(img[out & np.isfinite(img)])))
    ext = [(-0.5 - cx) * px, (nx - 0.5 - cx) * px, (-0.5 - cy) * px, (ny - 0.5 - cy) * px]
    ax[0].set_facecolor("white")
    ax[0].imshow(img, origin="lower", extent=ext, cmap="Greys",
                 norm=SymLogNorm(linthresh=3 * sg, vmin=-6 * sg, vmax=120 * sg, base=10))
    xs, ys = source_xy([planet[0]], [planet[1]], 1.0, 0.0, 0.0)
    ax[0].add_patch(Circle((xs[0], ys[0]), 0.12, fill=False, ec=PLANET_EC, lw=1.0))
    ax[0].set_xlim(-box, box); ax[0].set_ylim(-box, box)
    ax[0].set_title("optimized stitched reduction", fontsize=7.5)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    _show(ax[1], m, px, planet, r"per-pixel S/N", vlim=(-4, 6), box_as=box, snr=True)
    for a_ in ax:
        compass(a_)
    for a in r["annuli"][1:]:
        for axx in ax:
            axx.add_patch(Circle((0, 0), a["inrad_as"], fill=False, ec="0.45", lw=0.5, ls=":"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f9_stitch.pdf"))
    plt.close(fig)
    print("  f9_stitch.pdf")


# --------------------------------------------------------------------- f11 frame tags
def fig_frametags(s):
    """The three per-frame quality tags, for the two ground-based sequences."""
    sets = [(k, lab) for k, lab in (("A2", r"$\beta$ Pic (NACO $L'$)"),
                                    ("C", "HD 95086 (SPHERE $K1K2$)")) if k in s]
    if not sets:
        return
    fig, axes = plt.subplots(1, 2 * len(sets), figsize=(1.9 * 2 * len(sets), 2.1))
    axes = np.atleast_1d(axes)
    col = 0
    for key, lab in sets:
        red, _, _, _ = C.build(key)
        for pid, rr in red.reducers.items():
            t = rr.data.tags or {}
            if not t:
                continue
            c, n = np.asarray(t.get("corrs")), np.asarray(t.get("noises"))
            cn = np.asarray(t.get("coronoise", n))
            ax = axes[col]
            sc = ax.scatter(c, n, c=cn, s=6, cmap="viridis")
            ax.set_xlabel(r"correlation $\rho$")
            ax.set_ylabel("noise / median" if col == 0 else "")
            ax.set_title(f"{lab}\n{pid}", fontsize=6.5)
            ax.grid(alpha=.25)
            fig.colorbar(sc, ax=ax, pad=0.02).set_label("inner residual", fontsize=6)
            col += 1
            if col >= len(axes):
                break
        if col >= len(axes):
            break
    for ax in axes[col:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f11_frametags.pdf"))
    plt.close(fig)
    print("  f11_frametags.pdf")


# --------------------------------------------------------------------- f12 param compare
def fig_paramcompare(s, key="A2"):
    """Three configurations, the same data, the same injections."""
    from klip_tpe.metrics import Source
    if key not in s:
        return
    r = s[key]
    red, space, obj, _ = C.build(key)
    a = r["annuli"][min(1, len(r["annuli"]) - 1)]
    run = os.path.join(OUT, r["run"])
    recs = [json.loads(l) for l in open(os.path.join(run, "results.jsonl"))
            if json.loads(l).get("annulus") == a["annulus"]]
    ok = [x for x in recs if x.get("score") is not None]
    worst = min(ok, key=lambda x: x["score"])
    cfgs = [("a poor draw", worst["config"]["params"]),
            ("the seeded default", a["default_params"]),
            ("the validated optimum", a["winner_params"])]
    c = float(a["contrast"])
    rho = 0.5 * (a["inrad_as"] + a["outrad_as"])
    # The positions come from the framework's own sampler, with the real companion (and,
    # for beta Pic, the disk sectors) declared -- so the two injections keep the 1.5-FWHM
    # distance from the companion that every scored evaluation keeps.  Hard-wired azimuths
    # did not: 215 deg put one source 0.5 FWHM from beta Pic b, i.e. on top of the planet,
    # which is exactly the contamination Section "Known sources distort ..." measures.
    from klip_tpe.metrics import MawetPeakSNR
    from klip_tpe.positions import PositionSampler
    fpa = R.bp_disk(red)[0] if key in ("A", "A2", "B", "B2") else ()
    samp = PositionSampler(fwhm_as=red.fwhm * red.pxscale, known=[tuple(r["planet"])],
                           forbidden_pa=list(fpa))
    # One draw of two positions is not representative: the same configuration scores a
    # factor ~1.4 differently between draws, so a single pair could make all three panels
    # look better or worse than they typically are.  Draw eight admissible pairs, score the
    # SEEDED DEFAULT on each, and keep the pair that lands at the median of those eight --
    # a typical realization of this geometry rather than a lucky or an unlucky one.  The
    # choice is made on the default alone; the other two configurations then see exactly the
    # same positions.  (Both sources sit at the annulus' mid-radius so that the three panels
    # are comparable, which is a slightly harder place than the band-averaged draws the
    # search itself scores on, so these numbers run a little below Table 2's.)
    _mm = obj.metric          # the run's own metric: the companion is out of its noise ring
    _p0 = dict(a["default_params"], inrad=a["inrad_px"], outrad=a["outrad_px"])
    _p0.pop("width", None)
    cands = [samp.sample(2, rho, rho, np.random.default_rng(seed), c) for seed in range(8)]
    meds = []
    for cand in cands:
        im = red.reduce(ReductionRequest(params=_p0, injections=cand)).image
        meds.append(float(np.nanmedian(_mm.per_source(np.asarray(im, float), None,
                                                      [x.rho for x in cand], [x.theta for x in cand]))))
    srcs = cands[int(np.argsort(meds)[len(meds) // 2])]
    print(f"    (f12: default recovery over 8 admissible pairs "
          f"{np.min(meds):.1f}-{np.max(meds):.1f}, median {np.median(meds):.1f}; "
          f"using PAs " + ", ".join(f"{x.theta:.0f}" for x in srcs) + " deg)")
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.8))
    for j, (lab, prm) in enumerate(cfgs):
        prm = dict(prm, inrad=a["inrad_px"], outrad=a["outrad_px"])
        prm.pop("width", None)
        img = red.reduce(ReductionRequest(params=prm, injections=srcs)).image
        v = img[np.isfinite(img)]
        sg = 1.4826 * np.median(np.abs(v - np.median(v)))
        axes[0, j].set_facecolor("white")
        axes[0, j].imshow(img, origin="lower", **_img_kw(-2 * sg, 6 * sg))
        axes[0, j].set_title(f"{lab}\n$k$={prm.get('k_klip')}, $b$={prm.get('bin')}, "
                             f"$f$={prm.get('filter')}", fontsize=6.5)
        m = snr_map(img, red.fwhm, known=[tuple(r["planet"])], px=red.pxscale)
        axes[1, j].set_facecolor("white")
        axes[1, j].imshow(m, origin="lower", **_img_kw(-3, 8, snr=True))
        from klip_tpe.metrics import source_xy, star_center
        cx, cy = star_center(img.shape)
        xs, ys = source_xy([s_.rho for s_ in srcs], [s_.theta for s_ in srcs],
                           red.pxscale, cx, cy)
        px_, py_ = source_xy([r["planet"][0]], [r["planet"][1]], red.pxscale, cx, cy)
        sn = _mm.per_source(img, None, [s_.rho for s_ in srcs], [s_.theta for s_ in srcs])
        for x, y, v_ in zip(xs, ys, sn):
            axes[1, j].add_patch(Circle((x, y), 1.6 * red.fwhm, fill=False, ec="#0b8043", lw=0.9))
            axes[1, j].text(x, y + 2.2 * red.fwhm, f"{v_:.1f}", color="#0b8043", fontsize=6.5,
                            ha="center")
        # the real companion, marked but never injected on and never scored
        for ax_ in (axes[0, j], axes[1, j]):
            ax_.add_patch(Circle((px_[0], py_[0]), 1.6 * red.fwhm, fill=False, ec=PLANET_EC,
                                 lw=0.9, ls=(0, (3, 2))))
        axes[0, j].text(px_[0], py_[0] + 2.2 * red.fwhm, "b", color=PLANET_EC, fontsize=6.5,
                        ha="center")
        c0 = (img.shape[1] - 1) / 2.0
        h = 1.12 * a["outrad_px"]
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xlim(c0 - h, c0 + h); ax.set_ylim(c0 - h, c0 + h)
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("KLIP image", fontsize=7)
    axes[1, 0].set_ylabel("matched-filter S/N", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f12_paramcompare.pdf"))
    plt.close(fig)
    print("  f12_paramcompare.pdf")


def fig_bench():
    """The benchmark at both dimensionalities: convergence (upward-biased search score)
    and what survives validation."""
    import glob
    from klip_tpe.bench import bench_convergence, read_summary, summarize_bench
    # first directory that has rows; see the naming warning on collect.BENCH_DIRS -- the
    # *_unpaired_20260913/ archives are the ones that used the reference objective
    def _pick(*subs):
        for sub in subs:
            if os.path.exists(os.path.join(OUT, sub, "bench_summary.txt")):
                return os.path.join(OUT, sub)
        return os.path.join(OUT, subs[0])

    # ordered by searched dimension, so the ladder reads down the page.  The first two are
    # beta Pic, whose debris disk crosses the annulus they search; the last two are fields
    # with no scattered-light disk, which is what makes the trend a trend and not a target.
    sets = [(_pick("H2_bench_jwst_pyklip"), "HIP 65426, JWST/NIRCam\n5 searched dimensions (both rolls, searched mode + maxnumbasis)"),
            (_pick("E2_bench", "E_bench"), "beta Pic, VLT/NACO L'\n9 searched dimensions (one sequence)"),
            (_pick("G2_bench_sphere"), "HD 95086, SPHERE/IRDIS\n20 searched dimensions (K1 + K2)"),
            (_pick("F2_bench_highdim", "F_bench_highdim"), "beta Pic, VLT/NACO L'\n38 searched dimensions (four time groups)")]
    sets = [(d, lab) for d, lab in sets if os.path.exists(os.path.join(d, "bench_summary.txt"))]
    if not sets:
        return
    col = {"tpe": "#1f77b4", "random": "#ff7f0e", "grid": "#2ca02c"}
    # four rows at the two-row height would not fit a page with its caption
    row_h = 2.6 if len(sets) <= 2 else 2.05
    fig, axes = plt.subplots(len(sets), 2, figsize=(7.1, row_h * len(sets)),
                             gridspec_kw={"width_ratios": [2, 1]}, squeeze=False)
    summ_all = {}
    for row, (d, lab) in enumerate(sets):
        curves = bench_convergence(sorted(glob.glob(os.path.join(d, "bench_*_*_s*"))))
        rows = read_summary(d)
        summ = summarize_bench(d, log=lambda m: None)
        summ_all[os.path.basename(d)] = summ
        ax = axes[row]
        for m, c in curves.items():
            A = np.asarray(c["curves"], float)
            x = np.asarray(c["n"], float)
            ax[0].plot(x, np.nanmean(A, axis=0), "-", color=col.get(m), lw=1.4,
                       label=f"{m} ({A.shape[0]} seeds)")
            if A.shape[0] > 1:
                ax[0].fill_between(x, np.nanmin(A, axis=0), np.nanmax(A, axis=0),
                                   color=col.get(m), alpha=.15, lw=0)
        dflt = float(np.nanmedian([r["seeded_default_score"] for r in rows])) if rows else np.nan
        if np.isfinite(dflt):
            ax[0].axhline(dflt, color="0.35", ls="--", lw=0.9, zorder=0, label="seeded default")
        ax[0].set_xlabel("evaluation")
        ax[0].set_ylabel("running-best search score")
        ax[0].set_title(f"{lab}", fontsize=7.5)
        ax[0].legend(frameon=False, fontsize=6.5, loc="lower right")
        ax[0].grid(alpha=.25)

        ms = list(summ["modes"])
        v = [summ["modes"][m]["validated_mean"] for m in ms]
        e = [summ["modes"][m]["validated_sd"] for m in ms]
        sr = [summ["modes"][m]["search_mean"] for m in ms]
        x = np.arange(len(ms))
        ax[1].bar(x - 0.18, sr, 0.34, color="0.82", label="search best")
        ax[1].bar(x + 0.18, v, 0.34, yerr=e, capsize=2,
                  color=[col.get(m, "0.5") for m in ms], label="validated")
        if np.isfinite(dflt):
            ax[1].axhline(dflt, color="0.35", ls="--", lw=0.9)
        for xi, (a_, b_) in enumerate(zip(sr, v)):
            ax[1].annotate(f"{a_ - b_:+.1f}", (xi, max(a_, b_)), textcoords="offset points",
                           xytext=(0, 3), ha="center", fontsize=6)
        ax[1].set_xticks(x); ax[1].set_xticklabels(ms)
        ax[1].set_ylabel("median injected S/N")
        ax[1].set_title("search vs validated (gap = winner's curse)", fontsize=7)
        if row == 0:
            ax[1].legend(frameon=False, fontsize=6.5)
        ax[1].grid(alpha=.25, axis="y")
    json.dump(summ_all, open(os.path.join(OUT, "bench_summary.json"), "w"), indent=1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f7_bench.pdf"))
    plt.close(fig)
    print("  f7_bench.pdf")


if __name__ == "__main__":
    s = summary()
    print(f"figures -> {FIG}")
    for fn in (fig_gallery, fig_trace, fig_contrast, fig_partition, fig_paramverify,
               fig_stitch, fig_frametags, fig_paramcompare, fig_bench, copies):
        try:
            fn() if fn is fig_bench else fn(s)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  !! {fn.__name__}: {exc!r}")
