"""Diagrams of the noise apertures each phase of a run measures (``scripts/plot_noise_rings.py``).

:class:`klip_tpe.metrics.RingRecorder` keeps every noise ring the code measures while it is
active, one record per source: the ring's apertures and what became of each.  This module names
the phase a record came from, from the call stack (:func:`phase_from_stack`), and draws one
panel per phase (:func:`plot_rings`).  Nothing here measures anything: the apertures are the
ones the measuring code took.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["phase_from_stack", "plot_rings", "PHASE_ORDER", "STATUS_STYLE"]

#: (module, function) -> phase.  The innermost frame on the stack that names one wins, so the
#: final round inside validation is the final round, and the k-scan inside the calibration is
#: the k-scan.  Helpers (``evaluate``, the objective) are no phase of their own.
_OWNERS: Dict[Tuple[str, str], str] = {
    ("runner", "_ring_survives"): "source count check",
    ("runner", "_ring_clean"): "source count check",
    ("runner", "_grid_snr"): "contrast grid",
    ("runner", "_contrast_grid"): "contrast grid",
    ("runner", "_final_round"): "validation, final round",
    ("runner", "_rescore_winner"): "validation, winner re-scored",
    ("runner", "validate"): "validation",
    ("runner", "_scan_k"): "calibration k-scan",
    ("runner", "calibrate"): "calibration",
    ("runner", "_reset_history"): "search",
    ("runner", "_search_annulus"): "search",
    ("runner", "_restore_best_images"): "resume",
    ("runner", "_fm_pass"): "KLIP-FM",
    ("runner", "_param_verify_hook"): "param_verify",
    ("param_verify", "param_verify_cubes"): "param_verify",
    ("runner", "_verify_hook"): "verify",
    ("verify", "verify_injections"): "verify, injections",
    ("verify", "verify_candidates"): "verify, candidates",
    ("verify", "solved_contrasts"): "verify, solved contrasts",
    ("verify", "bootstrap_nights"): "verify, bootstrap",
    ("runner", "_candidates_hook"): "candidate search",
    ("candidates", "find_candidates"): "candidate search",
    ("runner", "_running_stitch"): "S/N map, running stitch",
    ("runner", "_final_stitch"): "S/N map, final stitch",
    ("runner", "rereduce"): "rereduce",
}

#: the panels' order; a phase not listed comes after these
PHASE_ORDER = ["calibration", "calibration k-scan", "search", "validation", "validation, final round",
               "validation, winner re-scored", "contrast grid", "param_verify", "verify, injections",
               "verify, candidates", "verify, solved contrasts", "candidate search", "S/N map, final stitch",
               "S/N map, running stitch"]


def phase_from_stack(skip: int = 2, depth: int = 80) -> Optional[str]:
    """The phase of the measurement under way, from the call stack (:data:`_OWNERS`).  A
    closure counts for the function it was defined in (``co_qualname``, Python 3.11 on), so a
    measurement on a worker thread is still named."""
    try:
        f = sys._getframe(skip)
    except ValueError:
        return None
    n = 0
    while f is not None and n < depth:
        code = f.f_code
        mod = os.path.splitext(os.path.basename(code.co_filename))[0]
        hit = _OWNERS.get((mod, code.co_name))
        if hit is None:
            qual = getattr(code, "co_qualname", "") or ""
            for part in reversed([p for p in qual.split(".") if p != "<locals>"][:-1]):
                hit = _OWNERS.get((mod, part))
                if hit is not None:
                    break
        if hit is not None:
            return hit
        f = f.f_back
        n += 1
    return None


#: what became of a ring aperture -> (label, face colour, edge colour, line style, filled).
#: Okabe-Ito colours, and fill and line style besides, so no status is told by colour alone.
STATUS_STYLE: Dict[str, Tuple[str, str, str, str, bool]] = {
    "used": ("noise aperture used", "#0072B2", "#0072B2", "-", True),
    "used, known": ("used: a companion's, fewer than 4 others", "#0072B2", "#CC79A7", "-", True),
    "unused": ("on the ring, not used (radial band)", "none", "#0072B2", ":", False),
    "source": ("skipped: within 1.5 FWHM of a source", "none", "#E69F00", "--", False),
    "known": ("skipped: within 1.5 FWHM of a companion", "none", "#CC79A7", "--", False),
    "excluded": ("skipped: another source or companion", "none", "#CC79A7", "--", False),
    "arc": ("skipped: in a sidelobe arc", "none", "#D55E00", "-", False),
    "masked": ("skipped: on the pixel mask", "none", "#009E73", "-", False),
    "off": ("skipped: off the data", "none", "#9a9a9a", ":", False),
}


def _label(rec: Dict[str, Any]) -> str:
    ph = rec.get("phase") or "other"
    if rec.get("role") == "clean":
        ph += ", clean image"
    if rec.get("fn") == "stim_fap":
        ph += ", STIM map"
    elif rec.get("fn") == "aperture_mawet_snr":
        ph += ", aperture photometry"
    if rec.get("noise_image") == "clean":
        ph += " (noise on the clean ring)"
    return ph


def _order_key(label: str) -> Tuple[int, str]:
    base = label.split(", clean image")[0].split(", STIM map")[0].split(", aperture")[0].split(" (noise")[0]
    return (PHASE_ORDER.index(base) if base in PHASE_ORDER else len(PHASE_ORDER), label)


def pick_records(records: Sequence[Dict[str, Any]], prefer_r: Optional[float] = None,
                 skip_phases: Sequence[str] = ("source count check", "resume", "KLIP-FM")) -> List[Dict[str, Any]]:
    """One record per phase (and image), in :data:`PHASE_ORDER`: the one measured last among
    those with the most sources, or for the S/N maps the last one with sources left out, and
    for the contrast grid the innermost position.  ``prefer_r`` (px): of the S/N maps' bands,
    the one at that radius."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        if (rec.get("phase") or "other") in skip_phases:
            continue
        groups.setdefault(_label(rec), []).append(rec)
    out = []
    for label in sorted(groups, key=_order_key):
        g = groups[label]
        if g[0].get("fn") == "snr_map":
            with_src = [r for r in g if r.get("sources_xy")]
            out.append(dict((with_src or g)[-1], _label=label))
            continue
        ok = [r for r in g if r.get("noise")] or g
        if (label.startswith("contrast grid")):
            rmin = min(float(r.get("r", np.inf)) for r in ok)
            ok = [r for r in ok if float(r.get("r", np.inf)) <= rmin + 0.5]
        most = max(len(r.get("all_sources_xy") or r.get("sources_xy") or []) for r in ok)
        ok = [r for r in ok if len(r.get("all_sources_xy") or r.get("sources_xy") or []) == most]
        out.append(dict(ok[-1], _label=label))
    return out


def _arc_polygon(arc, pxscale: float, fwhm: float, cx: float, cy: float, convention: str) -> np.ndarray:
    from .metrics import source_xy
    rho, theta, hw, rad = (float(v) for v in arc)
    r = rho / pxscale
    r1, r2 = max(r - rad * fwhm, 0.0), r + rad * fwhm
    th = np.linspace(theta - hw, theta + hw, 120)
    xo, yo = source_xy(np.full(th.size, r2 * pxscale), th, pxscale, cx, cy, convention)
    xi, yi = source_xy(np.full(th.size, r1 * pxscale), th[::-1], pxscale, cx, cy, convention)
    return np.column_stack([np.r_[xo, xi], np.r_[yo, yi]])


def _summary(rec: Dict[str, Any], n_other: int = 0) -> str:
    fn = rec.get("fn")
    ring = rec.get("ring") or []
    used = sum(1 for a in ring if str(a[2]).startswith("used"))
    n_src = len(rec.get("all_sources_xy") or rec.get("sources_xy") or [])
    src = f"{n_src} source{'s' * (n_src != 1)}"
    if n_other:
        src = f"1 of {n_other + 1} sources measured"
    if fn == "stim_fap":
        return f"{int(rec.get('nclean', 0))} px within 1 FWHM of r, {src}"
    if fn == "snr_map":
        return "each radius: a 1 px band, the sources left out"
    if rec.get("phase") == "verify, bootstrap":
        return "the peak alone over resampled nights, no ring"
    if rec.get("why"):
        return f"no S/N: {rec['why']}"
    noise = rec.get("noise")
    how = (f"{used} of {len(ring)} ring apertures" if noise == "ring" else
           "radial band (ring too short)" if noise == "band" else
           f"{used} apertures, short ring" if noise else "no noise estimate")
    return f"{how}, {src}"


def _notes(rec: Dict[str, Any]) -> List[str]:
    """The fallback and the band floor, for the panel's last line."""
    out = []
    if rec.get("fallback"):
        out.append(f"fallback {rec['fallback']}")
    elif rec.get("fallback_tried"):
        out.append(f"fallback {rec['fallback_tried']} failed")
    if rec.get("band_floor"):
        out.append("band floor")
    return out


def plot_rings(records: Sequence[Dict[str, Any]], path: str, title: str, *, pxscale: float,
               angle_convention: str = "pa", zones: Sequence[Tuple[float, float]] = (),
               pixel_mask: Optional[np.ndarray] = None, extra: Sequence[Dict[str, Any]] = (),
               ncols: int = 4, prefer_r: Optional[float] = None,
               known: Sequence[Tuple[float, float]] = ()) -> str:
    """One panel per phase of ``records`` (:func:`pick_records`), and the ``extra`` records
    after them (each with its own ``_label``), written to ``path``.  ``zones``: the annuli's
    edges (px) as dashed circles.  ``pixel_mask``: drawn under the apertures.  ``known``: the
    known companions ``(rho_arcsec, theta_deg)``, so the verify's left-out positions are told
    apart from the other injected sources."""
    from .metrics import source_xy
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch, Polygon

    picks = pick_records(records, prefer_r=prefer_r) + [dict(r) for r in extra]
    n = len(picks)
    if n == 0:
        raise ValueError("plot_rings: no records to draw")
    ncols = max(1, min(int(ncols), n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 4.35 * nrows + 1.3), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, rec in zip(axes.ravel(), picks):
        ny, nx = rec["shape"]
        cx, cy = rec["center"]
        fwhm = float(rec["fwhm"])
        exr = float(rec.get("exr", 1.5 * fwhm))
        r = float(rec.get("r", np.nan))
        if rec.get("fn") == "snr_map":
            sx = rec.get("sources_xy") or []
            r = float(np.hypot(sx[0][0] - cx, sx[0][1] - cy)) if sx else float(prefer_r or (zones[0][1] if zones else 10))
            r = float(np.round(r))
        half = (r if np.isfinite(r) else max(zones[-1]) if zones else 20.0) + 3.0 * fwhm
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.set_facecolor("#f7f7f7")
        if pixel_mask is not None and np.shape(pixel_mask) == (ny, nx):
            m = np.ma.masked_where(~np.asarray(pixel_mask, bool), np.ones((ny, nx)))
            ax.imshow(m, origin="lower", cmap=matplotlib.colors.ListedColormap(["#cfe8dc"]), alpha=0.9,
                      extent=(-0.5, nx - 0.5, -0.5, ny - 0.5), interpolation="nearest")
        if rec.get("fn") == "stim_fap" and rec.get("mask") is not None:
            mm = np.asarray(rec["mask"], bool)
            ax.imshow(np.ma.masked_where(~mm, np.ones(mm.shape)), origin="lower",
                      cmap=matplotlib.colors.ListedColormap(["#0072B2"]), alpha=0.45,
                      extent=(-0.5, nx - 0.5, -0.5, ny - 0.5), interpolation="nearest")
        if rec.get("fn") == "snr_map" and rec.get("keep") is not None:
            yy, xx = np.mgrid[0:ny, 0:nx]
            band = np.abs(np.hypot(xx - cx, yy - cy) - r) <= 0.5
            keep = np.asarray(rec["keep"], bool)
            img = np.full((ny, nx), np.nan)
            img[band & keep] = 1.0
            img[band & ~keep & np.asarray(rec.get("finite", keep), bool)] = 0.0
            ax.imshow(np.ma.masked_invalid(img), origin="lower",
                      cmap=matplotlib.colors.ListedColormap(["#E69F00", "#0072B2"]), vmin=0, vmax=1, alpha=0.6,
                      extent=(-0.5, nx - 0.5, -0.5, ny - 0.5), interpolation="nearest")
        for a, b in zones:
            for rr in (a, b):
                ax.add_patch(Circle((cx, cy), rr, fill=False, ec="#b0b0b0", lw=0.8, ls=(0, (4, 3))))
        for arc in rec.get("arcs") or []:
            try:
                poly = _arc_polygon(arc, pxscale, fwhm, cx, cy, angle_convention)
                ax.add_patch(Polygon(poly, closed=True, fc="#D55E00", ec="none", alpha=0.12))
            except Exception:                 # noqa: BLE001 - a drawing, never a failure
                pass
        if rec.get("noise") == "band" and np.isfinite(r) and rec.get("fn") in ("mawet_peak_snr",):
            bw = float(rec.get("band_fwhm", 0.6)) * fwhm
            for rr in (r - bw, r + bw):
                ax.add_patch(Circle((cx, cy), max(rr, 0.1), fill=False, ec="#56B4E9", lw=1.2, ls="-"))
        if np.isfinite(r) and rec.get("fn") not in ("snr_map",):
            ax.add_patch(Circle((cx, cy), r, fill=False, ec="#c8c8c8", lw=0.7))
        aprad = float(rec.get("aperture_px", fwhm / 2.0))
        bootstrap = rec.get("phase") == "verify, bootstrap"
        for x, y, what in rec.get("ring") or []:
            what = "unused" if bootstrap and str(what).startswith("used") else what
            _, fc, ec, ls, filled = STATUS_STYLE.get(str(what), STATUS_STYLE["off"])
            ax.add_patch(Circle((x, y), aprad, fc=fc if filled else "none", ec=ec, lw=1.1, ls=ls,
                                alpha=0.75 if filled else 1.0))
        srcs = list(rec.get("all_sources_xy") or rec.get("sources_xy") or [])
        # the known companions in this image's pixels; the verify's other left-out positions
        # (snr_fap's exclude_xy, kept as known_xy) are the other injected sources
        kpx = []
        if len(known):
            kx_, ky_ = source_xy([k[0] for k in known], [k[1] for k in known], pxscale, cx, cy, angle_convention)
            kpx = list(zip(kx_, ky_))
        comp, others = [], []
        for kx, ky in rec.get("known_xy") or []:
            is_known = (not rec.get("excluded_label")) or any(np.hypot(kx - a, ky - b) <= 1.5 for a, b in kpx)
            (comp if is_known else others).append((kx, ky))
        for sx, sy in srcs + others:
            ax.add_patch(Circle((sx, sy), exr, fill=False, ec="#E69F00", lw=0.7, ls=":"))
            ax.add_patch(Circle((sx, sy), aprad, fc="#E69F00", ec="#8a5a00", lw=0.8, alpha=0.9))
        if np.isfinite(rec.get("x", np.nan)) and rec.get("fn") != "snr_map":
            ax.add_patch(Circle((rec["x"], rec["y"]), aprad * 1.35, fill=False, ec="#222222", lw=1.4))
        for kx, ky in comp:
            ax.plot([kx], [ky], marker="*", ms=11, mfc="#CC79A7", mec="#5b2a4a", mew=0.6, ls="none")
            ax.add_patch(Circle((kx, ky), exr, fill=False, ec="#CC79A7", lw=0.7, ls=":"))
        ax.plot([cx], [cy], marker="+", ms=8, color="#444444", mew=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        last = ([f'r = {r * pxscale:.2f}"'] if np.isfinite(r) else []) + _notes(rec)
        ax.set_title(f"{rec['_label']}\n{_summary(rec, len(others))}\n" + ", ".join(last),
                     fontsize=8.2, color="#222222", loc="left")
    handles = [Line2D([], [], marker="o", ls="none", ms=9, mfc=fc if filled else "none", mec=ec, mew=1.2,
                      label=lab, alpha=0.8 if filled else 1.0)
               for key, (lab, fc, ec, ls, filled) in STATUS_STYLE.items() if key not in ("used, known",)]
    handles += [Line2D([], [], marker="o", ls="none", ms=9, mfc="#E69F00", mec="#8a5a00", label="injected source"),
                Line2D([], [], marker="o", ls="none", ms=12, mfc="none", mec="#222222", mew=1.4,
                       label="the source this ring measures"),
                Line2D([], [], marker="*", ls="none", ms=11, mfc="#CC79A7", mec="#5b2a4a", label="known companion"),
                Patch(fc="#D55E00", alpha=0.12, label="sidelobe arc left out"),
                Patch(fc="#cfe8dc", label="pixel mask (disk)"),
                Line2D([], [], color="#56B4E9", lw=1.2, label="radial band (noise when the ring is short)"),
                Line2D([], [], color="#b0b0b0", lw=0.8, ls=(0, (4, 3)), label="annulus edges")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.4, frameon=False)
    fig.suptitle(title, fontsize=11, x=0.01, ha="left", color="#222222")
    fig.tight_layout(rect=(0, 1.25 / (4.35 * nrows + 1.3), 1, 0.975), h_pad=2.2)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
