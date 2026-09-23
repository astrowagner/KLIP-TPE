"""Matplotlib diagnostics driven by a run directory (the logged history only; nothing
here talks to a reducer).  Inputs: ``results.jsonl``, ``run_setup.json``,
``final_results.json``, ``annulusNN/winner.json``, ``annulusNN/validation.json``.

Every ``plot_*`` returns the :class:`matplotlib.figure.Figure` and, with
``save=True``, writes a PNG under ``run_dir/plots/``.  Two conventions from the brief
are enforced in the labelling: the **search score is upward-biased** (best of many
noisy draws) and is never presented as the result -- the validated winner (raw metric
on fresh injections) is what is marked as such; and axes are labelled with the actual
metric name, never a generic "S/N map".
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

__all__ = ["plot_trace", "plot_param_hist", "plot_partition_snr", "plot_contrast_curve", "plot_validation",
           "plot_testbed", "plot_bench_convergence", "plot_all", "load_run"]

# Okabe-Ito (colour-blind safe): random draws grey, guided blue, local orange, explore green
PHASE_COLORS = {"seed": "#000000", "warmup": "#999999", "random": "#999999", "tpe": "#0072B2",
                "local": "#E69F00", "explore": "#009E73", "grid": "#CC79A7", "resume": "#56B4E9"}
PHASE_ORDER = ["seed", "warmup", "random", "grid", "tpe", "local", "explore", "resume"]
MODE_COLORS = {"tpe": "#0072B2", "random": "#999999", "grid": "#CC79A7"}
SEARCH_LABEL = "search score (upward-biased)"
VALID_LABEL = "validated (raw metric, fresh injections)"

_STYLE = {"axes.spines.top": True, "axes.spines.right": True, "axes.grid": False,
          "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "grid.alpha": 0.25,
          "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
          "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 130,
          "pdf.fonttype": 42}          # TrueType embedding: real text in the paper figures (see display._RC)


def _style():
    return matplotlib.rc_context(_STYLE)


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def _load_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_run(run_dir: str) -> Dict[str, Any]:
    """Everything the plots need: ``records`` (all annuli, all restarts), ``setup``,
    ``final``, ``winners`` ``{ia: winner.json}``, ``validation`` ``{ia: table}``."""
    from .bench import read_records
    out: Dict[str, Any] = {"run_dir": run_dir, "records": read_records(run_dir, None, last_segment=False),
                           "setup": _load_json(os.path.join(run_dir, "run_setup.json")),
                           "final": _load_json(os.path.join(run_dir, "final_results.json")),
                           "checkpoint": None, "winners": {}, "validation": {}}
    if out["setup"] is None:
        out["checkpoint"] = _load_json(os.path.join(run_dir, "checkpoint.json"))
    for name in sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else []:
        if name.startswith("annulus") and os.path.isdir(os.path.join(run_dir, name)):
            try:
                ia = int(name[7:]) - 1
            except ValueError:
                continue
            w = _load_json(os.path.join(run_dir, name, "winner.json"))
            v = _load_json(os.path.join(run_dir, name, "validation.json"))
            if w is not None:
                out["winners"][ia] = w
            if v is not None:
                out["validation"][ia] = v
            elif w is not None and w.get("validation_table"):
                out["validation"][ia] = w["validation_table"]
    return out


def _config(run: Dict[str, Any]) -> Dict[str, Any]:
    src = run.get("setup") or run.get("checkpoint") or {}
    return src.get("config", {})


def _space(run: Dict[str, Any]) -> Dict[str, Any]:
    src = run.get("setup") or run.get("checkpoint") or {}
    return src.get("space", {"params": []})


def _metric_name(run: Dict[str, Any]) -> str:
    st = run.get("setup") or {}
    try:
        m = st["objective"]["metric"]["name"]
        cs = st["objective"].get("clean_subtract")
        return f"{m}" + (" (clean-subtracted)" if cs else "")
    except Exception:
        return "metric"


def _annulus_records(run: Dict[str, Any], ia: int) -> List[Dict[str, Any]]:
    recs = [r for r in run["records"] if int(r.get("annulus", 0)) == ia]
    starts = [i for i, r in enumerate(recs) if int(r.get("index", 0)) == 0]
    return recs[starts[-1]:] if starts else recs


def _annuli(run: Dict[str, Any]) -> List[int]:
    return sorted({int(r.get("annulus", 0)) for r in run["records"]} | set(run["winners"]))


def _save(fig: Figure, run_dir: str, name: str, save: bool) -> Optional[str]:
    if not save:
        return None
    d = os.path.join(run_dir, "plots")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    fig.savefig(p, bbox_inches="tight")
    return p


def _score(r: Dict[str, Any]) -> float:
    s = r.get("score")
    return np.nan if s is None else float(s)


def draw_scores_of(recs) -> List[Optional[List[float]]]:
    """Per-record list of the individual draw scores, or None where a record has none.

    Reads ``meta['draw_scores']``, written when ``RunConfig.n_remeasure > 1``.  Records
    from a single-draw run simply have nothing here, which is how the callers know not to
    draw a range they cannot measure.
    """
    out: List[Optional[List[float]]] = []
    for r in recs:
        ds = ((r.get("meta") or {}) if isinstance(r, dict) else (getattr(r, "meta", {}) or {})).get("draw_scores")
        vals = [float(v) for v in (ds or []) if v is not None and np.isfinite(v)]
        out.append(vals if len(vals) > 1 else None)
    return out


def draw_ranges(ax, x, draws, colors=None, color: str = "#8a8a8a", lw: float = 0.9,
                alpha: float = 0.7, zorder: int = 2, label: Optional[str] = None) -> int:
    """Vertical min-to-max bar for every trial that was scored over more than one draw.

    The observed range, not a parametric interval: with three draws a +/- sd would be a
    two-degree-of-freedom estimate dressed up as a confidence band, whereas min-max is
    exactly what was measured.  Drawn under the points (``zorder`` below the scatter), so a
    trial's score still reads as the datum and the bar as its uncertainty.

    ``colors`` gives a colour per trial -- pass the same phase colours the points use and
    each bar belongs to its point rather than to a grey undifferentiated band.  Bars are
    grouped by colour so this stays a handful of ``vlines`` calls however many trials there
    are.  A single ``color`` applies to all of them.

    One legend entry, on a neutral proxy handle: per-colour entries would duplicate the
    phase legend the points already carry, and the entry is there to say what the bars MEAN,
    not which phase any one of them came from.

    Returns the number of bars drawn, so the caller can leave the legend alone when a run
    had a single draw per trial and there is nothing to show.
    """
    xx = np.atleast_1d(x)
    if colors is None:
        cols = [color] * len(xx)
    elif isinstance(colors, str):
        cols = [colors] * len(xx)
    else:
        cols = list(colors) + [color] * max(0, len(xx) - len(list(colors)))
    groups: Dict[Any, Tuple[List[float], List[float], List[float]]] = {}
    ns: List[int] = []
    for xi, vals, c in zip(xx, draws, cols):
        # sanitised here rather than trusted: a failed draw is recorded as None, and the
        # live-display caller passes meta['draw_scores'] through untouched
        v = [float(q) for q in (vals or []) if q is not None and np.isfinite(q)]
        if len(v) < 2:
            continue
        g = groups.setdefault(c, ([], [], []))
        g[0].append(float(xi))
        g[1].append(float(min(v)))
        g[2].append(float(max(v)))
        ns.append(len(v))
    if not ns:
        return 0
    for c, (xs, lo, hi) in groups.items():
        ax.vlines(xs, lo, hi, color=c, lw=lw, alpha=alpha, zorder=zorder)
    if label is None:
        nmode = int(max(set(ns), key=ns.count))
        label = f"draw range (min-max of {nmode})"
    if label:
        from matplotlib.lines import Line2D
        h = Line2D([], [], color="#8a8a8a", lw=lw, alpha=alpha, label=label)
        ax.add_line(h)                      # never drawn (no data), only carried by the legend
    return len(ns)


# ----------------------------------------------------------------------------
# trace
# ----------------------------------------------------------------------------
def plot_trace(run_dir: str, annulus: int = 0, save: bool = True, ax=None) -> Figure:
    """Score vs evaluation coloured by proposal phase, with the running best of the
    (upward-biased) search score and the validated winner marked."""
    run = load_run(run_dir)
    recs = _annulus_records(run, annulus)
    win = run["winners"].get(annulus)
    with _style():
        fig = ax.figure if ax is not None else plt.figure(figsize=(7.5, 4.2))
        ax = ax if ax is not None else fig.add_subplot(111)
        n = np.array([int(r["index"]) + 1 for r in recs])
        y = np.array([_score(r) for r in recs])
        ph = [r.get("phase", "?") for r in recs]
        # under the points, in each point's own phase colour: what its draws spanned
        draw_ranges(ax, n, draw_scores_of(recs),
                    colors=[PHASE_COLORS.get(p, "#333333") for p in ph])
        for p in PHASE_ORDER + sorted(set(ph) - set(PHASE_ORDER)):
            m = np.array([q == p for q in ph])
            if not m.any():
                continue
            ax.scatter(n[m], y[m], s=28 if p == "seed" else 14, marker="*" if p == "seed" else "o",
                       color=PHASE_COLORS.get(p, "#333333"), label=p, zorder=3, alpha=0.9)
        failed = ~np.isfinite(y)
        if failed.any():
            ax.scatter(n[failed], np.full(failed.sum(), np.nanmin(y) if np.isfinite(y).any() else 0), marker="x",
                       color="#d62728", s=18, label="failed", zorder=3)
        if len(y):
            from .bench import running_best
            ax.step(n, running_best(y), where="post", color="#444444", lw=1.0, label="running best (search)")
        if win is not None:
            wi, ws = int(win["winner_index"]) + 1, float(win["winner_score"])
            if win.get("validated"):
                ax.axhline(ws, color="#d62728", ls="--", lw=1.0)
                ax.scatter([wi], [ws], marker="D", s=46, facecolor="none", edgecolor="#d62728", lw=1.4, zorder=4,
                           label=f"winner: {VALID_LABEL} = {ws:.2f}")
                sb = float(win.get("search_best_score", np.nan))
                if np.isfinite(sb):
                    ax.annotate(f"search best {sb:.2f} -> validated {ws:.2f}", xy=(wi, ws), xytext=(6, 8),
                                textcoords="offset points", fontsize=8, color="#d62728")
            else:
                ax.scatter([wi], [ws], marker="D", s=46, facecolor="none", edgecolor="#444444", lw=1.4, zorder=4,
                           label="winner (NOT validated; search score)")
        ax.set_xlabel("evaluation")
        ax.set_ylabel(f"{SEARCH_LABEL}\n{_metric_name(run)}")
        cfg = _config(run)
        ax.set_title(f"{os.path.basename(os.path.abspath(run_dir))}  annulus {annulus + 1}  "
                     f"mode={cfg.get('search_mode', '?')} seed={cfg.get('seed', '?')}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
        _save(fig, run_dir, f"trace_ann{annulus + 1:02d}.png", save)
    return fig


# ----------------------------------------------------------------------------
# parameter histograms
# ----------------------------------------------------------------------------
def _param_groups(space: Dict[str, Any]) -> Dict[str, List[Tuple[int, Dict[str, Any]]]]:
    groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for i, p in enumerate(space.get("params", [])):
        groups.setdefault(p.get("base") or p["name"], []).append((i, p))
    return groups


def _bins_for(p: Dict[str, Any]) -> np.ndarray:
    if p.get("grid"):
        g = np.asarray(p["grid"], float)
        edges = np.concatenate([[g[0] - 0.5 * (g[1] - g[0]) if g.size > 1 else g[0] - 0.5],
                                0.5 * (g[1:] + g[:-1]), [g[-1] + 0.5 * (g[-1] - g[-2]) if g.size > 1 else g[-1] + 0.5]])
        return edges
    lo, hi = float(p["lo"]), float(p["hi"])
    if p.get("kind") in ("int", "categorical"):
        return np.arange(lo - 0.5, hi + 1.5, 1.0)
    return np.linspace(lo, hi, 16)


def plot_param_hist(run_dir: str, annulus: int = 0, gamma: Optional[float] = None, save: bool = True) -> Figure:
    """Per-parameter histograms: all evaluations vs the top-``gamma`` set (the TPE
    "good" set), winner marked.  Replicated per-partition dims are pooled by base name
    (one panel per base; the winner's per-partition values are all marked)."""
    run = load_run(run_dir)
    recs = _annulus_records(run, annulus)
    space = _space(run)
    if gamma is None:
        gamma = float(_config(run).get("gamma", 0.25))
    X = np.array([r["x"] for r in recs], float).reshape(len(recs), -1)
    y = np.array([_score(r) for r in recs])
    ok = np.isfinite(y)
    order = np.argsort(-np.where(ok, y, -np.inf), kind="stable")
    ng = max(int(round(gamma * ok.sum())), 1)
    top = order[:ng]
    win = run["winners"].get(annulus)
    wx = None if win is None else np.asarray(win["winner_x"], float)
    groups = _param_groups(space)
    nb = len(groups)
    ncol = min(4, max(nb, 1))
    nrow = int(np.ceil(nb / ncol)) if nb else 1
    with _style():
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow), squeeze=False)
        for ax in axes.ravel()[nb:]:
            ax.set_visible(False)
        for ax, (base, dims) in zip(axes.ravel(), groups.items()):
            p0 = dims[0][1]
            bins = _bins_for(p0)
            idx = [i for i, _ in dims]
            allv = X[:, idx].ravel() if len(X) else np.zeros(0)
            topv = X[top][:, idx].ravel() if len(X) else np.zeros(0)
            ax.hist(allv, bins=bins, color="#c8c8c8", label="all evaluations", density=True)
            ax.hist(topv, bins=bins, color="#1f77b4", alpha=0.6, label=f"top {gamma:.0%}", density=True)
            if wx is not None:
                for j, i in enumerate(idx):
                    ax.axvline(wx[i], color="#d62728", lw=1.2, label="winner" if j == 0 else None)
            role = p0.get("role", "reduction")
            npart = sum(1 for _, p in dims if p.get("partition") is not None)
            ttl = base + (f"  ({npart} partitions)" if npart > 1 else "") + ("  [selection]" if role == "selection" else "")
            if p0.get("choices"):
                ax.set_xticks(range(len(p0["choices"])))
                ax.set_xticklabels([str(c) for c in p0["choices"]], rotation=30)
            ax.set_title(ttl)
            ax.set_yticks([])
        axes.ravel()[0].legend(loc="upper right")
        fig.suptitle(f"{os.path.basename(os.path.abspath(run_dir))} annulus {annulus + 1}: parameter distributions "
                     f"(top set by {SEARCH_LABEL})", y=1.02)
        fig.tight_layout()
        _save(fig, run_dir, f"param_hist_ann{annulus + 1:02d}.png", save)
    return fig


# ----------------------------------------------------------------------------
# per-partition S/N
# ----------------------------------------------------------------------------
def plot_partition_snr(run_dir: str, annulus: int = 0, save: bool = True) -> Figure:
    """Raw per-partition score of the *included* partitions at every evaluation,
    with the combined raw score for reference (drop-outs show as gaps)."""
    run = load_run(run_dir)
    recs = _annulus_records(run, annulus)
    pids: List[str] = []
    for r in recs:
        for p in (r.get("partition_snr") or {}):
            if p not in pids:
                pids.append(p)
    with _style():
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        n = np.array([int(r["index"]) + 1 for r in recs])
        raw = np.array([np.nan if r.get("raw_score") is None else float(r["raw_score"]) for r in recs])
        ax.plot(n, raw, color="#444444", lw=1.0, alpha=0.6, label="combined (raw score)")
        cmap = plt.get_cmap("tab10")
        for k, pid in enumerate(pids):
            v = np.array([np.nan if (r.get("partition_snr") or {}).get(pid) is None
                          else float(r["partition_snr"][pid]) for r in recs])
            inc = np.array([pid in (r.get("meta", {}).get("selected") or r.get("config", {}).get("selected") or [])
                            for r in recs])
            v = np.where(inc, v, np.nan)
            ax.plot(n, v, "o-", ms=3, lw=0.8, color=cmap(k % 10), label=f"partition {pid} (n={int(inc.sum())} incl.)")
        if not pids:
            ax.text(0.5, 0.5, "no per-partition scores in results.jsonl\n(single-partition reducer?)",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("evaluation")
        ax.set_ylabel(f"raw per-partition score\n{_metric_name(run).replace(' (clean-subtracted)', '')}")
        ax.set_title(f"{os.path.basename(os.path.abspath(run_dir))} annulus {annulus + 1}: per-partition raw S/N of included partitions")
        ax.legend(loc="best", ncol=2)
        _save(fig, run_dir, f"partition_snr_ann{annulus + 1:02d}.png", save)
    return fig


# ----------------------------------------------------------------------------
# contrast curve
# ----------------------------------------------------------------------------
def _curve_xy(cc: Dict[str, Any], rkey: str = "r_as", ckey: str = "curve") -> Tuple[np.ndarray, np.ndarray]:
    r = np.array([np.nan if v is None else v for v in (cc or {}).get(rkey, [])], float)
    c = np.array([np.nan if v is None else v for v in (cc or {}).get(ckey, [])], float)
    n = min(r.size, c.size)
    return r[:n], c[:n]


def plot_contrast_curve(run_dir: str, save: bool = True, nsigma: float = 5.0) -> Figure:
    """Contrast curves of each annulus winner from ``winner.json``:

    * ``contrast_curve`` -- the validated, injection-calibrated ``nsigma`` curve (solid)
      with its per-sample points (``sample_r``/``sample_c``, else derived from
      ``validation_samples`` as ``nsigma * contrast / SNR``);
    * ``fm_curve`` -- the KLIP-FM cross-check curve when present (dashed + markers,
      same colour), so the two calibrations can be compared per annulus.
    """
    run = load_run(run_dir)
    with _style():
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        cmap = plt.get_cmap("tab10")
        any_curve = False
        any_fm = False
        for k, (ia, w) in enumerate(sorted(run["winners"].items())):
            col = cmap(k % 10)
            contrast = float(w.get("contrast") or np.nan)
            cc = w.get("contrast_curve") or {}
            fm = w.get("fm_curve") or {}
            if not cc and not fm:
                continue
            r, c = _curve_xy(cc)
            if r.size and np.isfinite(c).any() and (c[np.isfinite(c)] > 0).any():
                ax.plot(r, np.where(c > 0, c, np.nan), color=col, lw=1.4,
                        label=f"annulus {ia + 1}: validated winner ({nsigma:.0f}-sigma, injection-calibrated)")
                any_curve = True
            sr, sc = _curve_xy(cc, "sample_r", "sample_c")
            if not sr.size and np.isfinite(contrast):                # older winner.json: derive from the samples
                vs = w.get("validation_samples") or {}
                sr, snr = _curve_xy(vs, "r_as", "snr")
                sc = nsigma * contrast / np.where(snr > 0, snr, np.nan)
            if sr.size and np.isfinite(sc).any():
                ax.scatter(sr, sc, s=12, color=col, alpha=0.6, label=f"annulus {ia + 1}: validation samples")
            rf, cf = _curve_xy(fm)
            if rf.size and np.isfinite(cf).any() and (cf[np.isfinite(cf)] > 0).any():
                ax.plot(rf, np.where(cf > 0, cf, np.nan), color=col, lw=1.1, ls="--", marker="o", ms=3,
                        label=f"annulus {ia + 1}: KLIP-FM cross-check")
                any_curve = any_fm = True
            if np.isfinite(contrast) and contrast > 0:
                ax.axhline(contrast, color=col, ls=":", lw=0.8)
        if any_curve:
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "no contrast curve in winner.json", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("separation (arcsec)")
        ax.set_ylabel(f"{nsigma:.0f}-sigma contrast")
        ax.set_title(f"{os.path.basename(os.path.abspath(run_dir))}: contrast curve of the validated winners"
                     + (" vs KLIP-FM cross-check" if any_fm else "") + "\n(dotted: injection contrast)")
        ax.legend(loc="best", fontsize=7)
        _save(fig, run_dir, "contrast_curve.png", save)
    return fig


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def plot_validation(run_dir: str, save: bool = True) -> Figure:
    """Search score vs validated score of every validation candidate (all annuli);
    the vertical bar spans the validation trials, the 1:1 line shows the optimism."""
    run = load_run(run_dir)
    with _style():
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        cmap = plt.get_cmap("tab10")
        allv: List[float] = []
        for k, (ia, table) in enumerate(sorted(run["validation"].items())):
            w = run["winners"].get(ia)
            col = cmap(k % 10)
            for row in table:
                s = row.get("search_score")
                v = row.get("validated_score")
                tr = np.array([np.nan if t is None else t for t in row.get("trials", [])], float)
                if s is None:
                    continue
                if v is None or not np.isfinite(v):
                    ax.scatter([s], [np.nanmin(allv) if allv else 0], marker="x", color="#d62728")
                    continue
                allv += [s, v]
                if np.isfinite(tr).any():
                    allv += [float(np.nanmin(tr)), float(np.nanmax(tr))]
                    ax.plot([s, s], [np.nanmin(tr), np.nanmax(tr)], color=col, lw=0.8, alpha=0.5,
                            label="trial range" if not any(l.get_label() == "trial range" for l in ax.lines) else None)
                is_win = w is not None and w.get("validated") and int(w["winner_index"]) == int(row["eval_index"])
                ax.scatter([s], [v], s=60 if is_win else 24, color=col, edgecolor="#d62728" if is_win else "none",
                           lw=1.5, zorder=3, label=f"annulus {ia + 1}" if row is table[0] else None)
                ax.annotate(f"e{int(row['eval_index']) + 1}", (s, v), xytext=(4, 3), textcoords="offset points", fontsize=7)
        if allv:
            lo, hi = np.nanmin(allv), np.nanmax(allv)
            pad = 0.05 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#888888", ls="--", lw=0.8, label="1:1")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        else:
            ax.text(0.5, 0.5, "no validation table", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(SEARCH_LABEL)
        ax.set_ylabel(f"validated score (median of trials)\n{VALID_LABEL}")
        ax.set_title(f"{os.path.basename(os.path.abspath(run_dir))}: validation (red ring = winner)")
        ax.legend(loc="upper left")
        _save(fig, run_dir, "validation.png", save)
    return fig


# ----------------------------------------------------------------------------
# testbed / benchmark
# ----------------------------------------------------------------------------
def plot_testbed(results: Dict[str, Any], out_path: Optional[str] = None, title: Optional[str] = None) -> Figure:
    """Figure for :mod:`klip_tpe.testbed` outputs.

    * ``compare_density_models`` result (``{mode: {"true_mean", "true_sd", "obs_mean", ...}}``):
      bars of TRUE-at-observed-best (mean +/- sd) with the observed best marked, so the
      optimism (observed - true) is visible per mode.
    * ``{label: run_tpe_on_objective(...)}`` (values with ``best_obs`` / ``true_at_best``):
      the two curves vs iteration per label.
    """
    with _style():
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        first = next(iter(results.values())) if results else {}
        if isinstance(first, dict) and "best_obs" in first:
            cmap = plt.get_cmap("tab10")
            for k, (lab, r) in enumerate(results.items()):
                it = np.arange(1, len(r["best_obs"]) + 1)
                ax.plot(it, r["best_obs"], color=cmap(k % 10), lw=1.0, ls="--", label=f"{lab}: observed best (biased)")
                ax.plot(it, r["true_at_best"], color=cmap(k % 10), lw=1.4, label=f"{lab}: TRUE at observed best")
            ax.set_xlabel("iteration")
            ax.set_ylabel("score (true scale 0-10)")
            ax.legend(loc="lower right")
        else:
            labels = list(results)
            t = np.array([results[m].get("true_mean", np.nan) for m in labels])
            sd = np.array([results[m].get("true_sd", 0.0) for m in labels])
            o = np.array([results[m].get("obs_mean", np.nan) for m in labels])
            xs = np.arange(len(labels))
            ax.bar(xs, t, yerr=sd, color="#1f77b4", alpha=0.8, capsize=3, label="TRUE at observed best (mean +/- sd)")
            ax.scatter(xs, o, marker="_", s=400, color="#d62728", lw=2, label="observed best (upward-biased)", zorder=3)
            for x, m in zip(xs, labels):
                d = results[m].get("diff_vs_ref")
                if d is not None and np.isfinite(d) and x > 0:
                    ax.annotate(f"{d:+.2f} vs {labels[0]}", (x, t[x]), xytext=(0, 6), textcoords="offset points",
                                ha="center", fontsize=8)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
            ax.set_ylabel("score (true scale 0-10)")
            ax.legend(loc="lower right")
        ax.set_title(title or "testbed: density models (paired seeds)")
        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_bench_convergence(curves: Dict[str, Dict[str, Any]], out_path: Optional[str] = None,
                           title: Optional[str] = None, ax=None) -> Figure:
    """Curves from :func:`klip_tpe.bench.bench_convergence`: per-mode mean +/- sd
    (shaded) of the running best **search** score vs evaluation.  Upward-biased by
    construction -- read the validated numbers from ``summarize_bench``."""
    with _style():
        fig = ax.figure if ax is not None else plt.figure(figsize=(7.0, 4.2))
        ax = ax if ax is not None else fig.add_subplot(111)
        cmap = plt.get_cmap("tab10")
        defaults = []
        for k, (mode, c) in enumerate(curves.items()):
            col = MODE_COLORS.get(mode, cmap(k % 10))
            n, m, s = np.asarray(c["n"]), np.asarray(c["mean"], float), np.asarray(c["sd"], float)
            nrun = int(np.asarray(c["curves"]).shape[0])
            ax.plot(n, m, color=col, lw=1.6, label=f"{mode} (mean of {nrun} run{'s' if nrun != 1 else ''})")
            if nrun > 1:
                ax.fill_between(n, m - s, m + s, color=col, alpha=0.18, lw=0)
            if np.isfinite(c.get("default", np.nan)):
                defaults.append(float(c["default"]))
        if defaults:
            ax.axhline(float(np.mean(defaults)), color="#000000", ls=":", lw=0.9, label="seeded default")
        ax.set_xlabel("evaluation (incl. seeded default)")
        ax.set_ylabel(f"running best {SEARCH_LABEL}")
        ax.set_title(title or "matched-budget convergence -- search score, NOT validated")
        ax.legend(loc="lower right")
        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_all(run_dir: str, save: bool = True, close: bool = True, books: bool = True) -> Dict[str, Any]:
    """All per-run diagnostics for every annulus; returns ``{name: Figure}`` plus, with
    ``books=True`` (and ``save``), the per-annulus display books written by
    :mod:`klip_tpe.display` under ``{name: path}`` (corner, landscapes, parhist, edf,
    partition map, k-book, calibration panel)."""
    run = load_run(run_dir)
    figs: Dict[str, Any] = {}
    for ia in _annuli(run):
        figs[f"trace_ann{ia + 1:02d}"] = plot_trace(run_dir, ia, save)
        figs[f"param_hist_ann{ia + 1:02d}"] = plot_param_hist(run_dir, ia, save=save)
        figs[f"partition_snr_ann{ia + 1:02d}"] = plot_partition_snr(run_dir, ia, save)
    figs["contrast_curve"] = plot_contrast_curve(run_dir, save)
    figs["validation"] = plot_validation(run_dir, save)
    if close:
        for f in figs.values():
            if isinstance(f, Figure):
                plt.close(f)
    if books and save:
        from .display import plot_annulus_books, plot_calibration
        for ia in _annuli(run):
            for name, path in plot_annulus_books(run_dir, ia, run).items():
                figs[f"{name}_ann{ia + 1:02d}"] = path
            p = plot_calibration(run_dir, ia, run)
            if p:
                figs[f"calibration_ann{ia + 1:02d}"] = p
    return figs
