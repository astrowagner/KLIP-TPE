"""A live window that cannot freeze: an attachable viewer for a running optimization.

The optimizer's own window (``LiveDisplay(show=True)``) renders off-thread and only
*blits* a finished PNG, but the blit and the GUI event loop still belong to the main
thread -- and the main thread spends most of its life inside NumPy/LAPACK.  On macOS the
WindowServer marks a process that has not serviced events for a couple of seconds as
unresponsive, stops compositing it, and leaves the last painted frame on screen (a
beachball on hover).  The run is fine and the panels keep being written to
``steps/stepNNNN.png``; only the window is stale.

This module is the guarantee.  ``klip-tpe view --run-dir <dir>`` is a *separate process*
whose main thread does nothing but pump the event loop and blit whichever panel is newest
on disk.  It has no computation to block on, so it can never be starved -- and because it
talks to the run only through files it can be attached to, detached from and restarted
around a run that is already going, without touching it.

    klip-tpe view --run-dir runs/run_001          # attach to a running (or finished) run
    klip-tpe view --root /data/NEAR_py            # newest run under <root>/comb/opt

It also reports what the run is doing (evaluation count, rate, the age of the newest
panel), so a genuinely stuck run is distinguishable from a merely stale window.
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["newest_panel", "run_status", "find_run", "view"]

#: panels the viewer will show, newest first
PANEL_GLOBS = ("steps/step*.png", "annulus*/step_display_white.png", "annulus*/*_panel.png")


def find_run(run_dir: Optional[str] = None, root: Optional[str] = None) -> str:
    """The run directory to watch: ``run_dir`` as given, ``last``/``latest`` or a bare run
    name resolved under ``<root>/comb/opt``, or the newest run under ``root``."""
    if run_dir and run_dir not in ("last", "latest") and os.path.isdir(run_dir):
        return os.path.abspath(run_dir)
    cands: List[str] = []
    for base in ([root] if root else []) + ([os.getcwd()] if not root else []):
        if not base:
            continue
        for pat in ("comb/opt/run_*", "runs/run_*", "run_*"):
            cands += [d for d in glob.glob(os.path.join(base, pat)) if os.path.isdir(d)]
    if run_dir and run_dir not in ("last", "latest") and root:
        p = os.path.join(root, "comb", "opt", run_dir)
        if os.path.isdir(p):
            return os.path.abspath(p)
    if not cands:
        raise SystemExit(f"no run directory found (run_dir={run_dir!r}, root={root!r})")
    return os.path.abspath(max(cands, key=os.path.getmtime))


def newest_panel(run_dir: str) -> Optional[Tuple[str, float]]:
    """``(path, mtime)`` of the most recently written panel PNG, or None."""
    best: Optional[Tuple[str, float]] = None
    for pat in PANEL_GLOBS:
        for p in glob.glob(os.path.join(run_dir, pat)):
            try:
                m = os.path.getmtime(p)
            except OSError:
                continue
            if best is None or m > best[1]:
                best = (p, m)
    return best


def run_status(run_dir: str, prev: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What the run is doing, from its files alone: evaluation count, the annulus, the
    age of ``results.jsonl`` and of the newest panel, and a rate in evaluations/minute."""
    now = time.time()
    out: Dict[str, Any] = {"t": now, "n_eval": None, "annulus": None, "results_age": None,
                           "panel_age": None, "rate": None, "alive": None}
    res = os.path.join(run_dir, "results.jsonl")
    if os.path.exists(res):
        out["results_age"] = now - os.path.getmtime(res)
        try:                                    # count without reading the whole file twice
            with open(res, "rb") as f:
                n = sum(1 for _ in f)
            out["n_eval"] = n
            if prev and prev.get("n_eval") is not None and now > prev["t"]:
                dn, dt = n - prev["n_eval"], now - prev["t"]
                if dt > 0:
                    out["rate"] = 60.0 * dn / dt
        except OSError:
            pass
        try:
            with open(res, "rb") as f:          # the last line: which annulus we are in
                f.seek(0, os.SEEK_END)
                back = min(8192, f.tell())
                f.seek(-back, os.SEEK_END)
                last = f.read().splitlines()
            for line in reversed(last):
                try:
                    out["annulus"] = json.loads(line).get("annulus")
                    break
                except Exception:
                    continue
        except OSError:
            pass
    p = newest_panel(run_dir)
    if p:
        out["panel_age"] = now - p[1]
    # "alive" = something was written recently; the panel cadence is much slower than the
    # evaluation cadence, so results.jsonl is the honest signal
    ages = [a for a in (out["results_age"], out["panel_age"]) if a is not None]
    out["alive"] = (min(ages) < 300) if ages else None
    return out


def _fmt_age(s: Optional[float]) -> str:
    if s is None:
        return "--"
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.1f}h"


def status_line(run_dir: str, st: Dict[str, Any]) -> str:
    bits = [os.path.basename(run_dir)]
    if st.get("n_eval") is not None:
        a = st.get("annulus")
        bits.append(f"eval {st['n_eval']}" + (f" (annulus {a + 1})" if a is not None else ""))
    if st.get("rate"):
        bits.append(f"{st['rate']:.1f} eval/min")
    bits.append(f"last write {_fmt_age(st.get('results_age'))}")
    bits.append(f"panel {_fmt_age(st.get('panel_age'))} old")
    if st.get("alive") is False:
        bits.append("NO RECENT WRITES -- the run may have stopped")
    return "   |   ".join(bits)


def view(run_dir: Optional[str] = None, root: Optional[str] = None, interval: float = 1.0,
         scale: float = 1.0, once: bool = False, log=print) -> int:
    """Watch ``run_dir`` and show whichever panel is newest, refreshing every ``interval``
    seconds.  Returns 0 when the window is closed or the user interrupts.

    This call owns the main thread and does nothing but service the GUI and swap in a new
    image, so the window stays responsive no matter what the optimizer is doing.
    """
    d = find_run(run_dir, root)
    log(f"klip-tpe view: watching {d}")
    import matplotlib
    if not once:
        for cand in ([os.environ["KLIP_TPE_BACKEND"]] if os.environ.get("KLIP_TPE_BACKEND")
                     else ["MacOSX", "QtAgg", "TkAgg", "GTK3Agg"]):
            try:
                matplotlib.use(cand, force=True)
                break
            except Exception:
                continue
    matplotlib.rcParams["toolbar"] = "None"
    import matplotlib.pyplot as plt

    cur = newest_panel(d)
    if cur is None:
        log(f"no panel PNG under {d} yet (looked for {', '.join(PANEL_GLOBS)}); waiting...")
    st = run_status(d)
    log("  " + status_line(d, st))
    if once:
        return 0

    img = plt.imread(cur[0]) if cur else None
    h, w = (img.shape[0], img.shape[1]) if img is not None else (990, 1850)
    dpi = 100.0
    fig = plt.figure(figsize=(w / dpi * scale, h / dpi * scale), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(img, interpolation="lanczos" if abs(scale - 1) > 1e-3 else "nearest",
                   aspect="auto", interpolation_stage="rgba") if img is not None else None
    txt = fig.text(0.005, 0.004, "", color="#9adcff", fontsize=7, family="monospace", va="bottom")
    try:
        fig.canvas.manager.set_window_title(f"klip-tpe view -- {os.path.basename(d)}")
    except Exception:
        pass
    plt.ion()
    plt.show(block=False)

    shown = cur[1] if cur else -1.0
    prev = st
    last_status = 0.0
    try:
        while plt.fignum_exists(fig.number):
            p = newest_panel(d)
            if p and p[1] > shown:
                try:
                    a = plt.imread(p[0])
                    if im is None:
                        im = ax.imshow(a, interpolation="nearest", aspect="auto",
                                       interpolation_stage="rgba")
                    elif im.get_array().shape[:2] != a.shape[:2]:
                        ax.clear(); ax.set_axis_off()
                        im = ax.imshow(a, interpolation="nearest", aspect="auto",
                                       interpolation_stage="rgba")
                    else:
                        im.set_data(a)
                    shown = p[1]
                except Exception:
                    pass                        # a half-written PNG: just try again next tick
            now = time.time()
            if now - last_status > 2.0:
                prev = run_status(d, prev if now - prev["t"] > 10 else None) if prev else run_status(d)
                txt.set_text(status_line(d, prev))
                last_status = now
            fig.canvas.draw_idle()
            plt.pause(interval)                 # <- the whole point: the event loop always runs
    except KeyboardInterrupt:
        log("\nklip-tpe view: interrupted")
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass
    return 0
