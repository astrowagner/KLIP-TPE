"""Live and post-hoc display layer -- the matplotlib rebuild of the IDL panels
(``near2m_show``, ``near2m_savepanels``, ``near2m_corner``, ``near2m_pnlandscapes`` /
``pnkde`` / ``pnwalk``, ``near2m_parhist_page``, ``near2m_edf``, ``near2m_nightmap`` /
``npanel`` / ``nnpanel`` / ``nspanel``, ``near2m_kpanel`` / ``kklip_book``,
``near2m_snrhist``, ``near2m_cpanel``, ``near2m_calshow``, the validation display,
``near2m_intro_frame`` and ``near2m_makegif``).

Two entry points share every renderer:

* :class:`LiveDisplay` -- a :class:`~klip_tpe.runner.RunCallback` that draws the step
  panel during a run (``run_dir/steps/stepNNNN.png``), the calibration / validation
  panels, the per-annulus books (``annulusNN/*.pdf``) and, at the end, the
  progress movie (``opt_steps.gif`` / ``.mp4``).
* the ``plot_*`` / ``render_*`` functions -- the same panels rebuilt from a finished
  run directory (``results.jsonl``, ``checkpoint.json`` / ``run_setup.json``,
  ``annulusNN/{winner,validation,calibration}.json``, ``best_*.fits``).

Both paths first normalise the history into an :class:`AnnulusData` so that the
panel code never sees a runner or a file.

Labelling conventions (brief section 8.5-8.6): the search score is an upward-biased
quantity (best of many noisy draws) and is always labelled as such; only the
validated winner (raw metric on fresh injections) is marked as the result; and the
S/N *maps* are a display diagnostic with a different noise convention from the
score -- every map carries that caption.  Nothing here ever raises out of a callback.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
import matplotlib.cm
import matplotlib.lines
import matplotlib.patches
import matplotlib.ticker
from matplotlib import colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from .metrics import radprof, source_xy, star_center
from .plots import (PHASE_COLORS, PHASE_ORDER, SEARCH_LABEL, VALID_LABEL, _annulus_records, _bins_for,
                    _config, _metric_name, _space, load_run)
from .runner import RunCallback

__all__ = ["AnnulusData", "LiveDisplay", "annulus_from_run", "render_step", "render_steps",
           "render_calibration", "render_validation", "render_corner", "render_landscapes",
           "render_parhist_book", "render_edf", "render_nightmap", "render_kbook", "render_intro",
           "plot_annulus_books", "plot_calibration"]

# ----------------------------------------------------------------------------
# style
# ----------------------------------------------------------------------------
SCORE_CMAP = "viridis"          # score-coloured points (IDL cmap_plots)
PANEL_PX = (1850, 990)          # live panel size in pixels = the IDL window (window, 2, xsize=1850, ysize=990)


def panel_size(dpi: float) -> Tuple[float, float]:
    """Figure size in inches that renders to exactly ``PANEL_PX`` at ``dpi``."""
    return (PANEL_PX[0] / float(dpi), PANEL_PX[1] / float(dpi))
SNR_RANGE = (-3.0, 8.0)         # S/N map stretch (IDL: fixed -3..8)
SNR_CAPTION = "S/N map for display -- not the score"
RANDOM_PHASES = ("seed", "warmup", "random", "explore", "grid")
BEST_COLOR = "#000000"
VALID_COLOR = "#D55E00"
KNOWN_COLOR = "#FF2D95"         # real companions the objective was told about (never scored).
#: Magenta on purpose: SRC_COLOR is a spring green and VALID_COLOR a vermillion, and on
#: inferno (black -> purple -> orange -> yellow) a cyan reads as a dimmer green.  A known
#: companion must never be mistaken for an injection, so it gets the one hue nothing else uses.
CUR_COLOR = "#7f7f7f"
SRC_COLOR = "#00e5a0"

#: column-header abbreviations of the per-partition config table (step panel text block)
COL_ABBREV = {"bin": "bin", "n_ang": "nang", "filter": "filt", "angsep": "asep", "anglemax": "amax",
              "corr_thresh": "corr", "noise_max": "noise", "coronoise_max": "coro", "k_klip": "k", "k": "k"}
#: target width (characters) of the text block; wider config tables wrap into column blocks
TEXT_WIDTH = 72


def _abbrev(name: str) -> str:
    """Short column/label name.  The searched reference library's counts are
    ``nkeep_<group>``; a bare six-character cut used to turn ``nkeep_altroll`` and
    ``nkeep_psfref`` into the same ``nkeep_``, so the importance bars, the config table and
    the ``k=v`` lines could not say which pool they meant.  They read ``nkalt`` / ``nkref``
    (the IDL's own ``nkalt``), and any other group ``nk<first four letters>``."""
    if name in COL_ABBREV:
        return COL_ABBREV[name]
    if name.startswith("nkeep_"):
        grp = name[len("nkeep_"):]
        return {"altroll": "nkalt", "psfref": "nkref"}.get(grp, "nk" + grp[:4])
    return name[:6]


def _abbrev_unique(names: Sequence[str]) -> List[str]:
    """:func:`_abbrev` for a set of names shown together, with any collision resolved by
    falling back to the full names of the colliding entries -- two parameters must never
    share a label, whatever a future space calls them."""
    short = [_abbrev(n) for n in names]
    seen: Dict[str, int] = {}
    for s in short:
        seen[s] = seen.get(s, 0) + 1
    return [n if seen[s] > 1 else s for n, s in zip(names, short)]


def _fm_unavailable_note(desc: Optional[Dict[str, Any]]) -> Optional[str]:
    """The KLIP-FM cell's text when the reducer cannot forward-model at all, from its
    ``describe()`` dict (as saved in ``run_setup.json``); None when a model will appear.

    The cell used to read "(after 1st best)" whenever the image was missing, which on a
    pyKLIP / VIP / spaceKLIP run means for ever: KLIP-FM exists only in the built-in engine,
    so the promised model never arrives and the placeholder is still there on the finished
    panel."""
    if not isinstance(desc, dict) or desc.get("supports_fm", True):
        return None
    parts = desc.get("partitions") or {}
    backs = sorted({str(v.get("backend") or v.get("name")) for v in parts.values()
                    if isinstance(v, dict)} - {"None"}) if isinstance(parts, dict) else []
    who = "/".join(backs) or str(desc.get("name") or "this")
    return f"no KLIP-FM with the\n{who} backend\n(built-in engine only)"


_RC = {"axes.spines.top": True, "axes.spines.right": True, "axes.grid": False,
       "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "grid.alpha": 0.22,
       "grid.linewidth": 0.5, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
       "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
       "axes.titlepad": 3, "axes.labelpad": 2, "xtick.major.pad": 2, "ytick.major.pad": 2,
       # Every colour _RC_DARK touches is pinned light here, so the package style is a
       # complete theme rather than a patch on whatever is in rcParams.  The live panel
       # renders its frames on a worker thread inside _rc(idl=True, dark=True), and
       # matplotlib's rc_context mutates the one process-global rcParams dict: while that
       # frame is drawing, a book written from the main thread used to pick up the panel's
       # savefig.facecolor and come out black-on-black (corner.pdf and parhist.pdf, at
       # random, depending on which thread won).  Pinning the light values makes _rc()
       # immune to that -- the dark context still wins for the panel, because _RC_DARK is
       # applied after _RC.
       "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
       "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
       "xtick.color": "black", "ytick.color": "black", "grid.color": "#b0b0b0",
       # TrueType (Type 42) font embedding in the PDFs, not the default Type 3.  Type 3
       # embedding builds a 256-entry cp1252 width table and asks FreeType for the five
       # undefined cp1252 slots, which decode to U+FFFE; matplotlib silences the resulting
       # "Glyph 65534 missing" warning with a warnings.catch_warnings() -- which is
       # process-global, and the reducer's own catch_warnings() on the main thread restores
       # ITS snapshot of the filters while the panel thread is still inside matplotlib's,
       # so the warning leaked out of live runs at random (twice per tutorial notebook).
       # Type 42 never asks for those slots, and the text in the PDFs is real, selectable
       # text besides.
       "pdf.fonttype": 42}


_RC_IDL = {"axes.spines.top": True, "axes.spines.right": True, "axes.grid": False, "legend.frameon": False,
           "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
           "ytick.labelsize": 6.5, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
           "ytick.right": True}
_RC_DARK = {"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
            "text.color": "white", "axes.labelcolor": "white", "axes.edgecolor": "white",
            "xtick.color": "white", "ytick.color": "white", "grid.color": "#888888"}


def _rc(idl: bool = False, dark: bool = False):
    """Matplotlib rc context: the package style, or the IDL live-window look (boxed
    axes, inward ticks; ``dark`` = black background like the IDL X window)."""
    rc = dict(_RC)
    if idl:
        rc.update(_RC_IDL)
    if dark:
        rc.update(_RC_DARK)
    return matplotlib.rc_context(rc)


def _fg() -> str:
    """Foreground colour of the active rc (white on the dark live panel, else black)."""
    return matplotlib.rcParams.get("text.color", "black")


def _is_dark() -> bool:
    return mcolors.to_rgb(matplotlib.rcParams.get("figure.facecolor", "white")) == (0.0, 0.0, 0.0)


def _grey(light: str = "#cccccc", dark: str = "#5a5a5a") -> str:
    return dark if _is_dark() else light


def _in_notebook() -> bool:
    """True inside a Jupyter kernel (ZMQ shell), False in a terminal / plain IPython."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and type(ip).__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _fmt_clock(t: float) -> str:
    """Local wall-clock time of ``t`` (epoch s): ``HH:MM`` today, ``Mon HH:MM`` within a
    week, else ``Sep 14 HH:MM``."""
    lt = time.localtime(t)
    now = time.localtime()
    if lt.tm_yday == now.tm_yday and lt.tm_year == now.tm_year:
        return time.strftime("%H:%M", lt)
    if t - time.time() < 6 * 86400:
        return time.strftime("%a %H:%M", lt)
    return time.strftime("%b %d %H:%M", lt)


def _fmt_dur(s: float) -> str:
    """Duration with explicit units so it cannot be read as a clock time: ``2d 07h 33m``,
    ``7h 33m``, ``33m 20s``."""
    if not np.isfinite(s) or s < 0:
        return "--"
    s = int(round(s))
    d, h, m, sec = s // 86400, (s % 86400) // 3600, (s % 3600) // 60, s % 60
    if d:
        return f"{d}d {h:02d}h {m:02d}m"
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {sec:02d}s"


def _fmt_hms(s: float) -> str:
    if not np.isfinite(s) or s < 0:
        return "--"
    s = int(round(s))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _fnum(v, nd=2) -> str:
    return "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _is_random_phase(p: str) -> bool:
    return p in RANDOM_PHASES


def _phase_color(p: str) -> str:
    return PHASE_COLORS.get(p, "#333333")


# ----------------------------------------------------------------------------
# normalised history
# ----------------------------------------------------------------------------
@dataclass
class ParamInfo:
    name: str
    base: str
    lo: float
    hi: float
    kind: str
    partition: Optional[str]
    role: str
    grid: Optional[List[float]] = None
    choices: Optional[List[Any]] = None


@dataclass
class AnnulusData:
    """Everything the panels need for one annulus, independent of live/post-hoc."""

    run_name: str
    annulus: int
    nann: int
    inrad: float
    outrad: float
    pxscale: float
    fwhm: float
    params: List[ParamInfo]
    partitions: List[str]
    X: np.ndarray                       # (n, ndim)
    y: np.ndarray                       # (n,) nan = failed
    phases: List[str]
    k_used: List[Any]                   # scalar / dict / None per eval
    selected: List[List[str]]
    part_snr: List[Dict[str, Optional[float]]]
    sources: List[List[Tuple[float, float, float]]]
    per_source: List[List[Optional[float]]]
    raw_per_source: List[List[Optional[float]]]
    clean_per_source: List[Optional[List[Optional[float]]]]
    raw: np.ndarray
    wall: np.ndarray
    contrast: np.ndarray
    configs: List[Dict[str, Any]]
    n_init: int
    n_iter: int
    gamma: float
    metric_name: str
    search_mode: str
    winner: Optional[Dict[str, Any]] = None        # winner.json-like
    validation: Optional[List[Dict[str, Any]]] = None
    calibration: Optional[Dict[str, Any]] = None
    angle_convention: str = "pa"
    seed_default: bool = True
    partition_label: str = "night"      # what the partitions ARE, for the panel titles
    #: real companions the objective was told about, ``[(rho_arcsec, pa_deg), ...]``.  They
    #: are kept out of the noise rings and never scored, so the panel is the only place they
    #: appear -- circled on every image, with their measured S/N on the clean reductions.
    known: List[Tuple[float, float]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- basics -------------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def ndim(self) -> int:
        return len(self.params)

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.y)

    @property
    def replicated(self) -> bool:
        r = self.__dict__.get("_replicated")
        if r is None:
            r = self.__dict__["_replicated"] = any(p.partition is not None for p in self.params)
        return r

    @property
    def multi(self) -> bool:
        return len(self.partitions) > 1

    def best(self) -> Tuple[int, float]:
        v = self.valid
        if not v.any():
            return -1, np.nan
        yy = np.where(v, self.y, -np.inf)
        i = int(np.argmax(yy))
        return i, float(yy[i])

    def running_best(self) -> np.ndarray:
        out = np.full(self.n, np.nan)
        b = np.nan
        for i, s in enumerate(self.y):
            if np.isfinite(s):
                b = s if not np.isfinite(b) else max(b, s)
            out[i] = b
        return out

    def top_mask(self) -> np.ndarray:
        """The TPE 'good' set: top ``gamma`` fraction of the valid scores."""
        m = np.zeros(self.n, bool)
        idx = np.flatnonzero(self.valid)
        if idx.size == 0:
            return m
        ng = max(int(round(self.gamma * idx.size)), 1)
        order = idx[np.argsort(-self.y[idx], kind="stable")]
        m[order[:ng]] = True
        return m

    def is_random(self) -> np.ndarray:
        return np.array([_is_random_phase(p) for p in self.phases], bool)

    # -- parameter structure ---------------------------------------------------
    @property
    def bases(self) -> List[str]:
        out = self.__dict__.get("_bases")
        if out is None:
            out = []
            for p in self.params:
                if p.role == "reduction" and p.base not in out:
                    out.append(p.base)
            self.__dict__["_bases"] = out
        return out

    def dims_of_base(self, base: str) -> List[int]:
        by = self.__dict__.get("_dims_of_base")
        if by is None:
            by = self.__dict__["_dims_of_base"] = {}
            for i, p in enumerate(self.params):
                if p.role == "reduction":
                    by.setdefault(p.base, []).append(i)
        return by.get(base, [])

    def dim_for(self, base: str, pid: Optional[str]) -> Optional[int]:
        """Dimension of ``base`` for partition ``pid`` (falls back to the global slot).

        Cached on first use.  The panels ask for this once per (evaluation, partition,
        base): on a six-night NEAR run past 7000 evaluations that is ~310,000 calls, and
        each one used to rescan the whole parameter list -- a full second of every render,
        growing with the run.  ``params`` never changes after construction, so the answer
        is built once.  The lookup order is preserved exactly: the first slot matching
        ``pid`` wins, otherwise the last global slot for that base.
        """
        cache = self.__dict__.get("_dim_cache")
        if cache is None:
            cache, glob = {}, {}
            for i, p in enumerate(self.params):
                if p.role != "reduction":
                    continue
                if p.partition is None:
                    glob[p.base] = i                    # last global slot wins, as before
                else:
                    cache.setdefault((p.base, p.partition), i)   # first match wins, as before
            self.__dict__["_dim_cache"] = cache
            self.__dict__["_dim_global"] = glob
        d = cache.get((base, pid))
        return self.__dict__["_dim_global"].get(base) if d is None else d

    def has_k(self) -> bool:
        return "k_klip" in self.bases or any(k is not None for k in self.k_used)

    def k_scalar(self, i: int) -> float:
        """Representative k of eval ``i`` (median over its selected partitions)."""
        if "k_klip" in self.bases:
            vals = []
            for pid in (self.selected[i] if self.replicated else [None]):
                d = self.dim_for("k_klip", pid)
                if d is not None:
                    vals.append(self.X[i, d])
            if vals:
                return float(np.median(vals))
        k = self.k_used[i]
        if isinstance(k, dict):
            v = [float(x) for x in k.values() if x is not None]
            return float(np.median(v)) if v else np.nan
        return np.nan if k is None else float(k)

    def k_partition(self, i: int, pid: str) -> float:
        d = self.dim_for("k_klip", pid) if "k_klip" in self.bases else None
        if d is not None and self.params[d].partition == pid:
            return float(self.X[i, d])
        k = self.k_used[i]
        if isinstance(k, dict):
            v = k.get(str(pid))
            return np.nan if v is None else float(v)
        if d is not None:
            return float(self.X[i, d])
        return np.nan if k is None else float(k)

    def k_hi(self) -> float:
        ks = [self.k_scalar(i) for i in range(self.n)]
        ks = [k for k in ks if np.isfinite(k)]
        d = self.dims_of_base("k_klip")
        if d:
            return float(self.params[d[0]].hi)
        return float(max(ks)) if ks else 30.0

    def inclusion(self) -> np.ndarray:
        """(n, npart) boolean inclusion matrix.

        The selection is turned into a set once per evaluation rather than a fresh list
        per cell: the inner comprehension was rebuilt ``n x npartitions`` times per render.
        """
        cols = [str(pid) for pid in self.partitions]
        m = np.zeros((self.n, len(cols)), bool)
        for i, sel in enumerate(self.selected):
            chosen = {str(s) for s in sel}
            for j, pid in enumerate(cols):
                m[i, j] = pid in chosen
        return m

    def part_snr_matrix(self) -> np.ndarray:
        out = np.full((self.n, len(self.partitions)), np.nan)
        for i, d in enumerate(self.part_snr):
            for j, pid in enumerate(self.partitions):
                v = (d or {}).get(str(pid))
                out[i, j] = np.nan if v is None else float(v)
        return out

    # -- corner matrices --------------------------------------------------------
    def corner(self, partition: Optional[str] = None) -> Dict[str, Any]:
        """Rows for a corner plot.

        ``partition=None`` -> per-base view: one row per (eval, selected partition) with
        the partition's slot values (global dims repeated); ``partition=pid`` -> that
        partition's block (evals in which it was included).  A ``k_klip*`` column is
        appended when k is not a searched base but was recorded.
        """
        bases = [b for b in self.bases]
        names, lo, hi, kinds = [], [], [], []
        for b in bases:
            p = self.params[self.dims_of_base(b)[0]]
            names.append(b)
            lo.append(p.lo)
            hi.append(p.hi)
            kinds.append(p.kind)
        add_k = ("k_klip" not in bases) and any(k is not None for k in self.k_used)
        if add_k:
            names.append("k_klip*")
            lo.append(1.0)
            hi.append(self.k_hi())
            kinds.append("int")
        rows, ys, ev, pids = [], [], [], []
        for i in range(self.n):
            if not np.isfinite(self.y[i]):
                continue
            if partition is None:
                parts = [str(s) for s in self.selected[i]] if self.replicated else [None]
            else:
                if str(partition) not in [str(s) for s in self.selected[i]]:
                    continue
                parts = [str(partition)]
            for pid in parts:
                row = []
                for b in bases:
                    d = self.dim_for(b, pid)
                    row.append(np.nan if d is None else self.X[i, d])
                if add_k:
                    row.append(self.k_partition(i, pid) if pid is not None else self.k_scalar(i))
                rows.append(row)
                ys.append(self.y[i])
                ev.append(i)
                pids.append(pid)
        Xc = np.array(rows, float).reshape(len(rows), len(names))
        return {"names": names, "lo": np.array(lo, float), "hi": np.array(hi, float), "kinds": kinds,
                "X": Xc, "y": np.array(ys, float), "eval": np.array(ev, int), "pid": pids}

    def eval_point(self, i: int, cs: Dict[str, Any], pid: Optional[str] = None) -> Optional[np.ndarray]:
        """Corner coordinates of eval ``i`` (first matching row)."""
        if i < 0 or i >= self.n:
            return None
        for r, (e, p) in enumerate(zip(cs["eval"], cs["pid"])):
            if e == i and (pid is None or p == pid):
                return cs["X"][r]
        return None

    # -- text ------------------------------------------------------------------
    def config_lines(self, i: int, label: str, max_width: int = TEXT_WIDTH) -> List[str]:
        """Compact configuration summary of eval ``i``: representative per-base values
        and, for replicated spaces, a per-partition table with abbreviated headers
        (:data:`COL_ABBREV`).  Tables wider than ``max_width`` characters are wrapped
        into several column blocks (the partition column repeated)."""
        if i < 0 or i >= self.n:
            return [f"{label}: --"]
        cfg = self.configs[i] or {}
        params = cfg.get("params", {})
        per = cfg.get("per_partition", {})
        sel = [str(s) for s in cfg.get("selected", self.selected[i])]
        rep_bases = [b for b in self.bases if any(p.partition is not None for p in self.params if p.base == b)]
        glob = [b for b in self.bases if b not in rep_bases]
        lines = [label]
        if glob:
            lines.append("  " + "  ".join(f"{ab}={_fmtval(params.get(b), 4)}"
                                          for b, ab in zip(glob, _abbrev_unique(glob))))
        if rep_bases and per:
            cols = rep_bases + ([] if "k_klip" in rep_bases or not isinstance(self.k_used[i], dict) else ["k"])
            w = max(4, max(len(str(p)) for p in self.partitions)) if self.partitions else 4
            # cell text per (partition, column), then column widths from the content
            table: Dict[str, List[str]] = {}
            for pid in self.partitions:
                d = per.get(str(pid), {})
                row = []
                for c in cols:
                    if c == "k":
                        row.append(_fmtval(self.k_used[i].get(str(pid)), 4))
                    else:
                        row.append(_fmtval(d.get(c, params.get(c)), 4))
                table[str(pid)] = row
            heads = _abbrev_unique(cols)
            widths = [max(len(h), max((len(table[str(p)][k]) for p in self.partitions), default=1))
                      for k, h in enumerate(heads)]
            # wrap the columns into blocks that fit max_width: "  part " + cols + "  in"
            blocks: List[List[int]] = [[]]
            used = 2 + w + 4
            for k, cw in enumerate(widths):
                if blocks[-1] and used + cw + 1 > max_width:
                    blocks.append([])
                    used = 2 + w + 4
                blocks[-1].append(k)
                used += cw + 1
            for bk, blk in enumerate(blocks):
                lines.append("  " + "part".ljust(w) + " " + " ".join(heads[k].rjust(widths[k]) for k in blk)
                             + ("  in" if bk == 0 else ""))
                for pid in self.partitions:
                    row = table[str(pid)]
                    lines.append("  " + str(pid).ljust(w) + " " + " ".join(row[k].rjust(widths[k]) for k in blk)
                                 + (("   *" if str(pid) in sel else "   -") if bk == 0 else ""))
        elif not glob:
            items = list(params.items())[:8]
            lines.append("  " + "  ".join(f"{ab}={_fmtval(v, 4)}"
                                          for (k, v), ab in zip(items, _abbrev_unique([k for k, _ in items]))))
        return lines


def _fmtval(v, sig: int = 6) -> str:
    """Compact value text (``sig`` significant digits for non-integral floats)."""
    if v is None:
        return "--"
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "--"
        return f"{v:.{sig}g}" if abs(v - round(v)) > 1e-9 else str(int(round(v)))
    return str(v)


# ----------------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------------
def _params_from_space(space: Dict[str, Any]) -> List[ParamInfo]:
    out = []
    for p in space.get("params", []):
        out.append(ParamInfo(name=p["name"], base=p.get("base") or p["name"], lo=float(p["lo"]), hi=float(p["hi"]),
                             kind=p.get("kind", "float"),
                             partition=None if p.get("partition") is None else str(p["partition"]),
                             role=p.get("role", "reduction"), grid=p.get("grid"), choices=p.get("choices")))
    return out


def _per_annulus(val, ia: int) -> int:
    if isinstance(val, (list, tuple)):
        return int(val[min(ia, len(val) - 1)])
    return int(val)


def annulus_from_records(records: Sequence[Dict[str, Any]], space: Dict[str, Any], cfg: Dict[str, Any],
                         ia: int, pxscale: float, fwhm: float, metric_name: str = "metric",
                         run_name: str = "", winner: Optional[Dict[str, Any]] = None,
                         validation: Optional[List[Dict[str, Any]]] = None,
                         calibration: Optional[Dict[str, Any]] = None,
                         angle_convention: str = "pa",
                         partition_label: str = "night",
                         known: Sequence[Tuple[float, float]] = ()) -> AnnulusData:
    """Normalise ``results.jsonl``-style records (one annulus, one restart segment)."""
    params = _params_from_space(space)
    ndim = len(params)
    recs = list(records)
    n = len(recs)
    X = np.array([r.get("x", []) for r in recs], float).reshape(n, ndim) if n else np.zeros((0, ndim))
    y = np.array([np.nan if r.get("score") is None else float(r["score"]) for r in recs], float)
    raw = np.array([np.nan if r.get("raw_score") is None else float(r["raw_score"]) for r in recs], float)
    parts = [str(p) for p in space.get("partitions", [])]
    if not parts:
        seen: List[str] = []
        for r in recs:
            for s in (r.get("meta", {}).get("selected") or r.get("config", {}).get("selected") or []):
                if str(s) not in seen:
                    seen.append(str(s))
        parts = seen
    for r in recs:                                   # EvalRecord.meta carries the reducer's convention
        ac = (r.get("meta") or {}).get("angle_convention")
        if ac:
            angle_convention = str(ac)
            break
    edges = cfg.get("ann_edges", [])
    inrad = float(edges[ia]) if len(edges) > ia + 1 else 0.0
    outrad = float(edges[ia + 1]) if len(edges) > ia + 1 else (float(edges[0]) if edges else 0.0)
    nann = max(len(edges) - 1, 1)
    return AnnulusData(
        run_name=run_name, annulus=ia, nann=nann, inrad=inrad, outrad=outrad, pxscale=float(pxscale),
        fwhm=float(fwhm), params=params, partitions=parts, X=X, y=y,
        phases=[str(r.get("phase", "?")) for r in recs],
        k_used=[r.get("k_used") for r in recs],
        selected=[[str(s) for s in (r.get("meta", {}).get("selected") or r.get("config", {}).get("selected") or parts)]
                  for r in recs],
        part_snr=[dict(r.get("partition_snr") or {}) for r in recs],
        sources=[[tuple(s) for s in (r.get("sources") or [])] for r in recs],
        per_source=[list(r.get("per_source") or []) for r in recs],
        raw_per_source=[list(r.get("raw_per_source") or []) for r in recs],
        clean_per_source=[r.get("clean_per_source") for r in recs],
        raw=raw, wall=np.array([float(r.get("wall_s", 0.0) or 0.0) for r in recs]),
        contrast=np.array([float(r.get("contrast", np.nan) or np.nan) for r in recs]),
        configs=[dict(r.get("config") or {}) for r in recs],
        n_init=_per_annulus(cfg.get("n_init", 0), ia), n_iter=_per_annulus(cfg.get("n_iter", n), ia),
        gamma=float(cfg.get("gamma", 0.25)), metric_name=metric_name, search_mode=str(cfg.get("search_mode", "?")),
        winner=winner, validation=validation, calibration=calibration, angle_convention=angle_convention,
        seed_default=bool(cfg.get("seed_default", True)), partition_label=str(partition_label or "night"),
        known=[(float(k[0]), float(k[1])) for k in (known or ()) if k is not None and len(k) >= 2],
        # Per-eval remeasurement record under RunConfig.n_remeasure > 1: the panel has to say
        # that its score is a mean of n and how far the draws sat apart, or the number beside
        # a single displayed image reads as that image's measurement.
        extra={"draws": [{k: (r.get("meta") or {}).get(k)
                          for k in ("draw_scores", "draw_n", "draw_sd", "draw_spread", "draw_failed")}
                         for r in recs]})


def annulus_from_run(run_dir: str, ia: int, run: Optional[Dict[str, Any]] = None,
                     upto: Optional[int] = None) -> AnnulusData:
    """Post-hoc :class:`AnnulusData` of annulus ``ia`` from a run directory (last
    restart segment; ``upto`` truncates to the first ``upto`` evaluations)."""
    run = run or load_run(run_dir)
    recs = _annulus_records(run, ia)
    if upto is not None:
        recs = recs[:upto]
    setup = run.get("setup") or run.get("checkpoint") or {}
    px = float(setup.get("pxscale", 1.0))
    fw = float(setup.get("fwhm_px", 4.0))
    calib = None
    cpath = os.path.join(run_dir, f"annulus{ia + 1:02d}", "calibration.json")
    if os.path.exists(cpath):
        with open(cpath) as f:
            calib = json.load(f)
    ac = str(setup.get("angle_convention") or (setup.get("reducer") or {}).get("angle_convention") or "pa")
    return annulus_from_records(recs, _space(run), _config(run), ia, px, fw, _metric_name(run),
                                os.path.basename(os.path.abspath(run_dir)), run["winners"].get(ia),
                                run["validation"].get(ia), calib, ac,
                                str(setup.get("partition_label") or "night"),
                                ((setup.get("sampler") or {}).get("known")
                                 or ((setup.get("objective") or {}).get("metric") or {}).get("known")
                                 or ()))


def _record_dict(rec) -> Dict[str, Any]:
    return json.loads(rec.to_json())


def _runner_known(runner) -> List[Tuple[float, float]]:
    """Real companions this run was told about, from whichever component carries them.

    The objective's metric keeps them out of its noise rings and the sampler keeps
    injections away from them; either is authoritative, so take the first that has any.
    """
    for obj in (getattr(runner, "objective", None), getattr(runner, "sampler", None)):
        for holder in (getattr(obj, "metric", None), obj):
            k = getattr(holder, "known", None)
            if k:
                try:
                    return [(float(a), float(b)) for a, b in k]
                except Exception:
                    continue
    return []


def annulus_from_runner(runner, records: Sequence[Dict[str, Any]]) -> AnnulusData:
    """Live :class:`AnnulusData` from a runner and the record dicts collected so far."""
    space = runner.space.to_dict()
    cfg = runner.cfg.to_dict()
    ia = runner.ia
    try:
        d = runner.objective.describe()
        mname = d["metric"]["name"] + (" (clean-subtracted)" if d.get("clean_subtract") else "")
    except Exception:
        mname = "metric"
    winner = None
    if len(runner.results) > ia and runner.results[ia] is not None:
        winner = runner.results[ia].to_dict()
    calib = getattr(runner, "_display_calib", None)
    return annulus_from_records(records, space, cfg, ia, runner.pxscale, runner.fwhm, mname,
                                os.path.basename(os.path.abspath(runner.run_dir)), winner,
                                None if winner is None else winner.get("validation_table"), calib,
                                getattr(runner.reducer, "angle_convention",
                                        getattr(runner.objective.metric, "angle_convention", "pa")),
                                getattr(runner.reducer, "partition_label", "night"),
                                _runner_known(runner))


# ----------------------------------------------------------------------------
# image helpers
# ----------------------------------------------------------------------------
def _robust_sigma(img: np.ndarray) -> float:
    v = img[np.isfinite(img)]
    if v.size < 10:
        return 1.0
    med = np.median(v)
    s = 1.4826 * np.median(np.abs(v - med))
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(v)) or 1.0
    return float(s)


def _crop_extent(img: np.ndarray, pxscale: float, rmax_px: float) -> Tuple[np.ndarray, List[float]]:
    ny, nx = img.shape
    cx, cy = star_center(img.shape)
    r = int(min(math.ceil(rmax_px), math.floor(min(cx, cy))))
    x0, x1 = int(round(cx - r)), int(round(cx + r))
    y0, y1 = int(round(cy - r)), int(round(cy + r))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, nx - 1), min(y1, ny - 1)
    sub = img[y0:y1 + 1, x0:x1 + 1]
    ext = [(x0 - 0.5 - cx) * pxscale, (x1 + 0.5 - cx) * pxscale, (y0 - 0.5 - cy) * pxscale, (y1 + 0.5 - cy) * pxscale]
    return sub, ext


def _src_arcsec(sources, angle_convention="pa") -> Tuple[np.ndarray, np.ndarray]:
    if not sources:
        return np.zeros(0), np.zeros(0)
    rho = np.array([s[0] for s in sources], float)
    th = np.array([s[1] for s in sources], float)
    x, y = source_xy(rho, th, 1.0, 0.0, 0.0, angle_convention)
    return x, y


#: measured S/N of the known companions, cached per image so the several cells of one panel
#: pay for it once.  Each entry keeps a reference to the array it was measured on, which is
#: what makes ``id(img)`` a sound key: an array that is still referenced cannot be freed, so
#: its address cannot be handed to a different array while the entry is live.  Without that
#: reference the cache silently answered for a *previous* image that had been collected and
#: whose address had been reused -- a wrong S/N printed on a panel (never a score, since the
#: objective does not read this).  Small, because one panel needs only the last few.
_KNOWN_SNR_CACHE: "Dict[Any, Tuple[Any, Any]]" = {}
_KNOWN_SNR_CACHE_MAX = 8


def known_snr(ad: AnnulusData, img) -> Optional[List[float]]:
    """S/N of each known companion in ``img``, or None when there is nothing to measure.

    The objective never scores these -- that is what makes them an honest check -- so this
    is the only place the number is produced.  Measured on a clean (un-injected) reduction,
    with the other known sources excluded from the noise ring exactly as the metric does.
    """
    known = getattr(ad, "known", None)
    if not known or img is None or np.ndim(img) != 2:
        return None
    key = (id(img), float(ad.fwhm), float(ad.pxscale), tuple(map(tuple, known)))
    hit = _KNOWN_SNR_CACHE.get(key)
    if hit is not None and hit[0] is img:
        return hit[1]
    try:
        from .metrics import MawetPeakSNR
        m = MawetPeakSNR(pxscale=ad.pxscale, fwhm=ad.fwhm, known=list(known))
        out = [float(v) for v in m.per_source(np.asarray(img, float), None,
                                              [float(k[0]) for k in known],
                                              [float(k[1]) for k in known])]
    except Exception:
        return None
    if len(_KNOWN_SNR_CACHE) >= _KNOWN_SNR_CACHE_MAX:
        _KNOWN_SNR_CACHE.clear()                   # also releases the arrays it was holding
    _KNOWN_SNR_CACHE[key] = (img, out)             # the reference is load-bearing: see above
    return out


def draw_image(ax, img: Optional[np.ndarray], ad: AnnulusData, title: str, cmap: str = "inferno",
               sources=None, labels=None, snr: bool = False, note: Optional[str] = None,
               flatten: bool = False, vrange: Optional[Tuple[float, float]] = None, edges: bool = False,
               scale: str = "linear", known_labels=None):
    """One image cell: robust ``-1..5 sigma`` stretch of the image (or a fixed
    ``-3..8`` stretch for S/N maps), arcsec axes centred on the star, injected
    sources circled (0.9 FWHM) and labelled with their per-source metric value.

    ``flatten=False`` by default, matching the metrics: the panel shows the image the
    run actually scored.  The robust stretch is measured on whatever it is handed, so
    an unflattened panel is dominated by the stellar halo at small separations -- that
    is the honest picture, not a display fault.  Pass ``flatten=True`` for the old
    :func:`radprof` stretch."""
    ax.set_title(title)
    ax.grid(False)
    if img is None or np.ndim(img) != 2 or img.shape[0] < 4 or not np.isfinite(img).any():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, note or "collecting...", ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="#666666")
        return
    im = radprof(img) if (flatten and not snr) else np.asarray(img, float)
    rmax = ad.outrad + 1.5 * ad.fwhm if ad.outrad > 0 else min(img.shape) / 2
    sub, ext = _crop_extent(im, ad.pxscale, rmax)
    if snr:
        vmin, vmax = SNR_RANGE
    elif vrange is not None:
        vmin, vmax = vrange
    else:
        s = _robust_sigma(sub)
        vmin, vmax = -1.0 * s, 5.0 * s
    if scale in ("log", "symlog"):
        # log stretch over 3 decades below the peak (PSF models / KLIP-FM responses): symlog
        # keeps the negative self-subtraction lobes of a forward model visible
        peak = float(np.nanmax(np.abs(sub))) if np.isfinite(sub).any() else 1.0
        peak = peak if peak > 0 else 1.0
        if scale == "log":
            norm = mcolors.LogNorm(vmin=peak * 1e-3, vmax=peak)
            ax.imshow(np.where(sub > 0, sub, np.nan), origin="lower", extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
        else:
            norm = mcolors.SymLogNorm(linthresh=peak * 0.1, vmin=-peak, vmax=peak, base=10)
            ax.imshow(sub, origin="lower", extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
    else:
        ax.imshow(sub, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_facecolor("#111111")
    ax.set_xlabel("arcsec")
    ax.set_ylabel("arcsec")
    ax.set_aspect("equal")
    if edges:                                   # annulus edges (off by default: the IDL panels show none)
        for r in (ad.inrad, ad.outrad):
            if r > 0:
                ax.add_patch(matplotlib.patches.Circle((0, 0), r * ad.pxscale, fill=False, ec="white", lw=0.5, ls=":", alpha=0.6))
    rad = 0.9 * ad.fwhm * ad.pxscale
    if sources:
        xs, ys = _src_arcsec(sources, ad.angle_convention)
        for q, (x, y) in enumerate(zip(xs, ys)):
            ax.add_patch(matplotlib.patches.Circle((x, y), rad, fill=False, ec=SRC_COLOR, lw=1.1))
            if labels is not None and q < len(labels) and labels[q] is not None:
                lab = labels[q] if isinstance(labels[q], str) else (f"{labels[q]:.1f}" if np.isfinite(labels[q]) else None)
                if lab:
                    ax.text(x + 1.2 * rad, y, lab, color=SRC_COLOR, fontsize=7, va="center")
    # A known companion is never injected and never scored, so it appears in no other list.
    # Circle it on every image -- in its own colour, so it cannot be read as an injection --
    # and label it with its measured S/N where the caller has one (the clean reductions).
    for q, k in enumerate(getattr(ad, "known", None) or []):
        kx, ky = _src_arcsec([(float(k[0]), float(k[1]), 0.0)], ad.angle_convention)
        ax.add_patch(matplotlib.patches.Circle((kx[0], ky[0]), 1.15 * rad, fill=False,
                                               ec=KNOWN_COLOR, lw=1.3, ls="--"))
        v = known_labels[q] if (known_labels is not None and q < len(known_labels)) else None
        lab = v if isinstance(v, str) else (f"{v:.1f}" if v is not None and np.isfinite(v) else None)
        if lab:
            ax.text(kx[0] + 1.4 * rad, ky[0], lab, color=KNOWN_COLOR, fontsize=7,
                    va="center", fontweight="bold")
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=6.5, color="white", va="bottom")
    if snr and not getattr(ax, "_no_caption", False):
        ax.text(0.5, -0.30, SNR_CAPTION, transform=ax.transAxes, fontsize=6.5, ha="center", color="#555555")


def _fm_range(img: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
    """Linear ``0 .. max`` stretch for the (noiseless) KLIP-FM response images."""
    if img is None or not np.isfinite(img).any():
        return None
    hi = float(np.nanmax(img))
    return (0.0, hi) if hi > 0 else None


def _snr_map(img: Optional[np.ndarray], ad: AnnulusData, sources=None,
             flatten: bool = False) -> Optional[np.ndarray]:
    if img is None or np.ndim(img) != 2 or img.shape[0] < 8:
        return None
    from .products import snr_map
    try:
        ex = []
        if sources:
            cx, cy = star_center(img.shape)
            rho = [s[0] for s in sources]
            th = [s[1] for s in sources]
            xs, ys = source_xy(rho, th, ad.pxscale, cx, cy, ad.angle_convention)
            ex = list(zip(xs, ys))
        return snr_map(radprof(img) if flatten else img, ad.fwhm, exclude_xy=ex)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# panel primitives
# ----------------------------------------------------------------------------
def panel_trace(ax, ad: AnnulusData, current: Optional[int] = None, compact: bool = True):
    """near2m_show convergence trace: score vs evaluation coloured by phase, running
    best (search), seeded default level, validated winner if known, failed as x."""
    n = ad.n
    ax.set_title(f"search score vs evaluation  [{ad.metric_name}]", loc="left")
    if n == 0:
        ax.text(0.5, 0.5, "collecting...", ha="center", va="center", transform=ax.transAxes)
        return
    ev = np.arange(1, n + 1)
    y = ad.y
    # Under the points, when n_remeasure > 1: the min-to-max span of the draws the trial's
    # score is the mean of.  Without it a panel shows a mean as if it were a measurement,
    # and on this objective the draws span ~1.4 in S/N against a useful range of ~6.
    try:
        from .plots import draw_ranges
        _dw = (ad.extra or {}).get("draws") or []
        # _phase_color, not the marker facecolor: random phases draw open markers, so the
        # edge is the colour the eye reads for them.
        draw_ranges(ax, ev, [(d or {}).get("draw_scores") if isinstance(d, dict) else None
                             for d in (_dw + [None] * max(0, n - len(_dw)))][:n],
                    colors=[_phase_color(p) for p in ad.phases])
    except Exception:
        pass
    for p in PHASE_ORDER + sorted(set(ad.phases) - set(PHASE_ORDER)):
        m = np.array([q == p for q in ad.phases]) & np.isfinite(y)
        if not m.any():
            continue
        rnd = _is_random_phase(p)
        ax.scatter(ev[m], y[m], s=30 if p == "seed" else 13, marker="*" if p == "seed" else "o",
                   facecolor="none" if (rnd and p != "seed") else _phase_color(p), edgecolor=_phase_color(p),
                   lw=0.9, label=p, zorder=3)
    fail = ~np.isfinite(y)
    fin = y[np.isfinite(y)]
    if fail.any():
        y0 = float(np.nanmin(fin)) if fin.size else 0.0
        ax.scatter(ev[fail], np.full(fail.sum(), y0), marker="x", color="#d62728", s=16, label="failed", zorder=3)
    rb = ad.running_best()
    ax.step(ev, rb, where="post", color="#444444", lw=1.0, label="running best (search)")
    if ad.seed_default and n >= 1 and ad.phases[0] == "seed" and np.isfinite(y[0]):
        ax.axhline(y[0], color="#000000", ls=":", lw=0.8, label="seeded default")
    bi, bs = ad.best()
    if bi >= 0:
        ax.scatter([bi + 1], [bs], s=90, marker="o", facecolor="none", edgecolor="#009E73", lw=1.6, zorder=4)
    if current is not None and 0 <= current < n and np.isfinite(y[current]):
        ax.scatter([current + 1], [y[current]], s=70, marker="s", facecolor="none", edgecolor=CUR_COLOR, lw=1.2, zorder=4)
    w = ad.winner
    if w is not None and w.get("validated"):
        ws, wi = float(w["winner_score"]), int(w["winner_index"]) + 1
        ax.axhline(ws, color=VALID_COLOR, ls="--", lw=1.0)
        ax.scatter([wi], [ws], marker="D", s=44, facecolor="none", edgecolor=VALID_COLOR, lw=1.4, zorder=5,
                   label=f"validated winner {ws:.2f}")
    if ad.n_init and n > ad.n_init:
        ax.axvline(ad.n_init + 0.5, color="#999999", ls="--", lw=0.7)
    ax.set_xlim(0.5, max(n, 1) + 0.5)                # IDL: xrange=[0.5, nev+0.5] -- grows with the evals done
    if fin.size:
        lo, hi = float(min(fin.min(), 0.0)), float(fin.max())
        ax.set_ylim(lo - 0.05 * (hi - lo + 1e-9), hi + 0.15 * (hi - lo + 1e-9))
    ax.set_xlabel("evaluation")
    ax.set_ylabel(SEARCH_LABEL if not compact else "score (upward-biased)")
    ax.legend(loc="lower right", ncol=3 if compact else 4, handletextpad=0.2, columnspacing=0.8)


def panel_score_hist(ax, ad: AnnulusData):
    """near2m_snrhist: score distribution per phase group (random / guided / local) as
    fractions with Gaussian fits and means."""
    ax.set_title("score distribution by phase", loc="left")
    v = ad.valid
    if v.sum() < 2:
        ax.text(0.5, 0.5, "collecting...", ha="center", va="center", transform=ax.transAxes)
        return
    y = ad.y[v]
    ph = np.array(ad.phases)[v]
    rnd = np.array([_is_random_phase(p) for p in ph])
    groups = [("warm-up/explore", rnd, "#999999"), ("local", (ph == "local") & ~rnd, PHASE_COLORS["local"]),
              ("guided", ~rnd & (ph != "local"), PHASE_COLORS["tpe"])]
    lo, hi = float(y.min()), float(y.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, 13)
    bw = edges[1] - edges[0]
    k = 0
    present = [g for g in groups if g[1].any()]
    for name, m, col in present:
        h, _ = np.histogram(y[m], bins=edges)
        frac = h / max(m.sum(), 1)
        off = (k - (len(present) - 1) / 2) * bw * 0.28
        ax.bar(0.5 * (edges[1:] + edges[:-1]) + off, frac, width=bw * 0.26, color=col, alpha=0.85,
               label=f"{name} (n={int(m.sum())})")
        mu = float(np.mean(y[m]))
        ax.axvline(mu, color=col, ls="--", lw=0.8)
        if m.sum() >= 3:
            sd = float(np.std(y[m], ddof=1))
            if sd > 0:
                xg = np.linspace(lo, hi, 120)
                ax.plot(xg, bw / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((xg - mu) / sd) ** 2), color=col, lw=1.0)
        k += 1
    ax.set_xlabel("search score (upward-biased)")
    ax.set_ylabel("fraction of evals")
    ax.legend(loc="upper left")


def panel_marginals(fig, spec, ad: AnnulusData, current: Optional[int] = None, ncol: int = 3):
    """Per-parameter marginal histograms (grouped by base, per-partition slots pooled):
    all evaluations vs the top-gamma set, current (dashed) and best (solid) marked."""
    bases = ad.bases
    add_k = "k_klip" not in bases and any(k is not None for k in ad.k_used)
    items = list(bases) + (["k_klip*"] if add_k else [])
    nb = len(items)
    if nb == 0:
        ax = fig.add_subplot(spec)
        ax.text(0.5, 0.5, "no reduction parameters", ha="center", va="center", transform=ax.transAxes)
        return
    ncol = min(ncol, nb)
    nrow = int(math.ceil(nb / ncol))
    sub = GridSpecFromSubplotSpec(nrow + 1, ncol, subplot_spec=spec, hspace=0.7, wspace=0.25,
                                  height_ratios=[0.08] + [1.0] * nrow)
    top = ad.top_mask()
    bi, _ = ad.best()
    for k, base in enumerate(items):
        ax = fig.add_subplot(sub[1 + k // ncol, k % ncol])
        if base == "k_klip*":
            vals_all = np.array([ad.k_scalar(i) for i in range(ad.n)])
            vals_top = vals_all[top]
            cur = [ad.k_scalar(current)] if current is not None else []
            best = [ad.k_scalar(bi)] if bi >= 0 else []
            hi = ad.k_hi()
            bins = np.arange(0.5, hi + 1.5, max(1.0, round(hi / 15)))
            p0 = None
        else:
            dims = ad.dims_of_base(base)
            p0 = ad.params[dims[0]]
            bins = _bins_for({"grid": p0.grid, "lo": p0.lo, "hi": p0.hi, "kind": p0.kind})
            if len(bins) > 40:
                bins = np.linspace(p0.lo, p0.hi, 21)
            vals_all = ad.X[:, dims].ravel() if ad.n else np.zeros(0)
            vals_top = ad.X[top][:, dims].ravel() if ad.n else np.zeros(0)
            cur = list(ad.X[current, dims]) if current is not None and 0 <= current < ad.n else []
            best = list(ad.X[bi, dims]) if bi >= 0 else []
        vals_all = vals_all[np.isfinite(vals_all)]
        vals_top = vals_top[np.isfinite(vals_top)]
        if vals_all.size:
            ax.hist(vals_all, bins=bins, color="#d0d0d0", density=True)
        if vals_top.size:
            ax.hist(vals_top, bins=bins, color=PHASE_COLORS["tpe"], alpha=0.65, density=True)
        for v in best:
            ax.axvline(v, color=BEST_COLOR, lw=1.2)
        for v in cur:
            ax.axvline(v, color=CUR_COLOR, lw=1.0, ls="--")
        npart = len(dims) if base != "k_klip*" else 0
        ax.set_title(base + (f" (x{npart})" if npart > 1 else ""), fontsize=7.5)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        if p0 is not None and p0.choices:
            ax.set_xticks(range(len(p0.choices)))
            ax.set_xticklabels([str(c) for c in p0.choices], rotation=30, fontsize=6)
    pos = spec.get_position(fig)
    fig.text(pos.x0, pos.y1, f"marginals: all evals (grey) vs top {ad.gamma:.0%} by search score (blue); "
             "best solid, current dashed", fontsize=7.5, va="top")


def panel_k(ax, ad: AnnulusData, current: Optional[int] = None):
    """near2m_kpanel: search score vs k_klip (per partition when k is per-partition),
    the running best marked; 'k not searched' when the space has no k."""
    ax.set_title("score vs k_klip", loc="left")
    if not ad.has_k():
        ax.text(0.5, 0.5, "k_klip not searched\n(single k, no scan)", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    v = ad.valid
    if not v.any():
        ax.text(0.5, 0.5, "collecting...", ha="center", va="center", transform=ax.transAxes)
        return
    rnd = ad.is_random()
    per_part = ad.replicated and ("k_klip" in ad.bases or any(isinstance(k, dict) for k in ad.k_used))
    cmap = matplotlib.colormaps.get_cmap("tab10")
    if per_part and ad.partitions:
        inc = ad.inclusion()
        for j, pid in enumerate(ad.partitions):
            ks = np.array([ad.k_partition(i, pid) for i in range(ad.n)])
            m = v & inc[:, j] & np.isfinite(ks)
            if not m.any():
                continue
            jit = (j - (len(ad.partitions) - 1) / 2) * 0.12
            ax.scatter(ks[m & ~rnd] + jit, ad.y[m & ~rnd], s=12, color=cmap(j % 10), label=f"{pid}", zorder=3)
            ax.scatter(ks[m & rnd] + jit, ad.y[m & rnd], s=12, facecolor="none", edgecolor=cmap(j % 10), lw=0.8, zorder=3)
        ax.legend(loc="best", title="partition", ncol=2, handletextpad=0.1)
    else:
        ks = np.array([ad.k_scalar(i) for i in range(ad.n)])
        m = v & np.isfinite(ks)
        ax.scatter(ks[m & ~rnd], ad.y[m & ~rnd], s=12, color=PHASE_COLORS["tpe"], label="guided", zorder=3)
        ax.scatter(ks[m & rnd], ad.y[m & rnd], s=12, facecolor="none", edgecolor="#999999", lw=0.8, label="random", zorder=3)
        ax.legend(loc="lower right")
    bi, bs = ad.best()
    if bi >= 0:
        kb = ad.k_scalar(bi)
        if np.isfinite(kb):
            ax.scatter([kb], [bs], s=90, marker="o", facecolor="none", edgecolor="#009E73", lw=1.6, zorder=4)
    if current is not None and 0 <= current < ad.n and np.isfinite(ad.y[current]):
        kc = ad.k_scalar(current)
        if np.isfinite(kc):
            ax.scatter([kc], [ad.y[current]], s=70, marker="s", facecolor="none", edgecolor=CUR_COLOR, lw=1.2, zorder=4)
    ax.set_xlabel("k_klip (# PCs)" + ("  [median over selected partitions]" if per_part and not ad.replicated else ""))
    ax.set_ylabel("search score")
    ax.set_xlim(0.5, ad.k_hi() + 0.5)


def _evtick(n: int) -> int:
    """IDL ``near2m_evtick``: a 1/2/5 x 10^k tick interval giving ~6 ticks over ``n`` evals."""
    if n <= 1:
        return 1
    raw = n / 6.0
    mag = 10.0 ** np.floor(np.log10(max(raw, 1e-9)))
    f = raw / mag
    nice = 1.0 if f < 1.5 else (2.0 if f < 3.5 else (5.0 if f < 7.5 else 10.0))
    return max(int(nice * mag), 1)


def panel_inclusion(ax, ad: AnnulusData, colorbar: bool = True):
    """near2m_nspanel / nightmap top: inclusion matrix eval x partition, cells coloured
    by the eval's search score (failed = hatched)."""
    ax.set_title("partition inclusion (colour = search score)", loc="left")
    ax.grid(False)
    if not ad.multi:
        ax.text(0.5, 0.5, "single partition -- selection not searched", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    inc = ad.inclusion()
    n, npart = inc.shape
    fin = ad.y[np.isfinite(ad.y)]
    vmin, vmax = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1.0
    img = np.full((npart, n), np.nan)
    for i in range(n):
        for j in range(npart):
            if inc[i, j]:
                img[j, i] = ad.y[i] if np.isfinite(ad.y[i]) else vmin
    cm = matplotlib.colormaps.get_cmap(SCORE_CMAP).copy()
    cm.set_bad(_grey("#f2f2f2", "#000000"))
    im = ax.imshow(img, aspect="auto", origin="lower", cmap=cm, vmin=vmin, vmax=vmax,
                   extent=[0.5, n + 0.5, 0.5, npart + 0.5], interpolation="nearest")
    for i in range(n):
        if not np.isfinite(ad.y[i]):
            ax.axvspan(i + 0.5, i + 1.5, color="#d62728", alpha=0.25, lw=0)
    if ad.n_init and n > ad.n_init:
        ax.axvline(ad.n_init + 0.5, color=_fg(), ls="--", lw=0.8)
    ax.set_yticks(range(1, npart + 1))
    ax.set_yticklabels(ad.partitions)
    ax.set_xlabel("evaluation")
    # IDL near2m_nspanel: xrange=[0.5, nev+0.5] with near2m_evtick(nev) -- the axis grows with the
    # evaluations done in THIS annulus (so it starts over at each new annulus)
    ax.set_xlim(0.5, max(n, 1) + 0.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(_evtick(n)))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if colorbar:
        ax.figure.colorbar(im, ax=ax, pad=0.01, fraction=0.05).set_label("search score", fontsize=7)


def panel_partition_snr(ax, ad: AnnulusData):
    """Each partition's raw S/N (score of its own image) at the evals that included it,
    with the combined raw score for reference."""
    ax.set_title("per-partition raw S/N of included partitions", loc="left")
    if not ad.multi:
        panel_score_hist(ax, ad)
        return
    n = ad.n
    ev = np.arange(1, n + 1)
    ax.plot(ev, ad.raw, color="#444444", lw=1.0, alpha=0.6, label="combined (raw)")
    M = ad.part_snr_matrix()
    inc = ad.inclusion()
    cmap = matplotlib.colormaps.get_cmap("tab10")
    for j, pid in enumerate(ad.partitions):
        v = np.where(inc[:, j], M[:, j], np.nan)
        if np.isfinite(v).any():
            ax.plot(ev, v, "o-", ms=2.5, lw=0.7, color=cmap(j % 10), label=str(pid))
    ax.set_xlabel("evaluation")
    ax.set_ylabel("raw S/N")
    ax.set_xlim(0.5, max(n, 1) + 0.5)
    ax.legend(loc="lower right", ncol=3, handletextpad=0.2, columnspacing=0.6)


def panel_importance_live(ax, ad: AnnulusData, cs: Optional[Dict[str, Any]] = None):
    """IDL ``mwcm_pimppanel`` (single-night optimizer, in place of the night-inclusion
    panel): horizontal bars of |corr(parameter, S/N)| over the finite evaluations,
    strongest first, value printed at the bar end."""
    ax.set_title("parameter importance", fontsize=7.5)
    ax.grid(False)
    cs = cs or ad.corner(None)
    y = np.asarray(cs["y"], float)
    X = np.asarray(cs["X"], float)
    names = list(cs["names"])
    g = np.isfinite(y)
    if X.ndim != 2 or X.shape[1] == 0 or g.sum() < 3:
        ax.text(0.5, 0.5, "accumulating samples...", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        return
    imp = np.zeros(X.shape[1])
    for d in range(X.shape[1]):
        xv = X[g, d]
        if np.std(xv) > 0 and np.std(y[g]) > 0:
            imp[d] = abs(np.corrcoef(xv, y[g])[0, 1])
    imp[~np.isfinite(imp)] = 0.0
    order = np.argsort(-imp)
    nd = len(order)
    imax = max(float(imp.max()), 0.05)
    for row, d in enumerate(order):
        yb = nd - row
        ax.barh(yb, imp[d], height=0.7, color="#1e90ff")
        ax.text(imp[d] + imax * 0.02, yb, f"{imp[d]:.2f}", va="center", fontsize=6)
    ax.set_yticks(range(1, nd + 1))
    labels = _abbrev_unique(names)                                          # short, and never two alike
    ax.set_yticklabels([labels[d] for d in order[::-1]], fontsize=7)        # the cell is narrow
    ax.set_xlim(0, imax * 1.18)
    ax.set_ylim(0.25, nd + 0.75)
    ax.set_xlabel("|corr| with S/N")


def panel_partition_effect(ax, ad: AnnulusData):
    """near2m_npanel: per partition, the scores of evals that included it (dark) vs
    excluded it (grey) with group means."""
    ax.set_title("partition effect (dark = included, grey = excluded)", loc="left")
    v = ad.valid
    if not ad.multi or not v.any():
        ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
        return
    inc = ad.inclusion()
    rng = np.random.default_rng(0)
    for j, pid in enumerate(ad.partitions):
        for m, col, off in ((inc[:, j] & v, _fg(), -0.18), (~inc[:, j] & v, "#9a9a9a", 0.18)):
            if m.any():
                ax.scatter(j + 1 + off + rng.uniform(-0.08, 0.08, m.sum()), ad.y[m], s=9, color=col, zorder=3)
                mu = float(np.mean(ad.y[m]))
                ax.plot([j + 1 + off - 0.14, j + 1 + off + 0.14], [mu, mu], color=col, lw=2.2)
    ax.set_xticks(range(1, len(ad.partitions) + 1))
    ax.set_xticklabels(ad.partitions)
    ax.set_xlabel("partition")
    ax.set_ylabel("search score")


def panel_partition_count(ax, ad: AnnulusData):
    """near2m_nnpanel: score vs number of included partitions."""
    ax.set_title("score vs # partitions included", loc="left")
    v = ad.valid
    if not ad.multi or not v.any():
        ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
        return
    cnt = ad.inclusion().sum(1)
    rnd = ad.is_random()
    jit = 0.05 * ((np.arange(ad.n) % 7) - 3)
    m = v & rnd
    ax.scatter(cnt[m] + jit[m], ad.y[m], s=12, facecolor="none", edgecolor="#999999", label="random")
    m = v & ~rnd
    ax.scatter(cnt[m] + jit[m], ad.y[m], s=12, color=PHASE_COLORS["tpe"], label="guided")
    ax.set_xticks(range(1, len(ad.partitions) + 1))
    ax.set_xlabel("# partitions")
    ax.set_ylabel("search score")
    ax.legend(loc="best")


FM_COLOR = "#0072B2"


def panel_contrast(ax, curves: Sequence[Tuple[str, Dict[str, Any], Dict[str, Any]]], contrast: float,
                   title: Optional[str] = None, placeholder: Optional[str] = None):
    """near2m_cpanel: ``curves`` = ``[(label, curve_dict, style)]`` with ``curve_dict``
    holding ``r_as, curve[, sample_r, sample_c]``; dotted line = injection contrast.
    KLIP-FM cross-check curves (label containing ``KLIP-FM``) are drawn as markers +
    line so the two calibrations can be told apart; the title says when both are shown.
    ``placeholder`` (e.g. why the live preview could not be built) replaces the
    generic waiting text when there is nothing to draw."""
    has_fm = any("KLIP-FM" in lab for lab, cc, _ in curves if cc)
    if title is None:
        title = "5-sigma contrast (injection-calibrated" + (" vs KLIP-FM)" if has_fm else ")")
    ax.set_title(title, loc="left")
    any_c = False
    for lab, cc, st in curves:
        if not cc:
            continue
        r = np.array([np.nan if v is None else v for v in cc.get("r_as", [])], float)
        c = np.array([np.nan if v is None else v for v in cc.get("curve", [])], float)
        if r.size and np.isfinite(c).any() and (c[np.isfinite(c)] > 0).any():
            st2 = dict(st)
            if "KLIP-FM" in lab:
                st2.setdefault("marker", "o")
                st2.setdefault("ms", 3)
            ax.plot(r, np.where(c > 0, c, np.nan), label=lab, **st2)
            any_c = True
        sr = np.array([np.nan if v is None else v for v in cc.get("sample_r", [])], float)
        sc = np.array([np.nan if v is None else v for v in cc.get("sample_c", [])], float)
        if sr.size and np.isfinite(sc).any():
            ax.scatter(sr, sc, s=9, color=st.get("color", "#333333"), alpha=0.6)
    if any_c:
        ax.set_yscale("log")
        if np.isfinite(contrast) and contrast > 0:
            ax.axhline(contrast, color="#777777", ls=":", lw=0.8, label="injection contrast")
        ax.legend(loc="upper right")
    else:
        msg = "after first best / validation..."
        if placeholder:
            msg = f"no preview yet:\n{placeholder}\n\n(validated curve after the annulus)"
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=7.5, color="#555555",
                wrap=True)
    ax.set_xlabel("separation (arcsec)")
    ax.set_ylabel("5-sigma contrast")


def panel_text(ax, lines: Sequence[str], fontsize: float = 7.2, family: str = "monospace", fit: bool = True,
               min_fontsize: float = 4.8, linespacing: float = 1.25):
    """Monospace text block anchored top-left.  With ``fit`` the font is shrunk (down to
    ``min_fontsize``) so that the longest line fits the axes width and all lines fit
    its height -- so a wide per-partition config table is never clipped."""
    ax.set_axis_off()
    lines = list(lines)
    if fit and lines:
        try:
            fig = ax.figure
            bb = ax.get_position()
            w_pt = bb.width * fig.get_figwidth() * 72.0
            h_pt = bb.height * fig.get_figheight() * 72.0
            ncols = max(len(l) for l in lines)
            fs_w = w_pt / (0.62 * max(ncols, 1))            # DejaVu Sans Mono advance ~0.602 em
            fs_h = h_pt / (linespacing * 1.02 * len(lines))
            fontsize = float(np.clip(min(fontsize, fs_w, fs_h), min_fontsize, fontsize))
        except Exception:
            pass
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=fontsize,
            family=family, linespacing=linespacing)


# ----------------------------------------------------------------------------
# corner / landscapes
# ----------------------------------------------------------------------------
def _widen_flat(lo, hi):
    """Bounds with the pinned dimensions given a sliver of width.

    A searched dimension can arrive with ``lo == hi`` -- a partition the guard has fixed, a
    parameter whose range collapsed for this annulus -- and matplotlib warns on identical
    axis limits for every such cell of every panel.  On the 38-D beta Pic landscape that is
    thousands of "Attempting to set identical low and high xlims" lines in the stage log,
    which buries anything real.  Widening rather than silencing keeps the warning available
    for the cases it was meant for, and the cell still draws with its one value centred.
    """
    lo = np.asarray(lo, float).copy()
    hi = np.asarray(hi, float).copy()
    flat = ~(hi > lo)
    if flat.any():
        mid = 0.5 * (lo[flat] + hi[flat])
        pad = np.maximum(np.abs(mid) * 1e-3, 1e-6)
        lo[flat], hi[flat] = mid - pad, mid + pad
    return lo, hi


def _score_norm(y: np.ndarray, crange=None):
    fin = y[np.isfinite(y)]
    if crange is not None:
        lo, hi = crange
    elif fin.size >= 5:
        s = np.sort(fin)
        lo, hi = float(s[0]), float(max(s[int(0.97 * (s.size - 1))], s[-2]))
    elif fin.size:
        lo, hi = float(fin.min()), float(fin.max())
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    return mcolors.Normalize(lo, hi)


def _marg_hist(ax, vals, y, lo, hi, norm, cmap, best=None, cur=None, horizontal=False, nb=12):
    """near2m_marghist: histogram whose bars are coloured by the median score in the bin."""
    g = np.isfinite(vals) & np.isfinite(y)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if g.sum() < 2 or hi <= lo:
        return
    edges = np.linspace(lo, hi, nb + 1)
    idx = np.clip(np.floor((vals[g] - lo) / (edges[1] - edges[0])).astype(int), 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    for k in range(nb):
        if cnt[k] == 0:
            continue
        col = cmap(norm(np.median(y[g][idx == k])))
        if horizontal:
            ax.barh(0.5 * (edges[k] + edges[k + 1]), cnt[k], height=edges[k + 1] - edges[k], color=col, lw=0)
        else:
            ax.bar(0.5 * (edges[k] + edges[k + 1]), cnt[k], width=edges[k + 1] - edges[k], color=col, lw=0)
    for v, col, ls in ((best, _fg(), "-"), (cur, CUR_COLOR, "--")):
        if v is not None and np.isfinite(v):
            (ax.axhline if horizontal else ax.axvline)(v, color=col, ls=ls, lw=1.0)
    if horizontal:
        ax.set_ylim(lo, hi)
    else:
        ax.set_xlim(lo, hi)


#: markers per corner cell in the *live* step panel (the post-hoc books draw every point).
#: A cell this size saturates around here, so beyond it the extra markers are drawing time
#: spent on pixels that are already covered.
CORNER_MAX_POINTS = 6000


def _corner_subsample(y: np.ndarray, top: np.ndarray, max_points: Optional[int]) -> Optional[np.ndarray]:
    """Which rows to draw when there are more than a corner cell can show.

    A cell two inches across saturates long before 40,000 markers -- and a six-partition
    run past 7000 evaluations hands it exactly that, once per pair of parameters, on every
    live frame.  Past saturation the extra markers cost drawing time and add nothing a
    reader can see.  Every top-gamma point is kept, since those carry the result; the faint
    background is thinned with a fixed seed so the picture does not shimmer between frames.

    The two populations are thinned *separately and in proportion*, because they say
    different things: the faint grey cloud is where the search has been -- the warm-up
    trials and everything the optimizer tried and rejected -- and the coloured top-gamma
    set is where it settled.  Spending the whole budget on whichever is larger loses one of
    them completely: a six-night run past 7000 evaluations has ~12,000 top-gamma rows on
    its own, so an elite-first rule silently drew no grey points at all and the panel
    stopped showing the exploration.  Each population also has a floor, so a small one is
    never squeezed out by a large one, and the sample is drawn with a fixed seed so the
    picture does not shimmer between frames.

    Returns ``None`` when everything fits, so the post-hoc books draw every point.
    """
    n = int(len(y))
    if not max_points or n <= int(max_points):
        return None
    max_points = int(max_points)
    rng = np.random.default_rng(0)
    elite, rest = np.flatnonzero(top), np.flatnonzero(~top)
    if elite.size == 0 or rest.size == 0:               # only one population to thin
        only = elite if rest.size == 0 else rest
        return np.sort(rng.choice(only, max_points, replace=False))

    # proportional split, but neither population may fall below a quarter of the budget
    floor = max(int(0.25 * max_points), 1)
    k_top = int(round(max_points * elite.size / n))
    k_top = min(max(k_top, floor), max_points - floor)
    k_top, k_rest = min(k_top, elite.size), min(max_points - k_top, rest.size)
    k_top = min(k_top + (max_points - k_top - k_rest), elite.size)   # give back any slack

    keep_top = elite if k_top >= elite.size else rng.choice(elite, k_top, replace=False)
    keep_rest = rest if k_rest >= rest.size else rng.choice(rest, k_rest, replace=False)
    best = int(np.nanargmax(y)) if np.isfinite(y).any() else None
    out = np.concatenate([keep_top, keep_rest] + ([[best]] if best is not None else []))
    return np.unique(out.astype(int))


def draw_corner(fig, cs: Dict[str, Any], ad: AnnulusData, title: str, rect=(0.06, 0.06, 0.90, 0.82),
                warm: Optional[np.ndarray] = None, best_pt: Optional[np.ndarray] = None,
                cur_pt: Optional[np.ndarray] = None, cands: Sequence[Tuple[np.ndarray, bool]] = (),
                top_mask: Optional[np.ndarray] = None, crange=None, compact: bool = False,
                max_points: Optional[int] = None):
    """near2m_corner: lower-triangle pairwise scatter, diagonal marginal histograms coloured
    by the median score per bin, best (square), current (dashed cross-hairs), validation
    candidates (triangles, winner = star) and a score colourbar.

    Every point is coloured by its score; the marker is filled for a guided evaluation and
    open for a warm-up one.  Colour therefore means one thing throughout and the fill means
    another, rather than colour standing in for the top-gamma cut -- which during warm-up
    left a few arbitrary points coloured among grey ones that were no different in kind.
    ``top_mask`` is still used to decide what survives thinning on a long run."""
    names, lo, hi, X, y = cs["names"], cs["lo"], cs["hi"], cs["X"], cs["y"]
    lo, hi = _widen_flat(lo, hi)
    d = len(names)
    if X.shape[0] < 1 or d == 0:
        fig.text(0.5, 0.5, "collecting...", ha="center", va="center")
        return
    norm = _score_norm(y, crange)
    cmap = matplotlib.colormaps.get_cmap(SCORE_CMAP)
    warm = np.zeros(len(y), bool) if warm is None else warm
    top = np.ones(len(y), bool) if top_mask is None else top_mask
    keep = _corner_subsample(y, top, max_points)
    if keep is not None:                 # colour scale stays that of the full sample
        X, y, top, warm = X[keep], y[keep], top[keep], warm[keep]
    x0, y0, w, h = rect
    gs = GridSpec(d, d, left=x0, bottom=y0, right=x0 + w, top=y0 + h, wspace=0.08, hspace=0.08, figure=fig)
    ms = float(np.clip(np.sqrt(80.0 / max(len(y), 1)), 0.3, 1.0))
    for i in range(d):          # row (y param)
        for j in range(d):      # col (x param)
            if j > i:
                continue
            ax = fig.add_subplot(gs[i, j])
            if i == j:
                bi_v = None if best_pt is None else best_pt[i]
                cu_v = None if cur_pt is None else cur_pt[i]
                # the bottom-row marginal is rotated so it shares that row's y axis (IDL corner)
                _marg_hist(ax, X[:, i], y, lo[i], hi[i], norm, cmap, bi_v, cu_v, horizontal=(i == d - 1))
            else:
                ax.grid(False)
                # One channel per thing.  Colour is the score, for every point; fill is the
                # phase -- filled = guided, open = warm-up.  The older scheme spent colour
                # on the top-gamma cut, so during warm-up (when every point is random) the
                # only coloured markers were the handful above an arbitrary quantile and
                # everything else went grey, which read as "these few were special" when
                # nothing had guided them.  The colourmap already shows which scores are
                # good, and continuously; the gamma cut needs no second encoding.
                m = ~warm
                if m.any():
                    ax.scatter(X[m, j], X[m, i], s=14 * ms, c=y[m], cmap=cmap, norm=norm,
                               lw=0, zorder=3)
                if warm.any():
                    ax.scatter(X[warm, j], X[warm, i], s=14 * ms, facecolor="none",
                               edgecolor=cmap(norm(y[warm])), lw=0.6, zorder=2)
                if cur_pt is not None and np.isfinite(cur_pt[j]) and np.isfinite(cur_pt[i]):
                    ax.axvline(cur_pt[j], color=CUR_COLOR, ls="--", lw=0.7)
                    ax.axhline(cur_pt[i], color=CUR_COLOR, ls="--", lw=0.7)
                if best_pt is not None and np.isfinite(best_pt[j]) and np.isfinite(best_pt[i]):
                    ax.scatter([best_pt[j]], [best_pt[i]], s=60, marker="s", facecolor="none", edgecolor=_fg(), lw=1.3, zorder=5)
                for cp, is_win in cands:
                    if cp is None:
                        continue
                    ax.scatter([cp[j]], [cp[i]], s=90 if is_win else 45, marker="*" if is_win else "^",
                               facecolor="none", edgecolor=VALID_COLOR, lw=1.2, zorder=6)
                ax.set_xlim(lo[j], hi[j])
                ax.set_ylim(lo[i], hi[i])
            ax.tick_params(labelsize=5.5 if compact else 6)
            if compact:
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
                ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            lab = (lambda nm: COL_ABBREV.get(nm, nm)) if compact else (lambda nm: nm)
            lfs = 6.5 if compact else 7.5
            if i < d - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(lab(names[j]), fontsize=lfs)
            if j > 0 or i == j:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(lab(names[i]), fontsize=lfs)
            if i == j:
                ax.set_ylabel("")
                if i == d - 1:                 # rotated: values on y (shared with the row), counts on x
                    ax.set_ylim(lo[i], hi[i])
                    ax.set_xticks([]); ax.set_xticklabels([])
                    ax.set_xlabel(lab(names[i]), fontsize=lfs)
                else:
                    ax.set_xlim(lo[i], hi[i])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    if compact:      # IDL live panel: title over a centred colourbar, no legend line
        cax = fig.add_axes([x0 + 0.22 * w, y0 + h + 0.022, 0.56 * w, 0.011])
        fig.colorbar(sm, cax=cax, orientation="horizontal")
        cax.tick_params(labelsize=6, pad=1)
        fig.text(x0 + 0.5 * w, y0 + h + 0.040, title, fontsize=7.5, ha="center", va="bottom")
        return
    cax = fig.add_axes([x0 + 0.30 * w, y0 + h + 0.028, 0.70 * w, 0.012])
    fig.colorbar(sm, cax=cax, orientation="horizontal")
    cax.tick_params(labelsize=6, pad=1)
    fig.text(x0 + 0.29 * w, y0 + h + 0.034, "search score\n(upward-biased)", fontsize=6.5, ha="right", va="center")
    fig.text(x0, y0 + h + 0.085, title, fontsize=9, va="bottom")
    leg = (f"metric: {ad.metric_name}.  colour = search score, filled = guided, hollow = warm-up;  square = best,\n"
           "dashed = current, triangle = validation candidate, star = validated winner")
    fig.text(x0, y0 + h + 0.052, leg, fontsize=6.5, va="bottom", color=_grey("#444444", "#bbbbbb"))


def _nw_bandwidth(v, lo, hi, ng=24):
    """Kernel width for one axis of :func:`_nw_grid`.

    ``span/8`` is right for a search that samples an axis densely, and blanks the figure
    for one that does not.  A grid run with two values per axis puts its samples a full
    span apart -- eight bandwidths -- where the Gaussian weight is ``exp(-32)``, so every
    cell failed the support test except a few hard against the panel edge and the
    landscape rendered as one mark on black.  Widening to half the typical spacing
    between the values an axis actually took keeps dense searches exactly as they were
    (their spacing is far below ``span/8``) and makes a coarse design legible instead of
    invisible.
    """
    span = float(hi - lo)
    base = span / 8.0
    u = np.unique(np.asarray(v, float)[np.isfinite(v)])
    if u.size >= 2:
        base = max(base, 0.5 * float(np.median(np.diff(u))))
    return max(base, span / max(ng, 1), 1e-6)


def _nw_grid(x, y, v, xlo, xhi, ylo, yhi, ng=24):
    """near2m_nwgrid: Nadaraya-Watson kernel average of ``v`` on an ``ng x ng`` grid."""
    gx = xlo + (xhi - xlo) * (np.arange(ng) + 0.5) / ng
    gy = ylo + (yhi - ylo) * (np.arange(ng) + 0.5) / ng
    hx = _nw_bandwidth(x, xlo, xhi, ng)
    hy = _nw_bandwidth(y, ylo, yhi, ng)
    wx = np.exp(-0.5 * ((x[None, :] - gx[:, None]) / hx) ** 2)      # (ng, n)
    wy = np.exp(-0.5 * ((y[None, :] - gy[:, None]) / hy) ** 2)
    W = wy[:, None, :] * wx[None, :, :]                              # (ngy, ngx, n)
    sw = W.sum(-1)
    out = (W * v[None, None, :]).sum(-1) / np.where(sw > 0, sw, np.nan)
    out[sw <= 0.5] = np.nan
    return out


def draw_kde_corner(fig, cs: Dict[str, Any], title: str, rect=(0.06, 0.06, 0.90, 0.84),
                    best_pt: Optional[np.ndarray] = None):
    """near2m_kdecorner: kernel-smoothed score landscape per parameter pair."""
    names, lo, hi, X, y = cs["names"], cs["lo"], cs["hi"], cs["X"], cs["y"]
    lo, hi = _widen_flat(lo, hi)
    d = len(names)
    g = np.isfinite(y)
    if g.sum() < 6 or d < 2:
        fig.text(0.5, 0.5, "collecting... (needs >= 6 evals and >= 2 parameters)", ha="center", va="center")
        return
    x0, y0, w, h = rect
    gs = GridSpec(d - 1, d - 1, left=x0, bottom=y0, right=x0 + w, top=y0 + h, wspace=0.08, hspace=0.08, figure=fig)
    grids = {}
    for i in range(1, d):
        for j in range(i):
            grids[(i, j)] = _nw_grid(X[g, j], X[g, i], y[g], lo[j], hi[j], lo[i], hi[i])
    allv = np.concatenate([v[np.isfinite(v)] for v in grids.values()]) if grids else np.zeros(0)
    vmin, vmax = (float(allv.min()), float(allv.max())) if allv.size else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1
    cm = matplotlib.colormaps.get_cmap(SCORE_CMAP).copy()
    cm.set_bad("#111111")
    im = None
    for (i, j), grid in grids.items():
        ax = fig.add_subplot(gs[i - 1, j])
        ax.grid(False)
        im = ax.imshow(grid, origin="lower", extent=[lo[j], hi[j], lo[i], hi[i]], aspect="auto", cmap=cm,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        # The design itself, so a coarse one reads as "four configurations were tried"
        # rather than as an empty panel.  Marker area tracks how many evaluations landed
        # on each position: a grid run stacks hundreds of evaluations on one node.
        ux, iu = np.unique(np.column_stack([X[g, j], X[g, i]]), axis=0, return_inverse=True)
        if ux.shape[0] <= 400:
            mult = np.bincount(iu, minlength=ux.shape[0]).astype(float)
            s = 3.0 + 9.0 * np.sqrt(mult / max(mult.max(), 1.0))
            ax.scatter(ux[:, 0], ux[:, 1], s=s, facecolor="none", edgecolor="white", lw=0.45, alpha=0.75)
        if best_pt is not None and np.isfinite(best_pt[j]) and np.isfinite(best_pt[i]):
            ax.scatter([best_pt[j]], [best_pt[i]], s=60, marker="s", facecolor="none", edgecolor="white", lw=1.3)
        ax.set_xlim(lo[j], hi[j])
        ax.set_ylim(lo[i], hi[i])
        ax.tick_params(labelsize=6)
        if i < d - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(names[j], fontsize=7.5)
        if j > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(names[i], fontsize=7.5)
    if im is not None:
        cax = fig.add_axes([x0 + 0.45 * w, y0 + h + 0.035, 0.55 * w, 0.014])
        fig.colorbar(im, cax=cax, orientation="horizontal").set_label("kernel-averaged search score", fontsize=7)
        cax.tick_params(labelsize=6)
    fig.text(x0, y0 + h + 0.06, title, fontsize=9, va="bottom")
    fig.text(x0, y0 + h + 0.015,
             "black = unsupported; circles = evaluated (area ~ count); square = best",
             fontsize=6.0, color="#444444")


def draw_walk(fig, cs: Dict[str, Any], ad: AnnulusData, title: str, rect=(0.07, 0.06, 0.90, 0.86)):
    """near2m_pnwalk: chronological parameter walk per pair, random segments grey and
    guided segments in the TPE colour; the running-best path overlaid in orange."""
    names, lo, hi, X = cs["names"], cs["lo"], cs["hi"], cs["X"]
    lo, hi = _widen_flat(lo, hi)          # pinned dimensions: no "identical xlims" warnings
    ev = cs["eval"]
    d = len(names)
    if X.shape[0] < 2 or d < 2:
        fig.text(0.5, 0.5, "collecting...", ha="center", va="center")
        return
    # one row per eval (first partition row) for the walk
    first = np.array([k for k in range(len(ev)) if k == 0 or ev[k] != ev[k - 1]], int)
    Xw, evw = X[first], ev[first]
    rnd = np.array([_is_random_phase(ad.phases[e]) for e in evw])
    yb = ad.running_best()
    imp = np.array([np.isfinite(ad.y[e]) and ad.y[e] >= yb[e] and (e == 0 or not np.isfinite(yb[e - 1]) or ad.y[e] > yb[e - 1])
                    for e in evw])
    x0, y0, w, h = rect
    gs = GridSpec(d - 1, d - 1, left=x0, bottom=y0, right=x0 + w, top=y0 + h, wspace=0.08, hspace=0.08, figure=fig)
    for i in range(1, d):
        for j in range(i):
            ax = fig.add_subplot(gs[i - 1, j])
            ax.grid(False)
            for k in range(len(evw) - 1):
                ax.plot(Xw[k:k + 2, j], Xw[k:k + 2, i], color="#bbbbbb" if rnd[k + 1] else PHASE_COLORS["tpe"],
                        lw=0.9, alpha=0.8)
            if imp.sum() >= 2:
                ax.plot(Xw[imp, j], Xw[imp, i], "o-", color=PHASE_COLORS["local"], ms=3, lw=1.2)
            ax.scatter([Xw[-1, j]], [Xw[-1, i]], s=30, marker="s", facecolor="none", edgecolor="black", lw=1.0)
            ax.set_xlim(lo[j], hi[j])
            ax.set_ylim(lo[i], hi[i])
            ax.tick_params(labelsize=6)
            if i < d - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(names[j], fontsize=7.5)
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(names[i], fontsize=7.5)
    fig.text(x0, y0 + h + 0.03, title, fontsize=9, va="bottom")
    fig.text(x0, y0 + h + 0.005, "grey = random step, blue = guided step, orange = running-best path, square = last eval",
             fontsize=6.5, color="#444444")


def _best_pt(ad: AnnulusData, cs: Dict[str, Any], pid: Optional[str] = None) -> Optional[np.ndarray]:
    """Best eval's coordinates; when the overall best did not include ``pid``, the best
    row of this partition's own view."""
    bi, _ = ad.best()
    p = ad.eval_point(bi, cs, pid)
    if p is None and len(cs["y"]) and np.isfinite(cs["y"]).any():
        p = cs["X"][int(np.nanargmax(cs["y"]))]
    return p


def _cands(ad: AnnulusData, cs: Dict[str, Any], pid: Optional[str] = None):
    out = []
    if not ad.validation:
        return out
    win_idx = None
    if ad.winner is not None and ad.winner.get("validated"):
        win_idx = int(ad.winner["winner_index"])
    for row in ad.validation:
        e = int(row.get("eval_index", -1))
        out.append((ad.eval_point(e, cs, pid), e == win_idx))
    return out


def render_corner(ad: AnnulusData, out_pdf: str, current: Optional[int] = None) -> str:
    """Multi-page corner book: page 1 = per-base corner (per-partition slots pooled);
    then one page per partition with its own block (near2m_corner / pnlandscapes)."""
    with _rc(), PdfPages(out_pdf) as pdf:
        pages = [(None, "per-base view (per-partition slots pooled)")]
        if ad.replicated:
            pages += [(pid, f"partition {pid}") for pid in ad.partitions]
        for pid, lab in pages:
            cs = ad.corner(pid)
            fig = Figure(figsize=(8.0, 9.0))
            FigureCanvasAgg(fig)
            top_rows = np.zeros(len(cs["y"]), bool)
            tm = ad.top_mask()
            for r, e in enumerate(cs["eval"]):
                top_rows[r] = tm[e]
            warm = np.array([_is_random_phase(ad.phases[e]) for e in cs["eval"]], bool)
            ttl = f"{ad.run_name}  annulus {ad.annulus + 1}: search-score landscape -- {lab}"
            draw_corner(fig, cs, ad, ttl, warm=warm, best_pt=_best_pt(ad, cs, pid),
                        cur_pt=None if current is None else ad.eval_point(current, cs, pid),
                        cands=_cands(ad, cs, pid), top_mask=top_rows)
            pdf.savefig(fig)
    return out_pdf


def render_landscapes(ad: AnnulusData, out_pdf: str) -> str:
    """Per-partition landscape book: kernel-smoothed score landscape and the parameter
    walk for each partition (or the whole space when not replicated)."""
    with _rc(), PdfPages(out_pdf) as pdf:
        pages = [(pid, f"partition {pid}") for pid in ad.partitions] if ad.replicated else [(None, "all parameters")]
        for pid, lab in pages:
            cs = ad.corner(pid)
            for kind in ("kde", "walk"):
                fig = Figure(figsize=(8.0, 9.0))
                FigureCanvasAgg(fig)
                if kind == "kde":
                    draw_kde_corner(fig, cs, f"{ad.run_name} annulus {ad.annulus + 1}: kernel-smoothed landscape -- {lab}",
                                    best_pt=_best_pt(ad, cs, pid))
                else:
                    draw_walk(fig, cs, ad, f"{ad.run_name} annulus {ad.annulus + 1}: parameter walk -- {lab}")
                pdf.savefig(fig)
    return out_pdf


# ----------------------------------------------------------------------------
# parameter-histogram book, EDF, nightmap, k-book
# ----------------------------------------------------------------------------
def _parhist_blocks(d: int) -> Tuple[int, int]:
    """Column blocks (each a history / distribution / score triplet) and rows per block
    for ``d`` parameters.  One row per parameter stacked in a single block makes a page
    taller than it is wide as soon as d > 4 -- a nine-parameter annulus came out 8 x 18
    inches, which is unreadable once it is scaled to a journal text width.  Two or three
    blocks side by side keep the page close to landscape."""
    nblk = 1 if d <= 4 else (2 if d <= 10 else 3)
    return nblk, int(np.ceil(d / nblk))


def parhist_figsize(d: int) -> Tuple[float, float]:
    """Page size for :func:`_parhist_page` -- landscape-ish, so the book is legible at
    one column or one text width rather than a strip eighteen inches tall."""
    nblk, nrow = _parhist_blocks(max(d, 1))
    return 4.35 * nblk + 0.5, max(3.6, 1.50 * nrow + 0.85)


def _parhist_page(fig, ad: AnnulusData, cs: Dict[str, Any], title: str, current: Optional[int] = None):
    """near2m_parhist_page: one row per parameter -- history (value vs eval), distribution,
    score vs value; best value red, current dashed grey.  Parameters run down a block and
    then into the next block to the right, so the page stays roughly landscape."""
    names, lo, hi, X, y, ev = cs["names"], cs["lo"], cs["hi"], cs["X"], cs["y"], cs["eval"]
    lo, hi = _widen_flat(lo, hi)          # pinned dimensions: no "identical xlims" warnings
    d = len(names)
    if d == 0 or X.shape[0] == 0:
        fig.text(0.5, 0.5, "collecting...", ha="center", va="center")
        return
    nblk, nrow = _parhist_blocks(d)
    outer = GridSpec(1, nblk, left=0.055, right=0.985, bottom=0.055, top=0.905, wspace=0.30, figure=fig)
    inner = [GridSpecFromSubplotSpec(nrow, 3, subplot_spec=outer[0, b], hspace=0.68, wspace=0.42)
             for b in range(nblk)]
    bp = _best_pt(ad, cs, None if not cs["pid"] or cs["pid"][0] is None else cs["pid"][0])
    cp = None if current is None else ad.eval_point(current, cs)
    rnd = np.array([_is_random_phase(ad.phases[e]) for e in ev])
    fig.text(0.008, 0.978, title, fontsize=9)
    fig.text(0.008, 0.952, "red = best value, dashed = current; open grey = warm-up, filled = guided", fontsize=7.5)
    for k in range(d):
        gs, r = inner[k // nrow], k % nrow
        v = X[:, k]
        a1 = fig.add_subplot(gs[r, 0])
        a1.scatter(ev[rnd] + 1, v[rnd], s=8, facecolor="none", edgecolor="#999999", lw=0.7)
        a1.scatter(ev[~rnd] + 1, v[~rnd], s=8, color="#333333")
        a1.set_ylim(lo[k], hi[k])
        a1.set_xlim(0.5, ad.n + 0.5)
        a1.set_ylabel(names[k])
        a1.set_xlabel("evaluation")
        a2 = fig.add_subplot(gs[r, 1])
        g = np.isfinite(v)
        if g.sum() >= 2:
            a2.hist(v[g], bins=np.linspace(lo[k], hi[k], 13), color="#555555")
        a2.set_xlim(lo[k], hi[k])
        a2.set_xlabel(names[k])
        a2.set_ylabel("N evals")
        a3 = fig.add_subplot(gs[r, 2])
        a3.scatter(v[rnd], y[rnd], s=8, facecolor="none", edgecolor="#999999", lw=0.7)
        a3.scatter(v[~rnd], y[~rnd], s=8, color=PHASE_COLORS["tpe"])
        a3.set_xlim(lo[k], hi[k])
        a3.set_xlabel(names[k])
        a3.set_ylabel("search score")
        for ax_, hv in ((a1, "h"), (a2, "v"), (a3, "v")):
            if bp is not None and np.isfinite(bp[k]):
                (ax_.axhline if hv == "h" else ax_.axvline)(bp[k], color="#d62728", lw=1.1)
            if cp is not None and np.isfinite(cp[k]):
                (ax_.axhline if hv == "h" else ax_.axvline)(cp[k], color=CUR_COLOR, lw=0.9, ls="--")
        if r == 0:
            a1.set_title("history")
            a2.set_title("distribution")
            a3.set_title("score vs value")


def render_parhist_book(ad: AnnulusData, out_pdf: str, current: Optional[int] = None) -> str:
    with _rc(), PdfPages(out_pdf) as pdf:
        pages = [(pid, f"partition {pid}") for pid in ad.partitions] if ad.replicated else [(None, "all parameters")]
        for pid, lab in pages:
            cs = ad.corner(pid)
            fig = Figure(figsize=parhist_figsize(len(cs["names"])))
            FigureCanvasAgg(fig)
            _parhist_page(fig, ad, cs, f"{ad.run_name} annulus {ad.annulus + 1}: parameter histories -- {lab}", current)
            pdf.savefig(fig)
    return out_pdf


def draw_edf(ax, ad: AnnulusData):
    """near2m_edf: empirical CDF of the search score, warm-up vs guided, winner marked."""
    v = ad.valid
    rnd = ad.is_random()
    for lab, m, col in (("random / warm-up", v & rnd, "#999999"), ("guided", v & ~rnd, PHASE_COLORS["tpe"]),
                        ("all", v, "#000000")):
        if m.sum() < 1:
            continue
        s = np.sort(ad.y[m])
        ax.step(s, np.arange(1, s.size + 1) / s.size, where="post", color=col, lw=1.4 if lab != "all" else 0.8,
                ls="-" if lab != "all" else ":", label=f"{lab} (n={s.size})")
    if ad.winner is not None:
        ws = float(ad.winner["winner_score"])
        ax.axvline(ws, color=VALID_COLOR if ad.winner.get("validated") else "#444444", ls="--", lw=1.0,
                   label=("validated winner" if ad.winner.get("validated") else "winner (search score, NOT validated)") + f" {ws:.2f}")
    ax.set_xlabel(f"search score (upward-biased)  [{ad.metric_name}]")
    ax.set_ylabel("cumulative fraction of evaluations")
    ax.set_title(f"{ad.run_name} annulus {ad.annulus + 1}: objective EDF", loc="left")
    ax.legend(loc="lower right")


def render_edf(ad: AnnulusData, out: str) -> str:
    with _rc():
        fig = Figure(figsize=(6.5, 4.4))
        FigureCanvasAgg(fig)
        draw_edf(fig.add_subplot(111), ad)
        fig.savefig(out, bbox_inches="tight")
    return out


def render_nightmap(ad: AnnulusData, out: str) -> str:
    """near2m_nightmap: inclusion matrix + partition effect + score vs # partitions."""
    with _rc():
        fig = Figure(figsize=(8.0, 7.0))
        FigureCanvasAgg(fig)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1], hspace=0.45, wspace=0.3, left=0.09, right=0.97,
                      top=0.93, bottom=0.08)
        panel_inclusion(fig.add_subplot(gs[0, :]), ad)
        panel_partition_effect(fig.add_subplot(gs[1, 0]), ad)
        panel_partition_count(fig.add_subplot(gs[1, 1]), ad)
        fig.suptitle(f"{ad.run_name} annulus {ad.annulus + 1}: partition inclusion history", fontsize=9)
        fig.savefig(out)
    return out


def render_kbook(ad: AnnulusData, out_pdf: str) -> str:
    """k-klip book (near2m_kklip_book without scans): page 1 score vs k (per partition),
    page 2 the k history per partition coloured by phase."""
    with _rc(), PdfPages(out_pdf) as pdf:
        fig = Figure(figsize=(7.0, 5.0))
        FigureCanvasAgg(fig)
        panel_k(fig.add_subplot(111), ad)
        fig.suptitle(f"{ad.run_name} annulus {ad.annulus + 1}: score vs k_klip", fontsize=9)
        pdf.savefig(fig)
        if ad.has_k():
            fig = Figure(figsize=(7.0, 5.0))
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            ev = np.arange(1, ad.n + 1)
            cmap = matplotlib.colormaps.get_cmap("tab10")
            parts = ad.partitions if (ad.replicated and ad.partitions) else [None]
            for j, pid in enumerate(parts):
                ks = np.array([ad.k_partition(i, pid) if pid is not None else ad.k_scalar(i) for i in range(ad.n)])
                if pid is not None:
                    ks = np.where(ad.inclusion()[:, j], ks, np.nan)
                ax.plot(ev, ks, "o-", ms=3, lw=0.8, color=cmap(j % 10), label=str(pid) if pid is not None else "k")
            bi, _ = ad.best()
            if bi >= 0:
                ax.axvline(bi + 1, color="#009E73", lw=1.0, ls="--", label="best eval")
            if ad.n_init and ad.n > ad.n_init:
                ax.axvline(ad.n_init + 0.5, color="#999999", ls=":", lw=0.8)
            ax.set_xlabel("evaluation")
            ax.set_ylabel("k_klip")
            ax.legend(loc="best", ncol=3)
            ax.set_title(f"{ad.run_name} annulus {ad.annulus + 1}: k history per partition", loc="left")
            pdf.savefig(fig)
    return out_pdf


# ----------------------------------------------------------------------------
# calibration / validation / intro
# ----------------------------------------------------------------------------
def render_calibration(ad: AnnulusData, info: Dict[str, Any], out: str, clean: Optional[np.ndarray] = None,
                       inj: Optional[np.ndarray] = None, sources=None, per_source=None, clean_per_source=None,
                       cmap: str = "inferno", target=(4.0, 6.0)) -> str:
    """near2m_calshow: default-config clean / injected images with the sources marked,
    the k-scan curve (chosen k marked) and the contrast-trial table."""
    with _rc():
        fig = Figure(figsize=(12.0, 4.8))
        FigureCanvasAgg(fig)
        gs = GridSpec(2, 4, figure=fig, width_ratios=[1.2, 1.2, 1, 1.1], left=0.05, right=0.98, top=0.86, bottom=0.12,
                      wspace=0.35, hspace=0.6)
        note = None if inj is not None else "no images (seed eval not run)"
        draw_image(fig.add_subplot(gs[:, 0]), clean, ad, "default config -- clean (no injection)", cmap,
                   sources=sources, labels=clean_per_source, note=note)
        draw_image(fig.add_subplot(gs[:, 1]), inj, ad, "default config -- injected (per-source metric)", cmap,
                   sources=sources, labels=per_source, note=note)
        ax = fig.add_subplot(gs[0, 2:])
        ks = info.get("kscan")
        if ks:
            kv = np.array([np.nan if v is None else v for v in ks], float)
            kk = np.arange(1, kv.size + 1)
            ax.plot(kk, kv, "o-", ms=3, lw=1.0, color=PHASE_COLORS["tpe"])
            kd = info.get("k_default")
            if kd:
                ax.axvline(kd, color=VALID_COLOR, ls="--", lw=1.0, label=f"chosen k = {kd}")
                ax.legend(loc="lower right")
            ax.set_xlabel("k_klip")
            ax.set_ylabel("raw score")
            ax.set_title("k-scan of the default config", loc="left")
        else:
            ax.text(0.5, 0.5, "no k-scan (k fixed or scan disabled)", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        ax = fig.add_subplot(gs[1, 2:])
        trials = info.get("trials", [])
        lines = ["trial   contrast    median S/N   values"]
        for t, tr in enumerate(trials):
            vals = ", ".join(_fnum(None if v is None else float(v)) for v in tr.get("values", []))
            lines.append(f"{t + 1:>5d}   {tr.get('contrast', np.nan):.3e}   {_fnum(tr.get('snr')):>8s}     [{vals}]")
        lines.append("")
        lines.append(f"target window {target[0]:.1f} - {target[1]:.1f}   forced = {info.get('forced', 0) or 0:g}")
        lines.append(f"final contrast {info.get('contrast', np.nan):.3e}   median S/N {_fnum(info.get('snr'))}")
        panel_text(ax, lines, fontsize=7.5)
        ax.set_title("contrast-targeting trials", loc="left")
        fig.suptitle(f"{ad.run_name}: injection-contrast calibration -- annulus {ad.annulus + 1}/{ad.nann}  "
                     f"[{ad.inrad:.1f}, {ad.outrad:.1f}] px   metric: {ad.metric_name}", fontsize=10)
        fig.savefig(out)
    return out


def render_validation(ad: AnnulusData, table: List[Dict[str, Any]], out: str, inj: Optional[np.ndarray] = None,
                      clean: Optional[np.ndarray] = None, sources=None, cmap: str = "inferno",
                      winner_eval: Optional[int] = None, per_source=None, fm: Optional[np.ndarray] = None) -> str:
    """Validation panel: per candidate the search score (hollow) vs the validated
    median (filled) with the trial spread; committed winner thumbnails (the injected
    one labelled with the committed trial's raw per-source S/N) and, when the KLIP-FM
    cross-check ran, the forward-model response image."""
    with _rc():
        fig = Figure(figsize=(12.0, 5.6))
        FigureCanvasAgg(fig)
        gs = GridSpec(1, 4, figure=fig, width_ratios=[1.6, 1.2, 1, 1], left=0.05, right=0.98, top=0.86, bottom=0.14, wspace=0.35)
        ax = fig.add_subplot(gs[0, 0])
        if not table:
            ax.text(0.5, 0.5, "no validation (n_valid = 0)", ha="center", va="center", transform=ax.transAxes)
        for ci, row in enumerate(table):
            s = row.get("search_score")
            v = row.get("validated_score")
            tr = np.array([np.nan if t is None else t for t in row.get("trials", [])], float)
            e = int(row.get("eval_index", -1))
            is_win = winner_eval is not None and e == winner_eval
            ax.scatter([ci], [s], s=50, facecolor="none", edgecolor=PHASE_COLORS["tpe"], lw=1.3, zorder=3,
                       label="search score (upward-biased)" if ci == 0 else None)
            if np.isfinite(tr).any():
                ax.scatter(np.full(tr.size, ci) + 0.12, tr, s=10, color="#999999", zorder=2,
                           label="validation trials (raw metric)" if ci == 0 else None)
            if v is not None and np.isfinite(v):
                ax.scatter([ci], [v], s=80 if is_win else 50, color=VALID_COLOR, edgecolor="black" if is_win else "none",
                           lw=1.2, zorder=4, label=VALID_LABEL if ci == 0 else None)
                ax.annotate("", xy=(ci, v), xytext=(ci, s), arrowprops=dict(arrowstyle="->", color="#777777", lw=0.7))
            else:
                ax.scatter([ci], [s], marker="x", color="#d62728", s=40, zorder=5)
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels([f"cand {c + 1}\neval {int(r.get('eval_index', -1)) + 1}"
                            + ("\n(winner)" if winner_eval is not None and int(r.get("eval_index", -1)) == winner_eval else "")
                            for c, r in enumerate(table)])
        ax.set_xlim(-0.6, max(len(table) - 1, 0) + 0.6)
        ax.set_ylabel(f"score  [{ad.metric_name}]")
        ax.set_title("search vs validated score per candidate", loc="left")
        ax.legend(loc="best")
        ax2 = fig.add_subplot(gs[0, 1])
        lines = ["cand  eval   search   valid    trials"]
        for ci, row in enumerate(table):
            tr = ", ".join(_fnum(None if t is None else float(t), 1) for t in row.get("trials", []))
            lines.append(f"{ci + 1:>4d}  {int(row.get('eval_index', -1)) + 1:>4d}   {_fnum(row.get('search_score')):>6s}   "
                         f"{_fnum(row.get('validated_score')):>6s}   [{tr}]")
        lines.append("")
        lines.append("winner = argmax validated median; the search")
        lines.append("maximum is optimistic (best of noisy draws).")
        panel_text(ax2, lines, fontsize=7)
        draw_image(fig.add_subplot(gs[0, 2]), inj, ad, "winner: committed injected trial (raw S/N)", cmap,
                   sources=sources, labels=per_source, note=None if inj is not None else "no committed image")
        if fm is not None:
            sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 3], hspace=0.5)
            draw_image(fig.add_subplot(sub[0]), clean, ad, "winner: clean reduction", cmap,
                       note=None if clean is not None else "no clean image")
            draw_image(fig.add_subplot(sub[1]), fm, ad, "winner: KLIP-FM response (test spiral)", cmap,
                       flatten=False, vrange=_fm_range(fm))
        else:
            draw_image(fig.add_subplot(gs[0, 3]), clean, ad, "winner: clean reduction", cmap,
                       note=None if clean is not None else "no clean image")
        fig.suptitle(f"{ad.run_name}: validation -- annulus {ad.annulus + 1}/{ad.nann}", fontsize=10)
        fig.savefig(out)
    return out


def render_intro(setup: Dict[str, Any], out: str, size=(16.0, 10.0), dpi: int = 110) -> str:
    """Title card (the calm replacement of near2m_intro_frame): run setup summary."""
    cfg = setup.get("config", {}) or {}
    space = setup.get("space", {}) or {}
    with _rc():
        fig = Figure(figsize=size, dpi=dpi, facecolor="#101418")
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        name = os.path.basename(os.path.abspath(setup.get("run_dir", "run")))
        ax.text(0.5, 0.80, "KLIP-TPE", ha="center", va="center", fontsize=44, color="#e8e8e8", weight="bold")
        ax.text(0.5, 0.70, f"{name}", ha="center", va="center", fontsize=20, color="#9ecae1")
        edges = cfg.get("ann_edges", [])
        red = setup.get("reducer", {}) or {}
        parts = space.get("partitions", [])
        lines = [
            f"annuli: {max(len(edges) - 1, 1)}   edges (px): {', '.join(f'{e:g}' for e in edges)}",
            f"search: {cfg.get('search_mode', '?')}   n_iter {cfg.get('n_iter')}   n_init {cfg.get('n_init')}   gamma {cfg.get('gamma')}",
            f"reducer: {red.get('name', '?')}   partitions: {', '.join(parts) if parts else 'single'}   selection: {space.get('selection')}",
            f"searched: {', '.join(sorted({(p.get('base') or p['name']) for p in space.get('params', [])}))}",
            f"metric: {(setup.get('objective', {}) or {}).get('metric', {}).get('name', '?')}"
            f"{'  (clean-subtracted)' if (setup.get('objective', {}) or {}).get('clean_subtract') else ''}",
            f"seed {cfg.get('seed')}   {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]
        ax.text(0.5, 0.45, "\n".join(lines), ha="center", va="center", fontsize=14, color="#d0d0d0",
                family="monospace", linespacing=1.7)
        ax.text(0.5, 0.12, "search scores shown in the movie are upward-biased; the validated winner is marked at the end of each annulus",
                ha="center", va="center", fontsize=11, color="#f0a35e")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, facecolor=fig.get_facecolor())
    return out


# ----------------------------------------------------------------------------
# the step panel
# ----------------------------------------------------------------------------
@dataclass
class StepImages:
    cur_clean: Optional[np.ndarray] = None
    cur_inj: Optional[np.ndarray] = None
    best_inj: Optional[np.ndarray] = None
    best_clean: Optional[np.ndarray] = None
    best_index: int = -1
    #: sources that were injected into ``best_inj``, when they are NOT the ones the history
    #: holds at ``best_index``.  The best cells used to circle ``ad.sources[best_index]``
    #: while the image came from a different bookkeeping path, and the two part company
    #: whenever the picture is the winner's *committed* trial: validation re-scores the
    #: elected config on FRESH injections, so ``annulus01/best_inj.fits`` carries the
    #: validation positions while the history at that index carries the search ones.  On the
    #: annulus-done frame -- the one that stays on screen and is what a notebook shows at the
    #: end -- that drew every circle rotated by the azimuth step between the two draws
    #: (19.5 deg, ~5 px at 0.4"), which reads as a placement bug and is not one.  Pass the
    #: image's own sources here and the cells circle what is actually in the picture.
    best_sources: Optional[List[Tuple[float, float, float]]] = None
    best_labels: Optional[List[Any]] = None        # per-source values that go with best_sources
    note: Optional[str] = None
    fm_image: Optional[np.ndarray] = None          # KLIP-FM response (preview at the best, or the winner's cross-check)
    #: why ``fm_image`` will stay empty for the WHOLE run (a backend with no KLIP-FM), so the
    #: cell can say so instead of promising a model "(after 1st best)" that never comes
    fm_unavailable: Optional[str] = None
    fm_sources: Optional[List[Tuple[float, float, float]]] = None
    fm_label: str = "KLIP-FM response"
    cur_label: Optional[str] = None                # title prefix of the current cells (default "eval N")
    stitch_clean: Optional[np.ndarray] = None      # running stitch (multi-annulus); None = the best images (annulus 1)
    stitch_inj: Optional[np.ndarray] = None
    stitch_sources: Optional[List[Tuple[float, float, float]]] = None
    inj_model: Optional[np.ndarray] = None         # injected-PSF model image of the best eval (IDL disp_injmdl)
    cur_override: Optional[Dict[str, Any]] = None  # validation trials: sources / per_source / raw_per_source /
                                                   # clean_per_source / score / raw of the current cells


def _live_curve_info(clean: Optional[np.ndarray], ad: AnnulusData, i: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Per-eval 5-sigma contrast preview from this eval's sources.

    Returns ``(curve, reason)``: the curve dict (``r_as, curve, sample_r, sample_c``)
    or ``None`` with a one-line reason for the panel.  Like the IDL preview it needs
    only **two** positive per-source values: :func:`products.contrast_curve` fits a
    smooth ``K(r)`` for >= 3 samples, so with 2 the throughput is taken as the median
    ``K`` of the samples (``throughput_fit``'s own rule for < 3 points)."""
    if i < 0 or i >= ad.n:
        return None, "no evaluation yet"
    if clean is None or np.ndim(clean) != 2:
        return None, "no clean image for this eval"
    src = ad.sources[i]
    ps = ad.per_source[i]
    if len(src) < 2:
        return None, f"{len(src)} injected source{'s' if len(src) != 1 else ''} (need >= 2)"
    if len(ps) != len(src):
        return None, f"per-source values missing ({len(ps)} of {len(src)})"
    s = np.array([np.nan if v is None else float(v) for v in ps], float)
    npos = int(np.sum(np.isfinite(s) & (s > 0)))
    if npos < 2:
        return None, f"{npos} of {len(src)} per-source values positive (need >= 2)"
    from .products import contrast_curve, noise_profile
    try:
        r = np.array([float(x[0]) for x in src])
        c = float(ad.contrast[i])
        if not np.isfinite(c) or c <= 0:
            return None, f"injection contrast not positive ({c:g})"
        rin, rout = max(ad.inrad, ad.fwhm), max(ad.outrad, ad.inrad + 2 * ad.fwhm)
        cc = contrast_curve(clean, r, s, c, ad.fwhm, ad.pxscale, rin, rout)
        curve = np.asarray(cc.get("curve"), float)
        if not np.isfinite(curve).any():
            # 2 positive samples: median-K throughput instead of the polynomial fit
            g = np.isfinite(r) & np.isfinite(s) & (s > 0)
            sig, rprof = noise_profile(clean, ad.fwhm, rin, rout)
            ok = np.isfinite(sig)
            if ok.sum() < 2:
                return None, "noise profile undefined in this annulus"
            sig_at = np.interp(r[g] / ad.pxscale, rprof[ok], sig[ok])
            K = s[g] * sig_at / c
            K = K[np.isfinite(K) & (K > 0)]
            if K.size < 1:
                return None, "throughput samples not positive"
            cc["curve"] = 5.0 * sig / max(float(np.median(K)), 1e-12)
            cc["K_samples"] = K
            cc["fit"] = "median K (2 samples)"
        out = {k: (np.asarray(v, float).tolist() if not isinstance(v, str) else v) for k, v in cc.items()}
        return out, None
    except Exception as exc:
        return None, f"contrast preview failed: {type(exc).__name__}: {str(exc)[:60]}"


def _live_curve(clean: Optional[np.ndarray], ad: AnnulusData, i: int) -> Optional[Dict[str, Any]]:
    """Per-eval 5-sigma contrast preview (see :func:`_live_curve_info`); ``None`` when
    it cannot be built."""
    return _live_curve_info(clean, ad, i)[0]


def render_step_classic(ad: AnnulusData, i: int, images: StepImages, out_png: Optional[str] = None,
                out_pdf: Optional[str] = None, curves: Sequence[Tuple[str, Dict[str, Any], Dict[str, Any]]] = (),
                elapsed_s: float = np.nan, eta_s: float = np.nan, cmap: str = "inferno", dpi: int = 110,
                fig: Optional[Figure] = None, step_label: str = "", contrast_note: Optional[str] = None) -> Figure:
    """The original klip-tpe live panel (grid of package-style panels); kept as
    ``style="classic"`` of :func:`render_step`.  ``contrast_note``
    is shown in the contrast panel when no curve can be drawn (why the preview is
    missing); ``images.cur_label`` renames the current-eval cells (e.g. ``last eval N``
    on the annulus-done frame)."""
    with _rc():
        if fig is None:
            fig = Figure(figsize=(17.0, 10.5), dpi=dpi)
            FigureCanvasAgg(fig)
        else:
            fig.clf()
        gs = GridSpec(3, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1.6], height_ratios=[1.15, 1.0, 1.0],
                      left=0.035, right=0.99, top=0.93, bottom=0.06, hspace=0.55, wspace=0.32)
        bi = images.best_index if images.best_index >= 0 else ad.best()[0]
        cur_src = ad.sources[i] if 0 <= i < ad.n else []
        cur_ps = ad.per_source[i] if 0 <= i < ad.n else []
        cur_cps = ad.clean_per_source[i] if 0 <= i < ad.n else None
        best_src = ad.sources[bi] if 0 <= bi < ad.n else []
        best_ps = ad.per_source[bi] if 0 <= bi < ad.n else []
        if images.best_sources is not None:        # the image's own injections; see StepImages
            best_src = list(images.best_sources)
            best_ps = list(images.best_labels or [])
        y_i = ad.y[i] if 0 <= i < ad.n else np.nan
        y_b = ad.y[bi] if 0 <= bi < ad.n else np.nan
        note = images.note
        cl = images.cur_label or f"eval {i + 1}"
        # row 1: images.  A known companion is measured on the CLEAN reductions -- the ones
        # with no injected sources anywhere near it -- so those two cells carry its S/N and
        # the injected cells only circle it.
        k_cur = known_snr(ad, images.cur_clean)
        k_best = known_snr(ad, images.best_clean)
        draw_image(fig.add_subplot(gs[0, 0]), images.cur_clean, ad, f"{cl}: clean (no injection)", cmap,
                   sources=cur_src, labels=cur_cps, note=note, known_labels=k_cur)
        draw_image(fig.add_subplot(gs[0, 1]), images.cur_inj, ad,
                   f"{cl}: injected  score {_fnum(y_i)}", cmap, sources=cur_src, labels=cur_ps, note=note)
        draw_image(fig.add_subplot(gs[0, 2]), images.best_inj, ad,
                   f"best (eval {bi + 1}): injected  score {_fnum(y_b)}", cmap, sources=best_src, labels=best_ps, note=note)
        draw_image(fig.add_subplot(gs[0, 3]), images.best_clean, ad, f"best (eval {bi + 1}): clean", cmap,
                   note=note, known_labels=k_best)
        # row 2: S/N maps + trace
        draw_image(fig.add_subplot(gs[1, 0]), _snr_map(images.cur_inj, ad, cur_src), ad, f"{cl}: S/N map (injected)",
                   cmap, sources=cur_src, snr=True, note=note)
        draw_image(fig.add_subplot(gs[1, 1]), _snr_map(images.best_inj, ad, best_src), ad, "best: S/N map (injected)",
                   cmap, sources=best_src, snr=True, note=note)
        panel_trace(fig.add_subplot(gs[1, 2:4]), ad, current=i)
        # row 3
        panel_marginals(fig, gs[2, 0:2], ad, current=i)
        panel_k(fig.add_subplot(gs[2, 2]), ad, current=i)
        sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, 3], hspace=0.7)
        panel_inclusion(fig.add_subplot(sub[0]), ad, colorbar=False)
        panel_partition_snr(fig.add_subplot(sub[1]), ad)
        panel_contrast(fig.add_subplot(gs[2, 4]), curves, float(ad.contrast[i]) if 0 <= i < ad.n else np.nan,
                       placeholder=contrast_note)
        # text column (+ the KLIP-FM thumbnail below it when an FM image exists)
        fm_ok = images.fm_image is not None and np.ndim(images.fm_image) == 2 and np.isfinite(images.fm_image).any()
        if fm_ok:
            sub_t = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0:2, 4], height_ratios=[1.8, 1.0], hspace=0.2)
            ax_t = fig.add_subplot(sub_t[0])
            draw_image(fig.add_subplot(sub_t[1]), images.fm_image, ad, images.fm_label, cmap,
                       sources=images.fm_sources, flatten=False, vrange=_fm_range(images.fm_image))
        else:
            ax_t = fig.add_subplot(gs[0:2, 4])
        panel_text(ax_t, _step_text(ad, i, bi, elapsed_s, eta_s))
        ttl = (f"{ad.run_name}   annulus {ad.annulus + 1}/{ad.nann}  [{ad.inrad:.1f}, {ad.outrad:.1f}] px "
               f"({ad.inrad * ad.pxscale:.2f}-{ad.outrad * ad.pxscale:.2f}\")   eval {i + 1}/{ad.n_iter}   {step_label}")
        fig.suptitle(ttl, fontsize=11, x=0.035, ha="left")
        fig.text(0.99, 0.965, f"metric: {ad.metric_name}  --  scores are SEARCH scores (upward-biased); validated = raw metric on fresh injections",
                 ha="right", fontsize=7.5, color="#555555")
        if out_png:
            os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
            fig.savefig(out_png, dpi=dpi)
        if out_pdf:
            os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
            fig.savefig(out_pdf)
    return fig


# ----------------------------------------------------------------------------
# IDL-layout live panel (near2m_show)
# ----------------------------------------------------------------------------
def _box(fig, x0, y0, x1, y1):
    return fig.add_axes([x0, y0, x1 - x0, y1 - y0])


def _sq_box(fig, x0, y0, x1, y1):
    """Axes of the largest square (in figure inches) centred in the box (IDL keep_aspect)."""
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    w_in, h_in = (x1 - x0) * fw, (y1 - y0) * fh
    side = min(w_in, h_in)
    w, h = side / fw, side / fh
    return fig.add_axes([x0 + ((x1 - x0) - w) / 2, y0 + ((y1 - y0) - h) / 2, w, h])


def _lbl(v, raw=None) -> Optional[str]:
    """IDL ``near2m_srclbl``: ``raw/corr`` when the pre-subtraction value is known."""
    if v is None or not np.isfinite(v):
        return None
    if raw is not None and np.isfinite(raw):
        return f"{raw:.1f}/{v:.1f}"
    return f"{v:.1f}"


def _sma(y: np.ndarray, m: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean of the last ``w`` finite values of ``y`` within mask ``m`` (IDL SMA)."""
    out = np.full(y.size, np.nan)
    for i in range(y.size):
        v = y[:i + 1][m[:i + 1] & np.isfinite(y[:i + 1])]
        if v.size:
            out[i] = v[-w:].mean()
    return out


def panel_convergence_idl(ax, ad: AnnulusData, current: Optional[int] = None):
    """IDL convergence panel: grey hollow = warm-up/explore, TPE / local filled, running
    best (fg), SMA20/SMA50 of the guided evals, SMA20 of the random ones, best circled."""
    n = ad.n
    y = ad.y
    ev = np.arange(n)
    ph = np.array(ad.phases) if n else np.array([], str)
    rnd = np.array([_is_random_phase(p) for p in ph], bool)
    loc = (ph == "local") & ~rnd
    tpe = ~rnd & ~loc
    cm = matplotlib.colormaps.get_cmap(SCORE_CMAP)
    c_tpe, c_loc, c_s20, c_s50 = cm(0.35), cm(0.55), cm(0.80), cm(0.97)
    fg = _fg()
    n_bars, dlo, dhi = 0, np.inf, -np.inf
    if n:
        # Under the points when n_remeasure > 1: the min-to-max span of the draws each
        # trial's score is the mean of.  This panel is the one render_step draws -- panel_trace
        # belongs to render_step_classic -- so the bars have to be added here too or the live
        # window and the step PDFs show a mean with no indication of what it is a mean of.
        try:
            from .plots import draw_ranges
            _dw = list((ad.extra or {}).get("draws") or [])
            _ds = [(d or {}).get("draw_scores") if isinstance(d, dict) else None
                   for d in (_dw + [None] * max(0, n - len(_dw)))][:n]
            _cols = [("#9a9a9a" if r else (c_loc if l else c_tpe)) for r, l in zip(rnd, loc)]
            n_bars = draw_ranges(ax, ev, _ds, colors=_cols, label="")
            flat = [float(v) for d in _ds for v in (d or []) if v is not None and np.isfinite(v)]
            if flat:
                dlo, dhi = min(flat), max(flat)
        except Exception:
            n_bars = 0
        sma20, sma50, smag = _sma(y, ~rnd, 20), _sma(y, ~rnd, 50), _sma(y, rnd, 20)
        g = np.isfinite(sma20)
        slope = np.nan
        if g.sum() >= 2:
            idx = np.where(g)[0][-20:]
            slope = np.polyfit(ev[idx], sma20[idx], 1)[0]
        sstr = "n/a" if not np.isfinite(slope) else f"{slope:+.3f}"
        ax.set_title(f"TPE convergence  (SMA20 slope {sstr}/eval)  [{ad.metric_name}]", fontsize=7.5)
        if rnd.any():
            ax.scatter(ev[rnd], y[rnd], s=12, facecolor="none", edgecolor="#9a9a9a", lw=0.7)
        if tpe.any():
            ax.scatter(ev[tpe], y[tpe], s=12, color=c_tpe, lw=0)
        if loc.any():
            ax.scatter(ev[loc], y[loc], s=12, color=c_loc, lw=0)
        ax.plot(ev, ad.running_best(), color=fg, lw=1.6)
        ax.plot(ev, smag, color="#777777", lw=1.4, ls="--")
        ax.plot(ev, sma20, color=c_s20, lw=1.4)
        ax.plot(ev, sma50, color=c_s50, lw=1.4, ls="--")
        bi, bs = ad.best()
        if bi >= 0:
            ax.scatter([bi], [bs], s=110, facecolor="none", edgecolor="#00d000", lw=1.6, zorder=5)
        if current is not None and 0 <= current < n and np.isfinite(y[current]):
            ax.scatter([current], [y[current]], s=60, marker="s", facecolor="none", edgecolor=CUR_COLOR, lw=1.0, zorder=5)
        fin = y[np.isfinite(y)]
        lo, hi = (min(float(fin.min()), 0.0), float(fin.max()) * 1.1) if fin.size else (0.0, 1.0)
        if n_bars:                      # a clipped error bar is worse than none
            lo, hi = min(lo, float(dlo)), max(hi, float(dhi) * 1.02)
        if hi <= lo:
            hi = lo + 1.0
        ax.set_ylim(lo, hi)
        if ad.n_init and n > ad.n_init:
            ax.axvline(ad.n_init - 0.5, color="#777777", ls=":", lw=0.7)
    else:
        ax.set_title("TPE convergence", fontsize=7.5)
        lo, hi = 0.0, 1.0
    ax.set_xlim(-0.5, max(n - 1, 0) + 0.5 if n < 2 else n - 0.5)
    ax.set_xlabel("evaluation")
    nsrc = len(ad.sources[-1]) if (n and ad.sources and ad.sources[-1]) else 0
    ax.set_ylabel(f"{'mean' if nsrc == 2 else 'median'} S/N")
    # colour-coded flowing legend (IDL): one word per series
    x = 0.01
    series = [("warm-up/explore ", "#9a9a9a"), ("TPE ", c_tpe), ("local ", c_loc), ("SMA20 ", c_s20),
              ("SMA50 ", c_s50), ("random SMA20", "#9a9a9a")]
    if n_bars:
        nd = max((len(d.get("draw_scores") or []) for d in ((ad.extra or {}).get("draws") or [])
                  if isinstance(d, dict)), default=0)
        series.append((f"  | bars: min-max of {nd}" if nd else "  | bars: draw min-max", "#8a8a8a"))
    for txt, col in series:
        t = ax.text(x, 0.02, txt, transform=ax.transAxes, color=col, fontsize=6.3, va="bottom")
        try:
            ax.figure.canvas.draw()
            bb = t.get_window_extent().transformed(ax.transAxes.inverted())
            x = bb.x1
        except Exception:
            x += 0.16
    return (lo, hi)


def panel_snrhist_h(ax, ad: AnnulusData, yr):
    """IDL ``near2m_snrhist_h``: rotated marginal score histogram sharing the convergence
    panel's y-range; grey = random, colour = TPE / local, Gaussian fits."""
    ax.set_title("S/N dist", fontsize=7.5)
    v = ad.valid
    ax.set_ylim(*yr)
    ax.set_yticklabels([])
    ax.set_xlabel("fraction of evals")
    if v.sum() < 2:
        return
    y = ad.y[v]
    ph = np.array(ad.phases)[v]
    rnd = np.array([_is_random_phase(p) for p in ph])
    cm = matplotlib.colormaps.get_cmap(SCORE_CMAP)
    groups = [("random", rnd, "#9a9a9a"), ("local", (ph == "local") & ~rnd, cm(0.55)), ("TPE", ~rnd & (ph != "local"), cm(0.35))]
    edges = np.linspace(yr[0], yr[1], 17)
    bw = edges[1] - edges[0]
    fmax = 0.0
    for name, m, col in groups:
        if not m.any():
            continue
        h, _ = np.histogram(y[m], bins=edges)
        frac = h / max(m.sum(), 1)
        fmax = max(fmax, float(frac.max()))
        ax.barh(0.5 * (edges[1:] + edges[:-1]), frac, height=bw * 0.9, color=col, alpha=0.75 if name == "random" else 0.9, lw=0)
        if m.sum() >= 3:
            mu, sd = float(np.mean(y[m])), float(np.std(y[m], ddof=1))
            if sd > 0:
                yg = np.linspace(yr[0], yr[1], 120)
                ax.plot(bw / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((yg - mu) / sd) ** 2), yg, color=col, lw=0.9)
            ax.axhline(mu, color=col, ls="--", lw=0.7)
    ax.set_xlim(0, max(fmax * 1.12, 0.05))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))


def panel_contrast_idl(ax, curves, contrast: float, placeholder: Optional[str] = None):
    """IDL ``near2m_cpanel``: SNR=5 contrast (fg, solid) with the KLIP-FM cross-check dashed green."""
    ax.set_title("SNR=5 contrast (inj-calibrated)", fontsize=7.5)
    any_c = False
    for lab, cc, st in curves:
        if not cc:
            continue
        r = np.array([np.nan if v is None else v for v in cc.get("r_as", [])], float)
        c = np.array([np.nan if v is None else v for v in cc.get("curve", [])], float)
        if r.size and np.isfinite(c).any() and (c[np.isfinite(c)] > 0).any():
            fm = "KLIP-FM" in lab
            this = lab.startswith("this eval") or lab.startswith("eval ")
            ax.plot(r, np.where(c > 0, c, np.nan), color="#3cb44b" if fm else ("#8a8a8a" if this else _fg()),
                    ls="--" if (fm or this) else "-", lw=1.3 if fm else (0.9 if this else 1.5))
            any_c = True
    if any_c:
        ax.set_yscale("log")
        if np.isfinite(contrast) and contrast > 0:
            ax.axhline(contrast, color="#777777", ls=":", lw=0.7)
        if any("KLIP-FM" in lab for lab, cc, _ in curves if cc):
            ax.text(0.98, 0.04, "green dashed = KLIP-FM", transform=ax.transAxes, ha="right", color="#3cb44b", fontsize=6.5)
        if any(lab.startswith(("this eval", "eval ")) for lab, cc, _ in curves if cc):
            ax.text(0.98, 0.12, "grey dashed = this eval", transform=ax.transAxes, ha="right", color="#8a8a8a", fontsize=6.5)
    else:
        msg = placeholder or "after first best..."
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=6.5, color="#888888", wrap=True)
    ax.set_xlabel('separation (")')
    ax.set_ylabel("SNR=5 contrast")


def _cv(v) -> str:
    """Compact value for the BEST/TEST vectors: ints as ints, else 2 significant digits."""
    if v is None or not np.isfinite(v):
        return "-"
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.2g}"


def _vec_lines(ad: AnnulusData, i: int) -> List[str]:
    """IDL ``near2m_pnvecstr``: one ``name [v1 v2 ...]`` entry per base parameter (one value
    per partition, or one value), then k and the included partitions."""
    if not (0 <= i < ad.n):
        return []
    out = []
    parts = ad.partitions if ad.replicated else [None]
    for base in ad.bases:
        vals = []
        for pid in parts:
            dim = ad.dim_for(base, pid)
            vals.append(np.nan if dim is None else ad.X[i, dim])
        out.append(f"{COL_ABBREV.get(base, base)} [{' '.join(_cv(v) for v in vals)}]")
    kd = ad.k_used[i]
    if isinstance(kd, dict):
        out.append("k [" + " ".join(_cv(kd.get(str(p), kd.get(p, np.nan))) for p in ad.partitions) + "]")
    elif kd is not None:
        out.append(f"k [{_cv(kd)}]")
    if ad.multi:
        out.append("parts " + ",".join(str(p) for p in ad.selected[i]))
    return out


def _inj_lines(ad: AnnulusData, i: int, src=None) -> List[str]:
    """``src`` overrides the history row -- the BEST block has to quote the injections that
    are in the picture it labels, which after validation is the committed trial's fresh
    draw, not the search evaluation at index ``i``."""
    if src is None:
        if not (0 <= i < ad.n) or not ad.sources[i]:
            return []
        src = ad.sources[i]
    if not src:
        return []
    con = ad.contrast[i] if 0 <= i < ad.n else float("nan")
    return [f"sep = {', '.join(f'{s[0]:.2f}' for s in src)}\"",
            f"PA = {', '.join(f'{s[1]:.0f}' for s in src)} deg",
            f"inj contrast = {con:.2E}"]


def _text_block(fig, x_lab, x_c1, x_c2, y_top, label, lab_lines, vec_lines, pitch, fs) -> float:
    """IDL ``near2m_2col`` + label column: header, injection lines under it, the parameter
    vectors alternating between two right-hand columns.  Returns the y below the block."""
    fig.text(x_lab, y_top, label, fontsize=fs + 0.5, weight="bold", va="top")
    for k, t in enumerate(lab_lines):
        fig.text(x_lab, y_top - (k + 1) * pitch, t, fontsize=fs, va="top")
    ncol = 0
    for k, t in enumerate(vec_lines):
        col, row = k % 2, k // 2
        fig.text(x_c1 if col == 0 else x_c2, y_top - row * pitch, t, fontsize=fs, va="top", family="monospace")
        ncol = max(ncol, row + 1)
    nrows = max(len(lab_lines) + 1, ncol)
    return y_top - (nrows + 0.6) * pitch


def render_step(ad: AnnulusData, i: int, images: StepImages, out_png: Optional[str] = None,
                out_pdf: Optional[str] = None, curves: Sequence[Tuple[str, Dict[str, Any], Dict[str, Any]]] = (),
                elapsed_s: float = np.nan, eta_s: float = np.nan, cmap: str = "inferno", dpi: int = 100,
                fig: Optional[Figure] = None, step_label: str = "", contrast_note: Optional[str] = None,
                style: str = "idl", white: bool = False, loop_s: float = np.nan, last_s: float = np.nan,
                eta_full_s: float = np.nan, corner_partition: Optional[str] = "__cycle__") -> Figure:
    """The main live panel for evaluation ``i`` of ``ad``.  ``eta_s`` / ``eta_full_s`` are
    IDL's ``rem_ann`` / ``rem`` (remaining for this annulus / for the whole run).

    ``style="idl"`` (default) reproduces the IDL ``near2m_show`` window: two rows of
    five image cells (test / test no-inj / best / stitched no-inj / stitched inj, and
    their S/N maps with per-source ``raw/corr`` labels), the TPE convergence trace
    with its rotated S/N histogram and the KLIP-FM / injected-PSF squares, night
    inclusion and SNR=5 contrast, and on the right the BEST / TEST parameter
    vectors, elapsed / ETA, the night-effect panels and the scatter corner.  Black
    background like the live window unless ``white`` (the IDL PDF/PNG snapshots).
    ``style="classic"`` is the previous klip-tpe layout."""
    if style == "classic":
        return render_step_classic(ad, i, images, out_png, out_pdf, curves, elapsed_s, eta_s, cmap, dpi, fig,
                                   step_label, contrast_note)
    with _rc(idl=True, dark=not white):
        if fig is None:
            fig = Figure(figsize=panel_size(dpi), dpi=dpi)
            FigureCanvasAgg(fig)
        else:
            fig.clf()
            fig.set_size_inches(*panel_size(fig.get_dpi()))
        fig.patch.set_facecolor("black" if not white else "white")
        fg = _fg()
        bi = images.best_index if images.best_index >= 0 else ad.best()[0]
        n_ok = 0 <= i < ad.n
        cur_src = ad.sources[i] if n_ok else []
        cur_ps, cur_rps = (ad.per_source[i], ad.raw_per_source[i]) if n_ok else ([], [])
        cur_cps = (ad.clean_per_source[i] or []) if n_ok else []
        b_ok = 0 <= bi < ad.n
        best_src = ad.sources[bi] if b_ok else []
        best_ps, best_rps = (ad.per_source[bi], ad.raw_per_source[bi]) if b_ok else ([], [])
        if images.best_sources is not None:        # the image's own injections; see StepImages
            best_src = list(images.best_sources)
            best_ps = list(images.best_labels or [])
            best_rps = [None] * len(best_src)
        y_i, y_b = (ad.y[i] if n_ok else np.nan), (ad.y[bi] if b_ok else np.nan)
        raw_i, raw_b = (ad.raw[i] if n_ok else np.nan), (ad.raw[bi] if b_ok else np.nan)
        if images.cur_override:                    # validation trial in the current cells
            ov = images.cur_override
            cur_src = list(ov.get("sources") or [])
            cur_ps = list(ov.get("per_source") or [])
            cur_rps = list(ov.get("raw_per_source") or [])
            cur_cps = list(ov.get("clean_per_source") or [])
            y_i = float(ov.get("score", np.nan)); raw_i = float(ov.get("raw", np.nan))
        stat = "mean" if len(best_src) == 2 else "median"
        note = images.note
        # ---- geometry (IDL normal coordinates) -------------------------------------
        lx0, lx1, hx0, hx1 = 0.025, 0.605, 0.645, 0.995
        gap, cbw, cbgap = 0.013, 0.009, 0.006
        ncols = 5
        colw = (lx1 - lx0 - (ncols - 1) * gap - cbgap - cbw) / ncols
        xs = [lx0 + k * (colw + gap) for k in range(ncols)]
        cbx0 = xs[-1] + colw + cbgap
        img_y0, img_y1 = 0.769, 0.962          # the IDL near2m_show coordinates (panel = 1850 x 990 px)
        st_y0, st_y1 = 0.524, 0.717
        chA0, chA1 = 0.304, 0.469
        chB0, chB1 = 0.064, 0.229
        cxa0, cxa1, cxb0, cxb1 = 0.028, 0.300, 0.350, 0.608
        # stitched cells: the running stitch when given, else (annulus 1) the best images
        st_clean = images.stitch_clean if images.stitch_clean is not None else images.best_clean
        st_inj = images.stitch_inj if images.stitch_inj is not None else images.best_inj
        st_src = images.stitch_sources if images.stitch_inj is not None else best_src
        st_lab = None if images.stitch_inj is not None else [_lbl(v, r) for v, r in zip(best_ps, best_rps or [None] * len(best_ps))]

        def cell(x0, y0, y1, img, title, **kw):
            ax = _sq_box(fig, x0, y0, x0 + colw, y1)
            ax._no_caption = True
            draw_image(ax, img, ad, title, cmap, note=note, **kw)
            ax.set_title(title, fontsize=7.5)
            return ax

        # ---- row 1: images ---------------------------------------------------------
        t0 = images.cur_label or f"Ann {ad.annulus + 1}/{ad.nann}  Eval {i + 1}/{ad.n_iter}"
        cell(xs[0], img_y0, img_y1, images.cur_inj, t0)
        cell(xs[1], img_y0, img_y1, images.cur_clean, "Test (no inj)")
        cell(xs[2], img_y0, img_y1, images.best_inj, f"Best (eval {bi + 1})" if b_ok else "Best")
        cell(xs[3], img_y0, img_y1, st_clean, "Stitched best (no inj)")
        cell(xs[4], img_y0, img_y1, st_inj, "Stitched best (with inj)")
        cax = _box(fig, cbx0, img_y0 + 0.012, cbx0 + cbw, img_y1 - 0.012)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=mcolors.Normalize(-1, 5), cmap=cmap), cax=cax)
        cax.set_ylabel("KLIP image stretch (sigma)", fontsize=6.5)
        cax.tick_params(labelsize=6)
        # ---- row 2: S/N maps -------------------------------------------------------
        cur_lab = [_lbl(v, r) for v, r in zip(cur_ps, cur_rps or [None] * len(cur_ps))]
        best_lab = [_lbl(v, r) for v, r in zip(best_ps, best_rps or [None] * len(best_ps))]
        tt = f"Test S/N  orig {raw_i:.2f} / corr {y_i:.2f}" if np.isfinite(raw_i) and np.isfinite(y_i) else (
            f"Test SNR = {y_i:.2f}" if np.isfinite(y_i) else "Test SNR")
        tb = f"Best {stat} S/N  orig {raw_b:.2f} / corr {y_b:.2f}" if np.isfinite(raw_b) and np.isfinite(y_b) else (
            f"Best {stat} S/N = {y_b:.2f}" if np.isfinite(y_b) else "Best SNR")
        cell(xs[0], st_y0, st_y1, _snr_map(images.cur_inj, ad, cur_src), tt, sources=cur_src, labels=cur_lab, snr=True)
        cell(xs[1], st_y0, st_y1, _snr_map(images.cur_clean, ad, cur_src), "Test S/N (no inj)", sources=cur_src,
             labels=[_lbl(None if c is None else float(c)) for c in cur_cps] or None, snr=True)
        cell(xs[2], st_y0, st_y1, _snr_map(images.best_inj, ad, best_src), tb, sources=best_src, labels=best_lab, snr=True)
        cell(xs[3], st_y0, st_y1, _snr_map(st_clean, ad, None), "Stitched SNR (no inj)", snr=True)
        cell(xs[4], st_y0, st_y1, _snr_map(st_inj, ad, st_src), "Stitched SNR (with inj)", sources=st_src, labels=st_lab, snr=True)
        cax = _box(fig, cbx0, st_y0 + 0.012, cbx0 + cbw, st_y1 - 0.012)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=mcolors.Normalize(*SNR_RANGE), cmap=cmap), cax=cax)
        cax.set_ylabel("Image SNR", fontsize=6.5)
        cax.tick_params(labelsize=6)
        # ---- row A: convergence | S/N dist | KLIP-FM | injected PSF ------------------
        yr = panel_convergence_idl(_box(fig, cxa0, chA0, cxa1, chA1), ad, current=i)
        sqw = (chA1 - chA0) * fig.get_figheight() / fig.get_figwidth()
        xi0 = cxb1 - sqw
        xf0 = xi0 - 0.024 - sqw
        xh0, xh1 = cxa1 + 0.012, xf0 - 0.024
        panel_snrhist_h(_box(fig, xh0, chA0, max(xh1, xh0 + 0.02), chA1), ad, yr)
        ax = _box(fig, xf0, chA0, xf0 + sqw, chA1)
        fm_ok = images.fm_image is not None and np.ndim(images.fm_image) == 2 and np.isfinite(images.fm_image).any()
        if fm_ok:
            draw_image(ax, images.fm_image, ad, "KLIP-FM model (best)", cmap, flatten=False, scale="symlog")
        else:
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, getattr(images, "fm_unavailable", None) or "(after 1st best)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=6.5)
        ax.set_title("KLIP-FM model (best)  [symlog]", fontsize=7.5)
        ax = _box(fig, xi0, chA0, xi0 + sqw, chA1)
        im_ok = images.inj_model is not None and np.ndim(images.inj_model) == 2 and np.isfinite(images.inj_model).any()
        if im_ok:
            draw_image(ax, images.inj_model, ad, "injected PSF (best)", cmap, flatten=False,
                       vrange=(0.0, float(np.nanmax(images.inj_model)) or 1.0))
        else:
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, "(after 1st best)", ha="center", va="center", transform=ax.transAxes, fontsize=6.5)
        ax.set_title("injected PSF (best)", fontsize=7.5)
        # ---- row B: partition inclusion | contrast -------------------------------------
        plabel = str(getattr(ad, "partition_label", None) or "night")
        ax = _box(fig, cxa0, chB0, cxa1, chB1)
        if ad.multi:
            panel_inclusion(ax, ad, colorbar=False)
            ax.set_title("", loc="left")
            ax.set_title(f"{plabel} inclusion (color=SNR)", fontsize=7.5)
            ax.set_ylabel(plabel)
        else:                                   # single night: IDL mwcm_pimppanel instead
            panel_importance_live(ax, ad)
        panel_contrast_idl(_box(fig, cxb0, chB0, cxb1, chB1), curves, float(ad.contrast[i]) if n_ok else np.nan,
                           placeholder=contrast_note)
        # ---- right column: title, BEST / TEST blocks -------------------------------------
        fig.text((hx0 + hx1) / 2 - 0.02, 0.982, f"{ad.run_name}  --  annulus {ad.annulus + 1}/{ad.nann}  TPE optimization",
                 ha="center", va="top", fontsize=10.5)
        if step_label:
            fig.text(lx0, 0.985, step_label, fontsize=8, va="top", color="#00d000")
        pitch, fs = 0.0185, 6.4
        xlab, xc1, xc2 = hx0 + 0.008, hx0 + 0.118, hx0 + 0.238
        yb = 0.955
        if b_ok:
            yb = _text_block(fig, xlab, xc1, xc2, yb, "BEST",
                             _inj_lines(ad, bi, images.best_sources), _vec_lines(ad, bi), pitch, fs)
        if n_ok:
            _text_block(fig, xlab, xc1, xc2, yb, "TEST", _inj_lines(ad, i), _vec_lines(ad, i), pitch, fs)
        # ---- corner (lower-left triangle of the right region) -----------------------------
        # One partition at a time, cycling, as IDL does.  Pooling every partition into one
        # picture overlays landscapes that need not agree -- a parameter can be good on one
        # night and bad on the next, and the pooled cloud shows a blur that is true of no
        # night in particular.  ``corner_partition`` selects it: "__cycle__" steps through
        # by panel, a partition id pins one, None restores the pooled view.
        pid = corner_partition
        if pid == "__cycle__":
            pid = (ad.partitions[i % len(ad.partitions)]
                   if ad.replicated and len(ad.partitions) > 1 else None)
        if pid is not None and str(pid) not in [str(p) for p in ad.partitions]:
            pid = None
        cs = ad.corner(pid)
        top_rows = np.zeros(len(cs["y"]), bool)
        tm = ad.top_mask()
        for r, e in enumerate(cs["eval"]):
            top_rows[r] = tm[e]
        warm = np.array([_is_random_phase(ad.phases[e]) for e in cs["eval"]], bool)
        cur_pt = ad.eval_point(i, cs, pid) if n_ok else None
        # always say which landscape this is -- one night's, or the pooled blur
        npart = len(ad.partitions)
        if pid is not None:
            j = [str(p) for p in ad.partitions].index(str(pid)) + 1
            scope = f"   ({plabel} {j} of {npart}: {pid}, {len(cs['y'])} evals incl. it)"
        else:
            scope = (f"   ({npart} {plabel}s pooled, 1 point per {plabel} per eval)"
                     if ad.replicated and npart > 1 else "")
        draw_corner(fig, cs, ad, f"S/N landscape -- {stat} S/N{scope}",
                    rect=(hx0, 0.035, hx1 - hx0, 0.635), warm=warm,
                    best_pt=_best_pt(ad, cs, pid), cur_pt=cur_pt, cands=_cands(ad, cs, pid), top_mask=top_rows,
                    compact=True, max_points=CORNER_MAX_POINTS)
        # ---- upper-right triangle: elapsed/ETA, night panels, legend ------------------------
        wall = ad.wall[np.isfinite(ad.wall)]
        red_s = float(np.mean(wall[-50:])) if wall.size else np.nan
        last_red = float(wall[-1]) if wall.size else np.nan
        tot = elapsed_s + eta_full_s if np.isfinite(elapsed_s) and np.isfinite(eta_full_s) else np.nan
        # durations with explicit units (only the 'done' line is a clock time)
        eta_lines = [f"Elapsed  {_fmt_dur(elapsed_s)}", f"ETA(ann)  {_fmt_dur(eta_s)}",
                     f"ETA(full)  {_fmt_dur(eta_full_s)}", f"Est. total  {_fmt_dur(tot)}",
                     f"done  {_fmt_clock(time.time() + eta_full_s) if np.isfinite(eta_full_s) else '--'}",
                     (f"avg {loop_s:.1f} s/eval" if np.isfinite(loop_s) else "avg -- s/eval")
                     + (f"  ({red_s:.1f} s reduce)" if np.isfinite(red_s) else ""),
                     (f"last {last_s:.1f} s/eval" if np.isfinite(last_s) else "last -- s/eval")
                     + (f"  ({last_red:.1f} s reduce)" if np.isfinite(last_red) else "")]
        fig.text(hx0 + 0.41 * (hx1 - hx0), 0.668, "\n".join(eta_lines), fontsize=7, va="top", ha="center", linespacing=1.5)
        px0, px1 = hx0 + 0.50 * (hx1 - hx0) + 0.030, hx1 - 0.004
        hN, ptop = 0.086, 0.625
        if ad.multi:                            # single night: no night effect / S/N vs # nights panels
            ax = _box(fig, px0, ptop - hN, px1, ptop)
            panel_partition_effect(ax, ad)
            ax.set_title("", loc="left")
            ax.set_title(f"{plabel} effect ({'black' if white else 'white'}=in, gray=out)", fontsize=7)
            ax.set_xlabel(plabel); ax.set_ylabel(f"{stat} S/N", fontsize=6.5)
            nn_top = ptop - hN - 0.052
            ax = _box(fig, px0, nn_top - hN, px1, nn_top)
            panel_partition_count(ax, ad)
            if ax.get_legend():
                ax.get_legend().remove()
            ax.set_title("", loc="left")
            ax.set_title(f"S/N vs # {plabel}s", fontsize=7)
            ax.set_xlabel(f"# {plabel}s"); ax.set_ylabel(f"{stat} S/N", fontsize=6.5)
            yL = nn_top - hN - 0.050
        else:
            yL = ptop - 0.010
        for il, (mk, txt) in enumerate((("o", "warm-up / explore"), ("O", "TPE"), ("s", "best"), ("^", "valid"), ("*", "final"))):
            yy = yL - il * 0.022
            fig.text(px1 - 0.020, yy, txt, ha="right", va="center", fontsize=6.5)
            fig.add_artist(matplotlib.lines.Line2D([px1 - 0.010], [yy], marker=mk.lower(), ms=5, mfc=fg if mk == "O" else "none",
                                                   mec=fg, ls="none", transform=fig.transFigure))
        if out_png:
            os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
            fig.savefig(out_png, dpi=dpi, facecolor=fig.get_facecolor())
        if out_pdf:
            os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
            fig.savefig(out_pdf, facecolor=fig.get_facecolor())
    return fig


def _step_text(ad: AnnulusData, i: int, bi: int, elapsed_s: float, eta_s: float) -> List[str]:
    if not (0 <= i < ad.n):
        return ["no evaluation"]
    ph = ad.phases[i]
    y = ad.y[i]
    raw = ad.raw[i]
    cps = ad.clean_per_source[i]
    cl = np.nan
    if cps:
        v = np.array([np.nan if c is None else max(float(c), 0.0) for c in cps])
        cl = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
    kd = ad.k_used[i]
    if isinstance(kd, dict):
        kstr = " ".join(f"{p}={_fmtval(v)}" for p, v in kd.items())
    else:
        kstr = _fmtval(kd) if kd is not None else ("searched" if "k_klip" in ad.bases else "--")
    lines = [
        f"eval {i + 1}/{ad.n_iter}   phase {ph}   [{ad.search_mode}]",
        f"score      {_fnum(y)}   (search, upward-biased)",
    ]
    _dw = (ad.extra.get("draws") or [None] * ad.n)[i] if isinstance(ad.extra, dict) else None
    if _dw and _dw.get("draw_n"):
        _ds = _dw.get("draw_scores") or []
        _txt = ", ".join("--" if v is None else f"{float(v):.2f}" for v in _ds)
        _sd = _dw.get("draw_sd")
        lines.append(f"  = mean of {int(_dw['draw_n'])} draws [{_txt}]"
                     + (f"   sd {float(_sd):.2f}" if _sd is not None else "")
                     + ("   (image is the last draw)" if len(_ds) > 1 else ""))
    lines += [
        f"raw        {_fnum(raw)}   clean term {_fnum(cl)}",
        f"best       {_fnum(ad.y[bi]) if bi >= 0 else '--'} @ eval {bi + 1 if bi >= 0 else '--'}   (search)",
    ]
    if ad.winner is not None:
        w = ad.winner
        lines.append(f"validated  {_fnum(float(w['winner_score']))} @ eval {int(w['winner_index']) + 1}"
                     if w.get("validated") else "validated  -- (not validated)")
    else:
        lines.append("validated  -- (after search)")
    lines += [
        f"contrast   {ad.contrast[i]:.3e}   sources {len(ad.sources[i])}",
        f"k          {kstr}",
        f"wall       {ad.wall[i]:.1f} s/eval   elapsed {_fmt_hms(elapsed_s)}   ETA {_fmt_hms(eta_s)}",
        "",
    ]
    lines += ad.config_lines(i, "-- current config --")
    if bi >= 0 and bi != i:
        lines.append("")
        lines += ad.config_lines(bi, f"-- best config (eval {bi + 1}) --")
    if ad.multi:
        lines.append("")
        lines.append("included: " + ", ".join(ad.selected[i]))
    return lines


# ----------------------------------------------------------------------------
# post-hoc books
# ----------------------------------------------------------------------------
def _fits(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        from astropy.io import fits
        return np.asarray(fits.getdata(path), float)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# IDL parity books: importance / parallel coordinates / rank / slice / products sheet
# ----------------------------------------------------------------------------
def eta_squared(x: np.ndarray, y: np.ndarray, nbins: int = 8) -> float:
    """Main-effect variance fraction of ``y`` explained by binning on ``x``
    (``eta^2 = Var(E[y|bin]) / Var(y)``; the IDL ``eval_importance`` statistic)."""
    g = np.isfinite(x) & np.isfinite(y)
    x, y = x[g], y[g]
    if x.size < 2 * nbins or np.nanstd(y) == 0:
        return float("nan")
    u = np.unique(x)
    if u.size <= nbins:
        codes = np.searchsorted(u, x)
    else:
        edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
        codes = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nbins - 1)
    ybar = y.mean()
    num = sum(((y[codes == c].mean() - ybar) ** 2) * (codes == c).sum() for c in np.unique(codes))
    return float(num / (((y - ybar) ** 2).sum() or np.nan))


def _pages(ad: AnnulusData):
    """(label, corner-set) per page: one page per partition (IDL 'Night N') or one global page."""
    if ad.replicated and ad.partitions:
        return [(f"{pid}", ad.corner(pid)) for pid in ad.partitions]
    return [("all", ad.corner(None))]


def _subplots(nrow, ncol, figsize=(8, 6), squeeze=True):
    fig = Figure(figsize=figsize)
    axes = fig.subplots(nrow, ncol, squeeze=squeeze)
    return fig, axes


def render_importance(ad: AnnulusData, out_pdf: str) -> str:
    """IDL ``eval_importance``: horizontal bars of eta^2 per parameter, one page per partition."""
    from matplotlib.backends.backend_pdf import PdfPages
    with _rc(), PdfPages(out_pdf) as pdf:
        for label, cs in _pages(ad):
            fig = Figure(figsize=(7.5, 5.5))
            ax = fig.add_subplot(111)
            eta = np.array([eta_squared(cs["X"][:, j], cs["y"]) for j in range(len(cs["names"]))])
            order = np.argsort(np.nan_to_num(eta, nan=-1))
            ax.barh(np.arange(len(order)), np.nan_to_num(eta[order]), color="#1f77b4")
            ax.set_yticks(np.arange(len(order)))
            ax.set_yticklabels([cs["names"][j] for j in order], fontsize=8)
            for k, j in enumerate(order):
                if np.isfinite(eta[j]):
                    ax.text(eta[j] + 0.002, k, f"{eta[j]:.2f}", va="center", fontsize=7)
            ax.set_xlabel("main-effect variance fraction (eta^2)")
            ax.set_title(f"{ad.run_name}  annulus {ad.annulus + 1}  {label}: parameter importance  (n={cs['y'].size})", fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            pass
    return out_pdf


def render_paracoord(ad: AnnulusData, out_pdf: str, max_lines: int = 3000) -> str:
    """IDL ``eval_paracoord``: parallel coordinates (parameters normalised to [0, 1]),
    lines coloured by score, best in red; one page per partition."""
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import cm
    with _rc(), PdfPages(out_pdf) as pdf:
        for label, cs in _pages(ad):
            X, y = cs["X"], cs["y"]
            if X.size == 0:
                continue
            Z = (X - cs["lo"]) / np.where(cs["hi"] > cs["lo"], cs["hi"] - cs["lo"], 1.0)
            fig = Figure(figsize=(11, 5.5))
            ax = fig.add_subplot(111)
            norm = mcolors.Normalize(np.nanmin(y), np.nanmax(y))
            order = np.argsort(y)                                   # best drawn last
            if order.size > max_lines:
                order = order[-max_lines:]
            xs = np.arange(Z.shape[1])
            for r in order:
                ax.plot(xs, Z[r], color=cm.viridis(norm(y[r])), lw=0.5, alpha=0.35)
            ax.plot(xs, Z[order[-1]], color="red", lw=1.6, label=f"best {y[order[-1]]:.2f}")
            for j in xs:
                ax.axvline(j, color="k", lw=0.5, alpha=0.4)
            ax.set_xticks(xs)
            ax.set_xticklabels(cs["names"], rotation=45, ha="right", fontsize=8)
            ax.set_ylim(-0.02, 1.02); ax.set_ylabel("normalised parameter value")
            sm = cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
            fig.colorbar(sm, ax=ax, pad=0.01).set_label("search score")
            ax.set_title(f"{ad.run_name}  annulus {ad.annulus + 1}  {label}: parallel coordinates (colour = score)", fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            pass
    return out_pdf


def render_rank(ad: AnnulusData, out_pdf: str) -> str:
    """IDL ``eval_rank``: the two most important parameters (top eta^2) against each
    other, colour = score rank; one page per partition."""
    from matplotlib.backends.backend_pdf import PdfPages
    with _rc(), PdfPages(out_pdf) as pdf:
        for label, cs in _pages(ad):
            X, y = cs["X"], cs["y"]
            if X.shape[0] < 4 or X.shape[1] < 2:
                continue
            eta = np.array([eta_squared(X[:, j], y) for j in range(X.shape[1])])
            j1, j2 = np.argsort(np.nan_to_num(eta, nan=-1))[-2:][::-1]
            rank = np.argsort(np.argsort(y))
            fig = Figure(figsize=(7.5, 6))
            ax = fig.add_subplot(111)
            sc = ax.scatter(X[:, j1], X[:, j2], c=rank, cmap="viridis", s=12, alpha=0.85)
            b = np.nanargmax(y)
            ax.scatter([X[b, j1]], [X[b, j2]], marker="*", s=160, facecolor="none", edgecolor="red", lw=1.5, label="best")
            ax.set_xlabel(f"{cs['names'][j1]}  (eta^2 {eta[j1]:.2f})"); ax.set_ylabel(f"{cs['names'][j2]}  (eta^2 {eta[j2]:.2f})")
            fig.colorbar(sc, ax=ax).set_label("score rank (high = better)")
            ax.set_title(f"{ad.run_name}  annulus {ad.annulus + 1}  {label}: rank (top-2 eta^2 parameters)", fontsize=9)
            ax.legend(fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            pass
    return out_pdf


def render_slice(ad: AnnulusData, out_pdf: str) -> str:
    """IDL ``eval_slice``: score vs each parameter (all evaluations of the partition),
    best marked with a red line, binned median overplotted; one page per partition."""
    from matplotlib.backends.backend_pdf import PdfPages
    with _rc(), PdfPages(out_pdf) as pdf:
        for label, cs in _pages(ad):
            X, y = cs["X"], cs["y"]
            if X.size == 0:
                continue
            npar = X.shape[1]
            ncol = 3; nrow = int(np.ceil(npar / ncol))
            fig, axes = _subplots(nrow, ncol, figsize=(10, 3.0 * nrow), squeeze=False)
            b = int(np.nanargmax(y))
            for j, ax in enumerate(axes.flat):
                if j >= npar:
                    ax.axis("off"); continue
                x = X[:, j]
                ax.scatter(x, y, s=6, alpha=0.45, color="#1f77b4")
                g = np.isfinite(x) & np.isfinite(y)
                if g.sum() > 20:
                    edges = np.quantile(x[g], np.linspace(0, 1, 9))
                    mids, meds = [], []
                    for lo_, hi_ in zip(edges[:-1], edges[1:]):
                        m = g & (x >= lo_) & (x <= hi_)
                        if m.sum() >= 3:
                            mids.append(np.median(x[m])); meds.append(np.median(y[m]))
                    ax.plot(mids, meds, color="k", lw=1.2)
                ax.axvline(x[b], color="red", lw=1.0)
                ax.set_xlabel(cs["names"][j], fontsize=8); ax.set_ylabel("score", fontsize=8)
                ax.tick_params(labelsize=7)
            fig.suptitle(f"{ad.run_name}  annulus {ad.annulus + 1}  {label}: objective vs parameter (slices); red = best", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            pass
    return out_pdf


def render_products_sheet(run_dir: str, ia: int, out_pdf: str, run: Optional[Dict[str, Any]] = None) -> str:
    """One PDF with the annulus' final images (the IDL ``eval_best / eval_noinj /
    eval_nosub / eval_snrmap(_inj) / eval_stitch(_inj)`` set): winner injected + clean,
    per-partition stacks, S/N maps, and the run-level stitches when present."""
    from matplotlib.backends.backend_pdf import PdfPages
    ad = annulus_from_run(run_dir, ia, run)
    d = os.path.join(run_dir, f"annulus{ia + 1:02d}")
    items = [("winner: injected", os.path.join(d, "best_inj.fits")), ("winner: clean (no injection)", os.path.join(d, "best_clean.fits")),
             ("KLIP-FM response", os.path.join(d, "best_fm.fits")),
             ("stitch: clean", os.path.join(run_dir, "klip_stitched.fits")), ("stitch: injected", os.path.join(run_dir, "klip_stitched_inj.fits")),
             ("stitch: S/N (clean)", os.path.join(run_dir, "klip_stitched_snr.fits")), ("stitch: S/N (injected)", os.path.join(run_dir, "klip_stitched_snr_inj.fits"))]
    src = ad.winner.get("winner_sources") if ad.winner else None
    with _rc(), PdfPages(out_pdf) as pdf:
        fig, axes = _subplots(2, 4, figsize=(15, 7.6))
        k = 0
        for title, path in items:
            img = _fits(path)
            if img is None:
                continue
            if k >= 8:
                break
            ax = axes.flat[k]; k += 1
            draw_image(ax, img, ad, title, sources=src if "inj" in title else None,
                       snr="S/N" in title)
        for ax in axes.flat[k:]:
            ax.axis("off")
        fig.suptitle(f"{ad.run_name}  annulus {ia + 1}: final products", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig); pass
        # per-partition stacks
        for key, title in (("best_inj_partitions.fits", "winner per partition: injected"), ("best_clean_partitions.fits", "winner per partition: clean")):
            st = _fits(os.path.join(d, key))
            if st is None or st.ndim != 3:
                continue
            n = st.shape[0]
            fig, axes = _subplots(1, n, figsize=(3.2 * n, 3.6), squeeze=False)
            parts = (ad.winner or {}).get("partitions") or [str(i) for i in range(n)]
            for j in range(n):
                draw_image(axes[0][j], st[j], ad, f"{title.split(':')[1].strip()} {parts[j] if j < len(parts) else j}",
                           sources=src if "inj" in key else None)
            fig.suptitle(f"{ad.run_name}  annulus {ia + 1}: {title}", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig); pass
    return out_pdf


def render_verify_books(run_dir: str, ia: int, out_dir: Optional[str] = None,
                        run: Optional[Dict[str, Any]] = None, n_boot: int = 200) -> Dict[str, str]:
    """IDL ``verify_limits`` / ``verify_subsets_inj`` for one annulus, recomputed from
    the saved per-partition stacks (``best_*_partitions.fits``) and the winner's
    injections: (1) contrast limit vs separation per TAP, colour = FAP; (2) combined
    clean image + night-STIM map, and the five subset combines around every injection."""
    from matplotlib.backends.backend_pdf import PdfPages
    from .verify import (DEFAULT_FAP_LIST, DEFAULT_TAP_LIST, SUBSET_NAMES, contrast_limit, limit_tables,
                         verify_injections)
    ad = annulus_from_run(run_dir, ia, run)
    d = os.path.join(run_dir, f"annulus{ia + 1:02d}")
    out_dir = out_dir or d
    stack = _fits(os.path.join(d, "best_clean_partitions.fits"))
    istack = _fits(os.path.join(d, "best_inj_partitions.fits"))
    if stack is None:
        stack = _fits(os.path.join(d, "best_clean.fits"))
    if istack is None:
        istack = _fits(os.path.join(d, "best_inj.fits"))
    src = [tuple(map(float, s)) for s in ((ad.winner or {}).get("winner_sources") or [])]
    out: Dict[str, str] = {}
    if istack is None or not src:
        return out
    fwhm, px = float(ad.fwhm), float(ad.pxscale)
    rows, maps = verify_injections(istack, src, fwhm, px)
    # ---- page 1: limits vs separation (winner injections) + search-eval curve (all evals)
    ev_rows = []
    for i in range(len(ad.y)):
        for (rho, th, con), v in zip(ad.sources[i] or [], ad.per_source[i] or []):
            if v is not None and np.isfinite(v) and v > 0 and con > 0:
                ev_rows.append({"rho": rho, "theta": th, "contrast": con, "snr": float(v),
                                "nap": int(np.floor(2 * np.pi * (rho / px) / fwhm))})
    lim_ev = limit_tables(ev_rows, fwhm, px) if ev_rows else None
    fap_img = (0.05, 0.10, 0.20, 1.0 / 3.0, 0.50)                     # per-image FAPs (IDL legend)
    taps = (0.95, 0.90, 2.0 / 3.0, 0.50)
    cols = ["#3b8fd8", "#2e8b57", "#f5a623", "#d0021b", "#9013fe"]
    path1 = os.path.join(out_dir, "verify_limits.pdf")
    with _rc(), PdfPages(path1) as pdf:
        for label, lim in (("winner injections", maps.get("limits")), ("all search-eval injections", lim_ev)):
            if lim is None:
                continue
            smp = lim["samples"]
            fig, axes = _subplots(2, 2, figsize=(9, 7))
            for ax, tap in zip(axes.flat, taps):
                for fa, c in zip(fap_img, cols):
                    xs, ys = [], []
                    for b in lim["bins"]:
                        ww = np.abs(smp["rho"] - b["sep"]) <= lim["rbin_as"]
                        napm = max(float(np.mean(smp["nap"][ww])), 1.0)
                        ys.append(contrast_limit(fa / napm, tap, smp["contrast"][ww], smp["snr"][ww], smp["nap"][ww]))
                        xs.append(b["sep"])
                    ax.plot(xs, ys, "o-", ms=4, color=c, label=f"FAP = {fa*100:.0f}%/img")
                ax.set_yscale("log"); ax.set_xlim(0, max(ad.outrad * px * 1.1, 0.1))
                ax.set_title(f"TAP = {tap*100:.1f}%".replace(".0%", "%"))
                ax.set_xlabel("separation (arcsec)"); ax.set_ylabel("contrast (companion/star)")
            axes.flat[0].legend(fontsize=7, frameon=False)
            fig.suptitle(f"{ad.run_name}  annulus {ia + 1}: detection limits ({label}, n={lim['n_inj']})", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
    out["verify_limits"] = path1
    # ---- page 2: combined clean + night-STIM, subset stamps per injection
    path2 = os.path.join(out_dir, "verify_subsets_inj.pdf")
    cmaps = verify_injections(np.asarray(stack, float), src, fwhm, px)[1] if stack is not None else maps
    with _rc(), PdfPages(path2) as pdf:
        nsrc = len(rows)
        fig = Figure(figsize=(11, 3.4 + 2.4 * nsrc))
        gs = GridSpec(1 + nsrc, 5, figure=fig, height_ratios=[1.6] + [1.0] * nsrc)
        ax = fig.add_subplot(gs[0, 0:2]); draw_image(ax, cmaps["subsets"]["combined"], ad, "combined (clean)", sources=src)
        ax = fig.add_subplot(gs[0, 2:4]); draw_image(ax, cmaps["stim"], ad, "night-STIM", cmap="inferno", sources=src, flatten=False)
        ax = fig.add_subplot(gs[0, 4]); ax.axis("off")
        ax.text(0, 1, "\n".join([f"inj{r['index']}: rho={r['rho']:.2f}\"  PA={r['theta']:.0f}  c={r['contrast']:.1E}\n"
                                 f"   S/N={r['snr']:.2f}  FAP={r['fap']:.1E}\n   STIM={r['stim']:.2f}  STIM-FAP={r['stim_fap']:.1E}"
                                 for r in rows]), va="top", fontsize=7, family="monospace", transform=ax.transAxes)
        half = int(round(3.0 * fwhm))
        for m, r in enumerate(rows):
            x0, y0 = int(round(r["x"])), int(round(r["y"]))
            for j, name in enumerate(SUBSET_NAMES):
                ax = fig.add_subplot(gs[1 + m, j])
                img = maps["subsets"][name]
                ax.set_title(name, fontsize=8)
                if img is None or not np.isfinite(img).any():
                    ax.axis("off"); continue
                cut = img[max(y0 - half, 0):y0 + half + 1, max(x0 - half, 0):x0 + half + 1]
                v = cut[np.isfinite(cut)]
                lo, hi = (np.percentile(v, [1, 99.7]) if v.size else (0, 1))
                ax.imshow(cut, origin="lower", cmap="inferno", vmin=lo, vmax=hi)
                ax.add_patch(matplotlib.patches.Circle((x0 - max(x0 - half, 0), y0 - max(y0 - half, 0)), 0.9 * fwhm,
                                                       fill=False, ec="lime", lw=0.8))
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel(f"S/N {r['subsets'].get(name, float('nan')):.1f}", fontsize=8)
        fig.suptitle(f"{ad.run_name}  annulus {ia + 1}: winner injections, night subsets + STIM", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig)
    out["verify_subsets_inj"] = path2
    return out


def plot_annulus_books(run_dir: str, ia: int, run: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Post-hoc per-annulus books into ``run_dir/annulusNN/``: corner, landscapes,
    parameter-histogram book, EDF, partition map, k-book.  Returns ``{name: path}``."""
    ad = annulus_from_run(run_dir, ia, run)
    d = os.path.join(run_dir, f"annulus{ia + 1:02d}")
    os.makedirs(d, exist_ok=True)
    out = {}
    out["corner"] = render_corner(ad, os.path.join(d, "corner.pdf"))
    out["landscapes"] = render_landscapes(ad, os.path.join(d, "landscapes.pdf"))
    out["parhist"] = render_parhist_book(ad, os.path.join(d, "parhist.pdf"))
    out["edf"] = render_edf(ad, os.path.join(d, "edf.png"))
    out["nightmap"] = render_nightmap(ad, os.path.join(d, "partition_map.png"))
    out["kbook"] = render_kbook(ad, os.path.join(d, "kbook.pdf"))
    out["importance"] = render_importance(ad, os.path.join(d, "importance.pdf"))
    out["paracoord"] = render_paracoord(ad, os.path.join(d, "paracoord.pdf"))
    out["rank"] = render_rank(ad, os.path.join(d, "rank.pdf"))
    out["slice"] = render_slice(ad, os.path.join(d, "slice.pdf"))
    try:
        try:
            out.update(render_verify_books(run_dir, ia, run=run))
        except Exception as exc:  # pragma: no cover - diagnostic book only
            out["verify_error"] = repr(exc)
        out["products"] = render_products_sheet(run_dir, ia, os.path.join(d, "products.pdf"), run)
    except Exception:
        pass
    return out


def plot_calibration(run_dir: str, ia: int, run: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Post-hoc calibration panel from ``calibration.json`` (images: the seed eval has
    none on disk, so the winner FITS are shown with a note)."""
    ad = annulus_from_run(run_dir, ia, run)
    if ad.calibration is None:
        return None
    d = os.path.join(run_dir, f"annulus{ia + 1:02d}")
    seed = 0 if (ad.n and ad.phases[0] == "seed") else -1
    return render_calibration(ad, ad.calibration, os.path.join(d, "calibration_panel.png"),
                              sources=ad.sources[seed] if seed >= 0 else None,
                              per_source=ad.per_source[seed] if seed >= 0 else None,
                              clean_per_source=ad.clean_per_source[seed] if seed >= 0 else None)


def load_eval_images(run_dir: str, ia: int, i: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """The saved per-eval crops of eval ``i`` (0-based) of annulus ``ia`` re-embedded in a
    NaN frame of the full image size: ``(inj, clean)``; ``None`` where not saved."""
    out = []
    for tag in ("inj", "clean"):
        path = os.path.join(run_dir, f"annulus{ia + 1:02d}", "evals", f"eval{i + 1:04d}_{tag}.fits.gz")
        img = None
        if os.path.exists(path):
            try:
                from astropy.io import fits
                with fits.open(path) as h:
                    crop = np.asarray(h[0].data, float)
                    hd = h[0].header
                    full = np.full((int(hd.get("FULLNY", crop.shape[0])), int(hd.get("FULLNX", crop.shape[1]))), np.nan)
                    x0, y0 = int(hd.get("CROPX0", 0)), int(hd.get("CROPY0", 0))
                    full[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
                    img = full
            except Exception:
                img = None
        out.append(img)
    return out[0], out[1]


def render_steps(run_dir: str, annuli: Optional[Sequence[int]] = None, every: int = 1, dpi: int = 90,
                 out_dir: Optional[str] = None, movie: bool = True,
                 partition_label: Optional[str] = None) -> List[str]:
    """Rebuild the step PNGs of a finished run from its log (no live images: the image
    cells show the annulus winner FITS with a note).  Optionally assembles the movie.
    ``partition_label`` overrides what the partition panels call the partitions, for a run
    written before the reducer carried one."""
    run = load_run(run_dir)
    out_dir = out_dir or os.path.join(run_dir, "steps")
    os.makedirs(out_dir, exist_ok=True)
    try:
        with open(os.path.join(run_dir, "run_setup.json")) as f:
            fm_note = _fm_unavailable_note(json.load(f).get("reducer"))
    except Exception:
        fm_note = None
    from .plots import _annuli
    ias = list(annuli) if annuli is not None else _annuli(run)
    paths: List[str] = []
    step = 0
    fig = None
    for ia in ias:
        full = annulus_from_run(run_dir, ia, run)
        d = os.path.join(run_dir, f"annulus{ia + 1:02d}")
        b_inj, b_clean = _fits(os.path.join(d, "best_inj.fits")), _fits(os.path.join(d, "best_clean.fits"))
        b_fm = _fits(os.path.join(d, "best_fm.fits"))
        note = "winner FITS (rebuilt)" if b_inj is not None else "no images in log"
        wcurve = (full.winner or {}).get("contrast_curve")
        fmcurve = (full.winner or {}).get("fm_curve")
        for i in range(full.n):
            if i % max(every, 1) and i != full.n - 1:
                continue
            ad = annulus_from_run(run_dir, ia, run, upto=i + 1)
            if partition_label:
                ad.partition_label = str(partition_label)
            ad.winner = None if i < full.n - 1 else full.winner
            bi, _ = ad.best()
            c_inj, c_clean = load_eval_images(run_dir, ia, i)
            e_inj, e_clean = load_eval_images(run_dir, ia, bi) if bi != i else (c_inj, c_clean)
            have = c_inj is not None
            imgs = StepImages(fm_unavailable=fm_note, cur_clean=c_clean, cur_inj=c_inj,
                              best_inj=e_inj if e_inj is not None else b_inj,
                              best_clean=e_clean if e_clean is not None else b_clean, best_index=bi,
                              note=None if have else note, fm_image=b_fm if i == full.n - 1 else None)
            curves = []
            if have and i < full.n - 1:
                lc = _live_curve(c_clean, ad, i)
                if lc:
                    curves.append((f"eval {i + 1} preview", lc, {}))
            if wcurve and i == full.n - 1:
                curves.append(("validated winner (annulus)", wcurve, {"color": VALID_COLOR, "lw": 1.6}))
                if fmcurve:
                    curves.append(("KLIP-FM cross-check", fmcurve, {"color": FM_COLOR, "lw": 1.2}))
            p = os.path.join(out_dir, f"step{step:04d}.png")
            fig = render_step(ad, i, imgs, p, curves=curves, elapsed_s=float(np.nansum(full.wall[:i + 1])),
                              eta_s=float(np.nanmean(full.wall[:i + 1]) * (full.n - i - 1)) if i + 1 < full.n else 0.0,
                              dpi=dpi, fig=fig, step_label="(rebuilt from log)",
                              contrast_note=None if have else "rebuilt from log: no live images, so no per-eval preview")
            paths.append(p)
            step += 1
    if movie and paths:
        try:
            from .animate import make_movie
            setup = run.get("setup") or run.get("checkpoint") or {}
            intro = render_intro(dict(setup, run_dir=run_dir), os.path.join(out_dir, "intro.png"))
            make_movie(paths, os.path.join(run_dir, "opt_steps.gif"), os.path.join(run_dir, "opt_steps.mp4"), intro=intro)
        except Exception:
            pass
    return paths


# ----------------------------------------------------------------------------
# the live callback
# ----------------------------------------------------------------------------
def injected_model_image(runner, sources, shape) -> Optional[np.ndarray]:
    """IDL ``disp_injmdl``: the injected-PSF model of ``sources`` (rho, theta, contrast)
    as it appears in the final (north-up) image.

    Built at the data's OWN roll angles, then derotated and combined the way the science
    frames are.  It used to inject into a single ``parang = 0`` frame and derotate by the
    true-north offset alone, which silently assumed the telescope had been pointed at
    roll 0.  That was invisible while the injector span the stamp by the source's
    azimuth, and glaring once the template became spacecraft-fixed (``815fc55``): the
    lobe structure is carried into the sky frame entirely by the derotation, so a model
    built at roll 0 shows its lobes rotated away from the data by the roll itself --
    measured at ~112 degrees on HIP 65426's rolls of 108.0 and 117.4.

    Frames sharing a roll give the same picture, so only the distinct angles are rotated,
    weighted by how many frames carry each.  A multi-roll data set therefore shows the
    superposition that is really in the combined image, not one arbitrary roll's version.
    """
    try:
        from .injection import inject_sources
        from .klip import rotate_ccw
        from .metrics import Source
        red = runner.reducer
        sub = red if getattr(red, "model", None) is not None else next(
            (r for r in getattr(red, "reducers", {}).values() if getattr(r, "model", None) is not None), None)
        if sub is None or not sources or shape is None or len(shape) != 2:
            return None
        src = [Source(*tuple(float(v) for v in (s.as_tuple() if hasattr(s, "as_tuple") else s)[:3])) for s in sources]
        tn = float(getattr(sub, "truenorth", 0.0))
        try:
            ang = np.asarray(sub.frame_angles(), float).ravel()
            ang = ang[np.isfinite(ang)]
        except Exception:
            ang = np.zeros(0)
        if ang.size == 0:
            ang = np.zeros(1)
        uang, cnt = np.unique(np.round(ang, 2), return_counts=True)
        blank = np.zeros((uang.size,) + tuple(int(v) for v in shape), np.float32)
        img = inject_sources(blank, uang, src, sub.model, sub.pxscale, truenorth=tn,
                             fallback=getattr(sub, "fallback", None),
                             angle_convention=getattr(sub, "angle_convention", "pa"))
        # cval=0 rather than derotate()'s NaN fill: this is a display model, and NaN
        # corners would drag the panel's robust stretch around.
        der = np.stack([rotate_ccw(np.asarray(img[i], float), float(uang[i]) + tn, cval=0.0)
                        for i in range(uang.size)])
        return np.average(der, axis=0, weights=cnt.astype(float))
    except Exception:
        return None


class LiveDisplay(RunCallback):
    """Live display callback.

    Parameters
    ----------
    run_dir
        where to write (``steps/``, ``annulusNN/``, ``opt_steps.gif``).
    every
        render cadence in evaluations (a new best and the last evaluation of an
        annulus always render).
    pdf_every
        write ``annulusNN/eval_NNNN_panel.pdf`` every this many evaluations.
    save_png, show, cmap, dpi
        as named; ``show=True`` keeps an interactive window (``plt.ion``).
    movie
        assemble ``opt_steps.gif`` (and ``.mp4`` when an ffmpeg backend exists) at the end.
    """

    def __init__(self, run_dir: str, every: int = 1, pdf_every: int = 10, save_png: bool = True, show: bool = False,
                 cmap: str = "inferno", dpi: int = 100, movie: bool = True, log=None, aliens: bool = False,
                 window_scale: float = 1.0, movie_every: Optional[int] = None):
        """``aliens=True`` plays IDL's five-act launch movie in the live window during the
        first annulus' calibration and writes ``intro_frames/`` + ``intro.gif``; off by
        default (the calibration panel alone is shown).  ``movie_every`` (evaluations;
        default = ``pdf_every``) rebuilds the running per-annulus progress movie
        ``annulusNN/progress.gif`` / ``.mp4`` (IDL ``annNN_progress.gif`` on the pdf batch);
        0 = only at annulus end."""
        self.run_dir = run_dir
        self.movie_every = max(int(pdf_every if movie_every is None else movie_every), 0)
        self._ann_frames: List[str] = []
        self.aliens = bool(aliens)
        self.window_scale = float(window_scale)     # live window = panel size x this (1.0 = the IDL 1850 x 990 window, 1:1 pixels)
        self.every = max(int(every), 1)
        self.pdf_every = max(int(pdf_every), 0)
        # show: False | True/"window" (matplotlib GUI window) | "inline" (update an output cell in
        # place, Jupyter) | "auto" (inline inside a notebook kernel, window otherwise)
        mode = show if isinstance(show, str) else ("window" if show else "")
        mode = mode.lower()
        if mode == "auto":
            mode = "inline" if _in_notebook() else "window"
        # "inline" outside a Jupyter kernel used to push the panel to an IPython display
        # handle that renders nothing in a terminal -- no window, no warning, and no hint
        # that a live display exists at all.  The tutorials are .py files as well as
        # notebooks, so a student running one as a script hit exactly that.  Fall back to
        # the window and say so.
        self._inline_fallback = mode == "inline" and not _in_notebook()
        if self._inline_fallback:
            mode = "window"
        self.inline = mode == "inline"
        self.save_png, self.show, self.cmap, self.dpi, self.movie = save_png, bool(mode), cmap, int(dpi), movie
        self._handle = None                          # IPython display handle (inline mode)
        self._log = log
        self._records: List[Dict[str, Any]] = []
        self._ia = -1
        self._step = self._next_step()
        self._paths: List[str] = []
        self._fig = None
        self._best_curve: Optional[Dict[str, Any]] = None
        self._live_reason: Optional[str] = None                 # why the last live preview could not be built
        self._last_images: Dict[str, Any] = {}                  # last on_eval images (annulus-done frame)
        self._calib_info: Optional[Dict[str, Any]] = None
        self._calib_pending = False
        self._val_args: Optional[Dict[str, Any]] = None
        self._t0 = time.time()
        self._setup: Dict[str, Any] = {}

    def _next_step(self) -> int:
        """Carry on numbering panels where the run left off.

        A resumed run used to restart at ``step0000.png`` and overwrite the panels of every
        earlier segment one by one -- losing the visual history and scrambling the progress
        movie.  The panels on disk are the record of what the run has done, so a restart
        appends to them.
        """
        import glob as _glob
        import re as _re
        n = -1
        for p in _glob.glob(os.path.join(self.run_dir, "steps", "step*.png")):
            m = _re.search(r"step(\d+)\.png$", os.path.basename(p))
            if m:
                n = max(n, int(m.group(1)))
        return n + 1

    # -- plumbing ------------------------------------------------------------------
    _inj_model: Optional[np.ndarray] = None
    _inj_model_for: int = -1

    def _say(self, runner, msg: str) -> None:
        try:
            (self._log or (runner.log if runner is not None else print))(f"  [display] {msg}")
        except Exception:
            pass

    def _guard(self, runner, what: str, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as exc:
            self._say(runner, f"{what} failed: {exc!r}\n" + traceback.format_exc(limit=3))
            return None

    def _ann_dir(self, ia: int) -> str:
        d = os.path.join(self.run_dir, f"annulus{ia + 1:02d}")
        os.makedirs(d, exist_ok=True)
        return d

    def _interactive_backend(self, runner) -> bool:
        """Make sure pyplot has a GUI backend for ``show``; fall back to PNG-only with a
        log line when none is available (e.g. a headless Python or no GUI toolkit)."""
        if getattr(self, "_gui_checked", False) or self.inline:
            return self.show
        self._gui_checked = True
        if getattr(self, "_inline_fallback", False):
            self._say(runner, 'live display: show="inline" only works inside a Jupyter kernel -- '
                              "opening a window instead")
        import matplotlib
        matplotlib.rcParams["toolbar"] = "None"        # no home/arrows bar under the live panel
        cur = matplotlib.get_backend().lower()
        import sys as _sys
        forced = os.environ.get("KLIP_TPE_BACKEND")
        if forced:
            prefer = (forced,)
        elif _sys.platform == "darwin":
            prefer = ("MacOSX", "QtAgg", "TkAgg")
        else:
            prefer = ("QtAgg", "TkAgg", "GTK3Agg")
        if not forced and cur in ("macosx", "qtagg", "qt5agg", "tkagg", "gtk3agg", "gtk4agg", "wxagg"):
            return True
        for cand in prefer:
            try:
                matplotlib.use(cand, force=True)
                import matplotlib.pyplot as plt
                plt.figure(); plt.close()
                self._say(runner, f"live window: using the {cand} backend")
                return True
            except Exception:
                continue
        self._say(runner, "live window: no interactive matplotlib backend available (MacOSX/Qt/Tk) -- "
                          "showing nothing on screen; the panel is still written to steps/stepNNNN.png. "
                          "Install a GUI toolkit (pip install PyQt6) or run inside a framework Python.")
        self.show = False
        return False

    def _figure(self) -> Figure:
        """Offscreen (Agg) figure for the step panel -- used by the render thread only."""
        if self._fig is None:
            self._fig = Figure(figsize=panel_size(self.dpi), dpi=self.dpi)
            FigureCanvasAgg(self._fig)
        return self._fig

    def _figure2(self) -> Figure:
        """Second offscreen figure for what the main thread draws (intro, calibration panel)."""
        if getattr(self, "_fig2", None) is None:
            self._fig2 = Figure(figsize=panel_size(self.dpi), dpi=self.dpi)
            FigureCanvasAgg(self._fig2)
        return self._fig2

    # -- the live window is a viewer: it only ever blits a rendered PNG ------------------
    #
    # Deliberately CLASS attributes, so every LiveDisplay in a process shares one window.  A
    # benchmark is dozens of runs one after another, each with its own run directory and so
    # its own LiveDisplay; per-instance windows would open twenty-four of them and leave them
    # open, since on_finish tears down the render pool but never the window.  Sharing makes
    # the window follow whichever run is going, which is what watching a batch wants.  Only
    # one display is ever live at a time in a process (a batch runs its slots sequentially),
    # so there is nothing to contend over.
    _win = None
    _win_ax = None
    _win_im = None
    _win_dir = None

    def _window_show(self, png: Optional[str] = None, fig: Optional[Figure] = None) -> None:
        """Show ``png`` (or the current pixels of offscreen ``fig``) in the live window.
        Main thread only; a plain imshow, so it costs ~0.1 s whatever the panel is."""
        if not self.show:
            return
        if self.inline:
            self._inline_show(png=png, fig=fig)
            return
        if not getattr(self, "_gui_checked", False) and not self._interactive_backend(None):
            return                                  # also pins rcParams["toolbar"] = "None" before the window exists
        try:
            import matplotlib.pyplot as plt
            if fig is not None:
                fig.canvas.draw()
                img = np.asarray(fig.canvas.buffer_rgba())
            elif png:
                img = plt.imread(png)
            else:
                return
            C = LiveDisplay                       # the window lives on the class (see above)
            if C._win is None or not plt.fignum_exists(C._win.number):
                plt.ion()
                w_in, h_in = panel_size(self.dpi)
                C._win = plt.figure(figsize=(w_in * self.window_scale, h_in * self.window_scale), dpi=self.dpi)
                C._win.patch.set_facecolor("black")
                ax = C._win.add_axes([0, 0, 1, 1]); ax.set_axis_off()
                C._win_ax, C._win_im, C._win_dir = ax, None, None
                plt.show(block=False)
            if C._win_dir != self.run_dir:         # a batch moved on to the next run
                C._win_dir = self.run_dir
                try:
                    C._win.canvas.manager.set_window_title(f"klip-tpe live -- {os.path.basename(self.run_dir)}")
                except Exception:
                    pass
            if C._win_im is None or C._win_im.get_array().shape[:2] != img.shape[:2]:
                C._win_ax.clear(); C._win_ax.set_axis_off()
                # lanczos: crisp text when the 1870-px panel is resampled to the window (nearest blurred it)
                interp = "nearest" if abs(self.window_scale - 1.0) < 1e-3 else "lanczos"
                C._win_im = C._win_ax.imshow(img, interpolation=interp, aspect="auto",
                                             interpolation_stage="rgba", resample=True)
            else:
                C._win_im.set_data(img)
            C._win.canvas.draw_idle()
            C._win.canvas.flush_events()           # no show()/pause -> the window is never raised
            # the WINDOW is shared, the pump clock is not: a display that has just taken the
            # window over should service the event loop at once rather than inherit the
            # previous one's throttle (and shared clocks leak between runs in one process)
            self._last_pump = time.time()
            self._arm_pump()
        except Exception as exc:
            self._say(None, f"live window update failed: {exc!r}")
            self.show = False

    def _inline_show(self, png: Optional[str] = None, fig: Optional[Figure] = None) -> None:
        """Jupyter: push the panel into one output cell, updated in place (an IPython
        display handle), scaled to ``window_scale`` x the panel width."""
        try:
            from IPython.display import Image, display
        except Exception as exc:
            self._say(None, f"inline display needs IPython ({exc!r}); showing nothing")
            self.show = False
            return
        try:
            if fig is not None:
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=self.dpi, facecolor=fig.get_facecolor())
                data = buf.getvalue()
            elif png and os.path.exists(png):
                with open(png, "rb") as f:
                    data = f.read()
            else:
                return
            w = int(round(PANEL_PX[0] * self.window_scale))
            img = Image(data=data, format="png", width=w)
            if self._handle is None:
                self._handle = display(img, display_id=True)
            else:
                self._handle.update(img)
        except Exception as exc:
            self._say(None, f"inline display update failed: {exc!r}")
            self.show = False

    # -- background rendering of the step panel ------------------------------------------
    def _render_pool(self):
        import concurrent.futures as cf
        if getattr(self, "_rex", None) is None:
            self._rex = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="klip-render")
            self._renders: List[Tuple[Any, str]] = []
        return self._rex

    def _submit_render(self, png: Optional[str], fn, *args, **kw) -> None:
        fut = self._render_pool().submit(fn, *args, **kw)
        self._renders.append((fut, png))
        from .parallel import IDLE_HOOKS
        if self.show and self._show_latest not in IDLE_HOOKS:
            IDLE_HOOKS.append(self._show_latest)

    #: how long to let the render thread finish before giving up on this evaluation's panel
    RENDER_GRACE = 0.25
    _skipped_panels = 0
    _skip_warned = False

    def _render_room(self) -> bool:
        """True when a panel can be drawn now without putting one behind another.

        The window is a live view, so its lag must be bounded by how long *one* panel takes
        to draw -- never by how long the run has been going.  ``--display-every 1`` asks for
        a panel per evaluation; a NEAR panel takes far longer to draw than an evaluation
        takes to run, so without a gate the single render thread accumulated a queue that
        grew for the whole run.  The thread was always drawing a frame from minutes ago,
        the window drifted further behind the longer it lasted (~250 evaluations stale and
        still growing when it was reported), and every queued job held its image arrays
        alive.

        The gate belongs here, before the panel is promised, rather than on the queue:
        a panel that was never requested costs nothing, whereas cancelling one already
        promised would leave a hole in ``steps/`` that the progress movie and the resumed
        history both read.  So a busy render thread is given a short grace period -- long
        enough that a fast panel simply finishes and nothing is skipped at all -- and this
        evaluation goes without a panel only if the thread is still busy afterwards.
        """
        from .parallel import run_idle_hooks
        deadline = time.time() + self.RENDER_GRACE
        while True:
            # getattr: the gate runs on the first evaluation, before any render pool exists
            if not any(not f.done() for f, _ in (getattr(self, "_renders", None) or ())):
                return True
            if time.time() >= deadline:
                return False
            run_idle_hooks()                      # the window stays alive while we wait
            time.sleep(0.02)

    #: which landscape the live panel shows: "cycle" steps through the partitions one panel
    #: at a time (IDL), "pooled" overlays them all, a partition id pins one
    corner_mode: str = "cycle"
    _corner_i: int = 0

    def _next_corner_partition(self, ad) -> Optional[str]:
        """The partition whose landscape this panel should show.

        Pooling every partition into one corner overlays landscapes that need not agree --
        a parameter can be good on one night and poor on the next, and the pooled cloud is
        then a blur true of no night in particular.  So the panel shows one at a time and
        advances by one each frame, and the title says which.
        """
        parts = [str(p) for p in (getattr(ad, "partitions", None) or [])]
        if self.corner_mode == "pooled" or not getattr(ad, "replicated", False) or len(parts) < 2:
            return None
        if self.corner_mode not in ("cycle", "", None):
            return self.corner_mode if self.corner_mode in parts else None
        pid = parts[self._corner_i % len(parts)]
        self._corner_i += 1
        return pid

    def _skip_panel(self) -> None:
        """Record, and explain once, an evaluation that went without a panel."""
        self._skipped_panels += 1
        if self._skip_warned:
            return
        self._skip_warned = True
        self._say(None, f"a panel takes longer to draw than an evaluation takes to run, so some "
                        f"evaluations are going without one -- this keeps the window showing the "
                        f"current state instead of falling further behind. The run is unaffected; "
                        f"draw every panel afterwards with  klip-tpe render --run-dir {self.run_dir}")

    #: the main thread must hand the GUI a turn at least this often or macOS stops
    #: compositing the window (it is marked unresponsive after ~2 s)
    PUMP_EVERY = 0.05
    PUMP_WARN = 8.0
    _last_pump = 0.0
    _pump_warned = False

    def _pump(self) -> None:
        """Give the GUI event loop a turn.  Main thread only, throttled, never raises.

        This is what keeps the live window from freezing.  Rendering happens off-thread
        and the window only blits, but the blit and the event loop belong to the main
        thread -- and between renders (one per ``every`` evaluations, tens of seconds) the
        main thread is inside NumPy.  macOS marks a process that has not serviced events
        for ~2 s as unresponsive, stops compositing it and leaves the last painted frame on
        screen.  Pumping on every idle tick, not only when a new frame is ready, is the
        difference between a live window and a stale one.
        """
        if not self.show or self.inline or self._win is None:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        now = time.time()
        gap = now - (self._last_pump or now)
        if now - (self._last_pump or 0.0) < self.PUMP_EVERY:
            return
        self._last_pump = now               # per display, not per window (see _window_show)
        if gap > self.PUMP_WARN and not self._pump_warned:
            self._pump_warned = True
            self._say(None, f"the live window went {gap:.0f}s without a GUI turn, so it may look "
                            f"frozen (the run is unaffected). A window that cannot freeze: "
                            f"klip-tpe view --run-dir {self.run_dir}")
        try:
            import matplotlib.pyplot as plt
            if plt.fignum_exists(self._win.number):
                self._win.canvas.flush_events()
        except Exception:
            pass

    def _arm_pump(self) -> None:
        """Register the idle hook as soon as the window exists, so the event loop is
        serviced from the first evaluation rather than from the first render."""
        from .parallel import IDLE_HOOKS
        if self.show and not self.inline and self._show_latest not in IDLE_HOOKS:
            IDLE_HOOKS.append(self._show_latest)

    def _show_latest(self) -> None:
        """Main thread: put the newest finished render in the window (idle hook + on_eval)."""
        self._pump()                     # always, even with no render pending
        if not getattr(self, "_renders", None):
            return
        latest = None
        keep = []
        for fut, png in self._renders:
            if fut.done():
                try:
                    fut.result()
                    if png:
                        latest = png
                except Exception as exc:
                    self._say(None, f"render failed: {exc!r}")
            else:
                keep.append((fut, png))
        self._renders = keep
        if latest and self.show and not self._intro_on:
            self._window_show(png=latest)

    def _wait_renders(self) -> None:
        for fut in list(getattr(self, "_movies", []) or []):
            try:
                fut.result()
            except Exception:
                pass
        for fut, _ in list(getattr(self, "_renders", []) or []):
            try:
                fut.result()
            except Exception:
                pass
        self._show_latest()

    def _running_stitch(self, runner) -> Dict[str, Any]:
        """Running-stitch images for the stitched cells when there is more than one annulus
        (annulus 1 shows the best images, as IDL does)."""
        if getattr(runner, "ia", 0) <= 0:
            return {}
        out: Dict[str, Any] = {}
        for key, name in (("stitch_clean", "klip_stitched_running.fits"), ("stitch_inj", "klip_stitched_running_inj.fits")):
            img = _fits(os.path.join(self.run_dir, name))
            if img is not None and img.ndim == 2:
                out[key] = img
        if "stitch_inj" in out:
            src = []
            for r in getattr(runner, "results", []) or []:
                if r is not None:
                    src.extend(tuple(t) for t in getattr(r, "winner_sources", []) or [])
            b = self._best_images(runner)
            brec = b.get("record")
            if brec is not None and getattr(brec, "sources", None):
                src.extend(t.as_tuple() if hasattr(t, "as_tuple") else tuple(t) for t in brec.sources)
            out["stitch_sources"] = src or None
        return out

    def _data(self, runner) -> AnnulusData:
        if not self._records and getattr(runner, "history", None) is not None and len(runner.history):
            # resumed mid-annulus: the live record list is empty, take the annulus from the log on disk
            try:
                return annulus_from_run(self.run_dir, runner.ia)
            except Exception:
                pass
        return annulus_from_runner(runner, self._records)

    def _curves(self, runner, ad: AnnulusData, live: Optional[Dict[str, Any]], done: bool = False):
        """Contrast-panel curves: this eval's preview, the best eval's preview (+ its
        KLIP-FM preview with ``fm_preview``) and, once the annulus is finished, the
        validated injection curve together with the KLIP-FM cross-check curve."""
        out = []
        if live:
            out.append(("this eval (preview from its sources)", live, {"color": CUR_COLOR, "lw": 0.9, "ls": "--"}))
        if self._best_curve:
            out.append(("best eval (preview)", self._best_curve, {"color": PHASE_COLORS["tpe"], "lw": 1.3}))
        fm_prev = (self._best_images(runner).get("fm") or {}).get("curve") if not done else None
        if fm_prev:
            out.append(("KLIP-FM preview (best eval)", fm_prev, {"color": FM_COLOR, "lw": 1.0, "ls": "--"}))
        # validated winners of EVERY finished annulus stay on the panel (IDL's cumulative curve)
        for r in runner.results:
            if r is None:
                continue
            tag = f" annulus {r.annulus + 1}" if r.annulus != ad.annulus else ""
            if r.contrast_curve:
                out.append((f"validated winner (injection-calibrated){tag}", r.contrast_curve,
                            {"color": VALID_COLOR, "lw": 1.6}))
            if getattr(r, "fm_curve", None):
                out.append((f"KLIP-FM cross-check (winner){tag}", r.fm_curve, {"color": FM_COLOR, "lw": 1.4}))
        return out

    @staticmethod
    def _best_images(runner) -> Dict[str, Any]:
        b = getattr(runner, "best_images", None)
        if b is None:
            b = getattr(runner, "_best_images", None)
        return b or {}

    # -- hooks ---------------------------------------------------------------------
    # -- launch movie (IDL near2m_intro_frame) -------------------------------------
    _intro_f: int = 0
    _intro_on: bool = False
    _intro_last: float = 0.0

    def _intro_tick(self) -> None:
        """Advance the launch movie one frame in the live window (called from the main
        thread while it waits for worker results)."""
        if not self._intro_on or not self.show:
            return
        from .intro import FRAME_DT, N_RELEASE, draw_intro_frame
        now = time.time()
        if now - self._intro_last < FRAME_DT:
            return
        self._intro_last = now
        try:
            fig = self._figure2()
            draw_intro_frame(fig, self._intro_f)
            self._window_show(fig=fig)
            if self.save_png and self._intro_f % 2 == 0 and getattr(self, "_intro_dir", None):
                fig.savefig(os.path.join(self._intro_dir, f"introframe_{self._intro_f:05d}.png"), dpi=80,
                            facecolor=fig.get_facecolor())
        except Exception as exc:
            self._say(None, f"intro stopped: {exc!r}")
            self._intro_on = False
            return
        self._intro_f += 1
        if self._intro_f >= N_RELEASE:
            self._intro_on = False                 # hand the window to the calibration panel
            self._intro_gif()

    def _intro_gif(self) -> None:
        """Assemble intro.gif from whatever intro frames were captured."""
        d = getattr(self, "_intro_dir", None)
        if not d or not os.path.isdir(d):
            return
        paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("introframe_") and f.endswith(".png"))
        if len(paths) < 2 or os.path.exists(os.path.join(self.run_dir, "intro.gif")):
            return
        try:
            from .animate import make_movie
            from .intro import FRAME_DT
            make_movie(paths, os.path.join(self.run_dir, "intro.gif"), None, fps=1.0 / (FRAME_DT * 2 * 1.2))
        except Exception:
            pass

    def _intro_stop(self) -> None:
        was_on = self._intro_on
        self._intro_on = False
        if was_on:
            self._intro_gif()
        try:
            from .parallel import IDLE_HOOKS
            if self._intro_tick in IDLE_HOOKS:
                IDLE_HOOKS.remove(self._intro_tick)
        except Exception:
            pass

    # ---- ETA bookkeeping (IDL: measured per-annulus calib + validation reserves) ---------------
    def _eta_clock(self, runner) -> Dict[str, Any]:
        c = getattr(self, "_eta", None)
        if c is None:
            c = self._eta = dict(ia=-1, ann_t0=np.nan, search_t0=np.nan, val_t0=np.nan, val_times=[],
                                 t_cal_tot=0.0, n_cal=0, t_val_tot=0.0, n_val=0)
        if c["ia"] != runner.ia:                       # new annulus: its calibration starts now
            c.update(ia=runner.ia, ann_t0=time.time(), search_t0=np.nan, val_t0=np.nan, val_times=[])
        return c

    def _eta_mark(self, runner, phase: str) -> None:
        """``phase`` = 'cal' (a calibration trial), 'search' (an evaluation), 'val' (a
        validation trial) or 'done' (annulus finished): closes the calibration /
        validation stopwatches like IDL's ``t_cal_tot`` / ``t_val_tot``."""
        c = self._eta_clock(runner)
        now = time.time()
        if phase == "search" and not np.isfinite(c["search_t0"]):
            c["search_t0"] = now
            if np.isfinite(c["ann_t0"]):
                c["t_cal_tot"] += now - c["ann_t0"]; c["n_cal"] += 1
        elif phase == "val":
            if not np.isfinite(c["val_t0"]):
                c["val_t0"] = now
            c["val_times"].append(now)
        elif phase == "done" and np.isfinite(c["val_t0"]):
            c["t_val_tot"] += now - c["val_t0"]; c["n_val"] += 1
            c["val_t0"] = np.nan

    def _eta_estimate(self, runner, ad: AnnulusData, per: float, in_validation: bool = False,
                      trials_left: int = 0) -> Tuple[float, float]:
        """IDL ``rem_ann`` / ``rem`` (optimize_near_2_tpe, 'ETA / elapsed'): remaining wall
        time for this annulus and for the whole run.  Remaining evals (this annulus +
        future annuli) at the current per-eval loop time, plus the measured calibration and
        validation time per annulus for every future annulus, this annulus' pending
        validation (``cal_avg * n_top * (1 + n_valid) / 3`` until the first one is
        measured; during validation the measured per-trial time x trials left) and a
        reserve for still-available re-calibration restarts (``recal_check`` evals each)."""
        c = self._eta_clock(runner)
        cfg = runner.cfg
        av = float(per) if np.isfinite(per) else np.nan
        if not np.isfinite(av):
            return np.nan, np.nan
        vc, cc = cfg.validation, cfg.calibration
        n_fut = max(cfg.nann - 1 - runner.ia, 0)
        rem_this = 0 if in_validation else max(ad.n_iter - ad.n, 0)
        rem_fut = sum(cfg.per_annulus(cfg.n_iter, j) for j in range(runner.ia + 1, cfg.nann))
        cal_avg = c["t_cal_tot"] / c["n_cal"] if c["n_cal"] else 0.0
        has_val = vc.n_top >= 1 and vc.n_valid >= 1
        val_est = cal_avg * (vc.n_top * (1.0 + vc.n_valid)) / 3.0 if has_val else 0.0
        val_avg = c["t_val_tot"] / c["n_val"] if c["n_val"] else val_est
        if in_validation:
            vt = c["val_times"]
            per_trial = (vt[-1] - c["val_t0"]) / max(len(vt) - 1, 1) if len(vt) >= 2 else av
            val_this = per_trial * max(trials_left, 0)
        else:
            val_this = val_avg
        ovh_fut = (cal_avg + val_avg) * n_fut
        recal_left = max(cc.recal_budget - getattr(runner, "_recal_done", 0), 0) if cc.recal_check > 0 else 0
        recal_rsv = recal_left * cc.recal_check * av if (not in_validation and ad.n < cc.recal_check) else 0.0
        rem_ann = av * rem_this + val_this + recal_rsv
        rem_full = av * (rem_this + rem_fut) + val_this + ovh_fut + recal_rsv
        return float(rem_ann), float(rem_full)

    def on_setup(self, runner) -> None:
        self._t0 = time.time()
        if self.show:
            self._interactive_backend(runner)
        try:
            p = os.path.join(runner.run_dir, "run_setup.json")
            with open(p) as f:
                self._setup = json.load(f)
        except Exception:
            self._setup = {"config": runner.cfg.to_dict(), "space": runner.space.to_dict(), "run_dir": runner.run_dir}
        try:
            self._fm_note = _fm_unavailable_note(runner.reducer.describe())
        except Exception:
            self._fm_note = None
        resumed = bool(getattr(runner, "_resumed", False))
        if not resumed:
            self._guard(runner, "intro", render_intro, dict(self._setup, run_dir=runner.run_dir),
                        os.path.join(self.run_dir, "steps", "intro.png"), dpi=self.dpi)
        if self.show:
            # open the window now: on a resume with the newest frame of the run (the next
            # evaluation may be minutes away), on a fresh start with the intro card
            sd = os.path.join(self.run_dir, "steps")
            frames = sorted(f for f in (os.listdir(sd) if os.path.isdir(sd) else []) if f.startswith("step") and f.endswith(".png"))
            first = os.path.join(sd, frames[-1]) if frames else os.path.join(sd, "intro.png")
            if os.path.exists(first):
                self._guard(runner, "window", self._window_show, first)
        # launch movie (opt-in): with a live window the frames are captured from the window as
        # it plays (IDL cgsnapshot); otherwise they are rendered here, on the main thread -- a
        # background thread would garbage-collect Tk objects off the GUI thread.
        if self.aliens and self.save_png and getattr(runner, "ia", 0) == 0 and not resumed:
            self._intro_dir = os.path.join(self.run_dir, "intro_frames")
            os.makedirs(self._intro_dir, exist_ok=True)
            if not self.show:
                from .intro import render_intro_frames
                self._guard(runner, "intro movie", render_intro_frames, self.run_dir, log=lambda m: self._say(runner, m))
        if self.aliens and self.show and getattr(runner, "ia", 0) == 0 and not resumed:
            from .parallel import IDLE_HOOKS
            self._intro_on, self._intro_f = True, 0
            if self._intro_tick not in IDLE_HOOKS:
                IDLE_HOOKS.append(self._intro_tick)

    def on_calibration_trial(self, runner, ia: int, trial: int, contrast: float, msnr: float, images) -> None:
        """IDL ``near2m_calshow``: after every calibration trial draw the clean | injected
        panel (window, once the launch movie has released it) and save it as
        ``steps/calib_annNN_trialNNNN.png`` (IDL ``calib_annNN_iter_NNNN.png``)."""
        from .intro import draw_calshow
        self._eta_mark(runner, "cal")
        inj = clean = None
        src, ps, cps = [], None, None
        if isinstance(images, dict) and images.get("inj") is not None:
            ev_i, ev_c = images.get("inj"), images.get("clean")
            inj = getattr(ev_i, "image", ev_i)
            clean = None if ev_c is None else getattr(ev_c, "image", ev_c)
            src = [s.as_tuple() if hasattr(s, "as_tuple") else tuple(s) for s in images.get("sources") or []]
            ps, cps = images.get("per_source"), images.get("clean_per_source")
        self._records = []
        self._ia = ia
        ad = self._guard(runner, "calibration data", self._data, runner)
        if ad is None:
            return
        png = os.path.join(self.run_dir, "steps", f"calib_ann{ia + 1:02d}_trial{trial:04d}.png")
        os.makedirs(os.path.dirname(png), exist_ok=True)
        fig = self._figure2()
        draw_calshow(fig, ad, clean, inj, src, trial, contrast, msnr, per_source=ps, clean_per_source=cps, cmap=self.cmap)
        fig.savefig(png, dpi=self.dpi, facecolor=fig.get_facecolor())
        self._calib_paths = getattr(self, "_calib_paths", []) + [png]
        if self.show and not self._intro_on:
            self._window_show(fig=fig)

    def on_calibration(self, runner, info: Dict[str, Any], images=None) -> None:
        """``images`` (from the runner) = the last calibration measurement at the default
        config: ``{'inj': EvalImages, 'clean': EvalImages | None, 'sources': [...],
        'per_source': [...], 'clean_per_source': [...]}`` -> render the panel with them
        now.  Without images (failed measurement / older runner) fall back to the seed
        evaluation's images (same config, calibrated contrast), else render bare."""
        self._calib_info = dict(info or {})
        runner._display_calib = self._calib_info
        self._eta_mark(runner, "cal")
        self._records = []
        self._ia = runner.ia
        self._calib_pending = False
        inj = clean = None
        src = ps = cps = None
        if isinstance(images, dict) and images.get("inj") is not None:
            ev_i, ev_c = images.get("inj"), images.get("clean")
            inj = getattr(ev_i, "image", ev_i)
            clean = None if ev_c is None else getattr(ev_c, "image", ev_c)
            src = [s.as_tuple() if hasattr(s, "as_tuple") else tuple(s) for s in images.get("sources") or []]
            ps, cps = images.get("per_source"), images.get("clean_per_source")
        elif images is not None and getattr(images, "image", None) is not None:     # a bare EvalImages
            inj = images.image
        if inj is None or np.ndim(inj) != 2:
            self._calib_pending = bool(runner.cfg.seed_default)
            if self._calib_pending:
                return
        ad = self._guard(runner, "calibration data", self._data, runner)
        if ad is not None:
            self._guard(runner, "calibration panel", render_calibration, ad, self._calib_info,
                        os.path.join(self._ann_dir(runner.ia), "calibration_panel.png"), clean=clean, inj=inj,
                        sources=src, per_source=ps, clean_per_source=cps, cmap=self.cmap,
                        target=tuple(runner.cfg.calibration.target))

    def on_draw(self, runner, draw: int, n_draws: int, record, inj, clean) -> None:
        """Keep the live image current between a trial's remeasurements.

        Deliberately light: it refreshes the images the next panel will use and nothing
        else.  It does NOT touch ``_eval_times`` (the source of s/eval and both ETAs), the
        ``_records`` list, or the panel cadence counter -- a draw is part of a trial, not a
        trial, and counting it as one would divide the ETA by ``n_remeasure`` and report the
        gap between draws as the cost of an evaluation.  The trial's own panel arrives
        through :meth:`on_eval` once the draws are averaged, showing the last draw's image
        beside the mean.
        """
        try:
            img = None if inj is None else inj.image
            if img is not None and np.ndim(img) == 2 and img.shape[0] >= 4:
                self._last_images = {"inj": img, "clean": None if clean is None else clean.image,
                                     "index": record.index, "draw": int(draw) + 1, "n_draws": int(n_draws)}
        except Exception as exc:
            self._say(runner, f"on_draw failed: {exc!r}")

    def on_eval(self, runner, record, inj, clean, is_best: bool) -> None:
        try:
            self._on_eval(runner, record, inj, clean, is_best)
        except Exception as exc:
            self._say(runner, f"on_eval failed: {exc!r}\n" + traceback.format_exc(limit=3))

    def _on_eval(self, runner, record, inj, clean, is_best: bool) -> None:
        # Time every evaluation, before any gate decides whether this one gets a panel.
        # These timestamps are the run's clock: the panel's "s/eval" and both ETAs are
        # derived from them, so sampling them only on the evaluations that happen to be
        # drawn reports the interval between *panels* instead -- a run evaluating every
        # 8 s read 80 s/eval once one panel in ten was being drawn, and the ETA with it.
        self._eval_times = getattr(self, "_eval_times", [])
        self._eval_times.append(time.time())
        if len(self._eval_times) > 200:                 # only the last 50 are ever read
            del self._eval_times[:-200]
        if self._intro_on or record.index == 0:
            self._intro_stop()
        if runner.ia != self._ia or record.index == 0:
            self._records = []
            self._ann_frames = []
            self._ia = runner.ia
            self._best_curve = None
            self._live_reason = None
            self._last_images = {}
        self._eta_mark(runner, "search")
        if record.index > len(self._records):
            # resumed mid-annulus: the evaluations before the restart are in results.jsonl
            try:
                from .plots import load_run
                prior = {int(r.get("index", -1)): r for r in _annulus_records(load_run(self.run_dir), runner.ia)}
                if all(k in prior for k in range(record.index)):
                    self._records = [prior[k] for k in range(record.index)]
                    self._say(runner, f"display: restored {record.index} earlier evaluations of annulus "
                                      f"{runner.ia + 1} from results.jsonl")
            except Exception as exc:
                self._say(runner, f"display: could not restore earlier evaluations ({exc!r})")
        rec = _record_dict(record)
        if record.index < len(self._records):        # re-emission (should not happen) -> replace
            self._records[record.index] = rec
        else:
            self._records.append(rec)
        ad = self._data(runner)
        i = record.index
        cur_inj = None if inj is None else inj.image
        cur_clean = None if clean is None else clean.image
        if cur_inj is not None and (np.ndim(cur_inj) != 2 or cur_inj.shape[0] < 4):
            cur_inj = None
        if self._calib_pending and record.phase == "seed" and self._calib_info is not None:
            self._calib_pending = False
            self._guard(runner, "calibration panel", render_calibration, ad, self._calib_info,
                        os.path.join(self._ann_dir(runner.ia), "calibration_panel.png"), clean=cur_clean, inj=cur_inj,
                        sources=ad.sources[i], per_source=ad.per_source[i], clean_per_source=ad.clean_per_source[i],
                        cmap=self.cmap, target=tuple(runner.cfg.calibration.target))
        info = self._guard(runner, "contrast preview", _live_curve_info, cur_clean, ad, i)
        live, reason = info if info is not None else (None, "contrast preview raised (see log)")
        self._live_reason = None if live else f"eval {i + 1}: {reason}"
        if is_best and live:
            self._best_curve = live
        if cur_inj is not None or cur_clean is not None:
            self._last_images = {"inj": cur_inj, "clean": cur_clean, "index": i}
        last = (i + 1) >= ad.n_iter
        if not (is_best or last or (i % self.every == 0)):
            return
        # The panel for the annulus' final evaluation is part of the result and always gets
        # drawn; any other one is skipped when the render thread is still busy, so the
        # window's lag stays the cost of one panel instead of growing with the run.
        if not (last or self._render_room()):
            self._skip_panel()
            return
        bimg = self._best_images(runner)
        brec = bimg.get("record")
        bi = brec.index if brec is not None and getattr(brec, "annulus", runner.ia) == runner.ia else ad.best()[0]
        fm = bimg.get("fm") or {}
        fm_img = fm.get("fm_image")
        if fm_img is None and inj is not None and getattr(inj, "fm_image", None) is not None:
            fm_img = inj.fm_image
        fm_src = [s.as_tuple() if hasattr(s, "as_tuple") else tuple(s) for s in (fm.get("sources") or [])] or None
        b_src = ad.sources[int(bi)] if 0 <= int(bi) < ad.n else []
        key = (runner.ia, int(bi), tuple(tuple(np.round(t, 6)) for t in b_src))
        if is_best or self._inj_model is None or getattr(self, "_inj_model_key", None) != key:
            b_img = bimg.get("inj")
            self._inj_model = injected_model_image(runner, b_src, None if b_img is None else np.shape(b_img))
            self._inj_model_key = key
        imgs = StepImages(fm_unavailable=getattr(self, "_fm_note", None), cur_clean=cur_clean, cur_inj=cur_inj, best_inj=bimg.get("inj"), best_clean=bimg.get("clean"),
                          best_index=int(bi), fm_image=fm_img, fm_sources=fm_src,
                          fm_label=f"KLIP-FM preview (best, eval {int(bi) + 1})", inj_model=self._inj_model,
                          **self._running_stitch(runner))
        elapsed = time.time() - runner.wall0 + runner.wall_prev
        # the clock was stamped at the top of this call, for every evaluation, panel or not
        et = self._eval_times[-50:]
        per = (et[-1] - et[0]) / (len(et) - 1) if len(et) >= 2 else np.nan       # full loop time per eval
        last = et[-1] - et[-2] if len(et) >= 2 else np.nan                         # this evaluation's loop time
        eta, eta_full = self._eta_estimate(runner, ad, per)             # IDL rem_ann / rem
        png = os.path.join(self.run_dir, "steps", f"step{self._step:04d}.png") if self.save_png else None
        pdf = None
        if self.pdf_every and ((i + 1) % self.pdf_every == 0 or last):
            pdf = os.path.join(self._ann_dir(runner.ia), f"eval_{i + 1:04d}_panel.pdf")
        curves = self._curves(runner, ad, live)
        label = "NEW BEST" if is_best else ""
        note = self._live_reason
        if png:
            self._paths.append(png); self._ann_frames.append(png)
            self._step += 1

        # step the landscape to the next partition for this panel.  Counting *panels* rather
        # than evaluations is what makes the cycle complete: with --display-every 10 and six
        # nights, keying on the evaluation index would visit only nights 1, 3 and 5 for ever.
        pid = self._next_corner_partition(ad)

        def _job(fig=None):
            render_step(ad, i, imgs, png, pdf, curves=curves, elapsed_s=elapsed, eta_s=eta, eta_full_s=eta_full,
                        cmap=self.cmap, dpi=self.dpi, fig=self._figure(), loop_s=per, last_s=last, step_label=label,
                        contrast_note=note, corner_partition=pid)
        # the step panel renders on a worker thread (Agg figure, ~2-4 s) so the evaluation loop
        # does not wait for it; the window shows each panel as soon as it is finished
        self._submit_render(png, _job)
        if self.movie and self.movie_every and png and ((i + 1) % self.movie_every == 0 or last):
            self._submit_progress_movie(runner.ia)
        self._show_latest()

    def _movie_pool(self):
        """Movies get their own thread.

        Encoding a GIF of every frame so far takes far longer than drawing one panel and
        grows with the annulus.  Sharing the render thread meant that every ``movie_every``
        evaluations the live panel queued behind a whole movie rebuild -- and, once panels
        were gated on that thread being free, that the movie's own turn came round less
        often than asked.  Separating them keeps the window's latency the cost of one panel.
        """
        import concurrent.futures as cf
        if getattr(self, "_mex", None) is None:
            self._mex = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="klip-movie")
            self._movies: List[Any] = []
        return self._mex

    def _submit_progress_movie(self, ia: int) -> None:
        """Rebuild ``annulusNN/progress.gif`` / ``.mp4`` from this annulus' frames so far
        (calibration frames first, like IDL); runs on the movie thread, so neither the
        evaluation loop nor the live panel ever waits for it."""
        frames = [p for p in getattr(self, "_calib_paths", []) if f"calib_ann{ia + 1:02d}_" in os.path.basename(p)] \
            + list(self._ann_frames)
        d = self._ann_dir(ia)

        def _job():
            from .animate import make_movie
            os.makedirs(d, exist_ok=True)
            make_movie(frames, os.path.join(d, "progress.gif"), os.path.join(d, "progress.mp4"), intro=None,
                       intro_frames=0, hold_last=4)
        pool = self._movie_pool()
        self._movies = [f for f in self._movies if not f.done()]
        if self._movies:
            return              # one still encoding: this rebuild would only repeat it
        self._movies.append(pool.submit(_job))

    def on_validation_trial(self, runner, ia: int, ci: int, n_cand: int, e: int, t: int, n_valid: int,
                            inj, clean, sources, r, trials) -> None:
        """Validation trials go through the step panel like IDL's ``valid`` frames: the
        trial's images in the Test cells, candidate / trial / fresh sources in the title."""
        ad = self._guard(runner, "validation data", self._data, runner)
        if ad is None or ad.n == 0:
            return
        self._eta_mark(runner, "val")
        et = getattr(self, "_eval_times", [])[-50:]
        per = (et[-1] - et[0]) / (len(et) - 1) if len(et) >= 2 else np.nan
        trials_left = (n_cand - ci - 1) * n_valid + (n_valid - t - 1)
        eta, eta_full = self._eta_estimate(runner, ad, per, in_validation=True, trials_left=trials_left)
        ad.winner = None
        img_i = None if inj is None else getattr(inj, "image", inj)
        img_c = None if clean is None else getattr(clean, "image", clean)
        src = [s.as_tuple() if hasattr(s, "as_tuple") else tuple(s) for s in sources]
        ps = [None if not np.isfinite(v) else float(v) for v in getattr(r, "per_source", [])]
        rps = getattr(r, "raw_per_source", None)
        rps = None if rps is None else [None if not np.isfinite(v) else float(v) for v in rps]
        cps = getattr(r, "clean_per_source", None)
        cps = None if cps is None else [None if not np.isfinite(v) else float(v) for v in cps]
        done = [v for v in trials if v is not None and np.isfinite(v)]
        med = float(np.median(done)) if done else np.nan
        bimg = self._best_images(runner)
        # ``e`` is the candidate being validated; the best cells hold whatever image the
        # runner is keeping, which is a different evaluation for every candidate after the
        # first.  Circle that image's own injections rather than the candidate's.
        b_src = [s.as_tuple() if hasattr(s, "as_tuple") else tuple(s)
                 for s in (bimg.get("sources") or [])] or None
        imgs = StepImages(fm_unavailable=getattr(self, "_fm_note", None), cur_clean=img_c, cur_inj=img_i, best_inj=bimg.get("inj"), best_clean=bimg.get("clean"),
                          best_index=int(e), best_sources=b_src,
                          best_labels=(None if b_src is None else
                                       (list(ad.per_source[bimg["record"].index])
                                        if bimg.get("record") is not None
                                        and 0 <= bimg["record"].index < ad.n else [None] * len(b_src))),
                          fm_image=(bimg.get("fm") or {}).get("fm_image"), inj_model=self._inj_model,
                          cur_label=f"Ann {ia + 1}/{ad.nann}  Valid {ci + 1}/{n_cand} trial {t + 1}/{n_valid} (eval {e + 1})",
                          cur_override={"sources": src, "per_source": ps, "raw_per_source": rps, "clean_per_source": cps,
                                        "score": float(getattr(r, "score", np.nan)), "raw": np.nan},
                          **self._running_stitch(runner))
        png = os.path.join(self.run_dir, "steps", f"step{self._step:04d}.png") if self.save_png else None
        if png:
            self._paths.append(png); self._ann_frames.append(png); self._step += 1
        curves = self._curves(runner, ad, None)
        label = f"VALIDATION  cand {ci + 1}/{n_cand}  trial {t + 1}/{n_valid}  median so far {med:.2f}" if done else \
            f"VALIDATION  cand {ci + 1}/{n_cand}  trial {t + 1}/{n_valid}"
        elapsed = time.time() - runner.wall0 + runner.wall_prev

        def _job():
            render_step(ad, int(e), imgs, png, None, curves=curves, elapsed_s=elapsed, eta_s=eta, eta_full_s=eta_full,
                        cmap=self.cmap, dpi=self.dpi, fig=self._figure(), loop_s=per, step_label=label)
        self._submit_render(png, _job)
        self._show_latest()

    def on_validation(self, runner, table: List[Dict[str, Any]], winner: Dict[str, Any]) -> None:
        ad = self._guard(runner, "validation data", self._data, runner)
        if ad is None:
            return
        inj = winner.get("inj")
        clean = winner.get("clean")
        src = [s.as_tuple() if hasattr(s, "as_tuple") else tuple(s) for s in winner.get("sources", [])]
        self._val_args = dict(inj=None if inj is None else inj.image, clean=None if clean is None else clean.image,
                              sources=src, winner_eval=winner.get("eval_index"),
                              per_source=winner.get("per_source") or None)
        self._guard(runner, "validation panel", render_validation, ad, table,
                    os.path.join(self._ann_dir(runner.ia), "validation_panel.png"), cmap=self.cmap, **self._val_args)

    def on_annulus_done(self, runner, result, winner: Dict[str, Any]) -> None:
        self._eta_mark(runner, "done")
        ad = self._guard(runner, "annulus data", self._data, runner)
        if ad is None:
            return
        ad.winner = result.to_dict()
        ad.validation = result.validation_table
        d = self._ann_dir(result.annulus)
        self._guard(runner, "corner", render_corner, ad, os.path.join(d, "corner.pdf"))
        self._guard(runner, "landscapes", render_landscapes, ad, os.path.join(d, "landscapes.pdf"))
        self._guard(runner, "parhist", render_parhist_book, ad, os.path.join(d, "parhist.pdf"))
        self._guard(runner, "edf", render_edf, ad, os.path.join(d, "edf.png"))
        self._guard(runner, "partition map", render_nightmap, ad, os.path.join(d, "partition_map.png"))
        self._guard(runner, "kbook", render_kbook, ad, os.path.join(d, "kbook.pdf"))
        self._guard(runner, "importance", render_importance, ad, os.path.join(d, "importance.pdf"))
        self._guard(runner, "paracoord", render_paracoord, ad, os.path.join(d, "paracoord.pdf"))
        self._guard(runner, "rank", render_rank, ad, os.path.join(d, "rank.pdf"))
        self._guard(runner, "slice", render_slice, ad, os.path.join(d, "slice.pdf"))
        self._guard(runner, "verify books", render_verify_books, self.run_dir, result.annulus)
        self._guard(runner, "products sheet", render_products_sheet, self.run_dir, result.annulus,
                    os.path.join(d, "products.pdf"))
        # the KLIP-FM cross-check (A §9.1) came after validation: redraw the validation
        # panel with the FM response image next to the winner thumbnails
        fm = winner.get("fm") or {}
        if fm.get("fm_image") is not None and getattr(self, "_val_args", None) and result.validation_table:
            self._guard(runner, "validation panel (FM)", render_validation, ad, result.validation_table,
                        os.path.join(d, "validation_panel.png"), cmap=self.cmap, fm=fm["fm_image"], **self._val_args)
        # final frame of the annulus with the validated winner marked and both curves
        if self.save_png and ad.n:
            bimg = self._best_images(runner)
            # the current-eval cells keep the LAST evaluated images (from the last on_eval)
            li = self._last_images if self._last_images.get("index") is not None else {}
            i = int(li["index"]) if li else ad.n - 1
            if not li:
                c_inj, c_clean = load_eval_images(self.run_dir, result.annulus, i)
                li = {"inj": c_inj, "clean": c_clean}
            # Whatever picture ends up in the best cells, the circles have to be ITS
            # injections.  Three different images can land here and only the middle one is
            # the evaluation the history holds at winner_index:
            #   * bimg["inj"]        -- kept by the runner, paired with bimg["sources"];
            #   * the eval images    -- winner_index's own injected pass;
            #   * best_inj.fits      -- the winner's COMMITTED trial, whose injections are the
            #                           fresh validation draw, not the search one.
            b_src = None
            if bimg.get("inj") is not None:
                b_src = [s.as_tuple() if hasattr(s, "as_tuple") else tuple(s)
                         for s in (bimg.get("sources") or [])] or None
            else:
                b_inj, b_clean = load_eval_images(self.run_dir, result.annulus, int(result.winner_index))
                if b_inj is None:                  # the committed trial, with the fresh sources
                    b_src = [tuple(s) for s in (getattr(result, "winner_sources", None) or [])] or None
                bimg = dict(bimg, inj=b_inj if b_inj is not None else _fits(os.path.join(d, "best_inj.fits")),
                            clean=b_clean if b_clean is not None else _fits(os.path.join(d, "best_clean.fits")))
            b_lab = None
            if b_src is not None:
                ps = list(bimg.get("per_source") or getattr(result, "per_source", None) or [])
                b_lab = ps if len(ps) == len(b_src) else [None] * len(b_src)
            fm_src = [s.as_tuple() if hasattr(s, "as_tuple") else tuple(s) for s in (fm.get("sources") or [])] or None
            imgs = StepImages(fm_unavailable=getattr(self, "_fm_note", None), cur_clean=li.get("clean"), cur_inj=li.get("inj"), best_inj=bimg.get("inj"),
                              best_clean=bimg.get("clean"), best_index=int(result.winner_index),
                              best_sources=b_src, best_labels=b_lab,
                              note="annulus complete", fm_image=fm.get("fm_image"), fm_sources=fm_src,
                              fm_label="KLIP-FM cross-check (winner, test spiral)",
                              cur_label=f"Ann {result.annulus + 1}/{ad.nann}  last eval {i + 1}",
                              inj_model=injected_model_image(runner, ad.sources[int(result.winner_index)]
                                                             if 0 <= int(result.winner_index) < ad.n else [],
                                                             None if bimg.get("inj") is None else np.shape(bimg.get("inj"))),
                              **self._running_stitch(runner))
            curves = self._curves(runner, ad, None, done=True)
            png = os.path.join(self.run_dir, "steps", f"step{self._step:04d}.png")
            self._wait_renders()
            r = self._guard(runner, "final frame", render_step, ad, i, imgs, png, None, curves=curves,
                            elapsed_s=time.time() - runner.wall0 + runner.wall_prev, eta_s=0.0, cmap=self.cmap,
                            dpi=self.dpi, fig=self._figure(),
                            step_label="annulus done -- " + ("VALIDATED winner" if result.validated else "winner NOT validated"),
                            contrast_note="no validated curve for this annulus" if not curves else None)
            if r is not None:
                self._paths.append(png); self._ann_frames.append(png)
                self._step += 1
                self._window_show(png=png)
                if self.movie:
                    self._submit_progress_movie(result.annulus)     # final per-annulus progress movie
                self._guard(runner, "final frame (white)", render_step, ad, i, imgs,
                            os.path.join(d, "step_display_white.png"), None, curves=curves,
                            elapsed_s=time.time() - runner.wall0 + runner.wall_prev, eta_s=0.0, cmap=self.cmap,
                            dpi=self.dpi, fig=self._figure(), white=True,
                            step_label="annulus done -- " + ("VALIDATED winner" if result.validated else "winner NOT validated"),
                            contrast_note="no validated curve for this annulus" if not curves else None)

    def on_finish(self, runner) -> None:
        self._wait_renders()
        if not self.movie or not self._paths:
            return
        try:
            from .animate import make_movie
            intro = os.path.join(self.run_dir, "steps", "intro.png")
            frames = list(getattr(self, "_calib_paths", [])) + list(self._paths)   # IDL: calib frames lead the movie
            out = make_movie(frames, os.path.join(self.run_dir, "opt_steps.gif"),
                             os.path.join(self.run_dir, "opt_steps.mp4"), intro=intro if os.path.exists(intro) else None)
            self._say(runner, f"movie: {out}")
        except Exception as exc:
            self._say(runner, f"movie failed: {exc!r}")
        if getattr(self, "_rex", None) is not None:
            self._rex.shutdown(wait=True)
            self._rex = None
        self._fig = None
        try:
            from .parallel import IDLE_HOOKS
            for fn in (self._show_latest, self._intro_tick):
                if fn in IDLE_HOOKS:
                    IDLE_HOOKS.remove(fn)
        except Exception:
            pass
