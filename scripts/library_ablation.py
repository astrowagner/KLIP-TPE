#!/usr/bin/env python
"""What the searched reference library buys, measured on the run that chose it.

A ``run_miri.py`` search tunes the reference library per annulus alongside the usual KLIP
parameters, in the terms of the engine it ran on: on the built-in engine ``nkeep_altroll``
(correlation-ranked frames kept from the other roll) and ``nkeep_psfref`` (from the
reference star); on pyKLIP its own selection -- ``mode`` (which pools) and ``maxnumbasis``
(how many of the most-correlated frames of those pools each target keeps, per sector).
This script asks how much that is worth, by rebuilding the finished run's exact setup --
same loader, reducer, space, objective, sampler, annuli and per-annulus contrast -- and
scoring a set of configurations per annulus on COMMON injection draws: the same source
positions in every configuration.  On these data one draw scatters by ~0.84 in S/N from
the azimuth of the fakes alone, and most of that scatter is shared between
configurations, so a paired comparison resolves differences an unpaired one would bury.

Configurations, per annulus (the library variants differ from the winner ONLY in the
library, expressed in the engine's own terms):

  winner    the run's validated winner, tuned library and all
  all       the winner with both pools taken whole          -- ADI+RDI, "use everything"
  rdi       the winner with the reference star only, whole
  adi       the winner with the other roll only, whole
  rdi_third, ardi_half   (built-in engine) the best-correlated third of the reference
            star; the best half of each pool
  top_k     (pyKLIP) ADI+RDI keeping the k_klip best -- what pyKLIP does unasked
  carter    Carter et al. (2023)'s MIRI choices mapped into this space: every frame from
            both pools, no temporal binning, one azimuthal subsection, no high-pass,
            k = 6 -- reduced over the annulus' own zone
  carter_full  the same over ONE zone covering the whole crop from the 4QPM inner working
            angle out, as they reduced it ("a single annulus and a single subsection (i.e.,
            the entire image)")
  default   the run's own seeded default

Every injected reduction is scored twice: ``score_search`` (what the search maximised,
``s_inj - max(s_clean, 0)``) and ``score_raw`` (what validation reports).  HIP 65426 b's S/N
is measured in each configuration's clean image where its zone contains the planet.

The engine is the run's own unless ``--backend`` names the other, which makes it a
cross-engine test: each winner at its other parameters, with that engine's library in its
own terms.

A live window follows it (on by default; ``--no-show`` for none): per annulus, each
configuration's S/N draw by draw with its running mean, the planet's S/N under each name,
the other run's winner as a reference with ``--versus``, and the latest clean and injected
reductions.  The same picture is saved next to ``--out`` as a PNG.

usage:
  python scripts/library_ablation.py --data ~/Data/JWST/hip65426_miri \\
      --run-dir miri_HIP-65426_F1140C_v6 --n-draws 10 --out ablation_F1140C_v6.json

Two runs head to head (``--versus``) -- winner against winner, paired draw for draw -- need
the same injections in both: the same positions (same ``--seed`` and ``--n-draws``) AND the
same contrast.  Each run calibrated its own, so the second takes the first's with
``--contrast-from``:

  python scripts/library_ablation.py --data ~/Data/JWST/hip65426_miri \\
      --run-dir miri_HIP-65426_F1140C_v7_pyklip --n-draws 40 \\
      --out miri_HIP-65426_F1140C_v7_pyklip/library_ablation.json
  python scripts/library_ablation.py --data ~/Data/JWST/hip65426_miri \\
      --run-dir miri_HIP-65426_F1140C_v7_klip --n-draws 40 \\
      --contrast-from miri_HIP-65426_F1140C_v7_pyklip \\
      --out miri_HIP-65426_F1140C_v7_klip/library_ablation.json \\
      --versus miri_HIP-65426_F1140C_v7_pyklip/library_ablation.json

NIRCam F444W (paper_runs' D on pyKLIP, DK on the built-in engine) the same way; the
instrument is read from the run's pixel scale, and no --data is needed (run_demos loads its
own cubes).  The Carter et al. configurations are MIRI's and are left out:

  python scripts/library_ablation.py --run-dir paper_runs/D_hip65426_pyklip --n-draws 40 \\
      --out paper_runs/D_hip65426_pyklip/library_ablation.json
  python scripts/library_ablation.py --run-dir paper_runs/D_hip65426_klip --n-draws 40 \\
      --contrast-from paper_runs/D_hip65426_pyklip \\
      --out paper_runs/D_hip65426_klip/library_ablation.json \\
      --versus paper_runs/D_hip65426_pyklip/library_ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import run_miri                                                          # noqa: E402
from klip_tpe import RunConfig, Runner, ValidationConfig, CalibrationConfig   # noqa: E402
from klip_tpe.instruments import generic                                 # noqa: E402

#: Carter et al. (2023, ApJL 951, L20), MIRI: ADI+RDI with the full 9-point SGD library,
#: "a single annulus and a single subsection (i.e., the entire image)", contrast flat
#: beyond ~2 modes (ADI) and companion fits at 6.  Table 3 F1140C: 823 +/- 11 mas,
#: 149 +/- 1 deg, dF1140C = 8.264 +/- 0.021.
CARTER = dict(bin=1, n_ang=1, filter=0, k_klip=6)
PLANET = (0.823, 149.0)
IWA_AS = 0.36                                   # FQPM1140C nominal inner working angle


def _need_finished_run(run_dir: str, flag: str) -> None:
    """Stop, saying why, unless ``run_dir`` holds a FINISHED ``run_miri.py`` run: this script
    rebuilds a run from its ``run_setup.json`` (written once the search starts) and takes the
    winners from its ``final_results.json`` (written when it ends)."""
    log_file = os.path.join(run_dir, "run.log")
    where = f"; its progress is in {log_file}" if os.path.isfile(log_file) else ""
    if not os.path.isdir(run_dir):
        raise SystemExit(f"{flag} {run_dir}: no such directory -- is it the --out of a run_miri.py "
                         f"run, and are you in the directory that run was started from?")
    if not os.path.isfile(os.path.join(run_dir, "run_setup.json")):
        raise SystemExit(f"{flag} {run_dir}: the run has not started its search yet (no "
                         f"run_setup.json){where}.  This compares FINISHED runs -- run it once "
                         f"final_results.json appears.")
    if not os.path.isfile(os.path.join(run_dir, "final_results.json")):
        raise SystemExit(f"{flag} {run_dir}: the run has not finished (no final_results.json "
                         f"yet){where}.  Run this once it has.")


def _run_backend(full_setup: dict):
    """The engine a finished run reduced with, from its ``run_setup.json``: pyKLIP writes its
    name into the reducer's description and the built-in engine writes none, so a partition
    without one is ``'klip'``.  None when no reducer (or a mix) was recorded."""
    parts = (full_setup.get("reducer") or {}).get("partitions") or {}
    names = {str(p.get("backend") or "klip") for p in parts.values() if isinstance(p, dict)}
    return names.pop() if len(names) == 1 else None


def _run_dims(full_setup: dict) -> list:
    """The reduction dimensions the run searched, by name."""
    sp = full_setup.get("space") or {}
    ps = sp.get("params", []) if isinstance(sp, dict) else sp
    return [p["name"] for p in ps if isinstance(p, dict) and p.get("role", "reduction") == "reduction"]


def _space_check(run_dims, names, same_engine: bool):
    """``(problems, notes)`` from comparing the dimensions a run searched with the rebuilt
    space's.  On the run's own engine they must be the same: a winner is carried across BY
    NAME, so a dimension only the run had would be dropped silently, and one only the rebuild
    has would sit at its default -- the winner reduced would not be the winner.  On the other
    engine the library dimensions differ by design, and that is only noted."""
    names = list(names)
    lost = [n for n in run_dims if n not in names]
    extra = [n for n in names if n not in run_dims] if run_dims else []
    if not (lost or extra):
        return [], []
    what = "; ".join(s for s in (f"only the run has {lost}" if lost else "",
                                 f"only this rebuild has {extra}" if extra else "") if s)
    if same_engine:
        return [f"searched dimensions differ ({what}): the winners cannot be reproduced"], []
    return [], [f"  cross-engine: {what} -- each winner keeps its other parameters, and this "
                f"engine's library sits at its defaults"]


def _contrasts_from(run_dir: str, setup: dict) -> list:
    """Another run's calibrated contrast per annulus, to inject at here -- what pairing two
    runs needs, since each calibrates its own and S/N scales with the injected flux.  Its
    annuli must be this run's."""
    with open(os.path.join(run_dir, "run_setup.json")) as f:
        other = json.load(f)
    other = other.get("config", other)
    mine, theirs = np.asarray(setup["ann_edges"], float), np.asarray(other["ann_edges"], float)
    if mine.shape != theirs.shape or not np.allclose(mine, theirs):
        raise SystemExit(f"--contrast-from {run_dir}: its annuli {list(theirs)} are not this "
                         f"run's {list(mine)}")
    with open(os.path.join(run_dir, "final_results.json")) as f:
        c = [float(fr["contrast"]) for fr in json.load(f)["annuli"]]
    if len(c) != mine.size - 1:
        raise SystemExit(f"--contrast-from {run_dir}: {len(c)} annuli finished, {mine.size - 1} needed")
    return c


def _args_for(run_setup: dict, data: str, workers, backend: str = "pyklip") -> object:
    """``run_miri.build``'s argument object, mirroring the finished run (``backend`` may be
    the other engine's: a cross-engine test)."""
    a = dict(data=data, target="HIP-65426", filter="F1140C", partition="all", crop=40, backend=backend,
             ann=[float(v) for v in run_setup["ann_edges"]], known=[(0.826, 150.2)],
             star_flux=None, flux_density_jy=None, mode="ADI+RDI", min_throughput=0.30,
             dead_zones=True, nan_dead_zones=False, destripe=None,
             ref_target=["HIP-68245"], searched_library=True, star_center=None,
             workers=workers)
    return type("A", (), a)()


# ------------------------------------------------------------------ the live view
_SURFACE, _INK, _INK2, _MUTED, _GRID, _AXIS = "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
#: a configuration's ROLE carries the colour, not its name (nine names would outrun any
#: palette): dark-surface categorical slots 1-2 of the reference palette, validated all-pairs
#: with slot 3 (the other run's winner); baselines in neutral grey
_ROLE_COLOR = {"winner": "#3987e5", "library": "#d95926", "baseline": "#898781"}
_VERSUS_COLOR = "#199e70"
_ENGINE_NAME = {"pyklip": "pyKLIP", "klip": "built-in"}


def _role(name: str) -> str:
    if name == "winner":
        return "winner"
    if name in ("default", "carter", "carter_full"):
        return "baseline"
    return "library"


class AblationDisplay:
    """Live window for an ablation, and the same picture saved as a PNG.

    Top row, one panel per annulus: every configuration's injected-source S/N, draw by draw
    (faint dots) with its running mean +/- standard error, HIP 65426 b's S/N in the
    configuration's clean image (hollow star), and -- when the other run was injected at the
    same contrast (``--versus``) -- that run's winner as a dashed reference.  Bottom row: the
    latest clean and injected reductions (the search display's -1..5 sigma inferno stretch,
    arcsec about the star) with the fakes circled and the planet marked, and progress.

    The window is redrawn at most every ``every_s`` s and at the end of each configuration;
    the PNG is written at the end of a configuration, at most every ``png_every_s`` s (run
    folders live in Dropbox, which conflicts on fast rewrites).  Nothing here can stop the
    ablation: the first error turns the display off with one log line.
    """

    def __init__(self, png: str, title: str, show="window", versus=None, log=print,
                 every_s: float = 3.0, png_every_s: float = 30.0):
        self.png, self.title, self.log = png, title, log
        self.show = bool(show) and str(show).strip().lower() not in ("0", "off", "no", "none", "false")
        self.versus = versus or {}        # annulus number -> (label, contrast, mean, sem)
        self.every_s, self.png_every_s = float(every_s), float(png_every_s)
        self.plan_ann: List[tuple] = []    # (annulus number, title) of every annulus to come
        self.ann: Dict[int, dict] = {}
        self.cur = self.last = None
        self.clean_img = self.inj_img = None
        self.src, self.params, self.draw_i = [], {}, -1
        self.geom: Dict[str, Any] = {}
        self.summary: List[str] = []
        self.n_total = self.n_done = 0
        self.t0 = time.time()
        self._fig = None
        self._t_draw = self._t_png = 0.0
        self.ok = True
        self.no_image = "--"               # what an empty image panel says

    # -- what the ablation tells it -----------------------------------------------------
    def plan(self, annuli: List[tuple], n_total: int, geom: Dict[str, Any]) -> None:
        self.plan_ann, self.n_total, self.geom = list(annuli), int(n_total), dict(geom)
        self._refresh(force=True)

    def start_annulus(self, number: int, order: List[str], n_draws: int, zone_px, contrast: float) -> None:
        self.ann[number] = {"order": list(order), "n_draws": int(n_draws), "zone": tuple(zone_px),
                            "contrast": float(contrast),
                            "data": {k: {"raw": [], "planet": None} for k in order}}
        self.cur_ann = number
        self._refresh(force=True)

    def clean(self, name: str, img, planet_snr, params: Dict[str, Any]) -> None:
        a = self.ann[self.cur_ann]
        a["data"][name]["planet"] = planet_snr
        self.cur = self.last = name
        self.params, self.draw_i = dict(params), -1
        self.clean_img, self.inj_img, self.src = img, None, []
        self.n_done += 1
        self._refresh()

    def draw(self, name: str, d: int, score: float, img, sources) -> None:
        self.ann[self.cur_ann]["data"][name]["raw"].append(float(score))
        self.inj_img, self.src, self.draw_i = img, list(sources), d
        self.n_done += 1
        self._refresh()

    def config_done(self, name: str) -> None:
        self._refresh(force=True, png=True)

    def finish(self, lines=()) -> None:
        self.summary, self.cur = list(lines), None
        self._refresh(force=True, png=True, final=True)

    # -- drawing --------------------------------------------------------------------------
    def _refresh(self, force: bool = False, png: bool = False, final: bool = False) -> None:
        if not self.ok:
            return
        try:
            now = time.time()
            fig = self._figure()
            if force or now - self._t_draw >= self.every_s:
                self._render(fig)
                self._t_draw = now
                if self.show:
                    fig.canvas.draw_idle()
            if self.show:
                fig.canvas.flush_events()       # keeps the window alive between reductions
            if png and (final or now - self._t_png >= self.png_every_s):
                fig.savefig(self.png, dpi=100, facecolor=_SURFACE)
                self._t_png = now
        except Exception as exc:
            self.ok = False
            self.log(f"  display: turned off after an error ({exc!r}); the ablation carries on")

    def _gui(self) -> bool:
        """The live display's backend choice (MacOSX / Qt / Tk); PNG only when none works."""
        import matplotlib
        matplotlib.rcParams["toolbar"] = "None"
        cur = matplotlib.get_backend().lower()
        forced = os.environ.get("KLIP_TPE_BACKEND")
        if not forced and cur in ("macosx", "qtagg", "qt5agg", "tkagg", "gtk3agg", "gtk4agg", "wxagg"):
            return True
        prefer = (forced,) if forced else (("MacOSX", "QtAgg", "TkAgg") if sys.platform == "darwin"
                                           else ("QtAgg", "TkAgg", "GTK3Agg"))
        for cand in prefer:
            try:
                matplotlib.use(cand, force=True)
                import matplotlib.pyplot as plt
                plt.figure()
                plt.close()
                return True
            except Exception:
                continue
        self.log(f"  display: no interactive matplotlib backend (MacOSX/Qt/Tk) -- the panel goes to {self.png} only")
        return False

    def _figure(self):
        if self._fig is not None:
            return self._fig
        if self.show:
            self.show = self._gui()
        if self.show:
            import matplotlib.pyplot as plt
            plt.ion()
            self._fig = plt.figure(figsize=(13.5, 7.8), dpi=100)
            try:
                self._fig.canvas.manager.set_window_title(self.title)
            except Exception:
                pass
            plt.show(block=False)
        else:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            self._fig = Figure(figsize=(13.5, 7.8), dpi=100)
            FigureCanvasAgg(self._fig)
        return self._fig

    @staticmethod
    def _style(ax) -> None:
        ax.set_facecolor(_SURFACE)
        for s in ax.spines.values():
            s.set_color(_AXIS)
        ax.tick_params(colors=_MUTED, labelsize=8)
        ax.grid(axis="y", color=_GRID, lw=0.8)
        ax.set_axisbelow(True)

    def _render(self, fig) -> None:
        import matplotlib.lines as mlines
        fig.clf()
        fig.patch.set_facecolor(_SURFACE)
        k = max(len(self.plan_ann), 1)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.42, left=0.05, right=0.985,
                              top=0.90, bottom=0.07)
        top = gs[0].subgridspec(1, k, wspace=0.18)
        bot = gs[1].subgridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.28)
        fig.text(0.05, 0.965, self.title, color=_INK, fontsize=12, weight="bold", va="center")
        for j, (number, label) in enumerate(self.plan_ann or [(0, "")]):
            ax = fig.add_subplot(top[0, j])
            self._style(ax)
            ax.set_title(label, color=_INK, fontsize=9, loc="left")
            if j == 0:
                ax.set_ylabel("injected-source S/N", color=_INK2, fontsize=9)
            rec = self.ann.get(number)
            if rec is None:
                ax.text(0.5, 0.5, "to come", ha="center", va="center", color=_MUTED, fontsize=9,
                        transform=ax.transAxes)
                ax.set_xticks([])
                continue
            order = rec["order"]
            for x, name in enumerate(order):
                col = _ROLE_COLOR[_role(name)]
                v = np.asarray(rec["data"][name]["raw"], float)
                v = v[np.isfinite(v)]
                if v.size:
                    jit = (np.random.default_rng(x).random(v.size) - 0.5) * 0.36   # stable as draws arrive
                    ax.scatter(x + jit, v, s=10, color=col, alpha=0.5, linewidths=0, zorder=2)
                    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
                    ax.errorbar([x], [float(v.mean())], yerr=[se], fmt="o", ms=8, color=col, mec=_SURFACE,
                                mew=2, elinewidth=2, capsize=0, zorder=4)
            vs = self.versus.get(number)
            if vs and np.isclose(vs[1], rec["contrast"], rtol=1e-9, atol=0.0):
                lab, _, m, se = vs
                ax.axhline(m, color=_VERSUS_COLOR, lw=1.5, ls=(0, (4, 3)), zorder=1)
                ax.text(len(order) - 0.45, m, f"{lab} winner {m:.2f} ", color=_INK2, fontsize=8, va="bottom", ha="right")
            ax.set_xticks(range(len(order)))
            # HIP 65426 b's S/N in each configuration's clean image, under its name: a different
            # quantity on a different scale (14-17 against 4-8), so not on this axis
            labs = []
            for name in order:
                pl = rec["data"][name]["planet"]
                labs.append(name + (f"\nb {pl:.1f}" if pl is not None and np.isfinite(pl) else ""))
            ax.set_xticklabels(labs, rotation=0, ha="center", color=_INK2, fontsize=7.5)
            ax.set_xlim(-0.6, len(order) - 0.4)
        handles = [mlines.Line2D([], [], color=_ROLE_COLOR[r], marker="o", ls="none", ms=7, label=t)
                   for r, t in (("winner", "the run's winner"), ("library", "library variant"), ("baseline", "baseline"))]
        if self.versus:
            handles.append(mlines.Line2D([], [], color=_VERSUS_COLOR, ls=(0, (4, 3)), lw=1.5,
                                         label="other run's winner (same injections)"))
        leg = fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=8,
                         bbox_to_anchor=(0.985, 0.985))
        for t in leg.get_texts():
            t.set_color(_INK2)
        rec = self.ann.get(getattr(self, "cur_ann", None))
        n_draws = rec["n_draws"] if rec else 0
        self._image(fig.add_subplot(bot[0, 0]), self.clean_img, f"clean  ·  {self.cur or self.last or ''}", [])
        dtitle = f"injected  ·  draw {self.draw_i + 1} of {n_draws}" if self.draw_i >= 0 else "injected"
        self._image(fig.add_subplot(bot[0, 1]), self.inj_img, dtitle, self.src)
        axt = fig.add_subplot(bot[0, 2])
        axt.set_axis_off()
        el = time.time() - self.t0
        lines = []
        if rec is not None and self.cur is not None:
            order = rec["order"]
            i = order.index(self.cur) + 1 if self.cur in order else 0
            done_ann = [n for n, _ in self.plan_ann].index(self.cur_ann) + 1 if self.plan_ann else 1
            lines.append(f"annulus {done_ann} of {len(self.plan_ann)}  ·  {self.cur} ({i} of {len(order)})")
        if self.n_total:
            left = (self.n_total - self.n_done) * el / max(self.n_done, 1)
            lines.append(f"{self.n_done} of {self.n_total} reductions  ·  {el / 60:.0f} min so far"
                         + (f"  ·  ~{left / 60:.0f} min left" if self.n_done and self.n_done < self.n_total else ""))
        if self.params:
            lines.append("  ".join(f"{k}={v}" for k, v in self.params.items()))
        if self.summary:
            lines += [""] + self.summary
        axt.text(0.0, 1.0, "\n".join(lines), color=_INK2, fontsize=9, va="top", ha="left", family="monospace",
                 transform=axt.transAxes, wrap=True)

    def _image(self, ax, img, title: str, sources) -> None:
        from klip_tpe.display import _crop_extent, _robust_sigma
        ax.set_facecolor("#111111")
        ax.set_title(title, color=_INK, fontsize=9, loc="left")
        ax.tick_params(colors=_MUTED, labelsize=7)
        for s in ax.spines.values():
            s.set_color(_AXIS)
        g = self.geom
        if img is None or np.ndim(img) != 2 or not np.isfinite(img).any() or not g:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, self.no_image, ha="center", va="center", color=_MUTED, fontsize=8,
                    transform=ax.transAxes)
            return
        px, fw = float(g["px"]), float(g["fwhm"])
        zone = self.ann[self.cur_ann]["zone"] if getattr(self, "cur_ann", None) in self.ann else (0, min(img.shape) / 2)
        sub, ext = _crop_extent(np.asarray(img, float), px, float(zone[1]) + 1.5 * fw)
        s = _robust_sigma(sub)
        ax.imshow(sub, origin="lower", extent=ext, cmap="inferno", vmin=-s, vmax=5 * s, interpolation="nearest")
        import matplotlib.patches as mp
        for r in zone:
            ax.add_patch(mp.Circle((0, 0), float(r) * px, fill=False, color=_MUTED, lw=0.8, ls=(0, (3, 3))))
        off = 90.0 if g.get("angle_convention", "pa") == "pa" else 0.0

        def xy(rho, th):
            t = np.deg2rad(float(th) + off)
            return float(rho) * np.cos(t), float(rho) * np.sin(t)
        for sname in sources:
            ax.add_patch(mp.Circle(xy(sname.rho, sname.theta), 0.9 * fw * px, fill=False, color=_INK, lw=1.1))
        pl = g.get("planet")
        if pl is not None:
            x, y = xy(*pl)
            ax.add_patch(mp.Circle((x, y), 1.4 * fw * px, fill=False, color=_INK2, lw=1.0, ls=(0, (2, 2))))
            ax.text(x, y + 1.6 * fw * px, "b", color=_INK2, fontsize=8, ha="center", va="bottom")
        ax.set_xlabel("arcsec", color=_MUTED, fontsize=8)


#: pixel scales that identify the instrument of a finished run (its run_setup.json records it)
PXSCALE = {"miri": 0.1103, "nircam": 0.0630}


def _instrument(full_setup: dict, asked: str = "auto") -> str:
    """``asked`` unless it is ``'auto'``, else the instrument whose pixel scale the run
    recorded (MIRI 0.110"/px, NIRCam long-wave 0.063"/px)."""
    if asked != "auto":
        return asked
    px = full_setup.get("pxscale")
    for name, ref in PXSCALE.items():
        if px is not None and abs(float(px) - ref) < 0.01:
            return name
    raise SystemExit(f"cannot tell the instrument from the run's pixel scale ({px}); pass --instrument")


def _rebuild(inst: str, setup: dict, a, log) -> dict:
    """The finished run's reducer, space, objective, sampler and annuli, built by the code
    that built the run: ``run_miri.build`` for MIRI, ``run_demos``'s ``run_D`` recipe for
    NIRCam (``hip65426_objects``, ``k_klip_max=18``, ``n_min_ref=4``, the planet as a known
    source).  ``carter``: whether the Carter et al. (2023) mapping applies -- it encodes their
    MIRI reduction and the 4QPM's inner working angle, so not on NIRCam."""
    if inst == "miri":
        dsets, info, red, ann, obj, samp, m, px = run_miri.build(_args_for(setup, a.data, a.workers, a.backend), log)
        space = generic.make_space(red, k_klip_max=40, max_drop=0, search_angles=False)
        space.project = generic.make_guard(red, k_max=40)
        return dict(red=red, ann=ann, obj=obj, samp=samp, px=px, space=space, planet=PLANET, carter=True)
    if inst == "nircam":
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "paper_runs"))
        if str(a.workers) not in ("auto", "", "None"):
            os.environ["WORKERS"] = str(a.workers)       # run_demos.workers() reads it
        import run_demos as R                             # noqa: E402
        red = R.hip65426_objects(engine=a.backend)
        space = generic.make_space(red, k_klip_max=18, search_angles=False)
        space.project = generic.make_guard(red, k_max=18, n_min_ref=4)
        obj, samp = generic.default_config(red, known=[R.HIP])
        ann = [float(v) for v in setup["ann_edges"]]
        return dict(red=red, ann=ann, obj=obj, samp=samp, px=float(red.pxscale), space=space,
                    planet=tuple(R.HIP), carter=False)
    raise SystemExit(f"unknown instrument {inst!r}")


def _differs(a, b) -> bool:
    """Parameter values compared as the space holds them: a categorical (pyKLIP's ``mode``,
    ``"ADI+RDI"``) by equality, a number to 1e-9."""
    if isinstance(a, str) or isinstance(b, str):
        return str(a) != str(b)
    return abs(float(a) - float(b)) > 1e-9


def _x_from_params(space, params: dict, fallback) -> np.ndarray:
    """A vector for ``space`` from named parameters, falling back to ``fallback`` for any
    dimension the parameters do not name.  The run's winners were recorded in the space it
    searched, which on pyKLIP carried two ``nkeep_*`` dimensions that engine now refuses; by
    name, the live four map onto either space."""
    x = np.asarray(fallback, float).copy()
    for i, k in enumerate(space.names):
        if k in params and params[k] is not None:
            p = space.params[i]
            x[i] = float(p.encode(params[k])) if p.kind == "categorical" else float(params[k])
    return x


def _vec(runner, x, **over):
    """``x`` with named entries replaced, then sanitised and projected through the run's own
    guard by the runner itself (``runner.ia`` must already be the annulus) -- the path every
    searched vector took, so a modified configuration is feasible in the same sense."""
    space = runner.space
    x = np.asarray(x, float).copy()
    for k, v in over.items():
        if k in space.names:                  # a library count on an engine without one: ignored
            x[space.names.index(k)] = float(v)
    return runner._project(x, is_random=False)


def _versus_reference(th: dict) -> Dict[int, tuple]:
    """Per annulus of another run's ablation: (engine label, contrast, winner mean, sem)."""
    ref = {}
    for A in th.get("annuli", []):
        w = np.asarray(A.get("configs", {}).get("winner", {}).get("raw", []), float)
        w = w[np.isfinite(w)]
        if w.size:
            ref[int(A["annulus"])] = (_ENGINE_NAME.get(th.get("backend"), str(th.get("backend"))),
                                      float(A["contrast"]), float(w.mean()),
                                      float(w.std(ddof=1) / np.sqrt(w.size)) if w.size > 1 else 0.0)
    return ref


def _plot_finished(a, log) -> int:
    """``--plot``: the live view of a finished ablation, rebuilt from its output."""
    j = json.load(open(a.plot))
    th = json.load(open(a.versus)) if a.versus else None
    png = os.path.splitext(a.plot)[0] + ".png"
    disp = AblationDisplay(png, f"library ablation  ·  {os.path.basename(j.get('run_dir', a.plot))}  ·  "
                                f"{_ENGINE_NAME.get(j.get('backend'), j.get('backend'))} engine",
                           show=a.show, versus=_versus_reference(th) if th else {}, log=log)
    disp.no_image = "reductions are not stored\nin an ablation's output"
    px = j.get("pxscale")

    def title(A):
        z = A.get("zone_px") or (0, 0)
        where = (f"{z[0] * px:.2f}-{z[1] * px:.2f}\"" if px else
                 "injections at " + "-".join(f"{v:.2f}" for v in A.get("band_as", [])) + "\"")
        return f"annulus {A['annulus']}  ·  {where}  ·  contrast {A['contrast']:.2e}"
    disp.plan([(A["annulus"], title(A)) for A in j["annuli"]], 0, {})
    for A in j["annuli"]:
        disp.start_annulus(A["annulus"], list(A["configs"]), int(j.get("n_draws", 0)),
                           tuple(A.get("zone_px") or (0, 0)), float(A["contrast"]))
        for name, c in A["configs"].items():
            disp.clean(name, None, c.get("planet_snr"), {})
            for d, v in enumerate(c.get("raw", [])):
                disp.draw(name, d, float(v) if v is not None else float("nan"), None, [])
            disp.config_done(name)
    lines = [f"from {os.path.basename(a.plot)} ({j.get('n_draws')} draws)"]
    if th:
        lines.append(f"{_ENGINE_NAME.get(j.get('backend'), j.get('backend'))} winner / "
                     f"{_ENGINE_NAME.get(th.get('backend'), th.get('backend'))} winner, paired draws:")
        for row in versus(j, th, log):
            r, lo, hi, pb = row["raw"]
            lines.append(f"annulus {row['annulus']}  x{r:.2f} [{lo:.2f}-{hi:.2f}]  P(better) {pb:.3f}")
    disp.params = {}
    disp.finish(lines)
    log(f"wrote {png}")
    if disp.show and disp.ok:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()                                   # a finished picture: stay until closed
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=None, help="the MIRI calints directory (MIRI runs only)")
    ap.add_argument("--run-dir", default=None,
                    help="the finished run: a run_miri.py --out, or paper_runs' D_hip65426_pyklip / _klip")
    ap.add_argument("--plot", default=None, metavar="JSON",
                    help="draw the view of a FINISHED ablation from its output (the S/N panels; its "
                         "images are not stored) to JSON's .png, show it, and exit.  --versus adds "
                         "the other run's winner as the reference")
    ap.add_argument("--instrument", default="auto", choices=["auto", "miri", "nircam"],
                    help="how to rebuild the run: MIRI (run_miri.build) or NIRCam F444W "
                         "(paper_runs/run_demos.py's run_D).  auto: from the recorded pixel scale")
    ap.add_argument("--n-draws", type=int, default=10)
    ap.add_argument("--annuli", type=int, nargs="+", default=None, help="1-based; default all")
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--workers", default="auto")
    ap.add_argument("--backend", default=None, choices=["pyklip", "klip"],
                    help="engine to reduce with (default: the run's own).  The other one is a "
                         "cross-engine test: each winner at its other parameters, with that "
                         "engine's library in its own terms")
    ap.add_argument("--contrast-from", default=None, metavar="RUN_DIR",
                    help="inject at another run's calibrated contrast per annulus instead of "
                         "this run's -- what --versus needs, since each run calibrates its own")
    ap.add_argument("--show", nargs="?", const="window", default="window", metavar="MODE",
                    help="live window (on by default; '0' or --no-show for none).  The same "
                         "picture is written next to --out as a PNG either way")
    ap.add_argument("--no-show", dest="show", action="store_const", const=None,
                    help="no live window (the PNG is still written)")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", default="library_ablation.json")
    ap.add_argument("--versus", default=None, metavar="JSON",
                    help="another run's output of this script, made with the same --seed and "
                         "--n-draws: its winners are compared with these draw for draw (the "
                         "injection positions are checked to be identical)")
    a = ap.parse_args(argv)

    def log(s):
        print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)

    if a.plot:
        return _plot_finished(a, log)
    if not a.run_dir:
        ap.error("--run-dir is required (or --plot JSON)")

    # Everything this needs must exist before the first of hours of reductions, not after.
    _need_finished_run(a.run_dir, "--run-dir")
    if a.contrast_from:
        _need_finished_run(a.contrast_from, "--contrast-from")
    if a.versus and not os.path.isfile(a.versus):
        raise SystemExit(f"--versus {a.versus}: no such file yet.  It is the other run's ablation "
                         f"output -- make that one first (the two commands run one after the other).")
    with open(os.path.join(a.run_dir, "run_setup.json")) as f:
        full_setup = json.load(f)
    setup = full_setup.get("config", full_setup)
    with open(os.path.join(a.run_dir, "final_results.json")) as f:
        final = json.load(f)["annuli"]
    run_backend = _run_backend(full_setup)
    if a.backend is None:
        a.backend = run_backend or "pyklip"
    log(f"engine: {a.backend}" + ("" if a.backend == run_backend else
                                  f" -- the run reduced with {run_backend or 'an unrecorded engine'}"))
    contrasts = [float(fr["contrast"]) for fr in final]
    if a.contrast_from:
        theirs = _contrasts_from(a.contrast_from, setup)
        log(f"injecting at {a.contrast_from}'s contrasts {['%.3e' % c for c in theirs]} "
            f"(this run calibrated {['%.3e' % c for c in contrasts]})")
        contrasts = theirs

    inst = _instrument(full_setup, a.instrument)
    if inst == "miri" and not a.data:
        raise SystemExit("--data (the MIRI calints directory) is required for a MIRI run")
    rb = _rebuild(inst, setup, a, log)
    red, ann, obj, samp, px, space, planet = (rb[k] for k in ("red", "ann", "obj", "samp", "px", "space", "planet"))
    log(f"instrument: {inst};  space: {space.names}")

    # A comparison against a run is only as good as the rebuild of that run.  Refuse to
    # spend hours of reductions on a setup that differs from the one the winners came from.
    problems = []
    want_fpa = [tuple(map(float, v)) for v in full_setup.get("sampler", {}).get("forbidden_pa", [])]
    got_fpa = [tuple(map(float, v)) for v in (getattr(samp, "forbidden_pa", None) or [])]
    if want_fpa and not np.allclose(np.array(sorted(want_fpa)), np.array(sorted(got_fpa)), atol=0.05):
        problems.append(f"forbidden sectors {got_fpa} != run's {want_fpa}")
    want_mask = full_setup.get("objective", {}).get("metric", {}).get("pixel_mask_px")
    pm = getattr(obj.metric, "pixel_mask", None)
    got_mask = None if pm is None else int(np.count_nonzero(pm))
    if want_mask is not None and got_mask != int(want_mask):
        problems.append(f"dead-zone mask {got_mask} px != run's {want_mask}")
    dim_problems, dim_notes = _space_check(_run_dims(full_setup), space.names,
                                           same_engine=(a.backend == run_backend))
    problems += dim_problems
    for s in dim_notes:
        log(s)
    for fr in final:
        want = {k: v for k, v in fr["winner_config"]["params"].items() if k in space.names}
        xw_ = _x_from_params(space, fr["winner_config"]["params"], space.default_vector())
        got = {k: v for k, v in space.decode(xw_).params.items() if k in space.names}
        if any(_differs(got[k], want[k]) for k in want):
            problems.append(f"annulus {fr['annulus'] + 1}: winner decodes to {got}, run had {want}")
    if problems:
        for p in problems:
            log(f"MISMATCH: {p}")
        raise SystemExit("the rebuilt setup does not reproduce the run; not comparing against it")
    log("rebuild matches the run: forbidden sectors, dead-zone mask and every winner's parameters")

    cfg = RunConfig(ann_edges=[float(v) for v in ann], n_iter=1, n_init=1, seed=int(setup.get("seed", 21)),
                    validation=ValidationConfig(n_top=1, n_valid=1),
                    calibration=CalibrationConfig(forced=[1e-4]), n_remeasure=1,
                    inject_inset_fwhm=float(setup.get("inject_inset_fwhm", 1.0)),
                    pair_area_midpoint=bool(setup.get("pair_area_midpoint", True)),
                    # the run's own source count -- NIRCam D injected 2, where the rule would
                    # give more; a MIRI run that left it to the rule records None
                    n_sources=setup.get("n_sources"),
                    save_fits=False, save_eval_images=False, fm_curve=False, verify=False)
    runner = Runner(red, space, obj, samp, cfg, os.path.join(a.run_dir, "_ablation"), log=log)

    lo_sel = np.array(space.lo, float)
    hi_sel = np.array(space.hi, float)
    live = "nkeep_altroll" in space.names and "nkeep_psfref" in space.names
    native = "mode" in space.names and "maxnumbasis" in space.names
    if live:
        n_alt = int(hi_sel[space.names.index("nkeep_altroll")])
        n_ref = int(hi_sel[space.names.index("nkeep_psfref")])
    else:
        n_alt = n_ref = None
    pool = int(hi_sel[space.names.index("maxnumbasis")]) if native else None
    if not (live or native):
        log(f"  {a.backend}: no searched library in this space -- the library variants are skipped")
    x_def = space.default_vector()

    out = {"run_dir": os.path.abspath(a.run_dir), "n_draws": a.n_draws, "seed": a.seed, "backend": a.backend,
           "run_backend": run_backend, "contrast_from": a.contrast_from,
           "space": list(space.names), "pools": {"altroll": n_alt, "psfref": n_ref},
           "instrument": inst, "pxscale": float(px),
           "carter_mapping": CARTER if rb["carter"] else None, "annuli": []}
    annuli = [i - 1 for i in a.annuli] if a.annuli else list(range(len(final)))
    # the other run's winner, per annulus, for the view
    vref = _versus_reference(json.load(open(a.versus))) if a.versus else {}
    disp = AblationDisplay(os.path.splitext(a.out)[0] + ".png",
                           f"library ablation  ·  {os.path.basename(os.path.abspath(a.run_dir))}  ·  "
                           f"{_ENGINE_NAME.get(a.backend, a.backend)} engine",
                           show=a.show, versus=vref, log=log)
    disp.plan([(ia + 1, f"annulus {ia + 1}  ·  {float(final[ia]['inrad']) * px:.2f}-"
                        f"{float(final[ia]['outrad']) * px:.2f}\"  ·  contrast {contrasts[ia]:.2e}")
               for ia in annuli], 0,
              {"px": px, "fwhm": float(red.fwhm), "planet": planet,
               "angle_convention": getattr(red, "angle_convention", "pa")})
    for ia in annuli:
        fr = final[ia]
        runner.ia, runner.contrast = ia, contrasts[ia]
        xw = _x_from_params(space, fr["winner_config"]["params"], space.default_vector())
        zone = (float(fr["inrad"]), float(fr["outrad"]))
        # Library variants in each engine's own terms, everything else held at the winner.
        # carter* always takes "everything": all frames of both pools, however expressed.
        if live:                                  # built-in: one count per pool
            lib = {"all": dict(nkeep_altroll=n_alt, nkeep_psfref=n_ref),
                   "rdi": dict(nkeep_altroll=0, nkeep_psfref=n_ref),
                   "adi": dict(nkeep_altroll=n_alt, nkeep_psfref=0),
                   # the idea itself: keep only the best-correlated part of a pool
                   "rdi_third": dict(nkeep_altroll=0, nkeep_psfref=n_ref // 3),
                   "ardi_half": dict(nkeep_altroll=n_alt // 2, nkeep_psfref=n_ref // 2)}
            every = lib["all"]
        elif native:                              # pyKLIP: which pools, and how many of the best
            lib = {"all": dict(mode="ADI+RDI", maxnumbasis=pool),
                   "rdi": dict(mode="RDI", maxnumbasis=pool),
                   "adi": dict(mode="ADI", maxnumbasis=pool),
                   # what pyKLIP did unasked before maxnumbasis was searched: the k best
                   "top_k": dict(mode="ADI+RDI", maxnumbasis=0)}
            every = lib["all"]
        else:
            lib, every = {}, {}

        def enc(over):
            """Categorical values (``mode``) to their index; 0 maxnumbasis -> k_klip."""
            out = {}
            for k, v in over.items():
                if k == "mode":
                    pm = space.params[space.names.index("mode")]
                    v = float(list(pm.choices).index(v))
                out[k] = v
            return out

        def at(x, over):
            over = enc(over)
            if over.get("maxnumbasis") == 0:
                x = np.asarray(x, float).copy()
                over["maxnumbasis"] = float(x[space.names.index("k_klip")])
            return _vec(runner, x, **over)

        configs = {"winner": (_vec(runner, xw), None)}
        for k, over in lib.items():
            configs[k] = (at(xw, over), None)
        configs.update({
            **({"carter": (at(x_def, dict(every, **CARTER)), None),
                "carter_full": (at(x_def, dict(every, **CARTER)), (IWA_AS / px, float(ann[-1])))}
               if rb["carter"] else {}),
            "default": (_vec(runner, x_def), None),
        })
        if a.configs:
            configs = {k: v for k, v in configs.items() if k in a.configs}
        n = runner._nsrc(ia)
        rlo, rhi = runner._band(ia)
        log(f"=== annulus {ia + 1}: zone {zone[0]:.1f}-{zone[1]:.1f} px, {n} sources in "
            f"{rlo:.3f}-{rhi:.3f}\", contrast {runner.contrast:.3e} (run: validated "
            f"{fr['winner_score']:.3f} at {float(fr['contrast']):.3e})")
        draws = [runner.sampler.sample(n, rlo, rhi, np.random.default_rng([a.seed, ia, d]), runner.contrast)
                 for d in range(a.n_draws)]
        if not disp.n_total:
            disp.n_total = len(annuli) * len(configs) * (1 + a.n_draws)
        disp.start_annulus(ia + 1, list(configs), a.n_draws, zone, runner.contrast)
        rec = {"annulus": ia + 1, "zone_px": zone, "contrast": runner.contrast, "n_sources": n,
               "band_as": [rlo, rhi], "run_validated": fr["winner_score"],
               "draws": [[(s.rho, s.theta) for s in src] for src in draws], "configs": {}}
        for name, (x, zov) in configs.items():
            c = space.decode(x)
            p = {k: (v if isinstance(v, str) else int(v) if float(v).is_integer() else float(v))
                 for k, v in c.params.items() if k in space.names}
            t0 = time.time()
            clean = runner._reduce(c, None, tag=f"abl_a{ia + 1}_{name}_clean", zone=zov)
            t_clean = time.time() - t0
            planet_snr = None
            if zone[0] * px <= planet[0] <= zone[1] * px or zov is not None:
                try:
                    planet_snr = float(obj.metric.per_source(clean.image, None, [planet[0]], [planet[1]])[0])
                except Exception as exc:
                    log(f"   planet S/N failed: {exc!r}")
            disp.clean(name, clean.image, planet_snr, p)
            raw, srch, walls = [], [], []
            for d, src in enumerate(draws):
                t1 = time.time()
                inj = runner._reduce(c, src, tag=f"abl_a{ia + 1}_{name}_d{d}", zone=zov)
                fmk = runner._fm_for(inj, clean)
                r_raw = obj.score_raw(inj.image, src, clean.image, **fmk)
                r_srch = obj.score_search(inj.image, src, clean.image, **fmk)
                raw.append(float(r_raw.score))
                srch.append(float(r_srch.score))
                walls.append(time.time() - t1)
                disp.draw(name, d, float(r_raw.score), inj.image, src)
            rec["configs"][name] = {"params": p, "zone_override": zov, "raw": raw, "search": srch,
                                    "planet_snr": planet_snr, "wall_clean_s": t_clean,
                                    "wall_inj_s": float(np.mean(walls))}
            log(f"   {name:11s} {p}  raw {np.nanmean(raw):5.2f}  search {np.nanmean(srch):5.2f}"
                f"  planet {'--' if planet_snr is None else f'{planet_snr:5.2f}'}  "
                f"({np.mean(walls):.1f} s/reduction)")
            with open(a.out, "w") as f:                  # checkpoint after every configuration
                json.dump(out | {"annuli": out["annuli"] + [rec]}, f, indent=1)
            disp.config_done(name)
        out["annuli"].append(rec)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    log(f"wrote {a.out}")
    lines = []
    if a.versus:
        th = json.load(open(a.versus))
        lines.append(f"{_ENGINE_NAME.get(a.backend, a.backend)} winner / "
                     f"{_ENGINE_NAME.get(th.get('backend'), th.get('backend'))} winner, paired draws:")
        for row in versus(out, th, log):
            r, lo, hi, pb = row["raw"]
            lines.append(f"annulus {row['annulus']}  x{r:.2f} [{lo:.2f}-{hi:.2f}]  P(better) {pb:.3f}")
            if "planet" in row:
                lines.append(f"           planet b {row['planet'][0]:.1f} vs {row['planet'][1]:.1f}")
        if len(lines) == 1:
            lines.append("no annulus could be paired -- the log says why")
    disp.finish(lines)
    log(f"wrote {disp.png}")
    return 0


def versus(mine: dict, theirs: dict, log=print) -> List[Dict[str, Any]]:
    """Winner against winner, paired by draw: the head-to-head of two runs' answers.

    Pairing needs the same injections: the same positions -- with the same seed and draw
    count they are, whichever engine reduced them -- and the same contrast.  Each run
    calibrates its own contrast and S/N scales with the injected flux, so two runs' own
    contrasts would put the calibration into the ratio; one of the two must have been made
    with ``--contrast-from`` the other's run directory."""
    rng = np.random.default_rng(0)
    rows: List[Dict[str, Any]] = []
    log(f"head-to-head: {mine.get('backend')} ({mine['run_dir']}) vs "
        f"{theirs.get('backend')} ({theirs['run_dir']})")
    by_annulus = {B["annulus"]: B for B in theirs["annuli"]}
    for A in mine["annuli"]:
        B = by_annulus.get(A["annulus"])
        if B is None:
            log(f"  annulus {A['annulus']}: not in the other -- not pairing")
            continue
        da, db = np.asarray(A["draws"], float), np.asarray(B["draws"], float)
        if da.shape != db.shape or not np.allclose(da, db):
            log(f"  annulus {A['annulus']}: the injections differ -- not pairing")
            continue
        ca, cb = A.get("contrast"), B.get("contrast")
        if ca is None or cb is None or not np.isclose(float(ca), float(cb), rtol=1e-9, atol=0.0):
            log(f"  annulus {A['annulus']}: injected at contrast {ca} vs {cb} -- not pairing "
                f"(make one with --contrast-from the other's run directory)")
            continue
        row = {"annulus": A["annulus"]}
        for stat in ("raw", "search"):
            a = np.asarray(A["configs"]["winner"][stat], float)
            b = np.asarray(B["configs"]["winner"][stat], float)
            idx = rng.integers(0, a.size, (4000, a.size))
            r = a[idx].mean(1) / b[idx].mean(1)
            p_better = float(np.mean(a[idx].mean(1) > b[idx].mean(1)))
            row[stat] = (float(a.mean() / b.mean()), float(np.percentile(r, 16)), float(np.percentile(r, 84)), p_better)
            log(f"  annulus {A['annulus']} {stat:6s}: {a.mean():5.2f} vs {b.mean():5.2f}  "
                f"x{a.mean() / b.mean():.3f} [{np.percentile(r, 16):.3f}, {np.percentile(r, 84):.3f}]  "
                f"P(first better) {p_better:.3f}")
        pa, pb = A["configs"]["winner"]["planet_snr"], B["configs"]["winner"]["planet_snr"]
        if pa is not None and pb is not None:
            row["planet"] = (float(pa), float(pb))
            log(f"  annulus {A['annulus']} planet: {pa:.2f} vs {pb:.2f}")
        rows.append(row)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
