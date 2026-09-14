"""The IDL launch "movie" (``near2m_intro_frame``) and the calibration panel
(``near2m_calshow``), frame for frame.

The intro is a five-act ASCII boot sequence shown in the live window while the first
annulus calibrates (boot log -> proximity alert -> Earth mobilizes -> the approach ->
target acquired / dossier); ``N_RELEASE`` frames at 0.10 s.  Every second frame is
also written to ``intro_frames/introframe_NNNNN.png`` and assembled into
``intro.gif`` (IDL: delay 0.24 s).  The calibration panel shows the last clean /
injected calibration measurement with the per-source S/N and the trial line.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
import matplotlib.patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

N_BOOT, N_WARN, N_PREP, N_SCENE = 52, 34, 46, 40
B5 = N_BOOT + N_WARN + N_PREP + N_SCENE
N_RELEASE = B5 + 128                       # 300 frames -> ~30 s at 0.10 s/frame
FRAME_DT = 0.10

_C = {"bg": "black", "gr": "#00e000", "cy": "#00e5ff", "yl": "#ffe000", "rd": "#ff2020", "tm": "#ff6347",
      "am": "#ffa000", "wh": "white", "ch2": "#606060"}
_MONO = dict(family="monospace")


def _t(ax, x, y, s, size=11, color="wh", ha="center", weight="normal", va="center"):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=_C.get(color, color),
            fontweight=weight, **_MONO)


def _brackets(ax):
    for (x0, x1, y0, y1) in ((0.015, 0.055, 0.960, 0.985), (0.985, 0.945, 0.960, 0.985),
                             (0.015, 0.055, 0.040, 0.015), (0.985, 0.945, 0.040, 0.015)):
        ax.plot([x0, x0, x1], [y0, y1, y1], color=_C["rd"], lw=2, transform=ax.transAxes)


def _fill(ax, y0, y1, color):
    ax.add_patch(matplotlib.patches.Rectangle((0, y0), 1, y1 - y0, transform=ax.transAxes, color=_C[color], zorder=0))


def draw_intro_frame(fig: Figure, f: int) -> None:
    """Draw frame ``f`` of the launch movie into ``fig`` (cleared first)."""
    fig.clf()
    fig.patch.set_facecolor(_C["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    gr, cy, yl, rd, tm, am, wh, ch2 = "gr", "cy", "yl", "rd", "tm", "am", "wh", "ch2"
    if f < N_BOOT or f >= N_BOOT + N_WARN:
        _brackets(ax)
    # ================================ ACT 1 : BOOT ==============================
    if f < N_BOOT:
        sy = 0.90 - 0.82 * ((f % 26) / 25.0)
        ax.plot([0.06, 0.94], [sy, sy], color=_C["ch2"], lw=1, transform=ax.transAxes)
        _t(ax, 0.5, 0.945, 'ESO VLT-UT4 "YEPUN"  //  NEAR THREAT-ANALYSIS GRID  v2.6', 17, rd, weight="bold")
        _t(ax, 0.5, 0.905, 'VISIR CYBER-CORONAGRAPH TACTICAL CORE  ::  CERRO PARANAL', 12, am)
        log = ['[OK]  VISIR mid-IR core (10-12.5um) ...... ONLINE',
               '[OK]  deformable secondary mirror ........ SYNCED',
               '[OK]  adaptive optics loop ......... STREHL 97%',
               '[OK]  AGPM coronagraph mask .............. SEATED',
               '[OK]  chop @ 8.33 Hz : ELFN null ......... NULLED',
               '[OK]  Si:As Aquarius array ........ 42K NOMINAL',
               '[OK]  KLIP tactical core ................. ARMED',
               '[SCAN] sweeping HZ : alpha-Cen A / B .......',
               '>> MOTION DETECTED : 3 UNKNOWN CONTACTS',
               '>> CLASS: NON-TERRESTRIAL   ORIGIN alpha-CEN',
               '!! THREAT LEVEL ................. CRITICAL']
        logcol = [gr, gr, gr, gr, gr, gr, gr, am, yl, cy, rd]
        nlog = len(log); nshow = min(f // 4 + 1, nlog)
        y0, dy = 0.845, 0.045
        for i in range(nshow):
            cc = ((rd if f % 4 < 2 else yl) if i == nlog - 1 else logcol[i])
            _t(ax, 0.10, y0 - i * dy, log[i], 12.5, cc, ha="left", weight="bold" if i >= 6 else "normal")
        if f % 6 < 3 and nshow < nlog:
            _t(ax, 0.10, y0 - nshow * dy, '> _', 12.5, gr, ha="left")
        prog = min(f / float(N_BOOT), 1.0); nb = max(int(round(prog * 30)), 0)
        _t(ax, 0.5, 0.11, 'SCAN [' + '#' * nb + '-' * (30 - nb) + f'] {int(round(prog * 100))}%', 14, gr)
    # ================================ ACT 2 : WARN ==============================
    elif f < N_BOOT + N_WARN:
        fw = f - N_BOOT; on = (fw % 6) < 3
        band = rd if on else am
        _fill(ax, 0.86, 1.0, band); _fill(ax, 0.0, 0.14, band)
        _t(ax, 0.5, 0.915, '///////  PROXIMITY ALERT  ///////', 19, "bg", weight="bold")
        _t(ax, 0.5, 0.055, '///////  PROXIMITY ALERT  ///////', 19, "bg", weight="bold")
        _t(ax, 0.5, 0.585, '!!  ALIENS  APPROACHING  !!', 34, yl if on else rd, weight="bold")
        _t(ax, 0.5, 0.46, 'VECTOR alpha-CEN A/B  HABITABLE-ZONE BREACH   1.34 pc', 14, wh)
        _t(ax, 0.5, 0.39, 'WARNING ' + '> ' * ((fw % 8) + 1) + 'INCOMING', 14, rd if on else wh)
        for s in range(3):
            _t(ax, 0.30 + s * 0.20, 0.30 - 0.015 * ((fw + s * 3) % 5), '<oOo>', 17, cy if on else wh)
    # =========================== ACT 3 : EARTH PREPARES ========================
    elif f < N_BOOT + N_WARN + N_PREP:
        fp = f - (N_BOOT + N_WARN); onp = (fp % 6) < 3
        _t(ax, 0.5, 0.945, 'EARTH DEFENSE INITIATIVE  //  GLOBAL MOBILIZATION', 18, cy if onp else wh, weight="bold")
        _t(ax, 0.5, 0.905, 'ALL STATIONS -- CONDITION ZULU -- THIS IS NOT A DRILL', 12, am)
        mob = ['SCRAMBLE ORBITAL INTERCEPTORS ..... GO', 'RAILGUN BATTERIES ........... CHARGED',
               'VLT + ELT ARRAYS -> TARGETING .. LOCK', 'DEEP SPACE NETWORK UPLINK ...... LIVE',
               'PLANETARY SHIELD GRID ........ RAISED', 'DEFENSE FLEET ........... LAUNCHING']
        mobcol = [gr, gr, cy, cy, yl, rd]
        nsh = min(fp // 5 + 1, len(mob))
        for i in range(nsh):
            _t(ax, 0.06, 0.83 - i * 0.045, mob[i], 12, mobcol[i], ha="left", weight="bold" if i >= 4 else "normal")
        cd = max(5 - fp // 8, 0)
        _t(ax, 0.80, 0.31, 'LAUNCH', 16, am)
        _t(ax, 0.80, 0.24, f'T- {cd}', 30, rd if fp % 4 < 2 else yl, weight="bold")
        for li, line in enumerate([' .-~~~-. ', '( EARTH  )', "  '-...-' "]):
            _t(ax, 0.5, 0.14 - li * 0.030, line, 19, cy)
        for r in range(6):
            rxx = 0.30 + r * 0.06
            ryy = 0.20 + ((0.020 * fp + r * 0.16) % 1.0) * 0.32
            _t(ax, rxx, ryy, '^', 16, wh); _t(ax, rxx, ryy - 0.026, ':', 13, am)
    # =============================== ACT 4 : SCENE =============================
    elif f < B5:
        fc = f - (N_BOOT + N_WARN + N_PREP)
        rs = np.random.RandomState(777); stx = rs.uniform(size=70); sty = 0.18 + 0.78 * rs.uniform(size=70)
        for i in range(70):
            tw = (fc + i * 7) % 13; gl = tw < 2
            _t(ax, stx[i], sty[i], '*' if gl else ('+' if tw < 5 else '.'), 14 if gl else 9, wh if tw < 4 else ch2)
        sy = 0.80 - 0.54 * ((fc % 40) / 39.0)
        ax.plot([0.06, 0.94], [sy, sy], color=_C["ch2"], lw=1, transform=ax.transAxes)
        _t(ax, 0.5, 0.945, '>>>   INBOUND: VISITORS FROM ALPHA CENTAURI   <<<', 27, gr, weight="bold")
        if fc % 8 < 4:
            _t(ax, 0.5, 0.905, ':: carrier lock -- decoding transmission ::', 14, tm, weight="bold")
        mx = ((fc * 0.006) % 1.30) - 0.15
        _t(ax, mx, 0.855, '<==={####|####}===>', 17, cy, ha="left")
        _t(ax, 0.5, 0.155, '\\  |  /' if fc % 4 < 2 else ' \\ | / ', 20, yl)
        _t(ax, 0.5, 0.132, '(( * ))' if fc % 6 < 3 else '(  *  )', 28, yl)
        _t(ax, 0.5, 0.086, 'ALPHA CEN', 18, yl)
        pose = ['(o-o)', '/|#|\\', ' d b '] if fc % 2 == 0 else ['(o-o)', '/|#|\\', ' b d ']
        ylo, yhi = 0.26, 0.80; bh = yhi - ylo
        for c in range(7):
            axp = 0.125 + c * 0.126; ph = c * 0.19
            for kk in range(2):
                yy = ylo + ((0.011 * fc + ph + kk * 0.5) % 1.0) * bh
                col = cy if yy > yhi - 0.10 else gr
                for li, line in enumerate(pose):
                    _t(ax, axp, yy - li * 0.024, line, 16, col, ha="left")
        ufo = ['  .-==-.  ', ' ( o o ) ', "  '~~~'  "]
        ux = 0.5 + 0.20 * math.sin(fc * 0.21); uy = 0.55 + 0.20 * math.sin(fc * 0.11)
        if fc % 4 < 2:
            for li in range(1, 5):
                _t(ax, ux, uy - 0.05 - li * 0.030, ':' * li, 13, yl)
        for li, line in enumerate(ufo):
            _t(ax, ux, uy - li * 0.026, line, 19, cy)
        rc = rd if fc % 8 < 4 else yl
        _t(ax, ux - 0.075, uy + 0.010, '[', 26, rc); _t(ax, ux + 0.075, uy + 0.010, ']', 26, rc)
        _t(ax, ux, uy + 0.070, 'TRACKING', 9, rc)
        _t(ax, 0.5, 0.035, 'NEAR/VISIR decoding transmission' + '.' * ((fc % 5) + 1) + '  (calibrating injection contrast)', 14, gr)
    # ===================== ACT 5 : TARGET ACQUIRED / DISCOVERY =================
    else:
        f5 = f - B5
        rs = np.random.RandomState(888); stx = rs.uniform(size=60); sty = rs.uniform(size=60)
        for i in range(60):
            _t(ax, stx[i], sty[i], '+' if ((f5 + i * 5) % 11) < 2 else '.', 9, ch2)
        _t(ax, 0.5, 0.955, 'VISIR / NEAR  ::  DIRECT-IMAGING TARGET ACQUISITION', 16, gr, weight="bold")
        if f5 < 26:
            _t(ax, 0.42, 0.55, '(*)', 30, yl); _t(ax, 0.60, 0.47, '*', 20, am)
            _t(ax, 0.42, 0.63, 'A', 12, yl); _t(ax, 0.61, 0.41, 'B', 11, am)
            fr = 1.0 - f5 / 26.0; rc = rd if f5 % 4 < 2 else yl; gwid = 0.06 + 0.32 * fr
            _t(ax, 0.42 - gwid, 0.55, '[', 32, rc); _t(ax, 0.42 + gwid, 0.55, ']', 32, rc)
            _t(ax, 0.42, 0.63 + 0.28 * fr, 'v', 15, rc)
            _t(ax, 0.5, 0.28, 'ACQUIRING TARGET :: alpha CEN A + B', 15, cy)
            _t(ax, 0.5, 0.10, '1.34 pc  //  N-band 10-12.5 um  //  AGPM coronagraph', 11, gr)
        elif f5 < 50:
            gz = (f5 - 26) / 24.0
            _t(ax, 0.5, 0.54, '(( * ))', 30 + 30 * gz, yl)
            _t(ax, 0.5, 0.86, 'RESOLVING  alpha CEN A   (G2V, 1.1 Lsun)', 15, cy)
            _t(ax, 0.74, 0.42, 'o', 14, cy); _t(ax, 0.74, 0.37, 'HZ companion', 10, gr)
            _t(ax, 0.5, 0.10, 'thermal-IR point source :: SNR climbing', 11, gr)
        elif f5 < 76:
            gz = (f5 - 50) / 26.0; cs = 15 + 11 * gz
            planet = [' .-=========-. ', '/=============\\', '|==== O ======|', '|=============|', '\\=============/', " '-=========-' "]
            for li, line in enumerate(planet):
                _t(ax, 0.5, 0.66 - li * 0.055, line, cs, am)
            _t(ax, 0.5, 0.90, 'GIANT PLANET DETECTED', 17, yl if f5 % 4 < 2 else rd, weight="bold")
            _t(ax, 0.5, 0.06, 'forward-model + injection-recovery :: CONFIRMED', 11, gr)
        else:
            pcx, pcy = 0.29, 0.52
            planet = [' .-====-. ', '/========\\', '|== O ===|', '\\========/', " '-====-' "]
            for li, line in enumerate(planet):
                _t(ax, pcx, pcy + 0.10 - li * 0.050, line, 17, am)
            mnr = [0.11, 0.155, 0.155, 0.20]; msp = [0.13, -0.09, 0.10, -0.06]; mph = [0., 1.6, 3.1, 4.7]; mcl = [wh, gr, tm, cy]
            for m in range(4):
                ang = f5 * msp[m] + mph[m]
                _t(ax, pcx + mnr[m] * math.cos(ang), pcy + 0.72 * mnr[m] * math.sin(ang), '(o)' if m == 1 else 'o',
                   15 if m == 1 else 12, mcl[m])
            doss = ['== TARGET DOSSIER ==', 'PLANET : gas giant, 1.9 Mjup', 'ORBIT  : 1.1 AU  (inner HZ)',
                    'T_eff  : 295 K,  ringed', 'MOONS  : 4 confirmed', '  II : ocean + N2/O2/CH4',
                    '  << BIOSIGNATURE >>', 'LIFE   : CONFIRMED']
            dcl = [cy, wh, wh, wh, wh, gr, gr, gr]
            nd = len(doss); ns = min((f5 - 76) // 4 + 1, nd)
            for i in range(ns):
                cc = ((gr if f5 % 4 < 2 else yl) if i >= nd - 2 else dcl[i])
                _t(ax, 0.55, 0.80 - i * 0.075, doss[i], 12.5, cc, ha="left", weight="bold" if i >= nd - 2 else "normal")
            if ns >= nd and f5 % 6 < 3:
                _t(ax, 0.5, 0.05, 'TRANSMISSION COMPLETE _', 12, gr)


def render_intro_frames(run_dir: str, every: int = 2, dpi: int = 80, size=(1850 / 80.0, 990 / 80.0),
                        movie: bool = True, log=None) -> List[str]:
    """Write every ``every``-th frame of the launch movie to ``run_dir/intro_frames/`` and
    (``movie``) assemble ``run_dir/intro.gif`` (IDL: 0.24 s per frame)."""
    d = os.path.join(run_dir, "intro_frames")
    os.makedirs(d, exist_ok=True)
    fig = Figure(figsize=size, dpi=dpi)
    FigureCanvasAgg(fig)
    paths = []
    for f in range(0, N_RELEASE, max(every, 1)):
        p = os.path.join(d, f"introframe_{f:05d}.png")
        if not os.path.exists(p):
            draw_intro_frame(fig, f)
            fig.savefig(p, dpi=dpi, facecolor=_C["bg"])
        paths.append(p)
    if movie and paths:
        try:
            from .animate import make_movie
            make_movie(paths, os.path.join(run_dir, "intro.gif"), None, fps=1.0 / (FRAME_DT * every * 1.2))
        except Exception as exc:
            if log:
                log(f"  [display] intro movie not written: {exc!r}")
    return paths


# ----------------------------------------------------------------------------
# calibration panel (near2m_calshow)
# ----------------------------------------------------------------------------
def draw_calshow(fig: Figure, ad, clean: Optional[np.ndarray], inj: Optional[np.ndarray],
                 sources: Sequence[Tuple[float, float, float]], trial: int, contrast: float, msnr: float,
                 per_source=None, clean_per_source=None, cmap: str = "inferno", white: bool = False) -> None:
    """``near2m_calshow``: clean (no inj) | injected, same stretch, sources circled with
    their S/N, annulus / trial / contrast lines and the aggregate S/N vs the 4-6 target."""
    from .display import _rc, draw_image
    fg = "black" if white else "white"
    with _rc(idl=True, dark=not white):
        fig.clf()
        fig.patch.set_facecolor("white" if white else "black")
        if inj is None or np.ndim(inj) != 2:
            fig.text(0.5, 0.5, "Calibrating injection contrast...", ha="center", va="center", fontsize=20, color=fg)
            return
        fig.text(0.5, 0.95, "CALIBRATING INJECTION CONTRAST", ha="center", va="center", fontsize=18, color=fg)
        src = [tuple(s) for s in sources]
        have_clean = clean is not None and np.ndim(clean) == 2
        lab_i = [None if v is None else (f"{v:.1f}" if np.isfinite(v) else None) for v in (per_source or [])] or None
        lab_c = [None if v is None else (f"{v:.1f}" if np.isfinite(v) else None) for v in (clean_per_source or [])] or None
        if have_clean:
            ax = fig.add_axes([0.08, 0.30, 0.37, 0.54])
            draw_image(ax, clean, ad, "clean (no inj)", cmap, sources=src, labels=lab_c)
            ax.set_title("clean (no inj)", fontsize=13)
            ax = fig.add_axes([0.55, 0.30, 0.37, 0.54])
        else:
            ax = fig.add_axes([0.32, 0.30, 0.37, 0.54])
        draw_image(ax, inj, ad, "injected", cmap, sources=src, labels=lab_i)
        ax.set_title("injected", fontsize=13)
        fig.text(0.5, 0.16, f"Annulus {ad.annulus + 1} / {ad.nann}", ha="center", fontsize=14, color=fg)
        fig.text(0.5, 0.10, f"trial {trial}    contrast = {contrast:9.2E}", ha="center", fontsize=14, color=fg)
        stat = "mean" if len(src) == 2 else "median"
        if per_source and clean_per_source and all(v is not None for v in per_source):
            pi = np.array([np.nan if v is None else v for v in per_source], float)
            pc = np.array([np.nan if v is None else v for v in clean_per_source], float)
            orig = float(np.nanmedian(pi)); corr = float(np.nanmedian(pi - np.maximum(np.nan_to_num(pc), 0)))
            fig.text(0.5, 0.04, f"{stat} S/N:   original = {orig:.2f}     corrected = {corr:.2f}      (target 4 - 6)",
                     ha="center", fontsize=14, color=fg)
        else:
            fig.text(0.5, 0.04, f"{stat} SNR = {msnr:.2f}      (target 4 - 6)", ha="center", fontsize=14, color=fg)
