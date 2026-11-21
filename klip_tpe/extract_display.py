"""The live dashboard of ``klip-tpe extract``, after the IDL predecessor's ``mwcm_sub_dash``.

:class:`ExtractDisplay` takes the extraction's progress, as the ``progress`` callback of
:func:`klip_tpe.extract.extract_companion`, and draws panels of 1600 x 760 pixels on a black background
into ``steps/stepNNNN.png`` in the extraction's directory.  The run's panel process draws them
(:class:`klip_tpe.panels.PanelProcess`), so the fit never waits for one.  A panel asked for while the last
is still being drawn is skipped, but for the last of each stage and those of the partitions and the fake
sources.  ``klip-tpe view --run-dir <dir>`` shows the newest in a window that cannot stall the fit, and the
end of the extraction makes a progress movie of them, ``extract_steps.gif``.  The layout is the IDL
predecessor's.

* **The images**, two rows of five centered on the star, in inferno.  The upper row holds the original, the
  best null, the current trial and the injected models of the best and of the trial (the original less each
  null), on the original's scale.  The lower row holds the S/N maps of the first three (a matched filter
  with the small-sample correction, from -2 to 5) and the PSF models of the best and of the trial, with a
  peak of 1.  The star is a +, and the best an x on every panel.  A cyan circle marks the trial on its
  panels and the best on the others, a green circle the reference position and a dashed magenta one a fake
  source's true position.  The axes are offsets in arcsec, East to the left with the "pa" angle convention.
* **The zoom stamps**, 13 x 13 pixels centered on the best, one under each image of the upper row and on its
  scale.
* **The convergence trace**, the residual against the trial.  The random draws (warm-up and exploration)
  are gray open circles and the TPE's proposals filled circles, with the running best as a line and the best
  as a green circle.  A sideways histogram of the residuals shares its axis.
* **The parabolas**, the residual against each parameter for the best 40 % of the trials, with its
  parabola, minimum and 1-sigma.  The profile scans take their place once they run.
* **The corner plot** of the trials in ``(rho, PA, contrast)``, colored by ``-log`` residual.  Once the
  partitions or the fake sources are fitted, a 2 x 2 grid takes its place.  It holds the partitions'
  spectrum, their own positions less the combined fit against wavelength, a table of the fake sources'
  offsets, and as many pairs of the corner plot as there is room for.
* **The text**, with the stage and the trial count, the best and the trial's values, the time per trial,
  the elapsed time and the ETA, and the live minimum with its 1-sigma, from the trials and then from the
  profiles.

Each fit has its own panels, on its own reduction: the companion's search and profiles, each fake source's
fit and, with ``--group-positions``, each partition's own position fit.

:func:`render_saved` draws the panels afterwards from a finished extraction's saved tables, so an
extraction made without the display gets one without a new fit.

Notes
-----
It follows a user's note of 2026-11-21.  ``mwcm_sub_dash`` is in the IDL predecessor's
``subtract_mwc758c_tpe.pro``, read only.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ExtractDisplay", "PANEL_PX", "crop", "render_panel", "render_saved"]

#: the panel's size in pixels, at 100 dpi
PANEL_PX = (1600, 760)
#: half the side of the zoom stamps, in pixels
ZOOM_HALF = 6
#: the TPE's flags of its guided proposals.  The others, ``warmup`` and ``explore``, are random draws.
GUIDED = ("tpe", "local")
#: the longest wait for the panel process at the end, in seconds
PROCESS_WAIT = 600.0
#: the saved cut-outs of the companion's trials, for :func:`render_saved`
CROPS = "extract_trial_crops.npz"

_LABELS = ('rho (")', "PA (deg)", "contrast")
#: the image panels' left edges, width and rows, as the IDL predecessor placed them
_IX = (0.035, 0.14, 0.245, 0.40, 0.505)
_IW = 0.095
_PH = _IW * PANEL_PX[0] / PANEL_PX[1]
_ROWS = (0.77, 0.77 - 0.024 - _PH)
_ZW = 0.078
_ZH = _ZW * PANEL_PX[0] / PANEL_PX[1]
_ZTOP = _ROWS[1] - 0.048
_RC = {"figure.facecolor": "black", "savefig.facecolor": "black", "axes.facecolor": "black",
       "axes.edgecolor": "#8c8c8c", "axes.labelcolor": "white", "text.color": "white", "xtick.color": "white",
       "ytick.color": "white", "font.size": 7.0, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
       "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.titlepad": 3.0,
       "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.linewidth": 0.8}


def crop(img: np.ndarray, half: int) -> np.ndarray:
    """Return a cut-out of an image centered on the star, padded with NaN.

    Each side is ``2 half + 1`` pixels for an odd side of the image and ``2 half`` for an even one, so the star
    stays at the cut-out's center, ``((nx-1)/2, (ny-1)/2)``, as it is at the image's.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, with the star at ``((nx-1)/2, (ny-1)/2)``.
    half : int
        Half the cut-out's side, in pixels.

    Returns
    -------
    out : ndarray, shape (2 half + ny % 2, 2 half + nx % 2)
        The cut-out, float32.  An image of that size comes back as it is.
    """
    img = np.asarray(img, float)
    ny, nx = img.shape
    my, mx = 2 * int(half) + ny % 2, 2 * int(half) + nx % 2
    y0, x0 = (ny - my) // 2, (nx - mx) // 2
    out = np.full((my, mx), np.nan, np.float32)
    ys = slice(max(y0, 0), min(y0 + my, ny))
    xs = slice(max(x0, 0), min(x0 + mx, nx))
    if ys.start < ys.stop and xs.start < xs.stop:
        out[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = img[ys, xs]
    return out


def _dxdy(rho: float, pa: float, pxscale: float, conv: str) -> Tuple[float, float]:
    """Return a position's offset from the star, in pixels along the image's axes.

    Parameters
    ----------
    rho : float
        Separation, in arcsec.
    pa : float
        Angle, in degrees.
    pxscale : float
        Plate scale, in arcsec per pixel.
    conv : {"pa", "math"}
        How the reducer reads angles.

    Returns
    -------
    dx : float
        Offset along x, in pixels.
    dy : float
        Offset along y, in pixels.
    """
    from .metrics import source_xy
    x, y = source_xy([float(rho)], [float(pa)], float(pxscale), 0.0, 0.0, conv)
    return float(x[0]), float(y[0])


def _xy_sigma(rho: float, pa: float, s_rho: float, s_pa: float, pxscale: float, conv: str) -> Tuple[float, float]:
    """Return a position's sigmas along the image's axes, from its sigmas in separation and angle.

    Parameters
    ----------
    rho : float
        Separation, in arcsec.
    pa : float
        Angle, in degrees.
    s_rho : float
        Sigma in separation, in arcsec.
    s_pa : float
        Sigma in angle, in degrees.
    pxscale : float
        Plate scale, in arcsec per pixel.
    conv : {"pa", "math"}
        How the reducer reads angles.

    Returns
    -------
    s_dx : float
        Sigma along x, in pixels, NaN when a sigma is.
    s_dy : float
        Sigma along y, in pixels.
    """
    x0, y0 = _dxdy(rho, pa, pxscale, conv)
    xr, yr = _dxdy(rho + s_rho, pa, pxscale, conv)
    xp, yp = _dxdy(rho, pa + s_pa, pxscale, conv)
    return float(np.hypot(xr - x0, xp - x0)), float(np.hypot(yr - y0, yp - y0))


def _unwrap(pa: float, lo: float) -> float:
    """Return an angle on the turn that starts at a box's low edge.

    Parameters
    ----------
    pa : float
        The angle, in degrees.
    lo : float
        The box's low edge, in degrees.

    Returns
    -------
    pa : float
        The same angle in ``[lo, lo + 360)``.
    """
    return float(lo) + (float(pa) - float(lo)) % 360.0


def _hms(s: Optional[float]) -> str:
    """Return a duration as the IDL predecessor's ``mwcm_hms`` wrote it.

    Parameters
    ----------
    s : float or None
        The duration, in seconds.

    Returns
    -------
    text : str
        Such as ``3m 05s`` or ``1h 02m 07s``, ``-`` for None or NaN.
    """
    if s is None or not np.isfinite(s):
        return "-"
    s = int(max(s, 0))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"


def _colors() -> Dict[str, Any]:
    """Return the panel's colors, the IDL predecessor's on black.

    Returns
    -------
    colors : dict
        ``fg`` (text and lines), ``dim`` (random draws), ``best`` (green), ``circle`` (cyan), ``ref``,
        ``true``, ``tpe``, ``fit`` and ``sig`` (viridis, as ``mwcm_cmcolor`` takes it on black) and ``dx``
        and ``dy`` (the chromatic panel).
    """
    from matplotlib import colormaps
    vir = colormaps["viridis"]

    def dark(f: float):
        return vir(0.30 + 0.70 * min(max(float(f), 0.0), 1.0))
    return {"fg": "white", "dim": "#9a9a9a", "best": "#3fd05a", "circle": "cyan", "ref": "#3fd05a",
            "true": "#ff6bd6", "tpe": dark(0.15), "fit": dark(0.85), "sig": dark(0.42), "dx": "#5ec4ff",
            "dy": "#ffb347"}


def _nice(v: float) -> float:
    """Return the largest of 1, 2 and 5 times a power of 10 that is at most a value.

    Parameters
    ----------
    v : float
        The value, positive.

    Returns
    -------
    step : float
        The tick step, 1 for a value that is not positive.
    """
    if not v > 0:
        return 1.0
    e = math.floor(math.log10(v))
    for m in (5.0, 2.0, 1.0):
        if m * 10.0 ** e <= v:
            return m * 10.0 ** e
    return 10.0 ** e


def _psf_image(psf: Optional[Dict[str, Any]], shape: Tuple[int, int], x: float, y: float, pa: float,
               fwhm: float) -> np.ndarray:
    """Return the PSF model at a position of the cut-out, with a peak of 1.

    Parameters
    ----------
    psf : dict or None
        The model's stamp (:meth:`ExtractDisplay._psf`), None for a Gaussian of the FWHM.
    shape : tuple of 2 ints
        The cut-out's shape.
    x : float
        The position's column in the cut-out.
    y : float
        Its row.
    pa : float
        Its angle, in degrees.  The stamp is turned by its difference from the stamp's own.
    fwhm : float
        FWHM, in pixels.

    Returns
    -------
    model : ndarray, shape ``shape``
        The model.
    """
    ny, nx = int(shape[0]), int(shape[1])
    if psf is None:
        sig = float(fwhm) / 2.3548
        yy, xx = np.mgrid[0:ny, 0:nx]
        return np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / sig ** 2)
    from scipy import ndimage
    from .klip import rotate_ccw
    st = np.asarray(psf["stamp"], float)
    hs = st.shape[0] // 2
    turn = ((float(pa) - float(psf["pa"]) + 180.0) % 360.0) - 180.0
    if abs(turn) > 1e-6:
        st = rotate_ccw(st, turn, cval=0.0)
    ix, iy = int(round(x)), int(round(y))
    st = ndimage.shift(st, (y - iy - float(psf["fy"]), x - ix - float(psf["fx"])), order=1, mode="constant", cval=0.0)
    out = np.zeros((ny, nx))
    y0, x0 = iy - hs, ix - hs
    ys = slice(max(y0, 0), min(y0 + 2 * hs + 1, ny))
    xs = slice(max(x0, 0), min(x0 + 2 * hs + 1, nx))
    if ys.start < ys.stop and xs.start < xs.stop:
        out[ys, xs] = st[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    peak = float(out.max())
    return out / peak if peak > 0 else out


# ----------------------------------------------------------------------------- the panel's parts
def _images(fig, st: Dict[str, Any], C: Dict[str, Any]) -> None:
    """Draw the two rows of images, their colorbars and the zoom stamps.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The panel.
    st : dict
        The panel's state.
    C : dict
        The colors.
    """
    from matplotlib.patches import Circle
    from .products import snr_map
    g, run = st["geom"], st["run"]
    pxs, fwhm, conv = float(g["pxscale"]), float(g["fwhm_px"]), str(g["conv"])
    clean = np.asarray(run["clean"], float)
    ny, nx = clean.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0                # the star, as crop() keeps it
    sx = -1.0 if conv == "pa" else 1.0                     # East to the left, as on the sky
    ext = (-sx * nx / 2.0 * pxs, sx * nx / 2.0 * pxs, -ny / 2.0 * pxs, ny / 2.0 * pxs)

    def where(rho: float, pa: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        dx, dy = _dxdy(rho, pa, pxs, conv)
        return (cx + dx, cy + dy), (sx * dx * pxs, dy * pxs)

    best = np.asarray(run["best"] if run.get("best") is not None else clean, float)
    trial = np.asarray(run["trial"] if run.get("trial") is not None else best, float)
    fin = np.sort(clean[np.isfinite(clean)])
    if fin.size >= 3:
        vlo, vhi = float(fin[int(0.02 * fin.size)]), float(fin[min(int(0.995 * fin.size), fin.size - 1)])
    else:
        vlo, vhi = 0.0, 1.0
    if not vhi > vlo:
        vhi = vlo + 1.0
    bx, tx = run["best_x"], run.get("trial_x")
    (bpx, bpy), bd = where(bx[0], bx[1])
    (tpx, tpy), td = where(tx[0], tx[1]) if tx is not None else ((bpx, bpy), bd)
    excl = [(bpx, bpy)]
    true = run.get("true")
    tru = where(true[0], true[1]) if true is not None else None
    if tru is not None:
        excl.append(tru[0])
    snr = []
    for img in (clean, best, trial):
        try:
            snr.append(snr_map(img, fwhm, exclude_xy=excl))
        except Exception:                                  # noqa: BLE001 - a blank map, the panel goes on
            snr.append(np.full(img.shape, np.nan))
    psfs = [_psf_image(st.get("psf"), clean.shape, bpx, bpy, bx[1], fwhm),
            _psf_image(st.get("psf"), clean.shape, tpx, tpy, tx[1] if tx is not None else bx[1], fwhm)]
    imgs = [clean, best, trial, clean - best, clean - trial] + snr + psfs
    lims = [(vlo, vhi)] * 5 + [(-2.0, 5.0)] * 3 + [(0.0, 1.0)] * 2
    titles = ["original", "best null", "current trial", "injected, best", "injected, trial",
              "S/N", "S/N", "S/N", "PSF, best", "PSF, trial"]
    if run.get("trial_approx"):
        titles[2] = "current trial (approx.)"
    ref = st.get("reference")
    rd = where(ref[0], ref[1])[1] if ref is not None else None
    t = _nice(0.75 * min(nx, ny) / 2.0 * pxs)
    ims = []
    for k in range(10):
        row, col = divmod(k, 5)
        ax = fig.add_axes([_IX[col], _ROWS[row], _IW, _PH])
        lo_, hi_ = lims[k]
        im = ax.imshow(np.where(np.isfinite(imgs[k]), imgs[k], lo_), origin="lower", cmap="inferno", vmin=lo_,
                       vmax=hi_, extent=ext, interpolation="nearest")
        ims.append(im)
        ax.plot([0.0], [0.0], "+", color=C["fg"], ms=7, mew=1.2)
        ax.plot([bd[0]], [bd[1]], "x", color=C["fg"], ms=6, mew=1.3)
        ax.add_patch(Circle(td if col in (2, 4) else bd, 2.0 * pxs, fill=False, ec=C["circle"], lw=1.3))
        if rd is not None:
            ax.add_patch(Circle(rd, 3.0 * pxs, fill=False, ec=C["ref"], lw=1.1))
        if tru is not None:
            ax.add_patch(Circle(tru[1], 3.0 * pxs, fill=False, ec=C["true"], lw=1.1, ls="--"))
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([-t, 0.0, t])
        ax.set_yticks([-t, 0.0, t])
        if col:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Dec (")' if conv == "pa" else 'y (")', labelpad=1)
        if row == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('RA (")' if conv == "pa" else 'x (")', labelpad=1)
        ax.set_title(titles[k])
    for x0, row, im, name in ((0.343, 0, ims[0], "flux"), (0.343, 1, ims[5], "S/N"), (0.603, 0, ims[3], "flux"),
                              (0.603, 1, ims[8], "model")):
        _colorbar(fig, im, [x0, _ROWS[row], 0.012, _PH], name)
    # the zoom stamps, centered on the best and on the upper row's scale
    z = ZOOM_HALF
    zx, zy = int(round(bpx)), int(round(bpy))
    for col in range(5):
        cut = np.full((2 * z + 1, 2 * z + 1), vlo)
        ys = slice(max(zy - z, 0), min(zy + z + 1, ny))
        xs = slice(max(zx - z, 0), min(zx + z + 1, nx))
        if ys.start < ys.stop and xs.start < xs.stop:
            part = imgs[col][ys, xs]
            cut[ys.start - (zy - z):ys.stop - (zy - z), xs.start - (zx - z):xs.stop - (zx - z)] = np.where(
                np.isfinite(part), part, vlo)
        ax = fig.add_axes([_IX[col] + (_IW - _ZW) / 2.0, _ZTOP - _ZH, _ZW, _ZH])
        zext = (sx * (zx - z - 0.5 - cx) * pxs, sx * (zx + z + 0.5 - cx) * pxs, (zy - z - 0.5 - cy) * pxs,
                (zy + z + 0.5 - cy) * pxs)
        im = ax.imshow(cut, origin="lower", cmap="inferno", vmin=vlo, vmax=vhi, extent=zext, interpolation="nearest")
        ax.plot([bd[0]], [bd[1]], "x", color=C["fg"], ms=6, mew=1.3)
        ax.add_patch(Circle(bd, 2.0 * pxs, fill=False, ec=C["circle"], lw=1.3))
        ax.set_xlim(zext[0], zext[1])
        ax.set_ylim(zext[2], zext[3])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(_IX[0], _ZTOP - _ZH - 0.024, f"zoom on the best, {2 * z + 1} x {2 * z + 1} px", fontsize=7.5)
    _colorbar(fig, im, [_IX[4] + (_IW + _ZW) / 2.0 + 0.006, _ZTOP - _ZH, 0.012, _ZH], "flux")


def _colorbar(fig, im, rect: Sequence[float], name: str) -> None:
    """Draw a vertical colorbar with its two ends labeled and its name above.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The panel.
    im : matplotlib.image.AxesImage
        The image whose scale it shows.
    rect : sequence of 4 floats
        Where, ``(left, bottom, width, height)`` in the panel's fractions.
    name : str
        Its name.
    """
    cax = fig.add_axes(list(rect))
    cb = fig.colorbar(im, cax=cax)
    lo_, hi_ = im.get_clim()
    cb.set_ticks([lo_, hi_])
    cb.set_ticklabels([f"{lo_:.1g}", f"{hi_:.1g}"])
    cb.outline.set_edgecolor("#8c8c8c")
    fig.text(rect[0] + rect[2] / 2.0, rect[1] + rect[3] + 0.008, name, ha="center", fontsize=7)


def _trace(fig, run: Dict[str, Any], C: Dict[str, Any]) -> None:
    """Draw the convergence trace and the sideways histogram of the residuals.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The panel.
    run : dict
        The fit's state.
    C : dict
        The colors.
    """
    from matplotlib.lines import Line2D
    ax = fig.add_axes([0.06, 0.06, 0.24, 0.195])
    axh = fig.add_axes([0.30, 0.06, 0.06, 0.195])
    R = np.asarray(run["R"], float)
    n = R.size
    ax.set_title("convergence", loc="left")
    ax.set_xlabel("trial", labelpad=1)
    ax.set_ylabel("residual / original", labelpad=2)
    axh.set_xlabel("N", labelpad=1)
    axh.tick_params(labelleft=False)
    ok = np.isfinite(R)
    if not ok.any():
        ax.text(0.5, 0.5, "collecting...", ha="center", va="center", transform=ax.transAxes)
        return
    rnd = np.array([f not in GUIDED for f in list(run["flags"])[:n]], bool)
    ev = np.arange(1, n + 1)
    lo_, hi_ = float(R[ok].min()), float(R[ok].max())
    ylo, yhi = lo_ - 0.10 * abs(lo_), hi_ + 0.10 * abs(hi_)
    if not yhi > ylo:
        yhi = ylo + 1.0
    ax.plot(ev[rnd & ok], R[rnd & ok], "o", mfc="none", mec=C["dim"], ms=3.0, mew=0.7, ls="")
    ax.plot(ev[~rnd & ok], R[~rnd & ok], "o", color=C["tpe"], ms=3.0, mew=0.0, ls="")
    run_best = np.fmin.accumulate(np.where(ok, R, np.inf))
    ax.plot(ev, np.where(np.isfinite(run_best), run_best, np.nan), "-", color=C["fg"], lw=1.6)
    ib = int(np.nanargmin(np.where(ok, R, np.nan)))
    ax.plot([ev[ib]], [R[ib]], "o", mfc="none", mec=C["best"], ms=11, mew=1.6)
    ax.set_xlim(0.0, max(n, 2) + 0.5)
    ax.set_ylim(ylo, yhi)
    keys = [Line2D([], [], marker="o", ls="", mfc="none", mec=C["dim"], ms=4, label="warm-up / explore"),
            Line2D([], [], marker="o", ls="", color=C["tpe"], ms=4, label="TPE"),
            Line2D([], [], color=C["fg"], lw=1.6, label="running best"),
            Line2D([], [], marker="o", ls="", mfc="none", mec=C["best"], ms=6, label="best")]
    fig.legend(handles=keys, loc="lower right", bbox_to_anchor=(0.36, 0.252), ncol=4, frameon=False,
               labelcolor=C["fg"], handletextpad=0.3, columnspacing=0.9, borderaxespad=0.0)
    edges = np.linspace(ylo, yhi, 46)
    hg = np.histogram(R[rnd & ok], edges)[0]
    ht = np.histogram(R[~rnd & ok], edges)[0]
    mid, bw = 0.5 * (edges[1:] + edges[:-1]), edges[1] - edges[0]
    axh.barh(mid, hg, height=0.76 * bw, color=C["dim"], lw=0)
    axh.barh(mid, ht, left=hg, height=0.76 * bw, color=C["tpe"], lw=0)
    axh.set_ylim(ylo, yhi)
    axh.set_xlim(0.0, max(int((hg + ht).max()), 1) * 1.12)
    from matplotlib.ticker import MaxNLocator
    axh.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="lower"))


def _near_best(run: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return which trials are near the best, the IDL predecessor's 40 %.

    Parameters
    ----------
    run : dict
        The fit's state.

    Returns
    -------
    near : ndarray of bool or None
        The trials at or below the 40th percentile of the residuals, None with fewer than 12 trials or 10
        such.
    """
    R = np.asarray(run["R"], float)
    ok = np.isfinite(R)
    if R.size < 12 or ok.sum() < 10:
        return None
    rs = np.sort(R[ok])
    thr = rs[min(max(int(round(0.40 * R.size)) - 1, 9), rs.size - 1)]
    near = ok & (R <= thr)
    return near if near.sum() >= 10 else None


def _parabolas(fig, run: Dict[str, Any], C: Dict[str, Any]) -> Tuple[List[Tuple[float, float]], str]:
    """Draw the residual against each parameter, from the trials near the best or from the profile scans.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The panel.
    run : dict
        The fit's state.
    C : dict
        The colors.

    Returns
    -------
    live : list of tuple of 2 floats
        Each parameter's minimum and 1-sigma, NaN where the parabola failed.
    source : str
        ``"profiles"`` or ``"trials"``.
    """
    from .extract import parabola_sigma
    lo, hi = np.asarray(run["lo"], float), np.asarray(run["hi"], float)
    scale = np.where(hi - lo > 0, hi - lo, 1.0)
    bx = np.asarray(run["best_x"], float)
    prof = run.get("profiles") or None
    use_prof = bool(prof) and any(len(v) >= 4 for v in prof.get("values", []))
    X = np.asarray(run["X"], float).reshape(-1, 3)
    R = np.asarray(run["R"], float)
    near = None if use_prof else _near_best(run)
    hh = 0.245
    pph, gap = 0.215 * hh, 0.12 * hh
    live: List[Tuple[float, float]] = []
    for d in range(3):
        top = 0.29 - d * (pph + gap)
        ax = fig.add_axes([0.425, top - pph, 0.2, pph])
        ax.text(0.015, 0.92, _LABELS[d], transform=ax.transAxes, va="top", fontsize=6.5)
        if use_prof:
            vals = prof["values"][d] if d < len(prof["values"]) else []
            res = prof["residual"][d] if d < len(prof["residual"]) else []
            xv, yv = np.asarray(vals, float), np.asarray(res, float)
        elif near is not None:
            xv, yv = X[near, d], R[near]
        else:
            xv = yv = np.zeros(0)
        g = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[g], yv[g]
        if xv.size < 5 or float(np.ptp(xv)) <= 0:
            ax.text(0.5, 0.45, "collecting...", ha="center", va="center", transform=ax.transAxes, fontsize=6.5)
            ax.set_xticks([])
            ax.set_yticks([])
            live.append((float("nan"), float("nan")))
            continue
        f = parabola_sigma(xv, yv, bx[d], scale[d])
        xm = float(f["minimum"])
        sg = float(f["sigma"]) if f["ok"] else float("nan")
        x0, x1 = float(xv.min()), float(xv.max())
        if np.isfinite(sg):
            x0, x1 = min(x0, max(xm - 1.1 * sg, lo[d])), max(x1, min(xm + 1.1 * sg, hi[d]))
        pad = 0.06 * max(x1 - x0, 1e-30)
        y0, y1 = float(yv.min()), float(yv.max())
        yp = 0.08 * max(y1 - y0, 1e-6)
        ax.plot(xv, yv, "o", ms=2.2, mew=0.0, color=C["dim"], ls="")
        if f["coef"] is not None:
            xf = np.linspace(x0 - pad, x1 + pad, 60)
            u = (xf - bx[d]) / scale[d]
            c = f["coef"]
            ax.plot(xf, c[0] + c[1] * u + c[2] * u * u, "-", color=C["fit"], lw=1.8)
        ax.axvline(xm, ls="--", lw=1.0, color=C["best"])
        if np.isfinite(sg):
            for s in (-1.0, 1.0):
                ax.axvline(xm + s * sg, ls=":", lw=1.2, color=C["sig"])
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - yp, y1 + yp + 0.35 * (y1 - y0 + 2.0 * yp))      # room for the label
        ax.locator_params(axis="x", nbins=3)
        ax.locator_params(axis="y", nbins=2)
        if d == 2:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3), useMathText=True)
            ax.xaxis.get_offset_text().set_fontsize(6)
        live.append((xm, sg))
    fig.text(0.425, 0.297, "the profile scans" if use_prof else "the trials near the best", fontsize=7.5)
    return live, ("profiles" if use_prof else "trials")


def _scatter(ax, run: Dict[str, Any], a: int, b: int, C: Dict[str, Any], norm, cmap) -> None:
    """Draw one pair of the corner plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The cell.
    run : dict
        The fit's state.
    a : int
        The parameter along x.
    b : int
        The parameter along y.
    C : dict
        The colors.
    norm : matplotlib.colors.Normalize
        The scale of ``-log`` residual.
    cmap : matplotlib.colors.Colormap
        Its colors.
    """
    X = np.asarray(run["X"], float).reshape(-1, 3)
    R = np.asarray(run["R"], float)
    lo, hi = run["lo"], run["hi"]
    ok = np.isfinite(R)
    if ok.any():
        rnd = np.array([f not in GUIDED for f in list(run["flags"])[:R.size]], bool)
        col = cmap(norm(-np.log(np.clip(R, 1e-30, None))))
        s = 18.0 * min(max(math.sqrt(80.0 / max(int(ok.sum()), 1)), 0.30), 1.0) ** 2
        m = rnd & ok
        ax.scatter(X[m, a], X[m, b], s=s, facecolors="none", edgecolors=col[m], linewidths=0.6)
        m = ~rnd & ok
        ax.scatter(X[m, a], X[m, b], s=s, c=col[m], linewidths=0.0)
    bx, tx = run["best_x"], run.get("trial_x")
    if tx is not None and run.get("stage") in ("search", "profile"):
        ax.axvline(tx[a], ls="--", lw=0.8, color=C["dim"])
        ax.axhline(tx[b], ls="--", lw=0.8, color=C["dim"])
    ax.plot([bx[a]], [bx[b]], "s", mfc="none", mec=C["fg"], ms=6.5, mew=1.3)
    ax.set_xlim(lo[a], hi[a])
    ax.set_ylim(lo[b], hi[b])


def _fmt_axis(ax, which: str, d: int) -> None:
    """Give a corner plot's axis three ticks, with the contrast's exponent apart.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The cell.
    which : {"x", "y"}
        The axis.
    d : int
        Its parameter.
    """
    from matplotlib.ticker import MaxNLocator
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    if d == 2:
        ax.ticklabel_format(axis=which, style="sci", scilimits=(-2, 3), useMathText=True)
        axis.get_offset_text().set_fontsize(6)


def _score_norm(run: Dict[str, Any]):
    """Return the scale of ``-log`` residual, the IDL predecessor's: the lowest to the 97th percentile.

    Parameters
    ----------
    run : dict
        The fit's state.

    Returns
    -------
    norm : matplotlib.colors.Normalize
        The scale.
    """
    from matplotlib.colors import Normalize
    R = np.asarray(run["R"], float)
    v = np.sort(-np.log(np.clip(R[np.isfinite(R)], 1e-30, None)))
    if v.size == 0:
        return Normalize(0.0, 1.0)
    lo_ = float(v[0])
    hi_ = float(max(v[int(math.floor(0.97 * (v.size - 1)))], v[-2])) if v.size >= 5 else float(v[-1])
    return Normalize(lo_, hi_ if hi_ > lo_ else lo_ + 1.0)


def _corner(fig, run: Dict[str, Any], C: Dict[str, Any]) -> None:
    """Draw the corner plot of the trials, with its colorbar and key.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The panel.
    run : dict
        The fit's state.
    C : dict
        The colors.
    """
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    cmap = colormaps["viridis"]
    norm = _score_norm(run)
    left, bottom, right, top = 0.705, 0.09, 0.985, 0.83
    cw, ch = (right - left) / 3.0, (top - bottom) / 3.0
    padx, pady = 0.006, 0.016
    X = np.asarray(run["X"], float).reshape(-1, 3)
    R = np.asarray(run["R"], float)
    ok = np.isfinite(R)
    lo, hi = run["lo"], run["hi"]
    bx, tx = run["best_x"], run.get("trial_x")
    for i in range(3):
        for j in range(i + 1):
            ax = fig.add_axes([left + j * cw + padx, top - (i + 1) * ch + pady, cw - 2 * padx, ch - 2 * pady])
            if i == j:
                bins = np.linspace(lo[i], hi[i], 26)
                if ok.any():
                    ax.hist(X[ok, i], bins=bins, histtype="step", color=C["dim"], lw=1.0)
                    good = np.argsort(np.where(ok, R, np.inf))[:max(int(round(0.15 * ok.sum())), 1)]
                    ax.hist(X[good, i], bins=bins, color=C["fit"], lw=0.0, rwidth=0.85)
                if tx is not None and run.get("stage") in ("search", "profile"):
                    ax.axvline(tx[i], ls="--", lw=0.8, color=C["dim"])
                ax.axvline(bx[i], lw=1.2, color=C["best"])
                ax.set_xlim(lo[i], hi[i])
                ax.set_yticks([])
            else:
                _scatter(ax, run, j, i, C, norm, cmap)
                if j == 0:
                    _fmt_axis(ax, "y", i)
                    ax.set_ylabel(_LABELS[i], labelpad=1)
                else:
                    ax.set_yticklabels([])
            if i == 2:
                _fmt_axis(ax, "x", j)
                ax.set_xlabel(_LABELS[j], labelpad=1)
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
                ax.set_xticklabels([])
    cax = fig.add_axes([0.81, 0.795, 0.165, 0.014])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_edgecolor("#8c8c8c")
    cb.ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    fig.text(0.8925, 0.815, "-log residual, the best 15 % filled", ha="center", fontsize=7)
    keys = [Line2D([], [], marker="o", ls="", mfc="none", mec=C["fg"], ms=4, label="warm-up / explore"),
            Line2D([], [], marker="o", ls="", color=cmap(0.6), ms=4, label="TPE"),
            Line2D([], [], marker="s", ls="", mfc="none", mec=C["fg"], ms=5, label="best"),
            Line2D([], [], ls="--", color=C["dim"], lw=0.8, label="current trial")]
    fig.legend(handles=keys, loc="upper left", bbox_to_anchor=(0.893, 0.56), frameon=False, labelcolor=C["fg"],
               handletextpad=0.3, labelspacing=0.3)


def _spectrum(ax, st: Dict[str, Any], C: Dict[str, Any]) -> None:
    """Draw the partitions' contrasts, each with its sigma, against wavelength.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The slot.
    st : dict
        The panel's state.
    C : dict
        The colors.
    """
    sp = st["spectrum"]
    lam = [s.get("lam_um") for s in sp]
    by_lam = all(v is not None for v in lam)
    xs = np.array(lam if by_lam else range(len(sp)), float)
    sig = np.array([np.nan if s.get("sigma") is None else s["sigma"] for s in sp], float)
    ax.errorbar(xs, [s["contrast"] for s in sp], yerr=np.where(np.isfinite(sig), sig, 0.0), fmt="o", ms=3.5,
                color=C["dx"], ecolor=C["dx"], lw=1.0, capsize=0)
    ax.set_title("spectrum, the position held", loc="left")
    ax.set_ylabel("contrast", labelpad=1)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(6)
    _wavelength_axis(ax, xs, [s["partition"] for s in sp], by_lam)


def _wavelength_axis(ax, xs: np.ndarray, names: Sequence[str], by_lam: bool) -> None:
    """Label a slot's x axis with the wavelength, or with the partitions' names when one has none.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The slot.
    xs : ndarray
        The points' places.
    names : sequence of str
        The partitions' names.
    by_lam : bool
        Whether the places are wavelengths.
    """
    if by_lam:
        ax.set_xlabel("wavelength (um)", labelpad=1)
        ax.locator_params(axis="x", nbins=4)
    else:
        ax.set_xticks(xs)
        ax.set_xticklabels(list(names), rotation=45, fontsize=6)
        ax.set_xlabel("partition", labelpad=1)
    if xs.size == 1:
        ax.set_xlim(xs[0] - 1.0, xs[0] + 1.0)


def _chromatic(ax, st: Dict[str, Any], C: Dict[str, Any]) -> None:
    """Draw the partitions' own positions, less the combined fit, against wavelength.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The slot.
    st : dict
        The panel's state.
    C : dict
        The colors.
    """
    from matplotlib.lines import Line2D
    rows = st["chromatic"]["rows"]
    comb = st["chromatic"].get("combined")
    lam = [r.get("lam_um") for r in rows]
    by_lam = all(v is not None for v in lam)
    xs = np.array(lam if by_lam else range(len(rows)), float)
    base = comb if comb is not None else (float(np.mean([r["dx"] for r in rows])),
                                          float(np.mean([r["dy"] for r in rows])))
    off = 0.012 * (float(np.ptp(xs)) if xs.size > 1 else 1.0)
    for key, sk, mk, shift in (("dx", "sdx", "o", -off), ("dy", "sdy", "s", off)):
        v = np.array([r[key] for r in rows], float) - float(base[0 if key == "dx" else 1])
        s = np.array([np.nan if r.get(sk) is None else r[sk] for r in rows], float)
        ax.errorbar(xs + shift, v, yerr=np.where(np.isfinite(s), s, 0.0), fmt=mk, ms=3.5, color=C[key],
                    ecolor=C[key], lw=1.0, capsize=0)
    ax.axhline(0.0, ls="--", lw=0.9, color=C["fg"])
    ax.set_title("partitions' own positions", loc="left")
    ax.set_ylabel("less the combined fit (px)", labelpad=1)
    _wavelength_axis(ax, xs, [r["partition"] for r in rows], by_lam)
    keys = [Line2D([], [], marker="o", ls="", color=C["dx"], ms=4, label="dx"),
            Line2D([], [], marker="s", ls="", color=C["dy"], ms=4, label="dy"),
            Line2D([], [], ls="--", color=C["fg"], lw=0.9, label="combined fit")]
    ax.legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False,
              labelcolor=C["fg"], handletextpad=0.3, columnspacing=0.8, borderaxespad=0.0)


def _fakes_table(ax, st: Dict[str, Any]) -> None:
    """Write the fake sources' offsets from their true values as a table.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The slot.
    st : dict
        The panel's state.
    """
    fk = st["fakes"]
    rows = list(fk.get("rows") or [])
    ax.set_axis_off()
    ctx = fk.get("ctx") or ""
    ax.set_title("fake sources, " + (f"partition {ctx}" if ctx else "the companion's fit"), loc="left")
    lines = ["   #  d rho (mas)  d PA (deg)  d c (%)"]
    first = max(len(rows) - 9, 0)
    for k, f in enumerate(rows[first:], start=first + 1):
        lines.append(f"{k:4d}  {1e3 * f['d_rho']:+10.2f}  {f['d_pa']:+9.3f}  {100 * f['d_c']:+7.1f}")
    if len(rows) >= 2:
        a = np.array([[f["d_rho"], f["d_pa"], f["d_c"]] for f in rows], float)
        m, s = a.mean(axis=0), a.std(axis=0, ddof=1)
        lines += [f"mean  {1e3 * m[0]:+10.2f}  {m[1]:+9.3f}  {100 * m[2]:+7.1f}",
                  f"sd    {1e3 * s[0]:10.2f}  {s[1]:9.3f}  {100 * s[2]:7.1f}"]
    n = int(fk.get("n") or 0)
    if n and len(rows) < n:
        lines.append(f"fake {len(rows) + 1} of {n} being fitted")
    ax.text(0.0, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", family="monospace", fontsize=6.8,
            linespacing=1.3)


def _extras(st: Dict[str, Any]) -> List[str]:
    """Return the later stages' slots the panel has, in order.

    Parameters
    ----------
    st : dict
        The panel's state.

    Returns
    -------
    names : list of str
        Any of ``spectrum``, ``chromatic`` and ``fakes``.
    """
    out = []
    if st.get("spectrum"):
        out.append("spectrum")
    if (st.get("chromatic") or {}).get("rows"):
        out.append("chromatic")
    if st.get("fakes") is not None:
        out.append("fakes")
    return out


def _slots(fig, st: Dict[str, Any], run: Dict[str, Any], C: Dict[str, Any]) -> None:
    """Draw the later stages' 2 x 2 grid in the corner plot's place.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The panel.
    st : dict
        The panel's state.
    run : dict
        The fit's state.
    C : dict
        The colors.
    """
    from matplotlib import colormaps
    items: List[Any] = list(_extras(st))
    items += [(0, 1), (0, 2), (1, 2)][:4 - len(items)]
    cmap = colormaps["viridis"]
    norm = _score_norm(run)
    for k, item in enumerate(items):
        r, c = divmod(k, 2)
        x0, y0 = 0.675 + c * 0.155, 0.46 - r * 0.37
        ax = fig.add_axes([x0 + 0.045, y0 + 0.06, 0.155 - 0.06, 0.37 - 0.12])
        if item == "spectrum":
            _spectrum(ax, st, C)
        elif item == "chromatic":
            _chromatic(ax, st, C)
        elif item == "fakes":
            _fakes_table(ax, st)
        else:
            a, b = item
            _scatter(ax, run, a, b, C, norm, cmap)
            _fmt_axis(ax, "x", a)
            _fmt_axis(ax, "y", b)
            ax.set_xlabel(_LABELS[a], labelpad=1)
            ax.set_ylabel(_LABELS[b], labelpad=1)
            ax.set_title("trials by -log residual", loc="left")


def _text(fig, st: Dict[str, Any], run: Dict[str, Any], live: List[Tuple[float, float]], source: str) -> None:
    """Write the panel's text: the stage, the values, the times and the live minimum with its sigma.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The panel.
    st : dict
        The panel's state.
    run : dict
        The fit's state.
    live : list of tuple of 2 floats
        Each parameter's minimum and 1-sigma.
    source : str
        Where they come from, ``"trials"`` or ``"profiles"``.
    """
    label = st["geom"].get("label") or ""
    bx, br = run["best_x"], run["best_r"]
    tx = run.get("trial_x")
    per = run.get("per_trial")
    lines = [(f"{label}    " if label else "") + str(run.get("title") or ""),
             f"best : rho {bx[0]:.5f}\"  PA {bx[1] % 360:8.3f}  c {bx[2]:.3e}  resid {br:.4f}",
             (f"trial: rho {tx[0]:.5f}\"  PA {tx[1] % 360:8.3f}  c {tx[2]:.3e}  resid {run.get('trial_r', np.nan):.4f}"
              if tx is not None else "trial: -"),
             (f"{per:.2f} s/trial" if per is not None and np.isfinite(per) else "- s/trial")
             + f"    elapsed {_hms(run.get('elapsed'))}    ETA {_hms(run.get('eta'))}    all {_hms(st.get('elapsed_all'))}",
             f"min +- 1-sigma, from the {source}:"]
    if any(np.isfinite(s) for _, s in live):
        parts = []
        for name, d, fm, fs in (("rho", 0, ".5f", ".1e"), ("PA", 1, ".3f", ".3f"), ("c", 2, ".3e", ".1e")):
            m, s = live[d]
            parts.append(f"{name} --" if not np.isfinite(s) else
                         f"{name} {(m % 360 if d == 1 else m):{fm}} +- {s:{fs}}")
        lines.append("  ".join(parts))
    else:
        lines.append("(accumulating...)")
    fig.text(0.655, 0.988, "\n".join(lines), va="top", family="monospace", fontsize=7.5, linespacing=1.3)


# ----------------------------------------------------------------------------- the panel
def render_panel(st: Dict[str, Any], path: str) -> str:
    """Draw one panel of the extraction's dashboard.

    Parameters
    ----------
    st : dict
        The panel's state, :meth:`ExtractDisplay.state`.
    path : str
        Where the PNG goes.  It is written beside it first and moved into place, so a viewer never reads half
        of one.

    Returns
    -------
    path : str
        The PNG's path.
    """
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    C = _colors()
    run = st["run"]
    with matplotlib.rc_context(_RC):
        fig = Figure(figsize=(PANEL_PX[0] / 100.0, PANEL_PX[1] / 100.0), dpi=100)
        FigureCanvasAgg(fig)
        _images(fig, st, C)
        _trace(fig, run, C)
        live, source = _parabolas(fig, run, C)
        if _extras(st):
            _slots(fig, st, run, C)
        else:
            _corner(fig, run, C)
        _text(fig, st, run, live, source)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        part = path + ".part"
        fig.savefig(part, format="png", dpi=100)
    os.replace(part, path)
    return path


def _draw_quietly(st: Dict[str, Any], path: str, errors: List[str]) -> str:
    """Draw a panel, keeping its failure rather than raising it.

    Parameters
    ----------
    st : dict
        The panel's state.
    path : str
        Where the PNG goes.
    errors : list of str
        Where a failure is kept.

    Returns
    -------
    path : str
        The PNG's path, empty when the drawing failed.
    """
    try:
        return render_panel(st, path)
    except Exception as exc:                               # noqa: BLE001 - kept for the log
        errors.append(repr(exc))
        return ""


# ----------------------------------------------------------------------------- the live display
class ExtractDisplay:
    """Draw ``klip-tpe extract``'s dashboard as the extraction goes, from its progress.

    An instance is the ``progress`` callback of :func:`klip_tpe.extract.extract_companion`.  It keeps what
    the panels need, the cut-outs around the star and the trials, and hands each panel's state to the panel
    process.  A failure of the display stops the display alone, with a line in the log.

    Parameters
    ----------
    out_dir : str
        The extraction's directory.  The panels go to its ``steps/``, whose panels of an earlier extraction
        are removed first.
    label : str, default "companion"
        What is extracted, for the panels' text.
    every : int, default 1
        Number of trials between two panels.  A panel asked for while the last is still being drawn is
        skipped, but for the last of each stage and those of the partitions and the fake sources.
    reference : sequence of 2 floats or None, default None
        A reference position ``(rho, pa)``, in arcsec and degrees, drawn as a green circle.
    psf_fn : callable or None, default None
        Function ``psf_fn(rho, pa, shape)`` that returns the image of a unit source as the reduction shows it,
        such as :func:`klip_tpe.injection.injected_model` does.  It is called once, at the start.  None draws a
        Gaussian of the FWHM.
    engine : {"process", "thread", "inline"} or None, default None
        Where the panels are drawn: the panel process, a thread of this process, or this thread as they are
        asked for.  None takes :func:`klip_tpe.panels.engine`.
    log : callable or None, default None
        Function that takes a line.

    Attributes
    ----------
    paths : list of str
        The panels asked for, in order.
    skipped : int
        Number of panels skipped while one was being drawn.
    keep_crops : bool
        Whether :meth:`close` saves the companion's trials' cut-outs, True.

    Raises
    ------
    ValueError
        If ``engine`` is none of the three.
    """

    def __init__(self, out_dir: str, label: str = "companion", every: int = 1,
                 reference: Optional[Sequence[float]] = None, psf_fn: Optional[Callable[..., Any]] = None,
                 engine: Optional[str] = None, log: Optional[Callable[[str], None]] = None):
        if engine is None:
            from .panels import engine as _engine
            engine = _engine()
        if engine not in ("process", "thread", "inline"):
            raise ValueError(f"engine {engine!r} is not process, thread or inline")
        self.out_dir = os.path.abspath(out_dir)
        self.steps = os.path.join(self.out_dir, "steps")
        self.label = str(label)
        self.every = max(int(every), 1)
        self.reference = None if reference is None else (float(reference[0]), float(reference[1]))
        self.psf_fn = psf_fn
        self.engine = engine
        self.log = log or (lambda s: None)
        self.paths: List[str] = []
        self.skipped = 0
        self.keep_crops = True
        self.geom: Optional[Dict[str, Any]] = None
        self.psf: Optional[Dict[str, Any]] = None
        self.main: Optional[Dict[str, Any]] = None
        self.run: Optional[Dict[str, Any]] = None
        self.status: Optional[str] = None
        self.spectrum: List[Dict[str, Any]] = []
        self.chromatic: Dict[str, Any] = {"rows": [], "combined": None}
        self.fakes: Optional[Dict[str, Any]] = None
        self.crops: List[np.ndarray] = []
        self.crop_x: List[List[float]] = []
        self.t0 = time.time()
        self.off = False
        self._pp = None
        self._ex = None
        self._fut = None
        self._errors: List[str] = []
        self._told = 0

    # -- the extraction's progress ------------------------------------------------------------
    def __call__(self, ev: Dict[str, Any]) -> None:
        """Take one event of the extraction's progress.

        Parameters
        ----------
        ev : dict
            The event, its ``kind`` one of ``start``, ``trial`` and ``done`` (from
            :func:`klip_tpe.extract.fit_companion`), ``group`` and ``group_fakes`` (from
            :func:`klip_tpe.extract.extract_companion`).  Others are ignored.
        """
        if self.off:
            return
        try:
            fn = getattr(self, "_on_" + str(ev.get("kind")), None)
            if fn is not None:
                fn(ev)
        except Exception as exc:                           # noqa: BLE001 - the extraction goes on
            self.off = True
            self.log(f"  extract display: stopped ({exc!r}), the extraction goes on")

    def _on_start(self, ev: Dict[str, Any]) -> None:
        """Begin a fit's panels, and at the first fit the geometry and the PSF model.

        Parameters
        ----------
        ev : dict
            ``phase``, ``n``, ``lo``, ``hi``, ``clean``, ``start``, ``pxscale``, ``fwhm_px`` and
            ``angle_convention``, with ``true``, ``n_fakes`` and ``partition`` where they apply.
        """
        clean = np.asarray(ev["clean"], float)
        if self.geom is None:
            pxs, fwhm = float(ev["pxscale"]), float(ev["fwhm_px"])
            half = int(math.ceil(float(ev["start"][0]) / pxs + 3.0 * fwhm))
            self.geom = {"half": half, "pxscale": pxs, "fwhm_px": fwhm, "conv": str(ev.get("angle_convention", "pa")),
                         "label": self.label}
            self.psf = self._psf(ev["start"], clean.shape)
            self._clear()
            self.log(f"  extract display: panels to {self.steps}, shown by klip-tpe view --run-dir {self.out_dir}")
        lo, hi = [float(v) for v in ev["lo"]], [float(v) for v in ev["hi"]]
        start = [float(v) for v in ev["start"]]
        start[1] = _unwrap(start[1], lo[1])
        run = {"phase": str(ev.get("phase", "main")), "lo": lo, "hi": hi, "X": [], "R": [], "flags": [],
               "clean": crop(clean, self.geom["half"]), "best": None, "trial": None, "best_x": start,
               "best_r": float("inf"), "trial_x": None, "trial_r": float("nan"), "profiles": None, "stage": "search",
               "it": 0, "n_stage": int(ev["n"]), "count": 0, "t0": time.time(),
               "true": None if ev.get("true") is None else [float(v) for v in ev["true"]]}
        self.run = run
        self.status = None
        if self.main is None:
            self.main = run
        if run["true"] is not None:
            ctx = str(ev.get("partition") or "")
            if self.fakes is None or self.fakes["ctx"] != ctx:
                self.fakes = {"ctx": ctx, "rows": [], "n": int(ev.get("n_fakes") or 0)}

    def _on_trial(self, ev: Dict[str, Any]) -> None:
        """Take a trial of the search or a point of the profile scans, and draw a panel when one is due.

        Parameters
        ----------
        ev : dict
            ``stage`` (``search`` or ``profile``), ``it``, ``n``, ``x``, ``r``, ``img``, ``flag``, ``best_x``,
            ``best_r``, ``best_img``, ``new_best`` and, in the profile scans, ``profiles``.
        """
        run = self.run
        if run is None or self.geom is None:
            return
        h = self.geom["half"]
        x = [float(v) for v in ev["x"]]
        stage = str(ev.get("stage", "search"))
        if stage == "search":
            run["X"].append(x)
            run["R"].append(float(ev["r"]))
            run["flags"].append(str(ev.get("flag", "")))
            if run is self.main and self.keep_crops:
                self.crops.append(crop(ev["img"], h))
                self.crop_x.append(x)
        elif ev.get("profiles") is not None:
            run["profiles"] = ev["profiles"]
        run["stage"], run["it"], run["n_stage"] = stage, int(ev["it"]), int(ev["n"])
        run["count"] += 1
        run["trial_x"], run["trial_r"] = x, float(ev["r"])
        if ev.get("new_best") or run["best"] is None:
            bx = [float(v) for v in ev["best_x"]]
            bx[1] = _unwrap(bx[1], run["lo"][1])
            run["best"], run["best_x"], run["best_r"] = crop(ev["best_img"], h), bx, float(ev["best_r"])
        last = run["it"] >= run["n_stage"]
        if last or run["it"] % self.every == 0:
            run["trial"] = crop(ev["img"], h)
            self._draw(force=last)

    def _on_done(self, ev: Dict[str, Any]) -> None:
        """Finish a fit: its best, and a fake source's row of the table or a partition's own position.

        Parameters
        ----------
        ev : dict
            ``fit`` (:func:`klip_tpe.extract.fit_companion`'s result), with ``true`` for a fake source and
            ``partition`` and ``lam_um`` for a partition's own position.
        """
        run = self.run
        fit = ev.get("fit") or {}
        b = fit.get("best")
        if run is None or self.geom is None or not b:
            return
        g = self.geom
        bx = [float(b["rho"]), _unwrap(float(b["pa"]), run["lo"][1]), float(b["contrast"])]
        run["best_x"], run["best_r"], run["stage"] = bx, float(b.get("residual", run["best_r"])), "done"
        best_img = (fit.get("images") or {}).get("best")
        if best_img is not None:
            run["best"] = crop(best_img, g["half"])
        if ev.get("true") is not None:
            if self.fakes is not None:
                t = [float(v) for v in ev["true"]]
                self.fakes["rows"].append({"d_rho": bx[0] - t[0], "d_pa": ((bx[1] - t[1] + 180.0) % 360.0) - 180.0,
                                           "d_c": bx[2] / t[2] - 1.0})
        elif ev.get("partition") is not None:
            pf = fit.get("profile_fits") or []
            s = ([p["sigma"] for p in pf] if len(pf) >= 2
                 else list((fit.get("quadratic") or {}).get("sigma") or [np.nan] * 3))
            dx, dy = _dxdy(bx[0], bx[1], g["pxscale"], g["conv"])
            sdx, sdy = _xy_sigma(bx[0], bx[1], float(s[0]), float(s[1]), g["pxscale"], g["conv"])
            self.chromatic["rows"].append({"partition": str(ev["partition"]), "lam_um": ev.get("lam_um"), "dx": dx,
                                           "dy": dy, "sdx": sdx, "sdy": sdy})
        if run is self.main:
            self.chromatic["combined"] = _dxdy(bx[0], bx[1], g["pxscale"], g["conv"])
        self._draw(force=True)

    def _on_group(self, ev: Dict[str, Any]) -> None:
        """Add a partition's contrast, the position held, to the spectrum.

        Parameters
        ----------
        ev : dict
            ``k``, ``n`` and ``group``, with the partition's ``partition``, ``lam_um``, ``contrast`` and
            ``sigma``.
        """
        g = ev["group"]
        sig = g.get("sigma")
        self.spectrum.append({"partition": str(g["partition"]), "lam_um": g.get("lam_um"),
                              "contrast": float(g["contrast"]), "sigma": float("nan") if sig is None else float(sig)})
        self.run = self.main
        self.status = f"the partitions' contrasts, the position held: {ev.get('k')} of {ev.get('n')}"
        self._draw(force=True)

    def _on_group_fakes(self, ev: Dict[str, Any]) -> None:
        """Widen a partition's sigma in the spectrum to its fake sources' scatter.

        Parameters
        ----------
        ev : dict
            ``k``, ``n``, ``partition`` and ``scatter``, the fakes' scatter as a fraction of the contrast.
        """
        sc = float("nan") if ev.get("scatter") is None else float(ev["scatter"])
        for s in self.spectrum:
            if s["partition"] == str(ev.get("partition")) and np.isfinite(sc):
                s["sigma"] = float(np.nanmax([s["sigma"], sc * s["contrast"]]))
        self.run = self.main
        self.status = f"the partitions' fake sources: {ev.get('k')} of {ev.get('n')}"
        self._draw(force=True)

    # -- the panels ---------------------------------------------------------------------------
    def _psf(self, start: Sequence[float], shape: Tuple[int, ...]) -> Optional[Dict[str, Any]]:
        """Return the PSF model's stamp around the start, with a peak of 1.

        Parameters
        ----------
        start : sequence of float
            The start, ``(rho, pa, contrast)``.
        shape : tuple of int
            The reduced image's shape.

        Returns
        -------
        psf : dict or None
            ``stamp``, ``pa`` and the start's offsets ``fx`` and ``fy`` from the stamp's center pixel, None
            without ``psf_fn`` or when it gives nothing.
        """
        if self.psf_fn is None:
            return None
        from .metrics import star_center
        g = self.geom
        try:
            img = self.psf_fn(float(start[0]), float(start[1]), tuple(int(v) for v in shape))
            if img is None:
                return None
            img = np.nan_to_num(np.asarray(img, float))
            cx, cy = star_center(img.shape)
            dx, dy = _dxdy(start[0], start[1], g["pxscale"], g["conv"])
            x, y = cx + dx, cy + dy
            ix, iy = int(round(x)), int(round(y))
            hs = max(int(math.ceil(2.5 * g["fwhm_px"])), 3)
            stamp = np.zeros((2 * hs + 1, 2 * hs + 1))
            ys = slice(max(iy - hs, 0), min(iy + hs + 1, img.shape[0]))
            xs = slice(max(ix - hs, 0), min(ix + hs + 1, img.shape[1]))
            if ys.start < ys.stop and xs.start < xs.stop:
                stamp[ys.start - (iy - hs):ys.stop - (iy - hs), xs.start - (ix - hs):xs.stop - (ix - hs)] = img[ys, xs]
            peak = float(stamp.max())
            if not peak > 0:
                return None
            return {"stamp": (stamp / peak).astype(np.float32), "pa": float(start[1]), "fx": x - ix, "fy": y - iy}
        except Exception as exc:                           # noqa: BLE001 - the Gaussian stands in
            self.log(f"  extract display: no PSF model ({exc!r}), a Gaussian of the FWHM instead")
            return None

    def _clear(self) -> None:
        """Remove the panels an earlier extraction left in ``steps/``."""
        old = glob.glob(os.path.join(self.steps, "step*.png")) + glob.glob(os.path.join(self.steps, "step*.part"))
        for p in old:
            try:
                os.remove(p)
            except OSError:
                pass
        if old:
            self.log(f"  extract display: {len(old)} panels of an earlier extraction removed from {self.steps}")

    def _title(self, run: Dict[str, Any]) -> str:
        """Return the panel's first line, the stage and how far it is.

        Parameters
        ----------
        run : dict
            The fit's state.

        Returns
        -------
        title : str
            Such as ``trial 412 / 1000``, ``profile scans 12 / 63`` or ``fake 2 of 5: trial 40 / 300``.
        """
        if self.status:
            return self.status
        ph = "" if run["phase"] == "main" else run["phase"] + ": "
        if run["stage"] == "profile":
            return f"{ph}profile scans {run['it']} / {run['n_stage']}"
        if run["stage"] == "done":
            return f"{ph}done, {len(run['R'])} trials"
        return f"{ph}trial {run['it']} / {run['n_stage']}"

    def state(self) -> Dict[str, Any]:
        """Return the state a panel is drawn from, apart from what the extraction goes on changing.

        Returns
        -------
        st : dict
            ``geom`` (the cut-outs' half side, the plate scale, the FWHM, the angle convention and the label),
            ``run`` (the fit's cut-outs, trials, box, best and trial, profiles, stage, title and times),
            ``psf``, ``reference``, ``spectrum``, ``chromatic``, ``fakes`` and ``elapsed_all``.
        """
        src = self.run
        run = {k: v for k, v in src.items() if k not in ("X", "R", "flags", "t0")}
        run["X"] = np.asarray(src["X"], float).reshape(-1, 3)
        run["R"] = np.asarray(src["R"], float)
        run["flags"] = list(src["flags"])
        run["title"] = self._title(src)
        now = time.time()
        timed = src.get("timing") is not False
        run["elapsed"] = now - src["t0"] if timed else None
        run["per_trial"] = run["elapsed"] / src["count"] if timed and src["count"] else None
        run["eta"] = (run["per_trial"] * max(src["n_stage"] - src["it"], 0)
                      if run["per_trial"] is not None and src["stage"] in ("search", "profile") else None)
        return {"geom": dict(self.geom), "run": run, "psf": self.psf, "reference": self.reference,
                "spectrum": [dict(s) for s in self.spectrum],
                "chromatic": {"rows": [dict(r) for r in self.chromatic["rows"]],
                              "combined": self.chromatic["combined"]},
                "fakes": None if self.fakes is None else {"ctx": self.fakes["ctx"], "n": self.fakes["n"],
                                                          "rows": [dict(r) for r in self.fakes["rows"]]},
                "elapsed_all": now - self.t0 if timed else None}

    def _busy(self) -> bool:
        """Return whether a panel is being drawn.

        Returns
        -------
        busy : bool
            True while the panel process or the thread has one.
        """
        if self.engine == "process" and self._pp is not None and self._pp.broken is None:
            return self._pp.busy("extract_panel")
        return self._fut is not None and not self._fut.done()

    def _reap(self) -> None:
        """Read the panel process' reports, and log the first panel that failed."""
        if self._pp is not None:
            for rep in self._pp.take():
                if not rep.get("ok") and rep.get("kind") == "extract_panel":
                    err = str(rep.get("error") or "?").strip()
                    self._errors.append(err.splitlines()[0] if err else "?")
        if len(self._errors) > self._told:
            if not self._told:
                self.log(f"  extract display: a panel failed ({self._errors[0]})")
            self._told = len(self._errors)

    def _draw(self, force: bool = False) -> None:
        """Ask for a panel, unless one is being drawn and this one may be skipped.

        Parameters
        ----------
        force : bool, default False
            Whether to ask for it even while one is being drawn.
        """
        if self.off or self.geom is None or self.run is None:
            return
        self._reap()
        if not force and self._busy():
            self.skipped += 1
            return
        path = os.path.join(self.steps, f"step{len(self.paths) + 1:04d}.png")
        st = self.state()
        self.paths.append(path)
        if self.engine == "process":
            if self._pp is None:
                try:
                    from .panels import PanelProcess
                    self._pp = PanelProcess({"run_dir": self.out_dir, "dpi": 100, "corner_mode": None})
                except Exception as exc:                   # noqa: BLE001 - a thread draws instead
                    self.log(f"  extract display: the panel process did not start ({exc!r}), drawing on a thread")
                    self.engine = "thread"
            if self.engine == "process" and self._pp.broken is None:
                self._pp.send({"op": "extract_panel", "st": st, "png": path}, reply=True)
                return
            if self.engine == "process":
                self.log(f"  extract display: the panel process stopped ({self._pp.broken}), drawing on a thread")
                self.engine = "thread"
        if self.engine == "thread":
            if self._ex is None:
                import concurrent.futures as cf
                self._ex = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="klip-extract-panel")
            self._fut = self._ex.submit(_draw_quietly, st, path, self._errors)
            return
        _draw_quietly(st, path, self._errors)

    def close(self, movie: bool = True) -> Dict[str, str]:
        """Finish the panels, save the companion's trials' cut-outs and make the progress movie.

        Parameters
        ----------
        movie : bool, default True
            Whether to make ``extract_steps.gif``, and ``extract_steps.mp4`` where imageio has ffmpeg.

        Returns
        -------
        files : dict of str to str
            ``crops`` and ``movie``, where written.
        """
        out: Dict[str, str] = {}
        if self.keep_crops and self.crops and self.main is not None and self.geom is not None:
            p = os.path.join(self.out_dir, CROPS)
            try:
                os.makedirs(self.out_dir, exist_ok=True)
                np.savez_compressed(p, clean=self.main["clean"], trials=np.stack(self.crops).astype(np.float32),
                                    x=np.asarray(self.crop_x, float), half=int(self.geom["half"]))
                out["crops"] = p
            except Exception as exc:                       # noqa: BLE001 - the panels stand without it
                self.log(f"  extract display: the trials' cut-outs were not saved ({exc!r})")
        pp = self._pp
        if pp is not None and not pp.wait(timeout=PROCESS_WAIT):
            self.log(f"  extract display: the panel process did not finish ({pp.broken or 'timed out'})")
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None
        self._reap()
        frames = [p for p in self.paths if os.path.exists(p)]
        if movie and frames:
            gif = os.path.join(self.out_dir, "extract_steps.gif")
            mp4 = os.path.join(self.out_dir, "extract_steps.mp4")
            made = False
            if pp is not None and pp.broken is None:
                pp.send({"op": "movie", "frames": frames, "gif": gif, "mp4": mp4, "hold_last": 6}, reply=True)
                made = pp.wait(timeout=PROCESS_WAIT)
            if not made:
                try:
                    from .animate import make_movie
                    make_movie(frames, gif, mp4, intro=None, intro_frames=0, hold_last=6)
                except Exception as exc:                   # noqa: BLE001 - the panels stand without it
                    self.log(f"  extract display: no movie ({exc!r})")
            if os.path.exists(gif):
                out["movie"] = gif
        if pp is not None:
            pp.close()
            self._pp = None
        if self.paths:
            self.log(f"  extract display: {len(frames)} panels in {self.steps}"
                     + (f", {self.skipped} skipped while one was being drawn" if self.skipped else "")
                     + (f", movie {out['movie']}" if "movie" in out else ""))
        return out


# ----------------------------------------------------------------------------- afterwards
def _read_table(path: str) -> List[List[str]]:
    """Return a table's rows, its comment lines left out.

    Parameters
    ----------
    path : str
        The table.

    Returns
    -------
    rows : list of list of str
        Each row's fields.
    """
    with open(path) as f:
        return [ln.split() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def _approx_trial(clean: np.ndarray, best: np.ndarray, bx: Sequence[float], tx: Sequence[float],
                  geom: Dict[str, Any]) -> np.ndarray:
    """Return a trial's image as the best's injected model moved to the trial and scaled to its contrast.

    The best's injected model (the original less the best null) is moved by the trial's offset from the best,
    scaled by the ratio of their contrasts and taken from the original.  It stands in for a trial whose image
    was not saved, to first order in the injection.

    Parameters
    ----------
    clean : ndarray, shape (n, n)
        The original's cut-out.
    best : ndarray, shape (n, n)
        The best null's cut-out.
    bx : sequence of float
        The best's ``(rho, pa, contrast)``.
    tx : sequence of float
        The trial's.
    geom : dict
        The plate scale and the angle convention.

    Returns
    -------
    trial : ndarray, shape (n, n)
        The stand-in.
    """
    from scipy import ndimage
    model = np.nan_to_num(np.asarray(clean, float) - np.asarray(best, float))
    xb, yb = _dxdy(bx[0], bx[1], geom["pxscale"], geom["conv"])
    xt, yt = _dxdy(tx[0], tx[1], geom["pxscale"], geom["conv"])
    moved = ndimage.shift(model, (yt - yb, xt - xb), order=1, mode="constant", cval=0.0)
    return np.asarray(clean, float) - (float(tx[2]) / float(bx[2])) * moved


def render_saved(out_dir: str, every: int = 10, reference: Optional[Sequence[float]] = None, movie: bool = True,
                 log: Optional[Callable[[str], None]] = None) -> List[str]:
    """Draw the dashboard's panels afterwards, from a finished extraction's saved tables and images.

    The trials come from ``extract_trials.txt``, the profile scans from ``extract_profiles.txt``, the
    original and the best null from ``extract_clean.fits`` and ``extract_residual.fits``, and the partitions,
    their own positions and the fake sources from ``extract_summary.json``.  Each trial's image comes from
    ``extract_trial_crops.npz`` when the live display saved it for these trials.  Without it, a trial's image
    is the best's injected model moved to the trial and scaled to its contrast (:func:`_approx_trial`),
    titled as an approximation, and so is each point of the profile scans.  The fake sources' and the
    partitions' own fits were not saved trial by trial, so they come as their tables only.

    Parameters
    ----------
    out_dir : str
        The extraction's directory.
    every : int, default 10
        Number of trials between two panels.
    reference : sequence of 2 floats or None, default None
        A reference position ``(rho, pa)``, in arcsec and degrees.
    movie : bool, default True
        Whether to make the progress movie.
    log : callable or None, default None
        Function that takes a line.

    Returns
    -------
    paths : list of str
        The panels, in order, and the movie last when one was made.

    Raises
    ------
    FileNotFoundError
        If the directory lacks the summary, the trials or the images.

    Notes
    -----
    The panels go to ``steps/`` when it holds none, else to ``steps_saved/``, so a live display's panels are
    never overwritten.  The movie is ``extract_steps.gif``, or ``extract_steps_saved.gif`` beside a live
    display's.
    """
    from astropy.io import fits
    say = log or (lambda s: None)
    out_dir = os.path.abspath(out_dir)
    with open(os.path.join(out_dir, "extract_summary.json")) as f:
        s = json.load(f)
    rows = _read_table(os.path.join(out_dir, "extract_trials.txt"))
    clean = fits.getdata(os.path.join(out_dir, "extract_clean.fits")).astype(float)
    best_full = fits.getdata(os.path.join(out_dir, "extract_residual.fits")).astype(float)
    lo = [float(s["box"]["lo"][k]) for k in ("rho", "pa", "contrast")]
    hi = [float(s["box"]["hi"][k]) for k in ("rho", "pa", "contrast")]
    X = np.array([[float(r[1]), _unwrap(float(r[2]), lo[1]), float(r[3])] for r in rows], float).reshape(-1, 3)
    R = np.array([float(r[4]) for r in rows], float)
    flags = [r[5] if len(r) > 5 else "" for r in rows]
    live = bool(glob.glob(os.path.join(out_dir, "steps", "step*.png")))
    disp = ExtractDisplay(out_dir, label=str(s.get("label") or "companion"), every=max(int(every), 1),
                          reference=reference, engine="inline", log=say)
    disp.keep_crops = False
    if live:
        disp.steps = os.path.join(out_dir, "steps_saved")
        say(f"extract render: {os.path.join(out_dir, 'steps')} holds the live display's panels, these go to "
            f"{disp.steps}")
    start = [float(s["start"][k]) for k in ("rho", "pa", "contrast")]
    sb, tb = s["best"], s.get("tpe_best") or s["best"]
    final = [float(sb["rho"]), _unwrap(float(sb["pa"]), lo[1]), float(sb["contrast"])]
    center = [float(tb["rho"]), _unwrap(float(tb["pa"]), lo[1]), float(tb["contrast"])]
    disp({"kind": "start", "phase": "main", "n": len(R), "lo": lo, "hi": hi, "clean": clean, "start": start,
          "pxscale": float(s["pxscale"]), "fwhm_px": float(s["fwhm_px"]),
          "angle_convention": str(s.get("angle_convention", "pa"))})
    if disp.off or disp.run is None:
        return []
    g = disp.geom
    h = g["half"]
    c_clean, c_best = crop(clean, h), crop(best_full, h)
    crops = None
    cp = os.path.join(out_dir, CROPS)
    if os.path.exists(cp):
        with np.load(cp) as z:
            xs, half, trials = np.asarray(z["x"], float).reshape(-1, 3), int(z["half"]), z["trials"]
            if xs.shape == X.shape:
                xs[:, 1] = [_unwrap(v, lo[1]) for v in xs[:, 1]]
            if (xs.shape == X.shape and half == h and tuple(trials.shape[1:]) == c_clean.shape
                    and np.allclose(xs, X, rtol=1e-4, atol=1e-5)):
                crops = np.asarray(trials, float)
        say(f"extract render: the trials' images from {CROPS}" if crops is not None
            else f"extract render: {CROPS} belongs to other trials, left out")
    if crops is None:
        say("extract render: the trials' images were not saved, so each is the best's injected model moved to it")
    disp.run["timing"] = False
    disp.run["trial_approx"] = crops is None

    def image(i: int) -> np.ndarray:
        return crops[i] if crops is not None else _approx_trial(c_clean, c_best, final, X[i], g)

    ok = np.isfinite(R)
    ib = -1
    every_ = disp.every
    for i in range(len(R)):
        new = bool(ok[i] and (ib < 0 or R[i] < R[ib]))
        if new:
            ib = i
        due = (i + 1) % every_ == 0 or i + 1 == len(R)
        disp({"kind": "trial", "phase": "main", "stage": "search", "it": i + 1, "n": len(R), "x": X[i], "r": R[i],
              "img": image(i) if due else c_best, "flag": flags[i], "best_x": X[ib] if ib >= 0 else start,
              "best_r": R[ib] if ib >= 0 else float("inf"),
              "best_img": (image(ib) if ib >= 0 else c_clean) if (new or i == 0) else c_best,
              "new_best": new or i == 0})
    prof_path = os.path.join(out_dir, "extract_profiles.txt")
    if os.path.exists(prof_path) and not disp.off:
        by: Dict[str, List[Tuple[float, float]]] = {}
        for r in _read_table(prof_path):
            by.setdefault(r[0], []).append((float(r[1]), float(r[2])))
        names = [k for k in ("rho", "pa", "contrast") if k in by]
        npts = sum(len(by[k]) for k in names)
        disp.run["trial_approx"] = True
        prof: Dict[str, List[List[float]]] = {"values": [], "residual": []}
        k = 0
        for d, name in enumerate(names):
            vals = [v for v, _ in by[name]]
            res = [r for _, r in by[name]]
            for j in range(len(vals)):
                k += 1
                x = list(center)
                x[d] = vals[j]
                due = k % every_ == 0 or k == npts
                disp({"kind": "trial", "phase": "main", "stage": "profile", "it": k, "n": npts, "x": x, "r": res[j],
                      "img": _approx_trial(c_clean, c_best, final, x, g) if due else c_best, "best_x": final,
                      "best_r": float(sb["residual"]), "best_img": c_best, "new_best": k == 1,
                      "profiles": {"values": prof["values"] + [vals[:j + 1]],
                                   "residual": prof["residual"] + [res[:j + 1]]}})
            prof["values"].append(vals)
            prof["residual"].append(res)
    disp({"kind": "done", "phase": "main", "fit": {"best": dict(sb), "images": {"best": best_full}}})
    groups = s.get("groups") or []
    for k, gr in enumerate(groups):
        sig = gr.get("sigma_quoted")
        disp({"kind": "group", "phase": "partitions", "k": k + 1, "n": len(groups),
              "group": {"partition": gr["partition"], "lam_um": gr.get("lam_um"), "contrast": gr["contrast"],
                        "sigma": sig if sig is not None else gr.get("sigma")}})
    rec = s.get("recovery") or {}
    gp = (s.get("group_positions") or {}).get("rows") or []
    if (rec.get("fakes") or gp) and not disp.off:
        if rec.get("fakes"):
            disp.fakes = {"ctx": "", "n": len(rec["fakes"]), "rows": [
                {"d_rho": fk["fit"]["rho"] - fk["true"]["rho"],
                 "d_pa": ((fk["fit"]["pa"] - fk["true"]["pa"] + 180.0) % 360.0) - 180.0,
                 "d_c": fk["fit"]["contrast"] / fk["true"]["contrast"] - 1.0} for fk in rec["fakes"]]}
        for r in gp:
            disp.chromatic["rows"].append({"partition": r["partition"], "lam_um": r.get("lam_um"), "dx": r["dx_px"],
                                           "dy": r["dy_px"], "sdx": r.get("sigma_dx_px"), "sdy": r.get("sigma_dy_px")})
        if gp:
            disp.chromatic["combined"] = (float(sb["dx_px"]), float(sb["dy_px"]))
        disp.run, disp.status = disp.main, "the extraction's results"
        disp._draw(force=True)
    paths = [p for p in disp.paths if os.path.exists(p)]
    disp.close(movie=False)
    if disp.off:
        say("extract render: the display stopped early, see the line above")
    if movie and paths:
        try:
            from .animate import make_movie
            name = "extract_steps_saved" if live else "extract_steps"
            gif = os.path.join(out_dir, name + ".gif")
            make_movie(paths, gif, os.path.join(out_dir, name + ".mp4"), intro=None, intro_frames=0, hold_last=6)
            if os.path.exists(gif):
                paths.append(gif)
        except Exception as exc:                           # noqa: BLE001 - the panels stand without it
            say(f"extract render: no movie ({exc!r})")
    say(f"extract render: {sum(p.endswith('.png') for p in paths)} panels in {disp.steps}")
    return paths
