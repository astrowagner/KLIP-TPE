"""A known companion's position and contrast, with their uncertainties, from a finished run.

The companion is taken out of every frame by a negative injection, and a search finds the separation,
angle and contrast of the injection that leaves the least light where the companion was.  The step
ports the IDL predecessor's ``subtract_mwc758c_tpe.pro``, which fitted MWC 758c this way on
JWST/NIRCam data, and its ``subtract_profile_scan.pro``.

* **The residual.**  Each trial's reduction is read in a fixed circular patch of ``patch_fwhm`` FWHM
  around the starting position.  The patch's pixels, less their mean, give an L2 norm, which is divided
  by the same norm of the reduction without the injection.  0 is a perfect null and 1 is no
  cancellation.  The residual is deterministic, so no trial is drawn again.
* **The search.**  A TPE (:class:`klip_tpe.optimizers.TPE`) proposes ``(rho, pa, contrast)`` in a box
  around the start to maximize ``-log(residual)``, ``n_init`` random trials first.
* **Uncertainty 1, the quadratic.**  A full quadratic in the three parameters, each scaled by its box
  width, is fitted to the best ``best_frac`` of the trials (:func:`quadratic_sigma`).  With ``H`` its
  Hessian and the floor the rms of the fit's residuals, ``sigma = sqrt(2 floor diag(H^-1))`` times the
  width.  That is where the residual has risen by the floor.
* **Uncertainty 2, the profiles.**  Two parameters are held at the best and the third is stepped over
  ``4 nref + 1`` points across 4 times its first sigma, at most half the box.  A parabola per parameter
  (:func:`parabola_sigma`) gives the sigma at the same rise and its own minimum.
* **The recovery check.**  ``n_fakes`` sources at the fitted separation and contrast, at other angles,
  go into the data with the companion taken out at its fit, and each is fitted the same way
  (:func:`recovery_check`).  Their mean offsets are the bias, and their scatter the spread of a fit.
* **Wavelength groups.**  With several partitions, such as ALES's wavelength groups, the position comes
  from all of them combined.  Each group's contrast is then fitted alone with the position held
  (:func:`fit_contrast`), and with fakes its own recovery check fits fake sources' contrasts the same
  way (:func:`recovery_contrast`).  The contrasts with their sigmas are the spectrum.
* **The groups' own positions**, on request (``group_positions``).  Each group also gets a search of
  its own position and contrast in a smaller box around the combined fit, with its profiles and its
  fakes.  The summary gives each group's position, their weighted mean against the combined fit, and
  a chromatic test, the slope of each axis' offset with wavelength and its significance.

The two estimators measure how sharply the residual rises around its minimum, not the speckle noise
that moves the minimum.  On beta Pic b (``scripts/check_betapic_extract.py``) they came out 3 to 20
times smaller than the fake sources' scatter.  The summary therefore quotes the larger of the profile
sigma and the fakes' scatter, per parameter and per partition.

:func:`extract_companion` runs the step on a runner whose annuli are done, resumed from the run's
directory, and :func:`write_extract` writes what it found.  ``klip-tpe extract`` does both from the
command line, and a run made by a script resumes its runner with the script's reducer.

Which configuration
-------------------
The annulus' elected winner reduces, unless another configuration is named.  A negative injection
cancels the companion in every frame before the PSF subtraction, so at the null no reference frame,
ADI's included, holds any of it, and the residual's minimum lies at the companion.  ADI's
self-subtraction still flattens the residual around the null, which widens the sigmas, and the
recovery check measures the bias that remains in the configuration used.  The IDL predecessor fitted
MWC 758c in its production ARDI configuration the same way.

Notes
-----
Separations are in arcsec at the run's plate scale and angles in degrees East of North in the image
frame, as :class:`~klip_tpe.metrics.Source` takes them.  The step does not calibrate the plate scale or
north, which stay with the reduction.  Contrasts are in the injection model's flux unit.  During the
step the run's injection spectrum is set flat, so a fake source and the negative injection go into
every partition at the contrast given.  A source of ``RunConfig.subtract_known`` near the companion is
left out of the reductions, since the step fits it.

The step follows a user's note of 2026-11-20, "a companion parameter-extraction step".
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .companion import _xy
from .metrics import Source, star_center
from .optimizers import TPE, History
from .space import Param, SearchSpace

__all__ = ["ExtractConfig", "patch_mask", "patch_residual", "quadratic_sigma", "parabola_sigma", "fit_companion",
           "fit_contrast", "recovery_check", "recovery_contrast", "extract_companion", "write_extract", "NAMES"]

#: the three searched parameters, in their order in every vector
NAMES = ("rho", "pa", "contrast")

ReduceFn = Callable[[Optional[Sequence[Source]]], np.ndarray]


@dataclass
class ExtractConfig:
    """Settings of a companion extraction (:func:`extract_companion`).

    Attributes
    ----------
    box_rho_fwhm : float, default 0.5
        Half-width of the search box in separation, in FWHM.
    box_pa_fwhm : float, default 0.5
        Half-width of the box in angle, in FWHM along the arc at the starting separation.
    box_rho_as : float or None, default None
        Half-width in separation, in arcsec, in place of ``box_rho_fwhm``.
    box_pa_deg : float or None, default None
        Half-width in angle, in degrees, in place of ``box_pa_fwhm``.
    box_contrast : tuple of 2 floats, default (0.5, 2.0)
        Factors on the starting contrast that bound the box.
    patch_fwhm : float, default 2.0
        Radius of the patch the residual is read in, in FWHM.
    n_init : int, default 100
        Number of random trials before the TPE guides the search.
    n_iter : int, default 1000
        Number of trials in all.
    gamma : float, default 0.15
        Fraction of the trials the TPE takes as good.
    ncand : int, default 24
        Number of candidates the TPE draws per proposal.
    explore_frac : float, default 0.15
        Fraction of the guided trials drawn at random instead.
    best_frac : float, default 0.4
        Fraction of the trials, the best, that the quadratic is fitted to.
    nref : int, default 5
        Size of the profile scans, ``4 nref + 1`` points per parameter.  Below 3 skips them.
    n_fakes : int, default 5
        Number of fake sources of the recovery check, 0 for none.
    fake_iter : int or None, default None
        Number of trials of each fake source's fit, None for ``n_iter``.
    per_group : bool, default True
        Whether each partition's contrast is fitted alone with the position held, when there are several.
    group_points : int, default 11
        Number of points of each partition's coarse contrast scan.
    group_positions : bool, default False
        Whether each partition also gets a fit of its own position and contrast, a chromatic check.
    group_iter : int, default 300
        Number of trials of each partition's position fit.
    group_init : int, default 50
        Number of random trials of each partition's position fit.
    group_box : float, default 0.5
        The partitions' boxes, as a fraction of the combined fit's half-widths in separation and angle.
    group_fake_iter : int or None, default None
        Number of trials of each fake source's fit in a partition, None for ``group_iter``.
    seed : int or None, default None
        Seed of the random draws.
    """

    box_rho_fwhm: float = 0.5
    box_pa_fwhm: float = 0.5
    box_rho_as: Optional[float] = None
    box_pa_deg: Optional[float] = None
    box_contrast: Tuple[float, float] = (0.5, 2.0)
    patch_fwhm: float = 2.0
    n_init: int = 100
    n_iter: int = 1000
    gamma: float = 0.15
    ncand: int = 24
    explore_frac: float = 0.15
    best_frac: float = 0.4
    nref: int = 5
    n_fakes: int = 5
    fake_iter: Optional[int] = None
    per_group: bool = True
    group_points: int = 11
    group_positions: bool = False
    group_iter: int = 300
    group_init: int = 50
    group_box: float = 0.5
    group_fake_iter: Optional[int] = None
    seed: Optional[int] = None

    def check(self) -> None:
        """Check the settings.

        Raises
        ------
        ValueError
            If a setting is out of range.
        """
        lo, hi = (float(v) for v in self.box_contrast)
        if not 0.0 < lo < 1.0 < hi:
            raise ValueError(f"box_contrast {self.box_contrast}: needs 0 < low < 1 < high")
        for name in ("box_rho_fwhm", "box_pa_fwhm", "patch_fwhm"):
            if not float(getattr(self, name)) > 0:
                raise ValueError(f"{name} must be positive")
        for name in ("box_rho_as", "box_pa_deg"):
            v = getattr(self, name)
            if v is not None and not float(v) > 0:
                raise ValueError(f"{name} must be positive or None")
        if int(self.n_iter) < 20 or not 1 <= int(self.n_init) < int(self.n_iter):
            raise ValueError(f"n_iter {self.n_iter} and n_init {self.n_init}: needs n_iter >= 20 and "
                             f"1 <= n_init < n_iter")
        if not 0.0 < float(self.gamma) < 1.0 or not 0.0 <= float(self.explore_frac) < 1.0:
            raise ValueError("gamma must lie in (0, 1) and explore_frac in [0, 1)")
        if not 0.0 < float(self.best_frac) <= 1.0:
            raise ValueError("best_frac must lie in (0, 1]")
        if int(self.n_fakes) < 0 or int(self.group_points) < 5:
            raise ValueError("n_fakes must be at least 0 and group_points at least 5")
        if self.fake_iter is not None and int(self.fake_iter) < 20:
            raise ValueError("fake_iter must be at least 20, or None")
        if int(self.group_iter) < 20 or not 1 <= int(self.group_init) < int(self.group_iter):
            raise ValueError(f"group_iter {self.group_iter} and group_init {self.group_init}: needs group_iter >= 20 "
                             f"and 1 <= group_init < group_iter")
        if not 0.0 < float(self.group_box) <= 1.0:
            raise ValueError("group_box must lie in (0, 1]")
        if self.group_fake_iter is not None and int(self.group_fake_iter) < 20:
            raise ValueError("group_fake_iter must be at least 20, or None")


# ----------------------------------------------------------------------------- the residual
def patch_mask(shape: Tuple[int, int], rho: float, pa: float, radius_px: float, pxscale: float,
               angle_convention: str = "pa") -> np.ndarray:
    """Return the patch a residual is read in, a disk around a position.

    Parameters
    ----------
    shape : tuple of 2 ints
        Shape of the image, ``(ny, nx)``.
    rho : float
        Separation of the center, in arcsec.
    pa : float
        Angle of the center, in degrees.
    radius_px : float
        Radius, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    angle_convention : {"pa", "math"}, default "pa"
        How the reducer reads ``pa``.

    Returns
    -------
    mask : ndarray of bool, shape (ny, nx)
        True inside the patch.

    Raises
    ------
    ValueError
        If the patch holds fewer than 5 pixels.
    """
    x0, y0 = _xy(shape, rho, pa, pxscale, angle_convention)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= float(radius_px) ** 2
    if int(mask.sum()) < 5:
        raise ValueError(f"the patch at rho {rho:g}\", pa {pa:g} deg holds {int(mask.sum())} pixels of the "
                         f"image: check the position and the patch's radius ({radius_px:.2f} px)")
    return mask


def patch_residual(img: np.ndarray, mask: np.ndarray) -> float:
    """Return the L2 norm of an image's finite pixels in a patch, less their mean.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The reduced image.
    mask : ndarray of bool, shape (ny, nx)
        The patch (:func:`patch_mask`).

    Returns
    -------
    norm : float
        ``sqrt(sum((v - mean(v))**2))`` over the patch's finite pixels ``v``, NaN when fewer than 5 are
        finite.
    """
    v = np.asarray(img, float)[mask]
    v = v[np.isfinite(v)]
    if v.size < 5:
        return float("nan")
    v = v - v.mean()
    return float(np.sqrt(np.sum(v * v)))


# ----------------------------------------------------------------------------- the two estimators
def quadratic_sigma(X: np.ndarray, R: np.ndarray, x0: Sequence[float], scale: Sequence[float]
                    ) -> Dict[str, Any]:
    """Fit a full quadratic to residuals and return each parameter's sigma from its curvature.

    It is the IDL predecessor's ``mwcm_sub_quadfit`` for any number of parameters.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        The trials' parameters.
    R : ndarray, shape (n,)
        Their residuals.
    x0 : sequence of float, length d
        Where the parameters are centered, the best trial.
    scale : sequence of float, length d
        Each parameter's scale, the box's width.

    Returns
    -------
    fit : dict
        ``ok``, ``sigma`` (list of d floats, NaN where the fit fails), ``floor`` (the rms of the fit's
        residuals), ``hessian`` (d by d, in scaled units) and ``n``.

    Notes
    -----
    The quadratic is ``R = c + g.u + u^T H u / 2`` in the scaled parameters ``u = (x - x0) / scale``.
    It rises by the floor where ``u^T H u = 2 floor``, so a parameter's marginal sigma is
    ``sqrt(2 floor (H^-1)_ii)`` times its scale.  The fit needs 12 trials or the number of terms plus
    2, whichever is more, and fails when ``H`` cannot be inverted or a diagonal term of its inverse is
    not positive.
    """
    X = np.asarray(X, float)
    R = np.asarray(R, float)
    d = X.shape[1]
    out: Dict[str, Any] = {"ok": False, "sigma": [float("nan")] * d, "floor": float("nan"), "hessian": None,
                           "n": int(len(R))}
    good = np.isfinite(R) & np.all(np.isfinite(X), axis=1)
    X, R = X[good], R[good]
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    nterm = 1 + 2 * d + len(pairs)
    if len(R) < max(12, nterm + 2):
        return out
    sc = np.asarray(scale, float)
    u = (X - np.asarray(x0, float)) / sc
    cols = [np.ones(len(R))] + [u[:, i] for i in range(d)] + [u[:, i] ** 2 for i in range(d)] + \
        [u[:, i] * u[:, j] for i, j in pairs]
    M = np.column_stack(cols)
    cf, *_ = np.linalg.lstsq(M, R, rcond=None)
    floor = float(np.sqrt(np.mean((R - M @ cf) ** 2)))
    H = np.zeros((d, d))
    for i in range(d):
        H[i, i] = 2.0 * cf[1 + d + i]
    for k, (i, j) in enumerate(pairs):
        H[i, j] = H[j, i] = cf[1 + 2 * d + k]
    out.update(floor=floor, hessian=H.tolist())
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return out
    di = np.diag(Hi)
    if not np.all(np.isfinite(di)) or np.min(di) <= 0:
        return out
    out.update(ok=True, sigma=[float(v) for v in np.sqrt(2.0 * floor * di) * sc])
    return out


def parabola_sigma(xv: Sequence[float], yv: Sequence[float], x0: float, scale: float) -> Dict[str, Any]:
    """Fit a parabola to one parameter's residuals and return its minimum and sigma.

    It is the fit of the IDL predecessor's ``mwcm_sub_sigprof``, for a profile scan or for the trials
    near the best.

    Parameters
    ----------
    xv : sequence of float
        The parameter's values.
    yv : sequence of float
        The residuals there.
    x0 : float
        Where the parameter is centered, the best.
    scale : float
        The parameter's scale, the box's width.

    Returns
    -------
    fit : dict
        ``ok``, ``sigma``, ``minimum`` (the parabola's vertex, held within the values), ``floor`` (the rms
        of the fit's residuals) and ``coef`` (the parabola in ``u = (x - x0) / scale``, constant first).

    Notes
    -----
    With ``y = c0 + c1 u + c2 u^2``, the residual rises by the floor at ``u = sqrt(floor / c2)`` from the
    vertex, which times the scale is the sigma.  The fit needs 5 finite points with some spread, and a
    parabola that opens upward.
    """
    xv = np.asarray(xv, float)
    yv = np.asarray(yv, float)
    g = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[g], yv[g]
    out: Dict[str, Any] = {"ok": False, "sigma": float("nan"), "minimum": float(x0), "floor": float("nan"),
                           "coef": None}
    if xv.size < 5 or float(np.ptp(xv)) <= 0:
        return out
    sc = float(scale)
    u = (xv - float(x0)) / sc
    c2, c1, c0 = np.polyfit(u, yv, 2)
    floor = float(np.sqrt(np.mean((yv - np.polyval([c2, c1, c0], u)) ** 2)))
    out.update(floor=floor, coef=[float(c0), float(c1), float(c2)])
    if c2 > 0:
        vertex = float(x0) - c1 / (2.0 * c2) * sc
        out.update(ok=True, sigma=float(np.sqrt(floor / c2) * sc),
                   minimum=float(min(max(vertex, xv.min()), xv.max())))
    return out


# ----------------------------------------------------------------------------- the fits
def _box(rho: float, pa: float, contrast: float, fwhm_px: float, pxscale: float, cfg: ExtractConfig
         ) -> Tuple[np.ndarray, np.ndarray]:
    """Return the search box around a start.

    Parameters
    ----------
    rho : float
        Starting separation, in arcsec.
    pa : float
        Starting angle, in degrees.
    contrast : float
        Starting contrast.
    fwhm_px : float
        FWHM, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    cfg : ExtractConfig
        The settings.

    Returns
    -------
    lo : ndarray, shape (3,)
        The box's low corner, ``(rho, pa, contrast)``.
    hi : ndarray, shape (3,)
        Its high corner.
    """
    fw_as = float(fwhm_px) * float(pxscale)
    drho = float(cfg.box_rho_as) if cfg.box_rho_as is not None else float(cfg.box_rho_fwhm) * fw_as
    dpa = (float(cfg.box_pa_deg) if cfg.box_pa_deg is not None
           else math.degrees(float(cfg.box_pa_fwhm) * fw_as / max(float(rho), 1e-9)))
    lo = np.array([max(float(rho) - drho, 0.05 * float(rho)), float(pa) - dpa, float(contrast) * cfg.box_contrast[0]])
    hi = np.array([float(rho) + drho, float(pa) + dpa, float(contrast) * cfg.box_contrast[1]])
    return lo, hi


def _vec(x: Sequence[float]) -> Dict[str, float]:
    """Return a parameter vector as a dict.

    Parameters
    ----------
    x : sequence of float, length 3
        ``(rho, pa, contrast)``.

    Returns
    -------
    named : dict of str to float
        The values by name, ``pa`` in [0, 360).
    """
    return {"rho": float(x[0]), "pa": float(x[1]) % 360.0, "contrast": float(x[2])}


def fit_companion(reduce_fn: ReduceFn, rho: float, pa: float, contrast: float, *, fwhm_px: float,
                  pxscale: float, angle_convention: str = "pa", config: Optional[ExtractConfig] = None,
                  log: Optional[Callable[[str], None]] = None, log_every: int = 50) -> Dict[str, Any]:
    """Fit a companion's separation, angle and contrast by a TPE search of its negative injection.

    Parameters
    ----------
    reduce_fn : callable
        Function ``reduce_fn(sources) -> image`` that reduces one fixed configuration with the given
        extra injections, None for none.
    rho : float
        Starting separation, in arcsec.
    pa : float
        Starting angle, in degrees.
    contrast : float
        Starting contrast, positive.
    fwhm_px : float
        FWHM, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    angle_convention : {"pa", "math"}, default "pa"
        How the reducer reads ``pa``.
    config : ExtractConfig or None, default None
        The settings, None for the defaults.  Its ``n_fakes`` and ``per_group`` are not used here.
    log : callable or None, default None
        Function that takes a line of progress.
    log_every : int, default 50
        Number of trials between progress lines.

    Returns
    -------
    fit : dict
        The start, the box, the patch, the trials, the best of every reduction, the quadratic's and the
        profiles' sigmas and minima, the near-best trials' parabolas and the number of reductions.

    Raises
    ------
    ValueError
        If the start is not positive, the settings are out of range or the patch holds no light without
        the injection.
    """
    cfg = config or ExtractConfig()
    cfg.check()
    if not (rho > 0 and contrast > 0):
        raise ValueError(f"rho {rho!r} and contrast {contrast!r} must be positive")
    say = log or (lambda s: None)
    t0 = time.time()
    lo, hi = _box(rho, pa, contrast, fwhm_px, pxscale, cfg)
    scale = hi - lo
    clean = np.asarray(reduce_fn(None), float)
    radius = float(cfg.patch_fwhm) * float(fwhm_px)
    mask = patch_mask(clean.shape, rho, pa, radius, pxscale, angle_convention)
    ref = patch_residual(clean, mask)
    if not (np.isfinite(ref) and ref > 0):
        raise ValueError("the patch holds no light in the reduction without the injection: check the start")
    nred = 1
    best_img = clean
    best_r = float("inf")

    def resid(x: np.ndarray) -> Tuple[float, np.ndarray]:
        img = np.asarray(reduce_fn([Source(float(x[0]), float(x[1]), -float(x[2]))]), float)
        return patch_residual(img, mask) / ref, img

    space = SearchSpace([Param(name=n, lo=float(a), hi=float(b)) for n, a, b in zip(NAMES, lo, hi)])
    opt = TPE(space, n_init=int(cfg.n_init), gamma=float(cfg.gamma), ncand=int(cfg.ncand),
              explore_frac=float(cfg.explore_frac))
    hist = History(ndim=3)
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.n_iter)
    X = np.zeros((n, 3))
    R = np.full(n, np.nan)
    phases: List[str] = []
    say(f"  extract: {n} trials ({int(cfg.n_init)} random first) in rho {lo[0]:.4f}-{hi[0]:.4f}\", "
        f"pa {lo[1]:.2f}-{hi[1]:.2f} deg, contrast {lo[2]:.3e}-{hi[2]:.3e}, patch {radius:.2f} px")
    for it in range(n):
        p = opt.ask(hist, rng, 1)[0]
        x = np.clip(np.asarray(p.x, float), lo, hi)
        r, img = resid(x)
        nred += 1
        X[it], R[it] = x, r
        phases.append(str(p.flags.get("phase", "")))
        hist.append(x, -math.log(max(r, 1e-30)) if np.isfinite(r) else float("nan"), dict(p.flags))
        if np.isfinite(r) and r < best_r:
            best_r, best_img = r, img
        if log_every and ((it + 1) % int(log_every) == 0 or it + 1 == n):
            b = int(np.nanargmin(R[:it + 1]))
            say(f"    trial {it + 1}/{n}: best residual {R[b]:.4f} at rho {X[b, 0]:.4f}\", pa {X[b, 1] % 360:.3f} "
                f"deg, contrast {X[b, 2]:.4e}")
    if not np.any(np.isfinite(R)):
        raise ValueError("no trial gave a finite residual: the patch lies outside the reduced zone")
    ib = int(np.nanargmin(R))
    tpe_best = X[ib].copy()
    # uncertainty 1: the quadratic on the best trials
    order = np.argsort(np.where(np.isfinite(R), R, np.inf))
    nb = min(n, max(int(round(float(cfg.best_frac) * n)), 12))
    near = order[:nb]
    quad = quadratic_sigma(X[near], R[near], tpe_best, scale)
    marg = [parabola_sigma(X[near, d], R[near], tpe_best[d], scale[d]) for d in range(3)]
    # uncertainty 2: the profile scans, two parameters held at the best
    gbest, gbest_r = tpe_best.copy(), float(R[ib])
    prof: Dict[str, Any] = {"values": [], "residual": []}
    sig2: List[Dict[str, Any]] = []
    if int(cfg.nref) >= 3:
        npx = 4 * int(cfg.nref) + 1
        sig1 = np.asarray(quad["sigma"], float)
        span = np.where(np.isfinite(sig1) & (sig1 > 0), 4.0 * sig1, 0.05 * scale) if quad["ok"] else 0.05 * scale
        span = np.minimum(span, 0.5 * scale)
        say(f"  extract: profile scans, {3 * npx} reductions ({npx} per parameter)")
        for d in range(3):
            vals = np.clip(tpe_best[d] + span[d] * np.linspace(-1.0, 1.0, npx), lo[d], hi[d])
            rs = []
            for v in vals:
                g = tpe_best.copy()
                g[d] = v
                r, img = resid(g)
                nred += 1
                rs.append(r)
                if np.isfinite(r) and r < gbest_r:
                    gbest, gbest_r = g.copy(), float(r)
                if np.isfinite(r) and r < best_r:
                    best_r, best_img = r, img
            prof["values"].append([float(v) for v in vals])
            prof["residual"].append([float(v) for v in rs])
            sig2.append(parabola_sigma(vals, rs, tpe_best[d], scale[d]))
    secs = time.time() - t0
    return {
        "start": [float(rho), float(pa), float(contrast)], "lo": lo.tolist(), "hi": hi.tolist(),
        "patch_px": radius, "patch_center_px": list(_xy(clean.shape, rho, pa, pxscale, angle_convention)),
        "ref": float(ref), "shape": list(clean.shape),
        "trials": {"x": X.tolist(), "residual": R.tolist(), "phase": phases},
        "tpe_best": dict(_vec(tpe_best), residual=float(R[ib]), trial=ib + 1),
        "best": dict(_vec(gbest), residual=float(gbest_r)),
        "quadratic": quad, "marginal": marg, "profiles": prof, "profile_fits": sig2,
        "n_reductions": int(nred), "seconds": float(secs),
        "images": {"clean": clean, "best": best_img},
    }


def fit_contrast(reduce_fn: ReduceFn, rho: float, pa: float, c0: float, *, fwhm_px: float, pxscale: float,
                 angle_convention: str = "pa", patch_fwhm: float = 2.0, nref: int = 5, n_coarse: int = 11,
                 span: Tuple[float, float] = (0.2, 2.2)) -> Dict[str, Any]:
    """Fit a companion's contrast with its position held, over a coarse scan and a profile.

    Parameters
    ----------
    reduce_fn : callable
        Function ``reduce_fn(sources) -> image`` that reduces one fixed configuration, such as one
        partition, with the given extra injections.
    rho : float
        Separation, in arcsec, held.
    pa : float
        Angle, in degrees, held.
    c0 : float
        Starting contrast, positive.
    fwhm_px : float
        FWHM, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    angle_convention : {"pa", "math"}, default "pa"
        How the reducer reads ``pa``.
    patch_fwhm : float, default 2.0
        Radius of the patch, in FWHM.
    nref : int, default 5
        Size of the profile, ``4 nref + 1`` points.
    n_coarse : int, default 11
        Number of points of the coarse scan.
    span : tuple of 2 floats, default (0.2, 2.2)
        The coarse scan's range, as factors on ``c0``.  It moves once by its width when its minimum
        falls on an edge.

    Returns
    -------
    fit : dict
        ``contrast`` (the profile parabola's minimum, or the best point when the parabola fails),
        ``sigma``, ``ok``, ``floor``, ``best`` (the lowest point), ``points`` (the contrasts and their
        residuals), ``ref`` and ``n_reductions``.

    Raises
    ------
    ValueError
        If ``c0`` is not positive or the patch holds no light without the injection.
    """
    if not c0 > 0:
        raise ValueError(f"c0 {c0!r} must be positive")
    clean = np.asarray(reduce_fn(None), float)
    mask = patch_mask(clean.shape, rho, pa, float(patch_fwhm) * float(fwhm_px), pxscale, angle_convention)
    ref = patch_residual(clean, mask)
    if not (np.isfinite(ref) and ref > 0):
        raise ValueError("the patch holds no light in the reduction without the injection")
    pts: Dict[float, float] = {}

    def at(c: float) -> float:
        if c not in pts:
            img = np.asarray(reduce_fn([Source(float(rho), float(pa), -float(c))]), float)
            pts[c] = patch_residual(img, mask) / ref
        return pts[c]

    a, b = float(span[0]) * c0, float(span[1]) * c0
    for _ in range(2):
        cs = np.linspace(a, b, int(n_coarse))
        rs = np.array([at(float(c)) for c in cs])
        i = int(np.nanargmin(rs))
        if 0 < i < len(cs) - 1:
            break
        w = b - a
        a, b = (max(a - w, 0.0), a) if i == 0 else (b, b + w)
        if b <= 0:
            break
    c_all = np.array(sorted(pts))
    r_all = np.array([pts[c] for c in c_all])
    ib = int(np.nanargmin(r_all))
    width = float(c_all.max() - c_all.min()) or c0
    k = np.argsort(np.abs(c_all - c_all[ib]))[:5]
    first = parabola_sigma(c_all[k], r_all[k], c_all[ib], width)
    s0 = first["sigma"] if first["ok"] and first["sigma"] > 0 else 0.1 * c_all[ib]
    half = min(4.0 * s0, 0.5 * width)
    npx = 4 * int(nref) + 1
    vals = np.clip(c_all[ib] + half * np.linspace(-1.0, 1.0, npx), 1e-12 * c0, None)
    rs = [at(float(v)) for v in vals]
    prof = parabola_sigma(vals, rs, c_all[ib], width)
    c_all = np.array(sorted(pts))
    r_all = np.array([pts[c] for c in c_all])
    ib = int(np.nanargmin(r_all))
    return {"contrast": float(prof["minimum"] if prof["ok"] else c_all[ib]), "sigma": float(prof["sigma"]),
            "ok": bool(prof["ok"]), "floor": float(prof["floor"]), "best": float(c_all[ib]),
            "best_residual": float(r_all[ib]), "points": {"contrast": c_all.tolist(), "residual": r_all.tolist()},
            "profile": {"contrast": [float(v) for v in vals], "residual": [float(v) for v in rs]},
            "ref": float(ref), "n_reductions": int(len(pts) + 1)}


def recovery_check(reduce_fn: ReduceFn, fit: Dict[str, Any], *, fwhm_px: float, pxscale: float,
                   angle_convention: str = "pa", config: Optional[ExtractConfig] = None,
                   log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Fit fake sources like the companion and return the bias and spread of the fits.

    Each fake source goes in at the companion's fitted separation and contrast, at angles spread evenly
    from the companion's, with the companion taken out at its fit.  Its search starts as far from it as
    the companion's start was from the companion's fit, and has ``fake_iter`` trials and no profiles.

    Parameters
    ----------
    reduce_fn : callable
        Function ``reduce_fn(sources) -> image``, as :func:`fit_companion` takes it.
    fit : dict
        The companion's fit, :func:`fit_companion`'s.
    fwhm_px : float
        FWHM, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    angle_convention : {"pa", "math"}, default "pa"
        How the reducer reads angles.
    config : ExtractConfig or None, default None
        The settings, None for the defaults.
    log : callable or None, default None
        Function that takes a line of progress.

    Returns
    -------
    check : dict
        ``fakes`` (each one's true and fitted values), and per parameter the ``bias`` (mean offset, the
        contrast's as a fraction), its standard error ``bias_se`` and the ``scatter`` (standard
        deviation).  Empty lists and NaN without fakes.
    """
    cfg = config or ExtractConfig()
    say = log or (lambda s: None)
    nf = int(cfg.n_fakes)
    b = fit["best"]
    rho_b, pa_b, c_b = float(b["rho"]), float(b["pa"]), float(b["contrast"])
    s = fit["start"]
    d_rho, d_pa, f_c = s[0] - rho_b, ((s[1] - pa_b + 180.0) % 360.0) - 180.0, s[2] / c_b
    comp = Source(rho_b, pa_b, -c_b)
    n_it = int(cfg.fake_iter or cfg.n_iter)
    sub = dataclasses.replace(cfg, n_iter=n_it, n_init=min(int(cfg.n_init), max(10, n_it // 10)), nref=0,
                              n_fakes=0, seed=None if cfg.seed is None else int(cfg.seed) + 1)
    fakes: List[Dict[str, Any]] = []
    for k in range(1, nf + 1):
        pa_k = (pa_b + 360.0 * k / (nf + 1)) % 360.0
        true = Source(rho_b, pa_k, c_b)
        say(f"  extract: recovery check, fake source {k} of {nf} at pa {pa_k:.2f} deg")

        def rf(srcs, true=true):
            return reduce_fn([true, comp] + list(srcs or []))

        sub_k = dataclasses.replace(sub, seed=None if sub.seed is None else sub.seed + k)
        f = fit_companion(rf, rho_b + d_rho, pa_k + d_pa, c_b * f_c, fwhm_px=fwhm_px, pxscale=pxscale,
                          angle_convention=angle_convention, config=sub_k, log=None)
        fb = f["best"]
        fakes.append({"true": _vec([rho_b, pa_k, c_b]), "fit": {k2: float(fb[k2]) for k2 in NAMES},
                      "residual": float(fb["residual"]), "n_reductions": int(f["n_reductions"])})
    out: Dict[str, Any] = {"fakes": fakes, "bias": {}, "bias_se": {}, "scatter": {}}
    for name in NAMES:
        if not fakes:
            out["bias"][name] = out["bias_se"][name] = out["scatter"][name] = float("nan")
            continue
        if name == "pa":
            dv = np.array([((f["fit"]["pa"] - f["true"]["pa"] + 180.0) % 360.0) - 180.0 for f in fakes])
        elif name == "contrast":
            dv = np.array([f["fit"]["contrast"] / f["true"]["contrast"] - 1.0 for f in fakes])
        else:
            dv = np.array([f["fit"][name] - f["true"][name] for f in fakes])
        sd = float(np.std(dv, ddof=1)) if len(dv) > 1 else float("nan")
        out["bias"][name] = float(np.mean(dv))
        out["bias_se"][name] = sd / math.sqrt(len(dv)) if len(dv) > 1 else float("nan")
        out["scatter"][name] = sd
    out["n_reductions"] = int(sum(f["n_reductions"] for f in fakes))
    return out


def recovery_contrast(reduce_fn: ReduceFn, rho: float, pa: float, contrast: float, *, fwhm_px: float,
                      pxscale: float, angle_convention: str = "pa", patch_fwhm: float = 2.0, n_fakes: int = 5,
                      nref: int = 3, n_coarse: int = 11) -> Dict[str, Any]:
    """Fit fake sources' contrasts like a companion's, with their positions held, and return the bias and spread.

    Each fake source goes in at the companion's separation and contrast, at angles spread evenly from the
    companion's, with the companion taken out at its fit, and :func:`fit_contrast` fits it.

    Parameters
    ----------
    reduce_fn : callable
        Function ``reduce_fn(sources) -> image``, such as one partition's.
    rho : float
        The companion's separation, in arcsec.
    pa : float
        Its angle, in degrees.
    contrast : float
        Its fitted contrast, positive.
    fwhm_px : float
        FWHM, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    angle_convention : {"pa", "math"}, default "pa"
        How the reducer reads angles.
    patch_fwhm : float, default 2.0
        Radius of the patch, in FWHM.
    n_fakes : int, default 5
        Number of fake sources.
    nref : int, default 3
        Size of each fit's profile, ``4 nref + 1`` points.
    n_coarse : int, default 11
        Number of points of each fit's coarse scan.

    Returns
    -------
    check : dict
        ``fakes`` (each one's angle and true and fitted contrasts), ``bias`` (the mean of fitted over true
        less 1), its standard error ``bias_se``, ``scatter`` (the standard deviation of that ratio) and
        ``n_reductions``.
    """
    comp = Source(float(rho), float(pa), -float(contrast))
    fakes: List[Dict[str, Any]] = []
    for k in range(1, int(n_fakes) + 1):
        pa_k = (float(pa) + 360.0 * k / (int(n_fakes) + 1)) % 360.0
        true = Source(float(rho), pa_k, float(contrast))

        def rf(srcs, true=true):
            return reduce_fn([true, comp] + list(srcs or []))

        g = fit_contrast(rf, rho, pa_k, contrast, fwhm_px=fwhm_px, pxscale=pxscale,
                         angle_convention=angle_convention, patch_fwhm=patch_fwhm, nref=nref, n_coarse=n_coarse)
        fakes.append({"pa": pa_k, "true": float(contrast), "fit": float(g["contrast"]),
                      "n_reductions": int(g["n_reductions"])})
    dv = np.array([f["fit"] / f["true"] - 1.0 for f in fakes])
    sd = float(np.std(dv, ddof=1)) if len(dv) > 1 else float("nan")
    return {"fakes": fakes, "bias": float(np.mean(dv)) if len(dv) else float("nan"),
            "bias_se": sd / math.sqrt(len(dv)) if len(dv) > 1 else float("nan"), "scatter": sd,
            "n_reductions": int(sum(f["n_reductions"] for f in fakes))}


# ----------------------------------------------------------------------------- on a finished run
def _annulus_for(runner, rho: float, annulus: Optional[int]):
    """Return the result of the annulus whose winner reduces.

    Parameters
    ----------
    runner : Runner
        The finished run's runner.
    rho : float
        The companion's separation, in arcsec.
    annulus : int or None
        The annulus, counted from 0, None for the one that holds ``rho``.

    Returns
    -------
    result : AnnulusResult
        The annulus' result.

    Raises
    ------
    ValueError
        If the run has no finished annulus, or the one asked for is not finished.
    """
    res = [r for r in (runner.results or []) if r is not None]
    if not res:
        raise ValueError("the run has no finished annulus")
    if annulus is not None:
        for r in res:
            if int(r.annulus) == int(annulus):
                return r
        raise ValueError(f"annulus {int(annulus) + 1} is not finished in this run")
    rpx = float(rho) / float(runner.pxscale)
    inside = [r for r in res if float(r.inrad) <= rpx < float(r.outrad)]
    if inside:
        return inside[0]
    return min(res, key=lambda r: abs(0.5 * (float(r.inrad) + float(r.outrad)) - rpx))


def _record_config(run_dir: str, annulus: int, evaluation: int) -> Dict[str, Any]:
    """Return the configuration of one evaluation of a run, from its ``results.jsonl``.

    Parameters
    ----------
    run_dir : str
        The run's directory.
    annulus : int
        The annulus, counted from 0.
    evaluation : int
        The evaluation's number in the annulus, counted from 1 as the run's log counts it.

    Returns
    -------
    config : dict
        The configuration, in :meth:`klip_tpe.space.Config.to_dict` form.

    Raises
    ------
    ValueError
        If the run holds no such evaluation.
    """
    path = os.path.join(run_dir, "results.jsonl")
    found = None
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if int(rec.get("annulus", -1)) == int(annulus) and int(rec.get("index", -1)) == int(evaluation) - 1:
                found = rec.get("config")
    if not found:
        raise ValueError(f"annulus {int(annulus) + 1} has no evaluation {int(evaluation)} in {path}")
    return found


def extract_companion(runner, rho: float, pa: float, contrast: float, *, annulus: Optional[int] = None,
                      evaluation: Optional[int] = None, config: Optional[ExtractConfig] = None,
                      log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Extract a known companion's position and contrast, with uncertainties, from a finished run.

    Parameters
    ----------
    runner : Runner
        The finished run's runner, resumed from its directory (:meth:`klip_tpe.runner.Runner.resume`).
    rho : float
        Starting separation, in arcsec.
    pa : float
        Starting angle, in degrees East of North.
    contrast : float
        Starting contrast, positive.
    annulus : int or None, default None
        The annulus whose configuration reduces, counted from 0, None for the one that holds ``rho``.
    evaluation : int or None, default None
        The evaluation of that annulus whose configuration reduces, counted from 1, None for the
        annulus' elected winner.
    config : ExtractConfig or None, default None
        The settings, None for the defaults.
    log : callable or None, default None
        Function that takes a line of progress, None for the runner's log.

    Returns
    -------
    found : dict
        The fit (:func:`fit_companion`), ``groups`` (each partition's :func:`fit_contrast` with its
        wavelength, empty with one), ``recovery`` (:func:`recovery_check`, None without fakes), the
        configuration used and the run's geometry.  :func:`write_extract` writes it.

    Raises
    ------
    ValueError
        If the settings or the start are out of range, or the annulus or evaluation is not in the run.
    """
    cfg = config or ExtractConfig()
    cfg.check()
    say = log or getattr(runner, "log", None) or print
    res = _annulus_for(runner, rho, annulus)
    if evaluation is None:
        cdict, what = res.winner_config, f"the elected winner (eval {int(res.winner_index) + 1})"
    else:
        cdict, what = _record_config(runner.run_dir, int(res.annulus), int(evaluation)), f"eval {int(evaluation)}"
    ccfg = runner._config_from_dict(cdict)
    red = runner.reducer
    fwhm, pxs = float(runner.fwhm), float(runner.pxscale)
    conv = str(getattr(red, "angle_convention", "pa"))
    known = list(runner.cfg.subtract_known or [])
    shape_any = (int(2 * float(res.outrad) + 8),) * 2
    x0, y0 = _xy(shape_any, rho, pa, pxs, conv)
    near = [s for s in known if math.hypot(*(np.subtract(_xy(shape_any, float(s[0]), float(s[1]), pxs, conv),
                                                         (x0, y0)))) <= 2.0 * fwhm]
    keep = {"ia": runner.ia, "known": known, "scale": getattr(red, "injection_scale", None)}
    say(f"extract: annulus {int(res.annulus) + 1} ({float(res.inrad):.1f}-{float(res.outrad):.1f} px), {what}, "
        f"start rho {rho:.4f}\", pa {pa:.3f} deg, contrast {contrast:.4e}")
    if near:
        say(f"  extract: {len(near)} source(s) of subtract_known near the start left out, since the step fits it")
    selected = list(getattr(ccfg, "selected", None) or [])

    def reduce_fn(srcs, sel=None):
        c = ccfg if sel is None else dataclasses.replace(ccfg, selected=list(sel))
        return runner._reduce(c, list(srcs) if srcs else None, tag="extract").image

    try:
        runner.ia = int(res.annulus)
        runner.cfg.subtract_known = [s for s in known if s not in near]
        if keep["scale"] is not None:
            red.injection_scale = None
        fit = fit_companion(reduce_fn, rho, pa, contrast, fwhm_px=fwhm, pxscale=pxs, angle_convention=conv,
                            config=cfg, log=say)
        b = fit["best"]
        say(f"  extract: best rho {b['rho']:.5f}\", pa {b['pa']:.4f} deg, contrast {b['contrast']:.4e}, "
            f"residual {b['residual']:.4f}, after {fit['n_reductions']} reductions")
        groups: List[Dict[str, Any]] = []
        parts = getattr(red, "reducers", None)
        if cfg.per_group and parts and len(selected) > 1:
            say(f"  extract: each of {len(selected)} partitions' contrast with the position held")
            for pid in selected:
                g = fit_contrast(lambda s, pid=pid: reduce_fn(s, sel=[pid]), b["rho"], b["pa"], b["contrast"],
                                 fwhm_px=fwhm, pxscale=pxs, angle_convention=conv, patch_fwhm=cfg.patch_fwhm,
                                 nref=max(int(cfg.nref), 3), n_coarse=int(cfg.group_points))
                lam = getattr(parts.get(pid), "lam_m", None) if hasattr(parts, "get") else None
                g.update(partition=str(pid), lam_um=None if lam is None else float(lam) * 1e6)
                groups.append(g)
                say(f"    {pid}: contrast {g['contrast']:.4e} +- {g['sigma']:.2e}")
        rec = None
        if int(cfg.n_fakes) > 0:
            rec = recovery_check(reduce_fn, fit, fwhm_px=fwhm, pxscale=pxs, angle_convention=conv, config=cfg,
                                 log=say)
            for g in groups:
                pid = next(p for p in selected if str(p) == g["partition"])
                say(f"  extract: recovery check of {pid}'s contrast, {int(cfg.n_fakes)} fake sources")
                g["recovery"] = recovery_contrast(lambda s, pid=pid: reduce_fn(s, sel=[pid]), b["rho"], b["pa"],
                                                  g["contrast"], fwhm_px=fwhm, pxscale=pxs, angle_convention=conv,
                                                  patch_fwhm=cfg.patch_fwhm, n_fakes=int(cfg.n_fakes),
                                                  n_coarse=int(cfg.group_points))
        if cfg.group_positions and groups:
            lo, hi = fit["lo"], fit["hi"]
            half_rho = 0.5 * (hi[0] - lo[0]) * float(cfg.group_box)
            half_pa = 0.5 * (hi[1] - lo[1]) * float(cfg.group_box)
            say(f"  extract: each of {len(groups)} partitions' own position, {int(cfg.group_iter)} trials each")
            for k, g in enumerate(groups):
                pid = next(p for p in selected if str(p) == g["partition"])
                sub_cfg = dataclasses.replace(cfg, box_rho_as=half_rho, box_pa_deg=half_pa, n_iter=int(cfg.group_iter),
                                              n_init=int(cfg.group_init), n_fakes=int(cfg.n_fakes),
                                              fake_iter=int(cfg.group_fake_iter or cfg.group_iter),
                                              seed=None if cfg.seed is None else int(cfg.seed) + 101 + k)
                red_g = (lambda s, pid=pid: reduce_fn(s, sel=[pid]))
                fg = fit_companion(red_g, b["rho"], b["pa"], max(float(g["contrast"]), 1e-12), fwhm_px=fwhm,
                                   pxscale=pxs, angle_convention=conv, config=sub_cfg, log=None)
                fg.pop("images", None)
                rg = (recovery_check(red_g, fg, fwhm_px=fwhm, pxscale=pxs, angle_convention=conv, config=sub_cfg)
                      if int(cfg.n_fakes) > 0 else None)
                g["position"] = {"fit": fg, "recovery": rg}
                bb = fg["best"]
                say(f"    {pid}: rho {bb['rho']:.5f}\", pa {bb['pa']:.4f} deg, contrast {bb['contrast']:.4e}")
    finally:
        runner.ia = keep["ia"]
        runner.cfg.subtract_known = keep["known"]
        if keep["scale"] is not None:
            red.injection_scale = keep["scale"]
    return {"fit": fit, "groups": groups, "recovery": rec, "config": dataclasses.asdict(cfg),
            "configuration": {"what": what, "annulus": int(res.annulus), "inrad": float(res.inrad),
                              "outrad": float(res.outrad), "params": dict(ccfg.params),
                              "selected": [str(s) for s in selected]},
            "run_dir": os.path.abspath(runner.run_dir), "fwhm_px": fwhm, "pxscale": pxs,
            "angle_convention": conv, "left_out": [list(map(float, s)) for s in near]}


# ----------------------------------------------------------------------------- outputs
def _clean_json(v: Any) -> Any:
    """Return a value with NaN and infinity as None and arrays as lists, for JSON.

    Parameters
    ----------
    v : object
        The value.

    Returns
    -------
    clean : object
        The value made fit for :func:`json.dump`.
    """
    if isinstance(v, dict):
        return {str(k): _clean_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_json(x) for x in v]
    if isinstance(v, np.ndarray):
        return _clean_json(v.tolist())
    if isinstance(v, (np.floating, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def _position(found: Dict[str, Any], x: Dict[str, float]) -> Dict[str, float]:
    """Return a fitted position in pixels relative to the star.

    Parameters
    ----------
    found : dict
        :func:`extract_companion`'s result.
    x : dict
        The position, with ``rho`` and ``pa``.

    Returns
    -------
    pos : dict of str to float
        ``dx_px``, ``dy_px`` and ``rho_px``.
    """
    shape = tuple(found["fit"]["shape"])
    cx, cy = star_center(shape)
    px, py = _xy(shape, x["rho"], x["pa"], found["pxscale"], found["angle_convention"])
    return {"dx_px": float(px - cx), "dy_px": float(py - cy), "rho_px": float(x["rho"]) / float(found["pxscale"])}


def _xy_sigma(found: Dict[str, Any], x: Dict[str, float], s_rho: float, s_pa: float) -> Tuple[float, float]:
    """Return a position's sigmas in pixels along the image's axes, from its sigmas in separation and angle.

    Parameters
    ----------
    found : dict
        :func:`extract_companion`'s result.
    x : dict
        The position, with ``rho`` and ``pa``.
    s_rho : float
        Its sigma in separation, in arcsec.
    s_pa : float
        Its sigma in angle, in degrees.

    Returns
    -------
    s_dx : float
        The sigma along x, in pixels.
    s_dy : float
        The sigma along y, in pixels.
    """
    base = _position(found, x)
    out = []
    for key in ("dx_px", "dy_px"):
        terms = []
        for dr, dp in ((s_rho, 0.0), (0.0, s_pa)):
            if not np.isfinite(dr) or not np.isfinite(dp):
                terms.append(np.nan)
                continue
            moved = _position(found, {"rho": x["rho"] + dr, "pa": x["pa"] + dp})
            terms.append(moved[key] - base[key])
        out.append(float(np.sqrt(np.sum(np.square(terms)))))
    return out[0], out[1]


def _chromatic(lam: Sequence[float], v: Sequence[float], s: Sequence[float]) -> Dict[str, Any]:
    """Return a weighted straight line of a quantity against wavelength, and the test of a constant.

    Parameters
    ----------
    lam : sequence of float
        The wavelengths, in microns.
    v : sequence of float
        The values.
    s : sequence of float
        Their sigmas.

    Returns
    -------
    line : dict
        ``mean`` and ``mean_se`` (the weighted mean), ``slope`` and ``slope_se`` (per micron), ``z`` (the
        slope over its error), ``chi2`` and ``dof`` of the constant and ``p_const`` (its probability), NaN
        with fewer than 2 points that have sigmas.
    """
    lam, v, s = (np.asarray(a, float) for a in (lam, v, s))
    g = np.isfinite(lam) & np.isfinite(v) & np.isfinite(s) & (s > 0)
    out = {k: float("nan") for k in ("mean", "mean_se", "slope", "slope_se", "z", "chi2", "p_const")}
    out["dof"] = int(max(g.sum() - 1, 0))
    if g.sum() < 2:
        return out
    w = 1.0 / s[g] ** 2
    mean = float(np.sum(w * v[g]) / np.sum(w))
    l0 = float(np.sum(w * lam[g]) / np.sum(w))
    sxx = float(np.sum(w * (lam[g] - l0) ** 2))
    slope = float(np.sum(w * (lam[g] - l0) * (v[g] - mean)) / sxx) if sxx > 0 else float("nan")
    chi2 = float(np.sum(w * (v[g] - mean) ** 2))
    from scipy.stats import chi2 as _chi2
    out.update(mean=mean, mean_se=float(1.0 / np.sqrt(np.sum(w))), slope=slope,
               slope_se=float(1.0 / np.sqrt(sxx)) if sxx > 0 else float("nan"),
               chi2=chi2, p_const=float(_chi2.sf(chi2, out["dof"])))
    out["z"] = out["slope"] / out["slope_se"] if np.isfinite(out["slope_se"]) else float("nan")
    return out


def _group_positions(found: Dict[str, Any], best: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Return the partitions' own positions, their weighted mean against the combined fit, and the chromatic test.

    Parameters
    ----------
    found : dict
        :func:`extract_companion`'s result.
    best : dict
        The combined fit's best values, with ``dx_px`` and ``dy_px``.

    Returns
    -------
    positions : dict or None
        ``rows`` (per partition the best values, dx and dy, the profile sigmas, the fakes' scatter and the
        sigmas to quote), ``mean`` (the weighted mean of dx and dy, and its offset from the combined fit in
        pixels and in sigma) and ``chromatic`` (:func:`_chromatic` of dx and dy), None without the
        partitions' position fits.
    """
    rows = []
    for g in found["groups"]:
        pos = g.get("position")
        if not pos:
            continue
        fg, rg = pos["fit"], pos.get("recovery")
        b = dict(fg["best"])
        b.update(_position(found, b))
        prof = [p["sigma"] for p in fg["profile_fits"]] or list(fg["quadratic"]["sigma"])
        sc = ([rg["scatter"]["rho"], rg["scatter"]["pa"], rg["scatter"]["contrast"] * b["contrast"]]
              if rg and rg.get("fakes") else [float("nan")] * 3)
        q = [float(np.nanmax([a, c])) if np.isfinite(a) or np.isfinite(c) else float("nan") for a, c in zip(prof, sc)]
        sdx, sdy = _xy_sigma(found, b, q[0], q[1])
        rows.append({"partition": g["partition"], "lam_um": g["lam_um"], "rho": b["rho"], "pa": b["pa"],
                     "contrast": b["contrast"], "dx_px": b["dx_px"], "dy_px": b["dy_px"],
                     "sigma_profile": dict(zip(NAMES, prof)), "scatter": dict(zip(NAMES, sc)),
                     "sigma_quoted": dict(zip(NAMES, q)), "sigma_dx_px": sdx, "sigma_dy_px": sdy,
                     "residual": b["residual"], "n_reductions": int(fg["n_reductions"]) + int((rg or {}).get(
                         "n_reductions", 0))})
    if not rows:
        return None
    lam = [r["lam_um"] if r["lam_um"] is not None else float(k) for k, r in enumerate(rows)]
    cx = _chromatic(lam, [r["dx_px"] for r in rows], [r["sigma_dx_px"] for r in rows])
    cy = _chromatic(lam, [r["dy_px"] for r in rows], [r["sigma_dy_px"] for r in rows])
    mean = {"dx_px": cx["mean"], "dx_se": cx["mean_se"], "dy_px": cy["mean"], "dy_se": cy["mean_se"]}
    mean["ddx_px"] = mean["dx_px"] - best["dx_px"]
    mean["ddy_px"] = mean["dy_px"] - best["dy_px"]
    mean["ddx_z"] = mean["ddx_px"] / mean["dx_se"] if mean["dx_se"] and np.isfinite(mean["dx_se"]) else float("nan")
    mean["ddy_z"] = mean["ddy_px"] / mean["dy_se"] if mean["dy_se"] and np.isfinite(mean["dy_se"]) else float("nan")
    return {"rows": rows, "mean": mean, "chromatic": {"dx": cx, "dy": cy},
            "wavelength": "um" if all(r["lam_um"] is not None for r in rows) else "index"}


def _summary(found: Dict[str, Any]) -> Dict[str, Any]:
    """Return the numbers an extraction is reported with.

    Parameters
    ----------
    found : dict
        :func:`extract_companion`'s result.

    Returns
    -------
    summary : dict
        The best values with the position in pixels, both sets of sigmas in arcsec, degrees and pixels,
        the profiles' minima, the residual floors, the reductions, the partitions' contrasts and the
        recovery check.  ``sigma_quoted`` is the larger of the profile sigma, or the quadratic's without
        profiles, and the fake sources' scatter, per parameter and per partition.
    """
    fit = found["fit"]
    best = dict(fit["best"])
    best.update(_position(found, best))
    pxs = float(found["pxscale"])

    def sig(vals):
        s = {n: float(v) for n, v in zip(NAMES, vals)}
        s["rho_px"] = s["rho"] / pxs
        s["arc_px"] = best["rho_px"] * math.radians(s["pa"])
        s["contrast_frac"] = s["contrast"] / best["contrast"]
        return s

    quad = fit["quadratic"]
    prof = fit["profile_fits"]
    rec = found["recovery"]
    curv = [p["sigma"] for p in prof] if prof else list(quad["sigma"])
    if rec is not None and rec["fakes"]:
        spread = [rec["scatter"]["rho"], rec["scatter"]["pa"], rec["scatter"]["contrast"] * best["contrast"]]
    else:
        spread = [float("nan")] * 3
    quoted = [float(np.nanmax([a, b])) if np.isfinite(a) or np.isfinite(b) else float("nan")
              for a, b in zip(curv, spread)]
    groups = []
    for g in found["groups"]:
        e = {k: g[k] for k in ("partition", "lam_um", "contrast", "sigma", "ok", "floor", "best")}
        gr = g.get("recovery")
        e["bias"] = None if gr is None else gr["bias"]
        e["scatter"] = None if gr is None else gr["scatter"]
        sc = float("nan") if gr is None else gr["scatter"] * g["contrast"]
        e["sigma_quoted"] = float(np.nanmax([g["sigma"], sc])) if np.isfinite(g["sigma"]) or np.isfinite(sc) \
            else float("nan")
        groups.append(e)
    out = {
        "best": best, "tpe_best": fit["tpe_best"], "start": dict(zip(NAMES, fit["start"])),
        "box": {"lo": dict(zip(NAMES, fit["lo"])), "hi": dict(zip(NAMES, fit["hi"]))},
        "patch_px": fit["patch_px"], "patch_fwhm": found["config"]["patch_fwhm"],
        "sigma_quadratic": sig(quad["sigma"]) if quad["ok"] else None, "floor_quadratic": quad["floor"],
        "sigma_profile": sig([p["sigma"] for p in prof]) if prof else None,
        "minimum_profile": dict(zip(NAMES, [p["minimum"] for p in prof])) if prof else None,
        "floor_profile": dict(zip(NAMES, [p["floor"] for p in prof])) if prof else None,
        "sigma_near_best": sig([m["sigma"] for m in fit["marginal"]]),
        "minimum_near_best": dict(zip(NAMES, [m["minimum"] for m in fit["marginal"]])),
        "sigma_quoted": sig(quoted),
        "n_reductions": int(fit["n_reductions"] + sum(g["n_reductions"] for g in found["groups"])
                            + sum((g.get("recovery") or {}).get("n_reductions", 0) for g in found["groups"])
                            + sum(int(g["position"]["fit"]["n_reductions"])
                                  + int((g["position"].get("recovery") or {}).get("n_reductions", 0))
                                  for g in found["groups"] if g.get("position"))
                            + (found["recovery"] or {}).get("n_reductions", 0)),
        "n_trials": len(fit["trials"]["residual"]), "seconds": fit["seconds"],
        "groups": groups,
        "group_positions": _group_positions(found, best),
        "recovery": None if found["recovery"] is None else {k: found["recovery"][k] for k in
                                                           ("bias", "bias_se", "scatter", "fakes")},
        "configuration": found["configuration"], "run_dir": found["run_dir"], "pxscale": pxs,
        "fwhm_px": found["fwhm_px"], "angle_convention": found["angle_convention"],
        "subtract_sources": f"{best['rho']:.5f} {best['pa']:.4f} {best['contrast']:.6e}",
        "settings": found["config"], "left_out": found["left_out"],
    }
    return out


def _txt(s: Dict[str, Any], label: str) -> str:
    """Return an extraction's summary as text.

    Parameters
    ----------
    s : dict
        :func:`_summary`'s numbers.
    label : str
        What was extracted, such as the companion's name.

    Returns
    -------
    text : str
        The summary, a line per item.
    """
    b, c = s["best"], s["configuration"]
    L = [f"# klip-tpe extract: {label}, {time.strftime('%Y-%m-%d %H:%M')}",
         f"run              : {s['run_dir']}",
         f"configuration    : annulus {c['annulus'] + 1} ({c['inrad']:.1f}-{c['outrad']:.1f} px), {c['what']}",
         f"start            : rho {s['start']['rho']:.5f}\"  pa {s['start']['pa']:.3f} deg  contrast "
         f"{s['start']['contrast']:.4e}",
         f"box              : rho {s['box']['lo']['rho']:.4f}-{s['box']['hi']['rho']:.4f}\"  pa "
         f"{s['box']['lo']['pa']:.2f}-{s['box']['hi']['pa']:.2f} deg  contrast {s['box']['lo']['contrast']:.3e}-"
         f"{s['box']['hi']['contrast']:.3e}",
         f"patch            : radius {s['patch_px']:.2f} px ({s['patch_fwhm']:g} FWHM of {s['fwhm_px']:.2f} px)",
         f"reductions       : {s['n_reductions']} ({s['n_trials']} search trials, {s['seconds'] / 60:.1f} min for "
         f"the fit)",
         f"best             : rho {b['rho']:.5f}\" ({b['rho_px']:.3f} px)  pa {b['pa']:.4f} deg  contrast "
         f"{b['contrast']:.4e}  residual {b['residual']:.4f}",
         f"position         : dx {b['dx_px']:+.3f} px  dy {b['dy_px']:+.3f} px from the star, at "
         f"{s['pxscale']:g}\"/px, angles '{s['angle_convention']}'",
         "1-sigma, where the residual has risen by its floor:"]

    def row(name, sg, floor):
        if not sg:
            return f"  {name}: failed"
        return (f"  {name}: rho {sg['rho']:.2e}\" ({sg['rho_px']:.3f} px)  pa {sg['pa']:.3f} deg "
                f"({sg['arc_px']:.3f} px)  contrast {sg['contrast']:.2e} ({100 * sg['contrast_frac']:.1f} %)"
                + ("" if floor is None else f"  floor {floor:.2e}"))

    fq = s["floor_quadratic"]
    L.append(row("#1 quadratic   ", s["sigma_quadratic"], fq if fq == fq else None))
    if s["sigma_profile"] is not None:
        L.append(row("#2 profiles    ", s["sigma_profile"], None))
        m = s["minimum_profile"]
        L.append(f"profile minima   : rho {m['rho']:.5f}\"  pa {m['pa'] % 360:.4f} deg  contrast {m['contrast']:.4e}")
    rec = s["recovery"]
    q = s["sigma_quoted"]
    L.append(f"to quote         : rho {q['rho']:.2e}\" ({q['rho_px']:.3f} px)  pa {q['pa']:.3f} deg ({q['arc_px']:.3f} px)  "
             f"contrast {q['contrast']:.2e} ({100 * q['contrast_frac']:.1f} %), the larger of the profile sigma "
             f"and the fakes' scatter")
    if rec is not None and rec["fakes"]:
        bi, se, sc = rec["bias"], rec["bias_se"], rec["scatter"]
        L += [f"recovery check   : {len(rec['fakes'])} fake sources at the fit's separation and contrast",
              f"  bias           : rho {bi['rho']:+.2e} +- {se['rho']:.1e}\"  pa {bi['pa']:+.3f} +- {se['pa']:.3f} deg  "
              f"contrast {100 * bi['contrast']:+.1f} +- {100 * se['contrast']:.1f} %",
              f"  scatter        : rho {sc['rho']:.2e}\"  pa {sc['pa']:.3f} deg  contrast "
              f"{100 * sc['contrast']:.1f} %"]
    if s["groups"]:
        L.append("partitions, the position held: contrast +- the sigma to quote (profile sigma, fakes' bias and "
                 "scatter)")
        for g in s["groups"]:
            lam = "" if g["lam_um"] is None else f"  {g['lam_um']:.4f} um"
            fk = "" if g["scatter"] is None else (f", bias {100 * g['bias']:+.1f} %, scatter "
                                                 f"{100 * g['scatter']:.1f} %")
            L.append(f"  {g['partition']:>10s}{lam}  {g['contrast']:.4e} +- {g['sigma_quoted']:.2e} "
                     f"(profile {g['sigma']:.2e}{fk})" + ("" if g["ok"] else "  (parabola failed: the best point)"))
    gp = s.get("group_positions")
    if gp:
        L.append("partitions' own positions: dx, dy from the star and rho, pa, contrast +- the sigma to quote")
        for r in gp["rows"]:
            lam = "" if r["lam_um"] is None else f"  {r['lam_um']:.4f} um"
            q = r["sigma_quoted"]
            L.append(f"  {r['partition']:>10s}{lam}  dx {r['dx_px']:+.3f} +- {r['sigma_dx_px']:.3f}  dy {r['dy_px']:+.3f} "
                     f"+- {r['sigma_dy_px']:.3f} px  rho {r['rho']:.5f} +- {q['rho']:.1e}\"  pa {r['pa']:.3f} +- "
                     f"{q['pa']:.3f} deg  contrast {r['contrast']:.4e} +- {q['contrast']:.1e}")
        m, cx, cy = gp["mean"], gp["chromatic"]["dx"], gp["chromatic"]["dy"]
        unit = "px/um" if gp["wavelength"] == "um" else "px per partition"
        L += [f"  weighted mean: dx {m['dx_px']:+.3f} +- {m['dx_se']:.3f}  dy {m['dy_px']:+.3f} +- {m['dy_se']:.3f} px, "
              f"{m['ddx_px']:+.3f} ({m['ddx_z']:+.1f} sigma) and {m['ddy_px']:+.3f} ({m['ddy_z']:+.1f} sigma) from the "
              f"combined fit",
              f"  chromatic: dx {cx['slope']:+.3f} +- {cx['slope_se']:.3f} {unit} ({cx['z']:+.1f} sigma, a constant's "
              f"chi2 {cx['chi2']:.1f} for {cx['dof']}, p {cx['p_const']:.2g}), dy {cy['slope']:+.3f} +- "
              f"{cy['slope_se']:.3f} ({cy['z']:+.1f} sigma, chi2 {cy['chi2']:.1f}, p {cy['p_const']:.2g})"]
    L.append(f"--subtract-sources line: {s['subtract_sources']}")
    return "\n".join(L) + "\n"


def _figure(w: float, h: float):
    """Return a matplotlib figure that draws without pyplot.

    Parameters
    ----------
    w : float
        Width, in inches.
    h : float
        Height, in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure, with an Agg canvas.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h))
    FigureCanvasAgg(fig)
    return fig


def _plot_fit(found: Dict[str, Any], s: Dict[str, Any], path: str) -> str:
    """Draw the residual against each parameter, the near-best trials above and the profiles below.

    Parameters
    ----------
    found : dict
        :func:`extract_companion`'s result.
    s : dict
        :func:`_summary`'s numbers.
    path : str
        Where the figure goes.

    Returns
    -------
    path : str
        The figure's path.
    """
    fit = found["fit"]
    X = np.asarray(fit["trials"]["x"], float)
    R = np.asarray(fit["trials"]["residual"], float)
    order = np.argsort(np.where(np.isfinite(R), R, np.inf))
    nb = min(len(R), max(int(round(float(found["config"]["best_frac"]) * len(R))), 12))
    near = order[:nb]
    tb = X[int(fit["tpe_best"]["trial"]) - 1]
    rows = 2 if fit["profile_fits"] else 1
    fig = _figure(11.0, 3.4 * rows)
    labels = ("rho (arcsec)", "PA (deg)", "contrast")
    accent, ink = "#1f5fa8", "#333333"
    sq = s["sigma_quadratic"]
    for d in range(3):
        ax = fig.add_subplot(rows, 3, d + 1)
        ax.plot(X[near, d], R[near], ".", ms=3, color="#9a9a9a", label="trials")
        m = fit["marginal"][d]
        scale = fit["hi"][d] - fit["lo"][d]
        if m["coef"] is not None:
            xs = np.linspace(X[near, d].min(), X[near, d].max(), 100)
            u = (xs - tb[d]) / scale
            ax.plot(xs, m["coef"][0] + m["coef"][1] * u + m["coef"][2] * u * u, "-", lw=1.5, color=accent,
                    label="parabola")
        ax.axvline(tb[d], ls="--", lw=1, color=ink)
        if sq:
            for sgn in (-1, 1):
                ax.axvline(tb[d] + sgn * sq[NAMES[d]], ls=":", lw=1, color=accent)
        ax.set_xlabel(labels[d])
        ax.set_ylabel("residual")
        ax.set_title("best trials, #1 sigma dotted" if d == 0 else "", fontsize=9, loc="left")
    for d, pf in enumerate(fit["profile_fits"]):
        ax = fig.add_subplot(rows, 3, 4 + d)
        xv = np.asarray(fit["profiles"]["values"][d], float)
        yv = np.asarray(fit["profiles"]["residual"][d], float)
        ax.plot(xv, yv, "o", ms=3.5, color=ink)
        scale = fit["hi"][d] - fit["lo"][d]
        if pf["coef"] is not None:
            xs = np.linspace(xv.min(), xv.max(), 100)
            u = (xs - tb[d]) / scale
            ax.plot(xs, pf["coef"][0] + pf["coef"][1] * u + pf["coef"][2] * u * u, "-", lw=1.5, color=accent)
        if pf["ok"]:
            ax.axvline(pf["minimum"], ls="--", lw=1, color=ink)
            for sgn in (-1, 1):
                ax.axvline(pf["minimum"] + sgn * pf["sigma"], ls=":", lw=1, color=accent)
        ax.set_xlabel(labels[d])
        ax.set_ylabel("residual")
        ax.set_title("profiles, #2 sigma dotted" if d == 0 else "", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    return path


def _plot_images(found: Dict[str, Any], path: str) -> str:
    """Draw the reduction without the injection and the best fit's, around the companion.

    Parameters
    ----------
    found : dict
        :func:`extract_companion`'s result.
    path : str
        Where the figure goes.

    Returns
    -------
    path : str
        The figure's path.
    """
    from matplotlib.patches import Circle
    fit = found["fit"]
    clean = np.asarray(fit["images"]["clean"], float)
    best = np.asarray(fit["images"]["best"], float)
    x0, y0 = fit["patch_center_px"]
    r = float(fit["patch_px"])
    h = int(math.ceil(2.5 * r))
    ys = slice(max(int(y0) - h, 0), min(int(y0) + h + 1, clean.shape[0]))
    xs = slice(max(int(x0) - h, 0), min(int(x0) + h + 1, clean.shape[1]))
    a, b = clean[ys, xs], best[ys, xs]
    lim = float(np.nanmax(np.abs(a))) if np.any(np.isfinite(a)) else 1.0
    fig = _figure(8.0, 3.8)
    for k, (img, title) in enumerate(((a, "without the injection"), (b, "the best fit taken out"))):
        ax = fig.add_subplot(1, 2, k + 1)
        im = ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=(xs.start - 0.5, xs.stop - 0.5, ys.start - 0.5, ys.stop - 0.5))
        ax.add_patch(Circle((x0, y0), r, fill=False, ls="--", lw=1, color="#333333"))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)" if k == 0 else "")
    fig.colorbar(im, ax=fig.axes, shrink=0.85)
    fig.savefig(path, dpi=130)
    return path


def _plot_spectrum(s: Dict[str, Any], path: str) -> str:
    """Draw each partition's contrast with the sigma to quote.

    Parameters
    ----------
    s : dict
        :func:`_summary`'s numbers.
    path : str
        Where the figure goes.

    Returns
    -------
    path : str
        The figure's path.
    """
    g = s["groups"]
    lam = [x["lam_um"] for x in g]
    xs = np.array(lam if all(v is not None for v in lam) else range(len(g)), float)
    fig = _figure(6.0, 3.6)
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(xs, [x["contrast"] for x in g], yerr=[x["sigma_quoted"] for x in g], fmt="o", ms=4, lw=1,
                color="#1f5fa8", capsize=2)
    ax.set_xlabel("wavelength (um)" if all(v is not None for v in lam) else "partition")
    ax.set_ylabel("contrast")
    if not all(v is not None for v in lam):
        ax.set_xticks(xs)
        ax.set_xticklabels([x["partition"] for x in g], rotation=45)
    fig.tight_layout()
    fig.savefig(path)
    return path


def _plot_positions(s: Dict[str, Any], path: str) -> str:
    """Draw the partitions' own positions against wavelength, with the combined fit and the fitted line.

    Parameters
    ----------
    s : dict
        :func:`_summary`'s numbers, with ``group_positions``.
    path : str
        Where the figure goes.

    Returns
    -------
    path : str
        The figure's path.
    """
    gp = s["group_positions"]
    rows = gp["rows"]
    xs = np.array([r["lam_um"] if gp["wavelength"] == "um" else k for k, r in enumerate(rows)], float)
    fig = _figure(9.0, 3.6)
    accent, ink = "#1f5fa8", "#333333"
    for k, (key, sk, name) in enumerate((("dx_px", "sigma_dx_px", "dx"), ("dy_px", "sigma_dy_px", "dy"))):
        ax = fig.add_subplot(1, 2, k + 1)
        ax.errorbar(xs, [r[key] for r in rows], yerr=[r[sk] for r in rows], fmt="o", ms=4, lw=1, color=accent,
                    capsize=2, label="partitions")
        ax.axhline(s["best"][key], ls="--", lw=1, color=ink, label="combined fit")
        ch = gp["chromatic"][name]
        if np.isfinite(ch["slope"]):
            w = 1.0 / np.square([r[sk] for r in rows])
            l0 = float(np.sum(w * xs) / np.sum(w))
            xl = np.array([xs.min(), xs.max()])
            ax.plot(xl, ch["mean"] + ch["slope"] * (xl - l0), ":", lw=1.2, color=accent, label="weighted line")
        ax.set_xlabel("wavelength (um)" if gp["wavelength"] == "um" else "partition")
        ax.set_ylabel(f"{name} from the star (px)")
        if k == 0:
            ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    return path


def write_extract(found: Dict[str, Any], out_dir: str, label: str = "companion") -> Dict[str, str]:
    """Write an extraction's summary, tables, images and figures.

    Parameters
    ----------
    found : dict
        :func:`extract_companion`'s result.
    out_dir : str
        The directory, made if missing.
    label : str, default "companion"
        What was extracted, for the summary's title.

    Returns
    -------
    files : dict of str to str
        Each file written, by kind.

    Notes
    -----
    The files are ``extract_summary.json`` and ``extract_summary.txt``, ``extract_trials.txt``,
    ``extract_profiles.txt``, ``extract_partitions.txt``, ``extract_partition_positions.txt`` and
    ``extract_fakes.txt`` where there are any,
    ``extract_subtract_sources.txt``, the images ``extract_clean.fits`` and ``extract_residual.fits``,
    and the figures ``extract_fit.pdf``, ``extract_images.png``, ``extract_spectrum.pdf`` and
    ``extract_positions.pdf``.  A figure
    that fails is left out with a line in the summary, and the numbers are written regardless.
    """
    from . import durable
    os.makedirs(out_dir, exist_ok=True)
    s = _summary(found)
    fit = found["fit"]
    files: Dict[str, str] = {}

    def put(kind, name, text):
        p = os.path.join(out_dir, name)
        durable.write(p, text)
        files[kind] = p

    X = np.asarray(fit["trials"]["x"], float)
    R = np.asarray(fit["trials"]["residual"], float)
    rows = ["# trial  rho_as  pa_deg  contrast  residual  phase"]
    rows += [f"{i + 1:5d} {X[i, 0]:.6f} {X[i, 1] % 360:.5f} {X[i, 2]:.6e} {R[i]:.6e} {fit['trials']['phase'][i]}"
             for i in range(len(R))]
    put("trials", "extract_trials.txt", "\n".join(rows) + "\n")
    if fit["profile_fits"]:
        rows = ["# profile scans, the other two parameters held at the best: param value residual"]
        for d in range(3):
            for v, r in zip(fit["profiles"]["values"][d], fit["profiles"]["residual"][d]):
                rows.append(f"{NAMES[d]:>8s} {v:.7e} {r:.7e}")
        put("profiles", "extract_profiles.txt", "\n".join(rows) + "\n")
    if found["groups"]:
        rows = ["# partition  lam_um  contrast  sigma_quoted  sigma_profile  fake_bias  fake_scatter  floor  ok"]
        for g in s["groups"]:
            lam = "nan" if g["lam_um"] is None else f"{g['lam_um']:.5f}"
            fb = "nan" if g["bias"] is None else f"{g['bias']:+.4f}"
            fs = "nan" if g["scatter"] is None else f"{g['scatter']:.4f}"
            rows.append(f"{g['partition']} {lam} {g['contrast']:.6e} {g['sigma_quoted']:.3e} {g['sigma']:.3e} {fb} {fs} "
                        f"{g['floor']:.3e} {int(g['ok'])}")
        put("partitions", "extract_partitions.txt", "\n".join(rows) + "\n")
    gp = s.get("group_positions")
    if gp:
        rows = ["# partition  lam_um  dx_px  sigma_dx  dy_px  sigma_dy  rho_as  sigma_rho  pa_deg  sigma_pa  contrast  "
                "sigma_contrast  (the sigmas to quote)"]
        for r in gp["rows"]:
            q = r["sigma_quoted"]
            lam = "nan" if r["lam_um"] is None else f"{r['lam_um']:.5f}"
            rows.append(f"{r['partition']} {lam} {r['dx_px']:+.4f} {r['sigma_dx_px']:.4f} {r['dy_px']:+.4f} "
                        f"{r['sigma_dy_px']:.4f} {r['rho']:.6f} {q['rho']:.2e} {r['pa']:.4f} {q['pa']:.4f} "
                        f"{r['contrast']:.6e} {q['contrast']:.2e}")
        put("partition_positions", "extract_partition_positions.txt", "\n".join(rows) + "\n")
    rec = found["recovery"]
    if rec and rec["fakes"]:
        rows = ["# fake  true_rho  true_pa  true_contrast  fit_rho  fit_pa  fit_contrast  residual"]
        for k, f in enumerate(rec["fakes"]):
            t, q = f["true"], f["fit"]
            rows.append(f"{k + 1} {t['rho']:.6f} {t['pa']:.4f} {t['contrast']:.6e} {q['rho']:.6f} {q['pa'] % 360:.4f} "
                        f"{q['contrast']:.6e} {f['residual']:.5e}")
        put("fakes", "extract_fakes.txt", "\n".join(rows) + "\n")
    put("subtract_sources", "extract_subtract_sources.txt",
        f"# rho_as pa_deg contrast, the fitted companion, for --subtract-sources\n{s['subtract_sources']}\n")
    try:
        from astropy.io import fits
        for kind, key in (("clean_fits", "clean"), ("residual_fits", "best")):
            p = os.path.join(out_dir, f"extract_{'clean' if key == 'clean' else 'residual'}.fits")
            fits.PrimaryHDU(np.asarray(fit["images"][key], np.float32)).writeto(p, overwrite=True)
            files[kind] = p
    except Exception as exc:                                  # pragma: no cover  (astropy is a dependency)
        s["figure_errors"] = s.get("figure_errors", []) + [f"fits: {exc!r}"]
    for kind, fn in (("fit_pdf", lambda: _plot_fit(found, s, os.path.join(out_dir, "extract_fit.pdf"))),
                     ("images_png", lambda: _plot_images(found, os.path.join(out_dir, "extract_images.png"))),
                     ("spectrum_pdf", (lambda: _plot_spectrum(s, os.path.join(out_dir, "extract_spectrum.pdf")))
                      if s["groups"] else None),
                     ("positions_pdf", (lambda: _plot_positions(s, os.path.join(out_dir, "extract_positions.pdf")))
                      if s.get("group_positions") else None)):
        if fn is None:
            continue
        try:
            files[kind] = fn()
        except Exception as exc:
            s["figure_errors"] = s.get("figure_errors", []) + [f"{kind}: {exc!r}"]
    text = _txt(s, label)
    if s.get("figure_errors"):
        text += "".join(f"(a figure failed: {e})\n" for e in s["figure_errors"])
    put("summary_txt", "extract_summary.txt", text)
    put("summary_json", "extract_summary.json", json.dumps(_clean_json(dict(s, label=label)), indent=1) + "\n")
    return files
