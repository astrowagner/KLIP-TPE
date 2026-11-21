"""Injection-recovery contrast curve, one injected source per reduction.

It follows the protocol of the IDL predecessor's contrast generators (``contrastcurve_*.pro``, such
as ``contrastcurve_LMIRCam.pro``), run on each annulus's winning configuration:

* The test positions lie on a grid of separations, the annulus less ``edge_fwhm`` FWHM at each
  edge and ``sep_step_fwhm`` apart, with ``n_pa`` position angles per separation, offset from one
  separation to the next.  ``mode="random"`` draws ``n_samples`` positions instead.
* Each reduction holds one source, so no injection biases another's signal or noise.
* Each position iterates on its own contrast until its S/N is within ``precision`` of
  ``target_snr``, ``c <- c * target / snr``, or ``c <- 10 c`` when the source is not seen
  (S/N <= 0).  From the ``accept_after``-th reduction on (6th) a position at or above the target
  is accepted at its last contrast, and from the ``drop_after``-th (8th) one still short of
  ``(1 - precision) * target`` is dropped as blocked.  These are the predecessor's two rules,
  ``runs gt 5`` and ``runs gt 7``.
* The S/N is the predecessor's aperture photometry (``snr="aperture"``): FWHM/2 apertures, the
  source's at its injected position, the ring a FWHM apart, and the small-sample correction of
  Mawet et al. (2014).  ``snr="matched"`` takes the run's matched-filter Mawet S/N instead.
* The noise ring skips the apertures within 1.5 FWHM of the source and of every known companion
  (``sidelobe="off"``), as the predecessor's contrast generators skip the source and the aperture
  each side.  ``sidelobe="rotation"`` also leaves out each one's ADI sidelobes,
  ``+/- min(PA span of the data, the configuration's anglemax)`` and never less than
  ``sidelobe_fwhm`` FWHM of arc, and ``"fixed"`` that floor alone.  No position is placed in a
  companion's keep-out or sidelobes.  Where the arcs leave fewer than 4 apertures, the 4 farthest
  from the source are used, or the clean reduction's ring (``RunConfig.sidelobes``,
  :func:`klip_tpe.metrics.sidelobe_snr`).  Without the run's rule, a ring the arcs leave short
  gives no S/N below the metric's ``min_ring`` apertures.
* The curve is the median over the converged positions at each separation, with the 16th and 84th
  percentiles.

The driver takes two callables, ``reduce_many`` and ``measure``, so the arithmetic here is the same
whatever reduces the images (:meth:`klip_tpe.runner.Runner._contrast_grid`).

Notes
-----
The aperture S/N and ``sidelobe="off"`` have been the defaults since 2026-11-01.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ContrastGridConfig", "GridPosition", "separations_px", "grid_positions", "random_positions",
           "footprint_deg", "in_footprint", "next_contrast", "run_grid", "summarize"]

#: The default ``mode`` of a new :class:`ContrastGridConfig`.  The test suite sets it to
#: ``"off"`` (tests/conftest.py) so that its many short runs keep their pace, and a test of the
#: grid asks for it.
_DEFAULT = {"mode": "grid"}

STATUS_DONE = ("converged", "accepted")


@dataclass
class ContrastGridConfig:
    """Settings of the injection-recovery curve, whose protocol the module's docstring describes.

    Attributes
    ----------
    mode : {"grid", "random", "off"}, default "grid"
        How the positions are laid out, or no curve.  A run saved before the grid existed resumes with
        "off", and the test suite's default is "off" (``tests/conftest.py``).
    n_pa : int, default 12
        Number of position angles per separation.
    n_samples : int, default 96
        Number of random positions.
    sep_step_fwhm : float, default 1.0
        Step between separations, in FWHM.
    edge_fwhm : float, default 1.0
        Margin inside each edge of the zone, in FWHM.
    target_snr : float, default 5.0
        The S/N each position iterates to.
    precision : float, default 0.1
        Relative tolerance on the target.
    snr : {"aperture", "matched"}, default "aperture"
        The S/N, the predecessor's aperture photometry at the injected position
        (:func:`~klip_tpe.metrics.aperture_mawet_snr`) or the run's matched-filter Mawet S/N.
    sidelobe : {"off", "rotation", "fixed"}, default "off"
        What the ring leaves out besides the 1.5 FWHM keep-outs: nothing, the arc the field rotation
        sweeps, or ``sidelobe_fwhm`` FWHM of arc each side.
    sidelobe_fwhm : float, default 2.0
        Smallest half-width of an arc, in FWHM of arc.
    known_radial_fwhm : float, default 1.5
        Radial half-width of a known companion's arc, in FWHM.
    accept_after : int, default 6
        Reduction from which a position at or above the target is accepted.
    drop_after : int, default 8
        Reduction from which a position still short of the target is dropped as blocked.
    c_max : float, default 0.1
        Contrast cap, at which a position still short of the target is blocked.
    """

    mode: str = field(default_factory=lambda: _DEFAULT["mode"])
    n_pa: int = 12
    n_samples: int = 96
    sep_step_fwhm: float = 1.0
    edge_fwhm: float = 1.0
    target_snr: float = 5.0
    precision: float = 0.1
    #: ``"aperture"`` (default since 2026-11-01): the IDL predecessor's contrast generators'
    #: aperture photometry at the injected position (:func:`~klip_tpe.metrics.aperture_mawet_snr`,
    #: the predecessor's ``near2_snr``).  ``"matched"``: the run's matched-filter Mawet S/N
    snr: str = "aperture"
    #: ``"off"`` (default since 2026-11-01): no arcs, the ring skips the apertures within 1.5
    #: FWHM of the source and of each known companion, as the IDL predecessor's contrast
    #: generators skip the source and the aperture each side of it.  ``"rotation"``: also the arc
    #: the field rotation sweeps, ``"fixed"``: ``sidelobe_fwhm`` FWHM of arc each side
    sidelobe: str = "off"
    sidelobe_fwhm: float = 2.0
    known_radial_fwhm: float = 1.5
    accept_after: int = 6
    drop_after: int = 8
    c_max: float = 0.1

    @property
    def enabled(self) -> bool:
        """Whether the mode makes a curve, ``"grid"`` or ``"random"``.
        """
        return str(self.mode).lower() in ("grid", "random")


@dataclass
class GridPosition:
    """One test position and its history of trials.

    Attributes
    ----------
    index : int
        The position's index.
    r_px : float
        Separation, in pixels.
    theta : float
        Angle, in degrees, in the run's source convention.
    sep_index : int
        Index of its separation.
    trials : list of tuple of 2 floats
        Its ``(contrast, snr)`` trials.
    status : str, default "active"
        "active", "converged", "accepted", "blocked", "failed" or "excluded".
    contrast : float or None, default None
        The contrast it settled at.
    why : str or None, default None
        Why it failed, was blocked or was excluded.
    next_c : float or None, default None
        The contrast of its next trial.
    noise : str or None, default None
        Where its last measurement's noise came from when the ring beside the sidelobes held too few
        apertures (:func:`klip_tpe.metrics.sidelobe_snr`'s ``fallback``), else None.
    """

    index: int
    r_px: float
    theta: float
    sep_index: int
    trials: List[Tuple[float, float]] = field(default_factory=list)
    status: str = "active"
    contrast: Optional[float] = None
    why: Optional[str] = None
    next_c: Optional[float] = None
    #: where its last measurement's noise came from when the ring beside the sidelobes held
    #: too few apertures (:func:`klip_tpe.metrics.sidelobe_snr`'s ``fallback``), else None
    noise: Optional[str] = None

    def to_dict(self, pxscale: float) -> Dict[str, Any]:
        """Return the position as a JSON-ready dict.

        Parameters
        ----------
        pxscale : float
            Plate scale, in arcsec per pixel.

        Returns
        -------
        record : dict
            ``index``, ``rho_as``, ``theta``, ``sep_index``, ``status``, ``contrast``, ``why``, ``noise`` and
            ``trials``, None for a NaN S/N.
        """
        return {"index": self.index, "rho_as": float(self.r_px * pxscale), "theta": float(self.theta),
                "sep_index": int(self.sep_index), "status": self.status,
                "contrast": None if self.contrast is None else float(self.contrast), "why": self.why,
                "noise": self.noise,
                "trials": [[float(c), None if not np.isfinite(s) else float(s)] for c, s in self.trials]}


# ----------------------------------------------------------------------------- geometry
def separations_px(inrad: float, outrad: float, fwhm: float, cfg: ContrastGridConfig,
                   floor_px: float = 0.0) -> np.ndarray:
    """Return the test separations, ``edge_fwhm`` FWHM in from each edge of the zone and ``sep_step_fwhm`` FWHM apart.

    They are centered in the room left over, outside ``floor_px``, and where there is no room there is
    one separation at the middle of the zone.

    Parameters
    ----------
    inrad : float
        Inner edge of the zone, in pixels.
    outrad : float
        Outer edge of the zone, in pixels.
    fwhm : float
        FWHM of the PSF, in pixels.
    cfg : ContrastGridConfig
        The settings.
    floor_px : float, default 0.0
        Smallest separation, in pixels.

    Returns
    -------
    seps : ndarray
        The separations, in pixels.
    """
    lo = max(float(inrad) + cfg.edge_fwhm * fwhm, float(floor_px))
    hi = float(outrad) - cfg.edge_fwhm * fwhm
    step = max(cfg.sep_step_fwhm * fwhm, 1e-6)
    if hi < lo:
        return np.array([0.5 * (max(float(inrad), float(floor_px)) + float(outrad))])
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    span = (n - 1) * step
    return lo + 0.5 * ((hi - lo) - span) + step * np.arange(n)


def grid_positions(seps: Sequence[float], n_pa: int) -> List[GridPosition]:
    """Lay out ``n_pa`` evenly spaced angles per separation.

    Each separation turns the pattern by the golden fraction of the angle step, so no azimuth is tested
    at every separation.

    Parameters
    ----------
    seps : sequence of float
        The separations, in pixels.
    n_pa : int
        Number of angles per separation, at least 1.

    Returns
    -------
    positions : list of GridPosition
        The positions.
    """
    out: List[GridPosition] = []
    n_pa = max(int(n_pa), 1)
    step = 360.0 / n_pa
    gold = 0.5 * (np.sqrt(5.0) - 1.0)
    for j, r in enumerate(seps):
        off = ((j * gold) % 1.0) * step
        for k in range(n_pa):
            out.append(GridPosition(len(out), float(r), float((off + k * step) % 360.0), j))
    return out


def random_positions(seps: Sequence[float], step_px: float, n: int, rng: np.random.Generator) -> List[GridPosition]:
    """Draw positions uniform in area over the separations' span and in angle, each binned to the nearest separation.

    The span reaches half a step beyond the first and the last separation.

    Parameters
    ----------
    seps : sequence of float
        The separations, in pixels.
    step_px : float
        Step between separations, in pixels.
    n : int
        Number of positions.
    rng : numpy.random.Generator
        The generator.

    Returns
    -------
    positions : list of GridPosition
        The positions.
    """
    seps = np.asarray(seps, float)
    lo = max(float(seps.min()) - 0.5 * step_px, 0.0)
    hi = float(seps.max()) + 0.5 * step_px
    out: List[GridPosition] = []
    for i in range(int(n)):
        r = float(np.sqrt(lo * lo + rng.random() * (hi * hi - lo * lo)))
        th = float(rng.random() * 360.0)
        out.append(GridPosition(i, r, th, int(np.argmin(np.abs(seps - r)))))
    return out


def footprint_deg(r_px: float, fwhm: float, cfg: ContrastGridConfig, rotation_deg: Optional[float]) -> float:
    """Return the half-width of a source's sidelobe arc at a separation.

    It is the field rotation the references span (``rotation_deg``, ``sidelobe="rotation"``), never
    less than ``sidelobe_fwhm`` FWHM of arc, and at most 180 degrees.  With ``sidelobe="off"`` there is
    no arc, and it is the angle of the 1.5 FWHM keep-out, inside which no position goes beside a known
    companion.

    Parameters
    ----------
    r_px : float
        The separation, in pixels.
    fwhm : float
        FWHM of the PSF, in pixels.
    cfg : ContrastGridConfig
        The settings.
    rotation_deg : float or None
        The field rotation the references span, in degrees.

    Returns
    -------
    half_deg : float
        The half-width, in degrees.
    """
    if str(cfg.sidelobe).lower() == "off":
        return min(float(np.rad2deg(1.5 * fwhm / max(float(r_px), 1e-6))), 180.0)
    floor = float(np.rad2deg(cfg.sidelobe_fwhm * fwhm / max(float(r_px), 1e-6)))
    if str(cfg.sidelobe).lower() == "fixed" or rotation_deg is None or not np.isfinite(rotation_deg):
        return min(floor, 180.0)
    return min(max(float(rotation_deg), floor), 180.0)


def in_footprint(r_px: float, theta: float, rk_px: float, thk: float, half_deg: float, radial_px: float) -> bool:
    """Tell whether a position lies in an arc around a source, within a radial reach of its separation.

    Parameters
    ----------
    r_px : float
        Separation of the position, in pixels.
    theta : float
        Angle of the position, in degrees.
    rk_px : float
        Separation of the source, in pixels.
    thk : float
        Angle of the source, in degrees.
    half_deg : float
        Half-width of the arc, in degrees.
    radial_px : float
        Radial reach, in pixels.

    Returns
    -------
    inside : bool
        True inside the arc.
    """
    d = abs((float(theta) - float(thk) + 180.0) % 360.0 - 180.0)
    return abs(float(r_px) - float(rk_px)) <= radial_px and d <= half_deg


# ---------------------------------------------------------------------------- iteration
def next_contrast(c: float, snr: float, target: float, c_max: float) -> float:
    """Return a position's next contrast, by the IDL predecessor's update.

    It is ``c * target / snr``, or ``10 c`` for a source not seen, capped at ``c_max``.

    Parameters
    ----------
    c : float
        The contrast tried.
    snr : float
        Its S/N.
    target : float
        The target S/N.
    c_max : float
        The cap.

    Returns
    -------
    c_next : float
        The next contrast.
    """
    nxt = c * target / snr if (np.isfinite(snr) and snr > 0) else 10.0 * c
    return float(min(nxt, c_max))


def run_grid(positions: List[GridPosition], start: Callable[[GridPosition], float],
             reduce_many: Callable[[List[Tuple[GridPosition, float]]], Iterable[Any]],
             measure: Callable[[Any, GridPosition, float], Tuple[float, Optional[str]]],
             cfg: ContrastGridConfig, log: Callable[[str], None] = lambda s: None,
             stage: Callable[[str], None] = lambda s: None) -> Tuple[List[GridPosition], int]:
    """Iterate every active position to the target S/N, a wave at a time.

    Parameters
    ----------
    positions : list of GridPosition
        The positions, updated in place.
    start : callable
        Function ``start(pos)`` giving a position's first contrast, capped at ``c_max``.
    reduce_many : callable
        Function ``reduce_many([(pos, c), ...])`` yielding one result per entry, in order, an image or an
        exception for a failed reduction.
    measure : callable
        Function ``measure(result, pos, c)`` returning ``(snr, why)``, ``why`` a reason for a NaN.
    cfg : ContrastGridConfig
        The settings.
    log : callable, default a no-op
        Function that takes each line of the log.
    stage : callable, default a no-op
        Function that takes the label of each wave.

    Returns
    -------
    positions : list of GridPosition
        The positions, each settled as converged, accepted, blocked or failed.
    n_reductions : int
        Number of reductions made.
    """
    tgt, prec = float(cfg.target_snr), float(cfg.precision)
    for p in positions:
        if p.status == "active":
            c0 = float(start(p))
            p.next_c = min(c0, cfg.c_max) if np.isfinite(c0) and c0 > 0 else cfg.c_max
    n_red = 0
    wave = 0
    while True:
        active = [p for p in positions if p.status == "active"]
        if not active:
            break
        wave += 1
        stage(f"wave {wave}: {len(active)} position(s)")
        batch = [(p, float(p.next_c)) for p in active]
        for (p, c), res in zip(batch, reduce_many(batch)):
            n_red += 1
            if isinstance(res, BaseException):
                p.status, p.why = "failed", f"reduction failed: {type(res).__name__}: {str(res)[:80]}"
                continue
            snr, why = measure(res, p, c)
            snr = float(snr) if snr is not None else np.nan
            p.trials.append((c, snr))
            runs = len(p.trials)
            if not np.isfinite(snr):
                p.status, p.why = "failed", why or "no S/N"
                continue
            if abs(snr / tgt - 1.0) <= prec:
                p.status, p.contrast = "converged", c
                continue
            if runs >= cfg.accept_after and snr >= tgt:
                p.status, p.contrast = "accepted", c
                continue
            if runs >= cfg.drop_after or (c >= cfg.c_max and snr < tgt):
                p.status = "blocked"
                p.why = (f"S/N {snr:.2f} after {runs} reductions" if c < cfg.c_max
                         else f"S/N {snr:.2f} at the contrast cap {cfg.c_max:g}")
                continue
            p.next_c = next_contrast(c, snr, tgt, cfg.c_max)
        left = sum(p.status == "active" for p in positions)
        log(f"  contrast grid: wave {wave} done, {len(active) - left} of {len(active)} position(s) settled"
            + (f", {left} iterating" if left else ""))
    return positions, n_red


def _pct(v: np.ndarray, q: float) -> float:
    """Return a percentile, NaN for no values.

    Parameters
    ----------
    v : ndarray
        The values.
    q : float
        The percentile, from 0 to 100.

    Returns
    -------
    value : float
        The percentile.
    """
    return float(np.percentile(v, q)) if v.size else float("nan")


def summarize(positions: Sequence[GridPosition], seps: Sequence[float], pxscale: float,
              cfg: ContrastGridConfig, sidelobe_deg: Sequence[float], n_reductions: int) -> Dict[str, Any]:
    """Summarize the curve, per separation the settled positions' median contrast, its percentiles and the counts.

    Parameters
    ----------
    positions : sequence of GridPosition
        The positions.
    seps : sequence of float
        The separations, in pixels.
    pxscale : float
        Plate scale, in arcsec per pixel.
    cfg : ContrastGridConfig
        The settings.
    sidelobe_deg : sequence of float
        Each separation's :func:`footprint_deg`, the arc's half-width, or with ``sidelobe="off"`` the
        1.5 FWHM keep-out's angle, which is no arc.
    n_reductions : int
        Number of reductions made.

    Returns
    -------
    curve : dict
        ``r_as``, ``curve``, ``p16``, ``p84``, ``n`` (settled), ``n_pos`` (tested), ``n_all`` (laid out,
        the excluded too), ``n_blocked``, ``sidelobe_deg``, the settings (``mode``, ``n_pa``,
        ``n_samples``, ``target_snr``, ``precision``, ``snr``, ``sidelobe``), ``n_reductions`` and
        ``n_excluded``, with None for NaN.
    """
    seps = np.asarray(seps, float)
    med, p16, p84, n, npos, nblk, nall = [], [], [], [], [], [], []
    for j in range(seps.size):
        nall.append(int(sum(p.sep_index == j for p in positions)))
        here = [p for p in positions if p.sep_index == j and p.status != "excluded"]
        v = np.array([p.contrast for p in here if p.status in STATUS_DONE and p.contrast is not None], float)
        med.append(float(np.median(v)) if v.size else float("nan"))
        p16.append(_pct(v, 16.0))
        p84.append(_pct(v, 84.0))
        n.append(int(v.size))
        npos.append(len(here))
        nblk.append(int(sum(p.status == "blocked" for p in here)))

    def js(a):
        return [None if not np.isfinite(x) else float(x) for x in a]
    return {"r_as": js(seps * pxscale), "curve": js(med), "p16": js(p16), "p84": js(p84), "n": n, "n_pos": npos,
            "n_all": nall, "n_blocked": nblk, "sidelobe_deg": js(sidelobe_deg), "mode": str(cfg.mode), "n_pa": int(cfg.n_pa),
            "n_samples": int(cfg.n_samples), "target_snr": float(cfg.target_snr), "precision": float(cfg.precision),
            "snr": str(cfg.snr), "sidelobe": str(cfg.sidelobe), "n_reductions": int(n_reductions),
            "n_excluded": int(sum(p.status == "excluded" for p in positions))}
