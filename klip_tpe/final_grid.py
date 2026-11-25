"""The final-image grid, an election of each annulus' winner on the stitch and the run's contrast curve.

One stage after the searches (``RunConfig.final_grid``, on by default) measures single injections on
the assembled image, as the stitched verify measures them, and uses them twice:

* **The election.**  Each annulus' finalists, the configurations its validation scored and on
  request those of other runs of the same setup, go in turn into the annulus' slot of the stitch
  with the other annuli at their current winners.  Each is read on the grid points whose reading
  its slot reaches, at each tile's fixed contrast, and it replaces the winner only when its mean
  paired gain over the winner exceeds ``z`` times that gain's standard error.  Two sweeps go inner to
  outer and back, or stop sooner once a sweep changes nothing.
* **The curve.**  The elected winners are read on a second grid, the first turned by half a position
  angle step, each point's contrast stepped to S/N 5 as the contrast grid steps it
  (:func:`klip_tpe.contrast_grid.run_grid`).  The curve is the run's ``contrast_curve.txt`` in place
  of the per-annulus grids, with several separations per annulus.  A seam line per annulus gives the
  curve near each tile edge over its middle, and the seam steps read every point over a smooth fit of
  the whole curve, so a seam shows as a step between two tiles with the curve's slope taken out.

The grid is the same for every run of a setup.  Each tile has an odd number of separations, at least
3, at the middles of equal radial bands no wider than ``sep_step_fwhm`` FWHM, so the first and the
last lie near the tile's edges and one at its middle.  Each separation has ``n_pa`` position angles,
turned from one separation to the next by the golden fraction of the step.  No point lies in a known
companion's keep-out or where the run's sampler places no source, such as a forbidden sector.

Each point holds one injected source.  Every annulus whose part of the stitch lies within
:data:`READ_FWHM` FWHM of the point is reduced with it, and the other annuli take a clean image.  An
annulus whose part of the stitch lies farther cannot reach the point's reading, whose aperture,
matched filter and noise ring hold at most a few percent of their weight beyond that distance.  The
reductions an annulus' finalists share, its neighbors' at their current winners, are made once.
Every reading is kept in ``state/readings.jsonl`` under the points, contrasts and configurations it
was made with, so a stage started again on the same directory reads them back instead of reducing
again, and an annulus whose neighbors did not change is elected again from those readings alone.

:func:`run_final_grid` runs the stage on a runner whose annuli are all done, inside a run
(:meth:`klip_tpe.runner.Runner.finish`) or on a finished run (``klip-tpe elect``).

Notes
-----
The stage follows NEAR2's note of 2026-11-16, "one final-image grid, for the election and the
contrast curve".  It replaces the validation's final round (``ValidationConfig.n_final``), which a
run with the stage on skips.

At a fixed contrast the finalists' readings pair point by point, and in the linear regime the S/N at
a fixed contrast ranks configurations as their contrast at S/N 5 does.  Stepping each point's
contrast for every finalist would cost 3 to 6 reductions a point instead of 1 and would read each
finalist at its own contrast, so only the curve steps.

The election picks the largest of several noisy means, so the elected winner's gain on the election's
own points runs high.  The gain of each change on the curve's grid, at the same contrasts
(``held_out``), is the honest one.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import Source
from .refine import Entrant, StitchScorer, _check_setup, _nanmedian, config_key, paired_summary, run_entrants

__all__ = ["FinalGridConfig", "GridPoint", "FinalGrid", "READ_FWHM", "CURVE_SNRS", "tile_separations",
           "grid_points", "exclude_known", "exclude_sampled", "stitch_regions", "touched", "elect_one",
           "collect_finalists", "run_final_grid", "seam_ratios", "setup_of", "run_weights", "elect_products"]

#: The default ``mode`` of a new :class:`FinalGridConfig`.  The test suite sets it to ``"off"``
#: (tests/conftest.py) so that its many short runs keep their pace, and a test of the stage asks for
#: it.
_DEFAULT = {"mode": "on"}

#: Distance from a point within which an annulus' part of the stitch is reduced with the point's
#: source, in FWHM.  The matched filter's weight there is 1/16 of its peak, and the source's own light
#: times that weight 1/256.
READ_FWHM = 1.0

#: How the curve's points are read: the contrast grid's S/N, the stitched verify's (Path A), or no
#: curve.
CURVE_SNRS = ("grid", "path_a", "off")

#: Memory the reduced images kept for later readings may take, in bytes.
IMAGE_BUDGET = 512 * 2 ** 20


@dataclass
class FinalGridConfig:
    """Settings of the final-image grid, whose stage the module's docstring describes.

    Attributes
    ----------
    mode : {"on", "off"}, default "on"
        Whether the stage runs.  A run saved before the stage existed resumes with "off", and the test
        suite's default is "off" (``tests/conftest.py``).
    n_pa : int, default 12
        Number of position angles per separation.
    sep_step_fwhm : float, default 1.0
        Widest radial band of a tile per separation, in FWHM.
    cover_fwhm : float, default 0.5
        Distance from an annulus' part of the stitch within which a point counts toward the annulus'
        election, in FWHM.
    target : tuple of 2 floats, default (5.0, 8.0)
        Range of the median S/N the starting winners read on each tile's calibration points, at the
        tile's contrast.
    n_cal : int, default 12
        Number of a tile's points its contrast is calibrated on.
    max_cal : int, default 4
        Number of calibration steps at most.
    sweeps : int, default 2
        Number of sweeps of the election at most.
    z : float, default 2.0
        How many standard errors a finalist's mean paired gain must exceed to replace the winner.
    finalists : int, default 0
        Number of each run's finalists per annulus, its winner and its best by validated score.  0 takes
        every configuration its validation scored.
    entrants_from : list of str, default []
        Directories of other finished runs of the same setup whose finalists enter too.
    curve_snr : {"grid", "path_a", "off"}, default "grid"
        How the curve's points are read, with the contrast grid's S/N (``RunConfig.contrast_grid``, the
        IDL predecessor's aperture photometry by default), with the stitched verify's, or not at all.
    """

    mode: str = field(default_factory=lambda: _DEFAULT["mode"])
    n_pa: int = 12
    sep_step_fwhm: float = 1.0
    cover_fwhm: float = 0.5
    target: Tuple[float, float] = (5.0, 8.0)
    n_cal: int = 12
    max_cal: int = 4
    sweeps: int = 2
    z: float = 2.0
    finalists: int = 0
    entrants_from: List[str] = field(default_factory=list)
    curve_snr: str = "grid"

    @property
    def enabled(self) -> bool:
        """Whether the mode runs the stage, ``"on"``.
        """
        return str(self.mode).lower() == "on"

    def check(self) -> None:
        """Check the settings.

        Raises
        ------
        ValueError
            If a setting is out of range.
        """
        if str(self.mode).lower() not in ("on", "off"):
            raise ValueError(f"final_grid mode {self.mode!r}: on or off")
        if int(self.n_pa) < 1:
            raise ValueError("final_grid n_pa must be at least 1")
        if not float(self.sep_step_fwhm) > 0:
            raise ValueError("final_grid sep_step_fwhm must be positive")
        if float(self.cover_fwhm) < 0:
            raise ValueError("final_grid cover_fwhm must be 0 or more")
        if not 0 < float(self.target[0]) <= float(self.target[1]):
            raise ValueError("final_grid target must be two S/N, 0 < low <= high")
        if int(self.n_cal) < 1 or int(self.max_cal) < 1 or int(self.sweeps) < 1:
            raise ValueError("final_grid n_cal, max_cal and sweeps must be at least 1")
        if int(self.finalists) < 0:
            raise ValueError("final_grid finalists must be 0 or more")
        if str(self.curve_snr) not in CURVE_SNRS:
            raise ValueError(f"final_grid curve_snr {self.curve_snr!r}: one of {', '.join(CURVE_SNRS)}")


@dataclass
class GridPoint:
    """One point of the grid, an annulus' tile, a separation and a position angle.

    Attributes
    ----------
    index : int
        The point's index in its grid.
    annulus : int
        The annulus whose tile holds it, counted from 0.
    sep_index : int
        Index of its separation in the grid, over every tile.
    band : int
        Index of its separation in the tile, 0 at the inner edge.
    n_bands : int
        Number of the tile's separations, those under the grid's floor counted.
    r_px : float
        Separation, in pixels.
    theta : float
        Angle, in degrees, in the run's source convention.
    status : {"active", "excluded"}, default "active"
        Whether it is read, or excluded for a known companion.
    why : str or None, default None
        Why it is excluded.
    """

    index: int
    annulus: int
    sep_index: int
    band: int
    n_bands: int
    r_px: float
    theta: float
    status: str = "active"
    why: Optional[str] = None

    @property
    def role(self) -> str:
        """Where its separation lies in the tile, ``"inner"``, ``"middle"``, ``"outer"`` or ``"between"``.
        """
        if self.n_bands <= 1 or self.band == self.n_bands // 2:
            return "middle"
        if self.band == 0:
            return "inner"
        if self.band == self.n_bands - 1:
            return "outer"
        return "between"

    def source(self, pxscale: float, contrast: float) -> Source:
        """Return the point's source at a contrast.

        Parameters
        ----------
        pxscale : float
            Plate scale, in arcsec per pixel.
        contrast : float
            The contrast.

        Returns
        -------
        source : Source
            The source, its separation in arcsec.
        """
        return Source(float(self.r_px) * float(pxscale), float(self.theta), float(contrast))

    def to_dict(self, pxscale: float) -> Dict[str, Any]:
        """Return the point as a JSON-ready dict, its annulus counted from 1.

        Parameters
        ----------
        pxscale : float
            Plate scale, in arcsec per pixel.

        Returns
        -------
        record : dict
            ``index``, ``annulus``, ``sep_index``, ``band``, ``role``, ``r_px``, ``rho_as``, ``theta``,
            ``status`` and ``why``.
        """
        return {"index": int(self.index), "annulus": int(self.annulus) + 1, "sep_index": int(self.sep_index),
                "band": int(self.band), "role": self.role, "r_px": float(self.r_px),
                "rho_as": float(self.r_px) * float(pxscale), "theta": float(self.theta), "status": self.status,
                "why": self.why}


# ----------------------------------------------------------------------------- geometry
def tile_separations(t0: float, t1: float, fwhm: float, step_fwhm: float = 1.0) -> np.ndarray:
    """Return a tile's separations, the middles of an odd number of equal radial bands, at least 3.

    The bands are no wider than ``step_fwhm`` FWHM where the tile allows, so the first separation lies
    near the inner edge, the last near the outer edge and one at the middle.

    Parameters
    ----------
    t0 : float
        Inner edge of the tile, in pixels.
    t1 : float
        Outer edge of the tile, in pixels.
    fwhm : float
        FWHM of the PSF, in pixels.
    step_fwhm : float, default 1.0
        Widest band, in FWHM.

    Returns
    -------
    seps : ndarray, shape (n,)
        The separations, in pixels, empty for a tile of no width.
    """
    w = float(t1) - float(t0)
    if not w > 0:
        return np.zeros(0)
    n = max(3, int(np.ceil(w / max(float(step_fwhm) * float(fwhm), 1e-6) - 1e-9)))
    if n % 2 == 0:
        n += 1
    return float(t0) + (np.arange(n) + 0.5) * w / n


def grid_points(tiles: Sequence[Tuple[int, float, float]], fwhm: float, n_pa: int, step_fwhm: float = 1.0,
                floor_px: float = 0.0, turn: float = 0.0) -> List[GridPoint]:
    """Lay out the grid over the tiles, ``n_pa`` position angles per separation.

    Each separation turns the angles by the golden fraction of the step, counted over the separations
    of every tile, so no azimuth is read at every separation.  ``turn`` turns every angle by that
    fraction of the step more, half a step for the curve's grid.

    Parameters
    ----------
    tiles : sequence of tuple
        ``(annulus, inner, outer)`` per tile in the stitch's order, the annulus counted from 0 and the
        edges in pixels.
    fwhm : float
        FWHM of the PSF, in pixels.
    n_pa : int
        Number of position angles per separation, at least 1.
    step_fwhm : float, default 1.0
        Widest radial band of a tile per separation, in FWHM (:func:`tile_separations`).
    floor_px : float, default 0.0
        Smallest separation, in pixels.  A tile's separations under it are left out.
    turn : float, default 0.0
        Turn of every angle, in steps.

    Returns
    -------
    points : list of GridPoint
        The points, tile by tile and separation by separation.
    """
    n_pa = max(int(n_pa), 1)
    step = 360.0 / n_pa
    gold = 0.5 * (np.sqrt(5.0) - 1.0)
    out: List[GridPoint] = []
    g = 0
    for a, t0, t1 in tiles:
        seps = tile_separations(t0, t1, fwhm, step_fwhm)
        for b, r in enumerate(seps):
            if r < float(floor_px):
                continue
            off = ((g * gold) % 1.0) * step + float(turn) * step
            for m in range(n_pa):
                out.append(GridPoint(len(out), int(a), g, b, len(seps), float(r), float((off + m * step) % 360.0)))
            g += 1
    return out


def exclude_known(points: Sequence[GridPoint], known: Sequence[Tuple[float, float]], pxscale: float, fwhm: float,
                  gc) -> int:
    """Mark the points in a known companion's keep-out as excluded, as the contrast grid excludes them.

    The keep-out is :func:`klip_tpe.contrast_grid.footprint_deg` of arc around the companion, within
    ``gc.known_radial_fwhm`` FWHM of its separation.

    Parameters
    ----------
    points : sequence of GridPoint
        The points, marked in place.
    known : sequence of tuple of 2 floats
        The known companions, ``(rho_arcsec, theta_deg)``.
    pxscale : float
        Plate scale, in arcsec per pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    gc : ContrastGridConfig
        The run's contrast grid settings, whose ring rule sets the arc.

    Returns
    -------
    n : int
        Number of points excluded.
    """
    from . import contrast_grid as CG
    n = 0
    for p in points:
        for rk_as, tk in known:
            rk = float(rk_as) / float(pxscale)
            if CG.in_footprint(p.r_px, p.theta, rk, float(tk), CG.footprint_deg(rk, fwhm, gc, None),
                               float(gc.known_radial_fwhm) * float(fwhm)):
                p.status, p.why = "excluded", "in a known companion's keep-out"
                n += 1
                break
    return n


def exclude_sampled(points: Sequence[GridPoint], sampler, pxscale: float) -> int:
    """Mark the points the run's sampler would not place a source at as excluded.

    They are those in a known source's keep-out, a forbidden sector (``PositionSampler.forbidden_pa``,
    the dead zones of a four-quadrant mask) or on its mask (``PositionSampler.forbidden_mask``), as
    ``PositionSampler._bad`` says for a source alone.  A sampler without that test excludes none.

    Parameters
    ----------
    points : sequence of GridPoint
        The points, marked in place.
    sampler : PositionSampler or None
        The run's sampler.
    pxscale : float
        Plate scale, in arcsec per pixel.

    Returns
    -------
    n : int
        Number of points excluded.
    """
    bad = getattr(sampler, "_bad", None)
    if not callable(bad):
        return 0
    n = 0
    for p in points:
        if p.status == "active" and bad([float(p.r_px) * float(pxscale)], [float(p.theta)]):
            p.status, p.why = "excluded", "where the run's sampler places no source"
            n += 1
    return n


def stitch_regions(itiles: Sequence[Tuple[float, float]], pads: Sequence[Tuple[float, float]]
                   ) -> List[Tuple[float, float]]:
    """Return the radii over which each annulus' image enters the stitch, its tile widened by its pads.

    Parameters
    ----------
    itiles : sequence of tuple of 2 floats
        The tiles the stitched images are made over, in pixels.
    pads : sequence of tuple of 2 floats
        Each tile's ``(pad_in, pad_out)``, in pixels.

    Returns
    -------
    regions : list of tuple of 2 floats
        ``(inner, outer)`` per tile, in pixels.
    """
    return [(float(lo) - float(pi), float(hi) + float(po)) for (lo, hi), (pi, po) in zip(itiles, pads)]


def touched(r_px: float, regions: Sequence[Tuple[float, float]], reach_px: float) -> List[int]:
    """Return the tiles whose part of the stitch lies within a distance of a separation.

    Parameters
    ----------
    r_px : float
        The separation, in pixels.
    regions : sequence of tuple of 2 floats
        Each tile's part of the stitch (:func:`stitch_regions`).
    reach_px : float
        The distance, in pixels.

    Returns
    -------
    tiles : list of int
        The tiles' positions in ``regions``.
    """
    r = float(r_px)
    return [i for i, (lo, hi) in enumerate(regions) if max(lo - r, r - hi, 0.0) <= float(reach_px) + 1e-9]


def elect_one(readings: Dict[str, np.ndarray], order: Sequence[str], winner: str, z: float = 2.0
              ) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Return an annulus' winner after its election, and each finalist's statistics against the winner.

    A finalist replaces the winner when its mean paired gain over the points both read is positive and
    exceeds ``z`` times its standard error, with at least 2 such points.  Of several, the one of the
    largest mean gain is elected, the first in ``order`` on a tie.

    Parameters
    ----------
    readings : dict of str to ndarray
        Each finalist's readings on the annulus' points, by configuration key, NaN where none.
    order : sequence of str
        The finalists' keys, in the order a tie goes by.
    winner : str
        The current winner's key, among ``order``.
    z : float, default 2.0
        How many standard errors the gain must exceed.

    Returns
    -------
    winner : str
        The winner's key after the election.
    stats : dict of str to dict
        Each finalist's :func:`klip_tpe.refine.paired_summary` against the winner going in.
    """
    base = np.asarray(readings[winner], float)
    stats = {k: paired_summary(np.asarray(readings[k], float), base) for k in order}
    best, gain = winner, 0.0
    for k in order:
        if k == winner:
            continue
        s = stats[k]
        if (s["n_paired"] >= 2 and np.isfinite(s["diff_se"]) and s["diff_mean"] > gain
                and s["diff_mean"] > float(z) * s["diff_se"]):
            best, gain = k, float(s["diff_mean"])
    return best, stats


def seam_ratios(curve: Dict[str, Any], seps: Sequence[Tuple[int, int, int]], n_tiles: int,
                order: Sequence[int]) -> List[Dict[str, Any]]:
    """Return each annulus' seam line, the curve near each tile edge over its middle.

    Parameters
    ----------
    curve : dict
        The curve, :func:`klip_tpe.contrast_grid.summarize`'s, one entry per separation.
    seps : sequence of tuple
        ``(annulus, band, n_bands)`` per separation of the curve, the annulus counted from 0.
    n_tiles : int
        Number of tiles.
    order : sequence of int
        The annuli in the stitch's order.

    Returns
    -------
    seams : list of dict
        Per annulus in the stitch's order ``annulus`` (from 1), ``inner`` and ``outer`` (the ratios,
        None where the curve has no value there or the tile has a single separation) and
        ``inner_edge`` and ``outer_edge`` ("seam" between two tiles, "edge" at the stitch's own
        edges).
    """
    c = [None if v is None else float(v) for v in curve.get("curve") or []]
    out = []
    for pos, a in enumerate(order):
        mine = {b: (i, n) for i, (aa, b, n) in enumerate(seps) if aa == a}
        n = next((nn for _, nn in mine.values()), 0)

        def at(b: int) -> Optional[float]:
            i = mine.get(b, (None, None))[0]
            v = None if i is None or i >= len(c) else c[i]
            return v if v is not None and np.isfinite(v) and v > 0 else None
        mid = at(n // 2) if n else None
        lo, hi = (at(0), at(n - 1)) if n > 1 else (None, None)
        out.append({"annulus": int(a) + 1, "inner": None if lo is None or mid is None else lo / mid,
                    "outer": None if hi is None or mid is None else hi / mid,
                    "inner_edge": "edge" if pos == 0 else "seam",
                    "outer_edge": "edge" if pos == n_tiles - 1 else "seam"})
    return out


#: degree of the polynomial in log separation and log contrast that the seam steps are read against
SEAM_FIT_DEGREE = 3


def seam_steps(curve: Dict[str, Any], seps: Sequence[Tuple[int, int, int]], order: Sequence[int],
               degree: int = SEAM_FIT_DEGREE) -> Dict[str, Any]:
    """Return each annulus' curve over a smooth fit of the whole curve, and the step at each seam.

    The seam line (:func:`seam_ratios`) holds the curve's own slope inside a tile, which is steep inside
    0.8".  Here a polynomial in log separation and log contrast is fitted to every point of the curve,
    and each point is read over it.  A seam then shows as a step from one tile's outer edge to the next
    tile's inner edge, the slope between them taken out (NEAR2 note, 2026-11-17).

    Parameters
    ----------
    curve : dict
        The curve, :func:`klip_tpe.contrast_grid.summarize`'s, with ``r_as`` and ``curve`` per separation.
    seps : sequence of tuple
        ``(annulus, band, n_bands)`` per separation of the curve, the annulus counted from 0.
    order : sequence of int
        The annuli in the stitch's order.
    degree : int, default 3
        Degree of the polynomial, lowered to the number of points less 2 when there are fewer.

    Returns
    -------
    steps : dict
        ``degree`` (None when fewer than 4 points have a value), ``rms`` (the points' scatter about the
        fit, as a fraction) and ``annuli``, per annulus in the stitch's order ``annulus`` (from 1),
        ``inner``, ``middle`` and ``outer`` (the curve over the fit, None where there is no point) and
        ``step`` (the next annulus' ``inner`` over this one's ``outer``, None at the last or without
        both).
    """
    r = np.asarray(curve.get("r_as") or [], float)
    c = np.array([np.nan if v is None else float(v) for v in curve.get("curve") or []], float)
    m = min(len(r), len(c))
    r, c = r[:m], c[:m]
    good = np.isfinite(r) & np.isfinite(c) & (r > 0) & (c > 0)
    out: Dict[str, Any] = {"degree": None, "rms": None, "annuli": []}
    if int(good.sum()) < 4:
        return out
    deg = int(min(int(degree), int(good.sum()) - 2))
    lr, lc = np.log10(r[good]), np.log10(c[good])
    coef = np.polyfit(lr, lc, deg)
    ratio = np.full(m, np.nan)
    ratio[good] = 10.0 ** (lc - np.polyval(coef, lr))
    out["degree"] = deg
    out["rms"] = float(10.0 ** np.sqrt(np.mean((lc - np.polyval(coef, lr)) ** 2)) - 1.0)
    for a in order:
        mine = {b: i for i, (aa, b, _) in enumerate(seps) if aa == a and i < m}
        n = next((nn for aa, _, nn in seps if aa == a), 0)

        def at(b: int) -> Optional[float]:
            i = mine.get(b)
            return None if i is None or not np.isfinite(ratio[i]) else float(ratio[i])
        out["annuli"].append({"annulus": int(a) + 1, "inner": at(0) if n > 1 else None,
                              "middle": at(n // 2) if n else None, "outer": at(n - 1) if n > 1 else None,
                              "step": None})
    an = out["annuli"]
    for k in range(len(an) - 1):
        lo, hi = an[k]["outer"], an[k + 1]["inner"]
        an[k]["step"] = None if lo is None or hi is None else hi / lo
    return out


def _step_text(s: Dict[str, Any], nxt: Optional[int]) -> str:
    """Describe an annulus' curve over the smooth fit in words.

    Parameters
    ----------
    s : dict
        The annulus' entry of :func:`seam_steps`.
    nxt : int or None
        The next annulus in the stitch's order, from 1, None at the last.

    Returns
    -------
    text : str
        Such as ``inner 1.05, middle 0.98, outer 1.02, a step of 1.08 into annulus 3``.
    """
    return (f"inner {_f(s['inner'])}, middle {_f(s['middle'])}, outer {_f(s['outer'])}"
            + (f", a step of {_f(s['step'])} into annulus {nxt}" if nxt is not None and s["step"] is not None else ""))


# ----------------------------------------------------------------------------- caches
class _Readings:
    """Store of the readings, by key, kept in a JSON-lines file each new one is added to.

    Parameters
    ----------
    path : str or None
        Path of the file, None to keep the readings in memory alone.
    """

    def __init__(self, path: Optional[str]):
        self.path = path
        self.data: Dict[str, Tuple[float, Optional[str]]] = {}
        self._f = None
        if path and os.path.exists(path):
            with open(path) as f:
                for ln in f:
                    try:
                        d = json.loads(ln)
                    except ValueError:
                        continue
                    if isinstance(d, dict) and "k" in d:
                        s = d.get("s")
                        self.data[str(d["k"])] = (float("nan") if s is None else float(s), d.get("w"))
        self.n_loaded = len(self.data)

    def __contains__(self, key: str) -> bool:
        """Tell whether a reading is known.

        Parameters
        ----------
        key : str
            The reading's key.

        Returns
        -------
        known : bool
            True when the store holds it.
        """
        return key in self.data

    def get(self, key: str) -> Tuple[float, Optional[str]]:
        """Return a reading and why it is NaN, if it is.

        Parameters
        ----------
        key : str
            The reading's key.

        Returns
        -------
        snr : float
            The reading.
        why : str or None
            Why it is NaN.
        """
        return self.data[key]

    def put(self, key: str, snr: float, why: Optional[str], keep: bool = True) -> None:
        """Add a reading.

        Parameters
        ----------
        key : str
            The reading's key.
        snr : float
            The reading.
        why : str or None
            Why it is NaN.
        keep : bool, default True
            Whether it goes to the file, as every reading but a failed reduction's does.
        """
        self.data[key] = (float(snr), why)
        if keep and self.path:
            if self._f is None:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                self._f = open(self.path, "a")
            self._f.write(json.dumps({"k": key, "s": float(snr) if np.isfinite(snr) else None, "w": why}) + "\n")

    def flush(self) -> None:
        """Write the readings added to the disk.
        """
        if self._f is not None:
            self._f.flush()
            os.fsync(self._f.fileno())

    def close(self) -> None:
        """Close the file.
        """
        if self._f is not None:
            self.flush()
            self._f.close()
            self._f = None


class _Images:
    """Store of reduced images for later readings, the oldest dropped past a number.

    Parameters
    ----------
    max_n : int
        Number of images kept at most.
    """

    def __init__(self, max_n: int):
        self.max_n = max(int(max_n), 1)
        self.data: "OrderedDict[Any, np.ndarray]" = OrderedDict()

    def get(self, key) -> Optional[np.ndarray]:
        """Return an image, None when not kept.

        Parameters
        ----------
        key : tuple
            The image's key.

        Returns
        -------
        image : ndarray or None
            The image.
        """
        img = self.data.get(key)
        if img is not None:
            self.data.move_to_end(key)
        return img

    def put(self, key, img: np.ndarray) -> None:
        """Keep an image, dropping the oldest past the number.

        Parameters
        ----------
        key : tuple
            The image's key.
        img : ndarray
            The image.
        """
        self.data[key] = img
        self.data.move_to_end(key)
        while len(self.data) > self.max_n:
            self.data.popitem(last=False)


@dataclass
class _Job:
    """One reading to make, a point of a grid at a contrast under an assignment of the annuli.

    Attributes
    ----------
    key : str
        The reading's key.
    grid : str
        The grid, "elect" or "curve".
    point : GridPoint
        The point.
    c : float
        The contrast.
    slots : dict of int to str
        The configuration key of each annulus reduced with the source.
    bg : str
        The background, the clean images of the other annuli, "start" or "final".
    how : str
        The measure, "path_a" or "grid".
    """
    key: str
    grid: str
    point: GridPoint
    c: float
    slots: Dict[int, str]
    bg: str
    how: str


def _short(key: str) -> str:
    """Return a short hash of a configuration key.

    Parameters
    ----------
    key : str
        The key (:func:`klip_tpe.refine.config_key`).

    Returns
    -------
    short : str
        Its first 16 hexadecimal digits of SHA-1.
    """
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _ev1(v) -> str:
    """Return an evaluation as the text reports count it, from 1.

    Parameters
    ----------
    v : int or None
        The evaluation, counted from 0.

    Returns
    -------
    text : str
        The evaluation counted from 1, "-" for None.
    """
    return "-" if v is None else str(int(v) + 1)


def _f(v, fmt: str = "{:.2f}") -> str:
    """Format a number, "-" where it is None or not finite.

    Parameters
    ----------
    v : float or None
        The value.
    fmt : str, default "{:.2f}"
        The format.

    Returns
    -------
    text : str
        The value formatted.
    """
    return fmt.format(v) if v is not None and np.isfinite(v) else "-"


def _nanmean(v) -> float:
    """Return the mean of the finite values, NaN without any.

    Parameters
    ----------
    v : array_like
        The values.

    Returns
    -------
    mean : float
        The mean.
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _brief(e: Entrant) -> Dict[str, Any]:
    """Return a finalist's run, kind and evaluation.

    Parameters
    ----------
    e : Entrant
        The finalist.

    Returns
    -------
    brief : dict
        ``run``, ``kind`` and ``eval_index`` (counted from 0).
    """
    return {"run": e.run, "kind": e.kind, "eval_index": e.eval_index}


def _who(e: Entrant) -> str:
    """Name a finalist for the log, its run, kind and evaluation counted from 1.

    Parameters
    ----------
    e : Entrant
        The finalist.

    Returns
    -------
    text : str
        Such as ``run_a validation eval 12``.
    """
    return f"{e.run} {e.kind} eval {_ev1(e.eval_index)}"


class FinalGrid:
    """The final-image grid's points, readings and steps, on a runner whose annuli are all done.

    Parameters
    ----------
    runner : Runner
        The run.
    scorer : StitchScorer
        The run's stitch (:class:`klip_tpe.refine.StitchScorer`), for its tiles, pads and zones.
    config : FinalGridConfig
        The settings.
    finalists : dict of int to list of Entrant
        Each annulus' finalists by 0-based annulus, its starting winner first.
    start : dict of int to Entrant
        Each annulus' starting winner, the validation's.
    state_dir : str or None
        Directory of the readings kept for a later start, None to keep them in memory alone.
    weights : sequence of float or None, default None
        The stitch weights of every reading.  None works them out from each background's clean
        images, as the final stitch does.
    log : callable, default print
        Function that takes each line of the log.
    window : int or None, default None
        Number of reductions in flight at most.  None takes the runner's.
    card : bool, default False
        Whether the injected reductions may go to the card workers.
    stage : callable, default a no-op
        Function that takes a short status line, for the run's heartbeat.

    Attributes
    ----------
    points : dict of str to list of GridPoint
        The election's grid (``"elect"``) and the curve's (``"curve"``).
    reach : dict of str to list of list of int
        Per grid and point, the annuli reduced with its source.
    cover : dict of str to dict of int to list of int
        Per grid and annulus, the points that count toward its election.
    calibration : dict of int to dict
        Each tile's ``contrast``, ``converged``, ``history`` and calibration ``points``.
    sweeps : list of list of dict
        Each sweep's rows, one per annulus elected.
    assign : dict of int to str
        Each annulus' current winner, by configuration key.
    n_reductions : int
        Number of reductions made.
    """

    def __init__(self, runner, scorer: StitchScorer, config: FinalGridConfig, finalists: Dict[int, List[Entrant]],
                 start: Dict[int, Entrant], state_dir: Optional[str], weights: Optional[Sequence[float]] = None,
                 log: Callable[[str], None] = print, window: Optional[int] = None, card: bool = False,
                 stage: Callable[[str], None] = lambda s: None):
        self.r = runner
        self.sc = scorer
        self.fc = config
        self.log = log
        self.stage = stage
        self.card = bool(card)
        self.window = int(window) if window else None
        self.px = float(runner.pxscale)
        self.fwhm = float(runner.fwhm)
        self.order = list(scorer.order)
        self.regions = stitch_regions(scorer.itiles, scorer.pads)
        self.finalists = {a: list(v) for a, v in finalists.items()}
        self.entrant = {(a, e.key): e for a, v in finalists.items() for e in v}
        self.start = {a: e.key for a, e in start.items()}
        self.configs: Dict[str, Dict[str, Any]] = {}
        for v in finalists.values():
            for e in v:
                self.configs.setdefault(e.key, e.config)
        self.fixed_weights = None if weights is None else np.asarray(weights, float)
        self.state_dir = state_dir
        self.readings = _Readings(None)
        self.images = _Images(256)
        self.n_reductions = 0
        self._lock = threading.Lock()
        self._cfgs: Dict[str, Any] = {}
        self._cleans: Dict[Tuple[int, str], np.ndarray] = {}
        self._geo: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self._clean_st: Dict[str, np.ndarray] = {}
        self.bgs: Dict[str, Dict[str, Any]] = {}
        self._rules = runner._ring_rules()
        self._side = runner._verify_rule()
        self._exr = float(runner._excl_fwhm()) * self.fwhm
        metric = getattr(runner.objective, "metric", None)
        self.floor = max(float(getattr(metric, "inner_fwhm", 1.0) or 1.0) * self.fwhm, self.fwhm)
        tiles = [(a, float(t[0]), float(t[1])) for a, t in zip(self.order, scorer.tiles)]
        self.points = {g: grid_points(tiles, self.fwhm, config.n_pa, config.sep_step_fwhm, self.floor, turn)
                       for g, turn in (("elect", 0.0), ("curve", 0.5))}
        gc = runner.cfg.contrast_grid
        smp = getattr(runner.sampler, "base", None) or runner.sampler
        self.n_excluded = {g: (exclude_known(pts, scorer.known, self.px, self.fwhm, gc)
                               + exclude_sampled(pts, smp, self.px)) for g, pts in self.points.items()}
        self.read_px = max(READ_FWHM, float(config.cover_fwhm)) * self.fwhm
        cover_px = float(config.cover_fwhm) * self.fwhm
        self.reach: Dict[str, List[List[int]]] = {}
        self.cover: Dict[str, Dict[int, List[int]]] = {}
        for g, pts in self.points.items():
            self.reach[g] = [[self.order[i] for i in touched(p.r_px, self.regions, self.read_px)] for p in pts]
            cov: Dict[int, List[int]] = {a: [] for a in self.order}
            for p in pts:
                if p.status == "active":
                    for i in touched(p.r_px, self.regions, cover_px):
                        cov[self.order[i]].append(p.index)
            self.cover[g] = cov
        self.calibration: Dict[int, Dict[str, Any]] = {}
        self.sweeps: List[List[Dict[str, Any]]] = []
        self.assign: Dict[int, str] = dict(self.start)
        self.last_change: Dict[int, Dict[str, Any]] = {}
        self.held: Dict[int, Dict[str, Any]] = {}

    # ---------------------------------------------------------------- plan and state
    def plan(self) -> Dict[str, Any]:
        """Count the points and the reductions of the stage, without making any.

        Returns
        -------
        plan : dict
            ``annuli`` (per 0-based annulus its ``finalists``, ``points`` in its tile, ``cover`` and the
            ``reductions`` of a sweep's election without readings kept), ``election_sweep`` (their sum),
            ``calibration_step`` (the reductions of a step of every tile's calibration), ``curve_wave``
            (of a wave of the curve with every point active), ``start_cleans``, ``points`` and
            ``curve_points`` (the active ones) and ``excluded`` (per grid).
        """
        per = {}
        for a in self.order:
            cov = self.cover["elect"][a]
            nf = len(self.finalists.get(a, []))
            nb = sum(len(self.reach["elect"][i]) - 1 for i in cov)
            per[a] = {"finalists": nf, "cover": len(cov), "reductions": nf * len(cov) + nb,
                      "points": sum(1 for p in self.points["elect"] if p.annulus == a and p.status == "active")}
        cal = 0
        for a in self.order:
            pts = [p for p in self.points["elect"] if p.annulus == a and p.status == "active"]
            for p in pts[:: max(1, len(pts) // max(int(self.fc.n_cal), 1))][:int(self.fc.n_cal)]:
                cal += len(self.reach["elect"][p.index])
        act = [p for p in self.points["curve"] if p.status == "active"]
        return {"annuli": per, "election_sweep": int(sum(v["reductions"] for v in per.values())),
                "calibration_step": int(cal), "curve_wave": int(sum(len(self.reach["curve"][p.index]) for p in act)),
                "start_cleans": len(self.order), "excluded": dict(self.n_excluded),
                "points": sum(1 for p in self.points["elect"] if p.status == "active"), "curve_points": len(act)}

    def fingerprint(self) -> Dict[str, Any]:
        """Return what the kept readings depend on beside their keys, the stitch, the frames and the measure.

        Returns
        -------
        fingerprint : dict
            The zones, tiles, pads, FWHM, plate scale, frames, flattening, known companions, keep-out,
            ring rules, the grid's layout, the contrast grid's settings and the fixed weights.
        """
        from .runner import _core_edits, _describe_edits
        gc = asdict(self.r.cfg.contrast_grid)
        gc.pop("mode", None)
        # a run saved before 2026-11-25 sums whole pixels in its apertures, as its kept readings did, and
        # keeps the fingerprint it had before the key existed
        if str(gc.get("aperture")) == "whole":
            gc.pop("aperture")
        edits = dict(getattr(self.r.reducer, "data_edits", None) or {})
        # a run saved before 2026-11-25 reads sources at their peak (RunConfig.verify_position "peak"), as
        # its kept readings were made, and keeps the fingerprint it had before the key existed
        rk = {k: v for k, v in (self._rules or {}).items() if not (k == "position" and str(v) == "peak")}
        rules = hashlib.sha1(json.dumps(rk, sort_keys=True, default=str).encode()).hexdigest()
        return {"version": 1, "zones": [list(z) for z in self.sc.zones], "tiles": [list(t) for t in self.sc.tiles],
                "itiles": [list(t) for t in self.sc.itiles], "pads": [list(p) for p in self.sc.pads],
                "fwhm": self.fwhm, "pxscale": self.px, "frames": _describe_edits(_core_edits(edits)),
                "flatten": bool(self.r.flatten_products), "known": [list(k) for k in self.sc.known],
                "excl_px": self._exr, "verify_rule": repr(self._side), "rules": rules, "n_pa": int(self.fc.n_pa),
                "sep_step_fwhm": float(self.fc.sep_step_fwhm), "floor_px": self.floor, "read_px": self.read_px,
                "contrast_grid": gc,
                "weights": None if self.fixed_weights is None else [float(v) for v in self.fixed_weights]}

    def open_state(self) -> None:
        """Read the readings kept in the state directory, or start it afresh.

        A directory started on another stitch, other frames or another measure is moved aside to
        ``state_<time>`` and the stage starts afresh, saying so.
        """
        if not self.state_dir:
            return
        fp = self.fingerprint()
        opt = os.path.join(self.state_dir, "options.json")
        if os.path.exists(opt):
            try:
                with open(opt) as f:
                    old = json.load(f)
            except (OSError, ValueError):
                old = None
            if json.loads(json.dumps(fp, default=float)) != old:
                aside = self.state_dir.rstrip(os.sep) + "_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                os.replace(self.state_dir, aside)
                self.log(f"  final grid: {self.state_dir} was started on another stitch, frames or measure, so it "
                         f"moved to {os.path.basename(aside)} and the stage starts afresh")
        os.makedirs(self.state_dir, exist_ok=True)
        if not os.path.exists(opt):
            tmp = opt + ".tmp"
            with open(tmp, "w") as f:
                json.dump(fp, f, indent=1, default=float)
            os.replace(tmp, opt)
        self.readings = _Readings(os.path.join(self.state_dir, "readings.jsonl"))
        if self.readings.n_loaded:
            self.log(f"  final grid: {self.readings.n_loaded} readings kept from the last start")

    def close(self) -> None:
        """Close the readings' file.
        """
        self.readings.close()

    # ---------------------------------------------------------------- reductions
    def _window(self) -> int:
        """Return the number of reductions in flight.

        Returns
        -------
        window : int
            The window given, else the runner's.
        """
        return int(self.window or self.r._reduction_window())

    def _cfg(self, key: str):
        """Return a finalist's configuration as the runner reduces it.

        Parameters
        ----------
        key : str
            The configuration's key.

        Returns
        -------
        cfg : Config
            The configuration.
        """
        c = self._cfgs.get(key)
        if c is None:
            c = self._cfgs[key] = self.r._config_from_dict(self.configs[key])
        return c

    def _reduce(self, a: int, key: str, src: Optional[Source], tag: str) -> np.ndarray:
        """Reduce a configuration in an annulus' zone, with one source or none, without a KLIP-FM pass.

        Parameters
        ----------
        a : int
            The annulus, counted from 0.
        key : str
            The configuration's key.
        src : Source or None
            The source, None for the clean image.
        tag : str
            Label of the reduction.

        Returns
        -------
        image : ndarray, shape (ny, nx)
            The image.

        Raises
        ------
        ValueError
            If the reduction returns no image.
        """
        cfg = self._cfg(key)
        more = {"card": True} if (self.card and src is not None) else {}
        ev = self.r._reduce(cfg, None if src is None else [src], tag=tag, zone=self.r._zone(a, cfg), fm_sources=[],
                            **more)
        with self._lock:
            self.n_reductions += 1
        if ev.image is None:
            raise ValueError(f"annulus {a + 1}: the reduction returned no image")
        return np.asarray(ev.image, float)

    def _need_cleans(self, pairs: Sequence[Tuple[int, str]]) -> None:
        """Reduce the clean images not yet made of some annuli' configurations.

        Parameters
        ----------
        pairs : sequence of tuple
            ``(annulus, key)`` per clean image.

        Raises
        ------
        Exception
            The exception of a failed reduction.
        """
        todo = [p for p in dict.fromkeys(pairs) if p not in self._cleans]
        calls = [(lambda a=a, k=k: self._reduce(a, k, None, f"fg_a{a + 1}_clean")) for a, k in todo]
        for i, res in self.r._run_ordered(calls, self._window()):
            if isinstance(res, BaseException):
                raise res
            self._cleans[todo[i]] = res
            per = max(int(IMAGE_BUDGET // max(res.nbytes, 1)), 64)
            if per != self.images.max_n:
                self.images.max_n = per

    def background(self, name: str, assign: Dict[int, str]) -> None:
        """Set a background, the clean images of an assignment of the annuli and its stitch weights.

        Parameters
        ----------
        name : str
            The background's name, "start" or "final".
        assign : dict of int to str
            Each annulus' configuration key.
        """
        self._need_cleans([(a, assign[a]) for a in self.order])
        imgs = {a: self._cleans[(a, assign[a])] for a in self.order}
        if self.fixed_weights is not None:
            w = self.fixed_weights
        else:
            w = np.asarray(self.r._stitch_weights([imgs[a] for a in self.order], self.sc.zones), float)
        ident = hashlib.sha1(json.dumps([[a, _short(assign[a])] for a in self.order]
                                        + [[float(v) for v in w]]).encode()).hexdigest()[:16]
        self.bgs[name] = {"assign": dict(assign), "images": imgs, "weights": w, "id": ident}
        self._clean_st.pop(name, None)

    # ---------------------------------------------------------------- readings
    def job(self, grid: str, p: GridPoint, c: float, assign: Dict[int, str], bg: str, how: str) -> _Job:
        """Return the reading of a point at a contrast under an assignment of the annuli.

        Parameters
        ----------
        grid : {"elect", "curve"}
            The grid.
        p : GridPoint
            The point.
        c : float
            The contrast.
        assign : dict of int to str
            Each annulus' configuration key.  Those reduced with the source are the point's ``reach``.
        bg : str
            The background of the other annuli (:meth:`background`).
        how : {"path_a", "grid"}
            The measure.

        Returns
        -------
        job : _Job
            The reading, its key made of the grid, the point, the contrast, the measure, the reduced
            annuli's configurations and the background.
        """
        slots = {a: assign[a] for a in self.reach[grid][p.index]}
        key = hashlib.sha1(json.dumps([grid, int(p.index), repr(float(c)), how,
                                       [[a, _short(k)] for a, k in sorted(slots.items())],
                                       self.bgs[bg]["id"]]).encode()).hexdigest()
        return _Job(key, grid, p, float(c), slots, bg, how)

    def value(self, job: _Job) -> float:
        """Return a reading made.

        Parameters
        ----------
        job : _Job
            The reading.

        Returns
        -------
        snr : float
            The reading, NaN where it gave none.
        """
        return float(self.readings.get(job.key)[0])

    def compute(self, jobs: Sequence[_Job], label: str = "") -> int:
        """Make the readings not yet known, each image reduced once and kept while a reading needs it.

        A reading is measured as soon as its last image is in, and a reduction that fails gives its
        readings NaN, which are not kept for a later start.

        Parameters
        ----------
        jobs : sequence of _Job
            The readings, best ordered point by point.
        label : str, default ""
            What they are, for the heartbeat.

        Returns
        -------
        n : int
            Number of reductions made.
        """
        import contextlib
        todo: List[_Job] = []
        seen = set()
        for j in jobs:
            if j.key not in seen and j.key not in self.readings:
                seen.add(j.key)
                todo.append(j)
        if not todo:
            return 0
        n0 = self.n_reductions
        calls: List[Callable[[], Any]] = []
        iks: List[Any] = []
        where: Dict[Any, int] = {}
        held: Dict[Any, Optional[np.ndarray]] = {}
        need: List[List[Tuple[int, Any]]] = []
        last: List[int] = []
        for j in todo:
            mine, mx = [], -1
            for a, k in sorted(j.slots.items()):
                ik = (j.grid, j.point.index, a, k, repr(j.c))
                mine.append((a, ik))
                if ik in where:
                    mx = max(mx, where[ik])
                    continue
                if ik in held:
                    continue
                img = self.images.get(ik)
                if img is not None:
                    held[ik] = img
                    continue
                where[ik] = len(calls)
                calls.append(lambda a=a, k=k, src=j.point.source(self.px, j.c),
                             t=f"fg_{j.grid}_a{a + 1}_p{j.point.index}": self._reduce(a, k, src, t))
                iks.append(ik)
                mx = len(calls) - 1
            need.append(mine)
            last.append(mx)
        refs = Counter(ik for mine in need for _, ik in mine)
        ready: Dict[int, List[int]] = defaultdict(list)
        for i, mx in enumerate(last):
            ready[mx].append(i)
        failed: Dict[Any, str] = {}

        def finish(i: int) -> None:
            j = todo[i]
            why = next((failed[ik] for _, ik in need[i] if ik in failed), None)
            if why is not None:
                self.readings.put(j.key, float("nan"), why, keep=False)
            else:
                snr, w = self._measure(j, {a: held[ik] for a, ik in need[i]})
                self.readings.put(j.key, snr, w)
            for _, ik in need[i]:
                refs[ik] -= 1
                if refs[ik] <= 0:
                    held.pop(ik, None)

        for i in ready.pop(-1, []):
            finish(i)
        t0 = last_t = time.time()
        with contextlib.closing(self.r._run_ordered(calls, self._window())) as stream:
            for ci, res in stream:
                ik = iks[ci]
                if isinstance(res, BaseException):
                    failed[ik] = f"reduction failed: {type(res).__name__}: {str(res)[:80]}"
                    held[ik] = None
                else:
                    held[ik] = res
                    self.images.put(ik, res)
                for i in ready.pop(ci, []):
                    finish(i)
                if time.time() - last_t > 20.0:
                    last_t = time.time()
                    self.stage(f"{label}: {ci + 1} of {len(calls)} reductions ({(last_t - t0) / 60:.0f} min)")
        self.readings.flush()
        if failed:
            self.log(f"  final grid: {len(failed)} reduction(s) of {label} failed ({next(iter(failed.values()))}), "
                     f"their readings left out")
        return self.n_reductions - n0

    def _stitch(self, bg: str, imgs: Dict[int, np.ndarray]) -> np.ndarray:
        """Build a stitch from some annuli's images and the background's clean images for the rest.

        Parameters
        ----------
        bg : str
            The background.
        imgs : dict of int to ndarray
            The images, by 0-based annulus.

        Returns
        -------
        stitch : ndarray, shape (ny, nx)
            The stitch, built as the final stitch builds it, flattened where the run's products are.
        """
        from .stitch import stitch_tiles
        b = self.bgs[bg]
        st = stitch_tiles([imgs.get(a, b["images"][a]) for a in self.order], self.sc.itiles, b["weights"],
                          pads=self.sc.pads)
        return self.r._maybe_flatten(st)

    def _clean_stitch(self, bg: str) -> np.ndarray:
        """Return a background's clean stitch, for a ring rule that reads the clean image.

        Parameters
        ----------
        bg : str
            The background.

        Returns
        -------
        stitch : ndarray, shape (ny, nx)
            The stitch of the background's clean images.
        """
        if bg not in self._clean_st:
            self._clean_st[bg] = self._stitch(bg, {})
        return self._clean_st[bg]

    def path_a(self, img: np.ndarray, src: Source) -> float:
        """Measure a source alone on a stitch as the stitched verify's Path A measures it.

        It is the S/N :func:`klip_tpe.verify.verify_injections` gives a block of one source, made without
        that function's maps and tables, which a single S/N does not need.

        Parameters
        ----------
        img : ndarray, shape (ny, nx)
            The stitch.
        src : Source
            The source.

        Returns
        -------
        snr : float
            The S/N, NaN where none.
        """
        from .verify import _known_xy, ring_snr, search_px, snr_fap, star_center
        img = np.asarray(img, float)
        cx, cy = star_center(img.shape)
        conv = self.r.reducer.angle_convention
        kxy = _known_xy(self.sc.known, self.px, cx, cy, conv)
        # an injection's position is given: read at it (RunConfig.verify_position), or for a run saved
        # before 2026-11-25 at the peak within 1.5 px
        srad = search_px(self.fwhm, False, self._rules)
        if self._side is None:
            s = snr_fap(img, float(src.rho), float(src.theta), self.px, self.fwhm, self._exr, srad, exclude_xy=kxy,
                        angle_convention=conv, rules=self._rules)
        else:
            s = ring_snr(img, 0, [float(src.rho)], [float(src.theta)], self.px, self.fwhm, self._side, exr=self._exr,
                         srad=srad, exclude_xy=kxy, known=list(self.sc.known), angle_convention=conv, rules=self._rules)
        v = s.get("snr")
        return float("nan") if v is None else float(v)

    def _geometry(self, a: int, key: str) -> Dict[str, Any]:
        """Return the contrast grid's measuring settings for an annulus' configuration (:meth:`Runner._grid_geometry`).

        Parameters
        ----------
        a : int
            The annulus, counted from 0.
        key : str
            The configuration's key.

        Returns
        -------
        geometry : dict
            The settings.
        """
        g = self._geo.get((a, key))
        if g is None:
            g = self._geo[(a, key)] = self.r._grid_geometry(self._cfg(key))
        return g

    def _measure(self, job: _Job, imgs: Dict[int, np.ndarray]) -> Tuple[float, Optional[str]]:
        """Measure a reading on its stitch.

        Parameters
        ----------
        job : _Job
            The reading.
        imgs : dict of int to ndarray
            The reduced annuli's images with its source.

        Returns
        -------
        snr : float
            The reading, NaN where none.
        why : str or None
            Why it is NaN.
        """
        img = self._stitch(job.bg, imgs)
        if job.how == "path_a":
            v = self.path_a(img, job.point.source(self.px, job.c))
            return v, (None if np.isfinite(v) else "no S/N")
        a = job.point.annulus
        geo = self._geometry(a, job.slots.get(a, self.bgs[job.bg]["assign"][a]))
        snr, d = self.r._grid_snr(geo, img, job.point.r_px, job.point.theta, clean=self._clean_stitch(job.bg))
        if d.get("why"):
            return float("nan"), str(d["why"])
        n = int(d.get("nclean", 0) or 0)
        if n < int(geo["min_ring"]):
            return float("nan"), f"{n} noise apertures, fewer than {int(geo['min_ring'])}"
        return float(snr), None

    # ---------------------------------------------------------------- the stage's steps
    def _start_contrast(self, a: int) -> float:
        """Return a tile's first contrast, the run's injection contrast scaled by its winner's S/N to the target.

        Parameters
        ----------
        a : int
            The annulus, counted from 0.

        Returns
        -------
        contrast : float
            The contrast, 1e-4 without one, at most 0.1.
        """
        res = self.sc.results[a]
        c = float(res.contrast or 0.0)
        if not (np.isfinite(c) and c > 0):
            c = 1e-4
        s = float(res.winner_score) if res.winner_score is not None else float("nan")
        mid = 0.5 * (float(self.fc.target[0]) + float(self.fc.target[1]))
        if np.isfinite(s) and s > 0.5:
            c *= mid / s
        return float(min(c, 0.1))

    def contrast(self, a: int) -> float:
        """Return a tile's contrast, from its calibration.

        Parameters
        ----------
        a : int
            The annulus, counted from 0.

        Returns
        -------
        contrast : float
            The contrast.
        """
        return float(self.calibration[a]["contrast"])

    def calibrate(self) -> Dict[int, Dict[str, Any]]:
        """Find each tile's contrast, at which the starting winners' median reading of its points lies in the target.

        Each tile is calibrated on up to ``n_cal`` of its points spread over the tile, the starting winners
        in every annulus, by the steps :meth:`klip_tpe.refine.StitchScorer._next_contrast` takes.  Without
        convergence the contrast is the next one, not measured.

        Returns
        -------
        calibration : dict of int to dict
            Per 0-based annulus its ``contrast``, ``converged``, ``history`` (each step's ``contrast``,
            ``median_snr`` and ``n``) and ``points`` (the indices of the calibration's points).
        """
        lo, hi = float(self.fc.target[0]), float(self.fc.target[1])
        for a in self.order:
            pts = [p for p in self.points["elect"] if p.annulus == a and p.status == "active"]
            if not pts:
                self.calibration[a] = {"contrast": self._start_contrast(a), "converged": False, "history": [],
                                       "points": []}
                continue
            idx = sorted({int(round(v)) for v in np.linspace(0, len(pts) - 1, min(int(self.fc.n_cal), len(pts)))})
            cal = [pts[i] for i in idx]
            c = self._start_contrast(a)
            hist: List[Dict[str, Any]] = []
            ok = False
            for it in range(int(self.fc.max_cal)):
                jobs = [self.job("elect", p, c, self.start, "start", "path_a") for p in cal]
                self.compute(jobs, f"the calibration of tile {a + 1}")
                v = np.array([self.value(j) for j in jobs])
                m = _nanmedian(v)
                hist.append({"contrast": c, "median_snr": m, "n": int(np.isfinite(v).sum())})
                self.log(f"  final grid: tile {a + 1}: contrast {c:.3e} -> median S/N {m:.2f} on the stitch "
                         f"({len(cal)} points, step {it + 1})")
                if lo <= m <= hi:
                    ok = True
                    break
                c = float(StitchScorer._next_contrast(hist, (lo, hi)))
            if not ok:
                self.log(f"  final grid: tile {a + 1}: not inside S/N {lo:g}-{hi:g} after {len(hist)} steps, taking "
                         f"{c:.3e}")
            self.calibration[a] = {"contrast": float(c), "converged": ok, "history": hist,
                                   "points": [p.index for p in cal]}
        return self.calibration

    def _elect_annulus(self, a: int, sweep: int) -> Dict[str, Any]:
        """Elect one annulus' winner, its finalists read in its slot with the others at their current winners.

        Parameters
        ----------
        a : int
            The annulus, counted from 0.
        sweep : int
            The sweep, counted from 0.

        Returns
        -------
        row : dict
            ``annulus`` (from 1), ``sweep`` (from 1), ``before`` and ``after`` (the winner going in and
            out, :func:`_brief`), ``changed``, ``n_points``, ``reductions`` and ``finalists``, each with its
            ``run``, ``kind``, ``eval_index``, mean reading ``mean``, ``n`` read, and paired ``gain``,
            ``se``, ``median_gain``, ``p_better`` and ``n_paired`` against the winner going in.
        """
        pts = [self.points["elect"][i] for i in self.cover["elect"][a]]
        keys = [e.key for e in self.finalists[a]]
        before = self.assign[a]
        jobs = {k: [self.job("elect", p, self.contrast(p.annulus), {**self.assign, a: k}, "start", "path_a")
                    for p in pts] for k in keys}
        n_red = self.compute([jobs[k][i] for i in range(len(pts)) for k in keys],
                             f"sweep {sweep + 1}, annulus {a + 1}")
        R = {k: np.array([self.value(j) for j in jobs[k]], float) for k in keys}
        after, stats = elect_one(R, keys, before, float(self.fc.z)) if pts else (before, {})
        self.assign[a] = after
        fin = []
        for k in keys:
            e = self.entrant[(a, k)]
            st = stats.get(k) or {}
            fin.append(dict(_brief(e), mean=_nanmean(R[k]), n=int(np.isfinite(R[k]).sum()),
                            gain=st.get("diff_mean"), se=st.get("diff_se"), median_gain=st.get("diff_median"),
                            p_better=st.get("p_better"), n_paired=st.get("n_paired"),
                            current=k == before))
        row = {"annulus": a + 1, "sweep": sweep + 1, "before": _brief(self.entrant[(a, before)]),
               "after": _brief(self.entrant[(a, after)]), "changed": after != before, "n_points": len(pts),
               "reductions": int(n_red), "finalists": fin}
        eb, ea = self.entrant[(a, before)], self.entrant[(a, after)]
        if after != before:
            s = stats[after]
            self.last_change[a] = {"sweep": sweep + 1, "from": _brief(eb), "gain": s["diff_mean"], "se": s["diff_se"],
                                   "n": s["n_paired"]}
            self.log(f"  final grid: sweep {sweep + 1}, annulus {a + 1}: {_who(ea)} replaces {_who(eb)}, "
                     f"{s['diff_mean']:+.2f} +/- {_f(s['diff_se'])} S/N over {s['n_paired']} points "
                     f"(z = {float(self.fc.z):g}, {n_red} reductions)")
        else:
            others = [k for k in keys if k != before and stats.get(k) and np.isfinite(stats[k]["diff_mean"])]
            best = max(others, key=lambda k: stats[k]["diff_mean"]) if others else None
            tail = ("" if best is None else
                    f", the best challenger {_who(self.entrant[(a, best)])} at {stats[best]['diff_mean']:+.2f} +/- "
                    f"{_f(stats[best]['diff_se'])}")
            self.log(f"  final grid: sweep {sweep + 1}, annulus {a + 1}: {_who(eb)} stays over {len(pts)} points"
                     f"{tail} ({n_red} reductions)")
        return row

    def elect(self) -> List[List[Dict[str, Any]]]:
        """Run the election's sweeps, inner to outer and back, until a sweep changes no winner.

        Returns
        -------
        sweeps : list of list of dict
            Each sweep's rows (:meth:`_elect_annulus`).
        """
        self.assign = dict(self.start)
        self.sweeps = []
        self.last_change = {}
        n = int(self.fc.sweeps)
        for s in range(n):
            order = self.order if s % 2 == 0 else self.order[::-1]
            rows = [self._elect_annulus(a, s) for a in order]
            self.sweeps.append(rows)
            if not any(r["changed"] for r in rows):
                self.log(f"  final grid: sweep {s + 1} changed no winner"
                         + (", so the election stops" if s + 1 < n else ""))
                break
        return self.sweeps

    def held_out(self) -> Dict[int, Dict[str, Any]]:
        """Read each changed annulus' winner and its starting winner on the curve's grid, at the tiles' contrasts.

        The other annuli are at their elected winners.  The gain is the elected winner's paired gain over
        the starting winner on points the election did not read.

        Returns
        -------
        held_out : dict of int to dict
            Per changed 0-based annulus its ``gain``, ``se``, ``median_gain``, ``p_better`` and ``n``.
        """
        self.held = {}
        changed = [a for a in self.order if self.assign[a] != self.start[a]]
        if not changed:
            return self.held
        self.background("final", self.assign)
        for a in changed:
            pts = [self.points["curve"][i] for i in self.cover["curve"][a]]
            new = [self.job("curve", p, self.contrast(p.annulus), self.assign, "final", "path_a") for p in pts]
            old = [self.job("curve", p, self.contrast(p.annulus), {**self.assign, a: self.start[a]}, "final", "path_a")
                   for p in pts]
            n_red = self.compute([x for pair in zip(new, old) for x in pair], f"the held-out check of annulus {a + 1}")
            st = paired_summary(np.array([self.value(j) for j in new]), np.array([self.value(j) for j in old]))
            self.held[a] = {"gain": st["diff_mean"], "se": st["diff_se"], "median_gain": st["diff_median"],
                            "p_better": st["p_better"], "n": st["n_paired"]}
            self.log(f"  final grid: held out, annulus {a + 1}: {_who(self.entrant[(a, self.assign[a])])} over "
                     f"{_who(self.entrant[(a, self.start[a])])} by {_f(st['diff_mean'], '{:+.2f}')} +/- "
                     f"{_f(st['diff_se'])} S/N on {st['n_paired']} points of the curve's grid ({n_red} reductions)")
        return self.held

    def curve(self) -> Optional[Dict[str, Any]]:
        """Read the elected winners on the curve's grid, each point's contrast stepped to the target S/N.

        The steps are the contrast grid's (:func:`klip_tpe.contrast_grid.run_grid`, the settings of
        ``RunConfig.contrast_grid``), and each point starts from its tile's contrast scaled by the tile's
        calibration to the target.

        Returns
        -------
        curve : dict or None
            :func:`klip_tpe.contrast_grid.summarize`'s curve, one entry per separation, with ``annulus``
            (from 1), ``band`` and ``role`` per separation, ``seams`` (:func:`seam_ratios`), ``seam_steps``
            (:func:`seam_steps`), ``read_with`` (the measure), ``reductions`` and ``positions`` (each point's
            record).  None with ``curve_snr`` "off".
        """
        from . import contrast_grid as CG
        how = str(self.fc.curve_snr)
        if how == "off":
            return None
        if self.bgs.get("final", {}).get("assign") != self.assign:
            self.background("final", self.assign)
        gc = self.r.cfg.contrast_grid
        info: Dict[int, Tuple[int, int, int, float]] = {}
        for p in self.points["curve"]:
            info.setdefault(p.sep_index, (p.annulus, p.band, p.n_bands, p.r_px))
        seps = [info[g][3] for g in sorted(info)]
        byidx = {p.index: p for p in self.points["curve"]}
        positions = []
        for p in self.points["curve"]:
            gp = CG.GridPosition(p.index, p.r_px, p.theta, p.sep_index)
            if p.status != "active":
                gp.status, gp.why = "excluded", p.why
            positions.append(gp)
        tgt = float(gc.target_snr)

        def start(gp) -> float:
            cal = self.calibration.get(byidx[gp.index].annulus) or {}
            c = float(cal.get("contrast") or self.r.contrast)
            m = next((h["median_snr"] for h in cal.get("history") or [] if float(h["contrast"]) == c), None)
            return c * tgt / m if (m is not None and np.isfinite(m) and m > 0.5) else c
        n0 = self.n_reductions
        wave = [0]

        def reduce_many(batch) -> Iterator[Tuple[float, Optional[str]]]:
            wave[0] += 1
            jobs = [self.job("curve", byidx[gp.index], c, self.assign, "final", how) for gp, c in batch]
            self.compute(jobs, f"the curve, wave {wave[0]}")
            for j in jobs:
                yield self.readings.get(j.key)

        positions, _ = CG.run_grid(positions, start, reduce_many, lambda res, gp, c: res, gc,
                                   log=lambda s: self.log(s.replace("contrast grid:", "final grid, the curve:")),
                                   stage=lambda s: self.stage(f"the curve, {s}"))
        n_red = self.n_reductions - n0
        out = CG.summarize(positions, seps, self.px, gc, [CG.footprint_deg(r, self.fwhm, gc, None) for r in seps],
                           n_red)
        out["annulus"] = [info[g][0] + 1 for g in sorted(info)]
        out["band"] = [info[g][1] for g in sorted(info)]
        out["role"] = [GridPoint(0, 0, 0, info[g][1], info[g][2], 0.0, 0.0).role for g in sorted(info)]
        out["seams"] = seam_ratios(out, [info[g][:3] for g in sorted(info)], len(self.order), self.order)
        out["seam_steps"] = seam_steps(out, [info[g][:3] for g in sorted(info)], self.order)
        out["read_with"] = how
        out["reductions"] = int(n_red)
        out["positions"] = [dict(gp.to_dict(self.px), annulus=byidx[gp.index].annulus + 1) for gp in positions]
        n_ok = sum(gp.status in CG.STATUS_DONE for gp in positions)
        n_test = sum(gp.status != "excluded" for gp in positions)
        self.log(f"  final grid: the curve, {n_ok} of {n_test} points settled at S/N {tgt:g} in {n_red} reductions, "
                 f"{sum(out['n_blocked'])} blocked")
        for s in out["seams"]:
            self.log(f"  final grid: seam, annulus {s['annulus']}: {_seam_text(s)}")
        st = out["seam_steps"]
        if st["degree"] is not None:
            self.log(f"  final grid: the curve over a polynomial of degree {st['degree']} in log separation and log "
                     f"contrast, which the points scatter about by {100 * st['rms']:.0f} %:")
            an = st["annuli"]
            for k, s in enumerate(an):
                self.log(f"  final grid: seam step, annulus {s['annulus']}: "
                         f"{_step_text(s, an[k + 1]['annulus'] if k + 1 < len(an) else None)}")
        return out


def _seam_text(s: Dict[str, Any]) -> str:
    """Describe an annulus' seam line in words.

    Parameters
    ----------
    s : dict
        The seam line (:func:`seam_ratios`).

    Returns
    -------
    text : str
        Such as ``inner 1.08 (edge of the stitch), outer 0.97 (seam), the curve near each tile edge over
        its middle``.
    """
    def one(v, kind):
        return f"{_f(v)} ({'seam' if kind == 'seam' else 'edge of the stitch'})"
    return (f"inner {one(s['inner'], s['inner_edge'])}, outer {one(s['outer'], s['outer_edge'])}, the curve near each "
            f"tile edge over its middle")


# ----------------------------------------------------------------------------- finalists
def setup_of(runner) -> Dict[str, Any]:
    """Return a runner's setup as :func:`klip_tpe.refine._setup` reads a finished run's, from memory.

    Parameters
    ----------
    runner : Runner
        The run, every annulus done.

    Returns
    -------
    setup : dict
        ``zones`` (each annulus' ``(inrad, outrad)``, rounded), ``names`` (the search space's dimensions)
        and ``frames`` (the data edits that say what the frames are, as JSON).
    """
    from .runner import _core_edits
    res = sorted((r for r in runner.results if r is not None), key=lambda r: r.annulus)
    return {"zones": [(round(float(r.inrad), 6), round(float(r.outrad), 6)) for r in res],
            "names": [p.name for p in runner.space.params],
            "frames": json.dumps(_core_edits(runner.cfg.data_edits), sort_keys=True, default=str)}


def _take(ents: Sequence[Entrant], n: int) -> List[Entrant]:
    """Return a run's finalists for an annulus, its winner first and then its best by validated score.

    Parameters
    ----------
    ents : sequence of Entrant
        The run's candidates, one of kind "winner" at most.
    n : int
        Number kept, 0 for all, the others then in their order.

    Returns
    -------
    finalists : list of Entrant
        The finalists.
    """
    win = [e for e in ents if e.kind == "winner"][:1]
    rest = [e for e in ents if e not in win]
    if n > 0:
        rest = sorted(rest, key=lambda e: -(float(e.validated_score) if e.validated_score is not None
                                            and np.isfinite(float(e.validated_score)) else -np.inf))
        rest = rest[:max(n - len(win), 0)]
    return win + rest


def collect_finalists(runner, entrant_runs: Sequence[str] = (), n_best: int = 0,
                      log: Callable[[str], None] = print) -> Tuple[Dict[int, List[Entrant]], Dict[int, Entrant]]:
    """Return each annulus' finalists and its starting winner, the validation's.

    The run's own finalists are the configurations its validation scored, from its results in memory,
    so the stage runs inside a run as on a finished one.  Each other run's are its winner and its
    validation's entrants (:func:`klip_tpe.refine.run_entrants`), checked against the run's setup.  A
    configuration offered twice enters once, as the first run to offer it.

    Parameters
    ----------
    runner : Runner
        The run, every annulus done.
    entrant_runs : sequence of str, default ()
        Directories of other finished runs of the same setup.
    n_best : int, default 0
        Number of each run's finalists per annulus, its winner and its best by validated score.  0 takes
        all.
    log : callable, default print
        Function that takes each line of the log.

    Returns
    -------
    finalists : dict of int to list of Entrant
        Each annulus' finalists by 0-based annulus, its starting winner first.
    start : dict of int to Entrant
        Each annulus' starting winner.

    Raises
    ------
    ValueError
        If another run's annuli, search space or frames are not the run's.
    """
    base = os.path.abspath(runner.run_dir)
    name = os.path.basename(base.rstrip(os.sep))
    results = sorted((r for r in runner.results if r is not None), key=lambda r: r.annulus)
    others: List[str] = []
    ref = None
    for rd in entrant_runs or []:
        rd = os.path.abspath(rd)
        if rd == base or rd in others:
            continue
        if ref is None:
            ref = setup_of(runner)
        _check_setup(rd, ref)
        others.append(rd)
    fin: Dict[int, List[Entrant]] = {}
    start: Dict[int, Entrant] = {}
    for r in results:
        a = int(r.annulus)
        wkey = config_key(r.winner_config)
        own = [Entrant(name, "winner" if config_key(row["config"]) == wkey else "validation", row.get("eval_index"),
                       row["config"], row.get("search_score"), row.get("validated_score"))
               for row in (r.validation_table or []) if row.get("config")]
        win = next((e for e in own if e.kind == "winner"), None)
        if win is None:
            win = Entrant(name, "winner", int(r.winner_index), dict(r.winner_config), None, float(r.winner_score))
            own.insert(0, win)
        lst: List[Entrant] = []
        seen = set()
        for e in _take(own, int(n_best)) + [x for rd in others for x in _take(run_entrants(rd, a), int(n_best))]:
            if e.key not in seen:
                seen.add(e.key)
                lst.append(e)
        fin[a], start[a] = lst, win
    return fin, start


# ----------------------------------------------------------------------------- the stage
def run_final_grid(runner, out_dir: str, *, config: Optional[FinalGridConfig] = None,
                   entrant_runs: Optional[Sequence[str]] = None, weights: Optional[Sequence[float]] = None,
                   window: Optional[int] = None, plan: bool = False, card: bool = False, in_run: bool = False,
                   overlap_pads: Optional[bool] = None, log: Callable[[str], None] = print,
                   stage: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    """Run the final-image grid on a runner whose annuli are all done, and write its reports.

    It calibrates each tile's contrast, runs the election, reads each change on the curve's grid and
    reads the curve, then writes ``election.json``, ``election.txt``, ``winners.txt``, ``curve.json`` and
    ``curve.txt`` to ``out_dir``, with its readings in ``out_dir/state``, from which a stage started
    again on the same directory resumes.  ``winners.txt`` lists each annulus' winner as
    ``rereduce --base-winners`` reads it.  The runs are only read.

    Parameters
    ----------
    runner : Runner
        The run, every annulus done.
    out_dir : str
        Directory of the stage, ``final_grid`` in the run inside a run, beside the runs for a finished
        one.
    config : FinalGridConfig or None, default None
        The settings.  None takes the run's (``RunConfig.final_grid``).
    entrant_runs : sequence of str or None, default None
        Directories of other finished runs of the same setup whose finalists enter too.  None takes the
        settings' ``entrants_from``.
    weights : sequence of float or None, default None
        The stitch weights of every reading.  None works them out from each background's clean images,
        as the final stitch does.
    window : int or None, default None
        Number of reductions in flight at most.  None takes the runner's.
    plan : bool, default False
        Whether to count the points and the reductions and return, making none.
    card : bool, default False
        Whether the injected reductions may go to the card workers.
    in_run : bool, default False
        Whether the stage runs inside the run, its directory then inside the run's.
    overlap_pads : bool or None, default None
        Whether overlapping annuli take pads between them in the stitch the stage reads, as
        ``rereduce --overlap-pads`` sets it.  None keeps the run's (``RunConfig.stitch_overlap_pads``).
    log : callable, default print
        Function that takes each line of the log.
    stage : callable, default a no-op
        Function that takes a short status line, for the run's heartbeat.

    Returns
    -------
    record : dict
        What ``election.json`` holds: ``annuli`` (per annulus counted from 1 its ``start`` and ``winner``
        entrants, ``changed``, ``last_change`` and ``held_out``), ``stitch`` (the tiles and pads read),
        ``calibration``, ``sweeps``, ``curve``, ``plan``, the settings and the counts.  With ``plan``,
        ``plan``, ``stitch`` and ``finalists`` alone.

    Raises
    ------
    ValueError
        If a setting is out of range, ``out_dir`` lies inside a run of a finished stage, the run has not
        finished, searched its widths or scanned k, or another run's setup is not the run's.
    """
    fc = config if config is not None else runner.cfg.final_grid
    fc.check()
    t0 = time.time()
    base = os.path.abspath(runner.run_dir)
    out_dir = os.path.abspath(out_dir)
    runs_from = [os.path.abspath(p) for p in (fc.entrants_from if entrant_runs is None else entrant_runs) or []]
    if not in_run:
        for rd in [base] + runs_from:
            if out_dir == rd or out_dir.startswith(rd + os.sep):
                raise ValueError(f"{out_dir} is inside {rd}: the stage writes beside the runs")
    scorer = StitchScorer(runner, overlap_pads=overlap_pads, log=log, card=card)
    st = _stitch_record(scorer, runner)
    finalists, start = collect_finalists(runner, runs_from, int(fc.finalists), log)
    fg = FinalGrid(runner, scorer, fc, finalists, start, None if plan else os.path.join(out_dir, "state"), weights,
                   log, window, card, stage)
    pl = fg.plan()
    nf = [len(v) for v in finalists.values()]
    seps = sorted({(p.annulus, p.n_bands) for p in fg.points["elect"]})
    nb = sorted({n for _, n in seps})
    log(f"final grid of {os.path.basename(base)}: {len(fg.order)} annuli, "
        f"{min(nf) if min(nf) == max(nf) else f'{min(nf)} to {max(nf)}'} finalists each from "
        f"{1 + len(runs_from)} run(s), {pl['points']} points ({nb[0] if len(nb) == 1 else f'{nb[0]} to {nb[-1]}'} "
        f"separations per tile x {int(fc.n_pa)} angles"
        + (f", {pl['excluded']['elect']} left out where the run's sampler places no source"
           if pl["excluded"]["elect"] else "") + ")")
    log(f"  the stitch read: {_pads_text(st)}")
    for a in fg.order:
        p = pl["annuli"][a]
        log(f"  annulus {a + 1}: {p['finalists']} finalists x {p['cover']} points within "
            f"{float(fc.cover_fwhm):g} FWHM of its part of the stitch: {p['reductions']} reductions a sweep at most")
    log(f"  at most {pl['election_sweep']} reductions in the election's first sweep, {pl['calibration_step']} a step "
        f"of the calibration and {pl['curve_wave']} a wave of the curve; every annulus whose part of the stitch "
        f"lies within {fg.read_px / fg.fwhm:g} FWHM of a point reduced with its source")
    if plan:
        return {"plan": pl, "stitch": st,
                "finalists": {str(a + 1): [asdict(e) for e in v] for a, v in finalists.items()}}
    runner._start_workers()
    fg.open_state()
    try:
        fg.background("start", fg.start)
        fg.calibrate()
        fg.elect()
        fg.held_out()
        cv = fg.curve()
    finally:
        fg.close()
    rec = _record(fg, base, runs_from, pl, cv, time.time() - t0)
    rec["stitch"] = st
    os.makedirs(out_dir, exist_ok=True)
    _write_reports(out_dir, rec)
    changed = [int(k) for k, v in rec["annuli"].items() if v["changed"]]
    what = ("no winner changed" if not changed else ("annulus " if len(changed) == 1 else "annuli ")
            + ", ".join(str(k) for k in changed) + " took another winner")
    log(f"  final grid: {what}, {fg.n_reductions} reductions in {(time.time() - t0) / 60:.1f} min; wrote "
        f"{os.path.join(out_dir, 'election.txt')}, election.json, winners.txt, curve.txt and curve.json")
    return rec


def _stitch_record(scorer: StitchScorer, runner) -> Dict[str, Any]:
    """Return the tiles and pads of the stitch the stage reads, for its reports.

    Parameters
    ----------
    scorer : StitchScorer
        The stitch.
    runner : Runner
        The run.

    Returns
    -------
    stitch : dict
        ``overlap_pads`` (whether overlapping annuli take pads between them here), ``run_overlap_pads``
        (the run's own), ``blend`` (``RunConfig.stitch_blend`` here, ``"own"`` or ``"weighted"``),
        ``overlapping`` (whether any two annuli overlap), ``edge_pads_px`` (the pad at each edge between two
        tiles, in pixels), ``tiles_px`` and ``pads_px``.
    """
    z = scorer.zones
    edges = [max(float(scorer.pads[k][1]), float(scorer.pads[k + 1][0])) for k in range(len(z) - 1)]
    return {"overlap_pads": bool(scorer.overlap_pads),
            "run_overlap_pads": bool(getattr(runner.cfg, "stitch_overlap_pads", True)),
            "blend": str(getattr(scorer, "blend", "weighted")),
            "overlapping": any(float(z[k][1]) > float(z[k + 1][0]) + 1e-9 for k in range(len(z) - 1)),
            "edge_pads_px": edges, "tiles_px": [[float(v) for v in t] for t in scorer.tiles],
            "pads_px": [[float(v) for v in p] for p in scorer.pads]}


def _pads_text(st: Dict[str, Any]) -> str:
    """Describe the pads of the stitch the stage reads in words.

    Parameters
    ----------
    st : dict
        The stitch's record (:func:`_stitch_record`).

    Returns
    -------
    text : str
        Such as ``no pads between overlapping annuli, the tiles meeting sharply there (the run's own
        stitch had 2 px pads)``.  A stitch under ``RunConfig.stitch_blend = "own"`` pads no edge
        between tiles whatever ``overlap_pads`` says, and the text says so.
    """
    pads = sorted({round(float(v), 3) for v in st.get("edge_pads_px") or []})
    if not pads:
        return "one tile, no edges between tiles"
    width = "/".join(f"{v:g}" for v in pads)
    if not st.get("overlapping"):
        return (f"{width} px pads at the edges between tiles, the annuli meeting there" if max(pads) > 0 else
                "the tiles meeting sharply, without pads, each pixel from its tile's annulus")
    if max(pads) == 0 and str(st.get("blend", "weighted")).lower() == "own":
        return "no pads between overlapping annuli, each pixel from its tile's annulus (stitch blend own)"
    if st.get("overlap_pads"):
        text = f"{width} px pads at the edges between overlapping annuli"
    else:
        text = "no pads between overlapping annuli, the tiles meeting sharply there"
    if bool(st.get("overlap_pads")) != bool(st.get("run_overlap_pads")):
        text += " (the run's own stitch " + ("had pads there" if st.get("run_overlap_pads") else "had none") + ")"
    return text


def _record(fg: FinalGrid, base: str, runs_from: Sequence[str], pl: Dict[str, Any], cv: Optional[Dict[str, Any]],
            wall: float) -> Dict[str, Any]:
    """Return the stage's record, what ``election.json`` holds.

    Parameters
    ----------
    fg : FinalGrid
        The stage, run.
    base : str
        Directory of the run.
    runs_from : sequence of str
        Directories of the other runs whose finalists entered.
    pl : dict
        The plan (:meth:`FinalGrid.plan`).
    cv : dict or None
        The curve (:meth:`FinalGrid.curve`).
    wall : float
        Time the stage took, in seconds.

    Returns
    -------
    record : dict
        The record, the annuli counted from 1.
    """
    opts = asdict(fg.fc)
    opts["target"] = [float(v) for v in fg.fc.target]
    ann = {}
    for a in fg.order:
        w, s = fg.entrant[(a, fg.assign[a])], fg.entrant[(a, fg.start[a])]
        ann[str(a + 1)] = {"start": asdict(s), "winner": asdict(w), "changed": fg.assign[a] != fg.start[a],
                           "finalists": len(fg.finalists[a]), "last_change": fg.last_change.get(a),
                           "held_out": fg.held.get(a), "tile_px": list(fg.sc.tile(a)),
                           "contrast": fg.calibration.get(a, {}).get("contrast")}
    return {"run": base, "runs": [base] + list(runs_from), "options": opts, "read_fwhm": fg.read_px / fg.fwhm,
            "floor_px": fg.floor, "plan": {k: v for k, v in pl.items() if k != "annuli"},
            "calibration": {str(a + 1): v for a, v in fg.calibration.items()}, "sweeps": fg.sweeps, "annuli": ann,
            "grid": {g: [p.to_dict(fg.px) for p in pts] for g, pts in fg.points.items()},
            "curve": cv, "reductions": int(fg.n_reductions), "wall_s": float(wall),
            "finished": _dt.datetime.now().isoformat(timespec="seconds")}


def _atomic_text(path: str, text: str) -> None:
    """Write a text file through a temporary file replaced into place.

    Parameters
    ----------
    path : str
        Path of the file.
    text : str
        Its content.
    """
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def _write_winners(out_dir: str, rec: Dict[str, Any], now: str) -> None:
    """Write ``winners.txt``, each annulus' winner as ``rereduce --base-winners`` reads it.

    One line per annulus gives the annulus (from 1), the winner's run as a directory where the stage
    knows it, else its name, and its evaluation (from 0).  Every annulus is listed, so a composite made
    from the file does not depend on the winners the base run itself took.

    Parameters
    ----------
    out_dir : str
        Directory of the stage.
    rec : dict
        The record (:func:`_record`).
    now : str
        The time written in the header.
    """
    import shlex
    dirs = {os.path.basename(os.path.abspath(r).rstrip(os.sep)): os.path.abspath(r) for r in rec.get("runs") or []}
    L = [f"# klip_tpe final-image grid winners -- {now}", f"# run: {rec['run']}",
         "# one line per annulus, for rereduce --base-winners and rescore --base-winners, giving the annulus (from 1), "
         "the run and the evaluation (from 0, as election.json counts)",
         f"# the stitch read: {_pads_text(rec.get('stitch') or {})}"]
    for k, v in rec["annuli"].items():
        w = v["winner"]
        what = f"{w['run']} {w['kind']}" + (", changed" if v["changed"] else ", the validation's winner")
        if w.get("eval_index") is None:
            L.append(f"# annulus {k}: {what}, no evaluation to name it by")
            continue
        run = dirs.get(str(w["run"]), str(w["run"]))
        L.append(f"{int(k):<3d} {shlex.quote(run)}  {int(w['eval_index'])}    # {what}")
    _atomic_text(os.path.join(out_dir, "winners.txt"), "\n".join(L) + "\n")


def _write_reports(out_dir: str, rec: Dict[str, Any]) -> None:
    """Write the stage's ``election.json``, ``election.txt``, ``winners.txt``, ``curve.json`` and ``curve.txt``.

    Parameters
    ----------
    out_dir : str
        Directory of the stage.
    rec : dict
        The record (:func:`_record`).
    """
    cv = rec.get("curve")
    _atomic_text(os.path.join(out_dir, "election.json"),
                 json.dumps({k: v for k, v in rec.items() if k != "curve"}, indent=1, default=float))
    o = rec["options"]
    now = _dt.datetime.now().isoformat(timespec="seconds")
    L = [f"# klip_tpe final-image grid -- {now}", f"# run: {rec['run']}",
         f"# finalists from: {', '.join(os.path.basename(r) for r in rec['runs'])}",
         f"# grid: an odd number of separations per tile, at least 3, at the middles of equal bands no wider than "
         f"{o['sep_step_fwhm']:g} FWHM, {o['n_pa']} position angles each, turned by the golden fraction of the step "
         f"from one separation to the next; the curve's grid turned by half a step",
         f"# the stitch read: {_pads_text(rec.get('stitch') or {})}",
         f"# each point one injected source; every annulus whose part of the stitch lies within {rec['read_fwhm']:g} "
         f"FWHM of it reduced with it, the others clean; each source measured on the stitch as the stitched verify "
         f"measures it (Path A)",
         f"# election: an annulus' finalists in turn in its slot, the others at their current winners, read on the "
         f"points within {o['cover_fwhm']:g} FWHM of its part of the stitch at each tile's contrast; a finalist "
         f"replaces the winner when its mean paired gain exceeds {o['z']:g} x its standard error; up to "
         f"{o['sweeps']} sweeps, inner to outer, then back",
         "# eval: counted from 1, as results.txt counts (election.json counts from 0)", "#",
         "# tile contrasts: " + ", ".join(f"{k} {_f(v.get('contrast'), '{:.3e}')}"
                                          + ("" if v.get("converged") else " (?)")
                                          for k, v in rec["calibration"].items())]
    for s, rows in enumerate(rec["sweeps"]):
        L += ["", f"== sweep {s + 1} =="]
        for row in rows:
            b, a = row["before"], row["after"]
            what = (f"{a['run']} {a['kind']} eval {_ev1(a['eval_index'])} replaces {b['run']} eval "
                    f"{_ev1(b['eval_index'])}" if row["changed"] else f"{b['run']} {b['kind']} eval "
                    f"{_ev1(b['eval_index'])} stays")
            L.append(f"annulus {row['annulus']}  {row['n_points']} points  {what}  ({row['reductions']} reductions)")
            L.append(f"   {'run':<24} {'kind':<10} {'eval':>6}  {'mean S/N':>8}  {'gain vs winner':>16}  {'P':>4}  "
                     f"{'n':>4}")
            for x in sorted(row["finalists"], key=lambda x: -(x["mean"] if np.isfinite(x["mean"]) else -np.inf)):
                star = "*" if x["current"] else " "
                gain = "0" if x["current"] else f"{_f(x['gain'], '{:+.2f}')} +/- {_f(x['se'])}"
                L.append(f"  {star}{x['run']:<24} {x['kind']:<10} {_ev1(x['eval_index']):>6}  {_f(x['mean']):>8}  "
                         f"{gain:>16}  {'-' if x['current'] else _f(x['p_better']):>4}  {x['n']:>4}")
    L += ["", "== winners =="]
    for k, v in rec["annuli"].items():
        w, s = v["winner"], v["start"]
        if not v["changed"]:
            L.append(f"annulus {k}: {w['run']} {w['kind']} eval {_ev1(w['eval_index'])}, the validation's winner")
            continue
        lc, ho = v.get("last_change") or {}, v.get("held_out") or {}
        L.append(f"annulus {k}: {w['run']} {w['kind']} eval {_ev1(w['eval_index'])}, in place of {s['run']} eval "
                 f"{_ev1(s['eval_index'])}: {_f(lc.get('gain'), '{:+.2f}')} +/- {_f(lc.get('se'))} S/N on "
                 f"{lc.get('n')} points in sweep {lc.get('sweep')}; held out "
                 f"{_f(ho.get('gain'), '{:+.2f}')} +/- {_f(ho.get('se'))} on {ho.get('n')} points of the curve's grid")
    L += ["", f"# {rec['reductions']} reductions in {rec['wall_s'] / 3600:.2f} h"]
    _atomic_text(os.path.join(out_dir, "election.txt"), "\n".join(L) + "\n")
    _write_winners(out_dir, rec, now)
    if not cv:
        return
    _atomic_text(os.path.join(out_dir, "curve.json"), json.dumps(cv, indent=1, default=float))
    snr = ("the contrast grid's S/N (" + str(cv.get("snr")) + ")" if cv.get("read_with") == "grid"
           else "the stitched verify's S/N (Path A)")
    C = [f"# klip_tpe final-image contrast curve -- {now}", f"# run: {rec['run']}",
         f"# the elected winners' stitch, one injected source per reduction at each point of the curve's grid (the "
         f"election's turned by half a step), each point's contrast stepped to S/N {cv['target_snr']:g} +/- "
         f"{100 * cv['precision']:g} %, read with {snr}; the median over position angle at each separation, a "
         f"blocked point counted above the settled ones (median_settled leaves it out)",
         "# sep_arcsec   contrast_median   p16   p84   n_settled   n_tested   annulus   band   n_blocked   "
         "median_settled"]
    for i, r in enumerate(cv["r_as"]):
        def g(key):
            vals = cv.get(key) or []
            v = vals[i] if i < len(vals) else None
            return float("nan") if v is None else float(v)
        nb = (cv.get("n_blocked") or [0] * len(cv["r_as"]))[i]
        C.append(f"{r:10.4f}   {g('curve'):12.4e}   {g('p16'):12.4e}   {g('p84'):12.4e}   {cv['n'][i]:4d}   "
                 f"{cv['n_pos'][i]:4d}   {cv['annulus'][i]:3d}   {cv['role'][i]}   {int(nb):4d}   "
                 f"{g('curve_settled'):12.4e}")
    C.append("# seam per annulus, the curve near each tile edge over its middle:")
    for s in cv.get("seams") or []:
        C.append(f"# annulus {s['annulus']}: {_seam_text(s)}")
    st = cv.get("seam_steps") or {}
    if st.get("degree") is not None:
        C.append(f"# seam steps, the curve over a polynomial of degree {st['degree']} in log separation and log "
                 f"contrast fitted to all of it (points scatter by {100 * st['rms']:.0f} %):")
        an = st["annuli"]
        for k, s in enumerate(an):
            C.append(f"# annulus {s['annulus']}: {_step_text(s, an[k + 1]['annulus'] if k + 1 < len(an) else None)}")
    _atomic_text(os.path.join(out_dir, "curve.txt"), "\n".join(C) + "\n")


def run_weights(run_dir: str, nann: int) -> Optional[List[float]]:
    """Return a finished run's stitch weights from its ``final_results.json``.

    Parameters
    ----------
    run_dir : str
        Directory of the run.
    nann : int
        Number of the run's annuli.

    Returns
    -------
    weights : list of float or None
        The weights, None where the file holds none for every annulus.
    """
    try:
        with open(os.path.join(run_dir, "final_results.json")) as f:
            wl = json.load(f).get("stitch_weights")
    except (OSError, ValueError):
        return None
    return [float(v) for v in wl] if wl is not None and len(wl) == int(nann) else None


def elect_products(runner, rec: Dict[str, Any], out_dir: str, log: Callable[[str], None] = print,
                   overlap_pads: Optional[bool] = None, stitch_injections: Optional[str] = None) -> Dict[str, Any]:
    """Write a finished run's final products with the final-image grid's winners to the stage's directory.

    Each annulus whose winner changed reduces that winner in place of the validation's, and the others
    their validation's winner (:meth:`klip_tpe.runner.Runner.rereduce` with a composite), so the stitch,
    its S/N maps and the stitched verify are the elected winners'.  ``contrast_curve.txt`` takes the
    stage's curve.  The run is only read.

    Parameters
    ----------
    runner : Runner
        The run, resumed read-only.
    rec : dict
        The stage's record (:func:`run_final_grid`).
    out_dir : str
        Directory of the stage, beside the run.
    log : callable, default print
        Function that takes each line of the log.
    overlap_pads : bool or None, default None
        Whether overlapping annuli take pads between them, as the stage read them.  None keeps the run's.
    stitch_injections : {"all", "own"} or None, default None
        Which validation sources each annulus' injected reduction holds.  None keeps the run's.

    Returns
    -------
    record : dict
        :meth:`klip_tpe.runner.Runner.rereduce`'s record.
    """
    from .stitch import write_contrast_curve
    winners = {int(k) - 1: Entrant(**v["winner"]) for k, v in rec["annuli"].items() if v["changed"]}
    results = [r for r in runner.results if r is not None]
    keep = [(r, r.final_grid) for r in results]
    runner._ann_images.clear()
    try:
        # the stage starts from the validation's winners, and the products take them where it changed none
        for r in results:
            r.final_grid = None
        out = runner.rereduce(out_dir, winners=winners, log=log, overlap_pads=overlap_pads,
                              stitch_injections=stitch_injections)
    finally:
        for r, fg in keep:
            r.final_grid = fg
        runner._ann_images.clear()
    if rec.get("curve"):
        tl = runner._tiles([(float(r.inrad), float(r.outrad)) for r in results])
        edges = [t[0] for t in tl] + [tl[-1][1]]
        write_contrast_curve(os.path.join(out_dir, "contrast_curve.txt"), [r.contrast_curve for r in results], edges,
                             runner.fwhm, runner.pxscale, legacy=runner.cfg.legacy_stitch,
                             contrasts=[r.contrast for r in results], fm_curves=[r.fm_curve for r in results],
                             final_curve=rec["curve"])
        log(f"  wrote {os.path.join(out_dir, 'contrast_curve.txt')}, the elected winners' curve")
    return out
