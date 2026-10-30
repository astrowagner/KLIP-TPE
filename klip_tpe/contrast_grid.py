"""Injection-recovery contrast curve, one injected source per reduction.

The protocol of the IDL contrast generators (``contrastcurve_*.pro``, e.g.
``contrastcurve_LMIRCam.pro``), run on each annulus's winning configuration:

* test positions on a grid of separations (the annulus less ``edge_fwhm`` FWHM at each edge,
  ``sep_step_fwhm`` apart) and ``n_pa`` position angles per separation, offset from one
  separation to the next.  ``mode="random"`` draws ``n_samples`` positions instead;
* one source per reduction, so no injection biases another's signal or noise;
* each position iterated on its own contrast until its S/N is within ``precision`` of
  ``target_snr``: ``c <- c * target / snr``, ``c <- 10 c`` when the source is not seen
  (S/N <= 0).  From the ``accept_after``-th reduction on (6th) a position at or above the
  target is accepted at its last contrast, and from the ``drop_after``-th (8th) one still
  short of ``(1 - precision) * target`` is dropped as blocked.  These are the IDL's two
  rules, ``runs gt 5`` and ``runs gt 7``;
* the S/N is the run's matched-filter Mawet S/N (``snr="matched"``), or the IDL's aperture
  photometry (``snr="aperture"``): FWHM/2 apertures, the ring a FWHM apart, the small-sample
  correction of Mawet et al. (2014);
* the noise ring leaves out the injected source and its ADI sidelobes, every known companion
  and the companions' sidelobes on rings near their separation.  No position is placed in a
  companion's sidelobes either.  A sidelobe is the arc ADI over-subtracts a source along:
  ``+/- min(PA span of the data, the configuration's anglemax)`` (``sidelobe="rotation"``),
  never less than ``sidelobe_fwhm`` FWHM of arc, the IDL's two apertures each side
  (``sidelobe="fixed"``: that floor alone);
* the curve is the median over the converged positions at each separation, with the 16th and
  84th percentiles.

The driver takes two callables, ``reduce_many`` and ``measure``, so the arithmetic here is
the same whatever reduces the images (:meth:`klip_tpe.runner.Runner._contrast_grid`).
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
    """Settings of the injection-recovery curve (see the module docstring).

    ``mode``: ``"grid"`` (default), ``"random"`` or ``"off"``.  A run saved before the grid
    existed resumes with ``"off"``."""

    mode: str = field(default_factory=lambda: _DEFAULT["mode"])
    n_pa: int = 12
    n_samples: int = 96
    sep_step_fwhm: float = 1.0
    edge_fwhm: float = 1.0
    target_snr: float = 5.0
    precision: float = 0.1
    snr: str = "matched"
    sidelobe: str = "rotation"
    sidelobe_fwhm: float = 2.0
    known_radial_fwhm: float = 1.5
    accept_after: int = 6
    drop_after: int = 8
    c_max: float = 0.1

    @property
    def enabled(self) -> bool:
        return str(self.mode).lower() in ("grid", "random")


@dataclass
class GridPosition:
    """One test position: separation (px), angle (deg, the run's source convention), the
    separation's index, and its history of ``(contrast, snr)`` trials."""

    index: int
    r_px: float
    theta: float
    sep_index: int
    trials: List[Tuple[float, float]] = field(default_factory=list)
    status: str = "active"
    contrast: Optional[float] = None
    why: Optional[str] = None
    next_c: Optional[float] = None

    def to_dict(self, pxscale: float) -> Dict[str, Any]:
        return {"index": self.index, "rho_as": float(self.r_px * pxscale), "theta": float(self.theta),
                "sep_index": int(self.sep_index), "status": self.status,
                "contrast": None if self.contrast is None else float(self.contrast), "why": self.why,
                "trials": [[float(c), None if not np.isfinite(s) else float(s)] for c, s in self.trials]}


# ----------------------------------------------------------------------------- geometry
def separations_px(inrad: float, outrad: float, fwhm: float, cfg: ContrastGridConfig,
                   floor_px: float = 0.0) -> np.ndarray:
    """The test separations: ``edge_fwhm`` FWHM in from each zone edge (and outside
    ``floor_px``), ``sep_step_fwhm`` FWHM apart, centered in the room left over.  One
    separation at the zone's middle when the room is shorter than a step."""
    lo = max(float(inrad) + cfg.edge_fwhm * fwhm, float(floor_px))
    hi = float(outrad) - cfg.edge_fwhm * fwhm
    step = max(cfg.sep_step_fwhm * fwhm, 1e-6)
    if hi < lo:
        return np.array([0.5 * (max(float(inrad), float(floor_px)) + float(outrad))])
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    span = (n - 1) * step
    return lo + 0.5 * ((hi - lo) - span) + step * np.arange(n)


def grid_positions(seps: Sequence[float], n_pa: int) -> List[GridPosition]:
    """``n_pa`` angles per separation, evenly spaced.  Each separation turns the pattern by
    the golden fraction of the angle step, so no azimuth is tested at every separation."""
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
    """``n`` positions uniform in area over the separations' span (one bin each side of a
    test separation, ``step_px`` wide) and uniform in angle, binned to the nearest separation."""
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
    """Half-width (deg) of a source's sidelobe arc at separation ``r_px``: the field
    rotation the references span (``rotation_deg``, ``sidelobe="rotation"``), never less than
    ``sidelobe_fwhm`` FWHM of arc."""
    floor = float(np.rad2deg(cfg.sidelobe_fwhm * fwhm / max(float(r_px), 1e-6)))
    if str(cfg.sidelobe).lower() == "fixed" or rotation_deg is None or not np.isfinite(rotation_deg):
        return min(floor, 180.0)
    return min(max(float(rotation_deg), floor), 180.0)


def in_footprint(r_px: float, theta: float, rk_px: float, thk: float, half_deg: float, radial_px: float) -> bool:
    """Is ``(r_px, theta)`` inside the arc of half-width ``half_deg`` around ``(rk_px, thk)``,
    within ``radial_px`` of its separation?"""
    d = abs((float(theta) - float(thk) + 180.0) % 360.0 - 180.0)
    return abs(float(r_px) - float(rk_px)) <= radial_px and d <= half_deg


# ---------------------------------------------------------------------------- iteration
def next_contrast(c: float, snr: float, target: float, c_max: float) -> float:
    """The IDL's update: ``c * target / snr``, or ``10 c`` for a source not seen."""
    nxt = c * target / snr if (np.isfinite(snr) and snr > 0) else 10.0 * c
    return float(min(nxt, c_max))


def run_grid(positions: List[GridPosition], start: Callable[[GridPosition], float],
             reduce_many: Callable[[List[Tuple[GridPosition, float]]], Iterable[Any]],
             measure: Callable[[Any, GridPosition, float], Tuple[float, Optional[str]]],
             cfg: ContrastGridConfig, log: Callable[[str], None] = lambda s: None,
             stage: Callable[[str], None] = lambda s: None) -> Tuple[List[GridPosition], int]:
    """Iterate every active position to the target S/N, a wave at a time.

    ``start(pos)`` gives a position's first contrast.  ``reduce_many([(pos, c), ...])`` yields
    one result per entry, in order (an image, or an exception for a failed reduction).
    ``measure(result, pos, c)`` returns ``(snr, why)``, ``why`` a reason for a NaN.  Returns
    the positions and the number of reductions made."""
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
    return float(np.percentile(v, q)) if v.size else float("nan")


def summarize(positions: Sequence[GridPosition], seps: Sequence[float], pxscale: float,
              cfg: ContrastGridConfig, sidelobe_deg: Sequence[float], n_reductions: int) -> Dict[str, Any]:
    """The curve: per separation the median contrast of the settled positions with the 16th
    and 84th percentiles, and the counts (``n`` settled, ``n_pos`` tested, ``n_blocked``)."""
    seps = np.asarray(seps, float)
    med, p16, p84, n, npos, nblk = [], [], [], [], [], []
    for j in range(seps.size):
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
            "n_blocked": nblk, "sidelobe_deg": js(sidelobe_deg), "mode": str(cfg.mode), "n_pa": int(cfg.n_pa),
            "n_samples": int(cfg.n_samples), "target_snr": float(cfg.target_snr), "precision": float(cfg.precision),
            "snr": str(cfg.snr), "sidelobe": str(cfg.sidelobe), "n_reductions": int(n_reductions),
            "n_excluded": int(sum(p.status == "excluded" for p in positions))}
