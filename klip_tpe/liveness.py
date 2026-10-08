"""Is every searched dimension actually doing something?

A dimension can enter the search space and change nothing: the reducer ignores it, a
guard pins it, or it only matters in a region the space never visits.  The search cannot
tell.  TPE models noise in a dead dimension as happily as signal, and the importance panel
even hands it a correlation with S/N as the run converges.  GO 1386 MIRI run v6 searched two
such dimensions (``nkeep_altroll`` / ``nkeep_psfref`` on the pyKLIP backend, which ignored
them) for four hours, and the angular dimensions ``angsep`` / ``anglemax`` are provably inert
on two-roll JWST data for a different reason.  Both were found by accident, after the fact.

:func:`check_live_dimensions` is the check that finds them before the fact: for each searched
reduction dimension it moves that dimension alone -- after the space's own guards, so a
move the guards undo or that drags another dimension with it does not count -- reduces the
clean image, and asks whether anything changed.  The moves go to the bounds and the midpoint
first, and across a dense grid for whatever those leave unmoved: a frame-cut threshold can
act only in a narrow band of its range.  A handful of reductions against hours of search.

``klip-tpe near`` / ``generic`` run it before a new run's first evaluation
(:attr:`klip_tpe.runner.RunConfig.liveness_check`).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

__all__ = ["check_live_dimensions", "dead_dimensions"]


def _same(a, b) -> bool:
    """Bit-identical images, NaN pattern included."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return False
    fa, fb = np.isfinite(a), np.isfinite(b)
    return bool(np.array_equal(fa, fb) and np.array_equal(a[fa], b[fb]))


def _candidates(space, x, i) -> List[float]:
    """Values for dimension ``i`` away from ``x[i]``: the far end first, then the near end,
    then the middle; for a categorical every other choice."""
    p = space.params[i]
    v = float(x[i])
    if p.kind == "categorical":
        n = len(p.choices)
        return [float((int(round(v)) + j) % n) for j in range(1, n)]
    lo, hi = float(p.lo), float(p.hi)
    far = lo if (v - lo) > (hi - v) else hi
    near = hi if far == lo else lo
    return [far, near, 0.5 * (lo + hi)]


def _grid(space, x, i, n: int) -> List[float]:
    """A dense set of values for dimension ``i``, nearest to ``x[i]`` first: ``n`` evenly
    spaced across its range (every integer when an int range has no more than ``n``).  A
    threshold often acts only in a narrow band -- a frame cut above every frame's tag ratio
    cuts nothing, one far below it empties the night and is snapped back by the guard -- and
    the band is usually near the default, where the bounds and the midpoint never land."""
    p = space.params[i]
    v = float(x[i])
    if p.kind == "categorical":
        return []                               # the standard pass already tried every choice
    lo, hi = float(p.lo), float(p.hi)
    if p.kind == "int" and (hi - lo + 1) <= n:
        vals = np.arange(int(round(lo)), int(round(hi)) + 1, dtype=float)
    else:
        vals = np.linspace(lo, hi, int(n))
        if p.kind == "int":
            vals = np.unique(np.round(vals))
    vals = [float(u) for u in vals if abs(u - v) > 1e-9]
    return sorted(vals, key=lambda u: (abs(u - v), u))


def check_live_dimensions(runner, n_points: int = 3, annulus: int = 0, seed: int = 0,
                          log: Optional[Callable[[str], None]] = None,
                          dense: int = 41) -> Dict[str, Dict[str, Any]]:
    """For every searched reduction dimension of ``runner.space``: does moving it alone change
    the clean reduction of annulus ``annulus``?

    Tried at the default configuration first and then at up to ``n_points - 1`` random
    feasible ones, stopping for each dimension as soon as one move changes the image (so a
    healthy space costs ``1 + ndim`` reductions).  The first pass moves each dimension to its
    bounds and its midpoint; whatever that leaves unmoved is re-probed on a ``dense``-value
    grid (nearest the point's own value first) at the same points before it is called dead.
    Without the second pass the frame-selection thresholds of the 56-dimensional NEAR space
    came out dead on three nights of six, because each acts only in a band a few tenths
    wide between "culls nothing" and "culls too much" (found by the NEAR2 / IDL session,
    2026-10-07).  ``dense=0`` keeps the first pass alone.

    Returns ``{name: {"live", "tested", "pinned", "coupled", "live_at"}}``: ``tested`` moves
    reduced, ``pinned`` points where the guards would not let it move at all, ``coupled``
    points where every move also moved another dimension (so a difference could not be
    credited to this one), ``live_at`` the ``(from, to)`` move that first changed the image.
    """
    log = log or getattr(runner, "log", print)
    space = runner.space
    names = [p.name for p in space.params]
    todo = [i for i, p in enumerate(space.params) if p.role == "reduction"]
    status = {names[i]: {"live": False, "tested": 0, "pinned": 0, "coupled": 0, "live_at": None,
                         "dense": False} for i in todo}
    rng = np.random.default_rng(seed)
    old_ia = runner.ia
    runner.ia = int(annulus)
    points = []                                  # (x0, base image) per point, for the dense pass

    def probe(i, x0, base, values, tag):
        """Move dimension ``i`` of ``x0`` to each of ``values`` in turn until the image
        changes.  Returns (moved, coupled, live)."""
        st = status[names[i]]
        moved = coupled = False
        for v in values:
            x1 = x0.copy()
            x1[i] = v
            x1 = runner._project(x1, is_random=False)
            if abs(x1[i] - x0[i]) < 1e-9:              # the guards put it back
                continue
            others = np.abs(np.delete(x1 - x0, i)) > 1e-9
            if others.any():                           # it dragged another dimension along
                coupled = True
                continue
            moved = True
            img = runner._reduce(space.decode(x1), None, tag=f"live_{names[i]}_{tag}").image
            st["tested"] += 1
            if not _same(base, img):
                st["live"] = True
                st["live_at"] = (float(x0[i]), float(x1[i]))
                return moved, coupled, True
        return moved, coupled, False

    try:
        for pt in range(max(int(n_points), 1)):
            if not todo:
                break
            x0 = space.default_vector() if pt == 0 else space.random(rng)
            x0 = runner._project(np.asarray(x0, float), is_random=pt > 0)
            base = runner._reduce(space.decode(x0), None, tag=f"live_base{pt}").image
            points.append((x0, base))
            still = []
            for i in todo:
                st = status[names[i]]
                # Every candidate, not just the first that moves: a dimension can be inert in
                # one direction and live in another.  pyKLIP's ADI+RDI keeping its 3 best
                # frames reduces exactly like ADI when those 3 all come from the other roll,
                # yet RDI from the same point is a different reduction.
                moved, coupled, live = probe(i, x0, base, _candidates(space, x0, i), str(pt))
                if not moved:
                    st["coupled" if coupled else "pinned"] += 1
                if not live:
                    still.append(i)
            todo = still
        if dense and todo:
            log(f"  liveness: {len(todo)} dimension(s) unmoved by the bounds and midpoint; "
                f"re-probing each on a {int(dense)}-value grid")
            for pt, (x0, base) in enumerate(points):
                still = []
                for i in todo:
                    status[names[i]]["dense"] = True
                    moved, coupled, live = probe(i, x0, base, _grid(space, x0, i, int(dense)), f"g{pt}")
                    if not live:
                        still.append(i)
                todo = still
                if not todo:
                    break
    finally:
        runner.ia = old_ia
    for n, st in status.items():
        if st["live"]:
            verdict = "live" if not st["dense"] else (
                f"live -- acts in a band: {st['live_at'][0]:.4g} -> {st['live_at'][1]:.4g}; inert at the "
                f"bounds and the midpoint")
        elif st["tested"]:
            verdict = (f"DEAD -- inert at every value probed"
                       + (f" (bounds, midpoint and a {int(dense)}-value grid)" if st["dense"] else ""))
        else:
            verdict = "UNTESTED -- the guards never let it move alone"
        log(f"  liveness: {n:16s} {verdict}  ({st['tested']} move(s) reduced, pinned at "
            f"{st['pinned']}, coupled at {st['coupled']} point(s))")
    return status


def dead_dimensions(status: Dict[str, Dict[str, Any]]) -> List[str]:
    """Dimensions that never changed the reduction -- dead, or never movable alone."""
    return [n for n, st in status.items() if not st["live"]]
