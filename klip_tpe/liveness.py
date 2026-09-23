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
clean image, and asks whether anything changed.  A handful of reductions against hours of
search.
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


def check_live_dimensions(runner, n_points: int = 3, annulus: int = 0, seed: int = 0,
                          log: Optional[Callable[[str], None]] = None) -> Dict[str, Dict[str, Any]]:
    """For every searched reduction dimension of ``runner.space``: does moving it alone change
    the clean reduction of annulus ``annulus``?

    Tried at the default configuration first and then at up to ``n_points - 1`` random
    feasible ones, stopping for each dimension as soon as one move changes the image (so a
    healthy space costs ``1 + ndim`` reductions).  Returns ``{name: {"live", "tested",
    "pinned", "coupled"}}``: ``tested`` moves reduced, ``pinned`` points where the guards
    would not let it move at all, ``coupled`` points where every move also moved another
    dimension (so a difference could not be credited to this one).
    """
    log = log or getattr(runner, "log", print)
    space = runner.space
    names = [p.name for p in space.params]
    todo = [i for i, p in enumerate(space.params) if p.role == "reduction"]
    status = {names[i]: {"live": False, "tested": 0, "pinned": 0, "coupled": 0} for i in todo}
    rng = np.random.default_rng(seed)
    old_ia = runner.ia
    runner.ia = int(annulus)
    try:
        for pt in range(max(int(n_points), 1)):
            if not todo:
                break
            x0 = space.default_vector() if pt == 0 else space.random(rng)
            x0 = runner._project(np.asarray(x0, float), is_random=pt > 0)
            base = runner._reduce(space.decode(x0), None, tag=f"live_base{pt}").image
            still = []
            for i in todo:
                st = status[names[i]]
                moved = coupled = False
                # Every candidate, not just the first that moves: a dimension can be inert in
                # one direction and live in another.  pyKLIP's ADI+RDI keeping its 3 best
                # frames reduces exactly like ADI when those 3 all come from the other roll,
                # yet RDI from the same point is a different reduction.
                for v in _candidates(space, x0, i):
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
                    img = runner._reduce(space.decode(x1), None, tag=f"live_{names[i]}_{pt}").image
                    st["tested"] += 1
                    if not _same(base, img):
                        st["live"] = True
                        break
                if not moved:
                    st["coupled" if coupled else "pinned"] += 1
                if not st["live"]:
                    still.append(i)
            todo = still
    finally:
        runner.ia = old_ia
    for n, st in status.items():
        verdict = ("live" if st["live"] else
                   "DEAD -- moving it alone changed nothing" if st["tested"] else
                   "UNTESTED -- the guards never let it move alone")
        log(f"  liveness: {n:16s} {verdict}  ({st['tested']} move(s) reduced, pinned at "
            f"{st['pinned']}, coupled at {st['coupled']} point(s))")
    return status


def dead_dimensions(status: Dict[str, Dict[str, Any]]) -> List[str]:
    """Dimensions that never changed the reduction -- dead, or never movable alone."""
    return [n for n, st in status.items() if not st["live"]]
