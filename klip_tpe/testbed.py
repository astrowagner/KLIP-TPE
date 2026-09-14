"""Synthetic ground-truth testbed for the proposal density (port of ``near2_mvtest``).

There is no imagery here: the optimizer only ever sees ``(x, score)`` pairs, so the
pipeline is replaced by a formula with a *known* optimum and *known* noise.

* ``T(x)`` -- true score in ``[0, 10]``: ``ngroup`` groups of ``gsize`` dims, each an
  anisotropic **rotated ridge** (one broad direction sigma=2, the rest narrow
  sigma=``ridge``) scored ``1/(1+s)`` and averaged x10.  The rational form is
  essential: a Gaussian bump in 9-D is exponentially sparse and the warm-up finds
  nothing (a failure that is invisible on real data).
* ``Y = T(x) + N(0, noise)`` -- what the optimizer sees.
* Report ``max(Y)`` (observed best) and ``T(argmax Y)`` (TRUE at best).  Their gap is
  the *optimism* the production validation stage exists to defend against.

``compare_density_models`` runs the three density models with identical landscape
and warm-up per seed (paired), so the only difference is the proposal density.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .optimizers import tpe_propose
from .space import Param, SearchSpace

__all__ = ["RidgeObjective", "make_space", "run_tpe_on_objective", "compare_density_models"]


@dataclass
class RidgeObjective:
    nblock: int = 6
    bdim: int = 9
    ridge: float = 0.40
    transposed: bool = False
    seed: int = 1000

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        nb, bd = self.nblock, self.bdim
        self.ndim = nb * bd + 2
        if self.transposed:
            self.ngroup, self.gsize = bd, nb
            self.gidx = np.array([[g + k * bd for k in range(nb)] for g in range(bd)])
        else:
            self.ngroup, self.gsize = nb, bd
            self.gidx = np.array([[g * bd + k for k in range(bd)] for g in range(nb)])
        self.ctr = 0.15 + 0.70 * rng.random((self.ngroup, self.gsize))
        self.rot = np.empty((self.ngroup, self.gsize, self.gsize))
        for b in range(self.ngroup):
            m = rng.standard_normal((self.gsize, self.gsize))
            q, _ = np.linalg.qr(m)
            self.rot[b] = q.T          # rows = orthonormal directions

    def true(self, x: np.ndarray) -> float:
        x = np.asarray(x, float)
        tot = 0.0
        for b in range(self.ngroup):
            u = x[self.gidx[b]] - self.ctr[b]
            v = self.rot[b] @ u
            s = (v[0] / 2.0) ** 2 + np.sum((v[1:] / self.ridge) ** 2)
            tot += 1.0 / (1.0 + s)
        return 10.0 * tot / self.ngroup

    def blocks(self) -> List[List[int]]:
        """The 'by night' proposal blocking (always contiguous groups + selection dims)."""
        nb, bd = self.nblock, self.bdim
        return [[b * bd + k for k in range(bd)] for b in range(nb)] + [[nb * bd, nb * bd + 1]]


def make_space(ndim: int) -> SearchSpace:
    return SearchSpace([Param(f"x{i}", 0.0, 1.0) for i in range(ndim)])


def run_tpe_on_objective(obj: RidgeObjective, blocks, n_iter: int = 400, n_init: int = 50,
                         noise: float = 0.25, seed: int = 0, gamma: float = 0.25, ncand: int = 48,
                         pbest: float = 0.5) -> Dict[str, np.ndarray]:
    """One run; returns running observed-best and TRUE-at-observed-best curves.
    The warm-up stream depends only on ``seed`` so different density models are paired."""
    space = make_space(obj.ndim)
    X = np.zeros((n_iter, obj.ndim))
    Y = np.zeros(n_iter)
    T = np.zeros(n_iter)
    rw = np.random.default_rng(7000 + seed)
    for it in range(n_init):
        X[it] = rw.random(obj.ndim)
        T[it] = obj.true(X[it])
        Y[it] = T[it] + noise * rw.standard_normal()
    rp = np.random.default_rng(31000 + seed + 100 * (hash(str(blocks)) % 7))
    for it in range(n_init, n_iter):
        x = tpe_propose(X[:it], Y[:it], space, rp, gamma=gamma, ncand=ncand, pbest=pbest, blocks=blocks)
        X[it] = np.clip(x, 0, 1)
        T[it] = obj.true(X[it])
        Y[it] = T[it] + noise * rp.standard_normal()
    best_o = np.maximum.accumulate(Y)
    idx = np.array([int(np.argmax(Y[:i + 1])) for i in range(n_iter)])
    best_t = T[idx]
    return {"X": X, "Y": Y, "T": T, "best_obs": best_o, "true_at_best": best_t}


def compare_density_models(nblock: int = 6, bdim: int = 9, n_iter: int = 400, n_init: int = 50,
                           nseed: int = 8, noise: float = 0.25, ridge: float = 0.40,
                           transposed: bool = False, modes: Sequence[str] = ("univariate", "block", "full"),
                           pbest: float = 0.5, log=print) -> Dict[str, Dict[str, float]]:
    """Paired comparison of density models.  Returns per-mode mean/sd of TRUE-at-best,
    observed best, optimism, and the paired difference vs the first mode."""
    res: Dict[str, List[float]] = {m: [] for m in modes}
    obs: Dict[str, List[float]] = {m: [] for m in modes}
    for s in range(nseed):
        obj = RidgeObjective(nblock, bdim, ridge, transposed, seed=1000 + s)
        for m in modes:
            blocks = None if m == "univariate" else ("full" if m == "full" else obj.blocks())
            r = run_tpe_on_objective(obj, blocks, n_iter, n_init, noise, seed=s, pbest=pbest)
            res[m].append(float(r["true_at_best"][-1]))
            obs[m].append(float(r["best_obs"][-1]))
            log(f"  {m:<10s} seed {s}: observed-best {obs[m][-1]:6.2f}  TRUE at best {res[m][-1]:6.2f}")
    out: Dict[str, Dict[str, float]] = {}
    ref = np.array(res[modes[0]])
    for m in modes:
        t = np.array(res[m])
        o = np.array(obs[m])
        d = t - ref
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        out[m] = {"true_mean": float(t.mean()), "true_sd": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
                  "obs_mean": float(o.mean()), "optimism": float((o - t).mean()),
                  "diff_vs_ref": float(d.mean()), "diff_t": float(d.mean() / se) if se and se > 0 else np.nan,
                  "wins": int((d > 0).sum())}
        log(f"{m:<10s} TRUE@best {t.mean():.2f} +/- {out[m]['true_sd']:.2f}  observed {o.mean():.2f}  "
            f"optimism {out[m]['optimism']:.2f}  diff vs {modes[0]} {d.mean():+.2f} (t={out[m]['diff_t']:.2f}, wins {out[m]['wins']}/{len(d)})")
    return out
