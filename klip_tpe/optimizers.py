"""Search strategies behind one ``ask``/``tell`` interface.

:class:`TPE` is a faithful port of the IDL ``near2m_propose`` / ``near2m_bw`` /
``near2m_kde`` core plus the three adaptations that earn their keep in high
dimension (``pbest`` best-seeded candidates, ``p_local`` elite local moves and
``explore_frac`` epsilon-exploration).  :class:`RandomSearch` and
:class:`GridSearch` are the matched-budget baselines used by the benchmark harness.

All strategies are pure functions of the :class:`History` they are handed, so a
checkpointed history is all that is needed to resume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .space import SearchSpace

__all__ = ["History", "Proposal", "Optimizer", "TPE", "RandomSearch", "GridSearch",
           "parzen_bandwidth", "parzen_density", "block_log_density", "resolve_blocks", "tpe_propose"]


# ----------------------------------------------------------------------------
# history container
# ----------------------------------------------------------------------------
@dataclass
class History:
    """Evaluated vectors, scores and per-trial flags.

    ``y`` holds ``nan`` for failed / not-yet-scored trials.  ``valid`` is the mask of
    trials that may train the model.  ``flags[i]["phase"]`` records how the trial
    was proposed (``seed, warmup, tpe, local, explore, random, grid, resume``).
    """

    ndim: int
    X: np.ndarray = field(default=None)
    y: np.ndarray = field(default=None)
    flags: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.X is None:
            self.X = np.zeros((0, self.ndim))
        if self.y is None:
            self.y = np.zeros(0)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.y)

    def append(self, x: np.ndarray, y: float, flags: Optional[Dict[str, Any]] = None) -> int:
        self.X = np.vstack([self.X, np.asarray(x, dtype=float)[None, :]])
        self.y = np.append(self.y, np.nan if y is None else float(y))
        self.flags.append(dict(flags or {}))
        return len(self) - 1

    def set_score(self, i: int, y: float, **extra) -> None:
        self.y[i] = np.nan if y is None else float(y)
        self.flags[i].update(extra)

    def best(self) -> Tuple[int, float]:
        """Index and score of the best *valid* trial (ties -> earliest)."""
        v = self.valid
        if not v.any():
            return -1, -np.inf
        yv = np.where(v, self.y, -np.inf)
        i = int(np.argmax(yv))
        return i, float(yv[i])

    def order(self) -> np.ndarray:
        """Valid trial indices, best first (stable: ties keep evaluation order)."""
        idx = np.flatnonzero(self.valid)
        return idx[np.argsort(-self.y[idx], kind="stable")]

    def to_dict(self) -> Dict[str, Any]:
        return {"ndim": self.ndim, "X": self.X.tolist(),
                "y": [None if not np.isfinite(v) else float(v) for v in self.y],
                "flags": self.flags}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "History":
        h = cls(ndim=int(d["ndim"]))
        h.X = np.asarray(d["X"], dtype=float).reshape(-1, h.ndim)
        h.y = np.array([np.nan if v is None else float(v) for v in d["y"]])
        h.flags = [dict(f) for f in d["flags"]]
        return h


@dataclass
class Proposal:
    x: np.ndarray
    flags: Dict[str, Any]


# ----------------------------------------------------------------------------
# Parzen helpers (near2m_bw / near2m_kde)
# ----------------------------------------------------------------------------
def parzen_bandwidth(s: np.ndarray, lo: float, hi: float, floor_frac: float = 0.08) -> float:
    """Scott/Silverman bandwidth ``1.06 std n^-1/5`` floored at ``floor_frac*(hi-lo)``.
    The floor keeps the search from collapsing once the good set becomes tight."""
    s = np.asarray(s, dtype=float)
    n = s.size
    if n < 2:
        return 0.5 * (hi - lo)
    h = 1.06 * float(np.std(s, ddof=1)) * n ** (-0.2)
    return max(h, floor_frac * (hi - lo))


def parzen_density(x: float, s: np.ndarray, h: float, lo: float, hi: float,
                   prior_weight: float = 0.25) -> float:
    """1-D Gaussian Parzen density at ``x`` mixed with a uniform prior over [lo, hi]."""
    s = np.asarray(s, dtype=float)
    n = s.size
    if n < 1 or hi <= lo:
        return 1.0 / max(hi - lo, 1e-12)
    d = float(np.sum(np.exp(-0.5 * ((x - s) / h) ** 2))) / (n * h * np.sqrt(2.0 * np.pi))
    return (1.0 - prior_weight) * d + prior_weight / (hi - lo)


def _categorical_pmf(s: np.ndarray, ncat: int, prior_weight: float) -> np.ndarray:
    counts = np.bincount(np.clip(np.round(s).astype(int), 0, ncat - 1), minlength=ncat).astype(float)
    n = counts.sum()
    emp = counts / n if n > 0 else np.full(ncat, 1.0 / ncat)
    return (1.0 - prior_weight) * emp + prior_weight / ncat


def block_log_density(C: np.ndarray, X: np.ndarray, idx: np.ndarray, h: np.ndarray, dims: Sequence[int],
                      lo: np.ndarray, hi: np.ndarray, prior_weight: float = 0.25) -> np.ndarray:
    """Log of the mixture-of-products Parzen density over ``dims`` at each candidate
    row of ``C`` (``near2m_blkden``)::

        p(x) = (1-wp) (1/N) sum_i prod_d K(x_d - X[i,d]; h_d)  +  wp prod_d 1/(hi_d-lo_d)

    Every mixture component is anchored on ONE real observation, so a candidate is
    scored against configurations that actually occurred together.  Bandwidths stay
    diagonal (no covariance is estimated).  logsumexp throughout and the uniform prior
    is mixed in log space, so the result stays finite when the kernel sum underflows
    (it does for large blocks).
    """
    dims = np.asarray(dims, int)
    C = np.atleast_2d(np.asarray(C, float))
    Xi = X[np.asarray(idx, int)][:, dims]                      # (N, nd)
    Cd = C[:, dims]                                           # (nc, nd)
    hd = np.asarray(h, float)[dims]
    dl = (Cd[:, None, :] - Xi[None, :, :]) / hd[None, None, :]
    e = -0.5 * np.sum(dl * dl, axis=2)                         # (nc, N)
    cst = -np.sum(np.log(hd * np.sqrt(2.0 * np.pi)))
    lpr = -np.sum(np.log(np.maximum(hi[dims] - lo[dims], 1e-30)))
    m = e.max(axis=1)
    la = np.log(1.0 - prior_weight) + cst + m + np.log(np.sum(np.exp(e - m[:, None]), axis=1)) - np.log(Xi.shape[0])
    lb = np.log(prior_weight) + lpr
    mm = np.maximum(la, lb)
    return mm + np.log(np.exp(la - mm) + np.exp(lb - mm))


def resolve_blocks(space: SearchSpace, blocks) -> List[np.ndarray]:
    """Normalise the ``blocks`` option into a list of index arrays over the
    non-categorical dims.

    ``None`` / ``"univariate"``: one block per dimension (factorized TPE);
    ``"partitions"``: one block per partition plus one for the global/selection
    dims (the natural per-epoch blocking);
    ``"full"``: a single block;
    an explicit list of lists of dim names or indices.
    """
    cont = [i for i in range(space.ndim) if not space.is_categorical[i]]
    if blocks is None or blocks == "univariate":
        return [np.array([i]) for i in cont]
    if blocks == "full":
        return [np.array(cont)] if cont else []
    if blocks == "partitions":
        if not space.partitions:
            return [np.array([i]) for i in cont]      # no partitions -> univariate reference
        out: List[np.ndarray] = []
        seen = set()
        for pid in space.partitions:
            d = [i for i in space.dims_of_partition(pid) if i in cont]
            if d:
                out.append(np.array(d))
                seen.update(d)
        rest = [i for i in cont if i not in seen]
        if rest:
            out.append(np.array(rest))
        if not out:
            return [np.array([i]) for i in cont]
        return out
    out = []
    seen = set()
    for b in blocks:
        d = [space.index(v) if isinstance(v, str) else int(v) for v in b]
        d = [i for i in d if i in cont]
        if d:
            out.append(np.array(d))
            seen.update(d)
    for i in cont:
        if i not in seen:
            out.append(np.array([i]))
    return out


def tpe_propose(X: np.ndarray, y: np.ndarray, space: SearchSpace, rng: np.random.Generator,
                gamma: float = 0.25, ncand: int = 48, pbest: float = 0.0,
                prior_weight: float = 0.25, bw_floor: float = 0.08,
                blocks=None) -> np.ndarray:
    """One TPE proposal from history ``(X, y)`` (valid rows only).

    1. sort by score, top ``gamma`` fraction = good (l), rest = bad (g);
    2. Parzen estimators: per dimension (``blocks=None``, univariate -- the IDL
       reference behaviour) or mixture-of-products within each block of ``blocks``
       and independent across blocks (block-multivariate); categorical dims always
       use a smoothed pmf;
    3. ``ncand`` candidates; each block is seeded from ONE random good observation
       (or all blocks from the single best with probability ``pbest``) and jittered
       per dimension by the good-set bandwidth;
    4. return the candidate maximising the summed log-density ratio ``log l - log g``.

    Block-multivariate scoring is a noise-robustness mechanism: it stops the
    proposal from recombining pieces of good configurations that were never good
    together (see ``TPE_PYTHON_PORT_ADDENDUM_density.md``).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, ndim = X.shape
    lo, hi = space.lo, space.hi
    cat = space.is_categorical
    ncat = space.n_categories
    if n < 2:
        return space.random(rng)
    ord_ = np.argsort(-y, kind="stable")
    ng = max(int(round(gamma * n)), 1)
    if ng >= n:
        ng = n - 1
    good, bad = ord_[:ng], ord_[ng:]

    hg = np.array([parzen_bandwidth(X[good, d], lo[d], hi[d], bw_floor) for d in range(ndim)])
    hb = np.array([parzen_bandwidth(X[bad, d], lo[d], hi[d], bw_floor) for d in range(ndim)])
    pmf_g = {d: _categorical_pmf(X[good, d], ncat[d], prior_weight) for d in range(ndim) if cat[d]}
    pmf_b = {d: _categorical_pmf(X[bad, d], ncat[d], prior_weight) for d in range(ndim) if cat[d]}
    blk = resolve_blocks(space, blocks)

    C = np.empty((ncand, ndim))
    for c in range(ncand):
        anchor = good[0] if rng.random() < pbest else -1
        # one anchor observation per block (univariate: per dimension), or the running
        # best for every block with probability pbest
        for b in blk:
            j = anchor if anchor >= 0 else good[int(rng.random() * ng)]
            for d in b:
                v = X[j, d] + hg[d] * rng.standard_normal()
                C[c, d] = min(max(v, lo[d]), hi[d])
        for d in range(ndim):
            if cat[d]:
                C[c, d] = float(rng.choice(ncat[d], p=pmf_g[d]))

    ei = np.zeros(ncand)
    for b in blk:
        if len(b) == 1:
            d = int(b[0])
            for c in range(ncand):
                ld = parzen_density(C[c, d], X[good, d], hg[d], lo[d], hi[d], prior_weight)
                gd = parzen_density(C[c, d], X[bad, d], hb[d], lo[d], hi[d], prior_weight)
                ei[c] += np.log(ld) - np.log(gd)
        else:
            ei += block_log_density(C, X, good, hg, b, lo, hi, prior_weight) \
                - block_log_density(C, X, bad, hb, b, lo, hi, prior_weight)
    for d in range(ndim):
        if cat[d]:
            ci = C[:, d].astype(int)
            ei += np.log(pmf_g[d][ci]) - np.log(pmf_b[d][ci])
    best = int(np.argmax(ei))
    return space.sanitize(C[best])


# ----------------------------------------------------------------------------
# strategies
# ----------------------------------------------------------------------------
class Optimizer:
    """Base class.  ``ask(history, rng, n)`` returns ``n`` proposals computed from the
    same posterior (batch mode); ``tell`` is a no-op because the history object is
    the single source of truth (this is what makes resume trivial)."""

    name = "base"

    def __init__(self, space: SearchSpace, n_init: int = 40, warmstart_tie: bool = False,
                 link_params: Sequence[str] = ()):
        self.space = space
        self.n_init = int(n_init)
        self.warmstart_tie = bool(warmstart_tie)
        self.link_params = [b for b in link_params if b]

    def _finish(self, x: np.ndarray) -> np.ndarray:
        x = self.space.sanitize(x)
        if self.link_params:
            x = self.space.tie(x, self.link_params)
        return x

    def ask(self, history: History, rng: np.random.Generator, n: int = 1) -> List[Proposal]:
        raise NotImplementedError

    def tell(self, history: History, i: int, y: float) -> None:  # pragma: no cover
        pass

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "n_init": self.n_init, "warmstart_tie": self.warmstart_tie,
                "link_params": list(self.link_params)}


class RandomSearch(Optimizer):
    name = "random"

    def ask(self, history: History, rng: np.random.Generator, n: int = 1) -> List[Proposal]:
        return [Proposal(self._finish(self.space.random(rng, tie_all=self.warmstart_tie)),
                         {"phase": "random"}) for _ in range(n)]


class GridSearch(Optimizer):
    """Regular grid over ``axes`` (default: all reduction dims), cell-centred and
    refined in place when the cells run out.  Dims outside ``axes`` sit at the
    default vector.

    ``gpts = floor(budget**(1/naxes))`` points per axis, so the first pass fits inside
    the budget; ``floor`` rather than ``round`` because rounding up overruns it and the
    overrun is what used to wrap.  Few points per axis is the honest grid-search regime
    in many dimensions -- ``budget**(1/d)`` is the Bergstra & Bengio (2012) argument for
    why grid search loses to random search as ``d`` grows -- so a small ``gpts`` is
    reported, not corrected.  Two things about the old implementation were defects
    rather than that regime:

    *Cell centres, not endpoints.*  ``lo + span*i/(g-1)`` puts both points of a
    two-point axis on the bounds of the box, and in this problem the bounds are where
    the reduction degenerates: ``corr_thresh = 1`` keeps no frames, the largest
    ``filter`` smooths the signal away, ``k_klip = 1`` subtracts almost nothing.  A
    nine-axis grid at ``gpts = 2`` therefore spent its whole budget on the 512 corners
    of the box and never evaluated one interior configuration.  Cells are now centred --
    ``lo + span*(i + 0.5)/g`` -- which is the usual convention for a grid design and
    never lands on a degenerate face.

    *Refinement, not wrap-around.*  Cell ``k`` used to be ``k % g**naxes``, so once the
    cells ran out the grid re-proposed configurations it had already evaluated.  The
    objective is deterministic given the configuration, so those evaluations bought
    nothing: a 1000-evaluation run over 512 cells was charged 1000 reductions for 449
    distinct configurations.  Exhausting a stage now doubles ``gpts``; because the cells
    are centred, every node of stage ``s+1`` is new, so the budget always buys distinct
    configurations.
    """

    name = "grid"

    def __init__(self, space: SearchSpace, budget: int, axes: Optional[Sequence[str]] = None,
                 base_vector: Optional[np.ndarray] = None, gpts: Optional[int] = None, **kw):
        super().__init__(space, n_init=0, **kw)
        if axes is None:
            axes = [space.params[i].name for i in space.reduction_dims]
            # Axis order is not cosmetic: the cell index is an odometer, so axis 0 cycles
            # through all its values every g cells while the last axis only moves once the
            # budget has covered g**(naxes-1) cells -- which a real budget never does.  A
            # param carrying an explicit ``grid`` is one whose author wrote down the values
            # worth scanning, so those go first and are the ones a truncated pass resolves.
            # k_klip is the case that matters here: declared last, it took three values in a
            # thousand evaluations while ``bin`` took six.
            axes = ([n for n in axes if space[n].grid is not None]
                    + [n for n in axes if space[n].grid is None])
        self.axes = list(axes)
        self.budget = int(budget)
        na = max(len(self.axes), 1)
        if gpts is None:
            # floor: g**na must fit the budget, or the last cells of the pass wrap.
            g = int(max(self.budget, 1) ** (1.0 / na) + 1e-9)
            while g > 2 and g ** na > max(self.budget, 1):
                g -= 1
            self.gpts = max(g, 2)
        else:
            self.gpts = max(int(gpts), 2)
        self.base_vector = space.default_vector() if base_vector is None else np.asarray(base_vector, float)

    # ---------------------------------------------------------------- geometry
    def stage_of(self, k: int) -> Tuple[int, int, int]:
        """``(gpts, cell_within_stage, stage)`` for the ``k``-th grid proposal.

        Stage ``s`` has ``gpts * 2**s`` points per axis and ``(gpts * 2**s)**naxes``
        cells.  Stages are walked in order, so a budget larger than one stage keeps
        producing new configurations instead of repeating the first stage.
        """
        na = max(len(self.axes), 1)
        g, s, k = self.gpts, 0, int(k)
        while True:
            n = g ** na
            if k < n or s >= 24:                 # 24 doublings is past any real budget
                return g, k % max(n, 1), s
            k -= n
            g *= 2
            s += 1

    def cell(self, k: int) -> np.ndarray:
        x = self.base_vector.copy()
        g, kk, _ = self.stage_of(k)
        for a, name in enumerate(self.axes):
            i = (kk // (g ** a)) % g
            p = self.space[name]
            frac = (i + 0.5) / g                 # cell centre, never a box face
            if p.grid is not None:
                x[self.space.index(name)] = p.grid[min(int(frac * len(p.grid)), len(p.grid) - 1)]
            else:
                x[self.space.index(name)] = p.lo + p.span * frac
        return x

    def ask(self, history: History, rng: np.random.Generator, n: int = 1) -> List[Proposal]:
        ngrid = sum(1 for f in history.flags if f.get("phase") == "grid")
        out = []
        for k in range(n):
            g, cell, stage = self.stage_of(ngrid + k)
            out.append(Proposal(self._finish(self.cell(ngrid + k)),
                                {"phase": "grid", "cell": ngrid + k, "gpts": g, "stage": stage}))
        return out

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        na = max(len(self.axes), 1)
        d.update(axes=self.axes, gpts=self.gpts, budget=self.budget, cell_centred=True,
                 n_cells_stage0=self.gpts ** na,
                 stages=self.stage_of(max(self.budget - 1, 0))[2] + 1,
                 # True when the budget cannot give even three levels per axis, so the
                 # first pass only contrasts a low against a high value on each one.
                 # Worth stating in a caption: it is the budget**(1/d) limit, not a bug.
                 coarse_by_budget=self.gpts < 3)
        return d


class TPE(Optimizer):
    """Tree-structured Parzen Estimator with the KLIP-TPE adaptations.

    Parameters
    ----------
    n_init
        number of history rows (including any seed) before guided proposals start.
    gamma, ncand, prior_weight, bw_floor
        Parzen split fraction, candidates per proposal, uniform-prior mixing weight and
        bandwidth floor (fraction of range).
    pbest
        fraction of candidates seeded from the single best observation in *all*
        dimensions.  Default 0: on noisy objectives the running best is usually a
        noise winner and anchoring to it imports the selection bias into the
        proposal (-1.1 in TRUE score at noise 0.40 on the synthetic testbed,
        0/24 seeds won; addendum 2).  Kept as an option for parity experiments.
    p_local, n_elite
        probability of a local move (re-jitter ONE reduction dim of a random
        top-``n_elite`` config) instead of a TPE proposal.
    explore_frac
        probability (checked after the local coin) of a uniform random draw.
    blocks
        density model: ``"univariate"``/``None`` (default; factorized per-dimension
        Parzen estimators -- the IDL reference behaviour and the best configuration in
        the 2x2x3 factorial of addendum 2 once ``pbest=0``), ``"partitions"``
        (block-multivariate, one block per partition + one for the global/selection
        dims), ``"full"`` (single block; dominated, kept as a negative control) or an
        explicit list of dim-name lists.
    warmstart_tie
        tie all partition slots of each base in warm-up draws (global warm start).
    link_params
        bases whose partition slots are tied on every proposal.
    """

    name = "tpe"

    def __init__(self, space: SearchSpace, n_init: int = 40, gamma: float = 0.25, ncand: int = 48,
                 prior_weight: float = 0.25, bw_floor: float = 0.08, pbest: float = 0.0,
                 p_local: float = 0.15, n_elite: int = 5, explore_frac: float = 0.15,
                 blocks="univariate", warmstart_tie: bool = False, link_params: Sequence[str] = ()):
        super().__init__(space, n_init=n_init, warmstart_tie=warmstart_tie, link_params=link_params)
        self.blocks = blocks
        self.gamma, self.ncand = float(gamma), int(ncand)
        self.prior_weight, self.bw_floor = float(prior_weight), float(bw_floor)
        self.pbest, self.p_local, self.n_elite = float(pbest), float(p_local), int(n_elite)
        self.explore_frac = float(explore_frac)

    # -- individual moves ----------------------------------------------------
    def _local_move(self, history: History, rng: np.random.Generator) -> np.ndarray:
        sp = self.space
        order = history.order()
        if order.size == 0:
            return sp.random(rng)
        neu = int(np.clip(self.n_elite, 1, order.size))
        base = order[int(rng.random() * neu)]
        x = history.X[base].copy()
        ng = int(np.clip(round(self.gamma * order.size), 1, max(order.size - 1, 1)))
        good = order[:ng]
        rd = sp.reduction_dims
        d = int(rd[int(rng.random() * rd.size)])
        p = sp.params[d]
        if p.kind == "categorical":
            others = [c for c in range(len(p.choices)) if c != int(round(x[d]))]
            if others:
                x[d] = float(rng.choice(others))
        else:
            bw = parzen_bandwidth(history.X[good, d], p.lo, p.hi, self.bw_floor)
            x[d] = x[d] + bw * rng.standard_normal()
        return x

    def propose_one(self, history: History, rng: np.random.Generator) -> Proposal:
        sp = self.space
        if len(history) < self.n_init:
            x = sp.random(rng, tie_all=self.warmstart_tie)
            return Proposal(self._finish(x), {"phase": "warmup"})
        if rng.random() < self.p_local:
            return Proposal(self._finish(self._local_move(history, rng)), {"phase": "local"})
        if rng.random() < self.explore_frac:
            return Proposal(self._finish(sp.random(rng)), {"phase": "explore"})
        v = history.valid
        if v.sum() < 2:
            return Proposal(self._finish(sp.random(rng)), {"phase": "explore"})
        x = tpe_propose(history.X[v], history.y[v], sp, rng, gamma=self.gamma, ncand=self.ncand,
                        pbest=self.pbest, prior_weight=self.prior_weight, bw_floor=self.bw_floor,
                        blocks=self.blocks)
        return Proposal(self._finish(x), {"phase": "tpe"})

    def ask(self, history: History, rng: np.random.Generator, n: int = 1) -> List[Proposal]:
        # batch proposals share the same posterior (HPBoo-style); the warm-up counter
        # advances with the batch so a batch never over-runs n_init.
        out = []
        fake = History(history.ndim, history.X.copy(), history.y.copy(), list(history.flags))
        for _ in range(n):
            pr = self.propose_one(fake, rng)
            out.append(pr)
            if len(fake) < self.n_init:
                fake.append(pr.x, np.nan, pr.flags)
        return out

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(gamma=self.gamma, ncand=self.ncand, prior_weight=self.prior_weight,
                 bw_floor=self.bw_floor, pbest=self.pbest, p_local=self.p_local,
                 n_elite=self.n_elite, explore_frac=self.explore_frac,
                 blocks=self.blocks if isinstance(self.blocks, (str, type(None))) else [list(map(str, b)) for b in self.blocks],
                 resolved_blocks=[[self.space.params[i].name for i in b] for b in resolve_blocks(self.space, self.blocks)])
        return d


def make_optimizer(mode: str, space: SearchSpace, **kw) -> Optimizer:
    mode = mode.lower()
    if mode == "tpe":
        return TPE(space, **kw)
    if mode == "random":
        return RandomSearch(space, n_init=kw.get("n_init", 0), warmstart_tie=kw.get("warmstart_tie", False),
                            link_params=kw.get("link_params", ()))
    if mode == "grid":
        return GridSearch(space, budget=kw["budget"], axes=kw.get("axes"),
                          base_vector=kw.get("base_vector"), warmstart_tie=False,
                          link_params=kw.get("link_params", ()))
    raise ValueError(f"unknown search mode {mode!r}")
