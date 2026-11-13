"""Search strategies behind one ``ask`` and ``tell`` interface.

:class:`TPE` succeeds the IDL predecessor's ``near2m_propose``, ``near2m_bw`` and ``near2m_kde``
core, with the three adaptations that earn their keep in high dimension: ``pbest`` best-seeded
candidates, ``p_local`` elite local moves and ``explore_frac`` epsilon-exploration.
:class:`RandomSearch` and :class:`GridSearch` are the matched-budget baselines of the benchmark
harness.

Every strategy is a pure function of the :class:`History` it is handed, so a checkpointed history
is all a resume needs.
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
    """Evaluated vectors, their scores and each trial's flags.

    Attributes
    ----------
    ndim : int
        Number of dimensions.
    X : ndarray, shape (n, ndim), default None
        The ``n`` vectors.  None starts empty.
    y : ndarray, shape (n,), default None
        Their scores, NaN for a trial failed or not yet scored.  None starts empty.
    flags : list of dict
        Each trial's flags, ``phase`` recording how it was proposed (``seed``, ``warmup``, ``tpe``,
        ``local``, ``explore``, ``random``, ``grid``, ``resume``).
    """

    ndim: int
    X: np.ndarray = field(default=None)
    y: np.ndarray = field(default=None)
    flags: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Start ``X`` and ``y`` empty where they are None.
        """
        if self.X is None:
            self.X = np.zeros((0, self.ndim))
        if self.y is None:
            self.y = np.zeros(0)

    def __len__(self) -> int:
        """Return the number of trials.

        Returns
        -------
        n : int
            The number of rows of ``X``.
        """
        return int(self.X.shape[0])

    @property
    def valid(self) -> np.ndarray:
        """Mask of the trials that may train the model, those with a finite score.
        """
        return np.isfinite(self.y)

    def append(self, x: np.ndarray, y: float, flags: Optional[Dict[str, Any]] = None) -> int:
        """Append a trial.

        Parameters
        ----------
        x : ndarray, shape (ndim,)
            The vector.
        y : float or None
            Its score, None or NaN for one not yet scored.
        flags : dict or None, default None
            Its flags.

        Returns
        -------
        i : int
            The trial's index.
        """
        self.X = np.vstack([self.X, np.asarray(x, dtype=float)[None, :]])
        self.y = np.append(self.y, np.nan if y is None else float(y))
        self.flags.append(dict(flags or {}))
        return len(self) - 1

    def set_score(self, i: int, y: float, **extra) -> None:
        """Set a trial's score and add to its flags.

        Parameters
        ----------
        i : int
            The trial's index.
        y : float or None
            The score, None for NaN.
        extra : dict
            Flags to add.
        """
        self.y[i] = np.nan if y is None else float(y)
        self.flags[i].update(extra)

    def best(self) -> Tuple[int, float]:
        """Return the index and score of the best valid trial, the earliest of a tie.

        Returns
        -------
        i : int
            Its index, -1 without a valid trial.
        score : float
            Its score, -inf without a valid trial.
        """
        v = self.valid
        if not v.any():
            return -1, -np.inf
        yv = np.where(v, self.y, -np.inf)
        i = int(np.argmax(yv))
        return i, float(yv[i])

    def order(self) -> np.ndarray:
        """Return the valid trials' indices, best first, ties in the order of evaluation.

        Returns
        -------
        order : ndarray of int
            The indices.
        """
        idx = np.flatnonzero(self.valid)
        return idx[np.argsort(-self.y[idx], kind="stable")]

    def to_dict(self, n: Optional[int] = None) -> Dict[str, Any]:
        """Return the history as a JSON-ready dict.

        Parameters
        ----------
        n : int or None, default None
            Number of rows, the history as it stood when it held ``n``, since rows are only ever appended.
            None takes them all.

        Returns
        -------
        record : dict
            ``ndim``, ``X``, ``y`` (None for NaN) and ``flags``.
        """
        n = len(self) if n is None else int(n)
        return {"ndim": self.ndim, "X": self.X[:n].tolist(),
                "y": [None if not np.isfinite(v) else float(v) for v in self.y[:n]],
                "flags": self.flags[:n]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "History":
        """Rebuild a history from :meth:`to_dict`.

        Parameters
        ----------
        d : dict
            The record.

        Returns
        -------
        history : History
            The history.
        """
        h = cls(ndim=int(d["ndim"]))
        h.X = np.asarray(d["X"], dtype=float).reshape(-1, h.ndim)
        h.y = np.array([np.nan if v is None else float(v) for v in d["y"]])
        h.flags = [dict(f) for f in d["flags"]]
        return h


@dataclass
class Proposal:
    """A proposed vector and its flags.

    Attributes
    ----------
    x : ndarray, shape (ndim,)
        The vector.
    flags : dict
        Its flags, ``phase`` saying how it was proposed.
    """
    x: np.ndarray
    flags: Dict[str, Any]


# ----------------------------------------------------------------------------
# Parzen helpers (near2m_bw / near2m_kde)
# ----------------------------------------------------------------------------
def parzen_bandwidth(s: np.ndarray, lo: float, hi: float, floor_frac: float = 0.08) -> float:
    """Return the Scott-Silverman bandwidth ``1.06 std n^(-1/5)``, floored at ``floor_frac (hi - lo)``.

    The floor keeps the search from collapsing once the good set becomes tight.

    Parameters
    ----------
    s : ndarray
        The samples.
    lo : float
        Lower bound of the dimension.
    hi : float
        Upper bound of the dimension.
    floor_frac : float, default 0.08
        Floor, as a fraction of the range.

    Returns
    -------
    h : float
        The bandwidth, half the range for fewer than 2 samples.

    Notes
    -----
    It succeeds the IDL predecessor's ``near2m_bw``.
    """
    s = np.asarray(s, dtype=float)
    n = s.size
    if n < 2:
        return 0.5 * (hi - lo)
    h = 1.06 * float(np.std(s, ddof=1)) * n ** (-0.2)
    return max(h, floor_frac * (hi - lo))


def parzen_density(x: float, s: np.ndarray, h: float, lo: float, hi: float,
                   prior_weight: float = 0.25) -> float:
    """Return the 1-D Gaussian Parzen density at a point, mixed with a uniform prior over ``[lo, hi]``.

    Parameters
    ----------
    x : float
        The point.
    s : ndarray
        The samples.
    h : float
        The bandwidth.
    lo : float
        Lower bound.
    hi : float
        Upper bound.
    prior_weight : float, default 0.25
        Weight of the uniform prior.

    Returns
    -------
    density : float
        The density, the uniform one without samples or without range.

    Notes
    -----
    It succeeds the IDL predecessor's ``near2m_kde``.
    """
    s = np.asarray(s, dtype=float)
    n = s.size
    if n < 1 or hi <= lo:
        return 1.0 / max(hi - lo, 1e-12)
    d = float(np.sum(np.exp(-0.5 * ((x - s) / h) ** 2))) / (n * h * np.sqrt(2.0 * np.pi))
    return (1.0 - prior_weight) * d + prior_weight / (hi - lo)


def parzen_density_many(x: np.ndarray, s: np.ndarray, h: float, lo: float, hi: float,
                        prior_weight: float = 0.25) -> np.ndarray:
    """Return :func:`parzen_density` at every point at once, the same to the bit.

    It does the same operations on the same numbers: each row of the kernel matrix is one call's array,
    and a row sum of a C-ordered array is the same pairwise sum as ``np.sum`` of that row.  It makes one
    call per dimension instead of one per candidate.  With 48 candidates, 56 dimensions and both
    densities a TPE proposal made 5,376 calls, each with its own Python overhead and its own copy of
    the good or the bad set, which grows with the history (``tests/test_optimizers.py`` checks the
    equality).

    Parameters
    ----------
    x : ndarray
        The points.
    s : ndarray
        The samples.
    h : float
        The bandwidth.
    lo : float
        Lower bound.
    hi : float
        Upper bound.
    prior_weight : float, default 0.25
        Weight of the uniform prior.

    Returns
    -------
    density : ndarray
        The density at each point.
    """
    s = np.asarray(s, dtype=float)
    x = np.asarray(x, dtype=float)
    n = s.size
    if n < 1 or hi <= lo:
        return np.full(x.shape, 1.0 / max(hi - lo, 1e-12))
    d = np.sum(np.exp(-0.5 * ((x[:, None] - s[None, :]) / h) ** 2), axis=1) / (n * h * np.sqrt(2.0 * np.pi))
    return (1.0 - prior_weight) * d + prior_weight / (hi - lo)


def _categorical_pmf(s: np.ndarray, ncat: int, prior_weight: float) -> np.ndarray:
    """Return a categorical dimension's smoothed probabilities, its frequencies mixed with a uniform prior.

    Parameters
    ----------
    s : ndarray
        The samples, indices rounded and clipped to the choices.
    ncat : int
        Number of choices.
    prior_weight : float
        Weight of the uniform prior.

    Returns
    -------
    pmf : ndarray, shape (ncat,)
        The probabilities.
    """
    counts = np.bincount(np.clip(np.round(s).astype(int), 0, ncat - 1), minlength=ncat).astype(float)
    n = counts.sum()
    emp = counts / n if n > 0 else np.full(ncat, 1.0 / ncat)
    return (1.0 - prior_weight) * emp + prior_weight / ncat


def block_log_density(C: np.ndarray, X: np.ndarray, idx: np.ndarray, h: np.ndarray, dims: Sequence[int],
                      lo: np.ndarray, hi: np.ndarray, prior_weight: float = 0.25) -> np.ndarray:
    """Return the log of the mixture-of-products Parzen density over some dimensions at each candidate.

    The density is::

        p(x) = (1-wp) (1/N) sum_i prod_d K(x_d - X[i,d]; h_d)  +  wp prod_d 1/(hi_d-lo_d)

    Every component of the mixture sits on one real observation, so a candidate is scored against
    configurations that occurred together.  The bandwidths stay diagonal, with no covariance estimated.
    The sums are logsumexp and the uniform prior is mixed in log space, so the result stays finite
    where the kernel sum underflows, as it does for large blocks.

    Parameters
    ----------
    C : ndarray, shape (nc, ndim)
        The ``nc`` candidates.
    X : ndarray, shape (n, ndim)
        The observations.
    idx : ndarray of int
        Indices of the ``N`` observations used, the good or the bad set.
    h : ndarray, shape (ndim,)
        Bandwidth of each dimension.
    dims : sequence of int
        The block's dimensions.
    lo : ndarray, shape (ndim,)
        Lower bounds.
    hi : ndarray, shape (ndim,)
        Upper bounds.
    prior_weight : float, default 0.25
        Weight ``wp`` of the uniform prior.

    Returns
    -------
    log_density : ndarray, shape (nc,)
        The log density at each candidate.

    Notes
    -----
    It succeeds the IDL predecessor's ``near2m_blkden``.
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
    """Turn the ``blocks`` option into a list of index arrays over the dimensions that are not categorical.

    Parameters
    ----------
    space : SearchSpace
        The space.
    blocks : str, sequence or None
        None or ``"univariate"`` for one block per dimension (the factorized TPE), ``"partitions"`` for
        one block per partition and one for the global and selection dimensions, ``"full"`` for a single
        block, or a list of lists of dimension names or indices.

    Returns
    -------
    blocks : list of ndarray of int
        The blocks.  An explicit list leaves each dimension it misses in a block of its own, and
        ``"partitions"`` on a space without partitions is univariate.
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
    """Make one TPE proposal from a history.

    The steps are:

    1. sort by score, the top ``gamma`` fraction the good set (l) and the rest the bad set (g),
    2. fit Parzen estimators per dimension (``blocks=None``, univariate, the IDL predecessor's
       behavior), or a mixture of products within each block of ``blocks`` and independent across
       blocks (block-multivariate), with a smoothed pmf for each categorical dimension,
    3. draw ``ncand`` candidates, each block seeded from one random good observation (or every block
       from the single best with probability ``pbest``) and jittered per dimension by the good set's
       bandwidth,
    4. return the candidate that maximizes the summed log-density ratio ``log l - log g``.

    Block-multivariate scoring makes the proposal robust to noise: it stops it from recombining pieces
    of good configurations that were never good together (``TPE_PYTHON_PORT_ADDENDUM_density.md``).

    Parameters
    ----------
    X : ndarray, shape (n, ndim)
        The valid vectors.
    y : ndarray, shape (n,)
        Their scores.
    space : SearchSpace
        The space.
    rng : numpy.random.Generator
        The generator.
    gamma : float, default 0.25
        Fraction of the history in the good set.
    ncand : int, default 48
        Number of candidates.
    pbest : float, default 0.0
        Probability that a candidate is seeded from the single best observation in every block.
    prior_weight : float, default 0.25
        Weight of the uniform prior in each density.
    bw_floor : float, default 0.08
        Floor of the bandwidths, as a fraction of the range.
    blocks : str, sequence or None, default None
        The density model (:func:`resolve_blocks`).

    Returns
    -------
    x : ndarray, shape (ndim,)
        The proposal, sanitized, a random vector for fewer than 2 rows.
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
            ld = parzen_density_many(C[:, d], X[good, d], hg[d], lo[d], hi[d], prior_weight)
            gd = parzen_density_many(C[:, d], X[bad, d], hb[d], lo[d], hi[d], prior_weight)
            ei += np.log(ld) - np.log(gd)
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
    """Base class of the search strategies.

    ``ask(history, rng, n)`` returns ``n`` proposals computed from the same posterior, a batch, and
    ``tell`` does nothing, since the history is the one source of truth, which is what makes a resume
    trivial.

    Parameters
    ----------
    space : SearchSpace
        The space.
    n_init : int, default 40
        Number of history rows, any seed included, before guided proposals start.
    warmstart_tie : bool, default False
        Whether to tie every partition slot of each base in the warm-up draws, a global warm start.
    link_params : sequence of str, default ()
        Bases whose partition slots are tied on every proposal.

    Attributes
    ----------
    name : str
        Name of the strategy.
    """

    name = "base"

    def __init__(self, space: SearchSpace, n_init: int = 40, warmstart_tie: bool = False,
                 link_params: Sequence[str] = ()):
        self.space = space
        self.n_init = int(n_init)
        self.warmstart_tie = bool(warmstart_tie)
        self.link_params = [b for b in link_params if b]

    def _finish(self, x: np.ndarray) -> np.ndarray:
        """Sanitize a proposal and tie its linked bases.

        Parameters
        ----------
        x : ndarray, shape (ndim,)
            The proposal.

        Returns
        -------
        x : ndarray, shape (ndim,)
            The proposal finished.
        """
        x = self.space.sanitize(x)
        if self.link_params:
            x = self.space.tie(x, self.link_params)
        return x

    def ask(self, history: History, rng: np.random.Generator, n: int = 1) -> List[Proposal]:
        """Return proposals from a history.

        Parameters
        ----------
        history : History
            The history.
        rng : numpy.random.Generator
            The generator.
        n : int, default 1
            Number of proposals.

        Returns
        -------
        proposals : list of Proposal
            The proposals.

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError

    def tell(self, history: History, i: int, y: float) -> None:  # pragma: no cover
        """Do nothing, since the history is the one source of truth.

        Parameters
        ----------
        history : History
            The history.
        i : int
            Index of the trial scored.
        y : float
            Its score.
        """
        pass

    def describe(self) -> Dict[str, Any]:
        """Return the strategy's settings, for a run's record.

        Returns
        -------
        info : dict
            ``name``, ``n_init``, ``warmstart_tie`` and ``link_params``.
        """
        return {"name": self.name, "n_init": self.n_init, "warmstart_tie": self.warmstart_tie,
                "link_params": list(self.link_params)}


class RandomSearch(Optimizer):
    """Uniform random search, a baseline of matched budget, with the parameters of :class:`Optimizer`.
    """
    name = "random"

    def ask(self, history: History, rng: np.random.Generator, n: int = 1) -> List[Proposal]:
        """Return uniform random proposals.

        Parameters
        ----------
        history : History
            The history, unused.
        rng : numpy.random.Generator
            The generator.
        n : int, default 1
            Number of proposals.

        Returns
        -------
        proposals : list of Proposal
            The proposals, of phase "random".
        """
        return [Proposal(self._finish(self.space.random(rng, tie_all=self.warmstart_tie)),
                         {"phase": "random"}) for _ in range(n)]


class GridSearch(Optimizer):
    """Regular grid over ``axes``, cell-centered and refined in place when the cells run out.

    The dimensions outside ``axes`` sit at the base vector.  ``gpts = floor(budget**(1/naxes))`` points
    per axis make the first pass fit inside the budget, ``floor`` rather than ``round`` since rounding
    up overruns it.  Few points per axis is the honest regime of grid search in many dimensions, the
    ``budget**(1/d)`` argument of Bergstra & Bengio (2012) for why grid search loses to random search
    as ``d`` grows, so a small ``gpts`` is reported, not corrected.

    Parameters
    ----------
    space : SearchSpace
        The space.
    budget : int
        Number of evaluations.
    axes : sequence of str or None, default None
        The dimensions gridded.  None takes every reduction dimension, those with an explicit ``grid``
        first.
    base_vector : ndarray or None, default None
        Vector of the dimensions outside ``axes``.  None takes the default vector.
    gpts : int or None, default None
        Points per axis of the first stage, at least 2.  None takes the most whose grid fits the budget.
    kw : dict
        ``warmstart_tie`` and ``link_params``, as :class:`Optimizer` takes them.

    Attributes
    ----------
    axes : list of str
        The dimensions gridded, in order.
    budget : int
        The budget.
    gpts : int
        Points per axis of the first stage.
    base_vector : ndarray
        Vector of the other dimensions.

    Notes
    -----
    Two things about the earlier implementation were defects rather than that regime.

    Cells were endpoints, ``lo + span*i/(g-1)``, which puts both points of a two-point axis on the
    bounds of the box, where the reduction degenerates: ``corr_thresh = 1`` keeps no frames, the largest
    ``filter`` smooths the signal away, ``k_klip = 1`` subtracts almost nothing.  A nine-axis grid at
    ``gpts = 2`` spent its whole budget on the 512 corners of the box and never evaluated one interior
    configuration.  Cells are now centered, ``lo + span*(i + 0.5)/g``, the usual convention for a grid
    design, which never lands on a degenerate face.

    Cell ``k`` used to be ``k % g**naxes``, so once the cells ran out the grid proposed configurations
    it had already evaluated.  The objective is deterministic given the configuration, so those
    evaluations bought nothing: a 1000-evaluation run over 512 cells was charged 1000 reductions for
    449 distinct configurations.  Exhausting a stage now doubles ``gpts``, and since the cells are
    centered, every node of stage ``s+1`` is new.

    The axes with an explicit ``grid`` go first because the cell index is an odometer: axis 0 cycles
    through all its values every ``g`` cells, while the last moves only once the budget has covered
    ``g**(naxes-1)`` cells, which a real budget never does.  Declared last, ``k_klip`` took three values
    in a thousand evaluations while ``bin`` took six.
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
        """Return the stage of a grid proposal and its cell within the stage.

        Stage ``s`` has ``gpts * 2**s`` points per axis and ``(gpts * 2**s)**naxes`` cells.  The stages are
        walked in order, so a budget larger than one stage keeps producing new configurations instead of
        repeating the first stage.

        Parameters
        ----------
        k : int
            Index of the grid proposal.

        Returns
        -------
        gpts : int
            Points per axis of the stage.
        cell : int
            The cell within the stage.
        stage : int
            The stage, at most 24.
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
        """Return the vector of a grid proposal, each axis at its cell's center.

        An axis with an explicit ``grid`` takes the entry at the cell's fraction of it.

        Parameters
        ----------
        k : int
            Index of the grid proposal.

        Returns
        -------
        x : ndarray, shape (ndim,)
            The vector, the other dimensions at the base vector.
        """
        x = self.base_vector.copy()
        g, kk, _ = self.stage_of(k)
        for a, name in enumerate(self.axes):
            i = (kk // (g ** a)) % g
            p = self.space[name]
            frac = (i + 0.5) / g                 # cell center, never a box face
            if p.grid is not None:
                x[self.space.index(name)] = p.grid[min(int(frac * len(p.grid)), len(p.grid) - 1)]
            else:
                x[self.space.index(name)] = p.lo + p.span * frac
        return x

    def ask(self, history: History, rng: np.random.Generator, n: int = 1) -> List[Proposal]:
        """Return the next grid proposals, after those the history holds.

        Parameters
        ----------
        history : History
            The history, whose trials of phase "grid" are counted.
        rng : numpy.random.Generator
            The generator, unused.
        n : int, default 1
            Number of proposals.

        Returns
        -------
        proposals : list of Proposal
            The proposals, their flags giving ``cell``, ``gpts`` and ``stage``.
        """
        ngrid = sum(1 for f in history.flags if f.get("phase") == "grid")
        out = []
        for k in range(n):
            g, cell, stage = self.stage_of(ngrid + k)
            out.append(Proposal(self._finish(self.cell(ngrid + k)),
                                {"phase": "grid", "cell": ngrid + k, "gpts": g, "stage": stage}))
        return out

    def describe(self) -> Dict[str, Any]:
        """Return the grid's settings, for a run's record.

        Returns
        -------
        info : dict
            The base settings, ``axes``, ``gpts``, ``budget``, ``cell_centred``, ``n_cells_stage0``,
            ``stages`` and ``coarse_by_budget``, True when the budget cannot give three levels per axis.
        """
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

    Each proposal is a warm-up draw while the history holds fewer than ``n_init`` rows, else a local
    move with probability ``p_local``, else a uniform draw with probability ``explore_frac``, else a
    TPE proposal (:func:`tpe_propose`).

    Parameters
    ----------
    space : SearchSpace
        The space.
    n_init : int, default 40
        Number of history rows, any seed included, before guided proposals start.
    gamma : float, default 0.25
        Fraction of the history in the good set.
    ncand : int, default 48
        Number of candidates per proposal.
    prior_weight : float, default 0.25
        Weight of the uniform prior in each density.
    bw_floor : float, default 0.08
        Floor of the bandwidths, as a fraction of the range.
    pbest : float, default 0.0
        Fraction of candidates seeded from the single best observation in every dimension.  On a noisy
        objective the running best is usually a noise winner, and anchoring to it imports the selection
        bias into the proposal, so it is kept for parity experiments only.
    p_local : float, default 0.15
        Probability of a local move, which jitters one reduction dimension of a random configuration
        among the top ``n_elite``, in place of a TPE proposal.
    n_elite : int, default 5
        Number of top configurations a local move starts from.
    explore_frac : float, default 0.15
        Probability, after the local move's, of a uniform random draw.
    blocks : str, sequence or None, default "univariate"
        The density model (:func:`resolve_blocks`): ``"univariate"`` or None for factorized
        per-dimension estimators, ``"partitions"`` for block-multivariate ones, ``"full"`` for a single
        block, kept as a negative control, or a list of lists of dimension names.
    warmstart_tie : bool, default False
        Whether to tie every partition slot of each base in the warm-up draws, a global warm start.
    link_params : sequence of str, default ()
        Bases whose partition slots are tied on every proposal.

    Notes
    -----
    Anchoring to the best cost 1.1 in true score at noise 0.40 on the synthetic testbed and won none of
    24 seeds (addendum 2).  The univariate model is the IDL predecessor's behavior, and the best
    configuration in the 2x2x3 factorial of addendum 2 once ``pbest = 0``.
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
        """Jitter one random reduction dimension of a random configuration among the top ``n_elite``.

        A continuous dimension moves by the good set's bandwidth times a standard normal draw, and a
        categorical one takes another choice.

        Parameters
        ----------
        history : History
            The history.
        rng : numpy.random.Generator
            The generator.

        Returns
        -------
        x : ndarray, shape (ndim,)
            The moved vector, a random one without a valid trial.
        """
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
        """Make one proposal: a warm-up draw, a local move, an exploring draw or a TPE proposal.

        An exploring draw also stands in for a TPE proposal while fewer than 2 trials are valid.

        Parameters
        ----------
        history : History
            The history.
        rng : numpy.random.Generator
            The generator.

        Returns
        -------
        proposal : Proposal
            The proposal, its ``phase`` ``warmup``, ``local``, ``explore`` or ``tpe``.
        """
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
        """Return a batch of proposals from the same posterior.

        The warm-up count advances with the batch, so a batch never overruns ``n_init``.

        Parameters
        ----------
        history : History
            The history.
        rng : numpy.random.Generator
            The generator.
        n : int, default 1
            Number of proposals.

        Returns
        -------
        proposals : list of Proposal
            The proposals.
        """
        out = []
        fake = History(history.ndim, history.X.copy(), history.y.copy(), list(history.flags))
        for _ in range(n):
            pr = self.propose_one(fake, rng)
            out.append(pr)
            if len(fake) < self.n_init:
                fake.append(pr.x, np.nan, pr.flags)
        return out

    def describe(self) -> Dict[str, Any]:
        """Return the strategy's settings, with its blocks resolved, for a run's record.

        Returns
        -------
        info : dict
            The base settings, the TPE's parameters, ``blocks`` and ``resolved_blocks``, each block's
            dimension names.
        """
        d = super().describe()
        d.update(gamma=self.gamma, ncand=self.ncand, prior_weight=self.prior_weight,
                 bw_floor=self.bw_floor, pbest=self.pbest, p_local=self.p_local,
                 n_elite=self.n_elite, explore_frac=self.explore_frac,
                 blocks=self.blocks if isinstance(self.blocks, (str, type(None))) else [list(map(str, b)) for b in self.blocks],
                 resolved_blocks=[[self.space.params[i].name for i in b] for b in resolve_blocks(self.space, self.blocks)])
        return d


def make_optimizer(mode: str, space: SearchSpace, **kw) -> Optimizer:
    """Make the strategy of a search mode.

    Parameters
    ----------
    mode : {"tpe", "random", "grid"}
        The mode, in any case.
    space : SearchSpace
        The space.
    kw : dict
        The strategy's keywords, ``budget`` required for a grid.

    Returns
    -------
    optimizer : Optimizer
        The strategy.

    Raises
    ------
    ValueError
        If the mode is unknown.
    """
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
