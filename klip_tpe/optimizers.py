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

__all__ = ["History", "Proposal", "Optimizer", "TPE", "RandomSearch", "GridSearch", "DENSITIES",
           "parzen_bandwidth", "parzen_density", "parzen_density_many", "parzen_density_binned",
           "block_log_density", "resolve_blocks", "tpe_propose"]


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

    def _reserve(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Make room for ``n`` rows in the arrays behind ``X`` and ``y``.

        ``X`` and ``y`` are views of the first rows of two arrays with room to spare, which double when
        full, so an append copies one row instead of the whole history.  Arrays set from outside (a
        history built from given arrays, :meth:`from_dict`) are copied in at the first append.

        Parameters
        ----------
        n : int
            Number of rows wanted.

        Returns
        -------
        xbuf : ndarray, shape (cap, ndim)
            The array behind ``X``, ``cap`` at least ``n``.
        ybuf : ndarray, shape (cap,)
            The array behind ``y``.
        """
        st = self.__dict__
        xbuf, ybuf = st.get("_xbuf"), st.get("_ybuf")
        mine = xbuf is not None and self.X is st.get("_xview") and self.y is st.get("_yview")
        if not mine or xbuf.shape[0] < n:
            m = len(self)
            cap = max(64, 2 * n) if mine or m else max(16, n)
            nx, ny = np.empty((cap, self.ndim)), np.empty(cap)
            nx[:m], ny[:m] = self.X, self.y
            self._xbuf, self._ybuf = xbuf, ybuf = nx, ny
            self.X, self.y = nx[:m], ny[:m]
            self._xview, self._yview = self.X, self.y
        return xbuf, ybuf

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

        Notes
        -----
        Until 2026-11-17 each append stacked a new ``X``, a copy of the whole history per trial (4.5 MB at
        10,000 trials of a 56-D space).  ``X`` and ``y`` are now views of arrays with room to spare
        (:meth:`_reserve`).  The rows a view holds never change, so an ``X`` or ``y`` read earlier keeps
        the trials it had.
        """
        n = len(self)
        x = np.asarray(x, dtype=float).reshape(self.ndim)
        xbuf, ybuf = self._reserve(n + 1)
        xbuf[n] = x
        ybuf[n] = np.nan if y is None else float(y)
        self.X, self.y = xbuf[:n + 1], ybuf[:n + 1]
        self._xview, self._yview = self.X, self.y
        self.flags.append(dict(flags or {}))
        return n

    def with_rows(self, xs: Sequence[np.ndarray], flags: Sequence[Dict[str, Any]]) -> "History":
        """Return this history followed by rows without a score, sharing its arrays.

        The rows go into the room behind this history's arrays, so nothing of the history is copied but the
        list of flags.  The history returned is good until this one's next append, which writes over its
        first extra row.

        Parameters
        ----------
        xs : sequence of ndarray, each shape (ndim,)
            The extra rows' vectors.
        flags : sequence of dict
            Their flags.

        Returns
        -------
        history : History
            The trials, then the extra rows with NaN scores.
        """
        n, p = len(self), len(xs)
        if p == 0:
            return self
        xbuf, ybuf = self._reserve(n + p)
        xbuf[n:n + p] = np.asarray(xs, dtype=float).reshape(p, self.ndim)
        ybuf[n:n + p] = np.nan
        return History(self.ndim, xbuf[:n + p], ybuf[:n + p], self.flags + [dict(f or {}) for f in flags])

    def __getstate__(self) -> Dict[str, Any]:
        """Return the state to pickle, without the spare room behind ``X`` and ``y``.

        Returns
        -------
        state : dict
            ``ndim``, ``X``, ``y`` and ``flags``, the arrays copied to their rows.
        """
        return {"ndim": self.ndim, "X": np.array(self.X), "y": np.array(self.y), "flags": self.flags}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore a pickled state.

        Parameters
        ----------
        state : dict
            The state from :meth:`__getstate__`.
        """
        self.__dict__.update(state)

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
    # Each step in place, a few points at a time (2026-11-17): the same operations on the same
    # numbers as -0.5 * ((x - s) / h) ** 2 with a new array per step, in a block that stays in the
    # cache, and each point's row summed on its own as before
    rows = max(1, min(x.size, _DENSITY_BLOCK // n))
    t = np.empty((rows, n))
    sums = np.empty(x.shape)
    for i in range(0, x.size, rows):
        tb = t[:min(rows, x.size - i)]
        np.subtract.outer(x[i:i + rows], s, out=tb)
        tb /= h
        np.square(tb, out=tb)
        tb *= -0.5
        np.exp(tb, out=tb)
        sums[i:i + rows] = np.sum(tb, axis=1)
    d = sums / (n * h * np.sqrt(2.0 * np.pi))
    return (1.0 - prior_weight) * d + prior_weight / (hi - lo)


#: elements of the kernel block :func:`parzen_density_many` works on at a time, 1 MB
_DENSITY_BLOCK = 1 << 17


#: bins across a dimension's range for :func:`parzen_density_binned`, at least, and per bandwidth
BINS_PER_RANGE = 512
BINS_PER_BANDWIDTH = 16


def parzen_density_binned(x: np.ndarray, s: np.ndarray, h: float, lo: float, hi: float,
                          prior_weight: float = 0.25, integral: bool = False) -> np.ndarray:
    """Return :func:`parzen_density_many` with the samples gathered onto a grid.

    The kernel sum runs over the grid's occupied points, weighted by the samples they hold, so its cost
    stops growing once every point is occupied.  A dimension of whole numbers whose samples are all
    whole puts each value on a point of its own, and the sum is then exact but for its order.  Any other
    dimension shares each sample between its two neighboring points (linear binning), with
    :data:`BINS_PER_RANGE` points across the range or :data:`BINS_PER_BANDWIDTH` per bandwidth,
    whichever spacing is finer.

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
    integral : bool, default False
        Whether the dimension takes whole numbers only.

    Returns
    -------
    density : ndarray
        The density at each point.

    Notes
    -----
    With 16 points per bandwidth the density is within 1e-4 of the exact sum, 1e-5 typically, and on a
    dimension of whole numbers it is the exact sum.  ``tests/test_optimizers.py`` measures it.
    """
    s = np.asarray(s, dtype=float)
    x = np.asarray(x, dtype=float)
    n = s.size
    if n < 1 or hi <= lo:
        return np.full(x.shape, 1.0 / max(hi - lo, 1e-12))
    if not h > 0:                                  # no grid for a bandwidth of 0 (a floor of 0)
        return parzen_density_many(x, s, h, lo, hi, prior_weight)
    pts = w = None
    if integral:
        v = np.rint(s)
        if np.array_equal(v, s):
            iv = v.astype(np.int64)
            base = int(iv.min())
            if int(iv.max()) - base <= 1 << 16:
                cnt = np.bincount(iv - base)
                nz = np.flatnonzero(cnt)
                pts, w = (nz + base).astype(float), cnt[nz].astype(float)
    if pts is None:
        step = min((hi - lo) / BINS_PER_RANGE, h / BINS_PER_BANDWIDTH)
        nb = int(np.ceil((hi - lo) / step)) + 1
        u = np.clip((s - lo) / step, 0.0, nb - 1.0)
        k = np.minimum(u.astype(np.intp), nb - 2)
        f = u - k
        cnt = np.bincount(k, weights=1.0 - f, minlength=nb) + np.bincount(k + 1, weights=f, minlength=nb)
        nz = np.flatnonzero(cnt)
        pts, w = lo + nz * step, cnt[nz]
    t = np.subtract.outer(x, pts)
    t /= h
    np.square(t, out=t)
    t *= -0.5
    np.exp(t, out=t)
    d = (t @ w) / (n * h * np.sqrt(2.0 * np.pi))
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


def _columns(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Return the rows ``idx`` of ``X`` transposed, one row per dimension.

    Row ``d`` holds the same numbers in the same order as ``X[idx, d]``.  Each row starts 16 bytes after
    the one before, as a new array from the allocator does, so a reduction over it is the same one.

    Parameters
    ----------
    X : ndarray, shape (n, ndim)
        The vectors.
    idx : ndarray of int, shape (m,)
        The rows.

    Returns
    -------
    columns : ndarray, shape (ndim, m)
        The columns.
    """
    m = len(idx)
    out = np.empty((X.shape[1], m + (m & 1)))
    out[:, :m] = X[idx].T
    return out[:, :m]


#: the TPE's density evaluations, :func:`parzen_density_many` ("exact") or
#: :func:`parzen_density_binned` ("binned")
DENSITIES = ("exact", "binned")


def tpe_propose(X: np.ndarray, y: np.ndarray, space: SearchSpace, rng: np.random.Generator,
                gamma: float = 0.25, ncand: int = 48, pbest: float = 0.0,
                prior_weight: float = 0.25, bw_floor: float = 0.08,
                blocks=None, rows: Optional[np.ndarray] = None, density: str = "exact") -> np.ndarray:
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
        The valid vectors, or every vector of a history with ``rows``.
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
    rows : ndarray of int or None, default None
        The rows of ``X`` and ``y`` to use, in order, the valid ones of a history.  None uses them all.
    density : {"exact", "binned"}, default "exact"
        How the univariate densities are summed, over every sample (:func:`parzen_density_many`) or over
        a grid (:func:`parzen_density_binned`).  Blocks of several dimensions are summed exactly.

    Returns
    -------
    x : ndarray, shape (ndim,)
        The proposal, sanitized, a random vector for fewer than 2 rows.

    Raises
    ------
    ValueError
        If ``density`` is not one of :data:`DENSITIES`.

    Notes
    -----
    Since 2026-11-17 the good and the bad set are gathered once, a row per dimension, and the
    candidates are drawn with Python numbers.  The proposal and the generator's state after it are
    the same bit for bit as before (``tests/test_optimizers.py``).
    """
    if density not in DENSITIES:
        raise ValueError(f"density {density!r}: one of {DENSITIES}")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if rows is not None:
        rows = np.asarray(rows, dtype=np.intp)
        y = y[rows]
    n, ndim = y.shape[0], X.shape[1]
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
    if rows is not None:
        good, bad = rows[good], rows[bad]
    Xg = X[good]
    G, B = _columns(X, good), _columns(X, bad)

    hg = np.array([parzen_bandwidth(G[d], lo[d], hi[d], bw_floor) for d in range(ndim)])
    hb = np.array([parzen_bandwidth(B[d], lo[d], hi[d], bw_floor) for d in range(ndim)])
    cat_dims = [d for d in range(ndim) if cat[d]]
    pmf_g = {d: _categorical_pmf(G[d], ncat[d], prior_weight) for d in cat_dims}
    pmf_b = {d: _categorical_pmf(B[d], ncat[d], prior_weight) for d in cat_dims}
    blk = resolve_blocks(space, blocks)

    # Python numbers in the loop: the same IEEE arithmetic as numpy's scalars, at a fraction of
    # the cost, and the generator called in the same order with the same arguments
    blk_l = [[int(d) for d in b] for b in blk]
    hgl, lol, hil = hg.tolist(), lo.tolist(), hi.tolist()
    item = Xg.item
    rand, normal = rng.random, rng.standard_normal
    C = np.empty((ncand, ndim))
    for c in range(ncand):
        anchor = 0 if rand() < pbest else -1
        # one anchor observation per block (univariate: per dimension), or the running
        # best for every block with probability pbest
        for b in blk_l:
            j = anchor if anchor >= 0 else int(rand() * ng)
            for d in b:
                v = item(j, d) + hgl[d] * normal()
                C[c, d] = min(max(v, lol[d]), hil[d])
        for d in cat_dims:
            C[c, d] = float(rng.choice(ncat[d], p=pmf_g[d]))

    integral = space.is_int if density == "binned" else None
    ei = np.zeros(ncand)
    for b in blk:
        if len(b) == 1:
            d = int(b[0])
            if integral is None:
                ld = parzen_density_many(C[:, d], G[d], hg[d], lo[d], hi[d], prior_weight)
                gd = parzen_density_many(C[:, d], B[d], hb[d], lo[d], hi[d], prior_weight)
            else:
                ld = parzen_density_binned(C[:, d], G[d], hg[d], lo[d], hi[d], prior_weight, bool(integral[d]))
                gd = parzen_density_binned(C[:, d], B[d], hb[d], lo[d], hi[d], prior_weight, bool(integral[d]))
            ei += np.log(ld) - np.log(gd)
        else:
            ei += block_log_density(C, X, good, hg, b, lo, hi, prior_weight) \
                - block_log_density(C, X, bad, hb, b, lo, hi, prior_weight)
    for d in cat_dims:
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
    density : {"exact", "binned"}, default "exact"
        How a TPE proposal sums its univariate densities (:func:`tpe_propose`).

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
                 blocks="univariate", warmstart_tie: bool = False, link_params: Sequence[str] = (),
                 density: str = "exact"):
        super().__init__(space, n_init=n_init, warmstart_tie=warmstart_tie, link_params=link_params)
        if density not in DENSITIES:
            raise ValueError(f"density {density!r}: one of {DENSITIES}")
        self.blocks = blocks
        self.density = str(density)
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
        rows = np.flatnonzero(history.valid)
        if rows.size < 2:
            return Proposal(self._finish(sp.random(rng)), {"phase": "explore"})
        x = tpe_propose(history.X, history.y, sp, rng, gamma=self.gamma, ncand=self.ncand,
                        pbest=self.pbest, prior_weight=self.prior_weight, bw_floor=self.bw_floor,
                        blocks=self.blocks, rows=rows, density=self.density)
        return Proposal(self._finish(x), {"phase": "tpe"})

    def ask(self, history: History, rng: np.random.Generator, n: int = 1) -> List[Proposal]:
        """Return a batch of proposals from the same posterior.

        The warm-up count advances with the batch, so a batch never overruns ``n_init``.  The warm-up
        proposals enter a copy of the history, made only when there are any.

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
        fake = history
        for _ in range(n):
            pr = self.propose_one(fake, rng)
            out.append(pr)
            if len(fake) < self.n_init:
                if fake is history:
                    fake = History(history.ndim, history.X.copy(), history.y.copy(), list(history.flags))
                fake.append(pr.x, np.nan, pr.flags)
        return out

    def describe(self) -> Dict[str, Any]:
        """Return the strategy's settings, with its blocks resolved, for a run's record.

        Returns
        -------
        info : dict
            The base settings, the TPE's parameters, ``density``, ``blocks`` and ``resolved_blocks``, each
            block's dimension names.
        """
        d = super().describe()
        d.update(gamma=self.gamma, ncand=self.ncand, prior_weight=self.prior_weight,
                 bw_floor=self.bw_floor, pbest=self.pbest, p_local=self.p_local,
                 n_elite=self.n_elite, explore_frac=self.explore_frac, density=self.density,
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
