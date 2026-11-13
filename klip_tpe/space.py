"""Search spaces: parameter kinds, per-partition blocks, partition selection and the feasibility projection.

Every parameter vector ``x`` the optimizer handles is a 1-D float array of length ``space.ndim``.
``SearchSpace.decode(x)`` turns it into a :class:`Config` of plain dictionaries that a
:class:`~klip_tpe.reducer.Reducer` consumes.  Nothing in this module knows about telescopes.

Notes
-----
It succeeds these pieces of the IDL predecessor's ``optimize_near_2_tpe.pro``:

* the night-major blocks ``bin_n1 ... k_klip_n6``, by :meth:`SearchSpace.replicate`,
* ``near2m_pntie``, by :meth:`SearchSpace.tie`,
* ``near2m_pndecode`` and ``near2m_nightsel``, by :meth:`SearchSpace.decode`,
* ``near2m_kgrid`` and ``near2m_ksnap``, by :class:`Param`'s ``grid``,
* the two-slot jackknife ``drop1`` and ``drop2``, by ``selection="two_slot"``,
* the older binary flags ``n1 ... n6``, by ``selection="binary"``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = ["Param", "SearchSpace", "Config", "kgrid"]


def kgrid(kmax: int) -> np.ndarray:
    """Return the grid of allowed KL-mode counts, in steps of 1 to 20, of 5 to 50 and of 10 above.

    Drawing uniformly over the entries puts about 2/3 of random draws at k <= 20, where good solutions
    live.

    Parameters
    ----------
    kmax : int
        Largest count, at least 1.

    Returns
    -------
    grid : ndarray
        The counts up to ``kmax``.

    Notes
    -----
    It succeeds the IDL predecessor's ``near2m_kgrid``.
    """
    km = max(int(round(kmax)), 1)
    g = list(range(1, min(20, km) + 1))
    if km > 20:
        g += list(range(25, min(50, km) + 1, 5))
    if km > 50:
        g += list(range(60, km + 1, 10))
    return np.array([v for v in g if v <= km], dtype=float)


@dataclass
class Param:
    """One searched dimension.

    Attributes
    ----------
    name : str
        Name of the dimension.
    lo : float
        Lower bound, 0 for a categorical.
    hi : float
        Upper bound, the last index for a categorical.
    kind : {"float", "int", "categorical"}, default "float"
        Kind of the values.  A categorical is encoded as an index in ``[0, len(choices) - 1]``, which the
        TPE treats with a categorical Parzen estimator, without a spurious order.
    grid : sequence of float or None, default None
        Allowed values, kept sorted within the bounds.  Random draws pick an entry uniformly, and guided
        proposals snap to the nearest.
    choices : sequence or None, default None
        The labels of a categorical.
    default : object, default None
        Value of the seeded default configuration and of the slots not searched.  None takes the first
        choice, or the middle of the bounds, rounded for an int.
    base : str or None, default None
        The base parameter a replicated dimension stands for, set by :meth:`SearchSpace.replicate`.
        None takes ``name``.
    partition : object, default None
        The partition of a replicated dimension, None for a global one.
    role : {"reduction", "selection"}, default "reduction"
        Whether the value goes to the reducer, or encodes which partitions are included and never goes
        to it.
    doc : str, default ""
        Description of the dimension.

    Raises
    ------
    ValueError
        If the kind is unknown, a categorical has no choices, ``hi < lo``, or the grid has no entry
        within the bounds.
    """

    name: str
    lo: float
    hi: float
    kind: str = "float"
    grid: Optional[Sequence[float]] = None
    choices: Optional[Sequence[Any]] = None
    default: Any = None
    base: Optional[str] = None
    partition: Optional[Any] = None
    role: str = "reduction"
    doc: str = ""

    def __post_init__(self):
        """Check the dimension and settle its bounds, grid, base and default.

        A categorical's bounds become its indices, the grid is sorted and cut to the bounds, and a missing
        base or default is filled in.

        Raises
        ------
        ValueError
            If the kind is unknown, a categorical has no choices, ``hi < lo``, or the grid has no entry
            within the bounds.
        """
        if self.kind not in ("float", "int", "categorical"):
            raise ValueError(f"bad kind {self.kind!r} for {self.name}")
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"categorical {self.name} needs choices")
            self.lo, self.hi = 0.0, float(len(self.choices) - 1)
        self.lo = float(self.lo)
        self.hi = float(self.hi)
        if self.hi < self.lo:
            raise ValueError(f"{self.name}: hi < lo")
        if self.grid is not None:
            g = np.asarray(sorted(set(float(v) for v in self.grid)))
            g = g[(g >= self.lo) & (g <= self.hi)]
            if g.size == 0:
                raise ValueError(f"{self.name}: grid has no entries inside bounds")
            self.grid = g
        if self.base is None:
            self.base = self.name
        if self.default is None:
            if self.kind == "categorical":
                self.default = self.choices[0]
            else:
                self.default = 0.5 * (self.lo + self.hi)
                if self.kind == "int":
                    self.default = float(round(self.default))

    # -- encoding helpers --------------------------------------------------
    @property
    def is_int(self) -> bool:
        """Whether the dimension takes whole values, as an int or a categorical does.
        """
        return self.kind in ("int", "categorical")

    @property
    def span(self) -> float:
        """Width of the bounds, ``hi - lo``.
        """
        return self.hi - self.lo

    def encode(self, value: Any) -> float:
        """Return a value's coordinate in ``[lo, hi]``.

        Parameters
        ----------
        value : object
            The value, a label or an index for a categorical.

        Returns
        -------
        coord : float
            The coordinate, clipped to the bounds.
        """
        if self.kind == "categorical":
            if value in self.choices:
                return float(list(self.choices).index(value))
            return float(np.clip(round(float(value)), 0, len(self.choices) - 1))
        return float(np.clip(float(value), self.lo, self.hi))

    def decode(self, coord: float) -> Any:
        """Return the value at a coordinate, a label for a categorical and an int for an int.

        Parameters
        ----------
        coord : float
            The coordinate.

        Returns
        -------
        value : object
            The value, after :meth:`sanitize`.
        """
        c = self.sanitize(coord)
        if self.kind == "categorical":
            return self.choices[int(round(c))]
        if self.kind == "int":
            return int(round(c))
        return float(c)

    def sanitize(self, coord: float) -> float:
        """Clip a coordinate to the bounds, then snap it to the grid or round it to a whole value.

        Parameters
        ----------
        coord : float
            The coordinate.

        Returns
        -------
        coord : float
            The coordinate sanitized.
        """
        c = float(np.clip(coord, self.lo, self.hi))
        if self.grid is not None:
            g = self.grid
            c = float(g[int(np.argmin(np.abs(g - c)))])
        elif self.is_int:
            c = float(round(c))
        return c

    def random(self, rng: np.random.Generator) -> float:
        """Draw a coordinate uniformly, over the grid's entries where there is a grid.

        Parameters
        ----------
        rng : numpy.random.Generator
            The generator.

        Returns
        -------
        coord : float
            The coordinate, sanitized.
        """
        if self.grid is not None:
            return float(self.grid[rng.integers(len(self.grid))])
        v = self.lo + rng.random() * self.span
        return self.sanitize(v)


@dataclass
class Config:
    """A decoded parameter vector.

    Attributes
    ----------
    params : dict
        Global reduction parameters, ``{base: value}``, those not searched at their defaults, which a
        single-partition reducer consumes.
    per_partition : dict
        ``{partition_id: {base: value}}`` for every partition of the space, the partition's own slots
        over the global values.
    selected : list
        The partitions switched on, all of them when the space has no selection scheme, never empty.
    x : ndarray
        The sanitized vector it was decoded from.
    """

    params: Dict[str, Any]
    per_partition: Dict[Any, Dict[str, Any]]
    selected: List[Any]
    x: np.ndarray

    def params_for(self, partition) -> Dict[str, Any]:
        """Return a partition's parameters, the global ones for a partition the space does not know.

        Parameters
        ----------
        partition : object
            The partition's id.

        Returns
        -------
        params : dict
            The parameters.
        """
        return self.per_partition.get(partition, dict(self.params))

    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as JSON-ready dicts, the partition ids as strings.

        Returns
        -------
        record : dict
            ``params``, ``per_partition``, ``selected`` and ``x``.
        """
        return {
            "params": dict(self.params),
            "per_partition": {str(k): dict(v) for k, v in self.per_partition.items()},
            "selected": [str(s) for s in self.selected],
            "x": [float(v) for v in self.x],
        }


class SearchSpace:
    """Ordered collection of :class:`Param` with partition support.

    Parameters
    ----------
    params : iterable of Param, default ()
        The global dimensions, not replicated.
    fixed : dict or None, default None
        Reduction parameters not searched but still handed to the reducer, such as ``spat_mean=0``.
    project : callable or None, default None
        Feasibility projection ``f(x, space, **ctx) -> x'``, which the runner applies on every proposal
        path.  It should be a pure function.

    Attributes
    ----------
    partitions : list
        The partitions, set by :meth:`replicate` or :meth:`with_selection`.
    selection : {"two_slot", "binary"} or None
        The selection scheme.
    selection_max_drop : int
        Number of two-slot drop slots.
    p_include : float
        Probability that a random draw keeps the partitions of a selection slot or flag.
    """

    def __init__(self, params: Iterable[Param] = (), fixed: Optional[Dict[str, Any]] = None,
                 project: Optional[Callable[..., np.ndarray]] = None):
        self.params: List[Param] = []
        self.fixed: Dict[str, Any] = dict(fixed or {})
        self.partitions: List[Any] = []
        self.selection: Optional[str] = None
        self.selection_max_drop: int = 2
        self.p_include: float = 0.75
        self.project = project
        for p in params:
            self.add(p)

    # -- construction --------------------------------------------------------
    def add(self, p: Param) -> "SearchSpace":
        """Add a dimension.

        Parameters
        ----------
        p : Param
            The dimension.

        Returns
        -------
        space : SearchSpace
            The space itself.

        Raises
        ------
        ValueError
            If a dimension of that name exists.
        """
        if p.name in self.names:
            raise ValueError(f"duplicate parameter {p.name}")
        self.params.append(p)
        return self

    def replicate(self, block: Iterable[Param], partitions: Sequence[Any],
                  name_fmt: str = "{base}_{pid}") -> "SearchSpace":
        """Replicate a block of dimensions once per partition, partition-major.

        It is the IDL predecessor's layout, ``bin_n1 .. k_klip_n1, bin_n2 ...``.

        Parameters
        ----------
        block : iterable of Param
            The block.
        partitions : sequence
            The partitions' ids, which become the space's.
        name_fmt : str, default "{base}_{pid}"
            Format of each replica's name.

        Returns
        -------
        space : SearchSpace
            The space itself.
        """
        block = list(block)
        self.partitions = list(partitions)
        for pid in self.partitions:
            for b in block:
                self.add(replace(b, name=name_fmt.format(base=b.name, pid=pid),
                                 base=b.name, partition=pid))
        return self

    def with_selection(self, scheme: str = "two_slot", partitions: Optional[Sequence[Any]] = None,
                       max_drop: int = 2, p_include: float = 0.75) -> "SearchSpace":
        """Append the dimensions that select partitions.

        ``"two_slot"`` adds ``max_drop`` whole slots in ``[0, n]``, where a value c >= 1 drops partition
        ``c - 1`` by position and 0 drops none.  ``"binary"`` adds one 0/1 flag per partition.  A decode
        that drops every partition selects them all, and a space of fewer than 2 partitions gets no
        selection.

        Parameters
        ----------
        scheme : {"two_slot", "binary"}, default "two_slot"
            The scheme.
        partitions : sequence or None, default None
            The partitions' ids.  None keeps the space's.
        max_drop : int, default 2
            Number of two-slot slots.
        p_include : float, default 0.75
            Probability that a random draw keeps the partitions of a slot or flag.

        Returns
        -------
        space : SearchSpace
            The space itself.

        Raises
        ------
        ValueError
            If the scheme is unknown.
        """
        if partitions is not None:
            self.partitions = list(partitions)
        n = len(self.partitions)
        if n < 2:
            return self
        self.selection = scheme
        self.selection_max_drop = max_drop
        self.p_include = p_include
        if scheme == "two_slot":
            for s in range(max_drop):
                self.add(Param(f"drop{s+1}", 0, n, kind="int", default=0, role="selection",
                               doc="two-slot jackknife: 0 = no drop, c = drop partition #c"))
        elif scheme == "binary":
            for pid in self.partitions:
                self.add(Param(f"sel_{pid}", 0, 1, kind="int", default=1, role="selection",
                               partition=pid, base="sel"))
        else:
            raise ValueError(scheme)
        return self

    # -- introspection -------------------------------------------------------
    @property
    def names(self) -> List[str]:
        """Names of the dimensions, in order.
        """
        return [p.name for p in self.params]

    @property
    def ndim(self) -> int:
        """Number of dimensions.
        """
        return len(self.params)

    @property
    def lo(self) -> np.ndarray:
        """Lower bound of each dimension.
        """
        return np.array([p.lo for p in self.params])

    @property
    def hi(self) -> np.ndarray:
        """Upper bound of each dimension.
        """
        return np.array([p.hi for p in self.params])

    @property
    def is_int(self) -> np.ndarray:
        """Whether each dimension takes whole values.
        """
        return np.array([p.is_int for p in self.params])

    @property
    def is_categorical(self) -> np.ndarray:
        """Whether each dimension is categorical.
        """
        return np.array([p.kind == "categorical" for p in self.params])

    @property
    def n_categories(self) -> np.ndarray:
        """Number of choices of each dimension, 0 for one not categorical.
        """
        return np.array([len(p.choices) if p.kind == "categorical" else 0 for p in self.params])

    @property
    def reduction_dims(self) -> np.ndarray:
        """Indices of the dimensions that go to the reducer.
        """
        return np.array([i for i, p in enumerate(self.params) if p.role == "reduction"], dtype=int)

    @property
    def selection_dims(self) -> np.ndarray:
        """Indices of the dimensions that select partitions.
        """
        return np.array([i for i, p in enumerate(self.params) if p.role == "selection"], dtype=int)

    def index(self, name: str) -> int:
        """Return a dimension's index.

        Parameters
        ----------
        name : str
            The dimension's name.

        Returns
        -------
        index : int
            Its index.

        Raises
        ------
        ValueError
            If no dimension has that name.
        """
        return self.names.index(name)

    def __getitem__(self, name: str) -> Param:
        """Return a dimension by name.

        Parameters
        ----------
        name : str
            The dimension's name.

        Returns
        -------
        param : Param
            The dimension.
        """
        return self.params[self.index(name)]

    def __contains__(self, name: str) -> bool:
        """Tell whether a dimension of that name exists.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        present : bool
            True when it exists.
        """
        return name in self.names

    def dims_of_partition(self, pid) -> np.ndarray:
        """Return the indices of a partition's reduction dimensions.

        Parameters
        ----------
        pid : object
            The partition's id.

        Returns
        -------
        dims : ndarray of int
            The indices.
        """
        return np.array([i for i, p in enumerate(self.params)
                         if p.partition == pid and p.role == "reduction"], dtype=int)

    def dims_of_base(self, base: str) -> np.ndarray:
        """Return the indices of a base's reduction dimensions, the global one and its replicas.

        Parameters
        ----------
        base : str
            The base name.

        Returns
        -------
        dims : ndarray of int
            The indices.
        """
        return np.array([i for i, p in enumerate(self.params)
                         if p.base == base and p.role == "reduction"], dtype=int)

    @property
    def bases(self) -> List[str]:
        """Base names of the reduction dimensions, each once, in order.
        """
        out: List[str] = []
        for p in self.params:
            if p.role == "reduction" and p.base not in out:
                out.append(p.base)
        return out

    # -- vector operations ---------------------------------------------------
    def sanitize(self, x: np.ndarray) -> np.ndarray:
        """Sanitize each coordinate of a vector (:meth:`Param.sanitize`).

        Parameters
        ----------
        x : ndarray, shape (ndim,)
            The vector.

        Returns
        -------
        x : ndarray, shape (ndim,)
            A sanitized copy.
        """
        x = np.asarray(x, dtype=float).copy()
        for i, p in enumerate(self.params):
            x[i] = p.sanitize(x[i])
        return x

    def tie(self, x: np.ndarray, bases: Optional[Iterable[str]] = None) -> np.ndarray:
        """Give every partition slot of each base the first slot's value.

        Parameters
        ----------
        x : ndarray, shape (ndim,)
            The vector.
        bases : iterable of str or None, default None
            The bases tied.  None ties every replicated base.

        Returns
        -------
        x : ndarray, shape (ndim,)
            A tied copy.

        Notes
        -----
        It succeeds the IDL predecessor's ``near2m_pntie``.
        """
        x = np.asarray(x, dtype=float).copy()
        if bases is None:
            bases = self.bases
        for b in bases:
            if not b:
                continue
            idx = [i for i in self.dims_of_base(b) if self.params[i].partition is not None]
            if len(idx) >= 2:
                x[idx] = x[idx[0]]
        return x

    def random(self, rng: np.random.Generator, tie_all: bool = False) -> np.ndarray:
        """Draw a random vector, for the warm-up and the exploring draws.

        Grid dimensions draw over their entries, and each selection slot or flag keeps its partitions with
        probability ``p_include``.

        Parameters
        ----------
        rng : numpy.random.Generator
            The generator.
        tie_all : bool, default False
            Whether to tie every replicated base (:meth:`tie`).

        Returns
        -------
        x : ndarray, shape (ndim,)
            The vector.
        """
        x = np.array([p.random(rng) for p in self.params])
        if tie_all:
            x = self.tie(x)
        n = len(self.partitions)
        if self.selection == "two_slot":
            for i in self.selection_dims:
                x[i] = 0.0 if rng.random() < self.p_include else float(1 + int(rng.random() * n))
        elif self.selection == "binary":
            for i in self.selection_dims:
                x[i] = 1.0 if rng.random() < self.p_include else 0.0
        return x

    def default_vector(self, overrides: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Return the seeded default configuration as a vector, ``x0``.

        Parameters
        ----------
        overrides : dict or None, default None
            Values by base name, applied to every slot, or by full dimension name.

        Returns
        -------
        x0 : ndarray, shape (ndim,)
            The vector, sanitized, every partition selected unless a selection dimension is overridden.
        """
        overrides = overrides or {}
        x = np.zeros(self.ndim)
        for i, p in enumerate(self.params):
            v = p.default
            if p.base in overrides:
                v = overrides[p.base]
            if p.name in overrides:
                v = overrides[p.name]
            if p.role == "selection":
                v = 0 if self.selection == "two_slot" else 1
                if p.name in overrides:
                    v = overrides[p.name]
            x[i] = p.encode(v) if p.kind == "categorical" else float(v)
        return self.sanitize(x)

    def encode(self, params: Dict[str, Any], per_partition: Optional[Dict[Any, Dict[str, Any]]] = None,
               selected: Optional[Sequence[Any]] = None) -> np.ndarray:
        """Encode parameters as a vector, the inverse of :meth:`decode` as far as it goes.

        Parameters
        ----------
        params : dict
            Global parameters, by base or full dimension name.
        per_partition : dict or None, default None
            Each partition's parameters by base name, over the global ones.
        selected : sequence or None, default None
            The partitions selected, of which two-slot selection holds at most ``selection_max_drop``
            dropped.  None keeps the default selection.

        Returns
        -------
        x : ndarray, shape (ndim,)
            The vector, sanitized, the dimensions not given at their defaults.
        """
        x = self.default_vector()
        for i, p in enumerate(self.params):
            if p.role == "selection":
                continue
            src = None
            if per_partition and p.partition is not None and p.partition in per_partition \
                    and p.base in per_partition[p.partition]:
                src = per_partition[p.partition][p.base]
            elif p.base in params:
                src = params[p.base]
            elif p.name in params:
                src = params[p.name]
            if src is not None:
                x[i] = p.encode(src)
        if selected is not None and self.selection:
            mask = [pid in list(selected) for pid in self.partitions]
            if self.selection == "binary":
                for i in self.selection_dims:
                    x[i] = 1.0 if self.params[i].partition in list(selected) else 0.0
            else:
                dropped = [k + 1 for k, m in enumerate(mask) if not m][: self.selection_max_drop]
                for s, i in enumerate(self.selection_dims):
                    x[i] = float(dropped[s]) if s < len(dropped) else 0.0
        return self.sanitize(x)

    def selected_partitions(self, x: np.ndarray) -> List[Any]:
        """Return the partitions a vector selects, all of them where it would drop every one.

        Parameters
        ----------
        x : ndarray, shape (ndim,)
            The vector.

        Returns
        -------
        partitions : list
            The partitions selected, empty for a space without partitions.
        """
        n = len(self.partitions)
        if n == 0:
            return []
        sel = np.ones(n, dtype=bool)
        if self.selection == "two_slot":
            for i in self.selection_dims:
                c = int(np.clip(round(float(x[i])), 0, n))
                if c >= 1:
                    sel[c - 1] = False
        elif self.selection == "binary":
            for i in self.selection_dims:
                pid = self.params[i].partition
                sel[self.partitions.index(pid)] = bool(int(np.clip(round(float(x[i])), 0, 1)))
        if not sel.any():
            sel[:] = True
        return [pid for pid, s in zip(self.partitions, sel) if s]

    def decode(self, x: np.ndarray) -> Config:
        """Decode a vector into a :class:`Config`.

        The global parameters are the fixed ones, each base's default and the global dimensions' values.
        Each partition takes them with its own slots over them.  A replicated base's global value is the
        median over the partitions, or the most common choice of a categorical, as the IDL predecessor's
        representative values in logs and headers are.

        Parameters
        ----------
        x : ndarray, shape (ndim,)
            The vector, sanitized first.

        Returns
        -------
        config : Config
            The configuration.

        Notes
        -----
        It succeeds the IDL predecessor's ``near2m_pndecode`` and ``near2m_nightsel``.
        """
        x = self.sanitize(x)
        params: Dict[str, Any] = dict(self.fixed)
        # global defaults for every base (so unsearched partition slots inherit them)
        for p in self.params:
            if p.role == "reduction" and p.base not in params:
                params[p.base] = p.default
        for i, p in enumerate(self.params):
            if p.role == "reduction" and p.partition is None:
                params[p.base] = p.decode(x[i])
        per_partition: Dict[Any, Dict[str, Any]] = {}
        for pid in self.partitions:
            d = dict(params)
            for i in self.dims_of_partition(pid):
                d[self.params[i].base] = self.params[i].decode(x[i])
            per_partition[pid] = d
        # representative global values for replicated bases = median over partitions
        # (mirrors the IDL predecessor's "representative scalars" in logs and headers)
        if self.partitions:
            for b in self.bases:
                idx = self.dims_of_base(b)
                if any(self.params[i].partition is not None for i in idx):
                    vals = [per_partition[pid][b] for pid in self.partitions]
                    p0 = self.params[idx[0]]
                    if p0.kind == "categorical":
                        params[b] = max(set(vals), key=vals.count)
                    else:
                        m = float(np.median(vals))
                        params[b] = int(round(m)) if p0.kind == "int" else m
        return Config(params=params, per_partition=per_partition,
                      selected=self.selected_partitions(x), x=x)

    def distinct(self, xa: np.ndarray, xb: np.ndarray, tol: float = 1e-4) -> bool:
        """Tell whether two vectors differ by at least ``tol`` in their summed absolute difference.

        Parameters
        ----------
        xa : ndarray, shape (ndim,)
            The first vector.
        xb : ndarray, shape (ndim,)
            The second vector.
        tol : float, default 1e-4
            The threshold.

        Returns
        -------
        distinct : bool
            True when they differ by at least ``tol``.
        """
        return float(np.sum(np.abs(np.asarray(xa) - np.asarray(xb)))) >= tol

    def distance_to_bounds(self, x: np.ndarray) -> Dict[str, Optional[float]]:
        """Return each coordinate's distance to its nearest bound, over the bounds' span.

        It is a health check: a winner at 0 in many dimensions suggests the bounds are too tight.  It
        raises alarms, so a false one costs more than a missing number, and the dimensions where the
        question does not apply get None rather than 0.  These are the categoricals, whose index has no
        edge to widen, and the pinned dimensions (``span <= 0``), which are not searched.

        Parameters
        ----------
        x : ndarray, shape (ndim,)
            The vector.

        Returns
        -------
        distance : dict of str to float or None
            The distance of each dimension, from 0 at a bound to 0.5 in the middle.

        Notes
        -----
        On the MWC 758 run three of six entries read 0.0 before, and only one of them was a real edge, at
        the number of reference frames that exist, which no one can loosen either.
        """
        out: Dict[str, Optional[float]] = {}
        for i, p in enumerate(self.params):
            if p.kind == "categorical" or p.span <= 0:
                out[p.name] = None
            else:
                out[p.name] = float(min(x[i] - p.lo, p.hi - x[i]) / p.span)
        return out

    # -- serialization -------------------------------------------------------
    def describe(self) -> List[Dict[str, Any]]:
        """Return each dimension as a JSON-ready dict.

        Returns
        -------
        dims : list of dict
            ``name``, ``lo``, ``hi``, ``kind``, ``grid``, ``choices``, ``default``, ``base``, ``partition``,
            ``role`` and ``doc`` of each dimension.
        """
        return [{
            "name": p.name, "lo": p.lo, "hi": p.hi, "kind": p.kind,
            "grid": None if p.grid is None else [float(v) for v in p.grid],
            "choices": None if p.choices is None else list(p.choices),
            "default": p.default, "base": p.base,
            "partition": None if p.partition is None else str(p.partition),
            "role": p.role, "doc": p.doc,
        } for p in self.params]

    def to_dict(self) -> Dict[str, Any]:
        """Return the space as a JSON-ready dict, for :meth:`from_dict`.

        Returns
        -------
        record : dict
            ``params`` (:meth:`describe`), ``fixed``, ``partitions``, ``selection``, ``selection_max_drop`` and
            ``p_include``.
        """
        return {
            "params": self.describe(),
            "fixed": dict(self.fixed),
            "partitions": [str(p) for p in self.partitions],
            "selection": self.selection,
            "selection_max_drop": self.selection_max_drop,
            "p_include": self.p_include,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], project=None) -> "SearchSpace":
        """Rebuild a space from :meth:`to_dict`, the partition ids as strings.

        Parameters
        ----------
        d : dict
            The record.
        project : callable or None, default None
            The feasibility projection, which the record does not hold.

        Returns
        -------
        space : SearchSpace
            The space.
        """
        sp = cls(fixed=d.get("fixed", {}), project=project)
        parts = d.get("partitions", [])
        for e in d["params"]:
            part = e.get("partition")
            sp.params.append(Param(
                name=e["name"], lo=e["lo"], hi=e["hi"], kind=e["kind"], grid=e.get("grid"),
                choices=e.get("choices"), default=e.get("default"), base=e.get("base"),
                partition=part, role=e.get("role", "reduction"), doc=e.get("doc", "")))
        sp.partitions = list(parts)
        sp.selection = d.get("selection")
        sp.selection_max_drop = int(d.get("selection_max_drop", 2))
        sp.p_include = float(d.get("p_include", 0.75))
        return sp

    def setup_text(self) -> str:
        """Describe the space in text: its dimensions, fixed parameters and partitions.

        Returns
        -------
        text : str
            The lines, for a run's setup file.
        """
        lines = ["# searched parameters  (name: [lo, hi]  kind  grid/choices)"]
        for p in self.params:
            extra = ""
            if p.grid is not None:
                extra = f"  grid[{len(p.grid)}]"
            if p.choices is not None:
                extra = f"  choices={list(p.choices)}"
            lines.append(f"  {p.name:<16}: [{p.lo:.3f}, {p.hi:.3f}]  {p.kind}{extra}")
        if self.fixed:
            lines.append("# fixed (unsearched) reduction parameters")
            lines.append("  " + "  ".join(f"{k}={v}" for k, v in self.fixed.items()))
        if self.partitions:
            lines.append(f"# partitions: {', '.join(str(p) for p in self.partitions)}"
                         f"   selection={self.selection}")
        return "\n".join(lines)
