"""Search-space definition: parameters, bounds, integer / categorical kinds, custom
sampling grids, per-partition replication (per-night / per-dataset blocks), partition
selection schemes and the feasibility-projection hook.

Every parameter vector ``x`` handled by the optimizer is a 1-D float array of length
``space.ndim``.  ``SearchSpace.decode(x)`` turns it into a :class:`Config` (plain
dictionaries) that a :class:`~klip_tpe.reducer.Reducer` can consume.  Nothing in this
module knows about telescopes.

Port notes (IDL ``optimize_near_2_tpe.pro``):

* ``bin_n1 ... k_klip_n6`` night-major replication  -> :meth:`SearchSpace.replicate`
* ``near2m_pntie``                                   -> :meth:`SearchSpace.tie`
* ``near2m_pndecode`` / ``near2m_nightsel``          -> :meth:`SearchSpace.decode`
* ``near2m_kgrid`` / ``near2m_ksnap``                -> :class:`Param` ``grid``
* ``drop1``/``drop2`` two-slot jackknife             -> ``selection="two_slot"``
* legacy ``n1..n6`` binary flags                     -> ``selection="binary"``
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = ["Param", "SearchSpace", "Config", "kgrid"]


def kgrid(kmax: int) -> np.ndarray:
    """Non-uniform grid of allowed KL-mode counts: step 1 to 20, step 5 to 50,
    step 10 above (``near2m_kgrid``).  Sampling uniformly over the *entries* puts
    ~2/3 of random draws at k <= 20 where good solutions live."""
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

    kind
        ``"float"``, ``"int"`` or ``"categorical"``.  Categoricals are encoded as
        an index in ``[0, len(choices)-1]``; the TPE treats them with a categorical
        Parzen estimator (no spurious ordinal adjacency).
    grid
        optional explicit list of allowed values; random draws pick an entry
        uniformly, guided proposals are snapped to the nearest entry.
    default
        value used for the seeded default configuration and for unsearched slots.
    base / partition
        set by :meth:`SearchSpace.replicate`; ``partition`` is ``None`` for global
        dimensions.
    role
        ``"reduction"`` (goes to the reducer) or ``"selection"`` (partition
        inclusion encoding, never sent to the reducer).
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
        return self.kind in ("int", "categorical")

    @property
    def span(self) -> float:
        return self.hi - self.lo

    def encode(self, value: Any) -> float:
        """Physical value -> float coordinate in ``[lo, hi]``."""
        if self.kind == "categorical":
            if value in self.choices:
                return float(list(self.choices).index(value))
            return float(np.clip(round(float(value)), 0, len(self.choices) - 1))
        return float(np.clip(float(value), self.lo, self.hi))

    def decode(self, coord: float) -> Any:
        """Float coordinate -> physical value (label for categoricals, int for ints)."""
        c = self.sanitize(coord)
        if self.kind == "categorical":
            return self.choices[int(round(c))]
        if self.kind == "int":
            return int(round(c))
        return float(c)

    def sanitize(self, coord: float) -> float:
        """Clip to bounds, round integers, snap to the grid."""
        c = float(np.clip(coord, self.lo, self.hi))
        if self.grid is not None:
            g = self.grid
            c = float(g[int(np.argmin(np.abs(g - c)))])
        elif self.is_int:
            c = float(round(c))
        return c

    def random(self, rng: np.random.Generator) -> float:
        if self.grid is not None:
            return float(self.grid[rng.integers(len(self.grid))])
        v = self.lo + rng.random() * self.span
        return self.sanitize(v)


@dataclass
class Config:
    """Decoded parameter vector.

    params
        global reduction parameters ``{base: value}`` (including unsearched ones at
        their defaults) -- what a single-partition reducer consumes.
    per_partition
        ``{partition_id: {base: value}}`` for **every** partition of the space;
        partition-specific slots override the global value.
    selected
        list of the partition ids that are switched on (all of them when the space
        has no selection scheme).  Never empty.
    x
        the sanitized vector this config was decoded from.
    """

    params: Dict[str, Any]
    per_partition: Dict[Any, Dict[str, Any]]
    selected: List[Any]
    x: np.ndarray

    def params_for(self, partition) -> Dict[str, Any]:
        return self.per_partition.get(partition, dict(self.params))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": dict(self.params),
            "per_partition": {str(k): dict(v) for k, v in self.per_partition.items()},
            "selected": [str(s) for s in self.selected],
            "x": [float(v) for v in self.x],
        }


class SearchSpace:
    """Ordered collection of :class:`Param` objects with partition support.

    Parameters
    ----------
    params
        global (non-replicated) parameters.
    fixed
        ``{name: value}`` of reduction parameters that are *not* searched but must
        still be handed to the reducer (e.g. ``spat_mean=0``).
    project
        optional feasibility projection ``f(x, space, **ctx) -> x'`` applied by the
        runner on **every** proposal path.  Keep it a pure function.
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
        if p.name in self.names:
            raise ValueError(f"duplicate parameter {p.name}")
        self.params.append(p)
        return self

    def replicate(self, block: Iterable[Param], partitions: Sequence[Any],
                  name_fmt: str = "{base}_{pid}") -> "SearchSpace":
        """Replicate ``block`` once per partition (partition-major ordering, like the
        IDL ``bin_n1 .. k_klip_n1, bin_n2 ...`` layout)."""
        block = list(block)
        self.partitions = list(partitions)
        for pid in self.partitions:
            for b in block:
                self.add(replace(b, name=name_fmt.format(base=b.name, pid=pid),
                                 base=b.name, partition=pid))
        return self

    def with_selection(self, scheme: str = "two_slot", partitions: Optional[Sequence[Any]] = None,
                       max_drop: int = 2, p_include: float = 0.75) -> "SearchSpace":
        """Append partition-selection dimensions.

        ``"two_slot"``: ``max_drop`` integer slots in ``[0, n_part]``; slot value c>=1
        drops partition ``c-1`` (positional index), 0 = no drop.
        ``"binary"``: one 0/1 flag per partition.
        An all-dropped decode falls back to all partitions selected.
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
        return [p.name for p in self.params]

    @property
    def ndim(self) -> int:
        return len(self.params)

    @property
    def lo(self) -> np.ndarray:
        return np.array([p.lo for p in self.params])

    @property
    def hi(self) -> np.ndarray:
        return np.array([p.hi for p in self.params])

    @property
    def is_int(self) -> np.ndarray:
        return np.array([p.is_int for p in self.params])

    @property
    def is_categorical(self) -> np.ndarray:
        return np.array([p.kind == "categorical" for p in self.params])

    @property
    def n_categories(self) -> np.ndarray:
        return np.array([len(p.choices) if p.kind == "categorical" else 0 for p in self.params])

    @property
    def reduction_dims(self) -> np.ndarray:
        return np.array([i for i, p in enumerate(self.params) if p.role == "reduction"], dtype=int)

    @property
    def selection_dims(self) -> np.ndarray:
        return np.array([i for i, p in enumerate(self.params) if p.role == "selection"], dtype=int)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def __getitem__(self, name: str) -> Param:
        return self.params[self.index(name)]

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def dims_of_partition(self, pid) -> np.ndarray:
        return np.array([i for i, p in enumerate(self.params)
                         if p.partition == pid and p.role == "reduction"], dtype=int)

    def dims_of_base(self, base: str) -> np.ndarray:
        return np.array([i for i, p in enumerate(self.params)
                         if p.base == base and p.role == "reduction"], dtype=int)

    @property
    def bases(self) -> List[str]:
        out: List[str] = []
        for p in self.params:
            if p.role == "reduction" and p.base not in out:
                out.append(p.base)
        return out

    # -- vector operations ---------------------------------------------------
    def sanitize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        for i, p in enumerate(self.params):
            x[i] = p.sanitize(x[i])
        return x

    def tie(self, x: np.ndarray, bases: Optional[Iterable[str]] = None) -> np.ndarray:
        """Give every partition slot of each base the first slot's value
        (``near2m_pntie``).  ``bases=None`` ties all replicated bases."""
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
        """Uniform random vector (warm-up / explore draw).  Grid params draw over grid
        entries; selection dims use the ``p_include`` bias of the scheme."""
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
        """The seeded default configuration as a vector (``x0``).  ``overrides`` may
        address a base name (applied to every slot) or a full dim name."""
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
        """Inverse of :meth:`decode` (best effort)."""
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
        # (mirrors the IDL "representative scalars" used in logs/headers)
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
        return float(np.sum(np.abs(np.asarray(xa) - np.asarray(xb)))) >= tol

    def distance_to_bounds(self, x: np.ndarray) -> Dict[str, float]:
        """Normalised distance of each coordinate to its nearest bound (health check:
        a winner pinned at 0 for many dims suggests the bounds are too tight)."""
        out = {}
        for i, p in enumerate(self.params):
            if p.span <= 0:
                out[p.name] = 0.0
            else:
                out[p.name] = float(min(x[i] - p.lo, p.hi - x[i]) / p.span)
        return out

    # -- serialisation -------------------------------------------------------
    def describe(self) -> List[Dict[str, Any]]:
        return [{
            "name": p.name, "lo": p.lo, "hi": p.hi, "kind": p.kind,
            "grid": None if p.grid is None else [float(v) for v in p.grid],
            "choices": None if p.choices is None else list(p.choices),
            "default": p.default, "base": p.base,
            "partition": None if p.partition is None else str(p.partition),
            "role": p.role, "doc": p.doc,
        } for p in self.params]

    def to_dict(self) -> Dict[str, Any]:
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
