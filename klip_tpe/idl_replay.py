"""Read the IDL predecessor's ``optimize_tpe_results.txt`` logs and score their configurations again.

The configurations go through the Python objective, a regression test against the IDL predecessor.
The log of ``optimize_near_2_tpe.pro`` reads::

    # annulus iter  <opt_names...>   k_klip   medSNR
    ia it  x_1 ... x_ndim   k   score  [kpn=[k1,k2,...]]

:func:`read_idl_log` copes with ``********`` in the score column (a failed evaluation, whose -9999 did
not fit ``F8.3``), with an ``(annulus, iter)`` row repeated after a re-calibration restart, where the
last one wins, and with the trailing ``kpn=[...]`` token.  :func:`read_run_setup` copes with the
labels of ``run_setup.txt`` cut at 14 characters.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["IDLLog", "read_idl_log", "read_run_setup", "replay"]


@dataclass
class IDLLog:
    """The rows of an IDL predecessor's results log.

    Attributes
    ----------
    names : list of str
        The searched parameters' names, from the header.
    annulus : ndarray of int, shape (n,)
        Each row's annulus, as the log numbers it.
    iter : ndarray of int, shape (n,)
        Each row's iteration.
    X : ndarray, shape (n, d)
        The rows' parameter values.
    k : ndarray, shape (n,)
        The number of KL modes each row used.
    score : ndarray, shape (n,)
        The rows' scores, NaN for a failed evaluation.
    kpn : list of (list of int or None)
        Each row's KL modes per night, None where the log has none.
    """
    names: List[str]
    annulus: np.ndarray
    iter: np.ndarray
    X: np.ndarray
    k: np.ndarray
    score: np.ndarray            # nan = failed
    kpn: List[Optional[List[int]]]

    def rows(self, annulus: Optional[int] = None) -> np.ndarray:
        """Return the indices of the rows, of one annulus or of all.

        Parameters
        ----------
        annulus : int or None, default None
            The annulus, as the log numbers it, None for all.

        Returns
        -------
        idx : ndarray of int
            The rows' indices.
        """
        idx = np.arange(len(self.iter))
        if annulus is not None:
            idx = idx[self.annulus == annulus]
        return idx

    def config_dict(self, i: int) -> Dict[str, float]:
        """Return a row's parameter values by name.

        Parameters
        ----------
        i : int
            The row.

        Returns
        -------
        values : dict of str to float
            The values.
        """
        return {n: float(v) for n, v in zip(self.names, self.X[i])}


def read_idl_log(path: str) -> IDLLog:
    """Read an IDL predecessor's results log.

    Parameters
    ----------
    path : str
        The log, ``optimize_tpe_results.txt``.

    Returns
    -------
    log : IDLLog
        Its rows, one per ``(annulus, iter)``, in the order each first appears.
    """
    names: List[str] = []
    rows: Dict[Tuple[int, int], Tuple[List[float], float, float, Optional[List[int]]]] = {}
    order: List[Tuple[int, int]] = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                tok = s[1:].split()
                if tok[:2] == ["annulus", "iter"]:
                    names = tok[2:-2]
                continue
            tok = s.split()
            kpn = None
            if tok[-1].startswith("kpn="):
                kpn = [int(v) for v in tok[-1][5:-1].split(",") if v]
                tok = tok[:-1]
            ia, it = int(tok[0]), int(tok[1])
            sc_tok = tok[-1]
            score = np.nan if "*" in sc_tok else float(sc_tok)
            if score <= -9000:
                score = np.nan
            k = float(tok[-2])
            x = [float(v) for v in tok[2:-2]]
            key = (ia, it)
            if key not in rows:
                order.append(key)
            rows[key] = (x, k, score, kpn)       # last occurrence wins
    ndim = len(names) if names else (len(rows[order[0]][0]) if order else 0)
    X = np.array([rows[k][0] for k in order], float).reshape(-1, ndim)
    return IDLLog(names=names or [f"x{i}" for i in range(ndim)],
                  annulus=np.array([k[0] for k in order]), iter=np.array([k[1] for k in order]),
                  X=X, k=np.array([rows[k][1] for k in order]),
                  score=np.array([rows[k][2] for k in order]),
                  kpn=[rows[k][3] for k in order])


def read_run_setup(path: str) -> Dict[str, Any]:
    """Read an IDL predecessor's ``run_setup.txt``.

    Parameters
    ----------
    path : str
        The file.

    Returns
    -------
    setup : dict
        Each ``key : value`` line, with the parameters' bounds as ``params``, a list of dicts with
        ``name``, ``lo``, ``hi`` and ``int``.

    Notes
    -----
    The labels may be cut at 14 characters, so match a name by its start when in doubt.
    """
    out: Dict[str, Any] = {"params": []}
    with open(path) as f:
        for line in f:
            m = re.match(r"\s{2}(\S+)\s*:\s*\[([-\d.eE+]+),\s*([-\d.eE+]+)\]\s+int=(\d)", line)
            if m:
                out["params"].append({"name": m.group(1), "lo": float(m.group(2)), "hi": float(m.group(3)),
                                      "int": bool(int(m.group(4)))})
                continue
            if ":" in line and not line.startswith("#"):
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def idl_row_to_vector(log: IDLLog, i: int, space) -> np.ndarray:
    """Return a logged row as a vector of a Python search space, matched by parameter name.

    Parameters
    ----------
    log : IDLLog
        The log.
    i : int
        The row.
    space : SearchSpace
        The space, whose names match the IDL predecessor's, such as ``bin_n1``.

    Returns
    -------
    x : ndarray
        The space's default vector, with the row's values where the names match.
    """
    x = space.default_vector()
    d = log.config_dict(i)
    for j, p in enumerate(space.params):
        if p.name in d:
            x[j] = d[p.name]
    return x


def replay(runner, log: IDLLog, indices: Sequence[int], out_path: Optional[str] = None,
           n_repeat: int = 1, log_fn=print) -> List[Dict[str, Any]]:
    """Score logged configurations again with the Python objective, at new injection positions.

    Parameters
    ----------
    runner : Runner
        The runner whose objective scores them.
    log : IDLLog
        The log.
    indices : sequence of int
        The rows to score.
    out_path : str or None, default None
        A JSON file the results are rewritten to after each row, None for none.
    n_repeat : int, default 1
        Number of evaluations per row.
    log_fn : callable, default print
        Function that takes a line of progress.

    Returns
    -------
    results : list of dict
        Per row ``idl_index``, ``annulus``, ``iter``, ``idl_score``, ``py_score`` and ``py_raw`` (one per
        repeat), ``k_idl``, ``kpn_idl`` and ``wall_s``.

    Notes
    -----
    The injection positions differ from the IDL predecessor's, so the agreement is statistical.  Compare
    ranks and means.
    """
    from .space import Config
    results = []
    for n, i in enumerate(indices):
        x = idl_row_to_vector(log, i, runner.space)
        x = runner.space.sanitize(x)         # no projection: IDL already projected
        py = []
        raw = []
        t0 = time.time()
        for _ in range(n_repeat):
            rec, inj, clean = runner.evaluate(x, "replay", tag=f"replay{i}")
            py.append(np.nan if rec.score is None else rec.score)
            raw.append(np.nan if rec.raw_score is None else rec.raw_score)
        row = {"idl_index": int(i), "annulus": int(log.annulus[i]), "iter": int(log.iter[i]),
               "idl_score": None if not np.isfinite(log.score[i]) else float(log.score[i]),
               "py_score": [None if not np.isfinite(v) else float(v) for v in py],
               "py_raw": [None if not np.isfinite(v) else float(v) for v in raw],
               "k_idl": float(log.k[i]), "kpn_idl": log.kpn[i], "wall_s": time.time() - t0}
        results.append(row)
        log_fn(f"replay {n+1}/{len(indices)} (idl iter {log.iter[i]}): IDL {log.score[i]:.3f}  "
               f"PY {np.nanmean(py):.3f} (raw {np.nanmean(raw):.3f})  {row['wall_s']:.0f}s")
        if out_path:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=1)
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Return how the Python scores agree with the IDL predecessor's.

    Parameters
    ----------
    results : list of dict
        :func:`replay`'s results.

    Returns
    -------
    agreement : dict
        ``n`` (rows both scored), and with 3 or more ``spearman``, ``spearman_p``, ``pearson``,
        ``mean_idl``, ``mean_py`` and ``median_ratio`` (Python over the IDL predecessor).
    """
    from scipy import stats
    a = np.array([r["idl_score"] if r["idl_score"] is not None else np.nan for r in results])
    b = np.array([np.nanmean([v for v in r["py_score"] if v is not None]) if any(v is not None for v in r["py_score"]) else np.nan
                  for r in results])
    g = np.isfinite(a) & np.isfinite(b)
    if g.sum() < 3:
        return {"n": int(g.sum())}
    rho, p = stats.spearmanr(a[g], b[g])
    r, _ = stats.pearsonr(a[g], b[g])
    return {"n": int(g.sum()), "spearman": float(rho), "spearman_p": float(p), "pearson": float(r),
            "mean_idl": float(a[g].mean()), "mean_py": float(b[g].mean()),
            "median_ratio": float(np.median(b[g] / np.where(a[g] != 0, a[g], np.nan)))}
