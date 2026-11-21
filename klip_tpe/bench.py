"""Matched-budget benchmark of the search modes, TPE against random and grid search on the same data.

Every slot of a batch has the same data, metric, calibration and validation, and only the search mode
and the seed differ.  It succeeds the IDL predecessor's ``near2_bench``, ``near2_bench_conv``,
``near2_bench_plot`` and ``near2_bench_restart``.

Three lessons of the IDL predecessor's campaigns are built in (brief §8):

1. **Benchmark on a sub-threshold injection.**  With a bright injection the metric degenerates, since
   ``k -> 1`` maximizes throughput and a coarse grid "wins".  The synthetic benchmark therefore
   calibrates the injection contrast *once* with the default configuration and then **forces** that
   contrast for every run, so all modes and seeds are scored at the same sub-threshold contrast.
   Callers of :func:`run_benchmark` with real data should do the same, with
   ``CalibrationConfig(forced=[c])`` and ``c`` about half the 5-sigma limit.
2. **Report validated numbers.**  The convergence curves are running best *search* scores, biased
   upward and labeled so, and the comparison table of :func:`summarize_bench` uses the validated
   winner only.
3. **Tag the accumulating result files.**  ``bench_summary.txt`` is appended across campaigns, every
   row starts with ``bench_tag``, and every reader here filters on it.

The layout under ``out_dir`` is::

    bench_tag.txt                           the most recent batch tag
    bench_summary.txt                       appended, one row per (mode, seed, annulus):
        bench_tag mode seed annulus n_iter n_init seeded_default_score search_best validated_best run_dir
    <bench_tag>_<mode>_s<seed>/             one Runner run directory per slot
    fig_bench_conv_<bench_tag>.png          (CLI) convergence curves

Every run writes its own ``run_dir`` (``results.jsonl``, ``checkpoint.json`` and the rest).  A slot is
*finished* when ``final_results.json`` exists and *resumable* when only ``checkpoint.json`` does, or a
file-sync client's copy of it (:func:`klip_tpe.runner.checkpoint_candidates`).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .runner import Runner

try:
    from .runner import checkpoint_candidates
except ImportError:
    # A process started before runner.py grew this (2026-09-23) keeps its old runner module
    # in memory, and can still import THIS file for the first time late in its run
    # (plots.load_run imports bench lazily).  Its old runner reads only checkpoint.json, so
    # the old test is the consistent one there.
    def checkpoint_candidates(run_dir: str) -> List[str]:
        p = os.path.join(run_dir, "checkpoint.json")
        return [p] if os.path.exists(p) else []

__all__ = ["run_benchmark", "bench_convergence", "bench_status", "bench_restart", "summarize_bench",
           "read_summary", "summary_run_dirs", "read_records", "running_best", "slot_dir", "new_bench_tag",
           "SUMMARY_COLUMNS", "make_synthetic_bench", "main"]

#: the columns of a ``bench_summary.txt`` row, in order
SUMMARY_COLUMNS = ["bench_tag", "mode", "seed", "annulus", "n_iter", "n_init",
                   "seeded_default_score", "search_best", "validated_best", "run_dir"]

#: a function ``make_runner(mode, seed, run_dir)`` returning a runner on the caller's data
MakeRunner = Callable[[str, int, str], Runner]


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def new_bench_tag() -> str:
    """Return a new batch tag, ``bench_`` and the local time to the second.

    Returns
    -------
    tag : str
        The tag, such as ``bench_20261117093000``.
    """
    return "bench_" + time.strftime("%Y%m%d%H%M%S")


def slot_dir(out_dir: str, bench_tag: str, mode: str, seed: int) -> str:
    """Return the run directory of a slot of a batch.

    Parameters
    ----------
    out_dir : str
        The benchmark's directory.
    bench_tag : str
        The batch's tag.
    mode : str
        The search mode, such as ``"tpe"``.
    seed : int
        The seed.

    Returns
    -------
    run_dir : str
        ``out_dir/<bench_tag>_<mode>_s<seed>``.
    """
    return os.path.join(out_dir, f"{bench_tag}_{mode}_s{int(seed)}")


def _fmt(v: Optional[float]) -> str:
    """Return a score as the summary file writes it.

    Parameters
    ----------
    v : float or None
        The score.

    Returns
    -------
    text : str
        The score to 4 decimals, ``nan`` for None or a value that is not finite.
    """
    return "nan" if v is None or not np.isfinite(v) else f"{v:.4f}"


def read_records(run_dir: str, annulus: Optional[int] = None, last_segment: bool = True) -> List[Dict[str, Any]]:
    """Return the records of a run's ``results.jsonl``, each evaluation once.

    A re-calibration restart sets ``index`` back to 0, which begins a segment.  Repeated evaluations are
    dropped within each segment, the first of each kept.

    Parameters
    ----------
    run_dir : str
        The run directory.
    annulus : int or None, default None
        The annulus, counted from 0, None for every annulus.
    last_segment : bool, default True
        Whether to keep only the last segment, the search after the last restart.

    Returns
    -------
    records : list of dict
        The records, in the order of the file, empty without it.  A line half written by a crash is
        skipped.

    Notes
    -----
    ``results.jsonl`` is append-only, and an index can land in it twice.  Two processes on one run
    directory both append, and a resume can log again what it replayed.  Counting those twice inflates
    the evaluation count and puts a flat step in every running-best curve, the quantity the benchmark
    reports.  The de-duplication is per segment, since the calibration loop replays indices 0 to n at
    each trial contrast, and those are not repeats.
    """
    path = os.path.join(run_dir, "results.jsonl")
    if not os.path.exists(path):
        return []
    recs: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except ValueError:     # half a line left by a crash: the resume writes it again
                    continue
    if annulus is not None:
        recs = [r for r in recs if int(r.get("annulus", 0)) == int(annulus)]
    starts = [i for i, r in enumerate(recs) if int(r.get("index", 0)) == 0]
    if last_segment and recs and starts:
        recs = recs[starts[-1]:]
        starts = [0]
    # segment boundaries, always covering the whole list even if it does not open at index 0
    bounds = sorted({0, *starts, len(recs)})
    out: List[Dict[str, Any]] = []
    for a, b in zip(bounds, bounds[1:]):
        seen = set()
        for r in recs[a:b]:
            key = (r.get("annulus"), r.get("index"))
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out


def running_best(scores: Sequence[Optional[float]]) -> np.ndarray:
    """Return the running maximum of a sequence of scores, skipping failed evaluations.

    Parameters
    ----------
    scores : sequence of float or None
        The scores, None or NaN for a failed evaluation.

    Returns
    -------
    best : ndarray, shape (n,)
        The best score up to each evaluation, NaN before the first that succeeded.
    """
    out = np.full(len(scores), np.nan)
    best = np.nan
    for i, s in enumerate(scores):
        if s is not None and np.isfinite(s):
            best = s if not np.isfinite(best) else max(best, s)
        out[i] = best
    return out


def _run_mode(run_dir: str) -> Optional[str]:
    """Return a run's search mode, from its ``run_setup.json`` or ``checkpoint.json``.

    Parameters
    ----------
    run_dir : str
        The run directory.

    Returns
    -------
    mode : str or None
        The mode, None when neither file gives it.
    """
    for name in ("run_setup.json", "checkpoint.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)["config"]["search_mode"]
            except Exception:
                continue
    return None


def _run_config(run_dir: str) -> Optional[Dict[str, Any]]:
    """Return a run's configuration, from its ``checkpoint.json`` or ``run_setup.json``.

    Parameters
    ----------
    run_dir : str
        The run directory.

    Returns
    -------
    config : dict or None
        The configuration (``RunConfig.to_dict``), None when neither file gives it.
    """
    for name in ("checkpoint.json", "run_setup.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)["config"]
            except Exception:
                continue
    return None


# ----------------------------------------------------------------------------
# summary file
# ----------------------------------------------------------------------------
def _append_summary(out_dir: str, rows: Iterable[Dict[str, Any]]) -> None:
    """Append rows to a benchmark's ``bench_summary.txt``, a new file starting with the column header.

    Parameters
    ----------
    out_dir : str
        The benchmark's directory.
    rows : iterable of dict
        The rows, each with the keys of :data:`SUMMARY_COLUMNS`.
    """
    path = os.path.join(out_dir, "bench_summary.txt")
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("# " + " ".join(SUMMARY_COLUMNS) + "\n")
        for r in rows:
            f.write(f"{r['bench_tag']} {r['mode']:<8s} {int(r['seed']):6d} {int(r['annulus']) + 1:3d} "
                    f"{int(r['n_iter']):5d} {int(r['n_init']):4d} {_fmt(r['seeded_default_score']):>10s} "
                    f"{_fmt(r['search_best']):>10s} {_fmt(r['validated_best']):>10s} {r['run_dir']}\n")


def read_summary(out_dir: str, bench_tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the rows of a benchmark's ``bench_summary.txt`` for one batch.

    Parameters
    ----------
    out_dir : str
        The benchmark's directory.
    bench_tag : str or None, default None
        The batch.  None takes the tag in ``bench_tag.txt``, and ``"all"`` takes every row, mixing
        batches on purpose.

    Returns
    -------
    rows : list of dict
        Each row's :data:`SUMMARY_COLUMNS`, the annulus counted from 0, empty without the file.  Comment
        lines and short lines are skipped.
    """
    path = os.path.join(out_dir, "bench_summary.txt")
    if bench_tag is None:
        bench_tag = _current_tag(out_dir)
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            t = s.split()
            if len(t) < len(SUMMARY_COLUMNS):
                continue
            row = {"bench_tag": t[0], "mode": t[1], "seed": int(t[2]), "annulus": int(t[3]) - 1,
                   "n_iter": int(t[4]), "n_init": int(t[5]), "seeded_default_score": float(t[6]),
                   "search_best": float(t[7]), "validated_best": float(t[8]),
                   "run_dir": " ".join(t[9:])}
            if bench_tag == "all" or row["bench_tag"] == bench_tag:
                rows.append(row)
    return rows


def summary_run_dirs(out_dir: str, bench_tag: Optional[str] = None) -> List[str]:
    """Return the slot directories a batch's summary rows came from, the runs its figures draw.

    They are found under ``out_dir`` as it is now (:func:`slot_dir`), so a moved batch still resolves,
    and the absolute ``run_dir`` a row recorded is the fallback.

    Parameters
    ----------
    out_dir : str
        The benchmark's directory.
    bench_tag : str or None, default None
        The batch, as :func:`read_summary` takes it.

    Returns
    -------
    run_dirs : list of str
        The directories that exist, sorted, each once.

    Notes
    -----
    Globbing ``bench_*_*_s*`` instead also picked up other tags' slots and those
    ``scripts/supersede_bench_mode.py`` renamed ``..._sN_superseded_<date>``.  The paper's E2 panel drew
    16 grid seeds that way, 8 of them retired, beside a bar chart of the 8 that count.
    """
    out: List[str] = []
    for r in read_summary(out_dir, bench_tag):
        d = slot_dir(out_dir, r["bench_tag"], r["mode"], r["seed"])
        if not os.path.isdir(d) and os.path.isdir(r["run_dir"]):
            d = r["run_dir"]
        if os.path.isdir(d) and d not in out:
            out.append(d)
    return sorted(out)


def _current_tag(out_dir: str) -> Optional[str]:
    """Return the batch tag in a benchmark's ``bench_tag.txt``.

    Parameters
    ----------
    out_dir : str
        The benchmark's directory.

    Returns
    -------
    tag : str or None
        The tag, None without the file or with an empty one.
    """
    p = os.path.join(out_dir, "bench_tag.txt")
    if os.path.exists(p):
        with open(p) as f:
            return f.read().strip() or None
    return None


def _rows_from_run(run_dir: str, bench_tag: str, mode: str, seed: int) -> List[Dict[str, Any]]:
    """Return the summary rows of a finished run, one per annulus.

    Parameters
    ----------
    run_dir : str
        The run directory, with its ``final_results.json``.
    bench_tag : str
        The batch's tag.
    mode : str
        The search mode.
    seed : int
        The seed.

    Returns
    -------
    rows : list of dict
        Each annulus' :data:`SUMMARY_COLUMNS`.  The seeded default's score is the last ``seed`` record's,
        else the first record's, and the validated best is None for an annulus not validated.
    """
    with open(os.path.join(run_dir, "final_results.json")) as f:
        final = json.load(f)
    cfg = _run_config(run_dir) or {}
    rows = []
    for ann in final.get("annuli", []):
        ia = int(ann["annulus"])
        recs = read_records(run_dir, ia)
        seeded = [r["score"] for r in recs if r.get("phase") == "seed"]
        y0 = seeded[-1] if seeded else (recs[0]["score"] if recs else None)
        n_iter = cfg.get("n_iter", len(recs))
        n_init = cfg.get("n_init", 0)
        per = lambda v: int(v[min(ia, len(v) - 1)]) if isinstance(v, (list, tuple)) else int(v)
        rows.append({"bench_tag": bench_tag, "mode": mode, "seed": int(seed), "annulus": ia,
                     "n_iter": per(n_iter), "n_init": per(n_init),
                     "seeded_default_score": y0, "search_best": ann.get("search_best_score"),
                     "validated_best": ann.get("winner_score") if ann.get("validated") else None,
                     "run_dir": os.path.abspath(run_dir)})
    return rows


# ----------------------------------------------------------------------------
# slots
# ----------------------------------------------------------------------------
def bench_status(out_dir: str, bench_tag: Optional[str] = None, modes: Sequence[str] = ("tpe", "random", "grid"),
                 seeds: Sequence[int] = (0, 1, 2), n_iter: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return the state of each slot of a batch.

    Parameters
    ----------
    out_dir : str
        The benchmark's directory.
    bench_tag : str or None, default None
        The batch, None for the tag in ``bench_tag.txt``.
    modes : sequence of str, default ("tpe", "random", "grid")
        The search modes.
    seeds : sequence of int, default (0, 1, 2)
        The seeds.
    n_iter : int or None, default None
        The batch's budget, None to leave the budgets unchecked.

    Returns
    -------
    slots : list of dict
        One per (mode, seed), with ``bench_tag``, ``mode``, ``seed``, ``run_dir`` and ``status``.  The status
        is ``"finished"`` with a ``final_results.json``, ``"resumable"`` with a checkpoint, else
        ``"missing"``, and ``"budget_mismatch"`` when an existing run's ``n_iter`` is not ``n_iter``.
    """
    if bench_tag is None:
        bench_tag = _current_tag(out_dir)
    out = []
    for mode in modes:
        for seed in seeds:
            d = slot_dir(out_dir, bench_tag, mode, seed)
            st = "missing"
            if os.path.exists(os.path.join(d, "final_results.json")):
                st = "finished"
            elif checkpoint_candidates(d):           # checkpoint.json, or a synced copy of it
                st = "resumable"
            if st != "missing" and n_iter is not None:
                cfg = _run_config(d) or {}
                ni = cfg.get("n_iter")
                ni = ni[0] if isinstance(ni, list) else ni
                if ni is not None and int(ni) != int(n_iter):
                    st = "budget_mismatch"
            out.append({"bench_tag": bench_tag, "mode": mode, "seed": int(seed), "run_dir": d, "status": st})
    return out


def _run_slot(make_runner: MakeRunner, mode: str, seed: int, run_dir: str, bench_tag: str,
              n_iter: int, n_init: Optional[int], log) -> Runner:
    """Run one slot of a batch, from its checkpoint when it has one.

    A new slot takes the caller's runner with its search mode, batch tag, budget, warm-up and seed set.  A
    resumed one keeps the checkpoint's configuration and takes only the runner's data objects.

    Parameters
    ----------
    make_runner : callable
        Function ``make_runner(mode, seed, run_dir)`` returning a :class:`Runner` on the caller's data.
    mode : str
        The search mode.
    seed : int
        The seed.
    run_dir : str
        The slot's run directory.
    bench_tag : str
        The batch's tag.
    n_iter : int
        The budget.
    n_init : int or None
        The warm-up rows, None for the runner's own.
    log : callable
        Unused, since the runner logs.

    Returns
    -------
    runner : Runner
        The runner, after its run.
    """
    r = make_runner(mode, seed, run_dir)
    if checkpoint_candidates(run_dir):
        # resume: config comes from the checkpoint by design; we only reuse the data objects
        r = Runner.resume(run_dir, r.reducer, r.objective, r.sampler, project=r.space.project,
                          throughput_fn=r.throughput_fn, log=r.log)
    else:
        c = r.cfg
        c.search_mode, c.bench_tag, c.n_iter = mode, bench_tag, int(n_iter)
        if n_init is not None:
            c.n_init = int(n_init)
        if c.seed != int(seed):
            c.seed = int(seed)
            r.rng = np.random.default_rng(int(seed))
    r.run()
    return r


def run_benchmark(make_runner: MakeRunner, modes: Sequence[str] = ("tpe", "random", "grid"),
                  seeds: Sequence[int] = (0, 1, 2), n_iter: int = 60, n_init: Optional[int] = None,
                  bench_tag: Optional[str] = None, out_dir: str = "bench", log: Callable[[str], None] = print
                  ) -> List[Dict[str, Any]]:
    """Run every (mode, seed) slot of a batch at the same budget and append its summary rows.

    The harness sets ``search_mode``, ``seed``, ``n_iter``, ``n_init`` and ``bench_tag`` on each runner's
    configuration, so the slots differ in their search mode and seed only.  Slots already in the summary
    are skipped, finished ones summarized without running and checkpointed ones resumed, so calling it
    twice, or :func:`bench_restart`, is safe.  The batch's tag goes to ``bench_tag.txt``.

    Parameters
    ----------
    make_runner : callable
        Function ``make_runner(mode, seed, run_dir)`` returning a :class:`Runner` on the caller's data, its
        reducer, space, objective, sampler and configuration.
    modes : sequence of str, default ("tpe", "random", "grid")
        The search modes.
    seeds : sequence of int, default (0, 1, 2)
        The seeds.
    n_iter : int, default 60
        The budget of every slot.
    n_init : int or None, default None
        The warm-up rows of every slot, None for the runner's own.
    bench_tag : str or None, default None
        The batch, None for a new one (:func:`new_bench_tag`).
    out_dir : str, default "bench"
        The benchmark's directory.
    log : callable, default print
        Function taking each progress line.

    Returns
    -------
    rows : list of dict
        The summary rows this call appended.

    Raises
    ------
    ValueError
        If a slot would run at another budget or warm-up than the batch's rows so far.

    Notes
    -----
    A batch compares its arms at the budget they share.  E2's grid arm was run again after
    ``scripts/supersede_bench_mode.py`` without the ``BENCH_NITER=1000`` of its batch, and so ran 800
    evaluations beside TPE and random search at 1,000.  The check refuses before any slot runs, and
    finished slots are only read, whatever the call asks.
    """
    os.makedirs(out_dir, exist_ok=True)
    if bench_tag is None:
        bench_tag = new_bench_tag()
    with open(os.path.join(out_dir, "bench_tag.txt"), "w") as f:
        f.write(bench_tag + "\n")
    recorded = read_summary(out_dir, bench_tag)
    done = {(r["mode"], r["seed"]) for r in recorded}
    # One batch, one budget: its arms are compared at the n_iter they share.  E2's grid arm
    # was re-run (after supersede_bench_mode.py) without the BENCH_NITER=1000 the batch had
    # run under, and so ran 800 evaluations beside tpe and random at 1000.  Refuse before
    # running a slot.  Finished and summarized ones are only read, whatever this call asks.
    to_run = [(m, int(s)) for m in modes for s in seeds if (m, int(s)) not in done
              and not os.path.exists(os.path.join(slot_dir(out_dir, bench_tag, m, s), "final_results.json"))]
    if to_run and recorded:
        had_it = sorted({int(r["n_iter"]) for r in recorded})
        had_in = sorted({int(r["n_init"]) for r in recorded})
        if had_it != [int(n_iter)] or (n_init is not None and had_in != [int(n_init)]):
            raise ValueError(
                f"batch {bench_tag} in {out_dir} was run at n_iter={'/'.join(map(str, had_it))}, "
                f"n_init={'/'.join(map(str, had_in))}; this call would run {len(to_run)} slot(s) "
                f"({', '.join(f'{m} s{s}' for m, s in to_run[:4])}{'...' if len(to_run) > 4 else ''}) "
                f"at n_iter={n_iter}, n_init={n_init}.  A batch compares its arms at one budget: ask "
                f"for the batch's (paper_runs: BENCH_NITER={had_it[-1]}), or start a new batch.")
    rows: List[Dict[str, Any]] = []
    for mode in modes:
        for seed in seeds:
            d = slot_dir(out_dir, bench_tag, mode, seed)
            if (mode, int(seed)) in done:
                log(f"[bench {bench_tag}] {mode} seed {seed}: already summarized, skipping")
                continue
            if os.path.exists(os.path.join(d, "final_results.json")):
                log(f"[bench {bench_tag}] {mode} seed {seed}: finished run found, summarizing")
            else:
                log(f"[bench {bench_tag}] {mode} seed {seed}: running (n_iter={n_iter}, n_init={n_init})")
                _run_slot(make_runner, mode, int(seed), d, bench_tag, n_iter, n_init, log)
            new_rows = _rows_from_run(d, bench_tag, mode, int(seed))
            _append_summary(out_dir, new_rows)
            rows.extend(new_rows)
            for r in new_rows:
                log(f"[bench {bench_tag}] {mode} seed {seed} ann {r['annulus'] + 1}: default {_fmt(r['seeded_default_score'])}"
                    f"  search best {_fmt(r['search_best'])}  validated {_fmt(r['validated_best'])}")
    return rows


def bench_restart(out_dir: str, make_runner: MakeRunner, bench_tag: Optional[str] = None,
                  modes: Sequence[str] = ("tpe", "random", "grid"), seeds: Sequence[int] = (0, 1, 2),
                  n_iter: int = 60, n_init: Optional[int] = None, log: Callable[[str], None] = print
                  ) -> List[Dict[str, Any]]:
    """Tally the slots of a batch, resume the checkpointed ones and run the missing ones.

    Slots whose stored budget is not ``n_iter`` are reported and left alone.  It succeeds the IDL
    predecessor's ``near2_bench_restart``.

    Parameters
    ----------
    out_dir : str
        The benchmark's directory.
    make_runner : callable
        Function ``make_runner(mode, seed, run_dir)`` returning a :class:`Runner`, as :func:`run_benchmark`
        takes it.
    bench_tag : str or None, default None
        The batch, None for the tag in ``bench_tag.txt``.
    modes : sequence of str, default ("tpe", "random", "grid")
        The search modes.
    seeds : sequence of int, default (0, 1, 2)
        The seeds.
    n_iter : int, default 60
        The batch's budget.
    n_init : int or None, default None
        The warm-up rows, None for the runner's own.
    log : callable, default print
        Function taking each progress line.

    Returns
    -------
    rows : list of dict
        The summary rows appended (:func:`run_benchmark`).

    Raises
    ------
    FileNotFoundError
        If no tag is given and ``out_dir`` has no ``bench_tag.txt``.
    """
    if bench_tag is None:
        bench_tag = _current_tag(out_dir)
        if bench_tag is None:
            raise FileNotFoundError(f"no bench_tag.txt in {out_dir}; pass bench_tag")
    slots = bench_status(out_dir, bench_tag, modes, seeds, n_iter)
    for st in ("finished", "resumable", "missing", "budget_mismatch"):
        sl = [s for s in slots if s["status"] == st]
        if sl:
            log(f"[bench {bench_tag}] {st:<15s}: " + ", ".join(f"{s['mode']}/s{s['seed']}" for s in sl))
    todo = [(s["mode"], s["seed"]) for s in slots if s["status"] != "budget_mismatch"]
    rows: List[Dict[str, Any]] = []
    for mode in modes:
        sd = [s for m, s in todo if m == mode]
        if sd:
            rows.extend(run_benchmark(make_runner, (mode,), sd, n_iter, n_init, bench_tag, out_dir, log))
    return rows


# ----------------------------------------------------------------------------
# convergence + summary
# ----------------------------------------------------------------------------
def bench_convergence(run_dirs: Sequence[str], annulus: int = 0, modes: Optional[Dict[str, str]] = None
                      ) -> Dict[str, Dict[str, Any]]:
    """Return each mode's running-best curves of the search score, biased upward and not validated.

    Parameters
    ----------
    run_dirs : sequence of str
        The run directories.
    annulus : int, default 0
        The annulus, counted from 0.
    modes : dict or None, default None
        Each run directory's mode, None to read the modes from the runs' setups.

    Returns
    -------
    curves : dict
        Per mode, ``n`` (the evaluation numbers, from 1), ``mean`` and ``sd`` across the runs, which are the
        seeds, ``curves`` (shape ``(nrun, n)``, a shorter run padded with its last value), ``run_dirs`` and
        ``default``, the mean of the seeded defaults' scores.  A run without records is left out.
    """
    by_mode: Dict[str, List[np.ndarray]] = {}
    dirs: Dict[str, List[str]] = {}
    defaults: Dict[str, List[float]] = {}
    for d in run_dirs:
        recs = read_records(d, annulus)
        if not recs:
            continue
        mode = (modes or {}).get(d) or _run_mode(d) or "unknown"
        rb = running_best([r.get("score") for r in recs])
        by_mode.setdefault(mode, []).append(rb)
        dirs.setdefault(mode, []).append(d)
        seeded = [r["score"] for r in recs if r.get("phase") == "seed" and r.get("score") is not None]
        if seeded:
            defaults.setdefault(mode, []).append(float(seeded[-1]))
    out: Dict[str, Dict[str, Any]] = {}
    for mode, curves in by_mode.items():
        n = max(len(c) for c in curves)
        M = np.full((len(curves), n), np.nan)
        for i, c in enumerate(curves):
            M[i, :len(c)] = c
            if len(c) < n:
                M[i, len(c):] = c[-1]
        with np.errstate(all="ignore"):
            mean = np.nanmean(M, axis=0)
            sd = np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.zeros(n)
        out[mode] = {"n": np.arange(1, n + 1), "mean": mean, "sd": sd, "curves": M, "run_dirs": dirs[mode],
                     "default": float(np.mean(defaults[mode])) if defaults.get(mode) else np.nan}
    return out


def summarize_bench(out_dir: str, bench_tag: Optional[str] = None, ref: str = "tpe",
                    log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Log and return a batch's validated bests per mode and their paired differences from ``ref``.

    ``modes`` gives each mode's ``n``, ``validated_mean``, ``validated_sd``, ``search_mean`` and
    ``optimism``, the search mean less the validated one.  ``paired`` gives each other mode's ``n_pairs``,
    ``mean_diff`` (``ref`` less it), ``se``, ``t`` and ``wins``.

    Parameters
    ----------
    out_dir : str
        The benchmark's directory.
    bench_tag : str or None, default None
        The batch, as :func:`read_summary` takes it.
    ref : str, default "tpe"
        The mode the others are paired against, by seed and annulus.
    log : callable, default print
        Function taking each line of the table.

    Returns
    -------
    summary : dict
        ``bench_tag``, ``default`` (the median seeded default's score), ``n_rows``, ``modes`` and ``paired``.
        Without rows ``default`` is absent and ``modes`` and ``paired`` are empty.
    """
    rows = read_summary(out_dir, bench_tag)
    if bench_tag is None:
        bench_tag = _current_tag(out_dir)
    if not rows:
        log(f"no rows for bench_tag {bench_tag!r} in {out_dir}")
        return {"bench_tag": bench_tag, "modes": {}, "paired": {}, "n_rows": 0}
    modes = []
    for r in rows:
        if r["mode"] not in modes:
            modes.append(r["mode"])
    val = {m: {(r["seed"], r["annulus"]): r["validated_best"] for r in rows if r["mode"] == m} for m in modes}
    srch = {m: {(r["seed"], r["annulus"]): r["search_best"] for r in rows if r["mode"] == m} for m in modes}
    default = float(np.nanmedian([r["seeded_default_score"] for r in rows]))
    out: Dict[str, Any] = {"bench_tag": bench_tag, "default": default, "n_rows": len(rows), "modes": {}, "paired": {}}
    log(f"benchmark {bench_tag}: {len(rows)} rows, seeded default (median) {default:.3f}")
    log(f"{'mode':<8s} {'n':>3s} {'validated mean':>15s} {'sd':>7s} {'search mean':>12s}   (optimism = search - validated)")
    for m in modes:
        v = np.array(list(val[m].values()), float)
        s = np.array(list(srch[m].values()), float)
        st = {"n": int(v.size), "validated_mean": float(np.nanmean(v)),
              "validated_sd": float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0,
              "search_mean": float(np.nanmean(s)), "optimism": float(np.nanmean(s - v))}
        out["modes"][m] = st
        log(f"{m:<8s} {st['n']:3d} {st['validated_mean']:15.3f} {st['validated_sd']:7.3f} {st['search_mean']:12.3f}"
            f"   ({st['optimism']:+.3f})")
    if ref in val:
        for m in modes:
            if m == ref:
                continue
            keys = sorted(set(val[ref]) & set(val[m]))
            d = np.array([val[ref][k] - val[m][k] for k in keys], float)
            d = d[np.isfinite(d)]
            if d.size == 0:
                continue
            se = d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else np.nan
            pr = {"n_pairs": int(d.size), "mean_diff": float(d.mean()), "se": float(se) if np.isfinite(se) else None,
                  "t": float(d.mean() / se) if se and se > 0 else None, "wins": int((d > 0).sum())}
            out["paired"][m] = pr
            tstr = "" if pr["t"] is None else f"  t={pr['t']:.2f}"
            log(f"paired {ref} - {m}: {pr['mean_diff']:+.3f} (validated, n={pr['n_pairs']}){tstr}  "
                f"{ref} wins {pr['wins']}/{pr['n_pairs']}")
    return out


# ----------------------------------------------------------------------------
# synthetic benchmark
# ----------------------------------------------------------------------------
def make_synthetic_bench(n_partitions: int = 3, size: int = 80, ann_edges: Sequence[float] = (8.0, 30.0),
                         n_sources: int = 3, k_max: int = 30, n_valid: int = 6, n_top: int = 3,
                         forced_contrast: Optional[float] = None, calib_dir: Optional[str] = None,
                         log: Callable[[str], None] = lambda s: None) -> MakeRunner:
    """Return a ``make_runner`` for a benchmark on the synthetic partitions of :mod:`klip_tpe.synthetic`.

    The space has a ``bin``, ``filter``, ``angsep`` and ``k_klip`` per partition, ``k_klip`` on
    :func:`~klip_tpe.space.kgrid`, and the two-slot partition selection.  The objective is
    :class:`~klip_tpe.metrics.MawetPeakSNR` with clean subtraction.  The injection contrast is calibrated
    once with the default configuration, which puts the default's S/N in the calibration window and so
    below threshold, and forced for every run.  Grid mode searches the tied ``bin``, ``filter`` and
    ``k_klip`` only, as the IDL predecessor's baseline does.

    Parameters
    ----------
    n_partitions : int, default 3
        Number of partitions.
    size : int, default 80
        Side of the frames, in pixels.
    ann_edges : sequence of float, default (8.0, 30.0)
        Annulus edges, in pixels.
    n_sources : int, default 3
        Sources injected per evaluation.
    k_max : int, default 30
        Largest ``k_klip``.
    n_valid : int, default 6
        Validation trials per candidate.
    n_top : int, default 3
        Validation candidates.
    forced_contrast : float or None, default None
        The injection contrast, None to calibrate it.
    calib_dir : str or None, default None
        Directory of the calibration's run, None for a temporary one.
    log : callable, default a function that does nothing
        Function taking each log line, the runners' too.

    Returns
    -------
    make_runner : callable
        Function ``make_runner(mode, seed, run_dir)`` returning a :class:`Runner`, with the contrast every
        run takes as its ``contrast`` attribute.
    """
    from .metrics import MawetPeakSNR, Objective
    from .positions import PositionSampler
    from .reducer import PartitionedReducer
    from .contrast_grid import ContrastGridConfig
    from .runner import CalibrationConfig, RunConfig, ValidationConfig
    from .space import Param, SearchSpace, kgrid
    from .synthetic import make_synthetic_partitions

    pids = [f"n{i + 1}" for i in range(int(n_partitions))]

    def build(mode: str, seed: int, run_dir: str, forced: Optional[float]) -> Runner:
        parts = make_synthetic_partitions(pids, size=size)
        red = PartitionedReducer(parts)
        block = [Param("bin", 4, 32, kind="int", default=12),
                 Param("filter", 4, 20, kind="float", default=10.0),
                 Param("angsep", 0.2, 1.5, kind="float", default=0.5),
                 Param("k_klip", 1, k_max, kind="int", grid=kgrid(k_max), default=10)]
        sp = SearchSpace().replicate(block, pids).with_selection("two_slot")
        obj = Objective(MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm), clean_subtract=True)
        samp = PositionSampler(fwhm_as=red.fwhm * red.pxscale)
        link = ["bin", "filter"] if mode != "grid" else ["bin", "filter", "k_klip"]
        cfg = RunConfig(ann_edges=list(ann_edges), search_mode=mode, seed=int(seed), n_sources=n_sources,
                        link_params=link, grid_axes=[f"bin_{pids[0]}", f"filter_{pids[0]}", f"k_klip_{pids[0]}"],
                        calibration=CalibrationConfig(n_remeasure=2, forced=None if forced is None else [forced],
                                                      recal_budget=0 if forced is not None else 1),
                        validation=ValidationConfig(n_top=n_top, n_valid=n_valid), save_fits=False,
                        contrast_grid=ContrastGridConfig(mode="off"))     # a benchmark compares searches
        return Runner(red, sp, obj, samp, cfg, run_dir, log=log)

    contrast = forced_contrast
    if contrast is None:
        import tempfile
        d = calib_dir or tempfile.mkdtemp(prefix="klip_tpe_bench_calib_")
        r0 = build("tpe", 12345, d, None)
        contrast, kdef, info = r0.calibrate(0)
        log(f"[bench] calibrated injection contrast {contrast:.3e} (default-config S/N {info.get('snr')}, "
            f"k_default {kdef}); forcing it for every run")

    def make_runner(mode: str, seed: int, run_dir: str) -> Runner:
        return build(mode, seed, run_dir, contrast)

    make_runner.contrast = contrast          # type: ignore[attr-defined]
    return make_runner


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the synthetic benchmark from the command line, ``python -m klip_tpe.bench --synthetic``.

    It runs or completes a batch, logs its table (:func:`summarize_bench`) and draws its convergence
    curves into ``fig_bench_conv_<bench_tag>.png``.

    Parameters
    ----------
    argv : sequence of str or None, default None
        The arguments, None for ``sys.argv``.

    Returns
    -------
    code : int
        0.

    Raises
    ------
    SystemExit
        If ``--synthetic`` is not given, since the command line runs no other data.
    """
    ap = argparse.ArgumentParser(prog="python -m klip_tpe.bench",
                                 description="matched-budget benchmark of search modes (synthetic data)")
    ap.add_argument("--synthetic", action="store_true", help="benchmark on klip_tpe.synthetic partitions")
    ap.add_argument("--n_iter", type=int, default=60)
    ap.add_argument("--n_init", type=int, default=None, help="warm-up rows incl. seed (default: n_iter // 4)")
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (0..n-1)")
    ap.add_argument("--modes", default="tpe,random,grid")
    ap.add_argument("--out_dir", default="bench_synthetic")
    ap.add_argument("--tag", default=None, help="batch tag (default: new bench_YYYYMMDDHHMMSS)")
    ap.add_argument("--restart", action="store_true", help="resume/complete the batch in --out_dir/bench_tag.txt")
    ap.add_argument("--partitions", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if not a.synthetic:
        ap.error("only --synthetic is supported from the command line; call run_benchmark() for real data")
    log = (lambda s: None) if a.quiet else print
    if a.n_init is None:
        a.n_init = max(4, a.n_iter // 4)
    modes = tuple(m.strip() for m in a.modes.split(",") if m.strip())
    seeds = tuple(range(a.seeds))
    mk = make_synthetic_bench(n_partitions=a.partitions, log=log)
    if a.restart:
        rows = bench_restart(a.out_dir, mk, a.tag, modes, seeds, a.n_iter, a.n_init, log)
        tag = a.tag or _current_tag(a.out_dir)
    else:
        tag = a.tag or new_bench_tag()
        rows = run_benchmark(mk, modes, seeds, a.n_iter, a.n_init, tag, a.out_dir, log)
    print(f"injection contrast (forced, sub-threshold): {mk.contrast:.3e}")
    summarize_bench(a.out_dir, tag)
    curves = bench_convergence(summary_run_dirs(a.out_dir, tag))
    from .plots import plot_bench_convergence
    png = os.path.join(a.out_dir, f"fig_bench_conv_{tag}.png")
    plot_bench_convergence(curves, out_path=png, title=f"{tag}: running best search score (upward-biased)")
    print(f"convergence figure: {png}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
