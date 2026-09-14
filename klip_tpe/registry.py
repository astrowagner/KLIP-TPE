"""Which runs exist, which are actually running, and how to stop or resume one.

An optimization is a long-lived process that outlives the terminal it was started from,
and a campaign accumulates several of them.  Without a record, the only way to find out
what is on the machine is ``pgrep`` and reading command lines -- and a run whose parent
died leaves worker processes that nothing will ever reap or report.

So every run registers itself here when it starts: its directory, the exact argv that
created it, and its pid.  That single file is what makes ``klip-tpe runs`` able to say
what is going on, ``klip-tpe runs --stop`` able to clean up, and ``klip-tpe resume`` able
to work with no arguments at all -- the argv it needs is the one already on record.

The registry is advisory: it is never required to run, and a missing or unreadable file
degrades to "no runs known" rather than an error.  Liveness always comes from the
operating system and the run's own files, never from the registry, so a stale entry (a
machine that was rebooted, a directory that was deleted) can never masquerade as a
running job.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

__all__ = ["registry_path", "register", "known_runs", "processes_for", "stop_run",
           "run_state", "format_table", "records_for"]

STALE_AFTER = 900.0        # s without a write before a run with no process counts as stale


def registry_path() -> str:
    d = os.environ.get("KLIP_TPE_DATA") or os.path.join(os.path.expanduser("~"), ".klip_tpe")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return os.path.join(d, "runs.jsonl")


def register(run_dir: str, argv: Optional[List[str]] = None) -> None:
    """Record that this process is running ``run_dir``.  Never raises."""
    try:
        rec = {"run_dir": os.path.abspath(run_dir), "pid": os.getpid(),
               "argv": list(argv if argv is not None else sys.argv),
               "started": time.time(), "host": os.uname().nodename if hasattr(os, "uname") else ""}
        with open(registry_path(), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _read() -> List[Dict[str, Any]]:
    p = registry_path()
    if not os.path.exists(p):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except OSError:
        return []
    return out


def records_for(run_dir: str) -> List[Dict[str, Any]]:
    """Every entry recorded for ``run_dir``, newest first.

    The newest entry is not always the most useful one.  A resume that took the data root
    *from* this file records only what was typed -- ``klip-tpe resume --show`` -- so the
    next resume would inherit an entry with no root in it and stop, on a run that had been
    resuming all day.  Keeping the whole history lets a resume look back for an entry that
    carries what the reducer needs, while still preferring the newest one that does.
    """
    run_dir = os.path.abspath(run_dir)
    out = [r for r in _read() if r.get("run_dir") == run_dir]
    out.sort(key=lambda r: r.get("started", 0), reverse=True)
    return out


def _ps() -> List[tuple]:
    """``[(pid, command line), ...]`` for every process we can see.  Empty when ``ps``
    is unavailable, in which case liveness falls back to the run's own files."""
    try:
        r = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, args = line.partition(" ")
        try:
            out.append((int(pid), args.strip()))
        except ValueError:
            continue
    return out


#: command names that can mention a run directory without being one (a shell running a
#: heredoc, an editor, the very tools used to look for runs)
_NOT_A_RUN = {"bash", "sh", "zsh", "dash", "fish", "csh", "tcsh", "ps", "grep", "egrep",
              "pgrep", "pkill", "tail", "head", "less", "more", "cat", "vim", "vi", "nano",
              "emacs", "code", "open", "find", "rsync", "cp", "mv", "tar", "du", "ls"}


def _looks_like_a_run(args: str) -> bool:
    """A klip-tpe process, not something that merely mentions one."""
    if "klip-tpe" not in args and "klip_tpe" not in args:
        return False
    head = args.split(None, 1)[0] if args.strip() else ""
    return os.path.basename(head).split(".")[0].lower() not in _NOT_A_RUN


def processes_for(run_dir: str, table: Optional[List[tuple]] = None) -> List[int]:
    """Every pid whose command line names ``run_dir`` -- the parent and its workers alike,
    including workers orphaned by a parent that is already gone (a forked worker inherits
    the parent's command line, which is what makes them findable at all).

    Deliberately conservative: a shell, an editor or a ``grep`` that happens to mention the
    directory is not a run, and reporting one as running would block a legitimate resume.
    """
    run_dir = os.path.abspath(run_dir)
    base = os.path.basename(run_dir.rstrip("/"))
    me = os.getpid()
    hits = []
    for pid, args in (table if table is not None else _ps()):
        if pid == me or not (run_dir in args or (base and base in args)):
            continue
        if _looks_like_a_run(args):
            hits.append(pid)
    return sorted(hits)


def run_state(run_dir: str, table: Optional[List[tuple]] = None) -> Dict[str, Any]:
    """What this run is doing, from the operating system and the run's own files."""
    run_dir = os.path.abspath(run_dir)
    st: Dict[str, Any] = {"run_dir": run_dir, "exists": os.path.isdir(run_dir),
                          "pids": processes_for(run_dir, table), "n_eval": None,
                          "age": None, "annulus": None}
    res = os.path.join(run_dir, "results.jsonl")
    if os.path.exists(res):
        try:
            st["age"] = time.time() - os.path.getmtime(res)
            with open(res, "rb") as f:
                st["n_eval"] = sum(1 for _ in f)
                f.seek(0, os.SEEK_END)
                back = min(8192, f.tell())
                f.seek(-back, os.SEEK_END)
                for line in reversed(f.read().splitlines()):
                    try:
                        st["annulus"] = json.loads(line).get("annulus")
                        break
                    except Exception:
                        continue
        except OSError:
            pass
    # The heartbeat is the liveness clock when the run has one: it is stamped every few
    # seconds from the run's own thread, whereas ``results.jsonl`` advances once per
    # completed evaluation.  Judging by the latter alone called a run "stalled" whenever a
    # single reduction ran long -- which on a shared machine is often (6 s to 479 s within
    # one RX J0534 run) -- and could not tell that from a process that had died.
    from .heartbeat import read as _hb_read
    hb = _hb_read(run_dir)
    st["heartbeat"] = hb
    st["stage"] = (hb or {}).get("stage")
    st["stage_s"] = (hb or {}).get("stage_s")
    st["hb_age"] = (hb or {}).get("age")
    # state: running (processes + a fresh heartbeat, or recent writes)
    #      | stalled (processes, but the heartbeat has stopped / nothing written lately)
    #      | orphaned (processes, but nothing has been written for a long time)
    #      | finished/stopped (no processes)
    n, age = len(st["pids"]), st["age"]
    if n and hb is not None and hb.get("fresh"):
        st["state"] = "running"
    elif n and hb is not None:
        # the run has a heartbeat and it has gone quiet: the process is there but not ticking
        st["state"] = "stalled" if (age is None or age < STALE_AFTER) else "orphaned"
    elif n and age is not None and age < 120:
        st["state"] = "running"
    elif n and (age is None or age < STALE_AFTER):
        st["state"] = "stalled"
    elif n:
        st["state"] = "orphaned"
    else:
        st["state"] = "stopped"
    st["done"] = os.path.exists(os.path.join(run_dir, "final_results.json"))
    if st["done"] and not n:
        st["state"] = "finished"
    return st


def known_runs(root: Optional[str] = None, include_unregistered: bool = True) -> List[Dict[str, Any]]:
    """Every run we know about, newest first: the registry, plus any run directory found
    under ``root``, each annotated with :func:`run_state`."""
    seen: Dict[str, Dict[str, Any]] = {}
    for rec in _read():
        d = rec.get("run_dir")
        if not d:
            continue
        prev = seen.get(d)
        if prev is None or rec.get("started", 0) >= prev.get("started", 0):
            seen[d] = rec
    if include_unregistered and root:
        import glob
        for pat in ("comb/opt/run_*", "runs/run_*", "run_*"):
            for d in glob.glob(os.path.join(root, pat)):
                if os.path.isdir(d):
                    seen.setdefault(os.path.abspath(d), {"run_dir": os.path.abspath(d)})
    table = _ps()
    out = []
    for d, rec in seen.items():
        st = run_state(d, table)
        st["argv"] = rec.get("argv")
        st["started"] = rec.get("started")
        st["registered"] = "argv" in rec
        out.append(st)
    out.sort(key=lambda s: (s.get("age") if s.get("age") is not None else 9e18))
    return out


def stop_run(run_dir: str, grace: float = 10.0, log=print) -> int:
    """SIGTERM every process of ``run_dir``, then SIGKILL whatever is left.  Returns the
    number of processes signalled."""
    pids = processes_for(run_dir)
    if not pids:
        log(f"  {os.path.basename(run_dir)}: nothing running")
        return 0
    log(f"  {os.path.basename(run_dir)}: stopping {len(pids)} process(es) {pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    t0 = time.time()
    while time.time() - t0 < grace:
        alive = [p for p in pids if _alive(p)]
        if not alive:
            return len(pids)
        time.sleep(0.3)
    for pid in [p for p in pids if _alive(p)]:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return len(pids)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _age(s: Optional[float]) -> str:
    if s is None:
        return "--"
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.0f}m"
    if s < 172800:
        return f"{s / 3600:.1f}h"
    return f"{s / 86400:.1f}d"


def format_table(runs: List[Dict[str, Any]]) -> str:
    """One line per run: state, evaluations, how long since it last wrote, processes."""
    if not runs:
        return ("no runs known.  A run registers itself when it starts; runs from before this\n"
                "version, or on another machine, are found with  klip-tpe runs --root <data root>")
    w = max([len(os.path.basename(r["run_dir"])) for r in runs] + [8])
    head = (f"{'run':<{w}}  {'state':<9} {'evals':>7}  {'last write':>10}  {'procs':>5}  "
            f"{'ann':>3}  doing now")
    lines = [head, "-" * len(head)]
    for r in runs:
        name = os.path.basename(r["run_dir"])
        ann = "" if r.get("annulus") is None else str(r["annulus"] + 1)
        # what it is doing now, and for how long -- the answer to "is this one stuck?"
        from .heartbeat import describe as _hb_describe
        doing = _hb_describe(r.get("heartbeat")) if r["state"] in ("running", "stalled") else ""
        lines.append(f"{name:<{w}}  {r['state']:<9} "
                     f"{('' if r['n_eval'] is None else r['n_eval']):>7}  "
                     f"{_age(r['age']):>10}  {len(r['pids']):>5}  {ann:>3}  {doing}")
    quiet = [r for r in runs if r["state"] == "stalled" and (r.get("heartbeat") or {}).get("age") is not None]
    if quiet:
        lines.append("")
        lines.append(f"{len(quiet)} run(s) have processes but stopped stamping heartbeat.json -- wedged "
                     f"rather than slow (a slow reduction keeps stamping).")
        lines.append("  klip-tpe runs --stop-stale        then  klip-tpe resume  (exact, from the last checkpoint)")
    bad = [r for r in runs if r["state"] == "orphaned"]
    if bad:
        lines.append("")
        lines.append(f"{len(bad)} run(s) have processes but stopped writing long ago -- almost certainly "
                     f"workers orphaned by a parent that died.")
        lines.append("  klip-tpe runs --stop-stale        to clean them up")
    return "\n".join(lines)
