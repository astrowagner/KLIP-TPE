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
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

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
        from .durable import append_line
        append_line(registry_path(), json.dumps(rec))    # on a line of its own, whatever a crash left
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


class Proc(NamedTuple):
    """One process as the liveness scan sees it.

    Attributes
    ----------
    pid : int
        Its id.
    args : str
        Its command line as one string, as ``ps`` shows it.
    argv : tuple of str or None
        Its arguments exactly, from ``/proc/<pid>/cmdline`` or psutil, or split from ``args``.
    cwd : str or None
        Its working directory, None where it cannot be read.
    ppid : int or None
        Its parent's id.
    """

    pid: int
    args: str
    argv: Optional[Tuple[str, ...]] = None
    cwd: Optional[str] = None
    ppid: Optional[int] = None


def _proc_linux(pid: int) -> Optional[Proc]:
    """Read one process from ``/proc``.

    Parameters
    ----------
    pid : int
        The process.

    Returns
    -------
    proc : Proc or None
        The process, None when it is gone or unreadable.
    """
    base = f"/proc/{pid}"
    try:
        with open(base + "/cmdline", "rb") as f:
            raw = f.read()
    except OSError:
        return None
    argv = tuple(a.decode("utf-8", "replace") for a in raw.split(b"\0") if a)
    if not argv:
        return None
    try:
        cwd = os.readlink(base + "/cwd")
    except OSError:
        cwd = None
    ppid = None
    try:
        with open(base + "/stat") as f:
            stat = f.read()
        ppid = int(stat[stat.rindex(")") + 2:].split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return Proc(pid, " ".join(argv), argv, cwd, ppid)


def _ps() -> List[Proc]:
    """Return every process we can see, with its arguments, working directory and parent.

    ``/proc`` gives them on Linux and psutil elsewhere where it is installed.  Otherwise ``ps`` gives
    the command line as one string, split on spaces, with no working directory, so an argument
    relative to one cannot name a run.  Empty when nothing can list the processes, in which case
    liveness falls back to the run's own files and its lock.

    Returns
    -------
    procs : list of Proc
        The processes.
    """
    if os.path.isdir("/proc/self"):
        out = []
        for name in os.listdir("/proc"):
            if name.isdigit():
                pr = _proc_linux(int(name))
                if pr is not None:
                    out.append(pr)
        return out
    try:
        import psutil                                         # type: ignore
        out = []
        for q in psutil.process_iter(["pid", "ppid", "cmdline", "cwd"]):
            info = q.info
            argv = tuple(info.get("cmdline") or ())
            if argv:
                out.append(Proc(int(info["pid"]), " ".join(argv), argv, info.get("cwd"), info.get("ppid")))
        return out
    except Exception:
        pass
    try:
        r = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    out = []
    for line in r.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            out.append(Proc(int(parts[0]), parts[2].strip(), tuple(parts[2].split()), None, int(parts[1])))
        except ValueError:
            continue
    return out


def _as_proc(entry) -> Proc:
    """Return a process table's entry as a :class:`Proc`.

    Parameters
    ----------
    entry : Proc or tuple
        A :class:`Proc`, or ``(pid, command line)`` as tables built by hand give it.

    Returns
    -------
    proc : Proc
        The entry, its arguments split from the command line where it has none.
    """
    if isinstance(entry, Proc):
        return entry if entry.argv is not None else entry._replace(argv=tuple(entry.args.split()))
    pid, args = entry[0], entry[1]
    return Proc(int(pid), str(args), tuple(str(args).split()), *tuple(entry[2:4]))


#: interpreters that can run the klip-tpe script or module
_PYTHONS = ("python", "pypy")


def _entry_point(argv: Sequence[str]) -> Optional[int]:
    """Tell whether a command line runs klip-tpe, and where its own arguments start.

    It does when its program is ``klip-tpe``, or an interpreter running the ``klip-tpe`` script or
    ``-m klip_tpe`` (or a module of it).  A shell, ``tee``, an editor or a script of one's own that
    names a run is not klip-tpe, whatever folder it works in.

    Parameters
    ----------
    argv : sequence of str
        The command line.

    Returns
    -------
    start : int or None
        Index of the first argument after the program, the script or the module, None when the
        command line does not run klip-tpe.
    """
    if not argv:
        return None
    prog = os.path.basename(argv[0])
    if prog in ("klip-tpe", "klip-tpe.exe"):
        return 1
    if not prog.lower().startswith(_PYTHONS):
        return None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "-m":
            mod = argv[i + 1] if i + 1 < len(argv) else ""
            return i + 2 if (mod == "klip_tpe" or mod.startswith("klip_tpe.")) else None
        if a.startswith("-m") and len(a) > 2:
            mod = a[2:]
            return i + 1 if (mod == "klip_tpe" or mod.startswith("klip_tpe.")) else None
        if a in ("-c",):
            return None
        if a.startswith("-"):
            i += 1
            continue
        return i + 1 if os.path.basename(a) in ("klip-tpe", "klip-tpe.py") else None
    return None


def _looks_like_a_run(args: str) -> bool:
    """Tell whether a command line, as one string, runs klip-tpe (:func:`_entry_point`).

    Parameters
    ----------
    args : str
        The command line.

    Returns
    -------
    is_run : bool
        True for a klip-tpe process.
    """
    return _entry_point(tuple(str(args).split())) is not None


def _names_run(argv: Sequence[str], start: int, cwd: Optional[str], run_dir: str) -> bool:
    """Tell whether one of a command's arguments resolves to a run directory.

    Parameters
    ----------
    argv : sequence of str
        The command line.
    start : int
        Index of its first own argument (:func:`_entry_point`).
    cwd : str or None
        The process's working directory, against which a relative argument resolves.  None leaves
        relative arguments out.
    run_dir : str
        The run directory, absolute.

    Returns
    -------
    names : bool
        True when an argument, or the value of an ``--option=value``, is the run directory.
    """
    want = os.path.normpath(run_dir)
    for a in argv[start:]:
        for v in (a, a.split("=", 1)[1] if a.startswith("-") and "=" in a else None):
            if not v or v.startswith("-"):
                continue
            if os.path.isabs(v):
                path = v
            elif cwd:
                path = os.path.join(cwd, v)
            else:
                continue
            if os.path.normpath(path) == want:
                return True
    return False


def _recorded_pids(run_dir: str) -> Dict[int, Tuple[str, ...]]:
    """Return the pids the registry recorded as running a run, with the command lines they recorded.

    Parameters
    ----------
    run_dir : str
        The run directory, absolute.

    Returns
    -------
    pids : dict of int to tuple of str
        Each recorded pid and its recorded ``argv``.
    """
    out: Dict[int, Tuple[str, ...]] = {}
    for rec in records_for(run_dir):
        try:
            out.setdefault(int(rec["pid"]), tuple(str(a) for a in rec.get("argv") or ()))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def processes_for(run_dir: str, table: Optional[List[tuple]] = None) -> List[int]:
    """Return every pid running a run, the parent and its workers alike.

    A process counts when it runs klip-tpe (:func:`_entry_point`) and one of its arguments
    resolves to the run directory, relative ones against the process's working directory, or when
    the registry recorded its pid for the run and its command line still ends with the one recorded
    (a run a script started, or a bare ``klip-tpe resume``).  A process whose parent counts counts
    too, the workers a run forks.  A forked worker inherits the parent's command line, so an orphaned
    worker of a run named on its command line is found as well.

    Parameters
    ----------
    run_dir : str
        The run directory.
    table : list or None, default None
        The processes, :class:`Proc` or ``(pid, command line)`` entries.  None lists them
        (:func:`_ps`).

    Returns
    -------
    pids : list of int
        The pids, sorted.

    Notes
    -----
    Until 2026-11-25 a process counted when its command line held the run directory or only its
    name, held ``klip-tpe`` or ``klip_tpe``, and its program was not on a list of shells and tools.
    A ``tee -a .../klip_tpe/run_a.log`` in the resume's own pipeline, an analysis script given the
    run, and another folder's run of the same name all counted, so a resume was refused and
    ``runs --stop`` signaled them (a note from the NOMIC archive session, 2026-11-25).
    """
    run_dir = os.path.normpath(os.path.abspath(run_dir))
    me = os.getpid()
    procs = [_as_proc(e) for e in (table if table is not None else _ps())]
    recorded = _recorded_pids(run_dir)
    hit = set()
    for pr in procs:
        if pr.pid == me:
            continue
        argv = pr.argv or ()
        start = _entry_point(argv)
        if start is not None and _names_run(argv, start, pr.cwd, run_dir):
            hit.add(pr.pid)
            continue
        rec = recorded.get(pr.pid)
        if rec and len(argv) >= len(rec) and tuple(argv[len(argv) - len(rec):]) == rec:
            hit.add(pr.pid)
    # the workers of a process that counts, whatever their own command line
    parent = {pr.pid: pr.ppid for pr in procs}
    for pr in procs:
        seen, q = set(), pr.ppid
        while q and q not in seen and pr.pid != me:
            if q in hit:
                hit.add(pr.pid)
                break
            seen.add(q)
            q = parent.get(q)
    hit.discard(me)
    return sorted(hit)


LOCK_NAME = "run.lock"


class RunLock:
    """The exclusive ``flock`` that makes a process the one writer of a run directory.

    :meth:`Runner.run <klip_tpe.runner.Runner.run>` takes it before it writes and holds it to its end.
    The kernel drops it when the last process holding its descriptor ends, a crash included, so it
    never goes stale.  Workers forked while it is held inherit it.  Without ``fcntl``, as on
    Windows, it takes nothing and never refuses.

    Parameters
    ----------
    run_dir : str
        The run directory.

    Attributes
    ----------
    path : str
        The lock file, ``run.lock`` in the run directory.
    held : bool
        Whether this object holds the lock.
    """

    def __init__(self, run_dir: str):
        self.path = os.path.join(os.path.abspath(run_dir), LOCK_NAME)
        self._fd: Optional[int] = None
        self.held = False

    def acquire(self, wait: float = 0.5) -> bool:
        """Take the lock, and write this process's pid into the file.

        It tries for up to ``wait`` seconds, so that a :func:`lock_holder` probe, which holds a shared
        lock for an instant, never makes a starting run look refused.

        Parameters
        ----------
        wait : float, default 0.5
            Seconds to keep trying while another process holds it.

        Returns
        -------
        taken : bool
            True when this process now holds it, False when another holds it.  True without
            ``fcntl`` or where the file cannot be made, since the lock is a safeguard and never
            stands in the way of a run.
        """
        try:
            import fcntl
        except ImportError:                                   # pragma: no cover - Windows
            return True
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            return True
        deadline = time.monotonic() + max(float(wait), 0.0)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return False
                time.sleep(0.02)
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
        except OSError:
            pass
        self._fd, self.held = fd, True
        return True

    def release(self) -> None:
        """Drop the lock, when this object holds it."""
        if self._fd is None:
            return
        try:
            import fcntl
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except Exception:                                     # pragma: no cover - defensive
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd, self.held = None, False


def lock_holder(run_dir: str) -> Optional[int]:
    """Tell whether another process holds a run's lock (:class:`RunLock`), and which.

    It only reads.  The file is opened read-only and tried with a shared lock, which fails while a
    writer holds the exclusive one, so the commands that read a run (``rereduce``, ``render`` and
    the others that check it) change nothing in its directory.

    Parameters
    ----------
    run_dir : str
        The run directory.

    Returns
    -------
    pid : int or None
        The pid the holder wrote, 0 when it is held and the file names none, None when nobody else
        holds it, the run has no lock file, or ``fcntl`` is missing.
    """
    path = os.path.join(os.path.abspath(run_dir), LOCK_NAME)
    try:
        import fcntl
        fd = os.open(path, os.O_RDONLY)
    except (ImportError, OSError):                            # no lock file, or Windows
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            pass                                              # a writer holds it
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return None
        try:
            pid = int((os.read(fd, 64).decode(errors="replace").strip() or "0").split()[0])
        except (OSError, ValueError):
            return 0
    finally:
        os.close(fd)
    # a lock this process holds, through another descriptor, is no other process's
    return None if pid == os.getpid() else pid


def runs_named(name: str, runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the runs a ``--stop`` argument names: the run at a path, or every run of a bare name.

    Parameters
    ----------
    name : str
        A run directory, absolute or relative, or a run's name alone.
    runs : sequence of dict
        The runs on record (:func:`known_runs`).

    Returns
    -------
    runs : list of dict
        The run whose directory the argument is, or the runs whose directory has that name, which
        can be several in different folders.
    """
    n = str(name).rstrip("/")
    if os.sep in n or os.path.isdir(n):
        want = os.path.normpath(os.path.abspath(n))
        return [r for r in runs if os.path.normpath(r["run_dir"]) == want]
    return [r for r in runs if os.path.basename(os.path.normpath(r["run_dir"])) == n]


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
                # complete lines only: a crash can leave half of one at the end
                st["n_eval"] = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))
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
    number of processes signaled."""
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
