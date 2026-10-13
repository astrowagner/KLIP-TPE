"""Where the run's main thread spends its time, sampled inside the run itself.

A pipelined search keeps several trials in flight, and once the workers keep up, the main
thread sets the pace.  Scoring, the TPE proposal, the result files, the checkpoint and the
display all run there, one after another.  Setting ``KLIP_TPE_PROFILE_MAIN=S`` in the
environment of a run (or a resume) starts a daemon thread that looks at the main thread's
stack every few milliseconds.  Every ``S`` seconds it rewrites ``main_thread_profile.txt``
in the run directory with two tables, each for the last ``S`` seconds and since the start:

* inclusive: the share of the time each function was on the stack;
* where the stack ended: the innermost function, with waits (a lock, a queue, ``select``)
  counted together as ``(waiting)``, so the idle share of the thread is one number.

Each sample counts for the time since the one before.  The sampling thread needs the GIL
to look, and gets it less often while the main thread runs Python than while it waits, so
equal weights would overstate the waits (a half-busy loop read as 22% busy).

It needs no extra package and costs about one stack walk per sample.  Without the variable
nothing starts.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Callable, Optional

__all__ = ["MainThreadSampler", "sampler_from_env", "ENV"]

ENV = "KLIP_TPE_PROFILE_MAIN"
FILENAME = "main_thread_profile.txt"

#: innermost Python frames that mean the thread is waiting, not working
_WAITS = {("threading.py", "wait"), ("threading.py", "_wait_for_tstate_lock"), ("threading.py", "join"),
          ("queue.py", "get"), ("selectors.py", "select"), ("connection.py", "_poll"),
          ("connection.py", "wait"), ("connection.py", "_recv"), ("_base.py", "result"),
          ("_base.py", "wait"), ("_base.py", "as_completed")}


def _where(code) -> str:
    """``package/module.py:function`` for a code object."""
    path = code.co_filename
    parts = path.replace("\\", "/").split("/")
    short = "/".join(parts[-2:]) if len(parts) > 1 and parts[-2] in ("klip_tpe", "instruments") else parts[-1]
    return f"{short}:{code.co_name}"


class MainThreadSampler:
    """Sample the main thread's stack every ``period_s`` seconds and write the report to
    ``path`` every ``every_s`` seconds.  ``status`` returns a phrase for the report header
    (where the run is, e.g. the evaluation count)."""

    def __init__(self, path: str, every_s: float, period_s: float = 0.005,
                 status: Optional[Callable[[], str]] = None):
        self.path, self.every_s, self.period_s = path, max(float(every_s), 1.0), float(period_s)
        self.status = status
        self.main = threading.main_thread().ident
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self.total = {"n": 0, "w": 0.0, "incl": {}, "leaf": {}}
        self.window = {"n": 0, "w": 0.0, "incl": {}, "leaf": {}}
        self.t0 = self.t_win = time.time()
        self.win_status = self._status()

    def _status(self) -> str:
        try:
            return self.status() if self.status else ""
        except Exception:
            return ""

    def start(self) -> "MainThreadSampler":
        if self._th is None:
            self._th = threading.Thread(target=self._run, name="klip-tpe-profile", daemon=True)
            self._th.start()
        return self

    def stop(self) -> None:
        """Stop sampling and write the report one last time."""
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=5.0)
            self._th = None
        self.write()

    def sample(self, weight: float = 1.0) -> None:
        """Record the main thread's stack as it is now, for ``weight`` seconds."""
        f = sys._current_frames().get(self.main)
        if f is None:
            return
        leaf = f.f_code
        seen = set()
        while f is not None:
            seen.add(_where(f.f_code))
            f = f.f_back
        name = os.path.basename(leaf.co_filename)
        lk = "(waiting)" if (name, leaf.co_name) in _WAITS else _where(leaf)
        for acc in (self.total, self.window):
            acc["n"] += 1
            acc["w"] += weight
            for k in seen:
                acc["incl"][k] = acc["incl"].get(k, 0.0) + weight
            acc["leaf"][lk] = acc["leaf"].get(lk, 0.0) + weight

    def _run(self) -> None:
        t_prev = time.perf_counter()
        while not self._stop.wait(self.period_s):
            try:
                now = time.perf_counter()
                self.sample(now - t_prev)
                t_prev = now
                if time.time() - self.t_win >= self.every_s:
                    self.write()
                    self.window = {"n": 0, "w": 0.0, "incl": {}, "leaf": {}}
                    self.t_win, self.win_status = time.time(), self._status()
            except Exception:
                pass                       # a profile never stops a run

    def write(self, top: int = 40) -> None:
        win, tot = self.window, self.total
        now = time.time()

        def table(key):
            rows = sorted(tot[key], key=lambda k: (-win[key].get(k, 0), -tot[key][k], k))[:top]
            return [f"  {100.0 * win[key].get(k, 0.0) / max(win['w'], 1e-9):6.1f}%  "
                    f"{100.0 * tot[key][k] / max(tot['w'], 1e-9):6.1f}%   {k}" for k in rows]
        lines = [f"# main-thread profile, written {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 f"# last {now - self.t_win:.0f} s: {win['n']} samples"
                 + (f"  ({self.win_status} -> {self._status()})" if self.status else ""),
                 f"# since the start, {now - self.t0:.0f} s ago: {tot['n']} samples",
                 "#", "# inclusive: share of the time with the function on the stack",
                 "#    last   total   function"] + table("incl") + [
                 "#", "# where the stack ended (waits on a lock, queue or select count as (waiting))",
                 "#    last   total   function"] + table("leaf")
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp, self.path)
        except OSError:
            pass


def sampler_from_env(run_dir: str, status: Optional[Callable[[], str]] = None,
                     log: Optional[Callable[[str], None]] = None) -> Optional[MainThreadSampler]:
    """A started sampler when ``KLIP_TPE_PROFILE_MAIN`` holds a number of seconds, else None."""
    v = os.environ.get(ENV, "").strip()
    if not v:
        return None
    try:
        every = float(v)
    except ValueError:
        if log:
            log(f"  {ENV}={v!r} is not a number of seconds: no profile")
        return None
    if every <= 0:
        return None
    s = MainThreadSampler(os.path.join(run_dir, FILENAME), every, status=status).start()
    if log:
        log(f"  profiling the main thread: {FILENAME} every {every:g} s ({ENV})")
    return s
