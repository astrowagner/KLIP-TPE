"""Proof of life for a long run: ``heartbeat.json`` in the run directory.

A search runs for days, on a machine doing other things, and the only record of it was
``results.jsonl`` -- which advances once per *completed* evaluation.  That is a poor clock.
An evaluation on a busy machine takes anything from six seconds to eight minutes (measured:
a 147-D RX J0534 evaluation ran 479 s while its neighbours took 6 s, with four searches and
a benchmark sharing the cores), so a run that has written nothing for an hour is either
grinding through one slow reduction or dead, and the files cannot tell the two apart.  The
question "is it stuck or is it gone?" then has no answer short of hunting for the pid.

So a daemon thread stamps this file every few seconds for as long as the process lives:

* file fresh          -> the process is alive, whatever the rest of the directory shows
* file stale, pid up  -> alive but wedged at the OS level (swapping, stopped, deadlocked)
* file stale, no pid  -> it died; ``klip-tpe resume`` picks up from the last checkpoint

The thread is what makes the first line true.  Running this from the main loop instead
would only report where the main thread last *reached*, which is exactly what goes quiet
when a reduction is slow -- the thing we are trying to distinguish.

Alongside the clock it carries what the run is doing (``stage``, and when that stage
started), so a stale file also says what it was doing when it stopped, and a fresh one says
whether an hour in a single stage is the reduction or the display.  ``stall_after`` lets the
run state a duration past which the current stage is worth a line in the log -- scaled to
what that run's own evaluations cost, since no fixed number fits both a 6 s benchmark and a
479 s night stack.

Best-effort throughout: every write is guarded, the file is rewritten atomically so a reader
never sees half of one, and nothing here can interrupt a search.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

__all__ = ["Heartbeat", "FILENAME", "PERIOD", "read", "describe"]

#: file written in the run directory
FILENAME = "heartbeat.json"

#: seconds between stamps.  Small enough that "fresh" means seconds, large enough that a
#: run on a network filesystem is not writing continuously.
PERIOD = 5.0

#: a heartbeat older than this is not proof of life (six missed stamps)
FRESH_FOR = 30.0


class Heartbeat:
    """Stamps ``run_dir/heartbeat.json`` every ``period`` seconds from a daemon thread.

    ``stage()`` names what the run is doing and restarts the stage clock; ``update()``
    revises the details without restarting it.  Neither blocks and neither raises.
    """

    def __init__(self, run_dir: str, log=None, period: float = PERIOD):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, FILENAME)
        self.period = max(float(period), 0.2)
        self.log = log or (lambda s: None)
        self._lock = threading.Lock()
        now = time.time()
        self._state: Dict[str, Any] = {
            "pid": os.getpid(), "started": now, "stage": "starting", "stage_t0": now,
            "annulus": None, "eval": None, "n_eval": None, "note": None,
        }
        self._stall_after: Optional[float] = None
        self._warned: float = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ---------------------------------------------------------------------
    def start(self) -> "Heartbeat":
        """Begin stamping.  Idempotent; safe to call on a resumed run."""
        if self._thread is not None:
            return self
        self.write()
        t = threading.Thread(target=self._loop, name="klip-heartbeat", daemon=True)
        self._thread = t
        try:
            t.start()
        except Exception:                                    # pragma: no cover - no threads
            self._thread = None
        return self

    def stop(self, stage: Optional[str] = None) -> None:
        """Last stamp, then no more.  The file stays: it is the record of how this run
        ended, and a reader distinguishes it from a live one by the stage and the age.
        ``stage=None`` keeps whatever stage the run stopped in -- which is the useful thing
        to read when it stopped because of a Ctrl-C or an exception."""
        self._stop.set()
        if stage is None:
            self.write()
        else:
            self.stage(stage)
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=self.period + 1.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.period):
            self.write()
            self._check_stall()

    # -- what the run is doing ---------------------------------------------------------
    def stage(self, stage: str, **kw: Any) -> None:
        """Name the current stage and restart its clock (and so its stall warning)."""
        with self._lock:
            self._state["stage"] = str(stage)
            self._state["stage_t0"] = time.time()
            self._state.update(kw)
            self._warned = 0.0
        self.write()

    def update(self, **kw: Any) -> None:
        """Revise the details without restarting the stage clock."""
        with self._lock:
            self._state.update(kw)

    def stall_after(self, seconds: Optional[float]) -> None:
        """Log a line once the current stage passes ``seconds`` (then at each doubling).
        ``None`` (or a non-positive number) turns the warning off.  Scale it to the run's
        own evaluation cost -- see :attr:`klip_tpe.runner.Runner.STALL_FACTOR`."""
        with self._lock:
            s = None if seconds is None else float(seconds)
            self._stall_after = s if (s is not None and s > 0) else None

    # -- the file ----------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            st = dict(self._state)
        st["t"] = time.time()
        st["stage_s"] = st["t"] - st.get("stage_t0", st["t"])
        st["elapsed_s"] = st["t"] - st.get("started", st["t"])
        return st

    def write(self) -> None:
        """Atomic, guarded.  A reader never sees a partial file, and a full disk or a
        vanished directory never touches the search."""
        st = self.snapshot()
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(st, f)
                f.write("\n")
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def _check_stall(self) -> None:
        with self._lock:
            after, warned = self._stall_after, self._warned
            stage, t0 = self._state.get("stage"), self._state.get("stage_t0") or time.time()
        if not after:
            return
        s = time.time() - t0
        limit = after if not warned else warned * 2.0
        if s < limit:
            return
        with self._lock:
            self._warned = s
        self.log(f"  still in '{stage}' after {_dur(s)} (expected ~{_dur(after / 8.0)}); the run is "
                 f"alive -- heartbeat.json is being stamped. Fewer concurrent searches, or fewer "
                 f"workers each, is usually the cause; Ctrl-C is safe and  resume  is exact.")


def _dur(s: float) -> str:
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.0f}m"
    return f"{s / 3600:.1f}h"


def read(run_dir: str) -> Optional[Dict[str, Any]]:
    """The run's heartbeat, with ``age`` and ``fresh`` added, or ``None`` if there is none
    (a run from before this version, or one that never got as far as writing it)."""
    p = os.path.join(run_dir, FILENAME)
    try:
        with open(p) as f:
            hb = json.load(f)
        if not isinstance(hb, dict):
            return None
    except Exception:
        return None
    t = hb.get("t")
    if not isinstance(t, (int, float)):
        try:
            t = os.path.getmtime(p)
        except OSError:
            return None
    hb["age"] = max(time.time() - float(t), 0.0)
    hb["fresh"] = hb["age"] < FRESH_FOR
    return hb


def describe(hb: Optional[Dict[str, Any]]) -> str:
    """One phrase for a table or a log line."""
    if not hb:
        return ""
    stage = hb.get("stage") or "?"
    s = hb.get("stage_s")
    if isinstance(s, (int, float)):
        s = float(s) + (hb.get("age") or 0.0)
        return f"{stage} ({_dur(s)})"
    return str(stage)
