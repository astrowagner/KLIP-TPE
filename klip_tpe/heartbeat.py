"""Proof of life for a long run, ``heartbeat.json`` in the run directory.

A search runs for days on a machine doing other things, and its only record was ``results.jsonl``,
which advances once per finished evaluation.  That is a poor clock.  An evaluation on a busy machine
takes anything from six seconds to eight minutes (a 147-D RX J0534 evaluation ran 479 s while its
neighbors took 6 s, with four searches and a benchmark sharing the cores).  A run that has written
nothing for an hour is then either grinding through one slow reduction or dead, and its files cannot
tell the two apart.

So a daemon thread stamps this file every few seconds for as long as the process lives:

* a fresh file means the process is alive, whatever the rest of the directory shows,
* a stale file with the process still up means it is alive but stuck at the system level, swapping,
  stopped or deadlocked,
* a stale file without the process means it died, and ``klip-tpe resume`` carries on from the last
  checkpoint.

The thread is what makes the first true.  Stamps from the main loop would report only where the main
thread last reached, which is what goes quiet when a reduction is slow.

Beside the clock the file carries what the run is doing (``stage``, and when that stage started), so a
stale file also says where the run stopped, and a fresh one whether an hour in one stage is the
reduction or the display.  ``stall_after`` lets the run give a duration past which its stage is worth a
line in the log, scaled to what its own evaluations cost, since no fixed number fits both a 6 s
benchmark and a 479 s night stack.

Every write is guarded and atomic, so a reader never sees half a file, and nothing here can interrupt a
search.
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
    """Stamp ``heartbeat.json`` in a run directory from a daemon thread.

    :meth:`stage` names what the run is doing and restarts the stage's clock, and :meth:`update` changes
    the details without restarting it.  Neither blocks and neither raises.

    Parameters
    ----------
    run_dir : str
        The run's directory.
    log : callable or None, default None
        Function that takes a line for the run's log, for the stall warning.
    period : float, default 5.0
        Seconds between two stamps, at least 0.2.
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
        """Start stamping, once, a resumed run's heartbeat included.

        Returns
        -------
        hb : Heartbeat
            This heartbeat.
        """
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
        """Stamp a last time and stop.

        The file stays as the record of how the run ended, and a reader tells it from a live one by its stage
        and its age.

        Parameters
        ----------
        stage : str or None, default None
            The last stage, None to keep the stage the run stopped in, which is the one to read after a
            Ctrl-C or an exception.
        """
        self._stop.set()
        if stage is None:
            self.write()
        else:
            self.stage(stage)
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=self.period + 1.0)

    def _loop(self) -> None:
        """Stamp the file and look for a stall every ``period`` seconds until stopped.
        """
        while not self._stop.wait(self.period):
            self.write()
            self._check_stall()

    # -- what the run is doing ---------------------------------------------------------
    def stage(self, stage: str, write: bool = True, **kw: Any) -> None:
        """Name the current stage and restart its clock, and with it the stall warning.

        Parameters
        ----------
        stage : str
            What the run is doing.
        write : bool, default True
            Whether to write the file at once.  False leaves it to the thread's next stamp, at most
            ``period`` later, while the stall check sees the new stage at once.
        kw : dict
            Other details to record, such as ``annulus`` and ``eval``.

        Notes
        -----
        ``write=False`` is for stages that change every step, one per evaluation or validation trial.
        Written each time, a fast run of about 1 s evaluations rewrote the file faster than a synced folder
        uploads it, and Dropbox made a "conflicted copy" every few seconds (115 in 20 minutes on H2K,
        2026-09-23).  Changes of phase still write at once.
        """
        with self._lock:
            self._state["stage"] = str(stage)
            self._state["stage_t0"] = time.time()
            self._state.update(kw)
            self._warned = 0.0
        if write:
            self.write()

    def update(self, **kw: Any) -> None:
        """Change the details without restarting the stage's clock.

        Parameters
        ----------
        kw : dict
            The details.
        """
        with self._lock:
            self._state.update(kw)

    def stall_after(self, seconds: Optional[float]) -> None:
        """Log a line once the current stage passes a duration, then at each doubling.

        Parameters
        ----------
        seconds : float or None
            The duration, None or not positive to turn the warning off.  Scale it to the run's own
            evaluation cost (:attr:`klip_tpe.runner.Runner.STALL_FACTOR`).
        """
        with self._lock:
            s = None if seconds is None else float(seconds)
            self._stall_after = s if (s is not None and s > 0) else None

    # -- the file ----------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Return the state as the file holds it.

        Returns
        -------
        state : dict
            The state, with ``t``, ``stage_s`` (seconds in the stage) and ``elapsed_s`` (seconds since the
            start) added.
        """
        with self._lock:
            st = dict(self._state)
        st["t"] = time.time()
        st["stage_s"] = st["t"] - st.get("stage_t0", st["t"])
        st["elapsed_s"] = st["t"] - st.get("started", st["t"])
        return st

    def write(self) -> None:
        """Write the file atomically, never raising.

        A reader never sees a partial file, and a full disk or a vanished directory never touches the search.
        """
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
        """Log a line when the current stage has run past its stall limit.

        The first line comes at ``stall_after`` and each next one once the stage's time has doubled since the
        last.
        """
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
    """Return a duration as a short text.

    Parameters
    ----------
    s : float
        The duration, in seconds.

    Returns
    -------
    text : str
        Such as ``42s``, ``12m`` or ``3.4h``.
    """
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.0f}m"
    return f"{s / 3600:.1f}h"


def read(run_dir: str) -> Optional[Dict[str, Any]]:
    """Return a run's heartbeat, with its age and freshness.

    Parameters
    ----------
    run_dir : str
        The run's directory.

    Returns
    -------
    hb : dict or None
        The file's contents with ``age`` (seconds since its last stamp) and ``fresh`` (younger than
        30 s) added, None without a readable file, as for a run from before the heartbeat or one that
        never got as far as writing it.
    """
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
    """Return a heartbeat's stage and its time so far as one phrase.

    Parameters
    ----------
    hb : dict or None
        The heartbeat (:func:`read`).

    Returns
    -------
    text : str
        Such as ``search (12m)``, empty without a heartbeat.
    """
    if not hb:
        return ""
    stage = hb.get("stage") or "?"
    s = hb.get("stage_s")
    if isinstance(s, (int, float)):
        s = float(s) + (hb.get("age") or 0.0)
        return f"{stage} ({_dur(s)})"
    return str(stage)
