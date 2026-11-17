"""The live display's step panels, drawn by a process of their own.

A pipelined search's pace is set by its main thread once the workers keep up.  The step panel
was drawn on a thread of the run's own process, and matplotlib holds the interpreter while it
draws, 2 to 4 s a panel.  The main thread then ran at a fraction of its speed: NEAR2 measured
its steps 2 to 4 times slower in the fast searches of s7e (note of 2026-11-16), and a
NEAR-shaped synthetic search resumed at 9,800 trials in the sandbox took 373 ms a trial with
the panels against 83 ms without them.

:class:`PanelProcess` starts ``python -m klip_tpe.panels`` and keeps two pipes to it.  The
display sends each evaluation's record once, and for a panel only what the run alone has: the
evaluation's images, the best images when they change, the numbers of the panel's text and
the curves of the finished annuli.  The process keeps the annulus' records and builds its data
(:func:`klip_tpe.display.annulus_from_records`), works out the evaluation's contrast preview,
reads the running stitch, draws the panel and its PDF, and rebuilds the progress movie, all
off the run's interpreter.  Two threads of the run write and read the pipes, so the main
thread only queues a message.  The process runs at a lower priority than the workers, and a
Ctrl-C of the run does not stop it: it finishes what it was sent and exits when the run's end
of the pipe closes.

``KLIP_TPE_PANELS=thread`` draws the panels on the run's render thread, as before, and so does
a run whose process cannot start or stops.
"""
from __future__ import annotations

import os
import pickle
import queue
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

__all__ = ["ENV", "PanelProcess", "engine", "main"]

#: the environment variable that chooses where the step panels are drawn
ENV = "KLIP_TPE_PANELS"

_HDR = struct.Struct("<Q")
_LOG = "panels.log"


def engine() -> str:
    """Return where the step panels are drawn, from :data:`ENV`.

    Returns
    -------
    engine : {"process", "thread"}
        ``"thread"`` when the variable says ``thread`` (or ``inline``, ``off``, ``0``, ``no``), else
        ``"process"``.
    """
    v = os.environ.get(ENV, "").strip().lower()
    return "thread" if v in ("thread", "inline", "main", "0", "off", "no") else "process"


def _write(f, obj: Any) -> None:
    """Write one message to a pipe, its length and then its pickle.

    Parameters
    ----------
    f : file
        The pipe, opened for binary writing.
    obj : object
        The message.
    """
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    f.write(_HDR.pack(len(data)))
    f.write(data)
    f.flush()


def _exact(f, n: int) -> Optional[bytes]:
    """Read exactly ``n`` bytes from a pipe.

    Parameters
    ----------
    f : file
        The pipe, opened for binary reading.
    n : int
        Number of bytes.

    Returns
    -------
    data : bytes or None
        The bytes, None when the pipe closes first.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _read(f) -> Any:
    """Read one message from a pipe.

    Parameters
    ----------
    f : file
        The pipe, opened for binary reading.

    Returns
    -------
    message : object or None
        The message, None when the pipe has closed.
    """
    hdr = _exact(f, _HDR.size)
    if hdr is None:
        return None
    data = _exact(f, _HDR.unpack(hdr)[0])
    return None if data is None else pickle.loads(data)


# ----------------------------------------------------------------------------
# the run's side
# ----------------------------------------------------------------------------
class PanelProcess:
    """The process that draws a run's step panels, and the two threads that talk to it.

    Parameters
    ----------
    init : dict
        The settings of the panels: ``run_dir``, ``dpi`` and ``corner_mode``.

    Attributes
    ----------
    broken : str or None
        Why the process can no longer draw, None while it can.
    dir : str
        A directory of the process' own, holding its log.
    """

    def __init__(self, init: Dict[str, Any]):
        from .books import _env
        self.dir = tempfile.mkdtemp(prefix="klip_panels_")
        self._logf = open(os.path.join(self.dir, _LOG), "wb")
        try:
            # started in its own directory, so nothing in the run's working directory can stand
            # in for a module
            self.proc = subprocess.Popen([sys.executable, "-m", "klip_tpe.panels"], env=_env(), cwd=self.dir,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._logf,
                                         close_fds=True)
        except BaseException:
            self._logf.close()
            shutil.rmtree(self.dir, ignore_errors=True)
            raise
        self.broken: Optional[str] = None
        self._q: "queue.Queue[Any]" = queue.Queue()
        self._cv = threading.Condition()
        self._pending: Dict[int, str] = {}
        self._done: List[Dict[str, Any]] = []
        self._next = 1
        self._closing = False
        self._w = threading.Thread(target=self._writer, name="klip-panels-out", daemon=True)
        self._r = threading.Thread(target=self._reader, name="klip-panels-in", daemon=True)
        self._w.start()
        self._r.start()
        self.send(dict(init, op="init"))

    def send(self, msg: Dict[str, Any], reply: bool = False) -> Optional[int]:
        """Queue a message for the process.

        Parameters
        ----------
        msg : dict
            The message, its ``op`` naming what to do.
        reply : bool, default False
            Whether the process reports when it has done it.

        Returns
        -------
        id : int or None
            The message's number, which its report carries, None without a report or once the process
            is broken.
        """
        if self.broken is not None:
            return None
        mid = None
        if reply:
            with self._cv:
                mid = self._next
                self._next += 1
                self._pending[mid] = str(msg.get("op"))
            msg = dict(msg, id=mid)
        self._q.put(msg)
        return mid

    def busy(self, op: str = "panel") -> bool:
        """Return whether a message of one kind awaits its report.

        Parameters
        ----------
        op : str, default "panel"
            The kind.

        Returns
        -------
        busy : bool
            True while one is queued or being done.
        """
        with self._cv:
            return any(v == op for v in self._pending.values())

    def take(self) -> List[Dict[str, Any]]:
        """Return the reports that came in since the last call.

        Returns
        -------
        reports : list of dict
            Each report's ``id``, ``kind``, ``ok`` and, as the kind has them, ``png``, ``screen``,
            ``error`` and ``s``.
        """
        with self._cv:
            out, self._done = self._done, []
        return out

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until every message awaiting a report has its report.

        Parameters
        ----------
        timeout : float or None, default None
            Longest wait in seconds, None without a limit.

        Returns
        -------
        done : bool
            True when nothing is awaited any more, False on the timeout or once the process is broken.
        """
        end = None if timeout is None else time.time() + float(timeout)
        with self._cv:
            while self._pending and self.broken is None:
                if self.proc.poll() is not None:
                    break
                left = 0.5 if end is None else min(0.5, end - time.time())
                if left <= 0:
                    return False
                self._cv.wait(left)
        if self._pending and self.broken is None and self.proc.poll() is not None:
            self.fail(f"the panel process ended with code {self.proc.returncode}")
        return not self._pending and self.broken is None

    def log_tail(self, n: int = 600) -> str:
        """Return the end of the process' log.

        Parameters
        ----------
        n : int, default 600
            Number of characters.

        Returns
        -------
        text : str
            The last ``n`` characters, empty when there is no log.
        """
        try:
            with open(os.path.join(self.dir, _LOG), errors="replace") as f:
                return f.read()[-n:].strip()
        except OSError:
            return ""

    def close(self, timeout: float = 120.0) -> None:
        """Let the process finish what it was sent, then end it and remove its directory.

        Parameters
        ----------
        timeout : float, default 120.0
            Longest wait for the process in seconds, after which it is killed.
        """
        self._closing = True
        self._q.put({"op": "stop"})
        self._q.put(None)
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self._w.join(timeout=5.0)
        self._r.join(timeout=5.0)
        try:
            self._logf.close()
        except OSError:
            pass
        shutil.rmtree(self.dir, ignore_errors=True)

    def fail(self, why: str) -> None:
        """Mark the process broken and drop what awaited a report.

        Parameters
        ----------
        why : str
            The reason.
        """
        with self._cv:
            if self.broken is None:
                tail = self.log_tail()
                self.broken = why + (f" ({tail})" if tail else "")
            self._pending.clear()
            self._cv.notify_all()

    def _writer(self) -> None:
        """Write the queued messages to the process, one after another."""
        f = self.proc.stdin
        while True:
            m = self._q.get()
            if m is None:
                break
            try:
                _write(f, m)
            except Exception as exc:              # noqa: BLE001 - the display falls back
                if not self._closing:
                    self.fail(f"writing to the panel process failed: {exc!r}")
                break
        try:
            f.close()
        except OSError:
            pass

    def _reader(self) -> None:
        """Collect the process' reports until its pipe closes."""
        f = self.proc.stdout
        while True:
            try:
                m = _read(f)
            except Exception:                     # noqa: BLE001 - a torn message ends the stream
                m = None
            if m is None:
                if not self._closing:
                    self.fail("the panel process closed its pipe")
                with self._cv:
                    self._cv.notify_all()
                return
            with self._cv:
                self._pending.pop(m.get("id"), None)
                self._done.append(m)
                self._cv.notify_all()


# ----------------------------------------------------------------------------
# the process' side
# ----------------------------------------------------------------------------
class _Worker:
    """The panel process' state: the annulus' records and what its panels share.

    Parameters
    ----------
    reply : callable
        Function ``reply(report)`` that sends a report to the run.
    """

    def __init__(self, reply: Callable[[Dict[str, Any]], None]):
        self.reply = reply
        self.run_dir = ""
        self.dpi = 100
        self.corner_mode: Optional[str] = "cycle"
        self.corner_i = 0
        self.recipe: Optional[Dict[str, Any]] = None
        self.records: List[Dict[str, Any]] = []
        self.best: Dict[str, Any] = {}
        self.best_curve: Optional[Dict[str, Any]] = None
        self.fig = None
        self._movie: Optional[threading.Thread] = None

    def handle(self, msg: Dict[str, Any]) -> None:
        """Do what a message says.

        Parameters
        ----------
        msg : dict
            The message.

        Raises
        ------
        ValueError
            If its ``op`` is unknown.
        """
        fn = getattr(self, "_op_" + str(msg.get("op")), None)
        if fn is None:
            raise ValueError(f"unknown message {msg.get('op')!r}")
        fn(msg)

    def finish(self) -> None:
        """Wait for a movie being made."""
        if self._movie is not None:
            self._movie.join()

    def _figure(self):
        """Return the offscreen figure the panels are drawn on, made at the first panel.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure, on an Agg canvas.
        """
        if self.fig is None:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            from .display import panel_size
            self.fig = Figure(figsize=panel_size(self.dpi), dpi=self.dpi)
            FigureCanvasAgg(self.fig)
        return self.fig

    def _ad(self):
        """Return the annulus' data from the records so far.

        Returns
        -------
        ad : AnnulusData
            The data, as :func:`klip_tpe.display.annulus_from_records` builds it from the records and the
            recipe.
        """
        from .display import annulus_from_records
        return annulus_from_records(self.records, **self.recipe)

    def _op_init(self, msg: Dict[str, Any]) -> None:
        """Take the panels' settings.

        Parameters
        ----------
        msg : dict
            ``run_dir``, ``dpi`` and ``corner_mode``.
        """
        self.run_dir = str(msg.get("run_dir") or "")
        self.dpi = int(msg.get("dpi") or 100)
        self.corner_mode = msg.get("corner_mode", "cycle")

    def _op_recipe(self, msg: Dict[str, Any]) -> None:
        """Take an annulus' recipe, and with ``reset`` start its records and best images again.

        Parameters
        ----------
        msg : dict
            ``kw``, the keywords of :func:`klip_tpe.display.annulus_from_records` but the records, and
            ``reset``.
        """
        if msg.get("reset") or self.recipe is None:
            self.records, self.best, self.best_curve = [], {}, None
        self.recipe = dict(msg["kw"])

    def _op_records(self, msg: Dict[str, Any]) -> None:
        """Take records, from their index on.

        Parameters
        ----------
        msg : dict
            ``start``, the first record's index, and ``recs``, the record dicts.

        Raises
        ------
        ValueError
            If they start past the records held.
        """
        k = int(msg["start"])
        recs = list(msg["recs"])
        if k > len(self.records):
            raise ValueError(f"records from {k} after {len(self.records)}")
        self.records[k:k + len(recs)] = recs

    def _op_best(self, msg: Dict[str, Any]) -> None:
        """Take the best images.

        Parameters
        ----------
        msg : dict
            ``fields``, any of ``best_inj``, ``best_clean`` and ``inj_model``.
        """
        self.best.update(msg["fields"])

    def _op_best_curve(self, msg: Dict[str, Any]) -> None:
        """Keep a new best evaluation's contrast preview for the panels to come, its own panel skipped.

        Parameters
        ----------
        msg : dict
            ``i``, the evaluation's index, and ``clean``, its clean image.
        """
        from .display import _live_curve_info
        live, _ = _live_curve_info(msg.get("clean"), self._ad(), int(msg["i"]))
        if live:
            self.best_curve = live

    def _op_panel(self, msg: Dict[str, Any]) -> None:
        """Draw a panel and report it.

        The annulus' data is built from the records.  With ``live`` the evaluation's contrast preview is
        worked out, kept as the best one for a new best, and its reason given as the panel's note when
        there is none.  With ``stitch`` the running stitch is read, and with ``corner`` the landscape's
        partition steps on.  The report carries the window's copy when ``box`` asks for one.

        Parameters
        ----------
        msg : dict
            ``id``, ``i``, ``cur`` (the evaluation's images), ``img`` (the other :class:`StepImages` fields
            the run alone has, ``best_labels_from`` to read the best labels from a record), ``tail`` (the
            curves after the previews), ``kw`` (the keywords of :func:`klip_tpe.display.render_step`),
            ``png``, ``pdf``, ``box``, ``is_best``, ``live``, ``stitch``, ``stitch_sources``, ``corner``
            and ``winner_none``.
        """
        t0 = time.time()
        out: Dict[str, Any] = {"op": "done", "id": msg.get("id"), "kind": "panel", "png": msg.get("png")}
        try:
            from . import display as D
            from .winsize import screen_copy
            ad = self._ad()
            if msg.get("winner_none"):
                ad.winner = None
            i = int(msg["i"])
            cur = dict(msg.get("cur") or {})
            kw = dict(msg.get("kw") or {})
            live = None
            if msg.get("live"):
                live, reason = D._live_curve_info(cur.get("cur_clean"), ad, i)
                kw["contrast_note"] = None if live else f"eval {i + 1}: {reason}"
                if msg.get("is_best") and live:
                    self.best_curve = live
            curves = D._head_curves(live, self.best_curve) + list(msg.get("tail") or [])
            img = dict(self.best)
            img.update(msg.get("img") or {})
            img.update(cur)
            if img.get("best_index") is None:
                img["best_index"] = int(ad.best()[0])
            if img.get("fm_label", "") is None:
                img["fm_label"] = f"KLIP-FM preview (best, eval {int(img['best_index']) + 1})"
            lab = img.pop("best_labels_from", None)
            if lab is not None:
                idx, nsrc = lab
                img["best_labels"] = (list(ad.per_source[idx]) if idx is not None and 0 <= idx < ad.n
                                      else [None] * int(nsrc))
            if msg.get("stitch"):
                img.update(D.stitch_cells(self.run_dir, int(self.recipe.get("ia", 0)), msg.get("stitch_sources")))
            if msg.get("corner"):
                pid, step = D.corner_pick(ad, self.corner_mode, self.corner_i)
                self.corner_i += step
                kw["corner_partition"] = pid
            fig = self._figure()
            D.render_step(ad, i, D.StepImages(**img), msg.get("png"), msg.get("pdf"), curves=curves,
                          dpi=self.dpi, fig=fig, **kw)
            box = msg.get("box")
            if box and not D._fills((int(fig.bbox.width), int(fig.bbox.height)), box):
                out["screen"] = screen_copy(fig, box)
            out["ok"] = True
        except Exception as exc:                  # noqa: BLE001 - reported to the run
            out.update(ok=False, error=f"{exc!r}\n" + traceback.format_exc(limit=4))
        out["s"] = time.time() - t0
        self.reply(out)

    def _op_movie(self, msg: Dict[str, Any]) -> None:
        """Make the progress movie on a thread of its own, unless one is being made, and report it.

        Parameters
        ----------
        msg : dict
            ``id``, ``frames``, ``gif``, ``mp4`` and ``hold_last``.
        """
        mid = msg.get("id")
        if self._movie is not None and self._movie.is_alive():
            self.reply({"op": "done", "id": mid, "kind": "movie", "ok": True, "skipped": True})
            return

        def job():
            t0 = time.time()
            out: Dict[str, Any] = {"op": "done", "id": mid, "kind": "movie"}
            try:
                from .animate import make_movie
                os.makedirs(os.path.dirname(msg["gif"]), exist_ok=True)
                make_movie(list(msg["frames"]), msg["gif"], msg.get("mp4"), intro=None, intro_frames=0,
                           hold_last=int(msg.get("hold_last", 4)))
                out["ok"] = True
            except Exception as exc:              # noqa: BLE001 - reported to the run
                out.update(ok=False, error=f"{exc!r}\n" + traceback.format_exc(limit=4))
            out["s"] = time.time() - t0
            self.reply(out)
        self._movie = threading.Thread(target=job, name="klip-panels-movie", daemon=True)
        self._movie.start()

    def _op_wait(self, msg: Dict[str, Any]) -> None:
        """Report once the movie being made is done, and with it everything sent before.

        Parameters
        ----------
        msg : dict
            ``id``.
        """
        self.finish()
        self.reply({"op": "done", "id": msg.get("id"), "kind": "wait", "ok": True})


def main(argv=None) -> int:
    """``python -m klip_tpe.panels``: draw the panels the run sends until its pipe closes.

    Parameters
    ----------
    argv : sequence of str or None, default None
        Unused, for the entry point's signature.

    Returns
    -------
    code : int
        0.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)          # a Ctrl-C of the run: finish what was sent
    except Exception:
        pass
    try:
        os.nice(10)                                           # behind the workers
    except Exception:
        pass
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)                                              # anything printed goes to the log
    inp = sys.stdin.buffer
    import matplotlib
    matplotlib.use("Agg")
    lock = threading.Lock()

    def reply(obj: Dict[str, Any]) -> None:
        with lock:
            _write(out, obj)
    w = _Worker(reply)
    q: "queue.Queue[Any]" = queue.Queue()

    def reader() -> None:
        while True:
            try:
                m = _read(inp)
            except Exception:                     # noqa: BLE001 - a torn message ends the stream
                m = None
            q.put(m)
            if m is None:
                return
    threading.Thread(target=reader, name="klip-panels-read", daemon=True).start()
    ppid = os.getppid()

    def watch() -> None:
        # The run's workers are forked from it and hold its end of the pipe too, so a run that dies
        # leaves the pipe open as long as one of them lives.  Once the run is gone, the process
        # finishes what it was sent and stops.
        while os.getppid() == ppid:
            time.sleep(2.0)
        q.put(None)
    threading.Thread(target=watch, name="klip-panels-watch", daemon=True).start()
    while True:
        m = q.get()
        if m is None or m.get("op") == "stop":
            break
        try:
            w.handle(m)
        except Exception as exc:                  # noqa: BLE001 - reported to the run when it waits
            sys.stderr.write(f"panel message {m.get('op')!r} failed: {exc!r}\n{traceback.format_exc(limit=4)}")
            if m.get("id") is not None:
                reply({"op": "done", "id": m["id"], "kind": str(m.get("op")), "ok": False, "error": repr(exc)})
    w.finish()
    try:
        out.close()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
