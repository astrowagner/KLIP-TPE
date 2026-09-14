"""Worker / thread bookkeeping shared by the reducers and the runner.

klip-tpe parallelises with threads (numpy's SVD / matmul release the GIL): the
:class:`~klip_tpe.reducer.PartitionedReducer` maps partitions onto one pool, the KLIP
engine maps the per-target basis builds of one partition onto a second pool, and the
runner reduces the injected and the clean cube concurrently.  ``workers`` is the
budget for all of that: ``"auto"`` / ``None`` / ``0`` = every core the machine offers
(``os.sched_getaffinity`` where available, else ``os.cpu_count``); an integer pins it.
With more than one worker the BLAS thread pools are pinned to one thread inside the
KLIP loops (threadpoolctl, when installed) so the threads do not fight over cores.
"""
from __future__ import annotations

import atexit
import concurrent.futures as cf
import contextlib
import os
import threading
import time
from typing import Optional, Union

WorkerSpec = Union[int, str, None]


def cpu_count() -> int:
    """Cores available to this process."""
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except Exception:
        return max(os.cpu_count() or 1, 1)


def resolve_workers(spec: WorkerSpec = "auto") -> int:
    """``'auto'`` / ``None`` / ``0`` / ``'max'`` -> :func:`cpu_count`; ``-k`` -> all but k;
    an integer -> that many (at least 1)."""
    if spec is None:
        return cpu_count()
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("", "auto", "max", "all"):
            return cpu_count()
        spec = int(s)
    n = int(spec)
    if n <= 0:
        return max(cpu_count() + n, 1)
    return n


_lock = threading.Lock()
_pools: dict = {}
_reaper_installed = False


def _install_signal_reaper() -> None:
    """Terminate worker pools on SIGTERM/SIGHUP as well as on a normal exit, so that
    ``pkill`` or a closed terminal does not leave workers behind."""
    global _reaper_installed
    if _reaper_installed or threading.current_thread() is not threading.main_thread():
        return
    _reaper_installed = True
    import signal

    def handler(signum, frame):
        try:
            atexit._run_exitfuncs()           # includes every pool's close()
        except Exception:
            pass
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        try:
            if signal.getsignal(sig) in (signal.SIG_DFL, None):
                signal.signal(sig, handler)
        except (ValueError, OSError):
            pass

#: callables run by the main thread while it waits for worker results (the live
#: display animates the launch movie from here; IDL does the same in its bridge poll loop)
IDLE_HOOKS: list = []


def run_idle_hooks() -> None:
    if not IDLE_HOOKS or threading.current_thread() is not threading.main_thread():
        return
    for fn in list(IDLE_HOOKS):
        try:
            fn()
        except Exception:
            pass


def pump_wait(futures, tick: float = 0.1):
    """Wait for ``futures``, running the idle hooks every ``tick`` seconds.

    The main thread must hand the GUI event loop a turn every couple of seconds or macOS
    stops compositing the live window.  Waiting on a worker *process* already does that
    (:meth:`WorkerPool.gather` polls its queue with a timeout), but a bare
    ``Future.result()`` on a *thread* pool blocks with no such window -- so every place the
    main thread waits on threads goes through here instead.  Results come back in the order
    the futures were given, and the first exception is re-raised, exactly like ``result()``.
    """
    import concurrent.futures as _cf
    futs = list(futures)
    if threading.current_thread() is threading.main_thread() and IDLE_HOOKS:
        pending = set(futs)
        while pending:
            done, pending = _cf.wait(pending, timeout=tick, return_when=_cf.FIRST_COMPLETED)
            run_idle_hooks()
    return [f.result() for f in futs]


def target_pool(n: int) -> cf.ThreadPoolExecutor:
    """Process-wide pool for the per-target KLIP work (one per size; created lazily)."""
    with _lock:
        p = _pools.get(n)
        if p is None:
            p = _pools[n] = cf.ThreadPoolExecutor(max_workers=n, thread_name_prefix="klip-target")
        return p


@contextlib.contextmanager
def single_blas_thread(enabled: bool = True):
    """Pin BLAS/LAPACK to one thread inside a multi-threaded region (no-op without
    threadpoolctl)."""
    if not enabled:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        yield
        return
    with threadpool_limits(limits=1):
        yield


_pinned = False


def pin_blas(limit: int = 1) -> bool:
    """Pin the process' BLAS/LAPACK pools to ``limit`` threads for the rest of the run
    (klip-tpe's own threads provide the parallelism).  Returns True when applied."""
    global _pinned
    if _pinned:
        return True
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=limit)
        _pinned = True
    except Exception:
        return False
    return True


def describe(workers: int, n_partitions: int) -> str:
    conc = 2 if workers >= 2 else 1
    per = max(workers // max(conc * n_partitions, 1), 1)
    return (f"{workers} workers on {cpu_count()} cores: injected + clean reductions "
            f"{'concurrent' if conc == 2 else 'sequential'}, {min(workers, conc * n_partitions)} partition jobs in "
            f"parallel, {per} target thread(s) per partition job")


# ----------------------------------------------------------------------------
# forked worker processes (the IDL "bridges"): one copy of the loaded data, N processes
# ----------------------------------------------------------------------------
import multiprocessing as _mp
import pickle
import queue as _queue
import sys

# ``multiprocessing.Queue.__init__`` finishes with a *lazy* import -- CPython's own comment
# there reads "Can raise ImportError" -- and pays for it under threads:
#
#     from .synchronize import SEM_VALUE_MAX as maxsize
#
# A run builds its pool once the data are loaded, by which time the live display's render
# thread is alive and importing matplotlib.  If that thread happens to be part-way through
# importing ``multiprocessing.synchronize``, the pool's import of the same module returns
# the half-built one and raises
#
#     ImportError: cannot import name 'SEM_VALUE_MAX' from partially initialized module
#                  multiprocessing.synchronize (most likely due to a circular import)
#
# which the reducer catches and answers by falling back to threads -- for the whole run.  A
# 147-D RX J0534 search lost its worker processes to exactly this on a resume and spent the
# next 350 evaluations GIL-bound, single reductions swinging between 6 s and 479 s.  Doing
# the import here, at module import time and so before the run has any threads, means the
# lazy one always finds it finished.  Guarded: a platform with no working ``sem_open`` must
# still fall back rather than fail to import klip-tpe.
try:                                                    # pragma: no cover - platform detail
    import multiprocessing.synchronize as _mp_sync       # noqa: F401
except Exception:                                       # pragma: no cover
    _mp_sync = None


def _watch_parent(ppid: int, period: float = 2.0) -> None:
    """Exit this worker as soon as its parent is gone.

    ``daemon=True`` only covers a *clean* parent exit: multiprocessing's atexit hook
    terminates the children.  A parent that is SIGKILLed, or that dies with the terminal,
    never runs that hook, and the worker then blocks on ``inq.get()`` forever -- an
    invisible process holding a copy of the data until the machine is rebooted.  Polling
    ``getppid()`` catches every case, because an orphan is re-parented (to init, or to
    launchd on macOS) the moment its parent dies.
    """
    def watch():
        while True:
            time.sleep(period)
            try:
                if os.getppid() != ppid:
                    os._exit(0)               # not sys.exit: do not run atexit in a fork
            except Exception:
                os._exit(0)
    threading.Thread(target=watch, name="klip-parent-watch", daemon=True).start()


def _worker_main(reducers, inq, outq, ppid: Optional[int] = None):
    """Worker loop: ``(job_id, pid, request)`` -> ``(job_id, result | exception)``."""
    pin_blas()
    # Ctrl-C reaches every process in the foreground group, and a worker that takes the
    # KeyboardInterrupt dies inside inq.get() -- leaving the parent waiting on a queue
    # nobody feeds.  The parent owns the interrupt; workers stop when it tells them to,
    # or when the watchdog sees it gone.
    try:
        import signal as _sig
        _sig.signal(_sig.SIGINT, _sig.SIG_IGN)
    except Exception:
        pass
    if ppid:
        _watch_parent(ppid)
    while True:
        item = inq.get()
        if item is None:
            return
        job_id, pid, req = item
        try:
            res = reducers[pid].reduce(req)
        except BaseException as exc:      # noqa: BLE001 - report everything back to the parent
            res = exc
        try:
            outq.put((job_id, res))
        except Exception as exc:         # unpicklable result
            outq.put((job_id, RuntimeError(f"worker result not picklable: {exc!r}")))


class ProcessPool:
    """``n`` forked worker processes that each hold a (copy-on-write) reference to the
    reducers and execute ``reducer.reduce(request)`` jobs.  Fork happens once, after the
    data are loaded, so the cubes are shared with the parent and never pickled; only
    requests and the (small) results cross the pipes.  Falls back to ``None`` (caller
    uses threads) when forking is unavailable."""

    def __init__(self, reducers: dict, n: int, log=None):
        self.n = int(n)
        self.log = log or (lambda s: None)
        self.reducers = reducers
        ctx = _mp.get_context("fork") if "fork" in _mp.get_all_start_methods() else None
        if ctx is None or self.n < 1:
            raise RuntimeError("fork start method not available")
        if sys.platform == "darwin":
            os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
        pin_blas()                      # no BLAS thread pools alive across the fork
        self.inq, self.outq = self._queues(ctx)
        me = os.getpid()
        self.procs = [ctx.Process(target=_worker_main, args=(reducers, self.inq, self.outq, me),
                                  name=f"klip-worker-{i}", daemon=True) for i in range(self.n)]
        for p in self.procs:
            p.start()
        # belt and braces: a clean exit, a SIGTERM or an unhandled exception all reap the
        # pool; the watchdog in each worker covers SIGKILL, which no handler can catch
        atexit.register(self.close)
        _install_signal_reaper()
        self._next = 0
        self._lock = threading.Lock()
        self._pending: dict = {}

    @staticmethod
    def _queues(ctx, tries: int = 3):
        """``ctx.Queue()`` twice, retrying the import race described at the top of this
        section.  The pre-import above should make it impossible; this is the belt to that
        braces, because the cost of losing is a whole run in thread mode."""
        last = None
        for _ in range(max(tries, 1)):
            try:
                return ctx.Queue(), ctx.Queue()
            except ImportError as exc:              # partially initialized synchronize
                last = exc
                try:
                    import importlib
                    import multiprocessing.synchronize as s
                    importlib.reload(s)
                except Exception:
                    pass
                time.sleep(0.05)
        raise last if last is not None else RuntimeError("could not create worker queues")

    def alive(self) -> bool:
        return all(p.is_alive() for p in self.procs)

    def map(self, jobs):
        """``jobs = [(pid, request), ...]`` -> results in order (thread-safe: several
        callers may map concurrently)."""
        with self._lock:
            ids = []
            for pid, req in jobs:
                jid = self._next; self._next += 1
                ids.append(jid)
                self.inq.put((jid, pid, req))
        want = set(ids)
        got = {}
        while want:
            with self._lock:
                for jid in list(want):
                    if jid in self._pending:
                        got[jid] = self._pending.pop(jid); want.discard(jid)
            if not want:
                break
            try:
                jid, res = self.outq.get(timeout=0.1 if IDLE_HOOKS else 1.0)
            except _queue.Empty:
                if not self.alive():
                    dead = [p.name for p in self.procs if not p.is_alive()]
                    raise RuntimeError(
                        f"klip-tpe worker(s) {', '.join(dead)} died while {len(want)} job(s) were "
                        f"outstanding, so this evaluation can never complete. The run checkpoints "
                        f"after every evaluation -- restart it with  klip-tpe resume")
                run_idle_hooks()
                continue
            if jid in want:
                got[jid] = res; want.discard(jid)
            else:
                with self._lock:
                    self._pending[jid] = res
        out = []
        for jid in ids:
            r = got[jid]
            if isinstance(r, BaseException):
                raise r
            out.append(r)
        return out

    def close(self):
        for _ in self.procs:
            try:
                self.inq.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
