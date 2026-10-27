"""Worker / thread bookkeeping shared by the reducers and the runner.

klip-tpe parallelises with threads (numpy's SVD / matmul release the GIL): the
:class:`~klip_tpe.reducer.PartitionedReducer` maps partitions onto one pool, the KLIP
engine maps the per-target basis builds of one partition onto a second pool, and the
runner reduces the injected and the clean cube concurrently.  ``workers`` is the
budget for all of that: ``"auto"`` / ``None`` / ``0`` = one per physical core the process
may run on (:func:`physical_cores`), ``"all"`` = every logical CPU, hardware threads
included (:func:`cpu_count`); an integer pins it.  With more than one worker the BLAS thread
pools are pinned to one thread inside the KLIP loops (threadpoolctl, when installed) so the
threads do not fight over cores.
"""
from __future__ import annotations

import atexit
import concurrent.futures as cf
import contextlib
import os
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

WorkerSpec = Union[int, str, None]


def cpu_count() -> int:
    """Logical CPUs available to this process, hardware threads (SMT) included."""
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except Exception:
        return max(os.cpu_count() or 1, 1)


def _cpulist(text: str) -> Optional[frozenset]:
    """A sysfs CPU list (``0,96`` or ``0-3,8``) as a set, None when it does not parse."""
    out = set()
    try:
        for part in text.strip().split(","):
            if not part:
                continue
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    except ValueError:
        return None
    return frozenset(out) or None


_physical: Optional[int] = None


def physical_cores(sysfs: str = "/sys/devices/system/cpu") -> int:
    """Physical cores available to this process: the CPUs it may run on (:func:`cpu_count`)
    with each core's hardware threads counted once.  Linux reads each allowed CPU's
    siblings in sysfs (``topology/core_cpus_list``, or ``thread_siblings_list`` on older
    kernels), macOS asks ``sysctl hw.physicalcpu``.  Where neither answers, the logical
    count.

    The default worker budget since 2026-10-27.  On patrick (96 cores, 192 hardware threads)
    a farm benchmark of NEAR2's annulus 5 ran 1.25x faster with 48 jobs of 2 threads at a
    budget of 96 than with 48 jobs of 4 at 192.  The second thread of a core added nothing on
    this code and cost cache and memory bandwidth."""
    global _physical
    if _physical is not None and sysfs == "/sys/devices/system/cpu":
        return _physical
    n_log = cpu_count()
    found = None
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except Exception:
        cpus = None
    if cpus and os.path.isdir(sysfs):
        cores = set()
        for c in cpus:
            sib = None
            for name in ("core_cpus_list", "thread_siblings_list"):
                try:
                    with open(os.path.join(sysfs, f"cpu{c}", "topology", name)) as f:
                        sib = _cpulist(f.read())
                except OSError:
                    continue
                if sib is not None:
                    break
            if sib is None:
                cores = None
                break
            cores.add(sib)
        if cores:
            found = len(cores)
    if found is None and sys.platform == "darwin":
        try:
            import subprocess
            out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True, timeout=5)
            found = int(out.stdout.strip()) or None
        except Exception:
            found = None
    n = max(min(found or n_log, n_log), 1)
    if sysfs == "/sys/devices/system/cpu":
        _physical = n
    return n


def resolve_workers(spec: WorkerSpec = "auto") -> int:
    """The worker budget, the threads the run's jobs may use together.  ``'auto'`` / ``None``
    / ``0`` -> :func:`physical_cores`; ``'all'`` / ``'logical'`` / ``'max'`` -> every logical
    CPU (:func:`cpu_count`, what ``'auto'`` meant before 2026-10-27); ``-k`` -> ``'auto'``
    less k; an integer -> that many (at least 1)."""
    if spec is None:
        return physical_cores()
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("", "auto"):
            return physical_cores()
        if s in ("all", "logical", "max"):
            return cpu_count()
        spec = int(s)
    n = int(spec)
    if n <= 0:
        return max(physical_cores() + n, 1)
    return n


_lock = threading.Lock()
_pools: dict = {}
_reaper_installed = False


def _after_fork_in_child() -> None:
    """A forked worker inherits the parent's thread pools without their threads: work
    submitted to one would wait forever.  It also inherits ``_lock`` as it stood, possibly
    held by a parent thread that does not exist here.  Start both afresh."""
    global _lock
    _lock = threading.Lock()
    _pools.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


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


def mem_available() -> Optional[int]:
    """Bytes of memory the kernel says are available to new work (Linux ``MemAvailable``),
    or None."""
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cores_text() -> str:
    """``96 cores (192 logical CPUs)``, or ``2 cores`` where each core has one thread."""
    phys, log = physical_cores(), cpu_count()
    return f"{phys} cores ({log} logical CPUs)" if log != phys else f"{phys} cores"


def describe(workers: int, n_partitions: int, procs: Optional[int] = None,
             per_job: Optional[int] = None, cap: Optional[int] = None, budget: Optional[int] = None) -> str:
    """The run log's line about the worker budget.  ``procs`` / ``per_job``: the worker
    processes and threads per job (the thread-pool fallback passes None); ``cap``: the
    most threads a job expected to be long can get; ``budget``: the most the running jobs
    use together, when the pool holds them to it."""
    if procs is None:
        conc = 2 if workers >= 2 else 1
        per = max(workers // max(conc * n_partitions, 1), 1) if per_job is None else per_job
        return (f"{workers} workers on {_cores_text()}: injected + clean reductions "
                f"{'concurrent' if conc == 2 else 'sequential'}, {min(workers, conc * n_partitions)} partition "
                f"jobs in parallel, {per} target thread(s) per partition job")
    scaled = ""
    if cap and cap > 1:
        held = f", at most {budget} in flight" if budget else ""
        scaled = f" (1 to {cap} by the job's expected cost, longest first{held})"
    return (f"{workers} workers on {_cores_text()}: {procs} worker processes x {per_job} thread(s) per "
            f"job{scaled}, {n_partitions} partitions per reduction, so "
            f"{max(procs // max(n_partitions, 1), 1)} reduction(s) at full width at once")


# ----------------------------------------------------------------------------
# forked worker processes (the IDL "bridges"): one copy of the loaded data, N processes
# ----------------------------------------------------------------------------
import heapq
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

    def __init__(self, reducers: dict, n: int, log=None, budget: Optional[int] = None):
        self.n = int(n)
        self.log = log or (lambda s: None)
        self.reducers = reducers
        #: the threads the running jobs may use together, or None for no limit.  A job is
        #: sent with its request's ``threads`` cut to what the running jobs leave, keeping one
        #: for each idle worker, so the jobs in flight never use more than this (at least one
        #: thread a job, so never less than ``n``).  The images do not depend on the count.
        self.budget = None if budget is None else max(int(budget), self.n)
        self._threads: dict = {}           # job id -> the threads it was sent with
        self._threads_running = 0
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
        self._futures: dict = {}           # job id -> Future, settled by the collector
        self._waiting: list = []           # heap of (key, job id, pid, request) not yet sent
        self._busy = 0                     # jobs sent to the workers and not back yet
        self._closed = False
        # One thread reads every result and settles the job's future.  Several callers map
        # at once (the runner keeps a trial's reductions in flight together).  When each
        # caller read the queue itself, a result another caller picked up sat in a dict
        # until its owner's next read timed out: up to a second per reduction, more than a
        # synthetic reduction takes.  Started after the fork, so no worker has it.
        self._collector = threading.Thread(target=self._collect, name="klip-pool-results", daemon=True)
        self._collector.start()

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

    def memory(self) -> Optional[dict]:
        """Linux only, None elsewhere: each worker's peak resident set (``VmHWM``, which
        counts in full the copy-on-write data it shares with the parent and the other
        workers) and the pool's proportional set now (``Pss``, every shared page split
        between the processes using it), in bytes."""
        peaks, pss = [], 0
        for p in self.procs:
            try:
                with open(f"/proc/{p.pid}/status") as f:
                    peaks += [int(ln.split()[1]) * 1024 for ln in f if ln.startswith("VmHWM:")]
                with open(f"/proc/{p.pid}/smaps_rollup") as f:
                    pss += sum(int(ln.split()[1]) * 1024 for ln in f if ln.startswith("Pss:"))
            except (OSError, ValueError, IndexError):
                return None
        return {"peak": peaks, "pss": pss} if len(peaks) == len(self.procs) else None

    def map(self, jobs, keys=None):
        """``jobs = [(pid, request), ...]`` -> results in order (thread-safe: several
        callers may map concurrently).  A failed job raises its exception.  Waiting from
        the main thread services the live window (:func:`pump_wait`).

        ``keys``: one sort key per job.  A worker that comes free takes the waiting job with
        the smallest key, so a caller can send long jobs first.  Without keys, and between
        equal keys, jobs go in the order they came."""
        futs = []
        with self._lock:
            if self._closed or not self._collector.is_alive():
                raise RuntimeError("klip-tpe worker pool is closed")
            for k, (pid, req) in enumerate(jobs):
                jid = self._next
                self._next += 1
                f = cf.Future()
                self._futures[jid] = f
                futs.append(f)
                key = 0.0 if keys is None else float(keys[k])
                heapq.heappush(self._waiting, (key, jid, pid, req))
            self._dispatch()
        return pump_wait(futs)

    def _dispatch(self) -> None:
        """Send waiting jobs, smallest key first, until every worker has one (call with
        the lock held).  Only as many jobs as workers are ever in the queue, so the order
        is decided here, when a worker comes free, not when a job was submitted, and so is
        a job's share of the thread budget (:meth:`_fit`)."""
        while self._waiting and self._busy < self.n:
            _, jid, pid, req = heapq.heappop(self._waiting)
            self.inq.put((jid, pid, self._fit(jid, req)))
            self._busy += 1

    def _fit(self, jid: int, req):
        """``req`` with its ``threads`` cut to the budget's room (call with the lock held):
        the budget less the running jobs' threads and one for each worker still idle after
        this one.  That keeps the threads in flight within the budget at every moment, while
        every worker can still take a job.  Without a budget, or a request without
        ``threads``, it goes as it is.

        The by-cost plan (:meth:`klip_tpe.reducer.PartitionedReducer._plan`) gives a long
        job up to 4x the base and a short one 1, so over the jobs submitted the mean is near
        the base.  The jobs running at a given moment are not that mean: the long ones run
        longest, so they and their threads fill the pool.  On patrick, 48 jobs at a base of 4
        ran at a load average of 234 on 192 CPUs, and at a base of 2 at 134 on a budget of
        96."""
        t = getattr(req, "threads", None)
        if self.budget is None or t is None:
            return req
        room = self.budget - self._threads_running - (self.n - self._busy - 1)
        got = max(1, min(int(t), room))
        self._threads[jid] = got
        self._threads_running += got
        if got == int(t):
            return req
        import copy
        import dataclasses
        if dataclasses.is_dataclass(req):
            return dataclasses.replace(req, threads=got)
        req = copy.copy(req)
        req.threads = got
        return req

    def _collect(self) -> None:
        """The collector thread: every result off the queue to its job's future, and every
        outstanding job failed when a worker has died (it can never come back) or the pool
        closes."""
        while not self._closed:
            try:
                item = self.outq.get(timeout=1.0)
            except _queue.Empty:
                try:
                    if not self._closed and not self.alive():
                        dead = [p.name for p in self.procs if not p.is_alive()]
                        self._fail_all(lambda n: RuntimeError(
                            f"klip-tpe worker(s) {', '.join(dead)} died while {n} job(s) were outstanding, "
                            f"so this evaluation can never complete. The run checkpoints after every "
                            f"evaluation -- restart it with  klip-tpe resume"))
                except Exception:
                    pass
                continue
            except Exception:                  # the queue went away (interpreter exit)
                break
            if item is None:                   # close()
                break
            try:
                jid, res = item
                with self._lock:
                    f = self._futures.pop(jid, None)
                    self._busy = max(self._busy - 1, 0)
                    self._threads_running -= self._threads.pop(jid, 0)
                    self._dispatch()
                if f is not None and not f.done():
                    if isinstance(res, BaseException):
                        f.set_exception(res)
                    else:
                        f.set_result(res)
            except Exception:
                pass
        self._fail_all(lambda n: RuntimeError(f"klip-tpe worker pool closed with {n} job(s) outstanding"))

    def _fail_all(self, make_exc) -> None:
        with self._lock:
            futs, self._futures = self._futures, {}
            self._waiting = []
            self._threads, self._threads_running = {}, 0
        for f in futs.values():
            if not f.done():
                f.set_exception(make_exc(len(futs)))

    def close(self):
        self._closed = True
        for _ in self.procs:
            try:
                self.inq.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        try:
            self.outq.put(None)                # wakes the collector
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass



# ----------------------------------------------------------------------------
# arrays made once per reduction and read by every worker: shared memory, not pickles
# ----------------------------------------------------------------------------
class SharedArrays:
    """Named numpy arrays in one shared-memory block, for the worker processes to read in
    place.

    The SDI mode prepares every wavelength group's frames once per reduction and hands all of
    them to each group's job.  Through the job queue that is tens of MB pickled per job, hundreds
    per reduction, serialised by the parent; here the parent writes them once into a
    :class:`multiprocessing.shared_memory.SharedMemory` block and the jobs carry only this
    descriptor (the block's name and each array's offset, shape and dtype, plus ``extra``, small
    picklable things that go with them).  A worker :meth:`attach` es, gets views, and closes its
    handle when its copies are made; the parent :meth:`release` s the block when the reduction is
    done.  Picklable without the handle.  With threads instead of processes nothing of this is
    needed: the arrays are passed as they are."""

    def __init__(self, name: str, size: int, index: Dict[str, Tuple[int, Tuple[int, ...], str]],
                 extra: Optional[Dict[str, Any]] = None):
        self.name, self.size, self.index = str(name), int(size), dict(index)
        self.extra = dict(extra or {})
        self._shm = None

    @classmethod
    def create(cls, arrays: Dict[str, np.ndarray], extra: Optional[Dict[str, Any]] = None) -> "SharedArrays":
        """Write ``arrays`` into a new block (the parent's side).  Raises when shared memory
        is not available, so a caller can fall back to passing the arrays themselves."""
        from multiprocessing import shared_memory
        index: Dict[str, Tuple[int, Tuple[int, ...], str]] = {}
        off = 0
        for k, a in arrays.items():
            a = np.ascontiguousarray(a)
            index[k] = (off, tuple(int(v) for v in a.shape), a.dtype.str)
            off += int(a.nbytes)
            off += (-off) % 64                       # 64-byte alignment for the next array
        shm = shared_memory.SharedMemory(create=True, size=max(off, 1))
        for k, a in arrays.items():
            o, shape, dt = index[k]
            a = np.ascontiguousarray(a)
            view = np.ndarray(shape, dtype=np.dtype(dt), buffer=shm.buf, offset=o)
            view[...] = a
        out = cls(shm.name, max(off, 1), index, extra)
        out._shm = shm
        return out

    def attach(self):
        """``(arrays, closer)``: the arrays as views into the block, and a callable that
        drops this process's handle once the views are no longer needed (a view read after
        it is undefined)."""
        from multiprocessing import shared_memory
        # Attaching registers the block with the resource tracker as if this process owned
        # it, and the tracker (a worker's own, or the parent's when the worker inherited its
        # pipe, as on macOS) would then unlink it at exit or complain that it could not.  The
        # parent owns the block, so the registration is skipped here: the register call is
        # stood down for the duration of the attach (one job at a time in a worker).
        from multiprocessing import resource_tracker
        orig = resource_tracker.register
        resource_tracker.register = lambda *a, **k: None
        try:
            shm = shared_memory.SharedMemory(name=self.name, create=False)
        finally:
            resource_tracker.register = orig
        arrays = {k: np.ndarray(shape, dtype=np.dtype(dt), buffer=shm.buf, offset=o)
                  for k, (o, shape, dt) in self.index.items()}

        def closer():
            try:
                shm.close()
            except Exception:
                pass
        return arrays, closer

    def release(self) -> None:
        """The parent's side: close and unlink the block."""
        shm, self._shm = self._shm, None
        if shm is None:
            return
        for f in (shm.close, shm.unlink):
            try:
                f()
            except Exception:
                pass

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_shm"] = None
        return d

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
