"""Worker and thread bookkeeping shared by the reducers and the runner.

klip-tpe parallelizes with threads, since numpy's SVD and matmul release the GIL: the
:class:`~klip_tpe.reducer.PartitionedReducer` maps partitions onto one pool, the KLIP engine maps
the per-target basis builds of one partition onto a second pool, and the runner reduces the
injected and the clean cube concurrently.  It also forks worker processes (:class:`ProcessPool`),
with card workers that reduce on a GPU.

``workers`` is the budget for all of that.  ``"auto"``, None or 0 is one per physical core the
process may run on (:func:`physical_cores`), ``"all"`` every logical CPU, hardware threads included
(:func:`cpu_count`), and an integer pins it.  With more than one worker the BLAS thread pools are
pinned to one thread inside the KLIP loops (threadpoolctl, when installed), so the threads do not
fight over cores.
"""
from __future__ import annotations

import atexit
import concurrent.futures as cf
import contextlib
import math
import os
import sys
import threading
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

WorkerSpec = Union[int, str, None]


def cpu_count() -> int:
    """Return the logical CPUs available to this process, hardware threads (SMT) included.

    Returns
    -------
    n : int
        The count, at least 1.
    """
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except Exception:
        return max(os.cpu_count() or 1, 1)


def _cpulist(text: str) -> Optional[frozenset]:
    """Parse a sysfs CPU list, such as ``0,96`` or ``0-3,8``, as a set.

    Parameters
    ----------
    text : str
        The list.

    Returns
    -------
    cpus : frozenset of int or None
        The CPUs, None when the list does not parse or is empty.
    """
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
    """Return the physical cores available to this process, each core's hardware threads counted once.

    They are the CPUs it may run on (:func:`cpu_count`).  Linux reads each allowed CPU's siblings in
    sysfs (``topology/core_cpus_list``, or ``thread_siblings_list`` on older kernels), and macOS asks
    ``sysctl hw.physicalcpu``.  Where neither answers, it is the logical count.  The answer for the
    default ``sysfs`` is cached.

    Parameters
    ----------
    sysfs : str, default "/sys/devices/system/cpu"
        The sysfs CPU directory.

    Returns
    -------
    n : int
        The count, at least 1 and at most the logical count.

    Notes
    -----
    It has been the default worker budget since 2026-10-27.  On patrick (96 cores, 192 hardware
    threads) a farm benchmark of NEAR2's annulus 5 ran 1.25x faster with 48 jobs of 2 threads at a
    budget of 96 than with 48 jobs of 4 at 192.  The second thread of a core added nothing on this code
    and cost cache and memory bandwidth.
    """
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
    """Return the worker budget, the threads the run's jobs may use together.

    Parameters
    ----------
    spec : int, str or None, default "auto"
        ``"auto"``, None or 0 for :func:`physical_cores`, ``"all"``, ``"logical"`` or ``"max"`` for every
        logical CPU (:func:`cpu_count`, what ``"auto"`` meant before 2026-10-27), ``-k`` for ``"auto"``
        less k, or a positive count.

    Returns
    -------
    workers : int
        The budget, at least 1.
    """
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
    """Start the thread pools and their lock afresh in a forked worker.

    A forked worker inherits the parent's thread pools without their threads, so work submitted to one
    would wait forever.  It also inherits ``_lock`` as it stood, possibly held by a parent thread that
    does not exist there.
    """
    global _lock
    _lock = threading.Lock()
    _pools.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _install_signal_reaper() -> None:
    """Install handlers that terminate the worker pools on SIGTERM and SIGHUP, as on a normal exit.

    Then ``pkill`` or a closed terminal leaves no workers behind.  It installs once, from the main
    thread, and only over the default handlers.
    """
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

#: callables the main thread runs while it waits for worker results.  The live display
#: animates the launch movie from here, as the IDL predecessor does in its bridge poll loop.
IDLE_HOOKS: list = []


def run_idle_hooks() -> None:
    """Run the idle hooks of :data:`IDLE_HOOKS`, from the main thread only, ignoring a hook that fails.
    """
    if not IDLE_HOOKS or threading.current_thread() is not threading.main_thread():
        return
    for fn in list(IDLE_HOOKS):
        try:
            fn()
        except Exception:
            pass


def pump_wait(futures, tick: float = 0.1):
    """Wait for futures, running the idle hooks every ``tick`` seconds.

    The main thread must hand the GUI event loop a turn every couple of seconds, or macOS stops
    compositing the live window.  A bare ``Future.result()`` blocks with no such window, so every place
    the main thread waits on workers goes through here, threads and :meth:`ProcessPool.map` alike.

    Parameters
    ----------
    futures : iterable of concurrent.futures.Future
        The futures.
    tick : float, default 0.1
        Period of the idle hooks, in seconds.

    Returns
    -------
    results : list
        The results, in the order the futures were given.

    Raises
    ------
    Exception
        The first future's exception, as ``result()`` raises it.
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
    """Return the process-wide pool for the per-target KLIP work, one per size, made when first asked for.

    Parameters
    ----------
    n : int
        Number of threads.

    Returns
    -------
    pool : concurrent.futures.ThreadPoolExecutor
        The pool.
    """
    with _lock:
        p = _pools.get(n)
        if p is None:
            p = _pools[n] = cf.ThreadPoolExecutor(max_workers=n, thread_name_prefix="klip-target")
        return p


@contextlib.contextmanager
def single_blas_thread(enabled: bool = True):
    """Pin BLAS and LAPACK to one thread inside a multi-threaded region.

    Without threadpoolctl it does nothing.

    Parameters
    ----------
    enabled : bool, default True
        Whether to pin.

    Yields
    ------
    None
        Control inside the region.
    """
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
    """Pin the process' BLAS and LAPACK pools to ``limit`` threads for the rest of the run.

    klip-tpe's own threads give the parallelism.

    Parameters
    ----------
    limit : int, default 1
        Number of threads.

    Returns
    -------
    applied : bool
        True when the pin is in place, False without threadpoolctl.
    """
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
    """Return the bytes of memory the kernel says are available to new work, Linux's ``MemAvailable``.

    Returns
    -------
    nbytes : int or None
        The bytes, None where ``/proc/meminfo`` does not say.
    """
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cores_text() -> str:
    """Describe the cores, such as ``96 cores (192 logical CPUs)``, or ``2 cores`` where each core has one thread.

    Returns
    -------
    text : str
        The description.
    """
    phys, log = physical_cores(), cpu_count()
    return f"{phys} cores ({log} logical CPUs)" if log != phys else f"{phys} cores"


def describe(workers: int, n_partitions: int, procs: Optional[int] = None,
             per_job: Optional[int] = None, cap: Optional[int] = None, budget: Optional[int] = None) -> str:
    """Return the run log's line about the worker budget.

    Parameters
    ----------
    workers : int
        The worker budget.
    n_partitions : int
        Number of partitions per reduction.
    procs : int or None, default None
        Number of worker processes, None for the thread-pool fallback.
    per_job : int or None, default None
        Threads per job.
    cap : int or None, default None
        Most threads a job expected to be long can get.
    budget : int or None, default None
        Most threads the running jobs use together, when the pool holds them to it.

    Returns
    -------
    line : str
        The line.
    """
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
# forked worker processes (the IDL predecessor's "bridges"): one copy of the loaded data, N processes
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

    ``daemon=True`` only covers a clean exit of the parent, whose atexit hook terminates the children.
    A parent that is SIGKILLed, or dies with the terminal, never runs that hook, and the worker then
    blocks on ``inq.get()`` forever, an invisible process holding a copy of the data until the machine
    is rebooted.  Polling ``getppid()`` catches every case, because an orphan is re-parented, to init or
    to launchd on macOS, the moment its parent dies.

    Parameters
    ----------
    ppid : int
        The parent's process id.
    period : float, default 2.0
        Period of the poll, in seconds.
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
    """Run a worker's loop, ``(job_id, pid, request)`` in and ``(job_id, result or exception)`` out.

    The worker ignores SIGINT, since the parent owns the interrupt and the workers stop when it tells
    them to, or when the watchdog sees it gone.  A None job ends the loop.

    Parameters
    ----------
    reducers : dict
        The reducers, by partition id.
    inq : multiprocessing.Queue
        Queue of the jobs.
    outq : multiprocessing.Queue
        Queue of the results.
    ppid : int or None, default None
        The parent's process id, for the watchdog (:func:`_watch_parent`).
    """
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


# ----------------------------------------------------------------------------
# the card: worker processes that reduce on a GPU
# ----------------------------------------------------------------------------
def card_slot(tag: str) -> int:
    """Return a job's place in 0 to 99 for the card's share, from its tag.

    The tag names the evaluation, the draw and the partition, and the place comes from SHA-1, not
    Python's ``hash()``, which changes from one process to the next.  Under the card's ``"job"`` rule a
    job goes to the card when its slot is below the share, so a run and its resumes send the same jobs
    there.

    Parameters
    ----------
    tag : str
        The job's tag.

    Returns
    -------
    slot : int
        The place.
    """
    return _sha1_int(tag) % 100


def _sha1_int(text: str) -> int:
    """Return the first 4 bytes of a text's SHA-1 as an integer.

    Parameters
    ----------
    text : str
        The text.

    Returns
    -------
    value : int
        The integer, from 0 to 2**32 - 1.
    """
    import hashlib
    return int.from_bytes(hashlib.sha1(str(text).encode("utf-8")).digest()[:4], "big")


#: how the card's share picks jobs (``CardSpec.rule``): ``"trial"``, the lowest-hash jobs of each
#: trial up to the share, or ``"job"``, each job by its own :func:`card_slot`, as runs did before
#: 2026-11-15
CARD_RULES = ("trial", "job")


def card_picks(tags: Sequence[str], share: float) -> set:
    """Return the jobs of one trial the card takes under the ``"trial"`` rule.

    The count is ``share`` percent of the jobs, rounded down, plus one more when the remainder is above a
    number drawn from the trial's tags, so the card's long-run share is the share asked and no trial sends
    it more than its share rounded up.  The jobs taken are those with the lowest hash of their tag.  Both
    come from SHA-1, so a run and its resumes pick the same jobs.

    Parameters
    ----------
    tags : sequence of str
        Tags of the trial's jobs the card could take, each draw's partitions.
    share : float
        Percentage of the jobs the card takes.

    Returns
    -------
    picked : set of str
        Tags of the jobs the card takes.

    Notes
    -----
    Under the ``"job"`` rule each job goes to the card when its own slot is below the share, so the number
    of a trial's jobs on the card is binomial.  With 18 jobs a trial (NEAR2's six nights, three draws) at
    12 %, about one trial in six then sends the card 4 or more jobs for its 3 workers, and the whole trial
    waits on the card's second round, the tail the NEAR2 session found (2026-11-15).
    """
    tags = sorted({str(t) for t in tags})
    if not tags or share <= 0:
        return set()
    q = min(max(float(share), 0.0), 100.0) * len(tags) / 100.0
    k = math.floor(q)
    if q - k > _sha1_int("\n".join(tags)) / 2.0 ** 32:
        k += 1
    return set(sorted(tags, key=lambda t: (_sha1_int(t), t))[:k])


def card_present() -> bool:
    """Tell whether this machine shows an NVIDIA card, without loading CUDA.

    The parent never loads it, since a CUDA context does not survive a fork.

    Returns
    -------
    present : bool
        True on Linux with ``/dev/nvidia0`` or ``/dev/nvidiactl``.
    """
    return sys.platform.startswith("linux") and any(
        os.path.exists(p) for p in ("/dev/nvidia0", "/dev/nvidiactl"))


def describe_card(summary: Optional[dict]) -> str:
    """Return one line on the card workers, for the log.

    Parameters
    ----------
    summary : dict or None
        The summary, from :meth:`ProcessPool.card_summary`.

    Returns
    -------
    line : str
        The line, "none" without a summary.
    """
    if not summary:
        return "none"
    info = summary.get("info") or {}
    name = info.get("name") or info.get("card") or info.get("device") or "the card"
    drv = info.get("driver") or info.get("kernel_driver")
    k = int(summary.get("ready") or 0)
    n = int(summary.get("workers") or 0)
    who = f"{k} of {n} workers" if k != n else f"{n} worker{'s' * (n != 1)}"
    cap = info.get("memory_cap_gb")
    dev = summary.get("device")
    return (f"{who} on {name}" + (f" (CUDA_VISIBLE_DEVICES={dev})" if dev else "")
            + (f" (driver {drv})" if drv else "")
            + f", {summary.get('share', 0):g} % of the search's injected draws"
            + (", each job picked by its own slot" if summary.get("rule") == "job" else "")
            + (f", each worker held to {cap:g} GB of its memory" if cap else ""))


class CardSpec:
    """The card workers of a :class:`ProcessPool`.

    Parameters
    ----------
    reducers : dict
        The card reducers by partition id, built in the parent and pure Python there (``from_cpu``).
        Each worker calls their ``warm()`` after the fork, then ``reduce(req)``, and the parent asks
        ``supports(req)``.
    share : float
        Percentage of the eligible jobs (``ReductionRequest.card``) the card takes.
    workers : int, default 3
        Number of card processes sharing one queue, at least 1.  At large bins a job leaves the card
        idle while the host prepares the next, so several keep it busy.
    rule : {"trial", "job"}, default "trial"
        How the share picks the jobs.  ``"trial"`` takes the lowest-hash jobs of each trial up to the share
        (:func:`card_picks`, from the request's ``card_group``), and ``"job"`` each job whose own
        :func:`card_slot` is below the share, as runs did before 2026-11-15.
    device : str or None, default None
        The card, as ``CUDA_VISIBLE_DEVICES`` takes it (an index in CUDA's order or a UUID), set in the card
        workers alone before they warm up.  None leaves the environment's.
    on_ready : callable or None, default None
        Function ``on_ready(info) -> str or None`` called once, with what the first card worker reported at
        ``warm()``, as the card comes up.  A line it returns goes to the log.

    Raises
    ------
    ValueError
        If ``rule`` is not one of :data:`CARD_RULES`.
    """

    def __init__(self, reducers: dict, share: float, workers: int = 3, rule: str = "trial",
                 device: Optional[str] = None, on_ready=None):
        self.reducers = dict(reducers)
        self.share = float(share)
        self.workers = max(int(workers), 1)
        if rule not in CARD_RULES:
            raise ValueError(f"card rule {rule!r}: one of {', '.join(CARD_RULES)}")
        self.rule = rule
        self.device = None if device is None or str(device).strip() == "" else str(device).strip()
        self.on_ready = on_ready


#: card job failures after which the card is left out for the rest of the run.  A job that
#: runs out of the worker's share of the card's memory (:class:`CardMemoryError`) does not
#: count: it goes to the CPU workers, and the card stays in the run.
CARD_MAX_ERRORS = 3

#: the share of the card's memory the card workers' pools are held to, together
#: (``GPUKLIPReducer.limit_memory``)
CARD_MEMORY_FRACTION = 0.9


class CardMemoryError(RuntimeError):
    """Error of a card job that needed more memory than its worker's share of the card, the CuPy pool's limit.

    It is plain Python, so the parent unpickles it without CuPy.

    Parameters
    ----------
    tag : str
        The job's tag.
    message : str
        The error's message.
    """

    def __init__(self, tag: str, message: str):
        super().__init__(tag, message)
        self.tag, self.message = str(tag), str(message)

    def __str__(self) -> str:
        """Return ``"tag: message"``.

        Returns
        -------
        text : str
            The text.
        """
        return f"{self.tag}: {self.message}"


def _is_out_of_memory(exc: BaseException) -> bool:
    """Tell whether an exception is CuPy's out-of-memory error, by name, or a ``MemoryError``.

    The parent need not import CuPy to tell.

    Parameters
    ----------
    exc : BaseException
        The exception.

    Returns
    -------
    oom : bool
        True for a ``MemoryError``, or an ``OutOfMemoryError`` or ``CUDARuntimeError`` about memory.
    """
    return isinstance(exc, MemoryError) or (type(exc).__name__ in ("OutOfMemoryError", "CUDARuntimeError")
                                            and "memory" in str(exc).lower())


def _card_worker_main(reducers, inq, outq, ppid: Optional[int] = None, index: int = 0, workers: int = 1,
                      device: Optional[str] = None):
    """Run a card worker's loop.

    The worker sets ``CUDA_VISIBLE_DEVICES`` to ``device`` when one is given, calls every reducer's
    ``warm()``, holds its memory pool to its share of the card (``limit_memory(workers)``), and reports
    ``("card-ready", index, info)`` or ``("card-failed", index, why)``.  It then turns each
    ``(job_id, pid, request)`` into ``("card", job_id, result or exception, seconds)``.  After every job
    the reducer's ``after_job()`` gives the pool's cache back, and a job that runs out of memory comes
    back as a :class:`CardMemoryError` naming its tag.

    Parameters
    ----------
    reducers : dict
        The card reducers, by partition id.
    inq : multiprocessing.Queue
        Queue of the card's jobs.
    outq : multiprocessing.Queue
        Queue of the results, shared with the CPU workers.
    ppid : int or None, default None
        The parent's process id, for the watchdog.
    index : int, default 0
        The worker's index.
    workers : int, default 1
        Number of card workers sharing the card's memory.
    device : str or None, default None
        The card, as ``CUDA_VISIBLE_DEVICES`` takes it.  None leaves the environment's.
    """
    pin_blas()
    try:
        import signal as _sig
        _sig.signal(_sig.SIGINT, _sig.SIG_IGN)
    except Exception:
        pass
    if ppid:
        _watch_parent(ppid)
    if device is not None:
        # this process only, before CUDA is loaded: the parent and the CPU workers never load it
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    try:
        info = {}
        for pid, r in reducers.items():
            w = getattr(r, "warm", None)
            info[str(pid)] = w() if callable(w) else {}
        # one pool per process: the first reducer that can hold it does so for all of them
        for r in reducers.values():
            lim = getattr(r, "limit_memory", None)
            if callable(lim):
                cap = lim(workers, CARD_MEMORY_FRACTION)
                if cap is not None:
                    for v in info.values():
                        v["memory_cap_gb"] = round(cap / 1e9, 2)
                break
        outq.put(("card-ready", index, info))
    except BaseException as exc:          # noqa: BLE001 - no card, no CuPy, no memory: CPU-only
        try:
            outq.put(("card-failed", index, f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        return
    while True:
        item = inq.get()
        if item is None:
            return
        job_id, pid, req = item
        t = time.perf_counter()
        red = reducers[pid]
        try:
            res = red.reduce(req)
            meta = getattr(res, "meta", None)
            if isinstance(meta, dict):
                meta["card"] = True
        except BaseException as exc:      # noqa: BLE001
            tag = str(getattr(req, "tag", "?"))
            res = CardMemoryError(tag, f"{type(exc).__name__}: {exc}") if _is_out_of_memory(exc) else \
                RuntimeError(f"{tag}: {type(exc).__name__}: {exc}")
        after = getattr(red, "after_job", None)
        if callable(after):
            try:
                after()
            except Exception:             # noqa: BLE001 - the next job finds out
                pass
        dt = time.perf_counter() - t
        try:
            outq.put(("card", job_id, res, dt))
        except Exception as exc:
            outq.put(("card", job_id, RuntimeError(f"card result not picklable: {exc!r}"), dt))


class ProcessPool:
    """Forked worker processes that each hold the reducers, copy-on-write, and run their ``reduce`` jobs.

    The fork happens once, after the data are loaded, so the cubes are shared with the parent and never
    pickled, and only the requests and the small results cross the pipes.

    A :class:`CardSpec` adds card workers with a queue of their own.  A job goes to them when its
    request is marked ``card``, it is in the card's share (:meth:`_in_share`), and the card reducer
    ``supports`` it, so where a job runs does not depend on timing.  When the card cannot start, a worker
    dies, or :data:`CARD_MAX_ERRORS` jobs fail on it, the jobs it holds go back to the CPU workers and the
    pool goes on without it.

    Parameters
    ----------
    reducers : dict
        The reducers, by partition id.
    n : int
        Number of CPU workers.
    log : callable or None, default None
        Function that takes each line of the log.  None logs nothing.
    budget : int or None, default None
        Threads the running jobs may use together (:meth:`_fit`), at least ``n``.  None sets no limit,
        the default, and what :class:`~klip_tpe.reducer.PartitionedReducer` passes.
    card : CardSpec or None, default None
        The card workers.

    Attributes
    ----------
    procs : list of multiprocessing.Process
        The CPU workers.
    card_procs : list of multiprocessing.Process
        The card workers.
    card_info : dict or None
        What the first card worker reported at ``warm()``.
    card_busy : float
        Seconds the card workers spent in jobs.
    card_jobs : int
        Number of jobs the card returned.
    card_marked : int
        Jobs marked for the card and in its share, while it was up.
    card_refused : dict of str to int
        Of those, the ones its reducer turned away, by reason.
    card_memory : int
        Jobs sent back to the CPU for want of the worker's memory.
    card_why : str or None
        Why the card went away.

    Raises
    ------
    RuntimeError
        If the fork start method is not available, which the caller answers with threads.
    """

    def __init__(self, reducers: dict, n: int, log=None, budget: Optional[int] = None,
                 card: Optional[CardSpec] = None):
        self.n = int(n)
        self.log = log or (lambda s: None)
        self.reducers = reducers
        #: the threads the running jobs may use together, or None for no limit (the default,
        #: and what :class:`~klip_tpe.reducer.PartitionedReducer` passes, see its
        #: ``cap_threads``).  A job is sent with its request's ``threads`` cut to what the
        #: running jobs leave, keeping one for each idle worker, so the jobs in flight never use
        #: more than this (at least one thread a job, so never less than ``n``).  The images do
        #: not depend on the count.
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
        # the card workers: their own queue, the same result queue
        self.card = card if (card is not None and card.reducers and card.share > 0) else None
        self.card_procs: list = []
        self._card_ok = self.card is not None
        self._card_out: dict = {}          # job id -> (pid, request) sent to the card, not back yet
        self._card_ready = 0
        self._card_gone: set = set()       # card workers that reported they could not start
        self._card_errors = 0
        self.card_info: Optional[dict] = None
        self.card_busy = 0.0               # seconds the card workers spent in jobs
        self.card_jobs = 0
        self.card_marked = 0               # jobs marked for the card and in its share, while it was up
        self.card_refused: Dict[str, int] = {}   # of those, the ones supports() turned away, by reason
        self.card_memory = 0               # jobs sent back to the CPU for want of the worker's memory
        self.card_why: Optional[str] = None
        self._card_t0 = time.time()
        self._notes: list = []             # log lines for the caller's thread (take_notes)
        if self.card is not None:
            self.card_inq = ctx.Queue()
            self.card_procs = [ctx.Process(target=_card_worker_main,
                                           args=(self.card.reducers, self.card_inq, self.outq, me, i,
                                                 self.card.workers, getattr(self.card, "device", None)),
                                           name=f"klip-card-{i}", daemon=True) for i in range(self.card.workers)]
        for p in self.procs + self.card_procs:
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
        """Make the two queues, retrying the import race the comment above this section describes.

        The pre-import should make the race impossible, and this is the second guard, since losing it costs
        a whole run in thread mode.

        Parameters
        ----------
        ctx : multiprocessing.context.BaseContext
            The fork context.
        tries : int, default 3
            Number of tries.

        Returns
        -------
        inq : multiprocessing.Queue
            Queue of the jobs.
        outq : multiprocessing.Queue
            Queue of the results.

        Raises
        ------
        ImportError
            If every try met the race.
        """
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
        """Tell whether every CPU worker is alive, which the card workers are not needed for.

        Returns
        -------
        alive : bool
            True when every CPU worker is alive.
        """
        return all(p.is_alive() for p in self.procs)

    # -- the card ----------------------------------------------------------------------
    def _in_share(self, pid, req) -> bool:
        """Tell whether a job marked for the card is in its share, by the card's rule.

        Under the ``"trial"`` rule, a request with a ``card_group`` is in the share when :func:`card_picks`
        takes its tag from the group's jobs on partitions the card can reduce.  Otherwise, and under the
        ``"job"`` rule, it is in the share when its :func:`card_slot` is below the share.

        Parameters
        ----------
        pid : object
            The job's partition.
        req : ReductionRequest
            The job's request.

        Returns
        -------
        in_share : bool
            True when the job is in the share.
        """
        tag = str(getattr(req, "tag", ""))
        group = getattr(req, "card_group", None)
        if getattr(self.card, "rule", "trial") == "trial" and group:
            tags = [str(t) for t, p in group if p in self.card.reducers]
            return tag in card_picks(tags, self.card.share)
        return card_slot(tag) < self.card.share

    def _to_card(self, pid, req) -> bool:
        """Tell whether a job goes to the card, with the lock held.

        A job marked for it and in its share (:meth:`_in_share`) counts in ``card_marked``, and one its
        reducer turns away in ``card_refused``, under the reason (``refusal(req)`` where the reducer has it,
        else ``supports``).

        Parameters
        ----------
        pid : object
            The job's partition.
        req : ReductionRequest
            The job's request.

        Returns
        -------
        to_card : bool
            True when the job goes to the card.
        """
        if not self._card_ok or not getattr(req, "card", False) or pid not in self.card.reducers:
            return False
        if not self._in_share(pid, req):
            return False
        self.card_marked += 1
        red = self.card.reducers[pid]
        try:
            why = red.refusal(req) if hasattr(red, "refusal") else (None if red.supports(req) else "not supported")
        except Exception as exc:
            why = f"supports() raised {type(exc).__name__}"
        if why is not None:
            self.card_refused[str(why)] = self.card_refused.get(str(why), 0) + 1
            return False
        return True

    def _card_down(self, why: str) -> None:
        """Leave the card out from now on, with the lock held.

        Its jobs go back to the CPU workers, ahead of the rest, and its workers are told to stop.

        Parameters
        ----------
        why : str
            Why, for the log and ``card_why``.
        """
        if not self._card_ok:
            return
        self._card_ok = False
        self.card_why = why
        back = list(self._card_out.items())
        self._card_out.clear()
        for jid, (pid, req) in back:
            if jid in self._futures:
                heapq.heappush(self._waiting, (float("-inf"), jid, pid, req))
        self._notes.append(f"  card: {why}, so the run goes on without it"
                           + (f" ({len(back)} of its jobs go to the CPU workers)" if back else ""))
        for _ in self.card_procs:
            try:
                self.card_inq.put(None)
            except Exception:
                pass
        self._dispatch()

    def _check_card(self) -> None:
        """Leave the card out when a card worker died without saying so, the job it held lost with it.

        The collector's thread calls it once a second, and every job the card holds goes back to the CPU.
        """
        with self._lock:
            if not self._card_ok:
                return
            dead = [p.name for i, p in enumerate(self.card_procs) if i not in self._card_gone and not p.is_alive()]
            if dead:
                self._card_down(f"card worker(s) {', '.join(dead)} died")

    def take_notes(self) -> list:
        """Return the pool's log lines since the last call, such as the card coming up or going away.

        The caller writes them from its own thread.

        Returns
        -------
        notes : list of str
            The lines.
        """
        with self._lock:
            out, self._notes = self._notes, []
        return out

    def card_summary(self) -> Optional[dict]:
        """Summarize the card: its share, workers, state, report at ``warm()``, jobs and busy fraction.

        Returns
        -------
        summary : dict or None
            ``share``, ``workers``, ``rule``, ``device``, ``up``, ``ready``, ``why_down``, ``info``, ``jobs``,
            ``busy_s``, ``busy_fraction`` (of the time since the pool started), ``marked``, ``refused``,
            ``memory`` and ``t`` (the time of the summary, for a fraction over a stretch of the run), None
            without a card.
        """
        if self.card is None:
            return None
        now = time.time()
        wall = max(now - self._card_t0, 1e-9)
        return {"share": self.card.share, "workers": self.card.workers, "rule": getattr(self.card, "rule", "trial"),
                "device": getattr(self.card, "device", None), "up": bool(self._card_ok),
                "ready": self._card_ready, "why_down": self.card_why, "info": self.card_info,
                "jobs": self.card_jobs, "busy_s": round(self.card_busy, 3),
                "busy_fraction": round(self.card_busy / (wall * self.card.workers), 4),
                "marked": self.card_marked, "refused": dict(self.card_refused), "memory": self.card_memory,
                "t": now}

    def _card_result(self, item) -> None:
        """Take a card worker's message, in the collector's thread.

        A ready or failed worker is counted, and the card goes down when none could start.  A result
        settles its job's future, while a job that ran out of the worker's memory, or failed, goes back to
        the CPU workers, :data:`CARD_MAX_ERRORS` failures taking the card down.

        Parameters
        ----------
        item : tuple
            The message, ``("card-ready", index, info)``, ``("card-failed", index, why)`` or
            ``("card", job_id, result, seconds)``.
        """
        kind = item[0]
        with self._lock:
            if kind in ("card-ready", "card-failed"):
                if kind == "card-ready":
                    self._card_ready += 1
                    if self.card_info is None:
                        self.card_info = dict(next(iter(item[2].values()), {}) or {}) if item[2] else {}
                        cb = getattr(self.card, "on_ready", None)
                        if callable(cb):
                            try:
                                line = cb(dict(self.card_info))
                            except Exception as exc:  # noqa: BLE001 - a record for the log, never fatal
                                line = f"  card: on_ready failed ({exc!r})"
                            if line:
                                self._notes.append(str(line))
                else:
                    # a worker that cannot warm up never takes a job, so the others serve the
                    # card's queue alone; with none left the card is out
                    self._card_gone.add(item[1])
                    if len(self._card_gone) == len(self.card_procs):
                        self._card_down(f"it did not start ({item[2]})")
                        return
                    self._notes.append(f"  card: worker {item[1]} did not start ({item[2]})")
                if self._card_ok and self._card_ready + len(self._card_gone) == len(self.card_procs):
                    self._notes.append(f"  card: {describe_card(self.card_summary())}")
                return
            _, jid, res, dt = item
            self.card_busy += float(dt)
            sent = self._card_out.pop(jid, None)
            f = self._futures.get(jid)
            if isinstance(res, CardMemoryError) and sent is not None and f is not None:
                # more than the worker's share of the card: a property of the job, so it goes
                # to the CPU and the card stays in the run
                self.card_memory += 1
                heapq.heappush(self._waiting, (float("-inf"), jid, sent[0], sent[1]))
                if self.card_memory <= 5:
                    self._notes.append(f"  card: {res.tag} needed more than its worker's share of the card's "
                                       f"memory ({res.message}); reduced on the CPU instead"
                                       + ("  (further ones are counted, not listed)" if self.card_memory == 5 else ""))
                self._dispatch()
                return
            if isinstance(res, BaseException) and sent is not None and f is not None:
                # a job the card failed is reduced on the CPU instead
                self._card_errors += 1
                heapq.heappush(self._waiting, (float("-inf"), jid, sent[0], sent[1]))
                if self._card_errors >= CARD_MAX_ERRORS:
                    self._card_down(f"{self._card_errors} jobs failed on it, the last with {res!r}")
                else:
                    self._notes.append(f"  card: a job failed on it ({res!r}); reduced on the CPU instead")
                self._dispatch()
                return
            if sent is None or f is None:
                return                         # already reduced on the CPU
            self._futures.pop(jid, None)
            self.card_jobs += 1
        if not f.done():
            f.set_result(res)

    def memory(self) -> Optional[dict]:
        """Return each worker's peak resident set and the pool's proportional set now, on Linux.

        Returns
        -------
        memory : dict or None
            ``peak``, each worker's ``VmHWM`` in bytes, which counts in full the copy-on-write data it
            shares with the parent and the other workers, and ``pss``, the pool's ``Pss`` in bytes, every
            shared page split between the processes using it.  None elsewhere than Linux.
        """
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
        """Run jobs and return their results in order.

        It is thread-safe, several callers mapping at once, and waiting from the main thread services the
        live window (:func:`pump_wait`).

        Parameters
        ----------
        jobs : sequence of tuple
            ``(pid, request)`` per job.
        keys : sequence of float or None, default None
            One sort key per job.  A worker that comes free takes the waiting job with the smallest key, so a
            caller can send long jobs first.  Without keys, and between equal keys, jobs go in the order
            they came.

        Returns
        -------
        results : list
            The results, in the order of ``jobs``.

        Raises
        ------
        RuntimeError
            If the pool is closed.
        Exception
            A failed job's exception.
        """
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
                if self.card is not None and self._to_card(pid, req):
                    self._card_out[jid] = (pid, req)
                    self.card_inq.put((jid, pid, req))
                    continue
                key = 0.0 if keys is None else float(keys[k])
                heapq.heappush(self._waiting, (key, jid, pid, req))
            self._dispatch()
        return pump_wait(futs)

    def _dispatch(self) -> None:
        """Send waiting jobs, smallest key first, until every worker has one, with the lock held.

        Only as many jobs as workers are ever in the queue, so the order is decided here, when a worker
        comes free, not when a job was submitted, and so is a job's share of the thread budget
        (:meth:`_fit`).
        """
        while self._waiting and self._busy < self.n:
            _, jid, pid, req = heapq.heappop(self._waiting)
            self.inq.put((jid, pid, self._fit(jid, req)))
            self._busy += 1

    def _fit(self, jid: int, req):
        """Return a request with its ``threads`` cut to the budget's room, with the lock held.

        The room is the budget less the running jobs' threads and one for each worker still idle after this
        one.  That keeps the threads in flight within the budget at every moment, while every worker can
        still take a job.  Without a budget, or for a request without ``threads``, the request goes as it
        is.

        Parameters
        ----------
        jid : int
            The job's id.
        req : ReductionRequest
            The request.

        Returns
        -------
        req : ReductionRequest
            The request, a copy with fewer threads where it was cut.

        Notes
        -----
        The by-cost plan (:meth:`klip_tpe.reducer.PartitionedReducer._plan`) gives a long job up to 4x the
        base and a short one 1, so over the jobs submitted the mean is near the base.  The jobs running at
        a given moment are not that mean: the long ones run longest, so they and their threads fill the
        pool.  On patrick, 48 jobs at a base of 4 ran at a load average of 234 on 192 CPUs, and at a base
        of 2 at 134 on a budget of 96.

        The cut is not used by default.  Measured on patrick it starved the heavy jobs: with a budget of 96
        for 48 workers, the reserve for the idle workers leaves room for only a few jobs above one thread,
        so the heavy nights ran on 1 and every trial waited on them (0.339 s a reduction against 0.205 s
        uncut, the pool 25 % busy).
        """
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
        """Run the collector thread, which settles each result's future.

        Every outstanding job fails when a worker has died, since it can never come back, or when the pool
        closes.  A card worker's message goes to :meth:`_card_result`.
        """
        last_check = time.time()
        while not self._closed:
            if self.card is not None and time.time() - last_check > 1.0:
                last_check = time.time()
                self._check_card()
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
            if isinstance(item, tuple) and item and isinstance(item[0], str):
                try:
                    self._card_result(item)    # a card worker's message
                except Exception:
                    pass
                continue
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
        """Fail every outstanding job, and clear the waiting and the card's jobs.

        Parameters
        ----------
        make_exc : callable
            Function ``n -> exception``, given the number of outstanding jobs.
        """
        with self._lock:
            futs, self._futures = self._futures, {}
            self._waiting = []
            self._threads, self._threads_running = {}, 0
            self._card_out = {}
        for f in futs.values():
            if not f.done():
                f.set_exception(make_exc(len(futs)))

    def close(self):
        """Close the pool: stop the workers, terminate one still alive after 5 s, and wake the collector.
        """
        self._closed = True
        for _ in self.procs:
            try:
                self.inq.put(None)
            except Exception:
                pass
        for _ in self.card_procs:
            try:
                self.card_inq.put(None)
            except Exception:
                pass
        for p in self.procs + self.card_procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        try:
            self.outq.put(None)                # wakes the collector
        except Exception:
            pass

    def __del__(self):
        """Close the pool, ignoring any error.
        """
        try:
            self.close()
        except Exception:
            pass



# ----------------------------------------------------------------------------
# arrays made once per reduction and read by every worker: shared memory, not pickles
# ----------------------------------------------------------------------------
class SharedArrays:
    """Named numpy arrays in one shared-memory block, for the worker processes to read in place.

    The SDI mode prepares every wavelength group's frames once per reduction and hands all of them to
    each group's job.  Through the job queue that is tens of MB pickled per job, hundreds per
    reduction, serialized by the parent.  Here the parent writes them once into a
    :class:`multiprocessing.shared_memory.SharedMemory` block, and the jobs carry only this descriptor:
    the block's name, each array's offset, shape and dtype, and ``extra``, small picklable things that
    go with them.  A worker attaches (:meth:`attach`), takes views, and closes its handle when its
    copies are made, and the parent releases the block (:meth:`release`) when the reduction is done.
    It pickles without the handle.  With threads instead of processes none of this is needed, since the
    arrays are passed as they are.

    Parameters
    ----------
    name : str
        Name of the block.
    size : int
        Size of the block, in bytes.
    index : dict
        Each array's ``(offset, shape, dtype)``, by name.
    extra : dict or None, default None
        Small picklable things that go with the arrays.
    """

    def __init__(self, name: str, size: int, index: Dict[str, Tuple[int, Tuple[int, ...], str]],
                 extra: Optional[Dict[str, Any]] = None):
        self.name, self.size, self.index = str(name), int(size), dict(index)
        self.extra = dict(extra or {})
        self._shm = None

    @classmethod
    def create(cls, arrays: Dict[str, np.ndarray], extra: Optional[Dict[str, Any]] = None) -> "SharedArrays":
        """Write arrays into a new block, the parent's side.

        Each array starts on a 64-byte boundary.

        Parameters
        ----------
        arrays : dict of str to ndarray
            The arrays, by name.
        extra : dict or None, default None
            Small picklable things that go with them.

        Returns
        -------
        shared : SharedArrays
            The descriptor, holding the block's handle.

        Raises
        ------
        Exception
            If shared memory is not available, so a caller can fall back to passing the arrays themselves.
        """
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
        """Attach to the block, a worker's side.

        Attaching would register the block with the resource tracker as if this process owned it, and the
        tracker would then unlink it at exit or complain that it could not.  The parent owns the block, so
        the registration is stood down for the attach.

        Returns
        -------
        arrays : dict of str to ndarray
            The arrays, as views into the block.
        closer : callable
            Function that drops this process's handle once the views are no longer needed, after which a
            view's contents are undefined.
        """
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
        """Close and unlink the block, the parent's side.
        """
        shm, self._shm = self._shm, None
        if shm is None:
            return
        for f in (shm.close, shm.unlink):
            try:
                f()
            except Exception:
                pass

    def __getstate__(self):
        """Return the descriptor's state for pickling, without the block's handle.

        Returns
        -------
        state : dict
            The attributes, ``_shm`` None.
        """
        d = dict(self.__dict__)
        d["_shm"] = None
        return d

    def __del__(self):
        """Release the block, ignoring any error.
        """
        try:
            self.release()
        except Exception:
            pass
