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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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


def card_assign(tags: Sequence[str], shares: Sequence[float]) -> Dict[str, int]:
    """Return the jobs of one trial each card takes under the ``"trial"`` rule.

    Each card's count is its share of the jobs, rounded down, plus one more when the remainder is above a
    number drawn from the trial's tags, so each card's long-run share is the share asked and no trial sends
    a card more than its share rounded up.  The jobs go in the order of the hash of their tag, the lowest
    first: the first card takes the first of them, the second card the next, and so on.  Both come from
    SHA-1, so a run and its resumes give each card the same jobs.

    Parameters
    ----------
    tags : sequence of str
        Tags of the trial's jobs the cards could take, each draw's partitions.
    shares : sequence of float
        Percentage of the jobs each card takes, one per card.

    Returns
    -------
    assignment : dict of str to int
        The card of each job a card takes, by tag, the cards counted from 0.
    """
    tags = sorted({str(t) for t in tags})
    if not tags:
        return {}
    joined = "\n".join(tags)
    order = sorted(tags, key=lambda t: (_sha1_int(t), t))
    out: Dict[str, int] = {}
    start = 0
    for i, share in enumerate(shares):
        if share <= 0:
            continue
        q = min(max(float(share), 0.0), 100.0) * len(tags) / 100.0
        k = math.floor(q)
        if q - k > _sha1_int(joined if i == 0 else f"{i}\n{joined}") / 2.0 ** 32:
            k += 1
        k = min(k, len(order) - start)
        for t in order[start:start + k]:
            out[t] = i
        start += k
    return out


def card_picks(tags: Sequence[str], share: float) -> set:
    """Return the jobs of one trial the card takes under the ``"trial"`` rule, with one card.

    It is :func:`card_assign` with one card.

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
    return set(card_assign(tags, [share]))


#: the least share, in percent, the automatic share gives a card, so that its speed stays measured
CARD_SHARE_FLOOR = 2.0

#: the most share, in percent of the injected jobs, the automatic share gives the cards together
CARD_SHARE_CAP = 95.0


def card_balance(cpu: Tuple[int, float, int], cards: Sequence[Optional[Tuple[int, float, int]]], eligible: int,
                 shares: Sequence[float], damping: float = 0.5, dead_band: float = 0.05
                 ) -> Tuple[Optional[List[float]], Dict[str, Any]]:
    """Return the shares at which the cards and the CPU workers would finish a stretch of trials together.

    Each pool's rate is its workers over its mean job time, ``n / (busy / jobs)``, the jobs it would finish
    a second when every worker had one.  Card ``i`` should then take ``mu_i / (mu_cpu + sum(mu))`` of all the
    jobs, the clean ones among them, which only the CPU workers reduce, and its share is that number of
    jobs over the jobs it could have taken.  The share moves to it by a power ``damping`` of the ratio,
    at most a factor of 2 a step, and stays where it is when the ratio is within ``dead_band`` of 1.

    Parameters
    ----------
    cpu : tuple of (int, float, int)
        The CPU workers' jobs, their seconds in jobs and their number, over the stretch.
    cards : sequence of tuple of (int, float, int) or None
        The same for each card, None for a card out of the run.
    eligible : int
        Number of the stretch's jobs a card could have taken, the injected draws' partitions.
    shares : sequence of float
        Each card's share over the stretch, in percent.
    damping : float, default 0.5
        Power of the ratio of the balanced share to the current one that each step applies.
    dead_band : float, default 0.05
        Relative change below which a share stays.

    Returns
    -------
    new : list of float or None
        Each card's new share, from :data:`CARD_SHARE_FLOOR` up, together at most :data:`CARD_SHARE_CAP`,
        None when the stretch measured too few jobs (fewer than 10 on the CPU, 10 eligible, or 3 on every
        card).
    measured : dict
        ``cpu_rate`` and ``card_rates`` (jobs a second, None for a card not measured), ``total``, ``eligible``
        and ``balanced`` (each card's balanced share, None where not measured).
    """
    jc, bc, nc = int(cpu[0]), float(cpu[1]), max(int(cpu[2]), 1)
    measured: Dict[str, Any] = {"cpu_rate": None, "card_rates": [None] * len(shares), "total": None,
                                "eligible": int(eligible), "balanced": [None] * len(shares)}
    if jc < 10 or bc <= 0 or eligible < 10:
        return None, measured
    mu_c = nc * jc / bc
    rates: List[Optional[float]] = []
    total = jc
    for c in cards:
        if c is None:
            rates.append(None)
            continue
        total += int(c[0])
        rates.append(max(int(c[2]), 1) * int(c[0]) / float(c[1]) if int(c[0]) >= 3 and float(c[1]) > 0 else None)
    measured.update(cpu_rate=mu_c, card_rates=rates, total=total)
    if all(r is None for r in rates):
        return None, measured
    known = sum(r for r in rates if r is not None)
    new: List[float] = []
    for i, (r, cur) in enumerate(zip(rates, shares)):
        cur = float(cur)
        if r is None:
            new.append(cur)
            continue
        want = 100.0 * total * r / (mu_c + known) / float(eligible)
        measured["balanced"][i] = want
        ratio = min(max(want / max(cur, 1e-9), 0.5), 2.0)
        if abs(ratio - 1.0) <= dead_band:
            new.append(cur)
            continue
        new.append(min(max(round(cur * ratio ** float(damping), 1), CARD_SHARE_FLOOR), 100.0))
    if sum(new) > CARD_SHARE_CAP:
        new = [round(v * CARD_SHARE_CAP / sum(new), 1) for v in new]
    return new, measured


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
        The summary, from :meth:`ProcessPool.card_summary`, or one card's of its ``cards``.  A summary of
        several cards gives each card's part.

    Returns
    -------
    line : str
        The line, "none" without a summary.
    """
    if not summary:
        return "none"
    if len(summary.get("cards") or []) > 1:
        return " and ".join(f"card {c.get('index', k)}: {describe_card(dict(c, rule=summary.get('rule')))}"
                            for k, c in enumerate(summary["cards"]))
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
    """The card workers of a :class:`ProcessPool`, on one card or several.

    Parameters
    ----------
    reducers : dict
        The card reducers by partition id, built in the parent and pure Python there (``from_cpu``).
        Each worker calls their ``warm()`` after the fork, then ``reduce(req)``, and the parent asks
        ``supports(req)``.
    share : float or sequence of float
        Percentage of the eligible jobs (``ReductionRequest.card``) each card takes, one number for every
        card or one per card.  A request's ``card_shares`` replaces it for that request's trial.
    workers : int, default 3
        Number of worker processes on each card, sharing its queue, at least 1.  At large bins a job
        leaves the card idle while the host prepares the next, so several keep it busy.
    rule : {"trial", "job"}, default "trial"
        How the shares pick the jobs.  ``"trial"`` takes the lowest-hash jobs of each trial up to the
        shares (:func:`card_assign`, from the request's ``card_group``), and ``"job"`` each job by its own
        :func:`card_slot`, as runs did before 2026-11-15.
    device : str or sequence of str or None, default None
        The card, or one per card, as ``CUDA_VISIBLE_DEVICES`` takes it (an index in CUDA's order or a
        UUID), set in each card's workers alone before they warm up.  None leaves the environment's,
        one card.
    on_ready : callable or None, default None
        Function ``on_ready(info) -> str or None`` called once a card, as it comes up, with what its first
        worker reported at ``warm()`` and ``card_index``.  A line it returns goes to the log.

    Attributes
    ----------
    devices : list of str or None
        Each card's device.
    shares : list of float
        Each card's share.
    share : float
        The shares together.
    device : str or None
        The first card's device.

    Raises
    ------
    ValueError
        If ``rule`` is not one of :data:`CARD_RULES`, or ``share`` gives another number of cards than
        ``device``.
    """

    def __init__(self, reducers: dict, share, workers: int = 3, rule: str = "trial", device=None, on_ready=None):
        self.reducers = dict(reducers)
        devs = list(device) if isinstance(device, (list, tuple)) else [device]
        self.devices = [None if d is None or str(d).strip() == "" else str(d).strip() for d in devs] or [None]
        sh = list(share) if isinstance(share, (list, tuple)) else [share] * len(self.devices)
        if len(sh) != len(self.devices):
            raise ValueError(f"{len(sh)} card shares for {len(self.devices)} cards")
        self.shares = [float(v) for v in sh]
        self.share = float(sum(self.shares))
        self.device = self.devices[0]
        self.workers = max(int(workers), 1)
        if rule not in CARD_RULES:
            raise ValueError(f"card rule {rule!r}: one of {', '.join(CARD_RULES)}")
        self.rule = rule
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
                      device: Optional[str] = None, card: int = 0):
    """Run a card worker's loop.

    The worker sets ``CUDA_VISIBLE_DEVICES`` to ``device`` when one is given, calls every reducer's
    ``warm()``, holds its memory pool to its share of the card (``limit_memory(workers)``), and reports
    ``("card-ready", index, info)`` or ``("card-failed", index, why)``.  It then turns each
    ``(job_id, pid, request)`` into ``("card", job_id, result or exception, seconds, index)``, the result's
    ``meta`` marked with ``card`` and ``card_index``.  After every job the reducer's ``after_job()`` gives
    the pool's cache back, and a job that runs out of memory comes back as a :class:`CardMemoryError`
    naming its tag.

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
        The worker's index among every card's workers.
    workers : int, default 1
        Number of workers sharing its card's memory.
    device : str or None, default None
        The card, as ``CUDA_VISIBLE_DEVICES`` takes it.  None leaves the environment's.
    card : int, default 0
        The card's index.
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
                meta["card_index"] = int(card)
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
            outq.put(("card", job_id, res, dt, index))
        except Exception as exc:
            outq.put(("card", job_id, RuntimeError(f"card result not picklable: {exc!r}"), dt, index))


class _Card:
    """One card of a :class:`ProcessPool`: its workers, its queue and its counts.

    Parameters
    ----------
    index : int
        The card's place among the pool's cards, from 0.
    device : str or None
        The card, as ``CUDA_VISIBLE_DEVICES`` takes it.
    share : float
        The card's share of a request that gives none, in percent.
    workers : int
        Number of its worker processes.
    name : str
        What the log calls it, "card", or "card 1" among several.

    Attributes
    ----------
    inq : multiprocessing.Queue or None
        Queue of its jobs.
    procs : list of multiprocessing.Process
        Its workers.
    ok : bool
        Whether it is in the run.
    ready : int
        Number of its workers that warmed up.
    gone : set of int
        Its workers that could not start, by their place among its workers.
    errors : int
        Number of jobs that failed on it.
    info : dict or None
        What its first worker reported at ``warm()``.
    busy : float
        Seconds its workers spent in jobs.
    jobs : int
        Number of jobs it returned.
    marked : int
        Jobs given to it while it was up.
    refused : dict of str to int
        Of those, the ones its reducer turned away, by reason.
    memory : int
        Jobs sent back to the CPU for want of a worker's memory.
    why : str or None
        Why it went away.
    out : dict
        Its jobs not back yet, job id to ``(pid, request)``.
    """

    def __init__(self, index: int, device: Optional[str], share: float, workers: int, name: str):
        self.index, self.device, self.share, self.workers, self.name = index, device, float(share), workers, name
        self.inq = None
        self.procs: list = []
        self.ok = True
        self.ready = 0
        self.gone: set = set()
        self.errors = 0
        self.info: Optional[dict] = None
        self.busy = 0.0
        self.jobs = 0
        self.marked = 0
        self.refused: Dict[str, int] = {}
        self.memory = 0
        self.why: Optional[str] = None
        self.out: dict = {}

    def summary(self, wall: float) -> dict:
        """Return the card's part of :meth:`ProcessPool.card_summary`.

        Parameters
        ----------
        wall : float
            Seconds since the pool started.

        Returns
        -------
        summary : dict
            ``index``, ``device``, ``share``, ``workers``, ``up``, ``ready``, ``why_down``, ``info``, ``jobs``,
            ``busy_s``, ``busy_fraction``, ``marked``, ``refused`` and ``memory``.
        """
        return {"index": self.index, "device": self.device, "share": self.share, "workers": self.workers,
                "up": bool(self.ok), "ready": self.ready, "why_down": self.why, "info": self.info,
                "jobs": self.jobs, "busy_s": round(self.busy, 3),
                "busy_fraction": round(self.busy / (max(wall, 1e-9) * self.workers), 4),
                "marked": self.marked, "refused": dict(self.refused), "memory": self.memory}


class ProcessPool:
    """Forked worker processes that each hold the reducers, copy-on-write, and run their ``reduce`` jobs.

    The fork happens once, after the data are loaded, so the cubes are shared with the parent and never
    pickled, and only the requests and the small results cross the pipes.

    A :class:`CardSpec` adds card workers, on one card or several, each card with a queue of its own.  A
    job goes to a card when its request is marked ``card``, the card's share picks it (:meth:`_card_pick`),
    and the card reducer ``supports`` it, so where a job runs does not depend on timing.  When a card
    cannot start, a worker of it dies, or :data:`CARD_MAX_ERRORS` jobs fail on it, the jobs it holds go
    back to the CPU workers and the pool goes on without it.  The pool times the CPU workers' jobs as well
    as the cards', for the automatic share (:func:`card_balance`).

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
        Every card's workers.
    cpu_busy : float
        Seconds the CPU workers spent in jobs.
    cpu_jobs : int
        Number of jobs the CPU workers returned.
    card_eligible : int
        Jobs a card could have taken, marked ``card`` on a partition the cards reduce.

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
        # the card workers: a queue for each card, the same result queue
        shares = list(getattr(card, "shares", None) or [getattr(card, "share", 0.0)]) if card is not None else []
        self.card = card if (card is not None and card.reducers and max(shares) > 0) else None
        self.card_procs: list = []
        self._cards: List[_Card] = []
        self._card_worker: dict = {}       # worker index -> (card, its place among the card's workers)
        self.card_eligible = 0             # jobs a card could have taken
        self.cpu_busy = 0.0                # seconds the CPU workers spent in jobs
        self.cpu_jobs = 0
        self._sent: dict = {}              # CPU job id -> when it went to a worker
        self._card_t0 = time.time()
        self._notes: list = []             # log lines for the caller's thread (take_notes)
        if self.card is not None:
            devs = list(getattr(self.card, "devices", None) or [getattr(self.card, "device", None)])
            g = 0
            for k, (dev, sh) in enumerate(zip(devs, shares)):
                c = _Card(k, dev, sh, self.card.workers, "card" if len(devs) == 1 else f"card {k}")
                c.inq = ctx.Queue()
                for j in range(self.card.workers):
                    c.procs.append(ctx.Process(target=_card_worker_main,
                                               args=(self.card.reducers, c.inq, self.outq, me, g, self.card.workers,
                                                     dev, k),
                                               name=f"klip-card-{g}", daemon=True))
                    self._card_worker[g] = (c, j)
                    g += 1
                self._cards.append(c)
                self.card_procs.extend(c.procs)
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
    def _card_pick(self, pid, req) -> Optional[int]:
        """Return the card a job marked for the cards goes to, by the cards' rule, or None for the CPU.

        The shares are the request's ``card_shares``, when it gives one for each card, else the spec's.
        Under the ``"trial"`` rule, a request with a ``card_group`` goes where :func:`card_assign` puts its
        tag among the group's jobs on partitions the cards reduce.  Otherwise, and under the ``"job"`` rule,
        its :func:`card_slot` falls in one card's stretch of 0 to 99, the first card's from 0 up to its
        share and each next card's after it.

        Parameters
        ----------
        pid : object
            The job's partition.
        req : ReductionRequest
            The job's request.

        Returns
        -------
        card : int or None
            The card's index, None when no card's share takes the job.
        """
        tag = str(getattr(req, "tag", ""))
        shares = getattr(req, "card_shares", None)
        shares = [float(v) for v in shares] if shares and len(shares) == len(self._cards) else \
            [c.share for c in self._cards]
        group = getattr(req, "card_group", None)
        if getattr(self.card, "rule", "trial") == "trial" and group:
            tags = [str(t) for t, p in group if p in self.card.reducers]
            return card_assign(tags, shares).get(tag)
        slot, lo = card_slot(tag), 0.0
        for k, sh in enumerate(shares):
            if lo <= slot < lo + sh:
                return k
            lo += sh
        return None

    def _to_card(self, pid, req) -> Optional[_Card]:
        """Return the card a job goes to, or None for the CPU, with the lock held.

        A job a card could take counts in ``card_eligible``.  One its card's share picks (:meth:`_card_pick`)
        counts in that card's ``marked`` while the card is up, and one the card's reducer turns away in its
        ``refused``, under the reason (``refusal(req)`` where the reducer has it, else ``supports``).  The
        jobs of a card out of the run go to the CPU workers, not to another card.

        Parameters
        ----------
        pid : object
            The job's partition.
        req : ReductionRequest
            The job's request.

        Returns
        -------
        card : _Card or None
            The card, None when the job goes to the CPU workers.
        """
        if not getattr(req, "card", False) or pid not in self.card.reducers:
            return None
        self.card_eligible += 1
        k = self._card_pick(pid, req)
        if k is None or not self._cards[k].ok:
            return None
        c = self._cards[k]
        c.marked += 1
        red = self.card.reducers[pid]
        try:
            why = red.refusal(req) if hasattr(red, "refusal") else (None if red.supports(req) else "not supported")
        except Exception as exc:
            why = f"supports() raised {type(exc).__name__}"
        if why is not None:
            c.refused[str(why)] = c.refused.get(str(why), 0) + 1
            return None
        return c

    def _card_down(self, c: _Card, why: str) -> None:
        """Leave a card out from now on, with the lock held.

        Its jobs go back to the CPU workers, ahead of the rest, and its workers are told to stop.

        Parameters
        ----------
        c : _Card
            The card.
        why : str
            Why, for the log and the card's ``why``.
        """
        if not c.ok:
            return
        c.ok = False
        c.why = why
        back = list(c.out.items())
        c.out.clear()
        for jid, (pid, req) in back:
            if jid in self._futures:
                heapq.heappush(self._waiting, (float("-inf"), jid, pid, req))
        self._notes.append(f"  {c.name}: {why}, so the run goes on without it"
                           + (f" ({len(back)} of its jobs go to the CPU workers)" if back else ""))
        for _ in c.procs:
            try:
                c.inq.put(None)
            except Exception:
                pass
        self._dispatch()

    def _check_card(self) -> None:
        """Leave a card out when a worker of it died without saying so, the job it held lost with it.

        The collector's thread calls it once a second, and every job the card holds goes back to the CPU.
        """
        with self._lock:
            for c in self._cards:
                if not c.ok:
                    continue
                dead = [p.name for j, p in enumerate(c.procs) if j not in c.gone and not p.is_alive()]
                if dead:
                    self._card_down(c, f"card worker(s) {', '.join(dead)} died")

    def take_notes(self) -> list:
        """Return the pool's log lines since the last call, such as a card coming up or going away.

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
        """Summarize the cards and the CPU workers: shares, workers, state, warm-up reports, jobs and times.

        Returns
        -------
        summary : dict or None
            The cards together, ``share`` (their shares' sum), ``shares``, ``workers``, ``rule``, ``device``
            (the first card's), ``devices``, ``up`` (any card), ``ready``, ``why_down`` (when every card is
            out), ``info`` (the first card's), ``jobs``, ``busy_s``, ``busy_fraction`` (of the time since the
            pool started), ``marked``, ``refused`` and ``memory``, with ``cards`` (each card's part,
            :meth:`_Card.summary`), ``cpu`` (``jobs``, ``busy_s`` and ``workers``), ``eligible`` and ``t``
            (the time of the summary, for a stretch of the run), None without a card.
        """
        if self.card is None:
            return None
        now = time.time()
        wall = max(now - self._card_t0, 1e-9)
        with self._lock:
            cards = [c.summary(wall) for c in self._cards]
            cpu = {"jobs": self.cpu_jobs, "busy_s": round(self.cpu_busy, 3), "workers": self.n}
            eligible = self.card_eligible
        refused: Dict[str, int] = {}
        for c in cards:
            for k, v in c["refused"].items():
                refused[k] = refused.get(k, 0) + v
        workers = sum(c["workers"] for c in cards)
        busy = sum(c["busy_s"] for c in cards)
        up = any(c["up"] for c in cards)
        whys = [c["why_down"] for c in cards if c["why_down"]]
        return {"share": float(sum(c["share"] for c in cards)), "shares": [c["share"] for c in cards],
                "workers": workers, "rule": getattr(self.card, "rule", "trial"), "device": cards[0]["device"],
                "devices": [c["device"] for c in cards], "up": up, "ready": sum(c["ready"] for c in cards),
                "why_down": None if up or not whys else " and ".join(whys),
                "info": cards[0]["info"], "jobs": sum(c["jobs"] for c in cards), "busy_s": round(busy, 3),
                "busy_fraction": round(busy / (wall * max(workers, 1)), 4),
                "marked": sum(c["marked"] for c in cards), "refused": refused,
                "memory": sum(c["memory"] for c in cards), "cards": cards, "cpu": cpu, "eligible": eligible,
                "t": now}

    def _card_result(self, item) -> None:
        """Take a card worker's message, in the collector's thread.

        A ready or failed worker is counted, and its card goes down when none of its workers could start.
        A result settles its job's future, while a job that ran out of the worker's memory, or failed, goes
        back to the CPU workers, :data:`CARD_MAX_ERRORS` failures taking its card down.

        Parameters
        ----------
        item : tuple
            The message, ``("card-ready", index, info)``, ``("card-failed", index, why)`` or
            ``("card", job_id, result, seconds, index)``.
        """
        kind = item[0]
        with self._lock:
            if kind in ("card-ready", "card-failed"):
                c, j = self._card_worker.get(item[1], (self._cards[0], item[1]))
                if kind == "card-ready":
                    c.ready += 1
                    if c.info is None:
                        c.info = dict(next(iter(item[2].values()), {}) or {}) if item[2] else {}
                        cb = getattr(self.card, "on_ready", None)
                        if callable(cb):
                            try:
                                line = cb(dict(c.info, card_index=c.index))
                            except Exception as exc:  # noqa: BLE001 - a record for the log, never fatal
                                line = f"  {c.name}: on_ready failed ({exc!r})"
                            if line:
                                self._notes.append(str(line))
                else:
                    # a worker that cannot warm up never takes a job, so the card's others serve its
                    # queue alone, and with none left the card is out
                    c.gone.add(j)
                    if len(c.gone) == len(c.procs):
                        self._card_down(c, f"it did not start ({item[2]})")
                        return
                    self._notes.append(f"  {c.name}: worker {j} did not start ({item[2]})")
                if c.ok and c.ready + len(c.gone) == len(c.procs):
                    self._notes.append(f"  {c.name}: {describe_card(dict(c.summary(1.0), rule=self.card.rule))}")
                return
            jid, res, dt = item[1], item[2], item[3]
            c = self._card_worker[item[4]][0] if len(item) > 4 and item[4] in self._card_worker else None
            if c is None:
                c = next((x for x in self._cards if jid in x.out), self._cards[0])
            c.busy += float(dt)
            sent = c.out.pop(jid, None)
            f = self._futures.get(jid)
            if isinstance(res, CardMemoryError) and sent is not None and f is not None:
                # more than the worker's share of the card: a property of the job, so it goes
                # to the CPU and the card stays in the run
                c.memory += 1
                heapq.heappush(self._waiting, (float("-inf"), jid, sent[0], sent[1]))
                if c.memory <= 5:
                    self._notes.append(f"  {c.name}: {res.tag} needed more than its worker's share of the card's "
                                       f"memory ({res.message}); reduced on the CPU instead"
                                       + ("  (further ones are counted, not listed)" if c.memory == 5 else ""))
                self._dispatch()
                return
            if isinstance(res, BaseException) and sent is not None and f is not None:
                # a job the card failed is reduced on the CPU instead
                c.errors += 1
                heapq.heappush(self._waiting, (float("-inf"), jid, sent[0], sent[1]))
                if c.errors >= CARD_MAX_ERRORS:
                    self._card_down(c, f"{c.errors} jobs failed on it, the last with {res!r}")
                else:
                    self._notes.append(f"  {c.name}: a job failed on it ({res!r}); reduced on the CPU instead")
                self._dispatch()
                return
            if sent is None or f is None:
                return                         # already reduced on the CPU
            self._futures.pop(jid, None)
            c.jobs += 1
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
                c = self._to_card(pid, req) if self.card is not None else None
                if c is not None:
                    c.out[jid] = (pid, req)
                    c.inq.put((jid, pid, req))
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
            self._sent[jid] = time.perf_counter()
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
                    t_sent = self._sent.pop(jid, None)
                    if t_sent is not None:
                        self.cpu_busy += time.perf_counter() - t_sent
                        self.cpu_jobs += 1
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
            self._sent = {}
            for c in self._cards:
                c.out = {}
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
        for c in self._cards:
            for _ in c.procs:
                try:
                    c.inq.put(None)
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
