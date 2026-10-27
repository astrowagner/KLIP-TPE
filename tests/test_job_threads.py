"""A partition job's threads, and the order the worker pool runs jobs in.

From the NEAR2 session's second note of 2026-10-11: a job ran everything but its KLIP loop
on one core (high-pass of every frame, binning, derotation, ``nw_ang_comb``), and a heavy
job (a big night at a small bin) set the pace of its trial while most processes idled.
What must hold:

* the per-frame steps give the same numbers on any number of threads, and so does a whole
  reduction;
* a rotation of a finite frame skips the validity pass and changes nothing;
* the cost estimate grows with the work, and a job's threads with its cost, from 1 to
  ``4 x --threads-per-job``, a power of two;
* a free worker takes the waiting job with the smallest key;
* the jobs running together never use more threads than the budget (``--workers``), from
  the GPU session's farm benchmark on patrick of 2026-10-26.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from scipy import ndimage

from klip_tpe import GaussianPSF, KLIPReducer, PartitionedReducer
from klip_tpe.klip import derotate, highpass, highpass_stack, nw_ang_comb, rotate_ccw
from klip_tpe.metrics import Source
from klip_tpe.parallel import ProcessPool
from klip_tpe.reducer import ReductionRequest
from klip_tpe.synthetic import synthetic_klip_dataset

PX = 0.05


def _same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True) and np.asarray(a).dtype == np.asarray(b).dtype


def _rotate_reference(img, angle_deg, cval=np.nan, order=1, nan_mode="zero"):
    """rotate_ccw as it was before 2026-10-11: both interpolations, every time."""
    img = np.asarray(img, float)
    if angle_deg == 0.0:
        return img.copy()
    ny, nx = img.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx, dy = xx - cx, yy - cy
    xs = cx + ca * dx + sa * dy
    ys = cy - sa * dx + ca * dy
    fin = np.isfinite(img)
    out = ndimage.map_coordinates(np.where(fin, img, 0.0), [ys, xs], order=order, mode="constant", cval=0.0)
    w = ndimage.map_coordinates(fin.astype(float), [ys, xs], order=order, mode="constant", cval=0.0)
    inside = (xs >= 0) & (xs <= nx - 1) & (ys >= 0) & (ys <= ny - 1)
    if nan_mode == "propagate":
        out[(w < 1.0 - 1e-6) & inside] = np.nan
    else:
        out[(w <= 1e-6) & inside] = np.nan
    out[~inside] = cval
    return out


def test_a_rotation_is_what_it_was():
    rng = np.random.default_rng(3)
    for shape in [(21, 21), (20, 24), (64, 64)]:
        for kind in ("finite", "nan", "zeros"):
            img = rng.standard_normal(shape) * 10
            if kind == "nan":
                img[3, 4] = np.nan
                img[:, -2] = np.nan
            elif kind == "zeros":
                img[:] = 0.0
            for a in [0.0, 90.0, -90.0, 180.0, 45.0, 1e-9, 33.3, -179.99, 720.5]:
                for mode in ("zero", "propagate"):
                    for cval in (np.nan, 0.0):
                        assert _same(_rotate_reference(img, a, cval, nan_mode=mode),
                                     rotate_ccw(img, a, cval=cval, nan_mode=mode)), (shape, kind, a, mode)


def test_the_per_frame_steps_give_the_same_numbers_on_any_thread_count():
    rng = np.random.default_rng(4)
    cube = (rng.standard_normal((37, 41, 40)) * 20 + 7).astype(np.float32)
    cube[2, 5, 7] = np.nan
    cube[4, :, 3] = np.nan
    cube[6] = 0.0
    ang = np.sort(rng.uniform(-30, 30, cube.shape[0]))
    for w in (4, 5, 9):
        for nan_aware in (False, True):
            ref = np.stack([highpass(f, w, nan_aware=nan_aware) for f in cube])
            for th in (1, 2, 3):
                assert _same(ref, highpass_stack(cube, w, nan_aware=nan_aware, threads=th)), (w, nan_aware, th)
    der = derotate(cube, ang)
    for th in (2, 3, 5):
        assert _same(der, derotate(cube, ang, threads=th))
        assert _same(nw_ang_comb(der, ang), nw_ang_comb(der, ang, threads=th))
        ref = derotate(cube[::-1].copy(), ang)
        assert _same(nw_ang_comb(der, ang, ref_cube=ref), nw_ang_comb(der, ang, ref_cube=ref, threads=th))


def _reducer(nframes=40, size=48, seed=1):
    ds = synthetic_klip_dataset(nframes=nframes, size=size, pa_span=60, seed=seed)
    return KLIPReducer(ds, PX, 3.89 * PX / 206265 * 8.2, 8.2, outrad_cap=size // 2 - 4,
                       injection_model=GaussianPSF(4.0, star_flux=1e4))


def test_a_reduction_gives_the_same_images_on_any_thread_count():
    red = _reducer()
    src = [Source(0.6, 40.0, 3e-3)]
    for b, filt, extras, kscan, fm in [(1, 7, False, False, False), (3, 5, True, False, False),
                                       (2, 0, False, True, False), (4, 9, False, False, True)]:
        params = {"k_klip": 4, "inrad": 6, "outrad": 18, "bin": b, "filter": filt, "n_ang": 2,
                  "angsep": 0.5, "anglemax": 40.0}
        out = []
        for th in (1, 3):
            r = red.reduce(ReductionRequest(params, src, kscan, "t", fm_sources=src if fm else None,
                                            extras=extras, threads=th))
            out.append((r.image, r.fm_image, r.cadi))
        for x, y in zip(*out):
            assert (x is None and y is None) or _same(x, y), (b, filt, extras, kscan, fm)


def test_the_cost_estimate_grows_with_the_work():
    small, big = _reducer(nframes=30, seed=1), _reducer(nframes=90, seed=2)
    p = {"k_klip": 6, "inrad": 6, "outrad": 18, "filter": 7, "n_ang": 2, "angsep": 0.5, "anglemax": 30.0}
    c = {b: big.cost_estimate(dict(p, bin=b)) for b in (1, 3, 9)}
    assert c[1] > c[3] > c[9] > 0, "fewer binned frames, less work"
    assert big.cost_estimate(dict(p, bin=3)) > small.cost_estimate(dict(p, bin=3))
    assert big.cost_estimate(dict(p, bin=3), k_scan=True) > c[3]
    assert big.cost_estimate(dict(p, bin=3), fm=True) > c[3]
    assert big.cost_estimate(dict(p, bin=3), n_inject=4) > c[3]
    assert big.cost_estimate(dict(p, bin=3)) == c[3]                # cached, the same answer


def test_a_jobs_threads_follow_its_cost():
    reds = {"light": _reducer(nframes=30, seed=1), "heavy": _reducer(nframes=120, seed=2)}
    pr = PartitionedReducer(reds, max_workers=16, threads_per_job=2)
    p = {"k_klip": 6, "inrad": 6, "outrad": 18, "filter": 7, "n_ang": 2, "angsep": 0.5, "anglemax": 30.0}
    jobs = [(reds[k], ReductionRequest(dict(p, bin=b), None)) for k in ("light", "heavy") for b in (1, 6)]
    for _ in range(4):
        threads, walls = pr._plan(jobs, 2)
    costs = [r.cost_estimate(q.params) for r, q in jobs]
    assert all(t in (1, 2, 4, 8) for t in threads)
    assert threads[int(np.argmax(costs))] == max(threads) > 1
    assert threads[int(np.argmin(costs))] == 1
    assert max(walls) / min(walls) < max(costs) / min(costs), "threads even out the jobs' wall times"
    # without an estimate every job keeps the flat count
    class _Plain:
        def reduce(self, req):                     # pragma: no cover
            raise AssertionError
    assert pr._plan([(_Plain(), ReductionRequest(p, None))], 2) is None


class _Clock:
    """A partition that sleeps and says when it finished."""

    def reduce(self, req):
        time.sleep(req[0])
        return req[1], time.time()


def test_a_free_worker_takes_the_smallest_key_first():
    pool = ProcessPool({"a": _Clock()}, 1)
    try:
        t = threading.Thread(target=lambda: pool.map([("a", (0.6, "blocker"))]))
        t.start()
        time.sleep(0.2)                            # the one worker is busy
        got = pool.map([("a", (0.05, "third")), ("a", (0.05, "first")), ("a", (0.05, "second"))],
                       keys=[3.0, 1.0, 2.0])
        t.join(timeout=10)
        assert [g[0] for g in got] == ["third", "first", "second"], "results come back in the order asked"
        assert [g[0] for g in sorted(got, key=lambda g: g[1])] == ["first", "second", "third"]
        # equal keys keep the order they came in
        t = threading.Thread(target=lambda: pool.map([("a", (0.6, "blocker"))]))
        t.start()
        time.sleep(0.2)
        got = pool.map([("a", (0.05, k)) for k in "abc"])
        t.join(timeout=10)
        assert [g[0] for g in sorted(got, key=lambda g: g[1])] == list("abc")
    finally:
        pool.close()


class _Threaded:
    """A partition that uses the process-wide thread pool, as a KLIP job does."""

    def reduce(self, req):
        from klip_tpe.parallel import target_pool
        return sum(target_pool(req).map(lambda v: v * v, range(8)))


def test_a_worker_forked_after_the_parent_used_its_thread_pools_still_runs():
    """The parent's pools reach a forked worker without their threads.  Used there, they
    took work and never ran it: the worker hung."""
    from klip_tpe.parallel import target_pool
    gate = threading.Barrier(3)
    list(target_pool(3).map(lambda _: gate.wait(timeout=10), range(3)))  # the parent's pool of 3, all running
    pool = ProcessPool({"a": _Threaded()}, 1)
    try:
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("r", pool.map([("a", 3)])))
        t.start()
        t.join(timeout=20)
        assert not t.is_alive(), "the worker hung on a thread pool it inherited"
        assert out["r"] == [140]
    finally:
        pool.close()


class _Held:
    """A partition that sleeps and reports the threads it was sent with and when it ran."""

    def reduce(self, req):
        t0 = time.time()
        time.sleep(req.params["s"])
        return int(req.threads), t0, time.time()


def _most_at_once(got):
    """The most threads held by jobs running at the same moment, from their intervals."""
    ev = sorted([(t0, t) for t, t0, _ in got] + [(t1, -t) for t, _, t1 in got])
    run = most = 0
    for _, d in ev:                                # an end and a start at one instant: the end first
        run += d
        most = max(most, run)
    return most


def test_the_jobs_in_flight_never_use_more_threads_than_the_budget():
    """The by-cost plan gives a long job up to 4x the base and a short one 1, and the long
    ones run longest, so the jobs running at a moment held more threads than the budget: on
    patrick a load average of 234 on 192 CPUs at a base of 4, and 134 on a budget of 96 at
    a base of 2.  The pool now cuts a job's threads, when it sends it, to what the running
    jobs leave, keeping one for each idle worker.  A worker never waits for threads, every
    job gets at least one, and none more than it asked for."""
    planned = [8, 8, 1, 4, 2, 8, 1, 1, 4, 2, 8, 1, 6, 3]
    reqs = [("a", ReductionRequest({"s": 0.12 + 0.04 * (i % 3)}, None, threads=t)) for i, t in enumerate(planned)]
    pool = ProcessPool({"a": _Held()}, 4, budget=6)
    try:
        got = pool.map(reqs)
    finally:
        pool.close()
    threads = [g[0] for g in got]
    assert all(1 <= t <= p for t, p in zip(threads, planned)), list(zip(threads, planned))
    assert _most_at_once(got) <= 6, f"{_most_at_once(got)} threads in flight on a budget of 6"
    assert [q.threads for _, q in reqs] == planned, "the caller's requests were changed"
    assert threads[0] == 3, "the first job, sent with 3 workers idle, leaves them one thread each"
    # without a budget the plan goes as it is
    pool = ProcessPool({"a": _Held()}, 4)
    try:
        assert [g[0] for g in pool.map(reqs[:6])] == planned[:6]
    finally:
        pool.close()


def test_a_partitioned_reducer_holds_its_pool_to_the_budget():
    reds = {"n1": _reducer(nframes=30, seed=1), "n2": _reducer(nframes=60, seed=2)}
    pr = PartitionedReducer(reds, max_workers=6, threads_per_job=2)
    said = []
    pr._log = said.append
    try:
        assert pr.start_workers() == "processes"
        assert pr._procs.budget == 6
        assert "at most 6 in flight" in "\n".join(said)
    finally:
        pr.close()
    pr = PartitionedReducer(reds, max_workers=6, threads_per_job=2)
    pr.cap_threads = False                          # the plan as it was before 2026-10-27
    try:
        pr.start_workers()
        assert pr._procs.budget is None
    finally:
        pr.close()
