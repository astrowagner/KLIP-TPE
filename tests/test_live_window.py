"""The live window must never go quiet for long: the main thread has to hand the GUI
event loop a turn every couple of seconds, whatever the optimizer is doing.

macOS marks a process that has not serviced events for ~2 s as unresponsive, stops
compositing its windows and leaves the last painted frame on screen -- which looks exactly
like a crashed run even though the optimization is fine.  Rendering already happens
off-thread, so the fix is not "render faster": it is that the *idle hook* pumps on every
tick rather than only when a new frame happens to be ready.  These tests measure the gap
between pumps instead of trusting that it is small.
"""
import threading
import time

import numpy as np
import pytest

from klip_tpe import parallel
from klip_tpe.display import LiveDisplay


MAX_GAP = 2.0          # seconds; macOS gives up at about this


class _FakeCanvas:
    def __init__(self, log):
        self._log = log

    def flush_events(self):
        self._log.append(time.time())

    def draw_idle(self):
        pass


class _FakeWin:
    number = 1

    def __init__(self, log):
        self.canvas = _FakeCanvas(log)


def _display_with_fake_window(tmp_path):
    """A LiveDisplay wired to a fake window that records every event-loop turn."""
    pumps = []
    d = LiveDisplay(str(tmp_path), show=True, save_png=False, movie=False, log=lambda m: None)
    d._win = _FakeWin(pumps)
    d.inline = False
    d.show = True
    d.PUMP_EVERY = 0.0                       # do not throttle inside the test
    import matplotlib.pyplot as plt
    d._fignum_exists = plt.fignum_exists
    return d, pumps


@pytest.fixture(autouse=True)
def _clean_hooks():
    before = list(parallel.IDLE_HOOKS)
    parallel.IDLE_HOOKS.clear()
    yield
    parallel.IDLE_HOOKS[:] = before


def test_idle_hook_pumps_with_no_render_pending(tmp_path, monkeypatch):
    """The regression this file exists for: between renders there is nothing in the queue,
    and the hook used to return immediately without touching the event loop."""
    d, pumps = _display_with_fake_window(tmp_path)
    monkeypatch.setattr("matplotlib.pyplot.fignum_exists", lambda n: True)
    assert not getattr(d, "_renders", None), "no render pending is the interesting case"
    for _ in range(5):
        d._show_latest()
        time.sleep(0.01)
    assert len(pumps) >= 5, "the idle hook must pump the event loop even with no new frame"


def test_pump_is_throttled_and_main_thread_only(tmp_path, monkeypatch):
    d, pumps = _display_with_fake_window(tmp_path)
    monkeypatch.setattr("matplotlib.pyplot.fignum_exists", lambda n: True)
    d.PUMP_EVERY = 10.0
    d._pump(); d._pump(); d._pump()
    assert len(pumps) == 1, "pumping must be throttled, not once per call"

    d.PUMP_EVERY = 0.0
    pumps.clear()
    t = threading.Thread(target=d._pump)
    t.start(); t.join()
    assert pumps == [], "a worker thread must never touch the GUI"


def test_pump_wait_services_hooks_while_threads_work():
    """A bare Future.result() on a thread pool blocks with no window for the event loop;
    pump_wait must keep the hooks running for the whole wait."""
    import concurrent.futures as cf
    ticks = []
    parallel.IDLE_HOOKS.append(lambda: ticks.append(time.time()))

    def slow(x):
        time.sleep(0.6)                      # a "reduction"
        return x * 2

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        out = parallel.pump_wait([ex.submit(slow, 1), ex.submit(slow, 2)], tick=0.05)
    assert out == [2, 4], "results and order must match Future.result()"
    gaps = np.diff([t0] + ticks + [time.time()])
    assert ticks, "the hooks never ran during the wait"
    assert gaps.max() < MAX_GAP, f"the event loop went {gaps.max():.2f}s without a turn"


def test_pump_wait_propagates_the_first_exception():
    import concurrent.futures as cf
    parallel.IDLE_HOOKS.append(lambda: None)

    def boom():
        raise ValueError("reduction failed")

    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        with pytest.raises(ValueError, match="reduction failed"):
            parallel.pump_wait([ex.submit(boom)], tick=0.02)


def test_pump_wait_without_hooks_is_a_plain_wait():
    """No live window -> no reason to poll; pump_wait must not add latency."""
    import concurrent.futures as cf
    assert not parallel.IDLE_HOOKS
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        assert parallel.pump_wait([ex.submit(lambda: 7)]) == [7]


def test_the_event_loop_keeps_its_turn_across_a_long_evaluation(tmp_path, monkeypatch):
    """End to end: a reduction that takes far longer than the macOS limit, with the
    display's own hook registered, must still leave no gap longer than MAX_GAP."""
    import concurrent.futures as cf
    d, pumps = _display_with_fake_window(tmp_path)
    monkeypatch.setattr("matplotlib.pyplot.fignum_exists", lambda n: True)
    d._arm_pump()
    assert d._show_latest in parallel.IDLE_HOOKS, "arming must register the hook"

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        parallel.pump_wait([ex.submit(time.sleep, 5.0)], tick=0.1)
    t1 = time.time()
    assert pumps, "the window was never pumped during a 5 s evaluation"
    gaps = np.diff([t0] + pumps + [t1])
    assert gaps.max() < MAX_GAP, (
        f"the window went {gaps.max():.2f}s without a GUI turn during a 5 s evaluation "
        f"(macOS marks a process unresponsive after ~2 s)")
