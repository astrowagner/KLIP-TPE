"""One theme at a time: the live panel's dark rc context and the books' light one must never
be in force together.

matplotlib keeps a single rcParams dict per process.  ``rc_context`` writes its values into
it on entry and puts back its own snapshot on exit, so a context opened on one thread
restyles every figure being built on any other thread for as long as it is open.  The live
panel draws on the display's render thread inside ``_rc(idl=True, dark=True)``; the annulus
books are written on the main thread inside ``_rc()``.  On the README's first run 4 of 122
panels came out with white axes and black-on-black text -- exactly the ones drawn while
corner.pdf, landscapes.pdf ... products.pdf were being written.  The fix is a lock that every
rc context in the package holds for its whole extent; these tests pin it with forced
interleavings, so they fail deterministically without it.
"""
import threading
import time

import matplotlib

from klip_tpe import display as D
from klip_tpe import plots as P

KEYS = ("figure.facecolor", "axes.facecolor", "savefig.facecolor", "text.color", "axes.labelcolor",
        "xtick.color", "ytick.color", "axes.edgecolor")


def _theme():
    return {k: matplotlib.rcParams[k] for k in KEYS}


def _run(fn):
    th = threading.Thread(target=fn, daemon=True)
    th.start()
    return th


def test_a_book_cannot_restyle_a_panel_that_is_drawing():
    """The bug itself: a light context opened on the main thread while the panel was
    drawing changed the panel's colours mid-figure."""
    inside, release, seen = threading.Event(), threading.Event(), {}

    def panel():
        with D._rc(idl=True, dark=True):
            inside.set()
            release.wait(10)
            seen.update(face=matplotlib.rcParams["axes.facecolor"], text=matplotlib.rcParams["text.color"])

    entered = threading.Event()

    def book():
        with D._rc():
            entered.set()

    tp = _run(panel)
    assert inside.wait(10)
    tb = _run(book)
    time.sleep(0.3)
    try:
        assert not entered.is_set(), "a book's light context opened while the panel's dark one was in force"
    finally:
        release.set()
        tp.join(10)
        tb.join(10)
    assert entered.is_set(), "the book never got its turn once the panel had finished"
    assert seen == {"face": "black", "text": "white"}, f"the panel drew with {seen}"


def test_contexts_on_two_threads_leave_the_global_style_as_they_found_it():
    """Unlocked, the order of the exits decides what is left behind: a book opened inside
    the panel's context and closed after it restores a snapshot that holds the PANEL's
    theme, and every figure made afterwards outside a context -- a user's own plot in the
    same notebook -- comes out dark."""
    before = _theme()
    book_in, panel_out = threading.Event(), threading.Event()

    def panel():
        with D._rc(idl=True, dark=True):
            book_in.wait(0.5)            # unlocked, the book opens inside this context ...
        panel_out.set()

    def book():
        with D._rc():
            book_in.set()
            panel_out.wait(0.5)          # ... and closes after it

    tp = _run(panel)
    time.sleep(0.1)                      # the panel is inside first
    tb = _run(book)
    tp.join(10)
    tb.join(10)
    assert _theme() == before, "the two contexts left a different theme behind than they found"


def test_the_plots_style_waits_for_the_panel_too():
    """``plots._style()`` opens rc contexts as well (``klip-tpe plots``, the benchmark
    figures) and must take the same lock rather than a second one."""
    inside, release = threading.Event(), threading.Event()

    def panel():
        with D._rc(idl=True, dark=True):
            inside.set()
            release.wait(10)

    entered = threading.Event()

    def figure():
        with P._style():
            entered.set()

    tp = _run(panel)
    assert inside.wait(10)
    tf = _run(figure)
    time.sleep(0.3)
    try:
        assert not entered.is_set(), "plots._style() opened while the panel's context was in force"
    finally:
        release.set()
        tp.join(10)
        tf.join(10)
    assert entered.is_set()


def test_contexts_still_nest_on_one_thread():
    """The lock is reentrant: a context opened inside another on the same thread (a book
    helper inside a book, the classic panel inside a run) must not deadlock."""
    done = threading.Event()
    inner = {}

    def nested():
        with D._rc():
            with D._rc(idl=True, dark=True):
                with P._style():
                    inner["face"] = matplotlib.rcParams["axes.facecolor"]
        done.set()

    th = _run(nested)
    th.join(10)
    assert done.is_set(), "nested rc contexts on one thread deadlocked"
    assert inner["face"] == "black", "the innermost context did not see the dark theme it was opened in"


def test_the_lock_is_released_when_a_figure_fails():
    """A book that raises must not leave the lock held -- the next panel would wait forever."""
    try:
        with D._rc():
            raise RuntimeError("a book failed")
    except RuntimeError:
        pass
    got = threading.Event()

    def other():
        with D._rc(idl=True, dark=True):
            got.set()

    th = _run(other)
    th.join(10)
    assert got.is_set(), "the rc lock stayed held after an exception inside a context"
