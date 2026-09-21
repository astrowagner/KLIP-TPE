"""``klip-tpe view``: the separate-process window for watching a running optimization.

The point of it is that it is a *separate process*, so it cannot be starved by the
optimizer and closing it cannot take the run down.  That only holds if it also stays out
of the way of the terminal the run is in, which is what most of this file is about.
"""
from __future__ import annotations

import os
import time

import pytest

from klip_tpe import viewer


def _refresh_loop_code() -> str:
    """The body of ``view``'s refresh loop, comments stripped -- the comments name the call
    being avoided, so a naive substring test passes on the comment that explains it."""
    src = open(viewer.__file__).read()
    loop = src[src.index("while plt.fignum_exists"):]
    return "\n".join(l for l in loop.splitlines() if not l.strip().startswith("#"))


def test_the_refresh_loop_does_not_raise_the_window():
    """``plt.pause`` raises AND focuses the window on every call (macOS especially).

    At the default one-second refresh that takes the foreground once a second, so the
    terminal running the optimizer underneath is unusable -- which defeats the entire point
    of watching from a second shell.  ``draw_idle`` + ``flush_events`` services the same
    event loop and leaves the stacking order alone; ``display.LiveDisplay``'s own live window
    has always done it that way, and its code says why on the line itself.
    """
    loop = _refresh_loop_code()
    assert "plt.pause" not in loop, "plt.pause in the refresh loop raises the window"
    assert "flush_events" in loop
    disp = open(os.path.join(os.path.dirname(viewer.__file__), "display.py")).read()
    assert "the window is never raised" in disp, \
        "display.py's live window changed; the two should still agree on how to pump the GUI"


def test_the_sleep_is_chopped_up_so_the_window_stays_responsive():
    """Sleeping the whole refresh interval in one go makes the window ignore clicks and
    resizes for that long.  The wait is split, with the event loop serviced between pieces."""
    assert "time.sleep(min(" in _refresh_loop_code(), "the refresh wait is not chopped up"


def test_newest_panel_picks_the_latest_across_every_layout(tmp_path):
    """Panels land in ``steps/`` and in ``annulusNN/`` under three different names; the
    window shows whichever was written last, not whichever glob matched first."""
    (tmp_path / "steps").mkdir()
    (tmp_path / "annulus01").mkdir()
    old = tmp_path / "steps" / "step0001.png"
    new = tmp_path / "annulus01" / "step_display_white.png"
    old.write_bytes(b"x")
    time.sleep(0.01)
    new.write_bytes(b"y")
    got = viewer.newest_panel(str(tmp_path))
    assert got is not None and os.path.basename(got[0]) == "step_display_white.png"


def test_newest_panel_is_none_on_a_run_that_has_not_drawn_yet(tmp_path):
    assert viewer.newest_panel(str(tmp_path)) is None


def test_find_run_takes_a_path_as_given(tmp_path):
    d = tmp_path / "miri_HIP-65426_F1140C"
    d.mkdir()
    assert viewer.find_run(str(d)) == os.path.abspath(str(d))


def test_find_run_says_so_rather_than_watching_nothing(tmp_path):
    with pytest.raises(SystemExit, match="no run directory found"):
        viewer.find_run("last", root=str(tmp_path))
