"""The live window and ``klip-tpe view`` keep one size.

Both show a panel drawn at a fixed pixel size (``display.PANEL_PX`` x ``window_scale``), and
a resized window only rescales it.  :func:`fix_window_size` pins the size and switches the
toolkit's resizing off; :func:`keep_window_size`, called on every GUI turn, puts back a
window that could still be resized.  Kept apart from ``display`` so that the viewer, a
separate process, does not import the run machinery to get them.
"""
from __future__ import annotations

import time

import numpy as np

import matplotlib

__all__ = ["PIN_GRACE", "fix_window_size", "keep_window_size"]

#: seconds a window manager may hold the window at another size before that size is kept
#: (a screen smaller than the window), rather than asking for the pinned size forever
PIN_GRACE = 2.0


def fix_window_size(fig) -> bool:
    """Pin ``fig``'s window at its current size and switch the toolkit's resizing off.

    The live window and ``klip-tpe view`` show a panel drawn at a fixed pixel size, and a
    resized window only rescales it.  Tk, Qt, GTK and wx windows are made non-resizable.
    The macOS backend has no such switch in matplotlib, so there the resize control is
    removed through PyObjC (``pyobjc-framework-Cocoa``) when it is installed.  Wherever
    neither works, :func:`keep_window_size` puts the window back on the next GUI turn.
    Returns True when the toolkit's resizing is off."""
    fig._klip_pin = {"size": np.array(fig.get_size_inches(), float), "seen": None, "since": 0.0,
                     "asked": 0.0}
    mgr = getattr(fig.canvas, "manager", None)
    win = getattr(mgr, "window", None)
    try:
        if win is not None and hasattr(win, "resizable"):           # Tk
            win.resizable(False, False)
            return True
        if win is not None and hasattr(win, "setFixedSize"):         # Qt 5 / 6
            win.setFixedSize(win.size())
            return True
        if win is not None and hasattr(win, "set_resizable"):        # GTK 3 / 4
            win.set_resizable(False)
            return True
        frame = getattr(mgr, "frame", None)
        if frame is not None and hasattr(frame, "SetMaxSize"):       # wx
            size = frame.GetSize()
            frame.SetMinSize(size)
            frame.SetMaxSize(size)
            return True
        if mgr is not None and "macosx" in matplotlib.get_backend().lower():
            return _macos_fix_resize(mgr)
    except Exception:
        pass
    return False


def _macos_fix_resize(mgr) -> bool:
    """Clear ``NSWindowStyleMaskResizable`` on the window titled like ``mgr``'s.  Needs
    PyObjC; False without it."""
    try:
        from AppKit import NSApplication                 # pyobjc-framework-Cocoa
    except Exception:
        return False
    title = mgr.get_window_title()
    for w in NSApplication.sharedApplication().windows():
        if str(w.title()) == title:
            w.setStyleMask_(w.styleMask() & ~(1 << 3))   # NSWindowStyleMaskResizable
            return True
    return False


def keep_window_size(fig) -> None:
    """Put ``fig``'s window back at the size :func:`fix_window_size` pinned.

    Called on every GUI turn, and a no-op while the size holds.  A window the user resized
    is asked back to its size at once.  A size the window manager will not give up for
    ``PIN_GRACE`` seconds is kept instead of being fought over."""
    st = getattr(fig, "_klip_pin", None)
    if not st:
        return
    try:
        cur = np.array(fig.get_size_inches(), float)
        tol = 1.5 / float(getattr(fig, "_original_dpi", None) or fig.dpi)    # 1.5 px
        if np.all(np.abs(cur - st["size"]) <= tol):
            st["seen"] = None
            return
        now = time.time()
        if st["seen"] is None or not np.all(np.abs(cur - st["seen"]) <= tol):
            st["seen"], st["since"] = cur, now            # resized: ask for the pinned size
        elif now - st["since"] > PIN_GRACE:
            st["size"], st["seen"] = cur, None            # refused: keep the size it was given
            return
        elif now - st["asked"] < 0.5:
            return
        st["asked"] = now
        fig.set_size_inches(*st["size"], forward=True)
    except Exception:
        pass
