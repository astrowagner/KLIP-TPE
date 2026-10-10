"""The live window and ``klip-tpe view`` keep one size.

Both show a panel drawn at a fixed pixel size (``display.PANEL_PX`` x ``window_scale``), and
a resized window only rescales it.  Three things hold the size:

* :func:`prefer_x11_on_wayland`, before the GUI toolkit starts.  On a Wayland desktop
  (GNOME on Ubuntu, including its remote-login sessions) Qt and GTK draw their own window
  frame, and GNOME lets such a window be maximized whatever its size limits.  Run through
  XWayland instead, the window manager draws the frame and holds the size.
* :func:`fix_window_size`, once the window exists: pins the size and switches the toolkit's
  resizing off.
* :func:`keep_window_size`, on every GUI turn: puts back a window that was resized or
  maximized anyway.

Kept apart from ``display`` so that the viewer, a separate process, does not import the run
machinery to get them.
"""
from __future__ import annotations

import ctypes.util
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

import matplotlib

__all__ = ["PIN_GRACE", "prefer_x11_on_wayland", "fix_window_size", "keep_window_size",
           "window_platform"]

#: seconds after the window opens during which a size the window manager gives it is kept
#: (placement on a screen smaller than the window, a HiDPI rescale as it maps).  After that
#: a different size is always put back.
PIN_GRACE = 2.0


# -- before the toolkit starts -----------------------------------------------------------
def _qt_app_running() -> bool:
    for mod in ("PyQt6.QtWidgets", "PySide6.QtWidgets", "PyQt5.QtWidgets", "PySide2.QtWidgets"):
        m = sys.modules.get(mod)
        if m is not None:
            try:
                if m.QApplication.instance() is not None:
                    return True
            except Exception:
                pass
    return False


def prefer_x11_on_wayland() -> Optional[str]:
    """On a Wayland session, run Qt and GTK windows through XWayland.

    A native Wayland Qt window draws its own frame, and on GNOME a double-click on that frame
    maximizes a window whose size Qt has fixed.  Through XWayland GNOME's window manager
    draws the frame and refuses both resizing and maximizing.  The platforms are set as
    fallback lists (``xcb;wayland``, ``x11,wayland``), so a machine without the X11 pieces
    (Qt's xcb plugin needs ``libxcb-cursor0``) still gets a native window.  Variables the
    user set are left alone, and ``KLIP_TPE_WAYLAND=native`` turns this off.  Only acts
    before the toolkit has started.  Returns a line for the log, or None."""
    if not sys.platform.startswith("linux"):
        return None
    if not (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland"):
        return None
    if not os.environ.get("DISPLAY"):                     # no XWayland to go to
        return None
    if os.environ.get("KLIP_TPE_WAYLAND", "").strip().lower() in ("native", "wayland", "1", "yes"):
        return None
    set_, note = [], ""
    if "QT_QPA_PLATFORM" not in os.environ and not _qt_app_running():
        if ctypes.util.find_library("xcb-cursor"):
            os.environ["QT_QPA_PLATFORM"] = "xcb;wayland"
            set_.append("QT_QPA_PLATFORM=xcb;wayland")
        else:                       # Qt's xcb plugin would not load: stay native, say why
            note = ("; a Qt window stays native Wayland, where GNOME can maximize it "
                    "(sudo apt install libxcb-cursor0 lets it run through XWayland)")
    if "GDK_BACKEND" not in os.environ and "gi.repository.Gtk" not in sys.modules:
        os.environ["GDK_BACKEND"] = "x11,wayland"
        set_.append("GDK_BACKEND=x11,wayland")
    if not set_ and not note:
        return None
    head = ("Wayland session: the window opens through XWayland (" + ", ".join(set_) + ") so that "
            "the window manager holds its size; KLIP_TPE_WAYLAND=native keeps it native") if set_ \
        else "Wayland session"
    return head + note


def window_platform(fig) -> str:
    """Where ``fig``'s window lives: ``"x11"``, ``"wayland"``, ``"macos"``, ``"windows"`` or
    ``""`` (unknown), for the log."""
    mgr = getattr(fig.canvas, "manager", None)
    win = getattr(mgr, "window", None)
    try:
        if win is not None and hasattr(win, "setFixedSize"):
            from matplotlib.backends.qt_compat import QtWidgets
            name = QtWidgets.QApplication.instance().platformName().lower()
            return {"xcb": "x11", "cocoa": "macos"}.get(name, name)
        if win is not None and hasattr(win, "winfo_width"):
            return {"x11": "x11", "aqua": "macos", "win32": "windows"}.get(win.tk.call("tk", "windowingsystem"), "")
        if win is not None and hasattr(win, "set_resizable"):
            disp = type(win.get_display()).__name__.lower()
            return "wayland" if "wayland" in disp else "x11" if "x11" in disp else ""
    except Exception:
        return ""
    if "macosx" in matplotlib.get_backend().lower():
        return "macos"
    return ""


# -- the window itself -------------------------------------------------------------------
def _parts(fig):
    mgr = getattr(fig.canvas, "manager", None)
    return mgr, getattr(mgr, "window", None)


def _kind(mgr, win) -> str:
    if win is not None and hasattr(win, "winfo_width") and hasattr(win, "wm_geometry"):
        return "tk"
    if win is not None and hasattr(win, "setFixedSize"):
        return "qt"
    if win is not None and hasattr(win, "set_resizable"):
        return "gtk3" if hasattr(win, "resize") else "gtk4"
    frame = getattr(mgr, "frame", None)
    if frame is not None and hasattr(frame, "SetMaxSize"):
        return "wx"
    return ""


def _win_size(mgr, win, kind) -> Optional[Tuple[int, int]]:
    """The toolkit window's own size (its pixels), or None where it cannot be read."""
    try:
        if kind == "tk":
            return int(win.winfo_width()), int(win.winfo_height())
        if kind == "qt":
            return int(win.width()), int(win.height())
        if kind == "gtk3":
            w, h = win.get_size()
            return int(w), int(h)
        if kind == "gtk4":
            return int(win.get_width()), int(win.get_height())
        if kind == "wx":
            s = mgr.frame.GetSize()
            return int(s[0]), int(s[1])
    except Exception:
        pass
    return None


def _maximized(mgr, win, kind) -> bool:
    try:
        if kind == "tk":
            return win.state() == "zoomed" or bool(win.attributes("-zoomed")) \
                or bool(win.attributes("-fullscreen"))
        if kind == "qt":
            return bool(win.isMaximized() or win.isFullScreen())
        if kind in ("gtk3", "gtk4"):
            return bool(win.is_maximized())
        if kind == "wx":
            return bool(mgr.frame.IsMaximized() or mgr.frame.IsFullScreen())
    except Exception:
        pass
    return False


def _put_back(fig, st, mgr, win, kind) -> None:
    """Un-maximize and ask for the pinned size, in the toolkit's own terms."""
    if not st.get("win"):
        kind = ""                                    # size unknown to the toolkit: the figure's
    w, h = st["win"] if st.get("win") else (None, None)
    if kind == "tk":
        if _maximized(mgr, win, kind):
            for args in (("-fullscreen", False), ("-zoomed", False)):
                try:
                    win.attributes(*args)
                except Exception:
                    pass
            win.state("normal")
        # the toplevel's own geometry: after the window manager has resized it, Tk ignores
        # a new size requested by the canvas inside it
        win.geometry(f"{w}x{h}")
    elif kind == "qt":
        if _maximized(mgr, win, kind):
            win.showNormal()
        win.resize(w, h)
    elif kind == "gtk3":
        if _maximized(mgr, win, kind):
            win.unmaximize()
        win.resize(w, h)
    elif kind == "gtk4":
        if _maximized(mgr, win, kind):
            win.unmaximize()
        win.set_default_size(w, h)
    elif kind == "wx":
        if _maximized(mgr, win, kind):
            mgr.frame.Maximize(False)
        mgr.frame.SetSize((w, h))
    else:
        fig.set_size_inches(*st["size"], forward=True)


def fix_window_size(fig) -> bool:
    """Pin ``fig``'s window at its current size and switch the toolkit's resizing off.

    Tk, Qt, GTK and wx windows are made non-resizable.  The macOS backend has no such switch
    in matplotlib, so there the resize control is removed through PyObjC
    (``pyobjc-framework-Cocoa``) when it is installed.  Wherever neither works,
    :func:`keep_window_size` puts the window back on the next GUI turn.  Returns True when
    the toolkit's resizing is off."""
    mgr, win = _parts(fig)
    kind = _kind(mgr, win)
    fig._klip_pin = {"size": np.array(fig.get_size_inches(), float), "dpi": float(fig.dpi),
                     "win": None, "kind": kind, "opened": time.time(), "asked": 0.0, "misses": 0}
    try:
        if kind == "tk":
            win.resizable(False, False)
            return True
        if kind == "qt":
            win.setFixedSize(win.size())
            return True
        if kind in ("gtk3", "gtk4"):
            win.set_resizable(False)
            return True
        if kind == "wx":
            size = mgr.frame.GetSize()
            mgr.frame.SetMinSize(size)
            mgr.frame.SetMaxSize(size)
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

    Called on every GUI turn, and a no-op while the size holds.  The size is read from the
    toolkit window itself: a window manager that resizes a Tk window leaves the figure's own
    size alone.  For ``PIN_GRACE`` seconds after the window opens, a size the window manager
    gives it is kept (a screen smaller than the window).  After that a resized or maximized
    window is put back, at most every 0.5 s (every 5 s after ten requests that did not
    take).  A change of the figure's dpi (moving to a screen with another scale) is the
    toolkit's own doing and re-pins the size instead."""
    st = getattr(fig, "_klip_pin", None)
    if not st:
        return
    try:
        now = time.time()
        mgr, win = _parts(fig)
        kind = st.get("kind", "")
        if abs(float(fig.dpi) - st["dpi"]) > 1e-6:            # rescaled by the toolkit
            st["dpi"], st["size"], st["win"], st["opened"] = float(fig.dpi), \
                np.array(fig.get_size_inches(), float), None, now
            if kind == "tk":
                win.geometry("")                              # let the toplevel follow the canvas again
            return
        ws = _win_size(mgr, win, kind)
        if ws is not None and min(ws) < 50:                   # not mapped yet
            return
        settling = now - st["opened"] < PIN_GRACE
        if ws is not None:
            if st["win"] is None or settling:
                st["win"] = ws
                st["size"] = np.array(fig.get_size_inches(), float)
                return
            off = max(abs(ws[0] - st["win"][0]), abs(ws[1] - st["win"][1])) > 1
            if not off and not _maximized(mgr, win, kind):
                st["misses"] = 0
                return
        else:                                                 # macOS: the figure's size
            cur = np.array(fig.get_size_inches(), float)
            tol = 1.5 / float(getattr(fig, "_original_dpi", None) or fig.dpi)
            if np.all(np.abs(cur - st["size"]) <= tol):
                st["misses"] = 0
                return
            if settling:
                st["size"] = cur
                return
        wait = 0.5 if st["misses"] < 10 else 5.0
        if now - st["asked"] < wait:
            return
        st["asked"], st["misses"] = now, st["misses"] + 1
        _put_back(fig, st, mgr, win, kind)
    except Exception:
        pass
