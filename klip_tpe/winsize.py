"""The live window and ``klip-tpe view``: one size, and the panel drawn sharp.

Both show a panel drawn at a fixed pixel size (``display.PANEL_PX`` x ``window_scale``).
Five pieces make that hold:

* :func:`prefer_x11_on_wayland`, before the GUI toolkit starts.  On a GNOME Wayland session
  a Qt or GTK window draws its own frame, and GNOME maximizes such a window on a double-click
  of that frame whatever its size limits.  Through XWayland GNOME draws the frame and holds
  the size.  It is only used where XWayland draws as sharply as Wayland.
* :func:`open_panel_window` makes the window hidden, shrinks the panel to fit the screen,
  switches resizing off, and only then shows it.  GNOME maximizes a resizable window that
  covers most of the screen the moment it first appears.
* :func:`keep_window_size`, on every GUI turn, puts back a window that was resized or
  maximized anyway.  After ten tries that do not take, the window keeps the size the
  window manager insists on.
* :func:`show_panel` draws the panel pixel for pixel on the screen's own pixels, centered.
  A panel of another size is resampled once (Lanczos), never stretched.
* :func:`screen_copy` draws a figure at the size the window shows it, so that text is
  rendered at the screen's resolution instead of resampled to it.

Kept apart from ``display`` so that the viewer, a separate process, does not import the run
machinery to get them.
"""
from __future__ import annotations

import ctypes.util
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

import matplotlib

__all__ = ["PIN_GRACE", "MAX_MISSES", "prefer_x11_on_wayland", "open_panel_window",
           "fix_window_size", "keep_window_size", "window_platform", "screen_room", "panel_box",
           "fit_image", "screen_copy", "show_panel"]

#: seconds after the window opens during which a smaller size the window manager gives it is
#: kept (a screen smaller than the room it reported).  A maximized or larger window is always
#: put back.
PIN_GRACE = 2.0
#: put-backs in a row that do not take before the window keeps the size it has
MAX_MISSES = 10
#: room left around the window when it is fitted to the screen: a title bar, the window
#: border, and a desktop panel the toolkit cannot see (a Wayland client is not told where
#: GNOME's top bar is)
TITLE_PX, BORDER_PX, TOPBAR_PX = 48, 8, 32


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


def _gnome_scaling() -> Optional[Dict[str, Any]]:
    """GNOME's display layout, from ``org.gnome.Mutter.DisplayConfig`` (through ``gdbus``):
    ``{"logical": bool, "scales": [...], "native_x11": bool}``, or None off GNOME or when it
    cannot be read.  In the logical layout (fractional scaling) GNOME draws an X11 window at
    scale 1 and enlarges it, which blurs it, unless XWayland scales natively (GNOME 47+)."""
    gdbus = shutil.which("gdbus")
    if not gdbus:
        return None
    try:
        out = subprocess.run(
            [gdbus, "call", "--session", "--dest", "org.gnome.Mutter.DisplayConfig",
             "--object-path", "/org/gnome/Mutter/DisplayConfig",
             "--method", "org.gnome.Mutter.DisplayConfig.GetCurrentState"],
            capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return None
    m = re.search(r"'layout-mode': <uint32 (\d+)>", out)
    if not out or m is None:
        return None
    # logical monitors: (x, y, scale, transform, primary, [monitors], {props})
    scales = [float(s) for s in re.findall(r"\(-?\d+, -?\d+, ([0-9.]+), uint32 \d+, (?:true|false), \[",
                                             out)]
    native = False
    gs = shutil.which("gsettings")
    if gs:
        try:
            native = "xwayland-native-scaling" in subprocess.run(
                [gs, "get", "org.gnome.mutter", "experimental-features"],
                capture_output=True, text=True, timeout=3).stdout
        except Exception:
            native = False
    return {"logical": m.group(1) == "1", "scales": scales, "native_x11": native}


#: :func:`_gnome_scaling`, read once per process (a batch builds one display per run)
_SCALING: Dict[str, Any] = {}


def _gnome_scaling_once() -> Optional[Dict[str, Any]]:
    if "v" not in _SCALING:
        _SCALING["v"] = _gnome_scaling()
    return _SCALING["v"]


def prefer_x11_on_wayland() -> Optional[str]:
    """On a GNOME Wayland session, run Qt and GTK windows through XWayland.

    A native Wayland Qt window draws its own frame, and on GNOME a double-click on that frame
    maximizes a window whose size Qt has fixed.  Through XWayland GNOME's window manager
    draws the frame and refuses both resizing and maximizing.  This is skipped where GNOME
    would enlarge an X11 window to a scaled screen (fractional scaling), since the panel
    would then be blurred.  The platforms are set as fallback lists (``xcb;wayland``,
    ``x11,wayland``), so a machine without the X11 pieces (Qt's xcb plugin needs
    ``libxcb-cursor0``) still gets a native window.  Variables the user set are left alone,
    ``KLIP_TPE_WAYLAND=native`` turns this off, and ``KLIP_TPE_WAYLAND=x11`` forces it.
    Only acts before the toolkit has started.  Returns a line for the log, or None."""
    if not sys.platform.startswith("linux"):
        return None
    if not (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland"):
        return None
    if not os.environ.get("DISPLAY"):                     # no XWayland to go to
        return None
    choice = os.environ.get("KLIP_TPE_WAYLAND", "").strip().lower()
    if choice in ("native", "wayland", "1", "yes"):
        return None
    forced = choice in ("x11", "xwayland")
    if not forced and "gnome" not in os.environ.get("XDG_CURRENT_DESKTOP", "").lower():
        return None                 # other desktops (KDE) draw the frame of a Wayland window themselves
    want_qt = "QT_QPA_PLATFORM" not in os.environ and not _qt_app_running()
    want_gdk = "GDK_BACKEND" not in os.environ and "gi.repository.Gtk" not in sys.modules
    if not (want_qt or want_gdk):
        return None                 # set already (by the user, or by an earlier display)
    if not forced:
        sc = _gnome_scaling_once()
        if sc and sc["logical"] and not sc["native_x11"] and any(abs(s - 1.0) > 1e-3 for s in sc["scales"]):
            s = max(sc["scales"], key=lambda v: abs(v - 1.0))
            return (f"Wayland session at scale {s:g}: the window stays native Wayland, since GNOME "
                    f"would enlarge an XWayland window and blur the panel")
    set_, note = [], ""
    if want_qt:
        if ctypes.util.find_library("xcb-cursor"):
            os.environ["QT_QPA_PLATFORM"] = "xcb;wayland"
            set_.append("QT_QPA_PLATFORM=xcb;wayland")
        else:                       # Qt's xcb plugin would not load: stay native, say why
            note = ("; a Qt window stays native Wayland, where GNOME can maximize it "
                    "(sudo apt install libxcb-cursor0 lets it run through XWayland)")
    if want_gdk:
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


def screen_room(fig) -> Optional[Tuple[int, int]]:
    """The room for ``fig``'s window content on its screen, in the toolkit's pixels: the
    work area less a title bar and the window border.  None when the toolkit does not say
    (the macOS backend without PyObjC)."""
    mgr, win = _parts(fig)
    kind = _kind(mgr, win)
    w = h = None
    try:
        if kind == "qt":
            from matplotlib.backends.qt_compat import QtWidgets
            app = QtWidgets.QApplication.instance()
            scr = (win.screen() if hasattr(win, "screen") else None) or app.primaryScreen()
            g, full = scr.availableGeometry(), scr.geometry()
            w, h = g.width(), g.height()
            if g == full and app.platformName().lower().startswith("wayland"):
                h -= TOPBAR_PX
        elif kind == "tk":
            w, h = win.winfo_screenwidth(), win.winfo_screenheight() - TOPBAR_PX
        elif kind == "gtk3":
            from gi.repository import Gdk
            disp = Gdk.Display.get_default()
            mon = disp.get_primary_monitor() or disp.get_monitor(0)
            r = mon.get_workarea()
            w, h = r.width, r.height
        elif kind == "gtk4":
            from gi.repository import Gdk
            r = Gdk.Display.get_default().get_monitors().get_item(0).get_geometry()
            w, h = r.width, r.height - TOPBAR_PX
        elif kind == "wx":
            import wx
            i = wx.Display.GetFromWindow(mgr.frame)
            r = wx.Display(i if i >= 0 else 0).GetClientArea()
            w, h = r.width, r.height
        elif "macosx" in matplotlib.get_backend().lower():
            from AppKit import NSScreen                      # pyobjc-framework-Cocoa
            f = NSScreen.mainScreen().visibleFrame()
            w, h = f.size.width, f.size.height
    except Exception:
        return None
    if not w or not h:
        return None
    return int(w) - 2 * BORDER_PX, int(h) - TITLE_PX


def open_panel_window(px: Tuple[int, int], scale: float, dpi: float, title: str):
    """A window for a ``px`` panel at ``scale``: made hidden, shrunk to fit the screen, its
    resizing switched off, and then shown.

    A window shown first and fixed afterwards opened at the wrong size on GNOME, which
    maximizes a resizable window that covers most of the screen as it appears.  Returns
    ``(fig, info)``.  ``info`` holds the scale used and the scale asked for, the room on the
    screen, whether the toolkit's resizing is off, and where the window lives."""
    import matplotlib.pyplot as plt
    interactive = plt.isinteractive()
    plt.ioff()                      # in interactive mode pyplot shows a new window at once
    try:
        fig = plt.figure(figsize=(px[0] * scale / dpi, px[1] * scale / dpi), dpi=dpi)
    finally:
        if interactive:
            plt.ion()
    fig.patch.set_facecolor("black")
    try:
        fig.canvas.manager.set_window_title(title)       # first: on macOS the window is found by it
    except Exception:
        pass
    room = screen_room(fig)
    used = float(scale)
    if room:
        used = min(used, room[0] / px[0], room[1] / px[1])
    used = max(used, 0.1)
    if abs(used - scale) > 1e-4:
        fig.set_size_inches(px[0] * used / dpi, px[1] * used / dpi, forward=True)
    fixed = fix_window_size(fig)
    try:
        fig.canvas.manager.show()
    except Exception:
        plt.show(block=False)
    try:
        fig.canvas.flush_events()
    except Exception:
        pass
    return fig, {"scale": used, "asked": float(scale), "room": room, "fixed": fixed,
                 "platform": window_platform(fig)}


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


def _requested(fig, mgr, win, kind) -> Optional[Tuple[int, int]]:
    """The window size asked for, readable before the window is shown: Qt's and wx's own
    size, else the figure's (no toolbar, so the window is the canvas)."""
    try:
        if kind == "qt":
            return int(win.width()), int(win.height())
        if kind == "wx":
            s = mgr.frame.GetSize()
            return int(s[0]), int(s[1])
        if kind in ("tk", "gtk3", "gtk4"):
            w, h = np.round(np.asarray(fig.get_size_inches(), float) * float(fig.dpi))
            return int(w), int(h)
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
    target = st.get("win") or st.get("req")
    if not target:
        kind = ""                                    # size unknown to the toolkit: the figure's
    w, h = target if target else (None, None)
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

    Best called before the window is first shown (:func:`open_panel_window` does), so that
    the window manager never sees a resizable window.  Tk, Qt, GTK and wx windows are made
    non-resizable.  The macOS backend has no such switch in matplotlib, so there the resize
    control is removed through PyObjC (``pyobjc-framework-Cocoa``) when it is installed.
    Wherever neither works, :func:`keep_window_size` puts the window back on the next GUI
    turn.  Returns True when the toolkit's resizing is off."""
    mgr, win = _parts(fig)
    kind = _kind(mgr, win)
    fig._klip_pin = {"size": np.array(fig.get_size_inches(), float), "dpi": float(fig.dpi),
                     "win": None, "req": _requested(fig, mgr, win, kind), "kind": kind,
                     "opened": time.time(), "asked": 0.0, "misses": 0, "done": False}
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


def keep_window_size(fig) -> Optional[str]:
    """Put ``fig``'s window back at the size :func:`fix_window_size` pinned.

    Called on every GUI turn, and a no-op while the size holds.  The size is read from the
    toolkit window itself: a window manager that resizes a Tk window leaves the figure's own
    size alone.  A maximized window is always un-maximized, and its size is never adopted.
    Within ``PIN_GRACE`` seconds of opening, a smaller size the window manager gives the
    window is kept (a screen smaller than it reported).  Otherwise a resized window is put
    back, at most every 0.5 s.  After ``MAX_MISSES`` put-backs in a row that do not take,
    the window keeps the size it has, and a line for the log is returned (once).  A change
    of the figure's dpi (moving to a screen with another scale) is the toolkit's own doing
    and re-pins the size instead."""
    st = getattr(fig, "_klip_pin", None)
    if not st or st.get("done"):
        return None
    try:
        now = time.time()
        mgr, win = _parts(fig)
        kind = st.get("kind", "")
        if abs(float(fig.dpi) - st["dpi"]) > 1e-6:            # rescaled by the toolkit
            st.update(dpi=float(fig.dpi), size=np.array(fig.get_size_inches(), float), win=None,
                      req=None, opened=now, misses=0)
            if kind == "tk":
                win.geometry("")                              # let the toplevel follow the canvas again
            return None
        ws = _win_size(mgr, win, kind)
        if ws is not None and min(ws) < 50:                   # not mapped yet
            return None
        settling = now - st["opened"] < PIN_GRACE
        if ws is not None:
            mx = _maximized(mgr, win, kind)
            if st["win"] is None and not mx:
                req = st.get("req")
                if req is None or (settling and ws[0] <= req[0] + 1 and ws[1] <= req[1] + 1):
                    st["win"], st["misses"] = ws, 0
                    st["size"] = np.array(fig.get_size_inches(), float)
                    return None
            target = st["win"] or st.get("req")
            if target is None:
                return None
            off = max(abs(ws[0] - target[0]), abs(ws[1] - target[1])) > 1
            if not off and not mx:
                st["misses"] = 0
                return None
            if st["misses"] >= MAX_MISSES:                    # the window manager insists
                st["done"] = True
                return (f"the window manager holds the window at {ws[0]} x {ws[1]}"
                        + (" (maximized)" if mx else "") + "; the panel is drawn to fit it")
        else:                                                 # macOS: the figure's size
            cur = np.array(fig.get_size_inches(), float)
            tol = 1.5 / float(getattr(fig, "_original_dpi", None) or fig.dpi)
            if np.all(np.abs(cur - st["size"]) <= tol):
                st["misses"] = 0
                return None
            if settling:
                st["size"] = cur
                return None
            if st["misses"] >= MAX_MISSES:
                st["done"] = True
                return "the window keeps the size it was given; the panel is drawn to fit it"
        if now - st["asked"] < 0.5:
            return None
        st["asked"], st["misses"] = now, st["misses"] + 1
        _put_back(fig, st, mgr, win, kind)
    except Exception:
        pass
    return None


# -- the panel inside it -----------------------------------------------------------------
def panel_box(fig, px: Tuple[int, int]) -> Tuple[int, int]:
    """The device pixels a ``px`` panel fills in ``fig``'s canvas: as large as fits, at its
    own aspect.  Device pixels, so a 2x screen gives twice the toolkit's size."""
    try:
        W, H = fig.canvas.get_width_height(physical=True)
    except Exception:
        W, H = (int(v) for v in fig.bbox.size)
    s = min(W / float(px[0]), H / float(px[1]))
    return max(1, int(px[0] * s + 1e-6)), max(1, int(px[1] * s + 1e-6))


def _rgba8(img) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(a.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], axis=2)
    return a


def fit_image(img, box: Tuple[int, int]) -> np.ndarray:
    """``img`` (RGB or RGBA, float or uint8) as uint8 RGBA filling ``box`` at its own aspect.
    Unchanged when it already fills it, pixel-doubled for a whole-number enlargement (a 2x
    screen), Lanczos-resampled otherwise."""
    a = _rgba8(img)
    h, w = a.shape[:2]
    s = min(box[0] / float(w), box[1] / float(h))
    nw, nh = min(box[0], max(1, int(round(w * s)))), min(box[1], max(1, int(round(h * s))))
    if (nw, nh) == (w, h):
        return a
    k = int(round(s))
    if k >= 2 and abs(s - k) < 1e-6:
        return np.repeat(np.repeat(a, k, axis=0), k, axis=1)
    from PIL import Image
    return np.asarray(Image.fromarray(a).resize((nw, nh), Image.LANCZOS))


def screen_copy(fig, box: Tuple[int, int]) -> np.ndarray:
    """Offscreen ``fig`` drawn to fill ``box`` device pixels at its own aspect, as uint8
    RGBA: the text is rendered at the screen's resolution rather than resampled to it.
    The figure's dpi is put back afterwards."""
    w0, h0 = fig.bbox.size
    s = min(box[0] / float(w0), box[1] / float(h0))
    dpi0 = fig.dpi
    if abs(s - 1.0) > 1e-6:
        fig.set_dpi(dpi0 * s)
    try:
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba()).copy()
    finally:
        if abs(s - 1.0) > 1e-6:
            fig.set_dpi(dpi0)


def show_panel(fig, img, px: Tuple[int, int]) -> Tuple[int, int]:
    """Show ``img`` in window ``fig`` pixel for pixel, centered on a black ground, at the
    largest size a ``px`` panel fits (resampled once when it is not already that size).
    Returns the box it fills, in device pixels."""
    box = panel_box(fig, px)
    a = fit_image(img, box)
    try:
        W, H = fig.canvas.get_width_height(physical=True)
    except Exception:
        W, H = (int(v) for v in fig.bbox.size)
    xo, yo = max(0, (W - a.shape[1]) // 2), max(0, (H - a.shape[0]) // 2)
    old = getattr(fig, "_klip_panel", None)
    if old is not None and old.get_array().shape == a.shape and (old.ox, old.oy) == (xo, yo):
        old.set_data(a)
    else:
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
        fig._klip_panel = fig.figimage(a, xo=xo, yo=yo, origin="upper", zorder=0)
    return box
