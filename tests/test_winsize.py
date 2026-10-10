"""The live window and ``klip-tpe view`` keep one size: resizing off, and put back if not."""
from __future__ import annotations

import os

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from klip_tpe import winsize


class _Rec:
    """Records the calls made on it; ``_st`` holds what the getters return."""

    def __init__(self, names, **st):
        self.calls = []
        self._st = dict(st)
        for n in names:
            setattr(self, n, (lambda name: (lambda *a: self._call(name, a)))(n))

    def _call(self, name, a):
        self.calls.append((name, a))
        return self._st.get(name)


def _tk(size=(1017, 544), zoomed=False):
    w = _Rec(["resizable", "geometry", "wm_geometry", "state"], state="zoomed" if zoomed else "normal")
    w.winfo_width = lambda: w._st["size"][0]
    w.winfo_height = lambda: w._st["size"][1]
    w.attributes = lambda *a: False if len(a) == 1 else w.calls.append(("attributes", a))
    w._st.update(size=size)
    return w


def _qt(size=(1017, 544), maximized=False):
    w = _Rec(["setFixedSize", "showNormal", "resize"])
    w._st.update(size=size, maximized=maximized)
    w.size = lambda: w._st["size"]
    w.width = lambda: w._st["size"][0]
    w.height = lambda: w._st["size"][1]
    w.isMaximized = lambda: w._st["maximized"]
    w.isFullScreen = lambda: False
    return w


class _Fig:
    """Just enough of a figure for the pinning code: a manager with a window, and a size."""

    def __init__(self, win=None, frame=None, inches=(10.17, 5.44), dpi=100.0):
        mgr = type("M", (), {})()
        mgr.window = win
        if frame is not None:
            mgr.frame = frame
        self.canvas = type("C", (), {"manager": mgr})()
        self._inches, self.dpi = np.array(inches, float), dpi

    def get_size_inches(self):
        return self._inches

    def set_size_inches(self, *a, forward=True):
        self._inches = np.array(a, float)


def _settled(fig):
    fig._klip_pin["opened"] -= 10 * winsize.PIN_GRACE          # past the grace period


def test_each_toolkit_has_its_resizing_switched_off():
    """Tk ``resizable``, Qt ``setFixedSize``, GTK ``set_resizable``, wx min = max size."""
    tk = _tk()
    assert winsize.fix_window_size(_Fig(tk)) and ("resizable", (False, False)) in tk.calls
    qt = _qt()
    assert winsize.fix_window_size(_Fig(qt)) and qt.calls == [("setFixedSize", ((1017, 544),))]
    gtk3 = _Rec(["set_resizable", "resize", "get_size"])
    assert winsize.fix_window_size(_Fig(gtk3)) and gtk3.calls[0] == ("set_resizable", (False,))
    wx = _Rec(["SetMinSize", "SetMaxSize"])
    wx.GetSize = lambda: (1017, 544)
    assert winsize.fix_window_size(_Fig(None, frame=wx))
    assert wx.calls == [("SetMinSize", ((1017, 544),)), ("SetMaxSize", ((1017, 544),))]


def test_a_window_without_a_switch_is_pinned_all_the_same():
    """No toolkit window (Agg, or the macOS backend without PyObjC): False, but pinned."""
    fig = Figure(figsize=(10.17, 5.44), dpi=100)
    FigureCanvasAgg(fig)
    assert winsize.fix_window_size(fig) is False
    assert np.allclose(fig._klip_pin["size"], [10.17, 5.44])


def test_a_tk_window_the_window_manager_resized_is_put_back():
    """The size is read from the toplevel, and restored with its own geometry: a window
    manager that resizes a Tk window leaves the figure's size alone, and Tk then ignores a
    size the canvas asks for."""
    tk = _tk()
    fig = _Fig(tk)
    winsize.fix_window_size(fig)
    winsize.keep_window_size(fig)                         # first look: pins 1017 x 544
    _settled(fig)
    tk._st["size"] = (1400, 800)                         # resized from outside
    winsize.keep_window_size(fig)
    assert ("geometry", ("1017x544",)) in tk.calls
    geoms = lambda: [c for c in tk.calls if c[0] == "geometry"]
    n = len(geoms())
    tk._st["size"] = (1017, 544)
    winsize.keep_window_size(fig)
    assert len(geoms()) == n                               # nothing more while the size holds


def test_a_maximized_qt_window_is_unmaximized_and_resized():
    """GNOME maximizes a native Wayland Qt window whose size is fixed."""
    qt = _qt()
    fig = _Fig(qt)
    winsize.fix_window_size(fig)
    winsize.keep_window_size(fig)
    _settled(fig)
    qt._st.update(size=(1920, 1162), maximized=True)
    winsize.keep_window_size(fig)
    assert ("showNormal", ()) in qt.calls and ("resize", (1017, 544)) in qt.calls


def test_sizes_are_kept_only_while_the_window_settles(monkeypatch):
    """A size the window manager gives a window as it opens (a screen smaller than the
    window) is kept.  After ``PIN_GRACE`` a resize is always put back, and requests that do
    not take slow down to one every 5 s instead of stopping."""
    qt = _qt()
    fig = _Fig(qt)
    winsize.fix_window_size(fig)
    qt._st["size"] = (900, 500)                          # the window manager, at placement
    winsize.keep_window_size(fig)
    assert fig._klip_pin["win"] == (900, 500) and not any(c[0] == "resize" for c in qt.calls)
    _settled(fig)
    qt._st["size"] = (1300, 700)                         # the user, later
    clock = [winsize.time.time() + 100.0]
    monkeypatch.setattr(winsize.time, "time", lambda: clock[0])
    for _ in range(12):                                    # refused every time
        clock[0] += 0.6
        winsize.keep_window_size(fig)
    asks = [c for c in qt.calls if c[0] == "resize"]
    assert len(asks) == 10 and asks[0] == ("resize", (900, 500))
    clock[0] += 5.1
    winsize.keep_window_size(fig)
    assert len([c for c in qt.calls if c[0] == "resize"]) == 11


def test_a_dpi_change_repins_instead_of_fighting():
    """Moving to a screen with another scale changes the figure's dpi and the window's
    pixels.  That is the toolkit's doing, so the size is pinned again."""
    tk = _tk()
    fig = _Fig(tk)
    winsize.fix_window_size(fig)
    winsize.keep_window_size(fig)
    _settled(fig)
    fig.dpi, tk._st["size"] = 200.0, (2034, 1088)
    winsize.keep_window_size(fig)
    assert ("geometry", ("",)) in tk.calls and fig._klip_pin["win"] is None
    winsize.keep_window_size(fig)                          # inside the new grace: adopted
    assert fig._klip_pin["win"] == (2034, 1088)


def test_a_resized_figure_without_a_toolkit_window_is_put_back():
    """The macOS backend exposes no window object: the figure's size is the measure."""
    fig = Figure(figsize=(10.17, 5.44), dpi=100)
    FigureCanvasAgg(fig)
    winsize.fix_window_size(fig)
    _settled(fig)
    fig.set_size_inches(6.0, 4.0, forward=False)           # what a backend does on a user resize
    winsize.keep_window_size(fig)
    assert np.allclose(fig.get_size_inches(), [10.17, 5.44])


def test_wayland_sessions_open_the_window_through_xwayland(monkeypatch):
    for k in ("QT_QPA_PLATFORM", "GDK_BACKEND", "KLIP_TPE_WAYLAND"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(winsize, "_qt_app_running", lambda: False)
    monkeypatch.setattr(winsize.ctypes.util, "find_library", lambda name: "libxcb-cursor.so.0")
    note = winsize.prefer_x11_on_wayland()
    assert os.environ["QT_QPA_PLATFORM"] == "xcb;wayland" and "XWayland" in note
    if "gi.repository.Gtk" not in winsize.sys.modules:
        assert os.environ["GDK_BACKEND"] == "x11,wayland"
    # the user's own settings win, and the opt-out is honored
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland")
    monkeypatch.setenv("GDK_BACKEND", "wayland")
    assert winsize.prefer_x11_on_wayland() is None and os.environ["QT_QPA_PLATFORM"] == "wayland"
    monkeypatch.delenv("QT_QPA_PLATFORM")
    monkeypatch.setenv("KLIP_TPE_WAYLAND", "native")
    assert winsize.prefer_x11_on_wayland() is None and "QT_QPA_PLATFORM" not in os.environ


def test_no_xwayland_or_no_xcb_plugin_keeps_qt_native(monkeypatch):
    for k in ("QT_QPA_PLATFORM", "GDK_BACKEND", "KLIP_TPE_WAYLAND"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert winsize.prefer_x11_on_wayland() is None          # no XWayland at all
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("GDK_BACKEND", "x11")
    monkeypatch.setattr(winsize, "_qt_app_running", lambda: False)
    monkeypatch.setattr(winsize.ctypes.util, "find_library", lambda name: None)
    note = winsize.prefer_x11_on_wayland()
    assert "QT_QPA_PLATFORM" not in os.environ and "libxcb-cursor0" in note


def test_the_live_window_and_the_viewer_both_hold_their_size():
    here = os.path.dirname(winsize.__file__)
    disp = open(os.path.join(here, "display.py")).read()
    view = open(os.path.join(here, "viewer.py")).read()
    for src in (disp, view):
        assert "fix_window_size(" in src and "keep_window_size(" in src
        assert "prefer_x11_on_wayland(" in src
        assert 'aspect="auto"' not in src[src.index("fix_window_size("):]


def test_the_inline_panel_has_a_fixed_size(tmp_path, monkeypatch):
    """Jupyter: width and height both given, unconfined -- the notebook's width does not
    rescale it, and an update does not collapse the cell while the next frame loads."""
    ipd = pytest.importorskip("IPython.display")
    from klip_tpe import display as dm
    monkeypatch.setattr(dm, "_in_notebook", lambda: True)
    made = []

    class _Img:
        def __init__(self, **kw):
            made.append(kw)

    monkeypatch.setattr(ipd, "Image", _Img)
    monkeypatch.setattr(ipd, "display", lambda img, display_id=False: type("H", (), {"update": lambda s, i: None})())
    png = tmp_path / "p.png"
    fig = Figure(figsize=(1, 1), dpi=10)
    FigureCanvasAgg(fig).print_png(str(png))
    d = dm.LiveDisplay(str(tmp_path), show="inline", window_scale=0.55)
    d._inline_show(png=str(png))
    assert made and made[0]["width"] == 1017 and made[0]["height"] == 544 and made[0]["unconfined"] is True
