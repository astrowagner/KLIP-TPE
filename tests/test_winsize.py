"""The live window and ``klip-tpe view`` keep one size, and show the panel pixel for pixel."""
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


def _agg(w, h, dpi=100.0):
    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    return fig


# -------------------------------------------------------------------- holding the size
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


def test_the_size_asked_for_is_known_before_the_window_appears():
    """A Tk window not yet shown reads 1 x 1, so the size asked for comes from the figure
    (the window is the canvas, with no toolbar); Qt knows its own."""
    tk = _tk(size=(1, 1))
    fig = _Fig(tk)
    winsize.fix_window_size(fig)
    assert fig._klip_pin["req"] == (1017, 544)
    qt = _qt(size=(1424, 762))
    fig = _Fig(qt)
    winsize.fix_window_size(fig)
    assert fig._klip_pin["req"] == (1424, 762)


def test_a_window_without_a_switch_is_pinned_all_the_same():
    """No toolkit window (Agg, or the macOS backend without PyObjC): False, but pinned."""
    fig = _agg(1017, 544)
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


def test_a_window_maximized_as_it_opens_is_put_back_and_its_size_never_adopted():
    """What GNOME did to the 1850 x 990 window on a 1920 x 1080 screen: maximized as it
    appeared.  Adopting that size during the grace period made every later put-back aim
    at it, so the window was un-maximized and then resized over and over."""
    qt = _qt(size=(1850, 990))
    fig = _Fig(qt, inches=(18.5, 9.9))
    winsize.fix_window_size(fig)
    qt._st.update(size=(1920, 1043), maximized=True)     # within the grace period
    winsize.keep_window_size(fig)
    assert fig._klip_pin["win"] is None
    assert ("showNormal", ()) in qt.calls and ("resize", (1850, 990)) in qt.calls
    qt._st.update(size=(1850, 990), maximized=False)     # back where it was asked
    winsize.keep_window_size(fig)
    assert fig._klip_pin["win"] == (1850, 990)


def test_a_smaller_size_given_as_it_opens_is_kept_a_larger_one_is_not():
    """A screen smaller than it reported: the window manager's smaller size is kept.  A
    larger one is put back."""
    qt = _qt()
    fig = _Fig(qt)
    winsize.fix_window_size(fig)
    qt._st["size"] = (900, 500)
    winsize.keep_window_size(fig)
    assert fig._klip_pin["win"] == (900, 500) and not any(c[0] == "resize" for c in qt.calls)
    qt2 = _qt()
    fig2 = _Fig(qt2)
    winsize.fix_window_size(fig2)
    qt2._st["size"] = (1300, 700)
    winsize.keep_window_size(fig2)
    assert fig2._klip_pin["win"] is None and ("resize", (1017, 544)) in qt2.calls


def test_put_backs_stop_after_ten_refusals(monkeypatch):
    """A window manager that holds the window at its own size (a tiling one) is not fought
    for ever: after ``MAX_MISSES`` refused put-backs the window keeps its size, and one
    line says so."""
    qt = _qt()
    fig = _Fig(qt)
    winsize.fix_window_size(fig)
    winsize.keep_window_size(fig)
    _settled(fig)
    qt._st["size"] = (1300, 700)
    clock = [winsize.time.time() + 100.0]
    monkeypatch.setattr(winsize.time, "time", lambda: clock[0])
    notes = []
    for _ in range(14):
        clock[0] += 0.6
        notes.append(winsize.keep_window_size(fig))
    asks = [c for c in qt.calls if c[0] == "resize"]
    assert len(asks) == winsize.MAX_MISSES and asks[0] == ("resize", (1017, 544))
    said = [n for n in notes if n]
    assert len(said) == 1 and "1300 x 700" in said[0]
    clock[0] += 60.0
    assert winsize.keep_window_size(fig) is None and len([c for c in qt.calls if c[0] == "resize"]) == 10


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
    fig = _agg(1017, 544)
    winsize.fix_window_size(fig)
    _settled(fig)
    fig.set_size_inches(6.0, 4.0, forward=False)           # what a backend does on a user resize
    winsize.keep_window_size(fig)
    assert np.allclose(fig.get_size_inches(), [10.17, 5.44])


# -------------------------------------------------------------------- opening the window
def test_the_window_is_fitted_and_fixed_before_it_is_shown(monkeypatch):
    """GNOME maximizes a resizable window covering most of the screen the moment it appears.
    So the window is made hidden, shrunk to the room on the screen, fixed, and only then
    shown."""
    import matplotlib.pyplot as plt
    events = []
    qt = _qt(size=(1850, 990))
    qt.setFixedSize = lambda size: events.append(("fixed", tuple(size)))
    fig = _Fig(qt, inches=(18.5, 9.9))
    fig.patch = type("P", (), {"set_facecolor": lambda self, c: None})()
    mgr = fig.canvas.manager
    mgr.set_window_title = lambda t: events.append(("title", t))
    mgr.show = lambda: events.append(("show", plt.isinteractive()))
    fig.canvas.flush_events = lambda: None

    def _resize(*a, forward=True):
        fig._inches = np.array(a, float)
        qt._st["size"] = tuple(int(round(v * 100)) for v in a)
        events.append(("resize", qt._st["size"]))
    fig.set_size_inches = _resize

    def _figure(**kw):
        events.append(("created", plt.isinteractive()))
        return fig
    monkeypatch.setattr(plt, "figure", _figure)
    monkeypatch.setattr(winsize, "screen_room", lambda f: (1424, 852))     # a 1440 x 900 screen
    monkeypatch.setattr(winsize, "window_platform", lambda f: "x11")
    plt.ion()
    try:
        got, info = winsize.open_panel_window((1850, 990), 1.0, 100.0, "klip-tpe live -- t")
    finally:
        plt.ioff()
    names = [e[0] for e in events]
    assert names == ["created", "title", "resize", "fixed", "show"]
    assert events[0] == ("created", False)                 # hidden: pyplot does not show it
    assert events[2] == ("resize", (1424, 762)) and events[3] == ("fixed", (1424, 762))
    assert info["scale"] == pytest.approx(1424 / 1850) and info["fixed"] and info["asked"] == 1.0


def test_a_window_that_fits_is_not_shrunk(monkeypatch):
    import matplotlib.pyplot as plt
    qt = _qt(size=(1850, 990))
    fig = _Fig(qt, inches=(18.5, 9.9))
    fig.patch = type("P", (), {"set_facecolor": lambda self, c: None})()
    fig.canvas.manager.set_window_title = lambda t: None
    fig.canvas.manager.show = lambda: None
    fig.canvas.flush_events = lambda: None
    resized = []
    fig.set_size_inches = lambda *a, forward=True: resized.append(a)
    monkeypatch.setattr(plt, "figure", lambda **kw: fig)
    monkeypatch.setattr(winsize, "screen_room", lambda f: (1904, 1032))     # 1920 x 1080
    monkeypatch.setattr(winsize, "window_platform", lambda f: "x11")
    _, info = winsize.open_panel_window((1850, 990), 1.0, 100.0, "t")
    assert info["scale"] == 1.0 and not resized


# -------------------------------------------------------------------- the panel, sharp
def test_the_panel_box_is_the_largest_fit_at_the_panel_aspect():
    assert winsize.panel_box(_agg(1850, 990), (1850, 990)) == (1850, 990)
    assert winsize.panel_box(_agg(1017, 544), (1850, 990)) == (1016, 544)
    assert winsize.panel_box(_agg(3700, 1980, dpi=200.0), (1850, 990)) == (3700, 1980)
    assert winsize.panel_box(_agg(1920, 1043), (1850, 990)) == (1920, 1027)


def test_fit_image_keeps_a_fitting_panel_and_doubles_for_a_2x_screen():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (990, 1850, 4), dtype=np.uint8)
    assert winsize.fit_image(a, (1850, 990)) is not None
    assert np.array_equal(winsize.fit_image(a, (1850, 990)), a)
    big = winsize.fit_image(a, (3700, 1980))
    assert big.shape == (1980, 3700, 4) and np.array_equal(big[::2, ::2], a)
    small = winsize.fit_image(a, (1016, 544))
    assert small.shape[1] <= 1016 and small.shape[0] <= 544 and (small.shape[1] == 1016 or small.shape[0] == 544)
    f = winsize.fit_image(rng.random((99, 185, 3)), (185, 99))            # float RGB from imread
    assert f.dtype == np.uint8 and f.shape == (99, 185, 4) and np.all(f[..., 3] == 255)


def test_show_panel_draws_the_panel_pixel_for_pixel():
    """No resampling when the panel fills the window: the canvas holds the PNG's pixels."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (990, 1850, 4), dtype=np.uint8)
    a[..., 3] = 255
    win = _agg(1850, 990)
    box = winsize.show_panel(win, a, (1850, 990))
    win.canvas.draw()
    buf = np.asarray(win.canvas.buffer_rgba())
    assert box == (1850, 990) and np.array_equal(buf[..., :3], a[..., :3])
    # a window held wider than the panel: centered on black, still 1:1, never stretched
    wide = _agg(1950, 990)
    wide.patch.set_facecolor("black")
    winsize.show_panel(wide, a, (1850, 990))
    wide.canvas.draw()
    buf = np.asarray(wide.canvas.buffer_rgba())
    assert np.array_equal(buf[:, 50:1900, :3], a[..., :3]) and np.all(buf[:, :50, :3] == 0)


def test_screen_copy_draws_at_the_window_size_and_restores_the_dpi():
    fig = _agg(1850, 990)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.text(0.5, 0.5, "S/N = 5", fontsize=20)
    small = winsize.screen_copy(fig, (1016, 544))
    assert small.shape[:2] in ((544, 1016), (543, 1016), (544, 1015)) and fig.dpi == 100.0
    big = winsize.screen_copy(fig, (3700, 1980))
    assert big.shape[:2] == (1980, 3700) and fig.dpi == 100.0


# -------------------------------------------------------------------- Wayland
_GCS_LOGICAL = ("(uint32 2, [(('Meta-0', 'MetaVendor', 'MetaVirtualMonitor', '0x00'), [('2880x1620@60.000', 2880, "
                "1620, 60.0, 1.0, [1.0, 1.25, 1.5, 2.0], {'is-current': <true>, 'is-preferred': <true>})], "
                "{'is-builtin': <false>, 'display-name': <'MetaVendor'>})], [(0, 0, 1.5, uint32 0, true, [('Meta-0', "
                "'MetaVendor', 'MetaVirtualMonitor', '0x00')], @a{sv} {})], {'renderer': <'native'>, 'layout-mode': "
                "<uint32 1>, 'supports-changing-layout-mode': <true>, 'legacy-ui-scaling-factor': <1>})")
_GCS_PHYSICAL = _GCS_LOGICAL.replace("(0, 0, 1.5,", "(0, 0, 2.0,").replace("<uint32 1>", "<uint32 2>")


def test_gnome_scaling_is_read_from_mutter(monkeypatch):
    out = {"gdbus": _GCS_LOGICAL, "gsettings": "@as []"}

    class _R:
        def __init__(self, text):
            self.stdout = text
    monkeypatch.setattr(winsize.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(winsize.subprocess, "run", lambda cmd, **kw: _R(out[os.path.basename(cmd[0])]))
    assert winsize._gnome_scaling() == {"logical": True, "scales": [1.5], "native_x11": False}
    out["gdbus"] = _GCS_PHYSICAL
    assert winsize._gnome_scaling() == {"logical": False, "scales": [2.0], "native_x11": False}
    out["gsettings"] = "['scale-monitor-framebuffer', 'xwayland-native-scaling']"
    assert winsize._gnome_scaling()["native_x11"] is True
    out["gdbus"] = ""
    assert winsize._gnome_scaling() is None                 # not GNOME, or no session bus


def _wayland_env(monkeypatch, scaling=None):
    for k in ("QT_QPA_PLATFORM", "GDK_BACKEND", "KLIP_TPE_WAYLAND"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "ubuntu:GNOME")
    monkeypatch.setattr(winsize, "_qt_app_running", lambda: False)
    monkeypatch.setattr(winsize.ctypes.util, "find_library", lambda name: "libxcb-cursor.so.0")
    monkeypatch.setattr(winsize, "_gnome_scaling", lambda: scaling)
    monkeypatch.setattr(winsize, "_SCALING", {})


def test_gnome_wayland_sessions_open_the_window_through_xwayland(monkeypatch):
    _wayland_env(monkeypatch, {"logical": False, "scales": [2.0], "native_x11": False})
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


def test_fractional_scaling_keeps_the_window_native(monkeypatch):
    """In GNOME's logical layout (fractional scaling) an XWayland window is drawn at scale 1
    and enlarged, which blurs the panel.  The window stays native there."""
    _wayland_env(monkeypatch, {"logical": True, "scales": [1.5], "native_x11": False})
    note = winsize.prefer_x11_on_wayland()
    assert "QT_QPA_PLATFORM" not in os.environ and "1.5" in note and "native" in note
    # scale 1 in the logical layout, or XWayland scaling natively (GNOME 47+): XWayland is sharp
    for sc in ({"logical": True, "scales": [1.0], "native_x11": False},
               {"logical": True, "scales": [1.5], "native_x11": True}):
        _wayland_env(monkeypatch, sc)
        winsize.prefer_x11_on_wayland()
        assert os.environ.get("QT_QPA_PLATFORM") == "xcb;wayland"
    # and KLIP_TPE_WAYLAND=x11 forces it
    _wayland_env(monkeypatch, {"logical": True, "scales": [1.5], "native_x11": False})
    monkeypatch.setenv("KLIP_TPE_WAYLAND", "x11")
    winsize.prefer_x11_on_wayland()
    assert os.environ.get("QT_QPA_PLATFORM") == "xcb;wayland"


def test_other_desktops_and_missing_pieces_keep_qt_native(monkeypatch):
    _wayland_env(monkeypatch)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")           # KWin draws the frame itself
    assert winsize.prefer_x11_on_wayland() is None and "QT_QPA_PLATFORM" not in os.environ
    _wayland_env(monkeypatch)
    monkeypatch.delenv("DISPLAY", raising=False)
    assert winsize.prefer_x11_on_wayland() is None          # no XWayland at all
    _wayland_env(monkeypatch)
    monkeypatch.setenv("GDK_BACKEND", "x11")
    monkeypatch.setattr(winsize.ctypes.util, "find_library", lambda name: None)
    note = winsize.prefer_x11_on_wayland()
    assert "QT_QPA_PLATFORM" not in os.environ and "libxcb-cursor0" in note


# -------------------------------------------------------------------- wiring
def test_the_live_window_and_the_viewer_both_hold_their_size():
    here = os.path.dirname(winsize.__file__)
    disp = open(os.path.join(here, "display.py")).read()
    view = open(os.path.join(here, "viewer.py")).read()
    for src in (disp, view):
        assert "open_panel_window(" in src and "keep_window_size(" in src and "show_panel(" in src
        assert "prefer_x11_on_wayland(" in src
        assert "imshow(" not in src[src.index("open_panel_window("):src.index("open_panel_window(") + 6000]


def test_the_render_thread_draws_a_copy_at_the_window_size(tmp_path, monkeypatch):
    """The window shows text rendered for its screen.  After each panel the render thread
    draws the figure again at the window's pixel size, unless the PNG is that size."""
    from klip_tpe import display as dm, parallel
    hooks = list(parallel.IDLE_HOOKS)
    d = dm.LiveDisplay(str(tmp_path), show=True, movie=False, log=lambda m: None)
    d.inline = False
    png = str(tmp_path / "step0000.png")

    def _job():
        f = d._figure()
        f.clf()
        f.text(0.1, 0.5, "S/N = 5", fontsize=30)
        f.savefig(png, dpi=d.dpi)
    try:
        monkeypatch.setattr(dm.LiveDisplay, "_win_px", (1016, 544))
        d._submit_render(png, _job)
        arr = d._renders[-1][0].result(timeout=120)
        assert isinstance(arr, np.ndarray) and dm._fills(arr, (1016, 544))
        monkeypatch.setattr(dm.LiveDisplay, "_win_px", (1850, 990))
        d._submit_render(png, _job)
        assert d._renders[-1][0].result(timeout=120) is None      # the PNG on disk is that size
    finally:
        parallel.IDLE_HOOKS[:] = hooks
        if getattr(d, "_rex", None) is not None:
            d._rex.shutdown(wait=True)


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
