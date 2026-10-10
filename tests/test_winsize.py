"""The live window and ``klip-tpe view`` keep one size: resizing off, and put back if not."""
from __future__ import annotations

import os

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from klip_tpe import winsize


class _Win:
    """Stands in for a toolkit window; records what the pinning asked of it."""

    def __init__(self, *methods):
        self.calls = []
        for m in methods:
            setattr(self, m, (lambda name: (lambda *a: self.calls.append((name, a))))(m))

    def size(self):
        return (1017, 544)


class _Fig:
    def __init__(self, win=None, frame=None):
        mgr = type("M", (), {})()
        mgr.window = win
        if frame is not None:
            mgr.frame = frame
        self.canvas = type("C", (), {"manager": mgr})()

    def get_size_inches(self):
        return np.array([10.17, 5.44])


def test_each_toolkit_has_its_resizing_switched_off():
    """Tk ``resizable``, Qt ``setFixedSize``, GTK ``set_resizable``, wx min = max size."""
    tk = _Win("resizable")
    assert winsize.fix_window_size(_Fig(tk)) and tk.calls == [("resizable", (False, False))]
    qt = _Win("setFixedSize")
    assert winsize.fix_window_size(_Fig(qt)) and qt.calls == [("setFixedSize", ((1017, 544),))]
    gtk = _Win("set_resizable")
    assert winsize.fix_window_size(_Fig(gtk)) and gtk.calls == [("set_resizable", (False,))]
    wx = _Win("SetMinSize", "SetMaxSize")
    wx.GetSize = lambda: (1017, 544)
    assert winsize.fix_window_size(_Fig(None, frame=wx))
    assert wx.calls == [("SetMinSize", ((1017, 544),)), ("SetMaxSize", ((1017, 544),))]


def test_a_window_without_a_switch_is_pinned_all_the_same():
    """No toolkit window (Agg, or the macOS backend without PyObjC): False, but pinned."""
    fig = Figure(figsize=(10.17, 5.44), dpi=100)
    FigureCanvasAgg(fig)
    assert winsize.fix_window_size(fig) is False
    assert np.allclose(fig._klip_pin["size"], [10.17, 5.44])


def test_a_resized_window_is_put_back():
    fig = Figure(figsize=(10.17, 5.44), dpi=100)
    FigureCanvasAgg(fig)
    winsize.fix_window_size(fig)
    fig.set_size_inches(6.0, 4.0, forward=False)        # what a backend does on a user resize
    winsize.keep_window_size(fig)
    assert np.allclose(fig.get_size_inches(), [10.17, 5.44])
    winsize.keep_window_size(fig)                        # and nothing more while the size holds
    assert np.allclose(fig.get_size_inches(), [10.17, 5.44]) and fig._klip_pin["seen"] is None


def test_a_size_the_screen_forces_is_kept_rather_than_fought(monkeypatch):
    """A screen smaller than the window: the window manager refuses the pinned size.  After
    ``PIN_GRACE`` the size it gave is kept, instead of asking for the other one forever."""
    fig = Figure(figsize=(18.5, 9.9), dpi=100)
    FigureCanvasAgg(fig)
    winsize.fix_window_size(fig)
    real = fig.set_size_inches
    asked = []
    fig.set_size_inches = lambda *a, forward=True: asked.append(a)    # refused, size unchanged
    real(14.4, 8.0, forward=False)
    winsize.keep_window_size(fig)
    assert len(asked) == 1 and np.allclose(asked[0], [18.5, 9.9])
    monkeypatch.setattr(winsize, "PIN_GRACE", 0.0)
    winsize.keep_window_size(fig)
    assert np.allclose(fig._klip_pin["size"], [14.4, 8.0])
    winsize.keep_window_size(fig)
    assert len(asked) == 1                                           # no more asking


def test_the_live_window_and_the_viewer_both_hold_their_size():
    here = os.path.dirname(winsize.__file__)
    disp = open(os.path.join(here, "display.py")).read()
    view = open(os.path.join(here, "viewer.py")).read()
    for src in (disp, view):
        assert "fix_window_size(" in src and "keep_window_size(" in src
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
