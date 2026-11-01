"""The convergence panels' bottom comes from the scores alone, 0 in a floored run, and a draw
bar that reaches below it is cut there and marked (Kevin Wagner, 2026-11-01: one low draw of
three squashed the scores)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np                                # noqa: E402
from matplotlib.figure import Figure              # noqa: E402

from klip_tpe import display as D                 # noqa: E402
from klip_tpe.plots import draw_ranges            # noqa: E402

SCORES = [0.0, 1.5, 2.5, 0.0]
DRAWS = [[-2.0, 0.5, 1.0], [1.0, 1.5, 2.0], [-3.0, 4.0, 6.5], [-1.0, -0.5, 1.2]]


def _ad():
    recs = [{"x": [], "score": s, "raw_score": s, "phase": "tpe", "meta": {"draw_scores": d, "draw_n": len(d)}}
            for s, d in zip(SCORES, DRAWS)]
    return D.annulus_from_records(recs, {}, {"ann_edges": [8, 30]}, 0, 0.05, 4.0)


def _triangles(ax, floor):
    """x of the markers drawn at the floor by draw_ranges (the score points are circles)."""
    xs = []
    for c in ax.collections:
        off = np.asarray(c.get_offsets())
        if off.size and np.allclose(off[:, 1], floor) and len(c.get_paths()) == 1:
            verts = c.get_paths()[0].vertices
            if len(verts) <= 5:                   # a triangle's path, not a circle's
                xs.extend(off[:, 0].tolist())
    return sorted(xs)


def test_draw_ranges_cuts_at_the_floor_and_marks_the_cut_bars():
    fig = Figure()
    ax = fig.add_subplot(111)
    assert draw_ranges(ax, np.arange(4), DRAWS, floor=0.0) == 4
    lows = sorted(float(seg[0][1]) for c in ax.collections if hasattr(c, "get_segments") for seg in c.get_segments())
    assert min(lows) == 0.0
    assert _triangles(ax, 0.0) == [0.0, 2.0, 3.0]


def test_the_convergence_panel_starts_at_zero():
    fig = Figure()
    ax = fig.add_subplot(111)
    D.panel_convergence_idl(ax, _ad())
    lo, hi = ax.get_ylim()
    assert lo == 0.0 and hi >= 6.5, "the bottom from the scores, the bars' tops kept"
    assert _triangles(ax, 0.0) == [0.0, 2.0, 3.0]
    assert any("draw below the axis" in t.get_text() for t in ax.texts)


def test_the_trace_panel_starts_at_zero():
    fig = Figure()
    ax = fig.add_subplot(111)
    D.panel_trace(ax, _ad())
    lo, hi = ax.get_ylim()
    assert lo == 0.0 and hi >= 6.5
    assert "draw below the axis" in [t.get_text() for t in ax.get_legend().get_texts()]


def test_a_run_with_negative_scores_keeps_them_in_view():
    recs = [{"x": [], "score": s, "raw_score": s, "phase": "tpe", "meta": {"draw_scores": d, "draw_n": 3}}
            for s, d in zip([-0.5, 1.0], [[-2.0, -0.5, 1.0], [0.5, 1.0, 1.5]])]
    ad = D.annulus_from_records(recs, {}, {"ann_edges": [8, 30]}, 0, 0.05, 4.0)
    fig = Figure()
    ax = fig.add_subplot(111)
    D.panel_convergence_idl(ax, ad)
    assert ax.get_ylim()[0] == -0.5
    assert _triangles(ax, -0.5) == [0.0]
