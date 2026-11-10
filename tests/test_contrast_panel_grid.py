"""The contrast panel draws the injection grid as points with bars, joined across the annuli.

From the WISPIT 2 ALES session, 2026-11-10.  Its annuli are 4.5 to 6 px wide, and the grid
insets 1 FWHM from each edge and steps 1 FWHM, so each annulus' grid curve has one separation.
The panel drew it as a line and its 16-84 % as a band, which need two points each, so the panel
showed only the KLIP-FM curves.  What must hold:

* Every grid point is a marker, with its 16-84 % as a bar where the curve has both.
* One line joins the grid points of every annulus in order of separation.
* The other curves are drawn as before.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.container import ErrorbarContainer  # noqa: E402

from klip_tpe.display import panel_contrast_idl  # noqa: E402


def _grid(r, c, lo=None, hi=None):
    return {"r_as": r, "curve": c, "p16": lo if lo is not None else [None] * len(r),
            "p84": hi if hi is not None else [None] * len(r), "n": [12] * len(r)}


def test_one_separation_per_annulus_gives_points_bars_and_one_line():
    curves = [
        ("injection grid (median over PA) annulus 1", _grid([0.30], [2e-3], [1.5e-3], [2.6e-3]), {}),
        ("KLIP-FM cross-check (winner) annulus 1", {"r_as": [0.25, 0.30, 0.35], "curve": [3e-3, 2e-3, 1.5e-3]}, {}),
        ("injection grid (median over PA) annulus 3", _grid([0.65], [4e-4]), {}),
        ("injection grid (median over PA)", _grid([0.47], [8e-4], [6e-4], [1.1e-3]), {}),
    ]
    fig, ax = plt.subplots()
    try:
        panel_contrast_idl(ax, curves, 7.7e-4)
        bars = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
        assert len(bars) == 1
        data_line, _, (barcol,) = bars[0].lines
        assert np.allclose(data_line.get_xdata(), [0.30, 0.47]) and data_line.get_marker() == "o"
        segs = barcol.get_segments()
        assert np.allclose([s[:, 1].min() for s in segs], [1.5e-3, 6e-4])
        assert np.allclose([s[:, 1].max() for s in segs], [2.6e-3, 1.1e-3])
        lines = {(tuple(np.round(ln.get_xdata(), 6)), ln.get_linestyle(), ln.get_marker()) for ln in ax.lines}
        assert ((0.30, 0.47, 0.65), "-", "None") in lines, "one line joins every annulus' points"
        assert ((0.65,), "None", "o") in lines, "a point without its spread is a marker"
        assert ((0.25, 0.30, 0.35), "--", "None") in lines, "the KLIP-FM curve as before"
        assert ax.get_yscale() == "log"
        assert any(t.get_text().startswith("points = injection grid") for t in ax.texts)
    finally:
        plt.close(fig)


def test_a_wide_annulus_keeps_its_points_and_no_band():
    curves = [("injection grid (annulus)", _grid([0.3, 0.4, 0.5], [2e-3, 1e-3, 6e-4], [1.6e-3, 8e-4, 5e-4],
                                                  [2.5e-3, 1.3e-3, 7e-4]), {})]
    fig, ax = plt.subplots()
    try:
        panel_contrast_idl(ax, curves, 1e-3)
        assert not ax.collections or all(not isinstance(c, matplotlib.collections.PolyCollection)
                                         for c in ax.collections), "points and bars, no band"
        (bars,) = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
        assert np.allclose(bars.lines[0].get_xdata(), [0.3, 0.4, 0.5])
    finally:
        plt.close(fig)


def test_the_rebuilt_frames_keep_the_finished_annuli_curves(tmp_path, monkeypatch):
    """``klip-tpe render`` puts every finished annulus' grid curve on the later frames, as the live
    panel does, so the line joins the annuli on a finished run's frames too."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from conftest import build_synthetic_run

    import klip_tpe.display as D
    from klip_tpe import Runner
    from klip_tpe.contrast_grid import ContrastGridConfig
    # the suite turns the grid off (conftest): on here, small
    red, space, obj, samp, cfg = build_synthetic_run(
        n_iter=[4, 4], n_init=2, contrast_grid=ContrastGridConfig(mode="grid", n_pa=4, accept_after=2, drop_after=3))
    d = str(tmp_path / "run")
    results = Runner(red, space, obj, samp, cfg, d, log=lambda s: None).run()
    assert all(r.grid_curve for r in results)
    seen = []

    def capture(ad, i, imgs, p, curves=None, **kw):
        seen.append((ad.annulus, i, [lab for lab, _, _ in curves or []]))
        return None
    monkeypatch.setattr(D, "render_step", capture)
    D.render_steps(d, every=100, out_dir=str(tmp_path / "steps"), movie=False)
    last = {a: labs for a, i, labs in seen}
    assert "injection grid (median over PA)" in last[0] and not any("annulus 1" in s for s in last[0])
    assert "injection grid (median over PA) annulus 1" in last[1] and "injection grid (median over PA)" in last[1]
    assert "KLIP-FM cross-check (winner) annulus 1" in last[1] or not results[0].fm_curve
    first2 = [labs for a, i, labs in seen if a == 1][0]
    assert "injection grid (median over PA) annulus 1" in first2, "on every frame of the later annulus"
