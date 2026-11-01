"""The partition map (display.render_nightmap) with many partitions: its labels, its title
and its layout.  With an integral-field spectrograph's 20 wavelength groups the
partition-effect panel's labels ran into each other and a long run name lost both ends of
the title."""
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from klip_tpe import display
from klip_tpe.display import _fit_suptitle, _partition_tick_labels, annulus_from_records, render_nightmap

LONG = ("mwc758_ales_clike_nov14_allgroups_cmax2e-5_diskneg_rinj0.6_as0.01_kmax15_corr0.94-1_pinc0.5_"
        "anglemax127_nocuts_posthp0")
GROUPS20 = [f"n2_g{i:02d}" for i in range(20)]


def _annulus(names, n=400, run_name="run", seed=0):
    rng = np.random.default_rng(seed)
    space = {"params": [{"name": "bin", "lo": 1, "hi": 30, "kind": "int"}], "partitions": list(names)}
    recs = []
    for i in range(n):
        sel = [p for p in names if rng.random() < 0.5] or [names[0]]
        recs.append({"x": [float(rng.integers(1, 30))], "score": float(rng.normal(len(sel) / len(names), 0.3)),
                     "phase": "random" if i < n // 5 else "tpe", "meta": {"selected": sel}})
    return annulus_from_records(recs, space, {"ann_edges": [10, 20], "n_init": n // 5, "n_iter": n}, 0, 0.01, 4.0,
                                run_name=run_name)


def test_partition_tick_labels_drop_the_shared_prefix_only_when_safe():
    assert _partition_tick_labels(GROUPS20[:3]) == (["g00", "g01", "g02"], "n2_")
    two_nights = ["n1_g00", "n2_g00"]
    assert _partition_tick_labels(two_nights) == (two_nights, "")       # nothing shared up to a separator
    assert _partition_tick_labels(["x_", "x_a"]) == (["x_", "x_a"], "")   # it would leave an empty label
    assert _partition_tick_labels(["n1", "n2"]) == (["n1", "n2"], "")
    assert _partition_tick_labels(["only_one"]) == (["only_one"], "")


def test_a_long_run_name_title_fits_the_figure_and_a_short_one_stays_on_one_line():
    fig = Figure(figsize=(8.0, 7.0))
    FigureCanvasAgg(fig)
    t = _fit_suptitle(fig, LONG, "annulus 1: partition inclusion history")
    assert t.get_text() == LONG + "\nannulus 1: partition inclusion history"
    assert t.get_window_extent(fig.canvas.get_renderer()).width <= 0.98 * fig.bbox.width
    fig = Figure(figsize=(8.0, 7.0))
    FigureCanvasAgg(fig)
    t = _fit_suptitle(fig, "run_20261031", "annulus 1: partition inclusion history")
    assert t.get_text() == "run_20261031 annulus 1: partition inclusion history" and t.get_fontsize() == 9.0


def test_partition_effect_labels_lose_the_prefix_and_stand_on_end_past_ten_partitions():
    with display._rc():
        fig = Figure(figsize=(8.0, 3.0))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        display.panel_partition_effect(ax, _annulus(GROUPS20))
        labels = ax.get_xticklabels()
        assert [lab.get_text() for lab in labels] == [g[3:] for g in GROUPS20]
        assert all(lab.get_rotation() == 90 for lab in labels)
        assert ax.get_xlabel() == "partition (n2_…)"
        fig = Figure(figsize=(8.0, 3.0))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        display.panel_partition_effect(ax, _annulus(["n1", "n2"]))
        assert [lab.get_text() for lab in ax.get_xticklabels()] == ["n1", "n2"]
        assert ax.get_xlabel() == "partition" and ax.get_xticklabels()[0].get_rotation() == 0


@pytest.mark.parametrize("names, tall", [(["n1", "n2"], False), (GROUPS20, True),
                                          ([f"n{1 + i // 20}_g{i % 20:02d}" for i in range(40)], True)])
def test_many_partitions_get_full_width_rows(tmp_path, names, tall):
    out = str(tmp_path / "partition_map.png")
    assert render_nightmap(_annulus(names, run_name=LONG), out) == out
    w, h = Image.open(out).size
    assert (h / w > 1.0) == tall, (w, h)
