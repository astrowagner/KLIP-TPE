"""The live panel's settings block and upper-right region (MWC 758 ALES session, 2026-11-02, on a
frame of mwc758_ales_repeat: 20 wavelength groups, five global reduction parameters).

* A partition's selection flag does not make the reduction parameters per-partition: global
  bin, filter, angsep, corr and k print once each, and the corner pools the one landscape.
* The per-group k is each value once with its count, the groups are named without their shared
  prefix with runs merged, and every line is fitted to its column as measured.
* The upper-right region is laid out from the corner's diagonal cells and each box's whole
  extent: for 2 to 20 partitions and 2 to 9 corner parameters no two axes' boxes overlap, and
  every text lies inside the figure and inside its own column.
"""
from __future__ import annotations

import types

import matplotlib

matplotlib.use("Agg")
import numpy as np                                # noqa: E402
import pytest                                     # noqa: E402
from matplotlib.figure import Figure              # noqa: E402

from klip_tpe import display as D                 # noqa: E402

SPECS = [("bin", 5, 30, "int"), ("filter", 0, 25, "int"), ("angsep", 0.0, 3.0, "float"),
         ("corr_thresh", 0.94, 1.0, "float"), ("k_klip", 1, 15, "int"), ("n_ang", 1, 10, "int"),
         ("anglemax", 10, 90, "float"), ("noise_max", 1, 5, "float"), ("zone_in", 0, 5, "float")]


def _ad(npart=20, nbase=5, n=160, per_partition=False, prefix="n2_g", seed=0):
    """A run the shape of mwc758_ales_repeat: ``npart`` groups, each with a selection flag, and
    ``nbase`` reduction parameters, global unless ``per_partition``."""
    rng = np.random.default_rng(seed)
    parts = [f"{prefix}{j:02d}" for j in range(npart)]
    params = []
    for b, lo, hi, kind in SPECS[:nbase]:
        for pid in (parts if per_partition else [None]):
            params.append(D.ParamInfo(b if pid is None else f"{b}_{pid}", b, lo, hi, kind, pid, "reduction"))
    for pid in parts:
        params.append(D.ParamInfo(f"sel_{pid}", "sel", 0, 1, "int", pid, "selection"))
    X = np.zeros((n, len(params)))
    for k, p in enumerate(params):
        v = rng.uniform(p.lo, p.hi, n)
        X[:, k] = np.round(v) if p.kind == "int" else v
    sel = []
    for _ in range(n):
        m = rng.random(npart) < 0.55
        m[0] |= not m.any()
        sel.append([pid for pid, u in zip(parts, m) if u])
    for t in range(n):
        for k, p in enumerate(params):
            if p.role == "selection":
                X[t, k] = float(p.partition in sel[t])
    y = rng.gamma(2.0, 0.5, n)
    return D.AnnulusData(
        run_name="mwc758_ales_repeat", annulus=0, nann=1, inrad=8.0, outrad=30.0, pxscale=0.035, fwhm=4.0,
        params=params, partitions=parts, X=X, y=y, phases=(["seed"] * 30 + ["tpe"] * n)[:n],
        k_used=[{pid: (14 if pid in sel[t] else None) for pid in parts} for t in range(n)], selected=sel,
        part_snr=[{p: 1.0 for p in parts} for _ in range(n)],
        sources=[[(0.60, 322.0, 2e-5), (0.60, 142.0, 2e-5)] for _ in range(n)],
        per_source=[[1.9, 1.5]] * n, raw_per_source=[[1.9, 1.5]] * n, clean_per_source=[[0.2, 0.1]] * n,
        raw=y.copy(), wall=np.ones(n), contrast=np.full(n, 2e-5), configs=[{} for _ in range(n)],
        n_init=30, n_iter=3000, gamma=0.25, metric_name="mawet", search_mode="tpe", partition_label="group")


def _render(ad, i=None):
    img = np.random.default_rng(1).normal(size=(57, 57))
    i = ad.n - 1 if i is None else i
    fig = D.render_step(ad, i, D.StepImages(cur_inj=img, cur_clean=img, best_inj=img, best_clean=img,
                                            best_index=int(np.argmax(ad.y))),
                        None, None, fig=Figure(figsize=(8, 5)), elapsed_s=54000.0, eta_s=28000.0,
                        eta_full_s=28000.0, loop_s=29.1)
    fig.canvas.draw()
    return fig


def _layout_problems(fig):
    """Pairs of axes whose drawn boxes overlap, and texts outside the figure or their column."""
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    tol = 0.5
    boxes = [(a, a.get_tightbbox(r)) for a in fig.axes if a.get_visible()]
    out = []
    for k, (a, A) in enumerate(boxes):
        if A.x0 < -tol or A.y0 < -tol or A.x1 > W + tol or A.y1 > H + tol:
            out.append(f"axes '{a.get_title()}' leaves the figure: {A}")
        for b, B in boxes[k + 1:]:
            if min(A.x1, B.x1) - max(A.x0, B.x0) > tol and min(A.y1, B.y1) - max(A.y0, B.y0) > tol:
                out.append(f"axes '{a.get_title()}' at {a.get_position().bounds} overlaps "
                           f"'{b.get_title()}' at {b.get_position().bounds}")
    lay = fig._klip_layout
    for t in fig.texts:
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > W + tol or bb.y1 > H + tol:
            out.append(f"text {t.get_text()[:30]!r} leaves the figure")
        gid = t.get_gid()
        if gid in lay:
            x0, y0, x1, y1 = lay[gid]
            if bb.x0 < x0 * W - tol or bb.x1 > x1 * W + tol or bb.y0 < y0 * H - tol or bb.y1 > y1 * H + tol:
                out.append(f"text {t.get_text()[:30]!r} leaves its column {gid}: {bb} vs {lay[gid]}")
    for nm, ax in fig._klip_axes.items():
        x0, y0, x1, y1 = lay[nm]
        A = ax.get_tightbbox(r)
        if A.x0 < x0 * W - tol or A.x1 > x1 * W + tol or A.y0 < y0 * H - tol or A.y1 > y1 * H + tol:
            out.append(f"panel '{ax.get_title()}' leaves its box {nm}: {A} vs {lay[nm]}")
    return out


@pytest.mark.parametrize("npart,nbase", [(20, 5), (2, 5), (20, 9), (2, 2), (6, 3)])
def test_no_two_boxes_overlap_and_every_text_stays_in_its_column(npart, nbase):
    fig = _render(_ad(npart, nbase))
    assert _layout_problems(fig) == []
    lay = fig._klip_layout
    assert {"rhs:eta", "rhs:legend", "rhs:panel0", "rhs:panel1"} <= set(lay)
    assert 0.6 <= lay["rhs:scale"] <= 1.0


def test_a_space_with_a_slot_per_partition_lays_out_too():
    fig = _render(_ad(20, 4, per_partition=True))
    assert _layout_problems(fig) == []


def test_selection_flags_alone_do_not_make_the_parameters_per_partition():
    ad = _ad(20, 5)
    assert not ad.replicated and ad.partitioned
    assert ad.corner(None)["X"].shape[0] == int(ad.valid.sum())      # one point per evaluation
    rep = _ad(6, 3, per_partition=True, prefix="n")
    assert rep.replicated and rep.partitioned


def test_the_settings_lines():
    names = [f"n2_g{j:02d}" for j in range(20)]
    sel = ["n2_g00", "n2_g02", "n2_g10"] + [f"n2_g{j}" for j in range(12, 20)]
    assert D._parts_text(names, sel) == ("g00 g02 g10 g12-g19", " (11/20)")
    assert D._parts_text(["n1", "n2", "n3"], ["n1", "n2"]) == ("n1 n2", " (2/3)")
    assert D._counts([14] * 11 + [np.nan] * 9) == "14 x11, - x9"
    assert D._counts([5, 7, 5]) == "5 x2, 7"
    ad = _ad(20, 5)
    i = ad.n - 1
    lines = D._vec_lines(ad, i)
    assert lines[:5] == [f"{lab} {D._cv(ad.X[i, k])}" for k, lab in enumerate(("bin", "filt", "asep", "corr", "k"))]
    nsel = len(ad.selected[i])
    pair = [f"14 x{nsel}", f"- x{20 - nsel}"]
    assert lines[5] == "k used " + ", ".join(pair if ad.partitions[0] in ad.selected[i] else pair[::-1])
    assert lines[6].startswith("parts ") and lines[6].endswith(f" ({nsel}/20)")
    # a vector of a slot per partition: the values in order up to 8 partitions
    rep = _ad(6, 3, per_partition=True, prefix="n")
    vl = D._vec_lines(rep, 0)
    assert vl[0].startswith("bin [") and len(vl[0][5:-1].split()) == 6


def test_a_cut_keeps_whole_words_and_the_count():
    s = "parts g00 g04 g06-g08 g10 g11 g15 g16 g18 g19 (11/20)"
    c = D._cut(s, 40, " (11/20)")
    assert len(c) <= 40 and c.endswith(" ... (11/20)") and c.startswith("parts g00 g04")
    assert " g1 " not in c + " " and D._cut("bin 5", 40) == "bin 5"
    assert len(D._cut("asep [1.4 2 0.71 1.6 1.9 0.84 0.93 2.4 2 0.29]", 24)) <= 24


def test_global_parameters_pool_the_corner():
    """One landscape, one corner: no cycling through the groups' subsets of it."""
    ad = _ad(20, 5)
    fig = _render(ad)
    titles = [t.get_text() for t in fig.texts if t.get_text().startswith("S/N landscape")]
    assert titles and all("of 20" not in t and "pooled" not in t for t in titles)
    live = types.SimpleNamespace(corner_mode="cycle", _corner_i=0)
    assert D.LiveDisplay._next_corner_partition(live, ad) is None
    rep = _ad(6, 3, per_partition=True, prefix="n")
    assert D.LiveDisplay._next_corner_partition(live, rep) == "n00"


def test_a_long_vector_takes_a_line_of_its_own_or_is_cut():
    """With a slot per partition for 20 partitions the vectors do not fit a column: each takes a
    line across both, and what does not fit that is cut, never crossing the edge."""
    fig = _render(_ad(20, 4, per_partition=True))
    wide = [t for t in fig.texts if t.get_gid() == "settings:wide"]
    assert wide, "the long vectors did not get a line each"
    assert _layout_problems(fig) == []
