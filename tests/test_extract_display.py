"""Tests of the extraction's live dashboard (klip_tpe.extract_display), after the IDL predecessor's mwcm_sub_dash."""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from klip_tpe import Param, RunConfig, SearchSpace
from klip_tpe.extract import ExtractConfig, extract_companion, fit_companion, recovery_check, write_extract
from klip_tpe.extract_display import CROPS, PANEL_PX, ExtractDisplay, crop, render_panel, render_saved
from klip_tpe.runner import AnnulusResult, Runner

from test_extract import PX, TRUE, _finished_runner, _reduce_fn, red, two_partitions  # noqa: F401  (fixtures)


def _size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size, im.convert("RGB").getpixel((2, 2))


def _pngs(d):
    return sorted(glob.glob(os.path.join(d, "step*.png")))


def test_crop_keeps_the_star_at_the_center():
    img = np.zeros((64, 64))
    img[31:33, 31:33] = 1.0                       # the star at (31.5, 31.5), between four pixels
    c = crop(img, 10)
    assert c.shape == (20, 20) and c.dtype == np.float32, "an even side for an even image"
    assert np.array_equal(np.argwhere(c == 1.0), [[9, 9], [9, 10], [10, 9], [10, 10]]), "the star at (9.5, 9.5)"
    assert np.array_equal(crop(c, 10), c), "a cut-out of that size comes back as it is"
    odd = np.zeros((65, 65))
    odd[32, 32] = 1.0
    c = crop(odd, 10)
    assert c.shape == (21, 21) and c[10, 10] == 1.0 and np.nansum(c) == 1.0
    big = crop(img, 40)
    assert big.shape == (80, 80) and np.isnan(big[0, 0]) and big[39, 39] == 1.0


def test_a_panel_is_1600_by_760_on_black(tmp_path):
    rng = np.random.default_rng(0)
    n = 41
    yy, xx = np.mgrid[0:n, 0:n]
    clean = rng.normal(0, 0.05, (n, n)) + np.exp(-0.5 * ((xx - 28) ** 2 + (yy - 30) ** 2) / 2.0)
    X = np.c_[rng.uniform(0.85, 0.95, 50), rng.uniform(55, 65, 50), rng.uniform(2e-3, 8e-3, 50)]
    R = 0.3 + ((X - [0.9, 60, 4e-3]) / [0.1, 10, 6e-3]) ** 2 @ np.ones(3)
    run = {"phase": "main", "lo": [0.85, 55.0, 2e-3], "hi": [0.95, 65.0, 8e-3], "X": X, "R": R,
           "flags": ["warmup"] * 20 + ["tpe"] * 30, "clean": clean, "best": clean * 0.2, "trial": clean * 0.5,
           "best_x": [0.9, 60.0, 4e-3], "best_r": 0.3, "trial_x": [0.92, 61.0, 5e-3], "trial_r": 0.5,
           "profiles": None, "stage": "search", "it": 50, "n_stage": 100, "title": "trial 50 / 100",
           "elapsed": 61.0, "per_trial": 1.22, "eta": 61.0, "true": None}
    st = {"geom": {"half": 20, "pxscale": PX, "fwhm_px": 4.0, "conv": "pa", "label": "a test"}, "run": run,
          "psf": None, "reference": (0.91, 59.0), "spectrum": [], "chromatic": {"rows": [], "combined": None},
          "fakes": None, "elapsed_all": 61.0}
    p = render_panel(st, str(tmp_path / "steps" / "step0001.png"))
    size, corner = _size(p)
    assert size == PANEL_PX and corner == (0, 0, 0)
    assert not glob.glob(str(tmp_path / "steps" / "*.part")), "the panel is moved into place"
    # the later stages' grid, with a spectrum, a chromatic panel and a table of fakes, draws too
    st.update(spectrum=[{"partition": "a", "lam_um": 3.1, "contrast": 4e-3, "sigma": 3e-4},
                        {"partition": "b", "lam_um": 3.6, "contrast": 2e-3, "sigma": None}],
              chromatic={"rows": [{"partition": "a", "lam_um": 3.1, "dx": -10.1, "dy": 8.0, "sdx": 0.1, "sdy": 0.2},
                                  {"partition": "b", "lam_um": 3.6, "dx": -9.9, "dy": 8.1, "sdx": None, "sdy": 0.2}],
                         "combined": (-10.0, 8.0)},
              fakes={"ctx": "a", "n": 3, "rows": [{"d_rho": 1e-3, "d_pa": 0.1, "d_c": 0.02}] * 2})
    run.update(profiles={"values": [list(np.linspace(0.88, 0.92, 9))], "residual": [list(0.3 + np.linspace(-1, 1, 9) ** 2)]},
               stage="profile", true=[0.9, 180.0, 4e-3])
    size, _ = _size(render_panel(st, str(tmp_path / "steps" / "step0002.png")))
    assert size == PANEL_PX


def test_the_fakes_table_fits_its_slot_whatever_it_holds():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from klip_tpe.extract_display import _fakes_table
    fig = Figure(figsize=(PANEL_PX[0] / 100.0, PANEL_PX[1] / 100.0), dpi=100)
    FigureCanvasAgg(fig)
    x0, y0 = 0.675 + 0.155, 0.46                       # the later stages' top-right slot, at the panel's edge
    ax = fig.add_axes([x0 + 0.008, y0 + 0.06, 0.155 - 0.016, 0.37 - 0.12])
    rows = [{"d_rho": 0.0123, "d_pa": -12.345, "d_c": -0.987}] * 12
    _fakes_table(ax, {"fakes": {"ctx": "a partition with a long name", "n": 14, "rows": rows}})
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in ax.get_children() if isinstance(t, Text) and t.get_text()]
    assert len(texts) == 2, "the title and the table"
    for t in texts:
        assert t.get_window_extent(r).x1 <= fig.bbox.x1 - 5, t.get_text()[:40]


def test_the_last_panel_shows_the_values_to_quote():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from klip_tpe.extract_display import _text
    run = {"phase": "main", "best_x": [0.9, 60.0, 4e-3], "best_r": 0.3, "trial_x": None, "title": "results",
           "per_trial": None, "elapsed": None, "eta": None}
    st = {"geom": {"label": "a test"}, "elapsed_all": None,
          "quoted": {"value": {"rho": 0.90012, "pa": 59.85, "contrast": 4.1e-3},
                     "sigma": {"rho": 1.1e-3, "pa": 0.025, "contrast": 6e-5},
                     "corrected": {"rho": False, "pa": True, "contrast": False}}}
    live = [(0.9, 1e-4), (60.0, 0.01), (4e-3, 1e-5)]
    for quoted, want, gone in ((True, "PA 59.850* +- 0.025", "from the profiles"),
                               (False, "PA 60.000 +- 0.010", "to quote")):
        fig = Figure(figsize=(16, 7.6), dpi=100)
        FigureCanvasAgg(fig)
        _text(fig, dict(st, quoted=st["quoted"] if quoted else None), run, live, "profiles")
        text = fig.texts[-1].get_text()
        assert want in text and gone not in text, text
    assert "to quote, * less the fakes' bias:" in _last_text(_text, st, run, live)


def _last_text(fn, st, run, live):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(16, 7.6), dpi=100)
    FigureCanvasAgg(fig)
    fn(fig, st, run, live, "profiles")
    return fig.texts[-1].get_text()


def test_the_fit_tells_its_progress(red):  # noqa: F811
    cfg = ExtractConfig(n_iter=24, n_init=10, nref=3, n_fakes=0, seed=4)
    ev = []
    f = fit_companion(_reduce_fn(red), TRUE.rho + 0.01, TRUE.theta + 1.0, 3e-3, fwhm_px=red.fwhm, pxscale=PX,
                      angle_convention=red.angle_convention, config=cfg, progress=ev.append, phase="test")
    kinds = [e["kind"] for e in ev]
    npx = 4 * cfg.nref + 1
    assert kinds[0] == "start" and kinds[-1] == "done" and kinds.count("trial") == cfg.n_iter + 3 * npx
    assert all(e["phase"] == "test" for e in ev)
    s = ev[0]
    assert s["n"] == cfg.n_iter and s["lo"] == f["lo"] and s["clean"] is f["images"]["clean"]
    assert s["pxscale"] == PX and s["fwhm_px"] == red.fwhm and s["start"] == f["start"]
    trials = [e for e in ev if e["kind"] == "trial"]
    search = [e for e in trials if e["stage"] == "search"]
    assert [e["it"] for e in search] == list(range(1, cfg.n_iter + 1))
    assert [e["r"] for e in search] == pytest.approx(f["trials"]["residual"])
    assert [e["flag"] for e in search] == f["trials"]["phase"]
    best = np.minimum.accumulate([e["r"] for e in trials])
    assert [e["best_r"] for e in trials] == pytest.approx(list(best)), "the best so far, over the profiles too"
    assert all(e["new_best"] == (e["r"] == e["best_r"] and (i == 0 or e["r"] < trials[i - 1]["best_r"]))
               for i, e in enumerate(trials))
    prof = [e for e in trials if e["stage"] == "profile"]
    assert prof[-1]["it"] == prof[-1]["n"] == 3 * npx
    assert prof[-1]["profiles"] == f["profiles"], "the last point's profiles are the result's"
    assert ev[-1]["fit"] is f and f["best"]["residual"] == pytest.approx(trials[-1]["best_r"])


def test_the_display_draws_each_stage_and_saves_the_trials(red, tmp_path):  # noqa: F811
    out = str(tmp_path / "x")
    os.makedirs(os.path.join(out, "steps"))
    open(os.path.join(out, "steps", "step0099.png"), "wb").close()        # an earlier extraction's
    lines = []
    disp = ExtractDisplay(out, label="test", every=10, reference=(TRUE.rho, TRUE.theta), engine="inline",
                          psf_fn=lambda rho, pa, shape: (_ for _ in ()).throw(RuntimeError("no model")),
                          log=lines.append)
    cfg = ExtractConfig(n_iter=20, n_init=8, nref=0, n_fakes=2, fake_iter=20, seed=6)
    f = fit_companion(_reduce_fn(red), TRUE.rho, TRUE.theta, TRUE.contrast, fwhm_px=red.fwhm, pxscale=PX,
                      angle_convention=red.angle_convention, config=cfg, progress=disp)
    recovery_check(_reduce_fn(red), f, fwhm_px=red.fwhm, pxscale=PX, angle_convention=red.angle_convention,
                   config=cfg, progress=disp)
    files = disp.close(movie=False)
    assert not disp.off and not disp._errors
    assert any("no PSF model" in s for s in lines) and any("earlier extraction removed" in s for s in lines)
    # the companion and each fake: trials 10 and 20 and the end
    pngs = _pngs(os.path.join(out, "steps"))
    assert len(pngs) == 3 + 2 * 3 == len(disp.paths) and pngs == sorted(disp.paths)
    assert _size(pngs[0])[0] == PANEL_PX
    assert disp.fakes["ctx"] == "" and len(disp.fakes["rows"]) == 2
    with np.load(files["crops"]) as z:
        assert z["trials"].shape == (cfg.n_iter,) + disp.main["clean"].shape
        assert np.allclose(z["x"], f["trials"]["x"]) and int(z["half"]) == disp.geom["half"]
    assert disp.main["clean"].shape == (2 * disp.geom["half"],) * 2, "an even cut-out of the even image"
    from klip_tpe.viewer import newest_panel
    assert newest_panel(out)[0] == pngs[-1], "klip-tpe view --run-dir finds the panels"


def test_a_failure_stops_the_display_alone(tmp_path):
    lines = []
    disp = ExtractDisplay(str(tmp_path), engine="inline", log=lines.append)
    disp({"kind": "start", "phase": "main"})                               # no box, no image
    assert disp.off and any("stopped" in s for s in lines)
    disp({"kind": "trial"})                                                # ignored from then on
    assert disp.close(movie=False) == {}
    with pytest.raises(ValueError):
        ExtractDisplay(str(tmp_path), engine="window")


def test_partitions_fill_the_spectrum_and_the_chromatic_panel(two_partitions, tmp_path):  # noqa: F811
    pred = two_partitions
    space = SearchSpace().replicate([Param("k_klip", 1, 10, "int", default=5)], ["a", "b"])
    from klip_tpe import MawetPeakSNR, Objective, PositionSampler
    obj = Objective(MawetPeakSNR(pxscale=PX, fwhm=pred.fwhm), clean_subtract=True)
    samp = PositionSampler(fwhm_as=pred.fwhm * PX)
    cfg = RunConfig(ann_edges=[8, 26], n_iter=2, n_init=2, seed=1, save_fits=False, save_eval_images=False,
                    write_setup_files=False, fm_curve=False)
    r = Runner(pred, space, obj, samp, cfg, str(tmp_path / "run"), log=lambda s: None)
    win = space.decode(space.default_vector())
    r.results = [AnnulusResult(annulus=0, inrad=8.0, outrad=26.0, contrast=1e-3, search_best_index=0,
                               search_best_score=5.0, validated=True, winner_index=0, winner_score=5.0,
                               winner_x=list(win.x), winner_config=win.to_dict(), validation_table=[],
                               validation_samples={})]
    ec = ExtractConfig(n_iter=24, n_init=10, nref=3, n_fakes=2, fake_iter=20, seed=2, group_positions=True,
                       group_iter=20, group_init=6, group_fake_iter=20)
    out = str(tmp_path / "out")
    disp = ExtractDisplay(out, label="two partitions", every=1000, engine="inline")
    seen = []

    def both(e):
        seen.append((e["kind"], e.get("phase"), e.get("partition")))
        disp(e)

    found = extract_companion(r, TRUE.rho, TRUE.theta + 1.0, 3e-3, config=ec, log=lambda s: None, progress=both)
    phases = {p for k, p, _ in seen if k == "start"}
    assert phases == {"main", "fake 1 of 2", "fake 2 of 2", "partition a (1 of 2)", "partition b (2 of 2)",
                      "partition a, fake 1 of 2", "partition a, fake 2 of 2", "partition b, fake 1 of 2",
                      "partition b, fake 2 of 2"}
    assert [k for k, _, _ in seen if k in ("group", "group_fakes")] == ["group", "group", "group_fakes", "group_fakes"]
    assert all(pid in ("a", "b") for k, p, pid in seen if p and p.startswith("partition "))
    # the spectrum, widened to the fakes' scatter, and the partitions' own positions against the combined fit
    g = {x["partition"]: x for x in found["groups"]}
    assert [s["partition"] for s in disp.spectrum] == ["a", "b"]
    for s in disp.spectrum:
        assert s["contrast"] == pytest.approx(g[s["partition"]]["contrast"])
        prof = g[s["partition"]]["sigma"]
        assert not np.isfinite(prof) or s["sigma"] >= prof - 1e-15
    assert [row["partition"] for row in disp.chromatic["rows"]] == ["a", "b"]
    assert disp.chromatic["combined"] is not None
    assert disp.fakes["ctx"] == "b" and len(disp.fakes["rows"]) == 2, "the table follows the fakes being fitted"
    assert not disp.off and not disp._errors
    disp.close(movie=False)
    n_live = len(_pngs(os.path.join(out, "steps")))
    assert n_live == len(disp.paths) > 0
    # drawn again afterwards from the saved tables: beside the live panels, from the saved cut-outs
    write_extract(found, out, label="two partitions")
    lines = []
    paths = render_saved(out, every=8, movie=False, log=lines.append)
    assert any(f"from {CROPS}" in s for s in lines) and any("steps_saved" in s for s in lines)
    assert len(_pngs(os.path.join(out, "steps"))) == n_live, "the live panels stay"
    saved = _pngs(os.path.join(out, "steps_saved"))
    # the search's trials 8, 16 and 24, the profiles' 8 to 39 by 8 and 39, its end, the two partitions and the results
    assert saved == paths and len(saved) == 3 + 5 + 1 + 2 + 1


def test_render_saved_stands_in_for_trials_it_did_not_save(red, tmp_path):  # noqa: F811
    r = _finished_runner(red, tmp_path, subtract=False)
    cfg = ExtractConfig(n_iter=30, n_init=10, nref=3, n_fakes=0, seed=5)
    found = extract_companion(r, TRUE.rho + 0.01, TRUE.theta - 1.0, 3e-3, config=cfg, log=lambda s: None)
    out = str(tmp_path / "out")
    write_extract(found, out, label="no display")
    lines = []
    paths = render_saved(out, every=15, reference=(TRUE.rho, TRUE.theta), movie=True, log=lines.append)
    assert any("were not saved" in s for s in lines)
    pngs = _pngs(os.path.join(out, "steps"))
    # trials 15 and 30, profile points 15, 30 and 39, the end, and the results with the values to quote
    assert len(pngs) == 2 + 3 + 1 + 1 and paths[:-1] == pngs
    assert paths[-1].endswith("extract_steps.gif") and os.path.exists(paths[-1])
    assert not os.path.exists(os.path.join(out, CROPS)), "the stand-ins are never saved as the trials' images"


def test_the_panel_process_draws_the_panels(red, tmp_path):  # noqa: F811
    out = str(tmp_path / "p")
    lines = []
    disp = ExtractDisplay(out, every=1, engine="process", log=lines.append)
    cfg = ExtractConfig(n_iter=20, n_init=6, nref=0, n_fakes=0, seed=3)
    fit_companion(_reduce_fn(red), TRUE.rho, TRUE.theta, TRUE.contrast, fwhm_px=red.fwhm, pxscale=PX,
                  angle_convention=red.angle_convention, config=cfg, progress=disp)
    assert disp.engine == "process", lines
    disp.close(movie=False)
    pngs = _pngs(os.path.join(out, "steps"))
    assert pngs and pngs == sorted(disp.paths) and not disp._errors, lines
    assert disp.skipped + len(pngs) == cfg.n_iter + 1, "each trial asks for a panel, and the end"


def test_the_cli_parses_the_display_and_the_render():
    from klip_tpe.cli import _build_parser, cmd_extract
    p = _build_parser()
    a = p.parse_args(["extract", "--run-dir", "r", "--rho", "0.452", "--pa", "211.2", "--contrast", "7e-4"])
    assert a.display and a.display_every is None and a.reference is None and a.render is None
    a = p.parse_args(["extract", "--rho", "0.452", "--pa", "211.2", "--contrast", "7e-4", "--no-display",
                      "--display-every", "5", "--reference", "0.452", "211.2"])
    assert not a.display and a.display_every == 5 and a.reference == [0.452, 211.2]
    a = p.parse_args(["extract", "--render", "some_dir"])
    assert a.render == "some_dir" and a.rho is None
    with pytest.raises(SystemExit, match="no finished extraction"):
        cmd_extract(a)
    with pytest.raises(SystemExit, match="required"):
        cmd_extract(p.parse_args(["extract", "--run-dir", "r", "--rho", "0.452"]))
