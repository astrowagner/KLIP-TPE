"""The final-image grid, the election on the stitch and the run's curve (NEAR2's note of 2026-11-16).

What must hold:

* Each tile has an odd number of separations, at least 3, at the middles of equal bands, so the
  first and the last lie near its edges and one at its middle.  The curve's grid is the election's
  turned by half a position angle step, the same for every run of a setup, and no point lies in a
  known companion's keep-out.
* Every annulus whose part of the stitch lies within ``READ_FWHM`` FWHM of a point is reduced with its
  source, and an annulus' election reads the points within ``cover_fwhm`` of its part of the stitch.
  A reading is the stitched verify's Path A on the stitch built as the final stitch builds it.
* A finalist replaces the winner only when its mean paired gain is positive and exceeds ``z`` times its
  standard error.  The sweeps stop once one changes nothing.
* Inside a run the stage elects the winners the final products take: the validation's final round
  is left to it, ``contrast_curve.txt`` takes its curve and seam lines, and an annulus whose winner
  changed gets that winner's images, parameters and record, which a later resume, ``rereduce``, the
  re-score and the finalists of other runs take.
* A stage started again on its directory reads its readings back instead of reducing again.
* On finished runs (``klip-tpe elect``) the finalists of other runs of the same setup enter, the runs
  are only read, and the elected winners' products go to the stage's directory.  The election, the
  curve and the products read the stitch asked for, with or without pads between overlapping annuli,
  the reports name it, and ``winners.txt`` gives every annulus' winner as ``rereduce --base-winners``
  reads it (NEAR2, 2026-11-16).
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

import klip_tpe.final_grid as FG
from klip_tpe.final_grid import (READ_FWHM, FinalGrid, FinalGridConfig, GridPoint, collect_finalists, elect_one,
                                 elect_products, exclude_known, grid_points, run_final_grid, seam_ratios, seam_steps,
                                 stitch_regions, tile_separations, touched)
from klip_tpe.contrast_grid import ContrastGridConfig
from klip_tpe.refine import StitchScorer, config_key, run_entrants
from klip_tpe.runner import CalibrationConfig, RunConfig, Runner, ValidationConfig

from conftest import build_synthetic_run

QUIET = (lambda s: None)
EDGES = [8, 20, 32, 44]


def _make_run(path, seed, stage, n_final=0):
    red, space, obj, samp, cfg = build_synthetic_run(
        ann_edges=list(EDGES), n_iter=[5], n_init=3, seed=seed, param_verify=False, stitch_every=0, verify=False,
        write_setup_files=False, fm_curve=False, validation=ValidationConfig(n_top=3, n_valid=1, n_final=n_final),
        calibration=CalibrationConfig(forced=[2e-3, 2e-3, 2e-3]),
        final_grid=FinalGridConfig(mode="on" if stage else "off", n_pa=4))
    lines = []
    Runner(red, space, obj, samp, cfg, str(path), log=lines.append).run()
    return str(path), lines


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("final_grid")
    a, log_a = _make_run(base / "run_a", 11, True, n_final=2)
    b, _ = _make_run(base / "run_b", 23, False)
    return a, b, log_a


def _resume(run_dir, **kw):
    red, space, obj, samp, _ = build_synthetic_run()
    return Runner.resume(run_dir, red, obj, samp, log=QUIET, **kw)


def _snapshot(d):
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            st = os.stat(p)
            out[os.path.relpath(p, d)] = (st.st_size, st.st_mtime_ns)
    return out


# ---------------------------------------------------------------------------- the grid
def test_each_tile_has_an_odd_number_of_separations_at_the_middles_of_equal_bands():
    s = tile_separations(0.0, 10.0, 4.0)
    assert np.allclose(s, [10 / 6, 5.0, 50 / 6])
    s = tile_separations(10.0, 30.0, 4.0)                 # 5 bands of 4 px
    assert s.size == 5 and np.allclose(s, 12.0 + 4.0 * np.arange(5))
    s = tile_separations(10.0, 26.0, 4.0)                 # 4 bands would do, made 5
    assert s.size == 5 and np.allclose(np.diff(s), 16.0 / 5)
    assert tile_separations(3.0, 4.7, 6.4).size == 3      # a tile narrower than a FWHM keeps 3
    assert tile_separations(5.0, 5.0, 4.0).size == 0


def test_the_curves_grid_is_the_elections_turned_by_half_a_step():
    tiles = [(0, 8.0, 20.0), (1, 20.0, 32.0)]
    el = grid_points(tiles, 4.0, 6)
    cv = grid_points(tiles, 4.0, 6, turn=0.5)
    assert len(el) == len(cv) == 6 * 6
    assert [p.r_px for p in el] == [p.r_px for p in cv]
    for p, q in zip(el, cv):
        d = (q.theta - p.theta) % 360.0
        assert d == pytest.approx(30.0)                   # half of 60 degrees
    # each separation turned by the golden fraction of the step over the last one, counted over the tiles
    first = [el[i].theta for i in range(0, len(el), 6)]
    gold = 0.5 * (np.sqrt(5.0) - 1.0)
    assert np.allclose(first, [((g * gold) % 1.0) * 60.0 for g in range(6)])
    assert [p.role for p in el[:18:6]] == ["inner", "middle", "outer"]
    assert el[0].annulus == 0 and el[-1].annulus == 1 and el[-1].sep_index == 5
    # the floor leaves out a tile's separations under it, and the bands keep their places
    fl = grid_points(tiles, 4.0, 6, floor_px=12.0)
    assert min(p.r_px for p in fl) > 12.0 and fl[0].band == 1 and fl[0].role == "middle"
    assert grid_points(tiles, 4.0, 6) == el, "the same grid for every run of a setup"


def test_the_points_in_a_known_companions_keep_out_are_left_out():
    pts = grid_points([(0, 20.0, 32.0)], 4.0, 12)
    px = 0.01
    rk, tk = 26.0, pts[13].theta                         # on a point of the middle separation
    n = exclude_known(pts, [(rk * px, tk)], px, 4.0, ContrastGridConfig())
    out = [p for p in pts if p.status == "excluded"]
    assert n == len(out) >= 1 and pts[13] in out
    assert all(abs((p.theta - tk + 180) % 360 - 180) <= 15.0 for p in out)
    assert all(p.why == "in a known companion's keep-out" for p in out)


def test_the_points_the_sampler_would_not_use_are_left_out():
    from klip_tpe.final_grid import exclude_sampled
    from klip_tpe.positions import PositionSampler
    pts = grid_points([(0, 20.0, 32.0)], 4.0, 12)
    n = exclude_sampled(pts, PositionSampler(fwhm_as=0.04, forbidden_pa=[(90.0, 20.0)]), 0.01)
    out = [p for p in pts if p.status == "excluded"]
    assert n == len(out) >= 3
    assert all(abs((p.theta - 90.0 + 180) % 360 - 180) < 20.0 for p in out)
    assert all(abs((p.theta - 90.0 + 180) % 360 - 180) >= 20.0 for p in pts if p.status == "active")
    assert exclude_sampled(pts, None, 0.01) == 0


def test_the_annuli_a_point_reaches_and_an_annulus_cover():
    regions = stitch_regions([(8.0, 20.0), (20.0, 32.0), (32.0, 44.0)], [(0.0, 2.0), (2.0, 2.0), (2.0, 0.0)])
    assert regions == [(8.0, 22.0), (18.0, 34.0), (30.0, 44.0)]
    assert touched(10.0, regions, 4.0) == [0]
    assert touched(15.0, regions, 4.0) == [0, 1]           # 3 px from annulus 2's part of the stitch
    assert touched(26.0, regions, 4.0) == [0, 1, 2]
    assert touched(26.0, regions, 3.0) == [1]


def test_a_finalist_replaces_the_winner_only_past_its_paired_error():
    base = np.array([5.0, 6.0, 7.0, 6.0, 5.0])
    better = base + np.array([0.5, 0.6, 0.4, 0.5, 0.6])
    noisy = base + np.array([2.0, -1.5, 1.0, -1.0, 0.2])
    R = {"w": base, "b": better, "n": noisy, "x": base - 1.0}
    win, st = elect_one(R, ["w", "x", "n", "b"], "w")
    assert win == "b" and st["b"]["diff_mean"] == pytest.approx(0.52) and st["w"]["diff_mean"] == 0.0
    assert elect_one(R, ["w", "x", "n"], "w")[0] == "w", "a gain inside its error leaves the winner"
    assert elect_one(R, ["w", "b"], "w", z=100.0)[0] == "w"
    # of several past their error, the largest gain, the first on a tie
    R2 = {"w": base, "b": better, "c": better + 0.1, "d": better + 0.1}
    assert elect_one(R2, ["w", "b", "c", "d"], "w")[0] == "c"
    # NaN readings are left out of the pairs, and fewer than 2 pairs cannot replace the winner
    R3 = {"w": base, "b": np.array([9.0, np.nan, np.nan, np.nan, np.nan])}
    assert elect_one(R3, ["w", "b"], "w")[0] == "w"


def test_the_seam_line_is_the_curve_near_each_edge_over_its_middle():
    cv = {"curve": [2.0, 1.0, 1.5, None, 4.0, 2.0]}
    seps = [(0, 0, 3), (0, 1, 3), (0, 2, 3), (1, 0, 3), (1, 1, 3), (1, 2, 3)]
    s = seam_ratios(cv, seps, 2, [0, 1])
    assert s[0] == {"annulus": 1, "inner": 2.0, "outer": 1.5, "inner_edge": "edge", "outer_edge": "seam"}
    assert s[1]["inner"] is None and s[1]["outer"] == pytest.approx(0.5) and s[1]["outer_edge"] == "edge"


def test_the_seam_steps_take_the_curves_slope_out():
    """A smooth power law reads 1 everywhere, and a jump between two tiles shows as a step there
    (NEAR2's note of 2026-11-17, the seam line against a smooth fit)."""
    seps = [(a, b, 3) for a in range(4) for b in range(3)]
    r = np.linspace(0.2, 1.4, 12)
    smooth = 1e-3 * (r / 0.2) ** -2.5
    s = seam_steps({"r_as": list(r), "curve": list(smooth)}, seps, [0, 1, 2, 3])
    assert s["degree"] == 3 and s["rms"] < 1e-9
    assert all(abs(a[k] - 1.0) < 1e-6 for a in s["annuli"] for k in ("inner", "middle", "outer"))
    assert all(abs(a["step"] - 1.0) < 1e-6 for a in s["annuli"][:-1]) and s["annuli"][-1]["step"] is None
    jump = smooth * np.where(np.arange(12) >= 6, 1.3, 1.0)        # tiles 3 and 4 are 30 % worse
    st = seam_steps({"r_as": list(r), "curve": list(jump)}, seps, [0, 1, 2, 3])["annuli"]
    steps = [a["step"] for a in st[:-1]]
    assert max(range(3), key=lambda k: steps[k]) == 1 and steps[1] > 1.15, steps
    # too few points for a fit: no steps
    assert seam_steps({"r_as": [0.3, 0.4, 0.5], "curve": [1e-3, 8e-4, 6e-4]}, seps[:3], [0])["degree"] is None


def test_the_settings_are_checked_and_old_runs_resume_without_the_stage(monkeypatch):
    with pytest.raises(ValueError, match="curve_snr"):
        FinalGridConfig(curve_snr="peak").check()
    with pytest.raises(ValueError, match="n_pa"):
        FinalGridConfig(n_pa=0).check()
    with pytest.raises(ValueError, match="target"):
        FinalGridConfig(target=(6.0, 5.0)).check()
    monkeypatch.setitem(FG._DEFAULT, "mode", "on")
    cfg = RunConfig()
    assert cfg.final_grid.enabled and cfg.final_grid.n_pa == 12
    d = cfg.to_dict()
    assert RunConfig.from_dict(d).final_grid == cfg.final_grid
    d.pop("final_grid")
    assert RunConfig.from_dict(d).final_grid.mode == "off", "a run saved before the stage elects nothing again"


def test_the_command_line_takes_the_options():
    from klip_tpe.cli import _build_parser, _final_grid_config
    ap = _build_parser()
    a = ap.parse_args(["near", "--final-grid", "off", "--fg-pa", "8", "--fg-cover", "0.75", "--fg-z", "2",
                       "--fg-sweeps", "3", "--fg-finalists", "5", "--fg-entrants-from", "x", "y",
                       "--fg-curve", "path_a", "--fg-step", "0.8"])
    fc = _final_grid_config(a)
    assert (fc.mode, fc.n_pa, fc.cover_fwhm, fc.z, fc.sweeps, fc.finalists, fc.curve_snr, fc.sep_step_fwhm) == \
        ("off", 8, 0.75, 2.0, 3, 5, "path_a", 0.8)
    assert fc.entrants_from == [os.path.abspath("x"), os.path.abspath("y")]
    assert _final_grid_config(ap.parse_args(["near"])).mode == FG._DEFAULT["mode"]
    with pytest.raises(SystemExit):
        _final_grid_config(ap.parse_args(["near", "--fg-pa", "0"]))
    e = ap.parse_args(["elect", "--run-dir", "r", "--entrants-from", "b", "c", "--pa", "16", "--z", "2",
                       "--no-products", "--plan", "--curve", "off"])
    assert e.func.__name__ == "cmd_elect" and e.entrants_from == ["b", "c"] and e.pa == 16 and e.plan
    assert e.no_products and e.curve == "off" and e.z == 2.0
    assert e.overlap_pads is None and e.stitch_injections is None
    e = ap.parse_args(["elect", "--no-overlap-pads", "--stitch-injections", "all"])
    assert e.overlap_pads is False and e.stitch_injections == "all"
    assert ap.parse_args(["elect", "--overlap-pads"]).overlap_pads is True


def test_the_contrast_curve_takes_the_final_curve(tmp_path):
    from klip_tpe.stitch import write_contrast_curve
    fin = {"r_as": [0.1, 0.2, 0.3], "curve": [3e-4, None, 1e-4], "annulus": [1, 1, 1], "target_snr": 5.0,
           "read_with": "grid", "seams": [{"annulus": 1, "inner": 1.2, "outer": None, "inner_edge": "edge",
                                           "outer_edge": "edge"}]}
    grid = {"r_as": [0.15], "curve": [9e-4]}
    p = str(tmp_path / "cc.txt")
    out = write_contrast_curve(p, [None], [8.0, 20.0], 4.0, 0.01, grid_curves=[grid], final_curve=fin)
    text = open(p).read()
    assert list(out["r_as"]) == [0.1, 0.3] and list(out["curve"]) == [3e-4, 1e-4]
    assert text.startswith("# Final-image injection-recovery SNR=5 contrast curve") and "9.0000E-04" not in text
    assert "# seam, annulus 1: inner 1.20 (edge of the stitch), outer - (edge of the stitch)" in text
    assert "not trimmed at the seams" in text


# ---------------------------------------------------------------------------- inside a run
@pytest.mark.slow
def test_a_run_elects_on_the_final_grid_and_its_products_take_the_curve(runs):
    a, _, log = runs
    text = "\n".join(log)
    assert text.count("validation final round: left to the final-image grid") == 3
    assert "final grid of run_a: 3 annuli, 3 finalists each" in text
    fgd = os.path.join(a, "final_grid")
    for f in ("election.json", "election.txt", "curve.json", "curve.txt", "state/readings.jsonl", "state/options.json"):
        assert os.path.exists(os.path.join(fgd, f)), f
    rec = json.load(open(os.path.join(fgd, "election.json")))
    cv = json.load(open(os.path.join(fgd, "curve.json")))
    assert set(rec["annuli"]) == {"1", "2", "3"} and rec["sweeps"]
    # the rule held in every row: a change is a positive gain past z standard errors, and no stay passed it
    for rows in rec["sweeps"]:
        for row in rows:
            ch = [x for x in row["finalists"] if not x["current"] and x["n_paired"] and x["n_paired"] >= 2
                  and x["gain"] is not None and x["se"] is not None and x["gain"] > 0 and x["gain"] > x["se"]]
            if row["changed"]:
                best = max(ch, key=lambda x: x["gain"])
                assert (best["run"], best["eval_index"]) == (row["after"]["run"], row["after"]["eval_index"])
            else:
                assert not ch
    # the curve: 3 separations an annulus, each with its seam line, in contrast_curve.txt as its main section
    assert cv["annulus"] == [1, 1, 1, 2, 2, 2, 3, 3, 3] and cv["role"][:3] == ["inner", "middle", "outer"]
    assert len(cv["seams"]) == 3 and cv["read_with"] == "grid"
    cc = open(os.path.join(a, "contrast_curve.txt")).read()
    assert cc.startswith("# Final-image injection-recovery SNR=5 contrast curve") and "# seam, annulus 2:" in cc
    rows = [ln.split() for ln in cc.splitlines() if ln and not ln.startswith("#")]
    got = {round(float(r[0]), 3): float(r[1]) for r in rows[:sum(c is not None for c in cv["curve"])]}
    want = {round(r, 3): c for r, c in zip(cv["r_as"], cv["curve"]) if c is not None}
    assert got == pytest.approx(want, rel=1e-4)
    fr = json.load(open(os.path.join(a, "final_results.json")))
    assert all(x["final_grid"] is not None for x in fr["annuli"])
    assert all(x["final_grid"]["changed"] == rec["annuli"][str(x["annulus"] + 1)]["changed"] for x in fr["annuli"])
    # each annulus' winner.json carries the stage's record too, rather than saying it never ran
    for x in fr["annuli"]:
        wj = json.load(open(os.path.join(a, f"annulus{x['annulus'] + 1:02d}", "winner.json")))
        assert wj["final_grid"] == x["final_grid"]


@pytest.mark.slow
def test_a_reading_is_the_stitched_verify_on_the_stitch_of_its_reductions(runs):
    a, _, _ = runs
    r = _resume(a, same_frames=False)
    sc = StitchScorer(r, log=QUIET)
    fin, start = collect_finalists(r)
    fg = FinalGrid(r, sc, FinalGridConfig(mode="on", n_pa=4), fin, start, None, log=QUIET)
    fg.background("start", fg.start)
    # every annulus whose part of the stitch lies within READ_FWHM of a point is reduced with its source
    regions = stitch_regions(sc.itiles, sc.pads)
    for p in fg.points["elect"]:
        want = [sc.order[i] for i, (lo, hi) in enumerate(regions)
                if max(lo - p.r_px, p.r_px - hi, 0.0) <= READ_FWHM * fg.fwhm + 1e-9]
        assert fg.reach["elect"][p.index] == want
    for k, idx in fg.cover["elect"].items():
        assert all(k in fg.reach["elect"][i] for i in idx), "an annulus' cover is reduced with it"
        assert set(p.index for p in fg.points["elect"] if p.annulus == k) <= set(idx)
    # a reading by hand: the reductions of the reached annuli with the source, the others' clean images,
    # stitched with the weights of the final stitch, measured as the stitched verify measures it
    sc.clean, sc.weights = dict(fg.bgs["start"]["images"]), fg.bgs["start"]["weights"]
    for p in fg.points["elect"][::7]:
        src = p.source(fg.px, 4e-4)
        imgs = {k: fg._reduce(k, fg.start[k], src, "t") for k in fg.reach["elect"][p.index]}
        st = sc.stitch(imgs)
        assert fg.path_a(st, src) == pytest.approx(sc.measure(st, [src])[0], abs=1e-12)
        j = fg.job("elect", p, 4e-4, fg.start, "start", "path_a")
        fg.compute([j])
        assert fg.value(j) == pytest.approx(sc.measure(st, [src])[0], abs=1e-12)


@pytest.mark.slow
def test_a_stage_started_again_reads_its_readings_back(runs, tmp_path):
    a, _, _ = runs
    out = str(tmp_path / "again")
    shutil.copytree(os.path.join(a, "final_grid"), out)
    rec0 = json.load(open(os.path.join(out, "election.json")))
    r = _resume(a, same_frames=False)
    lines = []
    rec = run_final_grid(r, out, in_run=True, log=lines.append)
    assert any("readings kept from the last start" in ln for ln in lines)
    # the background's clean images are reduced again, every reading comes from the file
    assert rec["reductions"] == len(r.results)
    def same(sw):
        return [[{k: v for k, v in row.items() if k != "reductions"} for row in rows] for rows in sw]
    assert same(rec["sweeps"]) == same(rec0["sweeps"]) and rec["calibration"] == rec0["calibration"]
    assert all(row["reductions"] == 0 for rows in rec["sweeps"] for row in rows)
    # a directory started on another grid starts afresh and says so
    r2 = _resume(a, same_frames=False)
    lines = []
    run_final_grid(r2, out, config=FinalGridConfig(mode="on", n_pa=3, curve_snr="off"), in_run=True,
                   log=lines.append)
    assert any("starts afresh" in ln for ln in lines)
    assert any(n.startswith("state_") for n in os.listdir(out))


@pytest.mark.slow
def test_a_changed_winner_reaches_the_final_products(runs, tmp_path, monkeypatch):
    a, _, _ = runs
    run = str(tmp_path / "run_c")
    shutil.copytree(a, run)
    orig = FG.elect_one
    calls = []

    def forced(readings, order, winner, z=1.0):
        best, stats = orig(readings, order, winner, z)
        calls.append(order)
        if len(order) > 1 and winner == order[0] and order[1] in readings and len(calls) in (2, 5):
            best = order[1]                                  # annulus 2 takes its second finalist
        elif len(calls) in (2, 5):
            best = winner
        return best, stats
    monkeypatch.setattr(FG, "elect_one", forced)
    r = _resume(run)
    r.run()
    fr = json.load(open(os.path.join(run, "final_results.json")))
    fg2 = fr["annuli"][1]["final_grid"]
    assert fg2["changed"] and fg2["config"] and fg2["held_out"]["n"] > 0
    assert config_key(fg2["config"]) != config_key(fr["annuli"][1]["winner_config"])
    # winner.json keeps the validation's winner and names the grid's, and the record says where its image is
    from klip_tpe.runner import AnnulusResult
    wj = AnnulusResult.from_dict(json.load(open(os.path.join(run, "annulus02", "winner.json"))))
    assert wj.final_grid == fg2 and wj.winner_config == fr["annuli"][1]["winner_config"]
    assert wj.final_config == fg2["config"] and wj.final_label.startswith("the final grid's winner")
    assert wj.final_image_path(run) == os.path.join(run, "final_grid", "annulus02_best_clean.fits")
    assert (np.isnan(wj.final_score) if fg2["validated_score"] is None
            else wj.final_score == pytest.approx(fg2["validated_score"]))
    kept = AnnulusResult.from_dict(json.load(open(os.path.join(run, "annulus01", "winner.json"))))
    assert kept.final_image_path(run, "inj") == os.path.join(run, "annulus01", "best_inj.fits")
    for f in ("best_clean.fits", "best_inj.fits", "best_clean_partitions.fits", "best_inj_partitions.fits"):
        assert os.path.exists(os.path.join(run, "final_grid", f"annulus02_{f}")), f
    # the parameters table takes the elected winner
    pt = [ln.split() for ln in open(os.path.join(run, "klip_stitched_params.txt")) if ln.strip().startswith("2 ")]
    assert int(pt[0][3]) == int(fg2["config"]["params"]["k_klip"])
    # a later resume reads the elected winner's images, and its final configuration is the elected one
    r2 = _resume(run, same_frames=False)
    res = r2.results[1]
    assert res.final_config == fg2["config"] and res.winner_config == fr["annuli"][1]["winner_config"]
    from klip_tpe.stitch import read_fits
    d = r2._annulus_images(1)
    assert np.array_equal(d["clean"], read_fits(os.path.join(run, "final_grid", "annulus02_best_clean.fits")),
                          equal_nan=True)
    # the re-score's base and the run's candidates take the elected winner
    assert config_key(StitchScorer(r2, log=QUIET).base_cfg[1].to_dict()) == config_key(fg2["config"])
    win = [e for e in run_entrants(run, 1) if e.kind == "winner"]
    assert len(win) == 1 and win[0].key == config_key(fg2["config"])
    # rereduce reduces the elected winner
    seen = []
    keep = r2._reduce

    def spy(cfg, sources, *args, **kw):
        seen.append((kw.get("tag", ""), config_key(cfg.to_dict())))
        return keep(cfg, sources, *args, **kw)
    r2._reduce = spy
    r2.rereduce(str(tmp_path / "re"), verify=False, log=QUIET)
    assert [k for t, k in seen if t == "rereduce_a2_clean"] == [config_key(fg2["config"])]


# ---------------------------------------------------------------------------- finished runs
@pytest.mark.slow
def test_elect_on_finished_runs_takes_other_runs_finalists_and_writes_the_products(runs, tmp_path):
    a, b, _ = runs
    before = {d: _snapshot(d) for d in (a, b)}
    out = str(tmp_path / "el")
    with pytest.raises(ValueError, match="beside the runs"):
        run_final_grid(_resume(b, same_frames=False), os.path.join(b, "x"), config=FinalGridConfig(mode="on"))
    pl = run_final_grid(_resume(b, same_frames=False), out, entrant_runs=[a], plan=True, log=QUIET,
                        config=FinalGridConfig(mode="on", n_pa=4))
    assert not os.path.exists(out) and pl["plan"]["election_sweep"] > 0
    assert {e["run"] for e in pl["finalists"]["2"]} == {"run_a", "run_b"}
    assert pl["finalists"]["2"][0]["run"] == "run_b" and pl["finalists"]["2"][0]["kind"] == "winner"
    fin = pl["finalists"]["2"]
    assert len({config_key(e["config"]) for e in fin}) == len(fin), "each configuration once"
    r = _resume(b, same_frames=False)
    from klip_tpe.final_grid import run_weights
    rec = run_final_grid(r, out, entrant_runs=[a], weights=run_weights(b, 3), log=QUIET,
                         config=FinalGridConfig(mode="on", n_pa=4, finalists=2))
    assert all(v["finalists"] <= 4 for v in rec["annuli"].values())
    elect_products(r, rec, out, log=QUIET)
    for f in ("klip_stitched.fits", "klip_stitched_inj.fits", "rereduce.json", "contrast_curve.txt", "election.txt"):
        assert os.path.exists(os.path.join(out, f)), f
    rr = json.load(open(os.path.join(out, "rereduce.json")))
    changed = {k for k, v in rec["annuli"].items() if v["changed"]}
    assert set(rr["composite"] or {}) == changed
    assert open(os.path.join(out, "contrast_curve.txt")).read().startswith("# Final-image")
    assert {d: _snapshot(d) for d in (a, b)} == before, "the runs are only read"


@pytest.fixture(scope="module")
def padded_run(tmp_path_factory):
    """A finished run of overlapping annuli saved with pads between them and the 2 px blend, as s7 and
    every run before 2026-11-25 were (``stitch_blend="weighted"``)."""
    d = str(tmp_path_factory.mktemp("final_grid_pads") / "run_p")
    red, space, obj, samp, cfg = build_synthetic_run(
        ann_edges=[8, 20, 32], ann_extra=[[14, 26]], n_iter=[5], n_init=3, seed=7, param_verify=False,
        stitch_every=0, verify=False, write_setup_files=False, fm_curve=False, stitch_overlap_pads=True,
        stitch_blend="weighted", stitch_injections="own", validation=ValidationConfig(n_top=2, n_valid=1),
        calibration=CalibrationConfig(forced=[2e-3, 2e-3, 2e-3]))
    Runner(red, space, obj, samp, cfg, d, log=QUIET).run()
    return d


@pytest.mark.slow
def test_elect_reads_the_stitch_asked_and_lists_its_winners_for_rereduce(padded_run, tmp_path):
    from klip_tpe.refine import read_picks, resolve_picks
    run = padded_run
    before = _snapshot(run)
    lines_pad, lines_sharp = [], []
    fc = FinalGridConfig(mode="on", n_pa=2, curve_snr="off", finalists=2)
    pl_pad = run_final_grid(_resume(run, same_frames=False), str(tmp_path / "p0"), config=fc, plan=True,
                            log=lines_pad.append)
    pl_sharp = run_final_grid(_resume(run, same_frames=False), str(tmp_path / "p1"), config=fc, plan=True,
                              overlap_pads=False, log=lines_sharp.append)
    # the run's own stitch has 2 px pads between its overlapping annuli, and --no-overlap-pads none
    assert pl_pad["stitch"]["overlap_pads"] and pl_pad["stitch"]["run_overlap_pads"] and pl_pad["stitch"]["overlapping"]
    assert pl_pad["stitch"]["edge_pads_px"] == [2.0, 2.0] and pl_sharp["stitch"]["edge_pads_px"] == [0.0, 0.0]
    assert pl_pad["stitch"]["tiles_px"] == pl_sharp["stitch"]["tiles_px"]
    assert any("the stitch read: 2 px pads at the edges between overlapping annuli" in ln for ln in lines_pad)
    assert any("no pads between overlapping annuli, the tiles meeting sharply there (the run's own stitch had pads"
               in ln for ln in lines_sharp)
    cov_pad = sum(v["cover"] for v in pl_pad["plan"]["annuli"].values())
    cov_sharp = sum(v["cover"] for v in pl_sharp["plan"]["annuli"].values())
    assert cov_pad > cov_sharp, "the pads widen the part of the stitch an annulus reaches"
    # read with each pixel from its own annulus (a new run's default), no edge is padded either way
    lines_own = []
    r_own = _resume(run, same_frames=False)
    r_own.cfg.stitch_blend = "own"
    pl_own = run_final_grid(r_own, str(tmp_path / "p2"), config=fc, plan=True, log=lines_own.append)
    assert pl_own["stitch"]["blend"] == "own" and pl_own["stitch"]["edge_pads_px"] == [0.0, 0.0]
    assert pl_own["stitch"]["tiles_px"] == pl_sharp["stitch"]["tiles_px"]
    assert any("the stitch read: no pads between overlapping annuli, each pixel from its tile's annulus "
               "(stitch blend own)" in ln for ln in lines_own), [ln for ln in lines_own if "stitch read" in ln]
    # the stage without pads: its record, its report and its winners
    out = str(tmp_path / "el")
    r = _resume(run, same_frames=False)
    rec = run_final_grid(r, out, config=fc, overlap_pads=False, log=QUIET)
    assert rec["stitch"]["overlap_pads"] is False and rec["stitch"]["pads_px"] == pl_sharp["stitch"]["pads_px"]
    assert json.load(open(os.path.join(out, "election.json")))["stitch"]["overlap_pads"] is False
    assert "# the stitch read: no pads between overlapping annuli" in open(os.path.join(out, "election.txt")).read()
    picks = read_picks(os.path.join(out, "winners.txt"), run, one_per_annulus=True, nann=3)
    assert sorted(p.annulus for p in picks) == [0, 1, 2]
    got = {p.annulus: e.key for p, e in resolve_picks(run, picks)}
    assert got == {int(k) - 1: config_key(v["winner"]["config"]) for k, v in rec["annuli"].items()}
    # the products take the same stitch and the injections asked for
    elect_products(r, rec, out, log=QUIET, overlap_pads=False, stitch_injections="all")
    rr = json.load(open(os.path.join(out, "rereduce.json")))
    assert rr["overlap_pads"] is False and rr["stitch_injections"] == "all"
    assert _snapshot(run) == before, "the run is only read"


@pytest.mark.slow
def test_another_runs_setup_is_refused(runs, tmp_path):
    a, _, _ = runs
    red, space, obj, samp, cfg = build_synthetic_run(
        ann_edges=[8, 26, 44], n_iter=[4], n_init=3, seed=5, param_verify=False, stitch_every=0, verify=False,
        write_setup_files=False, fm_curve=False, validation=ValidationConfig(n_top=2, n_valid=1),
        calibration=CalibrationConfig(forced=[2e-3, 2e-3]))
    other = str(tmp_path / "other")
    Runner(red, space, obj, samp, cfg, other, log=QUIET).run()
    with pytest.raises(ValueError, match="annuli"):
        collect_finalists(_resume(a, same_frames=False), [other])


def test_a_point_knows_its_role_and_record():
    p = GridPoint(3, 1, 4, 2, 5, 25.0, 30.0)
    assert p.role == "middle"
    assert GridPoint(0, 1, 4, 0, 5, 25.0, 30.0).role == "inner"
    assert GridPoint(0, 1, 4, 4, 5, 25.0, 30.0).role == "outer"
    assert GridPoint(0, 1, 4, 1, 5, 25.0, 30.0).role == "between"
    assert GridPoint(0, 1, 4, 0, 1, 25.0, 30.0).role == "middle"
    d = p.to_dict(0.01)
    assert d["annulus"] == 2 and d["rho_as"] == pytest.approx(0.25) and d["role"] == "middle"
    s = p.source(0.01, 1e-4)
    assert (s.rho, s.theta, s.contrast) == (pytest.approx(0.25), 30.0, 1e-4)
