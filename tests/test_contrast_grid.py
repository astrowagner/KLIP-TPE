"""The injection-recovery contrast curve (klip_tpe.contrast_grid), the protocol of the IDL
contrast generators (contrastcurve_*.pro).  What must hold:

* a grid of separations inside the zone and position angles per separation, or random positions;
* one injected source per reduction;
* each position iterated on its own contrast until its S/N is within the tolerance of the target,
  with the IDL's rules for a position that never gets there;
* the noise ring without the injected source, the known companions and their ADI sidelobes, and
  no test position in a companion's sidelobes;
* the median over position angle per separation with the 16-84 % band, on the final frame and in
  the run's contrast_curve.txt;
* the run's own random draws untouched, and a run saved before the grid resumes without it.
"""
from __future__ import annotations

import json
import os
import types

import numpy as np
import pytest

from klip_tpe import ContrastGridConfig, Runner, RunConfig, cli
from klip_tpe import contrast_grid as CG
from klip_tpe.metrics import aperture_mawet_snr, mawet_peak_snr, source_xy, star_center
from klip_tpe.stitch import write_contrast_curve

from conftest import build_synthetic_run

QUIET = lambda s: None                                     # noqa: E731


def test_the_grid_spans_the_zone_and_turns_from_one_separation_to_the_next():
    cfg = ContrastGridConfig(mode="grid", n_pa=12)
    seps = CG.separations_px(30, 70, 4.0, cfg)
    assert seps[0] >= 34 - 1e-9 and seps[-1] <= 66 + 1e-9 and np.allclose(np.diff(seps), 4.0)
    assert seps.size == 9
    pos = CG.grid_positions(seps, 12)
    assert len(pos) == 9 * 12
    for j in range(seps.size):
        th = np.sort([p.theta for p in pos if p.sep_index == j])
        assert np.allclose(np.diff(th), 30.0) and (th >= 0).all() and (th < 360).all()
    a0 = sorted(p.theta for p in pos if p.sep_index == 0)[0]
    a1 = sorted(p.theta for p in pos if p.sep_index == 1)[0]
    assert abs(a0 - a1) > 1.0, "the pattern turns between separations"
    # a zone narrower than its edges: one separation at its middle
    assert np.allclose(CG.separations_px(30, 34, 4.0, cfg), [32.0])
    # random positions bin to the nearest separation
    rp = CG.random_positions(seps, 4.0, 50, np.random.default_rng(1))
    assert len(rp) == 50 and all(abs(p.r_px - seps[p.sep_index]) <= 2.0 + 1e-9 for p in rp)


def test_the_sidelobes_are_the_field_rotation_never_less_than_two_fwhm():
    cfg = ContrastGridConfig(sidelobe="rotation", sidelobe_fwhm=2.0)
    assert CG.footprint_deg(50.0, 4.0, cfg, 34.0) == pytest.approx(34.0)
    floor = np.rad2deg(2.0 * 4.0 / 8.0)                  # close in, two FWHM of arc is wider
    assert CG.footprint_deg(8.0, 4.0, cfg, 34.0) == pytest.approx(floor)
    assert CG.footprint_deg(50.0, 4.0, cfg, None) == pytest.approx(np.rad2deg(8.0 / 50.0))
    fixed = ContrastGridConfig(sidelobe="fixed", sidelobe_fwhm=2.0)
    assert CG.footprint_deg(50.0, 4.0, fixed, 34.0) == pytest.approx(np.rad2deg(8.0 / 50.0))
    assert CG.in_footprint(50.0, 140.0, 50.0, 110.0, 34.0, 6.0)
    assert not CG.in_footprint(50.0, 150.0, 50.0, 110.0, 34.0, 6.0)
    assert not CG.in_footprint(60.0, 110.0, 50.0, 110.0, 34.0, 6.0), "too far out to be on its arc"
    assert CG.in_footprint(50.0, 355.0, 50.0, 10.0, 20.0, 6.0), "across north"


def _run(positions, snr_of, cfg):
    """run_grid with an S/N model: ``snr_of(pos, c)``.  Returns the positions, the number of
    reductions and what each reduction was asked for."""
    asked = []

    def reduce_many(batch):
        for p, c in batch:
            asked.append((p.index, c))
            yield ("image", p.index, c)

    def measure(res, p, c):
        return snr_of(p, c), None
    out, n = CG.run_grid(positions, lambda p: 1e-5, reduce_many, measure, cfg)
    return out, n, asked


def test_each_position_iterates_on_its_own_contrast_to_the_target():
    cfg = ContrastGridConfig(mode="grid", target_snr=5.0, precision=0.1)
    pos = CG.grid_positions([20.0, 30.0], 3)
    gain = {p.index: 1e5 * (1 + p.index) for p in pos}           # S/N per unit contrast, per position
    out, n, asked = _run(pos, lambda p, c: gain[p.index] * c, cfg)
    for p in out:
        assert p.status == "converged" and abs(p.trials[-1][1] / 5.0 - 1) <= 0.1
        assert p.contrast == pytest.approx(5.0 / gain[p.index], rel=0.1)
        assert len(p.trials) <= 2, "a linear source converges on the first correction"
    assert n == len(asked) == sum(len(p.trials) for p in out)


def test_the_idl_rules_for_a_position_that_never_gets_there():
    cfg = ContrastGridConfig(mode="grid", target_snr=5.0, precision=0.1, c_max=1.0)
    # not seen at first: the contrast goes up tenfold
    assert CG.next_contrast(1e-5, -0.3, 5.0, 1.0) == pytest.approx(1e-4)
    assert CG.next_contrast(1e-5, 2.5, 5.0, 1.0) == pytest.approx(2e-5)
    # always above the target: accepted from the sixth reduction on, at its last contrast
    out, _, _ = _run(CG.grid_positions([20.0], 1), lambda p, c: 8.0, cfg)
    assert out[0].status == "accepted" and len(out[0].trials) == 6
    # always short of it: dropped as blocked from the eighth
    out, _, _ = _run(CG.grid_positions([20.0], 1), lambda p, c: 3.0, cfg)
    assert out[0].status == "blocked" and len(out[0].trials) == 8 and out[0].contrast is None
    # at the contrast cap and still short: blocked there
    capped = ContrastGridConfig(mode="grid", c_max=2e-5)
    out, _, _ = _run(CG.grid_positions([20.0], 1), lambda p, c: 1e4 * c, capped)
    assert out[0].status == "blocked" and "cap" in out[0].why
    # no S/N at all (inside the inner guard, say): failed, with the reason, no retry
    out, _, _ = _run(CG.grid_positions([20.0], 1), lambda p, c: np.nan, cfg)
    assert out[0].status == "failed" and len(out[0].trials) == 1


def test_the_curve_is_the_median_over_pa_with_the_band():
    cfg = ContrastGridConfig(mode="grid")
    pos = CG.grid_positions([20.0, 30.0], 5)
    for p, c in zip(pos, [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]):
        p.status, p.contrast = "converged", c * 1e-6
    pos[9].status, pos[9].contrast = "blocked", None
    out = CG.summarize(pos, [20.0, 30.0], 0.01, cfg, [30.0, 30.0], 17)
    assert out["curve"] == pytest.approx([3e-6, 25e-6])
    assert out["p16"][0] == pytest.approx(np.percentile([1, 2, 3, 4, 5], 16) * 1e-6)
    assert out["n"] == [5, 4] and out["n_pos"] == [5, 5] and out["n_blocked"] == [0, 1]
    assert out["r_as"] == pytest.approx([0.2, 0.3]) and out["n_reductions"] == 17


def _planted(shape=(121, 121), px=0.05, fwhm=4.0):
    img = np.random.default_rng(5).standard_normal(shape)
    cx, cy = star_center(shape)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]

    def blob(rho, pa, amp):
        x, y = source_xy([rho], [pa], px, cx, cy)
        return amp * np.exp(-((xx - x[0]) ** 2 + (yy - y[0]) ** 2) / (2 * (fwhm / 2.355) ** 2))
    return img, blob


def test_the_noise_ring_leaves_out_the_arcs_given():
    """A negative sidelobe on the source's own ring, 20 deg away, inflates the noise and pulls
    the background down; with its arc excluded the S/N is that of the ring without it."""
    px, fwhm = 0.05, 4.0
    img, blob = _planted()
    img = img + blob(1.5, 100.0, 12.0) - blob(1.5, 120.0, 25.0) - blob(1.5, 80.0, 25.0)
    for fn in (lambda **k: mawet_peak_snr(img, [1.5], [100.0], px, fwhm, return_details=True, **k),
               lambda **k: aperture_mawet_snr(img, [1.5], [100.0], px, fwhm, return_details=True, **k)):
        open_, d0 = fn()
        shut, d1 = fn(excl_arcs=[(1.5, 100.0, 30.0, 1.5)])
        assert shut[0] > open_[0], "the lobes no longer count as noise"
        assert d1[0]["nclean"] < d0[0]["nclean"]
    # no arcs: exactly what it was
    a = mawet_peak_snr(img, [1.5], [100.0], px, fwhm)
    b = mawet_peak_snr(img, [1.5], [100.0], px, fwhm, excl_arcs=())
    assert np.array_equal(a, b, equal_nan=True)


@pytest.mark.slow
def test_a_run_makes_the_grid_one_source_at_a_time_and_leaves_its_draws_alone(tmp_path):
    out = {}
    for name, cg in (("on", ContrastGridConfig(mode="grid", n_pa=8)), ("off", ContrastGridConfig(mode="off"))):
        red, space, obj, samp, cfg = build_synthetic_run(n_iter=[10, 8], n_init=4, contrast_grid=cg)
        seps = CG.separations_px(cfg.ann_edges[0], cfg.ann_edges[1], red.fwhm, ContrastGridConfig())
        rho_k = float(seps[seps.size // 2]) * red.pxscale          # a companion on annulus 1's middle separation
        obj.metric.known = [(rho_k, 50.0)]
        r = Runner(red, space, obj, samp, cfg, str(tmp_path / name), log=QUIET)
        orig, seen = r._reduce, []

        def spy(cfg_, src, tag="", **kw):
            if tag.startswith("a") and "_cg" in tag:
                seen.append(len(src or []))
            return orig(cfg_, src, tag=tag, **kw)
        r._reduce = spy
        out[name] = (r.run(), seen, tmp_path / name)
    (on, seen, d), (off, seen_off, d_off) = out["on"], out["off"]
    assert seen and set(seen) == {1}, "one injected source per reduction"
    assert not seen_off and all(x.grid_curve is None for x in off)
    rec = lambda p: [(r_["score"], r_["sources"]) for r_ in map(json.loads, open(p / "results.jsonl"))]  # noqa: E731
    assert rec(d) == rec(d_off), "the grid draws nothing from the run's generator"
    g = on[0].grid_curve
    assert g is not None and len(g["r_as"]) == len(g["curve"]) == len(g["p16"]) == len(g["n"])
    js = json.load(open(d / "annulus01" / "contrast_grid.json"))
    assert js["n_reductions"] == sum(len(p["trials"]) for p in js["positions"])
    done = [p for p in js["positions"] if p["status"] in ("converged", "accepted")]
    assert done and all(abs(p["trials"][-1][1] / 5.0 - 1) <= 0.1 for p in done if p["status"] == "converged")
    ex = [p for p in js["positions"] if p["status"] == "excluded"]
    hw = max(js["rotation_deg"], np.rad2deg(2.0 * red.fwhm * red.pxscale / rho_k))
    assert ex, "the positions on the companion's arc are left out"
    for p in js["positions"]:                            # excluded exactly where its sidelobes are
        dpa = abs((p["theta"] - 50.0 + 180) % 360 - 180)
        on_arc = abs(p["rho_as"] - rho_k) <= 1.5 * red.fwhm * red.pxscale + 1e-9 and dpa <= hw + 1e-9
        assert (p["status"] == "excluded") == on_arc
    assert "injection-recovery" in open(d / "annulus01" / "contrast_curve.txt").read()
    assert "Injection-recovery" in open(d / "contrast_curve.txt").read().splitlines()[0]
    assert os.path.exists(d / "annulus01" / "contrast_grid.txt")
    w = json.load(open(d / "annulus01" / "winner.json"))
    assert w["grid_curve"]["curve"] == g["curve"]


def test_the_final_frame_draws_the_grid_and_the_run_level_file_leads_with_it(tmp_path):
    from klip_tpe.display import LiveDisplay
    disp = LiveDisplay(str(tmp_path), save_png=False, movie=False, pdf_every=0)
    gcv = {"r_as": [0.3, 0.5], "curve": [1e-4, 5e-5], "p16": [8e-5, 4e-5], "p84": [1.2e-4, 6e-5], "n": [4, 4]}
    vcv = {"r_as": [0.3, 0.5], "curve": [2e-4, 9e-5], "sample_r": [0.4], "sample_c": [1e-4]}
    res = types.SimpleNamespace(annulus=0, contrast_curve=vcv, fm_curve=None, grid_curve=gcv)
    runner = types.SimpleNamespace(results=[res], best_images={})
    labels = [c[0] for c in disp._curves(runner, types.SimpleNamespace(annulus=0), None, done=True)]
    assert any(lab.startswith("injection grid") for lab in labels)
    assert not any(lab.startswith("validated winner") for lab in labels), "the grid replaces the samples' fit"
    p = tmp_path / "contrast_curve.txt"
    write_contrast_curve(str(p), [vcv], [10, 30], 3.0, 0.02, grid_curves=[gcv])
    txt = open(p).read()
    assert txt.splitlines()[0].startswith("# Injection-recovery")
    rows = [ln.split() for ln in txt.splitlines() if ln and not ln.startswith("#")]
    assert float(rows[0][1]) == pytest.approx(1e-4) and "per-sample" in txt


def test_off_for_a_run_saved_before_and_the_options_parse():
    d = RunConfig(ann_edges=[8, 20]).to_dict()
    d.pop("contrast_grid")
    assert RunConfig.from_dict(d).contrast_grid.mode == "off"
    new = RunConfig.from_dict(RunConfig(ann_edges=[8, 20], contrast_grid=ContrastGridConfig(mode="grid", n_pa=24)).to_dict())
    assert new.contrast_grid.mode == "grid" and new.contrast_grid.n_pa == 24
    a = cli._build_parser().parse_args(["near", "--root", "/r"])
    g = cli._contrast_grid_config(a)
    assert g.mode == ContrastGridConfig().mode, "the library's default (grid; off inside this test suite)"
    assert (g.n_pa, g.target_snr, g.snr, g.sidelobe) == (12, 5.0, "matched", "rotation")
    assert cli._contrast_grid_config(cli._build_parser().parse_args(["near", "--root", "/r", "--contrast-grid",
                                                                       "grid"])).mode == "grid"
    a = cli._build_parser().parse_args(["near", "--root", "/r", "--contrast-grid", "random", "--cg-pa", "72",
                                        "--cg-target", "3", "--cg-snr", "aperture", "--cg-samples", "40"])
    g = cli._contrast_grid_config(a)
    assert (g.mode, g.n_pa, g.target_snr, g.snr, g.n_samples) == ("random", 72, 3.0, "aperture", 40)
