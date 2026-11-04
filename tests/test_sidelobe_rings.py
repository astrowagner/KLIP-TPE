"""Noise rings without the ADI sidelobes (RunConfig.sidelobes, metrics.Sidelobes).

ADI over-subtracts a source along the arc the field rotation sweeps at its radius.  Those
negative arcs in a noise ring pull its mean down and its scatter up, so every S/N the run
measures leaves them out: the injected sources' arcs and the known companions'.  Where fewer
than 4 apertures are left, the 4 or more farthest from the source stand in ("farthest"), or
the clean reduction's ring through the source ("clean").  The source count keeps every source 4 apertures outside all the
arcs and out of the others' arcs, so a second source goes in only where it biases nothing.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from klip_tpe import ContrastGridConfig, Runner, RunConfig, ValidationConfig
from klip_tpe.candidates import coverage_rim
from klip_tpe.metrics import (InjectionDifferenceSNR, MawetPeakSNR, Objective, Sidelobes, Source, mawet_peak_snr,
                              sidelobe_snr, source_xy, star_center)
from klip_tpe.runner import _few_apertures, _grid_ring_text
from klip_tpe.stitch import write_contrast_curve

from conftest import build_synthetic_run

PX, FWHM, RHO, THETA = 0.05, 4.0, 1.0, 0.0           # 20 px out, 31 apertures on its ring


def _gauss(img, x, y, amp, fwhm=FWHM):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    s = fwhm / 2.3548
    img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))


def _images(arc_deg=60.0, seed=3):
    """A clean noise frame, and the same with a source at (RHO, THETA) and the negative arcs
    ADI leaves beside it out to +/- ``arc_deg``."""
    rng = np.random.default_rng(seed)
    clean = rng.normal(0.0, 1.0, (101, 101))
    inj = clean.copy()
    cx, cy = star_center(inj.shape)
    x, y = source_xy([RHO], [THETA], PX, cx, cy, "pa")
    _gauss(inj, x[0], y[0], 30.0)
    for dth in np.arange(12.0, arc_deg + 0.1, 6.0):
        for sgn in (-1, 1):
            xa, ya = source_xy([RHO], [THETA + sgn * dth], PX, cx, cy, "pa")
            _gauss(inj, xa[0], ya[0], -6.0)
    return clean, inj


def test_the_arcs_leave_the_sidelobes_out_of_the_ring():
    clean, inj = _images()
    s0, d0 = mawet_peak_snr(inj, [RHO], [THETA], PX, FWHM, return_details=True)
    s1, d1 = sidelobe_snr(inj, [RHO], [THETA], PX, FWHM, Sidelobes(rotation_deg=60.0))
    d1 = d1[0]
    assert d1["fallback"] is None and d1["arc_deg"] == pytest.approx(60.0)
    assert d1["n_arcs"] == 20, "31 apertures, less the source's own and the 10 within 60 deg of it"
    assert d1["bg"] > d0[0]["bg"] and d1["sigma"] < d0[0]["sigma"], "the negative arcs stayed in the ring"
    # left in, the arcs lower the ring's mean (which lifts the source above it) and widen its
    # scatter; here the scatter weighs more and they pull the S/N down
    assert s1[0] > 2 * s0[0]


def test_too_few_apertures_take_the_farthest():
    clean, inj = _images(arc_deg=168.0)
    s, d = sidelobe_snr(inj, [RHO], [THETA], PX, FWHM, Sidelobes(rotation_deg=170.0))
    d = d[0]
    assert d["n_arcs"] == 2 and d["fallback"] == "farthest"     # 2 left beside the arcs
    assert d["nclean"] == 4, "the four apertures farthest from the source"
    assert 150.0 < d["arc_deg"] < 170.0 and np.isfinite(s[0])


def test_too_few_apertures_take_the_clean_ring():
    clean, inj = _images(arc_deg=168.0)
    s, d = sidelobe_snr(inj, [RHO], [THETA], PX, FWHM, Sidelobes(rotation_deg=170.0, mode="clean"), img_clean=clean)
    d = d[0]
    assert d["fallback"] == "clean"
    _, di = mawet_peak_snr(inj, [RHO], [THETA], PX, FWHM, return_details=True)
    _, dc = mawet_peak_snr(clean, [RHO], [THETA], PX, FWHM, min_ring=4, return_details=True,
                           peak_xy=[(di[0]["x"], di[0]["y"])])
    assert dc[0]["r"] == di[0]["r"], "the clean ring runs through the source as located"
    want = (di[0]["peak"] - dc[0]["bg"]) / (dc[0]["sigma"] * np.sqrt(1 + 1 / max(dc[0]["nap"], 3)))
    assert s[0] == pytest.approx(want)
    assert d["nclean"] == dc[0]["nclean"] > 20, "the clean ring has no sidelobes to leave out"
    # without a clean image the mode takes the farthest apertures
    _, d2 = sidelobe_snr(inj, [RHO], [THETA], PX, FWHM, Sidelobes(rotation_deg=170.0, mode="clean"))
    assert d2[0]["fallback"] == "farthest"


def test_the_clean_ring_runs_through_the_injected_source():
    """The clean image holds no source, so a peak search of its own settles on a noise peak.
    Here a bump sits 1 px inside the source, where the ring holds 3 apertures.  The ring
    through the source as the injected image located it holds 4."""
    rng = np.random.default_rng(5)
    clean = rng.normal(0.0, 1.0, (101, 101))
    cx, cy = star_center(clean.shape)
    rho = 0.23                                       # 4.6 px: the peak lands 5 px out, 7 apertures
    x, y = source_xy([rho], [0.0], PX, cx, cy, "pa")
    _gauss(clean, x[0], y[0] - 0.8, 20.0)
    inj = clean.copy()
    _gauss(inj, x[0], y[0], 2000.0)
    _, di = mawet_peak_snr(inj, [rho], [0.0], PX, FWHM, return_details=True)
    _, dn = mawet_peak_snr(clean, [rho], [0.0], PX, FWHM, min_ring=4, return_details=True)
    assert di[0]["r"] == pytest.approx(5.0) and dn[0]["r"] == pytest.approx(4.0), "the setup"
    assert dn[0]["nclean"] == 3, "the noise peak's ring"
    s, d = sidelobe_snr(inj, [rho], [0.0], PX, FWHM, Sidelobes(rotation_deg=149.0, mode="clean"), img_clean=clean)
    d = d[0]
    assert d["n_arcs"] == 2 and d["fallback"] == "clean" and d["nclean"] == 4
    _, dc = mawet_peak_snr(clean, [rho], [0.0], PX, FWHM, min_ring=4, return_details=True,
                           peak_xy=[(di[0]["x"], di[0]["y"])])
    assert dc[0]["r"] == di[0]["r"] and d["sigma"] == pytest.approx(dc[0]["sigma"])
    assert np.isfinite(s[0])


def test_a_ring_too_small_for_four_says_so():
    """At 1 FWHM the whole ring holds 3 apertures outside 1.5 FWHM of the source, so neither
    fallback reaches 4, and the grid's reason names the ring rather than the sidelobes."""
    rng = np.random.default_rng(6)
    noise = rng.normal(0.0, 1.0, (101, 101))
    cx, cy = star_center(noise.shape)
    rho = 0.21                                       # 4.2 px: the peak lands 4 px out, 6 apertures
    x, y = source_xy([rho], [0.0], PX, cx, cy, "pa")
    img = noise.copy()
    _gauss(img, x[0], y[0], 2000.0)
    _, d = sidelobe_snr(img, [rho], [0.0], PX, FWHM, Sidelobes(rotation_deg=149.0))
    d = d[0]
    assert d["n_arcs"] == 1 and d["fallback"] is None
    assert d["fallback_tried"] == "farthest" and d["n_fallback"] == 3
    assert _few_apertures(d, 4, 149.0) == "3 noise apertures on the whole ring (fewer than 4)"
    _, dc = sidelobe_snr(img, [rho], [0.0], PX, FWHM, Sidelobes(rotation_deg=149.0, mode="clean"), img_clean=noise)
    assert dc[0]["fallback_tried"] == "clean"
    assert _few_apertures(dc[0], 4, 149.0) == "3 noise apertures on the clean reduction's ring (fewer than 4)"
    # with the rings as before, no fallback: the count beside the arcs, 1 or 2 no longer read as 0
    assert (_few_apertures({"nclean": 0, "n_left": 1}, 6, 170.0)
            == "1 noise aperture left beside the +/- 170 deg sidelobes (fewer than 6)")


def test_every_source_leaves_out_the_others_arcs():
    """Two sources on one ring, 180 deg apart: each ring loses the other's arc too."""
    clean, _ = _images()
    rule = Sidelobes(rotation_deg=40.0)
    _, one = sidelobe_snr(clean, [RHO], [THETA], PX, FWHM, rule)
    _, two = sidelobe_snr(clean, [RHO, RHO], [THETA, THETA + 180.0], PX, FWHM, rule)
    assert two[0]["n_arcs"] < one[0]["n_arcs"]
    assert two[0]["n_arcs"] == two[1]["n_arcs"] >= 4


def test_the_objective_hands_the_rule_to_a_metric_that_takes_it():
    clean, inj = _images()
    m = MawetPeakSNR(pxscale=PX, fwhm=FWHM)
    obj = Objective(m, clean_subtract=False)
    src = [Source(RHO, THETA, 1.0)]
    plain = obj.score_raw(inj, src).score
    obj.sidelobes = Sidelobes(rotation_deg=60.0)
    ringed = obj.score_raw(inj, src).score
    assert ringed == pytest.approx(sidelobe_snr(inj, [RHO], [THETA], PX, FWHM, obj.sidelobes,
                                                kernel=m.kernel([RHO]))[0][0])
    assert ringed != pytest.approx(plain)
    assert obj.describe()["sidelobes"]["rotation_deg"] == 60.0
    other = Objective(InjectionDifferenceSNR(pxscale=PX, fwhm=FWHM), clean_subtract=False)
    other.sidelobes = Sidelobes(rotation_deg=60.0)
    assert np.isfinite(other.score_raw(inj, src, clean).score), "a metric without the keyword is left as it was"
    assert "sidelobes" not in other.describe()


def test_a_new_run_installs_the_rule_and_an_old_one_keeps_its_rings(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run(sidelobes="farthest")
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "new"), log=lambda s: None)
    assert r.objective.sidelobes is not None and r.objective.sidelobes.mode == "farthest"
    assert r.objective.sidelobes.rotation_deg == pytest.approx(r._pa_coverage())
    d = cfg.to_dict()
    d.pop("sidelobes")
    assert RunConfig.from_dict(d).sidelobes == "off", "a run saved before the rule resumes without it"
    red, space, obj, samp, cfg = build_synthetic_run(sidelobes="off")
    assert Runner(red, space, obj, samp, cfg, str(tmp_path / "off"), log=lambda s: None).objective.sidelobes is None
    with pytest.raises(ValueError):
        red, space, obj, samp, cfg = build_synthetic_run(sidelobes="nearest")
        Runner(red, space, obj, samp, cfg, str(tmp_path / "bad"), log=lambda s: None)


@pytest.mark.parametrize("coverage,want", [(150.0, 1), (40.0, 2)])
def test_the_count_keeps_a_pair_out_of_each_others_sidelobes(tmp_path, monkeypatch, coverage, want):
    """A co-radial pair 180 deg apart: with +/-150 deg of rotation each source's ring beside
    its own arc lies in the other's, so one source goes in.  With +/-40 deg both do."""
    monkeypatch.setattr(Runner, "_pa_coverage", lambda self: coverage)
    monkeypatch.setattr(Runner, "_probe_shape", lambda self: (100, 100))   # the synthetic frames hold no cube
    logs = []
    red, space, obj, samp, cfg = build_synthetic_run(sidelobes="farthest", n_sources=2, ann_edges=[8, 30],
                                                     n_iter=[4], n_init=3)
    r = Runner(red, space, obj, samp, cfg, str(tmp_path), log=logs.append)
    assert r._nsrc(0) == want
    if want == 1:
        assert any("2 sources requested, 1 used" in l and "sidelobe" in l for l in logs), logs
    else:
        # an annulus the rule leaves alone says so too (NEAR2 s6 note, section 4)
        assert any("annulus 1: 2 sources a draw, as asked, and each keeps at least 4 noise apertures outside "
                   "the sources' ADI sidelobes (+/- 40 deg at the outer edge)" in l for l in logs), logs


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["farthest", "off"])
def test_the_grid_measures_inside_wide_sidelobes(tmp_path, monkeypatch, mode):
    """+/-170 deg of rotation leaves a ring 1 or 2 apertures beside a source's arcs.  Under the
    run's rule the grid measures there on the 4 or more farthest.  With the run's rings off and
    the grid's own rotation arcs it fails, and says why."""
    monkeypatch.setattr(Runner, "_rotation_deg", lambda self, cfg: 170.0)
    logs = []
    red, space, obj, samp, cfg = build_synthetic_run(sidelobes=mode, ann_edges=[8, 30], n_iter=[5], n_init=3,
                                                     validation=ValidationConfig(n_top=1, n_valid=1),
                                                     contrast_grid=ContrastGridConfig(mode="grid", n_pa=4, snr="matched",
                                                                                      sidelobe="rotation"))
    Runner(red, space, obj, samp, cfg, str(tmp_path), log=logs.append).run()
    pos = json.load(open(os.path.join(str(tmp_path), "annulus01", "contrast_grid.json")))["positions"]
    if mode == "farthest":
        far = [p for p in pos if p["noise"] == "farthest" and p["status"] in ("converged", "accepted")]
        assert far, pos
        assert any("measured on the 4 or more apertures farthest from the source" in l for l in logs), logs
    else:
        assert any(p["status"] == "failed" and "beside the +/- 170 deg sidelobes" in (p["why"] or "") for p in pos)
        assert any("position(s) failed:" in l and "sidelobes" in l for l in logs), logs


def test_the_contrast_curve_names_each_method(tmp_path):
    val = {"r_as": [0.3, 0.4], "curve": [1e-4, 8e-5], "sample_r": [], "sample_c": []}
    grid = {"r_as": [0.6, 0.7], "curve": [5e-5, 4e-5]}
    path = str(tmp_path / "cc.txt")
    write_contrast_curve(path, [val, dict(val, r_as=[0.6, 0.7])], [6.0, 10.0, 15.0], 4.0, 0.05,
                         grid_curves=[None, grid])
    head = open(path).readline()
    assert "two methods" in head and "Injection grid" in head and "annuli 2" in head
    assert "Validation injections (no grid curve): annuli 1" in head
    write_contrast_curve(path, [val, val], [6.0, 10.0, 15.0], 4.0, 0.05, grid_curves=[grid, grid])
    assert open(path).readline().startswith("# Injection-recovery SNR=5 contrast curve")


def test_the_candidate_search_stops_inside_the_rim():
    img = np.ones((121, 121))
    cx, cy = star_center(img.shape)
    yy, xx = np.mgrid[0:121, 0:121]
    rr = np.hypot(xx - cx, yy - cy)
    rim = img.copy()
    rim[(rr >= 50.5) & (xx < cx)] = np.nan                      # half of the rings past 50 px
    rim[rr >= 52.5] = np.nan
    assert coverage_rim(rim, None, 10, 58) == 51.0
    wedge = img.copy()
    wedge[np.abs(xx - cx) < 2] = np.nan                         # a share of every ring, as a 4QPM edge
    assert coverage_rim(wedge, None, 10, 58) is None
    # a night one annulus keeps and the next leaves out: half a ring at the seam, then nothing
    nights = np.stack([img.copy(), img.copy()])
    nights[1][(rr >= 30.5) & (yy < cy)] = np.nan
    nights[1][rr >= 31.5] = np.nan
    assert coverage_rim(img, nights, 10, 58) == 31.0
    assert coverage_rim(img, nights, 10, 58, stack_from=45.0) is None, "a seam is not the rim"
    nights[0][(rr >= 54.5) & (xx > cx)] = np.nan
    assert coverage_rim(img, nights, 10, 58, stack_from=45.0) == 55.0, "a night that stops covering the field"


def test_the_command_line_takes_the_rule():
    from klip_tpe import cli
    p = cli._build_parser()
    a = p.parse_args(["near", "--root", "x", "--sidelobes", "clean"])
    assert cli._protocol_config(a)["sidelobes"] == "clean"
    a = p.parse_args(["near", "--root", "x"])
    assert "sidelobes" not in cli._protocol_config(a), "without the option the library's default stands"
    g = cli._protocol_config(a)["contrast_grid"]
    assert (g.snr, g.sidelobe) == ("aperture", "off"), "the IDL contrast generators' S/N and ring"


def test_the_old_rings_are_the_default_again_and_the_grid_takes_the_idl_generators():
    """Kevin Wagner, 2026-11-01: the rule left NEAR's rings 1 to 3 apertures inside 0.93" and one
    source a draw, and the grid is to measure as the IDL contrast generators do.  Saved runs
    resume as they were."""
    import inspect
    from klip_tpe import runner as runner_mod
    # the suite pins the default itself (conftest), so the library's is read from its source
    assert '_SIDELOBES_DEFAULT = {"mode": "off"}' in inspect.getsource(runner_mod)
    assert (ContrastGridConfig().snr, ContrastGridConfig().sidelobe) == ("aperture", "off")
    d = RunConfig(contrast_grid=ContrastGridConfig(snr="matched", sidelobe="rotation")).to_dict()
    g = RunConfig.from_dict(d).contrast_grid
    assert (g.snr, g.sidelobe) == ("matched", "rotation"), "a run saved with the old grid keeps it"
    d["sidelobes"] = "farthest"
    assert RunConfig.from_dict(d).sidelobes == "farthest", "a run saved under the rule keeps it"


def test_the_aperture_ring_is_the_contrast_generators():
    """contrastcurve_near.pro: ntheta = round(2 pi r / FWHM) - 1 positions from the source's, the
    source and one each side skipped (ntheta >= 6), the source alone otherwise, and the source's
    aperture where it was put."""
    from klip_tpe.metrics import RingRecorder, aperture_mawet_snr
    rng = np.random.default_rng(5)
    img = rng.standard_normal((81, 81))
    for rho in (0.9, 0.3, 0.25):
        sep = rho / PX
        ntheta = int(round(2 * np.pi * sep / FWHM)) - 1
        with RingRecorder() as rec:
            _, d = aperture_mawet_snr(img, [rho], [30.0], PX, FWHM, return_details=True)
        ring = rec.records[0]["ring"]
        assert len(ring) == ntheta - 1, (rho, ntheta, len(ring))
        skipped = [k + 1 for k, a in enumerate(ring) if a[2] != "used"]
        assert skipped == ([1, ntheta - 1] if ntheta >= 6 else []), (rho, skipped)
        assert d[0]["nclean"] == (ntheta - 3 if ntheta >= 6 else ntheta - 1)


def test_the_sampler_draws_no_source_in_anothers_sidelobe(tmp_path):
    from klip_tpe.positions import PositionSampler
    smp = PositionSampler(fwhm_as=0.2, strategy="fixed_pa", sidelobes=Sidelobes(rotation_deg=62.0))
    rng = np.random.default_rng(5)
    for _ in range(50):
        a, b = smp.sample(2, 1.0, 1.0, rng)
        assert abs((a.theta - b.theta + 180.0) % 360.0 - 180.0) > 62.0
    # the run hands its rule to the sampler
    red, space, obj, samp, cfg = build_synthetic_run(sidelobes="farthest")
    r = Runner(red, space, obj, samp, cfg, str(tmp_path), log=lambda s: None)
    assert r.sampler.sidelobes is r.objective.sidelobes is not None


@pytest.mark.slow
def test_a_run_covered_out_to_its_last_edge_keeps_its_owa(tmp_path):
    """The rim rule must not bite where the frames cover the field: the candidate search of
    a synthetic run, whose frames reach past its last edge, keeps its outer edge."""
    logs = []
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30, 45], n_iter=[4, 4], n_init=3, candidates=True,
                                                     validation=ValidationConfig(n_top=1, n_valid=1))
    Runner(red, space, obj, samp, cfg, str(tmp_path), log=logs.append).run()
    assert os.path.exists(os.path.join(str(tmp_path), "cand"))
    assert not [l for l in logs if "candidate search: OWA" in l], [l for l in logs if "OWA" in l]


def test_the_grid_calls_its_keep_out_a_keep_out():
    """With no sidelobe arcs (the grid's "off", the default) the angle the grid logs is the
    1.5 FWHM keep-out, as the IDL contrast generators skip the source and the aperture each
    side.  mwc758_ales_repeat logged it as "sidelobes +/- 14.4 deg at 0.60"" (MWC 758 ALES
    session, 2026-11-03)."""
    from klip_tpe import contrast_grid as CG
    r_px, fw, px = 0.60 / 0.0345, 2.9083217, 0.0345
    hw = CG.footprint_deg(r_px, fw, CG.ContrastGridConfig(), 120.0)
    assert round(hw, 1) == 14.4
    assert (_grid_ring_text("off", None, hw, r_px, fw, px)
            == 'no sidelobe arcs, the ring skips +/- 14.4 deg (1.5 FWHM) beside the source at 0.60"')
    assert _grid_ring_text("rotation", None, 40.0, r_px, fw, px) == 'sidelobes +/- 40.0 deg at 0.60"'
    assert _grid_ring_text("off", Sidelobes(rotation_deg=149.0), hw, r_px, fw, px).startswith(
        "the run's sidelobe rule, +/- 149.0 deg")
    assert (_few_apertures({"nclean": 0, "n_left": 1}, 6, hw, keepout=True)
            == "1 noise aperture left beside the 1.5 FWHM keep-out, +/- 14 deg (fewer than 6)")
