"""The sidelobe rule in the verify and the candidate search (NEAR2 s6 note, section 4).

Since b992328 the run's own S/N (search, calibration, validation, the contrast grid) leaves the
ADI sidelobes of every injected source and known companion out of its noise rings.  The verify's
Path A and subsets, its STIM annulus and solved contrasts, the candidates' S/N and the blind S/N
map now take the same rule.  The verify's fallback is the farthest samples, then the other
sources' arcs let in, then the known companions'.  The blind map leaves out only the known
companions' arcs.  With no rule every number is the one before.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from klip_tpe.candidates import candidate_metrics, find_candidates
from klip_tpe.metrics import Sidelobes, sidelobe_snr, source_xy
from klip_tpe.products import snr_map
from klip_tpe.runner import Runner
from klip_tpe.verify import (mf_convolve, ring_snr, snr_fap, stim_fap, verify_candidates, verify_injections,
                             write_verify_table)

PX, FWHM, N = 0.0445, 6.4, 151
C = (N - 1) / 2.0


def _blob(img, rho, pa, amp, sigma=FWHM / 2.355):
    yy, xx = np.mgrid[0:N, 0:N]
    x, y = source_xy([rho], [pa], PX, C, C)
    img += amp * np.exp(-((xx - x[0]) ** 2 + (yy - y[0]) ** 2) / (2 * sigma ** 2))
    return img


def _lobed(rho, pa, amp=30.0, lobe=-9.0, span=50.0, seed=1):
    """White noise, a source at (rho, pa) and its ADI self-subtraction lobes: negative blobs
    along its arc out to +/- ``span`` deg."""
    img = np.random.default_rng(seed).normal(size=(N, N))
    _blob(img, rho, pa, amp)
    step = np.rad2deg(FWHM / (rho / PX))
    for d in np.arange(step, span + 1e-9, step):
        _blob(img, rho, pa + d, lobe)
        _blob(img, rho, pa - d, lobe)
    return img


def test_the_arcs_leave_their_samples_out_and_no_arcs_change_nothing():
    img = _lobed(0.8, 40.0)
    plain = snr_fap(img, 0.8, 40.0, PX, FWHM)
    same = snr_fap(img, 0.8, 40.0, PX, FWHM, excl_arcs=())
    assert {k: plain[k] for k in ("snr", "fap", "nap", "peak", "x", "y")} == \
        {k: same[k] for k in ("snr", "fap", "nap", "peak", "x", "y")}
    arc = snr_fap(img, 0.8, 40.0, PX, FWHM, excl_arcs=[(0.8, 40.0, 60.0, 1.5)])
    assert arc["n_left"] < plain["n_left"]
    # the lobes pull the ring's mean down; without them it sits at the noise
    assert plain["bg"] < arc["bg"] and abs(arc["bg"]) < abs(plain["bg"])


def test_the_ring_falls_back_to_the_farthest_samples_and_then_lets_the_arcs_in():
    img = _lobed(0.5, 100.0)
    rule = Sidelobes(rotation_deg=150.0)
    d = ring_snr(img, 0, [0.5], [100.0], PX, FWHM, rule)
    assert d["fallback"] == "farthest" and d["nclean"] >= 4 and 0 < d["arc_deg"] < 150.0
    assert np.isfinite(d["snr"]) and d["nap"] == d["nclean"]
    # two sources 180 deg apart at one radius: the other's arc takes the whole ring, so it is
    # let in, and the ring is the one without the rule (the other's keep-out stays)
    two = _lobed(0.6, 20.0)
    _blob(two, 0.6, 200.0, 30.0)
    xy = list(zip(*source_xy([0.6], [200.0], PX, C, C)))
    d2 = ring_snr(two, 0, [0.6, 0.6], [20.0, 200.0], PX, FWHM, rule, exclude_xy=xy)
    assert d2["fallback"] == "farthest, the other sources' arcs too" and d2["arcs"] == []
    assert d2["snr"] == snr_fap(two, 0.6, 20.0, PX, FWHM, exclude_xy=xy)["snr"]
    # a known companion's arc is let in last, after the other sources'
    d3 = ring_snr(two, 0, [0.6, 0.6], [20.0, 200.0], PX, FWHM, rule, exclude_xy=xy, known=[(0.62, 300.0)])
    assert d3["fallback"] == "farthest, every arc let in"
    # the run's own order is unchanged: known companions' arcs, and nothing after them
    snr_all, dets = sidelobe_snr(two, [0.6, 0.6], [20.0, 200.0], PX, FWHM, rule)
    assert all(dd.get("fallback") in (None, "farthest", "farthest, known arcs too") for dd in dets)
    snr1, dets1 = sidelobe_snr(two, [0.6, 0.6], [20.0, 200.0], PX, FWHM, rule, only=1)
    assert np.isnan(snr1[0]) and dets1[0] == {} and snr1[1] == snr_all[1] and dets1[1]["arcs"] == dets[1]["arcs"]


def _stack(rho, pas, nn=4, amp=6.0, seed=3):
    rng = np.random.default_rng(seed)
    st = []
    for j in range(nn):
        img = rng.normal(size=(N, N))
        for pa in pas:
            _blob(img, rho, pa, amp)
            step = np.rad2deg(FWHM / (rho / PX))
            for d in np.arange(step, 40.0, step):
                _blob(img, rho, pa + d, -0.3 * amp)
                _blob(img, rho, pa - d, -0.3 * amp)
        st.append(img)
    return np.stack(st)


def test_the_injection_block_takes_the_rule_and_without_it_nothing_changes():
    clean = _stack(1.4, [])
    src = [(1.4, 30.0, 1e-4), (1.4, 150.0, 1e-4), (1.4, 270.0, 1e-4)]
    inj = _stack(1.4, [s[1] for s in src])
    rows0, maps0 = verify_injections(inj, src, FWHM, PX, combined=inj.mean(axis=0), clean=clean.mean(axis=0))
    rows1, maps1 = verify_injections(inj, src, FWHM, PX, combined=inj.mean(axis=0), clean=clean.mean(axis=0),
                                     sidelobes=None)
    for a, b in zip(rows0, rows1):
        assert {k: a[k] for k in ("snr", "fap", "nap", "stim_fap", "c_det")} == \
            {k: b[k] for k in ("snr", "fap", "nap", "stim_fap", "c_det")}
        assert "noise" not in a
    rule = Sidelobes(rotation_deg=25.0)
    rows, maps = verify_injections(inj, src, FWHM, PX, combined=inj.mean(axis=0), clean=clean.mean(axis=0),
                                   sidelobes=rule)
    for r, r0 in zip(rows, rows0):
        assert r["noise"]["fallback"] is None and r["noise"]["n_used"] == r["nap"] < r0["nap"]
        assert np.isfinite(r["snr"]) and r["snr"] != r0["snr"]
        assert "c_det" in r and len(r["c_tap_solved"]) == 3 and set(r["subsets"]) == set(r0["subsets"])


def test_a_candidate_at_a_known_companion_keeps_the_others_arcs_and_not_its_own_twice():
    st = _stack(0.8, [60.0], amp=8.0)
    known = [(0.8, 60.0), (0.8, 240.0)]
    rule = Sidelobes(rotation_deg=40.0)
    rows = verify_candidates(st, None, [known[0]], FWHM, PX, known=known, n_boot=0, sidelobes=rule)
    r = rows[0]
    assert r["noise"]["fallback"] is None and np.isfinite(r["snr"])
    # its own arc once (+/- 40 deg about 60), and the other companion's (+/- 40 about 240)
    d = ring_snr(st.mean(axis=0), 0, [0.8], [60.0], PX, FWHM, rule, known=[known[1]],
                 exclude_xy=list(zip(*source_xy([k[0] for k in known], [k[1] for k in known], PX, C, C))))
    assert d["snr"] == r["snr"] and len(d["arcs"]) == 2
    assert rows[0]["meta"]["sidelobes"]["rotation_deg"] == 40.0
    old = verify_candidates(st, None, [known[0]], FWHM, PX, known=known, n_boot=0)
    assert "noise" not in old[0] and old[0]["meta"]["sidelobes"] is None and old[0]["nap"] > r["nap"]


def test_the_stim_annulus_leaves_the_arcs_out():
    stim = np.random.default_rng(5).normal(size=(N, N))
    x, y = source_xy([0.8], [10.0], PX, C, C)
    v0, f0 = stim_fap(stim, x[0], y[0], 0.8 / PX, FWHM)
    v1, f1 = stim_fap(stim, x[0], y[0], 0.8 / PX, FWHM, excl_arcs=[(0.8, 10.0, 90.0, 1.5)], pxscale=PX)
    assert v0 == v1 and f0 != f1


def test_the_candidates_take_the_rule_and_the_blind_map_only_the_known_arcs(tmp_path):
    st = _stack(0.8, [60.0], amp=10.0)
    img = st.mean(axis=0)
    known = [(0.8, 200.0)]
    rule = Sidelobes(rotation_deg=40.0)
    plain = snr_map(img, FWHM)
    arcs = [(0.8, 200.0, rule.half_width(0.8 / PX, FWHM), rule.radial_fwhm)]
    with_arcs = snr_map(img, FWHM, excl_arcs=arcs, pxscale=PX)
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.hypot(xx - C, yy - C)
    differ = ~np.isclose(plain, with_arcs, equal_nan=True)
    assert differ.any() and np.all(np.abs(rr[differ] - 0.8 / PX) <= 1.5 * FWHM + 1.0), \
        "only the rings the known companion's arc crosses"
    assert np.array_equal(snr_map(img, FWHM, excl_arcs=arcs), plain, equal_nan=True), "no pxscale, no arcs"
    rows = find_candidates(img, st, FWHM, PX, known=known, snrmin=3.0, verify_top=1,
                           verify_kwargs={"n_boot": 0}, sidelobes=rule)
    assert rows and "noise" in rows[0] and rows[0]["verify"]["noise"]["fallback"] in (None, "farthest")
    rows0 = find_candidates(img, st, FWHM, PX, known=known, snrmin=3.0)
    assert rows0 and "noise" not in rows0[0]
    px_, py_ = np.array([rows[0]["x"]]), np.array([rows[0]["y"]])
    m = candidate_metrics(px_, py_, np.array([5.0]), st, FWHM, PX, 1.2 * FWHM, 70.0, known=known, sidelobes=rule)
    assert m[0]["noise"]["n_used"] >= 4


def test_the_report_says_how_the_rings_came_out(tmp_path):
    clean = _stack(0.5, [])
    src = [(0.5, 30.0, 1e-4), (0.5, 210.0, 1e-4)]
    inj = _stack(0.5, [s[1] for s in src])
    rule = Sidelobes(rotation_deg=150.0)
    rows, maps = verify_candidates(clean, None, [], FWHM, PX, injected=(inj, src), n_boot=0, return_maps=True,
                                   combined=clean.mean(axis=0), combined_inj=inj.mean(axis=0), sidelobes=rule)
    txt = write_verify_table(str(tmp_path / "v.txt"), rows, maps.get("limits"))
    assert "# noise rings: the sidelobe rule." in txt and "+/- 150 deg" in txt
    assert "and the ADI arcs of the sidelobe rule (above)" in txt
    assert "#   sidelobe rule: 0 of 2 rings on their samples outside every arc, the others:" in txt
    assert "farthest, the other sources' arcs too" in txt
    rows0, maps0 = verify_candidates(clean, None, [], FWHM, PX, injected=(inj, src), n_boot=0, return_maps=True,
                                     combined=clean.mean(axis=0), combined_inj=inj.mean(axis=0))
    txt0 = write_verify_table(str(tmp_path / "v0.txt"), rows0, maps0.get("limits"))
    assert "sidelobe" not in txt0


def test_the_runner_hands_its_rule_to_the_products():
    rule = Sidelobes(rotation_deg=40.0)
    stub = types.SimpleNamespace(objective=types.SimpleNamespace(sidelobes=rule),
                                 sampler=types.SimpleNamespace(known=[(0.8, 200.0)]), pxscale=PX, fwhm=FWHM,
                                 reducer=types.SimpleNamespace(angle_convention="pa"))
    stub._verify_rule = lambda: Runner._verify_rule(stub)
    stub._known_arcs = lambda: Runner._known_arcs(stub)
    stub._pixel_mask = lambda: None
    arcs = Runner._known_arcs(stub)
    assert arcs == [(0.8, 200.0, rule.half_width(0.8 / PX, FWHM), rule.radial_fwhm)]
    img = _stack(0.8, [60.0]).mean(axis=0)
    assert np.array_equal(Runner._snr_map(stub, img), snr_map(img, FWHM, excl_arcs=arcs, pxscale=PX), equal_nan=True)
    stub.objective.sidelobes = None
    assert Runner._known_arcs(stub) == [] and np.array_equal(Runner._snr_map(stub, img), snr_map(img, FWHM),
                                                             equal_nan=True)


def test_mf_and_ring_agree_with_a_given_map():
    img = _lobed(0.7, 300.0)
    rule = Sidelobes(rotation_deg=30.0)
    a = ring_snr(img, 0, [0.7], [300.0], PX, FWHM, rule)
    b = ring_snr(img, 0, [0.7], [300.0], PX, FWHM, rule, mfmap=mf_convolve(img, FWHM))
    assert a["snr"] == pytest.approx(b["snr"], rel=0, abs=0) and a["arcs"] == b["arcs"]
