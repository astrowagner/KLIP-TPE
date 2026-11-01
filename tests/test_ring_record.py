"""The record of the noise rings (``metrics.RingRecorder``, ``klip_tpe.ringplot``,
``scripts/plot_noise_rings.py``): every ring an S/N is measured on, with what became of each
aperture, the phase it came from, and no S/N changed by recording it.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from klip_tpe.metrics import (MawetPeakSNR, Objective, RingRecorder, Sidelobes, Source, aperture_mawet_snr,
                              mawet_peak_snr, sidelobe_snr, source_xy, star_center)
from klip_tpe.products import snr_map
from klip_tpe.ringplot import phase_from_stack, pick_records, plot_rings
from klip_tpe.runner import Runner
from klip_tpe.verify import snr_fap, stim_fap

from conftest import build_synthetic_run

PX, FWHM = 0.05, 4.0


def _image(seed=3, n=81):
    rng = np.random.default_rng(seed)
    img = rng.standard_normal((n, n))
    cx, cy = star_center(img.shape)
    yy, xx = np.mgrid[0:n, 0:n]
    for rho, th, amp in ((0.9, 30.0, 6.0), (0.9, 210.0, 6.0)):
        x, y = source_xy([rho], [th], PX, cx, cy, "pa")
        img += amp * np.exp(-0.5 * ((xx - x[0]) ** 2 + (yy - y[0]) ** 2) / (FWHM / 2.355) ** 2)
    return img


def _same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def test_recording_changes_no_snr():
    img, clean = _image(), _image(seed=4)
    mask = np.zeros(img.shape, bool)
    mask[60:70, 20:40] = True
    kw = dict(known=[(0.9, 120.0)], pixel_mask=mask, return_details=True)
    rule = Sidelobes(rotation_deg=150.0, mode="farthest")
    calls = [lambda: mawet_peak_snr(img, [0.9, 0.9], [30.0, 210.0], PX, FWHM, **kw),
             lambda: sidelobe_snr(img, [0.9, 0.9], [30.0, 210.0], PX, FWHM, rule, known=[(0.9, 120.0)]),
             lambda: sidelobe_snr(img, [0.9], [30.0], PX, FWHM, Sidelobes(rotation_deg=170.0, mode="clean"),
                                  img_clean=clean),
             lambda: aperture_mawet_snr(img, [0.9], [30.0], PX, FWHM, known=[(0.9, 120.0)], return_details=True),
             lambda: snr_fap(img, 0.9, 30.0, PX, FWHM, exclude_xy=[(10.0, 10.0)]),
             lambda: stim_fap(img, 50.0, 50.0, 18.0, FWHM),
             lambda: snr_map(img, FWHM, exclude_xy=[(50.0, 50.0)])]
    for call in calls:
        plain = call()
        with RingRecorder() as rec:
            again = call()
        assert rec.records, call
        if isinstance(plain, tuple) and isinstance(plain[1], list):
            assert _same(plain[0], again[0]) and plain[1] == again[1], "the S/N and its details, to the bit"
            assert not any("_ring" in d for d in again[1]), "no record in the details"
        elif isinstance(plain, dict):
            assert plain == again or all(_same(plain[k], again[k]) for k in plain)
        else:
            assert _same(plain, again)


def test_each_aperture_says_what_became_of_it():
    img = _image()
    mask = np.zeros(img.shape, bool)
    cx, cy = star_center(img.shape)
    mx, my = source_xy([0.9], [300.0], PX, cx, cy, "pa")
    mask[int(my[0]) - 3:int(my[0]) + 4, int(mx[0]) - 3:int(mx[0]) + 4] = True
    with RingRecorder() as rec:
        snr, det = mawet_peak_snr(img, [0.9, 0.9], [30.0, 210.0], PX, FWHM, known=[(0.9, 120.0)], pixel_mask=mask,
                                  return_details=True)
    assert len(rec.records) == 2, "one record a source"
    r0 = rec.records[0]
    what = [a[2] for a in r0["ring"]]
    assert {"used", "source", "known", "masked"} <= set(what), what
    assert what.count("used") == det[0]["nclean"] and r0["noise"] == "ring"
    assert (r0["x"], r0["y"], r0["r"]) == (det[0]["x"], det[0]["y"], det[0]["r"])
    assert len(r0["ring"]) == det[0]["nap"] - 1 and r0["snr"] == snr[0]
    assert len(r0["sources_xy"]) == 2 and len(r0["known_xy"]) == 1


def test_the_rule_records_the_attempt_it_kept():
    img = _image()
    rule = Sidelobes(rotation_deg=170.0, mode="farthest")
    with RingRecorder() as rec:
        snr, det = sidelobe_snr(img, [0.9], [30.0], PX, FWHM, rule)
    assert len(rec.records) == 1, "the attempts it dropped are not recorded"
    r, d = rec.records[0], det[0]
    assert r["fallback"] == d["fallback"] == "farthest" and r["rule"] == "farthest"
    assert [a[2] for a in r["ring"]].count("used") == d["nclean"] >= 4
    assert "arc" in [a[2] for a in r["ring"]] and r["arc_deg"] == d["arc_deg"] < 170.0
    # a pair 180 deg apart: each one's arc covers the other's ring, and the fallback fails
    with RingRecorder() as rec:
        snr, det = sidelobe_snr(img, [0.9, 0.9], [30.0, 210.0], PX, FWHM, rule)
    assert [x["fallback_tried"] for x in rec.records] == ["farthest", "farthest"]
    assert all(len(x["all_sources_xy"]) == 2 for x in rec.records)


def test_the_objectives_clean_term_is_marked():
    img, clean = _image(), _image(seed=4)
    obj = Objective(MawetPeakSNR(pxscale=PX, fwhm=FWHM), clean_subtract=True)
    with RingRecorder() as rec:
        obj.score_search(img, [Source(0.9, 30.0, 1e-3), Source(0.9, 210.0, 1e-3)], clean)
    assert [r["role"] for r in rec.records] == [None, None, "clean", "clean"]


def test_a_band_fallback_marks_the_ring_unused():
    img = _image()
    with RingRecorder() as rec:
        _, det = mawet_peak_snr(img, [0.9], [30.0], PX, FWHM, min_ring=50, return_details=True)
    r = rec.records[0]
    assert not det[0]["ring_used"] and r["noise"] == "band"
    assert "unused" in [a[2] for a in r["ring"]] and "used" not in [a[2] for a in r["ring"]]


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("rings"))
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=[5], n_init=3, n_remeasure=2)
    keep = Runner.SCORE_EARLY
    Runner.SCORE_EARLY = False
    try:
        with RingRecorder(phase=phase_from_stack) as rec:
            Runner(red, space, obj, samp, cfg, d, log=lambda s: None).run()
    finally:
        Runner.SCORE_EARLY = keep
    return d, rec.records


@pytest.mark.slow
def test_a_run_names_every_rings_phase(recorded_run):
    d, recs = recorded_run
    phases = {r["phase"] for r in recs}
    assert {"calibration", "search", "validation"} <= phases, phases
    assert None not in phases, [r["fn"] for r in recs if r["phase"] is None][:5]
    search = [r for r in recs if r["phase"] == "search"]
    assert {r["role"] for r in search} == {None, "clean"}, "the injected and the clean image"
    assert all(len(r["sources_xy"]) == len(search[0]["sources_xy"]) for r in search)


@pytest.mark.slow
def test_the_diagram_has_a_panel_per_phase(recorded_run, tmp_path):
    d, recs = recorded_run
    picks = pick_records(recs)
    labels = [p["_label"] for p in picks]
    assert labels[:2] == ["calibration", "calibration, clean image"] and "validation" in labels
    assert "source count check" not in labels
    path = plot_rings(recs, str(tmp_path / "rings.png"), "synthetic", pxscale=PX, zones=[(8, 30)])
    assert os.path.getsize(path) > 10000
