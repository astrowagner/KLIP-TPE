"""Every ring and map leaves the run's pixel mask (the disk) out, and the verify, the candidate
search and the display's companion S/N measure with the search's rules (Kevin Wagner,
2026-11-01).

The field is white noise, a source at 0.8" and a bright lumpy arc across the same ring on the
other side, which the mask covers.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from klip_tpe import candidates as C
from klip_tpe import display as D
from klip_tpe import verify as V
from klip_tpe.metrics import MawetPeakSNR, aperture_mawet_snr, mawet_peak_snr, source_xy, star_center
from klip_tpe.products import contrast_curve, noise_profile, snr_map

from conftest import build_synthetic_run

PX, FWHM, N = 0.05, 4.0, 81
RHO, PA = 0.8, 40.0
KEEP_SPIKES = {"declip_nsig": None}


def _field(seed=2, amp=12.0):
    rng = np.random.default_rng(seed)
    img = rng.standard_normal((N, N))
    cx, cy = star_center(img.shape)
    yy, xx = np.mgrid[0:N, 0:N]
    ax, ay = source_xy(np.full(101, RHO), np.linspace(170.0, 270.0, 101), PX, cx, cy, "pa")
    arc = np.zeros((N, N), bool)
    for x, y in zip(ax, ay):
        arc |= np.hypot(xx - x, yy - y) <= 2.5
    img[arc] += amp * rng.random(int(arc.sum()))
    x, y = source_xy([RHO], [PA], PX, cx, cy, "pa")
    img += 10.0 * np.exp(-0.5 * ((xx - x[0]) ** 2 + (yy - y[0]) ** 2) / (FWHM / 2.3548) ** 2)
    return img, arc, (float(x[0]), float(y[0]))


def test_the_verify_leaves_the_mask_out():
    img, arc, _ = _field()
    s0 = V.snr_fap(img, RHO, PA, PX, FWHM, rules=KEEP_SPIKES)
    s1 = V.snr_fap(img, RHO, PA, PX, FWHM, rules=dict(KEEP_SPIKES, pixel_mask=arc))
    assert s1["nclean"] < s0["nclean"]
    assert s1["snr"] > s0["snr"], "the arc no longer counts as noise"
    ref = mawet_peak_snr(img, [RHO], [PA], PX, FWHM, declip_nsig=None, pixel_mask=arc)[0]
    assert s1["snr"] == pytest.approx(float(ref), rel=1e-12)


def test_the_stim_annulus_leaves_the_mask_out():
    img, arc, (x, y) = _field(amp=40.0)
    v0, f0 = V.stim_fap(img, x, y, RHO / PX, FWHM)
    v1, f1 = V.stim_fap(img, x, y, RHO / PX, FWHM, pixel_mask=arc)
    assert v1 == v0 and f1 < f0, (f0, f1)


def test_verify_candidates_takes_the_mask_everywhere():
    img, arc, _ = _field()
    stack = np.stack([img + np.random.default_rng(s).standard_normal(img.shape) for s in range(4)])
    kw = dict(n_boot=0, return_maps=True)
    r0 = V.verify_candidates(stack, None, [(RHO, PA)], FWHM, PX, rules=KEEP_SPIKES, **kw)[0][0]
    r1 = V.verify_candidates(stack, None, [(RHO, PA)], FWHM, PX, rules=dict(KEEP_SPIKES, pixel_mask=arc), **kw)[0][0]
    assert r1["snr"] > r0["snr"]
    assert r1["stim_fap"] <= r0["stim_fap"]


def test_the_maps_and_curves_leave_the_mask_out():
    img, arc, (x, y) = _field()
    ix, iy = int(round(x)), int(round(y))
    m0, m1 = snr_map(img, FWHM), snr_map(img, FWHM, pixel_mask=arc)
    assert m1[iy, ix] > m0[iy, ix]
    s0, r0 = noise_profile(img, FWHM, 10, 22)
    s1, r1 = noise_profile(img, FWHM, 10, 22, pixel_mask=arc)
    k = int(np.argmin(np.abs(r0 - RHO / PX)))
    assert s1[k] < s0[k]
    c0 = contrast_curve(img, [RHO], [8.0], 1e-4, FWHM, PX, 10, 22)
    c1 = contrast_curve(img, [RHO], [8.0], 1e-4, FWHM, PX, 10, 22, pixel_mask=arc)
    assert not np.allclose(np.asarray(c0["sigma"], float), np.asarray(c1["sigma"], float), equal_nan=True)


def test_the_grid_s_aperture_ring_leaves_the_mask_out():
    img, arc, _ = _field()
    s0, d0 = aperture_mawet_snr(img, [RHO], [PA], PX, FWHM, return_details=True)
    s1, d1 = aperture_mawet_snr(img, [RHO], [PA], PX, FWHM, return_details=True, pixel_mask=arc)
    assert d1[0]["nclean"] < d0[0]["nclean"] and s1[0] > s0[0]


def test_the_candidate_search_measures_as_the_search():
    img, arc, _ = _field()
    rules = dict(KEEP_SPIKES, pixel_mask=arc)
    rows = C.find_candidates(img, img[None], FWHM, PX, rules=rules, snrmin=4.0)
    near = [r for r in rows if np.hypot(r["rho"] - RHO, (r["theta"] - PA) * np.pi / 180 * RHO) < 2 * PX]
    assert near, [(r["rho"], r["theta"]) for r in rows]
    top = near[0]
    want = V.snr_fap(img, top["rho"], top["theta"], PX, FWHM, rules=rules)["snr"]
    assert top["snr"] == pytest.approx(want, rel=1e-12)
    assert top["snr"] > V.snr_fap(img, top["rho"], top["theta"], PX, FWHM, rules=KEEP_SPIKES)["snr"]


def test_the_runner_hands_over_its_rules_and_saves_its_mask(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run()
    size = red.reducers["n1"].size
    mask = np.zeros((size, size), bool)
    mask[10:20, 10:20] = True
    obj.metric.pixel_mask = mask
    obj.metric.declip_nsig = None
    from klip_tpe.runner import Runner
    run = Runner(red, space, obj, samp, cfg, str(tmp_path), log=lambda s: None)
    rules = run._ring_rules()
    assert rules["pixel_mask"] is mask and rules["declip_nsig"] is None and callable(rules["kernel_at"])
    run.write_setup()
    assert os.path.exists(tmp_path / "pixel_mask.fits")
    # the display reads the rules back from the setup, the despike left off included
    with open(tmp_path / "run_setup.json") as f:
        setup = json.load(f)
    pm = D._run_pixel_mask(str(tmp_path))
    assert np.array_equal(pm, mask)
    back = D._run_rules(setup, pm)
    assert "declip_nsig" in back and back["declip_nsig"] is None
    assert all(back[k] == rules[k] for k in ("search_px", "excl_fwhm", "band_fwhm", "min_ring", "penalty"))


def test_the_display_measures_the_companions_with_the_run_s_rules():
    img, arc, _ = _field()
    setup = {"objective": {"metric": {"name": "mawet_peak", "excl_fwhm": 1.5, "search_px": 1.5,
                                      "declip_nsig": None, "band_fwhm": 0.8, "min_ring": 7,
                                      "penalty": "geometric", "inner_fwhm": 1.0, "mf_fwhm": 4.5}}}
    rules = D._run_rules(setup, arc)
    ad = D.annulus_from_records([], {}, {}, 0, PX, FWHM, known=[(RHO, PA)], pixel_mask=arc, rules=rules)
    got = D.known_snr(ad, img)
    want = MawetPeakSNR(pxscale=PX, fwhm=FWHM, known=[(RHO, PA)], declip_nsig=None, band_fwhm=0.8, min_ring=7,
                        mf_fwhm=4.5, pixel_mask=arc).per_source(img, None, [RHO], [PA])
    assert got[0] == pytest.approx(float(want[0]), rel=1e-12)
    # without the rules, the metric's defaults and the mask alone
    ad0 = D.annulus_from_records([], {}, {}, 0, PX, FWHM, known=[(RHO, PA)], pixel_mask=arc)
    plain = MawetPeakSNR(pxscale=PX, fwhm=FWHM, known=[(RHO, PA)], pixel_mask=arc).per_source(img, None, [RHO], [PA])
    assert D.known_snr(ad0, img)[0] == pytest.approx(float(plain[0]), rel=1e-12)


def test_a_forward_model_metric_hands_over_the_kernel_it_falls_back_to():
    """The verify has no forward model, so an FMMF run's rings take the kernel the metric
    falls back to: its measured one, else the Gaussian of ``mf_fwhm``."""
    from klip_tpe.fmmf import FMMFSNR
    from klip_tpe.metrics import gaussian_kernel, ring_rules
    assert "kernel_at" not in ring_rules(FMMFSNR(pxscale=PX, fwhm=FWHM))
    k = ring_rules(FMMFSNR(pxscale=PX, fwhm=FWHM, mf_fwhm=5.0))["kernel_at"](RHO)
    assert np.array_equal(k, gaussian_kernel(5.0))
    box = np.ones((3, 3)) / 9.0
    assert ring_rules(FMMFSNR(pxscale=PX, fwhm=FWHM, kernel_fn=lambda r: box))["kernel_at"](RHO) is box


def test_the_s_n_map_widens_a_band_the_mask_starves():
    """A mask over whole rings left the S/N map blank there (dd433e7).  The band of such a
    radius widens past the mask until it holds products.SNR_BAND_MIN pixels, and the map says
    which radii borrowed (Kevin Wagner, 2026-11-01)."""
    from klip_tpe.products import snr_widened_header, snr_widened_note
    rng = np.random.default_rng(1)
    img = rng.standard_normal((N, N))
    cx, cy = star_center(img.shape)
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.hypot(xx - cx, yy - cy)
    img[rr < 4] = np.nan
    mask = (rr > 10.5) & (rr < 15.5)
    mask |= (rr > 15.5) & (rr < 17.5) & (xx < cx)
    info = {}
    m = snr_map(img, FWHM, pixel_mask=mask, info=info)

    def ring(r):
        return (np.abs(rr - r) <= 0.5) & np.isfinite(img)

    assert all(np.isfinite(m[ring(r)]).all() for r in range(11, 18))
    assert set(range(11, 16)) <= set(info["widened"]) and not {16, 17} & set(info["widened"])
    plain = snr_map(img, FWHM)
    for r in (6, 8, 25):                          # bands the mask leaves alone are as before
        assert np.array_equal(m[ring(r)], plain[ring(r)])
    info0 = {}
    assert np.array_equal(snr_map(img, FWHM, info=info0), plain, equal_nan=True) and info0["widened"] == {}
    hdr = snr_widened_header(info)
    assert hdr["SNRWIDEN"] == len(info["widened"]) and hdr["SNRWIDR"].startswith("11-")
    assert snr_widened_note(info, PX).startswith("noise band widened past the mask at 0.55-")
    # the display's S/N cell carries the note
    ad = D.annulus_from_records([], {}, {}, 0, PX, FWHM, pixel_mask=mask)
    got = D._snr_map(img, ad)
    assert np.array_equal(np.asarray(got), m, equal_nan=True) and "wider noise band" in got.snr_note
    assert getattr(D._snr_map(img, D.annulus_from_records([], {}, {}, 0, PX, FWHM)), "snr_note", "") == ""
