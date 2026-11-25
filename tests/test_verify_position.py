"""Where the verify reads a source, and the FAP of a candidate a blind search found (the audit of 2026-11-25).

A source whose position is given, a known companion or an injection, is read at it, so its small-sample
t FAP holds.  A candidate the blind search found is read at the matched filter's peak within FWHM/2,
and its FAP over the field comes from random-field theory with the S/N map's measured smoothness.  A
run saved before keeps the 1.5 px search and ``1 - (1 - p)^N_resel`` (``RunConfig.verify_position``).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage
from scipy.stats import norm

from klip_tpe import RunConfig
from klip_tpe.candidates import candidate_metrics, detect_peaks, find_candidates
from klip_tpe.metrics import VERIFY_IDL, source_xy, star_center
from klip_tpe.products import snr_map
from klip_tpe.verify import (LEGACY_SEARCH_PX, field_fap, field_geometry, field_smoothness, legacy_position,
                             search_px, snr_fap, verify_candidates)

PX, FW = 0.05, 4.0


def test_the_search_radius_follows_where_the_position_came_from():
    assert search_px(FW) == 0.0
    assert search_px(FW, found=True) == pytest.approx(FW / 2)
    assert search_px(FW, found=True, rules={"position": "given"}) == pytest.approx(FW / 2)
    for legacy in ({"position": "peak"}, dict(VERIFY_IDL)):
        assert legacy_position(legacy)
        assert search_px(FW, rules=legacy) == search_px(FW, True, legacy) == LEGACY_SEARCH_PX
    assert not legacy_position(None) and not legacy_position({"position": "given"})
    assert search_px(FW, True, {"position": "given"}, drift_px=0.7) == 0.7      # the caller's radius wins


def test_the_field_fap_is_the_expected_euler_characteristic():
    area, perim, fe = 9956.0, 2 * np.pi * (56.5 + 4.8), np.sqrt(2) * FW
    c = 4 * np.log(2)
    for p in (1e-3, 1e-4, 2.87e-7):
        u = norm.isf(p)
        ec = 0.5 * perim / fe * np.sqrt(c) / (2 * np.pi) * np.exp(-u * u / 2) \
            + area / fe ** 2 * c / (2 * np.pi) ** 1.5 * u * np.exp(-u * u / 2)
        assert field_fap(p, area, perim, fe) == pytest.approx(1 - np.exp(-ec), rel=1e-10)
    # at least the local FAP, at most 1, rising with p, and a disk adds its Euler characteristic
    vals = [field_fap(p, area, perim, fe) for p in (1e-9, 1e-7, 1e-5, 1e-3, 0.1)]
    assert all(np.diff(vals) > 0) and vals[-1] <= 1.0
    assert field_fap(1e-9, 1.0, 0.0, fe) == pytest.approx(1e-9)
    assert field_fap(0.6, area, perim, fe) == 1.0 and field_fap(0.0, area, perim, fe) == 0.0
    assert np.isnan(field_fap(np.nan, area, perim, fe)) and np.isnan(field_fap(1e-3, area, perim, 0.0))
    assert field_fap(1e-6, area, perim, fe, euler=1.0) > field_fap(1e-6, area, perim, fe)
    # far above the IDL predecessor's count at the operating FAP: about 10 times for this field
    nresel = np.pi * (56.5 ** 2 - 4.8 ** 2) / FW ** 2
    assert field_fap(2.87e-7, area, perim, fe) > 5 * (1 - (1 - 2.87e-7) ** nresel)


@pytest.mark.parametrize("s", [2.5, 4.0, 6.0])
def test_the_smoothness_of_smoothed_white_noise_is_its_kernel(s):
    rng = np.random.default_rng(int(10 * s))
    z = ndimage.gaussian_filter(rng.normal(size=(201, 201)), s / 2.3548)
    assert field_smoothness(z[20:-20, 20:-20]) == pytest.approx(s, rel=0.08)


def test_the_s_n_map_of_psf_correlated_noise_is_sqrt2_fwhm_smooth():
    rng = np.random.default_rng(4)
    img = ndimage.gaussian_filter(rng.normal(size=(161, 161)), FW / 2.3548)
    sm = snr_map(img, FW)
    fld = field_geometry(sm, 1.2 * FW, 80 - FW, FW)
    assert fld["fwhm_eff_from"] == "map"
    assert fld["fwhm_eff_px"] == pytest.approx(np.sqrt(2) * FW, rel=0.12)
    assert fld["area_px"] == pytest.approx(np.pi * ((80 - FW) ** 2 - (1.2 * FW) ** 2), rel=0.02)
    # too few pixels: the fallback
    tiny = field_geometry(sm, 10.0, 11.0, FW)
    assert tiny["fwhm_eff_from"] == "psf" and tiny["fwhm_eff_px"] == pytest.approx(np.sqrt(2) * FW)


def _offset_source(n=121, rho=0.9, pa=40.0, dx=1.0, amp=12.0, seed=7):
    """An image of correlated noise with a source 1 px from the position given as (rho, pa)."""
    rng = np.random.default_rng(seed)
    img = ndimage.gaussian_filter(rng.normal(size=(n, n)), FW / 2.3548)
    img /= img.std()
    cx, cy = star_center((n, n))
    (x,), (y,) = source_xy([rho], [pa], PX, cx, cy, "pa")
    yy, xx = np.mgrid[0:n, 0:n]
    img += amp * np.exp(-0.5 * ((xx - x - dx) ** 2 + (yy - y) ** 2) / (FW / 2.3548) ** 2)
    return img, (rho, pa)


def test_a_given_position_is_read_at_itself_and_a_found_one_at_its_peak():
    img, (rho, pa) = _offset_source()
    at = snr_fap(img, rho, pa, PX, FW)
    pk = snr_fap(img, rho, pa, PX, FW, srad=FW / 2)
    assert pk["snr"] > at["snr"]
    rows = verify_candidates(img, None, [(rho, pa)], FW, PX, n_boot=0)
    r = rows[0]
    assert r["position"] == "given" and r["search_px"] == 0.0
    assert r["snr"] == pytest.approx(at["snr"]) and r["fap"] == pytest.approx(at["fap"])
    assert r["snr_peak"] == pytest.approx(pk["snr"])
    found = verify_candidates(img, None, [(rho, pa)], FW, PX, n_boot=0, found=True,
                              field={"area_px": 9000.0, "perimeter_px": 400.0, "fwhm_eff_px": 5.66})[0]
    assert found["position"] == "found" and found["search_px"] == pytest.approx(FW / 2)
    assert found["snr"] == pytest.approx(pk["snr"]) and "snr_peak" not in found
    assert found["fap_field"] == pytest.approx(field_fap(found["fap"], 9000.0, 400.0, 5.66))
    # a run saved before 2026-11-25 keeps its 1.5 px search for every source
    old = verify_candidates(img, None, [(rho, pa)], FW, PX, n_boot=0, rules={"position": "peak"})[0]
    assert old["search_px"] == LEGACY_SEARCH_PX
    assert old["snr"] == pytest.approx(snr_fap(img, rho, pa, PX, FW, srad=1.5, rules={"position": "peak"})["snr"])


def test_injections_are_read_at_their_positions():
    img, (rho, pa) = _offset_source(dx=0.0)
    clean, _ = _offset_source(dx=0.0, amp=0.0)
    rows = verify_candidates(clean, None, [], FW, PX, n_boot=0, injected=(img[None], [(rho, pa, 1e-3)]))
    inj = next(r for r in rows if r.get("kind") == "injection")
    assert inj["snr"] == pytest.approx(snr_fap(img, rho, pa, PX, FW)["snr"])


def test_a_run_records_where_its_verify_reads_and_an_old_run_keeps_its_peak():
    assert RunConfig().verify_position == "given"
    d = RunConfig(ann_edges=[8, 20]).to_dict()
    assert RunConfig.from_dict(d).verify_position == "given"
    d.pop("verify_position")
    assert RunConfig.from_dict(d).verify_position == "peak"


def test_the_blind_search_corrects_its_peaks_over_the_field():
    rng = np.random.default_rng(12)
    img = ndimage.gaussian_filter(rng.normal(size=(121, 121)), FW / 2.3548)
    img /= img.std()
    iwa, owa = 1.2 * FW, 60.5 - FW
    sm = snr_map(img, FW)
    px, py, ps = detect_peaks(sm, 1.0, max(0.9 * FW, 2.0), iwa, owa, nmax=5)
    fld = field_geometry(sm, iwa, owa, FW)
    new = candidate_metrics(px, py, ps, img[None], FW, PX, iwa, owa, field=fld)
    old = candidate_metrics(px, py, ps, img[None], FW, PX, iwa, owa, rules={"position": "peak"})
    nresel = np.pi * (owa ** 2 - iwa ** 2) / FW ** 2
    for a in new:
        assert a["fap"] == pytest.approx(field_fap(a["fap_resel"], fld["area_px"], fld["perimeter_px"],
                                                   fld["fwhm_eff_px"]))
        assert a["fap"] >= a["fap_resel"]
    for b in old:
        assert b["fap"] == pytest.approx(min(1 - (1 - b["fap_resel"]) ** nresel, 1.0))
    rows = find_candidates(img, img[None], FW, PX, snrmin=2.5)
    assert rows and rows[0]["meta"]["look_elsewhere"] == "random field"
    assert rows[0]["meta"]["field"]["fwhm_eff_from"] == "map" and rows[0]["meta"]["search_px"] == FW / 2


@pytest.mark.slow
def test_the_field_fap_holds_on_source_free_fields():
    """600 fields of noise correlated at the PSF: the field FAP is reached about as often as it says."""
    rng = np.random.default_rng(31)
    n, iwa = 121, 1.2 * FW
    owa = n / 2.0 - FW
    g = []
    for _ in range(600):
        img = ndimage.gaussian_filter(rng.normal(size=(n, n)), FW / 2.3548)
        img /= img.std()
        sm = snr_map(img, FW)
        px, py, ps = detect_peaks(sm, 1.0, max(0.9 * FW, 2.0), iwa, owa, nmax=6)
        fld = field_geometry(sm, iwa, owa, FW)
        cm = candidate_metrics(px, py, ps, img[None], FW, PX, iwa, owa, field=fld)
        g.append(min(c["fap"] for c in cm))
    g = np.asarray(g)
    for alpha in (0.5, 0.2):
        assert abs(np.mean(g < alpha) / alpha - 1.0) < 0.25
