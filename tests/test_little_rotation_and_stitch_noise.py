"""After a user's note of 2026-11-22: the stitch's displayed noise, the known companions in the
S/N maps, the k = 1 warning's note on rotation and the injection model of the generic reducer.

The reference-count guard with a pinned ``anglemax`` is tested in ``tests/test_feasibility.py``.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import klip_tpe.display as D
from klip_tpe.metrics import source_xy, star_center
from klip_tpe.products import snr_map
from klip_tpe.runner import Runner

EDGES = [20.0, 45.0, 70.0, 95.0, 120.0]
FW = 6.0


def _stitch(seed=1, factors=(1.35, 0.8, 1.25, 0.85)):
    """Return a stitch whose noise follows a power law in radius times a jump in each tile."""
    rng = np.random.default_rng(seed)
    n = 261
    yy, xx = np.indices((n, n))
    cx, cy = star_center((n, n))
    rr = np.hypot(xx - cx, yy - cy)
    tile = np.clip(np.searchsorted(EDGES, rr, side="right") - 1, 0, len(EDGES) - 2)
    sig = 3.0 * np.maximum(rr, 1.0) ** -0.6 * np.asarray(factors)[tile]
    img = rng.normal(size=(n, n)) * sig
    img[(rr < EDGES[0]) | (rr >= EDGES[-1])] = np.nan
    return img, rr


def _band_ratio(disp, rr, keep):
    """Return the displayed noise 1 FWHM outside each inner edge over 1 FWHM inside it."""
    out = []
    for e in EDGES[1:-1]:
        a = disp[keep & (rr >= e) & (rr < e + FW)]
        b = disp[keep & (rr >= e - FW) & (rr < e)]
        out.append(D._robust_sigma(a) / D._robust_sigma(b))
    return np.array(out)


def _ad(known=()):
    """Return the annulus data the stitched cells read: the FWHM, the plate scale and the companions."""
    return SimpleNamespace(known=list(known), pxscale=0.01, fwhm=FW, angle_convention="pa")


def test_the_radial_gain_makes_the_stitch_s_noise_fall_continuously():
    img, rr = _stitch()
    keep = np.isfinite(img)
    gain, prof_d = D._radial_gain(img, EDGES, FW)
    assert prof_d["index"] < 0
    assert np.isnan(gain[~keep]).all() and np.isfinite(gain[keep]).all()
    # the bands' ratio the fitted power law gives between their middles
    law = np.array([((e + FW / 2) / (e - FW / 2)) ** prof_d["index"] for e in EDGES[1:-1]])
    tile, _, _, _ = D._stitch_cells(img, img, EDGES, "tile", _ad())
    radial, _, kw, label = D._stitch_cells(img, img, EDGES, "radial", _ad())
    assert kw == {"vrange": (-1.0, 5.0)} and "radially scaled" in label
    assert np.abs(_band_ratio(radial, rr, keep) / law - 1).max() < 0.06
    assert _band_ratio(tile, rr, keep).max() > 1.15                 # the sawtooth
    # the displayed noise in 2 FWHM rings falls at every step
    r0 = np.arange(EDGES[0], EDGES[-1] - 2 * FW + 1e-9, 2 * FW)
    prof = [D._robust_sigma(radial[keep & (rr >= r) & (rr < r + 2 * FW)]) for r in r0]
    assert np.all(np.diff(prof) < 0), prof


def test_a_bright_companion_does_not_set_the_gain():
    img, rr = _stitch(seed=2)
    cx, cy = star_center(img.shape)
    known = [(80.0 * 0.01, 60.0)]
    (kx,), (ky,) = source_xy([known[0][0]], [known[0][1]], 0.01, cx, cy, "pa")
    yy, xx = np.indices(img.shape)
    lit = img + 40.0 * np.exp(-0.5 * ((xx - kx) ** 2 + (yy - ky) ** 2) / (FW / 2.3548) ** 2)
    g0, _ = D._radial_gain(img, EDGES, FW)
    g_out, _ = D._radial_gain(lit, EDGES, FW, [(kx, ky)])
    g_in, _ = D._radial_gain(lit, EDGES, FW)
    ring = np.isfinite(img) & (np.abs(rr - 80.0) < 2.0)
    assert np.abs(g_out[ring] / g0[ring] - 1).max() < 0.05
    assert np.abs(g_in[ring] / g0[ring] - 1).max() > np.abs(g_out[ring] / g0[ring] - 1).max()
    disp, _, _, _ = D._stitch_cells(lit, lit, EDGES, "radial", _ad(known))
    np.testing.assert_allclose(disp * D._robust_sigma(lit * g_out), lit * g_out, rtol=1e-12, equal_nan=True)


def test_the_stitch_scale_is_checked_and_set_for_a_run(monkeypatch, tmp_path):
    assert D.STITCH_SCALE in D.STITCH_SCALES and D.check_stitch_scale(None) is None
    assert D.check_stitch_scale("Tile") == "tile"
    with pytest.raises(ValueError):
        D.check_stitch_scale("sawtooth")
    img, _ = _stitch()
    c, i, kw, label = D._stitch_cells(img, None, None, "radial", _ad())          # no edges: one stretch
    assert c is img and i is None and kw == {} and label == "KLIP image stretch (sigma)"
    monkeypatch.setattr(D, "STITCH_SCALE", "radial")
    LiveDisplay = D.LiveDisplay
    try:
        LiveDisplay(str(tmp_path / "run"), save_png=False, movie=False, stitch_scale="tile")
    except Exception:                                    # the window or the run directory, not the setting
        pass
    assert D.STITCH_SCALE == "tile"
    from klip_tpe.panels import _Worker
    srv = SimpleNamespace(recipe=None, records=[], best={}, best_curve=None)
    _Worker._op_recipe(srv, {"kw": {"ia": 0, "stitch_scale": "one"}, "reset": True})
    assert D.STITCH_SCALE == "one" and srv.recipe == {"ia": 0}


def test_the_products_snr_map_leaves_the_known_companions_out_of_the_noise():
    rng = np.random.default_rng(5)
    n, px = 161, 0.01
    img = rng.normal(size=(n, n))
    cx, cy = star_center((n, n))
    known = (0.40, 120.0)
    (kx,), (ky,) = source_xy([known[0]], [known[1]], px, cx, cy, "pa")
    yy, xx = np.indices((n, n))
    img += 25.0 * np.exp(-0.5 * ((xx - kx) ** 2 + (yy - ky) ** 2) / (4.0 / 2.3548) ** 2)
    stub = SimpleNamespace(objective=SimpleNamespace(metric=SimpleNamespace(known=[known])),
                           sampler=SimpleNamespace(known=[known]), pxscale=px, fwhm=4.0,
                           reducer=SimpleNamespace(angle_convention="pa"))
    stub._verify_rule = lambda: None
    stub._pixel_mask = lambda: None
    for name in ("_known", "_known_xy", "_inj_xy", "_known_arcs"):
        setattr(stub, name, getattr(Runner, name).__get__(stub))
    got = Runner._snr_map(stub, img)
    want = snr_map(img, 4.0, exclude_xy=[(kx, ky)])
    plain = snr_map(img, 4.0)
    iy, ix = int(round(ky)), int(round(kx))
    assert got[iy, ix] == pytest.approx(want[iy, ix]) and got[iy, ix] > 1.3 * plain[iy, ix]
    # injected sources the caller leaves out stay left out, beside the companion
    got2 = Runner._snr_map(stub, img, exclude_xy=[(20.0, 30.0)])
    np.testing.assert_allclose(got2, snr_map(img, 4.0, exclude_xy=[(20.0, 30.0), (kx, ky)]), equal_nan=True)


def test_the_k1_warning_names_the_field_rotation():
    stub = SimpleNamespace(pxscale=0.0107, fwhm=8.97)
    stub._rotation_deg = lambda cfg: 12.66
    note = Runner._rotation_note(stub, None, 0.30, 0.50)
    assert note.startswith(" (the frames turn 12.7 deg") and "FWHM of arc at 0.40\"" in note
    arc = np.deg2rad(12.66) * 0.40 / 0.0107 / 8.97
    assert f"{arc:.1f} FWHM" in note
    stub._rotation_deg = lambda cfg: None
    assert Runner._rotation_note(stub, None, 0.30, 0.50) == ""
