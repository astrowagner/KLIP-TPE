"""Stitched images over the whole KLIP zones of the innermost and outermost annuli (MWC 758 ALES
session, 2026-11-07).

What must hold:

* ``open_ends`` takes the innermost tile in to the star and the outermost out past the frame,
  and keeps the boundaries between annuli.  ``legacy_stitch`` keeps the scored tiles.
* A one-annulus run with ``zone_in`` / ``zone_out``: the stitch, the injected stitch, the
  per-partition stitch and the running stitch cover the winner's whole zone, and each is the
  winner's image.  The S/N map reaches past the annulus.  The search's sources stay in the
  scored annulus.
* With two annuli each end reaches its annulus' zone, and inside the scored span the stitch is
  what it was.
"""
from __future__ import annotations

import json
import os
import types

import numpy as np
import pytest
from astropy.io import fits

from klip_tpe.metrics import star_center
from klip_tpe.runner import CalibrationConfig, Runner, ValidationConfig
from klip_tpe.stitch import open_ends, stitch_stacks, stitch_tiles, tile_index

from conftest import build_synthetic_run

QUIET = (lambda s: None)
PX = 0.05
EDGES = [12.0, 22.0]
Z_IN, Z_OUT = 3.0, 4.0


def _radius(shape):
    cx, cy = star_center(shape)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.hypot(xx - cx, yy - cy)


def _span(img):
    """The radii of an image's finite pixels."""
    rr = _radius(np.shape(img))[np.isfinite(img)]
    return float(rr.min()), float(rr.max())


def test_the_ends_open_and_the_boundaries_stay():
    tiles = [(8.0, 17.0), (17.0, 23.0), (23.0, 32.0)]
    assert open_ends(tiles) == [(0.0, 17.0), (17.0, 23.0), (23.0, float("inf"))]
    assert open_ends([(10.0, 20.0)]) == [(0.0, float("inf"))]
    assert open_ends([]) == []
    idx = tile_index((61, 61), open_ends([(8.0, 18.0), (18.0, 28.0)]))
    rr = _radius((61, 61))
    assert np.all(idx[rr < 18.0] == 0) and np.all(idx[rr >= 18.0] == 1)
    for legacy, want in ((True, tiles), (False, open_ends(tiles))):
        ns = types.SimpleNamespace(cfg=types.SimpleNamespace(legacy_stitch=legacy))
        assert Runner._image_tiles(ns, tiles) == want


def test_two_annuli_reach_their_zones_and_keep_the_span_between():
    shape = (81, 81)
    rr = _radius(shape)
    rng = np.random.default_rng(0)
    zone_px = [(5.0, 26.0), (15.0, 36.0)]           # (8, 20) and (20, 32), padded 2 px between, zone 3 / 4
    imgs = [np.where((rr >= a) & (rr <= b), rng.standard_normal(shape), np.nan) for a, b in zone_px]
    tiles = [(8.0, 20.0), (20.0, 32.0)]
    pads = [(0.0, 2.0), (2.0, 0.0)]
    old = stitch_tiles(imgs, tiles, [1.0, 2.0], pads=pads)
    new = stitch_tiles(imgs, open_ends(tiles), [1.0, 2.0], pads=pads)
    span = (rr >= 8.0) & (rr <= 32.0)
    assert np.array_equal(old[span], new[span], equal_nan=True), "inside the scored span: as before"
    assert _span(old) == pytest.approx((8.0, 32.0), abs=1.0)
    assert _span(new) == pytest.approx((5.0, 36.0), abs=1.0)
    cube, ids = stitch_stacks([im[None] for im in imgs], [["n1"], ["n1"]], open_ends(tiles), [1.0, 2.0], pads=pads)
    assert ids == ["n1"] and np.array_equal(np.isfinite(cube[0]), np.isfinite(new))


@pytest.fixture(scope="module")
def one(tmp_path_factory):
    """One annulus, 12-22 px, its KLIP zone 3 px further in and 4 px further out."""
    d = str(tmp_path_factory.mktemp("openends"))
    red, space, obj, samp, cfg = build_synthetic_run(
        ann_edges=list(EDGES), n_iter=[5], n_init=3, param_verify=False, stitch_every=0, verify=True,
        verify_n_boot=5, write_setup_files=False, fm_curve=False, inject_rho=17.0 * PX,
        validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[2e-3]))
    space.fixed.update(zone_in=Z_IN, zone_out=Z_OUT)
    r = Runner(red, space, obj, samp, cfg, d, log=QUIET)
    results = r.run()
    return d, r, results


@pytest.mark.slow
def test_a_one_annulus_stitch_is_its_winner_image(one):
    d, r, results = one
    best = fits.getdata(os.path.join(d, "annulus01", "best_clean.fits")).astype(float)
    lo, hi = _span(best)
    assert lo == pytest.approx(EDGES[0] - Z_IN, abs=1.0) and hi == pytest.approx(EDGES[1] + Z_OUT, abs=1.0)
    for stitched, own in (("klip_stitched.fits", "best_clean.fits"), ("klip_stitched_inj.fits", "best_inj.fits"),
                          ("klip_stitched_running.fits", "best_clean.fits")):
        st = fits.getdata(os.path.join(d, stitched)).astype(float)
        im = fits.getdata(os.path.join(d, "annulus01", own)).astype(float)
        assert np.array_equal(np.isfinite(st), np.isfinite(im)), stitched
        f = np.isfinite(im)
        assert np.allclose(st[f], im[f], rtol=1e-6, atol=1e-9), stitched
    cube = fits.getdata(os.path.join(d, "klip_stitched_nights.fits.gz")).astype(float)
    parts = fits.getdata(os.path.join(d, "annulus01", "best_clean_partitions.fits")).astype(float)
    assert cube.shape[0] == parts.shape[0]
    assert all(np.array_equal(np.isfinite(c), np.isfinite(p)) for c, p in zip(cube, parts))
    snr = fits.getdata(os.path.join(d, "klip_stitched_snr.fits")).astype(float)
    s_lo, s_hi = _span(snr)
    assert s_lo < EDGES[0] - 1.0 and s_hi > EDGES[1] + 1.0, "the S/N map reaches past the annulus"


@pytest.mark.slow
def test_the_search_stays_in_the_scored_annulus(one):
    d, r, results = one
    rows = [json.loads(ln) for ln in open(os.path.join(d, "results.jsonl")) if ln.strip()]
    rho = [s[0] / PX for row in rows for s in (row.get("sources") or [])]
    assert rho and min(rho) >= EDGES[0] and max(rho) <= EDGES[1]
    assert (results[0].inrad, results[0].outrad) == tuple(EDGES)
    hdr = fits.getheader(os.path.join(d, "klip_stitched.fits"))
    assert hdr["EDGES"] == f"{EDGES[0]:.1f},{EDGES[1]:.1f}", "the header keeps the scored edges"


@pytest.mark.slow
def test_rereduce_gives_a_finished_run_the_whole_zone(one, tmp_path):
    d, r, results = one
    red, space, obj, samp, _ = build_synthetic_run()
    rr = Runner.resume(d, red, obj, samp, log=QUIET, same_frames=False)
    rr.rereduce(str(tmp_path / "re"), log=QUIET, verify=False)
    best = fits.getdata(os.path.join(d, "annulus01", "best_clean.fits")).astype(float)
    st = fits.getdata(os.path.join(str(tmp_path / "re"), "klip_stitched.fits")).astype(float)
    assert np.array_equal(np.isfinite(st), np.isfinite(best))
    assert _span(st) == pytest.approx((EDGES[0] - Z_IN, EDGES[1] + Z_OUT), abs=1.0)
