"""The final stitch with overlapping annuli, and the per-tile measure (NEAR2 s7 note,
2026-11-02).

What must hold:

* Each annulus' injected image in the final stitch holds every annulus' validation sources
  (``RunConfig.stitch_injections = "all"``, new runs): the run reduces each winner once more at
  the end, the injected stitch says so (``INJALL``), and the annuli' own ``best_inj.fits`` stay
  as the validation wrote them.  A source of one annulus that lies in the next annulus' tile is
  then in the stitch, where before only its own annulus' image held it.  A run saved before
  keeps each annulus' own (``"own"``), and so does its ``rereduce`` unless told otherwise.
* Where annuli overlap the tiles meet sharply (``stitch_overlap_pads`` False), where they only
  meet the pads stay, and a run saved before keeps its pads.
* The per-tile measure: each source and candidate measured on the stitch with the tile of the
  annulus that holds it put in (``tile_verify_report.txt``, since 2026-11-04), a source that
  annulus' image does not hold listed without an S/N, the stitched report closing with both S/N
  side by side, equal where the tiles take no blend, and the tile-wise S/N maps, on each tile the
  map of the stitch with that tile put in.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from astropy.io import fits

from klip_tpe import verify as V
from klip_tpe.metrics import source_xy, star_center
from klip_tpe.runner import CalibrationConfig, RunConfig, Runner, ValidationConfig
from klip_tpe.stitch import put_tile_in, stitch_by_tiles, tile_index, tile_of

from conftest import build_synthetic_run

QUIET = (lambda s: None)
PX = 0.05                                 # the synthetic reducer's plate scale
RHO_PX = 19.0                             # every source here: inside both zones, in the second tile


# -------------------------------------------------------------------------- pure helpers
def test_new_runs_take_the_new_stitch_and_old_runs_keep_theirs():
    cfg = RunConfig(ann_edges=[8, 24], ann_extra=[[12, 28]])
    assert cfg.stitch_injections == "all" and cfg.stitch_overlap_pads is False
    d = cfg.to_dict()
    assert RunConfig.from_dict(d).stitch_injections == "all"
    for k in ("stitch_injections", "stitch_overlap_pads"):
        d.pop(k)
    old = RunConfig.from_dict(d)
    assert old.stitch_injections == "own" and old.stitch_overlap_pads is True, "a run saved before keeps its stitch"


def test_which_tile_holds_a_radius_and_a_pixel():
    tiles = [(8.0, 18.0), (18.0, 28.0)]
    assert [tile_of(r, tiles) for r in (7.9, 8.0, 17.99, 18.0, 27.9, 28.0, 28.1)] == [None, 0, 0, 1, 1, 1, None]
    idx = tile_index((61, 61), tiles)
    yy, xx = np.mgrid[0:61, 0:61]
    rr = np.hypot(xx - 30.0, yy - 30.0)
    assert np.all(idx[(rr >= 8) & (rr < 18)] == 0) and np.all(idx[(rr >= 18) & (rr < 28)] == 1)
    assert np.all(idx[(rr < 8) | (rr > 28)] == -1)
    a, b = np.full((61, 61), 1.0), np.full((61, 61), 2.0)
    st = stitch_by_tiles([a, b], tiles)
    assert set(np.unique(st[np.isfinite(st)])) == {1.0, 2.0}, "no blend at the boundary"
    assert np.all(st[idx == 0] == 1.0) and np.all(st[idx == 1] == 2.0) and np.all(np.isnan(st[idx < 0]))


def _runner(tmp_path, **kw):
    base = dict(ann_edges=[8, 20, 32], ann_extra=[[14, 26]], n_iter=[6], n_init=3, param_verify=False,
                stitch_every=0, calibration=CalibrationConfig(recal_budget=0),
                validation=ValidationConfig(n_top=1, n_valid=1))
    base.update(kw)
    red, space, obj, samp, cfg = build_synthetic_run(**base)
    return Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)


def test_no_pads_where_annuli_overlap_and_pads_where_they_meet(tmp_path):
    r = _runner(tmp_path / "a")
    zones = r.cfg.zones
    tiles = r._tiles(zones)
    assert zones == [(8.0, 20.0), (14.0, 26.0), (20.0, 32.0)] and tiles == [(8.0, 17.0), (17.0, 23.0), (23.0, 32.0)]
    assert r._tile_pads(tiles, annuli=zones) == [(0.0, 0.0)] * 3
    # without the zones (an older caller), and for a run saved before: the pads it had
    assert r._tile_pads(tiles) == [(0.0, 2.0), (2.0, 2.0), (2.0, 0.0)]
    r.cfg.stitch_overlap_pads = True
    assert r._tile_pads(tiles, annuli=zones) == [(0.0, 2.0), (2.0, 2.0), (2.0, 0.0)]
    # an edge where two annuli only meet keeps its pad
    m = _runner(tmp_path / "b", ann_edges=[8, 20, 32, 44])
    zm = m.cfg.zones
    assert zm[-1] == (32.0, 44.0) and m._tile_pads(m._tiles(zm), annuli=zm) == [(0.0, 0.0), (0.0, 0.0),
                                                                                  (0.0, 2.0), (2.0, 0.0)]
    # without overlaps nothing changes
    p = _runner(tmp_path / "c", ann_extra=[])
    zp = p.cfg.zones
    assert p._tile_pads(p._tiles(zp), annuli=zp) == [(0.0, 2.0), (2.0, 0.0)]


def test_a_reduced_stack_is_put_in_the_validated_partitions_order():
    from klip_tpe.runner import _in_order
    st = np.stack([np.full((2, 2), v) for v in (1.0, 2.0, 3.0)])
    assert [float(o[0, 0]) for o in _in_order(st, ["n3", "n1", "n2"], ["n1", "n2", "n3"])] == [2.0, 3.0, 1.0]
    assert _in_order(st, ["n1", "n2", "n3"], ["n1", "n2", "n3"]) is st
    assert _in_order(st, ["n1", "n2", "n4"], ["n1", "n2", "n3"]) is st, "other partitions: left as it is"
    assert _in_order(None, ["n1"], ["n1"]) is None and _in_order(st, None, ["n1"]) is st


def test_close_pairs_from_different_annuli_are_logged(tmp_path):
    """With every annulus' sources in each image, two sources of different annuli within 2 FWHM
    add to each other's S/N: the log names them."""
    r = _runner(tmp_path)
    logs = []
    near = r._log_close_pairs([[(0.95, 10.0, 1e-3)], [(0.95, 15.0, 1e-3), (0.95, 190.0, 1e-3)]], logs.append)
    assert len(near) == 1 and near[0][:2] == ((0.95, 10.0), (0.95, 15.0)) and near[0][2] < 2.0
    assert len(logs) == 1 and "1 pair(s) of validation sources from different annuli" in logs[0]
    # two of one annulus are not such a pair: the validation measured them together already
    assert r._log_close_pairs([[(0.95, 10.0, 1e-3), (0.95, 12.0, 1e-3)]], logs.append) == [] and len(logs) == 1


# -------------------------------------------------------------------------- verify_tiles
def _blob(shape, x, y, amp, fwhm=4.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    s = fwhm / 2.3548
    return amp * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / s ** 2)


def test_each_position_is_measured_on_the_image_of_the_annulus_whose_tile_holds_it():
    shape, fw = (81, 81), 4.0
    cx, cy = star_center(shape)
    rng = np.random.default_rng(3)
    tiles = [(6.0, 18.0), (18.0, 34.0)]
    a_src, b_src, c_src = (12 * PX, 40.0, 1e-3), (25 * PX, 200.0, 1e-3), (14 * PX, 250.0, 1e-3)
    known = [(28 * PX, 120.0)]

    def xy(s):
        x, y = source_xy([s[0]], [s[1]], PX, cx, cy)
        return float(x[0]), float(y[0])

    clean = [rng.standard_normal(shape), rng.standard_normal(shape)]
    kx, ky = xy(known[0])
    clean[1] = clean[1] + _blob(shape, kx, ky, 12.0)               # a companion in the second tile
    # the first annulus' injected image holds A only, the second's A, B and C
    inj0 = clean[0] + _blob(shape, *xy(a_src), 12.0)
    inj1 = clean[1] + sum(_blob(shape, *xy(s), 12.0) for s in (a_src, b_src, c_src))
    annuli = [{"annulus": 0, "tile": tiles[0], "clean": clean[0], "inj": inj0, "inj_sources": [a_src]},
              {"annulus": 1, "tile": tiles[1], "clean": clean[1], "inj": inj1, "inj_sources": [a_src, b_src, c_src]}]
    rows, maps = V.verify_tiles(annuli, known, [a_src, b_src, c_src], fw, PX, known=known, n_boot=0)
    cand = [r for r in rows if r["kind"] == "candidate"]
    inj = [r for r in rows if r["kind"] == "injection"]
    assert [(r["index"], r["annulus"]) for r in cand] == [(1, 2)] and cand[0]["snr"] > 5
    assert [(r["index"], r["annulus"]) for r in inj] == [(1, 1), (2, 2), (3, 1)]
    assert inj[0]["snr"] > 5 and inj[1]["snr"] > 5
    # C lies in the first tile, whose annulus did not inject it: listed, not measured
    assert inj[2].get("absent") and not np.isfinite(inj[2]["snr"])
    assert maps["limits"] is not None and maps["limits"]["n_inj"] == 2
    # the report says whose image measured each row, and the stitched report gets both S/N
    text = V.write_verify_table(os.devnull, rows, maps["limits"])
    assert "measured on annulus 2's image, tile 18.00-34.00 px" in text and "| a2" in text
    assert "not in annulus 1's injected image (tile 6.00-18.00 px): not measured" in text
    stitched = [dict(r, snr=1.0) for r in rows]
    lines = V.tile_comparison_lines(stitched, rows)
    assert all(ln.startswith("#") for ln in lines) and len(lines) == 5 + len(rows)
    assert "not injected there" in lines[-1] and " a2 " in lines[5]


# -------------------------------------------------------------------------- a run
@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Two overlapping annuli, (8, 24) and (12, 28) px, tiles split at 18 px, and every source
    at 19 px: the first annulus' sources lie in the second annulus' tile."""
    d = str(tmp_path_factory.mktemp("allsrc"))
    red, space, obj, samp, cfg = build_synthetic_run(
        ann_edges=[8, 24], ann_extra=[[12, 28]], n_iter=[5], n_init=3, param_verify=False, stitch_every=0,
        inject_rho=RHO_PX * PX, verify=True, verify_n_boot=5, write_setup_files=False, fm_curve=False,
        validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[2e-3, 2e-3]))
    logs = []
    r = Runner(red, space, obj, samp, cfg, d, log=logs.append)
    results = r.run()
    return d, r, results, logs


def _rows(path):
    """The injection lines of a verify report: (rho, PA, S/N or None, annulus or None)."""
    out = []
    for ln in open(path).read().splitlines():
        if not ln.startswith("   rho="):
            continue
        rho = float(ln.split("rho=")[1].split('"')[0])
        pa = float(ln.split("PA=")[1].split()[0])
        snr = None if "not measured" in ln else float(ln.split("S/N=")[1].split()[0])
        ann = int(ln.rsplit("| a", 1)[1]) if "| a" in ln else None
        out.append((rho, pa, snr, ann))
    return out


@pytest.mark.slow
def test_the_final_stitch_holds_every_annulus_sources(run):
    d, r, results, logs = run
    own = [list(map(tuple, x.winner_sources)) for x in results]
    assert all(own) and sum(map(len, own)) >= 2
    assert sum("the winner reduced with all" in ln and "for the final stitch" in ln for ln in logs) == 2, logs
    h = fits.getheader(os.path.join(d, "klip_stitched_inj.fits"))
    assert h["INJALL"] == 1 and h["OVLPADS"] == 0 and h["INJNSRC"] == sum(map(len, own))
    # every source is measured in the injected stitch, the first annulus' ones in the second's
    # tile too
    rows = _rows(os.path.join(d, "stitched_verify_report.txt"))
    assert len(rows) == sum(map(len, own)) and all(s is not None and s > 3.0 for _, _, s, _ in rows), rows
    # the annuli's own injected images are the validation's: the first holds its own sources
    # only, so the second annulus' sources are not in it
    inj = fits.getdata(os.path.join(d, "klip_stitched_inj.fits")).astype(float)
    b0 = fits.getdata(os.path.join(d, "annulus01", "best_inj.fits")).astype(float)
    allxy = [tuple(float(v[0]) for v in source_xy([s[0]], [s[1]], PX, *star_center(inj.shape),
                                                   r.reducer.angle_convention)) for s in own[0] + own[1]]
    for j, s in enumerate(own[1]):
        me = len(own[0]) + j
        others = [p for k, p in enumerate(allxy) if k != me]
        on_b0 = V.snr_fap(b0, s[0], s[1], PX, r.fwhm, exclude_xy=others, angle_convention=r.reducer.angle_convention)
        on_st = V.snr_fap(inj, s[0], s[1], PX, r.fwhm, exclude_xy=others, angle_convention=r.reducer.angle_convention)
        assert on_st["snr"] > 3.0 and on_b0["snr"] < 0.5 * on_st["snr"], (s, on_b0["snr"], on_st["snr"])


@pytest.mark.slow
def test_the_tiles_meet_sharply_and_the_tile_snr_maps_put_each_tile_into_the_stitch(run):
    d, r, results, logs = run
    tiles = [(8.0, 18.0), (18.0, 28.0)]
    st = fits.getdata(os.path.join(d, "klip_stitched.fits")).astype(float)
    snr_t = fits.getdata(os.path.join(d, "klip_stitched_snr_tiles.fits")).astype(float)
    snr_st = fits.getdata(os.path.join(d, "klip_stitched_snr.fits")).astype(float)
    assert os.path.exists(os.path.join(d, "klip_stitched_snr_tiles_inj.fits"))
    idx = tile_index(st.shape, tiles)
    yy, xx = np.mgrid[0:st.shape[0], 0:st.shape[1]]
    cx, cy = star_center(st.shape)
    rr = np.hypot(xx - cx, yy - cy)
    on_edge = np.abs(rr - 18.0) < 1e-6
    for k in range(2):
        best = fits.getdata(os.path.join(d, f"annulus{k + 1:02d}", "best_clean.fits")).astype(float)
        m = (idx == k) & ~on_edge & np.isfinite(st) & np.isfinite(best)
        assert m.sum() > 200 and np.allclose(st[m], best[m], rtol=1e-6, atol=1e-9), "no blend inside the tiles"
        # the S/N map of the stitch with this tile put in, from the float32 files: to their precision
        own_snr = r._snr_map(put_tile_in(st, best, idx == k))
        m2 = (idx == k) & np.isfinite(own_snr)
        assert m2.sum() > 200 and np.allclose(snr_t[m2], own_snr[m2], rtol=1e-4, atol=1e-4)
        # no blend inside the tiles: the stitch's own S/N map there
        assert np.allclose(snr_t[m2], snr_st[m2], rtol=1e-4, atol=1e-4)
    assert np.all(np.isnan(snr_t[idx < 0]))


@pytest.mark.slow
def test_the_tile_report_and_the_two_snr_side_by_side(run):
    d, r, results, logs = run
    tile_rows = _rows(os.path.join(d, "tile_verify_report.txt"))
    nsrc = sum(len(x.winner_sources) for x in results)
    assert len(tile_rows) == nsrc and all(a == 2 and s is not None for _, _, s, a in tile_rows), tile_rows
    tile_text = open(os.path.join(d, "tile_verify_report.txt")).read()
    assert "PATH A = the stitched image with the tile of the annulus that holds the position put in" in tile_text
    text = open(os.path.join(d, "stitched_verify_report.txt")).read()
    assert "==== the stitch beside the tiles (tile_verify_report.txt) ====" in text
    assert "differ only where the stitch blends or patches that tile" in text
    side = [ln for ln in text.splitlines() if ln.startswith("#   injection")]
    assert len(side) == nsrc and all(" a2 " in ln for ln in side)
    # the tiles take no blend here: the stitch's S/N and the tile's are the same
    assert all(ln.split()[-1] == ln.split()[-2] for ln in side), side


@pytest.mark.slow
def test_rereduce_puts_every_source_everywhere_for_a_new_run_and_own_when_asked(run, tmp_path):
    d, r, results, logs = run
    red, space, obj, samp, _ = build_synthetic_run()
    rr = Runner.resume(d, red, obj, samp, log=QUIET, same_frames=False)
    rec = rr.rereduce(str(tmp_path / "all"), log=QUIET)
    assert rec["stitch_injections"] == "all" and rec["overlap_pads"] is False
    assert all(a["injected"] == sum(len(x.winner_sources) for x in results) for a in rec["annuli"])
    for f in ("klip_stitched.fits", "klip_stitched_inj.fits", "klip_stitched_snr_tiles.fits"):
        a0 = fits.getdata(os.path.join(d, f))
        a1 = fits.getdata(os.path.join(str(tmp_path / "all"), f))
        assert np.array_equal(a0, a1, equal_nan=True), f"the run's own frames give the run's {f}"
    assert rec["tile_verify_report"].endswith("tile_verify_report.txt")
    # each annulus its own, as a run saved before: the first annulus' sources, in the second
    # tile, are then in neither the stitch nor the second annulus' image
    rr = Runner.resume(d, red, obj, samp, log=QUIET, same_frames=False)
    rec2 = rr.rereduce(str(tmp_path / "own"), log=QUIET, stitch_injections="own", overlap_pads=True)
    assert rec2["stitch_injections"] == "own" and rec2["overlap_pads"] is True
    h = fits.getheader(os.path.join(str(tmp_path / "own"), "klip_stitched_inj.fits"))
    assert h["INJALL"] == 0 and h["OVLPADS"] == 1
    first = {(round(s[0], 3), round(s[1], 1)) for s in results[0].winner_sources}
    rows_own = _rows(os.path.join(str(tmp_path / "own"), "tile_verify_report.txt"))
    assert {(round(a, 3), round(b, 1)) for a, b, s, _ in rows_own if s is None} == first
    st_all = {(round(a, 3), round(b, 1)): s for a, b, s, _ in _rows(os.path.join(str(tmp_path / "all"),
                                                                              "stitched_verify_report.txt"))}
    st_own = {(round(a, 3), round(b, 1)): s for a, b, s, _ in _rows(os.path.join(str(tmp_path / "own"),
                                                                              "stitched_verify_report.txt"))}
    for k in first:
        assert st_all[k] > st_own[k] + 1.0, (k, st_all[k], st_own[k])
    with pytest.raises(ValueError, match="does not go with it"):
        rr.rereduce(str(tmp_path / "x"), log=QUIET, stitch_injections="own", verify_sources=[(0.95, 10.0, 3e-4)])


@pytest.mark.slow
def test_an_old_run_keeps_its_own_injections_and_pads(run, tmp_path):
    d, r, results, logs = run
    ck = os.path.join(d, "checkpoint.json")
    keep = open(ck).read()
    st = json.loads(keep)
    for k in ("stitch_injections", "stitch_overlap_pads"):
        st["config"].pop(k, None)
    try:
        open(ck, "w").write(json.dumps(st))
        red, space, obj, samp, _ = build_synthetic_run()
        rr = Runner.resume(d, red, obj, samp, log=QUIET, same_frames=False)
        assert rr.cfg.stitch_injections == "own" and rr.cfg.stitch_overlap_pads is True
        rec = rr.rereduce(str(tmp_path / "old"), log=QUIET, verify=False)
        assert rec["stitch_injections"] == "own" and rec["overlap_pads"] is True
        assert [a["injected"] for a in rec["annuli"]] == [len(x.winner_sources) for x in results]
    finally:
        open(ck, "w").write(keep)
