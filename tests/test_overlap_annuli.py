"""Overlapping annuli, ``--ann-extra`` (NEAR2 s6 note, section 5).

Annuli beside those of ``ann_edges`` may overlap them.  All are numbered by centre radius and
searched, validated and verified over their own zones.  Where two overlap, the stitch takes each
radius from the annulus with the nearest centre: the tile boundary is the midpoint between the
centres, with the 2 px blend an edge between annuli has.  The run's stitch, its per-night cubes,
the param maps, the seam patching, the contrast curve, ``rereduce`` and ``fit-chain`` all stitch
by the tiles, and ``final_results.json`` and ``klip_stitched_params.txt`` record them.  A run
without ``ann_extra`` has tiles equal to its zones and every product as before.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from astropy.io import fits

from klip_tpe import cli
from klip_tpe.display import _cfg_zones
from klip_tpe.metrics import star_center
from klip_tpe.runner import RunConfig, Runner, ValidationConfig
from klip_tpe.stitch import annulus_tiles

from conftest import build_synthetic_run

S6_EDGES = [2.19, 11.5, 20.81, 30.12, 39.43, 48.74, 58.05, 67.36]
S6_EXTRA = [[6.85, 16.16], [16.16, 25.47], [25.47, 34.78]]


def test_the_s6_layout_numbers_by_centre_and_tiles_by_the_nearest_centre():
    cfg = RunConfig(ann_edges=S6_EDGES, ann_extra=S6_EXTRA)
    zs = cfg.zones
    assert cfg.nann == 10 and zs[0] == (2.19, 11.5) and zs[1] == (6.85, 16.16) and zs[9] == (58.05, 67.36)
    c = [0.5 * (a + b) for a, b in zs]
    assert c == sorted(c)
    t = cfg.tiles
    assert t[0] == (2.19, pytest.approx(0.5 * (c[0] + c[1])))
    for k in range(6):
        assert t[k][1] == pytest.approx(0.5 * (c[k] + c[k + 1])) == t[k + 1][0]
    # from annulus 7 on the zones meet without overlapping, and the tiles keep their edges
    assert t[6] == (pytest.approx(32.45), 39.43) and t[7:] == zs[7:]
    for (a, b), (ta, tb) in zip(zs, t):
        assert a <= ta < tb <= b, "every tile inside its zone"


def test_without_extra_annuli_the_tiles_are_the_zones():
    for edges in ([8, 30, 45], [0, 20], [2.19, 11.5, 20.81], [10, 20, 40, 41]):
        cfg = RunConfig(ann_edges=edges)
        assert cfg.zones == [(float(a), float(b)) for a, b in zip(edges, edges[1:])] == cfg.tiles
        assert cfg.nann == len(edges) - 1
        assert annulus_tiles(cfg.zones) == cfg.zones
    assert RunConfig.from_dict(RunConfig(ann_edges=[8, 30]).to_dict()).ann_extra == []
    d = RunConfig(ann_edges=[8, 30]).to_dict()
    d.pop("ann_extra")
    assert RunConfig.from_dict(d).ann_extra == [], "a run saved before the option resumes without it"
    # unequal widths that meet: the edge, not the midpoint between the centres
    assert annulus_tiles([(8, 30), (30, 45)]) == [(8.0, 30.0), (30.0, 45.0)]
    # a midpoint outside the overlap is kept inside it
    assert annulus_tiles([(0, 100), (90, 120)]) == [(0.0, 90.0), (90.0, 120.0)]


@pytest.mark.parametrize("extra,why", [([[9, 19]], "one holds the other"), ([[8, 20]], "the same zone twice"),
                                       ([[15, 15]], "empty")])
def test_zones_that_cannot_be_tiled_are_refused(extra, why):
    with pytest.raises(ValueError, match=why):
        RunConfig(ann_edges=[8, 20, 32], ann_extra=extra).zones
    with pytest.raises(ValueError, match="opt_width"):
        RunConfig(ann_edges=[8], ann_extra=[[10, 20]], opt_width=True).zones


def test_the_command_line_takes_the_extra_annuli():
    a = cli._build_parser().parse_args(["near", "--root", "/r", "--ann-edges", *map(str, S6_EDGES),
                                        "--ann-extra", "6.85:16.16", "16.16:25.47", "25.47:34.78"])
    kw = cli._protocol_config(a)
    assert kw["ann_extra"] == S6_EXTRA
    assert "ann_extra" not in cli._protocol_config(cli._build_parser().parse_args(["near", "--root", "/r"]))
    bad = cli._build_parser().parse_args(["near", "--root", "/r", "--ann-extra", "6.85-16.16"])
    with pytest.raises(SystemExit, match="IN:OUT"):
        cli._protocol_config(bad)


def test_the_display_numbers_the_annuli_as_the_run_does():
    cfg = RunConfig(ann_edges=S6_EDGES, ann_extra=S6_EXTRA).to_dict()
    assert _cfg_zones(cfg) == RunConfig(ann_edges=S6_EDGES, ann_extra=S6_EXTRA).zones
    assert _cfg_zones(RunConfig(ann_edges=S6_EDGES).to_dict()) is None


@pytest.fixture(scope="module")
def overlap_run(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("overlap"))
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 20, 32], ann_extra=[[14, 26]], n_iter=[5],
                                                     n_init=3, validation=ValidationConfig(n_top=1, n_valid=1),
                                                     param_verify=False)
    logs = []
    r = Runner(red, space, obj, samp, cfg, d, log=logs.append)
    results = r.run()
    return d, r, results, logs


@pytest.mark.slow
def test_a_run_stitches_by_the_tiles_and_records_them(overlap_run):
    d, r, results, logs = overlap_run
    assert [(x.inrad, x.outrad) for x in results] == [(8.0, 20.0), (14.0, 26.0), (20.0, 32.0)]
    tiles = [(8.0, 17.0), (17.0, 23.0), (23.0, 32.0)]
    assert any("annuli by centre, zone (tile) px: 1: 8-20 (8.00-17.00), 2: 14-26 (17.00-23.00), "
               "3: 20-32 (23.00-32.00)" in ln for ln in logs), logs
    fr = json.load(open(os.path.join(d, "final_results.json")))
    assert fr["zones"] == [[8.0, 20.0], [14.0, 26.0], [20.0, 32.0]] and fr["tiles"] == [list(t) for t in tiles]
    params = open(os.path.join(d, "klip_stitched_params.txt")).read()
    assert "tile_in" in params and "   17.00     23.00  " in params
    # inside a tile, away from the 2 px blends, the stitch is that annulus' winner
    st = fits.getdata(os.path.join(d, "klip_stitched.fits")).astype(float)
    yy, xx = np.mgrid[0:st.shape[0], 0:st.shape[1]]
    cx, cy = star_center(st.shape)
    rr = np.hypot(xx - cx, yy - cy)
    for k, (a, b) in enumerate(tiles):
        best = fits.getdata(os.path.join(d, f"annulus{k + 1:02d}", "best_clean.fits")).astype(float)
        m = (rr >= a + 2.5) & (rr <= b - 2.5) & np.isfinite(st) & np.isfinite(best)
        assert m.sum() > 50 and np.allclose(st[m], best[m], rtol=1e-6, atol=1e-9), k
    # the per-night cube by the same tiles
    cube = fits.getdata(os.path.join(d, "klip_stitched_nights.fits.gz"))
    assert cube.shape[0] == 2
    # the contrast curve cut at the tile edges, with the seam trim
    head = open(os.path.join(d, "contrast_curve.txt")).read().splitlines()[2]
    assert "annulus edges (px) = 8.0, 17.0, 23.0, 32.0" in head


@pytest.mark.slow
def test_rereduce_keeps_the_runs_tiles_and_refuses_others(overlap_run, tmp_path):
    d, r, results, logs = overlap_run
    red, space, obj, samp, _ = build_synthetic_run()
    rr = Runner.resume(d, red, obj, samp, log=lambda s: None, same_frames=False)
    rec = rr.rereduce(str(tmp_path / "rr"), verify=False, log=lambda s: None)
    assert [a["tile_px"] for a in rec["annuli"]] == [[8.0, 17.0], [17.0, 23.0], [23.0, 32.0]]
    a0 = fits.getdata(os.path.join(d, "klip_stitched.fits"))
    a1 = fits.getdata(os.path.join(str(tmp_path / "rr"), "klip_stitched.fits"))
    assert np.array_equal(a0, a1, equal_nan=True), "the run's own frames give the run's stitch"
    fr_path = os.path.join(d, "final_results.json")
    fr = json.load(open(fr_path))
    keep = json.dumps(fr)
    try:
        fr["tiles"][0][1] = 16.0
        open(fr_path, "w").write(json.dumps(fr))
        rr2 = Runner.resume(d, red, obj, samp, log=lambda s: None, same_frames=False)
        with pytest.raises(ValueError, match="not the ones its zones give now"):
            rr2.rereduce(str(tmp_path / "rr2"), verify=False, log=lambda s: None)
    finally:
        open(fr_path, "w").write(keep)
