"""The stitch at an edge two annuli share (RunConfig.stitch_blend), after the audit of 2026-11-25.

``"own"`` (the default) takes each pixel from the annulus whose tile holds it.  ``"weighted"``, as runs
made before stitched, blends a 2 px band on each side of the edge by 1/sigma^2, which can give the last
2 px of a tile to the neighbor's padding.  On the paper's beta Pic run the neighbor's configuration kept
0.1 % of a source there, and the stitch a median 8 % of what the tile's own annulus kept.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import build_synthetic_run
from klip_tpe import CalibrationConfig, RunConfig, Runner, ValidationConfig
from klip_tpe.metrics import source_xy, star_center
from klip_tpe.param_verify import stitch_param_maps
from klip_tpe.stitch import stitch_stacks, stitch_tiles, tile_masks

QUIET = (lambda s: None)
N = 61


def _radius(n=N):
    cx, cy = star_center((n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    return np.hypot(xx - cx, yy - cy)


def test_an_unpadded_shared_edge_belongs_to_the_outer_tile():
    rr = _radius()
    tiles = [(0.0, 12.0), (12.0, 24.0), (24.0, np.inf)]
    m = tile_masks((N, N), tiles, [(0.0, 0.0)] * 3)
    on_edge = np.isclose(rr, 12.0)
    assert on_edge.any()
    assert not m[0][on_edge].any() and m[1][on_edge].all()
    assert np.array_equal(sum(mi.astype(int) for mi in m), np.ones((N, N), int))    # each pixel once
    # padded edges overlap as before, and the run's own ends are left as given
    p = tile_masks((N, N), tiles, [(0.0, 2.0), (2.0, 2.0), (2.0, 0.0)])
    assert p[0][on_edge].all() and p[1][on_edge].all()
    assert np.array_equal(p[0], (rr >= 0.0) & (rr <= 14.0))
    # an edge padded on one side only is not shared without a pad
    q = tile_masks((N, N), tiles[:2], [(0.0, 0.0), (2.0, 0.0)])
    assert q[0][on_edge].all()


def test_without_pads_each_pixel_comes_from_one_image():
    rr = _radius()
    a, b = np.full((N, N), 1.0), np.full((N, N), 2.0)
    tiles = [(0.0, 12.0), (12.0, np.inf)]
    own = stitch_tiles([a, b], tiles, [0.001, 1.0], pads=[(0.0, 0.0), (0.0, 0.0)])
    assert set(np.unique(own)) == {1.0, 2.0}
    assert np.all(own[rr < 12.0] == 1.0) and np.all(own[rr >= 12.0] == 2.0)
    blended = stitch_tiles([a, b], tiles, [0.001, 1.0], pads=[(0.0, 2.0), (2.0, 0.0)])
    band = (rr >= 10.0) & (rr <= 14.0)
    assert np.allclose(blended[band], (0.001 * 1.0 + 2.0) / 1.001)               # the quieter wins the band
    # per-partition stacks follow the same rule
    cube, ids = stitch_stacks([a[None], b[None]], [["p"], ["p"]], tiles, pads=[(0.0, 0.0), (0.0, 0.0)])
    assert ids == ["p"] and np.array_equal(cube[0], own)


def test_a_source_inside_its_tile_keeps_its_own_annulus_signal():
    """A source 1 px inside a seam: the outer annulus' padding kills it and is quiet, as on beta Pic."""
    rng = np.random.default_rng(3)
    rr = _radius()
    cx, cy = star_center((N, N))
    (x,), (y,) = source_xy([11.0 * 0.05], [40.0], 0.05, cx, cy, "pa")
    yy, xx = np.mgrid[0:N, 0:N]
    psf = np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / (3.6 / 2.3548) ** 2)
    inner = 3.0 * rng.normal(size=(N, N)) + 40.0 * psf             # keeps the source, noisy
    outer = 0.15 * rng.normal(size=(N, N)) + 0.04 * psf            # keeps 0.1 %, quiet
    w = np.array([1 / 3.0 ** 2, 1 / 0.15 ** 2]) / (1 / 0.15 ** 2)   # the final stitch's 1/sigma^2 weights
    tiles = [(0.0, 12.0), (12.0, np.inf)]
    ap = np.hypot(xx - x, yy - y) <= 1.8
    own = stitch_tiles([inner, outer], tiles, w, pads=[(0.0, 0.0), (0.0, 0.0)])
    old = stitch_tiles([inner, outer], tiles, w, pads=[(0.0, 2.0), (2.0, 0.0)])
    inside = ap & (rr < 12.0)
    assert np.allclose(own[inside], inner[inside], rtol=1e-12, atol=0)        # the weight cancels
    signal = float(np.sum(40.0 * psf[ap]))
    assert np.sum(own[ap]) > 0.7 * signal                           # the part across the seam is lost
    assert np.sum(old[ap]) < 0.25 * signal < 0.4 * np.sum(own[ap])  # the blend keeps the pixels below 10 px


def _runner(tmp_path, **kw):
    base = dict(ann_edges=[8, 20, 30, 45], n_iter=[6, 6, 6], n_init=3, param_verify=False, stitch_every=0,
                calibration=CalibrationConfig(recal_budget=0), validation=ValidationConfig(n_top=1, n_valid=1))
    base.update(kw)
    red, space, obj, samp, cfg = build_synthetic_run(**base)
    return Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)


def test_a_new_run_stitches_its_own_pixels_and_an_old_one_its_blend(tmp_path):
    assert RunConfig().stitch_blend == "own"
    d = RunConfig(ann_edges=[8, 20, 30]).to_dict()
    assert RunConfig.from_dict(d).stitch_blend == "own"
    d.pop("stitch_blend")
    assert RunConfig.from_dict(d).stitch_blend == "weighted"        # a run saved before 2026-11-25
    tiles = [(8.0, 20.0), (20.0, 30.0), (30.0, 45.0)]
    assert _runner(tmp_path / "n")._tile_pads(tiles) == [(0.0, 0.0)] * 3
    old = _runner(tmp_path / "o", stitch_blend="weighted")
    assert old._tile_pads(tiles) == [(0.0, 2.0), (2.0, 2.0), (2.0, 0.0)]
    # the run's own outermost edges keep the pads edge_padding "all" gives them
    every = _runner(tmp_path / "a", edge_padding="all")
    assert every._tile_pads(tiles) == [(2.0, 0.0), (0.0, 0.0), (0.0, 2.0)]


def test_the_parameter_maps_take_the_same_rule():
    rr = _radius()
    one = {"stim": np.full((N, N), 1.0)}
    two = {"stim": np.full((N, N), 3.0)}
    out = stitch_param_maps([one, two], [(4.0, 12.0), (12.0, 24.0)], kinds=("stim",),
                            pads=[(0.0, 0.0), (0.0, 0.0)])["stim"]
    on_edge = np.isclose(rr, 12.0)
    assert np.all(out[on_edge] == 3.0) and np.all(out[(rr >= 4.0) & (rr < 12.0)] == 1.0)


def test_the_commands_that_read_a_run_take_the_option():
    from klip_tpe import cli
    p = cli._build_parser()
    a = p.parse_args(["near", "--stitch-blend", "weighted"])
    assert cli._protocol_config(a)["stitch_blend"] == "weighted"
    assert "stitch_blend" not in cli._protocol_config(p.parse_args(["near"]))     # RunConfig's own default
    for cmd in (["rereduce"], ["elect"], ["rescore"]):
        assert p.parse_args(cmd + ["--stitch-blend", "own"]).stitch_blend == "own"
        assert p.parse_args(cmd).stitch_blend is None
    r = SimpleNamespace(cfg=SimpleNamespace(stitch_blend="weighted"))
    cli._stitch_blend_for_run(SimpleNamespace(stitch_blend="own"), r)
    assert r.cfg.stitch_blend == "own"
    cli._stitch_blend_for_run(SimpleNamespace(stitch_blend=None), r)
    assert r.cfg.stitch_blend == "own"


def test_the_final_grid_records_the_rule():
    from klip_tpe.final_grid import _pads_text, _stitch_record
    sc = SimpleNamespace(zones=[(8.0, 20.0), (20.0, 30.0)], tiles=[(8.0, 20.0), (20.0, 30.0)],
                         pads=[(0.0, 0.0), (0.0, 0.0)], overlap_pads=False, blend="own")
    rec = _stitch_record(sc, SimpleNamespace(cfg=SimpleNamespace(stitch_overlap_pads=False)))
    assert rec["blend"] == "own" and rec["edge_pads_px"] == [0.0]
    assert "each pixel from its tile's annulus" in _pads_text(rec)


@pytest.mark.parametrize("blend", ["own", "weighted"])
def test_the_stitch_scorer_takes_the_run_s_rule(tmp_path, blend):
    r = _runner(tmp_path, stitch_blend=blend)
    tiles = r._tiles(r.cfg.zones)
    pads = r._tile_pads(tiles, annuli=r.cfg.zones)
    assert (max(max(p) for p in pads[:-1]) == 0.0) == (blend == "own")
