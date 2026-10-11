"""Annulus edges are padded only where two annuli meet (RunConfig.edge_padding).

The reduction zone takes the reducer's ``zone_pad``, a stitch tile 2 px, a seam re-run 3 px
and a param_verify map 1 px, each on an edge another annulus shares.  The run's innermost
inner edge and outermost outer edge are reduced and stitched as given.  ``"all"`` pads every
edge, as the IDL does, and a run saved before the field existed resumes that way.
"""
from __future__ import annotations

import json
import os

import numpy as np

from conftest import build_synthetic_run
from klip_tpe import CalibrationConfig, Param, RunConfig, Runner, ValidationConfig
from klip_tpe import stitch as stitch_mod
from klip_tpe.injection import GaussianPSF
from klip_tpe.products import stitch_annuli
from klip_tpe.reducer import KLIPReducer, ReductionRequest
from klip_tpe.synthetic import synthetic_klip_dataset

QUIET = (lambda s: None)


def _runner(tmp_path, **kw):
    base = dict(ann_edges=[8, 20, 30, 45], n_iter=[6, 6, 6], n_init=3, param_verify=False, stitch_every=0,
                calibration=CalibrationConfig(recal_budget=0), validation=ValidationConfig(n_top=1, n_valid=1))
    base.update(kw)
    red, space, obj, samp, cfg = build_synthetic_run(**base)
    if cfg.opt_width:
        space.add(Param("width", int(cfg.width_range[0]), int(cfg.width_range[1]), "int"))
    return Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)


def test_the_reducer_pads_only_the_edges_it_is_told():
    ds = synthetic_klip_dataset(nframes=12, size=48, pa_span=40)
    px = 0.05
    red = KLIPReducer(ds, px, 3.89 * px / 206265 * 8.2, 8.2, injection_model=GaussianPSF(4.0, star_flux=1e4),
                      outrad_cap=22)
    base = {"k_klip": 3, "inrad": 8, "outrad": 18, "bin": 1, "filter": 0, "angsep": 0.0, "anglemax": 360}

    def zone(**flags):
        m = red.reduce(ReductionRequest({**base, **flags}, None)).meta
        return m["inrad"], m["outrad"]
    assert zone() == (6.0, 20.0)                                    # no marks: both edges, as the IDL
    assert zone(pad_in=False) == (8.0, 20.0)
    assert zone(pad_out=False) == (6.0, 18.0)
    assert zone(pad_in=False, pad_out=False) == (8.0, 18.0)


def test_the_runner_pads_only_edges_between_annuli(tmp_path):
    r = _runner(tmp_path)
    assert [r._pad_edges(*r._annulus(ia)) for ia in range(3)] == [(False, True), (True, True), (True, False)]
    assert r._tile_pads([(8, 20), (20, 30), (30, 45)]) == [(0.0, 2.0), (2.0, 2.0), (2.0, 0.0)]
    one = _runner(tmp_path / "one", ann_edges=[8, 30], n_iter=6)
    assert one._pad_edges(8, 30) == (False, False)
    every = _runner(tmp_path / "all", edge_padding="all")
    assert all(every._pad_edges(*every._annulus(ia)) == (True, True) for ia in range(3))


def test_adaptive_annuli_pad_their_outer_edge_only_when_another_annulus_follows(tmp_path):
    """opt_width: the run stops at r_cap, or when what is left is narrower than the smallest
    width.  An outer edge there is the run's last."""
    r = _runner(tmp_path, ann_edges=[8], opt_width=True, width_range=(8.0, 12.0), r_cap=40.0, n_iter=6)
    assert r._pad_edges(8, 18) == (False, True)
    assert r._pad_edges(18, 28) == (True, True)                     # 12 px left: another annulus
    assert r._pad_edges(28, 35) == (True, False)                    # 5 px left: the run stops here
    assert r._pad_edges(30, 40) == (True, False)                    # at r_cap


def test_every_reduction_request_carries_the_marks(tmp_path):
    r = _runner(tmp_path)
    seen = []
    for sub in r.reducer.reducers.values():
        orig = sub.reduce

        def spy(req, _orig=orig):
            seen.append((req.params["inrad"], req.params["outrad"], req.params["pad_in"], req.params["pad_out"]))
            return _orig(req)
        sub.reduce = spy
    cfg = r.space.decode(r.space.default_vector())
    for ia in range(3):
        r.ia = ia
        r._reduce(cfg, None)
    got = sorted(set(seen))
    assert got == [(8.0, 20.0, False, True), (20.0, 30.0, True, True), (30.0, 45.0, True, False)]


def test_stitch_tiles_take_their_own_pads():
    n = 60
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2)
    a = np.where(rr <= 30, 1.0, np.nan)
    zones = [(8, 16), (16, 24)]
    st = stitch_mod.stitch_tiles([a, a], zones, pads=[(0.0, 2.0), (2.0, 0.0)])
    assert not np.isfinite(st[(rr > 7.0) & (rr < 7.9)]).any()       # nothing inside the first edge
    assert not np.isfinite(st[(rr > 24.1) & (rr < 25.9)]).any()     # nothing beyond the last
    assert np.isfinite(st[(rr > 15) & (rr < 17)]).all()
    old = stitch_mod.stitch_tiles([a, a], zones)                     # the helper's default pads both
    assert np.isfinite(old[(rr > 6.2) & (rr < 7.9)]).all()
    cube, _ = stitch_mod.stitch_stacks([a[None], a[None]], [["n1"], ["n1"]], zones,
                                       pads=[(0.0, 2.0), (2.0, 0.0)])
    assert not np.isfinite(cube[0][(rr > 24.1) & (rr < 25.9)]).any()
    # products.stitch_annuli, the weighting recipe: the same, unless pad_ends
    s2, _ = stitch_annuli([a, a], [8, 16, 24], fwhm=3.0, legacy=True)
    assert not np.isfinite(s2[(rr > 24.1) & (rr < 25.9)]).any()
    s3, _ = stitch_annuli([a, a], [8, 16, 24], fwhm=3.0, legacy=True, pad_ends=True)
    assert np.isfinite(s3[(rr > 24.1) & (rr < 25.9)]).all()


def test_a_run_and_its_stitch_end_where_the_searched_range_ends(tmp_path):
    """End to end: the stitch holds nothing inside the first edge or beyond the last, and the
    two annuli overlap where they meet.  With edge_padding="all" it reaches 2 px past both."""
    from astropy.io import fits

    def run(d, **kw):
        red, space, obj, samp, cfg = build_synthetic_run(
            ann_edges=[8, 20, 32], n_iter=[5, 5], n_init=3, param_verify=False, stitch_every=0, fm_curve=False,
            calibration=CalibrationConfig(recal_budget=0), validation=ValidationConfig(n_top=1, n_valid=1), **kw)
        Runner(red, space, obj, samp, cfg, str(d), log=QUIET).run()
        st = fits.getdata(os.path.join(str(d), "klip_stitched.fits")).astype(float)
        ny, nx = st.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        rr = np.hypot(xx - (nx - 1) / 2, yy - (ny - 1) / 2)
        return st, rr

    st, rr = run(tmp_path / "between")
    assert not np.isfinite(st[rr < 7.9]).any() and not np.isfinite(st[rr > 32.1]).any()
    assert np.isfinite(st[(rr >= 8.5) & (rr <= 31.5)]).mean() > 0.99
    st_all, rr = run(tmp_path / "all", edge_padding="all")
    assert np.isfinite(st_all[(rr > 6.2) & (rr < 7.8)]).mean() > 0.9
    assert np.isfinite(st_all[(rr > 32.2) & (rr < 33.8)]).mean() > 0.9


def test_a_run_saved_before_the_field_resumes_with_every_edge_padded(tmp_path):
    cfg = RunConfig(ann_edges=[8, 20, 30])
    d = json.loads(json.dumps(cfg.to_dict()))
    assert RunConfig.from_dict(d).edge_padding == "between"
    d.pop("edge_padding")
    assert RunConfig.from_dict(d).edge_padding == "all"
