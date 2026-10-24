"""Coloured injections and the ALES paper reduction's ADI step, for a run tuned on MWC 758c.

What must hold:

* ``PartitionedReducer.set_injection_spectrum`` gives each partition the sources at its own
  point of the spectrum, from a function of wavelength, a table or one factor per partition,
  normalised at the representative partition, another partition or a wavelength; the
  search's, the SDI pool's and the KLIP-FM sources all get it; a negative source (a known
  companion taken out) keeps its contrast; None is the flat spectrum, to the bit; the
  factors are in the reducer's record;
* ``zone_in`` / ``zone_out`` widen the KLIP zone beyond the annulus, and the bin's PA span
  and ``angsep``'s arc stay the annulus'; zero is the reduction it was, to the bit;
* ``post_hp = 0`` derotates the residuals without the second high-pass, ``1`` (and absent)
  is the reduction it was;
* ``make_space(zone_range=..., search_post_hp=True)`` searches them, seeded from ``defaults``,
  and leaves the space as it was without them;
* a tiny search with all of it goes end to end, with the spectrum in the setup file.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from klip_tpe import CalibrationConfig, RunConfig, Runner, ValidationConfig
from klip_tpe.instruments import ales
from klip_tpe.klip import arcdist_deg
from klip_tpe.metrics import Source
from klip_tpe.space import Config

QUIET = (lambda s: None)
NW, NF, N = 12, 40, 48


def _groups(planet=None):
    """Four wavelength groups of a synthetic IFS sequence (a star with a halo, noise, PA -20 to
    +24), as ``(datasets, lam_m, fwhm_px, table)``."""
    rng = np.random.default_rng(3)
    waves = np.linspace(2.9, 4.2, NW)
    pa = np.linspace(-20.0, 24.0, NF)
    cx = cy = (N - 1) / 2.0
    yy, xx = np.mgrid[0:N, 0:N]
    cube = np.empty((NW, NF, N, N), np.float32)
    tpl = np.zeros((NW, 50, 54), np.float32)
    ty, tx = np.mgrid[0:50, 0:54]
    for w in range(NW):
        sg = 1.03 * (waves[w] * 1e-6 / 8.4) * 206265.0 / 0.0345 / 2.3548
        amp = 3000.0 * np.exp(-0.5 * ((waves[w] - 3.6) / 0.5) ** 2)
        star = amp * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sg ** 2) + 40 * amp / (1 + np.hypot(xx - cx, yy - cy) ** 2)
        tpl[w] = amp * np.exp(-0.5 * ((tx - 27) ** 2 + (ty - 25) ** 2) / sg ** 2)
        for f in range(NF):
            cube[w, f] = star + rng.standard_normal((N, N)) * 3.0
    return ales.load_groups(cube, pa, waves, tpl, c0=1, c1=10, ngrp=4, psf_center=(27.0, 25.0), log=QUIET)


def _reducer(**kw):
    ds, lam, fw, table = _groups()
    return ales.make_reducer(ds, lam, fw, max_workers=1, table=table, log=QUIET, **kw), lam


def _cfg(red, zone=(7.0, 13.0), **params):
    space = ales.make_space(red, defaults={"bin": 2, "k_klip": 3, "filter": 9})
    cfg = space.decode(space.default_vector())
    p = dict(cfg.params, inrad=zone[0], outrad=zone[1], pad_in=False, pad_out=False, **params)
    return Config(p, {q: dict(d, inrad=zone[0], outrad=zone[1], pad_in=False, pad_out=False, **params)
                      for q, d in cfg.per_partition.items()}, cfg.selected, cfg.x)


def _record_requests(red):
    """Every partition job's request, by partition (the jobs run in this process at max_workers=1)."""
    seen = {}
    for pid, r in red.reducers.items():
        orig = r.reduce

        def rec(req, pid=pid, orig=orig):
            seen.setdefault(pid, []).append(req)
            return orig(req)
        r.reduce = rec
    return seen


# ------------------------------------------------------------------------------- the spectrum
def test_the_spectrum_scales_each_partitions_sources():
    red, lam = _reducer()
    rep = red.representative
    um = {p: lam[p] * 1e6 for p in red.partitions()}
    fac = red.set_injection_spectrum(lambda w: w ** 4, label="lambda^4")
    for p in red.partitions():
        assert fac[p] == pytest.approx((um[p] / um[rep]) ** 4)
    assert fac[rep] == 1.0
    info = red.describe()["injection_spectrum"]
    assert info["label"] == "lambda^4" and info["normalised_at"].startswith(f"partition {rep}")
    assert info["factors"]["g00"] == pytest.approx(fac["g00"], rel=1e-5)
    seen = _record_requests(red)
    src = [Source(0.38, 240.0, 1e-3), Source(0.38, 60.0, -2e-3)]     # an injection and a known source taken out
    fm = [Source(0.30, 120.0, 5e-4)]
    red.reduce_config(_cfg(red), injections=src, fm_sources=fm)
    for p in red.partitions():
        req = seen[p][-1]
        assert req.injections[0].contrast == pytest.approx(1e-3 * fac[p])
        assert req.injections[1].contrast == -2e-3, "a known source keeps the contrast given"
        assert (req.injections[0].rho, req.injections[0].theta) == (0.38, 240.0)
        assert req.fm_sources[0].contrast == pytest.approx(5e-4 * fac[p])
    assert src[0].contrast == 1e-3, "the caller's sources are not changed"
    # the frames a partition injects carry its factor: the injected flux is linear in the contrast
    r = red.reducers["g00"]
    one = [Source(0.38, 240.0, 1e-3)]
    d_flat = r._inject(one) - r.data.cube
    d_col = r._inject(red.scaled_sources(one, "g00")) - r.data.cube
    np.testing.assert_allclose(d_col.sum(), fac["g00"] * d_flat.sum(), rtol=1e-4)


def test_tables_dicts_and_normalisations():
    red, lam = _reducer()
    um = {p: lam[p] * 1e6 for p in red.partitions()}
    lo, hi = min(um.values()), max(um.values())
    # a table: linear in wavelength, held at its end values
    tab = ([lo + 0.01, hi - 0.01], [1.0, 3.0])
    fac = red.set_injection_spectrum(tab, norm="g03")
    raw = {p: float(np.interp(um[p], *tab)) for p in um}
    assert fac["g03"] == 1.0 and fac["g00"] == pytest.approx(raw["g00"] / raw["g03"])
    assert raw["g00"] == 1.0 and raw["g03"] == 3.0, "held at the ends"
    # one factor per partition, normalised at a partition
    fac = red.set_injection_spectrum({"g00": 0.0, "g01": 1.0, "g02": 2.0, "g03": 4.0}, norm="g02")
    assert fac == {"g00": 0.0, "g01": 0.5, "g02": 1.0, "g03": 2.0}
    # a wavelength (m) as the normalisation
    fac = red.set_injection_spectrum(lambda w: w, norm=3.5e-6)
    assert fac["g01"] == pytest.approx(um["g01"] / 3.5)
    with pytest.raises(ValueError, match="per-partition"):
        red.set_injection_spectrum({"g00": 1.0, "g01": 1.0, "g02": 1.0, "g03": 1.0}, norm=3.5e-6)
    with pytest.raises(KeyError, match="no factor"):
        red.set_injection_spectrum({"g00": 1.0})
    with pytest.raises(ValueError, match="must be > 0"):
        red.set_injection_spectrum({"g00": 1.0, "g01": 1.0, "g02": 0.0, "g03": 1.0}, norm="g02")
    with pytest.raises(ValueError, match=">= 0"):
        red.set_injection_spectrum(lambda w: w - 0.5 * (um["g00"] + um["g01"]))   # < 0 in g00 only
    # the adapter takes it at construction
    red2, _ = _reducer(spectrum=lambda w: w ** 2, spectrum_label="red")
    assert red2.injection_scale["g03"] > 1.0 > red2.injection_scale["g00"]
    assert red2.describe()["injection_spectrum"]["label"] == "red"


def test_no_spectrum_is_the_flat_injection_to_the_bit():
    red, _ = _reducer()
    src = [Source(0.38, 240.0, 1e-3)]
    a = red.reduce_config(_cfg(red), injections=src).image
    red.set_injection_spectrum({p: 1.0 for p in red.partitions()})
    b = red.reduce_config(_cfg(red), injections=src).image
    red.set_injection_spectrum(None)
    c = red.reduce_config(_cfg(red), injections=src).image
    assert "injection_spectrum" not in red.describe()
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    red.set_injection_spectrum(lambda w: w ** 4)
    d = red.reduce_config(_cfg(red), injections=src).image
    assert not np.allclose(np.nan_to_num(a), np.nan_to_num(d))


def test_the_sdi_pool_borrows_coloured_frames():
    ds, lam, fw, table = _groups()
    red = ales.make_reducer(ds, lam, fw, max_workers=1, table=table, sdi=True, sdi_cap=8, log=QUIET,
                            spectrum=lambda w: w ** 4)
    seen = _record_requests(red)
    src = [Source(0.38, 240.0, 1e-3)]
    red.reduce_config(_cfg(red, nkeep_sdi=4), injections=src)
    for p in red.partitions():
        prep = [q for q in seen[p] if q.stage == "prepare"]
        assert prep, f"{p} lends its frames"
        assert prep[-1].injections[0].contrast == pytest.approx(1e-3 * red.injection_scale[p])


# --------------------------------------------------------------------- the zone and the post high-pass
def _one(red, pid="g01", **params):
    ev = red.reduce_config(_cfg(red, **params), selected=[pid])
    return ev.stack[0], ev.meta[pid]


def test_zone_in_and_out_widen_the_klip_zone_only():
    red, _ = _reducer()
    base, m0 = _one(red)
    zero, mz = _one(red, zone_in=0.0, zone_out=0.0)
    np.testing.assert_array_equal(base, zero)
    assert "annulus" not in m0 and "annulus" not in mz
    wide, mw = _one(red, zone_in=3.0, zone_out=2.0)
    assert (mw["inrad"], mw["outrad"]) == (4.0, 15.0) and mw["annulus"] == [7.0, 13.0]
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.hypot(xx - (N - 1) / 2.0, yy - (N - 1) / 2.0)
    inner = (rr >= 4.0) & (rr < 5.0)          # inside the annulus' zone even after derotation
    assert np.isnan(base[inner]).all() and np.isfinite(wide[inner]).all()
    # angsep's arc and the bins are the annulus'
    assert mw["arcdist"] == pytest.approx(arcdist_deg(7.0, 13.0, red.reducers["g01"].lam_over_d_px))
    assert mw["n_frames"] == m0["n_frames"]
    r = red.reducers["g01"]
    p = dict(_cfg(red).params_for("g01"))
    assert r.cost_estimate(dict(p, zone_in=3.0, zone_out=2.0)) > r.cost_estimate(p)
    # the outer edge stops at the reducer's cap
    capped, mc = _one(red, zone_out=99.0)
    assert mc["outrad"] == r.outrad_cap


def test_post_hp_off_skips_the_residuals_second_high_pass():
    red, _ = _reducer()
    base, _ = _one(red)
    on, _ = _one(red, post_hp=1)
    off, _ = _one(red, post_hp=0)
    np.testing.assert_array_equal(base, on)
    fin = np.isfinite(base) & np.isfinite(off)
    assert fin.sum() > 100 and not np.allclose(base[fin], off[fin])
    # without a high-pass at all the switch has nothing to switch
    a, _ = _one(red, filter=0, post_hp=1)
    b, _ = _one(red, filter=0, post_hp=0)
    np.testing.assert_array_equal(a, b)


def test_make_space_searches_them_when_asked():
    red, _ = _reducer()
    plain = ales.make_space(red)
    assert not {"zone_in", "zone_out", "post_hp"} & set(plain.names)
    sp = ales.make_space(red, zone_range=(0.0, 8.0), search_post_hp=True,
                         defaults={"zone_in": 6.6, "zone_out": 3.8, "post_hp": 0})
    i = {n: k for k, n in enumerate(sp.names)}
    assert sp.names.index("post_hp") < sp.names.index("sel_g00")
    assert (sp.params[i["zone_in"]].lo, sp.params[i["zone_in"]].hi) == (0.0, 8.0)
    assert (sp.params[i["post_hp"]].lo, sp.params[i["post_hp"]].hi) == (0.0, 1.0)
    cfg = sp.decode(sp.default_vector())
    assert cfg.params["zone_in"] == pytest.approx(6.6) and cfg.params["zone_out"] == pytest.approx(3.8)
    assert cfg.params["post_hp"] == 0
    assert ales.make_space(red, zone_range=(0, 8), search_post_hp=True).decode(
        ales.make_space(red, zone_range=(0, 8), search_post_hp=True).default_vector()).params["post_hp"] == 1


def test_a_tiny_coloured_search_with_the_paper_dimensions(tmp_path):
    red, lam = _reducer(spectrum=lambda w: w ** 4, spectrum_label="lambda^4")
    space = ales.make_space(red, defaults={"bin": 2, "k_klip": 3, "filter": 9, "zone_in": 2.0, "post_hp": 0},
                            k_klip_max=6, zone_range=(0.0, 4.0), search_post_hp=True)
    space.project = ales.make_guard(red, max_excluded=1, n_min_ref=3, ref_frac=0.5, k_max=6, prefer_m=4.0e-6)
    obj, samp = ales.default_config(red)
    cfg = RunConfig(ann_edges=[7, 13], n_iter=4, n_init=2, seed=2, fm_curve=False, param_verify=False,
                    validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[3e-3]))
    res = Runner(red, space, obj, samp, cfg, str(tmp_path / "run"), log=QUIET).run()
    assert len(res) == 1 and res[0].validated
    setup = json.load(open(tmp_path / "run" / "run_setup.json"))
    assert setup["reducer"]["injection_spectrum"]["label"] == "lambda^4"
    recs = [json.loads(l) for l in open(tmp_path / "run" / "results.jsonl")]
    seed = recs[0]["config"]["params"]
    assert seed["zone_in"] == 2.0 and seed["post_hp"] == 0


# ------------------------------------------------------------------------ the paper's geometry, held
def test_set_zone_center_moves_the_zones_and_none_restores_them():
    from klip_tpe.reducer import set_zone_center
    red, _ = _reducer()
    base, _ = _one(red)
    assert set_zone_center(red, (22.0, 22.0)) == red.partitions()
    assert all(r.defaults["zone_center"] == (22.0, 22.0) for r in red.reducers.values())
    moved, _ = _one(red)
    assert not np.array_equal(np.isfinite(base), np.isfinite(moved)), "the zone footprint moves with its centre"
    set_zone_center(red, None)
    assert all("zone_center" not in r.defaults for r in red.reducers.values())
    np.testing.assert_array_equal(_one(red)[0], base)
    with pytest.raises(ValueError, match="zone_center"):
        set_zone_center(red, (1.0, 2.0, 3.0))


def test_pin_holds_reduction_parameters_out_of_the_search():
    red, _ = _reducer()
    plain = ales.make_space(red)
    sp = ales.make_space(red, search_post_hp=True, pin={"n_ang": 6, "zone_in": 6.6, "zone_out": 3.8})
    assert "n_ang" in plain.names and "n_ang" not in sp.names
    assert not {"zone_in", "zone_out"} & set(sp.names) and "post_hp" in sp.names
    assert len(sp.names) == len(plain.names), "n_ang out, post_hp in"
    cfg = sp.decode(sp.default_vector())
    for d in [cfg.params, *cfg.per_partition.values()]:
        assert (d["n_ang"], d["zone_in"], d["zone_out"]) == (6, 6.6, 3.8)
    # a pinned dimension of the zone search goes too
    sp2 = ales.make_space(red, zone_range=(0.0, 8.0), pin={"zone_in": 1.0})
    assert "zone_in" not in sp2.names and "zone_out" in sp2.names
    assert sp2.decode(sp2.default_vector()).params["zone_in"] == 1.0


def test_angsep_range_keeps_the_search_off_the_auto_fast_corner():
    from klip_tpe.klip import KLIPParams, klip_annular
    red, _ = _reducer()
    plain = ales.make_space(red, defaults={"angsep": 0.0})
    sp = ales.make_space(red, defaults={"angsep": 0.0}, angsep_range=(0.01, 3.0))
    assert sp.names == plain.names
    i = sp.index("angsep")
    assert (plain.params[i].lo, sp.params[i].lo, sp.params[i].hi) == (0.0, 0.01, 3.0)
    # the default is clipped into the range, and a proposal past the floor (or the reference-count
    # guard's 0) onto it, as the runner sanitizes after the guard
    assert sp.decode(sp.default_vector()).params["angsep"] == pytest.approx(0.01)
    x = sp.default_vector()
    x[i] = -0.5
    a_plain, a_floor = plain.sanitize(x)[i], sp.sanitize(x)[i]
    assert a_plain == 0.0 and a_floor == pytest.approx(0.01)
    # 0 with no anglemax window is the shared basis that holds every frame; the floor is per-target KLIP
    rng = np.random.default_rng(0)
    cube, ang = rng.standard_normal((12, 21, 21)), np.linspace(0.0, 40.0, 12)
    for v, fast in [(a_plain, True), (a_floor, False)]:
        _, info = klip_annular(cube, ang, KLIPParams(k_klip=3, inrad=3.0, outrad=9.0, angsep=v, anglemax=360.0,
                                                     n_min_ref=2), 2.0)
        assert info["auto_fast"] is fast and info["fast"] is fast
    for bad in [(-0.1, 3.0), (2.0, 1.0)]:
        with pytest.raises(ValueError, match="angsep_range"):
            ales.make_space(red, angsep_range=bad)


def test_getzone2015_zones_are_the_paper_reductions_pixels():
    """The 2015 getzone (MWC758_ALES reduce_ales_mwc758_2.pro): dist_circle and angmask about (sc/2 - 1,
    sc/2 - 1) in single precision, ang = atan2 + pi, zone jj = [jj, jj + 1] * 360 / n_ang deg, both ends
    inclusive, with annmode_inout = [5, 27].  zone_indices with that centre and 'getzone2015' angles
    gives the same pixels in every sector, where the default angles move one row of 23 pixels."""
    from klip_tpe.klip import zone_indices
    n, c, n_ang = 62, np.float32(30.0), 6
    x2 = (np.arange(n, dtype=np.float32) - c) ** 2
    r = np.sqrt(x2[None, :] + x2[:, None]).astype(np.float32)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    ang = (np.arctan2(yy - c, xx - c).astype(np.float32) + np.float32(np.pi)).astype(np.float32)
    dtor, step = np.float32(np.pi / 180.0), np.float32(360.0) / np.float32(n_ang)
    idl = [np.flatnonzero(((ang >= np.float32(np.float32(j) * step) * dtor) & (ang <= np.float32(np.float32(j + 1) * step) * dtor)
                           & (r >= 5) & (r <= 27)).ravel()) for j in range(n_ang)]
    ours = [zone_indices((n, n), 5.0, 27.0, j * 60.0, (j + 1) * 60.0, (30.0, 30.0), "getzone2015") for j in range(n_ang)]
    for j in range(n_ang):
        np.testing.assert_array_equal(ours[j], idl[j])
    ccw = [zone_indices((n, n), 5.0, 27.0, j * 60.0, (j + 1) * 60.0, (30.0, 30.0)) for j in range(n_ang)]
    assert sum(len(set(ccw[(j + 3) % 6].tolist()) ^ set(idl[j].tolist())) for j in range(n_ang)) == 46
    with pytest.raises(ValueError, match="zone angles"):
        zone_indices((n, n), 5.0, 27.0, 0.0, 60.0, (30.0, 30.0), "cw")


def test_the_reducer_draws_its_zones_as_set():
    from klip_tpe.reducer import set_zone_center
    red, _ = _reducer()
    set_zone_center(red, (24.0, 24.0), angles="getzone2015")
    assert all(r.defaults["zone_center"] == (24.0, 24.0) and r.defaults["zone_angles"] == "getzone2015"
               for r in red.reducers.values())
    a, _ = _one(red, n_ang=4)
    set_zone_center(red, (24.0, 24.0))                       # the default angles, the same centre
    assert all("zone_angles" not in r.defaults for r in red.reducers.values())
    b, _ = _one(red, n_ang=4)
    assert np.array_equal(np.isfinite(a), np.isfinite(b)), "the angles move sectors, not the zone"
    with pytest.raises(ValueError, match="zone angles"):
        set_zone_center(red, (24.0, 24.0), angles="cw")
