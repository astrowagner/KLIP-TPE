"""The LBTI/ALES adapter: an IFS cube as wavelength groups, with group flags.

From the ALES reduction's handoff of 2026-10-13 (WISPIT 2, 535 frames x 99 channels of
62 x 62 spaxels).  What must hold:

* the group table is the handoff's rule, section 6.2: constant-R groups over channels 6-88,
  which on WISPIT 2's wavelengths gives its 20-row table exactly;
* the loader reads ``(wavelength, frame, y, x)``, means each group's channels (NaN-aware),
  adds the 180 deg to the angles once, gives every group its own template, wavelength and
  FWHM, and keeps the group table in the run's setup file;
* the reducer's FWHM and matched filter are the representative group's, not the bluest;
* the science frames' high-pass is NaN-aware, so a frame with NaN at its edges is not lost;
* the group bound restores groups to at most ``max_excluded`` off, preferred groups first
  for a guided proposal and random ones for a random draw, and the all-off draw is left to
  decode to all on;
* the metric does not de-spike a critically sampled source's peak;
* ``--instrument ales`` reaches the loader through the command line, and a tiny run on a
  synthetic IFS cube goes end to end with the flags recorded.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from astropy.io import fits

from klip_tpe import CalibrationConfig, RunConfig, Runner, ValidationConfig, cli
from klip_tpe.instruments import ales
from klip_tpe.metrics import mawet_peak_snr, source_xy, star_center
from klip_tpe.space import Config

QUIET = (lambda s: None)

#: WISPIT 2's 99 channel wavelengths (um), ``WISPIT2_waves.fits``
WAVES = [2.7615, 2.7852, 2.8086, 2.8316, 2.8543, 2.8766, 2.8986, 2.9203, 2.9418, 2.9629, 2.9838, 3.0044,
         3.0248, 3.0449, 3.0648, 3.0845, 3.1039, 3.1232, 3.1422, 3.161, 3.1797, 3.1982, 3.2165, 3.2346,
         3.2525, 3.2703, 3.2879, 3.3054, 3.3227, 3.3398, 3.3569, 3.3737, 3.3905, 3.4071, 3.4236, 3.4399,
         3.4561, 3.4722, 3.4882, 3.5041, 3.5198, 3.5354, 3.551, 3.5664, 3.5817, 3.5969, 3.612, 3.627, 3.642,
         3.6568, 3.6715, 3.6861, 3.7007, 3.7151, 3.7295, 3.7438, 3.758, 3.7721, 3.7861, 3.8001, 3.814, 3.8278,
         3.8415, 3.8552, 3.8687, 3.8822, 3.8957, 3.9091, 3.9224, 3.9356, 3.9487, 3.9618, 3.9749, 3.9878,
         4.0008, 4.0136, 4.0264, 4.0391, 4.0518, 4.0644, 4.0769, 4.0894, 4.1019, 4.1142, 4.1266, 4.1388,
         4.1511, 4.1632, 4.1753, 4.1874, 4.1994, 4.2114, 4.2233, 4.2352, 4.247, 4.2588, 4.2705, 4.2822, 4.2938]

#: the handoff's table: (first channel, last channel, mean wavelength to 3 decimals)
HANDOFF_TABLE = [(6, 7, 2.910), (8, 10, 2.963), (11, 13, 3.025), (14, 16, 3.084), (17, 19, 3.142),
                 (20, 22, 3.198), (23, 25, 3.253), (26, 29, 3.314), (30, 33, 3.382), (34, 37, 3.448),
                 (38, 41, 3.512), (42, 45, 3.574), (46, 50, 3.642), (51, 54, 3.708), (55, 59, 3.772),
                 (60, 65, 3.848), (66, 70, 3.922), (71, 76, 3.994), (77, 82, 4.071), (83, 88, 4.145)]


def test_the_group_table_is_the_handoffs():
    table = ales.group_table(WAVES)
    assert [(g["channels"][0], g["channels"][-1]) for g in table] == [t[:2] for t in HANDOFF_TABLE]
    assert all(abs(g["lam_mean"] - t[2]) <= 6e-4 for g, t in zip(table, HANDOFF_TABLE))     # the table's 3 decimals
    assert [g["name"] for g in table] == [f"g{i:02d}" for i in range(20)]
    assert sum(g["n"] for g in table) == 83 and table[0]["lam_lo"] == pytest.approx(0.5 * (WAVES[5] + WAVES[6]))
    assert table[-1]["lam_hi"] == pytest.approx(0.5 * (WAVES[88] + WAVES[89]))
    # 15 groups: one R = 40 element each, still a partition of the channels
    t15 = ales.group_table(WAVES, ngrp=15)
    assert len(t15) == 15 and sorted(c for g in t15 for c in g["channels"]) == list(range(6, 89))
    with pytest.raises(ValueError):
        ales.group_table(WAVES, c0=6, c1=88, ngrp=100)
    with pytest.raises(ValueError):
        ales.group_table(WAVES, c0=0, c1=99)


# ---------------------------------------------------------------------- a synthetic IFS cube
NW, NF, N = 12, 40, 48          # channels, frames, spaxels


def _ifs(tmp_path, planet=None, edge_nan=True):
    """``(NW, NF, N, N)`` cube of a speckle-free star with a halo, noise, NaN edges on some
    frames, PA -20..+24 like WISPIT 2; a template on a 50 x 54 grid with the star at (27, 25).
    ``planet``: ``(rho_spx, pa_deg, contrast)`` planted in every channel with the handoff's
    sense of rotation (the source moves clockwise on the detector as LBT_PARA grows)."""
    rng = np.random.default_rng(7)
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
            fr = star + rng.standard_normal((N, N)) * 3.0
            if planet is not None:
                rho, pa_sky, c = planet
                az = np.deg2rad(pa_sky - pa[f] - 180.0 + 90.0)       # azimuth on the detector, 'pa' convention
                px, py = cx + rho * np.cos(az), cy + rho * np.sin(az)
                fr += c * amp * np.exp(-0.5 * ((xx - px) ** 2 + (yy - py) ** 2) / sg ** 2)
            if edge_nan and f % 3 == 0:
                fr[:2, :] = np.nan                               # the registration edge
            cube[w, f] = fr
    tpl[3, 10, 10] = np.nan                                      # a dead template pixel
    d = tmp_path / "products"
    d.mkdir(exist_ok=True)
    hdr = fits.Header({"PIXSCALE": 0.0345})
    fits.writeto(d / "SYN_cube_cen_clean.fits", cube, hdr, overwrite=True)
    fits.writeto(d / "SYN_primary_angles_clean.fits", pa.astype(np.float32), overwrite=True)
    fits.writeto(d / "SYN_waves.fits", waves.astype(np.float32), overwrite=True)
    fits.writeto(d / "SYN_star_psf_cube.fits", tpl, fits.Header({"STARX": 27, "STARY": 25}), overwrite=True)
    fits.writeto(d / "REF_cube_cen_clean.fits", cube[:, :10] * 0.8, hdr, overwrite=True)
    return str(d), waves, pa, cube, tpl


def test_the_loader_reads_the_ifs_cube_as_groups(tmp_path):
    d, waves, pa, cube, tpl = _ifs(tmp_path)
    ds, lam, fw, table = ales.load_ales(d, "SYN", c0=1, c1=10, ngrp=4, cache=str(tmp_path / "cache"), log=QUIET)
    assert list(ds) == ["g00", "g01", "g02", "g03"] and sum(g["n"] for g in table) == 10
    g = table[1]
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            want = np.nanmean(cube[g["channels"]], axis=0)
    np.testing.assert_allclose(ds["g01"].cube, want.astype(np.float32), rtol=1e-6, atol=1e-3)
    assert np.isnan(ds["g01"].cube[0, 0, 0]) and not np.isnan(ds["g01"].cube[1, 0, 0]), "NaN edges kept"
    assert lam["g01"] == pytest.approx(g["lam_mean"] * 1e-6)
    assert fw["g01"] == pytest.approx(1.03 * (lam["g01"] / 8.4) * 206265.0 / 0.0345)
    np.testing.assert_allclose(ds["g01"].angles, pa + 180.0)
    assert ds["g01"].meta["angle_offset"] == 180.0 and ds["g01"].meta["psf_center"] == (27.0, 25.0)
    assert ds["g01"].texp == pytest.approx(NF * ales.ALES.texp_frame)
    t = ds["g01"].meta["psf"]
    assert t.shape == (50, 54) and np.isfinite(t).all() and np.argmax(t) == np.ravel_multi_index((25, 27), t.shape)
    p = ds["g01"].meta["provenance"]
    assert p["channels"] == g["channels"] and p["lam_mean_um"] == g["lam_mean"] and p["instrument"] == "ALES"
    assert ds["g01"].tags is not None and set(ds["g01"].tags) == {"corrs", "noises", "coronoise"}
    # the cache holds every group and a second load reads it without the cube
    assert sorted(os.listdir(tmp_path / "cache")) == [f"SYN_c1-10_n4_g{i:02d}.npy" for i in range(4)]
    os.rename(os.path.join(d, "SYN_cube_cen_clean.fits"), os.path.join(d, "away.fits"))
    ds2, _, _, _ = ales.load_ales(d, "SYN", c0=1, c1=10, ngrp=4, cache=str(tmp_path / "cache"), log=QUIET)
    np.testing.assert_array_equal(ds2["g02"].cube, ds["g02"].cube)
    os.rename(os.path.join(d, "away.fits"), os.path.join(d, "SYN_cube_cen_clean.fits"))
    # a subset, and a reference star as each group's library
    ds3, _, _, t3 = ales.load_ales(d, "SYN", ref="REF", c0=1, c1=10, ngrp=4, only=["g02", "g03"], log=QUIET)
    assert list(ds3) == ["g02", "g03"] and [g["name"] for g in t3] == ["g02", "g03"]
    assert ds3["g02"].ref_cube.shape == (10, N, N)
    with pytest.raises(ValueError, match="none of"):
        ales.load_ales(d, "SYN", c0=1, c1=10, ngrp=4, only=["g09"], log=QUIET)


def test_the_reducer_takes_the_representative_groups_fwhm_and_filter(tmp_path):
    d, waves, pa, cube, tpl = _ifs(tmp_path)
    ds, lam, fw, table = ales.load_ales(d, "SYN", c0=1, c1=10, ngrp=4, log=QUIET)
    red = ales.make_reducer(ds, lam, fw, max_workers=1, table=table, log=QUIET)
    rep = ales.representative_group(lam)
    assert rep in ("g01", "g02") and red.representative == rep
    assert red.fwhm == pytest.approx(fw[rep]) and red.fwhm > red.reducers["g00"].fwhm
    assert red.lam_over_d_px == pytest.approx(red.reducers[rep].lam_over_d_px)
    # a template model supplies no matched filter, so the metric filters with the Gaussian of
    # the representative group's FWHM; the kernel the reducer offers is that group's model's
    assert red.matched_filter_kernel(0.3) is None
    assert red.partition_label == "group" and red.provenance["groups"][0]["name"] == "g00"
    assert all(r.defaults["hp_nan_aware"] for r in red.reducers.values())
    # the flux unit is the group template's flux in 3 spaxels of its star, so a contrast is a flux ratio
    t0 = ds["g00"].meta["psf"]
    ty, tx = np.mgrid[0:t0.shape[0], 0:t0.shape[1]]
    assert red.reducers["g00"].model.flux_unit == pytest.approx(t0[np.hypot(tx - 27, ty - 25) <= 3.0].sum(), rel=1e-6)

    class _K:
        def __init__(self, v):
            self.v = v

        def matched_filter_kernel(self, rho_as, fwhm):
            return np.full((3, 3), self.v)
    red.reducers["g00"].model, red.reducers[rep].model = _K(1.0), _K(2.0)
    assert red.matched_filter_kernel(0.3)[0, 0] == 2.0
    red = ales.make_reducer(ds, lam, fw, max_workers=1, table=table, log=QUIET)
    assert red.describe()["representative"] == rep and red.describe()["provenance"]["representative"] == rep
    red2 = ales.make_reducer(ds, lam, fw, max_workers=1, representative="g03", log=QUIET)
    assert red2.fwhm == pytest.approx(fw["g03"])
    with pytest.raises(KeyError):
        ales.make_reducer(ds, lam, fw, max_workers=1, representative="g09", log=QUIET)
    obj, samp = ales.default_config(red, known=[(0.3, 240.0)], forbidden_pa=[(240.0, 25.0)])
    assert obj.metric.declip_nsig is None and obj.metric.fwhm == pytest.approx(red.fwhm)
    assert samp.fwhm_as == pytest.approx(red.fwhm * 0.0345) and samp.known == [(0.3, 240.0)]
    assert samp.forbidden_pa == [(240.0, 25.0)]


def _zone(cfg, zone):
    params = dict(cfg.params, inrad=zone[0], outrad=zone[1], pad_in=True, pad_out=True)
    per = {p: dict(d, inrad=zone[0], outrad=zone[1], pad_in=True, pad_out=True) for p, d in cfg.per_partition.items()}
    return Config(params, per, cfg.selected, cfg.x)


def test_nan_edges_do_not_empty_the_bins_and_the_planet_lands_where_it_should(tmp_path):
    """The handoff's convention check on a synthetic planet at 10 spaxels, PA 240."""
    d, waves, pa, cube, tpl = _ifs(tmp_path, planet=(10.0, 240.0, 3e-3))
    ds, lam, fw, table = ales.load_ales(d, "SYN", c0=1, c1=10, ngrp=4, log=QUIET)
    red = ales.make_reducer(ds, lam, fw, max_workers=1, table=table, log=QUIET)
    space = ales.make_space(red, defaults={"bin": 2, "k_klip": 3, "filter": 9})
    assert space.names[:9] == ["bin", "n_ang", "filter", "angsep", "anglemax", "corr_thresh", "noise_max",
                               "coronoise_max", "k_klip"]
    assert space.names[9:] == ["sel_g00", "sel_g01", "sel_g02", "sel_g03"]
    cfg = space.decode(space.default_vector())
    assert cfg.params["bin"] == 2 and cfg.params["corr_thresh"] == 0.0 and cfg.params["noise_max"] == 3.0
    ev = red.reduce_config(_zone(cfg, (7.0, 13.0)))
    # every frame kept: one in three frames has a NaN edge, which the IDL boxcar would have spread
    assert all(m["n_frames"] == NF // 2 for m in ev.meta.values()), {k: m["n_frames"] for k, m in ev.meta.items()}
    img = ev.image
    cx, cy = star_center(img.shape)
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.hypot(xx - cx, yy - cy)
    ann = (rr >= 7) & (rr <= 13) & np.isfinite(img)
    iy, ix = np.unravel_index(np.argmax(np.where(ann, img, -np.inf)), img.shape)
    x0, y0 = source_xy(10.0 * 0.0345, 240.0, 0.0345, cx, cy)
    assert abs(ix - x0[0]) <= 1 and abs(iy - y0[0]) <= 1, ((ix, iy), (x0[0], y0[0]))
    # the IDL high-pass on the same frames: the bins with a NaN frame are gone
    for r in red.reducers.values():
        r.defaults["hp_nan_aware"] = False
    ev0 = red.reduce_config(_zone(cfg, (7.0, 13.0)))
    assert all(m["n_frames"] < NF // 2 for m in ev0.meta.values())


def test_the_group_bound_restores_groups(tmp_path):
    d, waves, pa, cube, tpl = _ifs(tmp_path)
    ds, lam, fw, table = ales.load_ales(d, "SYN", c0=1, c1=10, ngrp=4, log=QUIET)
    red = ales.make_reducer(ds, lam, fw, max_workers=1, log=QUIET)
    space = ales.make_space(red)
    g = ales.GroupCountGuard(1, lam_m=lam, prefer_m=lam["g02"])
    assert g.prefer[0] == "g02" and g.describe()["max_excluded"] == 1
    x = space.default_vector({"sel_g00": 0, "sel_g02": 0, "sel_g03": 0})          # three off
    y = g(x, space)
    assert space.selected_partitions(y) == ["g01", "g02", "g03"], "g02 first, then g03 (nearest), g00 stays off"
    assert g.last_restored == 2
    assert np.array_equal(g(space.default_vector({"sel_g00": 0}), space), space.default_vector({"sel_g00": 0}))
    # the all-off draw decodes to all on and is left alone
    allx = space.default_vector({f"sel_g0{i}": 0 for i in range(4)})
    assert space.selected_partitions(allx) == ["g00", "g01", "g02", "g03"] and np.array_equal(g(allx, space), allx)
    # a random draw restores random groups, a guided one the preferred ones
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(40):
        seen.add(tuple(space.selected_partitions(g(x, space, rng=rng, is_random=True))))
    assert len(seen) > 1 and all(len(s) == 3 for s in seen)
    # no bound: untouched; the composed guard of make_guard carries it first
    assert np.array_equal(ales.GroupCountGuard(10)(x, space), x)
    proj = ales.make_guard(red, max_excluded=1, prefer_m=lam["g02"], n_min_ref=2, ref_frac=0.5)
    assert proj.describe()["steps"][0]["name"] == "group_count_guard"
    assert space.selected_partitions(proj(x, space, rng=rng, is_random=False, zone=(7.0, 13.0))) == ["g01", "g02", "g03"]
    none = ales.make_guard(red, max_excluded=None, n_min_ref=2, ref_frac=0.5)
    assert [s["name"] for s in none.describe()["steps"]][0] != "group_count_guard"


def test_the_metric_keeps_a_critically_sampled_peak():
    """A 2.6 px FWHM source on noise: the 6 sigma de-spiking takes its peak for a hot pixel."""
    n, fwhm, px = 81, 2.57, 0.0345
    rng = np.random.default_rng(1)
    cx, cy = star_center((n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    x0, y0 = source_xy(0.3, 240.0, px, cx, cy)
    img = 0.2 * rng.standard_normal((n, n)) + 3.0 * np.exp(-0.5 * ((xx - x0[0]) ** 2 + (yy - y0[0]) ** 2) / (fwhm / 2.3548) ** 2)
    with_declip = mawet_peak_snr(img, 0.3, 240.0, px, fwhm, declip_nsig=6.0)[0]
    without = mawet_peak_snr(img, 0.3, 240.0, px, fwhm, declip_nsig=None)[0]
    assert without > 1.3 * with_declip, (with_declip, without)


def _args(*extra):
    return cli._build_parser().parse_args(["near", "--instrument", "ales", "--root", "products"] + list(extra))


def test_the_command_line_reaches_the_loader(tmp_path, monkeypatch):
    a = _args("--ales-groups", "4", "--ales-channels", "1", "10", "--ales-only", "g01", "g02",
              "--ales-max-excluded", "3", "--known", "0.3", "240", "--forbidden-pa", "240", "25", "--bin-min", "2")
    assert a.instrument == "ales" and a.ales_groups == 4 and a.ales_channels == [1, 10] and a.ales_only == ["g01", "g02"]
    assert a.forbidden_pa == [[240.0, 25.0]] and a.bin_min == 2 and a.ales_target == "WISPIT2"
    d, waves, pa, cube, tpl = _ifs(tmp_path)
    a = _args("--ales-target", "SYN", "--ales-groups", "4", "--ales-channels", "1", "10", "--known", "0.3", "240",
              "--forbidden-pa", "240", "25", "--workers", "1", "--bin-min", "2", "--ales-max-excluded", "2")
    a.root = [d]
    red, obj, samp, guard, thr = cli._near_objects(a, log=QUIET)
    assert red.partitions() == ["g00", "g01", "g02", "g03"] and red.name == "ales"
    assert obj.metric.known == [(0.3, 240.0)] and samp.forbidden_pa == [(240.0, 25.0)] and obj.metric.declip_nsig is None
    steps = [s["name"] for s in guard.describe()["steps"]]
    assert steps[:2] == ["group_count_guard", "min_bin_guard"] and guard.describe()["steps"][0]["max_excluded"] == 2
    space = cli.build_near_space(red, a)
    assert space.selection == "binary" and len(space.selection_dims) == 4
    assert cli._default_run_dir(a).startswith("runs" + os.sep)


@pytest.mark.slow
def test_a_tiny_search_on_the_synthetic_ifs_cube(tmp_path):
    d, waves, pa, cube, tpl = _ifs(tmp_path)
    ds, lam, fw, table = ales.load_ales(d, "SYN", c0=1, c1=10, ngrp=4, log=QUIET)
    red = ales.make_reducer(ds, lam, fw, max_workers=1, table=table, log=QUIET)
    space = ales.make_space(red, defaults={"bin": 2, "k_klip": 3, "filter": 9}, k_klip_max=6)
    space.project = ales.make_guard(red, max_excluded=1, n_min_ref=3, ref_frac=0.5, k_max=6)
    obj, samp = ales.default_config(red)
    cfg = RunConfig(ann_edges=[7, 13], n_iter=5, n_init=2, seed=2, fm_curve=False, param_verify=False,
                    validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[3e-3]))
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "run"), log=QUIET)
    res = r.run()
    assert len(res) == 1 and res[0].validated
    setup = json.load(open(tmp_path / "run" / "run_setup.json"))
    assert [g["name"] for g in setup["reducer"]["provenance"]["groups"]] == ["g00", "g01", "g02", "g03"]
    assert setup["reducer"]["partitions"]["g01"]["frames"]["channels"] == table[1]["channels"]
    assert setup["projection"]["steps"][0] == {"name": "group_count_guard", "max_excluded": 1,
                                               "prefer": ales.GroupCountGuard(1, lam_m=lam).prefer,
                                               "prefer_um": 3.7, "random_redraw": True}
    recs = [json.loads(l) for l in open(tmp_path / "run" / "results.jsonl")]
    assert recs and all(len(rec["meta"]["selected"]) >= 3 for rec in recs), "at most one group off"
    flags = {n: recs[0]["x"][space.names.index(f"sel_{n}")] for n in red.partitions()}
    assert flags == {"g00": 1.0, "g01": 1.0, "g02": 1.0, "g03": 1.0}, "the seed includes every group"
    assert os.path.exists(tmp_path / "run" / "klip_stitched.fits")
