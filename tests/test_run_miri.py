"""The MIRI driver, end to end, on synthetic ``calints``.

``scripts/run_miri.py`` is where the MIRI pieces have to meet: ``load_calints`` (which is
instrument-agnostic and knows nothing about quadrant boundaries), the dead-zone mask, the
2-D throughput model, and the forbidden position-angle sectors the sampler needs.  Each
of those is unit-tested in ``test_miri.py`` against a synthetic throughput map; what is
left, and what these tests cover, is whether the driver actually wires them together --
the failure mode being a run that completes and reports contrasts computed with a radial
throughput on a four-quadrant mask.

The files are written here rather than fetched, so this needs no data and no network.
STPSF is stubbed out: the injection model's *identity* is what matters to the wiring, not
the pixels in its stamps.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from klip_tpe.instruments import miri                                    # noqa: E402


def synthetic_map(width=8.0, floor=0.15, top=0.95):
    seps = np.geomspace(0.3, 8.0, 10)
    az = miri.default_azimuths()
    d = np.min(np.abs(((az[None, :] - np.array([0.0, 90.0, 180.0, 270.0])[:, None] + 180.0)
                       % 360.0) - 180.0), axis=0)
    prof = floor + (top - floor) * np.clip(d / width, 0.0, 1.0)
    return {"seps": seps, "az": az, "trans": np.tile(prof, (seps.size, 1)),
            "pxscale": 0.109655, "fwhm_px": 3.1, "ee_radius_px": 4.0, "meta": {}}


def write_calints(path, target, roll, n_int=3, ny=120, nx=120, filt="F1065C", seed=0):
    """One stage-2 product, with the keywords ``load_calints`` actually reads."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    c = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    r = np.hypot(xx - c[0], yy - c[1])
    halo = 500.0 / (1.0 + (r / 6.0) ** 2)
    sci = np.array([halo + rng.normal(0.0, 1.0, (ny, nx)) for _ in range(n_int)], np.float32)
    dq = np.zeros_like(sci, np.int32)
    dq[:, 5, 5] = 1                                      # one DO_NOT_USE pixel to fill
    ph = fits.Header({"TARGPROP": target, "INSTRUME": "MIRI", "FILTER": filt,
                      "CORONMSK": "4QPM_1065", "EXP_TYPE": "MIR_4QPM"})
    sh = fits.Header({"ROLL_REF": float(roll), "V3I_YANG": 0.0, "VPARITY": 1,
                      "PIXAR_A2": 0.109655 ** 2, "PIXAR_SR": 2.8e-13,
                      "CRPIX1": c[0] + 1, "CRPIX2": c[1] + 1, "BUNIT": "MJy/sr",
                      "FILTER": filt, "INSTRUME": "MIRI", "EFFINTTM": 10.0})
    fits.HDUList([fits.PrimaryHDU(header=ph),
                  fits.ImageHDU(sci, header=sh, name="SCI"),
                  fits.ImageHDU(dq, name="DQ")]).writeto(path, overwrite=True)
    return path


@pytest.fixture
def tree(tmp_path):
    d = tmp_path / "miri"
    d.mkdir()
    write_calints(d / "jw00001001_calints.fits", "TARG", roll=0.0, seed=1)
    write_calints(d / "jw00001002_calints.fits", "TARG", roll=10.0, seed=2)
    write_calints(d / "jw00001003_calints.fits", "REFSTAR", roll=0.0, seed=3)
    return d


def _stub_library(g):
    def fake_library(filter="F1065C", star_flux=1.0, seps_as=None, date=None, log=print, **kw):
        seps = np.asarray(seps_as if seps_as is not None else np.geomspace(0.3, 8.0, 10), float)
        sl = np.zeros((seps.size, 15, 15))
        sl[:, 7, 7] = 1.0
        return miri.MIRILibraryPSF(sl, seps, center=(7.0, 7.0), ee_radius_px=3.0,
                                   refpa_deg=0.0, flux_unit=float(star_flux or 1.0),
                                   thru2d=miri.throughput_map_fn(g))
    return fake_library


@pytest.fixture
def stub_stpsf(monkeypatch):
    """A MIRI library without STPSF: right class, right flag, cheap stamps."""
    g = synthetic_map()
    monkeypatch.setattr(miri, "throughput_map", lambda *a, **k: g)
    monkeypatch.setattr(miri, "library", _stub_library(g))
    monkeypatch.setattr(miri, "pixelscale", lambda *a, **k: 0.109655)
    return g


def _args(**kw):
    import run_miri
    base = dict(data=None, target="TARG", filter=None, partition="roll", crop=40, ann=None,
                known=None, star_flux=None, mode="ADI+RDI", min_throughput=0.30,
                dead_zones=True, nan_dead_zones=False, n_iter=10, n_init=2, k_max=4,
                max_drop=0, out=None, seed=1, workers=1, check=True, default_only=False,
                fresh=False, show=False)
    base.update(kw)
    return run_miri, type("A", (), base)()


def test_build_wires_the_two_dimensional_model_into_the_reducer(tree, stub_stpsf):
    """The failure this exists for: a run that completes and reports contrasts computed
    with a radial throughput on a four-quadrant mask."""
    run_miri, a = _args(data=str(tree))
    dsets, info, red, ann, obj, samp, m, px = run_miri.build(a, log=lambda *_: None)
    model = red.reducers[list(red.reducers)[0]].model
    assert isinstance(model, miri.MIRILibraryPSF)
    assert model.azimuth_dependent is True
    assert m["filter"] == "F1065C" and m["image_mask"] == "FQPM1065"
    assert px == pytest.approx(0.109655, abs=1e-5)


def test_the_dead_zones_leave_the_noise_estimate_and_stay_in_the_cube(tree, stub_stpsf):
    """The regression this file exists for now.

    The first version NaN'd the dead zones into the cube.  The reducer high-passes at
    ``nan_aware=False``, which spreads each NaN over a box the filter's width, and the 4QPM
    dead zones are lines through the star reaching the frame edges -- so at the default
    width of 12 px every frame was dropped as empty and the run died before its first
    evaluation.  The pixels are attenuated measurements, not missing ones: they belong in
    the cube and out of the statistic.
    """
    run_miri, a = _args(data=str(tree))
    dsets, _, _, _, obj, _, _, _ = run_miri.build(a, log=lambda *_: None)
    for pid, ds in dsets.items():
        assert np.isfinite(np.asarray(ds.cube, float)).all(), f"{pid} has NaNs in the cube"
    pm = obj.metric.pixel_mask
    assert pm is not None and pm.sum() > 0, "the dead zones are still in the noise estimate"
    assert obj.metric.describe()["pixel_mask_px"] == int(pm.sum())


def test_the_pixel_mask_and_the_forbidden_sectors_are_one_geometry(tree, stub_stpsf):
    """Doing only one of the two is worse than doing neither: injections avoiding sectors
    that still inflate the ring sigma are scored against a noise level nothing is measuring
    them at.  So the mask has to agree with the sectors where they are both defined."""
    from klip_tpe.metrics import pa_wedge_mask
    run_miri, a = _args(data=str(tree))
    dsets, _, red, ann, obj, samp, m, px = run_miri.build(a, log=lambda *_: None)
    pm = obj.metric.pixel_mask
    mid = 0.5 * (ann[0] + ann[1])
    ny, nx = pm.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
    ring = np.abs(r - mid) <= 1.0                                 # where fpa was evaluated
    sect = pa_wedge_mask(pm.shape, list(samp.forbidden_pa))
    agree = (pm == sect)[ring].mean()
    assert agree > 0.9, f"mask and sectors disagree on {100 * (1 - agree):.0f}% of the ring"


def test_no_dead_zones_puts_them_back_in(tree, stub_stpsf):
    run_miri, a = _args(data=str(tree), dead_zones=False)
    _, _, _, _, obj, samp, _, _ = run_miri.build(a, log=lambda *_: None)
    assert obj.metric.pixel_mask is None
    assert list(samp.forbidden_pa) == [], "sectors were forbidden with the exclusion off"


def test_nan_dead_zones_is_opt_in_and_says_what_it_breaks(tree, stub_stpsf):
    """Kept for a no-filter reduction, and only that.  A cube carrying NaN dead zones plus
    the default high-pass is the failure above, so --check must not let it pass quietly."""
    run_miri, a = _args(data=str(tree), nan_dead_zones=True)
    dsets, *_ = run_miri.build(a, log=lambda *_: None)
    assert any(np.isnan(np.asarray(ds.cube, float)).any() for ds in dsets.values())
    for ds in dsets.values():
        assert ds.meta.get("quadrant_masked_px", 0) > 0


def test_one_nan_pixel_is_enough_to_lose_most_of_the_frame():
    """The mechanism behind all of the above, and it is worse than "spreads by the filter
    width".

    ``highpass(nan_aware=False)`` is ``ndimage.uniform_filter``, whose running-sum
    implementation poisons everything downstream of the first NaN along each axis -- not a
    13x13 box.  One NaN in the corner of an 81x81 frame takes three quarters of it; the MIRI
    dead zones reach the frame edges along all four axes, so they take all of it.  Nothing
    in the pipeline notices: ``np.nansum`` of an all-NaN frame is 0.0 and ``bin_frames``
    drops zero-sum bins.
    """
    from klip_tpe.klip import highpass
    img = np.ones((81, 81))
    img[3, 3] = np.nan
    assert np.mean(~np.isfinite(highpass(img, 12, nan_aware=False))) > 0.70
    # nan_aware=True is the contrast: only the input pixel stays NaN.  The reducer does not
    # use it on the science cube, which is why the cube must arrive finite.
    assert np.count_nonzero(~np.isfinite(highpass(img, 12, nan_aware=True))) == 1


def test_nan_dead_zones_costs_most_of_the_frame_at_the_default_filter(tree, monkeypatch):
    """Why ``--nan-dead-zones`` is opt-in and not what the driver does.

    The cube it produces is not merely missing its dead zones; after the default width-12
    high-pass most of the frame is gone.  On the real F1140C mask it is all of it, and the
    reducer raises -- the guard below is what keeps that failure legible.
    """
    from klip_tpe.klip import highpass
    g = synthetic_map(width=60.0)                     # T < 0.3 within ~11 deg of an axis
    monkeypatch.setattr(miri, "throughput_map", lambda *a, **k: g)
    monkeypatch.setattr(miri, "pixelscale", lambda *a, **k: 0.109655)
    monkeypatch.setattr(miri, "library", _stub_library(g))
    run_miri, a = _args(data=str(tree), nan_dead_zones=True)
    dsets, *_ = run_miri.build(a, log=lambda *_: None)
    cube = np.asarray(next(iter(dsets.values())).cube, float)
    assert np.mean(~np.isfinite(cube)) < 0.25, "the mask itself takes only a fifth"
    assert np.mean(~np.isfinite(highpass(cube[0], 12))) > 0.60, "the filter takes the rest"


def test_the_reducer_still_refuses_an_all_nan_cube(tree, stub_stpsf):
    """The guard that made this diagnosable.  Without it every downstream shape goes to
    zero and the run fails somewhere far away and unrecognisable."""
    from klip_tpe.instruments import generic
    run_miri, a = _args(data=str(tree))
    dsets, _, red, *_ = run_miri.build(a, log=lambda *_: None)
    for r in red.reducers.values():
        r.data.cube[:] = np.nan
    space = generic.make_space(red, k_klip_max=4)
    with pytest.raises(ValueError, match="dropped as empty"):
        red.reduce_config(space.decode(space.default_vector({"filter": 12})), None, tag="boom")


def test_build_bars_injections_from_the_dead_sectors(tree, stub_stpsf):
    """An injection into a dead zone recovers nothing, and the search reads that as a
    property of the parameters it was trying."""
    run_miri, a = _args(data=str(tree))
    *_, samp, _, _ = run_miri.build(a, log=lambda *_: None)
    assert len(samp.forbidden_pa) >= 4, list(samp.forbidden_pa)
    # every forbidden centre is a quadrant boundary carried into sky angle by some roll
    for c, w in samp.forbidden_pa:
        assert w > 0
        assert min(abs(((c - b + 180) % 360) - 180) for b in (0, 10, 90, 100, 180, 190, 270, 280)) < 6


def test_library_lets_the_throughput_map_keep_its_own_separations(monkeypatch):
    """The stamp ladder and the map ladder answer different questions.

    Forcing the map onto the stamps' separations cut a map computed over 0.3-10.8 arcsec
    down to 0.2-3.0, missed a 576-PSF cache on a key that no longer matched, and left a
    map that clamped across most of the field.
    """
    seen = {}
    monkeypatch.setattr(miri, "throughput_map",
                        lambda *a, **k: (seen.update(k), synthetic_map())[1])
    monkeypatch.setattr(miri, "offaxis_grid",
                        lambda **k: (seen.update(stamp_seps=np.asarray(k["seps_as"])),
                                     {"slices": np.zeros((len(k["seps_as"]), 9, 9)),
                                      "seps": np.asarray(k["seps_as"]),
                                      "center": (4.0, 4.0), "ee_radius_px": 3.0})[1])
    miri.library("F1065C", seps_as=[0.5, 1.0, 1.5], log=lambda *_: None)
    assert "seps_as" not in seen, "the map was forced onto the stamps' separations"
    np.testing.assert_allclose(seen["stamp_seps"], [0.5, 1.0, 1.5])


def test_a_nircam_sequence_is_refused_rather_than_reduced_as_miri(tmp_path, stub_stpsf):
    """This driver applies MIRI mask geometry; pointing it at NIRCam would mask the wrong
    pixels and say nothing about it."""
    d = tmp_path / "nircam"
    d.mkdir()
    write_calints(d / "jw00002001_calints.fits", "TARG", roll=0.0, filt="F444W")
    run_miri, a = _args(data=str(d))
    with pytest.raises(SystemExit, match="not a MIRI coronagraphic filter"):
        run_miri.build(a, log=lambda *_: None)


def test_missing_data_says_where_to_get_it(tmp_path, stub_stpsf):
    run_miri, a = _args(data=str(tmp_path / "nothing-here"))
    with pytest.raises(SystemExit, match="fetch_jwst_ar"):
        run_miri.build(a, log=lambda *_: None)


def test_check_refuses_a_radial_model_on_a_four_quadrant_mask(tree, stub_stpsf, monkeypatch,
                                                              tmp_path):
    """--check has to catch exactly the silent failure, so make it happen.

    A radial model here would produce a run that finishes and reports numbers; the check
    is the only place that can tell the difference.
    """
    from klip_tpe.injection import LibraryPSF

    def radial_library(filter="F1065C", star_flux=1.0, seps_as=None, date=None, log=print, **kw):
        seps = np.asarray(seps_as if seps_as is not None else np.geomspace(0.3, 8.0, 10), float)
        sl = np.zeros((seps.size, 15, 15))
        sl[:, 7, 7] = 1.0
        return LibraryPSF(sl, seps, center=(7.0, 7.0), ee_radius_px=3.0,
                          throughput_fn=lambda r: 0.5, flux_unit=1.0)

    monkeypatch.setattr(miri, "library", radial_library)
    run_miri, a = _args(data=str(tree), out=str(tmp_path / "o"))
    with pytest.raises(SystemExit, match="azimuthal average"):
        run_miri.main([f"--data={tree}", "--target=TARG", "--check", "--crop=40",
                       f"--out={tmp_path / 'o'}", "--k-max=4", "--workers=1"])


def test_check_runs_the_whole_path_and_passes(tree, stub_stpsf, tmp_path):
    out = tmp_path / "chk"
    rc = None
    import run_miri
    rc = run_miri.main([f"--data={tree}", "--target=TARG", "--check", "--crop=40",
                        f"--out={out}", "--k-max=4", "--workers=1"])
    assert rc == 0
    log = (out / "run.log").read_text()
    assert "miri_library" in log
    assert "azimuth_dependent=True" in log
    assert "check passed" in log
