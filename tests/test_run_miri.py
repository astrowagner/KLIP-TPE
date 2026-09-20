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


@pytest.fixture
def stub_stpsf(monkeypatch):
    """A MIRI library without STPSF: right class, right flag, cheap stamps."""
    g = synthetic_map()
    monkeypatch.setattr(miri, "throughput_map", lambda *a, **k: g)

    def fake_library(filter="F1065C", star_flux=1.0, seps_as=None, date=None, log=print, **kw):
        seps = np.asarray(seps_as if seps_as is not None else np.geomspace(0.3, 8.0, 10), float)
        sl = np.zeros((seps.size, 15, 15))
        sl[:, 7, 7] = 1.0
        return miri.MIRILibraryPSF(sl, seps, center=(7.0, 7.0), ee_radius_px=3.0,
                                   refpa_deg=0.0, flux_unit=float(star_flux or 1.0),
                                   thru2d=miri.throughput_map_fn(g))

    monkeypatch.setattr(miri, "library", fake_library)
    monkeypatch.setattr(miri, "pixelscale", lambda *a, **k: 0.109655)
    return g


def _args(**kw):
    import run_miri
    base = dict(data=None, target="TARG", filter=None, partition="roll", crop=40, ann=None,
                known=None, star_flux=None, mode="ADI+RDI", min_throughput=0.30,
                mask_quadrants=True, n_iter=10, n_init=2, k_max=4, max_drop=0, out=None,
                seed=1, workers=1, check=True, default_only=False, fresh=False, show=False)
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


def test_build_masks_the_dead_zones_out_of_the_cubes(tree, stub_stpsf):
    run_miri, a = _args(data=str(tree))
    dsets, *_ = run_miri.build(a, log=lambda *_: None)
    for pid, ds in dsets.items():
        assert ds.meta.get("quadrant_masked_px", 0) > 0, f"{pid} was not masked"
        assert np.isnan(np.asarray(ds.cube)).any()


def test_no_mask_quadrants_leaves_the_pixels_in(tree, stub_stpsf):
    run_miri, a = _args(data=str(tree), mask_quadrants=False)
    dsets, _, _, _, _, samp, _, _ = run_miri.build(a, log=lambda *_: None)
    for ds in dsets.values():
        assert "quadrant_mask" not in (ds.meta or {})
    assert list(samp.forbidden_pa) == [], "sectors were forbidden with masking off"


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
