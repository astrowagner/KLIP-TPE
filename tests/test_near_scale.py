"""NEAR's plate scale per run, and the PSF library's own separations (NEAR2 note, 2026-10-31).

The astrometry of the 2026 cubes gives 0.04450"/px, where klip-tpe used 0.0456.  A new run takes
the new scale and the constants that follow it.  A run keeps the scale it was made with, so a
resume, a re-reduction or a check of a run made at 0.0456 reproduces it, and a run whose setup
does not record a scale was made at 0.0456.  The PSF library v2 gives each slice's separation in
its index; v1's index has a column of that name klip-tpe never read, so v1 keeps the 75-mas grid.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import pytest
from astropy.io import fits

from klip_tpe import cli
from klip_tpe.instruments import near
from klip_tpe.instruments.near import NEAR, N4Library, near_constants

NF, FULL = 12, 152


def test_the_constants_follow_the_scale_and_the_old_scale_gives_the_old_ones():
    old = near_constants(0.0456)
    lod = (11.25e-6 / 8.2) * 206265.0 / 0.0456
    assert old == {"pxscale": 0.0456, "lam_over_d_px": lod, "fwhm_px": 1.028 * (11.0e-6 / 8.2) * 206265.0 / 0.0456,
                   "fwhm_bin_px": 1.028 * lod}, "to the bit, by NEAR's formulas"
    new = near_constants()
    assert new["pxscale"] == NEAR.pxscale == 0.04450 and NEAR.legacy_pxscale == 0.0456
    assert new["lam_over_d_px"] == pytest.approx(6.360, abs=1e-3)
    assert new["fwhm_px"] == pytest.approx(6.392, abs=1e-3)
    assert new["fwhm_bin_px"] == pytest.approx(6.538, abs=1e-3)
    assert (NEAR.lam_over_d_px, NEAR.fwhm_px, NEAR.fwhm_bin_px) == (new["lam_over_d_px"], new["fwhm_px"],
                                                                     new["fwhm_bin_px"])


@pytest.fixture()
def root(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    for seq in (1, 2):
        d = tmp_path / f"n{seq}"
        d.mkdir()
        fits.writeto(d / "AB_cube_skysub_cen_clean.fits",
                     (100.0 + 5.0 * rng.standard_normal((NF, FULL, FULL))).astype(np.float32))
        np.savez(d / "AB_parang_clean.npz", angles=np.linspace(-40.0, 40.0, NF))
        np.savez(d / "AB_frametags.npz", corrs=np.full(NF, 0.99), noises=np.ones(NF), coronoise=np.ones(NF))
        np.savez(d / "texp.npz", texp=np.array([600.0]))
        for name in ("AB_frametags.sav", "texp.sav"):
            (d / name).write_bytes(b"")
        yy, xx = np.mgrid[0:16, 0:16]
        fits.writeto(d / "PSF_ACenB.fits", np.exp(-0.5 * ((xx - 7.5) ** 2 + (yy - 7.5) ** 2) / 2.6 ** 2))

    def readsav(path):
        z = np.load(os.path.splitext(path)[0] + ".npz")
        return {k: z[k] for k in z.files}
    monkeypatch.setattr(near, "_readsav", readsav)
    return str(tmp_path)


def _library(path, index=None, header=None):
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(1)
    cube = np.abs(rng.normal(size=(14, 41, 41))) + 1.0
    hdu = fits.PrimaryHDU(cube)
    hdu.header["PLATESCL"] = 0.0445
    for k, v in (header or {}).items():
        hdu.header[k] = v
    hdu.writeto(os.path.join(path, "n4_psf_cube_EEnorm.fits"))
    if index is not None:
        cols, rows = index
        with open(os.path.join(path, "n4_psf_cube_index.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
    return path


V2 = [0.1619, 0.2305, 0.3045, 0.3178, 0.3863, 0.4677, 0.5412, 0.6201, 0.7029, 0.7658, 0.8515, 0.9339, 1.0142,
      1.0176]


def test_the_v2_index_gives_the_separations_and_v1_keeps_the_grid(tmp_path):
    v2 = _library(str(tmp_path / "v2"), index=(["slice", "ut", "sep_arcsec", "v1_grid_mas"],
                                               [[k, "", s, 150 + 75 * k] for k, s in enumerate(V2)]))
    lib = N4Library(v2)
    assert lib.sep_source == "n4_psf_cube_index.csv" and len(lib.seps) == 13
    assert np.allclose(lib.seps, V2[:12] + [0.5 * (V2[12] + V2[13])]), "the last pair averaged, its label the mean"
    assert lib.describe()["seps_arcsec"][0] == 0.1619
    # v1: an index with a sep_arcsec column of other values, and no v1_grid_mas: the grid
    v1 = _library(str(tmp_path / "v1"), index=(["slice", "sep_mas", "sep_arcsec"],
                                               [[k, 256 + 40 * k, 0.256 + 0.04 * k] for k in range(14)]))
    old = N4Library(v1)
    assert old.sep_source == "75-mas grid (v1)"
    assert np.array_equal(old.seps, (np.arange(13) + 2) * 0.075)
    # the header's SEPnn when there is no v2 index
    hd = _library(str(tmp_path / "hd"), header={f"SEP{k:02d}": s for k, s in enumerate(V2)})
    assert N4Library(hd).sep_source == "header SEPnn"
    bad = _library(str(tmp_path / "bad"), header={f"SEP{k:02d}": 1.0 - 0.05 * k for k in range(14)})
    with pytest.raises(ValueError, match="do not increase"):
        N4Library(bad)


def test_a_reducer_takes_the_scale_and_library_it_is_given_and_records_them(root, tmp_path):
    lib = _library(os.path.join(root, "psflib_v2"), index=(["slice", "sep_arcsec", "v1_grid_mas"],
                                                           [[k, s, 0] for k, s in enumerate(V2)]))
    logs = []
    red = near.make_reducer(root, nights=(1, 2), psflib=lib, max_workers=1, log=logs.append)
    new = near_constants()
    assert red.pxscale == new["pxscale"] and red.fwhm == new["fwhm_px"]
    assert red.provenance["instrument"] == "NEAR" and red.provenance["pxscale"] == new["pxscale"]
    assert red.provenance["psflib"]["sep_source"] == "n4_psf_cube_index.csv"
    assert any("psflib: " in ln and "13 stamps at 0.162-1.016" in ln for ln in logs), logs
    assert red.describe()["provenance"]["fwhm_bin_px"] == new["fwhm_bin_px"]
    old = near.make_reducer(root, nights=(1, 2), pxscale=0.0456, max_workers=1, log=lambda s: None)
    c = near_constants(0.0456)
    assert old.pxscale == 0.0456 and old.fwhm == c["fwhm_px"]
    assert all(r.lam_over_d_px == c["lam_over_d_px"] for r in old.reducers.values())
    obj, samp = near.default_config(old)
    assert obj.metric.pxscale == 0.0456 and obj.metric.fwhm == c["fwhm_px"]
    g = near.make_guard(old, frame_selection=False)
    assert (g.fwhm_px, g.lam_over_d_px) == (c["fwhm_bin_px"], c["lam_over_d_px"])


def _settle(tmp_path, run, extra=(), setup=None):
    d = tmp_path / run
    d.mkdir(exist_ok=True)
    if setup is not None:
        (d / "run_setup.json").write_text(json.dumps(setup))
    argv = ["resume", "--root", str(tmp_path), "--run-dir", str(d)] + list(extra)
    a = cli._build_parser().parse_args(argv)
    a._argv = argv
    cli._constants_for_run(a)
    return a.pxscale, a.psflib


def test_a_run_keeps_the_scale_and_library_it_was_made_with(tmp_path):
    lib = str(tmp_path / "psflib_v2")
    os.makedirs(lib)
    assert _settle(tmp_path, "new") == (None, None), "a new run: the class default"
    assert _settle(tmp_path, "old", setup={"reducer": {"pxscale": 0.0456}}) == (0.0456, None), \
        "a run that recorded no scale was made at 0.0456"
    rec = {"reducer": {"provenance": {"instrument": "NEAR", "pxscale": 0.0445, "psflib": {"path": lib}}}}
    assert _settle(tmp_path, "s6", setup=rec) == (0.0445, lib)
    assert _settle(tmp_path, "s6", ["--pxscale", "0.0456"], setup=rec) == (0.0456, lib)
    gone = {"reducer": {"provenance": {"instrument": "NEAR", "pxscale": 0.0445,
                                       "psflib": {"path": str(tmp_path / "elsewhere")}}}}
    with pytest.raises(SystemExit, match="not on this machine"):
        _settle(tmp_path, "moved", setup=gone)
    assert _settle(tmp_path, "moved", ["--psflib", lib], setup=gone) == (0.0445, lib)
