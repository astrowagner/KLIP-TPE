"""``klip_annular``'s inputs and outputs, as its docstring and docs/BACKENDS.md give them.

From Vivek Vijayakumar's question of 2026-11-11, whether the function's inputs and outputs are
documented anywhere.  What must hold:

* The residuals are float32 ``(n, ny, nx)``, or ``(k_klip, n, ny, nx)`` with ``k_scan``: NaN
  outside the zones and in dropped frames, finite inside, where an input NaN gives a finite
  residual.
* ``info`` holds the documented keys on both paths, and the per-target path its two more.
* With ``fm_cube`` a third, frame-aligned element comes back.
* Only the angles' differences matter: an offset gives the same residuals.
* The documented chain, ``nw_ang_comb(derotate(res, angles, truenorth), angles)``, is what the
  reducer's ``_subtract`` gives without its second high-pass.
"""
from __future__ import annotations

import numpy as np

from klip_tpe.injection import GaussianPSF
from klip_tpe.klip import KLIPParams, derotate, klip_annular, nw_ang_comb, zone_indices
from klip_tpe.reducer import KLIPReducer, ReductionRequest
from klip_tpe.synthetic import synthetic_klip_dataset

DOCUMENTED = {"fast", "auto_fast", "rdi", "klip_gram", "library_as_adi", "arcdist", "n_frames", "n_basis_frames",
              "k_requested", "k_effective", "k_capped", "fm", "fm_selfsub", "nref", "nref_used", "dropped",
              "n_dropped"}


def _data(n=24, size=61):
    ds = synthetic_klip_dataset(nframes=n, size=size, pa_span=40, seed=2)
    return ds, np.asarray(ds.cube, np.float32), np.asarray(ds.angles, float)


def test_the_documented_inputs_and_outputs():
    _, cube, ang = _data()
    cube = cube.copy()
    cube[:, 30, 45] = np.nan                            # a pixel inside the zone, NaN in every frame
    cube[::3, 22, 40] = np.nan                          # and one NaN in some frames
    n, ny, nx = cube.shape
    inside = np.zeros(ny * nx, bool)
    inside[zone_indices((ny, nx), 8, 25, 0, 360)] = True
    inside = inside.reshape(ny, nx)
    for fast in (False, True):
        p = KLIPParams(k_klip=5, inrad=8, outrad=25, n_ang=2, fast=fast, angsep=0.5, anglemax=90, n_min_ref=3)
        res, info = klip_annular(cube, ang, p, 4.0)
        assert res.dtype == np.float32 and res.shape == (n, ny, nx)
        kept = ~np.asarray(info["dropped"], bool)
        assert kept.any() and np.isnan(res[:, ~inside]).all()
        assert np.isfinite(res[kept][:, inside]).all(), "finite inside the zones, where the input was NaN too"
        assert np.isnan(res[~kept]).all()
        assert DOCUMENTED <= set(info), DOCUMENTED - set(info)
        assert info["n_frames"] == n and np.asarray(info["nref"]).shape == (n,)
        if not fast:
            assert {"n_starved", "drop_enabled"} <= set(info)
        # only the angles' differences matter
        res2, _ = klip_annular(cube, ang + 37.0, p, 4.0)
        assert np.array_equal(res, res2, equal_nan=True)
    scan, _ = klip_annular(cube, ang, KLIPParams(k_klip=4, inrad=8, outrad=25, k_scan=True, n_min_ref=3), 4.0)
    assert scan.shape == (4, n, ny, nx) and scan.dtype == np.float32
    out = klip_annular(cube, ang, KLIPParams(k_klip=4, inrad=8, outrad=25, n_min_ref=3), 4.0,
                       fm_cube=np.zeros_like(cube))
    assert len(out) == 3 and out[2].shape == (n, ny, nx) and out[2].dtype == np.float32


def test_the_documented_chain_is_the_reducers():
    ds, cube, ang = _data()
    red = KLIPReducer(ds, pxscale=0.05, lam_m=3.8e-6, diam_m=8.2, injection_model=GaussianPSF(3.0, star_flux=1e4),
                      fwhm_px=3.0, outrad_cap=28)
    p = KLIPParams(k_klip=6, inrad=8, outrad=25, n_ang=1, angsep=0.5, anglemax=90, n_min_ref=3)
    img, fm_img, info = red._subtract(cube, ang, p, {"comb_type": "nwadi", "fm_selfsub": True, "post_hp": False},
                                      0, ReductionRequest({}), None, None, None, {})
    res, _ = klip_annular(cube, ang, p, red.lam_over_d_px)
    doc = nw_ang_comb(derotate(res, ang, red.truenorth), ang)
    assert fm_img is None and img.shape == cube.shape[1:]
    assert np.array_equal(np.asarray(img, float), np.asarray(doc, float), equal_nan=True)
