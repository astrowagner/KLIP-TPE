"""The injected stamp and the star's flux are normalized in the same aperture (the audit of 2026-11-25).

``TemplatePSF`` and ``LibraryPSF`` normalized a stamp by the whole pixels inside ``ee_radius_px``, while
the star's flux is measured in the exact aperture (``generic.star_flux_from_aperture_photometry``,
``stpsf_psf.unocculted_ee``), so an injected source carried exact/whole of the flux its contrast asked
for.  The stamps are now normalized exactly, and ``RunConfig.ee_aperture = "whole"`` keeps the old stamps,
to the bit, for runs saved before.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from klip_tpe import RunConfig
from klip_tpe.injection import LibraryPSF, TemplatePSF
from klip_tpe.instruments import near
from klip_tpe.instruments.generic import aperture_sum, star_flux_from_aperture_photometry


def _gauss(n=25, fwhm=4.0, cx=None, cy=None):
    cx = (n - 1) / 2.0 if cx is None else cx
    cy = (n - 1) / 2.0 if cy is None else cy
    yy, xx = np.mgrid[0:n, 0:n]
    return np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / (fwhm / 2.3548) ** 2)


def test_a_template_injects_the_flux_its_contrast_asks_for():
    t = 37.0 * _gauss(cx=12.3, cy=11.8)
    c = (12.3, 11.8)
    starphot, ap = 5.0e5, 3.0
    for r_ee in (2.0, 3.0, 4.0):
        sf = star_flux_from_aperture_photometry(t, starphot, ap, center=c, ee_radius_px=r_ee)
        exact = TemplatePSF(t, center=c, ee_radius_px=r_ee, star_flux=sf)
        whole = TemplatePSF(t, center=c, ee_radius_px=r_ee, star_flux=sf, ee_aperture="whole")
        got = {m.ee_aperture: aperture_sum(m.stamp(0.5)[0] * m.flux_unit * 1e-3, c[0], c[1], ap)
               for m in (exact, whole)}
        assert got["exact"] == pytest.approx(1e-3 * starphot, rel=1e-9)
        # the old stamps carried exact over whole of it
        yy, xx = np.mgrid[0:25, 0:25]
        ratio = aperture_sum(t, c[0], c[1], r_ee) / t[np.hypot(xx - c[0], yy - c[1]) <= r_ee].sum()
        assert got["whole"] == pytest.approx(1e-3 * starphot * ratio, rel=1e-9)
        assert abs(ratio - 1.0) > 1e-3
    # without a radius both normalize the total
    a, b = TemplatePSF(t), TemplatePSF(t, ee_aperture="whole")
    assert np.array_equal(a.stamp(0.5)[0], b.stamp(0.5)[0]) and a.flux_unit == b.flux_unit == t.sum()


def test_whole_pixels_give_the_old_stamps_to_the_bit():
    t = 37.0 * _gauss(cx=12.3, cy=11.8)
    c, r_ee = (12.3, 11.8), 3.0
    yy, xx = np.mgrid[0:25, 0:25]
    f = float(t[np.hypot(xx - c[0], yy - c[1]) <= r_ee].sum())
    m = TemplatePSF(t, center=c, ee_radius_px=r_ee)
    m.set_ee_aperture("whole")
    assert np.array_equal(m.stamp(0.5)[0], t / f) and m.template_flux == f == m.flux_unit
    m.set_ee_aperture("exact")
    assert m.flux_unit == pytest.approx(aperture_sum(t, c[0], c[1], r_ee), rel=1e-12)
    # a star flux for each aperture
    m2 = TemplatePSF(t, center=c, ee_radius_px=r_ee, star_flux={"exact": 2.0, "whole": 3.0})
    assert m2.flux_unit == 2.0 and m2.set_ee_aperture("whole") == "whole" and m2.flux_unit == 3.0
    assert m2.describe()["ee_aperture"] == "whole"
    with pytest.raises(ValueError):
        m2.set_ee_aperture("round")


def _library(**kw):
    seps = [0.2, 0.4, 0.6]
    sl = np.stack([(1.0 + i) * _gauss(n=31, cx=15.4, cy=14.7) for i in range(3)])
    return sl, seps, LibraryPSF(sl.copy(), seps, center=(15.4, 14.7), ee_radius_px=3.5, **kw)


def test_a_library_normalizes_its_stamps_in_the_exact_aperture():
    sl, seps, m = _library(flux_unit={"exact": 10.0, "whole": 11.0})
    assert m.ee_aperture == "exact" and m.flux_unit == 10.0
    for rho in (0.2, 0.33, 0.6):
        st = m.stamp(rho)[0]
        assert aperture_sum(st, 15.4, 14.7, 3.5) == pytest.approx(1.0, rel=1e-12)
    # whole pixels: the slices and the stamps exactly as before 2026-11-25
    yy, xx = np.mgrid[0:31, 0:31]
    mask = np.hypot(xx - 15.4, yy - 14.7) <= 3.5
    old = sl.copy()
    for i in range(3):
        old[i] = old[i] / old[i][mask].sum()
    m.set_ee_aperture("whole")
    assert m.flux_unit == 11.0 and np.array_equal(m.slices, old)
    r = 0.27
    f = (r - 0.2) / (0.4 - 0.2)
    st = (1 - f) * old[0] + f * old[1]
    st = st / st[mask].sum()
    assert np.array_equal(m.stamp(r)[0], st)
    _, _, w = _library(ee_aperture="whole")
    assert np.array_equal(w.slices, old) and w.flux_unit == 1.0


def test_near_takes_the_flux_unit_of_each_aperture():
    t = _gauss(n=16, fwhm=3.5)
    lod = 3.2
    yy, xx = np.mgrid[0:16, 0:16]
    whole = near.NEAR.b_to_a * float(t[np.hypot(xx - 7.5, yy - 7.5) <= lod].sum())
    assert near.template_flux_unit(t, lod, exact=False) == whole
    assert near.template_flux_unit(t, lod) == pytest.approx(near.NEAR.b_to_a * aperture_sum(t, 7.5, 7.5, lod),
                                                             rel=1e-12)


def test_the_unocculted_ee_is_centered_on_the_psf(tmp_path, monkeypatch):
    """A cached unocculted PSF 0.8 px off the middle: the PSF-centered aperture holds more at small radii."""
    from astropy.io import fits

    from klip_tpe import stpsf_psf as S
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path))
    meta = {"kind": "unocculted_ee", "instrument": "NIRCam", "filter": "F444W", "pupil_mask": "MASKRND",
            "fov_arcsec": 30.0, "oversample": 2, "nlambda": 3, "aperture": None, "detector_position": None,
            "date": None, "version": S._CACHE_VERSION}
    img = _gauss(n=61, fwhm=4.0, cx=30.8, cy=30.0)
    hdr = fits.Header()
    hdr["PSFTOTAL"], hdr["PIXELSCL"] = float(img.sum()), 0.063
    os.makedirs(S.cache_dir(), exist_ok=True)
    fits.PrimaryHDU(img.astype(np.float32), header=hdr).writeto(
        os.path.join(S.cache_dir(), f"eeunocc_NIRCam_F444W_{S._key(meta)}.fits"))
    psf = S.unocculted_ee(2.0, log=lambda s: None)
    geo = S.unocculted_ee(2.0, log=lambda s: None, center="geometric")
    i32 = img.astype(np.float32).astype(float)
    assert geo == pytest.approx(aperture_sum(i32, 30.0, 30.0, 2.0) / img.sum(), rel=1e-9)
    assert psf > geo * 1.02
    with pytest.raises(ValueError):
        S.unocculted_ee(2.0, log=lambda s: None, center="peak")
    assert json.dumps(meta)                                   # the key's input stays serializable


def test_a_run_normalizes_exactly_and_an_old_run_by_whole_pixels(tmp_path):
    from conftest import build_synthetic_run
    from klip_tpe import CalibrationConfig, Runner, ValidationConfig, cli
    assert RunConfig().ee_aperture == "exact"
    d = RunConfig(ann_edges=[8, 20]).to_dict()
    assert RunConfig.from_dict(d).ee_aperture == "exact"
    d.pop("ee_aperture")
    assert RunConfig.from_dict(d).ee_aperture == "whole"
    p = cli._build_parser()
    assert cli._protocol_config(p.parse_args(["near", "--ee-aperture", "whole"]))["ee_aperture"] == "whole"
    assert "ee_aperture" not in cli._protocol_config(p.parse_args(["near"]))
    for mode in ("exact", "whole"):
        red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 20], n_iter=[4], n_init=2, param_verify=False,
                                                         calibration=CalibrationConfig(recal_budget=0),
                                                         validation=ValidationConfig(n_top=1, n_valid=1),
                                                         ee_aperture=mode)
        r = Runner(red, space, obj, samp, cfg, str(tmp_path / mode), log=lambda s: None)
        seen = []
        orig = r.reducer.reduce_config

        def spy(cfg2, *a, **kw):
            seen.append((cfg2.params.get("ee_aperture"), {v.get("ee_aperture") for v in cfg2.per_partition.values()}))
            return orig(cfg2, *a, **kw)
        r.reducer.reduce_config = spy
        r._reduce(space.decode(space.default_vector()), None, tag="t")
        assert seen and seen[0][0] == mode and seen[0][1] <= {mode}


def test_each_request_sets_the_models_aperture():
    from klip_tpe.reducer import Dataset, KLIPReducer, ReductionRequest
    from klip_tpe.synthetic import synthetic_klip_dataset
    ds = synthetic_klip_dataset(nframes=12, size=64, pa_span=60, seed=3)
    tpl = TemplatePSF(37.0 * _gauss(cx=12.3, cy=11.8), center=(12.3, 11.8), ee_radius_px=3.0, star_flux=1e4)
    px, lam = 0.05, 3.89 * 0.05 / 206265 * 8.2
    red = KLIPReducer(Dataset(ds.cube, ds.angles, ds.tags, texp=ds.texp), px, lam, 8.2, injection_model=tpl,
                      outrad_cap=30)
    params = dict(k_klip=4, inrad=8, outrad=26, n_ang=1, angsep=0.5, anglemax=360.0, fast=True, bin=1, filter=0)
    from klip_tpe.metrics import Source
    red.reduce(ReductionRequest(dict(params, ee_aperture="whole"), [Source(0.9, 60.0, 1e-3)]))
    assert tpl.ee_aperture == "whole"
    red.reduce(ReductionRequest(dict(params), [Source(0.9, 60.0, 1e-3)]))
    assert tpl.ee_aperture == "exact"
    assert red.use_ee_aperture("exact") is False and red.use_ee_aperture("whole") is True
