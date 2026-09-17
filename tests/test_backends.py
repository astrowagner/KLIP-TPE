"""Backends: pyKLIP, VIP, spaceKLIP/JWST ingestion, and the custom-pipeline wrappers.
pyKLIP / VIP tests are skipped when the packages are not installed."""
import os

import numpy as np
import pytest

# end-to-end on synthetic data (~20 s on two cores); `-m \"not slow\"` skips it
pytestmark = pytest.mark.slow
from astropy.io import fits

from klip_tpe import (CalibrationConfig, MawetPeakSNR, Objective, Param, PositionSampler, ReductionRequest,
                      ReductionResult, RunConfig, Runner, SearchSpace, Source, ValidationConfig, kgrid)
from klip_tpe.injection import GaussianPSF
from klip_tpe.reducer import PartitionedReducer
from klip_tpe.synthetic import synthetic_klip_dataset

QUIET = (lambda s: None)
PX, LAM, D = 0.05, 1.6e-6, 8.0


def _ds(seed=1, nframes=40):
    return synthetic_klip_dataset(nframes=nframes, size=64, pa_span=60, seed=seed)


def _params(**kw):
    p = dict(k_klip=5, bin=2, filter=0, n_ang=2, inrad=6, outrad=24, angsep=0.5, anglemax=90,
             corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0)
    p.update(kw)
    return p


def _expected_xy(rho_as, theta, size=64):
    c = (size - 1) / 2.0
    az = np.deg2rad(theta + 90.0)
    return c + rho_as / PX * np.cos(az), c + rho_as / PX * np.sin(az)


def _check_injection(red, p):
    """The injected companion lands where the package's own derotation convention puts it
    (PA east of north, north up) and a k-scan returns one image per k."""
    clean = red.reduce(ReductionRequest(p, None))
    inj = red.reduce(ReductionRequest(p, [Source(0.7, 45.0, 3e-3)]))
    d = inj.image - clean.image
    iy, ix = np.unravel_index(np.nanargmax(d), d.shape)
    ex, ey = _expected_xy(0.7, 45.0)
    assert abs(ix - ex) <= 1.5 and abs(iy - ey) <= 1.5, (ix, iy, ex, ey)
    scan = red.reduce(ReductionRequest(p, None, k_scan=True))
    assert scan.image.shape == (p["k_klip"], 64, 64)
    return clean


# ----------------------------------------------------------------------------
# pyKLIP
# ----------------------------------------------------------------------------
def test_pyklip_backend_adi_rdi_and_scan():
    pytest.importorskip("pyklip")
    from klip_tpe.backends.pyklip import PyKLIPReducer
    ds = _ds()
    red = PyKLIPReducer(ds, pxscale=PX, lam_m=LAM, diam_m=D, injection_model=GaussianPSF(4.0, star_flux=1e4),
                        fwhm_px=4.0, outrad_cap=28)
    assert red.supports_kscan and red.name == "pyklip"
    clean = _check_injection(red, _params())
    assert clean.meta["backend"] == "pyklip" and np.isfinite(clean.meta["movement_px"])
    ds.ref_cube = ds.cube[:12] * 1.01
    rdi = red.reduce(ReductionRequest(_params(mode="RDI"), None))
    assert rdi.meta["mode"] == "RDI" and np.isfinite(rdi.image[30, 45])
    try:
        nmf = red.reduce(ReductionRequest(_params(algo="nmf", k_klip=3), None))
        assert nmf.meta["algo"] == "nmf"
    except ModuleNotFoundError:                       # pyklip's NMF needs the optional NonnegMFPy
        pass


def test_dataset_from_pyklip_generic_data():
    pytest.importorskip("pyklip")
    from pyklip.instruments.Instrument import GenericData
    from klip_tpe.backends.pyklip import dataset_from_pyklip
    rng = np.random.default_rng(0)
    n, size = 8, 70
    imgs = rng.standard_normal((n, size, size)).astype(np.float32)
    centers = np.column_stack([np.full(n, 36.3), np.full(n, 33.8)])
    for i in range(n):
        imgs[i, 34, 36] += 100.0                                     # a bright pixel near the star
    data = GenericData(imgs, centers, parangs=np.linspace(-10, 10, n), filenames=[f"f{i//4}" for i in range(n)])
    data.filenums = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    out = dataset_from_pyklip(data, crop_half=20, name="g")
    assert set(out) == {"g0", "g1"} and out["g0"].cube.shape == (4, 40, 40)
    # the star centre (36.3, 33.8) is now at (19.5, 19.5): the bright pixel moved by the same shift
    m = out["g0"].cube[0]
    iy, ix = np.unravel_index(np.argmax(m), m.shape)
    assert abs(ix - (36 - 36.3 + 19.5)) <= 1 and abs(iy - (34 - 33.8 + 19.5)) <= 1
    one = dataset_from_pyklip(data, crop_half=20, partition_by=None, name="all")
    assert list(one) == ["all"] and one["all"].nframes == n


def test_pyklip_backend_in_runner(tmp_path):
    pytest.importorskip("pyklip")
    from klip_tpe.backends.pyklip import PyKLIPReducer
    reds = {f"r{i}": PyKLIPReducer(_ds(seed=i), pxscale=PX, lam_m=LAM, diam_m=D, fwhm_px=4.0, outrad_cap=28,
                                  injection_model=GaussianPSF(4.0, star_flux=1e4)) for i in (1, 2)}
    red = PartitionedReducer(reds)
    block = [Param("bin", 1, 4, "int", default=2), Param("n_ang", 1, 3, "int", default=2),
             Param("angsep", 0.0, 1.0, "float", default=0.3),
             Param("k_klip", 1, 8, "int", grid=kgrid(8), default=4)]
    space = SearchSpace().replicate(block, list(reds)).with_selection("two_slot", partitions=list(reds))
    obj = Objective(MawetPeakSNR(pxscale=PX, fwhm=4.0), clean_subtract=True)
    samp = PositionSampler(fwhm_as=4.0 * PX)
    cfg = RunConfig(ann_edges=[8, 24], n_iter=4, n_init=2, seed=2, save_fits=True, fm_curve=False,
                    validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[3e-3]))
    res = Runner(red, space, obj, samp, cfg, str(tmp_path / "pk"), log=QUIET).run()
    assert len(res) == 1 and res[0].validated and os.path.exists(tmp_path / "pk" / "klip_stitched.fits")


# ----------------------------------------------------------------------------
# VIP
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("algo", ["pca_annular", "pca", "median_sub"])
def test_vip_backend_algos(algo):
    pytest.importorskip("vip_hci")
    from klip_tpe.backends.vip import VIPReducer
    red = VIPReducer(_ds(), pxscale=PX, lam_m=LAM, diam_m=D, injection_model=GaussianPSF(4.0, star_flux=1e4),
                     fwhm_px=4.0, outrad_cap=28)
    p = _params(algo=algo, k_klip=3)
    clean = _check_injection(red, p)
    assert clean.meta["backend"] == "vip" and clean.meta["algo"] == algo
    # the image is confined to the zone like the built-in reducer
    assert np.isnan(clean.image[31, 31]) and np.isfinite(clean.image[31, 45])


def test_vip_rdi():
    pytest.importorskip("vip_hci")
    from klip_tpe.backends.vip import VIPReducer
    ds = _ds()
    ds.ref_cube = ds.cube[:12] * 1.01
    red = VIPReducer(ds, pxscale=PX, lam_m=LAM, diam_m=D, fwhm_px=4.0, outrad_cap=28)
    r = red.reduce(ReductionRequest(_params(use_rdi=True, rdi_mode="ardi"), None))
    assert np.isfinite(r.image[31, 45]) and "rdi_note" not in r.meta


# ----------------------------------------------------------------------------
# spaceKLIP / JWST files
# ----------------------------------------------------------------------------
def _write_calints(path, nints, roll, star=(40.2, 38.7), size=80, seed=0, filt="F1140C", inst="MIRI"):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    psf = 50 * np.exp(-0.5 * ((xx - star[0]) ** 2 + (yy - star[1]) ** 2) / 3.0 ** 2)
    data = np.stack([psf + rng.standard_normal((size, size)) for _ in range(nints)]).astype(np.float32)
    ph = fits.Header()
    ph["INSTRUME"], ph["FILTER"], ph["NINTS"], ph["TELESCOP"] = inst, filt, nints, "JWST"
    sh = fits.Header()
    sh["PIXAR_A2"] = 0.11 ** 2
    sh["STARCENX"], sh["STARCENY"] = star[0] + 1, star[1] + 1         # 1-indexed like the headers
    sh["CRPIX1"], sh["CRPIX2"] = star[0] + 1, star[1] + 1
    sh["ROLL_REF"], sh["V3I_YANG"], sh["VPARITY"] = roll, 4.8, -1
    sh["CRVAL1"], sh["CRVAL2"] = 10.0, -20.0
    fits.HDUList([fits.PrimaryHDU(header=ph), fits.ImageHDU(data, header=sh, name="SCI")]).writeto(path)


@pytest.fixture(scope="module")
def jwst_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("jwst")
    _write_calints(d / "sci_roll1_calints.fits", 5, roll=10.0, seed=1)
    _write_calints(d / "sci_roll2_calints.fits", 5, roll=20.0, seed=2)
    _write_calints(d / "ref_calints.fits", 6, roll=15.0, star=(39.6, 40.1), seed=3)
    return str(d)


def test_spaceklip_loader_builtin_reader(jwst_dir):
    from klip_tpe.backends import spaceklip as sk
    sci = [os.path.join(jwst_dir, f) for f in ("sci_roll1_calints.fits", "sci_roll2_calints.fits")]
    ref = [os.path.join(jwst_dir, "ref_calints.fits")]
    ds = sk.load_spaceklip(sci_files=sci, ref_files=ref, crop_half=25, name="miri", use_pyklip_reader=False, log=QUIET)
    assert set(ds) == {"miriroll1", "miriroll2"}              # one partition per position angle (default)
    a = ds["miriroll1"]
    assert a.cube.shape == (5, 50, 50) and a.ref_cube is not None and a.ref_cube.shape[0] == 6
    assert np.allclose(a.angles, 10.0 - 4.8 * -1) and np.allclose(ds["miriroll2"].angles, 20.0 + 4.8)
    # one partition per exposure is still available
    ds_f = sk.load_spaceklip(sci_files=sci, ref_files=ref, crop_half=25, name="miri", partition_by="filenums",
                             use_pyklip_reader=False, log=QUIET)
    assert set(ds_f) == {"miri0", "miri1"}
    assert np.isclose(a.meta["pxscale"], 0.11) and np.isclose(a.meta["wavelength_m"], 1.1315651557554e-05)
    # star at the crop centre after re-centring
    iy, ix = np.unravel_index(np.argmax(a.cube.mean(axis=0)), a.cube.shape[1:])
    assert abs(ix - 24.5) <= 1 and abs(iy - 24.5) <= 1
    # a database-like dict of tables works too
    from astropy.table import Table
    tab = Table({"TYPE": ["SCI", "SCI", "REF"], "FITSFILE": sci + ref, "FILTER": ["F1140C"] * 3, "NINTS": [5, 5, 6]})
    ds2 = sk.load_spaceklip({"JWST_MIRI_F1140C": tab}, crop_half=25, use_pyklip_reader=False, log=QUIET)
    assert set(ds2) == {"JWST_MIRI_F1140Croll1", "JWST_MIRI_F1140Croll2"}


def test_spaceklip_loader_pyklip_reader_and_reducer(jwst_dir, tmp_path):
    pytest.importorskip("pyklip")
    pytest.importorskip("pyklip.instruments.JWST")
    from klip_tpe.backends import spaceklip as sk
    sci = [os.path.join(jwst_dir, f) for f in ("sci_roll1_calints.fits", "sci_roll2_calints.fits")]
    ref = [os.path.join(jwst_dir, "ref_calints.fits")]
    try:
        ds = sk.load_spaceklip(sci_files=sci, ref_files=ref, crop_half=25, name="m", use_pyklip_reader=True, log=QUIET)
    except Exception as exc:                        # JWSTData needs WCS keywords the fake files may lack
        pytest.skip(f"pyklip JWSTData could not read the synthetic files: {exc!r}")
    assert set(ds) == {"mroll1", "mroll2"} and np.allclose(ds["mroll1"].angles, 14.8)
    red = sk.make_reducer(ds, log=QUIET)
    assert red.partitions() == ["mroll1", "mroll2"] and red.reducers["mroll1"].defaults["mode"] == "ADI+RDI"
    assert red.pool_kind == "threads"                        # pyKLIP forks its own workers
    r = red.reducers["mroll1"].reduce(ReductionRequest(dict(k_klip=3, bin=1, filter=0, n_ang=1, inrad=4, outrad=18,
                                                        angsep=0.0, anglemax=360), None))
    assert r.meta["mode"] == "ADI+RDI" and np.isfinite(r.image[24, 36])


def test_spaceklip_make_reducer_with_template(jwst_dir):
    pytest.importorskip("pyklip")
    from klip_tpe.backends import spaceklip as sk
    sci = [os.path.join(jwst_dir, "sci_roll1_calints.fits")]
    ds = sk.load_spaceklip(sci_files=sci, crop_half=25, name="t", use_pyklip_reader=False, log=QUIET)
    yy, xx = np.mgrid[0:21, 0:21]
    tmpl = np.exp(-0.5 * ((xx - 10) ** 2 + (yy - 10) ** 2) / 2.0 ** 2)
    red = sk.make_reducer(ds, psf_template=tmpl, star_flux=1e4, log=QUIET)
    r0 = red.reducers["troll1"]
    assert r0.model.name == "template" and r0.defaults["mode"] == "ADI"      # no refs -> ADI
    assert np.isclose(r0.lam_over_d_px, (1.1315651557554e-05 / 6.5) * 206265 / 0.11)


# ----------------------------------------------------------------------------
# custom pipelines
# ----------------------------------------------------------------------------
def test_function_reducer_wraps_user_subtraction():
    from klip_tpe.backends import FunctionReducer
    from klip_tpe.klip import derotate
    calls = {"n": 0, "kscan": 0}

    def my_sub(cube, angles, params, k_scan):
        calls["n"] += 1
        calls["kscan"] += int(k_scan)
        med = np.nanmedian(cube, axis=0)                          # classical ADI
        res = derotate(cube - med, angles, params["truenorth"])
        img = np.nanmean(res, axis=0)
        if k_scan:
            return np.stack([img] * params["k_klip"])
        return img

    red = FunctionReducer(_ds(), my_sub, pxscale=PX, lam_m=LAM, diam_m=D, injection_model=GaussianPSF(4.0, 1e4),
                          fwhm_px=4.0, outrad_cap=28, supports_kscan=True, extra_params={"my_knob": 3})
    p = _params(k_klip=2)
    clean = _check_injection(red, p)
    assert clean.meta["backend"] == "custom" and clean.meta["params"]["my_knob"] == 3
    assert calls["n"] == 3 and calls["kscan"] == 1


def test_external_reducer_runs_in_runner(tmp_path):
    from klip_tpe.backends import ExternalReducer
    from klip_tpe.injection import inject_sources
    from klip_tpe.klip import derotate
    ds = _ds()
    model = GaussianPSF(4.0, star_flux=1e4)

    def reduce_fn(req):
        cube = ds.cube if not req.injections else inject_sources(ds.cube, ds.angles, req.injections, model, PX)
        k = int(req.params.get("k_klip", 1))
        med = np.nanmedian(cube, axis=0)
        img = np.nanmean(derotate(cube - med, ds.angles, 0.0), axis=0)
        yy, xx = np.mgrid[0:64, 0:64]
        rr = np.hypot(xx - 31.5, yy - 31.5)
        img[(rr < req.params["inrad"]) | (rr > req.params["outrad"])] = np.nan
        img = img * (1 + 0.01 * k)
        return ReductionResult(img.astype(np.float32), 1.0, {"k": k})

    red = ExternalReducer(pxscale=PX, fwhm_px=4.0, angles=ds.angles, reduce_fn=reduce_fn, name="mine")
    space = SearchSpace()
    space.add(Param("k_klip", 1, 6, "int", default=2))
    obj = Objective(MawetPeakSNR(pxscale=PX, fwhm=4.0), clean_subtract=True)
    samp = PositionSampler(fwhm_as=4.0 * PX)
    cfg = RunConfig(ann_edges=[8, 24], n_iter=3, n_init=2, seed=1, save_fits=False, fm_curve=False,
                    validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[3e-3]),
                    param_verify=False)
    res = Runner(red, space, obj, samp, cfg, str(tmp_path / "ext"), log=QUIET).run()
    assert len(res) == 1 and res[0].validated


def test_pyklip_adi_at_angsep_zero_does_not_subtract_the_frame_from_itself():
    """pyKLIP's reference selection is ``moves >= movement``; at movement 0 the target frame
    is in its own KL basis and any source is annihilated (paper run D: ADI peak 3e-7 against
    2.8 in RDI).  angsep = 0 must mean 'the frame itself only', as in the built-in reducer."""
    pytest.importorskip("pyklip")
    from klip_tpe.backends.pyklip import PyKLIPReducer, MIN_MOVEMENT_PX
    from klip_tpe.metrics import Source
    ds = _ds(nframes=8)
    red = PyKLIPReducer(ds, pxscale=PX, lam_m=LAM, diam_m=D, injection_model=GaussianPSF(4.0, star_flux=1e4),
                        fwhm_px=4.0, outrad_cap=28)
    src = [Source(0.4, 90.0, 5e-2)]
    for mode in ("ADI", "ADI+RDI"):
        if mode == "ADI+RDI":
            ds.ref_cube = ds.cube[:4] * 1.02
        p = _params(mode=mode, angsep=0.0, k_klip=3, bin=1, n_ang=1)
        clean = red.reduce(ReductionRequest(p, None))
        inj = red.reduce(ReductionRequest(p, src))
        assert clean.meta["movement_px"] == pytest.approx(MIN_MOVEMENT_PX)
        diff = np.nan_to_num(inj.image - clean.image)
        x, y = _expected_xy(0.4, 90.0)
        peak = float(np.nanmax(diff[int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4]))
        # an annihilated source is round-off (~1e-7 of the injection); a recovered one is not
        assert peak > 1e-3 * 5e-2 * 1e4, f"{mode}: the injected source was subtracted away (peak {peak:.3g})"
