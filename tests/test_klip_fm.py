"""KLIP-FM forward model, RDI/ARDI basis, destriping, FM products and the optional
reference images (nosub / cadi)."""
import numpy as np
import pytest

from klip_tpe.injection import GaussianPSF, inject_sources
from klip_tpe.klip import (KLIPParams, destripe, klip_annular, klip_basis, klip_fm_zone, nw_ang_comb,
                           nw_ang_comb_ref)
from klip_tpe.metrics import Source, source_xy, star_center
from klip_tpe.products import fm_contrast_curve, fm_response, fm_test_sources
from klip_tpe.reducer import Dataset, KLIPReducer, PartitionedReducer, ReductionRequest
from klip_tpe.space import Config
from klip_tpe.synthetic import synthetic_klip_dataset

PX = 0.05
FWHM = 4.0
LOD = FWHM / 1.028
LAM = 1e-6
DIAM = LAM * 206265.0 / (LOD * PX)          # so that lam/D = LOD px at PX arcsec/px


def _reducer(ds, **defaults):
    d = {"corr_thresh": None}
    d.update(defaults)
    return KLIPReducer(ds, PX, LAM, DIAM, injection_model=GaussianPSF(FWHM, star_flux=1e4),
                       fwhm_px=FWHM, outrad_cap=28.0, defaults=d)


def _peak_box(img, src, half=3):
    cx, cy = star_center(img.shape)
    xs, ys = source_xy(src.rho, src.theta, PX, cx, cy)
    x, y = int(round(xs[0])), int(round(ys[0]))
    return img[y - half:y + half + 1, x - half:x + half + 1]


# ----------------------------------------------------------------------------
# (a) k = 0 -> FM residual equals the model; the eigen-perturbation is the derivative
# ----------------------------------------------------------------------------
def test_fm_zero_modes_returns_model_and_first_order_matches_finite_difference():
    ds = synthetic_klip_dataset(nframes=20, size=48, pa_span=30.0, seed=2)
    model = GaussianPSF(FWHM, star_flux=1e4)
    src = [Source(10 * PX, 40.0, 0.001)]
    mcube = inject_sources(np.zeros_like(ds.cube), ds.angles, src, model, PX)
    p = KLIPParams(k_klip=0, inrad=3, outrad=20, fast=True)
    res, info, fm = klip_annular(ds.cube, ds.angles, p, LOD, fm_cube=mcube)
    ok = np.isfinite(fm)
    assert ok.any()
    assert np.allclose(fm[ok], mcube[ok], atol=1e-5)
    assert np.allclose(res[ok], ds.cube[ok], atol=1e-4)
    assert info["nref"].shape == (20,)

    # per-target path, k=0 as well
    p = KLIPParams(k_klip=0, inrad=3, outrad=20, fast=False, angsep=0.3, anglemax=20, n_min_ref=2)
    res, info, fm = klip_annular(ds.cube, ds.angles, p, LOD, fm_cube=mcube)
    ok = np.isfinite(fm)
    assert np.allclose(fm[ok], mcube[ok], atol=1e-5)
    assert info["nref"].min() >= 1 and info["nref_used"].shape == (20,)

    # first-order FM == d/d eps KLIP(S + eps M) (basis rebuilt with the model in it)
    rng = np.random.default_rng(0)
    nref, npx, k = 12, 300, 5
    R = rng.standard_normal((nref, npx)) * np.linspace(3, 1, nref)[:, None] + rng.standard_normal(npx) * 5
    M = np.zeros((nref, npx))
    for i in range(nref):
        M[i, 20 + 3 * i:26 + 3 * i] = np.hanning(6)

    def klip(R_):
        Z = klip_basis(R_, k)
        return R_ - (R_ @ Z.T) @ Z

    fm_lin = klip_fm_zone(R, M, R, M, k, fm_selfsub=True)
    fm_over = klip_fm_zone(R, M, R, M, k, fm_selfsub=False)
    eps = 1e-3
    fd = (klip(R + eps * M) - klip(R)) / eps
    assert np.linalg.norm(fd - fm_lin) / np.linalg.norm(fd) < 1e-3
    assert np.linalg.norm(fd - fm_over) / np.linalg.norm(fd) > 0.1     # self-sub terms matter

    with pytest.raises(ValueError):
        klip_annular(ds.cube, ds.angles, KLIPParams(k_klip=3, fast=True, k_scan=True), LOD, fm_cube=mcube)


# ----------------------------------------------------------------------------
# (b) FM image vs injected - clean on the synthetic cube
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("params", [
    {"k_klip": 5, "fast": True, "inrad": 4, "outrad": 26, "filter": 0},
    {"k_klip": 5, "fast": False, "angsep": 0.5, "anglemax": 20, "n_min_ref": 3, "inrad": 4, "outrad": 26, "filter": 0},
    {"k_klip": 8, "fast": False, "angsep": 1.0, "anglemax": 30, "n_min_ref": 3, "inrad": 4, "outrad": 26,
     "filter": 9, "bin": 3},
])
def test_fm_image_matches_injection_difference(params):
    ds = synthetic_klip_dataset(nframes=60, size=64, pa_span=40.0, seed=1)
    red = _reducer(ds)
    src = [Source(12 * PX, 30.0, 0.002)]               # faint: first-order regime
    clean = red.reduce(ReductionRequest(params))
    inj = red.reduce(ReductionRequest(params, injections=src))
    fm = red.reduce(ReductionRequest(params, fm_sources=src))
    assert clean.fm_image is None and fm.fm_image is not None
    assert fm.fm_image.shape == clean.image.shape
    assert fm.meta["fm"] and fm.meta["fm_selfsub"] is True
    d_pk = np.nanmax(_peak_box(inj.image - clean.image, src[0]))
    f_pk = np.nanmax(_peak_box(fm.fm_image, src[0]))
    assert d_pk > 0 and f_pk > 0
    assert abs(f_pk / d_pk - 1.0) < 0.30
    # the FM image is the model response only: no speckles away from the planet
    far = fm.fm_image.copy()
    box = _peak_box(far, src[0], half=6)
    box[:] = np.nan
    assert np.nanmax(np.abs(far)) < 0.2 * f_pk
    # self-subtraction toggle off runs and differs (over-subtraction only)
    off = red.reduce(ReductionRequest({**params, "fm_selfsub": False}, fm_sources=src))
    assert off.meta["fm_selfsub"] is False
    assert not np.allclose(np.nan_to_num(off.fm_image), np.nan_to_num(fm.fm_image))
    # science image is unaffected by requesting the FM pass
    assert np.allclose(np.nan_to_num(fm.image), np.nan_to_num(clean.image))


def test_nw_ang_comb_ref_uses_reference_weights():
    rng = np.random.default_rng(0)
    n, s = 6, 20
    ang = np.linspace(-5, 5, n)
    ref = rng.standard_normal((n, s, s)) * np.linspace(1, 4, n)[:, None, None]
    model = np.ones((n, s, s))
    out = nw_ang_comb_ref(model, ang, ref)
    # constant frames combined with any positive weights stay constant (interior)
    assert np.allclose(out[6:14, 6:14], 1.0, atol=1e-6)
    # weights differ from those the model itself would produce
    self_w = nw_ang_comb(ref, ang)
    ref_w = nw_ang_comb(ref, ang, ref_cube=ref)
    assert np.allclose(np.nan_to_num(self_w), np.nan_to_num(ref_w))


# ----------------------------------------------------------------------------
# (c) RDI / ARDI vs ADI in a low-rotation, short sequence
# ----------------------------------------------------------------------------
def _speckle_modes(size, nmodes, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    rr = np.hypot(xx - c, yy - c)
    modes = []
    for _ in range(nmodes):
        b = np.zeros((size, size))
        for _ in range(12):
            x0, y0 = rng.uniform(c - 20, c + 20, 2)
            b += rng.uniform(0.3, 1) * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / 1.5 ** 2)
        modes.append(b * 100 * np.exp(-rr / 15))
    return np.array(modes), rr


def _mode_cube(modes, n, seed):
    r = np.random.default_rng(seed)
    coef = r.normal(1.0, 0.4, (n, modes.shape[0]))
    return (np.tensordot(coef, modes, axes=1) + r.standard_normal((n,) + modes.shape[1:])).astype(np.float32)


def test_rdi_reference_basis_beats_adi_at_low_rotation():
    size, n_sci, n_ref = 64, 6, 60
    modes, rr = _speckle_modes(size, 12, 3)
    sci, ref = _mode_cube(modes, n_sci, 10), _mode_cube(modes, n_ref, 11)
    ds = Dataset(sci, np.linspace(-1, 1, n_sci), ref_cube=ref)
    red = _reducer(ds)
    base = {"k_klip": 12, "inrad": 4, "outrad": 26, "filter": 0}

    def rms(img):
        m = np.isfinite(img) & (rr > 6) & (rr < 24)
        return float(np.nanstd(img[m]))

    adi = red.reduce(ReductionRequest({**base, "fast": False, "angsep": 0.0, "anglemax": 0.9, "n_min_ref": 1}))
    rdi = red.reduce(ReductionRequest({**base, "use_rdi": True, "rdi_mode": "rdi"}))
    ardi = red.reduce(ReductionRequest({**base, "use_rdi": True, "rdi_mode": "ardi"}))
    assert not adi.meta["rdi"] and rdi.meta["rdi"] and ardi.meta["rdi"]
    assert rdi.meta["fast"] and ardi.meta["fast"]                     # RDI forces the single-basis path
    assert rdi.meta["n_basis_frames"] == n_ref and ardi.meta["n_basis_frames"] == n_ref + n_sci
    assert rms(rdi.image) < 0.5 * rms(adi.image)
    assert rms(ardi.image) < 0.5 * rms(adi.image)
    # no reference cube -> ADI fallback with a note
    ds2 = Dataset(sci, np.linspace(-1, 1, n_sci))
    r2 = _reducer(ds2).reduce(ReductionRequest({**base, "use_rdi": True}))
    assert not r2.meta["rdi"] and "rdi_note" in r2.meta
    # FM in RDI (planet-free refs: no self-sub) and ARDI (science model in the basis) both run
    src = [Source(12 * PX, 30.0, 0.002)]
    f_rdi = red.reduce(ReductionRequest({**base, "use_rdi": True, "rdi_mode": "rdi"}, fm_sources=src))
    f_ardi = red.reduce(ReductionRequest({**base, "use_rdi": True, "rdi_mode": "ardi"}, fm_sources=src))
    assert np.nanmax(_peak_box(f_rdi.fm_image, src[0])) > 0
    assert np.nanmax(_peak_box(f_ardi.fm_image, src[0])) > 0
    # ARDI has the model in the library -> self-subtraction -> lower throughput than pure RDI
    assert np.nanmax(_peak_box(f_ardi.fm_image, src[0])) < np.nanmax(_peak_box(f_rdi.fm_image, src[0]))


# ----------------------------------------------------------------------------
# (d) FM products
# ----------------------------------------------------------------------------
def test_fm_test_sources_spiral():
    s = fm_test_sources(4.0, 26.0, PX, 1e-3)
    n = int(np.floor((26 - 1 - 4) / 1.5)) + 1
    assert len(s) == min(24, max(5, n))
    r = np.array([x.rho for x in s]) / PX
    assert np.isclose(r[0], 4.0) and np.isclose(r[-1], 25.0)
    assert np.allclose([x.theta for x in s], (np.arange(len(s)) * 137.508) % 360.0)
    assert len(fm_test_sources(10.0, 12.0, PX, 1e-3)) == 5
    assert len(fm_test_sources(0.0, 200.0, PX, 1e-3)) == 24


def test_fm_response_peak_and_contrast_curve_finite():
    ds = synthetic_klip_dataset(nframes=40, size=64, pa_span=40.0, seed=4)
    red = _reducer(ds)
    params = {"k_klip": 5, "fast": True, "inrad": 4, "outrad": 26, "filter": 0}
    contrast = 1e-3
    srcs = fm_test_sources(max(4.0, FWHM), 26.0, PX, contrast)
    clean = red.reduce(ReductionRequest(params))
    fm = red.reduce(ReductionRequest(params, fm_sources=srcs))
    # fm_response is the matched-filter peak: on a pure model image it is positive at every source
    resp = fm_response(fm.fm_image, [s.rho for s in srcs], [s.theta for s in srcs], PX, FWHM)
    assert resp.shape == (len(srcs),) and np.all(np.isfinite(resp)) and np.all(resp > 0)
    # per-source kernel hook is honoured
    calls = []

    def kfn(r):
        calls.append(r)
        return None
    fm_response(fm.fm_image, [s.rho for s in srcs], [s.theta for s in srcs], PX, FWHM, kernel_fn=kfn)
    assert len(calls) == len(srcs)
    cc = fm_contrast_curve(fm.fm_image, clean.image, contrast, FWHM, PX, 4.0, 26.0, srcs)
    assert set(cc) >= {"r_as", "curve", "K_fm", "resp"}
    assert cc["K_fm"].shape == (len(srcs),)
    fin = np.isfinite(cc["curve"])
    assert fin.sum() >= 3 and np.all(cc["curve"][fin] > 0)
    assert cc["r_as"].shape == cc["curve"].shape
    # too few valid responses -> NaN curve, no crash
    bad = fm_contrast_curve(np.zeros_like(fm.fm_image), clean.image, contrast, FWHM, PX, 4.0, 26.0, srcs)
    assert bad["curve"].size == 0 or not np.isfinite(bad["curve"]).any()


# ----------------------------------------------------------------------------
# (e) destriping
# ----------------------------------------------------------------------------
def test_destripe_removes_row_and_column_pattern():
    rng = np.random.default_rng(0)
    ny, nx = 40, 50
    base = rng.standard_normal((ny, nx)) * 0.1
    rows = rng.normal(0, 5, ny)[:, None]
    cols = rng.normal(0, 5, nx)[None, :]
    img = base + rows + cols
    img[5, 7] = np.nan
    out = destripe(destripe(img, 90.0, 0.0), 0.0, 0.0)
    assert np.isnan(out[5, 7]) and np.isfinite(out).sum() == ny * nx - 1
    resid = out - (base - np.nanmedian(base))
    fin = np.isfinite(resid)
    assert np.nanstd(resid[fin]) < 0.15 and np.nanstd(img[fin]) > 4.0
    # column direction only: row offsets survive, column offsets go
    c_only = destripe(img, 90.0)
    assert np.nanstd(np.nanmedian(c_only, axis=0)) < 0.05
    assert np.nanstd(np.nanmedian(c_only, axis=1)) > 3.0
    # sigma clipping ignores a bright source in the stripe statistics
    img2 = base + cols
    img2[10:14, 20] += 200.0
    plain = destripe(img2, 90.0, clip_level=0.0)
    clipped = destripe(img2, 90.0, clip_level=3.0)
    assert abs(np.nanmedian(clipped[:, 20]) - np.nanmedian(plain[:, 20])) < 0.5
    assert np.isclose(clipped[0, 20] - base[0, 20], -np.median(base[:, 20]), atol=0.2)
    with pytest.raises(ValueError):
        destripe(img, 45.0)
    # reducer switch runs
    ds = synthetic_klip_dataset(nframes=12, size=48, pa_span=20.0, seed=5)
    ds.cube += rng.normal(0, 3, (1, 1, 48)).astype(np.float32)        # column stripes
    red = _reducer(ds)
    p = {"k_klip": 3, "fast": True, "inrad": 3, "outrad": 20, "filter": 0}
    a = red.reduce(ReductionRequest({**p, "do_destripe": False}))
    b = red.reduce(ReductionRequest({**p, "do_destripe": True}))
    assert a.image.shape == b.image.shape and b.meta["params"]["do_destripe"]


# ----------------------------------------------------------------------------
# (f) nosub / cadi extras, mask hook, partitioned FM combine, nref exposure
# ----------------------------------------------------------------------------
def test_extras_shapes_mask_hook_and_partitioned_fm():
    ds = synthetic_klip_dataset(nframes=24, size=48, pa_span=30.0, seed=6)
    red = _reducer(ds)
    p = {"k_klip": 4, "fast": True, "inrad": 3, "outrad": 20, "filter": 7, "bin": 2}
    src = [Source(10 * PX, 60.0, 0.002)]
    plain = red.reduce(ReductionRequest(p))
    assert plain.nosub is None and plain.cadi is None
    clean = red.reduce(ReductionRequest(p, extras=True))
    inj = red.reduce(ReductionRequest(p, injections=src, extras=True))
    assert clean.nosub.shape == (48, 48) and clean.cadi is None
    assert inj.cadi.shape == (48, 48) and inj.nosub is None
    assert np.isfinite(clean.nosub).sum() > 1000 and np.isfinite(inj.cadi).sum() > 1000
    # the injected planet is visible in the cADI reference
    assert np.nanmax(_peak_box(inj.cadi, src[0])) > 3 * np.nanstd(inj.cadi[np.isfinite(inj.cadi)])
    assert "nref" in plain.meta and len(plain.meta["nref"]) == plain.meta["n_binned"]

    # mask hook: applied to the science image only
    def mask(img):
        out = img.copy()
        out[:, :10] = np.nan
        return out
    redm = KLIPReducer(ds, PX, LAM, DIAM, injection_model=GaussianPSF(FWHM, 1e4), fwhm_px=FWHM,
                       outrad_cap=28.0, defaults={"corr_thresh": None}, mask_fn=mask)
    rm = redm.reduce(ReductionRequest(p, fm_sources=src))
    assert np.isnan(rm.image[:, :10]).all()
    assert np.isfinite(rm.fm_image[:, :10]).any()
    ks = redm.reduce(ReductionRequest(p, k_scan=True))
    assert ks.image.shape == (4, 48, 48) and np.isnan(ks.image[:, :, :10]).all()

    # partitioned: FM images combined with the science weights; k_scan + FM -> note, no FM
    ds2 = synthetic_klip_dataset(nframes=24, size=48, pa_span=30.0, seed=7)
    pr = PartitionedReducer({"a": red, "b": _reducer(ds2)}, weighting="equal")
    assert pr.supports_fm
    cfg = Config(params=dict(p), per_partition={"a": dict(p), "b": dict(p)}, selected=["a", "b"], x=np.zeros(0))
    ev = pr.reduce_config(cfg, None, False, "t", fm_sources=src, extras=True)
    assert ev.fm_image.shape == (48, 48) and ev.fm_stack.shape == (2, 48, 48)
    assert ev.nosub.shape == (48, 48) and ev.cadi is None
    assert np.allclose(np.nan_to_num(ev.fm_image), np.nan_to_num(np.nanmean(ev.fm_stack, axis=0)), atol=1e-5)
    ev0 = pr.reduce_config(cfg, None, False, "t")
    assert ev0.fm_image is None and ev0.nosub is None
    rr = pr.reduce(ReductionRequest(p, fm_sources=src, k_scan=True))
    assert rr.fm_image is None and rr.image.shape == (4, 48, 48)
