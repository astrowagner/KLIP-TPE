import numpy as np
import pytest

from klip_tpe.injection import GaussianPSF, inject_sources
from klip_tpe.klip import (KLIPParams, arcdist_deg, bin_angles, bin_frames, derotate,
                           frame_selection_mask, highpass, klip_annular, klip_basis, nw_ang_comb,
                           reference_mask, rotate_ccw, zone_indices)
from klip_tpe.metrics import Source, mawet_peak_snr, source_xy, star_center
from klip_tpe.synthetic import synthetic_klip_dataset

LOD = 4.0 / 1.028


# ----------------------------------------------------------------------------
# frame selection
# ----------------------------------------------------------------------------
def test_frame_selection_mask_basic_and_skip_rules():
    nf = 20
    corrs = np.linspace(0.90, 0.99, nf)
    tags = {"corrs": corrs, "noises": np.ones(nf), "coronoise": np.ones(nf)}
    # threshold keeps the 10 best frames; bin*k = 2*3 = 6 <= 10 -> cut applied
    m = frame_selection_mask(nf, tags, corr_thresh=corrs[10], noise_max=None, coronoise_max=None, bin_=2, k_klip=3)
    assert m.sum() == 10 and m[10:].all() and not m[:10].any()
    # would keep fewer than bin*k = 4*3 = 12 -> skipped entirely
    m = frame_selection_mask(nf, tags, corrs[10], None, None, bin_=4, k_klip=3)
    assert m.all()
    # cull nothing -> all True; cull everything -> all True (min-keep 3)
    assert frame_selection_mask(nf, tags, 0.0, None, None, 1, 1).all()
    assert frame_selection_mask(nf, tags, 1.5, None, None, 1, 1).all()
    # no tags / no threshold -> all True
    assert frame_selection_mask(nf, None, 0.95, None, None, 1, 1).all()
    assert frame_selection_mask(nf, tags, None, None, None, 1, 1).all()
    # mismatched tag length -> ignored
    assert frame_selection_mask(nf, {"corrs": corrs[:5]}, 0.95, None, None, 1, 1).all()


def test_frame_selection_noise_cuts_and_nan_pass():
    nf = 12
    noises = np.ones(nf)
    noises[3] = 10.0
    noises[4] = np.nan
    tags = {"corrs": np.ones(nf), "noises": noises, "coronoise": np.ones(nf)}
    m = frame_selection_mask(nf, tags, 0.5, noise_max=2.0, coronoise_max=None, bin_=1, k_klip=1)
    assert not m[3] and m[4] and m.sum() == nf - 1
    cn = np.ones(nf)
    cn[7] = 50.0
    tags["coronoise"] = cn
    m = frame_selection_mask(nf, tags, 0.5, 2.0, 2.0, 1, 1)
    assert not m[3] and not m[7] and m.sum() == nf - 2


# ----------------------------------------------------------------------------
# binning
# ----------------------------------------------------------------------------
def test_bin_frames_groups_and_pa_span():
    n = 12
    cube = np.arange(1, n + 1, dtype=np.float32)[:, None, None] * np.ones((1, 4, 4), np.float32)
    angles = np.arange(n, dtype=float)                  # 1 deg per frame
    bc, ba = bin_frames(cube, angles, 4, 100.0)
    assert bc.shape == (3, 4, 4)
    np.testing.assert_allclose(ba, [1.5, 5.5, 9.5])
    np.testing.assert_allclose(bc[:, 0, 0], [2.5, 6.5, 10.5])
    # a PA span limit of 1.5 deg closes every bin after two frames
    bc2, ba2 = bin_frames(cube, angles, 4, 1.5)
    assert bc2.shape[0] == 6
    np.testing.assert_allclose(ba2, np.arange(6) * 2 + 0.5)
    np.testing.assert_allclose(bin_angles(angles, 4, 1.5), ba2)
    # bin <= 1 is a no-op (copy)
    bc3, ba3 = bin_frames(cube, angles, 1, 1.0)
    assert bc3.shape == cube.shape and np.array_equal(ba3, angles)
    np.testing.assert_array_equal(bin_angles(angles, 1, 1.0), angles)


def test_bin_frames_drops_all_zero_bins():
    cube = np.ones((6, 3, 3), np.float32)
    cube[2:4] = 0.0
    bc, ba = bin_frames(cube, np.arange(6.0), 2, 100.0)
    assert bc.shape[0] == 2 and ba.tolist() == [0.5, 4.5]


# ----------------------------------------------------------------------------
# high-pass
# ----------------------------------------------------------------------------
def test_highpass_zero_border_width_bump_and_nan():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((30, 30)) + 100.0
    h5 = highpass(img, 5)
    assert h5.dtype == np.float32
    assert np.all(h5[:2, :] == 0) and np.all(h5[-2:, :] == 0) and np.all(h5[:, :2] == 0) and np.all(h5[:, -2:] == 0)
    assert np.abs(h5[2:-2, 2:-2]).max() > 0
    assert abs(h5[2:-2, 2:-2].mean()) < 0.2          # the constant 100 is removed
    # even width bumps to the next odd one
    np.testing.assert_array_equal(highpass(img, 4), h5)
    assert np.all(highpass(img, 6)[:3, :] == 0) and np.any(highpass(img, 6)[3, 3:-3] != 0)
    # width <= 1 returns the input
    np.testing.assert_allclose(highpass(img, 1), img.astype(np.float32))
    # interior value = img - boxcar mean
    box = img[10 - 2:10 + 3, 12 - 2:12 + 3].mean()
    assert h5[10, 12] == pytest.approx(img[10, 12] - box, abs=1e-4)
    # NaN handling
    img[15, 15] = np.nan
    hn = highpass(img, 5, nan_aware=True)
    assert np.isnan(hn[15, 15]) and np.isfinite(hn[15, 16]) and np.isfinite(hn[14, 14])
    h0 = highpass(img, 5, nan_aware=False)
    assert np.isnan(h0[15, 15])


# ----------------------------------------------------------------------------
# rotation
# ----------------------------------------------------------------------------
def test_rotate_ccw_direction_and_inverse():
    img = np.zeros((51, 51))
    img[25, 35] = 1.0                                   # feature at +x
    r = rotate_ccw(img, 90.0)
    assert np.unravel_index(np.nanargmax(r), r.shape) == (35, 25)   # -> +y
    r2 = rotate_ccw(img, -90.0)
    assert np.unravel_index(np.nanargmax(r2), r2.shape) == (15, 25)  # -> -y
    assert rotate_ccw(img, 0.0) is not img and np.array_equal(rotate_ccw(img, 0.0), img)
    # smooth blob: rotate and rotate back recovers the image
    yy, xx = np.mgrid[0:51, 0:51]
    blob = np.exp(-0.5 * ((xx - 33) ** 2 + (yy - 20) ** 2) / 4.0 ** 2)
    back = rotate_ccw(rotate_ccw(blob, 37.0), -37.0)
    assert np.nanmax(np.abs(back - blob)[12:-12, 12:-12]) < 0.03      # bilinear smoothing only
    assert np.unravel_index(np.nanargmax(back), back.shape) == (20, 33)
    # NaN handling: default "zero" (IDL rot as measured) mixes a NaN neighbour in as 0 and
    # only a stencil with no finite member is NaN; "propagate" NaNs every touching stencil
    blob[20, 33] = np.nan
    rr = rotate_ccw(blob, 30.0, cval=-5.0)
    assert np.isnan(rr).sum() == 0 and rr[0, 0] == -5.0
    inside = rr != -5.0
    assert np.nanmin(rr[inside]) >= 0.0 and rr.max() < 1.0
    rp = rotate_ccw(blob, 30.0, cval=-5.0, nan_mode="propagate")
    assert 1 <= np.isnan(rp).sum() < 20
    # a NaN region's rim is damped, not dropped: zone of ones, NaN outside
    z = np.ones((51, 51)); yy2, xx2 = np.mgrid[0:51, 0:51]; z[np.hypot(xx2 - 25, yy2 - 25) > 15] = np.nan
    rz = rotate_ccw(z, 20.0)
    rim = (np.hypot(xx2 - 25, yy2 - 25) > 14.2) & (np.hypot(xx2 - 25, yy2 - 25) <= 15.0)
    assert np.isfinite(rz[rim]).mean() > 0.9 and 0.3 < np.nanmean(rz[rim]) < 1.0


def test_derotate_and_nw_ang_comb_concentrate_rotating_planet():
    n, size = 24, 64
    angles = np.linspace(-40, 40, n)
    cx, cy = star_center((size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    speckle = 5.0 * np.exp(-0.5 * ((xx - cx - 12) ** 2 + (yy - cy + 6) ** 2) / 1.5 ** 2)
    rng = np.random.default_rng(3)
    cube = np.empty((n, size, size), np.float32)
    for i in range(n):
        cube[i] = speckle + 0.2 * rng.standard_normal((size, size))
    planet = [Source(0.9, 45.0, 1.0)]
    cube = inject_sources(cube, angles, planet, GaussianPSF(4.0, star_flux=60.0), 0.05)
    der = derotate(cube, angles)
    assert der.shape == cube.shape and der.dtype == np.float32
    comb = nw_ang_comb(der, angles)
    assert comb.shape == (size, size)
    px, py = source_xy(0.9, 45.0, 0.05, cx, cy)
    # planet stacks coherently at its sky position ...
    assert comb[int(round(py[0])), int(round(px[0]))] > 1.0
    # ... while the (rotating, high-variance) speckle is down-weighted relative to a plain mean
    mean = np.nanmean(der, axis=0)
    ring = np.hypot(xx - cx, yy - cy)
    band = (ring > 10) & (ring < 16)
    band &= np.hypot(xx - px[0], yy - py[0]) > 6
    assert np.nanstd(comb[band]) < np.nanstd(mean[band])
    # S/N of the planet in the combined image is high
    assert mawet_peak_snr(comb, 0.9, 45.0, 0.05, 4.0)[0] > 8
    # single frame passes through
    np.testing.assert_array_equal(nw_ang_comb(der[:1], angles[:1]), der[0])


# ----------------------------------------------------------------------------
# KLIP core
# ----------------------------------------------------------------------------
def test_klip_basis_orthonormal_rows_and_cap(rng):
    R = rng.standard_normal((6, 50))
    Z = klip_basis(R, 4)
    assert Z.shape == (4, 50)
    np.testing.assert_allclose(Z @ Z.T, np.eye(4), atol=1e-10)
    # k capped at the number of frames, k < 1 floored
    assert klip_basis(R, 10).shape == (6, 50)
    assert klip_basis(R, 0).shape == (1, 50)
    # row-vector input
    assert klip_basis(R[0], 3).shape == (1, 50)
    # modes span the reference rows: projecting R onto all modes reproduces R
    Zf = klip_basis(R, 6)
    np.testing.assert_allclose((R @ Zf.T) @ Zf, R, atol=1e-8)


def test_klip_basis_zero_eigenvalues_and_means():
    R = np.ones((5, 30))
    R[1] = 2.0                                          # rank 1
    Z = klip_basis(R, 5)
    assert Z.shape == (5, 30)
    assert Z[0] @ Z[0] == pytest.approx(1.0)
    assert np.abs(Z[1:]).max() == 0.0                   # zero-eigenvalue modes are zero rows
    assert np.all(np.isfinite(Z))
    # spat_mean removes the per-frame mean -> a constant matrix becomes all-zero rows
    Zs = klip_basis(np.ones((4, 20)), 2, spat_mean=True)
    assert np.abs(Zs).max() == 0.0
    # temp_mean: identical frames -> zero after subtracting the temporal mean
    Zt = klip_basis(np.tile(np.arange(20.0), (4, 1)), 2, temp_mean=True)
    assert np.abs(Zt).max() == 0.0


def test_zone_indices_and_arcdist():
    idx = zone_indices((40, 40), 5, 10, 0, 360)
    yy, xx = np.mgrid[0:40, 0:40]
    r = np.hypot(xx - 20, yy - 20).ravel()
    assert np.all((r[idx] >= 5) & (r[idx] <= 10))
    assert idx.size == np.sum((r >= 5) & (r <= 10))
    half = zone_indices((40, 40), 5, 10, 0, 180)
    assert 0 < half.size < idx.size
    assert arcdist_deg(10, 30, LOD) == pytest.approx(360 * LOD / (2 * np.pi * 20))
    assert arcdist_deg(0, 1, LOD) == arcdist_deg(0, 2, LOD)     # mid radius floored at 1
    m = reference_mask(np.array([0.0, 1.0, 5.0, 10.0]), 0, 2.0, 6.0)
    assert m.tolist() == [False, False, True, False]


@pytest.fixture(scope="module")
def ds():
    return synthetic_klip_dataset(nframes=30, size=48, pa_span=40)


def test_klip_annular_fast_and_auto_fast(ds):
    p = KLIPParams(k_klip=5, inrad=5, outrad=20)
    res, info = klip_annular(ds.cube, ds.angles, p, LOD)
    assert res.shape == ds.cube.shape and res.dtype == np.float32
    assert info["fast"] and info["auto_fast"] and info["n_dropped"] == 0
    yy, xx = np.mgrid[0:48, 0:48]
    r = np.hypot(xx - 24, yy - 24)
    assert np.isnan(res[:, r < 5]).all() and np.isnan(res[:, r > 20]).all()
    assert np.isfinite(res[:, (r > 6) & (r < 19)]).all()
    # the speckle field is mostly removed
    zone = (r > 6) & (r < 19)
    assert np.nanstd(res[:, zone]) < 0.3 * np.std(ds.cube[:, zone])
    # explicit fast gives the same answer
    res2, info2 = klip_annular(ds.cube, ds.angles, KLIPParams(k_klip=5, inrad=5, outrad=20, fast=True), LOD)
    assert info2["fast"] and not info2["auto_fast"]
    np.testing.assert_allclose(res2[:, zone], res[:, zone], atol=1e-4)
    # anglemax below the PA span disables auto-fast
    _, info3 = klip_annular(ds.cube, ds.angles, KLIPParams(k_klip=5, inrad=5, outrad=20, anglemax=30), LOD)
    assert not info3["fast"]


def test_klip_annular_slow_path_and_k_scan(ds):
    p = KLIPParams(k_klip=5, inrad=5, outrad=20, angsep=1.0, anglemax=30, n_min_ref=4)
    res, info = klip_annular(ds.cube, ds.angles, p, LOD)
    assert not info["fast"] and info["n_starved"] == 0 and info["drop_enabled"]
    assert res.shape == ds.cube.shape
    scan, _ = klip_annular(ds.cube, ds.angles, KLIPParams(**{**p.__dict__, "k_scan": True}), LOD)
    assert scan.shape == (5,) + ds.cube.shape
    np.testing.assert_allclose(scan[4], res, atol=1e-5, equal_nan=True)
    for kk in (1, 3):
        one, _ = klip_annular(ds.cube, ds.angles, KLIPParams(**{**p.__dict__, "k_klip": kk}), LOD)
        np.testing.assert_allclose(scan[kk - 1], one, atol=1e-5, equal_nan=True)
    # fast k-scan too
    fs, finfo = klip_annular(ds.cube, ds.angles, KLIPParams(k_klip=3, inrad=5, outrad=20, k_scan=True), LOD)
    assert fs.shape == (3,) + ds.cube.shape and finfo["fast"]
    f1, _ = klip_annular(ds.cube, ds.angles, KLIPParams(k_klip=1, inrad=5, outrad=20), LOD)
    np.testing.assert_allclose(fs[0], f1, atol=1e-5, equal_nan=True)
    # k larger than the frame count in the fast path: the scan still returns one image per
    # REQUESTED k (so per-partition scans stack), the basis is capped at n-1 (a complete
    # basis built from the frames it subtracts would leave only round-off), and the slices
    # past the cap repeat the capped image rather than staying NaN.  See
    # test_klip_rank_collapse.py for why; this used to assert the NaN slices.
    big, binfo = klip_annular(ds.cube[:4], ds.angles[:4], KLIPParams(k_klip=6, inrad=5, outrad=20, k_scan=True), LOD)
    assert big.shape[0] == 6 and np.isfinite(big[0]).any()
    assert binfo["k_capped"] and binfo["k_effective"] == 3 and binfo["k_requested"] == 6
    np.testing.assert_allclose(big[5], big[2], atol=1e-6, equal_nan=True)
    assert not np.allclose(np.nan_to_num(big[0]), np.nan_to_num(big[2]))


def test_klip_annular_starved_targets_drop_vs_safety_floor(ds):
    # anglemax window of 8 deg on a 40 deg sequence: every edge frame has few refs
    p = KLIPParams(k_klip=3, inrad=5, outrad=20, angsep=0.3, anglemax=8.0, n_min_ref=6)
    res, info = klip_annular(ds.cube, ds.angles, p, LOD)
    assert not info["fast"]
    nan_frames = np.isnan(res).all(axis=(1, 2)).sum()
    if info["drop_enabled"]:
        assert info["n_dropped"] == info["n_starved"] == nan_frames > 0
    # tighten so that (n - n_starved) < max(n_min_ref, ceil(0.25 n)) -> dropping disabled
    p2 = KLIPParams(k_klip=3, inrad=5, outrad=20, angsep=3.0, anglemax=8.0, n_min_ref=10)
    res2, info2 = klip_annular(ds.cube, ds.angles, p2, LOD)
    assert info2["n_starved"] == ds.nframes and not info2["drop_enabled"] and info2["n_dropped"] == 0
    assert np.isnan(res2).all(axis=(1, 2)).sum() == 0     # every frame still reduced (fallback refs)
    # n_min_ref=1 keeps everything and drops nothing
    _, info3 = klip_annular(ds.cube, ds.angles, KLIPParams(k_klip=3, inrad=5, outrad=20, anglemax=8.0, n_min_ref=1), LOD)
    assert info3["n_dropped"] == 0


def test_klip_annular_n_ang_zones_and_means(ds):
    p = KLIPParams(k_klip=3, inrad=5, outrad=20, n_ang=4, spat_mean=True, temp_mean=True)
    res, info = klip_annular(ds.cube, ds.angles, p, LOD)
    yy, xx = np.mgrid[0:48, 0:48]
    r = np.hypot(xx - 24, yy - 24)
    zone = (r > 6) & (r < 19)
    assert np.isfinite(res[:, zone]).mean() > 0.95
    assert np.all(np.isfinite(res[:, zone]).sum(axis=1) > 0)
    # a NaN-only cube region does not crash the fast path
    cube = ds.cube.copy()
    cube[:, 20:24, 20:24] = np.nan
    r2, _ = klip_annular(cube, ds.angles, KLIPParams(k_klip=2, inrad=5, outrad=20), LOD)
    assert np.isfinite(r2[:, zone]).any()


# ----------------------------------------------------------------------------
# reference reducer end to end
# ----------------------------------------------------------------------------
def test_klip_reducer_recovers_injected_planets_and_kscan():
    from klip_tpe.metrics import radprof
    from klip_tpe.reducer import KLIPReducer, ReductionRequest, combine_stack

    ds = synthetic_klip_dataset(nframes=40, size=64, pa_span=60)
    px = 0.05
    lam_over_d = 3.89 * px / 206265                     # -> fwhm ~ 4 px
    red = KLIPReducer(ds, px, lam_over_d * 8.2, 8.2, injection_model=GaussianPSF(4.0, star_flux=1e4),
                      outrad_cap=30)
    assert red.fwhm == pytest.approx(4.0, abs=0.01) and red.supports_kscan
    src = [Source(0.8, 45.0, 2e-3), Source(1.0, 200.0, 2e-3)]
    params = {"k_klip": 5, "inrad": 8, "outrad": 26, "bin": 2, "filter": 0, "angsep": 0.5, "anglemax": 40}
    inj = red.reduce(ReductionRequest(params, src))
    clean = red.reduce(ReductionRequest(params, None))
    assert inj.image.shape == clean.image.shape == (64, 64)
    assert inj.meta["n_binned"] == 20 and not inj.meta["fast"] and inj.meta["outrad"] == 28   # zone_pad 2
    s_inj = mawet_peak_snr(radprof(inj.image), [0.8, 1.0], [45, 200], px, red.fwhm)
    s_cln = mawet_peak_snr(radprof(clean.image), [0.8, 1.0], [45, 200], px, red.fwhm)
    assert np.all(s_inj > 6) and np.all(s_inj > s_cln + 4)
    # k-scan cube and its last slice equals the single-k reduction
    scan = red.reduce(ReductionRequest({**params, "k_klip": 4}, src, k_scan=True))
    one = red.reduce(ReductionRequest({**params, "k_klip": 4}, src))
    assert scan.image.shape == (4, 64, 64)
    np.testing.assert_allclose(scan.image[3], one.image, atol=1e-3, equal_nan=True)
    # frame-selection tags are honoured through the chain
    sel = red.reduce(ReductionRequest({**params, "corr_thresh": float(np.sort(ds.tags["corrs"])[10])}, None))
    assert sel.meta["n_kept"] == 30
    # NaN-aware weighted combine used by PartitionedReducer
    np.testing.assert_allclose(combine_stack(np.array([[[1.0, np.nan]], [[3.0, 4.0]]]), [1, 3]), [[2.5, 4.0]])
