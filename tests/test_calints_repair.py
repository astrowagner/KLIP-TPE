"""load_calints must not rewrite pixels the pipeline did not flag.

Until 2026-09-16 its repair was a 5x5 median filter plus a 7-sigma clip against the
FRAME-WIDE robust scatter of the residual.  On a coronagraphic frame that scatter is set by
empty sky, so every structured pixel -- the star's six-lobed Lyot pattern and the companion's
three-bar core alike -- exceeded it: ~4,600-5,500 pixels per 320x320 ERS 1386 frame were
replaced by the local median, of which only 1,564 were DQ-flagged.  HIP 65426 b came out as
one smeared blob at 36% of its peak instead of Carter et al. (2023)'s Fig. 3 core, a
different set of pixels was rewritten in every frame (so the residual looked roll-dependent
and was mistaken for a speckle), and because fakes are injected AFTER the repair they kept
their cores while the companion lost 1.9x -- which a 0.561 "optics transmission" then hid.

The repair now fills DQ pixels from their neighbours (spaceKLIP's method) and touches
nothing else.  These tests run on synthetic frames so they need no data files.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from klip_tpe.backends.spaceklip import fill_dq_neighbours, sigma_clip_repair


def _coronagraphic_frame(seed=0, ny=160, nx=160):
    """Empty sky at sigma = 1 with a bright structured PSF: a smooth halo carrying sharp
    lobes and a small three-bar 'companion' core, like a NIRCam MASK335R field."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = 79.5, 79.5
    r = np.hypot(xx - cx, yy - cy)
    im = 300.0 * np.exp(-r / 12.0)
    for k in range(6):                                          # six lobes at 20 px
        a = np.deg2rad(60 * k + 15)
        lx, ly = cx + 20 * np.cos(a), cy + 20 * np.sin(a)
        im += 80.0 * np.exp(-((xx - lx) ** 2 + (yy - ly) ** 2) / 4.0)
    for _ in range(60):                                         # sharp speckles inside 40 px
        rr, aa = rng.uniform(8, 40), rng.uniform(0, 2 * np.pi)
        sx, sy = cx + rr * np.cos(aa), cy + rr * np.sin(aa)
        im += rng.uniform(30, 120) * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / 1.6)
    for dy in (-2, 0, 2):                                       # a hamburger core at 13 px, PA 150
        bx, by = cx + 13 * np.cos(np.deg2rad(240)), cy + 13 * np.sin(np.deg2rad(240)) + dy
        im += 20.0 * np.exp(-((xx - bx) ** 2) / 3.0 - ((yy - by) ** 2) / 0.6)
    return im + rng.normal(0.0, 1.0, im.shape)


def test_fill_dq_neighbours_touches_only_the_flagged_pixels():
    im = _coronagraphic_frame()
    flagged = im.copy()
    rng = np.random.default_rng(1)
    idx = rng.choice(im.size, 800, replace=False)
    flagged.flat[idx] = np.nan
    out, n = fill_dq_neighbours(flagged)
    assert n == 800
    assert np.isfinite(out).all(), "every flagged pixel is filled"
    mask = np.zeros(im.shape, bool); mask.flat[idx] = True
    assert np.array_equal(out[~mask], im[~mask]), "an unflagged pixel changed"
    # and the fill is a neighbourhood median: a flagged pixel on the halo stays on the halo
    # (a speckle core can lose its peak -- that is the price of any interpolation, and it is
    # paid on the ~1% of pixels the pipeline flagged, not on the whole PSF)
    assert np.median(np.abs(out[mask] - im[mask])) < 2.0


def test_fill_dq_neighbours_closes_a_cluster_from_its_rim():
    im = _coronagraphic_frame()
    flagged = im.copy(); flagged[70:75, 70:75] = np.nan          # a 5x5 hole, nothing finite inside
    out, n = fill_dq_neighbours(flagged)
    assert n == 25 and np.isfinite(out).all()


def test_the_old_sigma_clip_rewrote_the_psf_and_the_new_repair_does_not():
    """The exact failure: on a frame with 800 flagged pixels the sigma clip rewrites
    thousands, all of them on the PSF; the DQ fill rewrites 800."""
    im = _coronagraphic_frame()
    flagged = im.copy()
    idx = np.random.default_rng(2).choice(im.size, 800, replace=False)
    flagged.flat[idx] = np.nan
    old, n_hit, n_bad = sigma_clip_repair(flagged)
    new, n_new = fill_dq_neighbours(flagged)
    assert n_bad == 800 and n_new == 800
    assert n_hit > 800 + 500, f"the sigma clip should have rewritten far more than the flagged 800, got {n_hit}"
    # (on the real 320x320 F444W frames it is ~4,600-5,500 of which 1,564 are flagged)
    yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    r = np.hypot(xx - 79.5, yy - 79.5)
    changed_old = (old != np.nan_to_num(flagged, nan=old)) & np.isfinite(flagged)
    assert np.median(r[changed_old]) < 30, "the rewritten pixels are the PSF, not the sky"
    # the companion's core survives the new repair and not the old one
    bx, by = 79.5 + 13 * np.cos(np.deg2rad(240)), 79.5 + 13 * np.sin(np.deg2rad(240))
    box = (slice(int(by) - 4, int(by) + 5), slice(int(bx) - 4, int(bx) + 5))
    assert new[box].max() == pytest.approx(im[box].max(), abs=1e-9) or np.isnan(flagged[box]).any()
    assert old[box].max() < 0.85 * im[box].max(), "the old repair clipped the three-bar core"
    # ... and it is the brightest pixels it goes for: most of the 200 brightest inside 40 px
    # were rewritten (on the synthetic frame the medians are close to the peaks, so the
    # numerical damage is mild here; on the real F444W frames it took the companion to 36%)
    top = np.argsort(im[r < 40].ravel())[-200:]
    assert changed_old[r < 40].ravel()[top].mean() > 0.4, "the sigma clip spares the PSF peaks?"


def test_load_calints_default_is_the_dq_fill_and_sigma_is_opt_in(tmp_path):
    """End to end on two tiny synthetic calints files (one 'science' roll, one 'reference'):
    the default rewrites exactly the DQ pixels; repair='sigma' is still reachable, logs its
    warning and rewrites more; an unknown mode is refused."""
    from astropy.io import fits
    from klip_tpe.backends.spaceklip import load_calints

    def write(path, targ, roll, seed):
        n, ny, nx = 2, 120, 120
        data = np.stack([_coronagraphic_frame(seed + i, ny, nx) for i in range(n)]).astype(np.float32)
        dq = np.zeros((n, ny, nx), np.uint32)
        rng = np.random.default_rng(seed + 10)
        for i in range(n):
            dq[i].flat[rng.choice(ny * nx, 300, replace=False)] = 1
        ph = fits.Header(); ph["TARGPROP"] = targ; ph["FILTER"] = "F444W"; ph["EFFINTTM"] = 10.0
        ph["INSTRUME"] = "NIRCAM"; ph["CORONMSK"] = "MASKA335R"; ph["PUPIL"] = "MASKRND"
        sh = fits.Header(); sh["ROLL_REF"] = roll; sh["V3I_YANG"] = 0.0; sh["VPARITY"] = -1
        sh["PIXAR_A2"] = 0.0039; sh["PIXAR_SR"] = 9.2e-14; sh["BUNIT"] = "MJy/sr"
        sh["CRPIX1"] = 60.5; sh["CRPIX2"] = 60.5
        fits.HDUList([fits.PrimaryHDU(header=ph), fits.ImageHDU(data, header=sh, name="SCI"),
                      fits.ImageHDU(dq, name="DQ")]).writeto(path, overwrite=True)

    a, b = str(tmp_path / "jw0_sci_calints.fits"), str(tmp_path / "jw1_ref_calints.fits")
    write(a, "HIP-65426", 110.0, 0); write(b, "HIP-68245", 110.0, 5)
    logs = []
    dsets, info = load_calints([a, b], science_target="HIP65426", half_px=20, log=logs.append)
    assert info["repair"] == "dq"
    assert [f["n_repaired"] for f in info["frames"]] == [300, 300, 300, 300]
    assert all(f["repair"] == "dq" for f in info["frames"])
    assert not any("UNFLAGGED" in m for m in logs)
    logs.clear()
    _, info_s = load_calints([a, b], science_target="HIP65426", half_px=20, repair="sigma", log=logs.append)
    assert info_s["repair"] == "sigma"
    assert min(f["n_repaired"] for f in info_s["frames"]) > 300
    assert any("UNFLAGGED" in m for m in logs), "the opt-in old repair must say what it is doing"
    _, info_n = load_calints([a, b], science_target="HIP65426", half_px=20, repair=False, log=logs.append)
    assert info_n["repair"] == "none" and "n_repaired" not in info_n["frames"][0]
    with pytest.raises(ValueError):
        load_calints([a, b], science_target="HIP65426", half_px=20, repair="median", log=logs.append)


@pytest.mark.skipif(not os.path.exists(os.path.expanduser("~/.klip_tpe/data/jwst_hip65426")),
                    reason="ERS 1386 calints not downloaded")
def test_on_the_real_frames_only_the_dq_pixels_are_rewritten():
    import glob
    from klip_tpe.backends.spaceklip import load_calints
    files = sorted(glob.glob(os.path.expanduser("~/.klip_tpe/data/jwst_hip65426/jw*calints.fits")))
    if len(files) < 3:
        pytest.skip("not enough calints files")
    _, info = load_calints(files, science_target="HIP65426", star_center=(150.54, 172.98), log=lambda s: None)
    for f in info["frames"]:
        assert f["n_repaired"] == f["n_dq"], f"{f['file']}[{f['integration']}]: rewrote {f['n_repaired']} != DQ {f['n_dq']}"
    assert 1500 < info["frames"][0]["n_dq"] < 1700, "the ERS 1386 F444W DQ count is ~1,564 per frame"
