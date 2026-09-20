"""``load_calints``: the conventions that are silent when they are wrong.

Everything this function does is a convention -- which frames are science and which are
the reference library, how a roll angle becomes a position angle, where the star is, how
the partitions are cut.  None of them raise when they are wrong; they rotate, mis-centre
or mis-group the data and the run completes.  So they are pinned here, on ``calints``
the test writes itself: no data files and no network.

``test_calints_repair.py`` covers the repair functions directly.  These tests are about
the loader around them.
"""
from __future__ import annotations

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from klip_tpe.backends.spaceklip import load_calints                 # noqa: E402

PX = 0.063


def write(path, target, roll, n_int=2, ny=100, nx=100, v3i_yang=0.0, vparity=1,
          filt="F444W", crpix=None, seed=0, star=None, dq_px=((5, 5),)):
    """One stage-2 product carrying the keywords the loader reads."""
    rng = np.random.default_rng(seed)
    c = crpix if crpix is not None else ((nx - 1) / 2.0, (ny - 1) / 2.0)
    yy, xx = np.mgrid[0:ny, 0:nx]
    sx, sy = star if star is not None else c
    img = 800.0 / (1.0 + (np.hypot(xx - sx, yy - sy) / 4.0) ** 2)
    sci = np.array([img + rng.normal(0.0, 0.5, (ny, nx)) for _ in range(n_int)], np.float32)
    dq = np.zeros_like(sci, np.int32)
    for (py, px_) in dq_px:
        dq[:, py, px_] = 1                                    # DO_NOT_USE
    ph = fits.Header({"TARGPROP": target, "INSTRUME": "NIRCAM", "FILTER": filt,
                      "CORONMSK": "MASKA335R", "PUPIL": "MASKRND"})
    sh = fits.Header({"ROLL_REF": float(roll), "V3I_YANG": float(v3i_yang),
                      "VPARITY": int(vparity), "PIXAR_A2": PX ** 2, "PIXAR_SR": 9.3e-14,
                      "CRPIX1": c[0] + 1, "CRPIX2": c[1] + 1, "BUNIT": "MJy/sr",
                      "FILTER": filt, "INSTRUME": "NIRCAM", "EFFINTTM": 20.0})
    fits.HDUList([fits.PrimaryHDU(header=ph),
                  fits.ImageHDU(sci, header=sh, name="SCI"),
                  fits.ImageHDU(dq, name="DQ")]).writeto(path, overwrite=True)
    return str(path)


@pytest.fixture
def two_rolls(tmp_path):
    return [write(tmp_path / "jw01_calints.fits", "HIP65426", roll=0.0, seed=1),
            write(tmp_path / "jw02_calints.fits", "HIP65426", roll=10.0, seed=2),
            write(tmp_path / "jw03_calints.fits", "REFSTAR", roll=0.0, seed=3)]


def test_position_angle_is_roll_minus_v3i_yang_times_vparity(tmp_path):
    """``PA = ROLL_REF - V3I_YANG * VPARITY``.

    Get the sign of VPARITY wrong and every frame derotates the wrong way by twice the
    yang angle -- which does not fail, it just puts the companion somewhere else.
    """
    f = [write(tmp_path / "a_calints.fits", "T", roll=30.0, v3i_yang=4.0, vparity=1),
         write(tmp_path / "b_calints.fits", "T", roll=30.0, v3i_yang=4.0, vparity=-1)]
    ds, _ = load_calints(f, science_target="T", half_px=20, align=False,
                         partition="all", log=lambda *_: None)
    pas = sorted(set(np.round(ds["sci"].angles, 6)))
    assert pas == [26.0, 34.0], pas                      # 30 - 4*(+1) and 30 - 4*(-1)


def test_partition_roll_cuts_one_dataset_per_roll(two_rolls):
    ds, info = load_calints(two_rolls, science_target="HIP65426", half_px=20, align=False,
                            log=lambda *_: None)
    assert sorted(ds) == ["roll1", "roll2"]
    assert {float(np.unique(d.angles)[0]) for d in ds.values()} == {0.0, 10.0}
    assert all(d.ref_cube is not None and d.ref_cube.shape[0] == 2 for d in ds.values())
    assert info["n_sci"] == 4 and info["n_ref"] == 2


def test_partition_all_keeps_every_science_frame_together(two_rolls):
    """The choice decides what ADI can mean: with one partition per roll every frame in a
    partition shares a PA, so ADI has no reference frames at all."""
    ds, _ = load_calints(two_rolls, science_target="HIP65426", half_px=20, align=False,
                         partition="all", log=lambda *_: None)
    assert list(ds) == ["sci"]
    assert ds["sci"].cube.shape[0] == 4
    assert sorted(set(np.round(ds["sci"].angles, 6))) == [0.0, 10.0]


def test_a_bad_partition_mode_is_refused(two_rolls):
    with pytest.raises(ValueError, match="partition must be"):
        load_calints(two_rolls, science_target="HIP65426", half_px=20, align=False,
                     partition="per-file", log=lambda *_: None)


def test_the_science_target_selects_by_targprop_not_by_order(two_rolls):
    """The reference star is first alphabetically in some programmes; the split must come
    from TARGPROP rather than from whichever file sorted first."""
    ds, info = load_calints(two_rolls, science_target="REFSTAR", half_px=20, align=False,
                            partition="all", log=lambda *_: None)
    assert ds["sci"].cube.shape[0] == 2                   # the single REFSTAR file
    assert ds["sci"].ref_cube.shape[0] == 4               # HIP65426 becomes the library


def test_background_pointings_are_kept_out_of_the_rdi_library(tmp_path):
    """The split is "science, everything else is the library", so a background pointing
    lands in the RDI basis: blank sky used to model the star's diffraction.

    It contributes nothing to subtract with and dilutes the frames that do.  On MIRI
    programme 11225 the background pointings outnumber the real references.
    """
    f = [write(tmp_path / "s_calints.fits", "AU_MIC", roll=0.0, seed=1),
         write(tmp_path / "r_calints.fits", "AU_MIC_PSF_REFERENCE", roll=0.0, seed=2),
         write(tmp_path / "b1_calints.fits", "BKG-AU_MIC", roll=0.0, seed=3),
         write(tmp_path / "b2_calints.fits", "BKG-AU_MIC_PSF_REFERENCE", roll=0.0, seed=4),
         write(tmp_path / "b3_calints.fits", "AU_MIC-BACKGROUND", roll=0.0, seed=5)]
    msgs = []
    ds, info = load_calints(f, science_target="AU_MIC", half_px=20, align=False,
                            partition="all", log=msgs.append)
    assert ds["sci"].cube.shape[0] == 2                   # the one science file
    assert ds["sci"].ref_cube.shape[0] == 2               # the one reference, not the three bkg
    assert info["n_ref"] == 2
    assert any("background pointing" in m for m in msgs)


def test_a_reference_named_after_its_target_stays_in_the_library(tmp_path):
    """The science match is ``startswith``, so "HIP65426" finds "HIP-65426".

    A programme that names its reference after its target defeats that:
    ``AU_Mic_psf_reference`` normalises to AUMICPSFREFERENCE and starts with AUMIC, so it
    was classified as SCIENCE -- its frames derotated and stacked with the target's, and
    the RDI library empty.  Silently, since the only symptom is a science frame count
    nobody checks against the programme.
    """
    f = [write(tmp_path / "s_calints.fits", "AU_Mic", roll=0.0, seed=1),
         write(tmp_path / "r_calints.fits", "AU_Mic_psf_reference", roll=0.0, seed=2)]
    msgs = []
    ds, info = load_calints(f, science_target="AU_Mic", half_px=20, align=False,
                            partition="all", log=msgs.append)
    assert info["n_sci"] == 2 and info["n_ref"] == 2
    assert ds["sci"].ref_cube is not None and ds["sci"].ref_cube.shape[0] == 2
    assert any("named" in m and "PSF reference" in m for m in msgs)


def test_the_reference_can_still_be_asked_for_as_the_science_target(tmp_path):
    """Reducing the reference star to check it is a fair thing to want."""
    f = [write(tmp_path / "s_calints.fits", "AU_Mic", roll=0.0, seed=1),
         write(tmp_path / "r_calints.fits", "AU_Mic_psf_reference", roll=0.0, seed=2)]
    ds, info = load_calints(f, science_target="AU_Mic_psf_reference", half_px=20,
                            align=False, partition="all", log=lambda *_: None)
    assert info["n_sci"] == 2 and info["n_ref"] == 2


def test_a_set_with_no_reference_says_the_reduction_will_be_adi(tmp_path):
    """make_reducer drops from ADI+RDI to ADI on a dataset with no ref_cube, and reports
    it as one word in a log line.  Say it plainly at the point it becomes true."""
    f = [write(tmp_path / "only_calints.fits", "TARG", roll=0.0)]
    msgs = []
    ds, info = load_calints(f, science_target="TARG", half_px=20, align=False,
                            partition="all", log=msgs.append)
    assert ds["sci"].ref_cube is None and info["n_ref"] == 0
    assert any("ADI only" in m for m in msgs)


def test_an_unmatched_science_target_says_so(two_rolls):
    with pytest.raises(ValueError, match="no science files matching"):
        load_calints(two_rolls, science_target="NOTHERE", half_px=20, align=False,
                     log=lambda *_: None)


def test_targprop_matching_ignores_dashes_and_underscores(tmp_path):
    f = [write(tmp_path / "x_calints.fits", "HIP-65426", roll=0.0),
         write(tmp_path / "y_calints.fits", "REF_STAR", roll=0.0)]
    ds, _ = load_calints(f, science_target="HIP65426", half_px=20, align=False,
                         partition="all", log=lambda *_: None)
    assert ds["sci"].cube.shape[0] == 2


def test_the_crop_is_odd_so_the_star_lands_on_the_centre_pixel(two_rolls):
    """An even crop with the star on an integer pixel leaves it half a pixel off in each
    axis -- 0.7 px radially, which at 13 px separation is a 3 degree position-angle error
    and a throughput mismatch between a companion and the fakes calibrating it."""
    for half in (20, 25, 31):
        ds, info = load_calints(two_rolls, science_target="HIP65426", half_px=half,
                                align=False, log=lambda *_: None)
        n = 2 * half + 1
        assert info["crop_px"] == n and n % 2 == 1
        assert all(d.cube.shape[-2:] == (n, n) for d in ds.values())


def test_star_center_overrides_crpix(tmp_path):
    """CRPIX is the aperture reference point, not a measured star position -- identical in
    every file of a programme, dithers included."""
    off = (60.0, 44.0)
    f = [write(tmp_path / "s_calints.fits", "T", roll=0.0, star=off, seed=5)]
    half = 15
    ds_c, info_c = load_calints(f, science_target="T", half_px=half, align=False,
                                log=lambda *_: None)
    ds_s, info_s = load_calints(f, science_target="T", half_px=half, align=False,
                                star_center=off, log=lambda *_: None)
    assert info_c["star_center"] == info_c["crpix"]           # fell back to CRPIX
    assert info_s["star_center"] == off
    mid = half
    a = np.asarray(ds_c["roll1"].cube[0], float)
    b = np.asarray(ds_s["roll1"].cube[0], float)
    # with the true centre the star's peak is at the middle of the crop; with CRPIX it is not
    assert np.unravel_index(np.nanargmax(b), b.shape) == (mid, mid)
    assert np.unravel_index(np.nanargmax(a), a.shape) != (mid, mid)


def test_repair_dq_fills_the_flagged_pixels_and_leaves_the_rest(tmp_path):
    """``repair='dq'`` is the default for a reason: the 'sigma' path median-filters the
    whole PSF, star and companion alike."""
    f = [write(tmp_path / "r_calints.fits", "T", roll=0.0, dq_px=((30, 30), (31, 40)), seed=7)]
    ds_on, info_on = load_calints(f, science_target="T", half_px=45, align=False,
                                  log=lambda *_: None)
    ds_off, _ = load_calints(f, science_target="T", half_px=45, align=False, repair=False,
                             log=lambda *_: None)
    assert info_on["repair"] == "dq"
    assert np.isfinite(np.asarray(ds_on["roll1"].cube)).all()
    assert np.isnan(np.asarray(ds_off["roll1"].cube)).any()
    assert all(fr["n_dq"] == 2 for fr in info_on["frames"] if fr["role"] == "sci")


def test_info_carries_the_scale_and_filter_the_rest_of_the_run_needs(two_rolls):
    ds, info = load_calints(two_rolls, science_target="HIP65426", half_px=20, align=False,
                            log=lambda *_: None)
    assert info["pxscale"] == pytest.approx(PX, abs=1e-9)
    assert info["filter"] == "F444W"
    assert info["bunit"] == "MJy/sr"
    for d in ds.values():
        assert d.meta["pxscale"] == pytest.approx(PX, abs=1e-9)
        assert d.meta["header"]["INSTRUME"] == "NIRCAM"


def test_no_files_is_an_error_not_an_empty_run():
    with pytest.raises(ValueError, match="no files"):
        load_calints([], log=lambda *_: None)
