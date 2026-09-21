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


def test_a_mixed_filter_set_is_refused_rather_than_stacked(tmp_path):
    """The wavelength and pixel scale came from the FIRST file and were applied to every
    frame, so a directory holding more than one filter was stacked into a single dataset
    at a single lambda -- different masks, different lambda/D, no error.

    A whole programme downloaded at once is exactly that: GO 1386 holds HIP 65426 in
    F1140C and F1550C and HD 141569A in all three, together.
    """
    f = [write(tmp_path / "a_calints.fits", "T", roll=0.0, filt="F1140C", seed=1),
         write(tmp_path / "b_calints.fits", "T", roll=0.0, filt="F1550C", seed=2)]
    with pytest.raises(ValueError, match="span 2 filters"):
        load_calints(f, science_target="T", half_px=20, align=False, log=lambda *_: None)


def test_filter_selects_the_files_it_names(tmp_path):
    f = [write(tmp_path / "a_calints.fits", "T", roll=0.0, filt="F1140C", seed=1),
         write(tmp_path / "b_calints.fits", "T", roll=0.0, filt="F1550C", seed=2),
         write(tmp_path / "c_calints.fits", "R", roll=0.0, filt="F1140C", seed=3)]
    ds, info = load_calints(f, science_target="T", half_px=20, align=False, filter="F1140C",
                            partition="all", log=lambda *_: None)
    assert info["filter"] == "F1140C"
    assert ds["sci"].cube.shape[0] == 2            # the one F1140C science file
    assert ds["sci"].ref_cube.shape[0] == 2        # its F1140C reference, not the F1550C


def test_a_filter_that_is_not_there_is_named(tmp_path):
    f = [write(tmp_path / "a_calints.fits", "T", roll=0.0, filt="F1140C")]
    with pytest.raises(ValueError, match="no files with FILTER='F2300C'"):
        load_calints(f, science_target="T", half_px=20, align=False, filter="F2300C",
                     log=lambda *_: None)


def test_a_single_filter_set_still_needs_no_argument(tmp_path):
    """The refusal must only fire when there is a real ambiguity."""
    f = [write(tmp_path / "a_calints.fits", "T", roll=0.0, filt="F1140C", seed=1),
         write(tmp_path / "b_calints.fits", "R", roll=0.0, filt="F1140C", seed=2)]
    ds, info = load_calints(f, science_target="T", half_px=20, align=False, partition="all",
                            log=lambda *_: None)
    assert info["filter"] == "F1140C"


def test_a_large_dq_fraction_is_called_out_not_just_counted(tmp_path):
    """Flagged pixels inside the returned stamp are a smooth interpolation of their
    neighbours: no independent information, but they do enter the KLIP basis, where smooth
    and large is what the leading components are made of."""
    ny = nx = 60
    many = tuple((y, x) for y in range(10, 40) for x in range(10, 40))   # 900 of 3600 = 25%
    f = [write(tmp_path / "s_calints.fits", "T", roll=0.0, ny=ny, nx=nx, dq_px=many, seed=1)]
    msgs = []
    ds, info = load_calints(f, science_target="T", half_px=20, align=False,
                            partition="all", log=msgs.append)
    assert info["repaired_fraction"] == pytest.approx(0.25, abs=0.01)
    assert info["repaired_fraction_crop"] > 0.25            # the block is centred on the star
    assert any("over the whole subarray" in m for m in msgs)
    assert any("no independent information" in m for m in msgs)


def test_the_dq_warning_counts_the_stamp_not_the_subarray(tmp_path):
    """The number that bears on the reduction is the one inside the stamp.

    A MIRI coronagraphic subarray is about a quarter unilluminated and DQ-flagged, but that
    is the region outside the coronagraph field: of HIP 65426 F1140C's 17,790 flagged pixels
    per frame, 44 are inside the 81x81 stamp (0.7%).  Warning on the subarray figure -- 27%
    of every frame replaced by an interpolation! -- points at pixels nothing downstream ever
    sees, and it sent one real debugging session after the wrong thing entirely.
    """
    ny = nx = 60
    rim = tuple((y, x) for y in range(ny) for x in range(nx)
                if not (14 <= y < 46 and 14 <= x < 46))      # flagged OUTSIDE the stamp
    f = [write(tmp_path / "s_calints.fits", "T", roll=0.0, ny=ny, nx=nx, dq_px=rim, seed=1)]
    msgs = []
    ds, info = load_calints(f, science_target="T", half_px=12, align=False,
                            partition="all", log=msgs.append)
    assert info["repaired_fraction"] > 0.20, "the subarray really is a quarter flagged"
    assert info["repaired_fraction_crop"] == 0.0, "and none of it is in the stamp"
    assert not any("no independent information" in m for m in msgs), \
        "warned about pixels that are not in the returned data"
    assert np.isfinite(np.asarray(ds["sci"].cube, float)).all()


def test_a_small_dq_fraction_says_nothing_extra(tmp_path):
    """The warning is only worth having if a normal frame does not trigger it."""
    f = [write(tmp_path / "s_calints.fits", "T", roll=0.0, dq_px=((5, 5), (6, 6)), seed=1)]
    msgs = []
    ds, info = load_calints(f, science_target="T", half_px=20, align=False,
                            partition="all", log=msgs.append)
    assert info["repaired_fraction"] < 0.01
    assert not any("no independent information" in m for m in msgs)


# ------------------------------------------------- the two bugs that killed a real MIRI run

def test_a_gap_the_repair_cannot_close_does_not_take_the_whole_frame(tmp_path):
    """The bug that killed the first HIP 65426 F1140C run, and it was nowhere near the
    coronagraph.

    A MIRI MASK1140 subarray is ~28% DQ-flagged -- the unilluminated border outside the
    coronagraph field -- and that region is far too wide for ``fill_dq_neighbours`` to close
    from its rim, so 13% of the frame is still NaN when the alignment runs.  ``ndimage.shift``
    at ``order=3`` prefilters with a *recursive* spline filter, so those NaNs propagate to
    **every** pixel: the frame went in 13% NaN and came out 99.2% NaN, ``load_calints``
    returned a cube that was 100% NaN, and the run died two steps later in the reducer with
    "every frame was dropped as empty".  Nothing in between said a word.
    """
    ny = nx = 60
    rim = tuple((y, x) for y in range(ny) for x in range(nx)
                if x < 14 or x >= 46)                        # a 14-px-wide unfillable border
    f = [write(tmp_path / "s_calints.fits", "T", roll=0.0, ny=ny, nx=nx, dq_px=rim, seed=1)]
    ds, info = load_calints(f, science_target="T", half_px=12, align=True, partition="all",
                            log=lambda *_: None)
    cube = np.asarray(ds["sci"].cube, float)
    assert np.isfinite(cube).all(), \
        f"{100 * np.mean(~np.isfinite(cube)):.0f}% of the returned cube is NaN"
    assert info["nonfinite_fraction"] == 0.0
    # and the frame is still the star, not a shifted smear of nothing
    assert np.unravel_index(np.nanargmax(np.nanmedian(cube, axis=0)), cube.shape[-2:]) == (12, 12)


def test_shift_keeping_gaps_marks_what_the_spline_touched_and_nothing_else():
    """``shift_keeping_gaps`` has to do two opposite things: not let a gap spread, and not
    invent a gap where there was none.  A frame with no gaps must be untouched."""
    from scipy import ndimage
    from klip_tpe.backends.spaceklip import shift_keeping_gaps
    rng = np.random.default_rng(0)
    im = rng.normal(10.0, 1.0, (40, 40))
    np.testing.assert_array_equal(shift_keeping_gaps(im, (-0.3, 0.4)),
                                  ndimage.shift(im, (-0.3, 0.4), order=3))
    g = im.copy()
    g[20, 20] = np.nan
    out = shift_keeping_gaps(g, (-0.3, 0.4))
    bad = ~np.isfinite(out)
    assert bad.any(), "the gap has to survive as a gap"
    assert bad.sum() < 60, f"one gap became {int(bad.sum())} px -- the spline was let loose"
    assert not bad[:6].any() and not bad[-6:].any(), "the frame border is not a gap"
    assert np.isfinite(ndimage.shift(np.where(bad, 0.0, out), (0, 0))).all()
    # the naive call is the thing being avoided
    assert np.mean(~np.isfinite(ndimage.shift(g, (-0.3, 0.4), order=3))) > 0.9


def test_an_unmeasurable_offset_is_reported_not_guessed(tmp_path):
    """``_xs`` used to return the highest corner of its search box when there was no peak in
    it.  On HIP 65426 F1140C that was a confident six-pixel shift for every frame: the 41
    integrations of one exposure have no offset to register against each other, and
    ``np.argmax`` of an all-NaN correlation returns 0, which decodes to ``(-6, -6)``.
    """
    ny = nx = 60
    # flat frames: no structure to register, so no peak should be found
    f = [write(tmp_path / "s_calints.fits", "T", roll=0.0, ny=ny, nx=nx, seed=1)]
    import astropy.io.fits as _f
    with _f.open(f[0], mode="update") as h:
        rng = np.random.default_rng(3)
        h["SCI"].data = np.asarray(rng.normal(0.0, 1.0, h["SCI"].data.shape), np.float32)
    msgs = []
    ds, info = load_calints(f, science_target="T", half_px=12, align=True, partition="all",
                            log=msgs.append)
    off = [(p.get("dx"), p.get("dy")) for p in info["frames"]]
    assert all(abs(dx) < 6 and abs(dy) < 6 for dx, dy in off), \
        f"a corner of the search box was returned as a measurement: {off}"
    if any(not p.get("registered", True) for p in info["frames"]):
        assert any("left UNSHIFTED" in m for m in msgs), "an unmeasured offset must be said"
    assert np.isfinite(np.asarray(ds["sci"].cube, float)).all()
