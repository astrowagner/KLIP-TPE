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
          filt="F444W", crpix=None, seed=0, star=None, dq_px=((5, 5),),
          bkgsub=False, level=0.0, starflux=800.0):
    """One stage-2 product carrying the keywords the loader reads."""
    rng = np.random.default_rng(seed)
    c = crpix if crpix is not None else ((nx - 1) / 2.0, (ny - 1) / 2.0)
    yy, xx = np.mgrid[0:ny, 0:nx]
    sx, sy = star if star is not None else c
    img = starflux / (1.0 + (np.hypot(xx - sx, yy - sy) / 4.0) ** 2) + float(level)
    sci = np.array([img + rng.normal(0.0, 0.5, (ny, nx)) for _ in range(n_int)], np.float32)
    dq = np.zeros_like(sci, np.int32)
    for (py, px_) in dq_px:
        dq[:, py, px_] = 1                                    # DO_NOT_USE
    ph = fits.Header({"TARGPROP": target, "INSTRUME": "NIRCAM", "FILTER": filt,
                      "CORONMSK": "MASKA335R", "PUPIL": "MASKRND"})
    if bkgsub:
        ph["S_BKDSUB"] = "COMPLETE"          # Image2 already removed the dedicated background
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


def _load(files, msgs, **kw):
    return load_calints(files, science_target="T", half_px=20, align=False, partition="all",
                        log=msgs.append, **kw)

# ------------------------------------ the dedicated background, and the mixture in an archive

def _bkg_set(tmp_path, sci_done, ref_done, with_bkg=True, level=20.0):
    """A programme whose science target got a dedicated background and whose PSF reference
    did not -- which is the ordinary case, not a corner one."""
    f = [write(tmp_path / "s_calints.fits", "T", roll=0.0, seed=1,
               bkgsub=sci_done, level=0.0 if sci_done else level),
         write(tmp_path / "r_calints.fits", "REFSTAR", roll=5.0, seed=2,
               bkgsub=ref_done, level=0.0 if ref_done else level)]
    if with_bkg:
        f.append(write(tmp_path / "b_calints.fits", "T-BACKGROUND", roll=0.0, seed=3,
                       level=level, starflux=0.0))
    return f


def test_the_reference_gets_the_background_the_science_already_had_removed(tmp_path):
    """The mixture an archive download hides, and why it cannot be stacked.

    A programme's SCIENCE targets get dedicated background pointings and Image2 subtracts
    them (S_BKDSUB='COMPLETE'); its pure PSF REFERENCE stars usually do not get one, so the
    step never runs on them.  On ERS 1386 at F1140C that is 14 of 16 reference files arriving
    with a ~19 MJy/sr sky pedestal and the 4QPM glow sticks, against science frames with
    neither -- one KLIP library, two sky levels, and its dominant common mode the background
    rather than the star.
    """
    msgs = []
    ds, info = _load(_bkg_set(tmp_path, sci_done=True, ref_done=False), msgs)
    sci = np.asarray(ds["sci"].cube, float)
    ref = np.asarray(ds["sci"].ref_cube, float)
    assert abs(np.nanmedian(ref) - np.nanmedian(sci)) < 2.0, \
        f"reference sits {np.nanmedian(ref) - np.nanmedian(sci):.1f} above the science"
    assert any("subtracted the blank-sky median" in m for m in msgs)


def test_frames_the_pipeline_already_did_are_not_subtracted_twice(tmp_path):
    """Subtracting the background from a frame that already had it removed would push it
    negative by the sky level, which is just as wrong in the other direction."""
    ds, _ = _load(_bkg_set(tmp_path, sci_done=True, ref_done=True), [])
    sci = np.asarray(ds["sci"].cube, float)
    assert np.nanmedian(sci) > -1.0, f"science was double-subtracted: {np.nanmedian(sci):.1f}"


def test_a_mixture_with_no_background_to_fix_it_refuses(tmp_path):
    """Better to stop than to stack two sky levels into one basis and say nothing."""
    with pytest.raises(ValueError, match="one KLIP library mixes two sky levels|mixes two sky"):
        _load(_bkg_set(tmp_path, sci_done=True, ref_done=False, with_bkg=False), [])


def test_a_uniformly_unsubtracted_set_is_allowed_but_said_out_loud(tmp_path):
    """Self-consistent, so KLIP still works -- but the sky is in the basis and in the noise
    the objective is measured against, which the run's record should show."""
    msgs = []
    ds, _ = _load(_bkg_set(tmp_path, sci_done=False, ref_done=False, with_bkg=False), msgs)
    assert any("the sky is still in it" in m for m in msgs)


def test_background_false_stacks_it_anyway(tmp_path):
    msgs = []
    ds, _ = _load(_bkg_set(tmp_path, sci_done=True, ref_done=False), msgs, background=False)
    ref = np.asarray(ds["sci"].ref_cube, float)
    sci = np.asarray(ds["sci"].cube, float)
    assert np.nanmedian(ref) - np.nanmedian(sci) > 10.0, "the pedestal should still be there"
    assert not any("subtracted the blank-sky median" in m for m in msgs)


def test_blank_sky_medians_over_every_integration(tmp_path):
    """A cosmic ray in one background integration must not print itself onto every science
    frame, so the combination is a median over integrations and not a mean."""
    from klip_tpe.backends.spaceklip import blank_sky
    f = write(tmp_path / "b_calints.fits", "T-BACKGROUND", roll=0.0, n_int=5, level=20.0,
              starflux=0.0, seed=4)
    with fits.open(f, mode="update") as h:
        h["SCI"].data[2, 50, 50] = 1e5                   # one hot integration
    bg = blank_sky([f])
    assert abs(bg[50, 50] - 20.0) < 2.0, f"the outlier survived: {bg[50, 50]:.1f}"


# ----------------------------------------------------------------------------
# detector-frame destriping
# ----------------------------------------------------------------------------
def _striped_frame(ny=224, nx=288, cx=144.0, cy=112.0, seed=5):
    """A MIRI-shaped subarray: bright coronagraphic PSF, glow sticks along the 4QPM
    boundaries, and a per-row offset as large as the pixel noise (what the real data
    measures)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    im = rng.normal(0.0, 3.0, (ny, nx))
    im += 600.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0 ** 2))   # the star
    im += 40.0 * np.exp(-((yy - cy) ** 2) / (2 * 2.5 ** 2))                     # glow stick
    rows = rng.normal(0.0, 3.0, (ny, 1))                                        # the stripes
    return im + rows, rows


def test_destripe_removes_the_row_offset_without_eating_the_psf():
    from klip_tpe.backends.spaceklip import destripe_detector

    ny, nx, cx, cy = 224, 288, 144.0, 112.0
    im, rows = _striped_frame(ny, nx, cx, cy)
    out, stats = destripe_detector(im[None], (cx, cy))
    yy, xx = np.mgrid[0:ny, 0:nx]
    sky = ~((np.hypot(xx - cx, yy - cy) < 45) | (np.abs(yy - cy) < 12) | (np.abs(xx - cx) < 12))

    def scatter(a):
        v = a[sky]
        return float(np.nanmedian(np.abs(v - np.nanmedian(v))) * 1.4826)

    assert scatter(out[0]) < 0.75 * scatter(im), "the stripes are still there"
    assert stats["row_sigma"] > 1.0, "the row offsets were not even detected"
    # the star survives: a per-row constant cannot take more than the offset itself
    assert out[0].max() > 0.9 * im.max()


def test_destripe_measures_the_offset_on_sky_not_through_the_star():
    """The whole reason this runs on the full subarray.  Doubling the star's brightness
    must not change the offsets, because the star is masked out of the estimate."""
    from klip_tpe.backends.spaceklip import destripe_detector

    ny, nx, cx, cy = 224, 288, 144.0, 112.0
    im, _ = _striped_frame(ny, nx, cx, cy)
    bright = im + 600.0 * np.exp(
        -((np.mgrid[0:ny, 0:nx][1] - cx) ** 2 + (np.mgrid[0:ny, 0:nx][0] - cy) ** 2) / (2 * 3.0 ** 2))
    a, _ = destripe_detector(im[None], (cx, cy))
    b, _ = destripe_detector(bright[None], (cx, cy))
    # away from the star the two must be destriped identically
    far = np.zeros((ny, nx), bool)
    far[:, :60] = True
    assert np.allclose(a[0][far], b[0][far], atol=1e-9)


def test_destripe_is_nan_safe_and_leaves_empty_rows_alone():
    from klip_tpe.backends.spaceklip import destripe_detector

    im, _ = _striped_frame()
    im[10, :] = np.nan                       # a wholly dead row
    out, _ = destripe_detector(im[None], (144.0, 112.0))
    assert np.all(~np.isfinite(out[0][10]))  # still NaN, not turned into zeros
    assert np.isfinite(out[0][50]).all()


def test_destripe_leaves_no_line_uncorrected_through_the_star():
    """The bug this guards: the star disc and the 4QPM boundary band cross AT the star, so
    together they emptied 24 rows and 24 columns outright on MASK1140 -- rows 101-124 and
    columns 108-131, straight through the target, and the crop is centred on exactly that.
    A line with no offset estimate keeps its offset while every neighbour loses theirs, which
    turns a gradient into a step: measured on a real F1140C integration, line-to-line steps
    inside the 81x81 crop went from 1.87 raw to 4.05 'destriped'.  The global sky scatter
    never noticed (1.826 against 1.827), because the damage sat only where the science is.
    """
    from klip_tpe.backends.spaceklip import destripe_detector

    ny, nx, cx, cy = 224, 288, 144.0, 112.0
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:ny, 0:nx]
    im = rng.normal(0.0, 3.0, (ny, nx))
    im += 600.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0 ** 2))
    im += 40.0 * np.exp(-((yy - cy) ** 2) / (2 * 2.5 ** 2))          # glow stick
    rows = rng.normal(0.0, 3.0, (ny, 1))                             # the striping
    cols = rng.normal(0.0, 1.0, (1, nx))
    out, _ = destripe_detector((im + rows + cols)[None], (cx, cy))

    # every row that the star+boundary mask would have emptied must still be corrected: its
    # residual offset has to be far below the stripe amplitude it came in with
    disc_and_band = (np.hypot(xx - cx, yy - cy) < 45) | (np.abs(yy - cy) < 12) | (np.abs(xx - cx) < 12)
    dead_rows = [j for j in range(ny) if disc_and_band[j, :].all()]
    dead_cols = [i for i in range(nx) if disc_and_band[:, i].all()]
    assert dead_rows and dead_cols, "fixture must reproduce the geometry that caused the bug"

    sky = ~((np.hypot(xx - cx, yy - cy) < 55) | (np.abs(yy - cy) < 14) | (np.abs(xx - cx) < 14))
    res = np.where(sky, out[0], np.nan)
    live = [j for j in range(ny) if np.isfinite(res[j]).sum() > 20 and j not in dead_rows]
    ref = float(np.nanstd([np.nanmedian(res[j]) for j in live]))
    for j in dead_rows:
        v = res[j][np.isfinite(res[j])]
        if v.size > 20:
            assert abs(float(np.median(v))) < 5.0 * max(ref, 0.2), (
                f"row {j} was left uncorrected: offset {np.median(v):.2f} against a "
                f"corrected-row scatter of {ref:.2f}")


def test_destripe_recovers_the_injected_offset_off_the_glow_stick():
    """Inject a known per-row offset and check it comes back off the rows that have sky in
    them -- which is every row except the handful crossing the glow stick.

    Those are left alone on purpose.  The glow stick sits ~13 sigma above the sky right
    across its rows, so the sigma clip rejects them entirely and they get no estimate.  The
    obvious repair -- use their unclipped median instead -- was measured and is far worse
    than the disease: on a real F1140C integration the same rule empties 31 rows and 91
    COLUMNS, because those run through the PSF wings rather than sky, and their unclipped
    medians are 240-470 MJy/sr.  Subtracting them took the stellar peak from 627 to 256.
    """
    from klip_tpe.backends.spaceklip import destripe_detector

    ny, nx, cx, cy = 224, 288, 144.0, 112.0
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:ny, 0:nx]
    base = (rng.normal(0.0, 3.0, (ny, nx))
            + 600.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0 ** 2))
            + 40.0 * np.exp(-((yy - cy) ** 2) / (2 * 2.5 ** 2)))          # glow stick
    rows = rng.normal(0.0, 3.0, (ny, 1))
    clean, _ = destripe_detector(base[None], (cx, cy))
    striped, _ = destripe_detector((base + rows)[None], (cx, cy))

    sky = ~((np.hypot(xx - cx, yy - cy) < 55) | (np.abs(xx - cx) < 14))
    res, idx = [], []
    for j in range(ny):
        m = sky[j]
        if m.sum() > 30:
            res.append(float(np.median(striped[0][j][m] - clean[0][j][m])))
            idx.append(j)
    res, idx = np.array(res), np.array(idx)
    inj = float(np.std(rows))
    assert res.size > 150

    # overall: the offsets are gone to well within a third of what went in
    assert np.std(res) < inj / 3.0, f"residual sd {np.std(res):.3f} against injected {inj:.3f}"

    # and row by row, away from the glow stick, essentially nothing survives.  This is the
    # assertion that fails if the star disc and the boundary band are ever again allowed to
    # empty a line between them: those 24 rows sat at |y - cy| up to 12, i.e. right here.
    off = np.abs(idx - cy) > 8
    worst = float(np.max(np.abs(res[off] - np.median(res[off]))))
    assert worst < 0.75 * inj, (
        f"a row with sky in it kept {worst:.2f} of a {inj:.2f} offset -- a line the mask "
        f"emptied and the fallback did not reach")
    # the glow-stick rows are the known exception, and there are only a few of them
    assert int((~off).sum()) < 25, "the untreated band must stay narrow"


# ----------------------------------------------------------------------------
# which stars the RDI library may be built from
# ----------------------------------------------------------------------------
def test_ref_targets_keeps_a_disk_bearing_neighbour_out_of_the_library(tmp_path, capsys):
    """HIP 65426's MIRI directory also holds HD 141569A and HD 141569A's own reference
    HD 140986, and "science, and everything else is the library" put both in.  So F1140C's
    122-frame library was 90 frames of phi Cen, 20 of HD 140986 and 12 of HD 141569A -- a
    RESOLVED DISK in the KL basis of a bare star.  The numbers never said so; the loader
    reported "16 reference files" and nothing about whose."""
    d = tmp_path / "mixed"
    d.mkdir()
    for i, (t, roll) in enumerate([("HIP-65426", 103.2), ("HIP-65426", 112.6),
                                   ("HIP-68245", 104.4), ("HIP-68245", 104.4),
                                   ("HD-140986", 106.9),
                                   ("HD-141569A", 102.9)]):
        write(str(d / f"jw{i:05d}_calints.fits"), t, roll)
    files = sorted(str(p) for p in d.glob("*.fits"))

    msgs = []
    dsets, info = load_calints(files, science_target="HIP65426", half_px=20, align=False,
                               log=msgs.append)
    assert info["n_ref"] == 8, "default: every non-background non-science pointing (4 files)"
    assert any("RDI library from" in m and "HD141569A" in m.replace("-", "") for m in msgs), \
        "the composition must be logged by name -- that is the line whose absence hid the disk"

    msgs = []
    dsets, info = load_calints(files, science_target="HIP65426", half_px=20, align=False,
                               ref_targets=["HIP-68245"], log=msgs.append)
    assert info["n_ref"] == 4, f"phi Cen only: 2 files x 2 ints, got {info['n_ref']}"
    assert any("restricted to HIP68245" in m for m in msgs), msgs
    assert any("HD141569A x1" in m.replace("-", "") for m in msgs), \
        "it must say what it dropped, not just what it kept"


def test_ref_targets_that_matches_nothing_is_an_error_not_an_empty_library(tmp_path):
    """Silently reducing with no references because of a typo in a star name is exactly the
    kind of quiet wrong answer this package keeps finding."""
    d = tmp_path / "t"
    d.mkdir()
    write(str(d / "a_calints.fits"), "HIP-65426", 103.2)
    write(str(d / "b_calints.fits"), "HIP-68245", 104.4)
    files = sorted(str(p) for p in d.glob("*.fits"))
    with pytest.raises(ValueError, match="matched none"):
        load_calints(files, science_target="HIP65426", half_px=20, align=False,
                     ref_targets=["PHI-CEN"], log=lambda s: None)
