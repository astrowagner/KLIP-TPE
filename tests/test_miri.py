"""MIRI coronagraphy: the four-quadrant mask geometry and the 2-D throughput model.

None of these need STPSF.  The expensive part of :mod:`klip_tpe.instruments.miri` is
computing the throughput map, and everything downstream of it -- interpolation, the
azimuthal seam, the dead-zone mask, the injection model -- takes the map as data.  So the
tests build a synthetic map with a known four-quadrant structure and check the machinery
against it.  The one thing that genuinely requires STPSF, that the real map has minima on
the quadrant boundaries, is checked in ``test_the_real_map_is_a_four_quadrant_mask`` and
skipped when STPSF is not installed.
"""
import numpy as np
import pytest

from klip_tpe.instruments import miri
from klip_tpe.stpsf_psf import have_stpsf


def synthetic_map(boundary_deg=(0.0, 90.0, 180.0, 270.0), width=8.0, floor=0.15, top=0.95):
    """A map shaped like a 4QPM: ``top`` between the boundaries, ``floor`` on them.

    The azimuth grid is built around the boundaries, as the real one is -- that is the
    whole point of :func:`~klip_tpe.instruments.miri.locate_boundaries`.
    """
    seps = np.arange(0.4, 2.41, 0.4)
    az = miri.default_azimuths(boundary_deg)
    d = np.min(np.abs(((az[None, :] - np.asarray(boundary_deg)[:, None] + 180.0) % 360.0) - 180.0),
               axis=0)
    prof = floor + (top - floor) * np.clip(d / width, 0.0, 1.0)
    return {"seps": seps, "az": az, "trans": np.tile(prof, (seps.size, 1)),
            "pxscale": 0.109655, "fwhm_px": 3.1, "ee_radius_px": 4.0, "meta": {}}


def test_mode_for_filter_knows_the_four_coronagraphs():
    for f in ("F1065C", "F1140C", "F1550C"):
        m = miri.mode_for_filter(f)
        assert m["kind"] == "4qpm" and m["pupil_mask"] == "MASKFQPM"
    lyot = miri.mode_for_filter("F2300C")
    assert lyot["kind"] == "lyot" and lyot["spot_radius_as"] == 2.16
    assert miri.mode_for_filter("f1065c")["filter"] == "F1065C"      # case insensitive
    with pytest.raises(ValueError, match="not a MIRI coronagraphic filter"):
        miri.mode_for_filter("F444W")                                # that is NIRCam


def test_filter_wavelengths_are_the_published_ones():
    """Boccaletti et al. 2022 Table 1; a wrong lambda moves every annulus edge."""
    assert miri.MODES["F1065C"]["lam_m"] == pytest.approx(10.575e-6)
    assert miri.MODES["F1140C"]["lam_m"] == pytest.approx(11.30e-6)
    assert miri.MODES["F1550C"]["lam_m"] == pytest.approx(15.50e-6)
    assert miri.MODES["F2300C"]["lam_m"] == pytest.approx(22.75e-6)
    assert miri.DIAMETER_M == 6.5


def test_default_azimuths_are_dense_on_the_boundaries():
    """Sampled uniformly the dead zone is stepped over: at 30 degree spacing the mask
    came out 0.2% of the field, because the interpolant never saw the minimum."""
    az = miri.default_azimuths()
    assert az.size == 48 and az.min() >= 0 and az.max() < 360
    assert len(set(np.round(az, 6))) == az.size                      # no duplicates
    for b in (0.0, 90.0, 180.0, 270.0):
        d = np.abs(((az - b + 180.0) % 360.0) - 180.0)
        assert (d <= 3.0).any(), f"nothing within 3 deg of the boundary at {b}"
        assert (d <= 7.0).sum() >= 3, f"boundary at {b} sampled by fewer than 3 points"
    # and the gaps between boundaries are coarse, which is the point of the non-uniformity
    mid = az[(np.abs(az - 45.0) < 20)]
    assert np.diff(np.sort(mid)).max() >= 8.0


def test_default_azimuths_follow_the_boundaries_they_are_given():
    """Dense on the detector axes is dense 4.5 degrees away from the real boundaries."""
    rot = miri.default_azimuths((-4.5, 85.5, 175.5, 265.5))
    for b in (355.5, 85.5, 175.5, 265.5):
        d = np.abs(((rot - b + 180.0) % 360.0) - 180.0)
        assert d.min() < 1e-6, f"the grid does not sit on the boundary at {b}"
    on_axis = np.abs(((rot + 180.0) % 360.0) - 180.0)
    assert on_axis.min() > 1.0, "the grid is still pinned to the detector axis"


def test_throughput_interpolates_and_wraps_across_the_seam():
    g = synthetic_map()
    f = miri.throughput_map_fn(g)
    assert float(f(1.0, 45.0)) == pytest.approx(0.95, abs=0.02)       # between boundaries
    assert float(f(1.0, 0.0)) == pytest.approx(0.15, abs=0.02)        # on one
    # 359 and 1 straddle the seam and must both be near the floor, not clamped to an end
    assert float(f(1.0, 359.0)) < 0.35 and float(f(1.0, 1.0)) < 0.35
    assert float(f(1.0, 359.9)) == pytest.approx(float(f(1.0, -0.1)), abs=1e-6)
    # and the wrap is continuous: no jump across 360
    a = np.linspace(350.0, 370.0, 41)
    assert np.max(np.abs(np.diff(f(np.full(a.shape, 1.0), a)))) < 0.1


def test_throughput_is_clamped_outside_the_sampled_separations():
    g = synthetic_map()
    f = miri.throughput_map_fn(g)
    assert float(f(0.01, 45.0)) == pytest.approx(float(f(g["seps"][0], 45.0)))
    assert float(f(99.0, 45.0)) == pytest.approx(float(f(g["seps"][-1], 45.0)))


def test_quadrant_mask_is_a_cross_not_a_disc():
    """The whole point: a 4QPM's dead zone runs along two lines, not around a circle.

    A radial mask would either throw away every position angle at small separation or
    keep the boundary pixels at large separation; both are wrong, and the second is worse
    because the photometry there is unreliable rather than absent.
    """
    g = synthetic_map()
    bad = miri.quadrant_mask((161, 161), (80.0, 80.0), 0.109655, "F1065C", g=g,
                             min_throughput=0.30)
    yy, xx = np.mgrid[0:161, 0:161]
    dx, dy = xx - 80.0, yy - 80.0
    r = np.hypot(dx, dy) * 0.109655
    az = np.degrees(np.arctan2(dy, dx)) % 90.0
    ring = (r > 1.0) & (r < 2.0)
    on_axis = ring & ((az < 4) | (az > 86))
    between = ring & (np.abs(az - 45.0) < 20)
    assert bad[on_axis].mean() > 0.5, "the quadrant boundaries were not masked"
    assert bad[between].sum() == 0, "pixels between the boundaries were masked"
    assert bad[r < g["seps"][0]].all(), "inside the innermost sample nothing is known"


def test_quadrant_mask_follows_a_rotated_mask():
    """The real boundaries sit 4-5 degrees off the detector axes, so the mask must come
    from the measured map rather than from an assumed geometry."""
    g = synthetic_map(boundary_deg=(-4.5, 85.5, 175.5, 265.5))
    bad = miri.quadrant_mask((161, 161), (80.0, 80.0), 0.109655, "F1065C", g=g,
                             min_throughput=0.30)
    yy, xx = np.mgrid[0:161, 0:161]
    dx, dy = xx - 80.0, yy - 80.0
    r = np.hypot(dx, dy) * 0.109655
    az = np.degrees(np.arctan2(dy, dx))
    ring = (r > 1.5) & (r < 2.2)
    rotated = ring & (np.abs(((az + 4.5 + 180) % 360) - 180) < 3)
    on_detector_axis = ring & (np.abs(((az + 180) % 360) - 180) < 1.5)
    assert bad[rotated].mean() > 0.5, "the rotated boundary was not masked"
    assert bad[on_detector_axis].mean() < bad[rotated].mean()


def test_lyot_mode_masks_the_occulting_spot():
    g = synthetic_map()
    bad = miri.quadrant_mask((201, 201), (100.0, 100.0), 0.109655, "F2300C", g=g,
                             min_throughput=0.0, bar_width_px=6.0)
    yy, xx = np.mgrid[0:201, 0:201]
    r = np.hypot(xx - 100.0, yy - 100.0) * 0.109655
    assert bad[r < 2.1].all(), "the 2.16 arcsec Lyot spot is not masked"
    assert not bad[(r > 2.4) & (np.abs(xx - 100.0) > 4)].any()
    assert bad[(np.abs(xx - 100.0) <= 3) & (r > 2.4)].all(), "the support bar is not masked"


def test_library_psf_refuses_to_guess_an_azimuth_silently():
    """Called without an azimuth the model must say so: on a 4QPM the number it would
    otherwise return is wrong by up to a factor of three, in either direction."""
    g = synthetic_map()
    sl = np.zeros((3, 9, 9))
    sl[:, 4, 4] = 1.0
    m = miri.MIRILibraryPSF(sl, [0.5, 1.0, 1.5], center=(4.0, 4.0), ee_radius_px=3.0,
                            thru2d=miri.throughput_map_fn(g))
    assert m.throughput(1.0, 45.0) == pytest.approx(0.95, abs=0.02)
    assert m.throughput(1.0, 0.0) == pytest.approx(0.15, abs=0.02)
    miri.MIRILibraryPSF._warned = False
    with pytest.warns(RuntimeWarning, match="without a detector azimuth"):
        med = m.throughput(1.0)
    # the azimuthal median: above the boundary value, at most the plateau, and -- the
    # reason the warning exists -- nowhere near the boundary value it replaces
    assert m.throughput(1.0, 0.0) < med <= 0.95 + 1e-6
    assert med > 2 * m.throughput(1.0, 0.0)
    miri.MIRILibraryPSF._warned = False


def test_library_psf_without_a_map_behaves_like_the_base_class():
    sl = np.zeros((2, 9, 9))
    sl[:, 4, 4] = 1.0
    m = miri.MIRILibraryPSF(sl, [0.5, 1.0], center=(4.0, 4.0), ee_radius_px=3.0)
    assert m.throughput(0.7) == 1.0 and m.throughput(0.7, 12.0) == 1.0


def test_load_miri_masks_the_dead_zones_with_nan_not_zero():
    """Zero is a measurement of zero and pulls every statistic over the annulus towards
    it; the KLIP engine already treats NaN as missing."""
    rng = np.random.default_rng(0)
    cube = rng.normal(10.0, 1.0, (4, 81, 81)).astype(np.float32)
    ang = np.linspace(0, 12, 4)
    g = synthetic_map()
    import klip_tpe.instruments.miri as M
    real = M.throughput_map
    M.throughput_map = lambda *a, **k: g
    try:
        ds = miri.load_miri(cube, ang, filter="F1065C", tags=None, min_throughput=0.30)
    finally:
        M.throughput_map = real
    bad = ds.meta["quadrant_mask"]
    assert bad.sum() == ds.meta["quadrant_masked_px"] > 0
    assert np.isnan(np.asarray(ds.cube)[:, bad]).all()
    assert np.isfinite(np.asarray(ds.cube)[:, ~bad]).all()
    assert ds.meta["pxscale"] == pytest.approx(0.1097, abs=0.002)
    assert ds.meta["lam_m"] == pytest.approx(10.575e-6)
    assert ds.meta["image_mask"] == "FQPM1065"


def test_load_miri_needs_a_filter():
    cube = np.zeros((2, 21, 21), np.float32)
    with pytest.raises(ValueError, match="no MIRI filter"):
        miri.load_miri(cube, [0.0, 5.0], tags=None)


@pytest.mark.skipif(not have_stpsf(), reason="needs STPSF and its data files")
def test_the_real_map_is_a_four_quadrant_mask():
    """The one claim that cannot be made against a synthetic map: the instrument model
    really does suppress along two perpendicular lines, by a factor of a few."""
    g = miri.throughput_map("F1065C", seps_as=[1.0, 2.0],
                            az_deg=np.arange(0.0, 360.0, 30.0), log=lambda *_: None)
    t = g["trans"]
    assert t.max() <= 1.0 + 1e-6, "throughput above 1 means the reference is misplaced"
    assert t.max() / t.min() > 2.0, "no azimuthal structure: is the offset convention right?"
    # 180 degree symmetry is a property of the mask; 90 degree symmetry is not exact
    half = g["az"].size // 2
    np.testing.assert_allclose(t[:, :half], t[:, half:], atol=0.02)
