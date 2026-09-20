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


def test_default_separations_reach_the_edge_of_the_field():
    """A grid that stops short is worse than a coarse one: the interpolator clamps, so a
    map sampled to 3 arcsec reports the 3 arcsec throughput across a 24 arcsec field."""
    for f in ("F1065C", "F1140C", "F1550C"):
        s = miri.default_separations(f)
        assert s[0] <= 0.35 and s[-1] >= 10.0
        assert np.all(np.diff(s) > 0)
        # geometric: fine where the throughput climbs, coarse on the plateau
        assert np.diff(s)[0] < 0.2 and np.diff(s)[-1] > 1.5
    assert miri.default_separations("F2300C")[-1] > miri.default_separations("F1065C")[-1]


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


def test_the_injector_attenuates_each_frame_by_its_own_throughput():
    """The mask is fixed to the detector and the field rotates past it.

    A companion at one sky position angle therefore crosses the quadrant boundaries as the
    telescope rolls, so its attenuation differs frame to frame.  Hoisting the throughput
    out of the frame loop -- which is what the injector did before the azimuth existed --
    injects a source of constant brightness through a mask that is not, and the search
    then measures a throughput correction that no real companion experiences.
    """
    from klip_tpe.injection import Source, inject_sources, takes_azimuth
    g = synthetic_map(width=25.0)                       # wide boundary: rolls stay inside it
    sl = np.zeros((3, 11, 11))
    sl[:, 5, 5] = 1.0
    m = miri.MIRILibraryPSF(sl, [0.5, 1.0, 1.5], center=(5.0, 5.0), ee_radius_px=3.0,
                            flux_unit=1.0, thru2d=miri.throughput_map_fn(g))
    assert takes_azimuth(m), "the injector will not pass an azimuth to this model"
    cube = np.zeros((3, 61, 61), np.float32)
    # three rolls 20 degrees apart: the source sweeps from a boundary into a quadrant
    angles = np.array([0.0, 20.0, 40.0])
    out = inject_sources(cube, angles, [Source(rho=1.0, theta=270.0, contrast=1.0)], m,
                         pxscale=0.109655, truenorth=0.0)
    # the SUM, not the peak: add_stamp places the stamp with a bilinear sub-pixel shift,
    # which spreads a delta differently at every azimuth, so the peak is not proportional
    # to the injected amplitude and the sum is
    flux = np.array([float(f.sum()) for f in out])
    assert np.all(flux > 0), "nothing was injected"
    assert flux.max() / flux.min() > 1.5, (
        f"every frame got the same amplitude {flux} -- the azimuth is not reaching the "
        f"throughput, so the 2-D map is decorative")
    # and the frame-by-frame amplitudes are the map's own values at those azimuths
    az = 270.0 - 0.0 - 270.0 - angles
    np.testing.assert_allclose(flux / flux[0],
                               [m.throughput(1.0, a) / m.throughput(1.0, az[0]) for a in az],
                               rtol=2e-3)


def test_a_round_occulter_still_gets_one_throughput_for_the_sequence():
    """The per-frame path must not change what a radial model does."""
    from klip_tpe.injection import LibraryPSF, Source, inject_sources, takes_azimuth
    sl = np.zeros((2, 11, 11))
    sl[:, 5, 5] = 1.0
    m = LibraryPSF(sl, [0.5, 1.5], center=(5.0, 5.0), ee_radius_px=3.0,
                   throughput_fn=lambda r: 0.5, flux_unit=1.0)
    assert not takes_azimuth(m)
    out = inject_sources(np.zeros((3, 61, 61), np.float32), np.array([0.0, 20.0, 40.0]),
                         [Source(rho=1.0, theta=270.0, contrast=1.0)], m, pxscale=0.109655)
    flux = np.array([float(f.sum()) for f in out])
    np.testing.assert_allclose(flux, flux[0], rtol=1e-6)
    # and a MIRI model built without a map is a radial model, so it takes the same path
    assert not takes_azimuth(miri.MIRILibraryPSF(sl, [0.5, 1.5], center=(5.0, 5.0),
                                                 ee_radius_px=3.0))


def test_typical_throughput_does_not_warn():
    """The float32 headroom check runs on every injection and needs a scale, not a value.
    Making it warn each time would train the warning out of anyone's attention."""
    import warnings
    g = synthetic_map()
    sl = np.zeros((2, 9, 9))
    sl[:, 4, 4] = 1.0
    m = miri.MIRILibraryPSF(sl, [0.5, 1.5], center=(4.0, 4.0), ee_radius_px=3.0,
                            thru2d=miri.throughput_map_fn(g))
    miri.MIRILibraryPSF._warned = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = m.typical_throughput(1.0)
    assert 0.15 <= t <= 0.95
    assert not miri.MIRILibraryPSF._warned


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


def test_forbidden_pa_marks_the_dead_zones_in_sky_angle():
    """The dead zones are fixed to the detector; the sampler works in sky PA.

    The two are related only through each frame's roll, so this has to be computed over
    the whole sequence rather than for one angle.
    """
    g = synthetic_map()
    out = miri.forbidden_pa([0.0], rho_as=1.0, filter="F1065C", g=g, log=lambda *_: None)
    got = sorted(round(c) % 360 for c, _ in out)
    assert got == [0, 90, 180, 270], got                 # boundaries, in sky PA at roll 0
    assert all(w > 0 for _, w in out)
    # a roll shifts them: at parang = 30 the same detector boundaries sit 30 deg round
    out30 = miri.forbidden_pa([30.0], rho_as=1.0, filter="F1065C", g=g, log=lambda *_: None)
    assert sorted(round(c) % 360 for c, _ in out30) == [30, 120, 210, 300]
    # nothing is forbidden where the mask transmits
    assert not any(abs(((c - 45 + 180) % 360) - 180) < w for c, w in out)


def test_forbidden_pa_does_not_hinge_on_a_floating_point_tie_for_two_rolls():
    """Two rolls is the usual JWST sequence, and a PA a boundary eats in one of them is
    dead in exactly half the frames.  A ``> 0.5`` test forbids nothing there, on a tie."""
    g = synthetic_map()
    two = miri.forbidden_pa([0.0, 40.0], rho_as=1.0, filter="F1065C", g=g,
                            log=lambda *_: None)
    assert len(two) == 8, "both rolls' boundaries should be forbidden, not neither"
    assert sorted(round(c) % 360 for c, _ in two) == [0, 40, 90, 130, 180, 220, 270, 310]
    # and the strict-majority rule really would have dropped them
    assert miri.forbidden_pa([0.0, 40.0], rho_as=1.0, filter="F1065C", g=g,
                             dead_frac=0.5, log=lambda *_: None) == []


def test_forbidden_pa_wraps_a_sector_across_zero():
    """A sector centred on PA 0 spans 359-1 and must come back as one sector, not two."""
    g = synthetic_map(width=20.0)
    out = miri.forbidden_pa([0.0], rho_as=1.0, filter="F1065C", g=g, log=lambda *_: None)
    assert len(out) == 4, [round(c) for c, _ in out]
    c0 = [c for c, _ in out if min(c, 360 - c) < 5]
    assert len(c0) == 1 and min(c0[0], 360 - c0[0]) < 2


def test_forbidden_pa_with_no_frames_is_empty_not_everything():
    g = synthetic_map()
    assert miri.forbidden_pa([], rho_as=1.0, filter="F1065C", g=g, log=lambda *_: None) == []


def test_apply_quadrant_mask_leaves_non_miri_datasets_alone():
    """Safe on a mixed or NIRCam set: a filter that is not a MIRI coronagraphic one is
    passed through untouched rather than masked with somebody else's geometry."""
    from klip_tpe.reducer import Dataset
    g = synthetic_map()
    cube = np.ones((2, 81, 81), np.float32)
    ang = np.array([0.0, 10.0])
    mk = lambda f: Dataset(cube.copy(), ang, None, name="x",
                           meta={"filter": f, "pxscale": 0.109655, "center": (40.0, 40.0)})
    import klip_tpe.instruments.miri as M
    real, M.throughput_map = M.throughput_map, lambda *a, **k: g
    try:
        out = miri.apply_quadrant_mask({"miri": mk("F1065C"), "nircam": mk("F444W"),
                                        "blank": mk(None)}, log=lambda *_: None)
    finally:
        M.throughput_map = real
    assert np.isnan(np.asarray(out["miri"].cube)).any()
    assert out["miri"].meta["quadrant_masked_px"] > 0
    assert np.isfinite(np.asarray(out["nircam"].cube)).all()
    assert "quadrant_mask" not in (out["nircam"].meta or {})
    assert np.isfinite(np.asarray(out["blank"].cube)).all()


def test_model_for_datasets_routes_miri_to_the_two_dimensional_model():
    """`psf='stpsf'` must not hand MIRI a radial throughput.

    This is the path every driver takes, so if it returns a LibraryPSF the 2-D map is
    never reached no matter how correct the map itself is.
    """
    from klip_tpe.reducer import Dataset
    import klip_tpe.stpsf_psf as S

    calls = {}

    def fake_library(filter, star_flux=1.0, seps_as=None, log=print, **kw):
        calls.update(filter=filter, star_flux=star_flux)
        sl = np.zeros((2, 9, 9))
        sl[:, 4, 4] = 1.0
        return miri.MIRILibraryPSF(sl, [0.5, 1.5], center=(4.0, 4.0), ee_radius_px=3.0,
                                   thru2d=miri.throughput_map_fn(synthetic_map()))

    import klip_tpe.instruments.miri as M
    real, M.library = M.library, fake_library
    try:
        ds = Dataset(np.zeros((2, 61, 61), np.float32), np.array([0.0, 10.0]), None, name="m",
                     meta={"pxscale": 0.109655,
                           "header": {"INSTRUME": "MIRI", "FILTER": "F1065C",
                                      "CORONMSK": "4QPM_1065"}})
        m = S.model_for_datasets([ds], star_flux=7.0, log=lambda *_: None)
    finally:
        M.library = real
    assert calls == {"filter": "F1065C", "star_flux": 7.0}
    assert isinstance(m, miri.MIRILibraryPSF) and m.azimuth_dependent


def test_model_for_datasets_leaves_nircam_on_the_radial_path():
    """The MIRI branch must key off the instrument, not fire on anything with a mask."""
    from klip_tpe.reducer import Dataset
    import klip_tpe.stpsf_psf as S
    seen = {}

    def fake_grid(**kw):
        seen.update(kw)
        return {"slices": np.zeros((2, 9, 9)), "seps": np.array([0.5, 1.5]),
                "transmission": np.array([0.5, 0.9]), "center": (4.0, 4.0),
                "ee_radius_px": 3.0}

    real, S.offaxis_grid = S.offaxis_grid, fake_grid
    try:
        ds = Dataset(np.zeros((2, 61, 61), np.float32), np.array([0.0, 10.0]), None, name="n",
                     meta={"pxscale": 0.063,
                           "header": {"INSTRUME": "NIRCAM", "FILTER": "F444W",
                                      "CORONMSK": "MASKA335R"}})
        m = S.model_for_datasets([ds], seps_as=[0.5, 1.5], log=lambda *_: None)
    finally:
        S.offaxis_grid = real
    assert seen.get("instrument") == "NIRCam"
    assert not isinstance(m, miri.MIRILibraryPSF)
    assert not getattr(m, "azimuth_dependent", False)


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
