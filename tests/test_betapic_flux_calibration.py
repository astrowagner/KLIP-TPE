"""The beta Pic contrast axis, and the two traps that made it wrong.

The NACO L' set is the one tutorial data set whose star flux has to be *imported*: the
science frames are AGPM coronagraphic and the distributed template is normalised, so
neither file says how bright beta Pictoris is.  These tests pin the import, the exact
aperture arithmetic that converts it, and the float32 floor that silently eats an injection
whose ``flux_unit`` is a normalised template's own sum.

The end-to-end check -- beta Pic b at dL' 7.81 against Absil et al. (2013)'s 8.01 +/- 0.16
from the same data -- lives in ``scripts/check_betapic_contrast.py``; it needs a download
and a few minutes of KLIP, so it is a script rather than a test.  What is asserted here is
everything that feeds it.
"""
import os
import warnings

import numpy as np
import pytest

from klip_tpe import datasets
from klip_tpe.injection import GaussianPSF, TemplatePSF, inject_sources
from klip_tpe.instruments import generic
from klip_tpe.metrics import Source

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ exact aperture photometry

def test_aperture_sum_is_exact_on_a_flat_field():
    """Sub-pixel *sampling* converges as 1/nsub and is still 1e-3 at 40 samples per pixel,
    which lands straight on the contrast axis.  The partial-pixel areas are exact."""
    ones = np.ones((41, 41))
    for r in (0.3, 1.0, 2.0, 2.4005, 5.0, 12.0):
        assert generic.aperture_sum(ones, 20.0, 20.0, r) == pytest.approx(np.pi * r * r, rel=1e-12)
    # off-centre, non-integer centre, and a radius that clips no edge
    assert generic.aperture_sum(ones, 17.3, 22.9, 3.7) == pytest.approx(np.pi * 3.7 ** 2, rel=1e-12)


def test_aperture_sum_handles_edges_and_degenerate_radii():
    ones = np.ones((11, 11))
    assert generic.aperture_sum(ones, 5.0, 5.0, 0.0) == 0.0
    assert generic.aperture_sum(ones, 5.0, 5.0, -1.0) == 0.0
    assert generic.aperture_sum(ones, -50.0, 5.0, 2.0) == 0.0          # entirely off the array
    # centred on pixel 0, whose left half lies outside the array: the half-disc x >= 0 plus
    # the strip -0.5 <= x <= 0, i.e. pi r^2 / 2 + 2 * int_0^0.5 sqrt(r^2 - x^2) dx
    r = 3.0
    strip = 0.5 * 0.5 * np.sqrt(r * r - 0.25) + 0.5 * r * r * np.arcsin(0.5 / r)
    assert generic.aperture_sum(ones, 0.0, 5.0, r) == pytest.approx(np.pi * r * r / 2 + 2 * strip,
                                                                    rel=1e-12)
    assert np.isfinite(generic.aperture_sum(np.where(np.eye(11) > 0, np.nan, 1.0), 5.0, 5.0, 2.0))


def test_a_whole_pixel_mask_is_not_good_enough():
    """The reason aperture_sum exists: the boolean mask published photometry would be
    compared against differs from the true aperture by several percent at these radii."""
    ones = np.ones((41, 41))
    yy, xx = np.mgrid[0:41, 0:41]
    for r in (2.0, 2.4005):
        mask = float(((np.hypot(xx - 20.0, yy - 20.0) <= r)).sum())
        exact = generic.aperture_sum(ones, 20.0, 20.0, r)
        assert abs(mask / exact - 1) > 0.02, f"r={r}: the shortcut happens to be right here"


# ------------------------------------------------------------------ the conversion

def test_star_flux_from_aperture_photometry_rescales_between_apertures():
    """starphot is quoted in one aperture; TemplatePSF normalises in another.  The helper's
    whole job is that ratio, and it must agree with what TemplatePSF actually does."""
    g = np.exp(-((np.mgrid[0:41, 0:41][1] - 20.0) ** 2
                 + (np.mgrid[0:41, 0:41][0] - 20.0) ** 2) / (2 * 2.0 ** 2))
    a = generic.aperture_sum(g, 20.0, 20.0, 3.0)
    sf = generic.star_flux_from_aperture_photometry(g, 1000.0, 3.0)
    assert sf == pytest.approx(1000.0 * g.sum() / a, rel=1e-12)

    # with ee_radius_px the target normalisation changes and so must the answer
    sf5 = generic.star_flux_from_aperture_photometry(g, 1000.0, 3.0, ee_radius_px=5.0)
    assert sf5 == pytest.approx(1000.0 * generic.aperture_sum(g, 20.0, 20.0, 5.0) / a, rel=1e-12)
    assert sf5 < sf

    # and the pairing is the point: a source injected at contrast 1 carries starphot's
    # worth of flux through the aperture starphot was measured in
    m = TemplatePSF(g, star_flux=sf)
    stamp, _, _ = m.stamp(0.5)
    scaled = stamp * m.flux_unit
    assert generic.aperture_sum(scaled, 20.0, 20.0, 3.0) == pytest.approx(1000.0, rel=1e-9)


def test_star_flux_from_aperture_photometry_refuses_an_empty_aperture():
    z = np.zeros((21, 21))
    z[0, 0] = 1.0
    with pytest.raises(ValueError, match="no flux inside"):
        generic.star_flux_from_aperture_photometry(z, 1000.0, 2.0)


# ------------------------------------------------------------------ the beta Pic numbers

@pytest.mark.skipif(not os.path.exists(os.path.join(datasets.data_dir(), "naco_betapic_psf.fits")),
                    reason="needs the cached VIP beta Pic template")
def test_the_betapic_template_is_normalised_and_the_photometry_undoes_it():
    """This is the fact the whole entry rests on: naco_betapic_psf.fits carries unit flux
    inside r = 2 px, so its own counts are not beta Pictoris, and VIP's starphot is quoted
    in that same aperture.  If the distributed file is ever replaced, this fails first."""
    from astropy.io import fits
    t = fits.getdata(datasets.fetch("naco_betapic", quiet=True)["psf"]).astype(float)
    c = (t.shape[0] - 1) / 2.0
    assert generic.aperture_sum(t, c, c, 2.0) == pytest.approx(1.0, abs=1e-7)
    assert t.sum() == pytest.approx(4.3491, rel=1e-4)

    p = datasets.PHOTOMETRY["naco_betapic"]
    assert p["aperture_px"] == 2.0
    sf = generic.star_flux_from_aperture_photometry(t, p["starphot"], p["aperture_px"])
    assert sf == pytest.approx(3.3268e6, rel=1e-3)
    # the size of the mistake this replaces: star_flux unset would use the stamp's own sum
    assert sf / t.sum() == pytest.approx(7.65e5, rel=1e-2)


def test_the_photometry_entry_records_its_provenance_and_its_check():
    p = datasets.PHOTOMETRY["naco_betapic"]
    assert "VIP" in p["ref"] and "DIT" in p["ref"]
    assert "Absil" in p["check"] and "7.79" in p["check"]
    # the sets calibrated some other way must NOT acquire a starphot by copy-paste
    assert "sphere_hd95086" not in datasets.PHOTOMETRY
    assert "nircam_pds70_f480m" not in datasets.PHOTOMETRY


def test_the_betapic_call_sites_pass_a_star_flux():
    """Leaving it unset is not a neutral default here -- it is an axis 9.4e5 off."""
    for rel in (("tutorials", "01_naco_betapic.py"), ("README.md",)):
        src = open(os.path.join(HERE, *rel)).read()
        assert "star_flux_from_aperture_photometry" in src, f"{rel[-1]} has no star flux"
        assert "star_flux unset" not in src
    pr = next((p for p in (os.path.join(HERE, "paper_runs", "run_demos.py"),
                           os.path.join(os.path.dirname(HERE), "paper_runs", "run_demos.py"))
               if os.path.exists(p)), None)
    if pr:
        src = open(pr).read()
        assert "star_flux_from_aperture_photometry" in src
        assert "sf = None" not in src


# ------------------------------------------------------------------ the float32 floor

def test_an_injection_below_the_float32_quantum_is_lost_and_warned_about():
    """``inject_sources`` holds the cube in float32.  A stamp whose pixels sit below the
    float32 ULP of the science pixels they land on is not faint -- it is gone, and the loss
    grows as the contrast falls, so recovered S/N stops tracking contrast.  This is what a
    normalised template's flux_unit (order 1, against science pixels of order 1e3) walks
    into, and it is silent without the warning."""
    rng = np.random.default_rng(0)
    cube = (1000.0 + rng.normal(0, 10, (8, 61, 61))).astype(np.float32)
    ang = np.linspace(0, 40, 8)
    model = GaussianPSF(3.5, star_flux=1.0)       # flux_unit 1, as a normalised template gives

    def deposited(c):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = inject_sources(cube, ang, [Source(0.3, 70.0, c)], model, 0.027)
        return float((out.astype(np.float64) - cube.astype(np.float64)).sum()) / cube.shape[0]

    assert deposited(1e-3) / (1e-3 * model.flux_unit) < 0.9      # a quarter or more is gone
    assert deposited(1e2) / (1e2 * model.flux_unit) > 0.99       # and fine when it has headroom

    with pytest.warns(RuntimeWarning, match="float32 rounding"):
        inject_sources(cube, ang, [Source(0.3, 70.0, 1e-3)], model, 0.027)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                            # no warning with real headroom
        inject_sources(cube, ang, [Source(0.3, 70.0, 1e2)], model, 0.027)


def test_the_float32_warning_names_the_flux_unit_as_the_likely_cause():
    rng = np.random.default_rng(1)
    cube = (1000.0 + rng.normal(0, 10, (4, 61, 61))).astype(np.float32)
    with pytest.warns(RuntimeWarning) as rec:
        inject_sources(cube, np.zeros(4), [Source(0.3, 10.0, 1e-4)],
                       GaussianPSF(3.5, star_flux=1.0), 0.027)
    msg = str(rec[0].message)
    assert "flux_unit" in msg and "star_flux" in msg
    assert "star_flux_from_aperture_photometry" in msg


# ------------------------------------------------------------------ JWST: the coronagraph terms

def test_the_jwst_photometry_entry_separates_the_four_terms():
    """The HIP 65426 flux unit is S x (1/1e6 PIXAR_SR) x EE x T_optics, and the occulter's
    T(rho) is deliberately NOT among them -- it multiplies flux_unit inside inject_sources.

    T_optics is 1.0 and NOT anchored on anything: PHOTMJSR for PUPIL=MASKRND is derived
    from standards observed through the coronagraphic optics, so the calints already put an
    off-mask source at its true flux.  The 0.561 that sat here until 2026-09-16 was 1/1.9
    -- the factor by which the loader's sigma-clip repair had median-filtered the companion
    (and not the fakes, injected after it).  If a value other than 1.0 ever comes back, it
    needs a source that is not the companion, or the Carter comparison stops being a check."""
    p = datasets.PHOTOMETRY["hip65426_f444w"]
    assert p["filter"] == "F444W"
    assert p["flux_density_jy"] == pytest.approx(0.4026, rel=1e-3)
    assert p["optics_transmission"] == 1.0, \
        "0.561 was the sigma-clip repair's damage to the companion, not an optics number"
    assert "anchor" not in p["optics_transmission_source"].lower(), \
        "the Carter comparison is an independent check only while nothing is anchored on b"
    assert "PHOTMJSR" in p["optics_transmission_source"]
    assert "Carter" in p["check"] and "8.703" in p["check"]


def test_star_flux_from_flux_density_applies_each_term_once():
    """Arithmetic only -- no STPSF -- so it runs anywhere."""
    from klip_tpe import stpsf_psf
    calls = {}

    def fake_ee(radius_px, **kw):
        calls["radius"] = radius_px
        calls["pupil"] = kw.get("pupil_mask")
        return 0.25

    g = {"ee_radius_px": 8.0, "meta": {"instrument": "NIRCam", "filter": "F444W",
                                       "pupil_mask": "MASKRND"}}
    real, stpsf_psf.unocculted_ee = stpsf_psf.unocculted_ee, fake_ee
    try:
        sf = stpsf_psf.star_flux_from_flux_density(g, 1.0, 1e-13, optics_transmission=0.5,
                                                   log=lambda s: None)
    finally:
        stpsf_psf.unocculted_ee = real
    assert sf == pytest.approx(1.0 / (1e6 * 1e-13) * 0.25 * 0.5, rel=1e-12)
    assert calls["radius"] == 8.0
    assert calls["pupil"] == "MASKRND", "the EE must be the Lyot-stop PSF, not an imaging one"


def test_star_flux_from_flux_density_refuses_units_it_cannot_convert():
    from klip_tpe import stpsf_psf
    with pytest.raises(ValueError, match="MJy/sr"):
        stpsf_psf.star_flux_from_flux_density({"ee_radius_px": 8.0}, 1.0, 1e-13,
                                              bunit="DN/s", log=lambda s: None)


def test_load_calints_crops_odd_and_reports_the_centre_it_used():
    """An even crop with the star on an integer pixel leaves it half a pixel off in each
    axis -- 0.7 px radially, a 3 degree PA error at HIP 65426 b's separation."""
    import inspect

    from klip_tpe.backends import spaceklip as sk
    src = inspect.getsource(sk.load_calints)
    assert "n = 2 * H + 1" in src, "the crop has to be odd"
    assert "y0 + n" in src and "x0 + n" in src
    assert "star_center" in inspect.signature(sk.load_calints).parameters
    assert "APERTURE reference" in src, "CRPIX being the wrong centre must stay documented"


def test_every_frame_carries_its_own_integration_time():
    """One plane of a calints cube is one integration, and EFFINTTM is its exposure time --
    the JWST analogue of a DIT.  The provenance has to record it per frame rather than per
    exposure, because science (DEEP8, 307.884 s) and reference (MEDIUM8, 40.623 s) differ by
    7.6x and any per-frame weighting that assumed one number would be wrong for 18 of 22."""
    import inspect

    from klip_tpe.backends import spaceklip as sk
    src = inspect.getsource(sk.load_calints)
    for k in ("effinttm", "int_mid_mjd", "readpatt", "ngroups", "nframes", "groupgap", "tframe"):
        assert f'"{k}"' in src, f"per-frame provenance must record {k}"
    assert "INT_TIMES" in src, "the per-integration clock comes from INT_TIMES, not EXPSTART"
    # PRIMARY, not SCI: the same trap that made XOFFSET/YOFFSET silently zero
    assert "ph.get(\"EFFINTTM\"" in src


def test_the_readout_ladder_closes_for_both_hip65426_patterns():
    """EFFINTTM is the ramp span, (NGROUPS*NFRAMES + (NGROUPS-1)*GROUPGAP) * TFRAME -- NOT
    (NGROUPS-1)*TGROUP, which is off by 3% for DEEP8 and 21% for MEDIUM8.  Getting this wrong
    would misreport the exposure time in the paper's observation table."""
    tframe = 1.06904
    for readpatt, ngroups, nframes, groupgap, tgroup, effinttm, nints in (
            ("DEEP8", 15, 8, 12, 21.381, 307.884, 2),      # HIP 65426, each roll
            ("MEDIUM8", 4, 8, 2, 10.690, 40.623, 2)):      # HIP 68245, each dither
        assert (nframes + groupgap) * tframe == pytest.approx(tgroup, abs=5e-3), readpatt
        nfr = ngroups * nframes + (ngroups - 1) * groupgap
        assert nfr * tframe == pytest.approx(effinttm, abs=5e-3), readpatt
        # the formula that looks right and is not
        assert (ngroups - 1) * tgroup != pytest.approx(effinttm, rel=0.02), readpatt
    # 2 rolls x 2 integrations x 307.884 s
    assert 2 * 2 * 307.884 == pytest.approx(1231.5, abs=0.1)


def test_the_frame_count_matches_carter_2023_table_1():
    """Carter et al. 2023 (ApJL 951, L20) Table 1, MASK335R/F444W: HIP 65426 has N_ints=2
    over N_rolls=2, HIP 68245 N_ints=2 over N_dithers=9.  Four science integrations is the
    published design, not a truncated download -- and their t_exp is DURATION, not EFFEXPTM,
    which is the 2.2 s per exposure that makes 1235.892 s rather than 1231.5 s."""
    duration_sci, duration_ref = 617.946, 83.426      # Carter Table 1 t_exp
    effexptm_sci, effexptm_ref = 615.767, 81.247      # header, excludes the reset frames
    tframe, nints = 1.06904, 2

    assert 2 * duration_sci == pytest.approx(1235.892, abs=1e-3), "Carter t_total, 2 rolls"
    assert 9 * duration_ref == pytest.approx(750.835, abs=1e-2), "Carter t_total, 9 dithers"
    for dur, eff in ((duration_sci, effexptm_sci), (duration_ref, effexptm_ref)):
        # the gap is one reset frame per integration, give or take clock granularity
        assert (dur - eff) / (nints * tframe) == pytest.approx(1.0, abs=0.03)

    # 4 science, 18 reference -- what load_calints must produce for this programme
    assert 2 * 2 == 4 and 2 * 9 == 18


def test_the_jwst_call_sites_use_the_model_and_the_measured_centre():
    """Neither the runs nor the tutorial may fall back to a Gaussian of flux unit 1 or to
    CRPIX: those are the two things that made run D's axis 2.3e5 off and put the companion
    1.5 px inside its own separation."""
    p = datasets.PHOTOMETRY["hip65426_f444w"]
    cx, cy = p["star_center"]
    assert np.hypot(cx - 149.2, cy - 173.6) == pytest.approx(0.78, abs=0.05), \
        "the offset from CRPIX is the whole point of storing this (1.48 px was the solve on median-filtered frames)"

    tut = open(os.path.join(HERE, "tutorials", "03_jwst_nircam_hip65426.py")).read()
    assert "star_flux_from_flux_density" in tut and 'PHOT["star_center"]' in tut
    assert "psf_template=PSF_TEMPLATE" not in tut, "the old Gaussian fallback path is gone"

    pr = next((q for q in (os.path.join(HERE, "paper_runs", "run_demos.py"),
                           os.path.join(os.path.dirname(HERE), "paper_runs", "run_demos.py"))
               if os.path.exists(q)), None)
    if pr:
        src = open(pr).read()
        assert "star_flux_from_flux_density" in src
        assert "raise RuntimeError" in src and "GaussianPSF" in src, \
            "runs D/H2 must refuse rather than fall back silently"
        # the *argument*, not the comment that explains where 52.7 came from
        h2 = src[src.index('_bench_hi("H2"'):]
        assert "5.270e1" not in h2.split("\n")[0], \
            "H2's forced contrast must be on the absolute axis, not raw detector units"
        assert "2.324e-04" not in h2.split("\n")[0], \
            "2.324e-04 was measured on median-filtered frames (the pre-2026-09-16 repair)"
        assert "1.637e-04" in h2.split("\n")[0]
        # the hand-rolled loader is gone in favour of the shared one
        assert "load_calints" in src
