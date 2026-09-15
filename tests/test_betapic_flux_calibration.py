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
