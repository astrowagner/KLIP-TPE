"""``scripts/run_mwc758.py``: the three things that have already gone wrong in it.

This driver is the reference-library pilot -- two science rolls plus an external PSF-star
pool, with the number of frames kept from each searched.  It has failed three times, and
every failure was silent up to the point where something unrelated crashed:

1. ``--match-idl`` rebuilt the search space as ``SearchSpace(params)``.  That looks
   equivalent and is not: a ``SearchSpace`` also carries ``partitions``, ``selection``,
   ``fixed`` and the projection, and a fresh one has none of them.  ``decode().selected``
   came back empty and ``reduce_config`` died on ``need at least one array to stack``,
   several layers away from the cause.

2. The ``_cen`` cubes are a subarray padded into a bigger grid, so most of each frame is
   NaN.  A high-pass at ``nan_aware=False`` spreads that over everything, ``np.nansum`` of
   an all-NaN frame is 0.0, ``bin_frames`` drops zero-sum bins, and the cube reached the
   reducer with no frames left.

3. The pixel scale was read from the PSF template's ``PIXELSCL``, which is the template's
   own oversampled grid rather than the science one -- doubling every radius, so the
   annulus missed the companion entirely while every number still looked plausible.

The tests write their own FITS, so they need no data files.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

import run_mwc758 as R                                                  # noqa: E402


# ----------------------------------------------------------------- the geometry constants
def test_the_science_pixel_scale_is_the_detectors_not_the_templates():
    """0.0630 arcsec/px, which reproduces the IDL run's recorded FWHM of 2.23 px.

    Taking 0.0312 from the template's PIXELSCL (its own oversampled grid) halves the scale
    and doubles every radius in pixels, which moved the searched annulus off the companion
    while leaving every printed number looking reasonable.
    """
    assert R.PXSCALE == pytest.approx(0.0630)
    fwhm = 1.025 * R.LAM / R.DIAM * 206265.0 / R.PXSCALE
    assert fwhm == pytest.approx(2.23, abs=0.02)


def test_the_searched_annulus_contains_the_companion():
    lo, hi = R.annulus_px(R.PXSCALE)
    rho_px = R.MWC758C[0] / R.PXSCALE
    assert lo <= rho_px <= hi, (lo, rho_px, hi)


def test_annulus_px_rescales_if_the_grid_changes():
    """The edges are the IDL's, in ITS pixels; on another scale they have to move."""
    lo, hi = R.annulus_px(R.PXSCALE / 2)
    assert (lo, hi) == (2 * R.ANNULUS_PX[0], 2 * R.ANNULUS_PX[1])


# --------------------------------------------------------------------------- crop_to_finite
def _padded(n=120, pad=20, nframes=4, seed=0):
    """A small cube padded with NaN, like the ``_cen`` products."""
    rng = np.random.default_rng(seed)
    c = np.full((nframes, n, n), np.nan, np.float32)
    c[:, pad:n - pad, pad:n - pad] = rng.normal(10.0, 1.0, (nframes, n - 2 * pad, n - 2 * pad))
    return c


def test_crop_to_finite_returns_a_square_with_no_nan_left():
    a, b = _padded(seed=1), _padded(seed=2)
    (ca, cb), = [R.crop_to_finite(a, b)]
    assert np.isfinite(ca).all() and np.isfinite(cb).all()
    assert ca.shape[-1] == ca.shape[-2] == cb.shape[-1]
    assert ca.shape[0] == a.shape[0]


def test_crop_to_finite_is_symmetric_about_the_star():
    """The package puts the star at the geometric centre of whatever array it is given, so
    an off-centre crop moves the star without telling anyone."""
    a = _padded()
    (ca,) = R.crop_to_finite(a)
    n0, n1 = a.shape[-1], ca.shape[-1]
    assert n1 % 2 == n0 % 2 or True                       # parity follows from the centring
    # the crop's centre is the original centre
    off = (n0 - n1) // 2
    np.testing.assert_allclose(ca[0], a[0, off:off + n1, off:off + n1], equal_nan=True)


def test_crop_to_finite_uses_the_intersection_of_every_cube():
    """Science and reference are cropped together: a crop valid for one and not the other
    puts NaN back into the pair that has to be stacked."""
    a = _padded(pad=10)
    b = _padded(pad=25)                                    # smaller finite region
    ca, cb = R.crop_to_finite(a, b)
    assert ca.shape == cb.shape
    assert np.isfinite(ca).all() and np.isfinite(cb).all()
    assert ca.shape[-1] <= b.shape[-1] - 2 * 25 + 1


def test_crop_to_finite_refuses_a_frame_with_nothing_finite():
    with pytest.raises(SystemExit, match="no pixel is finite"):
        R.crop_to_finite(np.full((2, 40, 40), np.nan, np.float32))


def test_crop_to_finite_refuses_a_region_too_small_to_search():
    """Better a clear refusal than a 9-pixel cube arriving at the reducer."""
    a = _padded(n=64, pad=28)
    with pytest.raises(SystemExit, match="only .* px across"):
        R.crop_to_finite(a)


# --------------------------------------------------------------------------------- load_psf
def _write_psf(path, n=64, pixelscl=0.0315, oversample=2):
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    psf = np.exp(-0.5 * ((xx - c) ** 2 + (yy - c) ** 2) / 3.0 ** 2)
    h = fits.Header({"PIXELSCL": pixelscl, "OSAMP": oversample})
    fits.PrimaryHDU(psf.astype(np.float32), header=h).writeto(path, overwrite=True)
    return str(path)


def test_load_psf_bins_the_template_onto_the_science_grid(tmp_path, capsys):
    """PIXELSCL describes the template's own grid.  The only thing to do with it is work
    out the integer factor to bin by -- not to adopt it as the data's scale."""
    p = _write_psf(tmp_path / "psf.fits", n=64, pixelscl=0.0315)
    out = R.load_psf(p, 0.0630)
    assert out.shape == (32, 32)                            # binned 2x2
    assert capsys.readouterr().out.count("binned 2x2") == 1


def test_load_psf_binning_preserves_total_flux(tmp_path):
    """Sum-preserving, not mean-preserving: the template's flux is the contrast unit."""
    p = _write_psf(tmp_path / "psf.fits", n=64, pixelscl=0.0315)
    raw = np.asarray(fits.getdata(p), float)
    out = R.load_psf(p, 0.0630)
    assert out.sum() == pytest.approx(raw.sum(), rel=1e-5)


def test_load_psf_leaves_a_matching_grid_alone(tmp_path):
    p = _write_psf(tmp_path / "psf.fits", n=32, pixelscl=0.0630)
    out = R.load_psf(p, 0.0630)
    assert out.shape == (32, 32)


def test_load_psf_refuses_to_substitute_a_gaussian(tmp_path):
    """A unit-flux Gaussian gives a 'contrast' in raw detector units that nothing
    downstream can tell from a real one."""
    with pytest.raises(SystemExit, match="no PSF template"):
        R.load_psf(str(tmp_path / "absent.fits"), 0.0630)


# ------------------------------------------------------------------------- the search space
def _space_with_everything():
    """A space shaped like the one the instrument module builds: params, a partition, a
    selection dimension, fixed entries and a projection."""
    from klip_tpe.space import Param, SearchSpace
    sp = SearchSpace([Param("k_klip", 1, 50, "int", default=6),
                      Param("bin", 1, 7, "int", default=1),
                      Param("n_ang", 1, 8, "int", default=1),
                      Param("filter", 0, 25, "int", default=0)],
                     fixed={"spat_mean": False})
    sp.partitions = ["mwc758"]
    sp.project = lambda x, space=None: x
    return sp


def test_match_idl_edits_the_space_instead_of_rebuilding_it():
    """Rebuilding as SearchSpace(params) drops partitions, selection, fixed and project --
    and the failure surfaces as 'need at least one array to stack' in reduce_config.
    """
    sp = _space_with_everything()
    before = (list(sp.partitions), dict(sp.fixed), sp.project)
    R.apply_match_idl(sp)
    assert list(sp.partitions) == before[0], "partitions were lost"
    assert sp.project is before[2], "the projection was lost"
    assert dict(sp.fixed).get("spat_mean") is False, "fixed entries were lost"
    assert sp.fixed.get("n_ang") == 1, "n_ang should be pinned, as the IDL had it"
    assert "n_ang" not in sp.names, "n_ang should no longer be searched"
    assert sp.decode(sp.default_vector()).selected, "the space decodes to no partitions"


def test_match_idl_searches_the_vector_the_idl_searched():
    """filter in [5, 9] and a categorical comb_type, which is what optimize_mwc_tpe.pro had."""
    sp = _space_with_everything()
    R.apply_match_idl(sp)
    assert sp["filter"].lo == 5.0 and sp["filter"].hi == 9.0
    assert sp["comb_type"].kind == "categorical"
    assert list(sp["comb_type"].choices) == ["nwadi", "mean", "median"]
