"""Keeping a known disk out of the injections and out of the noise estimate.

beta Pic's debris disk runs through the 8-22 px annulus the beta Pic runs search, and until
now nothing excluded it from either side: ``PositionSampler`` had a ``forbidden_pa`` field
that ``default_config`` never set, and the metric's exclusion was point-source only.

Measured on the default reduction of that cube, so the paper can be specific:

* the azimuthal excess across the annulus peaks at PA 28-38 and 208-222 -- the catalogued
  near-edge-on axis -- and the two bands carry ~+0.2 sigma of median excess over the rest of
  the ring, while covering ~22% of it;
* at the innermost injection radius (12.7 px) the disk supplies 26% of the ring scatter,
  but comparable excesses sit at PA 244, 354 and 92, which are speckles and stay;
* cutting it moves the measured S/N from 3.96 to 4.26 and costs two noise apertures per
  source (15 -> 13), well clear of the ``min_ring`` floor.

The cut is defensible because the disk's position angle is known before the image is looked
at.  Dropping ring apertures *because they are bright* would bias the noise estimate down and
inflate every S/N, which is why this is a named-band mechanism and not a sigma clip.
"""
from __future__ import annotations

import numpy as np
import pytest

from klip_tpe.metrics import MawetPeakSNR, mawet_peak_snr, pa_wedge_mask
from klip_tpe.positions import PositionSampler


# ------------------------------------------------------------------------ the wedge mask

def test_the_mask_covers_the_bands_and_nothing_else():
    m = pa_wedge_mask((101, 101), [(29.0, 20.0)])
    ny, nx = m.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    pa = np.degrees(np.arctan2(yy - cy, xx - cx)) - 90.0
    d = np.abs((pa - 29.0 + 180.0) % 360.0 - 180.0)
    assert np.array_equal(m, d <= 20.0)


def test_two_bands_cover_about_a_fifth_of_the_frame():
    m = pa_wedge_mask((101, 101), [(29.0, 20.0), (209.0, 20.0)])
    assert 0.20 < m.mean() < 0.28, m.mean()          # 2 x 40 deg of 360


def test_a_radial_range_limits_the_wedges():
    full = pa_wedge_mask((101, 101), [(29.0, 20.0)])
    band = pa_wedge_mask((101, 101), [(29.0, 20.0)], r_range=(8, 22))
    assert band.sum() < full.sum()
    ny, nx = band.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
    assert rr[band].min() >= 8.0 and rr[band].max() <= 22.0


def test_no_bands_masks_nothing():
    assert not pa_wedge_mask((51, 51), []).any()


# ------------------------------------------------- the noise ring actually honours the mask

def _planted(shape=(101, 101), bands=((29.0, 20.0),), amp=50.0, seed=0):
    """A frame whose only structure is a bright wedge -- the disk, idealised."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1.0, shape)
    return img + amp * pa_wedge_mask(shape, bands)


def test_masking_the_wedge_removes_it_from_the_noise():
    img = _planted()
    kw = dict(pxscale=0.0272, fwhm=3.6, known=(), declip_nsig=None, return_details=True)
    _, hot = mawet_peak_snr(img, [0.40], [120.0], **kw)          # source away from the wedge
    _, cold = mawet_peak_snr(img, [0.40], [120.0], pixel_mask=pa_wedge_mask(img.shape, [(29.0, 20.0)]), **kw)
    assert cold[0]["sigma"] < 0.5 * hot[0]["sigma"], (hot[0]["sigma"], cold[0]["sigma"])


def test_masking_costs_apertures_but_keeps_the_ring_usable():
    img = _planted()
    kw = dict(pxscale=0.0272, fwhm=3.6, known=(), declip_nsig=None, return_details=True)
    _, hot = mawet_peak_snr(img, [0.40], [120.0], **kw)
    _, cold = mawet_peak_snr(img, [0.40], [120.0],
                             pixel_mask=pa_wedge_mask(img.shape, [(29.0, 20.0), (209.0, 20.0)]), **kw)
    assert cold[0]["nclean"] < hot[0]["nclean"], "the masked apertures were still counted"
    assert cold[0]["nclean"] >= 6, "the ring must not be starved by the cut"


def test_the_setup_record_says_whether_a_mask_was_in_force():
    """A masked noise estimate is not comparable with an unmasked one."""
    m = MawetPeakSNR(pxscale=0.0272, fwhm=3.6)
    assert m.describe()["pixel_mask_px"] is None
    masked = MawetPeakSNR(pxscale=0.0272, fwhm=3.6,
                          pixel_mask=pa_wedge_mask((101, 101), [(29.0, 20.0)]))
    assert masked.describe()["pixel_mask_px"] == int(masked.pixel_mask.sum())


# ------------------------------------------------------------ injections avoid the bands

def _pas(sampler, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return [s.theta for s in sum((list(sampler.sample(3, 0.30, 0.50, rng, 1e-3))
                                  for _ in range(n // 3)), [])]


def _on_band(pas, bands):
    def d(a, c):
        return abs((a - c + 180.0) % 360.0 - 180.0)
    return sum(1 for p in pas if any(d(p, c) <= hw for c, hw in bands))


def test_forbidden_pa_keeps_injections_off_the_bands():
    bands = [(29.0, 20.0), (209.0, 20.0)]
    free = PositionSampler(fwhm_as=0.098)
    cut = PositionSampler(fwhm_as=0.098, forbidden_pa=bands)
    assert _on_band(_pas(free), bands) > 0, "the test needs the free sampler to hit them"
    assert _on_band(_pas(cut), bands) == 0


def test_default_config_passes_the_bands_to_the_sampler():
    import klip_tpe.instruments.generic as G

    class _R:
        fwhm, pxscale = 3.6, 0.0272

        def matched_filter_kernel(self, rho):
            return None

    bands = [(29.0, 20.0), (209.0, 20.0)]
    _, samp = G.default_config(_R(), known=[(0.452, 211.9)], forbidden_pa=bands)
    assert list(samp.forbidden_pa) == [tuple(b) for b in bands]
    assert _on_band(_pas(samp), bands) == 0
    _, plain = G.default_config(_R(), known=[(0.452, 211.9)])
    assert list(plain.forbidden_pa) == []          # off unless asked for


def test_default_config_forwards_the_mask_to_the_metric():
    """Both halves or neither: the sampler's bands and the metric's mask have to agree."""
    import klip_tpe.instruments.generic as G

    class _R:
        fwhm, pxscale = 3.6, 0.0272

        def matched_filter_kernel(self, rho):
            return None

    mask = pa_wedge_mask((101, 101), [(29.0, 20.0)])
    obj, _ = G.default_config(_R(), known=[], forbidden_pa=[(29.0, 20.0)], pixel_mask=mask)
    assert obj.metric.describe()["pixel_mask_px"] == int(mask.sum())


# --------------------------------------------------------------- wired into the beta Pic runs

@pytest.fixture(scope="module")
def demos():
    import importlib.util
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    pr = next((d for d in (os.path.join(os.path.dirname(here), "paper_runs"),
                           os.path.join(os.path.dirname(os.path.dirname(here)), "paper_runs"))
               if os.path.exists(os.path.join(d, "run_demos.py"))), None)
    if pr is None:
        pytest.skip("paper_runs/ not next to the package")
    spec = importlib.util.spec_from_file_location("paper_run_demos_disk",
                                                  os.path.join(pr, "run_demos.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paper_run_demos_disk"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_bp_disk_gives_both_halves_and_disk_cut_0_gives_neither(demos, monkeypatch):
    monkeypatch.delenv("DISK_CUT", raising=False)
    bands, mask = demos.bp_disk(shape=(101, 101))
    assert list(bands) == demos.BP_DISK and mask is not None and mask.sum() > 0
    monkeypatch.setenv("DISK_CUT", "0")
    bands, mask = demos.bp_disk(shape=(101, 101))
    assert list(bands) == [] and mask is None


def test_the_bands_are_the_measured_disk_axis(demos):
    """PA 28-38 and 208-222 is where the azimuthal excess actually peaks on this cube."""
    centres = sorted(c for c, _ in demos.BP_DISK)
    assert centres == pytest.approx([29.0, 209.0])
    assert all(hw == 20.0 for _, hw in demos.BP_DISK)
    assert abs((centres[1] - centres[0]) - 180.0) < 1.0, "the two lobes must be opposite"


def test_only_beta_pic_stages_get_the_cut(demos):
    """HD 95086 and HIP 65426 have no scattered-light disk; they must be untouched."""
    import os
    src = open(os.path.join(os.path.dirname(demos.__file__), "run_demos.py")).read() \
        if hasattr(demos, "__file__") else ""
    assert "bp_disk(red) if kn == [BP] else ((), None)" in src
    assert src.count("forbidden_pa=fpa, pixel_mask=pmask") >= 5


# ---------------------------------------------- a pinned dimension must not spam the log

def test_pinned_dimensions_do_not_warn_on_identical_axis_limits():
    """A 38-D landscape with a fixed parameter produced thousands of matplotlib warnings in
    the stage log -- one per such cell per panel -- which buries anything real."""
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    import klip_tpe.display as D

    lo, hi = D._widen_flat(np.array([0.0, 5.0, -3.0]), np.array([1.0, 5.0, -3.0]))
    assert (hi > lo).all(), list(zip(lo, hi))
    assert lo[0] == 0.0 and hi[0] == 1.0, "an honest range must be left alone"
    assert abs(0.5 * (lo[1] + hi[1]) - 5.0) < 1e-9, "the pinned value stays centred"

    n = 3
    ad = D.AnnulusData(
        run_name="t", annulus=0, nann=1, inrad=8.0, outrad=30.0, pxscale=0.0272, fwhm=3.6,
        params=[D.ParamInfo("a", "a", 0.0, 1.0, "float", None, "reduction"),
                D.ParamInfo("b", "b", 5.0, 5.0, "float", None, "reduction")],
        partitions=["p1"], X=np.zeros((n, 2)), y=np.linspace(1, 3, n),
        phases=["tpe"] * n, k_used=[5] * n, selected=[["p1"]] * n,
        part_snr=[{"p1": 1.0}] * n, sources=[[(0.4, 30.0, 1e-4)]] * n,
        per_source=[[3.0]] * n, raw_per_source=[[3.0]] * n, clean_per_source=[[0.2]] * n,
        raw=np.linspace(1, 3, n), wall=np.ones(n), contrast=np.full(n, 1e-4),
        configs=[{}] * n, n_init=1, n_iter=n, gamma=0.25, metric_name="mawet",
        search_mode="tpe", known=[])
    cs = {"names": ["a", "b"], "lo": np.array([0.0, 5.0]), "hi": np.array([1.0, 5.0]),
          "X": np.array([[0.5, 5.0], [0.2, 5.0], [0.9, 5.0]]), "y": np.array([1.0, 2.0, 3.0])}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        D.draw_corner(Figure(figsize=(4, 4)), cs, ad, "t")
    assert not [x for x in w if "identical" in str(x.message)], [str(x.message) for x in w]
