"""A disk of any shape kept out of the noise estimate and the injections.

``pa_wedge_mask`` and ``forbidden_pa`` cover a disk seen near edge-on, at a position angle.
Spiral arms or a ring follow no wedge, so ``disk_mask_from_image`` masks the bright extended
structure of an image of the disk, and ``default_config(disk_mask=...)`` hands that one mask to
both halves: the metric's ``pixel_mask`` and the sampler's ``forbidden_mask``.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from klip_tpe.metrics import disk_mask_from_image, mawet_peak_snr, source_xy, star_center
from klip_tpe.positions import PositionSampler

PXS = 0.0345          # ALES arcsec per spaxel
FWHM = 2.9


def _arc_image(n=81, seed=1, amp=6.0):
    """Unit noise plus a spiral arc from 12 to 21 px, peak ``amp``; the arc's pixels."""
    rng = np.random.default_rng(seed)
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - c, yy - c)
    th = np.degrees(np.arctan2(yy - c, xx - c))
    arc = (np.abs(rr - (12.0 + 0.05 * (th + 180.0))) < 1.5) & (th > -120.0) & (th < 60.0)
    blur = ndimage.gaussian_filter(arc.astype(float), 1.0)
    return rng.normal(0.0, 1.0, (n, n)) + amp * blur / blur.max(), arc


# ------------------------------------------------------------------------- the builder

def test_the_mask_covers_the_arc_and_little_else():
    img, arc = _arc_image()
    m = disk_mask_from_image(img, FWHM, nsig=2.0, r_range=(5, 35))
    assert (m & arc).sum() == arc.sum(), "the arc must be masked whole"
    far = ndimage.distance_transform_edt(~arc) > 2 * FWHM
    assert (m & far).sum() < 0.02 * far.sum(), "isolated noise must not be masked"


def test_flat_noise_gives_next_to_nothing():
    rng = np.random.default_rng(3)
    m = disk_mask_from_image(rng.normal(0, 1, (81, 81)), FWHM, nsig=3.0, r_range=(5, 35))
    assert m.sum() < 0.01 * m.size


def test_growth_widens_and_the_radial_range_limits():
    img, arc = _arc_image()
    tight = disk_mask_from_image(img, FWHM, nsig=2.0, r_range=(5, 35), grow_fwhm=0.0)
    wide = disk_mask_from_image(img, FWHM, nsig=2.0, r_range=(5, 35), grow_fwhm=1.0)
    assert tight.sum() < wide.sum() and not (tight & ~wide).any()
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - (n - 1) / 2.0, yy - (n - 1) / 2.0)
    inner = disk_mask_from_image(img, FWHM, nsig=2.0, r_range=(5, 15))
    assert inner.any() and rr[inner].max() <= 15.0 and rr[inner].min() >= 5.0


def test_small_pieces_are_dropped():
    rng = np.random.default_rng(5)
    img = rng.normal(0, 0.2, (81, 81))
    img[20, 20] = 30.0                                   # one hot spaxel, well under two beams
    assert not disk_mask_from_image(img, FWHM, nsig=3.0, min_beams=2.0, smooth_fwhm=0.0).any()
    assert disk_mask_from_image(img, FWHM, nsig=3.0, min_beams=0.0, smooth_fwhm=0.0)[20, 20]


def test_a_protected_companion_stays_unmasked():
    img, arc = _arc_image()
    n = img.shape[0]
    cx, cy = star_center(img.shape)
    iy, ix = np.argwhere(arc)[len(np.argwhere(arc)) // 2]      # a point on the arc
    rho = float(np.hypot(ix - cx, iy - cy)) * PXS
    pa = float(np.degrees(np.arctan2(iy - cy, ix - cx))) - 90.0   # inverse of source_xy ('pa')
    m = disk_mask_from_image(img, FWHM, nsig=2.0, r_range=(5, 35), protect=[(rho, pa)], pxscale=PXS)
    yy, xx = np.mgrid[0:n, 0:n]
    assert not m[np.hypot(xx - ix, yy - iy) <= 1.5 * FWHM].any()
    assert m.sum() > 0.5 * arc.sum(), "the rest of the arc stays masked"
    with pytest.raises(ValueError):
        disk_mask_from_image(img, FWHM, protect=[(rho, pa)])


def test_nans_are_tolerated():
    img, arc = _arc_image()
    img[:, :6] = np.nan
    m = disk_mask_from_image(img, FWHM, nsig=2.0, r_range=(5, 35))
    assert m.dtype == bool and (m & arc).sum() > 0.9 * arc.sum()


def _arc_with_troughs(n=81, seed=1, amp=6.0, gap=4.0):
    """_arc_image's arc with an ADI-like trough of -amp/2 ``gap`` px either side of it, and one
    trough of -amp nowhere near it; the arc's, the troughs' and the far trough's pixels."""
    rng = np.random.default_rng(seed)
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - c, yy - c)
    th = np.degrees(np.arctan2(yy - c, xx - c))
    span = (th > -120.0) & (th < 60.0)
    r_arc = 12.0 + 0.05 * (th + 180.0)
    arc = (np.abs(rr - r_arc) < 1.5) & span
    trough = ((np.abs(rr - r_arc - gap) < 1.5) | (np.abs(rr - r_arc + gap) < 1.5)) & span
    far = np.hypot(xx - (c + 30.0 * np.cos(np.radians(150.0))), yy - (c + 30.0 * np.sin(np.radians(150.0)))) < 3.0

    def soft(m):
        b = ndimage.gaussian_filter(m.astype(float), 1.0)
        return b / b.max()

    img = rng.normal(0.0, 1.0, (n, n)) + amp * soft(arc) - 0.5 * amp * soft(trough) - amp * soft(far)
    return img, arc, trough, far


def test_the_disk_s_adi_negatives_join_the_mask_and_a_far_trough_does_not():
    img, arc, trough, far = _arc_with_troughs()
    bright = disk_mask_from_image(img, FWHM, nsig=2.0, r_range=(5, 35), grow_fwhm=0.0)
    both = disk_mask_from_image(img, FWHM, nsig=2.0, r_range=(5, 35), grow_fwhm=0.0, nsig_neg=1.5)
    assert not (bright & ~both).any(), "the negatives only add"
    assert (bright & trough).sum() < 0.1 * trough.sum(), "the bright cut must not take the troughs"
    assert (both & trough).sum() > 0.6 * trough.sum(), "the troughs beside the arc are the disk's"
    assert not (both & far).any(), "a trough nowhere near the disk is not"
    near_only = disk_mask_from_image(img, FWHM, nsig=2.0, r_range=(5, 35), grow_fwhm=0.0, nsig_neg=1.5,
                                     neg_reach_fwhm=0.1)
    assert (near_only & trough).sum() < (both & trough).sum(), "the reach decides which troughs count"


def test_without_a_bright_disk_no_negatives_are_taken():
    rng = np.random.default_rng(4)
    img = rng.normal(0, 1, (81, 81))
    yy, xx = np.mgrid[0:81, 0:81]
    img -= 8.0 * np.exp(-0.5 * ((xx - 20.0) ** 2 + (yy - 60.0) ** 2) / 2.0 ** 2)
    assert not disk_mask_from_image(img, FWHM, nsig=6.0, r_range=(5, 38), nsig_neg=2.0).any()


# ----------------------------------------------------------- the noise ring honours it

def test_masking_the_arc_lowers_the_noise_next_to_it():
    img, arc = _arc_image(amp=20.0)
    m = disk_mask_from_image(img, FWHM, nsig=2.0, r_range=(5, 35))
    kw = dict(pxscale=PXS, fwhm=FWHM, known=(), declip_nsig=None, return_details=True)
    _, hot = mawet_peak_snr(img, [16 * PXS], [150.0], **kw)
    _, cold = mawet_peak_snr(img, [16 * PXS], [150.0], pixel_mask=m, **kw)
    assert cold[0]["sigma"] < 0.6 * hot[0]["sigma"], (hot[0]["sigma"], cold[0]["sigma"])
    assert cold[0]["nclean"] >= 6


# ----------------------------------------------------------------- the injections avoid it

def _draws(sampler, n=150, r=16 * PXS):
    rng = np.random.default_rng(0)
    return [s for _ in range(n) for s in sampler.sample(2, r, r, rng, 1e-3)]


def test_no_injection_lands_on_the_mask():
    img, _ = _arc_image()
    m = disk_mask_from_image(img, FWHM, nsig=2.0, r_range=(5, 35))
    cx, cy = star_center(m.shape)

    def on(srcs):
        xs, ys = source_xy([s.rho for s in srcs], [s.theta for s in srcs], PXS, cx, cy)
        return sum(bool(m[int(round(y)), int(round(x))]) for x, y in zip(xs, ys))

    free = PositionSampler(fwhm_as=FWHM * PXS)
    cut = PositionSampler(fwhm_as=FWHM * PXS, forbidden_mask=m, mask_pxscale=PXS)
    assert on(_draws(free)) > 0, "the test needs the free sampler to hit the arc"
    assert on(_draws(cut)) == 0


def test_the_clearance_keeps_them_further_off():
    img, _ = _arc_image()
    m = disk_mask_from_image(img, FWHM, nsig=2.0, r_range=(5, 35), grow_fwhm=0.0)
    d = ndimage.distance_transform_edt(~m)
    cx, cy = star_center(m.shape)
    s = PositionSampler(fwhm_as=FWHM * PXS, forbidden_mask=m, mask_pxscale=PXS, mask_clear_fwhm=1.0)
    srcs = _draws(s)
    xs, ys = source_xy([q.rho for q in srcs], [q.theta for q in srcs], PXS, cx, cy)
    assert min(d[int(round(y)), int(round(x))] for x, y in zip(xs, ys)) > FWHM


def test_a_mask_needs_its_pixel_scale():
    s = PositionSampler(fwhm_as=0.1, forbidden_mask=np.ones((41, 41), bool))
    with pytest.raises(ValueError):
        s.sample(2, 0.3, 0.3, np.random.default_rng(0))


def test_the_record_says_whether_a_mask_was_in_force():
    plain = PositionSampler(fwhm_as=0.1).describe()
    assert "forbidden_mask_px" not in plain                  # runs without a mask: as they were
    m = np.zeros((41, 41), bool)
    m[5:9, 5:9] = True
    d = PositionSampler(fwhm_as=0.1, forbidden_mask=m, mask_pxscale=PXS, mask_clear_fwhm=0.5).describe()
    assert d["forbidden_mask_px"] == 16 and d["mask_clear_fwhm"] == 0.5


# ------------------------------------------------------------- one mask, both halves

class _R:
    fwhm, pxscale = FWHM, PXS

    def matched_filter_kernel(self, rho):
        return None


def test_default_config_gives_the_mask_to_the_metric_and_the_sampler():
    import klip_tpe.instruments.generic as G
    img, _ = _arc_image()
    m = disk_mask_from_image(img, FWHM, nsig=2.0, r_range=(5, 35))
    obj, samp = G.default_config(_R(), known=[], disk_mask=m, disk_mask_clear_fwhm=0.5)
    assert np.array_equal(obj.metric.pixel_mask, m)
    assert np.array_equal(samp.forbidden_mask, m) and samp.mask_pxscale == PXS and samp.mask_clear_fwhm == 0.5
    assert obj.metric.describe()["pixel_mask_px"] == samp.describe()["forbidden_mask_px"] == int(m.sum())
    obj0, samp0 = G.default_config(_R(), known=[])
    assert obj0.metric.pixel_mask is None and samp0.forbidden_mask is None


def test_a_given_pixel_mask_is_kept_alongside():
    import klip_tpe.instruments.generic as G
    m = np.zeros((41, 41), bool)
    m[2:4, 2:4] = True
    pm = np.zeros((41, 41), bool)
    pm[30:33, 30:33] = True
    obj, samp = G.default_config(_R(), known=[], disk_mask=m, pixel_mask=pm)
    assert np.array_equal(obj.metric.pixel_mask, m | pm)
    assert np.array_equal(samp.forbidden_mask, m)               # the sampler gets the disk only
    with pytest.raises(ValueError):
        G.default_config(_R(), known=[], disk_mask=m, pixel_mask=np.zeros((21, 21), bool))
