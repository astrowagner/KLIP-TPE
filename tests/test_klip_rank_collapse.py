"""A KL basis built from the frames it subtracts must never be allowed to be complete.

``auto_fast`` (and an explicit ``fast=True``) without RDI builds ONE basis per zone from the
science frames themselves and projects those same frames onto it.  That basis spans them
exactly, so asking for ``k_klip >= n_frames`` subtracts every frame from itself and leaves
float64 round-off -- an image of ~1e-15 whose S/N is a ratio of two round-off numbers, i.e.
an O(1) random draw.  ``klip_basis`` clamps ``k`` to the row count without complaint, so the
failure is silent, and an optimizer searching ``k_klip`` will chase the lucky draws: on
bench_20260915185229_tpe_s2 (HD 95086, 63 frames, bin 6 -> 12 binned) 548 of 800 evaluations
were degenerate, the best of them scored +2.410 and the best honest one +0.006.

The per-target path is safe (``reference_mask`` drops the target) and so is RDI (the basis
frames are not the target frames).  Only the shared-basis-without-RDI combination is capped.
"""
from __future__ import annotations

import numpy as np
import pytest

from klip_tpe.klip import KLIPParams, klip_annular, klip_basis, reference_mask


def _cube(n=8, ny=48, nx=48, seed=0):
    """A speckly cube with enough structure that a real reduction leaves a real residual."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - (nx - 1) / 2, yy - (ny - 1) / 2)
    halo = 1e3 * np.exp(-r / 6.0)
    speck = sum(rng.normal(0, 40) * np.exp(-((r - rng.uniform(6, 18)) ** 2) / 8.0)
                for _ in range(6))
    return np.stack([halo + speck + rng.normal(0, 1.0, (ny, nx)) for _ in range(n)]).astype(np.float32)


def _rms(a):
    return float(np.nanstd(a))


def test_klip_basis_still_clamps_k_silently():
    """The underlying sharp edge, documented so the cap above it is not removed by accident."""
    R = np.random.default_rng(0).normal(size=(6, 500))
    assert klip_basis(R, 6).shape[0] == 6
    assert klip_basis(R, 99).shape[0] == 6, "k is clamped to the row count, with no error"
    # a complete basis annihilates its own rows
    Z = klip_basis(R, 6)
    resid = R - (R @ Z.T) @ Z
    assert _rms(resid) / _rms(R) < 1e-12, "this is the collapse the cap exists to prevent"


def test_shared_basis_without_rdi_is_capped_below_the_frame_count():
    cube = _cube(n=8)
    ang = np.linspace(0.0, 20.0, 8)
    out = {}
    for k in (4, 7, 8, 12, 30):
        p = KLIPParams(k_klip=k, inrad=5, outrad=20, n_ang=1, fast=True, n_min_ref=1)
        img, info = klip_annular(cube, ang, p, 4.0)
        out[k] = (info, _rms(img))
        assert info["k_effective"] <= cube.shape[0] - 1, f"k={k} left a complete basis"
    assert out[7][0]["k_capped"] is False and out[8][0]["k_capped"] is True
    for k in (8, 12, 30):
        assert out[k][0]["k_effective"] == 7
        assert out[k][0]["k_requested"] == k, "the requested value must stay on the record"
        # capped k behaves exactly like the largest legal k, not like round-off
        assert out[k][1] == pytest.approx(out[7][1], rel=1e-9)
    # and the residual is a real image, not float64 dust
    assert out[7][1] / _rms(cube) > 1e-6


def test_auto_fast_is_the_path_that_reaches_it():
    """angsep=0 with anglemax >= the PA span silently switches to the shared basis, which is
    how a per-target config that looked safe ends up degenerate."""
    cube = _cube(n=8)
    ang = np.linspace(0.0, 20.0, 8)
    p = KLIPParams(k_klip=30, inrad=5, outrad=20, n_ang=1, fast=False, angsep=0.0,
                   anglemax=360.0, n_min_ref=1)
    _, info = klip_annular(cube, ang, p, 4.0)
    assert info["auto_fast"] is True and info["fast"] is True
    assert info["k_capped"] is True and info["k_effective"] == 7


def test_the_per_target_path_is_not_capped():
    """reference_mask drops the target, so its basis never spans the frame it subtracts and
    k = n_ref is legitimate.  Capping here would change results for no reason."""
    n = 8
    ang = np.linspace(0.0, 40.0, n)
    assert reference_mask(ang, 3, 0.0, 360.0)[3] == False  # noqa: E712 -- the whole point
    cube = _cube(n=n)
    p = KLIPParams(k_klip=30, inrad=5, outrad=20, n_ang=1, fast=False, angsep=0.5,
                   anglemax=20.0, n_min_ref=1)
    img, info = klip_annular(cube, ang, p, 4.0)
    assert info["auto_fast"] is False
    assert info["k_capped"] is False, "the per-target path must be left alone"
    assert _rms(img) / _rms(cube) > 1e-6


def test_rdi_is_not_capped_either():
    """With a reference cube the basis frames are not the target frames, so a complete
    reference basis does not annihilate the science frames."""
    cube = _cube(n=4, seed=1)
    ref = _cube(n=6, seed=2)
    ang = np.linspace(0.0, 10.0, 4)
    p = KLIPParams(k_klip=30, inrad=5, outrad=20, n_ang=1, fast=True, n_min_ref=1)
    img, info = klip_annular(cube, ang, p, 4.0, ref_cube=ref)
    assert info["rdi"] is True and info["k_capped"] is False
    assert info["n_basis_frames"] == 6
    assert _rms(img) / _rms(cube) > 1e-6, "RDI with a full reference basis is not degenerate"


def test_the_forward_model_collapses_with_the_image():
    """The KLIP-FM model is computed from the same basis, so it dies the same way -- which is
    what made the dashboard's 'KLIP-FM model' panel a symlog stretch of round-off."""
    cube = _cube(n=8)
    ang = np.linspace(0.0, 20.0, 8)
    model = np.zeros_like(cube)
    model[:, 24, 30] = 50.0                      # a crude point source, same in every frame
    p = KLIPParams(k_klip=30, inrad=5, outrad=20, n_ang=1, fast=True, n_min_ref=1)
    img, info, fm = klip_annular(cube, ang, p, 4.0, fm_cube=model)
    assert info["k_capped"] is True
    assert np.nanmax(np.abs(fm)) / 50.0 > 1e-6, "a capped FM must carry real planet response"


def test_a_k_scan_returns_one_image_per_REQUESTED_k_even_when_capped():
    """The cap must not change the k-scan's output length.

    A PartitionedReducer stacks the per-partition scans, so two channels whose binned frame
    counts differ by one would cap at different k and hand np.stack arrays of different
    length -- "all input arrays must have the same shape".  That is not hypothetical: it
    broke the calibration k-scan on HD 95086's [20, 36] px annulus, which then fell back to
    k = 10 where the optimum is k = 2, and the contrast loop diverged (median S/N 2.02 and
    falling as the contrast rose) instead of settling at 1.11e-05 / S/N 5.7.

    Slices past the cap repeat the capped image, which is what the non-scan path returns for
    those k anyway, so a k-scan plateaus instead of lying or crashing.
    """
    cube = _cube(n=6)
    ang = np.linspace(0.0, 20.0, 6)
    kreq = 11
    p = KLIPParams(k_klip=kreq, inrad=5, outrad=20, n_ang=1, fast=True, n_min_ref=1, k_scan=True)
    out, info = klip_annular(cube, ang, p, 4.0)
    assert info["k_capped"] is True and info["k_effective"] == 5
    assert out.shape[0] == kreq, f"k-scan must return {kreq} images, got {out.shape[0]}"
    # up to the cap the images differ; past it they repeat
    for kk in range(info["k_effective"], kreq):
        assert np.allclose(np.nan_to_num(out[kk]), np.nan_to_num(out[info["k_effective"] - 1])), \
            f"slice {kk} past the cap must repeat the capped image"
    assert not np.allclose(np.nan_to_num(out[0]), np.nan_to_num(out[3])), "and below it, not"
    # the capped scan's last useful slice is the same image the non-scan path gives
    p2 = KLIPParams(k_klip=kreq, inrad=5, outrad=20, n_ang=1, fast=True, n_min_ref=1)
    img, _ = klip_annular(cube, ang, p2, 4.0)
    assert np.allclose(np.nan_to_num(out[-1]), np.nan_to_num(img), atol=1e-6)


def test_two_partitions_capping_differently_still_stack():
    """The exact shape mismatch, in the form the reducer hits it: one channel with 6 frames
    and one with 7, scanned to the same requested k."""
    ang6, ang7 = np.linspace(0, 20, 6), np.linspace(0, 20, 7)
    kreq = 9
    shapes = []
    for cube, ang in ((_cube(n=6), ang6), (_cube(n=7, seed=3), ang7)):
        p = KLIPParams(k_klip=kreq, inrad=5, outrad=20, n_ang=1, fast=True, n_min_ref=1, k_scan=True)
        out, info = klip_annular(cube, ang, p, 4.0)
        shapes.append(out.shape[0])
        assert info["k_effective"] == cube.shape[0] - 1
    assert shapes[0] == shapes[1] == kreq, f"scans must be stackable, got {shapes}"
    np.stack([np.zeros((s, 4, 4)) for s in shapes])      # would raise before the fix
