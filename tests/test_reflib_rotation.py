"""A searched reference library over one rotating sequence: the ADI rule as the eligibility.

From the ALES reduction's note of 2026-10-14: one continuously rotating sequence has no
roll label to exclude references by, and a library that admitted frames with no rotation
would subtract every source from itself.  What must hold:

* a ``rotation`` group offers a target the rows ``angsep_deg <= |dPA| <= anglemax`` and never
  itself, the label rule on top of it only when there is more than one label, and the
  ``< 4`` fallback of ``klip_annular`` (every row but the target) outside the census;
* a pool of one label without the rotation rule stays empty, as the NIRCam precondition
  requires;
* with every eligible science frame kept and no reference frames the per-target reduction is
  plain ADI to the bit, through ``klip_annular`` and through the reducer, at several
  ``angsep`` / ``anglemax`` and with KLIP-FM;
* ``nkeep_psfref`` changes the image once it is above zero, and the reducer records the
  library.
"""
from __future__ import annotations

import numpy as np
import pytest

from klip_tpe.injection import GaussianPSF
from klip_tpe.klip import KLIPParams, klip_annular, reference_mask
from klip_tpe.reducer import KLIPReducer, ReductionRequest
from klip_tpe.reflib import ReferenceGroup, ReferenceLibrary, similarity_matrix
from klip_tpe.synthetic import synthetic_klip_dataset


def _lib(n, angles, ext=0, rotation=True, partition=None):
    sci_rows = np.arange(n)
    part = np.zeros(n, int) if partition is None else np.asarray(partition)
    groups = [ReferenceGroup("altroll", sci_rows, partition=part, rotation=rotation)]
    if ext:
        groups.append(ReferenceGroup("psfref", np.arange(n, n + ext)))
    sim = np.zeros((n + ext, n))
    return ReferenceLibrary(groups, sim, target_partition=part, angles=angles)


def test_the_rotation_rule_decides_the_eligible_rows():
    ang = np.array([0.0, 2.0, 5.0, 9.0, 14.0, 20.0, 27.0, 35.0])
    lib = _lib(8, ang)
    g = lib.groups[0]
    assert g.rotation and g.n_labels == 1 and g.max_keep == 7, "one label, every other frame reachable"
    for t in range(8):
        want = np.flatnonzero(reference_mask(ang, t, 4.0, 15.0))
        got = lib.eligible(t, g, 4.0, 15.0, census=True)
        np.testing.assert_array_equal(got, want), t
        assert t not in got
    # fewer than 4 admitted: the census says so, the selection falls back to every other row
    assert lib.eligible(0, g, 30.0, 40.0, census=True).size == 1
    np.testing.assert_array_equal(lib.eligible(0, g, 30.0, 40.0), np.arange(1, 8))
    # the library's angles must cover the rows
    with pytest.raises(ValueError, match="angles"):
        ReferenceLibrary([ReferenceGroup("altroll", np.arange(8), rotation=True)], np.zeros((8, 8)))
    # two labels and the rotation rule: both apply
    part = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    two = _lib(8, ang, partition=part)
    np.testing.assert_array_equal(two.eligible(0, two.groups[0], 0.0, 360.0, census=True), np.array([4, 5, 6, 7]))
    np.testing.assert_array_equal(two.eligible(0, two.groups[0], 15.0, 360.0, census=True), np.array([5, 6, 7]))


def test_one_label_without_the_rotation_rule_stays_empty():
    lib = _lib(5, np.arange(5.0), rotation=False)
    assert lib.groups[0].max_keep == 0 and lib.eligible(0, lib.groups[0]).size == 0


@pytest.mark.parametrize("angsep,anglemax,k", [(0.0, 360.0, 3), (1.0, 360.0, 4), (0.5, 12.0, 2), (3.0, 25.0, 5)])
def test_a_full_rotation_library_is_plain_adi_to_the_bit(angsep, anglemax, k):
    rng = np.random.default_rng(21)
    n, ny = 14, 40
    base = rng.normal(0, 1, (ny, ny))
    cube = np.stack([base * (1 + 0.1 * i) + rng.normal(0, 0.3, (ny, ny)) for i in range(n)]).astype(np.float32)
    ang = np.linspace(-20.0, 24.0, n)
    fm = (0.1 * rng.normal(0, 1, cube.shape)).astype(np.float32)
    kw = dict(k_klip=k, inrad=4.0, outrad=16.0, n_ang=2, n_min_ref=2, angsep=angsep, anglemax=anglemax)
    adi, info_a, fm_a = klip_annular(cube, ang, KLIPParams(**kw), 2.5, fm_cube=fm)
    lib = ReferenceLibrary([ReferenceGroup("altroll", np.arange(n), partition=np.zeros(n, int), rotation=True)],
                           similarity_matrix(cube, cube, inrad=4.0, outrad=16.0), target_partition=np.zeros(n, int),
                           angles=ang)
    out, info_l, fm_l = klip_annular(cube, ang, KLIPParams(ref_lib=lib, ref_keep={"altroll": n - 1}, **kw), 2.5,
                                     fm_cube=fm, ref_cube=cube, fm_ref_cube=fm)
    # the ADI code path, shortcuts included (no exclusion at all: one shared basis)
    assert info_l["library_as_adi"] is True and info_l["fast"] == info_a["fast"]
    assert info_a["fast"] == (angsep <= 0 and anglemax >= np.ptp(ang))
    np.testing.assert_array_equal(info_l["nref"], info_a["nref"])
    np.testing.assert_array_equal(info_l["nref_used"], info_a["nref_used"])
    np.testing.assert_array_equal(np.nan_to_num(out, nan=-7), np.nan_to_num(adi, nan=-7))
    np.testing.assert_array_equal(np.nan_to_num(fm_l, nan=-7), np.nan_to_num(fm_a, nan=-7))
    # a truncated science pool is a different reduction
    cut, info_c = klip_annular(cube, ang, KLIPParams(ref_lib=lib, ref_keep={"altroll": 3}, **kw), 2.5, ref_cube=cube)
    assert not np.array_equal(np.nan_to_num(cut), np.nan_to_num(adi))
    assert info_c["library_as_adi"] is False and info_c["fast"] is False
    assert (info_c["nref_used"] <= 3).all()


def _reducer(nref=12, rotation=True):
    ds = synthetic_klip_dataset(nframes=24, size=48, pa_span=40, seed=5)
    rng = np.random.default_rng(6)
    ds.ref_cube = np.stack([ds.cube[i % 24] * 1.02 + rng.normal(0, 0.05, (48, 48)) for i in range(nref)]).astype(np.float32)
    red = KLIPReducer(ds, pxscale=0.05, lam_m=3.6e-6, diam_m=8.4, injection_model=GaussianPSF(2.6, star_flux=1e4),
                      fwhm_px=2.6, outrad_cap=20)
    if rotation:
        red.set_reference_library(partition=np.zeros(24, int), ref_group="psfref", n_min_ref=2, rotation=True)
    return red


def _p(**kw):
    p = dict(k_klip=3, bin=2, filter=5, n_ang=2, inrad=4, outrad=16, angsep=0.8, anglemax=30.0, corr_thresh=0.0,
             noise_max=3.0, coronoise_max=3.0, n_min_ref=2, hp_nan_aware=True)
    p.update(kw)
    return p


def test_through_the_reducer_zero_reference_frames_is_adi_to_the_bit():
    plain = _reducer(rotation=False)
    lib = _reducer()
    assert {p.name for p in lib.reference_params()} == {"nkeep_altroll", "nkeep_psfref"}
    assert lib.describe()["reference_library"]["rotation"] is True
    a = plain.reduce(ReductionRequest(_p(use_rdi=False), None))
    for na in (23, 100):                                      # "all" at or above the pool
        b = lib.reduce(ReductionRequest(_p(use_rdi=True, nkeep_altroll=na, nkeep_psfref=0), None))
        np.testing.assert_array_equal(np.nan_to_num(b.image, nan=-7), np.nan_to_num(a.image, nan=-7))
        np.testing.assert_array_equal(b.meta["nref"], a.meta["nref"])
        assert b.meta["rdi_mode"] == "searched" and b.meta["ref_keep"] == {"altroll": na, "psfref": 0}
    # the same with injections and the forward model
    from klip_tpe.metrics import Source
    src = [Source(0.4, 70.0, 2e-3)]
    a = plain.reduce(ReductionRequest(_p(use_rdi=False), src, fm_sources=src))
    b = lib.reduce(ReductionRequest(_p(use_rdi=True, nkeep_altroll=23, nkeep_psfref=0), src, fm_sources=src))
    np.testing.assert_array_equal(np.nan_to_num(b.image, nan=-7), np.nan_to_num(a.image, nan=-7))
    np.testing.assert_array_equal(np.nan_to_num(b.fm_image, nan=-7), np.nan_to_num(a.fm_image, nan=-7))
    # reference frames in: a different reduction, their count recorded
    c = lib.reduce(ReductionRequest(_p(use_rdi=True, nkeep_altroll=23, nkeep_psfref=6), None))
    assert not np.array_equal(np.nan_to_num(c.image), np.nan_to_num(a.image))
    assert c.meta["ref_keep"]["psfref"] == 6 and (c.meta["nref_used"] >= 6).all()
    d = lib.reduce(ReductionRequest(_p(use_rdi=True, nkeep_altroll=0, nkeep_psfref=6), None))
    assert (d.meta["nref_used"] == 6).all(), "the reference star alone is pure RDI"
