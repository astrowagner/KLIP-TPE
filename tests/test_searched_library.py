"""The searched reference library on JWST: ``nkeep_altroll`` and ``nkeep_psfref``.

These two counts replace the ADI / RDI / ADI+RDI categorical.  ``nkeep_psfref = 0`` IS ADI
and ``nkeep_altroll = 0`` IS RDI, and unlike the categorical either pool can contribute part
of itself, ranked per target frame by cross-correlation -- the scheme of
``reduce_nircam_coro.pro`` + ``optimize_mwc_tpe.pro``.

The precondition is structural and easy to get wrong: every science frame must live in ONE
reducer.  Under ``partition='roll'`` each partition holds only its own roll and
``ReferenceLibrary.eligible`` drops every frame sharing the target's label, so the
alternate-roll pool comes back empty and ``nkeep_altroll`` searches nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from klip_tpe.feasibility import ReferenceLibraryGuard
from klip_tpe.instruments import generic
from klip_tpe.reducer import Dataset


def _jwst_like(n_per_roll=4, n_ref=18, n=41, rolls=(110.2, 120.3)):
    """Two rolls plus a dedicated reference star, shaped like HIP 65426 F444W."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:21, 0:21]
    psf = np.exp(-((xx - 10) ** 2 + (yy - 10) ** 2) / 2.0)
    ang = np.concatenate([np.full(n_per_roll, r) for r in rolls])
    ds = Dataset(cube=rng.normal(0, 1, (ang.size, n, n)).astype(np.float32), angles=ang, name="sci")
    ds.ref_cube = rng.normal(0, 1, (n_ref, n, n)).astype(np.float32)
    red = generic.make_reducer({"sci": ds}, pxscale=0.063, lam_m=4.44e-6, diam_m=6.5,
                               use_rdi=True, psf=psf, star_flux=1e5, log=lambda s: None)
    r0 = next(iter(red.reducers.values()))
    r0.set_reference_library(partition=np.round(ang, 1).astype(str), ref_group="psfref",
                             n_min_ref=2, metric="cc")
    return red, r0, ang


def test_the_two_counts_become_searched_dimensions_with_the_pool_sizes():
    red, r0, ang = _jwst_like()
    sp = generic.make_space(red, k_klip_max=18, search_angles=False, opt_framesel=False,
                            selection=None)
    rng = {p.name: (int(p.lo), int(p.hi)) for p in sp.params}
    assert "nkeep_altroll" in rng and "nkeep_psfref" in rng, sorted(rng)
    # altroll's ceiling is the largest ELIGIBLE subset -- the other roll -- not the whole
    # science cube, or the optimizer would spend draws on counts it can never fill
    assert rng["nkeep_altroll"] == (0, 4), rng["nkeep_altroll"]
    assert rng["nkeep_psfref"] == (0, 18), rng["nkeep_psfref"]
    assert "mode" not in rng, "the counts replace the categorical, they do not join it"


def test_a_target_never_draws_references_from_its_own_roll():
    """What stops a roll subtracting its own companion."""
    red, r0, ang = _jwst_like()
    lib = r0._reflib_stub()
    altroll = lib.groups[0]
    part = np.round(ang, 1).astype(str)
    for t in range(ang.size):
        el = lib.eligible(t, altroll)
        assert el.size == 4
        assert part[t] not in set(part[el]), f"target {t} was offered its own roll"


def test_zero_in_one_pool_is_exactly_the_pure_mode():
    red, r0, ang = _jwst_like()
    lib = r0._reflib_stub()
    nsci = ang.size
    rdi = lib.select(0, {"altroll": 0, "psfref": 18})
    adi = lib.select(0, {"altroll": 4, "psfref": 0})
    assert rdi.size == 18 and bool((rdi >= nsci).all()), "nkeep_altroll=0 must be pure RDI"
    assert adi.size == 4 and bool((adi < nsci).all()), "nkeep_psfref=0 must be pure ADI"
    both = lib.select(0, {"altroll": 2, "psfref": 5})
    assert both.size == 7 and (both < nsci).sum() == 2 and (both >= nsci).sum() == 5, \
        "and a mixture takes part of each, which the categorical could not express"


def test_the_empty_basis_is_unreachable_but_both_pure_modes_are_not():
    """Zero in BOTH pools is not a configuration -- KLIP needs at least two frames.  The
    guard raises the sum from the largest pool, so pure ADI and pure RDI both stay
    reachable while (0, 0) does not."""
    red, r0, ang = _jwst_like()
    sp = generic.make_space(red, k_klip_max=18, search_angles=False, opt_framesel=False,
                            selection=None)
    i = {p.name: k for k, p in enumerate(sp.params)}
    g = ReferenceLibraryGuard(n_min_ref=2, k_max=18)

    y = g(np.zeros(len(sp.params)), sp)
    assert y[i["nkeep_altroll"]] + y[i["nkeep_psfref"]] >= 2, "(0, 0) must be projected away"

    for name, other in (("nkeep_altroll", "nkeep_psfref"), ("nkeep_psfref", "nkeep_altroll")):
        x = np.zeros(len(sp.params))
        x[i[name]] = 4.0
        y = g(x, sp)
        assert y[i[other]] == 0.0, f"{other} must be allowed to stay 0 -- that is the pure mode"
        assert y[i[name]] >= 2.0


def test_k_klip_is_clamped_to_the_library_the_counts_buy():
    """Asking 18 KL modes of a 3-frame basis is not a worse configuration, it is not a
    configuration: several draws collapse onto the same reduction and the search pays to
    learn they tie."""
    red, r0, ang = _jwst_like()
    sp = generic.make_space(red, k_klip_max=18, search_angles=False, opt_framesel=False,
                            selection=None)
    i = {p.name: k for k, p in enumerate(sp.params)}
    x = np.zeros(len(sp.params))
    x[i["nkeep_altroll"]], x[i["nkeep_psfref"]] = 1.0, 2.0
    x[i["k_klip"]] = 18.0
    y = ReferenceLibraryGuard(n_min_ref=2, k_max=18)(x, sp)
    assert y[i["k_klip"]] <= 3.0, f"k_klip {y[i['k_klip']]} against a 3-frame library"


def test_roll_partitioning_cannot_feed_the_alternate_roll_pool():
    """The structural precondition, pinned so it cannot regress silently: with one reducer
    per roll every frame shares the target's label and the pool is empty."""
    from klip_tpe.reflib import ReferenceGroup, ReferenceLibrary

    n = 4
    one_roll = np.full(n, "110.2")
    lib = ReferenceLibrary([ReferenceGroup("altroll", np.arange(n), partition=one_roll)],
                           np.zeros((n, n)), target_partition=one_roll)
    assert lib.eligible(0, lib.groups[0]).size == 0, (
        "a reducer holding a single roll can offer no alternate-roll reference -- which is "
        "why the MIRI driver defaults to --partition all")


def test_the_library_changes_the_reduction():
    """The property the whole feature rests on, and the one nothing tested: different counts
    must give different images.  GO 1386 MIRI run v6 searched both counts for 600
    evaluations on a backend that ignored them -- pure ADI, pure RDI and everything gave
    bit-identical scores on 40 common draws -- and the space, the guard and the eligibility
    tests above all passed throughout, because none of them reduces anything."""
    from klip_tpe.reducer import ReductionRequest
    red, r0, ang = _jwst_like()
    base = dict(k_klip=3, inrad=3.0, outrad=9.0, bin=1, n_ang=1, filter=0, use_rdi=True)
    imgs = {}
    for label, (na, nr) in {"adi": (4, 0), "rdi": (0, 18), "all": (4, 18), "part": (2, 6)}.items():
        res = r0.reduce(ReductionRequest(params=dict(base, nkeep_altroll=na, nkeep_psfref=nr)))
        assert res.meta.get("rdi_mode") == "searched", res.meta.get("rdi_mode")
        assert res.meta.get("ref_keep") == {"altroll": na, "psfref": nr}, res.meta.get("ref_keep")
        imgs[label] = np.nan_to_num(np.asarray(res.image, float))
    labels = sorted(imgs)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            assert not np.array_equal(imgs[a], imgs[b]), f"{a} and {b} reduce identically"


@pytest.mark.parametrize("modname,clsname", [("klip_tpe.backends.pyklip", "PyKLIPReducer"),
                                             ("klip_tpe.backends.vip", "VIPReducer"),
                                             ("klip_tpe.backends.custom", "FunctionReducer")])
def test_backends_with_their_own_basis_refuse_a_library(modname, clsname):
    """A backend that builds its own basis must refuse a searched library rather than accept
    it, add the counts to the space, and ignore them.  Checked on a bare instance so the
    test needs neither pyKLIP nor VIP installed: the refusal comes before anything else."""
    import importlib
    cls = getattr(importlib.import_module(modname), clsname)
    r = object.__new__(cls)
    assert r.supports_reference_library is False
    with pytest.raises(NotImplementedError, match="does not apply a searched reference library"):
        r.set_reference_library(partition=np.array(["a", "b"]))
    r._reflib_spec = {"partition": np.array(["a", "b"])}          # even if set behind its back
    assert r.reference_params() == []


def test_the_built_in_engine_accepts_a_library():
    red, r0, ang = _jwst_like()
    assert r0.supports_reference_library is True
    assert [p.name for p in r0.reference_params()] == ["nkeep_altroll", "nkeep_psfref"]
