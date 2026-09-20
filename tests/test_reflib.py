"""Searched reference libraries (klip_tpe.reflib) -- the MWC 758 refsel scheme.

The properties that matter, and that a count-only reading of the IDL would get wrong:

* selection is PER TARGET FRAME, not per sequence;
* a target never draws references from its own partition (its own roll contains the
  companion, and a basis built from it subtracts the companion);
* the counts are what the optimizer searches, and asking for more KL modes than the
  retained library holds is projected away rather than evaluated;
* with every group at full size the result matches the plain fixed-library RDI, so the
  feature costs nothing when it is not used.
"""
import numpy as np
import pytest

from klip_tpe.klip import KLIPParams, klip_annular
from klip_tpe.reflib import (ReferenceGroup, ReferenceLibrary, similarity_matrix,
                             science_and_reference_library)


def _speckle_cube(n, ny=32, nx=32, seed=0, scale=1.0, common=None):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (ny, nx)) if common is None else common
    return np.stack([base * scale + rng.normal(0, 0.2, (ny, nx)) for _ in range(n)]), base


def test_similarity_ranks_the_frames_that_actually_look_alike():
    """A library frame built from the target's own speckle realisation must rank above
    one built from an unrelated realisation -- otherwise the ranking is noise and the
    whole selection is a lottery."""
    common = np.random.default_rng(1).normal(0, 1, (32, 32))
    target, _ = _speckle_cube(4, seed=2, common=common)
    like, _ = _speckle_cube(5, seed=3, common=common)          # same speckle field
    unlike, _ = _speckle_cube(5, seed=4)                        # different field
    lib = np.concatenate([like, unlike])
    cc = similarity_matrix(target, lib, inrad=3, outrad=14, filt=0, metric="cc", shift_px=0)
    assert cc.shape == (10, 4)
    assert cc[:5].mean() > cc[5:].mean() + 0.2, cc.mean(axis=1)
    ssr = similarity_matrix(target, lib, inrad=3, outrad=14, filt=0, metric="ssr", shift_px=0)
    assert ssr[:5].mean() < ssr[5:].mean(), "ssr is a residual: smaller is better"


def test_a_target_never_draws_references_from_its_own_partition():
    """The same-roll exclusion.  This is the one rule whose absence would still produce a
    plausible-looking image -- and a self-subtracted companion."""
    sci, _ = _speckle_cube(6, seed=5)
    part = np.array(["A", "A", "A", "B", "B", "B"])
    ext, _ = _speckle_cube(4, seed=6)
    _, lib = science_and_reference_library(sci, part, {"star": ext}, inrad=3, outrad=14)
    for t in range(6):
        rows = lib.select(t, {"altroll": 99, "star": 99})
        own = np.flatnonzero(part == part[t])
        assert not set(rows) & set(own.tolist()), f"target {t} drew from its own roll"
        assert set(rows) >= set(range(6, 10)), "the external star should always be eligible"


def test_counts_are_honoured_per_group_and_per_target():
    sci, _ = _speckle_cube(6, seed=7)
    part = np.array(["A"] * 3 + ["B"] * 3)
    ext, _ = _speckle_cube(8, seed=8)
    _, lib = science_and_reference_library(sci, part, {"star": ext}, inrad=3, outrad=14)
    for keep, want_alt, want_ext in (({"altroll": 2, "star": 3}, 2, 3),
                                     ({"altroll": 0, "star": 5}, 0, 5),
                                     ({"altroll": 99, "star": 0}, 3, 0)):
        for t in range(6):
            rows = lib.select(t, keep)
            assert (rows < 6).sum() == want_alt, (keep, t, rows)
            assert (rows >= 6).sum() == want_ext, (keep, t, rows)


def test_different_targets_get_different_libraries():
    """If every target ended up with the same rows the per-target machinery would be
    an expensive way to do what the shared basis already does."""
    rng = np.random.default_rng(9)
    sci = np.stack([rng.normal(0, 1, (32, 32)) for _ in range(6)])
    part = np.array(["A"] * 3 + ["B"] * 3)
    ext = np.stack([rng.normal(0, 1, (32, 32)) for _ in range(8)])
    _, lib = science_and_reference_library(sci, part, {"star": ext}, inrad=3, outrad=14)
    picks = {t: tuple(lib.select(t, {"altroll": 1, "star": 2})) for t in range(6)}
    assert len(set(picks.values())) > 1, picks


def test_project_clamps_k_and_rescues_a_starved_draw():
    sci, _ = _speckle_cube(6, seed=10)
    ext, _ = _speckle_cube(8, seed=11)
    _, lib = science_and_reference_library(sci, np.array(["A"] * 3 + ["B"] * 3),
                                           {"star": ext}, inrad=3, outrad=14, n_min_ref=4)
    # above the floor: the counts stand and only k_klip moves
    cfg = lib.project({"nkeep_altroll": 2, "nkeep_star": 3, "k_klip": 30})
    assert (cfg["nkeep_altroll"], cfg["nkeep_star"], cfg["k_klip"]) == (2, 3, 5), cfg
    # below it: the total is raised from the largest pool first, then k_klip follows
    rescued = lib.project({"nkeep_altroll": 1, "nkeep_star": 2, "k_klip": 30})
    assert (rescued["nkeep_altroll"], rescued["nkeep_star"], rescued["k_klip"]) == (1, 3, 4), rescued
    starved = lib.project({"nkeep_altroll": 0, "nkeep_star": 0, "k_klip": 5})
    assert starved["nkeep_altroll"] + starved["nkeep_star"] >= 4, starved
    assert starved["k_klip"] <= starved["nkeep_altroll"] + starved["nkeep_star"]


def test_params_expose_one_searched_count_per_group():
    sci, _ = _speckle_cube(4, seed=12)
    ext, _ = _speckle_cube(9, seed=13)
    _, lib = science_and_reference_library(sci, np.array(["A", "A", "B", "B"]), {"hd": ext},
                                           inrad=3, outrad=14, min_keep={"hd": 1})
    ps = {p.name: p for p in lib.params()}
    assert set(ps) == {"nkeep_altroll", "nkeep_hd"}
    # four science frames in two rolls of two: a target can reach the other roll only, so
    # the ceiling is 2, not 4.  Bounding at 4 would make half the range a tie.
    assert (ps["nkeep_altroll"].lo, ps["nkeep_altroll"].hi) == (0.0, 2.0)
    assert (ps["nkeep_hd"].lo, ps["nkeep_hd"].hi) == (1.0, 9.0), "min_keep must reach the bound"
    assert all(p.kind == "int" for p in ps.values())


def test_full_library_reproduces_the_shared_basis_rdi():
    """Every group at full size and no partition exclusion = the fixed library, so the
    per-target path must land on the same image as the single-basis path."""
    rng = np.random.default_rng(14)
    sci = np.stack([rng.normal(0, 1, (32, 32)) for _ in range(5)])
    ext = np.stack([rng.normal(0, 1, (32, 32)) for _ in range(9)])
    ang = np.linspace(0, 40, 5)
    kw = dict(k_klip=4, inrad=3, outrad=14, n_ang=1, n_min_ref=2)
    shared, _ = klip_annular(sci, ang, KLIPParams(**kw), 3.0, ref_cube=ext)

    grp = [ReferenceGroup("ext", np.arange(5, 14))]          # science rows excluded entirely
    lib = ReferenceLibrary(grp, similarity_matrix(sci, np.concatenate([sci, ext]),
                                                  inrad=3, outrad=14), n_min_ref=2)
    per_target, info = klip_annular(sci, ang, KLIPParams(ref_lib=lib, ref_keep={"ext": 9}, **kw),
                                    3.0, ref_cube=np.concatenate([sci, ext]))
    assert info["fast"] is False, "a searched library must take the per-target path"
    a, b = np.nan_to_num(shared), np.nan_to_num(per_target)
    assert np.allclose(a, b, atol=1e-5), np.abs(a - b).max()


def test_fewer_references_changes_the_image():
    rng = np.random.default_rng(15)
    sci = np.stack([rng.normal(0, 1, (32, 32)) for _ in range(5)])
    ext = np.stack([rng.normal(0, 1, (32, 32)) for _ in range(9)])
    full = np.concatenate([sci, ext])
    lib = ReferenceLibrary([ReferenceGroup("ext", np.arange(5, 14))],
                           similarity_matrix(sci, full, inrad=3, outrad=14), n_min_ref=2)
    kw = dict(k_klip=3, inrad=3, outrad=14, n_ang=1, n_min_ref=2)
    a, _ = klip_annular(sci, np.linspace(0, 40, 5),
                        KLIPParams(ref_lib=lib, ref_keep={"ext": 9}, **kw), 3.0, ref_cube=full)
    b, ib = klip_annular(sci, np.linspace(0, 40, 5),
                         KLIPParams(ref_lib=lib, ref_keep={"ext": 3}, **kw), 3.0, ref_cube=full)
    assert ib["ref_keep"] == {"ext": 3}
    assert not np.allclose(np.nan_to_num(a), np.nan_to_num(b))
    assert (ib["nref_used"] == 3).all(), ib["nref_used"]


def test_a_starved_count_falls_back_instead_of_crashing():
    rng = np.random.default_rng(16)
    sci = np.stack([rng.normal(0, 1, (32, 32)) for _ in range(4)])
    ext = np.stack([rng.normal(0, 1, (32, 32)) for _ in range(5)])
    full = np.concatenate([sci, ext])
    lib = ReferenceLibrary([ReferenceGroup("ext", np.arange(4, 9))],
                           similarity_matrix(sci, full, inrad=3, outrad=14), n_min_ref=2)
    out, info = klip_annular(sci, np.linspace(0, 30, 4),
                             KLIPParams(k_klip=2, inrad=3, outrad=14, n_ang=1, n_min_ref=1,
                                        ref_lib=lib, ref_keep={"ext": 0}), 3.0, ref_cube=full)
    assert np.isfinite(out).any(), "a zero-count draw should score badly, not produce NaN"
    assert (info["nref_used"] == 5).all(), info["nref_used"]


def test_library_built_for_a_different_frame_count_is_refused():
    rng = np.random.default_rng(17)
    sci = np.stack([rng.normal(0, 1, (32, 32)) for _ in range(4)])
    full = np.concatenate([sci, sci])
    lib = ReferenceLibrary([ReferenceGroup("ext", np.arange(4, 8))],
                           similarity_matrix(sci, full, inrad=3, outrad=14))
    with pytest.raises(ValueError, match="built for 4 target"):
        klip_annular(sci[:3], np.zeros(3),
                     KLIPParams(k_klip=1, inrad=3, outrad=14, ref_lib=lib), 3.0, ref_cube=full)


# ---------------------------------------------------------------- through the reducer
def _reducer_with_two_rolls(nframes=16, nref=10, size=48):
    """A two-roll science sequence plus a dedicated reference star, wired for a searched
    library the way the MWC 758 run is."""
    from klip_tpe.injection import GaussianPSF
    from klip_tpe.reducer import KLIPReducer
    from klip_tpe.synthetic import synthetic_klip_dataset
    ds = synthetic_klip_dataset(nframes=nframes, size=size, pa_span=20, seed=3)
    rng = np.random.default_rng(4)
    ds.ref_cube = np.stack([ds.cube[i % nframes] * 1.01 + rng.normal(0, 0.05, (size, size))
                            for i in range(nref)]).astype(np.float32)
    red = KLIPReducer(ds, pxscale=0.05, lam_m=4.4e-6, diam_m=6.5,
                      injection_model=GaussianPSF(3.0, star_flux=1e4), fwhm_px=3.0, outrad_cap=20)
    part = np.array(["rollA"] * (nframes // 2) + ["rollB"] * (nframes - nframes // 2))
    red.set_reference_library(partition=part, min_keep={"psfref": 1}, n_min_ref=2)
    return red, part


def _p(**kw):
    p = dict(k_klip=3, bin=1, filter=0, n_ang=1, inrad=4, outrad=16, angsep=0.0, anglemax=360.0,
             corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0, use_rdi=True, n_min_ref=2)
    p.update(kw)
    return p


def test_reducer_exposes_one_searched_count_per_pool():
    red, _ = _reducer_with_two_rolls()
    names = {p.name for p in red.reference_params()}
    assert names == {"nkeep_altroll", "nkeep_psfref"}, names


def test_reducer_honours_the_counts_and_records_them():
    from klip_tpe.reducer import ReductionRequest
    red, _ = _reducer_with_two_rolls()
    a = red.reduce(ReductionRequest(_p(nkeep_altroll=8, nkeep_psfref=10), None))
    b = red.reduce(ReductionRequest(_p(nkeep_altroll=1, nkeep_psfref=1), None))
    assert a.meta["rdi_mode"] == "searched"
    assert a.meta["ref_keep"] == {"altroll": 8, "psfref": 10}
    assert b.meta["ref_keep"] == {"altroll": 1, "psfref": 1}
    assert (a.meta["nref_used"] == 18).all(), a.meta["nref_used"]
    assert (b.meta["nref_used"] == 2).all(), b.meta["nref_used"]
    assert not np.allclose(np.nan_to_num(a.image), np.nan_to_num(b.image))


def test_dropping_the_science_pool_is_pure_rdi():
    """nkeep_altroll = 0 leaves only the reference star, which is RDI -- so the optimizer
    can elect the subtraction mode instead of being told it."""
    from klip_tpe.reducer import ReductionRequest
    red, _ = _reducer_with_two_rolls()
    r = red.reduce(ReductionRequest(_p(nkeep_altroll=0, nkeep_psfref=6), None))
    assert (r.meta["nref_used"] == 6).all(), r.meta["nref_used"]


def test_the_similarity_matrix_is_built_once_per_distinct_geometry():
    from klip_tpe.reducer import ReductionRequest
    red, _ = _reducer_with_two_rolls()
    for keep in (4, 6, 8, 10):
        red.reduce(ReductionRequest(_p(nkeep_altroll=keep, nkeep_psfref=keep), None))
    assert len(red._reflib_cache) == 1, "the counts must not invalidate the cache"
    red.reduce(ReductionRequest(_p(nkeep_altroll=4, nkeep_psfref=4, inrad=6), None))
    assert len(red._reflib_cache) == 2, "a different annulus is a different matrix"


def test_a_bin_that_straddles_two_rolls_is_refused():
    """Binning across the roll boundary would give the bin one label while its references
    came from both rolls -- the exclusion would then be silently wrong rather than loud.
    Driven at the mapping directly: whether the PA guard lets a bin get that wide depends
    on the annulus, and the check has to hold whenever it does."""
    red, part = _reducer_with_two_rolls(nframes=8)
    keep = np.ones(8, bool)
    ok = red._binned_partition(keep, np.repeat(np.arange(4), 2), np.ones(4, bool))
    assert list(ok) == ["rollA", "rollA", "rollB", "rollB"], ok
    # bins of three over four-and-four: the middle bin takes one rollA frame and two
    # rollB ones, so its single label would be a lie
    with pytest.raises(ValueError, match="mix partitions"):
        red._binned_partition(keep, np.array([0, 0, 0, 1, 1, 1, 2, 2]), np.ones(3, bool))


def test_frame_selection_and_binning_compose_into_the_right_labels():
    """The three maskings -- frame selection, bin grouping, empty-bin drop -- compose in
    that order, and a mislabelled roll is exactly the bug that would go unnoticed."""
    red, _ = _reducer_with_two_rolls(nframes=8)
    keep = np.array([1, 1, 0, 1, 1, 1, 1, 1], bool)        # one rollA frame cut
    # 7 selected frames (A A A B B B B) -> three bins, none straddling the boundary
    grp = np.array([0, 0, 0, 1, 2, 2, 2])
    lab = red._binned_partition(keep, grp, np.array([True, False, True]))
    assert list(lab) == ["rollA", "rollB"], lab            # middle bin dropped as empty


# ---------------------------------------------------------------- space + guard
def test_the_counts_reach_the_search_space_and_the_guard_clamps_k():
    from klip_tpe.feasibility import ReferenceLibraryGuard
    from klip_tpe.instruments import generic
    from klip_tpe.reducer import PartitionedReducer
    red, _ = _reducer_with_two_rolls()
    pr = PartitionedReducer({"sci": red})
    space = generic.make_space(pr, k_klip_max=40)
    names = {p.name for p in space.params}
    assert {"nkeep_altroll", "nkeep_psfref"} <= names, sorted(names)

    guard = generic.make_guard(pr, n_min_ref=2, k_max=40)
    idx = {p.name: i for i, p in enumerate(space.params)}
    x = np.array([p.default if p.default is not None else p.lo for p in space.params], float)
    x[idx["nkeep_altroll"]], x[idx["nkeep_psfref"]] = 2.0, 3.0
    kx = [i for i, p in enumerate(space.params) if p.base == "k_klip" or p.name == "k_klip"]
    for i in kx:
        x[i] = 40.0
    y = guard(x, space, is_random=False)
    for i in kx:
        assert y[i] <= 5.0, (y[i], "k_klip must not exceed the frames actually retained")

    # and a zero-everywhere draw is rescued rather than evaluated
    z = x.copy()
    z[idx["nkeep_altroll"]] = z[idx["nkeep_psfref"]] = 0.0
    w = guard(z, space, is_random=True)
    assert w[idx["nkeep_altroll"]] + w[idx["nkeep_psfref"]] >= 2
