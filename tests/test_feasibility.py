import numpy as np
import pytest

from klip_tpe.feasibility import (ReferenceCountGuard, compose, max_feasible_angsep, min_feasible_anglemax,
                                  ref_fraction)
from klip_tpe.klip import arcdist_deg
from klip_tpe.space import Param, SearchSpace

ANG = np.linspace(-20, 20, 100)          # 0.404 deg per frame
ARC = arcdist_deg(10, 30, 4.0)           # ~11.46 deg per lambda/D of arc at r=20


def test_ref_fraction_synthetic_angles():
    # no window: every frame has 99 refs
    assert ref_fraction(ANG, 0.0, 360.0, ARC, 10) == 1.0
    # nothing qualifies when angsep*arc exceeds the total span
    assert ref_fraction(ANG, 4.0, 360.0, ARC, 10) == 0.0
    # anglemax=0.5 deg: only immediate neighbors (<=2) -> 0 for nminref=3, 1 for nminref=1 (except edges)
    assert ref_fraction(ANG, 0.0, 0.5, ARC, 3) == 0.0
    assert ref_fraction(ANG, 0.0, 0.5, ARC, 1) == 1.0
    # hand count: window 2 deg = 4 frames each side (dpa 0.404..1.616) -> 8 refs in the interior
    f8 = ref_fraction(ANG, 0.0, 2.0, ARC, 8)
    f9 = ref_fraction(ANG, 0.0, 2.0, ARC, 9)
    assert 0.9 < f8 < 1.0 and f9 == 0.0
    # monotone in angsep and in nminref
    fr = [ref_fraction(ANG, a, 360.0, ARC, 10) for a in np.linspace(0, 3.5, 15)]
    assert all(a >= b for a, b in zip(fr, fr[1:]))
    assert ref_fraction(ANG, 1.0, 360.0, ARC, 5) >= ref_fraction(ANG, 1.0, 360.0, ARC, 50)
    # degenerate inputs
    assert ref_fraction(np.array([3.0]), 0.0, 360.0, ARC, 1) == 0.0
    assert ref_fraction(np.array([]), 0.0, 360.0, ARC, 1) == 0.0


def test_max_feasible_angsep_monotone_and_bounds():
    prev = -1.0
    for am in (5.0, 10.0, 20.0, 40.0, 360.0):
        a = max_feasible_angsep(ANG, am, ARC, 10, 0.95, 0.0, 5.0)
        assert a >= prev
        assert ref_fraction(ANG, a, am, ARC, 10) >= 0.95
        assert ref_fraction(ANG, a + 0.05, am, ARC, 10) < 0.95 or a == 5.0
        prev = a
    # ashi feasible -> returns ashi; aslo infeasible -> returns aslo
    assert max_feasible_angsep(ANG, 360.0, ARC, 10, 0.95, 0.0, 1.0) == 1.0
    assert max_feasible_angsep(ANG, 0.5, ARC, 10, 0.95, 0.0, 5.0) == 0.0


def test_min_feasible_anglemax():
    am = min_feasible_anglemax(ANG, 0.0, ARC, 10, 0.95, 1.0, 360.0)
    assert ref_fraction(ANG, 0.0, am, ARC, 10) >= 0.95
    assert ref_fraction(ANG, 0.0, am - 0.2, ARC, 10) < 0.95
    assert 2.0 < am < 5.0
    # amlo already feasible -> amlo ; amhi infeasible -> amhi
    assert min_feasible_anglemax(ANG, 0.0, ARC, 10, 0.95, 100.0, 360.0) == 100.0
    assert min_feasible_anglemax(ANG, 5.0, ARC, 10, 0.95, 1.0, 360.0) == 360.0


@pytest.fixture
def global_space():
    return SearchSpace([Param("angsep", 0, 5, "float", default=0.5),
                        Param("anglemax", 1, 360, "int", default=360),
                        Param("k_klip", 1, 20, "int", default=5),
                        Param("bin", 1, 10, "int", default=1)])


@pytest.fixture
def guard():
    return ReferenceCountGuard(lambda pid: ANG, fwhm_px=4.0, lam_over_d_px=4.0, n_min_ref=10, ref_frac=0.95)


def test_guard_projects_infeasible_pair_and_keeps_feasible(global_space, guard):
    sp = global_space
    # infeasible angsep for the given anglemax -> snapped down to the boundary
    x = sp.encode({"angsep": 4.5, "anglemax": 20})
    xp = guard(x, sp, zone=(10, 30))
    asf = max_feasible_angsep(ANG, 20.0, ARC, 10, 0.95, 0.0, 4.5)
    assert xp[0] == pytest.approx(asf) and xp[0] < 4.5
    assert xp[1] == 20 and xp[2] == 5 and xp[3] == 1
    assert ref_fraction(ANG, xp[0], xp[1], ARC, 10) >= 0.95
    # feasible pair untouched
    x2 = sp.encode({"angsep": 0.2, "anglemax": 360})
    np.testing.assert_array_equal(guard(x2, sp, zone=(10, 30)), x2)
    # anglemax too small even at angsep=0 -> angsep forced to 0, anglemax raised
    x3 = sp.encode({"angsep": 1.0, "anglemax": 2})
    xp3 = guard(x3, sp, zone=(10, 30))
    assert xp3[0] == 0.0 and xp3[1] > 2.0
    assert ref_fraction(ANG, 0.0, xp3[1], ARC, 10) >= 0.95
    # the input is not modified in place
    assert x3[1] == 2.0
    assert guard.describe()["n_min_ref"] == 10


def test_guard_random_redraw_inside_feasible_range(global_space, guard):
    sp = global_space
    x = sp.encode({"angsep": 4.5, "anglemax": 20})
    asf = max_feasible_angsep(ANG, 20.0, ARC, 10, 0.95, 0.0, 4.5)
    rng = np.random.default_rng(0)
    draws = np.array([guard(x, sp, rng=rng, is_random=True, zone=(10, 30))[0] for _ in range(40)])
    assert np.all(draws >= 0) and np.all(draws <= asf)
    assert draws.std() > 0.1 * asf                   # genuinely random, not snapped
    # guided (is_random False) always snaps even when an rng is passed
    guided = [guard(x, sp, rng=rng, is_random=False, zone=(10, 30))[0] for _ in range(5)]
    assert all(g == pytest.approx(asf) for g in guided)
    # random_redraw disabled -> snap
    g2 = ReferenceCountGuard(lambda pid: ANG, fwhm_px=4.0, lam_over_d_px=4.0, random_redraw=False)
    assert g2(x, sp, rng=rng, is_random=True, zone=(10, 30))[0] == pytest.approx(asf)


def test_guard_uses_zone_and_space_inrad_outrad(guard):
    sp = SearchSpace([Param("angsep", 0, 5), Param("anglemax", 1, 360, "int"),
                      Param("inrad", 0, 50), Param("outrad", 1, 70)])
    x = sp.encode({"angsep": 4.0, "anglemax": 360, "inrad": 10, "outrad": 30})
    a_zone = guard(x, sp, zone=(10, 30))[0]
    a_space = guard(x, sp)[0]
    assert a_zone == pytest.approx(a_space)
    # a wider annulus (larger mid radius) has a smaller arcdist, so more angsep is feasible
    a_far = guard(x, sp, zone=(40, 60))[0]
    assert a_far > a_zone
    # spaces without angsep/anglemax pass through untouched
    sp2 = SearchSpace([Param("k", 1, 5, "int")])
    x2 = sp2.default_vector()
    assert guard(x2, sp2) is x2


def test_guard_per_partition_and_binning():
    parts = ["a", "b"]
    angles = {"a": ANG, "b": np.linspace(-5, 5, 100)}       # b has 4x less rotation
    block = [Param("angsep", 0, 5, default=3.0), Param("anglemax", 1, 360, "int", default=360),
             Param("bin", 1, 20, "int", default=1), Param("k_klip", 1, 20, "int", default=5)]
    sp = SearchSpace().replicate(block, parts).with_selection("two_slot", partitions=parts)
    sp.add(Param("angsep", 0, 5, default=3.0))              # a global dim on top -> min over selected
    sp.add(Param("anglemax", 1, 360, "int", default=360))
    calls = []

    def angles_fn(pid):
        calls.append(pid)
        return angles[pid]

    g = ReferenceCountGuard(angles_fn, fwhm_px=4.0, lam_over_d_px=4.0)
    x = sp.default_vector()
    xp = g(x, sp, zone=(10, 30))
    a_a, a_b = xp[sp.index("angsep_a")], xp[sp.index("angsep_b")]
    assert a_a > a_b > 0                                   # less rotation -> tighter feasible angsep
    assert xp[sp.index("angsep")] == pytest.approx(min(a_a, a_b))
    assert xp[sp.index("anglemax")] == 360
    # angle census is cached per partition
    g(x, sp, zone=(10, 30))
    assert sorted(set(calls)) == parts and len(calls) == 2
    # binning reduces the frame count -> fewer references -> smaller feasible angsep
    xb = sp.default_vector({"bin": 8})
    xpb = g(xb, sp, zone=(10, 30))
    assert xpb[sp.index("angsep_a")] <= a_a
    ba = g.binned_angles("a", {"bin": 8, "k_klip": 5}, 30.0)
    assert 10 <= ba.size <= 13
    # tags with a corr threshold reduce the census
    g2 = ReferenceCountGuard(lambda pid: ANG, tags_fn=lambda pid: {"corrs": np.linspace(0, 1, 100)},
                             fwhm_px=4.0, lam_over_d_px=4.0)
    assert g2.binned_angles("a", {"bin": 1, "k_klip": 5, "corr_thresh": 0.5}, 30.0).size == 50


def test_compose_chains_projections():
    sp = SearchSpace([Param("a", 0, 10), Param("b", 0, 10)])
    p1 = lambda x, space, **kw: np.minimum(x, 5)
    p2 = lambda x, space, **kw: x + 1
    f = compose(p1, None, p2)
    np.testing.assert_array_equal(f(np.array([9.0, 2.0]), sp), [6.0, 3.0])


# --------------------------------------------------------------- frame-selection snapping
def _fs_space(replicate, parts):
    """bin, k and the three frame-selection thresholds; replicated per partition or shared.
    A shared space still names its partitions, as generic.make_space does for one sequence."""
    block = [Param("bin", 1, 4, "int", default=1), Param("corr_thresh", 0.0, 1.0, "float", default=0.9),
             Param("noise_max", 0.3, 3.0, "float", default=1.5),
             Param("coronoise_max", 0.3, 3.0, "float", default=1.5), Param("k_klip", 1, 10, "int", default=3)]
    sp = SearchSpace()
    if replicate:
        sp.replicate(block, parts, name_fmt="{base}_{pid}")
    else:
        for b in block:
            sp.add(b)
        sp.partitions = list(parts)
    return sp


def _guard(nf=20):
    from klip_tpe.feasibility import FrameSelectionGuard
    rng = np.random.default_rng(0)
    tags = {"corrs": np.full(nf, 0.99), "noises": 1.0 + 0.05 * rng.standard_normal(nf),
            "coronoise": 1.0 + 0.05 * rng.standard_normal(nf)}
    return FrameSelectionGuard(tags_fn=lambda pid: tags, nframes_fn=lambda pid: nf)


def test_frame_selection_guard_snaps_a_shared_space():
    """A single sequence: ``space.partitions == ['seq']`` but its thresholds carry
    ``partition=None``.  noise_max = 0.3 keeps no frame, so the reducer refuses the cut and the
    triplet must be recorded at the no-cut corner.  Until 2026-09-26 nothing was snapped here
    (the single-sequence beta Pic run logged 0 of 3030 evaluations at the corner)."""
    sp = _fs_space(False, ["seq"])
    g = _guard()
    x = np.array([1, 0.5, 0.3, 1.0, 3], float)          # bin, corr, noise, coro, k
    y = g(x.copy(), sp)
    assert g.last_snapped == 1
    assert y[1] == 0.0 and y[2] == 3.0 and y[3] == 3.0    # the no-cut corner
    assert y[0] == 1 and y[4] == 3                       # nothing else moves
    x2 = np.array([1, 0.5, 1.02, 3.0, 3], float)         # drops the noisier frames: it acts
    y2 = g(x2.copy(), sp)
    assert g.last_snapped == 0 and np.array_equal(y2, x2)


def test_frame_selection_guard_snaps_only_the_partition_whose_cut_did_not_act():
    sp = _fs_space(True, ["g1", "g2"])
    g = _guard()
    names = list(sp.names)
    x = np.zeros(len(names))
    for n, v in (("bin_g1", 1), ("k_klip_g1", 3), ("bin_g2", 1), ("k_klip_g2", 3),
                 ("corr_thresh_g1", 0.5), ("noise_max_g1", 0.3), ("coronoise_max_g1", 1.0),
                 ("corr_thresh_g2", 0.5), ("noise_max_g2", 1.02), ("coronoise_max_g2", 3.0)):
        x[names.index(n)] = v
    y = g(x.copy(), sp)
    assert g.last_snapped == 1
    assert y[names.index("noise_max_g1")] == 3.0 and y[names.index("corr_thresh_g1")] == 0.0
    assert y[names.index("noise_max_g2")] == 1.02 and y[names.index("corr_thresh_g2")] == 0.5


def test_guard_projects_angsep_when_the_space_pins_anglemax():
    """make_space pins anglemax at 360 for a PA span of 20 deg or less (a user's note of 2026-11-22)."""
    small = np.linspace(-6.33, 6.33, 189)                    # 12.66 deg of rotation
    arc = arcdist_deg(22, 52, 4.0)
    for parts in (None, ["n1"]):
        if parts is None:
            sp = SearchSpace([Param("angsep", 0, 3, default=3.0), Param("bin", 1, 10, "int", default=1),
                              Param("k_klip", 1, 20, "int", default=5)], fixed={"anglemax": 360.0})
        else:
            block = [Param("angsep", 0, 3, default=3.0), Param("bin", 1, 10, "int", default=1),
                     Param("k_klip", 1, 20, "int", default=5)]
            sp = SearchSpace(fixed={"anglemax": 360.0}).replicate(block, parts)
        assert "anglemax" not in sp.bases
        g = ReferenceCountGuard(lambda pid: small, fwhm_px=4.0, lam_over_d_px=4.0, n_min_ref=10, ref_frac=0.95)
        x = sp.default_vector()
        xp = g(x, sp, zone=(22, 52))
        a = float(xp[sp.dims_of_base("angsep")[0]])
        asf = max_feasible_angsep(small, 360.0, arc, 10, 0.95, 0.0, 3.0)
        assert a == pytest.approx(asf) and 0 < a < 3.0
        assert ref_fraction(small, a, 360.0, arc, 10) >= 0.95
        assert ref_fraction(small, 3.0, 360.0, arc, 10) == 0.0     # where it was: no target kept a reference
    # a space that neither searches nor pins anglemax passes through
    sp2 = SearchSpace([Param("angsep", 0, 3, default=3.0)])
    x2 = sp2.default_vector()
    assert g(x2, sp2, zone=(22, 52)) is x2


def test_make_space_s_guard_projects_angsep_on_a_sequence_of_little_rotation():
    from klip_tpe.instruments import generic
    from klip_tpe.instruments.search_space import make_space
    from klip_tpe.synthetic import synthetic_klip_dataset

    from klip_tpe.injection import GaussianPSF

    ds = synthetic_klip_dataset(nframes=60, size=64, pa_span=12.66, seed=2)
    model = GaussianPSF(4.0, star_flux=1.0)
    red = generic.make_reducer({"n1": ds}, pxscale=0.05, lam_m=3.8e-6, diam_m=8.4, injection_model=model,
                               log=lambda *a: None)
    r0 = next(iter(red.reducers.values()))
    assert r0.model is model                               # the model given, not one built from a template
    space = make_space(red, k_klip_max=10, search_angles=True)
    assert "anglemax" in space.fixed and "anglemax" not in space.bases
    guard = generic.make_guard(red, k_max=10)
    x = space.default_vector({"angsep": 3.0, "bin": 5})          # 12 binned frames, 11 references each
    xp = guard(x, space, zone=(8, 20))
    a = float(xp[space.dims_of_base("angsep")[0]])
    arc = arcdist_deg(8, 20, r0.lam_over_d_px)
    assert ref_fraction(np.sort(ds.angles), 3.0, 360.0, arc, 10) == 0.0
    assert 0 < a < 3.0
