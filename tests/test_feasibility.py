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
    # anglemax=0.5 deg: only immediate neighbours (<=2) -> 0 for nminref=3, 1 for nminref=1 (except edges)
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
