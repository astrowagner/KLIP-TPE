from collections import Counter

import numpy as np
import pytest
from scipy.integrate import quad

from klip_tpe.optimizers import (GridSearch, History, RandomSearch, TPE, block_log_density,
                                 make_optimizer, parzen_bandwidth, parzen_density, resolve_blocks,
                                 tpe_propose)
from klip_tpe.space import Param, SearchSpace, kgrid


# ----------------------------------------------------------------------------
# Parzen helpers
# ----------------------------------------------------------------------------
def test_parzen_bandwidth_scott_and_floor():
    s = np.array([0.1, 0.4, 0.5, 0.9])
    h = parzen_bandwidth(s, 0.0, 1.0)
    assert h == pytest.approx(1.06 * np.std(s, ddof=1) * 4 ** -0.2)
    # tight good set -> floor
    assert parzen_bandwidth(np.array([0.5, 0.5001, 0.5002]), 0.0, 1.0) == pytest.approx(0.08)
    assert parzen_bandwidth(np.array([0.5, 0.5001]), 0.0, 10.0, floor_frac=0.1) == pytest.approx(1.0)
    # fewer than two samples -> half range
    assert parzen_bandwidth(np.array([0.3]), 0.0, 2.0) == 1.0
    assert parzen_bandwidth(np.array([]), 0.0, 2.0) == 1.0


def test_parzen_density_normalises_with_prior():
    s = np.array([0.45, 0.5, 0.55, 0.6])
    h = 0.05
    for wp in (0.0, 0.25, 1.0):
        area = quad(lambda x: parzen_density(x, s, h, 0.0, 1.0, prior_weight=wp), 0.0, 1.0)[0]
        assert area == pytest.approx(1.0, abs=0.01)
    # far from the samples the density is exactly the prior share
    assert parzen_density(0.0, s, h, 0.0, 1.0, prior_weight=0.25) == pytest.approx(0.25)
    # empty sample -> uniform
    assert parzen_density(0.3, np.array([]), h, 0.0, 4.0) == pytest.approx(0.25)


def test_block_log_density_single_dim_matches_univariate(rng):
    X = rng.uniform(0, 1, (12, 3))
    C = rng.uniform(0, 1, (7, 3))
    h = np.array([0.1, 0.2, 0.3])
    lo, hi = np.zeros(3), np.ones(3)
    for d in range(3):
        bl = block_log_density(C, X, np.arange(12), h, [d], lo, hi, prior_weight=0.25)
        uv = np.log([parzen_density(C[c, d], X[:, d], h[d], 0.0, 1.0, 0.25) for c in range(7)])
        np.testing.assert_allclose(bl, uv, rtol=1e-10, atol=1e-12)


def test_block_log_density_finite_when_kernels_underflow(rng):
    # 30-dim block, candidates far from every observation: kernel product underflows,
    # log-space prior mixing must still give a finite value
    X = rng.uniform(0, 0.1, (10, 30))
    C = np.full((3, 30), 0.99)
    h = np.full(30, 0.01)
    v = block_log_density(C, X, np.arange(10), h, np.arange(30), np.zeros(30), np.ones(30))
    assert np.all(np.isfinite(v))
    assert v == pytest.approx(np.log(0.25))     # prior only: log(wp) + log(1/vol) with vol=1


def test_block_density_subset_selects_rows(rng):
    X = rng.uniform(0, 1, (10, 2))
    C = rng.uniform(0, 1, (4, 2))
    h = np.array([0.1, 0.1])
    a = block_log_density(C, X, [0, 1, 2], h, [0, 1], np.zeros(2), np.ones(2))
    b = block_log_density(C, X[:3], [0, 1, 2], h, [0, 1], np.zeros(2), np.ones(2))
    np.testing.assert_allclose(a, b)


# ----------------------------------------------------------------------------
# resolve_blocks
# ----------------------------------------------------------------------------
def test_resolve_blocks_modes(partition_space, simple_space):
    sp = partition_space
    uni = resolve_blocks(sp, None)
    assert [b.tolist() for b in uni] == [[i] for i in range(sp.ndim)]
    assert [b.tolist() for b in resolve_blocks(sp, "univariate")] == [b.tolist() for b in uni]
    full = resolve_blocks(sp, "full")
    assert len(full) == 1 and full[0].tolist() == list(range(sp.ndim))
    parts = resolve_blocks(sp, "partitions")
    assert len(parts) == 4
    for pid, b in zip(sp.partitions, parts):
        assert b.tolist() == sp.dims_of_partition(pid).tolist()
    assert parts[-1].tolist() == [sp.index("filter"), sp.index("drop1"), sp.index("drop2")]
    exp = resolve_blocks(sp, [["bin_p0", "bin_p1"], [sp.index("angsep_p2")]])
    assert exp[0].tolist() == [sp.index("bin_p0"), sp.index("bin_p1")]
    assert exp[1].tolist() == [sp.index("angsep_p2")]
    covered = sorted(int(i) for b in exp for i in b)
    assert covered == list(range(sp.ndim))            # leftovers become singletons
    # categorical dims are never part of a continuous block
    cat = simple_space.index("mode")
    assert all(cat not in b for b in resolve_blocks(simple_space, "full"))


def test_resolve_blocks_partitions_without_partitions_is_univariate(simple_space):
    blk = resolve_blocks(simple_space, "partitions")
    assert all(len(b) == 1 for b in blk)


# ----------------------------------------------------------------------------
# tpe_propose
# ----------------------------------------------------------------------------
def _quadratic_history(n=60, seed=0):
    sp = SearchSpace([Param("a", -1, 1), Param("b", -1, 1)])
    r = np.random.default_rng(seed)
    X = r.uniform(-1, 1, (n, 2))
    y = -((X[:, 0] - 0.3) ** 2 + (X[:, 1] + 0.2) ** 2)
    return sp, X, y


@pytest.mark.parametrize("blocks", [None, "full"])
def test_tpe_propose_near_optimum_more_than_random(blocks):
    sp, X, y = _quadratic_history()
    opt = np.array([0.3, -0.2])
    rng = np.random.default_rng(1)
    P = np.array([tpe_propose(X, y, sp, rng, blocks=blocks) for _ in range(60)])
    R = rng.uniform(-1, 1, (60, 2))
    near_tpe = (np.hypot(*(P - opt).T) < 0.3).mean()
    near_rnd = (np.hypot(*(R - opt).T) < 0.3).mean()
    assert near_tpe > 0.7
    assert near_tpe > near_rnd + 0.3
    assert np.all(P >= -1) and np.all(P <= 1)


def test_tpe_propose_sanitized_in_bounds(simple_space, rng):
    sp = simple_space
    X = np.array([sp.random(rng) for _ in range(30)])
    y = rng.standard_normal(30)
    for _ in range(20):
        x = tpe_propose(X, y, sp, rng, pbest=0.5)
        assert x.shape == (sp.ndim,)
        np.testing.assert_array_equal(x, sp.sanitize(x))
        assert np.all(x >= sp.lo) and np.all(x <= sp.hi)
        assert x[sp.index("k")] in set(kgrid(30).tolist())
        assert float(x[sp.index("mode")]).is_integer()


def test_tpe_propose_tiny_history_falls_back_to_random(simple_space, rng):
    sp = simple_space
    x = tpe_propose(np.zeros((1, sp.ndim)), np.zeros(1), sp, rng)
    assert x.shape == (sp.ndim,)
    # two points: one good, one bad, still works
    X = np.array([sp.random(rng) for _ in range(2)])
    x = tpe_propose(X, np.array([1.0, 0.0]), sp, rng)
    assert np.all(x >= sp.lo) and np.all(x <= sp.hi)


def test_tpe_propose_categorical_prefers_good_class(rng):
    sp = SearchSpace([Param("m", 0, 0, "categorical", choices=["a", "b", "c"]), Param("f", 0, 1)])
    n = 60
    X = np.column_stack([rng.integers(0, 3, n).astype(float), rng.uniform(0, 1, n)])
    y = np.where(X[:, 0] == 2, 1.0, 0.0) + 0.01 * rng.standard_normal(n)
    picks = [tpe_propose(X, y, sp, rng)[0] for _ in range(40)]
    assert Counter(picks)[2.0] >= 30


# ----------------------------------------------------------------------------
# History
# ----------------------------------------------------------------------------
def test_history_best_order_and_round_trip():
    h = History(2)
    assert h.best() == (-1, -np.inf) and h.order().size == 0
    h.append([0, 0], 1.0, {"phase": "seed"})
    h.append([1, 0], None)                   # failed
    h.append([0, 1], 3.0, {"phase": "tpe"})
    h.append([1, 1], 3.0)                    # tie: earlier wins
    h.append([2, 2], np.nan)
    assert len(h) == 5
    assert h.valid.tolist() == [True, False, True, True, False]
    assert h.best() == (2, 3.0)
    assert h.order().tolist() == [2, 3, 0]
    h.set_score(1, 5.0, k=7)
    assert h.best() == (1, 5.0) and h.flags[1]["k"] == 7
    d = h.to_dict()
    assert d["y"][4] is None
    h2 = History.from_dict(d)
    assert len(h2) == 5 and np.isnan(h2.y[4]) and h2.flags == h.flags
    np.testing.assert_array_equal(h2.X, h.X)
    assert h2.order().tolist() == h.order().tolist()


def test_history_from_dict_empty():
    h = History.from_dict(History(3).to_dict())
    assert len(h) == 0 and h.X.shape == (0, 3)


# ----------------------------------------------------------------------------
# TPE.ask
# ----------------------------------------------------------------------------
def test_tpe_ask_phases(simple_space):
    sp = simple_space
    opt = TPE(sp, n_init=5, p_local=0.15, explore_frac=0.15)
    h = History(sp.ndim)
    rng = np.random.default_rng(3)
    phases = []
    for _ in range(150):
        pr = opt.ask(h, rng)[0]
        phases.append(pr.flags["phase"])
        x = pr.x
        np.testing.assert_array_equal(x, sp.sanitize(x))
        h.append(x, -(x[0] - 0.3) ** 2 - (x[1] - 4) ** 2, pr.flags)
    assert phases[:5] == ["warmup"] * 5
    assert "warmup" not in phases[5:]
    c = Counter(phases[5:])
    assert c["tpe"] > c["local"] > 0 and c["explore"] > 0
    assert set(c) <= {"tpe", "local", "explore"}
    # the guided phase should have concentrated near the optimum
    tail = h.X[-40:]
    assert np.median(np.abs(tail[:, 0] - 0.3)) < 0.3


def test_tpe_batch_ask_count_and_warmup_accounting(simple_space, rng):
    sp = simple_space
    opt = TPE(sp, n_init=5)
    h = History(sp.ndim)
    props = opt.ask(h, rng, 7)
    assert len(props) == 7
    assert [p.flags["phase"] for p in props[:5]] == ["warmup"] * 5
    assert all(p.flags["phase"] != "warmup" for p in props[5:])
    assert len(h) == 0                                   # the real history is untouched
    for p in props:
        h.append(p.x, rng.standard_normal(), p.flags)
    props = opt.ask(h, rng, 3)
    assert len(props) == 3 and all(p.flags["phase"] in ("tpe", "local", "explore") for p in props)


def test_tpe_link_params_and_warmstart_tie(partition_space, rng):
    sp = partition_space
    opt = TPE(sp, n_init=4, warmstart_tie=True, link_params=["bin"], blocks="partitions")
    h = History(sp.ndim)
    for _ in range(30):
        pr = opt.ask(h, rng)[0]
        x = pr.x
        assert x[sp.index("bin_p0")] == x[sp.index("bin_p1")] == x[sp.index("bin_p2")]
        if pr.flags["phase"] == "warmup":
            assert x[sp.index("angsep_p0")] == x[sp.index("angsep_p2")]
        h.append(x, rng.standard_normal(), pr.flags)
    d = opt.describe()
    assert d["link_params"] == ["bin"] and len(d["resolved_blocks"]) == 4


def test_tpe_ask_with_only_failed_history(simple_space, rng):
    sp = simple_space
    opt = TPE(sp, n_init=2)
    h = History(sp.ndim)
    for _ in range(4):
        h.append(sp.random(rng), np.nan)
    pr = opt.ask(h, rng)[0]
    assert pr.flags["phase"] in ("explore", "local")
    assert np.all(pr.x >= sp.lo) and np.all(pr.x <= sp.hi)


# ----------------------------------------------------------------------------
# Random / Grid
# ----------------------------------------------------------------------------
def test_random_search(simple_space, rng):
    opt = RandomSearch(simple_space)
    props = opt.ask(History(simple_space.ndim), rng, 5)
    assert len(props) == 5 and all(p.flags["phase"] == "random" for p in props)
    assert opt.n_init == 40  # default from the base class; the runner passes 0


def test_grid_search_cells_cover_axes():
    sp = SearchSpace([Param("a", -1, 1), Param("b", 0, 10, "int"), Param("k", 1, 30, "int", grid=kgrid(30))])
    g = GridSearch(sp, budget=27, axes=["a", "b"])
    assert g.gpts == 5
    cells = np.array([g.cell(k) for k in range(25)])
    assert sorted(set(cells[:, 0])) == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert sorted(set(cells[:, 1])) == [0.0, 2.5, 5.0, 7.5, 10.0]
    assert len({tuple(c[:2]) for c in cells}) == 25            # every cell distinct
    assert np.all(cells[:, 2] == sp["k"].default)               # unsearched axis at default
    np.testing.assert_array_equal(g.cell(25), g.cell(0))         # wraps
    h = History(sp.ndim)
    rng = np.random.default_rng(0)
    p1 = g.ask(h, rng, 2)
    assert [p.flags["cell"] for p in p1] == [0, 1]
    for p in p1:
        h.append(p.x, 0.0, p.flags)
    assert g.ask(h, rng)[0].flags["cell"] == 2


def test_grid_search_grid_axis_uses_grid_entries():
    sp = SearchSpace([Param("k", 1, 30, "int", grid=kgrid(30))])
    g = GridSearch(sp, budget=4)
    vals = sorted(g.cell(i)[0] for i in range(g.gpts))
    assert vals[0] == 1.0 and vals[-1] == 30.0
    assert set(vals) <= set(kgrid(30).tolist())


def test_make_optimizer(simple_space):
    assert isinstance(make_optimizer("tpe", simple_space, n_init=3), TPE)
    assert isinstance(make_optimizer("random", simple_space), RandomSearch)
    assert isinstance(make_optimizer("grid", simple_space, budget=8), GridSearch)
    with pytest.raises(ValueError):
        make_optimizer("annealing", simple_space)
