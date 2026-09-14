import numpy as np
import pytest

from klip_tpe.space import Config, Param, SearchSpace, kgrid


# ----------------------------------------------------------------------------
# kgrid
# ----------------------------------------------------------------------------
def test_kgrid_100_layout():
    expected = list(range(1, 21)) + list(range(25, 51, 5)) + list(range(60, 101, 10))
    assert kgrid(100).tolist() == [float(v) for v in expected]


def test_kgrid_small_and_mid():
    assert kgrid(7).tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert kgrid(30).tolist() == [float(v) for v in list(range(1, 21)) + [25, 30]]
    assert kgrid(0).tolist() == [1.0]           # floor at 1


# ----------------------------------------------------------------------------
# Param kinds
# ----------------------------------------------------------------------------
def test_param_float_sanitize_clips():
    p = Param("f", -1.0, 2.0)
    assert p.sanitize(-5) == -1.0
    assert p.sanitize(7) == 2.0
    assert p.sanitize(0.3) == pytest.approx(0.3)
    assert p.decode(0.3) == pytest.approx(0.3)
    assert isinstance(p.decode(0.3), float)
    assert p.default == pytest.approx(0.5)


def test_param_int_rounds_and_default():
    p = Param("i", 1, 10, "int")
    assert p.decode(3.4) == 3 and isinstance(p.decode(3.4), int)
    assert p.decode(3.6) == 4
    assert p.decode(99) == 10
    assert p.default == 6.0  # round(5.5) -> 6 under banker's rounding of python? round(5.5)=6
    assert p.is_int


def test_param_grid_snapping():
    p = Param("k", 1, 100, "int", grid=kgrid(100))
    assert p.sanitize(22) == 20.0          # nearer to 20 than 25
    assert p.sanitize(23) == 25.0
    assert p.sanitize(57) == 60.0
    assert p.sanitize(1000) == 100.0
    assert p.decode(23) == 25 and isinstance(p.decode(23), int)


def test_param_grid_filtered_to_bounds_and_errors():
    p = Param("k", 5, 30, "int", grid=[1, 2, 5, 10, 30, 50])
    assert p.grid.tolist() == [5.0, 10.0, 30.0]
    with pytest.raises(ValueError):
        Param("k", 5, 6, "int", grid=[1, 2, 100])
    with pytest.raises(ValueError):
        Param("bad", 0, 1, "weird")
    with pytest.raises(ValueError):
        Param("hi_lt_lo", 1, 0)


def test_param_categorical_encode_decode():
    p = Param("c", 0, 0, "categorical", choices=["lin", "log", "sqrt"])
    assert (p.lo, p.hi) == (0.0, 2.0)
    assert p.default == "lin"
    for i, c in enumerate(["lin", "log", "sqrt"]):
        assert p.encode(c) == float(i)
        assert p.decode(float(i)) == c
    assert p.decode(1.4) == "log"
    assert p.decode(1.6) == "sqrt"
    assert p.encode(7) == 2.0                  # numeric encode is clipped
    with pytest.raises(ValueError):
        Param("c", 0, 1, "categorical")        # needs choices


def test_param_random_respects_grid_and_bounds(rng):
    p = Param("k", 1, 100, "int", grid=kgrid(100))
    draws = [p.random(rng) for _ in range(200)]
    assert set(draws) <= set(p.grid.tolist())
    q = Param("f", 2.0, 3.0)
    d = np.array([q.random(rng) for _ in range(200)])
    assert d.min() >= 2.0 and d.max() <= 3.0


# ----------------------------------------------------------------------------
# replicate / tie / selection
# ----------------------------------------------------------------------------
def test_replicate_partition_major_ordering(partition_space):
    names = partition_space.names
    assert names[:1] == ["filter"]
    assert names[1:4] == ["bin_p0", "k_klip_p0", "angsep_p0"]
    assert names[4:7] == ["bin_p1", "k_klip_p1", "angsep_p1"]
    assert names[-2:] == ["drop1", "drop2"]
    assert partition_space["bin_p1"].base == "bin"
    assert partition_space["bin_p1"].partition == "p1"
    assert partition_space["drop1"].role == "selection"
    assert set(partition_space.bases) == {"filter", "bin", "k_klip", "angsep"}
    with pytest.raises(ValueError):
        partition_space.add(Param("filter", 0, 1))


def test_tie_copies_first_slot(partition_space):
    sp = partition_space
    x = sp.default_vector()
    x[sp.index("bin_p0")] = 7
    x[sp.index("bin_p1")] = 3
    x[sp.index("bin_p2")] = 19
    x[sp.index("angsep_p1")] = 2.5
    xt = sp.tie(x, ["bin"])
    assert xt[sp.index("bin_p1")] == 7 and xt[sp.index("bin_p2")] == 7
    assert xt[sp.index("angsep_p1")] == 2.5             # untouched base
    xa = sp.tie(x)                                      # all replicated bases
    assert xa[sp.index("angsep_p1")] == xa[sp.index("angsep_p0")]
    assert xa[sp.index("angsep_p2")] == xa[sp.index("angsep_p0")]
    assert xa[sp.index("filter")] == x[sp.index("filter")]  # global dim not a tie target


def test_two_slot_positional_decode(partition_space):
    sp = partition_space
    x = sp.default_vector()
    assert sp.selected_partitions(x) == ["p0", "p1", "p2"]
    x[sp.index("drop1")] = 1                              # value c drops partition c-1
    assert sp.selected_partitions(x) == ["p1", "p2"]
    x[sp.index("drop2")] = 3
    assert sp.selected_partitions(x) == ["p1"]
    x[sp.index("drop1")] = 3                              # duplicate drop only removes one
    assert sp.selected_partitions(x) == ["p0", "p1"]
    x[sp.index("drop1")] = 2.4                            # rounding
    assert sp.selected_partitions(x) == ["p0"]


def test_two_slot_never_empty_rule():
    sp = SearchSpace().replicate([Param("bin", 1, 5, "int")], ["a", "b"]).with_selection("two_slot", partitions=["a", "b"])
    x = sp.default_vector()
    x[sp.index("drop1")] = 1
    x[sp.index("drop2")] = 2
    assert sp.selected_partitions(x) == ["a", "b"]
    assert sp.decode(x).selected == ["a", "b"]


def test_binary_selection_decode_and_never_empty():
    parts = ["a", "b", "c"]
    sp = SearchSpace().replicate([Param("bin", 1, 5, "int")], parts).with_selection("binary", partitions=parts)
    assert [p.name for p in sp.params if p.role == "selection"] == ["sel_a", "sel_b", "sel_c"]
    x = sp.default_vector()
    assert sp.selected_partitions(x) == parts
    x[sp.index("sel_b")] = 0
    assert sp.selected_partitions(x) == ["a", "c"]
    x[sp.index("sel_a")] = 0
    x[sp.index("sel_c")] = 0
    assert sp.selected_partitions(x) == parts             # all off -> all on


def test_selection_requires_two_partitions():
    sp = SearchSpace().replicate([Param("bin", 1, 5, "int")], ["only"]).with_selection("two_slot", partitions=["only"])
    assert sp.selection is None and sp.selection_dims.size == 0
    with pytest.raises(ValueError):
        SearchSpace().replicate([Param("bin", 1, 5, "int")], ["a", "b"]).with_selection("nonsense", partitions=["a", "b"])


def test_random_selection_bias(partition_space, rng):
    sp = partition_space
    sp.p_include = 0.75
    xs = np.array([sp.random(rng) for _ in range(600)])
    frac0 = (xs[:, sp.selection_dims] == 0).mean()
    assert 0.65 < frac0 < 0.85
    assert xs[:, sp.selection_dims].max() <= 3
    tied = np.array([sp.random(rng, tie_all=True) for _ in range(20)])
    assert np.all(tied[:, sp.index("bin_p0")] == tied[:, sp.index("bin_p2")])


# ----------------------------------------------------------------------------
# default_vector / encode / decode
# ----------------------------------------------------------------------------
def test_default_vector_and_overrides(partition_space):
    sp = partition_space
    x = sp.default_vector()
    cfg = sp.decode(x)
    assert cfg.params["filter"] == 10
    for pid in sp.partitions:
        assert cfg.per_partition[pid] == {"filter": 10, "bin": 4, "k_klip": 5, "angsep": 0.5}
    assert cfg.selected == sp.partitions
    x2 = sp.default_vector({"k_klip": 23, "bin_p1": 9, "drop1": 2})
    c2 = sp.decode(x2)
    assert all(c2.per_partition[p]["k_klip"] == 25 for p in sp.partitions)   # base override + grid snap
    assert c2.per_partition["p1"]["bin"] == 9 and c2.per_partition["p0"]["bin"] == 4
    assert c2.selected == ["p0", "p2"]


def test_decode_representative_globals_are_median(partition_space):
    sp = partition_space
    x = sp.default_vector()
    for pid, v in zip(sp.partitions, (2, 8, 20)):
        x[sp.index(f"bin_{pid}")] = v
    cfg = sp.decode(x)
    assert cfg.params["bin"] == 8 and isinstance(cfg.params["bin"], int)
    assert cfg.params_for("p2")["bin"] == 20
    assert cfg.params_for("unknown") == cfg.params
    d = cfg.to_dict()
    assert d["selected"] == ["p0", "p1", "p2"] and len(d["x"]) == sp.ndim


def test_encode_decode_round_trip(partition_space):
    sp = partition_space
    per = {"p0": {"bin": 3, "k_klip": 7, "angsep": 1.25},
           "p1": {"bin": 11, "k_klip": 25, "angsep": 0.0},
           "p2": {"bin": 20, "k_klip": 1, "angsep": 3.0}}
    x = sp.encode({"filter": 4}, per, selected=["p0", "p2"])
    cfg = sp.decode(x)
    assert cfg.params["filter"] == 4
    for pid in per:
        for k, v in per[pid].items():
            assert cfg.per_partition[pid][k] == pytest.approx(v)
    assert cfg.selected == ["p0", "p2"]
    assert x[sp.index("drop1")] == 2.0 and x[sp.index("drop2")] == 0.0
    # encode of a decode reproduces the sanitized vector
    np.testing.assert_allclose(sp.encode(cfg.params, cfg.per_partition, cfg.selected), x)


def test_encode_decode_simple_with_categorical(simple_space):
    sp = simple_space
    x = sp.encode({"a": 0.25, "b": 7.5, "k": 23, "mode": "z"})
    cfg = sp.decode(x)
    assert cfg.params == {"a": 0.25, "b": 7.5, "k": 25, "mode": "z"}     # 23 snaps to 25
    assert sp.decode(sp.encode({"k": 42})).params["k"] == 30              # clipped to hi
    assert cfg.selected == [] and cfg.per_partition == {}
    assert sp.decode(sp.default_vector()).params == {"a": 0.0, "b": 5.0, "k": 5, "mode": "y"}


def test_fixed_params_flow_into_decode():
    sp = SearchSpace([Param("k", 1, 10, "int")], fixed={"spat_mean": 0})
    assert sp.decode(sp.default_vector()).params["spat_mean"] == 0


def test_distinct_and_distance_to_bounds(simple_space):
    sp = simple_space
    x = sp.default_vector()
    assert not sp.distinct(x, x)
    assert sp.distinct(x, x + np.array([1e-3, 0, 0, 0]))
    d = sp.distance_to_bounds(x)
    assert d["a"] == pytest.approx(0.5) and d["b"] == pytest.approx(0.5)


# ----------------------------------------------------------------------------
# serialisation
# ----------------------------------------------------------------------------
def test_to_dict_from_dict_round_trip(partition_space, rng):
    sp = partition_space
    d = sp.to_dict()
    sp2 = SearchSpace.from_dict(d)
    assert sp2.names == sp.names
    assert sp2.partitions == sp.partitions
    assert sp2.selection == "two_slot" and sp2.selection_max_drop == 2
    assert sp2["k_klip_p1"].grid.tolist() == sp["k_klip_p1"].grid.tolist()
    for _ in range(20):
        x = sp.random(rng)
        c1, c2 = sp.decode(x), sp2.decode(x)
        assert c1.params == c2.params and c1.per_partition == c2.per_partition and c1.selected == c2.selected
    assert sp2.to_dict() == d


def test_from_dict_categorical_and_fixed(simple_space):
    sp = simple_space
    sp.fixed["comb_type"] = "mean"
    sp2 = SearchSpace.from_dict(sp.to_dict())
    assert sp2["mode"].choices == ["x", "y", "z"]
    assert sp2.decode(sp2.encode({"mode": "z"})).params["mode"] == "z"
    assert sp2.fixed == {"comb_type": "mean"}
    assert "mode" in sp2.setup_text()


def test_two_source_area_midpoint_band():
    """Addendum 2 §2b/§2c: with exactly two sources the Runner collapses the injection band
    to the annulus' area-weighted mid radius sqrt((r_in^2+r_out^2)/2) (14.1 px = 0.645" for
    [0, 20] px at NEAR scale); the placement routine then puts both there 180 deg apart.
    n >= 3 keeps the inset ladder; ``pair_area_midpoint=False`` restores the old ladder."""
    from klip_tpe.positions import PositionSampler
    px, fw = 0.0456, 6.238
    ps = PositionSampler(fwhm_as=fw * px)
    rng = np.random.default_rng(3)
    r_area = np.sqrt(0.5 * (0 ** 2 + 20 ** 2))
    src = ps.sample(2, r_area * px, r_area * px, rng, 6e-5)
    assert np.allclose([s.rho for s in src], r_area * px) and np.isclose(r_area, 14.14, atol=0.01)
    assert np.isclose(abs((src[0].theta - src[1].theta + 180) % 360 - 180), 180)
    # old ladder on the inset + IWA-clamped band [9.4, 13.8] px -> 10.5 / 12.7 px (IDL pre-change)
    old = ps.sample(2, (0 + fw) * px, (20 - fw) * px, np.random.default_rng(3))
    assert np.allclose(sorted(s.rho / px for s in old), [10.46, 12.66], atol=0.05)
    src3 = ps.sample(3, (0 + fw) * px, (20 - fw) * px, rng)
    assert len({round(s.rho, 6) for s in src3}) == 3
