import numpy as np
import pytest

from klip_tpe.testbed import RidgeObjective, compare_density_models, make_space, run_tpe_on_objective


def _centre_vector(obj):
    x = np.zeros(obj.ndim)
    for b in range(obj.ngroup):
        x[obj.gidx[b]] = obj.ctr[b]
    return x


@pytest.mark.parametrize("transposed", [False, True])
def test_ridge_objective_true_max_is_10_at_centres(transposed):
    obj = RidgeObjective(nblock=3, bdim=4, transposed=transposed, seed=5)
    assert obj.ndim == 3 * 4 + 2
    xc = _centre_vector(obj)
    assert obj.true(xc) == pytest.approx(10.0)
    rng = np.random.default_rng(0)
    vals = [obj.true(rng.random(obj.ndim)) for _ in range(200)]
    assert max(vals) < 10.0 and min(vals) > 0.0
    # moving off-centre along a narrow ridge direction costs more than along the broad one
    b = 0
    broad = obj.rot[b][0]
    narrow = obj.rot[b][1]
    xb, xn = xc.copy(), xc.copy()
    xb[obj.gidx[b]] += 0.3 * broad
    xn[obj.gidx[b]] += 0.3 * narrow
    assert obj.true(xn) < obj.true(xb) < 10.0
    # the two trailing 'selection' dims do not affect the score
    xs = xc.copy()
    xs[-2:] = 0.9
    assert obj.true(xs) == pytest.approx(10.0)
    # deterministic in the seed
    assert np.array_equal(RidgeObjective(3, 4, transposed=transposed, seed=5).ctr, obj.ctr)


def test_blocks_and_space():
    obj = RidgeObjective(nblock=2, bdim=3)
    assert obj.blocks() == [[0, 1, 2], [3, 4, 5], [6, 7]]
    sp = make_space(obj.ndim)
    assert sp.ndim == 8 and sp.names[0] == "x0" and sp.hi.max() == 1.0


def test_run_tpe_on_objective_paired_warmup():
    obj = RidgeObjective(nblock=2, bdim=3, seed=1001)
    a = run_tpe_on_objective(obj, None, n_iter=40, n_init=15, seed=3)
    b = run_tpe_on_objective(obj, obj.blocks(), n_iter=40, n_init=15, seed=3)
    for k in ("X", "Y", "T", "best_obs", "true_at_best"):
        assert a[k].shape[0] == 40
    np.testing.assert_array_equal(a["X"][:15], b["X"][:15])     # identical warm-up stream
    assert np.all(np.diff(a["best_obs"]) >= 0)
    assert np.all(a["X"] >= 0) and np.all(a["X"] <= 1)
    assert a["true_at_best"][-1] <= 10.0
    # observed best is optimistic relative to the truth at that point
    assert a["best_obs"][-1] >= a["true_at_best"][-1] - 1.0


def test_compare_density_models_short_paired_run():
    logs = []
    out = compare_density_models(nblock=2, bdim=3, n_iter=80, n_init=20, nseed=2, log=logs.append)
    assert set(out) == {"univariate", "block", "full"}
    for m, d in out.items():
        assert set(d) == {"true_mean", "true_sd", "obs_mean", "optimism", "diff_vs_ref", "diff_t", "wins"}
        assert 0.0 < d["true_mean"] <= 10.0
        assert 0 <= d["wins"] <= 2
        assert np.isfinite(d["true_sd"])
    assert out["univariate"]["diff_vs_ref"] == 0.0 and out["univariate"]["wins"] == 0
    assert len(logs) == 2 * 3 + 3
    # a subset of modes and a single seed also work
    out2 = compare_density_models(nblock=2, bdim=3, n_iter=40, n_init=15, nseed=1, modes=("block",), log=lambda s: None)
    assert list(out2) == ["block"] and out2["block"]["true_sd"] == 0.0
