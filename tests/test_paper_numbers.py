"""The helpers that turn run outputs into the paper's numbers: medians and win counts from
library_ablation.py (scripts/ablation_summary.py), the benchmark speed factor
(scripts/bench_numbers.py), and the cached F1140C companion fit (paper_runs/miri_fig.py)."""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def AS():
    return _load(os.path.join(ROOT, "scripts", "ablation_summary.py"), "ablation_summary")


@pytest.fixture(scope="module")
def BN():
    return _load(os.path.join(ROOT, "scripts", "bench_numbers.py"), "bench_numbers")


def _ablation(winner, other, draws, contrast=1e-4, backend="klip"):
    return {"backend": backend, "n_draws": len(winner),
            "annuli": [{"annulus": 1, "contrast": contrast, "draws": draws,
                        "configs": {"winner": {"raw": list(winner)}, "all": {"raw": list(other)}}}]}


def test_the_summary_uses_medians_and_counts_draws_not_means(AS):
    """library_ablation.py logs means; the paper quotes medians and draw counts."""
    w = [1.0, 2.0, 3.0, 100.0]           # mean 26.5, median 2.5
    o = [2.0, 1.0, 2.0, 2.0]             # median 2.0
    row = AS.summarize(_ablation(w, o, [[0.9, 10.0]] * 4))[0]
    assert row["configs"]["winner"]["median"] == pytest.approx(2.5)
    assert row["configs"]["all"]["ratio"] == pytest.approx(1.25)
    assert row["configs"]["all"]["winner_ahead"] == 3      # 1 < 2, then 2 > 1, 3 > 2, 100 > 2


def test_head_to_head_pairs_draw_for_draw_and_refuses_different_injections(AS):
    d = [[0.9, 10.0], [0.9, 190.0], [0.9, 100.0]]
    a = _ablation([3.0, 5.0, 4.0], [0, 0, 0], d)
    b = _ablation([4.0, 4.0, 4.0], [0, 0, 0], d, backend="pyklip")
    r = AS.head_to_head(a, b)[0]
    assert r["ratio"] == pytest.approx(1.0) and r["first_ahead"] == 1 and r["n"] == 3
    with pytest.raises(ValueError, match="same draws"):
        AS.head_to_head(a, _ablation([4.0] * 3, [0] * 3, d, contrast=2e-4))
    with pytest.raises(ValueError, match="same draws"):
        AS.head_to_head(a, _ablation([4.0] * 3, [0] * 3, [[0.9, 11.0]] * 3))


def test_speed_factor_is_the_first_evaluation_that_reaches_the_other_arms_final(BN):
    curve = np.array([1.0, 2.0, 2.0, 3.0, 3.5, 4.0, 4.0, 4.0])
    sf = BN.speed_factor(curve, 3.0)
    assert sf == {"eval": 4, "n": 8, "factor": 2.0}
    assert BN.speed_factor(curve, 5.0) is None


def test_the_f1140c_companion_fit_is_read_from_its_cache(tmp_path, monkeypatch):
    """miri_fig and companion_tests subtract and measure the same companion: the fit is made
    once and read back, without a reduction."""
    pr = os.path.join(ROOT, "paper_runs")
    if not os.path.exists(os.path.join(pr, "miri_fig.py")):
        pytest.skip("paper_runs/ not next to the package")
    monkeypatch.setenv("MIRI_RUNS", str(tmp_path))
    sys.path.insert(0, pr)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    M = _load(os.path.join(pr, "miri_fig.py"), "miri_fig_cache_test")
    assert M.NEGFC_F1140C == os.path.join(str(tmp_path), "_negfc", "hip65426b_f1140c.json")
    os.makedirs(os.path.dirname(M.NEGFC_F1140C))
    with open(M.NEGFC_F1140C, "w") as f:
        json.dump({"rho": 0.8407, "pa": 148.64, "contrast": 5.9428e-4}, f)
    assert M.miri_negfc(None, None, None, None) == (0.8407, 148.64, 5.9428e-4)
