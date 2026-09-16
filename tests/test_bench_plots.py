"""Benchmark harness + plotting diagnostics on the synthetic reducer (fast)."""
import os

import numpy as np
import pytest

# end-to-end on synthetic data (~20 s on two cores); `-m \"not slow\"` skips it
pytestmark = pytest.mark.slow

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from klip_tpe import bench, plots  # noqa: E402

MODES = ("tpe", "random", "grid")
SEEDS = (0, 1)
N_ITER, N_INIT = 30, 8


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("bench"))
    mk = bench.make_synthetic_bench(n_partitions=2, size=72, n_valid=3, n_top=2,
                                    calib_dir=os.path.join(out, "calib"))
    rows = bench.run_benchmark(mk, MODES, SEEDS, n_iter=N_ITER, n_init=N_INIT, bench_tag="bench_test",
                               out_dir=out, log=lambda s: None)
    return out, mk, rows


def test_summary_rows_and_files(batch):
    out, mk, rows = batch
    assert 0 < mk.contrast < 1e-2
    assert len(rows) == len(MODES) * len(SEEDS)              # one annulus
    parsed = bench.read_summary(out, "bench_test")
    assert len(parsed) == len(rows)
    assert {(r["mode"], r["seed"]) for r in parsed} == {(m, s) for m in MODES for s in SEEDS}
    for r in parsed:
        assert r["bench_tag"] == "bench_test" and r["n_iter"] == N_ITER and r["n_init"] == N_INIT
        assert np.isfinite(r["seeded_default_score"]) and np.isfinite(r["search_best"])
        assert np.isfinite(r["validated_best"])
        assert r["search_best"] >= r["seeded_default_score"] - 1e-9
        for f in ("results.jsonl", "results.txt", "checkpoint.json", "final_results.json", "run_setup.json"):
            assert os.path.exists(os.path.join(r["run_dir"], f))
        assert os.path.exists(os.path.join(r["run_dir"], "annulus01", "winner.json"))
    with open(os.path.join(out, "bench_tag.txt")) as f:
        assert f.read().strip() == "bench_test"
    # tag filtering: a foreign batch never leaks into the summary
    with open(os.path.join(out, "bench_summary.txt"), "a") as f:
        f.write("bench_other tpe 99 1 30 8 1.0 2.0 3.0 /nowhere\n")
    assert all(r["bench_tag"] == "bench_test" for r in bench.read_summary(out, "bench_test"))
    assert len(bench.read_summary(out, "all")) == len(rows) + 1


def test_summarize_and_convergence(batch):
    out, _, rows = batch
    lines = []
    s = bench.summarize_bench(out, "bench_test", log=lines.append)
    assert set(s["modes"]) == set(MODES)
    for m in MODES:
        assert s["modes"][m]["n"] == len(SEEDS)
        assert np.isfinite(s["modes"][m]["validated_mean"])
    assert set(s["paired"]) == {"random", "grid"}
    assert s["paired"]["random"]["n_pairs"] == len(SEEDS)
    assert any("paired tpe - grid" in l for l in lines)
    curves = bench.bench_convergence([r["run_dir"] for r in rows])
    assert set(curves) == set(MODES)
    for m, c in curves.items():
        assert c["curves"].shape == (len(SEEDS), N_ITER)
        assert np.all(np.diff(c["mean"]) >= -1e-9)          # running best is monotone
        assert np.isfinite(c["default"])
    png = os.path.join(out, "conv.png")
    fig = plots.plot_bench_convergence(curves, out_path=png)
    matplotlib.pyplot.close(fig)
    assert os.path.getsize(png) > 1000


def test_status_and_restart_noop(batch):
    out, mk, _ = batch
    st = bench.bench_status(out, "bench_test", MODES, SEEDS, n_iter=N_ITER)
    assert all(s["status"] == "finished" for s in st)
    assert bench.bench_status(out, "bench_test", MODES, (7,))[0]["status"] == "missing"
    assert [s["status"] for s in bench.bench_status(out, "bench_test", ("tpe",), SEEDS, n_iter=99)] == \
        ["budget_mismatch"] * len(SEEDS)
    # everything finished: restart adds no rows and does not re-run anything
    rows = bench.bench_restart(out, mk, "bench_test", MODES, SEEDS, N_ITER, N_INIT, log=lambda s: None)
    assert rows == []
    assert len(bench.read_summary(out, "bench_test")) == len(MODES) * len(SEEDS)


def test_restart_resumes_interrupted_run(tmp_path):
    out, tag = str(tmp_path / "br"), "bench_r"
    mk = bench.make_synthetic_bench(n_partitions=2, size=72, n_valid=2, n_top=1,
                                    calib_dir=str(tmp_path / "calib"))
    r = mk("tpe", 0, bench.slot_dir(out, tag, "tpe", 0))
    r.cfg.search_mode, r.cfg.bench_tag, r.cfg.n_iter, r.cfg.n_init = "tpe", tag, 20, 6

    class Stop(Exception):
        pass

    orig = r.checkpoint

    def interrupt(final_annulus=False):
        orig(final_annulus)
        if r.records_count >= 9:
            raise Stop

    r.checkpoint = interrupt
    with pytest.raises(Stop):
        r.run()
    with open(os.path.join(out, "bench_tag.txt"), "w") as f:
        f.write(tag)
    st = {s["mode"]: s["status"] for s in bench.bench_status(out, tag, ("tpe", "random"), (0,), 20)}
    assert st == {"tpe": "resumable", "random": "missing"}
    rows = bench.bench_restart(out, mk, None, ("tpe", "random"), (0,), 20, 6, log=lambda s: None)
    assert {(r_["mode"], r_["seed"]) for r_ in rows} == {("tpe", 0), ("random", 0)}
    recs = bench.read_records(bench.slot_dir(out, tag, "tpe", 0), 0)
    assert len(recs) == 20 and recs[0]["phase"] == "seed"
    assert all(s["status"] == "finished" for s in bench.bench_status(out, tag, ("tpe", "random"), (0,), 20))


def test_plots_per_run(batch):
    out, _, rows = batch
    run_dir = [r["run_dir"] for r in rows if r["mode"] == "tpe"][0]
    figs = plots.plot_all(run_dir)
    pdir = os.path.join(run_dir, "plots")
    for name in ("trace_ann01", "param_hist_ann01", "partition_snr_ann01", "contrast_curve", "validation"):
        assert name in figs
        p = os.path.join(pdir, name + ".png")
        assert os.path.exists(p) and os.path.getsize(p) > 1000, name
    # labels carry the bias warning
    fig = plots.plot_trace(run_dir, save=False)
    assert "upward-biased" in fig.axes[0].get_ylabel()
    assert any("validated" in t.get_text() for t in fig.axes[0].get_legend().get_texts())
    matplotlib.pyplot.close(fig)
    # a grid run has no per-partition selection changes but must still plot
    grid_dir = [r["run_dir"] for r in rows if r["mode"] == "grid"][0]
    for f in plots.plot_all(grid_dir, save=False).values():
        matplotlib.pyplot.close(f)


def test_plot_testbed(tmp_path):
    from klip_tpe.testbed import compare_density_models, RidgeObjective, run_tpe_on_objective
    res = compare_density_models(nblock=2, bdim=3, n_iter=40, n_init=10, nseed=2, log=lambda s: None)
    p1 = str(tmp_path / "testbed_bars.png")
    matplotlib.pyplot.close(plots.plot_testbed(res, out_path=p1))
    obj = RidgeObjective(2, 3, seed=1000)
    runs = {"univariate": run_tpe_on_objective(obj, None, 40, 10, seed=0),
            "block": run_tpe_on_objective(obj, obj.blocks(), 40, 10, seed=0)}
    p2 = str(tmp_path / "testbed_curves.png")
    matplotlib.pyplot.close(plots.plot_testbed(runs, out_path=p2))
    assert os.path.getsize(p1) > 1000 and os.path.getsize(p2) > 1000
