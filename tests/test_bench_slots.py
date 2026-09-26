"""A batch is one budget, and its figures draw the slots its summary counts, and only those."""
from __future__ import annotations

import importlib.util
import os

import pytest

from klip_tpe import bench

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _row(tag, mode, seed, annulus, run_dir):
    return dict(bench_tag=tag, mode=mode, seed=seed, annulus=annulus, n_iter=10, n_init=2,
                seeded_default_score=1.0, search_best=2.0, validated_best=1.5, run_dir=run_dir)


def test_summary_run_dirs_leaves_out_retired_and_foreign_slots(tmp_path):
    """``scripts/supersede_bench_mode.py`` retires an arm by renaming its slots
    ``..._sN_superseded_<date>`` and moving its rows out of the summary.  ``figs.py`` globbed
    ``bench_*_*_s*``, which still found them: the paper's E2 panel drew "grid (16 seeds)",
    eight of them retired, beside a bar chart of the eight that count."""
    out = tmp_path / "E2_bench"
    out.mkdir()
    (out / "bench_tag.txt").write_text("bench_1\n")
    live = []
    for m in ("tpe", "grid"):
        for s in (0, 1):
            d = out / f"bench_1_{m}_s{s}"
            d.mkdir()
            live.append(str(d))
    (out / "bench_1_grid_s0_superseded_20260920_150534").mkdir()      # the retired arm
    (out / "bench_0_tpe_s0").mkdir()                                  # an older batch
    moved = tmp_path / "elsewhere" / "bench_2_tpe_s0"                 # a slot recorded elsewhere
    moved.mkdir(parents=True)
    rows = [_row("bench_1", m, s, a, f"/was/here/before/a/move/bench_1_{m}_s{s}")
            for m in ("tpe", "grid") for s in (0, 1) for a in (0, 1)]   # two annuli, one slot
    rows += [_row("bench_0", "tpe", 0, 0, str(out / "bench_0_tpe_s0")),
             _row("bench_2", "tpe", 0, 0, str(moved))]
    bench._append_summary(str(out), rows)

    assert bench.summary_run_dirs(str(out)) == sorted(live)            # the current tag
    assert bench.summary_run_dirs(str(out), "bench_0") == [str(out / "bench_0_tpe_s0")]
    assert bench.summary_run_dirs(str(out), "bench_2") == [str(moved)]  # the recorded path
    assert bench.summary_run_dirs(str(tmp_path / "nowhere")) == []


def _never(mode, seed, run_dir):
    raise AssertionError(f"ran {mode} s{seed}")


def test_a_batch_is_extended_only_at_its_own_budget(tmp_path):
    """E2's grid arm was re-run -- after ``supersede_bench_mode.py`` retired it -- without the
    ``BENCH_NITER=1000`` the batch ran under, and ran 800 evaluations beside tpe and random at
    1000.  A slot about to run at another budget than the batch's rows is now refused."""
    out = tmp_path / "E2_bench"
    out.mkdir()
    (out / "bench_tag.txt").write_text("bench_1\n")
    bench._append_summary(str(out), [_row("bench_1", m, s, 0, str(out / f"bench_1_{m}_s{s}"))
                                     for m in ("tpe", "random") for s in (0, 1)])   # n_iter 10, n_init 2
    q = lambda s: None
    with pytest.raises(ValueError, match=r"n_iter=10, n_init=2.*grid s0, grid s1.*n_iter=8.*BENCH_NITER=10"):
        bench.run_benchmark(_never, ("grid",), (0, 1), n_iter=8, n_init=2, bench_tag="bench_1",
                            out_dir=str(out), log=q)
    with pytest.raises(ValueError, match="n_init=3"):
        bench.run_benchmark(_never, ("grid",), (0,), n_iter=10, n_init=3, bench_tag="bench_1",
                            out_dir=str(out), log=q)
    # resuming a batch with nothing left to run only reads it, whatever the call asks
    assert bench.run_benchmark(_never, ("tpe", "random"), (0, 1), n_iter=8, n_init=2,
                               bench_tag="bench_1", out_dir=str(out), log=q) == []


def test_supersede_says_which_budget_to_rerun_at(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("supersede_bench_mode",
                                                  os.path.join(ROOT, "scripts", "supersede_bench_mode.py"))
    sup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sup)
    out = tmp_path / "E2_bench"
    out.mkdir()
    (out / "bench_tag.txt").write_text("bench_1\n")
    for m in ("tpe", "grid"):
        (out / f"bench_1_{m}_s0").mkdir()
    earlier = out / "bench_1_grid_s0_superseded_20260920_150534"     # retired once already
    earlier.mkdir()
    bench._append_summary(str(out), [
        dict(_row("bench_1", "tpe", 0, 0, str(out / "bench_1_tpe_s0")), n_iter=1000, n_init=100),
        dict(_row("bench_1", "grid", 0, 0, str(out / "bench_1_grid_s0")), n_iter=800, n_init=80)])
    assert sup.main([str(out), "grid"]) == 0
    said = capsys.readouterr().out
    assert "BENCH_NITER=1000 BENCH_MODES=grid python3 paper_runs/run_demos.py E2" in said
    assert "the retired arm ran n_iter 800; the rest of the batch 1000" in said
    assert bench.summary_run_dirs(str(out)) == [str(out / "bench_1_tpe_s0")]
    # the live slot is retired; the one an earlier retirement renamed keeps its name (E2's
    # 2026-09-20 slots were stamped a second time on 09-25 by a prefix match)
    assert earlier.is_dir()
    retired = sorted(n for n in os.listdir(out) if n.startswith("bench_1_grid_s0_"))
    assert len(retired) == 2 and all(n.count("_superseded_") == 1 for n in retired), retired
