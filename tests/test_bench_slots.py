"""A batch's figures draw the slots its summary counts, and only those."""
from __future__ import annotations

import os

from klip_tpe import bench


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
