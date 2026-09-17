"""The paper's run driver: the core budget it actually uses, and the benchmark stages.

Two things this pins down.

``WORKERS`` had been exported by ``rerun_paper.sh`` since it was written and read by nobody:
every reducer in ``run_demos.py`` said ``max_workers="auto"``, which means every core on the
machine.  ``WORKERS=4 ./rerun_paper.sh bench`` therefore ran on all 32, and four searches at
32 workers each took single reductions from 6 s to 729 s and got one of them killed.

And the benchmark had only one subject.  E2 and F2 are both beta Pic, whose debris disk runs
through the [8, 22] px annulus they search, so the field the injected sources are measured
against is not empty.  G2 (HD 95086, SPHERE) and H2 (HIP 65426, JWST) put the same protocol
on two fields with no scattered-light disk, at 20 and 11 searched dimensions.

These tests do not run a benchmark -- they check that the driver is wired to do the right
thing when it is run, which is what a multi-hour stage gets wrong expensively.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [os.path.join(os.path.dirname(HERE), "paper_runs"),
              os.path.join(os.path.dirname(os.path.dirname(HERE)), "paper_runs")]
PAPER_RUNS = next((d for d in CANDIDATES if os.path.exists(os.path.join(d, "run_demos.py"))), None)
pytestmark = pytest.mark.skipif(PAPER_RUNS is None, reason="paper_runs/ not next to the package")


def _text(name):
    with open(os.path.join(PAPER_RUNS, name)) as f:
        return f.read()


@pytest.fixture(scope="module")
def demos():
    """``run_demos`` imported without running anything (it guards on ``__main__``)."""
    spec = importlib.util.spec_from_file_location("paper_run_demos",
                                                  os.path.join(PAPER_RUNS, "run_demos.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paper_run_demos"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ the core budget

def test_workers_reads_the_environment(demos, monkeypatch):
    for val, want in (("6", 6), ("1", 1), ("-2", -2), ("auto", "auto"), ("max", "max"), ("", "auto")):
        monkeypatch.setenv("WORKERS", val)
        assert demos.workers() == want, val
    monkeypatch.delenv("WORKERS", raising=False)
    assert demos.workers() == "auto"           # unset: every core, as before


def test_no_stage_hardwires_every_core():
    """The actual defect: a budget the driver could not be asked to lower."""
    src = _text("run_demos.py")
    assert 'max_workers="auto"' not in src
    assert src.count("max_workers=workers()") >= 9


def test_rerun_paper_exports_workers_and_show_to_the_stage():
    sh = _text("rerun_paper.sh")
    assert re.search(r'WORKERS="\$\{WORKERS:-auto\}" SHOW="\$\{SHOW:-window\}"', sh)
    assert 'bash -c "$(stage_cmd "$s")"' in sh


# ------------------------------------------------------------------ the new stages

def test_the_new_benchmark_stages_are_dispatchable(demos):
    src = _text("run_demos.py")
    for stage in ("G2", "H2"):
        assert hasattr(demos, f"run_{stage}"), stage
        assert f'"{stage}": run_{stage}' in src, stage


def test_rerun_paper_runs_and_recognises_the_new_stages():
    sh = _text("rerun_paper.sh")
    assert "BENCH=(E2 F2 G2 H2)" in sh
    assert "G2) echo G2_bench_sphere" in sh          # so a finished stage is skipped
    assert "H2) echo H2_bench_jwst" in sh


def test_collect_and_figs_know_where_the_new_benches_land():
    col = _text("collect.py")
    assert '"G": ("G2_bench_sphere",)' in col
    assert '"H": ("H2_bench_jwst",)' in col
    fig = _text("figs.py")
    assert "G2_bench_sphere" in fig and "H2_bench_jwst" in fig


def test_the_bench_figure_orders_its_rows_by_dimension():
    """The ladder only reads as a ladder if the rows are in order: 9, 11, 20, 38."""
    fig = _text("figs.py")
    order = [fig.index(d) for d in ("E2_bench", "H2_bench_jwst", "G2_bench_sphere", "F2_bench_highdim")]
    assert order == sorted(order), order


# -------------------------------------------------------- _bench_hi is dataset-agnostic

class _FakeSpace:
    def __init__(self):
        self.added, self.project = [], None

    def add(self, p):
        self.added.append(p)


def _capture(demos, monkeypatch):
    """Run one ``_bench_hi`` with everything below it stubbed, and return what it built."""
    seen = {}

    def make_red():
        seen["made_reducer"] = True
        return "REDUCER"

    monkeypatch.setattr(demos.generic, "default_config",
                        lambda red, known, **kw: seen.update(known=known, cfg_red=red, cfg_kw=kw)
                        or ("OBJ", "SAMP"))
    # bp_disk wants a real reducer to size the mask; the stub reducer is a string
    monkeypatch.setattr(demos, "bp_disk", lambda *a, **k: ((), None))
    monkeypatch.setattr(demos.generic, "make_space",
                        lambda red, **kw: seen.update(space_kw=kw) or _FakeSpace())
    monkeypatch.setattr(demos.generic, "make_guard", lambda red, **kw: seen.update(guard_kw=kw))
    monkeypatch.setattr(demos, "Runner", lambda *a, **kw: seen.update(cfg=a[4]) or "RUNNER")

    import klip_tpe.bench as B

    def fake_bench(make_runner, modes, seeds, n_iter, n_init, bench_tag, out_dir, log):
        seen.update(modes=modes, seeds=seeds, n_iter=n_iter, out_dir=out_dir)
        make_runner("tpe", 0, out_dir)          # build one, so the space/cfg are captured
    monkeypatch.setattr(B, "run_benchmark", fake_bench)
    return seen, make_red


def test_bench_hi_uses_the_reducer_it_is_given(demos, monkeypatch, tmp_path):
    seen, make_red = _capture(demos, monkeypatch)
    monkeypatch.setattr(demos, "OUT", str(tmp_path))
    monkeypatch.setattr(demos, "betapic_dataset",
                        lambda g: pytest.fail("beta Pic loaded for a non-beta-Pic benchmark"))
    demos._bench_hi("X", 1, ("tpe", "random"), 30, None, {"k_klip": 10}, 1e-4, [20, 45], "X_bench",
                    make_red=make_red, known=[(0.62, 145.0)], n_sources=3, n_min_ref=10)
    assert seen["made_reducer"] and seen["cfg_red"] == "REDUCER"
    assert seen["known"] == [(0.62, 145.0)]
    assert seen["cfg"].n_sources == 3
    assert seen["guard_kw"]["n_min_ref"] == 10
    assert seen["cfg"].ann_edges == [20, 45]


def test_bench_hi_still_defaults_to_beta_pic(demos, monkeypatch, tmp_path):
    """E2 and F2 must keep working unchanged: they pass no reducer at all."""
    seen, _ = _capture(demos, monkeypatch)
    monkeypatch.setattr(demos, "OUT", str(tmp_path))
    calls = []
    monkeypatch.setattr(demos, "betapic_dataset", lambda g: calls.append(g) or ({}, 1.0, {}))
    monkeypatch.setattr(demos.generic, "make_reducer", lambda *a, **kw: "BP_REDUCER")
    demos._bench_hi("F2", 4, ("tpe", "random"), 12, 2, {"k_klip": 5}, 2.087e-3, [8, 22], "F2_bench_highdim")
    assert calls == [4]                                  # the four-group split, as before
    assert seen["known"] == [demos.BP]
    assert seen["cfg"].n_sources == 3


def test_max_drop_none_leaves_make_space_its_own_default(demos, monkeypatch, tmp_path):
    """G2/H2 must search the same space their science runs did, drop slots included."""
    seen, make_red = _capture(demos, monkeypatch)
    monkeypatch.setattr(demos, "OUT", str(tmp_path))
    demos._bench_hi("G2", 1, ("tpe",), 30, None, {"k_klip": 10}, 1e-4, [20, 45], "G2_bench_sphere",
                    make_red=make_red)
    assert "max_drop" not in seen["space_kw"]
    seen2, make_red2 = _capture(demos, monkeypatch)
    demos._bench_hi("F2", 4, ("tpe",), 12, 2, {"k_klip": 5}, 1e-4, [8, 22], "F2x", make_red=make_red2)
    assert seen2["space_kw"]["max_drop"] == 2


def test_h2_carries_the_searched_klip_mode_and_unsearched_angles(demos, monkeypatch, tmp_path):
    """Run D searches ADI/RDI/ADI+RDI and does not search angles (two frames per roll)."""
    seen = {}
    space = _FakeSpace()
    monkeypatch.setattr(demos, "OUT", str(tmp_path))
    monkeypatch.setattr(demos, "hip65426_objects", lambda: "JWST_REDUCER")
    monkeypatch.setattr(demos.generic, "default_config", lambda red, known, **kw: ("OBJ", "SAMP"))
    monkeypatch.setattr(demos.generic, "make_space", lambda red, **kw: seen.update(kw) or space)
    monkeypatch.setattr(demos.generic, "make_guard", lambda red, **kw: None)
    monkeypatch.setattr(demos, "Runner", lambda *a, **kw: "RUNNER")
    import klip_tpe.bench as B
    monkeypatch.setattr(B, "run_benchmark",
                        lambda make_runner, modes, seeds, n_iter, n_init, bench_tag, out_dir, log:
                        make_runner("tpe", 0, out_dir))
    demos.run_H2()
    assert seen["search_angles"] is False
    assert [p.name for p in space.added] == ["mode"]
    assert space.added[0].choices == ["ADI", "RDI", "ADI+RDI"]


# ------------------------------------------------------------------ the live window

def test_show_mode_reads_the_environment(demos, monkeypatch):
    for val, want in (("window", "window"), ("inline", "inline"), ("auto", "auto"),
                      ("0", None), ("off", None), ("", "window")):
        monkeypatch.setenv("SHOW", val)
        assert demos.show_mode() == want, val
    monkeypatch.delenv("SHOW", raising=False)
    assert demos.show_mode() == "window"        # on by default: this is the interactive driver


def test_every_stage_passes_show_to_its_display(demos, monkeypatch):
    """The old bug: a LiveDisplay built without show= writes panels and opens nothing, which
    looks exactly like a broken display."""
    src = _text("run_demos.py")
    assert "LiveDisplay(d, every=10, pdf_every=0, movie=False, dpi=100)" not in src
    assert src.count("callbacks=[_display(d)]") == 5      # A, A2, B, C, D
    monkeypatch.setenv("SHOW", "window")
    seen = {}
    import klip_tpe.display as dmod
    monkeypatch.setattr(dmod, "LiveDisplay", lambda d, **kw: seen.update(kw) or "DISP")
    assert demos._display("/tmp/x") == "DISP"
    assert seen["show"] == "window" and seen["every"] == 10
    assert seen["movie"] is False and seen["pdf_every"] == 0


def test_the_benchmark_slots_get_a_display_too(demos, monkeypatch, tmp_path):
    """A benchmark had no window at all, and it is the run you most want to watch."""
    seen, make_red = _capture(demos, monkeypatch)
    monkeypatch.setattr(demos, "OUT", str(tmp_path))
    monkeypatch.setenv("SHOW", "window")
    built = []
    monkeypatch.setattr(demos, "_display", lambda d, **kw: built.append((d, kw)) or "DISP")
    monkeypatch.setattr(demos, "Runner",
                        lambda *a, **kw: seen.update(cbs=kw.get("callbacks")) or "RUNNER")
    demos._bench_hi("G2", 1, ("tpe",), 30, None, {"k_klip": 10}, 1e-4, [20, 45], "G2x",
                    make_red=make_red)
    assert seen["cbs"] == ["DISP"] and built and built[0][1]["every"] == 25
    # and SHOW=0 gets none
    monkeypatch.setenv("SHOW", "0")
    demos._bench_hi("G2", 1, ("tpe",), 30, None, {"k_klip": 10}, 1e-4, [20, 45], "G2y",
                    make_red=make_red)
    assert seen["cbs"] == []


def test_the_live_window_is_shared_across_displays():
    """One window for a whole batch: dozens of slots must not open dozens of windows."""
    from klip_tpe.display import LiveDisplay
    for name in ("_win", "_win_ax", "_win_im", "_win_dir"):
        assert name in vars(LiveDisplay), f"{name} must be a class attribute to be shared"


# ------------------------------------------------------------- LMIRCam as a paper stage

def test_lmircam_is_a_science_stage_with_its_own_driver():
    sh = _text("rerun_paper.sh")
    assert "SCIENCE=(A2 B2 C D I2)" in sh
    assert "run_rxj0534.py" in sh                        # it keeps its companion-ring guards
    assert 'I2) echo "$RXJ_OUT"' in sh                   # so a finished one is recognised


def test_the_science_group_covers_all_four_instruments():
    """NACO, SPHERE, NIRCam, LMIRCam -- the point of the grouping."""
    sh = _text("rerun_paper.sh")
    for word in ("NACO", "SPHERE", "NIRCam", "LMIRCam"):
        assert word in sh, word


def test_the_stage_command_is_dry_runnable():
    """``DRY=1`` prints what each stage would run -- how you check a night's plan before it
    starts, rather than after."""
    sh = _text("rerun_paper.sh")
    assert 'if [[ -n "${DRY:-}" ]]' in sh
    assert "would run" in sh


def test_force_on_a_benchmark_stage_retires_the_batch():
    """``FORCE=1`` on a benchmark used to get past the stage skip and then resume anyway.

    A benchmark resumes by reading ``bench_tag.txt`` and rejoining that tag's slots, so a
    stage re-run with a changed setting applied it to the new slots only -- F2 ended up with
    slot 0 uncut, slots 1-4 cut and slot 5 half of each, in one directory. Retiring the
    directory takes bench_tag.txt with it, which is what forces a fresh tag.
    """
    sh = _text("rerun_paper.sh")
    assert 'if [[ -n "${FORCE:-}" && -n "$d" && -d "$d" ]]' in sh
    assert 'if [[ -f "$d/bench_tag.txt" ]]' in sh
    assert "_superseded_" in sh
    assert 'mv "$d" "$keep"' in sh
    i_skip = sh.index('already finished ($d) -- skipping')
    i_force = sh.index('retiring the existing batch')
    assert i_skip < i_force, "the retire step has to come after the finished-stage skip"


def test_force_on_a_finished_science_stage_retires_the_run_too():
    """``FORCE=1 ./rerun_paper.sh A2`` on a finished A2 used to get past the skip and launch
    ``run_demos.py A2`` into the same directory -- where ``Runner.run()`` auto-resumes the
    checkpoint, finds every annulus complete, rewrites the products and reports "done in
    2 min".  Nothing was searched again and the old calibration stayed.  The finished
    directory is retired like a benchmark batch.  I2 -- days of LMIRCam compute -- is never
    retired by a flag."""
    sh = _text("rerun_paper.sh")
    assert 'elif [[ -f "$d/final_results.json" ]]' in sh
    assert "retiring the finished run" in sh
    i_i2 = sh.index('if [[ "$s" == "I2" ]]')
    i_mv = sh.index('say "$s: FORCE -- retiring the finished run')
    assert i_i2 < i_mv, "the I2 exception has to be checked before a science run is moved"
    assert "FORCE does not retire the LMIRCam run" in sh
    assert "FORCE would retire the finished run" in sh, "a dry run should say what FORCE would do"
    head = sh[:sh.index("set -uo pipefail")]
    assert "science run" in head and "I2" in head


def test_dry_reports_without_touching_the_tree():
    """A dry run that retires a batch is not a dry run.

    The retire step went in ahead of the DRY check and moved a real benchmark directory
    during what was meant to be a look-only invocation.
    """
    sh = _text("rerun_paper.sh")
    i_dry = sh.index('if [[ -n "${DRY:-}" ]]')
    i_retire = sh.index('mv "$d" "$keep"')
    assert i_dry < i_retire, "DRY must be checked before anything that changes the tree"
    assert "FORCE would retire batch" in sh, "a dry run should still say what FORCE would do"


def test_a_failed_preflight_stops_the_run():
    """``set -uo pipefail`` has no ``-e``: the guard printed its traceback and the stages ran
    anyway, so the assertion protecting the objective protected nothing."""
    sh = _text("rerun_paper.sh")
    assert "if ! python3 - <<'PY'" in sh
    assert "pre-flight failed" in sh and "exit 1" in sh


def test_force_is_documented_as_retiring_a_benchmark_batch():
    sh = _text("rerun_paper.sh")
    head = sh[:sh.index("set -uo pipefail")]
    assert "superseded" in head and "uniform" in head


# ------------------------------------------------------- the objective the rerun asserts

def test_the_rerun_refuses_to_run_with_frozen_injection_angles():
    """``rerun_paper.sh`` used to assert the opposite, and reran everything wrongly.

    ``optimize_near_2_tpe`` re-draws the injection azimuths every evaluation
    (``near2m_randpos``: "so sources still rotate eval-to-eval (anti-gaming)"), so the
    driver's pre-flight check has to fail when ``fixed_sources`` is on, not when it is off.
    """
    from klip_tpe import RunConfig
    sh = _text("rerun_paper.sh")
    assert "assert not RunConfig().fixed_sources" in sh
    assert "assert RunConfig().fixed_sources," not in sh
    assert RunConfig().fixed_sources is False       # what the driver is checking for


def test_the_unpaired_archives_are_flagged_as_the_correct_ones():
    """Whoever reads collect.py next must not delete the 2026-09-13 archives as stale."""
    col = _text("collect.py").lower()
    assert "_unpaired_20260913" in col and "reference objective" in col


def test_the_two_new_stages_use_their_science_runs_calibrated_contrast(demos):
    """A benchmark at a contrast the target was never calibrated for measures nothing."""
    src = _text("run_demos.py")
    assert "5.899e-9" in src            # run C, HD 95086 annulus 1
    assert "5.270e1" in src             # run D, HIP 65426 annulus 1


# ------------------------------------------------------- collect: the paired default
def test_collect_measures_the_projected_seed_and_pairs_it_with_the_winner():
    """The "default" of Table 2 is what the run was SEEDED with: the space defaults plus
    RunConfig.defaults at the calibration k-scan's k, projected through the reference-count
    guard (angsep -> 0, anglemax -> the PA span) as ``Runner._reset_history`` does.  Until
    2026-09-17 collect measured ``space.default_vector()`` unprojected -- a configuration
    no run ever evaluates -- and compared it with a validated score from other draws.
    Default and winner are now re-scored on the same injection sets."""
    col = _text("collect.py")
    assert "runner._project(runner._default_vector(k_seed), is_random=False)" in col
    assert 'a.get("k_default")' in col
    assert "sources=src, raw_only=True" in col
    assert '"paired_wins"' in col and '"winner_remeasured"' in col and '"gain_vs_validated"' in col
    assert "default_flat" in col, "the configured k (before the k-scan) is measured as well"
    assert "x0 = space.default_vector(seeded)" not in col


def test_figs_scale_the_default_curve_by_the_paired_gain():
    src = _text("figs.py")
    assert 'a.get("gain") or (a["winner_score"] / max(a["default_score"], 1e-9))' in src
