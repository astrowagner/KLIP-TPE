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
    assert _outdir("G2") == "G2_bench_sphere"         # so a finished stage is skipped
    assert _outdir("H2") == "H2_bench_jwst_pyklip"


def test_collect_and_figs_know_where_the_new_benches_land():
    col = _text("collect.py")
    assert '"G": ("G2_bench_sphere",)' in col
    # H2 on each engine, and never the old H2_bench_jwst, whose newest slots searched the
    # nkeep counts pyKLIP ignored
    assert '"H": (R.ENGINE_DIRS["H2"]["pyklip"],)' in col and '"HK": (R.ENGINE_DIRS["H2"]["klip"],)' in col
    assert '("H2_bench_jwst",)' not in col
    fig = _text("figs.py")
    assert "G2_bench_sphere" in fig and "H2_bench_jwst_pyklip" in fig


def test_collect_with_no_arguments_collects_what_the_figures_read():
    """``python3 collect.py && python3 figs.py`` is what the drivers say to run once the
    stages finish.  Its default was A C D B -- two runs whose directories are gone -- so it
    refreshed C and D, printed two tracebacks, and left the A2 and B2 the figures read."""
    col = _text("collect.py")
    names = re.findall(r'"(\w+)"', re.search(r"^PAPER = \((.*?)\)$", col, re.M).group(1))
    assert "or list(PAPER)" in col
    fig = _text("figs.py")
    read = set(re.findall(r':\s*"(\w+)"', re.search(r"^PRIMARY = \{(.*?)\}$", fig, re.M).group(1)))
    read |= {"B2"}                                       # fig_partition / fig_frametags
    assert read <= set(names), read - set(names)
    assert {"DK", "H", "HK"} <= set(names)               # both engines' D and H2
    for w in names:                                      # each a target or a benchmark
        assert f'"{w}": (' in col, w


def test_collect_skips_a_run_that_is_not_there_in_one_line(tmp_path):
    import subprocess
    env = dict(os.environ, RUNS_DIR=str(tmp_path), PYTHONPATH=os.path.dirname(PAPER_RUNS))
    r = subprocess.run([sys.executable, "collect.py", "A", "x"], cwd=PAPER_RUNS, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "Traceback" not in r.stdout + r.stderr
    assert "A: A_betapic has no final_results.json" in r.stdout
    assert "X: not a target" in r.stdout
    assert "nothing collected; summary.json unchanged" in r.stdout
    assert not os.path.exists(tmp_path / "summary.json")


def _outdir(stage, runs_dir=None, home=None):
    """rerun_paper.sh's own outdir(), run by bash: its RUNS_DIR block and the function."""
    import subprocess
    sh = _text("rerun_paper.sh")
    start = sh.index('if [[ -n "${RUNS_DIR:-}" ]]; then')
    end = sh.index("\n}\n", sh.index("outdir() {", start)) + 3
    env = {k: v for k, v in os.environ.items() if k != "RUNS_DIR"}
    if runs_dir is not None:
        env["RUNS_DIR"] = runs_dir
    if home is not None:
        env["HOME"] = home
    out = subprocess.run(["bash", "-c", sh[start:end] + f"\nRXJ_OUT=/x/rxj; outdir {stage}"],
                         env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_the_nircam_stages_have_both_engines_in_their_own_directories(demos, tmp_path):
    """D / H2 on pyKLIP and DK / H2K on the built-in engine, each into a directory nothing
    older can be resumed from -- here, or under $RUNS_DIR."""
    dirs = demos.ENGINE_DIRS
    assert dirs["D"] == {"pyklip": "D_hip65426_pyklip", "klip": "D_hip65426_klip"}
    assert dirs["H2"] == {"pyklip": "H2_bench_jwst_pyklip", "klip": "H2_bench_jwst_klip"}
    for stage, d in (("D", "D_hip65426_pyklip"), ("DK", "D_hip65426_klip"),
                     ("H2", "H2_bench_jwst_pyklip"), ("H2K", "H2_bench_jwst_klip")):
        assert _outdir(stage) == d, stage
        assert _outdir(stage, str(tmp_path / "runs") + "/") == str(tmp_path / "runs" / d), stage


def test_runs_dir_moves_every_stage_directory_and_the_scripts_follow(tmp_path, monkeypatch):
    """A synced folder cannot keep up with a fast search (Dropbox filed hundreds of
    conflicted copies of H2K's checkpoint and heartbeat, and once put a stale checkpoint back
    under the real name), so the stage directories can live elsewhere: $RUNS_DIR moves
    run_demos.OUT -- and with it collect.py / figs.py, which read R.OUT -- and the shell
    drivers' finished / live / retire checks look in the same place.  ~ is expanded on both
    sides, and I2 keeps its own RXJ_OUT."""
    def load():
        spec = importlib.util.spec_from_file_location("paper_run_demos_runsdir",
                                                      os.path.join(PAPER_RUNS, "run_demos.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RUNS_DIR", "~/klip_tpe_runs/paper")
    want = str(tmp_path / "home" / "klip_tpe_runs" / "paper")
    assert load().OUT == want and os.path.isdir(want)
    assert _outdir("H2K", "~/klip_tpe_runs/paper", home=str(tmp_path / "home")) == want + "/H2_bench_jwst_klip"
    assert _outdir("I2", "~/klip_tpe_runs/paper") == "/x/rxj"
    monkeypatch.delenv("RUNS_DIR")
    assert load().OUT == os.path.abspath(PAPER_RUNS)
    for script in ("collect.py", "figs.py"):
        assert "OUT = R.OUT" in _text(script), script
    assert 'os.environ.get("RUNS_DIR"' in _text("long_run.sh")        # its progress ticker too


def test_the_bench_figure_orders_its_rows_by_dimension():
    """The ladder only reads as a ladder if the rows are in order: 5, 9, 20, 38.

    It used to say 9, 11, 20, 38, with the JWST problem second.  That was wrong twice
    over: HIP 65426 searches five dimensions, not eleven, so it is the *smallest* problem
    and belongs first.  The figure was fixed and this test was not, which is how it went red.

    Still the smallest: n_ang, filter, k_klip, and pyKLIP's own library -- ``mode`` and
    ``maxnumbasis``.  ``bin`` is pinned (its range collapses to (1, 1) on four science
    frames).  (For a day the library was the nkeep counts, which pyKLIP ignored.)"""
    fig = _text("figs.py")
    order = [fig.index(d) for d in ("H2_bench_jwst_pyklip", "E2_bench", "G2_bench_sphere", "F2_bench_highdim")]
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
    monkeypatch.setattr(demos, "preflight", lambda runner, what: None)

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


def test_h2_searches_no_mode_categorical_and_no_angles(demos, monkeypatch, tmp_path):
    """H2 does not search angles (two frames per roll, so no field rotation to exploit) and
    adds no dimension of its own.

    The library comes with the reducer ``hip65426_objects`` builds, through
    ``reference_params()`` inside ``make_space``: pyKLIP's ``mode`` + ``maxnumbasis`` for H2,
    the built-in engine's ``nkeep_altroll`` / ``nkeep_psfref`` for H2K.  Nothing should be
    added here, and each stage must ask for its own engine."""
    seen = {}
    space = _FakeSpace()
    engines = []
    monkeypatch.setattr(demos, "OUT", str(tmp_path))
    monkeypatch.setattr(demos, "hip65426_objects", lambda **k: engines.append(k.get("engine")) or "JWST_REDUCER")
    monkeypatch.setattr(demos, "preflight", lambda runner, what: None)
    monkeypatch.setattr(demos.generic, "default_config", lambda red, known, **kw: ("OBJ", "SAMP"))
    monkeypatch.setattr(demos.generic, "make_space", lambda red, **kw: seen.update(kw) or space)
    monkeypatch.setattr(demos.generic, "make_guard", lambda red, **kw: None)
    monkeypatch.setattr(demos, "Runner", lambda *a, **kw: "RUNNER")
    import klip_tpe.bench as B
    monkeypatch.setattr(B, "run_benchmark",
                        lambda make_runner, modes, seeds, n_iter, n_init, bench_tag, out_dir, log:
                        make_runner("tpe", 0, out_dir))
    demos.run_H2()
    demos.run_H2("klip")
    assert engines == ["pyklip", "klip"], engines
    assert seen["search_angles"] is False
    assert [p.name for p in space.added] == [], (
        f"H2 should add no dimension of its own; got {[p.name for p in space.added]}")


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


# ------------------------------------------------------- one driver, one run per directory
def test_the_driver_refuses_a_second_instance_and_a_live_directory():
    """2026-09-17: launched under nohup the driver prints nothing, which read as "it didn't
    start"; a second launch 34 s later found a 34-second-old A2_betapic (no
    final_results.json yet, so no skip) and started a second A2 into it -- two searches
    appending to one results.jsonl.  A pid file refuses the second instance, and no stage
    starts into a directory whose heartbeat is fresh."""
    sh = _text("rerun_paper.sh")
    assert 'PIDFILE=.rerun_paper.pid' in sh
    assert 'kill -0 "$(cat "$PIDFILE"' in sh and "already running" in sh
    assert "trap 'rm -f \"$PIDFILE\"' EXIT" in sh
    assert "live_age()" in sh and "heartbeat.json" in sh
    assert 'a run is LIVE in $d' in sh
    i_live = sh.index('a run is LIVE in $d')
    i_retire = sh.index('mv "$d" "$keep"')
    assert i_live < i_retire, "a live directory must be recognised before anything could retire it"
    assert "tail -f rerun_paper.log" in sh[:sh.index("set -uo pipefail")], "the header says how to follow a nohup'd run"


def test_calibration_cannot_run_away_on_a_small_ring():
    """Run A2's [6, 12] px annulus: three sources four FWHM apart on a 9-px ring, S/N ~ 0-1
    from 3e-5 to 76, and the old run searched at contrast 4.6e+03.  The Runner now asks the
    k-scan for a k that sees the sources before the contrast passes a tenth of the star, and
    failing that stops AT that cap and marks the annulus uncalibrated -- it does not raise,
    because a default configuration that cannot see an injection is a statement about the
    default, not about the problem."""
    from klip_tpe import CalibrationConfig
    assert CalibrationConfig().max_contrast == 0.1
    src = open(os.path.join(os.path.dirname(HERE), "klip_tpe", "runner.py")).read()
    assert "walking again" in src and "could NOT be calibrated" in src
    assert 'info["uncalibrated"]' in src


def test_the_nircam_stages_remeasure_each_trial(demos, monkeypatch, tmp_path):
    """run_D and H2 score each trial as the mean of 3 fresh draws, as the MIRI driver does.

    Not cosmetic: on MIRI a single draw scatters with sd 0.84 against a useful range of ~6,
    and a benchmark that cannot separate that is measuring its own noise rather than the two
    search strategies.  NIRCam reductions are ~2.2 s against MIRI's 22.6, so this is the
    cheap end of the trade.  Pinned because the setting is easy to lose in a refactor and its
    absence is invisible -- the run completes, the panels just have no error bars.
    """
    import inspect

    seen = {}
    monkeypatch.setattr(demos, "OUT", str(tmp_path))
    monkeypatch.setattr(demos, "hip65426_objects", lambda *a, **k: "JWST_REDUCER")
    monkeypatch.setattr(demos.generic, "default_config", lambda red, known, **kw: ("OBJ", "SAMP"))
    monkeypatch.setattr(demos.generic, "make_space", lambda red, **kw: _FakeSpace())
    monkeypatch.setattr(demos.generic, "make_guard", lambda red, **kw: None)
    monkeypatch.setattr(demos, "Runner", lambda *a, **kw: seen.update(cfg=a[4]) or "RUNNER")
    monkeypatch.setattr(demos, "preflight", lambda runner, what: None)

    import klip_tpe.bench as B
    monkeypatch.setattr(B, "run_benchmark",
                        lambda make_runner, modes, seeds, n_iter, n_init, bench_tag, out_dir, log:
                        make_runner("tpe", 0, out_dir))
    demos.run_H2()
    assert seen["cfg"].n_remeasure == 3, f"H2 got n_remeasure={seen['cfg'].n_remeasure}"

    # run_D builds its own RunConfig rather than going through _bench_hi
    assert "n_remeasure=3" in inspect.getsource(demos.run_D)

    # and the default stays 1, so E2/F2/G2 are untouched until they are re-run deliberately
    assert inspect.signature(demos._bench_hi).parameters["n_remeasure"].default == 1
