"""End-to-end tests of the run protocol on the synthetic reducer.

All runs are tiny (2 partitions, <= 16 evaluations per annulus) so the whole file
takes a few tens of seconds.
"""
import json
import os

import numpy as np
import pytest

# end-to-end on synthetic data (~40 s on two cores); `-m \"not slow\"` skips it
pytestmark = pytest.mark.slow

from klip_tpe import (CalibrationConfig, MawetPeakSNR, Objective, Param, PositionSampler, RunConfig,
                      Runner, SearchSpace, ValidationConfig)
from klip_tpe.optimizers import History
from klip_tpe.runner import AnnulusResult, EvalRecord
from klip_tpe.space import kgrid
from klip_tpe.synthetic import SyntheticReducer

from conftest import build_synthetic_run

QUIET = (lambda s: None)


def _rows(path):
    return [l for l in open(path) if not l.startswith("#")]


def _strip_wall(line):
    return line.rsplit(None, 1)[0]


# ----------------------------------------------------------------------------
# full run: products
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("base"))
    red, space, obj, samp, cfg = build_synthetic_run()
    logs = []
    runner = Runner(red, space, obj, samp, cfg, d, log=logs.append)
    results = runner.run()
    return d, runner, results, logs


def test_full_run_products(base_run):
    d, runner, results, logs = base_run
    for f in ("results.txt", "results.jsonl", "checkpoint.json", "final_results.json",
              "run_setup.json", "run_setup.txt", "bench_summary.txt", "klip_stitched.fits"):
        assert os.path.exists(os.path.join(d, f)), f
    assert len(results) == 2
    for ia in range(2):
        ad = os.path.join(d, f"annulus{ia+1:02d}")
        for f in ("calibration.json", "validation.json", "winner.json", "best_clean.fits", "best_inj.fits",
                  "best_clean_partitions.fits", "contrast_curve.txt"):
            assert os.path.exists(os.path.join(ad, f)), (ia, f)
    fin = json.load(open(os.path.join(d, "final_results.json")))
    assert len(fin["annuli"]) == 2 and fin["seed"] == 11 and len(fin["stitch_weights"]) == 2
    assert fin["annuli"][0]["inrad"] == 8 and fin["annuli"][0]["outrad"] == 30
    assert fin["annuli"][1]["inrad"] == 30 and fin["annuli"][1]["outrad"] == 45
    # results.txt / jsonl are consistent with the histories
    rows = _rows(os.path.join(d, "results.txt"))
    recs = [json.loads(l) for l in open(os.path.join(d, "results.jsonl"))]
    assert len(rows) == len(recs) == runner.records_count
    assert all(r["annulus"] in (0, 1) for r in recs)
    last = [r for r in recs if r["annulus"] == 1]
    assert len(last) >= 14 and last[0]["phase"] == "seed" and last[-1]["index"] == 13
    assert {r["phase"] for r in recs} >= {"seed", "warmup"}
    assert {r["phase"] for r in recs} & {"tpe", "local", "explore"}
    # every record carries decoded config, sources and per-partition S/N
    r0 = last[-1]
    assert set(r0["config"]) == {"params", "per_partition", "selected", "x"}
    assert len(r0["sources"]) == 6 and set(r0["partition_snr"]) <= {"n1", "n2"}   # outer 2.25" -> 6 sources
    assert len([r for r in recs if r["annulus"] == 0][-1]["sources"]) == 3          # outer 1.5" -> 3
    assert r0["contrast"] == pytest.approx(results[1].contrast)
    assert isinstance(r0["k_used"], dict)
    setup = json.load(open(os.path.join(d, "run_setup.json")))
    assert setup["config"]["n_iter"] == [16, 14] and len(setup["space"]["params"]) == runner.space.ndim
    assert "run complete" in logs[-1]


def test_run_results_and_checkpoint_state(base_run):
    d, runner, results, _ = base_run
    for ia, r in enumerate(results):
        assert isinstance(r, AnnulusResult) and r.annulus == ia
        assert r.n_evaluations == [16, 14][ia]
        assert np.isfinite(r.winner_score) and r.contrast > 0
        assert len(r.winner_x) == runner.space.ndim
        assert set(r.distance_to_bounds) == set(runner.space.names)
        assert r.contrast_curve is not None and len(r.contrast_curve["r_as"]) > 0
        w = json.load(open(os.path.join(d, f"annulus{ia+1:02d}", "winner.json")))
        assert w["winner_index"] == r.winner_index and w["winner_score"] == pytest.approx(r.winner_score)
    ck = json.load(open(os.path.join(d, "checkpoint.json")))
    assert ck["annulus_done"] and ck["ia"] == 1 and len(ck["results"]) == 2
    assert ck["history"]["ndim"] == runner.space.ndim and len(ck["history"]["y"]) == 14
    assert "rng_state" in ck and ck["config"]["seed"] == 11


def test_validation_table_and_winner_election(base_run):
    d, runner, results, _ = base_run
    for ia, r in enumerate(results):
        table = json.load(open(os.path.join(d, f"annulus{ia+1:02d}", "validation.json")))
        assert len(table) == 2 == len(r.validation_table)         # n_top rows
        idx = [row["eval_index"] for row in table]
        assert len(set(idx)) == 2                                  # distinct configurations
        xs = [np.array(row["x"]) for row in table]
        assert runner.space.distinct(xs[0], xs[1])
        # candidates are the top of the search ordering, best first
        assert table[0]["search_score"] >= table[1]["search_score"]
        assert table[0]["search_score"] == pytest.approx(r.search_best_score)
        for row in table:
            assert len(row["trials"]) == 2                          # n_valid
            assert row["validated_score"] == pytest.approx(np.median(row["trials"]))
            assert 0 <= row["committed_trial"] < 2
        # winner = argmax validated score, its score is the validated one, not the search one
        best = max(table, key=lambda row: row["validated_score"])
        assert r.validated
        assert r.winner_index == best["eval_index"]
        assert r.winner_score == pytest.approx(best["validated_score"])
        assert np.allclose(r.winner_x, best["x"])
        assert r.winner_score != pytest.approx(r.search_best_score)  # search maxima are optimistic
        assert len(r.validation_samples["r_as"]) == 2 * [3, 6][ia]   # n_valid * n_sources


def test_validation_elects_lower_search_score_when_it_validates_better(tmp_path):
    """The election rule itself: forge a history in which the search best validates
    badly; the runner must pick the other candidate on the validated (raw) score."""
    from klip_tpe.metrics import ScoreResult
    from klip_tpe.reducer import EvalImages

    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=3,
                                                     validation=ValidationConfig(n_top=2, n_valid=3))
    runner = Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)
    runner.ia = 0
    runner._k_default = 5
    i_bin = space.index("bin_n1")

    # the "reduction" returns an image whose value is the config's bin_n1 ...
    def fake_reduce(cfg_, sources, k_scan=False, tag=""):
        return EvalImages(np.full((4, 4), float(cfg_.per_partition["n1"]["bin"])))
    runner._reduce = fake_reduce

    # ... and the raw metric returns that value (validated score == bin_n1)
    class FakeObjective:
        needs_clean = False
        metric = type("M", (), {"needs_clean": False})()

        def score_raw(self, img, sources, img_clean=None):
            v = np.full(len(sources), float(img[0, 0]))
            return ScoreResult(float(img[0, 0]), v, v, None)
    runner.objective = FakeObjective()

    runner.history = History(space.ndim)
    xa = space.default_vector({"bin_n1": 10})
    xb = space.default_vector({"bin_n1": 20})
    xc = space.default_vector({"bin_n1": 6})
    runner.history.append(xa, 100.0, {"phase": "seed"})     # search best, validates at 10
    runner.history.append(xb, 50.0, {"phase": "tpe"})       # validates at 20
    runner.history.append(xc, 75.0, {"phase": "tpe"})       # second-best search, validates at 6
    table, vwin, winner = runner.validate(0)
    assert [row["eval_index"] for row in table] == [0, 2]   # top-2 distinct by search score
    assert [row["validated_score"] for row in table] == [10.0, 6.0]
    assert vwin == 0 and winner["eval_index"] == 0 and winner["score"] == 10.0
    # now with n_top=3 the badly-searched-but-better-validated config wins
    runner.cfg.validation.n_top = 3
    table, vwin, winner = runner.validate(0)
    assert [row["eval_index"] for row in table] == [0, 2, 1]
    assert vwin == 2 and winner["eval_index"] == 1 and winner["score"] == 20.0
    assert winner["x"][i_bin] == 20.0
    # duplicates of an already-accepted vector are skipped when picking candidates
    runner.history.append(xa.copy(), 99.0, {"phase": "tpe"})
    table, _, _ = runner.validate(0)
    assert [row["eval_index"] for row in table] == [0, 2, 1]


# ----------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------
def test_calibration_converges_into_target_window(base_run):
    d, runner, results, _ = base_run
    cc = runner.cfg.calibration
    for ia in range(2):
        cal = json.load(open(os.path.join(d, f"annulus{ia+1:02d}", "calibration.json")))
        assert cal["forced"] == 0 and 1 <= len(cal["trials"]) <= cc.max_trials
        assert cc.target[0] <= cal["snr"] <= cc.target[1]
        assert cal["trials"][-1]["contrast"] == pytest.approx(cal["contrast"])
        assert len(cal["trials"][-1]["values"]) == cc.n_remeasure
        assert "kscan" in cal and 1 <= cal["k_default"] <= runner.cfg.k_scan_max
        # contrast moved away from contrast0 (3e-5 is deliberately too faint here)
        assert cal["contrast"] != pytest.approx(runner.cfg.contrast0)


def test_calibration_from_badly_off_contrast0(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run(contrast0=3e-3, ann_edges=[8, 30], n_iter=10,
                                                     validation=ValidationConfig(n_top=1, n_valid=1))
    runner = Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)
    contrast, kdef, info = runner.calibrate(0)
    assert len(info["trials"]) >= 3                       # needed several rescalings
    assert 4.0 <= info["snr"] <= 6.0
    assert contrast < 3e-3 / 5
    c = [t["contrast"] for t in info["trials"]]
    assert c[-1] < c[0] and c[1] < c[0]                    # descends from too bright
    assert all(0.1 <= b / a <= 10.0 for a, b in zip(c, c[1:]))   # step_clip
    # forced contrast bypasses the loop
    cfg.calibration.forced = [1e-4]
    runner2 = Runner(red, space, obj, samp, cfg, str(tmp_path / "f"), log=QUIET)
    contrast2, _, info2 = runner2.calibrate(0)
    assert contrast2 == 1e-4 and info2["forced"] == 1e-4 and len(info2["trials"]) == 1


def test_recalibration_revisit_triggers_when_contrast_is_off(tmp_path):
    # skip the calibration loop (max_trials=0) so the search starts 100x too bright
    red, space, obj, samp, cfg = build_synthetic_run(
        contrast0=3e-3, ann_edges=[8, 30], n_iter=12,
        calibration=CalibrationConfig(max_trials=0, recal_check=5, recal_ntop=3, recal_budget=2),
        validation=ValidationConfig(n_top=1, n_valid=1))
    logs = []
    runner = Runner(red, space, obj, samp, cfg, str(tmp_path), log=logs.append)
    res = runner.run()
    recal = [l for l in logs if "re-cal" in l]
    assert 1 <= len(recal) <= 2 and runner._recal_done == len(recal)
    assert res[0].contrast < 3e-3                          # was rescaled down
    assert res[0].contrast == pytest.approx(runner.contrast)
    # the aborted evaluations stay in results.txt: duplicate (annulus, iter) rows, last wins
    rows = _rows(tmp_path / "results.txt")
    keys = [tuple(r.split()[:2]) for r in rows]
    assert keys.count(("1", "1")) == len(recal) + 1
    assert len(rows) == 12 + len(recal) * 5
    recs = [json.loads(l) for l in open(tmp_path / "results.jsonl")]
    contrasts = sorted({r["contrast"] for r in recs}, reverse=True)
    assert contrasts[0] == pytest.approx(3e-3) and len(contrasts) == len(recal) + 1
    # the final history (checkpoint) holds only the restarted annulus
    ck = json.load(open(tmp_path / "checkpoint.json"))
    assert len(ck["history"]["y"]) == 12 and ck["recal_done"] == len(recal)


def test_no_recalibration_when_forced_or_budget_zero(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run(
        contrast0=3e-3, ann_edges=[8, 30], n_iter=8,
        calibration=CalibrationConfig(max_trials=0, recal_check=4, recal_budget=0),
        validation=ValidationConfig(n_top=1, n_valid=1))
    logs = []
    Runner(red, space, obj, samp, cfg, str(tmp_path / "a"), log=logs.append).run()
    assert not any("re-cal" in l for l in logs)
    assert len(_rows(tmp_path / "a" / "results.txt")) == 8
    cfg.calibration = CalibrationConfig(max_trials=0, recal_check=4, recal_budget=3, forced=[3e-3])
    logs = []
    Runner(red, space, obj, samp, cfg, str(tmp_path / "b"), log=logs.append).run()
    assert not any("re-cal" in l for l in logs)


# ----------------------------------------------------------------------------
# crash / resume
# ----------------------------------------------------------------------------
class Boom(Exception):
    pass


def test_crash_resume_reproduces_uninterrupted_run(tmp_path):
    full_dir, part_dir = str(tmp_path / "full"), str(tmp_path / "part")
    red, space, obj, samp, cfg = build_synthetic_run()
    full = Runner(red, space, obj, samp, cfg, full_dir, log=QUIET).run()

    # crash after 9 evaluations of annulus 1
    red, space, obj, samp, cfg = build_synthetic_run()
    r = Runner(red, space, obj, samp, cfg, part_dir, log=QUIET)
    orig = r.checkpoint

    def cp(final_annulus=False):
        orig(final_annulus)
        if r.ia == 0 and r.history is not None and len(r.history) == 9:
            raise Boom
    r.checkpoint = cp
    with pytest.raises(Boom):
        r.run()
    ck = json.load(open(os.path.join(part_dir, "checkpoint.json")))
    assert ck["ia"] == 0 and len(ck["history"]["y"]) == 9 and not ck["annulus_done"]

    # resume, crash again after 3 evaluations of annulus 2
    red, space, obj, samp, cfg = build_synthetic_run()
    logs = []
    r2 = Runner.resume(part_dir, red, obj, samp, log=logs.append)
    assert "resumed" in logs[0] and r2.ia == 0 and len(r2.history) == 9 and r2._resumed
    orig2 = r2.checkpoint

    def cp2(final_annulus=False):
        orig2(final_annulus)
        if r2.ia == 1 and r2.history is not None and len(r2.history) == 3:
            raise Boom
    r2.checkpoint = cp2
    with pytest.raises(Boom):
        r2.run()
    assert len(r2.results) == 1

    # resume once more and finish
    red, space, obj, samp, cfg = build_synthetic_run()
    r3 = Runner.resume(part_dir, red, obj, samp, log=QUIET)
    assert r3.ia == 1 and len(r3.history) == 3 and len(r3.results) == 1
    part = r3.run()

    a = [_strip_wall(l) for l in _rows(os.path.join(full_dir, "results.txt"))]
    b = [_strip_wall(l) for l in _rows(os.path.join(part_dir, "results.txt"))]
    assert len(a) == len(b) >= 30 and a == b                # >30 when re-calibration restarted an annulus
    ja = [json.loads(l) for l in open(os.path.join(full_dir, "results.jsonl"))]
    jb = [json.loads(l) for l in open(os.path.join(part_dir, "results.jsonl"))]
    for x, y in zip(ja, jb):
        for k in ("x", "score", "raw_score", "sources", "per_source", "contrast", "phase"):
            assert x[k] == y[k], k
    assert len(full) == len(part) == 2
    for x, y in zip(full, part):
        assert x.annulus == y.annulus
        assert x.winner_score == pytest.approx(y.winner_score)
        assert x.winner_x == y.winner_x and x.winner_index == y.winner_index
        assert x.contrast == pytest.approx(y.contrast)
        assert x.validation_table == y.validation_table
    fa = json.load(open(os.path.join(full_dir, "final_results.json")))
    fb = json.load(open(os.path.join(part_dir, "final_results.json")))
    assert fa["annuli"] == fb["annuli"]


def test_resume_after_completed_annulus_moves_on(tmp_path):
    d = str(tmp_path)
    red, space, obj, samp, cfg = build_synthetic_run()
    r = Runner(red, space, obj, samp, cfg, d, log=QUIET)
    r.run_annulus(0)                      # annulus 1 complete, checkpoint says annulus_done
    red, space, obj, samp, cfg = build_synthetic_run()
    r2 = Runner.resume(d, red, obj, samp, log=QUIET)
    assert r2.ia == 1 and r2.history is None and len(r2.results) == 1
    res = r2.run()
    assert len(res) == 2 and res[1].annulus == 1
    assert os.path.exists(os.path.join(d, "final_results.json"))


def test_resume_rejects_missing_checkpoint(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run()
    with pytest.raises(FileNotFoundError):
        Runner.resume(str(tmp_path), red, obj, samp, log=QUIET)


# ----------------------------------------------------------------------------
# other modes
# ----------------------------------------------------------------------------
def _scan_run(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run(k_mode="scan_rescore", ann_edges=[8, 30], n_iter=9,
                                                     n_init=4, k_scan_max=12,
                                                     calibration=CalibrationConfig(recal_budget=0),
                                                     validation=ValidationConfig(n_top=2, n_valid=1))
    runner = Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)
    return runner, runner.run()


def test_scan_rescore_mode_runs(tmp_path):
    runner, res = _scan_run(tmp_path)
    assert len(res) == 1 and np.isfinite(res[0].winner_score)
    recs = [json.loads(l) for l in open(tmp_path / "results.jsonl")]
    assert len(recs) == 9
    ok = [r for r in recs if not r["meta"]["failed"]]
    assert len(ok) >= 4                                        # seed + tied warm-up rows at least
    for r in ok:
        assert isinstance(r["k_used"], int) and 1 <= r["k_used"] <= 30
        # the chosen k is written back into the evaluated config for every partition
        for pid in r["config"]["selected"]:
            assert r["config"]["per_partition"][pid]["k_klip"] == r["k_used"]
    # validation used the rescored k, not the raw vector's k
    for row in res[0].validation_table:
        k_hist = runner.history.flags[row["eval_index"]]["k"]
        assert isinstance(k_hist, int)
        assert all(v["k_klip"] == k_hist for v in row["config"]["per_partition"].values())


def test_scan_rescore_untied_proposals_do_not_fail(tmp_path):
    runner, res = _scan_run(tmp_path)
    recs = [json.loads(l) for l in open(tmp_path / "results.jsonl")]
    assert not any(r["meta"]["failed"] for r in recs)


@pytest.mark.parametrize("mode", ["grid", "random"])
def test_grid_and_random_search_modes(tmp_path, mode):
    red, space, obj, samp, cfg = build_synthetic_run(search_mode=mode, ann_edges=[8, 30], n_iter=10,
                                                     calibration=CalibrationConfig(recal_budget=0),
                                                     validation=ValidationConfig(n_top=2, n_valid=1),
                                                     grid_axes=["bin_n1", "k_klip_n1"] if mode == "grid" else None)
    runner = Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)
    res = runner.run()
    assert len(res) == 1 and len(runner.history) == 10
    phases = [f["phase"] for f in runner.history.flags]
    assert phases[0] == "seed" and set(phases[1:]) == {mode}
    if mode == "grid":
        cells = [f.get("cell") for f in runner.history.flags[1:]]
        i1, i2 = runner.space.index("bin_n1"), runner.space.index("k_klip_n1")
        X = runner.history.X[1:]
        assert len({(x[i1], x[i2]) for x in X}) == 9         # 3x3 grid, distinct cells
        # link_params tie the bin slots even in grid mode
        assert np.all(X[:, i1] == X[:, runner.space.index("bin_n2")])
    else:
        assert any(f["phase"] == "random" for f in runner.history.flags)
    assert os.path.exists(tmp_path / "final_results.json")


def test_batch_mode(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run(batch=3, ann_edges=[8, 30], n_iter=11, n_init=4,
                                                     calibration=CalibrationConfig(recal_budget=0),
                                                     validation=ValidationConfig(n_top=1, n_valid=1))
    runner = Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)
    res = runner.run()
    assert len(runner.history) == 11                          # never over-runs n_iter
    phases = [f["phase"] for f in runner.history.flags]
    assert phases[0] == "seed" and phases[1:4] == ["warmup"] * 3
    assert "warmup" not in phases[4:]
    assert len(_rows(tmp_path / "results.txt")) == 11 and np.isfinite(res[0].winner_score)


def test_single_partition_reducer_and_no_validation(tmp_path):
    red = SyntheticReducer(k_opt=8, seed=3)
    space = SearchSpace([Param("bin", 5, 30, "int", default=12),
                         Param("k_klip", 1, 30, "int", grid=kgrid(30), default=5)])
    obj = Objective(MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm), clean_subtract=False)
    samp = PositionSampler(fwhm_as=red.fwhm * red.pxscale)
    cfg = RunConfig(ann_edges=[8, 30], n_iter=8, n_init=3, seed=5, save_fits=False,
                    validation=ValidationConfig(n_top=0, n_valid=0),
                    calibration=CalibrationConfig(recal_budget=0, scan_k=False, n_remeasure=1))
    runner = Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)
    res = runner.run()
    assert len(res) == 1 and not res[0].validated
    assert res[0].winner_index == res[0].search_best_index
    assert res[0].winner_score == pytest.approx(res[0].search_best_score)
    assert res[0].validation_table == []
    assert not os.path.exists(tmp_path / "annulus01" / "best_clean.fits")
    recs = [json.loads(l) for l in open(tmp_path / "results.jsonl")]
    assert all(r["partition_snr"] == {} and r["clean_per_source"] is None for r in recs)
    assert all(isinstance(r["k_used"], int) for r in recs)


def test_failed_evaluation_is_recorded_not_stale(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=4, n_init=2,
                                                     validation=ValidationConfig(n_top=1, n_valid=1))
    runner = Runner(red, space, obj, samp, cfg, str(tmp_path), log=QUIET)
    runner.ia = 0
    runner._k_default = 5
    orig = runner._reduce
    calls = {"n": 0}

    def flaky(cfg_, sources, k_scan=False, tag=""):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated reducer crash")
        return orig(cfg_, sources, k_scan, tag)
    runner._reduce = flaky
    rec, inj, clean = runner.evaluate(space.default_vector(), "test")
    assert isinstance(rec, EvalRecord) and rec.score is None and rec.meta["failed"]
    assert rec.per_source == [] and np.isnan(inj.image).all()
    rec2, _, _ = runner.evaluate(space.default_vector(), "test")
    assert rec2.score is not None and not rec2.meta["failed"]


def test_runconfig_round_trip_and_helpers():
    cfg = RunConfig(ann_edges=[5, 10, 20], n_iter=[3, 4], n_init=2)
    d = cfg.to_dict()
    cfg2 = RunConfig.from_dict(json.loads(json.dumps(d)))
    assert cfg2.n_iter == [3, 4] and cfg2.calibration.target == [4.0, 6.0] or cfg2.calibration.target == (4.0, 6.0)
    assert cfg2.per_annulus(cfg2.n_iter, 0) == 3 and cfg2.per_annulus(cfg2.n_iter, 5) == 4
    assert cfg2.per_annulus(7, 3) == 7 and cfg2.nann == 2


def test_validation_resumes_per_candidate(tmp_path):
    """An interruption during validation must not redo finished candidates: the
    resumed run reloads candidate 1 from val_cand01.pkl / validation.json, validates
    only candidate 2, and ends with the same winner as an uninterrupted run."""
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=8, n_init=3, seed=3,
                                                     save_fits=False, verify=False, candidates=False)
    d1 = str(tmp_path / "ref")
    ref = Runner(red, space, obj, samp, cfg, d1, log=QUIET)
    ref.run()
    ref_val = json.load(open(os.path.join(d1, "annulus01", "validation.json")))

    class Interrupt(BaseException):      # like Ctrl-C: not swallowed by the per-trial guards
        pass

    d2 = str(tmp_path / "cut")
    r = Runner(red, space, obj, samp, cfg, d2, log=QUIET)
    orig = r._reduce
    calls = {"val": 0}

    def cut(cfg_, src, tag="", **kw):
        if tag.startswith("a1_val1_"):          # second candidate: interrupt on first touch
            raise Interrupt
        return orig(cfg_, src, tag=tag, **kw)
    r._reduce = cut
    with pytest.raises(Interrupt):
        r.run()
    assert os.path.exists(os.path.join(d2, "annulus01", "val_cand01.pkl"))
    logs = []
    r2 = Runner.resume(d2, red, obj, samp, log=logs.append)
    r2.run()
    assert any("[resumed]" in l for l in logs)
    cut_val = json.load(open(os.path.join(d2, "annulus01", "validation.json")))
    assert [row["eval_index"] for row in cut_val] == [row["eval_index"] for row in ref_val]
    assert cut_val[0]["trials"] == ref_val[0]["trials"]
    assert np.allclose([row["validated_score"] for row in cut_val], [row["validated_score"] for row in ref_val])
    assert r2.results[0].to_dict()["winner_index"] == ref.results[0].to_dict()["winner_index"]


def test_validation_resumes_mid_candidate(tmp_path):
    """An interruption INSIDE a candidate's trials resumes after the last finished
    trial (val_candNN_trials.pkl + checkpointed RNG): the trials, validated scores and
    winner are identical to an uninterrupted run and no trial is re-reduced."""
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=8, n_init=3, seed=3,
                                                     save_fits=False, verify=False, candidates=False,
                                                     validation=ValidationConfig(n_top=2, n_valid=4))
    d1 = str(tmp_path / "ref")
    ref = Runner(red, space, obj, samp, cfg, d1, log=QUIET)
    ref.run()
    ref_val = json.load(open(os.path.join(d1, "annulus01", "validation.json")))

    class Interrupt(BaseException):
        pass

    d2 = str(tmp_path / "cut")
    r = Runner(red, space, obj, samp, cfg, d2, log=QUIET)
    orig = r._reduce

    def cut(cfg_, src, tag="", **kw):
        if tag == "a1_val1_t2":                 # second candidate, third trial
            raise Interrupt
        return orig(cfg_, src, tag=tag, **kw)
    r._reduce = cut
    with pytest.raises(Interrupt):
        r.run()
    assert os.path.exists(os.path.join(d2, "annulus01", "val_cand02_trials.pkl"))
    logs, tags = [], []
    r2 = Runner.resume(d2, red, obj, samp, log=logs.append)
    orig2 = r2._reduce

    def spy(cfg_, src, tag="", **kw):
        tags.append(tag)
        return orig2(cfg_, src, tag=tag, **kw)
    r2._reduce = spy
    r2.run()
    assert any("resuming after trial 2" in l for l in logs)
    assert "a1_val1_t0" not in tags and "a1_val1_t1" not in tags and "a1_val1_clean" not in tags
    assert not os.path.exists(os.path.join(d2, "annulus01", "val_cand02_trials.pkl"))
    cut_val = json.load(open(os.path.join(d2, "annulus01", "validation.json")))
    for a, b in zip(cut_val, ref_val):
        assert a["trials"] == b["trials"]
    assert r2.results[0].to_dict()["winner_index"] == ref.results[0].to_dict()["winner_index"]


def test_post_annulus_hooks_resume(tmp_path):
    """Hooks interrupted after the annulus checkpoint are finished on resume (not
    skipped, not repeated): verify is recorded as done, param_verify is re-run from
    its reduction cache, candidates run once."""
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=8, n_init=3, seed=4,
                                                     save_fits=True, verify=True, param_verify=True,
                                                     n_pv=3, candidates=True)
    d = str(tmp_path / "hooks")
    r = Runner(red, space, obj, samp, cfg, d, log=QUIET)

    class Interrupt(BaseException):
        pass
    orig = r._reduce
    n = {"pv": 0}

    def cut(cfg_, src, tag="", **kw):
        if tag.startswith("a1_pv_"):
            n["pv"] += 1
            if n["pv"] == 3:                 # inside param_verify, after one cached config
                raise Interrupt
        return orig(cfg_, src, tag=tag, **kw)
    r._reduce = cut
    with pytest.raises(Interrupt):
        r.run()
    ck = json.load(open(os.path.join(d, "checkpoint.json")))
    assert ck["hooks_done"] == {"0": ["stitch", "verify"]}
    assert len([f for f in os.listdir(os.path.join(d, "annulus01", "param_verify")) if f.startswith("cache_")]) == 1
    logs = []
    r2 = Runner.resume(d, red, obj, samp, log=logs.append)
    calls = {"pv": 0}
    orig2 = r2._reduce

    def count(cfg_, src, tag="", **kw):
        if tag.startswith("a1_pv_"):
            calls["pv"] += 1
        return orig2(cfg_, src, tag=tag, **kw)
    r2._reduce = count
    r2.run()
    assert any("finishing interrupted hooks ['param_verify', 'candidates']" in l for l in logs)
    ck = json.load(open(os.path.join(d, "checkpoint.json")))
    assert ck["hooks_done"]["0"] == ["stitch", "verify", "param_verify", "candidates"]
    nsel = len([f for f in os.listdir(os.path.join(d, "annulus01", "param_verify")) if f.startswith("cache_")])
    assert nsel >= 2 and calls["pv"] == 2 * (nsel - 1)      # the cached config was not re-reduced
    assert os.path.exists(os.path.join(d, "annulus01", "param_verify")) and r2._pv_results.get(0) is not None


def test_runner_band_two_source_area_midpoint():
    """Runner._band applies the area-weighted mid radius only for two sources and only when
    ``pair_area_midpoint`` is on; the IWA clamp is left to the placement routine."""
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=4, n_init=2, n_sources=2)
    r = Runner(red, space, obj, samp, cfg, "/tmp/_band_test_a", log=QUIET)
    lo, hi = r._band(0)
    r_area = np.sqrt(0.5 * (8 ** 2 + 30 ** 2))
    assert np.isclose(lo, hi) and np.isclose(lo, r_area * red.pxscale)
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=4, n_init=2, n_sources=2,
                                                     pair_area_midpoint=False)
    r = Runner(red, space, obj, samp, cfg, "/tmp/_band_test_b", log=QUIET)
    lo, hi = r._band(0)
    assert np.isclose(lo, (8 + red.fwhm) * red.pxscale) and np.isclose(hi, (30 - red.fwhm) * red.pxscale)
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=4, n_init=2, n_sources=3)
    r = Runner(red, space, obj, samp, cfg, "/tmp/_band_test_c", log=QUIET)
    lo, hi = r._band(0)
    assert hi > lo


def test_opt_width_stops_when_remaining_ring_below_w_lo():
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8], n_iter=3, n_init=2, opt_width=True,
                                                     width_range=(12, 20), r_cap=30, save_fits=False,
                                                     validation=ValidationConfig(n_top=1, n_valid=1))
    space.add(Param("width", 12, 20, "int", default=15))
    r = Runner(red, space, obj, samp, cfg, "/tmp/_ow_guard", log=QUIET)
    r.cfg.ann_edges = [8.0, 22.0]          # committed first annulus: 8 px remain (< w_lo = 12)
    assert r._done(1)
    r.cfg.ann_edges = [8.0, 17.0]          # 13 px remain (>= w_lo): another annulus is due
    assert not r._done(1)
