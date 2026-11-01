"""``RunConfig.score_floor`` (Kevin Wagner, 2026-11-01): a search trial's score, the mean of its
draws, is floored at 0 in new runs.  The draws keep their own scores, the calibration's revisit
reads the means, the validation's medians are not floored, and runs saved before resume without
a floor, as do the runs that mirror an IDL setup.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from klip_tpe import cli
from klip_tpe.contrast_grid import ContrastGridConfig
from klip_tpe.optimizers import History
from klip_tpe.runner import CalibrationConfig, EvalRecord, RunConfig, Runner, ValidationConfig

from conftest import build_synthetic_run


def _record(r: Runner, score, draws=None, raw=None) -> EvalRecord:
    x = [0.0] * r.space.ndim
    recs = []
    for s in (draws if draws is not None else [score]):
        recs.append(EvalRecord(annulus=0, index=-1, phase="tpe", x=x, config={}, sources=[(0.5, 10.0, 1e-3)],
                               score=s, raw_score=s if raw is None else raw, per_source=[s], raw_per_source=[s],
                               clean_per_source=None, partition_snr={}, k_used=5, contrast=1e-3, wall_s=1.0,
                               meta={"failed": s is None}))
    if draws is None:
        return recs[0]
    return Runner._combine_draws(recs, recs[-1])


def _runner(tmp_path, name="r", **kw) -> Runner:
    red, space, obj, samp, cfg = build_synthetic_run(write_setup_files=False, **kw)
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / name), log=lambda s: None)
    r.history = History(space.ndim)
    return r


def test_new_runs_floor_at_zero_and_saved_runs_do_not():
    assert RunConfig().score_floor == 0.0
    assert RunConfig.from_dict(RunConfig().to_dict()).score_floor == 0.0
    assert RunConfig.from_dict(RunConfig(score_floor=None).to_dict()).score_floor is None
    d = RunConfig().to_dict()
    d.pop("score_floor")
    assert RunConfig.from_dict(d).score_floor is None, "a run saved before 2026-11-01 resumes without a floor"


def test_a_run_that_mirrors_an_idl_setup_does_not_floor(tmp_path):
    from klip_tpe.idl_compare import matched_run_config
    setup = tmp_path / "run_setup.txt"
    setup.write_text("# KLIP-TPE run setup\nann_edges (px)     : 8.00, 30.00\nn_iter / n_init    : 6 / 3\n"
                     "n_valid / n_top    : 2 / 1\ngamma/ncand/explore: 0.250 / 48 / 0.15\n"
                     "p_local / n_elite  : 0.15 / 5\nseed / bench_tag   : 4242 / none\n")
    assert matched_run_config(str(setup), seed=1)[0].score_floor is None, "the IDL does not floor its scores"
    assert matched_run_config(str(setup), seed=1, score_floor=0.0)[0].score_floor == 0.0


def test_the_command_line_sets_or_clears_the_floor():
    p = cli._build_parser()
    assert "score_floor" not in cli._protocol_config(p.parse_args(["near", "--root", "/r"]))
    assert cli._protocol_config(p.parse_args(["near", "--root", "/r", "--score-floor", "off"]))["score_floor"] is None
    assert cli._protocol_config(p.parse_args(["near", "--root", "/r", "--score-floor", "0.5"]))["score_floor"] == 0.5
    with pytest.raises(SystemExit, match="a number or off"):
        cli._protocol_config(p.parse_args(["near", "--root", "/r", "--score-floor", "zero"]))


def test_a_mean_below_the_floor_enters_the_history_at_the_floor(tmp_path):
    r = _runner(tmp_path)
    rec = _record(r, None, draws=[-1.5, 0.5])            # the mean is -0.5, one draw is above 0
    assert rec.score == pytest.approx(-0.5), "draws are combined as before"
    i = r._append(rec)
    assert r.history.y[i] == 0.0 and rec.score == 0.0
    assert rec.meta["score_unfloored"] == pytest.approx(-0.5)
    assert r.history.flags[i]["unfloored"] == pytest.approx(-0.5)
    assert rec.meta["draw_scores"] == [-1.5, 0.5], "each draw keeps its own score"
    assert rec.raw_score == pytest.approx(-0.5), "the raw score is not floored"
    # a single draw is its own mean
    one = _record(r, -0.25)
    r._append(one)
    assert one.score == 0.0 and one.meta["score_unfloored"] == -0.25
    # above the floor, and a failed trial, as before
    up, failed = _record(r, 1.25, draws=[1.0, 1.5]), _record(r, None)
    r._append(up)
    r._append(failed)
    assert up.score == 1.25 and "score_unfloored" not in up.meta and "unfloored" not in r.history.flags[2]
    assert failed.score is None and np.isnan(r.history.y[3])
    assert list(r._unfloored_y()[:3]) == pytest.approx([-0.5, -0.25, 1.25])
    # results.jsonl holds the floored score and the mean beside it
    rows = [json.loads(ln) for ln in open(os.path.join(r.run_dir, "results.jsonl"))]
    assert [row["score"] for row in rows[:3]] == [0.0, 0.0, 1.25]
    assert rows[0]["meta"]["score_unfloored"] == pytest.approx(-0.5) and "score_unfloored" not in rows[2]["meta"]
    txt = open(os.path.join(r.run_dir, "results.txt")).read().splitlines()
    assert [ln.split()[-3] for ln in txt[1:4]] == ["0.0000", "0.0000", "1.2500"], "the score column"
    # a record is floored once: its mean stays the first one
    assert r._floor_score(rec).meta["score_unfloored"] == pytest.approx(-0.5)


def test_without_a_floor_the_history_keeps_the_means(tmp_path):
    r = _runner(tmp_path, score_floor=None)
    rec = _record(r, None, draws=[-1.5, 0.5])
    i = r._append(rec)
    assert r.history.y[i] == pytest.approx(-0.5) and "score_unfloored" not in rec.meta
    # a floor above 0 floors at its own value
    r2 = _runner(tmp_path, name="r2", score_floor=0.75)
    rec2 = _record(r2, 0.5)
    r2._append(rec2)
    assert rec2.score == 0.75 and rec2.meta["score_unfloored"] == 0.5


def test_the_calibration_revisit_reads_the_means(tmp_path):
    """The revisit's median of the top 4 of [6, 4, -1, -2, ...] is 1.5 on the means and would be
    2.0 on floored scores.  It reads the means, so its new contrast is the one it was."""
    out = []
    for floor in (0.0, None):
        r = _runner(tmp_path, name=f"r{floor}", score_floor=floor, seed_default=False,
                    calibration=CalibrationConfig(recal_check=6, recal_ntop=4))
        r.contrast = 1e-3
        for s in (6.0, 4.0, -1.0, -2.0, -3.0, -4.0):
            r._append(_record(r, s))
        out.append(r._recal_check(0, 5))
    assert out[0] == out[1] and out[0][0] == "reset" and out[0][1] == pytest.approx(1.5)


@pytest.fixture(scope="module")
def faint_run(tmp_path_factory):
    """Sources injected far below the noise, so that many trials' means fall below 0."""
    d = str(tmp_path_factory.mktemp("faint"))
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=[10], n_init=4, n_remeasure=2,
                                                     validation=ValidationConfig(n_top=1, n_valid=2),
                                                     param_verify=False, fm_curve=False, fm_preview=False,
                                                     contrast_grid=ContrastGridConfig(mode="off"),
                                                     calibration=CalibrationConfig(forced=[1e-7]))
    logs = []
    r = Runner(red, space, obj, samp, cfg, d, log=logs.append)
    r.run()
    return d, r, logs


@pytest.mark.slow
def test_a_run_floors_its_trials_and_not_their_draws(faint_run):
    d, r, logs = faint_run
    rows = [json.loads(ln) for ln in open(os.path.join(d, "results.jsonl"))]
    floored = [row for row in rows if "score_unfloored" in row["meta"]]
    assert floored, "the faint run has trials below 0"
    for row in rows:
        assert row["score"] is None or row["score"] >= 0.0
    for row in floored:
        draws = [v for v in row["meta"]["draw_scores"] if v is not None]
        assert row["score"] == 0.0 and row["meta"]["score_unfloored"] < 0.0
        assert row["meta"]["score_unfloored"] == pytest.approx(np.mean(draws)), "the mean of the draws"
    assert any(v < 0 for row in floored for v in row["meta"]["draw_scores"] if v is not None)
    assert any("floored from" in ln and "mean of 2" in ln for ln in logs)
    # the validation's medians are not floored
    val = json.load(open(os.path.join(d, "annulus01", "validation.json")))
    assert all(row["search_score"] >= 0.0 for row in val)
    # and the display says so
    from klip_tpe.display import _step_text, annulus_from_run
    ad = annulus_from_run(d, 0)
    k = next(j for j, row in enumerate(rows) if "score_unfloored" in row["meta"])
    text = "\n".join(_step_text(ad, k, 0, 0.0, 0.0))
    assert "floored from" in text and "  mean of 2 draws" in text and "= mean of" not in text
