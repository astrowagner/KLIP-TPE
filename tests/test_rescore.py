"""The re-score, the first piece of ``refine`` (NEAR2's notes of 2026-11-06 to 2026-11-09).

What must hold:

* A run's candidates for an annulus are its validation's entrants, the elected one marked the
  winner, and with ``top`` the best of its history not among them, each configuration once.
* The base winner's S/N on a draw is what ``rereduce`` with that draw's sources gives in its
  stitched verify: the stitch is built as the final stitch builds it, every annulus the sources
  reach is reduced with them, and each source is measured as the stitched verify measures it.
* The base winner's paired difference from itself is 0, every candidate of both runs gets a
  rank, the tiles' contrasts put the base winners' median S/N near the target, and the reports
  say so.
* The runs are only read, and a re-score started again on its directory resumes without
  scoring a candidate twice.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from klip_tpe.refine import (Entrant, StitchScorer, config_key, paired_summary, parse_annuli, rescore,
                             run_entrants)
from klip_tpe.runner import CalibrationConfig, Runner, ValidationConfig

from conftest import build_synthetic_run

QUIET = (lambda s: None)
EDGES = [8, 20, 32, 44]


def _make_run(path, seed):
    red, space, obj, samp, cfg = build_synthetic_run(
        ann_edges=list(EDGES), n_iter=[5], n_init=3, seed=seed, param_verify=False, stitch_every=0, verify=False,
        write_setup_files=False, fm_curve=False, validation=ValidationConfig(n_top=2, n_valid=1),
        calibration=CalibrationConfig(forced=[2e-3, 2e-3, 2e-3]))
    Runner(red, space, obj, samp, cfg, str(path), log=QUIET).run()
    return str(path)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("rescore")
    return _make_run(base / "run_a", 11), _make_run(base / "run_b", 23)


def _resume(run_dir):
    red, space, obj, samp, _ = build_synthetic_run()
    return Runner.resume(run_dir, red, obj, samp, log=QUIET, same_frames=False)


def _snapshot(d):
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            st = os.stat(p)
            out[os.path.relpath(p, d)] = (st.st_size, st.st_mtime_ns)
    return out


def test_annuli_and_paired_summaries():
    assert parse_annuli("1-3", 7) == [0, 1, 2] and parse_annuli("2,5", 7) == [1, 4] and parse_annuli("all", 3) == [0, 1, 2]
    with pytest.raises(ValueError, match="not among"):
        parse_annuli("8", 7)
    s = paired_summary(np.array([6.0, 8.0, 7.0, np.nan]), np.array([5.0, 6.0, 7.0, 6.0]))
    assert s["median"] == 7.0 and s["n"] == 3 and s["n_paired"] == 3
    assert s["diff_mean"] == pytest.approx(1.0) and s["p_better"] == pytest.approx(2 / 3)
    assert s["diff_se"] == pytest.approx(np.std([1.0, 2.0, 0.0], ddof=1) / np.sqrt(3))


@pytest.mark.slow
def test_a_runs_candidates(runs):
    a, _ = runs
    ents = run_entrants(a, 1)
    fr = json.load(open(os.path.join(a, "final_results.json")))
    ann = fr["annuli"][1]
    assert [e.kind for e in ents].count("winner") == 1
    win = next(e for e in ents if e.kind == "winner")
    assert win.key == config_key(ann["winner_config"]) and win.run == "run_a"
    assert len({e.key for e in ents}) == len(ents) == len({config_key(r["config"]) for r in ann["validation_table"]})
    more = run_entrants(a, 1, top=3)
    assert more[:len(ents)] == ents and 0 < len(more) - len(ents) <= 3
    assert all(e.kind == "top" and e.search_score is not None for e in more[len(ents):])


@pytest.mark.slow
def test_the_base_winner_reads_what_rereduce_gives(runs, tmp_path):
    a, _ = runs
    r = _resume(a)
    sc = StitchScorer(r, log=QUIET)
    sc.prepare(1)
    k = 1
    rng = np.random.default_rng(5)
    contrasts = {j: 4e-3 for j in sc.order}
    d = sc.draw([0, 1, 2], contrasts, rng)
    src = [s for _, s in d]
    assert sorted({j for j, _ in d}) == [0, 1, 2] and sc.affected(src, k) == [0, 1, 2]
    ours = sc.score_draws([d], k, window=1)[0]
    rr = _resume(a)
    rr.rereduce(str(tmp_path / "re"), verify_sources=[(s.rho, s.theta, s.contrast) for s in src], log=QUIET)
    got = []
    for ln in open(os.path.join(str(tmp_path / "re"), "stitched_verify_report.txt")):
        if ln.startswith("   rho="):
            got.append(float(ln.split("S/N=")[1].split()[0]))
    assert len(got) == len(src)
    assert np.allclose(ours, got, atol=0.006), (ours, got)        # the report prints 2 decimals


@pytest.mark.slow
def test_a_rescore_ranks_both_runs_candidates_and_resumes(runs, tmp_path):
    a, b = runs
    before = {d: _snapshot(d) for d in (a, b)}
    out = str(tmp_path / "rs")
    r = _resume(a)
    plan = rescore(r, [b], out, annuli=[1], n_draws=3, n_cal=2, max_iter=2, window=2, plan=True, log=QUIET)
    assert plan["plan"]["annuli"][1]["entrants"] >= 2 and not os.path.exists(out)
    rep = rescore(_resume(a), [b], out, annuli=[1], n_draws=3, n_cal=2, max_iter=2, window=2, log=QUIET)
    ann = rep["annuli"]["2"]
    names = {e["entrant"]["run"] for e in ann["entrants"]}
    assert names == {"run_a", "run_b"}
    base = [e for e in ann["entrants"] if e["base"]]
    assert len(base) == 1 and base[0]["entrant"]["run"] == "run_a" and base[0]["entrant"]["kind"] == "winner"
    assert base[0]["all"]["diff_mean"] == 0.0 and base[0]["all"]["p_better"] == 0.0
    assert sorted(e["rank"] for e in ann["entrants"]) == list(range(1, len(ann["entrants"]) + 1))
    assert ann["base_rank"] == base[0]["rank"] and ann["draw_tiles"] == [1, 2, 3]
    assert all(len(e["snr"]) == 3 for e in ann["entrants"])
    assert set(rep["calibration"]) == {"1", "2", "3"}
    text = open(os.path.join(out, "rescore.txt")).read()
    assert "== annulus 2" in text and "run_b" in text and "*" in text
    assert {d: _snapshot(d) for d in (a, b)} == before, "the runs are only read"
    # started again: the draws, the calibration and every candidate come from the state
    rep2 = rescore(_resume(a), [b], out, annuli=[1], n_draws=3, n_cal=2, max_iter=2, window=2, log=QUIET)
    assert rep2["reductions"] == len(EDGES) - 1, "only the base winners' clean images again"
    assert [e["all"] for e in rep2["annuli"]["2"]["entrants"]] == [e["all"] for e in ann["entrants"]]
    with pytest.raises(ValueError, match="holds 3 draws"):
        rescore(_resume(a), [b], out, annuli=[1], n_draws=4, n_cal=2, max_iter=2, window=2, log=QUIET)


@pytest.mark.slow
def test_runs_of_another_setup_are_refused(runs, tmp_path):
    a, _ = runs
    other = _make_run(tmp_path / "run_c", 5)
    fr_path = os.path.join(other, "final_results.json")
    fr = json.load(open(fr_path))
    fr["annuli"][0]["inrad"] = 9.0
    json.dump(fr, open(fr_path, "w"))
    with pytest.raises(ValueError, match="annuli"):
        rescore(_resume(a), [other], str(tmp_path / "x"), annuli=[1], n_draws=2, plan=True, log=QUIET)


def test_the_calibration_steps():
    step = StitchScorer._next_contrast
    # one step: scaled to the target's middle
    assert step([{"contrast": 2e-3, "median_snr": 52.0}], (5.0, 8.0)) == pytest.approx(2e-3 * 6.5 / 52.0)
    assert step([{"contrast": 1e-4, "median_snr": float("nan")}], (5.0, 8.0)) == pytest.approx(3e-4)
    # once bracketed: between the nearest two, in log contrast
    h = [{"contrast": 2e-3, "median_snr": 50.0}, {"contrast": 1e-4, "median_snr": 4.0},
         {"contrast": 4e-4, "median_snr": 12.0}]
    want = np.exp(np.log(1e-4) + (6.5 - 4.0) / (12.0 - 4.0) * (np.log(4e-4) - np.log(1e-4)))
    assert step(h, (5.0, 8.0)) == pytest.approx(want)


def test_an_entrant_tag_is_stable():
    e = Entrant("run_a", "winner", 3, {"params": {"bin": 12}, "per_partition": {}, "selected": ["n1"]})
    f = Entrant("run_a", "validation", 4, {"selected": ["n1"], "per_partition": {}, "params": {"bin": 12}})
    assert e.key == f.key and e.tag == f.tag and len(e.tag) == 12
    g = Entrant("run_b", "winner", 3, dict(e.config))
    assert g.key == e.key and g.tag != e.tag, "the same configuration from another run is another entrant's file"
