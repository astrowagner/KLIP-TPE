"""The re-score's tiles apart, a composite base and draws from another re-score (NEAR2,
2026-11-10).

NEAR2's re-score of s7 set each tile's contrast with that tile's sources alone, then put all
three tiles' sources in every reduction.  Tile 2's sources were 7 and 28 times as bright as
tiles 3 and 4's, and s7's winner read 3.47 and 1.77 there against the calibration's 5 to 8.
They then asked for a composite pass: each annulus at its chosen winner, scored again.  What
must hold:

* ``--tiles apart`` reduces each tile's sources on their own: no injected reduction holds two
  tiles' sources, and a draw's S/N is each tile's sub-draw scored as a draw of its own.  A new
  directory defaults to apart, one started before keeps together, and ``--draws-from`` takes
  that re-score's.
* A pick file names an annulus, a run and an evaluation counted from 0.  The run is a
  directory, a known run's name or a run beside the base run.  ``entrant_at`` finds the
  validation's entrant, the winner or the history's configuration.
* A composite base reads what ``rereduce`` with the same winners gives in its stitched verify.
  Its base winners are marked, the base run's own winner of those annuli enters as a candidate,
  and a directory refuses other base winners or another tiles mode when started again.
* ``--draws-from`` takes the draws and contrasts without a calibration and compares the bases
  draw for draw, which on the same base and mode is exactly 0.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from klip_tpe.refine import (Pick, StitchScorer, _subdraws, config_key, entrant_at, read_picks, rescore,
                             rescore_plan, resolve_picks, run_entrants)
from klip_tpe.runner import CalibrationConfig, Runner, ValidationConfig

from conftest import build_synthetic_run

QUIET = (lambda s: None)
EDGES = [8, 20, 32, 44]
FAST = dict(n_cal=2, max_iter=1, window=2, log=QUIET)


def _make_run(path, seed):
    red, space, obj, samp, cfg = build_synthetic_run(
        ann_edges=list(EDGES), n_iter=[5], n_init=3, seed=seed, param_verify=False, stitch_every=0, verify=False,
        write_setup_files=False, fm_curve=False, validation=ValidationConfig(n_top=2, n_valid=1),
        calibration=CalibrationConfig(forced=[2e-3, 2e-3, 2e-3]))
    Runner(red, space, obj, samp, cfg, str(path), log=QUIET).run()
    return str(path)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("composite")
    return _make_run(base / "run_a", 11), _make_run(base / "run_b", 23)


def _resume(run_dir):
    red, space, obj, samp, _ = build_synthetic_run()
    return Runner.resume(run_dir, red, obj, samp, log=QUIET, same_frames=False)


def _winner_eval(run_dir, k):
    fr = json.load(open(os.path.join(run_dir, "final_results.json")))
    return int(fr["annuli"][k]["winner_index"])


def _winner_key(run_dir, k):
    fr = json.load(open(os.path.join(run_dir, "final_results.json")))
    return config_key(fr["annuli"][k]["winner_config"])


def _other_eval(run_dir, other_run, k):
    """An evaluation of ``run_dir``'s annulus ``k`` whose configuration is not ``other_run``'s winner."""
    avoid = _winner_key(other_run, k)
    for e in run_entrants(run_dir, k):
        if e.eval_index is not None and e.key != avoid:
            return int(e.eval_index)
    for ln in open(os.path.join(run_dir, "results.jsonl")):
        r = json.loads(ln)
        if r["annulus"] == k and r.get("config") and config_key(r["config"]) != avoid:
            return int(r["index"])
    raise AssertionError("every configuration is the other run's winner")


def _spy(monkeypatch, sink):
    real = StitchScorer.reduce

    def reduce(self, ia, cfg, sources, tag):
        sink.append((ia, tuple(sorted({round(float(s.rho), 9) for s in sources})) if sources else (), tag))
        return real(self, ia, cfg, sources, tag)
    monkeypatch.setattr(StitchScorer, "reduce", reduce)


def _tile_of(sc, rho_as):
    r = float(rho_as) / float(sc.r.pxscale)
    return next(j for j in sc.order if sc.tile(j)[0] - 1e-6 <= r <= sc.tile(j)[1] + 1e-6)


# ---------------------------------------------------------------------------- pick files
@pytest.mark.slow
def test_pick_files_and_the_entrants_they_name(runs, tmp_path):
    a, b = runs
    ents = run_entrants(b, 1)
    ev = next(e.eval_index for e in ents if e.eval_index is not None)
    f = tmp_path / "picks.txt"
    f.write_text("# annulus  run  eval (from 0)\n"
                 f"2  run_b  {ev}   # by name, a run of --entrants-from\n"
                 f"1  {b}  0\n"
                 f"3  run_b  1\n")
    picks = read_picks(str(f), a, known=[b], nann=3)
    assert picks == [Pick(1, b, ev), Pick(0, b, 0), Pick(2, b, 1)]
    # a run beside the base run, though not named among the known ones; and JSON
    j = tmp_path / "picks.json"
    j.write_text(json.dumps([{"annulus": 2, "run": "run_b", "eval": ev}]))
    assert read_picks(str(j), a) == [Pick(1, b, ev)]
    for text, msg in (("2 run_b\n", "give the annulus, the run and the evaluation"),
                      ("4 run_b 0\n", "not among the run's 1-3"),
                      ("2 run_b -1\n", "count from 0"),
                      ("2 run_x 0\n", "not a directory"),
                      ("# nothing\n", "lists nothing")):
        f.write_text(text)
        with pytest.raises(ValueError, match=msg):
            read_picks(str(f), a, known=[b], nann=3)
    f.write_text(f"2 run_b {ev}\n2 run_a 0\n")
    with pytest.raises(ValueError, match="annulus 2 again"):
        read_picks(str(f), a, nann=3, one_per_annulus=True)
    assert len(read_picks(str(f), a, nann=3)) == 2, "candidates may list an annulus twice"
    # what each names: the validation's entrant, the winner, or the history's configuration
    e = entrant_at(b, 1, ev)
    assert e.kind in ("validation", "winner") and e.key == next(x.key for x in ents if x.eval_index == ev)
    w = entrant_at(b, 1, _winner_eval(b, 1))
    assert w.kind == "winner"
    listed = {x.eval_index for x in ents}
    hist = next(i for i in range(5) if i not in listed)
    h = entrant_at(b, 1, hist)
    row = next(json.loads(ln) for ln in open(os.path.join(b, "results.jsonl"))
               if json.loads(ln)["annulus"] == 1 and json.loads(ln)["index"] == hist)
    assert h.kind == "history" and h.config == row["config"] and h.run == "run_b"
    with pytest.raises(ValueError, match="no evaluation 999"):
        entrant_at(b, 1, 999)
    assert [x.key for _, x in resolve_picks(a, [Pick(1, b, ev)])] == [e.key]


# ---------------------------------------------------------------------------- tiles apart
@pytest.mark.slow
def test_apart_reduces_each_tile_on_its_own(runs, tmp_path, monkeypatch):
    a, _ = runs
    sc = StitchScorer(_resume(a), log=QUIET)
    sc.prepare(1)
    k = 1
    d = sc.draw([0, 1, 2], {j: 4e-3 for j in sc.order}, np.random.default_rng(5))
    subs = _subdraws(d, True)
    assert [j for _, _, j in subs] == [0, 1, 2] and _subdraws(d, False)[0][2] is None
    # a draw scored apart is each tile's sub-draw scored as a draw of its own
    apart = sc.score_draws([d], k, window=1, apart=True)[0]
    each = np.full(len(d), np.nan)
    for idx, src, j in subs:
        each[idx] = sc.score_draws([[(j, s) for s in src]], k, window=1)[0]
    assert np.allclose(apart, each, equal_nan=True)
    # the re-score: no injected reduction holds two tiles' sources, and the candidate is reduced
    # once for each tile whose sources reach its zone
    seen = []
    _spy(monkeypatch, seen)
    out = str(tmp_path / "apart")
    rep = rescore(_resume(a), [], out, annuli=[k], n_draws=2, **FAST)
    assert rep["tiles"] == "apart" and json.load(open(os.path.join(out, "state", "options.json")))["tiles"] == "apart"
    sc2 = StitchScorer(_resume(a), log=QUIET)
    inj = [(ia, rs, tag) for ia, rs, tag in seen if rs and "_d" in tag]
    assert inj and all(len({_tile_of(sc2, r) for r in rs}) == 1 for _, rs, _ in inj)
    n_ent = len(rep["annuli"]["2"]["entrants"])
    cand = [x for x in inj if "_c_d" in x[2]]
    assert len(cand) == n_ent * 2 * 3, "three tiles' parts per draw reach annulus 2"
    assert "# tiles apart" in open(os.path.join(out, "rescore.txt")).read()
    assert "the base winner per tile, median S/N: tile 1" in open(os.path.join(out, "rescore.txt")).read()
    base = next(e for e in rep["annuli"]["2"]["entrants"] if e["base"])
    assert set(base["tiles"]) == {"1", "2", "3"} and base["tiles"]["2"]["diff_mean"] == 0.0
    # together: one reduction per draw and annulus, with every tile's sources
    seen.clear()
    rep2 = rescore(_resume(a), [], str(tmp_path / "together"), annuli=[k], n_draws=2, tiles="together", **FAST)
    assert rep2["tiles"] == "together"
    cand2 = [x for x in seen if "_c_d" in x[2]]
    assert len(cand2) == n_ent * 2 and all(len({_tile_of(sc2, r) for r in rs}) == 3 for _, rs, _ in cand2)
    # the plan counts the same
    pl = rescore_plan(sc2, {k: [None] * n_ent}, 2, apart=True)
    assert pl["annuli"][k]["own_per_draw"] == 3 and pl["annuli"][k]["entrant"] == n_ent * 2 * 3
    assert rescore_plan(sc2, {k: [None] * n_ent}, 2)["annuli"][k]["entrant"] == n_ent * 2


@pytest.mark.slow
def test_a_directory_started_before_keeps_its_tiles_together(runs, tmp_path):
    a, _ = runs
    out = str(tmp_path / "old")
    rescore(_resume(a), [], out, annuli=[1], n_draws=2, tiles="together", **FAST)
    os.remove(os.path.join(out, "state", "options.json"))           # as a directory of 2026-11-10 is
    rep = rescore(_resume(a), [], out, annuli=[1], n_draws=2, **FAST)
    assert rep["tiles"] == "together" and rep["reductions"] == len(EDGES) - 1, "resumed, nothing scored again"
    with pytest.raises(ValueError, match="started with --tiles together"):
        rescore(_resume(a), [], out, annuli=[1], n_draws=2, tiles="apart", **FAST)


# ---------------------------------------------------------------------------- the composite
@pytest.mark.slow
def test_a_composite_base_reads_what_rereduce_with_its_winners_gives(runs, tmp_path):
    a, b = runs
    picks = [Pick(0, b, _winner_eval(b, 0)), Pick(2, b, _winner_eval(b, 2))]
    winners = {p.annulus: e for p, e in resolve_picks(a, picks)}
    sc = StitchScorer(_resume(a), log=QUIET)
    sc.use_winners(winners)
    sc.prepare(1)
    d = sc.draw([0, 1, 2], {j: 4e-3 for j in sc.order}, np.random.default_rng(5))
    src = [s for _, s in d]
    ours = sc.score_draws([d], 1, window=1)[0]
    out = str(tmp_path / "comp")
    rec = _resume(a).rereduce(out, verify_sources=[(s.rho, s.theta, s.contrast) for s in src], log=QUIET,
                              winners=winners)
    got = [float(ln.split("S/N=")[1].split()[0]) for ln in open(os.path.join(out, "stitched_verify_report.txt"))
           if ln.startswith("   rho=")]
    assert len(got) == len(src) and np.allclose(ours, got, atol=0.006), (ours, got)
    assert set(rec["composite"]) == {"1", "3"} and rec["composite"]["1"]["run"] == "run_b"
    assert rec["annuli"][0]["winner_from"]["run"] == "run_b" and rec["annuli"][1]["winner_eval"] is not None
    # the parameters table shows the composite's configurations
    tab = open(os.path.join(out, "klip_stitched_params.txt")).read()
    k_b = winners[0].config["params"].get("k_klip")
    assert tab and (k_b is None or str(k_b) in tab)
    # the base run's own stitch on the same draw differs where the composite changed annuli
    if any(_winner_key(a, k) != _winner_key(b, k) for k in (0, 2)):
        plain = StitchScorer(_resume(a), log=QUIET)
        plain.prepare(1)
        assert not np.allclose(plain.score_draws([d], 1, window=1)[0], ours)


@pytest.mark.slow
def test_a_composite_rescore_marks_its_base_and_keeps_its_options(runs, tmp_path):
    a, b = runs
    ev = _other_eval(b, a, 1)
    out = str(tmp_path / "rs")
    rep = rescore(_resume(a), [], out, annuli=[1], n_draws=2, tiles="together",
                  base_winners=[Pick(1, b, ev), Pick(0, b, _winner_eval(b, 0))], **FAST)
    ann = rep["annuli"]["2"]
    base = [e for e in ann["entrants"] if e["base"]]
    assert len(base) == 1 and base[0]["entrant"]["run"] == "run_b" and base[0]["entrant"]["eval_index"] == ev
    assert any(e["entrant"]["run"] == "run_a" and e["entrant"]["kind"] == "winner" and not e["base"]
               for e in ann["entrants"]), "the base run's own winner enters as a candidate"
    assert set(rep["base_winners"]) == {"1", "2"} and ann["base_winner"]["run"] == "run_b"
    text = open(os.path.join(out, "rescore.txt")).read()
    assert "# base winners: annulus 1 run_b" in text and "annulus 2 run_b" in text
    with pytest.raises(ValueError, match="other base winners"):
        rescore(_resume(a), [], out, annuli=[1], n_draws=2, **FAST)
    with pytest.raises(ValueError, match="--tiles together"):
        rescore(_resume(a), [], out, annuli=[1], n_draws=2, tiles="apart",
                base_winners=[Pick(1, b, ev), Pick(0, b, _winner_eval(b, 0))], **FAST)
    # the same options again: resumed
    rep2 = rescore(_resume(a), [], out, annuli=[1], n_draws=2,
                   base_winners=[Pick(0, b, _winner_eval(b, 0)), Pick(1, b, ev)], **FAST)
    assert rep2["reductions"] == len(EDGES) - 1
    with pytest.raises(ValueError, match="two base winners"):
        rescore(_resume(a), [], str(tmp_path / "x"), annuli=[1], n_draws=2, plan=True,
                base_winners=[Pick(1, b, ev), Pick(1, b, 0)], **FAST)


# ---------------------------------------------------------------------------- draws from another re-score
@pytest.mark.slow
def test_draws_from_another_rescore_pair_with_it(runs, tmp_path, monkeypatch):
    a, b = runs
    first = str(tmp_path / "first")
    rep_a = rescore(_resume(a), [], first, annuli=[1], n_draws=3, tiles="together", **FAST)
    # the same base and mode on its draws: no calibration, the same draws, a difference of 0
    seen = []
    _spy(monkeypatch, seen)
    second = str(tmp_path / "second")
    rep_b = rescore(_resume(a), [], second, annuli=[1], draws_from=first, **FAST)
    assert rep_b["tiles"] == "together" and rep_b["draws_from"] == first and rep_b["n_draws"] == 3
    assert rep_b["annuli"]["2"]["draws"] == rep_a["annuli"]["2"]["draws"]
    assert rep_b["calibration"]["2"]["contrast"] == rep_a["calibration"]["2"]["contrast"]
    n_ent = len(rep_b["annuli"]["2"]["entrants"])
    nb = sum(1 for _, rs, t in seen if rs and "_c_" not in t)
    cand = sum(1 for _, rs, t in seen if rs and "_c_" in t)
    assert cand == n_ent * 3 and nb == 3 * 2, "the neighbors and candidates only: no calibration"
    ag = rep_b["annuli"]["2"]["against"]
    assert not ag.get("missing") and ag["tiles"] == "together"
    assert ag["all"]["diff_mean"] == 0.0 and all(v["diff_mean"] == 0.0 for v in ag["per_tile"].values())
    assert "against first's base" in open(os.path.join(second, "rescore.txt")).read()
    # apart on its first two draws, with a pick file's candidates and a composite base
    ev = _other_eval(b, a, 1)
    third = str(tmp_path / "third")
    rep_c = rescore(_resume(a), [b], third, annuli=[1], n_draws=2, draws_from=first, tiles="apart",
                    base_winners=[Pick(1, b, ev)], candidates=[Pick(1, a, _winner_eval(a, 1))], **FAST)
    ann = rep_c["annuli"]["2"]
    assert rep_c["tiles"] == "apart" and len(ann["draws"]) == 2 and ann["draws"] == rep_a["annuli"]["2"]["draws"][:2]
    assert {(e["entrant"]["run"], e["base"]) for e in ann["entrants"]} == {("run_b", True), ("run_a", False)}
    assert ann["against"]["tiles"] == "together" and np.isfinite(ann["against"]["all"]["diff_mean"])
    # an annulus the other re-score has no draws of stops; placement options do not go with it
    rep_d = rescore(_resume(a), [], str(tmp_path / "fourth"), annuli=[0], draws_from=first, **FAST)
    assert "has no draws of annulus 1" in rep_d["annuli"]["1"]["stopped"]
    with pytest.raises(ValueError, match="come placed"):
        rescore(_resume(a), [], str(tmp_path / "fifth"), annuli=[1], draws_from=first, clear_fwhm=1.0, **FAST)
    with pytest.raises(ValueError, match="no re-score"):
        rescore(_resume(a), [], str(tmp_path / "sixth"), annuli=[1], draws_from=str(tmp_path), **FAST)


def test_the_command_line_takes_the_options():
    from klip_tpe.cli import _build_parser
    a = _build_parser().parse_args(["rescore", "--tiles", "apart", "--base-winners", "bw.txt", "--candidates",
                                    "c.txt", "--draws-from", "prev"])
    assert (a.tiles, a.base_winners, a.candidates, a.draws_from, a.draws) == ("apart", "bw.txt", "c.txt", "prev", None)
    r = _build_parser().parse_args(["rereduce", "--base-winners", "bw.txt"])
    assert r.base_winners == "bw.txt"
