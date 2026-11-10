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
* A draw places the tile with the most sources first, starts again when a tile finds no place,
  and draws as before where the tiles hold as many sources each.  An annulus whose draws find
  no place under the run's rule falls back to 1.5 FWHM, then to one source in each neighbor's
  tile, and says so.  One that finds none under any rule, or whose kept draws do not fit the
  options, stops with the reason and the re-score goes on (NEAR2, 2026-11-10).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from klip_tpe.refine import (Entrant, PlacementError, StitchScorer, _dist, _placements, _rule_text, config_key,
                             paired_summary, parse_annuli, rescore, run_entrants)
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
    # another number of draws on the same directory stops the annulus, with the reason
    rep3 = rescore(_resume(a), [b], out, annuli=[1], n_draws=4, n_cal=2, max_iter=2, window=2, log=QUIET)
    assert "holds 3 draws" in rep3["annuli"]["2"]["stopped"]


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


def _old_draw(sc, tiles, contrasts, rng):
    """The draw of the code before the placement rules (0531e2f), for the comparison."""
    smp, px = sc.r.sampler, float(sc.r.pxscale)
    dmin = float(getattr(smp, "min_mutual_fwhm", 2.0)) * float(getattr(smp, "fwhm_as", sc.r.fwhm * px))
    out = []
    for j in tiles:
        t0, t1 = sc.tile(j)
        for _ in range(300):
            src = smp.sample(sc.r._nsrc(j), t0 * px, t1 * px, rng, float(contrasts[j]))
            ok = all(_dist(s, o) >= dmin for s in src for _, o in out)
            if ok and out and hasattr(smp, "_in_sidelobes"):
                ok = not smp._in_sidelobes([o.rho for _, o in out] + [s.rho for s in src],
                                           [o.theta for _, o in out] + [s.theta for s in src])
            if ok:
                break
        else:
            raise RuntimeError("no placement")
        out += [(j, s) for s in src]
    return out


def test_a_draw_places_the_most_sources_first_and_the_rules_fall_back(runs):
    a, _ = runs
    sc = StitchScorer(_resume(a), log=QUIET)
    contrasts = {j: 4e-3 for j in sc.order}
    run_n = {j: int(sc.r._nsrc(j)) for j in sc.order}
    # the tiles hold as many sources each: the draws of the code before, stream for stream
    if len({run_n[j] for j in (0, 1, 2)}) == 1:
        r1, r2 = np.random.default_rng([1, 1]), np.random.default_rng([1, 1])
        for _ in range(5):
            assert sc.draw([0, 1, 2], contrasts, r1) == _old_draw(sc, [0, 1, 2], contrasts, r2)
    # a pair in the outer tile: placed first, given back in the tiles' order, all clear of each other
    d = sc.draw([0, 1, 2], contrasts, np.random.default_rng(3), counts={0: 1, 1: 1, 2: 2})
    assert [j for j, _ in d] == [0, 1, 2, 2]
    px = float(sc.r.pxscale)
    assert all(sc.tile(j)[0] - 1e-6 <= s.rho / px <= sc.tile(j)[1] + 1e-6 for j, s in d)
    dmin = sc.clear_fwhm * sc.fwhm_px * px
    assert all(_dist(s, o) >= dmin - 1e-9 for j, s in d for i, o in d if i != j)
    with pytest.raises(PlacementError, match="no placement in tile .* by 1000 FWHM"):
        sc.draw([0, 1, 2], contrasts, np.random.default_rng(3), clear_fwhm=1e3, restarts=2)
    # the rules: the run's, 1.5 FWHM, then one source in each neighbor's tile; or the one asked
    rules = _placements(sc, 1)
    assert rules[0] == {"counts": {j: run_n[j] for j in (0, 1, 2)}, "clear_fwhm": sc.clear_fwhm, "rule": "the run's"}
    assert rules[1] == {"counts": rules[0]["counts"], "clear_fwhm": 1.5, "rule": "relaxed"}
    if run_n[0] > 1 or run_n[2] > 1:
        assert rules[2]["counts"] == {0: 1, 1: run_n[1], 2: 1}
    assert _placements(sc, 1, sources_per_tile=1, clear_fwhm=1.0) == [
        {"counts": {0: 1, 1: 1, 2: 1}, "clear_fwhm": 1.0, "rule": "asked"}]
    assert _rule_text({"counts": {0: 1, 1: 1, 2: 2}, "clear_fwhm": 2.0}, 6.39) == \
        "1 source in tile 1, 1 in tile 2 and 2 in tile 3, 2 FWHM (12.8 px) apart"
    assert _rule_text({"counts": {3: 2}, "clear_fwhm": 1.5}, 4.0) == "2 sources in tile 4, 1.5 FWHM (6.0 px) apart"


@pytest.mark.slow
def test_an_annulus_without_a_placement_falls_back_or_stops(runs, tmp_path):
    a, _ = runs
    out = str(tmp_path / "rs")
    logs = []
    r = _resume(a)
    r.sampler.min_mutual_fwhm = 1e3                  # the run's distance places nothing
    rep = rescore(r, [], out, annuli=[0, 1], n_draws=2, n_cal=2, max_iter=1, window=2, log=logs.append)
    for ann in ("1", "2"):
        pl = rep["annuli"][ann]["placement"]
        assert pl["rule"] == "relaxed" and pl["clear_fwhm"] == 1.5, pl
    assert any(m.startswith("annulus 2: no placement with") and "1000 FWHM" in m for m in logs)
    assert any(m.startswith("annulus 2: 2 draws, ") and "1.5 FWHM" in m and m.endswith("(relaxed)") for m in logs)
    assert json.load(open(os.path.join(out, "state", "ann02_placement.json")))["clear_fwhm"] == 1.5
    text = open(os.path.join(out, "rescore.txt")).read()
    assert "  draws: " in text and "1.5 FWHM" in text
    # started again with another clearance: the kept annuli stop with the reason, the next is placed
    logs.clear()
    rep2 = rescore(_resume(a), [], out, annuli=[0, 1, 2], n_draws=2, n_cal=2, max_iter=1, window=2,
                   clear_fwhm=2.0, log=logs.append)
    for ann in ("1", "2"):
        assert "1.5 FWHM" in rep2["annuli"][ann]["stopped"] and "same options" in rep2["annuli"][ann]["stopped"]
    assert rep2["annuli"]["3"]["placement"]["rule"] == "asked" and rep2["annuli"]["3"]["entrants"]
    assert any(m.startswith("stopped: annuli 1, 2,") for m in logs)
    text = open(os.path.join(out, "rescore.txt")).read()
    assert text.count("stopped ==") == 2 and "== annulus 3" in text
    # the same options again: the kept draws, with their rule
    logs.clear()
    rep3 = rescore(_resume(a), [], out, annuli=[1], n_draws=2, n_cal=2, max_iter=1, window=2, log=logs.append)
    assert rep3["annuli"]["2"]["placement"]["clear_fwhm"] == 1.5
    assert any("kept from the last start" in m for m in logs)
