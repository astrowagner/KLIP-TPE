"""Validation's candidates, and the winner's reported score.

From the GPU session's note of 2026-10-28.  It re-validated every annulus of NEAR2's
run_20261024_153342 and run_20261027_222223 on 200 common draws and found the 25 candidates
of an annulus to be 7 to 20 distinct reductions (parameters a reduction never reads set two
vectors apart), and the elected winner's median high on its own draws by 0.22 S/N on average
(the candidates not elected were unbiased).  What must hold:

* parameters a reduction never reads give the same reduction, bit for bit: a deselected
  night's, an even high-pass width (run as the next odd one), frame cuts with no frame between
  them.  A live parameter does not;
* validation skips a configuration whose clean reduction equals an earlier candidate's for the
  next in the search's order, and ``dedup="vector"`` compares the vectors only, as before;
* the elected winner is scored on fresh draws: their median is ``winner_score`` and the
  election's ``election_score``.  The draws leave the run's generator where it stood, so the
  next annulus is the one it would have been, and ``n_rescore=0`` reports the election's
  median, as before;
* a lone candidate, whose draws were not selected on, is not re-scored, so ``n_top=1`` gives
  what it gave before;
* a run saved before 2026-10-28 resumes with the old rules.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

from klip_tpe import (GaussianPSF, KLIPReducer, Param, PartitionedReducer, Runner, RunConfig, SearchSpace,
                      ValidationConfig, cli)
from klip_tpe.metrics import nanmedian_even
from klip_tpe.optimizers import History
from klip_tpe.reducer import EvalImages
from klip_tpe.runner import _same_reduction
from klip_tpe.space import Config
from klip_tpe.synthetic import synthetic_klip_dataset

from conftest import build_synthetic_run

PX = 0.05
QUIET = lambda s: None                                     # noqa: E731
BASE = {"k_klip": 4, "inrad": 6, "outrad": 18, "bin": 2, "filter": 9, "n_ang": 2, "angsep": 0.5, "anglemax": 35.0,
        "corr_thresh": 0.0, "noise_max": 3.0, "coronoise_max": 3.0}


def _pr():
    reds = {f"n{i}": KLIPReducer(synthetic_klip_dataset(nframes=24, size=48, pa_span=50, seed=i), PX,
                                 3.89 * PX / 206265 * 8.2, 8.2, injection_model=GaussianPSF(4.0, star_flux=1e4),
                                 outrad_cap=22) for i in (1, 2)}
    return PartitionedReducer(reds, max_workers=1)


def test_parameters_a_reduction_never_reads_give_the_same_reduction():
    pr = _pr()

    def red(per, sel=("n1", "n2")):
        cfg = Config(params=dict(BASE), per_partition={k: dict(BASE, **per.get(k, {})) for k in ("n1", "n2")},
                     selected=list(sel), x=np.zeros(0))
        return pr.reduce_config(cfg)
    a = red({})
    # an even high-pass width runs as the next odd one (klip.highpass, IDL smooth)
    assert _same_reduction(a, red({"n1": {"filter": 8}, "n2": {"filter": 8}}))
    assert not _same_reduction(a, red({"n1": {"filter": 7}, "n2": {"filter": 7}}))
    # a deselected night's parameters
    one = red({}, ("n1",))
    assert _same_reduction(one, red({"n2": {"k_klip": 9, "angsep": 1.2, "filter": 3}}, ("n1",)))
    assert not _same_reduction(one, a)
    # frame cuts with no frame between them keep the same frames
    s = np.sort(pr.reducers["n1"].data.tags["corrs"])
    t1, t2 = s[3] + 0.25 * (s[4] - s[3]), s[3] + 0.75 * (s[4] - s[3])
    c1 = red({"n1": {"corr_thresh": float(t1)}})
    assert _same_reduction(c1, red({"n1": {"corr_thresh": float(t2)}}))
    assert not _same_reduction(c1, a), "the cut took four frames out"
    # a live parameter
    assert not _same_reduction(a, red({"n1": {"k_klip": 5}}))


def _stub_runner(hist):
    """A runner whose clean reduction reads parameter ``a`` only: ``b`` is dead."""
    r = Runner.__new__(Runner)
    r.space = SearchSpace([Param("a", 1, 5, "int"), Param("b", 0.0, 1.0)])
    r.history = History(ndim=2)
    for (a, b), y in hist:
        r.history.append(np.array([a, b], float), y, {"phase": "tpe"})
    r.cfg = RunConfig(ann_edges=[8, 20])
    r.hb = types.SimpleNamespace(stage=lambda *a, **k: None)
    said = []
    r.log = said.append
    r._reduction_window = lambda n=None: 1
    r._reduce = lambda cfg, src, tag="": EvalImages(image=np.full((4, 4), float(cfg.params["a"])))
    return r, said


def test_validation_skips_a_copy_for_the_next_distinct_reduction():
    hist = [((3, 0.1), 10.0), ((3, 0.7), 9.9), ((3, 0.4), 9.8), ((2, 0.5), 9.0), ((4, 0.2), 8.0), ((5, 0.9), 7.0)]
    r, said = _stub_runner(hist)
    cands, cleans, dups = r._validation_candidates(0, ValidationConfig(n_top=3, n_valid=2))
    assert cands == [0, 3, 4] and dups == [[1, 0], [2, 0]]
    assert [float(c.image[0, 0]) for c in cleans] == [3.0, 2.0, 4.0], "each candidate keeps its clean image"
    assert "2 of the 5 configurations examined reduce as an earlier candidate does" in "\n".join(said)
    # the vectors only, as before 2026-10-28: no reduction is made
    cands, cleans, dups = r._validation_candidates(0, ValidationConfig(n_top=3, n_valid=2, dedup="vector"))
    assert cands == [0, 1, 2] and cleans == [None] * 3 and dups == []
    # a look not deep enough gives what there is, and says so
    said.clear()
    cands, _, _ = r._validation_candidates(0, ValidationConfig(n_top=3, n_valid=2, dedup_depth=1))
    assert cands == [0] and "1 distinct reduction(s) among the first 3" in "\n".join(said)


@pytest.mark.slow
def test_the_winner_is_scored_on_fresh_draws_and_the_run_goes_on_as_it_would(tmp_path):
    runs = {}
    for name, n_re in (("fresh", None), ("election", 0)):
        red, space, obj, samp, cfg = build_synthetic_run(n_iter=[16, 12], validation=ValidationConfig(
            n_top=3, n_valid=4, n_rescore=n_re))
        runs[name] = (Runner(red, space, obj, samp, cfg, str(tmp_path / name), log=QUIET).run(), tmp_path / name)
    (a, da), (b, db) = runs["fresh"], runs["election"]
    for x, y in zip(a, b):
        assert x.winner_index == y.winner_index and x.validation_table == y.validation_table, "the same election"
        row = next(t for t in x.validation_table if t["eval_index"] == x.winner_index)
        assert x.election_score == pytest.approx(row["validated_score"]) and y.election_score is None
        assert y.winner_score == pytest.approx(row["validated_score"]), "n_rescore=0 reports the election's median"
        assert len(x.rescore_trials) == 4
        assert x.winner_score == pytest.approx(nanmedian_even(np.array(x.rescore_trials, float)))
        assert x.winner_score != pytest.approx(x.election_score)
    # the fresh draws did not move the run's generator: the second annulus is the same, record for record
    ra = [json.loads(ln) for ln in open(da / "results.jsonl")]
    rb = [json.loads(ln) for ln in open(db / "results.jsonl")]
    assert [(r["score"], r["sources"]) for r in ra] == [(r["score"], r["sources"]) for r in rb]
    w = json.load(open(da / "annulus01" / "winner.json"))
    assert w["election_score"] == pytest.approx(a[0].election_score) and len(w["rescore_trials"]) == 4
    c = w["rescore_committed"]
    assert len(w["rescore_sources"]) == 4 and w["rescore_sources"][c] == w["winner_sources"]
    fin = [v for v in w["rescore_trials"] if v is not None]
    assert abs(w["rescore_trials"][c] - w["winner_score"]) == min(abs(v - w["winner_score"]) for v in fin)


@pytest.mark.slow
def test_a_lone_candidate_gives_what_it_gave_before(tmp_path):
    """``n_top=1`` (the paper scripts' re-measurements): no election, so no re-score, and the
    one candidate's clean image is the one validation made before."""
    out = {}
    for name, vc in (("new", ValidationConfig(n_top=1, n_valid=3)),
                     ("old", ValidationConfig(n_top=1, n_valid=3, dedup="vector", n_rescore=0))):
        red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=10, n_init=4, validation=vc)
        res = Runner(red, space, obj, samp, cfg, str(tmp_path / name), log=QUIET).run()
        out[name] = (res, [json.loads(ln) for ln in open(tmp_path / name / "results.jsonl")])
    (a, ra), (b, rb) = out["new"], out["old"]
    assert [(r["score"], r["sources"]) for r in ra] == [(r["score"], r["sources"]) for r in rb]
    for x, y in zip(a, b):
        assert x.election_score is None and x.rescore_trials is None and x.rescore_committed is None
        assert x.winner_score == y.winner_score and x.validation_table == y.validation_table
        assert x.winner_sources == y.winner_sources and x.validation_samples == y.validation_samples


def test_a_run_saved_before_resumes_with_the_old_rules():
    d = RunConfig(ann_edges=[8, 20]).to_dict()
    for k in ("dedup", "dedup_depth", "n_rescore"):
        d["validation"].pop(k)
    old = RunConfig.from_dict(d)
    assert old.validation.dedup == "vector" and old.validation.n_rescore == 0
    new = RunConfig.from_dict(RunConfig(ann_edges=[8, 20]).to_dict())
    assert new.validation.dedup == "image" and new.validation.n_rescore is None


def test_the_options_parse():
    a = cli._build_parser().parse_args(["near", "--root", "/r"])
    assert a.n_rescore is None and a.validation_dedup is None, "None: the new rules, or the IDL's with --from-idl-setup"
    a = cli._build_parser().parse_args(["near", "--root", "/r", "--n-rescore", "0", "--validation-dedup", "vector"])
    assert a.n_rescore == 0 and a.validation_dedup == "vector"
