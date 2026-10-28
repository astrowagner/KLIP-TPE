"""Validation's final round (``ValidationConfig.n_final``), from the GPU session's addendum of
2026-10-28.  On 200 common draws the list of NEAR2's run_20261027_222223 held a better
reduction than the elected one in four of seven annuli, by 1 to 3 %.  What must hold:

* off by default, and a run saved before it resumes without it;
* the entrants are the first round's best by median, each scored on the same draws, and the
  largest median there is elected;
* the round draws nothing from the run's generator, so the first round, and everything after
  the validation, is what it was, and a run cut in the round resumes to the same round;
* the winner is then scored on fresh draws, not on the round's.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from klip_tpe import Runner, RunConfig, ValidationConfig, cli
from klip_tpe.metrics import nanmedian_even
from klip_tpe.runner import RunCallback

from conftest import build_synthetic_run

QUIET = lambda s: None                                     # noqa: E731


class _State(RunCallback):
    """The run's generator as validation hands its winner on."""

    def __init__(self):
        self.states = []

    def on_validation(self, runner, table, winner):
        self.states.append(json.dumps(runner.rng.bit_generator.state, sort_keys=True))


@pytest.mark.slow
def test_the_final_round_elects_on_common_draws_and_leaves_the_run_as_it_was(tmp_path):
    runs = {}
    for name, vc in (("off", ValidationConfig(n_top=3, n_valid=3)),
                     ("on", ValidationConfig(n_top=3, n_valid=3, n_final=3, n_final_draws=6))):
        red, space, obj, samp, cfg = build_synthetic_run(n_iter=[16, 12], validation=vc)
        st, said = _State(), []
        res = Runner(red, space, obj, samp, cfg, str(tmp_path / name), log=said.append, callbacks=[st]).run()
        runs[name] = (res, st.states, said)
    (a, sa, said), (b, sb, _) = runs["on"], runs["off"]
    assert sa == sb, "the round's draws are its own: the run's generator stands where it did"
    assert all(y.final_round is None for y in b), "off by default"
    n_rounds = 0
    for ia, (x, y) in enumerate(zip(a, b)):
        assert x.validation_table == y.validation_table, "the first round is what it was"
        fr = x.final_round
        if fr is None:
            assert sum(r["validated_score"] is not None for r in x.validation_table) < 2
            continue
        n_rounds += 1
        rows = fr["rows"]
        meds = sorted((r["validated_score"] for r in x.validation_table if r["validated_score"] is not None),
                      reverse=True)
        assert [r["validated_score"] for r in rows] == pytest.approx(meds[:3]), "the first round's best"
        for r in rows:
            assert len(r["trials"]) == 6
            assert r["final_score"] == pytest.approx(nanmedian_even(np.array(r["trials"], float)))
        best = max(range(len(rows)), key=lambda k: (rows[k]["final_score"], -k))
        assert fr["winner_candidate"] == rows[best]["candidate"] and x.winner_index == rows[best]["eval_index"]
        assert x.election_score == pytest.approx(rows[best]["final_score"])
        vf = json.load(open(tmp_path / "on" / f"annulus{ia + 1:02d}" / "validation_final.json"))
        ts = vf["trial_sources"]
        assert len(ts) == len(rows) and all(t == ts[0] for t in ts), "common draws: the same sources for each"
        assert len(ts[0]) == 6 and len({json.dumps(t) for t in ts[0]}) == 6, "six different draws"
        # then fresh draws for the reported median, none of them the round's
        assert len(x.rescore_trials) == 3 and x.winner_score == pytest.approx(
            nanmedian_even(np.array(x.rescore_trials, float)))
        assert not any(src in ts[0] for src in x.rescore_sources)
        assert x.rescore_sources[x.rescore_committed] == [list(s) for s in x.winner_sources]
        w = json.load(open(tmp_path / "on" / f"annulus{ia + 1:02d}" / "winner.json"))
        assert w["final_round"]["winner_candidate"] == fr["winner_candidate"]
    assert n_rounds >= 1
    assert any("on 6 common draws" in s for s in said)


@pytest.mark.slow
def test_a_run_cut_in_the_final_round_resumes_to_the_same_round(tmp_path):
    """The round keeps no record of its own: a resumed run reloads the first round and makes
    the same round again, on the same draws, and elects the same winner."""
    vc = ValidationConfig(n_top=3, n_valid=2, n_final=3, n_final_draws=4)
    red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 30], n_iter=8, n_init=3, seed=3, save_fits=False,
                                                     verify=False, candidates=False, validation=vc)
    ref = Runner(red, space, obj, samp, cfg, str(tmp_path / "ref"), log=QUIET)
    ref.run()

    class Interrupt(BaseException):      # like Ctrl-C: not swallowed by the per-trial guards
        pass

    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "cut"), log=QUIET)
    orig = r._reduce

    def cut(cfg_, src, tag="", **kw):
        if tag.startswith("a1_valfin") and tag.endswith("_t2"):
            raise Interrupt
        return orig(cfg_, src, tag=tag, **kw)
    r._reduce = cut
    with pytest.raises(Interrupt):
        r.run()
    said = []
    r2 = Runner.resume(str(tmp_path / "cut"), red, obj, samp, log=said.append)
    r2.run()
    assert any("[resumed]" in s_ for s_ in said) and any("common draws" in s_ for s_ in said)
    a, b = r2.results[0], ref.results[0]
    assert a.final_round is not None and a.final_round == b.final_round
    assert (a.winner_index, a.winner_score, a.election_score) == (b.winner_index, b.winner_score, b.election_score)
    assert a.rescore_trials == b.rescore_trials and a.rescore_sources == b.rescore_sources


def test_off_by_default_and_for_a_run_saved_before():
    assert ValidationConfig().n_final == 0
    d = RunConfig(ann_edges=[8, 20]).to_dict()
    for k in ("n_final", "n_final_draws"):
        d["validation"].pop(k)
    assert RunConfig.from_dict(d).validation.n_final == 0


def test_the_options_parse():
    a = cli._build_parser().parse_args(["near", "--root", "/r"])
    assert a.n_final == 0 and a.n_final_draws == 200
    a = cli._build_parser().parse_args(["near", "--root", "/r", "--n-final", "5", "--n-final-draws", "100"])
    assert a.n_final == 5 and a.n_final_draws == 100
