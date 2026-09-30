"""Every draw recorded, and recorded draws replayed.

A trial's score is the mean over several fresh injection draws, but until 2026-09-30 only
the last draw's positions were kept, so a search could not be re-scored on other frames: its
draws came from a sequential RNG stream, and a sampler's draws depend on the frames (their
rolls and dead zones set its forbidden sectors), so the same seed does not reproduce them on
another processing of the same data.  Now every draw's sources and raw score are recorded,
validation rows keep their trials' sources, and :class:`ReplaySampler` /
``Runner.replay_draws`` / ``library_ablation --draws-from`` put recorded positions back.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

from klip_tpe.positions import PositionSampler, ReplaySampler
from klip_tpe.runner import EvalRecord, Runner

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _rec(score, srcs, raw=None):
    return EvalRecord(annulus=0, index=0, phase="tpe", x=[0.0], config={"params": {}},
                      sources=[tuple(s) for s in srcs], score=score, raw_score=score if raw is None else raw,
                      per_source=[score], raw_per_source=[score], clean_per_source=None,
                      partition_snr={}, k_used=7, contrast=1e-4, wall_s=1.0, meta={})


def test_every_draw_of_a_trial_is_recorded_not_just_the_last():
    recs = [_rec(4.0, [(1.0, 10.0, 1e-4), (1.5, 190.0, 1e-4)], raw=4.5),
            _rec(6.0, [(1.0, 20.0, 1e-4), (1.5, 200.0, 1e-4)], raw=6.5),
            _rec(None, [(1.0, 30.0, 1e-4), (1.5, 210.0, 1e-4)], raw=None)]
    out = Runner._combine_draws(recs, recs[-1])
    assert out.meta["draw_sources"] == [[[1.0, 10.0, 1e-4], [1.5, 190.0, 1e-4]],
                                        [[1.0, 20.0, 1e-4], [1.5, 200.0, 1e-4]],
                                        [[1.0, 30.0, 1e-4], [1.5, 210.0, 1e-4]]]
    assert out.meta["draw_raw_scores"] == [4.5, 6.5, None]
    assert out.sources == [(1.0, 30.0, 1e-4), (1.5, 210.0, 1e-4)], "positions still from the last draw"


def test_the_replay_sampler_hands_back_the_recorded_draws_in_order():
    base = PositionSampler(fwhm_as=0.36, known=[(0.82, 150.0)], forbidden_pa=[(13.0, 4.5)])
    rec = [[(1.2, 40.0), (1.6, 220.0)], [(1.2, 80.0, 9.9), (1.6, 260.0, 9.9)]]
    rs = ReplaySampler(rec, base=base)
    a = rs.sample(2, 1.2, 1.75, np.random.default_rng(0), 2.8e-4)
    b = rs.sample(2, 1.2, 1.75, None, 2.8e-4)
    assert [(s.rho, s.theta, s.contrast) for s in a] == [(1.2, 40.0, 2.8e-4), (1.6, 220.0, 2.8e-4)]
    assert [(s.rho, s.theta) for s in b] == [(1.2, 80.0), (1.6, 260.0)]
    assert all(s.contrast == 2.8e-4 for s in b), "the contrast injected is the one asked for"
    with pytest.raises(IndexError, match="replayed"):
        rs.sample(2, 1.2, 1.75, None, 2.8e-4)
    assert rs.fwhm_as == 0.36 and rs.known == [(0.82, 150.0)], "geometry comes from the base sampler"
    d = rs.describe()
    assert d["replay"] is True and d["n_recorded"] == 2 and d["forbidden_pa"] == [[13.0, 4.5]]


def test_a_draw_of_the_wrong_size_is_refused():
    rs = ReplaySampler([[(1.2, 40.0)]])
    with pytest.raises(ValueError, match="holds 1 sources, 2 asked"):
        rs.sample(2, 1.2, 1.75, None, 1e-4)


def test_runner_replay_draws_swaps_in_a_replay_and_the_packing_probe_does_not_spend_it():
    """``_ring_survives`` samples positions to test the geometry; under a replay it must use
    the base sampler, or it would consume the recorded draws before the search saw them."""
    base = PositionSampler(fwhm_as=0.36)

    class _Stub:
        replay_draws = Runner.replay_draws

        def __init__(self):
            self.sampler = base

    r = _Stub()
    r.replay_draws([[(1.2, 40.0)], [(1.2, 50.0)]])
    assert isinstance(r.sampler, ReplaySampler) and r.sampler.base is base
    r.replay_draws([[(1.3, 60.0)]])                   # a second replay stands in for the same base
    assert r.sampler.base is base and r.sampler.draws == [[(1.3, 60.0)]]
    smp = getattr(r.sampler, "base", None) or r.sampler      # what _ring_survives now samples from
    smp.sample(1, 1.2, 1.2, np.random.default_rng(1))
    assert r.sampler.i == 0, "the probe must not have spent a recorded draw"


@pytest.fixture(scope="module")
def LA():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location("library_ablation_rd", os.path.join(ROOT, "scripts",
                                                                                      "library_ablation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_an_ablation_replays_another_ablations_recorded_positions(LA):
    abl = {"n_draws": 2, "annuli": [{"annulus": 1, "draws": [[[1.2, 40.0], [1.6, 220.0]],
                                                             [[1.2, 80.0], [1.6, 260.0]]]},
                                    {"annulus": 2, "draws": [[[2.3, 10.0]]]}]}
    assert LA.recorded_draws(abl, 1, 2) == [[(1.2, 40.0), (1.6, 220.0)], [(1.2, 80.0), (1.6, 260.0)]]
    with pytest.raises(SystemExit, match="recorded 1 sources per draw"):
        LA.recorded_draws(abl, 2, 3)
    with pytest.raises(SystemExit, match="no recorded draws for annulus 3"):
        LA.recorded_draws(abl, 3, 2)
