"""A draw that breaks the sampler's rules at every try is placed one source at a time (the audit of 2026-11-25).

``"spread"`` places the sources evenly and draws only the first angle, so 3 or 6 sources cannot all
miss sectors that lie 90 degrees apart.  Such a draw was injected as it was, silently.  It is now
placed again one source at a time, each clear of the rules, and both kinds of draw are counted,
warned of once and logged at each annulus' end.  ``RunConfig.sampler_fallback`` False keeps the old
draws for runs saved before.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from klip_tpe import RunConfig
from klip_tpe.positions import PositionSampler

QUAD = [(c, 16.0) for c in (0.0, 90.0, 180.0, 270.0)]


def _bad(s, src):
    return bool(s._bad(np.array([q.rho for q in src]), np.array([q.theta for q in src])))


def _draw(s, n, k=20, seed=2):
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        draws = [s.sample(n, 1.0, 2.5, rng) for _ in range(k)]
    return draws, [str(w.message) for w in got if "PositionSampler" in str(w.message)]


@pytest.mark.parametrize("n", [3, 6])
def test_sectors_a_quarter_turn_apart_take_the_sources_one_at_a_time(n):
    s = PositionSampler(fwhm_as=0.35, forbidden_pa=QUAD)
    draws, said = _draw(s, n)
    assert not any(_bad(s, d) for d in draws)
    assert s.n_fallbacks == 20 and s.n_rule_breaks == 0
    assert len(said) == 1 and "placed one at a time" in said[0] and f"{n} evenly spaced sources" in said[0]
    for d in draws:
        th = np.array([q.theta for q in d])
        gaps = np.abs((th[:, None] - th[None, :] + 180.0) % 360.0 - 180.0)[np.triu_indices(n, 1)]
        assert gaps.min() >= (45.0 if n == 3 else 20.0)              # as far apart as they fit
        # the radial ladder stays the evenly spaced draw's
        assert sorted(q.rho for q in d) == pytest.approx([1.0 + (i + 0.5) * 1.5 / n for i in range(n)])
    # without the fallback, as before 2026-11-25: every draw breaks a rule, counted and said once
    old = PositionSampler(fwhm_as=0.35, forbidden_pa=QUAD, fallback=False)
    draws, said = _draw(old, n)
    assert all(_bad(old, d) for d in draws) and old.n_rule_breaks == 20 and old.n_fallbacks == 0
    assert len(said) == 1 and "injected as it is" in said[0] and "one at a time" not in said[0]


@pytest.mark.parametrize("n", [2, 4])
def test_a_draw_that_fits_is_drawn_as_before(n):
    """Evenly spaced 2 or 4 sources miss the sectors, so the generator gives the same draws either way."""
    new, said = _draw(PositionSampler(fwhm_as=0.35, forbidden_pa=QUAD), n)
    old, _ = _draw(PositionSampler(fwhm_as=0.35, forbidden_pa=QUAD, fallback=False), n)
    assert [[(q.rho, q.theta) for q in d] for d in new] == [[(q.rho, q.theta) for q in d] for d in old]
    assert not said


def test_a_sampler_without_rules_never_counts():
    s = PositionSampler(fwhm_as=0.2)
    rng = np.random.default_rng(1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(50):
            s.sample(4, 0.5, 1.5, rng)
    assert s.n_rule_breaks == 0 and s.n_fallbacks == 0


def test_a_ring_with_no_room_hands_back_the_draw_and_says_so():
    s = PositionSampler(fwhm_as=0.35, forbidden_pa=[(0.0, 100.0), (180.0, 100.0)])     # the whole circle
    draws, said = _draw(s, 3, k=5)
    assert all(_bad(s, d) for d in draws) and s.n_rule_breaks == 5 and s.n_fallbacks == 0
    assert len(said) == 1 and "nor did placing them one at a time" in said[0]


def test_a_run_turns_the_fallback_on_and_an_old_run_keeps_it_off(tmp_path):
    from conftest import build_synthetic_run
    from klip_tpe import CalibrationConfig, Runner, ValidationConfig
    assert RunConfig().sampler_fallback is True
    d = RunConfig(ann_edges=[8, 20]).to_dict()
    assert RunConfig.from_dict(d).sampler_fallback is True
    d.pop("sampler_fallback")
    assert RunConfig.from_dict(d).sampler_fallback is False
    for on in (True, False):
        red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 20], n_iter=[4], n_init=2, param_verify=False,
                                                         calibration=CalibrationConfig(recal_budget=0),
                                                         validation=ValidationConfig(n_top=1, n_valid=1),
                                                         sampler_fallback=on)
        log = []
        r = Runner(red, space, obj, samp, cfg, str(tmp_path / str(on)), log=log.append)
        assert samp.fallback is on
        assert r._sampler_note(0) is None
        samp.n_fallbacks, samp.n_rule_breaks = 3, 1
        line = r._sampler_note(0)
        assert line in log and "3 draw(s) placed one source at a time" in line and "1 draw(s) injected" in line
        assert r._sampler_note(1) is None                                  # each annulus its own count
