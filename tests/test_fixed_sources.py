"""Where the search's injected sources go: fixed separations, rotating position angles.

This is what ``optimize_near_2_tpe`` does, and its own comment in ``near2m_randpos`` is the
specification:

    DETERMINISTIC radial anchor: for a given band [rlo,rhi] and count n the source
    SEPARATIONS are always the same set (a centered ladder rlo+(i+0.5)*rspan/n), so every
    eval in an annulus injects at identical radii -- best and test separations match and the
    median-S/N score is not biased by radial throughput/noise luck.  Only the AZIMUTH anchor
    (th0) stays random, so sources still rotate eval-to-eval (anti-gaming) without any
    radial effect.

``th0 = randomu(seed)*360.``, and ``near2m_randpos`` is called from inside
``for it=it_start, n_iter-1`` at four call sites.  :class:`PositionSampler`'s ``"spread"``
strategy reproduces it line for line.

The mistake this file now guards against
----------------------------------------
On 2026-09-13 a ``fixed_sources`` flag was added and defaulted ON, freezing the azimuths too,
on the belief that the per-evaluation redraw was a porting bug.  The evidence offered was
real -- the same configuration on a six-night NEAR run scored 8.45, 3.17, 8.56, 6.08, 6.69,
6.75 -- but the conclusion was wrong.  The reference accepts that variance deliberately; its
defence is the validated election, which re-scores the top candidates at FRESH positions.
Freezing the azimuths instead lets the optimizer tune to the speckle realisation at those
particular angles, which is precisely the gaming the comment names.  A NEAR run reached 2037
evaluations injecting at PA 87.39 / 267.39 before it was caught.

So the flag survives as an opt-in -- pairing every comparison on one realisation is a
legitimate measurement of how much of a search's gain is realisation luck -- but the default
is off, and these tests say so in a way that a future "fix" cannot quietly undo.
"""
import json
import os

import numpy as np
import pytest

from klip_tpe import RunConfig, Runner, Source
from conftest import build_synthetic_run


def _runner(tmp_path, **cfg_kw):
    red, space, obj, samp, cfg = build_synthetic_run(**cfg_kw)
    return Runner(red, space, obj, samp, cfg, str(tmp_path), log=lambda s: None)


def _draws(r, n=6, ia=0):
    """``n`` successive search draws, as ``n`` successive evaluations would make them."""
    rlo, rhi = r._band(ia, None)
    return [r.sampler.sample(r._nsrc(ia), rlo, rhi, r.rng, 3e-5) for _ in range(n)]


# --------------------------------------------------------------------- the reference rule

def test_the_reference_default_is_to_redraw_the_angles():
    """The default must stay off.  This assertion is the whole point of the file."""
    assert RunConfig().fixed_sources is False


def test_separations_are_identical_across_evaluations(tmp_path):
    r = _runner(tmp_path)
    radii = {tuple(round(s.rho, 9) for s in src) for src in _draws(r)}
    assert len(radii) == 1, f"the radial ladder moved between evaluations: {radii}"


def test_position_angles_rotate_across_evaluations(tmp_path):
    r = _runner(tmp_path)
    pas = {tuple(round(s.theta, 6) for s in src) for src in _draws(r)}
    assert len(pas) > 1, "the azimuth anchor did not rotate -- this is the 2026-09-13 mistake"


def test_the_ladder_is_the_idl_formula(tmp_path):
    """``r0 = rlo + 0.5*span/n``; ``rho[i] = rlo + ((r0-rlo) + i*span/n) mod span``."""
    r = _runner(tmp_path)
    rlo, rhi = r._band(0, None)
    n = r._nsrc(0)
    span = rhi - rlo
    if span <= 0:
        pytest.skip("this annulus collapses to one radius (two-source convention)")
    r0 = rlo + 0.5 * span / n
    want = [rlo + (((r0 - rlo) + i * span / n) % span) for i in range(n)]
    got = [s.rho for s in _draws(r, n=1)[0]]
    assert got == pytest.approx(want)


def test_a_whole_run_keeps_one_radius_set_and_many_angles(tmp_path):
    """End to end, which is the form the NEAR failure took: 2037 evaluations, one PA set."""
    red, space, obj, samp, cfg = build_synthetic_run(n_iter=16, n_init=8, seed=11)
    Runner(red, space, obj, samp, cfg, str(tmp_path), log=lambda s: None).run()
    rows = [json.loads(l) for l in open(os.path.join(str(tmp_path), "results.jsonl")) if l.strip()]
    search = [w for w in rows if w.get("phase") not in ("valid", "validation", "pv")]
    assert len(search) > 5
    per_ann = {}
    for w in search:
        radii = tuple(round(float(s[0]), 9) for s in w["sources"])
        pas = tuple(round(float(s[1]), 6) for s in w["sources"])
        d = per_ann.setdefault(int(w["annulus"]), {"radii": set(), "pas": set(), "n": 0})
        d["radii"].add(radii); d["pas"].add(pas); d["n"] += 1
    assert per_ann
    for ia, d in sorted(per_ann.items()):
        assert len(d["radii"]) == 1, f"annulus {ia + 1}: {len(d['radii'])} radius sets"
        assert len(d["pas"]) > 1, (f"annulus {ia + 1}: one PA set across {d['n']} evaluations "
                                   f"-- the azimuths are frozen")


def test_repeating_one_configuration_varies_and_that_is_deliberate(tmp_path):
    """The variance that prompted the mistake, asserted as the expected behaviour.

    Scoring the identical vector repeatedly gives different answers because the injections
    rotate.  That is the reference's design; the validated election is what handles it.
    """
    red, space, obj, samp, cfg = build_synthetic_run(seed=5)
    x = space.default_vector()
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "ref"), log=lambda s: None)
    r.ia, r.contrast = 0, 3e-5
    ref = [float(r.evaluate(x, "t", raw_only=True)[0].score) for _ in range(4)]

    red2, space2, obj2, samp2, cfg2 = build_synthetic_run(seed=5, fixed_sources=True)
    r2 = Runner(red2, space2, obj2, samp2, cfg2, str(tmp_path / "frozen"), log=lambda s: None)
    r2.ia, r2.contrast = 0, 3e-5
    frozen = [float(r2.evaluate(x, "t", raw_only=True)[0].score) for _ in range(4)]

    assert np.std(frozen) < 1e-6, f"frozen positions still varied: {frozen}"
    assert np.std(ref) > np.std(frozen), (
        f"the reference objective should vary: ref {np.std(ref):.4f} vs frozen {np.std(frozen):.4f}")


# ------------------------------------------------------------- the opt-in, when asked for

def test_the_opt_in_freezes_both(tmp_path):
    r = _runner(tmp_path, fixed_sources=True)
    r.ia = 0
    a = r.search_sources(0, 3e-5)
    b = r.search_sources(0, 3e-5)
    assert [(s.rho, s.theta) for s in a] == [(s.rho, s.theta) for s in b]


def test_the_opt_in_reaches_a_whole_run(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run(n_iter=14, n_init=6, seed=11,
                                                     fixed_sources=True)
    Runner(red, space, obj, samp, cfg, str(tmp_path), log=lambda s: None).run()
    rows = [json.loads(l) for l in open(os.path.join(str(tmp_path), "results.jsonl")) if l.strip()]
    search = [w for w in rows if w.get("phase") not in ("valid", "validation", "pv")]
    by_ann = {}
    for w in search:
        key = tuple(round(float(v), 9) for s in w["sources"] for v in s[:2])
        by_ann.setdefault(int(w["annulus"]), set()).add(key)
    for ia, pos in sorted(by_ann.items()):
        assert len(pos) == 1, f"annulus {ia + 1}: {len(pos)} sets with fixed_sources on"


def test_each_annulus_gets_its_own_frozen_set(tmp_path):
    r = _runner(tmp_path, ann_edges=[8, 20, 30], fixed_sources=True)
    a = [(s.rho, s.theta) for s in r.search_sources(0, 3e-5)]
    b = [(s.rho, s.theta) for s in r.search_sources(1, 3e-5)]
    assert a != b, "both annuli injected at the same place"
    assert [(s.rho, s.theta) for s in r.search_sources(0, 3e-5)] == a, "annulus 0 was redrawn"


def test_the_contrast_is_restamped_without_moving_the_frozen_sources(tmp_path):
    """Calibration changes the injected brightness many times; that must not move them."""
    r = _runner(tmp_path, fixed_sources=True)
    a = r.search_sources(0, 1e-5)
    b = r.search_sources(0, 9e-4)
    assert [(s.rho, s.theta) for s in a] == [(s.rho, s.theta) for s in b]
    assert [s.contrast for s in b] == [pytest.approx(9e-4)] * len(b)


def test_the_frozen_positions_survive_a_resume(tmp_path):
    """Drawn from the run seed and the annulus index, never from the search's own rng --
    otherwise a resumed paired run would carry on against a different realisation."""
    r1 = _runner(tmp_path / "a", seed=7, fixed_sources=True)
    first = [(s.rho, s.theta) for s in r1.search_sources(0, 3e-5)]
    for _ in range(50):                         # advance the search rng, as a search would
        r1.rng.random(16)
    r2 = _runner(tmp_path / "b", seed=7, fixed_sources=True)
    assert [(s.rho, s.theta) for s in r2.search_sources(0, 3e-5)] == first


def test_a_different_seed_gives_different_frozen_positions(tmp_path):
    a = [(s.rho, s.theta) for s in
         _runner(tmp_path / "a", seed=1, fixed_sources=True).search_sources(0, 3e-5)]
    b = [(s.rho, s.theta) for s in
         _runner(tmp_path / "b", seed=2, fixed_sources=True).search_sources(0, 3e-5)]
    assert a != b


# ---------------------------------------------------------------------------- opt_width

def test_opt_width_draws_per_configuration(tmp_path):
    """``opt_width`` searches the annulus width, so each configuration's injections belong at
    *its* zone's mid-radius -- there is no single band, and the radius follows the width."""
    from klip_tpe import CalibrationConfig, Param, ValidationConfig
    red, space, obj, samp, cfg = build_synthetic_run(
        ann_edges=[8, 99], n_iter=6, n_init=3, opt_width=True, width_range=(10, 20), r_cap=40,
        param_verify=False, stitch_every=0, calibration=CalibrationConfig(recal_budget=0),
        validation=ValidationConfig(n_top=1, n_valid=1))
    space.add(Param("width", 10, 20, "int", default=15))
    Runner(red, space, obj, samp, cfg, str(tmp_path), log=lambda s: None).run()
    rows = [json.loads(l) for l in open(os.path.join(str(tmp_path), "results.jsonl")) if l.strip()]
    widths, radii = set(), set()
    for w in rows:
        wd = w.get("config", {}).get("params", {}).get("width")
        if wd is None or not w.get("sources"):
            continue
        widths.add(round(float(wd), 6))
        radii.add(round(float(w["sources"][0][0]), 6))
    assert len(widths) > 1, "the test needs the width to actually vary"
    assert len(radii) > 1, "injection radius did not follow the searched width"
