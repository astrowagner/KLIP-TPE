"""``klip-tpe near`` / ``generic`` check every searched dimension before searching.

The JWST drivers have refused to start with a dead dimension since 2026-09-23, but the NEAR
launcher -- the one with the 56-dimensional space whose 18 frame-selection dimensions the
IDL workers once silently ignored (the ``parstr`` omission) -- never called the check.  Now
``RunConfig.liveness_check`` runs it before a new run's first evaluation and the command
line turns it on.  What must hold:

* a dimension the reducer never receives stops the run before any evaluation, naming it;
* a healthy space runs, and the check does not move the search RNG: a seeded run is the same
  run with or without it;
* a resumed run is not checked again;
* the command line turns it on, ``--no-liveness-check`` off; the Python API default is off.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from conftest import PARTS, build_synthetic_run
from klip_tpe import Param, RunConfig, SearchSpace
from klip_tpe import cli
from klip_tpe.runner import Runner
from klip_tpe.space import Config, kgrid

QUIET = lambda s: None  # noqa: E731


def _space_with_an_ignored_dimension():
    block = [Param("bin", 5, 30, "int", default=12),
             Param("k_klip", 1, 30, "int", grid=kgrid(30), default=5),
             Param("ignored", 0, 5, "int", default=2)]
    return SearchSpace().replicate(block, PARTS).with_selection("two_slot", partitions=PARTS)


def _deaf_to(runner, name):
    """The reducer never receives ``name`` -- the parstr bug, reproduced."""
    orig = runner._reduce

    def deaf(cfg_, sources, *a, **kw):
        per = {p: {k: v for k, v in d.items() if k != name} for p, d in cfg_.per_partition.items()}
        params = {k: v for k, v in cfg_.params.items() if k != name}
        return orig(Config(params, per, cfg_.selected, cfg_.x), sources, *a, **kw)
    runner._reduce = deaf


def test_a_dimension_the_reducer_never_receives_stops_the_run(tmp_path):
    red, _, obj, samp, cfg = build_synthetic_run(seed=2, ann_edges=[8, 30], n_iter=4, n_init=2,
                                                 liveness_check=True)
    d = str(tmp_path / "dead")
    logs = []
    r = Runner(red, _space_with_an_ignored_dimension(), obj, samp, cfg, d, log=logs.append)
    _deaf_to(r, "ignored")
    with pytest.raises(SystemExit) as e:
        r.run()
    msg = str(e.value)
    assert "ignored_n1" in msg and "ignored_n2" in msg and "bin_n1" not in msg
    assert not os.path.exists(os.path.join(d, "results.jsonl")), "not one evaluation was made"
    assert any("ignored_n1" in l and "DEAD" in l for l in logs)


def test_a_healthy_space_runs_and_the_check_leaves_the_rng_alone(tmp_path):
    runs = {}
    for check in (False, True):
        red, space, obj, samp, cfg = build_synthetic_run(seed=4, ann_edges=[8, 30], n_iter=6, n_init=3,
                                                         liveness_check=check, save_eval_images=False,
                                                         fm_curve=False, fm_preview=False)
        logs = []
        r = Runner(red, space, obj, samp, cfg, str(tmp_path / f"run_{check}"), log=logs.append)
        r.run()
        runs[check] = (np.asarray(r.history.y, float), logs)
    np.testing.assert_allclose(runs[False][0], runs[True][0], atol=1e-12)
    assert any("liveness: all" in l for l in runs[True][1])
    assert not any("liveness" in l for l in runs[False][1])


def test_a_resumed_run_is_not_checked_again(tmp_path, monkeypatch):
    red, space, obj, samp, cfg = build_synthetic_run(seed=5, ann_edges=[8, 30], n_iter=6, n_init=3,
                                                     liveness_check=True)
    d = str(tmp_path / "cut")
    r = Runner(red, space, obj, samp, cfg, d, log=QUIET)

    class Interrupt(BaseException):
        pass

    orig = r._reduce

    def cut(cfg_, src, tag="", **kw):
        if tag.startswith("a1_e4"):
            raise Interrupt
        return orig(cfg_, src, tag=tag, **kw)
    r._reduce = cut
    with pytest.raises(Interrupt):
        r.run()
    calls = []
    monkeypatch.setattr(Runner, "_liveness_preflight", lambda self: calls.append(1))
    r2 = Runner.resume(d, red, obj, samp, log=QUIET)
    assert r2.cfg.liveness_check is True, "the setting is the run's own"
    r2.run()
    assert calls == [], "the resume re-ran the pre-flight"


@pytest.mark.parametrize("argv", [["near", "--root", "/data/NEAR2_py"],
                                  ["generic", "--cube", "c.fits", "--angles", "a.fits"]])
def test_the_command_line_turns_it_on(argv):
    a = cli._build_parser().parse_args(argv)
    assert cli._protocol_config(a)["liveness_check"] is True
    a = cli._build_parser().parse_args(argv + ["--no-liveness-check"])
    assert cli._protocol_config(a)["liveness_check"] is False
    assert RunConfig.__dataclass_fields__["liveness_check"].default is False


# ------------------------------------------------------------------ narrow live bands

def _space_with(param):
    block = [Param("bin", 5, 30, "int", default=12),
             Param("k_klip", 1, 30, "int", grid=kgrid(30), default=5), param]
    return SearchSpace().replicate(block, PARTS).with_selection("two_slot", partitions=PARTS)


def _acts_only_in(runner, name, lo, hi):
    """The reduction sees ``name`` only inside (lo, hi): a frame-cut threshold above every
    frame's tag ratio cuts nothing, and one far below it empties the night and is snapped
    back -- live in a band, inert at both bounds and the midpoint."""
    orig = runner._reduce

    def banded(cfg_, sources, *a, **kw):
        vals = [d.get(name) for d in cfg_.per_partition.values()] + [cfg_.params.get(name)]
        per = {p: {k: v for k, v in d.items() if k != name} for p, d in cfg_.per_partition.items()}
        params = {k: v for k, v in cfg_.params.items() if k != name}
        ev = orig(Config(params, per, cfg_.selected, cfg_.x), sources, *a, **kw)
        if any(v is not None and lo < float(v) < hi for v in vals):
            ev.image = ev.image * 1.01
        return ev
    runner._reduce = banded


def test_a_threshold_live_only_in_a_narrow_band_is_not_called_dead(tmp_path):
    """noise_max on NEAR2 nights 3-5: range [0.3, 3.0], default 1.45, live only in about
    [1.22, 1.35].  The bounds and the midpoint (1.65) all miss it."""
    from klip_tpe.liveness import check_live_dimensions
    red, _, obj, samp, cfg = build_synthetic_run(seed=2, ann_edges=[8, 30], n_iter=4, n_init=2)
    sp = _space_with(Param("thr", 0.3, 3.0, default=1.45))
    r = Runner(red, sp, obj, samp, cfg, str(tmp_path / "band"), log=QUIET)
    _acts_only_in(r, "thr", 1.2, 1.36)
    logs = []
    first = check_live_dimensions(r, log=logs.append, dense=0)
    assert not first["thr_n1"]["live"], "the premise: bounds and midpoint alone miss the band"
    logs = []
    st = check_live_dimensions(r, log=logs.append)
    assert st["thr_n1"]["live"] and st["thr_n2"]["live"]
    a, b = st["thr_n1"]["live_at"]
    assert 1.2 < b < 1.36
    assert any("thr_n1" in l and "band" in l for l in logs)
    assert all(st[n]["live"] for n in ("bin_n1", "k_klip_n1"))


def test_a_dimension_inert_everywhere_is_still_dead_after_the_grid(tmp_path):
    from klip_tpe.liveness import check_live_dimensions, dead_dimensions
    red, _, obj, samp, cfg = build_synthetic_run(seed=2, ann_edges=[8, 30], n_iter=4, n_init=2)
    sp = _space_with(Param("thr", 0.3, 3.0, default=1.45))
    r = Runner(red, sp, obj, samp, cfg, str(tmp_path / "inert"), log=QUIET)
    _deaf_to(r, "thr")
    logs = []
    st = check_live_dimensions(r, log=logs.append, dense=11)
    assert dead_dimensions(st) == ["thr_n1", "thr_n2"]
    assert st["thr_n1"]["tested"] >= 3 * 10, "the grid was tried at every point"
    assert any("thr_n1" in l and "DEAD" in l and "grid" in l for l in logs)


def test_the_preflight_lets_a_banded_threshold_through(tmp_path):
    red, _, obj, samp, cfg = build_synthetic_run(seed=2, ann_edges=[8, 30], n_iter=4, n_init=2,
                                                 liveness_check=True, save_eval_images=False,
                                                 fm_curve=False, fm_preview=False)
    sp = _space_with(Param("thr", 0.3, 3.0, default=1.45))
    logs = []
    r = Runner(red, sp, obj, samp, cfg, str(tmp_path / "ok"), log=logs.append)
    _acts_only_in(r, "thr", 1.2, 1.36)
    r.run()
    assert any("liveness: all" in l for l in logs)
    assert os.path.exists(os.path.join(str(tmp_path / "ok"), "results.jsonl"))
