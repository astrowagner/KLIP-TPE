"""The README panel's script: RX J0534 by default, through run_rxj0534.py, titled by name.

The run takes two days on four nights of LMIRCam data, so a wrong argument found at the end
costs the run.  These tests check the command line it hands run_rxj0534.py, and that the
display's title replaces the run directory's name.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
readme_panel = pytest.importorskip("readme_panel")


def _opt(argv, name):
    return argv[argv.index(name) + 1]


def test_the_default_is_rxj0534_through_run_rxj0534(monkeypatch, tmp_path):
    seen = {}
    fake = types.ModuleType("run_rxj0534")
    fake.main = lambda argv: (seen.__setitem__("argv", list(argv)), 0)[1]
    monkeypatch.setitem(sys.modules, "run_rxj0534", fake)
    assert readme_panel.main(["--run-dir", str(tmp_path / "readme")]) == 0
    argv = seen["argv"]
    assert _opt(argv, "--partitions") == "4", "the README panel is the four-night search"
    assert _opt(argv, "--n-iter") == "3000"
    assert _opt(argv, "--title") == "RX J0534"
    assert _opt(argv, "--contrast-grid") == "grid"
    assert _opt(argv, "--cg-step") == "0.5", "half-FWHM steps: five separations in the +/- 2 FWHM annulus"
    assert _opt(argv, "--cg-pa") == "12"
    assert _opt(argv, "--out") == str(tmp_path / "readme")
    assert argv[-1] == "--show" and "--panels" not in argv
    assert "--seed" not in argv and "--n-init" not in argv, "run_rxj0534.py's own seed and warm-up"


def test_no_show_still_writes_the_panels(monkeypatch):
    a = readme_panel.parse(["--no-show", "--n-iter", "5000", "--cg-step", "1", "--seed", "4"])
    argv = readme_panel.rxj0534_argv(a)
    assert "--panels" in argv and "--show" not in argv
    assert _opt(argv, "--n-iter") == "5000"
    assert _opt(argv, "--cg-step") == "1"
    assert _opt(argv, "--seed") == "4"
    assert a.run_dir == os.path.expanduser("~/klip_tpe_runs/rxj0534/readme")


def test_hd95086_keeps_its_own_settings():
    a = readme_panel.parse(["--target", "hd95086"])
    assert (a.seed, a.n_init, a.cg_step, a.title) == (3, 200, 1.0, "HD 95086")
    assert a.run_dir == os.path.expanduser("~/klip_tpe_runs/hd95086")
    assert tuple(a.angsep) == (0.1, 2.0) and a.contrast == 3e-6


def test_the_title_replaces_the_run_directory_name(monkeypatch, tmp_path):
    """LiveDisplay(title=...) names the run on every panel and book.  Without it the name is
    the run directory's, as before."""
    import klip_tpe.display as D

    monkeypatch.setattr(D, "annulus_from_runner", lambda runner, recs: types.SimpleNamespace(run_name="readme"))
    monkeypatch.setattr(D, "annulus_from_run", lambda run_dir, ia: types.SimpleNamespace(run_name="readme"))
    fresh = types.SimpleNamespace(history=None, ia=0)
    resumed = types.SimpleNamespace(history=[1, 2], ia=0)              # resumed mid-annulus: read from disk
    for runner in (fresh, resumed):
        titled = D.LiveDisplay(str(tmp_path), save_png=False, movie=False, pdf_every=0, title="RX J0534")
        assert titled._data(runner).run_name == "RX J0534"
        plain = D.LiveDisplay(str(tmp_path), save_png=False, movie=False, pdf_every=0)
        assert plain._data(runner).run_name == "readme"
    assert D.LiveDisplay(str(tmp_path), save_png=False, movie=False, pdf_every=0, title="  ").title is None
