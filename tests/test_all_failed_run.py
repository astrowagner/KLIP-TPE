"""A run in which nothing can be reduced must stop, not finish.

Paper run D, 2026-09-17: the Mac's pyklip 2.10 called ``numpy.reshape(copy=False)`` against a
numpy older than 2.1, so every reduction raised the same TypeError.  The Runner logged
"calibration reduction failed" three times, kept going, logged "evaluation failed" 350
times, wrote a final_results.json with ``winner_index -1`` in both annuli, and the driver
said "D: done in 3 min" -- and would have skipped the stage as finished on the next pass.

Three things now hold that line:

* :meth:`Runner.calibrate` raises when the DEFAULT reduction fails on every attempt of the
  first trial (the search uses the same reducer on the same zone, so nothing could work);
* :meth:`Runner._finish_annulus` raises instead of logging when every evaluation failed;
* ``PyKLIPReducer`` refuses to construct when pyklip needs a newer numpy than is installed,
  with the two commands that fix it.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import build_synthetic_run
from klip_tpe import Runner

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Broken(RuntimeError):
    pass


def test_a_default_reduction_that_always_raises_stops_the_run_at_calibration(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run(n_iter=[6, 6], n_init=3)

    def boom(*a, **k):                     # PartitionedReducer.reduce_config is what Runner._reduce calls
        raise _Broken("reshape() got an unexpected keyword argument 'copy'")
    red.reduce_config = boom
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "run"), log=lambda s: None)
    with pytest.raises(RuntimeError, match="default reduction of annulus 1 failed on all") as ei:
        r.run()
    assert isinstance(ei.value.__cause__, _Broken), "the original error is chained, not swallowed"
    assert not os.path.exists(tmp_path / "run" / "final_results.json"), "no products from a run that never reduced"


def test_a_search_whose_evaluations_all_fail_raises_instead_of_finishing(tmp_path):
    """The calibration passes (the default reduction works), then every searched
    configuration fails.  The annulus must not be 'finished' with winner -1."""
    red, space, obj, samp, cfg = build_synthetic_run(n_iter=[5, 5], n_init=2)
    real = red.reduce_config

    def flaky(cfg_, sources, k_scan=False, tag="", **k):
        # the calibration's reductions are tagged calib_*; everything after that fails
        if str(tag).startswith("calib"):
            return real(cfg_, sources, k_scan, tag, **k)
        raise _Broken("every searched configuration dies")
    red.reduce_config = flaky
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "run"), log=lambda s: None)
    with pytest.raises(RuntimeError, match="every one of the .* evaluations of annulus 1 FAILED"):
        r.run()
    assert not os.path.exists(tmp_path / "run" / "final_results.json")


def test_pyklip_numpy_mismatch_is_caught_at_construction(monkeypatch):
    """Simulate numpy < 2.1 (reshape without copy=) against a pyklip source that uses it."""
    pytest.importorskip("pyklip")
    from klip_tpe.backends import pyklip as backend
    real_reshape = np.reshape

    def old_reshape(a, newshape, order="C", **kw):
        if "copy" in kw:
            raise TypeError("reshape() got an unexpected keyword argument 'copy'")
        return real_reshape(a, newshape, order=order)
    monkeypatch.setattr(np, "reshape", old_reshape)
    import inspect
    monkeypatch.setattr(inspect, "getsource", lambda obj: "x = np.reshape(x, s, copy=False)")
    with pytest.raises(ImportError, match="numpy >= 2.1") as ei:
        backend._check_pyklip_numpy()
    assert "pyklip<2.9" in str(ei.value) and "numpy>=2.1" in str(ei.value), "both remedies are named"
    # and a pyklip that does not use the keyword passes on the old numpy
    monkeypatch.setattr(inspect, "getsource", lambda obj: "x = np.reshape(x, s)")
    backend._check_pyklip_numpy()
    # on a numpy that has the keyword nothing is checked at all
    monkeypatch.setattr(np, "reshape", real_reshape)
    monkeypatch.setattr(inspect, "getsource", lambda obj: (_ for _ in ()).throw(OSError("no source")))
    backend._check_pyklip_numpy()


def test_rerun_paper_retires_a_stage_whose_results_have_no_winner(tmp_path):
    """The driver's skip test reads final_results.json; one with winner_index -1 in every
    annulus is retired to <dir>_failed_<stamp> and the stage runs again."""
    sh = next((p for p in (os.path.join(HERE, "paper_runs", "rerun_paper.sh"),
                           os.path.join(os.path.dirname(HERE), "paper_runs", "rerun_paper.sh"))
               if os.path.exists(p)), None)
    if sh is None:
        pytest.skip("paper_runs/ not next to the package")
    src = open(sh).read()
    assert "finished()" in src and "_failed_" in src
    # run just the finished() function against two synthetic result files
    fn = src[src.index("finished() {"):src.index("\n}\n", src.index("finished() {")) + 3]
    import json
    good, bad = tmp_path / "good", tmp_path / "bad"
    good.mkdir(); bad.mkdir()
    json.dump({"annuli": [{"winner_index": 3, "search_best_index": 3}]}, open(good / "final_results.json", "w"))
    json.dump({"annuli": [{"winner_index": -1, "search_best_index": -1}, {"winner_index": -1, "search_best_index": -1}]},
              open(bad / "final_results.json", "w"))
    for d, code in ((good, 0), (bad, 2), (tmp_path / "missing", 1)):
        rc = subprocess.run(["bash", "-c", fn + f'\nfinished "{d}"'], capture_output=True).returncode
        assert rc == code, f"{d.name}: exit {rc}, expected {code}"


def test_pyklip_progress_bars_are_disabled_by_the_backend(monkeypatch):
    """klip_parallelized draws a tqdm bar per call whatever `verbose` says; in a notebook
    with ipywidgets that is one widget per reduction (tutorial 03: 426 of them, 3.7 MB of
    widget state).  The backend swaps pyklip's trange/tqdm for disabled ones, once."""
    par = pytest.importorskip("pyklip.parallelized")
    from klip_tpe.backends import pyklip as backend
    monkeypatch.delenv("KLIP_TPE_PYKLIP_PROGRESS", raising=False)
    monkeypatch.setattr(par, "_klip_tpe_quiet", False, raising=False)
    backend._quiet_pyklip_progress(par)
    bar = par.trange(3)
    assert getattr(bar, "disable", False) is True, "pyklip's trange must be a disabled tqdm"
    assert list(bar) == [0, 1, 2], "and still iterate"
    assert getattr(par.tqdm(range(2)), "disable", False) is True
    # an explicit request keeps them
    monkeypatch.setenv("KLIP_TPE_PYKLIP_PROGRESS", "1")
    monkeypatch.setattr(par, "_klip_tpe_quiet", False, raising=False)
    from tqdm.auto import trange as real
    par.trange = real
    backend._quiet_pyklip_progress(par)
    assert par.trange is real
