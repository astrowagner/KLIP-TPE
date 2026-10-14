"""The injection floor on the command line, and a log line when one source starves its ring.

From the NEAR2 note of 2026-10-13: NEAR's AGPM reaches 0.9 lambda/D, and the next run wants
no injection inside 1.0 lambda/D rather than the sampler's 1.5 FWHM, which nothing on the
command line reached.  What must hold:

* ``--min-r-fwhm`` reaches ``RunConfig.min_r_fwhm`` from ``near`` and ``generic`` (1.5 by
  default), and a negative value is refused;
* the run's one sampler takes the floor, so every stage injects with it, and a resume puts it
  back on whatever sampler the caller built;
* when the count used still leaves a source's Mawet ring fewer than ``min_ring`` clean
  apertures (one source close to the star), the log says so.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from klip_tpe import (CalibrationConfig, GaussianPSF, KLIPReducer, MawetPeakSNR, Objective, Param,
                      PartitionedReducer, RunConfig, Runner, SearchSpace, ValidationConfig, cli)
from klip_tpe.positions import PositionSampler
from klip_tpe.synthetic import synthetic_klip_dataset

from conftest import build_synthetic_run


def _args(*extra, cmd="near"):
    base = ["near", "--root", "data"] if cmd == "near" else ["generic", "--cube", "c.fits"]
    return cli._build_parser().parse_args(base + list(extra))


def test_the_option_reaches_the_run_config():
    assert cli._protocol_config(_args())["min_r_fwhm"] == 1.5                # as before
    assert cli._protocol_config(_args("--min-r-fwhm", "1.0"))["min_r_fwhm"] == 1.0
    assert cli._protocol_config(_args("--min-r-fwhm", "0.8", cmd="generic"))["min_r_fwhm"] == 0.8
    with pytest.raises(SystemExit, match="0 or more"):
        cli._protocol_config(_args("--min-r-fwhm", "-1"))


def test_the_run_sampler_takes_the_floor_and_a_resume_puts_it_back(tmp_path):
    red, space, obj, samp, cfg = build_synthetic_run(
        ann_edges=[8, 30], n_iter=3, n_init=2, min_r_fwhm=1.0, param_verify=False,
        calibration=CalibrationConfig(recal_budget=0), validation=ValidationConfig(n_top=1, n_valid=1))
    assert samp.min_r_fwhm == 1.5
    r = Runner(red, space, obj, samp, cfg, str(tmp_path), log=lambda s: None)
    assert samp.min_r_fwhm == 1.0, "the run's one sampler"
    rng = np.random.default_rng(0)
    assert samp.sample(1, 0.0, 0.0, rng)[0].rho == pytest.approx(1.0 * samp.fwhm_as)
    r.run()
    setup = json.load(open(tmp_path / "run_setup.json"))
    assert setup["config"]["min_r_fwhm"] == 1.0 and setup["sampler"]["min_r_fwhm"] == 1.0
    fresh = PositionSampler(fwhm_as=samp.fwhm_as)
    Runner.resume(str(tmp_path), red, obj, fresh, log=lambda s: None)
    assert fresh.min_r_fwhm == 1.0, "a resume injects with the floor the run started with"
    # and so does a script's own Runner over the run's directory, which adopts the checkpoint
    mine = PositionSampler(fwhm_as=samp.fwhm_as)
    scripted = Runner(red, space, obj, mine, RunConfig(ann_edges=[8, 30]), str(tmp_path), log=lambda s: None)
    assert mine.min_r_fwhm == 1.5
    scripted._auto_resume()
    assert mine.min_r_fwhm == 1.0
    # a configuration from before the option leaves the caller's sampler alone
    old = RunConfig.from_dict({k: v for k, v in setup["config"].items() if k != "min_r_fwhm"})
    other = PositionSampler(fwhm_as=samp.fwhm_as)
    Runner(red, space, obj, other, old, str(tmp_path / "old"), log=lambda s: None)
    assert old.min_r_fwhm is None and other.min_r_fwhm == 1.5


def _klip_runner(tmp_path, edges, rmin):
    px = 0.05
    ds = synthetic_klip_dataset(nframes=20, size=64, pa_span=60, seed=1)
    kr = KLIPReducer(ds, px, 3.89 * px / 206265 * 8.2, 8.2, outrad_cap=28,
                     injection_model=GaussianPSF(4.0, star_flux=1e4))
    red = PartitionedReducer({"n1": kr})
    space = SearchSpace()
    space.add(Param("k_klip", 1, 10, "int", default=5))
    obj = Objective(MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm))
    logs = []
    r = Runner(red, space, obj, PositionSampler(fwhm_as=red.fwhm * red.pxscale),
               RunConfig(ann_edges=edges, n_sources=1, min_r_fwhm=rmin, seed=1), str(tmp_path), log=logs.append)
    return r, logs


def test_one_source_that_starves_its_ring_is_logged(tmp_path):
    r, logs = _klip_runner(tmp_path / "close", [2, 9], 0.3)        # one source at 1.4 FWHM
    assert r._nsrc(0) == 1
    said = [l for l in logs if "Mawet ring" in l]
    assert len(said) == 1 and "one source at 0.275\"" in said[0] and "radial band" in said[0], logs
    assert r._nsrc(0) == 1 and len([l for l in logs if "Mawet ring" in l]) == 1, "once per annulus"
    r2, logs2 = _klip_runner(tmp_path / "wide", [8, 30], 1.0)
    assert r2._nsrc(0) == 1 and not [l for l in logs2 if "Mawet ring" in l]
