"""A finished run's winners reduced again on other frames (``Runner.rereduce``,
``klip-tpe rereduce``).

What must hold:

* On the run's own frames the re-reduction gives the run's products: the stitch, its S/N map,
  the injected and per-night stitches, to the bit, and the same final verify.
* The run directory is only read.
* On other frames (a line subtracted, planets added) the products follow the frames, at the
  run's stitch weights, and ``rereduce.json`` says what the frames were.
* The final verify takes its injections from a given list (``verify_sources``).
* A plain resume on other frames is still refused, and so are a re-reduction of an unfinished
  run, a verify source outside the run's annuli and an output inside the run.
* On the command line, ``klip-tpe rereduce`` writes beside the run, keeps the run's frames when
  no data option is typed, and does not write over an earlier re-reduction unless told to
  (slow).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

import numpy as np
import pytest

from klip_tpe import CalibrationConfig, MawetPeakSNR, Objective, Param, PositionSampler, RunConfig, SearchSpace
from klip_tpe import ValidationConfig
from klip_tpe.injection import GaussianPSF, inject_sources
from klip_tpe.metrics import Source, line_mask, source_xy, star_center
from klip_tpe.reducer import Dataset, KLIPReducer, PartitionedReducer
from klip_tpe.runner import Runner
from klip_tpe.stitch import read_fits
from klip_tpe.synthetic import synthetic_klip_dataset

PX = 0.05
LAM = 3.89 * PX / 206265 * 8.2                       # FWHM about 4 px
LINE = [Source(round(0.35 + 0.025 * i, 4), pa, 1.2e-3) for i in range(37) for pa in (13.0, 193.0)]
PLANETS = [Source(0.62, 100.0, 4e-2), Source(1.05, 250.0, 4e-2)]
VERIFY = [(0.55, 60.0, 2.5e-2), (1.10, 300.0, 2.5e-2)]
FILES = ("klip_stitched.fits", "klip_stitched_snr.fits", "klip_stitched_inj.fits", "klip_stitched_snr_inj.fits",
         "klip_stitched_nights.fits.gz", "klip_stitched_nights_inj.fits.gz")


def _model():
    return GaussianPSF(4.0, star_flux=1e4)


def _night(seed):
    ds = synthetic_klip_dataset(nframes=36, size=64, pa_span=80, seed=seed)
    base = KLIPReducer(ds, PX, LAM, 8.2, injection_model=_model(), outrad_cap=30)
    cube = inject_sources(ds.cube, ds.angles, LINE, _model(), PX, truenorth=base.truenorth,
                          angle_convention=base.angle_convention)
    return Dataset(cube, ds.angles, ds.tags, texp=ds.texp, name=f"n{seed}", meta=dict(ds.meta or {}))


def _reducer(subtract=None, add=None):
    reds = {}
    for seed in (3, 4):
        r = KLIPReducer(_night(seed), PX, LAM, 8.2, injection_model=_model(), outrad_cap=30)
        reds[f"n{seed}"] = r
    pr = PartitionedReducer(reds, max_workers=1, log=lambda s: None)
    if subtract:
        pr.subtract_sources(subtract)
    if add:
        pr.inject_sources(add)
    return pr


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """A finished two-night, two-annulus run on data holding a line, with its final verify."""
    red = _reducer()
    space = SearchSpace().replicate([Param("k_klip", 2, 10, "int", default=5)], ["n3", "n4"])
    obj = Objective(MawetPeakSNR(pxscale=PX, fwhm=4.0), clean_subtract=True)
    samp = PositionSampler(fwhm_as=4.0 * PX, forbidden_pa=[(13.0, 20.0), (193.0, 20.0)])
    cfg = RunConfig(ann_edges=[7, 16, 27], n_iter=3, n_init=2, seed=2, n_sources=1, param_verify=False,
                    validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[3e-2]),
                    fm_curve=False, write_setup_files=False, verify=True, verify_n_boot=10)
    run = tmp_path_factory.mktemp("rereduce") / "run"
    Runner(red, space, obj, samp, cfg, str(run), log=lambda s: None).run()
    return {"run": str(run), "obj": obj, "samp": samp}


def _snap(d):
    return {str(p.relative_to(d)): (p.stat().st_mtime_ns, hashlib.md5(p.read_bytes()).hexdigest())
            for p in sorted(pathlib.Path(d).rglob("*")) if p.is_file()}


def _report_body(path):
    """The verify report without its time stamp and run line."""
    return [ln for ln in open(path).read().splitlines() if not ln.startswith(("# klip_tpe.verify", "# run:"))]


def test_on_the_runs_own_frames_the_products_are_the_runs(finished, tmp_path):
    run = finished["run"]
    before = _snap(run)
    r = Runner.resume(run, _reducer(), finished["obj"], finished["samp"], log=lambda s: None)
    rec = r.rereduce(str(tmp_path / "same"), log=lambda s: None)
    for name in FILES:
        a, b = read_fits(os.path.join(run, name)), read_fits(str(tmp_path / "same" / name))
        assert a is not None and b is not None, name
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True), name
    assert _report_body(os.path.join(run, "stitched_verify_report.txt")) == _report_body(
        str(tmp_path / "same" / "stitched_verify_report.txt"))
    fr = json.load(open(os.path.join(run, "final_results.json")))
    assert rec["stitch_weights"] == fr["stitch_weights"] and rec["weights_from"] == "final_results.json"
    assert rec["frames"] == rec["run_frames"] == "the default frames"
    assert _snap(run) == before                                   # the run is only read


def test_other_frames_follow_the_frames_at_the_runs_weights(finished, tmp_path):
    run = finished["run"]
    before = _snap(run)
    red = _reducer(subtract=LINE, add=PLANETS)
    with pytest.raises(ValueError, match="cannot change the frames"):          # a resume still refuses
        Runner.resume(run, red, finished["obj"], finished["samp"], log=lambda s: None)
    r = Runner.resume(run, red, finished["obj"], finished["samp"], log=lambda s: None, same_frames=False)
    rec = r.rereduce(str(tmp_path / "edited"), verify_sources=VERIFY, log=lambda s: None)
    assert "74 subtracted source(s)" in rec["frames"] and "2 added source(s)" in rec["frames"]
    assert rec["run_frames"] == "the default frames"
    assert rec["data_edits"]["injected_sources"] == [list(s.as_tuple()) for s in PLANETS]
    assert rec["stitch_weights"] == json.load(open(os.path.join(run, "final_results.json")))["stitch_weights"]
    old = np.asarray(read_fits(os.path.join(run, "klip_stitched.fits")), float)
    new = np.asarray(read_fits(str(tmp_path / "edited" / "klip_stitched.fits")), float)
    # the line is gone from its strip ...
    yy, xx = np.mgrid[0:64, 0:64]
    cx, cy = star_center((64, 64))
    rr = np.hypot(xx - cx, yy - cy)
    strip = line_mask((64, 64), [(13.0, 3.0)]) & (rr > 9) & (rr < 25) & np.isfinite(old) & np.isfinite(new)
    assert np.mean(new[strip] ** 2) < 0.2 * np.mean(old[strip] ** 2)
    # ... and the planets added at load are in the clean stitch (S/N about 0 there in the run's)
    snr = np.asarray(read_fits(str(tmp_path / "edited" / "klip_stitched_snr.fits")), float)
    snr0 = np.asarray(read_fits(os.path.join(run, "klip_stitched_snr.fits")), float)
    for p in PLANETS:
        x, y = source_xy([p.rho], [p.theta], PX, cx, cy, "pa")
        box = (slice(int(round(y[0])) - 2, int(round(y[0])) + 3), slice(int(round(x[0])) - 2, int(round(x[0])) + 3))
        assert np.nanmax(snr[box]) > 3.5 and np.nanmax(snr[box]) > np.nanmax(snr0[box]) + 3
    # the verify's injections are the given ones, each listed once
    txt = open(tmp_path / "edited" / "stitched_verify_report.txt").read()
    assert txt.count("injection calibration (2 sources)") == 1
    for rho, pa, _ in VERIFY:
        assert f'rho={rho:.3f}" PA={pa:.1f}' in txt
    assert "reduced again on" in txt
    assert _snap(run) == before


def test_what_it_refuses(finished, tmp_path):
    run = finished["run"]
    r = Runner.resume(run, _reducer(), finished["obj"], finished["samp"], log=lambda s: None)
    with pytest.raises(ValueError, match="inside the run"):
        r.rereduce(os.path.join(run, "again"), log=lambda s: None)
    with pytest.raises(ValueError, match="outside the run's annuli"):
        r.rereduce(str(tmp_path / "x"), verify_sources=[(1.5, 30.0, 1e-2)], log=lambda s: None)
    space = SearchSpace().replicate([Param("k_klip", 2, 10, "int", default=5)], ["n3", "n4"])
    fresh = Runner(_reducer(), space, finished["obj"], finished["samp"], RunConfig(ann_edges=[7, 16]),
                   str(tmp_path / "new"), log=lambda s: None)
    with pytest.raises(ValueError, match="has not finished"):
        fresh.rereduce(str(tmp_path / "y"), log=lambda s: None)


@pytest.mark.slow
def test_rereduce_on_the_command_line(tmp_path, monkeypatch):
    """generic cubes: a run on data holding a line, then ``klip-tpe rereduce`` with the line
    subtracted, planets added and the verify's injections given."""
    from astropy.io import fits
    from klip_tpe import cli
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))       # keep the run record here
    ds = synthetic_klip_dataset(nframes=40, size=64, pa_span=90, seed=3)
    psf = np.exp(-0.5 * ((np.mgrid[0:21, 0:21] - 10.0) ** 2).sum(0) / (4.0 / 2.3548) ** 2)
    for name, arr in (("angles", ds.angles), ("psf", psf)):
        fits.writeto(tmp_path / f"{name}.fits", arr)
    data = ["--instrument", "generic", "--angles", str(tmp_path / "angles.fits"), "--psf", str(tmp_path / "psf.fits"),
            "--pxscale", str(PX), "--lam", str(LAM), "--diam", "8.2", "--star-flux", "1e4", "--workers", "1",
            "--no-tags", "--names", "line"]
    fits.writeto(tmp_path / "base.fits", ds.cube)
    a = cli._build_parser().parse_args(["generic", "--cube", str(tmp_path / "base.fits"), *data[2:]])
    red, *_ = cli._generic_objects(a, log=lambda s: None)
    sub = next(iter(red.reducers.values()))
    fits.writeto(tmp_path / "line.fits", inject_sources(ds.cube, ds.angles, LINE, sub.model, PX,
                                                        truenorth=sub.truenorth, angle_convention=sub.angle_convention))
    data += ["--cube", str(tmp_path / "line.fits")]
    run = tmp_path / "run"
    cli.main(["near", *data, "--run-dir", str(run), "--ann-edges", "7", "25", "--n-iter", "3", "--n-init", "2",
              "--n-top", "1", "--n-valid", "1", "--use-contrast", "0.03", "--seed", "2", "--no-fm-curve",
              "--no-param-verify", "--no-display", "--forbidden-pa", "13", "20", "--forbidden-pa", "193", "20"])
    before = _snap(run)
    for name, lst in (("line.txt", LINE), ("planets.txt", PLANETS), ("verify.txt", [Source(*v) for v in VERIFY])):
        (tmp_path / name).write_text("".join(f"{s.rho} {s.theta} {s.contrast}\n" for s in lst))
    cli.main(["rereduce", *data, "--run-dir", str(run), "--subtract-sources", str(tmp_path / "line.txt"),
              "--inject-sources", str(tmp_path / "planets.txt"), "--verify-sources", str(tmp_path / "verify.txt")])
    out = str(run) + "_rereduce"
    rec = json.load(open(os.path.join(out, "rereduce.json")))
    assert "74 subtracted source(s)" in rec["frames"] and "2 added source(s)" in rec["frames"]
    assert all(os.path.exists(os.path.join(out, f)) for f in ("klip_stitched.fits", "klip_stitched_snr.fits",
                                                               "stitched_verify_report.txt"))
    with pytest.raises(SystemExit, match="already holds a re-reduction"):
        cli.main(["rereduce", *data, "--run-dir", str(run)])
    # no data option typed: the run's own frames, and --overwrite writes over the first one
    cli.main(["rereduce", *data, "--run-dir", str(run), "--overwrite", "--no-verify"])
    rec = json.load(open(os.path.join(out, "rereduce.json")))
    assert rec["frames"] == "the default frames" and rec["verify_report"] is None
    a, b = read_fits(str(run / "klip_stitched.fits")), read_fits(os.path.join(out, "klip_stitched.fits"))
    assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    assert _snap(run) == before
