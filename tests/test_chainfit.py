"""The chain fit (``klip_tpe.chainfit``, ``klip-tpe fit-chain``).

What must hold:

* On data holding a line of point sources with known node values, a finished run's winners
  forward-model each node's basis, and the noise-weighted fit in the strip recovers the values.
* The fitted chain, subtracted at load, takes the line out of the final reduction.
* The chain file, the line's PA and the candidate list are read as documented, and an
  unfinished run or a mixed file is refused.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from klip_tpe import CalibrationConfig, MawetPeakSNR, Objective, Param, PositionSampler, RunConfig, SearchSpace
from klip_tpe import ValidationConfig
from klip_tpe.chainfit import chain_pa, fit_chain, read_candidates, read_chain
from klip_tpe.injection import GaussianPSF, inject_sources, read_sources
from klip_tpe.metrics import Source, line_mask
from klip_tpe.reducer import Dataset, KLIPReducer, ReductionRequest
from klip_tpe.runner import Runner
from klip_tpe.synthetic import synthetic_klip_dataset

PX = 0.05
LAM = 3.89 * PX / 206265 * 8.2                       # FWHM about 4 px
RHOS = np.round(np.arange(0.45, 1.16, 0.1), 3)
BASIS = {"B": [Source(float(r), 13.0, 1e-2) for r in RHOS], "F": [Source(float(r), 193.0, 1e-2) for r in RHOS]}
TRUTH = {"B": 1.0, "F": 0.6}


def _reducer(cube, ds, model):
    return KLIPReducer(Dataset(cube, ds.angles, ds.tags, texp=ds.texp, name="line", meta=dict(ds.meta or {})),
                       PX, LAM, 8.2, injection_model=model, outrad_cap=30)


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """A finished one-annulus run on data with the line in it."""
    ds = synthetic_klip_dataset(nframes=40, size=64, pa_span=90, seed=3)
    model = GaussianPSF(4.0, star_flux=1e4)
    base = KLIPReducer(ds, PX, LAM, 8.2, injection_model=model, outrad_cap=30)
    line = [Source(s.rho, s.theta, s.contrast * TRUTH[n]) for n, src in BASIS.items() for s in src]
    cube = inject_sources(ds.cube, ds.angles, line, model, PX, truenorth=base.truenorth,
                          angle_convention=base.angle_convention)
    red = _reducer(cube, ds, model)
    space = SearchSpace([Param("k_klip", 2, 10, "int", default=5)])
    obj = Objective(MawetPeakSNR(pxscale=PX, fwhm=red.fwhm), clean_subtract=True)
    samp = PositionSampler(fwhm_as=red.fwhm * PX, forbidden_pa=[(13.0, 20.0), (193.0, 20.0)])
    cfg = RunConfig(ann_edges=[7, 25], n_iter=3, n_init=2, seed=2, n_sources=1, param_verify=False,
                    validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[3e-2]),
                    fm_curve=False, write_setup_files=False)
    run = tmp_path_factory.mktemp("chain") / "run"
    Runner(red, space, obj, samp, cfg, str(run), log=lambda s: None).run()
    return {"ds": ds, "model": model, "cube": cube, "run": str(run), "obj": obj, "samp": samp}


def _write_chain(path, nodes=BASIS):
    with open(path, "w") as f:
        f.write("# rho pa contrast node\n")
        for n, src in nodes.items():
            for s in src:
                f.write(f"{s.rho} {s.theta} {s.contrast} {n}\n")
    return str(path)


def test_the_fit_recovers_the_node_values_and_the_fitted_chain_removes_the_line(finished, tmp_path):
    f = finished
    red = _reducer(f["cube"], f["ds"], f["model"])
    r = Runner.resume(f["run"], red, f["obj"], f["samp"], log=lambda s: None)
    labels, nodes = read_chain(_write_chain(tmp_path / "chain.txt"))
    assert labels == ["B", "F"]
    rec = fit_chain(r, nodes, labels, halfwidth_px=4.0, out_dir=str(tmp_path / "fit"), log=lambda s: None)
    got = {n["label"]: n for n in rec["nodes"]}
    for n, v in TRUTH.items():
        assert got[n]["value"] == pytest.approx(v, rel=0.15), (n, got[n])
        assert got[n]["error"] > 0
    assert rec["line_pa_deg"] == pytest.approx(13.0, abs=1e-6)
    assert rec["pixels"] > 50 and np.isfinite(rec["chi2_per_dof"])
    for name in ("chainfit.json", "chain_fitted.txt", "chainfit_model.fits", "chainfit_residual.fits",
                 "chainfit_basis.fits", "chainfit_region.fits"):
        assert os.path.exists(tmp_path / "fit" / name), name
    # the fitted chain is the basis times the values, and subtracting it at load removes the line
    fitted = read_sources(str(tmp_path / "fit" / "chain_fitted.txt"))
    assert len(fitted) == sum(len(v) for v in BASIS.values())
    params = dict(k_klip=5, inrad=7, outrad=25, n_ang=1, angsep=0.0, anglemax=360.0, fast=True, bin=1, filter=0)
    raw = red.reduce(ReductionRequest(dict(params))).image
    clean = _reducer(f["cube"], f["ds"], f["model"])
    clean.subtract_sources(fitted)
    out = clean.reduce(ReductionRequest(dict(params))).image
    free = KLIPReducer(f["ds"], PX, LAM, 8.2, injection_model=f["model"], outrad_cap=30).reduce(
        ReductionRequest(dict(params))).image
    strip = line_mask(raw.shape, [(13.0, 3.0)]) & np.isfinite(raw)
    e_line = np.nansum((raw - free)[strip] ** 2)
    assert np.nansum((out - free)[strip] ** 2) < 0.1 * e_line


def test_a_mask_and_the_candidate_tiers_take_pixels_out(finished, tmp_path):
    f = finished
    r = Runner.resume(f["run"], _reducer(f["cube"], f["ds"], f["model"]), f["obj"], f["samp"], log=lambda s: None)
    labels, nodes = read_chain(_write_chain(tmp_path / "chain.txt"))
    a = fit_chain(r, nodes, labels, halfwidth_px=4.0, out_dir=str(tmp_path / "a"), log=lambda s: None)
    os.makedirs(os.path.join(f["run"], "cand"), exist_ok=True)
    with open(os.path.join(f["run"], "cand", "candidates.txt"), "w") as fh:
        fh.write("# header\n  rank    x    y   rho\"     PA    S/N Nnt  pn_md   STIM  sub1  sub2   PSF       FAP"
                 "   score  tier  flag\n"
                 "     1   40   40  0.700   13.0   6.00   1   1.00   1.00  1.00  1.00  0.90  1.00E-03    5.00   A   -\n")
    b = fit_chain(r, nodes, labels, halfwidth_px=4.0, masks=[(0.9, 193.0, None)], candidate_tiers="A",
                  out_dir=str(tmp_path / "b"), log=lambda s: None)
    assert b["pixels"] < a["pixels"]
    assert [m["why"] for m in b["masks"]] == ["mask", "candidate A"]
    os.remove(os.path.join(f["run"], "cand", "candidates.txt"))


def test_an_unfinished_run_is_refused(finished, tmp_path):
    f = finished
    r = Runner.resume(f["run"], _reducer(f["cube"], f["ds"], f["model"]), f["obj"], f["samp"], log=lambda s: None)
    r.results = []
    with pytest.raises(ValueError, match="has not finished"):
        fit_chain(r, BASIS, ["B", "F"], out_dir=str(tmp_path / "x"), log=lambda s: None)


def test_the_chain_file_the_line_pa_and_the_candidates_are_read(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("0.4 13 1e-6\n0.5, 193, 2e-6\n")
    labels, nodes = read_chain(str(p))
    assert labels == ["all"] and [s.as_tuple() for s in nodes["all"]] == [(0.4, 13.0, 1e-6), (0.5, 193.0, 2e-6)]
    p.write_text("0.4 13 1e-6 n1\n0.5 193 2e-6\n")
    with pytest.raises(ValueError, match="some lines have a node label"):
        read_chain(str(p))
    assert chain_pa({"a": [Source(0.4, 13.0), Source(0.5, 193.0), Source(0.6, 12.5)]}) == pytest.approx(12.83, abs=0.05)
    assert chain_pa({"a": [Source(0.4, 179.0), Source(0.5, 1.0)]}) % 180 == pytest.approx(0.0, abs=1e-6) or \
        chain_pa({"a": [Source(0.4, 179.0), Source(0.5, 1.0)]}) == pytest.approx(180.0, abs=1e-6)
    with pytest.raises(ValueError, match="do not lie on one line"):
        chain_pa({"a": [Source(0.4, 13.0), Source(0.5, 60.0)]})
    c = tmp_path / "candidates.txt"
    c.write_text("# x\n  rank    x    y   rho\"     PA    S/N Nnt  pn_md   STIM  sub1  sub2   PSF       FAP"
                 "   score  tier  flag\n"
                 "     1   40   40  0.700   13.0   6.00   1   1.00   1.00  1.00  1.00  0.90  1.00E-03    5.00   B   -\n")
    assert read_candidates(str(c)) == [{"rho": 0.7, "pa": 13.0, "snr": 6.0, "tier": "B"}]
    assert read_candidates(str(tmp_path / "none.txt")) == []


@pytest.mark.slow
def test_fit_chain_on_the_command_line(tmp_path, monkeypatch):
    """generic cubes: a run with the line in the data, then ``klip-tpe fit-chain`` on it."""
    from astropy.io import fits
    from klip_tpe import cli
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))       # keep the run record here
    ds = synthetic_klip_dataset(nframes=40, size=64, pa_span=90, seed=3)
    psf = np.exp(-0.5 * ((np.mgrid[0:21, 0:21] - 10.0) ** 2).sum(0) / (4.0 / 2.3548) ** 2)
    files = {"angles": tmp_path / "angles.fits", "psf": tmp_path / "psf.fits", "base": tmp_path / "base.fits"}
    fits.writeto(files["angles"], ds.angles)
    fits.writeto(files["psf"], psf)
    fits.writeto(files["base"], ds.cube)
    data = ["--instrument", "generic", "--angles", str(files["angles"]), "--psf", str(files["psf"]),
            "--pxscale", str(PX), "--lam", str(LAM), "--diam", "8.2", "--star-flux", "1e4", "--workers", "1",
            "--no-tags", "--names", "line"]
    # the line, put in with the reducer's own injection model
    a = cli._build_parser().parse_args(["generic", "--cube", str(files["base"]), *data[2:]])
    red, *_ = cli._generic_objects(a, log=lambda s: None)
    sub = next(iter(red.reducers.values()))
    line = [Source(s.rho, s.theta, s.contrast * TRUTH[n]) for n, src in BASIS.items() for s in src]
    cube = inject_sources(ds.cube, ds.angles, line, sub.model, PX, truenorth=sub.truenorth,
                          angle_convention=sub.angle_convention)
    fits.writeto(tmp_path / "line.fits", cube)
    data += ["--cube", str(tmp_path / "line.fits")]
    run = tmp_path / "run"
    cli.main(["near", *data, "--run-dir", str(run), "--ann-edges", "7", "25", "--n-iter", "3", "--n-init", "2",
              "--n-top", "1", "--n-valid", "1", "--use-contrast", "0.03", "--seed", "2", "--no-fm-curve",
              "--no-param-verify", "--no-display", "--forbidden-pa", "13", "20", "--forbidden-pa", "193", "20"])
    chain = _write_chain(tmp_path / "chain.txt")
    cli.main(["fit-chain", *data, "--run-dir", str(run), "--chain", chain, "--halfwidth", "4",
              "--rho-range", "0.3", "1.3", "--forbidden-pa", "13", "20", "--forbidden-pa", "193", "20"])
    rec = json.load(open(str(run) + "_chainfit/chainfit.json"))      # beside the run, which is left as it is
    got = {n["label"]: n["value"] for n in rec["nodes"]}
    assert got["B"] == pytest.approx(TRUTH["B"], rel=0.2) and got["F"] == pytest.approx(TRUTH["F"], rel=0.2)
    assert rec["rho_range_arcsec"] == [0.3, 1.3]
