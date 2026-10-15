"""A line through the star by its nodes (``klip_tpe.chainfit``: ``read_line_nodes``,
``psf_line_integral``, ``line_chain``, ``line_basis``, ``fit_per_night``; ``--subtract-line``).

What must hold:

* The node file reads as documented and refuses what it cannot mean.
* ``L = integral PSF(x, 0) dx / PSF(0, 0)``: sigma sqrt(2 pi) for a Gaussian, and along the
  reference (radial) axis of an anisotropic stamp.
* The chain is a source every ``step`` px from the first node to ``rho_max`` (the crop's edge by
  default), at ``a(s) step / L``, held at the last value, on both half-lines.  The node bases
  add up to that chain.
* Fitted on synthetic data holding such a line, the bases return the node values, and the
  fitted node file reads back.  One amplitude per night recovers a line twice as bright on one
  night as on the other.
* ``--subtract-line`` builds the chain from the first partition's injection model and records
  the nodes.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pytest

from klip_tpe import CalibrationConfig, MawetPeakSNR, Objective, Param, PositionSampler, RunConfig, SearchSpace
from klip_tpe import ValidationConfig, cli
from klip_tpe.chainfit import (fit_chain, fit_per_night, line_basis, line_chain, psf_line_integral,
                               read_line_nodes)
from klip_tpe.injection import GaussianPSF, LibraryPSF, inject_sources
from klip_tpe.metrics import Source
from klip_tpe.reducer import Dataset, KLIPReducer, PartitionedReducer
from klip_tpe.runner import Runner
from klip_tpe.synthetic import synthetic_klip_dataset

PX = 0.05
LAM = 3.89 * PX / 206265 * 8.2                       # FWHM about 4 px
NODES = "pa 13\nstep 0.5\n# |s| a1 a2\n0.40 4e-3 2e-3\n0.65 6e-3 3e-3\n0.90 5e-3 4e-3\n1.15 5e-3 3e-3\n"


def _spec(tmp_path, text=NODES):
    p = tmp_path / "nodes.txt"
    p.write_text(text)
    return read_line_nodes(str(p))


def test_the_node_file(tmp_path):
    s = _spec(tmp_path)
    assert (s["pa"], s["step"], s["rho_max"]) == (13.0, 0.5, None)
    assert s["nodes"] == [0.40, 0.65, 0.90, 1.15] and s["a1"][1] == 6e-3 and s["a2"][3] == 3e-3
    s = _spec(tmp_path, "pa 193\nrho_max 1.2\n0.5\n0.8\n")            # nodes alone, for a fit
    assert s["a1"] == [0.0, 0.0] and s["rho_max"] == 1.2 and s["step"] == 0.5
    for bad, msg in (("0.5 1 1\n", "no 'pa' line"), ("pa 13\n0.8 1 1\n0.5 1 1\n", "increasing"),
                     ("pa 13\n0.5 1\n", "a node is"), ("pa 13\nwidth 3\n", "neither a keyword")):
        with pytest.raises(ValueError, match=msg):
            _spec(tmp_path, bad)


def test_the_line_integral_of_the_psf():
    g = GaussianPSF(4.0, star_flux=1e4)
    assert psf_line_integral(g, 0.5) == pytest.approx(4.0 / 2.3548 * np.sqrt(2 * np.pi), rel=0.01)
    # an elongated stamp whose reference axis is x: the integral runs along x
    yy, xx = np.mgrid[0:41, 0:41]
    sx, sy = 3.0, 1.5
    st = np.exp(-0.5 * (((xx - 20) / sx) ** 2 + ((yy - 20) / sy) ** 2))
    lib = LibraryPSF(np.stack([st, st]), [0.1, 2.0], center=(20.0, 20.0), ee_radius_px=3.0, refpa_deg=0.0)
    assert psf_line_integral(lib, 0.5) == pytest.approx(sx * np.sqrt(2 * np.pi), rel=0.01)
    lib.refpa_deg = 90.0
    assert psf_line_integral(lib, 0.5) == pytest.approx(sy * np.sqrt(2 * np.pi), rel=0.02)


def test_the_chain_and_the_bases(tmp_path):
    s = _spec(tmp_path)
    g = GaussianPSF(4.0, star_flux=1e4)
    L = psf_line_integral(g, 0.5)
    chain = line_chain(s, g, PX, (64, 64))
    one = [c for c in chain if c.theta == 13.0]
    two = [c for c in chain if c.theta == 193.0]
    assert len(one) == len(two)
    rho = np.array([c.rho for c in one])
    assert rho[0] == pytest.approx(0.40) and np.allclose(np.diff(rho), 0.5 * PX)
    assert rho[-1] <= 31.5 * PX + 1e-9 and rho[-1] > 31.5 * PX - 0.5 * PX       # the crop's inscribed edge
    a = np.array([c.contrast for c in one]) * L / 0.5
    assert a[0] == pytest.approx(4e-3) and a[-1] == pytest.approx(5e-3)            # held at the last node
    j = int(np.argmin(np.abs(rho - 0.525)))                                         # half way 0.40 -> 0.65
    assert a[j] == pytest.approx(np.interp(rho[j], s["nodes"], s["a1"]))
    # the bases add up to the chain
    labels, basis = line_basis(s, g, PX, (64, 64))
    assert labels == [f"a1_{v:g}" for v in s["nodes"]] + [f"a2_{v:g}" for v in s["nodes"]]
    total = {}
    for lab in labels:
        side, node = lab.split("_")
        val = s[side][s["nodes"].index(float(node))]
        for src in basis[lab]:
            total[(src.rho, src.theta)] = total.get((src.rho, src.theta), 0.0) + val * src.contrast
    for c in chain:
        assert total[(c.rho, c.theta)] == pytest.approx(c.contrast, rel=1e-12)


def _run(reducers, run_dir, parts):
    red = PartitionedReducer(reducers, max_workers=1, log=lambda s: None) if parts else reducers
    space = (SearchSpace().replicate([Param("k_klip", 2, 10, "int", default=5)], parts) if parts
             else SearchSpace([Param("k_klip", 2, 10, "int", default=5)]))
    obj = Objective(MawetPeakSNR(pxscale=PX, fwhm=4.0), clean_subtract=True)
    samp = PositionSampler(fwhm_as=4.0 * PX, forbidden_pa=[(13.0, 20.0), (193.0, 20.0)])
    cfg = RunConfig(ann_edges=[7, 25], n_iter=3, n_init=2, seed=2, n_sources=1, param_verify=False,
                    validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[3e-2]),
                    fm_curve=False, write_setup_files=False)
    Runner(red, space, obj, samp, cfg, str(run_dir), log=lambda s: None).run()
    return red, obj, samp


def _with_line(ds, chain, scale=1.0):
    model = GaussianPSF(4.0, star_flux=1e4)
    base = KLIPReducer(ds, PX, LAM, 8.2, injection_model=model, outrad_cap=30)
    cube = inject_sources(ds.cube, ds.angles, [Source(c.rho, c.theta, c.contrast * scale) for c in chain], model, PX,
                          truenorth=base.truenorth, angle_convention=base.angle_convention)
    return KLIPReducer(Dataset(cube, ds.angles, ds.tags, texp=ds.texp, name=ds.name, meta=dict(ds.meta or {})),
                       PX, LAM, 8.2, injection_model=model, outrad_cap=30)


def test_the_node_fit_returns_the_values_and_a_node_file(tmp_path):
    s = _spec(tmp_path)
    g = GaussianPSF(4.0, star_flux=1e4)
    chain = line_chain(s, g, PX, (64, 64))
    ds = synthetic_klip_dataset(nframes=40, size=64, pa_span=90, seed=3)
    red, obj, samp = _run(_with_line(ds, chain), tmp_path / "run", None)
    r = Runner.resume(str(tmp_path / "run"), red, obj, samp, log=lambda m: None)
    labels, basis = line_basis(s, red.model, PX, (64, 64))
    rec = fit_chain(r, basis, labels, line_pa=13.0, halfwidth_px=4.0, line_spec=s, out_dir=str(tmp_path / "fit"),
                    log=lambda m: None)
    got = read_line_nodes(str(tmp_path / "fit" / "line_fitted.txt"))
    assert got["nodes"] == s["nodes"] and got["pa"] == 13.0
    for side, es in (("a1", "e1"), ("a2", "e2")):
        for j in range(4):
            tol = max(3.0 * rec["line"][es][j], 0.25 * s[side][j])
            assert abs(got[side][j] - s[side][j]) < tol, (side, j, got[side], s[side], rec["line"][es])
    assert rec["line"]["a1"] == got["a1"]
    # a radial range keeps the fit inside it, and is recorded
    rec2 = fit_chain(r, basis, labels, line_pa=13.0, halfwidth_px=4.0, rho_range=(0.6, 1.1),
                     out_dir=str(tmp_path / "fit2"), log=lambda m: None)
    from klip_tpe.stitch import read_fits
    region = read_fits(str(tmp_path / "fit2" / "chainfit_region.fits")).astype(bool)
    yy, xx = np.mgrid[0:64, 0:64]
    rr = np.hypot(xx - 31.5, yy - 31.5) * PX
    assert region.any() and rr[region].min() >= 0.6 - 1e-9 and rr[region].max() <= 1.1 + 1e-9
    assert rec2["pixels"] < rec["pixels"] and rec2["rho_range_arcsec"] == [0.6, 1.1]
    with pytest.raises(ValueError, match="outside the run's annuli"):
        fit_chain(r, basis, labels, line_pa=13.0, rho_range=(1.4, 2.0), out_dir=str(tmp_path / "fit3"),
                  log=lambda m: None)
    # a node with nothing in the fit region is refused, not reported as 0 +- 0
    off = {"in": [Source(0.8, 13.0, 4e-3)], "off": [Source(2.6, 13.0, 4e-3)]}       # 52 px out: off the frame
    with pytest.raises(ValueError, match=r"node\(s\) off put no signal"):
        fit_chain(r, off, line_pa=13.0, halfwidth_px=4.0, out_dir=str(tmp_path / "fit4"), log=lambda m: None)


def test_one_amplitude_per_night_follows_the_nights(tmp_path):
    s = _spec(tmp_path)
    g = GaussianPSF(4.0, star_flux=1e4)
    chain = line_chain(s, g, PX, (64, 64))
    d1 = synthetic_klip_dataset(nframes=40, size=64, pa_span=90, seed=3)
    d2 = synthetic_klip_dataset(nframes=40, size=64, pa_span=80, seed=4)
    d1.name, d2.name = "n1", "n2"
    reds = {"n1": _with_line(d1, chain, 1.0), "n2": _with_line(d2, chain, 0.5)}
    red, obj, samp = _run(reds, tmp_path / "run", ["n1", "n2"])
    r = Runner.resume(str(tmp_path / "run"), red, obj, samp, log=lambda m: None)
    rec = fit_per_night(r, chain, line_pa=13.0, halfwidth_px=4.0, out_dir=str(tmp_path / "pn"), rho_range=(0.3, 1.3),
                        log=lambda m: None)
    amp = {row["partition"]: row["amplitude"] for row in rec["nights"]}
    assert amp["n1"] == pytest.approx(1.0, rel=0.2) and amp["n2"] == pytest.approx(0.5, rel=0.25)
    assert os.path.exists(tmp_path / "pn" / "per_night.json") and rec["rho_range_arcsec"] == [0.3, 1.3]


def test_subtract_line_builds_the_chain_and_records_the_nodes(tmp_path):
    p = tmp_path / "nodes.txt"
    p.write_text(NODES)
    red = KLIPReducer(synthetic_klip_dataset(nframes=10, size=64, pa_span=40, seed=3), PX, LAM, 8.2,
                      injection_model=GaussianPSF(4.0, star_flux=1e4), outrad_cap=30)
    a = argparse.Namespace(subtract_sources=None, subtract_line=str(p))
    src, meta = cli._subtract_request(red, a)
    assert [c.as_tuple() for c in src] == [c.as_tuple() for c in line_chain(read_line_nodes(str(p)), red.model, PX,
                                                                             (64, 64))]
    assert meta["subtract_line"]["nodes"] == [0.40, 0.65, 0.90, 1.15] and meta["subtract_line"]["file"] == str(p)
    cli._subtract(red, a, log=lambda m: None)
    assert red.data_edits["subtract_line"]["pa"] == 13.0
    assert len(red.data_edits["subtracted_sources"]) == len(src)
