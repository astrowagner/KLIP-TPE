"""The NEAR data options: ``--cube-name``, ``--subtract-sources``, ``--forbidden-pa`` and
``--mask-line``, and the run's record of what its frames are (``RunConfig.data_edits``).

What must hold:

* ``near.make_reducer(cube_name=...)`` reads that cube in every night's directory.
* ``near.make_reducer(subtract_sources=...)`` gives the frames a reduction gets from a cube
  with the same sources subtracted in the full frame by the same injection code, to the bit.
* A new run records the reducer's ``data_edits`` in its configuration (so the checkpoint has
  them), a resume with the same frames goes on, and a resume with other frames is refused.
* On the command line a resume takes ``--cube-name`` / ``--subtract-sources`` from the
  checkpoint when they are not typed, and refuses typed values that differ.
* ``--forbidden-pa`` reaches NEAR's sampler and ``--mask-line`` its metric, as a strip about
  the line through the star on both sides.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pytest
from astropy.io import fits

from klip_tpe import CalibrationConfig, MawetPeakSNR, Objective, Param, PositionSampler, RunConfig, SearchSpace
from klip_tpe import ValidationConfig, cli
from klip_tpe.injection import GaussianPSF, TemplatePSF, inject_sources
from klip_tpe.instruments import near
from klip_tpe.metrics import Source, line_mask, source_xy, star_center
from klip_tpe.reducer import KLIPReducer
from klip_tpe.runner import Runner
from klip_tpe.synthetic import synthetic_klip_dataset

NF, FULL = 16, 152                     # frames, full-frame size (cropped to 150 at load)
LINE = [Source(0.30 + 0.12 * i, 13.0 if i % 2 else 193.0, 1.0 + 0.2 * i) for i in range(6)]   # bright: above float32 rounding


def _psf():
    yy, xx = np.mgrid[0:16, 0:16]
    return np.exp(-0.5 * ((xx - 7.5) ** 2 + (yy - 7.5) ** 2) / 2.6 ** 2)


def _model():
    psf = _psf()
    return TemplatePSF(psf, center=(7.5, 7.5), ee_radius_px=near.NEAR.lam_over_d_px,
                       star_flux=near.template_flux_unit(psf), throughput_fn=near.analytic_throughput)


@pytest.fixture()
def root(tmp_path, monkeypatch):
    """A one-night NEAR tree: the default cube, a second cube, the angles and the B template.
    IDL save files cannot be written here, so ``_readsav`` reads an .npz beside the name."""
    rng = np.random.default_rng(5)
    d = tmp_path / "n1"
    d.mkdir()
    cube = (100.0 + 5.0 * rng.standard_normal((NF, FULL, FULL))).astype(np.float32)
    fits.writeto(d / "AB_cube_skysub_cen_clean.fits", cube)
    fits.writeto(d / "AB_cube_other.fits", (cube + 1.0).astype(np.float32))
    angles = np.linspace(-30.0, 30.0, NF)
    np.savez(d / "AB_parang_clean.npz", angles=angles)
    fits.writeto(d / "PSF_ACenB.fits", _psf())

    def readsav(path):
        z = np.load(os.path.splitext(path)[0] + ".npz")
        return {k: z[k] for k in z.files}
    monkeypatch.setattr(near, "_readsav", readsav)
    return str(tmp_path), cube, angles


def test_cube_name_reads_that_cube_in_every_night(root):
    r, cube, _ = root
    red = near.make_reducer(r, [1], use_library=False, max_workers=1, log=lambda s: None,
                            cube_name="AB_cube_other.fits")
    got = red.reducers["n1"].data.cube
    assert np.array_equal(got, cube[:, 1:151, 1:151] + 1.0)
    assert red.data_edits == {"cube_name": "AB_cube_other.fits"}
    assert red.describe()["partitions"]["n1"]["data_edits"]["cube_name"] == "AB_cube_other.fits"


def test_subtracting_at_load_equals_an_abline_cube_made_in_the_full_frame(root):
    r, cube, angles = root
    neg = [Source(s.rho, s.theta, -s.contrast) for s in LINE]
    full = inject_sources(cube, angles, neg, _model(), near.NEAR.pxscale, truenorth=near.NEAR.truenorth)
    fits.writeto(os.path.join(r, "n1", "AB_cube_skysub_cen_clean_abline.fits"), full)
    a = near.make_reducer(r, [1], use_library=False, max_workers=1, log=lambda s: None,
                          cube_name="AB_cube_skysub_cen_clean_abline.fits")
    b = near.make_reducer(r, [1], use_library=False, max_workers=1, log=lambda s: None, subtract_sources=LINE)
    ca, cb = a.reducers["n1"].data.cube, b.reducers["n1"].data.cube
    assert np.array_equal(ca.view(np.uint32), cb.view(np.uint32))
    assert np.nanmax(np.abs(cb - cube[:, 1:151, 1:151])) > 0.1          # and it did take something out
    assert b.data_edits["subtracted_sources"] == [list(s.as_tuple()) for s in LINE]


# ---------------------------------------------------------------- the run's record
PX = 0.05
LAM = 3.89 * PX / 206265 * 8.2


def _red(edit=None):
    ds = synthetic_klip_dataset(nframes=20, size=64, pa_span=60, seed=3)
    red = KLIPReducer(ds, PX, LAM, 8.2, injection_model=GaussianPSF(4.0, star_flux=1e4), outrad_cap=30)
    if edit:
        red.subtract_sources(edit)
    return red


def _run(red, run_dir, n_iter=2):
    space = SearchSpace([Param("k_klip", 1, 8, "int", default=4)])
    obj = Objective(MawetPeakSNR(pxscale=PX, fwhm=red.fwhm), clean_subtract=True)
    samp = PositionSampler(fwhm_as=red.fwhm * PX)
    cfg = RunConfig(ann_edges=[8, 24], n_iter=n_iter, n_init=2, seed=1, n_sources=1, param_verify=False,
                    validation=ValidationConfig(n_top=1, n_valid=1), calibration=CalibrationConfig(forced=[2e-3]),
                    save_fits=False, save_eval_images=False, write_setup_files=False, fm_curve=False)
    return Runner(red, space, obj, samp, cfg, str(run_dir), log=lambda s: None), space, obj, samp


EDIT = [(0.6, 13.0, 1e-3), (0.6, 193.0, 8e-4)]


def test_a_new_run_records_its_frames_and_a_resume_must_reduce_the_same(tmp_path):
    from klip_tpe.runner import load_checkpoint
    r, space, obj, samp = _run(_red(EDIT), tmp_path / "run")
    r.run()
    st, _ = load_checkpoint(str(tmp_path / "run"))
    assert st["config"]["data_edits"]["subtracted_sources"] == [list(map(float, e)) for e in EDIT]
    # the same frames: the resume goes on
    Runner.resume(str(tmp_path / "run"), _red(EDIT), obj, samp, log=lambda s: None)
    # other frames: refused
    with pytest.raises(ValueError, match="cannot change the frames"):
        Runner.resume(str(tmp_path / "run"), _red(), obj, samp, log=lambda s: None)
    with pytest.raises(ValueError, match="cannot change the frames"):
        Runner.resume(str(tmp_path / "run"), _red(EDIT[:1]), obj, samp, log=lambda s: None)


def test_a_run_without_edits_resumes_as_before_and_refuses_edited_frames(tmp_path):
    r, space, obj, samp = _run(_red(), tmp_path / "run")
    r.run()
    Runner.resume(str(tmp_path / "run"), _red(), obj, samp, log=lambda s: None)
    with pytest.raises(ValueError, match="cannot change the frames"):
        Runner.resume(str(tmp_path / "run"), _red(EDIT), obj, samp, log=lambda s: None)
    # a configuration that asks for edits the reducer does not carry is refused up front
    r2, *_ = _run(_red(), tmp_path / "run2")
    r2.cfg.data_edits = {"subtracted_sources": [list(e) for e in EDIT]}
    with pytest.raises(ValueError, match="build the reducer with them"):
        r2.run()


def _ns(run_dir, argv):
    a = argparse.Namespace(run_dir=str(run_dir), cube_name=None, subtract_sources=None, subtract_line=None,
                           _argv=argv)
    return a


def test_the_command_line_takes_the_record_and_refuses_other_frames(tmp_path, capsys):
    r, *_ = _run(_red(EDIT), tmp_path / "run")
    r.cfg.data_edits = {}                   # the Runner records the reducer's
    r.reducer.data_edits = dict(r.reducer.data_edits, cube_name="AB_cube_x.fits", subtract_file="/x/chain.txt")
    r.run()
    base = ["resume", "--run-dir", str(tmp_path / "run")]
    a = _ns(tmp_path / "run", base)
    cli._data_edits_for_run(a)
    assert a.cube_name == "AB_cube_x.fits" and a._restored_meta["subtract_file"] == "/x/chain.txt"
    assert [s.as_tuple() for s in a._restored_sources] == [tuple(map(float, e)) for e in EDIT]
    src, meta = cli._subtract_request(None, a)       # the record wins over anything the registry restored
    assert [s.as_tuple() for s in src] == [tuple(map(float, e)) for e in EDIT] and meta == a._restored_meta
    out = capsys.readouterr().out
    assert "as run was started" in out and "the 2 source(s) run was started with" in out
    # typed and the same: fine; typed and different: refused
    f = tmp_path / "same.txt"
    f.write_text("".join(f"{r_} {p} {c}\n" for r_, p, c in EDIT))
    a = _ns(tmp_path / "run", base + ["--subtract-sources", str(f), "--cube-name", "AB_cube_x.fits"])
    a.subtract_sources, a.cube_name = str(f), "AB_cube_x.fits"
    cli._data_edits_for_run(a)
    g = tmp_path / "other.txt"
    g.write_text("0.6 13 2e-3\n")
    a = _ns(tmp_path / "run", base + ["--subtract-sources", str(g)])
    a.subtract_sources = str(g)
    with pytest.raises(SystemExit, match="cannot change the frames"):
        cli._data_edits_for_run(a)
    a = _ns(tmp_path / "run", base + ["--cube-name", "AB_cube_skysub_cen_clean.fits"])
    a.cube_name = "AB_cube_skysub_cen_clean.fits"
    with pytest.raises(SystemExit, match="cannot change the frames"):
        cli._data_edits_for_run(a)


def test_the_options_parse_and_resume_restores_them_from_the_record():
    a = cli._build_parser().parse_args(["near", "--root", "/r", "--cube-name", "c.fits", "--subtract-sources",
                                        "s.txt", "--mask-line", "13", "6", "--forbidden-pa", "13", "8"])
    assert (a.cube_name, a.subtract_sources, a.mask_line, a.forbidden_pa) == ("c.fits", "s.txt", [[13.0, 6.0]],
                                                                              [[13.0, 8.0]])
    for k in ("cube_name", "subtract_sources", "subtract_line", "mask_line"):
        assert k in cli._explicit_dests(["resume", "--cube-name", "c", "--subtract-sources", "s",
                                         "--subtract-line", "n", "--mask-line", "13", "6"])
    src = open(cli.__file__).read()
    assert '"cube_name", "subtract_sources", "subtract_line", "mask_line"' in src


# ---------------------------------------------------------------- injections and noise off the line
def test_line_mask_is_a_strip_through_the_star_on_both_sides():
    shape = (150, 150)
    m = line_mask(shape, [(13.0, 3.0)])
    cx, cy = star_center(shape)
    for rho_px in (10.0, 40.0, 70.0):
        for pa, inside in ((13.0, True), (193.0, True), (103.0, False), (283.0, False)):
            x, y = source_xy([rho_px], [pa], 1.0, cx, cy)
            assert m[int(round(y[0])), int(round(x[0]))] == inside, (rho_px, pa)
    # constant width: the masked pixels per radius do not grow with radius, unlike a wedge
    yy, xx = np.mgrid[0:150, 0:150]
    rr = np.hypot(xx - cx, yy - cy)
    n_in = [int(m[(rr >= r0) & (rr < r0 + 5)].sum()) for r0 in (20, 40, 60)]
    assert max(n_in) - min(n_in) <= 6
    assert line_mask(shape, [(13.0, 3.0)], r_range=(30, 60))[(rr < 29)].sum() == 0


def test_forbidden_pa_and_the_line_mask_reach_nears_sampler_and_metric():
    ds = synthetic_klip_dataset(nframes=10, size=64, pa_span=60, seed=3)
    red = KLIPReducer(ds, near.NEAR.pxscale, near.NEAR.lam_klip, near.NEAR.diam,
                      injection_model=GaussianPSF(6.2, star_flux=1e4))
    m = line_mask((64, 64), [(13.0, 4.0)])
    obj, samp = near.default_config(red, measured_kernel=False, forbidden_pa=[(13.0, 8.0), (193.0, 8.0)],
                                    pixel_mask=m)
    assert [tuple(b) for b in samp.forbidden_pa] == [(13.0, 8.0), (193.0, 8.0)]
    assert obj.metric.pixel_mask is not None and int(obj.metric.pixel_mask.sum()) == int(m.sum())
    obj0, samp0 = near.default_config(red, measured_kernel=False)
    assert list(samp0.forbidden_pa) == [] and obj0.metric.pixel_mask is None
    # the command line's helpers
    a = argparse.Namespace(forbidden_pa=[[13.0, 8.0]], mask_line=[[13.0, 4.0]])
    assert cli._forbidden(a) == [(13.0, 8.0)]
    pm = cli._line_mask(red, a, log=lambda s: None)
    assert pm.shape == (64, 64) and np.array_equal(pm, m)
    assert cli._line_mask(red, argparse.Namespace(mask_line=None)) is None


@pytest.mark.slow
def test_near_and_resume_on_the_command_line_keep_the_subtraction(root, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))       # keep the run record here
    r, cube, angles = root
    f = tmp_path / "chain.txt"
    f.write_text("# rho pa contrast\n" + "".join(f"{s.rho} {s.theta} {s.contrast}\n" for s in LINE))
    run = tmp_path / "run"
    common = ["--root", r, "--nights", "1", "--no-library", "--workers", "1", "--no-display"]
    cli.main(["near", *common, "--run-dir", str(run), "--ann-edges", "12", "30", "--n-iter", "2", "--n-init", "2",
              "--n-top", "1", "--n-valid", "1", "--use-contrast", "0.5", "--seed", "1", "--no-fm-curve",
              "--no-param-verify", "--no-setup-files", "--subtract-sources", str(f)])
    from klip_tpe.runner import load_checkpoint
    st, _ = load_checkpoint(str(run))
    rec = st["config"]["data_edits"]
    assert rec["subtracted_sources"] == [list(s.as_tuple()) for s in LINE]
    assert rec["subtract_file"] == str(f)
    out = capsys.readouterr().out
    assert "subtracted 6 source(s)" in out and "left the frames, on a sample of frames: n1 " in out
    # a resume that does not name the file reduces the same frames (the run is finished, so it
    # only reloads and re-checks), and one that names other sources is refused
    cli.main(["resume", *common, "--run-dir", str(run)])
    assert "the 6 source(s) run was started with" in capsys.readouterr().out
    g = tmp_path / "other.txt"
    g.write_text("0.5 13 1.0\n")
    with pytest.raises(SystemExit, match="cannot change the frames"):
        cli.main(["resume", *common, "--run-dir", str(run), "--subtract-sources", str(g)])


def test_an_all_negative_list_is_refused_since_it_would_double_the_line(tmp_path):
    f = tmp_path / "abline_chain.txt"
    f.write_text("0.95 13 -1.2e-8\n0.97 13 -1.3e-8\n")
    with pytest.raises(SystemExit, match="every contrast in .* is negative"):
        cli._sources_file(str(f))
    f.write_text("0.95 13 1.2e-8\n0.97 13 -1.3e-8\n")           # mixed signs: a fitted chain can be
    assert len(cli._sources_file(str(f))) == 2
