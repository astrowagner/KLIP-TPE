"""Each frame's own true north (NEAR2 s6 note, 2026-10-31, section 2).

The astrometry of the 2026 cubes gives a true north per frame, in ``n<seq>/AB_truenorth_clean.txt``.
``load_night`` reads it, sliced and dropped as the angles are, and the reducer uses it at derotation
(each bin by its frames' mean) and in the injection azimuth, on the CPU and on the card.  The frame
selection, the binning and the reference selection read the parallactic angles alone.  A list of one
value is that angle, so it reduces as the constant does, to the bit.  A run records which north it
used, with each file's SHA-1, and a resume, a re-reduction or a check keeps it.  A run that recorded
none was made at 36.5 deg.
"""
from __future__ import annotations

import json
import os
import types
from dataclasses import replace

import numpy as np
import pytest

from klip_tpe import cli
from klip_tpe.injection import GaussianPSF, inject_frames, inject_sources
from klip_tpe.instruments import near
from klip_tpe.instruments.near import NEAR, NORTH_FILE, NorthFileError, read_north
from klip_tpe.klip import bin_values, derotate, rotate_ccw
from klip_tpe.metrics import Source
from klip_tpe.reducer import KLIPReducer, ReductionRequest
from klip_tpe.runner import Runner

from test_gpu_reducer import A_IN, A_OUT, SOURCES, TOL, annulus_noise, synthetic_night  # noqa: E402
from test_near_scale import root  # noqa: E402,F401  (the two NEAR nights on disk)

PARAMS = dict(bin=5, n_ang=2, filter=7, angsep=0.3, anglemax=120.0, corr_thresh=0.0, noise_max=3.0,
              coronoise_max=3.0, k_klip=6, inrad=A_IN, outrad=A_OUT, pad_in=True, pad_out=True)


def _write_north(path, values, header=True):
    with open(path, "w") as f:
        if header:
            f.write("# true north per frame (deg), ICRS\n\n")
        for k, v in enumerate(values):
            f.write(f"{float(v)!r}" + ("   # frame 0\n" if k == 0 else "\n"))


def _with_north(red: KLIPReducer, north, truenorth=NEAR.truenorth, cube=None) -> KLIPReducer:
    """The night of ``red`` with ``north`` per frame (None: none), and ``cube`` in place of its
    frames when given, as a new reducer."""
    ds = replace(red.data, north=None if north is None else np.asarray(north, float),
                 cube=red.data.cube if cube is None else cube)
    out = KLIPReducer(ds, pxscale=red.pxscale, lam_m=red.lam_m, diam_m=red.diam_m, injection_model=red.model,
                      fallback_model=red.fallback, truenorth=truenorth, fwhm_px=red.fwhm, zone_pad=red.zone_pad,
                      outrad_cap=red.outrad_cap, defaults=dict(red.defaults))
    out.name = red.name
    return out


# ----------------------------------------------------------------------------- the file
def test_the_file_is_read_with_comments_and_the_frame_plan_applies(root, tmp_path):  # noqa: F811
    nf = os.path.join(root, "n1", NORTH_FILE)
    vals = 36.0 + 0.01 * np.arange(12)
    _write_north(nf, vals)
    assert np.array_equal(read_north(nf, 12), vals), "comments and blank lines skipped, the values to the bit"
    ds = near.load_night(root, 1)
    assert np.array_equal(ds.north, vals) and ds.meta["north"]["source"] == "file"
    assert ds.meta["north"]["sha1"] and ds.meta["north"]["file"] == NORTH_FILE
    # the frame plan: --max-frames and --drop-frames, as for the angles
    sub = near.load_night(root, 1, frames=slice(0, 9), drop_frames=[2, 5])
    assert np.array_equal(sub.north, vals[[0, 1, 3, 4, 6, 7, 8]])
    assert near.load_night(root, 1, north="constant").north is None
    assert near.load_night(root, 2).north is None and near.load_night(root, 2).meta["north"]["why"] == f"no {NORTH_FILE}"
    # a count that is not the cube's, and a line that is not one number
    _write_north(nf, vals[:11])
    with pytest.raises(NorthFileError, match="11 values for the cube's 12 frames"):
        near.load_night(root, 1)
    with open(nf, "w") as f:
        f.write("36.5\n36.5 36.6\n")
    with pytest.raises(NorthFileError, match="line 2"):
        read_north(nf)


def test_a_run_keeps_the_file_it_was_made_with(root):  # noqa: F811
    nf = os.path.join(root, "n1", NORTH_FILE)
    vals = 36.3 + 0.02 * np.arange(12)
    _write_north(nf, vals)
    red = near.make_reducer(root, nights=(1, 2), use_library=False, max_workers=1, log=lambda s: None)
    rec = red.provenance["north"]
    assert rec["n1"]["source"] == "file" and rec["n1"]["min"] == vals.min() and rec["n1"]["max"] == vals.max()
    assert rec["n2"] == {"source": "constant", "why": f"no {NORTH_FILE}", "deg": NEAR.truenorth}
    assert np.array_equal(red.reducers["n1"].north_frames, vals) and red.reducers["n2"].north_frames is None
    json.dumps(red.provenance)
    # the record reproduces the run: the same file is read, the constant stays the constant
    again = near.make_reducer(root, nights=(1, 2), use_library=False, max_workers=1, log=lambda s: None, north=rec)
    assert np.array_equal(again.reducers["n1"].north_frames, vals) and again.reducers["n2"].north_frames is None
    _write_north(os.path.join(root, "n2", NORTH_FILE), np.full(12, 36.9))
    again = near.make_reducer(root, nights=(1, 2), use_library=False, max_workers=1, log=lambda s: None, north=rec)
    assert again.reducers["n2"].data.north is None, "a night recorded on the constant keeps it"
    # a recorded file that has changed or gone is refused
    _write_north(nf, vals + 0.1)
    with pytest.raises(NorthFileError, match="not the file the run was made with"):
        near.make_reducer(root, nights=(1, 2), use_library=False, max_workers=1, log=lambda s: None, north=rec)
    os.remove(nf)
    with pytest.raises(NorthFileError, match="which is not there"):
        near.make_reducer(root, nights=(1, 2), use_library=False, max_workers=1, log=lambda s: None, north=rec)


def _settle(tmp_path, run, extra=(), setup=None):
    d = tmp_path / run
    d.mkdir(exist_ok=True)
    if setup is not None:
        (d / "run_setup.json").write_text(json.dumps(setup))
    argv = ["resume", "--root", str(tmp_path), "--run-dir", str(d)] + list(extra)
    a = cli._build_parser().parse_args(argv)
    a._argv = argv
    cli._constants_for_run(a)
    return a.north


def test_the_cli_settles_the_north_per_run(tmp_path):
    assert _settle(tmp_path, "new") == "files", "a new run: the nights' north files"
    assert _settle(tmp_path, "old", setup={"reducer": {"pxscale": 0.0456}}) == "constant", \
        "a run that recorded no north was made at 36.5"
    rec = {"n1": {"source": "file", "sha1": "ab"}, "n2": {"source": "constant", "deg": 36.5}}
    s6 = {"reducer": {"provenance": {"instrument": "NEAR", "pxscale": 0.0445, "north": rec}}}
    assert _settle(tmp_path, "s6", setup=s6) == rec
    assert _settle(tmp_path, "s6", ["--north", "constant"], setup=s6) == "constant"
    assert _settle(tmp_path, "s6", ["--north", "files"], setup=s6) == "files"


# ----------------------------------------------------------------------------- the arithmetic
def test_the_injection_and_the_derotation_take_each_frames_north():
    rng = np.random.default_rng(4)
    n, ny = 7, 61
    angles = np.linspace(-40.0, 50.0, n)
    north = 36.5 + rng.normal(scale=0.4, size=n)
    model = GaussianPSF(6.4, star_flux=1.0e4)
    src = [Source(0.9, 70.0, 1e-3), Source(0.6, 200.0, 5e-4)]
    cube = rng.normal(size=(n, ny, ny)).astype(np.float32)
    per = cube.copy()
    inject_frames(per, per, angles, src, model, 0.0445, truenorth=north)
    for j in range(n):
        one = cube[j:j + 1].copy()
        inject_frames(one, one, angles[j:j + 1], src, model, 0.0445, truenorth=float(north[j]))
        assert np.array_equal(per[j], one[0]), f"frame {j}: its own north, to the bit"
    with pytest.raises(ValueError, match="6 true-north values for 7 frames"):
        inject_frames(per, per, angles, src, model, 0.0445, truenorth=north[:6])
    d = derotate(per, angles, north)
    for i in range(n):
        want = rotate_ccw(per[i], float(angles[i]) + float(north[i])).astype(np.float32)
        assert np.array_equal(d[i], want, equal_nan=True)
    assert np.array_equal(derotate(per, angles, 36.5), derotate(per, angles, np.full(n, 36.5)), equal_nan=True)
    grp = np.array([0, 0, 0, 1, 1, 2, 2])
    assert np.allclose(bin_values(north, grp, 3), [north[:3].mean(), north[3:5].mean(), north[5:].mean()])
    assert np.array_equal(bin_values(north, np.arange(n), 1), north)


@pytest.mark.parametrize("fused", [True, False])
def test_one_value_per_frame_reduces_as_the_constant_to_the_bit(fused):
    base = synthetic_night("north_const", 90, seed=5)
    for c in (NEAR.truenorth, 36.4871):
        a = _with_north(base, None, truenorth=c)
        b = _with_north(base, np.full(base.data.nframes, c))
        assert b.north_frames is None and b.truenorth == c
        p = dict(PARAMS, prep_fused=fused)
        for req in (ReductionRequest(p, SOURCES, False, "inj", threads=2, fm_sources=SOURCES),
                    ReductionRequest(p, SOURCES, False, "inj_x", threads=2, extras=True),
                    ReductionRequest(p, None, False, "clean_x", threads=2, extras=True)):
            ra, rb = a.reduce(req), b.reduce(req)
            for k in ("image", "fm_image", "cadi", "nosub"):
                va, vb = getattr(ra, k), getattr(rb, k)
                assert (va is None) == (vb is None)
                if va is not None:
                    assert np.array_equal(va, vb, equal_nan=True), f"{req.tag} {k} at {c}"


def _centroid_pa(img, rho_px, theta_deg, box=4.0):
    """The PA (deg) of the light's centroid within ``box`` px of where (rho, theta) should land."""
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    c = (nx - 1) / 2.0
    az = np.deg2rad(theta_deg + 90.0)
    x0, y0 = c + rho_px * np.cos(az), c + rho_px * np.sin(az)
    m = (np.hypot(xx - x0, yy - y0) <= box) & np.isfinite(img)
    w = np.nan_to_num(np.where(m, np.clip(img, 0, None), 0.0))
    xc, yc = (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()
    return float(np.degrees(np.arctan2(yc - c, xc - c)) - 90.0) % 360.0


def test_a_planet_put_in_with_each_frames_north_comes_out_at_its_pa():
    """A planet put into the frames with a north running from 34.5 to 36.5 deg lands at its PA when
    the reducer derotates with the same values.  Derotated at the constant 36.5 instead, it lands
    1 deg away, the mean of the difference.  The images without subtraction (frames with only the
    planet) and the KLIP images (a night's frames) both."""
    base = synthetic_night("north_drift", 120, seed=6)
    n = base.data.nframes
    north = np.linspace(34.5, 36.5, n)
    theta, rho = 30.0, 1.0
    planet = [Source(rho, theta, 4e-3)]
    rho_px = rho / base.pxscale
    p = dict(PARAMS, bin=3, filter=7, inrad=12.0, outrad=34.0)
    for kind, cube0 in (("nosub", np.zeros_like(base.data.cube)), ("image", base.data.cube)):
        cube = inject_sources(cube0, base.data.angles, planet, base.model, base.pxscale, truenorth=north)
        per = _with_north(base, north, cube=cube)
        const = _with_north(base, None, truenorth=NEAR.truenorth, cube=cube)
        req = ReductionRequest(p, None, False, kind, threads=2, extras=True)
        got_per = _centroid_pa(getattr(per.reduce(req), kind), rho_px, theta)
        got_const = _centroid_pa(getattr(const.reduce(req), kind), rho_px, theta)
        assert abs(got_per - theta) < 0.25, (kind, got_per)
        assert abs(got_const - (theta + 1.0)) < 0.3, (kind, got_const)
    # the binned north is the frames' mean in each bin, and only the frames' own north makes one
    pr = per.prepare(ReductionRequest(p, None, False, "prep", threads=1))
    assert np.allclose(pr.bnorth, bin_values(north[pr.keep], pr.grp, 3)[pr.bkeep])
    assert const.prepare(ReductionRequest(p, None, False, "prep", threads=1)).bnorth is None


def test_the_card_agrees_with_the_cpu_with_each_frames_north():
    from klip_tpe.gpu_reducer import GPUKLIPReducer
    base = synthetic_night("north_card", 160, seed=7)
    rng = np.random.default_rng(8)
    cpu = _with_north(base, 36.5 + np.cumsum(rng.normal(scale=0.03, size=base.data.nframes)))
    assert cpu.north_frames is not None
    g = GPUKLIPReducer.from_cpu(cpu)
    g.backend_name = "numpy"
    assert g.north_frames is cpu.north_frames or np.array_equal(g.north_frames, cpu.north_frames)
    for params in (dict(PARAMS), dict(PARAMS, bin=1, filter=0), dict(PARAMS, comb_type="median")):
        req = ReductionRequest(params, SOURCES, False, "card", threads=2)
        assert g.supports(req)
        rc, rg = cpu.reduce(req), g.reduce(req)
        assert rg.meta["device"] == "gpu"
        sig, m = annulus_noise(rc.image, A_IN, A_OUT)
        both = m & np.isfinite(rg.image)
        assert np.array_equal(np.isfinite(rg.image), np.isfinite(rc.image))
        assert float(np.abs(rg.image - rc.image)[both].max() / sig) < TOL
    # the card's own constant path is unchanged by the per-frame code
    g0 = GPUKLIPReducer.from_cpu(_with_north(base, None))
    g0.backend_name = "numpy"
    g1 = GPUKLIPReducer.from_cpu(_with_north(base, np.full(base.data.nframes, NEAR.truenorth)))
    g1.backend_name = "numpy"
    req = ReductionRequest(dict(PARAMS), SOURCES, False, "card0", threads=2)
    assert np.array_equal(g0.reduce(req).image, g1.reduce(req).image, equal_nan=True)


def test_a_backend_that_derotates_by_one_angle_refuses_frames_with_their_own():
    from klip_tpe.backends.custom import FunctionReducer

    class OneAngle(KLIPReducer):
        def _subtract(self, bcube, bang, kp, p, filt, req, mcube, ref_basis, fm_ref, meta):
            raise AssertionError("never reached")

    base = synthetic_night("north_backend", 60, seed=9)
    ds = replace(base.data, north=36.5 + 0.01 * np.arange(base.data.nframes))
    red = OneAngle(ds, pxscale=base.pxscale, lam_m=base.lam_m, diam_m=base.diam_m, injection_model=base.model,
                   truenorth=36.5, fwhm_px=base.fwhm)
    assert not red.supports_frame_north and base.supports_frame_north
    with pytest.raises(ValueError, match="one angle"):
        red.reduce(ReductionRequest(dict(PARAMS), None, False, "x"))
    assert FunctionReducer._subtract is not KLIPReducer._subtract, "the custom backend overrides it, so refuses"


def test_the_stitched_header_says_per_frame_with_each_nights_values():
    rec = {"n1": {"source": "file", "file": NORTH_FILE, "min": 36.41, "median": 36.5, "max": 36.59},
           "n2": {"source": "constant", "deg": 36.5, "why": f"no {NORTH_FILE}"}}
    stub = types.SimpleNamespace(reducer=types.SimpleNamespace(provenance={"instrument": "NEAR", "north": rec}))
    h = Runner._north_header(stub)
    assert h["TRUENORT"] == "per frame (ICRS)"
    assert h["TN_n1"] == f"36.500 (36.410-36.590) deg, {NORTH_FILE}" and h["TN_n2"] == "36.5 deg, every frame"
    rec2 = {"n1": {"source": "constant", "deg": 36.5}, "n2": {"source": "constant", "deg": 36.5}}
    stub.reducer.provenance = {"north": rec2}
    assert Runner._north_header(stub)["TRUENORT"] == 36.5
    stub.reducer.provenance = None
    assert Runner._north_header(stub) == {}
    from klip_tpe.stitch import fits_header
    hh = fits_header(dict(h))
    assert hh["TRUENORT"] == "per frame (ICRS)" and hh["TN_N1"].startswith("36.500")


def test_both_preparation_paths_agree_with_each_frames_north():
    """The fused preparation (one bin group at a time) and the whole-cube steps place every
    source at its frame's own north and give the same images, to the bit, as they do with one."""
    base = synthetic_night("north_paths", 90, seed=10)
    rng = np.random.default_rng(11)
    red = _with_north(base, 36.5 + rng.normal(scale=0.3, size=base.data.nframes))
    for extras in (False, True):
        out = []
        for fused in (True, False):
            req = ReductionRequest(dict(PARAMS, prep_fused=fused), SOURCES, False, f"paths{int(fused)}", threads=2,
                                   fm_sources=None if extras else SOURCES, extras=extras)
            out.append(red.reduce(req))
        for k in ("image", "fm_image", "cadi"):
            va, vb = getattr(out[0], k), getattr(out[1], k)
            assert (va is None) == (vb is None)
            if va is not None:
                assert np.array_equal(va, vb, equal_nan=True), (k, extras)


@pytest.mark.slow
def test_a_run_on_the_command_line_records_the_files_and_refuses_a_changed_one(tmp_path, capsys, monkeypatch):
    """``klip-tpe near`` with n1's north file and none for n2: the log, ``run_setup.json``'s record
    and the stitched header.  A resume keeps the run's north, a resume after the file changed is
    refused, and ``--north files`` takes the new file."""
    from astropy.io import fits

    from test_near_drop_frames import NF
    from test_near_drop_frames import root as drop_root
    r, _ = drop_root.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setenv("KLIP_TPE_DATA", str(tmp_path / "home"))
    vals = 36.2 + 0.04 * np.arange(NF)
    _write_north(os.path.join(r, "n1", NORTH_FILE), vals)
    run = tmp_path / "run"
    common = ["--root", r, "--nights", "1", "2", "--no-library", "--workers", "1", "--no-display"]
    cli.main(["near", *common, "--run-dir", str(run), "--ann-edges", "12", "30", "--n-iter", "2", "--n-init", "2",
              "--n-top", "1", "--n-valid", "1", "--use-contrast", "0.5", "--seed", "1", "--no-fm-curve",
              "--no-param-verify"])
    out = capsys.readouterr().out
    assert f"n1: true north per frame from {NORTH_FILE}, 36.200-36.800 deg (median 36.500)" in out
    assert f"n2: true north 36.5 deg for every frame (no {NORTH_FILE})" in out
    with open(run / "run_setup.json") as f:
        rec = json.load(f)["reducer"]["provenance"]["north"]
    assert rec["n1"]["source"] == "file" and rec["n2"]["source"] == "constant"
    h = fits.getheader(run / "klip_stitched.fits")
    assert h["TRUENORT"] == "per frame (ICRS)" and h["TN_N1"].startswith("36.500 (36.200-36.800) deg")
    assert h["TN_N2"] == "36.5 deg, every frame"
    cli.main(["resume", *common, "--run-dir", str(run)])
    assert "n1: true north per frame from" in capsys.readouterr().out
    _write_north(os.path.join(r, "n1", NORTH_FILE), vals + 0.01)
    with pytest.raises(SystemExit, match="not the file the run was made with"):
        cli.main(["resume", *common, "--run-dir", str(run)])
    capsys.readouterr()
    cli.main(["resume", *common, "--run-dir", str(run), "--north", "files"])
    assert "36.210-36.810 deg" in capsys.readouterr().out
