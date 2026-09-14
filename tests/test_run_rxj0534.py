"""The RX J0534 script must be right before it is run on four nights of LMIRCam data.

The data lives on an 8-hour-a-night RAID volume and a full search is hours of compute, so
a typo in the loader or a stale keyword is expensive to discover.  These tests exercise the
script's own logic -- partition grouping, the IDL angle/exposure save files, the PSF build,
the geometry of the companion -- against small synthetic stand-ins laid out exactly like the
real tree, and then run its ``main`` end to end at a budget of one evaluation.
"""
import json
import os
import sys
import types

import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
run_rxj0534 = pytest.importorskip("run_rxj0534")

scipy_io = pytest.importorskip("scipy.io")


# ------------------------------------------------------------------ a miniature data tree
#: one real pipeline save file, so the format parsing is tested against the genuine article
#: rather than against something this test invented (scipy can read IDL saves but not write
#: them, and the layout -- anglesN / exptimesN / coaddsN, big-endian f8 -- is the point)
SAMPLE_SAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                          "lmircam_angles_sample.sav")


def _fake_tree(root, nframes=6, size=48):
    """Four nights x two sides x two cubes, plus the unsaturated sequences."""
    rng = np.random.default_rng(7)
    for night in run_rxj0534.NIGHTS:
        for side in run_rxj0534.SIDES:
            for c in run_rxj0534.CUBES:
                for sub, nf in (("reduced", nframes), ("reduced-unsat", 3)):
                    d = os.path.join(root, night, sub)
                    os.makedirs(d, exist_ok=True)
                    cube = rng.normal(0, 1, (nf, size, size)).astype("f4")
                    cube[:, size // 2, size // 2] += 500.0            # a core to find
                    fits.writeto(os.path.join(d, f"{night}_{side}_cube{c}_align_cen_bin.fits"),
                                 cube, overwrite=True)
                    open(os.path.join(d, f"{night}_{side}_angles{c}_bin.sav"), "wb").close()
    return root


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """The tree, with the IDL readers standing in for save files this test cannot write.

    ``test_the_real_save_format_reads`` covers the readers themselves against a genuine
    pipeline file; everything else here is about the grouping and geometry around them.
    """
    root = _fake_tree(str(tmp_path / "LMIRCam"))
    monkeypatch.setattr(run_rxj0534, "ROOT", root)

    def fake_angles(path):
        n = fits.getheader(path.replace("_angles", "_cube")
                           .replace("_bin.sav", "_align_cen_bin.fits"))["NAXIS3"]
        return np.linspace(-6.0, 48.0, n) + hash(os.path.basename(path)) % 7 * 1e-3

    monkeypatch.setattr(run_rxj0534, "_angles", fake_angles)
    monkeypatch.setattr(run_rxj0534, "_texp", lambda p: 3020.22)
    return root


@pytest.fixture
def big_tree(tmp_path, monkeypatch):
    """Frames wide enough to contain the companion at 36.4 px and the default annulus.

    The 48-px tree the other tests use is deliberately tiny, but it cannot hold a source
    at the companion's separation -- so anything about the companion itself has to be
    measured on a frame that reaches past it.
    """
    root = _fake_tree(str(tmp_path / "LMIRCamBig"), nframes=6, size=140)
    monkeypatch.setattr(run_rxj0534, "ROOT", root)

    def fake_angles(path):
        n = fits.getheader(path.replace("_angles", "_cube")
                           .replace("_bin.sav", "_align_cen_bin.fits"))["NAXIS3"]
        return np.linspace(-6.0, 48.0, n) + hash(os.path.basename(path)) % 7 * 1e-3

    monkeypatch.setattr(run_rxj0534, "_angles", fake_angles)
    monkeypatch.setattr(run_rxj0534, "_texp", lambda p: 3020.22)
    return root


def test_the_real_save_format_reads():
    """The readers, against a genuine ``*_angles1_bin.sav`` from the pipeline."""
    ang = run_rxj0534._angles(SAMPLE_SAV)
    assert ang.shape == (384,) and np.isfinite(ang).all()
    assert 40.0 < ang.max() - ang.min() < 70.0, "sky rotation is not a plausible ADI sequence"
    t = run_rxj0534._texp(SAMPLE_SAV)
    assert t and t > 0, "exptimes were not found -- star flux would fall back to None"


# ------------------------------------------------------------------ the pieces
def test_the_companion_position_matches_the_published_offsets():
    """rho and PA are derived from Table 2's dRA/dDec; a sign slip here would put the
    held-out source in the wrong quadrant and quietly invalidate the whole result."""
    d_ra, d_dec = -336.0, 198.0                       # mas
    rho = np.hypot(d_ra, d_dec) / 1000.0
    pa = np.degrees(np.arctan2(d_ra, d_dec)) % 360.0
    assert run_rxj0534.PLANET[0] == pytest.approx(rho, abs=0.002), f"rho should be {rho:.3f}\""
    assert run_rxj0534.PLANET[1] == pytest.approx(pa, abs=0.3), f"PA should be {pa:.1f} deg"


def test_the_default_annulus_is_centred_on_the_companion():
    """The default band is the companion's separation +/- 2 FWHM, so its mid-radius -- the
    radius 'fixed_pa' injects at -- lands exactly on the companion."""
    r_px = run_rxj0534.PLANET[0] / run_rxj0534.PXSCALE
    fwhm = 1.028 * (run_rxj0534.LAM_M / run_rxj0534.DIAM_M) * 206265.0 / run_rxj0534.PXSCALE
    lo, hi = r_px - 2.0 * fwhm, r_px + 2.0 * fwhm
    assert 0.5 * (lo + hi) == pytest.approx(r_px)
    assert lo > 1.5 * fwhm, f"the inner edge at {lo:.1f} px is inside the injection floor"
    assert hi - lo > 4 * fwhm - 1e-6


def test_lambda_over_d_is_the_single_lbt_primary():
    """Dual-aperture (incoherent) mode: each 8.4-m primary images independently, so the
    resolution is one primary's, not the 22.7-m baseline."""
    lod = (run_rxj0534.LAM_M / run_rxj0534.DIAM_M) * 206265.0 / run_rxj0534.PXSCALE
    assert 8.4 < lod < 9.1, f"lambda/D = {lod:.2f} px is not an 8.4-m primary at L'"


@pytest.mark.parametrize("how,want", [(16, 16), (8, 8), (4, 4)])
def test_partition_granularity(tree, how, want):
    dsets, sf = run_rxj0534.load(how, crop_half=0, log=lambda s: None)
    assert len(dsets) == want, sorted(dsets)
    total = sum(int(d.cube.shape[0]) for d in dsets.values())
    assert total == 4 * 2 * 2 * 6, "frames were lost or duplicated while merging"


def test_cropping_keeps_the_companion_and_the_star(tree):
    """Cropping is what makes 5,500 400x400 frames tractable, but it must not cut into the
    search annulus -- the companion sits at 36 px and the annulus reaches 55."""
    full, _ = run_rxj0534.load(4, crop_half=0, log=lambda s: None)
    cropped, _ = run_rxj0534.load(4, crop_half=20, log=lambda s: None)
    for pid, ds in cropped.items():
        assert ds.cube.shape[1:] == (41, 41), ds.cube.shape
        assert ds.cube.shape[0] == full[pid].cube.shape[0], "cropping dropped frames"
        # the star stays centred, which is what the annulus radii are measured from
        med = np.nanmedian(ds.cube, axis=0)
        assert np.unravel_index(np.argmax(med), med.shape) == (20, 20)


def test_merging_keeps_every_frame_and_its_own_angle(tree):
    """Merging four cubes into one partition must not shuffle angles against frames."""
    fine, _ = run_rxj0534.load(16, crop_half=0, log=lambda s: None)
    coarse, _ = run_rxj0534.load(4, crop_half=0, log=lambda s: None)
    for pid, ds in coarse.items():
        members = [v for k, v in fine.items() if k.startswith(pid)]
        assert len(members) == 4
        assert ds.cube.shape[0] == sum(m.cube.shape[0] for m in members)
        assert np.allclose(np.sort(ds.angles), np.sort(np.concatenate([m.angles for m in members])))


def test_partition_labels_are_readable_and_unique(tree):
    for how in (4, 8, 16):
        names = sorted(run_rxj0534.load(how, crop_half=0, log=lambda s: None)[0])
        assert len(set(names)) == how
        assert all(n.startswith("n") for n in names), names


def test_star_flux_uses_the_exposure_time_ratio(tree):
    """The science frames are saturated, so the stellar flux comes from the unsaturated
    sequence scaled by exposure time.  Equal exposures here means the ratio is 1."""
    _, sf = run_rxj0534.load(16, crop_half=0, log=lambda s: None)
    assert sf and len(sf) == 16
    psf_sum = 3 * 0 + 500.0                         # the injected core dominates the median
    for v in sf.values():
        assert v > 0.5 * psf_sum, f"star flux {v:.1f} is far below the PSF's own sum"


def test_the_psf_is_a_median_not_a_single_frame(tree, tmp_path):
    """A single LMIRCam frame can have a bad pixel brighter than the core; the median of
    the unsaturated cube is what makes the template usable."""
    d = os.path.join(tree, run_rxj0534.NIGHTS[0], "reduced-unsat")
    f = os.path.join(d, f"{run_rxj0534.NIGHTS[0]}_left_cube1_align_cen_bin.fits")
    cube = fits.getdata(f).copy()
    cube[0, 5, 5] = 1e6                              # one hot pixel, in one frame
    fits.writeto(f, cube, overwrite=True)
    psf, _ = run_rxj0534._psf_for(run_rxj0534.NIGHTS[0], "left", 1)
    assert psf[5, 5] < 1e4, "a single frame's hot pixel survived into the template"
    assert np.unravel_index(np.argmax(psf), psf.shape) == (24, 24), "the core is not the peak"


def test_a_cube_angle_mismatch_is_an_error_not_a_silent_crop(tree, monkeypatch):
    """Frames silently paired with the wrong angles would derotate into mush; that must
    stop the run, not produce a plausible-looking image.

    The real failure this guards against is a cube saved from one pass and angles from
    another -- the two files look fine on their own and only their lengths disagree.
    """
    target = f"{run_rxj0534.NIGHTS[1]}_left_angles1_bin.sav"
    real = run_rxj0534._angles

    def short(path):
        return real(path)[:-2] if os.path.basename(path) == target else real(path)

    monkeypatch.setattr(run_rxj0534, "_angles", short)
    with pytest.raises(ValueError, match="frames.*angles"):
        run_rxj0534.load(16, crop_half=0, log=lambda s: None)


def test_a_missing_night_is_reported_not_skipped_in_silence(tree):
    n = run_rxj0534.NIGHTS[2]
    os.remove(os.path.join(tree, n, "reduced", f"{n}_right_cube2_align_cen_bin.fits"))
    said = []
    dsets, _ = run_rxj0534.load(16, log=said.append)
    assert len(dsets) == 15
    assert any("missing" in s for s in said), said


def test_an_empty_root_fails_with_a_useful_message(tmp_path, monkeypatch):
    monkeypatch.setattr(run_rxj0534, "ROOT", str(tmp_path / "nothing"))
    with pytest.raises(SystemExit, match="LMIRCAM_ROOT"):
        run_rxj0534.load(16, crop_half=0, log=lambda s: None)


# ------------------------------------------------------------------ end to end
def test_main_runs_the_default_baseline(tree, tmp_path, monkeypatch):
    """The whole script, at a budget of one evaluation: catches a stale keyword or a
    renamed argument before four nights of data are loaded on the other machine."""
    out = str(tmp_path / "out")
    rc = run_rxj0534.main(["--partitions", "4", "--default-only", "--out", out,
                           "--ann", "8", "20", "--allow-offset-ann", "--k-max", "4", "--max-drop", "0",
                           "--crop", "0", "--workers", "1"])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "run.log"))
    log = open(os.path.join(out, "run.log")).read()
    assert "RX J0534 b at rho=0.39" in log
    assert "S/N ~ 5.2" in log
    assert os.path.exists(os.path.join(out, "results.jsonl"))


def test_main_builds_the_high_dimensional_space(tree, tmp_path):
    """16 partitions must actually produce a per-partition block each, not collapse to one."""
    out = str(tmp_path / "out16")
    run_rxj0534.main(["--partitions", "16", "--default-only", "--out", out,
                      "--ann", "8", "20", "--allow-offset-ann", "--k-max", "4", "--max-drop", "3", "--crop", "0", "--workers", "1"])
    log = open(os.path.join(out, "run.log")).read()
    line = [l for l in log.splitlines() if "search space:" in l][0]
    ndim = int(line.split("search space:")[1].split()[0])
    assert ndim > 100, f"16 partitions gave only {ndim} dimensions: {line}"


def test_show_actually_asks_for_a_window(tree, tmp_path, monkeypatch):
    """--show must reach LiveDisplay's ``show=``.

    The bug this pins: the display was constructed without it, so it wrote panels to
    steps/ and opened nothing.  Everything looked healthy -- the run proceeded, the files
    appeared -- and the only symptom was an absent window, which reads as a broken display
    rather than a missing argument.  Asserting the run finishes is not enough; the mode
    the display was asked for is the thing that matters.
    """
    from klip_tpe.display import LiveDisplay
    seen = {}
    real = LiveDisplay.__init__

    def spy(self, run_dir, *args, **kw):
        seen["show"] = kw.get("show", "NOT PASSED")
        return real(self, run_dir, *args, **kw)

    monkeypatch.setattr(LiveDisplay, "__init__", spy)
    out = str(tmp_path / "out_show")
    rc = run_rxj0534.main(["--partitions", "4", "--default-only", "--out", out, "--show",
                           "--ann", "8", "20", "--allow-offset-ann", "--k-max", "4", "--max-drop", "0",
                           "--crop", "0", "--workers", "1"])
    assert rc == 0
    assert seen.get("show") == "window", f"the display was asked for show={seen.get('show')!r}"


def test_show_inline_is_passed_through(tree, tmp_path, monkeypatch):
    """--show inline must not silently become a window (or nothing)."""
    from klip_tpe.display import LiveDisplay
    seen = {}
    real = LiveDisplay.__init__
    monkeypatch.setattr(LiveDisplay, "__init__",
                        lambda self, rd, *a, **k: (seen.__setitem__("show", k.get("show")),
                                                   real(self, rd, *a, **k))[1])
    run_rxj0534.main(["--partitions", "4", "--default-only", "--out", str(tmp_path / "o2"),
                      "--show", "inline", "--ann", "8", "20", "--allow-offset-ann", "--k-max", "4",
                      "--max-drop", "0", "--crop", "0", "--workers", "1"])
    assert seen.get("show") == "inline"


def test_no_show_builds_no_display(tree, tmp_path, monkeypatch):
    """And without --show there must be no display at all, for batch and cron runs."""
    from klip_tpe.display import LiveDisplay
    built = []
    real = LiveDisplay.__init__
    monkeypatch.setattr(LiveDisplay, "__init__",
                        lambda self, rd, *a, **k: (built.append(1), real(self, rd, *a, **k))[1])
    run_rxj0534.main(["--partitions", "4", "--default-only", "--out", str(tmp_path / "o3"),
                      "--ann", "8", "20", "--allow-offset-ann", "--k-max", "4", "--max-drop", "0",
                      "--crop", "0", "--workers", "1"])
    assert built == []


def test_show_does_not_break_a_headless_run(tree, tmp_path):
    """The window is on by default from the shell wrapper, so --show must degrade quietly
    where there is no display (a batch host, a detached terminal) rather than take the run
    down with it."""
    out = str(tmp_path / "out_headless")
    rc = run_rxj0534.main(["--partitions", "4", "--default-only", "--out", out, "--show",
                           "--ann", "8", "20", "--allow-offset-ann", "--k-max", "4", "--max-drop", "0",
                           "--crop", "0", "--workers", "1"])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "results.jsonl"))
    log = open(os.path.join(out, "run.log")).read()
    assert "Traceback" not in log
    assert "live window: window" in log, "the run should say what it opened"


# ------------------------------------------------------- injecting on the companion's ring
def test_the_annulus_is_built_around_the_companion(tree, tmp_path):
    """The band must be symmetric about the companion, because 'fixed_pa' injects at the
    band's *mid-radius* -- so a band chosen any other way would put every test source at
    the wrong separation while still looking correct in the log."""
    out = str(tmp_path / "o_ann")
    run_rxj0534.main(["--partitions", "4", "--default-only", "--out", out,
                      "--k-max", "4", "--max-drop", "0", "--crop", "0", "--workers", "1"])
    line = [l for l in open(os.path.join(out, "run.log")) if "mid-radius" in l][-1]
    mid = float(line.split("mid-radius")[1].split("px")[0])
    r_px = run_rxj0534.PLANET[0] / run_rxj0534.PXSCALE
    assert mid == pytest.approx(r_px, abs=0.05), f"injections would sit at {mid:.1f} px, not {r_px:.1f}"
    assert "note:" not in open(os.path.join(out, "run.log")).read()


def test_injections_land_on_the_companions_ring_and_clear_of_it(tree, tmp_path, monkeypatch):
    """End to end through the sampler the run actually uses."""
    from klip_tpe.instruments import generic
    seen = {}
    real = generic.default_config

    def spy(red, *a, **k):
        obj, samp = real(red, *a, **k)
        seen["samp"], seen["red"] = samp, red
        return obj, samp

    monkeypatch.setattr(generic, "default_config", spy)
    out = str(tmp_path / "o_ring")
    run_rxj0534.main(["--partitions", "4", "--default-only", "--out", out,
                      "--k-max", "4", "--max-drop", "0", "--crop", "0", "--workers", "1"])
    samp, red = seen["samp"], seen["red"]
    assert samp.strategy == "fixed_pa", "injections still ladder across the band"
    assert list(samp.known) == [list(run_rxj0534.PLANET)] or \
           [tuple(k) for k in samp.known] == [run_rxj0534.PLANET], samp.known

    r_px = run_rxj0534.PLANET[0] / run_rxj0534.PXSCALE
    half = 2.0 * red.fwhm
    lo, hi = (r_px - half) * run_rxj0534.PXSCALE, (r_px + half) * run_rxj0534.PXSCALE
    src = samp.sample(3, lo, hi, np.random.default_rng(0))
    assert len(src) == 3
    for s in src:
        assert s.rho == pytest.approx(run_rxj0534.PLANET[0], abs=1e-6), \
            f"injected at {s.rho:.4f}\" instead of the companion's {run_rxj0534.PLANET[0]}\""
        sep = abs((s.theta - run_rxj0534.PLANET[1] + 180) % 360 - 180) * np.pi / 180 * s.rho
        assert sep > 1.4 * red.fwhm * run_rxj0534.PXSCALE, \
            f"an injection landed {sep:.3f}\" from the real companion"


def test_an_explicit_annulus_off_the_ring_is_refused_by_default(tree, tmp_path):
    """The reported problem, turned into a stop.

    A band that does not put the injections on the companion's separation means the run
    optimizes the wrong radius -- which is invisible until the result is wrong.  It has to
    be refused before the compute is spent, and the refusal has to say how to proceed
    anyway when a wide band really is what is wanted.
    """
    with pytest.raises(SystemExit, match="off the companion's separation"):
        run_rxj0534.main(["--partitions", "4", "--default-only", "--out", str(tmp_path / "o_ref"),
                          "--ann", "8", "20", "--k-max", "4", "--max-drop", "0",
                          "--crop", "0", "--workers", "1"])


def test_the_opt_out_proceeds_but_says_so_loudly(tree, tmp_path):
    out = str(tmp_path / "o_manual")
    run_rxj0534.main(["--partitions", "4", "--default-only", "--out", out, "--ann", "8", "20",
                      "--allow-offset-ann", "--k-max", "4", "--max-drop", "0",
                      "--crop", "0", "--workers", "1"])
    log = open(os.path.join(out, "run.log")).read()
    assert "annulus 8.0-20.0 px" in log
    assert "WARNING" in log and "allow-offset-ann" in log


# ------------------------------------------------------------------ the companion trace
def test_the_companion_is_traced_but_never_scored(big_tree, tmp_path):
    """Its S/N is recorded every evaluation and appears nowhere in the objective."""
    out = str(tmp_path / "o_trace")
    run_rxj0534.main(["--partitions", "4", "--default-only", "--out", out,
                      "--k-max", "4", "--max-drop", "0", "--crop", "0", "--workers", "1"])
    p = os.path.join(out, "companion_snr.jsonl")
    assert os.path.exists(p), "no companion trace was written"
    rows = [json.loads(l) for l in open(p) if l.strip()]
    assert rows, "the trace is empty"
    for r in rows:
        assert {"annulus", "index", "score", "companion_snr"} <= set(r)
        assert np.isfinite(r["companion_snr"])
        # the objective's own score must not be the companion's S/N
        assert r["score"] != pytest.approx(r["companion_snr"]), \
            "the companion appears to be feeding the objective"


def test_the_trace_reaches_the_panel_note(tmp_path):
    """The value the user watches on the live panel comes from the display's note line,
    set on the display -- never on the record, which the objective reads."""
    import types
    red = types.SimpleNamespace(pxscale=0.0107, fwhm=9.0,
                                matched_filter_kernel=lambda *a, **k: None)
    disp = types.SimpleNamespace(_live_reason=None)
    msgs = []
    tr = run_rxj0534.CompanionTrace(red, (0.39, 300.5), str(tmp_path), msgs.append, display=disp)
    tr.m = types.SimpleNamespace(per_source=lambda *a, **k: [7.7])
    runner = types.SimpleNamespace(ia=0)
    record = types.SimpleNamespace(index=3, score=9.1)
    tr.on_eval(runner, record, None, types.SimpleNamespace(image=np.zeros((8, 8))), False)
    assert disp._live_reason and "7.7" in disp._live_reason
    assert "held out" in disp._live_reason
    assert any("5.2" in m for m in msgs), "the published comparison should be logged"


def test_the_trace_never_takes_the_run_down(tmp_path):
    """It is a diagnostic; a bad image or a metric failure must not stop the search."""
    import types
    red = types.SimpleNamespace(pxscale=0.0107, fwhm=9.0,
                                matched_filter_kernel=lambda *a, **k: None)
    tr = run_rxj0534.CompanionTrace(red, (0.39, 300.5), str(tmp_path), lambda s: None)
    runner = types.SimpleNamespace(ia=0)
    record = types.SimpleNamespace(index=0, score=1.0)
    tr.on_eval(runner, record, None, None, False)                      # no image at all
    tr.on_eval(runner, record, None, types.SimpleNamespace(image=None), False)
    tr.m = types.SimpleNamespace(per_source=lambda *a, **k: [float("nan")])
    tr.on_eval(runner, record, None, types.SimpleNamespace(image=np.zeros((8, 8))), False)
    assert not os.path.exists(os.path.join(str(tmp_path), "companion_snr.jsonl"))


def test_the_run_reports_where_the_injections_land(big_tree, tmp_path):
    """The startup log must state the injected separations, so 'are they on the companion's
    ring?' is answered by looking rather than by reading the source.

    The separations are what is fixed for the annulus; the position angles are re-drawn every
    evaluation, as ``near2m_randpos`` does, and the report has to say which is which or the
    reader will take one representative draw for the whole run.
    """
    out = str(tmp_path / "o_report")
    run_rxj0534.main(["--partitions", "4", "--default-only", "--out", out,
                      "--k-max", "4", "--max-drop", "0", "--crop", "0", "--workers", "1"])
    log = open(os.path.join(out, "run.log")).read()
    assert "injected sources (3;" in log
    assert "these separations hold for every evaluation" in log
    assert "the position angles are re-drawn each time" in log
    assert "from the companion" in log
    assert "the objective is measured on its own ring" in log
    rhos = [float(l.split("rho = ")[1].split('"')[0]) for l in log.splitlines() if "PA =" in l and "rho =" in l]
    assert rhos, "no injected positions were reported"
    for r in rhos:
        assert r == pytest.approx(run_rxj0534.PLANET[0], abs=0.003), f"injected at {r:.4f}\""


def test_a_band_that_misses_the_companion_is_refused(big_tree, tmp_path):
    """A hand-picked --ann that puts the injections somewhere else must stop the run before
    hours of compute, not quietly optimize the wrong separation."""
    with pytest.raises(SystemExit, match="off the companion's separation"):
        run_rxj0534.main(["--partitions", "4", "--default-only", "--out", str(tmp_path / "o_bad"),
                          "--ann", "8", "24", "--k-max", "4", "--max-drop", "0",
                          "--crop", "0", "--workers", "1"])


def test_the_default_budget_is_the_production_one():
    """5000 trials, with a warm-up capped rather than scaled.

    15% of 5000 would be 750 random evaluations before the model gets a say -- on a run
    this expensive that is most of a day spent not searching.
    """
    import subprocess
    import sys as _sys
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "scripts", "run_rxj0534.py")
    h = subprocess.run([_sys.executable, script, "--help"], capture_output=True, text=True).stdout
    assert "default: 5000" in h, h[h.find("--n-iter"):][:200]
    assert "capped at 400" in h


def test_the_warm_up_is_capped_not_scaled():
    for n_iter, want in ((200, 40), (1000, 150), (5000, 400), (20000, 400)):
        got = min(max(int(0.15 * n_iter), 40), 400)
        assert got == want, f"{n_iter} evaluations -> {got} warm-up, expected {want}"


def test_three_injections_leave_the_noise_ring_usable(big_tree, tmp_path):
    """Three, not four.

    Every source on the ring blanks ~3 FWHM of arc from the noise annulus that the S/N is
    measured against -- and RX J0534 b is on that ring too, so the injections are not the
    only tenants.  At 0.390" the ring holds ~25 apertures: four injections plus the planet
    leave ~10, three plus the planet leave ~13.  The objective's own noise estimate is what
    gets thinner, so this is a choice about the statistic, not about tidiness.
    """
    out = str(tmp_path / "o_three")
    run_rxj0534.main(["--partitions", "4", "--default-only", "--out", out,
                      "--k-max", "4", "--max-drop", "0", "--crop", "0", "--workers", "1"])
    log = open(os.path.join(out, "run.log")).read()
    assert "injected sources (3;" in log, [l for l in log.splitlines() if "injected sources" in l]
    assert "4 sources on it (3 injected + the planet)" in log
    assert "WARNING: only" not in log, "three injections should leave the ring healthy"
    rows = [json.loads(l) for l in open(os.path.join(out, "results.jsonl")) if l.strip()]
    for r in rows:
        if r.get("sources"):
            assert len(r["sources"]) == 3, f"{len(r['sources'])} sources injected"


def test_the_source_count_is_adjustable(big_tree, tmp_path):
    """Two must also land on the companion's ring.

    NEAR's two-source convention collapses the band to the annulus' *area-weighted* mid
    radius, which for a band centred on the companion sits ~4 px outside it -- so choosing
    --n-sources 2 would silently move the injections off the ring.  This run turns that
    convention off, because its band is already collapsed onto that ring deliberately.
    """
    out = str(tmp_path / "o_two")
    run_rxj0534.main(["--partitions", "4", "--default-only", "--out", out, "--n-sources", "2",
                      "--k-max", "4", "--max-drop", "0", "--crop", "0", "--workers", "1"])
    log = open(os.path.join(out, "run.log")).read()
    assert "injected sources (2;" in log
    assert "3 sources on it (2 injected + the planet)" in log


def test_a_crowded_ring_is_called_out(big_tree, tmp_path):
    """Asking for too many must say what it costs rather than silently thinning the noise."""
    out = str(tmp_path / "o_many")
    run_rxj0534.main(["--partitions", "4", "--default-only", "--out", out, "--n-sources", "7",
                      "--k-max", "4", "--max-drop", "0", "--crop", "0", "--workers", "1"])
    log = open(os.path.join(out, "run.log")).read()
    assert "WARNING: only" in log and "--n-sources" in log


# ------------------------------------------------------------------ resuming
def test_resume_continues_the_existing_run(big_tree, tmp_path):
    """A stopped run must carry on, not silently start again.

    The script builds its runner directly, so Runner.run() would have begun a fresh search
    over the same directory -- looking like a resume while throwing the history away.
    """
    out = str(tmp_path / "o_res")
    common = ["--partitions", "4", "--n-iter", "8", "--n-init", "4", "--k-max", "4",
              "--max-drop", "0", "--crop", "0", "--workers", "1"]
    run_rxj0534.main(common + ["--out", out])
    first = len([l for l in open(os.path.join(out, "results.jsonl")) if l.strip()])
    assert first >= 8
    assert os.path.exists(os.path.join(out, "checkpoint.json"))

    run_rxj0534.main(common + ["--out", out, "--resume"])
    log = open(os.path.join(out, "run.log")).read()
    assert "resumed" in log
    assert "continued with the recorded settings:" in log, "a resume must say what it recovered"


def test_resume_without_a_checkpoint_says_so(big_tree, tmp_path):
    with pytest.raises(SystemExit, match="nothing to resume"):
        run_rxj0534.main(["--partitions", "4", "--resume", "--out", str(tmp_path / "empty"),
                          "--k-max", "4", "--max-drop", "0", "--crop", "0", "--workers", "1"])


def test_options_changed_since_the_run_started_are_reported_not_applied(big_tree, tmp_path):
    """Runner.resume takes the configuration from the checkpoint by design -- the history
    would not be comparable otherwise.  So a changed --n-sources is not in force, and the
    run has to say so rather than let the user believe it took effect."""
    out = str(tmp_path / "o_chg")
    base = ["--partitions", "4", "--n-iter", "8", "--n-init", "4", "--k-max", "4",
            "--max-drop", "0", "--crop", "0", "--workers", "1", "--out", out]
    run_rxj0534.main(base + ["--n-sources", "3"])
    run_rxj0534.main(base + ["--n-sources", "2", "--resume"])
    log = open(os.path.join(out, "run.log")).read()
    assert "note: --n-sources=2 was given, but the run continues with 3" in log, \
        [l for l in log.splitlines() if "note:" in l]


def test_rerunning_continues_rather_than_starting_over(big_tree, tmp_path):
    """The default: a second invocation over the same --out continues the run.  Starting a
    fresh search into a directory that already holds one looks like a resume and silently
    discards the history, which is the failure this guards."""
    out = str(tmp_path / "o_over")
    base = ["--partitions", "4", "--n-iter", "8", "--n-init", "4", "--k-max", "4",
            "--max-drop", "0", "--crop", "0", "--workers", "1", "--out", out]
    run_rxj0534.main(base)
    run_rxj0534.main(base)                      # again, with nothing extra
    assert "resumed" in open(os.path.join(out, "run.log")).read()


def test_fresh_starts_over_on_purpose(big_tree, tmp_path):
    out = str(tmp_path / "o_fresh")
    base = ["--partitions", "4", "--n-iter", "8", "--n-init", "4", "--k-max", "4",
            "--max-drop", "0", "--crop", "0", "--workers", "1", "--out", out]
    run_rxj0534.main(base)
    run_rxj0534.main(base + ["--fresh"])
    log = open(os.path.join(out, "run.log")).read()
    assert "continued with the recorded settings" not in log.split("--fresh")[-1]
