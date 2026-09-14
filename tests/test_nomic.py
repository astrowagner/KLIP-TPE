"""pyNOMIC adapter: reading the notebook's products, partitions per chop state, Airy
injection model, and a tiny end-to-end run through the Runner."""
import json
import os

import numpy as np
import pytest
from astropy.io import fits

from klip_tpe import RunConfig, Runner, ValidationConfig
from klip_tpe.injection import AiryPSF
from klip_tpe.instruments import nomic

QUIET = (lambda s: None)
NF, NX = 60, 200          # frames, frame size (star registered at the frame centre)


def _airy(xy, amp, sx, sy, off, p, x0, y0, e=0.11):
    """pyNOMIC helper_functions.airy_disk, verbatim (ravel=False)."""
    from scipy.special import j1
    x, y = xy
    rad = np.pi * np.sqrt((((x - x0) * np.cos(p) + (y - y0) * np.sin(p)) / sx) ** 2
                          + (((x - x0) * np.sin(p) - (y - y0) * np.cos(p)) / sy) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (2 * j1(rad) / rad - 2 * e * j1(e * rad) / rad) ** 2
    m[rad == 0] = (1 - e ** 2) ** 2
    return amp * m + off


@pytest.fixture(scope="module")
def pynomic_dir(tmp_path_factory):
    """A fake pyNOMIC working directory: masked/*.fits + the three .npz side files in
    the notebooks' positional order."""
    d = tmp_path_factory.mktemp("pynomic")
    os.makedirs(d / "masked")
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:NX, 0:NX]
    c = (NX - 1) / 2.0
    static = np.zeros((NX, NX))
    for _ in range(20):
        x0, y0 = rng.uniform(c - 40, c + 40, 2)
        static += rng.uniform(0.5, 1.0) * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / 4 ** 2)
    static *= 30.0 * np.exp(-np.hypot(xx - c, yy - c) / 30.0)
    angles = np.linspace(-25, 25, NF)
    chops = np.array(["CHOP_A" if i % 2 == 0 else "CHOP_B" for i in range(NF)])
    files = []
    for i in range(NF):
        img = static * (1 + 0.1 * np.sin(i / 5.0)) + rng.standard_normal((NX, NX))
        img[(np.hypot(xx - 30, yy - 30) < 12)] = np.nan        # a NaN-masked ghost, like mask_files
        f = d / "masked" / f"frame_{i:05d}.fits"
        fits.PrimaryHDU(img.astype(np.float32)).writeto(f)
        files.append(str(f))
    files = np.asarray(files)
    header_info = np.array([np.ones(NF), angles, 2460000.0 + np.arange(NF) / 1440.0])
    np.savez(d / "star_NOMIC_chop_correction.npz", files, chops, header_info)
    np.savez(d / "star_NOMIC_aligned.npz", files, np.zeros((NF, 2)), np.zeros((NF, 7)), np.array([NX, NX]), 0.0)
    reffits = np.tile([1000.0, 12.0, 11.0, 0.0, 0.3, c, c], (NF, 1))
    np.savez(d / "star_NOMIC_evaluated.npz", np.full(NF, 12.4), np.full(NF, 0.1), np.full(NF, 1000.0),
             np.abs(rng.normal(1.0, 0.1, NF)), rng.uniform(0.9, 1.0, NF), np.abs(rng.normal(5.0, 0.5, NF)),
             np.zeros((NF, 3)), reffits)
    return str(d)


def test_airy_matches_pynomic_model():
    """AiryPSF(c*amp) == pyNOMIC airy_disk(c*amp) after our unit-flux normalisation."""
    m = AiryPSF(sigmax=12.0, sigmay=11.0, p=0.3, amp=1000.0)
    st, (cx, cy), ok = m.stamp(0.5)
    n = st.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    ref = _airy((xx, yy), 1e-3 * 1000.0, 12.0, 11.0, 0.0, 0.3, cx, cy)
    assert np.allclose(1e-3 * m.flux_unit * st, ref, atol=1e-9)
    assert np.isclose(m.fwhm_px, 1.028 * 11.5)


def test_load_pynomic_partitions_and_tags(pynomic_dir):
    ds = nomic.load_pynomic(pynomic_dir, "star", frames="masked", crop_half=40, log=QUIET)
    assert set(ds) == {"starA", "starB"}
    a = ds["starA"]
    assert a.cube.shape == (NF // 2, 80, 80) and np.isfinite(a.cube).all()   # NaN ghost zeroed
    assert set(a.tags) == {"corrs", "noises", "coronoise"}
    assert np.isclose(np.median(a.tags["noises"]), 1.0) and np.isclose(np.median(a.tags["coronoise"]), 1.0)
    assert a.meta["airy"].shape == (7,) and np.isclose(a.meta["airy"][1], 12.0)
    assert np.allclose(a.angles, np.linspace(-25, 25, NF)[::2])
    # crop geometry: frame centre (99.5, 99.5) -> crop centre (39.5, 39.5), i.e. rows/cols 60:140
    raw = fits.getdata(os.path.join(pynomic_dir, "masked", "frame_00000.fits")).astype(np.float32)
    exp = raw[60:140, 60:140].copy()
    exp[~np.isfinite(exp)] = 0.0
    assert np.array_equal(a.cube[0], exp)
    # parang sign flip and frame rejection
    ds2 = nomic.load_pynomic(pynomic_dir, "star", crop_half=40, parang_sign=-1, log=QUIET,
                             use_frame_rejection=np.arange(NF) % 3 != 0)
    assert np.allclose(ds2["starA"].angles, -a.angles[np.arange(NF // 2) % 3 != 0])
    # pre-binning
    ds3 = nomic.load_pynomic(pynomic_dir, "star", crop_half=40, pre_bin=3, log=QUIET)
    assert ds3["starA"].cube.shape[0] == (NF // 2) // 3


def test_nomic_end_to_end_tiny_run(pynomic_dir, tmp_path):
    ds = nomic.load_pynomic(pynomic_dir, "star", crop_half=40, log=QUIET)
    red = nomic.make_reducer(ds, log=QUIET)
    assert red.partitions() == ["starA", "starB"]
    assert red.reducers["starA"].model.name == "airy" and red.reducers["starA"].model.per_frame
    space = nomic.make_space(red, k_klip_max=8)
    space.project = nomic.make_guard(red, n_min_ref=3, ref_frac=0.5, k_max=8)
    obj, samp = nomic.default_config(red)
    cfg = RunConfig(ann_edges=[14, 34], n_iter=4, n_init=2, seed=1, save_fits=True, fm_curve=False,
                    validation=ValidationConfig(n_top=1, n_valid=1), defaults={"k_klip": 3},
                    calibration=__import__("klip_tpe").CalibrationConfig(forced=[2e-2]))
    r = Runner(red, space, obj, samp, cfg, str(tmp_path / "run"), log=QUIET)
    res = r.run()
    assert len(res) == 1 and res[0].validated
    w = json.load(open(tmp_path / "run" / "annulus01" / "winner.json"))
    assert set(w["partitions"]) <= {"starA", "starB"}
    assert os.path.exists(tmp_path / "run" / "klip_stitched.fits")


def test_frame_psf_injects_each_frames_own_star():
    """FramePSF: the companion in frame j is a scaled copy of frame j's star (not a
    median or analytic template) and its flux is contrast x that frame's core flux."""
    from klip_tpe.injection import FramePSF, inject_sources
    from klip_tpe.metrics import Source
    n, size = 6, 120
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    cube = np.zeros((n, size, size), np.float32)
    widths = np.linspace(3.0, 6.0, n)                 # seeing changes frame to frame
    fluxes = np.linspace(1000.0, 400.0, n)            # transparency changes too
    for j in range(n):
        g = np.exp(-0.5 * ((xx - c) ** 2 + (yy - c) ** 2) / widths[j] ** 2)
        g[0:15, 0:15] = 0.5                            # far-field junk that the taper must exclude
        cube[j] = fluxes[j] * g / g[np.hypot(xx - c, yy - c) <= 20].sum()
    m = FramePSF(cube, r_ee=20.0, r_stamp=30.0)
    assert m.per_frame and m.n_bad == 0
    assert np.allclose(m.frame_flux, fluxes, rtol=1e-4) and np.isclose(m.flux_unit, np.median(fluxes), rtol=1e-4)
    inj = inject_sources(np.zeros_like(cube), np.zeros(n), [Source(1.0, 0.0, 1e-2)], m, pxscale=0.05,
                         truenorth=0.0, copy=False)
    for j in range(n):
        # companion at (c + 20 px along the +x... whichever) -- compare shape to the frame's own star
        comp = inj[j]
        iy, ix = np.unravel_index(np.argmax(comp), comp.shape)
        star = cube[j]
        # peak ratio equals contrast (same normalisation: the star's own peak x c)
        assert np.isclose(comp.max(), 1e-2 * star.max(), rtol=2e-2)
        # width matches THIS frame's width: second moment of the companion vs star core
        w = np.hypot(xx - ix, yy - iy) <= 12
        ws = np.hypot(xx - c, yy - c) <= 12
        mom_c = np.sqrt((comp[w] * ((xx[w] - ix) ** 2 + (yy[w] - iy) ** 2)).sum() / comp[w].sum())
        mom_s = np.sqrt((star[ws] * ((xx[ws] - c) ** 2 + (yy[ws] - c) ** 2)).sum() / star[ws].sum())
        assert np.isclose(mom_c, mom_s, rtol=0.05), (j, mom_c, mom_s)
        assert not np.isclose(mom_c, np.sqrt(2) * widths[0], rtol=0.05) or j == 0
    # the taper kept the far-field junk out of the template
    assert inj[0].max() > 0 and (inj[0] > 0.1 * inj[0].max()).sum() < 400


def test_nomic_default_model_is_per_frame_airy(pynomic_dir):
    """Default = pyNOMIC's inject_source: frame j uses reffits[j]; the frame option is opt-in."""
    ds = nomic.load_pynomic(pynomic_dir, "star", crop_half=60, log=QUIET)
    assert ds["starA"].meta["airy_frames"].shape == (NF // 2, 7)
    red = nomic.make_reducer(ds, log=QUIET)
    m = red.reducers["starA"].model
    assert m.name == "airy" and m.per_frame and m.n_bad == 0
    red2 = nomic.make_reducer(ds, psf="frame", log=QUIET)
    assert red2.reducers["starA"].model.name == "frame"
    # pre-binned data: per-frame fits no longer line up -> median fit, not per frame
    ds3 = nomic.load_pynomic(pynomic_dir, "star", crop_half=60, pre_bin=3, log=QUIET)
    assert not nomic.make_reducer(ds3, log=QUIET).reducers["starA"].model.per_frame


def test_per_frame_airy_matches_pynomic_inject_source():
    """Frame j gets airy_disk(c*amp_j, sigmax_j, sigmay_j, p_j) -- different per frame."""
    from klip_tpe.injection import inject_sources
    from klip_tpe.metrics import Source
    fp = np.array([[1000.0, 12.0, 11.0, 0.0, 0.3, 0, 0], [800.0, 14.0, 13.0, 0.0, -0.2, 0, 0]])
    m = AiryPSF(sigmax=13.0, sigmay=12.0, p=0.05, amp=900.0, frame_params=fp)
    size = 160
    cube = np.zeros((2, size, size), np.float32)
    out = inject_sources(cube, np.zeros(2), [Source(0.6, 90.0, 1e-3)], m, pxscale=0.0179, copy=False)
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    xs, ys = c - 0.6 / 0.0179, c                    # PA 90, no rotation: az = -180 deg
    for j in range(2):
        ref = _airy((xx, yy), 1e-3 * fp[j, 0], fp[j, 1], fp[j, 2], 0.0, fp[j, 4], xs, ys)
        w = np.hypot(xx - xs, yy - ys) < 25
        assert np.allclose(out[j][w], ref[w], atol=1e-2 * out[j].max()), j   # bilinear sub-px placement
    assert not np.allclose(out[0], out[1])


def test_image_groups_split_on_position_jumps():
    """The pyNOMIC image-group port splits where the star position jumps and nowhere else."""
    rng = np.random.default_rng(3)
    t = 2460000.0 + np.arange(1200) / 1440.0
    pos = 100.0 + rng.normal(0, 0.3, 1200)
    pos[400:800] += 25.0                       # one nodded segment
    lab = nomic.image_groups(t, pos, smooth=20)
    assert lab.min() == 0 and lab.max() == 2
    assert (lab[:380] == 0).all() and (lab[420:780] == 1).all() and (lab[820:] == 2).all()
    assert (nomic.image_groups(t, 100.0 + rng.normal(0, 0.3, 1200), smooth=20) == 0).all()


def test_groups_become_partitions(pynomic_dir):
    """groups='auto' / labels / split times -> one partition per (chop, group); nights and
    groups are then just partitions for the selection dims."""
    # a psf_subtraction side file with a position jump in the middle of the sequence
    maxima = np.tile([100.0, 100.0], (NF, 1))
    maxima[NF // 2:, 0] += 30.0
    np.savez(os.path.join(pynomic_dir, "star_NOMIC_psf_subtraction.npz"), np.zeros(NF), maxima, np.zeros(NF),
             np.zeros((NF, 7)), np.zeros((NF, 3)), np.zeros((NF, 3)))
    ds = nomic.load_pynomic(pynomic_dir, "star", crop_half=40, groups="auto", group_smooth=3, group_min_frames=5,
                            log=QUIET)
    assert set(ds) == {"starAg1", "starBg1", "starAg2", "starBg2"}, set(ds)
    assert ds["starAg1"].meta["group"] == 1 and ds["starAg1"].cube.shape[0] + ds["starAg2"].cube.shape[0] == NF // 2
    # explicit labels and split times
    lab = (np.arange(NF) >= NF // 4).astype(int) + (np.arange(NF) >= 3 * NF // 4)
    ds2 = nomic.load_pynomic(pynomic_dir, "star", crop_half=40, groups=lab, group_min_frames=5, log=QUIET)
    assert len(ds2) == 6
    t_split = 2460000.0 + (NF // 2) / 1440.0
    ds3 = nomic.load_pynomic(pynomic_dir, "star", crop_half=40, groups=[t_split], group_min_frames=5, log=QUIET)
    assert set(ds3) == set(ds)
    # two "nights" x groups -> 8 partitions, one block each, drop slots over all of them
    ds_n2 = nomic.load_pynomic(pynomic_dir, "star", crop_half=40, groups="auto", group_smooth=3, group_min_frames=5,
                               name="n2", log=QUIET)
    red = nomic.make_reducer({**ds, **ds_n2}, log=QUIET)
    assert len(red.partitions()) == 8
    space = nomic.make_space(red, k_klip_max=8, max_drop=3)
    assert [p.name for p in space.params if p.role == "selection"] == ["drop1", "drop2", "drop3"]
    assert space.partitions == red.partitions()
    assert sum(1 for p in space.params if p.name.startswith("bin_")) == 8
