"""Static hot pixels the DQ array misses, the record of what a load did, and one exposure once.

* :func:`static_hot_pixels` finds, on the blank-sky median of the background pointings, a
  pixel that stands out from BOTH its row and its column neighbours -- a 4QPM glow stick
  stands out along one axis only and must not be flagged, and neither may the
  unilluminated border.
* ``load_calints(hot_pixels=...)`` repairs them in every frame, records which, and says so
  in the provenance the reducer's describe() puts into ``run_setup.json``.
* A directory tree with two processings of the same exposures is refused, not stacked.
* A rebuild of a finished run repairs hot pixels only if the run did (or, for a run from
  before the repair, only on frames that are not its own).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from klip_tpe.backends.spaceklip import (frames_provenance, frames_signature,  # noqa: E402
                                         load_calints, static_hot_pixels)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ----------------------------------------------------------------------------- the finder
def _sky(ny=60, nx=60, level=20.0, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return level + rng.normal(0.0, noise, (ny, nx))


def test_a_hot_and_a_dead_pixel_are_found():
    sky = _sky()
    sky[12, 10] += 50.0
    sky[45, 40] -= 50.0
    hot = static_hot_pixels(sky)
    assert hot[12, 10] and hot[45, 40]
    assert int(hot.sum()) == 2, np.argwhere(hot)


def test_a_glow_stick_is_level_with_its_neighbours_along_the_stick():
    """The AND: a pixel on a stick stands out from its column but not from its row.  (Where
    two sticks cross -- on a 4QPM, under the star, inside the inner working angle -- the
    crossing stands out both ways and is flagged, which costs nothing there.)"""
    for axis in (0, 1):
        sky = _sky()
        if axis == 0:
            sky[30, :] += 8.0                         # a stick along a row
        else:
            sky[:, 20] += 8.0                         # a stick along a column
        assert not static_hot_pixels(sky).any(), axis


def test_the_unilluminated_border_is_never_flagged():
    sky = _sky()
    sky[:, :4] = np.nan                               # the border
    sky[20, 4] += 50.0                                # a spike beside it: 3 finite row neighbours
    sky[40, 30] += 50.0                               # and one well inside
    hot = static_hot_pixels(sky)
    assert not hot[20, 4]
    assert hot[40, 30]


def test_a_small_excess_on_a_quiet_sky_is_below_the_floor():
    sky = _sky(noise=0.01)
    sky[25, 25] += 3.0                                # 300 sigma, but under min_excess = 5
    assert not static_hot_pixels(sky).any()
    assert static_hot_pixels(sky, min_excess=2.0)[25, 25]


# ----------------------------------------------------------------------------- the loader
PX = 0.11
HOT = (45, 55)                                        # (y, x) on the detector


def _write(path, target, roll, level=0.0, bkgsub=False, hot=0.0, starflux=800.0, n_int=2,
           seed=0, cal_ver="1.2.3", ny=100, nx=100):
    """A MIRI-like stage-2 product, star at (50, 50), a hot pixel of ``hot`` at HOT."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = starflux / (1.0 + (np.hypot(xx - 50.0, yy - 50.0) / 4.0) ** 2) + float(level)
    img[HOT] += float(hot)
    sci = np.array([img + rng.normal(0.0, 0.3, (ny, nx)) for _ in range(n_int)], np.float32)
    dq = np.zeros_like(sci, np.int32)
    ph = fits.Header({"TARGPROP": target, "INSTRUME": "MIRI", "FILTER": "F1140C",
                      "CAL_VER": cal_ver, "CRDS_CTX": "jwst_0000.pmap"})
    if bkgsub:
        ph["S_BKDSUB"] = "COMPLETE"
    sh = fits.Header({"ROLL_REF": float(roll), "V3I_YANG": 0.0, "VPARITY": 1, "PIXAR_A2": PX ** 2,
                      "PIXAR_SR": 2.8e-13, "CRPIX1": 51.0, "CRPIX2": 51.0, "BUNIT": "MJy/sr",
                      "FILTER": "F1140C", "INSTRUME": "MIRI", "EFFINTTM": 20.0})
    fits.HDUList([fits.PrimaryHDU(header=ph), fits.ImageHDU(sci, header=sh, name="SCI"),
                  fits.ImageHDU(dq, name="DQ")]).writeto(path, overwrite=True)
    return str(path)


def _set(root, with_bkg=True):
    """ERS 1386's arrangement: science background-subtracted by the pipeline with 12 of the
    hot pixel left in, the reference not subtracted (sky + 80), the blank sky + 52."""
    os.makedirs(root, exist_ok=True)
    f = [_write(os.path.join(root, "jw01_calints.fits"), "T", 0.0, bkgsub=True, hot=12.0, seed=1),
         _write(os.path.join(root, "jw02_calints.fits"), "T", 9.4, bkgsub=True, hot=12.0, seed=2),
         _write(os.path.join(root, "jw03_calints.fits"), "REFSTAR", 0.0, level=20.0, hot=80.0, seed=3)]
    if with_bkg:
        f.append(_write(os.path.join(root, "jw04_calints.fits"), "T-BACKGROUND", 0.0, level=20.0,
                        hot=52.0, starflux=0.0, seed=4))
    return f


def _load(files, **kw):
    msgs = []
    ds, info = load_calints(files, science_target="T", half_px=20, align=False, partition="all",
                            destripe=False, log=msgs.append, **kw)
    return ds, info, msgs


def _spike(cube, y=HOT[0] - 30, x=HOT[1] - 30):
    """The hot pixel's excess over its 8 neighbours, in the 41x41 crop (star at 50 -> 20)."""
    c = np.nanmedian(np.asarray(cube, float), axis=0)
    ring = np.concatenate([c[y - 1, x - 1:x + 2], c[y + 1, x - 1:x + 2], c[y, [x - 1, x + 1]]])
    return float(c[y, x] - np.median(ring))


def test_the_loader_repairs_them_in_every_frame_and_records_which(tmp_path):
    ds, info, msgs = _load(_set(str(tmp_path)))
    assert info["hot_pixels"] == [(HOT[1], HOT[0])]
    assert abs(_spike(ds["sci"].cube)) < 1.0 and abs(_spike(ds["sci"].ref_cube)) < 1.0
    rec = info["provenance"]
    assert rec["hot_pixels"]["applied"] is True and rec["hot_pixels"]["n"] == 1
    assert rec["hot_pixels"]["pixels"] == [[HOT[1], HOT[0]]]
    assert rec["cal_ver"] == ["1.2.3"] and rec["crds_ctx"] == ["jwst_0000.pmap"]
    assert rec["files"]["bkg"] == ["jw04_calints.fits"] and len(rec["files"]["sci"]) == 2
    assert rec["pxscale"] == pytest.approx(PX)
    assert ds["sci"].meta["provenance"] is rec or ds["sci"].meta["provenance"] == rec
    assert any("static hot/dead pixel" in m for m in msgs)


def test_hot_pixels_false_leaves_them(tmp_path):
    ds, info, _ = _load(_set(str(tmp_path)), hot_pixels=False)
    assert info["hot_pixels"] == [] and info["provenance"]["hot_pixels"]["applied"] is False
    assert _spike(ds["sci"].ref_cube) > 15.0          # the reference's +80, less the sky's +52


def test_hot_pixels_true_without_a_background_pointing_refuses(tmp_path):
    with pytest.raises(ValueError, match="hot_pixels=True but no dedicated background"):
        _load(_set(str(tmp_path), with_bkg=False)[:2] + [_write(os.path.join(str(tmp_path), "r.fits"),
                                                                  "REFSTAR", 0.0, bkgsub=True)],
              hot_pixels=True)


def test_the_same_exposure_twice_is_refused_not_stacked(tmp_path):
    """The archive calints and a re-reduction of them in a subfolder, found together."""
    a = _set(str(tmp_path / "mastDownload"))
    b = _set(str(tmp_path / "reproc" / "stage2"))
    with pytest.raises(ValueError, match="occur more than once"):
        _load(a + b)


def test_the_signature_tells_frame_treatments_apart(tmp_path):
    _, on, _ = _load(_set(str(tmp_path / "x")))
    _, off, _ = _load(_set(str(tmp_path / "y")), hot_pixels=False)
    _, on2, _ = _load(_set(str(tmp_path / "z")))
    assert frames_signature(on["provenance"]) == frames_signature(on2["provenance"])
    assert frames_signature(on["provenance"]) != frames_signature(off["provenance"])
    assert frames_signature(None) == "none"


def test_the_reducer_puts_the_frames_record_into_the_run_setup(tmp_path):
    from klip_tpe.reducer import KLIPReducer, PartitionedReducer
    ds, info, _ = _load(_set(str(tmp_path)))
    red = PartitionedReducer({"sci": KLIPReducer(ds["sci"], pxscale=PX, lam_m=11.3e-6, diam_m=6.5)})
    d = red.describe()
    assert d["partitions"]["sci"]["frames"]["hot_pixels"]["n"] == 1
    assert frames_provenance(red)["files"] == info["provenance"]["files"]


# ----------------------------------------------------------------- rebuilding a finished run
@pytest.fixture(scope="module")
def LA():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location("library_ablation_hp", os.path.join(ROOT, "scripts",
                                                                                      "library_ablation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _frames_at(root, px):
    os.makedirs(root, exist_ok=True)
    sh = fits.Header({"PIXAR_A2": px ** 2})
    fits.HDUList([fits.PrimaryHDU(header=fits.Header({"FILTER": "F1140C"})),
                  fits.ImageHDU(np.zeros((1, 4, 4), np.float32), header=sh, name="SCI")]).writeto(
        os.path.join(root, "jw01_calints.fits"), overwrite=True)
    return root


def test_an_override_wins(LA):
    assert LA.hot_pixels_for_rebuild({}, "/nonexistent", override=True) is True
    assert LA.hot_pixels_for_rebuild({}, "/nonexistent", override=False) is False


def test_a_run_that_recorded_its_frames_rebuilds_the_way_it_ran(LA):
    for applied in (True, False):
        setup = {"reducer": {"partitions": {"sci": {"frames": {"hot_pixels": {"applied": applied}}}}}}
        assert LA.hot_pixels_for_rebuild(setup, "/nonexistent") is applied


def test_an_old_run_on_its_own_frames_stays_unrepaired_and_on_others_the_loader_decides(LA, tmp_path):
    """v7 on the archive calints (0.1103"/px) must reproduce; v7's configuration on the
    reprocessed calints (0.1100"/px) is a different reduction and gets the loader default."""
    setup = {"pxscale": 0.11032674199848376, "config": {"ann_edges": [6.7, 20.0]}}
    own = _frames_at(str(tmp_path / "archive"), 0.11032674199848376)
    other = _frames_at(str(tmp_path / "reproc"), 0.1100)
    assert LA.hot_pixels_for_rebuild(setup, own) is False
    assert LA.hot_pixels_for_rebuild(setup, other) is None


def test_only_a_config_to_go_on_rebuilds_as_the_run_ran(LA, tmp_path):
    cfg = {"ann_edges": [6.7, 20.0]}
    a = LA._args_for(cfg, str(tmp_path), "auto")
    assert a.hot_pixels is False
    assert LA._args_for(cfg, str(tmp_path), "auto", hot_pixels=True).hot_pixels is True
    assert LA._args_for(cfg, str(tmp_path), "auto", hot_pixels=None).hot_pixels is None
