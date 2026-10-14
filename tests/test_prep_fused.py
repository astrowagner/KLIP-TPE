"""The frames before KLIP, made one bin group at a time.

From the NEAR2 note of 2026-10-14 on the per-frame passes of a draw.  A job used to read
and write the kept frames a whole cube per stage: the injected copy, the compact copy of
the kept frames, the high-pass, the binning.  ``KLIPReducer._prep_fused`` takes a bin group's
frames through every stage while they are in the processor's cache.  What must hold:

* Every product of the preparation (the binned cube, its angles, the groups, the bins
  kept, the KLIP-FM model cube) equals the whole-cube path's to the bit, for every filter,
  bin, frame cut, high-pass kind and order, destriping, injection model and thread count,
  with NaN in the frames and with bins dropped as empty, for a float64 cube too.
* Whole reductions agree to the bit, KLIP-FM, k-scan, the extras and the searched library
  included.
* A set of sources is placed without keeping anything the first time, keeps its stamps
  from the second time, places only the frames a later selection adds, and gives the same
  frames whichever way they were made.
* The headroom check runs once per new set, on a peak computed once.
* A reducer still pickles.
"""
from __future__ import annotations

import pickle
import threading
import warnings

import numpy as np
import pytest

import klip_tpe.reducer as R
from klip_tpe.injection import GaussianPSF
from klip_tpe.metrics import Source
from klip_tpe.reducer import Dataset, KLIPReducer, ReductionRequest
from test_injection_frames import SOURCES, _bits, _library, _models

PX = 0.05
N, SIZE = 29, 56


def _cube(seed=2, nan_edges=False, nan_frames=(), zero_frames=(), dtype=np.float32):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    star = 2e3 * np.exp(-0.5 * ((xx - 27.5) ** 2 + (yy - 27.5) ** 2) / 3.0 ** 2)
    cube = star[None] * rng.uniform(0.9, 1.1, (N, 1, 1)) + rng.standard_normal((N, SIZE, SIZE)) * 30 + 100
    cube = cube.astype(dtype)
    if nan_edges:
        cube[::3, :2, :] = np.nan                    # a registration edge on every third frame
        cube[1::4, :, -3:] = np.nan
    for j in nan_frames:
        cube[j] = np.nan
    for j in zero_frames:
        cube[j] = 0.0
    ang = np.sort(rng.uniform(-35, 35, N))
    ang[5] = 0.0
    return cube, ang


def _reducer(model=None, fallback=None, conv="pa", **cube_kw):
    cube, ang = _cube(**cube_kw)
    rng = np.random.default_rng(6)
    tags = {"corrs": rng.uniform(0.9, 1.0, N), "noises": rng.uniform(0.5, 1.5, N),
            "coronoise": rng.uniform(0.5, 1.5, N)}
    ds = Dataset(cube, ang, tags=tags, texp=10.0)
    return KLIPReducer(ds, PX, 3.89 * PX / 206265 * 8.2, 8.2, outrad_cap=SIZE // 2 - 4,
                       injection_model=model or _library(-20.0), fallback_model=fallback,
                       truenorth=1.25 if conv == "pa" else 0.0, angle_convention=conv,
                       fwhm_px=3.0)


def _params(**kw):
    p = {"k_klip": 3, "inrad": 5, "outrad": 20, "bin": 2, "filter": 5, "n_ang": 2, "angsep": 0.3,
         "anglemax": 50.0, "corr_thresh": 0.95, "noise_max": 1.2, "coronoise_max": 1.3}
    p.update(kw)
    return p


def _both(red, p, injections=None, fm=None, threads=1):
    """``prepare`` by the fused path and by the whole-cube reference, fresh caches each."""
    out = []
    for fused in (True, False):
        red._inj_cache, red._fm_cache = {}, {}
        red._crops = {"inj": R._CropCache(), "fm": R._CropCache()}
        out.append(red.prepare(ReductionRequest(dict(p, prep_fused=fused), injections, fm_sources=fm,
                                                threads=threads)))
    return out


def _same_prep(a, b, what):
    assert _bits(a.bcube, b.bcube), (what, "bcube")
    assert np.array_equal(a.bang, b.bang), (what, "bang")
    assert np.array_equal(a.grp, b.grp) and np.array_equal(a.bkeep, b.bkeep), (what, "groups")
    assert (a.mcube is None) == (b.mcube is None), (what, "mcube")
    if a.mcube is not None:
        assert _bits(a.mcube, b.mcube), (what, "mcube")


CASES = {
    "clean, bin 2": dict(),
    "clean, bin 1": dict(bin=1),
    "clean, bin 7, no cut": dict(bin=7, corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0),
    "clean, bin 29 (one group)": dict(bin=29, corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0, k_klip=1),
    "clean, no filter": dict(filter=0),
    "clean, filter 1": dict(filter=1, bin=3),
    "clean, even filter": dict(filter=12, bin=4),
    "clean, hp binned": dict(hp_order="binned", bin=4),
    "clean, hp binned, bin 1": dict(hp_order="binned", bin=1),
    "clean, destripe": dict(do_destripe=True, destripe_clip=3.0, bin=3),
    "clean, destripe, no clip": dict(do_destripe=True, destripe_clip=0.0, bin=1),
    "clean, nan-aware": dict(hp_nan_aware=True, bin=3),
    "injected, bin 2": dict(),
    "injected, bin 1": dict(bin=1),
    "injected, bin 5, no cut": dict(bin=5, corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0),
    "injected, no filter": dict(filter=0, bin=3),
    "injected, hp binned": dict(hp_order="binned", bin=3),
    "injected, destripe": dict(do_destripe=True, destripe_clip=2.5, bin=2),
    "injected + FM, bin 3": dict(bin=3),
    "injected + FM, bin 1": dict(bin=1),
    "injected + FM, hp binned": dict(hp_order="binned", bin=4),
    "injected + FM, nan-aware": dict(hp_nan_aware=True, bin=2),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("threads", [1, 3])
def test_the_fused_frames_are_the_whole_cube_frames_to_the_bit(name, threads):
    red = _reducer(nan_edges="nan-aware" in name)
    inj = SOURCES[:3] if "injected" in name else None
    fm = [Source(0.6, 120.0, 1e-3), Source(0.9, 250.0, 2e-3)] if "FM" in name else None
    a, b = _both(red, _params(**CASES[name]), inj, fm, threads)
    _same_prep(a, b, name)
    assert a.bcube.shape[0] > 0


@pytest.mark.parametrize("name", list(_models()))
def test_every_injection_model_through_the_groups(name):
    """The injection models of the frame-by-frame test (a shared stamp rotated or not, a
    fallback template, a stamp with a NaN, a PSF per frame, a throughput per frame)."""
    model, fallback, conv = _models()[name]
    red = _reducer(model, fallback, conv)
    src = SOURCES if fallback is not None else SOURCES[:3]
    for p in (_params(bin=3), _params(bin=1, corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0)):
        a, b = _both(red, p, src, src[:2], threads=2)
        _same_prep(a, b, name)


def test_nan_frames_empty_bins_and_a_float64_cube():
    # bins dropped as empty: all-NaN frames (nansum 0), all-zero frames, the IDL high-pass
    # spreading a NaN over its frame
    red = _reducer(nan_frames=(3, 4, 11), zero_frames=(20, 21, 22, 23), nan_edges=True)
    for p in (_params(bin=2, corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0),
              _params(bin=1, corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0),
              _params(bin=3, hp_order="binned", corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0),
              _params(bin=2, filter=0, corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0)):
        for inj, fm in ((None, None), (SOURCES[:2], SOURCES[2:3])):
            a, b = _both(red, p, inj, fm)
            _same_prep(a, b, p)
            assert not a.bkeep.all(), "some bins dropped"
    # a float64 cube: the clean frames keep their precision, the injected ones are float32 (as
    # the injected cube held them), both paths alike
    red = _reducer(dtype=np.float64)
    assert red.data.cube.dtype == np.float64
    for p in (_params(bin=3), _params(bin=1), _params(bin=3, filter=0), _params(bin=3, hp_order="binned")):
        for inj in (None, SOURCES[:2]):
            a, b = _both(red, p, inj)
            _same_prep(a, b, (p, inj is None))


def test_whole_reductions_agree_to_the_bit():
    red = _reducer()
    src = SOURCES[:3]
    fm = [Source(0.6, 120.0, 1e-3)]
    cases = [(_params(), src, {}), (_params(bin=1), src, {}), (_params(), None, {}),
             (_params(bin=3), src, {"fm_sources": fm}), (_params(k_klip=4), src, {"k_scan": True}),
             (_params(), src, {"extras": True}), (_params(), None, {"extras": True}),
             (_params(hp_order="binned", bin=4), src, {"fm_sources": fm})]
    for p, inj, kw in cases:
        got = {}
        for fused in (True, False):
            r = red.reduce(ReductionRequest(dict(p, prep_fused=fused), inj, threads=2, **kw))
            got[fused] = r
        a, b = got[True], got[False]
        assert _bits(a.image, b.image), (p, kw)
        for att in ("fm_image", "nosub", "cadi"):
            x, y = getattr(a, att), getattr(b, att)
            assert (x is None) == (y is None) and (x is None or _bits(x, y)), (att, p, kw)
        assert a.meta["n_kept"] == b.meta["n_kept"] and a.meta["n_binned"] == b.meta["n_binned"]


def test_the_searched_library_ranks_on_the_same_frames():
    """The reference library's ranking cube is the clean binned cube, made by the fused path
    when the reduction takes it."""
    from klip_tpe.synthetic import synthetic_klip_dataset
    ds = synthetic_klip_dataset(nframes=40, size=48, pa_span=20, seed=3)
    rng = np.random.default_rng(4)
    ds.ref_cube = np.stack([ds.cube[i % 40] * 1.01 + rng.normal(0, 0.05, (48, 48)) for i in range(12)]).astype(np.float32)
    red = KLIPReducer(ds, pxscale=0.05, lam_m=4.4e-6, diam_m=6.5, injection_model=GaussianPSF(3.0, star_flux=1e4),
                      fwhm_px=3.0, outrad_cap=20)
    red.set_reference_library(partition=np.array(["rollA"] * 20 + ["rollB"] * 20), min_keep={"psfref": 1}, n_min_ref=2)
    p = dict(k_klip=3, bin=2, filter=5, n_ang=1, inrad=4, outrad=16, angsep=0.0, anglemax=360.0,
             corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0, use_rdi=True, n_min_ref=2,
             nkeep_altroll=6, nkeep_psfref=5)
    src = [Source(0.4, 30.0, 5e-3), Source(0.6, 200.0, 5e-3)]
    imgs = []
    for fused in (True, False):
        red._reflib_cache.clear()
        imgs.append(red.reduce(ReductionRequest(dict(p, prep_fused=fused), src)).image)
    assert _bits(imgs[0], imgs[1])


def test_a_set_of_sources_keeps_its_stamps_from_its_second_reduction(monkeypatch):
    red = _reducer()
    placed = []
    real = R.place_source

    def spy(J, *a, **kw):
        placed.append(list(J))
        return real(J, *a, **kw)
    monkeypatch.setattr(R, "place_source", spy)
    src = SOURCES[:2]
    p1 = _params()
    p2 = _params(corr_thresh=0.93, noise_max=1.4)
    k1 = R.frame_selection_mask(N, red.data.tags, p1["corr_thresh"], p1["noise_max"], p1["coronoise_max"], 2, 3)
    k2 = R.frame_selection_mask(N, red.data.tags, p2["corr_thresh"], p2["noise_max"], p2["coronoise_max"], 2, 3)
    assert 0 < k1.sum() < k2.sum() < N and not (k1 & ~k2).any()
    ref1 = red.prepare(ReductionRequest(dict(p1, prep_fused=False), src)).bcube
    ref2 = red.prepare(ReductionRequest(dict(p2, prep_fused=False), src)).bcube

    def run(p):
        placed.clear()
        out = red.prepare(ReductionRequest(p, src)).bcube
        return out, sorted(set(j for J in placed for j in J))
    b1, f1 = run(p1)                              # first sighting: placed, nothing kept
    assert _bits(b1, ref1) and f1 == np.flatnonzero(k1).tolist()
    assert red._crops["inj"].key is None and red._crops["inj"].seen is not None
    b1, f1 = run(p1)                              # second: placed again, and kept
    assert _bits(b1, ref1) and f1 == np.flatnonzero(k1).tolist()
    assert sorted(red._crops["inj"].frames) == np.flatnonzero(k1).tolist()
    b2, f2 = run(p2)                              # a wider selection: only the frames it adds
    assert _bits(b2, ref2) and f2 == np.flatnonzero(k2 & ~k1).tolist()
    b1, f1 = run(p1)                              # everything kept: nothing placed
    assert _bits(b1, ref1) and f1 == []
    # another set passing through does not evict the kept one until it comes back itself
    run(_params())
    other = [Source(0.6, 100.0, 1e-2)]
    red.prepare(ReductionRequest(p1, other))
    b1, f1 = run(p1)
    assert _bits(b1, ref1) and f1 == []
    red.prepare(ReductionRequest(p1, other))      # the other set's second time: it is kept now
    assert red._crops["inj"].key == tuple(s.as_tuple() for s in other)
    # the FM model cube keeps its own
    fm = [Source(0.6, 120.0, 1e-3)]
    for _ in range(2):
        red.prepare(ReductionRequest(p1, None, fm_sources=fm))
    assert red._crops["fm"].key == tuple(s.as_tuple() for s in fm) and red._crops["fm"].frames


def test_the_headroom_check_runs_once_per_new_set_on_a_peak_taken_once(monkeypatch):
    red = _reducer()
    peaks, checks = [], []
    real_peak, real_check = R.cube_peak, R._check_float32_headroom
    monkeypatch.setattr(R, "cube_peak", lambda c: peaks.append(1) or real_peak(c))
    monkeypatch.setattr(R, "_check_float32_headroom", lambda *a, **k: checks.append(k.get("peak")) or real_check(*a, **k))
    p = _params()
    # new, back (kept), new, the kept one again: two sets seen for the first time
    for src in (SOURCES[:1], SOURCES[:1], SOURCES[1:2], SOURCES[:1]):
        red.prepare(ReductionRequest(p, src))
    assert len(peaks) == 1 and len(checks) == 2 and len(set(checks)) == 1
    # a faint injection still warns
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        red.prepare(ReductionRequest(p, [Source(0.6, 30.0, 1e-12)]))
    assert any("float32 rounding" in str(x.message) for x in w)


def test_two_threads_at_once_and_pickling():
    red = _reducer()
    src = SOURCES[:2]
    ps = [_params(), _params(corr_thresh=0.93, noise_max=1.4), _params(corr_thresh=0.0)]
    want = [_reducer().reduce(ReductionRequest(dict(p, prep_fused=False), src)).image for p in ps]
    for rep in range(3):
        red._crops = {"inj": R._CropCache(), "fm": R._CropCache()}
        if rep:
            red.reduce(ReductionRequest(ps[0], src))          # the set comes back: kept from here on
        got = [None] * len(ps)

        def run(i):
            got[i] = red.reduce(ReductionRequest(ps[i], src, threads=2)).image
        ts = [threading.Thread(target=run, args=(i,)) for i in range(len(ps))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert all(_bits(g, w) for g, w in zip(got, want)), rep
    assert red._crops["inj"].frames
    back = pickle.loads(pickle.dumps(red))
    assert back._crops["inj"].frames == {} and back._crops["inj"].seen is None and red._crops["inj"].frames
    assert _bits(back.reduce(ReductionRequest(ps[0], src)).image, want[0])
