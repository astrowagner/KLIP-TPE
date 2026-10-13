"""Injection on the job's threads, and only into the frames a reduction keeps.

From the NEAR2 parallelism note of 2026-10-13: ``inject_sources`` ran on one thread and was
the largest fixed cost of a job, 0.42-0.49 s per n6 job at any bin, for every frame of the
night whether the frame selection kept it or not.  What must hold:

* an injected frame is the same, bit for bit, as the old frame-by-frame loop made it, on
  any number of threads, for every kind of model (a stamp shared by all frames, rotated or
  not, a fallback template, a PSF per frame, a throughput per frame);
* ``rotate_ccw_many`` is ``rotate_ccw`` angle by angle;
* a reduction injects only its kept frames, hands back the same images, and injects a
  frame of the same sources at most once, also from two threads at once;
* a reducer still pickles.
"""
from __future__ import annotations

import pickle
import threading

import numpy as np
import pytest

from klip_tpe.injection import (AiryPSF, FramePSF, GaussianPSF, LibraryPSF, TemplatePSF, add_stamp,
                                inject_sources, takes_azimuth)
from klip_tpe.klip import frame_selection_mask, rotate_ccw, rotate_ccw_many
from klip_tpe.metrics import Source
from klip_tpe.reducer import Dataset, KLIPReducer, ReductionRequest
from klip_tpe.synthetic import synthetic_klip_dataset

PX = 0.05


def _bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {4: np.uint32, 8: np.uint64}[a.dtype.itemsize]
    return np.array_equal(a.view(view), b.view(view))


def _inject_reference(cube, angles, sources, model, pxscale, truenorth=0.0, fallback=None,
                      angle_convention="pa"):
    """inject_sources as it was before 2026-10-13: one source at a time, frame by frame."""
    out = np.array(cube, dtype=np.float32, copy=True)
    n, ny, nx = out.shape
    center = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    angles = np.asarray(angles, float)
    for s in sources:
        st, sc, ok = model.stamp(s.rho)
        m = model
        if not ok:
            st, sc, ok = fallback.stamp(s.rho)
            m = fallback
        st = np.asarray(st, float)
        r_px = s.rho / pxscale
        per_frame = bool(getattr(model, "per_frame", False)) and ok
        az_aware = takes_azimuth(model)
        thru = None if az_aware else float(model.throughput(s.rho))
        base = None if (az_aware or per_frame) else st * (s.contrast * m.flux_unit * thru)
        for j in range(n):
            if angle_convention == "pa":
                az = s.theta - truenorth - 270.0 - angles[j]
            else:
                az = s.theta - angles[j]
            azr = np.deg2rad(az)
            xs, ys = center[0] + r_px * np.cos(azr), center[1] + r_px * np.sin(azr)
            t_j = float(model.throughput(s.rho, az)) if az_aware else thru
            if per_frame:
                stj, sc, fu_j = model.frame_stamp(j, s.rho)
                stamp = np.asarray(stj, float) * (s.contrast * fu_j * t_j)
            elif az_aware:
                stamp = st * (s.contrast * m.flux_unit * t_j)
            else:
                stamp = base
            if m.refpa_deg is not None:
                stamp = rotate_ccw(stamp, az - m.refpa_deg, cval=0.0)
            add_stamp(out[j], stamp, sc, xs, ys)
    return out


class _AzimuthLibrary(LibraryPSF):
    """A library whose throughput depends on the detector azimuth (a 4QPM stands in)."""

    azimuth_dependent = True

    def throughput(self, rho_as, az_deg=None):
        if az_deg is None:
            return 0.6
        return float(0.35 + 0.6 * np.cos(np.deg2rad(2.0 * az_deg)) ** 2)


def _library(refpa, nan=False):
    rng = np.random.default_rng(11)
    sl = np.stack([GaussianPSF(2.0 + 0.4 * i, size=21).stamp(0)[0] * (1.0 + 0.3 * rng.random((21, 21)))
                   for i in range(4)])
    if nan:
        sl[:, 2, 3] = np.nan
    return LibraryPSF(sl, [0.2, 0.6, 1.0, 1.4], center=(10.0, 10.0), ee_radius_px=3.0, refpa_deg=refpa,
                      flux_unit=1e4)


def _cube(n=29, size=56, seed=2):
    rng = np.random.default_rng(seed)
    cube = (rng.standard_normal((n, size, size)) * 30 + 100).astype(np.float32)
    ang = np.sort(rng.uniform(-35, 35, n))
    ang[5] = 0.0
    return cube, ang


def _models():
    n = 29
    rng = np.random.default_rng(3)
    cube, _ = _cube(n)
    yy, xx = np.mgrid[0:56, 0:56]
    star = np.exp(-0.5 * ((xx - 27.5) ** 2 + (yy - 27.5) ** 2) / 2.0 ** 2) * 1e4
    fcube = (cube + star[None] * rng.uniform(0.8, 1.2, n)[:, None, None]).astype(np.float32)
    fp = np.column_stack([rng.uniform(0.8, 1.2, n), rng.uniform(2.0, 2.4, n), rng.uniform(2.0, 2.4, n),
                          np.zeros(n), rng.uniform(0, 1, n)])
    fp[3, 1] = np.nan                                   # a frame without a fit: the median model
    tmpl = TemplatePSF(GaussianPSF(3.0, size=15).stamp(0)[0], star_flux=1e4)
    return {
        "shared, rotated": (_library(-20.0), None, "pa"),
        "shared, rotated, zero angle": (_library(270.0), None, "math"),   # source 0 in frame 5: 0 deg
        "shared, not rotated": (GaussianPSF(3.0, star_flux=1e4), None, "pa"),
        "fallback template": (_library(15.0), tmpl, "pa"),
        "stamp with a NaN": (_library(-20.0, nan=True), None, "pa"),
        "PSF per frame (Airy)": (AiryPSF(2.2, frame_params=fp, amp=1e4), None, "pa"),
        "PSF per frame (frames)": (FramePSF(fcube, r_ee=3.0), None, "pa"),
        "throughput per frame": (_AzimuthLibrary(_library(None).slices, [0.2, 0.6, 1.0, 1.4], (10.0, 10.0),
                                                 3.0, flux_unit=1e4), None, "pa"),
        "throughput per frame, rotated": (_AzimuthLibrary(_library(None).slices, [0.2, 0.6, 1.0, 1.4],
                                                          (10.0, 10.0), 3.0, refpa_deg=30.0, flux_unit=1e4),
                                          None, "pa"),
    }


# the last two lie outside the library (0.2-1.4"): the fallback template takes them
SOURCES = [Source(0.7, 270.0, 1e-2), Source(0.45, 45.0, 2e-2), Source(1.2, 120.0, 5e-3),
           Source(1.6, 300.0, 5e-3), Source(0.05, 10.0, 1e-2)]


@pytest.mark.parametrize("name", list(_models()))
def test_an_injection_is_what_it_was(name):
    model, fallback, conv = _models()[name]
    cube, ang = _cube()
    src = SOURCES if fallback is not None else SOURCES[:3]
    ref = _inject_reference(cube, ang, src, model, PX, truenorth=1.25 if conv == "pa" else 0.0,
                            fallback=fallback, angle_convention=conv)
    kw = dict(truenorth=1.25 if conv == "pa" else 0.0, fallback=fallback, angle_convention=conv)
    for th in (1, 2, 3):
        assert _bits(ref, inject_sources(cube, ang, src, model, PX, threads=th, **kw)), (name, th)
    # a subset of the frames: those as before, the rest untouched
    sub = np.array([0, 3, 5, 6, 7, 20, 28])
    got = inject_sources(cube, ang, src, model, PX, frames=sub, threads=2, **kw)
    rest = np.setdiff1d(np.arange(cube.shape[0]), sub)
    assert _bits(ref[sub], got[sub]) and _bits(cube[rest], got[rest]), name


def test_rotate_ccw_many_is_rotate_ccw_angle_by_angle():
    rng = np.random.default_rng(5)
    angles = [0.0, 90.0, -90.0, 180.0, 45.0, 1e-9, 33.3, -179.99, 720.5, np.float64(-93.9) - np.float64(12.25)]
    for shape in [(21, 21), (20, 24), (40, 40)]:
        img = rng.standard_normal(shape) * 10
        for cval in (0.0, np.nan):
            many = rotate_ccw_many(img, angles, cval=cval)
            for j, a in enumerate(angles):
                assert _bits(many[j], rotate_ccw(img, a, cval=cval)), (shape, a, cval)
        bad = img.copy()
        bad[3, 4] = np.nan                       # not finite: rotate_ccw's own NaN handling
        many = rotate_ccw_many(bad, angles, cval=0.0)
        for j, a in enumerate(angles):
            assert np.array_equal(many[j], rotate_ccw(bad, a, cval=0.0), equal_nan=True), (shape, a)
    assert rotate_ccw_many(img, []).shape == (0,) + img.shape


# ----------------------------------------------------------------------------- the reducer
def _reducer(nframes=48, size=48, model=None):
    ds = synthetic_klip_dataset(nframes=nframes, size=size, pa_span=50, seed=4)
    rng = np.random.default_rng(6)
    ds.tags = {"corrs": rng.uniform(0.9, 1.0, nframes), "noises": rng.uniform(0.5, 1.5, nframes),
               "coronoise": rng.uniform(0.5, 1.5, nframes)}
    return KLIPReducer(ds, PX, 3.89 * PX / 206265 * 8.2, 8.2, outrad_cap=size // 2 - 4,
                       injection_model=model or _library(-20.0), fallback_model=GaussianPSF(3.0, star_flux=1e4))


def _params(**kw):
    p = {"k_klip": 3, "inrad": 5, "outrad": 18, "bin": 2, "filter": 5, "n_ang": 2, "angsep": 0.3,
         "anglemax": 50.0, "corr_thresh": 0.95, "noise_max": 1.2, "coronoise_max": 1.3}
    p.update(kw)
    return p


def _keep(red, p):
    return frame_selection_mask(red.data.nframes, red.data.tags, p["corr_thresh"], p["noise_max"],
                                p["coronoise_max"], p["bin"], p["k_klip"])


def test_only_the_kept_frames_are_injected(monkeypatch):
    import klip_tpe.reducer as R
    red = _reducer()
    src = [Source(0.5, 30.0, 5e-3), Source(0.8, 200.0, 5e-3)]
    full = _inject_reference(red.data.cube, red.data.angles, src, red.model, PX, fallback=red.fallback)
    asked = []
    real = R.inject_frames

    def spy(out, cube, angles, sources, model, pxscale, frames=None, **kw):
        asked.append(np.asarray(frames).tolist())
        return real(out, cube, angles, sources, model, pxscale, frames, **kw)
    monkeypatch.setattr(R, "inject_frames", spy)
    k1 = _keep(red, _params())
    k2 = _keep(red, _params(corr_thresh=0.93, noise_max=1.4))
    assert 0 < k1.sum() < k2.sum() < red.data.nframes and not (k1 & ~k2).any()
    assert _bits(red._inject(src, k1, threads=2), full[k1])
    assert asked[-1] == np.flatnonzero(k1).tolist()
    assert _bits(red._inject(src, k2), full[k2])
    assert asked[-1] == np.flatnonzero(k2 & ~k1).tolist(), "only the frames it adds"
    n = len(asked)
    assert _bits(red._inject(src, k1), full[k1]) and len(asked) == n, "nothing to add"
    assert _bits(red._inject(src), full)                 # every frame, as before
    assert asked[-1] == np.flatnonzero(~k2).tolist()
    # new sources start again
    red._inject([Source(0.6, 100.0, 1e-2)], k1)
    assert asked[-1] == np.flatnonzero(k1).tolist()


@pytest.mark.parametrize("case", ["cut", "no cut", "fm", "kscan", "extras", "hp binned"])
def test_a_reduction_with_a_frame_cut_is_what_it_was(case):
    """The reduction of injected frames equals a clean reduction of a cube the old code
    injected in full: the same frames reach the high-pass, in the same order."""
    red = _reducer()
    src = [Source(0.5, 30.0, 5e-3), Source(0.8, 200.0, 5e-3)]
    p = _params() if case != "no cut" else _params(corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0)
    if case == "hp binned":
        p["hp_order"] = "binned"
    fm = [Source(0.6, 120.0, 1e-3)] if case == "fm" else None
    req = ReductionRequest(p, src, k_scan=case == "kscan", fm_sources=fm, extras=case == "extras", threads=2)
    got = red.reduce(req)
    pre = Dataset(_inject_reference(red.data.cube, red.data.angles, src, red.model, PX, fallback=red.fallback),
                  red.data.angles, red.data.tags)
    old = KLIPReducer(pre, PX, red.lam_m, red.diam_m, outrad_cap=red.outrad_cap, injection_model=red.model,
                      fallback_model=red.fallback)
    if fm is not None:
        zero = _inject_reference(np.zeros_like(red.data.cube), red.data.angles, fm, red.model, PX,
                                 fallback=red.fallback)
        old.model_cube = lambda sources, keep=None, threads=1: zero if keep is None else zero[keep]
    want = old.reduce(ReductionRequest(p, None, k_scan=case == "kscan", fm_sources=fm, threads=1))
    assert got.meta["n_kept"] == want.meta["n_kept"]
    assert (got.meta["n_kept"] < red.data.nframes) == (case != "no cut")
    assert _bits(got.image, want.image), case
    if fm is not None:
        assert _bits(got.fm_image, want.fm_image)
    if case == "extras":
        assert got.cadi is not None


def test_the_same_sources_from_two_threads_at_once():
    red = _reducer()
    src = [Source(0.5, 30.0, 5e-3)]
    ps = [_params(), _params(corr_thresh=0.93, noise_max=1.4), _params(corr_thresh=0.0)]
    want = [_reducer().reduce(ReductionRequest(p, src)).image for p in ps]
    for rep in range(3):
        red._inj_cache = {}
        got = [None] * len(ps)

        def run(i):
            got[i] = red.reduce(ReductionRequest(ps[i], src, threads=2)).image
        ts = [threading.Thread(target=run, args=(i,)) for i in range(len(ps))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert all(_bits(g, w) for g, w in zip(got, want)), rep


def test_a_reducer_still_pickles():
    red = _reducer()
    src = [Source(0.5, 30.0, 5e-3)]
    a = red.reduce(ReductionRequest(_params(), src, fm_sources=src)).image
    assert red._inj_cache and red._fm_cache
    back = pickle.loads(pickle.dumps(red))
    assert back._inj_cache == {} and back._fm_cache == {} and red._inj_cache
    assert _bits(back.reduce(ReductionRequest(_params(), src, fm_sources=src)).image, a)
