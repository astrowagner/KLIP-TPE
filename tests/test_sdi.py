"""ADI, SDI and RDI in one KLIP step: the other wavelength groups' frames as a searched pool.

From the ALES reduction's note of 2026-10-14 (its section 3) and the answer of the same day.
What must hold:

* the rescaling magnifies a frame about the star by ``s`` (a source at ``p`` lands at
  ``c + s (p - c)``), keeps NaN where the data had none, returns the frame itself at ``s = 1``
  and the same numbers on any thread count;
* the displacement rule admits a scaled row when the planet has moved, by rotation and
  scaling together, at least the arc ``angsep`` asks for at the annulus' mid radius, within
  ``anglemax`` of rotation, and has no fallback; at ``s = 1`` it is the chord of the ADI rule;
* arrays shared between processes come back as they went, from a forked child too;
* through :class:`PartitionedReducer`, ``nkeep_sdi = 0`` is the reduction without an SDI pool
  to the bit (and prepares nothing), ``nkeep_sdi > 0`` brings the other groups' rows in, with
  worker processes (shared memory) and threads giving the same image, injected and with KLIP-FM
  too, and a count above the pool keeps every eligible row;
* the KLIP-FM rows of the SDI pool carry the planet at ``r * s``;
* ``sdi_lenders = "selected"`` keeps an excluded group's frames out of the pools;
* the library declares ``nkeep_sdi`` from 0 to the cap, records the pool, and the cost
  estimate grows with the counts.
"""
from __future__ import annotations

import multiprocessing as mp
import pickle

import numpy as np
import pytest

from klip_tpe.injection import GaussianPSF
from klip_tpe.metrics import Source
from klip_tpe.parallel import SharedArrays
from klip_tpe.reducer import Dataset, KLIPReducer, PartitionedReducer, ReductionRequest, SiblingFrames
from klip_tpe.reflib import ReferenceGroup, ReferenceLibrary
from klip_tpe.rescale import displacement_px, rescale_frame, rescale_frames
from klip_tpe.space import Config
from klip_tpe.synthetic import synthetic_klip_dataset

LAMS = {"a": 3.3e-6, "b": 3.6e-6, "c": 3.9e-6}
N = 48


def _same(a, b):
    return np.array_equal(np.nan_to_num(a, nan=-7.0), np.nan_to_num(b, nan=-7.0))


def _blob(x0, y0, sig=1.5, n=N):
    yy, xx = np.mgrid[0:n, 0:n]
    return np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sig ** 2)


def _centroid(img, h=4):
    iy, ix = np.unravel_index(np.nanargmax(img), img.shape)
    w = np.nan_to_num(img[iy - h:iy + h + 1, ix - h:ix + h + 1])
    gy, gx = np.mgrid[iy - h:iy + h + 1, ix - h:ix + h + 1]
    return float((w * gx).sum() / w.sum()), float((w * gy).sum() / w.sum())


def test_rescaling_moves_a_source_by_the_scale_and_keeps_nan():
    c = (N - 1) / 2.0
    img = _blob(33.0, 17.0)
    for s in (0.9, 1.08):
        x, y = _centroid(rescale_frame(img, s))
        assert abs(x - (c + s * (33.0 - c))) < 0.08 and abs(y - (c + s * (17.0 - c))) < 0.08, (s, x, y)
    # NaN: an edge of NaN rows stays NaN (and moves with the scale), nothing else is lost
    rng = np.random.default_rng(0)
    noisy = rng.normal(size=(N, N))
    noisy[:3, :] = np.nan
    out = rescale_frame(noisy, 0.95)
    assert np.isnan(out[:3]).all() and np.isfinite(out[8:-6, 6:-6]).all()
    out = rescale_frame(noisy, 1.05)
    assert np.isnan(out[:1]).all() and np.isfinite(out[4:, :]).all()
    # a cube: s = 1 is the cube itself, threads change nothing, the dtype is float32
    cube = rng.normal(size=(12, N, N)).astype(np.float32)
    assert np.array_equal(rescale_frames(cube, 1.0), cube)
    a, b = rescale_frames(cube, 0.97, threads=1), rescale_frames(cube, 0.97, threads=3)
    assert a.dtype == np.float32 and _same(a, b)
    np.testing.assert_allclose(a[3], rescale_frame(cube[3], 0.97), rtol=1e-6, atol=1e-6)
    # the displacement: the chord at s = 1, the spectral shift at dPA = 0
    np.testing.assert_allclose(displacement_px(10.0, [1.0, 1.0], [0.0, 10.0]), [0.0, 2 * 10 * np.sin(np.deg2rad(5))])
    np.testing.assert_allclose(displacement_px(10.0, [0.9, 1.2], [0.0, 0.0]), [1.0, 2.0])


def test_the_displacement_rule_admits_what_moved():
    ang = np.array([0.0, 4.0, 8.0, 12.0])                     # 4 target frames
    sci = ReferenceGroup("altroll", np.arange(4), partition=np.zeros(4, int), rotation=True)
    # 8 scaled rows: the same 4 angles at s = 0.9 and at s = 1.0 (a group at the same wavelength)
    scales = np.array([0.9] * 4 + [1.0] * 4)
    sdi = ReferenceGroup("sdi", 4 + np.arange(8), scales=scales, max_keep=100)
    lib = ReferenceLibrary([sci, sdi], np.zeros((12, 4)), target_partition=np.zeros(4, int),
                           angles=np.concatenate([ang, ang, ang]), mid_px=10.0)
    assert sdi.max_keep == 8, "the cap never exceeds the rows"
    # angsep_deg is the arc at the mid radius: 1 px of arc at mid 10 is 5.73 deg
    arc_1px = np.rad2deg(1.0 / 10.0)
    # at s = 0.9 the same-time row is displaced |s - 1| mid = 1 px: admitted at 1 px, not at 1.5
    got = lib.eligible(0, sdi, angsep_deg=arc_1px, anglemax=360.0)
    assert 4 in got and 8 not in got, got
    got = lib.eligible(0, sdi, angsep_deg=1.5 * arc_1px, anglemax=360.0)
    assert 4 not in got
    # at s = 1 the rule is the chord 2 mid sin(dPA / 2): 4 deg of rotation is 0.70 px
    chord = 2 * 10 * np.sin(np.deg2rad(2.0))
    got = lib.eligible(0, sdi, angsep_deg=np.rad2deg(0.99 * chord / 10.0), anglemax=360.0)
    assert 9 in got and 8 not in got
    got = lib.eligible(0, sdi, angsep_deg=np.rad2deg(1.01 * chord / 10.0), anglemax=360.0)
    assert 9 not in got and 10 in got
    # anglemax caps the rotation of scaled rows too, and there is no fallback below 4
    got = lib.eligible(0, sdi, angsep_deg=0.0, anglemax=5.0)
    assert set(got.tolist()) == {4, 5, 8, 9}
    assert lib.eligible(0, sdi, angsep_deg=50.0, anglemax=360.0).size == 0
    assert lib.eligible(0, sdi, angsep_deg=50.0, anglemax=360.0, census=True).size == 0
    # selection: a count above the eligible rows keeps them all, below it ranks
    sim = np.zeros((12, 4))
    sim[4:, 0] = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    lib2 = ReferenceLibrary([sci, sdi], sim, target_partition=np.zeros(4, int),
                            angles=np.concatenate([ang, ang, ang]), mid_px=10.0)
    sel = lib2.select(0, {"altroll": 0, "sdi": 2}, angsep_deg=0.0, anglemax=360.0)
    assert sel.tolist() == [5, 7]
    assert lib2.select(0, {"altroll": 0, "sdi": 50}, angsep_deg=0.0, anglemax=360.0).tolist() == list(range(4, 12))
    assert not lib2.is_plain_adi({"altroll": 3, "sdi": 1}) and lib2.is_plain_adi({"altroll": 3, "sdi": 0})
    with pytest.raises(ValueError, match="mid_px"):
        ReferenceLibrary([sci, sdi], np.zeros((12, 4)), angles=np.concatenate([ang, ang, ang]))
    with pytest.raises(ValueError, match="not both"):
        ReferenceGroup("x", np.arange(3), scales=np.ones(3), rotation=True)


def _child(desc, q):
    arrays, closer = desc.attach()
    q.put((float(arrays["p/bcube"].sum()), arrays["p/bcube"].shape, desc.extra["p"]["lam_m"]))
    closer()


def test_shared_arrays_round_trip():
    a = {"p/bcube": np.arange(24, dtype=np.float32).reshape(2, 3, 4), "q/bcube": np.ones((1, 3, 4), np.float32)}
    sa = SharedArrays.create(a, extra={"p": {"lam_m": 3.5e-6}})
    arrays, closer = pickle.loads(pickle.dumps(sa)).attach()
    assert np.array_equal(arrays["p/bcube"], a["p/bcube"]) and np.array_equal(arrays["q/bcube"], a["q/bcube"])
    closer()
    if "fork" in mp.get_all_start_methods():
        ctx = mp.get_context("fork")
        q = ctx.Queue()
        pr = ctx.Process(target=_child, args=(pickle.loads(pickle.dumps(sa)), q))
        pr.start()
        got = q.get(timeout=30)
        pr.join(timeout=30)
        assert got == (float(a["p/bcube"].sum()), (2, 3, 4), 3.5e-6) and pr.exitcode == 0
    sa.release()
    sa.release()                                          # twice is fine


@pytest.fixture(scope="module")
def groups():
    """Three wavelength groups of the synthetic sequence: the same speckles scaled with
    wavelength, noise on top."""
    rng = np.random.default_rng(3)
    base = synthetic_klip_dataset(nframes=24, size=N, pa_span=40, seed=5)
    cubes = {}
    for pid, lam in LAMS.items():
        cube = rescale_frames(base.cube, lam / 3.6e-6)
        cubes[pid] = (np.where(np.isfinite(cube), cube, 0.0) + rng.normal(0, 0.02, cube.shape)).astype(np.float32)
    return base, cubes


def _partitioned(groups, sdi, workers=1, pool="threads", cap=20, lenders="all", ref=False, searched="default"):
    base, cubes = groups
    rng = np.random.default_rng(11)
    reds = {}
    for pid, lam in LAMS.items():
        ds = Dataset(cubes[pid].copy(), base.angles.copy(), tags=None, texp=10.0, name=pid)
        ds.ref_cube = (cubes[pid][:6] * 1.02 + rng.normal(0, 0.03, (6, N, N))).astype(np.float32) if ref else None
        r = KLIPReducer(ds, pxscale=0.05, lam_m=lam, diam_m=8.4,
                        injection_model=GaussianPSF(2.6 * lam / 3.6e-6, star_flux=1e4),
                        fwhm_px=2.6 * lam / 3.6e-6, outrad_cap=20, defaults={"hp_nan_aware": True})
        if sdi:
            r.set_reference_library(partition=np.zeros(24, int), rotation=True,
                                    searched=(("sdi", "psfref") if ref else ("sdi",)) if searched == "default" else searched,
                                    sdi={"n_frames": 48, "cap": cap, "lenders": list(LAMS)})
        reds[pid] = r
    pr = PartitionedReducer(reds, max_workers=workers, pool=pool)
    pr.sdi_lenders = lenders
    return pr


P = dict(k_klip=3, bin=2, filter=5, n_ang=2, inrad=4, outrad=16, angsep=0.8, anglemax=30.0, corr_thresh=0.0,
         noise_max=3.0, coronoise_max=3.0, n_min_ref=2, hp_nan_aware=True, pad_in=True, pad_out=True, use_rdi=False)


def _cfg(selected=None, **kw):
    p = dict(P)
    p.update(kw)
    sel = list(LAMS) if selected is None else list(selected)
    return Config(p, {pid: dict(p) for pid in LAMS}, sel, np.zeros(0))


def test_no_sdi_rows_is_adi_to_the_bit_and_rows_change_it(groups):
    plain = _partitioned(groups, sdi=False)
    lib = _partitioned(groups, sdi=True)
    r = lib.reducers["a"]
    assert [p.name for p in r.reference_params()] == ["nkeep_sdi"]
    nk = r.reference_params()[0]
    assert (nk.lo, nk.hi) == (0.0, 20.0), "0 to the cap"
    assert r.describe()["reference_library"]["pools"] == ["nkeep_altroll all eligible frames (not searched)",
                                                           "nkeep_sdi 0-20 searched"]
    assert lib.sdi_lender_ids(["a"]) == ["a", "b", "c"] and plain.sdi_lender_ids(["a"]) is None
    e0 = plain.reduce_config(_cfg())
    e1 = lib.reduce_config(_cfg(nkeep_sdi=0))
    assert _same(e1.image, e0.image) and all(_same(x, y) for x, y in zip(e1.stack, e0.stack))
    assert all("n_sdi_rows" not in m and m["library_as_adi"] and m["ref_keep"] == {"altroll": 11, "sdi": 0}
               for m in e1.meta.values()), "the ADI point ran the ADI code, no rows prepared"
    e2 = lib.reduce_config(_cfg(nkeep_sdi=10))
    assert not _same(e2.image, e0.image)
    for pid, m in e2.meta.items():
        assert m["rdi_mode"] == "searched" and m["n_sdi_rows"] == 24 and m["ref_keep"] == {"altroll": 11, "sdi": 10}
        assert sorted(m["sdi_lenders"]) == sorted(q for q in LAMS if q != pid)
        assert (m["nref_used"] > m["nref"] * 0).all() and m["library_as_adi"] is False
    # above the pool: every eligible row, so the two counts give the same image (a cap above
    # the pool; the cap itself ranks and truncates, as it should)
    big = _partitioned(groups, sdi=True, cap=100)
    assert big.reducers["a"].reference_params()[0].hi == 48.0, "the cap never exceeds the pool"
    e3 = big.reduce_config(_cfg(nkeep_sdi=24))
    e4 = big.reduce_config(_cfg(nkeep_sdi=1000))
    assert _same(e3.image, e4.image) and not _same(e3.image, e2.image)
    assert all((m["nref_used"] <= 11 + 24).all() for m in e4.meta.values())
    # injected and with KLIP-FM, worker processes (shared memory) agree with threads to the bit
    src = [Source(0.4, 70.0, 2e-3)]
    e5 = lib.reduce_config(_cfg(nkeep_sdi=10), injections=src, fm_sources=src)
    assert e5.fm_image is not None and np.isfinite(e5.fm_image).any()
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("no fork, no worker processes")
    libp = _partitioned(groups, sdi=True, workers=2, pool="processes")
    try:
        assert libp.start_workers() == "processes"
        e6 = libp.reduce_config(_cfg(nkeep_sdi=10), injections=src, fm_sources=src)
        assert _same(e6.image, e5.image) and _same(e6.fm_image, e5.fm_image)
        e7 = libp.reduce_config(_cfg(nkeep_sdi=0))
        assert _same(e7.image, e0.image)
    finally:
        libp.close()


def test_the_three_pools_together_and_the_adi_point(groups):
    """ADI + SDI + RDI: zero counts are plain ADI to the bit, each pool alone changes the image."""
    plain = _partitioned(groups, sdi=False)
    lib = _partitioned(groups, sdi=True, ref=True)
    assert [p.name for p in lib.reducers["a"].reference_params()] == ["nkeep_sdi", "nkeep_psfref"]
    e0 = plain.reduce_config(_cfg())
    e1 = lib.reduce_config(_cfg(use_rdi=True, nkeep_sdi=0, nkeep_psfref=0))
    assert _same(e1.image, e0.image) and all(m["library_as_adi"] for m in e1.meta.values())
    es = lib.reduce_config(_cfg(use_rdi=True, nkeep_sdi=8, nkeep_psfref=0))
    er = lib.reduce_config(_cfg(use_rdi=True, nkeep_sdi=0, nkeep_psfref=4))
    eb = lib.reduce_config(_cfg(use_rdi=True, nkeep_sdi=8, nkeep_psfref=4))
    imgs = [e0.image, es.image, er.image, eb.image]
    assert all(not _same(imgs[i], imgs[j]) for i in range(4) for j in range(i + 1, 4))
    m = eb.meta["b"]
    assert m["ref_keep"] == {"altroll": 11, "sdi": 8, "psfref": 4} and m["n_ref_frames"] == 6 and m["n_sdi_rows"] == 24
    # the science count searched as well: the NIRCam form with three counts
    lib3 = _partitioned(groups, sdi=True, ref=True, searched=None)
    assert [p.name for p in lib3.reducers["a"].reference_params()] == ["nkeep_altroll", "nkeep_sdi", "nkeep_psfref"]
    e = lib3.reduce_config(_cfg(use_rdi=True, nkeep_altroll=0, nkeep_sdi=8, nkeep_psfref=0))
    assert all(m["ref_keep"]["altroll"] == 0 and (m["nref_used"] <= 8).all() for m in e.meta.values()), "pure SDI"


def test_fm_rows_carry_the_planet_at_the_rescaled_position(groups):
    lib = _partitioned(groups, sdi=True)
    r = lib.reducers["a"]                                  # 3.3 um, the bluest: s < 1 for both siblings
    c = (N - 1) / 2.0
    x0, y0 = 36.0, 20.0
    blob = _blob(x0, y0)[None].astype(np.float32)
    sibs = {pid: SiblingFrames(pid, LAMS[pid], np.zeros((1, N, N), np.float32), np.array([0.0]), blob)
            for pid in ("b", "c")}
    prep = r.prepare(ReductionRequest(dict(P), None, fm_sources=[Source(0.4, 70.0, 1e-3)]))
    sdi = r._sdi_rows(sibs, prep)
    assert sdi["rows"].shape == (2, N, N) and sdi["fm_rows"].shape == (2, N, N) and sdi["lenders"] == ["b", "c"]
    for i, pid in enumerate(("b", "c")):
        s = LAMS["a"] / LAMS[pid]
        assert sdi["scales"][i] == pytest.approx(s)
        x, y = _centroid(sdi["fm_rows"][i])
        assert abs(x - (c + s * (x0 - c))) < 0.1 and abs(y - (c + s * (y0 - c))) < 0.1
    # a sibling without a model cube cannot serve a KLIP-FM reduction
    sibs["b"].mcube = None
    with pytest.raises(ValueError, match="model cubes"):
        r._sdi_rows(sibs, prep)
    # its own frames are never its SDI rows
    sibs = {pid: SiblingFrames(pid, LAMS[pid], np.zeros((2, N, N), np.float32), np.zeros(2)) for pid in LAMS}
    prep = r.prepare(ReductionRequest(dict(P), None))
    out = r._sdi_rows(sibs, prep)
    assert out["lenders"] == ["b", "c"] and out["rows"].shape[0] == 4


def test_lenders_switch_and_the_cost_estimate(groups):
    lib_all = _partitioned(groups, sdi=True, lenders="all")
    lib_sel = _partitioned(groups, sdi=True, lenders="selected")
    sel = ["a", "b"]
    assert lib_all.sdi_lender_ids(sel) == ["a", "b", "c"] and lib_sel.sdi_lender_ids(sel) == ["a", "b"]
    ea = lib_all.reduce_config(_cfg(selected=sel, nkeep_sdi=10))
    eb = lib_sel.reduce_config(_cfg(selected=sel, nkeep_sdi=10))
    assert ea.meta["a"]["n_sdi_rows"] == 24 and eb.meta["a"]["n_sdi_rows"] == 12
    assert ea.meta["a"]["sdi_lenders"] == ["b", "c"] and eb.meta["a"]["sdi_lenders"] == ["b"]
    assert not _same(ea.image, eb.image)
    assert lib_sel.describe()["sdi_lenders"] == "selected" and "sdi_lenders" not in _partitioned(groups, False).describe()
    r = lib_all.reducers["a"]
    c0 = r.cost_estimate(dict(P, nkeep_sdi=0))
    c1 = r.cost_estimate(dict(P, nkeep_sdi=20))
    assert c0 is not None and c1 > c0
    with pytest.raises(ValueError, match="cap above 0"):
        r.set_reference_library(partition=np.zeros(24, int), rotation=True, sdi={"n_frames": 48, "cap": 0})
