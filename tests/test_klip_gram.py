"""One Gram matrix per zone, and only the modes a target uses (``KLIPParams.gram``).

From the NEAR2 session's note of 2026-10-14 on the per-target KLIP.  The per-target path
formed every target's covariance ``R R^T`` from its references' pixels and all M of its
modes, of which it kept k.  ``gram="zone"`` forms one Gram matrix per zone and reads each
target's covariance off it, and projects the target without forming the modes.  What must
hold:

* ``gram="target"`` (the default) is the KLIP it always was, to the bit, KLIP-FM included,
  though the FM now takes the basis' decomposition instead of making its own.
* ``gram="zone"`` gives the same residuals to float64 rounding (a few float32 units at most,
  on a few pixels), the same NaN masks, dropped targets and reference counts, for ADI, the
  k-scan, KLIP-FM, ``spat_mean``, a searched reference library and the fast path, on one
  thread or several.  ``temp_mean`` keeps the per-target path.
* The reducer, :func:`set_klip_gram` and ``--klip-gram`` carry the mode, and a run keeps
  the mode it was started or last resumed with (``klip_gram.json``).
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from klip_tpe import cli
from klip_tpe.klip import (KLIPParams, _klip_basis_eig, arcdist_deg, klip_annular, klip_basis, klip_fm_zone,
                           reference_mask, zone_indices)
from klip_tpe.reducer import KLIPReducer, ReductionRequest, set_klip_gram
from klip_tpe.synthetic import synthetic_klip_dataset

LOD = 3.0


@pytest.fixture(scope="module")
def cube():
    ds = synthetic_klip_dataset(nframes=40, size=48, pa_span=40, seed=3)
    c = ds.cube.astype(np.float32).copy()
    c[::5, :2, :] = np.nan                       # NaN edges on some frames
    return c, np.asarray(ds.angles, float)


def _ulps(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN masks differ"
    fin = np.isfinite(a)
    return np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))[fin]


def _close(a, b, max_ulp=8, frac=1e-3):
    u = _ulps(a, b)
    assert u.max(initial=0) <= max_ulp and (u > 0).mean() <= frac, (int(u.max(initial=0)), float((u > 0).mean()))


def _bits(a, b):
    return a.shape == b.shape and np.array_equal(np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8))


BASE = KLIPParams(k_klip=8, inrad=4, outrad=20, n_ang=2, angsep=1.0, anglemax=40, n_min_ref=4)
CASES = {
    "adi": {},
    "k 1": dict(k_klip=1),
    "k above the reference count": dict(k_klip=30),
    "k-scan": dict(k_scan=True),
    "spat_mean": dict(spat_mean=True),
    "starved targets dropped": dict(angsep=0.3, anglemax=8.0, n_min_ref=6),
    "the <4 fallback": dict(angsep=8.0, anglemax=10.0, n_min_ref=1),
    "one zone, wide": dict(n_ang=1, anglemax=360.0, angsep=0.5),
    "fast": dict(fast=True),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("threads", [1, 3])
def test_the_zone_gram_is_the_per_target_klip_to_rounding(cube, name, threads):
    c, ang = cube
    p = dataclasses.replace(BASE, threads=threads, **CASES[name])
    a, ia = klip_annular(c, ang, p, LOD)
    b, ib = klip_annular(c, ang, dataclasses.replace(p, gram="zone"), LOD)
    _close(a, b)
    assert np.array_equal(ia["dropped"], ib["dropped"]) and np.array_equal(ia["nref_used"], ib["nref_used"])
    assert ia["klip_gram"] == "target" and ib["klip_gram"] == "zone"
    if name == "starved targets dropped":
        assert ia["n_dropped"] > 0
    # the thread count changes nothing in either mode
    b1, _ = klip_annular(c, ang, dataclasses.replace(p, gram="zone", threads=1), LOD)
    assert _bits(b, b1)


def test_temp_mean_and_k0_keep_the_per_target_path(cube):
    c, ang = cube
    for kw in (dict(temp_mean=True), dict(temp_mean=True, spat_mean=True), dict(k_klip=0)):
        p = dataclasses.replace(BASE, **kw)
        a, _ = klip_annular(c, ang, p, LOD)
        b, ib = klip_annular(c, ang, dataclasses.replace(p, gram="zone"), LOD)
        assert _bits(a, b) and ib["klip_gram"] == "target", kw
    with pytest.raises(ValueError, match="gram"):
        klip_annular(c, ang, dataclasses.replace(BASE, gram="pixels"), LOD)


def _old_per_target(c, ang, p, fm=None):
    """The per-target path as it was before the FM took the basis' decomposition: a
    reference written from klip_basis and klip_fm_zone, each decomposing on its own."""
    n, ny, nx = c.shape
    flat = c.reshape(n, -1).astype(np.float64)
    out = np.full((n, ny * nx), np.nan, np.float32)
    fout = np.full((n, ny * nx), np.nan, np.float32)
    z = lambda A: np.where(np.isfinite(A), A, 0.0)
    asep = p.angsep * abs(arcdist_deg(p.inrad, p.outrad, LOD))
    for jj in range(p.n_ang):
        idx = zone_indices((ny, nx), p.inrad, p.outrad, jj * 360.0 / p.n_ang, min((jj + 1) * 360.0 / p.n_ang, 360.0))
        D = flat[:, idx]
        for t in range(n):
            refs = reference_mask(ang, t, asep, p.anglemax)
            if refs.sum() < 4:
                refs = np.ones(n, bool)
                refs[t] = False
            R = z(D[refs])
            Z = klip_basis(R, p.k_klip)
            T = z(D[t:t + 1])
            out[t, idx] = (T - (T @ Z.T) @ Z)[0]
            if fm is not None:
                F = fm.reshape(n, -1).astype(np.float64)[:, idx]
                fout[t, idx] = klip_fm_zone(R, z(F[refs]), T, z(F[t:t + 1]), p.k_klip)[0]
    return out.reshape(c.shape), fout.reshape(c.shape)


def test_the_target_mode_is_the_old_klip_to_the_bit_fm_included(cube):
    c, ang = cube
    rng = np.random.default_rng(1)
    fm = np.zeros_like(c)
    fm[:, 28:33, 30:35] = rng.uniform(0.5, 1.0, (c.shape[0], 5, 5))
    p = dataclasses.replace(BASE, n_min_ref=1)
    res, info, fmres = klip_annular(c, ang, p, LOD, fm_cube=fm)
    ores, ofm = _old_per_target(c, ang, p, fm)
    assert _bits(res, ores) and _bits(fmres, ofm)
    # the basis' decomposition is the FM's own: same eigenvalues, vectors and modes
    D = np.where(np.isfinite(c), c, 0).reshape(c.shape[0], -1)[:20, :300].astype(np.float64)
    Z, lam, V, _ = _klip_basis_eig(D, 6)
    T, M = D[:3] * 0.9, D[:3] * 0.01
    Mref = D * 0.01
    assert _bits(klip_fm_zone(D, Mref, T, M, 6), klip_fm_zone(D, Mref, T, M, 6, eig=(lam, V, Z)))
    assert _bits(klip_fm_zone(D, Mref, T, M, 6), klip_fm_zone(D, Mref, T, M, 6, eig=(lam, V, None)))


def test_klip_fm_in_zone_mode(cube):
    c, ang = cube
    fm = np.zeros_like(c)
    fm[:, 28:33, 30:35] = 1.0
    for kw in ({}, dict(spat_mean=True), dict(fast=True)):
        p = dataclasses.replace(BASE, **kw)
        a, _, fa = klip_annular(c, ang, p, LOD, fm_cube=fm)
        b, _, fb = klip_annular(c, ang, dataclasses.replace(p, gram="zone"), LOD, fm_cube=fm)
        _close(a, b)
        _close(fa, fb, max_ulp=16, frac=2e-3)


def _library_reducer():
    ds = synthetic_klip_dataset(nframes=40, size=48, pa_span=20, seed=3)
    rng = np.random.default_rng(4)
    ds.ref_cube = np.stack([ds.cube[i % 40] * 1.01 + rng.normal(0, 0.05, (48, 48)) for i in range(12)]).astype(np.float32)
    from klip_tpe.injection import GaussianPSF
    red = KLIPReducer(ds, pxscale=0.05, lam_m=4.4e-6, diam_m=6.5, injection_model=GaussianPSF(3.0, star_flux=1e4),
                      fwhm_px=3.0, outrad_cap=20)
    red.set_reference_library(partition=np.array(["rollA"] * 20 + ["rollB"] * 20), min_keep={"psfref": 1}, n_min_ref=2)
    return red


def test_a_searched_library_and_the_reducer(cube):
    from klip_tpe.metrics import Source
    red = _library_reducer()
    p = dict(k_klip=3, bin=1, filter=5, n_ang=1, inrad=4, outrad=16, angsep=0.0, anglemax=360.0,
             corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0, use_rdi=True, n_min_ref=2,
             nkeep_altroll=6, nkeep_psfref=5)
    src = [Source(0.4, 30.0, 5e-3)]
    for kw in ({}, {"fm_sources": [Source(0.5, 100.0, 1e-3)]}, {"k_scan": True}):
        a = red.reduce(ReductionRequest(dict(p), src, **kw))
        b = red.reduce(ReductionRequest(dict(p, klip_gram="zone"), src, **kw))
        assert a.meta["klip_gram"] == "target" and b.meta["klip_gram"] == "zone"
        _close(a.image, b.image)
        if a.fm_image is not None:
            _close(a.fm_image, b.fm_image, max_ulp=16, frac=2e-3)
    # the reducer's default, and what the record says
    set_klip_gram(red, "zone")
    assert red.defaults["klip_gram"] == "zone" and red.describe()["defaults"]["klip_gram"] == "zone"
    c = red.reduce(ReductionRequest(dict(p), src))
    assert c.meta["klip_gram"] == "zone"
    with pytest.raises(ValueError):
        set_klip_gram(red, "pixel")


def test_set_klip_gram_on_every_partition():
    from klip_tpe.reducer import Dataset, PartitionedReducer
    rng = np.random.default_rng(2)
    parts = {}
    for name in ("a", "b"):
        c = rng.normal(0, 1, (12, 32, 32)).astype(np.float32)
        parts[name] = Dataset(c, np.linspace(-20, 20, 12))
    reds = {k: KLIPReducer(v, 0.05, 4e-6, 8.0, outrad_cap=14) for k, v in parts.items()}
    pr = PartitionedReducer(reds, max_workers=1, pool="threads")
    assert sorted(set_klip_gram(pr, "zone")) == ["a", "b"]
    assert all(r.defaults["klip_gram"] == "zone" for r in pr.reducers.values())


def test_the_command_line_and_a_run_keep_the_mode(tmp_path, monkeypatch):
    run = tmp_path / "run_x"
    run.mkdir()
    base = ["resume", "--root", str(tmp_path), "--run-dir", str(run)]

    def settle(extra):
        a = cli._build_parser().parse_args(base + extra)
        a._argv = base + extra
        cli._klip_gram_for_run(a)
        return a.klip_gram

    rec = run / "klip_gram.json"
    assert settle([]) == "target" and not rec.exists(), "the default leaves no record"
    assert settle(["--klip-gram", "target"]) == "target" and not rec.exists()
    assert settle(["--klip-gram", "zone"]) == "zone" and rec.exists()
    assert settle([]) == "zone", "a later resume keeps it"
    assert settle(["--klip-gram", "target"]) == "target"
    d = json.loads(rec.read_text())
    assert d["klip_gram"] == "target" and [h["klip_gram"] for h in d["history"]] == ["zone", "target"]
    assert settle([]) == "target"
    # the option reaches every reducer before the workers start
    a = cli._build_parser().parse_args(["near", "--instrument", "generic", "--cube", "x.fits", "--klip-gram", "zone"])
    assert a.klip_gram == "zone"
    red = _library_reducer()
    cli._set_klip_gram(red, a, log=lambda s: None)
    assert red.defaults["klip_gram"] == "zone"


from test_sdi import groups  # noqa: E402,F401  (the SDI test's three wavelength groups)


def test_sdi_rows_a_reference_star_and_fm_in_zone_mode(groups):  # noqa: F811
    """ADI + SDI + RDI in one library: the pool's Gram holds every row a target can take."""
    from klip_tpe.metrics import Source
    from test_sdi import _cfg, _partitioned
    lib = _partitioned(groups, sdi=True, ref=True)
    for kw in (dict(nkeep_sdi=8, nkeep_psfref=4), dict(nkeep_sdi=8, nkeep_psfref=0), dict(nkeep_sdi=0, nkeep_psfref=4)):
        a = lib.reduce_config(_cfg(use_rdi=True, **kw))
        b = lib.reduce_config(_cfg(use_rdi=True, klip_gram="zone", **kw))
        _close(a.image, b.image)
        for pid in a.meta:
            assert b.meta[pid]["klip_gram"] == "zone" and np.array_equal(a.meta[pid]["nref_used"], b.meta[pid]["nref_used"])
    src = [Source(0.4, 70.0, 2e-3)]
    a = lib.reduce_config(_cfg(use_rdi=True, nkeep_sdi=8, nkeep_psfref=4), injections=src, fm_sources=src)
    b = lib.reduce_config(_cfg(use_rdi=True, nkeep_sdi=8, nkeep_psfref=4, klip_gram="zone"), injections=src,
                          fm_sources=src)
    _close(a.image, b.image)
    _close(a.fm_image, b.fm_image, max_ulp=16, frac=2e-3)
