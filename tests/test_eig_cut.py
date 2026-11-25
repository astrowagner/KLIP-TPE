"""The KL modes' eigenvalue cut is the decomposition's round-off floor (the audit of 2026-11-25).

The IDL predecessor dropped every mode whose eigenvalue lay below 1e-6, so frames of small values,
such as a cube in units of the star's peak, lost the modes asked for, silently.  The cut is now
``m eps lam_max``, below which an eigenvalue cannot be told from 0, and a cube times ``s`` reduces
to its image times ``s``.  ``RunConfig.klip_eig_cut = "absolute"`` keeps the old cut for runs saved
before.
"""
from __future__ import annotations

import numpy as np
import pytest

import klip_tpe.klip as K
import klip_tpe.klip_gpu as KG
from klip_tpe import PartitionedReducer, RunConfig
from klip_tpe.injection import GaussianPSF
from klip_tpe.metrics import Source
from klip_tpe.reducer import Dataset, KLIPReducer, ReductionRequest, _eig_tol, set_eig_cut, set_klip_gram
from klip_tpe.synthetic import synthetic_klip_dataset

PX = 0.05
LAM = 3.89 * PX / 206265 * 8.2
PARAMS = dict(k_klip=8, inrad=8, outrad=26, n_ang=1, angsep=0.5, anglemax=360.0, fast=True, bin=1, filter=0)
EPS = float(np.finfo(np.float64).eps)


def test_the_floor_is_the_decompositions_round_off():
    lam = np.array([4.0, 1.0, 1e-3, 0.0, -1e-17])
    assert K.eig_floor(lam) == pytest.approx(5 * EPS * 4.0)
    assert K.eig_floor(lam, 1e-6) == 1e-6 and K.eig_floor(np.array([-1.0, -2.0])) == 0.0
    assert list(K._keep_modes(lam)) == [True, True, True, False, False]
    # the absolute cut as before: the basis keeps an eigenvalue equal to it and the FM drops it
    edge = np.array([1.0, 1e-6, 1e-7])
    assert list(K._keep_modes(edge, 1e-6)) == [True, True, False]
    assert list(K._fm_valid(edge, 1e-6)) == [True, False, False]
    # the card keeps the same modes, problem by problem
    top = np.array([[4.0, 1.0, 1e-3, 3e-15], [1e-12, 1e-15, 1e-28, 0.0]])
    kept = KG.kept_modes(np, top, [5, 5])
    assert kept.tolist() == [[bool(v > 5 * EPS * 4.0) for v in top[0]], [True, True, False, False]]
    assert KG.kept_modes(np, top, [5, 5], KG.EIG_TOL).tolist() == (top > 1e-6).tolist()


def test_a_rank_deficient_basis_drops_its_null_mode_at_any_scale():
    rng = np.random.default_rng(1)
    for scale in (1e3, 1.0, 1e-6):
        R = rng.normal(size=(30, 400)) * scale
        R = R - R.mean(axis=0, keepdims=True)                 # the mean frame taken out: rank 29
        lam = np.sort(np.linalg.eigh(R @ R.T)[0])[::-1]
        assert K._keep_modes(lam).sum() == 29
        assert abs(lam[-1]) < 0.5 * K.eig_floor(lam)           # below the floor
        Z = K.klip_basis(R, 30)
        assert np.abs(Z[-1]).max() == 0.0
        assert np.allclose(Z[:29] @ Z[:29].T, np.eye(29), atol=1e-9)
    # with the absolute cut, frames of small values keep no mode at all
    assert np.abs(K.klip_basis(R, 30, tol=K.EIG_TOL_IDL)).max() == 0.0


def test_the_basis_and_the_forward_model_scale_with_the_frames():
    rng = np.random.default_rng(2)
    R = rng.normal(size=(12, 300)) + 3.0 * rng.normal(size=(12, 1)) * rng.normal(size=(1, 300))
    M = 0.2 * rng.normal(size=(12, 300))
    T = rng.normal(size=(2, 300))
    Mt = 0.2 * rng.normal(size=(2, 300))
    Z1 = K.klip_basis(R, 8)
    fm1 = K.klip_fm_zone(R, M, T, Mt, 8)
    for s in (1e-4, 1e-7, 1e-10):
        assert np.allclose(K.klip_basis(R * s, 8), Z1, rtol=1e-7, atol=1e-9)
        assert np.allclose(K.klip_fm_zone(R * s, M * s, T * s, Mt * s, 8), fm1 * s, rtol=1e-6, atol=1e-9 * s)
    # the absolute cut loses every mode at 1e-7: the forward model is the planet itself
    s = 1e-7
    assert np.array_equal(K.klip_fm_zone(R * s, M * s, T * s, Mt * s, 8, tol=K.EIG_TOL_IDL), Mt * s)


def _images(scale, cut, gram="target", fast=True, k=8, src=None):
    ds = synthetic_klip_dataset(nframes=30, size=64, pa_span=60, seed=3)
    d = Dataset((ds.cube * scale).astype(np.float32), ds.angles, ds.tags, texp=ds.texp)
    red = KLIPReducer(d, PX, LAM, 8.2, injection_model=GaussianPSF(4.0, star_flux=1e4 * scale), outrad_cap=30)
    set_klip_gram(red, gram)
    assert set_eig_cut(red, cut) == [None]
    params = dict(PARAMS, k_klip=k, fast=fast, anglemax=360.0 if fast else 40.0)
    return red.reduce(ReductionRequest(params, src)).image


@pytest.mark.parametrize("gram,fast", [("target", True), ("target", False), ("zone", False)])
def test_a_cube_in_small_units_reduces_as_the_cube_in_counts(gram, fast):
    src = [Source(0.9, 60.0, 5e-3)]
    ref = _images(1.0, "relative", gram, fast, src=src)
    ok = np.isfinite(ref)
    sd = float(np.nanstd(ref))
    for s in (1e-4, 1e-7, 1e-10):
        img = _images(s, "relative", gram, fast, src=src)
        assert np.array_equal(np.isfinite(img), ok)
        assert np.nanmax(np.abs(img[ok] / s - ref[ok])) < 1e-3 * sd
    # the absolute cut, as before 2026-11-25: one mode left at 1e-6 whatever k asks, none at 1e-7
    old = {s: _images(s, "absolute", gram, fast, src=src) / s for s in (1e-6, 1e-7)}
    assert np.allclose(_images(1e-6, "absolute", gram, fast, k=3, src=src) / 1e-6, old[1e-6], rtol=1e-4,
                       atol=1e-4 * sd, equal_nan=True)
    assert np.nanstd(old[1e-7] - ref) > 0.5 * sd


def test_frames_in_counts_reduce_as_before():
    """The floor keeps every mode the absolute cut kept on these frames, so their image is unchanged."""
    for gram, fast in (("target", True), ("target", False), ("zone", False)):
        a = _images(1.0, "relative", gram, fast)
        b = _images(1.0, "absolute", gram, fast)
        assert np.array_equal(a, b, equal_nan=True)


def test_a_run_records_its_cut_and_an_old_run_keeps_the_absolute_one(tmp_path):
    from conftest import build_synthetic_run
    from klip_tpe import CalibrationConfig, Runner, ValidationConfig
    assert RunConfig().klip_eig_cut == "relative"
    d = RunConfig(ann_edges=[8, 20]).to_dict()
    assert RunConfig.from_dict(d).klip_eig_cut == "relative"
    d.pop("klip_eig_cut")
    assert RunConfig.from_dict(d).klip_eig_cut == "absolute"
    assert _eig_tol({}) is None and _eig_tol({"klip_eig_cut": "absolute"}) == K.EIG_TOL_IDL
    with pytest.raises(ValueError):
        _eig_tol({"klip_eig_cut": "exact"})
    seen = []
    ds = synthetic_klip_dataset(nframes=12, size=64, pa_span=60, seed=3)
    for cut in ("relative", "absolute"):
        _, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 20], n_iter=[4], n_init=2, param_verify=False,
                                                       calibration=CalibrationConfig(recal_budget=0),
                                                       validation=ValidationConfig(n_top=1, n_valid=1),
                                                       klip_eig_cut=cut)
        red = PartitionedReducer({pid: KLIPReducer(Dataset(ds.cube, ds.angles, ds.tags, texp=ds.texp, name=pid), PX,
                                                   LAM, 8.2, injection_model=GaussianPSF(4.0, star_flux=1e4),
                                                   outrad_cap=30) for pid in ("n1", "n2")})
        log = []
        Runner(red, space, obj, samp, cfg, str(tmp_path / cut), log=log.append)
        seen.append({r.defaults["klip_eig_cut"] for r in red.reducers.values()})
        assert any("absolute 1e-6" in ln for ln in log) == (cut == "absolute")
    assert seen == [{"relative"}, {"absolute"}]


def test_the_command_line_and_the_idl_setup_take_the_cut():
    from klip_tpe import cli
    p = cli._build_parser()
    assert cli._protocol_config(p.parse_args(["near", "--klip-eig-cut", "absolute"]))["klip_eig_cut"] == "absolute"
    assert "klip_eig_cut" not in cli._protocol_config(p.parse_args(["near"]))


@pytest.mark.parametrize("compact", [0, 4])
def test_the_card_path_reduces_small_units_as_counts(compact):
    """The card's chain on the host (the NumPy backend): its weights keep each problem's modes above the floor."""
    from klip_tpe.gpu_reducer import GPUKLIPReducer
    ds = synthetic_klip_dataset(nframes=30, size=64, pa_span=60, seed=3)
    out = {}
    for s in (1.0, 1e-7):
        d = Dataset((ds.cube * s).astype(np.float32), ds.angles, ds.tags, texp=ds.texp)
        cpu = KLIPReducer(d, PX, LAM, 8.2, injection_model=GaussianPSF(4.0, star_flux=1e4 * s), outrad_cap=30)
        for cut in ("relative", "absolute"):
            set_eig_cut(cpu, cut)
            g = GPUKLIPReducer.from_cpu(cpu, compact=compact)
            g.backend_name = "numpy"
            params = dict(PARAMS, fast=False, anglemax=40.0)
            out[s, cut] = g.reduce(ReductionRequest(params, [Source(0.9, 60.0, 5e-3)])).image / s
    ref = out[1.0, "relative"]
    ok = np.isfinite(ref)
    sd = float(np.nanstd(ref))
    assert np.array_equal(out[1.0, "absolute"], ref, equal_nan=True)
    assert np.nanmax(np.abs(out[1e-7, "relative"][ok] - ref[ok])) < 1e-3 * sd
    assert np.nanstd(out[1e-7, "absolute"] - ref) > 0.5 * sd


def test_the_cut_travels_with_each_request(tmp_path):
    """Workers forked before the runner was built keep the reducers' old defaults, so the cut goes with the request."""
    from conftest import build_synthetic_run
    from klip_tpe import CalibrationConfig, Runner, ValidationConfig
    for cut in ("relative", "absolute"):
        red, space, obj, samp, cfg = build_synthetic_run(ann_edges=[8, 20], n_iter=[4], n_init=2, param_verify=False,
                                                         calibration=CalibrationConfig(recal_budget=0),
                                                         validation=ValidationConfig(n_top=1, n_valid=1),
                                                         klip_eig_cut=cut)
        r = Runner(red, space, obj, samp, cfg, str(tmp_path / cut), log=lambda s: None)
        seen = []
        orig = r.reducer.reduce_config

        def spy(cfg2, *a, **kw):
            seen.append((cfg2.params.get("klip_eig_cut"), {d.get("klip_eig_cut") for d in cfg2.per_partition.values()}))
            return orig(cfg2, *a, **kw)
        r.reducer.reduce_config = spy
        r._reduce(space.decode(space.default_vector()), None, tag="t")
        assert seen and seen[0][0] == cut and seen[0][1] <= {cut}
