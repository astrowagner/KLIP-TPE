"""Angles that cross a branch cut come onto one branch where a dataset is made."""
import numpy as np
import pytest

from klip_tpe.injection import GaussianPSF
from klip_tpe.klip import reference_mask, unwrap_angles
from klip_tpe.metrics import Source
from klip_tpe.reducer import Dataset, KLIPReducer, ReductionRequest
from klip_tpe.synthetic import synthetic_klip_dataset

PX = 0.05
LAM = 3.89 * PX / 206265 * 8.2


def _wrap180(a):
    """Return angles as stored in [-180, 180)."""
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def test_a_sequence_under_half_a_turn_comes_back_as_it_was():
    for a in (np.linspace(-40.0, 55.0, 30), np.array([10.0, -160.0]), np.linspace(300.0, 359.0, 12),
              np.array([5.0]), np.array([])):
        out = unwrap_angles(a)
        assert np.array_equal(out, np.asarray(a, float))


def test_a_sequence_through_the_cut_is_continuous_and_equal_modulo_a_turn():
    true = np.linspace(150.0, 210.0, 61)                     # 60 degrees through +-180
    for stored in (_wrap180(true), true % 360.0, _wrap180(true)[::-1], _wrap180(true)[np.random.default_rng(1).permutation(61)]):
        out = unwrap_angles(stored)
        assert np.ptp(out) == pytest.approx(60.0)
        assert np.allclose((out - stored) % 360.0, 0.0) or np.allclose((out - stored + 180.0) % 360.0 - 180.0, 0.0)
        assert out[0] == stored[0]                           # the first frame keeps its value
    through0 = np.linspace(-20.0, 30.0, 26) % 360.0          # stored in [0, 360): 340..359, 0..30
    out = unwrap_angles(through0)
    assert np.ptp(out) == pytest.approx(50.0) and out[0] == through0[0]
    # every pair differs as it does on the circle
    d = np.abs(out[:, None] - out[None, :])
    c = np.abs((through0[:, None] - through0[None, :] + 180.0) % 360.0 - 180.0)
    assert np.allclose(d, c)


def test_two_rolls_take_the_short_way_and_a_nan_stays():
    assert np.ptp(unwrap_angles([359.0, 9.0, 359.0, 9.0])) == pytest.approx(10.0)
    assert np.ptp(unwrap_angles([0.0, 200.0])) == pytest.approx(160.0)
    out = unwrap_angles([170.0, np.nan, -175.0])
    assert np.isnan(out[1]) and out[0] == 170.0 and out[2] == pytest.approx(185.0)
    again = unwrap_angles(out)                                # a second pass changes nothing
    assert np.array_equal(again, out, equal_nan=True)


def test_a_rotation_past_half_a_turn_that_never_wrapped_is_left_alone():
    a = np.linspace(-100.0, 100.0, 81)
    assert np.allclose(unwrap_angles(a), a) and unwrap_angles(a)[0] == a[0]


def test_a_dataset_takes_wrapped_angles_onto_one_branch():
    ds = synthetic_klip_dataset(nframes=40, size=64, pa_span=60, seed=3)
    true = np.asarray(ds.angles, float) + 180.0              # 150 .. 210
    meta = {"origin": "test"}
    wrapped = Dataset(ds.cube, _wrap180(true), ds.tags, texp=ds.texp, name="wrapped", meta=meta)
    plain = Dataset(ds.cube, true, ds.tags, texp=ds.texp, name="plain", meta=meta)
    assert plain.angles is true and "angles_unwrapped" not in plain.meta
    assert wrapped.meta["angles_unwrapped"] == 20 and meta == {"origin": "test"}
    assert np.ptp(wrapped.angles) == pytest.approx(60.0)
    # the references of a frame beside the cut: the same as on the continuous sequence
    for t in (0, 19, 20, 39):
        assert np.array_equal(reference_mask(np.asarray(wrapped.angles), t, 5.0, 40.0),
                              reference_mask(true, t, 5.0, 40.0))
    stored = _wrap180(true)                                   # as stored, half the frames count as references 1 deg away
    assert reference_mask(stored, 19, 5.0, 40.0).sum() < reference_mask(true, 19, 5.0, 40.0).sum()


def test_a_wrapped_sequence_reduces_as_the_continuous_one():
    ds = synthetic_klip_dataset(nframes=40, size=64, pa_span=60, seed=3)
    true = np.asarray(ds.angles, float) + 180.0
    model = GaussianPSF(4.0, star_flux=1e4)
    params = dict(k_klip=4, inrad=8, outrad=26, n_ang=1, angsep=1.0, anglemax=25.0, fast=False, bin=1, filter=0)
    src = [Source(0.9, 60.0, 4e-3)]
    out = {}
    for name, ang in (("plain", true), ("wrapped", _wrap180(true))):
        d = Dataset(ds.cube, ang, ds.tags, texp=ds.texp, name=name)
        red = KLIPReducer(d, PX, LAM, 8.2, injection_model=model, outrad_cap=30)
        out[name] = red.reduce(ReductionRequest(dict(params), src)).image
    both = np.isfinite(out["plain"]) & np.isfinite(out["wrapped"])
    assert both.sum() > 1000 and np.array_equal(np.isfinite(out["plain"]), np.isfinite(out["wrapped"]))
    assert np.nanmax(np.abs(out["plain"] - out["wrapped"])[both]) < 1e-4 * np.nanstd(out["plain"][both])


def test_the_run_log_names_the_partitions_whose_angles_moved():
    from types import SimpleNamespace

    from klip_tpe.reducer import PartitionedReducer
    from klip_tpe.runner import Runner
    ds = synthetic_klip_dataset(nframes=40, size=64, pa_span=60, seed=3)
    true = np.asarray(ds.angles, float) + 180.0
    model = GaussianPSF(4.0, star_flux=1e4)
    reds = {name: KLIPReducer(Dataset(ds.cube, ang, ds.tags, texp=ds.texp, name=name), PX, LAM, 8.2,
                              injection_model=model, outrad_cap=30)
            for name, ang in (("n1", _wrap180(true)), ("n2", true))}
    got = []
    lines = Runner._note_unwrapped_angles(SimpleNamespace(reducer=PartitionedReducer(reds), log=got.append))
    assert got == lines and len(lines) == 1
    assert "20 of 40 frames of partition n1" in lines[0] and "150.00 to 210.00 deg" in lines[0]
    one = Runner._note_unwrapped_angles(SimpleNamespace(reducer=reds["n2"], log=got.append))
    assert one == [] and len(got) == 1
