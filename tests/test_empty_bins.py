"""The bins the binning drops: those without data, never a bin whose pixels sum to zero.

From the WISPIT 2 ALES session, 2026-11-10.  One draw of 3000 failed with "similarity cube
(32, 62, 62) does not match the binned cube (31, 62, 62)".  The test for an empty bin was the
bin's ``nansum``, and a high-passed bin sums to nearly zero by design, so in float32 a bin with
data sums to exactly zero now and then.  The draw's sources moved one bin's sum onto zero, and
the injected cube lost a bin that the clean cube, the reference library's ranking cube, kept.
What must hold:

* A bin is dropped when none of its pixels is neither NaN nor zero: all-NaN and all-zero bins
  go, a bin whose pixels sum to exactly zero stays.  The fused path, the whole-cube path and
  the GPU reducer agree.
* The clean and the injected cubes keep the same bins when the data have them.
* A searched library ranks on the clean frames at the injected cube's own bins, keyed by them,
  so it never fails when the two cubes keep different bins (frames of zeros, which only the
  injected sources fill).
"""
from __future__ import annotations

import numpy as np
import pytest

from klip_tpe.injection import GaussianPSF
from klip_tpe.klip import bin_frames, bins_with_data
from klip_tpe.metrics import Source
from klip_tpe.reducer import KLIPReducer, ReductionRequest

SOURCES = [Source(0.4, 30.0, 5e-3), Source(0.6, 200.0, 5e-3)]


def _checkerboard(n: int) -> np.ndarray:
    """+1 and -1 in turn: a frame with data whose float32 sum is exactly zero in any order."""
    yy, xx = np.mgrid[0:n, 0:n]
    return np.where((xx + yy) % 2 == 0, 1.0, -1.0).astype(np.float32)


def test_a_bin_with_data_that_sums_to_zero_is_kept():
    cube = np.ones((8, 4, 4), np.float32)
    cube[2:4] = 0.0                                  # all zero: dropped
    cube[4:6] = np.nan                               # all NaN: dropped
    cube[6:8] = _checkerboard(4)                     # sums to exactly zero: kept
    assert np.nansum(cube[6]) == 0.0
    bc, ba, grp, keep = bin_frames(cube, np.arange(8.0), 2, 100.0, return_groups=True)
    assert keep.tolist() == [True, False, False, True]
    assert ba.tolist() == [0.5, 6.5] and np.array_equal(bc[1], _checkerboard(4))
    # NaN and zeros only: no data; an inf is data, as its sum kept it before
    f = np.zeros((3, 4, 4), np.float32)
    f[0, 0, 0] = np.nan
    f[1, 1, 1] = np.inf
    f[2, 2, 2] = -1e-30
    assert bins_with_data(f).tolist() == [False, True, True]
    assert bins_with_data(np.zeros((0, 4, 4))).shape == (0,)
    # the bins of another cube in place of the test
    bc2, ba2, _, keep2 = bin_frames(cube, np.arange(8.0), 2, 100.0, return_groups=True,
                                    bkeep=[True, True, False, True])
    assert keep2.tolist() == [True, True, False, True] and ba2.tolist() == [0.5, 2.5, 6.5]
    with pytest.raises(ValueError, match="bkeep has 3 bins"):
        bin_frames(cube, np.arange(8.0), 2, 100.0, bkeep=[True, True, True])


def _library_reducer(zero_frames=(), board_frames=()):
    from klip_tpe.synthetic import synthetic_klip_dataset
    ds = synthetic_klip_dataset(nframes=40, size=48, pa_span=20, seed=3)
    rng = np.random.default_rng(4)
    ds.ref_cube = np.stack([ds.cube[i % 40] * 1.01 + rng.normal(0, 0.05, (48, 48))
                            for i in range(12)]).astype(np.float32)
    ds.cube = np.asarray(ds.cube, np.float32)
    for j in zero_frames:
        ds.cube[j] = 0.0
    for j in board_frames:
        ds.cube[j] = _checkerboard(48)
    red = KLIPReducer(ds, pxscale=0.05, lam_m=4.4e-6, diam_m=6.5, injection_model=GaussianPSF(3.0, star_flux=1e4),
                      fwhm_px=3.0, outrad_cap=20)
    red.set_reference_library(partition=np.array(["rollA"] * 20 + ["rollB"] * 20), min_keep={"psfref": 1},
                              n_min_ref=2)
    return red


#: no high-pass and no binning, so a frame's bin is the frame as it is
PARAMS = dict(k_klip=3, bin=1, filter=0, n_ang=1, inrad=4, outrad=16, angsep=0.0, anglemax=360.0,
              corr_thresh=0.0, noise_max=3.0, coronoise_max=3.0, use_rdi=True, n_min_ref=2,
              nkeep_altroll=6, nkeep_psfref=5)


@pytest.mark.parametrize("fused", [True, False])
def test_the_clean_and_injected_cubes_keep_a_bin_that_sums_to_zero(fused):
    red = _library_reducer(board_frames=(7,))
    p = dict(PARAMS, prep_fused=fused)
    clean = red.prepare(ReductionRequest(p))
    inj = red.prepare(ReductionRequest(p, SOURCES))
    assert clean.bkeep.all() and inj.bkeep.all() and clean.bcube.shape == inj.bcube.shape == (40, 48, 48)
    assert np.nansum(clean.bcube[7]) == 0.0 and np.nansum(inj.bcube[7]) > 0
    # the injected reduction with the searched library: the draw that failed
    red._reflib_cache.clear()
    r = red.reduce(ReductionRequest(p, SOURCES))
    assert r.meta["n_binned"] == 40 and np.isfinite(r.image).any()
    # the clean reduction's library is the same one: the same bins
    red.reduce(ReductionRequest(p))
    assert len(red._reflib_cache) == 1


@pytest.mark.parametrize("fused", [True, False])
def test_the_library_ranks_on_the_clean_frames_at_the_injected_bins(fused):
    # frames of zeros: the clean cube drops them, the injected sources fill them
    red = _library_reducer(zero_frames=(10, 11))
    p = dict(PARAMS, prep_fused=fused)
    clean = red.prepare(ReductionRequest(p))
    inj = red.prepare(ReductionRequest(p, SOURCES))
    assert clean.bkeep.sum() == 38 and inj.bkeep.all()
    sim = red._clean_bins(inj.p, inj.keep, 1, 0, inj.dth, inj.hp_nan, inj.hp_binned, 1, bkeep=inj.bkeep)
    assert sim.shape == inj.bcube.shape and not sim[10:12].any()
    assert np.array_equal(sim[clean.bkeep], clean.bcube)
    red._reflib_cache.clear()
    r = red.reduce(ReductionRequest(p, SOURCES))
    assert r.meta["n_binned"] == 40 and np.isfinite(r.image).any()
    # the clean reduction ranks its own 38 bins: a library of its own
    red.reduce(ReductionRequest(p))
    assert len(red._reflib_cache) == 2


def test_the_gpu_reducer_keeps_the_same_bins():
    from klip_tpe.gpu_reducer import GPUKLIPReducer
    red = _library_reducer(zero_frames=(10, 11), board_frames=(7,))
    red._reflib_spec = None                           # the GPU reducer takes no searched library
    g = GPUKLIPReducer.from_cpu(red)
    g.backend_name = "numpy"
    p = {k: v for k, v in PARAMS.items() if k not in ("use_rdi", "nkeep_altroll", "nkeep_psfref")}
    req = ReductionRequest(p, None, threads=1)
    assert g.supports(req)
    rc, rg = red.reduce(req), g.reduce(req)
    assert rc.meta["n_binned"] == rg.meta["n_binned"] == 38
