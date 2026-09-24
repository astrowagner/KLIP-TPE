"""Both ways of searching the reference library, and the check that each is doing something.

* pyKLIP: its own selection, searched -- ``mode`` (which pools) and ``maxnumbasis`` (how
  many of the most-correlated frames each target keeps, per sector).
* built-in engine: ``nkeep_altroll`` / ``nkeep_psfref`` (``test_searched_library.py``).

and :func:`klip_tpe.liveness.check_live_dimensions`, which would have stopped MIRI run v6
before its first evaluation: it searched two counts the pyKLIP backend ignored.
"""
from __future__ import annotations

import numpy as np
import pytest

from klip_tpe.feasibility import PyKLIPLibraryGuard
from klip_tpe.liveness import check_live_dimensions, dead_dimensions
from klip_tpe.reducer import Dataset, PartitionedReducer, ReductionRequest
from klip_tpe.space import Param, SearchSpace, kgrid


def _two_rolls(n_per_roll=6, n_ref=10, n=41, seed=0):
    """Two rolls of a star with a speckle pattern that changes a little, plus a reference
    star with a similar one -- enough structure that the choice of references matters."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2)
    halo = 200.0 * np.exp(-r / 4.0)
    spk = rng.normal(0, 1, (n, n))
    ang = np.concatenate([np.full(n_per_roll, 108.0), np.full(n_per_roll, 117.4)])
    cube = np.stack([halo * (1 + 0.05 * spk + 0.02 * rng.normal(0, 1, (n, n))) + rng.normal(0, 1, (n, n))
                     for _ in ang]).astype(np.float32)
    ref = np.stack([halo * (1 + 0.05 * spk + 0.04 * rng.normal(0, 1, (n, n))) + rng.normal(0, 1, (n, n))
                    for _ in range(n_ref)]).astype(np.float32)
    ds = Dataset(cube=cube, angles=ang, name="sci")
    ds.ref_cube = ref
    return ds, np.round(ang, 1).astype(str)


# ------------------------------------------------------------------ the pyKLIP guard

def _pk_space():
    return SearchSpace([Param("bin", 1, 10, "int", default=10),
                        Param("k_klip", 1, 40, "int", grid=kgrid(40), default=6),
                        Param("mode", 0, 2, "categorical", choices=["ADI", "RDI", "ADI+RDI"], default="ADI+RDI"),
                        Param("maxnumbasis", 1, 131, "int", default=6)])


@pytest.mark.parametrize("mode,bin_,k,mnb,want_k,want_mnb", [
    ("ADI", 10, 20, 50, 5, 5),          # 41 frames / 10 -> 5 per roll: k and the library both capped
    ("RDI", 10, 20, 3, 20, 20),         # 90 references: k kept, maxnumbasis raised to k
    ("RDI", 10, 20, 130, 20, 90),       # ... and capped at the 90 there are
    ("ADI+RDI", 1, 30, 131, 30, 131),   # everything, unbinned
])
def test_pyklip_guard_keeps_k_and_maxnumbasis_live(mode, bin_, k, mnb, want_k, want_mnb):
    """pyKLIP clips numbasis to the frames it kept and keeps at most maxnumbasis, so outside
    k_klip <= maxnumbasis <= pool one of the two is inert."""
    sp = _pk_space()
    g = PyKLIPLibraryGuard(n_alt=41, n_ref=90)
    x = np.array([bin_, k, ["ADI", "RDI", "ADI+RDI"].index(mode), mnb], float)
    y = g(x, sp)
    assert (y[1], y[3]) == (want_k, want_mnb)
    assert y[1] <= y[3] <= g.pool(mode, bin_)


def test_take_everything_survives_the_grid_snap():
    """Run v7's ablation asked for "all" at bin 9 and reduced with 90 of the 95 frames: the
    guard clipped maxnumbasis to the pool (95) and the sanitize after it snapped that to the
    grid neighbour below.  Clipping to the first grid point at or above the pool keeps
    "everything" reachable -- pyKLIP keeps min(maxnumbasis, pool) -- and leaves interior
    values (a winner's 25, 70) alone."""
    grid = sorted({float(v) for v in kgrid(131)} | {131.0})
    sp = SearchSpace([Param("bin", 1, 10, "int", default=10),
                      Param("k_klip", 1, 40, "int", grid=kgrid(40), default=6),
                      Param("mode", 0, 2, "categorical", choices=["ADI", "RDI", "ADI+RDI"], default="ADI+RDI"),
                      Param("maxnumbasis", 1, 131, "int", grid=grid, default=6)])
    g = PyKLIPLibraryGuard(n_alt=41, n_ref=90)

    def proj(bin_, mode, mnb, k=20):
        x = np.array([bin_, k, ["ADI", "RDI", "ADI+RDI"].index(mode), mnb], float)
        return sp.sanitize(g(sp.sanitize(x), sp))
    y = proj(9, "ADI+RDI", 131)
    assert y[3] >= g.pool("ADI+RDI", 9) == 95 and y[3] == 100
    assert proj(9, "RDI", 131)[3] == 90 and proj(9, "ADI", 131)[3] == 5 == proj(9, "ADI", 131)[1]
    assert proj(1, "ADI+RDI", 131)[3] == 131
    assert proj(9, "ADI+RDI", 70)[3] == 70 and proj(4, "ADI+RDI", 25)[3] == 25


# ------------------------------------------------------------------ pyKLIP's own library, searched

def _pk_reducer(ds):
    pytest.importorskip("pyklip")
    from klip_tpe.backends.pyklip import PyKLIPReducer
    from klip_tpe.injection import GaussianPSF
    return PyKLIPReducer(ds, pxscale=0.11, lam_m=11.3e-6, diam_m=6.5, fwhm_px=3.3,
                         injection_model=GaussianPSF(3.3, star_flux=1e4), outrad_cap=18,
                         defaults={"mode": "ADI+RDI"})


def test_native_library_brings_mode_and_maxnumbasis():
    ds, part = _two_rolls()
    r = _pk_reducer(ds)
    assert r.reference_params() == []                     # nothing until asked for
    r.set_native_library(partition=part)
    ps = {p.name: p for p in r.reference_params()}
    assert set(ps) == {"mode", "maxnumbasis"}
    assert list(ps["mode"].choices) == ["ADI", "RDI", "ADI+RDI"] and ps["mode"].default == "ADI+RDI"
    assert (ps["maxnumbasis"].lo, ps["maxnumbasis"].hi) == (1, 6 + 10)   # other roll + references
    assert 16.0 in ps["maxnumbasis"].grid                # "take all" is reachable
    from klip_tpe.instruments import generic
    red = PartitionedReducer({"sci": r})
    sp = generic.make_space(red, k_klip_max=10, search_angles=False, opt_framesel=False, selection=None)
    k_def = next(p.default for p in sp.params if p.name == "k_klip")
    assert next(p.default for p in sp.params if p.name == "maxnumbasis") == k_def
    guard = generic.make_guard(red, k_max=10)
    parts = [c.cell_contents for c in (getattr(guard, "__closure__", None) or [])]
    flat = [g for c in parts for g in (c if isinstance(c, (list, tuple)) else [c])]
    assert isinstance(guard, PyKLIPLibraryGuard) or any(isinstance(g, PyKLIPLibraryGuard) for g in flat)


def test_mode_and_maxnumbasis_change_a_pyklip_reduction():
    """The property the searched nkeep_* counts never had on this backend.

    Not every pair has to differ: ADI+RDI keeps the ``maxnumbasis`` most-correlated frames of
    the union, so when its best 3 all come from the other roll it IS ADI -- as here, where
    the science frames correlate better with each other than with the reference star.  What
    must hold is that each dimension moves the reduction somewhere."""
    ds, part = _two_rolls()
    r = _pk_reducer(ds)
    base = dict(k_klip=3, inrad=3.0, outrad=15.0, bin=1, n_ang=1, filter=0)
    imgs = {}
    for label, (mode, mnb) in {"ardi3": ("ADI+RDI", 3), "ardi12": ("ADI+RDI", 12),
                               "rdi3": ("RDI", 3), "adi3": ("ADI", 3)}.items():
        res = r.reduce(ReductionRequest(params=dict(base, mode=mode, maxnumbasis=mnb)))
        assert res.meta["maxnumbasis"] == mnb and res.meta["mode"] == mode
        imgs[label] = np.nan_to_num(np.asarray(res.image, float))
    same = lambda a, b: np.array_equal(imgs[a], imgs[b])
    assert not same("ardi3", "ardi12"), "maxnumbasis changed nothing"
    assert not same("adi3", "rdi3") and not same("ardi3", "rdi3"), "mode changed nothing"
    assert same("adi3", "ardi3")          # the documented coincidence: its best 3 are all other-roll


# ------------------------------------------------------------------ the liveness pre-flight

def _runner(red, space, tmp_path):
    from klip_tpe import CalibrationConfig, RunConfig, Runner, ValidationConfig
    from klip_tpe.metrics import MawetPeakSNR, Objective
    from klip_tpe.positions import PositionSampler
    obj = Objective(MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm), clean_subtract=True)
    samp = PositionSampler(fwhm_as=red.fwhm * red.pxscale)
    cfg = RunConfig(ann_edges=[4, 15], n_iter=1, n_init=1, seed=1,
                    validation=ValidationConfig(n_top=1, n_valid=1),
                    calibration=CalibrationConfig(forced=[1e-3]), save_fits=False)
    return Runner(red, space, obj, samp, cfg, str(tmp_path / "live"), log=lambda s: None)


def test_liveness_flags_the_v6_bug(tmp_path):
    """nkeep_* on a pyKLIP space -- what run v6 searched -- is dead; mode and maxnumbasis
    and the ordinary dimensions are live."""
    ds, part = _two_rolls()
    r = _pk_reducer(ds)
    r.set_native_library(partition=part)
    red = PartitionedReducer({"sci": r})
    from klip_tpe.instruments import generic
    sp = generic.make_space(red, k_klip_max=10, search_angles=False, opt_framesel=False, selection=None)
    sp.add(Param("nkeep_altroll", 0, 6, "int", default=6))        # what set_reference_library used to add
    sp.project = generic.make_guard(red, k_max=10)
    logs = []
    st = check_live_dimensions(_runner(red, sp, tmp_path), log=logs.append)
    assert dead_dimensions(st) == ["nkeep_altroll"], st
    assert st["mode"]["live"] and st["maxnumbasis"]["live"] and st["k_klip"]["live"]
    assert any("nkeep_altroll" in l and "DEAD" in l for l in logs)


def test_liveness_passes_the_built_in_library(tmp_path):
    from klip_tpe.instruments import generic
    ds, part = _two_rolls()
    yy, xx = np.mgrid[0:21, 0:21]
    psf = np.exp(-((xx - 10) ** 2 + (yy - 10) ** 2) / (2 * 1.4 ** 2))
    red = generic.make_reducer({"sci": ds}, pxscale=0.11, lam_m=11.3e-6, diam_m=6.5, use_rdi=True,
                               psf=psf, star_flux=1e4, log=lambda s: None)
    next(iter(red.reducers.values())).set_reference_library(partition=part, n_min_ref=2)
    sp = generic.make_space(red, k_klip_max=10, search_angles=False, opt_framesel=False, selection=None)
    sp.project = generic.make_guard(red, k_max=10)
    st = check_live_dimensions(_runner(red, sp, tmp_path), log=lambda s: None)
    assert dead_dimensions(st) == [], st
    assert st["nkeep_altroll"]["live"] and st["nkeep_psfref"]["live"]


# ------------------------------------------------------------------ the head-to-head

def _la():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import library_ablation as la
    return la


def test_versus_pairs_winners_only_on_identical_injections():
    """Two runs' winners are compared draw for draw, and only when the injections match:
    the same positions AND the same contrast.  Each run calibrates its own contrast, and S/N
    scales with the injected flux -- pairing two runs' own contrasts would put their
    calibrations into the ratio."""
    la = _la()
    draws = [[[1.2, 30.0], [1.5, 210.0]]] * 4

    def res(backend, scores, planet, dr=draws, contrast=2.4e-4, annulus=1):
        return {"backend": backend, "run_dir": backend,
                "annuli": [{"annulus": annulus, "draws": dr, "contrast": contrast,
                            "configs": {"winner": {"raw": scores, "search": scores, "planet_snr": planet}}}]}
    logs = []
    la.versus(res("klip", [8.0, 8.2, 7.9, 8.1], 12.0), res("pyklip", [6.0, 6.1, 5.9, 6.2], 9.0), logs.append)
    assert any("x1.3" in l and "P(first better) 1.000" in l for l in logs), logs
    assert any("planet: 12.00 vs 9.00" in l for l in logs)
    for other, why in ((res("pyklip", [6.0] * 4, None, [[[1.2, 31.0], [1.5, 211.0]]] * 4),
                        "the injections differ -- not pairing"),
                       (res("pyklip", [6.0] * 3, None, draws[:3]), "the injections differ -- not pairing"),
                       (res("pyklip", [6.0] * 4, None, contrast=3.25e-4), "--contrast-from"),
                       (res("pyklip", [6.0] * 4, None, annulus=2), "not in the other")):
        logs = []
        la.versus(res("klip", [8.0] * 4, None), other, logs.append)
        assert any(why in l for l in logs), (why, logs)
        assert not any(" x" in l for l in logs), logs              # and no ratio was reported


def test_ablation_says_why_when_a_run_is_not_finished(tmp_path):
    """Run a minute after its searches started -- as happened on v7 -- the ablation raised a
    bare FileNotFoundError.  It now says what is missing and where the progress is, before
    any reduction, and checks the --versus file up front rather than crashing after hours."""
    la = _la()
    run = tmp_path / "v7_pyklip"

    def main(*extra):
        with pytest.raises(SystemExit) as e:
            la.main(["--data", str(tmp_path), "--run-dir", str(run), *extra])
        return str(e.value)
    assert "no such directory" in main()
    run.mkdir()
    (run / "run.log").write_text("[10:44:56] 63 calints\n")
    msg = main()
    assert "has not started its search yet" in msg and "run.log" in msg, msg
    (run / "run_setup.json").write_text("{}")
    assert "has not finished" in main()
    (run / "final_results.json").write_text('{"annuli": []}')
    msg = main("--contrast-from", str(tmp_path / "v7_klip"))
    assert msg.startswith("--contrast-from") and "no such directory" in msg, msg
    msg = main("--versus", str(tmp_path / "v7_klip_ablation.json"))
    assert msg.startswith("--versus") and "make that one first" in msg, msg


def test_ablation_follows_the_runs_own_engine_and_dimensions(tmp_path):
    """Run on a klip run without --backend, the ablation used to rebuild it on pyKLIP and
    carry the winners across by name: nkeep_* dropped silently, mode / maxnumbasis at their
    defaults, and a rebuild check that only compared the names both had -- so it passed.
    Now the engine defaults to the run's own, and on the same engine the searched dimensions
    must match exactly."""
    import json
    la = _la()
    assert la._run_backend({"reducer": {"partitions": {"sci": {"backend": "pyklip"}}}}) == "pyklip"
    assert la._run_backend({"reducer": {"partitions": {"sci": {"name": "spaceklip_sci"}}}}) == "klip"
    assert la._run_backend({}) is None
    setup = {"space": {"params": [{"name": n, "role": "reduction"}
                                  for n in ("bin", "n_ang", "filter", "k_klip", "nkeep_altroll", "nkeep_psfref")]}}
    dims = la._run_dims(setup)
    pk = ["bin", "n_ang", "filter", "k_klip", "mode", "maxnumbasis"]
    assert la._space_check(dims, dims, same_engine=True) == ([], [])
    probs, notes = la._space_check(dims, pk, same_engine=True)
    assert probs and "nkeep_altroll" in probs[0] and "maxnumbasis" in probs[0] and not notes
    probs, notes = la._space_check(dims, pk, same_engine=False)      # a deliberate cross-engine test
    assert not probs and notes and "cross-engine" in notes[0]

    # --contrast-from: the other run's calibrated contrasts, only for the same annuli
    def run(d, edges, contrasts):
        d.mkdir()
        (d / "run_setup.json").write_text(json.dumps({"config": {"ann_edges": edges}}))
        (d / "final_results.json").write_text(json.dumps({"annuli": [{"contrast": c} for c in contrasts]}))
        return str(d)
    edges = [6.7, 20.0, 26.9, 36.0]
    other = run(tmp_path / "pyklip", edges, [2.4e-4, 1.1e-4, 8.0e-5])
    assert la._contrasts_from(other, {"ann_edges": edges}) == [2.4e-4, 1.1e-4, 8.0e-5]
    with pytest.raises(SystemExit, match="annuli"):
        la._contrasts_from(other, {"ann_edges": [6.7, 20.0, 36.0]})
    short = run(tmp_path / "unfinished", edges, [2.4e-4])
    with pytest.raises(SystemExit, match="1 annuli finished, 3 needed"):
        la._contrasts_from(short, {"ann_edges": edges})
