#!/usr/bin/env python
"""What the searched reference library buys, measured on the run that chose it.

A ``run_miri.py`` search picks ``nkeep_altroll`` (correlation-ranked frames kept from the
other roll) and ``nkeep_psfref`` (from the reference star) per annulus, alongside the usual
KLIP parameters.  This script asks how much that is worth, by rebuilding the finished run's
exact setup -- same loader, reducer, space, objective, sampler, annuli and per-annulus
contrast -- and scoring a set of configurations per annulus on COMMON injection draws: the
same source positions in every configuration.  On these data one draw scatters by ~0.84 in
S/N from the azimuth of the fakes alone, and most of that scatter is shared between
configurations, so a paired comparison resolves differences an unpaired one would bury.

Configurations, per annulus (the first four differ ONLY in the library):

  winner    the run's validated winner, tuned library and all
  all       the winner with both pools taken whole          -- ADI+RDI, "use everything"
  rdi       the winner with the reference star only, whole  -- nkeep_altroll = 0
  adi       the winner with the other roll only, whole      -- nkeep_psfref = 0
  carter    Carter et al. (2023)'s MIRI choices mapped into this space: every frame from
            both pools, no temporal binning, one azimuthal subsection, no high-pass,
            k = 6 -- reduced over the annulus' own zone
  carter_full  the same over ONE zone covering the whole crop from the 4QPM inner working
            angle out, as they reduced it ("a single annulus and a single subsection (i.e.,
            the entire image)")
  default   the run's own seeded default

Every injected reduction is scored twice: ``score_search`` (what the search maximised,
``s_inj - max(s_clean, 0)``) and ``score_raw`` (what validation reports).  HIP 65426 b's S/N
is measured in each configuration's clean image where its zone contains the planet.

usage:
  python scripts/library_ablation.py --data ~/Data/JWST/hip65426_miri \\
      --run-dir miri_HIP-65426_F1140C_v6 --n-draws 10 --out ablation_F1140C_v6.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import run_miri                                                          # noqa: E402
from klip_tpe import RunConfig, Runner, ValidationConfig, CalibrationConfig   # noqa: E402
from klip_tpe.instruments import generic                                 # noqa: E402

#: Carter et al. (2023, ApJL 951, L20), MIRI: ADI+RDI with the full 9-point SGD library,
#: "a single annulus and a single subsection (i.e., the entire image)", contrast flat
#: beyond ~2 modes (ADI) and companion fits at 6.  Table 3 F1140C: 823 +/- 11 mas,
#: 149 +/- 1 deg, dF1140C = 8.264 +/- 0.021.
CARTER = dict(bin=1, n_ang=1, filter=0, k_klip=6)
PLANET = (0.823, 149.0)
IWA_AS = 0.36                                   # FQPM1140C nominal inner working angle


def _args_for(run_setup: dict, data: str, workers, backend: str = "pyklip") -> object:
    """``run_miri.build``'s argument object, mirroring the finished run (``backend`` aside:
    only the built-in ``'klip'`` engine applies the library, so a real test of it needs that)."""
    a = dict(data=data, target="HIP-65426", filter="F1140C", partition="all", crop=40, backend=backend,
             ann=[float(v) for v in run_setup["ann_edges"]], known=[(0.826, 150.2)],
             star_flux=None, flux_density_jy=None, mode="ADI+RDI", min_throughput=0.30,
             dead_zones=True, nan_dead_zones=False, destripe=None,
             ref_target=["HIP-68245"], searched_library=(backend == "klip"), star_center=None,
             workers=workers)
    return type("A", (), a)()


def _x_from_params(space, params: dict, fallback) -> np.ndarray:
    """A vector for ``space`` from named parameters, falling back to ``fallback`` for any
    dimension the parameters do not name.  The run's winners were recorded in the space it
    searched, which on pyKLIP carried two ``nkeep_*`` dimensions that engine now refuses; by
    name, the live four map onto either space."""
    x = np.asarray(fallback, float).copy()
    for i, k in enumerate(space.names):
        if k in params and params[k] is not None:
            x[i] = float(params[k])
    return x


def _vec(runner, x, **over):
    """``x`` with named entries replaced, then sanitised and projected through the run's own
    guard by the runner itself (``runner.ia`` must already be the annulus) -- the path every
    searched vector took, so a modified configuration is feasible in the same sense."""
    space = runner.space
    x = np.asarray(x, float).copy()
    for k, v in over.items():
        if k in space.names:                  # a library count on an engine without one: ignored
            x[space.names.index(k)] = float(v)
    return runner._project(x, is_random=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-dir", required=True, help="the finished run_miri.py output directory")
    ap.add_argument("--n-draws", type=int, default=10)
    ap.add_argument("--annuli", type=int, nargs="+", default=None, help="1-based; default all")
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--workers", default="auto")
    ap.add_argument("--backend", default="pyklip", choices=["pyklip", "klip"],
                    help="engine to reduce with.  The run's own was pyKLIP, which does NOT apply "
                         "the library (the four library variants then reduce identically); "
                         "'klip' does, so it is the one that tests whether the library matters")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", default="library_ablation.json")
    a = ap.parse_args(argv)

    def log(s):
        print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)

    with open(os.path.join(a.run_dir, "run_setup.json")) as f:
        full_setup = json.load(f)
    setup = full_setup.get("config", full_setup)
    with open(os.path.join(a.run_dir, "final_results.json")) as f:
        final = json.load(f)["annuli"]

    dsets, info, red, ann, obj, samp, m, px = run_miri.build(_args_for(setup, a.data, a.workers, a.backend), log)
    space = generic.make_space(red, k_klip_max=40, max_drop=0, search_angles=False)
    space.project = generic.make_guard(red, k_max=40)
    log(f"space: {space.names}")

    # A comparison against a run is only as good as the rebuild of that run.  Refuse to
    # spend hours of reductions on a setup that differs from the one the winners came from.
    problems = []
    want_fpa = [tuple(map(float, v)) for v in full_setup.get("sampler", {}).get("forbidden_pa", [])]
    got_fpa = [tuple(map(float, v)) for v in (getattr(samp, "forbidden_pa", None) or [])]
    if want_fpa and not np.allclose(np.array(sorted(want_fpa)), np.array(sorted(got_fpa)), atol=0.05):
        problems.append(f"forbidden sectors {got_fpa} != run's {want_fpa}")
    want_mask = full_setup.get("objective", {}).get("metric", {}).get("pixel_mask_px")
    pm = getattr(obj.metric, "pixel_mask", None)
    got_mask = None if pm is None else int(np.count_nonzero(pm))
    if want_mask is not None and got_mask != int(want_mask):
        problems.append(f"dead-zone mask {got_mask} px != run's {want_mask}")
    for fr in final:
        want = {k: v for k, v in fr["winner_config"]["params"].items() if k in space.names}
        xw_ = _x_from_params(space, fr["winner_config"]["params"], space.default_vector())
        got = {k: v for k, v in space.decode(xw_).params.items() if k in space.names}
        if any(abs(float(got[k]) - float(want[k])) > 1e-9 for k in want):
            problems.append(f"annulus {fr['annulus'] + 1}: winner decodes to {got}, run had {want}")
    if problems:
        for p in problems:
            log(f"MISMATCH: {p}")
        raise SystemExit("the rebuilt setup does not reproduce the run; not comparing against it")
    log("rebuild matches the run: forbidden sectors, dead-zone mask and every winner's parameters")

    cfg = RunConfig(ann_edges=[float(v) for v in ann], n_iter=1, n_init=1, seed=int(setup.get("seed", 21)),
                    validation=ValidationConfig(n_top=1, n_valid=1),
                    calibration=CalibrationConfig(forced=[1e-4]), n_remeasure=1,
                    inject_inset_fwhm=float(setup.get("inject_inset_fwhm", 1.0)),
                    pair_area_midpoint=bool(setup.get("pair_area_midpoint", True)),
                    save_fits=False, save_eval_images=False, fm_curve=False, verify=False)
    runner = Runner(red, space, obj, samp, cfg, os.path.join(a.run_dir, "_ablation"), log=log)

    lo_sel = np.array(space.lo, float)
    hi_sel = np.array(space.hi, float)
    live = "nkeep_altroll" in space.names and "nkeep_psfref" in space.names
    if live:
        n_alt = int(hi_sel[space.names.index("nkeep_altroll")])
        n_ref = int(hi_sel[space.names.index("nkeep_psfref")])
    else:
        n_alt = n_ref = None
        log(f"  {a.backend} does not apply a reference library: the library variants are skipped")
    x_def = space.default_vector()

    out = {"run_dir": os.path.abspath(a.run_dir), "n_draws": a.n_draws, "seed": a.seed, "backend": a.backend,
           "space": list(space.names), "pools": {"altroll": n_alt, "psfref": n_ref},
           "carter_mapping": CARTER, "annuli": []}
    annuli = [i - 1 for i in a.annuli] if a.annuli else list(range(len(final)))
    for ia in annuli:
        fr = final[ia]
        runner.ia, runner.contrast = ia, float(fr["contrast"])
        xw = _x_from_params(space, fr["winner_config"]["params"], space.default_vector())
        zone = (float(fr["inrad"]), float(fr["outrad"]))
        configs = {
            "winner": (_vec(runner, xw), None),
            "all": (_vec(runner, xw, nkeep_altroll=n_alt, nkeep_psfref=n_ref), None),
            "rdi": (_vec(runner, xw, nkeep_altroll=0, nkeep_psfref=n_ref), None),
            "adi": (_vec(runner, xw, nkeep_altroll=n_alt, nkeep_psfref=0), None),
            # the idea itself: keep only the best-correlated part of a pool
            "rdi_third": (_vec(runner, xw, nkeep_altroll=0, nkeep_psfref=n_ref // 3), None),
            "ardi_half": (_vec(runner, xw, nkeep_altroll=n_alt // 2, nkeep_psfref=n_ref // 2), None),
            "carter": (_vec(runner, x_def, nkeep_altroll=n_alt, nkeep_psfref=n_ref, **CARTER), None),
            "carter_full": (_vec(runner, x_def, nkeep_altroll=n_alt, nkeep_psfref=n_ref, **CARTER),
                            (IWA_AS / px, float(ann[-1]))),
            "default": (_vec(runner, x_def), None),
        }
        if not live:
            for k in ("all", "rdi", "adi", "rdi_third", "ardi_half"):
                configs.pop(k)
        if a.configs:
            configs = {k: v for k, v in configs.items() if k in a.configs}
        n = runner._nsrc(ia)
        rlo, rhi = runner._band(ia)
        log(f"=== annulus {ia + 1}: zone {zone[0]:.1f}-{zone[1]:.1f} px, {n} sources in "
            f"{rlo:.3f}-{rhi:.3f}\", contrast {runner.contrast:.3e} (run: validated {fr['winner_score']:.3f})")
        draws = [runner.sampler.sample(n, rlo, rhi, np.random.default_rng([a.seed, ia, d]), runner.contrast)
                 for d in range(a.n_draws)]
        rec = {"annulus": ia + 1, "zone_px": zone, "contrast": runner.contrast, "n_sources": n,
               "band_as": [rlo, rhi], "run_validated": fr["winner_score"],
               "draws": [[(s.rho, s.theta) for s in src] for src in draws], "configs": {}}
        for name, (x, zov) in configs.items():
            c = space.decode(x)
            t0 = time.time()
            clean = runner._reduce(c, None, tag=f"abl_a{ia + 1}_{name}_clean", zone=zov)
            t_clean = time.time() - t0
            planet = None
            if zone[0] * px <= PLANET[0] <= zone[1] * px or zov is not None:
                try:
                    planet = float(obj.metric.per_source(clean.image, None, [PLANET[0]], [PLANET[1]])[0])
                except Exception as exc:
                    log(f"   planet S/N failed: {exc!r}")
            raw, srch, walls = [], [], []
            for d, src in enumerate(draws):
                t1 = time.time()
                inj = runner._reduce(c, src, tag=f"abl_a{ia + 1}_{name}_d{d}", zone=zov)
                fmk = runner._fm_for(inj, clean)
                r_raw = obj.score_raw(inj.image, src, clean.image, **fmk)
                r_srch = obj.score_search(inj.image, src, clean.image, **fmk)
                raw.append(float(r_raw.score))
                srch.append(float(r_srch.score))
                walls.append(time.time() - t1)
            p = {k: (int(v) if float(v).is_integer() else float(v)) for k, v in c.params.items()
                 if k in space.names}
            rec["configs"][name] = {"params": p, "zone_override": zov, "raw": raw, "search": srch,
                                    "planet_snr": planet, "wall_clean_s": t_clean,
                                    "wall_inj_s": float(np.mean(walls))}
            log(f"   {name:11s} {p}  raw {np.nanmean(raw):5.2f}  search {np.nanmean(srch):5.2f}"
                f"  planet {'--' if planet is None else f'{planet:5.2f}'}  "
                f"({np.mean(walls):.1f} s/reduction)")
            with open(a.out, "w") as f:                  # checkpoint after every configuration
                json.dump(out | {"annuli": out["annuli"] + [rec]}, f, indent=1)
        out["annuli"].append(rec)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
