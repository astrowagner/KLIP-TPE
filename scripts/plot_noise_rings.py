#!/usr/bin/env python
"""Draw the noise apertures every phase of a run measures, as the code measures them.

A small synthetic run with a chosen geometry goes through every phase (calibration, search,
validation with its final round and re-scored winner, the contrast grid, param_verify, the
verify, the candidate search and the stitched S/N maps) while
:class:`klip_tpe.metrics.RingRecorder` keeps every noise ring.  One panel per phase shows the
ring of one measurement: the apertures used, the ones skipped and why, the sources, the known
companions, the pixel mask and any sidelobe arcs.  A second, short run under ``--compare``
(``farthest`` by default, the rule of 2026-10-31) adds its search rings for comparison.

Geometries::

    near-s6     NEAR, 0.0445"/px, FWHM 6.24 px, two nights of 153.6 and 105.3 deg of rotation,
                s6's annulus 3 (11.50-20.81 px, 2 sources a draw) and annulus 8 (39.43-48.74 px,
                6 sources a draw, the counts s6's rule gives there)
    mwc758      ALES, 0.0345"/px, FWHM 2.91 px, 126.9 deg of rotation, sources at 0.6",
                MWC 758c known at (0.603", 228.5 deg) with no injection within 25 deg of it,
                the run's disk mask (--disk-mask, its disk_mask.fits)

The frames are synthetic, so the S/N values mean nothing.  The layout is the code's own: the
counts, keep-outs, arcs, fallbacks and masks come from the measuring functions.

    python scripts/plot_noise_rings.py --geometry near-s6 --out rings_near_s6
    python scripts/plot_noise_rings.py --geometry mwc758 --disk-mask RUN/disk_mask.fits --out rings_mwc758
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from klip_tpe.contrast_grid import ContrastGridConfig  # noqa: E402
from klip_tpe.injection import GaussianPSF  # noqa: E402
from klip_tpe.metrics import MawetPeakSNR, Objective, RingRecorder  # noqa: E402
from klip_tpe.positions import PositionSampler  # noqa: E402
from klip_tpe.reducer import KLIPReducer, PartitionedReducer  # noqa: E402
from klip_tpe.ringplot import phase_from_stack, plot_rings  # noqa: E402
from klip_tpe.runner import CalibrationConfig, RunConfig, Runner, ValidationConfig  # noqa: E402
from klip_tpe.space import Param, SearchSpace, kgrid  # noqa: E402
from klip_tpe.synthetic import synthetic_klip_dataset  # noqa: E402

GEOMETRIES: Dict[str, Dict[str, Any]] = {
    "near-s6": dict(pxscale=0.0445, fwhm=6.24, size=150, spans=(153.6, 105.3), nframes=36,
                    annuli=[("annulus 3 of s6", [11.5, 20.81], 2), ("annulus 8 of s6", [39.43, 48.74], 6)],
                    known=(), forbidden_pa=(), inject_rho=None),
    "mwc758": dict(pxscale=0.0345, fwhm=2.9083216765240167, size=62, spans=(126.9, 126.9), nframes=36,
                   annuli=[("MWC 758, sources at 0.6\"", [11.594202898550725, 23.18840579710145], None)],
                   known=((0.603, 228.5),), forbidden_pa=((228.5, 25.0),), inject_rho=0.6),
}


def build(geo: Dict[str, Any], edges: Sequence[float], n_sources: Optional[int], mask: Optional[np.ndarray],
          sidelobes: str, full: bool, seed: int = 1):
    """The reducer, space, objective, sampler and config of one synthetic run."""
    px, fwhm, size = float(geo["pxscale"]), float(geo["fwhm"]), int(geo["size"])
    reds = {}
    for i, span in enumerate(geo["spans"]):
        ds = synthetic_klip_dataset(nframes=int(geo["nframes"]), size=size, pa_span=float(span), seed=seed + i)
        reds[f"n{i + 1}"] = KLIPReducer(ds, px, 11e-6, 8.2, fwhm_px=fwhm, outrad_cap=size / 2 - 2,
                                        injection_model=GaussianPSF(fwhm, star_flux=1e4))
    red = PartitionedReducer(reds)
    block = [Param("bin", 1, 3, "int", default=1), Param("n_ang", 1, 3, "int", default=2),
             Param("angsep", 0.3, 1.2, "float", default=0.6), Param("k_klip", 1, 10, "int", grid=kgrid(10), default=5)]
    parts = list(reds)
    space = SearchSpace().replicate(block, parts).with_selection("two_slot", partitions=parts)
    metric = MawetPeakSNR(pxscale=px, fwhm=fwhm, known=list(geo["known"]), pixel_mask=mask)
    obj = Objective(metric, clean_subtract=True)
    samp = PositionSampler(fwhm_as=fwhm * px, known=list(geo["known"]), forbidden_pa=list(geo["forbidden_pa"]),
                           **({"forbidden_mask": mask, "mask_pxscale": px} if mask is not None else {}))
    kw = dict(ann_edges=list(edges), n_iter=[8 if full else 4], n_init=3, seed=seed, n_remeasure=2,
              defaults={"k_klip": 5}, sidelobes=sidelobes, save_eval_images=False, write_setup_files=False,
              fm_curve=False, fm_preview=False, stitch_every=0,
              calibration=CalibrationConfig(max_trials=4, n_remeasure=2),
              **({"n_sources": int(n_sources)} if n_sources else {}),
              **({"inject_rho": float(geo["inject_rho"])} if geo.get("inject_rho") else {}))
    if full:
        kw.update(validation=ValidationConfig(n_top=2, n_valid=2, n_final=2, n_final_draws=3, n_rescore=2),
                  contrast_grid=ContrastGridConfig(n_pa=4), verify=True, verify_n_boot=8, candidates=True,
                  cand_verify_top=2, param_verify=True, n_pv=3)
    else:
        kw.update(validation=ValidationConfig(n_top=1, n_valid=1, n_rescore=0),
                  contrast_grid=ContrastGridConfig(mode="off"), verify=False, candidates=False, param_verify=False)
    return red, space, obj, samp, RunConfig(**kw)


def record(geo, edges, n_sources, mask, sidelobes: str, full: bool, log=lambda s: None) -> List[Dict[str, Any]]:
    """Every ring a synthetic run measures, each with its phase."""
    red, space, obj, samp, cfg = build(geo, edges, n_sources, mask, sidelobes, full)
    d = tempfile.mkdtemp(prefix="klip_tpe_rings_")
    keep = Runner.SCORE_EARLY
    Runner.SCORE_EARLY = False            # every draw scored on the main thread, with its caller on the stack
    try:
        with RingRecorder(phase=phase_from_stack) as rec:
            Runner(red, space, obj, samp, cfg, d, log=log).run()
    finally:
        Runner.SCORE_EARLY = keep
    return rec.records


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", choices=sorted(GEOMETRIES), default="near-s6")
    ap.add_argument("--disk-mask", default=None, help="FITS pixel mask (True = left out), the run's disk_mask.fits")
    ap.add_argument("--sidelobes", default="off", choices=["off", "farthest", "clean"],
                    help="the rings of the run drawn (default off, the default since 2026-11-01)")
    ap.add_argument("--compare", default="farthest", choices=["none", "off", "farthest", "clean"],
                    help="a short run's search rings under this setting, added for comparison (default farthest)")
    ap.add_argument("--out", default="noise_rings", help="file stem; one PNG per annulus")
    ap.add_argument("--replot", action="store_true",
                    help="draw again from the rings an earlier call kept (OUT_N.rings.pkl), without the runs")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    geo = GEOMETRIES[a.geometry]
    mask = None
    if a.disk_mask:
        from astropy.io import fits
        mask = np.asarray(fits.getdata(a.disk_mask)).astype(bool)
        if mask.shape != (geo["size"], geo["size"]):
            raise SystemExit(f"--disk-mask: {mask.shape}, the geometry's frames are {geo['size']} px square")
    log = print if a.verbose else (lambda s: None)
    out = []
    import pickle
    for k, (name, edges, nsrc) in enumerate(geo["annuli"]):
        path = f"{a.out}_{k + 1}.png" if len(geo["annuli"]) > 1 else f"{a.out}.png"
        kept = os.path.splitext(path)[0] + ".rings.pkl"
        if a.replot:
            with open(kept, "rb") as f:
                recs, extra = pickle.load(f)
        else:
            recs = record(geo, edges, nsrc, mask, a.sidelobes, True, log)
            extra = []
            if a.compare != "none" and a.compare != a.sidelobes:
                cmp_ = [r for r in record(geo, edges, nsrc, mask, a.compare, False, log)
                        if r.get("phase") == "search" and r.get("role") != "clean"]
                if cmp_:
                    most = max(len(r.get("all_sources_xy") or r.get("sources_xy") or []) for r in cmp_)
                    pick = [r for r in cmp_ if len(r.get("all_sources_xy") or r.get("sources_xy") or []) == most][-1]
                    extra.append(dict(pick, _label=f"search with --sidelobes {a.compare}"))
            with open(kept, "wb") as f:
                pickle.dump((recs, extra), f)
        title = (f"{a.geometry}: {name}, {edges[0]:.2f}-{edges[1]:.2f} px, --sidelobes {a.sidelobes} "
                 f"(synthetic frames: the layout is the code's, the S/N values are not data)")
        plot_rings(recs, path, title, pxscale=geo["pxscale"], zones=[tuple(edges)], pixel_mask=mask,
                   extra=extra, known=list(geo["known"]))
        n_by = {}
        for r in recs:
            n_by[r.get("phase")] = n_by.get(r.get("phase"), 0) + 1
        print(f"{path}: {len(recs)} rings recorded, " + ", ".join(f"{k_} {v}" for k_, v in n_by.items()))
        out.append(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
