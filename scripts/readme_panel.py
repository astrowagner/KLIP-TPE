#!/usr/bin/env python
"""The README's panel, from tutorial 02's search on HD 95086 run until it has converged.

    python3 scripts/readme_panel.py                       # 3,000 evaluations, about 2.5 h on the Mac Pro
    python3 scripts/readme_panel.py --n-iter 5000         # longer
    python3 scripts/readme_panel.py --no-show             # no window, panels still written

Tutorial 02 (VLT/SPHERE IRDIS K1+K2, 30-70 px, HD 95086 b passed as known) stops at 300
evaluations, where the running mean is still climbing.  This is the same search with the same
seed, data and settings, and a budget long enough to show the climb and the plateau after it.
The warm-up is 200 evaluations.  The injections spread over the annulus, as the tutorial's do,
so the injection-calibrated contrast curve and the KLIP-FM cross-check measure the same thing at
every separation.

The run writes to ``~/klip_tpe_runs/hd95086`` (``--run-dir``, whose name the panel shows as
its title), outside any synced folder, with a panel every fifth evaluation.  Its last panel,
the finished annulus with the validated winner, is the image at the top of the README
(``docs/img/step_display.png``).  Running the command again resumes the run.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from astropy.io import fits  # noqa: E402

from klip_tpe import CalibrationConfig, Runner, RunConfig, ValidationConfig, datasets  # noqa: E402
from klip_tpe.display import LiveDisplay  # noqa: E402
from klip_tpe.instruments import generic  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-iter", type=int, default=3000, help="evaluations (default 3000)")
    p.add_argument("--n-init", type=int, default=200, help="warm-up evaluations (default 200)")
    p.add_argument("--seed", type=int, default=3, help="tutorial 02's seed (default 3)")
    p.add_argument("--run-dir", default=os.path.expanduser("~/klip_tpe_runs/hd95086"),
                   help="run directory, whose name is the panel's title (default ~/klip_tpe_runs/hd95086)")
    p.add_argument("--every", type=int, default=5, help="a panel every this many evaluations (default 5)")
    p.add_argument("--workers", default="auto", help="reduction workers (default auto)")
    p.add_argument("--no-show", action="store_true", help="no live window")
    a = p.parse_args(argv)

    files = datasets.fetch("sphere_hd95086")
    inst = datasets.INSTRUMENT["sphere_hd95086"]
    angles = fits.getdata(files["angles"])
    dsets, star_flux, lam = {}, {}, {}
    for band in ("K1", "K2"):
        ds = generic.load_cube(files[f"cube_{band}"], angles, psf=files[f"psf_{band}"], name=band)
        dsets[band] = ds
        psf = ds.meta["psf"]
        star_flux[band] = float(psf[psf > 0].sum())      # the flux frames are on the science scale
        lam[band] = inst["lam_m"] if band == "K1" else inst["lam_m_K2"]
    red = generic.make_reducer(dsets, pxscale=inst["pxscale"], lam_m=lam, diam_m=inst["diam_m"],
                               star_flux=star_flux, max_workers=a.workers)
    space = generic.make_space(red, k_klip_max=30)
    space.project = generic.make_guard(red, k_max=30)
    objective, sampler = generic.default_config(red, known=[(0.62, 145.0)])
    sampler.excl_fwhm = 6.0                                  # injections >= 6 FWHM from HD 95086 b
    cfg = RunConfig(ann_edges=[30, 70], n_iter=a.n_iter, n_init=a.n_init, seed=a.seed, n_remeasure=3,
                    validation=ValidationConfig(n_top=6, n_valid=8),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=2),
                    defaults={"k_klip": 10}, fm_curve=True)
    display = LiveDisplay(a.run_dir, show=not a.no_show, every=a.every, pdf_every=0, movie=False)
    runner = Runner(red, space, objective, sampler, cfg, a.run_dir, callbacks=[display])
    t0 = time.time()
    r = runner.run()[0]
    print(f"{(time.time() - t0) / 3600:.2f} h: winner eval {r.winner_index + 1}, validated S/N {r.winner_score:.2f}")
    steps = sorted(glob.glob(os.path.join(a.run_dir, "steps", "step*.png")))
    if steps:
        print(f"last panel: {steps[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
