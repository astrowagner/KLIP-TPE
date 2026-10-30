#!/usr/bin/env python
"""The README's panel: RX J0534 b (LBTI/LMIRCam L', four nights), run until it has converged.

    python3 scripts/readme_panel.py                       # 3,000 evaluations, about two days on the Mac Pro
    python3 scripts/readme_panel.py --n-iter 5000         # longer
    python3 scripts/readme_panel.py --cg-step 1           # the contrast grid at FWHM steps
    python3 scripts/readme_panel.py --no-show             # no window, panels still written
    python3 scripts/readme_panel.py --target hd95086      # the panel on public data (VLT/SPHERE)

RX J0534 is ``scripts/run_rxj0534.py``'s four-night search (``--partitions 4``): the companion
held out of the objective, three injections on its ring at 0.39", the annulus its separation
+/- 2 FWHM.  The README's image before this one was the same search at 5,000 evaluations, whose
running median climbed for about 1,500 evaluations and then stayed flat, so 3,000 show the climb
and a plateau as long again.  The contrast grid steps half a FWHM (``--cg-step``), which in that
annulus gives five separations from 0.29" to 0.49" in place of three.  The panel's title is
``--title`` ("RX J0534"), not the run directory's name.  The data are read from ``LMIRCAM_ROOT``
(default /Volumes/RAID36TB/LMIRCam).

``--target hd95086`` is tutorial 02's search on HD 95086 (VLT/SPHERE IRDIS K1+K2, 30-70 px,
HD 95086 b passed as known) with the same data and seed, after a 200-evaluation warm-up.  Its
injections are at a fixed contrast of 3e-6 (``--contrast``, 0 calibrates it as the tutorial
does) and ``angsep`` is searched over 0.1-2.0 lambda/D (``--angsep``).  The floor keeps the
search off ``angsep = 0``, where, with ``anglemax`` at the PA span, the KLIP engine builds one
basis per zone from all the frames, the target included.

The contrast curve at the end is the injection grid (:mod:`klip_tpe.contrast_grid`): one
source per reduction at 12 position angles per separation (``--cg-pa``), each iterated to S/N 5
(``--cg-target``) with the run's matched-filter S/N or aperture photometry (``--cg-snr``).

The run writes to ``--run-dir`` (``~/klip_tpe_runs/rxj0534/readme``, or
``~/klip_tpe_runs/hd95086``), outside any synced folder.  Its last panel, the finished annulus
with the validated winner, is the image at the top of the README (``docs/img/step_display.png``).
Running the command again resumes the run.
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

#: per target, the settings that differ: run directory, title, seed, warm-up, grid step
TARGETS = {
    "rxj0534": dict(run_dir="~/klip_tpe_runs/rxj0534/readme", title="RX J0534", seed=None, n_init=None,
                    cg_step=0.5),
    "hd95086": dict(run_dir="~/klip_tpe_runs/hd95086", title="HD 95086", seed=3, n_init=200, cg_step=1.0),
}


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="rxj0534", choices=sorted(TARGETS),
                   help="RX J0534 (LBTI, default) or HD 95086 (VLT/SPHERE, public data)")
    p.add_argument("--n-iter", type=int, default=3000, help="evaluations (default 3000)")
    p.add_argument("--n-init", type=int, default=None,
                   help="warm-up evaluations (default: run_rxj0534.py's 400 for 3,000, or 200 on HD 95086)")
    p.add_argument("--seed", type=int, default=None, help="default: run_rxj0534.py's 21, or tutorial 02's 3")
    p.add_argument("--run-dir", default=None,
                   help="run directory (default ~/klip_tpe_runs/rxj0534/readme or ~/klip_tpe_runs/hd95086)")
    p.add_argument("--title", default=None, help="the panel's title (default \"RX J0534\" or \"HD 95086\")")
    p.add_argument("--cg-pa", type=int, default=12, help="contrast grid: position angles per separation (default 12)")
    p.add_argument("--cg-step", type=float, default=None,
                   help="contrast grid: separation step in FWHM (default 0.5 on RX J0534, 1 on HD 95086)")
    p.add_argument("--cg-target", type=float, default=5.0, help="contrast grid: target S/N (default 5)")
    p.add_argument("--cg-snr", default="matched", choices=["matched", "aperture"],
                   help="contrast grid: the run's matched-filter S/N (default) or aperture photometry")
    p.add_argument("--workers", default="auto", help="reduction workers (default auto)")
    p.add_argument("--no-show", action="store_true", help="no live window (the panels are still written)")
    hd = p.add_argument_group("HD 95086 only")
    hd.add_argument("--angsep", type=float, nargs=2, default=(0.1, 2.0), metavar=("LO", "HI"),
                    help="angsep search range in lambda/D (default 0.1 2.0)")
    hd.add_argument("--contrast", type=float, default=3e-6,
                    help="fixed injection contrast (default 3e-6), or 0 to calibrate it as tutorial 02 does")
    hd.add_argument("--every", type=int, default=5, help="a panel every this many evaluations (default 5)")
    a = p.parse_args(argv)
    for k, v in TARGETS[a.target].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.run_dir = os.path.expanduser(a.run_dir)
    return a


def rxj0534_argv(a: argparse.Namespace) -> list:
    """``scripts/run_rxj0534.py``'s command line for the README run."""
    argv = ["--partitions", "4", "--n-iter", str(a.n_iter), "--out", a.run_dir, "--title", a.title,
            "--contrast-grid", "grid", "--cg-pa", str(a.cg_pa), "--cg-step", f"{a.cg_step:g}",
            "--cg-target", f"{a.cg_target:g}", "--cg-snr", a.cg_snr, "--workers", str(a.workers)]
    if a.n_init is not None:
        argv += ["--n-init", str(a.n_init)]
    if a.seed is not None:
        argv += ["--seed", str(a.seed)]
    return argv + (["--panels"] if a.no_show else ["--show"])


def run_rxj0534(a: argparse.Namespace) -> int:
    scripts = os.path.join(_HERE, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_rxj0534 as rx
    rc = rx.main(rxj0534_argv(a))
    steps = sorted(glob.glob(os.path.join(a.run_dir, "steps", "step*.png")))
    if steps:
        print(f"last panel: {steps[-1]}")
    return rc


def run_hd95086(a: argparse.Namespace) -> int:
    from astropy.io import fits

    from klip_tpe import CalibrationConfig, ContrastGridConfig, Runner, RunConfig, ValidationConfig, datasets
    from klip_tpe.display import LiveDisplay
    from klip_tpe.instruments import generic

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
    space = generic.make_space(red, k_klip_max=30, angsep_range=tuple(a.angsep))
    space.project = generic.make_guard(red, k_max=30)
    objective, sampler = generic.default_config(red, known=[(0.62, 145.0)])
    sampler.excl_fwhm = 6.0                                  # injections >= 6 FWHM from HD 95086 b
    cfg = RunConfig(ann_edges=[30, 70], n_iter=a.n_iter, n_init=a.n_init, seed=a.seed, n_remeasure=3,
                    validation=ValidationConfig(n_top=6, n_valid=8),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=2,
                                                  forced=[a.contrast] if a.contrast > 0 else None),
                    defaults={"k_klip": 10}, fm_curve=True,
                    contrast_grid=ContrastGridConfig(mode="grid", n_pa=a.cg_pa, sep_step_fwhm=a.cg_step,
                                                     target_snr=a.cg_target, snr=a.cg_snr))
    display = LiveDisplay(a.run_dir, show=not a.no_show, every=a.every, pdf_every=0, movie=False, title=a.title)
    runner = Runner(red, space, objective, sampler, cfg, a.run_dir, callbacks=[display])
    t0 = time.time()
    r = runner.run()[0]
    print(f"{(time.time() - t0) / 3600:.2f} h: winner eval {r.winner_index + 1}, validated S/N {r.winner_score:.2f}")
    steps = sorted(glob.glob(os.path.join(a.run_dir, "steps", "step*.png")))
    if steps:
        print(f"last panel: {steps[-1]}")
    return 0


def main(argv=None) -> int:
    a = parse(argv)
    return run_rxj0534(a) if a.target == "rxj0534" else run_hd95086(a)


if __name__ == "__main__":
    sys.exit(main())
