#!/usr/bin/env python
"""The README's panel: RX J0534 b (LBTI/LMIRCam L', four nights), run until it has converged.

    python3 scripts/readme_panel.py                       # 3 annuli x 3,000 evaluations, about three days on the Mac Pro
    python3 scripts/readme_panel.py --extra-annuli 0      # the companion's annulus alone, about ten hours
    python3 scripts/readme_panel.py --n-iter 5000         # longer
    python3 scripts/readme_panel.py --cg-step 1           # the contrast grid at FWHM steps
    python3 scripts/readme_panel.py --no-show             # no window, panels still written
    python3 scripts/readme_panel.py --target hd95086      # the panel on public data (VLT/SPHERE)
    python3 scripts/readme_panel.py --extra-annuli 1 --seam-annuli   # two annuli and one on their boundary

RX J0534 is ``scripts/run_rxj0534.py``'s four-night search (``--partitions 4``), with the
companion held out of the objective.  The first annulus is its separation +/- 2 FWHM
(0.20-0.58"), and two more of the same width follow outward (``--extra-annuli``, 0.58-0.97" and
0.97-1.35"), each searched for ``--n-iter`` evaluations; the frames are cropped to hold the
outermost.  The injections go two at a time (``--n-sources``), 180 deg apart on each annulus'
mid-radius, which in the first is the companion's ring at 0.39" (``run_rxj0534.py --even-pa``),
from an anchor drawn afresh for every evaluation.  The README's image before this one was the
companion's annulus alone at 5,000 evaluations with three injections at random angles, whose
running median climbed for about 1,500 evaluations and then stayed flat, so 3,000 show the climb
and a plateau as long again.  The outer annuli hold two and three times the inner one's pixels,
so they take about that much longer.  The contrast grid steps half a FWHM (``--cg-step``): five
separations per annulus, a FWHM in from its edges, so the curve has no points within a FWHM of
the seams.  The panel's title is ``--title`` ("RX J0534"), not the run directory's name.  The
data are read from ``LMIRCAM_ROOT`` (default /Volumes/RAID36TB/LMIRCam).

``--seam-annuli`` adds an annulus on each boundary between the annuli (``run_rxj0534.py
--seam-annuli``), from the center of the one inside to the center of the one outside, numbered by
center among them; with ``--extra-annuli 1`` the annuli are 0.20-0.58", 0.39-0.77" and
0.58-0.97".  The stitch takes each radius from the annulus with the nearest center, so a
boundary's radii come from the middle of the seam annulus' zone, and every tile edge lies 1 FWHM
inside the zones on both sides.  The stitched images then hold none of the KLIP zone edges, where
the reductions ring: the three touching annuli show a dark ring just outside the 0.58" boundary.
The seam annulus' injections sit on the boundary.

``--target hd95086`` is tutorial 02's search on HD 95086 (VLT/SPHERE IRDIS K1+K2, 30-70 px,
HD 95086 b passed as known) with the same data and seed, after a 200-evaluation warm-up.  Its
injections are at a fixed contrast of 3e-6 (``--contrast``, 0 calibrates it as the tutorial
does) and ``angsep`` is searched over 0.1-2.0 lambda/D (``--angsep``).  The floor keeps the
search off ``angsep = 0``, where, with ``anglemax`` at the PA span, the KLIP engine builds one
basis per zone from all the frames, the target included.

The contrast curve at the end is the injection grid (:mod:`klip_tpe.contrast_grid`): one
source per reduction at 12 position angles per separation (``--cg-pa``), each iterated to S/N 5
(``--cg-target``) with the IDL contrast generators' aperture photometry or the run's
matched-filter S/N (``--cg-snr``).

The run writes to ``--run-dir`` (``~/klip_tpe_runs/rxj0534/readme_3ann``, ``readme_seam`` with
``--seam-annuli``, or ``~/klip_tpe_runs/hd95086``), outside any synced folder; the single-annulus
run of 2026-10-30 is in ``~/klip_tpe_runs/rxj0534/readme``.  The image at the top of the README
(``docs/img/step_display.png``) is the three-annulus run's panel at the end of its second annulus,
finished with its validated winner: its stitched images and contrast curve cover the first two
annuli, the companion's included, and its convergence and parameter panels are the second
annulus'.  Running the command again resumes the run.
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

#: per target, the settings that differ: run directory, title, seed, warm-up, grid step, sources
TARGETS = {
    "rxj0534": dict(run_dir="~/klip_tpe_runs/rxj0534/readme_3ann", title="RX J0534", seed=None, n_init=None,
                    cg_step=0.5, n_sources=2, extra_annuli=2),
    "hd95086": dict(run_dir="~/klip_tpe_runs/hd95086", title="HD 95086", seed=3, n_init=200, cg_step=1.0,
                    n_sources=None, extra_annuli=0),
}


def parse(argv=None) -> argparse.Namespace:
    """Read the command line, with the target's settings for the options not given.

    Parameters
    ----------
    argv : list of str or None, default None
        The command line without the program name; None reads ``sys.argv``.

    Returns
    -------
    a : argparse.Namespace
        The options.  Those not given take ``TARGETS``' values for the target (run directory,
        title, seed, warm-up, grid step, source count and extra annuli), and ``run_dir`` is
        expanded.  With ``--seam-annuli`` the default run directory is
        ``~/klip_tpe_runs/rxj0534/readme_seam``, so the touching annuli's run is not resumed.

    Raises
    ------
    SystemExit
        On a bad option, or ``--seam-annuli`` with ``--target hd95086``.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="rxj0534", choices=sorted(TARGETS),
                   help="RX J0534 (LBTI, default) or HD 95086 (VLT/SPHERE, public data)")
    p.add_argument("--n-iter", type=int, default=3000, help="evaluations (default 3000)")
    p.add_argument("--n-init", type=int, default=None,
                   help="warm-up evaluations (default: run_rxj0534.py's 400 for 3,000, or 200 on HD 95086)")
    p.add_argument("--seed", type=int, default=None, help="default: run_rxj0534.py's 21, or tutorial 02's 3")
    p.add_argument("--n-sources", type=int, default=None,
                   help="injected sources per evaluation (default 2 on RX J0534, evenly spaced in angle, "
                        "and the runner's rule on HD 95086)")
    p.add_argument("--extra-annuli", type=int, default=None,
                   help="RX J0534: annuli outside the companion's, each as wide (default 2; 0 for the "
                        "companion's annulus alone)")
    p.add_argument("--seam-annuli", action="store_true",
                   help="RX J0534: one more annulus on each boundary between the annuli, so the stitched "
                        "images hold no KLIP zone edge (run_rxj0534.py --seam-annuli; needs --extra-annuli 1 "
                        "or more). Default run directory: ~/klip_tpe_runs/rxj0534/readme_seam")
    p.add_argument("--run-dir", default=None,
                   help="run directory (default ~/klip_tpe_runs/rxj0534/readme_3ann or ~/klip_tpe_runs/hd95086)")
    p.add_argument("--title", default=None, help="the panel's title (default \"RX J0534\" or \"HD 95086\")")
    p.add_argument("--cg-pa", type=int, default=12, help="contrast grid: position angles per separation (default 12)")
    p.add_argument("--cg-step", type=float, default=None,
                   help="contrast grid: separation step in FWHM (default 0.5 on RX J0534, 1 on HD 95086)")
    p.add_argument("--cg-target", type=float, default=5.0, help="contrast grid: target S/N (default 5)")
    p.add_argument("--cg-snr", default="aperture", choices=["aperture", "matched"],
                   help="contrast grid: the IDL contrast generators' aperture photometry (default) or the "
                        "run's matched-filter S/N")
    p.add_argument("--workers", default="auto", help="reduction workers (default auto)")
    p.add_argument("--no-show", action="store_true", help="no live window (the panels are still written)")
    hd = p.add_argument_group("HD 95086 only")
    hd.add_argument("--angsep", type=float, nargs=2, default=(0.1, 2.0), metavar=("LO", "HI"),
                    help="angsep search range in lambda/D (default 0.1 2.0)")
    hd.add_argument("--contrast", type=float, default=3e-6,
                    help="fixed injection contrast (default 3e-6), or 0 to calibrate it as tutorial 02 does")
    hd.add_argument("--every", type=int, default=5, help="a panel every this many evaluations (default 5)")
    a = p.parse_args(argv)
    if a.seam_annuli and a.target != "rxj0534":
        p.error("--seam-annuli is for RX J0534's annuli")
    if a.seam_annuli and a.run_dir is None:
        a.run_dir = "~/klip_tpe_runs/rxj0534/readme_seam"
    for k, v in TARGETS[a.target].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.run_dir = os.path.expanduser(a.run_dir)
    return a


def rxj0534_argv(a: argparse.Namespace) -> list:
    """Return ``scripts/run_rxj0534.py``'s command line for the README run.

    Parameters
    ----------
    a : argparse.Namespace
        The options from :func:`parse`.

    Returns
    -------
    argv : list of str
        The four-night search (``--partitions 4``) with the sources 180 deg apart on each annulus'
        mid-radius (``--even-pa``) and the injection-grid contrast curve, with the run directory,
        title, budget, extra and seam annuli, warm-up, seed and window as given.
    """
    argv = ["--partitions", "4", "--n-iter", str(a.n_iter), "--out", a.run_dir, "--title", a.title,
            "--n-sources", str(a.n_sources), "--even-pa", "--contrast-grid", "grid", "--cg-pa", str(a.cg_pa), "--cg-step", f"{a.cg_step:g}",
            "--cg-target", f"{a.cg_target:g}", "--cg-snr", a.cg_snr, "--workers", str(a.workers)]
    if a.extra_annuli:
        argv += ["--extra-annuli", str(a.extra_annuli)]
    if a.seam_annuli:
        argv += ["--seam-annuli"]
    if a.n_init is not None:
        argv += ["--n-init", str(a.n_init)]
    if a.seed is not None:
        argv += ["--seed", str(a.seed)]
    return argv + (["--panels"] if a.no_show else ["--show"])


def run_rxj0534(a: argparse.Namespace) -> int:
    """Run the README search on RX J0534 through ``scripts/run_rxj0534.py``.

    Parameters
    ----------
    a : argparse.Namespace
        The options from :func:`parse`.

    Returns
    -------
    status : int
        ``run_rxj0534.main``'s status.  The run's products go to ``a.run_dir``, and the path of its
        last panel is printed.
    """
    scripts = os.path.join(_HERE, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_rxj0534 as rx
    rc = rx.main(rxj0534_argv(a))
    steps = sorted(glob.glob(os.path.join(a.run_dir, "steps", "step*.png")))
    if steps:
        print(f"last panel: {steps[-1]}")
    return rc


#: HD 95086 b (arcsec, deg), passed as known
HD95086_B = (0.62, 145.0)


def hd95086_reducer(workers="auto"):
    """Tutorial 02's reducer of HD 95086 (VLT/SPHERE IRDIS K1+K2), as :func:`run_hd95086` builds
    it.  ``scripts/fm_one_at_a_time.py`` rebuilds a run's with it."""
    from astropy.io import fits

    from klip_tpe import datasets
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
    return generic.make_reducer(dsets, pxscale=inst["pxscale"], lam_m=lam, diam_m=inst["diam_m"],
                                star_flux=star_flux, max_workers=workers)


def run_hd95086(a: argparse.Namespace) -> int:
    from klip_tpe import CalibrationConfig, ContrastGridConfig, Runner, RunConfig, ValidationConfig
    from klip_tpe.display import LiveDisplay
    from klip_tpe.instruments import generic

    red = hd95086_reducer(a.workers)
    space = generic.make_space(red, k_klip_max=30, angsep_range=tuple(a.angsep))
    space.project = generic.make_guard(red, k_max=30)
    objective, sampler = generic.default_config(red, known=[HD95086_B])
    sampler.excl_fwhm = 6.0                                  # injections >= 6 FWHM from HD 95086 b
    cfg = RunConfig(ann_edges=[30, 70], n_iter=a.n_iter, n_init=a.n_init, seed=a.seed, n_remeasure=3,
                    n_sources=a.n_sources,
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
    """Run the README panel's search on the chosen target.

    Parameters
    ----------
    argv : list of str or None, default None
        The command line without the program name; None reads ``sys.argv``.

    Returns
    -------
    status : int
        The status of :func:`run_rxj0534` or :func:`run_hd95086`, 0 when the run finishes.
    """
    a = parse(argv)
    return run_rxj0534(a) if a.target == "rxj0534" else run_hd95086(a)


if __name__ == "__main__":
    sys.exit(main())
