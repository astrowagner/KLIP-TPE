#!/usr/bin/env python3
"""Read the contrast curves of Fig. 8 of the KLIP-TPE paper with the current contrast curve.

The final-image grid (:mod:`klip_tpe.final_grid`) reads each paper run's S/N = 5 contrast curve
on the run's stitch, one injected source per reduction, each point's contrast stepped until its
S/N is within 10 % of 5, with the aperture photometry of the IDL predecessor's contrast-curve
generators (the contrast grid's current settings).  ``finalists=1`` gives each annulus its
validated winner alone, so no annulus can take another winner and Table 2 of the paper stands.

The runs are A2 (beta Pic, VLT/NACO), C (HD 95086, VLT/SPHERE) and D2 (HIP 65426, JWST/NIRCam on
pyKLIP, b subtracted).  Each is resumed read-only from a copy, with the reducer, objective,
sampler and guard that ``paper_runs/run_demos.py`` built for it.  Before the stage, each annulus'
winner is reduced again and compared with the run's ``best_clean.fits``, which shows whether the
current code reduces the frames the run was made on.

Usage::

    cd paper_runs && python3 fig8_curve.py [A2 C D2] [--plan]

reads the run directories where ``run_demos.py`` puts them (this folder, or ``$RUNS_DIR``) with
the package of this checkout, and writes to ``fig8/`` there.  ``--code``, ``--runs`` and
``--out`` override the three.  The paper's curves were read with commit ``233bbee``, from a clone
pinned to it and from copies of the three run directories without their ``steps/``::

    PYTHONPATH=<clone> python3 fig8_curve.py --code <clone> --runs <copies> --out <dir> \
        --commit 233bbee A2

one stage per call, in chunks that resume from the stage's saved readings.  ``fig8_plot.py``
then draws Fig. 8 from the outputs.

Outputs, in ``--out``: ``<run>_fig8/`` (the stage's ``election.*``, ``winners.txt``,
``curve.json``, ``curve.txt`` and its readings under ``state/``, from which a stage started
again resumes), ``fig8_<stage>.json`` (the curve, the check and the provenance) and
``<stage>.log``.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

#: The paper's stage names and their run directories.
STAGES = {"A2": "A2_betapic", "C": "C_hd95086", "D2": "D2_hip65426_pyklip_sub"}


def build(R, generic, stage):
    """Return the reducer, objective, sampler and guard that run_demos.py built for a stage.

    Parameters
    ----------
    R : module
        ``paper_runs/run_demos.py`` of the pinned clone.
    generic : module
        ``klip_tpe.instruments.generic``.
    stage : {"A2", "C", "D2"}
        The paper's stage.

    Returns
    -------
    reducer : Reducer
        The stage's reducer, on the frames its data sets give.
    objective : Objective
        The stage's detection metric.
    sampler : PositionSampler
        The stage's injection sampler.
    guard : callable
        The projection of the stage's search space.
    """
    if stage == "A2":
        dsets, sf, inst = R.betapic_dataset(1)
        red = generic.make_reducer(dsets, star_flux=sf, max_workers=R.workers(), log=R.log, **inst)
        guard = generic.make_guard(red, k_max=30)
        fpa, pmask = R.bp_disk(red)
        obj, samp = generic.default_config(red, known=[R.BP], forbidden_pa=fpa, pixel_mask=pmask,
                                           flatten=R.FLATTEN_GROUND)
    elif stage == "C":
        red = R.hd95086_objects()
        guard = generic.make_guard(red, k_max=30)
        obj, samp = generic.default_config(red, known=[R.HD], flatten=R.FLATTEN_GROUND)
    elif stage == "D2":
        red = R.hip65426_objects(engine="pyklip")
        guard = generic.make_guard(red, k_max=18, n_min_ref=4)
        obj, samp = generic.default_config(red, known=[R.HIP], flatten=False)
    else:
        raise SystemExit(f"unknown stage {stage!r}: one of {', '.join(STAGES)}")
    return red, obj, samp, guard


def check_winners(runner, log):
    """Reduce each annulus' winner again, clean, and compare it with the run's best_clean.fits.

    Parameters
    ----------
    runner : Runner
        The resumed run, its workers started.
    log : callable
        Function that takes each line of the log.

    Returns
    -------
    rows : list of dict
        Per annulus counted from 1: ``pixels`` compared (finite in both, inside the scored annulus
        less 1 px at each edge), ``rms_ref`` (the saved image's rms there), and ``rms_diff_rel`` and
        ``max_diff_rel`` (the difference's rms and largest absolute value over ``rms_ref``), or
        ``error`` where the comparison could not be made.
    """
    from astropy.io import fits
    from klip_tpe.refine import StitchScorer
    sc = StitchScorer(runner, log=log)
    edges = [float(e) for e in runner.cfg.ann_edges]
    rows = []
    for res in sorted((x for x in runner.results if x is not None), key=lambda x: x.annulus):
        ia = int(res.annulus)
        row = {"annulus": ia + 1}
        try:
            cfg = runner._config_from_dict(res.winner_config)
            img = np.asarray(sc.reduce(ia, cfg, None, f"check, annulus {ia + 1}"), float)
            ref = np.asarray(fits.getdata(os.path.join(runner.run_dir, f"annulus{ia + 1:02d}",
                                                       "best_clean.fits")), float)
            if img.shape != ref.shape:
                raise ValueError(f"shapes {img.shape} and {ref.shape} differ")
            ny, nx = ref.shape
            yy, xx = np.indices(ref.shape)
            rr = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
            core = ((rr >= edges[ia] + 1) & (rr <= edges[ia + 1] - 1) & np.isfinite(img)
                    & np.isfinite(ref))
            d = img[core] - ref[core]
            s = float(np.sqrt(np.mean(ref[core] ** 2))) or 1.0
            row.update(pixels=int(core.sum()), rms_ref=s, rms_diff_rel=float(np.sqrt(np.mean(d ** 2)) / s),
                       max_diff_rel=float(np.max(np.abs(d)) / s))
            log(f"  check, annulus {ia + 1}: the winner reduced again differs from best_clean.fits by "
                f"{row['rms_diff_rel']:.2e} rms and {row['max_diff_rel']:.2e} at most, relative to the "
                f"saved image's rms, over {row['pixels']} px")
        except Exception as exc:          # the check informs, it does not stop the stage
            row["error"] = repr(exc)
            log(f"  check, annulus {ia + 1}: could not compare ({exc!r})")
        rows.append(row)
    return rows


def versions():
    """Return the versions of Python and of the packages the stage uses, for the provenance.

    Returns
    -------
    versions : dict of str to str
        Python's version and executable, the platform, and numpy, scipy, astropy and pyklip's
        versions where they import.
    """
    out = {"python": platform.python_version(), "executable": sys.executable, "platform": platform.platform()}
    for name in ("numpy", "scipy", "astropy", "pyklip"):
        try:
            out[name] = str(__import__(name).__version__)
        except Exception as exc:
            out[name] = f"not imported ({type(exc).__name__})"
    return out


def main():
    """Run the final-image grid's curve on one or more paper runs, as the command line asks."""
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("stages", nargs="*", default=list(STAGES), help="A2, C and/or D2 (default: all three)")
    ap.add_argument("--code", default=os.path.dirname(here),
                    help="the klip-tpe checkout to run (default: the one that holds this script)")
    ap.add_argument("--runs", default=None,
                    help="folder holding the run directories (default: run_demos.OUT, this folder or $RUNS_DIR)")
    ap.add_argument("--out", default=None, help="folder for the outputs (default: fig8/ in the runs' folder)")
    ap.add_argument("--commit", default="", help="the checkout's commit, for the provenance")
    ap.add_argument("--plan", action="store_true", help="count the points and the reductions, reduce nothing")
    a = ap.parse_args()
    code = os.path.abspath(os.path.expanduser(a.code))
    sys.path.insert(0, os.path.join(code, "paper_runs"))
    sys.path.insert(0, code)
    if a.runs:
        os.environ["RUNS_DIR"] = os.path.abspath(os.path.expanduser(a.runs))   # run_demos.OUT
    import klip_tpe
    if not os.path.abspath(klip_tpe.__file__).startswith(code + os.sep):
        raise SystemExit(f"klip_tpe was imported from {klip_tpe.__file__}, not from {code}: "
                         f"run with PYTHONPATH={code}")
    import run_demos as R
    runs = os.path.abspath(R.OUT)          # the run directories, which nothing here writes
    out = os.path.abspath(os.path.expanduser(a.out)) if a.out else os.path.join(runs, "fig8")
    from klip_tpe import Runner
    from klip_tpe.contrast_grid import ContrastGridConfig
    from klip_tpe.final_grid import FinalGridConfig, run_final_grid, run_weights
    from klip_tpe.instruments import generic
    os.makedirs(out, exist_ok=True)
    for st in (s.upper() for s in a.stages):
        if st not in STAGES:
            raise SystemExit(f"unknown stage {st!r}: one of {', '.join(STAGES)}")
        run_dir = os.path.join(runs, STAGES[st])
        od = os.path.join(out, STAGES[st] + "_fig8")
        logf = open(os.path.join(out, f"{st}.log"), "a")

        def log(msg, _f=logf):
            """Print a line of the log and append it to the stage's log file.

            Parameters
            ----------
            msg : str
                The line.
            _f : file
                The stage's open log file.
            """
            print(msg, flush=True)
            _f.write(str(msg) + "\n")
            _f.flush()

        t0 = time.time()
        log(f"== {st} {'(plan) ' if a.plan else ''}{time.strftime('%Y-%m-%d %H:%M:%S')}: {run_dir} -> {od}, "
            f"klip_tpe from {os.path.dirname(klip_tpe.__file__)} {a.commit}")
        red, obj, samp, guard = build(R, generic, st)
        r = Runner.resume(run_dir, red, obj, samp, project=guard, same_frames=False, log=log)
        # A run made before the injection grid resumes with the grid's mode "off".  The curve takes
        # the contrast grid's current settings: aperture S/N, within 10 % of S/N 5, no sidelobe arcs.
        r.cfg.contrast_grid = ContrastGridConfig()
        fc = FinalGridConfig(mode="on", finalists=1)
        weights = run_weights(r.run_dir, r.cfg.nann)
        if a.plan:
            run_final_grid(r, od, config=fc, weights=weights, plan=True, log=log)
            logf.close()
            continue
        r._start_workers()
        check = check_winners(r, log)
        rec = run_final_grid(r, od, config=fc, weights=weights, log=log)
        cv = rec.get("curve") or {}
        keep = ("r_as", "curve", "p16", "p84", "n", "n_pos", "n_all", "n_blocked", "annulus", "band", "role",
                "read_with", "target_snr", "precision", "snr", "sidelobe", "n_pa", "reductions", "seams",
                "seam_steps")
        summary = {"stage": st, "run": STAGES[st], "commit": a.commit, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "minutes": (time.time() - t0) / 60.0, "versions": versions(), "check": check,
                   "changed": [k for k, v in (rec.get("annuli") or {}).items() if v.get("changed")],
                   "calibration": rec.get("calibration"), "ann_edges_px": [float(e) for e in r.cfg.ann_edges],
                   "pxscale": float(red.pxscale), "fwhm_px": float(red.fwhm),
                   "curve": {k: cv[k] for k in keep if k in cv}}
        with open(os.path.join(out, f"fig8_{st}.json"), "w") as f:
            json.dump(summary, f, indent=1, default=str)
        log(f"== {st} done in {summary['minutes']:.1f} min: {os.path.join(out, f'fig8_{st}.json')}")
        logf.close()


if __name__ == "__main__":
    main()
