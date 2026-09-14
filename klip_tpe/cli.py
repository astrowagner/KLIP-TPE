"""Command-line entry points.

    klip-tpe near   --root <NEAR root> --nights 1 2 3 4 5 6 --run-dir runs/run_001 [...]
    klip-tpe resume --run-dir runs/run_001 --root ...          (config comes from the checkpoint)
    klip-tpe extend --run-dir runs/run_001 --root ... --n-iter 800 800 0   (per-annulus new totals)
    klip-tpe replay --root ... --idl-log optimize_tpe_results.txt --n 30
    klip-tpe testbed --nseed 8 --noise 0.25                    (synthetic density-model comparison)
    klip-tpe plots  --run-dir runs/run_001
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _near_objects(a, log=print):
    """Reducer / objective / sampler / guard / throughput for the selected instrument
    (``--instrument near`` reads the NEAR campaign tree; ``--instrument nomic`` reads a
    pyNOMIC working directory -- see ``klip_tpe.instruments.nomic``)."""
    if getattr(a, "instrument", "near") == "nomic":
        return _nomic_objects(a, log)
    if getattr(a, "instrument", "near") == "generic":
        return _generic_objects(a, log)
    if not a.root:
        raise SystemExit("--root is required for --instrument near / nomic")
    from .instruments import near
    frames = slice(0, a.max_frames) if getattr(a, "max_frames", None) else None
    root = a.root[0] if isinstance(a.root, (list, tuple)) else a.root
    from .parallel import resolve_workers
    a.workers = resolve_workers(a.workers)
    red = near.make_reducer(root, a.nights, frames=frames, use_library=not a.no_library,
                            weighting=a.weighting, max_workers=a.workers, pool=a.pool, fast=a.fast, log=log,
                            n_min_ref=a.n_min_ref, backend=getattr(a, "backend", "klip"))
    objective, sampler = near.default_config(red, clean_subtract=not a.no_clean_subtract,
                                             metric=getattr(a, "metric", "mawet"),
                                             known=[tuple(k) for k in (a.known or [])])
    guard = near.make_guard(red, n_min_ref=a.n_min_ref, ref_frac=a.ref_frac) if not a.fast else None
    red.start_workers()                    # logs the worker budget and the pool kind it got
    thr = None
    try:
        thr = near.N4Library(os.path.join(root, "psflib")).throughput
    except Exception:
        thr = near.analytic_throughput
    return red, objective, sampler, guard, thr


def _generic_objects(a, log=print):
    """Reducer / objective / sampler / guard for plain registered cubes (``--cube``)."""
    import os as _os
    from .instruments import generic
    from .parallel import resolve_workers
    if not a.cube or not a.angles or a.pxscale is None or a.lam is None:
        raise SystemExit("--instrument generic needs --cube, --angles, --pxscale and --lam")
    n = len(a.cube)
    def per(lst, what):
        if lst is None:
            return [None] * n
        if len(lst) == 1:
            return list(lst) * n
        if len(lst) != n:
            raise SystemExit(f"--{what}: give one value or one per --cube ({n})")
        return list(lst)
    angs, psfs, refs = per(a.angles, "angles"), per(a.psf, "psf"), per(a.ref_cube, "ref-cube")
    names = per(a.names, "names") if a.names else [_os.path.splitext(_os.path.basename(c))[0].replace(".fits", "") for c in a.cube]
    sflux = per(a.star_flux, "star-flux")
    a.workers = resolve_workers(a.workers)
    datasets, star_flux = {}, {}
    for c, ang, psf, ref, nm, sf in zip(a.cube, angs, psfs, refs, names, sflux):
        ds = generic.load_cube(c, ang, psf=psf, ref_cube=ref, name=nm, crop_half=a.crop_half,
                               center=tuple(a.center) if a.center else None, angle_sign=a.angle_sign,
                               tags=None if a.no_tags else "auto", wv_index=a.wv_index)
        datasets[nm] = ds
        if sf is not None:
            star_flux[nm] = generic.star_flux_from_halo(ds) if str(sf).lower() == "halo" else float(sf)
        log(f"  {nm}: {ds.cube.shape} frames, angles {ds.angles.min():.1f}..{ds.angles.max():.1f} deg"
            + (f", star flux {star_flux[nm]:.4g}" if nm in star_flux else ""))
    red = generic.make_reducer(datasets, pxscale=a.pxscale, lam_m=a.lam, diam_m=a.diam, fwhm_px=a.fwhm_px,
                               truenorth=a.truenorth, star_flux=star_flux or None, weighting=a.weighting,
                               max_workers=a.workers, pool=a.pool, fast=a.fast, n_min_ref=a.n_min_ref,
                               backend=getattr(a, "backend", "klip"), log=log)
    objective, sampler = generic.default_config(red, clean_subtract=not a.no_clean_subtract,
                                                known=[tuple(k) for k in (a.known or [])],
                                                metric=getattr(a, "metric", "mawet"))
    guard = generic.make_guard(red, n_min_ref=a.n_min_ref, ref_frac=a.ref_frac) if not a.fast else None
    return red, objective, sampler, guard, None


def _groups_spec(v):
    """``--groups``: none | auto | a list of JD split times | @file with one label per frame."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            v = v[0]
        else:
            return [float(x) for x in v]
    v = str(v)
    if v.startswith("@"):
        import numpy as _np
        return _np.loadtxt(v[1:], dtype=int).ravel()
    try:
        return [float(v)] if v.lower() not in ("auto", "none", "off") else v
    except ValueError:
        return v


def _nomic_objects(a, log=print):
    from .instruments import nomic
    objs = a.obj if isinstance(a.obj, (list, tuple)) else [a.obj]
    roots = a.root if isinstance(a.root, (list, tuple)) else [a.root]
    if len(roots) == 1 and len(objs) > 1:
        roots = roots * len(objs)
    from .parallel import resolve_workers
    a.workers = resolve_workers(a.workers)
    datasets = {}
    for root, obj in zip(roots, objs):
        datasets.update(nomic.load_pynomic(root, obj, frames=a.frames, binned=a.binned, crop_half=75 if a.crop_half is None else a.crop_half,
                                           parang_sign=a.parang_sign, pre_bin=a.pre_bin, log=log,
                                           max_frames=getattr(a, "max_frames", None),
                                           groups=_groups_spec(getattr(a, "groups", None)),
                                           group_min_frames=getattr(a, "group_min_frames", 20),
                                           group_smooth=getattr(a, "group_smooth", 100.0),
                                           name=obj if len(objs) > 1 else (a.name or obj)))
    red = nomic.make_reducer(datasets, weighting=a.weighting, max_workers=a.workers, pool=a.pool, fast=a.fast,
                             n_min_ref=a.n_min_ref, truenorth=a.truenorth, log=log,
                             fwhm_px=a.fwhm_px, diam_m=a.diam, psf=(a.psf[0] if a.psf else "airy"), r_ee_px=a.r_ee,
                             backend=getattr(a, "backend", "klip"))
    objective, sampler = nomic.default_config(red, clean_subtract=not a.no_clean_subtract,
                                              metric=getattr(a, "metric", "mawet"),
                                              known=[tuple(k) for k in (a.known or [])])
    guard = nomic.make_guard(red, n_min_ref=a.n_min_ref, ref_frac=a.ref_frac) if not a.fast else None
    return red, objective, sampler, guard, None


def _add_near_args(p):
    p.add_argument("--instrument", default="near", choices=["near", "nomic", "generic"],
                   help="near: NEAR campaign tree under --root; nomic: pyNOMIC working directory under --root; "
                        "generic: any registered cube (--cube/--angles/--psf ...)")
    p.add_argument("--root", nargs="+", default=None,
                   help="NEAR: data root. nomic: pyNOMIC working directory (one per --obj, or one for all). "
                        "generic: optional base directory for the run directory")
    gg = p.add_argument_group("generic cubes (--instrument generic / klip-tpe generic)")
    gg.add_argument("--cube", nargs="+", default=None, help="registered cube(s), (n, ny, nx) FITS; one per partition")
    gg.add_argument("--angles", nargs="+", default=None, help="derotation angles FITS (one per cube; CCW to North-up)")
    gg.add_argument("--psf", nargs="+", default=None,
                    help="generic: off-axis PSF template FITS (one, or one per cube); nomic: airy (default) | frame | gaussian")
    gg.add_argument("--ref-cube", nargs="+", default=None, help="PSF-reference cube(s) for RDI/ARDI (one per cube)")
    gg.add_argument("--names", nargs="+", default=None, help="partition names (default: cube file stems)")
    gg.add_argument("--star-flux", nargs="+", default=None,
                    help="star flux in science-frame units per cube, or 'halo' (scale the PSF template to the halo)")
    gg.add_argument("--pxscale", type=float, default=None, help="arcsec / px")
    gg.add_argument("--lam", type=float, default=None, help="wavelength (m)")
    gg.add_argument("--diam", type=float, default=8.4, help="aperture diameter (m) for lambda/D")
    gg.add_argument("--angle-sign", type=float, default=1.0, help="-1 flips the derotation-angle sign")
    gg.add_argument("--crop-half", type=int, default=None,
                    help="crop to 2h+1 px around --center (generic default: full frame; nomic default: 75)")
    gg.add_argument("--center", type=float, nargs=2, default=None, help="star pixel (x y); default: array centre")
    gg.add_argument("--wv-index", type=int, default=None, help="channel of a 4-d IFS cube")
    gg.add_argument("--no-tags", action="store_true", help="no data-derived frame-quality tags (no frame selection)")
    p.add_argument("--nights", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--backend", default="klip", choices=["klip", "pyklip", "vip"],
                   help="PSF-subtraction engine: built-in annular KLIP (default), pyKLIP or VIP (must be installed)")
    g = p.add_argument_group("pyNOMIC (--instrument nomic)")
    g.add_argument("--obj", nargs="+", default=None, help="pyNOMIC object name(s) (the <obj>_NOMIC_*.npz prefix)")
    g.add_argument("--frames", default="masked", help="frame directory under the working dir (masked | aligned)")
    g.add_argument("--binned", action="store_true", help="use pyNOMIC's <obj>_NOMIC_binned_evaluated.npz frames")
    g.add_argument("--pre-bin", type=int, default=1, help="mean-bin every N frames at load time (memory)")
    g.add_argument("--parang-sign", type=float, default=1.0, help="-1 flips the parallactic-angle sign")
    g.add_argument("--truenorth", type=float, default=0.0)
    g.add_argument("--fwhm-px", type=float, default=None, help="override the FWHM (default: Airy fit / 1.028 l/D)")
    g.add_argument("--name", default=None, help="partition-name prefix (default: --obj)")
    g.add_argument("--r-ee", type=float, default=None, help="core radius (px) for the frame PSF normalisation (1.5 l/D)")
    g.add_argument("--groups", nargs="+", default=None,
                   help="image groups as partitions: 'auto' (pyNOMIC's star-position split), JD split times, "
                        "or @file with one integer label per frame; default: no grouping (partitions = chop states)")
    g.add_argument("--group-min-frames", type=int, default=20, help="groups with fewer frames are left out")
    g.add_argument("--group-smooth", type=float, default=100.0, help="smoothing kernel (frames) of the auto split")
    p.add_argument("--max-frames", type=int, default=None, help="use only the first N frames per night (tests)")
    p.add_argument("--workers", default="auto",
                   help="worker budget: 'auto' (default) = all cores, an integer pins it, -k = all but k")
    p.add_argument("--pool", default="auto", choices=["auto", "processes", "threads"],
                   help="how the budget is spent: forked worker processes (default; the IDL bridges) or threads")
    p.add_argument("--weighting", default="equal", choices=["equal", "sqrt_texp"])
    p.add_argument("--fast", action="store_true", help="single-basis KLIP (angsep/anglemax not searched)")
    p.add_argument("--no-library", action="store_true")
    p.add_argument("--no-clean-subtract", action="store_true")
    p.add_argument("--ladder-pair", action="store_true",
                   help="two-source annulus: use the pre-2026-09-05 two-rung radial ladder instead of "
                        "both sources at the annulus' area-weighted mid radius (only to reproduce older IDL runs)")
    p.add_argument("--n-min-ref", type=int, default=10)
    p.add_argument("--ref-frac", type=float, default=0.95)
    p.add_argument("--known", type=float, nargs=2, action="append", metavar=("RHO", "PA"))
    p.add_argument("--metric", default="mawet", choices=["mawet", "fmmf"],
                   help="detection metric: 'mawet' (default) filters with the injection PSF; "
                        "'fmmf' forward-models that PSF through each configuration's own subtraction "
                        "and filters with the result (klip_tpe.fmmf)")


def _protocol_args(p):
    """Run-protocol options shared by ``near`` (A §3.1 opt_width, §5.9 scan mode, §9 products, C hooks)."""
    p.add_argument("--opt-width", action="store_true",
                   help="adaptive annuli: --ann-edges gives the first inner edge; 'width' becomes a searched dim")
    p.add_argument("--width-range", type=float, nargs=2, default=[15.0, 30.0], metavar=("W0", "W1"))
    p.add_argument("--r-cap", type=float, default=70.0, help="outer radius cap (px) for --opt-width")
    p.add_argument("--k-mode", default="search", choices=["search", "scan_rescore", "scan"],
                   help="search: k_klip searched; scan/scan_rescore: per-eval k-scan (k not searched)")
    p.add_argument("--k-scan-max", type=int, default=30)
    p.add_argument("--stitch-every", type=int, default=10, help="running-stitch cadence (evals); 0 = annulus end only")
    p.add_argument("--verify", dest="verify", action="store_true", default=False,
                   help="klip_tpe.verify report + limit curve after each annulus")
    p.add_argument("--no-verify", dest="verify", action="store_false")
    p.add_argument("--verify-n-boot", type=int, default=100)
    p.add_argument("--param-verify", dest="param_verify", action="store_true", default=None,
                   help="parameter-ensemble persistence stage (default: on when per-night blocks exist)")
    p.add_argument("--no-param-verify", dest="param_verify", action="store_false")
    p.add_argument("--n-pv", type=int, default=20)
    p.add_argument("--pv-divmin", type=float, default=0.05)
    p.add_argument("--candidates", action="store_true", help="blind candidate search on the running/final stitch")
    p.add_argument("--write-setup-files", dest="write_setup_files", action="store_true", default=True)
    p.add_argument("--no-setup-files", dest="write_setup_files", action="store_false")
    p.add_argument("--legacy-stitch", action="store_true", help="equal-weight stitch, no contrast-curve seam trim")
    p.add_argument("--display", dest="display", action="store_true", default=True,
                   help="live display: per-eval step panels, books, progress movie (default on)")
    p.add_argument("--no-display", dest="display", action="store_false")
    p.add_argument("--display-every", type=int, default=1, help="render cadence in evaluations")
    p.add_argument("--pdf-every", type=int, default=10)
    p.add_argument("--movie-every", type=int, default=None,
                   help="rebuild annulusNN/progress.gif + .mp4 every N evaluations (default: --pdf-every; 0 = annulus end only)")
    p.add_argument("--show", nargs="?", const="window", default=False,
                   help="live panel on screen: --show (a matplotlib window), --show inline (update a Jupyter output "
                        "cell in place), --show auto (inline inside a notebook, window otherwise)")
    p.add_argument("--aliens", action="store_true", help="play the IDL launch movie during the first calibration (+ intro.gif)")
    p.add_argument("--window-scale", type=float, default=1.0,
                   help="live window size as a fraction of the 1850x990 panel (1.0 = the IDL window, 1:1 pixels)")
    p.add_argument("--no-fm-curve", dest="fm_curve", action="store_false", default=True,
                   help="skip the KLIP-FM cross-check curve after each annulus (A 9.1)")
    p.add_argument("--fm-preview", dest="fm_preview", action="store_true", default=True,
                   help="live KLIP-FM preview at each new best (A 9.3; one extra reduction per new best; default on)")
    p.add_argument("--no-fm-preview", dest="fm_preview", action="store_false")


def _protocol_config(a) -> dict:
    return dict(opt_width=a.opt_width, width_range=tuple(a.width_range), r_cap=a.r_cap, k_mode=a.k_mode,
                k_scan_max=a.k_scan_max, stitch_every=a.stitch_every, verify=a.verify, verify_n_boot=a.verify_n_boot,
                param_verify=a.param_verify, n_pv=a.n_pv, pv_divmin=a.pv_divmin, candidates=a.candidates,
                write_setup_files=a.write_setup_files, legacy_stitch=a.legacy_stitch,
                partition_weighting=a.weighting, fm_curve=a.fm_curve, fm_preview=a.fm_preview,
                pair_area_midpoint=not getattr(a, "ladder_pair", False))


def build_near_space(red, a):
    """The NEAR search space plus the protocol adjustments: a searched ``width`` dim for
    ``--opt-width`` and no ``k_klip`` dims in the scan modes (k comes from the k-scan)."""
    from .instruments import near
    if getattr(a, "instrument", "near") == "generic":
        from .instruments import generic
        has_tags = all(r.frame_tags() for r in red.reducers.values())
        return generic.make_space(red, per_night=not a.global_block, k_klip_max=a.k_max,
                                  opt_framesel=has_tags and not a.no_framesel, search_angles=not a.fast,
                                  selection=None if a.no_selection else "two_slot", max_drop=getattr(a, "max_drop", 2),
                                  width_range=tuple(a.width_range) if a.opt_width else None,
                                  search_k=a.k_mode not in ("scan", "scan_rescore"))
    return near.make_space(red, per_night=not a.global_block, k_klip_max=a.k_max,
                           opt_framesel=not a.no_framesel, search_angles=not a.fast,
                           selection=None if a.no_selection else "two_slot", max_drop=getattr(a, "max_drop", 2),
                           width_range=tuple(a.width_range) if a.opt_width else None,
                           search_k=a.k_mode not in ("scan", "scan_rescore"))


def _default_run_dir(a) -> str:
    """IDL layout: ``<root>/comb/opt/run_YYYYMMDD_HHMMSS`` (generic cubes: ``./runs/run_...``)."""
    root = a.root[0] if isinstance(a.root, (list, tuple)) else a.root
    if getattr(a, "instrument", "near") == "generic":
        return os.path.join(root or "runs", time.strftime("run_%Y%m%d_%H%M%S"))
    return os.path.join(root, "comb", "opt", time.strftime("run_%Y%m%d_%H%M%S"))


def cmd_near(a):
    from .runner import CalibrationConfig, RunConfig, Runner, ValidationConfig
    if not a.run_dir:
        a.run_dir = _default_run_dir(a)
        print(f"run directory: {a.run_dir}")
    _register(a.run_dir)
    if getattr(a, "from_idl_setup", None):
        # mirror an IDL run: nights, annulus, budget, contrast, validation and TPE settings
        from .idl_compare import matched_run_config
        cfg, info = matched_run_config(a.from_idl_setup, seed=a.seed, **_protocol_config(a))
        if info["nights"] and a.nights == [1, 2, 3, 4, 5, 6]:
            a.nights = info["nights"]
        print(f"matched IDL setup {a.from_idl_setup}: nights {a.nights}, ann_edges {cfg.ann_edges}, "
              f"n_iter {cfg.n_iter}, n_init {cfg.n_init}, forced contrast {cfg.calibration.forced}, "
              f"n_top/n_valid {cfg.validation.n_top}/{cfg.validation.n_valid}, blocks {cfg.blocks}, pbest {cfg.pbest}")
        red, objective, sampler, guard, thr = _near_objects(a)
        space = build_near_space(red, a)
        space.project = guard
        r = Runner(red, space, objective, sampler, cfg, a.run_dir, throughput_fn=thr, callbacks=_callbacks(a))
        r.run()
        return
    red, objective, sampler, guard, thr = _near_objects(a)
    space = build_near_space(red, a)
    space.project = guard
    edges = list(a.ann_edges[:1]) if a.opt_width else list(a.ann_edges)
    cfg = RunConfig(ann_edges=edges, n_iter=a.n_iter, n_init=a.n_init, search_mode=a.mode,
                    seed=a.seed, contrast0=a.contrast0, blocks=a.blocks, bench_tag=a.tag,
                    calibration=CalibrationConfig(forced=a.use_contrast, ceiling=a.cmax_cal),
                    validation=ValidationConfig(n_top=a.n_top, n_valid=a.n_valid),
                    **_protocol_config(a))
    r = Runner(red, space, objective, sampler, cfg, a.run_dir, throughput_fn=thr, callbacks=_callbacks(a))
    r.run()


def _callbacks(a):
    if not getattr(a, "display", True):
        return []
    from .display import LiveDisplay
    return [LiveDisplay(a.run_dir, every=a.display_every, pdf_every=a.pdf_every, show=getattr(a, "show", False),
                        aliens=getattr(a, "aliens", False), window_scale=getattr(a, "window_scale", 1.0),
                        movie_every=getattr(a, "movie_every", None))]


def latest_run_dir(root: str) -> str:
    """The newest ``run_*`` directory under ``<root>/comb/opt`` (by name, i.e. by start time)."""
    d = os.path.join(root, "comb", "opt")
    runs = sorted(x for x in (os.listdir(d) if os.path.isdir(d) else []) if x.startswith("run_")
                  and os.path.isdir(os.path.join(d, x)))
    if not runs:
        raise SystemExit(f"no run_* directory under {d}")
    return os.path.join(d, runs[-1])


def _resolve_run_dir(a) -> str:
    """``--run-dir last`` (also ``latest`` / ``newest``) -> the most recent run under
    ``<root>/comb/opt``; anything else is taken literally.  By this point the run record
    has already had its say (:func:`_fill_from_registry`), so reaching here with nothing to
    go on means there is genuinely no run to resume -- say so, rather than failing on a
    ``None`` path deep inside ``os.path.join``."""
    if str(a.run_dir).strip().lower() in ("last", "latest", "newest"):
        root = a.root[0] if isinstance(a.root, (list, tuple)) else a.root
        if not root:
            raise SystemExit(
                "no run to resume: nothing is on record, and no --root was given to look under.\n"
                "  klip-tpe runs --root <data root>     list the runs there\n"
                "  klip-tpe resume --run-dir <dir>      resume one by name\n"
                "Runs started from this version onwards record their own configuration, so a\n"
                "bare  klip-tpe resume  will work from then on.")
        a.run_dir = latest_run_dir(root)
        print(f"resuming the latest run: {a.run_dir}")
    return a.run_dir


def _register(run_dir, argv=None) -> None:
    """Put this run on the record so ``klip-tpe runs`` can see it and ``klip-tpe resume``
    can restart it without being told how.

    ``argv`` overrides what is recorded.  A resume passes the *complete* command line it
    resolved to rather than the sparse one that was typed: recording ``klip-tpe resume
    --show`` would leave the next resume with an entry carrying no data root.
    """
    try:
        from .registry import register
        register(run_dir, argv=argv)
    except Exception:
        pass


def _explicit_dests(argv=None) -> set:
    """The option names actually typed on this command line.

    argparse fills in defaults, so ``a.nights == [1..6]`` cannot be distinguished from a
    user who typed it -- and quietly preferring the default over the value a run was
    started with would resume a three-night run as a six-night one.  Re-parsing with every
    default suppressed leaves only what was really given.
    """
    import argparse as _ap
    import copy as _copy
    try:
        ap = _build_parser()
        for act in ap._actions:
            act.default = _ap.SUPPRESS
        for sp in [x for act in ap._actions if isinstance(act, _ap._SubParsersAction)
                   for x in act.choices.values()]:
            for act in sp._actions:
                act.default = _ap.SUPPRESS
        ns = ap.parse_args(_copy.copy(argv) if argv is not None else None)
        return set(vars(ns))
    except Exception:
        return set()


def _fill_from_registry(a, argv=None) -> None:
    """Make ``klip-tpe resume`` work with no arguments: take the data root (and the rest of
    what the reducer needs) from the argv the run recorded when it started."""
    from .registry import known_runs
    import shlex
    need_root = not getattr(a, "root", None)
    want = str(getattr(a, "run_dir", "last") or "last").strip().lower()
    if not need_root and want not in ("last", "latest", "newest"):
        return
    runs = known_runs(root=(a.root[0] if isinstance(a.root, (list, tuple)) else a.root))
    if want in ("last", "latest", "newest"):
        cand = [r for r in runs if r.get("argv") and r["exists"]] or [r for r in runs if r["exists"]]
        if not cand:
            return
        rec = cand[0]
        a.run_dir = rec["run_dir"]
    else:
        rec = next((r for r in runs if r["run_dir"] == os.path.abspath(a.run_dir)), None)
        if rec is None:
            return
    if rec["state"] in ("running", "stalled"):
        raise SystemExit(
            f"{os.path.basename(rec['run_dir'])} is still running ({len(rec['pids'])} process(es), "
            f"last wrote {rec['age']:.0f}s ago).\n"
            f"Resuming it now would put two writers on the same directory.\n"
            f"  klip-tpe runs                      see what is running\n"
            f"  klip-tpe runs --stop {os.path.basename(rec['run_dir'])}   stop it first")
    if not need_root:
        return
    # The newest entry is not always the useful one: a resume that took its root from here
    # records only what was typed, so a run resumed twice would inherit an entry with no
    # root in it.  Walk back until one carries the data root, preferring the newest that
    # does -- a run restarted against moved data must still resume against the new location.
    from .registry import records_for
    prev = unknown = None
    head = "near"
    rest: list = []
    for cand_rec in records_for(rec["run_dir"]) or ([rec] if rec.get("argv") else []):
        rec_argv = cand_rec.get("argv") or []
        if len(rec_argv) < 2:
            continue
        try:
            rest, skip = [], False
            for tok in rec_argv[2:]:                   # drop program name and subcommand
                if skip:
                    skip = False
                    continue
                if tok == "--run-dir":
                    skip = True
                    continue
                if tok.startswith("--run-dir="):
                    continue
                rest.append(tok)
            # re-read it as the subcommand that created the run: a `near` command line
            # carries options `resume` does not define, and parse_known_args tolerates
            # flags that have since been renamed or removed
            head = rec_argv[1]
            cand, unk = _build_parser().parse_known_args([head, "--run-dir", a.run_dir] + rest)
        except SystemExit:
            continue
        except Exception:
            continue
        if getattr(cand, "root", None):
            prev, unknown = cand, unk
            break
    if prev is None:
        return
    if unknown:
        print("  (ignoring options this version no longer has: "
              + " ".join(unknown[:6]) + (" ..." if len(unknown) > 6 else "") + ")")
    # what this resume should itself put on the record: the complete command line it is
    # actually running, so the history stops degrading with every resume
    a._record_argv = [sys.argv[0], head, "--run-dir", a.run_dir] + rest
    given = _explicit_dests(argv)
    restored = []
    for k in ("root", "nights", "instrument", "obj", "name", "binned", "frames",
              "crop_half", "parang_sign", "pre_bin", "cube", "angles", "psf",
              "pxscale", "lam", "diam", "ref_cube", "names", "star_flux",
              "workers", "pool", "known", "no_library", "fast", "n_min_ref",
              "truenorth", "fwhm_px", "metric", "backend"):
        if k in given:                     # the user said so on this command line
            continue
        v = getattr(prev, k, None)
        if v is None or v == getattr(a, k, None):
            continue
        setattr(a, k, v)
        restored.append(k)
    if restored:
        print("  restored from the record: "
              + ", ".join(f"--{k.replace('_', '-')} {getattr(a, k)}" for k in restored[:6])
              + (" ..." if len(restored) > 6 else ""))
    print(f"resume: {a.run_dir}")


def cmd_runs(a):
    from .registry import format_table, known_runs, stop_run
    root = a.root[0] if isinstance(a.root, (list, tuple)) else a.root
    runs = known_runs(root=root)
    targets = []
    if a.stop_all:
        targets = [r for r in runs if r["pids"]]
    elif a.stop_stale:
        targets = [r for r in runs if r["state"] == "orphaned"]
    elif a.stop:
        want = [os.path.basename(str(x).rstrip("/")) for x in a.stop]
        targets = [r for r in runs if os.path.basename(r["run_dir"]) in want or r["run_dir"] in a.stop]
        missing = set(want) - {os.path.basename(r["run_dir"]) for r in targets}
        for m in missing:
            print(f"  {m}: no such run on record")
    if targets:
        print(f"stopping {len(targets)} run(s):")
        for r in targets:
            stop_run(r["run_dir"])
        runs = known_runs(root=root)
    print(format_table(runs))


def cmd_resume(a):
    from .runner import Runner
    try:
        _fill_from_registry(a, getattr(a, "_argv", None))
    except SystemExit:
        raise                      # "that run is still going" is a real refusal
    except Exception as exc:       # bookkeeping must never stand between you and a resume
        print(f"  (could not read the run record: {exc!r}; carrying on with what was given)")
    _resolve_run_dir(a)
    # record the command line this resume resolved to, not the one that was typed, so the
    # next resume inherits an entry that still carries the data root
    _register(a.run_dir, getattr(a, "_record_argv", None))
    red, objective, sampler, guard, thr = _near_objects(a)
    r = Runner.resume(a.run_dir, red, objective, sampler, project=guard, throughput_fn=thr)
    r.callbacks = _callbacks(a)
    r.run()


def cmd_extend(a):
    from .runner import Runner
    _resolve_run_dir(a)
    red, objective, sampler, guard, thr = _near_objects(a)
    Runner.extend(a.run_dir, red, objective, sampler, a.n_iter, project=guard, throughput_fn=thr)


def cmd_replay(a):
    import numpy as np
    from .idl_replay import read_idl_log, replay, summarize
    from .instruments import near
    from .runner import CalibrationConfig, RunConfig, Runner
    red, objective, sampler, guard, thr = _near_objects(a)
    log = read_idl_log(a.idl_log)
    space = near.make_space(red, k_klip_max=a.k_max, anglemax_hi=a.anglemax_hi)
    if space.names != log.names:
        print("WARNING: space names differ from the IDL log header; mapping by name where possible")
    cfg = RunConfig(ann_edges=a.ann_edges, n_iter=1, n_init=1, seed=a.seed, contrast0=a.contrast,
                    calibration=CalibrationConfig(forced=[a.contrast]))
    r = Runner(red, space, objective, sampler, cfg, a.run_dir, throughput_fn=thr, log=lambda s: None)
    r.contrast = a.contrast
    fin = np.flatnonzero(np.isfinite(log.score))
    o = fin[np.argsort(log.score[fin])]
    pick = sorted(set(int(i) for i in list(o[np.linspace(0, len(o) - 1, a.n).astype(int)])))
    res = replay(r, log, pick, out_path=os.path.join(a.run_dir, "replay.json"), n_repeat=a.repeat)
    print(json.dumps(summarize(res), indent=1))


def cmd_testbed(a):
    from .testbed import compare_density_models
    out = compare_density_models(nblock=a.nblock, bdim=a.bdim, n_iter=a.n_iter, n_init=a.n_init, nseed=a.nseed,
                                 noise=a.noise, ridge=a.ridge, transposed=a.transposed, pbest=a.pbest)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


def cmd_plots(a):
    from .plots import plot_all
    figs = plot_all(a.run_dir, save=True)
    print("wrote", ", ".join(sorted(figs)))


def cmd_compare(a):
    """Replay the evaluations of a (running) IDL run through the Python objective."""
    from .idl_compare import LiveComparison
    from .runner import CalibrationConfig, RunConfig, Runner, ValidationConfig
    red, objective, sampler, guard, thr = _near_objects(a)
    space = build_near_space(red, a)
    space.project = guard
    cfg = RunConfig(ann_edges=list(a.ann_edges), n_iter=1, n_init=1, seed=a.seed, contrast0=a.contrast,
                    calibration=CalibrationConfig(forced=[a.contrast]), validation=ValidationConfig(n_top=1, n_valid=1),
                    fm_curve=False, save_fits=False, write_setup_files=False, param_verify=False)
    r = Runner(red, space, objective, sampler, cfg, os.path.join(a.out, "_runner"), throughput_fn=thr,
               log=lambda s: None)
    r.contrast = a.contrast
    lc = LiveComparison(a.idl_run, r, a.out, annulus=a.annulus, n_repeat=a.repeat)
    if a.contrast_from_idl:
        pass                                            # LiveComparison already took it from run_setup.txt
    else:
        lc.contrast = a.contrast
        r.contrast = a.contrast
    n = lc.sync(max_new=a.max_new, stride=a.stride, offset=a.stride_offset)
    print(f"compared {n} new evaluations ({len(lc.rows)} total) -> {a.out}/compare.txt")
    st = lc.stats()
    if "pearson" in st:
        print(f"pearson {st['pearson']:.3f}  spearman {st['spearman']:.3f}  PY = {st['slope']:.3f} IDL + {st['offset']:.3f}"
              f"  median PY/IDL {st['median_ratio']:.3f}  top10 overlap {st.get('top10_overlap', float('nan')):.2f}")


def _add_run_args(p):
    """The search / protocol arguments shared by ``near`` and ``generic``."""
    p.add_argument("--run-dir", default=None,
                   help="output directory (default: <root>/comb/opt/run_YYYYMMDD_HHMMSS, the IDL layout; "
                        "generic: ./runs/run_...)")
    p.add_argument("--ann-edges", type=float, nargs="+", default=[0.0, 20.0])
    p.add_argument("--n-iter", type=int, nargs="+", default=[200])
    p.add_argument("--n-init", type=int, nargs="+", default=[50])
    p.add_argument("--mode", default="tpe", choices=["tpe", "random", "grid"])
    p.add_argument("--blocks", default="partitions", help="partitions | univariate | full")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--contrast0", type=float, default=3e-5)
    p.add_argument("--use-contrast", type=float, nargs="+", default=None, help="forced contrast per annulus (0 = calibrate)")
    p.add_argument("--cmax-cal", type=float, default=None)
    p.add_argument("--k-max", type=int, default=None)
    p.add_argument("--n-top", type=int, default=3)
    p.add_argument("--n-valid", type=int, default=8)
    p.add_argument("--global-block", action="store_true", help="one parameter block for all nights")
    p.add_argument("--no-framesel", action="store_true")
    p.add_argument("--no-selection", action="store_true", help="no drop1/drop2 night-selection dims")
    p.add_argument("--max-drop", type=int, default=2,
                   help="drop slots of the partition selection (2 = IDL drop1/drop2; more for nights x groups)")
    p.add_argument("--tag", default="none")
    p.add_argument("--from-idl-setup", default=None, metavar="RUN_SETUP_TXT",
                   help="mirror an IDL run's settings (annulus, budget, contrast, validation, TPE) for a matched Python run")
    _protocol_args(p)
    p.set_defaults(func=cmd_near)



def cmd_view(a):
    from .viewer import view
    raise SystemExit(view(run_dir=a.run_dir, root=a.root, interval=a.interval,
                          scale=a.window_scale, once=a.once))


def _build_parser():
    ap = argparse.ArgumentParser(prog="klip-tpe", description="KLIP-TPE reduction-parameter optimizer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for cmd, hlp in (("near", "run the optimizer on NEAR / pyNOMIC data (--instrument)"),
                     ("generic", "run the optimizer on any registered cube (--cube --angles --psf ...)")):
        p = sub.add_parser(cmd, help=hlp)
        _add_near_args(p)
        if cmd == "generic":
            p.set_defaults(instrument="generic")
        _add_run_args(p)
    p = sub.add_parser("compare", help="replay a (running) IDL run's evaluations through the Python objective")
    _add_near_args(p)
    p.add_argument("--idl-run", required=True, help="IDL run directory (optimize_tpe_results.txt + run_setup.txt)")
    p.add_argument("--out", required=True, help="output directory (incremental: re-run to pick up new IDL evals)")
    p.add_argument("--annulus", type=int, default=1)
    p.add_argument("--ann-edges", type=float, nargs="+", default=[0.0, 20.0])
    p.add_argument("--contrast", type=float, default=6e-5)
    p.add_argument("--contrast-from-idl", action="store_true", default=True)
    p.add_argument("--max-new", type=int, default=None, help="replay at most N new IDL evaluations this call")
    p.add_argument("--stride", type=int, default=1, help="replay every N-th IDL iteration only")
    p.add_argument("--stride-offset", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1, help="Python re-scores per configuration (fresh azimuths)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-max", type=int, default=100)
    p.add_argument("--global-block", action="store_true")
    p.add_argument("--no-framesel", action="store_true")
    p.add_argument("--no-selection", action="store_true")
    p.add_argument("--opt-width", action="store_true", default=False)
    p.add_argument("--width-range", type=float, nargs=2, default=[15.0, 30.0])
    p.add_argument("--k-mode", default="search")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("resume", help="resume a run from its checkpoint (search config comes from the checkpoint)")
    _add_near_args(p)
    p.add_argument("--run-dir", default="last",
                   help="run directory, or 'last' (default) for the newest run_* under <root>/comb/opt")
    p.add_argument("--display", dest="display", action="store_true", default=True)
    p.add_argument("--no-display", dest="display", action="store_false")
    p.add_argument("--display-every", type=int, default=1)
    p.add_argument("--pdf-every", type=int, default=10)
    p.add_argument("--movie-every", type=int, default=None)
    p.add_argument("--show", nargs="?", const="window", default=False)
    p.add_argument("--aliens", action="store_true")
    p.add_argument("--window-scale", type=float, default=1.0)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("extend", help="reopen a run with new per-annulus totals (<=0 untouched, "
                                      "<=existing re-validate only, >existing continue the search)")
    _add_near_args(p)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n-iter", type=int, nargs="+", required=True, help="new total evaluations per annulus")
    p.set_defaults(func=cmd_extend)

    p = sub.add_parser("replay", help="re-score configurations from an IDL optimize_tpe_results.txt")
    _add_near_args(p)
    p.add_argument("--idl-log", required=True)
    p.add_argument("--run-dir", default="replay_run")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--contrast", type=float, default=6e-5)
    p.add_argument("--ann-edges", type=float, nargs="+", default=[0.0, 20.0])
    p.add_argument("--k-max", type=int, default=100)
    p.add_argument("--anglemax-hi", type=float, default=154.0)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("testbed", help="synthetic density-model comparison (univariate / block / full)")
    p.add_argument("--nblock", type=int, default=6)
    p.add_argument("--bdim", type=int, default=9)
    p.add_argument("--n-iter", type=int, default=400)
    p.add_argument("--n-init", type=int, default=50)
    p.add_argument("--nseed", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.25)
    p.add_argument("--ridge", type=float, default=0.40)
    p.add_argument("--pbest", type=float, default=0.5)
    p.add_argument("--transposed", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_testbed)

    p = sub.add_parser("plots", help="diagnostic plots for a run directory")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_plots)

    p = sub.add_parser("runs", help="what runs exist, which are actually running, and stop the strays")
    p.add_argument("--root", nargs="?", default=None,
                   help="also look for run directories under this data root")
    p.add_argument("--stop", nargs="+", metavar="RUN", help="stop these runs (name or directory)")
    p.add_argument("--stop-stale", action="store_true",
                   help="stop runs that still have processes but stopped writing long ago")
    p.add_argument("--stop-all", action="store_true", help="stop every run that has processes")
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("view", help="live window for a running (or finished) run -- a separate "
                                    "process, so it cannot be starved by the optimizer")
    p.add_argument("--run-dir", default="last",
                   help="run directory, a run name under <root>/comb/opt, or 'last' (default)")
    p.add_argument("--root", default=None, help="data root, to resolve --run-dir last")
    p.add_argument("--interval", type=float, default=1.0, help="refresh period (s)")
    p.add_argument("--window-scale", type=float, default=1.0)
    p.add_argument("--once", action="store_true", help="print the run status and exit (no window)")
    p.set_defaults(func=cmd_view)

    return ap


def main(argv=None):
    a = _build_parser().parse_args(argv)
    a._argv = argv
    a.func(a)


if __name__ == "__main__":
    main()
