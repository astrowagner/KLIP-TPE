"""Command-line entry points of ``klip-tpe``.

Each command has its own ``--help``::

    klip-tpe near      --root <NEAR root> --nights 1 2 3 4 5 6 --run-dir runs/run_001 [...]
    klip-tpe generic   --cube <cube> --angles <angles> --psf <psf> --pxscale ... --lam ...
    klip-tpe resume    --run-dir runs/run_001 --root ...      (config comes from the checkpoint)
    klip-tpe extend    --run-dir runs/run_001 --root ... --n-iter 800 800 0   (new totals per annulus)
    klip-tpe fit-chain --run-dir runs/run_001 --root ... --chain chain.txt    (a line's node values)
    klip-tpe rereduce  --run-dir runs/run_001 --root ... --subtract-line nodes.txt   (other frames)
    klip-tpe rescore   --run-dir runs/run_001 --root ... --entrants-from runs/run_002
    klip-tpe replay    --root ... --idl-log optimize_tpe_results.txt --n 30
    klip-tpe compare   --root ... --idl-run <IDL run directory> --out compare
    klip-tpe testbed   --nseed 8 --noise 0.25                 (synthetic density-model comparison)
    klip-tpe plots     --run-dir runs/run_001
    klip-tpe render    --run-dir runs/run_001                 (every evaluation's panel, after the fact)
    klip-tpe runs                                             (the runs on record, and stopping strays)
    klip-tpe view      --run-dir last                         (a live window on a run)

``replay`` and ``compare`` read runs of the IDL predecessor.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, Optional


def _near_objects(a, log=print):
    """Build the reducer, objective, sampler, guard and throughput of the instrument selected.

    ``--instrument near`` reads the NEAR campaign tree, and ``nomic``, ``generic`` and ``ales`` hand over
    to :func:`_nomic_objects`, :func:`_generic_objects` and :func:`_ales_objects`.  ``--rotate`` sets the
    rotation backend first.  The NEAR reducer takes the data edits (:func:`_subtract`), the threads per
    job, the card and the KLIP Gram mode before its workers start.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.
    log : callable, default print
        Function that takes each line of the log.

    Returns
    -------
    reducer : Reducer
        The reducer, its workers started.
    objective : Objective
        The objective.
    sampler : Sampler
        The source sampler.
    guard : callable or None
        The guard on the reference frames, None with ``--fast``.
    throughput : callable
        The coronagraph's transmission, from the PSF library, else the analytic one.

    Raises
    ------
    SystemExit
        If ``--root`` is missing, or ``--drop-frames`` or the north files do not fit the data.
    """
    if getattr(a, "rotate", None):
        # before the reducer and its workers: a new run records it, a resume takes its own
        from .klip import set_rotation_backend
        set_rotation_backend(a.rotate)
    if getattr(a, "instrument", "near") == "nomic":
        return _nomic_objects(a, log)
    if getattr(a, "instrument", "near") == "generic":
        return _generic_objects(a, log)
    if getattr(a, "instrument", "near") == "ales":
        return _ales_objects(a, log)
    if not a.root:
        raise SystemExit("--root is required for --instrument near / nomic / ales")
    from .instruments import near
    frames = slice(0, a.max_frames) if getattr(a, "max_frames", None) else None
    root = a.root[0] if isinstance(a.root, (list, tuple)) else a.root
    from .parallel import resolve_workers
    a.workers = resolve_workers(a.workers)
    try:
        red = near.make_reducer(root, a.nights, frames=frames, use_library=not a.no_library,
                                weighting=a.weighting, max_workers=a.workers, pool=a.pool, fast=a.fast, log=log,
                                n_min_ref=a.n_min_ref, backend=getattr(a, "backend", "klip"),
                                cube_name=getattr(a, "cube_name", None) or None,
                                drop_frames=_drop_frames(a) or None,
                                pxscale=getattr(a, "pxscale", None), psflib=getattr(a, "psflib", None),
                                north=getattr(a, "north", None) or "files")
    except near.DropFramesError as exc:
        raise SystemExit(f"--drop-frames: {exc}")
    except near.NorthFileError as exc:
        raise SystemExit(f"{exc}. --north files takes the north files as they are now, --north constant "
                         f"{near.NEAR.truenorth:g} deg for every frame; either one no longer reproduces the run")
    _subtract(red, a, log)
    _set_threads_per_job(red, a)
    _set_card(red, a)
    _set_klip_gram(red, a, log)
    objective, sampler = near.default_config(red, clean_subtract=not a.no_clean_subtract,
                                             metric=getattr(a, "metric", "mawet"),
                                             known=[tuple(k) for k in (a.known or [])],
                                             forbidden_pa=_forbidden(a), pixel_mask=_line_mask(red, a, log))
    guard = near.make_guard(red, n_min_ref=a.n_min_ref, ref_frac=a.ref_frac) if not a.fast else None
    red.start_workers()                    # logs the worker budget and the pool kind it got
    thr = None
    try:
        thr = near.N4Library(getattr(a, "psflib", None) or os.path.join(root, "psflib")).throughput
    except Exception:
        thr = near.analytic_throughput
    return red, objective, sampler, guard, thr


def _set_threads_per_job(red, a) -> None:
    """Set ``--threads-per-job`` on a reducer, before its worker pool starts, which it sizes.

    Parameters
    ----------
    red : Reducer
        The reducer.
    a : argparse.Namespace
        The parsed command line.
    """
    if hasattr(red, "threads_per_job"):
        red.threads_per_job = max(int(getattr(a, "threads_per_job", 2) or 2), 1)


def _set_klip_gram(red, a, log=print) -> None:
    """Set ``--klip-gram`` on every KLIP reducer, before the worker pool starts.

    The workers keep the reducers they were forked with.

    Parameters
    ----------
    red : Reducer
        The reducer.
    a : argparse.Namespace
        The parsed command line.
    log : callable, default print
        Function that takes each line of the log.
    """
    from .reducer import set_klip_gram
    mode = str(getattr(a, "klip_gram", None) or "target")
    if set_klip_gram(red, mode) and mode == "zone":
        log("  KLIP: each target's reference covariance read off one Gram matrix per zone, its "
            "projection made without forming the modes (--klip-gram zone, the same KLIP to "
            "float64 rounding, not to the bit)")


def _first_partition(red):
    """Return the first partition's reducer of a partitioned reducer, or the reducer itself.

    Parameters
    ----------
    red : Reducer
        The reducer.

    Returns
    -------
    reducer : Reducer
        The first partition's reducer, or ``red``.
    """
    subs = getattr(red, "reducers", None)
    return next(iter(subs.values())) if isinstance(subs, dict) and subs else red


def _sources_file(path: str, option: str = "--subtract-sources"):
    """Read the source list of ``--subtract-sources``, ``--inject-sources`` or ``--verify-sources``.

    All three take the sources' own contrasts, as they are or will be in the data.  A list whose
    contrasts are all negative is refused, since for ``--subtract-sources`` it is most likely the
    negatives written to take a line out by injection (``c = -a step / L``), which would double the
    line.  ``--verify-sources`` takes positive contrasts only.

    Parameters
    ----------
    path : str
        Path of the list.
    option : str, default "--subtract-sources"
        The option the list comes with, for the checks and the messages.

    Returns
    -------
    sources : list of Source
        The sources.

    Raises
    ------
    SystemExit
        If the file cannot be read, holds no sources or fails the option's check of the contrasts.
    """
    from .injection import read_sources
    try:
        lst = read_sources(path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{option}: {exc}")
    if not lst:
        raise SystemExit(f"{option}: no sources in {path}")
    base = os.path.basename(path)
    if option == "--verify-sources":
        if any(not s.contrast > 0 for s in lst):
            raise SystemExit(f"--verify-sources: every source in {path} needs a positive contrast, the contrast it "
                             f"is injected at")
        return lst
    if all(s.contrast <= 0 for s in lst) and any(s.contrast < 0 for s in lst):
        if option == "--inject-sources":
            raise SystemExit(f"--inject-sources: every contrast in {path} is negative.  The option adds the sources "
                             f"with their own contrasts, as they will be in the data, so these would take light out "
                             f"of the frames.  To take sources out, give their own contrasts to --subtract-sources.")
        raise SystemExit(f"--subtract-sources: every contrast in {path} is negative.  The option takes the sources' "
                         f"own contrasts, as they are in the data (positive for a planet or a line), and subtracts "
                         f"them.  A list of the negatives to inject, written to take a line out by injection "
                         f"(c = -a step / L), has the opposite sign, and subtracting it would double the line.  If "
                         f"that is what {base} holds, flip its signs, e.g. awk '{{print $1, $2, -$3}}' {base}.  To "
                         f"add sources to the frames, give their own contrasts to --inject-sources.")
    return lst


def _subtract_request(red, a):
    """Return the sources to take out of the frames, and the notes for the record.

    On a command that types neither option they are what the checkpoint recorded
    (:func:`_data_edits_for_run`).  Otherwise they are the sources of ``--subtract-sources`` and the
    chain ``--subtract-line`` describes, built with the first partition's injection model
    (:func:`klip_tpe.chainfit.line_chain`).

    Parameters
    ----------
    red : Reducer
        The reducer.
    a : argparse.Namespace
        The parsed command line.

    Returns
    -------
    sources : list of Source
        The sources to subtract.
    meta : dict
        The notes for the record, ``subtract_file`` and ``subtract_line``.

    Raises
    ------
    SystemExit
        If the line's file cannot be read.
    """
    restored = getattr(a, "_restored_sources", None)
    if restored is not None:
        return list(restored), dict(getattr(a, "_restored_meta", None) or {})
    src, meta = [], {}
    path = getattr(a, "subtract_sources", None)
    if path:
        src += _sources_file(path)
        meta["subtract_file"] = os.path.abspath(path) if os.path.exists(path) else str(path)
    lpath = getattr(a, "subtract_line", None)
    if lpath:
        from .chainfit import line_chain, read_line_nodes
        try:
            spec = read_line_nodes(lpath)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"--subtract-line: {exc}")
        first = _first_partition(red)
        src += line_chain(spec, first.model, first.pxscale, tuple(first.data.cube.shape[-2:]), fallback=first.fallback)
        meta["subtract_line"] = dict(spec, file=os.path.abspath(lpath))
    return src, meta


def _inject_request(a):
    """Return the sources to add to the frames, and the notes for the record.

    They are those of ``--inject-sources``, or the checkpoint's list on a command that does not type the
    option (:func:`_data_edits_for_run`).

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Returns
    -------
    sources : list of Source
        The sources to inject.
    meta : dict
        The notes for the record, ``inject_file``.
    """
    restored = getattr(a, "_restored_injected", None)
    if restored is not None:
        return list(restored), dict(getattr(a, "_restored_inject_meta", None) or {})
    path = getattr(a, "inject_sources", None)
    if not path:
        return [], {}
    return _sources_file(path, "--inject-sources"), {
        "inject_file": os.path.abspath(path) if os.path.exists(path) else str(path)}


def _subtract(red, a, log=print) -> None:
    """Apply ``--subtract-sources`` and ``--subtract-line``, then ``--inject-sources``, to a reducer just built.

    They go in before its worker processes start, and the reducer's ``data_edits`` records them.

    Parameters
    ----------
    red : Reducer
        The reducer.
    a : argparse.Namespace
        The parsed command line.
    log : callable, default print
        Function that takes each line of the log.
    """
    from .reducer import _taken_text
    src, meta = _subtract_request(red, a)
    if src:
        n = red.subtract_sources(src)
        red.data_edits = dict(getattr(red, "data_edits", None) or {}, **meta)
        c = [s.contrast for s in src]
        log(f"  subtracted {n} source(s) from every partition's frames, once, at load: rho "
            f"{min(s.rho for s in src):.3f}-{max(s.rho for s in src):.3f}\", contrast {min(c):.3g} to {max(c):.3g}"
            + (" (the run's own list, from its checkpoint)" if getattr(a, "_restored_sources", None) is not None
               else ""))
        took = _taken_text(red)
        if took:
            log(f"    the fraction of their flux that left the frames, on a sample of frames: {took}")
    add, meta = _inject_request(a)
    if add:
        n = red.inject_sources(add)
        red.data_edits = dict(getattr(red, "data_edits", None) or {}, **meta)
        c = [s.contrast for s in add]
        log(f"  added {n} source(s) to every partition's frames, once, at load: rho "
            f"{min(s.rho for s in add):.3f}-{max(s.rho for s in add):.3f}\", contrast {min(c):.3g} to {max(c):.3g}"
            + (" (the run's own list, from its checkpoint)" if getattr(a, "_restored_injected", None) is not None
               else ""))
        took = _taken_text(red, "injected_kept")
        if took:
            log(f"    the fraction of their flux that reached the frames, on a sample of frames: {took}")


def _no_cube_name(a, what: str) -> None:
    """Refuse ``--cube-name`` on a command that reads its cubes another way.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.
    what : str
        The command or instrument, for the message.

    Raises
    ------
    SystemExit
        If ``--cube-name`` is given.
    """
    if getattr(a, "cube_name", None):
        raise SystemExit(f"--cube-name names the cube in each NEAR night's directory; {what} reads its cubes "
                         f"another way")


def _parse_drop_frames(values) -> dict:
    """Parse ``--drop-frames NIGHT:INDEX[,INDEX...]`` values into ``{"n4": [551], ...}``.

    ``n4:551`` and ``4:551`` both name frame 551 of night 4, counted from 0 in the night's cube, and the
    option may repeat.  ``none`` alone is the empty list, which drops nothing.

    Parameters
    ----------
    values : str or sequence of str
        The option's values.

    Returns
    -------
    frames : dict of str to list of int
        The sorted indices of each night, the nights in order.

    Raises
    ------
    SystemExit
        If a value is malformed, an index is negative, or ``none`` comes with frames.
    """
    out = {}
    nones = []
    for v in ([values] if isinstance(values, str) else (values or [])):
        s = str(v).strip()
        if s.lower() == "none":
            nones.append(s)
            continue
        night, sep, rest = s.partition(":")
        lab = night.strip().lower()
        lab = lab[1:] if lab.startswith("n") else lab
        if not sep or not lab.isdigit():
            raise SystemExit(f"--drop-frames {s}: give the night, a colon and the frames' indices, "
                             f"NIGHT:INDEX[,INDEX...], e.g. n4:551")
        try:
            idx = [int(x) for x in rest.split(",") if x.strip()]
        except ValueError:
            raise SystemExit(f"--drop-frames {s}: the frames are given by their indices, whole numbers, "
                             f"e.g. n4:551,552")
        if not idx:
            raise SystemExit(f"--drop-frames {s}: no frame index after the colon")
        if any(i < 0 for i in idx):
            raise SystemExit(f"--drop-frames {s}: the indices count from 0 in the night's cube")
        out.setdefault(f"n{int(lab)}", set()).update(idx)
    if nones and out:
        raise SystemExit("--drop-frames none drops nothing, and cannot go with a list of frames")
    return {k: sorted(out[k]) for k in sorted(out, key=lambda x: int(x[1:]))}


def _drop_frames(a):
    """Return ``--drop-frames`` on this command line, or the run's list restored in its place.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Returns
    -------
    frames : dict of str to list of int or None
        The frames, as :func:`_parse_drop_frames` gives them, None when the option is not given.
    """
    vals = getattr(a, "drop_frames", None)
    return None if not vals else _parse_drop_frames(vals)


def _drop_argv(d) -> list:
    """Return a ``dropped_frames`` record as ``--drop-frames`` values, ``["n4:551", ...]``.

    Parameters
    ----------
    d : dict
        The record.

    Returns
    -------
    values : list of str
        The values.
    """
    from .runner import _dropped
    return [f"{k}:{','.join(str(i) for i in v)}" for k, v in _dropped(d).items()]


def _no_drop_frames(a, what: str) -> None:
    """Refuse ``--drop-frames`` on a command that does not take it.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.
    what : str
        The command or instrument, for the message.

    Raises
    ------
    SystemExit
        If ``--drop-frames`` gives frames.
    """
    if _drop_frames(a):
        raise SystemExit(f"--drop-frames leaves frames out of a NEAR night at load, and {what} does not take it")


def _forbidden(a):
    """Return ``--forbidden-pa`` as a list of ``(pa_deg, halfwidth_deg)`` tuples.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Returns
    -------
    bands : list of tuple of 2 floats
        The bands of position angle where no source is injected, empty without the option.
    """
    return [tuple(b) for b in (getattr(a, "forbidden_pa", None) or [])]


def _line_mask(red, a, log=print):
    """Return ``--mask-line PA HALFWIDTH`` as a pixel mask, the strips kept out of the noise apertures.

    Parameters
    ----------
    red : Reducer
        The reducer.
    a : argparse.Namespace
        The parsed command line.
    log : callable, default print
        Function that takes each line of the log.

    Returns
    -------
    mask : ndarray of bool or None
        The mask of the reducer's frame, True to exclude, None without the option.
    """
    lines = [tuple(float(v) for v in b) for b in (getattr(a, "mask_line", None) or [])]
    if not lines:
        return None
    from .metrics import line_mask
    first = _first_partition(red)
    shape = tuple(first.data.cube.shape[-2:])
    m = line_mask(shape, lines, angle_convention=getattr(first, "angle_convention", "pa"))
    log(f"  noise apertures: {int(m.sum())} pixels along "
        + ", ".join(f"PA {pa:g} deg (+-{hw:g} px)" for pa, hw in lines) + " left out (--mask-line)")
    return m


def _constants_for_run(a) -> None:
    """Settle ``--pxscale``, ``--psflib`` and ``--north`` for a NEAR command on a run directory.

    Each is what this command line says, else what the run was made with (``run_setup.json``'s reducer
    ``provenance``), else, for a run made before 2026-11-01, which recorded none of them,
    ``NEAR.legacy_pxscale``, ``ROOT/psflib`` and the constant north.  A new run takes ``NEAR.pxscale``
    and the nights' north files.

    The constants decide where every source is injected and how every S/N is measured, so a resume, an
    extension, a re-reduction and a check reproduce the run only with its own.  A recorded library that
    is not on this machine is refused rather than replaced, and so is a recorded north file that is
    gone or has changed (when the reducer is built).

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line, changed in place.

    Raises
    ------
    SystemExit
        If the run was made with another instrument, or its PSF library is not on this machine.
    """
    import json
    from .instruments.near import NEAR
    if getattr(a, "instrument", "near") not in (None, "near"):
        return
    given = _explicit_dests(getattr(a, "_argv", None))
    run_dir = getattr(a, "run_dir", None)
    setup = os.path.join(run_dir, "run_setup.json") if run_dir else None
    existed = bool(setup and os.path.exists(setup))
    prov = None
    if existed:
        try:
            with open(setup) as f:
                prov = (json.load(f).get("reducer") or {}).get("provenance") or None
        except (OSError, ValueError):
            prov = None
    other = _run_instrument(run_dir) if existed else None
    if other not in (None, "near"):
        raise SystemExit(f"{os.path.basename(os.path.abspath(str(run_dir)))} was made with --instrument {other} "
                         f"(run_setup.json), not NEAR: give --instrument {other} and its data arguments")
    prov = prov if isinstance(prov, dict) and prov.get("instrument") == "NEAR" else None
    if "pxscale" not in given:
        if prov is not None and prov.get("pxscale") is not None:
            a.pxscale = float(prov["pxscale"])
        elif existed:
            a.pxscale = float(NEAR.legacy_pxscale)
            print(f"  --pxscale {a.pxscale:g}: the run predates the recorded scale (run_setup.json)")
        else:
            a.pxscale = None
    if "psflib" not in given:
        lib = (prov or {}).get("psflib")
        path = lib.get("path") if isinstance(lib, dict) else None
        if path and not os.path.isdir(path):
            raise SystemExit(f"the run was made with the PSF library {path}, which is not on this machine: "
                             f"give --psflib")
        a.psflib = path or None
    if "north" not in given:
        rec = (prov or {}).get("north")
        if isinstance(rec, dict) and rec:
            a.north = rec
        elif existed:
            a.north = "constant"
            print(f"  --north constant: the run predates per-frame north (run_setup.json), "
                  f"{NEAR.truenorth:g} deg for every frame")
        else:
            a.north = "files"
    if existed and ("pxscale" in given or "psflib" in given or "north" in given):
        print("  --pxscale / --psflib / --north given: this command may not reproduce the run")


def _idl_constants(a) -> None:
    """Set the constants of the IDL predecessor's ``reduce_near_2.pro`` for a command that mirrors one of its runs.

    Such commands are ``replay``, ``compare`` and a new ``near --from-idl-setup``.  The constants are
    0.0456"/px and a north of 36.5 degrees for every frame, unless typed, since the predecessor
    derotates by one north and has not taken the 2026 plate scale.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line, changed in place.
    """
    from .instruments.near import NEAR
    given = _explicit_dests(getattr(a, "_argv", None))
    if "pxscale" not in given:
        a.pxscale = float(NEAR.legacy_pxscale)
    if "north" not in given:
        a.north = "constant"
    print(f"  the IDL's constants: --pxscale {a.pxscale:g}, --north {a.north if isinstance(a.north, str) else 'as given'}")


def _data_edits_for_run(a) -> None:
    """Settle the data edits of a command on a run directory that holds a checkpoint.

    They are ``--cube-name``, ``--drop-frames``, ``--subtract-sources`` and ``--inject-sources``.  A
    run's evaluations were made on particular frames, so a later command on it reduces the same ones.
    What the checkpoint recorded (``RunConfig.data_edits``) is used where an option is not typed, the
    sources restored as ``a._restored_sources`` and ``a._restored_injected``.  A typed value that
    differs from the record is refused rather than mixed in.  ``klip-tpe rereduce``, which only reads
    the run, uses :func:`_data_edits_for_rereduce`.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line, changed in place.

    Raises
    ------
    SystemExit
        If a typed option differs from the record.
    """
    run_dir = getattr(a, "run_dir", None)
    if not run_dir or not os.path.isdir(str(run_dir)):
        return
    from .runner import checkpoint_candidates, load_checkpoint
    if not checkpoint_candidates(run_dir):
        return
    st, _ = load_checkpoint(run_dir, log=lambda s: None)
    if st is None:
        return
    rec = dict((st.get("config") or {}).get("data_edits") or {})
    given = _explicit_dests(getattr(a, "_argv", None))
    name = os.path.basename(os.path.abspath(run_dir))
    was_cube = rec.get("cube_name") or None
    if "cube_name" in given:
        if (a.cube_name or None) != was_cube:
            raise SystemExit(f"{name} was made on {was_cube or 'the default cube'}, and this command asks for "
                             f"--cube-name {a.cube_name}.  A resume cannot change the frames its evaluations were "
                             f"made on: leave the option out, or start a new run directory.")
    elif was_cube:
        a.cube_name = was_cube
        print(f"  --cube-name {was_cube}, as {name} was started (its checkpoint)")
    from .runner import _dropped, _dropped_text
    was_drop = _dropped(rec.get("dropped_frames"))
    if "drop_frames" in given:
        now = _parse_drop_frames(a.drop_frames)
        if now != was_drop:
            raise SystemExit(f"{name} was made with {_dropped_text(was_drop)}, and this command asks for "
                             f"{_dropped_text(now)} (--drop-frames {' '.join(a.drop_frames)}).  A resume cannot "
                             f"change the frames its evaluations were made on: leave the option out (the "
                             f"checkpoint's list is used), or start a new run directory.")
    else:
        a.drop_frames = _drop_argv(was_drop) or None
        if a.drop_frames:
            print(f"  --drop-frames {' '.join(a.drop_frames)}, as {name} was started (its checkpoint)")
    was = [[float(v) for v in s] for s in (rec.get("subtracted_sources") or [])]
    typed_src, typed_line = "subtract_sources" in given, "subtract_line" in given
    if typed_src and not typed_line:
        now = [list(s.as_tuple()) for s in _sources_file(a.subtract_sources)]
        if now != was:
            raise SystemExit(f"{name} was made with {len(was)} source(s) subtracted, and --subtract-sources "
                             f"{a.subtract_sources} lists {'others' if now and was else len(now)}.  A resume cannot "
                             f"change the frames its evaluations were made on: leave the option out (the "
                             f"checkpoint's list is used), or start a new run directory.")
    elif not typed_src and not typed_line and was:
        from .metrics import Source
        a._restored_sources = [Source(*s) for s in was]
        a._restored_meta = {k: rec[k] for k in ("subtract_file", "subtract_line") if rec.get(k)}
        print(f"  --subtract-sources: the {len(was)} source(s) {name} was started with (its checkpoint)")
    # a typed --subtract-line is built from the reducer's model, and the Runner compares the
    # result with the record before anything is reduced
    was_add = [[float(v) for v in s] for s in (rec.get("injected_sources") or [])]
    if "inject_sources" in given:
        now = [list(s.as_tuple()) for s in _sources_file(a.inject_sources, "--inject-sources")]
        if now != was_add:
            raise SystemExit(f"{name} was made with {len(was_add)} source(s) added, and --inject-sources "
                             f"{a.inject_sources} lists {'others' if now and was_add else len(now)}.  A resume "
                             f"cannot change the frames its evaluations were made on: leave the option out (the "
                             f"checkpoint's list is used), or start a new run directory.")
    elif was_add:
        from .metrics import Source
        a._restored_injected = [Source(*s) for s in was_add]
        a._restored_inject_meta = {k: rec[k] for k in ("inject_file",) if rec.get(k)}
        print(f"  --inject-sources: the {len(was_add)} source(s) {name} was started with (its checkpoint)")


def _data_edits_for_rereduce(a) -> None:
    """Settle the frames ``klip-tpe rereduce`` reduces a finished run's winners on.

    Each option is what this command line types, else what the run was made on (its checkpoint).  A
    typed ``--subtract-sources`` or ``--subtract-line`` replaces the run's subtraction as a whole, a
    typed ``--inject-sources`` its added sources, and a typed ``--drop-frames`` its dropped frames
    (``none`` for none).  Nothing is refused, since the run is only read.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line, changed in place.
    """
    run_dir = getattr(a, "run_dir", None)
    rec = {}
    if run_dir and os.path.isdir(str(run_dir)):
        from .runner import checkpoint_candidates, load_checkpoint
        if checkpoint_candidates(run_dir):
            st, _ = load_checkpoint(run_dir, log=lambda s: None)
            rec = dict(((st or {}).get("config") or {}).get("data_edits") or {})
    from .metrics import Source
    given = _explicit_dests(getattr(a, "_argv", None))
    name = os.path.basename(os.path.abspath(str(run_dir or ".")))
    if "cube_name" not in given:
        a.cube_name = rec.get("cube_name") or None
        if a.cube_name:
            print(f"  --cube-name {a.cube_name}, as {name} was made (its checkpoint)")
    if "drop_frames" not in given:
        a.drop_frames = _drop_argv(rec.get("dropped_frames")) or None
        if a.drop_frames:
            print(f"  --drop-frames {' '.join(a.drop_frames)}, as {name} was made (its checkpoint)")
    if not ({"subtract_sources", "subtract_line"} & given):
        a.subtract_sources = a.subtract_line = None
        was = [[float(v) for v in s] for s in (rec.get("subtracted_sources") or [])]
        if was:
            a._restored_sources = [Source(*s) for s in was]
            a._restored_meta = {k: rec[k] for k in ("subtract_file", "subtract_line") if rec.get(k)}
            print(f"  --subtract-sources: the {len(was)} source(s) {name} was made with (its checkpoint)")
    else:
        for k in ("subtract_sources", "subtract_line"):
            if k not in given:
                setattr(a, k, None)
    if "inject_sources" not in given:
        a.inject_sources = None
        was_add = [[float(v) for v in s] for s in (rec.get("injected_sources") or [])]
        if was_add:
            a._restored_injected = [Source(*s) for s in was_add]
            a._restored_inject_meta = {k: rec[k] for k in ("inject_file",) if rec.get(k)}
            print(f"  --inject-sources: the {len(was_add)} source(s) {name} was made with (its checkpoint)")


_KLIP_GRAM_FILE = "klip_gram.json"


def _klip_gram_for_run(a, record: bool = True) -> None:
    """Settle ``--klip-gram`` for a command on a run directory.

    It is what this command line says, else what the run has been using (``<run>/klip_gram.json``),
    else ``target``.  The two modes give the same KLIP to float64 rounding but not to the bit, so a run
    keeps its mode across resumes, however they are typed.  A command line that changes it writes the
    new mode there with the time, and every later resume reads it back.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line, changed in place.
    record : bool, default True
        Whether a change is written to ``klip_gram.json``.  False keeps it for this command only.
    """
    import json
    import time
    run_dir = getattr(a, "run_dir", None)
    path = os.path.join(run_dir, _KLIP_GRAM_FILE) if run_dir else None
    rec = None
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            rec = None
    was = str((rec or {}).get("klip_gram", "target"))
    if "klip_gram" not in _explicit_dests(getattr(a, "_argv", None)):
        a.klip_gram = was
        if rec is not None and was != "target":
            print(f"  --klip-gram {was}, as the run has been using since "
                  f"{(rec.get('history') or [{}])[-1].get('since', '?')} ({_KLIP_GRAM_FILE})")
        return
    if path is None or a.klip_gram == was:
        return
    if not record:
        print(f"  --klip-gram {a.klip_gram} for this command only (the run used {was}; not recorded)")
        return
    hist = list((rec or {}).get("history") or [])
    hist.append({"klip_gram": a.klip_gram, "since": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "previous": was})
    os.makedirs(run_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"klip_gram": a.klip_gram, "history": hist}, f, indent=1)
    print(f"  --klip-gram {a.klip_gram} from now on (it was {was}), recorded in "
          f"{os.path.join(os.path.basename(run_dir), _KLIP_GRAM_FILE)} for every later resume")


_CARD_FILE = "card.json"

#: the card's share of ``--gpu-share auto``, in percent.  On NEAR2 (patrick, 2026-11-15, OpenCV
#: rotation, 3 card workers) 12 never ran slower than 30 and halved the search's time at small
#: bins.  It was 30 before then.
AUTO_CARD_SHARE = 12.0


def _auto_card_share() -> float:
    """Return the card's share of ``--gpu-share auto``.

    Returns
    -------
    share : float
        :data:`AUTO_CARD_SHARE` where Linux shows a card and CuPy is installed (found, not imported,
        since the parent never loads CUDA), else 0.
    """
    import importlib.util
    from .parallel import card_present
    from .reducer import _gpu_card_factory
    try:
        cupy = importlib.util.find_spec("cupy") is not None
    except (ImportError, ValueError):
        cupy = False
    return AUTO_CARD_SHARE if card_present() and cupy and _gpu_card_factory() is not None else 0.0


def _card_share_value(v):
    """Parse ``--gpu-share`` as a percentage.

    Parameters
    ----------
    v : str, float or None
        The option's value.

    Returns
    -------
    share : float or None
        The percentage, None for ``auto`` or no value.

    Raises
    ------
    SystemExit
        If the value is not a number from 0 to 100 or ``auto``.
    """
    if v is None or str(v).strip().lower() == "auto":
        return None
    try:
        f = float(v)
    except ValueError:
        raise SystemExit(f"--gpu-share {v!r}: a percentage from 0 to 100, or auto")
    if not 0.0 <= f <= 100.0:
        raise SystemExit(f"--gpu-share {v!r}: a percentage from 0 to 100, or auto")
    return f


def _card_device_value(v):
    """Parse ``--gpu-device`` as one card.

    Parameters
    ----------
    v : str or None
        The option's value.

    Returns
    -------
    device : str or None
        The card, as ``CUDA_VISIBLE_DEVICES`` takes it, None for no value.

    Raises
    ------
    SystemExit
        If the value names more than one card or none.
    """
    if v is None:
        return None
    d = str(v).strip()
    if not d or "," in d:
        raise SystemExit(f"--gpu-device {v!r}: one card, as CUDA_VISIBLE_DEVICES takes it (an index in CUDA's "
                         f"order, or a UUID from nvidia-smi -L)")
    return d


def _card_for_run(a, record: bool = True) -> None:
    """Settle the card's share, workers, rule and device for a command that searches a run directory.

    Each of ``--gpu-share``, ``--gpu-workers`` and ``--gpu-device`` is what this command line says, else
    what the run has been using (``<run>/card.json``), else ``auto``, 3 workers and the card
    ``CUDA_VISIBLE_DEVICES`` names, if any.  The share and its rule decide which draws the card reduces,
    and the card's images differ from the CPU's at about 1e-5 of the noise, so a run keeps them, and its
    card, across resumes, however they are typed.  A command line that changes one writes the new values
    there with the time.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line, changed in place: ``gpu_share``, ``gpu_workers``, ``gpu_device``,
        ``gpu_rule`` and ``gpu_moved``, whether this command line names the card.
    record : bool, default True
        Whether a change is written to ``card.json``.

    Raises
    ------
    SystemExit
        If ``--gpu-share`` is not a percentage or ``--gpu-device`` not one card.

    Notes
    -----
    The rule is ``"trial"`` for a new record and ``"job"`` for a record written before 2026-11-15, which
    has none, so such a run picks its card's jobs as it did.  A command line that names ``--gpu-share``
    moves the run to ``"trial"``.
    """
    import json
    import time
    run_dir = getattr(a, "run_dir", None)
    path = os.path.join(run_dir, _CARD_FILE) if run_dir else None
    rec = None
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            rec = None
    if rec is not None and not isinstance(rec, dict):
        rec = None
    given = _explicit_dests(getattr(a, "_argv", None))
    if "gpu_share" in given:
        share = _card_share_value(a.gpu_share)
        share = _auto_card_share() if share is None else share
    elif rec is not None:
        share = float(rec.get("gpu_share", 0.0) or 0.0)
    else:
        share = _auto_card_share()
    if "gpu_workers" in given and a.gpu_workers is not None:
        workers = max(int(a.gpu_workers), 1)
    else:
        workers = int((rec or {}).get("gpu_workers", 3) or 3)
    env = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip() or None
    moved = "gpu_device" in given and getattr(a, "gpu_device", None) is not None
    if moved:
        device = _card_device_value(a.gpu_device)
    elif (rec or {}).get("gpu_device"):
        device = str(rec["gpu_device"])
        if env is not None and env != device:
            print(f"  --gpu-device {device}, as the run has been using ({_CARD_FILE}), not CUDA_VISIBLE_DEVICES={env}")
    else:
        device = env
    was = None
    if rec is not None:
        was = {"gpu_share": float(rec.get("gpu_share", 0.0) or 0.0), "gpu_workers": int(rec.get("gpu_workers", 3) or 3),
               "gpu_device": rec.get("gpu_device"), "rule": str(rec.get("rule", "job"))}
    rule = "trial" if was is None or "gpu_share" in given else was["rule"]
    a.gpu_share, a.gpu_workers, a.gpu_device, a.gpu_rule, a.gpu_moved = share, workers, device, rule, moved
    now = {"gpu_share": share, "gpu_workers": workers, "gpu_device": device, "rule": rule}

    def opts(d):
        return (f"--gpu-share {d['gpu_share']:g} --gpu-workers {d['gpu_workers']}"
                + (f" --gpu-device {d['gpu_device']}" if d.get("gpu_device") else ""))
    job = ", each job picked by its own slot as before 2026-11-15" if rule == "job" else ""
    if was is not None and all(now[k] == was[k] for k in ("gpu_share", "gpu_workers", "rule")) \
            and (device is None or device == was["gpu_device"]):
        if share > 0:
            print(f"  {opts(now)}, as the run has been using ({_CARD_FILE}){job}")
        return
    if path is None or not record or (was is None and share <= 0 and "gpu_share" not in given):
        return
    hist = list((rec or {}).get("history") or [])
    hist.append(dict(now, since=time.strftime("%Y-%m-%dT%H:%M:%S%z"), previous=was))
    out = {k: v for k, v in now.items() if v is not None}
    for k in ("gpu_name",):                  # what the card said it is, kept by _card_came_up
        if (rec or {}).get(k):
            out[k] = rec[k]
    out["history"] = hist
    os.makedirs(run_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    if share > 0 or was is not None:
        switch = ", each trial sending the card its share (it was job by job)" \
            if was is not None and was["rule"] != rule else job
        env_note = ", the card from CUDA_VISIBLE_DEVICES" \
            if device is not None and device == env and not moved and not (was or {}).get("gpu_device") else ""
        print(f"  {opts(now)}" + (f" from now on (it was {opts(was)})" if was is not None else "") + switch + env_note
              + f", recorded in {os.path.join(os.path.basename(run_dir), _CARD_FILE)} for every later resume")


def _card_came_up(path: str, moved: bool, info: dict):
    """Record the card's name in ``card.json`` as it comes up, and say when it is not the run's.

    The pool calls it once, from its result thread, with what the first card worker reported at
    ``warm()`` (``klip_tpe.parallel.CardSpec.on_ready``).

    Parameters
    ----------
    path : str
        The run's ``card.json``.
    moved : bool
        Whether this command line names the card (``--gpu-device``), when another name than the record's
        is the one asked for.
    info : dict
        The card's report, ``name`` among it.

    Returns
    -------
    line : str or None
        A line for the log when the card is not the one the run has been using, or the record changed.

    Notes
    -----
    It writes ``gpu_name`` into ``card.json`` the first time, and replaces it when ``moved``.  Two cards
    of the same model have the same name, which the check cannot tell apart.
    """
    import json
    import time
    name = str((info or {}).get("name") or "").strip()
    if not name or name == "numpy":
        return None
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict):
        return None
    old = rec.get("gpu_name")
    if old == name:
        return None
    if old and not moved:
        return (f"  card: this is {name}, but the run has been reducing on {old} ({_CARD_FILE}).  "
                f"--gpu-device picks the card, or names this one for the rest of the run")
    rec["gpu_name"] = name
    if old:
        hist = list(rec.get("history") or [])
        hist.append({"gpu_name": name, "since": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "previous": {"gpu_name": old}})
        rec["history"] = hist
    try:
        with open(path, "w") as f:
            json.dump(rec, f, indent=1)
    except OSError:
        return None
    return f"  card: now on {name} (it was {old}), recorded in {_CARD_FILE}" if old else None


def _set_card(red, a) -> None:
    """Set the card's share, workers, rule and device on a reducer, before its worker pool starts.

    A command that did not settle them for a run (:func:`_card_for_run`) gets no card.  With a run
    directory, the pool records the card's name in ``card.json`` as it comes up (:func:`_card_came_up`).

    Parameters
    ----------
    red : Reducer
        The reducer.
    a : argparse.Namespace
        The parsed command line.
    """
    if not hasattr(red, "card_share"):
        return
    share = getattr(a, "gpu_share", None)
    red.card_share = float(share) if isinstance(share, (int, float)) else 0.0
    red.card_workers = max(int(getattr(a, "gpu_workers", None) or 3), 1)
    red.card_rule = str(getattr(a, "gpu_rule", None) or "trial")
    dev = getattr(a, "gpu_device", None)
    red.card_device = None if dev is None or not isinstance(share, (int, float)) else str(dev)
    run_dir = getattr(a, "run_dir", None)
    if red.card_share > 0 and run_dir and os.path.exists(os.path.join(run_dir, _CARD_FILE)):
        import functools
        red.card_on_ready = functools.partial(_card_came_up, os.path.join(run_dir, _CARD_FILE),
                                              bool(getattr(a, "gpu_moved", False)))
    else:
        red.card_on_ready = None


def _generic_objects(a, log=print):
    """Build the reducer, objective, sampler and guard of plain registered cubes (``--cube``).

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.
    log : callable, default print
        Function that takes each line of the log.

    Returns
    -------
    reducer : Reducer
        The reducer.
    objective : Objective
        The objective.
    sampler : Sampler
        The source sampler.
    guard : callable or None
        The guard on the reference frames, None with ``--fast``.
    throughput : None
        No transmission, which this instrument does not give.

    Raises
    ------
    SystemExit
        If ``--cube``, ``--angles``, ``--pxscale`` or ``--lam`` is missing, a per-cube option gives
        another number of values than cubes, or ``--star-flux halo`` is given.
    """
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
            if str(sf).lower() == "halo":
                raise SystemExit(
                    "--star-flux halo has been removed.  Fitting an off-axis template to the "
                    "science halo was never valid on coronagraphic or occulted data, and on the "
                    "beta Pic AGPM sequence it returned fluxes 5-8x apart on two runs of the same "
                    "method.  Give a measured number, or leave --star-flux off and read the "
                    "contrast axis as template units.")
            star_flux[nm] = float(sf)
        log(f"  {nm}: {ds.cube.shape} frames, angles {ds.angles.min():.1f}..{ds.angles.max():.1f} deg"
            + (f", star flux {star_flux[nm]:.4g}" if nm in star_flux else ""))
    red = generic.make_reducer(datasets, pxscale=a.pxscale, lam_m=a.lam, diam_m=a.diam, fwhm_px=a.fwhm_px,
                               truenorth=a.truenorth, star_flux=star_flux or None, weighting=a.weighting,
                               max_workers=a.workers, pool=a.pool, fast=a.fast, n_min_ref=a.n_min_ref,
                               backend=getattr(a, "backend", "klip"), log=log)
    _no_cube_name(a, "--instrument generic")
    _no_drop_frames(a, "--instrument generic")
    _subtract(red, a, log)
    _set_threads_per_job(red, a)
    _set_card(red, a)
    _set_klip_gram(red, a, log)
    pm = _line_mask(red, a, log)
    objective, sampler = generic.default_config(red, clean_subtract=not a.no_clean_subtract,
                                                known=[tuple(k) for k in (a.known or [])],
                                                metric=getattr(a, "metric", "mawet"), forbidden_pa=_forbidden(a),
                                                **({} if pm is None else {"pixel_mask": pm}))
    guard = generic.make_guard(red, n_min_ref=a.n_min_ref, ref_frac=a.ref_frac,
                               bin_min=getattr(a, "bin_min", 1) or 1) if not a.fast else None
    return red, objective, sampler, guard, None


def _groups_spec(v):
    """Parse ``--groups``, which is none, auto, a list of JD split times or ``@file`` with one label per frame.

    Parameters
    ----------
    v : str, sequence of str or None
        The option's value.

    Returns
    -------
    groups : str, list of float, ndarray of int or None
        The keyword, the split times, the labels read from the file, or None without the option.
    """
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
    """Build the reducer, objective, sampler and guard of pyNOMIC working directories.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.
    log : callable, default print
        Function that takes each line of the log.

    Returns
    -------
    reducer : Reducer
        The reducer.
    objective : Objective
        The objective.
    sampler : Sampler
        The source sampler.
    guard : callable or None
        The guard on the reference frames, None with ``--fast``.
    throughput : None
        No transmission, which this instrument does not give.
    """
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
    _no_cube_name(a, "--instrument nomic")
    _no_drop_frames(a, "--instrument nomic")
    _subtract(red, a, log)
    _set_threads_per_job(red, a)
    _set_card(red, a)
    _set_klip_gram(red, a, log)
    objective, sampler = nomic.default_config(red, clean_subtract=not a.no_clean_subtract,
                                              metric=getattr(a, "metric", "mawet"),
                                              known=[tuple(k) for k in (a.known or [])])
    guard = nomic.make_guard(red, n_min_ref=a.n_min_ref, ref_frac=a.ref_frac) if not a.fast else None
    return red, objective, sampler, guard, None


def _ales_objects(a, log=print):
    """Build the reducer, objective, sampler and guard of ALES integral-field cubes, wavelength groups as partitions.

    The cubes are a reduction's products under ``--root``, or a public data set in the release layout
    (``--ales-dataset``), one partition per night and group (:mod:`klip_tpe.instruments.ales`).

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.
    log : callable, default print
        Function that takes each line of the log.

    Returns
    -------
    reducer : Reducer
        The reducer.
    objective : Objective
        The objective.
    sampler : Sampler
        The source sampler.
    guard : callable or None
        The guard on the reference frames, None with ``--fast``.
    throughput : None
        No transmission, which this instrument does not give.

    Raises
    ------
    SystemExit
        If neither ``--root`` nor ``--ales-dataset`` is given, ``--ales-reflib`` lacks ``--ales-ref``,
        ``--ales-ref`` comes with a data set, or the data set is unknown or not in the release layout.
    """
    from .instruments import ales
    from .parallel import resolve_workers
    dataset = getattr(a, "ales_dataset", None)
    if not a.root and not dataset:
        raise SystemExit("--instrument ales needs --root, the reduction's products directory, or --ales-dataset NAME")
    a.workers = resolve_workers(a.workers)
    reflib = bool(getattr(a, "ales_reflib", False))
    if reflib and not a.ales_ref:
        raise SystemExit("--ales-reflib needs the reference star: --ales-ref NAME")
    if dataset:
        from . import datasets as _datasets
        if a.ales_ref:
            raise SystemExit(f"--ales-ref: the data set {dataset} carries no reference star")
        try:
            files = _datasets.fetch(dataset)
        except KeyError as exc:
            raise SystemExit(f"--ales-dataset: {exc.args[0]}")
        if not ales.release_nights(files):
            raise SystemExit(f"--ales-dataset {dataset} is not an ALES data set in the release layout")
        datasets, lam_m, fwhm_px, table = ales.load_release(
            files, nights=getattr(a, "ales_nights", None), only=a.ales_only, pxscale=a.pxscale, diam_m=a.diam,
            angle_offset=a.ales_angle_offset, angle_sign=a.angle_sign, tags=None if a.no_tags else "auto",
            tag_r_in=a.ales_tag_r_in, log=log)
        px = float(next(iter(datasets.values())).meta["provenance"]["pxscale"])
    else:
        root = a.root[0] if isinstance(a.root, (list, tuple)) else a.root
        c0, c1 = (int(a.ales_channels[0]), int(a.ales_channels[1])) if a.ales_channels else (ales.ALES.c0, ales.ALES.c1)
        px = a.pxscale if a.pxscale is not None else ales.ALES.pxscale
        datasets, lam_m, fwhm_px, table = ales.load_ales(
            root, a.ales_target, ref=a.ales_ref, c0=c0, c1=c1, ngrp=a.ales_groups, only=a.ales_only,
            pxscale=px, diam_m=a.diam, angle_offset=a.ales_angle_offset, angle_sign=a.angle_sign,
            tags=None if a.no_tags else "auto", tag_r_in=a.ales_tag_r_in, cache=a.ales_cache,
            ref_bin=int(getattr(a, "ales_ref_bin", 5) or 1), log=log)
    if a.fwhm_px:
        fwhm_px = {k: float(a.fwhm_px) for k in fwhm_px}
    red = ales.make_reducer(datasets, lam_m, fwhm_px, pxscale=px, diam_m=a.diam,
                            ee_radius_px=a.r_ee if a.r_ee else 3.0, representative=a.ales_representative,
                            weighting=a.weighting, max_workers=a.workers, pool=a.pool, fast=a.fast,
                            n_min_ref=a.n_min_ref, backend=getattr(a, "backend", "klip"), table=table,
                            reflib=reflib, reflib_science=getattr(a, "ales_reflib_science", "all") or "all",
                            sdi=bool(getattr(a, "ales_sdi", False)), sdi_cap=int(getattr(a, "ales_sdi_cap", 100) or 100),
                            sdi_lenders=getattr(a, "ales_sdi_lenders", "all") or "all", log=log)
    _no_cube_name(a, "--instrument ales")
    _no_drop_frames(a, "--instrument ales")
    _subtract(red, a, log)
    _set_threads_per_job(red, a)
    _set_card(red, a)
    _set_klip_gram(red, a, log)
    pm = _line_mask(red, a, log)
    objective, sampler = ales.default_config(red, clean_subtract=not a.no_clean_subtract,
                                             known=[tuple(k) for k in (a.known or [])],
                                             forbidden_pa=_forbidden(a),
                                             metric=getattr(a, "metric", "mawet"),
                                             **({} if pm is None else {"pixel_mask": pm}))
    guard = None
    if not a.fast:
        guard = ales.make_guard(red, max_excluded=None if a.ales_max_excluded < 0 else a.ales_max_excluded,
                                prefer_m=a.ales_prefer_um * 1e-6, n_min_ref=a.n_min_ref, ref_frac=a.ref_frac,
                                frame_selection=not getattr(a, "no_framesel", False),   # resume has no --no-framesel
                                bin_min=getattr(a, "bin_min", 1) or 1)
    return red, objective, sampler, guard, None


def _add_near_args(p):
    """Add the options that build a command's reducer, from its instrument and data, to its parser.

    Parameters
    ----------
    p : argparse.ArgumentParser
        The command's parser.
    """
    p.add_argument("--instrument", default="near", choices=["near", "nomic", "generic", "ales"],
                   help="near: NEAR campaign tree under --root; nomic: pyNOMIC working directory under --root; "
                        "generic: any registered cube (--cube/--angles/--psf ...); "
                        "ales: an LBTI/ALES reduction's products under --root, wavelength groups as partitions")
    p.add_argument("--root", nargs="+", default=None,
                   help="NEAR: data root. nomic: pyNOMIC working directory (one per --obj, or one for all). "
                        "generic: optional base directory for the run directory. "
                        "ales: the reduction's products directory, read only (the run goes to --run-dir), or --ales-dataset")
    gg = p.add_argument_group("generic cubes (--instrument generic / klip-tpe generic)")
    gg.add_argument("--cube", nargs="+", default=None, help="registered cube(s), (n, ny, nx) FITS; one per partition")
    gg.add_argument("--angles", nargs="+", default=None, help="derotation angles FITS (one per cube; CCW to North-up)")
    gg.add_argument("--psf", nargs="+", default=None,
                    help="generic: off-axis PSF template FITS (one, or one per cube); nomic: airy (default) | frame | gaussian")
    gg.add_argument("--ref-cube", nargs="+", default=None, help="PSF-reference cube(s) for RDI/ARDI (one per cube)")
    gg.add_argument("--names", nargs="+", default=None, help="partition names (default: cube file stems)")
    gg.add_argument("--star-flux", nargs="+", default=None,
                    help="star flux per cube, in the science frames' own units and in the "
                         "template's normalisation aperture; omit it and the contrast axis is "
                         "in template units")
    gg.add_argument("--pxscale", type=float, default=None,
                    help="arcsec / px. generic: the cubes' scale. near: the plate scale the pixel constants follow "
                         "(lambda/D, the FWHM). A new run takes 0.0445, the scale of the 2026 astrometry. A run keeps "
                         "the scale it was made with (run_setup.json), and one made before 2026-11-01, which did not "
                         "record it, 0.0456")
    gg.add_argument("--lam", type=float, default=None, help="wavelength (m)")
    gg.add_argument("--diam", type=float, default=8.4, help="aperture diameter (m) for lambda/D")
    gg.add_argument("--angle-sign", type=float, default=1.0, help="-1 flips the derotation-angle sign")
    gg.add_argument("--crop-half", type=int, default=None,
                    help="crop to 2h+1 px around --center (generic default: full frame; nomic default: 75)")
    gg.add_argument("--center", type=float, nargs=2, default=None, help="star pixel (x y); default: array centre")
    gg.add_argument("--wv-index", type=int, default=None, help="channel of a 4-d IFS cube")
    gg.add_argument("--no-tags", action="store_true", help="no data-derived frame-quality tags (no frame selection)")
    gg.add_argument("--forbidden-pa", type=float, nargs=2, action="append", metavar=("PA", "HALFWIDTH"),
                    help="near / generic / ales: no injection within HALFWIDTH deg of position angle PA "
                         "(a known companion's azimuth, a disk, a line through the star); may repeat")
    gg.add_argument("--mask-line", type=float, nargs=2, action="append", metavar=("PA", "HALFWIDTH"),
                    help="near / generic / ales: leave the pixels within HALFWIDTH px of the line through the "
                         "star at position angle PA (both sides) out of the noise apertures; may repeat. Pair "
                         "it with --forbidden-pa on the same PA and PA+180, so injections stay off it too")
    ga = p.add_argument_group("LBTI/ALES (--instrument ales)")
    ga.add_argument("--ales-dataset", default=None, metavar="NAME",
                    help="a public ALES data set in the release layout, downloaded once into $KLIP_TPE_DATA "
                         "(klip_tpe.datasets: lbti_ales_mwc758), one partition per night and wavelength group, "
                         "in place of --root")
    ga.add_argument("--ales-nights", nargs="+", default=None, metavar="NIGHT",
                    help="with --ales-dataset: the nights to load, as dates (20191114) or labels (n1), all by default")
    ga.add_argument("--ales-target", default="TARGET", metavar="NAME",
                    help="prefix of the product files under --root (NAME_cube_cen_clean.fits ...; default TARGET)")
    ga.add_argument("--ales-ref", default=None, metavar="NAME",
                    help="reference star whose products sit beside the target's (e.g. REF): its group means "
                         "become each group's RDI library (default: none, RDI off)")
    ga.add_argument("--ales-reflib", action="store_true",
                    help="search the reference star as a library: nkeep_psfref, the reference means most "
                         "correlated with each target frame, searched from 0 (plain ADI, to the bit) to the "
                         "whole pool, on top of the ADI references (needs --ales-ref)")
    ga.add_argument("--ales-ref-bin", type=int, default=5, metavar="N",
                    help="the reference star's frames in consecutive means of N before anything else "
                         "(default 5; 1 = every frame)")
    ga.add_argument("--ales-reflib-science", default="all", choices=["all", "searched"],
                    help="the science pool of the searched library: every frame the ADI rule admits "
                         "(default) or a searched count too, nkeep_altroll, whose 0 is pure RDI")
    ga.add_argument("--ales-sdi", action="store_true",
                    help="ADI, SDI and RDI in one KLIP step: the other groups' frames, rescaled to each group's "
                         "wavelength, join its library as a pool; nkeep_sdi (the rows most correlated with each target "
                         "frame among those the displacement rule admits) is searched from 0 (no SDI) to --ales-sdi-cap. "
                         "A reference star given (--ales-ref) is the library's third pool, searched. nkeep_sdi = "
                         "nkeep_psfref = 0 is plain ADI, to the bit")
    ga.add_argument("--ales-sdi-cap", type=int, default=100, metavar="N",
                    help="most-correlated SDI rows a target frame may take, the top of nkeep_sdi (default 100; the "
                         "per-target cost grows with the square of the basis)")
    ga.add_argument("--ales-sdi-lenders", default="all", choices=["all", "selected"],
                    help="which groups lend SDI rows: every group, excluded ones too (default), or only the groups "
                         "the configuration includes")
    ga.add_argument("--ales-groups", type=int, default=20, metavar="N",
                    help="wavelength groups over the usable channels, equally spaced in ln(lambda) (default 20; "
                         "15 = one R = 40 resolution element each)")
    ga.add_argument("--ales-channels", type=int, nargs=2, default=None, metavar=("C0", "C1"),
                    help="first and last usable channel, inclusive (default 6 88)")
    ga.add_argument("--ales-only", nargs="+", default=None, metavar="gNN",
                    help="load these groups only (checks and short runs)")
    ga.add_argument("--ales-max-excluded", type=int, default=10, metavar="N",
                    help="the group flags may switch off at most N groups; a proposal past that has groups "
                         "switched back on, nearest --ales-prefer-um first (default 10; -1 = no bound)")
    ga.add_argument("--ales-prefer-um", type=float, default=3.7, metavar="UM",
                    help="wavelength the group bound restores groups towards (default 3.7)")
    ga.add_argument("--ales-representative", default=None, metavar="gNN",
                    help="the group whose FWHM and matched filter stand for the combined image "
                         "(default: the group at the mean wavelength)")
    ga.add_argument("--ales-angle-offset", type=float, default=180.0, metavar="DEG",
                    help="added to LBT_PARA once: the derotation is CCW by LBT_PARA + this (default 180)")
    ga.add_argument("--ales-tag-r-in", type=float, default=2.0, metavar="PX",
                    help="inner radius of the frame-quality tags' annulus (default 2 spaxels: the star's "
                         "core and halo; 6, the generic default, is background-dominated here)")
    ga.add_argument("--ales-cache", default=None, metavar="DIR",
                    help="directory for the group cubes as .npy, written once and read back after "
                         "(a group mean reads the whole cube, which may sit on a slow volume)")
    ga.add_argument("--ales-selection", default="binary", choices=["binary", "two_slot"],
                    help="group flags: one 0/1 dimension per group (default) or --max-drop drop slots")
    p.add_argument("--nights", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--backend", default="klip", choices=["klip", "pyklip", "vip"],
                   help="PSF-subtraction engine: built-in annular KLIP (default), pyKLIP or VIP (must be installed)")
    p.add_argument("--rotate", default=None, choices=["auto", "scipy", "opencv"],
                   help="the library a new run rotates its frames with: auto (default, or KLIP_TPE_ROTATE): OpenCV 5 "
                        "when installed, about 5x faster, else scipy.  A resumed run keeps the one it recorded")
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
                   help="worker budget, the threads the jobs running together may use: 'auto' (default) = one per "
                        "physical core, 'all' = every logical CPU (hardware threads included; 'auto' before "
                        "2026-10-27), an integer pins it, -k = 'auto' less k")
    p.add_argument("--pool", default="auto", choices=["auto", "processes", "threads"],
                   help="how the budget is spent: forked worker processes (default; the IDL bridges) or threads")
    p.add_argument("--threads-per-job", type=int, default=2,
                   help="threads per partition job on average (default 2, 4 before 2026-10-27). The pool gets "
                        "workers / threads-per-job processes, never fewer than min(workers, 2 x partitions). A job "
                        "expected to take long gets up to 4x as many and a light one 1")
    p.add_argument("--gpu-share", default=None,
                   help="percentage of the search's injected draws reduced on the card (an NVIDIA GPU, through "
                        "klip_tpe.gpu_reducer), the rest on the CPU workers. Each trial sends the card the share "
                        "of its jobs with the lowest hash of the evaluation, the draw and the night, so a run and "
                        "its resumes reproduce. 'auto' (the default for a new run): 12 where Linux shows a card "
                        "and the module is there (30 before 2026-11-15), else 0. A run keeps its share across "
                        "resumes (card.json in the run directory)")
    p.add_argument("--gpu-workers", type=int, default=None,
                   help="card worker processes sharing the card (default 3). At large bins one leaves the card "
                        "idle while it prepares the next job")
    p.add_argument("--gpu-device", default=None,
                   help="the card the card workers use, as CUDA_VISIBLE_DEVICES takes it: an index in CUDA's "
                        "order or a UUID (nvidia-smi -L). Default: what the run has been using (card.json), "
                        "else CUDA_VISIBLE_DEVICES, which is then recorded, so a resume keeps the card")
    p.add_argument("--weighting", default="equal", choices=["equal", "sqrt_texp"])
    p.add_argument("--fast", action="store_true", help="single-basis KLIP (angsep/anglemax not searched)")
    p.add_argument("--klip-gram", default="target", choices=["target", "zone"],
                   help="how the per-target KLIP forms each target's reference covariance. 'target' "
                        "(default) works from its references' pixels and forms all modes. 'zone' reads "
                        "it off one Gram matrix per zone and projects the target without forming the "
                        "modes, the same KLIP to float64 rounding but not to the bit, and about twice "
                        "as fast per target at small bins. A run keeps the mode it was started or "
                        "last resumed with (its klip_gram.json) unless this is given")
    p.add_argument("--no-library", action="store_true")
    p.add_argument("--no-clean-subtract", action="store_true")
    p.add_argument("--ladder-pair", action="store_true",
                   help="two-source annulus: use the pre-2026-09-05 two-rung radial ladder instead of "
                        "both sources at the annulus' area-weighted mid radius (only to reproduce older IDL runs)")
    p.add_argument("--n-min-ref", type=int, default=10)
    p.add_argument("--ref-frac", type=float, default=0.95)
    p.add_argument("--bin-min", type=int, default=1, metavar="N",
                   help="generic / ales: floor of the temporal bin, as a projection (MinBinGuard): a draw below "
                        "it is evaluated at N frames per bin. Cost, not correctness: on a long sequence bin 1 "
                        "costs tens of times more than bin 5 (default 1 = no floor)")
    p.add_argument("--bin-max", type=int, default=None, metavar="N",
                   help="generic / ales: top of the searched temporal-bin range, frames per bin (default: ales 30, "
                        "generic nframes / 8). A winner sitting on it (distance_to_bounds bin = 0) wants it higher")
    p.add_argument("--known", type=float, nargs=2, action="append", metavar=("RHO", "PA"))
    p.add_argument("--psflib", default=None, metavar="DIR",
                   help="near: the PSF library's directory (default ROOT/psflib), e.g. psflib_v2. Recorded in "
                        "run_setup.json and kept on resume, extend and rereduce when not typed")
    p.add_argument("--north", default=None, choices=["files", "constant"],
                   help="near: each frame's true north at derotation and in the injection azimuth. files (a new "
                        "run's default): each night's AB_truenorth_clean.txt, one number per frame, where there "
                        "is one, else 36.5 deg. constant: 36.5 deg for every frame. Recorded in run_setup.json "
                        "with each file's SHA-1. resume, extend and rereduce keep the run's own and refuse a "
                        "recorded file that is gone or has changed. A run made before 2026-11-01 used 36.5")
    p.add_argument("--cube-name", default=None, metavar="NAME",
                   help="near: the cube read in each night's directory (default AB_cube_skysub_cen_clean.fits, "
                        "or AB_cube_crop150.fits where only that is there), e.g. a reduction's "
                        "AB_cube_skysub_cen_clean_abline.fits. Recorded in the run's checkpoint, and kept on "
                        "resume and extend when not typed")
    p.add_argument("--drop-frames", action="append", default=None, metavar="NIGHT:INDEX[,INDEX...]",
                   help="near: frames left out of a night at load, before the frame tags, the frame cut, binning "
                        "and the injections see them: the night (n4 or 4), a colon and the frames' indices in the "
                        "night's cube, counted from 0, e.g. n4:551. It may repeat, and 'none' drops nothing (on "
                        "rereduce, in place of the run's list). Recorded in the run's checkpoint, and kept on "
                        "resume and extend when not typed")
    p.add_argument("--subtract-sources", default=None, metavar="FILE",
                   help="point sources taken out of every frame once, at load: one per line, rho (arcsec), PA "
                        "(deg) and contrast, each injected with the negative of its contrast by the injection "
                        "code, so placement, true north, centre, flux unit and throughput are a draw's. Every "
                        "reduction sees the edited frames, the KLIP-FM model cube does not. Recorded in the "
                        "run's checkpoint, and kept on resume and extend when not typed")
    p.add_argument("--subtract-line", default=None, metavar="FILE",
                   help="a line through the star taken out of every frame once, at load, from a node file: 'pa', "
                        "'step' (px) and optional 'rho_max' lines, then |s| (arcsec) with the ridge's peak contrast "
                        "on each half-line. The chain is a source every step px from the first node to rho_max "
                        "(default the crop edge), each at a(s) step / L, with L the injection PSF's line integral; "
                        "as --subtract-sources otherwise (klip-tpe fit-chain --line-nodes writes such a file)")
    p.add_argument("--inject-sources", default=None, metavar="FILE",
                   help="point sources added to every frame once, at load, after any subtraction: one per line, "
                        "rho (arcsec), PA (deg) and contrast, by the code a draw's injection uses, e.g. planets "
                        "put on a line to see what its subtraction does to them. Every reduction sees them, the "
                        "KLIP-FM model cube does not. Recorded in the run's checkpoint, and kept on resume and "
                        "extend when not typed")
    p.add_argument("--metric", default="mawet", choices=["mawet", "fmmf"],
                   help="detection metric: 'mawet' (default) filters with the injection PSF; "
                        "'fmmf' forward-models that PSF through each configuration's own subtraction "
                        "and filters with the result (klip_tpe.fmmf)")


def _protocol_args(p):
    """Add the run-protocol options of a new run to a parser.

    They cover the adaptive annuli (A §3.1, ``--opt-width``), the scan modes (§5.9), the products (§9)
    and the verification hooks (C).

    Parameters
    ----------
    p : argparse.ArgumentParser
        The command's parser.
    """
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
    p.add_argument("--n-remeasure", type=int, default=1, metavar="N",
                   help="score each trial as the MEAN of N independent injection draws (default 1). "
                        "The objective's per-draw scatter is ~0.9 S/N on typical data, so N=3 divides "
                        "it by sqrt(3) and roughly halves the optimism of the reported best; see docs/BUDGET.md")
    p.add_argument("--score-floor", default=None, metavar="S",
                   help="the lowest score a trial takes (default 0). A trial's score is the mean of its draws, and a "
                        "mean below the floor is recorded as the floor, with the mean beside it. The draws keep "
                        "their own scores, and the calibration and the validation are not floored. TPE ranks the "
                        "trials, so a floor of 0 changes the search only where most trials score below 0. off sets "
                        "no floor, as before 2026-11-01 and with --from-idl-setup")
    p.add_argument("--verify-snr", default=None, choices=["search", "idl"],
                   help="how the verify, the candidate search and param_verify measure (default search: with the "
                        "search's own S/N under the run's rules, and every ring and map without the pixel mask). idl "
                        "measures as before 2026-11-01, with the IDL verify's S/N (a Gaussian kernel, no despike or "
                        "radial band) and the pixel mask in the search's rings alone, as --from-idl-setup does")
    p.add_argument("--fmmf-fm-apertures", action="store_true", default=False,
                   help="with --metric fmmf, each noise aperture takes a forward model of its own, from a "
                        "forward-model pass with a source alone there, and a sample between two apertures their two "
                        "models weighted by azimuth, the template a source there has where the PSF is not round "
                        "about the star. For the noise of the source's own filter, turning its template (the "
                        "default) is exact where the noise is round about the star, this where the noise changes "
                        "as the PSF does. It costs a pass per aperture, about 2 pi r / FWHM per source, on every "
                        "image the metric scores (default off)")
    p.add_argument("--n-sources", type=int, nargs="+", default=None, metavar="N",
                   help="injected sources per annulus, the last value repeating (default: klip-tpe's rule "
                        "by the annulus' outer edge, and 0 asks for the rule in that annulus). The cap that "
                        "keeps each source's noise ring clean still applies, and the log says when it binds")
    p.add_argument("--min-r-fwhm", type=float, default=1.5, metavar="X",
                   help="no injection inside X FWHM of the star, in every stage (default 1.5). On NEAR "
                        "1.0 FWHM is 1.0 lambda/D to half a percent")
    p.add_argument("--n-pv", type=int, default=20)
    p.add_argument("--pv-divmin", type=float, default=0.05)
    p.add_argument("--candidates", action="store_true", help="blind candidate search on the running/final stitch")
    p.add_argument("--write-setup-files", dest="write_setup_files", action="store_true", default=True)
    p.add_argument("--no-setup-files", dest="write_setup_files", action="store_false")
    p.add_argument("--legacy-stitch", action="store_true", help="equal-weight stitch, no contrast-curve seam trim")
    p.add_argument("--batch", type=int, default=1,
                   help="trials in flight at once (default 1). TPE then proposes from a posterior a few "
                        "trials old, which changes the search's trajectory")
    p.add_argument("--batch-mode", choices=("pipeline", "sync"), default=None,
                   help="with --batch: send the next proposal as soon as the oldest trial is in "
                        "(pipeline, the default), or propose whole batches from one posterior and wait "
                        "for all of a batch (sync)")
    p.add_argument("--edge-padding", choices=("between", "all"), default=None,
                   help="pad annulus edges only where two annuli meet (between, the default) or on every edge "
                        "including the innermost and outermost (all, as the IDL does)")
    _stitch_args(p)
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
                   help="live window size as a fraction of the 1850x990 panel (1.0 = the IDL window, 1:1 pixels), "
                        "shrunk to fit a smaller screen; the window keeps this size")
    p.add_argument("--no-fm-curve", dest="fm_curve", action="store_false", default=True,
                   help="skip the KLIP-FM cross-check curve after each annulus (A 9.1)")
    p.add_argument("--fm-preview", dest="fm_preview", action="store_true", default=True,
                   help="live KLIP-FM preview at each new best (A 9.3; one extra reduction per new best; default on)")
    p.add_argument("--no-fm-preview", dest="fm_preview", action="store_false")
    live = p.add_mutually_exclusive_group()
    live.add_argument("--no-liveness-check", dest="liveness_check", action="store_const", const="off",
                      default="warn",
                      help="skip the pre-flight that moves each searched reduction dimension alone and warns "
                           "when one changes nothing (klip_tpe.liveness; ~1 + ndim reductions, new runs only)")
    live.add_argument("--strict-liveness", dest="liveness_check", action="store_const", const="strict",
                      help="stop before the first evaluation when the pre-flight finds a dimension that "
                           "changes nothing, instead of warning")


def _stitch_args(p, rerun: bool = False):
    """Add ``--stitch-injections`` and ``--overlap-pads`` to a parser.

    Parameters
    ----------
    p : argparse.ArgumentParser
        The command's parser.
    rerun : bool, default False
        Whether the command is ``rereduce``, whose defaults are what the run did, rather than a new run
        (``near``, ``generic``).
    """
    was = ("the run's, all for a run made since 2026-11-03 and own before" if rerun else "all")
    p.add_argument("--stitch-injections", choices=("all", "own"), default=None,
                   help="what the final stitch's injected images hold. all: each annulus' winner reduced once "
                        "more with every annulus' validation sources, so the final verify measures each source on "
                        "the injected twin of the stitch, whichever tiles its footprint takes. own: each annulus' "
                        f"validated image, with its own sources only (default: {was})")
    pads = p.add_mutually_exclusive_group()
    pads.add_argument("--overlap-pads", dest="overlap_pads", action="store_const", const=True, default=None,
                      help="pad the stitch's tile edges between overlapping annuli (--ann-extra), as runs saved "
                           "before 2026-11-03 did" + (" (default: the run's pads)" if rerun else
                                                      ". Such an edge lies inside both zones, and by default the "
                                                      "tiles meet sharply there"))
    pads.add_argument("--no-overlap-pads", dest="overlap_pads", action="store_const", const=False,
                      help="no pads between overlapping annuli, the default for a new run"
                           + (". For a run that had them, the stitch is then not the run's" if rerun else ""))


def _n_sources(a):
    """Return ``--n-sources`` as ``RunConfig.n_sources``, one count or one per annulus.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Returns
    -------
    n_sources : int, list of int or None
        The count, the counts per annulus, or None without the option.

    Raises
    ------
    SystemExit
        If a count is negative.
    """
    ns = getattr(a, "n_sources", None)
    if ns is None:
        return None
    if any(int(v) < 0 for v in ns):
        raise SystemExit(f"--n-sources takes counts of 0 or more (0 = the rule), not {' '.join(map(str, ns))}")
    return int(ns[0]) if len(ns) == 1 else [int(v) for v in ns]


def _min_r_fwhm(a):
    """Return ``--min-r-fwhm`` as ``RunConfig.min_r_fwhm``.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Returns
    -------
    min_r_fwhm : float or None
        The radius, in FWHM, None without the option.

    Raises
    ------
    SystemExit
        If it is negative.
    """
    v = getattr(a, "min_r_fwhm", None)
    if v is None:
        return None
    if not v >= 0:
        raise SystemExit(f"--min-r-fwhm takes 0 or more FWHM, not {v}")
    return float(v)


def _contrast_grid_config(a):
    """Return ``RunConfig.contrast_grid`` from the ``--contrast-grid`` and ``--cg-*`` options.

    Without ``--contrast-grid`` the mode is :class:`~klip_tpe.contrast_grid.ContrastGridConfig`'s
    default, the grid.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Returns
    -------
    config : ContrastGridConfig
        The configuration.
    """
    from .contrast_grid import ContrastGridConfig
    mode = getattr(a, "contrast_grid", None)
    return ContrastGridConfig(**({"mode": mode} if mode else {}), n_pa=int(getattr(a, "cg_pa", 12)),
                              n_samples=int(getattr(a, "cg_samples", 96)),
                              sep_step_fwhm=float(getattr(a, "cg_step", 1.0)),
                              target_snr=float(getattr(a, "cg_target", 5.0)),
                              precision=float(getattr(a, "cg_precision", 0.1)),
                              snr=getattr(a, "cg_snr", "aperture") or "aperture",
                              sidelobe=getattr(a, "cg_sidelobe", "off") or "off",
                              sidelobe_fwhm=float(getattr(a, "cg_sidelobe_fwhm", 2.0)))


def _ann_extra(a) -> list:
    """Return ``--ann-extra IN:OUT ...`` as ``[[inner, outer], ...]``, in pixels.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Returns
    -------
    annuli : list of list of 2 floats
        The extra annuli, empty without the option.

    Raises
    ------
    SystemExit
        If an annulus is not ``IN:OUT``.
    """
    out = []
    for item in getattr(a, "ann_extra", None) or []:
        parts = str(item).replace(",", ":").split(":")
        try:
            lo, hi = (float(v) for v in parts)
        except ValueError:
            raise SystemExit(f"--ann-extra {item}: each annulus is IN:OUT in px, e.g. 6.85:16.16") from None
        out.append([lo, hi])
    return out


def _score_floor(text):
    """Parse ``--score-floor S`` as ``RunConfig.score_floor``.

    Parameters
    ----------
    text : str
        The option's value, a number or ``off``.

    Returns
    -------
    floor : float or None
        The floor, None for ``off``.

    Raises
    ------
    SystemExit
        If the value is not a finite number or ``off``.
    """
    t = str(text).strip().lower()
    if t in ("off", "none"):
        return None
    try:
        v = float(t)
    except ValueError:
        raise SystemExit(f"--score-floor takes a number or off, not {text}") from None
    if not math.isfinite(v):
        raise SystemExit(f"--score-floor takes a finite number or off, not {text}")
    return v


def _protocol_config(a) -> dict:
    """Return the ``RunConfig`` keywords the run-protocol options set.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Returns
    -------
    kwargs : dict
        The keywords, the optional ones only where their option is given.
    """
    ns = _n_sources(a)
    rmin = _min_r_fwhm(a)
    extra = _ann_extra(a)
    sf = getattr(a, "score_floor", None)
    return dict(contrast_grid=_contrast_grid_config(a), opt_width=a.opt_width, width_range=tuple(a.width_range), r_cap=a.r_cap, k_mode=a.k_mode,
                k_scan_max=a.k_scan_max, stitch_every=a.stitch_every, verify=a.verify, verify_n_boot=a.verify_n_boot,
                param_verify=a.param_verify, n_pv=a.n_pv, pv_divmin=a.pv_divmin, candidates=a.candidates,
                n_remeasure=max(int(getattr(a, "n_remeasure", 1) or 1), 1),
                write_setup_files=a.write_setup_files, legacy_stitch=a.legacy_stitch,
                partition_weighting=a.weighting, fm_curve=a.fm_curve, fm_preview=a.fm_preview,
                **({"sidelobes": a.sidelobes} if getattr(a, "sidelobes", None) else {}),
                pair_area_midpoint=not getattr(a, "ladder_pair", False),
                liveness_check=getattr(a, "liveness_check", "warn"),
                batch=max(int(getattr(a, "batch", 1) or 1), 1),
                **({"batch_mode": a.batch_mode} if getattr(a, "batch_mode", None) else {}),
                **({"edge_padding": a.edge_padding} if getattr(a, "edge_padding", None) else {}),
                **({"n_sources": ns} if ns is not None else {}),
                **({"min_r_fwhm": rmin} if rmin is not None else {}),
                **({"ann_extra": extra} if extra else {}),
                **({"score_floor": _score_floor(sf)} if sf is not None else {}),
                **({"verify_snr": a.verify_snr} if getattr(a, "verify_snr", None) else {}),
                **({"fmmf_fm_apertures": True} if getattr(a, "fmmf_fm_apertures", False) else {}),
                **({"stitch_injections": a.stitch_injections} if getattr(a, "stitch_injections", None) else {}),
                **({"stitch_overlap_pads": bool(a.overlap_pads)} if getattr(a, "overlap_pads", None) is not None
                   else {}))


def build_near_space(red, a):
    """Build the search space of the instrument selected, with the protocol's adjustments.

    ``--opt-width`` adds a searched ``width`` dimension, and the scan modes drop the ``k_klip``
    dimensions, k coming from the k-scan.  An ALES space has one block shared by every group, whatever
    ``--global-block`` says.

    Parameters
    ----------
    red : Reducer
        The reducer.
    a : argparse.Namespace
        The parsed command line.

    Returns
    -------
    space : SearchSpace
        The search space.
    """
    from .instruments import near
    if getattr(a, "instrument", "near") == "generic":
        from .instruments import generic
        has_tags = all(r.frame_tags() for r in red.reducers.values())
        bmax = getattr(a, "bin_max", None)
        return generic.make_space(red, per_night=not a.global_block, k_klip_max=a.k_max,
                                  opt_framesel=has_tags and not a.no_framesel, search_angles=not a.fast,
                                  selection=None if a.no_selection else "two_slot", max_drop=getattr(a, "max_drop", 2),
                                  width_range=tuple(a.width_range) if a.opt_width else None,
                                  search_k=a.k_mode not in ("scan", "scan_rescore"),
                                  **({"bin_range": (1, int(bmax))} if bmax else {}))
    if getattr(a, "instrument", "near") == "ales":
        from .instruments import ales
        has_tags = all(r.frame_tags() for r in red.reducers.values())
        # one block shared by every group (per_night=False), whatever --global-block says
        return ales.make_space(red, k_klip_max=a.k_max, opt_framesel=has_tags and not a.no_framesel,
                               search_angles=not a.fast, bin_range=(1, int(getattr(a, "bin_max", None) or 30)),
                               selection=None if a.no_selection else a.ales_selection,
                               max_drop=a.ales_max_excluded if a.ales_selection == "two_slot" and a.ales_max_excluded > 0
                               else getattr(a, "max_drop", 2),
                               width_range=tuple(a.width_range) if a.opt_width else None,
                               search_k=a.k_mode not in ("scan", "scan_rescore"))
    return near.make_space(red, per_night=not a.global_block, k_klip_max=a.k_max,
                           opt_framesel=not a.no_framesel, search_angles=not a.fast,
                           selection=None if a.no_selection else "two_slot", max_drop=getattr(a, "max_drop", 2),
                           width_range=tuple(a.width_range) if a.opt_width else None,
                           search_k=a.k_mode not in ("scan", "scan_rescore"))


def _default_run_dir(a) -> str:
    """Return a new run's directory, in the IDL predecessor's layout ``<root>/comb/opt/run_YYYYMMDD_HHMMSS``.

    Generic cubes take ``<root>/run_...``, or ``runs/run_...`` without ``--root``, and ALES takes
    ``runs/run_...``, since its ``--root`` stays as it is.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Returns
    -------
    run_dir : str
        The directory, named by the time.
    """
    root = a.root[0] if isinstance(a.root, (list, tuple)) else a.root
    if getattr(a, "instrument", "near") == "generic":
        return os.path.join(root or "runs", time.strftime("run_%Y%m%d_%H%M%S"))
    if getattr(a, "instrument", "near") == "ales":
        # --root is the reduction's products directory, which stays as it is
        return os.path.join("runs", time.strftime("run_%Y%m%d_%H%M%S"))
    return os.path.join(root, "comb", "opt", time.strftime("run_%Y%m%d_%H%M%S"))


def cmd_near(a):
    """Run ``klip-tpe near``, which starts a run, or goes on with one on a directory that holds a checkpoint.

    With ``--from-idl-setup`` it mirrors a run of the IDL predecessor: its nights, annulus, budget,
    contrast, validation and TPE settings.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Raises
    ------
    SystemExit
        If ``--ann-extra`` does not fit the annuli.
    """
    from .runner import CalibrationConfig, RunConfig, Runner, ValidationConfig
    if not a.run_dir:
        a.run_dir = _default_run_dir(a)
        print(f"run directory: {a.run_dir}")
    _register(a.run_dir)
    _klip_gram_for_run(a)
    _card_for_run(a)
    _data_edits_for_run(a)                 # a near command on a check-pointed directory resumes it
    _constants_for_run(a)
    if getattr(a, "from_idl_setup", None) and not os.path.exists(os.path.join(a.run_dir, "run_setup.json")):
        _idl_constants(a)
    if getattr(a, "from_idl_setup", None):
        # mirror a run of the IDL predecessor: nights, annulus, budget, contrast, validation and TPE settings
        from .idl_compare import matched_run_config
        cfg, info = matched_run_config(a.from_idl_setup, seed=a.seed, **_protocol_config(a))
        # the IDL predecessor's validation rules unless asked otherwise
        if getattr(a, "n_rescore", None) is not None:
            cfg.validation.n_rescore = int(a.n_rescore)
        if getattr(a, "validation_dedup", None):
            cfg.validation.dedup = a.validation_dedup
        if getattr(a, "n_final", 0):
            cfg.validation.n_final, cfg.validation.n_final_draws = int(a.n_final), int(a.n_final_draws)
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
    edges = list(a.ann_edges[:1]) if a.opt_width else list(a.ann_edges)
    if _ann_extra(a):
        # the annuli are checked before the data are loaded
        try:
            RunConfig(ann_edges=edges, ann_extra=_ann_extra(a), opt_width=a.opt_width).tiles
        except ValueError as exc:
            raise SystemExit(f"--ann-extra: {exc}") from None
    red, objective, sampler, guard, thr = _near_objects(a)
    space = build_near_space(red, a)
    space.project = guard
    cfg = RunConfig(ann_edges=edges, n_iter=a.n_iter, n_init=a.n_init, search_mode=a.mode,
                    seed=a.seed, contrast0=a.contrast0, blocks=a.blocks, bench_tag=a.tag,
                    calibration=CalibrationConfig(forced=a.use_contrast, ceiling=a.cmax_cal),
                    validation=ValidationConfig(n_top=a.n_top, n_valid=a.n_valid,
                                                n_rescore=getattr(a, "n_rescore", None),
                                                dedup=getattr(a, "validation_dedup", None) or "image",
                                                n_final=int(getattr(a, "n_final", 0) or 0),
                                                n_final_draws=int(getattr(a, "n_final_draws", 200) or 200)),
                    **_protocol_config(a))
    r = Runner(red, space, objective, sampler, cfg, a.run_dir, throughput_fn=thr, callbacks=_callbacks(a))
    r.run()


def _callbacks(a):
    """Return the run's callbacks, the live display unless it is turned off.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Returns
    -------
    callbacks : list
        The :class:`~klip_tpe.display.LiveDisplay`, or nothing with ``--no-display``.
    """
    if not getattr(a, "display", True):
        return []
    from .display import LiveDisplay
    return [LiveDisplay(a.run_dir, every=a.display_every, pdf_every=a.pdf_every, show=getattr(a, "show", False),
                        aliens=getattr(a, "aliens", False), window_scale=getattr(a, "window_scale", 1.0),
                        movie_every=getattr(a, "movie_every", None))]


def latest_run_dir(root: str) -> str:
    """Return the newest ``run_*`` directory under ``<root>/comb/opt``, by name and so by start time.

    Parameters
    ----------
    root : str
        The data root.

    Returns
    -------
    run_dir : str
        The directory.

    Raises
    ------
    SystemExit
        If there is none.
    """
    d = os.path.join(root, "comb", "opt")
    runs = sorted(x for x in (os.listdir(d) if os.path.isdir(d) else []) if x.startswith("run_")
                  and os.path.isdir(os.path.join(d, x)))
    if not runs:
        raise SystemExit(f"no run_* directory under {d}")
    return os.path.join(d, runs[-1])


def _resolve_run_dir(a) -> str:
    """Resolve ``--run-dir``, ``last``, ``latest`` or ``newest`` naming the newest run under ``<root>/comb/opt``.

    Anything else is taken as it is.  The run record has had its say by then (:func:`_fill_from_registry`),
    so reaching here with nothing to go on means there is no run to resume, which it says rather than
    failing on a None path.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line, changed in place.

    Returns
    -------
    run_dir : str
        The run directory.

    Raises
    ------
    SystemExit
        If ``last`` comes without ``--root``, or no run lies under it.
    """
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
    """Put a run on the record, which ``klip-tpe runs`` lists and a bare ``klip-tpe resume`` restarts.

    A failure is ignored.

    Parameters
    ----------
    run_dir : str
        The run directory.
    argv : sequence of str or None, default None
        The command line recorded in place of this one.  A resume passes the complete command line it
        resolved to, since recording ``klip-tpe resume --show`` would leave the next resume an entry
        with no data root.
    """
    try:
        from .registry import register
        register(run_dir, argv=argv)
    except Exception:
        pass


def _explicit_dests(argv=None) -> set:
    """Return the options typed on a command line, by their ``dest`` names.

    argparse fills in defaults, so ``a.nights == [1..6]`` cannot tell the default from a user who typed
    it, and preferring the default over the value a run was started with would resume a three-night run
    as a six-night one.  Parsing again with every default suppressed leaves only what was given.

    Parameters
    ----------
    argv : sequence of str or None, default None
        The command line, without the program's name.  None takes ``sys.argv``.

    Returns
    -------
    dests : set of str
        The options typed, empty when the line does not parse.
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
    """Make ``klip-tpe resume`` work with no arguments, from the command line the run recorded when it started.

    The data root and the rest of what the reducer needs come from the newest record of the run that
    carries a data root, each option only where this command line does not type it.
    ``a._record_argv`` gets the complete command line this resume runs, for the record.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line, changed in place.
    argv : sequence of str or None, default None
        This command line, for :func:`_explicit_dests`.

    Raises
    ------
    SystemExit
        If the run is still running.
    """
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
              "truenorth", "fwhm_px", "metric", "backend", "forbidden_pa", "r_ee", "angle_sign",
              "no_tags", "ales_target", "ales_ref", "ales_groups", "ales_channels", "ales_only",
              "ales_max_excluded", "ales_prefer_um", "ales_representative", "ales_angle_offset",
              "ales_tag_r_in", "ales_cache", "ales_selection", "ales_reflib", "ales_ref_bin",
              "ales_reflib_science", "ales_sdi", "ales_sdi_cap", "ales_sdi_lenders", "bin_min", "bin_max",
              "ales_dataset", "ales_nights", "cube_name", "subtract_sources", "subtract_line", "inject_sources",
              "mask_line", "drop_frames"):
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
    """Run ``klip-tpe runs``, which lists the runs on record and stops those asked.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.
    """
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
    """Run ``klip-tpe resume``, which restarts a run from its checkpoint.

    A failure to read the run record does not stop the resume, but a run still running does.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.
    """
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
    _klip_gram_for_run(a)
    _card_for_run(a)
    _data_edits_for_run(a)
    _constants_for_run(a)
    red, objective, sampler, guard, thr = _near_objects(a)
    r = Runner.resume(a.run_dir, red, objective, sampler, project=guard, throughput_fn=thr)
    r.callbacks = _callbacks(a)
    r.run()


def cmd_extend(a):
    """Run ``klip-tpe extend``, which reopens a run with new totals of evaluations per annulus.

    A total of 0 or less leaves its annulus untouched, one at most the annulus' count re-validates it
    only, and a larger one goes on with its search.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.
    """
    from .runner import Runner
    _resolve_run_dir(a)
    _klip_gram_for_run(a)
    _card_for_run(a)
    _data_edits_for_run(a)
    _constants_for_run(a)
    red, objective, sampler, guard, thr = _near_objects(a)
    Runner.extend(a.run_dir, red, objective, sampler, a.n_iter, project=guard, throughput_fn=thr)


def cmd_fit_chain(a):
    """Run ``klip-tpe fit-chain``, which fits a chain of point sources to a finished run's final image.

    The chain's node values, a line through the star from ``--chain`` or ``--line-nodes``, are fitted
    through the run's winners (:mod:`klip_tpe.chainfit`) and written to ``--out``, by default beside
    the run.  With
    ``--per-night`` each night gets its own amplitude of the fitted chain.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Raises
    ------
    SystemExit
        If both or neither of ``--chain`` and ``--line-nodes`` are given, a file cannot be read, or the
        fit fails.
    """
    from .chainfit import fit_chain, fit_per_night, line_basis, read_chain, read_line_nodes
    from .runner import Runner
    try:
        _fill_from_registry(a, getattr(a, "_argv", None))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"  (could not read the run record: {exc!r}; carrying on with what was given)")
    _resolve_run_dir(a)
    _klip_gram_for_run(a, record=False)    # the run directory is only read
    _data_edits_for_run(a)
    _constants_for_run(a)
    if bool(a.chain) == bool(a.line_nodes):
        raise SystemExit("fit-chain needs one of --chain FILE and --line-nodes FILE")
    spec = None
    if a.chain:
        try:
            labels, nodes = read_chain(a.chain)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"--chain: {exc}")
    else:
        try:
            spec = read_line_nodes(a.line_nodes)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"--line-nodes: {exc}")
    red, objective, sampler, guard, thr = _near_objects(a)
    if spec is not None:
        first = _first_partition(red)
        labels, nodes = line_basis(spec, first.model, first.pxscale, tuple(first.data.cube.shape[-2:]),
                                   fallback=first.fallback)
    line_pa = a.line_pa if a.line_pa is not None else (spec["pa"] if spec is not None else None)
    r = Runner.resume(a.run_dir, red, objective, sampler, project=guard, throughput_fn=thr)
    masks = [(m[0], m[1], m[2] if m[2] > 0 else None) for m in (a.mask or [])]
    rho_range = tuple(a.rho_range) if getattr(a, "rho_range", None) else None
    out = a.out or os.path.abspath(a.run_dir).rstrip(os.sep) + "_chainfit"
    try:
        fit_chain(r, nodes, labels, line_pa=line_pa, halfwidth_px=a.halfwidth, masks=masks,
                  mask_known=not a.no_mask_known, candidate_tiers=a.mask_candidates or "", out_dir=out,
                  line_spec=spec, rho_range=rho_range)
        if a.per_night:
            from .injection import read_sources
            print("  one amplitude per night for the fitted chain:")
            fit_per_night(r, read_sources(os.path.join(out, "chain_fitted.txt")), line_pa=line_pa,
                          halfwidth_px=a.halfwidth, masks=masks, mask_known=not a.no_mask_known, out_dir=out,
                          rho_range=rho_range)
    except ValueError as exc:
        raise SystemExit(f"fit-chain: {exc}")


def cmd_rereduce(a):
    """Run ``klip-tpe rereduce``, which reduces a finished run's winners again on other frames, beside the run.

    The frames may have a line subtracted or planets added (:func:`_data_edits_for_rereduce`).  The
    final stitch at the run's weights, its S/N map and the final verify follow
    (:meth:`klip_tpe.runner.Runner.rereduce`), and ``--base-winners`` takes other configurations as the
    winners of some annuli.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Raises
    ------
    SystemExit
        If ``--base-winners`` cannot be read, ``--out`` already holds a re-reduction without
        ``--overwrite``, or the re-reduction fails.
    """
    from .runner import Runner
    try:
        _fill_from_registry(a, getattr(a, "_argv", None))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"  (could not read the run record: {exc!r}; carrying on with what was given)")
    _resolve_run_dir(a)
    _klip_gram_for_run(a, record=False)    # the run directory is only read
    _data_edits_for_rereduce(a)
    _constants_for_run(a)
    vs = _sources_file(a.verify_sources, "--verify-sources") if a.verify_sources else None
    winners = None
    if getattr(a, "base_winners", None):
        from .refine import read_picks, resolve_picks
        try:
            picks = read_picks(a.base_winners, a.run_dir, one_per_annulus=True)
            winners = {p.annulus: e for p, e in resolve_picks(a.run_dir, picks)}
        except (OSError, ValueError) as exc:
            raise SystemExit(f"rereduce: --base-winners: {exc}")
    tail = "_composite" if winners else "_rereduce"
    out = os.path.abspath(a.out or (os.path.abspath(a.run_dir).rstrip(os.sep) + tail))
    if os.path.exists(os.path.join(out, "rereduce.json")) and not a.overwrite:
        raise SystemExit(f"{out} already holds a re-reduction (rereduce.json): give --out another directory, or "
                         f"--overwrite")
    red, objective, sampler, guard, thr = _near_objects(a)
    r = Runner.resume(a.run_dir, red, objective, sampler, project=guard, throughput_fn=thr, same_frames=False)
    if getattr(a, "verify_snr", None):
        # the re-reduction's verify, in place of the run's own (the run directory is only read)
        print(f"  verify: --verify-snr {a.verify_snr}, the run's own is {r.cfg.verify_snr}")
        r.cfg.verify_snr = a.verify_snr
    try:
        r.rereduce(out, verify_sources=vs, verify=not a.no_verify,
                   stitch_injections=getattr(a, "stitch_injections", None),
                   overlap_pads=getattr(a, "overlap_pads", None), winners=winners)
    except ValueError as exc:
        raise SystemExit(f"rereduce: {exc}")


def cmd_rescore(a):
    """Run ``klip-tpe rescore``, which scores candidates for annuli of a finished run on its stitch.

    The other annuli stay at the run's winners, and the candidates of other runs of the same setup
    enter too (:func:`klip_tpe.refine.rescore`).  The runs are only read, and the log also goes to
    ``rescore.log`` in ``--out``.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Raises
    ------
    SystemExit
        If a picks file cannot be read or the re-score fails.
    """
    from .refine import parse_annuli, rescore
    from .runner import Runner
    try:
        _fill_from_registry(a, getattr(a, "_argv", None))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"  (could not read the run record: {exc!r}; carrying on with what was given)")
    _resolve_run_dir(a)
    _klip_gram_for_run(a, record=False)    # the run directory is only read
    _data_edits_for_rereduce(a)
    _constants_for_run(a)
    out = os.path.abspath(a.out or (os.path.abspath(a.run_dir).rstrip(os.sep) + "_rescore"))
    red, objective, sampler, guard, thr = _near_objects(a)
    r = Runner.resume(a.run_dir, red, objective, sampler, project=guard, throughput_fn=thr, same_frames=False)
    if getattr(a, "verify_snr", None):
        print(f"  verify: --verify-snr {a.verify_snr}, the run's own is {r.cfg.verify_snr}")
        r.cfg.verify_snr = a.verify_snr
    logf = None
    if not a.plan:
        os.makedirs(out, exist_ok=True)
        logf = open(os.path.join(out, "rescore.log"), "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        if logf is not None:
            logf.write(msg + "\n")
            logf.flush()

    try:
        from .refine import read_picks
        known = list(a.entrants_from or [])
        bw = cand = None
        if getattr(a, "base_winners", None):
            bw = read_picks(a.base_winners, a.run_dir, known, r.cfg.nann, one_per_annulus=True)
        if getattr(a, "candidates", None):
            cand = read_picks(a.candidates, a.run_dir, known, r.cfg.nann)
        rescore(r, a.entrants_from or [], out, annuli=parse_annuli(a.annuli, r.cfg.nann), n_draws=a.draws,
                top=a.top, seed=a.seed, target=(float(a.target[0]), float(a.target[1])), n_cal=a.n_cal,
                window=a.window, overlap_pads=a.overlap_pads, plan=a.plan, card=a.card,
                sources_per_tile=a.sources_per_tile, clear_fwhm=a.clear_fwhm, base_winners=bw, candidates=cand,
                draws_from=getattr(a, "draws_from", None), tiles=getattr(a, "tiles", None), log=log)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"rescore: {exc}")
    finally:
        if logf is not None:
            logf.close()


def cmd_replay(a):
    """Run ``klip-tpe replay``, which reduces again a sample of the evaluations in the IDL predecessor's log.

    The ``--n`` evaluations are spread over the log by score, and each is scored through the Python
    objective.  The results go to ``replay.json`` in the run directory, with a summary printed.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.
    """
    import numpy as np
    from .idl_replay import read_idl_log, replay, summarize
    from .instruments import near
    from .runner import CalibrationConfig, RunConfig, Runner
    _idl_constants(a)
    red, objective, sampler, guard, thr = _near_objects(a)
    log = read_idl_log(a.idl_log)
    space = near.make_space(red, k_klip_max=a.k_max, anglemax_hi=a.anglemax_hi)
    if space.names != log.names:
        print("WARNING: space names differ from the IDL log header; mapping by name where possible")
    cfg = RunConfig(ann_edges=a.ann_edges, n_iter=1, n_init=1, seed=a.seed, contrast0=a.contrast,
                    calibration=CalibrationConfig(forced=[a.contrast]), edge_padding="all")   # the IDL predecessor's zones
    r = Runner(red, space, objective, sampler, cfg, a.run_dir, throughput_fn=thr, log=lambda s: None)
    r.contrast = a.contrast
    fin = np.flatnonzero(np.isfinite(log.score))
    o = fin[np.argsort(log.score[fin])]
    pick = sorted(set(int(i) for i in list(o[np.linspace(0, len(o) - 1, a.n).astype(int)])))
    res = replay(r, log, pick, out_path=os.path.join(a.run_dir, "replay.json"), n_repeat=a.repeat)
    print(json.dumps(summarize(res), indent=1))


def cmd_testbed(a):
    """Run ``klip-tpe testbed``, which compares the TPE density models on a synthetic problem.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.  The results go to ``--out`` as JSON when it is given.
    """
    from .testbed import compare_density_models
    out = compare_density_models(nblock=a.nblock, bdim=a.bdim, n_iter=a.n_iter, n_init=a.n_init, nseed=a.nseed,
                                 noise=a.noise, ridge=a.ridge, transposed=a.transposed, pbest=a.pbest)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


def cmd_plots(a):
    """Run ``klip-tpe plots``, which draws and saves a run's plots (:func:`klip_tpe.plots.plot_all`).

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.
    """
    from .plots import plot_all
    figs = plot_all(a.run_dir, save=True)
    print("wrote", ", ".join(sorted(figs)))


def cmd_render(a):
    """Run ``klip-tpe render``, which draws every evaluation's panel after the fact.

    The live display skips a panel whenever the render thread is still drawing the last one, so the
    window shows the present instead of falling behind, and says so with a pointer here.  The run's log
    and its per-evaluation crops hold everything a panel needs, so the complete set can be drawn
    afterwards.  It goes into its own directory, ``steps_rebuilt`` by default, because the live
    ``steps/`` also holds the calibration and validation frames this cannot redraw, and renumbering
    over them would leave the two sets interleaved.  With ``--with-data`` each frame's injected-PSF
    model is drawn from the run's frames (:func:`_injected_model_fn`), and a movie follows unless
    ``--no-movie``.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Raises
    ------
    SystemExit
        If the run directory does not exist.
    """
    from .display import render_steps
    if not os.path.isdir(a.run_dir):
        raise SystemExit(f"no run directory {a.run_dir}")
    out = a.out or os.path.join(a.run_dir, "steps_rebuilt")
    annuli = None if not a.annulus else [int(k) - 1 for k in a.annulus]
    fn = _injected_model_fn(a) if getattr(a, "with_data", False) else None
    paths = render_steps(a.run_dir, annuli=annuli, every=a.every, dpi=a.dpi, out_dir=out,
                         movie=not a.no_movie, movie_dir=out, inj_model_fn=fn)
    print(f"wrote {len(paths)} panels to {out}")
    gif = os.path.join(out, "opt_steps.gif")
    if not a.no_movie and paths and os.path.exists(gif):
        print(f"movie: {gif}")


#: run_setup.json's reducer name -> the CLI's --instrument
_RUN_INSTRUMENTS = {"near": "near", "ales": "ales", "nomic": "nomic", "generic": "generic"}


def _run_setup_reducer(run_dir) -> Dict[str, Any]:
    """Return ``run_setup.json``'s reducer record of a run directory.

    Parameters
    ----------
    run_dir : str
        The run directory.

    Returns
    -------
    record : dict
        The record, empty when the file is missing or unreadable.
    """
    import json
    try:
        with open(os.path.join(str(run_dir), "run_setup.json")) as f:
            return dict(json.load(f).get("reducer") or {})
    except (OSError, ValueError, TypeError):
        return {}


def _run_instrument(run_dir) -> Optional[str]:
    """Return the ``--instrument`` a run was made with, from its ``run_setup.json``.

    Parameters
    ----------
    run_dir : str
        The run directory.

    Returns
    -------
    instrument : str or None
        The instrument, from the reducer's name, else its provenance, None when the record does not say.
    """
    rec = _run_setup_reducer(run_dir)
    name = str(rec.get("name") or "").lower()
    if name in _RUN_INSTRUMENTS:
        return _RUN_INSTRUMENTS[name]
    inst = str(((rec.get("provenance") or {}).get("instrument")) or "").lower()
    return inst if inst in _RUN_INSTRUMENTS else None


def _injected_model_fn(a):
    """Return the injected-PSF model of ``render --with-data``, built on the run's reducer.

    The reducer is built from the data arguments as ``rereduce`` builds it, the ones left out filled
    from the run's record.  The instrument, the plate scale and the rotation library are the run's, from
    ``run_setup.json``.  The command refuses rather than take another instrument's defaults, and so does
    a run whose data arguments were never recorded (one started from a script), which the Python entry
    draws instead (:func:`klip_tpe.display.render_steps` with ``reducer=``).

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line, changed in place.

    Returns
    -------
    model_fn : callable
        Function ``fn(sources, shape)``, the injected-PSF model of ``sources``
        (:func:`klip_tpe.injection.injected_model`).

    Raises
    ------
    SystemExit
        If the instrument is unknown or not the run's, the reducer cannot be built, or its plate scale
        is not the run's.

    Notes
    -----
    The refusal dates from a user's note of 2026-11-10.
    """
    from .injection import injected_model
    try:
        _fill_from_registry(a, getattr(a, "_argv", None))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"  (could not read the run record: {exc!r}; carrying on with what was given)")
    given = _explicit_dests(getattr(a, "_argv", None))
    rec = _run_setup_reducer(a.run_dir)
    inst = _run_instrument(a.run_dir)
    script = ("this run did not record the data arguments it was made with (a run started from a script). "
              "Give them here, or draw from Python with the reducer the script built: "
              "klip_tpe.display.render_steps(run_dir, reducer=red, out_dir=...)")
    if "instrument" in given:
        if inst is not None and inst != a.instrument:
            raise SystemExit(f"render --with-data: the run was made with --instrument {inst} (run_setup.json), "
                             f"not {a.instrument}")
    elif inst is None:
        raise SystemExit("render --with-data: run_setup.json does not say which instrument made this run: give "
                         "--instrument and the data arguments, or " + script.split(". ", 1)[1])
    elif inst != getattr(a, "instrument", None):
        print(f"  --instrument {inst}, as run_setup.json records")
        a.instrument = inst
    px = rec.get("pxscale")
    if "pxscale" not in given and px is not None and a.instrument != "near":
        a.pxscale = float(px)                # NEAR's comes from its provenance (_constants_for_run)
    # the run's rotation library (RunConfig.rotation; scipy for a run recorded before it)
    from .klip import set_rotation_backend
    try:
        with open(os.path.join(str(a.run_dir), "run_setup.json")) as f:
            rot = (json.load(f).get("config") or {}).get("rotation")
    except (OSError, ValueError, TypeError):
        rot = None
    a.rotate = None                          # _near_objects is not to choose another
    set_rotation_backend(rot or "scipy")
    _klip_gram_for_run(a, record=False)    # the run directory is only read
    _data_edits_for_rereduce(a)
    _constants_for_run(a)
    try:
        red = _near_objects(a)[0]
    except SystemExit as exc:
        raise SystemExit(f"render --with-data: {exc}\n  {script}")
    if px is not None and abs(float(red.pxscale) - float(px)) > 1e-9 * max(abs(float(px)), 1.0):
        raise SystemExit(f"render --with-data: the reducer built here has {red.pxscale:g}\"/px, the run "
                         f"{float(px):g}\"/px (run_setup.json): its injected PSF would be drawn at another scale")
    print("  the run's frames loaded: each frame's injected-PSF model drawn from them")
    return lambda sources, shape: injected_model(red, sources, shape)


def cmd_compare(a):
    """Run ``klip-tpe compare``, which replays an IDL predecessor run's evaluations through the Python objective.

    The run may still be running.  Its new evaluations are synced
    (:class:`klip_tpe.idl_compare.LiveComparison`) into ``compare.txt`` in ``--out``, and the statistics
    of the agreement are printed.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.
    """
    from .idl_compare import LiveComparison
    from .runner import CalibrationConfig, RunConfig, Runner, ValidationConfig
    if getattr(a, "instrument", "near") in (None, "near"):
        _idl_constants(a)
    red, objective, sampler, guard, thr = _near_objects(a)
    space = build_near_space(red, a)
    space.project = guard
    cfg = RunConfig(ann_edges=list(a.ann_edges), n_iter=1, n_init=1, seed=a.seed, contrast0=a.contrast,
                    calibration=CalibrationConfig(forced=[a.contrast]), validation=ValidationConfig(n_top=1, n_valid=1),
                    fm_curve=False, save_fits=False, write_setup_files=False, param_verify=False,
                    edge_padding="all")                  # the IDL predecessor pads every zone edge
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
    """Add the search and protocol options shared by ``near`` and ``generic`` to a parser.

    Parameters
    ----------
    p : argparse.ArgumentParser
        The command's parser.
    """
    p.add_argument("--run-dir", default=None,
                   help="output directory (default: <root>/comb/opt/run_YYYYMMDD_HHMMSS, the IDL layout; "
                        "generic and ales: ./runs/run_...)")
    p.add_argument("--ann-edges", type=float, nargs="+", default=[0.0, 20.0])
    p.add_argument("--ann-extra", nargs="+", default=None, metavar="IN:OUT",
                   help="annuli beside those of --ann-edges, each IN:OUT px, which may overlap them, e.g. 6.85:16.16 "
                        "16.16:25.47. All are numbered by centre and searched, validated and verified over their own "
                        "zones. Where annuli overlap, the stitch takes each radius from the annulus with the nearest "
                        "centre (their tiles, in final_results.json and klip_stitched_params.txt). The per-annulus "
                        "lists of --n-iter, --n-init, --n-sources and --use-contrast follow the numbering by centre. "
                        "The first three repeat their last entry past their end, and an annulus past the end of "
                        "--use-contrast is calibrated")
    p.add_argument("--n-iter", type=int, nargs="+", default=[200])
    p.add_argument("--n-init", type=int, nargs="+", default=[50])
    p.add_argument("--mode", default="tpe", choices=["tpe", "random", "grid"])
    # The CLI default follows RunConfig.blocks and the TPE class.  It said "partitions" until
    # 2026-10-08, a month after the package default moved to univariate (IDL addendum 2), so
    # any `klip-tpe near` / `generic` call without --blocks ran block-multivariate densities.
    p.add_argument("--blocks", default="univariate",
                   help="TPE density model: univariate (default; the IDL reference) | partitions | full")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--contrast0", type=float, default=3e-5)
    p.add_argument("--use-contrast", type=float, nargs="+", default=None,
                   help="forced contrast per annulus, in the annuli's order (by centre with --ann-extra). 0 calibrates "
                        "that annulus, and so does every annulus past the end of the list. Unlike --n-iter, the list "
                        "does not repeat its last entry"),
    p.add_argument("--cmax-cal", type=float, default=None)
    p.add_argument("--k-max", type=int, default=None)
    p.add_argument("--n-top", type=int, default=3)
    p.add_argument("--n-valid", type=int, default=8)
    p.add_argument("--n-rescore", type=int, default=None, metavar="N",
                   help="fresh trials for the elected winner, whose median is then its validated score "
                        "(default --n-valid, 0 with --from-idl-setup). The election picks the largest of noisy "
                        "medians, so the winner's median on its own draws is high. A lone candidate is not "
                        "re-scored. 0 reports the election's median, as the IDL does")
    p.add_argument("--validation-dedup", default=None, choices=["image", "vector"],
                   help="how validation candidates must differ. With image (the default) a configuration whose "
                        "clean image equals an earlier candidate's is skipped for the next. With vector (the "
                        "default with --from-idl-setup) only the parameter vectors must differ, as in the IDL")
    p.add_argument("--n-final", type=int, default=0, metavar="K",
                   help="a final round for a sharper election (default 0, off). The K candidates with the largest "
                        "validated medians are scored on --n-final-draws common draws, the same injections for "
                        "each, and the largest median there is elected. On common draws the candidates' scores "
                        "move together, so their medians' differences scatter far less than on draws of their "
                        "own. The run's own random draws are untouched")
    p.add_argument("--n-final-draws", type=int, default=200, metavar="M",
                   help="common draws of the final round (default 200)")
    p.add_argument("--contrast-grid", default=None, choices=["grid", "random", "off"],
                   help="the contrast curve after validation (default grid). One injected source per reduction at "
                        "the winner config, on a grid of separations and position angles (grid) or at random "
                        "positions (random), each iterated on its contrast to the target S/N. The noise rings leave "
                        "out the source, the known companions and their ADI sidelobes. off skips it")
    p.add_argument("--cg-pa", type=int, default=12, metavar="N", help="position angles per separation (default 12)")
    p.add_argument("--cg-samples", type=int, default=96, metavar="N",
                   help="positions with --contrast-grid random (default 96)")
    p.add_argument("--cg-step", type=float, default=1.0, metavar="FWHM",
                   help="separation step of the grid in FWHM (default 1)")
    p.add_argument("--cg-target", type=float, default=5.0, metavar="SNR", help="target S/N of the grid (default 5)")
    p.add_argument("--cg-precision", type=float, default=0.1, metavar="F",
                   help="how close to the target a position must come, as a fraction of it (default 0.1)")
    p.add_argument("--cg-snr", default="aperture", choices=["aperture", "matched"],
                   help="the grid's S/N: the IDL contrast generators' aperture photometry at the injected position "
                        "(aperture, the default) or the run's matched-filter Mawet S/N (matched)")
    p.add_argument("--cg-sidelobe", default="off", choices=["off", "rotation", "fixed"],
                   help="the grid's noise ring. off (the default) skips the apertures within 1.5 FWHM of the source "
                        "and the known companions, as the IDL contrast generators skip the source and the aperture "
                        "each side. rotation also leaves out the arc the field rotation sweeps, capped by anglemax and "
                        "never under --cg-sidelobe-fwhm of arc, and fixed that width alone")
    p.add_argument("--cg-sidelobe-fwhm", type=float, default=2.0, metavar="FWHM",
                   help="the sidelobes' least half-width, in FWHM of arc (default 2)")
    p.add_argument("--sidelobes", default=None, choices=["farthest", "clean", "off"],
                   help="the noise rings of the search, calibration, validation, verify and candidates. off (the "
                        "default) skips the apertures within 1.5 FWHM of each injected source and known companion, "
                        "as every run before 2026-10-31 and the IDL did. farthest and clean also leave out the ADI "
                        "sidelobes of the sources and companions, +/- the data's PA coverage, and the source count "
                        "keeps 4 apertures outside them. Where fewer than 4 are left, farthest takes the 4 farthest "
                        "from the source and clean the clean reduction's ring. On NEAR this left 1 to 3 apertures "
                        "inside 0.93\" and one source a draw. The contrast grid has its own ring (--cg-sidelobe)")
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
    """Run ``klip-tpe view``, a live window on a run, running or finished, in a process of its own.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line.

    Raises
    ------
    SystemExit
        Always, with the viewer's exit status.
    """
    from .viewer import view
    raise SystemExit(view(run_dir=a.run_dir, root=a.root, interval=a.interval,
                          scale=a.window_scale, once=a.once))


class _Version(argparse.Action):
    """Action of ``--version``, which prints one unwrapped line, where argparse's own wraps a long path.

    Parameters
    ----------
    option_strings : sequence of str
        The option's strings.
    dest : str, default argparse.SUPPRESS
        The option's destination, none.
    default : object, default argparse.SUPPRESS
        The option's default, none.
    help : str or None, default None
        The option's help.
    """

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings=option_strings, dest=dest, default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        """Print the version (:func:`klip_tpe.version.describe`) and exit.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The parser.
        namespace : argparse.Namespace
            The namespace, unused.
        values : object
            The values, none.
        option_string : str or None, default None
            The option string, unused.
        """
        from .version import describe
        print(describe())
        parser.exit()


def _build_parser():
    """Build the parser of ``klip-tpe`` and its commands.

    Returns
    -------
    parser : argparse.ArgumentParser
        The parser, each command's ``func`` its ``cmd_*`` function.
    """
    ap = argparse.ArgumentParser(prog="klip-tpe", description="KLIP-TPE reduction-parameter optimizer")
    ap.add_argument("--version", action=_Version,
                    help="print the version, the git commit and where klip_tpe is imported from")
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

    p = sub.add_parser("fit-chain", help="fit the node values of a chain of point sources (a line through the "
                                         "star) to a finished run's final image, through its winners with KLIP-FM")
    _add_near_args(p)
    p.add_argument("--run-dir", default="last", help="the finished run (default: the newest on record)")
    p.add_argument("--chain", default=None, metavar="FILE",
                   help="one source per line: rho (arcsec), PA (deg), contrast per unit node value, and a node "
                        "label; a source shared by two nodes is on two lines. Without labels the whole chain is "
                        "one node and the fit scales it")
    p.add_argument("--line-nodes", default=None, metavar="FILE",
                   help="in place of --chain: a node file (as --subtract-line reads), one basis per node and "
                        "half-line, piecewise linear and held at the last node to rho_max; the fitted ridge "
                        "contrasts go to line_fitted.txt, ready for --subtract-line")
    p.add_argument("--per-night", action="store_true",
                   help="then one amplitude per night for the fitted chain, its shape held fixed (per_night.json)")
    p.add_argument("--line-pa", type=float, default=None, metavar="PA",
                   help="PA of the line the fit region follows (default: the chain's own line)")
    p.add_argument("--halfwidth", type=float, default=12.0, metavar="PX",
                   help="fit the pixels within PX of the line (default 12)")
    p.add_argument("--rho-range", type=float, nargs=2, default=None, metavar=("RMIN", "RMAX"),
                   help="fit only the pixels RMIN-RMAX arcsec from the star, e.g. 0.95 3.07 (default: the run's "
                        "annuli)")
    p.add_argument("--mask", type=float, nargs=3, action="append", metavar=("RHO", "PA", "RADIUS"),
                   help="leave a circle out of the fit and the noise: RHO (arcsec), PA (deg), RADIUS (px; 0 for "
                        "1.5 FWHM); may repeat")
    p.add_argument("--mask-candidates", default="", metavar="TIERS",
                   help="also leave out the run's candidates (cand/candidates.txt) of these tiers, e.g. A or AB. "
                        "The line's own peaks can be listed as candidates, so name only tiers that are not it")
    p.add_argument("--no-mask-known", action="store_true", help="keep the run's known sources in the fit")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="output directory (default <run>_chainfit beside the run, which is left as it is)")
    p.set_defaults(func=cmd_fit_chain)

    p = sub.add_parser("rereduce", help="a finished run's winners reduced again on other frames (a line "
                                        "subtracted, planets added), stitched with the run's weights, with the "
                                        "S/N map and the final verify, beside the run")
    _add_near_args(p)
    p.add_argument("--run-dir", default="last", help="the finished run (default: the newest on record)")
    p.add_argument("--verify-sources", default=None, metavar="FILE",
                   help="the final verify's injections: one per line, rho (arcsec), PA (deg) and contrast, all of "
                        "them in every annulus' injected reduction (default: the winners' validation injections, "
                        "as the run stitched them, see --stitch-injections)")
    _stitch_args(p, rerun=True)
    p.add_argument("--no-verify", action="store_true", help="the stitch and its S/N map only")
    p.add_argument("--verify-snr", default=None, choices=["search", "idl"],
                   help="how the verify and the S/N map measure (default: as the run did, idl for a run saved "
                        "before 2026-11-01). search: the search's own S/N under the run's rules and the pixel mask "
                        "in every ring and map. idl: the IDL verify's S/N and no mask outside the search")
    p.add_argument("--base-winners", default=None, metavar="FILE",
                   help="a composite: each listed annulus reduces another configuration in place of the run's "
                        "winner.  One per line: the annulus (from 1), the run (a directory, or a run beside this "
                        "one) and the evaluation (from 0, as rescore.json counts).  The run's verify injections, "
                        "weights, tiles and pads stay")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="output directory (default <run>_rereduce beside the run, which is left as it is, or "
                        "<run>_composite with --base-winners)")
    p.add_argument("--overwrite", action="store_true", help="write over an earlier re-reduction in --out")
    p.set_defaults(func=cmd_rereduce)

    p = sub.add_parser("rescore", help="candidates for annuli of a finished run (its own and other runs' winners and "
                                       "validation entrants) scored on its stitch, the other annuli at its winners, "
                                       "on common draws: the first piece of refine")
    _add_near_args(p)
    p.add_argument("--run-dir", default="last", help="the base run, whose stitch and winners the candidates are "
                                                     "scored in (default: the newest on record)")
    p.add_argument("--entrants-from", nargs="*", default=[], metavar="RUN",
                   help="other finished runs of the same frames, annuli and search space, whose candidates enter "
                        "too (the base run's always do)")
    p.add_argument("--annuli", default="all", help="the annuli to score, 1-based: 1-7, 2,3,5 or all (default all)")
    p.add_argument("--draws", type=int, default=None,
                   help="common draws per annulus (default 50, or all of --draws-from's)")
    p.add_argument("--tiles", default=None, choices=["apart", "together"],
                   help="apart: each tile's sources reduced and stitched on their own and measured there, at the "
                        "contrast set with them alone (the default for a new directory).  together: all of a draw's "
                        "sources in each reduction, as re-scores before 2026-11-11 did.  A directory started before "
                        "keeps its own, and with --draws-from the default is that re-score's")
    p.add_argument("--base-winners", default=None, metavar="FILE",
                   help="a composite base: each listed annulus' base winner is another configuration, for the "
                        "stitch, the neighbors' reductions and the paired differences, and the base run's own winner "
                        "enters as a candidate.  One per line: the annulus (from 1), the run (a directory, the base "
                        "run, a run of --entrants-from or a run beside the base run) and the evaluation (from 0, as "
                        "rescore.json counts)")
    p.add_argument("--candidates", default=None, metavar="FILE",
                   help="only these candidates enter (lines as --base-winners), with each annulus' base winner. An "
                        "annulus without a line scores its base winner alone")
    p.add_argument("--draws-from", default=None, metavar="DIR",
                   help="an earlier re-score of the same base run: its draws, their rule and its contrasts, without "
                        "a calibration, its first --draws of each annulus.  Each annulus' base winner is then "
                        "compared with DIR's, draw for draw and tile by tile")
    p.add_argument("--top", type=int, default=0,
                   help="each run's best N configurations of its history also enter (default 0: the winners and the "
                        "validation's entrants)")
    p.add_argument("--target", type=float, nargs=2, default=[5.0, 8.0], metavar=("LO", "HI"),
                   help="the median S/N the base winners should read on the stitch at each tile's contrast "
                        "(default 5 8)")
    p.add_argument("--n-cal", type=int, default=8, help="draws per step of each tile's contrast calibration "
                                                        "(default 8)")
    p.add_argument("--seed", type=int, default=1, help="seed of the draws (default 1)")
    p.add_argument("--sources-per-tile", type=int, default=None, metavar="N",
                   help="at most N sources in each tile of a draw (default: the run's number in each tile)")
    p.add_argument("--clear-fwhm", type=float, default=None, metavar="F",
                   help="the distance between the tiles' sources, in FWHM (default: the run's sampler's, 2).  "
                        "Without either option, an annulus whose draws find no place falls back to 1.5 FWHM, then "
                        "to one source in each neighbor's tile")
    p.add_argument("--window", type=int, default=None,
                   help="reductions in flight (default: enough to fill the reducer's workers)")
    pads = p.add_mutually_exclusive_group()
    pads.add_argument("--overlap-pads", dest="overlap_pads", action="store_const", const=True, default=None,
                      help="pad the stitch's tile edges between overlapping annuli (default: the base run's pads)")
    pads.add_argument("--no-overlap-pads", dest="overlap_pads", action="store_const", const=False,
                      help="no pads between overlapping annuli, as rereduce --no-overlap-pads")
    p.add_argument("--verify-snr", default=None, choices=["search", "idl"],
                   help="how each source is measured (default: as the base run's verify)")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="output directory (default <run>_rescore beside the run).  A re-score started again on "
                        "the same directory resumes")
    p.add_argument("--card", action="store_true",
                   help="the injected reductions may go to the card workers, as the search's draws do (default: "
                        "the CPU workers, as rereduce)")
    p.add_argument("--plan", action="store_true", help="count the candidates and the reductions, reduce nothing")
    p.set_defaults(func=cmd_rescore)

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

    p = sub.add_parser("render", help="draw every evaluation's panel after the fact, including the ones the "
                                      "live display skipped (from the run's log and saved crops)")
    _add_near_args(p)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--every", type=int, default=1,
                   help="every Nth evaluation of each annulus (its last is always drawn)")
    p.add_argument("--annulus", type=int, nargs="+", default=None, help="only these annuli (1-based)")
    p.add_argument("--dpi", type=int, default=100, help="text scale of the panel; 100 = the live panel's")
    p.add_argument("--out", default=None,
                   help="output directory (default <run-dir>/steps_rebuilt; the live steps/ and "
                        "opt_steps.gif are left as they are)")
    p.add_argument("--no-movie", action="store_true", help="skip <out>/opt_steps.gif")
    p.add_argument("--with-data", action="store_true",
                   help="load the run's frames, from the data arguments as rereduce takes them (the ones left out "
                        "filled from the run's record), to draw each frame's injected-PSF model, as the live "
                        "frames did.  Without it the frames draw the one the run saved at each annulus' end "
                        "(runs since 2026-11-10)")
    p.set_defaults(func=cmd_render)

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
    """Parse a command line and run its command.

    Parameters
    ----------
    argv : sequence of str or None, default None
        The command line, without the program's name.  None takes ``sys.argv``.
    """
    a = _build_parser().parse_args(argv)
    a._argv = argv
    a.func(a)


if __name__ == "__main__":
    main()
