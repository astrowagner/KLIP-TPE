#!/usr/bin/env python
"""The paper's MIRI figure: HIP 65426 b at F1140C under four configurations.

    python3 miri_fig.py                     # MIRI_DATA, MIRI_RUNS, MIRI_TAG as below
    MIRI_DATA=~/Data/JWST/hip65426_miri/mastDownload MIRI_RUNS=.. MIRI_TAG=v8 python3 miri_fig.py

Carter et al. (2023)'s reduction as their configuration file sets it out (``carter_published``
of scripts/library_ablation.py: their frame selection, the whole illuminated field as one zone,
ADI+RDI with every mode, mean-combined), the configured default on pyKLIP, and the validated
winner of each engine's search (MIRI_TAG, default v8) -- pyKLIP
(``miri_HIP-65426_F1140C_v8_pyklip``) and the built-in engine (``..._v8_klip``) -- each reduced
clean, with the companion's S/N measured at Carter et al.'s F1140C position.  The runs are
rebuilt by the code that ablated them (``library_ablation._rebuild``), which refuses a setup
that does not reproduce the run, so these are the configurations Table~\\ref{tab:miri} scores.

Every panel is shown from the inner working angle out to the searched annulus' outer edge.
Inside the annulus (0.36-0.74") the default and the winners, which were only ever reduced over
the annulus, are reduced once more over that inner zone with the same parameters and the two
zones are joined at the annulus' inner edge (dotted); Carter et al.'s single zone already
covers it.  The S/N above each column is the companion's in the scored reduction
(Table~\\ref{tab:miri}) -- the forward-model S/N of klip_tpe.companion, with the search metric's
kept in the JSON; the maps are computed on the joined image.  The stretch is set on the annulus,
so the brighter residuals inside it saturate.

Writes ``figs/f13_miri.pdf`` and ``figs/f13_miri.json`` (the planet S/N of each panel).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.dirname(HERE))

import library_ablation as LA                                      # noqa: E402
from klip_tpe import CalibrationConfig, RunConfig, Runner, ValidationConfig   # noqa: E402

def _default_data():
    """The archive calints.  ``~/Data/JWST/hip65426_miri`` also holds a re-reduction from the
    raw ramps (``reproc/``), which a recursive search would find beside MAST's products, so
    point at ``mastDownload/`` when it is there."""
    root = os.path.expanduser("~/Data/JWST/hip65426_miri")
    md = os.path.join(root, "mastDownload")
    return md if os.path.isdir(md) else root


DATA = os.path.expanduser(os.environ.get("MIRI_DATA", "") or _default_data())
RUNS = os.path.expanduser(os.environ.get("MIRI_RUNS", os.path.dirname(HERE)))
#: which MIRI runs the paper reports: v8 (archive calints, static hot pixels repaired, every
#: draw recorded); MIRI_TAG=v7 rebuilds the figure of 27 Sep from the runs before the repair
MIRI_TAG = os.environ.get("MIRI_TAG", "v8")


def miri_run_dir(engine: str, tag: str = None) -> str:
    return os.path.join(RUNS, f"miri_HIP-65426_F1140C_{tag or MIRI_TAG}_{engine}")


def rebuild_run(engine: str, log=print, tag: str = None):
    """A finished MIRI run rebuilt by the code that ablated it, with its whole run_setup.json
    in hand, so the rebuild repairs hot pixels exactly when the run did.  Returns
    ``(run_dir, full_setup, config, rebuilt)``."""
    run_dir = miri_run_dir(engine, tag)
    with open(os.path.join(run_dir, "run_setup.json")) as f:
        full = json.load(f)
    setup = full["config"]
    a = type("A", (), dict(data=DATA, workers="auto", backend=engine))()
    return run_dir, full, setup, LA._rebuild("miri", setup, a, log, full_setup=full)
PANELS = (("pyklip", "carter_published", "published configuration\n(pyKLIP)"),
          ("pyklip", "default", "configured default\n(pyKLIP)"),
          ("pyklip", "winner", "optimized\n(pyKLIP)"),
          ("klip", "winner", "optimized\n(built-in engine)"))


class _binned_as:
    """Bin the frames as the annulus does while reducing the zone inside it.

    The built-in engine lets a temporal bin span at most half a FWHM of rotation at the
    zone's outer radius, so the inner zone (outer radius 0.74") would take bins of up to 14
    deg -- wide enough to reach across the 9.4 deg roll change, which the engine refuses.  The
    panel shows the annulus' configuration, so the inner zone gets the annulus' bins."""

    def __init__(self, red, outrad):
        self.subs = [r for r in getattr(red, "reducers", {}).values() if hasattr(r, "dth_max_deg")]
        self.val = [float(r.dth_max_deg(outrad)) for r in self.subs]

    def __enter__(self):
        for r, v in zip(self.subs, self.val):
            r.dth_max_deg = (lambda outrad, _v=v: _v)

    def __exit__(self, *exc):
        for r in self.subs:
            del r.dth_max_deg
        return False


#: Table 3's other two rows, measured for the companion but not drawn
TABLE_ONLY = (("pyklip", "carter_published_annuli"), ("klip", "default"))

#: HIP 65426 b in F1140C by negative injection, kept so the figure and companion_tests.py
#: subtract and measure the same values (as run_demos.NEGFC_HIP does for F444W).  This name
#: holds the fit on the frames of 27 Sep (archive calints, no hot-pixel repair, 0.11033"/px);
#: any other frame treatment gets its own file, see :func:`negfc_cache_path`.
NEGFC_F1140C = os.path.join(RUNS, "_negfc", "hip65426b_f1140c.json")
_NEGFC_LEGACY_PX = 0.11032674199848376


def negfc_cache_path(red=None) -> str:
    """Where the companion fit for ``red``'s frames is cached.  A fit belongs to the frames it
    was made on -- repairing a hot pixel 0.26" from the companion, or another pipeline's
    calints, moves it -- so the cache is named by the loader's frame signature, except for
    the frames the original cache was made on, which keep :data:`NEGFC_F1140C`."""
    from klip_tpe.backends.spaceklip import frames_provenance, frames_signature
    rec = frames_provenance(red) if red is not None else None
    if rec is None or (not (rec.get("hot_pixels") or {}).get("applied")
                       and abs(float(rec.get("pxscale") or 0.0) - _NEGFC_LEGACY_PX) < 1e-6):
        return NEGFC_F1140C
    return os.path.join(RUNS, "_negfc", f"hip65426b_f1140c_{frames_signature(rec)}.json")


def miri_negfc(runner, space, red, px, log=print, refit=False):
    """HIP 65426 b's position and contrast in F1140C, fitted once by negative injection in
    pyKLIP's reference-star (RDI) basis over every reference frame -- a basis that does not
    hold the companion -- and cached per frame treatment (:func:`negfc_cache_path`).
    ``runner``/``space`` must be a pyKLIP run's.  Returns ``(rho, pa, contrast)``.  The fit
    leaves ~16% of the stamp's energy (NIRCam: 0.4%): the template matches this companion
    less well."""
    path = negfc_cache_path(red)
    if os.path.exists(path) and not refit:
        with open(path) as f:
            d = json.load(f)
        return float(d["rho"]), float(d["pa"]), float(d["contrast"])
    from klip_tpe.companion import fit_negative_companion
    if "maxnumbasis" not in space.names:
        raise ValueError("the F1140C companion is fitted in pyKLIP's RDI basis: pass the pyKLIP run's runner")
    pool = int(np.asarray(space.hi, float)[space.names.index("maxnumbasis")])
    choices = list(space.params[space.names.index("mode")].choices)
    c_rdi = space.decode(LA._vec(runner, space.default_vector(), mode=float(choices.index("RDI")), maxnumbasis=pool))
    f = fit_negative_companion(lambda s_: np.asarray(runner._reduce(c_rdi, s_, tag="fig_negfc").image, float),
                               LA.PLANET[0], LA.PLANET[1], 4.95e-4, red.fwhm, px,
                               angle_convention=red.angle_convention, max_evals=30, log=log)
    from klip_tpe.backends.spaceklip import frames_provenance, frames_signature
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(dict(f, engine="pyklip", mode="RDI", maxnumbasis=pool,
                       frames_signature=frames_signature(frames_provenance(red))), fh, indent=1)
    return float(f["rho"]), float(f["pa"]), float(f["contrast"])


def reduce_all(log=print, inner=True, companion=True):
    """Clean reductions of the inner annulus (and Carter's single zone), per panel; with
    ``inner``, also the zone between the inner working angle and the annulus, reduced with the
    same parameters, for the panels whose own zone starts at the annulus.

    With ``companion``, HIP 65426 b's forward-model S/N (klip_tpe.companion) in every panel and
    in Table 3's other two rows: its position and contrast are fitted once by negative injection
    in pyKLIP's reference-star (RDI) basis, which does not hold the companion, and each
    configuration then fits its own contrast at that position.  Its noise ring at 0.82" holds
    only about seven usable apertures (14 around the ring, less the quadrant boundaries and the
    companion's neighbours), so these values carry ~30% uncertainty; the search metric's (``snr``)
    is kept beside them."""
    from klip_tpe.companion import companion_snr, fit_negative_companion
    out = {}
    pos = None
    want = {(e, n) for e, n, _ in PANELS} | set(TABLE_ONLY)
    for engine in ("pyklip", "klip"):
        run_dir, _full, setup, rb = rebuild_run(engine, log)
        with open(os.path.join(run_dir, "final_results.json")) as f:
            fr = json.load(f)["annuli"][0]              # the annulus that holds the companion
        red, ann, obj, samp, px, space = (rb[k] for k in ("red", "ann", "obj", "samp", "px", "space"))
        m = obj.metric
        cfg = RunConfig(ann_edges=[float(v) for v in ann], n_iter=1, n_init=1, seed=int(setup.get("seed", 21)),
                        validation=ValidationConfig(n_top=1, n_valid=1),
                        calibration=CalibrationConfig(forced=[float(fr["contrast"])]), n_remeasure=1,
                        n_sources=setup.get("n_sources"), save_fits=False, save_eval_images=False,
                        fm_curve=False, verify=False)
        runner = Runner(red, space, obj, samp, cfg, os.path.join(run_dir, "_figure"), log=lambda s: None)
        runner.ia, runner.contrast = 0, float(fr["contrast"])
        x_def = space.default_vector()
        xs = {"winner": (LA._vec(runner, LA._x_from_params(space, fr["winner_config"]["params"], x_def)), None),
              "default": (LA._vec(runner, x_def), None)}
        # Carter et al. (2023)'s reduction as their configuration file sets it out: their frame
        # set, the whole illuminated field as one zone, every mode, mean-combined (library_ablation
        # CARTER_PUBLISHED); the central 81x81 px are what the figure shows and what is scored.
        field = {}
        if "maxnumbasis" in space.names:
            if companion and pos is None:
                # the companion's position and contrast, once, in a basis that does not hold it
                pos = miri_negfc(runner, space, red, px, log=log)
            if {(engine, "carter_published"), (engine, "carter_published_annuli")} & want:
                a_pub = type("A", (), dict(data=DATA, workers="auto", backend="pyklip", run_dir=run_dir,
                                           hot_pixels="auto"))()
                rFF, poolC, zFF, restore = LA.carter_published_setup(a_pub, setup, _full, space, obj, samp, cfg, log)
                rFF.ia, rFF.contrast = 0, runner.contrast
                cpub = LA.carter_published_config(space, runner, x_def, poolC)
                h = (int(np.shape(next(iter(red.reducers.values())).data.cube)[-1]) - 1) // 2
                field = {"carter_published": (cpub, zFF), "carter_published_annuli":
                         (cpub, (float(fr["inrad"]), float(fr["outrad"])))}
                xs.update({k: (None, "field") for k in field})

        def _reducer(name, x, zone):
            """``(configuration, zone marker, reduce(sources, tag) -> image)`` for one panel."""
            if name in field:
                cf, zf = field[name]
                return cf, zf, (lambda s_, tag, cf=cf, zf=zf:
                                LA.central(np.asarray(rFF._reduce(cf, s_, tag=tag, zone=zf).image, float), h))
            cr = space.decode(x)
            return cr, zone, (lambda s_, tag, cr=cr, zone=zone:
                              np.asarray(runner._reduce(cr, s_, tag=tag, zone=zone).image, float))

        for name, (x, zone) in xs.items():
            if (engine, name) not in want:
                continue
            c, zone, red_fn = _reducer(name, x, zone)
            img = red_fn(None, f"fig_{engine}_{name}")
            snr = float(m.per_source(img, None, [LA.PLANET[0]], [LA.PLANET[1]])[0])
            iwa = LA.IWA_AS / float(px)
            fm = None
            if companion and pos is not None:
                fn = (lambda s_, red_fn=red_fn, tag=f"fig_{engine}_{name}_fm": red_fn(s_, tag))
                fc = fit_negative_companion(fn, pos[0], pos[1], pos[2], red.fwhm, px,
                                            angle_convention=red.angle_convention, fit_position=False,
                                            rel_grid=(0.6, 0.8, 1.0, 1.2, 1.4, 1.7), clean=img)
                r = companion_snr(fn, pos[0], pos[1], fc["contrast"], red.fwhm, px, m.kernel([pos[0]]),
                                  pixel_mask=m.pixel_mask, angle_convention=red.angle_convention, clean=img)
                fm = dict(snr=r["snr"], snr_band=r["snr_band"], sigma_range=list(r["sigma_range"]), n_ap=r["n_ap"],
                          contrast=fc["contrast"], removed_fraction=fc["removed_fraction"],
                          rho_as=pos[0], pa_deg=pos[1])
            params = {k: (v if isinstance(v, str) else float(v)) for k, v in c.params.items()
                      if k in space.names or k == "comb_type"}
            if (engine, name) in TABLE_ONLY:
                out[(engine, name)] = dict(snr=snr, snr_fm=None if fm is None else fm["snr"], fm=fm, params=params,
                                           table_only=True)
            else:
                # the zone inside the annulus, same parameters: display only, never scored
                inn = None
                if inner and zone is None:
                    with _binned_as(red, float(fr["outrad"])):
                        inn = np.asarray(runner._reduce(c, None, tag=f"fig_{engine}_{name}_inner",
                                                        zone=(iwa, float(fr["inrad"]))).image, float)
                out[(engine, name)] = dict(img=img, inner=inn, snr=snr, snr_fm=None if fm is None else fm["snr"],
                                           fm=fm, px=float(px), fwhm=float(red.fwhm),
                                           inrad=float(fr["inrad"]), outrad=float(fr["outrad"]), iwa=iwa, params=params)
            log(f"  {engine:6s} {name:11s} planet S/N metric {snr:5.2f}, forward model "
                f"{'--' if fm is None else format(fm['snr'], '.2f')}  {params}")
        if field:
            restore()
    return out


def plot(res, path):
    import figs as F                       # the paper's style and helpers (reads RUNS_DIR too)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, axes = plt.subplots(2, len(PANELS), figsize=(7.1, 3.9))
    for j, (engine, name, title) in enumerate(PANELS):
        r = res[(engine, name)]
        img, px = r["img"].copy(), r["px"]
        ny, nx = img.shape
        cx, cy = F.star_center(img.shape)
        yy, xx = np.mgrid[0:ny, 0:nx]
        rr = np.hypot(xx - cx, yy - cy)
        ann = (rr >= r["inrad"]) & (rr <= r["outrad"])
        # the searched annulus as reduced; inside it, the same configuration over the zone in
        # to the inner working angle (Carter et al.'s single zone already reaches it)
        if r.get("inner") is not None:
            img = np.where(ann & np.isfinite(img), img, r["inner"])
        out = (rr < r["iwa"]) | (rr > r["outrad"])
        img[out] = np.nan
        # the stretch is set on the searched annulus, so the brighter residuals inside it
        # saturate instead of compressing the field Table 3 scores
        lo, hi = np.nanpercentile(img[ann & np.isfinite(img)], [1.0, 99.6])
        box = 1.08 * r["outrad"] * px
        mark = 0.8 * r["fwhm"] * px        # the ring clears the F1140C core (FWHM 0.37")
        s_q = r["snr_fm"] if r.get("snr_fm") is not None else r["snr"]
        F._show(axes[0, j], img, px, LA.PLANET, f"{title}\nS/N {s_q:.1f}", vlim=(lo, hi), box_as=box,
                mark_as=mark)
        m = F.snr_map(img, r["fwhm"], known=[LA.PLANET], px=px)
        m[out] = np.nan
        F._show(axes[1, j], m, px, LA.PLANET, "", vlim=(-4, 6), box_as=box, snr=True, mark_as=mark)
        for i in range(2):
            axes[i, j].add_patch(Circle((0, 0), r["inrad"] * px, fill=False, ec="w", lw=0.6, ls=":"))
            F.compass(axes[i, j])
    axes[0, 0].set_ylabel("image", fontsize=7.5)
    axes[1, 0].set_ylabel("S/N map", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    res = reduce_all()
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    plot(res, os.path.join(HERE, "figs", "f13_miri.pdf"))
    with open(os.path.join(HERE, "figs", "f13_miri.json"), "w") as f:
        json.dump({f"{e}_{n}": {"planet_snr_metric": r["snr"], "planet_snr_fm": r.get("snr_fm"),
                                "companion_fm": r.get("fm"), "params": r["params"],
                                "panel": not r.get("table_only", False)}
                   for (e, n), r in res.items()}, f, indent=1)
    print("  f13_miri.pdf")
