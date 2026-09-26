#!/usr/bin/env python
"""The paper's MIRI figure: HIP 65426 b at F1140C under four configurations.

    python3 miri_fig.py                     # MIRI_DATA, MIRI_RUNS as below
    MIRI_DATA=~/Data/JWST/hip65426_miri MIRI_RUNS=.. python3 miri_fig.py

Carter et al. (2023)'s MIRI choices over one zone (``carter_full`` of
scripts/library_ablation.py), the configured default on pyKLIP, and the validated winner of
each engine's v7 search -- pyKLIP (``miri_HIP-65426_F1140C_v7_pyklip``) and the built-in
engine (``..._v7_klip``) -- each reduced clean, with the companion's S/N measured by the
run's own metric at Carter et al.'s F1140C position.  The runs are rebuilt by the code that
ablated them (``library_ablation._rebuild``), which refuses a setup that does not reproduce
the run, so these are the configurations Table~\\ref{tab:miri} scores.

Every panel is shown from the inner working angle out to the searched annulus' outer edge.
Inside the annulus (0.36-0.74") the default and the winners, which were only ever reduced over
the annulus, are reduced once more over that inner zone with the same parameters and the two
zones are joined at the annulus' inner edge (dotted); Carter et al.'s single zone already
covers it.  The S/N above each column is the companion's in the scored reduction
(Table~\\ref{tab:miri}); the maps are computed on the joined image.  The stretch is set on the
annulus, so the brighter residuals inside it saturate.

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

DATA = os.path.expanduser(os.environ.get("MIRI_DATA", "~/Data/JWST/hip65426_miri"))
RUNS = os.path.expanduser(os.environ.get("MIRI_RUNS", os.path.dirname(HERE)))
PANELS = (("pyklip", "carter_full", "Carter et al. recipe\n(one zone, all frames, $k=6$)"),
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


def reduce_all(log=print, inner=True):
    """Clean reductions of the inner annulus (and Carter's single zone), per panel; with
    ``inner``, also the zone between the inner working angle and the annulus, reduced with the
    same parameters, for the panels whose own zone starts at the annulus."""
    out = {}
    for engine in sorted({e for e, _, _ in PANELS}):
        run_dir = os.path.join(RUNS, f"miri_HIP-65426_F1140C_v7_{engine}")
        with open(os.path.join(run_dir, "run_setup.json")) as f:
            setup = json.load(f)["config"]
        with open(os.path.join(run_dir, "final_results.json")) as f:
            fr = json.load(f)["annuli"][0]              # the annulus that holds the companion
        a = type("A", (), dict(data=DATA, workers="auto", backend=engine))()
        rb = LA._rebuild("miri", setup, a, log)
        red, ann, obj, samp, px, space = (rb[k] for k in ("red", "ann", "obj", "samp", "px", "space"))
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
        if "maxnumbasis" in space.names:
            pool = int(np.asarray(space.hi, float)[space.names.index("maxnumbasis")])
            mode = float(list(space.params[space.names.index("mode")].choices).index("ADI+RDI"))
            xs["carter_full"] = (LA._vec(runner, x_def, **dict(mode=mode, maxnumbasis=pool, **LA.CARTER)),
                                 (LA.IWA_AS / px, float(ann[-1])))
        for name, (x, zone) in xs.items():
            if (engine, name) not in {(e, n) for e, n, _ in PANELS}:
                continue
            c = space.decode(x)
            img = np.asarray(runner._reduce(c, None, tag=f"fig_{engine}_{name}", zone=zone).image, float)
            snr = float(obj.metric.per_source(img, None, [LA.PLANET[0]], [LA.PLANET[1]])[0])
            iwa = LA.IWA_AS / float(px)
            # the zone inside the annulus, same parameters: display only, never scored
            inn = None
            if inner and zone is None:
                with _binned_as(red, float(fr["outrad"])):
                    inn = np.asarray(runner._reduce(c, None, tag=f"fig_{engine}_{name}_inner",
                                                    zone=(iwa, float(fr["inrad"]))).image, float)
            out[(engine, name)] = dict(img=img, inner=inn, snr=snr, px=float(px), fwhm=float(red.fwhm),
                                       inrad=float(fr["inrad"]), outrad=float(fr["outrad"]), iwa=iwa,
                                       params={k: (v if isinstance(v, str) else float(v))
                                               for k, v in c.params.items() if k in space.names})
            log(f"  {engine:6s} {name:11s} planet S/N {snr:5.2f}  {out[(engine, name)]['params']}")
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
        F._show(axes[0, j], img, px, LA.PLANET, f"{title}\nS/N {r['snr']:.1f}", vlim=(lo, hi), box_as=box,
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
        json.dump({f"{e}_{n}": {"planet_snr": r["snr"], "params": r["params"]} for (e, n), r in res.items()},
                  f, indent=1)
    print("  f13_miri.pdf")
