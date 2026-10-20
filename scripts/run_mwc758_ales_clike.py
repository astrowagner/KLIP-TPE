#!/usr/bin/env python3
"""run_mwc758_ales_clike.py -- tutorial 7's search, tuned for a companion with MWC 758c's spectrum, in a
space that holds the ALES paper's reduction of the 2019 November 14 night (Wagner et al. 2023).

Tutorial 7 injects sources with the star's spectrum, one contrast in every wavelength group, and its winner
(5-px high-pass, a zone the width of the 0.4-0.8" annulus) suits them: c, a much redder source, comes out at
S/N 1.9 in it against 3.6-3.7 in the paper's images.  This run differs in four things:

* the injected sources have c's spectrum: a power law fitted to the c contrasts the paper's IDL injection
  tests used for November 14 (``MWC758_spec_2_Lp_fix.sav``), so a group gets the source at
  (lambda / 3.8 um)^alpha times the contrast quoted, which is c's at L' (``--alpha`` sets the index,
  ``--spectrum star`` is tutorial 7's flat spectrum);
* the KLIP zone may reach beyond the annulus (``zone_in``, ``zone_out``, 0-8 px each) and the residuals'
  second high-pass is a searched switch (``post_hp``), so the paper's ADI step is in the space: 15-frame
  bins, an 11-px high-pass, 15 modes, 6 sectors of one 5-27 px zone, references 0.01 lambda/D away, no
  frame cuts, no second high-pass;
* that configuration, on the 3.4-4.15 um groups, is the seed: trial 0, and the configuration the
  contrast is calibrated on;
* with both nights the partitions are weighted by the square root of their exposure, so January 5 (18 % of
  the exposure) no longer gets most of the weight when the search keeps more of its groups.

The metric, its matched filter (n2_g10, 3.50 um), the annulus, the budget and the validation are tutorial 7's,
so the two runs' S/N for c are on one yardstick.  ``--nights nov14`` searches November 14 alone, the night of
the paper's Fig. 1b; ``--sdi`` (one night only) adds the other groups' frames as a searched SDI pool.

  python3 scripts/run_mwc758_ales_clike.py                       # both nights, 3000 evaluations
  python3 scripts/run_mwc758_ales_clike.py --nights nov14        # November 14 alone
  python3 scripts/run_mwc758_ales_clike.py --quick               # a 12-evaluation smoke test first

Run from the repository root.  The run directory defaults to tutorials/runs/mwc758_ales_clike[_nov14][_sdi].
The same command again resumes an interrupted run.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
from klip_tpe import CalibrationConfig, RunConfig, Runner, ValidationConfig, datasets   # noqa: E402
from klip_tpe.instruments import ales                                                     # noqa: E402
from klip_tpe.metrics import MawetPeakSNR                                                 # noqa: E402
from klip_tpe.reducer import set_klip_gram                                                # noqa: E402

RHO, PA = 0.603, 228.5          # MWC 758c in the 2019 November 14 ALES data
SAV = "/Users/kevinwagner/Library/CloudStorage/Dropbox/Astrophysics/Research/MWC 758/MWC758_ALES/MWC758_spec_2_Lp_fix.sav"
ALPHA_DEFAULT = 6.76            # the fit to SAV (+- 1.22), when the file is not at hand

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--nights", choices=("both", "nov14"), default="both")
ap.add_argument("--spectrum", choices=("c", "star"), default="c")
ap.add_argument("--sav", default=SAV, help="c's contrasts (IDL save file with waves, contrasts, errors)")
ap.add_argument("--alpha", type=float, default=None, help="power-law index of c's contrast (default: fit to --sav)")
ap.add_argument("--weighting", choices=("sqrt_texp", "equal"), default=None,
                help="partition weights (default sqrt_texp with both nights)")
ap.add_argument("--sdi", action="store_true", help="a searched SDI pool (needs --nights nov14)")
ap.add_argument("--no-seed-paper", action="store_true", help="seed with tutorial 7's baseline instead")
ap.add_argument("--n-iter", type=int, default=3000)
ap.add_argument("--n-init", type=int, default=200)
ap.add_argument("--workers", default="auto")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--run-dir", default=None)
ap.add_argument("--show", choices=("window", "none"), default="window")
ap.add_argument("--quick", action="store_true", help="12 evaluations, 2 validation draws: a smoke test")
a = ap.parse_args()
if a.sdi and a.nights != "nov14":
    ap.error("--sdi needs --nights nov14: an SDI pool would lend one night's frames to the other's groups")

# ---------------------------------------------------------------------------------------- c's spectrum
spec_label = None
spectrum = None
if a.spectrum == "c":
    if a.alpha is not None:
        alpha, how = float(a.alpha), "given"
    elif os.path.exists(a.sav):
        from scipy.io import readsav
        from scipy.optimize import curve_fit
        s = readsav(a.sav)
        w, c, e = (np.asarray(s[k], float) for k in ("waves", "contrasts", "errors"))
        p, cov = curve_fit(lambda x, c38, al: c38 * (x / 3.8) ** al, w, c, p0=[1e-5, 4.0],
                           sigma=np.where(e > 0, e, np.median(e)))
        alpha, how = float(p[1]), f"fit to {os.path.basename(a.sav)} ({w.min():.2f}-{w.max():.2f} um): " \
                                  f"C(3.8 um) {p[0]:.2e}, alpha {p[1]:.2f} +- {np.sqrt(cov[1, 1]):.2f}"
    else:
        alpha, how = ALPHA_DEFAULT, f"{os.path.basename(a.sav)} not found, the fit made from it on 2026-10-18"
    spectrum = lambda lam_um, al=alpha: (lam_um / 3.8) ** al
    spec_label = f"MWC 758c: contrast proportional to (lambda / 3.8 um)^{alpha:.2f}, {how}"
    print("injected spectrum:", spec_label, flush=True)

# ---------------------------------------------------------------------------------------------- data
roles = None
if a.nights == "nov14":
    roles = [k for k in datasets.DATASETS["lbti_ales_mwc758"] if k.endswith("_20191114")]
files = datasets.fetch("lbti_ales_mwc758", roles=roles)
# the labels of the two-night data set (n1 = Jan 5, n2 = Nov 14), also when November 14 is fetched alone
dsets, lam, fwhm, table = ales.load_release(files, nights=["20191114"] if a.nights == "nov14" else None,
                                           labels={"20190105": "n1", "20191114": "n2"})
weighting = a.weighting or ("sqrt_texp" if a.nights == "both" else "equal")
red = ales.make_reducer(dsets, lam, fwhm, table=table, max_workers=a.workers, representative="n2_g10",
                        weighting=weighting, sdi=a.sdi, spectrum=spectrum, spectrum_norm=3.8e-6,
                        spectrum_label=spec_label)
set_klip_gram(red, "zone")

EDGES_AS = [0.4, 0.8]
EDGES = [e / red.pxscale for e in EDGES_AS]                    # 11.59, 23.19 spaxels
# the paper's ADI step (reduce_ales_mwc758_2.pro, MWC758_ALES @ 9ae925e) in this space's terms
PAPER = {"bin": 15, "filter": 11, "k_klip": 15, "n_ang": 6, "angsep": 0.01, "anglemax": 360.0,
         "corr_thresh": 0.0, "noise_max": 3.0, "coronoise_max": 3.0,
         "zone_in": EDGES[0] - 5.0, "zone_out": 27.0 - EDGES[1], "post_hp": 0}
space = ales.make_space(red, bin_range=(5, 30), zone_range=(0.0, 8.0), search_post_hp=True,
                        defaults=None if a.no_seed_paper else PAPER)
n_part = len(red.partitions())
space.project = ales.make_guard(red, max_excluded=n_part // 2, prefer_m=3.95e-6)
objective, sampler = ales.default_config(red, known=[(RHO, PA)], forbidden_pa=[(PA, 25.0)])
# the seed's groups: 3.4-4.15 um (g09-g19) on every night, the paper's band for c.  The calibration keeps the
# seed's k (scan_k=False): trial 0 is then the paper's configuration itself, not its best k.
seed_sel = {} if a.no_seed_paper else {f"sel_{p}": (1 if int(p.split("_g")[1]) >= 9 else 0) for p in red.partitions()}
print(f"{len(space.names)} dimensions: {', '.join(n for n in space.names if not n.startswith('sel_'))} "
      f"+ {len(space.selection_dims)} group flags; weights {weighting}", flush=True)

tag = "_nov14" if a.nights == "nov14" else ""
tag += "_sdi" if a.sdi else ""
tag += "_star" if a.spectrum == "star" else ""
run_dir = os.path.abspath(a.run_dir or os.path.join("tutorials", "runs", f"mwc758_ales_clike{tag}" + ("_quick" if a.quick else "")))
if a.quick:
    cfg = RunConfig(ann_edges=EDGES, n_iter=12, n_init=6, seed=a.seed, n_remeasure=1, defaults=seed_sel,
                    validation=ValidationConfig(n_top=2, n_valid=2),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=1, scan_k=a.no_seed_paper),
                    fm_curve=False,
                    param_verify=False, notes="c-coloured smoke test")
else:
    cfg = RunConfig(ann_edges=EDGES, n_iter=a.n_iter, n_init=a.n_init, seed=a.seed, n_remeasure=3, defaults=seed_sel,
                    validation=ValidationConfig(n_top=6, n_valid=8),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=2, scan_k=a.no_seed_paper),
                    fm_curve=True,
                    notes=f"tutorial 7 with c-coloured injections ({spec_label or 'star spectrum'}); the paper's ADI as seed")
# the same command again resumes the run (Runner.run picks up its checkpoint); a run started with another
# spectrum is not continued with this one
setup = os.path.join(run_dir, "run_setup.json")
if os.path.exists(setup):
    old = (json.load(open(setup)).get("reducer") or {}).get("injection_spectrum") or {}
    new = red.describe().get("injection_spectrum") or {}
    of, nf = old.get("factors") or {}, new.get("factors") or {}
    if set(of) != set(nf) or any(abs(of[k] - nf[k]) > 1e-5 * max(abs(of[k]), 1e-9) for k in of):
        sys.exit(f"{run_dir} was started with another injected spectrum ({old.get('label', 'the star spectrum')}); "
                 f"give --run-dir for a new run")
callbacks = []
if a.show != "none":
    from klip_tpe.display import LiveDisplay
    callbacks.append(LiveDisplay(run_dir, show="window", every=5, pdf_every=100, movie_every=500))
t0 = time.time()
results = Runner(red, space, objective, sampler, cfg, run_dir, callbacks=callbacks).run()
print(f"{(time.time() - t0) / 3600:.2f} h", flush=True)

# ----------------------------------------------------------------------------------- what it decided
from astropy.io import fits                                    # noqa: E402
r = results[0]
metric = MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel, declip_nsig=None)
blk = r.winner_config["per_partition"][red.partitions()[0]]
keep = ("bin", "filter", "k_klip", "n_ang", "angsep", "anglemax", "corr_thresh", "noise_max", "coronoise_max",
        "zone_in", "zone_out", "post_hp", "nkeep_sdi")
print(f"winner (eval {r.winner_index + 1}, validated {r.validated}): S/N {r.winner_score:.2f} for c-coloured "
      f"sources at contrast {r.contrast:.3g} (L')")
print("  " + "  ".join(f"{k}={blk[k]:.3g}" if isinstance(blk[k], float) else f"{k}={blk[k]}" for k in keep if k in blk))
print("  the paper's: " + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in PAPER.items()))
for n in sorted({p.split("_")[0] for p in red.partitions()}):
    on = [p.split("_")[1] for p in r.partitions if p.startswith(n + "_")]
    print(f"  {n}: {len(on)} groups kept: {' '.join(on)}")
img = fits.getdata(os.path.join(run_dir, "annulus01", "best_clean.fits"))
c_snr = float(metric.per_source(img, None, [RHO], [PA])[0])
print(f"MWC 758c in the winner's image: S/N {c_snr:.2f} (tutorial 7's winner 1.9; the paper's Nov 14 images 3.6-3.7)")
json.dump({"c_snr_winner": c_snr, "winner_score": r.winner_score, "contrast": r.contrast, "spectrum": spec_label,
           "winner_block": {k: blk.get(k) for k in keep}, "partitions": list(r.partitions), "paper_seed": PAPER},
          open(os.path.join(run_dir, "clike_summary.json"), "w"), indent=1, default=float)
