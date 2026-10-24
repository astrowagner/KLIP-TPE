#!/usr/bin/env python3
"""run_mwc758_ales_clike.py -- tutorial 7's search, tuned for a companion with MWC 758c's spectrum, in a
space that holds the ALES paper's reduction of the 2019 November 14 night (Wagner et al. 2023).

Tutorial 7 injects sources with the star's spectrum, one contrast in every wavelength group, and its winner
(5-px high-pass, a zone the width of the 0.4-0.8" annulus) suits them: c, a much redder source, comes out at
S/N 1.9 in it against 3.6-3.7 in the paper's images.  This run differs in five things:

* the injected sources have c's spectrum: a power law fitted to the c contrasts the paper's IDL injection
  tests used for November 14 (``MWC758_spec_2_Lp_fix.sav``), so a group gets the source at
  (lambda / 3.8 um)^alpha times the contrast quoted, which is c's at L' (``--alpha`` sets the index,
  ``--spectrum star`` is tutorial 7's flat spectrum);
* the KLIP zone is the paper's for every evaluation (``--geometry paper``, the default): one 5-27 px
  annulus about the paper's zone centre (30, 30) in 6 sectors, while the injections and the score stay in
  tutorial 7's 0.4-0.8" annulus round c.  ``--free-sectors`` searches the number of sectors again, and
  ``--geometry searched`` searches the zone's reach beyond the annulus as well (``zone_in``, ``zone_out``,
  0-8 px each), which is how the two-night run of 2026-10-18 ran;
* the residuals' second high-pass is a searched switch (``post_hp``), so the paper's ADI step is in the
  space: 15-frame bins, an 11-px high-pass, 15 modes, references 0.01 lambda/D away, no frame cuts, no
  second high-pass;
* that configuration, on the 3.4-4.15 um groups, is the seed: trial 0, and the configuration the
  contrast is calibrated on;
* with both nights the partitions are weighted by the square root of their exposure, so January 5 (18 % of
  the exposure) no longer gets most of the weight when the search keeps more of its groups.

The metric, its matched filter (n2_g10, 3.50 um), the annulus, the budget and the validation are tutorial 7's,
so the two runs' S/N for c are on one yardstick.  ``--nights nov14`` searches November 14 alone, the night of
the paper's Fig. 1b; ``--sdi`` (one night only) adds the other groups' frames as a searched SDI pool.
The search may switch off up to half of the groups (10 of November 14's 20; the guard switches groups back on
past that).  The paper's ADI band, 3.4-4.15 um (g09-g19), is within that bound and its Fig. 1b, 3.8-4.15 um
(g15-g19, 5 of 20), is not: ``--max-excluded 15`` lets the search go as far as Fig. 1b.

``--fixed-groups 15-19`` holds the combination at Fig. 1b's band for every evaluation instead: only those
groups are loaded and there are no group flags.  The matched filter is then the band's middle group's (g17,
3.97 um; g10 lies outside the band), and with it the FWHM and lambda/D the metric, the sampler and the guards
use; ``--kernel-group gNN`` picks another.  c's S/N is then on that group's yardstick, so compare it with
images measured with the same filter (``compare_mwc758_fig1b_clike.py`` uses the run's own).

``--disk-mask MASK.fits`` (1 = disk, the frames' 62 x 62; ``scripts/make_disk_mask.py`` builds one from an image
of the disk) keeps the disk out of the metric's noise samples and the injections off it, for the search, the
calibration, the validation and the products alike (``ales.default_config(disk_mask=...)``).  A copy goes into
the run directory, and c's S/N at the end is measured with it too.

The two sources go at the 0.4-0.8" annulus' area-weighted mid radius, 0.632"; ``--inject-rho 0.6`` puts every
injection of every stage at 0.6", c's own separation (0.603"), with the annulus, its zone and its scoring as
they were (``RunConfig.inject_rho``).

``--angsep-min 0.01`` searches ``angsep`` from the paper's 0.01 lambda/D instead of 0 (``make_space(angsep_range=
...)``).  At 0, with ``anglemax`` at the night's PA span (127 deg), the reduction is another algorithm: one basis
per zone from all the frames, each target frame inside its own basis (the IDL's auto-fast rule), where any
positive angsep leaves the target out.  Proposals past the bound are clipped onto it, so the search lands there
often, and every c-tuned winner up to 2026-10-24 sat there: it keeps 0.18 of the paper's arm flux where the same
configuration at angsep 0.01 keeps 0.55, with c a little fainter as well.  ``--n-remeasure N`` averages N
injection draws per evaluation (3 by default, N + 1 reductions); at a faint cap the mean of three scatters by
about as much as the configurations differ.  ``--k-max K`` searches at most K modes: past about 15 the per-target
basis starts to fit the arms, which barely move from frame to frame (at the paper's step with the winner's cuts the
arm flux falls from 1.0 at 15 modes to 0.66 at 18 and 0.38 at 25).

  python3 scripts/run_mwc758_ales_clike.py                       # both nights, 3000 evaluations
  python3 scripts/run_mwc758_ales_clike.py --nights nov14        # November 14 alone
  python3 scripts/run_mwc758_ales_clike.py --nights nov14 --fixed-groups 15-19   # Fig. 1b's band, held
  python3 scripts/run_mwc758_ales_clike.py --nights nov14 --disk-mask MASK.fits   # the disk blocked
  python3 scripts/run_mwc758_ales_clike.py --nights nov14 --angsep-min 0.01       # the target out of its basis
  python3 scripts/run_mwc758_ales_clike.py --quick               # a 12-evaluation smoke test first

Run from the repository root.  The run directory defaults to
tutorials/runs/mwc758_ales_clike[_nov14][_zonesearched][_cmaxC][_gLO-HI][_maxexN][_mfgNN][_disk][_rinjR][_asA][_nrmN][_kmaxK][_sdi][_star].
The same command again resumes an interrupted run.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.getcwd())
from klip_tpe import CalibrationConfig, RunConfig, Runner, ValidationConfig, datasets   # noqa: E402
from klip_tpe.instruments import ales                                                     # noqa: E402
from klip_tpe.metrics import MawetPeakSNR                                                 # noqa: E402
from klip_tpe.reducer import set_klip_gram, set_zone_center                               # noqa: E402

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
ap.add_argument("--geometry", choices=("paper", "searched"), default="paper",
                help="paper: the KLIP zone is the paper's 5-27 px about (30, 30) in 6 sectors, fixed; searched: "
                     "zone_in, zone_out and n_ang are searched (the two-night run of 2026-10-18)")
ap.add_argument("--free-sectors", action="store_true", help="with --geometry paper, search the number of sectors")
ap.add_argument("--no-seed-paper", action="store_true", help="seed with tutorial 7's baseline instead")
ap.add_argument("--max-contrast", type=float, default=None,
                help="cap on the injected contrast, through the calibration and every re-calibration "
                     "(CalibrationConfig.ceiling); with c's spectrum it is the contrast at 3.8 um, the number "
                     "the log reports, and the reddest group gets 1.7 times it")
ap.add_argument("--max-excluded", type=int, default=None,
                help="how many groups the search may switch off (default: half of them, 10 of November 14's 20); "
                     "the paper's Fig. 1b keeps 5 of 20 (g15-g19), so 15 lets the search reach it")
ap.add_argument("--fixed-groups", default=None, metavar="LO-HI",
                help="hold the combination at groups LO to HI for every evaluation (15-19: the 3.8-4.15 um of the "
                     "paper's Fig. 1b): only they are loaded, no group flags, and the matched filter is the band's "
                     "middle group's")
ap.add_argument("--kernel-group", default=None, metavar="gNN",
                help="the group whose PSF is the metric's matched filter, and whose FWHM and lambda/D the metric, "
                     "the sampler and the guards use (default g10, or the middle of --fixed-groups)")
ap.add_argument("--disk-mask", default=None, metavar="FITS",
                help="a disk mask (nonzero = disk, the frames' shape; scripts/make_disk_mask.py builds one): kept "
                     "out of the metric's noise samples, and no injection lands on it")
ap.add_argument("--inject-rho", type=float, default=None, metavar="ARCSEC",
                help="every injection at this separation (default: the annulus' area-weighted mid radius, "
                     "0.632\"); MWC 758c is at 0.603\"")
ap.add_argument("--angsep-min", type=float, default=None, metavar="LD",
                help="search angsep from this many lambda/D (default 0; the paper's 0.01 keeps every target frame "
                     "out of its own basis, which angsep 0 with anglemax at the PA span does not)")
ap.add_argument("--n-remeasure", type=int, default=3, metavar="N",
                help="injection draws averaged per evaluation (default 3; N + 1 reductions each)")
ap.add_argument("--k-max", type=int, default=None, metavar="K",
                help="search k_klip up to K modes (default: the space's rule, 100 here); the arms fade from about "
                     "15 modes on")
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
fixed = None
if a.fixed_groups:
    try:
        lo, hi = (int(v) for v in a.fixed_groups.lower().replace("g", "").split("-"))
    except ValueError:
        ap.error(f"--fixed-groups wants LO-HI, 15-19 say, not {a.fixed_groups!r}")
    if not 0 <= lo <= hi <= 19:
        ap.error(f"--fixed-groups wants LO-HI with 0 <= LO <= HI <= 19 (the groups are g00-g19), not {a.fixed_groups!r}")
    fixed = [f"g{i:02d}" for i in range(lo, hi + 1)]
    if a.max_excluded is not None:
        ap.error("--max-excluded has no meaning with --fixed-groups: every evaluation combines the whole band")
if a.inject_rho is not None and not 0.4 <= a.inject_rho <= 0.8:
    ap.error(f"--inject-rho {a.inject_rho:g}\" is outside the 0.4-0.8\" annulus the run scores")
if a.angsep_min is not None and not 0.0 <= a.angsep_min < 3.0:
    ap.error(f"--angsep-min {a.angsep_min:g} is outside the 0-3 lambda/D angsep is searched over")
if a.n_remeasure < 1:
    ap.error(f"--n-remeasure wants 1 or more draws, not {a.n_remeasure}")
if a.k_max is not None and not 1 <= a.k_max <= 100:
    ap.error(f"--k-max wants 1-100 modes, not {a.k_max}")
if a.kernel_group is not None:
    a.kernel_group = a.kernel_group.lower()
    if not (len(a.kernel_group) == 3 and a.kernel_group[0] == "g" and a.kernel_group[1:].isdigit()):
        ap.error(f"--kernel-group wants gNN, g17 say, not {a.kernel_group!r}")
    if fixed and a.kernel_group not in fixed:
        ap.error(f"--kernel-group {a.kernel_group} is outside --fixed-groups {fixed[0]}-{fixed[-1]}")

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
                                           labels={"20190105": "n1", "20191114": "n2"}, only=fixed)
weighting = a.weighting or ("sqrt_texp" if a.nights == "both" else "equal")
# the matched filter's group: n2_g10 (3.50 um) as in tutorial 7, or with the groups held the band's middle one,
# which ales.make_reducer would choose itself (the group nearest the band's mean wavelength)
rep_default = ales.representative_group({p: v for p, v in lam.items() if p.startswith("n2_")}) if fixed else "n2_g10"
rep_group = "n2_" + a.kernel_group if a.kernel_group else rep_default
if rep_group not in lam:
    ap.error(f"--kernel-group {a.kernel_group} is not one of the groups loaded")
red = ales.make_reducer(dsets, lam, fwhm, table=table, max_workers=a.workers, representative=rep_group,
                        weighting=weighting, sdi=a.sdi, spectrum=spectrum, spectrum_norm=3.8e-6,
                        spectrum_label=spec_label)
set_klip_gram(red, "zone")

EDGES_AS = [0.4, 0.8]
EDGES = [e / red.pxscale for e in EDGES_AS]                    # 11.59, 23.19 spaxels
# the paper's ADI step (reduce_ales_mwc758_2.pro, MWC758_ALES @ 9ae925e) in this space's terms
PAPER = {"bin": 15, "filter": 11, "k_klip": 15, "n_ang": 6, "angsep": 0.01, "anglemax": 360.0,
         "corr_thresh": 0.0, "noise_max": 3.0, "coronoise_max": 3.0,
         "zone_in": EDGES[0] - 5.0, "zone_out": 27.0 - EDGES[1], "post_hp": 0}
# angsep from --angsep-min: above 0 no evaluation reaches the auto-fast basis (every frame, the target's own included)
as_kw = {} if a.angsep_min is None else {"angsep_range": (float(a.angsep_min), 3.0)}
if a.k_max is not None:
    as_kw["k_klip_max"] = int(a.k_max)
if a.geometry == "paper":
    # the processing geometry of reduce_ales_mwc758_2.pro: adiklip's annmode_inout=[5, 27] and n_ang=6, the
    # zones drawn by the 2015 getzone about (sc/2 - 1, sc/2 - 1) = (30, 30), which the run never moves
    pin = {"zone_in": PAPER["zone_in"], "zone_out": PAPER["zone_out"]}
    if not a.free_sectors:
        pin["n_ang"] = PAPER["n_ang"]
    set_zone_center(red, (30.0, 30.0), angles="getzone2015")
    space = ales.make_space(red, bin_range=(5, 30), search_post_hp=True, pin=pin,
                            defaults=None if a.no_seed_paper else PAPER, selection=None if fixed else "binary", **as_kw)
else:
    space = ales.make_space(red, bin_range=(5, 30), zone_range=(0.0, 8.0), search_post_hp=True,
                            defaults=None if a.no_seed_paper else PAPER, selection=None if fixed else "binary", **as_kw)
as_par = next((p for p in space.params if p.base == "angsep"), None)
k_par = next((p for p in space.params if p.base == "k_klip"), None)
n_part = len(red.partitions())
# up to max_excl groups switched off; a draw past that gets groups switched back on (those nearest 3.95 um
# first).  Half of them by default: the paper's 3.4-4.15 um (g09-g19) is within reach, its Fig. 1b's
# 3.8-4.15 um (g15-g19) needs --max-excluded 15.  With --fixed-groups there are no flags and no bound.
if fixed:
    max_excl = None
else:
    max_excl = n_part // 2 if a.max_excluded is None else int(a.max_excluded)
    if not 0 <= max_excl < n_part:
        ap.error(f"--max-excluded must be 0-{n_part - 1} with {n_part} groups")
space.project = ales.make_guard(red, max_excluded=max_excl, prefer_m=3.95e-6)
disk_mask = None
if a.disk_mask:
    dm = np.squeeze(np.asarray(fits.getdata(a.disk_mask)))
    shape = tuple(red.reducers[red.partitions()[0]].data.cube.shape[1:])
    if dm.shape != shape:
        ap.error(f"--disk-mask {a.disk_mask} is {dm.shape}, the frames are {shape}")
    disk_mask = dm != 0
objective, sampler = ales.default_config(red, known=[(RHO, PA)], forbidden_pa=[(PA, 25.0)], disk_mask=disk_mask)
# the seed's groups: 3.4-4.15 um (g09-g19) on every night, the paper's band for c.  The calibration keeps the
# seed's k (scan_k=False): trial 0 is then the paper's configuration itself, not its best k.
seed_sel = {} if (a.no_seed_paper or fixed) else {f"sel_{p}": (1 if int(p.split("_g")[1]) >= 9 else 0)
                                                  for p in red.partitions()}
print(f"{len(space.names)} dimensions: {', '.join(n for n in space.names if not n.startswith('sel_'))} "
      f"+ {len(space.selection_dims)} group flags; weights {weighting}", flush=True)
if fixed:
    print(f"every evaluation combines all {n_part} groups, {fixed[0]}-{fixed[-1]} "
          f"({min(lam.values()) * 1e6:.2f}-{max(lam.values()) * 1e6:.2f} um), held", flush=True)
else:
    print(f"the search may switch off up to {max_excl} of the {n_part} groups", flush=True)
print(f"matched filter: {rep_group}'s PSF ({lam[rep_group] * 1e6:.3f} um)", flush=True)
if disk_mask is not None:
    _yy, _xx = np.mgrid[0:disk_mask.shape[0], 0:disk_mask.shape[1]]
    _rr = np.hypot(_xx - (disk_mask.shape[1] - 1) / 2.0, _yy - (disk_mask.shape[0] - 1) / 2.0)
    r_inj = (a.inject_rho / red.pxscale if a.inject_rho is not None       # where the two sources go
             else float(np.sqrt(0.5 * (EDGES[0] ** 2 + EDGES[1] ** 2))))
    band = np.abs(_rr - r_inj) <= 0.6 * red.fwhm
    print(f"disk mask {a.disk_mask}: {int(disk_mask.sum())} px, {100 * (disk_mask & band).sum() / band.sum():.1f}% of "
          f"the injections' noise band at {r_inj * red.pxscale:.3f}\"", flush=True)
if a.max_contrast:
    print(f"injected contrast capped at {a.max_contrast:.3g}" + (" at 3.8 um, so at most "
          f"{a.max_contrast * max(red.injection_scale.values()):.3g} in the reddest group" if red.injection_scale else ""),
          flush=True)
if a.inject_rho is not None:
    print(f"every injection at {a.inject_rho:g}\" (the annulus' rule: "
          f"{np.sqrt(0.5 * (EDGES_AS[0] ** 2 + EDGES_AS[1] ** 2)):.3f}\")", flush=True)
if as_par is not None:
    print(f"angsep searched over {as_par.lo:g}-{as_par.hi:g} lambda/D" + (
        ": every target frame stays out of its own basis (no auto-fast)" if as_par.lo > 0 else
        ": at 0 with anglemax at the PA span the basis holds every frame, the target's own included (auto-fast)"),
        flush=True)
if a.n_remeasure != 3:
    print(f"{a.n_remeasure} injection draws per evaluation ({a.n_remeasure + 1} reductions)", flush=True)
if a.k_max is not None and k_par is not None:
    print(f"k_klip searched up to {k_par.hi:g} modes", flush=True)
if a.geometry == "paper":
    print(f"KLIP zone held at the paper's: {EDGES[0] - pin['zone_in']:.2f}-{min(EDGES[1] + pin['zone_out'], 27.0):.2f} px "
          f"about (30, 30)" + (", 6 sectors" if "n_ang" in pin else ", sectors searched"), flush=True)

tag = "_nov14" if a.nights == "nov14" else ""
tag += "_zonesearched" if a.geometry == "searched" else ""
tag += f"_cmax{a.max_contrast:g}" if a.max_contrast else ""
tag += f"_g{fixed[0][1:]}-{fixed[-1][1:]}" if fixed else ""
tag += f"_maxex{max_excl}" if max_excl is not None and max_excl != n_part // 2 else ""
tag += f"_mf{rep_group.split('_')[1]}" if rep_group != rep_default else ""
tag += "_disk" if disk_mask is not None else ""
tag += f"_rinj{a.inject_rho:g}" if a.inject_rho is not None else ""
tag += f"_as{a.angsep_min:g}" if a.angsep_min is not None else ""
tag += f"_nrm{a.n_remeasure}" if a.n_remeasure != 3 else ""
tag += f"_kmax{a.k_max}" if a.k_max is not None else ""
tag += "_sdi" if a.sdi else ""
tag += "_star" if a.spectrum == "star" else ""
run_dir = os.path.abspath(a.run_dir or os.path.join("tutorials", "runs", f"mwc758_ales_clike{tag}" + ("_quick" if a.quick else "")))
if a.quick:
    cfg = RunConfig(ann_edges=EDGES, n_iter=12, n_init=6, seed=a.seed, n_remeasure=1, defaults=seed_sel,
                    inject_rho=a.inject_rho,
                    validation=ValidationConfig(n_top=2, n_valid=2),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=1, scan_k=a.no_seed_paper,
                                                  ceiling=a.max_contrast),
                    fm_curve=False,
                    param_verify=False, notes="c-coloured smoke test")
else:
    cfg = RunConfig(ann_edges=EDGES, n_iter=a.n_iter, n_init=a.n_init, seed=a.seed, n_remeasure=a.n_remeasure,
                    defaults=seed_sel,
                    inject_rho=a.inject_rho,
                    validation=ValidationConfig(n_top=6, n_valid=8),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=2, scan_k=a.no_seed_paper,
                                                  ceiling=a.max_contrast),
                    fm_curve=True,
                    notes=f"tutorial 7 with c-coloured injections ({spec_label or 'star spectrum'}); the paper's ADI as seed"
                          + (f"; groups {fixed[0]}-{fixed[-1]} held" if fixed else "")
                          + f"; matched filter {rep_group}"
                          + (f"; disk mask {os.path.basename(a.disk_mask)} ({int(disk_mask.sum())} px)"
                             if disk_mask is not None else "")
                          + (f"; injections at {a.inject_rho:g}\"" if a.inject_rho is not None else "")
                          + (f"; angsep from {a.angsep_min:g} lambda/D" if a.angsep_min is not None else "")
                          + (f"; {a.n_remeasure} draws per evaluation" if a.n_remeasure != 3 else "")
                          + (f"; k up to {a.k_max}" if a.k_max is not None else ""))
# the same command again resumes the run (Runner.run picks up its checkpoint); a run started with another
# spectrum is not continued with this one
setup = os.path.join(run_dir, "run_setup.json")
if os.path.exists(setup):
    # the same groups and the same matched filter
    old_red = json.load(open(setup)).get("reducer") or {}
    parts_old = sorted(map(str, old_red.get("partitions") or {}))
    if parts_old and parts_old != sorted(map(str, red.partitions())):
        sys.exit(f"{run_dir} was started on {len(parts_old)} groups ({parts_old[0]} to {parts_old[-1]}), this run has "
                 f"{n_part} (--fixed-groups); give --run-dir for a new run")
    rep_old = old_red.get("representative")
    if rep_old and str(rep_old) != str(red.representative):
        sys.exit(f"{run_dir} was started with {rep_old}'s matched filter, this run has {red.representative}'s "
                 f"(--kernel-group); give --run-dir for a new run")
    old = (json.load(open(setup)).get("reducer") or {}).get("injection_spectrum") or {}
    new = red.describe().get("injection_spectrum") or {}
    of, nf = old.get("factors") or {}, new.get("factors") or {}
    if set(of) != set(nf) or any(abs(of[k] - nf[k]) > 1e-5 * max(abs(of[k]), 1e-9) for k in of):
        sys.exit(f"{run_dir} was started with another injected spectrum ({old.get('label', 'the star spectrum')}); "
                 f"give --run-dir for a new run")
    p0 = red.partitions()[0]
    zc_old = (((json.load(open(setup)).get("reducer") or {}).get("partitions") or {}).get(str(p0)) or {}) \
        .get("defaults", {}).get("zone_center")
    zc_new = red.reducers[p0].defaults.get("zone_center")
    if (zc_old is None) != (zc_new is None) or (zc_old is not None and list(map(float, zc_old)) != list(map(float, zc_new))):
        sys.exit(f"{run_dir} was started with its zones about {zc_old or 'the frame centre'}, this run draws them about "
                 f"{zc_new or 'the frame centre'} (--geometry); give --run-dir for a new run")
    # the checkpoint holds the run's configuration, so a resume would keep the cap it was started with
    cap_old = ((json.load(open(setup)).get("config") or {}).get("calibration") or {}).get("ceiling")
    if (cap_old is None) != (a.max_contrast is None) or (cap_old is not None and abs(cap_old - a.max_contrast) > 1e-12):
        sys.exit(f"{run_dir} was started with the injected contrast capped at {cap_old}, not {a.max_contrast} "
                 f"(--max-contrast): resuming would keep {cap_old}.  Give --run-dir for a new run")
    # nor with another bound on the groups switched off
    proj = json.load(open(setup)).get("projection") or {}
    ex_old = next((st.get("max_excluded") for st in (proj.get("steps") or [proj])
                   if st.get("name") == "group_count_guard"), None)
    how = lambda v: "its groups held" if v is None else f"up to {v} groups switched off"
    if proj and ((ex_old is None) != (max_excl is None) or (ex_old is not None and int(ex_old) != int(max_excl))):
        sys.exit(f"{run_dir} was started with {how(ex_old)}, this run has {how(max_excl)} "
                 f"(--max-excluded, --fixed-groups); give --run-dir for a new run")
    # nor with the injections at another separation (the checkpoint's would win)
    rho_old = (json.load(open(setup)).get("config") or {}).get("inject_rho")
    if (rho_old is None) != (a.inject_rho is None) or (rho_old is not None and abs(rho_old - a.inject_rho) > 1e-9):
        where = lambda v: "the annulus' rule" if v is None else f"{v:g} arcsec"
        sys.exit(f"{run_dir} was started with its injections at {where(rho_old)}, this run has them at "
                 f"{where(a.inject_rho)} (--inject-rho); give --run-dir for a new run")
    # nor with another disk mask, or none where there was one
    px_old = ((json.load(open(setup)).get("objective") or {}).get("metric") or {}).get("pixel_mask_px")
    px_new = None if disk_mask is None else int(disk_mask.sum())
    copy = os.path.join(run_dir, "disk_mask.fits")
    same = px_old == px_new and (disk_mask is None or not os.path.exists(copy)
                                 or np.array_equal(np.asarray(fits.getdata(copy)) != 0, disk_mask))
    if not same:
        sys.exit(f"{run_dir} was started with {'no disk mask' if px_old is None else f'a {px_old}-px disk mask'}, this "
                 f"run has {'none' if px_new is None else f'{a.disk_mask} ({px_new} px)'} (--disk-mask); give --run-dir "
                 f"for a new run")
    # nor with angsep searched over another range (the checkpoint holds proposals from the old one)
    as_old = next((q for q in ((json.load(open(setup)).get("space") or {}).get("params") or [])
                   if q.get("base", q.get("name")) == "angsep"), None)
    if as_old is not None and as_par is not None and abs(float(as_old["lo"]) - as_par.lo) > 1e-12:
        sys.exit(f"{run_dir} was started with angsep searched from {float(as_old['lo']):g} lambda/D, this run searches "
                 f"it from {as_par.lo:g} (--angsep-min); give --run-dir for a new run")
    # nor with k searched up to another count
    k_old = next((q for q in ((json.load(open(setup)).get("space") or {}).get("params") or [])
                  if q.get("base", q.get("name")) == "k_klip"), None)
    if k_old is not None and k_par is not None and abs(float(k_old["hi"]) - k_par.hi) > 1e-9:
        sys.exit(f"{run_dir} was started with k_klip searched up to {float(k_old['hi']):g}, this run up to {k_par.hi:g} "
                 f"(--k-max); give --run-dir for a new run")
    # nor with another number of draws per evaluation (the checkpoint's would win)
    nrm_old = (json.load(open(setup)).get("config") or {}).get("n_remeasure")
    if not a.quick and nrm_old is not None and int(nrm_old) != a.n_remeasure:
        sys.exit(f"{run_dir} was started with {nrm_old} injection draws per evaluation, this run has {a.n_remeasure} "
                 f"(--n-remeasure): resuming would keep {nrm_old}.  Give --run-dir for a new run")
if disk_mask is not None and not os.path.exists(os.path.join(run_dir, "disk_mask.fits")):
    # the mask the run was scored with, beside its results
    os.makedirs(run_dir, exist_ok=True)
    hdr = fits.getheader(a.disk_mask)
    hdr["DMSOURCE"] = (os.path.basename(a.disk_mask)[:60], "the --disk-mask file this run was given")
    fits.writeto(os.path.join(run_dir, "disk_mask.fits"), disk_mask.astype(np.uint8), hdr, overwrite=True)
callbacks = []
if a.show != "none":
    from klip_tpe.display import LiveDisplay
    callbacks.append(LiveDisplay(run_dir, show="window", every=5, pdf_every=100, movie_every=500))
t0 = time.time()
results = Runner(red, space, objective, sampler, cfg, run_dir, callbacks=callbacks).run()
print(f"{(time.time() - t0) / 3600:.2f} h", flush=True)

# ----------------------------------------------------------------------------------- what it decided
r = results[0]
metric = MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel, declip_nsig=None,
                      pixel_mask=disk_mask)
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
print(f"MWC 758c in the winner's image: S/N {c_snr:.2f} with {rep_group}'s matched filter"
      + (", the disk masked" if disk_mask is not None else "")
      + (" (tutorial 7's winner 1.9; the paper's Nov 14 images 3.6-3.7)" if rep_group == "n2_g10" and disk_mask is None else ""))
json.dump({"c_snr_winner": c_snr, "winner_score": r.winner_score, "contrast": r.contrast, "spectrum": spec_label,
           "winner_block": {k: blk.get(k) for k in keep}, "partitions": list(r.partitions), "paper_seed": PAPER,
           "max_excluded": max_excl, "fixed_groups": fixed, "matched_filter": rep_group,
           "disk_mask": a.disk_mask, "disk_mask_px": None if disk_mask is None else int(disk_mask.sum()),
           "inject_rho": a.inject_rho, "angsep_range": None if as_par is None else [as_par.lo, as_par.hi],
           "n_remeasure": 1 if a.quick else a.n_remeasure, "k_max": None if k_par is None else k_par.hi},
          open(os.path.join(run_dir, "clike_summary.json"), "w"), indent=1, default=float)
