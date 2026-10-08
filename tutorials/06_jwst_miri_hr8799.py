# %% [markdown]
# # Tutorial 6: MIRI with One Roll and Four Planets (HR 8799)
#
# Tutorial 5 covered what makes MIRI's four-quadrant masks different. This one is about a
# different kind of observation, on a system where the mid-infrared payoff is clear. JWST GTO
# program 1194 observed HR 8799 through F1065C, F1140C and F1550C (Boccaletti et al. 2024, A&A
# 686, A33). It detected all four planets in the two shorter filters and resolved the inner
# warm dust belt, about 15 au (0.37″) from the star.
#
# Three features of the observation change how the run is set up, and none of them is about
# the coronagraph:
#
# 1. **One science pointing, not two rolls.** There is no field rotation at all, so ADI is
#    impossible. The reduction is RDI, and every parameter that matters is about how the
#    reference library is used. Most JWST coronagraphic sequences have two rolls, and a
#    single-roll sequence is easy to set up wrongly.
# 2. **A nine-point small-grid dither on the reference star** (HD 218261), in 10 mas steps.
#    Its eighteen integrations sample the PSF's response to pointing jitter, which is the
#    point of a small-grid dither and what makes RDI work at this contrast.
# 3. **Four companions from about 0.4″ to 1.7″, at four different position angles.** On a
#    4QPM the throughput depends on position angle (tutorial 5), so the four planets sit at
#    four quite different attenuations.
#
# **Data.**
# ```
# python3 scripts/fetch_jwst_ar.py --targets hr8799                                # list first
# python3 scripts/fetch_jwst_ar.py --targets hr8799 --download --outdir ~/Data/JWST
# ```
# The files land in `~/Data/JWST/hr8799/`, together with the program's NIRCam coronagraphy.
# Set `KLIP_TPE_JWST_HR8799` if you keep them elsewhere.
#
# **PSF model.** As in tutorial 5, the cells need STPSF or the three cache files for the
# filter (stamp grid, throughput map and encircled energy) in `$KLIP_TPE_DATA/stpsf_cache`. A
# missing file stops the notebook with its name.

# %%
import glob, os, time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from klip_tpe import Runner, RunConfig, ValidationConfig, CalibrationConfig, datasets
from klip_tpe import stpsf_psf
from klip_tpe.backends import spaceklip as sk
from klip_tpe.instruments import generic, miri
from klip_tpe.display import LiveDisplay

DATA = os.path.expanduser(os.environ.get("KLIP_TPE_JWST_HR8799", "~/Data/JWST/hr8799"))
if os.path.isdir(os.path.join(DATA, "mastDownload")):   # the archive files, not a reprocessing beside them:
    DATA = os.path.join(DATA, "mastDownload")           # load_calints refuses one exposure found twice
FILTER = "F1065C"                  # all four planets are detected here and in F1140C
RUN_DIR = os.path.abspath(f"runs/hr8799_{FILTER.lower()}")
# Re-running RESUMES this directory -- delete it to search again (see tutorial 01).

files = sorted(glob.glob(os.path.join(DATA, "**", "jw*_calints.fits"), recursive=True))
in_filt = [f for f in files if str(fits.getheader(f).get("FILTER", "")).upper() == FILTER]
HAVE_DATA = len(in_filt) >= 4
print(f"{len(files)} calints under {DATA}, {len(in_filt)} in {FILTER}"
      f"  ({'ok' if HAVE_DATA else 'run the fetch script'})")

# %% [markdown]
# ## 1. With One Roll, the Partition Layout Decides What Is Possible
#
# `partition="roll"` makes one partition per roll angle. With a single pointing that is one
# partition holding every science integration at one PA, which is right here. It also means
# that `mode="ADI"` would give pyKLIP nothing to build a basis from. Ask for it and you get
# NaN, or a number from a basis of one frame.
#
# The same layout on a two-roll sequence is a trap. With `partition="roll"` on two rolls, every
# frame in each partition shares a PA, so ADI has no reference frames and ADI+RDI duplicates
# RDI. The rule is that **a partition is reduced on its own**. For ADI to mean anything, the
# frames that provide the rotation have to be in the same partition (`partition="all"`, as in
# tutorials 3 and 5). Here there is no rotation, so the right setup is one partition and RDI.
#
# Every science and reference exposure in this program already carries
# `S_BKDSUB='COMPLETE'`, unlike the ERS program in tutorial 5. The header census below confirms
# it, and the loader then has no background to subtract.

# %%
if HAVE_DATA:
    for f in sorted(in_filt):
        h = fits.getheader(f)
        print(f"  {str(h.get('TARGPROP','')):24s} NINTS={h.get('NINTS'):<3} "
              f"S_BKDSUB={h.get('S_BKDSUB', 'absent')!s:9s} "
              f"PATTTYPE={str(h.get('PATTTYPE','')):20s}")

    dsets, info = sk.load_calints(in_filt, science_target="HR8799", half_px=40,
                                  partition="roll", filter=FILTER)
    px, m = float(info["pxscale"]), miri.mode_for_filter(info["filter"])
    d0 = list(dsets.values())[0]
    print(f"\n{len(dsets)} partition(s); science {np.asarray(d0.cube).shape}, "
          f"library {np.asarray(d0.ref_cube).shape}, roll(s) {info['rolls']}")

# %% [markdown]
# ## 2. The Companions, and Why This Tutorial Leaves Their Positions to You
#
# `known=[(ρ, PA), ...]` does two things. It keeps injected sources away from real ones, and it
# keeps real ones out of the noise ring that the injections are scored against. Without it, a
# real planet inflates σ in its own annulus, every contrast there is pessimistic, and the
# optimizer is rewarded for parameters that suppress the planet.
#
# So the positions have to be right, and they move. HR 8799's planets have orbital periods of
# about 45 to 460 years, and e advances several degrees of position angle per year. A table
# copied from a tutorial is wrong for your epoch, and it puts the exclusion zone next to the
# planet rather than on it. Boccaletti et al. (2024) observed on 2022 November 7 and 8 and
# report photometry, not astrometry. Take the positions for your epoch from an orbit fit or an
# astrometric compilation (`whereistheplanet`, Wang et al., computes them from the published
# astrometry), and enter them here:

# %%
# EDIT THESE for your epoch.  Left empty on purpose: the cells below work either way and say
# what changes.  Example shape: KNOWN = [(1.72, 65.0), (0.95, 325.0), (0.66, 218.0), (0.39, 265.0)]
KNOWN = []
if HAVE_DATA and not KNOWN:
    print("KNOWN is empty: the run below will treat the four planets as unknown signal.\n"
          "That is a legitimate blind-search configuration, but it is NOT the right way to\n"
          "measure a contrast curve for this system -- see section 5.")

# %% [markdown]
# ## 3. Throughput at Each Planet's Separation
#
# The separations change far more slowly than the position angles (b's period is about 460
# years), so they can be quoted approximately: about 1.7″, 0.95″, 0.66″ and 0.39″ for b, c, d
# and e. The figure shows, at each of those separations, how much the throughput varies with
# azimuth. At 1.7″ the attenuation a companion suffers depends on where it sits by about a
# factor of 6.4.

# %%
if HAVE_DATA:
    try:
        # Read from the cache when the map is there; STPSF is needed only to compute it.
        g = miri.throughput_map(FILTER)
    except (RuntimeError, ImportError) as exc:
        raise RuntimeError(
            f"This tutorial needs the STPSF model of the {FILTER} 4QPM: STPSF itself (Python >= 3.10, "
            f"with its data files), or the cached throughput map, stamp grid and encircled-energy "
            f"files for {FILTER} in $KLIP_TPE_DATA/stpsf_cache.\n{exc}") from exc
    T = miri.throughput_map_fn(g)
    az = np.linspace(0, 360, 721)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for rho, nm in ((0.39, "e"), (0.66, "d"), (0.95, "c"), (1.72, "b")):
        t = T(np.full(az.shape, rho), az)
        ax.plot(az, t, lw=1.4, label=f"{nm}  {rho:.2f}\"   x{t.max() / t.min():.1f}")
    ax.axhline(0.30, color="crimson", ls=":", lw=1)
    ax.text(4, 0.315, "min_throughput: below this is a dead zone", color="crimson", fontsize=7)
    ax.set_xlim(0, 360); ax.set_xlabel("detector azimuth (deg)"); ax.set_ylabel("throughput")
    ax.set_title(f"{FILTER}: how much the attenuation varies at each planet's separation",
                 fontsize=9)
    ax.legend(fontsize=7, title="planet, separation, max/min", title_fontsize=7)
    plt.tight_layout()

# %% [markdown]
# The cell below asks the map for the throughput at each planet's position in each roll of
# this observation. With one roll there is one value per planet. With two rolls there would be
# two, and the spread between them is the per-frame variation that `azimuth_dependent=True`
# handles.

# %%
if HAVE_DATA and KNOWN:
    angles = np.unique(np.round(np.asarray(d0.angles, float), 2))
    truenorth = 0.0        # 0 for these frames; in general the per-partition reducer's truenorth
    print(f"{'planet':>8s} {'rho':>6s} {'PA':>6s}   throughput per roll")
    for i, (rho, pa) in enumerate(KNOWN):
        az = np.asarray([pa - truenorth - 270.0 - a for a in angles])
        t = T(np.full(az.shape, rho), az)
        print(f"{'bcde'[i] if i < 4 else '?':>8s} {rho:6.2f} {pa:6.1f}   "
              + "  ".join(f"{v:.3f}" for v in np.atleast_1d(t))
              + ("   <- in a dead zone" if np.min(t) < 0.30 else ""))

# %% [markdown]
# ## 4. The Flux Unit, Checked Against an Independent Number
#
# The four terms are the same as in tutorial 5. HR 8799 has a check that tutorial 5's star
# does not.
#
# Boccaletti et al. (2024) interpolate the star's flux density at 15.5 µm between WISE W3 and
# AKARI L18W and get 154.2 mJy. Planck(7600 K, the effective temperature they adopt) through
# the F1550C bandpass, normalized to 2MASS Ks = 5.240, gives 155.8 mJy, a 1.0% agreement by a
# route that shares nothing with theirs. Over 7200 to 7800 K the synthetic value runs from
# 159.6 to 154.0 mJy, so the agreement holds to within 3.5% across that range. An error in Ks
# would scale straight through.
#
# `datasets.PHOTOMETRY` carries the entries for both stars, with their checks.

# %%
STAR_FLUX = None
if HAVE_DATA:
    phot = datasets.PHOTOMETRY[f"hr8799_{FILTER.lower()}"]
    STAR_FLUX = miri.star_flux_from_flux_density(FILTER, phot["flux_density_jy"],
                                                 info["pixar_sr"], bunit=info["bunit"])
    print(f"S = {phot['flux_density_jy'] * 1e3:.1f} mJy -> star_flux = {STAR_FLUX:.4e}")
    print(f"check: {phot['check']}")

# %% [markdown]
# ## 5. The Run
#
# Two settings follow from the single roll.
#
# * **`mode="RDI"`, fixed.** ADI has no frames to work with, so searching `mode` would only
#   spend evaluations.
# * **`search_angles=False`.** With every frame at one PA, `angsep` and `anglemax` change
#   nothing, so they are left out. The search covers the number of KL modes, the high-pass
#   filter and the azimuthal subdivision. With eighteen reference frames and no rotation, the
#   number of KL modes does most of the work.
#
# Two rolls in one partition (tutorial 5) would give ADI+RDI the other roll as well. The
# observation decides, not the instrument.

# %%
if HAVE_DATA and STAR_FLUX is not None:
    ann = (6.7, 36.0)
    mid_as = 0.5 * (ann[0] + ann[1]) * px
    angles = np.concatenate([np.asarray(d.angles, float).ravel() for d in dsets.values()])
    fpa = miri.forbidden_pa(angles, rho_as=mid_as, filter=FILTER)
    pmask = miri.dead_zone_pixel_mask((81, 81), px, angles, filter=FILTER)

    red = sk.make_reducer(dsets, pxscale=px, wavelength_m=m["lam_m"], diam_m=miri.DIAMETER_M,
                          psf="stpsf", star_flux=STAR_FLUX, mode="RDI", max_workers=3)
    obj, samp = generic.default_config(red, known=KNOWN, forbidden_pa=fpa, pixel_mask=pmask)
    space = generic.make_space(red, k_klip_max=16, search_angles=False)
    space.project = generic.make_guard(red, k_max=16)
    print(f"{space.names}, {len(fpa)} forbidden sector(s), "
          f"{int(pmask.sum())} px out of the noise estimate")

    dec = space.decode(space.default_vector())
    ev = red.reduce_config(dec, None, tag="check")
    img = np.asarray(ev.image if hasattr(ev, "image") else ev, float)
    yy, xx = np.mgrid[0:81, 0:81]
    rr = np.hypot(xx - 40, yy - 40)
    inann = (rr >= ann[0]) & (rr <= ann[1])
    print(f"default reduction: {100 * np.isfinite(img[inann]).mean():.1f}% of the annulus finite, "
          f"model {red.reducers[list(dec.selected)[0]].model.name}")

# %% [markdown]
# The search uses tutorial 1's budget: 300 evaluations with 40 of warm-up, three injection
# draws averaged per trial, and the six best candidates validated on eight fresh injection
# sets each. It has not been timed on these data, but with 9 science and 18 reference frames it
# runs faster than tutorial 5.
#
# The four injected sources step across the band in radius, and the inner two sit about a FWHM
# on either side of b's separation. With `KNOWN` empty nothing keeps them off the planets. An
# injection can land on b and is then scored against a ring that the planet also inflates. For
# a run whose panels or numbers you will use, fill in `KNOWN` (section 2), and consider keeping
# the injections a few FWHM from the planets as tutorial 5 does (`samp.excl_fwhm`).

# %%
if HAVE_DATA and STAR_FLUX is not None:
    cfg = RunConfig(ann_edges=[ann[0], ann[1]], n_iter=300, n_init=40, seed=21, n_remeasure=3,
                    validation=ValidationConfig(n_top=6, n_valid=8),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=2),
                    verify=True, save_eval_images=False)
    disp = LiveDisplay(RUN_DIR, every=5, pdf_every=0, movie=False, show="auto")
    runner = Runner(red, space, obj, samp, cfg, RUN_DIR, callbacks=[disp], resume="auto")
    t0 = time.time()
    results = runner.run()
    print(f"\n{(time.time() - t0) / 60:.1f} min -> {RUN_DIR}")

# %% [markdown]
# ## 6. The Same Target through Three Filters
#
# Change `FILTER` at the top and run again. Boccaletti et al. (2024) detect all four planets in
# F1065C and F1140C, and in F1550C only b, with c marginal. F1550C is therefore where the search
# works near its floor, where the optimizer's choices matter most and a wrong noise estimate
# does the most damage.
#
# * `datasets.PHOTOMETRY` has all three flux densities (324.4, 286.6 and 155.8 mJy for F1065C,
#   F1140C and F1550C), so `--flux-density-jy` is never needed by hand.
# * Each filter needs its own three cache files. A throughput map takes about 25 minutes on a
#   machine with STPSF, and it is checkpointed.
# * The star flux differs by a factor of two across the three filters, and so does the
#   background. Do not carry a `--star-flux` from one filter to another. Let the lookup do it.
#
# ## 7. Where This System Is Hard
#
# * **The inner dust belt** lies at about 0.15″ to 0.37″, inside this annulus (0.74″ to 3.97″).
#   An annulus that reaches in that far contains extended emission at every position angle,
#   because a ring around the star covers them all, so `forbidden_pa` cannot keep the
#   injections off it. `pixel_mask` can take it out of the noise estimate (tutorial 4).
# * **Which planets the run sees.** In this annulus only b lies where the injections go. c, at
#   0.95″, sits inside the inner edge of the injection band, and d and e lie inside the annulus.
#   A wider annulus brings more of them in. Each planet in `KNOWN` then removes a zone of
#   1.5 FWHM radius on top of the dead zones, and a narrow annulus can run short of legal
#   injection positions.
# * **e is at about 0.4″**, about 1.2 λ/D at 10.65 µm. That is inside the region where a radial
#   stamp library can be trusted on a 4QPM, and in the regime `min_throughput` masks rather than
#   extrapolates into. Treat any contrast quoted inside about 2 λ/D as a statement about the
#   mask rather than the reduction.
