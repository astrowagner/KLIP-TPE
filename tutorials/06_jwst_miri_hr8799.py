# %% [markdown]
# # Tutorial 6 — MIRI with one roll and four planets: HR 8799
#
# Tutorial 5 covered what makes MIRI's four-quadrant masks different.  This one is about a
# *different shape of observation*, on the system where the mid-infrared payoff is clearest.
# GO 1194 (Boccaletti et al. 2024, A&A 686, A33) observed HR 8799 through F1065C, F1140C and
# F1550C and detected all four planets in the two shorter filters — the first mid-infrared
# images of the system — with the inner dust belt as well.
#
# Three things about the observation change how you set the run up, and none of them are
# about the coronagraph:
#
# 1. **One science pointing, not two rolls.**  So there is no field rotation at all, and
#    **ADI is impossible**: the reduction is RDI, and every parameter that matters is about
#    how the reference library is used.  This is the commonest JWST coronagraphic layout and
#    the one most likely to be set up wrongly.
# 2. **A nine-point small-grid dither on the reference star**, 10 mas steps — eighteen
#    reference integrations that sample the PSF's response to pointing jitter, which is the
#    point of an SGD and what makes RDI work at this contrast.
# 3. **Four companions from ~0.4″ to ~1.7″, at four different position angles.**  One
#    companion tests a throughput model at one place.  Four test whether a
#    position-dependent throughput is right *everywhere at once*, which is the claim a 2-D
#    map actually makes.
#
# **Data.**
# ```
# python3 scripts/fetch_jwst_ar.py --list hr8799
# python3 scripts/fetch_jwst_ar.py --fetch hr8799 --out ~/Data/JWST/hr8799
# ```
# ~1.2 GB for the MIRI half (the programme also has a NIRCam half in eight filters).  Set
# `KLIP_TPE_JWST_HR8799` if you keep it elsewhere.

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
FILTER = "F1065C"                  # all four planets are detected here and in F1140C
RUN_DIR = os.path.abspath(f"runs/hr8799_{FILTER.lower()}")

files = sorted(glob.glob(os.path.join(DATA, "**", "jw*_calints.fits"), recursive=True))
in_filt = [f for f in files if str(fits.getheader(f).get("FILTER", "")).upper() == FILTER]
HAVE_DATA = len(in_filt) >= 4
print(f"{len(files)} calints under {DATA}, {len(in_filt)} in {FILTER}"
      f"  ({'ok' if HAVE_DATA else 'run the fetch script'})")

# %% [markdown]
# ## 1. One roll means the partition layout decides what is possible
#
# `partition="roll"` makes one partition per unique roll angle.  With a single pointing that
# is **one** partition containing every science integration at one PA — which is correct
# here, and it means `mode="ADI"` would give pyKLIP nothing to build a basis from.  Ask for
# it and you get NaN, or worse, a number from a basis of one frame.
#
# This is not a MIRI subtlety, it is the trap that invalidated two of this package's own
# paper runs: with `partition="roll"` on a *two*-roll NIRCam sequence, every frame in each
# partition shared a PA, so ADI had no reference frames and "ADI+RDI" was really RDI — and a
# run that searched over `mode` was choosing between three options of which two could not
# work.  The rule: **a partition is reduced on its own**, so if you want ADI to mean anything,
# the frames that provide the rotation have to be in the same partition (`partition="all"`).
# Here there is no rotation to have, so the honest setup is one partition and RDI.
#
# Note also what the loader does *not* say below: there is no background-subtraction line.
# Every science and reference exposure in this programme already carries
# `S_BKDSUB='COMPLETE'`, unlike the ERS programme in tutorial 5 where only the science half
# did.  Nothing to fix — but worth confirming rather than assuming, which is why the loader
# reports the census either way.

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
# ## 2. The companions, and why this tutorial will not hand you their positions
#
# `known=[(ρ, PA), ...]` does two things: it keeps injected sources away from real ones, and
# it keeps real ones out of the noise ring the injections are scored against.  Forget it and
# a real planet inflates σ in its own annulus, so every contrast in that annulus is
# pessimistic and the optimizer is rewarded for parameters that suppress it.
#
# So the positions have to be right — and **they move**.  HR 8799's planets are on 40-to-500
# year orbits; e advances several degrees of position angle per year.  A table copied out of a
# tutorial is wrong for your epoch, and wrong in the direction that puts the exclusion zone
# beside the planet rather than on it.  Boccaletti et al. (2024) observed on **2022 November
# 7–8** and report photometry rather than astrometry, so take positions from an orbit fit or
# an astrometric compilation at your own epoch — `whereistheplanet` (Wang et al.) does this
# from the published astrometry — and put them in here:

# %%
# EDIT THESE for your epoch.  Left empty on purpose: the cells below work either way and say
# what changes.  Example shape: KNOWN = [(1.72, 65.0), (0.95, 325.0), (0.66, 218.0), (0.39, 265.0)]
KNOWN = []
if HAVE_DATA and not KNOWN:
    print("KNOWN is empty: the run below will treat the four planets as unknown signal.\n"
          "That is a legitimate blind-search configuration, but it is NOT the right way to\n"
          "measure a contrast curve for this system -- see section 5.")

# %% [markdown]
# ## 3. What four companions test that one cannot
#
# The planets sit at four different position angles.  On a 4QPM the throughput at a fixed
# separation varies by up to a factor of six and a half with azimuth (tutorial 5), so the
# four of them sample four quite different attenuations — and the two things the map has to
# get right, the boundary *positions* and the depth *between* them, are tested at four places
# instead of one.
#
# %% [markdown]
# Their *separations* are another matter: those change far more slowly than the position
# angles (b's period is ~460 yr), so quoting them approximately is safe where quoting a PA is
# not.  Roughly 1.7″, 0.95″, 0.66″ and 0.39″ for b, c, d and e.  The figure below is the whole
# argument of this section: at each of those separations, how much the throughput varies with
# azimuth.  The planets do not choose their position angles to suit us, and at 1.7″ the
# attenuation a companion suffers depends on where it sits by nearly a factor of six.

# %%
if HAVE_DATA and stpsf_psf.have_stpsf():
    g = miri.throughput_map(FILTER)
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
# The cell below asks the map what it predicts at each planet's position for each roll of
# this observation.  With one roll there is one answer per planet; with two rolls you would
# get two, and the spread between them is exactly the per-frame variation that
# `azimuth_dependent=True` exists to handle.

# %%
if HAVE_DATA and KNOWN and stpsf_psf.have_stpsf():
    angles = np.unique(np.round(np.asarray(d0.angles, float), 2))
    truenorth = 0.0        # in a real run this is the reducer's: getattr(red, "truenorth", 0.0),
                           # which is what forbidden_pa is given in section 5
    print(f"{'planet':>8s} {'rho':>6s} {'PA':>6s}   throughput per roll")
    for i, (rho, pa) in enumerate(KNOWN):
        az = np.asarray([pa - truenorth - 270.0 - a for a in angles])
        t = T(np.full(az.shape, rho), az)
        print(f"{'bcde'[i] if i < 4 else '?':>8s} {rho:6.2f} {pa:6.1f}   "
              + "  ".join(f"{v:.3f}" for v in np.atleast_1d(t))
              + ("   <- in a dead zone" if np.min(t) < 0.30 else ""))

# %% [markdown]
# ## 4. The flux unit, checked against somebody else's number
#
# Same four terms as tutorial 5.  What is worth showing here is the *check*, because HR 8799
# has one that tutorial 5's star does not.
#
# Boccaletti et al. (2024) interpolate the stellar flux density at 15.5 µm from WISE and
# AKARI photometry and get **154.2 mJy**.  Planck(7600 K) through the F1550C bandpass,
# normalised to 2MASS Ks = 5.240, gives **155.8 mJy** — agreement to 1.0%, by a route with
# nothing in common with theirs.  Across the plausible Teff range (7200–7800 K) the synthetic
# value moves 159.6 → 154.0 mJy, so the agreement also pins the inputs: an error in Ks would
# scale straight through.
#
# That is the standard to hold a contrast axis to.  Getting a number is easy; the question is
# always what independent thing it agrees with.  `datasets.PHOTOMETRY` carries both stars'
# entries with their checks written into them.

# %%
STAR_FLUX = None
if HAVE_DATA and stpsf_psf.have_stpsf():
    phot = datasets.PHOTOMETRY[f"hr8799_{FILTER.lower()}"]
    STAR_FLUX = miri.star_flux_from_flux_density(FILTER, phot["flux_density_jy"],
                                                 info["pixar_sr"], bunit=info["bunit"])
    print(f"S = {phot['flux_density_jy'] * 1e3:.1f} mJy -> star_flux = {STAR_FLUX:.4e}")
    print(f"check: {phot['check']}")

# %% [markdown]
# ## 5. The run
#
# Two differences from tutorial 5's setup, both following from the single roll.
#
# **`mode="RDI"`, fixed, not searched.**  Searching a categorical whose other values cannot
# work wastes evaluations and produces a "choice" that is not one.
#
# **The searchable reference-library dimensions matter more.**  With eighteen SGD frames and
# no rotation, how many KL modes to keep and how the library is weighted *is* the reduction.
# `make_space` scales `k_klip` to the data (up to `nframes/5`), and `make_guard` keeps the
# reference census honest — a configuration that would leave KLIP fewer than `n_min_ref`
# usable references is projected back rather than evaluated and scored as a failure.
#
# Contrast with tutorial 5: there, two rolls 9.4° apart make ADI+RDI a real trade-off and
# `mode` is worth searching.  The observation decides, not the instrument.

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
    space = generic.make_space(red, k_klip_max=16)
    space.project = generic.make_guard(red, k_max=16)
    print(f"{space.ndim} dimensions, {len(fpa)} forbidden sector(s), "
          f"{int(pmask.sum())} px out of the noise estimate")

    dec = space.decode(space.default_vector())
    ev = red.reduce_config(dec, None, tag="check")
    img = np.asarray(ev.image if hasattr(ev, "image") else ev, float)
    yy, xx = np.mgrid[0:81, 0:81]
    rr = np.hypot(xx - 40, yy - 40)
    inann = (rr >= ann[0]) & (rr <= ann[1])
    print(f"default reduction: {100 * np.isfinite(img[inann]).mean():.1f}% of the annulus finite, "
          f"model {red.reducers[list(dec.selected)[0]].model.name}")

# %%
if HAVE_DATA and STAR_FLUX is not None:
    cfg = RunConfig(ann_edges=[ann[0], ann[1]], n_iter=60, n_init=15, seed=21,
                    validation=ValidationConfig(n_top=4, n_valid=8),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=3),
                    verify=True, save_eval_images=False)
    disp = LiveDisplay(RUN_DIR, every=5, pdf_every=0, movie=False, show="auto")
    runner = Runner(red, space, obj, samp, cfg, RUN_DIR, callbacks=[disp], resume="auto")
    t0 = time.time()
    results = runner.run()
    print(f"\n{(time.time() - t0) / 60:.1f} min -> {RUN_DIR}")

# %% [markdown]
# ## 6. The same target through three filters
#
# Change `FILTER` at the top and re-run.  What to expect, from Boccaletti et al. (2024): all
# four planets in **F1065C** and **F1140C**; in **F1550C** only b clearly, with c marginal.
# So the three filters are not three repetitions of one exercise — F1550C is the regime where
# the search is working near its floor, which is where the optimizer's choices matter most
# and where a wrong noise estimate does the most damage.
#
# Practical notes for running all three:
#
# * `datasets.PHOTOMETRY` has all three flux densities (324.4 / 286.6 / 155.8 mJy for
#   F1065C / F1140C / F1550C), so `--flux-density-jy` is never needed by hand.
# * Each filter needs its own cached throughput map and encircled-energy file.  F1065C and
#   F1140C are cached; **F1550C's map has to be computed once** (~10 minutes, check-pointed)
#   on a machine with STPSF, and the cache file copied over. The error names the file.
# * The star flux differs by a factor of two across the three, and so does the background.
#   Do not carry a `--star-flux` from one filter to another; let the lookup do it.
#
# ## 7. Where this system is genuinely hard
#
# * **The inner dust belt.**  Boccaletti et al. detect it, which means there is extended
#   emission in the search annulus — the case tutorial 4 is about.  If you are tuning for the
#   planets, `forbidden_pa` and the matching `pixel_mask` keep the belt out of both the
#   injections and the noise estimate; doing only one of the two is worse than neither.
# * **Four planets is four exclusion zones.**  At 1.5 FWHM each, plus 12% of the ring already
#   forbidden by the mask's dead zones, a narrow annulus can run out of legal injection
#   positions. The sampler will tell you; widen the annulus or inject fewer sources.
# * **e is at ~0.4″**, which at 10.65 µm is about 1.2 λ/D — inside where a radial stamp
#   library is trustworthy on a 4QPM, and in the regime `min_throughput` is designed to mask
#   rather than extrapolate into. Treat any contrast quoted inside ~2 λ/D as the mask's
#   statement rather than the reduction's.
