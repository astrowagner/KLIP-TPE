# %% [markdown]
# # Tutorial 5 — JWST/MIRI coronagraphy: the four-quadrant phase mask
#
# Tutorial 3 optimized the NIRCam half of ERS-1386 on HIP 65426.  This is the **MIRI** half
# of the same programme, on the same star: F1140C behind FQPM1140, two rolls 9.4° apart, and
# sixteen reference exposures of two other stars.  Running one target through two instruments
# is the cheapest way to see which parts of a reduction are the pipeline's and which are the
# coronagraph's.
#
# Because MIRI is not NIRCam with different numbers in it.  Three of its four coronagraphs
# are **four-quadrant phase masks**: instead of blocking a disc, they put a π phase step
# across two perpendicular lines through the star.  Everything downstream that assumes a
# round occulter is wrong here, and each of the three consequences below is a way to get a
# confident wrong answer rather than a crash.  All three were live bugs in this repository
# in September 2026; the sections say what they looked like.
#
# **Data.**  `scripts/fetch_jwst_ar.py` pulls the programme from MAST (~1.5 GB for all three
# MIRI filters, no login):
# ```
# python3 scripts/fetch_jwst_ar.py --list hip65426_miri        # look before downloading
# python3 scripts/fetch_jwst_ar.py --fetch hip65426_miri --out ~/Data/JWST/hip65426_miri
# ```
# Point `KLIP_TPE_JWST_MIRI` elsewhere if you keep it somewhere else.  Every cell below is
# guarded, so the notebook reads as a document without the files.
#
# **STPSF.**  The throughput map and the off-axis PSFs come from STPSF (Python ≥ 3.10 plus
# ~90 MB of data files), but they are *cached*, so a machine that cannot install STPSF runs
# from a copied cache — see section 7.

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

DATA = os.path.expanduser(os.environ.get("KLIP_TPE_JWST_MIRI", "~/Data/JWST/hip65426_miri"))
RUN_DIR = os.path.abspath("runs/hip65426_f1140c")
FILTER = "F1140C"
PLANET = (0.826, 150.2)            # HIP 65426 b, Carter et al. 2023 (F444W astrometry)

files = sorted(glob.glob(os.path.join(DATA, "**", "jw*_calints.fits"), recursive=True))
in_filt = [f for f in files if str(fits.getheader(f).get("FILTER", "")).upper() == FILTER]
HAVE_DATA = len(in_filt) >= 4
print(f"{len(files)} calints under {DATA}, {len(in_filt)} in {FILTER}"
      f"  ({'ok' if HAVE_DATA else 'run the fetch script'})")

# %% [markdown]
# ## 1. What an archive download of one programme actually contains
#
# Not just a target and a reference star.  A MIRI coronagraphic programme also takes
# **dedicated background pointings** — offset exposures of blank sky — because at 11 µm the
# thermal background is large and structured.  Sorting the three roles is the first thing
# `load_calints` does, and getting it wrong is silent in both directions: blank sky in the
# RDI library is empty frames used to model a star's diffraction, and a reference star named
# after its target (AU Mic's is `AU_Mic_psf_reference`, which starts with `AUMIC`) gets
# derotated and stacked with the science.
#
# The subtler half is that **the background may already have been subtracted — for some of
# the files.**  Image2 runs its background step when the association has background members,
# which a programme's *science* targets have and its pure *PSF reference* stars usually do
# not.  The header records it as `S_BKDSUB`.  For this programme at F1140C:
#
# | target | role | files | `S_BKDSUB` |
# |---|---|---|---|
# | HIP-65426 | science | 2 | `COMPLETE` |
# | HD-141569A | reference | 2 | `COMPLETE` |
# | HIP-68245 | reference | 9 | absent |
# | HD-140986 | reference | 5 | absent |
# | `*-BACKGROUND` | background | 8 | absent — they *are* the background |
#
# So an untreated load puts fourteen reference frames carrying a ~19 MJy/sr sky pedestal and
# the mask's glow sticks into one KLIP library beside two that carry neither and science
# frames that carry neither.  The library's dominant common mode is then the sky rather than
# the stellar PSF — and because the reducer's high-pass hides a smooth pedestal, the
# optimizer is handed a reason to prefer a hard high-pass and to report *that* as the best
# reduction parameter.  A search result that is really a workaround for a calibration
# mismatch is the worst thing this package can produce.
#
# `load_calints` reads `S_BKDSUB` per exposure, subtracts the median of the programme's own
# blank-sky pointings from whichever frames lack it, and leaves the rest alone.  A mixture
# with no background to fix it with raises rather than proceeding.

# %%
if HAVE_DATA:
    for f in sorted(in_filt)[:26]:
        h = fits.getheader(f)
        role = ("background" if "BACKGROUND" in str(h.get("TARGPROP", "")).upper()
                else "science" if str(h.get("TARGPROP", "")).upper().startswith("HIP-65426")
                else "reference")
        print(f"  {str(h.get('TARGPROP','')):24s} {role:11s} "
              f"S_BKDSUB={h.get('S_BKDSUB', 'absent')!s:9s} NINTS={h.get('NINTS')}")

# %%
if HAVE_DATA:
    dsets, info = sk.load_calints(in_filt, science_target="HIP-65426", half_px=40,
                                 partition="roll", filter=FILTER)
    px = float(info["pxscale"])
    m = miri.mode_for_filter(info["filter"])
    print(f"\n{m['filter']} / {m['image_mask']} ({m['kind']}), {px * 1e3:.2f} mas/px, "
          f"lambda = {m['lam_m'] * 1e6:.2f} um, partitions {list(dsets)}")

# %% [markdown]
# Two pixel scales appear in MIRI work and they differ by 0.6%: the instrument model says
# **0.109655″/px** and these science headers report **0.110327″/px** via `PIXAR_A2`.  The
# loader takes the data's own value, because that is what the astrometry of *these* frames
# is on; `miri.pixelscale()` returns the model's, for the model's own grids.  Over a 24″
# field 0.6% is a seventh of a pixel — small, but it is the kind of difference that moves a
# companion between annuli at the outer edge, so it is worth knowing which one you are using.
#
# ## 2. Throughput is a function of position, not radius
#
# This is the one that matters most.  A round occulter has a transmission `T(ρ)`; a 4QPM
# suppresses along two *lines*, so at one separation the throughput varies by a large factor
# with position angle.  Measured from STPSF for F1065C over the default grid:
#
# | ρ | min | max | ratio |
# |---|---|---|---|
# | 0.30″ | 0.176 | 0.421 | ×2.4 |
# | 0.58″ | 0.142 | 0.593 | ×4.2 |
# | 1.10″ | 0.159 | 1.007 | ×6.3 |
# | 1.53″ | 0.156 | 1.003 | ×6.5 |
# | 2.93″ | 0.170 | 0.998 | ×5.9 |
# | 5.63″ | 0.233 | 0.999 | ×4.3 |
# | 10.80″ | 0.406 | 0.991 | ×2.4 |
#
# A factor of up to **six and a half at constant separation**.  A radial model is not merely
# imprecise here — it is wrong in a way that tracks position angle, reporting a companion
# near a boundary up to three times fainter than it is and one between boundaries too
# bright, and a contrast curve built that way is an azimuthal average of two different
# things.  So `klip_tpe.instruments.miri` carries its own two-dimensional map, sampled on a
# (separation, detector azimuth) grid and cached like the radial grids are.
#
# Two details the map gives you that a hardcoded mask would not:
#
# * **The boundaries are not where they look.**  `locate_boundaries` measures them at az =
#   356°, 86°, 176° and 266° — four degrees off the detector axes, the same on all four, so
#   this is the mask's mounting angle and not noise.  Masking along detector rows and columns
#   would be off by ~1.5 px at 2″: it leaves the real dead zone in the data and discards good
#   pixels beside it.  An earlier version of that scan swept the circle in 15° steps and
#   reported all four boundaries *exactly* on the axes — which is the answer it would give if
#   they were there, and is why it looked right.
# * **The symmetry is two-fold, not four-fold.**  At 2″, `|T(az) − T(az+180)| ≤ 0.003` while
#   `|T(az) − T(az+90)|` reaches 0.18.  That is the mask's own asymmetry, and it is why the
#   map is measured over the whole circle rather than folded into one quadrant.
#
# The plateau between the boundaries sits at 1.00 within half a per cent, which is what it
# *should* be: the unocculted reference carries the same Lyot stop, so far from a boundary
# the phase mask takes nothing.

# %%
if HAVE_DATA and stpsf_psf.have_stpsf():
    g = miri.throughput_map(FILTER)
    T = miri.throughput_map_fn(g)
    az = np.linspace(0, 360, 721)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for rho in (0.6, 1.2, 2.4, 4.8):
        ax[0].plot(az, T(np.full(az.shape, rho), az), lw=1.4, label=f"{rho:.1f}\"")
    ax[0].set_xlabel("detector azimuth (deg)"); ax[0].set_ylabel("throughput")
    ax[0].set_title(f"{FILTER}: throughput vs position angle", fontsize=9)
    ax[0].legend(fontsize=7, title="separation", title_fontsize=7); ax[0].set_xlim(0, 360)
    for b in (356, 86, 176, 266):
        ax[0].axvline(b, color="0.7", lw=0.6, zorder=0)
    R, A = np.meshgrid(np.geomspace(0.3, 8.0, 160), az)
    im = ax[1].pcolormesh(A, R, T(R, A), shading="auto", cmap="magma", vmin=0, vmax=1.05)
    ax[1].set_yscale("log"); ax[1].set_xlabel("detector azimuth (deg)")
    ax[1].set_ylabel("separation (arcsec)"); ax[1].set_title("the map itself", fontsize=9)
    plt.colorbar(im, ax=ax[1], label="T"); plt.tight_layout()
    print("grey lines: the measured boundaries, four degrees off the detector axes")

# %% [markdown]
# ## 3. The dead zones: three things have to happen, and one must not
#
# Where the mask has taken most of the flux the photometry is unreliable and the stamp is
# *distorted*, not merely attenuated — the regime the radial stamp library cannot represent.
# `min_throughput=0.30`, roughly half the best throughput a 4QPM reaches, marks those pixels.
# On this frame that is 634 px, 9.7% of the 81×81 stamp.  Three things follow.
#
# **They must leave the noise estimate.**  Where the phase mask took the starlight it took
# the speckles too, so a dead-zone pixel is *quieter* than the ring it sits in.  Left in, it
# depresses σ, inflates every S/N, and gives the optimizer an incentive to choose parameters
# that preserve the dead zones.  `miri.dead_zone_pixel_mask` carries the detector geometry
# through every roll into the de-rotated frame and goes in as the metric's `pixel_mask`.
# Note the direction: taking them out *raises* σ and *lowers* the reported contrast.  Unlike
# a cut made because a region looked bright, this one is justified by the mask design before
# anyone looks at the image.
#
# **Injections must not land in them.**  The boundaries are fixed to the detector and the
# sampler works in sky position angle, so which sky angles are dead depends on how the
# telescope was pointed: `az = θ − truenorth − 270 − parang`.  `miri.forbidden_pa` takes the
# whole `angles` array and forbids a PA only when it is suppressed in more than `dead_frac`
# of the frames.  That fraction is 0.34 and not 0.5 on purpose — a JWST sequence is usually
# **two** rolls, so a PA one boundary eats is dead in exactly half the frames, and a `> 0.5`
# test would forbid nothing at all for the commonest observation there is, on an exact
# floating-point tie.  On these two rolls it returns eight sectors of ±2° covering 12% of the
# ring.  Injecting into a dead zone and "recovering" nothing is not a measurement of
# contrast; it is a measurement of the mask, presented as a measurement of the reduction.
#
# These two go **together**.  Injections that avoid sectors which still inflate the ring σ
# are scored against a noise level nothing is measuring them at — worse than doing neither.
#
# **And they must stay in the cube.**  The obvious move is to NaN them out, and it cannot
# work.  The reducer high-passes each frame at `nan_aware=False`, which is
# `ndimage.uniform_filter` — a *running-sum* filter, so a NaN poisons everything downstream
# of it along each axis rather than a box the filter's width.  One NaN near the corner of an
# 81×81 frame takes **73%** of it; the dead zones are lines through the star reaching all
# four frame edges, so they take all of it.  Then `np.nansum` of an all-NaN frame is 0.0,
# `bin_frames` drops zero-sum bins, and the reduction has no frames left.  No crop rescues
# it either — the dead zones cross the middle of the array.  The pixels are attenuated
# measurements, not missing ones: they belong in the cube and in the KLIP basis, and out of
# the *statistic*.

# %%
if HAVE_DATA and stpsf_psf.have_stpsf():
    angles = np.concatenate([np.asarray(d.angles, float).ravel() for d in dsets.values()])
    ann = (6.7, 36.0)
    mid_as = 0.5 * (ann[0] + ann[1]) * px
    fpa = miri.forbidden_pa(angles, rho_as=mid_as, filter=FILTER)
    pmask = miri.dead_zone_pixel_mask((81, 81), px, angles, filter=FILTER)
    print(f"rolls {sorted(set(np.round(angles, 1)))}, {len(fpa)} forbidden sector(s), "
          f"{int(pmask.sum())} px ({100 * pmask.mean():.1f}%) out of the noise estimate")

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].imshow(pmask, origin="lower", cmap="gray_r")
    ax[0].set_title("dead zones, de-rotated (two rolls)", fontsize=9)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    th = np.radians(np.linspace(0, 360, 721))
    ax[1] = plt.subplot(122, projection="polar")
    ax[1].plot(th, np.ones_like(th), color="0.8", lw=1)
    for c, w in fpa:
        s = np.radians(np.linspace(c - w, c + w, 32))
        ax[1].fill_between(s, 0, 1, alpha=0.6, color="crimson")
    ax[1].plot([np.radians(PLANET[1])], [1.0], "o", color="royalblue", label="HIP 65426 b")
    ax[1].set_yticks([]); ax[1].set_theta_zero_location("N"); ax[1].legend(fontsize=7)
    ax[1].set_title("forbidden sky position angles", fontsize=9)
    plt.tight_layout()

# %% [markdown]
# ## 4. The contrast axis
#
# HIP 65426 is behind the mask in every exposure, so its brightness has to be imported — the
# same problem as tutorial 3, and the same four terms.  What is different is where the flux
# density comes from, because there is no published MIRI stellar flux for this star to copy:
#
# | term | value | from |
# |---|---|---|
# | `S` | 0.06899 Jy | Planck(8600 K) through the F1140C bandpass, normalised to 2MASS Ks = 6.771, **ratio-anchored** to the F444W entry |
# | units | `S / (10⁶·PIXAR_SR)` = 2.4117e5 | `BUNIT = MJy/sr`, `PIXAR_SR = 2.8606e-13` |
# | `EE` | 0.4660 at 4.50 px | the model PSF unocculted *through the Lyot stop* |
# | `T_optics` | 1.0 | `PHOTMJSR` of the coronagraphic mode already carries its optics |
#
# giving `star_flux = 1.1238e5`.  Two things about that are worth copying as method.
#
# **Anchor, don't recompute.**  The Planck-through-the-bandpass recipe reproduces the repo's
# independently-checked F444W value of 0.40259 Jy to −3.3%, inside its own ±3%, and the
# residual is the 2MASS zero-point and effective-wavelength convention.  Taking the *ratio*
# to F444W cancels that convention and leaves only the model's shape between 4.4 and 11.3 µm
# — a Rayleigh–Jeans tail, where Planck and a real atmosphere differ by a per cent or two.
# Teff = 8600 K is Carter et al. (2023)'s own PHOENIX fit, so both entries rest on one model.
# The photosphere is the right thing to use here: the only excess those authors report is
# 3.5σ at 24 µm with `T_dust ≈ 300 K`, which contributes far less at 11 µm.
#
# **Then check it against something the recipe did not use.**  At `S = 0.0690 Jy`, Carter et
# al.'s quoted ~2.7 µJy F1140C sensitivity is a contrast floor of 3.9e-5, against their ~2e-4
# for the companion.  Those two hang together, and would not if `S` were wrong by a factor.
#
# **And 1.0 is not a safe default for this.**  `make_reducer(star_flux=None)` becomes
# `star_flux or 1.0`, which makes one unit of contrast worth *one count* against a cube whose
# pixels reach several hundred MJy/sr — 1.1e5 times too faint.  Injections then do nothing at
# any contrast: the calibration walks its whole ladder from 3e-5 to the 1e-1 cap with the
# median S/N flat at −0.11, 0.00, −0.02, −0.31, −0.04, reports that it could not calibrate,
# and the search ranks noise for as long as you let it.  A response flat over four orders of
# magnitude is not a faint source, it is an inert one.  `scripts/run_miri.py` now derives the
# flux unit or refuses to start.

# %%
STAR_FLUX = None
if HAVE_DATA and stpsf_psf.have_stpsf():
    phot = datasets.PHOTOMETRY[f"hip65426_{FILTER.lower()}"]
    STAR_FLUX = miri.star_flux_from_flux_density(FILTER, phot["flux_density_jy"],
                                                 info["pixar_sr"], bunit=info["bunit"])
    print(f"star_flux = {STAR_FLUX:.4e}   ({phot['ref']})")

# %% [markdown]
# ## 5. Reducer, space, objective — and the check that comes first
#
# `psf="stpsf"` routes MIRI to `miri.library`: radial stamps with the **two-dimensional**
# throughput attached.  The model's `azimuth_dependent` flag is what tells `inject_sources`
# to evaluate the throughput per frame, because a source at a fixed sky PA moves across the
# boundaries as the telescope rolls — so its attenuation genuinely differs frame to frame.
# The run log should say `injection miri_library`; `LibraryPSF` with a radial `throughput_fn`
# would also run, and would silently report azimuthal averages.
#
# Before spending hours, reduce once at the seeded default and look at what was built.  This
# is what `scripts/run_miri.py --check` does, and it is worth doing by hand once:

# %%
if HAVE_DATA and STAR_FLUX is not None:
    red = sk.make_reducer(dsets, pxscale=px, wavelength_m=m["lam_m"], diam_m=miri.DIAMETER_M,
                          psf="stpsf", star_flux=STAR_FLUX, mode="ADI+RDI", max_workers=3)
    obj, samp = generic.default_config(red, known=[PLANET], forbidden_pa=fpa, pixel_mask=pmask)
    space = generic.make_space(red, k_klip_max=20)
    space.project = generic.make_guard(red, k_max=20)

    dec = space.decode(space.default_vector())
    model = red.reducers[list(dec.selected)[0]].model
    cube = np.asarray(list(dsets.values())[0].cube, float)
    ev = red.reduce_config(dec, None, tag="check")
    img = np.asarray(ev.image if hasattr(ev, "image") else ev, float)
    yy, xx = np.mgrid[0:81, 0:81]
    inann = (np.hypot(xx - 40, yy - 40) >= ann[0]) & (np.hypot(xx - 40, yy - 40) <= ann[1])
    print(f"injection model      : {model.name}  azimuth_dependent="
          f"{getattr(model, 'azimuth_dependent', False)}")
    print(f"cube non-finite      : {100 * np.mean(~np.isfinite(cube)):.2f}%   (must be ~0)")
    print(f"default filter width : {dec.params.get('filter')} px")
    print(f"annulus finite       : {100 * np.isfinite(img[inann]).mean():.1f}%   (must be high)")

# %% [markdown]
# Those four lines are the whole check.  A radial model on a 4QPM, a cube with any NaN in it
# when the filter is 12 px wide, or an empty search annulus each produce a run that completes
# and reports numbers — so each is refused rather than warned about.  The annulus line in
# particular: measure finite pixels *inside the annulus*, not over the frame, because most of
# the frame is outside the reduced zones by design.  A single roll reduced ADI-only used to
# report "finite 0.0%" and then "check passed".
#
# ## 6. Optimize
#
# A small budget here so the notebook finishes; the paper runs use 1000.  Everything about
# the search itself is tutorial 1 — what is MIRI's is entirely in what has already been
# built above.

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
# On the real programme with `n_iter=1000` the calibration lands at S/N 5.08 for a contrast of
# 3e-4 and settles at 2.27e-4 — which is within ~15% of the companion's own published
# contrast, a pleasing accident that doubles as a sanity check on the flux scale.
#
# ## 7. Notes for real MIRI work
#
# * **The cache is the point, on a machine without STPSF.**  Every PSF grid, throughput map
#   and encircled-energy figure is cached as one file under `$KLIP_TPE_DATA/stpsf_cache`, so
#   a machine whose Python is too old for STPSF (3.9, say) runs from a cache computed
#   elsewhere.  A cache miss there is *"you are missing one file"*, not "install STPSF", and
#   the error names the file and the directory to copy it into.  `throughput_map` for one
#   filter is ~10 minutes and is check-pointed, so a killed process does not discard the
#   PSFs it already computed.
# * **The other two filters.**  F1065C is cached and ready; `datasets.PHOTOMETRY` has all
#   three (`S` = 0.07813 / 0.06899 / 0.03739 Jy for F1065C / F1140C / F1550C).  F1550C needs
#   its throughput map computed first.
# * **Which pixel scale.**  The loader uses the data's `PIXAR_A2` (0.110327″/px here); the
#   STPSF model says 0.109655.  Do not mix them within one analysis.
# * **`--star-center`.**  `CRPIX` is the *aperture reference point*, not the star.  Measure
#   the star by maximising the point symmetry of the stacked frame, **not** with a flux
#   centroid — on a four-quadrant residual the centroid is biased by the pattern rather than
#   the centre, and on these frames it lands 0.4 px on the opposite side and moves by 0.4 px
#   with the aperture radius.  The symmetry solution here is 0.39 px (45 mas) from CRPIX with
#   the two rolls agreeing to 0.05 px, so CRPIX is adequate for this data set and the flag is
#   for the ones where it is not.
# * **The Lyot mode (F2300C)** is a round occulter with a support bar, so the radial path is
#   the right one for it; `mode_for_filter("F2300C")["kind"]` is `"lyot"` and
#   `quadrant_mask` masks the catalogued 2.16″ spot instead of a cross.
# * **The whole path as one command**, which is what to use in anger:
#   ```
#   python3 scripts/run_miri.py --data ~/Data/JWST/hip65426_miri --target HIP-65426 \
#       --filter F1140C --crop 40 --workers 3 --known 0.826 150.2 --n-iter 1000 --show
#   ```
#   `--check` first: it builds everything, reduces once at the default, and stops.
