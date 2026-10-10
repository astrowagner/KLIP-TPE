# %% [markdown]
# # Tutorial 5: JWST/MIRI Coronagraphy with a Four-Quadrant Phase Mask (HIP 65426 b, F1140C)
#
# Tutorial 3 optimized the NIRCam half of ERS 1386 on HIP 65426. This is the MIRI half of the
# same program on the same star: F1140C behind FQPM1140, two rolls 9.4° apart, and the
# reference star HIP 68245 (φ Cen) in a 9-point small-grid dither. Running one target through
# two instruments separates what belongs to the pipeline from what belongs to the
# coronagraph.
#
# Three of MIRI's four coronagraphs are four-quadrant phase masks (4QPM). Instead of blocking
# a disk, they put a π phase step along two perpendicular lines through the star. Everything
# downstream that assumes a round occulter is wrong here, and each of the three consequences in
# sections 2 and 3 gives a wrong answer without an error if it is missed.
#
# **Data.** `scripts/fetch_jwst_ar.py` lists and downloads the program's MIRI coronagraphy
# (F1140C and F1550C for this star) from MAST, with no login:
# ```
# python3 scripts/fetch_jwst_ar.py --targets hip65426_miri                         # list first
# python3 scripts/fetch_jwst_ar.py --targets hip65426_miri --download --outdir ~/Data/JWST
# ```
# The files land in `~/Data/JWST/hip65426_miri/`. Set `KLIP_TPE_JWST_MIRI` if you keep them
# elsewhere. Every cell below is guarded, so the notebook reads as a document without them.
#
# **PSF model.** The throughput map and the off-axis PSFs come from STPSF (Python ≥ 3.10, with
# its data files), and they are cached. A machine without STPSF runs from three cache files per
# filter, copied into `$KLIP_TPE_DATA/stpsf_cache`: the stamp grid (`stpsf_MIRI_F1140C_*.fits`),
# the throughput map (`miri_thrumap_F1140C_*.npz`) and the encircled energy
# (`eeunocc_MIRI_F1140C_*.fits`). A missing file stops the notebook with its name.

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
if os.path.isdir(os.path.join(DATA, "mastDownload")):   # the archive files, not a reprocessing beside them:
    DATA = os.path.join(DATA, "mastDownload")           # load_calints refuses one exposure found twice
RUN_DIR = os.path.abspath("runs/hip65426_f1140c")
# Re-running RESUMES this directory -- delete it to search again (see tutorial 01).
FILTER = "F1140C"
PLANET = (0.820, 149.9)            # HIP 65426 b, Carter et al. 2023, Table 3 (F444W astrometry)

files = sorted(glob.glob(os.path.join(DATA, "**", "jw*_calints.fits"), recursive=True))
in_filt = [f for f in files if str(fits.getheader(f).get("FILTER", "")).upper() == FILTER]
HAVE_DATA = len(in_filt) >= 4
print(f"{len(files)} calints under {DATA}, {len(in_filt)} in {FILTER}"
      f"  ({'ok' if HAVE_DATA else 'run the fetch script'})")

# %% [markdown]
# ## 1. What an Archive Download of One Program Contains
#
# A MIRI coronagraphic program takes dedicated background pointings (offset exposures of blank
# sky) as well as its targets, because the thermal background at 11 µm is large and
# structured. The download for HIP 65426 also holds the program's second target, HD 141569A,
# a resolved disk, and its reference star HD 140986. `load_calints` sorts the files into
# science, reference and background, and `ref_targets="HIP-68245"` keeps the RDI library to
# φ Cen. Without it, every pointing that is neither the science target nor a background goes
# into the library, the disk included.
#
# The pipeline's background subtraction has already been applied to some of the files.
# Image2 runs its background step when the association has background members, which the
# science targets have and the pure PSF reference stars usually do not. The header records it
# as `S_BKDSUB`. For this program at F1140C:
#
# | target | role | files | `S_BKDSUB` |
# |---|---|---|---|
# | HIP-65426 | science | 2 | `COMPLETE` |
# | HIP-68245 (φ Cen) | reference | 9 | absent |
# | HD-141569A | second science target (disk) | 2 | `COMPLETE` |
# | HD-140986 | its reference | 5 | absent |
# | `*-BACKGROUND` | background | 8 | absent (they are the background) |
#
# Untreated, the reference frames carry a sky pedestal of about 19 MJy/sr and the glow of the
# mask edges, and the science frames carry neither. The leading KL mode is then the sky rather
# than the stellar PSF. `load_calints` reads `S_BKDSUB` per exposure, subtracts the median of
# the program's own blank-sky pointings from the frames that lack it, and leaves the rest
# alone. A mixture with no background pointings to fix it raises an error.
#
# On paired 1000-evaluation runs that differed only in this, the calibration contrast fell
# from 2.27 × 10⁻⁴ to 1.63 × 10⁻⁴. Once the library was consistent, the same default
# configuration reached S/N 5 on a source 28% fainter. The k-scan changed shape too. With the
# mismatch it was flat to 1.5% over k = 4 to 20, with k = 1 on top, because the leading KL
# mode was the pedestal. With a consistent library, low k is clearly penalized and more modes
# keep helping.

# %%
if HAVE_DATA:
    for f in sorted(in_filt)[:26]:
        h = fits.getheader(f)
        tp = str(h.get("TARGPROP", "")).upper()
        role = ("background" if "BACKGROUND" in tp else "science" if tp.startswith("HIP-65426")
                else "reference" if tp.startswith("HIP-68245") else "not used")
        print(f"  {str(h.get('TARGPROP','')):24s} {role:11s} "
              f"S_BKDSUB={h.get('S_BKDSUB', 'absent')!s:9s} NINTS={h.get('NINTS')}")

# %%
if HAVE_DATA:
    # partition="all": both rolls in ONE partition, so ADI+RDI really uses the other roll
    # (tutorial 3, section 2); ref_targets keeps the disk HD 141569A out of the library.
    dsets, info = sk.load_calints(in_filt, science_target="HIP-65426", half_px=40,
                                 partition="all", filter=FILTER, ref_targets="HIP-68245")
    px = float(info["pxscale"])
    m = miri.mode_for_filter(info["filter"])
    print(f"\n{m['filter']} / {m['image_mask']} ({m['kind']}), {px * 1e3:.2f} mas/px, "
          f"lambda = {m['lam_m'] * 1e6:.2f} um, partitions {list(dsets)}")

# %% [markdown]
# Two pixel scales appear in MIRI work, and they differ by 0.6%. The instrument model says
# 0.109655″/px, and these science headers give 0.110327″/px through `PIXAR_A2`. The loader
# takes the data's own value, because the astrometry of these frames is on it.
# `miri.pixelscale()` returns the model's value when STPSF is installed, and the documented
# 0.110 when it is not. Across a 24″ field 0.6% is 0.15″, 1.3 px from edge to edge, enough to
# move a companion between annuli at the outer edge. Use one scale within an analysis.
#
# ## 2. Throughput Is a Function of Position, Not Radius
#
# This matters most. A round occulter has a transmission `T(ρ)`. A 4QPM suppresses along two
# lines, so at one separation the throughput varies by a large factor with position angle.
# Measured from STPSF for F1065C over the default grid:
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
# At constant separation the throughput varies by up to a factor of 6.5. A radial model is
# wrong in a way that tracks position angle. It reports a companion near a boundary several
# times too faint and one between boundaries too bright, and a contrast curve built that way
# averages two different things. `klip_tpe.instruments.miri` therefore carries its own
# two-dimensional map, sampled on a grid of separation and detector azimuth and cached like
# the radial grids.
#
# The map shows two things a hardcoded mask would miss.
#
# * **The boundaries are not on the detector axes.** For F1065C, `locate_boundaries` measures
#   them at azimuths of 356°, 86°, 176° and 266°, four degrees off the axes and the same on all
#   four, which is the mask's mounting angle. Masking along detector rows and columns would be
#   off by 1.3 px at 2″, leaving the real dead zone in the data and discarding good pixels
#   beside it. The F1140C map carries its own measured boundaries (`g["boundaries"]`).
# * **The symmetry is two-fold, not four-fold.** At 2″, `|T(az) − T(az+180)| ≤ 0.003` while
#   `|T(az) − T(az+90)|` reaches 0.18. That is the mask's own asymmetry, and it is why the map
#   covers the whole circle rather than one quadrant.
#
# Between the boundaries the throughput sits at 1.00 to within about 1%, as it should. The
# unocculted reference carries the same Lyot stop, so far from a boundary the phase mask
# takes nothing.

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
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for rho in (0.6, 1.2, 2.4, 4.8):
        ax[0].plot(az, T(np.full(az.shape, rho), az), lw=1.4, label=f"{rho:.1f}\"")
    ax[0].set_xlabel("detector azimuth (deg)"); ax[0].set_ylabel("throughput")
    ax[0].set_title(f"{FILTER}: throughput vs position angle", fontsize=9)
    ax[0].legend(fontsize=7, title="separation", title_fontsize=7); ax[0].set_xlim(0, 360)
    if g.get("boundaries") is not None:
        for b in np.atleast_1d(g["boundaries"]):
            ax[0].axvline(float(b) % 360, color="0.7", lw=0.6, zorder=0)
        print(f"grey lines: the boundaries measured for {FILTER}, "
              f"{', '.join(f'{float(b) % 360:.0f}' for b in np.atleast_1d(g['boundaries']))} deg")
    R, A = np.meshgrid(np.geomspace(0.3, 8.0, 160), az)
    im = ax[1].pcolormesh(A, R, T(R, A), shading="auto", cmap="magma", vmin=0, vmax=1.05)
    ax[1].set_yscale("log"); ax[1].set_xlabel("detector azimuth (deg)")
    ax[1].set_ylabel("separation (arcsec)"); ax[1].set_title("the map itself", fontsize=9)
    plt.colorbar(im, ax=ax[1], label="T"); plt.tight_layout()

# %% [markdown]
# ## 3. The Dead Zones: Three Things Have to Happen, and One Must Not
#
# Where the mask has taken most of the flux, the photometry is unreliable and the stamp is
# distorted, not merely attenuated, which is the regime the radial stamp library cannot
# represent. `min_throughput=0.30`, 30% of the plateau, marks those pixels. On this frame that
# is 634 px, 9.7% of the 81×81 stamp. Three things follow.
#
# **They must leave the noise estimate.** Where the phase mask took the starlight it took the
# speckles too, so a dead-zone pixel is quieter than the ring it sits in. Left in, it depresses
# σ, inflates every S/N, and rewards parameters that preserve the dead zones.
# `miri.dead_zone_pixel_mask` carries the detector geometry through every roll into the
# derotated frame, and it goes in as the metric's `pixel_mask`. Taking these pixels out raises
# σ, so every S/N and contrast limit becomes more conservative. The cut follows from the mask
# design, before anyone looks at the image.
#
# **Injections must not land in them.** The boundaries are fixed to the detector and the
# sampler works in sky position angle, so which sky angles are dead depends on the pointing:
# `az = θ − truenorth − 270 − parang`. `miri.forbidden_pa` takes the whole `angles` array and
# forbids a PA only when it is suppressed in more than `dead_frac` of the frames. That
# fraction is 0.34 rather than 0.5 on purpose. A JWST sequence is usually two rolls, so a PA
# that one boundary removes is dead in exactly half the frames, and a `> 0.5` test would
# forbid nothing on an exact floating-point tie. Injecting into a dead zone and recovering
# nothing measures the mask, not the reduction.
#
# The two go together. An injection kept out of the dead zones, but scored against a noise ring
# that still contains them, is measured against a σ that is too low.
#
# **And they must stay in the cube.** Setting them to NaN cannot work. The reducer high-passes
# each frame with `nan_aware=False`, which is `ndimage.uniform_filter`, a running-sum filter.
# A NaN therefore spreads along each axis from where it sits, not just across a box the
# filter's width. One NaN near the corner of an 81×81 frame takes 73% of it, and the dead zones
# are lines through the star that reach all four edges, so they take all of it. `np.nansum` of
# an all-NaN frame is then 0.0, `bin_frames` drops zero-sum bins, and the reduction has no
# frames left. No crop helps, because the dead zones cross the middle of the array. The
# pixels are attenuated measurements, not missing ones. They belong in the cube and in the KLIP
# basis, and out of the statistic.

# %%
if HAVE_DATA:
    angles = np.concatenate([np.asarray(d.angles, float).ravel() for d in dsets.values()])
    ann = (6.7, 36.0)
    mid_as = 0.5 * (ann[0] + ann[1]) * px
    fpa = miri.forbidden_pa(angles, rho_as=mid_as, filter=FILTER)
    pmask = miri.dead_zone_pixel_mask((81, 81), px, angles, filter=FILTER)
    print(f"rolls {sorted(set(np.round(angles, 1)))}, {len(fpa)} forbidden sector(s) covering "
          f"{100 * sum(2 * w for _, w in fpa) / 360:.1f}% of the ring, "
          f"{int(pmask.sum())} px ({100 * pmask.mean():.1f}%) out of the noise estimate")

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].imshow(pmask, origin="lower", cmap="gray_r")
    ax[0].set_title("dead zones, derotated (two rolls)", fontsize=9)
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
# ## 4. The Contrast Axis
#
# HIP 65426 is behind the mask in every exposure, so its brightness has to be imported, as in
# tutorial 3 and with the same four terms. The flux density needs its own source, because no
# MIRI stellar flux has been published for this star:
#
# | term | value | from |
# |---|---|---|
# | `S` | 0.06899 Jy | Planck(8600 K) through the F1140C bandpass, normalized to 2MASS Ks = 6.771, as a ratio to the F444W entry |
# | units | `S / (10⁶·PIXAR_SR)` = 2.4117e5 | `BUNIT = MJy/sr`, `PIXAR_SR = 2.8606e-13` |
# | `EE` | 0.4660 at 4.50 px | the model PSF unocculted through the Lyot stop |
# | `T_optics` | 1.0 | `PHOTMJSR` of the coronagraphic mode already carries its optics |
#
# This gives `star_flux = 1.1238e5`. Two steps in it are worth copying.
#
# **Anchor the ratio rather than recomputing.** The Planck-through-the-bandpass recipe gives
# 3.3% less than the independently checked F444W value of 0.40259 Jy, about that value's own
# ±3% uncertainty. The difference is the 2MASS zero point and effective-wavelength convention.
# Taking the ratio to F444W cancels that convention and leaves only the spectral shape between
# 4.4 and 11.3 µm. That is the Rayleigh–Jeans tail, where Planck and a real atmosphere differ by
# a percent or two. Teff = 8600 K is the PHOENIX fit of Carter et al. (2023), so both entries
# rest on one model. The photosphere is the right input here, because the only excess those
# authors report is at 24 µm (3.5σ, `T_dust ≈ 300 K`), and it contributes far less at 11 µm.
#
# **Then check it against something the recipe did not use.** Carter et al. (2023, Table 3)
# give the companion as ΔF1140C = 8.264 ± 0.021, a contrast of 4.95 × 10⁻⁴, and as
# (7.40 ± 1.16) × 10⁻¹⁹ W m⁻² µm⁻¹, which is 31.5 µJy at 11.3 µm. Their ratio puts the star at
# 0.0637 Jy. Our `S = 0.0690 Jy` is 8.3% higher, against their ±3.5% on the stellar magnitude
# and our own ±5%, a 1.4σ difference. Their background-limited sensitivity of about 2.7 µJy is
# then a contrast floor of 3.9 × 10⁻⁵, in line with the 5 × 10⁻⁵ their contrast curve reaches
# beyond 3″. The 2 × 10⁻⁴ in their text is their S/N=5 limit at 1″, and the planet is 2.5×
# brighter than it.
#
# **A star flux of 1.0 does not work.** `make_reducer(star_flux=None)` uses `star_flux or 1.0`,
# which makes one unit of contrast worth one count in a cube whose pixels reach several hundred
# MJy/sr, 1.1 × 10⁵ times too faint. The injections then do nothing at any contrast. The
# calibration walks its whole ladder from 3 × 10⁻⁵ to the cap of 0.1 with the median S/N flat at
# −0.11, 0.00, −0.02, −0.31 and −0.04, reports that it could not calibrate, and the search ranks
# noise. A response that stays flat over 3.5 decades of contrast is not a faint source but an
# inert one. `scripts/run_miri.py` derives the flux unit or refuses to start.

# %%
STAR_FLUX = None
if HAVE_DATA:
    phot = datasets.PHOTOMETRY[f"hip65426_{FILTER.lower()}"]
    STAR_FLUX = miri.star_flux_from_flux_density(FILTER, phot["flux_density_jy"],
                                                 info["pixar_sr"], bunit=info["bunit"])
    print(f"star_flux = {STAR_FLUX:.4e}   ({phot['ref']})")

# %% [markdown]
# ## 5. Reducer, Space and Objective, and the Check That Comes First
#
# `psf="stpsf"` routes MIRI to `miri.library`, which attaches the two-dimensional throughput
# to radial stamps. The model's `azimuth_dependent` flag tells `inject_sources` to evaluate
# the throughput per frame, because a source at a fixed sky PA moves across the boundaries as
# the telescope rolls, so its attenuation differs from frame to frame. The run log should say
# `injection miri_library`. A `LibraryPSF` with a radial `throughput_fn` would also run, and
# would silently report azimuthal averages.
#
# Both rolls are in one partition, so `mode="ADI+RDI"` uses the other roll and φ Cen.
# `search_angles=False` leaves out `angsep` and `anglemax`, which do little with two rolls 9.4°
# apart (`scripts/run_miri.py` drops them too, and also searches the reference library).
#
# Before spending hours, reduce once at the default configuration and look at what was built.
# This is what `scripts/run_miri.py --check` does:

# %%
if HAVE_DATA and STAR_FLUX is not None:
    red = sk.make_reducer(dsets, pxscale=px, wavelength_m=m["lam_m"], diam_m=miri.DIAMETER_M,
                          psf="stpsf", star_flux=STAR_FLUX, mode="ADI+RDI", max_workers=3)
    obj, samp = generic.default_config(red, known=[PLANET], forbidden_pa=fpa, pixel_mask=pmask)
    space = generic.make_space(red, k_klip_max=20, search_angles=False)
    space.project = generic.make_guard(red, k_max=20)
    print(space.names)

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
# Those four lines are the whole check. A radial model on a 4QPM, a cube with any NaN in it when
# the filter is 12 px wide, and an empty search annulus each produce a run that completes and
# reports numbers, so `run_miri.py --check` refuses each one. Measure the finite pixels inside
# the annulus, not over the frame, because most of the frame lies outside the reduced zones by
# design.
#
# ## 6. Optimize
#
# The budget is tutorial 1's: 300 evaluations with 40 of warm-up, three injection draws averaged
# per trial, and the six best candidates validated on eight fresh injection sets each. On these
# data a single draw scatters by 0.84 in S/N, the example `RunConfig.n_remeasure` documents.
# With pyKLIP the run took just under two hours on a Mac. Everything else about the search is
# tutorial 1. What is MIRI's is in what has already been built above.
#
# One setting is about the picture rather than the score. The four injected sources step across
# the band in radius, and the innermost one passes within 1.6 FWHM of HIP 65426 b whenever it
# lands at the planet's position angle. The standard 1.5-FWHM exclusion allows that, and it does
# not affect the search score, but in the panel MIRI's PSF merges the two into one blob. The cell
# keeps every injection 3 FWHM (1.1″) from the planet.

# %%
if HAVE_DATA and STAR_FLUX is not None:
    samp.excl_fwhm = 3.0          # injections >= 3 FWHM (1.1") from HIP 65426 b
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
# The calibration contrast is a sensitivity, the contrast at which the default configuration
# sees an injected source at S/N ≈ 5. It says nothing about the flux scale on its own. Here it
# settled at 9.95 × 10⁻⁵, for injected sources between 1.4″ and 3.3″, and HIP 65426 b, at
# 4.95 × 10⁻⁴ in Carter et al. (2023, Table 3), is five times brighter. The flux-scale check is
# the one in section 4.
#
# The calibration's k-scan picked 20 modes, the top of its range, and the log warns that this
# is an edge hit. The data want more modes than the cap of 20 allows, and the cap limits the
# search as well. For a real analysis, raise it, to 40 for example, with `k_klip_max` in
# `make_space` and `k_max` in `make_guard`. The validated winner is evaluation 275, at 8.15.
#
# ## 7. Notes for Real MIRI Work
#
# * **The cache.** Every PSF grid, throughput map and encircled energy is cached as one file
#   under `$KLIP_TPE_DATA/stpsf_cache`, so a machine whose Python is too old for STPSF (3.9,
#   say) runs from files computed elsewhere. A cache miss names the missing file. A throughput
#   map takes about 25 minutes (576 PSFs) and is checkpointed, so a killed process keeps the
#   PSFs it already computed.
# * **The other filter.** HIP 65426 was also observed in F1550C. `datasets.PHOTOMETRY` has
#   `S` for F1065C, F1140C and F1550C (0.07813, 0.06899 and 0.03739 Jy). Each filter needs its
#   own cache files.
# * **Which pixel scale.** The loader uses the data's `PIXAR_A2` (0.110327″/px here), and the
#   STPSF model says 0.109655. Do not mix them within one analysis.
# * **`--star-center`.** `CRPIX` is the aperture reference point, not the star. Measure the
#   star by maximizing the point symmetry of the stacked frame, not with a flux centroid. On a
#   four-quadrant residual the centroid is biased by the pattern. On these frames it lands
#   0.4 px on the opposite side and moves by 0.4 px with the aperture radius. The symmetry
#   solution here is 0.39 px (43 mas) from CRPIX, with the two rolls agreeing to 0.05 px, so
#   CRPIX is adequate for this data set, and the flag is for data sets where it is not.
# * **The Lyot mode (F2300C)** is a round occulter with a support bar.
#   `mode_for_filter("F2300C")["kind"]` is `"lyot"`. It uses the same two-dimensional map as the
#   4QPM filters, and `quadrant_mask` adds the cataloged 2.16″ spot to the thresholded map.
# * **The whole path as one command.** `scripts/run_miri.py` is the production driver:
#   ```
#   python3 scripts/run_miri.py --data ~/Data/JWST/hip65426_miri/mastDownload --target HIP-65426 \
#       --ref-target HIP-68245 --filter F1140C --crop 40 --known 0.820 149.9 --show
#   ```
#   Its defaults differ from this notebook. It searches three annuli for 1000 evaluations each
#   (150 of warm-up), validates on ten injection sets, and also searches the reference library
#   (pyKLIP's `mode` and `maxnumbasis`). Run it with `--check` first. That builds everything,
#   moves each searched dimension alone to check that it changes the reduction, reduces once at
#   the default configuration, and stops.
