# %% [markdown]
# # Tutorial 3: JWST/NIRCam Coronagraphy of HIP 65426 b with ADI and RDI (pyKLIP Engine)
#
# The ERS 1386 NIRCam observations of HIP 65426 (Carter et al. 2023) are the reference JWST
# high-contrast data set: two science rolls through the MASK335R coronagraph, and a reference
# star, HIP 68245 (φ Cen), observed in a 9-point small-grid dither. With only 10° of roll this
# is an RDI problem. What matters is how the 18 reference frames are used, not the field
# rotation.
#
# spaceKLIP is the community pipeline for these data and hands them to pyKLIP. klip-tpe plugs
# in at that hand-over: both rolls in one partition, the reference exposures as the RDI
# library, pyKLIP's `klip_parallelized` as the engine, and the TPE search on top.
#
# **Data.** `tutorials/fetch_jwst_hip65426.py` downloads the F444W `calints` products from
# MAST (no login, about 60 MB):
# ```
# python3 -m pip install astroquery pyklip       # on Python 3.9 with numpy < 2.1: "pyklip<2.9"
# python3 tutorials/fetch_jwst_hip65426.py       # --dry-run lists the files first
# ```
# The notebook skips the run, and says so, when the files are not there.
#
# **PSF model.** Sections 3 to 5 need the STPSF model of MASK335R. Either install STPSF
# (Python ≥ 3.10, with its data files), or copy the cached grid files
# (`stpsf_NIRCam_F444W_MASK335R_<key>.fits` and `eeunocc_NIRCam_F444W_<key>.fits`) into
# `$KLIP_TPE_DATA/stpsf_cache` from a machine that has computed them. Section 3 stops with that
# message if neither is available.

# %%
import glob, os, time, warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage

from klip_tpe import Runner, RunConfig, ValidationConfig, CalibrationConfig, datasets, stpsf_psf
from klip_tpe.reducer import Dataset, ReductionRequest
from klip_tpe.backends import spaceklip as sk
from klip_tpe.instruments import generic
from klip_tpe.display import LiveDisplay
from klip_tpe.metrics import MawetPeakSNR

DATA = os.path.expanduser(os.environ.get("KLIP_TPE_JWST", "~/.klip_tpe/data/jwst_hip65426"))
RUN_DIR = os.path.abspath("runs/hip65426_f444w")
# Re-running RESUMES this directory -- delete it to search again (see tutorial 01).  Delete it
# also if it was made without the STPSF model: a resumed run keeps its old injection scale.
PLANET = (0.820, 149.9)                    # HIP 65426 b in F444W, Carter et al. 2023, Table 3

files = sorted(glob.glob(os.path.join(DATA, "**", "jw*calints.fits"), recursive=True))
files = [f for f in files if fits.getheader(f).get("FILTER") == "F444W"]   # the fetch script can add F300M
def _targ(f):
    h = fits.getheader(f)
    return str(h.get("TARGPROP", h.get("TARGNAME", ""))).replace("-", "").replace("_", "").upper()
sci_files = [f for f in files if _targ(f).startswith("HIP65426")]
ref_files = [f for f in files if f not in sci_files]
HAVE_DATA = len(sci_files) >= 2 and len(ref_files) >= 1
print(f"science exposures: {len(sci_files)}, reference exposures: {len(ref_files)}"
      f"  ({'ok' if HAVE_DATA else 'data not found - run the fetch script'})")

# %% [markdown]
# ## 1. Stage-2 Products Need Cleaning and Alignment First
#
# Stage-2 `calints` products flag bad pixels in the `DQ` extension and leave their repair and
# the frame alignment to post-processing. Each small-grid-dither reference exposure also sits
# at its own sub-pixel offset. Subtracting such references leaves a residual far brighter than
# any planet. Without this section the planet is undetectable.
#
# **In production, use spaceKLIP's `ImageTools`** (`quick_cleaning`, `align_frames` and so
# on). It does this properly and writes `STARCENX/Y` into the headers, and `load_spaceklip`
# then reads its products directly (section 7). So that this notebook stands alone, here is the
# minimal version: fill the flagged pixels from their neighbors, then register every frame on
# the median science frame by FFT cross-correlation.
#
# **Fill the flagged pixels and nothing else.** A coronagraphic PSF is supposed to be full of
# sharp, isolated blobs: the six-lobed Lyot-stop pattern of the star, and for a companion
# behind MASK335R a three-bar core (Carter et al. 2023, Fig. 3). A repair that decides from the
# pixel values what is an outlier, such as a sigma-clipped median filter, rewrites the PSF. On
# these frames such a filter changed about 5,000 pixels per frame, of which about 1,560 were
# flagged, and cut the peak of HIP 65426 b to 39%.

# %%
def read_calints(paths):
    """(frames, position angles) from stage-2 products; DQ=DO_NOT_USE pixels -> NaN."""
    ims, pas = [], []
    for f in paths:
        with fits.open(f) as h:
            d = np.asarray(h["SCI"].data, float)
            dq = np.asarray(h["DQ"].data, int) if "DQ" in h else np.zeros_like(d, int)
            s = h["SCI"].header
            pa = float(s["ROLL_REF"]) - float(s.get("V3I_YANG", 0.0)) * float(s.get("VPARITY", 1))
            d = np.where((dq & 1).astype(bool), np.nan, d)          # bit 0 = DO_NOT_USE
            for i in range(d.shape[0]):                             # one entry per integration
                ims.append(d[i]); pas.append(pa)
    return np.array(ims), np.array(pas)

def repair(cube, maxit=20):
    """Fill every DQ-flagged (NaN) pixel with the median of its finite 8 neighbours --
    spaceKLIP's treatment -- and touch nothing else.  Clusters close from the rim inward."""
    out = np.array(cube, float)
    for i, im in enumerate(out):
        for _ in range(maxit):
            bad = ~np.isfinite(im)
            if not bad.any():
                break
            p = np.pad(im, 1, constant_values=np.nan)
            st = np.stack([p[1 + dy:1 + dy + im.shape[0], 1 + dx:1 + dx + im.shape[1]]
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN stencils inside a cluster
                im[bad] = np.nanmedian(st, axis=0)[bad]
        out[i] = im
    return out

def xcorr_shift(im, ref, search=6):
    """Sub-pixel (dx, dy) that brings `im` onto `ref` (FFT cross-correlation + parabola)."""
    cc = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(im) * np.conj(np.fft.rfft2(ref)), s=im.shape))
    c = np.array(im.shape) // 2
    sub = cc[c[0] - search:c[0] + search + 1, c[1] - search:c[1] + search + 1]
    k = np.unravel_index(np.argmax(sub), sub.shape)
    dy, dx = k[0] - search, k[1] - search
    par = lambda a, b, c_: 0.0 if (a - 2 * b + c_) == 0 else 0.5 * (a - c_) / (a - 2 * b + c_)
    if 0 < k[0] < sub.shape[0] - 1: dy += par(sub[k[0] - 1, k[1]], sub[k[0], k[1]], sub[k[0] + 1, k[1]])
    if 0 < k[1] < sub.shape[1] - 1: dx += par(sub[k[0], k[1] - 1], sub[k[0], k[1]], sub[k[0], k[1] + 1])
    return dx, dy

def align(cube, ref):
    out = np.empty_like(cube)
    for i, im in enumerate(cube):
        dx, dy = xcorr_shift(im, ref)
        out[i] = ndimage.shift(im, (-dy, -dx), order=3, mode="constant", cval=0.0)
    return out

if HAVE_DATA:
    sci_raw, pas = read_calints(sci_files)
    ref_raw, _ = read_calints(ref_files)
    print(f"{sci_raw.shape[0]} science integrations at PA {np.unique(np.round(pas, 1))} deg, "
          f"{ref_raw.shape[0]} reference integrations, {sci_raw.shape[1]}x{sci_raw.shape[2]} px")
    sci_c, ref_c = repair(sci_raw), repair(ref_raw)
    anchor = np.median(sci_c, axis=0)
    sci_a, ref_a = align(sci_c, anchor), align(ref_c, anchor)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for a, im, t in zip(ax, (sci_raw[0], sci_a[0], ref_a[0]),
                        ("raw science integration (DQ masked)", "DQ filled + aligned", "reference star")):
        v = np.nanpercentile(im[np.isfinite(im)], [5, 99.5])
        a.imshow(im, origin="lower", cmap="inferno", vmin=v[0], vmax=v[1]); a.set_title(t, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()

# %% [markdown]
# ## 2. One Partition, Both Rolls, the Reference Star as the RDI Library
#
# klip-tpe expects the star at the center of the array, so the crop is taken about it. The
# four science integrations (two per roll, the two of a roll at the same position angle) go
# into **one** `Dataset`, and the 18 reference integrations go with it as its `ref_cube`.
#
# **Why one partition and not one per roll.** A partition is reduced on its own. With one
# partition per roll, every frame in it has the same PA, so pyKLIP's `ADI` has no reference
# frames and `ADI+RDI` is just `RDI`. The other roll is never in the basis, because it is in
# the other partition. Only with both rolls together does `ADI` mean what it means for JWST,
# subtracting roll 2 from roll 1 and the reverse. `spaceklip.load_calints` does section 1 and
# this in one call, where `partition='all'` is this layout and `'roll'` the other.
#
# **`CRPIX` marks the mask, not the star.** It is the aperture reference point, the same in
# every file of the program, dithers included. On these frames it misses HIP 65426 by 0.78 px.
# That would put the companion 0.8 px inside its own separation and mismatch its KLIP
# throughput against the injected fakes.
#
# There is no off-axis image of the star to centroid. HIP 65426 and the reference star are
# behind MASK335R in every exposure. A 180° symmetry fit to the coronagraphic residual is
# dominated by speckles, and on these data it moved the center by 2 px between the rolls.
# The companion itself works. Derotating about a center that is off by `δ` puts it at
# `u + R(PA_k)·δ` in roll `k`, so each roll gives `δ = R(−PA_k)·(measured − expected)`
# independently, and the two rolls agree to 0.12 px. `datasets.PHOTOMETRY` carries the
# result, and `scripts/check_hip65426_contrast.py` is the solve. The proper source is
# spaceKLIP's star-centering step (`STARCENX/Y`), which the path in section 7 uses.
#
# This fixes the geometry only. The contrast axis is section 3.

# %%
if HAVE_DATA:
    hdr = fits.getheader(sci_files[0], "SCI")
    pxscale = float(np.sqrt(hdr["PIXAR_A2"]))                      # 0.0626"/px (NIRCam LW)
    pixar_sr = float(hdr["PIXAR_SR"])                              # for the flux scale, section 3
    wavelength = 4.44e-6                                           # F444W mean wavelength
    PHOT = datasets.PHOTOMETRY["hip65426_f444w"]
    cx, cy = PHOT["star_center"]                                   # NOT CRPIX -- see above
    print(f"CRPIX ({hdr['CRPIX1']-1:.2f}, {hdr['CRPIX2']-1:.2f}) vs the star at "
          f"({cx:.2f}, {cy:.2f}): {np.hypot(cx-hdr['CRPIX1']+1, cy-hdr['CRPIX2']+1):.2f} px apart")
    H = 55                                                          # 111x111 px = 6.9" square

    def crop(cube):
        # ODD size, deliberately.  The shift puts the star on the integer pixel round(cx),
        # and the crop starts H pixels before it, so the star lands on index H -- which is
        # the array centre (n-1)/2 only when n = 2H+1.  With an even 2H crop the star sits
        # half a pixel off the centre klip-tpe assumes in EVERY axis (0.71 px in all), and
        # every separation and position angle downstream is measured from the wrong origin.
        fx, fy = cx - round(cx), cy - round(cy)
        x0, y0 = int(round(cx)) - H, int(round(cy)) - H
        return np.array([ndimage.shift(im, (-fy, -fx), order=3)[y0:y0 + 2 * H + 1, x0:x0 + 2 * H + 1]
                         for im in cube], np.float32)

    sci_x, ref_x = crop(sci_a), crop(ref_a)
    dsets = {"sci": Dataset(sci_x, pas, name="sci", ref_cube=ref_x,
                            meta={"pxscale": pxscale, "wavelength_m": wavelength})}
    d = dsets["sci"]
    print(f"sci: {d.cube.shape[0]} integrations at PA {np.unique(np.round(d.angles, 1))} deg, "
          f"RDI library {d.ref_cube.shape[0]} frames")

# %% [markdown]
# ## 3. Reducer, Space and Objective
#
# `spaceklip.make_reducer` builds a `PyKLIPReducer` per partition (one here), with λ/D from
# the filter and D = 6.5 m. The searched block is small. With four frames there is nothing to
# bin, so `make_space` pins `bin` at 1. `search_angles=False` leaves out the angular exclusion.
# `angsep` stays 0, which for pyKLIP excludes only the frames with no motion at all, the frame
# itself and its same-roll twin. What remains is the high-pass filter, `n_ang` (pyKLIP's
# `subsections`) and `k_klip` (`numbasis`).
#
# To that we add one global categorical dimension, pyKLIP's **`mode`** (`ADI`, `RDI`,
# `ADI+RDI`). Any `Param` whose name matches a backend option is passed straight to the
# engine, so the optimizer decides how to use the two rolls and the reference library. With a
# 10° roll it is a real trade-off. `ADI` has only the other roll's two frames for its basis and
# self-subtracts part of the companion, which moves about 1 FWHM between rolls. `RDI` keeps the
# most companion flux, and `ADI+RDI` gives up some throughput for whiter speckles. The last
# cell of this section compares the three at the default configuration. Carter et al. (2023)
# compare the same three modes.
#
# **The contrast axis.** HIP 65426 and the reference star are behind the mask in every
# exposure, so the star's brightness has to be imported. With a coronagraph the import has four
# terms:
#
# | term | value | from |
# |---|---|---|
# | `S` | 0.4026 Jy | synthetic photometry: Planck(8600 K) through F444W, normalized to 2MASS Ks = 6.771 |
# | units | `S / (10⁶·PIXAR_SR)` | `BUNIT = MJy/sr` and `PIXAR_SR` from the header |
# | `EE` | 0.696 at 16.5 px | the model PSF unocculted through the Lyot stop (an imaging PSF gives 0.928 and counts the stop twice) |
# | `T_optics` | 1.0 | `PHOTMJSR` for `PUPIL=MASKRND` already carries the coronagraphic optics |
#
# One term is deliberately not in `flux_unit`. The occulter's spatial transmission `T(ρ)`
# multiplies the injection inside `inject_sources` instead. Folding it into the star flux, or
# applying it twice, gets the contrast axis wrong by `1/T`.
#
# `T_optics` is 1 because STPSF's `calc_psf` normalizes at the entrance pupil and propagates
# only diffractive losses (`normalize='first'`), which is what makes the grid's measured
# transmission a real number. The model therefore lacks the COM substrate and the Lyot
# substrate, but so did every flux standard observed through them. `PHOTMJSR` for this pupil
# (2.486, against about 0.4 for CLEAR imaging) was derived in the same optical train, so the
# MJy/sr in the file already put an off-mask source at its true flux. `EE` is a fraction of the
# Lyot-stop PSF, in which the stop's own 0.18 cancels. The planet confirms it. With nothing
# tuned, HIP 65426 b measures ΔF444W = 8.796 ± 0.092, against the 8.703 ± 0.055 of Carter et al.
# (2023), a difference of 0.9σ (`scripts/check_hip65426_contrast.py`, and
# `docs/FLUX_CALIBRATION.md`).

# %%
STAR_FLUX, PSF_MODEL = None, None
if HAVE_DATA:
    try:
        # Read from the cache when the grid is there; STPSF is needed only to compute it.
        grid = stpsf_psf.offaxis_grid("NIRCam", "F444W", image_mask="MASK335R",
                                      seps_as=np.arange(0.2, 3.01, 0.2), stamp_px=41, nlambda=3)
        STAR_FLUX = stpsf_psf.star_flux_from_flux_density(
            grid, PHOT["flux_density_jy"], pixar_sr, bunit=hdr["BUNIT"],
            optics_transmission=PHOT["optics_transmission"])
        PSF_MODEL = stpsf_psf.library(grid, star_flux=STAR_FLUX)
    except (RuntimeError, ImportError) as exc:
        raise RuntimeError(
            "This tutorial needs the STPSF model of MASK335R: STPSF itself (Python >= 3.10, with its "
            "data files), or the cached stpsf_NIRCam_F444W_MASK335R_<key>.fits and "
            "eeunocc_NIRCam_F444W_<key>.fits in $KLIP_TPE_DATA/stpsf_cache.  Without it the "
            "injections have no physical scale, the calibration cannot reach S/N 5, and the search "
            f"ranks noise.\n{exc}") from exc

if HAVE_DATA:
    from klip_tpe import Param
    red = sk.make_reducer(dsets, injection_model=PSF_MODEL, mode="RDI",
                          max_workers="auto")          # pool="threads": pyKLIP forks its own workers
    space = generic.make_space(red, k_klip_max=18, search_angles=False)
    space.add(Param("mode", 0, 2, "categorical", choices=["ADI", "RDI", "ADI+RDI"], default="RDI",
                    doc="pyKLIP PSF-subtraction mode"))
    space.project = generic.make_guard(red, k_max=18, n_min_ref=4)
    objective, sampler = generic.default_config(red, known=[PLANET])
    print(space.names)

# %% [markdown]
# The STPSF model gives the injections the right shape and the mask throughput, which falls
# steeply inside 1″ (`grid` spans 0.2″ to 3.0″, the whole search annulus):

# %%
if HAVE_DATA:
    plt.figure(figsize=(9, 3.2))
    plt.subplot(1, 2, 1)
    plt.plot(grid["seps"], grid["transmission"], "-o", ms=3)
    plt.axvline(PLANET[0], color="c", ls=":", label="HIP 65426 b")
    plt.axhline(0.5, color="k", lw=.5)
    plt.xlabel("separation (arcsec)"); plt.ylabel("MASK335R throughput")
    plt.legend(); plt.grid(alpha=.3)
    plt.subplot(1, 2, 2)
    st, _, _ = PSF_MODEL.stamp(PLANET[0])
    plt.imshow(st ** 0.4, origin="lower", cmap="inferno")
    plt.title(f'off-axis PSF at {PLANET[0]}"'); plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    print(f"throughput at the planet: {PSF_MODEL.throughput(PLANET[0]):.3f}")

# %% [markdown]
# A default RDI reduction with 10 KL modes. HIP 65426 b is the point source at 0.82″, PA 150°
# (circled). Its core has three bars with six faint lobes around it, as in Carter et al.
# (2023, Fig. 3). That is what an off-axis source behind MASK335R looks like through the round
# Lyot stop. The cell after it runs the same reduction in all three modes.

# %%
if HAVE_DATA:
    cfg0 = space.decode(space.default_vector())
    res = red.reduce(ReductionRequest(params=dict(cfg0.params, inrad=6, outrad=45, k_klip=10, mode="RDI")))
    metric = MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel)
    c = (res.image.shape[0] - 1) / 2.0
    xb = c - PLANET[0] / pxscale * np.sin(np.deg2rad(PLANET[1]))
    yb = c + PLANET[0] / pxscale * np.cos(np.deg2rad(PLANET[1]))
    snr0 = float(metric.per_source(res.image, None, [PLANET[0]], [PLANET[1]])[0])
    plt.figure(figsize=(5, 5)); v = np.nanpercentile(res.image[np.isfinite(res.image)], [2, 99.7])
    plt.imshow(res.image, origin="lower", cmap="inferno", vmin=v[0], vmax=v[1])
    plt.plot(xb, yb, "o", mfc="none", mec="c", ms=20); plt.colorbar()
    plt.title(f"pyKLIP RDI, k=10   (planet S/N {snr0:.1f})")

# %%
if HAVE_DATA:                                    # the same reduction in each mode
    for m in ("ADI", "RDI", "ADI+RDI"):
        im = red.reduce(ReductionRequest(params=dict(cfg0.params, inrad=6, outrad=45, k_klip=10, mode=m))).image
        s_, det = metric.per_source(im, None, [PLANET[0]], [PLANET[1]]), None
        print(f"mode={m:8s} planet S/N {float(s_[0]):5.1f}")

# %% [markdown]
# At the default configuration `ADI` gives the planet the highest S/N, 14.1 against 12.3 with
# `RDI` and 12.8 with `ADI+RDI`.
#
# ## 4. Optimize
#
# The budget is tutorial 1's, 300 evaluations with 40 of warm-up and the six best candidates
# validated on eight fresh injection sets each, but with one injection draw per trial instead
# of three. With pyKLIP doing the reductions, this search took 11 minutes on a Mac, and the
# forward-modeled one in section 5 took 13. `n_remeasure=3` (tutorial 1, section 3) ranks the
# trials better and costs about twice as much.
#
# Each evaluation injects companions into the science frames only, so the reference library is
# never contaminated. It reduces with and without them and scores the difference. The four
# injected sources step across the band in radius, so one of them always sits near the
# separation of HIP 65426 b. `known=` keeps it 1.5 FWHM away, which is enough for the search
# score, but leaves it right next to the planet in the panel's images. The cell keeps every
# injection 4 FWHM (0.6″) clear.

# %%
if HAVE_DATA:
    sampler.excl_fwhm = 4.0      # injections >= 4 FWHM from HIP 65426 b (both searches use this sampler)
    cfg = RunConfig(ann_edges=[6, 45], n_iter=300, n_init=40, seed=5, n_sources=4,
                    validation=ValidationConfig(n_top=6, n_valid=8),
                    calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=2),
                    defaults={"k_klip": 10}, fm_curve=False)        # KLIP-FM: built-in engine only
    display = LiveDisplay(RUN_DIR, show="inline", window_scale=0.55, every=2, movie_every=10)
    runner = Runner(red, space, objective, sampler, cfg, RUN_DIR, callbacks=[display])
    t0 = time.time(); results = runner.run(); print(f"{(time.time() - t0) / 60:.1f} min")

# %%
if HAVE_DATA:
    r = results[0]
    print(f"winner: eval {r.winner_index + 1}, validated {r.validated}, injected S/N {r.winner_score:.2f}"
          f"   partitions: {r.partitions}")
    print(f"  mode = {r.winner_config['params'].get('mode')}")
    for pid, blk in r.winner_config["per_partition"].items():
        print(f"  {pid}: " + "  ".join(f"{k}={v}" for k, v in blk.items() if k in ("filter", "n_ang", "k_klip")))
    best_clean = fits.getdata(os.path.join(RUN_DIR, "annulus01", "best_clean.fits"))
    snr1 = float(metric.per_source(best_clean, None, [PLANET[0]], [PLANET[1]])[0])
    print(f"HIP 65426 b: S/N {snr0:.1f} (default)  ->  {snr1:.1f} (validated winner)")
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.4))
    for a, im, t in zip(ax, (res.image, best_clean), (f"default (S/N {snr0:.1f})", f"winner (S/N {snr1:.1f})")):
        v = np.nanpercentile(im[np.isfinite(im)], [2, 99.7])
        a.imshow(im, origin="lower", cmap="inferno", vmin=v[0], vmax=v[1]); a.set_title(t)
        a.plot(xb, yb, "o", mfc="none", mec="c", ms=20); a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()

# %% [markdown]
# The calibration was checked again after ten evaluations. The median of the five best scores
# was 7.7, above the 4 to 6 window, so the contrast went from 3.0 × 10⁻⁵ to 1.95 × 10⁻⁵ and the
# annulus restarted. Ten evaluations later the median was 3.9, below the window, and the
# contrast went to 2.5 × 10⁻⁵, so the log counts 320 evaluations in all. The best search score,
# 10.3 at evaluation 216, validated at 8.2. Evaluation 66 validated at 8.6 and is the winner.
# It uses `ADI` with 13 modes, one subsection and no high-pass filter (`filter=0`).
#
# HIP 65426 b goes from S/N 12.3 in the default reduction to 13.3. That is less than the 14.1
# `ADI` gives it at the default configuration (section 3). Only the injections were optimized.
# They sit near the detection limit, and the planet is far above it (tutorial 1, section 5).
#
# The planet stays in the frames in this tutorial. A companion this bright spreads light over
# its KLIP sector, beyond the 1.5 FWHM that `known=` excludes. For a contrast curve near it,
# subtract it before every reduction with `RunConfig(subtract_known=...)` (tutorial 4,
# section 4).

# %%
if HAVE_DATA:
    cc = np.loadtxt(os.path.join(RUN_DIR, "annulus01", "contrast_curve.txt"))
    plt.figure(figsize=(6, 4)); plt.semilogy(cc[:, 0], cc[:, 1], "-o", ms=3)
    plt.axvline(PLANET[0], color="c", ls=":", label="HIP 65426 b")
    plt.xlabel("separation (arcsec)"); plt.ylabel("S/N=5 contrast")
    plt.grid(alpha=.3); plt.legend(); plt.title("HIP 65426, NIRCam F444W");

# %% [markdown]
# ## 5. A Forward-Modeled Matched Filter
#
# KLIP does not conserve flux. It removes part of the planet and leaves negative lobes around
# the rest, by an amount that depends on the parameters being searched. `klip_tpe.fmmf.FMMFSNR`
# propagates the PSF model through each configuration's own subtraction and filters with that
# (Pueyo 2016; Ruffio et al. 2017). The Mawet small-sample ring statistics, the clean
# subtraction and the validation protocol are unchanged, so the two runs differ only in the
# filter.
#
# klip-tpe's pyKLIP backend has no analytic KLIP-FM, so the template is the numerical forward
# model `injected − clean`. That is the same response to first order, and it costs nothing,
# because the clean reduction is computed anyway. `fm_fraction` reports the fraction of filters
# that really were forward-modeled.

# %%
if HAVE_DATA:
    from klip_tpe.fmmf import FMMFSNR
    red_fm = sk.make_reducer(dsets, injection_model=PSF_MODEL,
                             mode="RDI", max_workers="auto")
    objective_fm, _ = generic.default_config(red_fm, metric="fmmf", known=[PLANET])
    metric_fm = objective_fm.metric
    cfg_fm = RunConfig(ann_edges=[6, 45], n_iter=300, n_init=40, seed=5, n_sources=4,
                       validation=ValidationConfig(n_top=6, n_valid=8),
                       calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=2),
                       defaults={"k_klip": 10}, fm_curve=False)
    run_fm = os.path.join(os.path.dirname(RUN_DIR), "hip65426_fmmf")
    runner_fm = Runner(red_fm, space, objective_fm, sampler, cfg_fm, run_fm,
                       callbacks=[LiveDisplay(run_fm, show="inline", window_scale=0.55, every=5)])
    t0 = time.time(); results_fm = runner_fm.run(); print(f"{(time.time() - t0) / 60:.1f} min")
    print(f"forward-modeled filters: {metric_fm.describe()['fm_fraction']:.0%} of "
          f"{metric_fm.describe()['n_filtered']}")

# %%
if HAVE_DATA:
    fm_clean = fits.getdata(os.path.join(run_fm, "annulus01", "best_clean.fits"))
    snr2 = float(metric.per_source(fm_clean, None, [PLANET[0]], [PLANET[1]])[0])
    print(f"HIP 65426 b, scored the same way for all three images:")
    print(f"   default k=10        S/N {snr0:5.1f}")
    print(f"   PSF matched filter  S/N {snr1:5.1f}")
    print(f"   forward-modeled MF S/N {snr2:5.1f}")
    print(f"   winner (fmmf): " + "  ".join(
        f"{k}={v}" for k, v in results_fm[0].winner_config["params"].items()
        if k in ("mode", "filter", "n_ang", "k_klip")))

# %% [markdown]
# Two thirds of the filters were forward-modeled (`fm_fraction`). The forward-modeled search
# picks `RDI` with 8 modes, two subsections and a high-pass filter (`filter=4`). HIP 65426 b
# reaches S/N 14.1, against 13.3 with the PSF-filter winner and 12.3 in the default reduction.
#
# ## 6. Notes for Real JWST Work
#
# * **Preprocessing sets the floor.** Section 1 is the minimum. spaceKLIP's `ImageTools`
#   (bad-pixel repair, background subtraction, sub-pixel alignment on the diffraction pattern,
#   frame selection) does better, and the optimizer works from whatever floor it is given.
# * **Photometry.** The STPSF model and the imported star flux of section 3 are what make the
#   injections, the calibration and the contrast curve physical. Without them the search has
#   nothing to rank.
# * **Small data sets.** With four science integrations, a single evaluation's score is noisy.
#   More injected sources per evaluation help only up to the cap that keeps enough of the noise
#   ring clean, which the runner applies to `n_sources`. `n_remeasure=3` averages fresh
#   positions and works on any annulus.
# * **What is searched** here is the number of KL modes, the high-pass filter, the azimuthal
#   subdivision and pyKLIP's `mode`, four dimensions. Any other backend option
#   (`annuli_spacing`, `algo`, `corr_smooth` and so on) becomes searchable the same way, as a
#   `Param` with that name.
# * **STPSF from the headers.** `make_reducer(dsets, psf="stpsf")` builds the same model from
#   the instrument mode, which it reads from a dataset header. The datasets here carry none, so
#   it would need `stpsf_kw=dict(filter="F444W", image_mask="MASK335R")`.
#
# ## 7. From a spaceKLIP Database
#
# After spaceKLIP has processed the data, sections 1 and 2 reduce to:
# ```python
# from spaceKLIP import database
# db = database.Database(output_dir="spaceklip/")
# db.read_jwst_s012_data(datapaths=sorted(glob.glob("spaceklip/IMGPROCESS/*_calints.fits")))
# dsets = sk.load_spaceklip(db, key="JWST_NIRCAM_NRCALONG_F444W_MASKRND_MASK335R_SUB320A335R",
#                           crop_half=55, partition_by=None)   # one Dataset; "roll" splits it
# red = sk.make_reducer(dsets, injection_model=PSF_MODEL, mode="RDI")   # section 3's STPSF model
# ```
# `make_reducer(dsets, psf_template="offset_psf_F444W.fits", star_flux=F_star)` works too.
# That template is normalized inside 2 λ/D and carries no mask throughput, so `F_star` has to
# be the star's flux in that aperture, and the contrast inside 1″ is only approximate.
