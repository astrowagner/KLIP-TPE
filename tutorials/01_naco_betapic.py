# %% [markdown]
# # Tutorial 1: Optimizing an ADI Reduction of β Pictoris (VLT/NACO L′)
#
# klip-tpe searches the parameters of a KLIP reduction (KL modes, temporal binning, high-pass
# filter, azimuthal zones, reference exclusion angles, frame selection, and which nights or
# groups to combine) with a Tree-structured Parzen Estimator. It scores every trial by the S/N
# of fake companions injected at fresh random positions. It then validates the best
# candidates on independent injections, so that the winner is not a lucky draw. It is the
# Python port of the IDL optimizer written for the NEAR (VLT/VISIR) campaign, with the same
# live display and products.
#
# This notebook runs the whole pipeline on the VIP tutorial sequence of β Pictoris (NACO L′,
# 61 frames of 101×101 px, Absil et al. 2013), which contains β Pic b at 0.45″. The search in
# section 4 makes about 1,300 reductions. That is 2.5 to 5 minutes of computation on one core,
# and it took 13 minutes with the live panel on a two-core machine. Watch the panel while it
# runs. Section 4 says what to look for, and section 3 says why the budget is this size.
#
# ```
# pip install "klip-tpe[plots]"          # or: pip install -e ".[plots]" from a clone
# ```
# Everything below also runs from a terminal (`klip-tpe generic ...`, section 7).

# %%
import os, json, time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from klip_tpe import Runner, RunConfig, ValidationConfig, CalibrationConfig
from klip_tpe import datasets
from klip_tpe.instruments import generic
from klip_tpe.display import LiveDisplay

RUN_DIR = os.path.abspath("runs/betapic_naco")      # everything the run produces goes here
# Re-running this notebook RESUMES that directory: a finished annulus is reported complete
# in 0.0 min and no new panels are drawn.  Delete runs/betapic_naco to search again.

# %% [markdown]
# ## 1. The Data
#
# `datasets.fetch` downloads the three files of the VIP data set once, into
# `~/.klip_tpe/data`: the registered cube, the derotation angles and an off-axis PSF.

# %%
files = datasets.fetch("naco_betapic")
files

# %%
cube = fits.getdata(files["cube"]); angles = fits.getdata(files["angles"]); psf = fits.getdata(files["psf"])
print(cube.shape, angles.shape, psf.shape, f"field rotation {np.ptp(angles):.1f} deg")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].imshow(np.log10(np.clip(np.nanmedian(cube, 0), 1, None)), origin="lower", cmap="inferno"); ax[0].set_title("median frame (log)")
ax[1].plot(angles); ax[1].set_xlabel("frame"); ax[1].set_ylabel("derotation angle (deg)"); ax[1].set_title("parallactic angles")
ax[2].imshow(psf, origin="lower", cmap="inferno"); ax[2].set_title("off-axis PSF")
plt.tight_layout()

# %% [markdown]
# ## 2. Dataset, Reducer and Search Space
#
# The generic adapter takes any cube registered on the star. Three conventions matter.
#
# * **Angles.** klip-tpe derotates each frame counterclockwise by its angle, the same sign as
#   VIP's `angle_list` and pyKLIP's `PAs`. Pass `angle_sign=-1` for the opposite convention.
# * **Star flux.** An injected companion is given as a contrast, so an absolute contrast axis
#   needs the star's flux in the science frames' units and in the template's normalization
#   aperture. Neither file carries it. The science frames are AGPM coronagraphic, so the star
#   is occulted and cannot be read off them. The distributed `naco_betapic_psf.fits` is a
#   template normalized to a flux of 1 inside r = 2 px, so its counts are not β Pictoris
#   either. Left unset, `star_flux` falls back to the stamp's sum (4.349), and the contrast
#   axis is then off by a factor of 7.65 × 10⁵.
#
#   The number has to come from photometry made by the observers. `datasets.PHOTOMETRY`
#   carries VIP's published `starphot = 764939.6` for this cube. It was measured on the
#   non-coronagraphic PSF, rescaled to the coronagraphic integration time, in the same 2 px
#   aperture that normalizes the template. `star_flux_from_aperture_photometry` converts it to
#   the normalization `TemplatePSF` uses. With that star flux, β Pic b measures
#   ΔL′ = 7.81 ± 0.08, against the 8.01 ± 0.16 that Absil et al. (2013) published from the same
#   data (1.1σ). `scripts/check_betapic_contrast.py` makes that measurement. Run it if you
#   change anything upstream of the contrast axis.
# * **Frame-quality tags.** The frame-selection thresholds (`corr_thresh`, `noise_max`,
#   `coronoise_max`) act on per-frame tags. When the data bring none, the adapter derives them
#   from the cube (`quality_tags`).
#
# `INSTRUMENT` holds the plate scale, wavelength and aperture (λ/D = 3.5 px here). `make_space`
# scales the parameter ranges to these data. `k_klip_max=30` allows 1 to 30 KL modes. Temporal
# binning goes up to `nframes/8` = 7 frames per bin, so that at least eight bins survive as
# KLIP references. The NEAR production range of 5 to 30 frames per bin would leave this
# 61-frame cube with two bins. `bin_range=(lo, hi)` sets the range by hand.

# %%
inst = datasets.INSTRUMENT["naco_betapic"]
ds = generic.load_cube(files["cube"], files["angles"], psf=files["psf"], name="betapic")
print(f"tags: {list(ds.tags)}")

phot = datasets.PHOTOMETRY["naco_betapic"]
star_flux = generic.star_flux_from_aperture_photometry(psf, phot["starphot"], phot["aperture_px"])
print(f"template sum {psf.sum():.4f}, flux in r={phot['aperture_px']} px "
      f"{generic.aperture_sum(psf, 19, 19, phot['aperture_px']):.9f} -> star_flux {star_flux:.4g}")

red = generic.make_reducer({"betapic": ds}, star_flux=star_flux, max_workers="auto", **inst)
space = generic.make_space(red, k_klip_max=30)      # 61 frames -> <=30 KL modes, <=7 frames per temporal bin
space.project = generic.make_guard(red, k_max=30)   # feasibility: enough references per frame etc.
objective, sampler = generic.default_config(red, known=[(0.452, 211.9)])   # keep injections away from beta Pic b
print(space.names)

# %% [markdown]
# A single reduction with the default parameters takes 0.1 s and already shows β Pic b at
# 0.452″, PA 211.9°:

# %%
from klip_tpe.reducer import ReductionRequest
cfg0 = space.decode(space.default_vector())
res = red.reduce(ReductionRequest(params=dict(cfg0.params, inrad=8, outrad=22, k_klip=10)))
rho_px = 0.452 / inst["pxscale"]; pa = np.deg2rad(211.9)
xb, yb = 50 - rho_px * np.sin(pa), 50 + rho_px * np.cos(pa)
plt.figure(figsize=(4.5, 4.5)); plt.imshow(res.image, origin="lower", cmap="inferno"); plt.plot(xb, yb, "o", mfc="none", mec="c", ms=18)
plt.title("default KLIP (k=10, 8-22 px)"); plt.colorbar();

# %% [markdown]
# ## 3. Configure the Optimization, and How Big a Budget It Needs
#
# **The annulus is where the optimization happens.** Companions are injected at the
# area-weighted mid radius of the annulus, √((8² + 22²)/2) = 16.6 px = 0.45″, which is
# β Pic b's separation. The parameters are therefore tuned for the region of interest. A wider
# annulus tunes for its own mid radius, which may be far from your target.
#
# The injection contrast is calibrated so that the default configuration detects the fakes at
# S/N ≈ 5, where the parameters matter most. `CalibrationConfig(forced=[c])` fixes it instead.
#
# **Keep the injections clear of β Pic b.** `known=` stops an injection from landing within
# `excl_fwhm` = 1.5 FWHM of the planet, the IDL convention. That is enough for the search score,
# which subtracts the S/N that the same reduction gives at each injection site without the
# injection, so the planet cancels. It is not enough for the eye. An injection 2 FWHM from a
# planet this bright sits next to it in every image of the live panel. The cell below sets
# 4 FWHM (0.39″). At this radius that also keeps every injection at least 49° in position angle
# from the disk (PA 29° and 209°). Tutorial 4 shows what the radius does.
#
# ### The Budget Is Not a Free Parameter
#
# A run spends its evaluations in three phases, and each phase has a size it must reach before
# it does anything. The numbers below were measured on this data set and space, at a forced
# contrast of 3 × 10⁻⁴ and on the search score, with four 1200-evaluation and eight
# 300-evaluation searches. [docs/BUDGET.md](../docs/BUDGET.md) has the full tables and how to
# size a budget for your own data.
#
# * **Warm-up, `n_init=40`.** Random draws that give TPE something to model. TPE sorts the
#   history and fits a density to the best quarter, so `n_init=15` leaves four points to
#   describe the nine dimensions below. It came out about 0.4 S/N worse than 40, 100 or 200,
#   which were indistinguishable (two seeds each, so the difference is suggestive).
# * **Search, `n_iter=300`.** The answer improves steeply up to 100 to 120 evaluations and is
#   flat after that. A search stopped at 20 evaluations returns a configuration worth
#   S/N ≈ 6.4, against the 7.6 this search reaches. 300 clears the plateau with margin.
# * **Validation, `n_top=6, n_valid=8`.** The six best distinct configurations are re-scored on
#   eight fresh injection sets each, and the median is reported. The best score a search has
#   seen is optimistic by +1.5 S/N at this budget, because the maximum of many noisy draws is
#   high partly by luck. Across twelve runs, the configuration the search ranked first was the
#   truly best of its own top eight in none of them.
#
# `n_remeasure=3` is the other half. Scoring the same configuration twice does not give the
# same number, because the injection positions are redrawn every time. On these data that
# scatter is σ ≈ 0.87 S/N, against a landscape only 1.44 wide. Averaging three draws per trial
# divides it by √3. The +1.5 above is already with three draws, and with single draws it grows
# by the same √3, to about +2.6. Three draws cost twice what one does, because they share one
# clean reduction (four reductions per trial instead of two).
#
# Without the display, the whole run takes 2.5 to 5 minutes on one core. The cost per
# evaluation grows as the optimizer moves toward more expensive configurations, so the second
# half is slower than the first. `max_workers="auto"` above already gave the reducer every
# core. Drawing the live panel costs CPU on top of that. On one core, 60 evaluations took 43 s
# without it and 290 s with a panel for every evaluation. The cell below therefore draws every
# second evaluation. Raise `every` if the run feels slow, or set `show=False` to keep the PNGs
# without the window. When the panel cannot keep up, the display skips panels and logs a
# note. The run is unaffected, and `klip-tpe render --run-dir ...` draws every evaluation's
# panel afterwards, into `steps_rebuilt/`.

# %%
sampler.excl_fwhm = 4.0          # injections >= 4 FWHM (0.39") from beta Pic b, see above
cfg = RunConfig(ann_edges=[8, 22], n_iter=300, n_init=40, seed=1, n_remeasure=3,
                validation=ValidationConfig(n_top=6, n_valid=8),
                calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=2),
                defaults={"k_klip": 10}, fm_curve=True)

# %% [markdown]
# ## 4. Run with the Live Display
#
# **Watch the panel while this runs.** `LiveDisplay(show="auto")` draws it inline in this
# output cell inside Jupyter, and opens a matplotlib window when you run this file as a script.
# It is the only way to tell a converging search from a stuck one. Every panel it draws is
# also written to `runs/betapic_naco/steps/stepNNNN.png`, numbered by panel. From another
# terminal, `klip-tpe view --run-dir runs/betapic_naco` opens a read-only window on a run that
# is already going.
#
# The phases, and when each starts to mean something:
#
# * **Calibration**, before evaluation 1. A few reductions set the injection contrast so that
#   the default parameters land at S/N ≈ 5, and a k-scan picks the default's number of KL
#   modes. The panel shows images but no trace yet.
# * **Warm-up**, evaluations 1 to 40. Evaluation 1 is the default configuration and the rest
#   are random draws. The trace jumps around and the running best rises in occasional steps.
#   Nothing is learned yet. This is the sample TPE will model.
# * **Search**, evaluations 41 to 300. About 70% of the proposals come from TPE. The rest are
#   local moves around the five best configurations and random exploration, in about equal
#   parts. The trace tightens around good values and the S/N histogram shifts right. The
#   running best keeps rising in ever smaller steps, partly by luck (section 3). Under each
#   point a bar spans its three draws. When the bars are as tall as the gaps between points,
#   the ranking is mostly noise, which is normal.
# * **Validation**, after evaluation 300. Six candidates, eight fresh injection sets each. The
#   trace stops, the validation trials appear, and the winner is picked.
#
# The calibration is checked again after the tenth evaluation. If the median of the five best
# scores is outside the 4 to 6 window, the contrast is rescaled and the annulus restarts from
# evaluation 1. The log below shows both steps.
#
# A progress movie (`annulus01/progress.gif`) is rebuilt every 10 evaluations. The books
# (corner, importance, landscapes, products, verification) are written at the end of the
# annulus.

# %%
display = LiveDisplay(RUN_DIR, show="auto", window_scale=0.55, movie_every=10, every=2)
runner = Runner(red, space, objective, sampler, cfg, RUN_DIR, callbacks=[display])
t0 = time.time()
results = runner.run()
print(f"done in {(time.time() - t0) / 60:.1f} min")

# %% [markdown]
# In this run the calibration's k-scan picked 7 KL modes for the default configuration, which
# then scored S/N 4.9 at a contrast of 3.0 × 10⁻⁴. After ten evaluations the median of the five
# best scores was 6.4, above the window, so the contrast was lowered to 2.34 × 10⁻⁴ and the
# annulus restarted. The run made 310 evaluations in all.
#
# ## 5. Results
#
# `results` holds one `AnnulusResult` per annulus. The same numbers are in
# `final_results.json` and `annulus01/winner.json`, and `results.txt` and `results.jsonl` log
# every evaluation.
#
# **Quote the validated score, not the search best.** The two are different statistics. The
# search maximizes `s_inj − max(s_clean, 0)`, which penalizes a speckle at each injection
# site. Validation reports the raw `s_inj`. On these data the penalty is worth 0.84 S/N, so
# validation sits about 0.8 above the search scale for the same configuration, while the
# search maximum sits about 1.5 above the truth. The two offsets largely cancel, which is why
# the step from search best to validated score often looks small. Compare search scores only
# with search scores.

# %%
r = results[0]
print(f"search best: eval {r.search_best_index + 1}, S/N {r.search_best_score:.2f}")
print(f"winner:      eval {r.winner_index + 1}, validated {r.validated}, validation median S/N {r.winner_score:.2f}")
print("winner parameters:", {k: v for k, v in r.winner_config['params'].items() if k in space.names})
print("validation table (candidate, trials):")
for row in r.validation_table:
    print(f"   eval {row['eval_index'] + 1}: search S/N {row['search_score']:.2f} -> validation trials "
          f"{np.round(row['trials'], 2)} (median {row['validated_score']:.2f})")

# %% [markdown]
# The winner's reduction with the injected companions, and its clean image with β Pic b. The
# real planet is an independent check. It was never injected, and its S/N is measured with the
# same matched filter.
#
# The objective injects at the annulus mid radius at random position angles and takes the
# median, so it optimizes a configuration's average performance around the annulus. β Pic b
# sits at one position angle and is about three times brighter than the injections. It is a
# different target, so its S/N can rise or fall when the injections improve. The second cell
# below prints both, for the default configuration and the winner, and β Pic b under every
# validated candidate.

# %%
best_inj = fits.getdata(os.path.join(RUN_DIR, "annulus01", "best_inj.fits"))
best_clean = fits.getdata(os.path.join(RUN_DIR, "annulus01", "best_clean.fits"))
fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
for a, im, t in zip(ax, (best_inj, best_clean), ("winner, with injected companions", "winner, clean (beta Pic b)")):
    a.imshow(im, origin="lower", cmap="inferno"); a.set_title(t)
ax[1].plot(xb, yb, "o", mfc="none", mec="c", ms=18); plt.tight_layout()

# %%
metric = objective.metric          # carries known=[(0.452, 211.9)]; a metric built without it
                                   # would count beta Pic b as noise in its own ring
def bpic(im):
    return float(metric.per_source(im, None, [0.452], [211.9])[0])

def reduce_x(x):
    """Clean reduction of a searched vector, in the run's annulus."""
    p = dict(space.decode(np.asarray(x, float)).params, inrad=8, outrad=22)
    return red.reduce(ReductionRequest(params=p)).image

# Eval 1 is the default configuration.  After a re-calibration, the last seed row is the one
# at the run's final contrast.  Its raw S/N is the statistic validation reports (here the mean
# of three draws).
rows = [json.loads(l) for l in open(os.path.join(RUN_DIR, "results.jsonl"))]
seed = [row for row in rows if row["phase"] == "seed"][-1]
print(f"injected companions: S/N {seed['raw_score']:.1f} (default, eval 1)  ->  {r.winner_score:.1f} (validated winner)")
print(f"beta Pic b:          S/N {bpic(reduce_x(seed['x'])):.1f} (default, eval 1)  ->  {bpic(best_clean):.1f} (validated winner)")
print("\nbeta Pic b under each validated candidate:")
for row in r.validation_table:
    print(f"   eval {row['eval_index'] + 1:4d}: validated {row['validated_score']:5.2f}   "
          f"beta Pic b {bpic(reduce_x(row['x'])):5.1f}")

# %% [markdown]
# The injections went from S/N 4.2 at the default configuration to 6.5 for the validated winner
# (+55%), and β Pic b from 15.4 to 20.9 (+36%). All six candidates put β Pic b between 19.4 and
# 20.9. Only the injections were optimized. They sit near the detection limit, where the
# parameters matter most, and β Pic b is far above it. To tune for a bright known companion,
# force the injection contrast to its own with `CalibrationConfig(forced=[c])`.
#
# Section 3's 7.6 does not compare with the validated 6.5. It was measured at a contrast of
# 3 × 10⁻⁴ and on the search score, while this run calibrated to 2.34 × 10⁻⁴, and validation
# reports the raw S/N.
#
# The S/N=5 contrast curve of the validated winner, with the throughput measured from the
# injections, and the KLIP forward-model cross-check. Known companions passed as `known=` are
# left out of the noise rings, so β Pic b does not raise its own detection limit:

# %%
cc = np.loadtxt(os.path.join(RUN_DIR, "annulus01", "contrast_curve.txt"))
plt.figure(figsize=(6, 4)); plt.semilogy(cc[:, 0], cc[:, 1], "-o", ms=3, label="winner (injections)")
if r.fm_curve:
    plt.semilogy(r.fm_curve["r_as"], r.fm_curve["curve"], "--", label="KLIP-FM cross-check")
plt.xlabel("separation (arcsec)"); plt.ylabel("S/N=5 contrast"); plt.legend(); plt.grid(alpha=.3); plt.title("beta Pic, NACO L'")

# %% [markdown]
# Everything else is in the run directory:
#
# | file | content |
# |---|---|
# | `steps/stepNNNN.png`, `annulus01/progress.gif`, `opt_steps.gif` | the live panels and the movies |
# | `annulus01/corner.pdf`, `landscapes.pdf` | the sampled parameter space, colored by S/N |
# | `annulus01/importance.pdf`, `paracoord.pdf`, `rank.pdf`, `slice.pdf` | which parameters matter |
# | `annulus01/products.pdf`, `verify_limits.pdf`, `verify_subsets_inj.pdf` | winner images, S/N maps, detection limits |
# | `annulus01/validation_panel.png`, `calibration_panel.png` | the validation and calibration stages |
# | `klip_stitched*.fits`, `contrast_curve.txt`, `results.txt` | final images and tables |

# %%
from IPython.display import Image
last_frame = sorted(f for f in os.listdir(os.path.join(RUN_DIR, "steps")) if f.startswith("step"))[-1]
Image(filename=os.path.join(RUN_DIR, "steps", last_frame), width=1000)     # the annulus-end frame

# %%
sorted(os.listdir(os.path.join(RUN_DIR, "annulus01")))[:40]

# %% [markdown]
# ## 6. Other PSF-Subtraction Engines: VIP and pyKLIP
#
# The optimizer does not depend on the PSF-subtraction engine. `backend="vip"` (VIP's
# `pca_annular`) and `backend="pyklip"` (`klip_parallelized`) replace the KLIP core and keep
# the injection, frame selection, filtering, binning and scoring. The same search space can
# therefore be compared across engines on the same data. A single reduction with each:

# %%
for backend in ("klip", "vip", "pyklip"):
    try:
        red_b = generic.make_reducer({"betapic": ds}, star_flux=star_flux, backend=backend, log=lambda s: None, **inst)
    except Exception as exc:                      # engine not installed
        print(f"{backend}: {exc}"); continue
    t = time.time()
    img = red_b.reduce(ReductionRequest(params=dict(cfg0.params, inrad=8, outrad=30, k_klip=10))).image
    print(f"{backend:7s} {time.time() - t:5.1f} s   peak {np.nanmax(img):.1f} at {np.unravel_index(np.nanargmax(np.nan_to_num(img, nan=-1e9)), img.shape)[::-1]}")

# %% [markdown]
# To optimize with another engine, build the reducer with `backend=` and run the same cells,
# for example into `RUN_DIR + "_vip"`. VIP has no equivalent of `anglemax`, so that dimension
# does nothing under VIP. KLIP-FM (the forward-model cross-check curve) exists only for the
# built-in engine.
#
# ## 7. The Same from a Terminal
#
# ```
# klip-tpe generic --cube naco_betapic_cube_cen.fits --angles naco_betapic_derot_angles.fits \
#     --psf naco_betapic_psf.fits --star-flux 3.3268e6 --pxscale 0.02719 --lam 3.8e-6 --diam 8.2 \
#     --known 0.452 211.9 --ann-edges 8 22 --n-iter 300 --n-init 40 --n-remeasure 3 --k-max 30 \
#     --n-top 6 --n-valid 8 --seed 1 --display-every 2 --run-dir runs/betapic_cli --show
# klip-tpe resume --run-dir runs/betapic_cli      # after an interruption
# klip-tpe plots  --run-dir runs/betapic_cli      # regenerate the figures
# ```
#
# Run it in `~/.klip_tpe/data`, where section 1 put the files. `--show` opens the live window, and
# `--backend vip` or `--backend pyklip` swaps the engine. Two settings of this notebook have no
# flag. The default configuration starts from the space's default of 6 KL modes rather than 10
# (the calibration's k-scan then picks the default's k either way), and injections keep the
# standard 1.5 FWHM from `--known` sources rather than 4. Before the first evaluation, the
# command line also moves each searched dimension alone and warns about any that changes
# nothing (`--strict-liveness` stops the run instead, `--no-liveness-check` skips the check).
#
# ## 8. Your Own Data
#
# Anything registered on the star works. `generic.load_cube(cube, angles, psf=...,
# ref_cube=...)` accepts arrays or FITS paths, and a 4-d IFS cube with `wv_index=`. Several
# `Dataset`s (nights, epochs, IRDIS channels, pyNOMIC image groups) become partitions that the
# optimizer tunes and selects individually (tutorial 2). `ref_cube` switches on RDI and ARDI.
# For pyNOMIC reductions use the dedicated adapter (`klip-tpe near --instrument nomic`). For
# JWST see tutorial 3.
