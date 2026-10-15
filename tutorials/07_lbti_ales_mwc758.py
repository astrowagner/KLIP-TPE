# %% [markdown]
# # Tutorial 7: An Integral-Field Spectrograph over Two Nights (MWC 758c, LBTI/ALES)
#
# ALES, the integral-field mode of LBTI/LMIRCam, splits the L band into about 100 channels at
# R ≈ 40. Each night is a `(wavelength, frame, y, x)` cube. klip-tpe reduces such a cube as
# wavelength groups, and each group is a partition with its own wavelength, FWHM and PSF
# template, the way the two IRDIS channels are in tutorial 2 (`docs/ALES.md`). A second night
# multiplies the partitions. Here two nights of 20 groups make 40, and the search decides which
# of them to combine.
#
# The target is MWC 758, a young star with a two-armed spiral disk, observed on 2019 November
# 13 and 14. MWC 758c, the protoplanet that Wagner et al. (2023, Nature Astronomy 7, 1208)
# propose drives the spiral arms, lies outside the southern arm. In their November 14 data it
# sits at 0.603″, PA 228.5°. It is faint and red: their November 14 spectrum reaches S/N ≈ 3
# per spectral channel at λ ≥ 3.8 µm. Here it is a known source, kept out of the injections and
# the noise rings, and a test of what the search finds.
#
# `datasets.fetch("lbti_ales_mwc758")` downloads the data set from the klip-tpe GitHub
# release, 786 MB in 42 files. For each night there is one cube per wavelength group and a
# meta file with the templates, the angles and the tables.

# %%
import os, time, warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from klip_tpe import Runner, RunConfig, ValidationConfig, CalibrationConfig
from klip_tpe import datasets
from klip_tpe.instruments import ales
from klip_tpe.display import LiveDisplay
from klip_tpe.metrics import MawetPeakSNR, source_xy
from klip_tpe.reducer import set_klip_gram

RUN_DIR = os.path.abspath("runs/mwc758_ales")
# Re-running RESUMES this directory -- delete it to search again (see tutorial 01).
files = datasets.fetch("lbti_ales_mwc758")
RHO, PA = 0.603, 228.5          # MWC 758c in the 2019 November 14 ALES data (Wagner et al. 2023)
inst = datasets.INSTRUMENT["lbti_ales_mwc758"]
inst

# %% [markdown]
# ## 1. Two Nights, One Wavelength Grid
#
# The two nights were reduced with K. Wagner's IDL ALES pipeline: the frames are registered on
# the star, their bad spaxels repaired, and only frames with the star at the center kept. Their
# cube-builder wavelength solutions differ, 100 channels from 2.750 µm on November 13 and 98 from
# 2.789 µm on November 14. The data set therefore puts both nights on one grid of 20 groups,
# equally spaced in ln λ over 2.90–4.15 µm. Each group's cube is the mean of the channels whose
# wavelengths fall inside it, 2 channels at the blue end and 6 at the red end. A group is the
# same band on both nights, while its mean wavelength follows that night's channels.
#
# The frames are 2.46 s each, on 62 × 62 spaxels of an assumed 0.0345″, and they are north up,
# east left once rotated counter-clockwise by LBT_PARA + 180°. `ales.load_release` reads it all:

# %%
dsets, lam, fwhm, table = ales.load_release(files)
for date in ales.release_nights(files):
    fr = fits.getdata(files[f"meta_{date}"], "FRAMES")
    print(f"{date}: {len(fr)} frames, UT {fr['UT'][0][:8]}-{fr['UT'][-1][:8]}, "
          f"LBT_PARA {fr['LBT_PARA'].min():.1f} to {fr['LBT_PARA'].max():.1f} deg")

# %% [markdown]
# The two nights cover 109° and 127° of rotation. November 13 has three gaps of about 20
# minutes, and November 14 alternates five minutes on the star with five on the sky. The star's
# flux per group, measured on each partition's template within 3 spaxels of the star (the
# template's flux unit), shows the telluric absorption between 3.1 and 3.3 µm:

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for date, c in zip(ales.release_nights(files), ("C0", "C1")):
    fr = fits.getdata(files[f"meta_{date}"], "FRAMES")
    ut = np.array([sum(float(v) / 60 ** i for i, v in enumerate(u.split(":"))) for u in fr["UT"]])
    ax[0].plot(ut, fr["LBT_PARA"], ".", ms=2, color=c, label=date)
rows = {r["name"]: r for r in table}
yy, xx = np.mgrid[0:62, 0:62]
core = np.hypot(yy - 30.5, xx - 30.5) < 3.0
for night, c in (("n1", "C0"), ("n2", "C1")):
    pids = [p for p in dsets if p.startswith(night)]
    ax[1].plot([rows[p]["lam_mean"] for p in pids], [dsets[p].meta["psf"][core].sum() for p in pids],
               "o-", ms=3, color=c, label=night)
ax[0].set_xlabel("UT (h)"); ax[0].set_ylabel("LBT_PARA (deg)"); ax[0].legend()
ax[1].set_xlabel("wavelength (µm)"); ax[1].set_ylabel("star within 3 spaxels (DN per frame)"); ax[1].legend()
plt.tight_layout()

# %% [markdown]
# ## 2. Forty Partitions
#
# Each partition is one night and one group, `n1_g00` to `n2_g19`. `make_reducer` builds one
# KLIP reducer per partition and combines the 40 images with equal weights. The partition at
# the mean wavelength (3.83 µm) stands for the combination: its FWHM sets the metric's apertures
# and its template the matched filter.
#
# `make_space` gives one block of reduction parameters shared by every partition, and one 0/1
# flag per partition. The guard bounds how many flags may be off. At 20 the search can drop a
# whole night, group by group, and a proposal with more partitions off has them switched back
# on, those nearest 3.7 µm first. The seed is the ALES reduction's baseline (`ales.BASELINE`):
# 5-frame bins, a 13-spaxel high-pass, 5 KL modes, references at least 0.77 λ/D of rotation
# away and within 45°.
#
# The data set keeps every 2.46 s frame, but the search starts its bins at 5 frames (12 s). On
# one thread a November 14 group between 0.41″ and 0.83″ takes 85 s to reduce at bin 1 and
# 1.3 s at bin 5, and an evaluation reduces 40 groups four times. One Gram matrix per zone (`set_klip_gram(red,
# "zone")`) shortens the per-target KLIP at these bins, and it is the same KLIP to
# floating-point rounding.

# %%
red = ales.make_reducer(dsets, lam, fwhm, table=table, max_workers="auto")
set_klip_gram(red, "zone")
space = ales.make_space(red, bin_range=(5, 30))
space.project = ales.make_guard(red, max_excluded=20)
objective, sampler = ales.default_config(red, known=[(RHO, PA)], forbidden_pa=[(PA, 25.0)])
print(len(space.names), "dimensions:", space.names[:9], "+", len(space.selection_dims), "flags")

# %% [markdown]
# `known` keeps MWC 758c out of every noise ring and the injections at least 1.5 FWHM from it,
# and `forbidden_pa` keeps them out of a 50° wedge around its position angle. The search runs
# in one annulus, 0.4–0.8″ (11.6 to 23.2 spaxels), with c in its middle. More edges in
# `EDGES_AS` add annuli, searched one after the other. A default reduction of the annulus,
# night by night and both together:

# %%
EDGES_AS = [0.4, 0.8]                               # the annulus' edges (arcsec)
EDGES = [e / red.pxscale for e in EDGES_AS]         # in spaxels: 11.6, 23.2
IC = int(np.searchsorted(EDGES_AS, RHO)) - 1        # the annulus with MWC 758c
nights = {n: [p for p in red.partitions() if p.startswith(n + "_")] for n in ("n1", "n2")}


def annulus(x, ia=IC):
    """The configuration of vector x in annulus ia, its edges padded where the runner pads them."""
    cfg = space.decode(x)
    for d in [cfg.params, *cfg.per_partition.values()]:
        d.update(inrad=EDGES[ia], outrad=EDGES[ia + 1], pad_in=ia > 0, pad_out=ia < len(EDGES) - 2)
    return cfg


metric = MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel, declip_nsig=None)
cfg0 = annulus(space.default_vector())
default = {"n1": red.reduce_config(cfg0, selected=nights["n1"]).image,
           "n2": red.reduce_config(cfg0, selected=nights["n2"]).image}
default["both"] = red.reduce_config(cfg0, selected=red.partitions()).image
xc, yc = source_xy(RHO, PA, red.pxscale, 30.5, 30.5)


def stretch(img, lo=-1.0, hi=5.0):
    """[lo, hi] x the image's own robust sigma (tutorial 2)."""
    from klip_tpe.display import _robust_sigma
    s_ = _robust_sigma(img)
    return dict(vmin=lo * s_, vmax=hi * s_)


def show(images, title):
    fig, ax = plt.subplots(1, len(images), figsize=(4.3 * len(images), 4.2))
    for a_, (name, im) in zip(np.atleast_1d(ax), images.items()):
        a_.imshow(im, origin="lower", cmap="inferno", **stretch(im))
        a_.plot(xc[0], yc[0], "o", mfc="none", mec="c", ms=18)
        snr = metric.per_source(im, None, [RHO], [PA])[0]
        a_.set_title(f"{title}, {name}: c at S/N {snr:.1f}", fontsize=10)
    plt.tight_layout()


show(default, "default")

# %% [markdown]
# ## 3. Optimize
#
# The space has 49 dimensions, the 9 shared parameters and the 40 flags, so the budget is the
# production one of `docs/BUDGET.md` rather than tutorial 1's. It is 3000 evaluations, the
# first 200 of them warm-up, each scored on three injection draws, and the six best candidates
# are validated on eight fresh injection sets each. The document's rules of thumb ask for about
# 30 evaluations and at least 4 warm-up draws per dimension. The contrast is calibrated so that
# the default configuration scores S/N ≈ 5. The injections have the star's spectrum, one
# contrast in every group, so the search optimizes for a companion with the star's colors. The
# search runs for many hours on a desktop.

# %%
cfg = RunConfig(ann_edges=EDGES, n_iter=3000, n_init=200, seed=7, n_remeasure=3,
                validation=ValidationConfig(n_top=6, n_valid=8),
                calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=2),
                fm_curve=True)
# a panel every 5th evaluation, a PDF of it every 100th, the running movie every 500th
display = LiveDisplay(RUN_DIR, show="inline", window_scale=0.55, every=5, pdf_every=100, movie_every=500)
runner = Runner(red, space, objective, sampler, cfg, RUN_DIR, callbacks=[display])
t0 = time.time(); results = runner.run(); print(f"{(time.time() - t0) / 3600:.1f} h")

# %% [markdown]
# ## 4. What Did It Decide?
#
# The winner's parameters, its validated score against the search's best, and the partitions
# it keeps, night by night:

# %%
groups = sorted({p.split("_")[1] for p in red.partitions()})
fig, axes = plt.subplots(len(results), 1, figsize=(10, 1.4 * len(results) + 1.0), sharex=True, squeeze=False)
for ia, (r, ax) in enumerate(zip(results, axes[:, 0])):
    band = f"{EDGES_AS[ia]:.1f}-{EDGES_AS[ia + 1]:.1f}″"
    print(f"annulus {ia + 1} ({band}): winner eval {r.winner_index + 1}, validated {r.validated}, "
          f"S/N {r.winner_score:.2f} (search best {r.search_best_score:.2f})")
    blk = r.winner_config["per_partition"][red.partitions()[0]]
    print("    " + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in blk.items()
                             if k not in ("spat_mean", "temp_mean", "inrad", "outrad", "pad_in", "pad_out")))
    kept = np.array([[1.0 if f"{n}_{g}" in r.partitions else 0.0 for g in groups] for n in ("n1", "n2")])
    print("    " + ", ".join(f"{n}: {int(row.sum())} of {len(groups)} groups kept" for n, row in zip(("n1", "n2"), kept)))
    ax.imshow(kept, cmap="Greys", vmin=-0.3, vmax=1.3, aspect="auto")
    ax.set_yticks([0, 1], ["Nov 13 (n1)", "Nov 14 (n2)"])
    ax.set_title(f"annulus {ia + 1} ({band}): partitions the winner keeps (dark)", fontsize=10)
ax.set_xticks(range(len(groups)), [f"{rows['n1_' + g]['lam_mean']:.2f}" for g in groups], rotation=60, fontsize=8)
ax.set_xlabel("group mean wavelength (µm)")
plt.tight_layout()

# %% [markdown]
# The winner's image, night by night and combined, with MWC 758c's S/N measured as the search
# measures the injections:

# %%
r = results[IC]
ad = os.path.join(RUN_DIR, f"annulus{IC + 1:02d}")
parts = fits.getdata(os.path.join(ad, "best_clean_partitions.fits"))
parts = parts[None] if parts.ndim == 2 else parts
winner = {}
for n in ("n1", "n2"):
    idx = [i for i, p in enumerate(r.partitions) if p.startswith(n + "_")]
    if idx:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # pixels outside the annulus are NaN everywhere
            winner[n] = np.nanmean(parts[idx], axis=0)
winner["both"] = fits.getdata(os.path.join(ad, "best_clean.fits"))
show(winner, "winner")
print(f"MWC 758c: S/N {metric.per_source(default['both'], None, [RHO], [PA])[0]:.1f} (default) -> "
      f"{metric.per_source(winner['both'], None, [RHO], [PA])[0]:.1f} (validated winner)")

# %% [markdown]
# ## 5. Contrast Curve and the Partition Map

# %%
cc = r.contrast_curve                       # also in annulus01/contrast_curve.txt, with the KLIP-FM curve after it
plt.figure(figsize=(6, 4)); plt.semilogy(cc["r_as"], cc["curve"], "-o", ms=3, label="validated winner")
if r.fm_curve:
    plt.semilogy(r.fm_curve["r_as"], r.fm_curve["curve"], "--", label="KLIP-FM cross-check")
plt.axvline(RHO, color="c", ls=":", label="MWC 758c")
plt.xlabel("separation (arcsec)"); plt.ylabel("S/N=5 contrast")
plt.legend(); plt.grid(alpha=.3); plt.title("MWC 758, ALES 2.90-4.15 µm, star-colored sources");

# %%
from IPython.display import Image
Image(filename=os.path.join(ad, "partition_map.png"), width=900)

# %% [markdown]
# The contrast is a flux ratio to the star in each group on each night, since every template
# is the night's own star. `annulus01/corner.pdf`, `importance.pdf` and `landscapes.pdf` show
# the sampled space, and `annulus01/progress.gif` is the movie of this run.
#
# ## 6. Variations
#
# * **One night.** `ales.load_release(files, nights=["n2"])` loads November 14 alone, as
#   `n2_g00` to `n2_g19`, and the same search then tunes 20 partitions.
# * **Part of the data set.** `datasets.fetch("lbti_ales_mwc758", roles=[...])` downloads only
#   the roles you name, `meta_<date>` and `gNN_<date>`, and `load_release(files, only=["g15",
#   ...])` keeps the same groups on every night.
# * **Weights.** `make_reducer(..., weighting="sqrt_texp")` weights each partition by the square
#   root of its exposure time, which favors the longer November 14 night.
# * **SDI.** `make_reducer(..., sdi=True)` adds the other groups' frames, rescaled to each
#   group's wavelength, as a searched pool (`docs/ALES.md` section 4). It needs one night, since
#   the pool would otherwise lend one night's frames to the other's groups.
# * **Terminal.**
#   ```
#   klip-tpe near --instrument ales --ales-dataset lbti_ales_mwc758 --known 0.603 228.5 \
#       --forbidden-pa 228.5 25 --ales-max-excluded 20 --bin-min 5 --klip-gram zone \
#       --ann-edges 11.594 23.188 --n-iter 3000 --n-init 200 --n-remeasure 3 --n-top 6 --n-valid 8 \
#       --seed 7 --display-every 5 --pdf-every 100 --run-dir runs/mwc758_ales_cli --show
#   ```
#   `--bin-min 5` floors the bin by projection, so the command line's random draws pile up at
#   bin 5 where this notebook's are spread over 5 to 30.
