# %% [markdown]
# # Tutorial 2: Two IRDIS Channels as Partitions (HD 95086 b, VLT/SPHERE K1/K2)
#
# Ground-based data often come in pieces that can be reduced together or apart: nights,
# epochs, the two IRDIS dual-band channels, IFS channels, chop states, or pyNOMIC image
# groups. klip-tpe calls each piece a **partition**. Every partition gets its own KLIP basis
# and its own block of parameters, and the optimizer also searches which partitions to
# combine (the `drop1`/`drop2` dimensions). A partition that lowers the combined S/N is
# dropped, and one that raises it is kept.
#
# Here the partitions are the two channels of a SPHERE/IRDIS DB_K12 sequence of HD 95086
# (63 frames each, 33.8° of field rotation, 2019 April 13). HD 95086 b (0.62″, PA ≈ 145°) is a
# real companion with ΔK ≈ 12 mag, so the validated winner can be tested on a real signal and
# not only on injected ones.
#
# `datasets.fetch("sphere_hd95086")` downloads the 241×241 px crops (0.01225″/px, star at the
# center) and the unsaturated flux frames from the klip-tpe GitHub release.

# %%
import os, time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from klip_tpe import Runner, RunConfig, ValidationConfig, CalibrationConfig
from klip_tpe import datasets
from klip_tpe.instruments import generic
from klip_tpe.display import LiveDisplay
from klip_tpe.reducer import ReductionRequest

RUN_DIR = os.path.abspath("runs/hd95086_irdis")
# Re-running RESUMES this directory -- delete it to search again (see tutorial 01).
files = datasets.fetch("sphere_hd95086")
inst = datasets.INSTRUMENT["sphere_hd95086"]
files, inst

# %% [markdown]
# ## 1. Two Partitions, One Star Flux per Channel
#
# An injected companion is given as a contrast, so the reducer needs the star's flux in the
# science frames' own units. The flux frames were taken with a 0.837 s DIT through the ND_1.0
# filter, and the science frames with a 96 s DIT and no ND. These products already carry that
# correction, so the star flux is the flux frame's own sum and nothing more.
#
# HD 95086 b confirms the scale to about 20%. The ratio of its S/N to that of the injections
# implies a contrast of 1.1 × 10⁻⁵, against the 1.3 × 10⁻⁵ (ΔK1 = 12.2 ± 0.1) of Chauvin
# et al. (2018). Applying the DIT and ND factor of 1347 a second time would put the planet at
# 8 × 10⁻⁹, and the optimizer would not notice, because it compares S/N only. **Check any star
# flux against a known companion before you trust a contrast curve.**
#
# Each channel has its own wavelength (2.110 and 2.251 µm), and therefore its own λ/D and
# FWHM. `make_reducer` takes a dictionary per partition for these.

# %%
angles = fits.getdata(files["angles"])
dsets, star_flux, lam = {}, {}, {}
# The flux frames in this distribution are already on the science frames' scale (the DIT
# ratio and the ND transmission were applied when the products were made), so the star flux
# is the flux frame's own sum.  inst["dit_science"] etc. are provenance, not a correction.
for band in ("K1", "K2"):
    ds = generic.load_cube(files[f"cube_{band}"], angles, psf=files[f"psf_{band}"], name=band)
    dsets[band] = ds
    psf = ds.meta["psf"]
    star_flux[band] = float(psf[psf > 0].sum())
    lam[band] = inst["lam_m"] if band == "K1" else inst["lam_m_K2"]
    print(f"{band}: {ds.cube.shape}, PA {ds.angles.min():.1f}..{ds.angles.max():.1f} deg, star flux {star_flux[band]:.3g}")

red = generic.make_reducer(dsets, pxscale=inst["pxscale"], lam_m=lam, diam_m=inst["diam_m"], star_flux=star_flux,
                           max_workers="auto")
space = generic.make_space(red, k_klip_max=30)      # ranges scaled to the data (<=30 KL modes, <=7 frames/bin)
space.project = generic.make_guard(red, k_max=30)
objective, sampler = generic.default_config(red, known=[(0.62, 145.0)])     # keep injections off the planet
print(space.names)

# %% [markdown]
# `space.names` shows the structure: a block of nine parameters per channel (`bin_K1` to
# `k_klip_K1`, `bin_K2` to `k_klip_K2`) and the two selection slots. A default reduction of
# the 30 to 70 px annulus (0.37″ to 0.86″) recovers the planet:

# %%
cfg0 = space.decode(space.default_vector())
res = red.reduce(ReductionRequest(params=dict(cfg0.params, inrad=30, outrad=70, k_klip=10)))
c = 120; rho = 0.62 / inst["pxscale"]; pa = np.deg2rad(145.0)
xb, yb = c - rho * np.sin(pa), c + rho * np.cos(pa)
def stretch(img, lo=-1.0, hi=5.0):
    """Per-image robust stretch, the one the live dashboard uses: [lo, hi] x the image's own
    robust sigma.  A fixed count range cannot serve both the default reduction and the winner --
    a more aggressive configuration leaves residuals several times smaller, so a stretch tuned
    for the default renders the winner black."""
    from klip_tpe.display import _robust_sigma
    s_ = _robust_sigma(img)
    return dict(vmin=lo * s_, vmax=hi * s_)

plt.figure(figsize=(5, 5)); plt.imshow(res.image, origin="lower", cmap="inferno", **stretch(res.image))
plt.plot(xb, yb, "o", mfc="none", mec="c", ms=20); plt.title("K1+K2 default KLIP (k=10), HD 95086 b circled"); plt.colorbar();

# %% [markdown]
# ## 2. Optimize
#
# The budget is tutorial 1's, for the reasons given there. The search makes 300 evaluations,
# the first 40 of them warm-up (the default configuration and 39 random draws), and averages
# three injection draws per trial. The six best candidates are then validated on eight fresh
# injection sets each. The contrast is calibrated so that the default configuration scores
# S/N ≈ 5, and the calibration's k-scan picks the default's number of KL modes. The run took
# 27 minutes on a two-core machine with the live panel, which draws every second evaluation
# (`every=2`).
#
# The injections are kept **6 FWHM (0.33″) from HD 95086 b** instead of the standard 1.5. The
# standard radius is enough for the search score, which subtracts the S/N that the same
# reduction gives at each injection site without the injection. But with it, about one
# injected source in twenty lands within 3 FWHM of the planet, next to it in the panel's
# images.

# %%
sampler.excl_fwhm = 6.0          # injections >= 6 FWHM (0.33") from HD 95086 b
cfg = RunConfig(ann_edges=[30, 70], n_iter=300, n_init=40, seed=3, n_remeasure=3,
                validation=ValidationConfig(n_top=6, n_valid=8),
                calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=2),
                defaults={"k_klip": 10}, fm_curve=True)
display = LiveDisplay(RUN_DIR, show="inline", window_scale=0.55, every=2, movie_every=10)
runner = Runner(red, space, objective, sampler, cfg, RUN_DIR, callbacks=[display])
t0 = time.time(); results = runner.run(); print(f"{(time.time() - t0) / 60:.1f} min")

# %% [markdown]
# ## 3. What Did It Decide?
#
# For partitions, the outputs to look at are the channels the winner keeps, the *dataset
# effect* panel (the mean S/N of evaluations with each partition in and out), and the
# per-partition images in `products.pdf`.
#
# The winner keeps K1 alone, and so do all six validated candidates. The search settled this
# early. After the warm-up, 84% of its evaluations dropped K2, and the 20 best evaluations all
# kept K1 alone. In the *dataset effect* panel, evaluations that include K2 score 2.5 lower
# on average than those without it. The injections have the star's colors, and for such a
# source K2 adds more speckle than signal in this annulus.

# %%
r = results[0]
print(f"winner: eval {r.winner_index + 1}, validated {r.validated}, S/N {r.winner_score:.2f}  (search best {r.search_best_score:.2f})")
print("partitions kept:", r.partitions, " of", list(dsets))
for pid, blk in r.winner_config["per_partition"].items():                      # every block, kept or not
    keep = "kept   " if pid in r.partitions else "dropped"
    print(f"  {pid} ({keep}): " + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in blk.items() if k not in ("spat_mean", "temp_mean")))
print("partitions kept by each validated candidate:",
      [space.decode(np.asarray(row["x"], float)).selected for row in r.validation_table])

# %%
best_clean = fits.getdata(os.path.join(RUN_DIR, "annulus01", "best_clean.fits"))
parts = fits.getdata(os.path.join(RUN_DIR, "annulus01", "best_clean_partitions.fits"))
parts = parts[None] if parts.ndim == 2 else parts                              # (n_kept, ny, nx)
fig, ax = plt.subplots(1, 1 + parts.shape[0], figsize=(4.5 * (1 + parts.shape[0]), 4.4), squeeze=False)
ax = ax.ravel()
ax[0].imshow(best_clean, origin="lower", cmap="inferno", **stretch(best_clean)); ax[0].set_title("winner, combined")
for a_, im, pid in zip(ax[1:], parts, r.partitions):
    a_.imshow(im, origin="lower", cmap="inferno", **stretch(im)); a_.set_title(f"winner, {pid}")
for a_ in ax:
    a_.plot(xb, yb, "o", mfc="none", mec="c", ms=20)
plt.tight_layout()

# %% [markdown]
# The planet's S/N in the default and in the optimized reduction, measured with the metric
# that scores the injections (matched-filter S/N with the Mawet et al. 2014 small-sample
# correction):

# %%
from klip_tpe.metrics import MawetPeakSNR
metric = MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel)   # no 'known' exclusion here
snr_default = metric.per_source(res.image, None, [0.62], [145.0])[0]
snr_winner = metric.per_source(best_clean, None, [0.62], [145.0])[0]
print(f"HD 95086 b: S/N {snr_default:.1f} (default k=10)  ->  {snr_winner:.1f} (validated winner)")

# %% [markdown]
# On a companion this bright the metric reads low. Its spike filter clips part of the
# companion's core, and the companion's own wings land in the noise ring.
# `klip_tpe.companion.companion_snr` measures a known companion without either effect.
#
# ## 4. Contrast Curves and the Partition Map

# %%
cc = np.loadtxt(os.path.join(RUN_DIR, "annulus01", "contrast_curve.txt"))
plt.figure(figsize=(6, 4)); plt.semilogy(cc[:, 0], cc[:, 1], "-o", ms=3, label="validated winner")
if r.fm_curve:
    plt.semilogy(r.fm_curve["r_as"], r.fm_curve["curve"], "--", label="KLIP-FM cross-check")
plt.axvline(0.62, color="c", ls=":", label="HD 95086 b"); plt.xlabel("separation (arcsec)"); plt.ylabel("S/N=5 contrast")
# (the planet was passed as known= and is left out of the noise rings, so it does not
#  raise the limit at its own separation)
plt.legend(); plt.grid(alpha=.3); plt.title(f"HD 95086, IRDIS {' + '.join(r.partitions)}");

# %%
from IPython.display import Image
Image(filename=os.path.join(RUN_DIR, "annulus01", "partition_map.png"), width=800)

# %% [markdown]
# `annulus01/corner.pdf`, `importance.pdf` and `landscapes.pdf` show the sampled space per
# partition, and `annulus01/progress.gif` is the movie of this run.
#
# ## 5. Variations
#
# * **More partitions.** Add nights or epochs as more `Dataset`s. `make_space(max_drop=3)` (or
#   `--max-drop` on the command line) gives the selection more slots when there are many.
# * **One shared block.** `make_space(red, per_night=False)` tunes one parameter set for all
#   partitions: fewer dimensions and faster convergence, but less flexibility.
# * **RDI.** Give `load_cube(..., ref_cube=...)` a reference-star cube. The reducer then builds
#   its KL basis from the references (`rdi_mode="rdi"`), or from the references plus the
#   angularly excluded science frames (`"ardi"`, the default). `datasets.fetch("sphere_sao206462")`
#   has such a cube.
# * **Terminal.** From `~/.klip_tpe/data`:
#   ```
#   klip-tpe generic --cube hd95086_irdis_K1_cube.fits hd95086_irdis_K2_cube.fits --names K1 K2 \
#       --angles hd95086_irdis_angles.fits --psf hd95086_irdis_K1_psf.fits hd95086_irdis_K2_psf.fits \
#       --star-flux 2.16e6 2.47e6 --pxscale 0.01225 --lam 2.18e-6 --diam 8.2 --known 0.62 145 \
#       --ann-edges 30 70 --n-iter 300 --n-init 40 --n-remeasure 3 --k-max 30 --n-top 6 --n-valid 8 \
#       --seed 3 --display-every 2 --run-dir runs/hd95086_cli --show
#   ```
#   Three settings of this notebook have no flag. `--lam` takes one wavelength for both
#   channels (the mean, 2.18 µm, here), the default configuration starts from 6 KL modes
#   rather than 10, and injections keep the standard 1.5 FWHM from `--known` sources rather
#   than 6.
