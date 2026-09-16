# %% [markdown]
# # Tutorial 2 — two IRDIS channels as partitions: HD 95086 b (VLT/SPHERE, K1/K2)
#
# Ground-based data usually come in several pieces that could be reduced together or
# apart: nights, epochs, the two IRDIS dual-band channels, IFS channels, chop states, the
# pyNOMIC *image groups*.  klip-tpe calls each such piece a **partition**.  Every
# partition gets its own KLIP basis and its own parameter block, and the optimizer also
# searches *which* partitions to combine (the `drop1`/`drop2` dimensions) — a partition
# that hurts the combined S/N gets dropped, one that helps stays.
#
# Here the two channels of a SPHERE/IRDIS DB_K12 sequence of HD 95086 (63 frames each,
# 33.8° of field rotation, 2019-04-13) are the partitions.  HD 95086 b (0.62″, PA ≈ 145°)
# is a real, faint (ΔK ≈ 12 mag) companion — a good test that the *validated* winner
# does better than the default on a real signal rather than on injected ones only.
#
# The 241×241 px crops (0.01225″/px, star at the centre) and the unsaturated flux frames
# are fetched from the klip-tpe GitHub release (`datasets.fetch("sphere_hd95086")`).

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
# ## 1. Two partitions, one star flux per channel
#
# Injected companions are expressed as a *contrast*, so the reducer needs the star's flux
# **in the science frames' own units**.  The flux frames here were taken with DIT 0.837 s
# through the ND_1.0 filter and the science frames with DIT 96 s and no ND — but **these
# products already carry that correction**, so the star flux is the flux frame's own sum
# and nothing further.
#
# Getting this wrong is invisible to the optimizer and fatal to the contrast axis, because
# the search only ever compares S/N.  Applying `(96/0.837)/T_ND ≈ 1347` a second time put
# the injections at 3.21e-09 for S/N 4.8 while HD 95086 b sat at S/N 12.2 in the same
# image — an implied planet contrast of 8.1e-09 against a published ΔK1 = 12.2 mag, i.e.
# 1.3e-05.  The ratio is 1625: the factor, within the accuracy of reading one off the
# other.  Removing it gives 1.1e-05.  **Check any star flux against a known companion
# before you believe a contrast curve.**
#
# Each channel has its own wavelength (2.110 / 2.251 µm), hence its own λ/D and FWHM —
# `make_reducer` takes per-partition dicts.

# %%
angles = fits.getdata(files["angles"])
dsets, star_flux, lam = {}, {}, {}
# The flux frames in this distribution are ALREADY on the science frames' scale -- the
# DIT ratio and the ND transmission were applied when the products were made -- so the
# star flux is the flux frame's own sum and nothing more.  Applying
# `dit_science/dit_flux/nd_transmission` here as well over-counted the star by 1347x and
# pushed the whole contrast axis down by that factor: the injections calibrated to 3.21e-09
# for S/N 4.8 while HD 95086 b sat at S/N 12.2 in the same image, implying a planet contrast
# of 8.1e-09 against a published dK1 = 12.2 mag, i.e. 1.3e-05.  The ratio, 1625, is that
# factor.  Without it the implied contrast is 1.1e-05 -- the published value.
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
# `space.names` shows the structure: one block of nine parameters per channel
# (`bin_K1 … k_klip_K1`, `bin_K2 … k_klip_K2`) plus the two selection slots.  A default
# reduction of the 30–70 px (0.37–0.86″) annulus recovers the planet in both channels:

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
# 80 evaluations (20 warm-up), the best two validated on three fresh injection sets each,
# on the 30–70 px annulus.  On a laptop this takes ~10 minutes; most of it is the live
# panel, `display_every=2` halves that.  The contrast is calibrated to S/N ≈ 5 for the
# default configuration.

# %%
cfg = RunConfig(ann_edges=[30, 70], n_iter=80, n_init=20, seed=3,
                validation=ValidationConfig(n_top=2, n_valid=3),
                calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=2),
                defaults={"k_klip": 10}, fm_curve=True)
display = LiveDisplay(RUN_DIR, show="inline", window_scale=0.55, every=2, movie_every=10)
runner = Runner(red, space, objective, sampler, cfg, RUN_DIR, callbacks=[display])
t0 = time.time(); results = runner.run(); print(f"{(time.time() - t0) / 60:.1f} min")

# %% [markdown]
# ## 3. What did it decide?
#
# Besides the winner's parameters, the interesting outputs for partitions are: which
# channels the winner keeps, the *night effect* panel (median S/N with each partition in
# vs out), and the per-partition images in `products.pdf`.
#
# Both channels were tuned, but the selection slots decide what enters the combination:
# in this run only K1 survives — K2's speckle residuals cost the combined image more than
# its photons add at this separation.  That is the decision you would otherwise make by
# hand, made on the validated injection S/N instead.

# %%
r = results[0]
print(f"winner: eval {r.winner_index + 1}, validated {r.validated}, S/N {r.winner_score:.2f}  (search best {r.search_best_score:.2f})")
print("partitions kept:", r.partitions, " of", list(dsets))
for pid, blk in r.winner_config["per_partition"].items():                      # every block, kept or not
    keep = "kept   " if pid in r.partitions else "dropped"
    print(f"  {pid} ({keep}): " + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in blk.items() if k not in ("spat_mean", "temp_mean")))

# %%
best_clean = fits.getdata(os.path.join(RUN_DIR, "annulus01", "best_clean.fits"))
parts = fits.getdata(os.path.join(RUN_DIR, "annulus01", "best_clean_partitions.fits"))
parts = parts[None] if parts.ndim == 2 else parts                              # one kept partition -> (ny, nx)
fig, ax = plt.subplots(1, 1 + parts.shape[0], figsize=(4.5 * (1 + parts.shape[0]), 4.4), squeeze=False)
ax = ax.ravel()
ax[0].imshow(best_clean, origin="lower", cmap="inferno", **stretch(best_clean)); ax[0].set_title("winner, combined")
for a_, im, pid in zip(ax[1:], parts, r.partitions):
    a_.imshow(im, origin="lower", cmap="inferno", **stretch(im)); a_.set_title(f"winner, {pid}")
for a_ in ax:
    a_.plot(xb, yb, "o", mfc="none", mec="c", ms=20)
plt.tight_layout()

# %% [markdown]
# The planet's S/N in the default versus the optimized reduction, measured the same way
# the optimizer scores its injections (matched-filter Mawet S/N with the small-sample
# correction):

# %%
from klip_tpe.metrics import MawetPeakSNR
metric = MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel)   # no 'known' exclusion here
snr_default = metric.per_source(res.image, None, [0.62], [145.0])[0]
snr_winner = metric.per_source(best_clean, None, [0.62], [145.0])[0]
print(f"HD 95086 b: S/N {snr_default:.1f} (default k=10)  ->  {snr_winner:.1f} (validated winner)")

# %% [markdown]
# ## 4. Contrast curves and the corner plot

# %%
cc = np.loadtxt(os.path.join(RUN_DIR, "annulus01", "contrast_curve.txt"))
plt.figure(figsize=(6, 4)); plt.semilogy(cc[:, 0], cc[:, 1], "-o", ms=3, label="validated winner")
if r.fm_curve:
    plt.semilogy(r.fm_curve["r_as"], r.fm_curve["curve"], "--", label="KLIP-FM cross-check")
plt.axvline(0.62, color="c", ls=":", label="HD 95086 b"); plt.xlabel("separation (arcsec)"); plt.ylabel("5-sigma contrast")
# (the planet was passed as known= and is excluded from the noise rings, so it does not
#  inflate the limit at its own separation)
plt.legend(); plt.grid(alpha=.3); plt.title("HD 95086, IRDIS K1+K2");

# %%
from IPython.display import Image
Image(filename=os.path.join(RUN_DIR, "annulus01", "partition_map.png"), width=800)

# %% [markdown]
# `annulus01/corner.pdf`, `importance.pdf` and `landscapes.pdf` show the sampled space per
# partition; `annulus01/progress.gif` is the movie of this run.
#
# ## 5. Variations
#
# * **More partitions** – add nights or epochs as more `Dataset`s; `--max-drop` /
#   `make_space(max_drop=3)` gives the selection more slots when there are many.
# * **One shared block** – `make_space(red, per_night=False)` tunes one parameter set for
#   all partitions (fewer dimensions, faster convergence, less flexibility).
# * **RDI** – give `load_cube(..., ref_cube=...)` a reference-star cube; the reducer then
#   builds its KL basis from the references (`rdi_mode="rdi"`) or from references plus the
#   angular-excluded science frames (`"ardi"`, default).  `datasets.fetch("sphere_sao206462")`
#   has such a cube.
# * **Terminal** –
#   ```
#   klip-tpe generic --cube hd95086_irdis_K1_cube.fits hd95086_irdis_K2_cube.fits --names K1 K2 \
#       --angles hd95086_irdis_angles.fits --psf hd95086_irdis_K1_psf.fits hd95086_irdis_K2_psf.fits \
#       --star-flux 2.91e9 2.66e9 --pxscale 0.01225 --lam 2.18e-6 --diam 8.2 --known 0.62 145 \
#       --ann-edges 30 70 --n-iter 80 --n-init 20 --n-top 2 --n-valid 3 --run-dir runs/hd95086_cli --show
#   ```
