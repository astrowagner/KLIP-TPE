# %% [markdown]
# # Tutorial 7: A Planet in a Disk, with an Integral-Field Spectrograph (MWC 758c, LBTI/ALES)
#
# ALES, the integral-field mode of LBTI/LMIRCam, splits the L band into about 100 channels at
# R ≈ 40. Each night is a `(wavelength, frame, y, x)` cube. klip-tpe reduces such a cube as
# wavelength groups, and each group is a partition with its own wavelength, FWHM and PSF
# template, the way the two IRDIS channels are in tutorial 2 (`docs/ALES.md`). Here one night
# in 20 groups makes 20 partitions, and the search decides which of them to combine.
#
# The target is MWC 758, a young star with a two-armed spiral disk. MWC 758c, the protoplanet
# that Wagner et al. (2023, Nature Astronomy 7, 1208) propose drives the spiral arms, lies just
# outside the southern arm, at 0.603″ and PA 228.5° in the 2019 November 14 data used here. It
# is faint, about 1e-5 of the star at 3.8 µm, and red.
#
# A search scores injected point sources and nothing else, so it is free to trade the disk for
# them, and many settings do exactly that. This tutorial tunes a reduction for a planet without
# losing the disk it sits in:
#
# 1. the night as 20 partitions, with injections that have c's spectrum (sections 1 and 2);
# 2. a mask that keeps the disk out of the noise and the injections (section 3);
# 3. the paper's reduction inside klip-tpe, the reference for c and for the arms (section 4);
# 4. the settings that erase the arms, which the search is therefore not given (section 5);
# 5. two searches of 3000 evaluations over the rest of the reduction, scored on c-coloured
#    sources at c's separation and close to its contrast: one on the band of the paper's Fig. 1b,
#    the other on all 20 groups with the group flags searched (sections 6 and 7);
# 6. the second search's group selection checked on common injection draws, and c group by
#    group (sections 8 and 9).
#
# `datasets.fetch("lbti_ales_mwc758", roles=...)` downloads November 14 from the klip-tpe
# GitHub release, 491 MB in 21 files: one cube per wavelength group and a meta file with the
# templates, the angles and the tables. The two disk masks are in `tutorials/data/`.

# %%
import json, os, time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage

from klip_tpe import Runner, RunConfig, ValidationConfig, CalibrationConfig
from klip_tpe import datasets
from klip_tpe.instruments import ales
from klip_tpe.display import LiveDisplay
from klip_tpe.metrics import MawetPeakSNR, Source, mawet_peak_snr, source_xy
from klip_tpe.reducer import combine_stack, set_klip_gram, set_zone_center

# The two searches of section 6 take about 15 and 20 hours on a 32-core Mac.  A finished run in
# these directories is read, not repeated, and an interrupted one is resumed (delete a directory
# to search again).  KLIP_TPE_TUT7_FIXED and KLIP_TPE_TUT7_FULL can point at finished runs of the
# same searches made elsewhere, by scripts/run_mwc758_ales_clike.py for one (section 11).
RUN_FIXED = os.path.abspath(os.environ.get("KLIP_TPE_TUT7_FIXED", "runs/mwc758_ales_nov14_g15-19"))
RUN_FULL = os.path.abspath(os.environ.get("KLIP_TPE_TUT7_FULL", "runs/mwc758_ales_nov14"))
NIGHT = "20191114"
files = datasets.fetch("lbti_ales_mwc758", roles=[f"meta_{NIGHT}"] + [f"g{g:02d}_{NIGHT}" for g in range(20)])
RHO, PA = 0.603, 228.5          # MWC 758c in the 2019 November 14 ALES data (Wagner et al. 2023)
inst = datasets.INSTRUMENT["lbti_ales_mwc758"]
inst

# %% [markdown]
# ## 1. One Night in 20 Wavelength Groups
#
# The night was reduced with K. Wagner's IDL ALES pipeline: the frames are registered on the
# star, their bad spaxels repaired, and only frames with the star at the center kept. The cube
# builder's wavelength solution gives 98 channels from 2.789 µm. The data set puts them on a
# grid of 20 groups, equally spaced in ln λ over 2.90–4.15 µm, which January 5 shares (section
# 11). Each group's cube is the mean of the channels whose wavelengths fall inside it, 2 or 3
# channels at the blue end and 6 at the red end.
#
# The 1594 frames are 2.46 s each, on 62 × 62 spaxels of an assumed 0.0345″, and they are north
# up, east left once rotated counter-clockwise by LBT_PARA + 180°. `ales.load_release` reads it
# all. The partitions are `n2_g00` to `n2_g19`, `n2` being the night's label in the two-night
# data set, so the names stay the same when January 5 is added:

# %%
dsets, lam, fwhm, table = ales.load_release(files, labels={NIGHT: "n2"})
fr = fits.getdata(files[f"meta_{NIGHT}"], "FRAMES")
print(f"{NIGHT}: {len(fr)} frames, UT {fr['UT'][0][:8]}-{fr['UT'][-1][:8]}, "
      f"LBT_PARA {fr['LBT_PARA'].min():.1f} to {fr['LBT_PARA'].max():.1f} deg")

# %% [markdown]
# The night covers 127° of rotation in 2.8 hours, alternating five minutes on the star with
# five on the sky. MWC 758 passes almost overhead, so the angle changes fastest at transit. The
# star's flux per group, measured on each partition's template within 3 spaxels of the star (the
# template's flux unit), shows the telluric absorption between 3.1 and 3.3 µm:

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ut = np.array([sum(float(v) / 60 ** i for i, v in enumerate(u.split(":"))) for u in fr["UT"]])
ax[0].plot(ut, fr["LBT_PARA"], ".", ms=2, color="C1")
rows = {r["name"]: r for r in table}
yy, xx = np.mgrid[0:62, 0:62]
rr = np.hypot(xx - 30.5, yy - 30.5)                 # spaxels from the star
pids = list(dsets)
ax[1].plot([rows[p]["lam_mean"] for p in pids], [dsets[p].meta["psf"][rr < 3.0].sum() for p in pids],
           "o-", ms=3, color="C1")
ax[0].set_xlabel("UT (h)"); ax[0].set_ylabel("LBT_PARA (deg)")
ax[1].set_xlabel("wavelength (µm)"); ax[1].set_ylabel("star within 3 spaxels (DN per frame)")
plt.tight_layout()

# %% [markdown]
# ## 2. Twenty Partitions with c's Colors
#
# `make_reducer` builds one KLIP reducer per group and combines the 20 images with equal
# weights. A search tunes for the sources it injects, and by default they have the star's
# spectrum, one contrast in every group. c is much redder than its star, so here the injections
# get c's spectrum: a power law of index 6.76 fitted to the c contrasts that the paper's IDL
# injection tests used for this night, normalized at 3.8 µm (`spectrum_norm`). A source of
# contrast C goes into each group at C (λ / 3.8 µm)^6.76, from 0.17 C at 2.92 µm to 1.73 C at
# 4.12 µm, and the contrasts the run reports are at 3.8 µm (`docs/ALES.md` section 8).
#
# The matched filter, and the FWHM the metric, the sampler and the guards use, are g17's (3.97
# µm, 2.91 spaxels), the middle of the 3.8–4.15 µm band of the paper's Fig. 1b. The KLIP zones
# are drawn as the paper's IDL drew them, one 5–27 spaxel annulus in six sectors about (30, 30)
# (`set_zone_center`), and one Gram matrix per zone (`set_klip_gram(red, "zone")`) shortens the
# per-target KLIP without changing it beyond floating-point rounding.

# %%
ALPHA = 6.7616          # c's contrast ~ (lambda / 3.8 um)^ALPHA, the fit to the paper's IDL injection contrasts (+- 1.22)
red = ales.make_reducer(dsets, lam, fwhm, table=table, max_workers="auto", representative="n2_g17",
                        spectrum=lambda um: (um / 3.8) ** ALPHA, spectrum_norm=3.8e-6,
                        spectrum_label=f"MWC 758c: contrast proportional to (lambda / 3.8 um)^{ALPHA:.2f}")
set_klip_gram(red, "zone")
set_zone_center(red, (30.0, 30.0), angles="getzone2015")     # the paper's zones (docs/ALES.md section 8)
GROUPS = [p.split("_")[1] for p in red.partitions()]          # g00 ... g19
LAM_UM = np.array([lam["n2_" + g] * 1e6 for g in GROUPS])
EDGES_AS = [0.4, 0.8]                                         # the annulus the run scores (arcsec), c in its middle
EDGES = [e / red.pxscale for e in EDGES_AS]                   # 11.6-23.2 spaxels
print(f"matched filter: n2_g17's PSF ({lam['n2_g17'] * 1e6:.3f} um), FWHM {red.fwhm:.2f} spaxels")
print("injected contrast for a unit contrast at 3.8 um:",
      ", ".join(f"{g} {red.injection_scale['n2_' + g]:.2f}" for g in GROUPS[::3]))

# %% [markdown]
# ## 3. The Disk, Kept out of the Noise and the Injections
#
# The Mawet et al. (2014) S/N compares a source with the apertures on its ring, and the arms of
# MWC 758 and the troughs that ADI digs beside them are signal, not noise. Two masks of the
# 62 × 62 frames hold them. `mwc758_disk_mask.fits` is the arms: the paper's ADI image smoothed
# by a FWHM, cut at 2 robust sigma, pieces under two beams dropped and the rest grown by a FWHM,
# with c kept clear. `mwc758_disk_mask_neg.fits` adds the
# troughs at least 2 sigma deep that come within 2 FWHM of the arms. `scripts/make_disk_mask.py`
# made both (`metrics.disk_mask_from_image`, `docs/ALES.md` section 9).
#
# `default_config(disk_mask=...)` gives one mask to both halves of the objective: it joins the
# metric's `pixel_mask`, so no noise sample is taken on it, and becomes the sampler's
# `forbidden_mask`, so no injection lands on it. `known` keeps c out of every noise ring and the
# injections at least 1.5 FWHM from it, and `forbidden_pa` keeps them out of a 50° wedge around
# its position angle. For your own data, build the mask from a disk-preserving reduction made
# before the search, never from a search's winner, and look at it: it masks whatever is bright.

# %%
ARMS = fits.getdata("data/mwc758_disk_mask.fits") != 0          # the spiral arms
DISK = fits.getdata("data/mwc758_disk_mask_neg.fits") != 0      # the arms and their ADI troughs
objective, sampler = ales.default_config(red, known=[(RHO, PA)], forbidden_pa=[(PA, 25.0)], disk_mask=DISK)
band = np.abs(rr - 0.6 / red.pxscale) <= 0.6 * red.fwhm        # the metric's radial band at c's separation
print(f"arms {ARMS.sum()} spaxels; arms and troughs {DISK.sum()} spaxels, "
      f"{100 * (DISK & band).sum() / band.sum():.1f}% of the noise band at 0.6\"")

# %% [markdown]
# ## 4. The Paper's Reduction
#
# The paper's ADI step for this night in klip-tpe's terms (`docs/ALES.md` section 8): 15-frame
# bins, an 11-spaxel high-pass, 15 KL modes, references at least 0.01 λ/D of rotation away with
# no limit on how far, no frame cuts, the paper's zones, and no second high-pass of the
# residuals. `reduce` makes one reduction of the 0.4–0.8″ annulus with given parameters in every
# group's block, combining the groups given.
#
# Two numbers follow every image from here on. c's S/N is the Mawet S/N with g17's matched filter
# and the disk out of the noise; and since one reduction choice moves the S/N of any position on
# c's ring by about as much as it moves c's, c is also placed among the 31 empty positions of its
# ring (every 10°, out of the 50° wedge, off the mask): z = (c − their mean) / their sd, and how
# many of them are as bright. The arms' flux is the mean over the arms mask of the image smoothed
# by λ/D, relative to the paper's step on the band of its Fig. 1b, and the arms' S/N divides that
# mean by the robust sigma of the smoothed image off the disk.

# %%
space0 = ales.make_space(red, bin_range=(5, 30), zone_range=(0.0, 8.0), search_post_hp=True, selection=None)


def reduce(params, groups, injections=None):
    """One reduction of the 0.4-0.8" annulus with `params` in every group's block, combining `groups`."""
    cfg = space0.decode(space0.default_vector())
    for d in [cfg.params, *cfg.per_partition.values()]:
        d.update(params); d.update(inrad=EDGES[0], outrad=EDGES[1], pad_in=False, pad_out=False)
    return red.reduce_config(cfg, injections=injections, selected=["n2_" + g for g in groups])


PAPER = dict(bin=15, filter=11, k_klip=15, n_ang=6, angsep=0.01, anglemax=127, corr_thresh=0.0, noise_max=3.0,
             coronoise_max=3.0, zone_in=EDGES[0] - 5.0, zone_out=27.0 - EDGES[1], post_hp=0)
FIG1B = [f"g{i}" for i in range(15, 20)]            # 3.8-4.15 um, the band of the paper's Fig. 1b
ADIBAND = [f"g{i:02d}" for i in range(9, 20)]       # 3.4-4.15 um, the paper's ADI band

metric = MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel, declip_nsig=None,
                      pixel_mask=DISK)
ring_metric = MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm, kernel_fn=red.matched_filter_kernel,
                           declip_nsig=None, known=[(RHO, PA)], pixel_mask=DISK)
RING = np.array([t for t in np.arange(0.0, 360.0, 10.0) if abs((t - PA + 180) % 360 - 180) > 25.0])
_x, _y = source_xy([RHO] * RING.size, RING, red.pxscale, 30.5, 30.5)
RING = RING[[not DISK[int(round(y)), int(round(x))] for x, y in zip(_x, _y)]]     # the empty positions
LOD = 3.967e-6 / 8.25 * 206264.8 / red.pxscale       # lambda/D at 3.97 um, spaxels
OFF = ~DISK & (rr >= 6) & (rr <= 22)                 # where the arms' noise is measured


def c_on_ring(img):
    """c's S/N, its z among the empty positions of its ring, and how many of them are as bright."""
    c = float(metric.per_source(img, None, [RHO], [PA])[0])
    e = np.asarray(ring_metric.per_source(img, None, [RHO] * RING.size, list(RING)), float)
    e = e[np.isfinite(e)]
    return c, float((c - e.mean()) / e.std()), int((e >= c).sum())


def smooth(img):
    fin = np.isfinite(img)
    return ndimage.gaussian_filter(np.where(fin, img, 0.0), LOD / 2.3548) / \
        np.maximum(ndimage.gaussian_filter(fin * 1.0, LOD / 2.3548), 1e-3)


def arms(img):
    """The arms' mean in the lambda/D-smoothed image, and that mean over the robust sigma off the disk."""
    sm = smooth(img)
    v = sm[OFF]
    return float(sm[ARMS].mean()), float(sm[ARMS].mean() / (1.4826 * np.median(np.abs(v - np.median(v)))))


paper = reduce(PAPER, FIG1B).image
ARM0 = arms(paper)[0]                               # the arms' flux in the paper's step on Fig. 1b's band


def report(name, img):
    c, z, n = c_on_ring(img)
    a, asn = arms(img)
    print(f"{name:38s} c S/N {c:4.2f}, z {z:+5.2f}, {n:2d} of {RING.size} as bright;  "
          f"arms {a / ARM0:4.2f} of the paper's, arm S/N {asn:4.2f}")


report("the paper's ADI step, g15-g19", paper)
report("the paper's ADI step, g09-g19", reduce(PAPER, ADIBAND).image)

# %% [markdown]
# On Fig. 1b's band c comes out at S/N 2.5, with three of the empty positions of its ring as
# bright, and at 3.6, the brightest of its ring, on the paper's ADI band. In the paper's own ASDI
# image of 3.8–4.14 µm, the source of Fig. 1b, it is at 3.35 (z +2.2) on this yardstick
# (`scripts/compare_mwc758_fig1b_clike.py`). The images, smoothed by λ/D as Fig. 1b is, with the
# mask's outline and c's position:

# %%
xc, yc = (v[0] for v in source_xy(RHO, PA, red.pxscale, 30.5, 30.5))


def show(images, ncol=4):
    nr, nc = -(-len(images) // ncol), min(len(images), ncol)
    fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.9 * nr), squeeze=False)
    for a_ in axs.flat[len(images):]:
        a_.axis("off")
    for a_, (name, im) in zip(axs.flat, images.items()):
        sm = smooth(im)
        v = sm[(rr >= 7) & (rr <= 21)]
        s_ = 1.4826 * np.median(np.abs(v - np.median(v)))
        a_.imshow(np.where((rr >= 6) & (rr <= 22), sm, np.nan), origin="lower", cmap="inferno",
                  vmin=-1.0 * s_, vmax=5.0 * s_)
        a_.contour(DISK.astype(float), levels=[0.5], colors="c", linewidths=0.6)
        a_.plot(xc, yc, "o", mfc="none", mec="w", ms=15)
        c, z, _ = c_on_ring(im)
        a_.set_title(f"{name}\nc S/N {c:.1f}, z {z:+.1f}", fontsize=9)
        a_.set_xlim(4, 57); a_.set_ylim(4, 57); a_.set_xticks([]); a_.set_yticks([])
    plt.tight_layout()


show({"the paper's ADI step, g15-g19": paper})

# %% [markdown]
# ## 5. How the Disk Gets Eaten
#
# The injections are point sources off the disk, so the objective never sees what a setting does
# to the arms: a setting that takes out more speckle near a point source scores higher even when
# it takes the arms out with it. Here is the paper's step on Fig. 1b's band with one setting
# changed at a time, scored on eight draws of two c-coloured sources at 0.6″ and contrast 2e-5,
# the same draws for every setting. Each setting is scored twice: by the run's objective (the
# clean-subtracted Mawet S/N the search maximizes) and by the throughput S/N (each source's own
# matched-filter peak in the injected-minus-clean image over the clean image's noise there, with
# no peak search, which scatters about half as much):

# %%
rng = np.random.default_rng(20261024)
DRAWS = [[Source(s.rho, s.theta, 2e-5) for s in sampler.sample(2, 0.6, 0.6, rng, 1.0)] for _ in range(8)]
m = objective.metric


def throughput(img, clean, src):
    """Each injected source's matched-filter peak in (injected - clean) over the clean image's noise there."""
    kw = dict(kernel=m.kernel(src[0].rho), search_px=m.search_px, excl_fwhm=m.excl_fwhm, known=m.known,
              declip_nsig=m.declip_nsig, band_fwhm=m.band_fwhm, min_ring=m.min_ring, penalty=m.penalty,
              inner_fwhm=m.inner_fwhm, angle_convention=m.angle_convention, return_details=True)
    rho, th = [s.rho for s in src], [s.theta for s in src]
    _, dc = mawet_peak_snr(clean, rho, th, red.pxscale, red.fwhm, pixel_mask=m.pixel_mask, **kw)
    _, dd = mawet_peak_snr(img - clean, rho, th, red.pxscale, red.fwhm, pixel_mask=None, **kw)
    return float(np.nanmedian([d2["peak"] / (d1["sigma"] * np.sqrt(1.0 + 1.0 / max(int(d1.get("nap", 3)), 3)))
                               for d1, d2 in zip(dc, dd)]))


def injections(params, groups):
    """The objective's score and the throughput S/N on each draw, and the clean image."""
    clean = reduce(params, groups).image
    sc, th = [], []
    for src in DRAWS:
        img = reduce(params, groups, injections=src).image
        sc.append(objective.score_search(img, src, clean).score)
        th.append(throughput(img, clean, src))
    return np.array(sc), np.array(th), clean


ROUTES = {"the paper's ADI step": PAPER,
          "angsep 0": dict(PAPER, angsep=0.0),
          "18 modes": dict(PAPER, k_klip=18),
          "25 modes": dict(PAPER, k_klip=25),
          "noise cut at 1.12": dict(PAPER, noise_max=1.12),
          "rotation window 98°": dict(PAPER, anglemax=98),
          "correlation cut 0.9933": dict(PAPER, corr_thresh=0.99326749)}    # the cut the searches find (section 7)
routes = {name: injections(prm, FIG1B) for name, prm in ROUTES.items()}
th0 = routes["the paper's ADI step"][1].mean()
for name, (sc, th, clean) in routes.items():
    c, z, n = c_on_ring(clean)
    a, asn = arms(clean)
    print(f"{name:24s} objective {sc.mean():4.2f}, throughput S/N {th.mean():4.2f} ({th.mean() / th0:4.2f} x);  "
          f"arms {a / ARM0:4.2f}, arm S/N {asn:4.2f};  c {c:4.2f} (z {z:+5.2f})")

# %% [markdown]
# Each of the first five lifts the injections and costs the arms. `angsep` 0 gains 21 % in
# throughput S/N for a quarter of the arms' flux, 18 modes 40 % for half of it (25 modes fall
# back to 22 % with three quarters gone), the noise cut 63 % for half, and a 98° window 14 % for
# three fifths. Each narrows the references toward the frames nearest the target in time: the
# target itself, more modes, the ends of the sequence cut, a shorter window. Those match the
# target's speckles best, so point sources gain, while an arm lying along the direction of
# rotation needs references rotated past its own length, so it loses. The correlation cut is the
# exception: it lifts the injections most, by 76 %, and keeps 0.93 of the arms at a higher arm
# S/N (0.83 against 0.58), because the frames it drops are the worst-correlated and noisiest.
# c's S/N moves between 2.3 and 3.6 with these settings.
#
# The search is not given these routes. `angsep` is searched from 0.01 λ/D (`angsep_range`), so
# no target frame is ever inside its own basis: at 0, with no limit on the rotation, the IDL's
# auto-fast rule builds one basis per zone from all the frames, the target's own included, which
# is another algorithm (`docs/ALES.md` section 10). The modes stop at 15 (`k_klip_max`). The noise
# and coronoise cuts are held at 3 times their mean, where they cut nothing, the residuals'
# second high-pass is held off, and the rotation window at the night's span of 127° (`pin`).
# Each was found by a search that had it, one after the other, and capping them in turn did not
# end: an objective that scores point sources alone trades the disk for them wherever it can.
# What is left to search is the bin, the high-pass box, `angsep` above 0.01, the modes up to 15,
# and the correlation cut, searched over 0.94–1 (`corr_range`), the range in which these frames'
# correlations lie, so that the random draws try cuts that cut. Section 8 comes back to it.
#
# The images:

# %%
show({name: im for name, (_, _, im) in routes.items()})

# %% [markdown]
# ## 6. Two Searches
#
# Both searches score two c-coloured sources per draw, three draws per evaluation, all at 0.6″,
# c's own separation (`RunConfig.inject_rho`), while the annulus, its zone and its noise rings
# stay 0.4–0.8″. The contrast is calibrated so that the seed scores S/N 4–6, but capped at 2e-5
# at 3.8 µm (`CalibrationConfig(ceiling=...)`), about twice c's contrast. The seeds score 1.0–1.3
# there, so the runs log the calibration as unreached and stay at the cap. The cap is deliberate:
# at 1e-4 the injected sources' own ADI wings raised the noise of the ring they are measured
# against by 25–120 % (more for wider high-pass boxes), so a search was paid to filter out
# structure the size of the wings, and the arms went with it. c, at about 1.1e-5, adds almost no
# wings of its own.
#
# Both search the five parameters left in section 5, seeded with the paper's step, on the
# production budget of `docs/BUDGET.md`: 3000 evaluations, the first 200 of them warm-up, and the
# six best candidates validated on eight fresh injection sets each. They differ in the groups:
#
# * the first holds the band of the paper's Fig. 1b, g15–g19, at every evaluation: only those
#   groups are loaded and there are no flags, so it has 5 dimensions;
# * the second loads all 20 and adds a 0/1 flag per group, 25 dimensions. The guard lets it switch
#   off up to 19 of them (`make_guard(max_excluded=19)`; past that, the groups nearest 3.95 µm are
#   switched back on first), and a random draw keeps each group with probability 0.5
#   (`space.p_include`), so the warm-up samples small selections as well as large ones. Its seed
#   keeps the paper's ADI band, g09–g19.
#
# `search` reads a run that has finished, without writing to it, and runs or resumes one that
# has not:

# %%
from klip_tpe.runner import AnnulusResult

PIN = {"zone_in": PAPER["zone_in"], "zone_out": PAPER["zone_out"], "n_ang": 6,    # the paper's zones
       "noise_max": 3.0, "coronoise_max": 3.0, "post_hp": 0, "anglemax": 127}    # the routes of section 5, held
SPACE = dict(bin_range=(5, 30), search_post_hp=True, pin=PIN, defaults=PAPER,
             angsep_range=(0.01, 3.0), k_klip_max=15, corr_range=(0.94, 1.0))


def run_config(**kw):
    return RunConfig(ann_edges=EDGES, n_iter=3000, n_init=200, seed=7, n_remeasure=3, inject_rho=0.6,
                     validation=ValidationConfig(n_top=6, n_valid=8), fm_curve=True,
                     calibration=CalibrationConfig(target=(4.0, 6.0), aim=5.0, n_remeasure=2, scan_k=False,
                                                   ceiling=2e-5), **kw)


def search(red_, space_, objective_, sampler_, cfg_, run_dir):
    """The results of the search in run_dir: read if it has finished (final_results.json), else run or resumed."""
    done = os.path.join(run_dir, "final_results.json")
    if os.path.exists(done):
        print(f"reading the finished run {os.path.basename(run_dir)}")
        return [AnnulusResult.from_dict(a) for a in json.load(open(done))["annuli"]]
    # a panel every 5th evaluation, a PDF of it every 100th, the running movie every 500th
    display = LiveDisplay(run_dir, show="inline", window_scale=0.55, every=5, pdf_every=100, movie_every=500)
    t0 = time.time()
    out = Runner(red_, space_, objective_, sampler_, cfg_, run_dir, callbacks=[display]).run()
    print(f"{(time.time() - t0) / 3600:.1f} h")
    return out


# Fig. 1b's band held: a reducer of those five groups, no flags
d5, lam5, fwhm5, table5 = ales.load_release(files, labels={NIGHT: "n2"}, only=FIG1B)
red5 = ales.make_reducer(d5, lam5, fwhm5, table=table5, max_workers="auto", representative="n2_g17",
                         spectrum=lambda um: (um / 3.8) ** ALPHA, spectrum_norm=3.8e-6)
set_klip_gram(red5, "zone")
set_zone_center(red5, (30.0, 30.0), angles="getzone2015")
space5 = ales.make_space(red5, selection=None, **SPACE)
space5.project = ales.make_guard(red5, max_excluded=None, prefer_m=3.95e-6)
objective5, sampler5 = ales.default_config(red5, known=[(RHO, PA)], forbidden_pa=[(PA, 25.0)], disk_mask=DISK)
fixed = search(red5, space5, objective5, sampler5, run_config(), RUN_FIXED)

# all 20 groups, the flags searched
space = ales.make_space(red, selection="binary", **SPACE)
space.project = ales.make_guard(red, max_excluded=19, prefer_m=3.95e-6)
space.p_include = 0.5
print(len(space.names), "dimensions:", [n for n in space.names if not n.startswith("sel_")], "+",
      len(space.selection_dims), "group flags")
SEED_GROUPS = {f"sel_n2_{g}": int(int(g[1:]) >= 9) for g in GROUPS}     # the paper's ADI band, g09-g19
full = search(red, space, objective, sampler, run_config(defaults=SEED_GROUPS), RUN_FULL)

# %% [markdown]
# ## 7. What Did They Decide?
#
# The winners' parameters, their scores, and the groups each keeps:

# %%
WIN = {}
for name, res in (("g15-g19 held", fixed), ("all 20 groups", full)):
    r_ = res[0]
    blk_ = dict(r_.winner_config["per_partition"]["n2_g17"])
    WIN[name] = (r_, blk_)
    el = "" if r_.election_score is None else f", {r_.election_score:.2f} in the election"
    print(f"{name}: winner evaluation {r_.winner_index + 1} of {r_.n_evaluations}; S/N {r_.winner_score:.2f}{el} "
          f"(best search score {r_.search_best_score:.2f}) at contrast {r_.contrast:.2g}")
    print("   " + "  ".join(f"{k} {blk_[k]:.4g}" for k in ("bin", "filter", "k_klip", "angsep", "corr_thresh"))
          + f";  {len(r_.partitions)} groups: {' '.join(p.split('_')[1] for p in r_.partitions)}")
r, blk = WIN["all 20 groups"]
KEPT = [p.split("_")[1] for p in r.partitions]
fig, ax = plt.subplots(figsize=(10, 1.6))
ax.imshow([[1.0 if g in KEPT else 0.0 for g in GROUPS]], cmap="Greys", vmin=-0.3, vmax=1.3, aspect="auto")
ax.set_yticks([]); ax.set_xticks(range(20), [f"{v:.2f}" for v in LAM_UM], rotation=60, fontsize=8)
ax.set_xlabel("group mean wavelength (µm)")
ax.set_title("groups the winner of the search on all 20 keeps (dark)", fontsize=10)
plt.tight_layout()

# %% [markdown]
# The two winners have the same settings: 5-frame bins, a 10-spaxel box, 15 modes, `angsep` at
# its floor and the correlation cut at 0.9933. The search on all 20 groups kept 11 of them: g00
# and g02 at the blue end, g10, and g12–g19, as did all six of its finalists and all of its 50 best
# evaluations. The g15–g19 run predates the re-scoring of a winner on fresh draws (2026-10-28), so
# its S/N is its median on the eight validation draws that elected it, and the two winners' S/Ns
# are not comparable; section 8 scores both on common draws.
#
# c and the arms in the winners' images: the g15–g19 winner, the other on all its groups and on
# its share of Fig. 1b's band (the mean of those groups' planes in `best_clean_partitions.fits`),
# next to the paper's step:

# %%
ad = os.path.join(RUN_FULL, "annulus01")
planes = fits.getdata(os.path.join(ad, "best_clean_partitions.fits"))         # one plane per kept group
images = {"the paper's ADI step, g15-g19": paper,
          "g15-g19 held: the winner": fits.getdata(os.path.join(RUN_FIXED, "annulus01", "best_clean.fits")),
          f"all 20: the winner, its {len(KEPT)} groups": fits.getdata(os.path.join(ad, "best_clean.fits"))}
if all(g in KEPT for g in FIG1B):
    images["all 20: the winner, its g15-g19"] = combine_stack(planes[[KEPT.index(g) for g in FIG1B]],
                                                              np.ones(len(FIG1B)))
for name, im in images.items():
    report(name, im)
show(images)

# %% [markdown]
# c comes out at S/N 4.0 (z +3.7) in the g15–g19 winner and at 4.5 (z +5.2) on the 11 groups of
# the other, with none of the empty positions of its ring as bright in either: the strongest c of
# any reduction of this night so far, against 3.35 (z +2.2) in the image of Fig. 1b and 2.5
# (z +1.4) at the paper's settings on the same band. The second winner's own g15–g19, at settings
# that differ from the first winner's only in the fifth decimal of the cut, give 3.9 (z +3.7). The
# arms come through: 0.93 of the paper's flux in the g15–g19 winner and 0.99 in the other, at
# arm S/Ns of 0.82 and 1.03 against the paper's 0.58. The cut drops the noisiest frames, and 11
# groups average the noise further down than 5.

# %% [markdown]
# ## 8. The Groups, Checked After the Search
#
# The two searches differ only in the groups, and the one that chose them kept six that the band
# of Fig. 1b leaves out. When did it choose them? `results.jsonl` holds every evaluation's vector,
# the five parameters and then the 20 flags. The share of the evaluations with each group on, per
# hundred, and next to it what the other settings were doing at the time:

# %%
recs = [json.loads(line) for line in open(os.path.join(RUN_FULL, "results.jsonl")) if line.strip()]
X = np.array([q["x"] for q in recs])
names = list(space.names)
flags = X[:, [names.index(f"sel_n2_{g}") for g in GROUPS]] > 0.5
same = (flags == np.array([g in KEPT for g in GROUPS])).all(axis=1)
first = int(np.argmax(same))
print(f"the winner's {len(KEPT)} groups first came up at evaluation {first + 1}, and are in {same.sum()} of the "
      f"{len(recs)} evaluations ({same[-1000:].sum()} of the last 1000)")
step = 100
lo = np.arange(0, len(recs), step)
mid = lo + step / 2
fig, ax = plt.subplots(3, 1, figsize=(10, 7.4), sharex=True, gridspec_kw={"height_ratios": [2, 1, 0.8]})
ax[0].imshow(np.array([flags[i:i + step].mean(axis=0) for i in lo]).T, origin="lower", aspect="auto",
             cmap="Greys", vmin=0, vmax=1, extent=(0, len(recs), -0.5, 19.5))
ax[0].set_yticks(range(20), [f"{g} {v:.2f}" for g, v in zip(GROUPS, LAM_UM)], fontsize=7)
ax[0].set_title("share of the evaluations with each group on, per 100 (black: all, white: none)", fontsize=10)
ax[1].plot(mid, [same[i:i + step].mean() for i in lo], label="the winner's groups")
ax[1].plot(mid, [(X[i:i + step, names.index("corr_thresh")] > 0.99).mean() for i in lo],
           label="a correlation cut at 0.99 or above")
ax[1].set_ylabel("share"); ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=8, loc="center right")
ax[2].plot(mid, [np.median(X[i:i + step, names.index("bin")]) for i in lo], color="0.3")
ax[2].set_ylabel("median bin\n(frames)"); ax[2].set_ylim(0, None); ax[2].set_xlabel("evaluation")
plt.tight_layout()

# %% [markdown]
# The winner's groups first came up at evaluation 222 and fill 2297 of the 3000 evaluations. They
# took over between evaluations 200 and 500, while the median bin was still 11–15 frames and a
# third of the evaluations or fewer cut at 0.99 or above; by evaluation 900 the median bin was 5
# and two thirds cut, but the flags had settled. With one flag per group, the TPE's model of each
# flag becomes nearly certain within a few hundred evaluations, 93–95 % on for those 11 groups
# and 5–7 % for the others, and a selection several flips away is then out of reach: no
# evaluation tried a selection inside g12–g19.
#
# The partitions are reduced independently and combined by a NaN-aware mean with equal weights
# (`reducer.combine_stack`), and `reduce_config` returns the partition images as `stack`. So one
# reduction of all 20 groups per draw scores every selection: the image of a selection is the
# mean of its planes. At the winner's settings, and the same without its correlation cut, on the
# draws of section 5:

# %%
def stacks(params):
    """All 20 groups reduced without injections and with each draw: (clean planes, [injected planes])."""
    return (np.asarray(reduce(params, GROUPS).stack),
            [np.asarray(reduce(params, GROUPS, injections=src).stack) for src in DRAWS])


def combine(planes, sel):
    return combine_stack(planes[[GROUPS.index(g) for g in sel]], np.ones(len(sel)))


def scores(st, sel, thru=True):
    """The objective (and the throughput S/N) of selection `sel` on each draw, and its clean image."""
    clean = combine(st[0], sel)
    sc = np.array([objective.score_search(combine(s, sel), src, clean).score for s, src in zip(st[1], DRAWS)])
    th = np.array([throughput(combine(s, sel), clean, src) for s, src in zip(st[1], DRAWS)]) if thru else None
    return sc, th, clean


ST = {"with its cut": stacks(blk), "without its cut": stacks(dict(blk, corr_thresh=0.0))}
SELECTIONS = {f"the search's {len(KEPT)}": KEPT, "g12-g19": [f"g{i}" for i in range(12, 20)],
              "g15-g19": FIG1B, "all 20": GROUPS}
FIXED_WINNER = injections(WIN["g15-g19 held"][1], FIG1B)      # the g15-g19 search's winner on the same draws


def row(name, sc, th, clean, th_base):
    c, z, n = c_on_ring(clean)
    d = th - th_base
    print(f"   {name:20s} objective {sc.mean():4.2f}, throughput {th.mean():4.2f} "
          f"({d.mean():+.2f} ± {d.std(ddof=1) / np.sqrt(d.size):.2f});  c {c:4.2f} (z {z:+5.2f});  "
          f"arms {arms(clean)[0] / ARM0:4.2f}")


for how, st in ST.items():
    th_base = scores(st, KEPT)[1]
    print(f"the second winner's settings {how}; throughput minus that of the search's selection:")
    for name, sel in SELECTIONS.items():
        row(name, *scores(st, sel), th_base)
    if how == "with its cut":
        row("the g15-g19 winner", *FIXED_WINNER, th_base)
from klip_tpe.klip import frame_selection_mask
kept_frames = [frame_selection_mask(len(t["corrs"]), t, blk["corr_thresh"], blk["noise_max"], blk["coronoise_max"],
                                    int(blk["bin"]), int(blk["k_klip"])).mean()
               for t in (red.frame_tags("n2_" + g) for g in GROUPS)]
print("frames the cut keeps: " + ", ".join(f"{g} {100 * v:.0f}%" for g, v in zip(GROUPS, kept_frames)))

# %% [markdown]
# With the cut, Fig. 1b's band beats the search's selection by 0.25 ± 0.03 in throughput S/N
# (2.85 against 2.59), the objective ranks them the same way (2.28 against 2.08), and g12–g19 lies
# between; the g15–g19 search's own winner does as well (2.89, +0.29 ± 0.02). So the search of
# the five groups found the better reduction for the objective both searches maximized, and the
# search of all 20 never reached it. Without the cut the order turns over: the 11 groups beat
# g15–g19 (1.72 against 1.55). The cut helps g15–g19 more (+1.30 against +0.87 for the 11 groups)
# because it drops more of their frames, a quarter against a fifth, as the last line shows. So
# the best selection depends on the other settings, and the search settled its groups under the
# early ones, where broad selections win, and did not revisit them once the cut made g15–g19
# better. A selection searched with one flag per group is worth checking this way, on common
# draws at the final settings.
#
# Greedy backward selection on the draws' mean objective, from all 20 groups down, takes out at
# each step the group whose removal scores best:

# %%
cur, path = list(GROUPS), []
while len(cur) > 1:
    s_, g_ = max((scores(ST["with its cut"], [h for h in cur if h != g], thru=False)[0].mean(), g) for g in cur)
    cur.remove(g_)
    path.append((list(cur), s_))
best_sel, best_s = max(path, key=lambda t: t[1])
print(f"best on the way down: {' '.join(best_sel)} ({len(best_sel)} groups), objective {best_s:.2f}; "
      f"the search's {scores(ST['with its cut'], KEPT, thru=False)[0].mean():.2f}")

# %% [markdown]
# Greedy selection ends at Fig. 1b's band, g15–g19. It is chosen on the eight draws it is scored
# on, so its own score is optimistic; g15–g19 was not chosen on them, and it wins on both scores
# above.

# %% [markdown]
# ## 9. c, Group by Group
#
# The same planes hold each group alone: c's place on its ring in each group's clean image, and
# the injections' throughput S/N in each group, at the winner's settings:

# %%
clean_planes, inj_planes = ST["with its cut"]
cz = np.array([c_on_ring(clean_planes[i])[1] for i in range(20)])
tp = np.array([np.median([throughput(s[i], clean_planes[i], src) for s, src in zip(inj_planes, DRAWS)])
               for i in range(20)])
fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
col = ["C0" if g in KEPT else "0.7" for g in GROUPS]
ax[0].bar(LAM_UM, cz, width=0.045, color=col); ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_ylabel("c's z on its ring")
ax[0].set_title("each group alone at the winner's settings (blue: the groups it keeps)", fontsize=10)
ax[1].bar(LAM_UM, tp, width=0.045, color=col)
ax[1].set_ylabel("injections' throughput S/N"); ax[1].set_xlabel("group mean wavelength (µm)")
plt.tight_layout()
for g, v, z_, t_ in zip(GROUPS, LAM_UM, cz, tp):
    if v > 3.6:
        print(f"{g} {v:.3f} um: c's z {z_:+5.2f}, injections {t_:4.2f}")

# %% [markdown]
# c does not follow the injections. The c-coloured sources are strongest in g14 and g16–g18
# (3.76–4.04 µm), as their spectrum says. c stands at z +2.1 to +5.8 in g12–g15 (3.63–3.83 µm)
# and below +1.3 in g16–g18. Adjacent groups share residuals, so these are not independent
# detections. Still, if c is less red than the α = 6.76 its injections assume, groups the
# objective would drop carry its signal: c is strongest on the search's 11 groups, the injections
# on g15–g19. A reduction tuned on injections is only as right as their spectrum, so where a
# companion's colors are uncertain, vary `spectrum=` before trusting the selection.

# %% [markdown]
# ## 10. Contrast Curves
#
# A run made with klip-tpe since 2026-10-30 ends with an injection grid at the winner's
# configuration: one source per reduction, each position's contrast iterated to S/N 5, the
# median over position angle with its 16–84 % band (`RunConfig.contrast_grid`, on by default,
# `AnnulusResult.grid_curve`). For a run saved before then, the curve is the fit to the validation
# injections. Both are in the run's `contrast_curve.txt`, with the KLIP-FM curve.

# %%
plt.figure(figsize=(6.5, 4.2))
for name, res in (("g15-g19 held", fixed), ("all 20 groups", full)):
    r_ = res[0]
    gc = getattr(r_, "grid_curve", None)     # the injection grid, when the run made one
    cc = gc or r_.contrast_curve             # else the fit to the validation injections
    ln = plt.semilogy(cc["r_as"], cc["curve"], "-o", ms=3, label=name + (", injection grid" if gc else ""))[0]
    if gc:
        plt.fill_between(gc["r_as"], gc["p16"], gc["p84"], alpha=0.2, color=ln.get_color())
    if r_.fm_curve:
        plt.semilogy(r_.fm_curve["r_as"], r_.fm_curve["curve"], "--", lw=0.8, color=ln.get_color(),
                     label=name + ", KLIP-FM cross-check")
plt.axvline(RHO, color="c", ls=":", label="MWC 758c")
plt.axhline(1.1e-5, color="c", ls="--", lw=0.8, label="c's contrast at 3.8 µm")
plt.xlabel("separation (arcsec)"); plt.ylabel("S/N=5 contrast at 3.8 µm")
plt.legend(fontsize=7); plt.grid(alpha=.3); plt.title("MWC 758, 2019 Nov 14, ALES, c-coloured sources");

# %% [markdown]
# The contrast is a flux ratio to the star at 3.8 µm for a source with c's spectrum, since every
# template is the night's own star. `annulus01/corner.pdf`, `importance.pdf` and
# `landscapes.pdf` show the sampled space, `partition_map.png` the flags' history of section 8,
# and `annulus01/progress.gif` is the movie of the run.

# %% [markdown]
# ## 11. Variations
#
# * **Terminal.** `scripts/run_mwc758_ales_clike.py` makes the two searches from the repository
#   root, and the notebook reads their run directories when they are the ones in `RUN_FIXED` and
#   `RUN_FULL`:
#   ```
#   python3 scripts/run_mwc758_ales_clike.py --nights nov14 --fixed-groups 15-19 --max-contrast 2e-5 \
#       --inject-rho 0.6 --disk-mask tutorials/data/mwc758_disk_mask_neg.fits --angsep-min 0.01 \
#       --k-max 15 --corr-range 0.94 1 --pin noise_max=3 --pin coronoise_max=3 --pin post_hp=0 \
#       --pin anglemax=127 --run-dir tutorials/runs/mwc758_ales_nov14_g15-19
#   python3 scripts/run_mwc758_ales_clike.py --nights nov14 --max-excluded 19 --p-include 0.5 \
#       --kernel-group g17 --max-contrast 2e-5 --inject-rho 0.6 \
#       --disk-mask tutorials/data/mwc758_disk_mask_neg.fits --angsep-min 0.01 --k-max 15 \
#       --corr-range 0.94 1 --pin noise_max=3 --pin coronoise_max=3 --pin post_hp=0 --pin anglemax=127 \
#       --run-dir tutorials/runs/mwc758_ales_nov14
#   ```
#   `--quick` makes a 12-evaluation smoke test of the same setup first. Each option is in the
#   script's help and in `docs/ALES.md` sections 8–10.
# * **Star-colored injections.** Without `spectrum=`, every group gets the same contrast, and
#   the search tunes for a companion with the star's colors. The earlier version of this tutorial
#   did that, with the default zones, no disk mask and the ALES baseline as the seed. A run of it
#   on both nights took a 5-spaxel high-pass and 12 modes and kept 10 of November 14's groups, and
#   c came out at S/N 1.9 in it (with g10's matched filter and no mask, so not on this notebook's
#   yardstick). Section 9 shows why the colors matter for the selection: the groups the injections
#   favor are not c's.
# * **Two nights.** `datasets.fetch("lbti_ales_mwc758")` downloads 2019 January 5 as well, 670
#   MB in 42 files, and `ales.load_release(files)` loads both nights as 40 partitions, `n1_g00`
#   (January 5) to `n2_g19`. January 5 covers 71° of rotation in 51 minutes around transit, and
#   November 14 has 2.7 times its frames and 4.6 times its exposure time. c's orbital motion
#   between the nights is well under a spaxel. `make_reducer(..., weighting="sqrt_texp")`
#   weights each partition by the square root of its exposure time (the script's default with
#   both nights).
# * **A third night.** `datasets.fetch("lbti_ales_mwc758_20191113")` is 2019 November 13 (959
#   frames of 2.46 s, 109° of rotation), which the paper does not use, in the same layout and
#   group grid. `ales.load_release({**files, **datasets.fetch("lbti_ales_mwc758_20191113")})`
#   loads it with November 14, the nights labelled in date order.
# * **Your own disk mask.** `metrics.disk_mask_from_image(img, fwhm_px, nsig=2, nsig_neg=2,
#   r_range=(6, 26), protect=[(RHO, PA)], pxscale=red.pxscale)` builds the mask of section 3 from
#   any disk-preserving image, and `scripts/make_disk_mask.py` does it from a FITS file with a
#   preview to look at.
# * **SDI.** `make_reducer(..., sdi=True)` adds the other groups' frames, rescaled to each
#   group's wavelength, as a searched pool (`docs/ALES.md` section 4).
