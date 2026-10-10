# %% [markdown]
# # Tutorial 4: Known Companions and Disks
#
# Optimizing by injection and recovery assumes that the field is empty apart from the fakes.
# A real companion or a bright disk breaks that assumption in three places:
#
# 1. An injected companion can land on top of the real one, and the blend scores nonsense.
# 2. The real source sits in the noise ring of the Mawet et al. (2014) small-sample S/N. The
#    noise at its separation is overestimated, every injection at that radius scores too low,
#    and the optimizer is tuned by a biased objective.
# 3. The same noise goes into the contrast curve, which gets a bump at the separation where
#    you detected something.
#
# klip-tpe takes one list, `known = [(ρ_arcsec, PA_deg), …]`, and applies it in all three
# places. This notebook shows what each does on β Pic, which has both a planet in the field
# (β Pic b at 0.45″ and PA ≈ 211°, Absil et al. 2013) and an edge-on debris disk (PA ≈ 29° and
# 209°).

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from klip_tpe import datasets
from klip_tpe.instruments import generic
from klip_tpe.reducer import ReductionRequest
from klip_tpe.metrics import MawetPeakSNR, Objective, Source
from klip_tpe.positions import PositionSampler
from klip_tpe.products import noise_profile, contrast_curve

files = datasets.fetch("naco_betapic")
inst = datasets.INSTRUMENT["naco_betapic"]
ds = generic.load_cube(files["cube"], files["angles"], psf=files["psf"], name="betapic")
# The distributed template is normalized (unit flux inside r = 2 px), so its counts are not
# beta Pictoris.  PHOTOMETRY carries VIP's published starphot for this cube in that aperture,
# and star_flux_from_aperture_photometry converts it to the template's whole-stamp
# normalization.  Without it the contrast axis is off by 7.65e5 (tutorial 1, section 2).
_p = datasets.PHOTOMETRY["naco_betapic"]
red = generic.make_reducer({"betapic": ds}, **inst, star_flux=generic.star_flux_from_aperture_photometry(
    ds.meta["psf"], _p["starphot"], _p["aperture_px"]))
space = generic.make_space(red, k_klip_max=30)
space.project = generic.make_guard(red, k_max=30)
cfg0 = space.decode(space.default_vector())
img = red.reduce(ReductionRequest(params=dict(cfg0.params, inrad=8, outrad=30, k_klip=10))).image
PX, FWHM = red.pxscale, red.fwhm
print(f"pixel scale {PX}\"/px, FWHM {FWHM:.2f} px")

# %% [markdown]
# ## 1. Where Is It?
#
# `known` uses the metric's convention: ρ in arcsec and PA in degrees east of north, with
# north up and east left after derotation. In pixels that is `x = cx − ρ/px · sin(PA)`,
# `y = cy + ρ/px · cos(PA)`, about the star at `((nx−1)/2, (ny−1)/2)`. Two helpers, and the
# position of the brightest source in the default reduction:

# %%
def to_pixels(rho_as, pa_deg, shape, pxscale):
    cx, cy = (shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0
    r = rho_as / pxscale
    return cx - r * np.sin(np.deg2rad(pa_deg)), cy + r * np.cos(np.deg2rad(pa_deg))

def to_sky(x, y, shape, pxscale):
    cx, cy = (shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0
    dx, dy = x - cx, y - cy
    return float(np.hypot(dx, dy) * pxscale), float(np.degrees(np.arctan2(-dx, dy)) % 360.0)

iy, ix = np.unravel_index(np.nanargmax(np.nan_to_num(img, nan=-np.inf)), img.shape)
rho_b, pa_b = to_sky(ix, iy, img.shape, PX)
print(f"brightest source: pixel ({ix}, {iy})  ->  rho {rho_b:.3f}\"  PA {pa_b:.1f} deg")
KNOWN = [(rho_b, pa_b)]                       # beta Pic b; add one entry per real source

plt.figure(figsize=(4.6, 4.6)); plt.imshow(img, origin="lower", cmap="inferno")
xb, yb = to_pixels(*KNOWN[0], img.shape, PX)
plt.plot(xb, yb, "o", mfc="none", mec="c", ms=18); plt.title("default reduction, beta Pic b"); plt.colorbar();

# %% [markdown]
# For a source known from the literature, type its position in. A position good to a FWHM is
# enough, because the exclusions have a radius of `excl_fwhm = 1.5` FWHM. A centroid can
# refine it, but do not fit the position on a strongly self-subtracted image.
#
# ## 2. Effect on the Noise and the Contrast Curve
#
# `noise_profile` measures σ(r) from `⌊2πr/FWHM⌋` apertures around each ring. β Pic b
# occupies one of them:

# %%
sig_raw, rr = noise_profile(img, FWHM, 8, 45)
sig_ex, _ = noise_profile(img, FWHM, 8, 45, known=KNOWN, pxscale=PX)
i = int(np.argmin(np.abs(rr * PX - KNOWN[0][0])))
print(f"sigma at the planet's separation: {sig_raw[i]:.3g} -> {sig_ex[i]:.3g} "
      f"({100 * (1 - sig_ex[i] / sig_raw[i]):.0f}% lower with the planet excluded)")

plt.figure(figsize=(6, 4))
plt.semilogy(rr * PX, sig_raw, label="all apertures")
plt.semilogy(rr * PX, sig_ex, label="known source excluded")
plt.axvline(KNOWN[0][0], color="c", ls=":", label="beta Pic b")
plt.xlabel("separation (arcsec)"); plt.ylabel("1$\\sigma$ noise"); plt.legend(); plt.grid(alpha=.3)
plt.title("the planet inflates its own noise ring");

# %% [markdown]
# The same σ sets the S/N=5 contrast curve, so an unmasked companion makes the limit look
# worse exactly where you found it. `Runner` passes the objective's `known` list to
# `contrast_curve` itself. Called by hand, it is the `known=` argument:

# %%
r_s = np.linspace(0.25, 0.75, 8)                       # pretend injection samples
s_s = np.full_like(r_s, 5.0)
cc_raw = contrast_curve(img, r_s, s_s, 5e-4, FWHM, PX, 8, 45)
cc_ex = contrast_curve(img, r_s, s_s, 5e-4, FWHM, PX, 8, 45, known=KNOWN)
plt.figure(figsize=(6, 4))
plt.semilogy(cc_raw["r_as"], cc_raw["curve"], label="all apertures")
plt.semilogy(cc_ex["r_as"], cc_ex["curve"], label="known source excluded")
plt.axvline(KNOWN[0][0], color="c", ls=":"); plt.xlabel("separation (arcsec)")
plt.ylabel("S/N=5 contrast"); plt.legend(); plt.grid(alpha=.3); plt.title("contrast curve");

# %% [markdown]
# ## 3. Effect on the Injections
#
# The position sampler redraws an injection that falls within `excl_fwhm` FWHM of a known
# source. Without the list, about one draw in ten lands on β Pic b in this annulus, and those
# evaluations score nonsense:

# %%
rng = np.random.default_rng(0)
band = (0.40, 0.50)                                    # the annulus the injections go into
def hits(known, n=4000):
    s = PositionSampler(fwhm_as=FWHM * PX, known=known)
    bad = 0
    for _ in range(n // 2):
        for src in s.sample(2, band[0], band[1], rng, 5e-4):
            dx = src.rho * np.sin(np.deg2rad(src.theta)) - KNOWN[0][0] * np.sin(np.deg2rad(KNOWN[0][1]))
            dy = src.rho * np.cos(np.deg2rad(src.theta)) - KNOWN[0][0] * np.cos(np.deg2rad(KNOWN[0][1]))
            bad += np.hypot(dx, dy) < 1.5 * FWHM * PX
    return 100.0 * bad / n
print(f"injections landing on beta Pic b: {hits([]):.1f}% without known=, {hits(KNOWN):.1f}% with it")

# %% [markdown]
# **How far is far enough** depends on what the run is for. For the search score, 1.5 FWHM is
# enough. The search score subtracts the S/N that the same reduction gives at each injection
# site without the injection, so the planet cancels. For the eye it is not enough. An
# injection 2 FWHM from a bright planet sits right next to it in every image of the live
# panel, and the panel is how a run is read. Raise the radius on the sampler for any run whose
# panels people will look at. Tutorial 1 uses 4 FWHM for β Pic b, and tutorial 2 uses 6 for
# HD 95086 b.
#
# Keep enough of the ring free. Two sources 180° apart at the planet's radius must both clear
# the radius. If it approaches the planet's separation, there is almost no room left, and after
# 30 failed draws the sampler keeps the last one, inside the exclusion.

# %%
def closest(excl, n=4000):
    s = PositionSampler(fwhm_as=FWHM * PX, known=KNOWN, excl_fwhm=excl)
    d = []
    for _ in range(n // 2):
        for src in s.sample(2, band[0], band[1], rng, 5e-4):
            dx = src.rho * np.sin(np.deg2rad(src.theta)) - KNOWN[0][0] * np.sin(np.deg2rad(KNOWN[0][1]))
            dy = src.rho * np.cos(np.deg2rad(src.theta)) - KNOWN[0][0] * np.cos(np.deg2rad(KNOWN[0][1]))
            d.append(np.hypot(dx, dy) / (FWHM * PX))
    d = np.asarray(d)
    return d.min(), 100.0 * np.mean(d < 3.0)

for excl in (1.5, 4.0):
    dmin, near = closest(excl)
    print(f"excl_fwhm = {excl}: closest injection {dmin:.1f} FWHM from beta Pic b, "
          f"{near:.1f}% of them within 3 FWHM")

# %% [markdown]
# The metric also leaves the known source out of the noise apertures when it scores an
# injection, so the score no longer depends on how far the injection happens to be from the
# planet:

# %%
m_raw = MawetPeakSNR(pxscale=PX, fwhm=FWHM, kernel_fn=red.matched_filter_kernel)
m_ex = MawetPeakSNR(pxscale=PX, fwhm=FWHM, kernel_fn=red.matched_filter_kernel, known=KNOWN)
inj = red.reduce(ReductionRequest(params=dict(cfg0.params, inrad=8, outrad=30, k_klip=10),
                                  injections=[Source(KNOWN[0][0], (KNOWN[0][1] + 90) % 360, 2e-3)])).image
pa_inj = (KNOWN[0][1] + 90) % 360
print(f"injection at rho {KNOWN[0][0]:.2f}\", PA {pa_inj:.0f} deg:")
print(f"  S/N with the planet in the noise ring : {m_raw.per_source(inj, None, [KNOWN[0][0]], [pa_inj])[0]:.2f}")
print(f"  S/N with the planet excluded          : {m_ex.per_source(inj, None, [KNOWN[0][0]], [pa_inj])[0]:.2f}")

# %% [markdown]
# ## 4. Declaring It Once, for a Whole Run
#
# `default_config(..., known=[...])` builds both the metric and the sampler with the list, and
# `Runner` forwards it to the contrast curve, so one argument covers all three uses:
#
# ```python
# objective, sampler = generic.default_config(red, known=[(0.452, 211.9)])
# Runner(red, space, objective, sampler, cfg, run_dir).run()
# ```
#
# On the command line the list is `--known rho pa`, repeated once per source:
#
# ```
# klip-tpe generic --cube ... --angles ... --known 0.452 211.9 --known <rho> <pa> ...
# ```

# %%
objective, sampler = generic.default_config(red, known=KNOWN)
print("metric knows :", objective.metric.known)
print("sampler knows:", sampler.known, f"(exclusion radius {sampler.excl_fwhm} FWHM)")

# %% [markdown]
# A bright companion can bias the run in a fourth way. Its light enters the KLIP basis, and
# the part of it beyond 1.5 FWHM (PSF lobes, the negative wings ADI leaves) still reaches the
# noise rings. On the ground the speckles dominate that light. Where the speckle floor is far
# below the companion, as on JWST, remove the companion before every reduction.
# `klip_tpe.companion.fit_negative_companion` fits its position and contrast, and
# `RunConfig(subtract_known=[(rho, pa, contrast)])` subtracts it with a negative injection.
# `companion.companion_snr` measures a companion's S/N without the search metric's biases.
#
# ## 5. Disks and Other Extended Signal
#
# A list of point sources does not describe a disk. Two more settings handle extended signal:
#
# * `PositionSampler(forbidden_pa=[(center_deg, half_width_deg), …])` keeps injections out of
#   whole position-angle sectors. That avoids an edge-on disk (β Pic's lies along PA ≈ 29° and
#   209°) or a diffraction feature.
# * `MawetPeakSNR(pixel_mask=mask)` takes a boolean image (`True` = ignore) that is left out of
#   the noise apertures, for a disk, a spider or a bad-pixel region.
#
# Both use the same geometry as everything else: north up, PA east of north.

# %%
yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
cx, cy = (img.shape[1] - 1) / 2.0, (img.shape[0] - 1) / 2.0
pa_pix = np.degrees(np.arctan2(-(xx - cx), yy - cy)) % 360.0
disk = np.minimum(np.abs(pa_pix - 29.0), np.abs(pa_pix - 209.0))          # distance to the disk axis
disk_mask = np.minimum(disk, 360.0 - disk) < 12.0                        # +-12 deg wedge
m_disk = MawetPeakSNR(pxscale=PX, fwhm=FWHM, kernel_fn=red.matched_filter_kernel,
                      known=KNOWN, pixel_mask=disk_mask)
sampler_disk = PositionSampler(fwhm_as=FWHM * PX, known=KNOWN, forbidden_pa=[(29.0, 15.0), (209.0, 15.0)])
pos = [sampler_disk.sample(2, 0.4, 0.5, rng, 5e-4) for _ in range(200)]
pas = np.array([s.theta for p in pos for s in p])
print(f"injected PAs inside the disk wedge: {100 * np.mean(np.minimum(np.abs(pas - 29) % 360, np.abs(pas - 209) % 360) < 15):.1f}%")

plt.figure(figsize=(4.6, 4.6)); plt.imshow(np.where(disk_mask, np.nan, img), origin="lower", cmap="inferno")
plt.plot(xb, yb, "o", mfc="none", mec="c", ms=18)
plt.title("disk wedge masked out of the noise"); plt.colorbar();

# %% [markdown]
# ## 6. When the Known Source Is the Point of the Run
#
# Excluding a companion keeps it from biasing the optimization. To tune the reduction for the
# companion itself, match the injections to it:
#
# * Center the injections on its separation. An annulus with an outer edge inside 1″ gets two
#   sources, both at the area-weighted mid radius √((r_in² + r_out²)/2), so choose `ann_edges`
#   to put that radius at ρ. Wider or farther annuli spread three or more sources across the
#   band.
# * Force the injection contrast to the companion's own instead of calibrating to S/N ≈ 5:
#   `CalibrationConfig(forced=[c])`, with `c` from its known Δmag.
# * Keep it in `known` all the same, so that no injection lands on it.
#
# The optimum depends on the contrast it is tuned for. The parameters that maximize the S/N of
# a companion at the detection limit are not the ones that maximize a bright companion's.
# Tutorial 1 tunes for faint injections and reports β Pic b beside them. The planet's S/N
# changes with the winner there, but the search was not aimed at it.
