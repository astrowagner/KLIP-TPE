# Injection templates and flux calibration

A contrast is a ratio, and the denominator has to come from somewhere. This document says,
for every observation the package ships or the paper uses, exactly what PSF is injected,
what the star flux is, over what aperture the two are normalised, and therefore what the
contrast axis means. It also names the places where a normalisation is *assumed* rather
than measured.

The optimizer never needs any of this: it compares S/N between configurations, and every
configuration sees the same injections, so a wrong star flux cancels. It matters for one
thing only — the absolute contrast axis — which is why it can be wrong for a long time
without anything looking wrong.

## The one line that sets the scale

`injection.py`, in `inject_sources`:

```python
amp = s.contrast * m.flux_unit * model.throughput(s.rho)
```

`stamp()` is normalised to unit flux **within some aperture**, `flux_unit` is the star's
flux **in that same aperture and in the science frames' units**, and `throughput(rho)` is
the coronagraph's transmission at that separation. Get the aperture pairing wrong and the
contrast axis is wrong by the ratio of encircled to total energy. Get `flux_unit` wrong and
it is wrong by that factor.

`products.contrast_curve` never sees `flux_unit`. It computes
`K(r) = SNR_inj · σ(r) / contrast_inj` and returns `nsigma · σ(r) / K_fit(r)`, so the
5σ curve inherits whatever units `flux_unit` carried. When `flux_unit = 1.0` the curve is
in raw detector units and nothing in the file says so — `runner.py` heads it
"injection-calibrated 5-sigma contrast" either way.

## Normalisation apertures in play

Four different ones, and the `star_flux` has to match whichever the model used:

| model | unit flux over |
|---|---|
| `GaussianPSF` | the whole 41×41 stamp |
| `TemplatePSF` | the whole stamp when `ee_radius_px is None`, else that circular aperture |
| `LibraryPSF` | `ee_radius_px` about the stamp centre |
| `AiryPSF`, `FramePSF` | a core aperture (`2.5σ`, `1.5 λ/D`) |

`generic.make_reducer` leaves `ee_radius_px=None`, so the generic path is whole-stamp.
`spaceklip.make_reducer` uses `2.0 λ/D` for a `psf_template`. `near.py` uses `1 λ/D`.
`stpsf_psf.offaxis_grid` uses `1.5 × EE50` of the unocculted PSF. These are not
interchangeable and nothing checks them against each other.

## Per observation

### β Pic — VLT/NACO L′ (tutorials 01 and 04; paper runs A, A2, B, B2, E, E2, F, F2)

* **Template**: `naco_betapic_psf.fits`, an off-axis PSF, as a `TemplatePSF` normalised to
  unit **total** flux (4.3491 counts in the file).
* **Star flux**: `3.3268e6`, from
  `generic.star_flux_from_aperture_photometry(psf, 764939.6, 2.0)`.
* **Throughput**: none is supplied, although the data are coronagraphic. See below.
* **Verified**: β Pic b measures **ΔL′ = 7.81 ± 0.08** against **8.01 ± 0.16** published by
  Absil et al. (2013, A&A 559, L12) *from this same sequence* — a 1.1 σ agreement.
  `scripts/check_betapic_contrast.py` is that measurement and it is the gate on this entry.

* **Why the template's own counts are not the star.** Two things about these files, both
  checked rather than assumed:
  1. The science frames are **AGPM coronagraphic, not saturated.** The median frame's
     radial profile *rises* outward from the centre — 1006, 1427, 1979, 2275 counts at
     r = 0, 1, 2, 3 px, peaking at r = 3 and falling beyond — which is an occulted core,
     not a clipped one. So the star cannot be read off the science frames.
  2. The template is **normalised**: its flux inside r = 2.000 px is 1.000000000 (to 2e-9,
     exact-aperture photometry). Its total, 4.3491, is that normalisation expressed over
     the whole stamp and says nothing about β Pictoris. The files carry no headers — no
     `EXPTIME`, no DIT, no ND keyword — so nothing in the distribution records the scale.

  Left unset, `flux_unit` becomes 4.3491 and the axis lands **9.4 × 10⁵** from a contrast:
  β Pic b then measures 586 instead of 6.25e-4. A normalised template is the easiest way to
  get an axis that looks plausible and means nothing.

* **Where the number comes from.** VIP's own metrics tutorial publishes
  `starphot = 764939.6` for this very cube, "obtained from the non-coronagraphic PSF before
  normalization and after rescaling to the integration time used in the coronagraphic
  observations" — i.e. the star's flux in the science frames' units, in the same 2 px
  aperture the template is normalised in.
  `star_flux_from_aperture_photometry` rescales it from that aperture into the whole-stamp
  normalisation `TemplatePSF` uses: `764939.6 × 4.3491 / 1.0 = 3.3268e6`.

  This is an import, and the β Pic b check above is what makes it more than an assumption:
  two independent quantities — VIP's photometry of the star and Absil's photometry of the
  planet — agree to 0.20 mag through the whole injection-and-recovery chain.

* **How the check is made.** Planet and fake are both point sources at 0.452″ reduced with
  the same default parameters, so the KLIP throughput divides out — *provided they are also
  equally bright*, because KLIP is not linear in the source (a fake 4× the planet loses a
  further 30% of itself; the script prints the curve). The contrast is therefore solved as a
  fixed point: inject, measure, re-inject at the answer. Six azimuths clear of both debris-
  disc ansae; the disc contributes ≤ 5% at the planet's position (its NE ansa peaks at 0.33
  in matched-filter units against the planet's 6.40).

* **What used to be here.** `star_flux_from_halo` scaled the template to the science
  frames' azimuthal profile over 6–14 px. That is a ratio of two different functions — an
  off-axis PSF against an AGPM-suppressed halo — with no background term, no goodness of
  fit and no uncertainty. The companion anchor measured the damage: **5.28× on run A2 and
  8.04× on run B2**, the same data and the same method, both wrong. Its answer, 9.1056e5,
  is 3.7× below the 3.3268e6 above, which is about what an AGPM removes from the halo it
  was fitted to. **It has been removed**, from `generic.py`, from the `--star-flux halo`
  CLI value and from every call site, with no opt-in.

### HD 95086 — VLT/SPHERE IRDIS DB_K12 (tutorial 02; paper runs C, G2)

* **Template**: `hd95086_irdis_{K1,K2}_psf.fits`, the unsaturated flux frames, per channel,
  as `TemplatePSF` normalised to unit total flux.
* **Star flux**: the flux frame's own sum, `float(psf[psf > 0].sum())`, **and nothing
  else**. These products already carry the DIT ratio and the ND transmission.
* **Throughput**: none is supplied, though the data are coronagraphic. The 30–70 px
  annulus is well outside the IWA, so the error is small, but it is not zero and it is not
  measured.
* **Status**: **measured, and now right to ~20 %.** Applying
  `dit_science/dit_flux/nd_transmission = 1347` a second time — which the code did until
  2026-09-15 — put the injections at 3.21e-09 for S/N 4.8 in an image where HD 95086 b
  scored 12.2, implying a companion contrast of 8.1e-09 against the published 1.32e-05
  (Chauvin et al. 2018). `collect.py` recorded `flux_scale = 9.34e-04`, i.e. 1/1071, which
  is that factor. Without it the implied contrast is 1.1e-05.
* `datasets.INSTRUMENT["sphere_hd95086"]` still carries `dit_science`, `dit_flux` and
  `nd_transmission`. They are **provenance, not a correction to apply**.
* Minor: `psf[psf > 0].sum()` excludes negative pixels while `TemplatePSF` divides by
  `t.sum()`, which includes them. A sub-percent inconsistency, but they should be the same
  sum.
* **The fix reached `hd95086_objects` on 2026-09-15 but not the constants downstream of
  it.** `run_G2` still forced `5.899e-9`, which is run C's annulus-1 calibration as written
  on 2026-09-13 — before the fix — so the harness and the reducer disagreed by the whole
  1347.0246. That contrast puts the injection peak at 2.4e-4 counts in a cube whose pixels
  reach 5.8e+02, below the float32 quantum, and the injections were rounded away before KLIP
  saw them; `reducer.py` raises the warning that says so. Now `7.946e-6`. **Run C's own
  `calibration.json` predates the fix too, so run C has to be redone** — annulus 1 moves from
  5.8989e-9 to 7.9459e-6 and annulus 2 from 4.6035e-9 to 6.2011e-6.

### HIP 65426 — JWST/NIRCam F444W (tutorial 03; paper runs D, H2)

* **Template**: `stpsf_psf.library(grid)` — the STPSF off-axis PSF of MASK335R on a ladder
  of separations, as a `LibraryPSF` normalised to unit flux inside `ee_radius_px = 16.5 px`,
  with the grid's measured mask throughput as `throughput(ρ)`.
* **Star flux**: `1.707e6` (MJy/sr summed inside that radius), from
  `stpsf_psf.star_flux_from_flux_density(grid, 0.40259, PIXAR_SR, optics_transmission=0.561)`.
* **Status**: **the chain is complete and closes on HIP 65426 b — but its last term is
  anchored on that companion, so it is a calibration of this mode, not an independent
  check.** Everything else in the chain is independent and is what the check exercises.
* **Runs D and H2 now build this model** (`run_demos.hip65426_objects`), and raise rather
  than fall back if STPSF is missing — the old silent `GaussianPSF(1.028 λ/D, star_flux=1.0)`
  is what put run D's axis in raw detector units (`flux_scale = 2.35e+05`, a companion
  "contrast" of 77; `run_H2`'s forced calibration contrast of `5.270e1` is the same tell —
  a contrast of 52.7 is not a contrast). **The runs themselves still have to be redone**:
  both the flux scale and the star centre changed.

There is no off-axis stellar image anywhere in ERS 1386 — HIP 65426 and the reference star
φ Cen are both behind MASK335R in every exposure — so the star has to be imported, and with
a coronagraph that import has four terms that are easy to confuse:

| term | value | where it comes from | what gets it wrong |
|---|---|---|---|
| `S` | 0.40259 Jy | synthetic photometry: Planck(8600 K) through the F444W bandpass, normalised to 2MASS Ks = 6.771 | ±3%, of which 1.9% is Ks and <2% the system-response shape |
| units | `S / (10⁶·PIXAR_SR)` = 4.372e6 | `BUNIT = MJy/sr`, `PIXAR_SR` from the SCI header | using `PIXAR_A2`, or forgetting the 10⁶ |
| `EE` | 0.6960 at 16.5 px | `unocculted_ee` — the model PSF **unocculted through the Lyot stop** | using an *imaging* PSF (EE 0.928 — counts the Lyot stop twice), or a stamp-sized field (the EE is still climbing at 5″) |
| `T_optics` | 0.561 | see below | leaving it at 1.0 — a factor of two |

and one term that is deliberately **not** in `flux_unit` at all: the occulter's spatial
transmission `T(ρ) = 0.778` at 0.826″, which multiplies it inside `inject_sources`. Folding
it into the star flux, or applying it twice, is the classic coronagraphic error.

* **The term nobody supplies.** STPSF's `calc_psf` defaults to `normalize='first'`, which
  normalises at the *entrance pupil* and then propagates "ignoring any reflective or
  transmissive losses from mirrors or filters … and calculates only the diffractive losses
  from slits and stops" (Perrin, `webbpsf#112`). That default is exactly what makes the
  grid's `transmission` a real measurement of the occulter — `normalize='last'` would
  renormalise every slice and report `T ≈ 1` everywhere — but it also means the model
  carries **only** the Lyot stop's diffractive loss. Measured from the model: 0.187 of the
  entrance pupil, matching JDox's "each Lyot stop has a throughput of ~20%". The
  *transmissive* losses — the COM sapphire substrate and its AR coating, which every
  coronagraphic beam crosses and which the NIRCam filter curves explicitly exclude, plus the
  BaF₂ Lyot substrate — are in neither the model nor the data: the NIRCam `photom`
  reference file has no column for the occulting mask at all
  ([spacetelescope/jwst#10309](https://github.com/spacetelescope/jwst/issues/10309)), so the
  `PHOTMJSR` applied to NRC_CORON data cannot be mask-specific.
* **How 0.561 was obtained, and what it is worth.** It is what puts HIP 65426 b at Carter et
  al. (2023)'s ΔF444W = 8.693. JDox brackets it independently: "the combined loss of light
  from the coronagraphic optics at distances > 1″ … is ~86–90%", i.e. a combined throughput
  of 0.10–0.14, which after removing the model's own 0.187 leaves 0.53–0.75. 0.561 sits
  inside that. It is a property of the mode, not of the target, so it transfers to other
  F444W/MASK335R programmes — but until it is replaced by the tabulated COM transmission
  (JDox "NIRCam Filters for Coronagraphy", or `webbpsf_ext`'s COM throughput) the HIP 65426 b
  comparison is a calibration and not a test.
* **The star is not at CRPIX.** `CRPIX` is the aperture reference point — identical in every
  file of the programme, dithers included — and misses HIP 65426 by **1.48 px**, which puts
  the companion 1.5 px inside its own separation and mismatches its KLIP throughput against
  the fakes injected to calibrate it. Two other routes failed: there is no off-axis stellar
  image to centroid, and a 180° symmetry fit to the coronagraphic residual moved the centre
  by 2 px between the two rolls of these very data. What works is the companion itself:
  derotation about a centre wrong by `δ` puts it at `u + R(PA_k)·δ` in roll `k`, so each roll
  gives `δ = R(−PA_k)·(measured − expected)` independently. The two rolls agree to 0.71 px.
  `load_calints(..., star_center=)` takes the answer; the proper source is spaceKLIP's own
  star-centring step (`STARCENX/Y`).
* **The "double peak" is gone, and what is left is a speckle, not a bug.** The companion
  looked like two blended peaks in earlier reductions — the signature of a derotation or
  registration failure — so it was measured rather than eyeballed
  (`scripts/check_hip65426_psf_shape.py`). Against the STPSF model's 1.08 axis ratio, the
  recovered companion is 1.27 combined and 1.24 / 1.92 in rolls 1 / 2, with **one** peak
  everywhere, and the centring fix improved the combined figure from 1.46 and moved the peak
  from 1.52 px off the catalogued position to 0.63 px. The residual stretch is not
  azimuthal — it lies 2.4–8.2° from the **detector** direction (which after derotation is the
  roll's own position angle, not zero). Four things say it is a speckle blended into the
  source and not an error in the pipeline: fakes injected at the same separation in the same
  reduction come out round (1.06–1.21, including at PA 120/130/170/180, right in the
  companion's neighbourhood); the frames are co-registered to 0.17 px; the shape does not
  move with `k_klip` (1.92–1.94 for k = 2…18, so it is not self-subtraction); and roll 2
  carries a residual at 1.03″, PA 166° at 89% of the companion's own peak that roll 1 does
  not have. A known speckle rotates by +8.97° between the rolls against the expected +10.08°,
  which is the geometry checking out. Combining the rolls dilutes the blend to 1.27 — which
  is what roll diversity is for.
* A Gaussian is also the wrong *shape*. Measured on these data: an STPSF off-axis template
  needs contrast 320 to reach the peak a Gaussian reaches at 40 — 8× — because the real PSF
  puts most of its light in wings and spikes.

**How much data there actually is.** Four science frames looks thin next to a ground-based
ADI sequence of hundreds, and it is worth being precise about why there is no more to be had.
A NIRCam exposure is a ladder, and only the top rung ever leaves the spacecraft as an image:

| rung | science (roll 1 and roll 2) | reference (each of 9 dithers) |
|---|---|---|
| frame — one non-destructive read of SUB320A335R | `TFRAME` = 1.06904 s | 1.06904 s |
| group — `NFRAMES` frames averaged **on the detector**, then `GROUPGAP` dropped | `DEEP8`: 8 + 12 → `TGROUP` = 21.381 s | `MEDIUM8`: 8 + 2 → 10.690 s |
| integration — `NGROUPS` groups up one charge ramp, fitted to a slope | 15 groups, `EFFINTTM` = **307.884 s** | 4 groups, **40.623 s** |
| exposure — `NINTS` integrations, one reset frame apart | 2 → `EFFEXPTM` = 615.767 s | 2 → 81.247 s |

`EFFINTTM` is the ramp span, `(NGROUPS·NFRAMES + (NGROUPS−1)·GROUPGAP) · TFRAME` — 288 frames
for `DEEP8`, 38 for `MEDIUM8`. It is *not* `(NGROUPS−1)·TGROUP`, which is the definition that
looks right and is off by 3% here and 21% for the reference. `INT_TIMES` confirms the
structure independently: each integration measures 288.04 frame times and the two are exactly
one frame apart, the reset.

So the JWST analogue of a DIT is `EFFINTTM`, and one plane of a calints cube is exactly one
of them. **There is nothing below it to recover.** The 8 frames of each group were averaged
in hardware before downlink, and the 15 groups are cumulative samples of a single charge
accumulation, not independent exposures — sub-ramp fitting would produce images that are
strongly correlated, share the same speckle realisation, and carry no new PSF diversity.
Total: 4 × 307.884 s = 1231.5 s on HIP 65426, 18 × 40.623 s = 731.2 s on φ Cen. The temporal
diversity available to KLIP is 2 integrations per roll; the *PSF* diversity comes from the
reference star's 18, which is the point of the 9-point dither.

`load_calints` records `effinttm`, `int_mid_mjd` (from `INT_TIMES`), `readpatt`, `ngroups`,
`nframes`, `groupgap` and `tframe` per frame in `info["frames"]`, because science and
reference integrations differ by 7.6× and any per-frame weighting that assumed a single
exposure time would be wrong for 18 of the 22 frames.

**Carter et al. (2023) agree.** Their Table 1 gives, for MASK335R/F444W: HIP 65426 —
`DEEP8`, N_groups 15, **N_ints 2**, t_exp 617.946 s, N_dithers 1, **N_rolls 2**, t_total
1235.892 s; HIP 68245 — `MEDIUM8`, N_groups 4, **N_ints 2**, t_exp 83.426 s, **N_dithers
9**, N_rolls 1, t_total 750.835 s. That is 2 × 2 = 4 science integrations and 2 × 9 = 18
reference integrations, matching the headers to the millisecond, and Section 2.2 confirms
they are not stacked: *"the subtraction is performed on each integration from both science
rolls individually, before being rotated to a common orientation … and summed together."*
That is what `load_calints` does. F444W is observed once, with one mask — Section 2.1 lists
MASK335R only, in F250M, F300M, F356W, F410M and F444W — so there is no second F444W
dataset to add.

Two bookkeeping notes for the paper's observation table:

* Their t_exp is `DURATION` (617.946 s), not `EFFEXPTM` (615.767 s); the 2.179 s difference
  is the two reset frames. Quote 1235.892 s to match Carter, 1231.5 s for time actually
  integrating. Either is defensible, but say which.
* Their prose quotes PA = 110.2° (ref), 110.0° (roll 1), 120.4° (roll 2), implying a 10.4°
  roll separation. The headers give **10.080°**, and `PA_V3` (10.085°) and `ROLL_REF`
  (10.080°) agree independently; roll 2's computed sky PA is 120.382°, which is their
  120.4°, while their other two look interchanged or rounded from the commanded attitude.
  The derotation here uses the headers. At 0.826″ the 0.3° is 0.15 px, so it does not move
  the companion, but it does matter if you try to reproduce their astrometry.

### α Cen — VLT/NEAR (`instruments/near.py`; `scripts/run_near2_production.sh`)

* **Template**: the measured off-axis AGPM-N4 library `n4_psf_cube_EEnorm.fits`, a
  `LibraryPSF` normalised to unit encircled energy within 1 λ/D, injected with
  `refpa_deg = -93.9`.
* **Star flux**: `template_flux_unit(PSF_ACenB.fits) = (129/57) × EE_B(1 λ/D)` — the
  measured encircled energy of the α Cen **B** template inside 1 λ/D, scaled to **A** by a
  hard-coded literature flux ratio.
* **Throughput**: the measured `throughput_merged.csv` table, falling back to the analytic
  AGPM curve. This is the only path where the Runner's `throughput_fn` is actually filled.
* **Status**: **the most complete chain in the package**, and absolute with respect to
  α Cen A — conditional on two things that are assumptions, not measurements from these
  data: the `129/57` constant, and the B template already being on the science frames'
  flux scale (no exposure-time or ND correction appears anywhere in the NEAR path).
* Minor: the library normalises inside `(LAMBDAD/1000)/0.0453 = 6.25` library px while
  `template_flux_unit` integrates inside `lam_over_d_px = 6.206` science px — 0.7 % in
  radius, and the library stamps are injected into 0.0456″/px data without resampling.

### RX J0534 — LBT/LMIRCam (`scripts/run_rxj0534.py`)

* **Template**: the median of the unsaturated PSFs, each normalised to unit sum first, as a
  `TemplatePSF` with unit total flux.
* **Star flux**: `mean over nights of (sum of the raw unsaturated PSF) × t_sci / t_un`.
* **Throughput**: none (`T ≡ 1`).
* **Status**: **measured and aperture-consistent** — total-sum normalisation on both sides,
  which is what `TemplatePSF`'s default wants. Assumes the data are in counts rather than
  counts/s, and detector linearity between the two exposure times. This is the cleanest of
  the generic-path calibrations and the one to copy.
* Watch: `generic.make_reducer` does `star_flux.get(pid)`, so a partition whose `.sav` has
  no exposure time gets neither a template nor a flux entry and silently falls back to
  `GaussianPSF(..., star_flux=1.0)` while its siblings keep real fluxes. The partitions are
  then combined as images, so one such night corrupts the combined scale. The only evidence
  is the per-partition `flux unit` in the startup log.

### LBTI/NOMIC (`instruments/nomic.py`)

* **Template**: pyNOMIC's own per-frame obstructed-Airy fit (`psf="airy"`, the default), so
  the injected source is literally `airy_disk(c · amp_j, …)` with frame *j*'s parameters;
  or `psf="frame"`, each frame's own unsaturated core within 1.5 λ/D.
* **Star flux**: **measured, per frame** — the fitted amplitude's core flux, or the frame's
  own core sum.
* **Throughput**: none, correct for the non-coronagraphic mode.
* **Status**: **the only per-frame flux calibration in the package.** `psf="gaussian"`
  exists for smoke tests and carries `star_flux = 1.0`.

## What the paper actually plots

`collect.py` does not trust any of the above. `anchor()` measures the companion's S/N
against three injections at its own separation and the same configuration, derives the
companion's contrast in template units, and divides by the published value to get
`flux_scale`; `figs.py` divides every plotted curve by it. So every published contrast axis
is anchored on Absil 2013 / Chauvin 2018 / Carter 2023, and gains, ratios and curve shapes
are untouched.

That is a defensible design and `collect.py` states it plainly. The consequence to be aware
of is that **it also hides how far off the star fluxes are**, which is how a 1347× error in
the SPHERE path and a complete absence of photometry in the JWST path both survived. The
recorded `flux_scale` is the diagnostic: it should be ≈ 1 when the star flux is right.

```
run                   flux_scale    reading
A2  beta Pic            5.28        halo estimate ~5x too small     [as run; see below]
B2  beta Pic            8.04        same method, same data, 8x -- the estimator is noisy
C   HD 95086         9.34e-04       1/1071: the DIT/ND factor applied twice (fixed)
D   HIP 65426        2.35e+05       no photometry at all: flux_unit = 1.0  [pre-fix; rerun needed]
```

Runs A2 and B2 above predate the β Pic star flux; with `star_flux = 3.3268e6` the same
measurement gives **1.21**, so `flux_scale` becomes the 0.2 mag residual against Absil
rather than a factor of five. **A, A2, B and B2 have to be re-collected** (`collect.py`)
before their contrast axes are quoted; the searches themselves are unaffected, since a
common star flux cancels out of every S/N comparison the optimizer makes.

## Open items

Closed on 2026-09-15: β Pic's absolute photometry (`star_flux = 3.3268e6` from VIP's
published `starphot`, checked against β Pic b at 1.1 σ) and HIP 65426's, which had none at
all. What remains:

1. **`optics_transmission = 0.561` is anchored on HIP 65426 b.** Replacing it with the
   tabulated COM substrate transmission (JDox "NIRCam Filters for Coronagraphy", or
   `webbpsf_ext`'s COM throughput) at 4.44 µm would turn the HIP 65426 b comparison from a
   calibration into a real check, and would let the same number serve other NIRCam
   coronagraphic programmes without re-anchoring. It is the last assumed number in the table.
2. **Runs C, D, G2 and H2 have to be redone.** D and H2 now build the model above, but the
   archived results predate both the flux scale and the 1.48 px star-centre correction. C and
   G2 predate the HD 95086 star-flux fix (see that section). G2 additionally has to be redone
   because of the rank collapse below.
2b. **Every archived run needs the rank-collapse audit.** A shared KL basis built from the
   frames it subtracts spans them exactly, so `k_klip >= n_binned_frames` returned float64
   round-off (image rms ~1e-15) and an S/N that is an O(1) random draw. `klip_annular` now
   caps `k` and records `k_requested`/`k_effective`/`k_capped`, but the archived runs were
   searched without the cap: on `bench_20260915185229_tpe_s2` **548 of 800 evaluations were
   degenerate**, the best of them scored +2.410 and the best honest one +0.006. Exposure is
   `k_max >= n_frames/bin`: β Pic (61 frames, `k_max` 30) from `bin >= 3`, HD 95086 (63) from
   `bin >= 3`, F2 (`k_max` 12) from `bin >= 6`. A run is safe only if its winner — and the
   scores it was chosen over — sit below that line.
3. **`star_center` for HIP 65426 is solved from published astrometry**, so it is the geometry
   the photometry needs rather than an astrometric measurement. spaceKLIP's own star-centring
   step (`STARCENX/Y`) would make it independent; `load_calints(star_center=)` takes it.
4. **`contrast_curve.txt` claims more than it knows.** The header should say when
   `flux_unit == 1.0` that the axis is not a contrast.
5. **`star_flux` is silently ignored when `injection_model` is passed**
   (`spaceklip.make_reducer`). No call site passes both any more, but nothing stops one.
6. **The aperture pairing is a convention nothing enforces.** Four different normalisation
   apertures are in use and a `star_flux` has to match whichever the model used. A check
   comparing the model's aperture with the provenance of the star flux would catch this
   class of error at construction rather than in a contrast curve.
7. **The live display's contrast curve omits the throughput correction** the written
   product applies (`display.py` calls `contrast_curve` with no `throughput_fn`). On the
   NEAR path the on-screen and saved curves are different functions of the same data.
8. **An injection can be too faint for the cube's dtype.** `inject_sources` holds the cube
   in float32, so a stamp whose pixels fall below the float32 quantum of the science pixels
   they land on is rounded away entirely — on β Pic a stamp peaking at 1e-4 counts loses a
   quarter of its flux, and the loss *grows* as the contrast falls, so the recovered S/N
   stops being proportional to the contrast and every curve derived from it bends. This
   never bites when `star_flux` is the star's real flux (the amplitudes are then of order
   1e2–1e3 counts) and it did not affect any paper run, but it is exactly what a normalised
   template's `flux_unit` of 4.35 walks into. `_check_float32_headroom` now raises a
   `RuntimeWarning` below 32 ULP. A cleaner fix is to keep float64 cubes in float64.
