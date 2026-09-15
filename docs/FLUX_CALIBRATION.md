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

### HIP 65426 — JWST/NIRCam F444W (tutorial 03; paper runs D, H2)

* **Template**: whatever `spaceklip.make_reducer` falls through to. Runs D and H2 pass no
  `psf_template`, no `star_flux` and no `injection_model`, so they get
  `GaussianPSF(1.028 λ/D, star_flux=1.0)`.
* **Star flux**: **1.0.** There is no photometry.
* **Status**: **absent.** The contrast axis is raw detector units. `run_H2`'s forced
  calibration contrast of `5.270e1` is the tell — a contrast of 52.7 is not a contrast.
  `collect.py` records `flux_scale = 2.35e+05` for run D, implying a companion "contrast"
  of 77 against a published 3.30e-04 (Carter et al. 2023).
* Tutorial 03 is honest about it: its contrast-curve label reads `"(template units)"`
  when `STAR_FLUX` is None. The paper runs are not, because they go through
  `contrast_curve.txt`, which always says "injection-calibrated".
* A Gaussian is also the wrong *shape*. Measured on these data: an STPSF off-axis template
  needs contrast 320 to reach the peak a Gaussian reaches at 40 — 8× — because the real
  PSF puts most of its light in wings and spikes. The Gaussian both misses the structure
  (see below) and mis-scales peak-to-flux.
* `psf="stpsf"` is available on the same `make_reducer` call and would supply both the
  right shape and a measured mask throughput. It is unused by runs D and H2.

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
D   HIP 65426        2.35e+05       no photometry at all: flux_unit = 1.0
```

Runs A2 and B2 above predate the β Pic star flux; with `star_flux = 3.3268e6` the same
measurement gives **1.21**, so `flux_scale` becomes the 0.2 mag residual against Absil
rather than a factor of five. **A, A2, B and B2 have to be re-collected** (`collect.py`)
before their contrast axes are quoted; the searches themselves are unaffected, since a
common star flux cancels out of every S/N comparison the optimizer makes.

## Open items

1. ~~**β Pic has no absolute photometry.**~~ **Done.** `star_flux = 3.3268e6` from VIP's
   published `starphot`, checked against β Pic b at 1.1 σ. See the β Pic entry above and
   `scripts/check_betapic_contrast.py`.
2. **HIP 65426 has no photometry.** Either give runs D and H2 `psf="stpsf"` plus a measured
   stellar flux (the target-acquisition image, or reference-star photometry in the same
   aperture), or keep the Gaussian and label the axis "template units" everywhere,
   including in `contrast_curve.txt`.
3. **`contrast_curve.txt` claims more than it knows.** The header should say when
   `flux_unit == 1.0` that the axis is not a contrast.
4. **`star_flux` is silently ignored when `injection_model` is passed**
   (`spaceklip.make_reducer`). Tutorial 03 passes both. It should warn.
5. **The aperture pairing is a convention nothing enforces.** Four different normalisation
   apertures are in use and a `star_flux` has to match whichever the model used. A check
   comparing the model's aperture with the provenance of the star flux would catch this
   class of error at construction rather than in a contrast curve.
6. **The live display's contrast curve omits the throughput correction** the written
   product applies (`display.py` calls `contrast_curve` with no `throughput_fn`). On the
   NEAR path the on-screen and saved curves are different functions of the same data.
7. **An injection can be too faint for the cube's dtype.** `inject_sources` holds the cube
   in float32, so a stamp whose pixels fall below the float32 quantum of the science pixels
   they land on is rounded away entirely — on β Pic a stamp peaking at 1e-4 counts loses a
   quarter of its flux, and the loss *grows* as the contrast falls, so the recovered S/N
   stops being proportional to the contrast and every curve derived from it bends. This
   never bites when `star_flux` is the star's real flux (the amplitudes are then of order
   1e2–1e3 counts) and it did not affect any paper run, but it is exactly what a normalised
   template's `flux_unit` of 4.35 walks into. `_check_float32_headroom` now raises a
   `RuntimeWarning` below 32 ULP. A cleaner fix is to keep float64 cubes in float64.
