# Changelog

## Unreleased — 2026-09-23 (the draws as an error bar)
- **Every score plot now shows the min-to-max span of the draws** behind each trial when
  `n_remeasure > 1`: the live panel's convergence trace (`panel_trace`) and the saved
  figure (`plots.plot_trace`, hence the PDFs and `plot_all`).  Without it a panel presents a
  mean as though it were a measurement, and on this objective the draws span ~1.4 in S/N
  against a useful range of ~6 — the uncertainty is a large fraction of the axis and ought
  to be on it.
- The observed range, not `± sd`: with three draws a standard deviation is a
  two-degree-of-freedom estimate dressed as a confidence band, whereas min-max is exactly
  what was measured.  Drawn in grey beneath the points so the trial's score still reads as
  the datum and the bar as its uncertainty, and labelled with the draw count
  (`draw range (min-max of 3)`).
- `plots.draw_ranges` / `plots.draw_scores_of` are shared by both callers (`display` imports
  them lazily, since `plots` must not import `display`).  A single-draw run adds nothing at
  all — no bar and no legend entry — and a wholly failed trial is skipped rather than
  plotted at zero.

## Unreleased — 2026-09-23 (per-trial remeasurement; dead search dimensions removed)
- **Search trials are scored as the mean of `RunConfig.n_remeasure` fresh source draws**
  (`Runner.evaluate_mean`), 3 by default on MIRI via `--n-remeasure`.  The objective's only
  stochastic input is the injected sources' azimuth anchor, and it is not small: on
  HIP 65426 F1140C, repeated evaluations of *identical* configurations scatter with
  **sd 0.84** against a total useful range of 0.86–7.11.  A search that cannot separate 0.84
  of S/N spends much of its budget ranking noise — it is also why the winner's curse runs
  ~1.13× and why single-dimension effects were invisible in v3's 1000 evaluations.
- Averaging 3 draws measured **0.48**, a factor 1.74 — the √3 an independent draw predicts,
  which is itself the evidence that the draws are independent.  The **mean**, not the median:
  median-of-3 reached only 0.68 on the same data, because the residuals are mildly
  left-skewed (−0.85) with no excess kurtosis (−0.03), so there are no heavy tails for a
  median to earn its efficiency back on.  `CalibrationConfig.n_remeasure` (the calibration's
  own, median-based) is untouched.
- One trial stays one history entry however many draws it took; appending them individually
  would let the optimizer read one configuration as n and triple the apparent budget.  The
  record carries the mean in `score` and everything positional from the **last** draw, so the
  panel shows one real measurement beside the average, and says so: `= mean of 3 draws
  [5.2, 5.8, 5.2]  sd 0.35  (image is the last draw)`.  `meta['draw_scores']` /
  `draw_sd` / `draw_spread` make the run report its own noise as it goes.  A failed draw is
  dropped from the mean rather than counted as zero.
- New `RunCallback.on_draw` carries each draw to the display.  Deliberately not `on_eval`:
  that hook's timestamps are the source of s/eval and both ETAs and it drives the panel
  cadence, so feeding draws through it would report the gap between draws as the cost of a
  trial and divide the ETA by `n_remeasure`.
- **MIRI drops `angsep` and `anglemax`** (`search_angles=False`) — 12 searched dimensions to
  8.  They are not weak on two-roll data, they are inert: each partition is one roll, so
  every science frame in it shares a position angle and `reference_mask`'s
  `dpa = |angle − angle[target]|` is identically zero.  Then `dpa <= anglemax` holds for any
  `anglemax >= 20` (the parameter's own floor), and `dpa >= angsep_deg` fails for every frame
  once `angsep > 0`, which empties the mask and falls through
  `if refs.sum() < 4: refs = all but the target` — exactly the `angsep = 0` set.  The same
  basis by two routes.  Corroborated in v3: groups differing only in `anglemax` spread 1.31
  in score, against 1.41 for groups differing in *nothing*.
- **A parameter whose range has collapsed to a point is pinned, not searched.**  NIRCam hit
  this for real — its cubes are short enough that `bin_range` came out `(1, 1)`, so every
  HIP 65426 run has carried `bin : [1.000, 1.000]` as a search dimension.  `fixed` reaches
  the reducer through `decode`, so the value is unchanged and only the coordinate goes away.
- **Fixed a latent crash:** `anglemax_hi` was taken from the sequence's PA span while the
  parameter's floor stayed at 20, so any span in (5, 20]° built `Param(20, span)` and raised
  `anglemax: hi < lo` in `make_space`, before a single evaluation ran.  Such a span cannot
  constrain anything, so `anglemax` is now pinned open at 360 instead.
- `n_ang` is **kept** as a searched dimension.  It is azimuthal subdivision of the KLIP zone,
  which changes the reduction whatever the number of rolls, so the two-roll argument that
  retires `angsep`/`anglemax` does not reach it.
- `tests/test_display.py::test_resume_restores_best_images` asserted the pre-crash incumbent
  was still best after the resumed evaluations finished, which tests the score landscape
  rather than the resume; the radprof change moved the landscape and eval 5 now wins
  legitimately.  It now checks that the restored images are the pre-crash incumbent's at the
  moment of restore, and that the images on hand afterwards belong to whatever is best then.

## Unreleased — 2026-09-22 (correction: what destriping is actually worth)
- **The destriping gain below is overstated, and this is the number to use: 1.28x, not
  1.63–1.77x.**  Measured in the running pipeline on HIP 65426 F1140C — per-row sigma 1.05,
  per-column 0.327, frame scatter 1.27 → 0.991.  The 1.63–1.77x was measured on a *raw*
  calints integration, and in the pipeline the destriper runs **after** the blank-sky
  background subtraction, which has already removed most of the pattern.
- The two are consistent, and the reconciliation is the interesting part.  The row pattern
  correlates at +0.997 between integrations, i.e. it is static; being static makes it
  largely common-mode, so subtracting a blank-sky median takes most of it away on its own.
  What destriping removes is the part that differs between the science and background
  pointings.  Still worth having — 28% in noise — but it is not the 1.7x the earlier entry
  claims, and the earlier claim was never measured anywhere the pipeline actually runs.

## Unreleased — 2026-09-22 (three MIRI annuli, and the packing cap measured properly)
- **The packing cap was testing a geometry the sampler never produces.**  My own bug, from
  the commit before this one: `max_sources_for_noise` assumes every source sits on the
  same ring.  `PositionSampler.sample`'s default strategy steps `rho` ACROSS the injection
  band, so sources sit at different radii and mostly stay out of each other's exclusion
  zones.  Measured with the real sampler and the real estimator, MIRI's 6.7–36 px annulus
  holds **8 sources with 10 clean apertures on the worst ring, and still passes at 14** —
  where the co-radial formula said 2.  `Runner._ring_survives` now samples actual positions
  and scores them with `mawet_peak_snr` itself, so it cannot drift from the estimator it is
  protecting; `max_sources_for_noise` stays as a cheap bound for callers with no sampler,
  documented as co-radial-only.  `_band` gained `_band_for(ia, n)` so the check can ask what
  a candidate count would place without recursing.
- **The ring criterion is almost never the binding constraint** once sources spread in
  radius — NIRCam passes at 8 too (14 apertures).  It bites only in the co-radial cases:
  `fixed_pa`, a collapsed band (`opt_width`), and the `pair_area_midpoint` pair.  So
  NIRCam's 2 rests entirely on mutual contamination, as Kevin said, and not on the ring.
- **MIRI now runs three annuli instead of one** (`run_miri.default_annuli`, `--ann` takes
  any number of increasing edges).  F1140C goes from `[6.68, 36]` to
  `[6.68, 20.05, 26.86, 36.00]`, i.e. 4.0 / 2.0 / 2.7 FWHM wide, with `n_sources_rule`
  supplying **4 / 6 / 6** — so more than two everywhere, which was the ask.  One annulus
  over 2–11 FWHM averages the inner working distance together with the background-limited
  outside, and the best KLIP parameters are not the same at both ends.
- **The first annulus ends at 6 FWHM because that is where the measurement puts it**, not
  by taste.  With the 4QPM dead zones eating ~10% of the ring and HIP 65426 b sitting on
  it, an inner zone ending at 6 FWHM leaves 8 clean apertures for 3 sources and 7 for 4;
  ending it at 5 FWHM leaves room for **2**, and at 4.5 FWHM the 1-FWHM inset collapses the
  band entirely.  Shrinking it further and keeping more than two sources are in direct
  tension, and 6 FWHM is where they meet.
- The two outer zones split at the geometric mean, so they are comparable in log radius
  rather than one thin and one huge; any zone under 1.5 FWHM wide is merged away, which is
  what takes **F1550C** to two annuli (its FWHM is 4.5 px, so the three-way split would
  leave sub-FWHM slices that cannot carry their own parameters).
- The forbidden sectors are now computed at the **innermost** annulus' mid-radius.  A dead
  zone of fixed physical width subtends a larger angle closer in, so one radius has to be
  chosen; over-masking the outer rings, which have aperture budget to spare, is the safe
  direction.

## Unreleased — 2026-09-22 (MIRI destriping, source packing, the display's model orientation)
- **The display's injected-PSF model was built at the wrong roll.**  Spotted by Kevin:
  the matched-PSF panel's lobes pointed the wrong way.  `injected_model_image` injected
  into a single `parang = 0` frame and derotated by the true-north offset alone — it
  assumed the telescope had been pointed at roll 0.  That was invisible while the
  injector span the stamp by the source's azimuth, and became glaring once the template
  went spacecraft-fixed (`815fc55`), because the derotation is then the *only* thing
  carrying lobe structure into the sky frame.  Measured on HIP 65426's rolls of 108.0 and
  117.4°, the panel's lobes sat **112° away** from what is in the data.  It now builds at
  the data's own angles (distinct rolls only, weighted by frame count) and derotates and
  combines them as the science frames are, so a multi-roll set shows the real
  superposition.  The *metric's* matched filter was never affected: `kernel_from_profile`
  azimuthally averages the stamp, so it is circularly symmetric and has no orientation to
  get wrong — which is also to say it discards the lobe structure entirely.
- **MIRI is destriped in the detector frame by default** (`spaceklip.destripe_detector`,
  `--no-destripe` to skip).  On a real F1140C integration the per-row offset has a scatter
  of 3.1–3.7 MJy/sr against a pixel-to-pixel scatter of 3.0–3.7 — *the striping is as
  large as the read noise* — and removing it drops the sky scatter by **1.63–1.77×**.
  Columns carry only ~0.3× and are taken too.  The row pattern correlates at **+0.997**
  between integrations of one exposure, so this is a static detector pattern, not random
  1/f per frame; it survives into the KLIP residual as a fixed shape that derotation then
  smears round the field instead of cancelling.
  It runs on the **full subarray, after the background subtraction and before the crop**.
  That is not a preference: a row of MIRI's 288×224 MASK1140 subarray is mostly sky, but
  every row of the 81×81 crop the reducer works with passes through the coronagraphic PSF,
  so destriping the crop would subtract the target.  The star (r < 45 px) and the 4QPM
  boundaries (< 12 px, where the glow sticks run) are masked out of the estimate, then
  what is left is sigma-clipped; the offset is subtracted from the whole row regardless.
  The existing per-evaluation `do_destripe` (the IDL's, on the crop) is untouched and
  still off.
- **Injected source counts now respect the noise ring.**  `positions.max_sources_for_noise`
  computes how many sources a ring can hold while leaving `min_ring` (6) clean apertures
  for the Mawet S/N, from `mawet_peak_snr`'s own geometry — the CHORD test it actually
  applies, not an arc approximation — and `n_sources_rule` caps on it, at the annulus's
  **inner** edge, where apertures are scarcest.  It applies to an explicit `n_sources`
  too, because past the cap the estimator abandons the ring for the radial band and the
  extra sources cost every separation its noise estimate; `Runner._nsrc` logs when it
  binds.  `mawet_peak_snr`'s details now report `nclean` / `ring_used`, so the number is
  visible rather than inferred.
- **What that measured, which was not what we expected.**  On the real `D_hip65426`
  stitch, NIRCam holds 4 sources with **9** clean apertures (5 with 7) — it passes
  comfortably.  MIRI is the one that starves: at its inner edge 3 sources already drop the
  ring to **4**, because the 4QPM dead zones eat ~10% of it before any source lands.  So
  the packing cap takes MIRI to 2 and leaves NIRCam alone.
- **NIRCam is set to 2 anyway, for a different reason.**  Kevin's call: four sources at one
  contrast in that small a field perturb the KLIP basis each other sees, which is mutual
  contamination and is invisible to an aperture count.  `run_D`, the `H2` bench,
  `calibrate_bench_contrast`'s H2 entry and `run_rxj0534`'s default all go to 2.
  **`run_mwc758.py` is deliberately left at 3**: its contrast is forced "as the IDL run had
  it" and changing the source count breaks that cross-check.  Say the word and it moves.

## Unreleased — 2026-09-22 (radial-profile subtraction off by default)
- **`radprof` is no longer applied anywhere by default.**  At Kevin's direction.  The
  azimuthal integer-radius-bin mean subtraction was the IDL's default and was inherited
  here as nine *independent* defaults — the three metrics, the FMMF map, the display
  panels, the saved stitches, the FM contrast curve, the verify subsets, `param_verify`
  and the candidate search.  Six of those were not switches at all: they called
  `radprof()` unconditionally, so there was no way to turn them off and no record in the
  output that they were on.  All nine now take `flatten=`, defaulting to `False`.
- **Why it is wrong as a default.**  `radprof` subtracts the mean of each integer-radius
  bin from every pixel in that bin, so anything that is not azimuthally uniform at a
  given radius leaks into that mean and is then removed from the whole ring — a bright
  companion subtracts a fraction of itself, and on MIRI the 4QPM dead zones and
  glow-stick residuals bias the ring they sit on.  Worse on a masked ring, where the
  mean is taken over the surviving pixels and removed from all of them.
- **The saved products now follow the metric** (`Runner.flatten_products`) instead of
  flattening regardless.  Previously a run could be optimised on one image and shipped
  with another, and the FITS header said `radprof-flattened` either way; the `IMGTYPE`
  strings now carry the tag only when it is true.
- **This changes the numbers, and not by a little.**  On a synthetic frame with an
  `exp(-r/8)` halo, the same source measures **S/N 34.5 flattened and 3.2 unflattened** —
  the radial gradient across each reference aperture inflates the ring scatter.  Real
  post-KLIP residuals are far shallower than that, but the direction holds: absolute S/N
  falls wherever a radial gradient survives, so the calibration lands at a different
  contrast and **no number from before this commit is comparable to one after it**.  That
  is the intended effect — the optimiser must now suppress the gradient itself rather
  than lean on `radprof` to hide it afterwards.  Every benchmark (E2/F2/G2/H2) and every
  MIRI contrast predates the change.
- Pass `flatten=True` anywhere to restore the IDL behaviour.  `tests/test_metrics.py`
  pins all nine defaults and both halves of the 34.5/3.2 split, so a silent flip fails.
- **Not changed, and needing a decision**: `scripts/check_hip65426_contrast.py`,
  `check_betapic_contrast.py` and `check_hd95086_klipfm.py` still call `radprof()`
  explicitly.  They are published-value cross-checks, and the HIP 65426 one is the source
  of the ΔF444W = 8.796 ± 0.092 below — changing them moves a manuscript number.

## Unreleased — 2026-09-22 (the injected PSF's orientation)
- **The injector rotated the PSF template by the source's position angle, and should not
  have.**  Spotted by eye on a live display: the injected sources' side lobes pointed the
  wrong way.  `stpsf_psf.library` and `miri.library` set `refpa_deg=0`, so `inject_sources`
  span the stamp by the source's detector azimuth — but what you can see in a JWST
  coronagraphic PSF (the Lyot stop's pattern, the segmented pupil's lobes) is fixed to the
  **spacecraft**, not to where a companion happens to sit relative to the mask.  It does not
  turn as the companion moves round the field, so neither should the template.  Measured
  against STPSF on F1140C at 2″: rotating by 90° misplaces **23%** of the stamp's flux
  (3.7% at 180°, where the mask and stop are symmetric).  The IDL reduction this package
  ports settles it — `reduce_nircam_v13.pro` reads one WebbPSF template per filter, centres
  it with `cntrd` + `fshift`, and at injection does `fshift(big_ref * contrast, …)` per frame
  and nothing else: one template per data set, in the spacecraft's orientation, never
  rotated.  Both libraries now default to `refpa_deg=None`; pass a number to reproduce an
  older run.
- **This moves the F444W cross-check**, which is the end-to-end test of the whole flux axis:
  HIP 65426 b now measures **ΔF444W = 8.796 ± 0.092** against Carter et al. (2023)'s
  8.703 ± 0.055, where it read 8.74 before.  That is +0.056 mag, *away* from the published
  value — 1.0σ instead of 0.4σ, still a pass, and still with nothing tuned.  Recorded because
  the physics decided it and not the agreement: the ±3% on the stellar flux density is
  ±0.033 mag on its own, so the previous closeness was partly luck.  Anything quoting 8.74
  (including the manuscript) needs updating.

## Unreleased — 2026-09-22 (the k-scan's argmax)
- **The calibration k-scan reported an edge of its own range as an optimum, twice.**  Chasing why
  the HIP 65426 F1140C run picked `k_default = 1`: with the background-mismatched RDI library the
  S/N-vs-k curve was flat to **1.5%** over k = 4..20 with k = 1 on top by 1.6%, which is the
  signature of a basis whose leading KL mode is the sky pedestal rather than the star — removing
  it was the only subtraction that helped, so nothing after k = 1 added anything.  That confirms
  the background mixture (`2b8a148`) was the cause.  But with the background fixed the argmax
  moved to **k = 20**, the *other* end of the range, winning by **0.1%**.  Neither was an interior
  optimum, and `--k-max 20` caps the *search* as well as the scan, so the run was working against
  a ceiling the data had already reached.  `Runner._scan_k` now names an argmax at either end as
  an edge hit (and at the top says which knob is limiting), keeps the draw-to-draw spread it used
  to discard with the median (`info["kscan_draw_spread"]`), and keeps the incumbent when the win
  is inside that spread rather than moving the run's seed on noise.  This is the guard
  `locate_boundaries` has always applied to its own scan — "an extremum at the edge of the window
  means the window is wrong" — finally applied to this one.  New `tests/test_kscan.py`, in the
  quick suite because the curve is dictated by a stub and no reduction happens.

## Unreleased — 2026-09-22 (the cache, on a machine without STPSF)
- **A missing cache file failed as `ModuleNotFoundError: stpsf`, six frames down.**  The cache
  exists precisely so a machine without STPSF can run from a copied one — the Mac this is
  developed against runs Python 3.9, which STPSF will never install on — so a cache miss there
  means "you are missing one file", not "install STPSF", and the error has to say *which* file.
  `throughput_map` already did this; `unocculted_ee` and `offaxis_grid` did not, and the new
  automatic flux route reaches `unocculted_ee`, so the F1140C run died on a bare ImportError
  after loading 26 files.  Both now check `have_stpsf()` before touching STPSF and name the
  exact cache filename and directory to copy it into.
- Shipped `eeunocc_MIRI_F1140C_…` and `eeunocc_MIRI_F1065C_…` to the cache, so both filters'
  flux units resolve offline (F1065C: S = 0.0781 Jy, EE(4.50 px) = 0.4753, star_flux = 1.2981e5).
  F1550C still needs its throughput map computed before that filter can run at all.

## Unreleased — 2026-09-22
- **The RDI library mixed background-subtracted and unsubtracted frames.**  A programme's
  *science* targets get dedicated background pointings and Image2 subtracts them
  (`S_BKDSUB='COMPLETE'`); its pure *PSF reference* stars usually do not get one, so the step
  never runs on them.  On ERS 1386 at F1140C that is both HIP-65426 rolls and both HD-141569A
  exposures subtracted, against HIP-68245 (9 files) and HD-140986 (5) not — so 14 of 16
  reference frames arrived carrying a ~19 MJy/sr sky pedestal and the 4QPM glow sticks, and
  `load_calints` stacked all of it into ONE KLIP library beside science frames with neither.
  The library's dominant common mode is then the background rather than the stellar PSF.
  (**Correction, 2026-09-22:** this entry and the `2b8a148` commit message went on to predict
  that the mismatch would push the optimizer towards a hard high-pass, to hide the pedestal,
  and that fixing it would bring the filter width down.  The paired 1000-evaluation runs say
  otherwise: the mismatched run chose `filter = 5` and the fixed one `filter = 6`.  The
  prediction was wrong.  Where the mismatch *did* show is the k-scan curve — see the entry
  above — and the calibration contrast, which fell from 2.27e-4 to 1.63e-4, so the default
  configuration reaches S/N 5 on a source 28% fainter.  The fix stands on the mixture itself
  and on those two measurements, not on the high-pass argument.)
  `load_calints` now reads `S_BKDSUB` per exposure and subtracts the median of the
  programme's own blank-sky pointings (`blank_sky`, new) from whichever frames lack it,
  leaving the rest alone; a mixture with no background to fix it with raises rather than
  proceeding, and `background=False` stacks it anyway for anyone reproducing an old run.
  Measured on the real files: reference-library median 22.20 → 3.14 MJy/sr against the
  science frames' 0.52, and the glow-stick excess along the horizontal mask boundary
  **+12.70 → +4.31 MJy/sr**.  The residual is plausibly the *starlight* scattered by the mask
  — present in proportion to each star and so not removable by any blank-sky frame — which is
  exactly what a PSF reference is supposed to carry and RDI to remove.  `calints` are in
  MJy/sr, a rate, so exposures of different length subtract with no scaling.

## Unreleased — 2026-09-21 (the flux unit)
- **The MIRI search spent two hours ranking noise, because `star_flux` defaulted to 1.0.**
  `make_reducer(star_flux=None)` becomes `star_flux or 1.0`, and 1.0 is not a neutral default
  — it makes one unit of *contrast* worth one count, against a MIRI cube whose pixels reach
  several hundred MJy/sr.  For HIP 65426 in F1140C the right value is **1.1238e5**, so every
  injected source was 1.1e5 times too faint.  The evidence is unmistakable once looked at: the
  calibration walked its entire ladder — 3e-5, 3e-4, 3e-3, 3e-2, 1e-1 — and got median S/N of
  −0.11, 0.00, −0.02, −0.31, −0.04.  A response flat over four orders of magnitude is not a
  faint source, it is an inert one.  The run then reported "could NOT be calibrated", blamed
  the injected sources limiting each other, pinned the contrast at the 1e-1 cap and searched
  on for 300 evaluations.  `reducer.py`'s existing `flux_unit is 1` warning did not fire: it
  tests whether the injection is *representable* in float32, not whether it is detectable, and
  at 1e-1 it was representable.  `scripts/run_miri.py` now derives the flux unit or refuses to
  start — `--star-flux`, `--flux-density-jy` (converted with the frames' own `PIXAR_SR` and the
  injection library's own EE radius, via the new `miri.star_flux_from_flux_density`), or
  `datasets.PHOTOMETRY['<target>_<filter>']`.  `--star-flux 1` still works, because a raw-units
  run is legitimate — but it has to be asked for.
- `datasets.PHOTOMETRY` gains `hip65426_f1065c` / `_f1140c` / `_f1550c` (0.07813 / 0.06899 /
  0.03739 Jy).  Same star and same method as the F444W entry, and deliberately **ratio-anchored**
  to it rather than computed from scratch: the Planck-through-the-bandpass recipe reproduces
  0.40259 Jy to −3.3% (photon-weighted), inside its own ±3%, and the residual is the 2MASS
  zero-point convention, which the ratio cancels.  Teff = 8600 K is Carter et al. (2023)'s own
  PHOENIX fit, so both rest on one model.  The photosphere is the right thing to use: the only
  excess those authors report is 3.5σ at 24 µm with T_dust ≈ 300 K, negligible at 11 µm.
  Independent check — at S = 0.0690 Jy the paper's ~2.7 µJy F1140C sensitivity is a 3.9e-5
  contrast floor against their ~2e-4 for the companion; the two hang together, and would not if
  S were wrong by a factor.

## Unreleased — 2026-09-21 (display)
- **`run_miri.py` wrote no panels unless a window was open.**  It built its `LiveDisplay` only
  under `--show`, so a run started without it produced no panel PNGs at all — and then
  `klip-tpe view --run-dir`, which the script itself prints as the way to watch a run, had
  nothing to watch, for the whole run, with no way to attach later.  The advice and the
  behaviour contradicted each other.  `klip_tpe.cli` has always kept the two apart, and this now
  does too: `--display` (default on) writes, `--show` opens a window, plus `--no-display`,
  `--display-every` (10) and `--pdf-every` (0).
- **`klip-tpe view` stole the foreground once a second.**  Its refresh loop ended in
  `plt.pause(interval)`, which raises *and focuses* the window on every call — on macOS that
  makes the terminal running the optimizer unusable, which defeats the point of watching from a
  second shell.  It now uses `draw_idle()` + `flush_events()`, the idiom
  `display.LiveDisplay`'s own live window already used with the comment "no show()/pause -> the
  window is never raised" on the line.  The wait between refreshes is chopped into 50 ms pieces
  so the window still answers clicks and resizes.  New `tests/test_viewer.py` pins both, and
  pins the two modules to keep agreeing.
- `run_miri.py --star-center X Y`, because `load_calints` logs "pass star_center= if you have
  one" on every MIRI run and the driver could not.  Measured on GO 1386 F1140C: the most
  point-symmetric centre is 0.39 px (45 mas, 0.12 FWHM) from CRPIX with the two rolls agreeing
  to 0.05 px, so CRPIX is adequate there.  Measure it by point symmetry, not with a flux
  centroid — on a four-quadrant residual the centroid is biased by the pattern and lands 0.4 px
  on the *opposite* side, varying by 0.4 px with the aperture radius.

## Unreleased — 2026-09-21 (later)
- **`load_calints` returned a cube that was 100% NaN, and said nothing.**  This is what actually
  killed the HIP 65426 F1140C run (GO 1386).  A MIRI MASK1140 subarray is 27.6% DQ `DO_NOT_USE`
  — the unilluminated border outside the coronagraph field — and that region is far too wide for
  `fill_dq_neighbours` to close from its rim, so 13.3% of each frame is still NaN when the
  alignment runs.  Both sub-pixel resamplings in `load_calints` (the cross-correlation shift and
  the crop's fractional offset) went through `ndimage.shift(..., order=3)`, whose spline
  prefilter is a *recursive* IIR filter: a single non-finite pixel propagates to every pixel of
  the output.  Measured on a real frame: 13.3% NaN in, **99.2%** NaN out (`order=1` gives 13.3%).
  Two such calls took the returned cube to 100% NaN, and the failure surfaced two steps
  downstream in the reducer as "every frame was dropped as empty" — which reads like a masking
  problem and is not one.  New `shift_keeping_gaps` zero-fills the gaps for the interpolation and
  runs the same interpolator over the finite-ness mask, which is a partition of unity: pixels
  where it does not come back at exactly 1 are the ones that would be carrying invented values,
  and those come back NaN.  The mask is extended with `mode='nearest'` so the frame border — a
  boundary condition, not a gap — is not flagged, and a frame with no gaps returns
  `ndimage.shift` unchanged, so NIRCam results are bit-for-bit as before.  The crop then closes
  what the two splines widened.  Verified end to end on the real data: cube 0.00% non-finite,
  reduced image 100% finite inside the search annulus.
- **The frame registration reported the corner of its search box as a measurement.**  `_xs` took
  `np.fft.rfft2` of a frame that could contain NaN — which makes the whole transform NaN, and
  `np.argmax` of an all-NaN array returns 0, decoding to a confident `(-6, -6)` px shift for
  every frame of every MIRI cube.  Even with the gaps zero-filled, `argmax` returns *something*
  when there is no peak inside the box, and on HIP 65426 F1140C the peak sits 100+ px away with
  zero lag a local *minimum*: the 41 integrations of one exposure have no relative offset to
  measure.  `_xs` now subtracts the median and zero-fills before the FFT, and returns
  `(0, 0, ok=False)` when the peak is on the box boundary or not positive; those frames are left
  **unshifted** rather than moved by the highest corner, and the count is logged.  On the real
  cube 37 of 41 frames register (interior, positive) and 4 do not.
- **The DQ-fraction warning counted the wrong pixels.**  It reported the flagged fraction of the
  whole subarray — 27% for MIRI — when the returned stamp is 81×81 and 44 of its 6,561 pixels
  (0.7%) were flagged.  Alarming about pixels nothing downstream sees sent one debugging session
  after the wrong thing entirely.  The log now gives both, the warning keys off the stamp, and
  `info` gains `repaired_fraction_crop` and `nonfinite_fraction`.
- **`run_miri.py --check`**: the cube's non-finite fraction and the filter width are now printed
  *before* the reduction rather than after it (a diagnostic printed after the step that fails
  never prints — that is why this took two runs to see), a non-finite cube plus a high-pass is
  refused with the reason, and the check no longer passes a reduction whose search annulus is
  empty.  It measures finite pixels *inside the annulus*, not over the frame: with one roll and
  no references the old check reported "finite 0.0%" and then said "check passed".

## Unreleased — 2026-09-21
- **The MIRI driver NaN'd the dead zones into the cube it was about to high-pass.**  Found while
  chasing the F1140C crash below, and it was *not* the cause of it — but it is the same mistake
  and would have caused it on its own.  `run_miri.py` set the 4QPM dead zones to NaN
  (`miri.apply_quadrant_mask`) before the reducer, which high-passes at `nan_aware=False`; that
  is `ndimage.uniform_filter`, a running-sum filter, so a NaN poisons everything *downstream of
  it along each axis* rather than a box the filter's width — one NaN near the corner of an 81×81
  frame takes 73% of it, and the dead zones are lines through the star reaching all four frame
  edges.  `np.nansum` of an all-NaN frame is 0.0, `bin_frames` drops zero-sum bins, and the
  reduction is left with no frames.  No crop fixes it — the dead zones cross the middle of the
  array.  The pixels are attenuated measurements, not missing ones, so they now
  stay in the cube and in the KLIP basis and are excluded from the *statistic* instead:
  `miri.dead_zone_pixel_mask` carries the detector geometry through every roll into the
  de-rotated frame and goes in as `MawetPeakSNR.pixel_mask`, paired with the `forbidden_pa`
  sectors it agrees with by construction (100% agreement on the ring `forbidden_pa` is evaluated
  on, measured against the real F1140C map: 8 sectors of ±2° at PA 5/85/95/175/185/265/275/355
  for a two-roll sequence, 11.4% of the search annulus excluded).  Note the sign of the bias
  this fixes: where the phase mask takes the starlight it takes the speckles too, so a dead-zone
  pixel is *quieter* than its ring — leaving it in depressed σ, inflated every S/N, and gave the
  optimizer an incentive to choose parameters that preserved the dead zones.  Taking it out
  raises σ and lowers the reported contrast.  `apply_quadrant_mask` survives for a reduction with
  no high-pass at all, behind `--nan-dead-zones`, with what it breaks written on it;
  `--no-mask-quadrants` is now `--no-dead-zones` (old spelling still accepted).
  `tests/test_run_miri.py` (the one-NaN propagation measurement, the cube arriving finite, the
  mask agreeing with the sectors, `--nan-dead-zones` costing most of the frame, and the reducer's
  guard) and `tests/test_miri.py` (`dead_zone_pixel_mask` geometry, rolls, and the empty-angles
  case masking only the unsampled core).

## Unreleased — 2026-09-17
- **The calibration k-scan was choosing the seed's k by noise.**  `Runner.calibrate` scanned k once
  at the *starting* contrast (`contrast0`, 3e-5) and only then walked the contrast into the S/N 4–6
  window — faithful to `optimize_near_2_tpe`, whose starting contrast was NEAR's per-annulus
  `use_contrast` and therefore already close.  On the public data sets 3e-5 is orders of magnitude
  from the calibrated values (8.8e-3 … 3.5e-6), the sources scored S/N ~0 (β Pic) or ~40 (HD 95086)
  at the scan, and `argmax_k` of that curve was a coin: the seeds of runs A2, B2, C and D came out
  k = 4/6/13, 8, 4/1 and 18/6 — a different "default" per annulus, and the k each contrast was
  calibrated at.  The scan now runs once the contrast has reached the window at the configured
  default k (median over `n_remeasure` draws instead of one), and the window is re-measured at the
  k it picks, with a full trial budget to walk the contrast back if the S/N moved.  A forced
  contrast still scans at that contrast and keeps its single trial (the benchmark slots are
  unchanged).  `calibration.json` gains `k_default_initial`, `kscan_contrast` and a `k` per trial;
  the log says `k-scan at contrast …`.  `tests/test_runner.py::
  test_the_k_scan_happens_at_the_calibrated_contrast_not_the_starting_one`.  **Runs A2, B2, C and D
  are to be redone under this protocol** (`FORCE=1 ./rerun_paper.sh A2 B2 C D`); E2 and F2 also
  predate the rank-collapse fix of 16037ac and go after them.
- **`collect.py` measured a default no run ever evaluates.**  Its "default" was
  `space.default_vector()` unprojected — angsep 1.923 λ/D, anglemax 26° — while the Runner seeds
  the guard-*projected* default (angsep 0, anglemax = the PA span, k capped) at the calibration's
  k; and it compared that with the run's validated score from a different set of draws, whose
  draw-to-draw scatter is a factor ~1.4.  collect now re-scores the projected seed (at the run's
  `k_default`), the configured default (k from `RunConfig.defaults`, before the k-scan) and the
  validated winner on the SAME `N_TRIALS = 8` injection sets with the validation metric, and
  records `winner_remeasured`, paired `gain`, `paired_wins`, `gain_vs_validated`,
  `default_flat_*`; the anchor check uses the companion annulus' projected seed.  `figs.py` scales
  the default curve by the paired gain.  On the reruns of 2026-09-17/18 (all four science runs under
  the new calibration), the paired gain is ×1.38 / ×1.73 / ×1.14 (A2), ×1.64 (B2), ×1.13 / ×1.30 (C)
  and ×1.21 / ×1.67 (D) — the winner better on **8 of 8** injection sets in every annulus — and the
  companion checks land at 0.91, 1.10, 1.29 and 1.13 times the published contrast (+0.10, −0.11,
  −0.28, −0.14 mag), against 1.55 / 3.03 / 1.23 / 1.10 before.
- **A calibration that could not see its sources ran away; now it changes k, then stops.**  Run A2's
  [6, 12] px annulus (three sources four FWHM apart on a 9-px ring, every reference frame holding the
  source): at k = 10 the median S/N sat at 0–1 from 3e-5 to 76, the walk went on ×10 per trial, the
  re-calibration revisit pushed it further, and the run searched at contrast **4.6e+03** — a source
  4,600 times the star.  (The 2026-09-13 A2 had done the same to 88, and only the four ×0.1 revisits
  brought it back to 8.8e-3.)  `CalibrationConfig.max_contrast = 0.1`: when the contrast is about to
  pass a tenth of the star, the k-scan is asked whether *any* k detects the sources at the last
  measured contrast — at that radius k = 1 sees them at S/N 7–9 where k = 10 gives 1 — and the walk
  restarts from the starting contrast at that k (A2 annulus 1 calibrated at 1.7e-3, k = 1, in the
  rerun, and its winner went 8.89 → 12.26).  If no k detects them either, the contrast stays AT the
  cap, the annulus is marked `uncalibrated` in `calibration.json`, the log says what that costs, and
  the search runs anyway — a default configuration that cannot see an injection is a statement about
  the default and not about the problem, and the synthetic RX J0534 fixture is exactly that (an
  earlier version of this guard raised there instead, and took all 18 of its end-to-end tests with
  it).  `collect.py` reads the flag and refuses to let such an annulus into the table unremarked.
  The revisit caps at the same value, as IDL's `cmax_cal` did.  `RunConfig.n_sources` now takes a
  per-annulus list, which is the other way out.  Four tests in `test_runner.py`.
- **`rerun_paper.sh` runs one instance, one run per directory.**  Launched under `nohup` it prints
  nothing, which read as "it didn't start" and got it started again 34 s later; the second instance
  found a 34-second-old `A2_betapic` (no `final_results.json` yet, so no skip) and launched a second
  A2 into it — two searches appending to one `results.jsonl`.  A pid file refuses a second instance,
  no stage starts into a directory whose heartbeat is younger than five minutes (checked before
  anything could retire it), and the header says to follow with `tail -f rerun_paper.log`.
- **`FORCE=1 ./rerun_paper.sh <science stage>` did not redo anything.**  It got past the
  finished-stage skip and launched the stage into its existing directory, where `Runner.run()`
  auto-resumes the checkpoint, finds every annulus complete, rewrites the products and reports
  "done in 2 min".  A finished science run is now retired to `<dir>_superseded_<stamp>` like a
  benchmark batch; I2 (days of LMIRCam compute) is never retired by a flag.  `DRY=1` says what
  FORCE would do.  `tests/test_paper_runs_stages.py` (+3).
- CI: the two RX J0534 tests that ran the default band on the 48-px tree now use the 140-px one
  (the all-failed guard stops such a run, correctly); `test_display_pdfs_never_ask_freetype_for_u_fffe`
  is matplotlib-3.11 aware.  Slow job green on Python 3.12 / numpy 2.5 / matplotlib 3.11.

## Unreleased — 2026-09-16
- **`load_calints` was median-filtering the planet.** Its repair step was a 5×5 median filter plus a
  7σ clip against the *frame-wide* robust scatter of the residual — a scatter set by empty sky, which
  every structured pixel of a coronagraphic PSF exceeds.  On ERS 1386 F444W it rewrote ~4,600–5,500
  pixels per 320×320 frame, of which only 1,564 were DQ-flagged; the rest was the star's Lyot pattern
  and HIP 65426 b, median-filtered.  The companion came out as one smeared blob at 39% of its peak
  (7.3 vs 18.6 MJy/sr) instead of the three-bar "hamburger" core of Carter et al. (2023)'s Fig. 3, and
  because the set of rewritten pixels depends on each frame's own values, a residual that differed
  between the two rolls was left beside it and was diagnosed — twice — as a speckle.  The repair now
  fills DQ `DO_NOT_USE` pixels from their eight neighbours (spaceKLIP's treatment, new
  `fill_dq_neighbours`) and touches nothing else; `repair='sigma'` (`sigma_clip_repair`) is kept,
  with a loud warning, only to reproduce old runs.  `scripts/check_hip65426_fig3.py` reproduces
  Carter's Fig. 3 panels from the MAST calints (one annulus, one subsection, ADI 2 / RDI 18 /
  ADI+RDI 20 modes) and injects the STPSF off-axis PSF at their published flux: F_measured/F_Carter
  = 1.01 ± 0.10 (ADI+RDI) and 1.16 ± 0.13 (RDI).  `tests/test_calints_repair.py` pins it.
- **`optics_transmission` for HIP 65426 is 1.0, not 0.561 — and the Carter comparison is now a check,
  not a calibration.** Fakes are injected *after* the repair, so they kept their cores while the
  companion's had been flattened: the companion looked 1.9× too faint relative to them, and the 0.561
  "anchored on HIP 65426 b" was 1/1.9.  `PHOTMJSR` for `PUPIL=MASKRND` is derived from standards
  observed through the coronagraphic optics, so the calints already put an off-mask source at its
  true flux; the earlier "no occulting-mask column in photom" argument confused the occulter
  (spatially varying, not in photom) with the substrate (uniform, in it).  With the repair fixed and
  nothing tuned, `scripts/check_hip65426_contrast.py` gives **ΔF444W = 8.735 ± 0.094 against Carter
  et al.'s 8.703 ± 0.055**.  `star_flux_from_flux_density` now warns when the value is *not* 1.
- **HIP 65426's star centre re-solved on clean frames**: (149.65, 172.96), 0.78 px from CRPIX, the two
  rolls agreeing to 0.12 px — against Carter et al.'s F444W astrometry (820 mas, 149.9°).  The old
  (150.54, 172.98) was the same solve on median-filtered frames, whose smeared companion peak had
  moved; the rolls disagreed by 0.71 px then, which was the tell.
- **H2's forced contrast re-measured**: `1.637e-04` (median S/N 4.05) on DQ-filled frames with
  `T_optics = 1` and the new centre, replacing `2.324e-04` (`scripts/calibrate_bench_contrast.py H2`, new entry).  **Every
  archived HIP 65426 result — runs D, H2, the psf-shape investigation and its figures — was made on
  median-filtered frames and is void.**  `scripts/check_hip65426_psf_shape.py` is removed; its
  "roll-2 over-subtraction" finding was the repair.
- **Fixed**: the STPSF cache was invisible across the two `KLIP_TPE_DATA` layouts.  `cache_dir()` is
  `$KLIP_TPE_DATA/stpsf_cache` when set, else `~/.klip_tpe/stpsf_cache`; a grid computed with
  `KLIP_TPE_DATA=~/.klip_tpe/data` therefore sat in `~/.klip_tpe/data/stpsf_cache`, where a shell
  without the variable never looked — and on a machine without STPSF that stopped paper run D at
  start-up ("the grid is not in the cache").  Reads now try both layouts (`_cache_path`); writes
  still go to `cache_dir()`.
- **A run that cannot reduce now stops instead of finishing.**  Paper run D on 2026-09-17: pyklip 2.10
  calls `numpy.reshape(copy=False)` (numpy ≥ 2.1 only, undeclared) against an older numpy, so every
  reduction raised the same TypeError; the Runner logged 350 failures, wrote a `final_results.json`
  with `winner_index -1`, and `rerun_paper.sh` said "done in 3 min".  Now: `PyKLIPReducer` refuses to
  construct on that pyklip/numpy pair, naming both remedies (`_check_pyklip_numpy`); `Runner.calibrate`
  raises when the default reduction fails on every attempt of its first trial; `_finish_annulus`
  raises instead of logging when no evaluation succeeded; and the driver retires a stage whose
  `final_results.json` has no winner to `<dir>_failed_<stamp>` rather than skipping it (after the
  `DRY` check).  `tests/test_all_failed_run.py`.
- **Notebook warnings.**  `draw_walk` and the parameter-history page set identical axis limits on a
  pinned dimension (one matplotlib UserWarning per cell per page; the KDE corner already widened them
  via `_widen_flat`, these two now do too).  The "Glyph 65534 (\ufffe) missing from font" lines were
  matplotlib's Type 3 font embedding asking FreeType for cp1252's five undefined slots, whose own
  `catch_warnings()` guard a concurrent `catch_warnings()` on the reducer's thread wiped at random;
  the display and paper-figure rc now embed TrueType (`pdf.fonttype 42`), which never builds that
  table and gives the PDFs real, selectable text.  Two regression tests in `test_display.py` (the
  Type 3 leak is a matplotlib ≤ 3.10 phenomenon — 3.11 builds the widths from the font's charmap and
  no longer imports `warnings` in `backend_pdf` — so on 3.11+ the test only checks that the rc
  selects Type 42 and the PDFs stay warning-free).
  pyKLIP's `klip_parallelized` draws a tqdm bar per call whatever `verbose` says — in a notebook with
  ipywidgets that is one widget per reduction (tutorial 03: 426 of them, 3.7 MB of widget state) —
  so the backend swaps pyklip's `trange`/`tqdm` for disabled ones (`KLIP_TPE_PYKLIP_PROGRESS=1`
  keeps them).  All four notebooks rebuilt: no warnings, no widgets.
- **pyKLIP ADI / ADI+RDI were subtracting each frame from itself.**  pyKLIP selects references with
  `moves >= movement` and nothing else, so at `angsep = 0` (`movement = 0`) the target frame — and every
  frame at its PA — sat in its own KL basis, and a companion came back at round-off (run D: peak 3e-7
  in ADI, 2e-6 in ADI+RDI, 2.8 in RDI).  The paper's "RDI 7.8 / ADI+RDI 1.7 / ADI 0.03" and tutorial
  03's "ADI −0.2 / ADI+RDI 1.2" were this, not the other roll eating the planet.  `movement` now floors
  at `MIN_MOVEMENT_PX = 1e-6`, which excludes exactly the zero-motion frames (the frame and its
  same-roll twins).  `test_pyklip_adi_at_angsep_zero_does_not_subtract_the_frame_from_itself`.
- **`load_calints(partition='all')`.**  A partition is reduced on its own, so with one partition per
  roll every frame in it shares a PA: ADI has no references and ADI+RDI *is* RDI — the other roll is
  never in the basis.  `'all'` puts both rolls in one partition, where pyKLIP's `mode` is a real
  choice (ERS 1386 F444W, k=10: injected S/N 6.2 ADI, 6.9 RDI, 7.2 ADI+RDI).  Default stays `'roll'`.
- **`paper_runs/collect.py`**: the "default" column now uses the vector the run was seeded with
  (`RunConfig.defaults` on top of the space's defaults — k=10, not the space's k=6); the companion
  anchor compares matched-filter *peaks* (fakes in inj−clean) instead of S/N values, which the
  NIRCam PSF's lobes in the noise ring biased by ~1.9×, and records `flux_scale` as a check without
  rescaling the axis unless `ANCHOR_APPLY=1` (`figs.py` follows `flux_scale_applied`).  Earlier
  β Pic / HD 95086 anchor values were made with the S/N method and must be re-collected.
- Tutorial 03's `repair()` cell — the version students copy — now fills DQ pixels only, and the
  text says why a value-based outlier filter must never be run on a coronagraphic PSF.  The notebook
  now builds ONE partition with both rolls (section 2 explains why per-roll partitions make `mode`
  meaningless) and its mode comparison is real: at k=10 in [6, 45] px the companion is at S/N 13.3
  (ADI), 12.8 (RDI), 13.2 (ADI+RDI); the 50-evaluation search elected ADI with a strong high-pass
  filter.  `docs/BACKENDS.md`, `docs/TUTORIALS.md` and `docs/FLUX_CALIBRATION.md` corrected likewise.

## Unreleased — 2026-09-15
- **Flux calibration audited end to end** (`docs/FLUX_CALIBRATION.md`): what PSF is injected, what the
  star flux is and over what aperture, for every observation the package ships or the paper uses.  Three
  axes were wrong and are fixed, each checked against the companion's published brightness:
  - **β Pic** — the distributed `naco_betapic_psf.fits` is a *normalised* template (unit flux inside
    r = 2.000 px), so its own counts are not the star; unset, `star_flux` put the axis 9.4e5 from a
    contrast.  Now `3.3268e6`, converted from VIP's published `starphot` by the new
    `generic.star_flux_from_aperture_photometry` + `datasets.PHOTOMETRY`.  **Verified**: β Pic b measures
    ΔL′ = 7.81 ± 0.08 against Absil et al. (2013)'s 8.01 ± 0.16 from the same data — 1.1 σ
    (`scripts/check_betapic_contrast.py`).
  - **HD 95086** — the SPHERE flux frames are already on the science frames' scale; applying
    `dit_science/dit_flux/nd_transmission` a second time over-counted the star by 1347×.
  - **HIP 65426** — had no stellar photometry at all (`flux_unit = 1.0`, `flux_scale = 2.35e5`).
    It now has a full chain (see below), and runs D and H2 build it rather than falling back to a
    Gaussian.  **Runs D and H2 have to be redone**: both the flux scale and the star centre changed.
- **`star_flux_from_halo` removed**, with no opt-in (also the `--star-flux halo` CLI value).  Fitting an
  off-axis template to a *coronagraphic* halo compares two different functions: the same method on the
  same data gave star fluxes 5.28× and 8.04× apart on paper runs A2 and B2.  A PSF template is now
  **required** — the old `GaussianPSF(flux_unit=1)` fallback silently called raw detector units a contrast.
- **HIP 65426 / NIRCam coronagraphy — the flux chain, and the term neither side supplies.**
  New `stpsf_psf.unocculted_ee` and `stpsf_psf.star_flux_from_flux_density` turn a stellar flux
  density into `flux_unit` for MJy/sr JWST data: `S / (10^6 PIXAR_SR) x EE x T_optics`, with the
  occulter's `T(rho)` deliberately left out (it multiplies `flux_unit` inside `inject_sources`).
  The `EE` is of the **unocculted-through-the-Lyot-stop** PSF, not an imaging one.  `T_optics`
  covers the *transmissive* losses of the coronagraphic optics (COM sapphire substrate, BaF2 Lyot
  substrate): STPSF's `normalize='first'` models only *diffractive* losses, and the NIRCam `photom`
  reference file has no occulting-mask column ([jwst#10309](https://github.com/spacetelescope/jwst/issues/10309)),
  so neither the model nor the pipeline carries it -- a factor of ~1.8 on the contrast axis.
- **New**: `spaceklip.load_calints` -- stage-2 `*_calints.fits` into partitions without spaceKLIP,
  with an **odd** crop (an even one leaves the star half a pixel off in each axis) and a
  `star_center=` override.  `CRPIX` is the aperture reference point and misses HIP 65426 by 1.48 px.
- **New**: `generic.aperture_sum` (exact partial-pixel circular photometry, no photutils dependency) and
  `generic.star_flux_from_aperture_photometry`.
- **Fixed**: `inject_sources` builds the cube in float32, so an injected stamp below the float32 quantum
  of the science pixels it lands on was silently rounded away — a quarter of the flux at 1e-4 counts, and
  worse as the contrast falls, which bends every curve derived from it.  Now a `RuntimeWarning` naming
  `flux_unit` as the likely cause.  No paper run was affected.

## Unreleased — 2026-09-13
- **Public release preparation**: README rewritten for a general audience, `docs/TUTORIALS.md`,
  `docs/PYNOMIC.md` (pyNOMIC start-to-finish guide incl. image groups), `docs/DISPLAY.md` (panel anatomy
  and product glossary), generated `docs/CLI.md` (`scripts/gen_cli_doc.py`), `CONTRIBUTING.md`,
  `CITATION.cff`, CI workflow; private paths removed from the shipped files.
- **Generic cube adapter** (`klip_tpe.instruments.generic`, `klip-tpe generic`): any registered ADI cube
  (+ angles, optional PSF template and reference cube; FITS or arrays, 4-d IFS with `wv_index`) becomes a
  `Dataset` → reducer → space → guard → objective.  Data-derived frame-quality tags (`quality_tags`),
  per-partition wavelength / FWHM dicts (IRDIS K1/K2).  (This entry originally listed a
  `star_flux_from_halo` helper for saturated cores; it was removed on 2026-09-15, see above.)
- **Example data** (`klip_tpe.datasets`): `naco_betapic`, `sphere_sao206462`, `nircam_pds70_*` (VIP_extras)
  and `sphere_hd95086` (SPHERE IRDIS K1+K2 crops, klip-tpe release), downloaded on first use into
  `~/.klip_tpe/data` (`$KLIP_TPE_DATA`).
- **Tutorials** (`tutorials/`, authored as `.py` with `# %%` markers, built and executed by
  `tutorials/_build_notebooks.py`): 01 NACO β Pic end to end (+ VIP / pyKLIP engines), 02 SPHERE HD 95086
  partitions, 03 JWST NIRCam HIP 65426 via spaceKLIP/pyKLIP (+ `tutorials/fetch_jwst_hip65426.py`).
- **JWST / pyKLIP fixes found while building tutorial 3**: `load_spaceklip(partition_by="roll")` is the new
  default (one partition per position angle; `filenums` gave one partition per exposure, often a single
  frame) and the pixel scale is read from the science header when pyKLIP's reader does not carry it;
  the pyKLIP and VIP backends default to `pool="threads"` because they fork their own workers and a
  daemonic worker may not have children.  Backend options are searchable as ordinary parameters — the
  tutorial searches pyKLIP's `mode` and the optimizer picks `RDI` over `ADI+RDI` for a 10-degree roll.
- **Known sources**: `known=[(rho, pa), ...]` (already honoured by the position sampler and the metric) is
  now also excluded from the noise rings of `products.noise_profile` / `contrast_curve`, and `Runner`
  forwards the objective's list automatically — a detected companion no longer inflates the 5-sigma limit
  at its own separation (58% at HD 95086 b).  New tutorial 4 covers the whole topic, incl. `forbidden_pa`
  and `pixel_mask` for disks.
- **Search ranges scaled to the data** (`instruments.generic.make_space`): temporal binning up to
  `nframes / min_bins` (default 8 bins) instead of the NEAR production 5-30 frames per bin, which
  collapsed short sequences to two bins and made the search degenerate; `bin_range=(lo, hi)` overrides,
  `near.make_space(bin_range=...)` too.
- **Inline Jupyter display**: `LiveDisplay(show="inline")` updates one output cell in place;
  `show="auto"` picks inline inside a notebook and a window otherwise; `--show [window|inline|auto]`.

## Unreleased — 2026-09-12
- pyNOMIC image groups as partitions (`--groups auto | <JD splits> | @labels`, `nomic.load_pynomic(groups=...)`):
  the port of pyNOMIC's `hf.image_groups` star-position split (`nomic.image_groups`) or explicit labels make
  every (night x chop state x group) a partition `<name><A|B>g<k>`, so nights and groups go through the
  same selection / per-partition-block / display machinery; `--group-min-frames`, `--group-smooth`;
  `make_space(max_drop=...)` / `--max-drop` sets the number of drop slots for many partitions.
- Progress movie while the run goes: `annulusNN/progress.gif` + `.mp4` rebuilt every `--movie-every`
  evaluations (default = `--pdf-every`, IDL's `annNN_progress.gif` cadence) and at annulus end, on the
  render thread; movies are thinned to 400 evenly spaced frames (last kept) so long annuli stay bounded.
- Display: validation trials now update the live/step display (IDL's `valid` frames):
  `RunCallback.on_validation_trial` fires after every trial with the trial's images, fresh sources and
  score; the panel shows them in the Test cells (title `Valid c/n trial t/n (eval e)`, green header
  `VALIDATION ... median so far`) and the frames join the progress movie.
- Display: the elapsed/ETA block carries IDL's full estimate — `ETA(ann)`, `ETA(full)` and `total`
  (elapsed + remaining) computed as in `optimize_near_2_tpe`: remaining evals of this and future annuli
  at the current per-eval loop time, plus the measured calibration + validation time per annulus for
  each future annulus, this annulus' pending validation (`cal_avg · n_top · (1 + n_valid) / 3` until one
  is measured; per-trial time × trials left during validation) and a reserve for still-available
  re-calibration restarts.  (`ETA(ann)` used to be the remaining-evals-only figure for the whole run.)
- Display, single-night runs: the night-inclusion, night-effect and S/N-vs-#nights panels are dropped and
  IDL's `mwcm_pimppanel` parameter-importance bars (|corr| with S/N, strongest first) take the inclusion
  panel's place; the ETA block reads `Est. total` and adds a `done <clock time>` line.
- Display: evaluation axes of the night-inclusion panel (and the trace / partition-S/N books) grow with
  the evaluations done in the current annulus, `xrange=[0.5, nev+0.5]` with IDL's `near2m_evtick` tick
  rule, so they start over at every annulus instead of spanning the annulus budget.
- Display: live window at IDL's size (1850 × 990, 1:1, `--window-scale`), `last` eval time under `avg`,
  corner bottom-row marginal rotated to share the row's y axis, contrast panel keeps the validated
  winners of earlier annuli, KLIP-FM model square symlog, injected-PSF square derotated by true north.
- Resume with the live window: `on_setup` is emitted on a resume too (window opens at once with the run's
  newest frame, no toolbar), and a restart in the middle of an annulus restores the earlier evaluations of
  that annulus from `results.jsonl`, so the panel shows the whole annulus rather than the post-restart evals.
  The launcher accepts lowercase settings (`show=1`, `smoke=1`, ...).  The incumbent's images are not
  checkpointed, so a resume re-reduces the best-so-far evaluation once (same config, positions and
  contrast; the logged score stands) to refill the Best cells, the KLIP-FM preview and the fallback products
  instead of showing `collecting...` until the next new best.
- Resume without the run id: `scripts/run_near2_production.sh resume` (or `resume last`) and `klip-tpe
  resume` / `extend --run-dir last` (the CLI default) pick the newest `run_*` under `<root>/comb/opt`;
  an explicit run directory or name still works.
- Launcher: `SMOKE=1` end-to-end test preset (25 warm-up + 50 TPE per annulus, 5 candidates × 10 trials).
- IDL calibration display phase (opt-in `--fun` / `LiveDisplay(fun=True)` / `ALIENS=1`): the five-act launch movie (`klip_tpe/intro.py`, port of
  `near2m_intro_frame`) plays in the live window during the first annulus' calibration (animated from the
  main thread while it waits on the workers) and is written as `intro_frames/` + `intro.gif`; every
  calibration trial draws the `near2m_calshow` panel (clean | injected, per-source S/N, trial / contrast /
  target line) to `steps/calib_annNN_trialNNNN.png` and the window (`RunCallback.on_calibration_trial`);
  the calibration frames lead the progress movie like IDL's.
- Reduction: `rotate_ccw` now treats NaN neighbours as 0 in the bilinear stencil (`klip.ROT_NAN_MODE =
  "zero"`), which is what IDL `rot(/interp)` does on the production images: the 1–2 px rim of the padded
  KLIP zone is damped like IDL's instead of a noisy few-frame ring (rim std 0.027 → 0.007 vs IDL 0.006 at
  r 21–22 px on eval 1826).  Images inside the zone are unchanged; scores move by a few % because the
  matched-filter noise apertures near the outer sources reached into the rim.  `"propagate"` restores the
  strict semantics.
- Display: no annulus-edge circles on the image cells (IDL has none), injected-PSF square uses the image
  colour map, panel 17 × 8.6 in (946 px at 110 dpi, fits 1080p with menu bar and dock), rows shifted up,
  no toolbar under the live window.
- Production launcher: `scripts/run_near2_production.sh` and `notebooks/near2_production_run.ipynb` run the
  IDL `run_20260906_173000` protocol from a terminal / Jupyter (docs/RUNNING.md); `--run-dir` defaults to
  `<root>/comb/opt/run_YYYYMMDD_HHMMSS` (IDL layout).
- Parallelism: `--workers auto` (default) uses every core — forked worker processes sharing the loaded data
  (`klip_tpe.parallel.ProcessPool`, the IDL bridges; `--pool threads` for the single-process version),
  injected and clean reductions concurrent, partitions mapped onto the workers, the remainder as per-target
  threads inside `klip_annular` (`KLIPParams.threads`), BLAS pinned to one thread per worker (threadpoolctl).
- Validation checkpoints after every trial (`val_candNN_trials.pkl`) and resumes mid-candidate.
- Live/step display redrawn to the IDL `near2m_show` layout (`render_step(style="idl")`, default): black
  live window, 5+5 image/S-N cells with `raw/corr` per-source labels and the stitched cells, convergence
  trace with SMA20/50 + rotated S/N histogram, KLIP-FM and injected-PSF squares, night inclusion, SNR=5
  contrast with the FM curve dashed, BEST/TEST parameter vectors, elapsed/ETA, night-effect panels and the
  scatter corner on the right; a white snapshot (`annulusNN/step_display_white.png`) at annulus end.
  The previous layout is `style="classic"`.  Runs now save per-eval image crops
  (`annulusNN/evals/evalNNNN_{inj,clean}.fits.gz`, `RunConfig.save_eval_images`, like IDL's
  `evalNNNN_score_inj.fits.gz`) so `render_steps` rebuilds complete frames post hoc; `fm_preview`
  (live KLIP-FM at each new best) is now on by default (`--no-fm-preview`); the annulus-end hook
  writes all books (importance / paracoord / rank / slice / verify / products) and survives a resume.
- Display parity with the IDL figure set: `plot_annulus_books` now also writes `importance.pdf`
  (η² parameter importance), `paracoord.pdf`, `rank.pdf`, `slice.pdf`, `products.pdf` (final images /
  S/N maps / stitches / per-partition winners) and the verify books `verify_limits.pdf`,
  `verify_subsets_inj.pdf` (recomputed from the saved partition stacks).  IDL↔Python product mapping
  and side-by-side sheets: `KLIP-TPE/idl_vs_python/displays/`.
- `klip-tpe compare` / `klip_tpe.idl_compare`: live incremental comparison against a running IDL run.

## 0.1.0 — 2026-09-11 (first shared version)
- Port of the IDL `optimize_near_2_tpe` / `reduce_near_2` optimizer: TPE (univariate default,
  `pbest`=0, `blocks` option), calibration → search → validation protocol, feasibility projections,
  KLIP/ADI/RDI reference reducer with KLIP-FM, injection models, Mawet small-sample metric, products
  (contrast curve, FM cross-check, stitching, S/N maps), verification stack (verify / param_verify /
  candidates), live display + movies, checkpoint / resume at any phase, `extend`, `opt_width`, scan k-modes.
- Instrument adapters: VLT/VISIR NEAR (`instruments/near.py`), LBTI/NOMIC via pyNOMIC
  (`instruments/nomic.py`).
- PSF-subtraction backends: pyKLIP, VIP; spaceKLIP/JWST ingestion; custom-pipeline wrappers
  (`FunctionReducer`, `ExternalReducer`).
- Cross-validation against IDL documented in the (internal) IDL findings notes (incl. the `parstr` frame-selection
  bug found in the IDL production code, now fixed there).
