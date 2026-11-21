# A known companion's position and contrast: `klip-tpe extract`

`klip-tpe extract` fits a known companion in a finished run.  It gives the companion's separation,
angle and contrast with two sets of 1-sigma uncertainties, the bias and spread of the fit from fake
sources, and for data in wavelength groups, such as ALES's, the contrast per group with the position
held.  It ports the IDL predecessor's `subtract_mwc758c_tpe.pro` and `subtract_profile_scan.pro`,
which fitted MWC 758c on JWST/NIRCam data the same way.

The code is `klip_tpe/extract.py`.

## Running it

On a run made with the command line:

```
klip-tpe extract --run-dir RUN --rho 0.286 --pa 25.65 --contrast 5.5e-3 --label B
```

The data options come from the run's record, as `klip-tpe elect` takes them, and the run is only
read.  The output goes to `RUN_extract_B` beside the run (`--out` to choose), with its log in
`extract.log`.

On a run made by a script, resume the runner with the script's reducer and call the step:

```python
from klip_tpe.extract import ExtractConfig, extract_companion, write_extract
from klip_tpe.runner import Runner

runner = Runner.resume(run_dir, reducer, objective, sampler)    # as the script built them
found = extract_companion(runner, 0.286, 25.65, 5.5e-3, config=ExtractConfig(seed=1))
write_extract(found, run_dir + "_extract_B", label="B")
```

## What it does

1. **The configuration.**  The elected winner of the annulus that holds `--rho` reduces
   (`--annulus N` for another annulus, `--config EVAL` for another evaluation, numbered as the run's
   log numbers it).
2. **The residual.**  Each trial puts the companion's negative into every frame and reduces.  The
   reduced image is read in a fixed disk of `--patch` FWHM (default 2) around the starting position.
   The disk's pixels, less their mean, give an L2 norm, divided by the same norm of the reduction
   without the injection.  0 is a perfect null and 1 is no cancellation.  The residual is
   deterministic, so no trial is drawn again.
3. **The search.**  A TPE proposes `(rho, pa, contrast)` in a box around the start and maximizes
   `-log(residual)`, `--n-init` random trials first and `--n-iter` in all.  The box is 0.5 FWHM either
   side in separation, 0.5 FWHM along the arc in angle and 0.5 to 2 times the starting contrast
   (`--box-rho`, `--box-pa`, `--box-contrast`).
4. **Uncertainty 1, the quadratic.**  A full quadratic in the three parameters, each scaled by its box
   width, is fitted to the best 40 % of the trials (`--best-frac`).  With `H` its Hessian and the floor
   the rms of the fit's residuals, a parameter's sigma is `sqrt(2 floor (H^-1)_ii)` times its width.
   That is where the residual has risen by the floor.
5. **Uncertainty 2, the profiles.**  Two parameters are held at the best and the third is stepped over
   `4 nref + 1` points (21 for `--nref 5`) across 4 times its first sigma, at most half the box.  A
   parabola per parameter gives its sigma at the same rise and its own minimum.
6. **The recovery check.**  `--fakes` sources (default 5) at the fitted separation and contrast, at
   angles spread evenly from the companion's, go into the data with the companion taken out at its
   fit.  Each is fitted like the companion, from a start as far off as the companion's was, with
   `--fake-iter` trials and no profiles.  Their mean offsets are the bias with its standard error,
   and their standard deviation is the spread of a fit.
7. **Wavelength groups.**  With several partitions the position comes from all of them combined.
   Each partition's contrast is then fitted alone with the position held.  A coarse scan of 11 points
   from 0.2 to 2.2 times the combined contrast, moved once when its minimum falls on an edge, is
   followed by a profile of `4 nref + 1` points and its parabola.  With fakes, each partition also
   gets its own recovery check: fake sources at its fitted contrast, at the same angles as above, with
   the companion taken out at that contrast, each fitted the same way with its position held.  The
   contrasts with their sigmas are the spectrum.  `--no-groups` skips them.

## Which sigma to quote

The quadratic and the profiles measure how sharply the residual rises around its minimum.  They do
not measure the speckle noise in the patch, which moves the minimum itself, and they shrink as the
trials crowd the minimum.  The fake sources measure that spread directly.  On beta Pic b
(`scripts/check_betapic_extract.py`, the defaults):

| | rho | pa | contrast |
|---|---|---|---|
| fit | 0.4516" | 210.64 deg | 7.17e-4 (dL' 7.86) |
| #1 quadratic | 0.0007" | 0.046 deg | 0.9 % |
| #2 profiles | 0.0002" | 0.013 deg | 0.2 % |
| 5 fakes: scatter | 0.0023" | 0.15 deg | 4.3 % |
| 5 fakes: bias | -0.0001 +- 0.0010" | -0.13 +- 0.07 deg | +3.0 +- 1.9 % |
| Absil et al. 2013, these data | 0.452 +- 0.010" | 211.2 +- 1.3 deg | dL' 8.01 +- 0.16 |

The fit lies within 1 sigma of the published values in all three, and the fakes' scatter is 3 to 20
times either estimator.  The summary's `to quote` line therefore gives the larger of the profile sigma
and the fakes' scatter, per parameter, and each partition's contrast carries the larger of its
profile sigma and its own fakes' scatter.  With 5 fakes the scatter is itself uncertain by about a
third.  More fakes, or a second check with another seed, firm it up.

## Which configuration

The annulus' elected winner reduces unless another is named.  A negative injection cancels the
companion in every frame before the PSF subtraction, so at the null no reference frame, ADI's
included, holds any of it, and the residual's minimum lies at the companion in any configuration.
Away from the null, an ADI basis holds part of the mismatch and subtracts it, which flattens the
residual around the minimum.  The sigmas then come out wider, as they should, and the recovery check
measures the bias that remains in the configuration used.  The IDL predecessor fitted MWC 758c in
its production ARDI configuration the same way.

A configuration whose references never hold the companion (RDI) gives a sharper minimum, but not
always a lower floor.  Name it with `--config` to compare.

## What it writes

| File | What it holds |
|---|---|
| `extract_summary.txt`, `extract_summary.json` | The best values, the position in pixels from the star, both sets of sigmas in arcsec, degrees, pixels and percent, the profiles' minima, the floors, the reductions, the partitions' contrasts, the recovery check and the `--subtract-sources` line |
| `extract_trials.txt` | Every search trial: rho, pa, contrast, residual and how it was proposed |
| `extract_profiles.txt` | The profile scans' points |
| `extract_partitions.txt` | Each partition's wavelength, contrast, the sigma to quote, the profile sigma, its fakes' bias and scatter, and the floor |
| `extract_fakes.txt` | Each fake source's true and fitted values |
| `extract_subtract_sources.txt` | The fitted companion, for `--subtract-sources` in later runs |
| `extract_clean.fits`, `extract_residual.fits` | The reduction without the injection and the best fit's |
| `extract_fit.pdf` | The residual against each parameter, the best trials with their parabolas and the quadratic's sigmas above, the profiles with theirs below |
| `extract_images.png` | Both reductions around the companion, with the patch |
| `extract_spectrum.pdf` | Each partition's contrast with the sigma to quote |

## Cost

A fit takes `1 + n_iter + 3 (4 nref + 1)` reductions, 1064 with the defaults.  Each partition adds
about 30 reductions of that partition alone, and as many again per fake source for its recovery
check.  Each fake source of the combined fit adds `1 + fake_iter` reductions.  These are the larger
part with the defaults, 5005 reductions, so `--fake-iter 300` or `--fakes 3` makes a quicker first
pass.  On beta Pic b's 61 NACO frames the defaults took 4 minutes on 2 cores.

## Notes

- Separations are in arcsec at the run's plate scale, and angles in degrees East of North in the
  image frame.  The step does not calibrate the plate scale or north, which stay with the reduction.
  Contrasts are in the injection model's flux unit.
- During the step the run's injection spectrum is flat, so a fake source and the negative injection
  go into every partition at the contrast given.  The combined fit's contrast is then a weighted
  mean over the partitions, and the partitions' own fits give the spectrum.
- A source of `RunConfig.subtract_known` within 2 FWHM of the start is left out of the reductions,
  since the step fits it, and the summary lists it.  A run whose frames had a source near the start
  taken out by `--subtract-sources` is refused, since there is nothing left to fit.  Give
  `--subtract-sources` a list without it.
- The IDL predecessor searched a box of 0.04" and 5 degrees either side and 0.5 to 2 times the
  contrast on MWC 758c, and read a 12-pixel square on it, about 2.7 FWHM either side of the
  companion.  `--box-rho 0.04 --box-pa 5` sets that box.  The other defaults are its own: 100
  random trials of 1000, gamma 0.15, 24 candidates, 15 % random draws, the best 40 % for the
  quadratic and 21 points per profile.
- `klip_tpe.companion.fit_negative_companion` is the quicker fit without uncertainties, a contrast
  grid and a Nelder-Mead.
