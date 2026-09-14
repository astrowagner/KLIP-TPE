# Findings from cross-checking the Python port against the IDL pipeline

Date: 2026-09-05.  Reference data: `/Volumes/RAID36TB/NEAR2_py` (copy of the six cleaned
NEAR nights), completed IDL run `run_20260818_223040` (annulus 1, 1500 evaluations) and
live products of the paired A/B run `run_20260904_174912`.

## 1. IDL production bug: frame-selection parameters never reach the reducer  (ACTION NEEDED)

`reduce_near_2.pro` dispatches per-night reductions to persistent worker processes by
serialising the call into a command string (`parstr`, lines 2237–2260).  That string
carries `k_klip, angsep, anglemax, bin, inrad, outrad, n_ang, filter, fast, spat_mean,
temp_mean, do_destripe`, the injections, `out_suffix`, `lean`, `klip_scan`, `nthreads`,
`use_near2_throughput`, RDI flags and `n_min_ref` — but **not `corr_thresh`, `noise_max`
or `coronoise_max`**.  Inside each worker `reduce_near_2` therefore falls back to its own
defaults (`corr_thresh=0.95`, `noise_max=2.0`, `coronoise_max=2.0`, lines 346–351)
regardless of the values the optimizer searched.

Consequences for every run with `nbridges > 1` (i.e. all production runs):

* the 18 `corr_thresh_n*/noise_max_n*/coronoise_max_n*` dimensions of the 56-D space are
  dead — they change nothing in the reduction (they do still enter the reference-count
  census / feasibility projection, which uses the *searched* values, so the guardrail is
  computed for a frame set that is not the one reduced);
* `corr_thresh=0.95` keeps only a handful of frames on most nights, so the min-keep rule
  (`>= k_klip*bin`) usually disables selection entirely — the workers reduce **all** frames;
* the "winner" frame-selection values in `final_setup.txt` / `klip_stitched_params.txt`
  are meaningless.

Evidence: reproducing 14 IDL per-night products (`AB_median_klip_pn[cs].fits` harvested
from the running job, whose headers record all other parameters) with the *logged*
thresholds gives clean-image correlations of 0.26–0.93; with the worker defaults the poor
cases jump to 0.91–0.93 (n4, anglemax 21 and k=5 configs) and none get worse.

Fix (one line, no semantic change to anything else): append
`', corr_thresh='+strtrim(corr_thresh,2)+', noise_max='+strtrim(noise_thresh,2)+', coronoise_max='+strtrim(coronoise_thresh,2)`
to `parstr`.  Note this will change the objective for the running A/B pair if applied
mid-run; apply between runs.

The Python port passes the thresholds through (`KLIPReducer.reduce` applies
`frame_selection_mask` with the searched values), so the two implementations will differ on
these dimensions until the IDL fix is in.

**Status 2026-09-05 (post-fix):** the IDL run `run_20260905_102602` forwards the thresholds
(`parstr` line 2272); its log rows now show the searched `corr_thresh/noise_max/coronoise_max`
per night with the no-cut corner (`0, 3, 3`) appearing where the projection snapped them.

### 1b. Keyword audit: `reduce_near_2` signature vs. what `parstr` forwards (addendum 2 request)

Every keyword of `pro reduce_near_2` that is not orchestration (`root, nocomb, nbridges,
seq_range, seq_values, cache, nowait, jobflags, out_suffix`) was checked against the
`parstr` string built in the `else` branch of `if nbridges le 1`:

| keyword | forwarded? | note |
|---|---|---|
| `k_klip angsep anglemax bin inrad outrad n_ang filter fast spat_mean temp_mean do_destripe` | yes | emitted after the serial-path resolution (`outrad < 70`, `inrad` clamp, `angsep` default 1), so children see the resolved values |
| `rho theta contrast`, `fm_rho fm_theta fm_contrast` | yes (when set) | |
| `block_burn aa bb ba ab`, `block_airy` | yes (when set) | |
| `lean`, `klip_scan`, `nthreads`, `use_near2_throughput`, `use_rdi rdi_mode`, `n_min_ref` | yes | |
| `corr_thresh noise_max coronoise_max` | **was missing** → fixed 2026-09-04 | §1 |
| `dthmax` | **not forwarded** | only matters if a caller passes `dthmax`; the optimizer never does today, so children fall back to the same `0.5·FWHM/outrad` rule as the parent. Forward it anyway for safety: `if n_elements(dthmax) gt 0 then parstr += ', dthmax='+strtrim(dthmax,2)` |

Non-keyword settings the children *must* agree on because they are hard-coded in
`reduce_near_2` and therefore identical in parent and child: `bin_type='mean'`,
`comb_type='nwadi'`, `noise_clean`/`coronoise_clean` = 2.0 (always on inside
`near2_framesel`), the two per-frame `smooth(...,filter)` high-passes (pre-KLIP and
post-KLIP), the destripe order (90° then 0°), and the cube crop.  The Python
`KLIPReducer` mirrors all of these (`defaults` dict + `reduce()` chain).

## 2. Metric agreement

Scoring IDL's own combined image for eval 1353 (per-night stack `eval1353_nights_inj.fits.gz`,
uniform mean — identical to IDL's `score_inj` to 3e-9) with the Python `MawetPeakSNR`:
raw per-source S/N `[3.77, 8.83]` (median 6.30), clean-subtracted 5.59; IDL logged 6.59.
The Python matched-filter kernel (azimuthal profile of the measured N4 PSF at the median
separation) and the Gaussian fallback give 6.30 vs 6.42.  Residual ~5–15 % differences
are attributable to the `aper`-vs-pixel ring sampling and de-spike details; the sign,
location logic, small-sample penalty and clean-subtraction clamp behave identically.

## 3. Reduction agreement (per night, same config, same injections)

With the worker-default thresholds:

| config (night) | clean-image corr | noise ratio PY/IDL | injected peak PY/IDL |
|---|---|---|---|
| k=70 bin=27 amax=126 (n2, n4) | 0.58 / 0.77 | 1.08 / 0.96 | 0.91 / 0.74 |
| k=2  bin=22 amax=137 (n2, n4) | 0.65 / 0.64 | 1.16 / 1.09 | 1.07 / 1.03 |
| k=5  bin=17 amax=128 (n2, n4) | 0.84 / 0.91 | 0.89 / 1.12 | 0.80 / 1.05 |
| k=15 bin=25 amax=21  (n2, n4) | 0.63 / 0.93 | 1.60 / 1.11 | 1.04 / 1.01 |
| k=12 bin=17 amax=24  (n2)     | 0.57        | 1.63        | 0.98 |
| k=100 bin=8 amax=33  (n2, n4) | 0.56 / 0.76 | 1.71 / 1.30 | 1.00 / 1.00 |
| k=100 bin=18 amax=29 (n4)     | 0.77        | 1.09        | 0.83 |
| eval 1353 (bin 11, k 12–40, angsep 0, amax 57–88), 5 nights | 0.49–0.64 | 1.5–2.0 | 0.97–1.14 (n4 fast: 2.4) |

**Post-fix re-check (2026-09-05, `run_20260905_102602`, thresholds now honoured on both
sides, 12 products from n2/n4/n6, annulus 0–20 px):**

| night | k | bin | n_ang | filter | angsep | amax | clean corr | rms IDL | rms PY | ratio | inj-diff corr | peak IDL / PY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n2 | 12 | 23 | 5 | 11 | 1.13 | 103 | 0.70 | 0.0487 | 0.0537 | 1.10 | 0.86 | 0.067 / 0.079 |
| n2 | 19 | 30 | 6 | 20 | 0.65 | 47 | 0.81 | 0.0526 | 0.0653 | 1.24 | 0.99 | 0.104 / 0.110 |
| n2 | 25 | 17 | 2 | 9 | 0.03 | 46 | 0.54 | 0.0116 | 0.0223 | **1.92** | 0.96 | 0.021 / 0.022 |
| n4 | 50 | 23 | 2 | 11 | 0.07 | 129 | 0.93 | 0.0584 | 0.0624 | 1.07 | 0.99 | 0.048 / 0.049 |
| n4 | 7 | 30 | 1 | 20 | 0.33 | 35 | 0.96 | 0.1067 | 0.1112 | 1.04 | 0.98 | 0.097 / 0.093 |
| n4 | 25 | 17 | 2 | 9 | 0.42 | 46 | 0.95 | 0.0685 | 0.0734 | 1.07 | 0.99 | 0.086 / 0.087 |
| n4 | 10 | 7 | 2 | 21 | 0.08 | 57 | 0.81 | 0.0483 | 0.0599 | 1.24 | 0.99 | 0.061 / 0.057 |
| n6 | 18 | 23 | 6 | 11 | 0.05 | 130 | 0.56 | 0.0131 | 0.0246 | **1.88** | 0.84 | 0.016 / 0.024 |
| n6 | 1 | 30 | 2 | 20 | 0.00 | 23 | 0.88 | 0.0592 | 0.0687 | 1.16 | 1.00 | 0.096 / 0.097 |
| n6 | 25 | 17 | 2 | 9 | 0.67 | 68 | 0.81 | 0.0493 | 0.0614 | 1.24 | 0.98 | 0.088 / 0.089 |

Injected-source response (difference image) now correlates at 0.84–1.00 with peaks within
±5 % in 8/10 cases, and n4 (552 frames) agrees to 4–7 % in residual noise for every config.
The remaining discrepancy has a clear signature: it appears only when IDL's residual is
very low (rms ≈ 0.012 — long nights n2/n6 with ~110 binned frames, `angsep` ≈ 0 and
k ≥ 18), where Python is 1.9× noisier and the clean-image correlation drops to ~0.55.
For the n2 k=25 case, raising k to 50, setting `angsep=0`, `anglemax=90` or disabling the
cuts moves the Python rms only from 0.0223 to 0.0202–0.0220, so it is not a
parameter-mapping issue; and n_kept/n_binned (1901/112) match what IDL's rules give.
The bisection still needs one IDL intermediate dump (see below).

Planet throughput agrees to ~10 % everywhere except in single-basis ("fast") mode, and the
speckle residual correlates at 0.6–0.9 for wide reference windows.  **Open item:** for
narrow reference windows (small `anglemax` relative to the night's PA span) and for the
auto-fast n4 case, the Python residual noise is 1.3–2× higher than IDL's at the same nominal
parameters (IDL's residual looks like Python's at roughly 1.5–2× the number of KL modes).
Things ruled out by direct experiment: eigenvalue ordering, float32 covariance, mean
subtraction, pre/post high-pass placement, combination weights (nwadi vs mean),
rotation direction / true-north sign, rotation centre (`(n-1)/2` vs `n/2`), bilinear vs
cubic, binning rule, frame selection, self-inclusion of the target in its basis, the
`n_min_ref` drop.  The next diagnostic requires IDL-side intermediates: one serial
`reduce_near_2, seq_range=[2,2], nbridges=1, cache=0, bin=17, k_klip=25, n_ang=2, filter=9,
angsep=0.0316197, anglemax=46, inrad=0, outrad=22, corr_thresh=0, noise_max=3, coronoise_max=3`
run **without `/lean`** (so `AB_cube_klip.fits` is written) plus a `save` of the binned
`angles`, the binned cube and the per-frame `nottarget` counts, to bisect the stage where
the two diverge (`klip_tpe.klip.klip_annular` returns the same intermediates).  Until then, Python and IDL
scores agree in rank (Spearman 0.58, p=0.001 on 29 replayed configurations) but Python is
systematically ~15–40 % lower in S/N on the high-k configurations that win the IDL search.

### 3b. Zone-rim behaviour of IDL `rot(/interp)` (2026-09-12)

On the production images the outer 1–2 px of the padded KLIP zone (r = 21–23 px for annulus
1) are *smoother* than the interior in IDL (std 0.0066 → 0.0059 at r 20–22 px, then 57 % NaN
at 22–23) whereas the port showed a noisy ring there (0.0096 → 0.027 → 0.062).  A synthetic
test reproduces IDL's profile only if the bilinear derotation treats NaN neighbours as 0
(pixels whose stencil is partly outside the zone are damped toward 0 and stay covered by every
frame) rather than propagating NaN (few-frame coverage → noisy ring).  `klip.rotate_ccw` now
does the former by default (`ROT_NAN_MODE = "zero"`); re-reduced evals 1826 / 3051 give rim
profiles 0.0067 / 0.0042 (IDL 0.0059 / 0.0049).  Interior pixels are identical; the scores of
those two evals move by +3 % / +11 % because the matched-filter noise apertures of sources at
0.645" reach r ≈ 21 px.  Whether IDL's `rot` genuinely zero-fills NaN stencil members is
inferred from the images, not from the IDL source (`rot` → `poly_2d` is built in).

## 4. TPE density-model experiment reproduced

`klip_tpe.testbed` (port of `near2_mvtest`) with the IDL settings (56-D, 400 evals,
50 warm-up, 8 seeds, ridge 0.40):

| noise | univariate | block-MV | full-MV | block − uni | IDL (24 seeds) |
|---|---|---|---|---|---|
| 0.25 | 3.34 | 4.34 | 3.44 | +1.00 (7/8) | 3.41 / 4.33 / 3.51, +0.92 |
| 0.40 | 2.69 | 4.05 | 2.95 | +1.36 (8/8) | 2.61 / 3.67 / 3.02, +1.06 |
| 0.25 transposed | 4.73 | 5.60 | 4.84 | +0.87 (8/8) | 4.83 / 5.30 / 4.93, +0.47 |

Same ordering, same magnitudes, same noise trend and the mismatch result survives.  The
Python `TPE` defaults to `blocks="partitions"` (block-multivariate over the per-partition
sub-vectors); `blocks="univariate"` is the byte-for-byte IDL reference sampler.

## 4b. Injection geometry for the two-source annulus (addendum 2 §2b / §2c)

Ported exactly as IDL does it: the injection band is the annulus inset by one FWHM per edge
(`RunConfig.inject_inset_fwhm=1`), the placement routine (`PositionSampler`, unchanged) clamps
the inner edge to 1.5 FWHM and lays the centred ladder; for **two sources** the *caller*
(`Runner._band`) collapses the band to the annulus' area-weighted mid radius
`sqrt((r_in²+r_out²)/2)` so both land there 180° apart (`RunConfig.pair_area_midpoint=True`;
CLI `--ladder-pair` restores the pre-2026-09-05 ladder).  One sampler object serves search,
calibration, validation, param_verify (its fixed set now comes from `_band` too) and the
products.

Check against IDL: annulus 1 ([0, 20] px) injects at `rho = 0.645"` (14.1 px) in both codes
(IDL `run_20260906_173000` calibration setup: 0.645" at PA 308.86/128.86; Python: 0.645" at
185.52/5.52).  The old ladder on the inset+clamped band [9.4, 13.8] px reproduces IDL's
previous 10.5/12.7 px as well.  (An intermediate IDL variant that injected at the feasible-band
midpoint, 0.527", was live for about an hour on 2026-09-05 and is what the first Python port
matched; it is no longer used anywhere.)

`opt_width`: the port now stops when the remaining ring to `r_cap` is narrower than the
lower width bound (`r_cap - inner < w_lo`), avoiding the 1-px sliver annulus; IDL will add the
same guard at its next pause (addendum 2 §2c).

## 5. Other IDL observations recorded while porting (not bugs in production)

* Global (non-pernight) + `clean_subtract` subtracts the clean S/N twice
  (`cleansub_done` is only set in the pernight branch) — production is pernight, so
  unaffected; the port subtracts once on every path.
* Grid search mode always evaluates `k_klip=1` when `search_k` (k is not in the grid
  `case`), so the grid baseline is only meaningful with `scan_mode`.
* `run_setup.txt` truncates parameter labels at 14 characters (`coronoise_max_n1` →
  `coronoise_max_`), which breaks name-based parsing; the port writes JSON.
* The calibration seed (`X[*,0]`) is not feasibility-projected in IDL and is excluded from
  the TPE training set; the port projects it and keeps it.
* `optimize_tpe_results.txt` can contain duplicate `(annulus, iter)` rows after a
  re-calibration restart and `********` for failed scores; `klip_tpe.idl_replay` handles both.

## 6. Live comparison against a running IDL run (tooling)

`klip-tpe compare --idl-run <run dir> --root <NEAR2_py> --nights 1 2 3 4 5 6 --out <dir>`
replays every evaluation the IDL run has logged through the Python objective on the same
nights, annulus, contrast and injection geometry (exact positions when the IDL
`annulus01/evalNNNN_setup.txt` exists, else a fresh azimuth at the same radius), and
writes `compare.txt` / `compare.png` / `compare_stats.json` (Pearson, Spearman, regression
slope, median PY/IDL, running-best curves, top-5/10/20 set overlap).  It is incremental:
re-run it while the IDL run is going and only the new rows are scored.  A matched Python
search with the IDL settings is started with
`klip-tpe near --from-idl-setup <run dir>/run_setup.txt --seed <other> --run-dir <dir>`;
`klip_tpe.idl_compare.compare_runs` puts the two best-so-far curves and winners side by
side.  What to expect from the earlier checks: rank agreement (Spearman ≳ 0.6) and slope
< 1 (Python scores ~15–40 % lower on the high-k configurations, §3), tightening as the
low-residual discrepancy is resolved.

### 6.1 Result on the production run `run_20260906_173000` (2026-09-12)

Setup: 6 nights, annulus [0, 20] px, contrast 6e-5, univariate TPE, `pbest` 0, post-fix
frame selection, area-midpoint pair geometry; 4469 evaluations logged when the RAID filled
up (the run stalled at 22:20 UTC on 2026-09-11).  IDL writes a setup file and the combined
injected image for every evaluation, so both checks below use IDL's **exact** injection
positions.  Sample: every 25th evaluation (178 configurations).

*Metric only* — Python `MawetPeakSNR` on IDL's own `evalNNNN_score_inj` images:
Pearson 0.91, Spearman 0.91, `PY_raw = 0.98·IDL + 0.79` (the offset is the clean term:
IDL logs `S/N_inj − S/N_clean`, only the injected image is saved).

*Full replay* — Python reduction + metric, same configurations, same positions:

| n | Pearson | Spearman | fit | median PY/IDL | rms diff | top-10 / top-20 overlap |
|---|---|---|---|---|---|---|
| 178 | 0.953 | 0.956 | `PY = 0.895·IDL + 0.31` | 0.992 | 0.54 | 0.60 / 0.75 |

Residual structure: the difference is flat in k up to 40 (median ratio 0.99) and only the
six k ≥ 40 configurations sit at 0.94; configurations with `angsep < 0.2` average
−0.19 below IDL while `angsep ≥ 0.2` average +0.07 — the low-residual corner of §3, now
quantified at the score level: on the 47 configurations IDL scores above 5, Python is 5 %
lower (median ratio 0.95).  IDL's best sampled configuration (7.78) scores 6.96 in Python;
Python's best (7.01) scores 7.07 in IDL.  Files: `compare.txt`, `compare.png`,
`compare_stats.json` (klip-tpe-py/docs/compare_run_20260906_173000/).


### 6.2 Independent Python search on the same problem (2026-09-12)

A Python run with the IDL run's exact problem (`klip-tpe near --from-idl-setup`: six nights,
annulus [0, 20] px, forced contrast 6e-5, two sources at the area-weighted mid radius 0.645",
TPE gamma 0.25 / ncand 48 / explore 0.15 / p_local 0.15 / pbest 0, `n_valid` 8 / `n_top` 3)
but its own random stream and a 50-eval warm-up instead of IDL's 500, run for 300
evaluations, gives the same kind of search:

| | warm-up median (p90) | guided evals 1–250: median | p90 | max | frac > 5 | best after 250 guided |
|---|---|---|---|---|---|---|
| IDL run_20260906_173000 | 2.38 (4.09), n = 500 | 3.40 | 5.34 | 7.90 | 0.16 | 7.90 |
| Python matched run | 2.21 (3.97), n = 50 | 4.00 | 6.09 | 8.17 | 0.28 | 8.17 |

Warm-up (random) distributions agree — the same objective on the same data.  In the guided
phase Python climbs slightly faster in this single realisation (one run each, so the
difference is within run-to-run scatter: IDL itself needed until eval 1255 for its first
score above 8 and reached 9.87 by eval 4747).  Python's validation: candidates at evals
200 / 263 / 268 (search 8.17 / 8.12 / 7.91) re-validated 6.22 / 8.04 / 8.25 on fresh
injections — the search best was an upward fluctuation, the runner-up configurations hold
their scores; winner = eval 268 (bin 5, filter 9, per-night k 13–40, all six nights kept).
IDL's incumbent at that time (eval 4381, 9.87 search) uses the same `bin`/`filter` regime
(bin 30, filter 10, k 14–18, nights 2–6) at a much later stage of the search.  Figure and
numbers: `KLIP-TPE/idl_vs_python/run_comparison.{png,json}`; Python run products in
`idl_vs_python/py_match_run/`.
