# Changelog

## Unreleased — 2026-10-30 (the README panel on RX J0534)

- `scripts/readme_panel.py` makes the README's panel from RX J0534 b (LBTI/LMIRCam L', four
  nights) by default.  It runs `scripts/run_rxj0534.py --partitions 4` for 3,000 evaluations,
  with the contrast grid at half-FWHM steps (five separations from 0.29" to 0.49" in the
  annulus, the companion's separation +/- 2 FWHM) and the title "RX J0534".  `--target
  hd95086` runs the earlier panel on public data.
- `LiveDisplay(title=...)` names the run on the panels and the annulus books in place of the
  run directory's name.
- `scripts/run_rxj0534.py` takes `--title`, `--panels` (the panels without a window) and the
  contrast grid's options (`--contrast-grid`, `--cg-pa`, `--cg-step`, `--cg-target`,
  `--cg-snr`).

## Unreleased — 2026-10-30 (the contrast curve from an injection grid, one source at a time)

The protocol of Kevin Wagner's IDL contrast generators (`contrastcurve_*.pro`), after
validation at each annulus's winning configuration (`RunConfig.contrast_grid`,
`klip_tpe.contrast_grid`).

- **One injected source per reduction**, so no injection biases another's signal or noise, at
  12 position angles (`--cg-pa`) on separations a FWHM apart (`--cg-step`), the annulus less a
  FWHM at each edge.  `--contrast-grid random` draws `--cg-samples` positions instead, from a
  generator of their own.
- **Each position iterated on its own contrast** until its S/N is within 10 % (`--cg-precision`)
  of 5 (`--cg-target`), by the IDL's rule: `c * 5 / snr`, ten times `c` for a source not seen.
  From the sixth reduction on, a position at or above the target is accepted at its last
  contrast.  From the eighth, one still short of 90 % of it is dropped as blocked.  The first
  guesses come from the KLIP-FM curve, so on HD 95086 a position takes about two reductions.
- **The S/N** is the run's matched-filter Mawet S/N, or the IDL's aperture photometry
  (`--cg-snr aperture`, `metrics.aperture_mawet_snr`).
- **The noise rings** leave out the injected source, the known companions and their ADI
  sidelobes.  A sidelobe is the arc the field rotation sweeps, `+/- min(PA span, anglemax)`,
  never less than 2 FWHM of arc, the IDL's two apertures each side (`--cg-sidelobe`,
  `--cg-sidelobe-fwhm`).  No test
  position goes on a companion's sidelobes.  `mawet_peak_snr` and `aperture_mawet_snr` take the
  arcs as `excl_arcs`.
- **The curve** is the median over position angle at each separation, with the 16th and 84th
  percentiles (`AnnulusResult.grid_curve`).  The final frame draws it with its band beside the
  KLIP-FM curve, in place of the fit to the validation injections.  It leads the run-level
  `contrast_curve.txt` and has a section in each annulus's.  `contrast_grid.json` keeps every
  position's trials and `contrast_grid.txt` the curve.
- On by default.  A run saved before it resumes without it, the benchmark harness leaves it
  off, and the test suite turns it off unless a test asks for it.  The grid draws nothing from
  the run's generator.

## Unreleased — 2026-10-30 (known companions out of the live contrast preview)

- The live contrast preview (this evaluation's curve and the best evaluation's) leaves known
  companions out of its noise rings, as the run's own curve and the KLIP-FM cross-check
  already did.  HD 95086 b had put a spike in tutorial 02's preview at 0.62", about 1.7 times
  the limit beside it.  The panels `klip-tpe render` draws afterwards get the same.

## Unreleased — 2026-10-30 (the finished annulus's contrast panel, the README panel's script)

- The finished annulus's frame no longer draws the best evaluation's preview curve.  The
  preview is that evaluation's own few injections, and validation's curve replaces it.  The
  IDL panel colours both alike, so the final frame showed two solid curves that disagreed for
  no reason a reader could see.  While the search runs, the preview stays.
- `scripts/readme_panel.py` makes the README's panel.  It is tutorial 02's search on HD 95086
  with the same data, seed and settings, run for 3,000 evaluations instead of 300, long enough
  to show the climb and the plateau after it.

## Unreleased — 2026-10-28 (a final round of validation on common draws)

From the addendum to the GPU session's note of 2026-10-28.  On its 200 common draws the
validation list of NEAR2's run_20261027_222223 held a better reduction than the elected one in
four of seven annuli, by 1 to 3 % (by 12.6 % in an annulus of run_20261024_153342), each time
outside the bootstrap's 16–84 % interval.

- **A final round** (`ValidationConfig.n_final`, `n_final_draws`, `--n-final K`,
  `--n-final-draws M`, off by default).  The K candidates with the largest validated medians
  are scored on M common draws, the same injections for each, and the largest median there is
  elected, the better first-round median on a tie.  The first round's medians are each on draws
  of their own and scatter by a few tenths of S/N, about as far as the top candidates lie
  apart.  The candidates' scores correlate at 0.92 to 1.00 across draws, so on common draws
  their differences scatter far less.  The GPU session suggests 5 × 200, about 1,000 more
  reductions an annulus, which roughly doubles a NEAR2 validation on patrick.
- The round's draws come from a generator seeded from the run's, which stays where it stood.
  The first round, the search after it and a resume are what they were, and a resumed run makes
  the same round again.  The winner is then scored on fresh draws (`n_rescore`), and
  `election_score` is its median in the round.
- `winner.json: final_round` keeps the entrants' medians and trials, `validation_final.json`
  the draws as well.  The validation panel draws the round's medians as diamonds.
- README and docs/BUDGET.md now describe the winner's fresh draws of 7a5bc15.

## Unreleased — 2026-10-28 (distinct reductions in validation, the winner on fresh draws)

From the GPU session's note of 2026-10-28.  It re-validated every annulus of NEAR2's
run_20261024_153342 and run_20261027_222223 on 200 common draws per configuration, with a port
that reproduces klip-tpe's validation trials to 6e-6.

- **The candidates are distinct reductions** (`ValidationConfig.dedup = "image"`,
  `--validation-dedup`).  The 25 candidates of an annulus were 7 to 20 distinct reductions.  The
  copies were apart from an earlier candidate only in parameters the reduction never reads (a
  deselected night's, frame cuts with no frame between them, an even high-pass width, which
  `klip.highpass` runs as the next odd one).  74 of run_20261027_222223's 175 validation slots
  re-scored a reduction already in the list, and the election among copies went by draw noise.
  Validation now makes each configuration's clean image first (its first reduction anyway) and
  skips one equal to an earlier candidate's, bit for bit, for the next in the search's order,
  `dedup_depth x n_top` deep.  That costs one clean reduction per copy and takes no draw.
  `winner.json` lists the copies as `validation_duplicates`.  `"vector"` compares the vectors
  only, as the IDL does.
- **The winner's validated score is its median on fresh draws** (`ValidationConfig.n_rescore`,
  `--n-rescore`, `n_valid` by default).  The election takes the largest of the candidates'
  medians, so the elected one is high on its own draws.  Against the 200 common draws the
  winners were high by 0.22 S/N on average in run_20261027_222223 (up to 0.43) and by up to 0.94
  in run_20261024_153342, while the candidates not elected were unbiased (+0.005).  After the
  election the winner is reduced on `n_rescore` fresh injection sets.  Their median is
  `winner_score` (`best_SNR`, the per-annulus SNR line, `bench_summary.txt`), the trial nearest
  it is the committed one (the products, the final verify's injections), and their per-source
  S/N give the contrast curve.  The election's median stays as `election_score`, the trials as
  `rescore_trials`.  The draws come from a generator seeded from the run's, which stays where it
  stood, so everything after is what it would have been.  A lone candidate (`n_top=1`, or one
  median among failures) is not re-scored, since its draws were not selected on, so its run
  gives what it gave before.  `--n-rescore 0` reports the election's median, as the IDL does.
- A run saved before 2026-10-28 resumes with the vectors and the election's median, so its
  annuli stay alike.  `--from-idl-setup` (`idl_compare.matched_run_config`) mirrors the IDL's
  validation too, unless `--validation-dedup` or `--n-rescore` is given.  The validation panel
  draws the fresh median as a star and gives both.

## Unreleased — 2026-10-28 (the thread cap off by default)

From the GPU session's A/B of 9097654 on patrick (2026-10-27), with the same farm benchmark of
NEAR2's annulus 5 as before.

- **`PartitionedReducer.cap_threads` is False.**  Holding the jobs running together to the
  budget took a six-night reduction from 0.205 s to 0.339 s at 96 / 2 / batch 4, 1.65x slower
  and slower than the old 192 / 4 layout (0.267 s).  At 128 and 192 workers it was as bad
  (0.347 and 0.355 s), so the loss is the policy, not the budget.  With 48 workers and a budget
  of 96, `ProcessPool._fit` kept a thread for each idle worker, so the few jobs sent first took
  8 threads and nearly every later one, the heavy nights included, ran on 1.  Each trial then
  waited on one-thread stragglers, and the pool sat 25 % busy at a load average of 39.
  Uncapped, the plan's jobs hold a load average of 133 on the budget of 96, which patrick's 192
  logical CPUs absorb, and that is the fastest layout measured there.  `cap_threads = True`
  turns the cut on again, for measuring a bound.  A pinned `--workers N` can again be exceeded
  for a while by the jobs a long reduction is given.
- `--workers auto` = physical cores and `--threads-per-job 2` stay: uncapped, 96 / 2 ran at
  0.205 s, the best of the thirteen layouts measured over two days.

## Unreleased — 2026-10-27 (the worker farm: a thread per physical core, 2 per job, the budget held)

From the GPU session's farm benchmark on patrick of 2026-10-26.  It replayed 200 recorded
evaluations of run_20261024_153342's annulus 5 through `PartitionedReducer.reduce_config` as
the pipelined search sends them, 6 min per layout on an idle machine (96 cores, 192 hardware
threads).  A six-night reduction took 0.267 s at the production layout (`--workers 192
--threads-per-job 4`, 48 processes of 4 threads), 0.213 s at 96 / 2 (48 processes of 2,
1.25x), 0.221 s at 128 / 2, 0.230 s at 96 / 1 and 0.245 s at 192 / 2.  The second hardware
thread of a core added nothing, and 24 x 8 lost (0.79x).

- **`--workers auto` is one thread per physical core** (`parallel.physical_cores`: each
  allowed CPU's siblings in sysfs on Linux, `sysctl hw.physicalcpu` on macOS, else the logical
  count).  `--workers all` (or `logical`, `max`) is every logical CPU, what `auto` was, and
  `-k` is `auto` less k.  On a machine without SMT nothing changes.
- **`--threads-per-job` is 2 by default** (4 before), on the command line and in
  `PartitionedReducer`.  With the new `auto` that is the layout patrick ran fastest, 48 x 2,
  now held to its budget of 96 (below).
- **The jobs running together never use more threads than the budget.**  The by-cost plan
  gives a long job up to 4x the base and a short one 1, and the long jobs run longest, so the
  jobs in flight held more than the budget: a load average of 234 on 192 CPUs at a base of 4,
  134 on a budget of 96 at a base of 2.  `ProcessPool` now cuts a job's threads, when it sends
  it, to what the running jobs leave, keeping one for each idle worker
  (`PartitionedReducer.cap_threads`, True, a switch for measuring the two).  A pinned
  `--workers 8` now uses at most 8 threads.  The images do not depend on the thread count
  (`tests/test_job_threads.py`), so none of this changes a result.

## Unreleased — 2026-10-25 (the main thread: no per-night S/N in the search, no wait in validation)

From the NEAR2 session's note of 2026-10-25 on run_20261024_153342 (seven annuli of 10,000
evaluations, 21.3 h on patrick, 5 draws, batch 4).  The main thread set the search's pace there,
0.6-0.75 s per evaluation in five of six annuli, with the farm 55-70 % busy.  It set
validation's pace too.

- **The search measures the combined image only.**  Each draw of each trial scored the
  combined images, injected and clean (the score), and then every selected night's own image
  for `EvalRecord.partition_snr`.  That was 25-30 S/N measurements per 5-draw trial beside the
  10 that make the score, and the record kept only the last draw's.  Nothing in the search read
  them.  Kevin decided that only the combined S/N matters.  New records carry
  `partition_snr = {}`, as failed ones did, so `results.jsonl` keeps its fields and old runs
  read as before.  `Runner._score_stack` stays for scripts, and `klip-tpe plots`' per-partition
  figure and the classic panel's per-partition curve say the values were not computed.
- **Validation does not wait for the display.**  A validation trial that found the render
  thread busy waited out the search's 0.25 s grace on the main thread before going without a
  panel.  A validation panel takes several times longer to draw than a trial takes to score, so
  nearly every trial waited, about 5 of each annulus' 12-14 min in the NEAR2 run.
  `LiveDisplay.VALIDATION_GRACE` (0) skips such a panel at once, and the annulus' data is built
  once per validation instead of per trial (77 ms at 10,000 evaluations).  The suite holds
  both graces open (`tests/conftest.py`), and `tests/test_display_lag.py` runs validation with
  the production values.
- **Measured** on a NEAR2-shaped synthetic run in the sandbox (six nights of 150 px frames at a
  tenth of NEAR2's frame counts, NEAR's loader, space, guard and objective, 5 draws, batch 4,
  100 trials, validation 3 x 8, 2 cores).  The main thread's CPU per search trial went from
  825 to 222 ms, a trial was scored 0.34 s after its last reduction was back instead of
  1.24 s, and validation's display work per trial went from 0.27 s to 0.6 ms.  The records
  are the same apart from their times and `partition_snr`, and so are the validation table,
  the stitches and `final_results.json` (its paths aside).  On 2 cores the workers set the pace, so the run took
  564 s against 592 s.  On patrick the farm has room, and the note expects the search to run
  at the farm's pace, 1.3-1.6x faster in such annuli.

## Unreleased — 2026-10-24 (frames left out of a NEAR night at load)

From the NEAR2 session's note of 2026-10-23, revised at night (its section 5, n4's frame 551).

- **`--drop-frames NIGHT:INDEX[,INDEX...]`** (near, on `near`, `resume`, `extend`,
  `fit-chain`, `rereduce`, `replay` and `compare`), **`near.load_night(drop_frames=[551])`**
  and **`near.make_reducer(drop_frames={"n4": [551]})`**: frames left out of a night at load,
  by their index in the night's cube counted from 0, before the frame tags, the frame cut, the
  binning, the injections and KLIP-FM see them.  `texp` scales with the frames kept, and the
  files are not changed.  n4's last frame, 551, has its parallactic angle recorded as 12.07 deg
  where its neighbors are at 24.1 deg (its header's angle at the end of the exposure is 0), and
  a frame-tag correlation of 0.16 against a median of 0.995.  The frame cut keeps it wherever a
  configuration sets the correlation threshold below 0.16, as run_20261015_171115's n4 blocks
  in annuli 1, 3 and 5 do, and wherever the cut would leave fewer than bin x k frames.  In
  those three blocks, on n4 alone, dropping it changes the clean residual by 3, 31 and 23 % of
  the annulus' RMS.  In annulus 3 it raises the S/N of six planets injected one at a time at
  every position, by 2 to 15 %, with frame 25 held out of both reductions (the coronoise cut
  compares each frame with the mean, which frame 551 pulls down).  Where the cut removes the
  frame already, a rereduce of a test run on nights 4 and 6 with `--drop-frames n4:551` gives
  the same products, to the bit.  An index past the cube, a night not loaded and a list that
  leaves no frame are refused.  generic, nomic and ales refuse the option.
- **The run's record.**  The list goes into the reducer's `data_edits` as `dropped_frames`,
  and so into `RunConfig.data_edits`, the checkpoint and `describe()`.  `resume`, `extend` and
  `fit-chain` take it from the checkpoint when it is not typed and refuse a typed list that
  differs, and the `Runner` refuses a resume whose reducer drops other frames.  `rereduce`
  takes a typed list in place of the run's (`none` for none) and writes `NDROPFR`, the number
  of frames dropped, into its FITS headers.

## Unreleased — 2026-10-24 (angsep off the auto-fast corner)

Every c-tuned MWC 758 winner so far has `angsep` 0 and `anglemax` 127 deg, the night's PA span:
the corner where `klip_annular` takes the auto-fast shortcut, one basis per zone from all the
frames with each target frame inside its own (the IDL's rule, `reduce_near_2.pro` L1839-1845),
where any positive `angsep` leaves the target out.  The corner is not of measure zero:
`Param.sanitize` clips a proposal past the bound onto it, and the reference-count guard
projects onto 0 as well, so in the 4e-5 run 74 % of the TPE's proposals and 93 % of the local
search's had `angsep` exactly 0.  It costs the disk.  The 4e-5 winner keeps 0.18 of the paper's
arm flux; the same configuration at `angsep` 0.01 keeps 0.55 (at 1e-6 the same, and with
`anglemax` 120, where the shortcut stands down, 0.53), and c comes out at S/N 2.9 against 2.4.

- **`near.make_space(..., angsep_range=(lo, hi))`** (and through it `generic.make_space` and
  `ales.make_space`): `angsep` searched over lo-hi lambda/D instead of 0-3, its default clipped
  into the range.  A floor above 0 keeps every evaluation on the per-target path.
- **`scripts/run_mwc758_ales_clike.py --angsep-min 0.01`** (the paper's value),
  **`--n-remeasure N`**, the injection draws averaged per evaluation (3, as before, by default),
  and **`--k-max K`**, the most KL modes searched (`make_space(k_klip_max=)`; 100 here by
  default).  Off the corner the objective still trades the disk for point-source S/N along k:
  at the paper's step with the 4e-5 winner's frame cuts the arm flux goes from 1.01 at 15 modes
  to 0.66 at 18 and 0.38 at 25, while with the winner's 13-px box the injections' S/N gains
  22 % from 15 to 18 modes (`_sync/check_objective_floor.py`).  The
  run directory is tagged `_asA`, `_nrmN` and `_kmaxK`, a run is not resumed with another
  floor, count or cap, and `clike_summary.json` records all three.

## Unreleased — 2026-10-23 (injections at a chosen separation)

- **`RunConfig.inject_rho`** (arcsec): every injection of every stage (search, k-scan re-score,
  calibration, validation, param_verify) at that one separation, in place of the band the
  annulus gives.  Two sources are still 180 deg apart, and the sampler's floor still applies.
  A radius outside an annulus the run scores is refused.  None, the default, keeps the band;
  checkpoints written before the field resume with None.
- **`scripts/run_mwc758_ales_clike.py --inject-rho 0.6`**: the injections at c's own
  separation (0.603") instead of the 0.4-0.8" annulus' area-weighted mid radius, 0.632".  The
  annulus, its zone and its scoring are unchanged.  The run directory is tagged `_rinjR`, and a
  run is not resumed with its injections elsewhere.  The disk mask covers 8% of the noise band
  at 0.6", against 2% at 0.632".

## Unreleased — 2026-10-23 (a disk of any shape out of the noise and the injections)

The MWC 758 winners leave the disk's spiral arms all but gone, so the c-tuned run gets a way to
block the disk.  `pa_wedge_mask` and `forbidden_pa` covered a disk seen near edge-on; arms and
rings follow no wedge.

- **`metrics.disk_mask_from_image(img, fwhm_px, nsig=2, r_range=, min_beams=2, grow_fwhm=1,
  smooth_fwhm=1, protect=[(rho, pa)], pxscale=)`**: True on the bright extended structure of an
  image of the disk.  The image is smoothed by a FWHM and cut at `nsig` robust sigma; pieces
  under `min_beams` beams are dropped and the rest grown by `grow_fwhm` FWHM, the matched
  filter's reach; protected positions (a companion) stay clear.
- **`disk_mask_from_image(..., nsig_neg=, neg_reach_fwhm=2)`** adds the disk's ADI negatives:
  the troughs `nsig_neg` robust sigma deep, in pieces of at least `min_beams` beams, that come
  within `neg_reach_fwhm` FWHM of the bright structure; a trough nowhere near the disk stays.
  `make_disk_mask.py --neg-nsig S [--neg-reach-fwhm 2]`.  On MWC 758 at 2 sigma the mask grows
  from 391 to 592 px, and from 7.5 to 9.2% of the noise band at 0.6".
- **`PositionSampler(forbidden_mask=, mask_pxscale=, mask_clear_fwhm=0)`**: no injection on the
  mask, or within `mask_clear_fwhm` FWHM of it.  With a mask the sampler tries up to 400
  anchors, since two opposite sources must both miss it; without one the draws are what they
  were.  `describe()` adds `forbidden_mask_px` only when a mask is set.
- **`default_config(..., disk_mask=, disk_mask_clear_fwhm=0)`** (generic, ales): one mask for
  both halves, OR-ed into the metric's `pixel_mask` and given to the sampler.
- **`scripts/make_disk_mask.py IMAGE --fwhm PX --out MASK.fits`**: the mask from any FITS image,
  its parameters and the image's SHA-1 in the header, and a preview.
- **`scripts/run_mwc758_ales_clike.py --disk-mask MASK.fits`**: the run scored with the disk
  masked.  A copy goes into the run directory, c's S/N at the end is measured with it, the run
  directory is tagged `_disk`, and a run is not resumed with another mask or without one.
  **`compare_mwc758_fig1b_clike.py`** measures a masked run with its own mask (`--disk-mask
  FITS|none` overrides), drops the ring positions on the mask from the empty ones, and draws
  the mask on every panel.
- On MWC 758 the arms lie inside 0.6": a mask of them (arms above 2 sigma in the paper's ADI
  image, grown a FWHM) covers about 2% of the noise band at 0.632", where the runs inject.
- `docs/ALES.md` section 9; `tests/test_disk_mask.py`.

## Unreleased — 2026-10-21 (the c-tuned run's groups: how many may drop, or held)

- **`scripts/run_mwc758_ales_clike.py --max-excluded N`**: how many groups the search may switch
  off, half of them by default (10 of November 14's 20) as before.  The paper's ADI band,
  3.4-4.15 µm (g09-g19), is within that bound and its Fig. 1b, 3.8-4.15 µm (g15-g19), is not;
  `--max-excluded 15` reaches it.  A bound other than the default tags the run directory
  `_maxexN`, and a run is not resumed with another bound.
- **`--fixed-groups LO-HI`** of the same script holds the combination at those groups for every
  evaluation (`15-19`, the 3.8-4.15 µm of Fig. 1b): only they are loaded, the space has no group
  flags and the guard no group bound.  The matched filter, with the FWHM and lambda/D of the
  metric, the sampler and the guards, is then the band's middle group's (g17, 3.97 µm, for
  15-19), since g10 lies outside the band; **`--kernel-group gNN`** picks another.  Run directory
  tags `_gLO-HI` and `_mfgNN`; a run is not resumed with other groups or another matched filter.
- **`--max-contrast C`** of the same script (in since 6c9813e, not logged then): a cap on the
  injected contrast through the calibration and every re-calibration
  (`CalibrationConfig.ceiling`); with c's spectrum it is the contrast at 3.8 µm.  The cap tags
  the run directory `_cmaxC`, and a run is not resumed with another cap.
- **`scripts/compare_mwc758_fig1b_clike.py --kernel-group gNN`**: every panel's c S/N and ring
  with that group's PSF as the matched filter, by default the run's own; the files are then
  `fig1b_clike_mfgNN.*`.  A run with its groups held gets one winner panel.

## Unreleased — 2026-10-24 (the verify on the stitched image, TABLE 3, tau's cap named)

From the NEAR2 session's notes of 2026-10-20 (the verify's combined image) and 2026-10-23
(revised that evening and at night, on limits for faint planets and at small separations).

- **The final verify measures the stitch itself.**  Path A, the combined S/N and FAP of the
  candidates and the injections and the limit tables, ran on the equal-weight mean of the
  per-night stitches (`klip_stitched_nights*.fits.gz`), as the IDL does.  That is not the
  stitch.  At an edge between two annuli each tile reaches into the other annulus, a night
  kept by only one of the two holds that annulus' value there at full weight, and the stitch
  weighs the annuli by their sensitivity, not by night.  In NEAR2's run_20261015_171115, n4,
  dropped by annulus 2 and kept by annuli 1 and 3, filled 43 % of annulus 2 with those annuli's
  reductions.  The final verify now measures `klip_stitched.fits` and `klip_stitched_inj.fits`
  (`verify_candidates(..., combined=, combined_inj=)`), and each annulus' verify and the
  verify books measure the annulus' combined image.  Path B (STIM, bootstrap, subsets) stays
  on the per-night images.  The candidate list measures its combined S/N, FAP and PSF-likeness on
  the stitch too (`find_candidates(image, ...)`), and its header says so.  On one annulus, or
  where neighboring annuli keep the same nights, nothing changes.
- **TABLE 3 in the verify report.**  With the clean image beside the injected one, each
  injection's S/N is solved on clean + a (injected - clean), the image's noise as it is and
  only the source's response scaled (`verify.solved_contrasts`).  Per bin TABLE 3 gives the
  mean of the contrasts at which each source reaches tau + tau_(1-TAP), with the exact tau
  (TABLE 1 solved instead of scaled), the faintest contrast at which 90 % of the bin's
  sources are detected, and how many are detected at TABLE 1's c@TAP90, both with every
  source evaluated at that contrast.  Each source's first detected contrast and solved
  c@TAP90 follow.  TABLE 1 stays the headline, as in the IDL, and the injection lines keep
  their format.
- **TABLE 1 says where tau's cap binds.**  `t_thresh` caps tau at 50, as the IDL's bisection
  does, while `t_fap`, which decides detection, is exact.  With six noise samples or fewer
  the exact tau at the operating FAP is 57 to 1.1e6, so a source at TABLE 1's c@TAP there
  does not reach the FAP.  A planet's own residual in its ring also caps its S/N there (6 to
  14 for NEAR2 inside about 0.56"), so no contrast reaches it.  TABLE 1 keeps the IDL's
  numbers and names those bins below the table, and TABLE 3 shows `inf` there.
  `contrast_limit` and `t_thresh` are unchanged.

## Unreleased — 2026-10-23 (the params table's summary row, the candidate list's header)

From the NEAR2 session's note of 2026-10-23.

- **`klip_stitched_params.txt`'s summary row** is the median over the partitions the winner
  keeps, as its header said, rather than over all of them (`winner_config.params`).  Its values
  and those of the per-partition block carry 6 significant digits, so they can be rounded again
  without the error of a second rounding.
- **The candidate list's header** gives the look-elsewhere FAP as it is computed, 1 - (1 -
  p)^N_resel, where it said "x N_resel".  The IDL `near2_candidates` header has the same
  wording (line 379) over the same formula (line 294).

## Unreleased — 2026-10-20 (tutorial 7 on November 14)

- **Tutorial 7** searches 2019 November 14 alone, the longer of the paper's two nights: 20
  wavelength groups as 20 partitions, `n2_g00` to `n2_g19`, the labels of the two-night data
  set.  It downloads that night only, 491 MB in 21 files, with `datasets.fetch(...,
  roles=...)`, and the guard lets the search drop up to 10 groups.  January 5, and the
  partition weights by exposure time, are a variation now.  Its run directory is
  `runs/mwc758_ales_nov14`, so an earlier two-night run is not resumed into it.
  `docs/TUTORIALS.md` and the README follow.

## Unreleased — 2026-10-20 (the ALES paper's processing geometry, held)

The c-tuned two-night run searched the KLIP zone's reach beyond the annulus and the number of
sectors, and its winner processed 11.5-27 spaxels in one sector, where the paper processes one
5-27 spaxel annulus in six.

- **`make_space(..., pin={name: value})`** (near, generic and ales): reduction parameters held
  at a value for every evaluation, out of the search (`SearchSpace.fixed`).
- **`zone_center`** and **`zone_angles`** (KLIPReducer parameters; `set_zone_center(red, (x, y),
  angles)` sets them on every reducer): the point the zones are drawn about, by default
  `(nx / 2, ny / 2)`, and how sector angles are counted, `'ccw'` from +x by default or
  `'getzone2015'`, atan2 + 180 as the 2015 IDL `getzone` counts them.  `KLIPParams.zone_angles`,
  `klip.zone_indices(..., angles=)`.  With `(30, 30)` and `'getzone2015'` the zones of the
  MWC 758 ALES reduction are its pixels, sector by sector.  Unset, every reduction is as it was.
- **`scripts/run_mwc758_ales_clike.py --geometry paper`**, now the default: the zone held at the
  paper's 5-27 spaxels about (30, 30) in six sectors, the injections and the score still in the
  0.4-0.8" annulus; `--free-sectors` searches the sectors; `--geometry searched` is the run of
  2026-10-18.  A run is not resumed with another zone centre.
- **`scripts/compare_mwc758_fig1b_clike.py`**: a finished run's winner next to Fig. 1b's source.

## Unreleased — 2026-10-20 (the verify measures each injection without the others)

The NEAR2 session's note of 2026-10-20 found that the final verify counted the other injected
planets as noise.

- **`verify_injections`** (the calibration block of the per-annulus and final verify, the
  verify books and `rereduce`) leaves out of each injected source's noise ring, in the
  combined image and the four subsets, the other sources of the block and the known ones,
  within the metric's keep-out of their positions (`excl_fwhm`, 1.5 FWHM by default), as the
  metric does in the search.  The STIM FAP's annulus drops them within the same keep-out.
  Two sources 180 deg apart at one radius put a ring sample on the other source when `nap`
  is even, and two beside it when `nap` is odd, so each S/N came out low.  Now each source
  gets the S/N it gets injected alone, to the noise (`tests/test_verify_stack.py`).  The
  runner and the verify books pass the metric's `excl_fwhm` to the verify, which used its
  own default.
- **The limits of earlier runs' final images change.**  Wherever two injections share a
  separation, their S/N rises and the TABLE 1 and TABLE 2 contrasts of `verify_report.txt`
  and `stitched_verify_report.txt` fall.  On run_20261013_194614 the NEAR2 session found the
  pairs' S/N doubled (7.3 and 9.4 to 14.6 and 20.2 at 0.77") and c@TAP90 fell by 1.4 to 3.2
  times.  An annulus with one source does not change, and neither do the search, its scores
  or `verify_curve.txt`, whose S/N is the metric's.
- **To verify a finished run again,** run `klip-tpe rereduce` on its own frames.  The products
  are the run's, to the bit, and `<run>_rereduce/stitched_verify_report.txt` holds the new
  calibration block.  A report says above that block how its rings were measured, so an
  earlier one can be told apart.
- The candidates' STIM FAP drops the known sources from its annulus, as their S/N rings
  always did.  The verify books (`verify_limits.pdf`, `verify_subsets_inj.pdf`) pass the
  run's known sources and angle convention, as the runner's verify does.
- The IDL `near2_verify` calls `near2v_snrfap` without `exclude` (verify 588-591 and
  655-656) and keeps every pixel of the STIM annulus (660).  So its calibration block, and
  `near2v_collect` behind its `verify_curve.txt` (430), leave the other injections in the
  rings, and since 2026-09-05 `near2m_randpos` puts an annulus' two sources at one radius,
  180 deg apart.  The module docstring lists this with the other deliberate deviations.

## Unreleased — 2026-10-18 (coloured injections; the ALES paper's ADI step in the space)

Tutorial 7's winner suits star-coloured injections and not MWC 758c, a much redder source
(S/N 1.9 against 3.6-3.7 in the paper's November 14 images), and two steps of the paper's
reduction were outside the space.

- **`PartitionedReducer.set_injection_spectrum`** (`ales.make_reducer(spectrum=,
  spectrum_norm=, spectrum_label=)`): the injected sources get a spectrum across the
  partitions, a function of wavelength, a table or one factor per partition, normalised at
  the representative partition, another one or a wavelength.  Every injection of the run gets
  it (search, calibration, validation, KLIP-FM, the SDI pool's borrowed frames); a negative
  source keeps its contrast.  None, the default, is the flat spectrum, to the bit.  Recorded
  under `injection_spectrum` in the reducer's description.
- **`zone_in`, `zone_out`** (KLIPReducer parameters, px, default 0): the KLIP zone reaches
  beyond the annulus, as the paper's one 5-27 spaxel zone does.  `angsep`'s arc and the
  bin's PA span stay the annulus' (`KLIPParams.arc_radii`, the SDI pool's `mid_px`).
  **`post_hp`** (default 1): 0 derotates the KLIP residuals without the second high-pass, as
  the paper's ADI images are made.  `make_space(zone_range=(lo, hi), search_post_hp=True)`
  (near, generic and ales) searches them, seeded from `defaults`; without them the space and
  every reduction are as they were.  The live display labels them `zin`, `zout` and `hp2`.
- **`scripts/run_mwc758_ales_clike.py`**: tutorial 7's search with c's spectrum (a power law
  fitted to the c contrasts of the paper's injection tests), these dimensions, the paper's
  configuration as the seed and square-root-exposure weights; `--nights nov14`, `--sdi`,
  `--quick`.  `docs/ALES.md` section 8; `tests/test_ales_clike.py`.

## Unreleased — 2026-10-17 (tutorial 7 on the two ALES nights of Wagner et al. 2023)

- **`datasets.fetch("lbti_ales_mwc758")`** is now 2019 Jan 5 and Nov 14, the two ALES nights
  of Wagner et al. (2023): 583 frames of 1.48 s (71 deg of rotation) and 1594 of 2.46 s
  (127 deg), 670 MB in 42 files, on the same 20-group grid.  2019 Nov 13, which the paper does
  not use, is its own data set, **`lbti_ales_mwc758_20191113`** (21 files); merged with the
  two, `load_release` reads three nights, `n1` to `n3`.  The data sets' `texp_frame` in
  `datasets.INSTRUMENT` is per night (each meta header's `EXPTIME`, which `load_release`
  reads).  The January 5 files go into the `tutorial-data` release.
- **Tutorial 7** searches the paper's two nights, `n1` = Jan 5 and `n2` = Nov 14.
  `docs/ALES.md` section 7, `tests/test_ales_release.py` and `tests/test_dataset_cache.py`
  follow.
- **Live display:** `show="inline"` outside a notebook falls back to a window at the full panel
  size (shrunk to fit the screen).  It used to keep the scale meant for a notebook cell
  (tutorial 7 asks for 0.55), which made that fixed-size window too small to read when the
  tutorial ran as a script.

## Unreleased — 2026-10-15e (a finished run's winners on other frames, sources added at load, the verify's injections given)

From the NEAR2 session's reply of 2026-10-15 15:30 MST: the checks on the line need
run_20261013_194614's winners reduced on frames with the line subtracted, and planets put on
the line.

- **`klip-tpe rereduce` / `Runner.rereduce`**: each annulus' winning configuration reduced
  again on the frames the command line asks for (the run's own for every option it leaves
  out), stitched with the run's weights and seam patching, with the S/N map, the injected and
  per-night stitches and the final verify, in `<run>_rereduce/` beside the run, which it only
  reads.  On the run's own frames the products are the run's, to the bit.  `rereduce.json`
  records the frames, the weights and the injections, and an earlier re-reduction is not
  written over without `--overwrite`.  `Runner.resume(..., same_frames=False)` builds the
  runner for it, and a plain resume on other frames is refused as before.
- **`--verify-sources FILE`** (rereduce): the final verify's calibration injections at given
  rho, PA and contrast, all of them in every annulus' injected reduction, in place of the
  winners' validation draws.
- **`--inject-sources FILE`** and **`KLIPReducer.inject_sources` / `PartitionedReducer.inject_sources`**
  (`near.make_reducer(inject_sources=)`): point sources added to the frames once, at load,
  after any subtraction, by the path `--subtract-sources` takes with the sign the other way.
  Recorded in `RunConfig.data_edits` as `injected_sources`, kept on resume and extend, checked
  by the `Runner`, and measured as the subtraction is (`injected_kept`).  An all-negative list
  is refused, pointing to `--subtract-sources`.
- **The all-negative refusal of `--subtract-sources`** now says when the minus applies: to a
  list of the negatives to inject.  A line in its own sign goes in as it is, and columns
  after the third are ignored.
- `_final_stitch(weights=, extra_header=)` and `_final_verify(header=)` for the re-reduction.
- `docs/RUNNING.md` (a section on `rereduce`, `--inject-sources`), `docs/CLI.md` (now with
  `rereduce`), `tests/README.md`.  New tests in `test_rereduce.py`, more in
  `test_subtract_sources.py` and `test_near_data_edits.py`.

## Unreleased — 2026-10-15d (edited frames, another cube or point sources taken out at load, a line through the star, the chain fit)

From the NEAR2 session's note of 2026-10-15 on the alpha Cen A-B line (PA 13/193 deg) in
run_20261013_194614's final image.

- **`KLIPReducer.subtract_sources` / `PartitionedReducer.subtract_sources`** take point sources
  out of `data.cube` once, at load, through `inject_frames` with the contrasts negated, so the
  placement, true north, center, flux unit and throughput are a draw's.  Every reduction sees
  the edited frames (the clean pass, the draws, `_clean_bins`, validation, param-verify) and the
  KLIP-FM model cube does not.  Contrasts of 0 are skipped, so a list of zeros changes nothing
  to the bit.  The edited frames are a copy, and a partitioned reducer refuses once its worker
  processes run.  The edit is in the reducer's `data_edits` and `describe()`.  The fraction of
  the sources' flux that left the frames is measured on 32 frames against the sources in zeros
  (`subtracted_kept`, in the log), and a loss to float32 rounding above 0.1 % warns.  This
  replaces the per-source headroom check there, which warned for the faintest sources of a
  chain whose contrasts pass through 0, where they carry almost none of the chain's flux.  On
  NEAR nights 4 and 6 a 216-source line left the frames to 1.3e-4 or better.
  `injection.read_sources` reads `rho pa contrast` files.
- **`--cube-name NAME`** (near: the cube in each night's directory) and **`--subtract-sources
  FILE`** (near, generic, nomic, ales) on `near`, `generic`, `resume`, `extend`, `replay` and
  `fit-chain`; `near.make_reducer(cube_name=, subtract_sources=)`.
- **`RunConfig.data_edits`**: a new run records the reducer's `data_edits` (cube name,
  subtracted sources, the file), so the checkpoint holds them.  `resume` and `extend` take them
  from the checkpoint when the options are not typed and refuse typed values that differ, and
  the `Runner` refuses a resume whose reducer reduces other frames.
- **`--forbidden-pa` on `near`** (`near.default_config(forbidden_pa=, pixel_mask=)`), and
  **`--mask-line PA HALFWIDTH`** (near, generic, ales): the strip about a line through the star
  left out of the noise apertures, `metrics.line_mask`.
- **`klip-tpe fit-chain`** (`klip_tpe.chainfit`): a chain's node values fitted to a finished
  run's final image.  Each node's basis goes through every annulus' winner as the KLIP-FM model,
  the FM images are stitched with the final stitch's weights, and the fit is noise-weighted in a
  strip about the line, with masks, the run's known sources and chosen candidate tiers left out,
  inside the run's annuli or `--rho-range RMIN RMAX`.  A node with no signal in that region is
  refused rather than fitted to 0 +- 0.  It writes the values with errors, the fitted chain for
  `--subtract-sources`, and the model, residual, basis and region images, beside the run
  (`<run>_chainfit/`), which it only reads.  On synthetic data with a two-node line it recovers
  0.991 +- 0.023 and 0.605 +- 0.023 for 1.0 and 0.6, and the 8 values of a four-node line on
  both half-lines come back within 0.6 sigma.  `--per-night` fits one amplitude per night to the
  fitted chain (`fit_per_night`).
- **A line by its nodes** (`read_line_nodes`, `line_chain`, `line_basis`,
  `psf_line_integral`): the PA, the step, `rho_max` and the ridge's peak contrast at each node on
  both half-lines, piecewise linear and held at the last node.  The chain puts a source every
  `step` px at `a(s) step / L`, with `L` the injection PSF's line integral along the radius
  (7.6 px for NEAR's n4 library PSF, and 7.1 and 7.2 px for the B templates of nights 4 and 6
  beyond 1.05").
  `--subtract-line FILE` subtracts it at load, `fit-chain --line-nodes FILE` fits the node values
  and writes `line_fitted.txt`.
- **`--subtract-sources` refuses a file whose contrasts are all negative**: it takes the
  sources' own contrasts, and a chain written to be injected (`c = -a step / L`) would double
  the line.
- **The float32 headroom check reads the 99.9th percentile of the bright pixels** in place of
  the brightest one (`injection.cube_peak`).  On NEAR night 6 one pixel at 1.9e3 counts, 29 px
  out, made it warn that 1e-6 sources at 0.5" would be lost, where they land on pixels of order
  10 counts.  The new level there is 67 counts.
  It judges a negative contrast by its size, as before, and only ever warns.
- Tests: `test_subtract_sources.py`, `test_near_data_edits.py` (one slow, the command line from
  `near` to `resume`), `test_chainfit.py` (one slow, `klip-tpe fit-chain` on generic cubes),
  `test_line_model.py`.

## Unreleased — 2026-10-15c (ALES over several nights: a data set in the release layout, the MWC 758 data of tutorial 7)

- **`ales.load_release`** reads an ALES data set in the release layout as one partition per
  night and wavelength group, `<night>_<group>` (`n1_g07`).  Per night the layout has a meta
  file (a template per group, then `ANGLES`, `FRAMES`, `GROUPS` and `WAVES`) and one
  `(frame, y, x)` cube per group, and every night shares one group grid, so a group is one
  band on all of them.  Nights whose grids differ are refused.  `nights=` picks nights by date
  or label (a night loaded alone keeps its label), `only=` picks groups on every night, and
  `labels=` renames the nights.  The plate scale and the angle convention come from the meta
  header.  `ales.release_nights` lists a data set's nights.
- **`datasets.fetch("lbti_ales_mwc758")`**: LBTI/ALES of MWC 758 on 2019 Nov 13 and 14, from
  the `tutorial-data` release.  It holds 959 and 1594 frames of 2.46 s, 109 and 127 deg of
  rotation, as 20 groups over 2.90-4.15 um, 786 MB in 42 files.  The frames come from
  K. Wagner's IDL ALES reduction, and they are the frames with the star at the center.
  `fetch(..., roles=[...])` downloads part of a set.
- **`make_reducer(..., sdi=True)` refuses partitions from more than one night**: the SDI
  pool would lend one night's frames to the other's groups.
- **Command line:** `--ales-dataset NAME` in place of `--root`, and `--ales-nights NIGHT ...`.
  `klip-tpe resume` restores both.
- **Tutorial 7** (`tutorials/07_lbti_ales_mwc758.py` and its notebook, without outputs until it
  has run on the data): the two nights as 40 partitions, MWC 758c (0.603", PA 228.5 deg in the
  Nov 14 data, Wagner et al. 2023) as a known source, a search of 0.4-0.8" with c in its
  middle at the production budget of `docs/BUDGET.md` (3000 evaluations, 200 of them warm-up),
  the group flags bounded at 20 and bins from 5, and the winner's partitions, images and
  contrast curve.  README and `docs/TUTORIALS.md` list it.
- `docs/ALES.md` section 7.  `tests/test_ales_release.py`: the loader on two synthetic nights,
  picking nights and groups, mismatched grids, SDI across nights, the space (one block and a
  flag per partition), the data set entry and the command line, and a tiny search (slow).
  `tests/test_dataset_cache.py` uses a three-file set by name, since the ALES set now sorts
  first, and checks `roles`.
- **The quick suite passes on macOS.**  The Wayland tests of `tests/test_winsize.py` set a
  Linux platform for the session they simulate, and the pyklip/numpy test simulates a numpy
  with `reshape(copy=)` as well as one without, so neither depends on the machine (a Mac with
  numpy 1.26 failed five tests).
- **`klip-tpe view` shows the calibration panels.**  While an annulus calibrates its
  injection contrast, for minutes before the search draws its first panel, the window
  showed a black frame.  It now shows the newest `steps/calib_annNN_trialNNNN.png` until a
  search panel is newer.

## Unreleased — 2026-10-15b (one Gram matrix per zone for the per-target KLIP behind `--klip-gram zone`, KLIP-FM on the basis' own decomposition)

From the NEAR2 session's note of 2026-10-14 on the per-target KLIP.  At a small bin a night job
is KLIP-bound, and for every target the per-target path formed the covariance `R Rᵀ` of its
references from their pixels and all M of its modes, of which it kept k.

- **`KLIPParams.gram`, the request parameter `klip_gram`, `--klip-gram {target,zone}`.**
  `zone` forms one Gram matrix `G = D Dᵀ` per zone before the targets (the pool's under a
  searched library, with the targets' products with it beside), reads each target's
  covariance off it as `G[refs, refs]`, and projects the target on its first k modes as a
  weighted sum of its references, `T - w R`, with `w` from `G[t, refs]` and the eigenvectors,
  without forming the modes.  The k-scan takes every k from one set of cumulative weights,
  `spat_mean` centres the zone once, and `temp_mean` keeps the per-target path.  The fast path
  forms only the k modes it keeps.  It is the same KLIP to float64 rounding, not to the bit,
  so `target` stays the default.
  - On NEAR2 night 6, one thread, the night's KLIP: bin 5 with annulus 5's zones (37.4-50.7
    px, two zones, k 45, about 200 references per target) 7.61 s against 3.51 s, bin 9
    1.36 s against 0.76 s, bin 20 0.21 s against 0.08 s, zones of 56-69 px at bin 5 8.75 s
    against 3.89 s, and an inner annulus with five zones and about 300 references 19.4 s
    against 15.2 s, where the eigendecomposition dominates.  Up to 18 residual pixels in a
    million differ, by one float32 unit.
  - Over the 52 A/B reductions (NEAR2 nights 6 and 4, a searched library, an ALES cube), 108
    of 113 arrays are the same to the bit, and 26 pixels of 211,670 differ, by at most 1.2e-7
    of the image's noise.  NaN masks, dropped targets and reference counts are the same.
- **A run keeps its mode.**  A command line that changes it writes the new mode and its time
  into `klip_gram.json` in the run directory, and every later resume reads it back, however it
  is typed.  `klip-tpe resume --klip-gram zone` switches a stopped run from there on.
  `docs/RUNNING.md`.
- **KLIP-FM takes the basis' eigendecomposition** (`klip_fm_zone(..., eig=)`) instead of
  decomposing the same references a second time, per target and in the fast path.  The numbers
  are the same to the bit, FM images included, and at bin 5 with annulus 5's zones a reduction
  with KLIP-FM takes 23.7 s where it took 29.9 s (20.7 s with `zone`).
- `klip_tpe.reducer.set_klip_gram(reducer, mode)` sets the mode on every partition.
  `tests/test_klip_gram.py` holds the two modes to each other on ADI, the k-scan, KLIP-FM,
  `spat_mean`, dropped targets, the fallback below four references, the fast path, a searched
  library and ADI + SDI + RDI, on one thread and three.  It holds `target` to the old
  per-target path to the bit, and checks the command line and the run's record.

## Unreleased — 2026-10-15 (`ales.WISPIT2B` removed)

- `klip_tpe.instruments.ales.WISPIT2B` is gone.  A companion's position goes to
  `ales.default_config(known=[(RHO, PA)], forbidden_pa=[(PA, 25.0)])`, or to `--known RHO PA
  --forbidden-pa PA 25` on the command line, as for any target.
- The ALES documentation, docstrings, command-line help and tests no longer quote numbers from
  the first ALES data set.  `tests/test_ales.py` checks the group rule on an ALES-like
  wavelength solution and the reference means on a synthetic block layout.

## Unreleased — 2026-10-14c (the frames before KLIP one bin group at a time: a night read once per reduction, no injected cube held)

From the NEAR2 session's note of 2026-10-14 on the per-frame passes of a draw (its items 1 and
2).  Before KLIP a draw wrote and read whole cubes about seven times (the injected copy of the
night, the compact copy of its kept frames, the high-pass, the binning), a clean reduction about
five times.

- **`KLIPReducer.prepare` goes one bin group at a time** (`_prep_fused`).  A batch of whole
  groups, at most five frames of 150² past its first group, is copied out of the night, given
  its sources, high-passed, destriped when asked and averaged into its binned frames while it is
  in the processor's cache.  The groups are spread over the job's threads.  A job reads its kept
  frames from memory once and writes only the binned cube.  Each step is the whole-cube step's
  own function on the same frames in the same order, so every number is the same to the bit.
  `prep_fused=False` keeps the whole-cube path, the reference the tests hold the fused one to.
- **No injected cube.**  The sources go straight into the batch
  (`klip_tpe.injection.place_source`, the arithmetic of `inject_frames` in the same order), and
  the KLIP-FM model cube is made the same way from zero frames.  A set of sources reduced a
  second time on a night (param-verify's fixed set, a fixed-source search) keeps its stamps from
  then on, about 70 MB a night for three sources on NEAR, where the frame cache held the whole
  injected night.  After a run of draws on one NEAR night a worker's resident peak is 546 MB,
  where it was 884 MB.
- **The float32 headroom check** sampled the cube for its peak on every new set of sources, so
  on every draw.  Its stride touches most of the cube's cache lines, which made it most of
  another read of the night.  The reducer now takes the peak once (`KLIPReducer.cube_peak`,
  `klip_tpe.injection.cube_peak`).
- New helpers: `klip_tpe.klip.group_mean` (the binning's mean, which `bin_by_groups` now calls)
  and `group_bounds`, and in `klip_tpe.injection` `stamp_crop` and `add_crop` (where `add_stamp`
  puts a stamp and what it adds there) and `prepare_sources`.
- On NEAR2 night 6 on one thread, with memory kept warm, a draw's preparation takes 1.15 s where
  the whole-cube path takes 1.25 s at bin 29, and 1.16 s for 1.26 s at bin 20.  That machine's
  260 MiB cache holds most of a night, so the passes cost little there.  Where many workers
  share the memory bus each pass goes to main memory.  Each cube-sized temporary of the
  whole-cube path is also a fresh mapping, which the kernel zeroes on first touch, and the new
  path makes none.
- `tests/test_prep_fused.py` holds the fused frames to the whole-cube ones to the bit: 22
  configurations on 1 and 3 threads, every injection model, NaN frames, empty bins, a float64
  cube, whole reductions, the searched library's ranking, the kept stamps, how often the
  headroom check runs, two threads at once and pickling.  On NEAR2 nights 6 and 4, a synthetic
  searched library and an ALES cube with NaN registration edges, every image of 52 reductions
  (113 arrays with the FM, nosub and cADI images and the frame counts) is the same to the bit
  as before, on 1 and 2 threads.  So are the prepared frames themselves (the binned cube, its
  angles, groups and kept bins, the model cube) of 180 preparations on the two nights, ten
  settings each clean, injected and with KLIP-FM, for a new set of sources, the same set again
  and the set from its kept stamps.

## Unreleased — 2026-10-14b (ADI, SDI and RDI in one KLIP step for IFS data, reference means within observing blocks)

From the ALES reduction's answer of 2026-10-14 (its sections 1 and 2).  An IFS cube loaded as
wavelength groups gets a third kind of reference pool: the other groups' frames, rescaled to the
target group's wavelength, with their count searched, so ADI, SDI and RDI are points of one
space and the optimizer elects the mix.

- **`klip_tpe.rescale`**: `rescale_frames` magnifies frames about the star by `s` with a
  NaN-aware cubic B-spline (`scipy.ndimage.map_coordinates`, the ALES reduction's own choice),
  `displacement_px` is how far a planet at the annulus' mid radius moves between a target and a
  reference rotated by `dPA` and magnified by `s`: `r sqrt(1 + s^2 - 2 s cos dPA)`, the ADI chord
  at `s = 1`, the spectral shift `r |s - 1|` at `dPA = 0`.
- **`ReferenceGroup(scales=...)`** (`reflib.py`): a pool of rescaled rows, one scale per row,
  eligible by the displacement rule at the library's `mid_px` (the arc `angsep` asks for, in
  pixels, within `anglemax` of rotation), no fallback; `is_plain_adi` counts it among the pools
  that must be at zero.
- **`KLIPReducer.prepare`** is the first half of `reduce` (selection, injections, model cube,
  high-pass, binning), and a request with `stage="prepare"` returns it as `SiblingFrames`.
  `set_reference_library(sdi={...})` declares the SDI pool (`n_frames`, `cap`, `lenders`);
  `reduce` with `ReductionRequest.siblings` rescales every sibling's prepared frames to its own
  wavelength (`_sdi_rows`), puts them between the science frames and the reference star's in the
  basis `[bcube | sdi | ref]`, ranks them at zero lag against this evaluation's target frames
  (never cached: a cached ranking would depend on which evaluation came first, a clean prep of
  every sibling would double the rescaling), and gives KLIP-FM the siblings' model cubes rescaled
  the same way, so the forward model's planet sits at `r s`.  `cost_estimate` counts the extra
  rows and the rescaling.
- **`PartitionedReducer`** runs a first round of `prepare` jobs over the lending partitions when
  the configuration's `nkeep_sdi` is above 0 (nothing otherwise: that point is plain ADI at plain
  ADI's cost) and hands the frames to the partition jobs in one shared-memory block
  (`klip_tpe.parallel.SharedArrays`, a descriptor the jobs carry, attached without a resource
  tracker registration of their own) with worker processes, as they are with threads.
  `sdi_lenders` is `"all"` or `"selected"`; `describe()` records it.
- **ALES: `--ales-sdi`**, `--ales-sdi-cap N` (100), `--ales-sdi-lenders {all,selected}`:
  `nkeep_sdi` joins the shared block beside `nkeep_psfref`, both seeded at 0; a reference star
  given is the third pool, searched; `run_setup.json` records the pools, the rule, the rescaling
  and the ranking; `resume` restores the options.  `docs/ALES.md` section 4.
- **Reference means within observing blocks**: `reference_means(blocks=...)` never lets a mean
  straddle a block, the blocks come from the `block` column of `<ref>_frametags.csv` beside the
  products (`frame_blocks`), the whole sequence is one block without it; `run_setup.json` records
  the block sizes.  A reference star's means now match the reduction's own `bin5` column, where
  one straight run had put means across the gaps between its blocks.
- `tests/test_sdi.py` (the rescaling, the displacement rule, the shared arrays from a forked
  child, `nkeep_sdi = 0` to the bit through the partitioned reducer with processes and threads,
  the three pools together and the pure-SDI corner, the FM rows at `r s`, the lenders switch,
  the cost estimate) and `tests/test_ales.py` (the adapter, the command line, a tiny run with the
  three pools and KLIP-FM on worker processes).  Checked on an ALES cube over three annuli:
  `nkeep_sdi = nkeep_psfref = 0` is plain ADI to the bit, the SDI count changes the images from
  1 up.  The run itself is reported with the data.

## Unreleased — 2026-10-14 (a searched reference library for one rotating sequence: the ADI rule as its eligibility, a reference star's count from plain ADI to the whole pool)

From the ALES reduction's note of 2026-10-14.  A searched library (`klip_tpe.reflib`) chose a
target's science references by partition label alone: the roll, night or channel a frame
belongs to.  One continuously rotating sequence has no label to exclude by, and under the
library `reference_mask`'s rule (`angsep`, `anglemax`) was not applied, so switched on as it was
the library would have admitted references with no rotation and left `angsep` and `anglemax`
dead.

- **`ReferenceGroup(rotation=True)`** (`reflib.py`): a science pool the ADI rule selects from, per
  target and per annulus.  `ReferenceLibrary` carries the frames' angles; `eligible` admits the
  rows with `angsep_deg <= |dPA| <= anglemax`, never the target, the label rule on top only when
  there is more than one label, and falls back to every other row below 4 as `klip_annular` does;
  `select` and `mask` take the two angles, and a `census` argument asks for the count before the
  fallback (what the starved-target rule reads).  `is_plain_adi(keep)` says whether a count
  vector is the ADI point: every eligible science frame kept, no reference frame.
- **`klip_annular`** passes `angsep` and `anglemax` into the library path, and at the ADI point
  runs the ADI code itself (`info['library_as_adi']`), its shortcuts and k cap included, so "no
  reference frames" is plain ADI to the bit and not a per-target imitation of it.
- **`KLIPReducer.set_reference_library(rotation=..., searched=...)`**: `rotation` builds the
  science pool that way and hands the binned angles to the library; `searched` names the pools
  whose counts are search dimensions (default: all).  A pool left out is not a dimension at all,
  every one of its eligible frames is in the basis, as without a library.  `describe()` records
  the library (metric, floor, pools and which are searched).  `generic.make_guard` adds
  `ReferenceLibraryGuard` only when every pool is searched, since the guard reads the basis size
  off the searched counts.  `tests/test_reflib_rotation.py`: the rule decides the rows, a
  one-label pool without the rule stays empty (the NIRCam precondition), the full library is
  plain ADI to the bit through `klip_annular` and the reducer at several `angsep` / `anglemax`
  and with KLIP-FM, a truncated pool is not.
- **ALES: `--ales-ref NAME --ales-reflib`** searches the reference star as a library: each
  target frame's basis is the ADI-eligible science frames plus the `nkeep_psfref` reference means
  most correlated with it in the annulus, `nkeep_psfref` one more dimension of the shared block
  from 0 (plain ADI, the seed) to the whole pool.  `--ales-ref-bin N` (5) takes the reference
  frames in consecutive means first, so the count means one thing whatever the science `bin` is;
  the cache keeps them unbinned.  `--ales-reflib-science {all,searched}` keeps the science pool
  whole (default) or searches `nkeep_altroll` as well, with the library guard then last.
  `run_setup.json` records the pool (`reducer.provenance.reference_library`, and per group).
  `resume` restores the three options.  `docs/ALES.md` section 3, `tests/test_ales.py`.
- **`--bin-max N`** (generic and ALES): the top of the searched temporal-bin range, for a winner
  that sits on it.
- Checked on an ALES cube over the whole field: `nkeep_psfref = 0` through the adapter is
  plain ADI to the bit (the clean image, every group, an injected image and its score), and the
  count changes the images from 1 up; at bin 5 on two cores the whole pool costs a 20-group
  evaluation about a third more than ADI.  The run itself is reported with the data.

## Unreleased — 2026-10-14 (LBTI/ALES integral-field cubes: wavelength groups as partitions with on/off flags)

From the ALES reduction's handoff of 2026-10-13 (the package's first IFS data set).

- **`klip_tpe.instruments.ales`, `klip-tpe near --instrument ales`** reads an ALES reduction's
  products (the `(wavelength, frame, y, x)` cube, `LBT_PARA`, the wavelengths, the per-channel
  template) and makes one partition per wavelength group: the usable channels 6-88 cut into
  20 groups of equal width in ln λ (the handoff's rule, §6.2, reproduced to the channel), each
  group's channels meaned into one cube, with its own wavelength and FWHM (1.03 λ/D), its own
  template normalised in 3 spaxels, its own quality tags from 2 spaxels out, and the group
  mean of a reference star's cube as its RDI library when asked.  The angles carry the +180°
  once.  The group table and each group's channels go into `run_setup.json`.  `docs/ALES.md`.
  - One parameter block is shared by every group and `selection='binary'` adds a 0/1 flag per
    group.  `GroupCountGuard` bounds the flags: a proposal with more than `max_excluded` (10)
    groups off has groups switched back on, nearest 3.7 µm first, at random for a random draw.
  - `PartitionedReducer` takes a `representative` partition whose FWHM, λ/D and matched filter
    stand for the combined image (the first partition before, the bluest group here); the
    adapter names the group at the mean wavelength.  `describe()` records it, the partition
    label and a `provenance` dict.
  - `KLIPReducer` default `hp_nan_aware` (False as before): the science frames' high-pass with
    `smooth(/nan)` semantics.  The registered ALES frames carry NaN along their edges, and IDL's
    boxcar carries a NaN along the whole frame, which empties every bin holding such a frame.
    The adapter sets it.
  - The adapter's metric does not de-spike (`declip_nsig=None`): at 2.1-3.0 spaxels per FWHM a
    source's peak stands 6 robust σ above its 3 x 3 median and was replaced by it.
  - `--bin-min N` on the command line floors the temporal bin (`MinBinGuard`) for the generic
    and ALES guards; `--forbidden-pa PA HALFWIDTH` reaches the sampler for both.
  - On the first ALES cube the derotation puts the known companion where the reduction puts it,
    and the reduction's own parameters give it close to the S/N the reduction measures.  The
    run itself is reported with the data.  `tests/test_ales.py`.
  - `generic.make_reducer` centres a template at the loader's `meta['psf_center']` when there is
    one (the ALES header's `STARX`, `STARY`), and at the centroid of its peak otherwise, as before.

## Unreleased — 2026-10-13 (injection only into the kept frames and on the job's threads, the display's annulus books in a process of their own, `--n-sources`, `--min-r-fwhm`, the metric's inner guard at the source's own radius, failed draws that say why)

From four NEAR2 notes of 2026-10-13: the parallelism note, the note on the source count, the
note on the injection floor and the note on the metric's inner guard.

- **A job injects only the frames its selection keeps, on its own threads.**  On patrick
  `inject_sources` was the largest fixed cost of a job, 0.42-0.49 s per n6 job at any bin,
  on one thread and into all 2024 frames of the night.
  - The frame selection reads the frame tags, not the pixels, so `KLIPReducer.reduce` now
    selects first and injects the kept frames alone.  Each night keeps the frames it has
    injected for its latest sources, so a reduction of the same sources with another
    selection injects only the frames it adds.  The KLIP-FM model cube works the same way.
  - `inject_sources` takes `frames` and `threads`, and the new `inject_frames` writes the
    injected frames into a cube the caller holds.  The frames go to the job's threads in
    blocks.  Within a block one `rotate_ccw_many` call rotates a source's stamp to every
    frame's angle, without the GIL.  The shift and the add stay frame by frame.
  - Every injected frame is the same as before, bit for bit.  Each frame takes its stamps in
    the order of the sources, with the sines, cosines, sample positions and interpolation
    calls of the old loop.  `tests/test_injection_frames.py` compares the injection with
    the old loop for a stamp shared by all frames (rotated or not), the fallback template, a
    PSF per frame and a throughput per frame, on one to three threads.  Whole reductions of
    NEAR2 n4 and n6 gave the same images, KLIP-FM, nosub and cADI images as 7fbe593, with
    cuts that act and cuts below `k_klip × bin`, a reused cache, k-scan, `hp_order="binned"`,
    destriping and the searched reference library.
  - On the sandbox (2 cores, n6, 3 sources, median of 5 draws, three runs), the injection
    took 1.0-1.2 s per job before.  It now takes 0.87-0.91 s on one thread and 0.66-0.73 s
    on two with every frame kept, 0.61-0.71 s and 0.52 s with 1349 frames kept, and 0.12-0.14 s
    and 0.08-0.09 s with 165.  A whole n6 reduction with 1934 frames kept went from 2.3-2.6 s
    to 2.0-2.1 s on one thread and 1.4-1.6 s on two.  Three fifths of what is left are the two
    scipy interpolations, whose cost per pixel does not change with the batch.
  - The cost estimate that orders the jobs counts the injection per kept frame, at the
    measured cost relative to the high-pass.
  - The high-pass already ran after the frame selection, as in `reduce_near_2`.  A job that
    high-passes all 2024 frames of n6 kept all of them, because a cut that would leave fewer
    than `k_klip × bin` frames (at least 3) is skipped, the rule of `near2_framekeep.pro`.
- **The display's annulus books are drawn by a process of their own.**  `heartbeat.json` and
  the stall note named the candidate search for the minutes between annuli.  On the running
  stitch of `run_20261008_093817` the search takes 0.15 s in the sandbox (20 peaks), and the
  final, verified pass 5 s.  The minutes were the display's books, drawn on the run's main
  thread after the hooks while every worker waited.  In that run (Mac-Pro-5, 10,000
  evaluations in annulus 1) param_verify finished at 15:23:54, `corner.pdf` was written at
  15:27:46, `landscapes.pdf` at 15:46:08 and the final panel at 15:49:54, and annulus 2 began
  calibrating at 15:50:06.  Its parameter walks were still drawn line by line, which 7fbe593
  replaced.  The "expected ~38s" was the stall threshold of the evaluations before
  (300 s / 8).
  - The ten books of an annulus (corner, landscapes, parhist, edf, partition map, kbook,
    importance, paracoord, rank, slice) are drawn by `python -m klip_tpe.books` from a
    pickled snapshot of the same annulus data, with the same functions in the same order
    (`klip_tpe/books.py`).  At 5,000 evaluations the run spent 218 s on them and now spends
    0.09 s handing them over, and the process wrote all ten in 205 s, byte for byte the same
    files (with `SOURCE_DATE_EPOCH` set, which fixes the PDFs' creation date).
  - The process runs at a lower priority than the workers.  The display reports the books
    in the log once they are written, and the run waits for the last of them at its end.  A
    Ctrl-C of the run does not stop the process, which finishes the books.  When it ends
    without finishing, the run draws them from the snapshot.  `KLIP_TPE_BOOKS=inline` draws
    them on the main thread as before.
  - The verify books and the products sheet read the run directory as it stands, and the
    validation panel and the annulus' last frames come from images the run holds in memory.
    They stay on the main thread, a few seconds under the stage `annulus N display books`.
    The post-annulus hooks and these carry no stall threshold, as the final products do, and
    the next evaluation sets it again.
- **`--n-sources` on the command line.**  `klip-tpe near` and `klip-tpe generic` take
  `--n-sources N [N ...]`, the injected sources per annulus, the last value repeating, and a
  0 asks for the rule in that annulus.  It sets `RunConfig.n_sources`, which the command line
  never set before, and `run_setup.json` and `run_setup.txt` record it with the other fields.
  Each annulus' first log line gives the count used and where it came from, as in
  `2 sources (n_sources 2)` or `6 sources (the rule)`.  The cap that keeps each source's noise
  ring clean still applies, and its own log line says when it binds.
- **The injection floor on the command line.**  `--min-r-fwhm X` (`near`, `generic` and
  `--instrument nomic`) keeps every injection outside X FWHM of the star, 1.5 by default as
  before.  It sets `RunConfig.min_r_fwhm`, which puts the floor on the run's one sampler, so
  search, calibration, validation, param_verify and the products inject with it, and a resume
  puts it back on the sampler the resumed run builds.  On NEAR 1.0 FWHM is 1.0 λ/D to half a
  percent (0.284″).  With the NEAR2 annuli from 0.10″, annulus 1's single source moves from
  0.427″, where the old floor clamped it, to its mid-radius of 0.312″.
  - When the count used still leaves a source's Mawet ring fewer than `min_ring` clean
    apertures, the log says so once per annulus, since that source's S/N then comes from the
    radial band.  It counts the apertures at the source over 12 azimuths, because where the
    source falls on the pixel grid moves the count by one.  On NEAR annulus 1 a source keeps
    3 to 4 at 0.312″, and 5 to 6 at 0.427″.  The candidate search keeps its own inner limit.
- **The metric's inner guard asks where a source was put.**  `run_20261013_170907`
  (`--min-r-fwhm 1.0 --n-sources 1 2 --n-remeasure 5`) was stopped after 443 evaluations with
  1,110 of its 2,219 draws failed.  Annulus 1's one source sat at 0.312″, 1.10 FWHM.
  `mawet_peak_snr` locates a source as the filtered image's peak within 1.5 px and gave no S/N
  to a peak inside 1 FWHM, as `near2_snrpk` does.  On the injected image that peak is the
  planet.  On the clean image it is the highest noise pixel nearby, which lay inside 1 FWHM
  about half the time, so the clean term was NaN and the draw failed.  It depended on position
  angle through the pixel grid, it left `partition_snr` empty for 7-14 % of the pairs of
  evaluation and night, and it made calibration trial 3 a NaN, so the annulus ran
  uncalibrated.  IDL's 1.5 FWHM floor never let a peak move that far in.
  - The guard, and the rule that a ring needs at least 5 apertures, are now asked of the
    radius the source was given.  The ring is still laid through the located peak.  Wherever
    the old guard passed, the S/N is the same.  Of 2,729 random sources both versions scored,
    all scored the same to the bit, and only sources put inside 1 FWHM lost a score.  On NEAR2
    n6 the same short run of annulus 1 failed 14 of its 30 draws before and none now, and a
    calibration trial that came back NaN before is gone.
  - The threshold is `MawetPeakSNR.inner_fwhm`, 1.0 as before, also on `FMMFSNR`.  The Runner
    lowers it to the injection floor when the floor is closer in, so a run with
    `--min-r-fwhm 0.9` is scored from 0.9 FWHM, and says so.  It never raises it.
    param_verify's contrast curve scores with the same guard, `run_setup.json` records it
    under the objective, and a resume puts it back.  Below 0.80 FWHM a ring holds fewer than 5
    apertures and a source there still has no S/N.
- **A draw without a score says why.**  `score=failed` said nothing else for 400 evaluations.
  - Its log line gives the reason, as in `draw 3/5 score=failed (clean S/N NaN (inside the
    inner guard))`.  The reason names the image that had no S/N and the metric's own reason
    (`inside the inner guard`, `fewer than 5 apertures on its ring`, `no noise samples on its
    ring or band`), or the error of a reduction that raised.  `results.jsonl` keeps it as
    `meta['why']`, and a trial's draws as `meta['draw_why']`.  `calibration.json` and
    `validation.json` keep theirs as `why` and `trial_why`.
  - A trial's line counts its failed draws, as in `(mean of 5, sd 0.38, 2 failed)`, and the
    calibration and validation lines count theirs with the commonest reasons.  The log warns
    once per annulus when a tenth or more of at least 20 draws have failed, and the end of each
    search gives the count.

## Unreleased — 2026-10-12 (checkpoints that survive a machine crash, less work per evaluation on the main thread, each trial's own times on the panel)

From three NEAR2 notes of 2026-10-12: the checkpoint-durability note, the 552d9cc section of
the parallelism note, and the note on per-trial timing.

- **Checkpoints survive a crash of the machine.**  patrick went down hard 2.7 h into
  `run_20261011_145111`, and every file the run wrote in its last ~30 s came back with 0
  bytes, `checkpoint.json` among them, so the run could not resume.  Writing a temporary file
  and renaming it made each write atomic but not durable.
  - Every checkpoint write fsyncs the new file before the rename and the directory after it
    (`klip_tpe/durable.py`).  On macOS the new file also gets `F_FULLFSYNC`, since `fsync`
    stops at the drive's cache there.
  - The save before the newest moves to `checkpoint.prev.json` before the new one takes the
    name, so a crash at any moment leaves one complete checkpoint.  A `checkpoint.json` that
    cannot be read never replaces a good `checkpoint.prev.json`.  A resume still reads every
    `checkpoint*.json` and takes the one furthest along, so the interim copies
    (`checkpoint.bak.json`) keep working, and the backup loop can stop once patrick runs this.
  - `results.jsonl` and `results.txt` are fsynced before each checkpoint write, so no
    checkpoint counts a line that a crash can take away.  The products a resume reads back
    for a finished annulus (`best_*.fits`, the partition stacks, `winner.json`,
    `contrast_curve.txt`), the verify tables and the param_verify maps reach the disk before
    the checkpoint that records them.  `validation.json` and `val_candNN.pkl` reach it before
    the checkpoint that pairs with their RNG state, and are now written atomically as well.
    `run_setup.json`, `run_setup.txt` and `calibration.json` go with the next checkpoint.
    `final_results.json`, which marks a run finished, is written durably after the final
    stitch it lists.
  - The fsyncs come with the checkpoint writes, at most every 10 s in a search plus once per
    validation candidate and post-annulus hook.  On the sandbox's disk a 4.6 MB checkpoint
    took 9-15 ms to write with them and 5-9 ms without.
  - A resume cuts a half line off the end of `results.jsonl` and `results.txt` before it
    appends, since the next record would fuse with it.  `klip-tpe runs` counts complete lines
    only, and the bench and plot readers skip a half line or an empty JSON file.  The run
    registry starts each record on a line of its own.
  - The validation pickles carry the RNG state from before and after their candidate's
    draws.  A resume from `checkpoint.prev.json`, after a crash between its two renames, then
    draws the next candidate's positions as an uninterrupted run does.
  - An extend's saves outrank everything from before it (`extends` in the checkpoint).  A
    re-validation adds no evaluation and goes back an annulus, so by the other counts the
    finished run's `checkpoint.prev.json` looked further along.
  - A param_verify cache entry cut short by a crash is reduced again instead of failing the
    hook, which the checkpoint would then have recorded as done.  Its entries are fsynced
    before their rename, and so are downloaded data sets.
  - An empty `heartbeat.json` already read as "unknown" in `klip-tpe runs`, and a test now
    keeps it that way.
- **Less work per evaluation on the main thread.**  With four trials in flight (552d9cc) the
  NEAR2 search slowed from 0.50 s/eval over evaluations 501-2500 to 0.67 s over 4501-5040, at
  a constant bin mix.  Three things the main thread did after every evaluation took longer as
  the history grew.
  - The routine checkpoint is still taken after every evaluation, but serialized only when
    it is written, at most every 10 s.  `json.dumps` of the whole history took ~100 ms at
    5,040 evaluations of a 56-D space.  A save held back is written as it stood when taken.
  - A TPE proposal computes its 48 candidates' densities with one call per dimension instead
    of one per candidate.  The proposals and the generator state after them are the same bit
    for bit (`tests/test_optimizers.py` compares them with the old loop).
  - The live display rebuilt its annulus data from every record after each evaluation, 66 ms
    at 5,000 records, for a panel drawn one evaluation in `--display-every`.  It now does
    that, and the contrast preview, only for an evaluation it draws.
  - The parameter-walk pages of `landscapes.pdf`, drawn on the run's main thread at the end
    of an annulus, draw each pair's walk as one collection.  One line per step made 180,000
    artists per partition page at 5,000 evaluations.  A page took 16 s at 1,000 evaluations,
    growing with them, and now takes 3 s at 5,000 and looks the same.
  - A NEAR2-shaped synthetic search (6 nights, 56 dimensions, 3 draws, 4 trials in flight,
    fast reductions), resumed at evaluation 4,800 in the sandbox, ran evaluations 4800-5000 at
    346-354 ms each and now runs them at 209-214 ms.  Its main thread never waited, so that is
    the main thread's time per evaluation.  From evaluations 200-400 to 4800-5000 it now grows
    by 38 ms.  At 5,000 evaluations scoring takes 71% of it, the TPE proposal 21% and the
    evaluation's files 7%.  With the live display attached (`--display-every 20`) the same
    evaluations took 1.97 s each on average and now take 1.20 s, with a median of 0.97 s in both.
    The panel renderer holds the GIL while it draws, and on two cores it then sets the main
    thread's pace.
  - `KLIP_TPE_PROFILE_MAIN=S` profiles the main thread from inside a run
    (`klip_tpe/profiling.py`).  Every `S` seconds `main_thread_profile.txt` lists the share of
    the time each function held it, over the last `S` seconds and since the start, and the
    share spent waiting.
- **Each trial's own times on the panel and in `run.log`.**  With four trials in flight the
  panel read "avg 0.6 s/eval (2.1 s reduce)".  The interval between two evaluations is the
  throughput, and `wall_s` runs on to a trial's scoring, so it also holds the wait for workers
  and for the main thread.
  - A record carries `reduce_s`, from its dispatch until the last of its reductions was back,
    and its dispatch time (`meta["t_dispatch"]`).  `wall_s` keeps its meaning.
  - The display is now the last thing the runner does for a trial, after its files, the
    checkpoint and the running stitch.  The panel shows `avg eval X s (reduce Y s)` over the
    last 50 trials and `last eval X s (reduce Y s)` for the newest, with `eval` counted from
    the dispatch until the trial was processed.  `1 done every X s (N in flight)` gives the
    throughput, which still sets both ETAs.  The classic panel's `wall` line shows the same
    two times.
  - Each `run.log` evaluation line ends `reduce Xs / eval Ys`.
  - With one trial at a time, the two numbers mean what the panel's numbers meant before
    trials overlapped.

## Unreleased — 2026-10-11 (pipelined batches, a job's per-frame steps on its threads, the longest job first; high-pass after binning under test)

From the NEAR2 session's second note of 2026-10-11.  It measured da36baf on patrick at 1.66x
(`--workers 96`) and 2.50x (`--workers 192 --batch 2`), with the machine at about 10% CPU.

- **`--batch` keeps its trials in flight.**  A whole batch waited for its slowest trial, often
  one with a KLIP-heavy job, while most of the pool idled.  Now, as soon as the oldest trial is
  in, it enters the history, the next proposal is asked and its reductions go out, and only
  then are the finished trial's files, display and checkpoint written.
  - Every generator call still happens on the main thread in index order, so a seed gives one
    trajectory on any machine and however the reductions finish.  The trials in flight enter
    the optimizer's history unscored, so they count toward the warm-up and the grid.  Guided
    proposals start at eval `n_init`, and a grid never repeats a cell.
  - The checkpoint records the trials in flight (`pending`), and a resume reduces them again
    before it asks for anything new, so the resumed run is the run that never stopped.
  - A re-calibration drops the trials in flight and puts the generator back where it was
    after the trial that triggered it.
  - `RunConfig.batch_mode` (`--batch-mode`): `"pipeline"`, the default, or `"sync"`, whole
    batches as in da36baf.  A run saved before the field existed resumes with `"sync"`.  The
    k-scan modes, whose draws pick their k as they run, use `"sync"`.
- **A partition job runs its per-frame steps on its own threads.**  The high-pass, binning,
  derotation and `nw_ang_comb` ran on one core, and only the KLIP loop used the job's threads.
  - The high-pass filters a few frames per `ndimage` call, which runs without the GIL.  Frame
    by frame, the per-call overhead held it.
  - Derotation and the rotations of `nw_ang_comb` go to the job's threads, and `nw_ang_comb`
    still sums in frame order.
  - A rotation of a finite frame skips its validity interpolation, which marks nothing there.
    All three rotations per frame in `nw_ang_comb` are of finite frames, so at 229 binned
    frames of 150 x 150 it took 2.1 s and now takes 1.1 s on one thread.
  - Every step gives the same numbers on any thread count, bit for bit, and so does a whole
    reduction, checked against the previous code on real KLIP reductions.  One 2024-frame job
    of 150 x 150 frames went from 2.2 to 1.5 s at bin 22 and from 4.4 to 2.6 s at bin 8, on two
    threads of a two-core sandbox.
- **The longest job first, and threads by expected cost.**  `KLIPReducer.cost_estimate`
  predicts a job's single-thread time from the frames kept, the frames after binning, the zone
  and the reference counts.
  - A job gets threads in proportion to its predicted cost against the mean of the latest
    jobs, a power of two from 1 to 4 x `--threads-per-job`, so the average stays at
    `--threads-per-job`.
  - The worker pool keeps only as many jobs in its queue as it has workers, and a free worker
    takes the waiting job with the smallest key.  Among jobs submitted together that is the
    longest.  An older job goes ahead of a newer one unless the newer is longer by more than
    the time between them.
  - A lone partition job goes to the workers too, instead of running in the runner's process.
  - Neither the thread count nor the order changes an image.
- **A forked worker starts its thread pools afresh.**  It inherited the parent's per-target
  pools without their threads, and work submitted to a pool that had all its threads never
  ran, so the worker hung.  The command line starts its workers before any reduction and is
  not expected to have met this.  In the test suite, a pool started after earlier tests had
  used the thread pools did hang.
- **High-pass after binning, under test.**  `hp_order="binned"` in a reduction's parameters
  bins the frames first and high-passes the binned frames.  The means are taken in double
  precision, since the frames still carry their background when they are binned.  It is off
  by default and not on the command line.  `scripts/compare_highpass_order.py` re-reduces a
  run's evaluations both ways and reports the image and score differences against the run's
  own draw-to-draw spread, and one job's time both ways.  On two synthetic nights the scores
  agreed to 8 x 10^-7 and the images to 4 x 10^-6 of their scatter, also under a 10^4-count
  pedestal.  A 2024-frame job at bin 22 took 1.06 s instead of 1.52 s on one thread.
- `tests/test_job_threads.py` checks the steps and whole reductions across thread counts, the
  cost estimate, the thread plan, the pool's order and a worker forked after the parent used
  its thread pools.  `tests/test_concurrency.py` adds the pipeline on one process and on many,
  a resume with trials in flight, the warm-up and the grid, and the generator after a
  re-calibration.

## Unreleased — 2026-10-11 (several reductions in flight: one round per trial, validation in parallel, a pool sized by the thread budget, `--batch`)

From the NEAR2 session's note of 2026-10-11 (proposals 1 to 4).  The first three change when
a reduction runs, never what it returns.  A run writes the records the one-at-a-time schedule
wrote, and a stopped run resumes to them.

- **The worker pool hands each result to its caller at once.**  Every caller of
  `ProcessPool.map` read the shared result queue itself.  A result another caller read first
  waited until its owner's next read timed out, 0.1 s with a live window in the process and
  1 s without one.  A trial's injected and clean reductions are two such callers.  With six
  jobs of 0.5 s each on 12 processes, a round of the two took 1.13 s on average instead of
  0.50 s.  One collector thread now settles each job's future, and the round takes 0.50 s.  A
  synthetic run on 4 worker processes took 32 s against 5.6 s on one process, and now takes
  5.8 s.  A worker that dies still fails the jobs it held.
- **A trial's reductions go out in one round.**  On a parallel reducer the clean image and all
  `--n-remeasure` injected images are reduced together, `(n + 1) x partitions` jobs, and each
  draw is scored as soon as its image and the clean image are back.  They went in two rounds:
  the first draw with the clean image, then the other draws two at a time.  The positions are
  drawn first, in order, and the tags stay `a1_e123_d2_inj`.  A failed clean reduction still
  fails the first draw, and the later draws reduce their own.
- **Validation runs several trials at a time.**  A candidate's trial positions are drawn
  first, in order, each record keeping the generator state after its own draw.  The clean
  image and the trials are then reduced a few at a time, enough to fill the pool, and scored,
  checkpointed and reported in trial order.  param_verify reduces its configurations' clean
  and injected pairs the same way, and so do the calibration remeasurements, the calibration
  k-scan and the scan pair of the k-scan modes.  The KLIP-FM curve is a single reduction.
- **The worker pool is sized by the thread budget.**  `PartitionedReducer` forked
  `min(workers, 2 x partitions)` processes, 12 for NEAR2's six nights whatever `--workers`
  said, so a larger budget only raised the threads per job.  It now forks
  `workers // threads_per_job` processes, never fewer than before, and gives each job
  `workers // processes` threads.  `--threads-per-job` sets it (default 4).  At
  `--workers 96` six nights get 24 processes of 4 threads, and a trial at `--n-remeasure 3`
  (24 partition jobs) runs in one round.  At `--workers 50` the pool stays at 12 processes.
  The thread-pool fallback keeps its old rule.
- **`--batch B` evaluates B proposals concurrently** (`RunConfig.batch`, default 1).  Each
  proposal is projected, given its eval index and its draws in proposal order.  Then all
  their reductions go out together, and the trials are scored and appended in index order.
  The same batch evaluated one proposal at a time gives the same history.  TPE proposes a
  batch from one posterior, which changes the search's trajectory, so it stays off by
  default, and `run_setup.txt` records it.
  - A batch is checkpointed once it is all in, also when it is evaluated one proposal at a
    time.  A run stopped part way through a batch resumes at its start and repeats it
    exactly.
  - A re-calibration inside a batch drops the proposals after it and puts the generator back
    where a one-at-a-time loop leaves it.
- After the search and after validation the log reports the worker processes' peak resident
  memory and what they hold together (Linux), for sizing the pool against the machine.
- The worker processes are forked before the heartbeat thread and before any reduction
  thread starts.
- `tests/test_concurrency.py` compares a serial run with the same run on worker processes,
  record for record, with reductions finishing out of order on purpose.  Runs stopped
  mid-round, mid-validation-chunk and mid-batch resume to the same records.  A concurrent
  batch matches the batch evaluated one proposal at a time, with a re-calibration and a
  failed clean reduction inside it.

## Unreleased — 2026-10-11 (annulus edges padded only where two annuli meet)

- **Only the edges between annuli are padded.**  The inner edge of the innermost annulus and
  the outer edge of the outermost were padded like every other edge: the KLIP zone by the
  reducer's `zone_pad` (2 px), the stitch tile by 2 px, a seam re-run by 3 px and a
  param_verify map by 1 px.  Now each of these applies only to an edge another annulus
  shares, so the reduction and the stitch end where the searched range ends.  On
  `--ann-edges 8 20 32` the stitch used to reach from 6 to 34 px, and now runs from 8 to 32
  px, with the two annuli still overlapping at 20 px.
  - `RunConfig.edge_padding` (`--edge-padding`): `"between"`, the default, or `"all"`,
    which pads every edge as the IDL does.  With `opt_width` an outer edge counts as the
    last when it reaches `r_cap`, or leaves less than the smallest width, where the run
    stops.
  - The runner marks each reduction request with `pad_in` / `pad_out`, and `KLIPReducer`,
    with the pyKLIP and VIP backends and `FunctionReducer` behind it, pads only the marked
    edges.  A request without the marks pads both, so direct reducer calls are unchanged.
  - A run saved before the field existed resumes with `"all"`, so its evaluations stay
    comparable.  `klip-tpe compare`, `klip-tpe replay` and `--from-idl-setup` keep `"all"`,
    since they reproduce IDL reductions.
  - `stitch_tiles`, `stitch_stacks` and `param_verify.stitch_param_maps` take per-tile
    pads, and `products.stitch_annuli` leaves the two end edges unpadded unless `pad_ends`.

## Unreleased — 2026-10-10 (the live window keeps one size and shows the panel sharp; tutorials 3, 5 and 6 with outputs; the trace figure)

- **The live window opens at its size, keeps it, and shows the panel sharp.**  On a 1920 ×
  1080 GNOME session the 1850 × 990 window (the CLI default, `--window-scale 1`) opened
  maximized.  GNOME maximizes a resizable window that covers most of the screen as it
  appears, and the window was fixed only after it appeared.  The 1850 × 990 panel was then
  resampled into the larger window, which blurred the text.  Two seconds later the window was
  un-maximized, and every later put-back aimed at the maximized size it had adopted while it
  settled.  Reproduced on mutter 46, the GNOME of Ubuntu 24.04.  Now:
  - `winsize.open_panel_window` makes the window hidden, shrinks the panel to the room on the
    screen, fixes the size, and only then shows it.  On 1920 × 1080 the window opens at
    1850 × 990, and a title double-click and Super+Up leave it there.  On 1440 × 900 it opens
    at 1424 × 762 (the panel at 0.77), and the log says so.
  - The panel is drawn pixel for pixel on the screen's own pixels (`winsize.show_panel`,
    centered on black, never stretched).  After each panel the render thread draws the figure
    once more at the window's pixel size (about 1 s, against about 4 s for the panel), so the
    text is rendered for the screen instead of resampled to it.  Checked at scale 1, at 2×
    (3700 × 1980 device pixels) and at GNOME's fractional 1.5× (2775 × 1485).
  - With GNOME's fractional scaling the window stays native Wayland.  There GNOME draws an
    XWayland window at scale 1 and enlarges it, which blurs it.  The layout is read from
    mutter's `DisplayConfig` through `gdbus`.  Desktops other than GNOME keep the window
    native, since they draw a Wayland window's frame themselves.  `KLIP_TPE_WAYLAND=x11`
    forces XWayland.
  - A maximized size or a larger one is never adopted, and put-backs stop after ten that do
    not take (a tiling window manager), with one line in the log.
  - A window that runs through X11 on a GNOME session with fractional scaling, as a Tk window
    always does, gets a line in the log: GNOME enlarges it, so the panel is soft, and with
    PyQt6 installed and `KLIP_TPE_BACKEND` unset the window opens native and sharp.
  - Every run's log starts with the version, the git commit and the folder klip-tpe runs
    from (`klip-tpe 0.1.0 (git d2fbb74) from ...`), and `klip-tpe --version` prints the
    same line.  A copy installed with `pip install .` reads `(installed copy)`.  It keeps the
    code it was installed with when the repository is updated.
  - `klip-tpe view` picked the macOS backend on Linux and failed, since `matplotlib.use` only
    records a name before pyplot is imported.  It now tries Qt, Tk and GTK in turn.
- **The live window holds its size on GNOME's Wayland desktop too.**  On Ubuntu's GNOME
  session (port 3389 remote login included) the window could still be maximized and resized,
  and then jumped back and forth.  A native Wayland Qt window draws its own frame, and GNOME
  maximizes it on a double-click whatever size Qt has fixed.  Reproduced on mutter 46 with
  Qt 6.11's GNOME decorations: a `setFixedSize` window maximizes to 1920 × 1162.  Now:
  - on a Wayland session Qt and GTK open through XWayland (`QT_QPA_PLATFORM=xcb;wayland`,
    `GDK_BACKEND=x11,wayland`, set only when unset and before the toolkit starts), where
    mutter draws the frame and refuses both: a corner drag and a title double-click leave
    the live window at 1017 × 544.  Qt's xcb plugin needs `libxcb-cursor0`; without it Qt
    stays native and the log says how to install it.  `KLIP_TPE_WAYLAND=native` opts out;
  - `keep_window_size` reads the toolkit window's own size, un-maximizes, and restores
    through the toolkit (Tk `geometry`, Qt `showNormal`/`resize`, GTK `unmaximize`/`resize`,
    wx `SetSize`).  A window manager that resizes a Tk window leaves the figure's size
    alone, so the old check missed it, and Tk ignores a size requested by the canvas after
    such a resize.  Resized from outside with no window manager to stop it, Tk and Qt live
    windows and `klip-tpe view` return to their size within 0.6 s.  On native Wayland a
    maximized live window is un-maximized at the next GUI turn;
  - a size is adopted only while the window settles (2 s after it opens, or after a dpi
    change from moving to a screen with another scale).  Later sizes are always put back,
    at most every 0.5 s and every 5 s after ten requests that did not take.  Adopting any
    size held for 2 s let a user's resize or a maximize stick;
  - the run log names the backend and the windowing system: `live window: QtAgg on x11,
    1017 x 544 px, fixed size`.
  Checked under metacity, xfwm4 and mutter (X11, and Wayland with Xwayland), with Tk and Qt,
  at 96 and 192 dpi.
- **The paper's trace figure (f2) shows only the evaluations at each annulus' final injected
  contrast** (`paper_runs/figs.py`).  The β Pic and NIRCam panels began with the ten
  evaluations made before the warm-up check rescaled the contrast.  On NIRCam those ran at
  1.6× the final contrast, and one of them set the panel's running maximum (9.18, against
  8.95 at the final contrast and a validated 8.92).
- `docs/BACKENDS.md` gives the MASK335R inner working angle as the JWST documentation does,
  0.64″ (it said 0.63″).
- **Notebooks 3, 5 and 6 are stored with their outputs**, from runs on a Mac with the MAST
  data and the STPSF cache.  The searches took 11.1 and 13.4 minutes (tutorial 3), 116.5
  minutes (tutorial 5) and 18.8 minutes (tutorial 6).  The text now quotes what the runs
  give.  Tutorial 3: the planet's S/N in each `mode` at the default configuration (ADI 14.1,
  RDI 12.3, ADI+RDI 12.8), the two re-calibrations that make its log 320 evaluations long,
  both validated winners, and why the PSF-filter winner gives the planet 13.3, less than
  default ADI (only the injections are optimized).  Tutorial 5: the calibration contrast of
  9.95 × 10⁻⁵, and the k-scan's edge hit at the cap of 20, with how to raise it.  Tutorial 6:
  its run time against tutorial 5's.  Home-directory paths in the outputs are written as `~`.
- Tutorials 1 to 6 read more plainly: colons that joined two sentences are two sentences, and
  in tutorial 6 the inner dust belt and planets d and e are placed inside the annulus's inner
  edge, and e at 1.2 λ/D closer in than a radial stamp library can be trusted.
  `docs/TUTORIALS.md` gives the measured run times and says that building a notebook without
  `--execute` replaces its stored outputs.
- **The live window keeps one size, `window_scale` × 1850 × 990 px, for the whole run.**
  `klip_tpe.winsize.fix_window_size` switches the toolkit's resizing off: Tk
  `resizable(False, False)`, Qt `setFixedSize`, GTK `set_resizable(False)`, wx min = max
  size.  matplotlib's macOS backend has no such switch, so there the resize control is
  removed through PyObjC when it is installed (`pip install pyobjc-framework-Cocoa`).
  Without it, `keep_window_size`, called on every GUI turn, puts a resized window back.  A
  size the window manager will not give up for 2 s (a screen smaller than the window) is
  kept instead of being fought over.  The run log says which applies: `live window: 1017 x
  544 px, fixed size`.  Checked on QtAgg under Xvfb: a resize to 500 × 300 leaves the window
  at 1017 × 544, and with the switch lifted a resize to 600 × 400 is put back.
- The panel is drawn with `aspect="equal"`, so a window held smaller than asked shows it
  whole with black margins instead of stretched.
- `klip-tpe view` does the same, and the Jupyter inline panel is given its width and
  height and shown unconfined, so the notebook's width no longer rescales it and an update
  no longer collapses the cell while the next frame loads.

## Unreleased — 2026-10-08 (the dead-dimension check warns and reads the frame tags; display cadence; tutorials and docs reviewed)

- **The command line's dead-dimension check warns instead of stopping** (`--strict-liveness`
  stops).  Added earlier today, it refused the tutorials' own terminal commands: on the NaCo
  β Pic cube it called `corr_thresh` dead, and on HD 95086 `coronoise_max_K2`.  Both act, but
  only in a sliver of their range.  The frame correlations of the β Pic cube run from 0.980
  to 0.997, so `corr_thresh` acts on 2% of its [0, 1] range and the 41-value grid stepped
  over it.  A frame-selection threshold is now moved first to the cut points of the frames'
  own tags (midpoints between consecutive tag values, nearest its value first), which change
  the selection one frame at a time.  Both commands now pass the check, in 0.9 s and 17 s (it
  took 158 s to refuse the second).  `liveness_check` takes `"warn"`, `"strict"` (or `True`)
  and `"off"` (or `False`).  With `--backend vip` it reports `anglemax` dead, which is right:
  VIP has no such parameter.
- **The panel's PDF and progress movie are due every `pdf_every` / `movie_every`
  evaluations**, whatever the panel cadence, and a due PDF is never skipped.  Both used to be
  decided after the `every` gate, and against a `last` that the loop time had overwritten, so
  every drawn panel wrote a PDF and asked for a movie, and with `every=2` neither came round
  at the evaluations named.
- **No broken `opt_steps.mp4` without imageio-ffmpeg.**  imageio handed `.mp4` to its TIFF
  writer, which left an 8-byte file behind; the writer is now asked for by name.
- **Tutorials 1–6 reviewed for accuracy and rewritten for clarity.**  The code changes:
  - tutorials 3, 5 and 6 skipped their PSF-model cells whenever STPSF could not be imported,
    even with its cache present, and tutorial 3 then injected a Gaussian with a flux unit of
    1, which no contrast can calibrate.  They now read the cache, and stop with the name of
    the missing file when neither STPSF nor the cache is there.  A run made in the fallback
    state has to be deleted, not resumed;
  - tutorial 2's terminal command gave the star fluxes ×1347 (`2.91e9 2.66e9`; `2.16e6
    2.47e6` is right), and tutorials 1 and 2 now give `--n-remeasure 3 --k-max 30 --seed`
    and say which settings have no flag;
  - tutorial 5 loads both rolls into one partition (with one per roll, ADI+RDI was RDI),
    keeps the resolved disk HD 141569A out of the RDI library (`ref_targets="HIP-68245"`, as
    the paper run does), and leaves out `angsep` / `anglemax`, as `scripts/run_miri.py` does;
  - tutorials 5 and 6 average three draws per trial (the 3 had been set on the calibration
    instead), and their fetch commands are the ones `scripts/fetch_jwst_ar.py` accepts;
  - HIP 65426 b is at the 0.820″, PA 149.9° of Carter et al. (2023, Table 3).

  Notebooks 1, 2 and 4 are re-executed with these changes (tutorial 2 took 27 minutes on two
  cores and keeps K1 alone, as all six validated candidates do).  Notebooks 3, 5 and 6 are
  stored without outputs until they are run on a machine with the MAST data and the STPSF
  model.
- **Docs corrected**: `docs/BACKENDS.md` (the spaceKLIP database key is `NRCALONG`; a
  unit-flux Gaussian is not "fine for ranking"; the STPSF stamps are not rotated; it is this
  package's pyKLIP backend that has no analytic KLIP-FM), `docs/FLUX_CALIBRATION.md` (a wrong
  star flux cancels only within the calibration's reach; β Pic's unset flux unit is off by
  7.65 × 10⁵; HIP 65426 b is ΔF444W = 8.796 ± 0.092), `datasets.PHOTOMETRY` and
  `scripts/check_betapic_contrast.py` (ΔL′ = 7.81 ± 0.08, 1.1σ), `docs/RUNNING.md`,
  `docs/BUDGET.md`, `docs/TUTORIALS.md` and the README's tutorial table.

## Unreleased — 2026-10-08 (runs: one clean reduction per trial, a linear validation checkpoint, the CLI's TPE default, a dead-dimension pre-flight)

All of these were found by the NEAR2 / IDL session while it set up the Python NEAR2 production
run (`--n-remeasure 3`, validation 25 x 50).  None changes a score.

- **An averaged trial reduces its clean image once** (`Runner.evaluate_mean`).  Each of a
  trial's `n_remeasure` draws used to run a complete `evaluate`, injected and clean image
  both, but the clean image depends on the configuration alone: 2n reductions where n + 1 do,
  6 where 4 do at n = 3.  Every draw's positions are now drawn first, in order, by the same
  generator calls, so positions, scores and the RNG stream are the per-draw loop's (a run
  started on the old code resumes on the new one).  On a parallel reducer the injected
  reductions after the first run two at a time, the pair the thread budget was already sized
  for, so the saving shows in wall time: tutorial 1's search, headless on one core, took 2.48
  min instead of 3.34 with an identical history, and a trial on its reducer 0.47 s instead
  of 0.65 on one core, 0.38 s instead of 0.56 on two.  The k-scan modes keep the
  per-draw loop, since each draw picks its own k there.  A trial's `wall_s` is now its elapsed
  time rather than the sum of its draws'.
- **The validation trial checkpoint is linear in `n_valid`.**  `val_candNN_trials.pkl` was
  rewritten whole after every trial (the clean image and every trial image so far, ~1,300
  image sets per candidate at 50 trials) and `checkpoint.json` with it (the whole search
  history, 14 MB at 10 000 evaluations, once per trial).  It is now a header plus one appended
  record per trial carrying the RNG state after its draw, which is what the per-trial
  `checkpoint.json` was for; a resume restores it.  Files written by the old code still
  resume, a record cut short by an interruption is dropped and cut off, and a failed write
  rewrites the stream whole at the next trial.  `checkpoint.json` is written once when
  validation starts.
- **`klip-tpe near` / `generic` default to `--blocks univariate`**, as `RunConfig.blocks` and
  `TPE` have since 2026-09-06 (IDL addendum 2: block-multivariate lost on real data and was the
  least stable of the three).  Until now a command line without `--blocks` silently ran
  `partitions`.  Both NEAR scripts pass the flag, so production runs are unaffected; the README's
  and the tutorials' terminal examples do not pass it.
- **`check_live_dimensions` no longer calls a narrow-band threshold dead.**  It moved each
  dimension only to its bounds and its midpoint, and on the first real run against the
  56-dimensional NEAR2 space that called `noise_max` dead on nights 3–5 and
  `coronoise_max` on night 4.  Each acts only in a band a few tenths wide: above it the cut is
  above every frame's tag ratio and removes nothing, below it the night drops under its
  minimum frame count and the guard snaps the cut back.  None of the three probe values
  lands in the band.  Whatever that first pass
  leaves unmoved is now re-probed on a 41-value grid (`dense=`), nearest its own value first,
  at the same points before it is called dead, and the verdict says what was probed.  A
  healthy dimension still costs one reduction.
- **`klip-tpe near` / `generic` check every searched dimension before searching**
  (`RunConfig.liveness_check`: a warning on the command line, `--strict-liveness` to stop the
  run instead, `--no-liveness-check` to skip; off in the Python API).  A new run first moves
  each reduction dimension alone and reduces annulus 1's clean image
  (`klip_tpe.liveness.check_live_dimensions`, about one reduction per dimension); a dimension
  that never changes it is reported before the first evaluation, one the guards never let
  move alone is logged.  The search RNG is put back, so a seeded run
  is the same run with or without the check, and a resume is not re-checked.  The JWST
  drivers have done this since 2026-09-23; the NEAR launcher, whose 56-dimensional space is
  where the IDL `parstr` omission left 18 dimensions dead, never did.
- **docs/IDL_FINDINGS.md corrected**: the §1 snippet was the first version of the
  frame-selection fix, which references serial-only variables and fails on the parallel
  path (now the applied version); the §3 bisection call needs `fast=0`, production's
  per-target path (`reduce_near_2` defaults to the single-basis `fast=1`); `dthmax` is
  forwarded since 2026-10-07; run_20260906_173000 continued to 4881 evaluations after the
  4469 that §6.1 compared, and was retired on 2026-10-07; the TPE default noted in §4 was stale.

## Unreleased — 2026-10-08 (tutorials: one budget, and injections clear of the companions)

- **Every tutorial search uses tutorial 1's budget**: 300 evaluations, the first 40 of them
  warm-up, the six best candidates validated on eight fresh injection sets each, and three
  draws averaged per trial (tutorial 3 keeps one draw, to hold its two pyKLIP searches to
  about two hours each).  Tutorials 2, 3, 5 and 6 had demo
  budgets of 50–80 evaluations with 15–20 warm-up, which tutorial 1 and docs/BUDGET.md
  explain give a meaningless answer rather than a rough one.  docs/TUTORIALS.md has the new
  run times.
- **Injections are kept clear of the companion where its position is given**
  (`sampler.excl_fwhm`): 4 FWHM for β Pic b (tutorial 1) and for HIP 65426 b on NIRCam
  (tutorial 3), 6 for HD 95086 b (tutorial 2), 3 for HIP 65426 b on MIRI (tutorial 5).  The
  default 1.5 FWHM (the IDL's) is enough for the search score, which subtracts the S/N the
  same reduction gives at each injection site without the injection, but it let about one
  injected source in twenty land within 3 FWHM of HD 95086 b, and beside it in every image
  of the panel.  Tutorial 4 shows what the
  radius does; tutorial 6 still leaves its planets' positions to the reader, and now says
  what that costs.  The default is unchanged.
- **Tutorials 1, 2 and 4 re-executed** with this release's display.  Tutorial 1's text now
  quotes its own run (injections from S/N 4.2 at the default configuration to 6.5 validated,
  β Pic b from 15.4 to 20.9).  Tutorials 3, 5 and 6 need the MAST data and STPSF and are
  executed separately.
- **The MIRI tutorials read `<data>/mastDownload` when it exists**: an archive download beside
  a reprocessing of the same exposures (`reproc/`) made the recursive search find each exposure
  twice, which `load_calints` refuses.

## Unreleased — 2026-10-07 (live display: validation panels gated, one rc context at a time; `klip-tpe render`)

- **Validation panels wait their turn.**  `LiveDisplay.on_validation_trial` now passes the
  render gate the search panels already used.  Every validation trial (`n_top x n_valid`, 48
  in the README's first run) was queued unconditionally, so where a panel draws slower than a
  trial runs, the window fell minutes behind and the annulus then waited in `_wait_renders`
  for the backlog: 3.7 min on that run on two cores, long enough for the heartbeat to report a
  stall in `param_verify` that was really the display.  A trial is skipped while the render
  thread is busy; the annulus-done frame is drawn regardless.
- **One rc context at a time.**  Every rc context the package opens (`display._rc`,
  `plots._style`) now holds one reentrant lock (`plots._RC_LOCK`) for its whole extent.
  rcParams is a single dict per process, and the panel (dark, on the render thread) and the
  annulus books (light, on the main thread) used to be open together: 4 of that run's 122
  panels came out with white axes and black-on-black text, every one drawn while
  `corner.pdf` ... `products.pdf` were being written, and the wrong order of exits could
  leave the dark theme behind for figures made afterwards.  Pinning the light colours in
  `_RC` had fixed only the books' side of the race.
- **`klip-tpe render --run-dir <dir> [--every N] [--annulus K ...] [--out DIR]`** draws every
  evaluation's step panel after the fact, from the log and the saved per-eval crops, into
  `<dir>/steps_rebuilt/` with its own `opt_steps.gif` -- the live `steps/` (which also holds
  the calibration and validation frames) and the live movie are left as they are.  The live
  display's skip notice, tutorial 1 and docs/BUDGET.md have all pointed at this command, which
  did not exist; `display.render_steps` was there, Python-only.  `render_steps(movie_dir=)`
  puts its movie beside the panels.  docs/CLI.md regenerated, which also adds the
  `--n-remeasure` the README's terminal example already uses.

## Unreleased — 2026-09-30 (MIRI frames: hot pixels, provenance, one exposure once; replayable draws)

- **Static hot pixels the DQ array misses** (`load_calints(hot_pixels=None)`, on whenever
  there are background pointings; `static_hot_pixels`).  On the blank-sky median of the
  dedicated background pointings, a pixel above (or below) both the median of its seven row
  neighbours and that of its seven column neighbours by more than `max(10 robust sigma, 5)`
  MJy/sr is flagged in every science and reference frame and repaired like a DQ pixel.  The
  AND keeps the 4QPM glow sticks off the list.  ERS 1386 F1140C has one five pixels (0.6")
  from the star, 52 MJy/sr above its neighbours on the blank sky, that the pipeline's
  background subtraction leaves at +12-15 MJy/sr in the science frames and the loader's at
  +25-30 in the references: a fixed detector feature no ADI+RDI reduction can fit, which sat
  at 11:30 o'clock in every reduction.  75 pixels flagged on the archive calints.
- **One exposure, once.**  `load_calints` refuses a file list in which a product name occurs
  twice.  A tree holding the archive calints and a re-reduction of them from `uncal`
  (`reproc/stage2/` beside `mastDownload/`) was found whole by the drivers' recursive search
  and stacked: every F1140C frame twice, from two pipelines.  `paper_runs/miri_fig.py` and
  `companion_tests.py` now default to `~/Data/JWST/hip65426_miri/mastDownload` when it exists.
- **The run records its frames.**  `info['provenance']` (in every Dataset's `meta`, and through
  `KLIPReducer.describe()` in `run_setup.json` as `reducer.partitions.<id>.frames`): the
  files, `CAL_VER` and `CRDS_CTX` from their headers, the pixel scale as read (0.1103"/px in
  jwst 2.0.1 MIRI products, 0.1100 in jwst 1.13.4), the repaired pixels, what was
  background-subtracted, and how many frames registration left unshifted.
  `frames_provenance(reducer)` / `frames_signature(provenance)` name a frame treatment;
  `miri_fig.miri_negfc` caches the companion fit per treatment (the original file keeps the
  frames it was made on).
- **Rebuilding a finished run** (`library_ablation._args_for` / `_rebuild`, `miri_fig`,
  `companion_tests`) repairs hot pixels exactly when the run did
  (`library_ablation.hot_pixels_for_rebuild`).  A run from before the repair stays unrepaired
  on its own frames, recognised by the pixel scale it recorded, and gets the loader's default
  on other frames; with only a run's `config` to go on it is rebuilt unrepaired.
  `--hot-pixels on|off` (`run_miri.py`, `library_ablation.py`) and `hot_pixels=` override.
- **Every draw recorded, and recorded draws replayed.**  A trial averaged over several draws
  now keeps every draw's sources and raw score (`meta['draw_sources']`,
  `meta['draw_raw_scores']`), and validation rows keep their trials' sources
  (`trial_sources`).  `positions.ReplaySampler` / `Runner.replay_draws(recorded)` inject
  recorded positions instead of drawing, and `library_ablation.py --draws-from JSON` scores a
  run on another ablation's recorded injections.  A sampler's draws depend on the frames (their
  rolls and dead zones set its forbidden sectors), so the same seed does not reproduce them on
  another processing of the same data.
- **pyKLIP on macOS no longer spends most of a search starting processes.**  pyKLIP builds a
  worker pool for every reduction, and macOS's `spawn` start method makes each worker a fresh
  interpreter that imports numpy, scipy and pyklip: MIRI's v7 pyKLIP search took a median 24 s
  per three-draw evaluation on the Mac against 5 s for the built-in engine.  On macOS the
  backend now gives pyKLIP a `forkserver` context (`KLIP_TPE_PYKLIP_START` overrides).
  Measured with spawn forced: 18.0 / 12.6 s per evaluation against 8.7 / 4.0 s with the
  forkserver, scores identical.  The built-in engine spends 85% of a MIRI evaluation in the
  per-target KL basis (BLAS), 2-15 s per three-draw evaluation; a shortcut computing only the
  first k modes changes the last bits of the basis, so it is not taken.
- **Carter et al. (2023) as published** (`library_ablation.py`, configuration
  `carter_published`, `CARTER_PUBLISHED`): their Figure 3 reduction from their configuration
  file -- the first integration of every exposure dropped (80 + 81 frames), pyKLIP ADI+RDI over
  the whole illuminated field (211x211 px, one zone), every mode (121), mean-combined -- scored
  on the search's central 81x81 px.  `CARTER` (the k = 6 mapping) keeps its meaning for the
  scripts that import it.  `paper_runs/miri_fig.py` shows it in f13's first panel ("published
  configuration") and scores it for Table 3, over one zone and over the inner annulus.
- **`companion_tests.py miri` also runs with the companion taken out** (`<engine>_<contrast>_removed`):
  the same draws at HIP 65426 b's separation, on frames with b subtracted at its
  negative-injection fit.  Left in, b's own light raises the ring scatter at its separation in
  some configurations (1.8x in v8's pyKLIP winner), which the injections beside it feel and its
  forward-model S/N does not.
- **`paper_runs/miri_v8.sh`**: the paper's MIRI runs on the archive calints with the hot pixels
  repaired -- both searches (v7's spaces, budgets, annuli and seed), both library ablations, the
  companion tests and f13.

## Unreleased — 2026-09-27 (known companions, and the built-in engine's library cache)

- **The built-in engine's searched library handed one reduction another's frames.**
  `KLIPReducer._reflib` cached the basis rows (`[science | references]`) with the ranking,
  keyed on the binned frame count, filter and annulus only.  Every later reduction with the
  same key took its other-roll frames from whichever reduction filled the cache first -- a
  different evaluation's injections, or none -- so an injected source was not self-subtracted
  by the other roll, a clean reduction carried someone else's sources, and a score depended on
  the order of the reductions.  The cache now holds only the selection (groups and the
  similarity ranking, computed on the frames without injections), keyed also on the binning,
  the rotation threshold and the frame-selection mask; the basis is always the reduction's own
  cube.  `tests/test_searched_library.py` has a test that fails on the old code.  Affected:
  every built-in-engine run with a searched library and an other-roll pool (NIRCam DK, D2K,
  H2K; MIRI `v7_klip`; any MWC 758 pilot on the built-in engine).  pyKLIP runs, which build their own basis, and the
  NACO / SPHERE runs are not.  Rerun: MIRI's built-in winner now uses the reference star alone
  inside 2.2″, and the 2026-09-25 head-to-head below is superseded (built-in / pyKLIP on common
  draws: ×0.91, ×1.08, ×0.91 by annulus; HIP 65426 b 9.9 against 10.1).
- **Known companions can be taken out of the frames** (`RunConfig.subtract_known`: a negative
  injection of the template at a fitted position and contrast, in every reduction).  HIP 65426
  b's light, spread over its KLIP sector, set the NIRCam ring scatter at 7-10× its
  companion-free value across the inner annulus, well beyond the 1.5-FWHM exclusion, and the
  searches optimized its suppression.  `paper_runs` stages D2 / D2K and H3 / H3K subtract it
  (`run_demos.hip65426b_negfc`); H3's injection contrast was recalibrated with it gone
  (1.497e-5, `calibrate_bench_contrast.py H3`).
- **`klip_tpe.companion`**: the negative-fake-companion fit (`fit_negative_companion`) and a
  forward-model companion S/N (`companion_snr`): the matched-filter peak of (reduction −
  reduction with the companion subtracted) over the ring scatter of the subtracted reduction,
  so the companion's own light is not in its noise.  `collect.py` reports it for default and
  winner (`companion_fm_default` / `_optimized`) and measures the companion's photometry with
  it; `miri_fig.py` titles its panels with it.
- **Paper numbers from repo code**: `companion_tests.py miri_light` (the F1140C version of the
  NIRCam companion-light test), `scripts/ablation_summary.py` (medians and win counts from
  `library_ablation.py`, which logs means), `scripts/bench_numbers.py` (Table 4: validated
  means, paired t, inflation, speed factors, raw-scale inflation).

## Unreleased — 2026-09-25 (the MIRI head-to-head, and collecting what the paper reads)

- **MIRI v7, head to head** (40 paired draws per annulus, both engines injected at pyKLIP's
  calibrated contrast): the built-in engine's winner beats pyKLIP's by ×1.33 [1.29, 1.37] in
  annulus 1 (0.74–2.21″; better on 39 of 40 draws), and pyKLIP's is ahead by ×1.06 in annulus
  2 (2.21–2.96″) and ×1.07 in annulus 3 (2.96–3.97″).  HIP 65426 b: S/N 13.4 against 7.8.
  Only this direction was run; on NIRCam one annulus' verdict changed with the contrast.
  Results beside each run in `miri_HIP-65426_F1140C_v7_{pyklip,klip}/library_ablation.json`.
- **What the searched library buys there.**  pyKLIP: ×1.5 over taking every frame in annuli
  2–3 and nothing in annulus 1 (×0.99), but its own unsearched choice — keep the k best —
  comes within 2–4 % of the searched `maxnumbasis` everywhere.  Built-in engine: ×1.25–1.41
  over everything in every annulus, and the other roll alone matches its winner in annulus 1
  (×1.00).  Carter et al. (2023)'s MIRI choices, mapped into this space and scored with this
  metric (not their pipeline or their S/N), come out ×2.0–2.7 below either winner on the
  injections, and b at 5.4–5.5 against 7.8 and 13.4.
- **`python3 collect.py` skipped the runs the figures read.**  Its default was `A C D B`: A and
  B are the runs from before 2026-09-14, whose directories are gone, so it refreshed C and D,
  printed two tracebacks and left A2 and B2 as they were.  It now collects everything the
  paper reads (`PAPER`: A2 B2 C D DK E F G H HK), and a target whose run is missing or
  unfinished is one line, not a traceback.  `collect.py D` and `DK` were run end to end.
- **The ablation view, from the v7 pictures.**  Nine configurations ran `rdi_third` into
  `ardi_half`: names now break at their `_`, with the planet level on the third line.  A long
  run name ran the title into the legend: the title shrinks to fit.  The reference line's
  label sits inside the frame on a backing that reads over the last column.  `--plot` of an
  output from before the pixel scale was recorded takes it from the instrument, so the zone
  is shown rather than the injection band; a finished run says how long it took, not "so far".
- **The bench figure drew E2's retired grid arm.**  `figs.py` found the convergence curves by
  globbing `bench_*_*_s*`, which also matches the slots `supersede_bench_mode.py` renamed
  `..._sN_superseded_<date>`: the E2 panel said "grid (16 seeds)" — eight retired, eight
  current — beside bars of the eight.  The curves now come from the slots the summary counts
  (`bench.summary_run_dirs`, which `python -m klip_tpe.bench` uses too); only E2 had retired
  slots, and its bars and every collected number are unchanged.  Rerun `figs.py`.
- **E2's grid arm ran 800 evaluations; tpe and random ran 1000.**  The arm was retired and
  re-run on 2026-09-20 with the command `supersede_bench_mode.py` printed —
  `BENCH_MODES=grid python3 paper_runs/run_demos.py E2` — which leaves out the
  `BENCH_NITER=1000` that `long_run.sh` had run the batch under, so the stage fell back to
  its own 800.  The convergence panel shows it (the grid curve stops at 800), and the bars
  compare grid's validated winners at 800 with the others' at 1000.  The tool now prints the
  budget of the arms the re-run will be compared with (and knows H2's per-engine
  directories), and `run_benchmark` refuses to run a slot into a batch whose rows are at
  another `n_iter`/`n_init` — a resume with nothing left to run still only reads.  Re-run at
  1000 on 2026-09-25: grid validates at 6.62 ± 0.29 (6.56 at 800), against random 6.91 ± 0.35
  and TPE 7.91 ± 0.30; paired by seed, TPE leads grid by +1.28 ± 0.17 and random by +1.00 ±
  0.07, 8 of 8 seeds each.  All 24 E2 rows are now at one budget.
- **`supersede_bench_mode.py` re-stamped slots it had already retired.**  It took every
  directory starting `<tag>_<mode>_s`, which includes `..._s0_superseded_20260920_150534`:
  retiring E2's 800-evaluation grid arm renamed the arm retired on 09-20 a second time
  (`..._superseded_20260920_150534_superseded_20260925_095843`).  It now takes the live slots
  only (`<tag>_<mode>_sN`); the eight double-stamped directories were renamed back.
- **Figure legibility, from the regenerated set.**  f7: with H2 on top its bars filled the
  panel and the key sat on them — the panels now leave headroom for the key and the gap
  labels.  f6: the key moved under the row, off β Pic's inner curves.  f12: the seeded
  default printed as k=9.0, b=7.0, f=12.0 beside the winner's k=17, b=1, f=8, and the
  injected sources' S/N labels, green, vanished on the map's dark blue — integers now print
  as integers, and the labels have a white edge.
- **The synced-folder fix held.**  After it and the cleanup, the last eight hours of H2 and
  both MIRI ablations left 6 conflicted copies, in one slot within 12 s; before it, dozens to
  hundreds a day.

## Unreleased — 2026-09-24 (the v7 ablation, and two source-count regressions)

- **`library_ablation.py` crashed on the first pyKLIP-native run.**  Its rebuild check compared
  every winner parameter as a float, and v7 is the first run whose space holds a categorical
  (`mode = "ADI+RDI"`): `ValueError: could not convert string to float`.  Categoricals now
  compare by equality.  The whole v7 chain — pyKLIP ablation, klip ablation with
  `--contrast-from`, `--versus` — was then run end to end on the real runs before handing it
  back.
- **"Take everything" was out of reach on pyKLIP at any bin but 1.**  `PyKLIPLibraryGuard`
  clipped `maxnumbasis` to the pool (95 at bin 9), and the sanitize after it snapped that to
  the grid neighbour below (90).  pyKLIP keeps `min(maxnumbasis, pool)` frames, so the guard
  now clips to the first grid point at or above the pool (100): every search state and the
  ablation's "all" variant can use every frame.  Interior values — every v7 winner — are
  unchanged, so the finished runs rebuild exactly.
- **RX J0534 injected 2 sources, not 3.**  On 2026-09-22 (471c017) the NIRCam limit of two —
  a small coronagraphic field, where simultaneous sources perturb each other's basis — was
  also applied to `run_rxj0534.py`'s `--n-sources` default, with NIRCam's reason in its help.
  RX J0534 is LMIRCam, and three is its documented choice (three plus the planet leave the
  noise ring ~13 apertures).  Restored.  No run was made with the wrong default: every
  directory under `~/klip_tpe_runs/rxj0534` predates it.
- **A 0 in a per-annulus `n_sources` injected nothing.**  The documented meaning is "this
  annulus uses the rule", but only `None` was converted; `[0, 3]` asked annulus 1 for zero
  sources and the ring-survival loop had nothing to try.  0 now means the rule, in a list
  and as a single value.
- These were the three slow-suite failures noted on 2026-09-23 (they failed identically
  before that day's changes).
- **`library_ablation.py` does NIRCam too** (`--instrument`, read from the run's pixel scale
  by default).  A NIRCam run is rebuilt by `run_demos`'s own `run_D` recipe, needs no
  `--data`, and leaves out the Carter et al. configurations, which encode their MIRI
  reduction and the 4QPM's inner working angle.  The ablation now also injects the run's own
  `n_sources` — it had always used the rule, which matched MIRI v7 (recorded `None`) by luck
  and would not have matched D's 2.
- **D vs DK, head to head** (40 paired draws per annulus, both engines injected at one
  contrast, done both ways): pyKLIP's winner beats the built-in engine's in annulus 1
  (injections at 0.92″) by ×1.12–1.13 at either engine's contrast, and in annulus 2 (2.18″)
  ties at pyKLIP's fainter contrast (×0.98 [0.94, 1.02]) and wins ×1.13 at klip's.  HIP 65426 b
  in annulus 1: S/N 16.7 vs 14.0.  pyKLIP's winning library is pure ADI with its two
  other-roll frames (its top-2-correlated ADI+RDI reduces identically); the built-in
  engine's own search left a better library on the table in annulus 1 (its best third of the
  reference star beats its winner).  Results in `paper_runs/D_hip65426_{pyklip,klip}/`.
- **The ablation has a live view.**  It called the reducer directly, so none of the search's
  display callbacks fired and it ran blind.  `AblationDisplay`: per annulus, each
  configuration's injected-source S/N draw by draw with its running mean ± standard error,
  coloured by role (the run's winner, library variants, baselines — validated dark-surface
  palette), the planet's S/N under each name rather than on the injected-S/N axis (14–17
  against 4–8), the other run's winner as a dashed reference with `--versus` when the
  injections match, the latest clean and injected reductions in the search display's own
  stretch, and the paired result at the end.  On by default (`--no-show`), PNG beside
  `--out` either way, written at most every 30 s (Dropbox); a display error turns it off
  and never stops the ablation.  `versus()` now also returns its numbers, and `--plot JSON`
  draws the same view for an ablation that has already finished (its S/N panels; an
  ablation's reductions are not stored).

## Unreleased — 2026-09-23 (runs that survive a synced folder)

`paper_runs/` and the MIRI run directories live in Dropbox, and a fast search outran it.
Twenty minutes into H2K (~1 s per evaluation) its slot held 115 conflicted copies of
`heartbeat.json` and 49 of `checkpoint.json` (MIRI klip at ~3 s: 53; MIRI pyKLIP at ~30 s:
none), and Dropbox had put its own older, online-only version back under the name
`checkpoint.json` while the runner's newest save (800 evaluations, annulus done,
param_verify recorded) sat in `checkpoint (… conflicted copy … 48).json`.  Every
`results.jsonl` was intact; the exposure was a resume, which trusted the name.

- **Resume takes the checkpoint furthest along.**  `klip_tpe.runner.load_checkpoint` reads
  `checkpoint.json` and every sync-client copy beside it (`checkpoint (… conflicted copy
  …).json`, `checkpoint 2.json`, `checkpoint (1).json`, `checkpoint.sync-conflict-…`) and
  takes the one with the most evaluations (then annuli finished, hooks run, wall time; ties to
  the real name), saying so when it is a copy.  `Runner.resume`, `Runner.extend`, every
  auto-resume, and `bench`'s "resumable" all use it.  A test replays the H2K swap end to end:
  with the old loader it resumes from the stale save and logs evaluations twice.
- **Routine checkpoints are written at most every 10 s** (`Runner.CHECKPOINT_EVERY_S`).  The
  save after each evaluation is still taken every time — serialised on the spot, so it is
  exactly the state between two evaluations — but held when the last write was under 10 s
  ago, and flushed on any exit: Ctrl-C and exceptions included, so a resume is as exact as
  before (tested against an uninterrupted run: same evaluations, same scores).  Only a hard
  kill loses the held save, and then a resume replays at most 10 s from the same RNG state,
  whose re-logged records every reader drops (`figs.py` now reads through
  `bench.read_records` like the rest).  Phase saves stay immediate: annulus ends, hooks,
  segment starts, and validation candidates and trials, whose pickles pair with the RNG state.
- **The heartbeat's per-step stages wait for the next stamp.**  `Heartbeat.stage(...,
  write=False)` for the per-evaluation and per-validation-trial stages: live at once for the
  stall check, on disk with the thread's next 5 s stamp.  Phase changes still write at once.
- **`RUNS_DIR` moves the paper runs out of the synced folder.**  `run_demos.OUT` (and with it
  `collect.py`'s summary and `figs.py`'s figures), `rerun_paper.sh`'s finished / live /
  retire checks and `long_run.sh`'s progress ticker all follow it; `~` is expanded, I2 keeps
  `RXJ_OUT`.  Default unchanged.  `run_miri.py --out` already takes any path.
- Safe to land under running searches: they keep the modules they started with.  The one
  late import, `bench` from `plots.load_run`, falls back to the old checkpoint test when the
  process's `runner` predates `checkpoint_candidates`.

## Unreleased — 2026-09-23 (a head-to-head needs one contrast, and the run's own engine)

- **`library_ablation.py --versus` paired two runs' injections by position only.**  Each run
  calibrates its own contrast per annulus, the ablation injects at its run's, and S/N scales
  with the injected flux — so a pyKLIP-run-versus-klip-run ratio would have carried the two
  calibrations inside it.  Measured on v6's annulus 3, built-in engine, same winner, same
  positions: S/N 3.63 at the run's contrast and 6.33 at twice it, which `versus` would have
  reported as a ×0.57 head-to-head.  Now `--contrast-from RUN_DIR` injects at another run's
  contrasts (same annuli required), and `versus` refuses to pair unless positions AND
  contrasts are identical; it also pairs by annulus number and refuses, rather than
  crashes, on a different draw count.  Repeating a configuration at matched contrast gives
  ×1.000 exactly.
- **The engine defaults to the run's own.**  `--backend` defaulted to pyKLIP, and winners
  are carried across by name — so a klip run ablated without `--backend klip` would have
  dropped its `nkeep_*` silently and reduced "the winner" at pyKLIP's library defaults,
  while a rebuild check that compared only the names both spaces had passed it.  On the
  run's own engine the searched dimensions must now match exactly (v6 on pyKLIP is refused:
  its dead counts have no counterpart); naming the other engine is a cross-engine test and
  is noted as one.
- The `make_reducer` docstring no longer says pyKLIP cannot search a library.
- **Run on an unfinished run, the ablation now says so.**  Started a minute after the v7
  searches, it raised a bare `FileNotFoundError` on `run_setup.json`.  It now checks
  `--run-dir` and `--contrast-from` for a finished run (`run_setup.json` appears once the
  search starts, `final_results.json` when it ends) and points at the run's `run.log`, and
  it checks that the `--versus` file exists before the first reduction rather than failing
  at the end of hours of them.

## Unreleased — 2026-09-23 (both libraries, and a check that every searched dimension is live)

- **pyKLIP's own library, searched.**  `PyKLIPReducer.set_native_library` makes `mode`
  (ADI / RDI / ADI+RDI) and `maxnumbasis` (how many of the most-correlated frames of those
  pools each target keeps, per sector) searched dimensions, contributed through
  `reference_params()` exactly as the built-in engine's counts are.  `maxnumbasis` now
  reaches pyKLIP — it never did, so it was `k_klip` — and defaults to the space's `k_klip`
  default, so the seeded configuration is what pyKLIP always did.  `PyKLIPLibraryGuard`
  keeps `k_klip <= maxnumbasis <= pool(mode, bin)`: pyKLIP clips the KL count to the frames
  it kept (`klip_math`), so outside that region one of the two changes nothing.
- **Both engines, in both drivers.**  `run_miri.py --backend pyklip` searches `mode` +
  `maxnumbasis`, `--backend klip` the two counts.  `run_demos.py`: D and H2 on pyKLIP, new
  DK and H2K on the built-in engine, each into a directory nothing older can be resumed
  from (`D_hip65426_pyklip`, `D_hip65426_klip`, `H2_bench_jwst_pyklip`,
  `H2_bench_jwst_klip`), wired through `collect.py` (D, DK, and D0 for the valid mode-only
  run of 2026-09-20; H, HK), `figs.py`, `rerun_paper.sh` and `long_run.sh`.  The old
  `H2_bench_jwst` is no longer collected: its newest slots searched the dead counts.
- **`klip_tpe.liveness.check_live_dimensions`, before every search.**  It moves each
  searched dimension alone — after the space's guards, rejecting a move they undo or one that
  drags another dimension along, so a difference is never credited to the wrong one —
  reduces, and compares bit for bit.  `run_miri.py` (also under `--check`;
  `--no-liveness-check` to skip) and D / DK / H2 / H2K refuse to start with a dimension that
  changes nothing.  On the real data every searched dimension is live on both engines: six
  for MIRI F1140C (6 s on pyKLIP, 2 s built-in), five for NIRCam D.  Run v6's configuration —
  the counts on pyKLIP — comes back dead, and a test replays exactly that.
- A dimension can be inert in one direction and live in another: pyKLIP's ADI+RDI keeping
  its 3 best frames reduces exactly like ADI when all 3 come from the other roll, while RDI
  from the same point does not.  So the check tries every candidate value before it calls a
  point dead.
- `scripts/library_ablation.py` handles both engines' libraries (pyKLIP variants: all / rdi
  / adi / top_k), categorical parameters, and `--versus`, a paired head-to-head of two runs'
  winners on identical injection draws (refused when the draws differ).  A first look at v6's
  annulus 3, two draws only: pyKLIP keeping its 20 best-correlated frames scores 6.23 against
  3.56 with all 99 — how many frames matters a great deal on pyKLIP's per-sector ranking,
  where it barely did on the built-in engine's per-annulus one.

## Unreleased — 2026-09-23 (the searched library was never applied on pyKLIP)

- **Retraction: `nkeep_altroll` / `nkeep_psfref` changed nothing on the pyKLIP backend** —
  which is every JWST run since the entry below that introduced them.
  `set_reference_library` lives on the built-in `KLIPReducer` and adds the two counts to the
  search space; `PyKLIPReducer._subtract` never used the basis they select, handing
  `klip_parallelized` the whole reference cube every time.  MIRI F1140C run v6 searched both
  for 600 evaluations.  Re-reduced on 40 common injection draws per annulus
  (`scripts/library_ablation.py`), pure ADI (41, 0), pure RDI (0, 90), everything (41, 90)
  and the elected counts give **bit-identical** scores in all three annuli, and the same
  HIP 65426 b S/N.  The "RDI inside, ARDI outside" pattern was where TPE left two dead
  dimensions, and their apparent importance in the panel is what an inert dimension picks
  up as a search converges.
- **NIRCam `run_D` / `H2` lost a live dimension to it.**  That entry replaced the `mode`
  categorical — live on pyKLIP — with the two counts, and those stages build pyKLIP with
  `mode="RDI"` fixed, so as configured they could not choose ADI / RDI / ADI+RDI at all.
- **Guard.**  A backend that overrides `_subtract` now refuses `set_reference_library`
  (`NotImplementedError`) and contributes no dimensions unless it declares
  `_honours_reference_library`.  `run_miri.py` refuses at build time, before any
  reduction, and names the two ways forward; `run_D` / `H2` stop with the same choice.
  New tests: the built-in engine's reduction must *change* with the counts — the property
  the feature rests on, which no earlier test reduced anything to check — and pyKLIP, VIP and
  `FunctionReducer` must refuse.
- **`spaceklip.make_reducer(backend="klip")`, `run_miri.py --backend klip`**: the built-in
  annular KLIP with the STPSF 2-D injection model, the one engine that applies the library.
- **pyKLIP was already ranking references.**  Per target frame and per sector, pyKLIP 2.10's
  ADI+RDI keeps the `maxnumbasis` most-correlated frames of the union of both pools — and our
  backend never set `maxnumbasis`, so it defaulted to `k_klip`.  On every pyKLIP run,
  `k_klip` has been tuning a one-pool library size as well as the KL truncation.
- **The first real test of the library** (built-in engine, v6's other parameters, 40 common
  draws per annulus): *which* pool matters a great deal and *how many* frames much less.  The
  other roll alone beats anything involving φ Cen by ×1.41 at 1.1–1.8″ (S/N 10.68 against
  7.56 for every frame), ×1.14 at 2.6″ and ×1.20 at 3.3–3.6″; keeping the best-correlated
  third of φ Cen instead of all 90 moves the injections by under 5%.  HIP 65426 b at 0.82″
  reverses it — the roll moves it only 0.37 FWHM, so ADI takes 12.4 (top-third RDI) down to
  4.9.  At the same parameters pyKLIP is much the stronger engine at 2.6–3.6″ (6.79 against
  4.52), consistent with it re-ranking references per sector where the built-in library
  ranks per annulus.
- **What v6 does show** (pyKLIP, 40 common draws per annulus): its validated winners beat the
  untuned default by ×1.37 at 1.1–1.8″, ×1.28 at 2.6″ and ×1.01 at 3.3–3.6″, and take
  HIP 65426 b from S/N 9.3 to 12.2.  Its injection-calibrated 5σ curve (2.06e-4 at 0.99″,
  5.0e-5 at 3.09″) sits on Carter et al. (2023)'s (~2e-4 at 1″, ~5e-5 beyond 3″) — at par,
  not ahead, with the caveat that the two 5σ definitions differ.
- Display: the library counts are labelled `nkalt` / `nkref` instead of both `nkeep_`, and
  no two parameters can share a label any more; the KLIP-FM cell says why it is empty on a
  backend without forward modelling instead of promising a model "(after 1st best)".
- Carter et al. (2023) F1140C, corrected in `datasets.py` and tutorial 05: the companion is
  ΔF1140C = 8.264 ± 0.021 (4.95e-4, 31.5 µJy), not the "~2e-4 / 13.8 µJy" quoted — that
  was their 5σ *limit* at 1″.  Their Table 3 then implies a star of 0.0637 Jy, 8.3% below
  ours (1.3σ), a real flux-scale check the old note missed.

## Unreleased — 2026-09-23 (a tutorial budget that can actually find something)

A student ran tutorial 1, stopped at ~20 evaluations because nothing seemed to be happening,
and concluded it was not working.  Both halves of that were the package's fault.

- **Fix: `show="inline"` outside a Jupyter kernel silently displayed nothing.**  Inline mode
  pushes the panel to an IPython display handle, which renders nothing in a terminal — no
  window, no warning, no hint that a live display exists at all.  The tutorials ship as
  `.py` files as well as notebooks, so running one as a script hit exactly this.  It now
  falls back to a window and logs why.  Tutorial 1 also asked for `"inline"` explicitly
  where `"auto"` was meant.
- **The tutorial's budget could not have worked, and now there are numbers for that.**
  Four 1200-evaluation searches plus eight 300-evaluation searches on the tutorial's own
  data, with every incumbent re-scored on fresh injections rather than trusted at its
  search value:
  - the per-injection-set scatter of the objective is **σ = 0.87 S/N** (853 repeated-draw
    trials) against a landscape only **1.44** wide, so a single-draw score barely separates
    two configurations at all;
  - the answer improves steeply to **~100–120 evaluations** and is flat after that, while
    the best score *seen* keeps climbing — selection bias reaching **+1.6 S/N at 1200**,
    which matches `σ/√n_remeasure × Φ⁻¹(1−1/N)` to 0.02;
  - `n_init=15` leaves TPE four points in nine dimensions and measurably loses ~0.4 S/N
    against 40, 100 or 200, which are indistinguishable;
  - the configuration a search ranks **first** was the truly-best of its own top eight in
    **none** of twelve runs, so `n_top` matters more than `n_valid`.
  Tutorial 1 goes to `n_iter=300, n_init=40, n_top=6, n_valid=8, n_remeasure=3`, and the
  README's quick start with it.  New `docs/BUDGET.md` carries the measurements and the
  rules of thumb (`n_init` ≈ 4–8 × dimensions, `n_iter` ≈ 30 × dimensions).
- **The quick start now points at the live display before the code**, says what each phase
  looks like on the panel and when it starts to mean something, and mentions `klip-tpe view`
  for attaching to a run in progress.
- **New `--n-remeasure` CLI flag.**  It was reachable only from the API, so every
  command-line run was single-draw.  Default stays 1; nothing existing changes.
- **Measured: drawing the live panel is not free.**  60 evaluations took 43 s headless,
  171 s at `every=5` and 290 s at `every=1` on one core.  The display already skips frames
  rather than blocking, but on a machine short of cores the panel dominates the wall clock,
  so tutorial 1 throttles to `every=2` and `docs/BUDGET.md` says to raise `every` before
  cutting `n_iter`.
- **Tutorial 1 no longer implies the winner must improve β Pic b.**  It is an independent
  check and is allowed to disagree: across fifteen runs the validated winner put β Pic b
  between 11 and 22 against ~16 for the default, because the objective medians over position
  angle while the planet sits at one (within a single winner, an injected source varies
  across PA with sd 1.5 on a mean of 9 — the same fractional scatter).  Section 5 now prints
  β Pic b for every validated candidate so the spread is visible in the reader's own run,
  and uses the objective's own metric — one built without `known=` counts β Pic b as noise
  in its own ring and reads ~3× low.

## Unreleased — 2026-09-23 (the NIRCam stages remeasure too)
- **`run_D` and `H2` now score each trial as the mean of 3 fresh draws**, as the MIRI driver
  does.  They had `n_remeasure=3` on `CalibrationConfig` — the calibration's own, older
  setting — while `RunConfig.n_remeasure` stayed at its default of 1, so every NIRCam trial
  was a single measurement and the new error bars had nothing to draw.  That looked like a
  plotting bug and was not; `draw_ranges` correctly adds nothing for a one-draw run.
- A benchmark exists to separate TPE from random, and on MIRI a single draw scatters with
  sd 0.84 against a useful range of ~6 — most of what such a benchmark measures is that
  noise.  NIRCam reductions are **2.2 s** against MIRI's 22.6, so H2's 16 slots × 800
  evaluations go from ~8 h to ~24 h.  This batch is already incomparable to E2/F2/G2 (search
  space, radprof, the reference library, source count), so nothing further is lost by also
  fixing what it measures.
- `_bench_hi` takes `n_remeasure`, defaulting to **1**, so E2/F2/G2 are untouched until they
  are re-run deliberately.  Pinned by a test, because the setting is easy to lose in a
  refactor and its absence is invisible: the run completes, the panels just have no bars.

## Unreleased — 2026-09-23 (the draw bars went into a panel nothing draws)
- **Fix: the error bars were added to `panel_trace`, which only `render_step_classic` calls.**
  The live window and the step PDFs go through `render_step` → `panel_convergence_idl`, so
  neither showed anything — the feature was invisible in exactly the two places it was asked
  for.  `panel_convergence_idl` now draws them too, in each point's own series colour
  (grey for warm-up/explore, the TPE and local colours otherwise) with the flowing legend
  gaining a `bars: min-max of N` word.
- That panel sets its own y-range from the means, and the draws reach past them by
  construction, so the range is widened to contain them: a clipped error bar is worse than
  none.
- Three regression tests, which fail on the old behaviour: both panels draw a bar per
  remeasured trial, the convergence panel's limits contain the extreme draws, and a
  single-draw run still adds nothing.

## Unreleased — 2026-09-23 (nkalt / nkref replace the ADI/RDI mode, on NIRCam and MIRI)
- **The searched reference library is wired into both JWST paths.**  `nkeep_altroll` and
  `nkeep_psfref` — how many correlation-ranked frames to keep from the other roll and from
  the reference star — replace the ADI / RDI / ADI+RDI categorical on `run_D` / `H2`, and
  arrive on MIRI for the first time.  `nkeep_psfref = 0` **is** ADI and
  `nkeep_altroll = 0` **is** RDI, and unlike the categorical either pool can contribute
  *part* of itself, ranked per target frame.  `ReferenceLibraryGuard` already floors the two
  counts' sum at `n_min_ref`, so the empty basis is unreachable while both pure modes stay
  reachable, and it clamps `k_klip` to the basis the counts buy.
- **MIRI defaults to `--partition all`.**  This is a precondition, not a preference: under
  `partition='roll'` each partition holds only its own roll and `ref_cube` is the reference
  star alone, so `ReferenceLibrary.eligible` — which drops every frame sharing the target's
  label — returns nothing for the alternate-roll pool.  The price, as the loader's own
  docstring has always said, is one `bin`/`n_ang`/`filter`/`k_klip` block for both rolls
  instead of one per roll; the dimension count goes *down*, not up.  `--partition roll` still
  works and now says plainly that the library cannot be searched there.
- Ranges, from the headers: F444W `nkeep_altroll` 0–2 over 4 science frames with 18 φ Cen;
  F1140C 0–41 over 82 with 90 φ Cen; F1550C 0–60 over 120 with 171.  **F1065C has no
  HIP 65426 data at all** — no science and no φ Cen — so that filter is not available for
  this target whatever the STPSF cache holds.
- **`ref_targets` restricts the library by star, and the composition is now always logged.**
  "Science, and everything else is the library" is right for one programme's download and
  wrong for a directory holding several.  HIP 65426's MIRI directory also holds HD 141569A
  and HD 141569A's own reference HD 140986, so F1140C's "16 reference files" were 90 frames
  of φ Cen, 20 of HD 140986 and **12 of HD 141569A — a resolved disk, in the KL basis of a
  bare star**.  Nothing in the numbers said so, because the loader only ever reported a file
  count.  At Kevin's direction the MIRI runs now pass φ Cen alone; the loader names every
  star in the library with its file count either way, and refuses a `ref_targets` that
  matches nothing rather than reducing with an empty library.
- H2's `n_min_ref` drops 4 → 2: it is now the floor on the two counts' sum, i.e. the basis
  size, rather than on an angular reference census the searched library does not use.

## Unreleased — 2026-09-23 (destriping left 24 rows and 24 columns uncorrected)
- **My bug, spotted by Kevin as "striping at 90 degrees".**  Columns were already being
  destriped — but the star disc (r < 45) and the 4QPM boundary band (< 12 px) cross AT the
  star, and between them they covered **rows 101–124 and columns 108–131 outright** on
  MASK1140.  24 of each, with no unmasked pixel, so `nanmedian` returned NaN and
  `np.where(isfinite…)` quietly substituted a zero offset.
- A line with no offset is worse than no destriping at all: every neighbour loses its offset
  while that one keeps it, turning a gradient into a step — and the crop is centred on
  exactly those lines.  Measured on a real F1140C integration, line-to-line steps inside the
  81×81 crop went from **1.87 raw to 4.05 "destriped"** along rows.  The uncorrected
  24-column band is a vertical stripe, which is what was visible.
- **The global sky metric never noticed**: 1.826 against 1.827.  The damage was confined to
  the one region the science comes from, which is why the 1.28× in-pipeline gain reported
  earlier was real and yet the product was worse where it mattered.
- Fixed with a per-line fallback (`min_line_px`): where the full mask leaves a line too thin
  to measure, that line falls back to the star disc alone.  The ~200 rows away from the
  boundary keep their glow-stick protection and the ones crossing it are still corrected.
  In-crop row steps now **1.87 → 1.45** (1.28× better), columns 0.77 → 0.56 (1.39×), sky
  scatter unchanged at 1.69×, stellar peak −1.86% as before.
- **A repair I measured and rejected.**  Rows crossing the glow stick are also emptied, by
  the sigma clip rather than the mask, and the tempting fix is to use their unclipped median.
  On the real frame that rule empties 31 rows and **91 columns** — those run through the PSF
  wings, not sky, and their unclipped medians are 240–470 MJy/sr.  Subtracting them took the
  stellar peak from **627 to 256**.  A line with no sky in it has no sky offset to measure,
  so it is left alone; the untreated band is now ~9 rows at |y − y*| ≤ 4 instead of 24 + 24.
- Two regression tests, both of which fail on the old behaviour: one that every row the mask
  would have emptied is still corrected, and one that recovers a known injected offset row by
  row away from the glow stick.

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
