# Changelog

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
  table and gives the PDFs real, selectable text.  Two regression tests in `test_display.py`.
  pyKLIP's `klip_parallelized` draws a tqdm bar per call whatever `verbose` says — in a notebook with
  ipywidgets that is one widget per reduction (tutorial 03: 426 of them, 3.7 MB of widget state) —
  so the backend swaps pyklip's `trange`/`tqdm` for disabled ones (`KLIP_TPE_PYKLIP_PROGRESS=1`
  keeps them).  All four notebooks rebuilt: no warnings, no widgets.
- Tutorial 03's `repair()` cell — the version students copy — now fills DQ pixels only, and the
  text says why a value-based outlier filter must never be run on a coronagraphic PSF.  Notebook
  rebuilt from scratch (RDI k=10: planet S/N 12.0; ADI −0.2; ADI+RDI 1.2).

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
