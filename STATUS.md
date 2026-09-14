# klip-tpe-py — status (2026-09-12)

Package is functional end to end; 209 tests pass (`python -m pytest -q`, ~2 min).

Done
- Core: SearchSpace, TPE (univariate default, block-multivariate via `blocks`, `pbest`=0 default per addendum 2),
  Random/Grid, Mawet metric + clean-subtraction clamp, position sampler, KLIP engine (ADI, RDI/ARDI, k-scan,
  KLIP-FM), injection models (Gaussian / template / N4 library), partitioned reducer, feasibility projections
  (FrameSelectionGuard + ReferenceCountGuard, composed; one shared keep rule `frame_selection_mask`),
  Runner (calibrate/search/validate/products, per-eval checkpoint, deterministic resume, `extend`, `k_mode`
  search|scan_rescore|scan, `opt_width`).  Resume is now complete: validation checkpoints per candidate
  (`val_candNN.pkl`), post-annulus hooks are recorded in the checkpoint and finished on resume, param_verify
  caches its per-config reductions — a run can be interrupted anywhere and picks up where it stopped.
- Injection geometry per addendum 2 §2b/§2c: two sources at the annulus' area-weighted mid radius
  sqrt((r_in²+r_out²)/2) = 0.645" for annulus 1 (verified against IDL run_20260906_173000), applied by the caller
  collapsing the band (`RunConfig.pair_area_midpoint`; `--ladder-pair` restores the old ladder); one sampler
  object serves every stage.  `opt_width` stops when r_cap − inner < w_lo.
- Products: contrast curve + KLIP-FM cross-check curve, stitched annuli (tile + legacy), S/N maps, FITS/params
  tables/setup files, verification stack (verify / candidates / param_verify), bench harness + plots.
- Display: `LiveDisplay` (step panels in the IDL `near2m_show` layout, black live window + white snapshot, corner, landscapes, parameter histories, EDF, night map, k bookkeeping,
  calibration/validation panels, intro) + `animate.make_movie`; `render_steps` reproduces panels post hoc.
  Every IDL figure family has a counterpart (importance, paracoord, rank, slice, products sheet, verify
  limits / subsets+STIM books added 2026-09-12; mapping table and side-by-side sheets in
  `KLIP-TPE/idl_vs_python/displays/`); only `eval_walkbest` and the classical-ADI cell are partial.
- Synthetic testbed (`testbed.py`, port of near2_mvtest) and synthetic reducers for tests.
- NEAR adapter (instruments/near.py) and pyNOMIC/LBTI-NOMIC adapter (instruments/nomic.py: chop-state partitions,
  frame-quality tags, pyNOMIC's per-frame Airy fit as injection PSF like its own `inject_source`; `FramePSF` empirical option) + CLI (`klip-tpe near|resume|extend|replay|testbed|plots`,
  `--instrument near|nomic`, `--backend klip|pyklip|vip`).  Backends: pyKLIP and VIP engines behind the shared
  pre-processing (`backends/`), spaceKLIP/JWST ingestion, `FunctionReducer`/`ExternalReducer` for custom pipelines
  (docs/BACKENDS.md, docs/CUSTOM_PIPELINE.md).  Handoff notes: docs/HANDOFF.md.  Repo files for GitHub in place.
- IDL cross-checks (docs/IDL_FINDINGS.md): parstr bug found and now fixed on the IDL side; keyword audit done
  (only `dthmax` still not forwarded — harmless today); metric agrees; density-model experiment reproduced;
  post-fix reductions agree to 4–24 % in residual noise and 0.84–1.00 in injected-source response for 10/12
  products.

Open
1. Residual-noise discrepancy in the very-low-residual regime (long nights, angsep≈0, k≥18: Python ≈1.9× noisier,
   clean-image corr ≈0.55).  Not a parameter-mapping issue (k, angsep, anglemax, cuts probed).  Needs one IDL
   serial reduction without `/lean` plus binned angles (exact recipe in docs/IDL_FINDINGS.md §3).
2. Real-data heavy tests done 2026-09-05: 6-night 60-eval full-protocol run (validated winner 3.63, FM curve,
   verify, param_verify, candidates, stitch), `extend` 60→68 (re-validated winner 4.22), `k_mode=scan_rescore`
   and `opt_width` (3 adaptive annuli to r_cap) smoke runs, and repeated interrupt/resume across all phases.
   Not yet exercised on real data: RDI/ARDI, multi-annulus fixed edges beyond the smoke tests, and the pyNOMIC
   adapter (tested on synthetic pyNOMIC-format files only — PA sign / FWHM / contrast checks in HANDOFF.md §4).
3. Not ported: `near2_view` interactive viewer; MWC / LMIRCam / HPBoo adapters (pattern: instruments/nomic.py); IDL worker pool (Python uses threads in `PartitionedReducer`; a process pool is a
   small change).

Data: /Volumes/RAID36TB/NEAR2_py — n1..n6 (full cubes + AB_cube_crop150.fits, angles, tags, texp, PSF), psflib/,
ref_run/run_20260818_223040 (log, setup, eval1353 products), idl_products/ (harvested per-night products + logs,
snaps/ = pre-fix run 20260904_174912, snaps2/ = post-fix run 20260905_102602).
