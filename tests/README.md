# klip_tpe test suite

## Running

```bash
cd /home/claude/klip-tpe-py
python3 -m pytest -q                 # whole suite, ~20 s on 2 cores
python3 -m pytest -q tests/test_runner.py -k resume     # one area / one test
python3 -m pytest -q -rx             # also list the xfail reasons (known core bugs)
```

Only `numpy`, `scipy`, `astropy` (FITS products) and `pytest` are needed. All runs
use the synthetic reducers in `klip_tpe.synthetic`, so no data files are read and
every test is deterministic (fixed seeds).

Tests marked `xfail(strict=True)` document behaviour that contradicts the reference
semantics or the docstrings; they start passing (and therefore *fail* as
`XPASS(strict)`) once the core code is fixed, at which point the marker should be
removed.

## Layout

| file | covers |
|---|---|
| `conftest.py` | `build_synthetic_run()` (2 partitions, per-partition `bin`/`k_klip`, two-slot selection, clean-subtracted Mawet objective, tiny budgets) and small space fixtures |
| `test_space.py` | `kgrid` layout; `Param` float/int/categorical kinds, grid snapping, encode/decode; `replicate` ordering; `tie`; two-slot and binary selection decode (positional index semantics, never-empty rule); `default_vector` / `encode` / `decode` round trip; representative medians; `to_dict` / `from_dict` |
| `test_optimizers.py` | `parzen_bandwidth` (Scott rule + floor), `parzen_density` normalisation, `block_log_density` vs univariate and underflow safety, `resolve_blocks` modes, `tpe_propose` (in-bounds sanitized output, concentrates near a 2-D quadratic optimum vs random, categorical preference), `History` best/order/NaN round trip, `TPE.ask` phase flags and batch accounting, `link_params`/`warmstart_tie`, `GridSearch` (cell coverage, cell centres never on a box face, refinement instead of wrap-around so no cell repeats, `gpts` fits the budget, declared grid axes scanned first), `RandomSearch`, `make_optimizer` |
| `test_miri.py` | MIRI coronagraphic modes and published wavelengths, `default_azimuths`/`default_separations` (dense on the boundaries given, reaching the field edge), 2-D throughput interpolation including the 360-degree seam, `quadrant_mask` is a cross rather than a disc and follows a rotated mask, Lyot spot and support bar, `forbidden_pa` in sky angle over a whole roll sequence (and its two-roll floating-point tie), the injector attenuating each frame by its own throughput while a radial model still gets one for the sequence, `typical_throughput` not warning, `load_miri`/`apply_quadrant_mask` masking dead zones with NaN, `dead_zone_pixel_mask` describing the SAME geometry as `forbidden_pa` on the ring they are both defined on (the pair is passed together, and a sign error in either composition would score injections against noise measured on the other side of a boundary), `model_for_datasets` routing MIRI to the 2-D model and NIRCam to the radial one; one STPSF-backed check that the real map has the four-quadrant structure |
| `test_run_mwc758.py` | The three failures this driver has already had: `apply_match_idl` editing the space in place rather than rebuilding it (partitions, `fixed` and the projection survive, and it still decodes to a selected partition), `crop_to_finite` on NaN-padded `_cen` cubes (square, symmetric about the star, intersection of every cube, and refusing a region too small), and `load_psf` binning the template by the PIXELSCL *ratio* while preserving total flux rather than adopting the template's scale as the data's. Plus the pixel scale and annulus geometry that put the companion inside the searched band |
| `test_fetch_jwst_ar.py` | The MAST helpers that decide what a query MEANS, offline: planned-vs-delivered observations (GO 11225 listed sixteen public rows with no products behind any of them), the science / reference / background split, numeric programme sort, de-duplication |
| `test_load_calints.py` | The JWST loading conventions that fail silently rather than loudly, on `calints` the test writes itself: `PA = ROLL_REF - V3I_YANG * VPARITY` (both VPARITY signs), `partition='roll'` vs `'all'`, the science/reference split by TARGPROP (dashes and underscores ignored, and not by file order), the deliberately odd crop, `star_center` overriding CRPIX, `repair='dq'` filling only the flagged pixels, and the scale/filter carried into `info` |
| `test_run_miri.py` | `scripts/run_miri.py` end to end on synthetic `calints` written by the test: the 2-D model reaching the reducer, injections barred from the dead sectors, a NIRCam sequence refused rather than reduced with MIRI geometry, and `--check` refusing a radial throughput on a four-quadrant mask. Plus the failure that killed the first real run: the dead zones must leave the noise estimate (`pixel_mask`, agreeing with the forbidden sectors) and STAY IN the cube, because `highpass(nan_aware=False)` is a running-sum filter that poisons everything downstream of the first NaN -- one NaN in the corner of an 81x81 frame takes three quarters of it, the MIRI dead zones reach the frame edges, and `bin_frames` then drops every all-NaN frame as empty. `--nan-dead-zones` is opt-in and measured; the reducer's guard is pinned |
| `test_metrics.py` | `source_xy` PA conventions (PA 0 = +y, PA 90 = -x), `radprof`, `declip`, `gaussian_kernel`, `kernel_from_profile`, `clip0`, `mawet_peak_snr` (planted source, empty ring, negative, NaN inside 1 FWHM, exclusions, penalty modes), other estimators, metric objects, `Objective.score_search == raw - max(clean, 0)` with NaN dropping, aggregates, `nanmedian_even` |
| `test_klip.py` | `frame_selection_mask` rules, `bin_frames` / `bin_angles` PA-span closing, `highpass` border/width bump/NaN, `rotate_ccw` direction + inverse, `derotate` + `nw_ang_comb` on a rotating planet, `klip_basis` orthonormality / cap / zero eigenvalues, `zone_indices`, `klip_annular` fast vs slow (auto-fast rule), `k_scan` slices == single-k, starved-target drop vs safety floor, `n_ang` zones, `KLIPReducer` end to end |
| `test_injection.py` | `shift_bilinear`, `add_stamp` sub-pixel centroid + clipping, `GaussianPSF` / `TemplatePSF` / `LibraryPSF` normalisation, interpolation and `ok` flag, fallback model, detector azimuth for given parang/truenorth and PA after derotation (checked with `metrics.source_xy`), math convention, anisotropic stamp rotation |
| `test_feasibility.py` | `ref_fraction` on synthetic angles, `max_feasible_angsep` monotone, `min_feasible_anglemax`, `ReferenceCountGuard` projection (infeasible pair snapped, feasible untouched, anglemax raised), random redraw inside the feasible range, zone vs searched `inrad`/`outrad`, per-partition + global dims, binning/tags census, `compose` |
| `test_runner.py` | full run products (txt/jsonl/checkpoint/final_results/annulus dirs), checkpoint state, validation table (`n_top` rows, `n_valid` trials, winner elected on the validated score incl. a forged-history election test), calibration convergence into the target window (also from a 100x wrong `contrast0`, forced contrast), re-calibration revisit (triggers, restarts the annulus, duplicate rows in `results.txt`, disabled when forced / budget 0), crash + resume twice reproducing an uninterrupted run exactly, resume after a completed annulus, `k_mode="scan_rescore"`, grid and random modes, `batch=3`, single (non-partitioned) reducer without validation, failed evaluations, `RunConfig` round trip |
| `test_testbed.py` | `RidgeObjective` true maximum 10 at the centres (both orientations), blocks, paired warm-up stream in `run_tpe_on_objective`, short `compare_density_models` (2 seeds, 80 evals) |

`test_verify_stack.py` and `test_bench_plots.py` (verification stack, benchmark
harness and plots) were added separately and are not described here.

## Adding tests

* Keep budgets tiny: the synthetic reducer costs ~10 ms per reduction, and the
  runner does 2 reductions per evaluation (+ 2 per calibration trial, + `n_valid`
  + 1 per validation candidate).
* `build_synthetic_run(**overrides)` accepts any `RunConfig` field; pass
  `calibration=CalibrationConfig(recal_budget=0)` when a test counts rows in
  `results.txt`, because a re-calibration restart appends the aborted rows.
* Anything that must not modify `klip_tpe/` but works around a known issue should
  be an `xfail(strict=True)` with the file:line of the suspected bug in the reason.
