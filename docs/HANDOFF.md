# klip-tpe — handoff notes (2026-09-11)

Python port of the IDL `optimize_near_2_tpe` / `reduce_near_2` optimizer.  This page is
the entry point for someone picking the package up; `README.md` has the layout,
`STATUS.md` the ledger, `docs/IDL_FINDINGS.md` the cross-checks against IDL, and
`design_notes/A–D.md` the specification the code follows.

## 1. Where things are

* Package: `klip-tpe-py/` (this folder).  `pip install -e ".[plots,test]"`, Python ≥ 3.9,
  deps numpy / scipy / astropy (+ matplotlib for plots and the live display).  It installs
  fine next to pyNOMIC (which wants Python ≥ 3.14 and vip_hci / pyklip; klip-tpe needs
  neither and never imports pyNOMIC — it only reads its files).
* Tests: `python -m pytest -q` — 208 tests, ~2.5 min, all synthetic (no data needed); the
  pyKLIP / VIP tests skip when those packages are not installed.
* GitHub-ready: MIT `LICENSE`, `.gitignore`, `CHANGELOG.md`, CI workflow in `.github/workflows/tests.yml`.
* NEAR data copy for regression work: `/Volumes/RAID36TB/NEAR2_py` (see `STATUS.md`).

## 2. State of the port

Everything in the IDL production protocol is ported and has been exercised on the NEAR
data: TPE (univariate default, `pbest = 0`, `blocks` option), calibration, search,
validation, KLIP-FM cross-check curve, stitching, verify / param_verify / candidates,
live display and movies, `extend`, `opt_width`, scan k-modes, checkpoint / resume at any
point (search, validation, post-annulus hooks, param_verify).  Reduction agreement with
IDL is 4–24 % in residual noise and 0.84–1.00 in injected-source response for typical
configurations; one open discrepancy remains (below).

Injection geometry follows the 2026-09-05 IDL change: two sources at the annulus'
area-weighted mid radius, 180° apart (`RunConfig.pair_area_midpoint`); the band is the
annulus inset by one FWHM per edge with a 1.5-FWHM inner clamp; n ≥ 3 keeps the ladder.

### Open items

1. **Low-residual regime discrepancy** (docs/IDL_FINDINGS.md §3): for long sequences with
   `angsep ≈ 0` and `k ≥ 18` the Python residual is ~1.9× noisier than IDL's at the same
   parameters (elsewhere they agree).  Not a parameter-mapping issue; needs one IDL serial
   reduction without `/lean` plus the binned angles to bisect.  Until then, treat Python
   and IDL absolute S/N values as *not* interchangeable in that corner; rankings agree.
2. `dthmax` is not forwarded to IDL night-workers (harmless today).  IDL-side `opt_width`
   guard `r_cap − inner < w_lo` is queued; the port already has it.
3. Not ported: `near2_view` (interactive viewer), MWC/LMIRCam/HPBoo IDL adapters (write
   them like `instruments/nomic.py`), a process pool (threads are used; numpy releases the
   GIL, so `--workers 4` already parallelises across partitions).

## 3. Running

Production protocol from a terminal or notebook: `docs/RUNNING.md` (`scripts/run_near2_production.sh`,
`notebooks/near2_production_run.ipynb`; all cores by default, products in `<root>/comb/opt/run_*`).

NEAR (regression / comparison with IDL):

```
klip-tpe near --root /Volumes/RAID36TB/NEAR2_py --nights 1 2 3 4 5 6 --run-dir runs/ann1 \
    --ann-edges 0 20 --n-iter 500 --n-init 50 --use-contrast 6e-5 --workers 4 --verify --candidates
```

Interrupted?  `klip-tpe resume --run-dir runs/ann1 --root ... [same instrument flags]`.
A run can be killed anywhere and resumed; validation candidates, post-annulus hooks and
param_verify reductions are all checkpointed.  `klip-tpe extend --n-iter 800` reopens a
finished annulus (IDL `extend_ann`).  `klip-tpe plots --run-dir ...` regenerates figures.

## 4. Using it with pyNOMIC (LBTI/NOMIC)

`klip_tpe.instruments.nomic` reads what the pyNOMIC notebooks leave behind:

| pyNOMIC product | used for |
|---|---|
| `masked/*.fits` (or `aligned/`), sorted | the frames; cropped to `2*crop_half` around the frame centre where `frame_registration` put the star (150 px like pyNOMIC's `crop_size=150`) |
| `<obj>_NOMIC_chop_correction.npz` | `chops` (`CHOP_A`/`CHOP_B` → partitions), `header_info[1]` (parallactic angles) |
| `<obj>_NOMIC_evaluated.npz` | `correlations → corrs`, `background_dev/median → noises`, `residual_dev/median → coronoise` (the three frame-selection dimensions), `reffits` (per-frame Airy fit → injection PSF and FWHM) |
| `<obj>_NOMIC_binned_evaluated.npz` | the same for pyNOMIC's own temporal bins (`--binned`) |

Command line:

```
klip-tpe near --instrument nomic --root <pyNOMIC working dir> --obj procyon \
    --run-dir runs/procyon_ann1 --ann-edges 20 45 --n-iter 300 --n-init 50 --workers 4 \
    --use-contrast 2e-4 [--binned | --pre-bin 20] [--parang-sign -1] [--truenorth 0]
```

or from Python:

```python
from klip_tpe.instruments import nomic
from klip_tpe import Runner, RunConfig, CalibrationConfig
ds = nomic.load_pynomic("/data/procyon_reduction", "procyon", pre_bin=20)     # {'procyonA': Dataset, 'procyonB': Dataset}
red = nomic.make_reducer(ds, max_workers=4)
space = nomic.make_space(red); space.project = nomic.make_guard(red)
obj, samp = nomic.default_config(red)
cfg = RunConfig(ann_edges=[20, 45], n_iter=300, n_init=50, calibration=CalibrationConfig(forced=[2e-4]))
Runner(red, space, obj, samp, cfg, "runs/procyon_ann1").run()
```

### Things to check on the first real data set (in this order)

1. **Memory.**  Unbinned pyNOMIC sequences are ~40 000 frames; at 150×150 float32 that is
   3.6 GB per chop state.  Use `--binned` (pyNOMIC's 200-frame bins, ~200 frames) or
   `--pre-bin N` (plain mean of N consecutive frames per chop state at load time).  The
   searched `bin` then acts on top of that.  `--max-frames` is for quick tests.
2. **Parallactic-angle sign / true north.**  The loader assumes pyNOMIC's `header_info[1]`
   is the angle such that rotating a frame counter-clockwise by `parang` puts north up
   (pyNOMIC's own `inject_source` uses detector azimuth `PA − parang`, which is that
   convention).  Verify on a known companion or binary: run a short search, look at
   `annulus01/best_clean.fits` (north up, east left after derotation) and check the PA.
   If it comes out mirrored in angle, use `--parang-sign -1`; a constant offset is
   `--truenorth`.  This cannot be established from the code alone.
3. **FWHM and λ/D.**  Defaults: `pxscale 0.0179"`, N′ 11.1 µm, D = 8.4 m → λ/D = 15.2 px;
   the FWHM comes from the median Airy fit (`1.028·σ`, ≈ 12–14 px in practice) — the
   notebooks use `fwhm=14`.  Override with `--fwhm-px`; for Fizeau (double-sided) data the
   relevant baseline is not 8.4 m — set `--diam` to what the `angsep` unit should mean.
4. **Injection PSF and contrast normalisation.**  Default `--psf airy` = what pyNOMIC's
   own `inject_source` does: frame *j* is injected with pyNOMIC's obstructed-Airy fit of
   frame *j* (`reffits[j]`: `airy_disk(c·amp_j, σx_j, σy_j, p_j)`), so contrasts are
   directly comparable between the two tools and are relative to the fitted stellar
   amplitude of each frame.  (With `--binned` / `--pre-bin` the per-frame fits no longer
   line up with the frames, so the median fit is used for all of them.)  If the star is
   saturated or the fit sits on a masked core, calibrate the fitted amplitude against an
   unsaturated exposure and scale `--use-contrast`.  `--psf frame` is an alternative that
   injects each frame's own star as an empirical template (needs an unsaturated core in
   the registered frames); `--psf gaussian` is for smoke tests only.
5. **NaN ghosts.**  `mask_files` NaN-masks the over-subtracted PSF ghosts; the loader
   zeroes them (`nan_policy='zero'`).  Keep the search annuli away from the ghost radius
   or expect a dead sector there.
6. **Search ranges** are the NEAR production ones (`bin 5–30`, `filter 0–25 px`,
   `anglemax` up to the sequence's PA span, `k_klip` on the non-uniform grid up to
   `nframes/5`).  With NOMIC's larger FWHM the `filter` upper bound (25 px < 2 FWHM) is
   the first thing to widen: `nomic.make_space(red)` returns a `SearchSpace` whose params
   can be edited before running (`space["filter_procyonA"].hi = 60`), or pass
   `defaults=`/`k_klip_max=`.

## 4b. pyKLIP, VIP, spaceKLIP and custom pipelines

`--backend pyklip|vip` (CLI) or `make_reducer(..., backend=...)` swaps the PSF-subtraction
engine under the same pre-processing and optimizer; `backends/spaceklip.py` ingests JWST
products (spaceKLIP database or `*_calints.fits`) into per-roll partitions with the RDI
library and reduces them with pyKLIP as spaceKLIP does.  Parameter mapping and the
convention checks are in `docs/BACKENDS.md`.  To optimize a pipeline of your own, see
`docs/CUSTOM_PIPELINE.md` (`FunctionReducer`: one function; `ExternalReducer`: full control).
pyKLIP / VIP are optional extras (`pip install -e ".[pyklip,vip]"`); their tests skip when
absent.

### Adding another instrument

Copy `instruments/nomic.py`: produce `Dataset(cube, angles, tags, texp, name, meta)` per
partition, pick an `InjectionModel` (`AiryPSF`, `TemplatePSF`, `GaussianPSF`, `FramePSF`,
`LibraryPSF`), build reducers into a `PartitionedReducer`, and reuse `near.make_space` /
`nomic.make_guard` / `nomic.default_config`.  Nothing in `Runner` is instrument-specific.

## 5. Practicalities

* Long runs: one evaluation on six NEAR nights is 20–200 s with 2 threads; budget
  accordingly.  Checkpoints are written after every evaluation, so preemption is cheap.
* Products per annulus land in `run_dir/annulusNN/` (`winner.json`, `best_*.fits`,
  `contrast_curve.txt` with the FM section, `verify_report.txt`, `param_verify/`), run-level
  stitches and `cand/` in `run_dir/`; `results.txt` mirrors IDL's `optimize_tpe_results.txt`.
* `results.jsonl` + `checkpoint.json` are the source of truth for resume; do not edit.
