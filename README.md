# klip-tpe

**Bayesian optimization of high-contrast imaging reductions.**  klip-tpe searches the
parameter space of a KLIP / ADI / RDI reduction — KL modes, temporal binning, high-pass
filter, azimuthal zones, reference-exclusion angles, frame selection, and *which* nights,
epochs, channels or groups to combine — with a Tree-structured Parzen Estimator.  Every
trial is scored by the S/N of synthetic companions injected at fresh positions; the best
candidates are then re-scored on independent injections, and only the *validated* winner is
reported.  It ships with its own annular KLIP engine and drives pyKLIP and VIP as
alternatives, reads VLT/VISIR-NEAR, LBTI/NOMIC (pyNOMIC), JWST (spaceKLIP) and any
registered cube, and reproduces the live display, movies and product set of the IDL
optimizer it was ported from (the KLIP-TPE code used for the NEAR and LBTI campaigns).

<p align="center"><img src="docs/img/step_display.png" width="900" alt="live panel"></p>

## Install

```
git clone https://github.com/astrowagner/KLIP-TPE && cd KLIP-TPE
pip install -e ".[plots]"                     # numpy, scipy, astropy, matplotlib, pillow, imageio, threadpoolctl
pip install -e ".[all]"                       # + jupyter, pytest, pyklip, vip_hci
# without a clone:
pip install "klip-tpe[plots] @ git+https://github.com/astrowagner/KLIP-TPE"
python -m pytest -q -m "not slow"             # the unit tests, a few minutes; the full suite runs
                                              # end-to-end searches and takes ~30 min
```

klip-tpe is not on PyPI yet; install from GitHub as above.

Python ≥ 3.9.  A GUI matplotlib backend (MacOSX, Qt, Tk) is needed for the live window;
Jupyter for the inline display.

## Sixty seconds

```python
from klip_tpe import Runner, RunConfig, ValidationConfig, datasets
from klip_tpe.instruments import generic
from klip_tpe.display import LiveDisplay

f = datasets.fetch("naco_betapic")                                   # public VIP tutorial data (beta Pic b)
ds = generic.load_cube(f["cube"], f["angles"], psf=f["psf"], name="betapic")
p = datasets.PHOTOMETRY["naco_betapic"]                              # the template is normalised, so the
sf = generic.star_flux_from_aperture_photometry(                     # star's flux has to come from published
    ds.meta["psf"], p["starphot"], p["aperture_px"])                 # photometry or the axis is not a contrast
red = generic.make_reducer({"betapic": ds}, star_flux=sf, **datasets.INSTRUMENT["naco_betapic"])
space = generic.make_space(red, k_klip_max=30); space.project = generic.make_guard(red, k_max=30)
objective, sampler = generic.default_config(red, known=[(0.452, 211.9)])   # keep injections off the planet

cfg = RunConfig(ann_edges=[8, 22], n_iter=60, n_init=15, validation=ValidationConfig(n_top=2, n_valid=3))
Runner(red, space, objective, sampler, cfg, "runs/betapic", callbacks=[LiveDisplay("runs/betapic", show="auto")]).run()
```

or, the same from a terminal:

```
klip-tpe generic --cube naco_betapic_cube_cen.fits --angles naco_betapic_derot_angles.fits \
    --psf naco_betapic_psf.fits --star-flux 3.3268e6 --pxscale 0.02719 --lam 3.8e-6 --diam 8.2 \
    --known 0.452 211.9 --ann-edges 8 22 --n-iter 60 --n-init 15 --run-dir runs/betapic --show
klip-tpe resume --run-dir runs/betapic ...        # after an interruption (or: --run-dir last)
klip-tpe plots  --run-dir runs/betapic            # regenerate the figures
```

## Tutorials (Jupyter, run end to end)

| notebook | data | what it shows |
|---|---|---|
| [`tutorials/01_naco_betapic.ipynb`](tutorials/01_naco_betapic.ipynb) | VLT/NACO L′ β Pic (public, 5 MB, auto-download) | the whole protocol on one cube, inline live display, results and products, VIP / pyKLIP engines, your own data |
| [`tutorials/02_sphere_hd95086.ipynb`](tutorials/02_sphere_hd95086.ipynb) | VLT/SPHERE IRDIS K1+K2 HD 95086 (30 MB, auto-download) | **partitions**: two channels tuned and selected individually, real planet S/N before / after, per-partition products |
| [`tutorials/03_jwst_nircam_hip65426.ipynb`](tutorials/03_jwst_nircam_hip65426.ipynb) | JWST/NIRCam F444W HIP 65426 (MAST, `tutorials/fetch_jwst_hip65426.py`) | ADI + RDI through the **spaceKLIP / pyKLIP** path, per-roll partitions, reference library |
| [`tutorials/04_known_sources.ipynb`](tutorials/04_known_sources.ipynb) | VLT/NACO β Pic (same as 1) | **known companions and disks**: `known=[(ρ, PA)]` in the injections, the noise rings and the contrast curve; `forbidden_pa` / `pixel_mask` for disks |
| [`notebooks/near2_production_run.ipynb`](notebooks/near2_production_run.ipynb) | NEAR campaign (private) | the production protocol (6 nights × 3 annuli × 15 000 evaluations) from Jupyter |

`docs/TUTORIALS.md` has the reading order and what each one assumes.

## How it works

1. **Calibrate.**  The injection contrast is tuned so that the *default* configuration
   detects the fake companions at S/N ≈ 5 — the regime where parameters matter.
2. **Search.**  Random warm-up, then TPE: each evaluation injects 2–3 companions at fresh
   positions in the annulus, reduces the injected and the clean cube, scores the
   matched-filter Mawet S/N (small-sample corrected, clean-subtracted) and updates the
   density model.  Multi-partition data get one parameter block per partition plus
   `drop1/drop2` selection slots; feasibility guards keep every proposal reducible.
3. **Validate.**  The top-`n_top` search configurations are re-scored on `n_valid` fresh
   injection sets; the winner is the best *validation* median.  Search scores are
   upward-biased and are never reported.
4. **Products.**  Winner images and S/N maps, injection-calibrated 5σ contrast curve with a
   KLIP forward-model cross-check, sensitivity-weighted stitch over annuli, parameter
   importance, corner and landscape books, verification and blind candidate search —
   see `docs/DISPLAY.md` for the panel anatomy and the product glossary.

Everything is checkpointed after every evaluation: `klip-tpe resume` continues a run
from any phase; `extend` adds evaluations to a finished one.

## Data and engines

| your data | entry point |
|---|---|
| a registered cube + angles (+ PSF, + reference cube), FITS or arrays — VIP, your pipeline | `klip_tpe.instruments.generic` / `klip-tpe generic` |
| LBTI/NOMIC through **pyNOMIC** | `klip_tpe.instruments.nomic` / `klip-tpe near --instrument nomic` — **[docs/PYNOMIC.md](docs/PYNOMIC.md)** |
| VLT/VISIR **NEAR** campaign tree | `klip_tpe.instruments.near` / `klip-tpe near` — `docs/RUNNING.md` |
| **JWST** via spaceKLIP (database or `calints`) | `klip_tpe.backends.spaceklip` — tutorial 3, `docs/BACKENDS.md` |
| any **pyKLIP** `Data` object (GPI, CHARIS, JWST, GenericData) | `klip_tpe.backends.pyklip.dataset_from_pyklip` |

| PSF subtraction | `backend=` |
|---|---|
| built-in annular KLIP (single-basis and per-target, RDI/ARDI, KLIP-FM) | `"klip"` (default) |
| pyKLIP `klip_parallelized` | `"pyklip"` |
| VIP `pca_annular` / `pca` / `median_sub` | `"vip"` |
| your own function or external program | `backends.custom` — `docs/CUSTOM_PIPELINE.md` |

The engines share injection, frame selection, filtering, binning and scoring, so a
parameter means the same thing whichever code subtracts (`docs/BACKENDS.md` maps them).

## The live display

`LiveDisplay(run_dir, show=...)` is the port of the IDL `near2m_show` window: five
image / S/N cells (test, test clean, best, running stitches), the TPE convergence trace,
S/N histogram, KLIP-FM model and injected-PSF squares, night/partition inclusion, the
SNR = 5 contrast curve, the BEST / TEST parameter vectors, elapsed / ETA / total / done
time, night-effect panels and the corner plot; single-partition runs show parameter
importance instead of the night panels.

* `show=True` — a 1850 × 990 matplotlib window (MacOSX / Qt / Tk), 1:1 pixels; `window_scale=0.6` to shrink.
* `show="inline"` — inside Jupyter, the panel updates in place in one output cell.
* `show="auto"` — inline in a notebook, window otherwise.
* `every`, `pdf_every`, `movie_every` — render / PDF / progress-movie cadence.  Frames go to
  `steps/`, the per-annulus movie to `annulusNN/progress.gif` (+ `.mp4`), the full movie to `opt_steps.gif`.
* `aliens=True` — the IDL launch movie during the first calibration.

## Command line

`klip-tpe <command> --help` for everything; the main ones:

```
klip-tpe generic  --cube ... --angles ... [--psf ...] [--ref-cube ...] --pxscale --lam --diam ...   # any cube
klip-tpe near     --root <NEAR tree> --nights 1 2 3 [...]                                            # NEAR
klip-tpe near     --instrument nomic --root <pyNOMIC workdir> --obj <name> [--groups auto] [...]     # pyNOMIC
klip-tpe resume   --run-dir <dir>|last  <same data arguments>                                        # continue
klip-tpe extend   --run-dir <dir> --n-iter <new totals>                                              # more evaluations
klip-tpe plots    --run-dir <dir>                                                                    # figures
```

Common search arguments: `--ann-edges r0 r1 [r2 ...]` (px), `--n-iter`, `--n-init` (per
annulus), `--n-top/--n-valid` (validation), `--use-contrast` (forced injection contrast,
0 = calibrate), `--k-max`, `--global-block` (one block for all partitions), `--no-selection`,
`--max-drop`, `--workers auto|N` (every core by default; forked workers share the loaded
data), `--backend klip|pyklip|vip`, `--show [window|inline|auto]`, `--display-every`,
`--movie-every`, `--verify --param-verify --candidates` (post-search stack).  `docs/CLI.md`
lists all of them.

## Package map

| module | contents |
|---|---|
| `space.py` | `Param` / `SearchSpace`: float / int / categorical dims, sampling grids, per-partition replication, tying, two-slot / binary selection, feasibility hook |
| `optimizers.py` | `TPE` (univariate or block densities, elite local moves, ε-exploration), `RandomSearch`, `GridSearch`, `History` |
| `metrics.py`, `positions.py` | matched-filter Mawet S/N, clean-subtracted `Objective`, injection-position samplers |
| `klip.py`, `injection.py` | the reference annular KLIP (frame selection, angle-aware binning, high-pass, NaN-faithful rotation, KL basis, KLIP-FM); `GaussianPSF`, `TemplatePSF`, `AiryPSF`, `FramePSF`, `LibraryPSF` |
| `reducer.py`, `feasibility.py` | `Dataset`, `KLIPReducer`, `PartitionedReducer` (per-partition blocks, forked workers), `FrameSelectionGuard`, `ReferenceCountGuard` |
| `runner.py` | calibrate → search → validate, products, checkpoints, `resume` / `extend`, callbacks |
| `display.py`, `intro.py`, `animate.py`, `plots.py` | live panel, books, launch movie, GIF / MP4, diagnostics |
| `products.py`, `stitch.py`, `verify.py`, `param_verify.py`, `candidates.py` | contrast curves, stitches, S/N maps, the verification stack |
| `instruments/` | `generic`, `near`, `nomic` adapters |
| `backends/` | `pyklip`, `vip`, `spaceklip`, `custom` |
| `datasets.py` | public example data (auto-download) |
| `cli.py`, `parallel.py` | the command line; worker / thread bookkeeping |

## Documentation

* `docs/TUTORIALS.md` — the notebooks and what to read when.
* `docs/PYNOMIC.md` — LBTI/NOMIC via pyNOMIC: loading, image groups as partitions, checks.
* `docs/RUNNING.md` — the NEAR production protocol from a terminal or Jupyter.
* `docs/DISPLAY.md` — the live panel, cell by cell, and every product file.
* `docs/BACKENDS.md` — pyKLIP / VIP / spaceKLIP: parameter mapping, conventions, ingestion.
* `docs/CUSTOM_PIPELINE.md` — plugging in your own reduction (three levels).
* `docs/CLI.md` — every command-line argument.
* `CHANGELOG.md`.

## Citing

If klip-tpe contributes to a publication, please cite the KLIP-TPE method paper
(Wagner et al., in prep.; `CITATION.cff` will carry the reference once available) and,
for the engines and data you use, pyKLIP (Wang et al. 2015), VIP (Gomez Gonzalez et al.
2017; Christiaens et al. 2023), spaceKLIP (Kammerer et al. 2022; Carter et al. 2023) and
pyNOMIC as appropriate.

## License

MIT — see `LICENSE`.
