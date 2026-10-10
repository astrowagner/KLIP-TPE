# PSF-subtraction backends: pyKLIP, VIP, spaceKLIP

klip-tpe ships its own annular KLIP (`klip_tpe.klip`, the port of `reduce_near_2`), but
the optimizer does not care who does the subtraction. Three external engines are wired
in; each is a `KLIPReducer` subclass that keeps the pre-processing chain (injection,
frame selection, high-pass, binning, RDI reference handling) and replaces only the
subtract-derotate-combine step, so results and products are directly comparable across
engines on the same data.

Install what you need: `pip install pyklip`, `pip install vip_hci`; spaceKLIP itself is
only needed to *produce* the JWST products (its pyKLIP hand-over is reproduced here).

## Selecting a backend

```
klip-tpe near --instrument near  --backend pyklip ...      # NEAR data, pyKLIP engine
klip-tpe near --instrument nomic --backend vip    ...      # pyNOMIC data, VIP engine
```

```python
from klip_tpe import reducer_class
Red = reducer_class("pyklip")            # KLIPReducer | PyKLIPReducer | VIPReducer
red = Red(dataset, pxscale=..., lam_m=..., diam_m=..., injection_model=..., fwhm_px=...)
```

The instrument adapters accept `backend=` in `make_reducer(...)`.

## Parameter mapping

| searched | built-in KLIP | pyKLIP (`klip_parallelized`) | VIP (`pca_annular` / `pca` / `median_sub`) |
|---|---|---|---|
| `k_klip` | KL modes | `numbasis` (k-scan native) | `ncomp` (k-scan = one call per k) |
| `n_ang` | azimuthal zones | `subsections` | `n_segments` (pca_annular) |
| `inrad, outrad` | annulus | `IWA, OWA`, `annuli = n_annuli` | `radius_int`, `asize = outrad − inrad` |
| `angsep` (λ/D of arc) | reference exclusion | `movement = angsep·λ/D` px | `delta_rot = angsep·(λ/D)/FWHM` |
| `anglemax` | max reference PA span | `maxrot` | — (ignored; drop it from the space) |
| `bin, filter, corr_thresh, noise_max, coronoise_max` | pre-processing, shared by all backends | | |
| `comb_type` | `nwadi` / mean / median | same (combine done here) | `collapse` median / mean |
| RDI | `use_rdi`, `rdi_mode` | `mode = 'RDI' / 'ADI+RDI'` (pyKLIP `PSFLibrary`) | `cube_ref` (`use_rdi`, `rdi_mode`) |

Backend-specific fixed options live in the reducer's `defaults` (pass `defaults={...}` or
set them after construction); any of them becomes searchable by adding a `Param` of that
name to the space:

* pyKLIP: `algo` (`klip` \| `nmf` \| `empca`), `mode`, `n_annuli`, `annuli_spacing`,
  `corr_smooth`, `minrot`, `numthreads`.
* VIP: `algo` (`pca_annular` \| `pca` \| `median_sub`), `n_annuli`, `svd_mode`, `imlib`,
  `interpolation`, `min_frames_lib`, `max_frames_lib`, `nproc`.

Conventions were checked numerically (tests/test_backends.py): a companion injected at
(rho, PA) by klip-tpe is recovered at the same (rho, PA) through each engine's own
derotation (pyKLIP: aligned output derotated here; VIP: `angle_list = parang + truenorth`,
VIP derotates CCW by `angle_list`).

KLIP-FM (the forward-model cross-check curve) is only available with the built-in KLIP;
`fm_curve` is skipped silently for the others.

## Ingesting data

### pyKLIP `Data` objects

```python
from klip_tpe.backends.pyklip import dataset_from_pyklip
datasets = dataset_from_pyklip(data, crop_half=75, partition_by="roll")   # GPI, CHARIS, JWST, GenericData, ...
```

Frames are re-centred (bilinear) onto the common pixel `((nx-1)/2, (ny-1)/2)`, cropped,
and split into partitions by `partition_by`: `"roll"` (one per distinct position angle —
the JWST default), `"filenums"` (one per exposure), `"filenames"`, or `None` (one dataset);
`data.psflib` becomes `Dataset.ref_cube`. Multi-wavelength cubes: `wv_index=` picks a
channel.

### spaceKLIP / JWST

```python
from klip_tpe.backends import spaceklip as sk
datasets = sk.load_spaceklip(database, key="JWST_NIRCAM_NRCALONG_F444W_MASKRND_MASK335R_SUB320A335R")
# or: sk.load_spaceklip(sci_files=[...calints.fits], ref_files=[...])
red = sk.make_reducer(datasets, psf_template="offset_psf_F444W.fits", star_flux=F_star, max_workers=4)
```

`load_spaceklip` takes the science/reference split from the database `obs` table (`TYPE`
SCI/REF, `FITSFILE`), reads the ImageTools products with pyKLIP's `JWSTData` when
available (else an equivalent built-in reader: `STARCENX/Y` or `CRPIX`,
`PA = ROLL_REF − V3I_YANG·VPARITY`, wavelength from `CWAVEL` or pyKLIP's filter tables),
makes one partition per roll (`partition_by="roll"`) or one for everything
(`partition_by=None`) and attaches the reference exposures as the RDI library.  **Use one
partition if `mode` is to mean anything**: a partition is reduced on its own, so with one
per roll every frame in it shares a PA — pyKLIP's `ADI` then has no reference frames and
`ADI+RDI` is `RDI`.  (`load_calints(partition='all')` is the same choice for raw stage-2
files.)  `make_reducer` builds `PyKLIPReducer`s in `ADI+RDI` (spaceKLIP's default; with a
few degrees of roll `ADI+RDI` gives up companion flux to the other roll, and `mode` can be
*searched* by adding a categorical `Param`).  `angsep = 0` is handed to pyKLIP as a
`movement` of 1e-6 px, not 0: pyKLIP keeps every reference with `moves >= movement`, so 0
would put the target frame in its own basis and annihilate any source.
mapping the partitions onto threads (`pool="threads"`: pyKLIP forks its own workers) with λ/D from
the filter wavelength and D = 6.5 m. For injection pass a webbpsf / `webbpsf_ext` offset
PSF at the data's pixel scale (`psf_template`, e.g. from spaceKLIP's
`analysistools.get_offsetpsf`) and the star flux in image units (`star_flux`).  Without
them a Gaussian of `1.028 λ/D` with a flux unit of 1 is injected, so a "contrast" is a
fraction of one image unit.  On frames in MJy/sr that is far fainter than any companion:
the calibration reaches its contrast cap (`CalibrationConfig.max_contrast = 0.1`) with the
injections still invisible, and the search ranks noise.  Give both for any real run.

**pyKLIP starts a worker pool for every reduction.**  `klip_parallelized` builds a new
`multiprocessing.Pool` on each call, and under `spawn` -- macOS's default -- each worker is a
fresh interpreter importing numpy, scipy and pyklip first.  On macOS the backend therefore hands
pyKLIP a `forkserver` context: one server with pyklip imported, the workers forked from it.
Measured on MIRI F1140C evaluations (three draws each), spawn cost 18.0 and 12.6 s against 8.7
and 4.0 s with the forkserver, and the scores are identical to the last bit.
`KLIP_TPE_PYKLIP_START=spawn|fork|forkserver` overrides; elsewhere the platform default stands.

**pyKLIP's `maxnumbasis` selects the library; it does not truncate it.**  Per target frame and
sector, pyKLIP keeps the `maxnumbasis` frames of the chosen pools most correlated with the
target, and builds its modes from those.  Left unset, `maxnumbasis = numbasis`, so every mode
count `k` also picks a library of `k` frames.  In HIP 65426's F1140C data every frame of the
other roll ranks ahead of every frame of the reference star (counted per target frame, pyKLIP
2.10.1, archive calints): in Carter et al.'s frame set ADI+RDI keeps the other roll's 40 frames
alone up to `k = 40` and adds the reference star's above it (40 + 41 at `k = 81`), and every
ADI+RDI winner and default of the v8 searches holds the whole other roll plus some of the
reference star (the default, `maxnumbasis = 6` at bin 10, the other roll's 5 binned frames and
one reference frame).  The pyKLIP backend searches `mode` and `maxnumbasis` separately from
`k_klip` for this reason (`set_native_library`).

#### Raw stage-2 files: `load_calints`

`load_calints(files, ...)` reads `*_calints.fits` directly, without spaceKLIP, and does five
things to them before KLIP sees them, each one recorded:

* **One exposure, once.** A product is named after its exposure, so a name that occurs twice
  is refused.  The usual cause is two processings of one data set in one tree — MAST's
  `mastDownload/` beside a re-reduction from `uncal` — found together by a recursive search.
  Point the loader at one of them.
* **Background.** The blank-sky median of the programme's dedicated background pointings is
  subtracted from whichever frames the pipeline did not already do (`S_BKDSUB`): an archive
  download mixes the two, because science targets get the step and PSF references usually do
  not.  A mixture with no background pointing to fix it is refused.
* **Static hot pixels** (`hot_pixels=None`, the default: whenever there are background
  pointings).  A pixel that stands above (or below) both the median of its seven row
  neighbours and the median of its seven column neighbours on the blank sky, by more than
  `max(10 × robust σ, 5)` MJy/sr, is flagged in every science and reference frame and
  repaired like a DQ pixel.  The AND keeps the 4QPM glow sticks, which stand out along one
  axis only, off the list (where two sticks cross, under the star, the crossing is flagged);
  the unilluminated border is never flagged.  On ERS 1386 F1140C the pipeline's DQ array
  misses a hot pixel five pixels (0.6") from the star that no ADI+RDI reduction can fit,
  because the library carries it at a different level than the science frames.
  `hot_pixels=False` leaves them; `True` insists, and raises without a background pointing.
* **Destriping** (on by default for MIRI): row and column offsets removed on the full
  subarray, star and 4QPM boundaries masked.  On the F1140C frames it halves the ring noise
  of an every-mode ADI+RDI reduction.
* **Registration** by cross-correlation; a frame whose offset cannot be measured is left
  unshifted and counted.

`info['provenance']`, which every `Dataset`'s `meta` carries and the reducer's `describe()`
writes into `run_setup.json` (`reducer.partitions.<id>.frames`), records the files, the
pipeline version and CRDS context from their headers (`CAL_VER`, `CRDS_CTX`), the pixel
scale as read (jwst 2.0.1 MIRI products carry 0.1103"/px, jwst 1.13.4 ones 0.1100"/px), the
repaired hot pixels, what was background-subtracted, and how many frames the registration
left unshifted.  Which pipeline made the calints is a parameter of the reduction, so a run
says which it used.  `frames_signature(provenance)` names a frame treatment, for keying
anything cached from one (a companion fit, say).

Rebuilding a finished run (`scripts/library_ablation.py`, `paper_runs/miri_fig.py`) repairs
hot pixels exactly when the run did; a run from before the repair existed stays unrepaired on
its own frames (recognised by the pixel scale it recorded) and gets the loader's default on
anyone else's.  `--hot-pixels on|off` overrides.

#### STPSF off-axis PSFs

`psf="stpsf"` replaces that template with the off-axis PSF of the coronagraph itself,
computed with [STPSF](https://stpsf.readthedocs.io) (the renamed WebbPSF) from the mode in
the data's own headers:

```python
red = sk.make_reducer(datasets, psf="stpsf", star_flux=F_star, max_workers=4)
# or build it yourself and reuse the grid:
from klip_tpe import stpsf_psf
grid  = stpsf_psf.offaxis_grid("NIRCam", "F444W", image_mask="MASK335R",
                               seps_as=np.arange(0.2, 2.61, 0.2))
model = stpsf_psf.library(grid, star_flux=F_star)
red   = sk.make_reducer(datasets, injection_model=model)
```

Inside a few λ/D of a mask the PSF is neither a Gaussian nor separation-independent, so
this matters twice: injected sources get the right shape *and* the right amplitude, since
the grid also measures the mask throughput `T(ρ)` (for MASK335R it reaches half
transmission at 0.65″, against the 0.64″ IWA in the JWST documentation) and `LibraryPSF` applies it. Each
PSF costs seconds, so a grid is computed once and cached as a FITS under
`$KLIP_TPE_DATA/stpsf_cache` (default `~/.klip_tpe/stpsf_cache`).  The stamps are stored
source-centred and injected translated, never rotated (`refpa_deg=None`): the structure in a
JWST coronagraphic PSF is fixed to the spacecraft and does not turn with the companion.  Pass
`refpa_deg` to reproduce a run from before 2026-09-22, when the injector rotated them. STPSF is an optional dependency (`pip install stpsf`, Python ≥
3.10, plus its ~90 MB data files via `STPSF_PATH`); nothing else imports the module.

#### Forward-modelled matched filter (`--metric fmmf`)

KLIP is not flux-conserving: it eats part of the planet and leaves negative
self-subtraction lobes around what remains, and *how much* depends on the very parameters
being searched. Filtering with the instrument PSF therefore mismatches the signal, and the
mismatch is worst exactly where the optimizer is working hardest. `klip_tpe.fmmf.FMMFSNR`
propagates the PSF model through each configuration's own subtraction and filters with the
result (Pueyo 2016; Ruffio et al. 2017):

```bash
klip-tpe generic --cube ... --metric fmmf          # or metric="fmmf" in default_config()
```

Everything else is unchanged — the same Mawet small-sample ring statistics, the same
clean-subtraction rule, the same validation protocol — so an FMMF run and a PSF-matched
filter run differ only in the filter. The template comes from the analytic KLIP-FM image
when the reducer has one (the built-in annular KLIP); this package's pyKLIP, VIP and
spaceKLIP backends do not, and
there the runner passes the *numerical* forward model `injected − clean`, which is the same
quantity to first order and costs nothing extra because `clean_subtract=True` already
computes both. `FMMFSNR.describe()["fm_fraction"]` reports what fraction of the filters
were really forward-modelled rather than fallbacks (the per-evaluation k-scan has no
forward model and always falls back). `klip_tpe.fmmf.fmmf_map` turns a ring of test
sources into a full-frame FMMF amplitude/S-N map for the final image.

### VIP cubes

VIP keeps cubes as plain arrays: `Dataset(cube, angle_list, texp=...)` is all that is
needed (VIP's `angle_list` has the same sign as klip-tpe's `angles`).

## Adding another engine

Subclass `KLIPReducer` and override `_subtract(bcube, bang, kp, p, filt, req, mcube,
ref_basis, fm_ref, meta) -> (image, fm_image_or_None, info_dict)`; `kp` is a
`KLIPParams` (k, zone, n_ang, angsep, anglemax, k_scan). Register it in
`klip_tpe.reducer.reducer_class` if you want a CLI name. `backends/vip.py` is the
smallest example.
