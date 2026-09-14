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
datasets = sk.load_spaceklip(database, key="JWST_NIRCAM_NRCA5_F444W_MASKRND_MASK335R_SUB320A335R")
# or: sk.load_spaceklip(sci_files=[...calints.fits], ref_files=[...])
red = sk.make_reducer(datasets, psf_template="offset_psf_F444W.fits", star_flux=F_star, max_workers=4)
```

`load_spaceklip` takes the science/reference split from the database `obs` table (`TYPE`
SCI/REF, `FITSFILE`), reads the ImageTools products with pyKLIP's `JWSTData` when
available (else an equivalent built-in reader: `STARCENX/Y` or `CRPIX`,
`PA = ROLL_REF − V3I_YANG·VPARITY`, wavelength from `CWAVEL` or pyKLIP's filter tables),
makes one partition per roll and attaches the reference exposures as the RDI library.
`make_reducer` builds `PyKLIPReducer`s in `ADI+RDI` (spaceKLIP's default; `mode="RDI"` when
the roll is too small to separate a companion from itself — with a few degrees of roll
`ADI+RDI` self-subtracts, and `mode` can be *searched* by adding a categorical `Param`),
mapping the partitions onto threads (`pool="threads"`: pyKLIP forks its own workers) with λ/D from
the filter wavelength and D = 6.5 m. For injection pass a webbpsf / `webbpsf_ext` offset
PSF at the data's pixel scale (`psf_template`, e.g. from spaceKLIP's
`analysistools.get_offsetpsf`) and the star flux in image units (`star_flux`); without one
a Gaussian of `1.028 λ/D` and flux unit 1 is used, which is fine for parameter *ranking*
but not for calibrated contrasts.

### VIP cubes

VIP keeps cubes as plain arrays: `Dataset(cube, angle_list, texp=...)` is all that is
needed (VIP's `angle_list` has the same sign as klip-tpe's `angles`).

## Adding another engine

Subclass `KLIPReducer` and override `_subtract(bcube, bang, kp, p, filt, req, mcube,
ref_basis, fm_ref, meta) -> (image, fm_image_or_None, info_dict)`; `kp` is a
`KLIPParams` (k, zone, n_ang, angsep, anglemax, k_scan). Register it in
`klip_tpe.reducer.reducer_class` if you want a CLI name. `backends/vip.py` is the
smallest example.
