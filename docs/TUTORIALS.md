# Tutorials

Executed Jupyter notebooks in `tutorials/` (outputs included, so they read as documents
without running anything).  Authored as `tutorials/NN_name.py` (cell markers `# %%`);
`python tutorials/_build_notebooks.py [--execute]` regenerates the `.ipynb` files.

| # | notebook | data | you learn |
|---|---|---|---|
| 1 | `01_naco_betapic.ipynb` | VLT/NACO L′ β Pic, 61 frames (public VIP data, auto-download) | the full protocol on one cube: `Dataset` → reducer → space → `Runner`, the inline live display, reading `AnnulusResult` / the run directory, contrast curves, swapping in the VIP and pyKLIP engines, the equivalent command line |
| 2 | `02_sphere_hd95086.ipynb` | VLT/SPHERE IRDIS K1 + K2 HD 95086, 63 frames × 2 (auto-download from the GitHub release) | **partitions** — several data pieces tuned and selected individually (the mechanism behind nights, epochs, channels, pyNOMIC image groups), per-partition star flux and wavelength, the real planet's S/N before and after, per-partition products |
| 4 | `04_known_sources.ipynb` | VLT/NACO β Pic (same data as 1) | **known companions and disks** — what `known=[(ρ, PA)]` does to the injections, the noise rings and the contrast curve (and what it costs to forget it), `forbidden_pa` and `pixel_mask` for extended signal, tuning *for* a known companion |
| 3 | `03_jwst_nircam_hip65426.ipynb` | JWST/NIRCam F444W HIP 65426 + reference star (MAST; `tutorials/fetch_jwst_hip65426.py`) | the spaceKLIP / pyKLIP path: minimal stage-2 cleaning and alignment (DQ fill only — never sigma-clip a coronagraphic PSF), both rolls in one partition with the reference library, and **searching a backend option** (pyKLIP's `mode`: a real ADI / RDI / ADI+RDI trade-off) |
| 5 | `05_jwst_miri_hip65426.ipynb` | JWST/MIRI F1140C HIP 65426 + 16 reference + 8 background exposures (MAST; `scripts/fetch_jwst_ar.py --fetch hip65426_miri`) | **the four-quadrant phase mask** — throughput as a function of *position* rather than radius (a factor of 6.5 at one separation, boundaries 4° off the detector axes, two-fold not four-fold symmetry); the three things that have to happen to the dead zones and the one that must not (NaN ahead of a running-sum high-pass takes the whole frame); what an archive download of one programme really contains, including the half of it the pipeline has already background-subtracted; and a contrast axis anchored to another filter and then checked against a published sensitivity |
| 6 | `06_jwst_miri_hr8799.ipynb` | JWST/MIRI F1065C (and F1140C, F1550C) HR 8799 + 9-point SGD reference (MAST; `--fetch hr8799`) | **a one-roll observation** — no field rotation, so ADI is impossible and the partition layout is what decides whether `mode` means anything (the trap that invalidated two of this package's own paper runs); the reference-library dimensions that matter when RDI is all there is; four companions at four position angles as the only real test of a 2-D throughput; why the tutorial refuses to hand you their astrometry; and a stellar flux that agrees to 1% with one derived from WISE and AKARI |
| — | `notebooks/near2_production_run.ipynb` | NEAR campaign (private) | the production run from Jupyter (background thread, watch cell, resume) |

Reading order: 1, then 4 if your field has a real companion or disk, then 2 or 3 depending
on your data.  JWST/MIRI: 3 first for the spaceKLIP path, then 5 for everything the
four-quadrant masks change, then 6 if your observation has a single roll (most do).  pyNOMIC users: 1 for the
concepts, then `docs/PYNOMIC.md` (the NOMIC adapter does what tutorial 2 does by hand).

## Requirements

```
pip install "klip-tpe[all]"          # includes jupyter, pyklip, vip_hci
jupyter lab tutorials/
```

Run times on a laptop (8 cores): tutorial 1 ≈ 5–10 min, tutorial 2 ≈ 10–15 min, tutorial 3
≈ 15 min after the download, tutorial 4 ≈ 1 min (no optimization run), tutorials 5 and 6
≈ 10 min each after the download (both need a cached STPSF throughput map for their
filter — see their last section; F1550C's is not computed yet).  Most of the time goes into the live panel; `every=2` in
`LiveDisplay` halves it, `show=False` removes the window but keeps the frames on disk.

## Running the notebooks headless

```
python tutorials/_build_notebooks.py --execute            # 3, 5 and 6 skip themselves without the MAST files
python tutorials/_build_notebooks.py --only 05_ 06_      # rebuild the structure without running
python tutorials/_build_notebooks.py --execute --only 01_
```
