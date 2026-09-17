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
| — | `notebooks/near2_production_run.ipynb` | NEAR campaign (private) | the production run from Jupyter (background thread, watch cell, resume) |

Reading order: 1, then 4 if your field has a real companion or disk, then 2 or 3 depending
on your data.  pyNOMIC users: 1 for the
concepts, then `docs/PYNOMIC.md` (the NOMIC adapter does what tutorial 2 does by hand).

## Requirements

```
pip install "klip-tpe[all]"          # includes jupyter, pyklip, vip_hci
jupyter lab tutorials/
```

Run times on a laptop (8 cores): tutorial 1 ≈ 5–10 min, tutorial 2 ≈ 10–15 min, tutorial 3
≈ 15 min after the download, tutorial 4 ≈ 1 min (no optimization run).  Most of the time goes into the live panel; `every=2` in
`LiveDisplay` halves it, `show=False` removes the window but keeps the frames on disk.

## Running the notebooks headless

```
python tutorials/_build_notebooks.py --execute            # all three (3 skips itself without the MAST files)
python tutorials/_build_notebooks.py --execute --only 01_
```
