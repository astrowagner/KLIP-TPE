# Tutorials

The tutorials are Jupyter notebooks in `tutorials/`. Each one is written as
`tutorials/NN_name.py`, with `# %%` cell markers, and `python tutorials/_build_notebooks.py`
turns the scripts into `.ipynb` files (`--execute` also runs them and stores the outputs).
All six are stored with their outputs, so they read as documents without running anything.
Notebooks 1, 2 and 4 run on data the package downloads itself. Notebooks 3, 5 and 6 need
JWST data from MAST and the STPSF model of the coronagraph.

| # | Notebook | Data | What it covers |
|---|---|---|---|
| 1 | `01_naco_betapic.ipynb` | VLT/NACO L′ β Pic, 61 frames (public VIP data, downloaded automatically) | The whole protocol on one cube: `Dataset`, reducer, search space and `Runner`. The live display, the results in `AnnulusResult` and the run directory, contrast curves, the VIP and pyKLIP engines, and the same run from a terminal. |
| 2 | `02_sphere_hd95086.ipynb` | VLT/SPHERE IRDIS K1 and K2 of HD 95086, 63 frames each (downloaded from the GitHub release) | **Partitions.** Pieces of data tuned and selected individually, the mechanism behind nights, epochs, channels and pyNOMIC image groups. A star flux and a wavelength per partition, the real planet's S/N before and after, and the per-partition products. |
| 3 | `03_jwst_nircam_hip65426.ipynb` | JWST/NIRCam F444W of HIP 65426 and its reference star (MAST, `tutorials/fetch_jwst_hip65426.py`) | The spaceKLIP and pyKLIP path. Minimal stage-2 cleaning and alignment (fill the flagged pixels and never sigma-clip a coronagraphic PSF), both rolls in one partition with the reference library, and **searching a backend option**, pyKLIP's `mode` (ADI, RDI or ADI+RDI). |
| 4 | `04_known_sources.ipynb` | VLT/NACO β Pic (the data of tutorial 1) | **Known companions and disks.** What `known=[(ρ, PA)]` does to the injections, the noise rings and the contrast curve, and what forgetting it costs. `forbidden_pa` and `pixel_mask` for extended signal, and tuning *for* a known companion. |
| 5 | `05_jwst_miri_hip65426.ipynb` | JWST/MIRI F1140C of HIP 65426, its reference star and the program's background pointings (MAST, `scripts/fetch_jwst_ar.py --targets hip65426_miri`) | **The four-quadrant phase mask.** Throughput as a function of position rather than radius: a factor of 6.5 at one separation, boundaries 4° off the detector axes, and two-fold rather than four-fold symmetry. The three things that have to happen to the dead zones, and the one that must not. What an archive download of one program contains, including the files the pipeline has already background-subtracted. A contrast axis anchored to another filter and checked against the published values. |
| 6 | `06_jwst_miri_hr8799.ipynb` | JWST/MIRI F1065C (also F1140C and F1550C) of HR 8799 and its reference star in a 9-point small-grid dither (MAST, `scripts/fetch_jwst_ar.py --targets hr8799`) | **A one-roll observation.** With no field rotation, ADI is impossible, and the partition layout decides whether `mode` means anything. The reference-library dimensions that matter when RDI is all there is, four companions at four position angles as a test of a two-dimensional throughput, why the tutorial leaves their positions to you, and a stellar flux that agrees to 1% with one derived from WISE and AKARI. |
| — | `notebooks/near2_production_run.ipynb` | NEAR campaign (private) | The production run from Jupyter: a background thread, a watch cell, and resuming. |

Every notebook that searches draws a **live panel** while it runs. The panel is where you
watch a search converge, and without it a run is a silent process for several minutes. Every
search uses the budget tutorial 1 explains: 300 evaluations, the first 40 of them random
warm-up, and the six best candidates validated on eight fresh injection sets each. Each
evaluation averages three injection draws, except in tutorial 3, which uses one. A smaller
budget does not give a rougher answer but a meaningless one ([BUDGET.md](BUDGET.md) has the
measurements, on tutorial 1's data). Where a companion's position is given, the injections are
also kept a few FWHM clear of it (`sampler.excl_fwhm`), so that none of them lands beside it in
the panel. Tutorial 4 shows what the radius does, and tutorial 6 leaves its planets' positions
to you.

Reading order: 1, then 4 if your field has a real companion or disk, then 2 or 3 depending on
your data. For JWST/MIRI, read 3 first for the spaceKLIP path, then 5 for everything the
four-quadrant masks change, then 6 if your observation has a single roll. pyNOMIC users need 1
for the concepts and then `docs/PYNOMIC.md`, because the NOMIC adapter does what tutorial 2
does by hand.

## Requirements

```
pip install "klip-tpe[all]"          # includes jupyter, pyklip and vip_hci
jupyter lab tutorials/
```

Tutorials 3, 5 and 6 also need `astroquery` for the download, and the STPSF model of the
coronagraph. Either install STPSF (Python ≥ 3.10, with its data files), or copy its cache files
into `$KLIP_TPE_DATA/stpsf_cache` from a machine that has computed them. Tutorial 3 needs two
cache files, and tutorials 5 and 6 need three per filter (the stamp grid, the throughput map
and the encircled energy). A MIRI throughput map takes about 25 minutes to compute, and the
computation is checkpointed. When the data are there but a cache file is missing, the notebook
stops with that file's name.

## Run Times

Tutorial 1 makes about 1,300 reductions, which take 2.5 to 5 minutes on one core without the
display. With the live panel it took 13 minutes on a two-core machine. Tutorial 2 took 27
minutes on the same machine with the panel. On a Mac, with pyKLIP doing the reductions,
tutorial 3's two searches took 11 and 13 minutes, tutorial 5's just under two hours (82
science and 90 reference frames), and tutorial 6's 19 minutes (9 and 18). Tutorial 4 runs in
about a minute, because it makes no search. Most of the time with the display goes into
drawing it. `every=2` in `LiveDisplay` draws every second
evaluation, and `show=False` removes the window but keeps the frames on disk
([BUDGET.md](BUDGET.md) has both measurements).

## Running the Notebooks Headless

```
python tutorials/_build_notebooks.py --execute --only 01_ 02_ 04_   # the three that download their own data
python tutorials/_build_notebooks.py --execute --only 05_           # one JWST notebook
python tutorials/_build_notebooks.py --only 03_                     # rewrite without running
```

Without `--execute`, the builder writes the notebook without outputs, in place of the stored
one. Without the MAST files, notebooks 3, 5 and 6 skip their cells and say so. With the files
but without the STPSF model, they stop with the name of the missing cache file, which also
stops a `--execute` over all six.
