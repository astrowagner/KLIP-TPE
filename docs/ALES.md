# LBTI/ALES integral-field cubes

klip-tpe reads the products of an ALES reduction (a `(wavelength, frame, y, x)` cube, its
parallactic angles, its wavelengths and a per-channel PSF template) and optimizes the
KLIP/ADI post-processing over the usual parameters plus one on/off flag per wavelength
group.  The adapter follows the ALES reduction's handoff of 2026-10-13.  The reports on the
runs live with the data.  A data set of several nights in the release layout, such as the
MWC 758 data of tutorial 7, is read as one partition per night and group (section 7).

## 1. The cube as wavelength groups

ALES delivers 99 channels at R ~ 40-55 over 2.76-4.29 um.  Neighbouring channels share a
resolution element, so a channel is not the unit to switch on and off.  The adapter cuts
the usable channels (6-88 by default) into `ngrp` groups of equal width in `ln lambda`,
about one resolution element each at `ngrp = 15` and 0.73 at the default 20, and means each
group's channels into one `(frame, y, x)` cube.  Within a group the speckle pattern scales
by at most 2 % in radius, 0.2 spaxels at 0.3", far below the FWHM, so the mean loses
nothing and the reduction costs 1/(channels per group) of a per-channel one.

The rule (`ales.group_table`): a channel's edges are the midpoints to its neighbours, the
group edges are equally spaced in `ln lambda` from the lower edge of channel `c0` to the
upper edge of `c1`, and a channel belongs to the group whose edges contain its wavelength.
At the default 20 the groups hold from 2 channels at the blue end to 6 at the red end.  The
realised table goes into the run's `run_setup.json` under `reducer.provenance.groups`, and
each group's channels and wavelengths under `reducer.partitions.gNN.frames`.

Each group is one partition of a `PartitionedReducer`, like a night or a roll:

* its own wavelength `lam_m` and FWHM `1.03 lambda/D` (about 2.1 spaxels at the blue end and
  3.0 at the red end), so `angsep` in lambda/D and the Mawet apertures are right for it;
* its own PSF template, the group mean of the per-channel template, normalised in 3
  spaxels (`ee_radius_px`).  The template is the median of the registered science frames,
  in the frames' units, so `flux_unit` is the star's flux in that group and an injected
  `contrast` is a flux ratio to the star there;
* frame-quality tags from its own frames (`quality_tags`, `tag_r_in = 2`: from 6 spaxels
  out ALES frames are background-dominated and their correlations fall);
* optionally the group mean of a reference star's cube as its RDI library (`ref=`,
  `--ales-ref`), off unless asked.

One parameter block is shared by every group (`bin`, `n_ang`, `filter`, `angsep`,
`anglemax`, the three frame-selection thresholds, `k_klip`), and the group at the mean
wavelength (g10 at the default 20) stands for the combined image: the metric's FWHM
and the sampler's floor are its (`PartitionedReducer.representative`).  The groups are
combined with equal weights.

## 2. The group flags

`selection='binary'` adds one 0/1 dimension per group, `sel_g00 .. sel_g19`.  A random
draw includes each group with probability 0.75, a proposal may switch any number off, and
an all-off draw decodes to all on.  `GroupCountGuard` (first in `ales.make_guard`) bounds
it: a proposal with more than `max_excluded` (10) groups off has groups switched back on,
nearest `prefer_um` (3.7 um by default) first, or at
random for a random draw, until exactly `max_excluded` are off.  The projected vector is
what is evaluated and recorded, so `results.jsonl` never holds a selection past the bound;
`run_setup.json` records the guard under `projection`.

An excluded group is out of the target set and the combination.  In `results.jsonl` a
record's flags are its `x` at the `sel_gNN` positions of `space.names` and, decoded,
`meta.selected` and `config.selected` (the groups included).

## 3. A reference star as a searched library

`--ales-ref NAME` attaches a reference star's group means (`NAME_cube_cen_clean.fits` beside
the target's products) to every group.  On its own that is the fixed ARDI basis of any
adapter: every reference frame in every target's basis.  `--ales-reflib` makes it a searched
library instead (`KLIPReducer.set_reference_library`, the `rotation` form, one per group):

* A target frame's basis is the science frames the ADI rule admits for it (`angsep` in the
  group's lambda/D, `anglemax`, per annulus, as without a library) plus the `nkeep_psfref`
  reference means most correlated with it in the annulus (`cc`, the NIRCam library's
  ranking).  `nkeep_psfref` is one more dimension of the shared block, from 0 to the whole
  pool, and 0 is plain ADI to the bit: that point runs the ADI code itself, shortcuts and
  the k cap included (`klip_annular`'s `library_as_adi`).  The seed sits at 0, so the
  search's first evaluation is the ADI baseline and every library draw is read against it.
* The reference frames are taken in consecutive means of `--ales-ref-bin N` (5) before
  anything else, so the count means one thing whatever the science `bin` is, and never across
  an observing block: the blocks come from the `block` column of `NAME_frametags.csv` beside
  the products (blocks of 23, 41 and 37 frames make 4 + 8 + 7 = 19 means of five, a block's
  last frames unused), the whole sequence is one block without the file.  The cache
  keeps the reference group means unbinned, so `N` may change between runs.  `run_setup.json`
  records the pool under `reducer.provenance.reference_library` (source, frames, blocks,
  `ref_bin`, means, the metric, the science pool) and per group under
  `partitions.gNN.reference_library` and `.frames`.
* The science pool is not a dimension by default (`--ales-reflib-science all`): the ADI rule
  decides it, as in a plain ADI run.  `searched` adds `nkeep_altroll`, the NIRCam form, the
  science frames ranked by correlation and the count searched, whose 0 is pure RDI.  In that
  form `ReferenceLibraryGuard` comes last in the guard chain (the searched counts are then the
  basis size, so their sum is floored and `k_klip` capped by it).  With the science pool
  whole, the census guard governs as before and `nkeep_psfref` runs free from 0.
* The similarity matrix is built once per group per distinct (binning, frame selection,
  filter, annulus) and cached, on the frames without injections.  The basis rows are the
  evaluation's own frames.  At bin 5 on 2 cores the whole pool costs a 20-group evaluation
  about a third more than plain ADI, nothing at `nkeep_psfref = 0`.
* `nkeep_psfref` appears as `nkref` in the live window and the books, like the NIRCam count.

`klip-tpe resume` restores `--ales-reflib`, `--ales-ref-bin` and `--ales-reflib-science` from
the run's record with the other ALES options.

## 4. ADI, SDI and RDI in one KLIP step

`--ales-sdi` gives every group's library a third kind of pool: the other groups' frames,
rescaled to its wavelength (the ALES reduction's note of 2026-10-14, section 3).  For a target
frame of group g at an annulus of mid radius r:

| pool | rows | a row is eligible when | count |
|---|---|---|---|
| ADI | group g's other binned frames | the planet moved by `angsep` (lambda_g/D of arc at r) through rotation, within `anglemax` | every eligible frame (not searched) |
| SDI | the other groups' binned frames of this evaluation, magnified about the star by s = lambda_g / lambda_g' | the planet's displacement by rotation and scaling together, `r sqrt(1 + s^2 - 2 s cos dPA)`, is at least that arc, and `abs(dPA) <= anglemax` | `nkeep_sdi`, 0 to `--ales-sdi-cap` (100) |
| RDI | the reference star's means (`--ales-ref`) | always | `nkeep_psfref`, 0 to the pool |

* Within a pool the eligible rows are ranked by correlation with the target in the annulus and
  the count keeps the best.  `nkeep_sdi` and `nkeep_psfref` join the shared block and seed at 0.
  `nkeep_sdi = nkeep_psfref = 0` is plain ADI to the bit and costs plain ADI: the other groups
  are prepared only when `nkeep_sdi` asks for rows.  ADI (0, 0), SDI and ASDI (`nkeep_sdi` up,
  all of the science frames), RDI and ARDI (`nkeep_psfref` up) and all three together are
  points of one space, and the optimizer elects the mix.
* One `angsep` serves both rules.  At s = 1 the displacement is the chord `2 r sin(dPA / 2)`,
  the ADI arc to first order, so the SDI rule is the ADI rule with the scale in it.  A row of
  the same time (dPA = 0) is eligible when the spectral displacement `r abs(s - 1)` alone is
  enough, which it is at large radii and between distant groups.
* The SDI rows are prepared as the group prepares its own frames (the same selection,
  injections, high-pass and binning with the searched `bin`), then magnified about the star by
  a NaN-aware cubic B-spline (`klip_tpe.rescale`, the reduction's own `asdi.rescale`): the
  high-pass before the rescaling, the injections in every group before the rescaling, so a fake
  moves the way a planet does and sits at `r s` in the rows.  A KLIP-FM model cube goes through
  the same chain, so the forward model's planet sits at `r s` too.
* Each group's frames are prepared once per reduction (`ReductionRequest.stage = "prepare"`, a
  first round of jobs) and handed to the group jobs in one shared-memory block
  (`klip_tpe.parallel.SharedArrays`) with worker processes, as they are with threads.
  `--ales-sdi-lenders all` (default) prepares every group, an excluded one too; `selected` only
  the groups the configuration includes.
* The SDI rows are ranked at zero lag against this evaluation's target frames, injections
  included, and the ranking is never cached: a cached one would make the selection depend on
  which evaluation came first, and a clean preparation of every sibling would double the
  rescaling.  The science and reference pools keep their cached ranking on the frames without
  injections.
* Cost.  A target's KLIP basis grows by up to `nkeep_sdi` rows and the Gram matrix with its
  square, and every sibling frame is rescaled once per group per reduction (about 0.5 ms per
  62 x 62 frame): each of 20 groups rescales the other 19 groups' binned frames.  At bin 5 that
  is most of the cost of the mode on two cores, and about a second per reduction on 32 threads.
  `--bin-min 5` keeps the search off the bins where it would be tens of seconds.
  `nkeep_sdi` appears as `nksdi` in the live window and the books.

`klip-tpe resume` restores `--ales-sdi`, `--ales-sdi-cap` and `--ales-sdi-lenders` as well.

## 5. Conventions

* The products are north up, east left once every frame is rotated counter-clockwise by
  `LBT_PARA + 180` deg.  The adapter adds the 180 to the angle array once
  (`angle_offset`, `--ales-angle-offset`) and keeps `truenorth = 0`.  A source therefore
  moves clockwise on the detector as the parallactic angle grows.
* The star is at the array centre `((nx - 1) / 2, (ny - 1) / 2)` of the science cube and
  on the header's `STARX`, `STARY` of the template's larger grid.
* 0.0345"/spaxel and the 180 deg are the reduction's assumptions, not calibrated on the
  night.  A companion's position must not be used to calibrate either.
* The registered frames carry NaN along their edges (the rows and columns the registration
  shifted in).  The science frames' high-pass is NaN-aware here (`KLIPReducer` default
  `hp_nan_aware`, True from `ales.make_reducer`): IDL's `smooth` without `/nan` carries a NaN
  along the whole frame, which empties every bin holding such a frame.
* The metric does not de-spike (`declip_nsig=None`): ALES samples the PSF at 2.1-3.0
  spaxels per FWHM, and the 6 sigma single-pixel de-spiking takes a companion's peak for a
  hot pixel.
* The template is centred on the header's `STARX`, `STARY` (`meta['psf_center']`, which
  `generic.make_reducer` honours).  The centroid of its peak lands within a small fraction of a
  spaxel of it on every group, and the 3-spaxel flux unit differs little through the
  whole-pixel aperture.

## 6. Running it

```
klip-tpe near --instrument ales --root PRODUCTS --ales-target NAME --ales-cache CACHE \
    --known RHO PA --forbidden-pa PA 25 \
    --ann-edges 3 6.5 11 16 22 27 --n-iter 300 --n-init 50 --n-remeasure 3 \
    --use-contrast C1 C2 C3 C4 C5 \
    --run-dir RUN_DIR --workers 16
```

* `--use-contrast` forces the injected contrast per annulus (0 calibrates it to S/N 5 as
  usual); the reduction's own 5 sigma curve is the place to take the values from.
* `--root` is the reduction's products directory and is only read.  `--ales-cache` holds
  the group cubes as `.npy`, so later runs skip reading the full cube.  `--run-dir` is where
  everything else goes (default `./runs/run_...`).
* `--known RHO PA` keeps a companion's 1.5 FWHM out of every noise ring and the injections
  away from it; `--forbidden-pa PA HALFWIDTH` keeps the injections out of a wedge around
  it.  Say so in the products.
* `--ales-groups`, `--ales-channels C0 C1`, `--ales-only gNN ...`, `--ales-max-excluded`,
  `--ales-prefer-um`, `--ales-representative`, `--ales-selection {binary,two_slot}`,
  `--ales-ref NAME`, `--ales-tag-r-in`: `docs/CLI.md`.
* `--ales-ref NAME --ales-reflib` searches the reference star as a library (section 3), with
  `--ales-ref-bin N` (5) and `--ales-reflib-science {all,searched}` (all).  `--ales-sdi`
  adds the other groups' rescaled frames as a pool (section 4), with `--ales-sdi-cap N` (100)
  and `--ales-sdi-lenders {all,selected}` (all); with `--ales-ref` the reference star is the
  third pool.
* `--bin-min N` floors the temporal bin.  A group at bin 1 costs many times a bin-5 one.  On
  a machine with many cores it is affordable, on two it is not.  `--bin-max N` is the top of
  the searched bin range (30 by default here): a winner sitting on it (`distance_to_bounds`
  bin 0 in `final_results.json`) wants it higher.
* The search space's seed is the ALES reduction's baseline (`ales.BASELINE`): bin 5,
  filter 13, k 5, angsep 0.77 lambda/D, anglemax 45, no frame cuts.  `--n-sources`,
  `--n-remeasure`, `--min-r-fwhm` and the rest are as for any run.
* `klip-tpe resume --run-dir <dir>` continues a run; the ALES options are restored from the
  run's record.  `klip-tpe plots --run-dir <dir>` draws the figures and books again.

From Python:

```python
from klip_tpe import CalibrationConfig, RunConfig, Runner
from klip_tpe.instruments import ales
datasets, lam_m, fwhm_px, table = ales.load_ales(PRODUCTS, NAME, cache=CACHE)
red = ales.make_reducer(datasets, lam_m, fwhm_px, max_workers=16, table=table)
# with a reference star as a searched library: load_ales(..., ref=REF, ref_bin=5, cache=CACHE)
# and make_reducer(..., reflib=True); make_space then adds nkeep_psfref.  ADI, SDI and RDI in one
# step: make_reducer(..., sdi=True, sdi_cap=100, sdi_lenders="all"), which adds nkeep_sdi
space = ales.make_space(red)
space.project = ales.make_guard(red, max_excluded=10)
objective, sampler = ales.default_config(red, known=[(RHO, PA)], forbidden_pa=[(PA, 25.0)])
cfg = RunConfig(ann_edges=[6.5, 11], n_iter=300, n_init=50, n_remeasure=3,
                calibration=CalibrationConfig(forced=[C]))         # the injected contrast
Runner(red, space, objective, sampler, cfg, RUN_DIR).run()
```

A single reduction outside the runner, for checks: `cfg = space.decode(space.default_vector({...}))`,
add `inrad`, `outrad`, `pad_in`, `pad_out` to `cfg.params` and each `cfg.per_partition[pid]`,
then `red.reduce_config(cfg)` gives `EvalImages` with `.image`, `.stack` (one image per group)
and `.meta` per group.

## 7. Several nights: a data set in the release layout

klip-tpe treats a night as a partition, and a wavelength group as one too, so two nights of
ALES data are 2 x 20 partitions.  `ales.load_release` reads a data set in the layout that
`datasets.fetch("lbti_ales_mwc758")` downloads (the MWC 758 data of tutorial 7):

* per night, a meta file `<name>_<date>_meta.fits`: one template per group in its primary
  HDU, then `ANGLES` (LBT_PARA per frame), `FRAMES` (raw file, UT, LBT_PARA and airmass per
  frame), `GROUPS` (name, edges, mean wavelength and channels per group) and `WAVES` (the
  night's channel wavelengths).
* per night and group, a cube `<name>_<date>_gNN.fits`, `(frame, y, x)`, the mean of the
  group's channels.

The nights share one group grid (`GRIDLO`, `GRIDHI` and `NGROUP` in the meta header), so
`g07` is one band on both nights, while its mean wavelength follows each night's own channels.
The two MWC 758 nights, 2019 Jan 5 and Nov 14, have different cube-builder wavelength
solutions (99 and 98 channels, from 2.738 and 2.789 um).  Their grid, 20 groups over
2.90-4.15 um, puts 2 to 6 channels in each group on both nights.

```python
from klip_tpe import datasets
from klip_tpe.instruments import ales
files = datasets.fetch("lbti_ales_mwc758")                  # 670 MB, downloaded once
ds, lam_m, fwhm_px, table = ales.load_release(files)         # n1_g00 ... n2_g19
red = ales.make_reducer(ds, lam_m, fwhm_px, table=table)
space = ales.make_space(red)                                 # one block, 40 flags
space.project = ales.make_guard(red, max_excluded=20, bin_min=5)
```

* The partitions are `<night>_<group>`.  The nights are `n1`, `n2`, ... in date order
  (`labels=` renames them), and a night loaded alone keeps its label:
  `load_release(files, nights=["n2"])` gives `n2_g00` to `n2_g19`.  `only=["g15", ...]` keeps
  the same groups on every night.
* Each partition has its night's angles, its night's template (the star at the array center)
  and its night's frame tags, so a contrast is a flux ratio to the star in that group on that
  night.
* One parameter block is shared by every night and group, and there is one flag per
  partition.  The search drops a night group by group, so `max_excluded` must be at least 20
  for it to drop a whole night of 20 groups.
* An SDI pool (`sdi=True`) would lend one night's frames to the other night's groups, and
  `make_reducer` refuses it on more than one night.  Load one night for SDI.
* `datasets.fetch(..., roles=[...])` downloads part of a data set, such as a few groups.  The
  roles are `meta_<date>` and `gNN_<date>`.
* A third MWC 758 night, 2019 Nov 13, is its own data set, `lbti_ales_mwc758_20191113`, in the
  same layout and group grid.  `load_release({**files, **datasets.fetch("lbti_ales_mwc758_20191113")})`
  loads the three nights, `n1` to `n3` in date order.
* On the command line, `klip-tpe near --instrument ales --ales-dataset lbti_ales_mwc758`
  replaces `--root`, and `--ales-nights n2` (or a date) loads one night.  `klip-tpe resume`
  restores both options.

## 8. A companion's colours, and the paper reduction's ADI step

The search scores injected sources, so it tunes for the sources it injects.  By default they
have the star's spectrum: one contrast in every group.  A companion much redder than its star,
such as MWC 758c, puts most of its flux in the red groups, and a configuration tuned on
star-coloured sources need not suit it.  `make_reducer(..., spectrum=...)` gives the injected
sources a spectrum (`PartitionedReducer.set_injection_spectrum`):

```python
red = ales.make_reducer(ds, lam_m, fwhm_px, table=table,
                        spectrum=lambda lam_um: (lam_um / 3.8) ** 6.76,   # relative contrast to the star
                        spectrum_norm=3.8e-6, spectrum_label="MWC 758c power law")
```

* A source of contrast `c` goes into each group at `c` times the spectrum there, over the
  spectrum at `spectrum_norm`: the representative group by default, another partition id, or
  a wavelength in metres.  The run's calibrated contrast and its contrast curves are then the
  contrast at that wavelength, for a source of that spectrum.
* The spectrum is a callable of the wavelength in microns, a `(lambda_um, values)` table
  (linear in wavelength, held at its end values outside it) or `{partition: factor}`.  Every
  injection of the run gets it: the search's, the calibration's and the validation's, the
  KLIP-FM sources and the frames an SDI pool borrows.  A negative source, a known companion
  taken out of the frames (`RunConfig.subtract_known`), keeps its contrast.  The factors are in
  the setup file, under `reducer.injection_spectrum`.

Two steps of the built-in KLIP that the production space leaves at their reference values can
be searched, which the ALES reduction of MWC 758 in Wagner et al. (2023) needs:

* `make_space(..., zone_range=(0, 8))` adds `zone_in` and `zone_out`: the KLIP zone reaches
  that many spaxels inside and outside the annulus the run scores, so the basis is fitted over
  a wider ring.  `angsep`'s lambda/D of arc and the bin's PA span stay the annulus', where the
  sources are.  The frames' size caps the outer edge (27 spaxels on the 62-spaxel MWC 758
  frames).
* `make_space(..., search_post_hp=True)` adds `post_hp`: 1 high-passes the KLIP residuals again,
  with the same box, before they are derotated (the reference chain), and 0 derotates them as
  KLIP leaves them, as the paper's ADI images do.

The paper's ADI step for 2019 Nov 14 in these terms, in the 0.4-0.8" annulus: `bin` 15,
`filter` 11, `k_klip` 15, `n_ang` 6, `angsep` 0.01, `anglemax` at its top, no frame cuts,
`zone_in` 6.6, `zone_out` 3.8 (one 5-27 spaxel zone) and `post_hp` 0, on the 3.4-4.15 um
groups.

To keep the paper's processing geometry rather than search it, hold the zone and its sectors
and draw them as its IDL did:

```python
from klip_tpe.reducer import set_zone_center
E0, E1 = 0.4 / red.pxscale, 0.8 / red.pxscale       # the annulus the run scores, 11.6-23.2 spaxels
set_zone_center(red, (30, 30), "getzone2015")       # the 2015 getzone: about (sc/2 - 1, sc/2 - 1)
space = ales.make_space(red, search_post_hp=True,
                        pin={"zone_in": E0 - 5.0, "zone_out": 27.0 - E1, "n_ang": 6})
```

`pin` takes those parameters out of the search and gives every evaluation the values given.
klip-tpe draws its zones about `(nx / 2, ny / 2)` = (31, 31) on these 62-spaxel frames, and
counts sector angles from +x; the 2015 `getzone` of the paper's reduction draws them about
(30, 30) and counts from -x (atan2 + 180).  The star is at (30.5, 30.5) in both, and so is the
centre of the derotation.  With `set_zone_center(red, (30, 30), "getzone2015")` the 5-27 spaxel
zone holds the paper's 2220 spaxels and each of the six sectors the paper's own.

It still differs from the paper in two ways: KLIP runs on group means of 2 to 6
channels rather than on each channel, and the paper's ASDI images follow the ADI step with a
separate SDI step, where `sdi=True` is one KLIP step with an SDI pool (section 4).
`scripts/run_mwc758_ales_clike.py` is tutorial 7's search with c's spectrum and that
configuration as its seed.  By default it holds the paper's geometry (`--geometry paper`);
`--geometry searched` searches `zone_in`, `zone_out` and `n_ang` instead.  Its two sources go at the 0.4-0.8"
annulus' area-weighted mid radius, 0.632"; `--inject-rho 0.6` (`RunConfig.inject_rho`) puts
every injection of every stage at c's own separation instead, with the annulus as it was.

## 9. A disk of any shape out of the noise and the injections

A disk at a known position angle, seen near edge-on, is kept out of the metric's noise samples
by `pixel_mask` (`metrics.pa_wedge_mask`) and out of the injections by `forbidden_pa`.  Spiral
arms or a ring follow no wedge.  `metrics.disk_mask_from_image` masks the bright extended
structure of an image of the disk, and `default_config(..., disk_mask=mask)` hands that one
mask to both halves: it joins the metric's `pixel_mask` and becomes the sampler's
`forbidden_mask`, so no injection lands on it (`disk_mask_clear_fwhm` keeps them further off).

```python
from klip_tpe.metrics import disk_mask_from_image
mask = disk_mask_from_image(disk_img, fwhm_px=2.8, nsig=2.0, r_range=(6, 26),
                            protect=[(0.603, 228.5)], pxscale=red.pxscale)   # c stays unmasked
objective, sampler = ales.default_config(red, known=[(0.603, 228.5)], disk_mask=mask)
```

The image is smoothed by a FWHM, cut at `nsig` robust sigma, pieces under `min_beams` beams are
dropped, and the rest is grown by `grow_fwhm` FWHM, because the metric samples the
matched-filtered image, which carries the disk about a FWHM past its edge.  `nsig_neg` adds the
disk's ADI negatives, the self-subtraction troughs beside its emission: pieces `nsig_neg` robust
sigma below the median that come within `neg_reach_fwhm` FWHM of the bright structure
(`make_disk_mask.py --neg-nsig 2`).
`scripts/make_disk_mask.py IMAGE --fwhm PX --out MASK.fits` builds one from any FITS image and
draws it for a look, and `run_mwc758_ales_clike.py --disk-mask MASK.fits` runs with it.  The
setup file records the masked pixel counts (`objective.metric.pixel_mask_px`,
`sampler.forbidden_mask_px`).

Build the mask from an image of the disk made before the search (a disk-preserving reduction,
another epoch or instrument), never from a search's winner, and look at it: it masks whatever
is bright, and a bright speckle group masked with the disk biases the noise estimate down.  It
also only matters where the disk crosses the injections' ring.  On MWC 758 the arms lie inside
0.6", and a mask of them covers about 2% of the noise band at 0.632", where the 0.4-0.8" runs
inject.

## 10. angsep = 0 is another algorithm, and k trades the disk for point sources

`angsep` (lambda/D of arc at the annulus) is the least rotation between a target frame and its
references, and any positive value leaves the target out of its own basis.  `angsep = 0` with
`anglemax` at the sequence's PA span does not: `klip.klip_annular` then takes the IDL's auto-fast
shortcut (`reduce_near_2.pro`) and builds one basis per zone from all the frames, each target
frame inside its own.  A spiral arm, which rotates along itself and so barely moves from frame
to frame, goes into that basis and is subtracted with the star.

A search finds the corner.  `Param.sanitize` clips a proposal past a bound onto it and the
reference-count guard projects onto `angsep = 0`, so the bound carries real probability, and
the objective scores the corner well.  Every c-tuned MWC 758 winner sat there.  On Nov 14,
g15-g19, with c's spectrum, the injections at 0.6" and the arms-and-negatives mask (8 paired
draws; arm flux = the mean over the arms of the image smoothed by lambda/D, over the same for
the paper's ADI step; c's S/N with the disk masked):

| configuration                              | injections' S/N at 2e-5 | at 4e-5 | arm flux | c's S/N |
|--------------------------------------------|------:|------:|-----:|-----:|
| the paper's ADI step                       | 1.14 | 2.24 | 1.00 | 2.48 |
| the paper's with the 4e-5 winner's frame cuts | 1.64 | 3.03 | 1.01 | 3.63 |
| the 4e-5 winner (angsep 0, 18 modes, 13-px box) | 2.63 | 4.73 | 0.18 | 2.37 |
| the winner at angsep 0.01                  | 2.01 | 3.75 | 0.55 | 2.92 |
| the winner at angsep 0.01 with 15 modes    | 1.65 | 3.07 | 0.97 | 3.36 |

Off the corner, k makes the same trade: at the paper's step with those frame cuts the arm flux
is 1.01 at 15 modes, 0.84 at 16, 0.66 at 18, 0.57 at 20 and 0.38 at 25, while the injections
gain 22 % from 15 to 18 modes.  (Below 15 modes the arm flux rises past the paper's, 1.7 at
12 and 4.1 at 10, with starlight the basis no longer removes.)  The injections sit off the
disk and are scored as point sources, so the arms' loss never reaches the objective, and c
gains from neither trade (though its S/N moves by about this much with any one setting, as
the empty positions of its ring do).  Halving the contrast leaves the ranking as it is.

`make_space(..., angsep_range=(0.01, 3.0), k_klip_max=15)` keeps the search on the disk's side
of both (`run_mwc758_ales_clike.py --angsep-min 0.01 --k-max 15`).  It does not keep the disk.
With both closed, at 2e-5, the search took a third route: a noise cut at 1.12 times the mean,
which drops a third of the frames, most of them from the start and the end of the sequence, and
leaves 0.41 of the arm flux (14 modes, a 17-px box, the residual high-pass on).  At that
winner's other settings the arm flux is 1.09 with no cuts, 0.84 with the cut at 1.66, 0.71 at
1.4 and 0.41 at 1.12, and the injections' S/N at 2e-5 is 1.15, 1.60, 1.59 and 2.06 (the paper's
step 1.14): the cut is the whole of the gain.  A cap per route does not end:
an objective that scores point sources alone trades the disk for them wherever it can, so a
search for a planet in a disk has to score the disk too.
