# LBTI/ALES integral-field cubes

klip-tpe reads the products of an ALES reduction (a `(wavelength, frame, y, x)` cube, its
parallactic angles, its wavelengths and a per-channel PSF template) and optimizes the
KLIP/ADI post-processing over the usual parameters plus one on/off flag per wavelength
group.  The first data set was WISPIT 2 (UT 2026-06-05, DX aperture, 535 frames x 99
channels of 62 x 62 spaxels); the report on that run lives with the data
(`WISPIT2_ALES/klip_tpe/REPLY_2026-10-14_ALES_WISPIT2.md`).

## 1. The cube as wavelength groups

ALES delivers 99 channels at R ~ 40-55 over 2.76-4.29 um.  Neighbouring channels share a
resolution element, so a channel is not the unit to switch on and off.  The adapter cuts
the usable channels (6-88, 2.899-4.175 um) into `ngrp` groups of equal width in `ln lambda`,
about one resolution element each at `ngrp = 15` and 0.73 at the default 20, and means each
group's channels into one `(frame, y, x)` cube.  Within a group the speckle pattern scales
by at most 2 % in radius, 0.2 spaxels at 0.3", far below the FWHM, so the mean loses
nothing and the reduction costs 1/(channels per group) of a per-channel one.

The rule (`ales.group_table`): a channel's edges are the midpoints to its neighbours, the
group edges are equally spaced in `ln lambda` from the lower edge of channel `c0` to the
upper edge of `c1`, and a channel belongs to the group whose edges contain its wavelength.
On WISPIT 2's wavelengths the 20 groups are g00 = channels 6-7 (2.910 um) to g19 = channels
83-88 (4.145 um), 2 to 6 channels each.  The realised table goes into the run's
`run_setup.json` under `reducer.provenance.groups`, and each group's channels and
wavelengths under `reducer.partitions.gNN.frames`.

Each group is one partition of a `PartitionedReducer`, like a night or a roll:

* its own wavelength `lam_m` and FWHM `1.03 lambda/D` (2.13 spaxels at 2.91 um, 3.04 at
  4.15), so `angsep` in lambda/D and the Mawet apertures are right for it;
* its own PSF template, the group mean of the per-channel template, normalised in 3
  spaxels (`ee_radius_px`).  The template is the median of the registered science frames,
  in the frames' units, so `flux_unit` is the star's flux in that group and an injected
  `contrast` is a flux ratio to the star there;
* frame-quality tags from its own frames (`quality_tags`, `tag_r_in = 2`: from 6 spaxels
  out the frames are background-dominated and the correlations fall to 0.14-0.50);
* optionally the group mean of a reference star's cube as its RDI library (`ref=`,
  `--ales-ref`), off unless asked.

One parameter block is shared by every group (`bin`, `n_ang`, `filter`, `angsep`,
`anglemax`, the three frame-selection thresholds, `k_klip`), and the group at the mean
wavelength (g10, 3.512 um, on WISPIT 2) stands for the combined image: the metric's FWHM
and the sampler's floor are its (`PartitionedReducer.representative`).  The groups are
combined with equal weights.

## 2. The group flags

`selection='binary'` adds one 0/1 dimension per group, `sel_g00 .. sel_g19`.  A random
draw includes each group with probability 0.75, a proposal may switch any number off, and
an all-off draw decodes to all on.  `GroupCountGuard` (first in `ales.make_guard`) bounds
it: a proposal with more than `max_excluded` (10) groups off has groups switched back on,
nearest `prefer_um` (3.7 um, where the companion's S/N per group peaks) first, or at
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
  the products (248 HIP 91748 frames in blocks of 78, 78 and 92 make 15 + 15 + 18 = 48 means,
  a block's last frames unused), the whole sequence is one block without the file.  The cache
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
  62 x 62 frame).  At bin 5 on WISPIT 2 that rescaling is 20 x 19 x 108 frames per reduction,
  which on two cores is most of the cost of the mode, and on 32 threads about a second per
  reduction.  `--bin-min 5` keeps the search off the bins where it would be tens of seconds.
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
  shifted in, all beyond 27 spaxels on WISPIT 2).  The science frames' high-pass is
  NaN-aware here (`KLIPReducer` default `hp_nan_aware`, True from `ales.make_reducer`):
  IDL's `smooth` without `/nan` carries a NaN along the whole frame, and at bin 5 it had
  emptied 40 of WISPIT 2's 108 bins.
* The metric does not de-spike (`declip_nsig=None`): ALES samples the PSF at 2.1-3.0
  spaxels per FWHM, and the 6 sigma single-pixel de-spiking took WISPIT 2b's peak for a hot
  pixel (matched-filter S/N 6.2 to 3.9).
* The template is centred on the header's `STARX`, `STARY` (`meta['psf_center']`, which
  `generic.make_reducer` honours); the centroid of its peak lands within 0.05 spaxel of it on
  every group, and the 3-spaxel flux unit then differs by up to 1.5 % through the whole-pixel
  aperture.

## 6. Running it

```
klip-tpe near --instrument ales --root /Volumes/RAID36TB/LMIRCam/WISPIT2_ALES/reduction/products \
    --ales-target WISPIT2 --ales-cache /Volumes/RAID36TB/LMIRCam/WISPIT2_ALES/klip_tpe/cache \
    --known 0.29888 241.46 --forbidden-pa 241.46 25 \
    --ann-edges 3 6.5 11 16 22 27 --n-iter 300 --n-init 50 --n-remeasure 3 \
    --use-contrast C1 C2 C3 C4 C5 \
    --run-dir /Volumes/RAID36TB/LMIRCam/WISPIT2_ALES/klip_tpe/run_YYYYMMDD_HHMMSS --workers 16
```

* `--use-contrast` forces the injected contrast per annulus (0 calibrates it to S/N 5 as
  usual); the reduction's own 5 sigma curve is the place to take the values from.
* `--root` is the reduction's products directory and is only read; `--ales-cache` holds
  the 20 group cubes as `.npy` (157 MB) so later runs skip the 814 MB cube; `--run-dir` is
  where everything else goes (default `./runs/run_...`).
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
* `--bin-min N` floors the temporal bin.  A 535-frame group at bin 1 costs 30 times a bin-5
  one (97 s against 3 s for 20 groups on two cores); on a machine with many cores it is
  affordable, on two it is not.  `--bin-max N` is the top of the searched bin range (30 by
  default here): a winner sitting on it (`distance_to_bounds` bin 0 in `final_results.json`)
  wants it higher.
* The search space's seed is the ALES reduction's baseline (`ales.BASELINE`): bin 5,
  filter 13, k 5, angsep 0.77 lambda/D, anglemax 45, no frame cuts.  `--n-sources`,
  `--n-remeasure`, `--min-r-fwhm` and the rest are as for any run.
* `klip-tpe resume --run-dir <dir>` continues a run; the ALES options are restored from the
  run's record.  `klip-tpe plots --run-dir <dir>` draws the figures and books again.

From Python:

```python
from klip_tpe import CalibrationConfig, RunConfig, Runner
from klip_tpe.instruments import ales
datasets, lam_m, fwhm_px, table = ales.load_ales(PRODUCTS, "WISPIT2", cache=CACHE)
red = ales.make_reducer(datasets, lam_m, fwhm_px, max_workers=16, table=table)
# with a reference star as a searched library: load_ales(..., ref="HIP91748", ref_bin=5, cache=CACHE)
# and make_reducer(..., reflib=True); make_space then adds nkeep_psfref.  ADI, SDI and RDI in one
# step: make_reducer(..., sdi=True, sdi_cap=100, sdi_lenders="all"), which adds nkeep_sdi
space = ales.make_space(red)
space.project = ales.make_guard(red, max_excluded=10)
objective, sampler = ales.default_config(red, known=[ales.WISPIT2B], forbidden_pa=[(ales.WISPIT2B[1], 25.0)])
cfg = RunConfig(ann_edges=[6.5, 11], n_iter=300, n_init=50, n_remeasure=3,
                calibration=CalibrationConfig(forced=[C]))         # the injected contrast
Runner(red, space, objective, sampler, cfg, RUN_DIR).run()
```

A single reduction outside the runner, for checks: `cfg = space.decode(space.default_vector({...}))`,
add `inrad`, `outrad`, `pad_in`, `pad_out` to `cfg.params` and each `cfg.per_partition[pid]`,
then `red.reduce_config(cfg)` gives `EvalImages` with `.image`, `.stack` (one image per group)
and `.meta` per group.
