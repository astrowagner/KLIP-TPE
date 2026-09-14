# LBTI/NOMIC with pyNOMIC

klip-tpe reads the products of a **pyNOMIC** reduction directly — the directory where the
pyNOMIC notebook ran — and optimizes the KLIP/ADI post-processing of those frames.  This
page is the start-to-finish guide for pyNOMIC users.

## 1. What klip-tpe needs from pyNOMIC

Run the pyNOMIC notebook (`singlesidedreduction.ipynb` or `doublesidedreduction.ipynb`)
at least through **frame evaluation**.  klip-tpe then reads, from `<workdir>`:

| file | used for |
|---|---|
| `<obj>_NOMIC_chop_correction.npz` | `files`, `chops`, `header_info` (`header_info[1]` = parallactic angle, `[2]` = JD) |
| `<masked|aligned>/*.fits` | the registered, background-subtracted frames (sorted) |
| `<obj>_NOMIC_evaluated.npz` | `correlations`, `background_dev`, `residual_dev` → frame-quality tags; `reffits` → the per-frame Airy fit used as **injection PSF** |
| `<obj>_NOMIC_binned_evaluated.npz` | the same for pyNOMIC's temporal bins (`--binned`) |
| `<obj>_NOMIC_psf_subtraction.npz` / `<obj>_NOMIC_aligned.npz` | star positions for the image-group split (`--groups auto`), optional |

Nothing is written into the pyNOMIC directory.

## 2. First run

```
pip install "klip-tpe[plots]"
klip-tpe near --instrument nomic --root /data/procyon_reduction --obj procyon \
    --run-dir runs/procyon_ann1 --ann-edges 20 45 --n-iter 300 --n-init 50 \
    --n-top 2 --n-valid 4 --show
```

* `--ann-edges 20 45` — the annulus in pixels (0.0179″/px → 0.36–0.81″).  Several edges
  make several annuli, optimized one after the other and stitched at the end.
* `--n-iter / --n-init` — evaluations per annulus / random warm-up.  300 is a smoke test;
  production runs use thousands (the NEAR protocol: 500 warm-up + 10 000 for the first
  annulus, 2500 for the others).
* `--n-top / --n-valid` — validation: the best `n_top` search configurations are
  re-scored on `n_valid` fresh injection sets; the winner is the best validation median.
* `--show` opens the live window (`--show inline` inside Jupyter); `--workers auto`
  (default) uses every core.
* `--binned` uses pyNOMIC's own temporal bins instead of the unbinned frames;
  `--pre-bin 20` mean-bins every 20 frames at load time (memory); `--max-frames N` for tests.

Products land in `runs/procyon_ann1/` (see `docs/DISPLAY.md`); interrupt at any time and
continue with

```
klip-tpe resume --instrument nomic --root /data/procyon_reduction --obj procyon --run-dir runs/procyon_ann1
```

## 3. Partitions: chop states, nights, image groups

Every *partition* gets its own KLIP basis and its own parameter block (`bin, n_ang,
filter, angsep, anglemax, corr_thresh, noise_max, coronoise_max, k_klip`), and the
optimizer searches which partitions to combine through two selection slots
(`drop1/drop2`: 0 = drop nothing, c = drop partition c).

* **Chop states** are always partitions: `procyonA`, `procyonB`.
* **Nights / objects**: several `--obj` (and `--root`, one per object) give several
  nights: `--obj procyon_n1 procyon_n2 --root /data/n1 /data/n2`.
* **Image groups**: pyNOMIC's notebooks split a sequence into *image groups* — contiguous
  time segments between jumps of the star position (nods, pointing offsets,
  `hf.image_groups(header_info[2], maxima[:, 0])`) — and drop the shifted ones by hand.
  klip-tpe can make each group a partition and let the optimizer decide:

  ```
  --groups auto                       # pyNOMIC's split (port of hf.image_groups), star positions from
                                      #   <obj>_NOMIC_psf_subtraction.npz 'maxima' or <obj>_NOMIC_aligned.npz
  --groups 2460345.61 2460345.68      # split at these JDs
  --groups @labels.txt                # one integer label per frame
  --group-min-frames 20               # smaller groups are left out
  --group-smooth 100                  # smoothing kernel (frames) of the auto split
  ```

  Partitions are then `procyonAg1, procyonAg2, …, procyonBg1, …`.  Nights and groups are
  treated identically — they are all partitions — so `nights × chops × groups` simply
  gives more of them; raise `--max-drop 3` (or more) so the selection can drop several.

`--global-block` tunes one parameter set for all partitions (fewer dimensions);
`--no-selection` disables the drop slots.

## 4. Injection PSF and contrast

The injection model is pyNOMIC's own: frame *j* receives `airy_disk(c · amp_j, σx_j, σy_j,
p_j)` from `reffits[j]` — exactly what pyNOMIC's `inject_source` does — so a contrast in
klip-tpe means the same thing as in pyNOMIC.  `--psf frame` uses each frame's own
(unsaturated) star as an empirical template instead; `--psf gaussian` is for smoke tests.

The injection contrast is **calibrated** per annulus so that the default configuration
detects the fakes at S/N ≈ 5 (`--use-contrast c` forces it; `--cmax-cal` caps it).
Contrast curves (`annulusNN/contrast_curve.txt`, 5σ, throughput from the injections) are
in the same units.

## 5. Checks to make on a new data set

1. **Parallactic-angle sign.**  klip-tpe derotates counter-clockwise by
   `parang_sign · header_info[1] + truenorth`.  Reduce once with a bright injected
   companion (or a known binary) and confirm it lands at the requested PA; if it is
   mirrored, `--parang-sign -1`; a fixed offset goes into `--truenorth`.
2. **FWHM.**  Default = 1.028 × the median Airy fit's σ; `--fwhm-px` overrides.  Compare
   with a Gaussian fit of the flux frames.
3. **Memory.**  Unbinned NOMIC sequences are large (150 × 150 px × many thousand frames per
   chop); `--pre-bin`, `--binned` or `--max-frames` keep the cube in RAM.  The forked
   workers share one copy of the data.
4. **The annulus.**  Start inside the well-corrected region and away from the edge of the
   150 px crop (`--crop-half 75`); `--ann-edges 10 30 50 70` gives three annuli.

## 6. Python instead of the command line

```python
from klip_tpe import Runner, RunConfig, ValidationConfig
from klip_tpe.instruments import nomic
from klip_tpe.display import LiveDisplay

ds = nomic.load_pynomic("/data/procyon_reduction", "procyon", groups="auto", group_min_frames=50)
red = nomic.make_reducer(ds, max_workers="auto")
space = nomic.make_space(red, max_drop=3)
space.project = nomic.make_guard(red)
objective, sampler = nomic.default_config(red, known=[(0.9, 210.0)])     # a known companion to avoid
cfg = RunConfig(ann_edges=[20, 45], n_iter=300, n_init=50, validation=ValidationConfig(n_top=2, n_valid=4))
Runner(red, space, objective, sampler, cfg, "runs/procyon", callbacks=[LiveDisplay("runs/procyon", show="auto")]).run()
```

`nomic.datasets_from_arrays(files, chops, angles, correlations, background_dev,
residual_dev, reffits, ...)` builds the datasets from arrays you already have in a
notebook session, without the `.npz` files.

## 7. Comparing with pyNOMIC's own reduction

The winner's parameters are in `annulusNN/winner.json` (`winner_config`) and
`results.txt`; `annulusNN/best_clean.fits` is the winner's image without injections,
`klip_stitched.fits` the final stitch over annuli, `klip_stitched_snr.fits` its S/N map.
`--backend vip` or `--backend pyklip` reduce with those engines instead of the built-in
KLIP for a like-for-like comparison on the same frames and the same injections.
