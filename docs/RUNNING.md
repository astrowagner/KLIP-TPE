# Running the NEAR production optimisation

This is the IDL `optimize_near_2_tpe` production protocol (the NEAR/VISIR campaign: 6
nights, 3 annuli, 500 warm-up + 10 000 / 2500 / 2500 evaluations) driven from a terminal or
a Jupyter notebook.  Products land in `$ROOT/comb/opt/run_YYYYMMDD_HHMMSS/` — the same
layout as the IDL runs.  The NEAR data themselves are not public; the same commands apply
to any other data through `--instrument nomic | generic` (see `docs/PYNOMIC.md` and the
tutorials).

## Install (once)

```
cd klip-tpe
python3 -m pip install -e ".[plots]"
python3 -m pip install jupyter
klip-tpe --help
```

(`[plots]` pulls in numpy, scipy, astropy, threadpoolctl, matplotlib, pillow, imageio; jupyter is
only needed for the notebook.  zsh treats a trailing `# comment` as arguments, so keep the
lines bare.)  Python ≥ 3.9.  `ffmpeg` (e.g. `brew install ffmpeg`) is optional and only adds the `.mp4`
next to the `.gif` progress movie.

## Terminal

```
ROOT=/path/to/NEAR_py scripts/run_near2_production.sh
```

Variants: `WORKERS=8 scripts/run_near2_production.sh` pins the thread budget (`WORKERS=-2` leaves
two cores free); `SHOW=1 scripts/run_near2_production.sh` also opens the live panel window;
`ALIENS=1 scripts/run_near2_production.sh` adds IDL's launch movie to the calibration phase (`--aliens`); `scripts/run_near2_production.sh resume` continues the newest run under `$ROOT/comb/opt` after an interruption (`resume <run dir>` or `resume run_YYYYMMDD_HHMMSS` picks a specific one).

The script is the following command (edit freely):

```
klip-tpe near --root $ROOT --nights 1 2 3 4 5 6 \
    --ann-edges 0 20 40 60 --n-iter 10000 2500 2500 --n-init 500 500 500 \
    --use-contrast 6e-5 0 0 --cmax-cal 6e-5 --blocks univariate --n-top 3 --n-valid 8 --k-max 100 \
    --verify --param-verify --candidates --stitch-every 10 --pdf-every 10 --workers auto
```

`--run-dir` defaults to `<root>/comb/opt/run_YYYYMMDD_HHMMSS`; `--seed N` fixes the random
stream (IDL used 1789071027 — the streams are not interchangeable, only the protocol is).
Everything the IDL run writes has a counterpart (`docs/DISPLAY.md`).

Before its first evaluation a new run moves each searched reduction dimension alone and
reduces annulus 1's clean image, about one reduction per dimension
(`klip_tpe.liveness.check_live_dimensions`).  A frame-selection threshold is moved to the cut
points of the frames' own tags, because it acts only between the smallest and largest tag.
A dimension that changes nothing is reported as a warning, naming it.  This is the class of
problem `docs/IDL_FINDINGS.md` §1 describes, which the search itself cannot see.
`--strict-liveness` stops the run there instead.  A dimension the guards never let move
alone is logged and the run goes on.  `--no-liveness-check` skips the check, and a resume
is not re-checked.

## Jupyter

`notebooks/near2_production_run.ipynb`: the same protocol built from the Python API
(`near.make_reducer / make_space / make_guard / default_config`, `RunConfig`, `Runner`,
`LiveDisplay`), started in a background thread so the notebook stays live, with a `watch()`
cell that shows the newest step panel and the log tail, and a resume cell.

## Cores

`--workers auto` (the default, `max_workers="auto"` in the API) gives the run one thread per
physical core the process can use.  On patrick (96 cores, 192 logical CPUs) a farm benchmark of
NEAR2's nights reduced them 1.25x faster with 48 jobs of 2 threads at a budget of 96 than with
48 jobs of 4 at 192, so the second hardware thread of a core (SMT) adds nothing on this code.  `--workers all`
uses every logical CPU, as `auto` did before 2026-10-27.  `--workers 8` pins the budget and
`--workers -2` leaves two cores free.  The budget is spent like IDL's bridges.  After the nights are loaded the reducer forks
`workers / threads-per-job` worker processes, and never fewer than `min(workers, 2 × nights)`.
They share the loaded cubes copy-on-write, and only requests and the small result images
cross between processes.  A partition job runs its injection, its per-target KLIP work, its
high-pass, binning, derotation and combination on its own threads, and injects only the frames
its frame selection keeps.  Everything before KLIP goes one bin group at a time: a group's kept
frames are copied out of the night, given their sources, high-passed and averaged while they
are in the processor's cache, so a job reads its kept frames from memory once and writes only
the binned cube.  `--threads-per-job` (2 by default, 4 before
2026-10-27) sets the average.  A job expected to take long, a big night at a small bin, gets
more threads, up to four times as many, and a light one gets one.  The long jobs run longest,
so the jobs running at a moment hold more threads than the budget, a load average of 133 on
patrick's budget of 96.  Its 192 logical CPUs absorb that, and it is the fastest layout
measured there.  Holding the jobs to the budget (`PartitionedReducer.cap_threads`) ran 1.65x slower,
since the heavy nights then ran on one thread each.  A free worker takes the waiting job
expected to take longest.  The images do not depend on the thread count or the order.  BLAS
is pinned to one thread per worker thread.  `--pool threads` keeps everything in one process.
It is slower, since Python's GIL limits the thread version to roughly one to two cores' worth
of work.

The runner keeps as many reductions in flight as the pool can serve, and takes their results
in order:

* A trial reduces its clean image and all `--n-remeasure` injected images in one round.  Each
  draw is scored as soon as its image and the clean image are back.
* Validation reduces a candidate's clean image and its trials several at a time, enough to
  fill the pool, and reports the trials in order.  The candidates' clean images come first, a
  batch at a time, so a copy of an earlier candidate is skipped before its trials go out.  The
  final round's trials (`--n-final`) and the elected winner's fresh draws go out together after
  the first round.
* param_verify, the calibration remeasurements and the k-scan pair work the same way.

Every draw's positions are drawn first, in the order a one-at-a-time loop draws them.  The
schedule changes when a reduction finishes, not what it returns, so a run writes the records a
one-at-a-time loop writes, and a resumed run continues exactly where it stopped.

A trial of N nights at `--n-remeasure 5` is 6N partition jobs.  On patrick `--workers auto` is
96, and the default 2 threads per job make 48 processes.  The startup log prints the plan:

```
parallel: 96 workers on 96 cores (192 logical CPUs): 48 worker processes x 2 thread(s) per job (1 to 8 by the job's expected cost, longest first), 6 partitions per reduction, so 8 reduction(s) at full width at once; pool = processes
```

`--batch B` keeps B trials in flight (default 1).  As soon as the oldest trial is in, the next
proposal is asked from the history so far and its reductions go out, before the finished
trial's files, display and checkpoint are written.  The trials still in flight count toward
the warm-up and the grid.  TPE proposes from a history a few trials old, so a batched search
follows a different trajectory from `--batch 1`, and the same seed gives the same trajectory
on any machine.  The checkpoint records the trials in flight, and a resume reduces them again
before it asks for anything new.  `--batch-mode sync` proposes each batch from one posterior
and waits for all of it before the next.  It checkpoints once a batch is in, so a run stopped
part way through a batch repeats that batch.  A run started before `--batch-mode` existed
resumes with `sync`, and the k-scan modes always use it.  `run_setup.txt` records both
settings.

With trials in flight, the time between two finished evaluations is the throughput, not the
time any of them takes.  The panel and each `run.log` evaluation line give a trial's own
times: `eval`, from its dispatch until it was processed (scored, files written, checkpoint,
display), and `reduce`, until the last of its reductions was back.  `results.jsonl` keeps
`reduce_s` in each record.  The panel's `1 done every ... (N in flight)` is the throughput,
and both ETAs come from it.

With several trials in flight, the main thread sets the pace once the workers keep up: it
scores every trial, asks for the next proposal and writes the files, the checkpoint and the
display.  `KLIP_TPE_PROFILE_MAIN=300 klip-tpe resume ...` writes `main_thread_profile.txt`
into the run directory every 300 s.  It lists the share of the time each function held the
main thread over the last 300 s and since the start, and the share it spent waiting.  It
needs no other package.

Six NEAR nights are ~2000 frames × 150² each, and the workers share them.  Each running job
holds its own binned cube and KL bases.  No worker holds injected frames.  A set of sources
reduced a second time on a night (param-verify's fixed set, a fixed-source search) keeps its
stamps there, about 70 MB a night for three sources on NEAR, one set at a time, so a later
selection places only the frames it adds.  On Linux the log reports, after the
search and after validation, the workers' peak resident memory and what they hold together,
which is what a given pool size needs on that machine.

### The per-target KLIP at small bins (`--klip-gram zone`)

At a small bin the KLIP step is most of a night job.  For every target the per-target path
forms the covariance of its references, `R Rᵀ`, from their pixels, decomposes it, and forms
all M modes, of which it keeps k.  `--klip-gram zone` forms one Gram matrix per zone instead,
before the targets, reads each target's covariance off it, and projects the target on its
first k modes without forming them.  On NEAR night 6 at bin 5, with annulus 5's zones and
k 45, the night's KLIP takes 3.5 s where it took 7.6 s on one thread, and 1 of 1.4 million
residual pixels differs, by one float32 unit.  It is the same KLIP to float64 rounding but
not to the bit, since the Gram's dot products are summed in another order, so `target` is the
default.  `temp_mean` keeps the per-target path in either mode.

A run keeps the mode it was started with.  `klip-tpe resume --klip-gram zone` switches a
stopped run from that point on and writes the switch, with its time, into `klip_gram.json`
in the run directory, so every later resume keeps it.  `--klip-gram target` switches it back.
The log says when a run uses `zone`.

## Edited frames (`--cube-name`, `--drop-frames`, `--subtract-sources`, `--inject-sources`)

These options change the frames a run reduces, and the run keeps them.

* `--cube-name NAME` reads another cube in each night's directory, such as a reduction's
  line-subtracted `AB_cube_skysub_cen_clean_abline.fits`.
* `--drop-frames NIGHT:INDEX[,INDEX...]` leaves frames out of a NEAR night at load, by their
  index in the night's cube counted from 0.  The option may repeat.  `--drop-frames n4:551`
  drops n4's last frame, whose parallactic angle is recorded as 12.07 deg where its neighbors
  are at 24.1 deg.  The frames go before anything else sees them, so the frame tags, the frame
  cut, the binning, the injections and KLIP-FM all work on the frames kept, and `texp` scales
  with them.  The files in the night's directory are not changed.  An index past the cube is
  refused, and so are a night the run does not load and a list that leaves no frame.  An index
  past `--max-frames` drops nothing, and the log says so.  The frame cut does not do this job.
  A configuration can set its correlation threshold below the frame's, and the cut stands
  down when it would leave fewer than bin x k frames.
* `--subtract-sources FILE` takes point sources out of every frame once, at load.  The file
  has one source per line: rho (arcsec), PA (deg) and contrast, with `#` comments.  Columns
  after the third are ignored.  The contrasts are the sources' own, as they are in the data,
  so a planet's or a line's are positive and the list goes in as it is.  Each source goes in
  with the negative of its contrast, through the code that injects a draw's sources, so its
  placement, true north, center, flux unit and throughput are a draw's.  A file whose
  contrasts are all negative is refused, since a list of the negatives to inject, written to
  take a line out by injection (`c = -a step / L`), would double the line.
* `--subtract-line FILE` does the same for a line through the star given by its nodes (below).
* `--inject-sources FILE` adds point sources to every frame once, at load, after any
  subtraction: planets put on a line, say, to see what its subtraction does to them.  The file
  is as for `--subtract-sources`, with the contrasts the sources will have in the data.

Every reduction then sees the edited frames: the clean pass, each draw, the reference ranking,
validation and param-verify.  The KLIP-FM model cube does not, since it is built from zeros.
A source of contrast 0 is skipped, so a list of zeros changes nothing, to the bit.  On NEAR
night 6, a 42-source chain subtracted at load gives the frames that an outside subtraction of
the same list with the same injection code gives, to the bit.  Sources added at load give the
frames of an outside injection, to the bit, in the tests.

The log gives the fraction of the sources' flux that left the frames, measured on 32 frames
per night against the sources injected into zeros.  float32 rounding keeps the part of a source
that sits below half a unit in the last place of the pixels it lands on, and a warning says so
when that is more than 0.1 % of the list's flux.  On NEAR nights 4 and 6 a 216-source line of
contrasts 1e-8 to 3e-7 left the frames to 1.3e-4 or better.

The frames go into the run's configuration (`RunConfig.data_edits`, in `run_setup.json` and
the checkpoint).  `klip-tpe resume` and `extend` read them from the checkpoint when the
options are not typed, and refuse a typed value that differs, since a resume cannot change the
frames its evaluations were made on.  A script that resumes a run with a reducer whose frames
differ gets the same refusal from the `Runner`.  From Python the options are
`near.make_reducer(..., cube_name=..., drop_frames={"n4": [551]}, subtract_sources=...,
inject_sources=...)`, or
`reducer.subtract_sources(...)` and `reducer.inject_sources(...)` on any KLIP reducer before
its worker processes start.  `klip-tpe rereduce` (below) reduces a finished run's winners on
other frames without a new search.

`RunConfig.subtract_known` is a different tool.  It adds the negatives to the injections of
every reduction, for a companion whose contrast was fitted, and the reference ranking does not
see them.

## A line through the star (`--forbidden-pa`, `--mask-line`, `klip-tpe fit-chain`)

For a feature fixed on the sky along one direction on both sides of the star, such as the NEAR
line through alpha Cen A along the A-B direction (PA 13 and 193 deg):

* `--forbidden-pa PA HALFWIDTH` keeps the injections out of a wedge, on `near` as on `generic`
  and `ales`.  Give it for PA and for PA + 180.
* `--mask-line PA HALFWIDTH` leaves the pixels within HALFWIDTH px of the line out of the noise
  apertures, on both sides of the star (`metrics.line_mask`).  As for any noise mask, choose it
  from what is known before the image is looked at.
* `klip-tpe fit-chain --run-dir RUN --chain FILE` fits the node values of a chain of point
  sources to a finished run's final image.

The chain file has one source per line, `rho pa contrast node`.  A node's lines are its basis,
each source at its contrast per unit node value, so a source shared by two nodes is on two
lines.  Without labels the chain is one node and the fit scales it.  For each node, its sources
go through every annulus' winner as the KLIP-FM model, and the FM images are stitched with the
final stitch's weights and tiles.  The fit takes the pixels within `--halfwidth` px of the line
(default 12) inside the run's annuli, or inside `--rho-range RMIN RMAX` (arcsec) within them,
weighted by 1/sigma(r)² from the image outside the strip.  `--mask RHO PA RADIUS` circles, the
run's known sources and, with `--mask-candidates A`, its candidates of that tier are left out.
A node that puts no signal in the fit region, one beyond the run's annuli say, is refused
rather than fitted to 0 ± 0.  The errors are scaled by the square root of the pixels per beam.
It writes, in `<run>_chainfit/` beside the run:

* `chainfit.json`: the node values, their errors and covariance, chi², the masks and the
  annuli,
* `chain_fitted.txt`: the fitted chain, ready for `--subtract-sources`,
* `chainfit_model.fits`, `chainfit_residual.fits`, `chainfit_basis.fits` (one stitched FM
  image per node) and `chainfit_region.fits`.

It rebuilds the run's reducer, so it takes the run's data options as `resume` does.  It only
reads the run directory, so a finished run stays as it is, and `--out` puts its output
elsewhere.  On a run made with the line subtracted, it fits what is left of the line.

A line can also be given by its nodes, in a file `--subtract-line` and `fit-chain --line-nodes`
read:

```
pa 13            # PA of the first half-line (deg), the second is pa + 180
step 0.5         # px between the chain's sources (default 0.5)
rho_max 3.4      # arcsec, the last value is held out to here (default: the crop's edge)
# |s| (arcsec)   a, first half-line   a, second half-line
1.0  1.0e-6  0.5e-6
1.5  2.0e-6  1.0e-6
2.0  1.5e-6  1.0e-6
3.0  3.0e-6  2.5e-6
```

`a` is the ridge's peak contrast: across the line, the ridge peaks where a point source of
contrast `a` would.  It is piecewise linear between the nodes and held at the last one.
The chain has a source every `step` px from the first node to `rho_max`, each at
`a(s) step / L`, where `L` is the injection PSF's line integral through its center along the
radial direction, over its central value (`chainfit.psf_line_integral`).  On NEAR, `L` is 7.6 px
for the n4 library PSF out to 1.05", and beyond it 7.1 and 7.2 px for the B-star templates of
nights 4 and 6.  The chain takes `L` from the first night's models.
`fit-chain --line-nodes FILE` makes one basis per node and half-line from the same file (its
values are not needed) and writes the fitted values as `line_fitted.txt`, ready for
`--subtract-line`.  `--per-night` then fits one amplitude per night to the fitted chain, its
shape held fixed (`per_night.json`), which says whether the line follows something that
changes from night to night.

## A finished run's winners on other frames (`klip-tpe rereduce`)

`klip-tpe rereduce --run-dir RUN` reduces each annulus' winning configuration again and
writes the final products beside the run, in `<run>_rereduce/` (or `--out`).  The frames are
what the command line types (`--cube-name`, `--drop-frames`, `--subtract-sources`,
`--subtract-line`, `--inject-sources`), and for each option it leaves out, what the run was
made on.  A typed subtraction replaces the run's as a whole, and so does a typed
`--drop-frames` list.  `--drop-frames none` reduces every frame of a run that dropped some.
So a line subtracted, planets added, or a bad frame left out cost one reduction per annulus
and kind instead of a new search.

* Each winner is reduced clean, and with the injections of the final verify's calibration
  block: by default the winner's own validation injections, as in the run.  `--verify-sources
  FILE` (rho, PA and contrast per line) puts those in instead, all of them into every
  annulus' injected reduction, each listed under the annulus that holds it.
* The tiles are stitched with the run's weights (`final_results.json`) and its seam patching.
  On the run's own frames the products are the run's, to the bit.
* It writes `klip_stitched.fits`, `klip_stitched_snr.fits`, the injected and per-night
  stitches, `klip_stitched_params.txt`, `stitched_verify_report.txt` (not with `--no-verify`)
  and `rereduce.json`, the record of the frames, the weights and the injections.
* The run directory is only read.  An output that already holds a re-reduction is refused
  unless `--overwrite` is given.
* On the run's own frames it verifies a finished run again with the current verify.  Since
  2026-10-20 each injection's noise ring leaves out the other injections, and the report says
  so above its calibration block.  Since 2026-10-24 Path A (the combined S/N and FAP of the
  candidates and the injections, and the limit tables) is measured on the stitched images,
  `klip_stitched.fits` and `klip_stitched_inj.fits`, rather than on the mean of the per-night
  stitches, and the report adds TABLE 3, each injection's contrasts solved on its own image.
  The candidate list measures its S/N there too.
* `--known RHO PA` puts a position into the verify as a candidate, a planet added with
  `--inject-sources` say, and keeps it out of the noise apertures.

From Python, `Runner.resume(run_dir, reducer, objective, sampler, same_frames=False)` with the
edited reducer, then `runner.rereduce(out_dir, verify_sources=...)`.

## Interruptions

The run checkpoints after every evaluation, every validation trial, every post-annulus hook
and every param_verify reduction; `klip-tpe resume --root ... --nights 1 2 3 4 5 6` (`--run-dir` defaults to `last`, the newest run under
`<root>/comb/opt`; give a directory to pick another) or the script's `resume` continues exactly where it stopped.  `klip-tpe extend --n-iter ...`
reopens a finished annulus with a larger budget (IDL `extend_ann`).

The checkpoint also survives a crash of the machine.  Each write is fsynced before its
rename, and the directory after it, and the save before the newest stays beside it as
`checkpoint.prev.json`.  `results.jsonl` and `results.txt` reach the disk before the
checkpoint that counts their lines, and so do the products of a finished annulus or hook
before the checkpoint that records them.  After a crash, `resume` takes `checkpoint.json`, or
`checkpoint.prev.json` when the crash left `checkpoint.json` empty or missing.  It cuts any
half line off the end of `results.jsonl` and `results.txt`, and the run ends where it would
have ended without the crash.  A resume reads every `checkpoint*.json` in the directory and
takes the one furthest along, so a backup copy such as `checkpoint.bak.json` is used when it
is the newest.

## While it runs

* `steps/stepNNNN.png` — the live panel (also `--show` for a window); `annulusNN/eval_NNNN_panel.pdf` every 10 evals.
* `results.txt` mirrors IDL's `optimize_tpe_results.txt`; `annulusNN/evalNNNN_setup.txt` per evaluation.
* `klip_stitched_running*.fits` every 10 evaluations.
* `klip-tpe compare --idl-run <IDL run dir> --out cmp ...` replays a running IDL run through Python (for cross-validation against the IDL original).
* A draw without a score says why on its line, as in `draw 3/5 score=failed (clean S/N NaN (inside the inner guard))`,
  and `results.jsonl` keeps the reason (`meta['why']`, a trial's draws in `meta['draw_why']`), as do
  `calibration.json` (`why`) and `validation.json` (`trial_why`).  A failed draw does not count toward its
  trial's mean.  The log warns once per annulus when a tenth or more of at least 20 draws have failed, and
  the end of each search gives the count with the commonest reasons.

## Afterwards

`klip-tpe plots --run-dir <dir>` regenerates every figure and book; `klip-tpe render --run-dir
<dir> [--every 10]` draws the step panels again from the saved per-eval crops -- every one,
including those the live display skipped -- into `<dir>/steps_rebuilt/` with its own movie,
leaving the live `steps/` alone (`display.render_steps` from Python).

## Re-scoring a run on other frames

Every trial is the mean of `n_remeasure` fresh injection draws, and the run keeps all of them:
`results.jsonl` records each draw's sources and raw score (`meta['draw_sources']`,
`meta['draw_raw_scores']`), and each annulus' `validation.json` keeps its trials' sources
(`trial_sources`).  The positions cannot be re-drawn from the seed on another processing of the
same data -- the sampler's forbidden sectors come from the frames' rolls and dead zones -- so
re-score from the record instead:

```python
runner.replay_draws(recorded)        # recorded: [[(rho, theta), ...], ...], one list per draw
rec, inj, clean = runner.evaluate(x, "replay")
```

or, for a library ablation, `scripts/library_ablation.py ... --draws-from other/library_ablation.json`.
`run_setup.json` says which frames a run used (`reducer.partitions.<id>.frames`: files,
pipeline version, pixel scale, repaired hot pixels).

## Knowing what is running, and getting a run back

A run is a long-lived process that outlives the terminal it was started from, and a
campaign accumulates several. Three commands cover the whole lifecycle.

```bash
klip-tpe runs                      # what exists, what is actually running
klip-tpe runs --stop-stale         # clean up strays
klip-tpe resume                    # restart the newest stopped run, no arguments
```

`klip-tpe runs` lists every run it knows about with its real state — taken from the
operating system and the run's own files, never from the bookkeeping, so a stale record
can never claim a run is alive:

```
run                  state       evals  last write  procs  annulus
------------------------------------------------------------------
run_20260912_160024  running      6761         12s      8  1
run_20260912_104108  orphaned     1204        19.4h     10  1
```

* **running** — processes, and writing.
* **stalled** — processes, but nothing written for a while. Worth a look.
* **orphaned** — processes, but writing stopped long ago. These are workers left behind by
  a parent that died; `--stop-stale` removes them.
* **stopped** / **finished** — no processes; `finished` means the run completed.

`--root <data root>` also picks up runs started before this version, or by someone else.

### Resume needs nothing

Every run records its own command line when it starts, so:

```bash
klip-tpe resume
```

finds the newest stopped run and restores the data root, the night list and the rest from
that record. Anything you pass explicitly wins over the record, so
`klip-tpe resume --workers 4` changes the worker count and nothing else. It refuses to
resume a run that is still going, rather than putting two writers on one directory.

Runs checkpoint after **every** evaluation, so an interrupted run loses at most one.  With
`--batch-mode sync` they checkpoint after every batch, and an interrupted run repeats that
batch.

### The live window

`--show` opens a window in the optimizer's own process. Rendering happens off-thread and
the window only blits a finished panel, but the blit still belongs to the main thread, so
a very long evaluation can leave it unrepainted (on macOS the system marks the process
unresponsive after about two seconds and stops compositing it — the run is unaffected, the
picture is just stale).

For a window that cannot be starved, run the viewer as its own process:

```bash
klip-tpe view                                  # newest run
klip-tpe view --run-dir <dir> --once           # status only, no window
```

It watches the panels the run writes to `steps/` and shows whichever is newest, with the
evaluation count, the rate and the age of the newest panel. While an annulus calibrates its
injection contrast, before the search draws anything, the newest panel is the latest
calibration trial's clean and injected pair. It can be attached to and detached from a run
that is already going, and it is read-only.
