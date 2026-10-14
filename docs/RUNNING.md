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

`--workers auto` uses every core the process can see (the default, `max_workers="auto"` in
the API).  `--workers 8` pins the budget and `--workers -2` leaves two cores free.  The budget
is spent like IDL's bridges.  After the nights are loaded the reducer forks
`workers / threads-per-job` worker processes, and never fewer than `min(workers, 2 × nights)`.
They share the loaded cubes copy-on-write, and only requests and the small result images
cross between processes.  A partition job runs its injection, its per-target KLIP work, its
high-pass, binning, derotation and combination on its own threads, and injects only the frames
its frame selection keeps.  `--threads-per-job` (4 by default)
sets the average.  A job expected to take long, a big night at a small bin, gets more threads,
up to four times as many, and a light one gets one.  A free worker takes the waiting job
expected to take longest.  The images do not depend on the thread count or the order.  BLAS
is pinned to one thread per worker thread.  `--pool threads` keeps everything in one process.
It is slower, since Python's GIL limits the thread version to roughly one to two cores' worth
of work.

The runner keeps as many reductions in flight as the pool can serve, and takes their results
in order:

* A trial reduces its clean image and all `--n-remeasure` injected images in one round.  Each
  draw is scored as soon as its image and the clean image are back.
* Validation reduces a candidate's clean image and its trials several at a time, enough to
  fill the pool, and reports the trials in order.
* param_verify, the calibration remeasurements and the k-scan pair work the same way.

Every draw's positions are drawn first, in the order a one-at-a-time loop draws them.  The
schedule changes when a reduction finishes, not what it returns, so a run writes the records a
one-at-a-time loop writes, and a resumed run continues exactly where it stopped.

A trial of N nights at `--n-remeasure 3` is 4N partition jobs, so for six nights
`--workers 96` (24 processes of 4 threads) runs a trial in one round.  The startup log prints
the plan:

```
parallel: 96 workers on 192 cores: 24 worker processes x 4 thread(s) per job (1 to 16 by the job's expected cost, longest first), 6 partitions per reduction, so 4 reduction(s) at full width at once; pool = processes
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
holds its own binned cube and KL bases, and each worker keeps the frames it injected for the
latest sources of every night it has reduced, at most about 0.85 GB for six NEAR nights.  On Linux the log reports, after the
search and after validation, the workers' peak resident memory and what they hold together,
which is what a given pool size needs on that machine.

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
evaluation count, the rate and the age of the newest panel. It can be attached to and
detached from a run that is already going, and it is read-only.
