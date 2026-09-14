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

## Jupyter

`notebooks/near2_production_run.ipynb`: the same protocol built from the Python API
(`near.make_reducer / make_space / make_guard / default_config`, `RunConfig`, `Runner`,
`LiveDisplay`), started in a background thread so the notebook stays live, with a `watch()`
cell that shows the newest step panel and the log tail, and a resume cell.

## Cores

`--workers auto` (the default; `max_workers="auto"` in the API) uses every core the process
can see.  The budget is spent like IDL's bridges: after the nights are loaded the reducer forks
`min(workers, 2 × nights)` worker processes that share the loaded cubes (copy-on-write, nothing
is pickled but requests and the small result images); the injected and the clean reduction of
an evaluation run concurrently, each mapping its selected nights onto those workers; whatever
budget is left over becomes per-target threads inside each night's KLIP loop; BLAS is pinned to
one thread per worker.  `--pool threads` keeps everything in one process (slower: Python's GIL
limits the thread version to roughly one to two cores' worth of work), `--workers 8` pins the
budget, `--workers -2` leaves two cores free.  The startup log prints the plan, e.g.
`parallel: 12 workers on 12 cores: injected + clean reductions concurrent, 12 partition jobs in
parallel, 1 target thread(s) per partition job; pool = processes`.

Memory: six NEAR nights are ~2000 frames × 150² each and are shared by the workers; each
worker holds one binned cube while it reduces — a few GB in total.

## Interruptions

The run checkpoints after every evaluation, every validation trial, every post-annulus hook
and every param_verify reduction; `klip-tpe resume --root ... --nights 1 2 3 4 5 6` (`--run-dir` defaults to `last`, the newest run under
`<root>/comb/opt`; give a directory to pick another) or the script's `resume` continues exactly where it stopped.  `klip-tpe extend --n-iter ...`
reopens a finished annulus with a larger budget (IDL `extend_ann`).

## While it runs

* `steps/stepNNNN.png` — the live panel (also `--show` for a window); `annulusNN/eval_NNNN_panel.pdf` every 10 evals.
* `results.txt` mirrors IDL's `optimize_tpe_results.txt`; `annulusNN/evalNNNN_setup.txt` per evaluation.
* `klip_stitched_running*.fits` every 10 evaluations.
* `klip-tpe compare --idl-run <IDL run dir> --out cmp ...` replays a running IDL run through Python (for cross-validation against the IDL original).

## Afterwards

`klip-tpe plots --run-dir <dir>` regenerates every figure and book; `python -c "from klip_tpe
import display; display.render_steps('<dir>', every=10)"` rebuilds the step frames and the
movie from the saved per-eval crops.

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

Runs checkpoint after **every** evaluation, so an interrupted run loses at most one.

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
