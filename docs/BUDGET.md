# How many trials does a run need?

A klip-tpe run spends its evaluations in three phases — a random **warm-up**, the
TPE-guided **search**, and **validation** of the best candidates on fresh injections — and
the right size for each is set by one measurable quantity: how noisy the objective is on
your data.  This page measures that on the β Pictoris tutorial set and turns it into
numbers you can copy.

Everything below comes from four independent 1200-evaluation searches plus eight
300-evaluation searches on `tutorials/01_naco_betapic.py`'s configuration (61 NACO L′
frames, one 8–22 px annulus, nine searched dimensions, injection contrast forced to
3 × 10⁻⁴ so the runs are comparable).  The scripts are not shipped; the numbers are
reproducible from the tutorial by changing `n_iter` / `n_init`.

## The objective is noisy, and that is the whole story

Score the *same* configuration twice and you do not get the same number, because each
evaluation injects companions at fresh random positions.  Over 853 trials that each
measured one configuration three times, the per-injection-set scatter on these data is

> **σ ≈ 0.87 S/N** (pooled within-configuration standard deviation)

against a total spread of only **1.44** between trials across the whole search.  The same
measurement on JWST/MIRI HIP 65426 F1140C gives σ ≈ 0.84, so this is not a quirk of one
data set.  A single-draw score therefore separates two configurations about as well as it
separates a configuration from itself.

Two consequences run through everything else.  First, **average several draws per trial**:
`RunConfig(n_remeasure=3)` scores each trial as the mean of three independent injection
sets, which divides the scatter by √3 to ≈ 0.50 (the draws are independent — the reduction
in scatter follows √n to within the measurement error).  Second, **the best score you see
is not the score you have**; see the search section.

## Warm-up — `n_init`

The warm-up is a pure random sample.  Its job is to give TPE enough observations to build
its two densities: the sampler sorts the history by score, calls the top `gamma = 0.25` the
"good" set, and fits a Parzen estimator per dimension to it.  So after `n_init` warm-up
trials the good set holds `n_init / 4` points, and those have to populate every searched
dimension.  The β Pic space has nine.  At the old tutorial value of `n_init = 15` the good
set is **four points in nine dimensions** — TPE cannot model that, it can only echo it back.

Measured at a fixed total budget of 300, as the true value of the configuration the run
finally picked (two seeds each):

| `n_init` | good set | true S/N of the winner |
|---:|---:|---:|
| 15 | 4 | 6.73 ± 0.24 |
| 40 | 10 | 7.14 ± 0.23 |
| 100 | 25 | 7.15 ± 0.10 |
| 200 | 50 | 7.05 ± 0.21 |

Below ~40 the answer is about 0.4 S/N worse; from 40 up it is flat, and spending a third of
the budget on random draws costs nothing measurable.  With two seeds per point this is
suggestive rather than decisive, but it agrees with what the algorithm structurally needs.

**Rule of thumb: `n_init` ≈ 4–8 × the number of searched dimensions, and never below the
library default of 40.**  `print(space.names)` tells you the dimension count.

## Search — `n_iter`

Two curves matter and they are not the same curve.  One is the **best score the run has
seen**, which is what the live display's convergence trace shows.  The other is what that
configuration is **actually worth**, measured by re-scoring it on fresh injections.  Averaged
over four independent 1200-evaluation searches:

| evaluations | best score seen | true value of that configuration | optimism |
|---:|---:|---:|---:|
| 10 | 6.68 | 5.76 ± 0.45 | +0.92 |
| 20 | 7.24 | 6.41 ± 0.69 | +0.84 |
| 40 | 7.58 | 6.97 ± 0.52 | +0.61 |
| 60 | 7.61 | 6.83 ± 0.41 | +0.78 |
| 100 | 7.95 | 7.53 ± 0.67 | +0.42 |
| 120 | 8.35 | 7.69 ± 0.53 | +0.66 |
| 200 | 8.49 | 7.50 ± 0.69 | +0.99 |
| 300 | 8.85 | 7.34 ± 0.04 | +1.51 |
| 600 | 8.96 | 7.62 ± 0.62 | +1.34 |
| 1200 | 9.23 | 7.63 ± 0.63 | +1.61 |

Read the middle column: the answer improves steeply to about **100–120 evaluations** and is
flat after that, within the seed-to-seed spread of ±0.6.  Read the left column: the number
on screen keeps climbing to the last evaluation.  The gap between them is selection bias —
the maximum of many noisy draws is high partly because it got lucky — and it grows with
budget, reaching **+1.6 S/N at 1200 evaluations**.

That bias is predictable.  For `N` trials scored with scatter `σ_eff = σ / √n_remeasure`,
the expected optimism is about `σ_eff × z`, with `z = Φ⁻¹(1 − 1/N)` ≈ 2.2 at N = 60, 2.7 at
300 and 3.1 at 1200.  Here that predicts +1.6 at N = 1200 against +1.6 measured.  Running
single-draw would double it.

Three practical points follow.  A run that stops at 20 evaluations has not searched — its
answer is worth ~6.4 where the same search reaches ~7.6, and most of the improvement it
*does* show on screen is not real.  The incumbent can also get **worse** as you add trials:
in one of our four seeds the true value of the running best went 8.42 at 80 evaluations,
7.31 at 250, and back to 8.56 by 600, because the incumbent is chosen by a noisy score.  And
past the plateau, seed-to-seed variation (±0.6) is larger than anything more budget buys, so
a second run with a different seed tells you more than doubling `n_iter`.

**Rule of thumb: `n_iter` ≈ 30 × the number of searched dimensions, minimum ~200.**  On the
nine-dimensional β Pic space that is ~300, comfortably past the plateau.  Production runs
use 1000–3000 because the spaces are larger (partition selection adds a dimension per night
or channel) and the plateau moves right, not because the extra evaluations keep paying on a
small space.

## Validation — `n_top` and `n_valid`

Validation exists to fix exactly the bias above: it re-scores the top `n_top` distinct
configurations on `n_valid` fresh injection sets each and reports the median.  **This is the
number to quote.**  Skipping it (`n_top = 1`) means publishing the +1.5.

By the time a search has produced eight good candidates they are nearly tied: across twelve
runs the median true gap between the best and the second-best candidate is **0.22 S/N**,
against a per-trial scatter of 1.16 among those candidates.  So validation is much better at
rejecting an outlier than at ranking near-equals, and the useful question is not "how often
does it pick the exact best" but "how much do I give up when it doesn't":

| `n_valid` | agrees with the truth | S/N given up by picking wrong |
|---:|---:|---:|
| 1 | 25 % | 0.37 |
| 3 | 31 % | 0.29 |
| 8 | 44 % | 0.18 |
| 12 | 47 % | 0.15 |
| 24 | 57 % | 0.10 |

The agreement rate stays low because the candidates are genuinely within noise of one
another; the shortfall column shows that this hardly matters.  Eight trials is the library
default and the sensible stopping point.

`n_top` matters more.  Across those same twelve runs, the configuration the search ranked
**first** was the truly-best of its own top eight in **none** of them:

| `n_top` | truly-best candidate is inside the top `n_top` of the search ranking |
|---:|---:|
| 1 | 0 % |
| 2 | 42 % |
| 3 | 42 % |
| 4 | 50 % |
| 6 | 67 % |
| 8 | 100 % |

(The last row is 100 % by construction — the pool was eight.)  Validating two candidates
catches it 42 % of the time; six catches 67 %.  Since a validation trial is one reduction, the
whole validation stage at `n_top = 6, n_valid = 8` is 48 reductions — a few percent of a
300-evaluation search.  There is no reason to be stingy here.

**Rule of thumb: `n_top = 6, n_valid = 8`.**

## The search number and the validation number are different statistics

Do not read `search best 8.9 → validated 7.6` as the answer getting worse.  The search
maximises `score_search`, which subtracts a one-sided speckle term at each injection site
(`s_inj − max(s_clean, 0)`); validation deliberately reports `score_raw = s_inj`.  Measured
on these data that term is worth **+0.84 S/N**, so validation sits about 0.8 *above* the
search scale for the same configuration, while the reported search maximum sits ~1.5 *above*
the truth.  The two biases roughly cancel, which is why the two numbers often look similar
for no good reason.  Compare search scores with search scores and validated scores with
validated scores.

## What to set

| | `n_init` | `n_iter` | `n_top` | `n_valid` | `n_remeasure` |
|---|---:|---:|---:|---:|---:|
| smoke test (does it run?) | 5 | 10 | 1 | 1 | 1 |
| β Pic tutorial | 40 | 300 | 6 | 8 | 3 |
| a real answer on a small space | 60 | 400 | 6 | 8 | 3 |
| production (NEAR, JWST programmes) | 100–200 | 1000–3000 | 6–8 | 8–12 | 3 |

A smoke test is for checking that the plumbing works and produces a `winner.json`.  Its
score means nothing; do not report it.

## How long that takes

The β Pic tutorial's 300 evaluations at `n_remeasure=3` — 900 reductions plus validation —
took **3.2 to 6.6 minutes** on a single core across eight runs, with no display.  The spread
is real and not a timing artefact: the optimizer drifts toward configurations that are more
expensive to reduce (less temporal binning, more KL modes), so the cost per evaluation
roughly tripled between the start and the end of a run, and two 1200-evaluation searches on
the same data took 14.7 and 35.5 minutes.  Estimate from the trial count, not from the first
minute, and give `max_workers="auto"` the cores.

**Drawing the live panel is not free**, and on a machine without spare cores it dominates.
Measured on one core, 60 evaluations of this search took 43 s headless, 171 s with
`LiveDisplay(every=5)` and 290 s with `every=1` — a panel costs ~10 s of CPU there.  The
display already protects the run: when it cannot keep up it skips panels rather than
blocking, logs that it is doing so, and `klip-tpe render --run-dir ...` draws the missing
ones afterwards.  On a laptop with cores to spare the rendering thread overlaps with the
reductions and the penalty is much smaller; if a run feels slow, raise `every` before you
cut `n_iter`.

## Watching it happen

Every run can draw a live panel — convergence trace, S/N maps, the current and best
reductions, the parameter vectors, ETA.  Pass a `LiveDisplay` callback (`show="auto"`), or
watch any running directory from another terminal with `klip-tpe view`.  Without it a run is
a silent process for several minutes and there is no way to tell a converging search from a
stuck one.  See [DISPLAY.md](DISPLAY.md).
