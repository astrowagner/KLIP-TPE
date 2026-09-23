#!/usr/bin/env bash
# Re-run every paper search at a converged budget, with the live display on.
#
# Why
# ---
# The convergence traces in the 2026-09 draft (the science trace figure and the benchmark
# figure) show the running best still climbing at the end of several stages.  A search
# reported at a budget it has not converged at understates every strategy -- and understates
# the guided one most, because that is the one still finding things.  So: at least 1000
# evaluations per annulus everywhere, and per benchmark arm.
#
# This is a thin wrapper around rerun_paper.sh, which owns all the safety: the pid file that
# refuses a second instance, the heartbeat check that refuses to start into a directory
# something is already writing, the FORCE retirement that moves a finished run aside to
# <dir>_superseded_<stamp> instead of resuming it, and the per-stage resume.  Everything
# here does is set the budget, keep the window open, and say how long it will take.
#
#   ./long_run.sh                 everything: A2 B2 C D, then E2 F2 G2 H2
#   ./long_run.sh science         A2 B2 C D only  (~5 h -- the paper's numbers)
#   ./long_run.sh bench           E2 F2 G2 H2 only (~38 h -- the strategy comparison)
#   NITER=1500 ./long_run.sh science          deeper science budget
#   BENCH_NITER=1500 ./long_run.sh bench      deeper benchmark budget
#   ITER_SCALE=2 ./long_run.sh                double each stage's own budget instead
#   DRY=1 ./long_run.sh                       print the plan and the projection, run nothing
#   SHOW=0 ./long_run.sh                      headless (panels still written to steps/)
#
# The budgets reach the stages through $NITER / $BENCH_NITER / $ITER_SCALE, which
# run_demos.budget() applies; n_init follows at each annulus' own warm-up fraction, so a
# longer search does not quietly become a smaller fraction of warm-up.  Nothing is edited in
# run_demos.py, so a plain `python3 run_demos.py A2` still reproduces the draft.
#
# The old directories are kept.  FORCE=1 retires each finished run to
# <dir>_superseded_<stamp> before starting, so the September numbers stay on disk and
# summary.json can be rebuilt from either.
#
# Watching it.  The live window is on (SHOW=window): one window, shared, following whichever
# stage -- or, inside a benchmark, whichever slot -- is running.  Alongside it:
#     tail -f long_run.log        stage starts, finishes, and a progress line every 10 min
#     tail -f <stage>.log         that stage's own output
#     ls <dir>/steps/ | tail      the panel PNGs, written whether or not the window is up
#
# Sleep.  A 40-hour run and an idle Mac do not mix, so this re-execs itself under
# `caffeinate -is` (idle + system sleep held off; the display is still allowed to sleep).
# Closing the laptop lid still suspends it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -z "${KLIPTPE_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null 2>&1; then
  export KLIPTPE_CAFFEINATED=1
  exec caffeinate -is "$0" "$@"
fi

export NITER="${NITER:-1000}"
export BENCH_NITER="${BENCH_NITER:-1000}"
export ITER_SCALE="${ITER_SCALE:-1}"
export SHOW="${SHOW:-window}"
export WORKERS="${WORKERS:-auto}"
export FORCE="${FORCE:-1}"

case "${1:-}" in
  "")      STAGES=(A2 B2 C D E2 F2 G2 H2) ;;
  science) STAGES=(A2 B2 C D) ;;
  bench)   STAGES=(E2 F2 G2 H2) ;;
  *)       STAGES=("$@") ;;
esac
# I2 (the LMIRCam run) is deliberately not in the default set: it is days of compute on a
# data set that is not in this paper, and rerun_paper.sh will not retire it anyway.

LOG=long_run.log
say() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- projection
# Seconds per evaluation, measured on this machine from the 2026-09-17/18 runs:
#   A2 28 min / 1000 evals   B2 12 / 400    C 29 / 650     D 19 / 350
#   E2 335 min / 19200       F2 226 / 12800  H2 484 / 12800   (G2 not timed; F2's rate used)
# and the slot count of each benchmark (modes x seeds x annuli).
python3 - "${STAGES[@]}" <<'PY' | tee -a "$LOG"
import os, sys
NITER = int(os.environ["NITER"]); BENCH = int(os.environ["BENCH_NITER"])
SCALE = float(os.environ["ITER_SCALE"])
# stage -> (list of built-in per-annulus budgets, seconds/eval, benchmark slot count or 0)
S = {"A2": ([400, 300, 300], 1.68, 0), "B2": ([400], 1.80, 0),
     "C":  ([350, 300],      2.68, 0), "D":  ([200, 150], 3.26, 0), "DK": ([200, 150], 3.26, 0),
     "E2": ([800], 1.05, 24), "F2": ([800], 1.06, 16),
     "G2": ([800], 1.06, 32), "H2": ([800], 2.27, 16), "H2K": ([800], 2.27, 16)}
tot = 0.0
print("  stage   evaluations          projected")
for s in sys.argv[1:]:
    if s not in S:
        print(f"  {s:<6}  (not projected)"); continue
    base, rate, slots = S[s]
    floor = BENCH if slots else NITER
    per = [max(floor, round(b * SCALE)) for b in base]
    n = sum(per) * (slots or 1)
    h = n * rate / 3600.0
    tot += h
    shape = "+".join(str(p) for p in per) + (f" x {slots} slots" if slots else "")
    print(f"  {s:<6}  {n:>7,}  ({shape:<22})  {h:>5.1f} h")
days = f"  ({tot / 24:.1f} days)" if tot > 24 else ""
print(f"  {'total':<6}  {'':>7}   {'':<24}  {tot:>5.1f} h{days}")
PY

say "budget: NITER=$NITER  BENCH_NITER=$BENCH_NITER  ITER_SCALE=$ITER_SCALE"
say "stages: ${STAGES[*]}   (SHOW=$SHOW  WORKERS=$WORKERS  FORCE=$FORCE)"

if [[ -n "${DRY:-}" ]]; then
  DRY=1 ./rerun_paper.sh "${STAGES[@]}"
  say "DRY -- nothing run"
  exit 0
fi

# ------------------------------------------------------- progress ticker
# rerun_paper.sh logs a stage's start and its finish, which on a 16-hour benchmark is two
# lines in sixteen hours.  This adds a line every ten minutes naming the directory being
# written and how many evaluations are in it, so `tail -f long_run.log` answers "is it
# moving" without opening the window.
progress() {
  while sleep 600; do
    python3 - <<'PY' >> "$LOG" 2>/dev/null
import glob, json, os, time
newest, best = None, 0.0
for p in glob.glob("*/heartbeat.json") + glob.glob("*/*/heartbeat.json"):
    try:
        t = float(json.load(open(p)).get("t", 0))
    except Exception:
        continue
    if t > best:
        newest, best = p, t
if newest is None:
    raise SystemExit
d = os.path.dirname(newest)
rj = os.path.join(d, "results.jsonl")
n = sum(1 for _ in open(rj)) if os.path.exists(rj) else 0
age = int(time.time() - best)
print(f"[{time.strftime('%m-%d %H:%M:%S')}]   ... {d}: {n} evaluations, heartbeat {age}s old")
PY
  done
}
progress & TICKER=$!
trap 'kill $TICKER 2>/dev/null' EXIT

t0=$SECONDS
./rerun_paper.sh "${STAGES[@]}"
say "all stages attempted in $(( (SECONDS - t0) / 3600 ))h $(( ((SECONDS - t0) % 3600) / 60 ))m"
say "now:  python3 collect.py && python3 figs.py"
