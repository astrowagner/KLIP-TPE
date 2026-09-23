#!/usr/bin/env bash
# Regenerate every paper run with the reference objective (RunConfig.fixed_sources=False).
#
# Why this exists
# ---------------
# On 2026-09-13 this driver was written to rerun everything with fixed_sources=True, on the
# belief that optimize_near_2_tpe draws its injected positions once per annulus and that the
# port's per-evaluation redraw was a porting bug.  That was wrong, and reading
# near2m_randpos settles it: the source SEPARATIONS are a deterministic ladder, identical for
# every evaluation of an annulus, but the azimuth anchor (th0 = randomu(seed)*360.) is
# re-drawn every evaluation -- the IDL's own comment gives the reason, "so sources still
# rotate eval-to-eval (anti-gaming)" -- and near2m_randpos is called from inside
# for it=it_start, n_iter-1 at four call sites.  PositionSampler's "spread" strategy already
# reproduced that exactly, so the port was faithful all along.
#
# Freezing the azimuths lets the optimizer tune to the speckle realisation at those
# particular position angles: one NEAR run spent 2037 evaluations injecting at PA 87.39 and
# 267.39 before this was caught.  The variance that prompted the change is real (the same
# configuration scoring between 3.17 and 8.56) and the reference accepts it deliberately --
# its defence is the validated election, which re-scores the top candidates at FRESH
# positions.
#
# So every fixed_sources=True run from 2026-09-13 to 2026-09-14 has to be redone, and the
# *_unpaired_20260913/ directories archived that day are misnamed: those were the correct
# ones.  Keep them.
#
# Usage
#   ./rerun_paper.sh              science runs then benchmarks (A2 B2 C D E2 F2 G2 H2)
#   ./rerun_paper.sh science      A2 B2 C D
#   ./rerun_paper.sh bench        E2 F2 G2 H2
#   ./rerun_paper.sh A2 C         just those
#   WORKERS=4 ./rerun_paper.sh    leave cores for another run on the same machine
#   SHOW=0 ./rerun_paper.sh       headless (panels still written to each run's steps/)
#   FORCE=1 ./rerun_paper.sh A2   redo a stage that already finished.  The finished directory
#                                 is retired to <dir>_superseded_<stamp> first -- a benchmark
#                                 as a whole batch, so the re-run is uniform (one that merely
#                                 resumed would mix the old slots' settings with the new ones
#                                 in a single directory), and a science run because
#                                 Runner.run() resumes any directory holding a checkpoint:
#                                 re-launched in place it would reload the finished search,
#                                 rewrite the products and say "done in 2 min".  I2 (days of
#                                 LMIRCam compute) is never retired by a flag: move it by hand.
#
# The live window is ON by default and is shared: one window that follows whichever stage --
# or, inside a benchmark, whichever slot -- is running.
#
# The four benchmark stages are the same protocol on four problems -- beta Pic 9-D (E2) and
# 38-D (F2), HD 95086 20-D (G2), HIP 65426 5-D (H2) -- so the TPE-vs-random question is
# answered as a trend across dimension and instrument rather than on one target.  E2 and F2
# search an annulus that beta Pic's debris disk runs through; G2 and H2 do not.
#
# WORKERS is a real cap as of 2026-09-14.  It had always been exported here and
# run_demos.py never read it -- every reducer said max_workers="auto", meaning every core --
# so a run asked to leave cores free took them all anyway.  Four searches at 32 workers each
# on one machine took single reductions from 6 s to 729 s and got one of them killed.
#
# Resumable: a stage whose output already carries final_results.json is skipped, and the
# benchmarks skip their finished slots internally.  So if this is interrupted -- a closed
# terminal, a reaped background job -- just run it again.
#
# One at a time.  Launched under nohup this prints nothing to the terminal, which on
# 2026-09-17 read as "it didn't start" and got it started again 34 s later: the second
# instance found a 34-second-old A2_betapic (no final_results.json yet, so no skip) and
# launched a second A2 into it, two searches appending to one results.jsonl.  So: a pid
# file refuses a second instance, and no stage starts into a directory whose heartbeat is
# fresh.  Follow progress with  tail -f rerun_paper.log  (and <stage>.log).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PIDFILE=.rerun_paper.pid
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  printf '[%s] another rerun_paper.sh (pid %s) is already running -- not starting a second one.  tail -f rerun_paper.log to follow it.\n' \
    "$(date +%H:%M:%S)" "$(cat "$PIDFILE")" | tee -a rerun_paper.log
  exit 1
fi
if [[ -z "${DRY:-}" ]]; then
  echo $$ > "$PIDFILE"
  trap 'rm -f "$PIDFILE"' EXIT
fi

# The science group covers all four example instruments: NACO (A2, B2), SPHERE (C),
# NIRCam (D) and LMIRCam (I2).  I2 goes last because it is the long one -- a 5000-evaluation
# four-night LMIRCam search is days, where the other four are hours -- so the short stages
# are finished and collectable before it starts.
SCIENCE=(A2 B2 C D I2)
BENCH=(E2 F2 G2 H2)

case "${1:-}" in
  "")        STAGES=("${SCIENCE[@]}" "${BENCH[@]}") ;;
  science)   STAGES=("${SCIENCE[@]}") ;;
  bench)     STAGES=("${BENCH[@]}") ;;
  *)         STAGES=("$@") ;;
esac

# where each stage writes, so a finished one can be recognised
outdir() {
  case "$1" in
    A2) echo A2_betapic ;;         B2) echo B2_betapic_groups ;;
    C)  echo C_hd95086 ;;          D)  echo D_hip65426_pyklip ;;
    DK) echo D_hip65426_klip ;;    H2K) echo H2_bench_jwst_klip ;;
    E2) echo E2_bench ;;           F2) echo F2_bench_highdim ;;
    G2) echo G2_bench_sphere ;;    H2) echo H2_bench_jwst_pyklip ;;
    I2) echo "$RXJ_OUT" ;;
    *)  echo "" ;;
  esac
}

# LMIRCam has its own driver (scripts/run_rxj0534.py): it carries the companion-ring
# geometry, the noise-aperture budget check and the CompanionTrace callback, none of which
# belong in run_demos.py.  So this stage shells out to it rather than duplicating it.
RXJ_PARTITIONS="${RXJ_PARTITIONS:-4}"
RXJ_OUT="${RXJ_OUT:-$HOME/klip_tpe_runs/rxj0534/p$RXJ_PARTITIONS}"

stage_cmd() {
  case "$1" in
    I2) local sh=""
        [ "${SHOW:-window}" = "0" ] || sh="--show ${SHOW:-window}"
        printf '%s' "python3 ../scripts/run_rxj0534.py --partitions $RXJ_PARTITIONS \
          --n-iter ${RXJ_N_ITER:-5000} --out $RXJ_OUT --workers ${WORKERS:-auto} $sh" ;;
    *)  printf '%s' "python3 run_demos.py $1" ;;
  esac
}

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a rerun_paper.log; }

say "rerunning: ${STAGES[*]}   (reference objective, fixed_sources=False)"
# `set -u -o pipefail` without -e: a failing pre-flight (no klip_tpe on the path, a
# syntax error, the wrong interpreter) printed its traceback and the run carried on
# regardless -- so the assertion guarding the objective was not a guard at all.
if ! python3 - <<'PY' | tee -a rerun_paper.log
from klip_tpe import RunConfig, datasets
assert not RunConfig().fixed_sources, (
    "fixed_sources is ON -- that freezes the injection azimuths, which optimize_near_2_tpe "
    "re-draws every evaluation on purpose (near2m_randpos: 'anti-gaming'). This rerun would "
    "reproduce the 2026-09-13 mistake.")
print("  confirmed: RunConfig.fixed_sources = False (azimuths re-drawn per evaluation, radii fixed)")
# Warm the download cache before any stage starts.  The science and benchmark halves write
# to different directories and can run side by side, but they read the SAME cached cubes --
# so a cold cache is the one thing they share, and two cold fetches at once is the one way
# they could tread on each other.  Doing it here, once, makes running them together safe
# rather than merely advisable.  Costs a couple of stat() calls when it is already warm.
for ds in ("naco_betapic", "sphere_hd95086"):
    try:
        datasets.fetch(ds, quiet=True)
    except Exception as exc:
        print(f"  could not pre-fetch {ds}: {exc!r} (a stage that needs it will retry)")
print("  data cache warm")
PY
then
  say "pre-flight failed (see above) -- not starting any stage"
  exit 1
fi

# A final_results.json is proof of completion only if it holds a winner.  On 2026-09-17
# run D wrote one with winner_index -1 and score -inf in every annulus (350 evaluations,
# every one a pyklip/numpy TypeError), this script called it "done in 3 min", and the skip
# below would have kept it forever.  The Runner now refuses to write such a file, but the
# check is cheap and older directories exist.
finished() {
  python3 - "$1" <<'PY'
import json, sys
try:
    j = json.load(open(sys.argv[1] + "/final_results.json"))
except Exception:
    sys.exit(1)
ann = j.get("annuli") or []
ok = bool(ann) and all(a.get("winner_index", -1) >= 0 or a.get("search_best_index", -1) >= 0 for a in ann)
sys.exit(0 if ok else 2)
PY
}

# Age in seconds of the youngest heartbeat.json under a stage directory (its own, or a
# benchmark slot's), or nothing when there is none.  A heartbeat younger than 5 minutes
# means a process is writing there now.
live_age() {
  python3 - "$1" <<'PY'
import glob, json, os, sys, time
d = sys.argv[1]
ages = []
for p in glob.glob(os.path.join(d, "heartbeat.json")) + glob.glob(os.path.join(d, "*", "heartbeat.json")):
    try:
        ages.append(time.time() - float(json.load(open(p)).get("t", 0)))
    except Exception:
        pass
print(int(min(ages)) if ages else "")
PY
}
LIVE_S=300

for s in "${STAGES[@]}"; do
  d="$(outdir "$s")"
  failed=""
  live=""
  if [[ -n "$d" && -d "$d" ]]; then
    age="$(live_age "$d")"
    [[ -n "$age" && "$age" -lt "$LIVE_S" ]] && live="$age"
  fi
  if [[ -n "$live" && -z "${DRY:-}" ]]; then
    say "$s: a run is LIVE in $d (heartbeat ${live}s old) -- not starting a second one into it; skipping.  tail -f $s.log to follow it"
    continue
  fi
  if [[ -n "$d" && -f "$d/final_results.json" ]]; then
    finished "$d"; fin=$?
    if [[ $fin -eq 2 ]]; then
      failed=1
    elif [[ -z "${FORCE:-}" ]]; then
      say "$s: already finished ($d) -- skipping.  FORCE=1 to redo"
      continue
    fi
  fi
  if [[ -n "${DRY:-}" ]]; then
    extra=""
    [[ -n "$failed" ]] && extra="   [would retire $d: its final_results.json has no winner]"
    [[ -n "${FORCE:-}" && -n "$d" && -f "$d/bench_tag.txt" ]] && \
      extra="$extra   [FORCE would retire batch $(cat "$d/bench_tag.txt")]"
    [[ -n "${FORCE:-}" && -n "$d" && ! -f "$d/bench_tag.txt" && -f "$d/final_results.json" && "$s" != "I2" ]] && \
      extra="$extra   [FORCE would retire the finished run $d]"
    [[ -n "${FORCE:-}" && "$s" == "I2" && -n "$d" && -f "$d/final_results.json" ]] && \
      extra="$extra   [FORCE does not retire I2; move $d aside by hand to redo it]"
    [[ -n "$live" ]] && extra="$extra   [LIVE: heartbeat ${live}s old -- would be skipped]"
    say "$s: would run  $(stage_cmd "$s")   [WORKERS=${WORKERS:-auto} SHOW=${SHOW:-window}]$extra"
    continue
  fi
  if [[ -n "$failed" ]]; then
    keep="${d}_failed_$(date +%Y%m%d_%H%M%S)"
    say "$s: $d has a final_results.json with NO winner (every evaluation failed) -- retiring it to $(basename "$keep") and running again"
    mv "$d" "$keep"
  fi
  # FORCE on a FINISHED stage has to retire its directory, not just get past the skip above.
  # A benchmark resumes by reading bench_tag.txt and rejoining the slots of that tag, so a
  # stage re-run with FORCE=1 used to march straight back into the batch it was meant to
  # replace -- and any setting changed for the re-run (DISK_CUT, the objective, the space)
  # then applied to the NEW slots only, leaving one directory holding two protocols.  That
  # is how F2 ended up with slot 0 uncut, slots 1-4 cut, and slot 5 half of each.
  # Retiring the directory takes bench_tag.txt with it, so the stage mints a fresh tag.
  # A science run is no different in effect: Runner.run() auto-resumes any directory that
  # holds a checkpoint.json, so a finished A2 re-launched in place reloads the finished
  # search, finds every annulus done, rewrites the final products and reports "done in
  # 2 min" -- nothing searched again, and the old calibration kept.  I2 is the exception:
  # a 5000-evaluation four-night LMIRCam search is days of compute, and no flag retires it.
  if [[ -n "${FORCE:-}" && -n "$d" && -d "$d" ]]; then
    if [[ -f "$d/bench_tag.txt" ]]; then
      keep="${d}_superseded_$(date +%Y%m%d_%H%M%S)"
      say "$s: FORCE -- retiring the existing batch $(cat "$d/bench_tag.txt") to $(basename "$keep")"
      mv "$d" "$keep"
    elif [[ -f "$d/final_results.json" ]]; then
      if [[ "$s" == "I2" ]]; then
        say "$s: FORCE does not retire the LMIRCam run ($d: days of compute) -- move it aside by hand to redo it; skipping"
        continue
      fi
      keep="${d}_superseded_$(date +%Y%m%d_%H%M%S)"
      say "$s: FORCE -- retiring the finished run to $(basename "$keep") (re-launched in place it would only resume)"
      mv "$d" "$keep"
    fi
  fi
  t0=$SECONDS
  say "$s: starting"
  if WORKERS="${WORKERS:-auto}" SHOW="${SHOW:-window}" \
       bash -c "$(stage_cmd "$s")" >> "$s.log" 2>&1; then
    say "$s: done in $(( (SECONDS - t0) / 60 )) min"
  else
    say "$s: STOPPED after $(( (SECONDS - t0) / 60 )) min -- see $s.log; re-run this script to continue"
  fi
done

say "stages attempted.  Collect with:  python3 collect.py && python3 figs.py"
