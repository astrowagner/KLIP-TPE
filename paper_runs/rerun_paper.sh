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
#   FORCE=1 ./rerun_paper.sh A2   redo a stage that already finished
#
# The live window is ON by default and is shared: one window that follows whichever stage --
# or, inside a benchmark, whichever slot -- is running.
#
# The four benchmark stages are the same protocol on four problems -- beta Pic 9-D (E2) and
# 38-D (F2), HD 95086 20-D (G2), HIP 65426 11-D (H2) -- so the TPE-vs-random question is
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
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

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
    C)  echo C_hd95086 ;;          D)  echo D_hip65426 ;;
    E2) echo E2_bench ;;           F2) echo F2_bench_highdim ;;
    G2) echo G2_bench_sphere ;;    H2) echo H2_bench_jwst ;;
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
python3 - <<'PY' | tee -a rerun_paper.log
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

for s in "${STAGES[@]}"; do
  d="$(outdir "$s")"
  if [[ -z "${FORCE:-}" && -n "$d" && -f "$d/final_results.json" ]]; then
    say "$s: already finished ($d) -- skipping.  FORCE=1 to redo"
    continue
  fi
  t0=$SECONDS
  if [[ -n "${DRY:-}" ]]; then
    say "$s: would run  $(stage_cmd "$s")   [WORKERS=${WORKERS:-auto} SHOW=${SHOW:-window}]"
    continue
  fi
  say "$s: starting"
  if WORKERS="${WORKERS:-auto}" SHOW="${SHOW:-window}" \
       bash -c "$(stage_cmd "$s")" >> "$s.log" 2>&1; then
    say "$s: done in $(( (SECONDS - t0) / 60 )) min"
  else
    say "$s: STOPPED after $(( (SECONDS - t0) / 60 )) min -- see $s.log; re-run this script to continue"
  fi
done

say "stages attempted.  Collect with:  python3 collect.py && python3 figs.py"
