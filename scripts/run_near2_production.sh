#!/usr/bin/env bash
# NEAR2 production run in Python -- the exact protocol of the IDL run_20260906_173000:
#   6 nights, 3 annuli [0,20,40,60] px, 10000/2500/2500 evals with 500 warm-up each,
#   annulus 1 forced to contrast 6e-5 (annuli 2-3 calibrated, ceiling 6e-5), two sources at the
#   area-weighted mid radius, TPE gamma 0.25 / ncand 48 / explore 0.15 / p_local 0.15 / pbest 0,
#   univariate densities, validation 8 trials x 3 candidates, k_klip up to 100, drop1/drop2 night
#   selection, KLIP-FM cross-check + live FM preview, verify / param_verify / candidates, all
#   display products and the progress movie.
#
# Products land in  $ROOT/comb/opt/run_YYYYMMDD_HHMMSS/  (IDL layout) unless RUN_DIR is set.
# Every core is used by default (WORKERS=auto); WORKERS=8 pins it, WORKERS=-2 leaves two free.
# SHOW=1 opens the live window, ALIENS=1 adds the launch movie, SMOKE=1 runs the short end-to-end test.
# Interrupted?  klip-tpe resume        (no arguments: the run's own record has the rest)
# What is running? klip-tpe runs       A window that cannot freeze:  klip-tpe view
set -euo pipefail
# lowercase spellings are accepted too (show=1 ... == SHOW=1 ...)
ROOT="${ROOT:-${root:-}}"; WORKERS="${WORKERS:-${workers:-}}"; SEED="${SEED:-${seed:-}}"; WINDOW="${WINDOW:-${window:-}}"
SHOW="${SHOW:-${show:-}}"; ALIENS="${ALIENS:-${aliens:-}}"; SMOKE="${SMOKE:-${smoke:-}}"; RUN_DIR="${RUN_DIR:-${run_dir:-}}"
NITER="${NITER:-${niter:-}}"; NINIT="${NINIT:-${ninit:-}}"; NTOP="${NTOP:-${ntop:-}}"; NVALID="${NVALID:-${nvalid:-}}"
for v in ROOT WORKERS SEED WINDOW SHOW ALIENS SMOKE RUN_DIR NITER NINIT NTOP NVALID; do [[ -z "${!v}" ]] && unset "$v"; done
export PYTHONUNBUFFERED=1   # the log reaches the terminal and run.log line by line (not in 8 kB chunks)
WORKERS="${WORKERS:-auto}"
SEED="${SEED:-}"
WINDOW="${WINDOW:-1.0}"    # live window size as a fraction of the 1850x990 panel (IDL window size)
# budget: production defaults, or SMOKE=1 for an end-to-end smoke test (25 warm-up + 50 TPE per annulus,
# 5 validation candidates x 10 trials); NITER / NINIT / NTOP / NVALID override individually
if [[ -n "${SMOKE:-}" ]]; then
  NITER="${NITER:-75 75 75}"; NINIT="${NINIT:-25 25 25}"; NTOP="${NTOP:-5}"; NVALID="${NVALID:-10}"
else
  NITER="${NITER:-10000 2500 2500}"; NINIT="${NINIT:-500 500 500}"; NTOP="${NTOP:-3}"; NVALID="${NVALID:-8}"
fi

# resume:  ./run_near2_production.sh resume            -> the newest run_* under $ROOT/comb/opt
#          ./run_near2_production.sh resume last       -> same
#          ./run_near2_production.sh resume <run dir>  -> that run (a name under $ROOT/comb/opt or a full path)
if [[ "${1:-}" == "resume" ]]; then
  RUN_DIR="${2:-last}"
  # No ROOT needed: every run records its own command line, so `klip-tpe resume` recovers
  # the data root and the night list by itself.  (`klip-tpe resume` on its own does the same
  # thing without this script at all.)
  if [[ -z "${ROOT:-}" ]]; then
    exec klip-tpe resume --run-dir "$RUN_DIR" ${WORKERS:+--workers "$WORKERS"} \
         ${SHOW:+--show} ${ALIENS:+--aliens} --window-scale "$WINDOW"
  fi
  if [[ "$RUN_DIR" == "last" || "$RUN_DIR" == "latest" ]]; then
    RUN_DIR=$(ls -d "$ROOT"/comb/opt/run_* 2>/dev/null | sort | tail -1)
    [[ -n "$RUN_DIR" ]] || { echo "no run_* directory under $ROOT/comb/opt" >&2; exit 1; }
  elif [[ ! -d "$RUN_DIR" && -d "$ROOT/comb/opt/$RUN_DIR" ]]; then
    RUN_DIR="$ROOT/comb/opt/$RUN_DIR"
  fi
  echo "resuming: $RUN_DIR"
  exec klip-tpe resume --run-dir "$RUN_DIR" --root "$ROOT" --nights 1 2 3 4 5 6 --workers "$WORKERS" ${SHOW:+--show} ${ALIENS:+--aliens} --window-scale "$WINDOW"
fi

# everything below starts a NEW run, which does need to know where the data are
ROOT="${ROOT:?set ROOT=/path/to/NEAR_py (the NEAR data root; products go to \$ROOT/comb/opt).
     A resume needs no ROOT:  ./run_near2_production.sh resume   (or just: klip-tpe resume)}"

RUN_DIR="${RUN_DIR:-$ROOT/comb/opt/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
echo "run directory: $RUN_DIR"
klip-tpe near \
  --root "$ROOT" --nights 1 2 3 4 5 6 \
  --run-dir "$RUN_DIR" \
  --ann-edges 0 20 40 60 \
  --n-iter $NITER --n-init $NINIT \
  --use-contrast 6e-5 0 0 --cmax-cal 6e-5 \
  --blocks univariate --n-top "$NTOP" --n-valid "$NVALID" --k-max 100 \
  --verify --param-verify --candidates \
  --stitch-every 10 --pdf-every 10 --display-every 1 \
  --workers "$WORKERS" ${SEED:+--seed "$SEED"} ${SHOW:+--show} ${ALIENS:+--aliens} --window-scale "$WINDOW" \
  2>&1 | tee -a "$RUN_DIR/run.log"
