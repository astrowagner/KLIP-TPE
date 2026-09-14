#!/usr/bin/env bash
# RX J0534.0-0221 (LBTI/LMIRCam L'), four nights, sixteen partitions.
#
#   scripts/run_rxj0534.sh check          look at the data, evaluate the default, stop
#   scripts/run_rxj0534.sh                the full 16-partition search (~150-D)
#   scripts/run_rxj0534.sh 4              one partition per night (~38-D), for comparison
#   scripts/run_rxj0534.sh resume         carry on where it stopped
#
# The live window is on by default; `show=0 scripts/run_rxj0534.sh ...` turns it off.
# A second, freeze-proof view of the same run is always available from another terminal:
#   klip-tpe view --run-dir <the run directory this prints>
#
# Every run checkpoints after each evaluation, so Ctrl-C is safe and `resume` is exact.
# Set LMIRCAM_ROOT if the data is not at /Volumes/RAID36TB/LMIRCam.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export LMIRCAM_ROOT="${LMIRCAM_ROOT:-/Volumes/RAID36TB/LMIRCam}"
OUT_BASE="${OUT_BASE:-$HOME/klip_tpe_runs/rxj0534}"

# the live window is on by default here -- this is the interactive entry point.
# show=0 turns it off for a headless or batch run (the panels are still written to disk).
SHOW="--show"
[ "${show:-1}" = "0" ] && SHOW=""

case "${1:-}" in
  resume)
    # this script builds its runner directly rather than through the klip-tpe CLI, so the
    # CLI's registry knows nothing about it: resuming goes back through the script, which
    # picks the search configuration up from the run's own checkpoint.
    P="${2:-16}"
    exec python3 scripts/run_rxj0534.py --resume --partitions "$P" \
         --out "$OUT_BASE/p$P" ${SHOW}
    ;;
  check)
    exec python3 scripts/run_rxj0534.py --partitions 16 --default-only \
         --out "$OUT_BASE/check" ${SHOW}
    ;;
esac

P="${1:-16}"
case "$P" in 4|8|16) ;; *) echo "partitions must be 4, 8 or 16 (got '$P')" >&2; exit 2 ;; esac
shift || true

python3 scripts/run_rxj0534.py \
  --partitions "$P" \
  --n-iter "${N_ITER:-5000}" \
  --k-max "${K_MAX:-20}" \
  --max-drop "${MAX_DROP:-3}" \
  --out "$OUT_BASE/p$P" \
  ${SHOW} "$@"
