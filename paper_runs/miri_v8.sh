#!/usr/bin/env bash
# The paper's MIRI runs, v8: HIP 65426 in F1140C, both engines, on the archive calints with the
# loader's static hot pixels repaired and every draw recorded.
#
#   paper_runs/miri_v8.sh                 search, ablate, companion tests, figure
#   paper_runs/miri_v8.sh search          the two searches (pyKLIP ~5 h, built-in ~1.5 h on 2 cores)
#   paper_runs/miri_v8.sh ablate          the library ablations (pyKLIP with Carter et al. as
#                                         published, ~1.5 h; built-in at pyKLIP's contrasts, ~20 min)
#   paper_runs/miri_v8.sh after           companion_tests.py miri miri_light, then miri_fig.py
#   MIRI_DATA=... WORKERS=... paper_runs/miri_v8.sh
#
# v7 differs only in its frames: the same spaces, budgets, annuli and seed, on frames without the
# hot-pixel repair (and before the searched library's cache fix, for the built-in engine).  Run it
# from the repository root.  MIRI_DATA must hold ONE processing of the data: the loader refuses a
# tree with the same exposure twice (MAST's mastDownload/ beside a re-reduction under reproc/).
set -euo pipefail
cd "$(dirname "$0")/.."
# paper_runs/ imports klip_tpe from this checkout, so the stages work in an environment where
# the package is not installed (scripts/ insert the checkout themselves; paper_runs/ does not)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
DATA="${MIRI_DATA:-$HOME/Data/JWST/hip65426_miri/mastDownload}"
W="${WORKERS:-auto}"
ANN="6.68240733102219 20.047221993066568 26.864474529578956 36.0"   # v7's annuli, px
export MPLBACKEND="${MPLBACKEND:-Agg}" TQDM_DISABLE=1
what="${1:-all}"

search() {
  for eng in pyklip klip; do
    python3 scripts/run_miri.py --data "$DATA" --target HIP-65426 --filter F1140C \
      --ref-target HIP-68245 --partition all --crop 40 --known 0.826 150.2 --ann $ANN \
      --n-iter 200 --n-init 40 --n-remeasure 3 --k-max 40 --seed 21 --backend "$eng" \
      --hot-pixels on --workers "$W" --no-display --out "miri_HIP-65426_F1140C_v8_$eng"
  done
}

ablate() {
  P=miri_HIP-65426_F1140C_v8_pyklip
  K=miri_HIP-65426_F1140C_v8_klip
  python3 scripts/library_ablation.py --data "$DATA" --run-dir "$P" --n-draws 40 --workers "$W" \
    --no-show --out "$P/library_ablation.json"
  python3 scripts/library_ablation.py --data "$DATA" --run-dir "$K" --n-draws 40 --workers "$W" \
    --no-show --contrast-from "$P" --versus "$P/library_ablation.json" --out "$K/library_ablation.json"
}

after() {
  (cd paper_runs && MIRI_DATA="$DATA" MIRI_TAG=v8 python3 companion_tests.py miri miri_light \
     && MIRI_DATA="$DATA" MIRI_TAG=v8 python3 miri_fig.py)
}

case "$what" in
  all) search; ablate; after ;;
  search) search ;;
  ablate) ablate ;;
  after) after ;;
  *) echo "usage: $0 [all|search|ablate|after]" >&2; exit 2 ;;
esac
