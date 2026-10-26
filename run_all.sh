#!/bin/bash
# =============================================================================
# run_all.sh -- the whole pipeline, in order, resumable.
# =============================================================================
# Every stage skips work already on disk, so an interrupted run is restarted by
# re-invoking this script. Stages are gated on the DONE markers the individual
# runners write, not on exit codes, because an ns-3 worker can die without the
# runner noticing.
#
# Total: roughly 70 h on 8 cores with 4 parallel workers. Stage times are in
# README.md section 8. To rebuild only the paper from frozen results, skip to
# stage 7 with:  ./run_all.sh --paper-only
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
LOG="$HERE/results/run_all.log"
mkdir -p "$HERE/results"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

paper(){
  say "STAGE 7  freeze -> tables -> figures -> pdf"
  "$PY" "$HERE/analysis-scripts/freeze_results.py" || return 1
  "$PY" "$HERE/analysis-scripts/gen_tables.py"     || return 1
  "$PY" "$HERE/analysis-scripts/make_figs.py"      || return 1
  ( cd "$HERE/paper" && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1 )
  local p; p=$(pdfinfo "$HERE/paper/main.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
  say "  paper built: ${p:-?} pages"
}

if [ "${1:-}" = "--paper-only" ]; then paper; exit $?; fi

say "===== pipeline start ====="
stage(){ # stage <n> <marker> <script...>
  local n="$1" marker="$2"; shift 2
  if [ -f "$HERE/results/$marker" ]; then say "STAGE $n  already complete ($marker)"; return 0; fi
  say "STAGE $n  $1"
  "$@" || { say "STAGE $n FAILED -- fix, then re-run run_all.sh"; exit 1; }
}

stage 1 FINAL_DONE      "$HERE/collect-scripts/1-collect-dataset.sh"
"$PY" "$HERE/collect-scripts/3-build-energy-reward.py" || exit 1
stage 2 SEEDS10_DONE    "$HERE/offline-rl-scripts/1-train-all-seeds.sh"
stage 3 FQE_DONE        "$HERE/offline-rl-scripts/2-select-checkpoints.sh"
stage 4 ROWS_DONE       "$HERE/eval-scripts/1-baseline-rows.sh"
stage 5 HEUR2_DONE      "$HERE/eval-scripts/2-tune-hand-designed.sh"
stage 6 ORIGREWARD_DONE "$HERE/eval-scripts/3-control-reward-repair.sh"
stage 6 NSTA9_DONE      "$HERE/eval-scripts/4-control-solicitation-limit.sh"
paper
say "===== pipeline complete ====="
