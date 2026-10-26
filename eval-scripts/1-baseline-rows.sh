#!/bin/bash
# =============================================================================
# run_rows.sh -- the four missing baseline rows, all service-period aligned
# =============================================================================
# The intended comparison is RL controlling TWT *and* OFDMA jointly, measured
# against hand-designed policies that also run both. The main run compared
# against TWT-only baselines, which is the wrong reference for that question.
#
#   row 3  static TWT  + OFDMA      fixed:11:17:1
#   row 5  random TWT  + OFDMA      randomtwt:1
#   row 6  OFDMA only, no TWT       alwayson:1
#   row 7  random OFDMA, no TWT     random  (samples the joint 760 space, ~50% OFDMA,
#                                            with TWT disabled)
#
# Every OFDMA row uses --sp-aligned --sleep-guard-us 0: triggers are only issued
# to a station whose service period still has room for the whole exchange, so no
# station can sleep mid-exchange and the 8 ms deferral workaround is not needed.
# Using the naive scheduler instead would inflate every OFDMA row by ~45% and
# flatter the learned arms for the wrong reason.
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# Repo root is one level up: scripts live in <root>/<topic>-scripts/.
ROOT="$(cd "$HERE/.." && pwd)"
# Override with:  PY=/path/to/python3.11 ./script.sh
PY="${PY:-python3}"
R="$ROOT/results/final/rows"; LOG="$ROOT/results/rows.log"
PAR="${PAR:-3}"; RESERVE_MB=2500
mkdir -p "$R"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
exec 8>"$ROOT/run/rows.lock"; flock -n 8 || { say "another rows run holds the lock"; exit 0; }
avail(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
job(){
  local tag="$1" seg="$2" pol="$3"; shift 3
  local d="$R/$tag"; mkdir -p "$d"
  local have; have=$(ls "$d"/*.json 2>/dev/null | wc -l)
  [ "$have" -ge 48 ] && { say "  $tag complete ($have)"; return 0; }
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ] || [ "$(avail)" -lt "$RESERVE_MB" ]; do sleep 20; done
  say "  start $tag ($pol $*)"
  ( "$PY" "$ROOT/eval-scripts/evaluate.py" --policy "$pol" --seeds 90000-90049 \
      --out "$d" --seg "$seg" --progress-every 0 "$@" >> "$LOG" 2>&1
    echo "[$(date '+%F %T')]   finished $tag ($(ls "$d"/*.json 2>/dev/null | wc -l))" >> "$LOG" ) &
}
say "===== missing baseline rows, SP-aligned ====="
SP="--sp-aligned --sleep-guard-us 0"
job r3_static_ofdma  r3 "fixed:11:17:1"  $SP
job r5_random_ofdma  r5 "randomtwt:1"    $SP
job r6_ofdma_no_twt  r6 "alwayson:1"     $SP
job r7_rand_no_twt   r7 "random"         $SP
wait
say "===== rows complete ====="
touch "$ROOT/results/ROWS_DONE"
