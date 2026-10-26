#!/bin/bash
# =============================================================================
# run_heur2.sh -- re-tune the hand-designed controller under a SERVICE floor
# =============================================================================
# The original sweep picked "lowest energy subject to violations no worse than
# max-duty". No throughput or demand constraint, so it converged on a policy
# that saves energy by serving 14.85 of 22.10 Mb/s offered. Comparing a learned
# policy against that partly credits the learner for doing more work -- the same
# flaw that made us discard the s=8 static baseline.
#
# This re-tunes with the selection changed to: LOWEST ENERGY SUBJECT TO SERVING
# AT LEAST 95% OF THE BEST THROUGHPUT ANY CANDIDATE ACHIEVES. That is what an
# operator deploying TWT would actually ask for, and it gives the hand-designed
# policy its strongest form before the learned policies are compared to it.
#
# Tuning on validation seeds 80000-80014 only; the winner is then evaluated on
# the 50 paired evaluation seeds, which the search never sees.
#
# Two winners are produced: one constrained to OFDMA-on (row 8 as specified --
# TWT and OFDMA both active) and one left free to choose its OFDMA mode, so we
# can report whether the best hand-designed policy would use OFDMA at all.
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# Repo root is one level up: scripts live in <root>/<topic>-scripts/.
ROOT="$(cd "$HERE/.." && pwd)"
# Override with:  PY=/path/to/python3.11 ./script.sh
PY="${PY:-python3}"
R="$ROOT/results/final/heur2"; LOG="$ROOT/results/heur2.log"
PAR=2; RESERVE_MB=2600
mkdir -p "$R"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
exec 5>"$ROOT/run/heur2.lock"; flock -n 5 || { say "another heur2 holds the lock"; exit 0; }
avail(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
SP="--sp-aligned --sleep-guard-us 0"

run(){ # tag seeds policy...
  local tag="$1" sds="$2" pol="$3"; shift 3
  local d="$R/$tag"; mkdir -p "$d"
  local need=14; [ "$sds" = "90000-90049" ] && need=48
  local have; have=$(ls "$d"/*.json 2>/dev/null|wc -l)
  [ "$have" -ge "$need" ] && return 0
  while [ "$(jobs -rp|wc -l)" -ge "$PAR" ] || [ "$(avail)" -lt "$RESERVE_MB" ]; do sleep 30; done
  say "  $tag"
  ( "$PY" "$ROOT/eval-scripts/evaluate.py" --policy "$pol" --seeds "$sds" --out "$d" \
      --seg "$tag" --progress-every 0 "$@" >> "$LOG" 2>&1 ) &
}

say "===== heuristic re-tune under a service floor ====="
say "STAGE 1  thresholds, schedule pair 11/4, OFDMA on, validation seeds"
for HI in 2800 3100 3400; do for LO in 2200 2450 2700; do
  run "t_${HI}_${LO}" 80000-80014 "latency:${HI}:${LO}:11:4:17:on" $SP
done; done
wait
say "STAGE 2  schedule pairs at the winning thresholds"
BEST=$("$PY" - "$R" <<'PYEOF'
import glob,json,os,sys,re
import numpy as np
R=sys.argv[1]; rows=[]
for d in glob.glob(os.path.join(R,"t_*")):
    J=[json.load(open(f)) for f in glob.glob(d+"/*.json")]
    if len(J)<10: continue
    rows.append((os.path.basename(d),np.mean([x["energy_mj_total"] for x in J])/1000,
                 np.mean([x["throughput_mbps"] for x in J])))
if not rows: print("3400_2700"); sys.exit()
tmax=max(r[2] for r in rows)
ok=[r for r in rows if r[2]>=0.95*tmax] or rows
for n,e,t in sorted(rows,key=lambda r:r[1]):
    print(f"    {n:14} {e:7.1f} J  {t:6.2f} Mb/s {'  <= meets floor' if t>=0.95*tmax else ''}",file=sys.stderr)
w=min(ok,key=lambda r:r[1])
print(f"    floor = 95% of {tmax:.2f} = {0.95*tmax:.2f} Mb/s -> {w[0]}",file=sys.stderr)
print(w[0].replace("t_",""))
PYEOF
)
HI=${BEST%%_*}; LO=${BEST##*_}
say "STAGE 2  winning thresholds hi=$HI lo=$LO"
for PAIRS in "11:4" "0:4" "11:8"; do
  run "p_${PAIRS/:/-}" 80000-80014 "latency:${HI}:${LO}:${PAIRS}:17:on" $SP
done
run "p_off" 80000-80014 "latency:${HI}:${LO}:11:4:17:off" $SP
wait
say "STAGE 3  evaluating the two winners on the 50 evaluation seeds"
run eval_ofdma_on 90000-90049 "latency:${HI}:${LO}:11:4:17:on"  $SP
run eval_free     90000-90049 "latency:${HI}:${LO}:11:4:17:off" $SP
wait
say "===== heur2 COMPLETE ====="
touch "$ROOT/results/HEUR2_DONE"
