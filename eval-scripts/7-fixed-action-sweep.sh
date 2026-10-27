#!/bin/bash
# =============================================================================
# run_fixedsweep.sh -- the best fixed (schedule, grouping) policy
# =============================================================================
# Both reviewers ask the same question: IQL and the tuned controller BOTH
# disable uplink OFDMA, so the 59 J gap between them cannot come from the OFDMA
# bit. Does IQL beat the best FIXED schedule/grouping, or does it merely
# discover a good static configuration?
#
# Only 5 of 20 schedules were ever searched, all at assignment 17. This
# completes the sweep on the TUNING seeds (80000-80014, disjoint from the 50
# evaluation seeds), then the Pareto-relevant configurations are evaluated on
# the untouched evaluation set by stage 2.
#
# Stage 1: 15 remaining schedules x assignment 17, OFDMA off  = 225 episodes
# Stage 2: for the best schedules, 4 further assignments      = ~180 episodes
# Writes only results/final/fixsweep/.
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/shuvo/NS3-project/EHRL/bin/python3.11
R="$HERE/results/final/fixsweep"; LOG="$HERE/results/fixsweep.log"
mkdir -p "$R" "$HERE/run"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
exec 4>"$HERE/run/fixsweep.lock"; flock -n 4 || { say "another fixsweep holds the lock"; exit 0; }
avail(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
SP="--sp-aligned --sleep-guard-us 0"
PAR=4
TUNE=80000-80014

run(){ # run <tag> <policy> <seeds>
  local tag="$1" pol="$2" sd="$3"
  local d="$R/$tag"; mkdir -p "$d"
  local want; want=$(( $(echo "$sd"|cut -d- -f2) - $(echo "$sd"|cut -d- -f1) + 1 ))
  if [ "$(ls "$d"/*.json 2>/dev/null | wc -l)" -ge "$want" ]; then say "  $tag complete"; return; fi
  while [ "$(jobs -rp|wc -l)" -ge "$PAR" ] || [ "$(avail)" -lt 2500 ]; do sleep 20; done
  say "  $tag  ($pol)"
  ( "$PY" "$HERE/code/evaluate.py" --policy "$pol" --seeds "$sd" --out "$d" \
      --seg "fx${tag}" $SP --progress-every 0 >> "$LOG" 2>&1 ) &
}

say "===== STAGE 1: all 20 schedules at assignment 17, OFDMA off, tuning seeds ====="
for S in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
  run "s${S}_a17" "fixed:${S}:17:0" "$TUNE"
done
wait
say "===== STAGE 1 complete ====="
touch "$HERE/results/FIXSWEEP_S1_DONE"

"$PY" -I - <<'PYEOF' | tee -a "$LOG"
import glob, json, os
import numpy as np
H=os.path.dirname(os.path.dirname(os.path.abspath("x")))
R="results/final/fixsweep"
rows=[]
for d in sorted(glob.glob(R+"/s*_a17")):
    rs=[json.load(open(f)) for f in glob.glob(d+"/*.json")]
    rs=[r for r in rs if r.get("steps")==42]
    if len(rs)>=14:
        rows.append((os.path.basename(d), np.mean([r["energy_mj_total"] for r in rs])/1000,
                     np.mean([r["throughput_mbps"] for r in rs]), len(rs)))
rows.sort(key=lambda r:r[1])
print("\n=== fixed schedules on TUNING seeds (assignment 17, OFDMA off) ===")
print(f"{'config':12}{'E (J)':>9}{'tput':>9}{'n':>4}")
for r in rows: print(f"  {r[0]:12}{r[1]:9.1f}{r[2]:9.2f}{r[3]:>4}")
# Pareto frontier: minimal energy at or above each throughput level
par=[]
for r in sorted(rows,key=lambda x:-x[2]):
    if not par or r[1]<par[-1][1]: par.append(r)
print("\n  Pareto-efficient (lower energy, higher throughput):")
for r in par: print(f"    {r[0]:12}{r[1]:9.1f} J {r[2]:8.2f} Mb/s")
print("\n  IQL on EVAL seeds for reference: 107.3 J at 20.76 Mb/s")
PYEOF
