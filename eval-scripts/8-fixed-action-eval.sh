#!/bin/bash
# =============================================================================
# run_fixsweep2.sh -- stage 2: the Pareto-best fixed schedules on EVAL seeds
# =============================================================================
# Stage 1 ranks all 20 schedules on the TUNING seeds (80000-80014). Comparing
# those directly against IQL's 107.3 J would be a cross-seed-set comparison and
# is not admissible. This re-evaluates the Pareto-efficient schedules -- plus
# the single cheapest and the cheapest meeting the service floor -- on the
# untouched evaluation seeds (90000-90049), so the comparison is like for like.
#
# Selection uses tuning data only; evaluation seeds are never used to choose.
# Waits for stage 1, then runs ~3-4 arms x 50 seeds.
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/shuvo/NS3-project/EHRL/bin/python3.11
R="$HERE/results/final/fixsweep_eval"; LOG="$HERE/results/fixsweep2.log"
mkdir -p "$R" "$HERE/run"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
exec 4>"$HERE/run/fixsweep2.lock"; flock -n 4 || { say "another fixsweep2 holds the lock"; exit 0; }
avail(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

if [ ! -f "$HERE/results/FIXSWEEP_S1_DONE" ]; then
  say "waiting for stage 1"
  while [ ! -f "$HERE/results/FIXSWEEP_S1_DONE" ]; do sleep 120; done
fi
say "stage 1 done; choosing the Pareto set from TUNING data only"

mapfile -t PICK < <("$PY" -I - <<'PYEOF'
import glob, json, os
import numpy as np
rows=[]
for d in sorted(glob.glob("results/final/fixsweep/s*_a17")):
    rs=[json.load(open(f)) for f in glob.glob(d+"/*.json")]
    rs=[r for r in rs if r.get("steps")==42]
    if len(rs)>=14:
        sid=os.path.basename(d).replace("_a17","").lstrip("s")
        rows.append((sid, np.mean([r["energy_mj_total"] for r in rs])/1000,
                     np.mean([r["throughput_mbps"] for r in rs])))
if not rows: raise SystemExit
tmax=max(r[2] for r in rows)
pick=set()
pick.add(min(rows,key=lambda r:r[1])[0])                                  # cheapest overall
floor=[r for r in rows if r[2] >= 0.95*tmax]                              # meets service floor
if floor: pick.add(min(floor,key=lambda r:r[1])[0])
# Pareto frontier: no other config is both cheaper and faster
for r in rows:
    if not any((o[1] < r[1] and o[2] > r[2]) for o in rows): pick.add(r[0])
# Rank by INFORMATIVENESS, not schedule id: the arms that matter are the one
# nearest IQL's energy, the one nearest its throughput, the cheapest, and the
# cheapest meeting the floor. Truncating a numerically sorted list drops them.
IQL_E, IQL_T = 107.3, 20.76
key=[]
key.append(min(rows,key=lambda r:abs(r[1]-IQL_E))[0])   # energy-matched
key.append(min(rows,key=lambda r:abs(r[2]-IQL_T))[0])   # throughput-matched
key.append(min(rows,key=lambda r:r[1])[0])              # cheapest
if floor: key.append(min(floor,key=lambda r:r[1])[0])   # cheapest meeting floor
out=[]
for s in key:
    if s not in out: out.append(s)
for s in out[:4]: print(s)
PYEOF
)
say "selected schedules: ${PICK[*]}"

SP="--sp-aligned --sleep-guard-us 0"; PAR=4
for S in "${PICK[@]}"; do
  [ -n "$S" ] || continue
  d="$R/s${S}_a17"; mkdir -p "$d"
  [ "$(ls "$d"/*.json 2>/dev/null|wc -l)" -ge 48 ] && { say "  s$S complete"; continue; }
  while [ "$(jobs -rp|wc -l)" -ge "$PAR" ] || [ "$(avail)" -lt 2500 ]; do sleep 20; done
  say "  eval s$S on 90000-90049"
  ( "$PY" "$HERE/code/evaluate.py" --policy "fixed:${S}:17:0" --seeds 90000-90049 \
      --out "$d" --seg "fe${S}" $SP --progress-every 0 >> "$LOG" 2>&1 ) &
done
wait
say "===== fixsweep2 COMPLETE ====="
touch "$HERE/results/FIXSWEEP_S2_DONE"

"$PY" -I - <<'PYEOF' | tee -a "$LOG"
import glob, json, os
import numpy as np
from scipy import stats
def load(d):
    o={}
    for f in glob.glob(f"results/final/{d}/*.json"):
        try: r=json.load(open(f))
        except Exception: continue
        if r.get("steps")==42: o[r["seed"]]=(r["energy_mj_total"]/1000, r["throughput_mbps"])
    return o
iql=[load(f"seeds10/s{s}_iql") for s in range(42,52)]; iql=[d for d in iql if d]
print("\n=== fixed schedules vs IQL, SAME 50 evaluation seeds ===")
print(f"{'policy':16}{'E (J)':>9}{'tput':>9}   vs IQL")
best=None
for d in sorted(glob.glob("results/final/fixsweep_eval/s*_a17")):
    o=load("fixsweep_eval/"+os.path.basename(d))
    if len(o)<45: continue
    k=sorted(set(o)&set.intersection(*[set(x) for x in iql]))
    e=np.array([o[i][0] for i in k]); t=np.array([o[i][1] for i in k])
    qe=np.array([np.mean([x[i][0] for x in iql]) for i in k])
    qt=np.array([np.mean([x[i][1] for x in iql]) for i in k])
    p=stats.wilcoxon(qe,e).pvalue
    print(f"  {os.path.basename(d):16}{e.mean():9.1f}{t.mean():9.2f}   IQL {100*(qe.mean()/e.mean()-1):+6.1f}% energy, "
          f"{100*(qt.mean()/t.mean()-1):+6.1f}% tput, p={p:.2g}")
    if best is None or e.mean()<best[1]: best=(os.path.basename(d), e.mean(), t.mean())
if best and iql:
    k=sorted(set.intersection(*[set(x) for x in iql]))
    qe=np.mean([np.mean([x[i][0] for x in iql]) for i in k]); qt=np.mean([np.mean([x[i][1] for x in iql]) for i in k])
    print(f"\n  IQL             {qe:9.1f}{qt:9.2f}")
    print(f"  -> cheapest fixed is {best[0]} at {best[1]:.1f} J / {best[2]:.2f} Mb/s")
    print(f"  -> IQL {'DOMINATES (cheaper AND faster)' if qe<best[1] and qt>best[2] else 'does NOT dominate -- narrow the claim'}")
PYEOF
