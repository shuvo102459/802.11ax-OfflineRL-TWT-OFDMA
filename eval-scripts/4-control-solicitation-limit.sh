#!/bin/bash
# =============================================================================
# run_nsta9.sh -- does the TWT/OFDMA substitution survive a higher
#                 multiplexing degree?
# =============================================================================
# Every OFDMA arm in the paper ran at ns-3's RrMultiUserScheduler default of
# NStations=4, which was never set deliberately and never reported. A 20 MHz
# channel admits nine 26-tone RUs, so the trigger overhead was amortised over
# four uplinks instead of nine. The claim "TWT and uplink OFDMA substitute"
# therefore rests on an unexamined default -- the same failure mode as the
# inherited 273 mA CCA current.
#
# This re-runs the two arms where the degree can matter, at N=9, on the same 50
# paired seeds with identical flags otherwise. The N=4 counterparts are already
# frozen, so each pair differs in exactly one parameter:
#
#   static TWT + OFDMA   fixed:11:17:1              vs r3_static_twt_ofdma (282.2 J)
#   hand-designed, on    latency:3100:2200:11:4:17:on  vs r8 (265.5 J)
#
# The no-OFDMA controls (r2 at 173.3 J, r8b at 166.5 J) are unaffected by
# NStations and are reused as-is.
#
# Reads: nothing the paper depends on. Writes: results/final/nsta9/ only.
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# Repo root is one level up: scripts live in <root>/<topic>-scripts/.
ROOT="$(cd "$HERE/.." && pwd)"
# Override with:  PY=/path/to/python3.11 ./script.sh
PY="${PY:-python3}"
R="$ROOT/results/final/nsta9"; LOG="$ROOT/results/nsta9.log"
mkdir -p "$R" "$ROOT/run"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

exec 4>"$ROOT/run/nsta9.lock"; flock -n 4 || { say "another nsta9 holds the lock"; exit 0; }
avail(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

# Wait for the reward-correction run to finish before competing for memory.
if [ ! -f "$ROOT/results/ORIGREWARD_DONE" ]; then
  say "waiting for origreward to finish before starting"
  while [ ! -f "$ROOT/results/ORIGREWARD_DONE" ]; do sleep 120; done
  say "origreward done; proceeding"
fi

SP="--sp-aligned --sleep-guard-us 0"
PAR=4

run(){  # run <tag> <policy>
  local tag="$1"; shift
  local pol="$1"; shift
  local d="$R/$tag"; mkdir -p "$d"
  local have; have=$(ls "$d"/*.json 2>/dev/null | wc -l)
  if [ "$have" -ge 48 ]; then say "  $tag already has $have episodes, skipping"; return; fi
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ] || [ "$(avail)" -lt 2500 ]; do sleep 20; done
  say "  launching $tag  ($pol, nStations=9)"
  ( "$PY" "$ROOT/eval-scripts/evaluate.py" --policy "$pol" --seeds 90000-90049 \
       --out "$d" --seg "n9${tag}" --n-stations 9 $SP --progress-every 0 \
       >> "$LOG" 2>&1 ) &
}

say "===== nStations=9 re-run of the OFDMA arms, 50 paired seeds ====="
run static_ofdma "fixed:11:17:1"
run heur_ofdma   "latency:3100:2200:11:4:17:on"
wait
say "===== nsta9 COMPLETE ====="
touch "$ROOT/results/NSTA9_DONE"

"$PY" -I - <<'PYEOF' | tee -a "$LOG"
import glob, json
import numpy as np
from scipy import stats
H="$ROOT"
def load(d):
    o={}
    for f in glob.glob(f"{H}/results/final/{d}/*.json"):
        try: r=json.load(open(f))
        except Exception: continue
        if r.get("steps")==42: o[r["seed"]]=(r["energy_mj_total"]/1000, r["throughput_mbps"])
    return o
print("\n=== does the OFDMA penalty survive N=9? (paired, same seeds) ===")
for lab, off, on4, on9 in (
    ("static TWT",    "main/static_s11",  "rows/r3_static_ofdma", "nsta9/static_ofdma"),
    ("hand-designed", "heur2/eval_free",  "heur2/eval_ofdma_on",  "nsta9/heur_ofdma")):
    A,B,C = load(off), load(on4), load(on9)
    k = sorted(set(A)&set(B)&set(C))
    if not k: print(f"  {lab}: incomplete"); continue
    a=np.array([A[i][0] for i in k]); b=np.array([B[i][0] for i in k]); c=np.array([C[i][0] for i in k])
    ta=np.mean([A[i][1] for i in k]); tc=np.mean([C[i][1] for i in k])
    print(f"  {lab:14} OFDMA off {a.mean():6.1f} J | N=4 {b.mean():6.1f} J ({100*(b.mean()/a.mean()-1):+5.1f}%)"
          f" | N=9 {c.mean():6.1f} J ({100*(c.mean()/a.mean()-1):+5.1f}%)  n={len(k)}")
    print(f"  {'':14} N=9 vs N=4: {100*(c.mean()/b.mean()-1):+5.1f}%  p={stats.wilcoxon(c,b).pvalue:.2g}"
          f" | tput {ta:.2f} -> {tc:.2f} Mb/s")
    print(f"  {'':14} VERDICT: OFDMA at N=9 is "
          f"{'STILL a cost' if c.mean()>a.mean() else 'now a SAVING -- claim reverses'}")
PYEOF
