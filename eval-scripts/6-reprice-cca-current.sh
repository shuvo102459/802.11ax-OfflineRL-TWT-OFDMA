#!/bin/bash
# =============================================================================
# run_cca66.sh -- the same comparison under a coherent power profile
# =============================================================================
# The scenario sets TX/RX/idle/sleep to 232/66/50/0.12 mA but leaves CCA-busy at
# ns-3's unset default of 273 mA. Sensing the medium therefore costs more than
# transmitting on it, which is not a realisable radio. Every absolute joule in
# the paper is conditional on that profile.
#
# This re-evaluates the arms the claims rest on at --ccaCurrentMa 66, charging
# carrier sense at the receive current. No retraining: the policies are the same
# weights the frozen results used, so this isolates the energy model.
#
# 3 non-learned arms (150 ep) + IQL at all 10 FQE-best fits (500 ep) = 650
# episodes, ~5-6 h at 4 workers. Resumable: an arm with >=48 episodes is skipped,
# so partial results are usable at any point.
#
# Writes only results/final/cca66v2/. Nothing the paper currently cites is
# touched.
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/shuvo/NS3-project/EHRL/bin/python3.11
R="$HERE/results/final/cca66v2"; LOG="$HERE/results/cca66v2.log"
mkdir -p "$R" "$HERE/run"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
exec 4>"$HERE/run/cca66v2.lock"; flock -n 4 || { say "another cca66v2 holds the lock"; exit 0; }
avail(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

SP="--sp-aligned --sleep-guard-us 0"
CCA="--cca-current-ma 66"
PAR=4

run(){ # run <tag> <policy> [extra flags]
  local tag="$1" pol="$2"; shift 2
  local d="$R/$tag"; mkdir -p "$d"
  if [ "$(ls "$d"/*.json 2>/dev/null | wc -l)" -ge 48 ]; then say "  $tag complete, skipping"; return; fi
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ] || [ "$(avail)" -lt 2500 ]; do sleep 20; done
  say "  launching $tag"
  ( "$PY" "$HERE/code/evaluate.py" --policy "$pol" --seeds 90000-90049 \
      --out "$d" --seg "c6${tag}" $CCA "$@" --progress-every 0 >> "$LOG" 2>&1 ) &
}

say "===== repricing CCA at the receive current (66 mA) ====="

# --- non-learned arms, exactly the policies the frozen rows used ---
run static_twt   "fixed:11:17:0" $SP
run static_ofdma "fixed:11:17:1" $SP
run handdesigned "latency:3100:2200:11:4:17:off" $SP

# --- IQL at all ten FQE-best checkpoints, so the repriced mean pairs with 107.3 +/- 2.6 ---
while IFS=$'\t' read -r S P; do
  [ -n "${P:-}" ] && [ -f "$P" ] || { say "  seed ${S:-?}: weights missing, skipped"; continue; }
  run "iql_s${S}" "learned:$P" $SP
done < <(cat "$HERE/results/iql_fqebest_paths.txt"; echo)   # trailing newline guard:
# a final line without \n makes `read` return non-zero and the body never runs,
# which silently dropped seed 51 on the first pass.

wait
say "===== cca66v2 COMPLETE ====="
touch "$HERE/results/CCA66V2_DONE"

"$PY" -I - <<'PYEOF' | tee -a "$LOG"
import glob, json
import numpy as np
from scipy import stats
H="/home/shuvo/NS3-project/ns-allinone-3.44/ns-3.44/contrib/ai/examples/MobiCom/twt/wcnc2027-v3-nonstationary"
def load(d):
    o={}
    for f in glob.glob(f"{H}/results/final/{d}/*.json"):
        try: r=json.load(open(f))
        except Exception: continue
        if r.get("steps")==42: o[r["seed"]]=r["energy_mj_total"]/1000
    return o
print("\n=== 273 mA (shipped) vs 66 mA (coherent), same policies, same 50 seeds ===")
pairs=[("static TWT","main/static_s11","cca66v2/static_twt"),
       ("static+OFDMA","rows/r3_static_ofdma","cca66v2/static_ofdma"),
       ("hand-designed","heur2/eval_free","cca66v2/handdesigned")]
out={}
for lab,a,b in pairs:
    A,B=load(a),load(b); k=sorted(set(A)&set(B))
    if not k: print(f"  {lab}: incomplete"); continue
    x=np.array([A[i] for i in k]); y=np.array([B[i] for i in k])
    out[lab]=(x.mean(),y.mean())
    print(f"  {lab:14} {x.mean():6.1f} -> {y.mean():6.1f} J  ({100*(y.mean()/x.mean()-1):+5.1f}%)  n={len(k)}")
# IQL across the ten fits
f273=[load(f"seeds10/s{s}_iql") for s in range(42,52)]
f66 =[load(d) for d in sorted(glob.glob(f"{H}/results/final/cca66v2/iql_s*"))]
f66 =[load(f"cca66v2/{d.split('/')[-1]}") for d in sorted(glob.glob(f"{H}/results/final/cca66v2/iql_s*"))]
f273=[d for d in f273 if d]; f66=[d for d in f66 if d]
if f66:
    m273=np.array([np.mean(list(d.values())) for d in f273])
    m66 =np.array([np.mean(list(d.values())) for d in f66])
    print(f"  {'IQL':14} {m273.mean():6.1f} +/- {m273.std(ddof=1):.1f} -> {m66.mean():6.1f} +/- {m66.std(ddof=1):.1f} J "
          f"({100*(m66.mean()/m273.mean()-1):+5.1f}%)  {len(m66)} fits")
    if "hand-designed" in out:
        hd=out["hand-designed"][1]
        print(f"\n  HEADLINE under the coherent profile: IQL {m66.mean():.1f} J vs hand-designed {hd:.1f} J "
              f"= {100*(m66.mean()/hd-1):+.1f}%")
    if "static TWT" in out and "static+OFDMA" in out:
        a,b=out["static TWT"][1],out["static+OFDMA"][1]
        print(f"  OFDMA penalty under the coherent profile: {100*(b/a-1):+.1f}% (was +62.8%)")
PYEOF
