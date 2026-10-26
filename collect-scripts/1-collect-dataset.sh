#!/bin/bash
# =============================================================================
# run_final.sh -- the single experiment every claim in the paper will cite
# =============================================================================
# ONE harness, ONE window (42-decision RL window), ONE evaluation seed set
# (90000-90049, paired across every arm), disjoint from collection (30000+) and
# from the heuristic's tuning seeds (80000-80014).
#
# The main comparison runs OFDMA OFF: with the bit clear the 8 ms sleep guard
# never fires, so the headline is structurally free of the workaround rather
# than corrected for it. OFDMA is its own experiment, run service-period
# aligned with the guard disabled.
#
# PARALLELISM. evaluate.py runs one episode at a time (ns3ai's Experiment is a
# singleton per process), so a single invocation yields ~13 episodes/h at the
# measured 4.6 min each. Arms are therefore run through a worker pool. The pool
# is sized by MEMORY, not cores: an always-awake episode holds ~2.8 GB and five
# concurrent ones drove this machine to 293 MB once already.
#
# TIERED. Tier 1 is everything the headline needs; tier 2 adds breadth; tier 3
# is OFDMA, the power-profile sweep and cadence. If the run is cut short the
# essential arms are already complete.
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# Repo root is one level up: scripts live in <root>/<topic>-scripts/.
ROOT="$(cd "$HERE/.." && pwd)"
# Override with:  PY=/path/to/python3.11 ./script.sh
PY="${PY:-python3}"
# contrib/ai/examples/MobiCom/twt/<repo> -> up 6 to the ns-3 root; override with NS3=...
NS3="${NS3:-$(cd "$ROOT/../../../../../.." && pwd)}"
R="$ROOT/results/final"; LOG="$ROOT/results/final.log"
PAR="${PAR:-3}"; RESERVE_MB="${RESERVE_MB:-2500}"
mkdir -p "$R"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
exec 7>"$ROOT/run/final.lock"; flock -n 7 || { say "another final run holds the lock"; exit 0; }
avail(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

job(){ # <dir> <seg> <policy> [extra...]
  local tag="$1" d="$R/$1" seg="$2" pol="$3"; shift 3
  mkdir -p "$d"
  local have; have=$(ls "$d"/*.json 2>/dev/null | wc -l)
  [ "$have" -ge "${NEED:-48}" ] && { say "  $tag complete ($have)"; return 0; }
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ] || [ "$(avail)" -lt "$RESERVE_MB" ]; do sleep 20; done
  say "  start $tag  ($pol $*)"
  ( "$PY" "$ROOT/eval-scripts/evaluate.py" --policy "$pol" --seeds "${SEEDS:-90000-90049}" \
      --out "$d" --seg "$seg" --progress-every 0 "$@" >> "$LOG" 2>&1
    echo "[$(date '+%F %T')]   finished $tag ($(ls "$d"/*.json 2>/dev/null | wc -l))" >> "$LOG" ) &
}
drain(){ wait; }

say "===== FINAL RUN (pool=$PAR) ====="

# ── Stage 0: static baseline chosen on VALIDATION seeds ──────────────────────
say "STAGE 0  static-schedule selection, validation seeds 80000-80014"
SEEDS=80000-80014 NEED=14
for S in 4 11 8 17 0; do job "sel/s$S" "sel$S" "fixed:$S:17:0"; done
drain
unset SEEDS NEED
BEST=$("$PY" - "$R/sel" <<'PYEOF'
import glob,json,os,sys
import numpy as np
best=(None,1e18)
for d in sorted(glob.glob(os.path.join(sys.argv[1],"s*"))):
    E=[json.load(open(f))["energy_mj_total"] for f in glob.glob(os.path.join(d,"*.json"))]
    if len(E)<10: continue
    m=float(np.mean(E)); print(f"    {os.path.basename(d)}: {m/1000:7.1f} J  n={len(E)}",file=sys.stderr)
    if m<best[1]: best=(os.path.basename(d)[1:],m)
print(best[0] or "11")
PYEOF
)
say "STAGE 0  selected static schedule s=$BEST"
BESTIQL=$(ls "$HERE"/data/ms_e2/*iql*ep50.npz 2>/dev/null | head -1)
"$PY" "$ROOT/offline-rl-scripts/resolve_policies.py" > "$R/policies.tsv" 2>"$R/policies.err" || true

# ── Tier 1: everything the headline needs ────────────────────────────────────
say "TIER 1  headline arms"
job main/static m_static "fixed:${BEST}:17:0"
job main/notwt  m_notwt  "alwayson"
job main/heur   m_heur   "latency:3400:2700:11:4:17:off"
[ -n "$BESTIQL" ] && job main/iql_fqebest m_iqlb "learned:$BESTIQL"
for t in 43 44; do
  p=$(awk -F'\t' -v k="e2s${t}_iql" '$1==k{print $2}' "$R/policies.tsv")
  [ -n "${p:-}" ] && job "main/iql_s$t" "m_iql$t" "learned:$p"
done
drain
say "TIER 1 complete"

# ── Tier 2: breadth across algorithms and fits ───────────────────────────────
say "TIER 2  remaining learned arms"
job main/random m_rand "randomtwt:0"
while IFS=$'\t' read -r tag path how v; do
  [ -n "${path:-}" ] || continue
  case "$tag" in e2s43_iql|e2s44_iql) continue;; esac
  job "main/$tag" "m_$tag" "learned:$path"
done < "$R/policies.tsv"
drain
say "TIER 2 complete"

# ── Tier 3: OFDMA, power profile, cadence ────────────────────────────────────
say "TIER 3  OFDMA / profile / cadence"
job ofdma/heur_sp    o_hsp "latency:3400:2700:11:4:17:on" --sp-aligned --sleep-guard-us 0
job ofdma/heur_naive o_hnv "latency:3400:2700:11:4:17:on"
[ -n "$BESTIQL" ] && job ofdma/iql_sp o_isp "learned:$BESTIQL" --sp-aligned --sleep-guard-us 0
drain
job cca66/static c_static "fixed:${BEST}:17:0"            --cca-current-ma 66
job cca66/heur   c_heur   "latency:3400:2700:11:4:17:off" --cca-current-ma 66
[ -n "$BESTIQL" ] && job cca66/iql c_iql "learned:$BESTIQL" --cca-current-ma 66
drain
[ -n "$BESTIQL" ] && job cadence/iql_hold10 d_iql "learned:$BESTIQL" --hold-steps 10
drain

say "harvesting per-STA logs"
mkdir -p "$R/csv"; m=0
for f in "$NS3"/{m_,o_,c_,d_,sel}*9*_persta.csv; do [ -f "$f" ] && { mv -f "$f" "$R/csv/"; m=$((m+1)); }; done
rm -f "$NS3"/{m_,o_,c_,d_,sel}*9*_{overall,actions,rewards}.csv 2>/dev/null
say "collected $m per-STA logs"
touch "$ROOT/results/FINAL_DONE"
say "===== FINAL RUN COMPLETE ====="
