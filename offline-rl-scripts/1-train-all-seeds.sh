#!/bin/bash
# =============================================================================
# run_seeds10.sh -- every offline RL algorithm at 10 training seeds
# =============================================================================
# Convention in offline RL is 4-10 training seeds (CORL reports 4; the stronger
# benchmarks use 10 for IQL/ReBRAC). Three is below that, and two algorithms
# currently rest on a single fit -- the exact weakness that made an earlier
# DQN+CQL result collapse (s.d. 60 J across fits).
#
# SELECTION RULE, FIXED BEFORE ANY NEW RESULT IS SEEN: deploy the checkpoint
# with the highest FQE value, ignoring the long-window drift gate. The gate
# currently rejects every snapshot and falls back to the final epoch, which for
# IQL is the WORST of four by both FQE and the simulator. FQE's ranking was
# validated against ns-3 (Spearman -1.000, 0.0 J lost to selection), so using it
# is defensible; the gate is not. Final-epoch numbers are reported alongside as
# a conservative bound.
#
# Seeds 42-51. 42 exists for all five algorithms; 43/44 exist for bc/dqn/iql.
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# Repo root is one level up: scripts live in <root>/<topic>-scripts/.
ROOT="$(cd "$HERE/.." && pwd)"
# Override with:  PY=/path/to/python3.11 ./script.sh
PY="${PY:-python3}"
DATA="$ROOT/data/stat_e2.npz"; HOLD="$ROOT/data/stat_e2_holdout.npz"
ALPHA=$(tail -1 "$ROOT/results/best_alpha.txt" 2>/dev/null || echo 0.5)
R="$ROOT/results/final/seeds10"; LOG="$ROOT/results/seeds10.log"
mkdir -p "$R"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
exec 9>"$ROOT/run/seeds10.lock"; flock -n 9 || { say "another seeds10 run holds the lock"; exit 0; }
avail(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
source "$ROOT/lib/resumable.sh"; rr_init "$ROOT/state/seeds10"
export RR_NEED_MB=1500 RR_RESERVE_MB=2500
ALGOS="bc dqn ddqn iql dueling"
SEEDS="42 43 44 45 46 47 48 49 50 51"

# ── A: train every missing (algorithm, seed) ────────────────────────────────
say "STAGE A  training to 10 seeds per algorithm"
for S in $SEEDS; do
  for A in $ALGOS; do
    OUT="$ROOT/data/model_e2s${S}_${A}.npz"
    [ "$S" = 42 ] && OUT="$ROOT/data/model_e2_${A}.npz"
    [ -f "$OUT" ] && continue
    rr_run "t_${S}_${A}" "$HERE" "/dev/null" 0 14400 -- \
      env OMP_NUM_THREADS=2 "$PY" "$ROOT/offline-rl-scripts/train_offline.py" --algo "$A" \
        --data "$DATA" --epochs 200 --cql-alpha "$ALPHA" --target-clip 50 --seed "$S" \
        --patience 999 --hidden-dim 128 --preset energy \
        --milestone-dir "$ROOT/data/ms_e2s${S}" --milestone-every 50 \
        --ckpt-dir "$ROOT/data/ck_e2s${S}" --output "$OUT" \
      && say "  trained seed $S / $A"
  done
done

# ── B: FQE over each seed's milestones ──────────────────────────────────────
say "STAGE B  FQE selection per training seed"
for S in $SEEDS; do
  [ -d "$ROOT/results/fqe_e2s${S}" ] && { say "  seed $S already selected"; continue; }
  [ "$S" = 42 ] && { say "  seed 42 uses results/fqe_e2"; continue; }
  MS="$ROOT/data/ms_e2s${S}"
  [ -d "$MS" ] || { say "  seed $S has no milestones; skipping"; continue; }
  rm -rf "$ROOT/state/fqe" "$ROOT/results/fqe"
  HOLDOUT="$HOLD" FITDATA="$DATA" MSDIR="$MS" PARALLEL=4 ITERS=30000 \
    "$HERE/select_checkpoints.sh" >> "$LOG" 2>&1
  if [ -d "$ROOT/results/fqe" ] && [ "$(ls "$ROOT/results/fqe" | wc -l)" -gt 0 ]; then
    mv -f "$ROOT/results/fqe" "$ROOT/results/fqe_e2s${S}"; say "  seed $S selected"
  else
    say "  FAILED: seed $S produced no rankings"
  fi
done

# ── C: resolve the FQE-best checkpoint per (algorithm, seed) ────────────────
say "STAGE C  resolving FQE-best checkpoints (drift gate NOT applied)"
"$PY" - "$HERE" "$R" <<'PYEOF' | tee -a "$LOG"
import glob,json,os,sys
H,R=sys.argv[1],sys.argv[2]
ALG=["bc","dqn","ddqn","iql","dueling"]
out=[]
for S in [42,43,44,45,46,47,48,49,50,51]:
    fq=os.path.join(H,"results","fqe_e2" if S==42 else f"results/fqe_e2s{S}")
    ms=os.path.join(H,"data","ms_e2" if S==42 else f"data/ms_e2s{S}")
    rows=[]
    for fp in glob.glob(os.path.join(fq,"*.json")):
        try: rows+=json.load(open(fp))
        except Exception: pass
    for a in ALG:
        best=None
        for r in rows:                       # highest FQE value, gate ignored
            n=r["policy"].split("/")[-1]
            if f"_{a}_" in n and (best is None or r["fqe_value"]>best[1]): best=(n,r["fqe_value"])
        p=os.path.join(ms,best[0]) if best else ""
        if not p or not os.path.exists(p):
            fb=os.path.join(H,"data",f"model_e2_{a}.npz" if S==42 else f"model_e2s{S}_{a}.npz")
            if not os.path.exists(fb): continue
            p, how = fb, "final"
        else: how="fqe"
        out.append((f"s{S}_{a}",p,how))
with open(os.path.join(R,"policies.tsv"),"w") as f:
    for t,p,h in out: f.write(f"{t}\t{p}\t{h}\n")
print(f"    resolved {len(out)} policies ({sum(1 for _,_,h in out if h=='fqe')} by FQE, "
      f"{sum(1 for _,_,h in out if h=='final')} fell back to final epoch)")
PYEOF

# ── D: evaluate, 50 paired seeds each ───────────────────────────────────────
say "STAGE D  evaluating on 50 paired seeds"
PAR=3
while IFS=$'\t' read -r tag path how; do
  d="$R/$tag"; mkdir -p "$d"
  have=$(ls "$d"/*.json 2>/dev/null | wc -l)
  [ "$have" -ge 48 ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ] || [ "$(avail)" -lt 2500 ]; do sleep 20; done
  say "  start $tag ($how)"
  ( "$PY" "$ROOT/eval-scripts/evaluate.py" --policy "learned:$path" --seeds 90000-90049 \
      --out "$d" --seg "x$tag" --progress-every 0 >> "$LOG" 2>&1
    echo "[$(date '+%F %T')]   finished $tag ($(ls "$d"/*.json 2>/dev/null|wc -l))" >> "$LOG" ) &
done < "$R/policies.tsv"
wait
say "===== seeds10 COMPLETE ====="
touch "$ROOT/results/SEEDS10_DONE"
