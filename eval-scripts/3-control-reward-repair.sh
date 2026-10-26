#!/bin/bash
# =============================================================================
# run_origreward.sh -- the reward-correction control
# =============================================================================
# The paper claims repairing the saturated energy term improved the result. That
# is currently unproven: the corrected reward was trained at ten seeds, the
# original at one. Without the original repeated at the SAME seeds the
# improvement cannot be separated from initialisation luck.
#
# IQL only. It is the method the headline rests on, and running all five would
# quintuple the cost to answer a question about one claim.
#
#   original preset : stat_ofdma.npz, --preset energy  (0.25, 0.15, 0.60), the
#                     formulation whose energy term is saturated for 99.99% of
#                     transitions
#   corrected       : stat_e2.npz, already run, 107.3 +/- 2.6 J
#
# Same 10 training seeds, same FQE-best selection rule, same 50 paired seeds.
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# Repo root is one level up: scripts live in <root>/<topic>-scripts/.
ROOT="$(cd "$HERE/.." && pwd)"
# Override with:  PY=/path/to/python3.11 ./script.sh
PY="${PY:-python3}"
DATA="$ROOT/data/stat_ofdma.npz"; HOLD="$ROOT/data/stat_ofdma_holdout.npz"
ALPHA=$(tail -1 "$ROOT/results/best_alpha.txt" 2>/dev/null || echo 0.5)
R="$ROOT/results/final/origreward"; LOG="$ROOT/results/origreward.log"
mkdir -p "$R"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
exec 4>"$ROOT/run/origreward.lock"; flock -n 4 || { say "another origreward holds the lock"; exit 0; }
avail(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
source "$ROOT/lib/resumable.sh"; rr_init "$ROOT/state/origreward"
export RR_NEED_MB=1500 RR_RESERVE_MB=2500

say "STAGE A  training IQL on the ORIGINAL energy preset, seeds 42-51"
for S in 42 43 44 45 46 47 48 49 50 51; do
  OUT="$ROOT/data/model_orig_s${S}_iql.npz"
  [ -f "$OUT" ] && continue
  rr_run "o_${S}" "$HERE" "/dev/null" 0 14400 -- \
    env OMP_NUM_THREADS=2 "$PY" "$ROOT/offline-rl-scripts/train_offline.py" --algo iql \
      --data "$DATA" --epochs 200 --cql-alpha "$ALPHA" --target-clip 50 --seed "$S" \
      --patience 999 --hidden-dim 128 --preset energy \
      --milestone-dir "$ROOT/data/ms_orig_s${S}" --milestone-every 50 \
      --ckpt-dir "$ROOT/data/ck_orig_s${S}" --output "$OUT" \
    && say "  trained orig seed $S"
done

say "STAGE B  FQE selection per seed"
for S in 42 43 44 45 46 47 48 49 50 51; do
  [ -d "$ROOT/results/fqe_orig_s${S}" ] && continue
  MS="$ROOT/data/ms_orig_s${S}"; [ -d "$MS" ] || continue
  rm -rf "$ROOT/state/fqe" "$ROOT/results/fqe"
  HOLDOUT="$HOLD" FITDATA="$DATA" MSDIR="$MS" PARALLEL=4 ITERS=30000 \
    "$HERE/select_checkpoints.sh" >> "$LOG" 2>&1
  if [ -d "$ROOT/results/fqe" ] && [ "$(ls "$ROOT/results/fqe"|wc -l)" -gt 0 ]; then
    mv -f "$ROOT/results/fqe" "$ROOT/results/fqe_orig_s${S}"; say "  seed $S selected"
  else say "  FAILED: seed $S produced no rankings"; fi
done

say "STAGE C  evaluating on the 50 paired seeds"
PAR=4
for S in 42 43 44 45 46 47 48 49 50 51; do
  d="$R/s${S}"; mkdir -p "$d"
  [ "$(ls "$d"/*.json 2>/dev/null|wc -l)" -ge 48 ] && continue
  P=$("$PY" - "$HERE" "$S" <<'PYEOF'
import glob,json,os,sys
H,S=sys.argv[1],sys.argv[2]
rows=[]
for f in glob.glob(f"{H}/results/fqe_orig_s{S}/*.json"):
    try: rows+=json.load(open(f))
    except Exception: pass
best=None
for r in rows:
    n=r["policy"].split("/")[-1]
    if "_iql_" in n and (best is None or r["fqe_value"]>best[1]): best=(n,r["fqe_value"])
p=f"{H}/data/ms_orig_s{S}/{best[0]}" if best else ""
print(p if p and os.path.exists(p) else f"{H}/data/model_orig_s{S}_iql.npz")
PYEOF
)
  [ -f "$P" ] || { say "  seed $S: no weights"; continue; }
  while [ "$(jobs -rp|wc -l)" -ge "$PAR" ] || [ "$(avail)" -lt 2500 ]; do sleep 20; done
  say "  eval seed $S"
  ( "$PY" "$ROOT/eval-scripts/evaluate.py" --policy "learned:$P" --seeds 90000-90049 \
      --out "$d" --seg "og$S" --progress-every 0 >> "$LOG" 2>&1 ) &
done
wait
say "===== origreward COMPLETE ====="
touch "$ROOT/results/ORIGREWARD_DONE"
