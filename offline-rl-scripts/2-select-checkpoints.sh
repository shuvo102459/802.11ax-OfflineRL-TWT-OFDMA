#!/bin/bash
# =============================================================================
# select_checkpoints.sh -- FQE checkpoint selection, parallelised
# =============================================================================
# FQE costs 24.4 min per policy at 40,000 iterations, single-threaded, measured
# on this machine. Selecting across 10 arms x 5 snapshots = 50 policies would be
# ~20 h serially, which does not fit the budget. The runs are independent, so
# they go N-at-a-time instead: 50 policies / 5 parallel ~ 4 h.
#
# Each job is single-threaded ON PURPOSE (OMP_NUM_THREADS=1). Letting torch take
# all 8 cores per job and then running 5 jobs oversubscribes the box ~5x and is
# slower than this, besides starving anything else running.
#
# ITERS=30000 comes from the measured convergence curve, not a guess. On the pilot
# holdout V_hat went -6.42 (2k) -> -12.60 (10k) -> -13.33 (14k) -> -13.99 (30k) ->
# -14.21 (40k). At 30k it is within 1.6% of the 40k value.
#
# The gate is LONG-window drift (< 2% over 10,000 iterations), not the 2,000-step
# figure: that fell below 2% at ~14k while V_hat still had 6.6% to travel. An
# earlier version of this script gated on the short window and would have accepted
# those estimates.
#
# Resumable per policy: an interrupted sweep continues where it stopped.
# =============================================================================
set -u
PARALLEL="${PARALLEL:-5}"
ITERS="${ITERS:-30000}"
HOLDOUT="${HOLDOUT:?set HOLDOUT=data/full_constrained_holdout.npz}"
FITDATA="${FITDATA:?set FITDATA=data/full_constrained.npz}"
MSDIR="${MSDIR:?set MSDIR=the milestone directory to rank}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# Repo root is one level up: scripts live in <root>/<topic>-scripts/.
ROOT="$(cd "$HERE/.." && pwd)"
# Override with:  PY=/path/to/python3.11 ./script.sh
PY="${PY:-python3}"
OUT="$ROOT/results/fqe"
source "$ROOT/lib/resumable.sh"
rr_init "$ROOT/state/fqe"
rr_guard "$ROOT/run/fqe.lock"
export RR_NEED_MB=1200 RR_RESERVE_MB=3000
mkdir -p "$OUT" "$ROOT/results"
LOG="$ROOT/results/fqe_select.log"
echo "=== FQE selection (re)started $(date): iters=$ITERS parallel=$PARALLEL ===" | tee -a "$LOG"

mapfile -t POLS < <(ls "$MSDIR"/*.npz 2>/dev/null)
[ "${#POLS[@]}" -gt 0 ] || { echo "no snapshots in $MSDIR"; exit 1; }
echo "    ${#POLS[@]} snapshots to rank" | tee -a "$LOG"

K=0; PIDS=()
for P in "${POLS[@]}"; do
  KEY="fqe_$(basename "$P" .npz)"
  if ! rr_done "$KEY"; then
    # one policy per job, its own json; aggregated after
    rr_run "$KEY" "$HERE" "/dev/null" 0 7200 -- \
      env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$ROOT/offline-rl-scripts/fqe.py" \
        --data "$HOLDOUT" --fit-data "$FITDATA" --policies "$P" \
        --iters "$ITERS" --log-every 2000 --trace-convergence \
        --out "$OUT/$KEY.json" &
    PIDS+=($!); K=$((K+1))
  fi
  if [ "$K" -ge "$PARALLEL" ]; then for Q in "${PIDS[@]}"; do wait "$Q"; done; PIDS=(); K=0; fi
done
for Q in "${PIDS[@]:-}"; do wait "$Q" 2>/dev/null; done

echo | tee -a "$LOG"
echo "=== ranking (LONG-window drift must be < 2% for a row to count) ===" | tee -a "$LOG"
"$PY" - "$OUT" <<'PYEOF' | tee -a "$LOG"
import json, glob, os, sys
rows = []
for fp in glob.glob(os.path.join(sys.argv[1], "*.json")):
    rows += json.load(open(fp))
if not rows:
    print("  no results"); raise SystemExit
def _ld(r):
    v = r.get("long_drift")
    return v if v is not None and v == v else 1.0   # NaN/missing -> not converged
ok  = [r for r in rows if _ld(r) < 0.02]
bad = [r for r in rows if _ld(r) >= 0.02]
for r in sorted(ok, key=lambda x: -x["fqe_value"]):
    print(f"  {r['policy']:52s} V_hat {r['fqe_value']:+8.3f} +/- {r['fqe_se']:.3f}  "
          f"drift {100*_ld(r):4.2f}% over {r.get('long_window_iters',0):,}  "
          f"OFDMA {100*r['ofdma_share']:5.1f}%")
if bad:
    print(f"\n  EXCLUDED -- drift >= 2%, estimate not converged ({len(bad)} of {len(rows)}):")
    for r in sorted(bad, key=_ld):
        print(f"    {r['policy']:50s} long-drift {100*_ld(r):5.2f}%  "
              f"(V_hat {r['fqe_value']:+.3f}, not usable)")
    print("  Raise --iters for these before selecting; an unconverged V_hat is noise.")
if ok:
    best = max(ok, key=lambda x: x["fqe_value"])
    print(f"\n  SELECTED: {best['policy']}  V_hat {best['fqe_value']:+.3f}")
PYEOF
echo "=== FQE selection finished $(date) ===" | tee -a "$LOG"
