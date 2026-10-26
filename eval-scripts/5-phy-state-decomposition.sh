#!/bin/bash
# =============================================================================
# run_phystate.sh -- PHY-state energy decomposition + the sleep-guard control
# =============================================================================
# Answers two reviewer demands with one set of runs.
#
# (1) ENERGY DECOMPOSITION. The paper claims OFDMA's cost falls inside the awake
#     period but could only show an aggregate. The rebuilt scenario now logs
#     time in each PHY state and the energy the configured currents attribute to
#     it, so E_total can be split into idle / CCA / TX / RX / sleep.
#
# (2) THE SLEEP-GUARD CONFOUND. The 8 ms sleep deferral followed the per-decision
#     OFDMA bit, so the OFDMA-on arm deferred sleep and the off arm did not: the
#     comparison varied two things. Three arms separate them, all on the same 50
#     seeds (90000-90049), all paired:
#
#       A  OFDMA off, no guard          (historical off arm)
#       B  OFDMA on,  guard 8 ms        (historical on arm -- produced by
#                                        run_heur_indep.sh, already running)
#       C  OFDMA off, guard 8 ms        (off arm WITH the guard)
#
#     B vs A is the historical, confounded comparison. B vs C holds the guard
#     fixed and varies only OFDMA. C vs A is the guard's own cost. If B vs C is
#     much smaller than B vs A, the reported OFDMA penalty was substantially the
#     guard, and the paper's central comparison has to be restated.
#
# Waits for the independent-seed ablation to finish rather than competing with
# it for the box. Resumable: evaluate.py skips episodes whose json exists.
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# Repo root is one level up: scripts live in <root>/<topic>-scripts/.
ROOT="$(cd "$HERE/.." && pwd)"
# Override with:  PY=/path/to/python3.11 ./script.sh
PY="${PY:-python3}"
# contrib/ai/examples/MobiCom/twt/<repo> -> up 6 to the ns-3 root; override with NS3=...
NS3="${NS3:-$(cd "$ROOT/../../../../../.." && pwd)}"
OUT="$ROOT/results/phystate"; CSV="$ROOT/results/phystate_csv"
LOG="$ROOT/results/phystate.log"
mkdir -p "$OUT" "$CSV"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
exec 4>"$ROOT/run/phystate.lock"; flock -n 4 || { say "another phystate run holds the lock"; exit 0; }

say "=== waiting for the independent-seed ablation to finish ==="
for _ in $(seq 1 180); do
  [ -f "$ROOT/results/HEUR_INDEP_DONE" ] && break
  pgrep -f 'bash \./run_heur_indep' >/dev/null 2>&1 || break
  sleep 60
done
say "proceeding (heur_indep done marker: $([ -f "$ROOT/results/HEUR_INDEP_DONE" ] && echo yes || echo no))"

run_arm() {   # <tag> <policy> <extra args...>
  # Each arm gets its OWN output directory. evaluate.py names result files after
  # the POLICY, not the seg tag, so two arms that differ only in a simulator flag
  # (here --sleep-guard-symmetric) produce identical filenames: the second arm
  # found the first arm's files, reported "50/50 present" and ran nothing. The
  # guard control silently did not happen.
  local tag="$1" pol="$2"; shift 2
  local dir="$OUT/$tag"; mkdir -p "$dir"
  local n; n=$(ls "$dir"/*.json 2>/dev/null | wc -l)
  if [ "$n" -ge 48 ]; then say "arm $tag already has $n episodes"; return 0; fi
  say "arm $tag: $pol $*"
  "$PY" "$ROOT/eval-scripts/evaluate.py" --policy "$pol" --seeds 90000-90049 \
      --out "$dir" --seg "$tag" --progress-every 0 "$@" >> "$LOG" 2>&1 \
    && say "  $tag done ($(ls "$dir"/*.json 2>/dev/null | wc -l) episodes)" \
    || say "  $tag returned non-zero"
}

run_arm ga "latency:3400:2700:11:4:17:off"
run_arm gc "latency:3400:2700:11:4:17:off" --sleep-guard-symmetric

# Harvest the per-STA CSVs the scenario writes into the ns-3 root. Names are
# prefixed by segName+seed, so they are unique per episode and can be collected
# after the fact rather than moved one at a time.
say "harvesting per-STA state logs"
m=0
for pfx in ga gc hi1 hi2; do
  for f in "$NS3"/${pfx}9*_persta.csv; do
    [ -f "$f" ] || continue
    mv -f "$f" "$CSV"/ && m=$((m+1))
  done
done
say "collected $m per-STA logs into results/phystate_csv"
# the other three logs are not needed; drop them so the ns-3 root stays clean
for pfx in ga gc hi1 hi2; do
  rm -f "$NS3"/${pfx}9*_overall.csv "$NS3"/${pfx}9*_actions.csv "$NS3"/${pfx}9*_rewards.csv
done
say "=== done ==="
touch "$ROOT/results/PHYSTATE_DONE"
