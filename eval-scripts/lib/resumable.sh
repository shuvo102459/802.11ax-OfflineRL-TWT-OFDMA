#!/bin/bash
# =============================================================================
# resumable.sh -- shared restart-safe execution for every runner in this repo.
# =============================================================================
# A forced shutdown killed a sweep mid-flight and the work was only recoverable
# because the ns-3 scenario happens to truncate its CSVs on open. That is luck,
# not design. Every long phase sources this file so that an interrupted run
# resumes at the exact unit of work it was in, whatever killed it.
#
# The contract is one marker file per unit of work, written only after the unit
# has been validated. A unit is therefore in exactly one of three states:
#   marker present  -> finished and checked, skip it
#   marker absent   -> never ran, or died partway; run it again from scratch
#   (no third state) -- partial output is never trusted, only overwritten.
#
# Callers must make each unit idempotent, i.e. re-running it must overwrite its
# outputs rather than append to them.
# =============================================================================

rr_init() {
  RR_STATE="$1"
  mkdir -p "$RR_STATE/done" "$RR_STATE/logs"
}

# rr_done <key> -- true if this unit already finished and passed validation
rr_done() { [ -f "$RR_STATE/done/$1" ]; }

# rr_mark <key> <note> -- atomic: a half-written marker can never be observed
rr_mark() {
  printf '%s\t%s\n' "$(date '+%F %T')" "${2:-ok}" > "$RR_STATE/done/.$1.tmp"
  mv -f "$RR_STATE/done/.$1.tmp" "$RR_STATE/done/$1"
}

# rr_run <key> <workdir> <verify_file> <min_lines> <timeout_s> -- <cmd...>
#
# Runs cmd in workdir, then requires BOTH a zero exit status and verify_file
# holding at least min_lines lines before marking the unit done. A run that
# times out, crashes, or produces a short file leaves no marker, so the next
# invocation of the phase repeats exactly that unit and nothing else.
rr_run() {
  local key="$1" wd="$2" vf="$3" minl="$4" tmo="$5"; shift 5
  [ "${1:-}" = "--" ] && shift
  if rr_done "$key"; then return 0; fi
  mkdir -p "$wd"
  # Hold the launch until it fits beside the desktop reserve, then run it as the
  # preferred OOM victim: if memory still runs out, the kernel takes this
  # resumable simulation instead of the user's editor.
  if [ -n "${RR_NEED_MB:-}" ]; then
    rr_wait_for_mem "$RR_NEED_MB" "$key" || return 1
  fi
  # Run inside its OWN systemd scope when available. setsid gives a new session
  # but does NOT leave the parent's cgroup: on 2026-09-30 an OOM-killed simulation
  # was still a member of VS Code's app-code-*.scope, so systemd tore that scope
  # down and the user's editor died with it -- even though oom_score_adj had
  # correctly steered the kill to the simulation. A private scope with MemoryMax
  # contains both the kill and the blast radius; exceeding the cap kills only this
  # run, which is resumable.
  if [ -n "${RR_SCOPE_MAX:-}" ] && command -v systemd-run >/dev/null 2>&1 \
     && systemd-run --user --scope --collect true >/dev/null 2>&1; then
    ( cd "$wd" && rr_selfish_off
      exec systemd-run --user --scope --collect -q \
           -p MemoryMax="$RR_SCOPE_MAX" -p MemorySwapMax=0 \
           -- timeout -k 10 "$tmo" "$@" ) >"$RR_STATE/logs/$key.log" 2>&1
  else
    ( cd "$wd" && rr_selfish_off; exec timeout -k 10 "$tmo" "$@" ) \
        >"$RR_STATE/logs/$key.log" 2>&1
  fi
  local rc=$?
  local have=0
  [ -f "$wd/$vf" ] && have=$(wc -l < "$wd/$vf")
  if [ "$rc" -eq 0 ] && [ "$have" -ge "$minl" ]; then
    rr_mark "$key" "rc=0 lines=$have"
    return 0
  fi
  echo "    !! $key incomplete (rc=$rc lines=$have) -- will retry on next run"
  return 1
}

# rr_adopt <key> <verify_file_path> <min_lines>
# Marks a unit done if a previous run (from before markers existed) already
# produced a complete output. Lets this machinery be added to a sweep that is
# already half finished without discarding that work.
rr_adopt() {
  local key="$1" vfp="$2" minl="$3" have=0
  rr_done "$key" && return 0
  [ -f "$vfp" ] && have=$(wc -l < "$vfp")
  [ "$have" -ge "$minl" ] && { rr_mark "$key" "adopted lines=$have"; return 0; }
  return 1
}

# rr_guard <lockfile> -- refuse to start if the same phase is already running.
# Two supervisors racing on one output tree is its own corruption mode.
rr_guard() {
  exec 9>"$1"
  if ! flock -n 9; then
    echo "another instance is already running (lock: $1); exiting"
    exit 0
  fi
  echo $$ >&9
}

# ── Resource governance ──────────────────────────────────────────────────────
# A 5-way parallel sweep once consumed 13.7 GB of 15 GB and the kernel killed
# the user's IDE to reclaim it. ns-3 memory, not core count, sets how many runs
# fit: each holds GBs and grows with simulated time. These keep a sweep from
# ever again taking the machine down.

RR_RESERVE_MB=${RR_RESERVE_MB:-5000}   # never encroach on the desktop's memory

# rr_avail_mb -- MemAvailable, the kernel's own estimate of what is claimable
rr_avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

# rr_wait_for_mem <needed_mb> [label] -- block until a run of this size fits
# alongside the reserve. Returns 1 if it cannot fit even on an idle machine, so
# the caller can degrade instead of deadlocking.
rr_wait_for_mem() {
  local need="$1" label="${2:-run}" waited=0 avail
  local total; total=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)
  if [ "$((need + RR_RESERVE_MB))" -gt "$total" ]; then
    echo "    !! $label needs ${need}MB + ${RR_RESERVE_MB}MB reserve > ${total}MB total"
    return 1
  fi
  while :; do
    avail=$(rr_avail_mb)
    [ "$avail" -ge "$((need + RR_RESERVE_MB))" ] && return 0
    if [ "$waited" -eq 0 ]; then
      echo "    .. holding $label: ${avail}MB available, need $((need + RR_RESERVE_MB))MB"
    fi
    sleep 30; waited=$((waited + 30))
  done
}

# rr_selfish_off -- mark THIS process as the preferred OOM victim and deprioritise
# it on CPU. Call inside the child, before exec. A killed simulation is resumable;
# a killed editor is the user's work. Raising oom_score_adj needs no privileges.
rr_selfish_off() {
  echo 800 > /proc/self/oom_score_adj 2>/dev/null || true
  renice -n 10 -p $$ >/dev/null 2>&1 || true
}

# rr_peak_rss_mb <pid> -- peak resident size, for calibrating the budget
rr_peak_rss_mb() { awk '/VmHWM/{print int($2/1024)}' "/proc/$1/status" 2>/dev/null || echo 0; }
