#!/bin/bash
# Regression test for lib/resumable.sh. Every long phase depends on these
# semantics, so they are checked rather than assumed. Run: ./lib/test_resumable.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# Repo root is one level up: scripts live in <root>/<topic>-scripts/.
ROOT="$(cd "$HERE/.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
cd "$T"
source "$HERE/resumable.sh"
rr_init "$T/state"
P=0; F=0
chk(){ if [ "$2" = "$3" ]; then echo "  PASS  $1"; P=$((P+1)); else echo "  FAIL  $1 (got '$2' want '$3')"; F=$((F+1)); fi; }

rr_run u_ok "$T/w1" out.csv 3 60 -- bash -c 'printf "a\nb\nc\n" > out.csv'
chk "success+complete -> marked" "$(rr_done u_ok && echo y || echo n)" "y"

rr_run u_short "$T/w2" out.csv 10 60 -- bash -c 'printf "a\n" > out.csv' >/dev/null
chk "success+short file -> not marked" "$(rr_done u_short && echo y || echo n)" "n"

rr_run u_crash "$T/w3" out.csv 1 60 -- bash -c 'printf "a\nb\n" > out.csv; exit 3' >/dev/null
chk "nonzero exit -> not marked" "$(rr_done u_crash && echo y || echo n)" "n"

rr_run u_tmo "$T/w4" out.csv 1 1 -- bash -c 'printf "a\n" > out.csv; sleep 30' >/dev/null
chk "timeout -> not marked" "$(rr_done u_tmo && echo y || echo n)" "n"

rr_run u_ok "$T/w1" out.csv 3 60 -- bash -c 'echo CLOBBERED > out.csv'
chk "completed unit is skipped, output intact" "$(tr -d '\n' < w1/out.csv)" "abc"

rr_run u_short "$T/w2" out.csv 3 60 -- bash -c 'printf "x\ny\nz\n" > out.csv'
chk "incomplete unit retried on next pass" "$(rr_done u_short && echo y || echo n)" "y"
chk "retry overwrote the partial file" "$(wc -l < w2/out.csv)" "3"

mkdir -p w5 && printf "1\n2\n3\n4\n" > w5/out.csv
rr_adopt u_adopt "$T/w5/out.csv" 4
chk "adopt complete prior output" "$(rr_done u_adopt && echo y || echo n)" "y"
rr_adopt u_adopt2 "$T/w5/out.csv" 99
chk "adopt refuses short prior output" "$(rr_done u_adopt2 && echo y || echo n)" "n"

echo; echo "$P passed, $F failed"; [ "$F" -eq 0 ]
