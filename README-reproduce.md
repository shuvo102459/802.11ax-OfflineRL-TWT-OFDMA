# Reproducing the results

Runtimes are from an 8-core machine with 4 parallel ns-3 workers, which is what
the scripts size themselves for. ns-3 episodes dominate: roughly **2 minutes per
episode**, and everything below is counted in episodes.

## 0. Build the scenario

```bash
cp ns3/scenario4_rl_twt_ofdma.cc  $NS3/scratch/scenario4-rl-agent/
cd $NS3 && ./ns3 build scenario4_rl_twt_ofdma
```

The scenario adds these flags beyond the stock scenario. Each one exists because
a result depended on it:

| Flag | Default | Why it exists |
|---|---|---|
| `--spAligned` | false | Solicit a station only while its service period can still contain the whole 5.8 ms exchange. Removes the race where a solicited station sleeps mid-exchange. |
| `--spMarginUs` | 5800 | The exchange duration reserved by `--spAligned`. |
| `--sleepGuardUs` | 8000 | Defers a PHY sleep so an in-flight exchange can finish. Historical workaround; **set to 0** with `--spAligned`. |
| `--sleepGuardSymmetric` | false | Applies the guard in *both* OFDMA arms. Without it the arms differ in sleep handling as well as in OFDMA. |
| `--ccaCurrentMa` | 273 | ns-3's unset default. 66 makes the power profile coherent. |
| `--nStations` | 4 | Max stations granted an RU per trigger. ns-3's default; a 20 MHz channel admits 9. |

## 1. Collect the dataset  (~2,400 episodes, ≈ 20 h)

```bash
scripts/run_final.sh          # resumable; re-run after an interrupt
python src/build_energy_v2.py # corrected energy reward -> data/stat_e2.npz
```

Actions are drawn uniformly over the 760-way space and **held for ten
decisions**. The hold is deliberate: with a fresh draw every decision, measured
energy is flat against schedule tightness and the gradient is invisible.

## 2. Train (10 seeds × 5 algorithms, ≈ 11 h)

```bash
scripts/run_seeds10.sh
```

Episode-disjoint split, never transition-disjoint: 2,043 training episodes,
361 held out.

## 3. Select checkpoints without touching the network (≈ 3 h)

```bash
scripts/select_checkpoints.sh
```

Ranks snapshots by fitted Q evaluation. **Do not re-enable the convergence
gate**: it admitted only snapshots whose FQE drift fell below 2 %, none passed,
and selection silently fell back to the final epoch — which both FQE and the
simulator rank *last* of four. FQE's ranking validates against the simulator at
Spearman ρ = −1.000; the gate was throwing away a correct ranking at a cost of
6.5 % energy.

## 4. Baselines and evaluation (≈ 30 h)

```bash
scripts/run_rows.sh    # the non-learned scenarios
scripts/run_heur2.sh   # tune + evaluate the hand-designed controller
```

All evaluation uses seeds 90000–90049, disjoint from collection (30000+) and
from heuristic tuning (80000–80014).

## 5. The two controls (≈ 12 h)

```bash
scripts/run_origreward.sh   # was the reward repair real, or seed luck?
scripts/run_nsta9.sh        # does the OFDMA penalty survive 9 STAs per trigger?
```

## 6. Freeze and build

```bash
python src/freeze_results.py    # -> results/frozen.json
python src/gen_tables.py        # -> paper/tables/*.tex
python src/make_figs_v3.py      # -> paper/figures/*.pdf
cd paper && latexmk -pdf main.tex
```

---

## Pitfalls that cost us real time

Recorded because each one produced a wrong number that looked plausible.

**Forward every simulation-changing flag into child processes.** `evaluate.py`
spawns a child per episode. A flag honoured in-process but dropped from the
child command yields a "control" arm byte-identical to its control. Probe a
child-spawned seed, not the first episode.

**Never compare arms that differ in two things.** The 8 ms sleep guard applied
only when OFDMA was enabled, so the OFDMA comparison varied sleep handling too.
Isolated, the guard alone was worth +62 % — two thirds of a penalty we had
charged to the protocol.

**Aggregate statistics hide timing changes.** Mean sleep occupancy moved under
0.8 % between the guard arms, which read as harmless. The guard changed *when*
stations slept, not *how much*; no occupancy average can see that.

**Check the measurement window.** One harness included a pre-TWT warm-up that
burns 26.6 % of episode energy in 9.8 % of the episode. The same weights read
153.6 J and 104–111 J. Matched to the 42-decision window they agree to 4.5 %.

**Check reward scales against the data range.** The inherited reward charged
`tanh(E/400 mJ)` against an E spanning 1390–14,167 mJ — saturated for 99.99 % of
transitions, so the preset named for energy barely correlated with energy.

**`pkill -f` matches your own command line.** Filter by age and PID, never by
pattern alone.
