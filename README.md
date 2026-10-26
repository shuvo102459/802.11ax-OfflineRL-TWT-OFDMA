# 802.11ax Offline RL — Joint TWT + Uplink OFDMA Scheduling

![NS-3](https://img.shields.io/badge/NS--3-3.44-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![WiFi](https://img.shields.io/badge/IEEE-802.11ax-orange)
![RL](https://img.shields.io/badge/Offline%20RL-IQL%20%7C%20CQL%20%7C%20BC-red)
![Paper](https://img.shields.io/badge/Paper-WCNC%202027-lightgrey)

**Do Target Wake Time and Uplink OFDMA Combine? An Offline Reinforcement
Learning Study of Joint Scheduling in IEEE 802.11ax**

IEEE 802.11ax introduces Target Wake Time (TWT) and uplink OFDMA to reduce
station energy. The two are usually configured separately, and how they
interact when an access point controls both is not established. A controller
must adapt to changing traffic, yet online learning is impractical on a
production network and a hand-tuned rule needs an expert. Offline reinforcement
learning offers a middle path: a policy trained from logged data alone, with no
exploration on the live network. We cast joint control as a 760-way action —
TWT schedule, station-to-group assignment, and whether uplink OFDMA is enabled
at each decision — and train five offline algorithms on 100,235 logged
transitions in ns-3.44. **Implicit Q-Learning reaches 107.3 ± 2.6 J against a
tuned hand-designed controller's 166.5 J, a 35.6 % reduction at higher
throughput, while four of the five methods lose to that controller and the
energy of all five is ordered by how often each enables uplink OFDMA.**

**Authors:** Ahmed Selim Anwar, Ahmed Maksud, Joarder Jafor Sadique,
Charles Wiame, Marcelo Menezes De Carvalho
**PI:** Dr. Marcelo Menezes De Carvalho, Department of Electrical Engineering,
Texas State University

**Paper:** [`paper/wcnc2027-twt-ofdma-offline-rl.pdf`](paper/wcnc2027-twt-ofdma-offline-rl.pdf)

---

## Prerequisites

This project builds on the NS3–NS3AI installation. It expects the standard
layout:

```
NS3-project/
└── ns-allinone-3.44/
    └── ns-3.44/
        ├── contrib/ai/                       <- ns3-ai
        └── contrib/ai/examples/MobiCom/twt/
            └── 802.11ax-OfflineRL-TWT-OFDMA-WCNC27/   <- this repo
```

Python 3.11 with `numpy`, `scipy`, `matplotlib`:

```bash
pip install numpy scipy matplotlib
```

## Quick Start

```bash
# 1. clone into the twt examples directory
cd $NS3/contrib/ai/examples/MobiCom/twt
git clone https://github.com/<user>/802.11ax-OfflineRL-TWT-OFDMA-WCNC27

# 2. register the scenario
cp 802.11ax-OfflineRL-TWT-OFDMA-WCNC27/twt-ofdma-main-simulation.cc \
   $NS3/scratch/scenario4-rl-agent/scenario4_rl_twt_ofdma.cc
cd $NS3 && ./ns3 build scenario4_rl_twt_ofdma

# 3. run the whole pipeline (long -- see section 8 for stage times)
cd contrib/ai/examples/MobiCom/twt/802.11ax-OfflineRL-TWT-OFDMA-WCNC27
./run_all.sh

# 4. or just rebuild the paper from the frozen results
python analysis-scripts/gen_tables.py
python analysis-scripts/make_figs.py
cd paper && latexmk -pdf main.tex
```

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [The Control Problem](#3-the-control-problem)
4. [Simulation Source and Custom Flags](#4-simulation-source-and-custom-flags)
5. [Offline RL Pipeline](#5-offline-rl-pipeline)
6. [Results](#6-results)
7. [The Two Controls](#7-the-two-controls)
8. [One-Stop Pipeline](#8-one-stop-pipeline)
9. [Known Limitations](#9-known-limitations)
10. [Quick Reference](#10-quick-reference)
11. [Citation](#11-citation)

---

## 1. Overview

| | |
|---|---|
| Stations | 16, four device classes (IoT / camera / voice / streamer) |
| Channel | 20 MHz, 802.11ax, single BSS |
| Decision interval | 20 beacon intervals = 2.048 s, 42 decisions per episode |
| Action space | **760** = 20 schedules × 19 assignments × 2 OFDMA modes |
| Dataset | 100,235 transitions, 2,404 episodes |
| Algorithms | BC, DQN+CQL, Dueling DQN+CQL, DDQN, IQL — 10 seeds each |
| Selection | Fitted Q Evaluation, no simulator in the loop |
| Evaluation | 50 paired seeds, disjoint from collection and tuning |

**Headline:** IQL cuts station energy **35.6 %** against a tuned hand-designed
controller (*p* = 1.8 × 10⁻¹⁵) at higher throughput *and* lower delay.
**Four of five offline methods lose** to that controller, so this is a result
about IQL, not about offline RL as a class.

---

## 2. System Architecture

```
          ┌──────────────────── ns-3.44 ────────────────────┐
          │                                                 │
          │   AP  ── TwtAwareScheduler (UL OFDMA trigger)   │
          │    │     + TWT schedule / group assignment      │
          │    │                                            │
          │   16 STAs  ── WifiRadioEnergyModel              │
          │              PhyStateHelper trace               │
          └────────────────────┬────────────────────────────┘
                               │  ns3-ai shared memory
                               │  (state 130-d  /  action 0..759)
          ┌────────────────────┴────────────────────────────┐
          │              Python policy process              │
          │   offline-trained Q-network, argmax over 760     │
          └─────────────────────────────────────────────────┘
```

Every 2.048 s the scenario packs a 130-dimensional state into shared memory,
the Python side returns one action index, and the scenario decodes and applies
it. During **data collection** the action is drawn uniformly and held for ten
decisions; during **evaluation** it is the learned policy's argmax.

---

## 3. The Control Problem

TWT splits each 102.4 ms beacon interval into service periods. A station sleeps
outside its own. Uplink OFDMA replaces contention with a trigger frame, but a
solicited station must stay awake through the whole exchange — which is what
makes the two mechanisms compete:

```
 time within one beacon interval (ms)
 0                      50          80        102.4
 |---- Group A awake ----|- Group B -|- all asleep -|
   8 of 16 STAs            8 STAs

 inside Group A's window, one trigger exchange (5.8 ms):
   [ TF ]·[    TB PPDU    ]·[ BA ]
     every solicited STA awake and listening throughout
```

The access point selects, once per decision:

```
    a = b·380 + s·19 + g          b ∈ {0,1}   uplink OFDMA on/off
                                  s ∈ {0..19} TWT schedule
                                  g ∈ {0..18} station-to-group assignment
```

Making the OFDMA bit part of the **action** rather than a property of the run
is what makes the problem joint: the policy can enable the scheduler when it
pays and disable it when it does not.

---

## 4. Simulation Source and Custom Flags

`twt-ofdma-main-simulation.cc` is the scenario. Beyond the stock flags it adds
the following — each one exists because a reported result depended on it.

| Flag | Default | Why it exists |
|---|---|---|
| `--spAligned` | `false` | Solicit a station only while its service period can still contain the whole 5.8 ms exchange. Removes the race where a solicited station sleeps mid-exchange. |
| `--spMarginUs` | `5800` | Exchange duration reserved by `--spAligned`. |
| `--sleepGuardUs` | `8000` | Defers a PHY sleep so an in-flight exchange can finish. Historical workaround — **set to 0** when using `--spAligned`. |
| `--sleepGuardSymmetric` | `false` | Applies the guard in *both* OFDMA arms. Without it the arms differ in sleep handling as well as in OFDMA. |
| `--ccaCurrentMa` | `273` | ns-3's unset default. `66` makes the power profile coherent. |
| `--nStations` | `4` | Max stations granted an RU per trigger. ns-3's default; a 20 MHz channel admits 9. |
| `--outPrefix` | `segName` | CSV prefix, unique per episode so parallel runs cannot collide. |

Energy is attributed by **measured deltas**, not time × current: the
`WifiRadioEnergyModel` runs its own listener state machine independent of
`WifiPhyStateHelper`, so a time×current reconstruction does not close (we saw
9 % unattributed).

---

## 5. Offline RL Pipeline

```
collect-scripts/      1-collect-dataset.sh      uniform action, 10-decision hold
                      2-build-dataset.py        episode-disjoint split
                      3-build-energy-reward.py  corrected energy term
                           │
offline-rl-scripts/   1-train-all-seeds.sh      5 algorithms × 10 seeds
                      2-select-checkpoints.sh   FQE ranking, no simulator
                           │
eval-scripts/         1-baseline-rows.sh        9 non-learned scenarios
                      2-tune-hand-designed.sh   coordinate search + service floor
                      3,4-control-*.sh          the two controls (section 7)
                           │
analysis-scripts/     freeze_results.py         -> results/frozen.json
                      gen_tables.py             -> paper/tables/*.tex
                      make_figs.py              -> paper/figures/*.pdf
```

**No number in the paper is typed by hand.** The tables and figures read only
`results/frozen*.json`, so the text and the artefacts cannot disagree.

**Checkpoint selection.** Importance-sampling OPE is degenerate at this action
cardinality — for a deterministic policy against a uniform behaviour policy the
effective sample size collapses to the state-action matches, a few hundred out
of 15,025 held-out transitions. We rank by FQE instead, and **validate that
ranking against the simulator** (Spearman ρ = −1.000) rather than assume it.

> ⚠️ Do not re-enable the convergence gate. It admitted only snapshots whose FQE
> drift fell below 2 %; none passed, so selection silently fell back to the
> final epoch — which both FQE and the simulator rank *last* of four. The gate
> was discarding a correct ranking at a cost of 6.5 % energy.

---

## 6. Results

All on the same 50 paired evaluation seeds, one simulator, one 42-decision
window. A seed fixes the traffic, the load-change time and its multiplier, so
every policy meets an identical environment.

| Scenario | Energy (J) | Throughput (Mb/s) | Delay (ms) | OFDMA |
|---|---|---|---|---|
| No TWT, no OFDMA | 441.1 | 24.95 | 2637 | 0 % |
| Static TWT | 173.3 | 21.03 | 2789 | 0 % |
| Static TWT + OFDMA | 282.2 | 19.82 | 2823 | 100 % |
| OFDMA only, no TWT | 477.8 | 24.30 | 2652 | 100 % |
| Hand-designed, OFDMA on | 265.5 | 18.37 | 2904 | 100 % |
| **Hand-designed, free choice** | **166.5** | 19.22 | 2888 | 0 % |
| **IQL** | **107.3 ± 2.6** | **20.76** | **2805** | 0 % |
| DQN+CQL | 193.0 ± 10.1 | 20.54 | 2829 | 23 % |
| Dueling+CQL | 194.1 ± 14.7 | 20.49 | 2834 | 25 % |
| DDQN | 222.7 ± 13.5 | 15.64 | 3076 | 43 % |
| Behaviour cloning | 255.1 ± 12.4 | 19.78 | 2875 | 48 % |

Two comparisons are reported per algorithm, and they answer different questions:

- **typical fit** — the per-seed mean over ten fits. What training *once*
  yields. IQL −35.6 %; DQN+CQL **+15.9 %**, Dueling+CQL **+16.6 %**.
- **best fit** — the single lowest-energy fit of ten. Only identifiable by
  measuring all ten on the network, which is the cost offline RL exists to
  avoid. There the two CQL variants tie the baseline (+1.6 %, +0.7 %).

The gap between those columns is the paper's sharpest point: **checkpoint
selection, not policy optimisation, is the remaining difficulty.**

**TWT and OFDMA substitute rather than combine.** Energy across the five
methods is ordered almost deterministically by how often each enables the
scheduler (*r* = 0.981), and an independent coordinate search over the
hand-designed controller disables OFDMA when given the choice.

---

## 7. The Two Controls

Both exist because a claim in the paper would otherwise be unsupported.

### `eval-scripts/3-control-reward-repair.sh`

*Was repairing the saturated energy reward real, or initialisation luck?*
The inherited reward charged `tanh(E/400 mJ)` against an *E* spanning
1390–14,167 mJ — saturated for 99.99 % of transitions. Retraining all ten IQL
fits on the original reward:

| | mean over 10 fits | s.d. |
|---|---|---|
| original (saturated) | 120.7 J | 5.63 |
| corrected | **107.3 J** | **2.60** |

Worth **13.3 J** (*p* = 3.6 × 10⁻¹⁵ paired, 1.2 × 10⁻⁴ across fits, so it is
the reward and not the seed). The broken reward **still** beat the baseline by
27.5 % — the repair improves the result rather than producing it.

### `eval-scripts/4-control-solicitation-limit.sh`

*ns-3 solicits 4 stations per trigger by default while 8 are awake. Did we
cripple OFDMA and then report that it lost?*

| arm | OFDMA off | N = 4 | N = 9 |
|---|---|---|---|
| static TWT | 173.3 J | 282.2 J (+62.8 %) | 324.9 J (**+87.4 %**) |
| hand-designed | 166.5 J | 265.5 J (+59.5 %) | 305.3 J (**+83.3 %**) |

Raising the limit makes OFDMA **worse**, both +15 % on the default at
*p* = 2 × 10⁻¹⁵, with throughput falling at each step. Solicitation cost scales
with the stations a trigger names; the benefit scales with the data they have
to send.

---

## 8. One-Stop Pipeline

`run_all.sh` runs every stage in order and is resumable — re-running skips work
already on disk. Stage times are for 8 cores with 4 parallel ns-3 workers
(≈ 2 min per episode).

| Stage | Script | Episodes | Time |
|---|---|---|---|
| 1 | `collect-scripts/1-collect-dataset.sh` | ~2,400 | ≈ 20 h |
| 2 | `offline-rl-scripts/1-train-all-seeds.sh` | — | ≈ 11 h |
| 3 | `offline-rl-scripts/2-select-checkpoints.sh` | — | ≈ 3 h |
| 4 | `eval-scripts/1-baseline-rows.sh` | 450 | ≈ 8 h |
| 5 | `eval-scripts/2-tune-hand-designed.sh` | ~800 | ≈ 14 h |
| 6 | `eval-scripts/3,4-control-*.sh` | 600 | ≈ 12 h |
| 7 | `analysis-scripts/*` + `latexmk` | — | < 1 min |

See [`README-reproduce.md`](README-reproduce.md) for the full walkthrough,
including the pitfalls that cost us real time.

---

## 9. Known Limitations

Stated plainly, because they bound what the results mean.

- **One topology.** 16 stations, 20 MHz, four device classes, one traffic
  model. Station mix and traffic are identical across every arm, so comparisons
  are internally controlled but not shown to transfer.
- **The energy model mixes two power profiles.** TX/RX/idle/sleep currents are
  consistent with a low-power part, but CCA-busy keeps ns-3's default 273 mA —
  measured on an 802.11n laptop card — so sensing costs more than transmitting.
  Repricing CCA at the receive current moves arm energies 18–30 %, unevenly. No
  ordering changes; every magnitude is conditional on that profile.
- **The behaviour policy is uniform random** with a ten-decision hold. A real
  deployment logs from a working policy, which is a different and easier
  offline RL problem.
- **No online RL baseline.** We argue offline RL is the practical middle path
  but do not measure the online upper bound.

---

## 10. Quick Reference

```
802.11ax-OfflineRL-TWT-OFDMA-WCNC27/
├── twt-ofdma-main-simulation.cc    ns-3.44 scenario (AP, 16 STAs, TWT+OFDMA)
├── run_all.sh                      one-stop pipeline, resumable
├── collect-scripts/                1-3  dataset collection and reward build
├── offline-rl-scripts/             1-2  training and FQE checkpoint selection
├── eval-scripts/                   1-5  baselines, tuning, the two controls
├── analysis-scripts/               freeze -> tables -> figures
├── results/                        frozen*.json, the single source of truth
├── paper/                          LaTeX source and built PDF
└── README-reproduce.md             full walkthrough and pitfalls
```

**Key configuration points**

| What | Where |
|---|---|
| TWT schedule catalogue (20) | `twt-ofdma-main-simulation.cc`, `SCHEDULES` |
| Group assignment rules (19) | `twt-ofdma-main-simulation.cc`, `InitAssignments()` |
| Device classes and rates | `twt-ofdma-main-simulation.cc`, `DEVICE_PROFILES` |
| Reward weights | `collect-scripts/3-build-energy-reward.py` |
| Evaluation seeds | `eval-scripts/*.sh`, `--seeds 90000-90049` |

**Build and run**

```bash
cd $NS3 && ./ns3 build scenario4_rl_twt_ofdma
./ns3 run "scenario4_rl_twt_ofdma -- --seed=90000 --spAligned --sleepGuardUs=0"
```

**Licence:** GNU General Public License v2 — consistent with ns-3. See
individual source files.

---

## 11. Citation

```bibtex
@inproceedings{anwar2027twtofdma,
  author    = {Anwar, Ahmed Selim and Maksud, Ahmed and Sadique, Joarder Jafor
               and Wiame, Charles and De Carvalho, Marcelo Menezes},
  title     = {Do Target Wake Time and Uplink {OFDMA} Combine? An Offline
               Reinforcement Learning Study of Joint Scheduling in
               {IEEE} 802.11ax},
  booktitle = {IEEE Wireless Communications and Networking Conference (WCNC)},
  year      = {2027}
}
```

Related work by the same group:

```bibtex
@inproceedings{maksud2026drltwt,
  author    = {Maksud, Ahmed and De Carvalho, Marcelo Menezes},
  title     = {Deep Reinforcement Learning-Based Dynamic {TWT} Scheduling for
               Heterogeneous {Wi-Fi} Networks},
  booktitle = {ICCCN},
  year      = {2026}
}
```
