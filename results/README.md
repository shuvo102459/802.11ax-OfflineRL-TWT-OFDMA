# Results

`frozen*.json` are the single source of truth for every number in the paper.
`src/gen_tables.py` and `src/make_figs_v3.py` read only these files, so the
tables, the figures and the text cannot disagree.

| File | Produced by | Contents |
|---|---|---|
| `frozen.json` | `src/freeze_results.py` | 9 non-learned arms, 5 algorithms × 10 fits, each on 50 paired seeds. Per-fit energy, throughput, delay, violations, OFDMA share; paired Wilcoxon and bootstrap CIs against the hand-designed baseline. |
| `frozen_origreward.json` | `scripts/run_origreward.sh` | The reward-correction control: IQL retrained on the original saturated reward at the same 10 seeds and selection rule. |
| `frozen_nsta9.json` | `scripts/run_nsta9.sh` | The multiplexing-degree control: both OFDMA arms re-run at 9 stations per trigger instead of ns-3's default 4. |

Two comparisons appear in `frozen.json` for each algorithm and they answer
different questions:

- `vs_hand_meanfit_pct` — the per-seed **mean over ten fits**. This is what
  training once yields, and it is the number that pairs arithmetically with
  `mean_J`. The paper's headline.
- `vs_handdesigned_free_pct` — the single **lowest-energy fit** of ten. Only
  identifiable by measuring all ten on the network, which is the cost offline RL
  exists to avoid. Reported separately for exactly that reason.

Quoting the best-fit delta next to the mean-of-fits energy is an arithmetic
error (107.3 vs 166.5 J is −35.6 %, not −37.2 %). An earlier draft did this.

Raw per-episode JSON (~259 MB) and training datasets (~2.1 GB) are not tracked.
Regenerate with `scripts/`, or ask the authors.
