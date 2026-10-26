#!/usr/bin/env python3
"""gen_tables.py -- emit every table from results/frozen.json.

Nothing here is typed by hand. If a number changes in the data it changes in the
paper, which is the whole point: hand-copied figures are what produced
"20.23 -> 18.00" and "372.2 vs 372.1" in earlier drafts.
"""
import json, os
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(R, "paper", "tables")
F = json.load(open(os.path.join(R, "results", "frozen.json")))
A, G = F["arms"], F["algorithms"]

ROWS = [("No TWT, no OFDMA", "r1_plain"), ("Static TWT", "r2_static_twt"),
        ("Static TWT $+$ OFDMA", "r3_static_twt_ofdma"), ("Random TWT", "r4_random_twt"),
        ("Random TWT $+$ OFDMA", "r5_random_twt_ofdma"), ("OFDMA only, no TWT", "r6_ofdma_only"),
        ("Random joint, no TWT", "r7_random_joint"),
        ("Hand-designed, OFDMA on", "r8_handdesigned_ofdma_on"),
        ("Hand-designed, \\emph{free choice}", "r8b_handdesigned_free")]
NAME = {"iql": "IQL", "dqn": "DQN$+$CQL", "dueling": "Dueling$+$CQL",
        "ddqn": "DDQN", "bc": "BC"}

import math
def _p(x):
    if x != x: return "---"
    return f"${x:.2f}$" if x >= 0.01 else f"$10^{{{int(math.floor(math.log10(x)))}}}$"

B = A["r8b_handdesigned_free"]

# One table, not two. The learned rows previously appeared in both tab_main and
# tab_algos with the same mean energy, which is a duplication a reader has to
# reconcile by hand.
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Every scenario on the same $50$ paired evaluation seeds, one",
     r"simulator and one $42$-decision measurement window. Where uplink OFDMA is",
     r"enabled the scheduler is service-period aligned and the sleep guard is",
     r"disabled, so no row depends on that workaround. The hand-designed controller",
     r"is tuned on disjoint seeds under a service floor; the row marked",
     r"\emph{free choice} lets that search pick its own OFDMA mode, and it picks",
     r"off. Learned rows give the mean and standard deviation over $10$ training",
     r"seeds. The last three columns compare each learned method against the",
     r"\emph{free choice} controller at $%.1f$\,J, Wilcoxon signed-rank:" % B["energy_J"],
     r"\emph{typical} uses the per-seed mean over the ten fits, which is what",
     r"training once yields, and \emph{best} uses the single lowest-energy fit,",
     r"which can only be identified by measuring all ten on the network.}",
     r"\label{tab:main}", r"\footnotesize", r"\setlength{\tabcolsep}{4.5pt}",
     r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
     r" & & & & & & \multicolumn{2}{c}{typical fit} & best fit \\",
     r"\cmidrule(lr){7-8}\cmidrule(lr){9-9}",
     r"Scenario & $E$ (J) & tput (Mb/s) & delay (ms) & viol. & OFDMA & "
     r"$\Delta$ vs hand & $p$ & $\Delta$ vs hand \\", r"\midrule"]
for lbl, k in ROWS:
    if k not in A: continue
    r = A[k]
    L.append(f"{lbl} & ${r['energy_J']:.1f}$ & ${r['throughput_mbps']:.2f}$ & "
             f"${r['delay_ms']:.0f}$ & ${r['viol']:.2f}$ & "
             f"${r['ofdma_share']:.0f}\\,\\%$ & & & \\\\")
L.append(r"\midrule")
L.append(r"\multicolumn{9}{l}{\emph{offline RL, 10 training seeds each}} \\")
for a, r in sorted(G.items(), key=lambda kv: kv[1]["mean_J"]):
    _d = np.mean([v["delay_ms"] for v in r["per_fit"].values()])
    _v = np.mean([v["viol"] for v in r["per_fit"].values()])
    L.append(f"\\quad {NAME[a]} & ${r['mean_J']:.1f} \\pm {r['sd_J']:.1f}$ & "
             f"${r['throughput_mbps']:.2f}$ & ${_d:.0f}$ & ${_v:.2f}$ & "
             f"${r['ofdma_share']:.0f}\\,\\%$ & "
             f"${r.get('vs_hand_meanfit_pct', 0):+.1f}\\,\\%$ & "
             f"{_p(r.get('wilcoxon_p_meanfit', float('nan')))} & "
             f"${r.get('vs_handdesigned_free_pct', 0):+.1f}\\,\\%$ \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(os.path.join(T, "tab_main.tex"), "w").write("\n".join(L) + "\n")
print("  wrote tab_main.tex (merged) from frozen.json")
