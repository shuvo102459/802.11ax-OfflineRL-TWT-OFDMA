#!/usr/bin/env python3
"""Figures for the v3 (non-stationary joint TWT+OFDMA) submission.

Every number is read from results/frozen*.json -- the same files the tables are
generated from -- so a figure can never disagree with the text. The earlier
figures in figures/ were built from a 150-seed design that this paper no longer
uses; their captions quote 24.6% and 150 seeds against the current 35.6% and 50,
so they are superseded, not reusable.

Palette is inherited from code/make_paper_figs.py, where it passed the six
checks (OKLab dE, Machado CVD at severity 1.0, greyscale luminance separation,
contrast). Marker shape carries identity as well as colour because IEEE
proceedings are printed in black and white.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

C_LEARNED, C_STATIC, C_HEUR = "#2a78d6", "#4a3aa7", "#eb6834"
C_WORSE, C_MID = "#a8322a", "#f0efec"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COL = 3.45

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 600, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
})

F  = json.load(open(f"{R}/results/frozen.json"))
N9 = json.load(open(f"{R}/results/frozen_nsta9.json"))
A, G = F["arms"], F["algorithms"]


def fig_system():
    """Top: one beacon interval of TWT. Bottom: a trigger exchange, drawn to its
    own scale -- 5.8 ms is 5.7% of a beacon interval and cannot be labelled at
    the top panel's scale without misrepresenting its width."""
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(COL, 2.06),
                                 gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": .95})
    BI = 102.4
    # --- top: the schedule (schedule 11 = two service periods, 50 ms + 30 ms) ---
    ax.add_patch(Rectangle((0, .18), 50, .64, fc=C_STATIC, ec="none", alpha=.24))
    ax.add_patch(Rectangle((50, .18), 30, .64, fc=C_LEARNED, ec="none", alpha=.24))
    ax.add_patch(Rectangle((80, .18), BI-80, .64, fc=GRID, ec="none", alpha=.55))
    ax.text(25, .50, "Group A awake\n8 of 16 STAs", ha="center", va="center", fontsize=6.0)
    ax.text(65, .50, "Group B\n8 STAs", ha="center", va="center", fontsize=6.0)
    ax.text(91, .50, "all\nasleep", ha="center", va="center", fontsize=6.0, color=INK2)
    # mark where the zoomed exchange sits
    ax.add_patch(Rectangle((8, .18), 5.8, .64, fc="none", ec=C_HEUR, lw=.9))
    ax.set_xlim(-2, BI+2); ax.set_ylim(0, 1.0)
    ax.set_yticks([]); ax.set_xticks([0, 50, 80, BI])
    ax.set_xticklabels(["0", "50", "80", "102.4"])
    ax.set_xlabel("time within one beacon interval (ms)", labelpad=1.5)
    for sp in ("left", "right", "top"): ax.spines[sp].set_visible(False)

    # --- bottom: the exchange, own scale ---
    segs = [("TF", 0, 1.0), ("", 1.0, 1.16), ("TB PPDU", 1.16, 4.9), ("", 4.9, 5.06), ("BA", 5.06, 5.8)]
    for lab, a0, a1 in segs:
        fc = C_HEUR if lab else GRID
        bx.add_patch(Rectangle((a0, .34), a1-a0, .40, fc=fc, ec="none",
                               alpha=.9 if lab else .8))
        if lab:
            bx.text((a0+a1)/2, .54, lab, ha="center", va="center",
                    fontsize=5.6, color="white")
    bx.text(1.08, .80, "SIFS", ha="center", fontsize=5.0, color=INK2)
    bx.text(4.98, .80, "SIFS", ha="center", fontsize=5.0, color=INK2)
    bx.annotate("", xy=(0, .24), xytext=(5.8, .24),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=.6))
    bx.text(2.9, .06, "5.8 ms: solicited STAs awake and listening throughout",
            ha="center", va="bottom", fontsize=5.6)
    bx.set_xlim(-.3, 6.1); bx.set_ylim(0, 1.0)
    bx.set_yticks([]); bx.set_xticks([0, 2, 4, 5.8])
    bx.set_xticklabels(["0", "2", "4", "5.8"])
    bx.set_xlabel("time within one trigger exchange (ms)", labelpad=1.5)
    for sp in ("left", "right", "top"): bx.spines[sp].set_visible(False)
    for sp in ("bottom",): bx.spines[sp].set_color(C_HEUR)
    fig.savefig(f"{OUT}/fig_system.pdf"); fig.savefig(f"{OUT}/fig_system.png")
    plt.close(fig); print("  fig_system")


def fig_ordering():
    """Energy against how often each policy enables uplink OFDMA."""
    fig, ax = plt.subplots(figsize=(COL, 2.18))
    names = {"iql": "IQL", "dqn": "DQN+CQL", "dueling": "Duel+CQL", "ddqn": "DDQN", "bc": "BC"}
    mk = {"iql": "o", "dqn": "s", "dueling": "^", "ddqn": "D", "bc": "v"}
    xs, ys = [], []
    for a, r in G.items():
        x, y, e = r["ofdma_share"], r["mean_J"], r["sd_J"]
        xs.append(x); ys.append(y)
        ax.errorbar(x, y, yerr=e, fmt=mk[a], ms=4.6, mfc=C_LEARNED, mec="white",
                    mew=.7, ecolor=C_LEARNED, elinewidth=.8, capsize=2, zorder=3)
        off = {"dqn": (6, -7.5), "dueling": (6, 3.5)}.get(a, (6, -1.5))
        ax.annotate(names[a], (x, y), textcoords="offset points",
                    xytext=off, fontsize=6.4, color=INK)
    b = A["r8b_handdesigned_free"]
    ax.axhline(b["energy_J"], color=C_HEUR, lw=1.0, ls="--", zorder=2)
    ax.text(49, b["energy_J"]+5, "hand-designed, tuned", ha="right", fontsize=6.4, color=C_HEUR)
    s, p = np.polyfit(xs, ys, 1), np.corrcoef(xs, ys)[0, 1]
    gx = np.linspace(-2, 52, 10)
    ax.plot(gx, s[0]*gx+s[1], color=INK2, lw=.7, ls=":", zorder=1)
    ax.text(2, 243, f"$r$ = {p:.3f}", fontsize=6.4, color=INK2)
    ax.set_xlabel("decisions with uplink OFDMA enabled (%)")
    ax.set_ylabel("station energy (J)")
    ax.set_xlim(-3, 54); ax.grid(True, color=GRID, lw=.5, alpha=.7); ax.set_axisbelow(True)
    for sp in ("right", "top"): ax.spines[sp].set_visible(False)
    fig.savefig(f"{OUT}/fig_ordering.pdf"); fig.savefig(f"{OUT}/fig_ordering.png")
    plt.close(fig); print(f"  fig_ordering (r={p:.3f})")


def fig_nsta():
    """OFDMA cost at both solicitation limits, both baseline arms."""
    fig, ax = plt.subplots(figsize=(COL, 1.78))
    labs = ["static TWT", "hand-designed"]
    off = [N9["static"]["off_J"], N9["handdesigned"]["off_J"]]
    n4  = [N9["static"]["n4_J"],  N9["handdesigned"]["n4_J"]]
    n9  = [N9["static"]["n9_J"],  N9["handdesigned"]["n9_J"]]
    x = np.arange(2); w = 0.26
    ax.bar(x-w, off, w, color=C_LEARNED, label="OFDMA off", zorder=3)
    ax.bar(x,   n4,  w, color=C_HEUR,    label="OFDMA, 4 STAs/trigger", zorder=3)
    ax.bar(x+w, n9,  w, color=C_WORSE,   label="OFDMA, 9 STAs/trigger", zorder=3)
    for i in range(2):
        ax.text(x[i],   n4[i]+7, f"+{N9[['static','handdesigned'][i]]['n4_vs_off_pct']:.0f}%",
                ha="center", fontsize=6.0, color=INK)
        ax.text(x[i]+w, n9[i]+7, f"+{N9[['static','handdesigned'][i]]['n9_vs_off_pct']:.0f}%",
                ha="center", fontsize=6.0, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("station energy (J)")
    ax.set_ylim(0, max(n9)*1.22)
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.1)
    ax.grid(True, axis="y", color=GRID, lw=.5, alpha=.7); ax.set_axisbelow(True)
    for sp in ("right", "top"): ax.spines[sp].set_visible(False)
    fig.savefig(f"{OUT}/fig_nsta.pdf"); fig.savefig(f"{OUT}/fig_nsta.png")
    plt.close(fig); print("  fig_nsta")


if __name__ == "__main__":
    print("writing figures from frozen*.json:")
    fig_system(); fig_ordering(); fig_nsta()
    print(f"  -> {OUT}")
