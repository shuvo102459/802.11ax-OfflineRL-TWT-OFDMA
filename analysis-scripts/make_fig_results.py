#!/usr/bin/env python3
"""Figure 2: the energy comparison this literature expects, plus what the
policy actually does.

(a) follows the convention of TWT energy papers (e.g. Busacca et al. 2025,
    Fig. 9): mean station energy per arm with 95% confidence intervals, so the
    comparison is visible rather than buried in a table.
(b) answers "what does the learned policy do?": schedule index per decision for
    one IQL fit, with the load step marked. The correlation scatter it replaces
    plotted five method-level means, which is descriptive, not mechanistic.

Reads results/frozen*.json and results/final/trace/ only.
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

C_LEARNED, C_STATIC, C_HEUR = "#2a78d6", "#4a3aa7", "#eb6834"
C_WORSE = "#a8322a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COL = 3.45
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
})

def load(d):
    o = {}
    for f in glob.glob(os.path.join(R, "results", "final", d, "*.json")):
        try: r = json.load(open(f))
        except Exception: continue
        if r.get("steps") == 42: o[r["seed"]] = r["energy_mj_total"] / 1000.0
    return o

def ci95(x, n=20000):
    rng = np.random.default_rng(0)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)

ARMS = [("No power save", "main/notwt", C_WORSE),
        ("Static TWT", "main/static_s11", C_STATIC),
        ("Static+OFDMA", "rows/r3_static_ofdma", C_STATIC),
        ("Hand-designed", "heur2/eval_free", C_HEUR)]
LEARNED = [("BC", "bc"), ("DDQN", "ddqn"), ("Duel+CQL", "dueling"),
           ("DQN+CQL", "dqn"), ("IQL", "iql")]

fig, (ax, bx) = plt.subplots(2, 1, figsize=(COL, 3.25),
                             gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": .62})

# ---- (a) energy per arm, 95% CI -------------------------------------------
labels, means, los, his, cols = [], [], [], [], []
for lab, d, c in ARMS:
    o = load(d)
    if len(o) < 40: continue
    x = np.array(list(o.values())); lo, hi = ci95(x)
    labels.append(lab); means.append(x.mean()); los.append(x.mean()-lo); his.append(hi-x.mean()); cols.append(c)
F = json.load(open(os.path.join(R, "results", "frozen.json")))
for lab, key in LEARNED:
    g = F["algorithms"].get(key)
    if not g: continue
    per = [v["energy_J"] for v in g["per_fit"].values()]
    x = np.array(per); lo, hi = ci95(x)
    labels.append(lab); means.append(x.mean()); los.append(x.mean()-lo); his.append(hi-x.mean())
    cols.append(C_LEARNED)
y = np.arange(len(labels))
ax.barh(y, means, xerr=[los, his], color=cols, height=.68,
        error_kw=dict(ecolor=INK2, elinewidth=.7, capsize=2), zorder=3)
ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlabel("station energy (J)")
for i, m in enumerate(means):
    ax.text(m + his[i] + max(means)*.022, i, f"{m:.0f}", va="center",
            fontsize=6.0, color=INK)
ax.set_xlim(0, max(means)*1.18)
ax.grid(True, axis="x", color=GRID, lw=.5, alpha=.7); ax.set_axisbelow(True)
for sp in ("right", "top"): ax.spines[sp].set_visible(False)
ax.text(.985, .04, "bars: 95% CI over 50 paired seeds", transform=ax.transAxes,
        ha="right", fontsize=5.8, color=INK2)

# ---- (b) what IQL does over an episode -------------------------------------
tr = sorted(glob.glob(os.path.join(R, "results", "final", "trace", "*.json")))
drawn = 0
for f in tr:
    try: r = json.load(open(f))
    except Exception: continue
    t = r.get("trace")
    if not t or not r.get("stepped"): continue
    sched = [(s["action"] % 380) // 19 for s in t]
    bx.step(range(len(sched)), sched, where="post", lw=1.0,
            color=C_LEARNED, alpha=.75 if drawn else 1.0, zorder=3)
    if drawn == 0 and r.get("load_step_at_s", -1) > 0:
        k = r["load_step_at_s"] / 2.048
        bx.axvline(k, color=C_HEUR, lw=1.0, ls="--", zorder=2)
        bx.text(k + .6, 1.5, f"load x{r['load_step_mult']:g}",
                fontsize=6.0, color=C_HEUR)
    drawn += 1
    if drawn >= 3: break
if drawn:
    bx.set_xlabel("decision within episode"); bx.set_ylabel("TWT schedule")
    bx.set_xlim(0, 42); bx.set_ylim(-0.6, 19.6); bx.set_yticks([0, 5, 10, 15, 19])
    bx.grid(True, color=GRID, lw=.5, alpha=.7); bx.set_axisbelow(True)
    for sp in ("right", "top"): bx.spines[sp].set_visible(False)
else:
    bx.text(.5, .5, "no traces yet", ha="center", transform=bx.transAxes, color=INK2)
    print("  WARNING: panel (b) empty -- trace run not finished", file=sys.stderr)

fig.savefig(f"{OUT}/fig_results.pdf"); fig.savefig(f"{OUT}/fig_results.png")
plt.close(fig)
print(f"  fig_results: {len(labels)} arms in (a), {drawn} traced episodes in (b)")
