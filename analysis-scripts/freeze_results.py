#!/usr/bin/env python3
"""
freeze_results.py -- one JSON holding every number the paper may cite.

Every figure in the manuscript is currently typed into prose by hand, which is
how "20.23 -> 18.00", "372.2 vs 372.1" and "roughly 7%" reached reviewers. This
emits results/frozen.json: one record per arm, with the experiment definition
attached, so tables and in-text numbers can be generated from it and checked
against it rather than retyped.
"""
import glob, json, os, re
import numpy as np
from scipy import stats

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF = dict(harness="scenario4_rl_twt_ofdma", window="42-decision RL window",
           eval_seeds="90000-90049 (50, paired)",
           tuning_seeds="80000-80014 (disjoint)",
           collection_seeds="30000+ (disjoint)",
           ofdma_scheduler="service-period aligned, sleep guard disabled",
           cca_current_ma=273.0, selection_rule="highest FQE value, drift gate not applied")


def load(d):
    o = {}
    for f in glob.glob(os.path.join(R, "results/final", d, "*.json")):
        m = re.search(r"_s(9\d+)\.json$", os.path.basename(f))
        if m:
            o[int(m.group(1))] = json.load(open(f))
    return o


def summarise(o):
    g = lambda k, s=1.0: float(np.mean([x[k] for x in o.values()]) / s)
    return dict(n=len(o), energy_J=g("energy_mj_total", 1000), throughput_mbps=g("throughput_mbps"),
                delay_ms=g("delay_ms_mean"), viol=g("delay_budget_viol_frac"),
                drops=g("drops"), ofdma_share=100 * g("ofdma_on_share"))


ARMS = {"r1_plain": "main/notwt", "r2_static_twt": "main/static_s11",
        "r3_static_twt_ofdma": "rows/r3_static_ofdma", "r4_random_twt": "main/random",
        "r5_random_twt_ofdma": "rows/r5_random_ofdma", "r6_ofdma_only": "rows/r6_ofdma_no_twt",
        "r7_random_joint": "rows/r7_rand_no_twt",
        "r8_handdesigned_ofdma_on": "heur2/eval_ofdma_on",
        "r8b_handdesigned_free": "heur2/eval_free"}

out = {"definition": DEF, "arms": {}, "algorithms": {}}
base = None
for k, d in ARMS.items():
    o = load(d)
    if o:
        out["arms"][k] = summarise(o)
        if k == "r8b_handdesigned_free":
            base = o

for a in ("iql", "dqn", "dueling", "ddqn", "bc"):
    fits, per = [], {}
    for s in range(42, 52):
        o = load(f"seeds10/s{s}_{a}")
        if len(o) >= 45:
            per[s] = summarise(o)
            fits.append(per[s]["energy_J"])
    if not fits:
        continue
    e = np.array(fits)
    rec = dict(n_fits=len(e), mean_J=float(e.mean()), sd_J=float(e.std(ddof=1)),
               min_J=float(e.min()), max_J=float(e.max()),
               throughput_mbps=float(np.mean([v["throughput_mbps"] for v in per.values()])),
               ofdma_share=float(np.mean([v["ofdma_share"] for v in per.values()])),
               per_fit={str(k): v for k, v in per.items()})
    if base is not None:
        # paired test of the best fit against the strongest hand-designed baseline
        bo = load(f"seeds10/s{min(per, key=lambda s: per[s]['energy_J'])}_{a}")
        kk = sorted(set(bo) & set(base))
        x = np.array([bo[i]["energy_mj_total"] for i in kk]) / 1000
        y = np.array([base[i]["energy_mj_total"] for i in kk]) / 1000
        rec["vs_handdesigned_free_pct"] = float(100 * (x.mean() - y.mean()) / y.mean())
        rec["wilcoxon_p"] = float(stats.wilcoxon(x, y).pvalue)
        # The headline comparison: the MEAN over the ten fits, per seed, against
        # the same baseline. This is the number that pairs arithmetically with
        # mean_J; vs_handdesigned_free_pct above is the best fit of ten and is
        # the right number only where the text says "best fit".
        allo = [load(f"seeds10/s{s2}_{a}") for s2 in per]
        kk2 = sorted(set(base).intersection(*[set(o) for o in allo]))
        xm = np.array([np.mean([o[i]["energy_mj_total"] for o in allo]) for i in kk2]) / 1000
        ym = np.array([base[i]["energy_mj_total"] for i in kk2]) / 1000
        rec["vs_hand_meanfit_pct"] = float(100 * (xm.mean() - ym.mean()) / ym.mean())
        rec["wilcoxon_p_meanfit"] = float(stats.wilcoxon(xm, ym).pvalue)
        rng = np.random.default_rng(0)
        bs = [100 * (xm[j].mean() - ym[j].mean()) / ym[j].mean()
              for j in (rng.integers(0, len(xm), len(xm)) for _ in range(20000))]
        rec["vs_hand_meanfit_ci95"] = [float(np.percentile(bs, 2.5)),
                                       float(np.percentile(bs, 97.5))]
        rec["n_paired_seeds"] = len(kk2)
    out["algorithms"][a] = rec

p = os.path.join(R, "results", "frozen.json")
json.dump(out, open(p, "w"), indent=2)
print(f"  wrote {p}")
print(f"  {len(out['arms'])} non-learned arms, {len(out['algorithms'])} algorithms x 10 fits")
for a, r in sorted(out["algorithms"].items(), key=lambda kv: kv[1]["mean_J"]):
    print(f"    {a:9} {r['mean_J']:7.1f} +/- {r['sd_J']:4.1f} J   "
          f"vs best hand-designed {r.get('vs_handdesigned_free_pct', float('nan')):+6.1f}%  "
          f"p={r.get('wilcoxon_p', float('nan')):.1e}")
