#!/usr/bin/env python3
"""Compare policies on the per-episode JSONs that evaluate.py writes.

PAIRED, because every policy is run on the same seed list and a seed fixes the
whole world (traffic, the load-step time, the step multiplier). Comparing
unpaired means across seeds would bury the effect under between-seed variance
that the pairing removes outright.

Reported per metric, against a named reference policy:
  - mean difference with a paired bootstrap CI (resampling SEEDS, not episodes)
  - Wilcoxon signed-rank p, which assumes no distributional shape
  - Cohen's dz on the paired differences
  - Holm-Bonferroni across the policies tested, because testing five baselines on
    one metric and quoting the smallest p is how a null result gets published

Energy is the headline: mJ per station per decision window. Lower is better, and
the sign convention is handled per metric so "improvement" always reads positive.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

# metric -> (lower_is_better, label, unit)
METRICS = {
    "energy_mj_per_sta_per_step": (True,  "Energy/STA/step", "mJ"),
    "delay_ms_mean":              (True,  "Mean queue delay", "ms"),
    "delay_ms_p95":               (True,  "p95 queue delay", "ms"),
    "delay_budget_viol_frac":     (True,  "Budget violation", "frac"),
    "drops":                      (True,  "MPDU drops", "count"),
    "throughput_mbps":            (False, "Throughput", "Mbps"),
    "fairness":                   (False, "Jain fairness", "-"),
}


def load(dirs):
    """policy -> {seed: record}"""
    by = defaultdict(dict)
    for d in dirs:
        for fp in glob.glob(os.path.join(d, "*.json")):
            r = json.load(open(fp))
            by[r["policy"]][int(r["seed"])] = r
    return by


def paired_bootstrap(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon(d):
    """Signed-rank, normal approximation with tie correction. No SciPy needed."""
    d = d[d != 0]
    n = len(d)
    if n < 6:
        return float("nan")
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, dtype=float)
    a = np.abs(d)[order]
    i = 0
    while i < n:                       # average ranks within ties
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        ranks[i:j + 1] = np.mean(np.arange(i + 1, j + 2))
        i = j + 1
    r = np.empty(n, dtype=float); r[order] = ranks
    W = float(np.sum(r[d > 0]))
    mu = n * (n + 1) / 4.0
    _, tc = np.unique(np.abs(d), return_counts=True)
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - np.sum(tc ** 3 - tc) / 48.0)
    if sd == 0:
        return float("nan")
    z = (W - mu) / sd
    # two-sided normal tail without SciPy
    from math import erfc, sqrt
    return float(erfc(abs(z) / sqrt(2.0)))


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--ref", required=True, help="reference policy name (substring match)")
    ap.add_argument("--metrics", nargs="*", default=list(METRICS))
    ap.add_argument("--subset", choices=["all", "stepped", "stationary"], default="all")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    by = load(a.dirs)
    if not by:
        raise SystemExit("no results found")
    refs = [p for p in by if a.ref in p]
    if not refs:
        raise SystemExit(f"--ref '{a.ref}' matched nothing; policies present: {sorted(by)}")
    if len(refs) > 1:
        # An aborted run can leave a stray policy whose name also contains the ref
        # substring -- e.g. a heuristic config from a previous sweep with a single
        # episode. Exiting here would throw away a finished evaluation at the last
        # step, so take the candidate with the most seeds and say what was ignored.
        refs.sort(key=lambda p: -len(by[p]))
        print(f"  NOTE: --ref '{a.ref}' matched {len(refs)} policies; using the one")
        print(f"        with the most seeds and ignoring the rest:")
        for p in refs:
            mark = "USING " if p is refs[0] else "ignore"
            print(f"          {mark} {p}  ({len(by[p])} seeds)")
    ref = refs[0]
    others = sorted(p for p in by if p != ref)

    print(f"  reference: {ref}")
    print(f"  compared : {len(others)} policies   subset={a.subset}")
    print()
    out = {}
    for metric in a.metrics:
        lower_better, label, unit = METRICS[metric]
        rows, ps = [], []
        for p in others:
            seeds = sorted(set(by[ref]) & set(by[p]))
            if a.subset == "stepped":
                seeds = [s for s in seeds if by[ref][s]["stepped"]]
            elif a.subset == "stationary":
                seeds = [s for s in seeds if not by[ref][s]["stepped"]]
            if len(seeds) < 6:
                continue
            r = np.array([by[ref][s][metric] for s in seeds], float)
            o = np.array([by[p][s][metric] for s in seeds], float)
            # positive = reference is better than p
            diff = (o - r) if lower_better else (r - o)
            lo, hi = paired_bootstrap(diff)
            pv = wilcoxon(diff)
            dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan")
            rows.append(dict(policy=p, n=len(seeds), ref_mean=float(r.mean()),
                             other_mean=float(o.mean()), gain=float(diff.mean()),
                             ci_lo=lo, ci_hi=hi, p=pv, dz=dz))
            ps.append(pv)
        if not rows:
            continue
        adj = holm(np.array([x if x == x else 1.0 for x in ps]))
        for row, q in zip(rows, adj):
            row["p_holm"] = float(q)
        out[metric] = rows

        print(f"  {label} ({unit}) -- positive gain = {ref} better; lower_is_better={lower_better}")
        print(f"    {'policy':44s} {'n':>4s} {'ref':>9s} {'other':>9s} "
              f"{'gain':>9s} {'95% CI':>19s} {'p_holm':>9s} {'dz':>6s}")
        for row in sorted(rows, key=lambda x: -x["gain"]):
            sig = "*" if row["p_holm"] < 0.05 and row["ci_lo"] * row["ci_hi"] > 0 else " "
            print(f"    {row['policy'][:44]:44s} {row['n']:4d} {row['ref_mean']:9.3f} "
                  f"{row['other_mean']:9.3f} {row['gain']:+9.3f} "
                  f"[{row['ci_lo']:+8.3f},{row['ci_hi']:+8.3f}] {row['p_holm']:9.2e} "
                  f"{row['dz']:+6.2f}{sig}")
        print()

    if a.out:
        json.dump(dict(ref=ref, subset=a.subset, results=out), open(a.out, "w"), indent=2)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
