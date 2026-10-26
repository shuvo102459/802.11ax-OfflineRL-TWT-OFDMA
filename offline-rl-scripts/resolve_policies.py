#!/usr/bin/env python3
"""
resolve_policies.py -- name the exact checkpoint each learned arm deploys.

Every learned row in the final comparison must be traceable to one file. This
resolves (algorithm, training seed) to the FQE-selected milestone, applying the
same rule used throughout: highest FQE value among snapshots whose long-window
drift is below 2%, falling back to the final epoch when none converged. It
prints the fallback explicitly rather than hiding it, because four of ten pairs
hit it last time.
"""
import glob, json, os, sys

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGOS = ["bc", "dqn", "ddqn", "iql", "dueling"]


def best(fqe_dir, ms_dir, algo):
    rows = []
    for fp in glob.glob(os.path.join(fqe_dir, "*.json")):
        try:
            rows += json.load(open(fp))
        except Exception:
            pass
    cand = None
    for r in rows:
        ld = r.get("long_drift")
        if ld is None or ld != ld or ld >= 0.02:
            continue
        n = r["policy"].split("/")[-1]
        if f"_{algo}_" in n and (cand is None or r["fqe_value"] > cand[1]):
            cand = (n, r["fqe_value"])
    if cand and os.path.exists(os.path.join(ms_dir, cand[0])):
        return os.path.join(ms_dir, cand[0]), cand[1], "fqe"
    return None, None, "fallback"


def main():
    out = []
    for tseed, fq, ms, fin in (
            ("42", "results/fqe_e2", "data/ms_e2", "data/model_e2_%s.npz"),
            ("43", "results/fqe_e2s43", "data/ms_e2s43", "data/model_e2s43_%s.npz"),
            ("44", "results/fqe_e2s44", "data/ms_e2s44", "data/model_e2s44_%s.npz")):
        for a in ALGOS:
            p, v, how = best(os.path.join(R, fq), os.path.join(R, ms), a)
            if how == "fallback":
                p = os.path.join(R, fin % a)
                if not os.path.exists(p):
                    print(f"  MISSING  seed {tseed} {a}", file=sys.stderr); continue
            out.append((tseed, a, p, how, v))
    for tseed, a, p, how, v in out:
        tag = f"e2s{tseed}_{a}"
        print(f"{tag}\t{p}\t{how}\t{'' if v is None else f'{v:+.2f}'}")


if __name__ == "__main__":
    main()
