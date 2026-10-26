#!/usr/bin/env python3
"""
build_energy_v2.py -- a reward preset that actually targets energy.

THE DEFECT THIS FIXES. The shipped "energy" preset is (0.25, 0.15, 0.60) over
(r_sum, r_fairness, r_network). Measured on the training data it is the LEAST
energy-correlated of the three presets, corr(r, E) = -0.092 against -0.165
(balanced) and -0.199 (throughput), and in evaluation it produced the highest
energy of the three for all five algorithms. Two compounding causes:

  1. r_network's energy term is -0.20*tanh(total_energy / 400), but total_energy
     is the per-decision sum over 16 stations, which spans 1390-14167 mJ here.
     tanh(E/400) therefore lies in [0.99808, 1.00000] with sd 4.1e-5: saturated
     for 99.99% of transitions. r_network carries no energy signal at all --
     corr(r_network, E) = +0.0017.
  2. The preset puts 0.60 on that energy-blind component (0.445 of reward sd)
     and only 0.25 on r_sum, which is the one component that IS energy-aligned
     (corr = -0.238). That is why the *throughput* preset, which puts 0.55 on
     r_sum, beats the *energy* preset on energy.

Fixing the tanh scale alone is necessary but not sufficient: a corrected term
contributes 0.024 of reward sd against r_network's 0.445, so it is drowned. This
builder instead replaces the energy component with an explicit, unsaturated one
and reweights:

    r = 0.30 * z(-tanh(E / 8000))  +  0.55 * z(r_sum)  +  0.15 * z(r_fairness)

E* = 8000 mJ is ~1.36x the mean per-decision energy, chosen so the term stays in
tanh's responsive region over the whole observed range: tanh spans 0.172-0.944
instead of 0.998-1.000.

WHY 0.55 ON r_sum AND NOT LESS. An energy-only objective (weight 1.0 on the
energy term) reaches corr(r, E) = -0.98 but has corr(r, drops) = +0.258 -- it
pays the policy to starve traffic. The weights above are the point where the
objective still targets energy hard (corr = -0.712, 8x the shipped preset) while
keeping corr(r, drops) negative, so dropping packets is penalised as it is under
every other preset.

NO RE-COLLECTION IS NEEDED. Per-decision per-station energy is already in the
state vector: scenario4_collect.py:106 writes 8 fields per station and the last
is d_energy[i], so E = sum_i state[2 + i*8 + 7].

NORMALISATION IS FIT ON TRAIN ONLY and applied to the holdout, so the holdout
contributes nothing to the reward definition. The result is then affinely mapped
onto the shipped preset's scale (mean -1.156, sd 0.601); an affine map leaves the
optimal policy and every correlation unchanged, and it keeps |discounted return|
in the range the trainer's target clip was set for.

THE PROVENANCE COMPONENTS ARE DELIBERATELY RENAMED. train_offline.py:324
recombines a reward from preset weights whenever r_sum/r_fairness/r_network are
present, which would silently discard this objective. They are stored as
r_sum_orig/r_fairness_orig/r_network_orig so the trainer falls through to the
stored `rewards` (train_offline.py:334).
"""
import argparse, os
import numpy as np

E_SCALE = 8000.0
W_ENERGY, W_SUM, W_FAIR = 0.30, 0.55, 0.15
IDX = 2 + np.arange(16) * 8          # per-station block starts
OFF_ENERGY = 7                       # d_energy is the 8th field


def total_energy(states):
    return states[:, IDX + OFF_ENERGY].sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/stat_ofdma.npz")
    ap.add_argument("--holdout", default="data/stat_ofdma_holdout.npz")
    ap.add_argument("--out", default="data/stat_e2.npz")
    a = ap.parse_args()

    tr = dict(np.load(a.train))
    ho = dict(np.load(a.holdout))

    Etr, Eho = total_energy(tr["states"]), total_energy(ho["states"])
    etr, eho = -np.tanh(Etr / E_SCALE), -np.tanh(Eho / E_SCALE)

    # z-statistics fit on TRAIN ONLY
    stats = {}
    def zfit(name, v):
        stats[name] = (float(v.mean()), float(v.std()))
        return (v - stats[name][0]) / stats[name][1]
    def zapp(name, v):
        m, s = stats[name]
        return (v - m) / s

    r_tr = (W_ENERGY * zfit("e", etr)
            + W_SUM * zfit("s", tr["r_sum"])
            + W_FAIR * zfit("f", tr["r_fairness"]))
    r_ho = (W_ENERGY * zapp("e", eho)
            + W_SUM * zapp("s", ho["r_sum"])
            + W_FAIR * zapp("f", ho["r_fairness"]))

    # affine map onto the shipped energy preset's scale (train statistics)
    shipped = 0.25 * tr["r_sum"] + 0.15 * tr["r_fairness"] + 0.60 * tr["r_network"]
    k = float(shipped.std() / r_tr.std())
    b = float(shipped.mean() - k * r_tr.mean())
    r_tr, r_ho = k * r_tr + b, k * r_ho + b

    for name, d, r, E in (("train", tr, r_tr, Etr), ("holdout", ho, r_ho, Eho)):
        out = dict(states=d["states"], actions=d["actions"],
                   next_states=d["next_states"], dones=d["dones"],
                   episode_ids=d["episode_ids"],
                   rewards=r.astype(np.float64),
                   r_sum_orig=d["r_sum"], r_fairness_orig=d["r_fairness"],
                   r_network_orig=d["r_network"])
        base, ext = os.path.splitext(a.out)
        fp = a.out if name == "train" else f"{base}_holdout{ext}"
        np.savez_compressed(fp, **out)
        tp = d["states"][:, 0]
        dr = d["states"][:, IDX + 6].sum(axis=1)
        print(f"{name:8} n={len(r):>7,}  r: mean={r.mean():+.3f} sd={r.std():.3f}  "
              f"corr(r,E)={np.corrcoef(r, E)[0, 1]:+.4f}  "
              f"corr(r,tp)={np.corrcoef(r, tp)[0, 1]:+.4f}  "
              f"corr(r,drops)={np.corrcoef(r, dr)[0, 1]:+.4f}  -> {fp}")

    # return gate: the trainer clips Bellman targets at 50
    G, EP = 0.95, tr["episode_ids"]
    rets = [abs(sum(G ** k_ * x for k_, x in enumerate(r_tr[EP == e])))
            for e in np.unique(EP)[:500]]
    mx = max(rets)
    print(f"\n|discounted return| over 500 train episodes: mean={np.mean(rets):.1f} max={mx:.1f}"
          f"   (target clip 50) -> {'OK' if mx < 50 else 'EXCEEDS CLIP'}")
    if mx >= 50:
        raise SystemExit("return gate failed: discounted return exceeds the trainer's clip")
    print(f"affine map applied: r <- {k:.4f}*r + {b:+.4f}")


if __name__ == "__main__":
    main()
