#!/usr/bin/env python3
"""Build a training dataset from collected JSONL, with a selectable reward.

Two things this fixes relative to the historical pipeline:

1. DONES. The shipped 100k dataset has `dones` all zero, so the Bellman backup
   bootstraps across episode boundaries -- pairing the last state of one episode
   with the first of an unrelated one. Each JSONL file is one episode, so the
   boundary is known exactly and the final transition of each is marked terminal.

2. REWARD IS RECOMPUTED, NOT REPLAYED. Transitions store raw per-station
   observations, so any objective can be built offline from one collection. That
   is what lets the energy-weighted and latency-constrained variants share a
   single 20-hour collection instead of needing two.

Rewards:
  balanced|energy|throughput  the existing three presets (reward_functions.py)
  constrained                 minimise energy subject to a queueing-delay budget,
                              via Lagrangian relaxation:
                                  r = -(E / E_scale) - lam * mean_i max(0, d_i - L)/L
                              Sweeping L traces the Pareto frontier and shows where
                              OFDMA enters the optimal policy -- rather than picking
                              one budget that happens to give the answer we want.
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.abspath(os.path.join(HERE, "..", "..", "RL-room"))
sys.path.insert(0, RL)
from reward_functions import reward_per_sta_with_fairness as preset_reward  # noqa: E402

# E_SCALE must keep the DISCOUNTED RETURN inside the trainer's target clip, not
# merely keep the reward "O(1)". train-dqn-offline-v2.py clips targets to [-10, 10]
# at four sites (lines 497, 522, 549, 589) and the default gamma is 0.95, so the
# largest representable value is 10 and |V| ~ |r|/(1-gamma) = 20*|r|. The original
# E_SCALE = 50 gave mean r = -7.98 on pilot data => |V| ~ 160, SIXTEEN TIMES past
# the clip: every target saturates at -10, all 760 actions get the same target, and
# the Q-function carries no action information at all. The constrained arm would
# have trained to completion and learned nothing, with no error raised.
#
# Median per-station energy is ~368 mJ and the max ~998, so 1000 puts the energy
# term in [0, 1] and |V| ~ 7.4 -- inside the clip with headroom.
E_SCALE_DEFAULT = 1000.0


def constrained_reward(obs, budget_ms, lam, e_scale=E_SCALE_DEFAULT):
    e = float(np.mean([s["energy_consumed_mj"] for s in obs]))
    d = np.array([s.get("queue_delay_ms", 0.0) for s in obs], dtype=float)
    viol = float(np.mean(np.maximum(0.0, d - budget_ms) / max(budget_ms, 1e-9)))
    return -(e / e_scale) - lam * viol, e, viol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory of per-episode .jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reward", default="balanced",
                    choices=["balanced", "energy", "throughput", "constrained"])
    ap.add_argument("--budget-ms", type=float, default=3000.0)
    # Measured on the live data at E_SCALE=1000: energy-term sd 0.182, violation-term
    # sd 0.126. lam ~ 1.5 weights the two comparably; lam << 1 is energy-only (the
    # violation term then contributes ~3% of the objective's variation) and lam >> 10
    # is latency-only. Sweep {0.5, 1.5, 5, 15} to trace the frontier.
    ap.add_argument("--lam", type=float, default=1.5)
    ap.add_argument("--gamma", type=float, default=0.95,
                    help="only used to check the return fits the trainer's clip")
    # A pre-registered EPISODE-level holdout. The trainer's own split is
    # transition-level and seeded separately, so an FQE run that makes its own
    # split evaluates the policy on states it MEMORIZED: measured 83.69% greedy
    # match on the trainer's train portion vs 0.00% on its val portion (uniform
    # null 0.13%). V_hat computed on those states is optimistically biased.
    # Reserving whole episodes up front, written to a separate file, makes the
    # contamination impossible rather than merely unlikely.
    ap.add_argument("--holdout-frac", type=float, default=0.15,
                    help="fraction of EPISODES reserved for FQE; 0 disables")
    ap.add_argument("--holdout-seed", type=int, default=12345)
    ap.add_argument("--e-scale", type=float, default=E_SCALE_DEFAULT,
                    help="mJ divisor for the energy term; 1000 keeps it in [0,1]")
    # code/train_offline.py raises the clip to 50 (RL-room's copy hardcodes 10 and
    # is left untouched). At gamma=0.95 that admits |r| up to 2.5.
    ap.add_argument("--clip", type=float, default=50.0,
                    help="trainer's target clip; build fails if the return exceeds it")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.raw, "*.jsonl")))
    S, A, R, NS, D = [], [], [], [], []
    RS, RF, RN, EM, VI = [], [], [], [], []
    EP, SEED = [], []          # episode id + source seed, for trajectory-disjoint splits
    episodes = skipped = 0

    for fp in files:
        rows = [json.loads(l) for l in open(fp) if l.strip()]
        if not rows:
            continue
        ep_id = episodes
        episodes += 1
        spawn = int(rows[0].get("spawn_id", ep_id))
        for k, t in enumerate(rows):
            s = np.asarray(t["state"], dtype=np.float32)
            ns = np.asarray(t["next_state"], dtype=np.float32)
            if s.shape[0] != 130 or ns.shape[0] != 130:
                skipped += 1
                continue
            obs = t["sta_observations"]
            if a.reward == "constrained":
                r, e, v = constrained_reward(obs, a.budget_ms, a.lam, a.e_scale)
                rs = rf = rn = 0.0
            else:
                r, rs, rf, rn = preset_reward({"sta_observations": obs},
                                              return_components=True, preset=a.reward)
                e = float(np.mean([x["energy_consumed_mj"] for x in obs])); v = 0.0
            S.append(s); NS.append(ns); A.append(int(t["action"])); R.append(float(r))
            # terminal iff last transition of this episode -- the fix for all-zero dones
            D.append(1.0 if k == len(rows) - 1 else 0.0)
            RS.append(rs); RF.append(rf); RN.append(rn); EM.append(e); VI.append(v)
            EP.append(ep_id); SEED.append(spawn)

    D = np.asarray(D, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    arrays = dict(
        states=np.asarray(S, dtype=np.float32), actions=np.asarray(A, dtype=np.int64),
        rewards=np.asarray(R, dtype=np.float64), next_states=np.asarray(NS, dtype=np.float32),
        dones=D, energy_mj=np.asarray(EM, dtype=np.float32),
        episode_ids=np.asarray(EP, dtype=np.int32),
        spawn_ids=np.asarray(SEED, dtype=np.int32),
        delay_violation=np.asarray(VI, dtype=np.float32),
        reward_kind=np.array(a.reward), budget_ms=np.array(a.budget_ms), lam=np.array(a.lam))
    # train-dqn-offline-v2.py:301 RECOMBINES the reward from r_sum/r_fairness/
    # r_network whenever all three keys exist, ignoring `rewards` entirely. The
    # constrained objective has no such decomposition, so writing zeros for them
    # would silently train on reward == 0 with no error raised. Omitting the keys
    # makes the trainer fall back to the stored scalar (its line 311), which is
    # what the constrained variant needs.
    if a.reward != "constrained":
        arrays.update(r_sum=np.asarray(RS, dtype=np.float32),
                      r_fairness=np.asarray(RF, dtype=np.float32),
                      r_network=np.asarray(RN, dtype=np.float32))
    # Split by EPISODE, never by transition, so no episode contributes to both.
    if a.holdout_frac > 0:
        eps = np.unique(arrays["episode_ids"])
        rng = np.random.default_rng(a.holdout_seed)
        n_ho = max(1, int(round(a.holdout_frac * len(eps))))
        ho_eps = set(rng.permutation(eps)[:n_ho].tolist())
        is_ho = np.isin(arrays["episode_ids"], list(ho_eps))
        base, ext = os.path.splitext(a.out)
        ho_path = f"{base}_holdout{ext}"
        np.savez_compressed(ho_path, **{k: (v[is_ho] if getattr(v, "ndim", 0) and v.shape[:1] == is_ho.shape else v)
                                        for k, v in arrays.items()})
        np.savez_compressed(a.out, **{k: (v[~is_ho] if getattr(v, "ndim", 0) and v.shape[:1] == is_ho.shape else v)
                                      for k, v in arrays.items()})
        print(f"  holdout: {n_ho} of {len(eps)} episodes ({int(is_ho.sum()):,} transitions) "
              f"-> {ho_path}")
        print(f"           train on {a.out}; FQE evaluates on the holdout ONLY")
    else:
        np.savez_compressed(a.out, **arrays)

    R = np.asarray(R)
    # GATE: a reward whose discounted return exceeds the trainer's clip trains
    # silently to a constant Q. Caught here rather than after hours of training.
    v_est = abs(float(R.mean())) / (1.0 - a.gamma)
    v_max = abs(float(R.min())) / (1.0 - a.gamma)
    print(f"  wrote {a.out}")
    _split_note = " BEFORE the holdout split" if a.holdout_frac > 0 else ""
    print(f"    {len(S):,} transitions from {episodes} episodes "
          f"({skipped} skipped){_split_note}")
    print(f"    dones: {int(D.sum())} terminal  (must equal episode count: {episodes})")
    print(f"    reward '{a.reward}': mean {R.mean():+.4f}  range [{R.min():+.3f},{R.max():+.3f}]")
    acts = np.asarray(A)
    print(f"    actions: {len(np.unique(acts))} distinct of 760, "
          f"OFDMA-on share {100*(acts>=380).mean():.1f}%")
    if a.reward == "constrained":
        print(f"    budget {a.budget_ms:.0f} ms, lambda {a.lam}; "
              f"mean violation {np.mean(VI):.3f}")
        es, vs = float(np.std(np.asarray(EM) / a.e_scale)), float(a.lam * np.std(VI))
        print(f"    objective balance: energy sd {es:.3f} vs violation sd {vs:.3f} "
              f"-> energy is {100*es/max(es+vs,1e-9):.0f}% of the variation")
    print(f"    return check: |V| ~ {v_est:.1f} (worst-case {v_max:.1f}) "
          f"vs trainer clip {a.clip:.0f}  [gamma {a.gamma}]")
    if v_est > a.clip:
        print(f"    *** FAIL: |V| {v_est:.1f} exceeds the clip {a.clip:.0f}. Every target "
              f"saturates and all actions become indistinguishable.")
        print(f"    *** Raise E_SCALE by ~{v_est/a.clip:.1f}x, or raise the trainer's clip.")
        raise SystemExit(2)
    if v_max > a.clip:
        print(f"    warning: the worst-case return {v_max:.1f} exceeds the clip; the "
              f"reward tail is compressed, though the mean fits.")


if __name__ == "__main__":
    main()
