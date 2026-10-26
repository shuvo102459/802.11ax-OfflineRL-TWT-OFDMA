#!/usr/bin/env python3
"""Fitted Q Evaluation for offline checkpoint selection.

WHY NOT IMPORTANCE SAMPLING. The behaviour policy is uniform over 760 actions
and the learned policies are deterministic greedy, so a transition contributes
to a one-step IS estimate only when argmax Q(s) happens to equal the logged
action -- probability ~1/760. For a deterministic target policy the Kish ESS of
the self-normalised estimator reduces exactly to that match count:

    w_i in {0, 760}  =>  ESS = (sum w)^2 / sum w^2 = #matches

Measured on the pilot: 26 matches in 12,600 transitions (0.21%), ESS = 26.00;
4 matches on a 20% validation split. At 100k transitions that is ~200 matches
over the whole dataset and ~40 on a held-out split -- still far short of what
ranking ten checkpoints needs. The failure is the ACTION-SPACE CARDINALITY, so
neither more data nor a different split fixes it: a trajectory-disjoint split
changes which transitions are held out, not the 1/760 match probability.

FQE has no importance weights. It fits a separate critic Q_psi to the target
policy by repeated regression on the logged transitions,

    Q_psi(s,a)  <-  r + gamma (1-d) Q_psi'(s', pi(s'))          pi = argmax Q_ckpt

and reports V_hat(pi) = E_{s0}[Q_psi(s0, pi(s0))] over held-out trajectory
starts. This is the standard offline-selection protocol (Le, Voloshin & Yue,
ICML 2019; Paine et al., "Hyperparameter Selection for Offline RL", 2020).

FQE is a model fit, not an unbiased estimate: it can be optimistic on actions
the data covers poorly, which is exactly the regime a greedy policy walks into.
It is used here to RANK checkpoints of one method, and the ranking is reported
with its own convergence diagnostic. Absolute performance claims come from the
ns-3 evaluation, never from this number.
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn

NUM_TWT = 380


# ── loading a checkpoint's greedy policy (numpy; matches the trainer's nets) ───

def load_q(path, mu_fallback=None, sd_fallback=None):
    """Return a callable S -> Q(S), handling plain and dueling saves.

    NORMALIZATION. A finished model saved by Net.save() carries state_mean/
    state_std, but a mid-training CHECKPOINT written by save_checkpoint() does
    NOT -- it stores only qnet_*/qtgt_*/curve_*. Without a fallback this function
    silently evaluated Q on RAW states, giving wrong Q values, wrong greedy
    actions, and therefore a meaningless FQE ranking -- on exactly the inputs FQE
    exists to rank. The caller must pass the statistics derived from the dataset
    the policy was trained on, computed the way the trainer computes them
    (train_offline.py:358-359, over the WHOLE dataset before any split):

        mu = states.mean(axis=0);  sd = states.std(axis=0) + 1e-8
    """
    m = np.load(path, allow_pickle=True)
    keys = set(m.files)
    trained_on = str(m["train_data"]) if "train_data" in keys else None

    # a full training checkpoint prefixes the online net with qnet_
    pre = "qnet_" if any(k.startswith("qnet_") for k in keys) else ""
    g = lambda k: m[pre + k]

    if "state_mean" in keys:
        mu, sd = m["state_mean"], m["state_std"]
    elif mu_fallback is not None:
        mu, sd = mu_fallback, sd_fallback
    else:
        raise SystemExit(
            f"{os.path.basename(path)} carries no state_mean/state_std (it is a "
            f"training checkpoint) and no fallback was supplied. Evaluating it on "
            f"raw states would silently produce a wrong ranking.")
    dueling = (pre + "W_V") in keys

    def q(S):
        x = S.astype(np.float64)
        if mu is not None:
            x = (x - mu) / np.where(sd > 0, sd, 1.0)
        h = np.maximum(0.0, x @ g("W1") + g("b1"))
        h = np.maximum(0.0, h @ g("W2") + g("b2"))
        if dueling:
            V = h @ g("W_V") + g("b_V")
            A = h @ g("W_A") + g("b_A")
            return V + A - A.mean(axis=1, keepdims=True)
        return h @ g("W3") + g("b3")

    return q, dueling, pre, trained_on


def greedy_actions(q, S, chunk=8192):
    out = np.empty(len(S), dtype=np.int64)
    for i in range(0, len(S), chunk):
        out[i:i + chunk] = q(S[i:i + chunk]).argmax(axis=1)
    return out


# ── the FQE critic ────────────────────────────────────────────────────────────

class Critic(nn.Module):
    def __init__(self, sdim, adim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sdim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, adim))

    def forward(self, x):
        return self.net(x)


def episode_starts(ep_ids):
    """First index of each contiguous episode block."""
    starts = [0]
    for i in range(1, len(ep_ids)):
        if ep_ids[i] != ep_ids[i - 1]:
            starts.append(i)
    return np.asarray(starts, dtype=np.int64)


def fqe(data, pi_next, pi_eval_states, gamma, iters, batch, lr, tgt_every,
        device="cpu", seed=0, log_every=0):
    """Fit Q_psi to the target policy; return (value, history)."""
    torch.manual_seed(seed)
    S  = torch.as_tensor(data["S"], dtype=torch.float32, device=device)
    A  = torch.as_tensor(data["A"], dtype=torch.long, device=device)
    R  = torch.as_tensor(data["R"], dtype=torch.float32, device=device)
    NS = torch.as_tensor(data["NS"], dtype=torch.float32, device=device)
    Dn = torch.as_tensor(data["D"], dtype=torch.float32, device=device)
    PN = torch.as_tensor(pi_next, dtype=torch.long, device=device)

    mu = S.mean(0, keepdim=True)
    sd = S.std(0, keepdim=True).clamp_min(1e-6)
    Sn, NSn = (S - mu) / sd, (NS - mu) / sd

    adim = int(max(int(A.max()), int(PN.max()),
                   int(pi_eval_states["A"].max()))) + 1
    net = Critic(S.shape[1], adim).to(device)
    tgt = Critic(S.shape[1], adim).to(device)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    E0 = torch.as_tensor(pi_eval_states["S"], dtype=torch.float32, device=device)
    E0n = (E0 - mu) / sd
    E0a = torch.as_tensor(pi_eval_states["A"], dtype=torch.long, device=device)

    n = len(S)
    g = torch.Generator(device="cpu").manual_seed(seed)
    hist = []
    for it in range(1, iters + 1):
        idx = torch.randint(0, n, (batch,), generator=g).to(device)
        with torch.no_grad():
            qn = tgt(NSn[idx]).gather(1, PN[idx].unsqueeze(1)).squeeze(1)
            y = R[idx] + gamma * (1.0 - Dn[idx]) * qn
        q = net(Sn[idx]).gather(1, A[idx].unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        if it % tgt_every == 0:
            tgt.load_state_dict(net.state_dict())
        if log_every and (it % log_every == 0 or it == iters):
            with torch.no_grad():
                v = net(E0n).gather(1, E0a.unsqueeze(1)).squeeze(1).mean().item()
            hist.append((it, float(loss.item()), v))

    with torch.no_grad():
        vals = net(E0n).gather(1, E0a.unsqueeze(1)).squeeze(1)
    return float(vals.mean()), float(vals.std() / max(len(vals) ** 0.5, 1)), hist


# ── the IS diagnostic we are replacing (reported, not used) ───────────────────

def snis_ess(logged_a, greedy_a, n_actions=760):
    m = (logged_a == greedy_a)
    k = int(m.sum())
    return k, (k / len(m) if len(m) else 0.0), float(k)   # ESS == match count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="the HOLDOUT .npz -- supplies the s0 set where the policy "
                         "is evaluated. Must be data the policy never trained on.")
    ap.add_argument("--fit-data", default=None,
                    help="optional separate .npz for FITTING the critic (default: "
                         "--data). The critic is a SEPARATE network, so fitting it "
                         "on the training set is sound and uses far more data; what "
                         "must stay uncontaminated is the s0 set, because a policy "
                         "that memorized a state picks its logged action there and "
                         "so is unrepresentative of deployment.")
    ap.add_argument("--policies", nargs="+", required=True, help="checkpoint .npz files")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tgt-every", type=int, default=200)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of TRAJECTORIES held out; their starts are the s0 set")
    ap.add_argument("--seed", type=int, default=0)
    # Drift must be measured over a FIXED iteration interval, otherwise it is not
    # comparable between runs: with log_every = iters//5 a 3k run and a 30k run
    # report drift over 600 and 6000 iterations respectively, and the longer run
    # looks worse purely because its window is wider.
    ap.add_argument("--log-every", type=int, default=2000,
                    help="fixed interval for the convergence trace and drift")
    ap.add_argument("--trace-convergence", action="store_true",
                    help="print V_hat at each interval, to size the iteration budget")
    ap.add_argument("--out", default=None, help="write rankings to this .json")
    a = ap.parse_args()

    d = np.load(a.data, allow_pickle=True)
    S, A, R = d["states"], d["actions"], d["rewards"].astype(np.float32)
    # Exactly as train_offline.py:358-359 does it, over the whole dataset.
    MU = S.mean(axis=0)
    SD = S.std(axis=0) + 1e-8
    NS, D = d["next_states"], d["dones"]
    if "episode_ids" in d.files:
        ep = d["episode_ids"]
    else:                               # derive from terminals: one per episode
        ep = np.concatenate([[0], np.cumsum(D[:-1])]).astype(np.int32)
        print("  note: no episode_ids in dataset; derived from dones")

    uniq = np.unique(ep)
    if a.fit_data:
        # Critic fits on the separate (larger) file; EVERY episode of --data
        # contributes an s0, since none of it was used for fitting.
        f = np.load(a.fit_data, allow_pickle=True)
        tr = dict(S=f["states"], A=f["actions"], R=f["rewards"].astype(np.float32),
                  NS=f["next_states"], D=f["dones"])
        is_val = np.ones(len(S), dtype=bool)
        Sv = S[episode_starts(ep)]
        n_val = len(uniq)
        print(f"  critic fit on {a.fit_data} ({len(tr['S']):,} transitions); "
              f"s0 from ALL {n_val} episodes of the holdout")
    else:
        rng = np.random.default_rng(a.seed)
        perm = rng.permutation(uniq)
        n_val = max(1, int(a.val_frac * len(uniq)))
        val_eps = set(perm[:n_val].tolist())
        is_val = np.isin(ep, list(val_eps))
        # TRAJECTORY-disjoint: no episode contributes to both fit and s0 set.
        tr = dict(S=S[~is_val], A=A[~is_val], R=R[~is_val], NS=NS[~is_val], D=D[~is_val])
        Sv = S[is_val][episode_starts(ep[is_val])]

    print(f"  dataset {a.data}")
    print(f"    {len(S):,} transitions, {len(uniq)} episodes "
          f"(gamma {a.gamma}, effective horizon {1/(1-a.gamma):.0f})")
    _fit_eps = "a separate file" if a.fit_data else f"{len(uniq)-n_val} episodes"
    print(f"    fit on {len(tr['S']):,} transitions from {_fit_eps}; "
          f"{len(Sv)} held-out trajectory starts as s0")
    print()

    rows = []
    for p in a.policies:
        label = os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p))
        q, dueling, pre, trained_on = load_q(p, MU, SD)
        if trained_on and trained_on == os.path.basename(a.data):
            raise SystemExit(
                f"REFUSING: {os.path.basename(p)} was trained on {trained_on}, which is "
                f"the dataset being used to evaluate it. The policy memorizes its "
                f"training actions (83.7% greedy match on train vs 0.0% on held-out "
                f"data), so V_hat would be optimistically biased. Pass the "
                f"*_holdout.npz as --data (the training file may be passed as "
                f"--fit-data, which is sound).")
        pin = greedy_actions(q, tr["NS"])          # pi(s') on the fit split
        a0 = greedy_actions(q, Sv)                 # pi(s0) on held-out starts
        v, se, hist = fqe(tr, pin, dict(S=Sv, A=a0), a.gamma, a.iters,
                          a.batch, a.lr, a.tgt_every, seed=a.seed,
                          log_every=max(a.log_every, 1))
        if a.trace_convergence and hist:
            print(f"    convergence trace ({label}):")
            pv = None
            for it, ls, vv in hist:
                d = "" if pv is None else f"  drift {100*abs(vv-pv)/max(abs(vv),1e-9):5.2f}%"
                print(f"      iter {it:7,d}  loss {ls:9.4f}  V_hat {vv:+8.3f}{d}")
                pv = vv
        # the IS diagnostic, on the held-out episodes, for the record
        gv = greedy_actions(q, S[is_val])
        k, rate, ess = snis_ess(A[is_val], gv)
        conv = [h[2] for h in hist]
        # SHORT-window drift is not a convergence test. Measured curve (40,000
        # iterations, pilot holdout): the 2,000-iteration drift fell below 2% at
        # ~14,000 iterations with V_hat = -13.33, yet V_hat continued to -14.21 by
        # 40,000 -- a further 6.6%. The estimate trends slowly and monotonically,
        # so a narrow window reads as converged while real movement remains.
        # LONG_W intervals (= LONG_W * log_every iterations) is the gate that
        # actually holds; the short figure is kept only for reporting.
        LONG_W = 5
        drift = (abs(conv[-1] - conv[-2]) / max(abs(conv[-1]), 1e-9)) if len(conv) > 1 else float("nan")
        long_drift = (abs(conv[-1] - conv[-1 - LONG_W]) / max(abs(conv[-1]), 1e-9)
                      if len(conv) > LONG_W else float("nan"))
        # `label` keeps the parent directory: basenames alone are ambiguous when a
        # sweep writes identically-named snapshots to different directories.
        rows.append(dict(policy=label, fqe_value=v, fqe_se=se,
                         ofdma_share=float((a0 >= NUM_TWT).mean()),
                         top_action_share=float(np.bincount(a0).max() / len(a0)),
                         is_matches=k, is_ess=ess, final_drift=drift,
                         long_drift=long_drift,
                         long_window_iters=LONG_W * max(a.log_every, 1),
                         dueling=bool(dueling)))
        _ld = rows[-1]["long_drift"]
        print(f"  {label:48s} V_hat {v:+8.3f} +/- {se:.3f}   "
              f"drift {100*drift:4.1f}% (over {LONG_W * max(a.log_every,1):,}: "
              f"{100*_ld:4.1f}%)   OFDMA {100*rows[-1]['ofdma_share']:5.1f}%   "
              f"[IS: {k} matches, ESS {ess:.0f}]")

    rows.sort(key=lambda r: -r["fqe_value"])
    print()
    print("  ranking by FQE value (best first):")
    for i, r in enumerate(rows, 1):
        print(f"    {i}. {r['policy']}  {r['fqe_value']:+.3f}")
    if max(r["is_ess"] for r in rows) < 400:
        print()
        print(f"  IS unusable as expected: best ESS {max(r['is_ess'] for r in rows):.0f} "
              f"< 400 needed. FQE ranking used instead.")
    if a.out:
        import json
        json.dump(rows, open(a.out, "w"), indent=2)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
