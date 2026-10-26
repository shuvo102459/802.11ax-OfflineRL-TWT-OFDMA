#!/usr/bin/env python3
"""
Multi-algorithm offline RL trainer for TWT comparison.

Supports 5 algorithms (all trained on the same offline dataset):
  bc        — Behavior Cloning (supervised, no reward signal)
  dqn       — Vanilla DQN (single-network target, + CQL)
  ddqn      — Double DQN, NO CQL  (cql_alpha forced to 0)
  iql       — Implicit Q-Learning (Kostrikov et al. 2021)
  dueling   — Dueling DQN + CQL   (V+A heads, same CQL as ddqn_cql)

Usage:
  python3 train-dqn-offline-v2.py --algo bc     --data rl-data/rl_dataset_3k.npz
  python3 train-dqn-offline-v2.py --algo dqn    --data rl-data/rl_dataset_3k.npz
  python3 train-dqn-offline-v2.py --algo ddqn   --data rl-data/rl_dataset_3k.npz
  python3 train-dqn-offline-v2.py --algo iql    --data rl-data/rl_dataset_3k.npz
  python3 train-dqn-offline-v2.py --algo dueling --data rl-data/rl_dataset_3k.npz

Output: rl-data/model_<algo>.npz  (loaded by export_dqn_v2_header.py)
"""

import argparse
import os
import sys
import time
import numpy as np
import random


# ── Shared helpers ─────────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)


def huber_grad(err, delta=1.0):
    return err if abs(err) <= delta else np.sign(err)


def huber_loss(err, delta=1.0):
    return err**2 if abs(err) <= delta else abs(err)


# ── Standard 3-layer Q-network (BC / DQN / DDQN / IQL) ───────────────────────

class SimpleQNetwork:
    """130 → 128 → 128 → action_dim  (used for Q and V networks)."""

    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.lr         = lr
        scale = 0.5
        self.W1 = np.random.randn(state_dim,  hidden_dim) * np.sqrt(2/state_dim)  * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2/hidden_dim) * scale
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, action_dim) * np.sqrt(2/hidden_dim) * scale
        self.b3 = np.zeros(action_dim)
        self._cache = None
        # Adam state
        self.t = 0
        self.m = {k: 0 for k in ["W1","b1","W2","b2","W3","b3"]}
        self.v = {k: 0 for k in ["W1","b1","W2","b2","W3","b3"]}

    def forward(self, x):
        z1 = x @ self.W1 + self.b1;  a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2; a2 = relu(z2)
        out = a2 @ self.W3 + self.b3
        self._cache = (x, z1, a1, z2, a2)
        return out

    def backward(self, dL_dout, clip_grad=1.0):
        x, z1, a1, z2, a2 = self._cache
        bs = x.shape[0]
        dL_da2 = dL_dout @ self.W3.T
        dL_dW3 = a2.T @ dL_dout / bs
        dL_db3 = dL_dout.mean(axis=0)
        dL_dz2 = dL_da2 * (z2 > 0)
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dW2 = a1.T @ dL_dz2 / bs
        dL_db2 = dL_dz2.mean(axis=0)
        dL_dz1 = dL_da1 * (z1 > 0)
        dL_dW1 = x.T @ dL_dz1 / bs
        dL_db1 = dL_dz1.mean(axis=0)
        grads = {"W1":dL_dW1,"b1":dL_db1,"W2":dL_dW2,"b2":dL_db2,"W3":dL_dW3,"b3":dL_db3}
        norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        if norm > clip_grad:
            s = clip_grad / (norm + 1e-8)
            grads = {k: v*s for k, v in grads.items()}
        self._adam_update(grads)

    def _adam_update(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for name in grads:
            g = grads[name]
            self.m[name] = beta1 * self.m[name] + (1-beta1) * g
            self.v[name] = beta2 * self.v[name] + (1-beta2) * g**2
            mh = self.m[name] / (1 - beta1**self.t)
            vh = self.v[name] / (1 - beta2**self.t)
            setattr(self, name, getattr(self, name) - self.lr * mh / (np.sqrt(vh) + eps))

    def get_state(self):
        """Return all weights + Adam optimizer state as a flat dict of numpy arrays."""
        keys = ["W1","b1","W2","b2","W3","b3"]
        d = {k: getattr(self, k).copy() for k in keys}
        d["adam_t"] = np.array([self.t], dtype=np.int64)
        for k in keys:
            mv = self.m[k]
            vv = self.v[k]
            d[f"m_{k}"] = np.atleast_1d(np.array(mv, dtype=np.float32))
            d[f"v_{k}"] = np.atleast_1d(np.array(vv, dtype=np.float32))
        return d

    def set_state(self, d, prefix=""):
        """Restore weights + Adam optimizer state from a dict (prefix distinguishes networks)."""
        keys = ["W1","b1","W2","b2","W3","b3"]
        for k in keys:
            setattr(self, k, d[f"{prefix}{k}"].copy())
        self.t = int(d[f"{prefix}adam_t"][0])
        for k in keys:
            self.m[k] = d[f"{prefix}m_{k}"].copy().reshape(getattr(self, k).shape) if d[f"{prefix}m_{k}"].size > 1 else float(d[f"{prefix}m_{k}"][0])
            self.v[k] = d[f"{prefix}v_{k}"].copy().reshape(getattr(self, k).shape) if d[f"{prefix}v_{k}"].size > 1 else float(d[f"{prefix}v_{k}"][0])

    def copy_from(self, other):
        for a in ["W1","b1","W2","b2","W3","b3"]:
            setattr(self, a, getattr(other, a).copy())

    def soft_update_from(self, other, tau=0.005):
        for a in ["W1","b1","W2","b2","W3","b3"]:
            setattr(self, a, tau*getattr(other,a) + (1-tau)*getattr(self,a))

    def save(self, path, state_mean=None, state_std=None, algo="unknown",
             train_data=None):
        d = dict(W1=self.W1,b1=self.b1,W2=self.W2,b2=self.b2,
                 W3=self.W3,b3=self.b3,
                 state_dim=self.state_dim, action_dim=self.action_dim,
                 algo=algo, is_dueling=False)
        if state_mean is not None:
            d["state_mean"] = state_mean
            d["state_std"]  = state_std
        # Recorded so fqe.py can refuse to evaluate a policy on the data it was
        # trained on; the policy memorizes its training actions (83.7% greedy
        # match on train vs 0.0% held out), which biases V_hat upward.
        if train_data is not None:
            d["train_data"] = np.array(str(train_data))
        np.savez(path, **d)


# ── Dueling Q-network (Dueling DQN + CQL) ─────────────────────────────────────

class DuelingQNetwork:
    """
    130 → 128 → 128 → { V-head: 1 } + { A-head: 380 }
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.lr         = lr
        scale = 0.5
        self.W1  = np.random.randn(state_dim,  hidden_dim) * np.sqrt(2/state_dim)  * scale
        self.b1  = np.zeros(hidden_dim)
        self.W2  = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2/hidden_dim) * scale
        self.b2  = np.zeros(hidden_dim)
        self.W_V = np.random.randn(hidden_dim, 1)          * np.sqrt(2/hidden_dim) * scale
        self.b_V = np.zeros(1)
        self.W_A = np.random.randn(hidden_dim, action_dim) * np.sqrt(2/hidden_dim) * scale
        self.b_A = np.zeros(action_dim)
        self._cache = None
        self.t = 0
        keys = ["W1","b1","W2","b2","W_V","b_V","W_A","b_A"]
        self.m = {k: 0 for k in keys}
        self.v = {k: 0 for k in keys}

    def forward(self, x):
        z1 = x @ self.W1 + self.b1;   a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2;  a2 = relu(z2)
        V   = (a2 @ self.W_V + self.b_V)          # (batch, 1)
        A   = a2 @ self.W_A + self.b_A            # (batch, action_dim)
        Q   = V + A - A.mean(axis=1, keepdims=True)
        self._cache = (x, z1, a1, z2, a2, V, A)
        return Q

    def backward(self, dL_dQ, clip_grad=1.0):
        x, z1, a1, z2, a2, V, A = self._cache
        bs = x.shape[0]
        # dQ/dA: identity minus 1/action_dim (for the mean subtraction)
        dL_dA = dL_dQ - dL_dQ.mean(axis=1, keepdims=True)
        # dQ/dV: sum over actions
        dL_dV = dL_dQ.sum(axis=1, keepdims=True)
        dL_dW_A = a2.T @ dL_dA / bs
        dL_db_A = dL_dA.mean(axis=0)
        dL_dW_V = a2.T @ dL_dV / bs
        dL_db_V = dL_dV.mean(axis=0).squeeze()
        dL_da2  = dL_dA @ self.W_A.T + dL_dV @ self.W_V.T
        dL_dz2  = dL_da2 * (z2 > 0)
        dL_da1  = dL_dz2 @ self.W2.T
        dL_dW2  = a1.T @ dL_dz2 / bs
        dL_db2  = dL_dz2.mean(axis=0)
        dL_dz1  = dL_da1 * (z1 > 0)
        dL_dW1  = x.T @ dL_dz1 / bs
        dL_db1  = dL_dz1.mean(axis=0)
        grads = {"W1":dL_dW1,"b1":dL_db1,"W2":dL_dW2,"b2":dL_db2,
                 "W_V":dL_dW_V,"b_V":np.atleast_1d(dL_db_V),
                 "W_A":dL_dW_A,"b_A":dL_db_A}
        norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        if norm > clip_grad:
            s = clip_grad / (norm + 1e-8)
            grads = {k: v*s for k, v in grads.items()}
        self._adam_update(grads)

    def _adam_update(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for name in grads:
            g = grads[name]
            self.m[name] = beta1 * self.m[name] + (1-beta1) * g
            self.v[name] = beta2 * self.v[name] + (1-beta2) * g**2
            mh = self.m[name] / (1 - beta1**self.t)
            vh = self.v[name] / (1 - beta2**self.t)
            setattr(self, name, getattr(self, name) - self.lr * mh / (np.sqrt(vh) + eps))

    def get_state(self):
        """Return all weights + Adam optimizer state as a flat dict of numpy arrays."""
        keys = ["W1","b1","W2","b2","W_V","b_V","W_A","b_A"]
        d = {k: getattr(self, k).copy() for k in keys}
        d["adam_t"] = np.array([self.t], dtype=np.int64)
        for k in keys:
            mv = self.m[k]
            vv = self.v[k]
            d[f"m_{k}"] = np.atleast_1d(np.array(mv, dtype=np.float32))
            d[f"v_{k}"] = np.atleast_1d(np.array(vv, dtype=np.float32))
        return d

    def set_state(self, d, prefix=""):
        """Restore weights + Adam optimizer state from a dict."""
        keys = ["W1","b1","W2","b2","W_V","b_V","W_A","b_A"]
        for k in keys:
            setattr(self, k, d[f"{prefix}{k}"].copy())
        self.t = int(d[f"{prefix}adam_t"][0])
        for k in keys:
            self.m[k] = d[f"{prefix}m_{k}"].copy().reshape(getattr(self, k).shape) if d[f"{prefix}m_{k}"].size > 1 else float(d[f"{prefix}m_{k}"][0])
            self.v[k] = d[f"{prefix}v_{k}"].copy().reshape(getattr(self, k).shape) if d[f"{prefix}v_{k}"].size > 1 else float(d[f"{prefix}v_{k}"][0])

    def copy_from(self, other):
        for a in ["W1","b1","W2","b2","W_V","b_V","W_A","b_A"]:
            setattr(self, a, getattr(other, a).copy())

    def soft_update_from(self, other, tau=0.005):
        for a in ["W1","b1","W2","b2","W_V","b_V","W_A","b_A"]:
            setattr(self, a, tau*getattr(other,a) + (1-tau)*getattr(self,a))

    def save(self, path, state_mean=None, state_std=None, algo="dueling",
             train_data=None):
        d = dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W_V=self.W_V, b_V=self.b_V, W_A=self.W_A, b_A=self.b_A,
                 state_dim=self.state_dim, action_dim=self.action_dim,
                 algo=algo, is_dueling=True)
        if state_mean is not None:
            d["state_mean"] = state_mean
            d["state_std"]  = state_std
        if train_data is not None:
            d["train_data"] = np.array(str(train_data))
        np.savez(path, **d)


# ── Trainer ────────────────────────────────────────────────────────────────────

class OfflineDQNV2Trainer:

    # Reward preset weight vectors (w_sum, w_fairness, w_network)
    REWARD_PRESETS = {
        "balanced":   (0.40, 0.25, 0.35),
        "energy":     (0.25, 0.15, 0.60),
        "throughput": (0.55, 0.20, 0.25),
    }

    # Per-epoch curve columns (single source of truth for checkpoint/curve I/O).
    #   val_reward_mean      : off-policy estimate of the learned policy's expected
    #                          immediate reward on held-out transitions (raw units)
    #   val_reward_ess       : Kish effective-sample-size fraction of that estimate
    #   val_reward_data_mean : constant behaviour-policy reward baseline (raw units)
    CURVE_KEYS = ["epoch", "loss", "q_mean", "q_std", "val_q_mean",
                  "val_reward_mean", "val_reward_ess", "val_reward_data_mean"]

    def __init__(self, data_file, algo="ddqn_cql", hidden_dim=128, lr=0.001,
                 gamma=0.95, batch_size=512, cql_alpha=0.05, tau=0.005, target_clip=50.0,
                 iql_tau=0.7, preset="balanced", val_split=0.1,
                 reward_weights=None, seed=42):
        self.algo       = algo
        self.gamma      = gamma
        # Needed by _ckpt_path to key the checkpoint on the run configuration.
        self.lr         = lr
        self.data_file  = data_file
        self.seed       = seed
        self.hidden_dim = hidden_dim
        self.val_split  = val_split
        # RL-room's copy hardcodes +/-10. With gamma=0.95 that caps |V| at 10, but
        # the balanced reward implies |V| ~ 33 and the constrained one ~20, so every
        # target saturated and all 760 actions received the SAME target -- a Q with
        # no action information, whose argmax is an arbitrary near-constant policy.
        # That is a candidate explanation for the 91.4% top-action share reported in
        # PILOT_FINDINGS.md. Parameterised here; build_dataset.py gates against it.
        self.target_clip = float(target_clip)
        self.batch_size = batch_size
        self.cql_alpha  = 0.0 if algo == "ddqn" else cql_alpha
        self.tau        = tau
        self.iql_tau    = iql_tau
        self.preset     = preset
        # explicit (w_sum, w_fair, w_net); overrides `preset` when set (stage 5
        # reward-weight sensitivity sweep). Not required to sum to 1.
        self.reward_weights = tuple(reward_weights) if reward_weights else None

        # ── load data ──────────────────────────────────────────────────────
        data = np.load(data_file)
        self.reward_kind = str(data["reward_kind"]) if "reward_kind" in data else "unknown"
        states      = data["states"].astype(np.float32)
        actions     = data["actions"].astype(int)
        next_states = data["next_states"].astype(np.float32)
        dones       = data["dones"].astype(np.float32)

        # Build reward from preset weights if reward components are stored
        if "r_sum" in data and "r_fairness" in data and "r_network" in data:
            if self.reward_weights is not None:
                w_sum, w_fair, w_net = self.reward_weights
            else:
                w_sum, w_fair, w_net = self.REWARD_PRESETS.get(preset, self.REWARD_PRESETS["balanced"])
            rewards = (w_sum  * data["r_sum"].astype(np.float32)
                     + w_fair * data["r_fairness"].astype(np.float32)
                     + w_net  * data["r_network"].astype(np.float32))
            print(f"Preset '{preset}': w_sum={w_sum}, w_fair={w_fair}, w_net={w_net}")
        else:
            rewards = data["rewards"].astype(np.float32)
            print(f"No reward components in dataset — using stored scalar rewards (preset ignored)")

        # ── train / val split ──────────────────────────────────────────────
        n_total = len(states)
        n_val   = max(1, int(n_total * val_split))
        idx     = np.random.permutation(n_total)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        self.val_states  = states[val_idx]
        self.val_actions = actions[val_idx]
        self.val_rewards = rewards[val_idx]

        self.states      = states[train_idx]
        self.actions     = actions[train_idx]
        self.rewards     = rewards[train_idx]
        self.next_states = next_states[train_idx]
        self.dones       = dones[train_idx]
        self.num_samples = len(self.states)
        self.state_dim   = self.states.shape[1]
        # Data-driven, not hardcoded: 380 for the old (sched,assign)-only action
        # space, 760 for the 2026-09-16 extension that adds a per-step OFDMA
        # on/off bit (ofdma*380 + sched*19 + assign). Floor at 380 so a tiny/
        # degenerate dataset that happens not to sample the top action index
        # still gets a correctly-sized (not truncated) output layer.
        self.num_actions = max(380, int(self.actions.max()) + 1)

        # normalise rewards (fit on train, apply to both)
        self.reward_mean = self.rewards.mean()
        self.reward_std  = self.rewards.std() + 1e-8
        self.rewards      = (self.rewards     - self.reward_mean) / self.reward_std
        self.val_rewards  = (self.val_rewards - self.reward_mean) / self.reward_std
        print(f"Reward: mean={self.reward_mean:.4f}  std={self.reward_std:.4f}")

        # Raw-unit validation rewards + the constant behaviour-policy baseline
        # (mean reward the uniform-random exploration policy earned on the val set).
        self._val_rewards_raw       = self.val_rewards * self.reward_std + self.reward_mean
        self._val_reward_data_mean  = float(self._val_rewards_raw.mean())

        # normalise states (fit on train, apply to both)
        self.state_mean  = self.states.mean(axis=0)
        self.state_std   = self.states.std(axis=0) + 1e-8
        self.states      = (self.states      - self.state_mean) / self.state_std
        self.next_states = (self.next_states - self.state_mean) / self.state_std
        self.val_states  = (self.val_states  - self.state_mean) / self.state_std

        print(f"Loaded {n_total} transitions | train={self.num_samples} val={n_val} "
              f"| state_dim={self.state_dim} | algo={algo} | preset={preset}")

        # ── networks ───────────────────────────────────────────────────────
        if algo == "dueling":
            self.q_net    = DuelingQNetwork(self.state_dim, self.num_actions, hidden_dim, lr)
            self.q_target = DuelingQNetwork(self.state_dim, self.num_actions, hidden_dim, lr)
        else:
            self.q_net    = SimpleQNetwork(self.state_dim, self.num_actions, hidden_dim, lr)
            self.q_target = SimpleQNetwork(self.state_dim, self.num_actions, hidden_dim, lr)
        self.q_target.copy_from(self.q_net)

        # IQL needs an additional V-network (output dim = 1)
        if algo == "iql":
            self.v_net    = SimpleQNetwork(self.state_dim, 1, hidden_dim, lr)
            self.v_target = SimpleQNetwork(self.state_dim, 1, hidden_dim, lr)
            self.v_target.copy_from(self.v_net)

    # ── checkpoint helpers ─────────────────────────────────────────────────────

    def _run_tag(self):
        """Short tag identifying this RUN, for milestone filenames.

        Milestones were named `model_{algo}_{preset}_ep{N}.npz`. A constrained
        run reports preset "balanced" (preset is ignored when the dataset stores
        scalar rewards), so the two reward arms of one algorithm wrote the SAME
        filenames and silently overwrote each other's snapshots -- the same trap
        as the checkpoint key, on the files FQE later ranks.
        """
        import hashlib
        # EVERY hyperparameter that changes the trained model must appear here.
        # hidden_dim was missing on the first attempt, so a capacity sweep produced
        # three snapshots with IDENTICAL filenames -- they survived only because
        # each went to its own directory. In a shared milestone dir they would have
        # overwritten each other, which is the same collision this tag exists to
        # prevent. iql_tau and val_split are included for the same reason.
        sig = "|".join(str(x) for x in [
            self.algo, self.preset, getattr(self, "reward_kind", "?"),
            self.cql_alpha, self.target_clip, self.gamma, self.lr,
            self.batch_size, getattr(self, "seed", "?"),
            getattr(self, "hidden_dim", "?"), getattr(self, "iql_tau", "?"),
            getattr(self, "val_split", "?"),
            os.path.basename(getattr(self, "data_file", "?"))])
        return hashlib.sha1(sig.encode()).hexdigest()[:10]

    def _ckpt_path(self, ckpt_dir):
        """Checkpoint file keyed on the whole RUN CONFIGURATION.

        RL-room's copy named it `ckpt_{algo}_{preset}.npz` in a shared default
        directory, so runs differing only in cql_alpha, reward kind, target_clip,
        lr, gamma or seed all collided on ONE file and silently resumed from each
        other. Observed directly: two runs at cql_alpha 0.05 and 0.5 resumed from
        a cql_alpha 0.0 run's epoch 82 and early-stopped after a single epoch,
        reporting its best_loss as their own. A sweep over methods x rewards x
        lambda would have cross-contaminated every arm.

        `preset` is also the wrong key on its own: a constrained-reward run
        reports preset "balanced" (the preset is ignored when the dataset stores
        scalar rewards), so the two reward arms of one algorithm collided too.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        return os.path.join(
            ckpt_dir, f"ckpt_{self.algo}_{self.preset}_{self._run_tag()}.npz")

    def save_checkpoint(self, ckpt_dir, epoch):
        """Atomically save full trainer state (weights + Adam + curve) to checkpoint."""
        path = self._ckpt_path(ckpt_dir)
        # np.savez force-appends ".npz"; keep that suffix on the temp name so the
        # written file and the rename source match.
        tmp  = path[:-4] + ".tmp.npz"
        d = {"ckpt_epoch": np.array([epoch], dtype=np.int32),
             "algo":       np.array([self.algo]),
             "preset":     np.array([self.preset])}
        # Q-network weights + Adam
        for k, v in self.q_net.get_state().items():
            d[f"qnet_{k}"] = v
        for k, v in self.q_target.get_state().items():
            d[f"qtgt_{k}"] = v
        # V-network for IQL
        if self.algo == "iql":
            for k, v in self.v_net.get_state().items():
                d[f"vnet_{k}"] = v
            for k, v in self.v_target.get_state().items():
                d[f"vtgt_{k}"] = v
        # Training curve so far
        for key in self.CURVE_KEYS:
            d[f"curve_{key}"] = np.array(self.curve.get(key, []), dtype=np.float32)
        np.savez(tmp, **d)
        os.replace(tmp, path)   # atomic rename — safe against power-off mid-write

    def load_checkpoint(self, ckpt_dir):
        """Load checkpoint if it exists and matches algo+preset. Returns start_epoch or 0."""
        path = self._ckpt_path(ckpt_dir)
        if not os.path.exists(path):
            return 0
        try:
            d = np.load(path, allow_pickle=True)
            saved_algo   = str(d["algo"][0])
            saved_preset = str(d["preset"][0])
            if saved_algo != self.algo or saved_preset != self.preset:
                print(f"[ckpt] Ignoring checkpoint: algo/preset mismatch "
                      f"({saved_algo}/{saved_preset} vs {self.algo}/{self.preset})")
                return 0
            start_epoch = int(d["ckpt_epoch"][0]) + 1
            # Restore Q-network
            q_d   = {k[len("qnet_"):]: d[k] for k in d if k.startswith("qnet_")}
            qt_d  = {k[len("qtgt_"):]: d[k] for k in d if k.startswith("qtgt_")}
            self.q_net.set_state(q_d)
            self.q_target.set_state(qt_d)
            # Restore V-network for IQL
            if self.algo == "iql":
                v_d  = {k[len("vnet_"):]: d[k] for k in d if k.startswith("vnet_")}
                vt_d = {k[len("vtgt_"):]: d[k] for k in d if k.startswith("vtgt_")}
                self.v_net.set_state(v_d)
                self.v_target.set_state(vt_d)
            # Restore curve (tolerate checkpoints written before a column existed)
            for key in self.CURVE_KEYS:
                ck = f"curve_{key}"
                self.curve[key] = d[ck].tolist() if ck in d.files else []
            n = len(self.curve["epoch"])
            for key in self.CURVE_KEYS:
                if len(self.curve[key]) < n:
                    self.curve[key] += [float("nan")] * (n - len(self.curve[key]))
            print(f"[ckpt] Resumed {self.algo}/{self.preset} from epoch {start_epoch}")
            return start_epoch
        except Exception as e:
            print(f"[ckpt] Failed to load checkpoint ({e}) — starting fresh")
            return 0

    # ── batch sampler ──────────────────────────────────────────────────────────

    def _batch(self):
        idx = np.random.randint(0, self.num_samples, self.batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    # ── CQL gradient helper (shared by dqn, ddqn, dueling) ────────────────────

    def _cql_grad(self, q_values, batch_actions):
        """CQL gradient to add to dL_dq, plus the scalar CQL loss.

        TWO BUGS FIXED relative to RL-room's copy, both silent:

        1. q_taken indexed `self.actions[: self.batch_size]` -- a FIXED PREFIX of
           the dataset, not the actions of the randomly sampled batch. Every
           sampled state was paired with an unrelated transition's action, so the
           reported cql_loss was meaningless. `batch_actions` is now required.

        2. The gradient omitted the `-1[a == taken]` term:

               d/dQ_a [ alpha*(logsumexp(Q) - Q_taken) ]
                   = alpha*( softmax_a - 1[a == taken] )

           Without it the penalty only ever pushed Q in one direction and never
           compensated at the action the data actually took -- a uniform
           Q-shrinkage term, not Conservative Q-Learning. No value of alpha
           recovers the correct gradient, so this was never a tuning problem.
           Holding Q up on in-distribution actions while suppressing
           out-of-distribution ones is precisely the mechanism that bounds the
           max-operator overestimation over 760 actions.
        """
        if self.cql_alpha <= 0:
            return np.zeros_like(q_values), 0.0
        B         = q_values.shape[0]
        q_max     = q_values.max(axis=1, keepdims=True)
        exp_q     = np.exp(q_values - q_max)
        softmax   = exp_q / exp_q.sum(axis=1, keepdims=True)
        logsumexp = q_max.squeeze(axis=1) + np.log(exp_q.sum(axis=1))
        rows      = np.arange(B)
        q_taken   = q_values[rows, batch_actions]
        cql_loss  = self.cql_alpha * float((logsumexp - q_taken).mean())
        cql_grad  = self.cql_alpha * softmax / B
        cql_grad[rows, batch_actions] -= self.cql_alpha / B
        return cql_grad, cql_loss

    # ── per-algorithm train steps ──────────────────────────────────────────────

    def _step_bc(self):
        """Behavior Cloning: cross-entropy(softmax(Q), one-hot(action))."""
        states, actions, _, _, _ = self._batch()
        q = self.q_net.forward(states)

        # Softmax
        q_shifted = q - q.max(axis=1, keepdims=True)
        exp_q     = np.exp(q_shifted)
        softmax   = exp_q / exp_q.sum(axis=1, keepdims=True)

        # Cross-entropy gradient: softmax - one_hot
        dL_dq = softmax.copy()
        for i in range(self.batch_size):
            dL_dq[i, actions[i]] -= 1.0
        dL_dq /= self.batch_size

        loss = -np.log(softmax[np.arange(self.batch_size), actions] + 1e-8).mean()
        self.q_net.backward(dL_dq)
        return float(loss)

    def _step_dqn(self):
        """Vanilla DQN: target uses max over target network directly (no double)."""
        states, actions, rewards, next_states, dones = self._batch()

        # Single-network target: max_a' Q_target(s', a')
        next_q  = self.q_target.forward(next_states)
        max_nq  = next_q.max(axis=1)
        targets = np.clip(rewards + self.gamma * max_nq * (1-dones), -self.target_clip, self.target_clip)

        q       = self.q_net.forward(states)
        dL_dq   = np.zeros_like(q)
        loss    = 0.0
        for i in range(self.batch_size):
            e = q[i, actions[i]] - targets[i]
            dL_dq[i, actions[i]] = huber_grad(e)
            loss += huber_loss(e)
        loss /= self.batch_size

        cql_g, cql_l = self._cql_grad(q, actions)
        self.q_net.backward(dL_dq / self.batch_size + cql_g)
        self.q_target.soft_update_from(self.q_net, self.tau)
        return loss + cql_l

    def _step_ddqn(self):
        """Double DQN (with or without CQL, controlled by self.cql_alpha)."""
        states, actions, rewards, next_states, dones = self._batch()

        # Double DQN: online selects, target evaluates
        nq_online  = self.q_net.forward(next_states)
        best_next  = np.argmax(nq_online, axis=1)
        nq_target  = self.q_target.forward(next_states)
        max_nq     = nq_target[np.arange(self.batch_size), best_next]
        targets    = np.clip(rewards + self.gamma * max_nq * (1-dones), -self.target_clip, self.target_clip)

        q      = self.q_net.forward(states)
        dL_dq  = np.zeros_like(q)
        loss   = 0.0
        for i in range(self.batch_size):
            e = q[i, actions[i]] - targets[i]
            dL_dq[i, actions[i]] = huber_grad(e)
            loss += huber_loss(e)
        loss /= self.batch_size

        cql_g, cql_l = self._cql_grad(q, actions)
        self.q_net.backward(dL_dq / self.batch_size + cql_g)
        self.q_target.soft_update_from(self.q_net, self.tau)
        return loss + cql_l

    def _step_iql(self):
        """
        Implicit Q-Learning (Kostrikov et al. 2021).

        V-update: expectile regression  L_τ(V(s), Q(s,a))
        Q-update: TD with V(s') target — never takes max over actions (no OOD risk)
        """
        states, actions, rewards, next_states, dones = self._batch()

        # ── Q-network update ───────────────────────────────────────────────
        v_next  = self.v_target.forward(next_states).squeeze()  # (batch,)
        targets = np.clip(rewards + self.gamma * v_next * (1-dones), -self.target_clip, self.target_clip)

        q      = self.q_net.forward(states)
        dL_dq  = np.zeros_like(q)
        q_loss = 0.0
        for i in range(self.batch_size):
            e = q[i, actions[i]] - targets[i]
            dL_dq[i, actions[i]] = huber_grad(e)
            q_loss += huber_loss(e)
        q_loss /= self.batch_size
        self.q_net.backward(dL_dq / self.batch_size)

        # ── V-network update: expectile regression ─────────────────────────
        q_sa   = q[np.arange(self.batch_size), actions]  # detached from q graph
        v_pred = self.v_net.forward(states).squeeze()     # (batch,)
        diff   = q_sa - v_pred                            # positive = underestimation

        # Asymmetric L2: weight τ when under, (1-τ) when over
        tau    = self.iql_tau
        weight = np.where(diff > 0, tau, 1 - tau)
        v_loss = (weight * diff**2).mean()

        # Gradient of expectile loss w.r.t. v_pred
        dL_dv  = (-2 * weight * diff / self.batch_size).reshape(-1, 1)
        self.v_net.backward(dL_dv)

        # Soft-update both target networks
        self.v_target.soft_update_from(self.v_net, self.tau)
        self.q_target.soft_update_from(self.q_net, self.tau)

        return q_loss + v_loss

    def _step_dueling(self):
        """Dueling DQN architecture + Double DQN training + CQL."""
        states, actions, rewards, next_states, dones = self._batch()

        nq_online = self.q_net.forward(next_states)
        best_next = np.argmax(nq_online, axis=1)
        nq_target = self.q_target.forward(next_states)
        max_nq    = nq_target[np.arange(self.batch_size), best_next]
        targets   = np.clip(rewards + self.gamma * max_nq * (1-dones), -self.target_clip, self.target_clip)

        q      = self.q_net.forward(states)
        dL_dq  = np.zeros_like(q)
        loss   = 0.0
        for i in range(self.batch_size):
            e = q[i, actions[i]] - targets[i]
            dL_dq[i, actions[i]] = huber_grad(e)
            loss += huber_loss(e)
        loss /= self.batch_size

        cql_g, cql_l = self._cql_grad(q, actions)
        self.q_net.backward(dL_dq / self.batch_size + cql_g)
        self.q_target.soft_update_from(self.q_net, self.tau)
        return loss + cql_l

    # ── Off-policy validation-reward estimate ─────────────────────────────────
    def _ope_val_reward(self, eps=0.05):
        """
        Off-policy estimate of the learned policy's expected *immediate* reward on
        the held-out validation transitions — a per-epoch signal of policy quality
        that (unlike the training loss) keeps moving as the policy improves.

        The exploration dataset was collected with a uniform-random behaviour
        policy (epsilon = 1.0 over all A = 380 actions), so mu(a|s) = 1/A exactly
        and per-decision importance sampling carries no behaviour-model error.

        Target policy: the deployed policy, i.e. epsilon-greedy on the current
        Q-network with a small eps (0.05) for weight stability. Estimate:
        self-normalised 1-step importance sampling,
            V_hat = sum_i w_i r_i / sum_i w_i ,  w_i = pi(a_i|s_i) / mu(a_i|s_i),
        with mu = 1/A so w_i = A * pi(a_i|s_i).

        Returns (v_raw, ess_frac, greedy_match):
          v_raw        estimate of E_pi[r] in raw reward units
          ess_frac     Kish effective sample size / N. THE COMMENT THAT USED TO SIT
                       HERE -- "1.0 = ideal, ->0 = noisy" -- IS INVERTED FOR THIS
                       ESTIMATOR. The target policy is deterministic greedy and the
                       behaviour policy is uniform over 760 actions, so pi_a is
                       (1-eps) on matches and eps/(A-1) elsewhere. When NOTHING
                       matches, every weight is the same small constant and

                           ESS = (sum w)^2 / (N * sum w^2) = 1.0  exactly

                       so ess_frac reports 1.00 precisely when the estimate has
                       degenerated to the unweighted data mean -- i.e. the
                       BEHAVIOUR policy's reward, carrying zero information about
                       the learned policy. Verified: v_raw then equals the base
                       exactly. At eps=0 the same situation reports 0.00 instead;
                       the readout flips between extremes on eps alone while the
                       estimator is uninformative either way.

                       USE greedy_match, NOT ess_frac. Checkpoint selection uses
                       code/fqe.py, which needs no importance weights.
          greedy_match fraction of val states where argmax_a Q == logged action
        """
        A      = self.num_actions
        q      = self.q_net.forward(self.val_states)                  # (N, A)
        rows   = np.arange(len(self.val_actions))
        greedy = q.argmax(axis=1)
        match  = greedy == self.val_actions

        # pi(a_i|s_i): (1-eps) on the greedy action, eps/(A-1) elsewhere
        pi_a = np.where(match, 1.0 - eps, eps / (A - 1))
        w    = A * pi_a                                               # rho_i, mu = 1/A
        w_sum = w.sum() + 1e-12

        v_raw        = float((w * self._val_rewards_raw).sum() / w_sum)
        ess_frac     = float(w_sum ** 2 / (np.sum(w ** 2) + 1e-12) / len(w))
        greedy_match = float(match.mean())
        # Make the degenerate case impossible to misread in a log or a curve.
        if match.sum() == 0:
            ess_frac = 0.0
        return v_raw, ess_frac, greedy_match

    # ── Main train loop ────────────────────────────────────────────────────────

    def train(self, num_epochs=300, log_freq=10, ckpt_dir="rl-data/checkpoints",
              ckpt_every=10, milestone_dir=None, milestone_every=25, patience_limit=60):
        step_fn = {
            "bc":      self._step_bc,
            "dqn":     self._step_dqn,
            "ddqn":    self._step_ddqn,
            "iql":     self._step_iql,
            "dueling": self._step_dueling,
        }[self.algo]

        steps_per_epoch = self.num_samples // self.batch_size
        print(f"\nTraining [{self.algo.upper()}] — {num_epochs} epochs × {steps_per_epoch} steps")
        print("=" * 60)

        t0 = time.time()
        best_loss = float("inf")
        patience  = 0

        # Training curve accumulators — pre-seeded from checkpoint if resuming
        if not hasattr(self, "curve"):
            self.curve = {k: [] for k in self.CURVE_KEYS}

        # Resume from checkpoint if available
        start_epoch = self.load_checkpoint(ckpt_dir)
        if start_epoch > 0 and self.curve["loss"]:
            best_loss = min(self.curve["loss"])
            # Restore patience counter from how many trailing epochs didn't improve
            for l in reversed(self.curve["loss"]):
                if l <= best_loss:
                    break
                patience += 1

        for epoch in range(start_epoch, num_epochs):
            epoch_losses = [step_fn() for _ in range(steps_per_epoch)]
            avg = float(np.mean(epoch_losses))

            q_sample = self.q_net.forward(self.states[:500])

            # Validation: mean Q at dataset actions, + off-policy reward estimate
            val_q_all  = self.q_net.forward(self.val_states)
            val_q_taken = val_q_all[np.arange(len(self.val_actions)), self.val_actions]
            val_q_mean  = float(val_q_taken.mean())
            val_r_ope, val_r_ess, val_greedy = self._ope_val_reward()

            self.curve["epoch"].append(epoch + 1)
            self.curve["loss"].append(avg)
            self.curve["q_mean"].append(float(q_sample.mean()))
            self.curve["q_std"].append(float(q_sample.std()))
            self.curve["val_q_mean"].append(val_q_mean)
            self.curve["val_reward_mean"].append(val_r_ope)
            self.curve["val_reward_ess"].append(val_r_ess)
            self.curve["val_reward_data_mean"].append(self._val_reward_data_mean)

            if (epoch + 1) % log_freq == 0:
                print(f"Epoch {epoch+1:4d}/{num_epochs} | loss={avg:.4f} | "
                      f"Q: {q_sample.mean():.2f}±{q_sample.std():.2f} | "
                      f"val_Q={val_q_mean:.3f} | val_R={val_r_ope:.3f} "
                      f"(base={self._val_reward_data_mean:.3f}, ess={val_r_ess:.2f}, "
                      f"match={val_greedy:.2f}) | t={time.time()-t0:.0f}s")
            else:
                print(f"Epoch {epoch+1:4d}/{num_epochs} | loss={avg:.4f} | "
                      f"val_Q={val_q_mean:.3f} | val_R={val_r_ope:.3f}")

            if avg < best_loss:
                best_loss = avg
                patience  = 0
            else:
                patience += 1

            # Save checkpoint every ckpt_every epochs (atomic write)
            if (epoch + 1) % ckpt_every == 0:
                self.save_checkpoint(ckpt_dir, epoch)

            # Milestone snapshot: exportable model weights at fixed epoch intervals,
            # kept under distinct per-epoch filenames (unlike the resume checkpoint,
            # which is overwritten in place). Used for real-simulator checkpoint
            # evaluation (see eval_checkpoints.sh) — the resume checkpoint alone
            # cannot answer "how good was the policy at epoch N" after training moves on.
            if milestone_dir is not None and (epoch + 1) % milestone_every == 0:
                os.makedirs(milestone_dir, exist_ok=True)
                mpath = os.path.join(milestone_dir,
                                      f"model_{self.algo}_{self.preset}_{self._run_tag()}_ep{epoch+1}.npz")
                self.save(mpath)

            if patience >= patience_limit and epoch > 80:
                print(f"Early stop at epoch {epoch+1}")
                self.save_checkpoint(ckpt_dir, epoch)
                if milestone_dir is not None:
                    os.makedirs(milestone_dir, exist_ok=True)
                    mpath = os.path.join(milestone_dir,
                                          f"model_{self.algo}_{self.preset}_{self._run_tag()}_ep{epoch+1}.npz")
                    self.save(mpath)
                break
            if avg > 100 or np.isnan(avg):
                print(f"Diverged at epoch {epoch+1}")
                break

        print(f"\nDone in {time.time()-t0:.1f}s  best_loss={best_loss:.4f}")
        if milestone_dir is not None:
            os.makedirs(milestone_dir, exist_ok=True)
            final_epoch = self.curve["epoch"][-1] if self.curve["epoch"] else 0
            mpath = os.path.join(milestone_dir,
                                  f"model_{self.algo}_{self.preset}_{self._run_tag()}_ep{final_epoch}.npz")
            if not os.path.exists(mpath):
                self.save(mpath)

    def save_curve(self, path):
        """Save per-epoch training + validation curve to .npz for plotting."""
        cols = {k: np.array(self.curve.get(k, []), dtype=np.float32)
                for k in self.CURVE_KEYS}
        np.savez(path, algo=self.algo, preset=self.preset, **cols)
        print(f"Curve saved: {path}")

    def save(self, path):
        self.q_net.save(path, self.state_mean, self.state_std, algo=self.algo,
                        train_data=os.path.basename(self.data_file))
        print(f"Saved: {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",       required=True,
                        choices=["bc","dqn","ddqn","iql","dueling"],
                        help="Algorithm to train")
    parser.add_argument("--data",       required=True, help="Training data .npz")
    parser.add_argument("--epochs",     type=int,   default=300)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--gamma",      type=float, default=0.95)
    parser.add_argument("--target-clip", type=float, default=50.0,
                        help="Bellman target clip. RL-room used a hardcoded 10, which\n"
                             "saturated every target for these reward scales.")
    parser.add_argument("--cql-alpha",  type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int,   default=128)
    parser.add_argument("--iql-tau",    type=float, default=0.5,
                        help="Expectile level for IQL V-network (default 0.5)")
    parser.add_argument("--preset",     type=str,   default="balanced",
                        choices=["balanced", "energy", "throughput"],
                        help="Reward preset: balanced (0.40/0.25/0.35), "
                             "energy (0.25/0.15/0.60), throughput (0.55/0.20/0.25)")
    parser.add_argument("--val-split",  type=float, default=0.1,
                        help="Fraction of data held out for per-epoch validation (default 0.1)")
    parser.add_argument("--output",     type=str,   default=None,
                        help="Output .npz path (default: rl-data/model_<algo>_<preset>.npz)")
    parser.add_argument("--milestone-dir", type=str, default=None,
                        help="If set, save exportable model snapshots every "
                             "--milestone-every epochs here (for real-simulator "
                             "checkpoint evaluation; see eval_checkpoints.sh)")
    parser.add_argument("--milestone-every", type=int, default=25,
                        help="Epoch interval for milestone snapshots (default 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for weight init / minibatch sampling / val split "
                             "(default 42). Use distinct values for training-seed reruns.")
    parser.add_argument("--ckpt-dir", type=str, default="rl-data/checkpoints",
                        help="Resume-checkpoint directory. MUST be unique per (seed) when "
                             "running training-seed reruns — checkpoints are keyed only by "
                             "algo+preset, so a shared dir makes seed N resume from seed N-1.")
    parser.add_argument("--reward-weights", type=str, default=None,
                        help="Explicit 'w_sum,w_fair,w_net' overriding --preset "
                             "(stage 5 reward-weight sensitivity sweep).")
    parser.add_argument("--patience", type=int, default=60,
                        help="Early-stop patience in epochs-without-improvement on raw "
                             "training loss (default 60, fires once epoch > 80). NOTE: "
                             "in TD/expectile-regression training, raw loss legitimately "
                             "*rises* early on as Q-values differentiate from their "
                             "near-zero init, so epoch-1 loss is often never beaten again "
                             "and this silently caps every run at epoch ~82 regardless of "
                             "--epochs. Pass a very large value (e.g. 100000) to disable.")
    args = parser.parse_args()

    rw = None
    if args.reward_weights:
        rw = tuple(float(x) for x in args.reward_weights.split(","))
        assert len(rw) == 3, "--reward-weights needs 3 comma-separated numbers"

    np.random.seed(args.seed)
    random.seed(args.seed)

    out = args.output or f"rl-data/model_{args.algo}_{args.preset}.npz"

    trainer = OfflineDQNV2Trainer(
        data_file   = args.data,
        algo        = args.algo,
        hidden_dim  = args.hidden_dim,
        lr          = args.lr,
        gamma       = args.gamma,
        target_clip = args.target_clip,
        seed        = args.seed,
        batch_size  = args.batch_size,
        cql_alpha   = args.cql_alpha,
        iql_tau     = args.iql_tau,
        preset      = args.preset,
        val_split   = args.val_split,
        reward_weights = rw,
    )
    ckpt_dir = args.ckpt_dir
    trainer.curve = {k: [] for k in OfflineDQNV2Trainer.CURVE_KEYS}
    trainer.train(num_epochs=args.epochs, log_freq=10, ckpt_dir=ckpt_dir,
                  milestone_dir=args.milestone_dir, milestone_every=args.milestone_every,
                  patience_limit=args.patience)
    trainer.save(out)
    curve_path = out.replace(".npz", "_curve.npz")
    trainer.save_curve(curve_path)


if __name__ == "__main__":
    main()
