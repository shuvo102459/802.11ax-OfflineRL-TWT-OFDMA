#!/usr/bin/env python3
"""Evaluate a policy in ns-3 and report the metrics the paper's claims rest on.

Everything here runs the SAME scenario the dataset was collected from
(scenario4_rl_twt_ofdma, with the non-stationary load step), so a learned policy
and a baseline see identical traffic for a given seed. That is what makes the
comparison paired, and paired is what lets a 150-seed bootstrap resolve the
differences we care about.

POLICIES
  random                      uniform over the 760 joint actions. This is the
                              behaviour policy, so it is also the dataset's own
                              performance -- the floor any learning must clear.
  fixed:<sched>:<asg>:<ofdma> one action for the whole episode. The static-TWT
                              baselines, including sched 0 (90 ms wake of a
                              102.4 ms BI = 87.9% duty), the highest-duty
                              schedule the action space contains. Note this is
                              the in-space ceiling, NOT a true always-on station;
                              report it as max-duty TWT, not as "always-on".
  latency:<hi>:<lo>:...       THE HEURISTIC THAT MATTERS. A reviewer will ask why
                              a learned policy is needed when an engineer would
                              just raise the duty cycle once latency climbs. So
                              that engineer's policy is implemented here, with
                              hysteresis, and its thresholds are swept -- the
                              comparison is against the BEST heuristic found, not
                              a strawman.
  learned:<ckpt.npz>          greedy argmax over the 760 actions.

Energy is per-decision-window (verified non-monotonic in the logs), so episode
energy is the sum over steps, and the headline figure is mJ per station per
decision, which is comparable across episodes of unequal length.
"""
import argparse, json, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RL   = os.path.abspath(os.path.join(HERE, "..", "..", "RL-room"))
sys.path.insert(0, RL)

# Reused rather than re-derived: any drift between the collector's state/obs
# construction and the evaluator's would silently invalidate the comparison.
from reward_functions import reward_per_sta_with_fairness          # noqa: E402
from scenario4_collect import (                                   # noqa: E402
    extract_state, build_sta_observations, jain_fairness,
    ACTION_TABLE, NUM_TWT_ACTIONS, NUM_ACTIONS, NUM_STA, SHM_SIZE, _NS3_ROOT,
)
# Importing the collector above already put ns-3's python_utils and the binding
# directory on sys.path, so these resolve the same way they do for collection.
import ns3ai_scenario4_py as py_binding                           # noqa: E402
from ns3ai_utils import Experiment                                # noqa: E402

BI_MS = 102.4


# ── policies ──────────────────────────────────────────────────────────────────

class Random:
    name = "random"
    def __init__(self, seed): self.rng = random.Random(seed)
    def reset(self): pass
    def act(self, state, obs): return self.rng.randint(0, NUM_ACTIONS - 1)


class RandomTwt:
    """Uniform over the 380 TWT actions with the OFDMA bit HELD FIXED.

    Distinct from `random`, which samples the joint 760-way space and therefore
    mixes ~50% OFDMA into one arm. The submitted paper separated these as
    "Random TWT only" and "Random TWT + OFDMA", and the separation is the point:
    with the TWT distribution identical and only the OFDMA bit differing, the
    pair is a controlled estimate of what OFDMA costs under uninformed
    scheduling.
    """
    def __init__(self, ofdma, seed=0):
        self.of = int(ofdma)
        self.name = f"random_twt_ofdma{self.of}"
        self._seed = seed
    def reset(self): self.rng = random.Random(self._seed)
    def act(self, state, obs):
        return self.of * NUM_TWT_ACTIONS + self.rng.randint(0, NUM_TWT_ACTIONS - 1)


class Fixed:
    def __init__(self, sched, asg, ofdma):
        self.a = int(ofdma) * NUM_TWT_ACTIONS + int(sched) * 19 + int(asg)
        self.name = f"fixed_s{sched}_a{asg}_o{ofdma}"
    def reset(self): pass
    def act(self, state, obs): return self.a


class LatencyTriggered:
    """Raise duty (and optionally enable OFDMA) when queueing delay climbs.

    Hysteresis (hi > lo) prevents flapping between schedules every window, which
    would pay the schedule-change cost without gaining anything. Holding the
    previous action in the dead band is the whole point of the design.
    """
    def __init__(self, hi_ms, lo_ms, hi_sched, lo_sched, asg, ofdma_mode):
        self.hi, self.lo = float(hi_ms), float(lo_ms)
        self.hs, self.ls, self.asg = int(hi_sched), int(lo_sched), int(asg)
        self.mode = ofdma_mode          # off | on | with_load
        self.name = (f"latency_hi{self.hi:g}_lo{self.lo:g}_"
                     f"s{self.hs}-{self.ls}_a{self.asg}_o-{ofdma_mode}")
        self.reset()

    def reset(self):
        self.sched = self.ls
        self.loaded = False

    def act(self, state, obs):
        d = float(np.mean([s["queue_delay_ms"] for s in obs]))
        if d > self.hi:
            self.sched, self.loaded = self.hs, True
        elif d < self.lo:
            self.sched, self.loaded = self.ls, False
        ofdma = 1 if self.mode == "on" else (1 if (self.mode == "with_load" and self.loaded) else 0)
        return ofdma * NUM_TWT_ACTIONS + self.sched * 19 + self.asg


class AlwaysOn:
    """No power save at all: the scenario never sleeps a station.

    Distinct from max-duty TWT (schedule 0), which still sleeps for 12.1% of every
    beacon interval because its wake window is 90 ms of 102.4 ms. The submitted
    paper carried a true always-on row and this study did not, which made its
    energy numbers harder to place; `--enableTwt=false` restores it. The action
    sent each step is irrelevant to energy here -- nothing is ever put to sleep --
    so a fixed, OFDMA-off action is used.
    """
    name = "always_on_no_twt"
    twt_enabled = False
    def __init__(self, ofdma=0):
        self.a = int(ofdma) * NUM_TWT_ACTIONS + 0 * 19 + 17
        if int(ofdma): self.name = "always_on_no_twt_ofdma"
    def reset(self): pass
    def act(self, state, obs): return self.a


class Learned:
    def __init__(self, path):
        sys.path.insert(0, HERE)
        from fqe import load_q, greedy_actions
        self.q, _, _, _ = load_q(path)
        self._g = greedy_actions
        self.name = "learned_" + os.path.basename(path).replace(".npz", "")
    def reset(self): pass
    def act(self, state, obs):
        return int(self._g(self.q, state.reshape(1, -1))[0])


def make_policy(spec, seed):
    kind, *rest = spec.split(":")
    if kind == "random":   return Random(seed)
    if kind == "alwayson": return AlwaysOn(*rest)
    if kind == "randomtwt": return RandomTwt(rest[0], seed)
    if kind == "fixed":    return Fixed(*rest)
    if kind == "latency":  return LatencyTriggered(*rest)
    if kind == "learned":  return Learned(":".join(rest))
    raise SystemExit(f"unknown policy '{spec}'")


# ── one episode ───────────────────────────────────────────────────────────────

def run_episode(policy, seed, seg_name, nonstationary=True, trace=False,
                verbose=False, progress_every=0, sleep_guard_us=None, sleep_guard_symmetric=False,
                hold_steps=1, sp_aligned=False, cca_current_ma=None,
                n_stations=None):
    random.seed(seed); np.random.seed(seed)

    # The load profile MUST be derived exactly as the collector derives it, so a
    # given seed is the same world for every policy compared on it.
    load_step_at, load_step_mult = -1.0, 2.0
    if nonstationary:
        _r = random.Random(seed * 7919 + 13)
        if _r.random() < 0.60:
            load_step_at   = round(_r.uniform(30.0, 70.0), 2)
            load_step_mult = round(_r.choice([1.5, 2.0, 2.5]), 2)

    # One segment per EPISODE. Reusing a name across episodes raises
    # boost::interprocess_exception::library_error on the second one, because the
    # first segment is still mapped.
    seg = f"{seg_name}{seed}"
    exp = Experiment(targetName="scenario4_rl_twt_ofdma", ns3Path=_NS3_ROOT,
                     msgModule=py_binding, handleFinish=True,
                     shmSize=SHM_SIZE, segName=seg)
    # sleepGuardUs / sleepGuardSymmetric expose the 8 ms PHY-sleep deferral the
    # scenario applies when the OFDMA bit is set. Historically it applied in the
    # OFDMA-on arm only, so that comparison varied sleep handling as well as
    # OFDMA. The defaults below reproduce the historical behaviour exactly.
    setting = {"seed": seed, "preset": "energy", "segName": seg,
               "enableOfdma": "false",
               "enableTwt": "true" if getattr(policy, "twt_enabled", True) else "false",
               "loadStepAtS": load_step_at,
               "loadStepMult": load_step_mult}
    if sleep_guard_us is not None:
        setting["sleepGuardUs"] = int(sleep_guard_us)
    if sleep_guard_symmetric:
        setting["sleepGuardSymmetric"] = "true"
    if sp_aligned:
        setting["spAligned"] = "true"
    if cca_current_ma is not None:
        setting["ccaCurrentMa"] = float(cca_current_ma)
    if n_stations is not None:
        setting["nStations"] = int(n_stations)
    msg = exp.run(setting=setting, show_output=verbose)

    policy.reset()
    held = None
    # Per-step reward under the CORRECTED energy objective, so the simulator's
    # discounted return can be compared against the FQE value that selected the
    # checkpoint. FQE estimates a discounted return, not energy; validating it
    # against energy alone (as we first did) checks a correlate rather than the
    # quantity itself.
    RW = []
    E, TP, FA, DL, DR, ACT = [], [], [], [], [], []
    tr, step = [], 0
    try:
        while True:
            msg.PyRecvBegin()
            if msg.PyGetFinished():
                break
            env   = msg.GetCpp2PyStruct()
            state = extract_state(env)
            obs   = build_sta_observations(env)
            msg.PyRecvEnd()

            if step > 0:                      # step 0 precedes any action of ours
                E.append(float(np.mean([s["energy_consumed_mj"] for s in obs])))
                TP.append(float(sum(s["throughput_mbps"] for s in obs)))
                FA.append(jain_fairness([s["throughput_mbps"] for s in obs]))
                DL.append(float(np.mean([s["queue_delay_ms"] for s in obs])))
                DR.append(float(sum(s["mpdu_drops_expired"] for s in obs)))
                # returns (R_final, R_sum, R_fairness, R_network) -- unpack all
                # four. Do NOT swallow a failure here: an empty reward trace that
                # looks like "no data" is how several earlier checks silently
                # reported nothing while appearing to succeed.
                # same call convention as build_dataset.py:110 -- the function
                # takes a metrics DICT, not a bare observation list
                _rf, _rs, _rfa, _rn = reward_per_sta_with_fairness(
                    {"sta_observations": obs}, return_components=True, preset="energy")
                RW.append(dict(r_sum=float(_rs), r_fair=float(_rfa), r_net=float(_rn),
                               energy_mj=float(sum(s["energy_consumed_mj"] for s in obs))))

            # ACTION HOLD. The dataset was collected holding each action for ten
            # decisions, so a policy deployed at one-step cadence is operating at a
            # cadence its training data barely contains. hold_steps=10 deploys at
            # the behaviour policy's cadence; comparing the two measures whether
            # that mismatch costs anything.
            if hold_steps > 1 and step % hold_steps != 0 and held is not None:
                a = held
            else:
                a = policy.act(state, obs)
                held = a
            msg.PySendBegin()
            act = msg.GetPy2CppStruct()
            act.ofdma_enabled = a // NUM_TWT_ACTIONS
            act.schedule_id, act.assignment_id = ACTION_TABLE[a % NUM_TWT_ACTIONS]
            msg.PySendEnd()
            ACT.append(a)
            if progress_every and step % progress_every == 0:
                print(f"    [s{seed} step {step}] d={np.mean([x['queue_delay_ms'] for x in obs]):.0f}ms "
                      f"a={a}", flush=True)
            if trace:
                tr.append(dict(step=step, action=a,
                               delay_ms=float(np.mean([s["queue_delay_ms"] for s in obs]))))
            step += 1
    finally:
        del exp

    if not E:
        return None
    A = np.asarray(ACT[:-1] if len(ACT) > len(E) else ACT)
    DLa = np.asarray(DL)
    return dict(
        seed=seed, policy=policy.name, steps=len(E),
        load_step_at_s=load_step_at, load_step_mult=load_step_mult,
        stepped=bool(load_step_at >= 0),
        energy_mj_per_sta_per_step=float(np.mean(E)),
        energy_mj_total=float(np.sum(E) * NUM_STA),
        throughput_mbps=float(np.mean(TP)),
        fairness=float(np.mean(FA)),
        delay_ms_mean=float(DLa.mean()),
        delay_ms_p95=float(np.percentile(DLa, 95)),
        delay_budget_viol_frac=float((DLa > 3000.0).mean()),
        reward_trace=RW,
        drops=float(np.sum(DR)),
        ofdma_on_share=float((A >= NUM_TWT_ACTIONS).mean()),
        distinct_actions=int(len(np.unique(A))),
        modal_schedule=int(np.bincount((A % NUM_TWT_ACTIONS) // 19).argmax()),
        trace=tr if trace else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    help="random | fixed:s:a:o | latency:hi:lo:hs:ls:asg:mode | learned:ckpt.npz")
    ap.add_argument("--seeds", required=True, help="e.g. 90000-90149")
    ap.add_argument("--sp-aligned", action="store_true",
                    help="only solicit a station whose TWT service period still has room "
                         "for the whole trigger exchange; makes the sleep guard unnecessary")
    ap.add_argument("--n-stations", type=int, default=None,
                    help="Max stations granted an RU per trigger frame. ns-3's "
                         "RrMultiUserScheduler default is 4; a 20 MHz channel "
                         "admits 9 26-tone RUs. Bounds the multiplexing degree.")
    ap.add_argument("--cca-current-ma", type=float, default=None,
                    help="CCA-busy current in mA (273 = ns-3 default, 66 = receive current)")
    ap.add_argument("--hold-steps", type=int, default=1,
                    help="repeat each chosen action for N decisions (the dataset used N=10)")
    ap.add_argument("--sleep-guard-us", type=int, default=None,
                    help="PHY-sleep deferral in us (scenario default 8000; 0 = immediate)")
    ap.add_argument("--sleep-guard-symmetric", action="store_true",
                    help="apply the sleep guard in BOTH OFDMA arms, removing the "
                         "asymmetry from the OFDMA comparison")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seg", default="evalNS", help="shm segment name (unique per worker)")
    ap.add_argument("--tag", default=None, help="label for filenames; defaults to policy name")
    ap.add_argument("--stationary", action="store_true", help="disable the load step")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="show ns-3 stdout")
    ap.add_argument("--progress-every", type=int, default=10,
                    help="heartbeat every N decisions; 0 disables. A silent run is\n"
                         "indistinguishable from a hung one, which cost a smoke test.")
    a = ap.parse_args()

    lo, hi = (a.seeds.split("-") + [None])[:2]
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    os.makedirs(a.out, exist_ok=True)

    pol = make_policy(a.policy, seed=12345)
    tag = a.tag or pol.name

    # MULTI-SEED RUNS FAN OUT TO ONE CHILD PROCESS PER EPISODE.
    # ns3ai_utils.Experiment is a singleton (ns3ai_utils.py:110 raises
    # 'Experiment is singleton'), so several episodes in one process is against
    # the library's design: the second Experiment() died with
    # boost::interprocess_exception::library_error, which cost an overnight run --
    # every policy produced exactly ONE episode and the rest failed.
    # Each child is resumable on its own output file, so an interrupted sweep
    # resumes at the episode it stopped on.
    if len(seeds) > 1:
        import subprocess
        done = fail = 0
        for s in seeds:
            fp = os.path.join(a.out, f"{tag}_s{s}.json")
            if os.path.exists(fp):
                done += 1
                continue
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--policy", a.policy, "--seeds", str(s), "--out", a.out,
                   "--seg", a.seg, "--progress-every", str(a.progress_every),
                   "--hold-steps", str(a.hold_steps)]
            if a.sp_aligned:              cmd += ["--sp-aligned"]
            if a.cca_current_ma is not None:
                cmd += ["--cca-current-ma", str(a.cca_current_ma)]
            if a.n_stations is not None:
                cmd += ["--n-stations", str(a.n_stations)]
            if a.tag:        cmd += ["--tag", a.tag]
            if a.stationary: cmd += ["--stationary"]
            if a.trace:      cmd += ["--trace"]
            if a.verbose:    cmd += ["--verbose"]
            # EVERY option that changes the simulation must be forwarded here.
            # These two reached the parser and the ns-3 setting dict but not this
            # child command, so a single-seed run applied them and a multi-seed
            # run silently did not. The symmetric-guard arm therefore produced 50
            # episodes of no-guard data, byte-identical to its control, twice.
            if a.sleep_guard_us is not None:
                cmd += ["--sleep-guard-us", str(a.sleep_guard_us)]
            if a.sleep_guard_symmetric:
                cmd += ["--sleep-guard-symmetric"]
            rc = subprocess.call(cmd)
            if rc == 0 and os.path.exists(fp):
                done += 1
            else:
                fail += 1
                print(f"  seed {s}: FAILED (rc={rc})", flush=True)
        print(f"  {tag}: {done}/{len(seeds)} episodes present, {fail} failed", flush=True)
        # a mostly-failed sweep must not look like success to the caller
        return 0 if fail == 0 else 1

    s = seeds[0]
    fp = os.path.join(a.out, f"{tag}_s{s}.json")
    if os.path.exists(fp):
        return 0
    r = run_episode(pol, s, a.seg, nonstationary=not a.stationary, trace=a.trace,
                    sleep_guard_us=a.sleep_guard_us,
                    sleep_guard_symmetric=a.sleep_guard_symmetric,
                    hold_steps=a.hold_steps, sp_aligned=a.sp_aligned,
                    cca_current_ma=a.cca_current_ma,
                    n_stations=a.n_stations,
                    verbose=a.verbose, progress_every=a.progress_every)
    if r is None:
        print(f"  seed {s}: NO DATA", flush=True)
        return 1
    tmp = fp + ".tmp"
    json.dump(r, open(tmp, "w"), indent=1); os.replace(tmp, fp)
    print(f"  {tag} s{s}: E {r['energy_mj_per_sta_per_step']:7.2f} mJ  "
          f"TP {r['throughput_mbps']:6.2f}  d {r['delay_ms_mean']:8.1f} ms  "
          f"OFDMA {100*r['ofdma_on_share']:5.1f}%  sched {r['modal_schedule']}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
