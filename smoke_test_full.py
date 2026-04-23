"""Smoke test: every (policy x algorithm) combo runs cleanly on M=2, K=20."""
from __future__ import annotations

import time
import numpy as np
import torch

from src.envs import ARRIVAL_RATES, batch_rollout
from src.policy import make_policy
from src.objectives import AlphaFairness
from src.train import train


def run(policy_kind, algorithm, algo_kwargs, M=2, K=20, H=500, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env_config = {"M": M, "L_max": 10, "lambdas": ARRIVAL_RATES[M]}
    policy = make_policy(policy_kind, M=M, L_max=10)
    obj = AlphaFairness(alpha=2.0, delta=1e-3, H=500)
    rng = np.random.default_rng(seed)
    rollout_fn = lambda H, B: batch_rollout(env_config, policy, H, B, rng)
    t0 = time.time()
    h = train(algorithm, policy, rollout_fn, obj, gamma=0.9999, K=K, H=H,
              rng=rng, lr=0.005, algo_kwargs=algo_kwargs, verbose=False)
    return h, time.time() - t0


def main():
    algo_configs = [
        ("bai_pg",      {"N1": 8, "N2": 8}),
        ("reinforce",   {"N": 8}),
        ("ganesh_npg",  {"B1": 8, "B2": 8, "N_inner": 3, "beta": 0.01}),
        ("ganesh_mlmc", {"B_max": 64, "B_inner": 8, "N_inner": 3, "beta": 0.01}),
    ]
    print(f"{'policy':<12} {'algorithm':<14} {'d':>6} {'f_init':>10} {'f_final':>10} {'time[s]':>8}")
    print("-" * 70)
    for policy_kind in ["log_linear", "tabular", "nn"]:
        for algo, kw in algo_configs:
            try:
                h, t = run(policy_kind, algo, kw)
                f0, f1 = h["f"][0], h["f"][-1]
                # d depends on policy; recompute from the policy dict quickly:
                from src.policy import make_policy as _mp
                d = _mp(policy_kind, M=2, L_max=10).d
                ok = np.isfinite(f0) and np.isfinite(f1)
                flag = "OK" if ok else "NAN"
                print(f"{policy_kind:<12} {algo:<14} {d:>6} {f0:>10.3f} {f1:>10.3f} {t:>8.1f}  {flag}")
            except Exception as e:
                print(f"{policy_kind:<12} {algo:<14} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
