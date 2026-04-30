"""Longer sanity run: does f actually improve across all M?

M=4 and M=8 start further from optimum under uniform policy and should
exhibit clearer learning signal in 80-100 outer iterations.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from src.envs import ARRIVAL_RATES, batch_rollout
from src.policy import LogLinearPolicy
from src.objectives import AlphaFairness
from src.train import train


def make_rollout_fn(env_config, policy, rng):
    def rollout_fn(H, B):
        return batch_rollout(env_config, policy, H, B, rng)
    return rollout_fn


def run_one(algorithm, algo_kwargs, M, K=80, H=500, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env_config = {"M": M, "L_max": 10, "lambdas": ARRIVAL_RATES[M]}
    policy = LogLinearPolicy(M=M, L_max=10)
    obj = AlphaFairness(alpha=2.0, delta=1e-3)
    rng = np.random.default_rng(seed)
    rollout_fn = make_rollout_fn(env_config, policy, rng)

    theta_init = policy.theta.clone()
    t0 = time.time()
    h = train(algorithm=algorithm, policy=policy, rollout_fn=rollout_fn,
              obj=obj, gamma=0.9999, K=K, H=H, rng=rng, lr=0.005,
              algo_kwargs=algo_kwargs, verbose=False)
    h["wall_time_sec"] = time.time() - t0
    h["theta_shift_L2"] = float(torch.linalg.vector_norm(policy.theta - theta_init).item())
    h["theta_dim"] = int(policy.d)
    return h


def main():
    configs = [
        ("bai_pg",      {"N1": 16, "N2": 16}),
        ("ganesh_npg",  {"B1": 16, "B2": 16, "N_inner": 5, "beta": 0.01}),
        ("ganesh_mlmc", {"B_max": 128, "B_inner": 16, "N_inner": 5, "beta": 0.01}),
    ]

    for M in (2, 4, 8):
        print(f"\n=== M = {M}  (d = {M*(M+1)}) ===")
        print(f"{'algorithm':<15} {'f_init':>10} {'f_mid':>10} {'f_final':>10} "
              f"{'|dtheta|':>10} {'traj':>8} {'time[s]':>8}")
        print("-" * 85)
        for name, kw in configs:
            K = 80
            h = run_one(name, kw, M=M, K=K)
            f = h["f"]
            print(f"{name:<15} {f[0]:>10.4f} {f[K//2]:>10.4f} {f[-1]:>10.4f} "
                  f"{h['theta_shift_L2']:>10.3f} {h['trajectories'][-1]:>8d} "
                  f"{h['wall_time_sec']:>8.1f}")


if __name__ == "__main__":
    main()
