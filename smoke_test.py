"""Smoke test: run all three algorithms briefly on M=2 and print progress.

Intended purely for sanity-checking the implementation end-to-end. Uses tiny
budgets so it completes in a minute or so on CPU.
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


def run_smoke(algorithm: str, algo_kwargs: dict, K: int = 40, M: int = 2,
              H: int = 500, seed: int = 0) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_config = {"M": M, "L_max": 10, "lambdas": ARRIVAL_RATES[M]}
    policy = LogLinearPolicy(M=M, L_max=10)
    obj = AlphaFairness(alpha=2.0, delta=1e-3)
    rng = np.random.default_rng(seed)
    rollout_fn = make_rollout_fn(env_config, policy, rng)

    t0 = time.time()
    history = train(
        algorithm=algorithm,
        policy=policy,
        rollout_fn=rollout_fn,
        obj=obj,
        gamma=0.9999,
        K=K,
        H=H,
        rng=rng,
        lr=0.005,
        algo_kwargs=algo_kwargs,
        verbose=False,
    )
    history["wall_time_sec"] = time.time() - t0
    return history


def main():
    K = 30
    configs = [
        ("bai_pg",      {"N1": 8, "N2": 8}),
        ("ganesh_npg",  {"B1": 8, "B2": 8, "N_inner": 3, "beta": 0.01}),
        ("ganesh_mlmc", {"B_max": 64, "B_inner": 8, "N_inner": 3, "beta": 0.01}),
    ]
    print(f"Smoke test: M=2, H=500, K={K}, alpha=2 fairness\n")
    print(f"{'algorithm':<15} {'f_init':>10} {'f_final':>10} {'J_final':>30} {'traj':>8} {'time[s]':>8}")
    print("-" * 90)
    for name, kw in configs:
        h = run_smoke(name, kw, K=K)
        f_init, f_final = h["f"][0], h["f"][-1]
        J_final = np.round(h["J_hat"][-1], 2)
        ntraj = h["trajectories"][-1]
        print(f"{name:<15} {f_init:>10.3f} {f_final:>10.3f} {str(J_final):>30} "
              f"{ntraj:>8d} {h['wall_time_sec']:>8.1f}")

        # Sanity assertions (don't hard-fail; just flag).
        if not np.isfinite(f_final):
            print(f"  WARNING: non-finite objective at final iter for {name}")
        if f_final < f_init - 1.0:
            print(f"  NOTE: {name} got worse over training ({f_init:.3f} -> {f_final:.3f})")


if __name__ == "__main__":
    main()
