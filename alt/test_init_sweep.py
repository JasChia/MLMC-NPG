"""Quick empirical test: sweep q_init in {0, 2, 4, 6, 8} and confirm it
shifts the f curve to match the paper's reported values.

Runs a very short K=60 Bai-PG on M in {2, 4, 8}, N=64, 2 seeds per q_0.
Reports the mean f at the last iteration for each q_0.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from src.envs import ARRIVAL_RATES
from src.policy import LogLinearPolicy
from src.objectives import AlphaFairness
from src.train import train
from alt.envs_v2 import batch_rollout_v2


def run(q_init, M, K=60, N=64, H=500, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env_config = {"M": M, "L_max": 10, "lambdas": ARRIVAL_RATES[M]}
    policy = LogLinearPolicy(M=M, L_max=10)
    obj = AlphaFairness(alpha=2.0, delta=1e-3, H=500)
    rng = np.random.default_rng(seed)

    def rollout_fn(H, B):
        return batch_rollout_v2(env_config, policy, H, B, rng, q_init=q_init)

    h = train("bai_pg", policy, rollout_fn, obj, gamma=0.9999, K=K, H=H,
              rng=rng, lr=0.005, algo_kwargs={"N1": N, "N2": N}, verbose=False)
    return h


def main():
    print(f"Sweep q_init on M in {{2,4,8}}, K=60, N=64, 2 seeds/cfg.\n")
    print(f"{'M':>2} {'q_0':>4}  {'f_init_mean':>13}  {'f_final_mean':>13}  {'f_final_std':>13}")
    print("-" * 70)
    for M in (2, 4, 8):
        for q0 in (0, 2, 4, 6, 8):
            f_inits, f_finals = [], []
            for seed in (0, 1):
                t0 = time.time()
                h = run(q0, M, seed=seed)
                f_inits.append(h["f"][0]); f_finals.append(h["f"][-1])
            print(f"{M:>2} {q0:>4}  {np.mean(f_inits):>13.2f}  "
                  f"{np.mean(f_finals):>13.2f}  {np.std(f_finals):>13.2f}")
        print()


if __name__ == "__main__":
    main()
