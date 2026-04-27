"""Test stochastic q_0 variants against paper's f values."""
from __future__ import annotations

import numpy as np
import torch

from src.envs import ARRIVAL_RATES
from src.policy import LogLinearPolicy
from src.objectives import AlphaFairness
from src.train import train
from alt.envs_v2 import batch_rollout_v2


def run(q_init, M, K=60, N=64, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)
    env_config = {"M": M, "L_max": 10, "lambdas": ARRIVAL_RATES[M]}
    policy = LogLinearPolicy(M=M, L_max=10)
    obj = AlphaFairness(alpha=2.0, delta=1e-3, H=500)
    rng = np.random.default_rng(seed)

    def rollout_fn(H, B):
        return batch_rollout_v2(env_config, policy, H, B, rng, q_init=q_init)

    h = train("bai_pg", policy, rollout_fn, obj, gamma=0.9999, K=K, H=500,
              rng=rng, lr=0.005, algo_kwargs={"N1": N, "N2": N}, verbose=False)
    return h["f"][-1]


def main():
    q_inits = {
        "q_0 = 0       ": 0,
        "q_0 = 5       ": 5,
        "uniform[0,10] ": "uniform",
    }
    paper_targets = {2: -7.6, 4: -25.3, 8: -126.0}

    print(f"{'q_init':<16} {'M=2':>10} {'M=4':>10} {'M=8':>10}")
    print("-" * 60)
    for label, qi in q_inits.items():
        vals = []
        for M in (2, 4, 8):
            res = [run(qi, M, seed=s) for s in (0, 1)]
            vals.append(np.mean(res))
        print(f"{label:<16} {vals[0]:>10.2f} {vals[1]:>10.2f} {vals[2]:>10.2f}")
    print(f"{'paper target':<16} "
          f"{paper_targets[2]:>10.2f} {paper_targets[4]:>10.2f} {paper_targets[8]:>10.2f}")


if __name__ == "__main__":
    main()
