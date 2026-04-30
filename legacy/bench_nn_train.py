"""Verify vmap is in use for scores() in NNPolicy, and measure train() wall time
on CPU vs CUDA, for both Bai-PG and Ganesh-NPG on M=2 and M=4.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from src.envs import ARRIVAL_RATES, batch_rollout
from src.policy import NNPolicy
from src.objectives import AlphaFairness
from src.train import train


def bench(device, M, algorithm, algo_kwargs, K=20, H=500, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = NNPolicy(M=M, L_max=10, device=device)
    env_config = {"M": M, "L_max": 10, "lambdas": ARRIVAL_RATES[M]}
    rng = np.random.default_rng(seed)

    def rollout_fn(H, B):
        return batch_rollout(env_config, policy, H, B, rng)

    obj = AlphaFairness(alpha=2.0, delta=1e-3, H=500)
    t0 = time.time()
    h = train(algorithm, policy, rollout_fn, obj, gamma=0.9999, K=K, H=H,
              rng=rng, lr=0.005, algo_kwargs=algo_kwargs, verbose=False)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return elapsed, h["f"][0], h["f"][-1]


def main():
    print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
    print()
    print(f"{'device':<6} {'M':>3} {'algorithm':<14} {'K':>4} {'time[s]':>10}  f_init  f_final")
    print("-" * 70)

    configs = [
        ("bai_pg",     {"N1": 16, "N2": 16}),
        ("ganesh_npg", {"B1": 16, "B2": 16, "N_inner": 3, "beta": 0.01}),
    ]
    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        for M in [2, 4]:
            for name, kw in configs:
                K = 20
                t, f0, f1 = bench(device, M, name, kw, K=K)
                print(f"{device:<6} {M:>3} {name:<14} {K:>4} {t:>10.2f}  {f0:7.2f} {f1:7.2f}")


if __name__ == "__main__":
    main()
