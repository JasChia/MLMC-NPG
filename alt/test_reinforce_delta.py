"""Calibrate REINFORCE's delta_guard to match Paper A Fig 2 (where PG > REINFORCE)."""
import numpy as np
import torch

from src.envs import ARRIVAL_RATES
from src.policy import LogLinearPolicy
from src.train import train
from alt.envs_v2 import batch_rollout_v2
from alt.objectives_v2 import GeneralAlphaFairness


def run(delta_reinforce, M=8, N=16, K=200, q_init=5, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)
    env_config = {"M": M, "L_max": 10, "lambdas": ARRIVAL_RATES[M]}
    policy = LogLinearPolicy(M=M, L_max=10)
    obj = GeneralAlphaFairness(alpha=2.0, delta=1e-3, H=500)
    rng = np.random.default_rng(seed)
    rf = lambda H, B: batch_rollout_v2(env_config, policy, H, B, rng, q_init=q_init)
    kw = {"N": N, "delta_guard": delta_reinforce}
    h = train("reinforce", policy, rf, obj, gamma=0.9999, K=K, H=500,
              rng=rng, lr=0.005, algo_kwargs=kw, verbose=False)
    return h["f"][-1]


print("M=8, N=16, K=200, q_0=5:")
print(f"{'delta_reinforce':>18} {'f_final (mean of 3 seeds)':>28}")
print("-" * 50)
for d in (1e-3, 0.1, 1.0, 10.0, 100.0):
    fs = [run(d, seed=s) for s in (0, 1, 2)]
    print(f"{d:>18g} {np.mean(fs):>18.2f}  ±{np.std(fs):>5.2f}")
