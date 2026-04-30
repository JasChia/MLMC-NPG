"""Quick smoke test for GeneralAlphaFairness + MLMC under different alpha values."""
import time
import numpy as np
import torch

from src.envs import ARRIVAL_RATES
from src.policy import LogLinearPolicy
from src.train import train
from alt.envs_v2 import batch_rollout_v2
from alt.objectives_v2 import GeneralAlphaFairness


def run(algo, alpha, M=8, N=4, K=30, seed=0, q_init=5):
    np.random.seed(seed); torch.manual_seed(seed)
    env_config = {"M": M, "L_max": 10, "lambdas": ARRIVAL_RATES[M]}
    policy = LogLinearPolicy(M=M, L_max=10)
    obj = GeneralAlphaFairness(alpha=alpha, delta=1e-3, H=500)
    rng = np.random.default_rng(seed)

    def rollout_fn(H, B):
        return batch_rollout_v2(env_config, policy, H, B, rng, q_init=q_init)

    if algo == "bai_pg":
        kw = {"N1": N, "N2": N}
    elif algo == "ganesh_npg":
        kw = {"B1": N, "B2": N, "N_inner": 10, "beta": 0.01}
    elif algo == "ganesh_mlmc":
        kw = {"B_max": 128, "B_inner": N, "N_inner": 10, "beta": 0.01}
    t0 = time.time()
    h = train(algo, policy, rollout_fn, obj, gamma=0.9999, K=K, H=500,
              rng=rng, lr=0.005, algo_kwargs=kw, verbose=False)
    return h["f"][0], h["f"][-1], h["trajectories"][-1], time.time() - t0


def main():
    print(f"{'algo':<14} {'alpha':>6} {'f_init':>10} {'f_final':>10} {'traj':>8} {'time[s]':>8}")
    print("-" * 70)
    for alpha in (2.0, 3.0, 4.0):
        for algo in ("bai_pg", "ganesh_npg", "ganesh_mlmc"):
            f0, f1, traj, t = run(algo, alpha)
            print(f"{algo:<14} {alpha:>6.1f} {f0:>10.4f} {f1:>10.4f} {traj:>8d} {t:>8.1f}")
        print()


if __name__ == "__main__":
    main()
