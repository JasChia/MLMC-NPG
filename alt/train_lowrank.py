"""Training loop using low-rank NPG products for Ganesh algorithms.

Bai-PG and REINFORCE are delegated to src.algorithms unchanged. Ganesh NPG and
MLMC-NPG use exact low-rank Fisher-vector products from alt.algorithms_lowrank.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import torch

from alt.algorithms_lowrank import ganesh_mlmc_step_lowrank, ganesh_npg_step_lowrank
from src.algorithms import bai_pg_step, reinforce_step
from src.train import AdamAscent


def train(
    algorithm: str,
    policy,
    rollout_fn: Callable,
    obj,
    gamma: float,
    K: int,
    H: int,
    rng: np.random.Generator,
    lr: float = 0.005,
    algo_kwargs: Dict[str, Any] = None,
    verbose: bool = False,
) -> Dict[str, list]:
    algo_kwargs = dict(algo_kwargs or {})
    opt = AdamAscent(policy.theta.shape, lr=lr, device=policy.device, dtype=policy.dtype)

    history = {"iter": [], "J_hat": [], "f": [], "trajectories": []}
    total_traj = 0

    for k in range(K):
        if algorithm == "bai_pg":
            omega, J_hat, ntraj = bai_pg_step(policy, rollout_fn, obj, gamma, H, **algo_kwargs)
        elif algorithm == "ganesh_npg":
            omega, J_hat, ntraj = ganesh_npg_step_lowrank(
                policy, rollout_fn, obj, gamma, H, **algo_kwargs
            )
        elif algorithm == "ganesh_mlmc":
            omega, J_hat, ntraj = ganesh_mlmc_step_lowrank(
                policy, rollout_fn, obj, gamma, H, rng=rng, **algo_kwargs
            )
        elif algorithm == "reinforce":
            omega, J_hat, ntraj = reinforce_step(policy, rollout_fn, obj, gamma, H, **algo_kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        opt.step(policy.theta, omega)
        total_traj += ntraj

        history["iter"].append(k)
        history["J_hat"].append([float(x) for x in J_hat])
        history["f"].append(obj.value(J_hat))
        history["trajectories"].append(total_traj)

        if verbose and (k % max(1, K // 20) == 0 or k == K - 1):
            print(
                f"  iter {k:4d} | f = {history['f'][-1]: .4f} | "
                f"J = {np.round(J_hat, 3)} | ntraj = {total_traj}"
            )

    return history
