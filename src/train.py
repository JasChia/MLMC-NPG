"""Outer training loop + Adam updater for the three algorithms.

Adam is applied to the computed ascent direction omega, not to a conventional
loss gradient: we treat omega as the "gradient of f w.r.t. theta" and take
an ascent step. This matches Paper A's simulation setup (they use Adam as well).
"""
from __future__ import annotations

from typing import Callable, Dict, Any

import numpy as np
import torch

from .algorithms import (bai_pg_step, ganesh_mlmc_step,
                          reinforce_step, scalarized_reinforce_step,
                          linear_reinforce_step)


class AdamAscent:
    """Adam with explicit ascent direction (maximize f).

    theta <- theta + lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, shape, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8,
                 device="cpu", dtype=torch.float32):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.t = 0

    def step(self, theta: torch.Tensor, omega) -> None:
        self.t += 1
        g = torch.as_tensor(omega, device=theta.device, dtype=theta.dtype)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g * g
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        theta += self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)


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
    """Run K outer iterations and return a per-iteration history.

    history keys:
        iter           : list[int]
        J_hat          : list[list[float]]   per-objective return estimates
        f              : list[float]         scalarized objective at J_hat
        trajectories   : list[int]           cumulative trajectory count
    """
    algo_kwargs = dict(algo_kwargs or {})
    opt = AdamAscent(policy.theta.shape, lr=lr,
                     device=policy.device, dtype=policy.dtype)

    history = {"iter": [], "J_hat": [], "f": [], "trajectories": []}
    total_traj = 0

    for k in range(K):
        if algorithm == "bai_pg":
            omega, J_hat, ntraj = bai_pg_step(policy, rollout_fn, obj, gamma, H, **algo_kwargs)
        elif algorithm == "ganesh_mlmc":
            omega, J_hat, ntraj = ganesh_mlmc_step(policy, rollout_fn, obj, gamma, H, rng=rng, **algo_kwargs)
        elif algorithm == "reinforce":
            omega, J_hat, ntraj = reinforce_step(policy, rollout_fn, obj, gamma, H, **algo_kwargs)
        elif algorithm == "scalarized_reinforce":
            omega, J_hat, ntraj = scalarized_reinforce_step(policy, rollout_fn, obj, gamma, H, **algo_kwargs)
        elif algorithm == "linear_reinforce":
            omega, J_hat, ntraj = linear_reinforce_step(policy, rollout_fn, obj, gamma, H, **algo_kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        opt.step(policy.theta, omega)
        total_traj += ntraj

        history["iter"].append(k)
        history["J_hat"].append([float(x) for x in J_hat])
        history["f"].append(obj.value(J_hat))
        history["trajectories"].append(total_traj)

        if verbose and (k % max(1, K // 20) == 0 or k == K - 1):
            print(f"  iter {k:4d} | f = {history['f'][-1]: .4f} | J = {np.round(J_hat, 3)} | ntraj = {total_traj}")

    return history
