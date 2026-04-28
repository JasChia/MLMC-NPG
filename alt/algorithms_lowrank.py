"""Memory-efficient NPG/MLMC-NPG steps for high-dimensional policies.

This keeps the paper algorithm unchanged while avoiding explicit construction
of the empirical Fisher matrix. The identity used is exact:

    F_hat @ omega = X.T @ (X @ omega) / B

where X is the gamma-weighted score matrix used by src.algorithms to form
F_hat = X.T @ X / B. This is required for NN d-scale sweeps where materializing
the d x d Fisher matrix is infeasible.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch


RolloutFn = Callable[[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    H = rewards.shape[1]
    gammas = gamma ** np.arange(H)
    return (rewards * gammas[None, :, None]).sum(axis=1)


def _reward_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    H = rewards.shape[1]
    gammas = (gamma ** np.arange(H))[None, :, None]
    discounted = rewards * gammas
    return np.cumsum(discounted[:, ::-1, :], axis=1)[:, ::-1, :]


def _pg_direction_from_scores(policy, scores, rewards, gamma, grad_f) -> torch.Tensor:
    B = rewards.shape[0]
    rtg = _reward_to_go(rewards, gamma)
    scalar_return = rtg @ grad_f
    scalar_return_t = torch.as_tensor(
        scalar_return, dtype=policy.dtype, device=policy.device
    )
    return (scores * scalar_return_t.unsqueeze(-1)).sum(dim=(0, 1)) / B


def _fisher_vec_product(scores: torch.Tensor, omega: torch.Tensor, gamma: float) -> torch.Tensor:
    """Return F_hat @ omega without materializing F_hat."""
    B, H, _ = scores.shape
    gammas = torch.tensor(gamma ** np.arange(H), dtype=scores.dtype, device=scores.device)
    weighted = scores * torch.sqrt(gammas).view(1, H, 1)
    flat = weighted.reshape(B * H, -1)
    return flat.T @ (flat @ omega) / B


def _solve_inner_npg_lowrank(policy, rollout_fn, grad_f, gamma, H, B2, N_inner, beta):
    d = policy.d
    omega = torch.zeros(d, device=policy.device, dtype=policy.dtype)
    total_traj = 0
    for _ in range(N_inner):
        states, actions, rewards = rollout_fn(H, B2)
        total_traj += B2
        scores = policy.scores(states, actions)
        g_hat = _pg_direction_from_scores(policy, scores, rewards, gamma, grad_f)
        f_omega = _fisher_vec_product(scores, omega, gamma)
        omega = omega - beta * (f_omega - g_hat)
    return omega.detach().cpu().numpy(), total_traj


def ganesh_npg_step_lowrank(policy, rollout_fn: RolloutFn, obj, gamma, H,
                            B1, B2, N_inner, beta):
    _, _, rr1 = rollout_fn(H, B1)
    J_hat = _discounted_returns(rr1, gamma).mean(axis=0)
    grad_f = obj.grad(J_hat)
    omega, inner_traj = _solve_inner_npg_lowrank(
        policy, rollout_fn, grad_f, gamma, H, B2, N_inner, beta
    )
    return omega, J_hat, B1 + inner_traj


def ganesh_mlmc_step_lowrank(policy, rollout_fn: RolloutFn, obj, gamma, H,
                             B_max, B_inner, N_inner, beta,
                             rng: np.random.Generator):
    Q = int(rng.geometric(0.5))
    Q_max = int(np.floor(np.log2(B_max)))
    if Q > Q_max:
        Q_bar = 0
        B_k = 1
    else:
        Q_bar = Q
        B_k = 2 ** Q_bar

    _, _, rr = rollout_fn(H, B_k)
    J_samples = _discounted_returns(rr, gamma)
    J_H1 = J_samples[0]

    if Q_bar == 0:
        grad_f = obj.grad(J_H1)
        J_report = J_H1
    else:
        J_full = J_samples.mean(axis=0)
        J_half = J_samples[: B_k // 2].mean(axis=0)
        grad_f = obj.grad(J_H1) + (2 ** Q_bar) * (
            obj.grad(J_full) - obj.grad(J_half)
        )
        J_report = J_full

    omega, inner_traj = _solve_inner_npg_lowrank(
        policy, rollout_fn, grad_f, gamma, H, B_inner, N_inner, beta
    )
    return omega, J_report, B_k + inner_traj
