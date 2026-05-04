"""Three concave-MORL algorithm steps on a shared rollout/policy/objective interface.

bai_pg_step       — Vanilla PG with biased plug-in estimator (prior-work baseline, Alg. 1).
ganesh_mlmc_step  — MLMC-NPG (this work, Algorithm 2): no 2nd-order smoothness required.
reinforce_step    — REINFORCE baseline (prior-work eq. 38) with per-step training reward.

Each function signature:
    (policy, rollout_fn, obj, gamma, H, <hyperparams>) -> (omega, J_hat, n_traj)

where `rollout_fn(H, B)` returns (states, actions, rewards) of shape
    (B, H, M) int32, (B, H) int64, (B, H, M) float32.

`omega` is the ascent direction (numpy, shape (d,)); the outer loop feeds
omega to an Adam updater applied to policy.theta.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch


RolloutFn = Callable[[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """rewards: (B, H, M) -> J: (B, M), J[b] = sum_t gamma^t rewards[b, t]."""
    H = rewards.shape[1]
    gammas = gamma ** np.arange(H)
    return (rewards * gammas[None, :, None]).sum(axis=1)


def _reward_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """PGT reward-to-go: rtg[b, t, m] = sum_{h>=t} gamma^h rewards[b, h, m]."""
    H = rewards.shape[1]
    gammas = (gamma ** np.arange(H))[None, :, None]
    discounted = rewards * gammas
    return np.cumsum(discounted[:, ::-1, :], axis=1)[:, ::-1, :]


def _pg_direction(policy, states, actions, rewards, gamma, grad_f) -> torch.Tensor:
    """Compute   (1/B) sum_i sum_t score_{i,t} * [ grad_f . rtg_{i,t} ].

    Shared gradient estimator for vanilla PG (with grad_f = df(J_hat)) and
    the MLMC-NPG inner loop (with a different grad_f but the same form).
    """
    B = rewards.shape[0]
    rtg = _reward_to_go(rewards, gamma)  # (B, H, M)
    scalar_return = rtg @ grad_f  # (B, H) numpy
    scalar_return_t = torch.as_tensor(scalar_return, dtype=policy.dtype, device=policy.device)
    scores = policy.scores(states, actions)  # (B, H, d)
    return (scores * scalar_return_t.unsqueeze(-1)).sum(dim=(0, 1)) / B


def _empirical_fisher(scores: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute F_hat = (1/B) sum_i sum_t gamma^t score_{i,t} score_{i,t}^T.

    scores: (B, H, d). Returns (d, d).
    """
    B, H, d = scores.shape
    gammas = torch.tensor(gamma ** np.arange(H), dtype=scores.dtype, device=scores.device)
    weighted = scores * torch.sqrt(gammas).view(1, H, 1)
    flat = weighted.reshape(B * H, d)
    return (flat.T @ flat) / B


def _solve_inner_npg(policy, rollout_fn, grad_f, gamma, H, B2, N_inner, beta):
    """Inner SGD loop: omega_{n+1} = omega_n - beta (F_hat omega_n - g_hat).

    Fresh batch of B2 trajectories per inner step. Returns (omega numpy (d,),
    trajectories consumed).
    """
    d = policy.d
    omega = torch.zeros(d, device=policy.device, dtype=policy.dtype)
    total_traj = 0
    for _ in range(N_inner):
        states, actions, rewards = rollout_fn(H, B2)
        total_traj += B2
        g_hat = _pg_direction(policy, states, actions, rewards, gamma, grad_f)
        scores = policy.scores(states, actions)
        F_hat = _empirical_fisher(scores, gamma)
        omega = omega - beta * (F_hat @ omega - g_hat)
    return omega.detach().cpu().numpy(), total_traj


# -----------------------------------------------------------------------------
# Vanilla PG with biased plug-in estimator (prior-work baseline, Algorithm 1)
# -----------------------------------------------------------------------------

def bai_pg_step(policy, rollout_fn: RolloutFn, obj, gamma, H, N1, N2):
    """One outer iteration of the biased plug-in vanilla PG step.

    Uses the biased plug-in estimator: df(J_hat) with J_hat from an independent
    batch of N2 trajectories, then forms the gradient from another N1
    trajectories so that J_hat and omega are independent.
    """
    _, _, rr2 = rollout_fn(H, N2)
    J_hat = _discounted_returns(rr2, gamma).mean(axis=0)  # (M,)
    grad_f = obj.grad(J_hat)

    ss1, aa1, rr1 = rollout_fn(H, N1)
    omega = _pg_direction(policy, ss1, aa1, rr1, gamma, grad_f)
    return omega.detach().cpu().numpy(), J_hat, N1 + N2


# -----------------------------------------------------------------------------
# MLMC-NPG (this work, Algorithm 2)
# -----------------------------------------------------------------------------

def ganesh_mlmc_step(policy, rollout_fn: RolloutFn, obj, gamma, H,
                     B_max, B_inner, N_inner, beta, rng: np.random.Generator):
    """One outer iteration of MLMC-NPG (Algorithm 2).

    MLMC return estimator:
        Q ~ Geom(1/2)
        if 2^Q > B_max : fall back to J_{H,1} (single trajectory)
        else:
            df(J_MLMC) = df(J_{H,1}) + 2^Q [df(J_{H,2^Q}) - df(J_{H, 2^{Q-1}})]
        with J_{H, 2^{Q-1}} using the FIRST HALF of the 2^Q trajectories
        that produced J_{H, 2^Q}  (the nested coupling).
    """
    Q = int(rng.geometric(0.5))
    Q_max = int(np.floor(np.log2(B_max)))
    if Q > Q_max:
        Q_bar = 0
        B_k = 1
    else:
        Q_bar = Q
        B_k = 2 ** Q_bar

    _, _, rr = rollout_fn(H, B_k)
    J_samples = _discounted_returns(rr, gamma)  # (B_k, M)
    J_H1 = J_samples[0]

    if Q_bar == 0:
        grad_f = obj.grad(J_H1)
        J_report = J_H1
    else:
        J_full = J_samples.mean(axis=0)
        J_half = J_samples[: B_k // 2].mean(axis=0)
        grad_f = obj.grad(J_H1) + (2 ** Q_bar) * (obj.grad(J_full) - obj.grad(J_half))
        J_report = J_full

    omega, inner_traj = _solve_inner_npg(policy, rollout_fn, grad_f, gamma, H,
                                         B_inner, N_inner, beta)
    return omega, J_report, B_k + inner_traj


# -----------------------------------------------------------------------------
# REINFORCE baseline (prior-work eq. 38)
# -----------------------------------------------------------------------------

def reinforce_step(policy, rollout_fn: RolloutFn, obj, gamma, H, N, delta_guard=1e-3,
                   reward_transform=None):
    """REINFORCE with the per-step training reward:

        r_train(t) = - sum_m  t / ( sum_{tau=1}^{t} gamma^{tau-1} r_{m,tau} )

    This collapses the M-dim reward vector to a scalar per step, then runs
    plain REINFORCE (policy gradient theorem) on that scalar using reward-to-go.

    The numerator t is 1-indexed in the source equation; we use t = 0..H-1
    with gamma^t and apply (t+1) accordingly. delta_guard prevents division
    by zero on channels that have not yet accumulated reward.
    """
    ss, aa, rr = rollout_fn(H, N)  # (N, H, M)
    rr_train = reward_transform(rr) if reward_transform is not None else rr
    H_len = rr_train.shape[1]
    gammas = gamma ** np.arange(H_len)
    discounted = rr_train * gammas[None, :, None]
    cum_disc = np.cumsum(discounted, axis=1)

    t_vec = (np.arange(H_len) + 1).astype(np.float64)
    r_train = -(t_vec[None, :, None] / (cum_disc + delta_guard)).sum(axis=2)  # (N, H)

    r_train_disc = r_train * gammas[None, :]
    rtg = np.ascontiguousarray(np.cumsum(r_train_disc[:, ::-1], axis=1)[:, ::-1])

    scores = policy.scores(ss, aa)  # (N, H, d)
    rtg_t = torch.as_tensor(rtg, dtype=policy.dtype, device=policy.device)
    omega = (scores * rtg_t.unsqueeze(-1)).sum(dim=(0, 1)) / N

    J_hat = _discounted_returns(rr, gamma).mean(axis=0)
    return omega.detach().cpu().numpy(), J_hat, N
