"""Four concave-MORL algorithms on a shared rollout/policy/objective interface.

bai_pg_step         — Paper A, Algorithm 1 (vanilla PG, biased plug-in estimator).
ganesh_npg_step     — Paper B, Algorithm 1 (vanilla NPG; valid when f is 2nd-order smooth).
ganesh_mlmc_step    — Paper B, Algorithm 2 (MLMC-NPG; no 2nd-order smoothness required).
reinforce_step      — Paper A baseline (eq. 38): REINFORCE with per-step r_train(t).

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
    """Compute   (1/B) sum_i sum_t score_{i,t} * [ grad_f · rtg_{i,t} ].

    Shared gradient estimator for Bai-PG (with grad_f = df(J_hat)) and both
    Ganesh inner loops (with a different grad_f but otherwise identical form).
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
    """Inner SGD loop per Paper B: omega_{n+1} = omega_n - beta (F_hat omega_n - g_hat).

    Fresh batch of B2 trajectories per inner step (per the theory).
    Returns (omega as numpy (d,), trajectories consumed).
    """
    d = policy.d
    omega = torch.zeros(d, device=policy.device, dtype=policy.dtype) #Line 7
    total_traj = 0
    for _ in range(N_inner): #Line 8
        states, actions, rewards = rollout_fn(H, B2) #Line 9
        total_traj += B2 #Reporting only
        g_hat = _pg_direction(policy, states, actions, rewards, gamma, grad_f) #Line 10
        scores = policy.scores(states, actions) #psi_{i,t}, shape (B2, H, d)
        F_hat = _empirical_fisher(scores, gamma) #Line 11
        omega = omega - beta * (F_hat @ omega - g_hat) #Line 12
    return omega.detach().cpu().numpy(), total_traj


# -----------------------------------------------------------------------------
# Paper A, Algorithm 1 — Vanilla PG
# -----------------------------------------------------------------------------

def bai_pg_step(policy, rollout_fn: RolloutFn, obj, gamma, H, N1, N2):
    """One outer iteration of Bai-Agarwal-Aggarwal (2022), Algorithm 1.

    Uses biased plug-in estimator: df(J_hat) with J_hat from an independent
    batch of N2 trajectories, then forms the gradient from another N1
    trajectories so that J_hat and omega are independent.
    """
    # Step 1: estimate J with N2 independent trajectories.
    _, _, rr2 = rollout_fn(H, N2)
    J_hat = _discounted_returns(rr2, gamma).mean(axis=0)  # (M,)
    grad_f = obj.grad(J_hat)

    # Step 2: estimate omega with N1 fresh trajectories.
    ss1, aa1, rr1 = rollout_fn(H, N1)
    omega = _pg_direction(policy, ss1, aa1, rr1, gamma, grad_f)
    return omega.detach().cpu().numpy(), J_hat, N1 + N2


# -----------------------------------------------------------------------------
# Paper B, Algorithm 1 — Vanilla NPG (requires 2nd-order-smooth f)
# -----------------------------------------------------------------------------

def ganesh_npg_step(policy, rollout_fn: RolloutFn, obj, gamma, H,
                    B1, B2, N_inner, beta):
    """One outer iteration of Ganesh-Aggarwal (2026), Algorithm 1.

    Differs from Bai-PG: instead of a single gradient step, solves the NPG
    direction via an inner SGD on the quadratic (1/2) omega^T F omega - omega^T g,
    using the empirical Fisher.
    """
    _, _, rr1 = rollout_fn(H, B1)
    J_hat = _discounted_returns(rr1, gamma).mean(axis=0)
    grad_f = obj.grad(J_hat)  # frozen across the inner loop

    omega, inner_traj = _solve_inner_npg(policy, rollout_fn, grad_f, gamma, H,
                                         B2, N_inner, beta)
    return omega, J_hat, B1 + inner_traj


# -----------------------------------------------------------------------------
# Paper B, Algorithm 2 — MLMC-NPG
# -----------------------------------------------------------------------------

def ganesh_mlmc_step(policy, rollout_fn: RolloutFn, obj, gamma, H,
                     B_max, B_inner, N_inner, beta, rng: np.random.Generator):
    """One outer iteration of Ganesh-Aggarwal (2026), Algorithm 2.

    MLMC return estimator:
        Q ~ Geom(1/2)
        if 2^Q > B_max : fall back to J_{H,1} (single trajectory)
        else:
            df(J_MLMC) = df(J_{H,1}) + 2^Q [df(J_{H,2^Q}) - df(J_{H, 2^{Q-1}})]
        with J_{H, 2^{Q-1}} using the FIRST HALF of the 2^Q trajectories
        that produced J_{H, 2^Q}  (the nested coupling).

    The nested coupling is what yields the variance bound in Paper B's Lemma 6;
    sampling J_{H, 2^{Q-1}} independently would make the estimator unbiased but
    blow up the variance (the telescoping argument breaks).
    """
    # Line 3
    Q = int(rng.geometric(0.5)) # Q in {1, 2, ...}, P(Q = q) = 2^{-q}
    # End Line 3
    # Line 4
    Q_max = int(np.floor(np.log2(B_max)))
    if Q > Q_max:
        Q_bar = 0
        B_k = 1
    else:
        Q_bar = Q
        B_k = 2 ** Q_bar
    # End Line 4
    # Line 5
    _, _, rr = rollout_fn(H, B_k)
    J_samples = _discounted_returns(rr, gamma)  # (B_k, M)
    # End Line 5
    # Line 6: Compute ∂_m f(Ĵ_{H,MLMC}) using eq. (17)
    # ∂_m f(Ĵ_{H,MLMC}) = ∂_m f(Ĵ_{H,1}) + 1{2^Q ≤ B_max} · 2^Q · [∂_m f(Ĵ_{H,2^Q}) − ∂_m f(Ĵ_{H,2^{Q-1}})]
    J_H1 = J_samples[0]                          # Ĵ_{H,1} — single-trajectory return

    if Q_bar == 0:                                # 1{2^Q ≤ B_max} = 0, correction term vanishes
        grad_f = obj.grad(J_H1)                   # = ∂_m f(Ĵ_{H,1})
        J_report = J_H1
    else:                                         # 1{2^Q ≤ B_max} = 1
        J_full = J_samples.mean(axis=0)           # Ĵ_{H,2^Q}
        J_half = J_samples[: B_k // 2].mean(axis=0)  # Ĵ_{H,2^{Q-1}} (nested coupling)
        # ∂f(Ĵ_{H,1})  +  2^Q · [∂f(Ĵ_{H,2^Q}) − ∂f(Ĵ_{H,2^{Q-1}})]
        grad_f = obj.grad(J_H1) + (2 ** Q_bar) * (obj.grad(J_full) - obj.grad(J_half))
        J_report = J_full
    # End Line 6
    # Lines 7-14: Inner NPG loop (solved in _solve_inner_npg)
    # Line 7:  ω_0 = 0
    # Line 8:  for n = 0, ..., N-1
    # Line 9:    Sample B trajectories
    # Line 10:   ĝ = (1/B) Σ g(τ, Ĵ_{H,MLMC} | θ_k)
    # Line 11:   F̂ = (1/B) Σ Σ_t γ^t ψ_t ψ_t⊤
    # Line 12:   ω_{n+1} = ω_n − β(F̂ ω_n − ĝ)
    # Line 13: end for
    # Line 14: ω_k = ω_N
    omega, inner_traj = _solve_inner_npg(policy, rollout_fn, grad_f, gamma, H,
                                         B_inner, N_inner, beta)
    # Line 15: θ_{k+1} = θ_k + α·ω_k  (applied by caller)
    return omega, J_report, B_k + inner_traj


# -----------------------------------------------------------------------------
# REINFORCE baseline — Paper A eq. (38)
# -----------------------------------------------------------------------------

def reinforce_step(policy, rollout_fn: RolloutFn, obj, gamma, H, N, delta_guard=1e-3,
                   reward_transform=None):
    """REINFORCE with the per-step training reward from Paper A eq. (38):

        r_train(t) = - sum_m  t / ( sum_{tau=1}^{t} gamma^{tau-1} r_{m,tau} )

    This collapses the M-dim reward vector to a scalar per step, then runs
    plain REINFORCE (policy gradient theorem) on that scalar using reward-to-go
    with an H-weighting consistent with Paper A eq. (37).

    Notes on indexing:
      - Paper uses t = 1..H with gamma^{t-1}; we use t = 0..H-1 with gamma^t.
        The numerator `t` in eq. (38) corresponds to (t+1) in 0-indexed time.
      - A small delta guards against division by zero when a queue has no
        reward yet.
    """
    ss, aa, rr = rollout_fn(H, N)  # (N, H, M)
    rr_train = reward_transform(rr) if reward_transform is not None else rr
    H_len = rr_train.shape[1]
    gammas = gamma ** np.arange(H_len)  # (H,)
    # cum_disc[b, t, m] = sum_{tau=0}^{t} gamma^tau r[b, tau, m]
    discounted = rr_train * gammas[None, :, None]
    cum_disc = np.cumsum(discounted, axis=1)

    # Paper's numerator t (1-indexed) == (t+1) in 0-indexed.
    t_vec = (np.arange(H_len) + 1).astype(np.float64)  # (H,)
    r_train = -(t_vec[None, :, None] / (cum_disc + delta_guard)).sum(axis=2)  # (N, H)

    # Reward-to-go in the original discount gamma: R_t = sum_{h>=t} gamma^h r_train(h)
    r_train_disc = r_train * gammas[None, :]
    # np.ascontiguousarray strips the negative stride so torch.as_tensor works.
    rtg = np.ascontiguousarray(np.cumsum(r_train_disc[:, ::-1], axis=1)[:, ::-1])  # (N, H)

    scores = policy.scores(ss, aa)  # (N, H, d)
    rtg_t = torch.as_tensor(rtg, dtype=policy.dtype, device=policy.device)
    omega = (scores * rtg_t.unsqueeze(-1)).sum(dim=(0, 1)) / N

    # For logging, report the paper-scaled f(J_hat):
    J_hat = _discounted_returns(rr, gamma).mean(axis=0)
    return omega.detach().cpu().numpy(), J_hat, N


# -----------------------------------------------------------------------------
# Scalarized REINFORCE — best-effort episodic baseline for concave MORL
# -----------------------------------------------------------------------------

def scalarized_reinforce_step(policy, rollout_fn: RolloutFn, obj, gamma, H, N):
    """One outer iteration of scalarized REINFORCE.

    Standard MORL adaptation of single-objective REINFORCE:
      1. Roll out N trajectories, compute J_hat = (1/N) sum_i J(tau_i) (M-vector).
      2. Compute grad_f = obj.grad(J_hat) (M-vector). This is the linearization
         of f at the current J_hat — the standard concave-Frank-Wolfe direction.
      3. Build a per-step scalar reward r_scalar(t) = grad_f . r(t) and run
         vanilla PGT on it: omega = (1/N) sum_i sum_t score_{i,t} * rtg_scalar_{i,t}
         where rtg_scalar is the discounted reverse-cumsum of r_scalar.

    Differences vs reinforce_step (Paper A eq. (38)):
      - No t/(cum_disc + delta_guard) per-step surrogate; no divide-by-zero
        pathology on negative reward channels (DST time penalty).
      - Works on any reward sign; no `reward_transform` workaround needed.
      - Uses the SAME N trajectories for both J_hat and the gradient (one batch),
        distinguishing it from Bai-PG which uses N1 + N2 = 2N independent batches
        and gets the cleaner unbiased-conditional-on-J_hat estimator.

    Differences vs bai_pg_step:
      - Single batch instead of two => half the per-iter trajectory cost,
        introducing some bias from correlation between J_hat and the gradient
        on the same batch.
      - Otherwise identical gradient form — this is "Bai-PG with shared batch."

    Per-iter trajectory cost: N (matches reinforce_step).
    """
    ss, aa, rr = rollout_fn(H, N)  # (N, H, M)
    H_len = rr.shape[1]
    gammas = gamma ** np.arange(H_len)  # (H,)

    # Discounted return per trajectory (N, M); mean is plug-in J_hat.
    discounted = rr * gammas[None, :, None]
    J_per = discounted.sum(axis=1)
    J_hat = J_per.mean(axis=0)
    grad_f = obj.grad(J_hat)  # (M,)

    # Per-step scalarized reward r_scalar(t) = grad_f . r(t)  shape (N, H).
    r_scalar = (rr * grad_f[None, None, :]).sum(axis=-1)
    # Discounted then PGT-style reverse cumsum.
    r_scalar_disc = r_scalar * gammas[None, :]
    rtg = np.ascontiguousarray(np.cumsum(r_scalar_disc[:, ::-1], axis=1)[:, ::-1])

    scores = policy.scores(ss, aa)  # (N, H, d)
    rtg_t = torch.as_tensor(rtg, dtype=policy.dtype, device=policy.device)
    omega = (scores * rtg_t.unsqueeze(-1)).sum(dim=(0, 1)) / N
    return omega.detach().cpu().numpy(), J_hat, N
