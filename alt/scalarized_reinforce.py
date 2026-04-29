"""Scalarized REINFORCE — best-effort episodic REINFORCE for concave MORL.

This is the standard MORL adaptation of single-objective REINFORCE: at each
outer iteration

  1. Roll out N trajectories, compute J_hat = (1/N) sum_i J(tau_i) (M-vector).
  2. Compute grad_f = obj.grad(J_hat) (M-vector). This is the linearization
     of f at the current J_hat — the standard concave-Frank-Wolfe direction.
  3. Build a per-step scalar reward r_scalar(t) = grad_f · r(t) and run
     vanilla PGT on it: omega = (1/N) sum_i sum_t score_{i,t} * rtg_scalar_{i,t}
     where rtg_scalar is the discounted reverse-cumsum of r_scalar.

Differences vs `src.algorithms.reinforce_step` (Paper A eq. (38)):
  - No t/(cum_disc + delta_guard) per-step surrogate; no divide-by-zero
    pathology on negative reward channels (DST time penalty).
  - Works on any reward sign; no `reward_transform` workaround needed.
  - Uses the SAME N trajectories for both J_hat and the gradient (one batch),
    distinguishing it from Bai-PG which uses N1 + N2 = 2N independent
    batches and gets the cleaner unbiased-conditional-on-J_hat estimator.

Differences vs `src.algorithms.bai_pg_step`:
  - Single batch instead of two => half the per-iter trajectory cost,
    introducing some bias from correlation between J_hat and the gradient
    on the same batch.
  - Otherwise identical gradient form — this is "Bai-PG with shared batch."

Outputs match the four-tuple convention: (omega, J_hat, n_traj).
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch


RolloutFn = Callable[[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]


def scalarized_reinforce_step(policy, rollout_fn: RolloutFn, obj, gamma, H, N):
    """One outer iteration of scalarized REINFORCE.

    Same trajectory budget per iter as paper-A REINFORCE: N trajectories.
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


def train_scalarized_reinforce(
    policy, rollout_fn: RolloutFn, obj, gamma: float, K: int, H: int,
    rng: np.random.Generator, lr: float, N: int, verbose: bool = False,
) -> dict:
    """Outer K-iteration loop with Adam ascent on `policy.theta`.

    Mirrors src.train.train but only supports scalarized REINFORCE so we
    don't have to touch src/train.py.
    """
    from src.train import AdamAscent

    opt = AdamAscent(policy.theta.shape, lr=lr,
                     device=policy.device, dtype=policy.dtype)
    history = {"iter": [], "J_hat": [], "f": [], "trajectories": []}
    total = 0
    for k in range(K):
        omega, J_hat, ntraj = scalarized_reinforce_step(
            policy, rollout_fn, obj, gamma, H, N,
        )
        opt.step(policy.theta, omega)
        total += ntraj
        history["iter"].append(k)
        history["J_hat"].append([float(x) for x in J_hat])
        history["f"].append(obj.value(J_hat))
        history["trajectories"].append(total)
        if verbose and (k % max(1, K // 20) == 0 or k == K - 1):
            print(f"  iter {k:4d} | f = {history['f'][-1]: .4f} | ntraj={total}")
    return history
