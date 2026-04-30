"""M-queue server environment (Paper A, Bai et al. 2022).

State: q in [0, L_max]^M (truncated queue lengths).
Action: choose queue to serve, in {0, ..., M-1}.
Dynamics (per step):
    1. If q[a] > 0: r[a] = 1, q[a] <- q[a] - 1.
    2. Arrivals: for each m, q[m] += Bernoulli(lambda_m), capped at L_max.

Reward is an M-dim vector; a scalarization (e.g. alpha-fairness) is applied
externally on the discounted return J.
"""
from __future__ import annotations

import numpy as np

# Paper A, Table 2 arrival rates.
ARRIVAL_RATES = {
    2: np.array([0.16, 0.64]),
    4: np.array([0.08, 0.16, 0.24, 0.32]),
    8: np.array([0.0125, 0.0375, 0.0625, 0.0875, 0.1125, 0.1375, 0.1625, 0.1875]),
}


class QueueingEnv:
    """Single-trajectory interface; see `batch_rollout` for the vectorized loop."""

    def __init__(self, M: int, arrival_rates=None, L_max: int = 10, seed=None):
        if arrival_rates is None:
            arrival_rates = ARRIVAL_RATES[M]
        self.M = M
        self.lambdas = np.asarray(arrival_rates, dtype=np.float64)
        assert self.lambdas.shape == (M,)
        assert self.lambdas.sum() < 1.0, "total arrival rate must be < 1 for stability"
        self.L_max = L_max
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros(M, dtype=np.int32)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.q = np.zeros(self.M, dtype=np.int32)
        return self.q.copy()

    def step(self, action: int):
        r = np.zeros(self.M, dtype=np.float32)
        if self.q[action] > 0:
            self.q[action] -= 1
            r[action] = 1.0
        arrivals = (self.rng.random(self.M) < self.lambdas).astype(np.int32)
        self.q = np.minimum(self.q + arrivals, self.L_max)
        return self.q.copy(), r, False, {}


def batch_rollout(env_config: dict, policy, H: int, B: int, rng: np.random.Generator):
    """Run B independent trajectories of length H in parallel under `policy`.

    Returns
    -------
    states  : (B, H, M) int32 — state observed at each step
    actions : (B, H)    int64 — action taken at each step
    rewards : (B, H, M) float32 — per-objective reward at each step
    """
    M = env_config["M"]
    L_max = env_config["L_max"]
    lambdas = env_config["lambdas"]

    q = np.zeros((B, M), dtype=np.int32)
    states = np.zeros((B, H, M), dtype=np.int32)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, M), dtype=np.float32)

    idx = np.arange(B)

    for t in range(H):
        states[:, t] = q

        probs = policy.action_probs_batch(q)  # (B, M)
        # Sample via inverse CDF
        u = rng.random(B)
        cum = probs.cumsum(axis=1)
        a = (u[:, None] < cum).argmax(axis=1)
        actions[:, t] = a

        # Service for rows where q[a] > 0
        served = q[idx, a] > 0
        rewards[served, t, a[served]] = 1.0
        q[idx[served], a[served]] -= 1

        # Arrivals (Bernoulli per queue), capped at L_max
        arrivals = (rng.random((B, M)) < lambdas[None, :]).astype(np.int32)
        q = np.minimum(q + arrivals, L_max)

    return states, actions, rewards
