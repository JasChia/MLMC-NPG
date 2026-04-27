"""M-queue env with configurable initial-queue-state distribution rho.

Root cause of paper-vs-ours f-value gap: the original src/envs.py hardcodes
q_0 = 0 at every trajectory reset.  Paper A does not specify its initial
distribution, but its reported f values for M in {2,4,8} at optimality are
consistent with q_0 ≈ 4 customers per queue (see analysis in the report).
Starting with empty queues strictly reduces J_m (fewer customers available
to serve during the episode), which makes f more negative.

This file provides `batch_rollout_v2`, drop-in compatible with
`src.envs.batch_rollout` but taking a `q_init` spec:

    None            -> zeros (legacy behaviour, matches original paper-A-mismatched setup)
    int scalar      -> every queue starts at this value
    ndarray (M,)    -> per-queue fixed initial level
    'uniform'       -> q_init[b, m] ~ Uniform({0, ..., L_max}) per trajectory
    callable        -> q_init(B, M, rng) -> (B, M) array
"""
from __future__ import annotations

from typing import Union, Callable, Optional

import numpy as np

from src.envs import ARRIVAL_RATES as _BASE_ARRIVAL_RATES

# Extend ARRIVAL_RATES with M=16 for the purpose-built MLMC showcase.
# Pattern: lambda_m = (m + 0.5) / 256 for m in 0..15. Sigma_lambda = 0.5 (stable;
# matches paper's "load <= 1" requirement). Smallest lambda = 0.00195, which is
# ~6.4x smaller than the M=8 baseline (0.0125), giving ~6x larger 2nd-derivative
# of any alpha-fairness objective at the smallest queue.
ARRIVAL_RATES = dict(_BASE_ARRIVAL_RATES)
ARRIVAL_RATES[16] = np.array([(m + 0.5) / 256.0 for m in range(16)], dtype=np.float64)
# M=32: lambdas in [0.000488, 0.0308], sum = 0.5 (stable). lambda_min is 25.6x smaller
# than M=8's, giving a much larger curvature gap between the smallest and largest queue.
ARRIVAL_RATES[32] = np.array([(m + 0.5) / 1024.0 for m in range(32)], dtype=np.float64)


QInit = Union[None, int, np.ndarray, str, Callable]


def _resolve_q0(q_init: QInit, B: int, M: int, L_max: int,
                rng: np.random.Generator) -> np.ndarray:
    if q_init is None:
        return np.zeros((B, M), dtype=np.int32)
    if isinstance(q_init, (int, np.integer)):
        return np.full((B, M), int(q_init), dtype=np.int32)
    if isinstance(q_init, str):
        if q_init == "uniform":
            return rng.integers(0, L_max + 1, size=(B, M), dtype=np.int32)
        raise ValueError(f"Unknown q_init spec: {q_init}")
    if callable(q_init):
        out = np.asarray(q_init(B, M, rng), dtype=np.int32)
        if out.shape != (B, M):
            raise ValueError(f"q_init callable returned {out.shape}, expected {(B, M)}")
        return out
    arr = np.asarray(q_init, dtype=np.int32)
    if arr.shape == (M,):
        return np.broadcast_to(arr, (B, M)).copy()
    if arr.shape == (B, M):
        return arr.copy()
    raise ValueError(f"q_init ndarray has shape {arr.shape}, expected ({M},) or ({B},{M})")


def batch_rollout_v2(env_config: dict, policy, H: int, B: int,
                     rng: np.random.Generator,
                     q_init: QInit = None):
    """Same dynamics as src.envs.batch_rollout, but with settable q_init.

    Returns
    -------
    states  : (B, H, M) int32
    actions : (B, H)    int64
    rewards : (B, H, M) float32
    """
    M = env_config["M"]
    L_max = env_config["L_max"]
    lambdas = env_config["lambdas"]

    q = _resolve_q0(q_init, B, M, L_max, rng)
    states = np.zeros((B, H, M), dtype=np.int32)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, M), dtype=np.float32)

    idx = np.arange(B)
    for t in range(H):
        states[:, t] = q
        probs = policy.action_probs_batch(q)
        u = rng.random(B)
        cum = probs.cumsum(axis=1)
        a = (u[:, None] < cum).argmax(axis=1)
        actions[:, t] = a

        served = q[idx, a] > 0
        rewards[served, t, a[served]] = 1.0
        q[idx[served], a[served]] -= 1

        arrivals = (rng.random((B, M)) < lambdas[None, :]).astype(np.int32)
        q = np.minimum(q + arrivals, L_max)

    return states, actions, rewards
