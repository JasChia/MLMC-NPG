"""Episodic M-queue env — extends alt/envs_v2 with done flag and absorbing state.

The standard queueing env in src.envs / alt.envs_v2 is non-episodic (fixed
horizon H, no termination). Several MLMC mechanisms (sparse-reward bias
amplification) are masked because per-trajectory returns concentrate fast
under the dense per-step service-reward signal.

This file provides an *episodic* variant that terminates the episode on
the *first arrival event to the lowest-priority queue (queue 0)*. This
makes per-trajectory return very heavy-tailed: episode length is
geometric with rate lambda_0, so P(T = k) = lambda_0 * (1 - lambda_0)^{k-1},
giving E[T] = 1/lambda_0 and Var(T) = (1 - lambda_0)/lambda_0^2 -- both
massive when lambda_0 is small.

Other queue dynamics (service, other arrivals, reward) are unchanged.
After done, agents move to an absorbing sentinel state (q = -1 sentinel)
and emit zero rewards. The standard PG/MLMC math is unbiased on variable-
length episodic returns under this convention.

Reward layout matches base queueing: r[b, t, m] = service-success indicator.

env_config: {M, L_max, lambdas, T_max, q_init}
"""
from __future__ import annotations

from typing import Union, Callable, Optional

import numpy as np

from alt.envs_v2 import _resolve_q0, ARRIVAL_RATES


def episodic_queue_batch_rollout(env_config: dict, policy, H: int, B: int,
                                  rng: np.random.Generator,
                                  q_init=None):
    """Same dynamics as batch_rollout_v2, but terminates each trajectory on
    the first arrival to queue 0 (lowest-priority). After termination,
    states are zeroed and rewards are 0.

    Returns
    -------
    states  : (B, H, M) int32  (zeros after done)
    actions : (B, H)    int64  (zeros after done; action 0 is a valid no-op surrogate)
    rewards : (B, H, M) float32  (zeros after done)
    """
    M = env_config["M"]
    L_max = env_config["L_max"]
    lambdas = env_config["lambdas"]
    T_max = int(env_config.get("T_max", H))

    q = _resolve_q0(q_init, B, M, L_max, rng)
    states = np.zeros((B, H, M), dtype=np.int32)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, M), dtype=np.float32)
    done = np.zeros(B, dtype=bool)

    idx = np.arange(B)
    for t in range(H):
        cur_q = np.where(done[:, None], 0, q)
        states[:, t] = cur_q

        probs = policy.action_probs_batch(cur_q)
        u = rng.random(B)
        cum = probs.cumsum(axis=1)
        a = (u[:, None] < cum).argmax(axis=1)
        actions[:, t] = a

        active = ~done
        if active.any():
            served = active & (q[idx, a] > 0)
            rewards[served, t, a[served]] = 1.0
            q[idx[served], a[served]] -= 1

            arrivals = (rng.random((B, M)) < lambdas[None, :]).astype(np.int32)
            arrivals[~active] = 0
            q = np.minimum(q + arrivals, L_max)

            # Terminate: first arrival to queue 0 ends the episode for this traj.
            new_done = active & (arrivals[:, 0] > 0)
            done = done | new_done

        if t + 1 >= T_max:
            done[:] = True

    return states, actions, rewards
