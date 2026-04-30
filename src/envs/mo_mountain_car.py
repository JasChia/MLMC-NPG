"""Multi-Objective Mountain Car env (mo-gymnasium spec).

Standard Mountain Car physics with discrete actions:
    0 = full backward (push left)
    1 = zero throttle
    2 = full forward (push right)

Continuous state (position, velocity) discretized onto a grid for the
existing one-hot-input policies (TabularGridPolicy, NNStatePolicy).

State encoding (default 12x16 grid = 192 cells, +1 absorbing sentinel = 193):
    state_idx = pos_bin * NUM_VEL_BINS + vel_bin

Three reward channels (M=3), all non-positive penalties to be MAXIMIZED
(i.e. their absolute count to be minimized):

    r[0] = -1 each active step (time penalty)
    r[1] = -1 if action == 2 (forward acceleration penalty)
    r[2] = -1 if action == 0 (backward / reversal penalty)
    -- all zero after the goal is reached or T_max steps elapse.

Episode terminates when position >= GOAL_POS (0.5) or after T_MAX steps.

Convention matches existing DST envs: post-done state is a sentinel idx
NUM_REAL_STATES, and rewards after done are zero so the score-times-RTG
product stays zero in the inner PG/MLMC math.
"""
from __future__ import annotations

import numpy as np


POS_LOW, POS_HIGH = -1.2, 0.6
VEL_LOW, VEL_HIGH = -0.07, 0.07
GOAL_POS = 0.5
INIT_POS_LOW, INIT_POS_HIGH = -0.6, -0.4   # standard mountain-car start range

FORCE = 0.001
GRAVITY = 0.0025
T_MAX = 200    # truncated relative to the 500 of the original to keep runs cheap

# Discretization defaults
NUM_POS_BINS = 12
NUM_VEL_BINS = 16
NUM_REAL_STATES = NUM_POS_BINS * NUM_VEL_BINS
NUM_STATES = NUM_REAL_STATES + 1
NUM_ACTIONS = 3
M_REWARD = 3


def _discretize(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    p = np.clip(pos, POS_LOW, POS_HIGH - 1e-9)
    v = np.clip(vel, VEL_LOW, VEL_HIGH - 1e-9)
    pi = ((p - POS_LOW) / (POS_HIGH - POS_LOW) * NUM_POS_BINS).astype(np.int64)
    vi = ((v - VEL_LOW) / (VEL_HIGH - VEL_LOW) * NUM_VEL_BINS).astype(np.int64)
    pi = np.clip(pi, 0, NUM_POS_BINS - 1)
    vi = np.clip(vi, 0, NUM_VEL_BINS - 1)
    return pi * NUM_VEL_BINS + vi


def mo_mountain_car_batch_rollout(env_config: dict, policy, H: int, B: int,
                                    rng: np.random.Generator):
    """Returns (states, actions, rewards) shapes (B,H), (B,H), (B,H,3)."""
    states = np.zeros((B, H), dtype=np.int64)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, M_REWARD), dtype=np.float32)

    pos = rng.uniform(INIT_POS_LOW, INIT_POS_HIGH, size=B).astype(np.float64)
    vel = np.zeros(B, dtype=np.float64)
    done = np.zeros(B, dtype=bool)

    for t in range(H):
        cur_state = np.where(done, NUM_REAL_STATES, _discretize(pos, vel)).astype(np.int64)
        states[:, t] = cur_state

        probs = policy.action_probs_batch(cur_state)
        u = rng.random(B)
        cum = np.cumsum(probs, axis=-1)
        a = (u[:, None] < cum).argmax(axis=-1).astype(np.int64)
        actions[:, t] = a

        active = ~done
        if active.any():
            # Mountain Car physics: vel += (action-1)*FORCE - cos(3*pos)*GRAVITY
            force_u = (a.astype(np.float64) - 1.0) * FORCE
            vel_new = vel + force_u + np.cos(3.0 * pos) * (-GRAVITY)
            vel_new = np.clip(vel_new, VEL_LOW, VEL_HIGH)
            pos_new = np.clip(pos + vel_new, POS_LOW, POS_HIGH)
            # If at left wall with negative velocity, snap velocity to 0
            at_left = (pos_new <= POS_LOW + 1e-9) & (vel_new < 0)
            vel_new = np.where(at_left, 0.0, vel_new)
            pos = np.where(active, pos_new, pos)
            vel = np.where(active, vel_new, vel)

            # Rewards (all -1, channel-specific)
            r_active = np.where(active, 1.0, 0.0)
            rewards[:, t, 0] = -r_active                          # time penalty
            rewards[:, t, 1] = -np.where((a == 2) & active, 1.0, 0.0)  # forward
            rewards[:, t, 2] = -np.where((a == 0) & active, 1.0, 0.0)  # reverse

            # Goal reached
            new_done = active & (pos >= GOAL_POS)
            done = done | new_done

        if t + 1 >= T_MAX:
            done[:] = True

    return states, actions, rewards


class MOMountainCarObj:
    """f(J) = sum_m sqrt(max(0, T_max + J_m) + delta).

    All three penalties are non-positive, so T_max + J_m ranges in [0, T_max]
    (similar to the DST time channel). Smaller |J_m| -> larger T_max + J_m
    -> larger f. So maximizing f minimizes the three counts.

    Compatible drop-in for objectives in src/objectives/.
    """
    def __init__(self, delta: float = 0.5, T_max: int = T_MAX):
        self.delta = float(delta)
        self.T_max = float(T_max)

    def value(self, J) -> float:
        J = np.asarray(J, dtype=np.float64)
        s = 0.0
        for m in range(len(J)):
            x = max(0.0, self.T_max + float(J[m]))
            s += float(np.sqrt(x + self.delta))
        return s

    def grad(self, J) -> np.ndarray:
        J = np.asarray(J, dtype=np.float64)
        g = np.zeros(len(J), dtype=np.float64)
        for m in range(len(J)):
            x = max(0.0, self.T_max + float(J[m]))
            g[m] = 0.5 / np.sqrt(x + self.delta)
        return g
