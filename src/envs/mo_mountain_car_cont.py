"""Multi-Objective Mountain Car Continuous (mo-gymnasium spec).

Same physics as MO Mountain Car but with continuous action u in [-1, 1].
We discretize u into 5 levels {-1, -0.5, 0, 0.5, 1} so the existing
NNStatePolicy / TabularGridPolicy works unchanged. This is the standard
discretization used in classical control RL when comparing to discrete
algorithms.

Three reward channels (M=3):
    r[0] = -1 each active step                                  (time penalty)
    r[1] = -|u_continuous|^2 if u != 0  (otherwise 0)            (energy penalty)
    r[2] = -1 if action != neutral (any non-zero force)          (action-count)
All zero after goal or T_max.

Actions (5 discrete representing continuous force):
    0: u = -1.0    (full reverse)
    1: u = -0.5
    2: u =  0.0    (zero throttle)
    3: u = +0.5
    4: u = +1.0    (full forward)

Goal: position >= 0.45 (slightly more lenient than discrete MC's 0.5).
T_max = 200.
"""
from __future__ import annotations

import numpy as np


POS_LOW, POS_HIGH = -1.2, 0.6
VEL_LOW, VEL_HIGH = -0.07, 0.07
GOAL_POS = 0.45
INIT_POS_LOW, INIT_POS_HIGH = -0.6, -0.4

POWER = 0.0015
GRAVITY_FORCE = 0.0025
T_MAX = 200

# 5-level discretization of the continuous action space
ACTION_VALUES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
NUM_ACTIONS = 5

# State discretization (re-using MC values)
NUM_POS_BINS = 12
NUM_VEL_BINS = 16
NUM_REAL_STATES = NUM_POS_BINS * NUM_VEL_BINS
NUM_STATES = NUM_REAL_STATES + 1

M_REWARD = 3


def _discretize(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    p = np.clip(pos, POS_LOW, POS_HIGH - 1e-9)
    v = np.clip(vel, VEL_LOW, VEL_HIGH - 1e-9)
    pi = ((p - POS_LOW) / (POS_HIGH - POS_LOW) * NUM_POS_BINS).astype(np.int64)
    vi = ((v - VEL_LOW) / (VEL_HIGH - VEL_LOW) * NUM_VEL_BINS).astype(np.int64)
    pi = np.clip(pi, 0, NUM_POS_BINS - 1)
    vi = np.clip(vi, 0, NUM_VEL_BINS - 1)
    return pi * NUM_VEL_BINS + vi


def mc_cont_batch_rollout(env_config: dict, policy, H: int, B: int,
                           rng: np.random.Generator):
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
        a_idx = (u[:, None] < cum).argmax(axis=-1).astype(np.int64)
        actions[:, t] = a_idx

        active = ~done
        if active.any():
            u_cont = ACTION_VALUES[a_idx]
            force = u_cont * POWER
            vel_new = vel + force + np.cos(3.0 * pos) * (-GRAVITY_FORCE)
            vel_new = np.clip(vel_new, VEL_LOW, VEL_HIGH)
            pos_new = np.clip(pos + vel_new, POS_LOW, POS_HIGH)
            at_left = (pos_new <= POS_LOW + 1e-9) & (vel_new < 0)
            vel_new = np.where(at_left, 0.0, vel_new)
            pos = np.where(active, pos_new, pos)
            vel = np.where(active, vel_new, vel)

            r_active = np.where(active, 1.0, 0.0)
            rewards[:, t, 0] = -r_active                                     # time
            rewards[:, t, 1] = -(u_cont ** 2).astype(np.float32) * r_active  # energy
            rewards[:, t, 2] = -np.where((a_idx != 2) & active, 1.0, 0.0).astype(np.float32)  # action-count

            new_done = active & (pos >= GOAL_POS)
            done = done | new_done

        if t + 1 >= T_MAX:
            done[:] = True

    return states, actions, rewards


class MCContObj:
    """f(J) = sum_m sqrt(max(0, T_max + J_m) + delta).

    Since channel 1 (energy = -u^2) ranges in [-T_max, 0] (max -1 per
    step, T_max steps), all channels share the same shifted-positive
    structure. Maximizing f minimizes time, energy, and action-count.
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
