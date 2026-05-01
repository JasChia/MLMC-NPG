"""Simplified Multi-Objective Lunar Lander (mo-gymnasium-inspired).

The full mo-gymnasium lunar-lander uses Box2D physics with rotation,
legs, and irregular terrain. We implement a simplified planar version
that captures the multi-objective spirit without the full physics:

State (4D): (x, y, vx, vy)
    x  in [-1, 1]   (horizontal position; landing pad at x=0)
    y  in [0, 1]    (altitude; goal y=0)
    vx in [-1, 1]   (clipped)
    vy in [-1, 1]   (clipped)

4 discrete actions:
    0: do nothing
    1: fire left engine    (force +x, lateral fuel)
    2: fire main engine    (force -y opposing gravity, main fuel)
    3: fire right engine   (force -x, lateral fuel)

4 reward channels (M=4, all penalty):
    r[0] = -1 each active step                        (time / shaping)
    r[1] = -1 if action == 2 (main-engine fuel)       (main fuel)
    r[2] = -1 if action in {1, 3} (lateral-engine)    (lateral fuel)
    r[3] = -1 if not soft-landed at goal              (landing penalty per step)

Terminate on landing (y <= 0 AND |vx|+|vy| < 0.2 AND |x| < 0.3) or
T_max=100 steps. Crash if y <= 0 with velocity above threshold (no
explicit penalty channel; just terminate without success).

State discretization: 5x4x4x4 = 320 cells + 1 sentinel = 321 states.
"""
from __future__ import annotations

import numpy as np


X_LOW, X_HIGH = -1.0, 1.0
Y_LOW, Y_HIGH = 0.0, 1.0
V_LOW, V_HIGH = -1.0, 1.0

GRAVITY = 0.02
MAIN_FORCE = 0.04
SIDE_FORCE = 0.02
DT = 1.0
T_MAX = 100

# Discretization
NUM_X_BINS = 5
NUM_Y_BINS = 4
NUM_VX_BINS = 4
NUM_VY_BINS = 4
NUM_REAL_STATES = NUM_X_BINS * NUM_Y_BINS * NUM_VX_BINS * NUM_VY_BINS
NUM_STATES = NUM_REAL_STATES + 1
NUM_ACTIONS = 4
M_REWARD = 4


def _discretize(x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    xi = np.clip(((x - X_LOW) / (X_HIGH - X_LOW) * NUM_X_BINS).astype(np.int64), 0, NUM_X_BINS - 1)
    yi = np.clip(((y - Y_LOW) / (Y_HIGH - Y_LOW) * NUM_Y_BINS).astype(np.int64), 0, NUM_Y_BINS - 1)
    vxi = np.clip(((vx - V_LOW) / (V_HIGH - V_LOW) * NUM_VX_BINS).astype(np.int64), 0, NUM_VX_BINS - 1)
    vyi = np.clip(((vy - V_LOW) / (V_HIGH - V_LOW) * NUM_VY_BINS).astype(np.int64), 0, NUM_VY_BINS - 1)
    return ((xi * NUM_Y_BINS + yi) * NUM_VX_BINS + vxi) * NUM_VY_BINS + vyi


def lunar_lander_batch_rollout(env_config: dict, policy, H: int, B: int,
                                 rng: np.random.Generator):
    states = np.zeros((B, H), dtype=np.int64)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, M_REWARD), dtype=np.float32)

    # Initial conditions: hover above pad with small random offset
    x = rng.uniform(-0.4, 0.4, size=B).astype(np.float64)
    y = np.full(B, 0.9, dtype=np.float64)
    vx = np.zeros(B, dtype=np.float64)
    vy = np.zeros(B, dtype=np.float64)
    done = np.zeros(B, dtype=bool)

    for t in range(H):
        cur_state = np.where(done, NUM_REAL_STATES, _discretize(x, y, vx, vy)).astype(np.int64)
        states[:, t] = cur_state

        probs = policy.action_probs_batch(cur_state)
        u = rng.random(B)
        cum = np.cumsum(probs, axis=-1)
        a = (u[:, None] < cum).argmax(axis=-1).astype(np.int64)
        actions[:, t] = a

        active = ~done
        if active.any():
            # Apply forces from action
            ax = np.where(a == 1,  SIDE_FORCE, np.where(a == 3, -SIDE_FORCE, 0.0))
            ay = np.where(a == 2,  MAIN_FORCE, 0.0) - GRAVITY
            # Update velocities and positions
            vx_new = np.clip(vx + ax * DT, V_LOW, V_HIGH)
            vy_new = np.clip(vy + ay * DT, V_LOW, V_HIGH)
            x_new = np.clip(x + vx_new * DT, X_LOW, X_HIGH)
            y_new = np.clip(y + vy_new * DT, Y_LOW, Y_HIGH)
            x = np.where(active, x_new, x)
            y = np.where(active, y_new, y)
            vx = np.where(active, vx_new, vx)
            vy = np.where(active, vy_new, vy)

            # Soft landing detection
            soft_land = active & (y <= 0.05) & (np.abs(vx) + np.abs(vy) < 0.2) & (np.abs(x) < 0.3)
            crash = active & (y <= 0.0) & ~soft_land

            r_active = np.where(active, 1.0, 0.0)
            rewards[:, t, 0] = -r_active                                       # time
            rewards[:, t, 1] = -np.where((a == 2) & active, 1.0, 0.0).astype(np.float32)  # main fuel
            rewards[:, t, 2] = -np.where(((a == 1) | (a == 3)) & active, 1.0, 0.0).astype(np.float32)  # lateral fuel
            # landing penalty: -1 per step until soft-landed
            rewards[:, t, 3] = -np.where(active & ~soft_land, 1.0, 0.0).astype(np.float32)

            done = done | soft_land | crash

        if t + 1 >= T_MAX:
            done[:] = True

    return states, actions, rewards


class MOLunarLanderObj:
    """f(J) = sum_m sqrt(max(0, T_max + J_m) + delta).

    All 4 channels are non-positive penalties. T_max + J_m shifts
    each into [0, T_max]. Maximizing f minimizes time, main fuel,
    lateral fuel, and unlanded-step count.
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
