"""Multi-Objective Cliff Walking.

Standard 4x12 cliff-walking grid. Start at bottom-left (3, 0), goal at
bottom-right (3, 11). Cliff cells (3, 1..10): stepping into them sends
the agent back to start (no episode termination on cliff fall — keeps
trying until goal or T_MAX).

4 actions: 0=up, 1=down, 2=right, 3=left. Walls clip movement.

Three reward channels (M=3, all penalties):
    r[0] = -1 each active step  (time penalty)
    r[1] = -100 if just fell off cliff this step (catastrophic)
    r[2] = -1 if action was "right" (encourages efficient pathfinding via up-around)

The point of r[2] is to give a non-trivial multi-objective trade-off:
the safe path (up + right + down) takes more steps but avoids cliffs.
"""
from __future__ import annotations

import numpy as np


GRID_ROWS = 4
GRID_COLS = 12
START = (3, 0)
GOAL = (3, 11)
T_MAX = 100
NUM_REAL_STATES = GRID_ROWS * GRID_COLS
NUM_STATES = NUM_REAL_STATES + 1
NUM_ACTIONS = 4
M_REWARD = 3

_DR = np.array([-1, +1, 0, 0], dtype=np.int32)
_DC = np.array([0, 0, +1, -1], dtype=np.int32)


def _is_cliff(row: np.ndarray, col: np.ndarray) -> np.ndarray:
    return (row == 3) & (col >= 1) & (col <= 10)


def cliff_walking_batch_rollout(env_config: dict, policy, H: int, B: int,
                                  rng: np.random.Generator):
    states = np.zeros((B, H), dtype=np.int64)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, M_REWARD), dtype=np.float32)

    row = np.full(B, START[0], dtype=np.int32)
    col = np.full(B, START[1], dtype=np.int32)
    done = np.zeros(B, dtype=bool)

    for t in range(H):
        cur_state = np.where(done, NUM_REAL_STATES,
                             row * GRID_COLS + col).astype(np.int64)
        states[:, t] = cur_state

        probs = policy.action_probs_batch(cur_state)
        u = rng.random(B)
        cum = np.cumsum(probs, axis=-1)
        a = (u[:, None] < cum).argmax(axis=-1).astype(np.int64)
        actions[:, t] = a

        active = ~done
        if active.any():
            new_row = np.clip(row + _DR[a], 0, GRID_ROWS - 1)
            new_col = np.clip(col + _DC[a], 0, GRID_COLS - 1)
            row = np.where(active, new_row, row)
            col = np.where(active, new_col, col)

            cliff_hit = active & _is_cliff(row, col)
            r_active = np.where(active, 1.0, 0.0)
            rewards[:, t, 0] = -r_active                         # time penalty
            rewards[:, t, 1] = -np.where(cliff_hit, 100.0, 0.0)  # cliff penalty
            rewards[:, t, 2] = -np.where((a == 2) & active, 1.0, 0.0)  # right-step penalty

            # Reset to start on cliff hit
            row = np.where(cliff_hit, START[0], row)
            col = np.where(cliff_hit, START[1], col)

            # Goal reached
            new_done = active & (row == GOAL[0]) & (col == GOAL[1])
            done = done | new_done

        if t + 1 >= T_MAX:
            done[:] = True

    return states, actions, rewards


class CliffWalkingObj:
    """f(J) = sqrt(max(0, T_max + J[0]) + delta)
              + sqrt(max(0, 50 + J[1]) + delta)
              + sqrt(max(0, T_max + J[2]) + delta).

    All channels are non-positive. T_max + J_m shifts to non-negative
    (cliff penalty channel uses 50 as cap since one cliff hit is -100;
    for a few cliff hits, J[1] in [-200, 0]). Smaller absolute values
    in each channel -> larger f. Maximizing f minimizes time, cliff
    falls, and right-steps.
    """
    def __init__(self, delta: float = 0.5, T_max: int = T_MAX):
        self.delta = float(delta)
        self.T_max = float(T_max)

    def value(self, J) -> float:
        J = np.asarray(J, dtype=np.float64)
        a = max(0.0, self.T_max + float(J[0]))
        b = max(0.0, 50.0 + float(J[1]))
        c = max(0.0, self.T_max + float(J[2]))
        return float(np.sqrt(a + self.delta) + np.sqrt(b + self.delta) + np.sqrt(c + self.delta))

    def grad(self, J) -> np.ndarray:
        J = np.asarray(J, dtype=np.float64)
        a = max(0.0, self.T_max + float(J[0]))
        b = max(0.0, 50.0 + float(J[1]))
        c = max(0.0, self.T_max + float(J[2]))
        g = np.zeros(3, dtype=np.float64)
        g[0] = 0.5 / np.sqrt(a + self.delta)
        g[1] = 0.5 / np.sqrt(b + self.delta)
        g[2] = 0.5 / np.sqrt(c + self.delta)
        return g
