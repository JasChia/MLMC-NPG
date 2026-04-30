"""Bonus World env (Vamplew, Dazeley, Berry, Issabekov, Dekker 2011,
"Empirical evaluation methods for multiobjective reinforcement learning
algorithms", Machine Learning 84:51-80).

9x9 grid. Agent starts at (0, 0). Two goal squares at (8, 0) and (0, 8) emit
respective objective rewards. A "bonus" pickup at (1, 7) doubles whichever
goal is reached next.

State encoding: (row, col, has_bonus). 9*9*2 = 162 internal states + 1 absorbing.
M=3 reward channels:
    r[0] = reward for goal-1 (at (8, 0)): 1 if reached without bonus, 2 if reached with bonus
    r[1] = reward for goal-2 (at (0, 8)): 1 if reached without bonus, 2 if reached with bonus
    r[2] = -1 each active step; 0 after done

Episode terminates on reaching either goal or T_max=50 steps. Once a goal is
reached, the agent is in the absorbing state.
"""
from __future__ import annotations

import numpy as np


GRID_ROWS = 9
GRID_COLS = 9
HOME = (0, 0)
GOAL1 = (8, 0)
GOAL2 = (0, 8)
BONUS = (1, 7)
T_MAX = 50
NUM_REAL_STATES = GRID_ROWS * GRID_COLS * 2  # *2 for has_bonus
NUM_STATES = NUM_REAL_STATES + 1
NUM_ACTIONS = 4

_DR = np.array([-1, +1,  0,  0], dtype=np.int32)
_DC = np.array([ 0,  0, +1, -1], dtype=np.int32)


def state_idx(row: int, col: int, has_bonus: bool) -> int:
    return (int(row) * GRID_COLS + int(col)) * 2 + (1 if has_bonus else 0)


def bonus_world_batch_rollout(env_config: dict, policy, H: int, B: int,
                               rng: np.random.Generator):
    """Returns (states, actions, rewards) with shapes (B, H), (B, H), (B, H, 3)."""
    states = np.zeros((B, H), dtype=np.int64)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, 3), dtype=np.float32)

    row = np.full(B, HOME[0], dtype=np.int32)
    col = np.full(B, HOME[1], dtype=np.int32)
    has_bonus = np.zeros(B, dtype=bool)
    done = np.zeros(B, dtype=bool)

    for t in range(H):
        cur_state = np.where(
            done, NUM_REAL_STATES,
            (row * GRID_COLS + col) * 2 + has_bonus.astype(np.int64)
        ).astype(np.int64)
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

            on_bonus = active & (row == BONUS[0]) & (col == BONUS[1]) & ~has_bonus
            has_bonus = has_bonus | on_bonus

            on_goal1 = active & (row == GOAL1[0]) & (col == GOAL1[1])
            on_goal2 = active & (row == GOAL2[0]) & (col == GOAL2[1])

            mult = np.where(has_bonus, 2.0, 1.0).astype(np.float32)
            rewards[:, t, 0] = np.where(on_goal1, mult, 0.0)
            rewards[:, t, 1] = np.where(on_goal2, mult, 0.0)
            rewards[:, t, 2] = np.where(active, -1.0, 0.0).astype(np.float32)

            new_done = active & (on_goal1 | on_goal2)
            done = done | new_done

        if t + 1 >= T_MAX:
            done[:] = True

    return states, actions, rewards


class BonusWorldObj:
    """f(J) = sqrt(J[0] + delta) + sqrt(J[1] + delta) + sqrt(max(0, T_max + J[2]) + delta).

    Encourages reaching either goal (one of the first two channels) and
    minimizing time. Concave fairness rewards balance: best to combine the
    bonus pickup with whichever goal is closer, but the policy must learn
    that the bonus also encourages variety in goal choice.
    """
    def __init__(self, delta: float = 0.5, T_max: int = T_MAX):
        self.delta = float(delta)
        self.T_max = float(T_max)

    def value(self, J) -> float:
        J = np.asarray(J, dtype=np.float64)
        a = max(0.0, float(J[0]))
        b = max(0.0, float(J[1]))
        c = max(0.0, self.T_max + float(J[2]))
        return float(np.sqrt(a + self.delta) + np.sqrt(b + self.delta) + np.sqrt(c + self.delta))

    def grad(self, J) -> np.ndarray:
        J = np.asarray(J, dtype=np.float64)
        a = max(0.0, float(J[0]))
        b = max(0.0, float(J[1]))
        c = max(0.0, self.T_max + float(J[2]))
        g = np.zeros(3, dtype=np.float64)
        g[0] = 0.5 / np.sqrt(a + self.delta)
        g[1] = 0.5 / np.sqrt(b + self.delta)
        g[2] = 0.5 / np.sqrt(c + self.delta)
        return g
