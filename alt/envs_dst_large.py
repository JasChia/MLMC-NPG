"""Larger Deep Sea Treasure: 20x10 grid with 10 treasures along a diagonal,
giving 4x more states (200 vs 50) for the tabular policy and a longer
T_max=100 horizon. Same M=2 reward shape as the original DST and same
sqrt-fairness scalarization.

Tabular d = 201*4 = 804 (vs the original DST's 204) — exercises mechanism 2
(Fisher preconditioning) without changing the reward sparsity.
"""
from __future__ import annotations

import numpy as np


# 10 treasures along increasing depth columns. Values follow a sublinear
# growth so deeper treasures are clearly more valuable but not by orders
# of magnitude (keeps the gradient scale comparable across columns).
LARGE_DST_TREASURES = {
    (1, 0): 1.0,
    (2, 1): 2.0,
    (3, 2): 3.0,
    (5, 3): 6.0,
    (8, 4): 14.0,
    (10, 5): 26.0,
    (12, 6): 43.0,
    (14, 7): 63.0,
    (16, 8): 85.0,
    (19, 9): 124.0,
}
GRID_ROWS = 20
GRID_COLS = 10
T_MAX = 100
NUM_REAL_STATES = GRID_ROWS * GRID_COLS
NUM_STATES = NUM_REAL_STATES + 1
NUM_ACTIONS = 4

_DR = np.array([-1, +1,  0,  0], dtype=np.int32)
_DC = np.array([ 0,  0, +1, -1], dtype=np.int32)


def large_dst_batch_rollout(env_config: dict, policy, H: int, B: int,
                             rng: np.random.Generator):
    states = np.zeros((B, H), dtype=np.int64)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, 2), dtype=np.float32)

    row = np.zeros(B, dtype=np.int32)
    col = np.zeros(B, dtype=np.int32)
    done = np.zeros(B, dtype=bool)

    for t in range(H):
        cur_state = np.where(done, NUM_REAL_STATES, row * GRID_COLS + col).astype(np.int64)
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

            tr_vals = np.zeros(B, dtype=np.float32)
            for (rr, cc), v in LARGE_DST_TREASURES.items():
                hit = active & (row == rr) & (col == cc)
                tr_vals[hit] = v
            rewards[:, t, 0] = tr_vals
            rewards[:, t, 1] = np.where(active, -1.0, 0.0).astype(np.float32)

            new_done = tr_vals > 0
            done = done | new_done

        if t + 1 >= T_MAX:
            done[:] = True

    return states, actions, rewards


class LargeDSTObj:
    """f(J) = sqrt(J[0] + delta) + sqrt(max(0, T_max + J[1]) + delta).

    Same shape as DSTreasureObj but with T_max=100 to match the longer grid.
    """
    def __init__(self, delta: float = 0.5, T_max: int = T_MAX):
        self.delta = float(delta)
        self.T_max = float(T_max)

    def value(self, J) -> float:
        J = np.asarray(J, dtype=np.float64)
        a = max(0.0, float(J[0]))
        c = max(0.0, self.T_max + float(J[1]))
        return float(np.sqrt(a + self.delta) + np.sqrt(c + self.delta))

    def grad(self, J) -> np.ndarray:
        J = np.asarray(J, dtype=np.float64)
        a = max(0.0, float(J[0]))
        c = max(0.0, self.T_max + float(J[1]))
        g = np.zeros(2, dtype=np.float64)
        g[0] = 0.5 / np.sqrt(a + self.delta)
        g[1] = 0.5 / np.sqrt(c + self.delta)
        return g
