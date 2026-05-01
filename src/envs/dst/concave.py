"""Concave Deep Sea Treasure (mo-gymnasium deep-sea-treasure-concave-v0).

Same dynamics as the standard 10x11 Vamplew DST, but treasure values are
chosen so the Pareto front is concave (vs the convex front of the
standard DST). Concave Pareto fronts are challenging for linear
scalarization — only nonlinear concave scalarization can recover the
interior points.

Treasure positions (row, col) and values (mo-gymnasium spec):
    (1, 0): 0.7,  (2, 1): 8.2,  (3, 2): 11.5,
    (4, 3): 14.0, (4, 4): 15.1, (4, 5): 16.1,
    (7, 6): 19.6, (7, 7): 20.3,
    (8, 8): 22.4,
    (9, 9): 23.7

Reward shape (M=2): r[0] = treasure_value if hit, r[1] = -1 active step.
Sea floor masking and bounce-back logic identical to the original DST.
"""
from __future__ import annotations

import numpy as np


CONCAVE_DST_TREASURES = {
    (1, 0): 0.7,
    (2, 1): 8.2,
    (3, 2): 11.5,
    (4, 3): 14.0,
    (4, 4): 15.1,
    (4, 5): 16.1,
    (7, 6): 19.6,
    (7, 7): 20.3,
    (8, 8): 22.4,
    (9, 9): 23.7,
}

GRID_ROWS = 10
GRID_COLS = 11
T_MAX_SPEC = 1000
T_MAX = 200
NUM_REAL_STATES = GRID_ROWS * GRID_COLS
NUM_STATES = NUM_REAL_STATES + 1
NUM_ACTIONS = 4


def _build_passable_mask() -> np.ndarray:
    deepest_in_col: dict = {}
    for (r, c) in CONCAVE_DST_TREASURES.keys():
        deepest_in_col[c] = max(deepest_in_col.get(c, -1), r)
    mask = np.ones((GRID_ROWS, GRID_COLS), dtype=bool)
    for c, deepest_r in deepest_in_col.items():
        for r in range(deepest_r + 1, GRID_ROWS):
            mask[r, c] = False
    return mask


PASSABLE = _build_passable_mask()
_DR = np.array([-1, +1, 0, 0], dtype=np.int32)
_DC = np.array([0, 0, +1, -1], dtype=np.int32)


def concave_dst_batch_rollout(env_config: dict, policy, H: int, B: int,
                                rng: np.random.Generator):
    states = np.zeros((B, H), dtype=np.int64)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, 2), dtype=np.float32)

    row = np.zeros(B, dtype=np.int32)
    col = np.zeros(B, dtype=np.int32)
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
            prop_row = np.clip(row + _DR[a], 0, GRID_ROWS - 1)
            prop_col = np.clip(col + _DC[a], 0, GRID_COLS - 1)
            allowed = PASSABLE[prop_row, prop_col]
            new_row = np.where(allowed, prop_row, row)
            new_col = np.where(allowed, prop_col, col)
            row = np.where(active, new_row, row)
            col = np.where(active, new_col, col)

            tr_vals = np.zeros(B, dtype=np.float32)
            for (rr, cc), v in CONCAVE_DST_TREASURES.items():
                hit = active & (row == rr) & (col == cc)
                tr_vals[hit] = v
            rewards[:, t, 0] = tr_vals
            rewards[:, t, 1] = np.where(active, -1.0, 0.0).astype(np.float32)
            done = done | (tr_vals > 0)

        if t + 1 >= T_MAX_SPEC:
            done[:] = True

    return states, actions, rewards


class ConcaveDSTObj:
    """f(J) = sqrt(J[0] + delta) + sqrt(max(0, T_max + J[1]) + delta).

    Same form as DSTreasureObj/OriginalDSTObj. T_max=200 matches the
    rollout cap; treasure scale ~24 (smaller than original DST's 124).
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
