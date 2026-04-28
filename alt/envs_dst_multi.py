"""Multi-treasure Deep Sea Treasure: M=6 reward channels for per-treasure visit
rates plus a step-cost channel. Same 10x5 grid as DST but the objective is
sqrt-fairness over per-treasure visit rates — encourages the policy to be
stochastic enough to visit *all* treasures, not just the deepest one.

This is a more challenging concave-MORL problem than the original DST: bias
in any single J_m hurts the scalar objective via 1/sqrt(J_m+delta), so the
plug-in bias correction has 5x the leverage compared to DST's M=2.

States/dynamics identical to alt/envs_dst.py. Reward layout differs:
    r[0..4] = 1 if just picked treasure k, else 0
    r[5]    = -1 each active step; 0 after done
"""
from __future__ import annotations

import numpy as np


DST_TREASURES = [
    (1, 0),
    (2, 1),
    (4, 2),
    (7, 3),
    (9, 4),
]
DST_GRID_ROWS = 10
DST_GRID_COLS = 5
DST_T_MAX = 50
NUM_REAL_STATES = DST_GRID_ROWS * DST_GRID_COLS
NUM_STATES = NUM_REAL_STATES + 1
NUM_ACTIONS = 4
NUM_TREASURES = len(DST_TREASURES)
M_REWARD = NUM_TREASURES + 1  # 5 treasure flags + 1 step penalty


_DR = np.array([-1, +1,  0,  0], dtype=np.int32)
_DC = np.array([ 0,  0, +1, -1], dtype=np.int32)


def dst_multi_batch_rollout(env_config: dict, policy, H: int, B: int,
                             rng: np.random.Generator):
    """Returns
        states  : (B, H) int64
        actions : (B, H) int64
        rewards : (B, H, 6) float32
    """
    states = np.zeros((B, H), dtype=np.int64)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, M_REWARD), dtype=np.float32)

    row = np.zeros(B, dtype=np.int32)
    col = np.zeros(B, dtype=np.int32)
    done = np.zeros(B, dtype=bool)

    for t in range(H):
        cur_state = np.where(done, NUM_REAL_STATES, row * DST_GRID_COLS + col).astype(np.int64)
        states[:, t] = cur_state

        probs = policy.action_probs_batch(cur_state)
        u = rng.random(B)
        cum = np.cumsum(probs, axis=-1)
        a = (u[:, None] < cum).argmax(axis=-1).astype(np.int64)
        actions[:, t] = a

        active = ~done
        if active.any():
            new_row = np.clip(row + _DR[a], 0, DST_GRID_ROWS - 1)
            new_col = np.clip(col + _DC[a], 0, DST_GRID_COLS - 1)
            row = np.where(active, new_row, row)
            col = np.where(active, new_col, col)

            for k, (rr, cc) in enumerate(DST_TREASURES):
                hit = active & (row == rr) & (col == cc)
                rewards[hit, t, k] = 1.0
                done = done | hit

            rewards[:, t, M_REWARD - 1] = np.where(active, -1.0, 0.0).astype(np.float32)

        if t + 1 >= DST_T_MAX:
            done[:] = True

    return states, actions, rewards


class DSTMultiObj:
    """f(J) = sum_{k=0..4} sqrt(J_k + delta) + sqrt(max(0, T_max + J[5]) + delta).

    Encourages visiting *all* treasures (per the concave-fairness shape) rather
    than locking onto whichever one is easiest. Step-cost dimension is treated
    DST-style.
    """
    def __init__(self, delta: float = 0.5, T_max: int = DST_T_MAX):
        self.delta = float(delta)
        self.T_max = float(T_max)
        self._n_treas = NUM_TREASURES

    def value(self, J) -> float:
        J = np.asarray(J, dtype=np.float64)
        s = 0.0
        for k in range(self._n_treas):
            s += float(np.sqrt(max(0.0, float(J[k])) + self.delta))
        s += float(np.sqrt(max(0.0, self.T_max + float(J[self._n_treas])) + self.delta))
        return s

    def grad(self, J) -> np.ndarray:
        J = np.asarray(J, dtype=np.float64)
        g = np.zeros(M_REWARD, dtype=np.float64)
        for k in range(self._n_treas):
            jk = max(0.0, float(J[k]))
            g[k] = 0.5 / np.sqrt(jk + self.delta)
        c = max(0.0, self.T_max + float(J[self._n_treas]))
        g[self._n_treas] = 0.5 / np.sqrt(c + self.delta)
        return g
