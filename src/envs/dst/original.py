"""Original Deep Sea Treasure (Vamplew, Dazeley, Berry, Issabekov, Dekker 2011),
faithful to the Fig. 10 specification used in Vamplew's MORL benchmark suite.

Grid: 10 rows x 11 columns. Submarine starts at (0, 0). Four actions
(up/down/right/left), with two restrictions:
  1. Any move that would leave the grid leaves the agent in place (per spec).
  2. The "sea floor" cells (the black region in Vamplew's Fig. 10) are
     impassable — a move into a sea-floor cell leaves the agent in place.
     Note: the user-quoted spec only mentions the grid edges, but the
     canonical Vamplew implementation (and every reproduction in
     mo-gymnasium / MORL-baselines / Yang 2019) treats the sea floor as
     a wall. Without it, the deepest treasure is trivially reachable in
     ~9 steps via the bottom row, and the 10-point Pareto front collapses.

Ten treasures along a descending staircase (row, col):
    (1, 0)=1, (2, 1)=2, (3, 2)=3,
    (4, 3)=5, (4, 4)=8, (4, 5)=16,
    (7, 6)=24, (7, 7)=50,
    (8, 8)=74,
    (9, 9)=124.

Reward vector per step (M=2):
    r[0] = treasure_value if at_treasure_cell else 0
    r[1] = -1 each active step; 0 after done

Episode terminates when the agent steps onto any treasure cell, or after
T_MAX actions. The spec sets T_MAX=1000 as a safety cap; we expose it as
T_MAX_SPEC and additionally define T_MAX=200 (>> any plausible optimal
path of <=18 steps) as the rollout horizon used by the runner — this
keeps (B, H, M) arrays small while remaining functionally equivalent for
any policy that ever reaches a treasure. Set H=T_MAX_SPEC to match the
spec exactly.

Done-padding follows the same convention as alt/envs_dst.py: post-done
state = absorbing sentinel (= NUM_REAL_STATES), and rewards after done
are zero, so the score x RTG product is zero and the gradient estimator
is unbiased w.r.t. the true variable-length episodic return.
"""
from __future__ import annotations

import numpy as np


# Treasure positions (row, col) and values, per Vamplew 2011 Fig. 10.
ORIGINAL_DST_TREASURES = {
    (1, 0): 1.0,
    (2, 1): 2.0,
    (3, 2): 3.0,
    (4, 3): 5.0,
    (4, 4): 8.0,
    (4, 5): 16.0,
    (7, 6): 24.0,
    (7, 7): 50.0,
    (8, 8): 74.0,
    (9, 9): 124.0,
}

GRID_ROWS = 10
GRID_COLS = 11
T_MAX_SPEC = 1000   # Vamplew 2011 spec: episode terminates after 1000 actions
T_MAX = 200         # rollout horizon used by the runner (see module docstring)
NUM_REAL_STATES = GRID_ROWS * GRID_COLS  # 110
NUM_STATES = NUM_REAL_STATES + 1         # +1 absorbing sentinel
NUM_ACTIONS = 4                          # up/down/right/left


def state_idx(row: int, col: int) -> int:
    return int(row) * GRID_COLS + int(col)


def idx_to_rowcol(idx: int) -> tuple[int, int]:
    return divmod(int(idx), GRID_COLS)


def _build_passable_mask() -> np.ndarray:
    """True where the cell is water or treasure, False where sea floor.

    Sea floor follows Vamplew 2011 Fig. 10: in each column, every row
    strictly below that column's deepest treasure is impassable. Columns
    with no treasure (here, col 10) are entirely passable water.
    """
    deepest_in_col: dict[int, int] = {}
    for (r, c) in ORIGINAL_DST_TREASURES.keys():
        deepest_in_col[c] = max(deepest_in_col.get(c, -1), r)
    mask = np.ones((GRID_ROWS, GRID_COLS), dtype=bool)
    for c, deepest_r in deepest_in_col.items():
        for r in range(deepest_r + 1, GRID_ROWS):
            mask[r, c] = False
    return mask


PASSABLE = _build_passable_mask()  # (10, 11) bool

_DR = np.array([-1, +1,  0,  0], dtype=np.int32)
_DC = np.array([ 0,  0, +1, -1], dtype=np.int32)


def original_dst_batch_rollout(env_config: dict, policy, H: int, B: int,
                               rng: np.random.Generator):
    """Run B independent original-DST episodes for up to H steps in parallel.

    Returns
        states  : (B, H) int64
        actions : (B, H) int64
        rewards : (B, H, 2) float32
    """
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
            # Sea-floor block: revert to current position if proposed is impassable.
            allowed = PASSABLE[prop_row, prop_col]
            new_row = np.where(allowed, prop_row, row)
            new_col = np.where(allowed, prop_col, col)
            row = np.where(active, new_row, row)
            col = np.where(active, new_col, col)

            tr_vals = np.zeros(B, dtype=np.float32)
            for (rr, cc), v in ORIGINAL_DST_TREASURES.items():
                hit = active & (row == rr) & (col == cc)
                tr_vals[hit] = v
            rewards[:, t, 0] = tr_vals
            rewards[:, t, 1] = np.where(active, -1.0, 0.0).astype(np.float32)

            done = done | (tr_vals > 0)

        if t + 1 >= T_MAX_SPEC:
            done[:] = True

    return states, actions, rewards


class OriginalDSTObj:
    """f(J) = sqrt(J[0] + delta) + sqrt(max(0, T_max + J[1]) + delta).

    Same shape as DSTreasureObj / LargeDSTObj. Default T_max here is the
    rollout horizon (200) rather than the 1000-step spec cap, so the
    time-term is on a comparable scale to J[0] (max treasure 124).
    Reaching treasure 124 in 18 steps gives sqrt(124+delta) + sqrt(182+delta)
    ~= 11.2 + 13.5; the trade-off is meaningful at this scale.
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
