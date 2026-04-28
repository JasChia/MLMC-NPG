"""Resource Gathering env (Barrett & Narayanan 2008; standard MORL benchmark).

5x5 grid. Agent starts at home (4,0). Two resource piles (gold at (0,4),
gems at (0,0)) must be collected. Enemy at (2,2) terminates the episode
when stepped onto.

Reward vector (M=3):
    r[0] = gold pickup indicator (0 or 1)
    r[1] = gems pickup indicator (0 or 1)
    r[2] = -1 each active step; 0 after done.

Terminal: gold AND gems collected, OR enemy hit, OR T_max steps.

Like DST, we use an absorbing-state sentinel after termination so that
score-times-RTG remains 0 in the inner PG/MLMC math. Reward = 0 after done.
"""
from __future__ import annotations

import numpy as np


GRID_ROWS = 5
GRID_COLS = 5
HOME = (4, 0)
GOLD_POS = (0, 4)
GEMS_POS = (0, 0)
ENEMY_POS = (2, 2)
T_MAX = 25
NUM_ACTIONS = 4  # 0=up, 1=down, 2=right, 3=left

# State encoding: (row, col, has_gold, has_gems). 5*5*2*2 = 100 + 1 absorbing.
NUM_REAL_STATES = GRID_ROWS * GRID_COLS * 2 * 2
NUM_STATES = NUM_REAL_STATES + 1


def state_idx(row: int, col: int, has_gold: bool, has_gems: bool) -> int:
    return (int(row) * GRID_COLS + int(col)) * 4 + (2 if has_gold else 0) + (1 if has_gems else 0)


_DR = np.array([-1, +1,  0,  0], dtype=np.int32)
_DC = np.array([ 0,  0, +1, -1], dtype=np.int32)


def resource_batch_rollout(env_config: dict, policy, H: int, B: int,
                            rng: np.random.Generator):
    """B trajectories in parallel, up to H steps.

    Returns
        states  : (B, H) int64
        actions : (B, H) int64
        rewards : (B, H, 3) float32
    """
    states = np.zeros((B, H), dtype=np.int64)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, 3), dtype=np.float32)

    row = np.full(B, HOME[0], dtype=np.int32)
    col = np.full(B, HOME[1], dtype=np.int32)
    has_gold = np.zeros(B, dtype=bool)
    has_gems = np.zeros(B, dtype=bool)
    done = np.zeros(B, dtype=bool)

    for t in range(H):
        # Encode state idx (absorbing for done agents)
        cur_state = np.where(
            done, NUM_REAL_STATES,
            (row * GRID_COLS + col) * 4 + has_gold.astype(np.int64) * 2 + has_gems.astype(np.int64)
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

            # Pickup (only if not yet picked)
            on_gold = active & (row == GOLD_POS[0]) & (col == GOLD_POS[1]) & ~has_gold
            on_gems = active & (row == GEMS_POS[0]) & (col == GEMS_POS[1]) & ~has_gems
            on_enemy = active & (row == ENEMY_POS[0]) & (col == ENEMY_POS[1])

            r0 = on_gold.astype(np.float32)
            r1 = on_gems.astype(np.float32)
            r2 = np.where(active, -1.0, 0.0).astype(np.float32)

            rewards[:, t, 0] = r0
            rewards[:, t, 1] = r1
            rewards[:, t, 2] = r2

            has_gold = has_gold | on_gold
            has_gems = has_gems | on_gems

            # Terminal: enemy hit, or both collected
            new_done = active & (on_enemy | (has_gold & has_gems))
            done = done | new_done

        if t + 1 >= T_MAX:
            done[:] = True

    return states, actions, rewards


class ResourceObj:
    """f(J) = sqrt(J[0]+delta) + sqrt(J[1]+delta) + sqrt(max(0, T_max + J[2]) + delta).

    J[0] = gold-pickup indicator returned by an episode (in [0, 1] roughly,
    discounted), J[1] = gems-pickup indicator, J[2] = sum of -1 per step
    (discounted). T_max + J[2] is approximately "steps remaining / not used".

    delta-guarded so the gradient and 2nd derivative stay bounded; MLMC's
    2^Q multiplier is therefore safe.
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
        g[2] = 0.5 / np.sqrt(c + self.delta)  # +derivative w.r.t. J[2] (since sqrt(T_max + J[2]))
        return g
