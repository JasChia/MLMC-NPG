"""Deep Sea Treasure (Vamplew et al. 2011), episodic, 10x5 layout per Paper B.

Grid: 10 rows x 5 columns. Agent starts at (0, 0).
4 actions: 0=up, 1=down, 2=right, 3=left (capped at grid edges).
Five treasures along increasing-depth columns; episode terminates when the
agent steps onto a treasure cell, OR when t == T_max (default 50).

Reward vector per step (M=2):
    r[0] = treasure_value if at_treasure_cell else 0
    r[1] = -1 each step until done; 0 after done.

Done-padding: once an agent terminates, its state stays at a sentinel
"absorbing state" index (= num_states, distinct from all real states) and
all subsequent rewards are 0. Because reward-to-go after done is 0, the
score-times-RTG product in the PG/MLMC inner loops is 0 — the gradient
estimator is unbiased w.r.t. the true (variable-length) episodic return.
The Fisher matrix accumulates absorbing-state score outer products, but
that bias is identical across Bai-PG / NPG / MLMC and so does not affect
their comparison.

Compatible with src/algorithms.py: dst_batch_rollout returns
    states  : (B, H) int64
    actions : (B, H) int64
    rewards : (B, H, 2) float32
"""
from __future__ import annotations

import numpy as np


# Treasure positions (row, col) and values. Column index = depth choice;
# row index increases with reward, giving a clear Pareto frontier.
DST_TREASURES = {
    (1, 0): 1.0,
    (2, 1): 2.0,
    (4, 2): 5.0,
    (7, 3): 16.0,
    (9, 4): 50.0,
}
DST_GRID_ROWS = 10
DST_GRID_COLS = 5
DST_T_MAX = 50  # max steps per episode (also the t in 50-t scalarization)
NUM_REAL_STATES = DST_GRID_ROWS * DST_GRID_COLS  # 50
NUM_STATES = NUM_REAL_STATES + 1                 # +1 absorbing sentinel
NUM_ACTIONS = 4                                  # up/down/right/left


def state_idx(row: int, col: int) -> int:
    return int(row) * DST_GRID_COLS + int(col)


def idx_to_rowcol(idx: int) -> tuple[int, int]:
    return divmod(int(idx), DST_GRID_COLS)


# Action effects: dr, dc
_DR = np.array([-1, +1,  0,  0], dtype=np.int32)
_DC = np.array([ 0,  0, +1, -1], dtype=np.int32)


def dst_batch_rollout(env_config: dict, policy, H: int, B: int,
                      rng: np.random.Generator):
    """Run B independent DST episodes for up to H steps in parallel.

    env_config: ignored (kept for signature uniformity with batch_rollout_v2).

    Returns (states_batch, actions_batch, rewards_batch) where shapes match
    the queueing infrastructure but states are int state-indices and rewards
    have M=2 (treasure_value, -1 step cost).
    """
    states = np.zeros((B, H), dtype=np.int64)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, 2), dtype=np.float32)

    # Per-trajectory current row/col and done flag.
    row = np.zeros(B, dtype=np.int32)
    col = np.zeros(B, dtype=np.int32)
    done = np.zeros(B, dtype=bool)

    for t in range(H):
        # State index for currently-non-done agents; absorbing sentinel for done.
        cur_state = np.where(done, NUM_REAL_STATES, row * DST_GRID_COLS + col).astype(np.int64)
        states[:, t] = cur_state

        # Sample action from current policy (vectorized over B).
        # Policy expects (B,) int state indices.
        probs = policy.action_probs_batch(cur_state)  # (B, NUM_ACTIONS)
        # Sample
        u = rng.random(B)
        cum = np.cumsum(probs, axis=-1)
        a = (u[:, None] < cum).argmax(axis=-1).astype(np.int64)
        actions[:, t] = a

        # Active agents take a step
        active = ~done
        if active.any():
            new_row = np.clip(row + _DR[a], 0, DST_GRID_ROWS - 1)
            new_col = np.clip(col + _DC[a], 0, DST_GRID_COLS - 1)
            row = np.where(active, new_row, row)
            col = np.where(active, new_col, col)

            # Treasure-value reward and step-cost reward
            tr_vals = np.zeros(B, dtype=np.float32)
            for (rr, cc), v in DST_TREASURES.items():
                hit = active & (row == rr) & (col == cc)
                tr_vals[hit] = v
            rewards[:, t, 0] = tr_vals
            rewards[:, t, 1] = np.where(active, -1.0, 0.0).astype(np.float32)

            # Mark done if treasure picked up
            new_done = tr_vals > 0
            done = done | new_done

        # Auto-terminate at H or T_MAX
        # (T_MAX should be set <= H by the runner; safe-guarded here.)
        if t + 1 >= DST_T_MAX:
            done[:] = True

    return states, actions, rewards
