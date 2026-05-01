"""Breakable Bottles env (mo-gymnasium spec).

Agent walks back and forth between source (cell 0) and home (cell N-1).
Picks up empty bottles at source, fills them with water by walking,
and breaks them if not handled with care. Multi-objective:
    obj 0:  +1 per bottle of water successfully delivered to home
    obj 1:  -1 per bottle broken
    obj 2:  -1 each step (time penalty)

Simplified state encoding (works with our int-state policies):
    state = cell_position * (NUM_BOTTLE_STATES) + bottle_state_idx

bottle_states (small, discretized):
    0: hands_empty, no bottle
    1: holding_empty_bottle
    2: holding_full_bottle
    3: bottle_broken (just broke this step)

NUM_CELLS = 5 (source -> 3 transit -> home)
NUM_BOTTLE_STATES = 4
NUM_REAL_STATES = 5 * 4 = 20

3 actions: 0=move_left, 1=move_right, 2=interact
T_max = 60 steps.

This is a simplified faithful approximation of mo-gymnasium's
breakable-bottles env. The original is a richer symbolic env, but
the core multi-objective trade-off (water collected vs bottles broken
vs time) is preserved.
"""
from __future__ import annotations

import numpy as np


NUM_CELLS = 5
NUM_BOTTLE_STATES = 4
NUM_REAL_STATES = NUM_CELLS * NUM_BOTTLE_STATES
NUM_STATES = NUM_REAL_STATES + 1
NUM_ACTIONS = 3
T_MAX = 60
M_REWARD = 3

# Probability of breaking a held bottle when moving
BREAK_PROB_PER_MOVE = 0.05


def _state_idx(cell: np.ndarray, bottle: np.ndarray) -> np.ndarray:
    return cell * NUM_BOTTLE_STATES + bottle


def breakable_bottles_batch_rollout(env_config: dict, policy, H: int, B: int,
                                      rng: np.random.Generator):
    states = np.zeros((B, H), dtype=np.int64)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, M_REWARD), dtype=np.float32)

    cell = np.zeros(B, dtype=np.int32)        # start at source
    bottle = np.zeros(B, dtype=np.int32)      # start with empty hands
    done = np.zeros(B, dtype=bool)

    for t in range(H):
        cur_state = np.where(done, NUM_REAL_STATES,
                             _state_idx(cell, bottle)).astype(np.int64)
        states[:, t] = cur_state

        probs = policy.action_probs_batch(cur_state)
        u = rng.random(B)
        cum = np.cumsum(probs, axis=-1)
        a = (u[:, None] < cum).argmax(axis=-1).astype(np.int64)
        actions[:, t] = a

        active = ~done
        if active.any():
            r_water = np.zeros(B, dtype=np.float32)     # +1 per bottle delivered
            r_broken = np.zeros(B, dtype=np.float32)    # -1 per break
            r_time = np.where(active, -1.0, 0.0).astype(np.float32)

            # Reset bottle_broken sentinel state (it's transient)
            bottle = np.where(bottle == 3, 0, bottle)

            for b in range(B):
                if not active[b]:
                    continue
                act = int(a[b])
                if act == 0 and cell[b] > 0:                     # move left
                    cell[b] -= 1
                    if bottle[b] in (1, 2) and rng.random() < BREAK_PROB_PER_MOVE:
                        bottle[b] = 3
                        r_broken[b] = -1.0
                elif act == 1 and cell[b] < NUM_CELLS - 1:       # move right
                    cell[b] += 1
                    if bottle[b] in (1, 2) and rng.random() < BREAK_PROB_PER_MOVE:
                        bottle[b] = 3
                        r_broken[b] = -1.0
                elif act == 2:                                    # interact
                    if cell[b] == 0 and bottle[b] == 0:
                        # at source, pick up empty bottle
                        bottle[b] = 1
                    elif cell[b] == 0 and bottle[b] == 1:
                        # at source, fill with water
                        bottle[b] = 2
                    elif cell[b] == NUM_CELLS - 1 and bottle[b] == 2:
                        # at home, deliver water
                        bottle[b] = 0
                        r_water[b] = 1.0

            rewards[:, t, 0] = r_water
            rewards[:, t, 1] = r_broken
            rewards[:, t, 2] = r_time

        if t + 1 >= T_MAX:
            done[:] = True

    return states, actions, rewards


class BreakableBottlesObj:
    """f(J) = sqrt(max(0, J[0]) + delta)
              + sqrt(max(0, T_max + J[1]) + delta)
              + sqrt(max(0, T_max + J[2]) + delta).

    J[0] = water delivered (>= 0). J[1] = -break_count (<= 0). J[2] = -time.
    Maximize: more water, fewer breaks, less time.
    """
    def __init__(self, delta: float = 0.5, T_max: int = T_MAX):
        self.delta = float(delta)
        self.T_max = float(T_max)

    def value(self, J) -> float:
        J = np.asarray(J, dtype=np.float64)
        a = max(0.0, float(J[0]))
        b = max(0.0, self.T_max + float(J[1]))
        c = max(0.0, self.T_max + float(J[2]))
        return float(np.sqrt(a + self.delta) + np.sqrt(b + self.delta) + np.sqrt(c + self.delta))

    def grad(self, J) -> np.ndarray:
        J = np.asarray(J, dtype=np.float64)
        a = max(0.0, float(J[0]))
        b = max(0.0, self.T_max + float(J[1]))
        c = max(0.0, self.T_max + float(J[2]))
        g = np.zeros(3, dtype=np.float64)
        g[0] = 0.5 / np.sqrt(a + self.delta)
        g[1] = 0.5 / np.sqrt(b + self.delta)
        g[2] = 0.5 / np.sqrt(c + self.delta)
        return g
