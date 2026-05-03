"""Breakable Bottles — easier variant B: lower break + longer episode + shorter corridor.

Identical to src.envs.breakable_bottles except:
  Mod #1: BREAK_PROB_PER_MOVE: 0.05 -> 0.01
  Mod #3: NUM_CELLS:           5    -> 3 (only 1 transit cell between source and home)
  Mod #6: T_MAX:               60   -> 200

3 cells means: source (cell 0), middle (cell 1), home (cell 2).
Per-attempt successful walks: 0.99^2 = 0.9801 (vs 0.95^4 = 0.8145 in original).

Same M=3 channels, 3 actions.
"""
from __future__ import annotations

import numpy as np


NUM_CELLS = 3
NUM_BOTTLE_STATES = 4
NUM_REAL_STATES = NUM_CELLS * NUM_BOTTLE_STATES
NUM_STATES = NUM_REAL_STATES + 1
NUM_ACTIONS = 3
T_MAX = 200
M_REWARD = 3

BREAK_PROB_PER_MOVE = 0.01


def _state_idx(cell: np.ndarray, bottle: np.ndarray) -> np.ndarray:
    return cell * NUM_BOTTLE_STATES + bottle


def breakable_bottles_short_batch_rollout(env_config: dict, policy, H: int, B: int,
                                            rng: np.random.Generator):
    states = np.zeros((B, H), dtype=np.int64)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, M_REWARD), dtype=np.float32)

    cell = np.zeros(B, dtype=np.int32)
    bottle = np.zeros(B, dtype=np.int32)
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
            r_water = np.zeros(B, dtype=np.float32)
            r_broken = np.zeros(B, dtype=np.float32)
            r_time = np.where(active, -1.0, 0.0).astype(np.float32)

            bottle = np.where(bottle == 3, 0, bottle)

            for b in range(B):
                if not active[b]:
                    continue
                act = int(a[b])
                if act == 0 and cell[b] > 0:
                    cell[b] -= 1
                    if bottle[b] in (1, 2) and rng.random() < BREAK_PROB_PER_MOVE:
                        bottle[b] = 3
                        r_broken[b] = -1.0
                elif act == 1 and cell[b] < NUM_CELLS - 1:
                    cell[b] += 1
                    if bottle[b] in (1, 2) and rng.random() < BREAK_PROB_PER_MOVE:
                        bottle[b] = 3
                        r_broken[b] = -1.0
                elif act == 2:
                    if cell[b] == 0 and bottle[b] == 0:
                        bottle[b] = 1
                    elif cell[b] == 0 and bottle[b] == 1:
                        bottle[b] = 2
                    elif cell[b] == NUM_CELLS - 1 and bottle[b] == 2:
                        bottle[b] = 0
                        r_water[b] = 1.0

            rewards[:, t, 0] = r_water
            rewards[:, t, 1] = r_broken
            rewards[:, t, 2] = r_time

        if t + 1 >= T_MAX:
            done[:] = True

    return states, actions, rewards


class BreakableBottlesShortObj:
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
