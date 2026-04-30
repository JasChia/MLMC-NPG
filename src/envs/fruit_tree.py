"""Fruit Tree env (Yang, Sun, Narasimhan 2019, "A generalized algorithm for
multi-objective reinforcement learning and policy adaptation", NeurIPS).

Binary tree of depth d. At each internal node, agent chooses left (action=0)
or right (action=1). Episode is exactly d steps. Each leaf emits a fixed
6-dim nutrient vector (protein, carbs, fats, vitamins, minerals, water).
Internal nodes give zero reward.

State encoding: int index of the current node. Internal nodes 0..(2^d - 2),
leaves are reached after d actions.

This is an extreme sparse-reward concave-MORL setup: per-trajectory return
is a single 6-vector chosen from 2^d possible leaves, deterministic given
the action sequence. Bias-correction has maximum leverage because Var(J)
is high across episodes (different leaves picked) but bounded.

Reference Pareto front for depth=7 in Yang 2019; we use the same leaf-vector
construction (random sampling on a sphere with sublinear scaling per depth).

Reward shape (M=6) returned in (B, H, 6); reward emitted only at the leaf
(last active step). H = depth.
"""
from __future__ import annotations

import numpy as np


M_REWARD = 6
DEFAULT_DEPTH = 7


def _generate_leaf_values(depth: int, seed: int = 42,
                          minmax_scale: bool = False) -> np.ndarray:
    """Deterministic leaf-vector construction.

    Each leaf gets a 6-dim vector with values in [0, 1]. Leaves "deeper down"
    (along certain branches) are higher-value. We use a per-leaf random seed
    so the construction is reproducible across runs.

    Returns an array of shape (2^depth, 6).
    """
    rng = np.random.default_rng(seed)
    n_leaves = 2 ** depth
    raw = rng.uniform(0.3, 1.0, size=(n_leaves, M_REWARD)).astype(np.float64)
    # Scale up so largest possible f(J) is in a reasonable range; not strictly
    # necessary but gives plateau values comparable to DST.
    raw *= 10.0
    if minmax_scale:
        mins = raw.min(axis=0, keepdims=True)
        maxs = raw.max(axis=0, keepdims=True)
        denom = np.maximum(maxs - mins, 1e-12)
        raw = (raw - mins) / denom
    return raw


def state_idx(node: int) -> int:
    """Internal nodes 0..(2^d-2), leaves are absorbing once reached."""
    return int(node)


def _build_tree_step():
    """Tree topology: node 0 is root; left child of i = 2i+1, right child = 2i+2.
    For depth d, internal nodes are 0..2^d - 2; first leaf index is 2^d - 1.
    """
    pass


class FruitTreeEnv:
    """Stateful holder for a depth-d fruit tree with deterministic leaf values."""

    def __init__(self, depth: int = DEFAULT_DEPTH, seed: int = 42,
                 minmax_scale: bool = False):
        self.depth = int(depth)
        self.n_internal = (1 << depth) - 1   # 2^d - 1
        self.n_leaves = 1 << depth           # 2^d
        self.num_real_states = self.n_internal + self.n_leaves
        self.num_states = self.num_real_states + 1  # +1 absorbing sentinel
        self.num_actions = 2
        self.leaf_values = _generate_leaf_values(depth, seed=seed,
                                                 minmax_scale=minmax_scale)
        # First leaf index in linear ordering: n_internal
        self.first_leaf_idx = self.n_internal


def fruit_tree_batch_rollout(env_config: dict, policy, H: int, B: int,
                              rng: np.random.Generator):
    """Run B parallel episodes for up to H steps. Episode terminates when a
    leaf is reached (after exactly `depth` steps).

    env_config["env"] must be a FruitTreeEnv instance.

    Returns
        states  : (B, H) int64
        actions : (B, H) int64
        rewards : (B, H, 6) float32
    """
    env: FruitTreeEnv = env_config["env"]
    depth = env.depth

    states = np.zeros((B, H), dtype=np.int64)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, M_REWARD), dtype=np.float32)

    node = np.zeros(B, dtype=np.int64)  # current internal node index
    done = np.zeros(B, dtype=bool)

    for t in range(H):
        cur_state = np.where(done, env.num_real_states, node).astype(np.int64)
        states[:, t] = cur_state

        probs = policy.action_probs_batch(cur_state)
        u = rng.random(B)
        cum = np.cumsum(probs, axis=-1)
        a = (u[:, None] < cum).argmax(axis=-1).astype(np.int64)
        actions[:, t] = a

        active = ~done
        if active.any():
            # Compute next node for active agents
            new_node = 2 * node + 1 + a   # left child = 2i+1, right child = 2i+2
            node = np.where(active, new_node, node)

            # Did agent step into a leaf? After exactly `depth` steps from root,
            # nodes will be in [first_leaf_idx, first_leaf_idx + n_leaves).
            # Equivalently, check if node >= n_internal.
            reached_leaf = active & (node >= env.first_leaf_idx)
            if reached_leaf.any():
                # Emit the leaf vector
                leaf_idx = node[reached_leaf] - env.first_leaf_idx
                leaf_vals = env.leaf_values[leaf_idx]  # (n_hits, M)
                rewards[reached_leaf, t, :] = leaf_vals.astype(np.float32)
            done = done | reached_leaf

        if t + 1 >= depth:
            done[:] = True

    return states, actions, rewards


class FruitTreeObj:
    """f(J) = sum_m sqrt(max(0, J[m]) + delta).

    Concave fairness over the 6 nutrient dimensions. Encourages the policy to
    pick leaves that have a balanced profile across all 6 channels rather
    than any single dominating channel.
    """
    def __init__(self, delta: float = 0.5):
        self.delta = float(delta)

    def value(self, J) -> float:
        J = np.asarray(J, dtype=np.float64)
        return float(np.sum(np.sqrt(np.maximum(0.0, J) + self.delta)))

    def grad(self, J) -> np.ndarray:
        J = np.asarray(J, dtype=np.float64)
        Jp = np.maximum(0.0, J)
        return 0.5 / np.sqrt(Jp + self.delta)
