"""Drop-in replacement for src.train.train() that supports per-step gradient
(= ascent direction ω) L2-norm clipping before Adam. Mirrors src/train.py
otherwise, so any algorithm returning (omega, J_hat, ntraj) works unchanged.

Clipping rule: ω_clipped = ω · min(1, grad_clip / ||ω||₂).
With grad_clip = None (default) this is a no-op.

Motivated by NN-policy training where transient large-norm ascent directions
cause Adam to move θ into degenerate regions (visible as big downward spikes
in the NN figures of v2).
"""
from __future__ import annotations

from typing import Callable, Dict, Any, Optional

import numpy as np
import torch

from src.algorithms import (
    bai_pg_step, ganesh_npg_step, ganesh_mlmc_step, reinforce_step,
)
from src.train import AdamAscent


def train_v2(
    algorithm: str,
    policy,
    rollout_fn: Callable,
    obj,
    gamma: float,
    K: int,
    H: int,
    rng: np.random.Generator,
    lr: float = 0.005,
    algo_kwargs: Dict[str, Any] = None,
    verbose: bool = False,
    grad_clip: Optional[float] = None,
) -> Dict[str, list]:
    algo_kwargs = dict(algo_kwargs or {})
    opt = AdamAscent(policy.theta.shape, lr=lr,
                     device=policy.device, dtype=policy.dtype)
    history = {"iter": [], "J_hat": [], "f": [], "trajectories": [],
               "clip_events": 0}
    total_traj = 0

    for k in range(K):
        if algorithm == "bai_pg":
            omega, J_hat, ntraj = bai_pg_step(policy, rollout_fn, obj, gamma, H, **algo_kwargs)
        elif algorithm == "ganesh_npg":
            omega, J_hat, ntraj = ganesh_npg_step(policy, rollout_fn, obj, gamma, H, **algo_kwargs)
        elif algorithm == "ganesh_mlmc":
            omega, J_hat, ntraj = ganesh_mlmc_step(policy, rollout_fn, obj, gamma, H, rng=rng, **algo_kwargs)
        elif algorithm == "reinforce":
            omega, J_hat, ntraj = reinforce_step(policy, rollout_fn, obj, gamma, H, **algo_kwargs)
        else:
            raise ValueError(algorithm)

        # Grad clip (L2-norm) on the ascent direction.
        if grad_clip is not None:
            omega_arr = np.asarray(omega, dtype=np.float64)
            norm = float(np.linalg.norm(omega_arr))
            if norm > grad_clip:
                omega = (omega_arr * (grad_clip / norm)).astype(omega_arr.dtype)
                history["clip_events"] += 1

        opt.step(policy.theta, omega)
        total_traj += ntraj

        history["iter"].append(k)
        history["J_hat"].append([float(x) for x in J_hat])
        history["f"].append(obj.value(J_hat))
        history["trajectories"].append(total_traj)

        if verbose and (k % max(1, K // 20) == 0 or k == K - 1):
            print(f"  iter {k:4d} | f = {history['f'][-1]: .3f}")
    return history
