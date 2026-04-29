"""v13 DST sweep — all 6 variants (3 maps x {open, floor}) and tab/NN policies.

Variants:
    dst_open            10x5  no sea floor
    dst_floor           10x5  Vamplew sea floor
    dst_large_open      20x10 no sea floor
    dst_large_floor     20x10 Vamplew sea floor
    dst_original_open   10x11 no sea floor
    dst_original_floor  10x11 Vamplew sea floor

Hyperparams (matched ~5k-traj fair budget):
    MLMC ref     : N (=B_inner) = 4, K = 35,   N_inner = 35, B_max = 16
    Bai-PG       : N1 = N2 = 4, K = 612
    REINFORCE    : N = 4, K = 1225, with DST reward_transform fix

Each run saves history (incl. final theta + greedy trajectory) under
results/sweep_v13_dst_<tag>/{algo}_{env}_{policy_tag}_N{N}_K{K}_v13_seed{s}.json
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from alt.envs_dst import DST_GRID_COLS, DST_GRID_ROWS, DST_TREASURES, DST_T_MAX
from alt.envs_dst_large import (
    GRID_COLS as LARGE_COLS, GRID_ROWS as LARGE_ROWS,
    LARGE_DST_TREASURES, T_MAX as LARGE_T_MAX,
)
from alt.envs_dst_original import (
    GRID_COLS as ORIG_COLS, GRID_ROWS as ORIG_ROWS,
    ORIGINAL_DST_TREASURES, T_MAX as ORIG_T_MAX,
)
from alt.objectives_dst import DSTreasureObj
from alt.envs_dst_large import LargeDSTObj
from alt.envs_dst_original import OriginalDSTObj
from alt.policy_grid import TabularGridPolicy
from alt.policy_nn_state import NNStatePolicy
from alt.scalarized_reinforce import train_scalarized_reinforce
from src.train import train


_DR = np.array([-1, +1, 0, 0], dtype=np.int32)
_DC = np.array([0, 0, +1, -1], dtype=np.int32)
ACTION_NAMES = ["up", "down", "right", "left"]
NUM_ACTIONS = 4


def dst_reinforce_reward_transform(rewards: np.ndarray) -> np.ndarray:
    transformed = rewards.copy()
    transformed[..., 1] = (rewards[..., 1] < 0).astype(rewards.dtype)
    return transformed


@dataclass(frozen=True)
class DstSpec:
    env: str
    rows: int
    cols: int
    treasures: dict
    t_max: int
    obj_kind: str   # "small" | "large" | "original"
    has_floor: bool

    @property
    def num_real_states(self) -> int:
        return self.rows * self.cols

    @property
    def num_states(self) -> int:
        return self.num_real_states + 1


SPECS = {
    "dst_open":           DstSpec("dst_open",           DST_GRID_ROWS, DST_GRID_COLS, DST_TREASURES,           DST_T_MAX,   "small",    False),
    "dst_floor":          DstSpec("dst_floor",          DST_GRID_ROWS, DST_GRID_COLS, DST_TREASURES,           DST_T_MAX,   "small",    True),
    "dst_large_open":     DstSpec("dst_large_open",     LARGE_ROWS,    LARGE_COLS,    LARGE_DST_TREASURES,     LARGE_T_MAX, "large",    False),
    "dst_large_floor":    DstSpec("dst_large_floor",    LARGE_ROWS,    LARGE_COLS,    LARGE_DST_TREASURES,     LARGE_T_MAX, "large",    True),
    "dst_original_open":  DstSpec("dst_original_open",  ORIG_ROWS,     ORIG_COLS,     ORIGINAL_DST_TREASURES,  ORIG_T_MAX,  "original", False),
    "dst_original_floor": DstSpec("dst_original_floor", ORIG_ROWS,     ORIG_COLS,     ORIGINAL_DST_TREASURES,  ORIG_T_MAX,  "original", True),
}


def sea_floor_mask(spec: DstSpec) -> np.ndarray:
    depth: dict = {}
    for (r, c) in spec.treasures:
        depth[c] = max(depth.get(c, -1), r)
    mask = np.zeros((spec.rows, spec.cols), dtype=bool)
    for c, deepest in depth.items():
        mask[deepest + 1:, c] = True
    return mask


SEA_FLOORS = {env: sea_floor_mask(spec) for env, spec in SPECS.items()}


def make_obj(spec: DstSpec, delta: float):
    if spec.obj_kind == "small":
        return DSTreasureObj(delta=delta, T_max=spec.t_max)
    if spec.obj_kind == "large":
        return LargeDSTObj(delta=delta, T_max=spec.t_max)
    if spec.obj_kind == "original":
        return OriginalDSTObj(delta=delta, T_max=spec.t_max)
    raise ValueError(spec.obj_kind)


def dst_rollout(spec: DstSpec, policy, H: int, B: int, rng):
    states = np.zeros((B, H), dtype=np.int64)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, 2), dtype=np.float32)
    row = np.zeros(B, dtype=np.int32)
    col = np.zeros(B, dtype=np.int32)
    done = np.zeros(B, dtype=bool)

    floor_mask = SEA_FLOORS[spec.env] if spec.has_floor else None

    for t in range(H):
        cur_state = np.where(done, spec.num_real_states,
                             row * spec.cols + col).astype(np.int64)
        states[:, t] = cur_state

        probs = policy.action_probs_batch(cur_state)
        u = rng.random(B)
        cum = np.cumsum(probs, axis=-1)
        action = (u[:, None] < cum).argmax(axis=-1).astype(np.int64)
        actions[:, t] = action

        active = ~done
        if active.any():
            prop_row = np.clip(row + _DR[action], 0, spec.rows - 1)
            prop_col = np.clip(col + _DC[action], 0, spec.cols - 1)
            if floor_mask is not None:
                allowed = ~floor_mask[prop_row, prop_col]
                prop_row = np.where(allowed, prop_row, row)
                prop_col = np.where(allowed, prop_col, col)
            row = np.where(active, prop_row, row)
            col = np.where(active, prop_col, col)

            tr_vals = np.zeros(B, dtype=np.float32)
            for (rr, cc), v in spec.treasures.items():
                hit = active & (row == rr) & (col == cc)
                tr_vals[hit] = v
            rewards[:, t, 0] = tr_vals
            rewards[:, t, 1] = np.where(active, -1.0, 0.0).astype(np.float32)
            done = done | (tr_vals > 0)

        if t + 1 >= spec.t_max:
            done[:] = True

    return states, actions, rewards


def record_greedy_trajectory(spec: DstSpec, policy) -> dict:
    row, col = 0, 0
    cells = [(row, col)]
    states, actions, action_names, action_probs, rewards = [], [], [], [], []
    terminated = False
    treasure = 0.0
    floor_mask = SEA_FLOORS[spec.env] if spec.has_floor else None

    for _ in range(spec.t_max):
        state = row * spec.cols + col
        probs = policy.action_probs_batch(np.array([state], dtype=np.int64))[0]
        action = int(np.argmax(probs))
        next_row = int(np.clip(row + int(_DR[action]), 0, spec.rows - 1))
        next_col = int(np.clip(col + int(_DC[action]), 0, spec.cols - 1))
        if floor_mask is not None and floor_mask[next_row, next_col]:
            next_row, next_col = row, col
        reward = [0.0, -1.0]
        if (next_row, next_col) in spec.treasures:
            treasure = float(spec.treasures[(next_row, next_col)])
            reward[0] = treasure
            terminated = True
        states.append(int(state))
        actions.append(action)
        action_names.append(ACTION_NAMES[action])
        action_probs.append([float(x) for x in probs])
        rewards.append(reward)
        row, col = next_row, next_col
        cells.append((row, col))
        if terminated:
            break

    return {
        "cells": [[int(r), int(c)] for r, c in cells],
        "states": states,
        "actions": actions,
        "action_names": action_names,
        "action_probs": action_probs,
        "rewards": rewards,
        "terminated": terminated,
        "treasure": treasure,
        "steps": len(actions),
    }


# v13 fair-budget rows
def base_rows():
    return [
        # algo,                   N,  K,    mode
        ("ganesh_mlmc",           4,  35,   "v13"),  # B_inner=N=4, K=35; N_inner=35, B_max=16
        ("bai_pg",                4,  612,  "v13"),  # N1=N2=N=4
        ("reinforce",             4,  1225, "v13"),  # paper-A eq.(38) (kept for reference)
        ("scalarized_reinforce",  4,  1225, "v13"),  # best-effort scalarized REINFORCE
    ]


def configs(args) -> list:
    out = []
    for env in args.envs:
        if env not in SPECS:
            raise ValueError(f"unknown env {env}")
        for algo, N, K, mode in base_rows():
            K = int(K * args.K_multiplier)
            if args.algos and algo not in args.algos:
                continue
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                out.append({
                    "env": env, "algorithm": algo, "N": N, "K": K, "mode": mode,
                    "seed": seed,
                    "policy": args.policy,
                    "hidden": list(args.hidden) if args.policy == "nn" else None,
                    "gamma": args.gamma, "lr": args.lr, "delta": args.delta,
                    "N_inner": args.N_inner, "beta": args.beta, "B_max": args.B_max,
                    "delta_reinforce": args.delta_reinforce,
                    "out_dir": args.out_dir,
                })
    return out


def policy_tag(cfg: dict) -> str:
    if cfg.get("policy") == "nn":
        return "nn" + "x".join(str(x) for x in cfg["hidden"])
    return "tab"


def filename(cfg: dict) -> str:
    return (
        f"{cfg['algorithm']}_{cfg['env']}_{policy_tag(cfg)}_"
        f"N{cfg['N']}_K{cfg['K']}_{cfg['mode']}_seed{cfg['seed']}.json"
    )


def run_one(cfg: dict) -> dict:
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)

    spec = SPECS[cfg["env"]]
    rng = np.random.default_rng(cfg["seed"])

    if cfg.get("policy") == "nn":
        policy = NNStatePolicy(spec.num_states, NUM_ACTIONS, hidden=tuple(cfg["hidden"]))
    else:
        policy = TabularGridPolicy(spec.num_states, NUM_ACTIONS)

    obj = make_obj(spec, cfg["delta"])
    rollout_fn = lambda H, B: dst_rollout(spec, policy, H, B, rng)

    N = cfg["N"]
    t0 = time.time()
    if cfg["algorithm"] == "scalarized_reinforce":
        history = train_scalarized_reinforce(
            policy=policy, rollout_fn=rollout_fn, obj=obj,
            gamma=cfg["gamma"], K=cfg["K"], H=spec.t_max, rng=rng,
            lr=cfg["lr"], N=N, verbose=False,
        )
    else:
        if cfg["algorithm"] == "bai_pg":
            algo_kwargs = {"N1": N, "N2": N}
        elif cfg["algorithm"] == "reinforce":
            algo_kwargs = {"N": N, "delta_guard": cfg["delta_reinforce"],
                           "reward_transform": dst_reinforce_reward_transform}
        elif cfg["algorithm"] == "ganesh_mlmc":
            algo_kwargs = {"B_max": cfg["B_max"], "B_inner": N,
                           "N_inner": cfg["N_inner"], "beta": cfg["beta"]}
        else:
            raise ValueError(cfg["algorithm"])
        history = train(
            algorithm=cfg["algorithm"], policy=policy, rollout_fn=rollout_fn, obj=obj,
            gamma=cfg["gamma"], K=cfg["K"], H=spec.t_max, rng=rng,
            lr=cfg["lr"], algo_kwargs=algo_kwargs, verbose=False,
        )
    history["wall_time_sec"] = time.time() - t0
    history["config"] = cfg
    history["final_theta"] = [float(x) for x in policy.theta.detach().cpu().numpy()]
    history["greedy_trajectory"] = record_greedy_trajectory(spec, policy)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / filename(cfg), "w") as f:
        json.dump(history, f)
    return {"file": str(out_dir / filename(cfg)), "time": history["wall_time_sec"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--envs", nargs="+", default=list(SPECS.keys()))
    p.add_argument("--algos", nargs="+", default=None)
    p.add_argument("--policy", choices=["tabular", "nn"], default="tabular")
    p.add_argument("--hidden", type=int, nargs="+", default=[32])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.9999)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--N_inner", type=int, default=35)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--B_max", type=int, default=16)
    p.add_argument("--delta_reinforce", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--K_multiplier", type=float, default=1.0,
                   help="multiplicative scale for all K values (e.g. 2.0 doubles K)")
    args = p.parse_args()

    cfgs = configs(args)
    if args.skip_existing:
        cfgs = [c for c in cfgs if not (Path(c["out_dir"]) / filename(c)).exists()]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "sweep_manifest.json", "w") as f:
        json.dump({"args": vars(args), "n": len(cfgs)}, f, indent=2)

    print(f"v13 DST sweep: {len(cfgs)} runs, {args.workers} workers, policy={args.policy}")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, c): c for c in cfgs}
        for fut in as_completed(futures):
            cfg = futures[fut]
            try:
                res = fut.result()
                done += 1
                elapsed = time.time() - t0
                eta = elapsed / done * (len(cfgs) - done)
                print(f"  [{done:3d}/{len(cfgs)}] {cfg['env']:<19} {cfg['algorithm']:<12} "
                      f"N={cfg['N']:<2} K={cfg['K']:<5} seed={cfg['seed']} "
                      f"{res['time']:6.1f}s (ETA {eta/60:5.1f} min)")
            except Exception as e:
                print(f"  FAILED {cfg['algorithm']} {cfg['env']} seed={cfg['seed']}: "
                      f"{type(e).__name__}: {e}")

    print(f"v13 DST sweep done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
