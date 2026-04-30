"""v12 DST trajectory sweeps.

Runs large-DST tabular fair-budget experiments at gamma=0.9999 for two layout
variants:
  - dst_large_open: existing large DST, no seafloor
  - dst_large_floor: same treasures, Vamplew-style seafloor below treasures

Unlike v11, this saves final policy parameters and records final-policy greedy
trajectories, so path plots are based on learned policies rather than inferred
from return summaries.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from alt.envs_dst_large import (
    GRID_COLS,
    GRID_ROWS,
    LARGE_DST_TREASURES,
    LargeDSTObj,
    NUM_ACTIONS,
    NUM_REAL_STATES,
    NUM_STATES,
    T_MAX,
)
from alt.policy_grid import TabularGridPolicy
from src.train import train


_DR = np.array([-1, +1, 0, 0], dtype=np.int32)
_DC = np.array([0, 0, +1, -1], dtype=np.int32)
ACTION_NAMES = ["up", "down", "right", "left"]


def dst_reinforce_reward_transform(rewards: np.ndarray) -> np.ndarray:
    transformed = rewards.copy()
    transformed[..., 1] = (rewards[..., 1] < 0).astype(rewards.dtype)
    return transformed


def sea_floor_mask() -> np.ndarray:
    depth: dict[int, int] = {}
    for (r, c) in LARGE_DST_TREASURES:
        depth[c] = max(depth.get(c, -1), r)
    mask = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    for c, deepest in depth.items():
        mask[deepest + 1 :, c] = True
    return mask


SEA_FLOOR = sea_floor_mask()


def _is_passable(row: np.ndarray, col: np.ndarray, with_floor: bool) -> np.ndarray:
    if not with_floor:
        return np.ones_like(row, dtype=bool)
    return ~SEA_FLOOR[row, col]


def dst_large_rollout(with_floor: bool, policy, H: int, B: int, rng: np.random.Generator):
    states = np.zeros((B, H), dtype=np.int64)
    actions = np.zeros((B, H), dtype=np.int64)
    rewards = np.zeros((B, H, 2), dtype=np.float32)

    row = np.zeros(B, dtype=np.int32)
    col = np.zeros(B, dtype=np.int32)
    done = np.zeros(B, dtype=bool)

    for t in range(H):
        cur_state = np.where(done, NUM_REAL_STATES, row * GRID_COLS + col).astype(np.int64)
        states[:, t] = cur_state

        probs = policy.action_probs_batch(cur_state)
        u = rng.random(B)
        cum = np.cumsum(probs, axis=-1)
        action = (u[:, None] < cum).argmax(axis=-1).astype(np.int64)
        actions[:, t] = action

        active = ~done
        if active.any():
            prop_row = np.clip(row + _DR[action], 0, GRID_ROWS - 1)
            prop_col = np.clip(col + _DC[action], 0, GRID_COLS - 1)
            allowed = _is_passable(prop_row, prop_col, with_floor)
            new_row = np.where(allowed, prop_row, row)
            new_col = np.where(allowed, prop_col, col)
            row = np.where(active, new_row, row)
            col = np.where(active, new_col, col)

            tr_vals = np.zeros(B, dtype=np.float32)
            for (rr, cc), value in LARGE_DST_TREASURES.items():
                hit = active & (row == rr) & (col == cc)
                tr_vals[hit] = value
            rewards[:, t, 0] = tr_vals
            rewards[:, t, 1] = np.where(active, -1.0, 0.0).astype(np.float32)
            done = done | (tr_vals > 0)

        if t + 1 >= T_MAX:
            done[:] = True

    return states, actions, rewards


def record_greedy_trajectory(policy, with_floor: bool, H: int = T_MAX) -> dict:
    row = 0
    col = 0
    cells = [(row, col)]
    states = []
    actions = []
    action_names = []
    rewards = []
    action_probs = []
    treasure = 0.0
    terminated = False

    for _ in range(H):
        state = row * GRID_COLS + col
        probs = policy.action_probs_batch(np.array([state], dtype=np.int64))[0]
        action = int(np.argmax(probs))
        prop_row = int(np.clip(row + int(_DR[action]), 0, GRID_ROWS - 1))
        prop_col = int(np.clip(col + int(_DC[action]), 0, GRID_COLS - 1))
        if not bool(_is_passable(np.array([prop_row]), np.array([prop_col]), with_floor)[0]):
            prop_row, prop_col = row, col

        reward = [0.0, -1.0]
        if (prop_row, prop_col) in LARGE_DST_TREASURES:
            treasure = float(LARGE_DST_TREASURES[(prop_row, prop_col)])
            reward[0] = treasure
            terminated = True

        states.append(int(state))
        actions.append(action)
        action_names.append(ACTION_NAMES[action])
        action_probs.append([float(x) for x in probs])
        rewards.append(reward)
        row, col = prop_row, prop_col
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


def configs(args) -> list[dict]:
    rows = [
        ("ganesh_mlmc", 4, 300, "ref"),
        ("bai_pg", 4, 1000, "moreK"),
        ("bai_pg", 13, 300, "largerN"),
        ("reinforce", 4, 2000, "moreK"),
        ("reinforce", 26, 300, "largerN"),
    ]
    out = []
    for variant in ("open", "floor"):
        for algo, N, K, mode in rows:
            if args.algos and algo not in args.algos:
                continue
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                out.append({
                    "variant": variant,
                    "algorithm": algo,
                    "N": N,
                    "K": K,
                    "mode": mode,
                    "seed": seed,
                    "gamma": args.gamma,
                    "lr": args.lr,
                    "delta": args.delta,
                    "N_inner": args.N_inner,
                    "beta": args.beta,
                    "B_max": args.B_max,
                    "delta_reinforce": args.delta_reinforce,
                    "out_dir": args.out_dir,
                })
    return out


def filename(cfg: dict) -> str:
    return (
        f"{cfg['algorithm']}_dst_large_{cfg['variant']}_tab_"
        f"N{cfg['N']}_K{cfg['K']}_{cfg['mode']}_seed{cfg['seed']}.json"
    )


def run_one(cfg: dict) -> dict:
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)

    rng = np.random.default_rng(cfg["seed"])
    policy = TabularGridPolicy(NUM_STATES, NUM_ACTIONS)
    obj = LargeDSTObj(delta=cfg["delta"], T_max=T_MAX)
    with_floor = cfg["variant"] == "floor"
    rollout_fn = lambda H, B: dst_large_rollout(with_floor, policy, H, B, rng)

    N = cfg["N"]
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

    t0 = time.time()
    history = train(
        algorithm=cfg["algorithm"],
        policy=policy,
        rollout_fn=rollout_fn,
        obj=obj,
        gamma=cfg["gamma"],
        K=cfg["K"],
        H=T_MAX,
        rng=rng,
        lr=cfg["lr"],
        algo_kwargs=algo_kwargs,
        verbose=False,
    )
    history["wall_time_sec"] = time.time() - t0
    history["config"] = cfg
    history["final_theta"] = [float(x) for x in policy.theta.detach().cpu().numpy()]
    history["greedy_trajectory"] = record_greedy_trajectory(policy, with_floor, H=T_MAX)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename(cfg)
    with open(out_path, "w") as f:
        json.dump(history, f)
    return {"file": str(out_path), "time": history["wall_time_sec"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/sweep_v12_dst_traj_gamma09999")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--algos", nargs="+", default=None)
    p.add_argument("--gamma", type=float, default=0.9999)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--N_inner", type=int, default=5)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--B_max", type=int, default=64)
    p.add_argument("--delta_reinforce", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    all_cfgs = configs(args)
    if args.skip_existing:
        kept = []
        for cfg in all_cfgs:
            if not (Path(cfg["out_dir"]) / filename(cfg)).exists():
                kept.append(cfg)
        print(f"skip_existing: {len(all_cfgs) - len(kept)} present, {len(kept)} to run")
        all_cfgs = kept

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "sweep_manifest.json", "w") as f:
        json.dump({"args": vars(args), "n": len(all_cfgs)}, f, indent=2)

    print(f"v12 dst trajectory sweep: {len(all_cfgs)} runs, {args.workers} workers")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, cfg): cfg for cfg in all_cfgs}
        for fut in as_completed(futures):
            cfg = futures[fut]
            res = fut.result()
            done += 1
            elapsed = time.time() - t0
            avg = elapsed / done
            eta = avg * (len(all_cfgs) - done)
            print(
                f"  [{done:2d}/{len(all_cfgs)}] {cfg['variant']:<5} "
                f"{cfg['algorithm']:<10} N={cfg['N']:<2} K={cfg['K']:<4} "
                f"{cfg['mode']:<8} seed={cfg['seed']} {res['time']:6.1f}s "
                f"(ETA {eta/60:4.1f} min)"
            )

    print(f"v12 dst trajectory sweep done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
