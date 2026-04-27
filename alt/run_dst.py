"""Sweep runner for Deep Sea Treasure (Paper B env 2). Mirrors run_sweep_v3
but wires in DST-specific env / objective / policy. Uses src/train.train()
and src/algorithms.py unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def run_one(cfg: dict) -> dict:
    from src.train import train
    from alt.envs_dst import dst_batch_rollout, NUM_STATES, NUM_ACTIONS, DST_T_MAX
    from alt.objectives_dst import DSTreasureObj
    from alt.policy_grid import TabularGridPolicy

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)

    policy = TabularGridPolicy(NUM_STATES, NUM_ACTIONS)
    obj = DSTreasureObj(delta=cfg["delta"], T_max=DST_T_MAX)
    rng = np.random.default_rng(cfg["seed"])

    def rollout_fn(H, B):
        return dst_batch_rollout({}, policy, H, B, rng)

    N = cfg["N"]
    if cfg["algorithm"] == "bai_pg":
        algo_kwargs = {"N1": N, "N2": N}
    elif cfg["algorithm"] == "reinforce":
        algo_kwargs = {"N": N, "delta_guard": cfg.get("delta_reinforce", 1.0)}
    elif cfg["algorithm"] == "ganesh_mlmc":
        algo_kwargs = {"B_max": cfg.get("B_max", 64), "B_inner": N,
                       "N_inner": cfg["N_inner"], "beta": cfg["beta"]}
    elif cfg["algorithm"] == "ganesh_npg":
        algo_kwargs = {"B1": N, "B2": N,
                       "N_inner": cfg["N_inner"], "beta": cfg["beta"]}
    else:
        raise ValueError(cfg["algorithm"])

    t0 = time.time()
    history = train(
        algorithm=cfg["algorithm"],
        policy=policy, rollout_fn=rollout_fn, obj=obj,
        gamma=cfg["gamma"], K=cfg["K"], H=DST_T_MAX,
        rng=rng, lr=cfg["lr"], algo_kwargs=algo_kwargs, verbose=False,
    )
    history["wall_time_sec"] = time.time() - t0
    history["config"] = cfg

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (
        f"{cfg['algorithm']}_dst_N{N}_seed{cfg['seed']}.json"
    )
    with open(out_file, "w") as f:
        json.dump(history, f)
    return {"file": str(out_file), "time": history["wall_time_sec"]}


def build_configs(args) -> List[Dict]:
    configs = []
    for algo in args.algos:
        for N in args.Ns:
            for seed in range(args.seeds):
                configs.append({
                    "algorithm": algo, "N": N, "seed": seed,
                    "K": args.K, "gamma": args.gamma,
                    "lr": args.lr, "delta": args.delta,
                    "N_inner": args.N_inner, "beta": args.beta,
                    "B_max": args.B_max,
                    "delta_reinforce": args.delta_reinforce,
                    "out_dir": args.out_dir,
                })
    return configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/sweep_dst")
    p.add_argument("--Ns", type=int, nargs="+", default=[2, 4, 16])
    p.add_argument("--algos", nargs="+",
                   default=["bai_pg", "reinforce", "ganesh_mlmc"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--K", type=int, default=400)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--N_inner", type=int, default=5)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--B_max", type=int, default=64)
    p.add_argument("--delta_reinforce", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    configs = build_configs(args)
    if args.skip_existing:
        out_dir = Path(args.out_dir)
        def out_file(c):
            return out_dir / f"{c['algorithm']}_dst_N{c['N']}_seed{c['seed']}.json"
        kept = [c for c in configs if not out_file(c).exists()]
        print(f"skip_existing: {len(configs) - len(kept)} present, {len(kept)} to run")
        configs = kept

    print(f"DST sweep: {len(configs)} runs, {args.workers} workers")
    print(f"  algos={args.algos} Ns={args.Ns} K={args.K} gamma={args.gamma} lr={args.lr}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "sweep_manifest.json", "w") as f:
        json.dump({"args": vars(args), "n": len(configs)}, f, indent=2)

    t0 = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, cfg): cfg for cfg in configs}
        for fut in as_completed(futures):
            cfg = futures[fut]
            try:
                res = fut.result()
                completed += 1
                elapsed = time.time() - t0
                avg = elapsed / completed
                eta = avg * (len(configs) - completed)
                tag = f"{cfg['algorithm']:<12} N={cfg['N']:<3} seed={cfg['seed']}"
                print(f"  [{completed:3d}/{len(configs)}] {tag}  "
                      f"{res['time']:6.1f}s  (ETA {eta/60:5.1f} min)")
            except Exception as e:
                tag = f"{cfg['algorithm']} N={cfg['N']} seed={cfg['seed']}"
                print(f"  FAILED {tag}: {type(e).__name__}: {e}")

    print(f"\nDST sweep done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
