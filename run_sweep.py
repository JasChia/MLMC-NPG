"""Parallel sweep runner over (algorithm, M, N, seed, policy).

Spawns a multiprocessing worker pool (one run per worker process). Each run
writes its history to results/<sweep_name>/<algo>_<policy>_M<M>_N<N>_seed<seed>.json.

Default sweep matches Paper A's Figure 1 / Figure 2:
    algorithms : bai_pg, reinforce           (extend with ganesh_{npg,mlmc} via --include_ganesh)
    M          : 2, 4, 8
    N          : 4, 16, 64, 256
    seeds      : 3 by default (paper uses 10; bump via --seeds)

Total Bai-PG + REINFORCE runs with default settings:
    2 algos * 3 M * 4 N * 3 seeds = 72 runs
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
    """Worker entry point. `cfg` has all parameters for a single run."""
    # Local imports so each worker starts from a clean module state.
    from src.envs import ARRIVAL_RATES, batch_rollout
    from src.policy import make_policy
    from src.objectives import AlphaFairness
    from src.train import train

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)  # avoid thread contention across workers

    env_config = {"M": cfg["M"], "L_max": cfg["L_max"],
                  "lambdas": ARRIVAL_RATES[cfg["M"]]}
    policy = make_policy(cfg["policy"], M=cfg["M"], L_max=cfg["L_max"], device="cpu")
    obj = AlphaFairness(alpha=2.0, delta=cfg["delta"], H=cfg["H"])
    rng = np.random.default_rng(cfg["seed"])

    def rollout_fn(H, B):
        return batch_rollout(env_config, policy, H, B, rng)

    # Build algo_kwargs from cfg['N'] + cfg['algorithm'].
    N = cfg["N"]
    if cfg["algorithm"] == "bai_pg":
        algo_kwargs = {"N1": N, "N2": N}
    elif cfg["algorithm"] == "reinforce":
        algo_kwargs = {"N": N}
    elif cfg["algorithm"] == "ganesh_npg":
        algo_kwargs = {"B1": N, "B2": N, "N_inner": cfg["N_inner"], "beta": cfg["beta"]}
    elif cfg["algorithm"] == "ganesh_mlmc":
        algo_kwargs = {"B_max": cfg.get("B_max", 256), "B_inner": N,
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
        H=cfg["H"],
        rng=rng,
        lr=cfg["lr"],
        algo_kwargs=algo_kwargs,
        verbose=False,
    )
    history["wall_time_sec"] = time.time() - t0
    history["config"] = cfg

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (
        f"{cfg['algorithm']}_{cfg['policy']}_M{cfg['M']}_N{N}_seed{cfg['seed']}.json"
    )
    with open(out_file, "w") as f:
        json.dump(history, f)
    return {"file": str(out_file), "time": history["wall_time_sec"]}


def build_configs(args) -> List[Dict]:
    if args.only_ganesh:
        algos = ["ganesh_npg", "ganesh_mlmc"]
    else:
        algos = ["bai_pg", "reinforce"]
        if args.include_ganesh:
            algos += ["ganesh_npg", "ganesh_mlmc"]
    configs = []
    for algo in algos:
        for M in args.Ms:
            for N in args.Ns:
                for seed in range(args.seeds):
                    configs.append({
                        "algorithm": algo,
                        "policy": args.policy,
                        "M": M,
                        "N": N,
                        "seed": seed,
                        "K": args.K,
                        "H": args.H,
                        "gamma": args.gamma,
                        "L_max": args.L_max,
                        "lr": args.lr,
                        "delta": args.delta,
                        "N_inner": args.N_inner,
                        "beta": args.beta,
                        "B_max": args.B_max,
                        "out_dir": args.out_dir,
                    })
    return configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/sweep")
    p.add_argument("--policy", default="log_linear",
                   choices=["log_linear", "tabular", "nn"])
    p.add_argument("--Ms", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--Ns", type=int, nargs="+", default=[4, 16, 64, 256])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--H", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.9999)
    p.add_argument("--L_max", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--N_inner", type=int, default=5)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--B_max", type=int, default=256)
    p.add_argument("--include_ganesh", action="store_true")
    p.add_argument("--only_ganesh", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()

    configs = build_configs(args)
    n_algos = (2 if args.only_ganesh else (2 + 2 * int(args.include_ganesh)))
    print(f"Sweep: {len(configs)} runs, {args.workers} workers")
    print(f"  algos x Ms x Ns x seeds = "
          f"{n_algos} x {len(args.Ms)} x {len(args.Ns)} x {args.seeds}")
    print(f"  out_dir = {args.out_dir}  |  K = {args.K}  H = {args.H}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "sweep_manifest.json", "w") as f:
        json.dump({"args": vars(args), "configs": configs}, f, indent=2)

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
                tag = f"{cfg['algorithm']:<12} M={cfg['M']} N={cfg['N']:<3} seed={cfg['seed']}"
                print(f"  [{completed:3d}/{len(configs)}] {tag}  {res['time']:6.1f}s  "
                      f"(ETA {eta/60:5.1f} min)")
            except Exception as e:
                tag = f"{cfg['algorithm']} M={cfg['M']} N={cfg['N']} seed={cfg['seed']}"
                print(f"  FAILED  {tag}: {type(e).__name__}: {e}")

    print(f"\nSweep done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
