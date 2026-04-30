"""Sweep runner that uses batch_rollout_v2 with configurable q_init.

Same CLI as run_sweep.py but adds --q_init (int, "uniform", or None) and
propagates it through the worker to the rollout callable.
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
    # Fresh imports per worker.
    from src.envs import ARRIVAL_RATES
    from src.policy import make_policy
    from src.objectives import AlphaFairness
    from src.train import train
    from alt.envs_v2 import batch_rollout_v2

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if cfg.get("device", "cpu") == "cpu":
        torch.set_num_threads(1)

    env_config = {"M": cfg["M"], "L_max": cfg["L_max"],
                  "lambdas": ARRIVAL_RATES[cfg["M"]]}
    policy = make_policy(cfg["policy"], M=cfg["M"], L_max=cfg["L_max"],
                         device=cfg.get("device", "cpu"))
    obj = AlphaFairness(alpha=2.0, delta=cfg["delta"], H=cfg["H"])
    rng = np.random.default_rng(cfg["seed"])

    q_init = cfg["q_init"]
    if isinstance(q_init, str) and q_init.lstrip("-").isdigit():
        q_init = int(q_init)
    elif q_init in ("none", "None"):
        q_init = None

    def rollout_fn(H, B):
        return batch_rollout_v2(env_config, policy, H, B, rng, q_init=q_init)

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
                        "algorithm": algo, "policy": args.policy, "M": M, "N": N,
                        "seed": seed, "K": args.K, "H": args.H, "gamma": args.gamma,
                        "L_max": args.L_max, "lr": args.lr, "delta": args.delta,
                        "N_inner": args.N_inner, "beta": args.beta,
                        "B_max": args.B_max, "device": args.device,
                        "q_init": args.q_init, "out_dir": args.out_dir,
                    })
    return configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/sweep_v2")
    p.add_argument("--policy", default="log_linear", choices=["log_linear", "tabular", "nn"])
    p.add_argument("--Ms", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--Ns", type=int, nargs="+", default=[4, 16, 64, 256])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--K", type=int, default=500)
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
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--q_init", default="5",
                   help="initial queue state: int, 'uniform', or 'none'")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()

    configs = build_configs(args)
    if args.skip_existing:
        out_dir = Path(args.out_dir)
        kept = [c for c in configs
                if not (out_dir / f"{c['algorithm']}_{c['policy']}_M{c['M']}"
                                  f"_N{c['N']}_seed{c['seed']}.json").exists()]
        print(f"--skip_existing: {len(configs) - len(kept)} present, {len(kept)} to run")
        configs = kept

    print(f"Sweep v2: {len(configs)} runs, {args.workers} workers, q_init={args.q_init}")
    print(f"  out_dir = {args.out_dir} | K = {args.K} H = {args.H}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "sweep_manifest.json", "w") as f:
        json.dump({"args": vars(args), "n_configs": len(configs)}, f, indent=2)

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
                tag = (f"{cfg['algorithm']:<12} {cfg['policy']:<10} "
                       f"M={cfg['M']} N={cfg['N']:<3} seed={cfg['seed']}")
                print(f"  [{completed:3d}/{len(configs)}] {tag}  "
                      f"{res['time']:6.1f}s  (ETA {eta/60:5.1f} min)")
            except Exception as e:
                tag = f"{cfg['algorithm']} M={cfg['M']} N={cfg['N']} seed={cfg['seed']}"
                print(f"  FAILED {tag}: {type(e).__name__}: {e}")

    print(f"\nSweep done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
