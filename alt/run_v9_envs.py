"""Unified runner for the four new episodic envs:
  - resource         : Resource Gathering (5x5, M=3 sparse, tabular d=404)
  - dst_multi        : Multi-treasure DST (10x5, M=6 sparse, tabular d=204)
  - dst_large        : Larger DST (20x10, M=2, tabular d=804)
  - episodic_queue   : Episodic queueing (terminate on first arrival to queue 0)

All envs are wired through src.train.train without touching src/ algorithms.
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

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)

    env_kind = cfg["env"]
    rng = np.random.default_rng(cfg["seed"])

    if env_kind == "resource":
        from alt.envs_resource import (resource_batch_rollout, ResourceObj,
                                        NUM_STATES, NUM_ACTIONS, T_MAX)
        from alt.policy_grid import TabularGridPolicy
        policy = TabularGridPolicy(NUM_STATES, NUM_ACTIONS)
        obj = ResourceObj(delta=cfg["delta"], T_max=T_MAX)

        def rollout_fn(H, B):
            return resource_batch_rollout({}, policy, H, B, rng)

        H = T_MAX
    elif env_kind == "dst_multi":
        from alt.envs_dst_multi import (dst_multi_batch_rollout, DSTMultiObj,
                                         NUM_STATES, NUM_ACTIONS, DST_T_MAX)
        from alt.policy_grid import TabularGridPolicy
        policy = TabularGridPolicy(NUM_STATES, NUM_ACTIONS)
        obj = DSTMultiObj(delta=cfg["delta"], T_max=DST_T_MAX)

        def rollout_fn(H, B):
            return dst_multi_batch_rollout({}, policy, H, B, rng)

        H = DST_T_MAX
    elif env_kind == "dst_large":
        from alt.envs_dst_large import (large_dst_batch_rollout, LargeDSTObj,
                                         NUM_STATES, NUM_ACTIONS, T_MAX)
        from alt.policy_grid import TabularGridPolicy
        policy = TabularGridPolicy(NUM_STATES, NUM_ACTIONS)
        obj = LargeDSTObj(delta=cfg["delta"], T_max=T_MAX)

        def rollout_fn(H, B):
            return large_dst_batch_rollout({}, policy, H, B, rng)

        H = T_MAX
    elif env_kind == "episodic_queue":
        from alt.envs_episodic_queue import episodic_queue_batch_rollout
        from alt.envs_v2 import ARRIVAL_RATES, ARRIVAL_RATES_CHALLENGING
        from alt.objectives_v2 import GeneralAlphaFairness
        from src.policy import make_policy as _src_make_policy

        policy = _src_make_policy(cfg["policy"], M=cfg["M"], L_max=cfg["L_max"],
                                  device="cpu")
        arrivals_kind = cfg.get("arrivals", "standard")
        lambdas = (ARRIVAL_RATES_CHALLENGING[cfg["M"]] if arrivals_kind == "challenging"
                   else ARRIVAL_RATES[cfg["M"]])
        env_config = {"M": cfg["M"], "L_max": cfg["L_max"], "lambdas": lambdas,
                      "T_max": cfg["H"]}
        obj = GeneralAlphaFairness(alpha=cfg["alpha"], delta=cfg["delta"], H=cfg["H"])

        q_init = cfg.get("q_init", 5)
        if isinstance(q_init, str) and q_init.lstrip("-").isdigit():
            q_init = int(q_init)
        elif q_init in ("none", "None"):
            q_init = None

        def rollout_fn(H, B):
            return episodic_queue_batch_rollout(env_config, policy, H, B, rng, q_init=q_init)

        H = cfg["H"]
    else:
        raise ValueError(f"Unknown env: {env_kind}")

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
        gamma=cfg["gamma"], K=cfg["K"], H=H,
        rng=rng, lr=cfg["lr"], algo_kwargs=algo_kwargs, verbose=False,
    )
    history["wall_time_sec"] = time.time() - t0
    history["config"] = cfg

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = _filename(cfg)
    with open(out_dir / fname, "w") as f:
        json.dump(history, f)
    return {"file": str(out_dir / fname), "time": history["wall_time_sec"]}


def _filename(cfg) -> str:
    env = cfg["env"]
    if env in ("resource", "dst_multi", "dst_large"):
        return f"{cfg['algorithm']}_{env}_N{cfg['N']}_seed{cfg['seed']}.json"
    if env == "episodic_queue":
        return (f"{cfg['algorithm']}_{cfg['policy']}_{env}_M{cfg['M']}_N{cfg['N']}"
                f"_alpha{cfg['alpha']:.1f}_seed{cfg['seed']}.json")
    return f"{cfg['algorithm']}_{env}_N{cfg['N']}_seed{cfg['seed']}.json"


def build_configs(args) -> List[Dict]:
    configs = []
    for algo in args.algos:
        for N in args.Ns:
            for seed in range(args.seeds):
                c = {
                    "algorithm": algo, "N": N, "seed": seed,
                    "env": args.env, "K": args.K, "gamma": args.gamma,
                    "lr": args.lr, "delta": args.delta,
                    "N_inner": args.N_inner, "beta": args.beta,
                    "B_max": args.B_max,
                    "delta_reinforce": args.delta_reinforce,
                    "out_dir": args.out_dir,
                }
                if args.env == "episodic_queue":
                    c.update({
                        "policy": args.policy, "M": args.M, "L_max": args.L_max,
                        "alpha": args.alpha, "H": args.H,
                        "q_init": args.q_init,
                        "arrivals": args.arrivals,
                    })
                configs.append(c)
    return configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True,
                   choices=["resource", "dst_multi", "dst_large", "episodic_queue"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--Ns", type=int, nargs="+", default=[2, 4, 16])
    p.add_argument("--algos", nargs="+",
                   default=["bai_pg", "reinforce", "ganesh_mlmc"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--K", type=int, default=300)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--N_inner", type=int, default=5)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--B_max", type=int, default=64)
    p.add_argument("--delta_reinforce", type=float, default=1.0)

    # Episodic-queue-specific knobs
    p.add_argument("--policy", default="log_linear")
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--L_max", type=int, default=10)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--H", type=int, default=500)
    p.add_argument("--q_init", default="5")
    p.add_argument("--arrivals", default="standard", choices=["standard", "challenging"])

    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    configs = build_configs(args)
    if args.skip_existing:
        out_dir = Path(args.out_dir)
        kept = [c for c in configs if not (out_dir / _filename(c)).exists()]
        print(f"skip_existing: {len(configs) - len(kept)} present, {len(kept)} to run")
        configs = kept

    print(f"Sweep {args.env}: {len(configs)} runs, {args.workers} workers")
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

    print(f"\nSweep {args.env} done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
