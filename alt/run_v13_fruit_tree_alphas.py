"""v13 Fruit Tree sweep — multiple alpha-fairness levels.

Same depth-7 binary tree as alt/envs_fruit_tree.py but objective is
AlphaFairObj(alpha) over the 6 nutrient channels (delta=0.5).

Alphas swept: 0 (utilitarian), 0.25, 0.5 (~current sqrt), 1 (log-prop), 2 (-1/x).

v13 fair-budget hyperparams (~5k traj each):
    MLMC ref     : N=B_inner=4, K=35,   N_inner=35, B_max=16
    Bai-PG       : N1=N2=4, K=612
    REINFORCE    : N=4, K=1225  (no reward_transform — fruit tree leaf rewards
                   are non-negative so eq.(38) denominators stay positive
                   given delta_guard>0; first leaf hit gives strictly
                   positive cum_disc within the first few steps)
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from alt.envs_fruit_tree import FruitTreeEnv, fruit_tree_batch_rollout
from alt.objectives_alpha_fair import AlphaFairObj
from alt.policy_grid import TabularGridPolicy
from alt.scalarized_reinforce import train_scalarized_reinforce
from src.train import train


ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def base_rows():
    return [
        ("ganesh_mlmc",           4, 35,   "v13"),
        ("bai_pg",                4, 612,  "v13"),
        ("reinforce",             4, 1225, "v13"),
        ("scalarized_reinforce",  4, 1225, "v13"),
    ]


def configs(args) -> list:
    out = []
    for alpha in args.alphas:
        for algo, N, K, mode in base_rows():
            K = int(K * args.K_multiplier)
            if args.algos and algo not in args.algos:
                continue
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                out.append({
                    "alpha": alpha, "algorithm": algo, "N": N, "K": K, "mode": mode,
                    "seed": seed, "depth": args.depth,
                    "gamma": args.gamma, "lr": args.lr, "delta": args.delta,
                    "N_inner": args.N_inner, "beta": args.beta, "B_max": args.B_max,
                    "delta_reinforce": args.delta_reinforce,
                    "out_dir": args.out_dir,
                })
    return out


def alpha_tag(alpha: float) -> str:
    # Filesystem-safe label: 0.25 -> "025", 1.0 -> "100", 2.0 -> "200".
    return f"a{int(round(alpha * 100)):03d}"


def filename(cfg: dict) -> str:
    return (
        f"{cfg['algorithm']}_fruit_tree_{alpha_tag(cfg['alpha'])}_tab_"
        f"N{cfg['N']}_K{cfg['K']}_{cfg['mode']}_seed{cfg['seed']}.json"
    )


def run_one(cfg: dict) -> dict:
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)

    rng = np.random.default_rng(cfg["seed"])
    env = FruitTreeEnv(depth=cfg["depth"], seed=42)
    obj = AlphaFairObj(alpha=cfg["alpha"], delta=cfg["delta"])
    policy = TabularGridPolicy(env.num_states, env.num_actions)

    rollout_fn = lambda H, B: fruit_tree_batch_rollout({"env": env}, policy, H, B, rng)

    N = cfg["N"]
    t0 = time.time()
    if cfg["algorithm"] == "scalarized_reinforce":
        history = train_scalarized_reinforce(
            policy=policy, rollout_fn=rollout_fn, obj=obj,
            gamma=cfg["gamma"], K=cfg["K"], H=env.depth, rng=rng,
            lr=cfg["lr"], N=N, verbose=False,
        )
    else:
        if cfg["algorithm"] == "bai_pg":
            algo_kwargs = {"N1": N, "N2": N}
        elif cfg["algorithm"] == "reinforce":
            algo_kwargs = {"N": N, "delta_guard": cfg["delta_reinforce"]}
        elif cfg["algorithm"] == "ganesh_mlmc":
            algo_kwargs = {"B_max": cfg["B_max"], "B_inner": N,
                           "N_inner": cfg["N_inner"], "beta": cfg["beta"]}
        else:
            raise ValueError(cfg["algorithm"])
        history = train(
            algorithm=cfg["algorithm"], policy=policy, rollout_fn=rollout_fn, obj=obj,
            gamma=cfg["gamma"], K=cfg["K"], H=env.depth, rng=rng,
            lr=cfg["lr"], algo_kwargs=algo_kwargs, verbose=False,
        )
    history["wall_time_sec"] = time.time() - t0
    history["config"] = cfg
    history["final_theta"] = [float(x) for x in policy.theta.detach().cpu().numpy()]

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / filename(cfg), "w") as f:
        json.dump(history, f)
    return {"file": str(out_dir / filename(cfg)), "time": history["wall_time_sec"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    p.add_argument("--algos", nargs="+", default=None)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--N_inner", type=int, default=35)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--B_max", type=int, default=16)
    p.add_argument("--delta_reinforce", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--K_multiplier", type=float, default=1.0,
                   help="multiplicative scale for all K values")
    args = p.parse_args()

    cfgs = configs(args)
    if args.skip_existing:
        cfgs = [c for c in cfgs if not (Path(c["out_dir"]) / filename(c)).exists()]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "sweep_manifest.json", "w") as f:
        json.dump({"args": vars(args), "n": len(cfgs)}, f, indent=2)

    print(f"v13 fruit-tree-alpha sweep: {len(cfgs)} runs, {args.workers} workers")
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
                print(f"  [{done:3d}/{len(cfgs)}] alpha={cfg['alpha']:<5} "
                      f"{cfg['algorithm']:<12} N={cfg['N']:<2} K={cfg['K']:<5} "
                      f"seed={cfg['seed']} {res['time']:6.1f}s (ETA {eta/60:5.1f} min)")
            except Exception as e:
                print(f"  FAILED alpha={cfg['alpha']} {cfg['algorithm']} seed={cfg['seed']}: "
                      f"{type(e).__name__}: {e}")

    print(f"v13 fruit-tree-alpha sweep done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
