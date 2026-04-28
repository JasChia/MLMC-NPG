"""Fair-budget sweep: in each env where MLMC-NPG dominated at small N,
also run Bai-PG and REINFORCE at:
  (a) same N, more K (so cumulative trajectories match MLMC's)
  (b) larger N, same K (so per-iter trajectories match MLMC's)

This is the §10.5 "trajectory-matched" methodology applied to v9 envs.

Goal: confirm that MLMC's win in the §13/§14 plateau tables is not just an
artefact of different per-iter sample counts.

Each result file is named with the CONFIG'S K and N so multiple budgets can
share a single sweep dir without collisions.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch


# (env, target_N_mlmc, K_mlmc, mlmc_rate_per_iter_approx)
# For each MLMC-dominant case: list of (algo, N, K) tuples to also run for fair
# budget. K's chosen so K*per_iter_cost >= K_mlmc * mlmc_rate.
# bai per-iter = 2N; reinforce per-iter = N.
CONFIGS = {
    "v10_fair_resource": {
        "env": "resource",
        "K_mlmc": 300,
        "mlmc_rate": 17,  # per-iter trajectories for MLMC
        "fair_runs": [
            # (algo, N, K, mode_label)
            ("bai_pg",    2,  1300, "moreK"),     # 2*2*1300 = 5200 >= 5100
            ("bai_pg",    8,   300, "largerN"),   # 2*8*300 = 4800 ≈ 5100
            ("reinforce", 2,  2600, "moreK"),     # 2*2600 = 5200
            ("reinforce", 16,  300, "largerN"),   # 16*300 = 4800
            ("bai_pg",    16, 800, "moreK"),      # for MLMC N=16: 86*300 = 25800; bai N=16 → K = 800 (32*800=25600)
            ("bai_pg",    44,  300, "largerN"),   # 2*44*300=26400 ≈ 25800
            ("reinforce", 16, 1600, "moreK"),     # 16*1600=25600
            ("reinforce", 86,  300, "largerN"),   # 86*300=25800
        ],
    },
    "v10_fair_dst_large": {
        "env": "dst_large",
        "K_mlmc": 300,
        "mlmc_rate": 26,
        "fair_runs": [
            ("bai_pg",    4,   975, "moreK"),     # 8*975=7800 ≈ 7800
            ("bai_pg",    13,  300, "largerN"),   # 2*13*300=7800
            ("reinforce", 4,  1950, "moreK"),     # 4*1950=7800
            ("reinforce", 26,  300, "largerN"),
            ("bai_pg",    16,  800, "moreK"),     # 32*800=25600 ≈ 25800 for N=16
            ("bai_pg",    44,  300, "largerN"),
            ("reinforce", 16, 1600, "moreK"),
            ("reinforce", 86,  300, "largerN"),
        ],
    },
}


def run_one(cfg):
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
    elif env_kind == "dst_large":
        from alt.envs_dst_large import (large_dst_batch_rollout, LargeDSTObj,
                                         NUM_STATES, NUM_ACTIONS, T_MAX)
        from alt.policy_grid import TabularGridPolicy
        policy = TabularGridPolicy(NUM_STATES, NUM_ACTIONS)
        obj = LargeDSTObj(delta=cfg["delta"], T_max=T_MAX)

        def rollout_fn(H, B):
            return large_dst_batch_rollout({}, policy, H, B, rng)

        H = T_MAX
    else:
        raise ValueError(env_kind)

    N = cfg["N"]
    if cfg["algorithm"] == "bai_pg":
        algo_kwargs = {"N1": N, "N2": N}
    elif cfg["algorithm"] == "reinforce":
        algo_kwargs = {"N": N, "delta_guard": cfg.get("delta_reinforce", 1.0)}
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
    fname = (f"{cfg['algorithm']}_{env_kind}_N{N}_K{cfg['K']}_{cfg['mode']}_seed{cfg['seed']}.json")
    with open(out_dir / fname, "w") as f:
        json.dump(history, f)
    return {"file": str(out_dir / fname), "time": history["wall_time_sec"]}


def build_configs(args, key, spec):
    out = []
    out_dir = f"results/sweep_{key}"
    for (algo, N, K, mode) in spec["fair_runs"]:
        for seed in range(args.seeds):
            out.append({
                "algorithm": algo, "N": N, "K": K, "mode": mode,
                "env": spec["env"], "seed": seed,
                "gamma": args.gamma, "lr": args.lr, "delta": args.delta,
                "delta_reinforce": args.delta_reinforce,
                "out_dir": out_dir,
            })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--delta_reinforce", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--keys", nargs="+", default=list(CONFIGS.keys()))
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    all_configs = []
    for key in args.keys:
        if key not in CONFIGS:
            raise ValueError(f"unknown key {key}")
        all_configs.extend(build_configs(args, key, CONFIGS[key]))

    if args.skip_existing:
        kept = []
        for c in all_configs:
            fname = (f"{c['algorithm']}_{c['env']}_N{c['N']}_K{c['K']}_{c['mode']}_seed{c['seed']}.json")
            if not (Path(c['out_dir'])/fname).exists():
                kept.append(c)
        print(f"skip_existing: {len(all_configs) - len(kept)} present, {len(kept)} to run")
        all_configs = kept

    print(f"Fair-budget sweep: {len(all_configs)} runs, {args.workers} workers")

    for c in all_configs[:5]:
        print(f"  e.g. {c['algorithm']} N={c['N']} K={c['K']} mode={c['mode']} env={c['env']}")

    t0 = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, cfg): cfg for cfg in all_configs}
        for fut in as_completed(futures):
            cfg = futures[fut]
            try:
                res = fut.result()
                completed += 1
                elapsed = time.time() - t0
                avg = elapsed / completed
                eta = avg * (len(all_configs) - completed)
                tag = (f"{cfg['env']:<10} {cfg['algorithm']:<10} N={cfg['N']:<3} K={cfg['K']:<5} "
                       f"{cfg['mode']:<8} seed={cfg['seed']}")
                print(f"  [{completed:3d}/{len(all_configs)}] {tag}  "
                      f"{res['time']:6.1f}s  (ETA {eta/60:5.1f} min)")
            except Exception as e:
                tag = f"{cfg['algorithm']} N={cfg['N']} K={cfg['K']}"
                print(f"  FAILED {tag}: {type(e).__name__}: {e}")

    print(f"\nFair-budget sweep done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
