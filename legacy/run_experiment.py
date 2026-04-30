"""Run a single (algorithm, M, seed) configuration and save history to JSON.

Example:
    python run_experiment.py --algorithm bai_pg --M 2 --seed 0 --K 200
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.envs import ARRIVAL_RATES, batch_rollout
from src.policy import make_policy
from src.objectives import AlphaFairness
from src.train import train


def build_rollout_fn(env_config, policy, rng):
    def rollout_fn(H, B):
        return batch_rollout(env_config, policy, H, B, rng)
    return rollout_fn


def algo_kwargs_from_args(args):
    if args.algorithm == "bai_pg":
        return {"N1": args.N1, "N2": args.N2}
    if args.algorithm == "ganesh_npg":
        return {"B1": args.B1, "B2": args.B2,
                "N_inner": args.N_inner, "beta": args.beta}
    if args.algorithm == "ganesh_mlmc":
        return {"B_max": args.B_max, "B_inner": args.B_inner,
                "N_inner": args.N_inner, "beta": args.beta}
    if args.algorithm == "reinforce":
        return {"N": args.N1}
    raise ValueError(args.algorithm)


def run_one(args) -> dict:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_config = {
        "M": args.M,
        "L_max": args.L_max,
        "lambdas": ARRIVAL_RATES[args.M],
    }
    policy = make_policy(args.policy, M=args.M, L_max=args.L_max, device=args.device)
    obj = AlphaFairness(alpha=2.0, delta=args.delta, H=args.H)
    rng = np.random.default_rng(args.seed)
    rollout_fn = build_rollout_fn(env_config, policy, rng)

    t0 = time.time()
    history = train(
        algorithm=args.algorithm,
        policy=policy,
        rollout_fn=rollout_fn,
        obj=obj,
        gamma=args.gamma,
        K=args.K,
        H=args.H,
        rng=rng,
        lr=args.lr,
        algo_kwargs=algo_kwargs_from_args(args),
        verbose=args.verbose,
    )
    history["wall_time_sec"] = time.time() - t0
    history["config"] = vars(args).copy()
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", required=True,
                   choices=["bai_pg", "ganesh_npg", "ganesh_mlmc", "reinforce"])
    p.add_argument("--policy", default="log_linear",
                   choices=["log_linear", "tabular", "nn"])
    p.add_argument("--M", type=int, default=2, choices=[2, 4, 8])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--H", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.9999)
    p.add_argument("--L_max", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--delta", type=float, default=1e-3,
                   help="positivity guard in alpha-fairness denominator")
    # Bai-PG
    p.add_argument("--N1", type=int, default=16)
    p.add_argument("--N2", type=int, default=16)
    # Ganesh vanilla NPG
    p.add_argument("--B1", type=int, default=16)
    p.add_argument("--B2", type=int, default=16)
    # Ganesh MLMC-NPG
    p.add_argument("--B_max", type=int, default=256)
    p.add_argument("--B_inner", type=int, default=16)
    # Shared NPG inner loop
    p.add_argument("--N_inner", type=int, default=5)
    p.add_argument("--beta", type=float, default=0.01)
    # Output
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    history = run_one(args)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_file = Path(args.out_dir) / f"{args.algorithm}_M{args.M}_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history to {out_file} (wall time {history['wall_time_sec']:.1f} s)")


if __name__ == "__main__":
    main()
