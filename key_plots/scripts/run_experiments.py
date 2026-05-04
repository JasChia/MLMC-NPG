"""Run all experiments needed by the figures in this repository.

Generates per-seed JSON files into:
    key_plots/data/fruit_tree/         (3 algos × 4 α × 20 seeds)
    key_plots/data/breakable_bottles/  (3 algos × 3 B values × 20 seeds)

Each file matches the regex consumed by plot_fruit_tree.py / plot_breakable_bottles.py.
Existing JSON files are not overwritten; delete them to force regeneration.

Usage:
    cd key_plots/scripts
    python run_experiments.py                       # everything (slow)
    python run_experiments.py --env fruit_tree      # only fruit tree
    python run_experiments.py --env breakable_bottles --B 4 8 16
    python run_experiments.py --seeds 5             # fewer seeds for a smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

# Repo root = parent of "key_plots"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from src.envs.fruit_tree import FruitTreeEnv, fruit_tree_batch_rollout, M_REWARD as FT_M
from src.envs.breakable_bottles import (
    breakable_bottles_batch_rollout,
    BreakableBottlesObj,
    NUM_STATES as BB_NUM_STATES,
    NUM_ACTIONS as BB_NUM_ACTIONS,
    T_MAX as BB_T_MAX,
)
from src.objectives.alpha_fair import AlphaFairObj
from src.policies.nn_state import NNStatePolicy
from src.train import train


FT_DATA = ROOT / "key_plots" / "data" / "fruit_tree"
BB_DATA = ROOT / "key_plots" / "data" / "breakable_bottles"

FT_ALPHAS = [(0.10, "a010"), (0.25, "a025"), (0.50, "a050"), (1.00, "a100")]
BB_BS     = [4, 8, 16]
ALGORITHMS = ["ganesh_mlmc", "bai_pg", "reinforce"]


@dataclass
class ExperimentConfig:
    env: str
    algo: str
    seed: int
    # Shared
    B: int
    N: int            # MLMC inner iterations / N1 for bai_pg / batch for reinforce
    K_mlmc: int       # MLMC outer iterations (the budget anchor)
    beta: float = 0.05
    lr: float = 0.05
    delta: float = 0.5
    hidden: int = 32
    # Per-env
    alpha: float = None         # fruit_tree
    depth: int = 7              # fruit_tree
    gamma: float = None
    H: int = None
    # Computed
    K: int = None               # outer iterations actually used by `algo`

    def derive(self):
        # Match total trajectory budget across algorithms.
        # MLMC-NPG per outer iter: ~B + N·B trajectories (geometric coupling
        # adds a small expectation; we use the conservative B + N·B upper bound
        # as in the published runs).
        per_iter_mlmc = self.B + self.N * self.B
        T = self.K_mlmc * per_iter_mlmc
        if self.algo == "ganesh_mlmc":
            self.K = self.K_mlmc
        elif self.algo == "bai_pg":
            self.K = max(1, T // (2 * self.B))
        elif self.algo == "reinforce":
            self.K = max(1, T // self.B)
        else:
            raise ValueError(self.algo)


def filename(cfg: ExperimentConfig) -> str:
    if cfg.env == "fruit_tree":
        atag = f"a{int(round(cfg.alpha * 100)):03d}"
        return (f"{cfg.algo}_fruit_tree_{atag}_d{cfg.depth}_nn{cfg.hidden}"
                f"_B{cfg.B}_N{cfg.N}_K{cfg.K}_beta{cfg.beta}_lr{cfg.lr}"
                f"_seed{cfg.seed}.json")
    if cfg.env == "breakable_bottles":
        return (f"{cfg.algo}_breakable_bottles_nn{cfg.hidden}"
                f"_B{cfg.B}_N{cfg.N}_K{cfg.K}_beta{cfg.beta}_lr{cfg.lr}"
                f"_seed{cfg.seed}.json")
    raise ValueError(cfg.env)


def setup_env_policy_obj(cfg: ExperimentConfig):
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.env == "fruit_tree":
        env = FruitTreeEnv(depth=cfg.depth, seed=42)
        policy = NNStatePolicy(num_states=env.num_states, num_actions=env.num_actions,
                                hidden=(cfg.hidden,), device=device)
        obj = AlphaFairObj(alpha=cfg.alpha, delta=cfg.delta)

        env_config = {"env": env}
        H = cfg.depth
        gamma = 0.99

        def rollout_fn(H_, B_):
            return fruit_tree_batch_rollout(env_config, policy, H_, B_, rng)

        return rng, policy, obj, rollout_fn, gamma, H

    if cfg.env == "breakable_bottles":
        policy = NNStatePolicy(num_states=BB_NUM_STATES, num_actions=BB_NUM_ACTIONS,
                                hidden=(cfg.hidden,), device=device)
        obj = BreakableBottlesObj(delta=cfg.delta, T_max=BB_T_MAX)
        env_config = {}
        H = BB_T_MAX
        gamma = 0.9999

        def rollout_fn(H_, B_):
            return breakable_bottles_batch_rollout(env_config, policy, H_, B_, rng)

        return rng, policy, obj, rollout_fn, gamma, H

    raise ValueError(cfg.env)


def algo_kwargs(cfg: ExperimentConfig) -> dict:
    if cfg.algo == "ganesh_mlmc":
        return {"B_max": cfg.B, "B_inner": cfg.B, "N_inner": cfg.N, "beta": cfg.beta}
    if cfg.algo == "bai_pg":
        return {"N1": cfg.B, "N2": cfg.B}
    if cfg.algo == "reinforce":
        # delta_guard=1.0 matches the breakable-bottles runs; FT uses default 1e-3
        delta_guard = 1.0 if cfg.env == "breakable_bottles" else 1e-3
        return {"N": cfg.B, "delta_guard": delta_guard}
    raise ValueError(cfg.algo)


def run_one(cfg: ExperimentConfig, out_dir: Path) -> bool:
    """Execute a single experiment and write its JSON. Returns False if skipped."""
    cfg.derive()
    out = out_dir / filename(cfg)
    if out.exists():
        return False

    rng, policy, obj, rollout_fn, gamma, H = setup_env_policy_obj(cfg)
    history = train(
        algorithm=cfg.algo, policy=policy, rollout_fn=rollout_fn,
        obj=obj, gamma=gamma, K=cfg.K, H=H, rng=rng,
        lr=cfg.lr, algo_kwargs=algo_kwargs(cfg),
    )
    payload = {
        "algo": cfg.algo,
        "env":  cfg.env,
        "config": {k: v for k, v in asdict(cfg).items() if v is not None},
        "f":            history["f"],
        "J_hat":        history["J_hat"],
        "trajectories": history["trajectories"],
        "iter":         history["iter"],
        "seed":         cfg.seed,
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    return True


def fruit_tree_jobs(seeds: Iterable[int]):
    for alpha, _ in FT_ALPHAS:
        for algo in ALGORITHMS:
            for seed in seeds:
                yield ExperimentConfig(
                    env="fruit_tree", algo=algo, seed=seed,
                    B=16, N=20, K_mlmc=40, alpha=alpha,
                )


def breakable_bottles_jobs(seeds: Iterable[int], Bs: Iterable[int]):
    for B in Bs:
        for algo in ALGORITHMS:
            for seed in seeds:
                yield ExperimentConfig(
                    env="breakable_bottles", algo=algo, seed=seed,
                    B=B, N=20, K_mlmc=40,
                )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=["fruit_tree", "breakable_bottles", "all"],
                   default="all")
    p.add_argument("--seeds", type=int, default=20,
                   help="number of seeds per (algo, config); seeds are 0..N-1")
    p.add_argument("--B", type=int, nargs="+", default=BB_BS,
                   help="B values for breakable_bottles (default: 4 8 16)")
    args = p.parse_args()

    seeds = range(args.seeds)
    FT_DATA.mkdir(parents=True, exist_ok=True)
    BB_DATA.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.env in ("fruit_tree", "all"):
        jobs += [(c, FT_DATA) for c in fruit_tree_jobs(seeds)]
    if args.env in ("breakable_bottles", "all"):
        jobs += [(c, BB_DATA) for c in breakable_bottles_jobs(seeds, args.B)]

    print(f"queued {len(jobs)} runs")
    t0 = time.time()
    ran = skipped = 0
    for i, (cfg, out_dir) in enumerate(jobs, 1):
        did_run = run_one(cfg, out_dir)
        if did_run: ran += 1
        else:       skipped += 1
        if i % 10 == 0 or i == len(jobs):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(jobs)}] ran={ran} skipped={skipped} "
                  f"elapsed={elapsed:.1f}s")

    print(f"done: {ran} new runs written, {skipped} skipped (already existed).")


if __name__ == "__main__":
    main()
