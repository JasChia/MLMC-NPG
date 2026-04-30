"""v14 — search for MLMC-NPG wins under user-specified constraints.

Constraint set (from user prompt):
    For MLMC-NPG:
        - K can be a bit greater than N (inner iters), but should be similar
        - 2^B_max should equal B  (i.e., the codebase's B_max param == inner B)
    For Bai-PG (PG):
        - B = N_1 = N_2  (same batch size both halves)
    For REINFORCE / scalarized REINFORCE:
        - batch size N_codebase = B
    Total trajectory budget matched across algorithms.

Per-iter trajectory counts:
    MLMC:        E[B_k] + N_inner * B_inner
                 = (~log2(B_max) + 1/B_max) + N * B   when B_inner = B_max = B
                 ≈ N * B                              for moderate sizes
    Bai-PG:      2 * B
    REINFORCE:   B

For matched total T:
    K_MLMC outer with K ≈ N (inner) → ~K*(log2(B) + N*B) trajectories
    K_PG     = T / (2*B)
    K_REIN   = T / B

This runner takes a single (B, N, K, env, policy) config and runs all 4
algorithms at fair budget. Saves to results/sweep_v14_<tag>/.

Each tag describes the configuration concisely.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from src.envs.fruit_tree import FruitTreeEnv, fruit_tree_batch_rollout
from src.objectives.alpha_fair import AlphaFairObj
from src.policies.tabular import TabularGridPolicy
from src.policies.nn_state import NNStatePolicy
from alt.run_v13_dst import (
    SPECS as DST_SPECS, SEA_FLOORS as DST_SEA_FLOORS,
    dst_rollout, make_obj as make_dst_obj,
    dst_reinforce_reward_transform,
    record_greedy_trajectory as record_dst_greedy,
)
from src.train import train


def _env_setup(cfg: dict, rng):
    """Returns (policy, obj, rollout_fn, H, num_states, env_meta_for_traj)."""
    if cfg["env_kind"] == "fruit_tree":
        env = FruitTreeEnv(depth=cfg["depth"], seed=42)
        obj = AlphaFairObj(alpha=cfg["alpha"], delta=cfg["delta"])
        H = env.depth
        ns, na = env.num_states, env.num_actions
        if cfg["policy"] == "tabular":
            policy = TabularGridPolicy(ns, na)
        else:
            policy = NNStatePolicy(ns, na, hidden=tuple(cfg["hidden"]))
        rollout_fn = lambda H_, B_: fruit_tree_batch_rollout({"env": env}, policy, H_, B_, rng)
        return policy, obj, rollout_fn, H, ns, {"env_kind": "fruit_tree"}
    elif cfg["env_kind"] == "dst":
        spec = DST_SPECS[cfg["env"]]
        obj = make_dst_obj(spec, cfg["delta"])
        H = spec.t_max
        ns, na = spec.num_states, 4
        if cfg["policy"] == "tabular":
            policy = TabularGridPolicy(ns, na)
        else:
            policy = NNStatePolicy(ns, na, hidden=tuple(cfg["hidden"]))
        rollout_fn = lambda H_, B_: dst_rollout(spec, policy, H_, B_, rng)
        return policy, obj, rollout_fn, H, ns, {"env_kind": "dst", "spec": spec}
    else:
        raise ValueError(cfg["env_kind"])


def _per_iter_traj_mlmc(B: int, N: int) -> float:
    """Expected trajectory cost per outer MLMC iter when B_max = B (count cap).
    E[B_k] = sum_{q=1..Q_max} 2^q * 2^-q + 1/(2^Q_max) where Q_max = floor(log2(B)).
            = Q_max + 1/B
    """
    Qmax = max(0, int(math.floor(math.log2(B))))
    e_bk = float(Qmax) + 1.0 / B
    return e_bk + N * B


def derive_K_pg_reinforce(K_mlmc: int, N: int, B: int) -> tuple[int, int]:
    """Solve K_PG and K_REIN to match MLMC's total budget."""
    T = K_mlmc * _per_iter_traj_mlmc(B, N)
    K_pg = max(1, int(round(T / (2 * B))))
    K_rein = max(1, int(round(T / B)))
    return K_pg, K_rein


def make_configs(args) -> list:
    K_pg, K_rein = derive_K_pg_reinforce(args.K, args.N, args.B)
    out = []
    for algo in ["ganesh_mlmc", "bai_pg", "reinforce", "scalarized_reinforce"]:
        if args.algos and algo not in args.algos:
            continue
        if algo == "ganesh_mlmc":
            K_use = args.K
        elif algo == "bai_pg":
            K_use = K_pg
        else:
            K_use = K_rein
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            cfg = dict(args.__dict__)
            cfg.update({"algorithm": algo, "K_use": K_use, "seed": seed})
            out.append(cfg)
    return out


def filename(cfg: dict) -> str:
    if cfg["env_kind"] == "fruit_tree":
        env_tag = f"fruit_tree_a{int(round(cfg['alpha']*100)):03d}_d{cfg['depth']}"
    else:
        env_tag = cfg["env"]
    pol = cfg["policy"]
    if pol == "nn":
        pol = "nn" + "x".join(str(x) for x in cfg["hidden"])
    else:
        pol = "tab"
    return (f"{cfg['algorithm']}_{env_tag}_{pol}_"
            f"B{cfg['B']}_N{cfg['N']}_K{cfg['K_use']}_"
            f"beta{cfg['beta']:g}_lr{cfg['lr']:g}_"
            f"v14_{cfg['tag']}_seed{cfg['seed']}.json")


def run_one(cfg: dict) -> dict:
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    rng = np.random.default_rng(cfg["seed"])

    policy, obj, rollout_fn, H, ns, env_meta = _env_setup(cfg, rng)

    B = cfg["B"]
    N = cfg["N"]
    K_use = cfg["K_use"]

    t0 = time.time()
    if cfg["algorithm"] == "bai_pg":
        algo_kwargs = {"N1": B, "N2": B}
    elif cfg["algorithm"] == "reinforce":
        kw = {"N": B, "delta_guard": cfg["delta_reinforce"]}
        if cfg["env_kind"] == "dst":
            kw["reward_transform"] = dst_reinforce_reward_transform
        algo_kwargs = kw
    elif cfg["algorithm"] == "scalarized_reinforce":
        algo_kwargs = {"N": B}
    elif cfg["algorithm"] == "ganesh_mlmc":
        algo_kwargs = {"B_max": B, "B_inner": B,
                       "N_inner": N, "beta": cfg["beta"]}
    else:
        raise ValueError(cfg["algorithm"])
    history = train(
        algorithm=cfg["algorithm"], policy=policy, rollout_fn=rollout_fn,
        obj=obj, gamma=cfg["gamma"], K=K_use, H=H, rng=rng,
        lr=cfg["lr"], algo_kwargs=algo_kwargs, verbose=False,
    )
    history["wall_time_sec"] = time.time() - t0
    history["config"] = cfg
    history["final_theta"] = [float(x) for x in policy.theta.detach().cpu().numpy()]
    if env_meta["env_kind"] == "dst":
        history["greedy_trajectory"] = record_dst_greedy(env_meta["spec"], policy)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / filename(cfg), "w") as f:
        json.dump(history, f)
    return {"file": str(out_dir / filename(cfg)), "time": history["wall_time_sec"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", required=True, help="short tag for filenames")
    p.add_argument("--env_kind", choices=["fruit_tree", "dst"], required=True)
    p.add_argument("--env", default="dst_open",
                   help="DST env if env_kind=dst")
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--policy", choices=["tabular", "nn"], default="nn")
    p.add_argument("--hidden", type=int, nargs="+", default=[32])
    p.add_argument("--B", type=int, required=True,
                   help="MLMC inner B = B_max (count cap), Bai N1=N2=B, REINFORCE N=B")
    p.add_argument("--N", type=int, required=True,
                   help="MLMC N_inner")
    p.add_argument("--K", type=int, required=True,
                   help="MLMC outer iterations (≈ N per user constraint)")
    p.add_argument("--algos", nargs="+", default=None)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--delta_reinforce", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    K_pg, K_rein = derive_K_pg_reinforce(args.K, args.N, args.B)
    print(f"v14 search [{args.tag}] | env={args.env_kind} env={args.env if args.env_kind=='dst' else f'depth{args.depth}_a{args.alpha}'} policy={args.policy}")
    print(f"  B={args.B}  N={args.N}  K={args.K}  -> K_pg={K_pg}  K_rein={K_rein}")
    print(f"  MLMC per-iter ~ {_per_iter_traj_mlmc(args.B, args.N):.1f} traj, total ~ {args.K * _per_iter_traj_mlmc(args.B, args.N):.0f}")
    print(f"  Bai-PG total = {2 * args.B * K_pg}, REINFORCE total = {args.B * K_rein}")

    cfgs = make_configs(args)
    if args.skip_existing:
        cfgs = [c for c in cfgs if not (Path(c["out_dir"]) / filename(c)).exists()]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / f"sweep_manifest_{args.tag}.json", "w") as f:
        json.dump({"args": vars(args), "K_pg": K_pg, "K_rein": K_rein, "n": len(cfgs)}, f, indent=2)

    print(f"  {len(cfgs)} runs, {args.workers} workers")
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
                print(f"  [{done:3d}/{len(cfgs)}] {cfg['algorithm']:<22} "
                      f"K={cfg['K_use']:<5} seed={cfg['seed']} "
                      f"{res['time']:6.1f}s (ETA {eta/60:5.1f} min)")
            except Exception as e:
                print(f"  FAILED {cfg['algorithm']} seed={cfg['seed']}: "
                      f"{type(e).__name__}: {e}")
    print(f"v14 [{args.tag}] done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
