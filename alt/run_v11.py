"""v11 unified runner — fair-budget × multi-policy × multi-env sweeps.

Supports envs: dst, dst_large, fruit_tree, bonus_world, resource
Supports policies: tabular, nn (with --hidden specifying widths)

Filename schema:
  {algo}_{env}_{policy_tag}_N{N}_K{K}_{mode}_seed{seed}.json
where policy_tag is "tab" or "nn{h1}_{h2}_..."
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


def _make_env_and_policy(cfg, rng):
    env_kind = cfg["env"]
    if env_kind == "dst":
        from alt.envs_dst import dst_batch_rollout, NUM_STATES, NUM_ACTIONS, DST_T_MAX
        from alt.objectives_dst import DSTreasureObj
        obj = DSTreasureObj(delta=cfg["delta"], T_max=DST_T_MAX)
        H = DST_T_MAX
        ns, na = NUM_STATES, NUM_ACTIONS
        rollout_kwargs = {}
        rollout_fn_factory = lambda policy: (lambda H, B:
            dst_batch_rollout({}, policy, H, B, rng))
    elif env_kind == "dst_large":
        from alt.envs_dst_large import (large_dst_batch_rollout, LargeDSTObj,
                                          NUM_STATES, NUM_ACTIONS, T_MAX)
        obj = LargeDSTObj(delta=cfg["delta"], T_max=T_MAX)
        H = T_MAX
        ns, na = NUM_STATES, NUM_ACTIONS
        rollout_fn_factory = lambda policy: (lambda H, B:
            large_dst_batch_rollout({}, policy, H, B, rng))
    elif env_kind == "dst_original":
        from alt.envs_dst_original import (original_dst_batch_rollout,
                                             OriginalDSTObj,
                                             NUM_STATES, NUM_ACTIONS, T_MAX)
        obj = OriginalDSTObj(delta=cfg["delta"], T_max=T_MAX)
        H = T_MAX
        ns, na = NUM_STATES, NUM_ACTIONS
        rollout_fn_factory = lambda policy: (lambda H, B:
            original_dst_batch_rollout({}, policy, H, B, rng))
    elif env_kind == "fruit_tree":
        from alt.envs_fruit_tree import (FruitTreeEnv, FruitTreeObj,
                                          fruit_tree_batch_rollout)
        env = FruitTreeEnv(depth=cfg.get("depth", 7), seed=42)
        obj = FruitTreeObj(delta=cfg["delta"])
        H = env.depth
        ns, na = env.num_states, env.num_actions
        rollout_fn_factory = lambda policy: (lambda H, B:
            fruit_tree_batch_rollout({"env": env}, policy, H, B, rng))
    elif env_kind == "bonus_world":
        from alt.envs_bonus_world import (bonus_world_batch_rollout, BonusWorldObj,
                                            NUM_STATES, NUM_ACTIONS, T_MAX)
        obj = BonusWorldObj(delta=cfg["delta"], T_max=T_MAX)
        H = T_MAX
        ns, na = NUM_STATES, NUM_ACTIONS
        rollout_fn_factory = lambda policy: (lambda H, B:
            bonus_world_batch_rollout({}, policy, H, B, rng))
    elif env_kind == "resource":
        from alt.envs_resource import (resource_batch_rollout, ResourceObj,
                                          NUM_STATES, NUM_ACTIONS, T_MAX)
        obj = ResourceObj(delta=cfg["delta"], T_max=T_MAX)
        H = T_MAX
        ns, na = NUM_STATES, NUM_ACTIONS
        rollout_fn_factory = lambda policy: (lambda H, B:
            resource_batch_rollout({}, policy, H, B, rng))
    else:
        raise ValueError(f"Unknown env: {env_kind}")

    pol_kind = cfg.get("policy", "tabular")
    if pol_kind == "tabular":
        from alt.policy_grid import TabularGridPolicy
        policy = TabularGridPolicy(ns, na)
    elif pol_kind == "nn":
        from alt.policy_nn_state import NNStatePolicy
        hidden = tuple(cfg["hidden"])
        policy = NNStatePolicy(ns, na, hidden=hidden)
    else:
        raise ValueError(pol_kind)

    return policy, obj, H, rollout_fn_factory(policy)


def run_one(cfg):
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)

    rng = np.random.default_rng(cfg["seed"])
    policy, obj, H, rollout_fn = _make_env_and_policy(cfg, rng)
    use_lowrank_npg = (
        cfg.get("policy") == "nn"
        and cfg["algorithm"] in {"ganesh_npg", "ganesh_mlmc"}
        and policy.d >= cfg.get("lowrank_min_d", 10_000)
    )
    if use_lowrank_npg:
        from alt.train_lowrank import train
        cfg["lowrank_npg"] = True
    else:
        from src.train import train
        cfg["lowrank_npg"] = False

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

    # Compose filename
    pol_tag = cfg.get("policy", "tabular")
    if pol_tag == "nn":
        h = "x".join(str(x) for x in cfg["hidden"])
        pol_tag = f"nn{h}"
    else:
        pol_tag = "tab"
    mode = cfg.get("mode", "ref")
    fname = (f"{cfg['algorithm']}_{cfg['env']}_{pol_tag}_N{N}_K{cfg['K']}_"
             f"{mode}_seed{cfg['seed']}.json")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / fname, "w") as f:
        json.dump(history, f)
    return {"file": str(out_dir / fname), "time": history["wall_time_sec"]}


def configs_from_csv(csv_text: str, base: dict):
    """Parse semicolon-separated config lines:
       env, policy, hidden, algo, N, K, mode
    Each becomes a config dict, multiplied by seeds."""
    rows = []
    for line in csv_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        env, pol, hidden_str, algo, N, K, mode = parts
        hidden = tuple(int(x) for x in hidden_str.split("x")) if hidden_str else None
        rows.append({"env": env, "policy": pol,
                     "hidden": list(hidden) if hidden else None,
                     "algorithm": algo, "N": int(N), "K": int(K), "mode": mode})
    out = []
    for r in rows:
        for seed in range(base["seeds"]):
            cfg = dict(base); cfg.update(r); cfg["seed"] = seed
            out.append(cfg)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config_csv", required=True,
                   help="Path to file with one row per (env, policy, hidden, algo, N, K, mode)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--N_inner", type=int, default=5)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--B_max", type=int, default=64)
    p.add_argument("--delta_reinforce", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    with open(args.config_csv) as f:
        csv_text = f.read()
    base = {
        "seeds": args.seeds, "gamma": args.gamma, "lr": args.lr,
        "delta": args.delta, "N_inner": args.N_inner, "beta": args.beta,
        "B_max": args.B_max, "delta_reinforce": args.delta_reinforce,
        "out_dir": args.out_dir,
    }
    configs = configs_from_csv(csv_text, base)

    if args.skip_existing:
        kept = []
        for c in configs:
            pol_tag = c.get("policy", "tabular")
            if pol_tag == "nn":
                h = "x".join(str(x) for x in c["hidden"])
                pol_tag = f"nn{h}"
            else:
                pol_tag = "tab"
            fname = (f"{c['algorithm']}_{c['env']}_{pol_tag}_N{c['N']}_K{c['K']}_"
                     f"{c['mode']}_seed{c['seed']}.json")
            if not (Path(c['out_dir'])/fname).exists():
                kept.append(c)
        print(f"skip_existing: {len(configs) - len(kept)} present, {len(kept)} to run")
        configs = kept

    print(f"v11 sweep: {len(configs)} runs, {args.workers} workers")
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
                pol_tag = cfg.get("policy", "tabular")
                if pol_tag == "nn":
                    pol_tag = f"nn{'x'.join(str(x) for x in cfg['hidden'])}"
                tag = (f"{cfg['env']:<11} {pol_tag:<10} {cfg['algorithm']:<10} "
                       f"N={cfg['N']:<3} K={cfg['K']:<5} {cfg['mode']:<8} seed={cfg['seed']}")
                print(f"  [{completed:3d}/{len(configs)}] {tag} {res['time']:6.1f}s "
                      f"(ETA {eta/60:5.1f} min)")
            except Exception as e:
                tag = f"{cfg['algorithm']} N={cfg['N']} K={cfg['K']}"
                print(f"  FAILED {tag}: {type(e).__name__}: {e}")

    print(f"\nv11 sweep done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
