"""v15 — extended search runner supporting more envs and tweakable objectives.

Same constraint set as v14:
    For MLMC-NPG: K ≈ N (slightly larger OK), 2^B_max = B (B_max = B in code).
    For Bai-PG: B = N1 = N2.
    For REINFORCE: batch = B.
    Total trajectory budget matched.

Supported envs:
    fruit_tree            depth-7 binary tree, M=6 nutrient channels
    dst (6 variants)      10x5/20x10/10x11 × {open, floor}
    bonus_world           9x9 grid M=3
    resource              5x5 grid M=3 with enemy

Tweakable objectives:
    --alpha               alpha-fairness exponent (0 not allowed; user wants nonzero)
    --delta               delta-guard (smaller -> more concave near 0)
    --obj_kind            "alpha_fair" (default) | "alpha_fair_time" | "env_default"

Tweakable algo params:
    --B --N --K           v14-style
    --beta                MLMC inner step-size
    --lr                  Adam learning rate
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
from src.objectives.alpha_fair import AlphaFairObj, AlphaFairTimeObj
from src.objectives.dst import DSTreasureObj
from src.envs.bonus_world import (bonus_world_batch_rollout, BonusWorldObj,
                                    NUM_STATES as BW_NUM_STATES,
                                    NUM_ACTIONS as BW_NUM_ACTIONS,
                                    T_MAX as BW_T_MAX)
from src.envs.resource import (resource_batch_rollout, ResourceObj,
                                NUM_STATES as RS_NUM_STATES,
                                NUM_ACTIONS as RS_NUM_ACTIONS,
                                T_MAX as RS_T_MAX)
from src.envs.dst.multi import (dst_multi_batch_rollout, DSTMultiObj,
                                  NUM_STATES as DSTM_NUM_STATES,
                                  NUM_ACTIONS as DSTM_NUM_ACTIONS,
                                  DST_T_MAX as DSTM_T_MAX, M_REWARD as DSTM_M)
from src.envs.mo_mountain_car import (mo_mountain_car_batch_rollout, MOMountainCarObj,
                                        NUM_STATES as MC_NUM_STATES,
                                        NUM_ACTIONS as MC_NUM_ACTIONS,
                                        T_MAX as MC_T_MAX)
from src.envs.cliff_walking import (cliff_walking_batch_rollout, CliffWalkingObj,
                                      NUM_STATES as CW_NUM_STATES,
                                      NUM_ACTIONS as CW_NUM_ACTIONS,
                                      T_MAX as CW_T_MAX)
from src.envs.dst.concave import (concave_dst_batch_rollout, ConcaveDSTObj,
                                    NUM_STATES as CDST_NUM_STATES,
                                    NUM_ACTIONS as CDST_NUM_ACTIONS,
                                    T_MAX as CDST_T_MAX)
from src.policies.tabular import TabularGridPolicy
from src.policies.nn_state import NNStatePolicy
from alt.run_v13_dst import (
    SPECS as DST_SPECS, dst_rollout, make_obj as make_dst_default_obj,
    dst_reinforce_reward_transform,
    record_greedy_trajectory as record_dst_greedy,
)
from src.train import train


def make_obj(cfg: dict, env_meta: dict):
    """Build the objective per cfg["obj_kind"]."""
    obj_kind = cfg.get("obj_kind", "alpha_fair")
    delta = cfg["delta"]

    if obj_kind == "alpha_fair":
        return AlphaFairObj(alpha=cfg["alpha"], delta=delta)
    if obj_kind == "alpha_fair_time":
        T_max = env_meta.get("t_max", 50)
        return AlphaFairTimeObj(alpha=cfg["alpha"], delta=delta, T_max=T_max)
    if obj_kind == "env_default":
        if env_meta["env_kind"] == "fruit_tree":
            from src.envs.fruit_tree import FruitTreeObj
            return FruitTreeObj(delta=delta)
        if env_meta["env_kind"] == "dst":
            return make_dst_default_obj(env_meta["spec"], delta)
        if env_meta["env_kind"] == "bonus_world":
            return BonusWorldObj(delta=delta, T_max=BW_T_MAX)
        if env_meta["env_kind"] == "resource":
            return ResourceObj(delta=delta, T_max=RS_T_MAX)
        if env_meta["env_kind"] == "dst_multi":
            return DSTMultiObj(delta=delta, T_max=DSTM_T_MAX)
        if env_meta["env_kind"] == "mo_mountain_car":
            return MOMountainCarObj(delta=delta, T_max=MC_T_MAX)
        if env_meta["env_kind"] == "cliff_walking":
            return CliffWalkingObj(delta=delta, T_max=CW_T_MAX)
        if env_meta["env_kind"] == "dst_concave":
            return ConcaveDSTObj(delta=delta, T_max=CDST_T_MAX)
    raise ValueError(f"unknown obj_kind={obj_kind}")


def env_setup(cfg: dict, rng):
    """Returns (policy, obj, rollout_fn, H, env_meta)."""
    if cfg["env_kind"] == "fruit_tree":
        env = FruitTreeEnv(depth=cfg["depth"], seed=42)
        H = env.depth
        ns, na = env.num_states, env.num_actions
        meta = {"env_kind": "fruit_tree", "t_max": H}
        rollout_fn_factory = lambda pol: (lambda H_, B_:
            fruit_tree_batch_rollout({"env": env}, pol, H_, B_, rng))
    elif cfg["env_kind"] == "dst":
        spec = DST_SPECS[cfg["env"]]
        H = spec.t_max
        ns, na = spec.num_states, 4
        meta = {"env_kind": "dst", "spec": spec, "t_max": spec.t_max}
        rollout_fn_factory = lambda pol: (lambda H_, B_:
            dst_rollout(spec, pol, H_, B_, rng))
    elif cfg["env_kind"] == "bonus_world":
        H = BW_T_MAX
        ns, na = BW_NUM_STATES, BW_NUM_ACTIONS
        meta = {"env_kind": "bonus_world", "t_max": BW_T_MAX}
        rollout_fn_factory = lambda pol: (lambda H_, B_:
            bonus_world_batch_rollout({}, pol, H_, B_, rng))
    elif cfg["env_kind"] == "resource":
        H = RS_T_MAX
        ns, na = RS_NUM_STATES, RS_NUM_ACTIONS
        meta = {"env_kind": "resource", "t_max": RS_T_MAX}
        rollout_fn_factory = lambda pol: (lambda H_, B_:
            resource_batch_rollout({}, pol, H_, B_, rng))
    elif cfg["env_kind"] == "dst_multi":
        H = DSTM_T_MAX
        ns, na = DSTM_NUM_STATES, DSTM_NUM_ACTIONS
        meta = {"env_kind": "dst_multi", "t_max": DSTM_T_MAX}
        rollout_fn_factory = lambda pol: (lambda H_, B_:
            dst_multi_batch_rollout({}, pol, H_, B_, rng))
    elif cfg["env_kind"] == "mo_mountain_car":
        H = MC_T_MAX
        ns, na = MC_NUM_STATES, MC_NUM_ACTIONS
        meta = {"env_kind": "mo_mountain_car", "t_max": MC_T_MAX}
        rollout_fn_factory = lambda pol: (lambda H_, B_:
            mo_mountain_car_batch_rollout({}, pol, H_, B_, rng))
    elif cfg["env_kind"] == "cliff_walking":
        H = CW_T_MAX
        ns, na = CW_NUM_STATES, CW_NUM_ACTIONS
        meta = {"env_kind": "cliff_walking", "t_max": CW_T_MAX}
        rollout_fn_factory = lambda pol: (lambda H_, B_:
            cliff_walking_batch_rollout({}, pol, H_, B_, rng))
    elif cfg["env_kind"] == "dst_concave":
        H = CDST_T_MAX
        ns, na = CDST_NUM_STATES, CDST_NUM_ACTIONS
        meta = {"env_kind": "dst_concave", "t_max": CDST_T_MAX}
        rollout_fn_factory = lambda pol: (lambda H_, B_:
            concave_dst_batch_rollout({}, pol, H_, B_, rng))
    else:
        raise ValueError(cfg["env_kind"])

    if cfg["policy"] == "tabular":
        policy = TabularGridPolicy(ns, na)
    else:
        policy = NNStatePolicy(ns, na, hidden=tuple(cfg["hidden"]),
                                activation=cfg.get("activation", "tanh"))
    obj = make_obj(cfg, meta)
    rollout_fn = rollout_fn_factory(policy)
    return policy, obj, rollout_fn, H, meta


def per_iter_traj_mlmc(B: int, N: int) -> float:
    Qmax = max(0, int(math.floor(math.log2(B))))
    e_bk = float(Qmax) + 1.0 / B
    return e_bk + N * B


def derive_K(K_mlmc: int, N: int, B: int) -> tuple[int, int]:
    T = K_mlmc * per_iter_traj_mlmc(B, N)
    return max(1, int(round(T / (2 * B)))), max(1, int(round(T / B)))


def make_configs(args) -> list:
    K_pg, K_rein = derive_K(args.K, args.N, args.B)
    out = []
    for algo in ["ganesh_mlmc", "bai_pg", "reinforce", "scalarized_reinforce"]:
        if args.algos and algo not in args.algos:
            continue
        K_use = args.K if algo == "ganesh_mlmc" else (K_pg if algo == "bai_pg" else K_rein)
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            out.append({**vars(args), "algorithm": algo, "K_use": K_use, "seed": seed})
    return out


def env_label(cfg: dict) -> str:
    if cfg["env_kind"] == "fruit_tree":
        return f"fruit_tree_a{int(round(cfg['alpha']*100)):03d}_d{cfg['depth']}"
    if cfg["env_kind"] == "dst":
        return cfg["env"]
    if cfg["env_kind"] == "mo_mountain_car":
        return "mo_mc"
    return cfg["env_kind"]


def filename(cfg: dict) -> str:
    pol = cfg["policy"]
    if pol == "nn":
        pol = "nn" + "x".join(str(x) for x in cfg["hidden"])
        act = cfg.get("activation", "tanh")
        if act != "tanh":
            pol = f"{pol}_{act}"
    else:
        pol = "tab"
    obj_tag = "" if cfg.get("obj_kind", "alpha_fair") == "alpha_fair" else f"_o{cfg['obj_kind']}"
    return (f"{cfg['algorithm']}_{env_label(cfg)}_{pol}_"
            f"B{cfg['B']}_N{cfg['N']}_K{cfg['K_use']}_"
            f"beta{cfg['beta']:g}_lr{cfg['lr']:g}{obj_tag}_"
            f"v15_{cfg['tag']}_seed{cfg['seed']}.json")


def run_one(cfg: dict) -> dict:
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    rng = np.random.default_rng(cfg["seed"])

    policy, obj, rollout_fn, H, meta = env_setup(cfg, rng)

    B = cfg["B"]; N = cfg["N"]; K_use = cfg["K_use"]
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
    history["config"] = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list, type(None)))}
    history["final_theta"] = [float(x) for x in policy.theta.detach().cpu().numpy()]
    if meta["env_kind"] == "dst":
        history["greedy_trajectory"] = record_dst_greedy(meta["spec"], policy)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / filename(cfg), "w") as f:
        json.dump(history, f)
    return {"file": str(out_dir / filename(cfg)), "time": history["wall_time_sec"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--env_kind", choices=["fruit_tree", "dst", "bonus_world", "resource", "dst_multi", "mo_mountain_car", "cliff_walking", "dst_concave"], required=True)
    p.add_argument("--env", default="dst_open")
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--obj_kind", default="alpha_fair",
                   choices=["alpha_fair", "alpha_fair_time", "env_default"])
    p.add_argument("--policy", choices=["tabular", "nn"], default="nn")
    p.add_argument("--hidden", type=int, nargs="+", default=[32])
    p.add_argument("--activation", default="tanh",
                   choices=["tanh", "relu", "elu", "sigmoid", "gelu", "leaky_relu"])
    p.add_argument("--B", type=int, required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
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

    K_pg, K_rein = derive_K(args.K, args.N, args.B)
    print(f"v15 [{args.tag}] env={args.env_kind} alpha={args.alpha} obj={args.obj_kind} policy={args.policy}{args.hidden} "
          f"B={args.B} N={args.N} K={args.K} -> K_pg={K_pg} K_rein={K_rein} "
          f"beta={args.beta} lr={args.lr} gamma={args.gamma} delta={args.delta}")

    cfgs = make_configs(args)
    if args.skip_existing:
        cfgs = [c for c in cfgs if not (Path(c["out_dir"]) / filename(c)).exists()]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
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
    print(f"v15 [{args.tag}] done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
