"""MLMC-NPG showcase sweep runner: generalized alpha-fairness + MLMC-focused comparison.

Compared to run_sweep_v2.py:
- Uses alt.objectives_v2.GeneralAlphaFairness so alpha can be > 2
- Passes --alpha through workers
- Otherwise identical interface, separate out_dir

The point of this runner is to sweep (algorithm, alpha, N) at small N on M=8
where the bias in Bai-PG's plug-in gradient estimator bites hardest. We
explicitly match per-outer-iteration trajectory counts where possible, and
the downstream plotter uses cumulative trajectories on the x-axis.
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
    from src.policy import make_policy as _src_make_policy
    from src.train import train
    from alt.train_v2 import train_v2
    from alt.envs_v2 import batch_rollout_v2, ARRIVAL_RATES
    from alt.objectives_v2 import GeneralAlphaFairness, MinFairness

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if cfg.get("device", "cpu") == "cpu":
        torch.set_num_threads(1)

    env_config = {"M": cfg["M"], "L_max": cfg["L_max"],
                  "lambdas": ARRIVAL_RATES[cfg["M"]]}
    if cfg["policy"] == "nn":
        # Use compat NN policy (per-sample autograd loop) on torch < 2.0.
        from alt.policy_nn_compat import NNPolicyCompat
        policy = NNPolicyCompat(M=cfg["M"], L_max=cfg["L_max"],
                                device=cfg.get("device", "cpu"))
    else:
        policy = _src_make_policy(cfg["policy"], M=cfg["M"], L_max=cfg["L_max"],
                                  device=cfg.get("device", "cpu"))
    obj_kind = cfg.get("objective", "alpha")
    if obj_kind == "alpha":
        obj = GeneralAlphaFairness(alpha=cfg["alpha"], delta=cfg["delta"], H=cfg["H"])
    elif obj_kind == "minfair":
        obj = MinFairness(H=cfg["H"])
    else:
        raise ValueError(f"Unknown objective: {obj_kind}")
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
        algo_kwargs = {"N": N, "delta_guard": cfg.get("delta_reinforce", 1e-3)}
    elif cfg["algorithm"] == "ganesh_npg":
        algo_kwargs = {"B1": N, "B2": N,
                       "N_inner": cfg["N_inner"], "beta": cfg["beta"]}
    elif cfg["algorithm"] == "ganesh_mlmc":
        algo_kwargs = {"B_max": cfg.get("B_max", 128), "B_inner": N,
                       "N_inner": cfg["N_inner"], "beta": cfg["beta"]}
    else:
        raise ValueError(cfg["algorithm"])

    t0 = time.time()
    grad_clip = cfg.get("grad_clip")
    if grad_clip is not None and grad_clip > 0:
        history = train_v2(
            algorithm=cfg["algorithm"],
            policy=policy, rollout_fn=rollout_fn, obj=obj,
            gamma=cfg["gamma"], K=cfg["K"], H=cfg["H"],
            rng=rng, lr=cfg["lr"], algo_kwargs=algo_kwargs, verbose=False,
            grad_clip=grad_clip,
        )
    else:
        history = train(
            algorithm=cfg["algorithm"],
            policy=policy, rollout_fn=rollout_fn, obj=obj,
            gamma=cfg["gamma"], K=cfg["K"], H=cfg["H"],
            rng=rng, lr=cfg["lr"], algo_kwargs=algo_kwargs, verbose=False,
        )
    history["wall_time_sec"] = time.time() - t0
    history["config"] = cfg

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (
        f"{cfg['algorithm']}_{cfg['policy']}_M{cfg['M']}_N{N}"
        f"_alpha{cfg['alpha']:.1f}_seed{cfg['seed']}.json"
    )
    with open(out_file, "w") as f:
        json.dump(history, f)
    return {"file": str(out_file), "time": history["wall_time_sec"]}


def build_configs(args) -> List[Dict]:
    algos = args.algos
    configs = []
    for algo in algos:
        for alpha in args.alphas:
            for M in args.Ms:
                for N in args.Ns:
                    for seed in range(args.seeds):
                        configs.append({
                            "algorithm": algo, "policy": args.policy,
                            "M": M, "N": N, "seed": seed, "alpha": alpha,
                            "K": args.K, "H": args.H, "gamma": args.gamma,
                            "L_max": args.L_max, "lr": args.lr,
                            "delta": args.delta, "N_inner": args.N_inner,
                            "beta": args.beta, "B_max": args.B_max,
                            "device": args.device, "q_init": args.q_init,
                            "delta_reinforce": args.delta_reinforce,
                            "grad_clip": args.grad_clip,
                            "objective": args.objective,
                            "out_dir": args.out_dir,
                        })
    return configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/sweep_v3_mlmc")
    p.add_argument("--policy", default="log_linear")
    p.add_argument("--Ms", type=int, nargs="+", default=[8])
    p.add_argument("--Ns", type=int, nargs="+", default=[4])
    p.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0])
    p.add_argument("--algos", nargs="+",
                   default=["bai_pg", "ganesh_npg", "ganesh_mlmc"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--K", type=int, default=500)
    p.add_argument("--H", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.9999)
    p.add_argument("--L_max", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--N_inner", type=int, default=10)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--B_max", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--q_init", default="5")
    p.add_argument("--delta_reinforce", type=float, default=1e-3,
                   help="δ-guard in REINFORCE eq. 38 denominator; larger values weaken "
                        "the fairness shortcut and match the paper's implied setup.")
    p.add_argument("--grad_clip", type=float, default=None,
                   help="L2-norm clip on ascent direction ω before Adam. None = no clip.")
    p.add_argument("--objective", default="alpha", choices=["alpha", "minfair"],
                   help="alpha = α-fairness; minfair = max-min (Lipschitz, non-2nd-order-smooth).")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    configs = build_configs(args)
    if args.skip_existing:
        out_dir = Path(args.out_dir)
        def out_file(c):
            return (out_dir / f"{c['algorithm']}_{c['policy']}_M{c['M']}"
                              f"_N{c['N']}_alpha{c['alpha']:.1f}_seed{c['seed']}.json")
        kept = [c for c in configs if not out_file(c).exists()]
        print(f"skip_existing: {len(configs) - len(kept)} present, {len(kept)} to run")
        configs = kept

    print(f"Sweep v3 (MLMC showcase): {len(configs)} runs, {args.workers} workers")
    print(f"  algos={args.algos} alphas={args.alphas} Ms={args.Ms} Ns={args.Ns}")
    print(f"  K={args.K} H={args.H} N_inner={args.N_inner} q_init={args.q_init}")

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
                tag = (f"{cfg['algorithm']:<12} alpha={cfg['alpha']:.1f} "
                       f"M={cfg['M']} N={cfg['N']:<3} seed={cfg['seed']}")
                print(f"  [{completed:3d}/{len(configs)}] {tag}  "
                      f"{res['time']:6.1f}s  (ETA {eta/60:5.1f} min)")
            except Exception as e:
                tag = f"{cfg['algorithm']} alpha={cfg['alpha']} M={cfg['M']} N={cfg['N']}"
                print(f"  FAILED {tag}: {type(e).__name__}: {e}")

    print(f"\nSweep done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
