"""v11 queueing fair-budget runner for Paper A queue sizes M=2,4,8.

Filename schema:
  {algo}_log_linear_M{M}_N{N}_K{K}_{mode}_alpha{alpha}_seed{seed}.json
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch


def configs_from_csv(csv_text: str, base: dict) -> list[dict]:
    rows = []
    for line in csv_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        M, algo, N, K, mode = [p.strip() for p in line.split(",")]
        rows.append({
            "M": int(M),
            "algorithm": algo,
            "N": int(N),
            "K": int(K),
            "mode": mode,
        })
    configs = []
    for row in rows:
        for seed in range(base["seeds"]):
            cfg = dict(base)
            cfg.update(row)
            cfg["seed"] = seed
            configs.append(cfg)
    return configs


def output_path(cfg: dict) -> Path:
    return Path(cfg["out_dir"]) / (
        f"{cfg['algorithm']}_log_linear_M{cfg['M']}_N{cfg['N']}_K{cfg['K']}_"
        f"{cfg['mode']}_alpha{cfg['alpha']:.1f}_seed{cfg['seed']}.json"
    )


def run_one(cfg: dict) -> dict:
    from alt.envs_v2 import ARRIVAL_RATES, batch_rollout_v2
    from alt.objectives_v2 import GeneralAlphaFairness
    from src.policy import make_policy
    from src.train import train

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)

    rng = np.random.default_rng(cfg["seed"])
    lambdas = ARRIVAL_RATES[cfg["M"]]
    env_config = {"M": cfg["M"], "L_max": cfg["L_max"], "lambdas": lambdas}
    policy = make_policy("log_linear", M=cfg["M"], L_max=cfg["L_max"], device="cpu")
    obj = GeneralAlphaFairness(alpha=cfg["alpha"], delta=cfg["delta"], H=cfg["H"])

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
        algo_kwargs = {"N": N, "delta_guard": cfg["delta_reinforce"]}
    elif cfg["algorithm"] == "ganesh_mlmc":
        algo_kwargs = {
            "B_max": cfg["B_max"],
            "B_inner": N,
            "N_inner": cfg["N_inner"],
            "beta": cfg["beta"],
        }
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

    out = output_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(history, f)
    return {"file": str(out), "time": history["wall_time_sec"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/sweep_v11_queue_fair")
    p.add_argument("--config_csv", default="alt/configs/v11_queue_fair.csv")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--H", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.9999)
    p.add_argument("--L_max", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--q_init", default="5")
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
        "seeds": args.seeds,
        "H": args.H,
        "gamma": args.gamma,
        "L_max": args.L_max,
        "lr": args.lr,
        "alpha": args.alpha,
        "delta": args.delta,
        "q_init": args.q_init,
        "N_inner": args.N_inner,
        "beta": args.beta,
        "B_max": args.B_max,
        "delta_reinforce": args.delta_reinforce,
        "out_dir": args.out_dir,
    }
    configs = configs_from_csv(csv_text, base)
    if args.skip_existing:
        kept = [cfg for cfg in configs if not output_path(cfg).exists()]
        print(f"skip_existing: {len(configs) - len(kept)} present, {len(kept)} to run")
        configs = kept

    print(f"v11 queue sweep: {len(configs)} runs, {args.workers} workers")
    print(
        f"  H={args.H} gamma={args.gamma} lr={args.lr} q_init={args.q_init} "
        f"B_max={args.B_max} beta={args.beta}"
    )
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
                eta = elapsed / completed * (len(configs) - completed)
                print(
                    f"  [{completed:3d}/{len(configs)}] M={cfg['M']:<2} "
                    f"{cfg['algorithm']:<11} N={cfg['N']:<3} K={cfg['K']:<4} "
                    f"{cfg['mode']:<8} seed={cfg['seed']} {res['time']:6.1f}s "
                    f"(ETA {eta/60:4.1f} min)"
                )
            except Exception as e:
                print(
                    f"  FAILED M={cfg['M']} {cfg['algorithm']} "
                    f"N={cfg['N']} K={cfg['K']}: {type(e).__name__}: {e}"
                )

    print(f"\nv11 queue sweep done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
