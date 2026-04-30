"""Visualize learned NN policies on the Fruit Tree environment.

The training JSONs do not store final policy parameters, so this script
deterministically replays a selected run config, then plots P(right) at every
internal tree node and highlights the greedy root-to-leaf path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from alt.envs_fruit_tree import FruitTreeEnv, FruitTreeObj
from alt.run_v11 import _make_env_and_policy
from alt.train_lowrank import train as train_lowrank
from src.train import train as train_src


def replay_policy(run_json: Path):
    with open(run_json) as f:
        saved = json.load(f)
    cfg = dict(saved["config"])

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    rng = np.random.default_rng(cfg["seed"])

    policy, obj, H, rollout_fn = _make_env_and_policy(cfg, rng)
    N = cfg["N"]
    if cfg["algorithm"] == "bai_pg":
        algo_kwargs = {"N1": N, "N2": N}
    elif cfg["algorithm"] == "reinforce":
        algo_kwargs = {"N": N, "delta_guard": cfg.get("delta_reinforce", 1.0)}
    elif cfg["algorithm"] == "ganesh_mlmc":
        algo_kwargs = {
            "B_max": cfg.get("B_max", 64),
            "B_inner": N,
            "N_inner": cfg["N_inner"],
            "beta": cfg["beta"],
        }
    else:
        raise ValueError(cfg["algorithm"])

    # The low-rank path computes the same F_hat @ omega product without
    # materializing F_hat; use it for NN MLMC visualizations to keep policy
    # reconstruction practical.
    use_lowrank = (
        cfg.get("lowrank_npg", False)
        or (cfg.get("policy") == "nn" and cfg["algorithm"] in {"ganesh_npg", "ganesh_mlmc"})
    )
    trainer = train_lowrank if use_lowrank else train_src
    trainer(
        algorithm=cfg["algorithm"],
        policy=policy,
        rollout_fn=rollout_fn,
        obj=obj,
        gamma=cfg["gamma"],
        K=cfg["K"],
        H=H,
        rng=rng,
        lr=cfg["lr"],
        algo_kwargs=algo_kwargs,
        verbose=False,
    )
    return cfg, policy


def tree_positions(depth: int) -> dict[int, tuple[float, float]]:
    pos = {}
    for d in range(depth):
        first = (1 << d) - 1
        count = 1 << d
        for j in range(count):
            idx = first + j
            x = (j + 0.5) / count
            y = -float(d)
            pos[idx] = (x, y)
    return pos


def greedy_path(policy, env: FruitTreeEnv) -> tuple[list[int], int]:
    node = 0
    path = []
    for _ in range(env.depth):
        path.append(node)
        probs = policy.action_probs_batch(np.asarray([node], dtype=np.int64))[0]
        action = int(np.argmax(probs))
        node = 2 * node + 1 + action
    leaf = node - env.first_leaf_idx
    return path, leaf


def plot_policy_panel(ax, cfg, policy):
    env = FruitTreeEnv(depth=cfg.get("depth", 7), seed=42)
    obj = FruitTreeObj(delta=cfg["delta"])
    pos = tree_positions(env.depth)
    internal = np.arange(env.n_internal, dtype=np.int64)
    probs = policy.action_probs_batch(internal)
    p_right = probs[:, 1]
    path, leaf = greedy_path(policy, env)
    path_edges = set()
    for node in path:
        pr = policy.action_probs_batch(np.asarray([node], dtype=np.int64))[0]
        action = int(np.argmax(pr))
        child = 2 * node + 1 + action
        path_edges.add((node, child))

    for parent in range(env.n_internal):
        if parent not in pos:
            continue
        for action in (0, 1):
            child = 2 * parent + 1 + action
            if child >= env.n_internal:
                # Leaf positions are one row below the last internal level.
                depth = env.depth
                leaf_j = child - env.first_leaf_idx
                child_pos = ((leaf_j + 0.5) / env.n_leaves, -float(depth))
            else:
                child_pos = pos[child]
            is_path = (parent, child) in path_edges
            ax.plot(
                [pos[parent][0], child_pos[0]],
                [pos[parent][1], child_pos[1]],
                color="black" if is_path else "0.75",
                lw=2.3 if is_path else 0.45,
                alpha=0.9 if is_path else 0.4,
                zorder=1,
            )

    xs = [pos[i][0] for i in internal]
    ys = [pos[i][1] for i in internal]
    sc = ax.scatter(xs, ys, c=p_right, cmap="coolwarm", vmin=0, vmax=1,
                    s=18, edgecolors="none", zorder=2)
    path_x = [pos[i][0] for i in path]
    path_y = [pos[i][1] for i in path]
    ax.scatter(path_x, path_y, facecolors="none", edgecolors="black",
               s=42, linewidths=1.0, zorder=3)

    leaf_vals = env.leaf_values[leaf]
    title = f"{cfg['algorithm']} {cfg['mode']} seed={cfg['seed']}"
    ax.set_title(title, fontsize=9)
    ax.text(
        0.02, 0.02,
        f"greedy leaf {leaf}\nf(leaf)={obj.value(leaf_vals):.2f}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.8},
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-env.depth - 0.35, 0.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    return sc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", default="figures/v11_fair_budget/fruit_tree_nn32_learned_policy.png")
    args = p.parse_args()

    run_paths = [Path(r) for r in args.runs]
    fig, axes = plt.subplots(1, len(run_paths), figsize=(4.2 * len(run_paths), 4.8))
    if len(run_paths) == 1:
        axes = [axes]

    last_sc = None
    for ax, run_path in zip(axes, run_paths):
        cfg, policy = replay_policy(run_path)
        last_sc = plot_policy_panel(ax, cfg, policy)

    fig.suptitle("Fruit Tree NN-32 learned policy: node color = P(right), black = greedy path")
    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes, shrink=0.75, pad=0.02)
        cbar.set_label("P(right)")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
