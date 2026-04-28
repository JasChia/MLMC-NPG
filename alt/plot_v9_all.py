"""Generate all v9 plots in one go: per-env iter and sample-axis convergence,
plus the cross-env summary. Idempotent — re-runnable any time after sweeps
update."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SWEEPS = [
    # (sweep_dir, output_subdir, title, dual_panel)
    ("results/sweep_v9_resource",          "v9_resource",       "Resource Gathering (M=3, sparse, episodic, tabular d=404)",   False),
    ("results/sweep_v9_dst_multi",         "v9_dst_multi",      "Multi-treasure DST (M=6, 5 treasures + step, tabular d=204)", False),
    ("results/sweep_v9_dst_large",         "v9_dst_large",      "Larger DST (20x10 grid, M=2, tabular d=804)",                 False),
    ("results/sweep_v9_episodic_queue",    "v9_episodic_queue", "Episodic Queueing M=8 (rare-arrival termination)",            False),
    ("results/sweep_v9_M32_challenging",   "v9_M32_challenging","M=32 log_linear, challenging-load arrivals (Sigma=0.8, d=1056)", True),
    ("results/sweep_v9_M64_challenging",   "v9_M64_challenging","M=64 log_linear, challenging-load arrivals (Sigma=0.8, d=4160)", True),
]


def run(cmd):
    print(" $", " ".join(cmd))
    subprocess.run(cmd, check=False)


def main():
    figures_dir = Path("figures")
    for sweep_dir, fig_subdir, title, dual_panel in SWEEPS:
        if not Path(sweep_dir).exists() or not list(Path(sweep_dir).glob("*.json")):
            print(f"  skip {sweep_dir}: not present yet")
            continue
        out_iter = figures_dir / fig_subdir / f"{fig_subdir}_iter.png"
        out_traj = figures_dir / fig_subdir / f"{fig_subdir}_traj.png"
        flag = ["--dual_panel"] if dual_panel else []
        run([sys.executable, "-m", "alt.plot_sample_axis",
             "--dir", sweep_dir, "--output", str(out_iter),
             "--x_axis", "iter", "--title", title, "--print_table"] + flag)
        run([sys.executable, "-m", "alt.plot_sample_axis",
             "--dir", sweep_dir, "--output", str(out_traj),
             "--x_axis", "trajectories", "--title", title + " (sample-axis)"] + flag)

    # Cross-env summary
    run([sys.executable, "-m", "alt.plot_v9_summary",
         "--output", "figures/v9_summary/policy_effect_summary_v9.png"])


if __name__ == "__main__":
    main()
