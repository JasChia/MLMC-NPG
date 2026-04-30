# v16 — regeneration commands

Each v16 plot was produced in two steps: run a sweep with `alt.run_v15_search`,
then render the plot with `alt.plot_v14`.

All sweeps use NN-32 policy, fair-budget across MLMC-NPG / Bai-PG / paper-A
REINFORCE / scalarized REINFORCE. Output goes to `results/sweep_v16/` and
`figures/v16/`.

## Running everything

```bash
# 1) install deps
pip install numpy torch scipy matplotlib

# 2) regenerate sweeps (~3-4 hours on a typical workstation)
bash figures/v16/regen_all.sh   # see "All commands" section below

# 3) re-render plots from cached results (seconds)
bash figures/v16/replot_all.sh  # see "All commands" section below
```

## Per-plot commands

Each block has the SWEEP command (slow) and the PLOT command (fast). If
`results/sweep_v16/<tag>...json` already exists, skip the sweep step.

---

### 1. `ft_nn32_B16_NK40_a000_fair_budget.png` — α=0 (linear), B=16, K=40 [v14 headline at K=40]

MLMC vs Bai p=0.007. Strongest single-environment win.

```bash
# sweep
python -m alt.run_v15_search \
  --out_dir results/sweep_v16 --tag ft_nn32_B16_NK40_a000 \
  --env_kind fruit_tree --alpha 0.0 --policy nn --hidden 32 \
  --B 16 --N 20 --K 40 --delta 0.5 --seeds 7 --workers 2

# plot
python -m alt.plot_v14 --results_dir results/sweep_v16 \
  --tag ft_nn32_B16_NK40_a000 --out_dir figures/v16 \
  --title "Fruit Tree NN-32 alpha=0 B=16 N=20 K=40 [7 seeds]"
```

---

### 2. `ft_nn32_B16_NK40_a010_d010_fair_budget.png` — α=0.1, δ=0.1, B=16, K=40 [v15 headline at K=40]

MLMC vs Bai p=0.038. The non-zero-α significant win.

```bash
python -m alt.run_v15_search \
  --out_dir results/sweep_v16 --tag ft_nn32_B16_NK40_a010_d010 \
  --env_kind fruit_tree --alpha 0.1 --policy nn --hidden 32 \
  --B 16 --N 20 --K 40 --delta 0.1 --seeds 7 --workers 4

python -m alt.plot_v14 --results_dir results/sweep_v16 \
  --tag ft_nn32_B16_NK40_a010_d010 --out_dir figures/v16 \
  --title "Fruit Tree NN-32 alpha=0.1 delta=0.1 B=16 N=20 K=40 [7 seeds]"
```

---

### 3. `ft_nn32_B16_NK40_a010_fair_budget.png` — α=0.1, default δ=0.5, B=16, K=40

MLMC vs Bai p=0.053 (borderline).

```bash
python -m alt.run_v15_search \
  --out_dir results/sweep_v16 --tag ft_nn32_B16_NK40_a010 \
  --env_kind fruit_tree --alpha 0.1 --policy nn --hidden 32 \
  --B 16 --N 20 --K 40 --delta 0.5 --seeds 7 --workers 2

python -m alt.plot_v14 --results_dir results/sweep_v16 \
  --tag ft_nn32_B16_NK40_a010 --out_dir figures/v16 \
  --title "Fruit Tree NN-32 alpha=0.1 B=16 N=20 K=40 [7 seeds]"
```

---

### 4. `ft_nn32_B16_NK40_a025_d010_fair_budget.png` — α=0.25, δ=0.1, B=16, K=40

MLMC vs Bai p=0.026.

```bash
python -m alt.run_v15_search \
  --out_dir results/sweep_v16 --tag ft_nn32_B16_NK40_a025_d010 \
  --env_kind fruit_tree --alpha 0.25 --policy nn --hidden 32 \
  --B 16 --N 20 --K 40 --delta 0.1 --seeds 7 --workers 2

python -m alt.plot_v14 --results_dir results/sweep_v16 \
  --tag ft_nn32_B16_NK40_a025_d010 --out_dir figures/v16 \
  --title "Fruit Tree NN-32 alpha=0.25 delta=0.1 B=16 N=20 K=40 [7 seeds]"
```

---

### 5. `ft_nn32_B16_NK40_a025_fair_budget.png` — α=0.25, default δ=0.5, B=16, K=40

MLMC vs Bai p=0.014.

```bash
python -m alt.run_v15_search \
  --out_dir results/sweep_v16 --tag ft_nn32_B16_NK40_a025 \
  --env_kind fruit_tree --alpha 0.25 --policy nn --hidden 32 \
  --B 16 --N 20 --K 40 --delta 0.5 --seeds 7 --workers 2

python -m alt.plot_v14 --results_dir results/sweep_v16 \
  --tag ft_nn32_B16_NK40_a025 --out_dir figures/v16 \
  --title "Fruit Tree NN-32 alpha=0.25 B=16 N=20 K=40 [7 seeds]"
```

---

### 6. `ft_nn32_B16_NK40_a050_fair_budget.png` — α=0.5 (canonical √-fairness), B=16, K=40

MLMC vs Bai p=0.55 (no significant difference; alpha=0.5 erases the v14/v15 advantage).

```bash
python -m alt.run_v15_search \
  --out_dir results/sweep_v16 --tag ft_nn32_B16_NK40_a050 \
  --env_kind fruit_tree --alpha 0.5 --policy nn --hidden 32 \
  --B 16 --N 20 --K 40 --delta 0.5 --seeds 5 --workers 4

python -m alt.plot_v14 --results_dir results/sweep_v16 \
  --tag ft_nn32_B16_NK40_a050 --out_dir figures/v16 \
  --title "Fruit Tree NN-32 alpha=0.5 B=16 N=20 K=40 [5 seeds]"
```

---

### 7. `ft_nn32_B16_NK40_a100_fair_budget.png` — α=1.0 (log-fairness), B=16, K=40

MLMC vs Bai p=0.40.

```bash
python -m alt.run_v15_search \
  --out_dir results/sweep_v16 --tag ft_nn32_B16_NK40_a100 \
  --env_kind fruit_tree --alpha 1.0 --policy nn --hidden 32 \
  --B 16 --N 20 --K 40 --delta 0.5 --seeds 3 --workers 4

python -m alt.plot_v14 --results_dir results/sweep_v16 \
  --tag ft_nn32_B16_NK40_a100 --out_dir figures/v16 \
  --title "Fruit Tree NN-32 alpha=1.0 B=16 N=20 K=40 [3 seeds]"
```

---

### 8. `ft_nn32_B16_NK60_a050_fair_budget.png` — α=0.5, B=16, N=30, K=60 (extension of v14 N=K=30 winner)

MLMC vs Bai p=0.70 (gain disappears as Bai catches up at large K).

```bash
python -m alt.run_v15_search \
  --out_dir results/sweep_v16 --tag ft_nn32_B16_NK60_a050 \
  --env_kind fruit_tree --alpha 0.5 --policy nn --hidden 32 \
  --B 16 --N 30 --K 60 --delta 0.5 --seeds 3 --workers 4

python -m alt.plot_v14 --results_dir results/sweep_v16 \
  --tag ft_nn32_B16_NK60_a050 --out_dir figures/v16 \
  --title "Fruit Tree NN-32 alpha=0.5 B=16 N=30 K=60 [3 seeds]"
```

---

### 9. `ft_nn32_B4_NK40_a050_fair_budget.png` — α=0.5, B=4, K=40

MLMC vs Bai p=0.10 (matches v15 K=40 result; included as a small-B point).

```bash
python -m alt.run_v15_search \
  --out_dir results/sweep_v16 --tag ft_nn32_B4_NK40_a050 \
  --env_kind fruit_tree --alpha 0.5 --policy nn --hidden 32 \
  --B 4 --N 20 --K 40 --delta 0.5 --seeds 3 --workers 2

python -m alt.plot_v14 --results_dir results/sweep_v16 \
  --tag ft_nn32_B4_NK40_a050 --out_dir figures/v16 \
  --title "Fruit Tree NN-32 alpha=0.5 B=4 N=20 K=40 [3 seeds]"
```

---

### 10. `ft_nn32_B8_NK40_a000_fair_budget.png` — α=0, B=8, K=40 [biggest M-X jump]

MLMC vs Bai p=0.016, M-X = +5.60 — biggest improvement vs the K=20 baseline (+2.27 at K=20).

```bash
python -m alt.run_v15_search \
  --out_dir results/sweep_v16 --tag ft_nn32_B8_NK40_a000 \
  --env_kind fruit_tree --alpha 0.0 --policy nn --hidden 32 \
  --B 8 --N 20 --K 40 --delta 0.5 --seeds 5 --workers 4

python -m alt.plot_v14 --results_dir results/sweep_v16 \
  --tag ft_nn32_B8_NK40_a000 --out_dir figures/v16 \
  --title "Fruit Tree NN-32 alpha=0 B=8 N=20 K=40 [5 seeds]"
```

---

### 11. `ft_nn32_B8_NK40_a010_d010_fair_budget.png` — α=0.1, δ=0.1, B=8, K=40

MLMC vs Bai p=0.053 (borderline, robustness check at smaller B).

```bash
python -m alt.run_v15_search \
  --out_dir results/sweep_v16 --tag ft_nn32_B8_NK40_a010_d010 \
  --env_kind fruit_tree --alpha 0.1 --policy nn --hidden 32 \
  --B 8 --N 20 --K 40 --delta 0.1 --seeds 7 --workers 2

python -m alt.plot_v14 --results_dir results/sweep_v16 \
  --tag ft_nn32_B8_NK40_a010_d010 --out_dir figures/v16 \
  --title "Fruit Tree NN-32 alpha=0.1 delta=0.1 B=8 N=20 K=40 [7 seeds]"
```

---

### 12. `ft_nn32_B8_NK40_a050_fair_budget.png` — α=0.5, B=8, K=40

MLMC vs Bai p=0.056 (borderline).

```bash
python -m alt.run_v15_search \
  --out_dir results/sweep_v16 --tag ft_nn32_B8_NK40_a050 \
  --env_kind fruit_tree --alpha 0.5 --policy nn --hidden 32 \
  --B 8 --N 20 --K 40 --delta 0.5 --seeds 5 --workers 2

python -m alt.plot_v14 --results_dir results/sweep_v16 \
  --tag ft_nn32_B8_NK40_a050 --out_dir figures/v16 \
  --title "Fruit Tree NN-32 alpha=0.5 B=8 N=20 K=40 [5 seeds]"
```

---

### 13. `resource_nn32_B16_NK40_a025_fair_budget.png` — Resource Gathering, α=0.25, B=16, K=40

MLMC ties with Bai (M-X=+0.00). Different env (M=3 channels: gold + gems + step penalty).

```bash
python -m alt.run_v15_search \
  --out_dir results/sweep_v16 --tag resource_nn32_B16_NK40_a025 \
  --env_kind resource --alpha 0.25 --policy nn --hidden 32 \
  --B 16 --N 20 --K 40 --gamma 0.999 --seeds 3 --workers 2

python -m alt.plot_v14 --results_dir results/sweep_v16 \
  --tag resource_nn32_B16_NK40_a025 --out_dir figures/v16 \
  --title "Resource NN-32 alpha=0.25 B=16 N=20 K=40 [3 seeds]"
```

---

## Notes on parameters

- `--K 40` for MLMC-NPG; the runner auto-derives `K_pg` for Bai-PG and
  `K_rein` for the two REINFORCE variants so that all four algorithms
  consume the same total trajectory budget. See
  `derive_K_pg_reinforce` in `alt/run_v15_search.py`.
- `--B` sets the inner-batch size and (per the v14/v15 user constraint)
  also the MLMC outer-batch cap (`B_max = B`). Bai-PG's `N1 = N2 = B`,
  REINFORCE batch = `B`.
- `--delta` sets the δ-guard in the α-fairness objective
  `f(J) = Σ_m U_α(max(0, J_m) + δ)`. Smaller δ ⇒ more concave near 0.
- `--gamma` defaults to 0.99 (Fruit Tree) or 0.9999 (DST). Resource and
  Bonus World use 0.999 here.
- `--workers` controls process-pool parallelism. Use 2 if you've seen
  `BrokenProcessPool` errors; use 4 if memory allows.

## Output

Each sweep produces in `results/sweep_v16/`:
- `{algo}_{env_tag}_{policy_tag}_B{B}_N{N}_K{K_use}_beta{beta}_lr{lr}_v15_{tag}_seed{s}.json`

Each plot run produces in `figures/v16/`:
- `{tag}_fair_budget.png` — curve of `f(J)` vs cumulative trajectories,
  4 algorithms, mean ± std across seeds
- `{tag}_plateaus.csv` — last-K/10 plateau means + std per algorithm
- `{tag}_pvalues.txt` — Wilcoxon rank-sum (MWU exact, two-sided) of MLMC
  vs each alternative

## Cross-reference

For interpretation and the full v14/v15/v16 narrative, see
`figures/v16/v16_summary.md`.
