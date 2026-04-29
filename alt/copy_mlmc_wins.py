"""Identify and copy MLMC-NPG winning configurations to a single folder.

Definition of "winning": MLMC mean > Bai-PG mean (positive M-X). Optionally
require statistical significance via p-value threshold.

Reads each <tag>_pvalues.txt to extract M-X vs Bai and the p-value, then
copies the corresponding fair_budget.png to figures/mlmc_wins/.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SRC_DIRS = [REPO / "figures" / "v14", REPO / "figures" / "v15"]
DST_DIR = REPO / "figures" / "mlmc_wins"


def parse_pvalues(txt_path: Path) -> dict | None:
    """Returns {tag, m_x_bai, p_bai, m_x_rein, p_rein, m_x_scal, p_scal}.
    Returns None if no Bai contest found.
    """
    if not txt_path.exists():
        return None
    text = txt_path.read_text()
    out = {"file": txt_path.stem.replace("_pvalues", "")}
    for line in text.splitlines():
        m = re.search(
            r"vs\s+(\S+)\s+.*?M-X=\s*([+\-]?\d+\.\d+)\s+p=([\d.]+)", line
        )
        if not m:
            continue
        algo, mx, pv = m.group(1), float(m.group(2)), float(m.group(3))
        if algo == "bai_pg":
            out["m_x_bai"] = mx
            out["p_bai"] = pv
        elif algo == "reinforce":
            out["m_x_rein"] = mx
            out["p_rein"] = pv
        elif algo == "scalarized_reinforce":
            out["m_x_scal"] = mx
            out["p_scal"] = pv
    return out if "m_x_bai" in out else None


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for src in SRC_DIRS:
        if not src.exists():
            continue
        for pv_path in sorted(src.glob("*_pvalues.txt")):
            info = parse_pvalues(pv_path)
            if info is None:
                continue
            png = pv_path.with_name(info["file"] + "_fair_budget.png")
            if not png.exists():
                continue
            wins_bai = info["m_x_bai"] > 0
            rows.append({
                "src_dir": src.name,
                "tag": info["file"],
                "m_x_bai": info["m_x_bai"],
                "p_bai": info["p_bai"],
                "m_x_rein": info.get("m_x_rein"),
                "p_rein": info.get("p_rein"),
                "m_x_scal": info.get("m_x_scal"),
                "p_scal": info.get("p_scal"),
                "wins_bai": wins_bai,
                "png": png,
            })

    # Copy winners
    print(f"\nMLMC-NPG winning configurations (MLMC mean > Bai mean):\n")
    print(f"{'src':<6} {'tag':<55} {'M-X(Bai)':>9} {'p_bai':>7} "
          f"{'M-X(R)':>8} {'p_R':>7} {'M-X(sR)':>8} {'p_sR':>7}")
    print("-" * 116)
    win_count = 0
    for r in sorted(rows, key=lambda x: -x["m_x_bai"]):
        if not r["wins_bai"]:
            continue
        win_count += 1
        # Copy with src-prefix to disambiguate v14/v15 same tags
        dst_name = f"{r['src_dir']}_{r['tag']}_fair_budget.png"
        shutil.copy(r["png"], DST_DIR / dst_name)
        print(f"{r['src_dir']:<6} {r['tag']:<55} "
              f"{r['m_x_bai']:>+9.3f} {r['p_bai']:>7.4f} "
              f"{(r['m_x_rein'] if r['m_x_rein'] is not None else 0):>+8.3f} "
              f"{(r['p_rein'] if r['p_rein'] is not None else 0):>7.4f} "
              f"{(r['m_x_scal'] if r['m_x_scal'] is not None else 0):>+8.3f} "
              f"{(r['p_scal'] if r['p_scal'] is not None else 0):>7.4f}")
    print(f"\n{win_count} winning configs copied to {DST_DIR}")

    # Also list non-wins for reference
    losses = [r for r in rows if not r["wins_bai"]]
    if losses:
        print(f"\nMLMC LOSES vs Bai in {len(losses)} configs (NOT copied):")
        for r in sorted(losses, key=lambda x: x["m_x_bai"]):
            print(f"  {r['src_dir']}/{r['tag']}: M-X={r['m_x_bai']:+.3f}")


if __name__ == "__main__":
    main()
