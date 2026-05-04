"""Remove all curves for one color from a pgfplots tikz file.

Usage:
    python strip_algo_from_tikz.py <input.tikz> <color_name> [--label "old label"] \
        [--rename "new label"] [--legend-columns N] [--out output.tikz]

Removes:
  - \\path [draw=<color>,...] ... --cycle;    (confidence bands)
  - \\addplot [..., <color>] table {%...};    (mean lines)
  - \\addlegendentry{<label>}                 (if --label given)

Optionally renames another legend entry and updates legend columns count.
"""
import re
import argparse
from pathlib import Path


def strip_color_blocks(lines: list[str], color: str) -> list[str]:
    """Remove \\path and \\addplot blocks that reference the given color."""
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Confidence band: \path [draw=COLOR, fill=COLOR, ...]
        if re.match(rf'\\path\s+\[draw={re.escape(color)},', line):
            # Skip until --cycle;
            while i < len(lines):
                if lines[i].strip().endswith('--cycle;'):
                    i += 1
                    break
                i += 1
            continue
        # Mean line: \addplot [..., COLOR]
        if re.match(rf'\\addplot\s+\[.*{re.escape(color)}.*\]', line):
            # Skip until };  (end of table data)
            while i < len(lines):
                if lines[i].strip() == '};':
                    i += 1
                    break
                i += 1
            continue
        out.append(line)
        i += 1
    return out


def process(src: Path, color: str, label: str | None, rename: str | None,
            legend_cols: int | None, dst: Path) -> None:
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)

    lines = strip_color_blocks(lines, color)

    result = []
    for line in lines:
        # Remove the legend entry for the stripped algorithm
        if label and line.strip() == f'\\addlegendentry{{{label}}}':
            continue
        # Rename another legend entry
        if rename and label:
            pass  # rename handled separately below
        result.append(line)

    # Handle rename as a string replacement pass
    if rename:
        result = [l.replace(f'\\addlegendentry{{REINFORCE (scalar)}}',
                             f'\\addlegendentry{{{rename}}}') for l in result]

    # Update legend columns
    if legend_cols is not None:
        result = [re.sub(r'legend columns=\d+', f'legend columns={legend_cols}', l)
                  for l in result]

    dst.write_text("".join(result), encoding="utf-8")
    print(f"Written to {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("color")
    p.add_argument("--label", default=None, help="Legend entry text to remove")
    p.add_argument("--rename", default=None, help="New label for REINFORCE (scalar)")
    p.add_argument("--legend-columns", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    src = Path(args.input)
    dst = Path(args.out) if args.out else src
    process(src, args.color, args.label, args.rename, args.legend_columns, dst)


if __name__ == "__main__":
    main()
