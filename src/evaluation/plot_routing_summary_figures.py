"""
Create compact routing figures for paper/report use.

Example:
    python -m src.evaluation.plot_routing_summary_figures --tokens-csv results/routing_heatmaps/routing_tokens.csv --summary-csv results/routing_heatmaps/routing_summary.csv --output-dir results/routing_paper_figures
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


EXPERTS = ("fr", "ca", "oc")
CASE_ORDER = ("French", "Catalan", "Occitan", "Code-Switch")
MODE_ORDER = ("Soft/default routing", "Sharpened soft routing", "Hard argmax routing")


def slug(text: str) -> str:
    text = text.lower().replace("/", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_-]+", "", text)
    return text.strip("_")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value: str) -> float:
    return float(value) * 100.0


def short_token(token: str, max_len: int = 12) -> str:
    token = token.replace("\\n", "\\\\n")
    if token == "":
        token = "[space]"
    if len(token) <= max_len:
        return token
    return token[: max_len - 3] + "..."


def plot_mean_activation_grid(
    summary_rows: list[dict[str, str]],
    out_path: Path,
    *,
    level: str,
    scope: str,
    modes: tuple[str, ...] = MODE_ORDER,
    cases: tuple[str, ...] = CASE_ORDER,
) -> None:
    rows = [
        row
        for row in summary_rows
        if row["level"] == level and row["scope"] == scope and row["mode"] in modes
    ]
    by_key = {(row["mode"], row["case"]): row for row in rows}

    fig, axes = plt.subplots(
        1,
        len(modes),
        figsize=(4.1 * len(modes), 3.0),
        sharey=True,
        constrained_layout=True,
    )
    if len(modes) == 1:
        axes = [axes]

    image = None
    for ax, mode in zip(axes, modes):
        matrix = []
        for expert in EXPERTS:
            matrix.append(
                [
                    pct(by_key[(mode, case)][f"{expert}_mean"])
                    if (mode, case) in by_key
                    else 0.0
                    for case in cases
                ]
            )

        image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=100.0, cmap="viridis")
        ax.set_title(mode.replace(" routing", ""), fontsize=10)
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels(cases, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(EXPERTS)))
        ax.set_yticklabels(EXPERTS, fontsize=9)
        ax.set_xlabel("Prompt case", fontsize=9)

        for y, expert in enumerate(EXPERTS):
            for x, case in enumerate(cases):
                value = matrix[y][x]
                label_color = "white" if value < 45.0 else "black"
                ax.text(
                    x,
                    y,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    color=label_color,
                    fontsize=8,
                )

    axes[0].set_ylabel("Expert", fontsize=9)
    fig.suptitle(f"Mean Expert Activation ({level}, {scope} tokens)", fontsize=12)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
        cbar.set_label("Mean gate mass (%)", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def response_token_rows(
    token_rows: list[dict[str, str]],
    *,
    mode: str,
    case: str,
    level: str = "Token-level",
) -> list[dict[str, str]]:
    return [
        row
        for row in token_rows
        if row["mode"] == mode
        and row["case"] == case
        and row["level"] == level
        and row["in_response"] == "True"
    ]


def plot_trimmed_token_heatmap(
    token_rows: list[dict[str, str]],
    out_path: Path,
    *,
    mode: str,
    case: str,
    start: int = 0,
    max_tokens: int = 40,
) -> None:
    rows = response_token_rows(token_rows, mode=mode, case=case)
    if not rows:
        raise ValueError(f"No response token rows found for mode={mode!r}, case={case!r}")

    rows = rows[start : start + max_tokens]
    data = [[float(row[expert]) for row in rows] for expert in EXPERTS]
    tokens = [short_token(row["token"]) for row in rows]

    fig, ax = plt.subplots(figsize=(max(7.0, 0.24 * len(rows)), 2.55))
    image = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(f"{mode.replace(' routing', '')} | {case} | response token routing", fontsize=10)
    ax.set_yticks(range(len(EXPERTS)))
    ax.set_yticklabels(EXPERTS, fontsize=9)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=70, ha="right", fontsize=7)
    ax.set_xlabel("Generated response token", fontsize=9)
    ax.set_ylabel("Expert", fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Gate mass", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_latex_table(summary_rows: list[dict[str, str]], out_path: Path, *, level: str, scope: str) -> None:
    rows = [
        row
        for row in summary_rows
        if row["level"] == level and row["scope"] == scope and row["mode"] in MODE_ORDER
    ]
    by_key = {(row["mode"], row["case"]): row for row in rows}

    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Mode & Expert & French & Catalan & Occitan & Code-Switch \\\\",
        "\\midrule",
    ]
    for mode in MODE_ORDER:
        short_mode = mode.replace(" routing", "")
        for expert in EXPERTS:
            values = []
            for case in CASE_ORDER:
                row = by_key.get((mode, case))
                values.append(f"{pct(row[f'{expert}_mean']):.1f}" if row else "--")
            lines.append(f"{short_mode} & {expert} & " + " & ".join(values) + " \\\\")
        if mode != MODE_ORDER[-1]:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_case_trim(value: str) -> tuple[str, int, int]:
    parts = value.split(":")
    if len(parts) == 1:
        return parts[0], 0, 40
    if len(parts) == 2:
        return parts[0], int(parts[1]), 40
    if len(parts) == 3:
        return parts[0], int(parts[1]), int(parts[2])
    raise argparse.ArgumentTypeError(
        "Trim spec must be CASE, CASE:START, or CASE:START:COUNT"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compact paper figures from MoLE routing CSVs."
    )
    parser.add_argument(
        "--tokens-csv",
        default="results/routing_heatmaps/routing_tokens.csv",
        help="Token-level routing CSV from plot_routing_heatmaps_from_trace.py.",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/routing_heatmaps/routing_summary.csv",
        help="Summary routing CSV from plot_routing_heatmaps_from_trace.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/routing_paper_figures",
        help="Directory for paper-sized figures.",
    )
    parser.add_argument("--scope", default="response", choices=["response", "full"])
    parser.add_argument(
        "--trim-mode",
        default="Sharpened soft routing",
        help="Mode used for trimmed token-level strips.",
    )
    parser.add_argument(
        "--trim-case",
        action="append",
        type=parse_case_trim,
        default=None,
        help=(
            "Trimmed token strip spec: CASE, CASE:START, or CASE:START:COUNT. "
            "Can be repeated. Defaults to Occitan and Code-Switch."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary_rows = read_csv(Path(args.summary_csv))
    token_rows = read_csv(Path(args.tokens_csv))

    plot_mean_activation_grid(
        summary_rows,
        output_dir / f"mean_activation_sequence_{args.scope}.png",
        level="Sequence-level",
        scope=args.scope,
    )
    plot_mean_activation_grid(
        summary_rows,
        output_dir / f"mean_activation_token_{args.scope}.png",
        level="Token-level",
        scope=args.scope,
    )
    write_latex_table(
        summary_rows,
        output_dir / f"mean_activation_sequence_{args.scope}.tex",
        level="Sequence-level",
        scope=args.scope,
    )
    write_latex_table(
        summary_rows,
        output_dir / f"mean_activation_token_{args.scope}.tex",
        level="Token-level",
        scope=args.scope,
    )

    trim_specs = args.trim_case or [("Occitan", 0, 40), ("Code-Switch", 0, 40)]
    for case, start, count in trim_specs:
        plot_trimmed_token_heatmap(
            token_rows,
            output_dir
            / f"trimmed_token_heatmap__{slug(args.trim_mode)}__{slug(case)}.png",
            mode=args.trim_mode,
            case=case,
            start=start,
            max_tokens=count,
        )

    print(f"Wrote paper routing figures to: {output_dir}")


if __name__ == "__main__":
    main()
