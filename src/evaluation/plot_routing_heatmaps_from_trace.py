"""
Plot token-level and sequence-level routing heatmaps from a smoke-test trace.

Example:
    python -m src.evaluation.plot_routing_heatmaps_from_trace --input artifacts/routing_smoke_trace_t8_aux001_sup002_cs000.txt --output-dir results/routing_heatmaps --response-only --also-full --max-tokens 90
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt


EXPERTS = ("fr", "ca", "oc")
MODE_NAMES = {
    "Soft/default routing",
    "Sharpened soft routing",
    "Hard argmax routing",
}


@dataclass
class RoutingRow:
    checkpoint: str
    mode: str
    case: str
    layer_name: str
    level: str
    token_index: int
    token: str
    fr: float
    ca: float
    oc: float
    in_response: bool


@dataclass
class RoutingTable:
    checkpoint: str
    mode: str
    case: str
    layer_name: str
    level: str
    rows: list[RoutingRow] = field(default_factory=list)


def slug(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("/", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_<>-]+", "", text)
    text = text.replace("<-", "to_").replace("->", "to_")
    return text.strip("_")


def short_token(token: str, max_len: int = 14) -> str:
    token = token.replace("\\n", "\\\\n")
    if token == "":
        token = "[space]"
    if len(token) <= max_len:
        return token
    return token[: max_len - 3] + "..."


def parse_probability_row(line: str) -> tuple[str, float, float, float] | None:
    if "|" not in line:
        return None
    # Token strings can contain pipe characters, e.g. <|begin_of_text|>.
    parts = line.rsplit("|", 3)
    if len(parts) != 4:
        return None
    token = parts[0].rstrip()
    try:
        fr = float(parts[1].strip())
        ca = float(parts[2].strip())
        oc = float(parts[3].strip())
    except ValueError:
        return None
    return token, fr, ca, oc


def mark_response_rows(raw_rows: list[tuple[str, float, float, float]]) -> list[bool]:
    flags = [False] * len(raw_rows)
    response_idx = None
    for idx, (token, *_vals) in enumerate(raw_rows):
        if token.strip() == "Response":
            response_idx = idx
            break
    if response_idx is None:
        return flags

    start = response_idx + 1
    if start < len(raw_rows) and raw_rows[start][0].strip() in {":\\n", ":"}:
        start += 1

    for idx in range(start, len(raw_rows)):
        token = raw_rows[idx][0].strip()
        if token.startswith("<|end_of_text|>"):
            break
        flags[idx] = True
    return flags


def parse_smoke_file(path: Path) -> tuple[str, list[RoutingTable]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty smoke file: {path}")

    checkpoint = lines[0].strip()
    current_mode: str | None = None
    current_case: str | None = None
    tables: list[RoutingTable] = []
    i = 1

    while i < len(lines):
        line = lines[i].strip()
        if line in MODE_NAMES:
            current_mode = line
            current_case = None
            i += 1
            continue

        if line.startswith("Smoke Test Case:"):
            current_case = line.split(":", 1)[1].strip()
            i += 1
            continue

        if line.startswith("Showing Probabilities for Layer:"):
            if current_mode is None:
                raise ValueError(f"Found routing table before mode at line {i + 1}")
            if current_case is None:
                # the first case. The fixed smoke-test order is French first.
                current_case = "French"
            match = re.match(r"Showing Probabilities for Layer:\s*(.*?)\s*\((.*?)\)", line)
            if not match:
                raise ValueError(f"Could not parse layer line: {line}")
            layer_name = match.group(1).strip()
            level = match.group(2).strip()

            i += 1
            while i < len(lines) and not lines[i].startswith("Token"):
                i += 1
            i += 2

            raw_rows: list[tuple[str, float, float, float]] = []
            while i < len(lines):
                candidate = lines[i]
                stripped = candidate.strip()
                if not stripped:
                    break
                if stripped in MODE_NAMES or stripped.startswith("Smoke Test Case:"):
                    break
                if stripped.startswith("Showing Probabilities for Layer:"):
                    break
                parsed = parse_probability_row(candidate)
                if parsed is None:
                    break
                raw_rows.append(parsed)
                i += 1

            in_response = mark_response_rows(raw_rows)
            table = RoutingTable(
                checkpoint=checkpoint,
                mode=current_mode,
                case=current_case,
                layer_name=layer_name,
                level=level,
            )
            for idx, ((token, fr, ca, oc), flag) in enumerate(zip(raw_rows, in_response)):
                table.rows.append(
                    RoutingRow(
                        checkpoint=checkpoint,
                        mode=current_mode,
                        case=current_case,
                        layer_name=layer_name,
                        level=level,
                        token_index=idx,
                        token=token,
                        fr=fr,
                        ca=ca,
                        oc=oc,
                        in_response=flag,
                    )
                )
            tables.append(table)
            continue

        i += 1

    if not tables:
        raise ValueError(f"No routing tables parsed from {path}")
    return checkpoint, tables


def write_rows_csv(path: Path, tables: list[RoutingTable]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checkpoint",
                "mode",
                "case",
                "layer_name",
                "level",
                "token_index",
                "token",
                "fr",
                "ca",
                "oc",
                "in_response",
            ],
        )
        writer.writeheader()
        for table in tables:
            for row in table.rows:
                writer.writerow(row.__dict__)


def write_summary_csv(path: Path, tables: list[RoutingTable]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checkpoint",
                "mode",
                "case",
                "level",
                "layer_name",
                "scope",
                "num_tokens",
                "fr_mean",
                "ca_mean",
                "oc_mean",
                "top_expert",
                "top_mean",
            ],
        )
        writer.writeheader()
        for table in tables:
            for scope, rows in (
                ("full", table.rows),
                ("response", [row for row in table.rows if row.in_response]),
            ):
                if not rows:
                    continue
                means = {
                    expert: sum(getattr(row, expert) for row in rows) / len(rows)
                    for expert in EXPERTS
                }
                top_expert = max(EXPERTS, key=lambda expert: means[expert])
                writer.writerow(
                    {
                        "checkpoint": table.checkpoint,
                        "mode": table.mode,
                        "case": table.case,
                        "level": table.level,
                        "layer_name": table.layer_name,
                        "scope": scope,
                        "num_tokens": len(rows),
                        "fr_mean": means["fr"],
                        "ca_mean": means["ca"],
                        "oc_mean": means["oc"],
                        "top_expert": top_expert,
                        "top_mean": means[top_expert],
                    }
                )


def make_heatmap(
    table: RoutingTable,
    out_path: Path,
    *,
    response_only: bool,
    max_tokens: int,
    dpi: int,
) -> None:
    rows = [row for row in table.rows if row.in_response] if response_only else list(table.rows)
    if not rows:
        return
    rows = rows[:max_tokens]

    data = [[getattr(row, expert) for row in rows] for expert in EXPERTS]
    labels = [short_token(row.token) for row in rows]

    width = max(8.0, min(28.0, 0.32 * len(rows)))
    fig, ax = plt.subplots(figsize=(width, 3.2))
    im = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")

    ax.set_yticks(range(len(EXPERTS)))
    ax.set_yticklabels(EXPERTS)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    scope = "response tokens" if response_only else "full prompt+response"
    ax.set_title(f"{table.mode} | {table.case} | {table.level} | {scope}", fontsize=10)
    ax.set_xlabel("Token")
    ax.set_ylabel("Expert")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Routing probability")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MoLE expert-routing heatmaps from a smoke-test text dump."
    )
    parser.add_argument("--input", required=True, help="Smoke-test routing text file.")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG/CSV outputs.")
    parser.add_argument(
        "--response-only",
        action="store_true",
        help="Plot only generated response tokens instead of full prompt+response.",
    )
    parser.add_argument(
        "--also-full",
        action="store_true",
        help="If --response-only is set, also emit full prompt+response heatmaps.",
    )
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["Token-level", "Sequence-level"],
        help="Routing table levels to plot.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Optional mode filter, e.g. 'Sharpened soft routing'.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional case filter, e.g. French Catalan Occitan Code-Switch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    checkpoint, tables = parse_smoke_file(input_path)

    if args.modes:
        wanted_modes = set(args.modes)
        tables = [table for table in tables if table.mode in wanted_modes]
    if args.cases:
        wanted_cases = set(args.cases)
        tables = [table for table in tables if table.case in wanted_cases]
    wanted_levels = set(args.levels)
    tables = [table for table in tables if table.level in wanted_levels]

    if not tables:
        raise ValueError("No routing tables remained after filters.")

    write_rows_csv(output_dir / "routing_tokens.csv", tables)
    write_summary_csv(output_dir / "routing_summary.csv", tables)

    scopes = ["response"] if args.response_only else ["full"]
    if args.response_only and args.also_full:
        scopes.append("full")

    for table in tables:
        for scope in scopes:
            response_only = scope == "response"
            filename = (
                f"{slug(table.mode)}__{slug(table.case)}__"
                f"{slug(table.level)}__{scope}.png"
            )
            make_heatmap(
                table,
                output_dir / filename,
                response_only=response_only,
                max_tokens=args.max_tokens,
                dpi=args.dpi,
            )

    print(f"Parsed checkpoint: {checkpoint}")
    print(f"Parsed/plotted tables: {len(tables)}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
