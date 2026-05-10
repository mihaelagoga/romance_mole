"""
Build a compact French repair set for the French Romance expert.

Example:
    python -m src.data_processing.build_french_repair_set --output data/cousin_data/cousin_fr_repair.jsonl
"""

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

from datasets import concatenate_datasets, load_dataset

OUTPUT_DEFAULT = "data/cousin_data/cousin_fr_repair.jsonl"

DEFAULT_SOURCES = [
    "OpenAssistant/oasst2",
    "tbboukhari/Alpaca_french_instruct",
    "AIffl/Alpaca_french_mixtral",
    "timpearce/alpaca-cleaned-french",
]

DEFAULT_LIMITS = {
    "OpenAssistant/oasst2": 2500,
    "tbboukhari/Alpaca_french_instruct": 750,
    "AIffl/Alpaca_french_mixtral": 750,
    "timpearce/alpaca-cleaned-french": 750,
}

ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)
ALPACA_TEMPLATE_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _contains_unk(*values) -> bool:
    return any("<unk>" in str(value) for value in values)


def _normalize_text_key(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def format_alpaca(instruction: str, inp: str, output: str) -> str:
    if inp:
        return ALPACA_TEMPLATE.format(instruction=instruction, input=inp, output=output)
    return ALPACA_TEMPLATE_NO_INPUT.format(instruction=instruction, output=output)


def _record_text(
    text: str,
    *,
    seen_texts: set[str],
    sink: list[dict],
    stats: dict[str, int],
) -> None:
    normalized = _normalize_text_key(text)
    if not normalized:
        stats["skipped_empty"] += 1
        return
    if _contains_unk(normalized):
        stats["filtered_unk"] += 1
        return
    if normalized in seen_texts:
        stats["duplicates_removed"] += 1
        return

    seen_texts.add(normalized)
    sink.append({"text": normalized, "lang": "fr"})
    stats["kept"] += 1


def _pick_first(row: dict, candidates: list[str]) -> str:
    for key in candidates:
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def _concat_splits(dataset_name: str):
    loaded = load_dataset(dataset_name)
    splits = [loaded[split_name] for split_name in loaded.keys()]
    if not splits:
        raise ValueError(f"No splits found for dataset '{dataset_name}'")
    if len(splits) == 1:
        return splits[0]
    return concatenate_datasets(splits)


def load_oasst2_pairs(
    dataset_name: str,
    *,
    target_count: int,
    seed: int,
    seen_texts: set[str],
    sink: list[dict],
) -> dict[str, int]:
    stats = {"loaded": 0, "kept": 0, "skipped_empty": 0, "filtered_unk": 0, "duplicates_removed": 0}
    print(f"\nLoading repair source: {dataset_name}")

    dataset = _concat_splits(dataset_name).shuffle(seed=seed)
    rows_by_id = {}
    for row in dataset:
        rows_by_id[str(row.get("message_id"))] = row

    for row in dataset:
        stats["loaded"] += 1
        if clean_text(row.get("role")) != "assistant":
            continue
        if clean_text(row.get("lang")) != "fr":
            continue
        if bool(row.get("deleted")):
            continue
        review_result = row.get("review_result")
        if review_result is False:
            continue

        parent_id = clean_text(row.get("parent_id"))
        if not parent_id or parent_id not in rows_by_id:
            continue

        parent = rows_by_id[parent_id]
        if clean_text(parent.get("role")) != "prompter":
            continue
        if clean_text(parent.get("lang")) != "fr":
            continue
        if bool(parent.get("deleted")):
            continue
        parent_review_result = parent.get("review_result")
        if parent_review_result is False:
            continue

        instruction = clean_text(parent.get("text"))
        output = clean_text(row.get("text"))
        if not instruction or not output:
            stats["skipped_empty"] += 1
            continue
        if _contains_unk(instruction, output):
            stats["filtered_unk"] += 1
            continue

        text = format_alpaca(instruction, "", output)
        _record_text(text, seen_texts=seen_texts, sink=sink, stats=stats)
        if stats["kept"] >= target_count:
            break

    return stats


def load_generic_instruct_dataset(
    dataset_name: str,
    *,
    target_count: int,
    seed: int,
    seen_texts: set[str],
    sink: list[dict],
) -> dict[str, int]:
    stats = {"loaded": 0, "kept": 0, "skipped_empty": 0, "filtered_unk": 0, "duplicates_removed": 0}
    print(f"\nLoading repair source: {dataset_name}")

    dataset = _concat_splits(dataset_name).shuffle(seed=seed)
    for row in dataset:
        stats["loaded"] += 1
        instruction = _pick_first(row, ["instruction", "Instruction"])
        inp = _pick_first(row, ["input", "Input", "saisir"])
        output = _pick_first(row, ["output", "Output", "sortir"])

        if not instruction or not output:
            stats["skipped_empty"] += 1
            continue
        if _contains_unk(instruction, inp, output):
            stats["filtered_unk"] += 1
            continue

        text = format_alpaca(instruction, inp, output)
        _record_text(text, seen_texts=seen_texts, sink=sink, stats=stats)
        if stats["kept"] >= target_count:
            break

    return stats


def _load_source(
    dataset_name: str,
    *,
    target_count: int,
    seed: int,
    seen_texts: set[str],
    sink: list[dict],
) -> dict[str, int]:
    if dataset_name == "OpenAssistant/oasst2":
        return load_oasst2_pairs(
            dataset_name,
            target_count=target_count,
            seed=seed,
            seen_texts=seen_texts,
            sink=sink,
        )
    return load_generic_instruct_dataset(
        dataset_name,
        target_count=target_count,
        seed=seed,
        seen_texts=seen_texts,
        sink=sink,
    )


def print_stats(label: str, stats: dict[str, int]) -> None:
    print(label)
    print(f"  - Loaded rows:        {stats['loaded']:,}")
    print(f"  - Kept rows:          {stats['kept']:,}")
    print(f"  - Empty skipped:      {stats['skipped_empty']:,}")
    print(f"  - <unk> filtered:     {stats['filtered_unk']:,}")
    print(f"  - Duplicates removed: {stats['duplicates_removed']:,}")


def main(args):
    output_path = _resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_texts: set[str] = set()
    merged_rows: list[dict] = []

    print("Building French Repair Set")
    print("=" * 60)
    print(f"Sources:      {args.sources}")
    print(f"Output path:  {output_path}")
    print(f"Seed:         {args.seed}")

    for dataset_name in args.sources:
        target_count = int(args.source_limits.get(dataset_name, DEFAULT_LIMITS.get(dataset_name, args.default_limit)))
        stats = _load_source(
            dataset_name,
            target_count=target_count,
            seed=args.seed,
            seen_texts=seen_texts,
            sink=merged_rows,
        )
        print_stats(f"Stats for {dataset_name} (target={target_count:,}):", stats)

    with open(output_path, "w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("DONE")
    print(f"Final repair rows: {len(merged_rows):,}")
    print(f"Saved to:          {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a French repair set for the French cousin adapter")
    parser.add_argument(
        "--output_path",
        default=OUTPUT_DEFAULT,
        help="Output JSONL path (relative to project root by default).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        help="Hugging Face dataset ids to include in the repair set.",
    )
    parser.add_argument(
        "--default_limit",
        type=int,
        default=750,
        help="Fallback per-source sample cap when no special limit is defined.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset shuffling.")
    args = parser.parse_args()
    args.source_limits = dict(DEFAULT_LIMITS)
    main(args)
