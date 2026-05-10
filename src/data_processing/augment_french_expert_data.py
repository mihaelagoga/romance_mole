"""
Augment the French Romance expert training corpus.

Example:
    python -m src.data_processing.augment_french_expert_data --local-source data/cousin_data/cousin_fr_seed.jsonl --output-path data/cousin_data/cousin_fr_augmented.jsonl
"""

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Load project-root .env into os.environ so HF_TOKEN is seen by huggingface_hub."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

from datasets import load_dataset

LOCAL_SOURCE_DEFAULT = "data/cousin_data/cousin_fr_seed.jsonl"
OUTPUT_DEFAULT = "data/cousin_data/cousin_fr_augmented.jsonl"

HF_SOURCES_DEFAULT = [
    "tbboukhari/Alpaca_french_instruct",
    "AIffl/Alpaca_french_mixtral",
    "timpearce/alpaca-cleaned-french",
]

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


def format_alpaca(instruction: str, inp: str, output: str) -> str:
    if inp:
        return ALPACA_TEMPLATE.format(instruction=instruction, input=inp, output=output)
    return ALPACA_TEMPLATE_NO_INPUT.format(instruction=instruction, output=output)


def _contains_unk(*values) -> bool:
    return any("<unk>" in str(value) for value in values)


def _normalize_text_key(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def _record_row(
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


def load_local_jsonl(path: Path, *, seen_texts: set[str], sink: list[dict]) -> dict[str, int]:
    stats = {"loaded": 0, "kept": 0, "skipped_empty": 0, "filtered_unk": 0, "duplicates_removed": 0}
    print(f"\nLoading local source: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats["loaded"] += 1
            row = json.loads(line)
            text = clean_text(row.get("text"))
            _record_row(text, seen_texts=seen_texts, sink=sink, stats=stats)

    return stats


def _pick_first(row: dict, candidates: list[str]) -> str:
    for key in candidates:
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def load_hf_instruct_dataset(hf_name: str, *, seen_texts: set[str], sink: list[dict], seed: int) -> dict[str, int]:
    stats = {"loaded": 0, "kept": 0, "skipped_empty": 0, "filtered_unk": 0, "duplicates_removed": 0}
    print(f"\nLoading HF source: {hf_name}")

    dataset = load_dataset(hf_name, split="train").shuffle(seed=seed)
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
        _record_row(text, seen_texts=seen_texts, sink=sink, stats=stats)

    return stats


def print_stats(label: str, stats: dict[str, int]) -> None:
    print(label)
    print(f"  - Loaded rows:        {stats['loaded']:,}")
    print(f"  - Kept rows:          {stats['kept']:,}")
    print(f"  - Empty skipped:      {stats['skipped_empty']:,}")
    print(f"  - <unk> filtered:     {stats['filtered_unk']:,}")
    print(f"  - Duplicates removed: {stats['duplicates_removed']:,}")


def main(args):
    local_source = _resolve_path(args.local_source) if args.local_source else None
    output_path = _resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_texts: set[str] = set()
    merged_rows: list[dict] = []

    print("Augmenting French Cousin Dataset")
    print("=" * 60)
    print(f"Local source:  {local_source or '(none)'}")
    print(f"HF sources:    {args.hf_sources}")
    print(f"Output path:   {output_path}")

    if local_source is not None:
        if not local_source.exists():
            raise FileNotFoundError(
                f"Local source not found: {local_source}. "
                "Pass --local-source to your seed French JSONL, or pass --no-local-source to use only HF sources."
            )
        local_stats = load_local_jsonl(local_source, seen_texts=seen_texts, sink=merged_rows)
        print_stats("Local source stats:", local_stats)

    hf_stats = {}
    for hf_name in args.hf_sources:
        stats = load_hf_instruct_dataset(hf_name, seen_texts=seen_texts, sink=merged_rows, seed=args.seed)
        hf_stats[hf_name] = stats
        print_stats(f"Stats for {hf_name}:", stats)

    with open(output_path, "w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("DONE")
    print(f"Final unique rows: {len(merged_rows):,}")
    print(f"Saved to:          {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment and deduplicate the French cousin dataset")
    parser.add_argument(
        "--local-source",
        "--local_source",
        dest="local_source",
        default=LOCAL_SOURCE_DEFAULT,
        help="Local French JSONL to augment (relative to project root by default).",
    )
    parser.add_argument(
        "--no-local-source",
        action="store_const",
        const=None,
        dest="local_source",
        help="Use only the Hugging Face instruction sources.",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        dest="output_path",
        default=OUTPUT_DEFAULT,
        help="Output JSONL path (relative to project root by default).",
    )
    parser.add_argument(
        "--hf-sources",
        "--hf_sources",
        dest="hf_sources",
        nargs="+",
        default=HF_SOURCES_DEFAULT,
        help="French Hugging Face dataset ids to merge into the local source.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for HF dataset shuffling.")
    main(parser.parse_args())
