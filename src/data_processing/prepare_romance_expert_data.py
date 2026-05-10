"""
Prepare French and Catalan Alpaca-style data for Romance expert LoRAs.

Example:
    python -m src.data_processing.prepare_romance_expert_data --output_dir data/cousin_data
"""

import json
import os
import random
import argparse
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

CATALAN_TARGET = 60_000
FRENCH_TARGET = CATALAN_TARGET * 2  # 120k
DEFAULT_FR_DATASET = "jpacifico/French-Alpaca-dataset-Instruct-110K"
DEFAULT_CA_DATASET = "saillab/alpaca-catalan-cleaned"

ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

ALPACA_TEMPLATE_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def clean_input(val):
    if val is None:
        return ""
    if isinstance(val, float) and val != val:
        return ""
    val_str = str(val).strip()
    return "" if val_str.lower() == "nan" else val_str


def _contains_unk(*values) -> bool:
    return any("<unk>" in str(value) for value in values)


def format_alpaca(instruction: str, inp: str, output: str) -> str:
    if inp:
        return ALPACA_TEMPLATE.format(instruction=instruction, input=inp, output=output)
    return ALPACA_TEMPLATE_NO_INPUT.format(instruction=instruction, output=output)


def collect_from_hf(hf_name: str, target: int, lang_label: str, seed: int) -> list[dict]:
    """Load, filter, format, and return up to *target* examples from a HF Alpaca dataset."""
    print(f"\nLoading {lang_label} data from '{hf_name}'...")
    ds = load_dataset(hf_name, split="train")
    ds = ds.shuffle(seed=seed)

    collected = []
    skipped = 0
    unk_filtered = 0
    seen = set()
    for row in ds:
        instruction = str(row.get("instruction", "")).strip()
        output = str(row.get("output", "")).strip()
        if not instruction or not output:
            skipped += 1
            continue

        inp = clean_input(row.get("input"))
        if _contains_unk(instruction, inp, output):
            unk_filtered += 1
            continue

        row_key = (instruction, inp, output)
        if row_key in seen:
            continue
        seen.add(row_key)

        text = format_alpaca(instruction, inp, output)
        collected.append({"text": text, "lang": lang_label})

        if len(collected) >= target:
            break

    print(
        f"  Collected {len(collected):,} / {target:,}  "
        f"(skipped {skipped} empty rows, filtered {unk_filtered} <unk> rows)"
    )
    if len(collected) < target:
        print(f"  Warning: only {len(collected):,} valid rows available in '{hf_name}'")
    return collected


def main(args):
    random.seed(args.seed)
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fr_target = args.fr_target
    ca_target = args.ca_target

    print("Rich Cousin Data Preparation  (Alpaca Instruct)")
    print("=" * 60)
    print(f"French target:  {fr_target:,}")
    print(f"Catalan target: {ca_target:,}")
    print(f"Ratio:          {fr_target / ca_target:.0f}:1")
    print(f"Output dir:     {out_dir}")

    print(f"French dataset:  {args.fr_dataset}")
    print(f"Catalan dataset: {args.ca_dataset}")

    fr_data = collect_from_hf(args.fr_dataset, fr_target, "fr", args.seed)
    ca_data = collect_from_hf(args.ca_dataset, ca_target, "ca", args.seed)

    for lang_label, data, filename in [
        ("French", fr_data, "cousin_fr.jsonl"),
        ("Catalan", ca_data, "cousin_ca.jsonl"),
    ]:
        path = out_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(data):,} {lang_label} examples -> {path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 60)
    print("ALL DONE")
    print(f"  French:  {len(fr_data):,}  ({out_dir / 'cousin_fr.jsonl'})")
    print(f"  Catalan: {len(ca_data):,}  ({out_dir / 'cousin_ca.jsonl'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Alpaca-style data for Rich Cousin LoRA adapters")
    parser.add_argument("--output_dir", default="data/cousin_data", help="Output directory (relative to project root)")
    parser.add_argument("--fr_target", type=int, default=FRENCH_TARGET, help=f"French examples (default: {FRENCH_TARGET:,})")
    parser.add_argument("--ca_target", type=int, default=CATALAN_TARGET, help=f"Catalan examples (default: {CATALAN_TARGET:,})")
    parser.add_argument(
        "--fr_dataset",
        default=DEFAULT_FR_DATASET,
        help=f"French Hugging Face dataset id (default: {DEFAULT_FR_DATASET})",
    )
    parser.add_argument(
        "--ca_dataset",
        default=DEFAULT_CA_DATASET,
        help=f"Catalan Hugging Face dataset id (default: {DEFAULT_CA_DATASET})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
