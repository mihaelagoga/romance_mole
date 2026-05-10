"""
Create deterministic train/dev/test splits for Alpaca Occitan data.

Example:
  python -m src.data_processing.split_alpaca_occitan --input data/synthetic/alpaca_occitan_v2/alpaca_occitan_v2.jsonl --output-dir data/synthetic/alpaca_occitan/splits
"""

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "synthetic" / "alpaca_occitan_v2" / "alpaca_occitan_v2.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic" / "alpaca_occitan" / "splits"

DEFAULT_DEV_SIZE = 200
DEFAULT_TEST_SIZE = 200
DEFAULT_SEED = 42


def normalize_record(item: dict) -> dict | None:
  instruction = str(item.get("instruction", "")).strip()
  model_input = str(item.get("input", "")).strip()
  output = str(item.get("output", "")).strip()
  if not instruction or not output:
    return None
  return {
    "instruction": instruction,
    "input": model_input,
    "output": output,
  }


def read_rows(path: Path) -> list[dict]:
  rows: list[dict] = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        item = json.loads(line)
      except json.JSONDecodeError:
        continue
      normalized = normalize_record(item)
      if normalized is not None:
        rows.append(normalized)
  return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
  with open(path, "w", encoding="utf-8") as f:
    for item in rows:
      f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
  parser = argparse.ArgumentParser(description="Prepare Alpaca train/dev/test splits.")
  parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input Alpaca Occitan JSONL.")
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
  parser.add_argument("--dev-size", type=int, default=DEFAULT_DEV_SIZE, help="Dev split size.")
  parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE, help="Test split size.")
  parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed.")
  args = parser.parse_args()

  print("=" * 60)
  print("PREPARE ALPACA OCCITAN SPLITS ")
  print("=" * 60)
  print(f"Input:   {args.input}")
  print(f"Output:  {args.output_dir}")
  print(f"Dev size: {args.dev_size}")
  print(f"Test size: {args.test_size}")
  print(f"Seed:   {args.seed}")
  print("=" * 60)

  if not args.input.exists():
    raise FileNotFoundError(f"Input file not found: {args.input}")

  rows = read_rows(args.input)
  total = len(rows)
  if total <= args.dev_size + args.test_size:
    raise ValueError(
      f"Not enough rows ({total}) for dev+test="
      f"{args.dev_size + args.test_size}."
    )

  rng = random.Random(args.seed)
  rng.shuffle(rows)

  dev_rows = rows[: args.dev_size]
  test_rows = rows[args.dev_size : args.dev_size + args.test_size]
  train_rows = rows[args.dev_size + args.test_size :]

  args.output_dir.mkdir(parents=True, exist_ok=True)
  train_path = args.output_dir / "train_alpaca.jsonl"
  dev_path = args.output_dir / "dev_alpaca_200.jsonl"
  test_path = args.output_dir / "test_alpaca_200.jsonl"
  meta_path = args.output_dir / "split_metadata.json"

  print("Writing split files...")
  write_jsonl(train_path, train_rows)
  write_jsonl(dev_path, dev_rows)
  write_jsonl(test_path, test_rows)

  metadata = {
    "input_file": str(args.input),
    "seed": args.seed,
    "total_rows": total,
    "train_rows": len(train_rows),
    "dev_rows": len(dev_rows),
    "test_rows": len(test_rows),
    "train_file": str(train_path),
    "dev_file": str(dev_path),
    "test_file": str(test_path),
  }
  with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

  print("Done.")
  print(f" Train: {len(train_rows):,} -> {train_path}")
  print(f" Dev:  {len(dev_rows):,} -> {dev_path}")
  print(f" Test: {len(test_rows):,} -> {test_path}")
  print(f" Meta: {meta_path}")


if __name__ == "__main__":
  main()
