"""
Create deterministic HPLT train/dev/test splits.

Example:
  python -m src.data_processing.split_hplt --input data/hplt_v3/cleaned/all_data.jsonl --output-dir data/hplt_v3/splits
"""

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "hplt_v3" / "cleaned" / "all_data.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "hplt_v3" / "splits"

DEFAULT_DEV_SIZE = 1000
DEFAULT_TEST_SIZE = 2000
DEFAULT_SEED = 42


def read_text_rows(path: Path) -> list[dict]:
  rows: list[dict] = []
  with open(path, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        item = json.loads(line)
      except json.JSONDecodeError:
        continue
      text = str(item.get("text", "")).strip()
      if not text:
        continue
      rows.append({"text": text})
      if line_no % 50000 == 0:
        print(f" Parsed {line_no:,} lines...")
  return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
  with open(path, "w", encoding="utf-8") as f:
    for item in rows:
      f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
  parser = argparse.ArgumentParser(description="Prepare HPLT train/dev/test splits.")
  parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input HPLT JSONL.")
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
  parser.add_argument("--dev-size", type=int, default=DEFAULT_DEV_SIZE, help="Dev split size.")
  parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE, help="Test split size.")
  parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed.")
  args = parser.parse_args()

  print("=" * 60)
  print("PREPARE HPLT SPLITS ")
  print("=" * 60)
  print(f"Input:   {args.input}")
  print(f"Output:  {args.output_dir}")
  print(f"Dev size: {args.dev_size}")
  print(f"Test size: {args.test_size}")
  print(f"Seed:   {args.seed}")
  print("=" * 60)

  if not args.input.exists():
    raise FileNotFoundError(f"Input file not found: {args.input}")

  rows = read_text_rows(args.input)
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
  train_path = args.output_dir / "train_hplt.jsonl"
  dev_path = args.output_dir / "dev_hplt_1000.jsonl"
  test_path = args.output_dir / "test_hplt_2000.jsonl"
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
