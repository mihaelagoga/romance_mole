"""
Create deterministic splits for synthetic Occitan data.

Example:
  python -m src.data_processing.split_synthetic --output-dir data/synthetic/synth_splits
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIALOGUE = PROJECT_ROOT / "data" / "synthetic" / "dialogocc" / "synthetic_dialogue_250.jsonl"
DEFAULT_FLORES = PROJECT_ROOT / "data" / "synthetic" / "flores200occ" / "synthetic_flores_500.jsonl"
DEFAULT_MORPHO = PROJECT_ROOT / "data" / "synthetic" / "morphostressocc" / "synthetic_stress_test_250.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic" / "synth_splits"

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_DEV_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_SEED = 42


@dataclass
class SourceSpec:
  name: str
  path: Path


def normalize_text(value) -> str:
  return str(value or "").strip()


def read_dialogue(path: Path) -> list[dict]:
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
      a = normalize_text(item.get("speaker_A"))
      b = normalize_text(item.get("speaker_B"))
      if not a and not b:
        continue
      if a and b:
        text = f"{a}\n{b}"
      else:
        text = a or b
      rows.append(
        {
          "text": text,
          "source": "synthetic_dialogue",
          "source_type": "dialogue",
          "scenario": normalize_text(item.get("scenario")),
        }
      )
  return rows


def read_flores(path: Path) -> list[dict]:
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
      text = normalize_text(item.get("target_text") or item.get("text"))
      if not text:
        continue
      rows.append(
        {
          "text": text,
          "source": "synthetic_flores200occ",
          "source_type": "flores_localization",
        }
      )
  return rows


def read_morpho(path: Path) -> list[dict]:
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
      text = normalize_text(item.get("text"))
      if not text:
        continue
      rows.append(
        {
          "text": text,
          "source": "synthetic_morphostressocc",
          "source_type": normalize_text(item.get("category")) or "morphological_stress",
        }
      )
  return rows


def split_rows(rows: list[dict], train_ratio: float, dev_ratio: float, test_ratio: float, rng: random.Random):
  if not rows:
    return [], [], []
  shuffled = list(rows)
  rng.shuffle(shuffled)

  n = len(shuffled)
  n_train = int(n * train_ratio)
  n_dev = int(n * dev_ratio)
  n_test = n n_train n_dev

  if n >= 10:
    if n_train == 0:
      n_train = 1
    if n_dev == 0:
      n_dev = 1
    n_test = n n_train n_dev
    if n_test <= 0:
      n_test = 1
      n_train = max(1, n_train 1)

  train_rows = shuffled[:n_train]
  dev_rows = shuffled[n_train : n_train + n_dev]
  test_rows = shuffled[n_train + n_dev :]
  return train_rows, dev_rows, test_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
  with open(path, "w", encoding="utf-8") as f:
    for item in rows:
      f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args():
  parser = argparse.ArgumentParser(description="Prepare synthetic 80/10/10 splits.")
  parser.add_argument("--dialogue", type=Path, default=DEFAULT_DIALOGUE, help="Dialogue JSONL source.")
  parser.add_argument("--flores", type=Path, default=DEFAULT_FLORES, help="FLORES synthetic JSONL source.")
  parser.add_argument("--morpho", type=Path, default=DEFAULT_MORPHO, help="Morphological stress JSONL source.")
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
  parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Train ratio.")
  parser.add_argument("--dev-ratio", type=float, default=DEFAULT_DEV_RATIO, help="Dev ratio.")
  parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="Test ratio.")
  parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed.")
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  ratio_sum = args.train_ratio + args.dev_ratio + args.test_ratio
  if abs(ratio_sum 1.0) > 1e-8:
    raise ValueError(
      f"Ratios must sum to 1.0; got {ratio_sum:.6f} "
      f"({args.train_ratio}, {args.dev_ratio}, {args.test_ratio})."
    )

  sources = [
    SourceSpec("dialogue", args.dialogue),
    SourceSpec("flores", args.flores),
    SourceSpec("morpho", args.morpho),
  ]
  for src in sources:
    if not src.path.exists():
      raise FileNotFoundError(f"Input file not found for {src.name}: {src.path}")

  print("=" * 60)
  print("PREPARE synthetic SPLITS")
  print("=" * 60)
  print(f"Dialogue:  {args.dialogue}")
  print(f"FLORES:   {args.flores}")
  print(f"Morpho:   {args.morpho}")
  print(f"Output dir: {args.output_dir}")
  print(f"Ratios:   train={args.train_ratio}, dev={args.dev_ratio}, test={args.test_ratio}")
  print(f"Seed:    {args.seed}")
  print("=" * 60)

  readers = {
    "dialogue": read_dialogue,
    "flores": read_flores,
    "morpho": read_morpho,
  }
  rng = random.Random(args.seed)

  merged_train: list[dict] = []
  merged_dev: list[dict] = []
  merged_test: list[dict] = []
  per_source_meta = {}

  for src in sources:
    rows = readers[src.name](src.path)
    train_rows, dev_rows, test_rows = split_rows(
      rows, args.train_ratio, args.dev_ratio, args.test_ratio, rng
    )
    merged_train.extend(train_rows)
    merged_dev.extend(dev_rows)
    merged_test.extend(test_rows)

    per_source_meta[src.name] = {
      "input_file": str(src.path),
      "total_rows": len(rows),
      "train_rows": len(train_rows),
      "dev_rows": len(dev_rows),
      "test_rows": len(test_rows),
    }
    print(
      f"{src.name:>8}: total={len(rows):>4} "
      f"train={len(train_rows):>4} dev={len(dev_rows):>4} test={len(test_rows):>4}"
    )

  rng.shuffle(merged_train)
  rng.shuffle(merged_dev)
  rng.shuffle(merged_test)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  train_path = args.output_dir / "train_synth.jsonl"
  dev_path = args.output_dir / "dev_synth.jsonl"
  test_path = args.output_dir / "test_synth.jsonl"
  meta_path = args.output_dir / "synth_split_metadata.json"

  write_jsonl(train_path, merged_train)
  write_jsonl(dev_path, merged_dev)
  write_jsonl(test_path, merged_test)

  metadata = {
    "seed": args.seed,
    "ratios": {
      "train": args.train_ratio,
      "dev": args.dev_ratio,
      "test": args.test_ratio,
    },
    "totals": {
      "train_rows": len(merged_train),
      "dev_rows": len(merged_dev),
      "test_rows": len(merged_test),
      "all_rows": len(merged_train) + len(merged_dev) + len(merged_test),
    },
    "files": {
      "train_file": str(train_path),
      "dev_file": str(dev_path),
      "test_file": str(test_path),
    },
    "per_source": per_source_meta,
  }
  with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

  print("Done.")
  print(f" Train: {len(merged_train):,} -> {train_path}")
  print(f" Dev:  {len(merged_dev):,} -> {dev_path}")
  print(f" Test: {len(merged_test):,} -> {test_path}")
  print(f" Meta: {meta_path}")


if __name__ == "__main__":
  main()
