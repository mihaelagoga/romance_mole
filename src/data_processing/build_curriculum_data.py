"""
Merge synthetic and Alpaca Occitan data for curriculum training.

Example:
  python -m src.data_processing.build_curriculum_data --output-dir data/synthetic/merged_splits
"""

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_STAGE3B_DIR = PROJECT_ROOT / "data" / "synthetic" / "synth_splits"
DEFAULT_ALPACA_DIR = PROJECT_ROOT / "data" / "synthetic" / "alpaca_occitan" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic" / "merged_splits"

DEFAULT_STAGE3B_TRAIN = "train_synth.jsonl"
DEFAULT_STAGE3B_DEV = "dev_synth.jsonl"
DEFAULT_ALPACA_TRAIN = "train_alpaca.jsonl"
DEFAULT_ALPACA_DEV = "dev_alpaca_200.jsonl"

OUT_TRAIN = "train_merged.jsonl"
OUT_DEV = "dev_merged.jsonl"
OUT_META = "merge_metadata.json"

DEFAULT_SEED = 42

STAGE3B_INSTRUCTION_TEMPLATES = [
  "Escriu un text natural en occitan lengadocian (norma classica).",
  "Produz una responsa en occitan lengadocian corrècta e idiomatica.",
  "Redigis un pichon passatge en occitan lengadocian, estil natural.",
]


def _read_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError:
        continue
  return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
  with open(path, "w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_alpaca_row(row: dict) -> dict | None:
  instruction = str(row.get("instruction", "")).strip()
  model_input = str(row.get("input", "")).strip()
  output = str(row.get("output", "")).strip()
  if not instruction or not output:
    return None
  return {
    "instruction": instruction,
    "input": model_input,
    "output": output,
    "source": "alpaca_occitan",
  }


def _convert_text_row(row: dict, rng: random.Random) -> dict | None:
  text = str(row.get("text", "")).strip()
  if not text:
    return None
  return {
    "instruction": rng.choice(STAGE3B_INSTRUCTION_TEMPLATES),
    "input": "",
    "output": text,
    "source": "synth",
    "source_type": str(row.get("source_type", "")).strip(),
  }


def _load_alpaca_split(path: Path) -> list[dict]:
  out: list[dict] = []
  for row in _read_jsonl(path):
    normalized = _normalize_alpaca_row(row)
    if normalized is not None:
      out.append(normalized)
  return out


def _load_split(path: Path, rng: random.Random) -> list[dict]:
  out: list[dict] = []
  for row in _read_jsonl(path):
    converted = _convert_text_row(row, rng)
    if converted is not None:
      out.append(converted)
  return out


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Merge synthetic + Alpaca splits into Alpaca-style splits."
  )
  parser.add_argument("--synth-dir", type=Path, default=DEFAULT_STAGE3B_DIR)
  parser.add_argument("--alpaca-dir", type=Path, default=DEFAULT_ALPACA_DIR)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
  args = parser.parse_args()

  train_path = args.dir / DEFAULT_STAGE3B_TRAIN
  dev_path = args.dir / DEFAULT_STAGE3B_DEV
  alpaca_train_path = args.alpaca_dir / DEFAULT_ALPACA_TRAIN
  alpaca_dev_path = args.alpaca_dir / DEFAULT_ALPACA_DEV

  for path in [train_path, dev_path, alpaca_train_path, alpaca_dev_path]:
    if not path.exists():
      raise FileNotFoundError(f"Required input split not found: {path}")

  rng = random.Random(args.seed)

  train = _load_split(train_path, rng)
  dev = _load_split(dev_path, rng)
  alpaca_train = _load_alpaca_split(alpaca_train_path)
  alpaca_dev = _load_alpaca_split(alpaca_dev_path)

  merged_train = train + alpaca_train
  merged_dev = dev + alpaca_dev
  rng.shuffle(merged_train)
  rng.shuffle(merged_dev)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  out_train = args.output_dir / OUT_TRAIN
  out_dev = args.output_dir / OUT_DEV
  out_meta = args.output_dir / OUT_META

  _write_jsonl(out_train, merged_train)
  _write_jsonl(out_dev, merged_dev)

  metadata = {
    "seed": args.seed,
    "inputs": {
      "train": str(train_path),
      "dev": str(dev_path),
      "alpaca_train": str(alpaca_train_path),
      "alpaca_dev": str(alpaca_dev_path),
    },
    "counts": {
      "train_rows": len(train),
      "dev_rows": len(dev),
      "alpaca_train_rows": len(alpaca_train),
      "alpaca_dev_rows": len(alpaca_dev),
      "merged_train_rows": len(merged_train),
      "merged_dev_rows": len(merged_dev),
    },
    "outputs": {
      "train_file": str(out_train),
      "dev_file": str(out_dev),
    },
  }
  with open(out_meta, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

  print("=" * 60)
  print("PREPARE MERGED SPLITS")
  print("=" * 60)
  print(f"Train: {len(merged_train):,} -> {out_train}")
  print(f"Dev:  {len(merged_dev):,} -> {out_dev}")
  print(f"Meta: {out_meta}")
  print("=" * 60)


if __name__ == "__main__":
  main()

