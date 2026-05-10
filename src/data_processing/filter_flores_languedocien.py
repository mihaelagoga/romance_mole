"""
Filter external Occitan evaluation data toward Languedocien/Norma Classica.

Example:
  python -m src.data_processing.filter_flores_languedocien
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DEFAULT_OUTPUT = DEFAULT_EVAL_DIR / "eval_external_languedocien_filtered.jsonl"
DEFAULT_METADATA = DEFAULT_EVAL_DIR / "eval_external_languedocien_filter_metadata.json"


REJECT_PATTERNS = {
  "gascon_articles_eth_era": re.compile(r"\b(eth|era|eths|eras)\b", flags=re.IGNORECASE),
  "aranese_contractions_deth": re.compile(r"\b(deth|dera|deths|deras)\b", flags=re.IGNORECASE),
  "provençal_contractions_dau": re.compile(r"\b(dau|daus)\b", flags=re.IGNORECASE),
  "gascon_indefinite_ua": re.compile(r"\bua\b", flags=re.IGNORECASE),
  # Gascon enunciative "Que" + clitic/aux start (e.g., "Que'm...", "Que soi...").
  "gascon_enunciative_que_initial": re.compile(
    r"^\s*['\"(\-]*que(?:\s+|')(m|t|s|n|v|u|soi|sèm|es|ei|èra|eren)\b",
    flags=re.IGNORECASE,
  ),
}


POSITIVE_PATTERNS = {
  "articles_lo_la_los_las": re.compile(r"\b(lo|la|los|las)\b", flags=re.IGNORECASE),
  "languedocien_contractions_del_pel_al": re.compile(r"\b(del|pel|al|dels|pels|als)\b", flags=re.IGNORECASE),
  "classical_diacritics": re.compile(r"[òèç]"),
  "elision_apostrophe": re.compile(r"\b([ldqsnmt])'"),
  "interrogatives_norma_classica": re.compile(r"\b(ont|cossi|perque|quora|qual)\b", flags=re.IGNORECASE),
}


def normalize_space(text: str) -> str:
  return re.sub(r"\s+", " ", text).strip()


def discover_external_files(eval_dir: Path) -> list[Path]:
  """
  Auto-discover external eval source files:
  FLORES split files
  UD file
  """
  patterns = [
    "*flores200*_oc.jsonl",
    "test_ud_occitan.jsonl",
  ]
  seen = set()
  files: list[Path] = []
  for pattern in patterns:
    for path in sorted(eval_dir.glob(pattern)):
      if path.is_file() and path not in seen:
        seen.add(path)
        files.append(path)
  return files


def score_row(text: str) -> tuple[bool, list[str], list[str]]:
  reject_hits = [name for name, rx in REJECT_PATTERNS.items() if rx.search(text)]
  if reject_hits:
    return False, reject_hits, []

  positive_hits = [name for name, rx in POSITIVE_PATTERNS.items() if rx.search(text)]
  return True, [], positive_hits


def read_jsonl(path: Path) -> list[dict]:
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
      text = normalize_space(str(item.get("text", "")))
      if not text:
        continue
      item["text"] = text
      rows.append(item)
  return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
  with open(path, "w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
  parser = argparse.ArgumentParser(
    description="Filter external eval rows (FLORES + UD) to Languedocien-like subset."
  )
  parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR, help="Directory containing eval files.")
  parser.add_argument(
    "--input-file",
    action="append",
    default=[],
    help="Optional eval file(s). If omitted, auto-discovers FLORES and UD eval files.",
  )
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Filtered output JSONL file.")
  parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="Metadata JSON output path.")
  parser.add_argument(
    "--min-positive-signals",
    type=int,
    default=1,
    help="Minimum number of positive markers required to keep a sentence.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.min_positive_signals < 0:
    raise ValueError("--min-positive-signals must be >= 0")

  input_files = [Path(p) for p in args.input_file] if args.input_file else discover_external_files(args.eval_dir)
  if not input_files:
    raise FileNotFoundError(
      f"No external eval files found. Use --input-file or run build_external_occitan_eval.py first. "
      f"Searched: {args.eval_dir}"
    )
  for path in input_files:
    if not path.exists():
      raise FileNotFoundError(f"Input file not found: {path}")

  print("=" * 70)
  print("FILTER external EVAL (FLORES + UD) -> LANGUEDOCIEN SUBSET")
  print("=" * 70)
  print("Input files:")
  for p in input_files:
    print(f" {p}")
  print(f"Min positive signals: {args.min_positive_signals}")
  print(f"Output: {args.output}")
  print("=" * 70)

  kept_rows: list[dict] = []
  per_file_counts = {}
  reject_reason_counts: Counter[str] = Counter()
  positive_signal_counts: Counter[str] = Counter()

  for file_path in input_files:
    rows = read_jsonl(file_path)
    file_total = len(rows)
    file_kept = 0

    for row in rows:
      text = row["text"]
      pass_reject_check, reject_hits, positive_hits = score_row(text)

      if not pass_reject_check:
        for reason in reject_hits:
          reject_reason_counts[reason] += 1
        continue

      if len(positive_hits) < args.min_positive_signals:
        reject_reason_counts["insufficient_positive_signals"] += 1
        continue

      for sig in positive_hits:
        positive_signal_counts[sig] += 1

      kept = dict(row)
      kept["source_file"] = str(file_path.name)
      kept["filter_positive_signals"] = positive_hits
      kept["source"] = "external_languedocien_filtered"
      kept_rows.append(kept)
      file_kept += 1

    per_file_counts[file_path.name] = {
      "input_rows": file_total,
      "kept_rows": file_kept,
      "dropped_rows": file_total file_kept,
    }
    print(f"{file_path.name}: kept {file_kept:,} / {file_total:,}")

  args.output.parent.mkdir(parents=True, exist_ok=True)
  write_jsonl(args.output, kept_rows)

  metadata = {
    "input_files": [str(p) for p in input_files],
    "output_file": str(args.output),
    "min_positive_signals": args.min_positive_signals,
    "totals": {
      "input_rows": sum(v["input_rows"] for v in per_file_counts.values()),
      "kept_rows": len(kept_rows),
      "dropped_rows": sum(v["dropped_rows"] for v in per_file_counts.values()),
    },
    "per_file_counts": per_file_counts,
    "reject_reason_counts": dict(reject_reason_counts),
    "positive_signal_counts": dict(positive_signal_counts),
  }
  with open(args.metadata, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

  print("Done.")
  print(f" Filtered rows: {len(kept_rows):,}")
  print(f" Output:    {args.output}")
  print(f" Metadata:   {args.metadata}")


if __name__ == "__main__":
  main()
