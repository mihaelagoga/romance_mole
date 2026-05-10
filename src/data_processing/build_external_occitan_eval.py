"""
Build external Occitan evaluation files from FLORES and UD sources.

Example:
  python -m src.data_processing.build_external_occitan_eval --output-dir data/eval
"""

import argparse
import json
import re
from urllib.request import urlopen
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"


def normalize_space(text: str) -> str:
  return re.sub(r"\s+", " ", text).strip()


def write_text_jsonl(path: Path, texts: list[str], source: str) -> None:
  with open(path, "w", encoding="utf-8") as f:
    for t in texts:
      f.write(json.dumps({"text": t, "source": source}, ensure_ascii=False) + "\n")


def dedupe_preserve_order(texts: list[str]) -> list[str]:
  seen = set()
  out = []
  for t in texts:
    key = normalize_space(t).lower()
    if not key or key in seen:
      continue
    seen.add(key)
    out.append(normalize_space(t))
  return out


def _load_flores_occitan_split(split_candidates: list[str]) -> tuple[list[str], str]:
  """
  Try each split name in order and return (texts, split_name) for the first
  that loads successfully. Raises RuntimeError if all fail.
  """
  errors: list[str] = []
  for split_name in split_candidates:
    try:
      ds = load_dataset(
        "DGME/FLORES-200",
        "flores_oc",
        split=split_name,
        trust_remote_code=False,
      )
      texts = [normalize_space(str(row.get("text", ""))) for row in ds]
      texts = [t for t in texts if t]
      texts = dedupe_preserve_order(texts)
      if texts:
        return texts, split_name
    except Exception as e:
      errors.append(f"{split_name}: {e}")
      continue
  raise RuntimeError(
    f"Could not load FLORES Occitan (tried {split_candidates}). "
    f"Errors: {errors}"
  )


def load_flores_occitan_dev() -> tuple[list[str], str]:
  """
  Load FLORES dev split for training-time monitoring.
  Priority: 'dev' → 'devtest' (never 'test', to keep test clean).
  """
  return _load_flores_occitan_split(["dev", "devtest"])


def load_flores_occitan_test() -> tuple[list[str], str]:
  """
  Load FLORES test split for final evaluation.
  Priority: 'devtest' → 'test' (never 'dev', which is used for monitoring).
  """
  return _load_flores_occitan_split(["devtest", "test"])


def _extract_ud_sentence(row: dict) -> str:
  if isinstance(row.get("text"), str) and row["text"].strip():
    return normalize_space(row["text"])
  if isinstance(row.get("sentence"), str) and row["sentence"].strip():
    return normalize_space(row["sentence"])
  tokens = row.get("tokens")
  if isinstance(tokens, list) and tokens:
    return normalize_space(" ".join(str(x) for x in tokens))
  return ""


def _parse_conllu_sentences(conllu_text: str) -> list[str]:
  """
  Extract sentence texts from CoNLL-U content.
  Prefers '# text = ...' comments; falls back to token reconstruction.
  """
  sentences: list[str] = []
  current_tokens: list[str] = []
  current_text = ""

  for raw_line in conllu_text.splitlines():
    line = raw_line.strip()
    if not line:
      if current_text:
        sentences.append(normalize_space(current_text))
      elif current_tokens:
        sentences.append(normalize_space(" ".join(current_tokens)))
      current_tokens = []
      current_text = ""
      continue

    if line.startswith("#"):
      if line.startswith("# text ="):
        current_text = line.split("=", 1)[1].strip()
      continue

    parts = line.split("\t")
    if len(parts) < 2:
      continue
    token_id = parts[0]
    form = parts[1]

    if "-" in token_id or "." in token_id:
      continue
    if form and form != "_":
      current_tokens.append(form)

  # flush trailing sentence
  if current_text:
    sentences.append(normalize_space(current_text))
  elif current_tokens:
    sentences.append(normalize_space(" ".join(current_tokens)))

  return [s for s in sentences if s]


def _download_text(url: str) -> str:
  with urlopen(url, timeout=30) as resp:
    return resp.read().decode("utf-8", errors="replace")


def load_ud_occitan_from_github(max_rows: int | None = None) -> list[str]:
  """
  Fallback for environments where HF datasets script-based UD loaders are disabled.
  """
  candidate_bases = [
    ("UD_Occitan-CorAG", "oc_corag"),
    ("UD_Occitan-TTB", "oc_ttb"),
    ("UD_Occitan-BNE", "oc_bne"),
    ("UD_Occitan-ProvenalRovenc", "oc_provenalrovenc"),
  ]
  splits = ["train", "dev", "test"]

  collected: list[str] = []
  for repo_name, prefix in candidate_bases:
    for split in splits:
      url = (
        "https://raw.githubusercontent.com/UniversalDependencies/"
        f"{repo_name}/master/{prefix}-ud-{split}.conllu"
      )
      try:
        text = _download_text(url)
      except Exception:
        continue
      rows = _parse_conllu_sentences(text)
      if rows:
        collected.extend(rows)
        if max_rows is not None and len(collected) >= max_rows:
          return dedupe_preserve_order(collected)[:max_rows]

  deduped = dedupe_preserve_order(collected)
  if max_rows is not None:
    return deduped[:max_rows]
  return deduped


def load_ud_occitan(max_rows: int | None = None) -> list[str]:
  dataset_candidates = [
    ("universal-dependencies/universal_dependencies", "oc_corag"),
    ("universal-dependencies/universal_dependencies", "oc_borest"),
    ("universal_dependencies", "oc_corag"),
    ("universal_dependencies", "oc_borest"),
  ]
  split_candidates = ["train", "validation", "test"]

  errors: list[str] = []
  for dataset_name, config_name in dataset_candidates:
    try:
      collected = []
      for split_name in split_candidates:
        try:
          ds = load_dataset(
            dataset_name,
            config_name,
            split=split_name,
            trust_remote_code=False,
          )
        except Exception as e:
          errors.append(f"{dataset_name}/{config_name}/{split_name}: {e}")
          continue
        for row in ds:
          sent = _extract_ud_sentence(row)
          if sent:
            collected.append(sent)
            if max_rows is not None and len(collected) >= max_rows:
              return dedupe_preserve_order(collected)
      if collected:
        return dedupe_preserve_order(collected)[:max_rows]
    except Exception as e:
      errors.append(f"{dataset_name}/{config_name}: {e}")
      continue

  # Fallback: fetch official UD files directly from GitHub.
  github_rows = load_ud_occitan_from_github(max_rows=max_rows)
  if github_rows:
    return github_rows

  details = errors[-5:] if errors else ["No candidate split/config could be loaded."]
  raise RuntimeError(
    "Could not load UD Occitan data from known configs or GitHub fallback. "
    f"Recent errors: {details}"
  )


def _extract_tatoeba_occitan(row: dict) -> str:
  if isinstance(row.get("translation"), dict):
    occ = row["translation"].get("oc")
    if isinstance(occ, str) and occ.strip():
      return normalize_space(occ)
  if isinstance(row.get("sourceString"), str) and row["sourceString"].strip():
    return normalize_space(row["sourceString"])
  if isinstance(row.get("source_sentence"), str) and row["source_sentence"].strip():
    return normalize_space(row["source_sentence"])
  if isinstance(row.get("sourceSentence"), str) and row["sourceSentence"].strip():
    return normalize_space(row["sourceSentence"])
  return ""


def load_tatoeba_occitan(max_rows: int | None = None) -> list[str]:
  dataset_candidates = [
    ("Helsinki-NLP/tatoeba", {"lang1": "oc", "lang2": "en"}),
    ("tatoeba", {"lang1": "oc", "lang2": "en"}),
  ]
  split_candidates = ["train", "validation", "test"]

  errors: list[str] = []
  for dataset_name, kwargs in dataset_candidates:
    try:
      collected = []
      for split_name in split_candidates:
        try:
          ds = load_dataset(
            dataset_name,
            split=split_name,
            trust_remote_code=False,
            **kwargs,
          )
        except Exception as e:
          errors.append(f"{dataset_name}/{split_name}: {e}")
          continue
        for row in ds:
          sent = _extract_tatoeba_occitan(row)
          if sent:
            collected.append(sent)
            if max_rows is not None and len(collected) >= max_rows:
              return dedupe_preserve_order(collected)
      if collected:
        return dedupe_preserve_order(collected)[:max_rows]
    except Exception as e:
      errors.append(f"{dataset_name}: {e}")
      continue

  details = errors[-5:] if errors else ["No candidate split/dataset could be loaded."]
  raise RuntimeError(
    "Could not load Tatoeba Occitan data from known dataset IDs. "
    f"Recent errors: {details}"
  )


def load_occitan_wikipedia(max_rows: int | None = None) -> list[str]:
  """
  Fallback external corpus when Tatoeba is unavailable in the local HF setup.
  """
  ds = load_dataset(
    "wikimedia/wikipedia",
    "20231101.oc",
    split="train",
    trust_remote_code=False,
  )
  rows: list[str] = []
  for row in ds:
    text = normalize_space(str(row.get("text", "")))
    if len(text) < 80:
      continue
    rows.append(text)
    if max_rows is not None and len(rows) >= max_rows:
      break
  return dedupe_preserve_order(rows)


def main() -> None:
  parser = argparse.ArgumentParser(description="Prepare external dev/test eval sets for Stage 3b.")
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--ud-max", type=int, default=800, help="Max UD Occitan sentences.")
  args = parser.parse_args()

  output_dir = args.output_dir
  output_dir.mkdir(parents=True, exist_ok=True)

  print("=" * 60)
  print("PREPARE EXTERNAL EVAL SETS ")
  print("=" * 60)

  # FLORES 'dev' split: used during 3b training for perplexity monitoring.
  print("Loading FLORES-200 Occitan dev split (for training-time monitoring)...")
  flores_dev, flores_dev_split = load_flores_occitan_dev()
  flores_dev_path = output_dir / f"dev_flores200_{flores_dev_split}_oc.jsonl"
  write_text_jsonl(flores_dev_path, flores_dev, source=f"flores200_{flores_dev_split}_oc")
  print(f" Split used: {flores_dev_split} | rows: {len(flores_dev):,}")

  # Test source 1: UD Occitan (human-annotated treebank sentences).
  print("Loading UD Occitan for test...")
  ud_rows = load_ud_occitan(max_rows=args.ud_max)
  ud_path = output_dir / "test_ud_occitan.jsonl"
  write_text_jsonl(ud_path, ud_rows, source="ud_occitan")
  print(f" UD test rows: {len(ud_rows):,}")

  # Test source 2: FLORES 'test'/'devtest' split — kept fully held-out.
  flores_test_rows: list[str] = []
  flores_test_split = ""
  flores_test_path = output_dir / "test_flores200_test_oc.jsonl"
  wikipedia_fallback_used = False

  print("Loading FLORES-200 Occitan test split (held-out, for final evaluation)...")
  try:
    flores_test_rows, flores_test_split = load_flores_occitan_test()
    write_text_jsonl(
      flores_test_path, flores_test_rows,
      source=f"flores200_{flores_test_split}_oc",
    )
    print(f" Split used: {flores_test_split} | rows: {len(flores_test_rows):,}")
  except Exception as e:
    print(f" WARNING: FLORES test loading failed ({e}).")
    print(" Falling back to Occitan Wikipedia (last resort)...")
    flores_test_rows = load_occitan_wikipedia(max_rows=args.ud_max)
    flores_test_path = output_dir / "test_wikipedia_occitan_fallback.jsonl"
    write_text_jsonl(flores_test_path, flores_test_rows, source="wikipedia_20231101_oc")
    wikipedia_fallback_used = True
    print(f" Wikipedia fallback rows: {len(flores_test_rows):,}")

  if not flores_test_rows:
    raise RuntimeError("No rows collected for FLORES test / Wikipedia fallback.")

  # Combined test: UD + FLORES test (or UD + Wikipedia if fallback triggered).
  combined_test = dedupe_preserve_order(ud_rows + flores_test_rows)
  combined_test_path = output_dir / "test_external_occitan_combined.jsonl"
  write_text_jsonl(combined_test_path, combined_test, source="combined_ud_flores_test")

  meta = {
    "dev_file": str(flores_dev_path),
    "flores_dev_split_used": flores_dev_split,
    "test_ud_file": str(ud_path),
    "test_flores_file": str(flores_test_path),
    "flores_test_split_used": flores_test_split if not wikipedia_fallback_used else "wikipedia_fallback",
    "test_combined_file": str(combined_test_path),
    "wikipedia_fallback_used": wikipedia_fallback_used,
    "counts": {
      f"dev_flores200_{flores_dev_split}_oc": len(flores_dev),
      "test_ud_occitan": len(ud_rows),
      "test_flores_or_fallback": len(flores_test_rows),
      "test_combined": len(combined_test),
    },
    "params": {"ud_max": args.ud_max},
  }
  meta_path = output_dir / "external_eval_metadata.json"
  with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

  print("Done.")
  print(f" Dev:       {flores_dev_path}")
  print(f" Test (UD):    {ud_path}")
  test_label = "FLORES test" if not wikipedia_fallback_used else "Wikipedia fallback"
  print(f" Test ({test_label}): {flores_test_path}")
  print(f" Test combined:  {combined_test_path}")
  print(f" Metadata:     {meta_path}")


if __name__ == "__main__":
  main()
