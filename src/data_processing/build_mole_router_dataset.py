"""
Build a mixed language-labeled dataset for Romance-MoLE router training.

Example:
    python -m src.data_processing.build_mole_router_dataset --output_dir data/router_training
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEGMENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+|(?<=[;:])\s+")


def clean_input(val):
    if val is None:
        return ""
    if isinstance(val, float) and val != val:  # NaN
        return ""
    val_str = str(val).strip()
    return "" if val_str.lower() == "nan" else val_str


def clean_text(val):
    return clean_input(val)


def iter_json_objects_from_line(line: str):
    """
    Parse one or more JSON objects from a single line.

    Some generated JSONL files can accidentally contain multiple objects on one line.
    """
    decoder = json.JSONDecoder()
    idx = 0
    n = len(line)

    while idx < n:
        while idx < n and line[idx].isspace():
            idx += 1
        if idx >= n:
            break

        obj, end = decoder.raw_decode(line, idx)
        yield obj
        idx = end


def _make_stats() -> dict[str, int]:
    return {
        "loaded": 0,
        "kept": 0,
        "skipped_invalid": 0,
        "filtered_unk": 0,
        "duplicates_removed": 0,
        "codeswitched_created": 0,
    }


def _row_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("lang", "")),
        str(row.get("instruction", "")),
        str(row.get("input", "")),
        str(row.get("output", "")),
    )


def _contains_unk(row: dict) -> bool:
    fields = (row.get("instruction", ""), row.get("input", ""), row.get("output", ""))
    return any("<unk>" in str(value) for value in fields)


def _normalize_row(row: dict, lang: str) -> dict:
    return {
        "instruction": clean_text(row.get("instruction")),
        "input": clean_input(row.get("input")),
        "output": clean_text(row.get("output")),
        "lang": lang,
    }


def _keep_row(
    row: dict,
    *,
    lang: str,
    stats: dict[str, int],
    seen_keys: set[tuple[str, str, str, str]],
    allow_unk: bool,
) -> dict | None:
    stats["loaded"] += 1
    normalized = _normalize_row(row, lang)
    if not normalized["instruction"] or not normalized["output"]:
        stats["skipped_invalid"] += 1
        return None
    if not allow_unk and _contains_unk(normalized):
        stats["filtered_unk"] += 1
        return None

    key = _row_key(normalized)
    if key in seen_keys:
        stats["duplicates_removed"] += 1
        return None

    seen_keys.add(key)
    stats["kept"] += 1
    return normalized


def _load_occitan_rows(oc_path: Path, *, allow_unk: bool) -> tuple[list[dict], dict[str, int]]:
    print(f"Loading Occitan data from {oc_path}...")
    rows = []
    stats = _make_stats()
    seen_keys: set[tuple[str, str, str, str]] = set()

    with open(oc_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                for item in iter_json_objects_from_line(line):
                    kept = _keep_row(
                        item,
                        lang="oc",
                        stats=stats,
                        seen_keys=seen_keys,
                        allow_unk=allow_unk,
                    )
                    if kept is not None:
                        rows.append(kept)
            except json.JSONDecodeError:
                stats["skipped_invalid"] += 1
                print(f"Warning: Skipping malformed Occitan JSON at line {line_no}.")

    return rows, stats


def _load_hf_rows(
    dataset_name: str,
    *,
    lang: str,
    target_count: int,
    seed: int,
    allow_unk: bool,
) -> tuple[list[dict], dict[str, int]]:
    print(f"\nLoading {lang.upper()} data from '{dataset_name}'...")
    dataset = load_dataset(dataset_name, split="train").shuffle(seed=seed)
    rows = []
    stats = _make_stats()
    seen_keys: set[tuple[str, str, str, str]] = set()

    for item in dataset:
        kept = _keep_row(
            item,
            lang=lang,
            stats=stats,
            seen_keys=seen_keys,
            allow_unk=allow_unk,
        )
        if kept is not None:
            rows.append(kept)
        if len(rows) >= target_count:
            break

    return rows, stats


def _print_quality_report(label: str, stats: dict[str, int]) -> None:
    print(f"{label} quality report:")
    print(f"  - Loaded rows:        {stats['loaded']}")
    print(f"  - Kept rows:          {stats['kept']}")
    print(f"  - Invalid skipped:    {stats['skipped_invalid']}")
    print(f"  - <unk> filtered:     {stats['filtered_unk']}")
    print(f"  - Duplicates removed: {stats['duplicates_removed']}")
    if stats["codeswitched_created"] > 0:
        print(f"  - Code-switched made: {stats['codeswitched_created']}")


def _split_segments(text: str) -> list[str]:
    chunks = [segment.strip() for segment in SEGMENT_SPLIT_RE.split(text) if segment.strip()]
    if len(chunks) >= 2:
        return chunks
    fallback = [segment.strip() for segment in re.split(r"\s*,\s*", text) if segment.strip()]
    return fallback


def _build_codeswitched_output(anchor_output: str, donor_output: str, rng: random.Random) -> tuple[str, int] | tuple[None, None]:
    anchor_segments = _split_segments(anchor_output)
    donor_segments = _split_segments(donor_output)
    if len(anchor_segments) < 2 or not donor_segments:
        return None, None

    insert_at = rng.randrange(1, len(anchor_segments))
    donor_segment = rng.choice(donor_segments)
    mixed_segments = anchor_segments[:insert_at] + [donor_segment] + anchor_segments[insert_at:]
    return " ".join(mixed_segments), insert_at


def _generate_codeswitched_rows(
    rows_by_lang: dict[str, list[dict]],
    *,
    frac: float,
    seed: int,
    stats_by_lang: dict[str, dict[str, int]],
) -> list[dict]:
    if frac <= 0.0:
        return []

    total_base_rows = sum(len(rows) for rows in rows_by_lang.values())
    target_rows = int(round(total_base_rows * frac))
    if target_rows <= 0:
        return []

    rng = random.Random(seed)
    generated = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    lang_pairs = [
        ("fr", "oc"),
        ("fr", "ca"),
        ("ca", "oc"),
        ("ca", "fr"),
        ("oc", "fr"),
        ("oc", "ca"),
    ]
    pair_counts: Counter[str] = Counter()
    max_attempts = max(target_rows * 30, 100)

    for _ in range(max_attempts):
        if len(generated) >= target_rows:
            break
        anchor_lang, donor_lang = rng.choice(lang_pairs)
        anchor_rows = rows_by_lang.get(anchor_lang, [])
        donor_rows = rows_by_lang.get(donor_lang, [])
        if not anchor_rows or not donor_rows:
            continue

        anchor_row = rng.choice(anchor_rows)
        donor_row = rng.choice(donor_rows)
        mixed_output, insert_at = _build_codeswitched_output(
            anchor_row["output"],
            donor_row["output"],
            rng,
        )
        if mixed_output is None:
            continue

        mixed_row = {
            "instruction": anchor_row["instruction"],
            "input": anchor_row["input"],
            "output": mixed_output,
            "lang": f"{anchor_lang}+{donor_lang}",
            "is_codeswitched": True,
            "codeswitch_meta": {
                "anchor_lang": anchor_lang,
                "inserted_lang": donor_lang,
                "strategy": "segment_insert",
                "insert_after_segment": insert_at,
            },
        }
        key = _row_key(mixed_row)
        if key in seen_keys:
            continue

        seen_keys.add(key)
        generated.append(mixed_row)
        pair_counts[f"{anchor_lang}+{donor_lang}"] += 1
        stats_by_lang[anchor_lang]["codeswitched_created"] += 1

    print(f"\nGenerated {len(generated)} code-switched rows (target={target_rows}).")
    if pair_counts:
        print("Code-switch pair counts:")
        for pair, count in sorted(pair_counts.items()):
            print(f"  - {pair}: {count}")
    return generated


def _lang_histogram(rows: list[dict]) -> Counter:
    return Counter(str(row.get("lang", "unknown")) for row in rows)


def _print_lang_histogram(title: str, rows: list[dict]) -> None:
    hist = _lang_histogram(rows)
    total = max(1, len(rows))
    print(title)
    for lang, count in sorted(hist.items()):
        print(f"  - {lang}: {count} rows ({count / total * 100:.1f}%)")


def _stratified_split(rows: list[dict], *, dev_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("lang", "unknown")), []).append(row)

    train_rows = []
    dev_rows = []
    for lang, bucket in buckets.items():
        bucket_copy = list(bucket)
        rng.shuffle(bucket_copy)
        if len(bucket_copy) <= 1:
            train_bucket = bucket_copy
            dev_bucket = []
        else:
            dev_size = int(round(len(bucket_copy) * dev_frac))
            dev_size = max(1, dev_size) if dev_frac > 0 else 0
            dev_size = min(dev_size, len(bucket_copy) - 1)
            dev_bucket = bucket_copy[:dev_size]
            train_bucket = bucket_copy[dev_size:]
        train_rows.extend(train_bucket)
        dev_rows.extend(dev_bucket)
        print(f"Split '{lang}': train={len(train_bucket)}, dev={len(dev_bucket)}")

    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    return train_rows, dev_rows


def main(args):
    oc_path = PROJECT_ROOT / args.oc_data
    if not oc_path.exists():
        raise FileNotFoundError(f"Could not find Occitan data at {oc_path}")

    oc_data, oc_stats = _load_occitan_rows(oc_path, allow_unk=args.allow_unk)
    num_oc = len(oc_data)
    if num_oc == 0:
        raise ValueError("No valid Occitan rows loaded; cannot build mixed dataset.")

    num_fr = int(round(num_oc * args.fr_multiplier))
    num_ca = int(round(num_oc * args.ca_multiplier))

    print(f"\nLoaded {num_oc} Occitan examples.")
    print(
        f"Targeting {num_fr} French and {num_ca} Catalan examples "
        f"(multipliers: fr={args.fr_multiplier}, ca={args.ca_multiplier})."
    )

    fr_data, fr_stats = _load_hf_rows(
        "timpearce/alpaca-cleaned-french",
        lang="fr",
        target_count=num_fr,
        seed=args.seed,
        allow_unk=args.allow_unk,
    )
    ca_data, ca_stats = _load_hf_rows(
        "saillab/alpaca-catalan-cleaned",
        lang="ca",
        target_count=num_ca,
        seed=args.seed,
        allow_unk=args.allow_unk,
    )

    if len(fr_data) < num_fr:
        print(f"Warning: Requested {num_fr} French rows, got {len(fr_data)} after filtering.")
    if len(ca_data) < num_ca:
        print(f"Warning: Requested {num_ca} Catalan rows, got {len(ca_data)} after filtering.")

    print()
    _print_quality_report("Occitan", oc_stats)
    _print_quality_report("French", fr_stats)
    _print_quality_report("Catalan", ca_stats)

    rows_by_lang = {"oc": oc_data, "fr": fr_data, "ca": ca_data}
    stats_by_lang = {"oc": oc_stats, "fr": fr_stats, "ca": ca_stats}

    print("\nMerging datasets...")
    merged_data = oc_data + fr_data + ca_data
    if not merged_data:
        raise ValueError("Merged dataset is empty.")

    codeswitched_rows = _generate_codeswitched_rows(
        rows_by_lang,
        frac=args.codeswitch_frac,
        seed=args.seed,
        stats_by_lang=stats_by_lang,
    )
    merged_data.extend(codeswitched_rows)

    print()
    _print_quality_report("Occitan", oc_stats)
    _print_quality_report("French", fr_stats)
    _print_quality_report("Catalan", ca_stats)
    _print_lang_histogram("Merged language histogram:", merged_data)

    print(f"\nPerforming stratified train/dev split (seed={args.seed}, dev={args.dev_frac:.0%})...")
    train_data, dev_data = _stratified_split(merged_data, dev_frac=args.dev_frac, seed=args.seed)

    print(f"\nTotal merged dataset size: {len(merged_data)} rows")
    print(f"  - Train: {len(train_data)} rows")
    print(f"  - Dev:   {len(dev_data)} rows")
    _print_lang_histogram("Train language histogram:", train_data)
    _print_lang_histogram("Dev language histogram:", dev_data)

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = out_dir / "mole_router_train.jsonl"
    dev_out = out_dir / "mole_router_dev.jsonl"

    with open(train_out, "w", encoding="utf-8") as handle:
        for row in train_data:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(dev_out, "w", encoding="utf-8") as handle:
        for row in dev_data:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved training data to: {train_out.relative_to(PROJECT_ROOT)}")
    print(f"Saved validation data to: {dev_out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare mixed routing dataset for Romance-MoLE")
    parser.add_argument(
        "--oc_data",
        default="data/synthetic/alpaca_occitan_v2/alpaca_occitan_v2.jsonl",
        help="Path to Occitan Alpaca data",
    )
    parser.add_argument("--output_dir", default="data/mole_router_data", help="Output directory")
    parser.add_argument(
        "--dev_frac",
        type=float,
        default=0.10,
        help="Fraction of each language bucket used for validation (default: 0.10 = 10%%)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle and train/dev split")
    parser.add_argument(
        "--fr_multiplier",
        type=float,
        default=2.0,
        help="French rows to sample relative to the number of kept Occitan rows (default: 2.0).",
    )
    parser.add_argument(
        "--ca_multiplier",
        type=float,
        default=1.0,
        help="Catalan rows to sample relative to the number of kept Occitan rows (default: 1.0).",
    )
    parser.add_argument(
        "--codeswitch_frac",
        type=float,
        default=0.0,
        help="Fraction of additional synthetic code-switched rows to add relative to the monolingual base set.",
    )
    parser.add_argument(
        "--allow_unk",
        action="store_true",
        help="Keep rows containing literal '<unk>' tokens instead of filtering them out.",
    )
    args = parser.parse_args()
    main(args)
