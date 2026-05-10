"""
Fetch aligned FLORES French, Catalan, and Occitan evaluation files.

Example:
    python -m src.data_processing.fetch_flores_eval_data --output-dir flores_eval_data
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "flores_eval_data"
EXPECTED_DEVTEST_COUNT = 1012

LANGUAGES = ("fra_Latn", "oci_Latn", "cat_Latn")
DGME_CONFIG_MAP = {
    "fra_Latn": "flores_fr",
    "oci_Latn": "flores_oc",
    "cat_Latn": "flores_ca",
}
DGME_SPLIT_MAP = {
    # DGME mirror often exposes FLORES devtest as "test".
    "dev": "dev",
    "devtest": "test",
    "test": "test",
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _require_non_empty(value: str, field_name: str, row_idx: int) -> str:
    text = normalize_space(value)
    if not text:
        raise ValueError(f"Empty {field_name} at row {row_idx}.")
    return text


def load_from_facebook_all(split: str) -> list[dict]:
    ds = load_dataset(
        "facebook/flores",
        "all",
        split=split,
        trust_remote_code=False,
    )

    rows: list[dict] = []
    for idx, row in enumerate(ds, start=1):
        rows.append(
            {
                "id": int(row.get("id", idx)),
                "fra_Latn": _require_non_empty(row.get("sentence_fra_Latn", ""), "sentence_fra_Latn", idx),
                "oci_Latn": _require_non_empty(row.get("sentence_oci_Latn", ""), "sentence_oci_Latn", idx),
                "cat_Latn": _require_non_empty(row.get("sentence_cat_Latn", ""), "sentence_cat_Latn", idx),
            }
        )

    return rows


def load_from_facebook_individual(split: str) -> list[dict]:
    per_lang_rows: dict[str, list[dict]] = {}

    for lang in LANGUAGES:
        ds = load_dataset(
            "facebook/flores",
            lang,
            split=split,
            trust_remote_code=False,
        )
        rows = []
        for idx, row in enumerate(ds, start=1):
            rows.append(
                {
                    "id": int(row.get("id", idx)),
                    "sentence": _require_non_empty(row.get("sentence", ""), f"{lang}.sentence", idx),
                }
            )
        per_lang_rows[lang] = rows

    counts = {lang: len(rows) for lang, rows in per_lang_rows.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Language-specific facebook/flores counts do not align: {counts}")

    aligned: list[dict] = []
    total = counts["fra_Latn"]
    for idx in range(total):
        fra = per_lang_rows["fra_Latn"][idx]
        oci = per_lang_rows["oci_Latn"][idx]
        cat = per_lang_rows["cat_Latn"][idx]

        ids = {fra["id"], oci["id"], cat["id"]}
        if len(ids) != 1:
            raise ValueError(f"Mismatched ids across languages at index {idx}: {ids}")

        aligned.append(
            {
                "id": fra["id"],
                "fra_Latn": fra["sentence"],
                "oci_Latn": oci["sentence"],
                "cat_Latn": cat["sentence"],
            }
        )

    return aligned


def load_from_dgme_mirror(split: str) -> list[dict]:
    dgme_split = DGME_SPLIT_MAP.get(split, split)
    per_lang_texts: dict[str, list[str]] = {}

    for lang in LANGUAGES:
        ds = load_dataset(
            "DGME/FLORES-200",
            DGME_CONFIG_MAP[lang],
            split=dgme_split,
            trust_remote_code=False,
        )
        texts = [
            _require_non_empty(row.get("text", ""), f"{lang}.text", idx)
            for idx, row in enumerate(ds, start=1)
        ]
        per_lang_texts[lang] = texts

    counts = {lang: len(texts) for lang, texts in per_lang_texts.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"DGME/FLORES-200 counts do not align: {counts}")

    total = counts["fra_Latn"]
    return [
        {
            "id": idx,
            "fra_Latn": per_lang_texts["fra_Latn"][idx - 1],
            "oci_Latn": per_lang_texts["oci_Latn"][idx - 1],
            "cat_Latn": per_lang_texts["cat_Latn"][idx - 1],
        }
        for idx in range(1, total + 1)
    ]


def load_aligned_flores(split: str) -> tuple[list[dict], str]:
    attempts: list[tuple[str, callable]] = [
        ("facebook/flores [all config]", load_from_facebook_all),
        ("facebook/flores [language configs]", load_from_facebook_individual),
        ("DGME/FLORES-200 mirror (devtest->test mapping)", load_from_dgme_mirror),
    ]

    errors: list[str] = []
    for label, loader in attempts:
        try:
            rows = loader(split)
            if rows:
                return rows, label
            errors.append(f"{label}: returned 0 rows")
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    raise RuntimeError(
        "Could not load aligned FLORES data from any known source. "
        f"Recent errors: {errors}"
    )


def write_txt(path: Path, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "id": row["id"],
                "fra_Latn": row["fra_Latn"],
                "oci_Latn": row["oci_Latn"],
                "cat_Latn": row["cat_Latn"],
                "french_source": row["fra_Latn"],
                "occitan_target": row["oci_Latn"],
                "catalan_target": row["cat_Latn"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def verify_counts(output_dir: Path) -> None:
    files = [
        output_dir / "fra_Latn.txt",
        output_dir / "oci_Latn.txt",
        output_dir / "cat_Latn.txt",
        output_dir / "flores_aligned.jsonl",
    ]

    counts = {path.name: count_lines(path) for path in files}
    if any(count != EXPECTED_DEVTEST_COUNT for count in counts.values()):
        raise ValueError(
            "Verification failed. Expected 1012 lines in every generated file, "
            f"got: {counts}"
        )

    print(
        "Verification passed: all generated files contain exactly "
        f"{EXPECTED_DEVTEST_COUNT} aligned devtest lines."
    )
    for name, count in counts.items():
        print(f"  {name}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch FLORES-200 devtest data for fra_Latn, oci_Latn, and cat_Latn."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for FLORES files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="devtest",
        choices=["dev", "devtest"],
        help="FLORES split to fetch (default: devtest).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"FETCH FLORES ({args.split})")
    print("=" * 60)
    rows, source_label = load_aligned_flores(split=args.split)
    print(f"Loaded {len(rows)} aligned rows from {source_label}.")

    fra_lines = [row["fra_Latn"] for row in rows]
    oci_lines = [row["oci_Latn"] for row in rows]
    cat_lines = [row["cat_Latn"] for row in rows]

    write_txt(output_dir / "fra_Latn.txt", fra_lines)
    write_txt(output_dir / "oci_Latn.txt", oci_lines)
    write_txt(output_dir / "cat_Latn.txt", cat_lines)
    write_jsonl(output_dir / "flores_aligned.jsonl", rows)

    if args.split == "devtest" and len(rows) != EXPECTED_DEVTEST_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DEVTEST_COUNT} devtest rows, found {len(rows)}."
        )

    verify_counts(output_dir)
    print(f"Saved files to: {output_dir}")


if __name__ == "__main__":
    main()
