"""
Generate a hand-curatable Occitan minimal-pair challenge set with Gemini.

Example:
    GEMINI_API_KEY=... python -m src.data_processing.generate_occitan_minimal_pairs --items-per-category 10 --output data/eval/occitan_minimal_pairs_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from tqdm import tqdm


DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "eval"
    / "occitan_minimal_pairs_raw.jsonl"
)
DEFAULT_REJECTED_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "eval"
    / "occitan_minimal_pairs_rejected.jsonl"
)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
DEFAULT_BATCH_SIZE = 5
DEFAULT_ITEMS_PER_CATEGORY = 10


SYSTEM_PROMPT = """You are a computational linguist building a diagnostic benchmark for Languedocien Occitan.

Your task is to create MINIMAL PAIRS: one correct Occitan sentence and one incorrect Occitan sentence that differ only in the targeted linguistic phenomenon.

Critical requirements:
1. Output ONLY valid JSON. No Markdown, no explanations outside JSON.
2. Use Classical Norm / Norma Classica Languedocien Occitan.
3. The incorrect sentence must be plausible but linguistically wrong for the specified phenomenon.
4. The correct and incorrect Occitan sentences should be as similar as possible.
5. Use straight apostrophes (') only, never curly apostrophes.
6. Do not include offensive, political, sexual, or personally identifying content.
7. Avoid named entities unless needed; prefer everyday sentences.
8. The French source should express the same meaning as the correct Occitan sentence.
"""


@dataclass(frozen=True)
class CategorySpec:
    name: str
    phenomenon: str
    prompt: str
    required_correct_any: tuple[str, ...] = ()
    required_incorrect_any: tuple[str, ...] = ()
    forbidden_correct_any: tuple[str, ...] = ()
    min_correct_apostrophes: int = 0


CATEGORIES: list[CategorySpec] = [
    CategorySpec(
        name="elision",
        phenomenon="apostrophe elision before vowel or mute h",
        min_correct_apostrophes=1,
        required_incorrect_any=(" se ", " de ", " que ", " lo ", " la "),
        prompt="""Generate {count} minimal-pair benchmark items for Languedocien Occitan elision.

Target phenomenon:
- Correct Occitan must use apostrophe elision before vowel where appropriate:
  s'es, d'aquí, d'ont, l'òme, l'aiga, qu'aviá, n'i, m'a, t'ai.
- Incorrect Occitan should undo the elision or use a non-elided form:
  se es, de aquí, de ont, lo òme, la aiga, que aviá.

Return a JSON array. Each item must have:
{{
  "source_fr": "French sentence",
  "correct_oc": "Correct Languedocien Occitan sentence",
  "incorrect_oc": "Nearly identical but wrong Occitan sentence",
  "explanation": "One short sentence explaining the contrast"
}}""",
    ),
    CategorySpec(
        name="articles",
        phenomenon="Languedocien definite articles",
        required_correct_any=(" lo ", " la ", " los ", " las "),
        required_incorrect_any=(" el ", " els ", " eth ", " era ", " eras ", " eths "),
        forbidden_correct_any=(" el ", " els ", " eth ", " era ", " eras ", " eths "),
        prompt="""Generate {count} minimal-pair benchmark items for Languedocien Occitan definite articles.

Target phenomenon:
- Correct Occitan must use Languedocien/Classical articles: lo, la, los, las.
- Incorrect Occitan must replace exactly one or two of them with a Catalan-like or Gascon-like article:
  el, els, eth, era, eths, eras.

Examples of the contrast:
- correct: "Lo can es davant la pòrta."
- incorrect: "El can es davant la pòrta."

Return a JSON array. Each item must have source_fr, correct_oc, incorrect_oc, explanation.""",
    ),
    CategorySpec(
        name="contractions",
        phenomenon="preposition plus article contractions",
        required_correct_any=(" al ", " del ", " pel ", " als ", " dels ", " pels "),
        required_incorrect_any=(
            " a lo ",
            " de lo ",
            " per lo ",
            " a los ",
            " de los ",
            " per los ",
        ),
        prompt="""Generate {count} minimal-pair benchmark items for Occitan contractions.

Target phenomenon:
- Correct Occitan should use contracted forms: al, del, pel, als, dels, pels.
- Incorrect Occitan should incorrectly split the same contraction:
  a lo, de lo, per lo, a los, de los, per los.

Return a JSON array. Each item must have source_fr, correct_oc, incorrect_oc, explanation.""",
    ),
    CategorySpec(
        name="object_clitics",
        phenomenon="object clitic choice and placement",
        required_correct_any=(" lo ", " la ", " los ", " las ", " ne ", " n'", " i "),
        prompt="""Generate {count} minimal-pair benchmark items for Occitan object clitics.

Target phenomenon:
- Correct Occitan must use the appropriate object/adverbial clitic:
  lo, la, los, las, ne/n', i.
- Incorrect Occitan should use a wrong clitic, omit the clitic, or use a French/Catalan-like placement.
- Keep the pair as minimal as possible.

Examples of contrasts:
- correct: "Lo vesi cada matin."
- incorrect: "Vesi el cada matin."
- correct: "N'ai crompat tres."
- incorrect: "Ai crompat tres d'eles."

Return a JSON array. Each item must have source_fr, correct_oc, incorrect_oc, explanation.""",
    ),
    CategorySpec(
        name="negation",
        phenomenon="Occitan negation pattern",
        required_correct_any=(" pas", "non "),
        prompt="""Generate {count} minimal-pair benchmark items for Occitan negation.

Target phenomenon:
- Correct Occitan must use a natural Occitan negation pattern, typically with pas and/or non depending on the sentence.
- Incorrect Occitan should use a malformed French-like or Catalan-like negation while preserving the meaning.
- Avoid making the two sentences differ in meaning; only the negation form should be wrong.

Return a JSON array. Each item must have source_fr, correct_oc, incorrect_oc, explanation.""",
    ),
    CategorySpec(
        name="agreement",
        phenomenon="gender/number agreement",
        prompt="""Generate {count} minimal-pair benchmark items for Occitan gender and number agreement.

Target phenomenon:
- Correct Occitan must have correct agreement between article/noun/adjective and/or participle.
- Incorrect Occitan should change only the agreement marker.

Examples of contrasts:
- correct: "Las flors rojas son sus la taula."
- incorrect: "Las flors roge son sus la taula."
- correct: "La pòrta es dubèrta."
- incorrect: "La pòrta es dubèrt."

Return a JSON array. Each item must have source_fr, correct_oc, incorrect_oc, explanation.""",
    ),
    CategorySpec(
        name="auxiliaries_tense",
        phenomenon="auxiliary choice and tense morphology",
        required_correct_any=(" ai ", " as ", " a ", " avèm ", " son ", " es ", " èra ", " aviá "),
        prompt="""Generate {count} minimal-pair benchmark items for Occitan auxiliaries and tense morphology.

Target phenomenon:
- Correct Occitan must use appropriate aver/èsser auxiliary or tense morphology.
- Incorrect Occitan should use the wrong auxiliary, wrong person ending, or a French-like tense form.
- Keep all other words as similar as possible.

Return a JSON array. Each item must have source_fr, correct_oc, incorrect_oc, explanation.""",
    ),
    CategorySpec(
        name="languedocien_shibboleths",
        phenomenon="Languedocien forms vs Catalan/Gascon/French interference",
        required_correct_any=(" lo ", " la ", " los ", " las ", " aquò", " aquela", " aquel"),
        forbidden_correct_any=(" el ", " els ", " eth ", " era ", " eras ", " eths "),
        prompt="""Generate {count} minimal-pair benchmark items for Languedocien Occitan shibboleths.

Target phenomenon:
- Correct Occitan should sound Languedocien/Classical and use forms such as:
  lo, la, los, las, aquò, aquel/aquela, amb, dins, sus.
- Incorrect Occitan should contain one clear interference form from Catalan, Gascon, or French:
  el/els, eth/era, amb malformed as avec, dins replaced by dans, sus replaced by sur.
- Keep the semantic meaning unchanged.

Return a JSON array. Each item must have source_fr, correct_oc, incorrect_oc, explanation.""",
    ),
]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    return text


def contains_any(text: str, needles: tuple[str, ...]) -> bool:
    if not needles:
        return True
    padded = f" {text.lower()} "
    return any(needle.lower() in padded for needle in needles)


def validate_item(raw: dict[str, Any], category: CategorySpec) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []

    source_fr = normalize_space(str(raw.get("source_fr", "")))
    correct_oc = normalize_space(str(raw.get("correct_oc", ""))).replace("’", "'")
    incorrect_oc = normalize_space(str(raw.get("incorrect_oc", ""))).replace("’", "'")
    explanation = normalize_space(str(raw.get("explanation", "")))

    if not source_fr:
        errors.append("missing source_fr")
    if not correct_oc:
        errors.append("missing correct_oc")
    if not incorrect_oc:
        errors.append("missing incorrect_oc")
    if not explanation:
        errors.append("missing explanation")
    if correct_oc and incorrect_oc and correct_oc.lower() == incorrect_oc.lower():
        errors.append("correct_oc and incorrect_oc are identical")
    if correct_oc.count("'") < category.min_correct_apostrophes:
        errors.append(
            f"correct_oc has fewer than {category.min_correct_apostrophes} apostrophes"
        )
    if category.required_correct_any and not contains_any(correct_oc, category.required_correct_any):
        errors.append("correct_oc lacks required category marker")
    if category.required_incorrect_any and not contains_any(
        incorrect_oc, category.required_incorrect_any
    ):
        errors.append("incorrect_oc lacks required error marker")
    if category.forbidden_correct_any and contains_any(correct_oc, category.forbidden_correct_any):
        errors.append("correct_oc contains forbidden interference marker")

    # Reject extremely long or suspicious outputs; these are meant to be compact
    # diagnostic items, not paragraph-length generations.
    if len(correct_oc.split()) > 24:
        errors.append("correct_oc is too long")
    if len(incorrect_oc.split()) > 24:
        errors.append("incorrect_oc is too long")

    if errors:
        return None, errors

    item = {
        "category": category.name,
        "phenomenon": category.phenomenon,
        "source_lang": "fr",
        "source": source_fr,
        "correct": correct_oc,
        "incorrect": incorrect_oc,
        "explanation": explanation,
        "needs_manual_review": True,
    }
    return item, []


def request_batch(
    client: genai.Client,
    *,
    model: str,
    category: CategorySpec,
    count: int,
    temperature: float,
    max_retries: int,
) -> list[dict[str, Any]]:
    prompt = category.prompt.format(count=count)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=temperature,
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            text = strip_code_fence(response.text or "")
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
            print("Warning: Gemini response was not a JSON array")
        except json.JSONDecodeError as exc:
            print(f"JSON parse error for {category.name}: {exc}")
        except Exception as exc:
            print(f"Gemini request failed for {category.name}: {exc}")

        if attempt < max_retries - 1:
            time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))

    return []


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set and --api-key was not provided.")

    output_path = Path(args.output)
    rejected_path = Path(args.rejected_output)

    client = genai.Client(api_key=api_key)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    categories = CATEGORIES
    total_target = len(categories) * args.items_per_category
    print(f"Generating Occitan minimal-pair challenge set with {args.model}")
    print(f"Target accepted items: {total_target}")
    print(f"Categories: {', '.join(cat.name for cat in categories)}")
    print(f"Output: {output_path}")
    print(f"Rejected: {rejected_path}")
    print("=" * 72)

    with tqdm(total=total_target, desc="Accepted items") as pbar:
        for category in categories:
            accepted_for_category = 0
            attempts = 0
            max_category_attempts = args.max_category_attempts
            while accepted_for_category < args.items_per_category and attempts < max_category_attempts:
                attempts += 1
                needed = args.items_per_category - accepted_for_category
                batch_count = min(args.batch_size, max(needed * 2, args.batch_size))
                raw_items = request_batch(
                    client,
                    model=args.model,
                    category=category,
                    count=batch_count,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                )

                for raw in raw_items:
                    item, errors = validate_item(raw, category)
                    if item is None:
                        rejected.append(
                            {
                                "category": category.name,
                                "raw": raw,
                                "errors": errors,
                            }
                        )
                        continue

                    pair_key = (item["correct"].lower(), item["incorrect"].lower())
                    if pair_key in seen_pairs:
                        rejected.append(
                            {
                                "category": category.name,
                                "raw": raw,
                                "errors": ["duplicate minimal pair"],
                            }
                        )
                        continue

                    item["id"] = f"{category.name}_{accepted_for_category + 1:03d}"
                    accepted.append(item)
                    seen_pairs.add(pair_key)
                    accepted_for_category += 1
                    pbar.update(1)
                    if accepted_for_category >= args.items_per_category:
                        break

                time.sleep(args.sleep)

            if accepted_for_category < args.items_per_category:
                print(
                    f"Warning: {category.name} accepted {accepted_for_category}/"
                    f"{args.items_per_category} after {attempts} attempts."
                )

    write_jsonl(output_path, accepted)
    write_jsonl(rejected_path, rejected)

    print("\n" + "=" * 72)
    print("GENERATION COMPLETE")
    print(f"Accepted: {len(accepted)} -> {output_path}")
    print(f"Rejected: {len(rejected)} -> {rejected_path}")
    print("\nAccepted by category:")
    for category in categories:
        n = sum(1 for row in accepted if row["category"] == category.name)
        print(f"  {category.name:24s} {n:3d}/{args.items_per_category}")
    print("\nNext step: manually review accepted JSONL before using it as a benchmark.")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic Occitan minimal-pair challenge set using "
            "the Gemini API."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None, help="Defaults to GEMINI_API_KEY.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--rejected-output", default=str(DEFAULT_REJECTED_OUTPUT))
    parser.add_argument("--items-per-category", type=int, default=DEFAULT_ITEMS_PER_CATEGORY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument(
        "--max-category-attempts",
        type=int,
        default=8,
        help="Maximum Gemini batches per category before moving on.",
    )
    parser.add_argument("--sleep", type=float, default=1.5, help="Sleep between API calls.")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
