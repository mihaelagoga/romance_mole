"""
Localize Alpaca-style instruction data into Languedocien Occitan with Gemini.

Example:
    GEMINI_API_KEY=... python -m src.data_processing.generate_synthetic_alpaca_occitan --num-rows 2000
"""
import argparse
import os
import json
import re
import time
from pathlib import Path

from datasets import load_dataset
from google import genai
from google.genai import types
from tqdm import tqdm

NUM_ROWS = 2000
MODEL_NAME = "gemini-2.5-flash"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "alpaca_occitan_v2"
OUTPUT_FILE = OUTPUT_DIR / "alpaca_occitan_v2.jsonl"
BUFFER_FILE = OUTPUT_DIR / "buffer_alpaca_occitan_v2.jsonl"

MAX_RETRIES = 5
RETRY_DELAY = 2
THINKING_LEVEL_CHOICES = ("minimal", "low", "medium", "high")

_SKIP_PATTERNS = re.compile(
    r"|".join([
        r"preu\s+(actual|de\s+les\s+accions)",   # stock prices
        r"cotitza\s+a\s+la\s+borsa",              # stock exchange
        r"NASDAQ|NYSE|S&P\s*500",
        r"GDPR|CCPA|RGPD",                        # US/EU law acronyms
        r"Califòrnia|California|Texas|Florida",    # US states
        r"com\s+a\s+IA",                           # "as an AI" disclaimer (Catalan)
        r"no\s+puc\s+proporcionar\s+dades\s+en\s+temps\s+real",
        r"no\s+tinc\s+accés\s+a\s+internet",
        r"no\s+tinc\s+la\s+capacitat\s+de",
        r"https?://",                              # URLs
        r"\[(\d+)\]",                              # footnote-style refs
        r"font:\s",                                # "source:" citation markers
        r"Merriam.Webster|Wikipedia|Britannica",   # encyclopedia refs
        r"Facebook|Instagram|Twitter|TikTok|YouTube",  # social media
        r"Amazon|Google|Apple|Microsoft",          # tech company names
        r"API\s+de\s+Google|OpenAI",
    ]),
    flags=re.IGNORECASE,
)


def _should_skip_source(row: dict) -> bool:
    combined = " ".join([
        row.get("instruction", ""),
        row.get("input", ""),
        row.get("output", ""),
    ])
    return bool(_SKIP_PATTERNS.search(combined))


_REJECT_PATTERNS = re.compile(
    r"|".join([
        r"https?://",
        r"www\.",
        r"\[(\d+)\]",                             # footnote refs
        r"Facebook|Instagram|Twitter|TikTok|YouTube",
        r"Merriam.Webster|Wikipedia|Britannica",
        r"California|Texas|Florida|CCPA|GDPR|RGPD",
        r"NASDAQ|NYSE",
        r"Amazon\.com|Google\.com",
        r"com\s+a\s+IA",                          # Catalan "as an AI" leak
        r"coma\s+IA",                              # Occitan "as an AI" leak
    ]),
    flags=re.IGNORECASE,
)


def _passes_post_validation(row: dict) -> bool:
    combined = " ".join([
        row.get("instruction", ""),
        row.get("input", ""),
        row.get("output", ""),
    ])
    return not bool(_REJECT_PATTERNS.search(combined))


def _resolve_thinking_level(model_name: str, requested: str | None) -> str | None:
    if requested:
        return requested
    if model_name == "gemini-3-flash-preview":
        return "high"
    return None


def _build_generate_config(model_name: str, thinking_level: str | None) -> types.GenerateContentConfig:
    config_kwargs = {
        "system_instruction": SYSTEM_PROMPT,
        "temperature": 0.3,
    }
    effective_thinking_level = _resolve_thinking_level(model_name, thinking_level)
    if effective_thinking_level:
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=effective_thinking_level
            )
        except Exception:
            print(
                "Warning: Installed google-genai SDK does not expose ThinkingConfig. "
                "Continuing without thinking level."
            )
    try:
        return types.GenerateContentConfig(**config_kwargs)
    except Exception as e:
        # (e.g., "extra_forbidden" for thinking_config) rather than TypeError.
        err = str(e).lower()
        unsupported_thinking_config = (
            "thinking_config" in err
            and (
                "extra_forbidden" in err
                or "extra inputs are not permitted" in err
                or "unexpected keyword argument" in err
            )
        )
        unsupported_thinking_level = (
            "thinking_level" in err
            and (
                "extra_forbidden" in err
                or "extra inputs are not permitted" in err
                or "unexpected keyword argument" in err
            )
        )
        if unsupported_thinking_config or unsupported_thinking_level:
            print(
                "Warning: Installed google-genai SDK does not support thinking config. "
                "Continuing without it."
            )
            config_kwargs.pop("thinking_config", None)
            return types.GenerateContentConfig(**config_kwargs)
        raise


SYSTEM_PROMPT = """You are an expert translator and linguist specializing in Occitan,
specifically the Languedocien (Lengadocian) dialect following the Classical Norm (Norma Classica)
established by Louis Alibert.

Your task is to **localize** Catalan text into Languedocien Occitan. This is NOT a literal
word-for-word translation—you must produce natural, idiomatic Occitan that a native speaker
of the Lengadocian dialect would use. Catalan and Occitan are closely related; preserve
the meaning and adapt spelling and grammar to Norma Classica.

## STRICT GRAMMATICAL RULES (Norma Classica - Alibert):

### 1. Definite Articles:
- USE: lo (masc. sing.), la (fem. sing.), los (masc. plur.), las (fem. plur.)
- DO NOT USE Gascon articles: eth, era, eths, eras

### 2. Contractions and Prepositions:
- USE: del (de + lo), pel (per + lo), al (a + lo)
- USE: dels, pels, als for plurals
- DO NOT USE Provençal contractions: dau, daus, pòu

### 3. Elision (MANDATORY - Apply Strictly):
- Apply elision before vowels and silent 'h':
  - lo/la → l' (l'òme, l'aiga)
  - de → d' (d'aquò, d'aquel)
  - que → qu' (qu'es, qu'avèm)
  - se → s' (s'escapa)
  - ne → n' (n'i a)
  - me, te → m', t' (m'agrada, t'ai vist)

### 4. Apostrophe Style:
- USE ONLY straight apostrophes: '
- DO NOT USE curly/typographic apostrophes: ' or '

### 5. Spelling Conventions:
- Use 'ò' for open 'o' sounds
- Use 'è' for open 'e' sounds
- Use 'ç' before a, o, u for /s/ sound
- Use 'nh' for palatal nasal (like Spanish 'ñ')
- Use 'lh' for palatal lateral (like Italian 'gl' in famiglia)

### 6. Verb Forms (Languedocien conjugation):
- Follow Alibert's verb conjugation tables
- Use -èm for 1st person plural present
- Use -ètz for 2nd person plural
- Use -on/-an for 3rd person plural depending on conjugation class

## CONTENT ADAPTATION RULES (CRITICAL):

### 7. Cultural Localization:
- Replace references to non-Occitan geography, companies, or laws with
  Occitan/Southern-French equivalents when possible.
- If a concept is universally applicable (science, math, cooking), keep it
  but express it naturally in Occitan.
- Do NOT invent Occitan-specific facts. If the content cannot be meaningfully
  adapted, translate it faithfully without adding extra commentary.

### 8. Forbidden Output Patterns:
- NEVER include URLs, web links, or references to websites.
- NEVER include footnote-style references like [1], [2], etc.
- NEVER include "Works Cited", bibliographies, or source attributions.
- NEVER mention social media platforms (Facebook, Instagram, Twitter, etc.).
- NEVER add translations into other languages (Spanish, Italian, etc.).
- NEVER say "as an AI" or disclaim inability to access real-time data.
- NEVER reference the Merriam-Webster dictionary or any encyclopedia by name.
- Keep responses self-contained: the answer must stand on its own without
  pointing the reader elsewhere.

## OUTPUT REQUIREMENTS:
- You will receive a Catalan Alpaca example with three fields: instruction, input, output.
- Return a valid JSON object with exactly three keys: "instruction", "input", "output".
- Each value must be the Occitan localization of the corresponding Catalan text.
- If the original "input" is empty or "nan", return "" for "input".
- Preserve the original meaning and tone; adapt cultural references when appropriate.
- No explanations, no markdown code fences—only the raw JSON object."""


def load_alpaca_catalan(num_rows: int = 2000):
    """
    Load and pre-filter Catalan Alpaca examples.

    Returns:
        List of dicts with keys: instruction, input, output (Catalan).
    """
    print("Load Catalan Alpaca (train)")
    dataset = load_dataset(
        "saillab/alpaca-catalan-cleaned",
        split="train",
        trust_remote_code=False,
    )

    rows = []
    skipped = 0
    for ex in dataset:
        instruction = (ex.get("instruction") or "").strip()
        inp = ex.get("input") or ""
        output = (ex.get("output") or "").strip()

        if isinstance(inp, float) and (inp != inp):
            inp = ""
        if isinstance(inp, str) and inp.strip().lower() == "nan":
            inp = ""
        inp = inp.strip() if isinstance(inp, str) else ""

        if not instruction or not output:
            skipped += 1
            continue

        candidate = {"instruction": instruction, "input": inp, "output": output}
        if _should_skip_source(candidate):
            skipped += 1
            continue

        rows.append(candidate)
        if len(rows) >= num_rows:
            break

    print(f"Catalan rows: {len(rows)} (skipped {skipped})")
    return rows


def translate_row_to_occitan(
    client: genai.Client,
    instruction: str,
    input_text: str,
    output_text: str,
    model_name: str,
    thinking_level: str | None,
    max_retries: int = MAX_RETRIES,
) -> dict | None:
    """
    Localize one Alpaca example (instruction, input, output) from Catalan to Occitan.

    Returns:
        Dict with keys instruction, input, output (Occitan), or None on failure.
    """
    input_display = input_text if input_text else "(empty)"
    prompt = f"""Localize this Catalan Alpaca example into Languedocien Occitan. Return only a JSON object with keys "instruction", "input", "output".

Catalan:
- instruction: {instruction}
- input: {input_display}
- output: {output_text}

Occitan (JSON only):"""

    config = _build_generate_config(model_name=model_name, thinking_level=thinking_level)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()
            if not text:
                print("  Warning: Empty response.")
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                continue

            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)

            data = json.loads(text)
            for key in ("instruction", "input", "output"):
                if key not in data:
                    data[key] = ""
                else:
                    data[key] = str(data[key]).strip()

            if not _passes_post_validation(data):
                print("  Rejected (post-validation): hallucination markers detected.")
                return None

            return data

        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            continue
        except Exception as e:
            error_msg = str(e).lower()
            is_overloaded = (
                "429" in error_msg
                or "resource_exhausted" in error_msg
                or "overloaded" in error_msg
            )
            if is_overloaded:
                wait_time = 60 * (attempt + 1)
                print(f"  [OVERLOADED] Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                print(f"  Retry {attempt + 1}/{max_retries}: {e}")
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None

    return None


def generate_synthetic_dataset(
    max_requests: int | None = None,
    model_name: str = MODEL_NAME,
    thinking_level: str | None = None,
):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if BUFFER_FILE.exists():
        print(f"\nFound buffer: {BUFFER_FILE}. Merging...")
        try:
            with open(BUFFER_FILE, "r", encoding="utf-8") as bf, open(
                OUTPUT_FILE, "a", encoding="utf-8"
            ) as of:
                for line in bf:
                    line = line.strip()
                    if line:
                        of.write(line + "\n")
            BUFFER_FILE.unlink()
            print("Buffer merged and cleared.")
        except Exception as e:
            print(f"ERROR merging buffer: {e}")

    existing_count = 0
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_count = sum(1 for line in f if line.strip())
        print(f"Output has {existing_count} rows")

    catalog = load_alpaca_catalan(NUM_ROWS)
    if existing_count >= len(catalog):
        print(f"All {len(catalog)} eligible rows have already been localized.")
        return

    to_process = catalog[existing_count:]
    print(f"Rows remaining to localize: {len(to_process)} (indices {existing_count}–{len(catalog) - 1})")

    if max_requests is not None:
        print(f"Limit: {max_requests} requests this run (then exit for key rotation).")
    effective_thinking_level = _resolve_thinking_level(model_name, thinking_level)
    print(f"\nLocalize Catalan -> Occitan ({model_name})")
    print(f"Thinking level: {effective_thinking_level or 'default'}")
    results = []
    failed_count = 0
    rejected_count = 0
    requests_this_run = 0

    for idx, row in enumerate(tqdm(to_process, desc="Localizing")):
        if max_requests is not None and requests_this_run >= max_requests:
            print(f"\nReached limit of {max_requests} requests this run. Exiting for key rotation.")
            break
        occ = translate_row_to_occitan(
            client,
            row["instruction"],
            row["input"],
            row["output"],
            model_name=model_name,
            thinking_level=effective_thinking_level,
        )
        if occ is None:
            failed_count += 1
        elif not _passes_post_validation(occ):
            rejected_count += 1
        else:
            out_item = {
                "instruction": occ["instruction"],
                "input": occ["input"],
                "output": occ["output"],
                "source_instruction": row["instruction"],
                "source_input": row["input"],
                "source_output": row["output"],
                "type": "alpaca_occitan_v2",
                "index": existing_count + idx,
            }
            results.append(out_item)
            try:
                with open(BUFFER_FILE, "a", encoding="utf-8") as bf:
                    bf.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"\nWarning: Could not write buffer: {e}")

        requests_this_run += 1
        time.sleep(1)

    if results:
        print(f"\nMerge {len(results)} rows -> {OUTPUT_FILE}")
        try:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as of:
                for item in results:
                    of.write(json.dumps(item, ensure_ascii=False) + "\n")
            if BUFFER_FILE.exists():
                BUFFER_FILE.unlink()
        except Exception as e:
            print(f"Error writing output: {e}. Data remains in {BUFFER_FILE}.")

    print("\n" + "=" * 60)
    print("ALPACA OCCITAN v2 LOCALIZATION BATCH COMPLETE")
    print("=" * 60)
    print(f"  Eligible source rows:  {len(catalog)}")
    print(f"  Already completed:     {existing_count}")
    print(f"  Processed this run:    {requests_this_run}")
    print(f"  Successful:            {len(results)}")
    print(f"  Failed (API/parse):    {failed_count}")
    print(f"  Rejected (validation): {rejected_count}")
    print(f"  Output file:           {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Localize Catalan Alpaca to Languedocien Occitan via Gemini (v2, filtered)."
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Stop after this many API requests this run (for key rotation; default: no limit).",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=NUM_ROWS,
        help=f"Target number of eligible source rows to localize (default: {NUM_ROWS}).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Gemini model name to use (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--thinking-level",
        type=str,
        choices=THINKING_LEVEL_CHOICES,
        default=None,
        help="Reasoning budget: minimal | low | medium | high. If omitted and model is gemini-3-flash-preview, high is used.",
    )
    args = parser.parse_args()
    NUM_ROWS = args.num_rows
    generate_synthetic_dataset(
        max_requests=args.max_requests,
        model_name=args.model_name,
        thinking_level=args.thinking_level,
    )
