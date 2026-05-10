"""
Align new Occitan tokens to French and Catalan equivalents with Gemini.

Example:
    GEMINI_API_KEY=... python -m src.tokenization.align_occitan_tokens_with_gemini --input notebooks/occitan_llama_tokenizer_patched/added_tokens_list.txt --output data/alignments_gemini.json
"""

import os
import json
import time
import argparse
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "notebooks" / "occitan_llama_tokenizer_patched" / "added_tokens_list.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "alignments_gemini.json"
BUFFER_FILE = PROJECT_ROOT / "data" / "alignments_buffer.jsonl"


MODEL_NAME = "gemini-3-pro-preview"
BATCH_SIZE = 50
MAX_RETRIES = 5
RETRY_DELAY = 2  


SYSTEM_PROMPT = """You are an expert Romance linguist specializing in Occitan, French, and Catalan.

Your task is to translate Occitan words into their closest French and Catalan equivalents.

## CRITICAL RULES:

### 1. Morphological Preservation
When an Occitan word contains contractions, elisions, or specific morphological markers:
- Find the STRUCTURALLY EQUIVALENT form in French/Catalan
- Examples:
  - d'aquesta (de + aquesta) → French: "de cette" OR "d'une" / Catalan: "d'aquesta"
  - l'òme (l' + òme) → French: "l'homme" / Catalan: "l'home"
  - qu'es (qu' + es) → French: "qu'est" OR "qui est" / Catalan: "que és"

### 2. Single Words
For simple words without contractions:
- Provide the most common translation
- Examples:
  - òme → French: "homme" / Catalan: "home"
  - femna → French: "femme" / Catalan: "dona"

### 3. Function Words
For articles, prepositions, conjunctions:
- Provide the direct equivalent
- Examples:
  - lo → French: "le" / Catalan: "el"
  - del → French: "du" / Catalan: "del"

### 4. Diacritics and Special Characters
For single diacritic characters or very short tokens:
- If it's just a character (like ò, è, ç), return it unchanged for both languages
- Example: ò → French: "ò" / Catalan: "ò"

### 5. Output Format
Return ONLY valid JSON with no additional text, markdown formatting, or code blocks.
The JSON must be an object where:
- Keys are the Occitan words (exactly as provided)
- Values are objects with "fr" (French) and "ca" (Catalan) keys

Example output for ["l'òme", "femna", "lo"]:
{"l'òme": {"fr": "l'homme", "ca": "l'home"}, "femna": {"fr": "femme", "ca": "dona"}, "lo": {"fr": "le", "ca": "el"}}"""


def load_tokens(input_file: Path) -> list[str]:
    
    print(f"Loading tokens from {input_file}...")
    
    tokens = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token: 
                tokens.append(token)
    
    print(f"Loaded {len(tokens)} tokens.")
    return tokens


def load_existing_translations(output_file: Path) -> dict:
   
    if not output_file.exists():
        return {}
    
    print(f"Loading existing translations from {output_file}...")
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Found {len(data)} existing translations.")
        return data
    except (json.JSONDecodeError, Exception) as e:
        print(f"Warning: Could not load existing file: {e}")
        return {}


def load_buffer(buffer_file: Path) -> dict:
    
    if not buffer_file.exists():
        return {}
    
    print(f"Found buffer file, loading intermediate results.")
    buffer_data = {}
    try:
        with open(buffer_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        buffer_data.update(item)
                    except json.JSONDecodeError:
                        continue
        print(f"Recovered {len(buffer_data)} translations from buffer.")
    except Exception as e:
        print(f"Warning: Could not read buffer: {e}")
    
    return buffer_data


def translate_batch(
    client: genai.Client,
    tokens: list[str],
    max_retries: int = MAX_RETRIES
) -> dict | None:
    
    prompt = f"""Translate these Occitan words to French and Catalan.

Words to translate:
{json.dumps(tokens, ensure_ascii=False)}

Return ONLY the JSON object, no other text."""

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2  
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config
            )
            
            if not response.text:
                print(f"  Warning: Empty response for batch")
                continue
            
            
            text = response.text.strip()
            if text.startswith("```"):
                
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            
            
            result = json.loads(text)
            
            
            if not isinstance(result, dict):
                print(f"  Warning: Response is not a dict, retrying...")
                continue
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
                
        except Exception as e:
            error_msg = str(e).lower()
            is_overloaded = "429" in error_msg or "resource_exhausted" in error_msg or "overloaded" in error_msg
            
            if is_overloaded:
                wait_time = 60 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None
    
    return None


def char_ngram_cosine_similarity(s1: str, s2: str, n: int = 2) -> float:
    """Calculate character n-gram cosine similarity between two strings."""
    import math
    from collections import Counter
    
    if not s1 or not s2:
        return 0.0
        
    s1_lower, s2_lower = s1.lower(), s2.lower()
    
    s1_pad = f"^{s1_lower}$"
    s2_pad = f"^{s2_lower}$"
    
    vec1 = Counter(s1_pad[i:i+n] for i in range(len(s1_pad) - n + 1))
    vec2 = Counter(s2_pad[i:i+n] for i in range(len(s2_pad) - n + 1))
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)
    
    sum1 = sum(vec1[x]**2 for x in vec1.keys())
    sum2 = sum(vec2[x]**2 for x in vec2.keys())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    return float(numerator) / denominator if denominator else 0.0


def convert_to_alignment_format(translations: dict) -> dict:
    
    alignments = {}
    for occ_word, trans in translations.items():
        if isinstance(trans, dict) and "fr" in trans and "ca" in trans:
            sim_fr = char_ngram_cosine_similarity(occ_word, trans["fr"])
            sim_ca = char_ngram_cosine_similarity(occ_word, trans["ca"])
            
            # Add small epsilon to prevent 0 weights, and normalize to sum to 1
            eps = 0.01
            weight_fr = sim_fr + eps
            weight_ca = sim_ca + eps
            total = weight_fr + weight_ca
            
            alignments[occ_word] = [
                [trans["fr"], round(weight_fr / total, 3)],
                [trans["ca"], round(weight_ca / total, 3)]
            ]
    return alignments


def main():
    parser = argparse.ArgumentParser(
        description="Translate Occitan tokens to French/Catalan using Gemini API"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input file with Occitan tokens (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=BATCH_SIZE,
        help=f"Tokens per API call (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making API calls"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tokens to process (for testing)"
    )
    args = parser.parse_args()

    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Set it with: export GEMINI_API_KEY='your-key'"
        )

    
    all_tokens = load_tokens(args.input)
    
    
    if args.limit:
        all_tokens = all_tokens[:args.limit]
        print(f"Limited to {len(all_tokens)} tokens for testing.")

    
    existing = load_existing_translations(args.output)
    buffer_data = load_buffer(BUFFER_FILE)
    
    
    existing.update(buffer_data)
    
    
    tokens_to_translate = [t for t in all_tokens if t not in existing]
    
    if not tokens_to_translate:
        print("All tokens have already been translated!")
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        alignments = convert_to_alignment_format(existing)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(alignments, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(alignments)} alignments to {args.output}")
        return

    print(f"\nTokens remaining to translate: {len(tokens_to_translate)}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would translate the following batches:")
        for i in range(0, len(tokens_to_translate), args.batch_size):
            batch = tokens_to_translate[i:i + args.batch_size]
            print(f"  Batch {i // args.batch_size + 1}: {len(batch)} tokens")
            if i == 0:
                print(f"    First few: {batch[:5]}")
        return

    
    client = genai.Client(api_key=api_key)
    
   
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    
    translations = dict(existing) 
    success_count = 0
    fail_count = 0
    
    num_batches = (len(tokens_to_translate) + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(0, len(tokens_to_translate), args.batch_size), 
                  desc="Translating", total=num_batches):
        batch = tokens_to_translate[i:i + args.batch_size]
        
        result = translate_batch(client, batch)
        
        if result:
            translations.update(result)
            success_count += len(result)
            
           
            try:
                with open(BUFFER_FILE, "a", encoding="utf-8") as bf:
                    bf.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"\nWarning: Could not write to buffer: {e}")
        else:
            fail_count += len(batch)
        
        
        time.sleep(1)
    
    
    alignments = convert_to_alignment_format(translations)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(alignments, f, ensure_ascii=False, indent=2)
    
   
    if BUFFER_FILE.exists():
        BUFFER_FILE.unlink()
        print("Buffer cleared.")
    
    
    print("TRANSLATION COMPLETE")
    print("-" * 60)
    print(f"  Total tokens:              {len(all_tokens)}")
    print(f"  Previously translated:     {len(existing)}")
    print(f"  Translated this run:       {success_count}")
    print(f"  Failed this run:           {fail_count}")
    print(f"  Output file:               {args.output}")
    print("-" * 60)


if __name__ == "__main__":
    main()
