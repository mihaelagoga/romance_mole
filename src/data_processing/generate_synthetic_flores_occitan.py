"""
Generate synthetic FLORES-style Languedocien Occitan data with Gemini.

Example:
    GEMINI_API_KEY=... python -m src.data_processing.generate_synthetic_flores_occitan
"""

import os
import json
import time
from pathlib import Path

from datasets import load_dataset
from google import genai
from google.genai import types
from tqdm import tqdm

NUM_SENTENCES = 500
MODEL_NAME = "gemini-3-pro-preview"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "synthetic" / "flores200occ"
OUTPUT_FILE = OUTPUT_DIR / "synthetic_flores_500.jsonl"
BUFFER_FILE = OUTPUT_DIR / "buffer_flores_500.jsonl"

MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds

SYSTEM_PROMPT = """You are an expert translator and linguist specializing in Occitan, 
specifically the Languedocien (Lengadocian) dialect following the Classical Norm (Norma Classica) 
established by Louis Alibert.

Your task is to **localize** French text into Languedocien Occitan. This is NOT a literal 
word-for-word translation—you must produce natural, idiomatic Occitan that a native speaker 
of the Lengadocian dialect would use.

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
- Use -èm for 1st person plural present (parlam → parlam is also acceptable)
- Use -ètz for 2nd person plural
- Use -on/-an for 3rd person plural depending on conjugation class

## OUTPUT REQUIREMENTS:
- Provide ONLY the Occitan translation, nothing else
- No explanations, notes, or alternatives
- No quotation marks around the translation
- Preserve the original meaning and tone
- Adapt cultural references appropriately when needed"""


def load_flores_french(num_sentences: int = 500) -> list[str]:
    """Load French sentences from the FLORES-200 dataset."""
    print(f"Loading FLORES-200 French (flores_fr) subset from DGME/FLORES-200...")
    
    try:
        dataset = load_dataset(
            "DGME/FLORES-200",
            "flores_fr",
            split="dev",  
            trust_remote_code=False
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        dataset = load_dataset(
            "DGME/FLORES-200",
            "flores_fr",
            split="devtest", 
            trust_remote_code=False
        )
    
    sentences = []
    for i, example in enumerate(dataset):
        if i >= num_sentences and len(sentences) >= num_sentences:
            break
        
        # DGME/FLORES-200 uses 'text' column
        text = example.get("text", "")
        if text and text not in sentences:
            sentences.append(text)
            
        if len(sentences) >= num_sentences:
            break
    
    print(f"Loaded {len(sentences)} unique French sentences.")
    return sentences[:num_sentences]


def translate_to_occitan(
    client: genai.Client,
    french_text: str,
    max_retries: int = MAX_RETRIES
) -> str | None:
    """Translate a French sentence to Languedocien Occitan using Gemini API."""
    prompt = f"Translate the following French text into Languedocien Occitan:\n\n{french_text}"
    
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.3
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config
            )
            
            if response.text:
                return response.text.strip()
            else:
                print(f"  Warning: Empty response for: {french_text[:50]}...")
                
        except Exception as e:
            error_msg = str(e).lower()
            is_overloaded = "429" in error_msg or "resource_exhausted" in error_msg or "overloaded" in error_msg
            
            if is_overloaded:
                wait_time = 60 * (attempt + 1)
                print(f"  [OVERLOADED] Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None
    
    return None


def generate_synthetic_dataset():


    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
        )
    
    client = genai.Client(api_key=api_key)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if BUFFER_FILE.exists():
        print(f"\nFound existing buffer file: {BUFFER_FILE}")
        print("Merging buffer into main output file before starting...")
        try:
            buffer_count = 0
            with open(BUFFER_FILE, "r", encoding="utf-8") as bf, open(OUTPUT_FILE, "a", encoding="utf-8") as of:
                for line in bf:
                    line = line.strip()
                    if line:
                        of.write(line + "\n")
                        buffer_count += 1
            print(f"Merged {buffer_count} items from buffer.")
            BUFFER_FILE.unlink()
            print("Buffer cleared.")
        except Exception as e:
            print(f"ERROR merging buffer: {e}")
            print("Suggest manual inspection of buffer_flores_500.jsonl")

    existing_source_texts = set()
    if OUTPUT_FILE.exists():
        print(f"Checking existing file for progress: {OUTPUT_FILE}")
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if "source_text" in data:
                                existing_source_texts.add(data["source_text"])
                        except json.JSONDecodeError:
                            continue
            print(f"Found {len(existing_source_texts)} already translated sentences.")
        except Exception as e:
            print(f"Warning: Could not read existing file: {e}")

    all_french_sentences = load_flores_french(NUM_SENTENCES)
    
    sentences_to_process = [s for s in all_french_sentences if s not in existing_source_texts]
    
    if not sentences_to_process:
        print("All sentences have already been translated!")
        return

    print(f"Sentences remaining to translate: {len(sentences_to_process)}")
    
    print(f"\nGenerating Occitan translations using {MODEL_NAME}...")
    results = []
    failed_count = 0
    
    
    for french_text in tqdm(sentences_to_process, desc="Translating"):
        occitan_text = translate_to_occitan(client, french_text)
        
        if occitan_text:
            result_item = {
                "source_text": french_text,
                "target_text": occitan_text,
                "type": "flores_localization"
            }
            results.append(result_item)
            
            try:
                with open(BUFFER_FILE, "a", encoding="utf-8") as bf:
                    bf.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"\nWarning: Could not write to buffer: {e}")
                
        else:
            failed_count += 1
        
        time.sleep(1)
    
    if results:
        print(f"\nRun complete. Merging {len(results)} new translations from buffer to {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as of:
                for item in results:
                    of.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            if BUFFER_FILE.exists():
                BUFFER_FILE.unlink()
                print("Buffer cleared.")
        except Exception as e:
            print(f"Error writing to main file: {e}")
            print(f"Data is safe in {BUFFER_FILE}")

    
    print("\n" + "=" * 60)
    print("GENERATION BATCH COMPLETE")
    print("=" * 60)
    print(f"  Total required:            {len(all_french_sentences)}")
    print(f"  Previously done:           {len(existing_source_texts)}")
    print(f"  Processed this run:        {len(sentences_to_process)}")
    print(f"  Successful (this run):     {len(results)}")
    print(f"  Failed (this run):         {failed_count}")
    print(f"  Output file:               {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    generate_synthetic_dataset()
