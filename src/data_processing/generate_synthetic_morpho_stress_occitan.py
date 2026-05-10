"""
Generate synthetic morphosyntactic stress-test Occitan sentences with Gemini.

Example:
    GEMINI_API_KEY=... python -m src.data_processing.generate_synthetic_morpho_stress_occitan
"""

import os
import json
import time
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm


MODEL_NAME = "gemini-3-pro-preview"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "synthetic" / "morphostressocc"
OUTPUT_FILE = OUTPUT_DIR / "synthetic_stress_test_250.jsonl"

MAX_RETRIES = 3
RETRY_DELAY = 2 
BATCH_SIZE = 10  


CATEGORIES = [
    {
        "name": "elision_overload",
        "count": 100,
        "prompt": """Generate {count} distinct sentences in Languedocien Occitan.

RULES (MANDATORY):
- Each sentence MUST contain AT LEAST 3-4 elisions
- Target elision patterns: d', l', qu', s', n', m', t'
- Sentences should be natural but elision-dense

EXAMPLES of elision-heavy sentences:
- "S'es n'anat d'aquí sens dire d'ont veniá."
- "L'òme qu'aviá vist s'es escapat d'aquela traïna."
- "N'i a pas qu'un qu'o sabiá, e s'es n'anat."
- "M'an dit qu'es l'ora d'anar s'adormir."
- "T'ai vist qu'anaves d'ont veniá l'aiga."

Return ONLY a JSON array of sentences, no explanations:
["sentence1", "sentence2", ...]"""
    },
    {
        "name": "diacritic_density",
        "count": 100,
        "prompt": """Generate {count} distinct sentences in Languedocien Occitan.

RULES (MANDATORY):
- Sentences MUST be packed with diacritic-heavy words
- Required characters: ò, à, è, é, í, ú, ç
- Each sentence should contain at least 4-5 accented words
- Focus on words with open vowels (ò, è) and cedilla (ç)

EXAMPLES of diacritic-dense sentences:
- "Çò que vòls es la vertat sus l'istòria d'aquela bòria."
- "L'òme vièlh parlèt de la tèrra e de l'òrt pròche."
- "Aquò's la bèla cançon qu'ausiguèrem ièr al sèr."
- "Lo còr de la vilòta èra plen de gènt que cantava."
- "La fèsta comencèt amb una dança e un còp de canòn."

Return ONLY a JSON array of sentences, no explanations:
["sentence1", "sentence2", ...]"""
    },
    {
        "name": "shibboleth",
        "count": 50,
        "prompt": """Generate {count} distinct sentences in Languedocien Occitan.

RULES (MANDATORY):
- Focus on Languedocien-specific features that distinguish it from Gascon and Catalan
- USE definite articles: lo, la, los, las (NEVER Gascon "eth/era" or Catalan "el")
- USE contractions: del, pel, al, dels, pels, als
- Include characteristic Languedocien vocabulary

EXAMPLES of dialect-marking sentences:
- "Lo lop e la lèbre son dins lo bòsc."
- "Los mainatges an passat pel pont e son anats al mercat."
- "La femna del vailet trabalhava dins los camps."
- "Las flors del prat son las mai bèlas de la valada."
- "Al ser, lo solelh se cocha darrièr los tucs."

NEVER USE:
- Gascon articles: eth, era, eths, eras
- Catalan articles: el, els
- Gascon enunciative "Que" at sentence start

Return ONLY a JSON array of sentences, no explanations:
["sentence1", "sentence2", ...]"""
    }
]

SYSTEM_PROMPT = """You are a computational linguist generating training data for an Occitan tokenizer.

Your role is to produce sentences that follow EXACT morphological specifications.
You must output ONLY the requested Occitan sentences in valid JSON format.

CRITICAL RULES:
1. Output ONLY valid JSON - no explanations, no translations, no notes
2. Each sentence must be in Languedocien Occitan (Classical Norm / Norma Classica)
3. Follow the specific morphological requirements given in each prompt EXACTLY
4. Use straight apostrophes (') only, never curly quotes
5. Sentences must be grammatically correct and semantically coherent
6. Each sentence must be UNIQUE - no duplicates

You are NOT translating. You are GENERATING authentic Occitan text for NLP training."""

def generate_sentences_batch(
    client: genai.Client,
    prompt_template: str,
    count: int,
    max_retries: int = MAX_RETRIES
) -> list[str] | None:
    prompt = prompt_template.format(count=count)
    
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.5
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config
            )
            
            if response.text:
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text
                if text.endswith("```"):
                    text = text.rsplit("```", 1)[0]
                text = text.strip()
                if text.startswith("json"):
                    text = text[4:].strip()
                
                
                sentences = json.loads(text)
                if isinstance(sentences, list):
                    return sentences
                else:
                    print(f"  Warning: Response is not a list")
            else:
                print(f"  Warning: Empty response")
                
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None
    
    return None


def generate_category_data(
    client: genai.Client,
    category: dict,
    pbar: tqdm
) -> list[dict]:
    results = []
    remaining = category["count"]
    category_name = category["name"]
    
    while remaining > 0:
        batch_size = min(BATCH_SIZE, remaining)
        
        sentences = generate_sentences_batch(
            client,
            category["prompt"],
            batch_size
        )
        
        if sentences:
            for sentence in sentences:
                if isinstance(sentence, str) and sentence.strip():
                    results.append({
                        "type": "morphological_stress",
                        "category": category_name,
                        "text": sentence.strip()
                    })
                    pbar.update(1)
            remaining -= len(sentences)
        else:
            if batch_size > 5:
                remaining = remaining  
            else:
                print(f"  Skipping remaining {remaining} for {category_name}")
                break
        
        time.sleep(2)
    
    return results


def generate_stress_test_data():

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set."
        )
    
    client = genai.Client(api_key=api_key)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_sentences = sum(cat["count"] for cat in CATEGORIES)
    
    print(f"Generating Morphological Stress Test Data using {MODEL_NAME}")
    print("=" * 60)
    for cat in CATEGORIES:
        print(f"  {cat['name']}: {cat['count']} sentences")
    print(f"  Total: {total_sentences} sentences")
    print("=" * 60 + "\n")
    
    all_results = []
    
    with tqdm(total=total_sentences, desc="Generating sentences") as pbar:
        for category in CATEGORIES:
            print(f"\nProcessing category: {category['name']}")
            category_results = generate_category_data(client, category, pbar)
            all_results.extend(category_results)
    
    print(f"\nWriting {len(all_results)} sentences to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    
    category_counts = {}
    for item in all_results:
        cat = item["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat_name, count in category_counts.items():
        expected = next(c["count"] for c in CATEGORIES if c["name"] == cat_name)
        status = "OK" if count >= expected else f"({expected - count} missing)"
        print(f"  {cat_name}: {count} sentences {status}")
    
    print(f"\n  Total sentences generated: {len(all_results)}")
    print(f"  Output file: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    generate_stress_test_data()
