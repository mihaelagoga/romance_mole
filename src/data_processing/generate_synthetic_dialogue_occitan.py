"""
Generate synthetic Languedocien Occitan dialogue data with Gemini.

Example:
    GEMINI_API_KEY=... python -m src.data_processing.generate_synthetic_dialogue_occitan
"""

import os
import json
import time
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm


MODEL_NAME = "gemini-3-pro-preview"
EXCHANGES_PER_SCENARIO = 10
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "synthetic" / "dialogocc"
OUTPUT_FILE = OUTPUT_DIR / "synthetic_dialogue_250.jsonl"

MAX_RETRIES = 3
RETRY_DELAY = 2  


SCENARIOS = [
    "Ordering coffee at a café",
    "Asking for directions in town",
    "Complaining about the weather",
    "Buying train tickets at the station",
    "Gossip between neighbors",
    "Haggling at the market",
    "Making a doctor's appointment",
    "Discussing the harvest with a farmer",
    "Planning a village festival",
    "Greeting an old friend after years",
    "Ordering food at a restaurant",
    "Asking about family news",
    "Complaining about noisy neighbors",
    "Discussing a football match",
    "Buying bread at the bakery",
    "Asking about bus schedules",
    "Talking about children's school",
    "Planning a weekend trip",
    "Discussing local politics",
    "Borrowing tools from a neighbor",
    "Returning something to a shop",
    "Inviting someone to dinner",
    "Apologizing for being late",
    "Discussing a recipe",
    "Saying goodbye before a journey",
]


SYSTEM_PROMPT = """You are an expert linguist specializing in spoken Occitan, 
specifically the Languedocien (Lengadocian) dialect following the Classical Norm (Norma Classica).

Your task is to generate natural, conversational dialogue exchanges in Languedocien Occitan.
These dialogues will be used to stress-test a tokenizer, so they MUST be rich in clitics,
elisions, and colloquial speech patterns that are common in spoken Occitan but rare in formal text.

## MANDATORY LINGUISTIC FEATURES:

### 1. Clitic Pronouns (USE FREQUENTLY):
Force the use of weak/clitic pronouns attached to verbs:
- m' (me) → "M'agrada", "Diga-m"
- t' (te/you) → "T'ai vist", "Te'n vas?"
- s' (se/oneself) → "S'escapa", "Se n'anèt"
- li (to him/her) → "Li diguèri", "Diga-li"
- los/las (them) → "Los vesi", "Porta-los"
- o (it) → "O sabi", "Dóna-m'o"
- ne/n' (of it, some) → "N'i a", "Ne vòli mai"
- i (there, to it) → "I anam", "Pòrta-m'i"

### 2. Complex Clitic Combinations (ESSENTIAL):
Create sentences with multiple clitics:
- "Te'n vas?" (Are you leaving? - lit. you-of-it go)
- "Dóna-m'o" (Give it to me)
- "No n'i a pas" (There isn't any)
- "Me'n vau" (I'm leaving)
- "T'o diguèri" (I told you so)
- "Se n'es anat" (He/she left)

### 3. Interrogative Words (USE THESE):
- Ont / Ont es / Ont se tròba (Where)
- Cossí (How)
- Perqué (Why)
- Quora (When)
- Qual (Who)
- Qué / Qu'es aquò (What)
- Quant/Quanta (How much/many)

### 4. Elision Rules (APPLY STRICTLY):
- lo/la → l' before vowels: l'òme, l'aiga
- de → d' before vowels: d'aquò, d'aquel
- que → qu' before vowels: qu'es, qu'avèm
- se → s' before vowels: s'escapa
- ne → n' before vowels: n'i a
- me/te → m'/t' before vowels: m'agrada, t'ai vist

### 5. Common Colloquial Expressions:
- "Qu'es aquò?" (What's that?)
- "Vai plan!" (Take it easy!)
- "Te calhas!" (Be quiet!)
- "Ont vas?" (Where are you going?)
- "N'i a pus" (There's no more)
- "Fas pas res" (It doesn't matter / You're welcome)

## NEGATIVE CONSTRAINTS (NEVER DO):
- NO Gascon enunciative "Que" at sentence start (e.g., DON'T write "Que vòli...")
- NO French sentence structures or calques
- NO Gascon articles (eth, era) - USE lo, la, los, las
- NO Provençal contractions (dau) - USE del, pel, al

## APOSTROPHE STYLE:
- USE ONLY straight apostrophes: '
- DO NOT USE curly/typographic apostrophes: ' or '

## OUTPUT FORMAT:
You will be given a scenario. Generate EXACTLY 10 distinct dialogue exchanges.
Return ONLY valid JSON array, no other text. Each exchange must have:
- "speaker_A": First speaker's line in Occitan
- "speaker_B": Second speaker's response in Occitan

Example output format:
[
  {"speaker_A": "Pardon, ont se tròba la gara?", "speaker_B": "Es al cap del carrièra, viratz a dreita."},
  {"speaker_A": "...", "speaker_B": "..."}
]"""


def generate_dialogues_for_scenario(
    client: genai.Client,
    scenario: str,
    num_exchanges: int = EXCHANGES_PER_SCENARIO,
    max_retries: int = MAX_RETRIES
) -> list[dict] | None:
    """Generate dialogue exchanges for a given scenario."""
    prompt = f"""Generate {num_exchanges} conversational exchanges in Languedocien Occitan 
for the following scenario: "{scenario}"

Remember:
- Pack the dialogues with clitics (m', t', s', li, o, ne, n', i)
- Use complex clitic combinations like "Te'n vas?", "Dóna-m'o", "No n'i a pas"
- Use natural spoken patterns, not formal written style
- Return ONLY the JSON array, no explanations"""

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.7  # Higher temperature for creative dialogue
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
                
                exchanges = json.loads(text)
                return exchanges
            else:
                print(f"  Warning: Empty response for scenario: {scenario}")
                
        except json.JSONDecodeError as e:
            print(f"  JSON parse error for '{scenario}': {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries} for '{scenario}': {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None
    
    return None


def generate_synthetic_dialogues():


    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
        )
    
    client = genai.Client(api_key=api_key)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating Occitan dialogues using {MODEL_NAME}...")
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Exchanges per scenario: {EXCHANGES_PER_SCENARIO}")
    print(f"Target total: {len(SCENARIOS) * EXCHANGES_PER_SCENARIO} exchanges\n")
    
    results = []
    failed_scenarios = []
    
    for scenario in tqdm(SCENARIOS, desc="Processing scenarios"):
        exchanges = generate_dialogues_for_scenario(client, scenario)
        
        if exchanges:
            for exchange in exchanges:
                results.append({
                    "type": "dialogue",
                    "scenario": scenario,
                    "speaker_A": exchange.get("speaker_A", ""),
                    "speaker_B": exchange.get("speaker_B", "")
                })
        else:
            failed_scenarios.append(scenario)
        
        time.sleep(1)
    
    print(f"\nWriting {len(results)} dialogue exchanges to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total scenarios processed: {len(SCENARIOS)}")
    print(f"  Successful exchanges:      {len(results)}")
    print(f"  Failed scenarios:          {len(failed_scenarios)}")
    if failed_scenarios:
        print(f"  Failed list: {failed_scenarios}")
    print(f"  Output file:               {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    generate_synthetic_dialogues()
