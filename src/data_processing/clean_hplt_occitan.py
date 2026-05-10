"""
Clean extracted HPLT Occitan text with language and junk filters.

Example:
    python -m src.data_processing.clean_hplt_occitan --input data/hplt_v3/extracted_raw/all_data.jsonl --output data/hplt_v3/cleaned/all_data.jsonl

    python -m src.data_processing.clean_hplt_occitan --input data/hplt_v3/extracted_raw/all_data.jsonl --output data/hplt_v3/cleaned/all_data.jsonl --lid-model src/auxiliary/lid.176.bin
"""

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm


DEFAULT_INPUT = Path("data/hplt_v3/extracted_raw/all_data.jsonl")
DEFAULT_OUTPUT = Path("data/hplt_v3/cleaned/all_data.jsonl")

REGEX_WIKI_EDIT = re.compile(r'\[\s*modificar.*?\s*\]', re.IGNORECASE)

JUNK_PHRASES = [
    "modificar la font",           # Modify the font
    "clicar sul ligam",            # Click the link
    "terminar lo procès de validacion", # Finish the validation process
    "vòstre comentari es a mand",  # Your comment is awaiting
    "recebre per e",               # Receive by email 
    "anatz recebre per",           # You will receive by...
    "cal encara clicar",           # Must still click...
    "dins lo navigador"            # In the browser 
]


BLOCKED_LANGS = {
    '__label__fr', '__label__en', '__label__pt', '__label__es', 
    '__label__it', '__label__de', '__label__ro', '__label__nl',
    '__label__ca' 
}

def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"FastText language-id model not found: {model_path}. "
            "Download lid.176.bin from https://fasttext.cc/docs/en/language-identification.html "
            "or pass --lid-model /path/to/lid.176.bin."
        )
    try:
        import fasttext
    except ImportError as exc:
        raise ImportError(
            "FastText filtering was requested with --lid-model, but fasttext is not installed. "
            "Install it with: pip install fasttext-wheel"
        ) from exc
    fasttext.FastText.eprint = lambda x: None
    return fasttext.load_model(str(model_path))

def clean_text_segment(text, model):
    cleaned_lines = []
    lines = text.split('\n')
    
    for line in lines:
        line = REGEX_WIKI_EDIT.sub("", line)
        line = line.strip()

        if any(junk in line.lower() for junk in JUNK_PHRASES):
            continue

        if len(line.split()) < 5:
            continue
            
        if model is None:
            cleaned_lines.append(line)
            continue

        prediction = model.predict(line, k=1)
        label = prediction[0][0]
        score = prediction[1][0]

        if label in BLOCKED_LANGS and score > 0.8:
            continue
        
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def main():
    parser = argparse.ArgumentParser(description="Clean extracted HPLT Occitan JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--lid-model",
        type=Path,
        default=None,
        help="Optional path to FastText lid.176.bin. If omitted, language-ID filtering is skipped.",
    )
    args = parser.parse_args()

    model = load_model(args.lid_model) if args.lid_model else None
    if model is None:
        print("FastText LID filtering disabled; pass --lid-model to enable it.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:
        
        total_kept = 0
        
        for line in tqdm(fin):
            try:
                data = json.loads(line)
                
                if 'text' in data and data['text']:
                    cleaned_text = clean_text_segment(data['text'], model)
                    
                    if cleaned_text:
                        new_record = {
                            "id": data.get("id"),
                            "url": data.get("url"),
                            "text": cleaned_text
                        }
                        fout.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                        total_kept += 1
                        
            except json.JSONDecodeError:
                continue
    
    print(f"Cleanup complete. Total valid documents kept: {total_kept}")

if __name__ == "__main__":
    main()
