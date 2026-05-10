"""
Patch a Llama tokenizer with Occitan-specific lexical and morphosyntactic tokens.

Example:
    python -m src.tokenization.build_occitan_tokenizer --base-model proxectonos/Llama-3.1-Carballo --output models/occitan_llama_tokenizer_patched
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

from transformers import AutoTokenizer
from tqdm import tqdm


DEFAULT_BASE_MODEL = "proxectonos/Llama-3.1-Carballo"

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "tokenizer_training_corpus" / "training_corpus.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "occitan_llama_tokenizer_patched"


DEFAULT_CORPUS_WORDS = 500


def bytes_to_unicode():
    """Returns list of utf-8 byte and a corresponding list of unicode strings."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def string_to_byte_string(s: str) -> str:
    """
    Convert a normal string (e.g. "Occitània") into the GPT-2 byte-string
    representation (e.g. "OccitÃłnia") required by ByteLevel BPE.
    """
    b2u = bytes_to_unicode()
    return "".join([b2u[b] for b in s.encode("utf-8")])


def byte_string_to_string(s: str) -> str:
    """
    Convert a GPT-2 byte-string representation back to a normal string.
    """
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    return bytes([u2b.get(c, ord(c)) for c in s]).decode("utf-8", errors="replace")


def get_morphology_tokens() -> list[str]:
    """Build Occitan-specific tokens to inject into the vocabulary."""

    tokens = []
    
    elisions = [
        "d'", "l'", "qu'", "s'", "n'", "m'", "t'",
        "D'", "L'", "Qu'", "S'", "N'", "M'", "T'",
    ]
    tokens.extend(elisions)
    
    elision_combos = [
        # d' combinations
        "d'aquò", "d'aquel", "d'aquela", "d'aquí", "d'ont", "d'una", "d'un",
        # l' combinations
        "l'òme", "l'aiga", "l'ostal", "l'ora", "l'autre", "l'altra",
        # qu' combinations
        "qu'es", "qu'èra", "qu'aviá", "qu'an", "qu'avèm",
        # s' combinations
        "s'escapa", "s'es", "s'en", "s'anar",
        # n' combinations
        "n'i", "n'a", "n'avèm",
        # m' combinations
        "m'agrada", "m'an", "m'a",
        # t' combinations
        "t'ai", "t'an", "t'a",
    ]
    tokens.extend(elision_combos)
    tokens.extend([w.capitalize() for w in elision_combos if not w[0].isupper()])
    
    diacritic_words = [
        # ò words
        "çò", "aquò", "aiçò", "bòria", "còr", "còp", "fòrt", "jòc", "mòrt",
        "nòu", "pòrta", "sòm", "tròp", "vòl",
        # è words  
        "tèrra", "fèsta", "lèu", "bèl", "bèla", "cèl", "pèl", "sèr", "vèrb",
        # combined
        "cançon", "garçon", "plaça", "peça", "braç", "dolç",
    ]
    tokens.extend(diacritic_words)
    
    tokens.extend([w.capitalize() for w in diacritic_words])
    
    common_words = [
        # pronouns
        "ieu", "ela", "eles", "elas",
        # verbs
        "èsser", "aver", "dire", "anar", "venir", "poder", "voler",
        "saber", "veire", "parlar", "cantar", "manjar", "beure",
        # time
        "ièr", "uèi", "deman", "nuèit", "setmana",
        # conjunctions / prepositions
        "perqué", "cossí", "quora", "dins",
        # common nouns
        "òme", "femna", "mainatge", "ostal", "vilòta", "carrièra", "país",
    ]
    tokens.extend(common_words)
    tokens.extend([w.capitalize() for w in common_words])
    
    seen = set()
    unique_tokens = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)
    
    return unique_tokens


def extract_fragmented_words(
    tokenizer,
    corpus_file: Path,
    max_words: int = 500,
    min_word_length: int = 3,
    min_frequency: int = 5,
) -> list[str]:
    print(f"Scanning corpus for frequently fragmented words")
    
    if not corpus_file.exists():
        print(f"  Warning: Corpus file not found: {corpus_file}")
        return []
    
    
    with open(corpus_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    
    words = re.findall(r"[a-zA-ZàèéíòóúçÀÈÉÍÒÓÚÇ']+", text.lower())
    
    
    word_counts = Counter(words)
    
    
    fragmented_words = []
    
    for word, count in tqdm(word_counts.most_common(10000), desc="  Checking words"):
        
        if len(word) < min_word_length or count < min_frequency:
            continue
        
      
        try:
            tokens = tokenizer.tokenize(word)
            if len(tokens) > 1:
                fragmented_words.append(word)
                
                fragmented_words.append(word.capitalize())
        except Exception:
            continue
        
        if len(fragmented_words) >= max_words * 2: 
            break
    
    
    seen = set()
    unique = []
    for w in fragmented_words:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    
    print(f"  Found {len(unique)} fragmented words to repair")
    return unique[:max_words]


def patch_tokenizer(
    base_model: str,
    corpus_file: Path,
    output_dir: Path,
    corpus_words: int = DEFAULT_CORPUS_WORDS,
    skip_corpus_scan: bool = False,
):

    print("=" * 60)
    print("TARGETED VOCABULARY INJECTION")
    print("=" * 60)
    print(f"  Base model:        {base_model}")
    print(f"  Corpus file:       {corpus_file}")
    print(f"  Output directory:  {output_dir}")
    print(f"  Corpus words:      {corpus_words}")
    print(f"  Skip corpus scan:  {skip_corpus_scan}")
    print("=" * 60 + "\n")
    
    
    print("Step 1: Loading base tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except Exception as e:
        print(f"Error loading model '{base_model}': {e}")
        return None
    
    original_vocab_size = len(tokenizer)
    print(f"  Initial vocab size: {original_vocab_size:,}")
    
   
    print("\nStep 2: Collecting morphology tokens.")
    morphology_tokens = get_morphology_tokens()
    print(f"  Morphology tokens: {len(morphology_tokens)}")
    
    
    corpus_tokens = []
    if not skip_corpus_scan and corpus_file.exists():
        print("\nStep 3: Extracting fragmented corpus words.")
        corpus_tokens = extract_fragmented_words(
            tokenizer, corpus_file, max_words=corpus_words
        )
    else:
        print("\nStep 3: Skipping corpus scan")
    
    all_tokens = morphology_tokens + corpus_tokens
    seen = set()
    unique_tokens = []
    for t in all_tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)
    
    existing_vocab = set(tokenizer.get_vocab().keys())
    before_filter = len(unique_tokens)

    def token_already_exists(tok: str) -> bool:
        if tok in existing_vocab:
            return True
        # Normalization-aware check: catches cases where add_tokens would map
        # the candidate to an existing token ID despite string mismatch.
        tid = tokenizer.convert_tokens_to_ids(tok)
        return tid is not None and tid != tokenizer.unk_token_id

    unique_tokens = [t for t in unique_tokens if not token_already_exists(t)]
    
    # The Llama-3/GPT-2 BPE tokenizer works on ByteLevel tokens.
    # To add tokens with non-ASCII chars (like à, ò), we MUST map the UTF-8 bytes 
    # to the corresponding BPE unicode string.
    # For example, "l'òme" -> "l'Ã²me"
    unique_tokens_bpe = [string_to_byte_string(t) for t in unique_tokens]
    
    skipped = before_filter - len(unique_tokens)
    
    print(f"\nStep 4: Injecting tokens.")
    print(f"  Candidates:        {before_filter}")
    print(f"  Already in vocab:  {skipped} (skipped — avoids ByteLevel BPE collision)")
    print(f"  To add:            {len(unique_tokens_bpe)}")
    
    num_added = tokenizer.add_tokens(unique_tokens_bpe)
    print(f"  Successfully added: {num_added} unique tokens")
    print(f"  New vocab size:     {len(tokenizer):,}")

    # Safety invariant:
    # Any token in added vocab must have an ID >= original base vocab size.
    # If this is violated, we'd reintroduce byte-level collisions (e.g. à -> 156).
    added_vocab = tokenizer.get_added_vocab()
    injected_set = set(unique_tokens_bpe)
    colliding = {
        tok: tid for tok, tid in added_vocab.items() 
        if tid < original_vocab_size and tok in injected_set
    }
    if colliding:
        print("\nERROR: Detected colliding added tokens that map into base vocab IDs.")
        print("This would reintroduce replacement-character decoding issues.")
        for tok, tid in sorted(colliding.items(), key=lambda x: x[1])[:30]:
            print(f"  - {repr(tok)} -> {tid}")
        raise RuntimeError(
            "Refusing to save broken tokenizer. Remove colliding tokens from candidate list."
        )
    
   
    print(f"\nStep 5: Saving patched tokenizer to {output_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    
    
    metadata = {
        "base_model": base_model,
        "corpus_file": str(corpus_file),
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": len(tokenizer),
        "tokens_added": num_added,
        "morphology_tokens": len(morphology_tokens),
        "corpus_tokens": len(corpus_tokens),
        "created_at": datetime.now().isoformat(),
    }
    
    with open(output_dir / "patch_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    
    added_tokens_sorted = [
        byte_string_to_string(tok) for tok, tid in sorted(added_vocab.items(), key=lambda x: x[1])
        if tok in injected_set
    ]
    with open(output_dir / "added_tokens_list.txt", "w", encoding="utf-8") as f:
        for token in added_tokens_sorted:
            f.write(token + "\n")
    
    
    print("\n" + "=" * 60)
    print("VALIDATION CHECK")
    print("=" * 60)
    
    test_cases = [

        # elision combinations
        (string_to_byte_string("d'aquò"), "elision"),
        (string_to_byte_string("l'òme"), "elision"),
        (string_to_byte_string("qu'es"), "elision"),
        (string_to_byte_string("s'escapa"), "elision"),
        (string_to_byte_string("n'i"), "elision"),
        (string_to_byte_string("m'agrada"), "elision"),
        (string_to_byte_string("t'ai"), "elision"),

        # diacritic words
        (string_to_byte_string("çò"), "diacritic"),
        (string_to_byte_string("bòria"), "diacritic"),
        (string_to_byte_string("tèrra"), "diacritic"),
        (string_to_byte_string("fèsta"), "diacritic"),
        (string_to_byte_string("cançon"), "diacritic"),

        # articles
        (string_to_byte_string("lo"), "article"),
        (string_to_byte_string("la"), "article"),
        (string_to_byte_string("los"), "article"),
        (string_to_byte_string("las"), "article"),
        (string_to_byte_string("del"), "contraction"),
        (string_to_byte_string("pel"), "contraction"),
        (string_to_byte_string("al"), "contraction"),
    ]
    
    fixed = 0
    for text, category in test_cases:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text, add_special_tokens=False)

        decoded_text = byte_string_to_string(text)

        is_fixed = False
        if "'" in decoded_text:
            is_fixed = len(tokens) <= 2
        else:
            is_fixed = len(tokens) == 1
        
        status = "FIXED" if is_fixed else " Still split"
        if is_fixed:
            fixed += 1
        
        print(f"  '{decoded_text}' ({category})")
        print(f"     Tokens: {tokens} ({len(tokens)})")
        print(f"     {status}")
    
    print(f"\nResult: {fixed}/{len(test_cases)} test words fixed")
    
    print("\n" + "=" * 60)
    print("TOKENIZER PATCHING COMPLETE")
    print("=" * 60)
    
    return tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Patch Llama 3 tokenizer with the targeted Occitan vocabulary"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_BASE_MODEL})"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"Path to training corpus (default: {DEFAULT_CORPUS})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--corpus-words",
        type=int,
        default=DEFAULT_CORPUS_WORDS,
        help=f"Number of corpus words to add (default: {DEFAULT_CORPUS_WORDS})"
    )
    parser.add_argument(
        "--skip-corpus-scan",
        action="store_true",
        help="Skip corpus scan and only add morphology tokens"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    patch_tokenizer(
        base_model=args.base_model,
        corpus_file=args.corpus,
        output_dir=args.output,
        corpus_words=args.corpus_words,
        skip_corpus_scan=args.skip_corpus_scan,
    )


if __name__ == "__main__":
    main()
