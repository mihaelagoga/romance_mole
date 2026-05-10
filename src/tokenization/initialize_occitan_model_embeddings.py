"""
Initialize embeddings for the patched Occitan tokenizer.

Example:
    python -m src.tokenization.initialize_occitan_model_embeddings --base-model proxectonos/Llama-3.1-Carballo --patched-tokenizer models/occitan_llama_tokenizer_patched --alignments data/alignments_gemini.json --output models/llama-3.1-occitan-initialized
"""

import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_BASE_MODEL = "proxectonos/Llama-3.1-Carballo"
DEFAULT_PATCHED_TOKENIZER = PROJECT_ROOT / "models" / "occitan_llama_tokenizer_patched"
DEFAULT_ALIGNMENTS = PROJECT_ROOT / "data" / "alignments_gemini.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "llama-3.1-occitan-initialized"


def load_alignments(alignments_file: Path) -> dict:
    
    print(f"Loading alignments from {alignments_file}...")
    with open(alignments_file, "r", encoding="utf-8") as f:
        alignments = json.load(f)
    print(f"Loaded {len(alignments)} alignments.")
    return alignments


def get_token_embedding(
    token: str,
    tokenizer,
    embedding_matrix: torch.Tensor
) -> torch.Tensor | None:
    
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    
    if not token_ids:
        return None
    
   
    embeddings = []
    for tid in token_ids:
        if tid < embedding_matrix.shape[0]:
            embeddings.append(embedding_matrix[tid])
    
    if not embeddings:
        return None
    
    
    if len(embeddings) == 1:
        return embeddings[0]
    else:
        return torch.stack(embeddings).mean(dim=0)


def bytes_to_unicode():
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

def byte_string_to_string(s: str) -> str:
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    return bytes([u2b.get(c, ord(c)) for c in s]).decode("utf-8", errors="replace")


def tokenizer_roundtrip_sanity(tokenizer) -> tuple[bool, list[str]]:
    """Quick sanity check to catch broken ByteLevel added-token artifacts."""
    probes = [
        "automàtic",
        "és",
        "què",
        "l'òme",
        "Occitània",
    ]
    failures = []
    for text in probes:
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        if decoded != text:
            failures.append(f"{text!r} -> {decoded!r} ids={ids}")
    return len(failures) == 0, failures


def initialize_new_embeddings(
    model,
    original_tokenizer,
    patched_tokenizer,
    alignments: dict,
    dry_run: bool = False
) -> dict:
    
   
    input_embeddings = model.get_input_embeddings().weight
    output_embeddings = model.get_output_embeddings().weight
    
    original_vocab_size = len(original_tokenizer)
    patched_vocab_size = len(patched_tokenizer)
    
    print(f"\nOriginal vocabulary size: {original_vocab_size}")
    print(f"Patched vocabulary size: {patched_vocab_size}")
    print(f"New tokens to initialize: {patched_vocab_size - original_vocab_size}")
    
    stats = {
        "total_new_tokens": patched_vocab_size - original_vocab_size,
        "initialized_from_cognates": 0,
        "random_init": 0,
        "failed_lookups": [],
        "multi_token_cognates": 0
    }
    
    
    new_token_ids = list(range(original_vocab_size, patched_vocab_size))
    
    for new_id in tqdm(new_token_ids, desc="Initializing embeddings"):
        
        new_token = patched_tokenizer.decode([new_id])
        
        
        token_found = None
        for token, tid in patched_tokenizer.get_vocab().items():
            if tid == new_id:
                token_found = token
                break
        
        if token_found:
            new_token = byte_string_to_string(token_found)
        
       
        if new_token not in alignments:
            
            new_token_stripped = new_token.lstrip("Ġ").lstrip("▁").strip()
            if new_token_stripped in alignments:
                new_token = new_token_stripped
            else:
                stats["random_init"] += 1
                stats["failed_lookups"].append(new_token)
                continue
        
        cognate_data = alignments[new_token]
        
        
        weighted_embedding = None
        total_weight = 0.0
        
        for cognate_word, weight in cognate_data:
            cognate_embedding = get_token_embedding(
                cognate_word, 
                original_tokenizer, 
                input_embeddings[:original_vocab_size]
            )
            
            if cognate_embedding is not None:
                if weighted_embedding is None:
                    weighted_embedding = weight * cognate_embedding
                else:
                    weighted_embedding = weighted_embedding + weight * cognate_embedding
                total_weight += weight
                
                
                cognate_ids = original_tokenizer.encode(cognate_word, add_special_tokens=False)
                if len(cognate_ids) > 1:
                    stats["multi_token_cognates"] += 1
        
        if weighted_embedding is not None and total_weight > 0:
           
            weighted_embedding = weighted_embedding / total_weight
            
            if not dry_run:
                
                with torch.no_grad():
                    input_embeddings[new_id] = weighted_embedding
                    output_embeddings[new_id] = weighted_embedding
            
            stats["initialized_from_cognates"] += 1
        else:
            stats["random_init"] += 1
            stats["failed_lookups"].append(new_token)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Initialize embeddings for new Occitan tokens via trans-tokenization"
    )
    parser.add_argument(
        "--base-model", "-m",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model to load (default: {DEFAULT_BASE_MODEL})"
    )
    parser.add_argument(
        "--patched-tokenizer", "-t",
        type=Path,
        default=DEFAULT_PATCHED_TOKENIZER,
        help=f"Path to patched tokenizer (default: {DEFAULT_PATCHED_TOKENIZER})"
    )
    parser.add_argument(
        "--alignments", "-a",
        type=Path,
        default=DEFAULT_ALIGNMENTS,
        help=f"Path to alignments JSON (default: {DEFAULT_ALIGNMENTS})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for initialized model (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without modifying embeddings or saving"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)"
    )
    args = parser.parse_args()

    print("TRANS-TOKENIZATION EMBEDDING INITIALIZATION")
    print("-" * 60)
    
    if args.dry_run:
        print("[DRY RUN MODE - No changes will be saved]")
    
    if not args.alignments.exists():
        raise FileNotFoundError(
            f"Alignments file not found: {args.alignments}\n"
            "Run align_occitan_tokens_with_gemini.py first to generate alignments."
        )
    
   
    alignments = load_alignments(args.alignments)
    
   
    print(f"\nLoading original tokenizer from {args.base_model}...")
    original_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    original_vocab_size = len(original_tokenizer)
    print(f"Original vocab size: {original_vocab_size}")
    
   
    print(f"\nLoading patched tokenizer from {args.patched_tokenizer}...")
    patched_tokenizer = AutoTokenizer.from_pretrained(args.patched_tokenizer)
    
    patched_vocab_size = len(patched_tokenizer)
    print(f"Patched vocab size: {patched_vocab_size}")

    ok, failures = tokenizer_roundtrip_sanity(patched_tokenizer)
    if not ok:
        print("\nERROR: Patched tokenizer failed round-trip sanity check.")
        for line in failures:
            print(f"  - {line}")
        raise RuntimeError(
            "Refusing to initialize embeddings with a broken tokenizer. "
            "Rebuild tokenizer with build_occitan_tokenizer.py and retry."
        )
    
   
    print(f"\nLoading model from {args.base_model}...")
    print(f"Device: {args.device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map=args.device if args.device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    print(f"\nResizing embeddings from {original_vocab_size} to {patched_vocab_size}...")
    model.resize_token_embeddings(patched_vocab_size)
    
   
    stats = initialize_new_embeddings(
        model=model,
        original_tokenizer=original_tokenizer,
        patched_tokenizer=patched_tokenizer,
        alignments=alignments,
        dry_run=args.dry_run
    )
    
    
    print("INITIALIZATION STATISTICS")
    print("-" * 60)
    print(f"  Total new tokens:           {stats['total_new_tokens']}")
    print(f"  Initialized from cognates:  {stats['initialized_from_cognates']}")
    print(f"  Random initialization:      {stats['random_init']}")
    print(f"  Multi-token cognates used:  {stats['multi_token_cognates']}")
    
    init_rate = (stats['initialized_from_cognates'] / stats['total_new_tokens'] * 100 
                 if stats['total_new_tokens'] > 0 else 0)
    print(f"  Initialization rate:        {init_rate:.1f}%")
    
    if stats['failed_lookups'] and len(stats['failed_lookups']) <= 20:
        print(f"\n  Tokens with random init:")
        for token in stats['failed_lookups'][:20]:
            print(f"    - {repr(token)}")
    elif stats['failed_lookups']:
        print(f"\n  First 20 tokens with random init:")
        for token in stats['failed_lookups'][:20]:
            print(f"    - {repr(token)}")
        print(f"    ... and {len(stats['failed_lookups']) - 20} more")
    
    
    if not args.dry_run:
        print(f"\nSaving model to {args.output}...")
        args.output.mkdir(parents=True, exist_ok=True)
        
        
        model.save_pretrained(args.output)
        
        
        patched_tokenizer.save_pretrained(args.output)
        
        
        metadata = {
            "base_model": args.base_model,
            "patched_tokenizer": str(args.patched_tokenizer),
            "alignments_file": str(args.alignments),
            "original_vocab_size": original_vocab_size,
            "patched_vocab_size": patched_vocab_size,
            "stats": {
                "total_new_tokens": stats['total_new_tokens'],
                "initialized_from_cognates": stats['initialized_from_cognates'],
                "random_init": stats['random_init'],
                "init_rate_percent": init_rate
            }
        }
        with open(args.output / "trans_tokenization_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("Model saved successfully!")
    
    print("-" * 60)


if __name__ == "__main__":
    main()
