"""
Compare tokenization behavior across stock, Carballo, and patched tokenizers.

Example:
    python -m src.evaluation.tokenizer_comparative_test --patched models/llama-3.1-occitan-initialized
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_STOCK = "meta-llama/Meta-Llama-3.1-8B"
DEFAULT_STOCK_FALLBACK = "unsloth/Meta-Llama-3.1-8B"
DEFAULT_CARBALLO = "proxectonos/Llama-3.1-Carballo"
DEFAULT_PATCHED = "models/llama-3.1-occitan-initialized"

TEST_PHRASES = [
    "d'aquestas",
    "d’aquestas",
    "l'òme",
    "qu'ei",
    "s'es",
    "Occitània",
    "automàtic",
    "El cel és blau perquè la llum és desviada per l'atmosfera.",
    "totas las comunas d'Occitània",
]


@dataclass
class TokenizerSpec:
    name: str
    ref: str
    fallback: str | None = None
    local_only: bool = False


def load_tokenizer(spec: TokenizerSpec):
    if spec.local_only:
        ref_path = Path(spec.ref)
        if not ref_path.exists():
            raise FileNotFoundError(f"Local tokenizer path does not exist: {spec.ref}")
        tok = AutoTokenizer.from_pretrained(str(ref_path), local_files_only=True)
        return tok, str(ref_path)

    try:
        tok = AutoTokenizer.from_pretrained(spec.ref)
        return tok, spec.ref
    except OSError:
        if spec.fallback:
            tok = AutoTokenizer.from_pretrained(spec.fallback)
            return tok, spec.fallback
        raise


def inspect(tokenizer, text: str) -> dict:
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids)
    word_count = max(1, len(text.split()))
    fertility = len(tokens) / word_count
    return {
        "tokens": tokens,
        "ids": ids,
        "decoded": decoded,
        "fertility": fertility,
        "roundtrip_ok": decoded == text,
    }


def has_bytelevel_artifacts(tokens: list[str]) -> bool:
    # ByteLevel BPE often represents accented bytes as tokens like "Ã²", "Ã©", etc.
    return any(("Ã" in tok or "â" in tok) for tok in tokens)


def print_model_header(name: str, source: str, tok):
    print("=" * 90)
    print(f"{name}")
    print(f"Source: {source}")
    print(f"Vocab size: {len(tok):,}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Compare stock, Carballo, and patched tokenizers")
    parser.add_argument("--stock", default=DEFAULT_STOCK, help="Stock Llama 3.1 tokenizer ref")
    parser.add_argument(
        "--stock_fallback",
        default=DEFAULT_STOCK_FALLBACK,
        help="Fallback stock tokenizer ref if --stock cannot be loaded",
    )
    parser.add_argument("--carballo", default=DEFAULT_CARBALLO, help="Carballo tokenizer ref")
    parser.add_argument("--patched", default=DEFAULT_PATCHED, help="Patched tokenizer path/ref")
    args = parser.parse_args()

    specs = [
        TokenizerSpec("Stock Llama 3.1", args.stock, args.stock_fallback),
        TokenizerSpec("Llama 3.1 Carballo", args.carballo),
        TokenizerSpec("Patched Occitan Tokenizer", args.patched, local_only=True),
    ]

    tokenizers = []
    for spec in specs:
        print(f"Loading {spec.name} from {spec.ref}...")
        tok, source = load_tokenizer(spec)
        tokenizers.append((spec.name, source, tok))
    print()

    for model_name, source, tok in tokenizers:
        print_model_header(model_name, source, tok)
        mismatches = 0
        for phrase in TEST_PHRASES:
            out = inspect(tok, phrase)
            status = "OK" if out["roundtrip_ok"] else "MISMATCH"
            if not out["roundtrip_ok"]:
                mismatches += 1
            frag_flag = "HIGH_FRAGMENTATION" if out["fertility"] > 2.0 else "normal"
            bytelevel_note = (
                "bytelevel tokens expected"
                if out["roundtrip_ok"] and has_bytelevel_artifacts(out["tokens"])
                else "none"
            )

            print(f"\nInput:    {phrase!r}")
            print(f"Tokens:   {out['tokens']}")
            print(f"IDs:      {out['ids']}")
            print(f"Decoded:  {out['decoded']!r} [{status}]")
            print(f"Fertility:{out['fertility']:.2f} tokens/word ({frag_flag})")
            print(f"ByteLevel artifact note: {bytelevel_note}")

        print("\n" + "-" * 90)
        print(f"Round-trip mismatches for {model_name}: {mismatches}/{len(TEST_PHRASES)}")
        print("-" * 90 + "\n")

    print("#" * 90)
    print("COMPACT CROSS-MODEL SUMMARY")
    print("#" * 90)
    for phrase in TEST_PHRASES:
        print(f"\nPhrase: {phrase!r}")
        for model_name, _, tok in tokenizers:
            out = inspect(tok, phrase)
            if out["roundtrip_ok"]:
                mark = "OK(bytelevel)" if has_bytelevel_artifacts(out["tokens"]) else "OK"
            else:
                mark = "BAD"
            print(
                f"  - {model_name:<26} | "
                f"tokens={len(out['tokens']):<3} | "
                f"fertility={out['fertility']:.2f} | "
                f"roundtrip={mark}"
            )


if __name__ == "__main__":
    main()
