"""
Reweight token alignments using character n-gram similarity.

Example:
    python -m src.tokenization.weight_token_alignments --input data/alignments_gemini.json
"""

import argparse
import json
import shutil
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALIGNMENTS = PROJECT_ROOT / "data" / "alignments_gemini.json"


def update_ratios(alignments_file: Path):
    if not alignments_file.exists():
        print(f"Error: {alignments_file} not found.")
        return False

    print(f"Loading {alignments_file} ...")
    with open(alignments_file, "r", encoding="utf-8") as f:
        alignments = json.load(f)

    print(f"Found {len(alignments)} entries. Updating ratios...")
    
    updated_count = 0
    for occ_word, langs in alignments.items():
        if len(langs) == 2:
            fr_word = langs[0][0]
            ca_word = langs[1][0]
            
            sim_fr = char_ngram_cosine_similarity(occ_word, fr_word)
            sim_ca = char_ngram_cosine_similarity(occ_word, ca_word)
            
            eps = 0.01
            weight_fr = sim_fr + eps
            weight_ca = sim_ca + eps
            total = weight_fr + weight_ca
            
            alignments[occ_word][0][1] = round(weight_fr / total, 3)
            alignments[occ_word][1][1] = round(weight_ca / total, 3)
            updated_count += 1

    backup = alignments_file.with_suffix(".json.bak")
    shutil.copy2(alignments_file, backup)
    print(f"Created backup at {backup}")

    with open(alignments_file, "w", encoding="utf-8") as f:
        json.dump(alignments, f, indent=2, ensure_ascii=False)

    print(f"Successfully updated {updated_count} entries with cosine similarity ratios.")
    print(f"Saved to {alignments_file}")
    
    print("\nExamples of new ratios:")
    examples = list(alignments.items())[:5]
    for occ_word, langs in examples:
        fr_w, fr_r = langs[0]
        ca_w, ca_r = langs[1]
        print(f"  {occ_word!r:<12} -> FR: {fr_w!r:<12} ({fr_r:.3f}) | CA: {ca_w!r:<12} ({ca_r:.3f})")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update alignment ratios using cosine similarity"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_ALIGNMENTS,
        help=f"Path to alignments JSON (default: {DEFAULT_ALIGNMENTS})"
    )
    args = parser.parse_args()
    
    update_ratios(args.input)
