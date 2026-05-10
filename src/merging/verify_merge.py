"""
Smoke-test the TIES-merged Occitan initialization adapter.

Example:
    python -m src.merging.verify_merge --base-model models/llama-3.1-occitan-initialized --adapter-path models/adapter_oc_init --device cuda
"""

import argparse
import math
import torch
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE_MODEL = "models/llama-3.1-occitan-initialized"
DEFAULT_ADAPTER_PATH = "models/adapter_oc_init"


TEST_PROMPTS = [
    ("French", "Bonjour, je m'appelle"),
    ("Catalan", "Bon dia, em dic"),
    ("Occitan", "Adieu, me soi"),
]


def load_merged_model(base_model: str, adapter_path: str, device: str):
    """Load base model with merged adapter."""
    print("=" * 60)
    print("MERGE VERIFICATION")
    print("=" * 60)
    
    print(f"\n1. Tokenizer: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer_vocab = len(tokenizer)
    print(f"   Vocab size: {tokenizer_vocab:,}")
    
    from safetensors.torch import load_file
    adapter_weights = load_file(Path(adapter_path) / "adapter_model.safetensors")
    
    target_vocab = None
    for key in adapter_weights.keys():
        if "lm_head.lora_B" in key or "lm_head.base_layer" in key:
            target_vocab = adapter_weights[key].shape[0]
            break
            
    if target_vocab is None:
        target_vocab = math.ceil(len(tokenizer) / 64) * 64
        
    print(f"   Target vocab size: {target_vocab:,}")
    del adapter_weights  # Free memory
    
    print(f"\n2. Base model: {base_model}")
    print("   (CPU initially)")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        tie_word_embeddings=False,
    )
    
    if model.config.vocab_size != target_vocab:
        print(f"   Resize embeddings: {model.config.vocab_size} -> {target_vocab}")
        model.resize_token_embeddings(target_vocab, mean_resizing=False)
    
    print(f"\n3. Merged adapter: {adapter_path}")
    model = PeftModel.from_pretrained(
        model, 
        adapter_path,
        device_map="cpu",
    )
    print("Adapter loaded")
    
    print(f"\n4. Move to {device}")
    model = model.to(device)
    print(f"   Model on {device}")
    
    return model, tokenizer


def check_adapter_weights(model):
    """Verify adapter weights are non-zero and reasonable."""
    print("\n5. Adapter stats")
    
    lora_stats = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            abs_mean = param.data.abs().mean().item()
            abs_max = param.data.abs().max().item()
            nonzero_pct = (param.data != 0).float().mean().item() * 100
            
            layer_type = name.split(".")[-2]  # e.g., "lora_A" or "lora_B"
            if layer_type not in lora_stats:
                lora_stats[layer_type] = []
            lora_stats[layer_type].append({
                "name": name,
                "mean": abs_mean,
                "max": abs_max,
                "nonzero": nonzero_pct,
            })
    
    for layer_type, stats in lora_stats.items():
        avg_mean = sum(s["mean"] for s in stats) / len(stats)
        avg_nonzero = sum(s["nonzero"] for s in stats) / len(stats)
        print(f"   {layer_type}: avg|value|={avg_mean:.6f}, avg_nonzero={avg_nonzero:.1f}%")
    
    total_params = sum(len(s) for s in lora_stats.values())
    dead_params = sum(1 for stats in lora_stats.values() for s in stats if s["mean"] < 1e-8)
    
    if dead_params > 0:
        print(f"    Warning: {dead_params}/{total_params} layers have near-zero weights")
    else:
        print(f"    All {total_params} LoRA layers have non-zero weights")
    
    return dead_params == 0


def test_generation(model, tokenizer):
    """Test text generation with the merged model."""
    print("\n6. Text generation")
    
    model.eval()
    results = []
    
    for lang, prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n   [{lang}]")
        print(f"   Prompt: {prompt}")
        print(f"   Output: {generated}")
        results.append((lang, generated))
    
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a TIES-merged Occitan adapter.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_merged_model(args.base_model, args.adapter_path, args.device)
    
    weights_ok = check_adapter_weights(model)
    
    results = test_generation(model, tokenizer)
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f" Model loaded successfully")
    print(f"{'Yes' if weights_ok else 'No '} Adapter weights: {'OK' if weights_ok else 'Some dead layers'}")
    print(f" Generation: {len(results)} prompts completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
