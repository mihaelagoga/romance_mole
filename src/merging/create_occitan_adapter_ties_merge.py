"""
Create an Occitan initialization adapter by TIES-merging French and Catalan LoRAs.

Example:
    python -m src.merging.create_occitan_adapter_ties_merge --base-model models/llama-3.1-occitan-initialized --adapter-fr checkpoints/lora_fr/final --adapter-ca checkpoints/lora_ca/final --output-dir models/adapter_oc_init
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import torch

DEFAULT_BASE_MODEL = "models/llama-3.1-occitan-initialized"
DEFAULT_ADAPTER_FR = "checkpoints/lora_fr/final"
DEFAULT_ADAPTER_CA = "checkpoints/lora_ca/final"
DEFAULT_OUTPUT_DIR = "models/adapter_oc_init"

def _load_mergekit_ties_symbols():
    """Import mergekit's TIES symbols, with local-repo fallback."""
    try:
        from mergekit.merge_methods.generalized_task_arithmetic import get_mask
        from mergekit.sparsify import SparsificationMethod, sparsify
        return get_mask, SparsificationMethod, sparsify
    except ModuleNotFoundError as e:
        project_root = Path(__file__).resolve().parents[2]
        local_mergekit_repo = project_root / "mergekit"
        if local_mergekit_repo.exists():
            sys.path.insert(0, str(local_mergekit_repo))
            try:
                from mergekit.merge_methods.generalized_task_arithmetic import get_mask
                from mergekit.sparsify import SparsificationMethod, sparsify
                return get_mask, SparsificationMethod, sparsify
            except ModuleNotFoundError as e2:
                raise ModuleNotFoundError(
                    f"{e2}. When using the local mergekit repo, install its dependencies, "
                    "e.g.: pip install immutables"
                ) from e2
        raise


GET_MASK, SPARSIFICATION_METHOD, SPARSIFY = _load_mergekit_ties_symbols()


def build_ties_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Standard mergekit-like TIES config with exposed hyperparameters."""
    return {
        "merge_method": "ties",
        "base_model": args.base_model,
        "models": [
            {
                "model": args.adapter_fr,
                "parameters": {
                    "density": args.density_fr,
                    "weight": args.weight_fr,
                },
            },
            {
                "model": args.adapter_ca,
                "parameters": {
                    "density": args.density_ca,
                    "weight": args.weight_ca,
                },
            },
        ],
        "parameters": {
            "normalize": args.normalize,
            "int8_mask": args.int8_mask,
            "lambda": args.scale,
        },
    }


def load_peft_state_dict(adapter_path):
    """
    Manually loads the LoRA state dict.
   It is done manually to perform arithmetic on the tensors directly.
    """
    print(f"Adapter: {adapter_path}")
    try:
        from safetensors.torch import load_file
        state_dict = load_file(Path(adapter_path) / "adapter_model.safetensors")
    except Exception:
        state_dict = torch.load(Path(adapter_path) / "adapter_model.bin", map_location="cpu")
    return state_dict


def ties_merge_with_mergekit(state_dict_fr, state_dict_ca, config: Dict[str, Any]):
    """
    Merge two LoRA adapter state dicts with mergekit's TIES implementation.
    We treat LoRA adapter tensors as task vectors relative to a zero base.
    """
    logging.info("Merging adapters with mergekit TIES...")
    merged_state_dict = {}

    fr_params = config["models"][0]["parameters"]
    ca_params = config["models"][1]["parameters"]
    global_params = config["parameters"]

    keys = list(state_dict_fr.keys())
    for key in keys:
        tensor_fr = state_dict_fr[key].float()
        tensor_ca = state_dict_ca[key].float()

        # Embeddings and lm_head (full tensors, not LoRA A/B) must not be sparsified or vocabulary is corrupted.
        is_embed_or_head = (
            ("embed_tokens" in key or "lm_head" in key)
            and "lora_A" not in key
            and "lora_B" not in key
        )
        density_fr = 1.0 if is_embed_or_head else fr_params["density"]
        density_ca = 1.0 if is_embed_or_head else ca_params["density"]

        delta_fr = SPARSIFY(
            tensor_fr,
            density=density_fr,
            method=SPARSIFICATION_METHOD.magnitude,
            rescale_norm=None,
        )
        delta_ca = SPARSIFY(
            tensor_ca,
            density=density_ca,
            method=SPARSIFICATION_METHOD.magnitude,
            rescale_norm=None,
        )

        deltas = torch.stack([delta_fr, delta_ca], dim=0)
        weights = torch.tensor(
            [fr_params["weight"], ca_params["weight"]],
            dtype=deltas.dtype,
            device=deltas.device,
        )
        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)
        weighted_deltas = deltas * weights

        mask_dtype = torch.int8 if global_params["int8_mask"] else deltas.dtype
        mask = GET_MASK(weighted_deltas, method="sum", mask_dtype=mask_dtype)
        mixed_delta = (weighted_deltas * mask).sum(dim=0)

        if global_params["normalize"]:
            divisor = (weights * mask).sum(dim=0)
            divisor[divisor == 0] = 1
            mixed_delta /= divisor

        if global_params["lambda"] != 1:
            mixed_delta *= global_params["lambda"]

        merged_state_dict[key] = mixed_delta.to(state_dict_fr[key].dtype)

    return merged_state_dict


def check_and_tie_embeddings(state_dict, drift_threshold=0.1):
    """Check if input and output embeddings have drifted significantly."""
    embed_key = None
    head_key = None
    
    for key in state_dict.keys():
        if "embed_tokens" in key and "lora_A" not in key and "lora_B" not in key:
            embed_key = key
        if "lm_head" in key and "lora_A" not in key and "lora_B" not in key:
            head_key = key
    
    if embed_key is None or head_key is None:
        print("  Missing embed/head keys for tying check")
        return state_dict
    
    embed_weights = state_dict[embed_key]
    head_weights = state_dict[head_key]
    
    if embed_weights.shape != head_weights.shape:
        if embed_weights.shape == head_weights.T.shape:
            head_weights = head_weights.T
        else:
            print(f"  Shape mismatch: embed {embed_weights.shape} vs head {head_weights.shape}")
            return state_dict
    
    embed_flat = embed_weights.flatten().float()
    head_flat = head_weights.flatten().float()
    
    cosine_sim = torch.nn.functional.cosine_similarity(
        embed_flat.unsqueeze(0), head_flat.unsqueeze(0)
    ).item()
    drift = 1 - cosine_sim
    
    print(f"  Drift: {drift:.4f} (thresh: {drift_threshold})")
    
    if drift > drift_threshold:
        print("  Tying embeddings (drift > thresh)")
        tied = (embed_weights + head_weights) / 2
        state_dict[embed_key] = tied.clone()
        state_dict[head_key] = tied.clone()
        print("  Embeddings tied")
    else:
        print("  Embeddings stable")
    
    return state_dict


def clean_weight_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove PEFT internal artifacts from key names so the adapter loads correctly
    when re-attached. Strips .modules_to_save and .original_module from keys.
    """
    def key_priority(raw_key: str) -> int:
        """Collision precedence after key cleaning:."""
        if ".modules_to_save" in raw_key:
            return 3
        if ".original_module" in raw_key:
            return 1
        return 2

    cleaned: Dict[str, torch.Tensor] = {}
    chosen_source: Dict[str, str] = {}
    for key, value in state_dict.items():
        new_key = key.replace(".modules_to_save", "").replace(".original_module", "")
        if new_key not in cleaned:
            cleaned[new_key] = value
            chosen_source[new_key] = key
            continue

        prev_key = chosen_source[new_key]
        if key_priority(key) > key_priority(prev_key):
            logging.warning(
                "Key collision for %s: replacing %s with %s",
                new_key,
                prev_key,
                key,
            )
            cleaned[new_key] = value
            chosen_source[new_key] = key
        else:
            logging.warning(
                "Key collision for %s: keeping %s, dropping %s",
                new_key,
                prev_key,
                key,
            )
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TIES-merge French and Catalan LoRA adapters.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-fr", default=DEFAULT_ADAPTER_FR)
    parser.add_argument("--adapter-ca", default=DEFAULT_ADAPTER_CA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--density-fr", type=float, default=0.2, help="TIES density for the French adapter.")
    parser.add_argument("--density-ca", type=float, default=0.2, help="TIES density for the Catalan adapter.")
    parser.add_argument("--weight-fr", type=float, default=0.3, help="Merge weight for the French adapter.")
    parser.add_argument("--weight-ca", type=float, default=0.7, help="Merge weight for the Catalan adapter.")
    parser.add_argument("--scale", type=float, default=1.0, help="Global lambda/scale for the merged task vector.")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable TIES normalization.")
    parser.add_argument("--no-int8-mask", dest="int8_mask", action="store_false", help="Disable int8 mask storage.")
    parser.set_defaults(normalize=True, int8_mask=True)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("TIES-MERGING ADAPTERS")
    logging.info("-" * 30)

    ties_config = build_ties_config(args)
    logging.info("Merge parameters: %s", json.dumps(ties_config, indent=2))

    sd_fr = load_peft_state_dict(args.adapter_fr)
    sd_ca = load_peft_state_dict(args.adapter_ca)

    merged_sd = ties_merge_with_mergekit(sd_fr, sd_ca, ties_config)

    print("\nEmbedding stability check")
    merged_sd = check_and_tie_embeddings(merged_sd)

    merged_sd = clean_weight_keys(merged_sd)

    print(f"Save adapter: {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file
    save_file(merged_sd, Path(args.output_dir) / "adapter_model.safetensors")

    LORA_CONFIG_WHITELIST = frozenset({
        "peft_type", "auto_mapping", "base_model_name_or_path", "revision",
        "task_type", "inference_mode", "r", "lora_alpha", "target_modules",
        "lora_dropout", "bias", "modules_to_save", "init_lora_weights",
        "layers_to_transform", "layers_pattern", "fan_in_fan_out",
        "use_rslora", "use_dora", "loftq_config",
    })
    with open(Path(args.adapter_fr) / "adapter_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    config = {k: v for k, v in config.items() if k in LORA_CONFIG_WHITELIST}

    with open(Path(args.output_dir) / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json", 
        "special_tokens_map.json",
        "tokenizer.model",  # For sentencepiece-based tokenizers
    ]
    for fname in tokenizer_files:
        src = Path(args.adapter_fr) / fname
        if src.exists():
            shutil.copy(src, Path(args.output_dir) / fname)
            print(f"  Copied {fname}")

    with open(Path(args.output_dir) / "merge_trace.json", "w", encoding="utf-8") as f:
        json.dump(ties_config, f, indent=2)
    logging.info("Saved merge trace to %s", Path(args.output_dir) / "merge_trace.json")
    print("Done")

if __name__ == "__main__":
    main()
