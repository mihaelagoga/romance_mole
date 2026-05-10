"""
Run MoLE smoke tests and print routing probabilities.

Example:
    python -m src.evaluation.inspect_mole_routing --router_dir checkpoints/router_ablation_ladder/t8_aux001_sup002_cs000/final --base_model models/llama-3.1-occitan-initialized --adapters checkpoints/lora_fr/final checkpoints/lora_ca/final_compat checkpoints/occitan_3b2/final_compat --prompt_suite default --max_new_tokens 80
"""
import argparse
import json
import sys
import re
import inspect
import shutil
import tempfile
import torch
from pathlib import Path
from peft import LoraConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import TextStreamer
from src.models.mole.romance_mole import RomanceMoLEModel, _LAYER_INDEX_RE
from src.training.train_romance_mole_router import _load_base_model, _load_tokenizer

DEFAULT_PROMPT_SUITE = [
    (
        "fr",
        "French",
        "Tu es un assistant virtuel. Raconte-moi ta routine matinale avant d'aller au travail. "
        "Utilise des verbes pronominaux.",
    ),
    (
        "ca",
        "Catalan",
        "Ets un assistent virtual. Explica la teva rutina del mati abans d'anar a la feina. "
        "Utilitza verbs pronominals.",
    ),
    (
        "oc",
        "Occitan",
        "Sès un assistent virtual. Conta-me ta rutina del matin abans d'anar al trabalh. "
        "Utiliza de vèrbs pronominals.",
    ),
    (
        "code_switch",
        "Code-Switch",
        "Escriu un pichon paragraf majoritàriament en francés, mas inserís naturalament una corta "
        "expression occitana e una autra catalana.",
    ),
]


def _resolve_path(p: str) -> str:
    """Resolve local path relative to PROJECT_ROOT if not absolute."""
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)

def _resolve_model_ref(model_ref: str) -> str:
    """Resolve a base-model reference."""
    path = Path(model_ref)
    if path.is_absolute():
        return str(path)
    project_candidate = PROJECT_ROOT / path
    if project_candidate.exists():
        return str(project_candidate)
    return model_ref


def _supported_lora_keys() -> set[str]:
    keys = set(inspect.signature(LoraConfig.__init__).parameters.keys())
    keys.discard("self")
    return keys


def _metadata_keys() -> set[str]:
    return {
        "peft_type",
        "auto_mapping",
        "base_model_name_or_path",
        "revision",
        "task_type",
        "inference_mode",
    }


def _sanitize_adapter_runtime_copy(adapter_dir: Path) -> tuple[Path, bool]:
    """
    Create a temporary PEFT-compatible adapter copy when adapter_config.json
    contains keys unsupported by the local PEFT LoraConfig schema.
    Returns (path_to_use, is_temporary_copy).
    """
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    keep_keys = _supported_lora_keys() | _metadata_keys()
    cleaned = {k: v for k, v in cfg.items() if k in keep_keys}
    dropped = sorted(set(cfg.keys()) - set(cleaned.keys()))

    if not dropped:
        return adapter_dir, False

    tmp_dir = Path(tempfile.mkdtemp(prefix="peft_runtime_compat_"))
    with open(tmp_dir / "adapter_config.json", "w", encoding="utf-8") as handle:
        json.dump(cleaned, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    copied_any = False
    for fname in ("adapter_model.safetensors", "adapter_model.bin", "tokenizer.json", "tokenizer_config.json"):
        src = adapter_dir / fname
        if src.exists():
            shutil.copy2(src, tmp_dir / fname)
            copied_any = True

    if not copied_any:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")

    print(
        f"[smoke_test_mole] Runtime adapter compat copy: {adapter_dir} -> {tmp_dir} "
        f"(dropped keys: {dropped})"
    )
    return tmp_dir, True


def _register_router_bias_hooks(model, expert_index: int, bias_value: float):
    """Inject a positive logit bias for one expert on all router layers."""
    if not hasattr(model, "routers"):
        return []

    def bias_hook(module, args, output):
        if isinstance(output, torch.Tensor) and output.shape[-1] > expert_index:
            hacked = output.clone()
            hacked[..., expert_index] += bias_value
            return hacked
        return output

    handles = []
    for router in model.routers.values():
        handles.append(router.register_forward_hook(bias_hook))
    return handles


def _get_expert_index(adapter_names: list[str], expert_name: str, fallback_index: int) -> int:
    """Resolve an expert index by name, with a positional fallback."""
    if expert_name in adapter_names:
        return adapter_names.index(expert_name)
    if fallback_index < len(adapter_names):
        return fallback_index
    raise ValueError(
        f"Could not resolve expert '{expert_name}' in adapters {adapter_names}; "
        f"fallback index {fallback_index} is out of range."
    )


def _normalize_cli_prompt(text: str) -> str:
    """Interpret common escaped newlines from shell-passed prompts."""
    normalized = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return normalized.replace("\r\n", "\n")


def _looks_like_alpaca_prompt(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("### Instruction:") and "### Response:" in stripped


def _build_prompt_text(text: str, raw_prompt: bool) -> tuple[str, str]:
    normalized = _normalize_cli_prompt(text)
    if raw_prompt:
        return normalized, "Raw"
    if _looks_like_alpaca_prompt(normalized):
        return normalized, "Detected Alpaca"
    prompt_text = (
        f"### Instruction:\n{normalized}\n\n"
        "### Response:\n"
    )
    return prompt_text, "Alpaca"


def _iter_prompt_cases(args) -> list[tuple[str, str]]:
    if args.prompt_suite == "default":
        return [(label, prompt) for _, label, prompt in DEFAULT_PROMPT_SUITE]
    return [("Custom", args.prompt)]


def main(args):
    args.router_dir = _resolve_path(args.router_dir)
    args.adapters = [_resolve_path(a) for a in args.adapters]
    args.base_model = _resolve_model_ref(args.base_model)

    temp_adapter_dirs = []
    runtime_adapters = []
    for adapter_path in args.adapters:
        compat_path, is_temp = _sanitize_adapter_runtime_copy(Path(adapter_path))
        runtime_adapters.append(str(compat_path))
        if is_temp:
            temp_adapter_dirs.append(compat_path)

    try:
        print("Load router config")
        router_config_path = Path(args.router_dir) / "router_config.json"
        if not router_config_path.exists():
            print(f"Error: Could not find router config at {router_config_path}")
            print("Make sure you've provided the correct --router_dir")
            return

        with open(router_config_path, "r") as f:
            router_cfg = json.load(f)

        adapter_names = router_cfg["adapter_names"]
        router_temperature = (
            args.router_temperature
            if args.router_temperature is not None
            else float(router_cfg.get("router_temperature", 1.0))
        )
        hard_router_argmax = (
            args.hard_router_argmax
            if args.hard_router_argmax is not None
            else bool(router_cfg.get("hard_router_argmax", False))
        )
        print(f"Router expects adapters: {adapter_names}")
        
        if len(runtime_adapters) != len(adapter_names):
            raise ValueError(f"Expected {len(adapter_names)} adapters, got {len(runtime_adapters)} (--adapters)")

        print("\nLoad tokenizer")
        tokenizer = _load_tokenizer(runtime_adapters, args.base_model)

        print("\nLoad base model")
        base_model = _load_base_model(args.base_model, runtime_adapters, tokenizer)

        print("\nBuild RomanceMoLEModel")
        print(
            f"Router sharpening: temperature={router_temperature}, "
            f"hard_argmax={hard_router_argmax}"
        )
        mole = RomanceMoLEModel(
            base_model=base_model,
            adapter_paths=runtime_adapters,
            adapter_names=adapter_names,
            sequence_route_threshold=router_cfg.get("sequence_route_threshold"),
            router_aux_loss_coef=0.0, # Disable aux loss for inference
            router_temperature=router_temperature,
            hard_router_argmax=hard_router_argmax,
        )
        
        del base_model # Free reference to save memory
        
        print("\nLoad router weights")
        router_weights_path = Path(args.router_dir) / "router_weights.pt"
        if not router_weights_path.exists():
            print(f"Error: Could not find router weights at {router_weights_path}")
            return

        mole.routers.load_state_dict(torch.load(router_weights_path, map_location="cpu"))
        
        print("Model to CUDA (bfloat16)")
        mole = mole.to(torch.bfloat16).to("cuda")
        mole.eval()

        hook_handles = []
        bias_specs = [
            ("fr", "French", 0, args.fr_bias),
            ("ca", "Catalan", 1, args.ca_bias),
            ("oc", "Occitan", 2, args.oc_bias),
        ]
        for expert_name, expert_label, fallback_index, bias_value in bias_specs:
            if bias_value <= 0.0:
                continue
            expert_index = _get_expert_index(adapter_names, expert_name, fallback_index)
            print(
                f"Injecting +{bias_value} router logit bias to expert index "
                f"{expert_index} ({expert_label})."
            )
            hook_handles.extend(
                _register_router_bias_hooks(
                    model=mole,
                    expert_index=expert_index,
                    bias_value=bias_value,
                )
            )

        def get_layer_idx(key):
            orig_name = key.replace("__", ".")
            match = _LAYER_INDEX_RE.search(orig_name)
            return int(match.group(1)) if match else -1

        def print_routing_table(gen_seq, router_key, is_sequence=False):
            full_gates = mole.last_gates.get(router_key)
            if full_gates is None:
                print(f"No gates cached for {router_key}")
                return
                
            full_gates = full_gates[0] # [Seq_len, Num_Experts]
            mode_str = "Sequence-level" if is_sequence else "Token-level"
            print(f"\nProbabilities: Layer {router_key.replace('__', '.')} ({mode_str})")
            header = f"{'Token':<25} | " + " | ".join([f"{name:>10}" for name in adapter_names])
            print(header)
            print("-" * len(header))
            
            decoded_so_far = ""
            for i, token_id in enumerate(gen_seq):
                current_decoded = tokenizer.decode(gen_seq[:i+1])
                if current_decoded.endswith("\ufffd"):
                    token_str = current_decoded[len(decoded_so_far):-1] + "<byte>"
                else:
                    token_str = current_decoded[len(decoded_so_far):]
                    decoded_so_far = current_decoded
                    
                token_str = repr(token_str).strip("'\"")
                if len(token_str) > 23:
                    token_str = token_str[:20] + "..."
                    
                probs = full_gates[i]
                probs_str = " | ".join([f"{p.item():>10.4f}" for p in probs])
                print(f"{token_str:<25} | {probs_str}")

        def run_case(case_label: str, case_prompt: str):
            prompt_text, prompt_mode = _build_prompt_text(case_prompt, args.raw_prompt)
            print("\n" + "=" * 100)
            print(f"Smoke Test Case: {case_label}")
            print("=" * 100)
            print(f"\n[Prompt ({prompt_mode})]: {prompt_text}")

            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

            print("\nGenerate")
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            with torch.no_grad():
                outputs = mole.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                )

            gen_seq = outputs[0]
            print("\n")

            print("-" * 80)
            print("Token-by-Token Routing Probabilities")
            print("-" * 80)

            print("\nForward pass to extract gates")
            with torch.no_grad():
                _ = mole(gen_seq.unsqueeze(0))

            token_routers = [k for k, v in mole.router_modes.items() if v == "token"]
            if token_routers:
                last_token_router_key = sorted(token_routers, key=get_layer_idx)[-1]
                print_routing_table(gen_seq, last_token_router_key, is_sequence=False)
            else:
                print("No token-level routers found.")

            seq_routers = [k for k, v in mole.router_modes.items() if v == "sequence"]
            if seq_routers:
                first_seq_router_key = sorted(seq_routers, key=get_layer_idx)[0]
                print_routing_table(gen_seq, first_seq_router_key, is_sequence=True)
            else:
                print("No sequence-level routers found.")

        for case_label, case_prompt in _iter_prompt_cases(args):
            run_case(case_label, case_prompt)

        for h in hook_handles:
            h.remove()
    finally:
        for tmp in temp_adapter_dirs:
            shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--router_dir", default="checkpoints/mole_router_ddp/final")
    
    parser.add_argument("--adapters", nargs="+", default=[
        "checkpoints/lora_fr/final_compat",
        "checkpoints/lora_ca/final_compat",
        "checkpoints/occitan_3b2/final"
    ])
    
    parser.add_argument(
        "--base_model",
        default="models/llama-3.1-occitan-initialized",
        help="Base LM local path or Hugging Face model id.",
    )
    parser.add_argument("--prompt", type=str, default="Bonjorn, cossí anatz uèi? Es un bèl jorn per parlar occitan.")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Generation repetition penalty (1.0 disables penalty).",
    )
    parser.add_argument(
        "--router_temperature",
        type=float,
        default=None,
        help="Temperature applied to router logits at inference (<1 sharpens). Defaults to router_config.json.",
    )
    parser.add_argument(
        "--hard_router_argmax",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force one-hot (argmax) routing at inference. Defaults to router_config.json.",
    )
    parser.add_argument(
        "--raw_prompt",
        action="store_true",
        help="Skip wrapping the prompt in the Alpaca instruction template.",
    )
    parser.add_argument(
        "--prompt_suite",
        choices=("none", "default"),
        default="none",
        help="Run a built-in prompt suite (`default` = French, Catalan, Occitan, code-switch).",
    )
    parser.add_argument(
        "--fr_bias",
        type=float,
        default=0.0,
        help="Add positive router logit bias to French expert (default: 0.0 = disabled).",
    )
    parser.add_argument(
        "--ca_bias",
        type=float,
        default=0.0,
        help="Add positive router logit bias to Catalan expert (default: 0.0 = disabled).",
    )
    parser.add_argument(
        "--oc_bias",
        type=float,
        default=0.0,
        help="Add positive router logit bias to Occitan expert (default: 0.0 = disabled).",
    )
    args = parser.parse_args()
    main(args)
