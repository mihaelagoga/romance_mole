"""
Run a quick generation smoke test for the French expert adapter.

Example:
    python -m src.evaluation.smoke_test_french_expert --adapter checkpoints/lora_fr/final_compat --max_new_tokens 80
"""

import argparse
import inspect
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_BASE_MODEL = "models/llama-3.1-occitan-initialized"
DEFAULT_FR_ADAPTER = "checkpoints/lora_fr/final"

FR_PROMPTS = [
    "Explique brievement pourquoi le ciel est bleu.",
    "Ecris une courte reponse polie a un email professionnel en francais.",
    "Raconte-moi ta routine matinale avant d'aller au travail. Utilise des verbes pronominaux.",
    "Resume en 3 phrases le concept d'apprentissage automatique.",
]

FR_MARKERS = {
    "le",
    "la",
    "les",
    "des",
    "une",
    "dans",
    "avec",
    "pour",
    "est",
    "etre",
    "vous",
    "bonjour",
}
CA_MARKERS = {
    "el",
    "els",
    "una",
    "amb",
    "per",
    "es",
    "aixo",
    "mati",
    "feina",
}
OC_MARKERS = {
    "lo",
    "los",
    "una",
    "amb",
    "per",
    "es",
    "bonjorn",
    "trabalh",
    "matin",
}

ENCODING_PROBE = "Encoding probe: a e e i o o u u n c (francais, reponse, etrange) -- replacement char: \ufffd"


def _resolve_path(p: str) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _resolve_model_ref(model_ref: str) -> str:
    path = Path(model_ref)
    if path.is_absolute():
        return str(path)
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return str(candidate)
    return model_ref


def _format_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def _tokenize_for_markers(text: str) -> list[str]:
    return re.findall(r"\b[\wÀ-ÿ']+\b", text.lower())


def _lexical_probe(text: str) -> tuple[int, int, int]:
    toks = _tokenize_for_markers(text)
    fr_hits = sum(1 for tok in toks if tok in FR_MARKERS)
    ca_hits = sum(1 for tok in toks if tok in CA_MARKERS)
    oc_hits = sum(1 for tok in toks if tok in OC_MARKERS)
    return fr_hits, ca_hits, oc_hits


def _generate(model, tokenizer, instruction: str, max_new_tokens: int, greedy: bool, temperature: float, top_p: float) -> str:
    prompt = _format_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = out[0][prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


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


def _build_runtime_compat_copy(adapter_path: str) -> Path:
    src = Path(adapter_path)
    cfg_src = src / "adapter_config.json"
    if not cfg_src.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {src}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="peft_runtime_compat_"))

    with open(cfg_src, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    keep = _metadata_keys() | _supported_lora_keys()
    cleaned = {k: v for k, v in cfg.items() if k in keep}
    with open(tmp_dir / "adapter_config.json", "w", encoding="utf-8") as handle:
        json.dump(cleaned, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    st_path = src / "adapter_model.safetensors"
    bin_path = src / "adapter_model.bin"
    if st_path.exists():
        weights = load_file(st_path)
        save_file(weights, str(tmp_dir / "adapter_model.safetensors"))
    elif bin_path.exists():
        weights = torch.load(bin_path, map_location="cpu")
        torch.save(weights, tmp_dir / "adapter_model.bin")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise FileNotFoundError(f"Missing adapter weights in {src}")

    return tmp_dir


def _load_adapter_with_fallback(base_model, adapter_path: str):
    temp_dirs: list[Path] = []
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        return model, temp_dirs
    except TypeError as exc:
        msg = str(exc)
        if "unexpected keyword argument" not in msg:
            raise
        print(
            "Adapter config is newer than local PEFT; retrying with "
            "runtime-sanitized adapter_config.json ..."
        )
        compat_dir = _build_runtime_compat_copy(adapter_path)
        temp_dirs.append(compat_dir)
        model = PeftModel.from_pretrained(base_model, str(compat_dir))
        return model, temp_dirs


def _echo_encoding_probe():
    print(ENCODING_PROBE)


def _load_best_tokenizer(adapter_path: str, base_model_ref: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        print(f"Loaded tokenizer from adapter: {adapter_path}")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_ref)
        print(f"Loaded tokenizer from base model: {base_model_ref}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def main(args):
    _echo_encoding_probe()
    if args.encoding_test:
        print("(Run without --encoding_test to load model and run smoke test.)")
        return

    base_model_ref = _resolve_model_ref(args.base_model)
    adapter_path = _resolve_path(args.adapter)

    print("Loading tokenizer...")
    tokenizer = _load_best_tokenizer(adapter_path, base_model_ref)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_ref,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=None,
        attn_implementation="sdpa",
    )

    target_vocab = ((len(tokenizer) + 63) // 64) * 64
    if model.config.vocab_size != target_vocab:
        print(f"Resizing embeddings: {model.config.vocab_size} -> {target_vocab}")
        model.resize_token_embeddings(target_vocab, mean_resizing=False)

    temp_dirs: list[Path] = []
    try:
        print("Loading FR adapter...")
        model, adapter_tmp = _load_adapter_with_fallback(model, adapter_path)
        temp_dirs.extend(adapter_tmp)

        model.to(args.device)
        model.eval()

        print("\n" + "=" * 90)
        print("FRENCH ADAPTER SMOKE TEST")
        print("=" * 90)

        for idx, prompt in enumerate(FR_PROMPTS, start=1):
            text = _generate(
                model=model,
                tokenizer=tokenizer,
                instruction=prompt,
                max_new_tokens=args.max_new_tokens,
                greedy=args.greedy,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            fr_hits, ca_hits, oc_hits = _lexical_probe(text)
            print(f"\n[{idx}] Prompt: {prompt}")
            print("-" * 90)
            print(text)
            print(
                "\nLexical probe -> "
                f"FR markers: {fr_hits} | CA markers: {ca_hits} | OC markers: {oc_hits}"
            )
    finally:
        for tmp_dir in temp_dirs:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(
        "\nDone. If the standalone French adapter still shows strong cross-language bleed, "
        "a larger/cleaner French instruction set is likely justified."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick smoke test for the French adapter")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL, help="Base model local path or HF id")
    parser.add_argument("--adapter", default=DEFAULT_FR_ADAPTER, help="French adapter directory")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--encoding_test",
        action="store_true",
        help="Only print the encoding probe line and exit (to test terminal UTF-8 display).",
    )
    main(parser.parse_args())
