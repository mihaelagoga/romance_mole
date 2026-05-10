"""
Run a quick generation smoke test for the Catalan expert adapter.

Example:
    python -m src.evaluation.smoke_test_catalan_expert --adapter checkpoints/lora_ca/final_compat --max_new_tokens 80
"""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_BASE_MODEL = "models/llama-3.1-occitan-initialized"
DEFAULT_CA_ADAPTER = "checkpoints/lora_ca/final"

PROMPTS = [
    "Explica breument per què el cel és blau.",
    "Escriu una resposta curta i educada a un correu professional en català.",
    "Resumeix en 3 frases el concepte d'aprenentatge automàtic.",
]


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
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = out[0][prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def main(args):
    base_model_ref = _resolve_model_ref(args.base_model)
    adapter_path = _resolve_path(args.adapter)

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        print(f"Loaded tokenizer from adapter: {adapter_path}")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_ref)
        print(f"Loaded tokenizer from base model: {base_model_ref}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

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

    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(args.device)
    model.eval()

    print("\n" + "=" * 90)
    print("CATALAN ADAPTER SMOKE TEST")
    print("=" * 90)
    for i, prompt in enumerate(PROMPTS, start=1):
        text = _generate(
            model=model,
            tokenizer=tokenizer,
            instruction=prompt,
            max_new_tokens=args.max_new_tokens,
            greedy=args.greedy,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\n[{i}] Prompt: {prompt}")
        print("-" * 90)
        print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick smoke test for Catalan adapter")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL, help="Base model local path or HF id")
    parser.add_argument("--adapter", default=DEFAULT_CA_ADAPTER, help="Catalan adapter directory")
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
    main(parser.parse_args())
