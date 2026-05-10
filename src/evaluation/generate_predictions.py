"""
Generate line-aligned model predictions for translation scoring.

Example:
    python -m src.evaluation.generate_predictions --model-path checkpoints/occitan_3b2/final_compat --source-file flores_eval_data/fra_Latn.txt --output-file results/preds/fr_to_oc.txt --base-model models/llama-3.1-occitan-initialized
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.evaluation.score_metrics import load_model_and_tokenizer, read_text_lines


DEFAULT_PROMPT_TEMPLATE = (
    "Sès un assistent d'intelligéncia artificiala que parla unicament en occitan lengadocian.\n\n"
    "### Instruction:\n"
    "Traduís en occitan lengadocian la frasa francesa seguenta.\n\n"
    "### Input:\n"
    "{source}\n\n"
    "### Response:\n"
)


def generate_predictions(
    model_path: str,
    source_file: str,
    output_file: str,
    base_model: str | None = None,
    adapters: list[str] | None = None,
    device: str | None = None,
    max_new_tokens: int = 150,
    do_sample: bool = False,
    temperature: float = 0.3,
    top_p: float = 0.9,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    raw_input: bool = False,
    ablate_expert_idx: int | None = None,
    oc_bias: float = 1.0,
) -> dict:
    model, tokenizer, device, load_info = load_model_and_tokenizer(
        model_path=model_path,
        device=device,
        base_model=base_model,
        adapter_paths=adapters,
    )

    source_lines = read_text_lines(source_file)
    source_lines = [line.strip() for line in source_lines if line.strip()]
    if not source_lines:
        raise ValueError(f"Source file is empty: {source_file}")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    hook_handles = []
    if hasattr(model, "routers"):
        if ablate_expert_idx is not None:
            print(f"!!! WARNING: Ablating expert at index {ablate_expert_idx} !!!")

            def hook_fn(module, args, kwargs, output):
                if isinstance(output, torch.Tensor) and output.shape[-1] > ablate_expert_idx:
                    hacked = output.clone()
                    hacked[..., ablate_expert_idx] = -1e4
                    return hacked
                return output

        elif oc_bias > 0.0:
            print(f"!!! INJECTING +{oc_bias} LOGIT BIAS TO OCCITAN EXPERT (Index 2) !!!")

            def hook_fn(module, args, kwargs, output):
                if isinstance(output, torch.Tensor) and output.shape[-1] > 2:
                    hacked = output.clone()
                    hacked[..., 2] += oc_bias
                    return hacked
                return output

        else:
            hook_fn = None

        if hook_fn is not None:
            for router in model.routers.values():
                hook_handles.append(router.register_forward_hook(hook_fn, with_kwargs=True))

    generated_count = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for idx, source in enumerate(source_lines, start=1):
            if raw_input:
                prompt = source
            else:
                prompt = prompt_template.format(source=source)

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = int(inputs["input_ids"].shape[1])

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            generated_ids = outputs[0][input_len:]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().replace("\n", " ")
            out.write(prediction + "\n")
            generated_count += 1

            if idx % 100 == 0:
                print(f"Generated {idx}/{len(source_lines)} lines...")

    for h in hook_handles:
        h.remove()

    return {
        "model_path": model_path,
        "loader_type": load_info.get("loader_type"),
        "resolved_path": load_info.get("resolved_path"),
        "source_file": source_file,
        "output_file": output_file,
        "num_lines": generated_count,
        "device": device,
        "ablated_expert": ablate_expert_idx,
        "oc_bias": oc_bias,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FLORES prediction file for chrF++ evaluation."
    )
    parser.add_argument("--model-path", required=True, help="Model/adapter/router path.")
    parser.add_argument("--source-file", required=True, help="Source file (one sentence per line).")
    parser.add_argument("--output-file", required=True, help="Output predictions file path.")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model required when --model-path is an adapter or MoLE router checkpoint.",
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=None,
        help="Adapter list required for MoLE router checkpoints.",
    )
    parser.add_argument("--device", default=None, help="Device, e.g. cuda or cpu.")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling (default is greedy).")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--raw-input",
        action="store_true",
        help="Use source line directly as prompt (no instruction template).",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template with {source} placeholder (ignored with --raw-input).",
    )
    parser.add_argument(
        "--ablate-expert-idx",
        type=int,
        default=None,
        help="Index of the expert to ablate (set logits to -inf). For Occitan, this is usually 2.",
    )
    parser.add_argument(
        "--oc-bias",
        type=float,
        default=1.0,
        help="Positive router logit bias added to the Occitan expert (index 2). Set to 0.0 to disable.",
    )
    args = parser.parse_args()

    result = generate_predictions(
        model_path=args.model_path,
        source_file=args.source_file,
        output_file=args.output_file,
        base_model=args.base_model,
        adapters=args.adapters,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_template=args.prompt_template,
        raw_input=args.raw_input,
        ablate_expert_idx=args.ablate_expert_idx,
        oc_bias=args.oc_bias,
    )
    print("\nPrediction generation complete.")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
