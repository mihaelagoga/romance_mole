"""
Train the Romance-MoLE router over frozen LoRA experts.

Example:
    torchrun --nproc_per_node=2 -m src.training.train_romance_mole_router --base_model models/llama-3.1-occitan-initialized --adapters checkpoints/lora_fr/final checkpoints/lora_ca/final_compat checkpoints/occitan_3b2/final_compat --adapter_names fr ca oc --data data/router_training/mole_router_train.jsonl --eval_data data/router_training/mole_router_dev.jsonl --output_dir checkpoints/router_ablation_ladder/t8_aux001_sup002_cs000 --sequence_route_threshold 8 --router_aux_loss_coef 0.01 --router_supervision_coef 0.02 --expandable-cuda-segments --disable-nccl-p2p --isolate-visible-gpus
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import sys
from pathlib import Path

if "--expandable-cuda-segments" in sys.argv:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if "--disable-nccl-p2p" in sys.argv:
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")

# Optional workaround for 24 GB multi-GPU runs. It must run before importing
# torch because CUDA_VISIBLE_DEVICES is read during CUDA initialization.
if "--isolate-visible-gpus" in sys.argv and "LOCAL_RANK" in os.environ:
    _lr = int(os.environ["LOCAL_RANK"])
    _vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _vis:
        _gpus = [g.strip() for g in _vis.split(",")]
        if len(_gpus) > 1 and _lr < len(_gpus):
            os.environ["CUDA_VISIBLE_DEVICES"] = _gpus[_lr]
            os.environ["LOCAL_RANK"] = "0"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from src.models.mole.romance_mole import RomanceMoLEModel
except ModuleNotFoundError:
    from models.mole.romance_mole import RomanceMoLEModel

try:
    from src.training.train_occitan_curriculum_lora import (
        DEFAULT_BASE_MODEL,
        LossGapLoggerCallback,
        ParanoidDataCollator,
        detect_adapter_vocab_size,
        load_dataset_safe,
    )
except ModuleNotFoundError:
    from training.train_occitan_curriculum_lora import (
        DEFAULT_BASE_MODEL,
        LossGapLoggerCallback,
        ParanoidDataCollator,
        detect_adapter_vocab_size,
        load_dataset_safe,
    )


MOLE_ROUTER_DEFAULTS = {
    "learning_rate": 5e-4,
    "warmup_steps": 50,
    "weight_decay": 0.01,
    "default_max_steps": 500,
    "max_seq_length": 128,
}


class MoLEDDPTrainer(Trainer):
    """Trainer that saves only router weights/config."""

    def _save(self, output_dir: str | None = None, state_dict=None):
        if not self.args.should_save:
            return

        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model = self.model
        if hasattr(model, "module"):
            model = model.module

        torch.save(
            model.routers.state_dict(),
            os.path.join(output_dir, "router_weights.pt"),
        )

        router_cfg = {
            "adapter_names": model.adapter_names,
            "num_experts": model.num_experts,
            "sequence_route_threshold": model.sequence_route_threshold,
            "router_aux_loss_coef": model.router_aux_loss_coef,
            "router_supervision_coef": getattr(model, "router_supervision_coef", 0.0),
            "router_supervision_class_weights": getattr(
                model,
                "router_supervision_class_weights",
                None,
            ).tolist() if hasattr(model, "router_supervision_class_weights") else None,
            "router_temperature": getattr(model, "router_temperature", 1.0),
            "hard_router_argmax": getattr(model, "hard_router_argmax", False),
            "router_modes": model.router_modes,
        }
        with open(os.path.join(output_dir, "router_config.json"), "w", encoding="utf-8") as f:
            json.dump(router_cfg, f, indent=2)

        print(f"[MoLE/DDP] Saved: {output_dir}")


def _load_tokenizer(adapter_paths: list[str], base_model_path: str) -> AutoTokenizer:
    for path in adapter_paths:
        try:
            tok = AutoTokenizer.from_pretrained(path)
            print(f"[MoLE] Tokenizer (adapter): {path}")
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
                tok.pad_token_id = tok.eos_token_id
            return tok
        except Exception:
            continue

    print(f"[MoLE] Tokenizer (base fallback): {base_model_path}")
    tok = AutoTokenizer.from_pretrained(base_model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def _load_base_model(base_model_path: str, adapter_paths: list[str], tokenizer: AutoTokenizer):
    target_vocab = detect_adapter_vocab_size(adapter_paths[0])
    if target_vocab:
        print(f"[MoLE] Detected adapter embedding size: {target_vocab:,}")
    else:
        target_vocab = math.ceil(len(tokenizer) / 64) * 64
        print(f"[MoLE] Target vocab (64-aligned): {target_vocab:,}")

    print(f"[MoLE] Loading base model on CPU: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    if model.config.vocab_size != target_vocab:
        print(f"[MoLE] Resize embeddings: {model.config.vocab_size} -> {target_vocab}")
        model.resize_token_embeddings(target_vocab, mean_resizing=False)

    return model


def _patch_for_trainer(mole: RomanceMoLEModel) -> None:
    mole.supports_gradient_checkpointing = True
    mole.gradient_checkpointing_enable = lambda **kw: mole.model.gradient_checkpointing_enable(**kw)
    mole.gradient_checkpointing_disable = lambda: mole.model.gradient_checkpointing_disable()


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _load_router_dataset(
    data_path: str,
    tokenizer,
    *,
    max_seq_length: int,
    expert_label_to_index: dict[str, int],
    router_supervision_coef: float,
):
    """Router-specific dataset loader with neutral Alpaca formatting."""
    print(f"Dataset: {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"Items: {len(dataset):,}")

    if "text" in dataset.column_names:
        def tokenize_text(examples):
            model_inputs = tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )
            if router_supervision_coef > 0.0 and "lang" in examples:
                model_inputs["expert_target"] = [
                    int(expert_label_to_index.get(str(lang), -100))
                    for lang in examples["lang"]
                ]
            return model_inputs

        print("[MoLE/DDP] Tokenizing text dataset")
        return dataset.map(
            tokenize_text,
            batched=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
            desc="Tokenizing text",
        )

    required_cols = {"instruction", "output"}
    if required_cols.issubset(set(dataset.column_names)):
        print(
            "[MoLE/DDP] Detected instruction-format dataset. "
            "Building neutral Alpaca prompts for router training..."
        )

        def tokenize_instruction(examples):
            input_col = examples.get("input", [""] * len(examples["instruction"]))
            lang_col = examples.get("lang", [None] * len(examples["instruction"]))
            codeswitch_col = examples.get("is_codeswitched", [False] * len(examples["instruction"]))
            include_expert_target = router_supervision_coef > 0.0
            model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
            if include_expert_target:
                model_inputs["expert_target"] = []

            for instruction, inp, output, lang, is_codeswitched in zip(
                examples["instruction"], input_col, examples["output"], lang_col, codeswitch_col
            ):
                instruction = _safe_text(instruction)
                inp = _safe_text(inp)
                output = _safe_text(output)
                lang = _safe_text(lang)

                if inp:
                    prompt = (
                        f"### Instruction:\n{instruction}\n\n"
                        f"### Input:\n{inp}\n\n"
                        "### Response:\n"
                    )
                else:
                    prompt = (
                        f"### Instruction:\n{instruction}\n\n"
                        "### Response:\n"
                    )

                full_text = prompt + output
                full_ids = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                    add_special_tokens=False,
                )["input_ids"]
                prompt_ids = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                    add_special_tokens=False,
                )["input_ids"]

                if tokenizer.eos_token_id is not None and len(full_ids) < max_seq_length:
                    full_ids = full_ids + [tokenizer.eos_token_id]
                if not full_ids:
                    continue

                prompt_len = min(len(prompt_ids), len(full_ids))
                labels = full_ids.copy()
                labels[:prompt_len] = [-100] * prompt_len
                if all(x == -100 for x in labels):
                    continue

                model_inputs["input_ids"].append(full_ids)
                model_inputs["attention_mask"].append([1] * len(full_ids))
                model_inputs["labels"].append(labels)
                if include_expert_target:
                    expert_target = -100
                    if not bool(is_codeswitched):
                        expert_target = int(expert_label_to_index.get(lang, -100))
                    model_inputs["expert_target"].append(expert_target)

            return model_inputs

        tokenized = dataset.map(
            tokenize_instruction,
            batched=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
            desc="Tokenizing instruction data",
        )
        print("[MoLE/DDP] Filter empty/masked rows")
        tokenized = tokenized.filter(
            lambda ex: len(ex["input_ids"]) > 0 and any(lbl != -100 for lbl in ex["labels"]),
            desc="Filtering invalid rows",
        )
        print(f"Kept rows: {len(tokenized):,}")
        return tokenized

    raise ValueError(
        "Unsupported dataset format. Expected either a 'text' column "
        "or Alpaca columns including 'instruction' and 'output'."
    )


def _build_mole_model(
    *,
    base_model,
    adapter_paths: list[str],
    adapter_names: list[str],
    sequence_route_threshold: int,
    router_aux_loss_coef: float,
    router_supervision_coef: float,
    router_supervision_class_weights: list[float] | None,
    router_temperature: float,
    hard_router_argmax: bool,
):
    """Compatibility wrapper around `RomanceMoLEModel`."""
    params = inspect.signature(RomanceMoLEModel.__init__).parameters
    kwargs = {
        "base_model": base_model,
        "adapter_paths": adapter_paths,
        "adapter_names": adapter_names,
        "sequence_route_threshold": sequence_route_threshold,
        "router_aux_loss_coef": router_aux_loss_coef,
    }
    if "router_supervision_coef" in params:
        kwargs["router_supervision_coef"] = router_supervision_coef
    elif router_supervision_coef > 0.0:
        print(
            "[MoLE/DDP] Warning: imported RomanceMoLEModel does not support "
            "router_supervision_coef; explicit router supervision is disabled "
            "for this run. Make sure the updated MoLE model file is on the "
            "runtime PYTHONPATH if you intended to use --router_supervision_coef."
        )
    if "router_supervision_class_weights" in params:
        kwargs["router_supervision_class_weights"] = router_supervision_class_weights
    elif router_supervision_class_weights is not None:
        print(
            "[MoLE/DDP] Warning: imported RomanceMoLEModel does not support "
            "router_supervision_class_weights; class-weighted supervision is disabled "
            "for this run."
        )
    if "router_temperature" in params:
        kwargs["router_temperature"] = router_temperature
    elif router_temperature != 1.0:
        print(
            "[MoLE/DDP] Warning: imported RomanceMoLEModel does not support "
            "router_temperature; using the model default instead."
        )
    if "hard_router_argmax" in params:
        kwargs["hard_router_argmax"] = hard_router_argmax
    elif hard_router_argmax:
        print(
            "[MoLE/DDP] Warning: imported RomanceMoLEModel does not support "
            "hard_router_argmax; using the model default instead."
        )
    return RomanceMoLEModel(**kwargs)


def train(args: argparse.Namespace) -> None:
    torch.cuda.empty_cache()
    gc.collect()

    cfg = MOLE_ROUTER_DEFAULTS
    lr = args.learning_rate if args.learning_rate is not None else cfg["learning_rate"]
    max_seq_length = args.max_seq_length
    expert_label_to_index = {
        str(name): index for index, name in enumerate(args.adapter_names)
    }
    router_supervision_class_weights = None
    if args.router_class_weights is not None:
        if len(args.router_class_weights) != len(args.adapter_names):
            raise ValueError(
                f"--router_class_weights must have the same length as --adapter_names; "
                f"got {len(args.router_class_weights)} weights for {len(args.adapter_names)} experts."
            )
        router_supervision_class_weights = [float(weight) for weight in args.router_class_weights]

    if len(args.adapters) != len(args.adapter_names):
        raise ValueError(
            f"--adapters ({len(args.adapters)}) and --adapter_names ({len(args.adapter_names)}) "
            "must have the same length"
        )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    eval_enabled = args.eval_data is not None
    ta_params = inspect.signature(TrainingArguments.__init__).parameters

    eval_kwargs: dict = {"eval_steps": 50} if eval_enabled else {}
    if "evaluation_strategy" in ta_params:
        eval_kwargs["evaluation_strategy"] = "steps" if eval_enabled else "no"
    elif "eval_strategy" in ta_params:
        eval_kwargs["eval_strategy"] = "steps" if eval_enabled else "no"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name="mole_router_ddp",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        remove_unused_columns=False,
        gradient_accumulation_steps=16,
        max_steps=args.max_steps,
        warmup_steps=cfg["warmup_steps"],
        learning_rate=lr,
        lr_scheduler_type="cosine",
        weight_decay=cfg["weight_decay"],
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        prediction_loss_only=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        ddp_find_unused_parameters=True,
        **eval_kwargs,
    )

    local_rank = training_args.local_rank

    print("=" * 70)
    print("  ROMANCE-MoLE ROUTER TRAINING (DDP)")
    print("=" * 70)
    print(f"  Adapters:       {args.adapters}")
    print(f"  Adapter names:  {args.adapter_names}")
    print(f"  Base model:     {args.base_model}")
    print(f"  Data:           {args.data}")
    print(f"  Eval data:      {args.eval_data or 'None'}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Max steps:      {args.max_steps}")
    print(f"  Learning rate:  {lr}")
    print(f"  HMoRA thresh:   {args.sequence_route_threshold}")
    print(f"  Aux-loss coef:  {args.router_aux_loss_coef}")
    print(f"  Sup-loss coef:  {args.router_supervision_coef}")
    print(f"  Sup class wts:  {router_supervision_class_weights}")
    print(f"  Router temp:    {args.router_temperature}")
    print(f"  Hard argmax:    {args.hard_router_argmax}")
    print(f"  Max seq len:    {max_seq_length}")
    print(f"  Local rank:     {local_rank}")
    print("=" * 70 + "\n")

    tokenizer = _load_tokenizer(args.adapters, args.base_model)
    
    with training_args.main_process_first(desc="dataset map"):
        dataset = _load_router_dataset(
            args.data,
            tokenizer,
            max_seq_length=max_seq_length,
            expert_label_to_index=expert_label_to_index,
            router_supervision_coef=args.router_supervision_coef,
        )
        eval_dataset = None
        if args.eval_data:
            eval_dataset = _load_router_dataset(
                args.eval_data,
                tokenizer,
                max_seq_length=max_seq_length,
                expert_label_to_index=expert_label_to_index,
                router_supervision_coef=args.router_supervision_coef,
            )
            if local_rank in [-1, 0]:
                print(f"Eval items: {len(eval_dataset):,}")

    print("\n[MoLE] Load models")
    base_model = _load_base_model(args.base_model, args.adapters, tokenizer)
    
    print("\n[MoLE] Build RomanceMoLEModel")
    mole = _build_mole_model(
        base_model=base_model,
        adapter_paths=args.adapters,
        adapter_names=args.adapter_names,
        sequence_route_threshold=args.sequence_route_threshold,
        router_aux_loss_coef=args.router_aux_loss_coef,
        router_supervision_coef=args.router_supervision_coef,
        router_supervision_class_weights=router_supervision_class_weights,
        router_temperature=args.router_temperature,
        hard_router_argmax=args.hard_router_argmax,
    )
    mole = mole.to(torch.bfloat16)
    
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    _patch_for_trainer(mole)

    device = training_args.device  # cuda:0 for every process
    print(f"\n[MoLE] Model to {device}")
    mole = mole.to(device)

    if local_rank in [-1, 0]:
        mole.print_parameter_summary()
        mole.print_router_summary()

    data_collator = ParanoidDataCollator(tokenizer=tokenizer, vocab_limit=mole.config.vocab_size)

    trainer = MoLEDDPTrainer(
        model=mole,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[LossGapLoggerCallback()] if eval_enabled else None,
    )

    print(f"\n{'=' * 70}")
    print(f"  Starting MoLE Router Training (DDP) ({args.max_steps} steps, lr={lr})")
    print(f"{'=' * 70}\n")
    trainer.train()

    final_path = os.path.join(args.output_dir, "final")
    print(f"\n[MoLE/DDP] Saving final router -> {final_path}")
    trainer.save_model(final_path)
    if local_rank in [-1, 0]:
        tokenizer.save_pretrained(final_path)

    print("\n" + "=" * 70)
    print("  MoLE Router DDP Training Complete!")
    print(f"  Output: {final_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Romance-MoLE router weights (DDP)")

    parser.add_argument("--adapters", nargs="+", required=True, help="Paths to frozen LoRA adapters")
    parser.add_argument("--adapter_names", nargs="+", required=True, help="Adapter names, e.g. fr ca oc")

    parser.add_argument(
        "--data",
        default="data/synthetic/merged_splits/train_merged.jsonl",
    )
    parser.add_argument(
        "--eval_data",
        default="data/synthetic/merged_splits/dev_merged.jsonl",
        help="Set empty string to disable evaluation",
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output_dir", default="checkpoints/mole_router_ddp")

    parser.add_argument("--sequence_route_threshold", type=int, default=12)
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.001)
    parser.add_argument(
        "--router_supervision_coef",
        type=float,
        default=0.0,
        help="Optional coefficient for explicit expert-label router supervision.",
    )
    parser.add_argument(
        "--router_class_weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional per-expert class weights for router supervision, aligned with --adapter_names.",
    )
    parser.add_argument(
        "--router_temperature",
        type=float,
        default=1.0,
        help="Optional router softmax temperature during training/eval (default: 1.0).",
    )
    parser.add_argument(
        "--hard_router_argmax",
        action="store_true",
        help="Force one-hot router decisions when gradients are disabled (mainly for eval/inference).",
    )

    parser.add_argument("--max_steps", type=int, default=MOLE_ROUTER_DEFAULTS["default_max_steps"])
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=MOLE_ROUTER_DEFAULTS["max_seq_length"],
        help="Tokenization max length (128 default for 24GB GPUs)",
    )
    parser.add_argument(
        "--expandable-cuda-segments",
        action="store_true",
        help="Opt in to PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before torch import.",
    )
    parser.add_argument(
        "--disable-nccl-p2p",
        action="store_true",
        help="Opt in to NCCL_P2P_DISABLE=1 for clusters where peer buffers cause OOM.",
    )
    parser.add_argument(
        "--isolate-visible-gpus",
        action="store_true",
        help="Opt in to per-rank CUDA_VISIBLE_DEVICES isolation for memory-constrained torchrun jobs.",
    )

    cli_args = parser.parse_args()
    if cli_args.eval_data is not None and cli_args.eval_data.strip() == "":
        cli_args.eval_data = None

    train(cli_args)
