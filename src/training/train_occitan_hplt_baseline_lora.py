"""
Train the HPLT-only Occitan baseline LoRA adapter.

Example:
  python -m src.training.train_occitan_hplt_baseline_lora --data data/hplt_v3/splits/train_hplt.jsonl --eval_data data/hplt_v3/splits/dev_hplt_1000.jsonl --output_dir checkpoints/oc_simple_carballo_hplt
"""

from __future__ import annotations

import argparse
import gc
import inspect
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

from src.training.train_occitan_curriculum_lora import (
  LossGapLoggerCallback,
  ParanoidDataCollator,
  load_dataset_safe,
)


DEFAULT_BASE_MODEL = "proxectonos/Llama-3.1-Carballo"

BASELINE_CONFIG = {
  "learning_rate": 5e-5,
  "warmup_steps": 100,
  "weight_decay": 0.0,
  "neftune_noise_alpha": None,
  "default_max_steps": 2000,
  "max_seq_length": 512,
  "description": "Simple HPLT-only Occitan fine-tuning (baseline for docs/benchmarking_plan.md)",
}

DEFAULT_LORA_TARGET_MODULES = [
  "q_proj",
  "k_proj",
  "v_proj",
  "o_proj",
  "gate_proj",
  "up_proj",
  "down_proj",
]


def load_model_and_tokenizer(
  base_model_path: str,
  lora_r: int,
  lora_alpha: int,
  lora_dropout: float,
):
  print(f"Tokenizer: {base_model_path}")
  tokenizer = AutoTokenizer.from_pretrained(base_model_path)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

  print(f"Base model (CPU): {base_model_path}")
  model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map=None,
    attn_implementation="sdpa",
  )

  model.config.use_cache = False

  print("Create baseline LoRA adapter")
  lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=list(DEFAULT_LORA_TARGET_MODULES),
    bias="none",
    task_type="CAUSAL_LM",
  )
  model = get_peft_model(model, lora_config)

  model.print_trainable_parameters()

  print("Model to CUDA")
  model = model.to("cuda")
  model.gradient_checkpointing_enable()
  if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

  return model, tokenizer


def train(args):
  torch.cuda.empty_cache()
  gc.collect()

  cfg = BASELINE_CONFIG
  max_seq_length = cfg["max_seq_length"]

  print("=" * 60)
  print("OCCITAN SIMPLE BASELINE TRAINING (Carballo + HPLT)")
  print(cfg["description"])
  print("=" * 60)
  print(f" Data:     {args.data}")
  print(f" Eval Data:  {args.eval_data or 'None'}")
  print(f" Base Model:  {args.base_model}")
  print(f" Output:    {args.output_dir}")
  print(f" Max Steps:  {args.max_steps}")
  print(f" LR:      {cfg['learning_rate']}")
  print(f" Max Seq Len: {max_seq_length}")
  print(f" Patience:   {args.patience}")
  print("=" * 60 + "\n")

  Path(args.output_dir).mkdir(parents=True, exist_ok=True)

  model, tokenizer = load_model_and_tokenizer(
    base_model_path=args.base_model,
    lora_r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
  )

  dataset = load_dataset_safe(args.data, tokenizer, max_seq_length=max_seq_length)
  if "labels" in dataset.column_names:
    raise ValueError(
      "Baseline training expects plain-text data with a 'text' field only. "
      "Instruction-format datasets are intentionally rejected so the baseline "
      "cannot accidentally inherit instruction-tuning signal from stage 3b2."
    )

  eval_dataset = None
  if args.eval_data:
    print(f"Validation data: {args.eval_data}")
    eval_dataset = load_dataset_safe(args.eval_data, tokenizer, max_seq_length=max_seq_length)
    if "labels" in eval_dataset.column_names:
      raise ValueError(
        "Baseline eval data must also be plain text only (a 'text' field). "
        "Instruction-format eval would not match the baseline definition."
      )
    print(f"Eval items: {len(eval_dataset):,}")

  vocab_limit = model.config.vocab_size
  print(f"Vocab limit: {vocab_limit}")
  data_collator = ParanoidDataCollator(tokenizer=tokenizer, vocab_limit=vocab_limit)

  eval_enabled = eval_dataset is not None
  ta_params = inspect.signature(TrainingArguments.__init__).parameters
  eval_kwargs = {"eval_steps": 50} if eval_enabled else {}
  if "evaluation_strategy" in ta_params:
    eval_kwargs["evaluation_strategy"] = "steps" if eval_enabled else "no"
  elif "eval_strategy" in ta_params:
    eval_kwargs["eval_strategy"] = "steps" if eval_enabled else "no"

  if eval_enabled:
    eval_kwargs["load_best_model_at_end"] = True
    eval_kwargs["metric_for_best_model"] = "eval_loss"
    eval_kwargs["greater_is_better"] = False

  neftune_alpha = cfg["neftune_noise_alpha"]
  neftune_kwargs = {"neftune_noise_alpha": neftune_alpha} if neftune_alpha is not None else {}

  training_args = TrainingArguments(
    output_dir=args.output_dir,
    run_name="oc_simple_baseline_carballo_hplt",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    max_steps=args.max_steps,
    warmup_steps=cfg["warmup_steps"],
    learning_rate=cfg["learning_rate"],
    lr_scheduler_type="cosine",
    weight_decay=cfg["weight_decay"],
    bf16=True,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    prediction_loss_only=True,
    logging_steps=25,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=3,
    report_to="none",
    **neftune_kwargs,
    **eval_kwargs,
  )

  callbacks = []
  if eval_enabled:
    callbacks.append(LossGapLoggerCallback())
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=callbacks or None,
  )

  print(f"\nStart simple baseline: {args.max_steps} steps")
  print(f"Base: {args.base_model}")
  if eval_enabled:
    print(f"Early stopping: patience={args.patience}")
  trainer.train()

  final_path = f"{args.output_dir}/final"
  if eval_enabled:
    print(f"\nSave best model: {final_path}")
  else:
    print(f"\nSave: {final_path}")
  trainer.save_model(final_path)
  tokenizer.save_pretrained(final_path)

  print("\n" + "=" * 60)
  print("Done")
  print(f"Output: {final_path}")
  print("=" * 60)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description=(
      "Train the OccitanExpert_Simple baseline from docs/benchmarking_plan.md: "
      "stock proxectonos/Llama-3.1-Carballo + fresh LoRA, single-stage HPLT-only "
      "continued pretraining, no anchor, no curriculum, no resume."
    )
  )
  parser.add_argument(
    "--data",
    required=True,
    help="Path to plain-text JSONL training data (must contain a 'text' field, e.g. HPLT Occitan).",
  )
  parser.add_argument(
    "--eval_data",
    default=None,
    help="Path to plain-text JSONL validation data (enables early stopping).",
  )
  parser.add_argument("--output_dir", required=True, help="Output directory")
  parser.add_argument(
    "--base_model",
    default=DEFAULT_BASE_MODEL,
    help=f"Base model path or HF id (default: {DEFAULT_BASE_MODEL}).",
  )
  parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help=(
      "Override training steps. Defaults to BASELINE_CONFIG['default_max_steps'] "
      f"({BASELINE_CONFIG['default_max_steps']})."
    ),
  )
  parser.add_argument("--save_steps", type=int, default=500)
  parser.add_argument(
    "--patience",
    type=int,
    default=5,
    help="Early stopping patience in eval rounds (default: 5).",
  )
  parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank for the fresh adapter.")
  parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha for the fresh adapter.")
  parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout for the fresh adapter.")

  args = parser.parse_args()

  if args.max_steps is None:
    args.max_steps = BASELINE_CONFIG["default_max_steps"]

  train(args)
