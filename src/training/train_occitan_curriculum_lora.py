"""
Train the two-stage Occitan curriculum LoRA adapter.

Examples:
  python -m src.training.train_occitan_curriculum_lora --stage hplt --data data/hplt_v3/splits/train_hplt.jsonl --eval_data data/hplt_v3/splits/dev_hplt_1000.jsonl --adapter models/adapter_oc_init --output_dir checkpoints/occitan_hplt --expandable-cuda-segments
  
  python -m src.training.train_occitan_curriculum_lora --stage synth --data data/synthetic/merged_splits/train_merged.jsonl --eval_data data/synthetic/merged_splits/dev_merged.jsonl --adapter checkpoints/occitan_hplt/final --output_dir checkpoints/occitan_synth --expandable-cuda-segments
"""

import argparse
import gc
import inspect
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if "--expandable-cuda-segments" in sys.argv:
  os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from datasets import load_dataset
from peft import PeftConfig, PeftModel
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  EarlyStoppingCallback,
  Trainer,
  TrainerCallback,
  TrainingArguments,
)


DEFAULT_BASE_MODEL = "models/llama-3.1-occitan-initialized"


STAGE_CONFIG = {
  "hplt": {
    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "weight_decay": 0.0,
    "neftune_noise_alpha": None,
    "default_max_steps": 2000,
    "max_seq_length": 512,
    "description": "General Adaptation (HPLT)",
  },
  "synth": {
    "learning_rate": 4e-5,      # Increased to force weight shifting
    "warmup_steps": 50,       # Slightly longer warmup for stability
    "weight_decay": 0.05,      # Higher decay to prevent memorizing synth quirks
    "neftune_noise_alpha": 1,    # Low noise to avoid disrupting fragile new syntactic patterns
    "default_max_steps": 2500,    # Extended steps to overcome CA/FR bias in MLP layers
    "max_seq_length": 1024,
    "description": "Aggressive Dialect Steering (Occitan Expert)",
  },
}


@dataclass
class ParanoidDataCollator:
  """
  Pads causal-LM batches and clamps labels to model vocab limits.
  Handles both plain text (labels = input_ids) and precomputed labels
  (instruction masking where prompt tokens are -100).
  """
  tokenizer: AutoTokenizer
  vocab_limit: int = 128000

  def __call__(self, examples):
    has_precomputed_labels = "labels" in examples[0]
    label_rows = [ex.get("labels", []) for ex in examples] if has_precomputed_labels else None
    passthrough_keys = {
      key for key in examples[0]
      if key not in {"input_ids", "attention_mask", "token_type_ids", "special_tokens_mask", "labels"}
    }
    passthrough_values = {
      key: [ex.get(key) for ex in examples]
      for key in passthrough_keys
    }
    model_examples = [
      {
        key: value
        for key, value in ex.items()
        if key in {"input_ids", "attention_mask", "token_type_ids", "special_tokens_mask"}
      }
      for ex in examples
    ]
    batch = self.tokenizer.pad(model_examples, padding=True, return_tensors="pt")

    if has_precomputed_labels and label_rows is not None:
      seq_len = int(batch["input_ids"].shape[1])
      padded_labels = []
      for row in label_rows:
        row_list = list(row) if isinstance(row, list) else list(row or [])
        if len(row_list) > seq_len:
          row_list = row_list[:seq_len]
        if len(row_list) < seq_len:
          row_list = row_list + ([-100] * (seq_len len(row_list)))
        padded_labels.append(row_list)
      labels = torch.tensor(padded_labels, dtype=torch.long)
    else:
      labels = batch["input_ids"].clone()
      if self.tokenizer.pad_token_id is not None:
        labels[labels == self.tokenizer.pad_token_id] = -100

    ignore_mask = labels == -100
    clamped = torch.clamp(labels, min=0, max=self.vocab_limit 1)
    labels = torch.where(ignore_mask, torch.full_like(clamped, -100), clamped)
    batch["labels"] = labels
    for key, values in passthrough_values.items():
      if all(value is None for value in values):
        continue
      example_value = next((value for value in values if value is not None), None)
      if isinstance(example_value, bool):
        batch[key] = torch.tensor([bool(value) for value in values], dtype=torch.bool)
      elif isinstance(example_value, int):
        normalized = [-100 if value is None else int(value) for value in values]
        batch[key] = torch.tensor(normalized, dtype=torch.long)
    return batch


class LossGapLoggerCallback(TrainerCallback):
  """Logs train-vs-eval loss gap at logging intervals."""

  def __init__(self):
    self.latest_train_loss = None
    self.latest_eval_loss = None

  def on_log(self, args, state, control, logs=None, **kwargs):
    if not logs:
      return
    if "loss" in logs:
      self.latest_train_loss = logs["loss"]
    if "eval_loss" in logs:
      self.latest_eval_loss = logs["eval_loss"]
    if self.latest_train_loss is not None and self.latest_eval_loss is not None:
      gap = self.latest_eval_loss self.latest_train_loss
      print(
        f"[Loss Monitor] step={state.global_step} "
        f"train_loss={self.latest_train_loss:.4f} "
        f"eval_loss={self.latest_eval_loss:.4f} "
        f"gap={gap:+.4f}"
      )


def detect_adapter_vocab_size(adapter_path):
  """Detect the embedding size from the adapter's saved weights."""
  from safetensors.torch import load_file

  adapter_file = Path(adapter_path) / "adapter_model.safetensors"
  if not adapter_file.exists():
    return None

  weights = load_file(adapter_file)
  embed_keys = [k for k in weights.keys() if "embed_tokens" in k]
  if not embed_keys:
    embed_keys = [k for k in weights.keys() if "original_module" in k and "embed" in k.lower()]

  if embed_keys:
    vocab_size = weights[embed_keys[0]].shape[0]
    del weights
    return vocab_size

  del weights
  return None


def load_model_chain(base_model_path, adapter_path):
  """
  Load base model, resize embeddings (before PEFT), attach adapter with modules_to_save
  so PEFT manages and saves embed_tokens and lm_head. All embedding rows are fully
  trainable to allow overcoming CA/FR bias from the TIES-merged base adapter.
  """

  print(f"Tokenizer: {adapter_path}")
  try:
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
  except Exception:
    print(f"Fallback to base tokenizer: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

  target_vocab = detect_adapter_vocab_size(adapter_path)
  if target_vocab:
    print(f"Detected adapter embedding size: {target_vocab:,}")
  else:
    target_vocab = math.ceil(len(tokenizer) / 64) * 64
    print(f"Calculated target vocab (64-aligned): {target_vocab:,}")

  print(f"Base model (CPU): {base_model_path}")
  model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map=None,
    attn_implementation="sdpa",
  )

  # Resize *before* wrapping with PEFT so modules_to_save wraps the correctly sized layers.
  if model.config.vocab_size != target_vocab:
    print(f"Resizing embeddings: {model.config.vocab_size} -> {target_vocab}")
    model.resize_token_embeddings(target_vocab, mean_resizing=False)

  print(f"Adapter config: {adapter_path}")
  peft_config = PeftConfig.from_pretrained(adapter_path)
  current_mts = getattr(peft_config, "modules_to_save", None)
  if current_mts is None:
    peft_config.modules_to_save = ["embed_tokens", "lm_head"]
  else:
    mts_list = list(current_mts) if isinstance(current_mts, (list, tuple)) else [current_mts]
    for m in ["embed_tokens", "lm_head"]:
      if m not in mts_list:
        mts_list.append(m)
    peft_config.modules_to_save = mts_list
  print(f" modules_to_save: {peft_config.modules_to_save}")

  print(f"Wrap PEFT and load adapter weights")
  model = PeftModel(model, peft_config, adapter_name="default")
  model.load_adapter(adapter_path, adapter_name="default")


  model.print_trainable_parameters()

  print("Model to CUDA")
  model = model.to("cuda")
  model.gradient_checkpointing_enable()

  return model, tokenizer


def load_dataset_safe(
  data_path,
  tokenizer,
  max_seq_length: int = 1024,
  expert_label_to_index: dict[str, int] | None = None,
):
  """Load JSONL data — handles both plain 'text' and Alpaca instruction format."""
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
      if expert_label_to_index is not None and "lang" in examples:
        model_inputs["expert_target"] = [
          int(expert_label_to_index.get(str(lang), -100))
          for lang in examples["lang"]
        ]
      return model_inputs

    print("Tokenizing text dataset")
    return dataset.map(
      tokenize_text,
      batched=True,
      remove_columns=dataset.column_names,
      load_from_cache_file=False,
      desc="Tokenizing text",
    )

  required_cols = {"instruction", "output"}
  if required_cols.issubset(set(dataset.column_names)):
    print("Instruction format detected (Alpaca). Masking responses")

    def _safe_text(value):
      if value is None:
        return ""
      if isinstance(value, float) and value != value:
        return ""
      text = str(value).strip()
      return "" if text.lower() == "nan" else text

    def tokenize_instruction(examples):
      input_col = examples.get("input", [""] * len(examples["instruction"]))
      lang_col = examples.get("lang", [None] * len(examples["instruction"]))
      codeswitch_col = examples.get("is_codeswitched", [False] * len(examples["instruction"]))
      include_expert_target = expert_label_to_index is not None
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

        system_anchor = "Sès un assistent d'intelligéncia artificiala que parla unicament en occitan lengadocian.\n\n"

        if inp:
          prompt = (
            system_anchor
            + f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            "### Response:\n"
          )
        else:
          prompt = (
            system_anchor
            + f"### Instruction:\n{instruction}\n\n"
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

    print("Filter empty/masked rows")
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


def train(args):
  torch.cuda.empty_cache()
  gc.collect()

  cfg = STAGE_CONFIG[args.stage]
  max_seq_length = cfg["max_seq_length"]

  print("=" * 60)
  print(f"OCCITAN TRAINING Stage {args.stage}")
  print(cfg["description"])
  print("=" * 60)
  print(f" Data:     {args.data}")
  print(f" Eval Data:  {args.eval_data or 'None'}")
  print(f" Adapter:   {args.adapter}")
  print(f" Output:    {args.output_dir}")
  print(f" Max Steps:  {args.max_steps}")
  print(f" LR:      {cfg['learning_rate']}")
  print(f" Max Seq Len: {max_seq_length}")
  print(f" Patience:   {args.patience}")
  print("=" * 60 + "\n")

  Path(args.output_dir).mkdir(parents=True, exist_ok=True)

  model, tokenizer = load_model_chain(args.base_model, args.adapter)

  dataset = load_dataset_safe(args.data, tokenizer, max_seq_length=max_seq_length)
  eval_dataset = None
  if args.eval_data:
    print(f"Loading validation dataset from {args.eval_data}...")
    eval_dataset = load_dataset_safe(args.eval_data, tokenizer, max_seq_length=max_seq_length)
    print(f"Validation examples: {len(eval_dataset):,}")

  vocab_limit = model.config.vocab_size
  print(f"Collator vocab limit: {vocab_limit}")
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
    run_name=f"occitan_stage_{args.stage}",
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

  print(f"\nStarting Training (Stage {args.stage}, {args.max_steps} steps)...")
  if eval_enabled:
    print(f"Early stopping enabled (patience={args.patience}, eval every {50} steps)")
  trainer.train()

  final_path = f"{args.output_dir}/final"
  if eval_enabled:
    print(f"\nSaving best model to {final_path}...")
  else:
    print(f"\nSaving to {final_path}...")
  # PEFT save_pretrained writes adapter_model.safetensors and, when modules_to_save
  # is set, the saved embed_tokens/lm_head (e.g. original_module), so next stage loads full vocab.
  trainer.save_model(final_path)
  tokenizer.save_pretrained(final_path)

  print("\n" + "=" * 60)
  print(f"Stage {args.stage} Complete!")
  print(f"Output: {final_path}")
  print("=" * 60)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Occitan Curriculum Training (HPLT / Synth) with Early Stopping"
  )
  parser.add_argument(
    "--stage", type=str, default="hplt", choices=["hplt", "synth"],
    help="Training stage: hplt (general adaptation) or synth (joint dialect + instruction)",
  )
  parser.add_argument("--data", required=True, help="Path to JSONL training data")
  parser.add_argument("--eval_data", default=None, help="Path to JSONL validation data (enables early stopping)")
  parser.add_argument("--adapter", required=True, help="Path to adapter to fine-tune")
  parser.add_argument("--output_dir", required=True, help="Output directory")
  parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
  parser.add_argument(
    "--max_steps", type=int, default=None,
    help="Override training steps. Defaults to stage value (hplt: 2000, synth: 2500).",
  )
  parser.add_argument("--save_steps", type=int, default=500)
  parser.add_argument(
    "--patience", type=int, default=5,
    help="Early stopping patience in eval rounds (default: 5, i.e. 250 steps without improvement)",
  )
  parser.add_argument(
    "--expandable-cuda-segments",
    action="store_true",
    help="Opt in to PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before torch import.",
  )

  args = parser.parse_args()

  if args.max_steps is None:
    args.max_steps = STAGE_CONFIG[args.stage]["default_max_steps"]

  train(args)
