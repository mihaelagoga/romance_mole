"""
Train a French or Catalan Romance expert LoRA adapter.

Example:
    python -m src.training.train_romance_expert_lora --lang fr --data data/cousin_data/cousin_fr_augmented.jsonl data/cousin_data/cousin_fr_repair.jsonl --data_repeat 1 3 --output_dir checkpoints/lora_fr

    python -m src.training.train_romance_expert_lora --lang ca --data data/cousin_data/cousin_ca.jsonl --output_dir checkpoints/lora_ca
"""

import argparse
import gc
import math
import shutil
import tempfile
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL = "models/llama-3.1-occitan-initialized"
DEFAULT_CHECKPOINTS_ROOT = "checkpoints"
MAX_SEQ_LENGTH = 1024

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "lm_head",
]
ADAPTER_ARTIFACTS = {
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
}


def _symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
        return
    except OSError:
        pass

    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _build_model_compat_copy(model_name: str) -> Path | None:
    src = Path(model_name)
    if not src.is_dir():
        return None

    has_adapter_artifacts = any((src / name).exists() for name in ADAPTER_ARTIFACTS)
    if not has_adapter_artifacts:
        return None

    tmp_dir = Path(tempfile.mkdtemp(prefix="base_model_runtime_compat_"))
    for child in src.iterdir():
        if child.name in ADAPTER_ARTIFACTS:
            continue
        _symlink_or_copy(child, tmp_dir / child.name)
    return tmp_dir


def load_model_and_tokenizer(model_name: str):
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        print("FlashAttention-2 available — using it (required for correct packing)")
    except ImportError:
        attn_impl = "sdpa"
        print("FlashAttention not installed — falling back to SDPA (packing may cross-contaminate)")

    print("Loading model on CPU for safe resize")
    compat_dir: Path | None = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
            attn_implementation=attn_impl,
        )
    except ImportError as exc:
        if "PEFT_TYPE_TO_PREFIX_MAPPING" not in str(exc):
            raise

        compat_dir = _build_model_compat_copy(model_name)
        if compat_dir is None:
            raise

        print(
            "Detected incompatible PEFT auto-loading in the base model directory; "
            "retrying from a temporary copy without adapter metadata."
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(compat_dir),
            torch_dtype=torch.bfloat16,
            device_map=None,
            attn_implementation=attn_impl,
        )
    finally:
        if compat_dir is not None:
            shutil.rmtree(compat_dir, ignore_errors=True)

    current_len = len(tokenizer)
    target_vocab = math.ceil(current_len / 64) * 64

    print(f"Alignment check:")
    print(f"  Tokenizer: {current_len}")
    print(f"  Target:    {target_vocab}")
    print(f"  Model Old: {model.config.vocab_size}")

    if model.config.vocab_size != target_vocab:
        print(f"Resizing on CPU ({model.config.vocab_size} -> {target_vocab})")
        model.resize_token_embeddings(target_vocab)
        assert model.get_input_embeddings().weight.shape[0] == target_vocab
        assert model.get_output_embeddings().weight.shape[0] == target_vocab
        print("CPU Resize Confirmed.")

    print("Moving model to CUDA")
    model = model.to("cuda")
    model.gradient_checkpointing_enable()

    return model, tokenizer


def _resolve_data_path(path_str: str) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _expand_repeat_factors(data_paths: list[str], repeat_factors: list[int] | None) -> list[int]:
    if not repeat_factors:
        return [1] * len(data_paths)
    if len(repeat_factors) == 1:
        return [max(1, int(repeat_factors[0]))] * len(data_paths)
    if len(repeat_factors) != len(data_paths):
        raise ValueError(
            f"--data_repeat expects either 1 value or one per --data entry; "
            f"got {len(repeat_factors)} repeats for {len(data_paths)} datasets."
        )
    return [max(1, int(value)) for value in repeat_factors]


def _load_training_dataset(data_paths: list[str], repeat_factors: list[int] | None = None):
    resolved_paths = [_resolve_data_path(path) for path in data_paths]
    expanded_repeats = _expand_repeat_factors(data_paths, repeat_factors)
    datasets = []
    print("Loading dataset(s):")
    for path, repeat in zip(resolved_paths, expanded_repeats):
        ds = load_dataset("json", data_files=path, split="train")
        print(f"  - {path}: {len(ds):,} examples (repeat={repeat}x)")
        datasets.extend([ds] * repeat)

    if len(datasets) == 1:
        return datasets[0]

    merged = concatenate_datasets(datasets)
    print(f"Merged dataset size: {len(merged):,} examples")
    return merged


def train(
    lang: str,
    base_model: str,
    data_paths: list[str],
    output_dir: str,
    num_epochs: int,
    patience: int,
    repeat_factors: list[int] | None = None,
):
    torch.cuda.empty_cache()
    gc.collect()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(base_model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    full_dataset = _load_training_dataset(data_paths, repeat_factors=repeat_factors)
    print(f"Full dataset size: {len(full_dataset):,} examples")

    split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train split: {len(train_dataset):,} | Eval split: {len(eval_dataset):,}")

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        optim="paged_adamw_32bit",
        report_to="none",
        gradient_checkpointing=True,
        warmup_steps=50,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    print(f"Starting {lang.upper()} LoRA training  (epochs={num_epochs}, packing=ON, patience={patience})")
    trainer.train()

    final_dir = f"{output_dir}/final"
    print(f"Saving best model to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Rich-Cousin LoRA adapter")
    parser.add_argument("--lang", required=True, help="Language code (fr / ca)")
    parser.add_argument(
        "--base_model",
        default=DEFAULT_BASE_MODEL,
        help=(
            "Base causal LM checkpoint to train on top of. "
            "Default: models/llama-3.1-occitan-initialized"
        ),
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="One or more JSONL files produced by the data preparation scripts.",
    )
    parser.add_argument(
        "--data_repeat",
        nargs="*",
        type=int,
        default=None,
        help="Optional repeat factors for --data (one value applied to all, or one per dataset).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Where to save checkpoints and final adapter. "
            "Default: checkpoints/lora_<lang>"
        ),
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 2)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience in eval rounds (default: 3)")
    args = parser.parse_args()

    if args.output_dir is None:
        lang = args.lang.strip().lower()
        if lang == "fr":
            args.output_dir = f"{DEFAULT_CHECKPOINTS_ROOT}/lora_fr"
        elif lang == "ca":
            args.output_dir = f"{DEFAULT_CHECKPOINTS_ROOT}/lora_ca"
        else:
            args.output_dir = f"{DEFAULT_CHECKPOINTS_ROOT}/lora_{lang}"
        print(f"No --output_dir provided, defaulting to: {args.output_dir}")

    train(args.lang, args.base_model, args.data, args.output_dir, args.epochs, args.patience, args.data_repeat)
