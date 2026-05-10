"""
Shared evaluation utilities for perplexity, chrF++, and model loading.

Example:
    python -m src.evaluation.score_metrics --help
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import torch
from peft import LoraConfig, PeftModel
from sacrebleu.metrics import CHRF
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.mole.romance_mole import RomanceMoLEModel
from src.training.train_romance_mole_router import _load_base_model, _load_tokenizer

OCCITAN_SYSTEM_ANCHOR = "Sès un assistent d'intelligéncia artificiala que parla unicament en occitan lengadocian.\n\n"


def read_text_lines(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def read_plain_text(path: str | Path) -> str:
    lines = read_text_lines(path)
    non_empty = [line.strip() for line in lines if line.strip()]
    if not non_empty:
        raise ValueError(f"No non-empty text found in {path}.")
    return "\n".join(non_empty)


def read_non_empty_lines(path: str | Path) -> list[str]:
    return [line.strip() for line in read_text_lines(path) if line.strip()]


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _model_dtype_for_device(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def _safe_load_tokenizer(model_ref: str | Path):
    try:
        return AutoTokenizer.from_pretrained(str(model_ref))
    except Exception:
        return AutoTokenizer.from_pretrained(str(model_ref), use_fast=False)


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
        f"[score_metrics] Runtime adapter compat copy: {adapter_dir} -> {tmp_dir} "
        f"(dropped keys: {dropped})"
    )
    return tmp_dir, True


def resolve_model_artifact_dir(model_path: str | Path) -> Path | str:
    path = Path(model_path).expanduser()
    if not path.exists():
        return str(model_path)
    if not path.is_dir():
        return path

    final_dir = path / "final"
    if final_dir.exists() and final_dir.is_dir():
        return final_dir
    return path


def is_full_model_dir(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def is_peft_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists()


def is_mole_router_dir(path: Path) -> bool:
    return (path / "router_config.json").exists() and (path / "router_weights.pt").exists()


def infer_base_model(adapter_dir: Path, explicit_base_model: str | None) -> str:
    if explicit_base_model:
        return explicit_base_model

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise ValueError(
            "Missing adapter_config.json, and no --base-model was provided."
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    inferred = cfg.get("base_model_name_or_path")
    if not inferred:
        raise ValueError(
            f"Could not infer base model from {config_path}. Pass --base-model explicitly."
        )
    return str(inferred)


def detect_adapter_vocab_size(adapter_dir: Path) -> int | None:
    """
    Detect embedding vocab rows saved inside a PEFT adapter.
    Needed when modules_to_save contains resized embed/lm_head weights.
    """
    st_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"

    weights = None
    if st_path.exists():
        weights = load_file(str(st_path))
    elif bin_path.exists():
        weights = torch.load(bin_path, map_location="cpu")
    else:
        return None

    for key, tensor in weights.items():
        if "lora_A" in key or "lora_B" in key:
            continue
        if ("embed_tokens" in key or "lm_head" in key) and getattr(tensor, "ndim", 0) == 2:
            return int(tensor.shape[0])
    return None


def load_full_model_and_tokenizer(model_dir: Path | str, device: str):
    tokenizer = _safe_load_tokenizer(model_dir)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=_model_dtype_for_device(device),
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def load_peft_adapter_model_and_tokenizer(
    adapter_dir: Path,
    device: str,
    base_model: str | None = None,
):
    temp_dirs: list[Path] = []
    try:
        adapter_dir_to_use, is_temp = _sanitize_adapter_runtime_copy(adapter_dir)
        if is_temp:
            temp_dirs.append(adapter_dir_to_use)

        resolved_base_model = infer_base_model(adapter_dir_to_use, base_model)
        tokenizer_source = adapter_dir_to_use if (adapter_dir_to_use / "tokenizer.json").exists() else resolved_base_model
        tokenizer = _safe_load_tokenizer(tokenizer_source)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            resolved_base_model,
            dtype=_model_dtype_for_device(device),
        )
        adapter_vocab = detect_adapter_vocab_size(adapter_dir_to_use)
        if adapter_vocab is not None:
            target_vocab = max(adapter_vocab, len(tokenizer))
            if int(model.config.vocab_size) != int(target_vocab):
                print(
                    f"[score_metrics] Resizing base embeddings before adapter load: "
                    f"{model.config.vocab_size} -> {target_vocab}"
                )
                try:
                    model.resize_token_embeddings(target_vocab, mean_resizing=False)
                except TypeError:
                    model.resize_token_embeddings(target_vocab)

        model = PeftModel.from_pretrained(model, str(adapter_dir_to_use))
        model.to(device)
        model.eval()
        return model, tokenizer, resolved_base_model
    finally:
        for tmp in temp_dirs:
            shutil.rmtree(tmp, ignore_errors=True)


def load_mole_model_and_tokenizer(
    router_dir: Path,
    device: str,
    base_model: str | None,
    adapter_paths: Sequence[str] | None,
):
    if not adapter_paths:
        raise ValueError(
            "MoLE router checkpoints require --adapters so the frozen expert adapters can be loaded."
        )
    if not base_model:
        raise ValueError(
            "MoLE router checkpoints require --base-model so the underlying causal LM can be loaded."
        )

    resolved_adapters = [str(Path(p).expanduser()) for p in adapter_paths]
    resolved_base_model = str(Path(base_model).expanduser()) if str(base_model).startswith("~") else base_model

    temp_dirs: list[Path] = []
    runtime_adapters: list[str] = []
    for path in resolved_adapters:
        compat_path, is_temp = _sanitize_adapter_runtime_copy(Path(path))
        runtime_adapters.append(str(compat_path))
        if is_temp:
            temp_dirs.append(compat_path)

    with open(router_dir / "router_config.json", "r", encoding="utf-8") as handle:
        router_cfg = json.load(handle)

    adapter_names = router_cfg["adapter_names"]
    if len(adapter_names) != len(runtime_adapters):
        raise ValueError(
            f"Router expects {len(adapter_names)} adapters but received {len(runtime_adapters)} via --adapters."
        )

    try:
        tokenizer = _load_tokenizer(runtime_adapters, resolved_base_model)
        base = _load_base_model(resolved_base_model, runtime_adapters, tokenizer)
        mole = RomanceMoLEModel(
            base_model=base,
            adapter_paths=runtime_adapters,
            adapter_names=adapter_names,
            sequence_route_threshold=router_cfg.get("sequence_route_threshold"),
            router_aux_loss_coef=0.0,
            router_temperature=float(router_cfg.get("router_temperature", 1.0)),
            hard_router_argmax=bool(router_cfg.get("hard_router_argmax", False)),
        )
        mole.routers.load_state_dict(torch.load(router_dir / "router_weights.pt", map_location="cpu"))
        mole = mole.to(_model_dtype_for_device(device)).to(device)
        mole.eval()
        return mole, tokenizer, resolved_base_model
    finally:
        for tmp in temp_dirs:
            shutil.rmtree(tmp, ignore_errors=True)


def load_model_and_tokenizer(
    model_path: str,
    device: str | None = None,
    base_model: str | None = None,
    adapter_paths: Sequence[str] | None = None,
):
    device = resolve_device(device)
    resolved = resolve_model_artifact_dir(model_path)

    if isinstance(resolved, Path):
        if is_mole_router_dir(resolved):
            model, tokenizer, resolved_base_model = load_mole_model_and_tokenizer(
                router_dir=resolved,
                device=device,
                base_model=base_model,
                adapter_paths=adapter_paths,
            )
            return model, tokenizer, device, {
                "loader_type": "mole_router",
                "resolved_path": str(resolved),
                "base_model": resolved_base_model,
                "adapters": list(adapter_paths or []),
            }
        if is_peft_adapter_dir(resolved):
            model, tokenizer, resolved_base_model = load_peft_adapter_model_and_tokenizer(
                adapter_dir=resolved,
                device=device,
                base_model=base_model,
            )
            return model, tokenizer, device, {
                "loader_type": "peft_adapter",
                "resolved_path": str(resolved),
                "base_model": resolved_base_model,
            }
        if is_full_model_dir(resolved):
            model, tokenizer = load_full_model_and_tokenizer(resolved, device=device)
            return model, tokenizer, device, {
                "loader_type": "full_model",
                "resolved_path": str(resolved),
            }

    model, tokenizer = load_full_model_and_tokenizer(str(resolved), device=device)
    return model, tokenizer, device, {
        "loader_type": "full_model",
        "resolved_path": str(resolved),
    }


def _resolve_max_length(model, tokenizer) -> int:
    max_length = getattr(model.config, "max_position_embeddings", None)
    if max_length is None or max_length <= 0:
        max_length = getattr(tokenizer, "model_max_length", 1024)
    if max_length is None or max_length > 100000:
        max_length = 1024
    return int(max_length)


def _build_prefixed_line_corpus(
    tokenizer,
    test_file_path: str,
    prompt_prefix: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build one concatenated token stream where every non-empty line is prefixed."""
    lines = read_non_empty_lines(test_file_path)
    if not lines:
        raise ValueError(f"No non-empty lines found in {test_file_path}.")

    prefix_ids = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
    if not prefix_ids:
        raise ValueError("Prompt prefix tokenized to zero tokens.")

    separator_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]

    all_ids: list[int] = []
    all_masked: list[bool] = []
    for idx, line in enumerate(lines):
        line_ids = tokenizer(line, add_special_tokens=False)["input_ids"]
        all_ids.extend(prefix_ids)
        all_masked.extend([True] * len(prefix_ids))

        all_ids.extend(line_ids)
        all_masked.extend([False] * len(line_ids))

        if idx < len(lines) - 1 and separator_ids:
            all_ids.extend(separator_ids)
            # Separator is synthetic formatting, so exclude it from scored loss.
            all_masked.extend([True] * len(separator_ids))

    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
    masked_positions = torch.tensor([all_masked], dtype=torch.bool, device=device)
    return input_ids, masked_positions, len(lines)


def _init_routing_accumulator(model) -> dict | None:
    if not hasattr(model, "adapter_names") or not hasattr(model, "last_gates"):
        return None
    expert_names = [str(x) for x in getattr(model, "adapter_names", [])]
    if not expert_names:
        return None
    router_modes = dict(getattr(model, "router_modes", {}) or {})
    return {
        "expert_names": expert_names,
        "gate_sums": [0.0 for _ in expert_names],
        "event_count": 0,
        "layers_observed": 0,
        "router_modes": router_modes,
        "per_layer": {},
    }


def _accumulate_routing_stats(model, acc: dict | None) -> None:
    if acc is None:
        return
    gates_by_layer = getattr(model, "last_gates", None)
    if not isinstance(gates_by_layer, dict) or not gates_by_layer:
        return

    for layer_key, gates in gates_by_layer.items():
        if gates is None or getattr(gates, "ndim", 0) != 3:
            continue
        g = gates.detach().float()
        # g shape: [B, S, E]
        if int(g.shape[-1]) != len(acc["expert_names"]):
            continue
        token_events = int(g.shape[0]) * int(g.shape[1])
        if token_events <= 0:
            continue

        layer_entry = acc["per_layer"].setdefault(
            layer_key,
            {
                "gate_sums": [0.0 for _ in acc["expert_names"]],
                "event_count": 0,
                "mode": acc["router_modes"].get(layer_key, "unknown"),
            },
        )

        for i in range(len(acc["expert_names"])):
            gate_sum = float(g[..., i].sum().item())
            acc["gate_sums"][i] += gate_sum
            layer_entry["gate_sums"][i] += gate_sum

        layer_entry["event_count"] += token_events
        acc["event_count"] += token_events
        acc["layers_observed"] += 1


def _finalize_routing_stats(acc: dict | None) -> dict | None:
    if acc is None or acc["event_count"] <= 0:
        return None
    averages = {
        name: gate_sum / acc["event_count"]
        for name, gate_sum in zip(acc["expert_names"], acc["gate_sums"])
    }
    dominant_expert = max(averages, key=averages.get)
    dominant_weight = averages[dominant_expert]

    per_layer_expert_average_gates = {}
    per_layer_dominant_expert = {}
    per_mode_acc = {}

    for raw_layer_key, layer_data in acc["per_layer"].items():
        layer_events = int(layer_data["event_count"])
        if layer_events <= 0:
            continue
        layer_name = str(raw_layer_key).replace("__", ".")
        layer_avg = {
            name: gate_sum / layer_events
            for name, gate_sum in zip(acc["expert_names"], layer_data["gate_sums"])
        }
        per_layer_expert_average_gates[layer_name] = layer_avg
        layer_dom = max(layer_avg, key=layer_avg.get)
        per_layer_dominant_expert[layer_name] = {
            "expert": layer_dom,
            "weight": layer_avg[layer_dom],
            "mode": layer_data.get("mode", "unknown"),
        }

        mode = layer_data.get("mode", "unknown")
        mode_entry = per_mode_acc.setdefault(
            mode,
            {
                "gate_sums": [0.0 for _ in acc["expert_names"]],
                "event_count": 0,
            },
        )
        for i, val in enumerate(layer_data["gate_sums"]):
            mode_entry["gate_sums"][i] += val
        mode_entry["event_count"] += layer_events

    per_mode_expert_average_gates = {}
    for mode, mode_data in per_mode_acc.items():
        events = int(mode_data["event_count"])
        if events <= 0:
            continue
        per_mode_expert_average_gates[mode] = {
            name: gate_sum / events
            for name, gate_sum in zip(acc["expert_names"], mode_data["gate_sums"])
        }

    return {
        "expert_average_gates": averages,
        "dominant_expert": dominant_expert,
        "dominant_weight": dominant_weight,
        "router_collapse_suspected": dominant_weight >= 0.80,
        "token_events": acc["event_count"],
        "layers_observed": acc["layers_observed"],
        "per_mode_expert_average_gates": per_mode_expert_average_gates,
        "per_layer_expert_average_gates": per_layer_expert_average_gates,
        "per_layer_dominant_expert": per_layer_dominant_expert,
    }


def calculate_perplexity_linewise(
    model,
    tokenizer,
    test_file_path: str,
    prompt_prefix: str | None = None,
    score_only_after_prefix: bool = False,
) -> dict:
    lines = [line.strip() for line in read_text_lines(test_file_path) if line.strip()]
    if not lines:
        raise ValueError(f"No non-empty lines found in {test_file_path}.")

    max_length = _resolve_max_length(model, tokenizer)
    routing_acc = _init_routing_accumulator(model)

    total_nll = 0.0
    total_loss_tokens = 0
    total_input_tokens = 0

    prefix = prompt_prefix or ""
    prefix_ids = None
    prefix_len = 0
    if prefix:
        prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        prefix_len = int(prefix_ids.shape[1])

    for line in lines:
        full_text = f"{prefix}{line}" if prefix else line
        encoded = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"].to(model.device)

        labels = encoded.clone()
        if score_only_after_prefix:
            if not prefix:
                raise ValueError("score_only_after_prefix=True requires a non-empty prompt prefix.")
            effective_prefix = min(prefix_len, int(encoded.shape[1]))
            labels[:, :effective_prefix] = -100

        with torch.no_grad():
            outputs = model(encoded, labels=labels)

        _accumulate_routing_stats(model, routing_acc)

        loss_token_count = int((labels[:, 1:] != -100).sum().item())
        if loss_token_count > 0:
            total_nll += float(outputs.loss.item()) * loss_token_count
            total_loss_tokens += loss_token_count
        total_input_tokens += int(encoded.shape[1])

    if total_loss_tokens == 0:
        raise ValueError("Linewise perplexity produced zero valid loss tokens.")

    avg_loss = total_nll / total_loss_tokens
    return {
        "mode": "linewise",
        "num_lines": len(lines),
        "max_length": max_length,
        "prompt_prefix_used": bool(prefix),
        "score_only_after_prefix": bool(score_only_after_prefix),
        "total_input_tokens": total_input_tokens,
        "total_loss_tokens": total_loss_tokens,
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "routing_summary": _finalize_routing_stats(routing_acc),
    }


def calculate_perplexity(
    model_path: str,
    test_file_path: str,
    stride: int = 512,
    device: str | None = None,
    base_model: str | None = None,
    adapter_paths: Sequence[str] | None = None,
    linewise: bool = False,
    prompt_prefix: str | None = None,
    score_only_after_prefix: bool = False,
    prefix_each_line: bool = False,
) -> dict:
    """
    Calculate average loss and perplexity over a plain-text corpus.

    Uses a sliding-window evaluation strategy so long files can be processed
    without truncating the full corpus.
    """
    model, tokenizer, device, load_info = load_model_and_tokenizer(
        model_path,
        device=device,
        base_model=base_model,
        adapter_paths=adapter_paths,
    )

    if linewise:
        linewise_result = calculate_perplexity_linewise(
            model=model,
            tokenizer=tokenizer,
            test_file_path=test_file_path,
            prompt_prefix=prompt_prefix,
            score_only_after_prefix=score_only_after_prefix,
        )
        return {
            "metric": "perplexity",
            "model_path": model_path,
            "test_file_path": str(test_file_path),
            "device": device,
            "stride": int(stride),
            "loader_type": load_info["loader_type"],
            "resolved_path": load_info["resolved_path"],
            "base_model": load_info.get("base_model"),
            "adapters": load_info.get("adapters"),
            **linewise_result,
        }

    prefix = prompt_prefix or ""
    repeated_prefix_mask = None
    line_count = None
    if prefix_each_line:
        if not prefix:
            raise ValueError("prefix_each_line=True requires a non-empty prompt prefix.")
        input_ids, repeated_prefix_mask, line_count = _build_prefixed_line_corpus(
            tokenizer=tokenizer,
            test_file_path=test_file_path,
            prompt_prefix=prefix,
            device=device,
        )
    else:
        text = read_plain_text(test_file_path)
        full_text = f"{prefix}{text}" if prefix else text
        encodings = tokenizer(full_text, return_tensors="pt")
        input_ids = encodings["input_ids"].to(device)

    max_length = _resolve_max_length(model, tokenizer)

    routing_acc = _init_routing_accumulator(model)

    prefix_token_count = 0
    if prefix:
        prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        prefix_token_count = int(prefix_ids.shape[1])

    total_nll = 0.0
    total_loss_tokens = 0
    total_input_tokens = int(input_ids.size(1))
    previous_end = 0

    for begin in range(0, total_input_tokens, stride):
        end = min(begin + max_length, total_input_tokens)
        target_length = end - previous_end

        input_ids_chunk = input_ids[:, begin:end]
        labels = input_ids_chunk.clone()
        if target_length < input_ids_chunk.size(1):
            labels[:, :-target_length] = -100

        if score_only_after_prefix:
            chunk_start = begin
            chunk_end = end

            if prefix_each_line and repeated_prefix_mask is not None:
                local_mask = repeated_prefix_mask[:, chunk_start:chunk_end]
                labels = labels.masked_fill(local_mask, -100)
            elif prefix_token_count > 0:
                overlap_start = max(chunk_start, 0)
                overlap_end = min(chunk_end, prefix_token_count)
                if overlap_end > overlap_start:
                    local_start = overlap_start - chunk_start
                    local_end = overlap_end - chunk_start
                    labels[:, local_start:local_end] = -100

        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=labels)

        _accumulate_routing_stats(model, routing_acc)

        loss_token_count = int((labels[:, 1:] != -100).sum().item())
        if loss_token_count > 0:
            total_nll += float(outputs.loss.item()) * loss_token_count
            total_loss_tokens += loss_token_count

        previous_end = end
        if end >= total_input_tokens:
            break

    if total_loss_tokens == 0:
        raise ValueError("Perplexity evaluation produced zero valid loss tokens.")

    average_loss = total_nll / total_loss_tokens
    perplexity = math.exp(average_loss)

    return {
        "metric": "perplexity",
        "model_path": model_path,
        "test_file_path": str(test_file_path),
        "device": device,
        "max_length": int(max_length),
        "stride": int(stride),
        "total_input_tokens": total_input_tokens,
        "total_loss_tokens": total_loss_tokens,
        "loss": average_loss,
        "perplexity": perplexity,
        "loader_type": load_info["loader_type"],
        "resolved_path": load_info["resolved_path"],
        "base_model": load_info.get("base_model"),
        "adapters": load_info.get("adapters"),
        "mode": "sliding_window",
        "prompt_prefix_used": bool(prefix),
        "score_only_after_prefix": bool(score_only_after_prefix),
        "prefix_each_line": bool(prefix_each_line),
        "num_lines": line_count,
        "routing_summary": _finalize_routing_stats(routing_acc),
    }


def score_suffix_given_prefix(
    model,
    tokenizer,
    prefix: str,
    suffix: str,
    device: str | None = None,
) -> dict:
    """
    Score only the suffix tokens conditioned on a shared prefix.

    This isolates the grammatical choice in minimal-pair evaluations so the
    long shared prefix does not dilute the probability difference.
    """
    device = resolve_device(device or str(model.device))
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    full_ids = tokenizer(prefix + suffix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    prefix_len = int(prefix_ids.size(1))
    full_len = int(full_ids.size(1))
    suffix_len = full_len - prefix_len
    if suffix_len <= 0:
        raise ValueError("Suffix added zero new tokens; choose a suffix with distinct tokenization.")

    labels = full_ids.clone()
    labels[:, :prefix_len] = -100

    with torch.no_grad():
        outputs = model(full_ids, labels=labels)

    scored_suffix_tokens = int((labels[:, 1:] != -100).sum().item())
    if scored_suffix_tokens <= 0:
        raise ValueError("No suffix tokens were scored after prefix masking.")

    total_nll = float(outputs.loss.item()) * scored_suffix_tokens
    average_suffix_loss = total_nll / scored_suffix_tokens

    return {
        "prefix_token_count": prefix_len,
        "full_token_count": full_len,
        "suffix_token_count": suffix_len,
        "scored_suffix_tokens": scored_suffix_tokens,
        "suffix_loss": average_suffix_loss,
        "suffix_nll": total_nll,
        "suffix_perplexity": math.exp(average_suffix_loss),
    }


def calculate_minimal_pair(
    model_path: str,
    prefix: str,
    correct_suffix: str,
    incorrect_suffix: str,
    device: str | None = None,
    base_model: str | None = None,
    adapter_paths: Sequence[str] | None = None,
) -> dict:
    """
    Compare two suffixes under the same prefix and score only the suffix tokens.
    """
    model, tokenizer, device, load_info = load_model_and_tokenizer(
        model_path,
        device=device,
        base_model=base_model,
        adapter_paths=adapter_paths,
    )

    correct = score_suffix_given_prefix(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        suffix=correct_suffix,
        device=device,
    )
    incorrect = score_suffix_given_prefix(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        suffix=incorrect_suffix,
        device=device,
    )

    return {
        "metric": "minimal_pair",
        "model_path": model_path,
        "device": device,
        "prefix": prefix,
        "correct_suffix": correct_suffix,
        "incorrect_suffix": incorrect_suffix,
        "correct": correct,
        "incorrect": incorrect,
        "preferred": "correct" if correct["suffix_loss"] < incorrect["suffix_loss"] else "incorrect",
        "loss_margin": incorrect["suffix_loss"] - correct["suffix_loss"],
        "ppl_margin": incorrect["suffix_perplexity"] - correct["suffix_perplexity"],
        "loader_type": load_info["loader_type"],
        "resolved_path": load_info["resolved_path"],
        "base_model": load_info.get("base_model"),
        "adapters": load_info.get("adapters"),
    }


def calculate_chrf(predictions_file: str, references_file: str) -> dict:
    """
    Compute corpus chrF++ with sacrebleu.
    """
    predictions = read_text_lines(predictions_file)
    references = read_text_lines(references_file)

    if len(predictions) != len(references):
        raise ValueError(
            "Prediction and reference files must have the same number of lines: "
            f"{len(predictions)} vs {len(references)}."
        )
    if not predictions:
        raise ValueError("Prediction/reference files are empty.")

    chrf = CHRF(word_order=2)
    score = chrf.corpus_score(predictions, [references])

    return {
        "metric": "chrf++",
        "predictions_file": str(predictions_file),
        "references_file": str(references_file),
        "num_sentences": len(predictions),
        "score": score.score,
        "signature": str(score),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute perplexity or chrF++ for thesis evaluation."
    )
    parser.add_argument(
        "--metric",
        required=True,
        choices=["ppl", "perplexity", "chrf", "chrf++", "minimal_pair", "minpair"],
        help="Metric to compute.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model path or Hugging Face model id for perplexity.",
    )
    parser.add_argument(
        "--test-file-path",
        type=str,
        default=None,
        help="Plain text file for perplexity evaluation.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding-window stride for perplexity (default: 512).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for perplexity, e.g. cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model path or HF id required when --model-path is a PEFT adapter or MoLE router checkpoint.",
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=None,
        help="Frozen expert adapter paths required when --model-path points to a MoLE router checkpoint.",
    )
    parser.add_argument(
        "--linewise",
        action="store_true",
        help="Compute perplexity per line instead of concatenated sliding-window corpus mode.",
    )
    parser.add_argument(
        "--prompt-prefix",
        type=str,
        default=None,
        help="Optional prompt prefix prepended to each line in linewise mode.",
    )
    parser.add_argument(
        "--use-occitan-anchor",
        action="store_true",
        help="Use the exact Occitan system anchor as --prompt-prefix.",
    )
    parser.add_argument(
        "--score-only-after-prefix",
        action="store_true",
        help="When using a prefix, mask prefix tokens and score only the continuation tokens.",
    )
    parser.add_argument(
        "--prefix-each-line",
        action="store_true",
        help="In sliding-window mode, prepend the prompt prefix to every non-empty line before concatenation.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Shared prefix for minimal-pair suffix scoring.",
    )
    parser.add_argument(
        "--correct-suffix",
        type=str,
        default=None,
        help="Correct suffix for minimal-pair scoring.",
    )
    parser.add_argument(
        "--incorrect-suffix",
        type=str,
        default=None,
        help="Incorrect suffix for minimal-pair scoring.",
    )
    parser.add_argument(
        "--preds",
        type=str,
        default=None,
        help="Predictions text file for chrF++.",
    )
    parser.add_argument(
        "--refs",
        type=str,
        default=None,
        help="Reference text file for chrF++.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    metric = args.metric.lower()
    if metric in {"ppl", "perplexity"}:
        if not args.model_path or not args.test_file_path:
            parser.error("--model-path and --test-file-path are required for perplexity.")
        prompt_prefix = args.prompt_prefix
        if args.use_occitan_anchor:
            if prompt_prefix is not None and prompt_prefix != OCCITAN_SYSTEM_ANCHOR:
                parser.error("--use-occitan-anchor conflicts with a different --prompt-prefix value.")
            prompt_prefix = OCCITAN_SYSTEM_ANCHOR
        result = calculate_perplexity(
            model_path=args.model_path,
            test_file_path=args.test_file_path,
            stride=args.stride,
            device=args.device,
            base_model=args.base_model,
            adapter_paths=args.adapters,
            linewise=args.linewise,
            prompt_prefix=prompt_prefix,
            score_only_after_prefix=args.score_only_after_prefix,
            prefix_each_line=args.prefix_each_line,
        )
        print(f"Loss:       {result['loss']:.6f}")
        print(f"Perplexity: {result['perplexity']:.6f}")
        if result.get("routing_summary"):
            rs = result["routing_summary"]
            print(f"Routing dominant expert: {rs['dominant_expert']} ({rs['dominant_weight']:.4f})")
            if rs.get("router_collapse_suspected"):
                print("Routing warning: potential router collapse detected (dominant weight >= 0.80).")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if metric in {"chrf", "chrf++"}:
        if not args.preds or not args.refs:
            parser.error("--preds and --refs are required for chrF++.")
        result = calculate_chrf(args.preds, args.refs)
        print(f"chrF++:     {result['score']:.2f}")
        print(f"Signature:  {result['signature']}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if metric in {"minimal_pair", "minpair"}:
        if not args.model_path or args.prefix is None or args.correct_suffix is None or args.incorrect_suffix is None:
            parser.error(
                "--model-path, --prefix, --correct-suffix, and --incorrect-suffix "
                "are required for minimal-pair scoring."
            )
        result = calculate_minimal_pair(
            model_path=args.model_path,
            prefix=args.prefix,
            correct_suffix=args.correct_suffix,
            incorrect_suffix=args.incorrect_suffix,
            device=args.device,
            base_model=args.base_model,
            adapter_paths=args.adapters,
        )
        print(f"Preferred:        {result['preferred']}")
        print(f"Loss margin:      {result['loss_margin']:.6f}")
        print(f"Correct loss:     {result['correct']['suffix_loss']:.6f}")
        print(f"Incorrect loss:   {result['incorrect']['suffix_loss']:.6f}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    parser.error(f"Unsupported metric: {args.metric}")


if __name__ == "__main__":
    main()
