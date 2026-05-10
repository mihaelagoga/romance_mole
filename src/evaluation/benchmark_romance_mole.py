"""
Benchmark Romance-MoLE against the single Occitan experts.

Example:
    python -m src.evaluation.benchmark_romance_mole --mole-model checkpoints/router_ablation_ladder/t8_aux001_sup002_cs000/final --mole-base models/llama-3.1-occitan-initialized --mole-adapters checkpoints/lora_fr/final checkpoints/lora_ca/final_compat checkpoints/occitan_3b2/final_compat --router-temperature 0.5 --output-dir results/benchmark2
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from sacrebleu.metrics import CHRF
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.mole.romance_mole import RomanceMoLEModel



DEFAULT_PROMPT_OC = (
    "### Instruction:\n"
    "Traduís en occitan lengadocian la frasa francesa seguenta.\n\n"
    "### Input:\n"
    "{source}\n\n"
    "### Response:\n"
)

DEFAULT_PROMPT_CA = (
    "### Instruction:\n"
    "Tradueix al català la frase francesa següent.\n\n"
    "### Input:\n"
    "{source}\n\n"
    "### Response:\n"
)

DEFAULT_PROMPT_FR = (
    "### Instruction:\n"
    "Traduisez la phrase suivante en français.\n\n"
    "### Input:\n"
    "{source}\n\n"
    "### Response:\n"
)


def read_text_lines(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def read_non_empty_lines(path: str | Path) -> list[str]:
    return [line.strip() for line in read_text_lines(path) if line.strip()]


def read_eval_corpus(path: str | Path) -> str:
    """Load held-out monolingual text. Supports .txt and .jsonl (text field)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        chunks: list[str] = []
        with open(p, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        if not chunks:
            raise ValueError(f"No usable 'text' fields in {path}")
        return "\n\n".join(chunks)

    chunks = [line.strip() for line in read_text_lines(p) if line.strip()]
    if not chunks:
        raise ValueError(f"No non-empty lines in {path}")
    return "\n\n".join(chunks)


def _resolve_artifact_dir(model_path: str | Path) -> Path | str:
    p = Path(model_path).expanduser()
    if not p.exists():
        return str(model_path)
    if p.is_dir() and (p / "final").is_dir():
        return p / "final"
    return p


def _is_peft_adapter_dir(p: Path) -> bool:
    return (p / "adapter_config.json").exists() and (p / "adapter_model.safetensors").exists()


def _is_full_model_dir(p: Path) -> bool:
    return any(
        (p / name).exists()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def _is_mole_router_dir(p: Path) -> bool:
    return (p / "router_config.json").exists() and (p / "router_weights.pt").exists()


def _model_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def _ensure_pad(tokenizer):
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _detect_adapter_vocab_size(adapter_dir: Path) -> int | None:
    st_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
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


def _load_full_or_peft(model_path: str, base_model: str | None, device: str):
    resolved = _resolve_artifact_dir(model_path)
    dtype = _model_dtype(device)

    if isinstance(resolved, Path):
        if _is_peft_adapter_dir(resolved):
            cfg_path = resolved / "adapter_config.json"
            with open(cfg_path, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)
            inferred_base = base_model or cfg.get("base_model_name_or_path")
            if not inferred_base:
                raise ValueError(
                    f"PEFT adapter {resolved} has no base_model_name_or_path; pass --*-base."
                )
            tokenizer_source = (
                resolved if (resolved / "tokenizer.json").exists() else inferred_base
            )
            tokenizer = _ensure_pad(AutoTokenizer.from_pretrained(str(tokenizer_source)))
            base = AutoModelForCausalLM.from_pretrained(inferred_base, dtype=dtype)
            adapter_vocab = _detect_adapter_vocab_size(resolved)
            target_vocab = max(adapter_vocab or 0, len(tokenizer))
            if int(base.config.vocab_size) != int(target_vocab):
                try:
                    base.resize_token_embeddings(target_vocab, mean_resizing=False)
                except TypeError:
                    base.resize_token_embeddings(target_vocab)
            model = PeftModel.from_pretrained(base, str(resolved))
            model.to(device)
            model.eval()
            return model, tokenizer, {"loader": "peft_adapter", "base": inferred_base, "path": str(resolved)}

        if _is_full_model_dir(resolved):
            tokenizer = _ensure_pad(AutoTokenizer.from_pretrained(str(resolved)))
            model = AutoModelForCausalLM.from_pretrained(str(resolved), dtype=dtype)
            model.to(device)
            model.eval()
            return model, tokenizer, {"loader": "full_model", "base": None, "path": str(resolved)}

    tokenizer = _ensure_pad(AutoTokenizer.from_pretrained(str(resolved)))
    model = AutoModelForCausalLM.from_pretrained(str(resolved), dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, {"loader": "full_model", "base": None, "path": str(resolved)}


def load_mole(
    router_dir: str,
    base_model: str,
    adapter_paths: list[str],
    router_temperature: float,
    device: str,
):
    router_dir_p = Path(router_dir).expanduser()
    if not _is_mole_router_dir(router_dir_p):
        raise ValueError(
            f"--mole-model {router_dir} is not a MoLE router directory "
            "(missing router_config.json or router_weights.pt)."
        )
    if not adapter_paths or len(adapter_paths) < 2:
        raise ValueError("--mole-adapters requires >= 2 frozen expert adapter paths.")

    with open(router_dir_p / "router_config.json", "r", encoding="utf-8") as handle:
        router_cfg = json.load(handle)
    adapter_names = router_cfg["adapter_names"]
    if len(adapter_names) != len(adapter_paths):
        raise ValueError(
            f"Router expects {len(adapter_names)} adapters but received "
            f"{len(adapter_paths)} via --mole-adapters."
        )

    adapter_paths_resolved = [str(Path(p).expanduser()) for p in adapter_paths]

    tokenizer = None
    for path in adapter_paths_resolved:
        try:
            tokenizer = _ensure_pad(AutoTokenizer.from_pretrained(path))
            break
        except Exception:
            continue
    if tokenizer is None:
        tokenizer = _ensure_pad(AutoTokenizer.from_pretrained(base_model))

    target_vocab = _detect_adapter_vocab_size(Path(adapter_paths_resolved[0]))
    if not target_vocab:
        target_vocab = math.ceil(len(tokenizer) / 64) * 64

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if int(base.config.vocab_size) != int(target_vocab):
        base.resize_token_embeddings(target_vocab, mean_resizing=False)

    mole = RomanceMoLEModel(
        base_model=base,
        adapter_paths=adapter_paths_resolved,
        adapter_names=adapter_names,
        sequence_route_threshold=router_cfg.get("sequence_route_threshold"),
        router_aux_loss_coef=0.0,
        router_temperature=float(router_cfg.get("router_temperature", router_temperature)),
        hard_router_argmax=bool(router_cfg.get("hard_router_argmax", False)),
    )
    mole.router_temperature = float(router_temperature)
    mole.routers.load_state_dict(torch.load(router_dir_p / "router_weights.pt", map_location="cpu"))
    mole = mole.to(_model_dtype(device)).to(device)
    mole.eval()

    return mole, tokenizer, {
        "loader": "mole_router",
        "base": base_model,
        "path": str(router_dir_p),
        "adapters": adapter_paths_resolved,
        "adapter_names": adapter_names,
        "router_temperature": float(router_temperature),
    }


def greedy_generate(
    model,
    tokenizer,
    source_lines: list[str],
    prompt_template: str,
    max_new_tokens: int,
    device: str,
    log_every: int = 25,
) -> list[str]:
    predictions: list[str] = []
    for idx, source in enumerate(source_lines, start=1):
        prompt = prompt_template.format(source=source)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = outputs[0][input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().replace("\n", " ")
        predictions.append(text)
        if idx % log_every == 0:
            print(f"    generated {idx}/{len(source_lines)}")
    return predictions


def compute_chrf(predictions: list[str], references: list[str]) -> dict:
    if len(predictions) != len(references):
        raise ValueError(
            f"prediction/reference length mismatch: {len(predictions)} vs {len(references)}"
        )
    chrf = CHRF(word_order=2)
    score = chrf.corpus_score(predictions, [references])
    return {
        "metric": "chrF++",
        "score": float(score.score),
        "signature": str(score),
        "num_sentences": len(predictions),
    }


def bootstrap_chrf_diff(
    left_predictions: list[str],
    right_predictions: list[str],
    references: list[str],
    *,
    left_name: str,
    right_name: str,
    num_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap confidence interval for chrF++ difference.

    Resamples sentence indices with replacement and computes:
      diff = chrF(right) - chrF(left)
    """
    if not (len(left_predictions) == len(right_predictions) == len(references)):
        raise ValueError(
            "bootstrap_chrf_diff length mismatch: "
            f"{len(left_predictions)} vs {len(right_predictions)} vs {len(references)}"
        )
    n = len(references)
    if n == 0:
        raise ValueError("bootstrap_chrf_diff received empty inputs.")
    if num_samples <= 0:
        raise ValueError("--bootstrap-samples must be > 0.")

    left_obs = compute_chrf(left_predictions, references)["score"]
    right_obs = compute_chrf(right_predictions, references)["score"]
    observed_diff = float(right_obs - left_obs)

    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(num_samples):
        idxs = [rng.randrange(n) for _ in range(n)]
        left_sample = [left_predictions[i] for i in idxs]
        right_sample = [right_predictions[i] for i in idxs]
        ref_sample = [references[i] for i in idxs]
        left_s = compute_chrf(left_sample, ref_sample)["score"]
        right_s = compute_chrf(right_sample, ref_sample)["score"]
        diffs.append(float(right_s - left_s))

    diffs_sorted = sorted(diffs)
    lo_idx = int(0.025 * (num_samples - 1))
    hi_idx = int(0.975 * (num_samples - 1))
    ci_low = float(diffs_sorted[lo_idx])
    ci_high = float(diffs_sorted[hi_idx])

    count_le_zero = sum(1 for d in diffs if d <= 0.0)
    count_ge_zero = sum(1 for d in diffs if d >= 0.0)
    p_two_sided = 2.0 * min(
        (count_le_zero + 1) / (num_samples + 1),
        (count_ge_zero + 1) / (num_samples + 1),
    )
    p_two_sided = float(min(1.0, p_two_sided))

    return {
        "metric": "bootstrap_chrF_diff",
        "left_model": left_name,
        "right_model": right_name,
        "left_score": float(left_obs),
        "right_score": float(right_obs),
        "observed_diff_right_minus_left": observed_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value_two_sided": p_two_sided,
        "significant_0p05": bool(p_two_sided < 0.05),
        "num_sentences": n,
        "num_samples": int(num_samples),
        "seed": int(seed),
    }


def _init_routing_tracker(model) -> dict | None:
    gates = getattr(model, "last_gates", None)
    if not isinstance(gates, dict):
        return None
    expert_names = list(getattr(model, "adapter_names", []))
    return {
        "expert_names": expert_names,
        "sum_gates": None,  # torch.Tensor [E]
        "num_positions": 0,
        "num_layers_observed": 0,
    }


def _accumulate_routing_from_model(model, tracker: dict | None) -> None:
    if tracker is None:
        return
    gates = getattr(model, "last_gates", None)
    if not isinstance(gates, dict) or not gates:
        return
    for gate in gates.values():
        if not torch.is_tensor(gate) or gate.ndim < 2:
            continue
        flat = gate.detach().float().reshape(-1, gate.shape[-1]).cpu()
        if flat.numel() == 0:
            continue
        gate_sum = flat.sum(dim=0)
        tracker["sum_gates"] = gate_sum if tracker["sum_gates"] is None else tracker["sum_gates"] + gate_sum
        tracker["num_positions"] += int(flat.shape[0])
        tracker["num_layers_observed"] += 1


def _finalize_routing_tracker(tracker: dict | None) -> dict | None:
    if not tracker or tracker["sum_gates"] is None or tracker["num_positions"] <= 0:
        return None
    avg = (tracker["sum_gates"] / float(tracker["num_positions"])).tolist()
    expert_names = tracker["expert_names"] or [f"expert_{i}" for i in range(len(avg))]
    if len(expert_names) < len(avg):
        expert_names = expert_names + [f"expert_{i}" for i in range(len(expert_names), len(avg))]
    pairs = list(zip(expert_names[: len(avg)], avg))
    pairs_sorted = sorted(pairs, key=lambda kv: kv[1], reverse=True)
    return {
        "expert_average_gates": {name: float(val) for name, val in pairs},
        "expert_ranking": [{"expert": name, "average_gate": float(val)} for name, val in pairs_sorted],
        "num_positions": int(tracker["num_positions"]),
        "num_layers_observed": int(tracker["num_layers_observed"]),
    }


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    device: str,
    stride: int = 512,
    max_length: int | None = None,
) -> dict:
    """Sliding-window causal-LM perplexity plus tokenizer-invariant."""
    total_chars = len(text)
    total_bytes = len(text.encode("utf-8"))

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    total_tokens = int(input_ids.size(1))
    if total_tokens < 2:
        raise ValueError("Eval text tokenized to <2 tokens.")

    if max_length is None or max_length <= 0:
        max_length = 2048
    model_cap = getattr(model.config, "max_position_embeddings", None) or max_length
    if model_cap > 0:
        max_length = min(int(max_length), int(model_cap))

    total_nll = 0.0
    total_loss_tokens = 0
    previous_end = 0

    for begin in range(0, total_tokens, stride):
        end = min(begin + max_length, total_tokens)
        target_length = end - previous_end
        chunk = input_ids[:, begin:end]
        labels = chunk.clone()
        if target_length < chunk.size(1):
            labels[:, :-target_length] = -100
        with torch.no_grad():
            out = model(chunk, labels=labels)
        loss_tokens = int((labels[:, 1:] != -100).sum().item())
        if loss_tokens > 0:
            total_nll += float(out.loss.item()) * loss_tokens
            total_loss_tokens += loss_tokens
        previous_end = end
        if end >= total_tokens:
            break

    if total_loss_tokens == 0:
        raise ValueError("Perplexity computation produced zero scored tokens.")

    avg_loss = total_nll / total_loss_tokens
    total_nll_bits = total_nll / math.log(2)
    bits_per_char = total_nll_bits / total_chars if total_chars > 0 else None
    bits_per_byte = total_nll_bits / total_bytes if total_bytes > 0 else None
    return {
        "metric": "perplexity",
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "bits_per_char": bits_per_char,
        "bits_per_byte": bits_per_byte,
        "scored_tokens": total_loss_tokens,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "stride": stride,
        "max_length": max_length,
    }


def compute_conditional_perplexity(
    model,
    tokenizer,
    sources: list[str],
    references: list[str],
    prompt_template: str,
    device: str,
    max_length: int | None = None,
    collect_routing: bool = False,
) -> dict:
    """Perplexity of the reference *conditioned* on an instruction-formatted prompt."""
    if len(sources) != len(references):
        raise ValueError(
            f"sources/references length mismatch: {len(sources)} vs {len(references)}"
        )
    if not sources:
        raise ValueError("No (source, reference) pairs provided.")

    if max_length is None or max_length <= 0:
        max_length = 2048
    model_cap = getattr(model.config, "max_position_embeddings", None) or max_length
    if model_cap > 0:
        max_length = min(int(max_length), int(model_cap))

    total_nll = 0.0
    total_scored_tokens = 0
    total_ref_chars = 0
    total_ref_bytes = 0
    num_pairs_used = 0
    num_pairs_skipped = 0
    routing_tracker = _init_routing_tracker(model) if collect_routing else None

    for source, reference in zip(sources, references):
        if not reference.strip():
            num_pairs_skipped += 1
            continue

        prompt = prompt_template.format(source=source)
        full_text = prompt + reference

        full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = int(prompt_ids.size(1))

        if prompt_len >= int(full_ids.size(1)):
            num_pairs_skipped += 1
            continue
        if int(full_ids.size(1)) > max_length:
            num_pairs_skipped += 1
            continue

        full_ids = full_ids.to(device)
        labels = full_ids.clone()
        labels[:, :prompt_len] = -100

        with torch.no_grad():
            out = model(full_ids, labels=labels)
        _accumulate_routing_from_model(model, routing_tracker)

        num_scored = int((labels[:, 1:] != -100).sum().item())
        if num_scored <= 0:
            num_pairs_skipped += 1
            continue

        total_nll += float(out.loss.item()) * num_scored
        total_scored_tokens += num_scored
        total_ref_chars += len(reference)
        total_ref_bytes += len(reference.encode("utf-8"))
        num_pairs_used += 1

    if total_scored_tokens == 0:
        raise ValueError(
            "Conditional PPL computation produced zero scored tokens "
            "(all pairs skipped)."
        )

    avg_loss = total_nll / total_scored_tokens
    total_nll_bits = total_nll / math.log(2)
    routing_summary = _finalize_routing_tracker(routing_tracker)
    return {
        "metric": "conditional_perplexity",
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "bits_per_char": total_nll_bits / total_ref_chars if total_ref_chars > 0 else None,
        "bits_per_byte": total_nll_bits / total_ref_bytes if total_ref_bytes > 0 else None,
        "scored_tokens": total_scored_tokens,
        "num_pairs_used": num_pairs_used,
        "num_pairs_skipped": num_pairs_skipped,
        "total_ref_chars": total_ref_chars,
        "total_ref_bytes": total_ref_bytes,
        "max_length": max_length,
        "routing_summary": routing_summary,
    }


@dataclass
class LangSpec:
    code: str
    flores_pairs: list[tuple[str, str, str]]
    ppl_corpora: list[tuple[str, str]]  # [(name, path), ...]
    prompt_template: str

    @property
    def has_translation_task(self) -> bool:
        return bool(self.flores_pairs)

    @property
    def has_ppl_task(self) -> bool:
        return bool(self.ppl_corpora)


@dataclass
class ModelSpec:
    name: str
    kind: str  # "single" or "mole"
    model_path: str
    base_model: str | None = None
    mole_adapters: list[str] = field(default_factory=list)
    router_temperature: float = 0.5


def _parse_ppl_specs(raw: list[str] | None) -> list[tuple[str, str]]:
    """
    Parse `NAME=PATH` (or bare `PATH`) entries from a --*-ppl-files flag.
    Bare paths get a name from the file stem. Duplicate names are
    disambiguated with a numeric suffix.
    """
    if not raw:
        return []
    specs: list[tuple[str, str]] = []
    used: set[str] = set()
    for entry in raw:
        if "=" in entry:
            name, _, path = entry.partition("=")
            name = name.strip()
            path = path.strip()
        else:
            path = entry.strip()
            name = Path(path).stem
        if not name or not path:
            raise ValueError(f"Bad --ppl-files entry: {entry!r}")
        base = name
        n = 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        specs.append((name, path))
    return specs


def _parse_flores_pairs(raw: list[str] | None) -> list[tuple[str, str, str]]:
    """Parse `SRCLANG=SRCPATH:REFPATH` entries from a --*-flores-pairs flag."""
    if not raw:
        return []
    pairs: list[tuple[str, str, str]] = []
    used: set[str] = set()
    for entry in raw:
        if "=" not in entry or ":" not in entry.split("=", 1)[1]:
            raise ValueError(
                f"Bad --flores-pairs entry: {entry!r}. "
                "Expected format SRCLANG=SRCPATH:REFPATH"
            )
        src_code, _, rest = entry.partition("=")
        src_path, _, ref_path = rest.partition(":")
        src_code = src_code.strip()
        src_path = src_path.strip()
        ref_path = ref_path.strip()
        if not src_code or not src_path or not ref_path:
            raise ValueError(
                f"Bad --flores-pairs entry: {entry!r}. "
                "Expected format SRCLANG=SRCPATH:REFPATH"
            )
        base = src_code
        n = 2
        while src_code in used:
            src_code = f"{base}_{n}"
            n += 1
        used.add(src_code)
        pairs.append((src_code, src_path, ref_path))
    return pairs


def evaluate_model_on_languages(
    spec: ModelSpec,
    langs: list[LangSpec],
    output_dir: Path,
    device: str,
    max_new_tokens: int,
    ppl_stride: int,
    ppl_max_length: int,
) -> dict:
    print(f"\n=== Evaluating {spec.name} ({spec.kind}) ===")
    print(f"    model_path = {spec.model_path}")
    if spec.kind == "mole":
        model, tokenizer, load_info = load_mole(
            router_dir=spec.model_path,
            base_model=spec.base_model or "",
            adapter_paths=spec.mole_adapters,
            router_temperature=spec.router_temperature,
            device=device,
        )
    else:
        model, tokenizer, load_info = _load_full_or_peft(
            model_path=spec.model_path,
            base_model=spec.base_model,
            device=device,
        )

    per_lang: dict = {}
    for lang in langs:
        if not (lang.has_translation_task or lang.has_ppl_task):
            print(f"  [{lang.code}] no inputs provided, skipping.")
            continue
        print(f"  [{lang.code}] evaluating ...")

        directions: dict[str, dict] = {}
        for src_code, src_path, ref_path in lang.flores_pairs:
            direction_label = f"{src_code}->{lang.code}"
            print(f"    direction {direction_label}: {src_path} -> {ref_path}")
            sources = read_non_empty_lines(src_path)
            references = read_text_lines(ref_path)
            if len(sources) != len(references):
                raise ValueError(
                    f"[{direction_label}] FLORES source/ref length mismatch: "
                    f"{len(sources)} vs {len(references)}"
                )
            print(f"      Greedy translation: {len(sources)} items")
            preds = greedy_generate(
                model=model,
                tokenizer=tokenizer,
                source_lines=sources,
                prompt_template=lang.prompt_template,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            preds_dir = output_dir / "preds"
            preds_dir.mkdir(parents=True, exist_ok=True)
            preds_path_p = preds_dir / f"{spec.name}_{src_code}_to_{lang.code}.txt"
            with open(preds_path_p, "w", encoding="utf-8") as handle:
                for line in preds:
                    handle.write(line + "\n")
            chrf_result = compute_chrf(preds, references)
            print(f"      chrF++ = {chrf_result['score']:.2f}")

            print("      Conditional perplexity:")
            conditional_ppl_result = compute_conditional_perplexity(
                model=model,
                tokenizer=tokenizer,
                sources=sources,
                references=references,
                prompt_template=lang.prompt_template,
                device=device,
                max_length=ppl_max_length,
                collect_routing=(spec.kind == "mole"),
            )
            cbpc = conditional_ppl_result.get("bits_per_char")
            cbpc_s = f"{cbpc:.4f}" if cbpc is not None else "n/a"
            print(
                f"      PPL: {conditional_ppl_result['perplexity']:.4f}  "
                f"cBPC: {cbpc_s}  "
                f"(used: {conditional_ppl_result['num_pairs_used']}, "
                f"skipped: {conditional_ppl_result['num_pairs_skipped']})"
            )
            routing_summary = conditional_ppl_result.get("routing_summary")
            if routing_summary:
                top = (routing_summary.get("expert_ranking") or [{}])[0]
                top_name = top.get("expert", "n/a")
                top_gate = top.get("average_gate")
                top_gate_s = f"{top_gate:.3f}" if isinstance(top_gate, (int, float)) else "n/a"
                print(
                    f"      routing top-expert = {top_name} "
                    f"(avg gate {top_gate_s})"
                )

            directions[src_code] = {
                "source_lang": src_code,
                "target_lang": lang.code,
                "flores_src": src_path,
                "flores_ref": ref_path,
                "predictions_file": str(preds_path_p),
                "chrf": chrf_result,
                "conditional_ppl": conditional_ppl_result,
            }

        perplexities: dict[str, dict] = {}
        for ppl_name, ppl_path in lang.ppl_corpora:
            print(f"    Perplexity [{ppl_name}]:")
            text = read_eval_corpus(ppl_path)
            ppl_result = compute_perplexity(
                model=model,
                tokenizer=tokenizer,
                text=text,
                device=device,
                stride=ppl_stride,
                max_length=ppl_max_length,
            )
            perplexities[ppl_name] = ppl_result
            bpc = ppl_result.get("bits_per_char")
            bpc_s = f"{bpc:.4f}" if bpc is not None else "n/a"
            print(
                f"      PPL: {ppl_result['perplexity']:.4f}  "
                f"BPC: {bpc_s}"
            )

        per_lang[lang.code] = {
            "directions": directions,
            "perplexities": perplexities,
            "flores_pairs": {
                src_code: {"src": src_path, "ref": ref_path}
                for src_code, src_path, ref_path in lang.flores_pairs
            },
            "ppl_corpora": {name: path for name, path in lang.ppl_corpora},
            "prompt_template": lang.prompt_template,
        }

    payload = {"model": {"name": spec.name, **load_info}, "per_language": per_lang}
    per_model_path = output_dir / f"{spec.name}.json"
    with open(per_model_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"  wrote {per_model_path}")

    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return payload


def _add_lang_args(parser: argparse.ArgumentParser, code: str, default_template: str) -> None:
    grp = parser.add_argument_group(f"{code.upper()} held-out evaluation")
    grp.add_argument(
        f"--{code}-flores-pairs",
        nargs="+",
        default=None,
        help=(
            f"One or more translation directions INTO {code.upper()} as "
            f"SRCLANG=SRCPATH:REFPATH entries. SRCLANG is a short label "
            f"(e.g. 'fr', 'ca') identifying the source language; it also "
            f"names the output predictions file. Example: "
            f"--{code}-flores-pairs fr=flores_eval_data/fra_Latn.txt:flores_eval_data/{code}_Latn.txt"
        ),
    )
    grp.add_argument(
        f"--{code}-flores-src",
        default=None,
        help=(
            f"{code.upper()} translation source file. Legacy; equivalent to a single "
            f"--{code}-flores-pairs src=<path>:<ref_path>. Combined with --{code}-flores-ref."
        ),
    )
    grp.add_argument(
        f"--{code}-flores-ref",
        default=None,
        help=f"{code.upper()} translation reference file. Pair with --{code}-flores-src.",
    )
    grp.add_argument(
        f"--{code}-ppl-file",
        default=None,
        help=(
            f"Single {code.upper()} held-out PPL corpus (.txt or .jsonl with a 'text' field). "
            f"Legacy; equivalent to --{code}-ppl-files main=<path>."
        ),
    )
    grp.add_argument(
        f"--{code}-ppl-files",
        nargs="+",
        default=None,
        help=(
            f"One or more {code.upper()} held-out PPL corpora as NAME=PATH pairs "
            f"(or bare PATH; file stem is used as the name). Each corpus is scored "
            f"independently; register-stratified PPL (e.g. web vs clean edited text) "
            f"is the recommended way to read Benchmark 2 results on Occitan."
        ),
    )
    grp.add_argument(
        f"--{code}-prompt-template",
        default=default_template,
        help=f"Prompt template with {{source}} placeholder for {code.upper()} translation.",
    )


def _build_lang_spec(args, code: str) -> LangSpec:
    raw_files = getattr(args, f"{code}_ppl_files", None)
    legacy_ppl = getattr(args, f"{code}_ppl_file", None)
    ppl_specs = _parse_ppl_specs(raw_files)
    if not ppl_specs and legacy_ppl:
        ppl_specs = _parse_ppl_specs([legacy_ppl])
        ppl_specs = [("main", ppl_specs[0][1])]

    raw_pairs = getattr(args, f"{code}_flores_pairs", None)
    flores_pairs = _parse_flores_pairs(raw_pairs)
    legacy_src = getattr(args, f"{code}_flores_src", None)
    legacy_ref = getattr(args, f"{code}_flores_ref", None)
    if legacy_src and legacy_ref:
        legacy_label = "src"
        used = {src for src, _, _ in flores_pairs}
        if legacy_label in used:
            n = 2
            while f"{legacy_label}_{n}" in used:
                n += 1
            legacy_label = f"{legacy_label}_{n}"
        flores_pairs.append((legacy_label, legacy_src, legacy_ref))

    return LangSpec(
        code=code,
        flores_pairs=flores_pairs,
        ppl_corpora=ppl_specs,
        prompt_template=getattr(args, f"{code}_prompt_template"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark 2 (docs/benchmarking_plan.md): MoLE_Final vs OccitanExpert_FullPipeline "
            "and OccitanExpert_Simple across Occitan, French, and Catalan."
        )
    )

    parser.add_argument("--simple-model", required=True, help="OccitanExpert_Simple checkpoint.")
    parser.add_argument("--simple-base", default=None, help="Base model for the simple expert.")
    parser.add_argument("--full-model", required=True, help="OccitanExpert_FullPipeline checkpoint.")
    parser.add_argument("--full-base", default=None, help="Base model for the full-pipeline expert.")

    parser.add_argument("--mole-model", required=True, help="MoLE router directory (router_config.json + router_weights.pt).")
    parser.add_argument("--mole-base", required=True, help="Underlying causal LM for MoLE.")
    parser.add_argument(
        "--mole-adapters",
        nargs="+",
        required=True,
        help="Frozen expert adapter directories in the same order as router_config.json adapter_names.",
    )
    parser.add_argument("--router-temperature", type=float, default=0.5, help="Inference router temperature for MoLE.")

    _add_lang_args(parser, "oc", DEFAULT_PROMPT_OC)
    _add_lang_args(parser, "fr", DEFAULT_PROMPT_FR)
    _add_lang_args(parser, "ca", DEFAULT_PROMPT_CA)

    parser.add_argument("--output-dir", required=True, help="Where to save predictions and result JSONs.")
    parser.add_argument(
        "--expert-ablation-mode",
        action="store_true",
        help=(
            "Expert-ablation/routing-analysis run mode. Redirects output folder "
            "name to `benchmark2_ablation` (or `<name>_ablation`) so prior "
            "benchmark artifacts are not overwritten."
        ),
    )
    parser.add_argument("--device", default=None, help="Device, e.g. cuda or cpu.")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--ppl-stride", type=int, default=512)
    parser.add_argument(
        "--ppl-max-length",
        type=int,
        default=2048,
        help=(
            "Per-window context length for sliding-window PPL. Default 2048. "
            "Capped to the model's max_position_embeddings."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help=(
            "Number of paired-bootstrap resamples for chrF++ confidence "
            "intervals and p-values. Set 0 to disable bootstrap reporting."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for paired-bootstrap resampling.",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    if args.expert_ablation_mode:
        if output_dir.name == "benchmark2_ablation":
            pass
        elif output_dir.name == "benchmark2":
            output_dir = output_dir.with_name("benchmark2_ablation")
        else:
            output_dir = output_dir.with_name(f"{output_dir.name}_ablation")
        print(f"Expert-ablation mode enabled: writing outputs to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for code in ("oc", "fr", "ca"):
        legacy_src = getattr(args, f"{code}_flores_src", None)
        legacy_ref = getattr(args, f"{code}_flores_ref", None)
        if bool(legacy_src) != bool(legacy_ref):
            parser.error(
                f"--{code}-flores-src and --{code}-flores-ref must be provided together."
            )

    langs = [_build_lang_spec(args, code) for code in ("oc", "fr", "ca")]
    if not any(lang.has_translation_task or lang.has_ppl_task for lang in langs):
        parser.error(
            "No language has any held-out inputs. "
            "Provide at least one of --{lang}-flores-pairs, "
            "--{lang}-flores-src/--{lang}-flores-ref, "
            "--{lang}-ppl-file, or --{lang}-ppl-files."
        )

    specs = [
        ModelSpec(
            name="MoLE_Final",
            kind="mole",
            model_path=args.mole_model,
            base_model=args.mole_base,
            mole_adapters=list(args.mole_adapters),
            router_temperature=args.router_temperature,
        ),
        ModelSpec(
            name="OccitanExpert_FullPipeline",
            kind="single",
            model_path=args.full_model,
            base_model=args.full_base,
        ),
        ModelSpec(
            name="OccitanExpert_Simple",
            kind="single",
            model_path=args.simple_model,
            base_model=args.simple_base,
        ),
    ]

    summary: dict = {
        "benchmark": "benchmark2",
        "device": device,
        "max_new_tokens": args.max_new_tokens,
        "ppl_stride": args.ppl_stride,
        "ppl_max_length": args.ppl_max_length,
        "expert_ablation_mode": bool(args.expert_ablation_mode),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "router_temperature": args.router_temperature,
        "languages": {lang.code: {
            "flores_pairs": {
                src_code: {"src": src_path, "ref": ref_path}
                for src_code, src_path, ref_path in lang.flores_pairs
            },
            "ppl_corpora": {name: path for name, path in lang.ppl_corpora},
            "prompt_template": lang.prompt_template,
        } for lang in langs},
        "results": {},
    }

    for spec in specs:
        payload = evaluate_model_on_languages(
            spec=spec,
            langs=langs,
            output_dir=output_dir,
            device=device,
            max_new_tokens=args.max_new_tokens,
            ppl_stride=args.ppl_stride,
            ppl_max_length=args.ppl_max_length,
        )
        summary["results"][spec.name] = payload

    if args.bootstrap_samples > 0:
        model_names = list(summary["results"].keys())
        pairwise: list[tuple[str, str]] = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                pairwise.append((model_names[i], model_names[j]))

        boot_by_direction: dict[str, dict] = {}
        for lang in langs:
            tgt = lang.code
            for src, _, _ in lang.flores_pairs:
                key = f"{tgt}<-{src}"
                per_cmp: dict[str, dict] = {}
                for left_name, right_name in pairwise:
                    left_payload = (
                        (summary["results"].get(left_name) or {})
                        .get("per_language", {})
                        .get(tgt, {})
                        .get("directions", {})
                        .get(src)
                    )
                    right_payload = (
                        (summary["results"].get(right_name) or {})
                        .get("per_language", {})
                        .get(tgt, {})
                        .get("directions", {})
                        .get(src)
                    )
                    if not left_payload or not right_payload:
                        continue
                    left_pred_path = left_payload.get("predictions_file")
                    right_pred_path = right_payload.get("predictions_file")
                    ref_path = right_payload.get("flores_ref") or left_payload.get("flores_ref")
                    if not left_pred_path or not right_pred_path or not ref_path:
                        continue
                    print(f"Computing paired-bootstrap chrF++ CI for {key}: {right_name} - {left_name} ...")
                    left_preds = read_text_lines(left_pred_path)
                    right_preds = read_text_lines(right_pred_path)
                    refs = read_text_lines(ref_path)
                    per_cmp[f"{right_name}_minus_{left_name}"] = bootstrap_chrf_diff(
                        left_predictions=left_preds,
                        right_predictions=right_preds,
                        references=refs,
                        left_name=left_name,
                        right_name=right_name,
                        num_samples=args.bootstrap_samples,
                        seed=args.bootstrap_seed,
                    )
                if per_cmp:
                    boot_by_direction[key] = per_cmp

        summary["bootstrap"] = {
            "method": "paired_bootstrap_sentence_resampling",
            "metric": "chrF++",
            "num_samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "by_direction": boot_by_direction,
        }

    mole_payload = summary["results"].get("MoLE_Final") or {}
    routing_by_direction: dict[str, dict] = {}
    for lang in langs:
        tgt = lang.code
        directions = (
            mole_payload.get("per_language", {})
            .get(tgt, {})
            .get("directions", {})
        )
        for src, _, _ in lang.flores_pairs:
            direction = directions.get(src) or {}
            cond = direction.get("conditional_ppl") or {}
            routing = cond.get("routing_summary")
            if routing:
                routing_by_direction[f"{tgt}<-{src}"] = routing
    if routing_by_direction:
        summary["mole_routing"] = {
            "source": "conditional_perplexity_pass",
            "by_direction": routing_by_direction,
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    lang_codes = ("oc", "fr", "ca")

    direction_columns: list[tuple[str, str]] = []  # [(tgt, src), ...]
    for tgt in lang_codes:
        lang = next((l for l in langs if l.code == tgt), None)
        if lang is None:
            continue
        for src_code, _, _ in lang.flores_pairs:
            direction_columns.append((tgt, src_code))

    bpc_columns: list[tuple[str, str]] = []  # [(lang, corpus_name), ...]
    for code in lang_codes:
        lang = next((l for l in langs if l.code == code), None)
        if lang is None:
            continue
        for ppl_name, _ in lang.ppl_corpora:
            bpc_columns.append((code, ppl_name))

    print("\n" + "=" * 110)
    print("Benchmark 2 complete")
    print(f"  summary: {summary_path}")

    if direction_columns:
        print("\n  chrF++ by direction (tgt <- src)")
        print("  " + "-" * 90)
        header_cells = [f"{'Model':30s}"]
        for tgt, src in direction_columns:
            header_cells.append(f"{tgt + '<-' + src:>8s}")
        print("  " + "  ".join(header_cells))
        for name, payload in summary["results"].items():
            row = [f"{name:30s}"]
            for tgt, src in direction_columns:
                entry = payload["per_language"].get(tgt) or {}
                direction = (entry.get("directions") or {}).get(src)
                chrf = direction.get("chrf") if direction else None
                chrf_s = f"{chrf['score']:.2f}" if chrf else "   -   "
                row.append(f"{chrf_s:>8s}")
            print("  " + "  ".join(row))

        have_cond = any(
            (
                (payload["per_language"].get(tgt) or {}).get("directions") or {}
            ).get(src, {}).get("conditional_ppl")
            for payload in summary["results"].values()
            for tgt, src in direction_columns
        )
        if have_cond:
            print(
                "\n  Conditional BPC by direction "
                "(NLL of reference | instruction prompt)"
            )
            print("  " + "-" * 90)
            header_cells = [f"{'Model':30s}"]
            for tgt, src in direction_columns:
                header_cells.append(f"{tgt + '<-' + src:>8s}")
            print("  " + "  ".join(header_cells))
            for name, payload in summary["results"].items():
                row = [f"{name:30s}"]
                for tgt, src in direction_columns:
                    entry = payload["per_language"].get(tgt) or {}
                    direction = (entry.get("directions") or {}).get(src)
                    cond = direction.get("conditional_ppl") if direction else None
                    cbpc = cond.get("bits_per_char") if cond else None
                    cbpc_s = f"{cbpc:.4f}" if cbpc is not None else "   -   "
                    row.append(f"{cbpc_s:>8s}")
                print("  " + "  ".join(row))
            print(
                "\n  Lower cBPC = higher probability mass on the correct translation\n"
                "  given the instruction-formatted prompt. Probabilistic analog of\n"
                "  chrF++; expected to favor instruction/curriculum-tuned models\n"
                "  even when unconditional BPC does not."
            )

    routing = (summary.get("mole_routing") or {}).get("by_direction") or {}
    if routing:
        print("\n  MoLE routing attribution (average gate mass by direction)")
        print("  " + "-" * 90)
        print("  " + f"{'Direction':12s}  {'Top expert':20s}  {'Avg gate':>8s}")
        for direction in sorted(routing.keys()):
            ranking = routing[direction].get("expert_ranking") or []
            top = ranking[0] if ranking else {}
            top_name = str(top.get("expert", "n/a"))
            top_gate = top.get("average_gate")
            top_gate_s = f"{top_gate:.3f}" if isinstance(top_gate, (int, float)) else "n/a"
            print("  " + f"{direction:12s}  {top_name:20s}  {top_gate_s:>8s}")

    if bpc_columns:
        print("\n  BPC (bits-per-character, tokenizer-invariant) by language.corpus")
        print("  " + "-" * 90)
        header_cells = [f"{'Model':30s}"]
        for code, ppl_name in bpc_columns:
            header_cells.append(f"{code + '.' + ppl_name:>14s}")
        print("  " + "  ".join(header_cells))
        for name, payload in summary["results"].items():
            row = [f"{name:30s}"]
            for code, ppl_name in bpc_columns:
                entry = payload["per_language"].get(code) or {}
                perps = entry.get("perplexities") or {}
                ppl = perps.get(ppl_name)
                bpc = ppl.get("bits_per_char") if ppl else None
                bpc_s = f"{bpc:.4f}" if bpc is not None else "   -   "
                row.append(f"{bpc_s:>14s}")
            print("  " + "  ".join(row))
        print(
            "\n  Lower BPC = better language-modeling fit on that corpus. Use\n"
            "  multiple corpora per language to stratify by register (e.g. oc.hplt =\n"
            "  noisy web text, oc.ud = clean edited text, oc.flores = neutral news prose)."
        )

    boot = summary.get("bootstrap") or {}
    boot_dirs = boot.get("by_direction") or {}
    if boot_dirs:
        print("\n  Paired-bootstrap chrF++ comparisons (right - left)")
        print("  " + "-" * 110)
        print(
            "  "
            + f"{'Direction':12s}  {'Comparison':40s}  {'Observed':>9s}  {'CI95 Low':>9s}  {'CI95 High':>10s}  {'p-value':>8s}"
        )
        for direction in sorted(boot_dirs.keys()):
            comps = boot_dirs[direction]
            for name, stats in comps.items():
                print(
                    "  "
                    + f"{direction:12s}  {name:40s}  "
                    + f"{stats['observed_diff_right_minus_left']:>9.2f}  "
                    + f"{stats['ci95_low']:>9.2f}  "
                    + f"{stats['ci95_high']:>10.2f}  "
                    + f"{stats['p_value_two_sided']:>8.4f}"
                )
    print("=" * 110)


if __name__ == "__main__":
    main()
