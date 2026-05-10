"""
Evaluate Occitan minimal-pair preference accuracy.

Example:
    python -m src.evaluation.benchmark_occitan_minimal_pairs --minimal-pairs data/eval/occitan_minimal_pairs_raw.jsonl --full-model checkpoints/occitan_3b2/final_compat --full-base models/llama-3.1-occitan-initialized --output-dir results/benchmark3
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.mole.romance_mole import RomanceMoLEModel


DEFAULT_PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Traduís en occitan lengadocian la frasa francesa seguenta.\n\n"
    "### Input:\n"
    "{source}\n\n"
    "### Response:\n"
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} of {path} is not a JSON object.")
            rows.append(obj)
    if not rows:
        raise ValueError(f"No JSONL rows found in {path}")
    return rows


def validate_minimal_pair_rows(
    rows: list[dict[str, Any]],
    *,
    exclude_needs_review: bool,
) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    required = ("id", "category", "source", "correct", "incorrect")
    for idx, row in enumerate(rows, start=1):
        missing = [key for key in required if not str(row.get(key, "")).strip()]
        if missing:
            raise ValueError(f"Minimal-pair row {idx} missing required fields: {missing}")
        if exclude_needs_review and bool(row.get("needs_manual_review", False)):
            continue
        correct = str(row["correct"]).strip()
        incorrect = str(row["incorrect"]).strip()
        if correct == incorrect:
            raise ValueError(f"Minimal-pair row {idx} has identical correct/incorrect text.")
        normalized = dict(row)
        normalized["id"] = str(row["id"])
        normalized["category"] = str(row["category"])
        normalized["phenomenon"] = str(row.get("phenomenon", row["category"]))
        normalized["source_lang"] = str(row.get("source_lang", "fr"))
        normalized["source"] = str(row["source"]).strip()
        normalized["correct"] = correct
        normalized["incorrect"] = incorrect
        valid.append(normalized)
    if not valid:
        raise ValueError("No minimal-pair rows remained after filtering.")
    return valid


def _resolve_artifact_dir(model_path: str | Path) -> Path | str:
    p = Path(model_path).expanduser()
    if not p.exists():
        return str(model_path)
    if p.is_dir() and (p / "final").is_dir():
        return p / "final"
    return p


def _is_peft_adapter_dir(p: Path) -> bool:
    return (p / "adapter_config.json").exists() and (
        (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()
    )


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


def load_model_and_tokenizer(model_path: str, base_model: str | None, device: str):
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
            tokenizer_source = resolved if (resolved / "tokenizer.json").exists() else inferred_base
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
            return model, tokenizer, {
                "loader": "peft_adapter",
                "base": inferred_base,
                "path": str(resolved),
                "adapter_vocab": adapter_vocab,
                "tokenizer_vocab": len(tokenizer),
                "resized_to": int(target_vocab),
            }

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
    hard_router_argmax: bool,
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
        router_temperature=float(router_temperature),
        hard_router_argmax=bool(hard_router_argmax),
    )
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
        "hard_router_argmax": bool(hard_router_argmax),
    }


def _resolve_expert_index(model, selector: str) -> int:
    names = list(getattr(model, "adapter_names", []))
    if not names:
        raise ValueError("MoLE model has no adapter_names; cannot resolve expert selector.")

    if selector.isdigit():
        idx = int(selector)
        if idx < 0 or idx >= len(names):
            raise ValueError(
                f"Expert index {idx} out of range for {len(names)} experts: {names}"
            )
        return idx

    if selector in names:
        return names.index(selector)

    lowered = selector.lower()
    for i, name in enumerate(names):
        if name.lower() == lowered:
            return i

    raise ValueError(f"Unknown expert selector '{selector}'. Available experts: {names}")


def _install_mole_router_hooks(
    model,
    *,
    force_expert: str | None,
    disable_experts: list[str],
) -> dict[str, Any]:
    names = list(getattr(model, "adapter_names", []))
    if not names:
        return {"enabled": False, "reason": "no_adapter_names"}

    force_idx = _resolve_expert_index(model, force_expert) if force_expert else None
    disable_idxs = sorted({_resolve_expert_index(model, name) for name in disable_experts})
    if force_idx is not None and force_idx in disable_idxs:
        raise ValueError(
            "The same expert cannot be both forced and disabled. "
            f"force={force_expert}, disable={disable_experts}"
        )

    if force_idx is None and not disable_idxs:
        return {"enabled": False, "reason": "no_controls"}

    neg = -1e4
    pos = 1e4

    def hook_fn(_module, _inputs, output):
        if not torch.is_tensor(output):
            return output
        logits = output.clone()
        if force_idx is not None:
            logits.fill_(neg)
            logits[..., force_idx] = pos
        if disable_idxs:
            for idx in disable_idxs:
                logits[..., idx] = neg
        return logits

    handles: list[Any] = []
    for router in model.routers.values():
        handles.append(router.register_forward_hook(hook_fn))
    setattr(model, "_benchmark3_router_hook_handles", handles)

    return {
        "enabled": True,
        "force_expert": names[force_idx] if force_idx is not None else None,
        "disable_experts": [names[i] for i in disable_idxs],
        "num_router_hooks": len(handles),
    }


@dataclass
class ModelSpec:
    name: str
    kind: str
    model_path: str
    base_model: str | None = None
    mole_adapters: list[str] = field(default_factory=list)
    router_temperature: float = 0.5
    hard_router_argmax: bool = False
    force_expert: str | None = None
    disable_experts: list[str] = field(default_factory=list)


@dataclass
class CandidateScore:
    text: str
    total_nll: float
    total_nll_bits: float
    loss: float
    perplexity: float
    bits_per_char: float | None
    bits_per_byte: float | None
    scored_tokens: int
    chars: int
    bytes: int


@dataclass
class PairScore:
    item_id: str
    category: str
    phenomenon: str
    source_lang: str
    source: str
    correct: str
    incorrect: str
    correct_score: dict[str, Any]
    incorrect_score: dict[str, Any]
    score_field: str
    margin_incorrect_minus_correct: float | None
    correct_preferred: bool | None
    tie: bool


@dataclass
class ModelResult:
    model: dict[str, Any] = field(default_factory=dict)
    aggregate: dict[str, Any] = field(default_factory=dict)
    by_category: dict[str, Any] = field(default_factory=dict)
    items: list[dict[str, Any]] = field(default_factory=list)


def score_candidate(
    model,
    tokenizer,
    *,
    prompt: str,
    candidate: str,
    device: str,
    max_length: int,
) -> CandidateScore:
    full_text = prompt + candidate
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = int(prompt_ids.size(1))

    if prompt_len >= int(full_ids.size(1)):
        raise ValueError("Candidate produced no scoreable tokens.")
    if int(full_ids.size(1)) > max_length:
        raise ValueError(
            f"Prompt+candidate token length {int(full_ids.size(1))} exceeds max_length={max_length}"
        )

    full_ids = full_ids.to(device)
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        out = model(full_ids, labels=labels)

    scored_tokens = int((labels[:, 1:] != -100).sum().item())
    if scored_tokens <= 0:
        raise ValueError("Candidate produced zero scored tokens.")

    total_nll = float(out.loss.item()) * scored_tokens
    total_nll_bits = total_nll / math.log(2)
    chars = len(candidate)
    byte_count = len(candidate.encode("utf-8"))
    avg_loss = total_nll / scored_tokens
    return CandidateScore(
        text=candidate,
        total_nll=total_nll,
        total_nll_bits=total_nll_bits,
        loss=avg_loss,
        perplexity=math.exp(avg_loss),
        bits_per_char=total_nll_bits / chars if chars > 0 else None,
        bits_per_byte=total_nll_bits / byte_count if byte_count > 0 else None,
        scored_tokens=scored_tokens,
        chars=chars,
        bytes=byte_count,
    )


def candidate_to_dict(score: CandidateScore) -> dict[str, Any]:
    return {
        "text": score.text,
        "total_nll": score.total_nll,
        "total_nll_bits": score.total_nll_bits,
        "loss": score.loss,
        "perplexity": score.perplexity,
        "bits_per_char": score.bits_per_char,
        "bits_per_byte": score.bits_per_byte,
        "scored_tokens": score.scored_tokens,
        "chars": score.chars,
        "bytes": score.bytes,
    }


def _score_value(candidate: CandidateScore, field_name: str) -> float | None:
    value = getattr(candidate, field_name)
    if value is None:
        return None
    return float(value)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    spread = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (centre - spread) / denom, (centre + spread) / denom


def binomial_two_sided_p_value(k: int, n: int, p: float = 0.5) -> float | None:
    if n <= 0:
        return None
    if not 0.0 < p < 1.0:
        raise ValueError("p must be between 0 and 1.")

    def prob(i: int) -> float:
        return math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))

    observed_prob = prob(k)
    total = sum(prob(i) for i in range(n + 1) if prob(i) <= observed_prob + 1e-15)
    return min(1.0, float(total))


def summarize_pair_scores(pair_scores: list[PairScore]) -> dict[str, Any]:
    scored = [item for item in pair_scores if item.correct_preferred is not None]
    correct = sum(1 for item in scored if item.correct_preferred)
    ties = sum(1 for item in pair_scores if item.tie)
    n = len(scored)
    ci_low, ci_high = wilson_ci(correct, n)
    p_value = binomial_two_sided_p_value(correct, n)
    margins = [
        item.margin_incorrect_minus_correct
        for item in scored
        if item.margin_incorrect_minus_correct is not None
    ]
    return {
        "num_items": len(pair_scores),
        "num_scored": n,
        "num_correct_preferred": correct,
        "num_incorrect_preferred": n - correct,
        "num_ties": ties,
        "accuracy": correct / n if n else None,
        "accuracy_ci95_low": ci_low,
        "accuracy_ci95_high": ci_high,
        "binomial_p_value_vs_0p5": p_value,
        "mean_margin_incorrect_minus_correct": (
            sum(margins) / len(margins) if margins else None
        ),
    }


def evaluate_model(
    spec: ModelSpec,
    items: list[dict[str, Any]],
    *,
    output_dir: Path,
    device: str,
    prompt_template: str,
    score_field: str,
    max_length: int,
    log_every: int,
) -> ModelResult:
    print(f"\n=== Evaluating {spec.name} ===")
    print(f"    model_path = {spec.model_path}")
    print(f"    base_model = {spec.base_model}")

    if spec.kind == "mole":
        model, tokenizer, load_info = load_mole(
            router_dir=spec.model_path,
            base_model=spec.base_model or "",
            adapter_paths=spec.mole_adapters,
            router_temperature=spec.router_temperature,
            hard_router_argmax=spec.hard_router_argmax,
            device=device,
        )
        routing_controls = _install_mole_router_hooks(
            model,
            force_expert=spec.force_expert,
            disable_experts=spec.disable_experts,
        )
        load_info["router_controls"] = routing_controls
        if routing_controls.get("enabled"):
            print(
                f"  Router controls: force={routing_controls.get('force_expert')} "
                f"disable={routing_controls.get('disable_experts')}"
            )
    else:
        model, tokenizer, load_info = load_model_and_tokenizer(
            spec.model_path,
            spec.base_model,
            device,
        )
    result = ModelResult(model={"name": spec.name, **load_info})

    pair_scores: list[PairScore] = []
    for idx, item in enumerate(items, start=1):
        prompt = prompt_template.format(source=item["source"])
        correct = score_candidate(
            model,
            tokenizer,
            prompt=prompt,
            candidate=item["correct"],
            device=device,
            max_length=max_length,
        )
        incorrect = score_candidate(
            model,
            tokenizer,
            prompt=prompt,
            candidate=item["incorrect"],
            device=device,
            max_length=max_length,
        )

        correct_value = _score_value(correct, score_field)
        incorrect_value = _score_value(incorrect, score_field)
        margin = None
        correct_preferred = None
        tie = False
        if correct_value is not None and incorrect_value is not None:
            margin = incorrect_value - correct_value
            tie = math.isclose(correct_value, incorrect_value, rel_tol=1e-9, abs_tol=1e-12)
            correct_preferred = bool(correct_value < incorrect_value and not tie)

        pair_scores.append(
            PairScore(
                item_id=item["id"],
                category=item["category"],
                phenomenon=item.get("phenomenon", item["category"]),
                source_lang=item.get("source_lang", "fr"),
                source=item["source"],
                correct=item["correct"],
                incorrect=item["incorrect"],
                correct_score=candidate_to_dict(correct),
                incorrect_score=candidate_to_dict(incorrect),
                score_field=score_field,
                margin_incorrect_minus_correct=margin,
                correct_preferred=correct_preferred,
                tie=tie,
            )
        )

        if log_every > 0 and idx % log_every == 0:
            running = summarize_pair_scores(pair_scores)
            acc = running.get("accuracy")
            acc_s = f"{acc:.3f}" if acc is not None else "n/a"
            print(f"  scored {idx}/{len(items)}  accuracy={acc_s}")

    result.aggregate = summarize_pair_scores(pair_scores)
    by_category: dict[str, list[PairScore]] = defaultdict(list)
    for score in pair_scores:
        by_category[score.category].append(score)
    result.by_category = {
        category: summarize_pair_scores(scores)
        for category, scores in sorted(by_category.items())
    }
    result.items = [
        {
            "id": score.item_id,
            "category": score.category,
            "phenomenon": score.phenomenon,
            "source_lang": score.source_lang,
            "source": score.source,
            "correct": score.correct,
            "incorrect": score.incorrect,
            "score_field": score.score_field,
            "margin_incorrect_minus_correct": score.margin_incorrect_minus_correct,
            "correct_preferred": score.correct_preferred,
            "tie": score.tie,
            "correct_score": score.correct_score,
            "incorrect_score": score.incorrect_score,
        }
        for score in pair_scores
    ]

    per_model_path = output_dir / f"{spec.name}.json"
    with open(per_model_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": result.model,
                "aggregate": result.aggregate,
                "by_category": result.by_category,
                "items": result.items,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  wrote {per_model_path}")

    handles = getattr(model, "_benchmark3_router_hook_handles", [])
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark 3: minimal-pair preference accuracy for Occitan "
            "morphosyntax challenge items."
        )
    )
    parser.add_argument("--minimal-pairs", required=True, help="Minimal-pair JSONL file.")
    parser.add_argument("--simple-model", required=True, help="OccitanExpert_Simple checkpoint path.")
    parser.add_argument("--simple-base", default=None, help="Base model for the simple expert.")
    parser.add_argument("--full-model", required=True, help="OccitanExpert_FullPipeline checkpoint path.")
    parser.add_argument("--full-base", default=None, help="Base model for the full-pipeline expert.")
    parser.add_argument("--mole-model", default=None, help="Optional MoLE router directory.")
    parser.add_argument("--mole-base", default=None, help="Underlying causal LM for optional MoLE.")
    parser.add_argument(
        "--mole-adapters",
        nargs="+",
        default=None,
        help="Frozen expert adapter directories for optional MoLE, in router_config.json order.",
    )
    parser.add_argument(
        "--router-temperature",
        type=float,
        default=0.5,
        help="Inference router temperature for optional MoLE.",
    )
    parser.add_argument(
        "--mole-hard-router-argmax",
        action="store_true",
        help="For optional MoLE: use hard one-expert routing at inference.",
    )
    parser.add_argument(
        "--mole-force-expert",
        default=None,
        help=(
            "For optional MoLE: force routing to a single expert by name "
            "(e.g. occitan) or index (e.g. 2)."
        ),
    )
    parser.add_argument(
        "--mole-disable-expert",
        nargs="+",
        default=None,
        help=(
            "For optional MoLE: disable one or more experts by name or index. "
            "Example: --mole-disable-expert occitan"
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Where to save result JSONs.")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda or cpu.")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template with {source} placeholder. Same template is used for both models.",
    )
    parser.add_argument(
        "--score-field",
        default="bits_per_char",
        choices=("bits_per_char", "bits_per_byte", "loss", "total_nll"),
        help=(
            "Candidate score used for preference. Lower is better. "
            "bits_per_char is the default because it reduces length bias."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum prompt+candidate token length. Default 2048.",
    )
    parser.add_argument(
        "--exclude-needs-review",
        action="store_true",
        help="Skip items whose needs_manual_review field is true.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if bool(args.mole_model) or bool(args.mole_base) or bool(args.mole_adapters):
        if not (args.mole_model and args.mole_base and args.mole_adapters):
            parser.error(
                "--mole-model, --mole-base, and --mole-adapters must be provided together."
            )
    else:
        if args.mole_hard_router_argmax or args.mole_force_expert or args.mole_disable_expert:
            parser.error(
                "MoLE router controls require --mole-model/--mole-base/--mole-adapters."
            )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_items = read_jsonl(args.minimal_pairs)
    items = validate_minimal_pair_rows(
        raw_items,
        exclude_needs_review=args.exclude_needs_review,
    )
    print(f"Items: {len(items)} ({args.minimal_pairs})")
    print(f"Score: {args.score_field} (lower is better)")

    specs = [
        ModelSpec(
            name="OccitanExpert_Simple",
            kind="single",
            model_path=args.simple_model,
            base_model=args.simple_base,
        ),
        ModelSpec(
            name="OccitanExpert_FullPipeline",
            kind="single",
            model_path=args.full_model,
            base_model=args.full_base,
        ),
    ]
    if args.mole_model:
        specs.insert(
            0,
            ModelSpec(
                name="MoLE_Final",
                kind="mole",
                model_path=args.mole_model,
                base_model=args.mole_base,
                mole_adapters=list(args.mole_adapters),
                router_temperature=args.router_temperature,
                hard_router_argmax=bool(args.mole_hard_router_argmax),
                force_expert=args.mole_force_expert,
                disable_experts=list(args.mole_disable_expert or []),
            ),
        )

    summary: dict[str, Any] = {
        "benchmark": "benchmark3",
        "minimal_pairs": args.minimal_pairs,
        "device": device,
        "prompt_template": args.prompt_template,
        "score_field": args.score_field,
        "max_length": args.max_length,
        "exclude_needs_review": bool(args.exclude_needs_review),
        "router_temperature": args.router_temperature if args.mole_model else None,
        "mole_hard_router_argmax": bool(args.mole_hard_router_argmax) if args.mole_model else None,
        "mole_force_expert": args.mole_force_expert if args.mole_model else None,
        "mole_disable_expert": list(args.mole_disable_expert or []) if args.mole_model else None,
        "num_items": len(items),
        "categories": sorted({item["category"] for item in items}),
        "results": {},
    }

    for spec in specs:
        result = evaluate_model(
            spec,
            items,
            output_dir=output_dir,
            device=device,
            prompt_template=args.prompt_template,
            score_field=args.score_field,
            max_length=args.max_length,
            log_every=args.log_every,
        )
        summary["results"][spec.name] = {
            "model": result.model,
            "aggregate": result.aggregate,
            "by_category": result.by_category,
            "items_file": str(output_dir / f"{spec.name}.json"),
        }

    simple = summary["results"]["OccitanExpert_Simple"]
    full = summary["results"]["OccitanExpert_FullPipeline"]
    simple_acc = simple["aggregate"].get("accuracy")
    full_acc = full["aggregate"].get("accuracy")
    summary["comparison"] = {
        "full_minus_simple_accuracy": (
            full_acc - simple_acc if full_acc is not None and simple_acc is not None else None
        ),
        "simple_accuracy": simple_acc,
        "full_accuracy": full_acc,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("Benchmark 3 complete")
    print(f"  summary: {summary_path}\n")
    print("  Minimal-pair preference accuracy")
    print("  " + "-" * 80)
    print(f"  {'Model':30s}  {'Accuracy':>8s}  {'Correct':>7s}  {'Total':>7s}  {'CI95':>19s}  {'p(vs .5)':>9s}")
    for name, payload in summary["results"].items():
        agg = payload["aggregate"]
        acc = agg["accuracy"]
        ci_low = agg["accuracy_ci95_low"]
        ci_high = agg["accuracy_ci95_high"]
        p_val = agg["binomial_p_value_vs_0p5"]
        print(
            "  "
            + f"{name:30s}  "
            + f"{acc:>8.3f}  "
            + f"{agg['num_correct_preferred']:>7d}  "
            + f"{agg['num_scored']:>7d}  "
            + f"[{ci_low:.3f}, {ci_high:.3f}]  "
            + f"{p_val:>9.4f}"
        )

    print("\n  Accuracy by category")
    print("  " + "-" * 80)
    categories = sorted({item["category"] for item in items})
    header = [f"{'Category':28s}"]
    for name in summary["results"]:
        header.append(f"{name.replace('OccitanExpert_', ''):>18s}")
    print("  " + "  ".join(header))
    for category in categories:
        row = [f"{category:28s}"]
        for name, payload in summary["results"].items():
            cat = payload["by_category"].get(category)
            if cat and cat["accuracy"] is not None:
                row.append(f"{cat['accuracy']:.3f} ({cat['num_correct_preferred']}/{cat['num_scored']})".rjust(18))
            else:
                row.append("   -".rjust(18))
        print("  " + "  ".join(row))

    delta = summary["comparison"]["full_minus_simple_accuracy"]
    delta_s = f"{delta:+.3f}" if delta is not None else "n/a"
    print(f"\n  FullPipeline - Simple accuracy delta: {delta_s}")
    print(
        "\n  Preference = score(correct) < score(incorrect), where score is "
        f"`{args.score_field}`. Lower is better."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
