"""
Benchmark the full Occitan pipeline against the HPLT-only baseline.

Example:
    python -m src.evaluation.benchmark_occitan_pipeline --simple-model checkpoints/oc_simple_carballo_hplt/final --simple-base proxectonos/Llama-3.1-Carballo --full-model checkpoints/occitan_3b2/final_compat --full-base models/llama-3.1-occitan-initialized --flores-pairs fr=flores_eval_data/fra_Latn.txt:flores_eval_data/oci_Latn.txt ca=flores_eval_data/cat_Latn.txt:flores_eval_data/oci_Latn.txt --ppl-files hplt=data/eval/oc_ppl_hplt_500.jsonl ud=data/eval/test_ud_occitan.jsonl flores=flores_eval_data/oci_Latn.txt --output-dir results/benchmark1
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import PeftModel
from sacrebleu.metrics import CHRF
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Traduís en occitan lengadocian la frasa francesa seguenta.\n\n"
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
    """Load held-out evaluation text. Supports:."""
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


def _model_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def _ensure_pad(tokenizer):
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _detect_adapter_vocab_size(adapter_dir: Path) -> int | None:
    """Peek inside a PEFT adapter and return the vocab size of any saved."""
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
    """Load either a full HF model directory or a PEFT adapter directory."""
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
            print(f"  generated {idx}/{len(source_lines)}")
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
    """Paired bootstrap confidence interval for chrF++ difference."""
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


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    device: str,
    stride: int = 512,
    max_length: int | None = None,
) -> dict:
    """Sliding-window causal-LM perplexity. Loss is averaged over only the new."""
    total_chars = len(text)
    total_bytes = len(text.encode("utf-8"))

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    total_tokens = int(input_ids.size(1))
    if total_tokens < 2:
        raise ValueError("Eval text tokenized to <2 tokens; nothing to score.")

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
            # FLORES pairs are far shorter than max_length; skipping the rare
            # overflow is cleaner than partial scoring (which would bias BPC).
            num_pairs_skipped += 1
            continue

        full_ids = full_ids.to(device)
        labels = full_ids.clone()
        labels[:, :prompt_len] = -100

        with torch.no_grad():
            out = model(full_ids, labels=labels)

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
    }


@dataclass
class ModelSpec:
    name: str
    model_path: str
    base_model: str | None = None


@dataclass
class Benchmark1Result:
    model: dict = field(default_factory=dict)
    # Per-direction results keyed by source-language code (e.g. "fr", "ca").
    # Each value has: chrf, conditional_ppl, predictions_file, flores_src, flores_ref.
    directions: dict[str, dict] = field(default_factory=dict)
    # Unconditional PPL on held-out Occitan corpora, keyed by corpus name.
    perplexities: dict = field(default_factory=dict)


def _parse_flores_pairs(raw: list[str] | None) -> list[tuple[str, str, str]]:
    """Parse `SRCLANG=SRCPATH:REFPATH` entries from a --flores-pairs flag."""
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


def _parse_ppl_specs(raw: list[str] | None) -> list[tuple[str, str]]:
    """Parse `NAME=PATH` (or bare `PATH`) entries from --ppl-files."""
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


def evaluate_model(
    spec: ModelSpec,
    flores_pairs: list[tuple[str, str, str]],
    ppl_corpora: dict[str, str],
    output_dir: Path,
    device: str,
    prompt_template: str,
    max_new_tokens: int,
    ppl_stride: int,
    ppl_max_length: int,
) -> Benchmark1Result:
    print(f"\n=== Evaluating {spec.name} ===")
    print(f"    model_path = {spec.model_path}")
    print(f"    base_model = {spec.base_model}")

    model, tokenizer, load_info = load_model_and_tokenizer(spec.model_path, spec.base_model, device)
    result = Benchmark1Result(model={"name": spec.name, **load_info})

    for src_code, src_path, ref_path in flores_pairs:
        direction_label = f"{src_code}->oc"
        print(f"  direction {direction_label}: {src_path} -> {ref_path}")
        sources = read_non_empty_lines(src_path)
        references = read_text_lines(ref_path)
        if len(sources) != len(references):
            raise ValueError(
                f"[{direction_label}] FLORES source/ref length mismatch: "
                f"{len(sources)} vs {len(references)}"
            )
        print(f"    Greedy translation: {len(sources)} items")
        preds = greedy_generate(
            model=model,
            tokenizer=tokenizer,
            source_lines=sources,
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        preds_path = output_dir / "preds" / f"{spec.name}_{src_code}_to_oc.txt"
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(preds_path, "w", encoding="utf-8") as handle:
            for line in preds:
                handle.write(line + "\n")
        chrf_result = compute_chrf(preds, references)
        print(f"    chrF++ = {chrf_result['score']:.2f}")

        print("    Conditional perplexity:")
        conditional_ppl_result = compute_conditional_perplexity(
            model=model,
            tokenizer=tokenizer,
            sources=sources,
            references=references,
            prompt_template=prompt_template,
            device=device,
            max_length=ppl_max_length,
        )
        cbpc = conditional_ppl_result.get("bits_per_char")
        cbpc_s = f"{cbpc:.4f}" if cbpc is not None else "n/a"
        print(
            f"    PPL: {conditional_ppl_result['perplexity']:.4f}  "
            f"cBPC: {cbpc_s}  "
            f"(used: {conditional_ppl_result['num_pairs_used']}, "
            f"skipped: {conditional_ppl_result['num_pairs_skipped']})"
        )

        result.directions[src_code] = {
            "source_lang": src_code,
            "target_lang": "oc",
            "flores_src": src_path,
            "flores_ref": ref_path,
            "predictions_file": str(preds_path),
            "chrf": chrf_result,
            "conditional_ppl": conditional_ppl_result,
        }

    for name, text in ppl_corpora.items():
        print(f"  Perplexity [{name}]:")
        ppl = compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            text=text,
            device=device,
            stride=ppl_stride,
            max_length=ppl_max_length,
        )
        result.perplexities[name] = ppl
        bpc = ppl.get("bits_per_char")
        bpc_s = f"{bpc:.4f}" if bpc is not None else "n/a"
        print(
            f"    PPL: {ppl['perplexity']:.4f}  "
            f"BPC: {bpc_s}"
        )

    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark 1 (docs/benchmarking_plan.md): OccitanExpert_Simple vs "
            "OccitanExpert_FullPipeline on a held-out Occitan suite."
        )
    )
    parser.add_argument("--simple-model", required=True, help="OccitanExpert_Simple checkpoint path.")
    parser.add_argument("--simple-base", default=None, help="Base model for the simple expert (PEFT adapters).")
    parser.add_argument("--full-model", required=True, help="OccitanExpert_FullPipeline checkpoint path.")
    parser.add_argument("--full-base", default=None, help="Base model for the full-pipeline expert (PEFT adapters).")

    parser.add_argument(
        "--flores-pairs",
        nargs="+",
        default=None,
        help=(
            "One or more translation directions INTO Occitan as "
            "SRCLANG=SRCPATH:REFPATH entries. SRCLANG is a short label "
            "(e.g. 'fr', 'ca') identifying the source language; it also "
            "names the output predictions file. Example: "
            "--flores-pairs fr=flores_eval_data/fra_Latn.txt:flores_eval_data/oci_Latn.txt "
            "ca=flores_eval_data/cat_Latn.txt:flores_eval_data/oci_Latn.txt"
        ),
    )
    parser.add_argument(
        "--flores-src",
        default=None,
        help=(
            "Held-out source (one sentence per line). Legacy; equivalent to a "
            "single --flores-pairs src=<path>:<ref_path>. Combined with --flores-ref."
        ),
    )
    parser.add_argument(
        "--flores-ref",
        default=None,
        help="Held-out reference (one sentence per line). Pair with --flores-src.",
    )
    parser.add_argument(
        "--ppl-file",
        default=None,
        help=(
            "Single held-out PPL corpus (.txt or .jsonl with a 'text' field). "
            "Legacy; equivalent to --ppl-files main=<path>. Prefer --ppl-files "
            "for register-stratified reporting across multiple held-out sets."
        ),
    )
    parser.add_argument(
        "--ppl-files",
        nargs="+",
        default=None,
        help=(
            "One or more held-out PPL corpora as NAME=PATH pairs (or bare PATH, "
            "in which case the file stem is used as the name). Each corpus is "
            "scored independently and appears as its own column in the summary "
            "table and as its own entry under 'perplexities' in the per-model "
            "JSON. Typical usage: "
            "--ppl-files hplt=data/eval/oc_ppl_hplt_500.jsonl "
            "ud=data/eval/test_ud_occitan.jsonl "
            "flores=flores_eval_data/oci_Latn.txt"
        ),
    )

    parser.add_argument("--output-dir", required=True, help="Where to save predictions and result JSONs.")
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help=(
            "CI/significance run mode. Redirects output folder name to "
            "`benchmark1_ci` (or `<name>_ci`) so prior benchmark artifacts "
            "are not overwritten."
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
            "Capped to the model's max_position_embeddings. Keep small "
            "(1024-4096) unless you specifically want long-context PPL."
        ),
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template with {source} placeholder. Same template is used for both models.",
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

    ppl_specs = _parse_ppl_specs(args.ppl_files)
    if not ppl_specs and args.ppl_file:
        ppl_specs = _parse_ppl_specs([args.ppl_file])
        ppl_specs = [("main", ppl_specs[0][1])]

    if bool(args.flores_src) != bool(args.flores_ref):
        parser.error("--flores-src and --flores-ref must be provided together.")

    flores_pairs = _parse_flores_pairs(args.flores_pairs)
    if args.flores_src and args.flores_ref:
        legacy_label = "src"
        used = {src for src, _, _ in flores_pairs}
        if legacy_label in used:
            n = 2
            while f"{legacy_label}_{n}" in used:
                n += 1
            legacy_label = f"{legacy_label}_{n}"
        flores_pairs.append((legacy_label, args.flores_src, args.flores_ref))

    if not (flores_pairs or ppl_specs):
        parser.error(
            "At least one of --flores-pairs, --flores-src/--flores-ref, "
            "--ppl-file, or --ppl-files must be provided."
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    if args.ci_mode:
        if output_dir.name == "benchmark1_ci":
            pass
        elif output_dir.name == "benchmark1":
            output_dir = output_dir.with_name("benchmark1_ci")
        else:
            output_dir = output_dir.with_name(f"{output_dir.name}_ci")
        print(f"CI mode enabled: writing outputs to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ppl_corpora: dict[str, str] = {}
    for name, path in ppl_specs:
        print(f"Load PPL [{name}]: {path}")
        ppl_corpora[name] = read_eval_corpus(path)

    specs = [
        ModelSpec(name="OccitanExpert_Simple", model_path=args.simple_model, base_model=args.simple_base),
        ModelSpec(name="OccitanExpert_FullPipeline", model_path=args.full_model, base_model=args.full_base),
    ]

    summary = {
        "benchmark": "benchmark1",
        "device": device,
        "prompt_template": args.prompt_template,
        "max_new_tokens": args.max_new_tokens,
        "ppl_stride": args.ppl_stride,
        "ppl_max_length": args.ppl_max_length,
        "ci_mode": bool(args.ci_mode),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "flores_pairs": {
            src_code: {"src": src_path, "ref": ref_path}
            for src_code, src_path, ref_path in flores_pairs
        },
        "ppl_corpora": {name: path for name, path in ppl_specs},
        "results": {},
    }

    for spec in specs:
        result = evaluate_model(
            spec=spec,
            flores_pairs=flores_pairs,
            ppl_corpora=ppl_corpora,
            output_dir=output_dir,
            device=device,
            prompt_template=args.prompt_template,
            max_new_tokens=args.max_new_tokens,
            ppl_stride=args.ppl_stride,
            ppl_max_length=args.ppl_max_length,
        )
        per_model_payload = {
            "model": result.model,
            "directions": result.directions,
            "perplexities": result.perplexities,
        }
        summary["results"][spec.name] = per_model_payload
        per_model_path = output_dir / f"{spec.name}.json"
        with open(per_model_path, "w", encoding="utf-8") as handle:
            json.dump(per_model_payload, handle, ensure_ascii=False, indent=2)
        print(f"  wrote {per_model_path}")

    ppl_names = [name for name, _ in ppl_specs]
    direction_codes = [src_code for src_code, _, _ in flores_pairs]
    if args.bootstrap_samples > 0 and direction_codes:
        boots: dict[str, dict] = {}
        left_name = "OccitanExpert_Simple"
        right_name = "OccitanExpert_FullPipeline"
        for src in direction_codes:
            left_payload = (summary["results"].get(left_name) or {}).get("directions", {}).get(src)
            right_payload = (summary["results"].get(right_name) or {}).get("directions", {}).get(src)
            if not left_payload or not right_payload:
                continue
            left_pred_path = left_payload.get("predictions_file")
            right_pred_path = right_payload.get("predictions_file")
            ref_path = right_payload.get("flores_ref") or left_payload.get("flores_ref")
            if not left_pred_path or not right_pred_path or not ref_path:
                continue
            print(f"Computing paired-bootstrap chrF++ CI for {src}->oc ...")
            left_preds = read_text_lines(left_pred_path)
            right_preds = read_text_lines(right_pred_path)
            refs = read_text_lines(ref_path)
            boots[f"{src}->oc"] = bootstrap_chrf_diff(
                left_predictions=left_preds,
                right_predictions=right_preds,
                references=refs,
                left_name=left_name,
                right_name=right_name,
                num_samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
        summary["bootstrap"] = {
            "method": "paired_bootstrap_sentence_resampling",
            "metric": "chrF++",
            "num_samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "comparisons": boots,
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 120)
    print("Benchmark 1 complete")
    print(f"  summary: {summary_path}\n")

    if direction_codes:
        print("  chrF++ by direction (src -> oc)")
        print("  " + "-" * 90)
        header = [f"{'Model':30s}"]
        for src in direction_codes:
            header.append(f"{'chrF[' + src + '->oc]':>14s}")
        print("  " + "  ".join(header))
        for name, payload in summary["results"].items():
            dirs = payload.get("directions") or {}
            row = [f"{name:30s}"]
            for src in direction_codes:
                d = dirs.get(src) or {}
                chrf = d.get("chrf")
                chrf_s = f"{chrf['score']:.2f}" if chrf else "   -   "
                row.append(f"{chrf_s:>14s}")
            print("  " + "  ".join(row))

        print("\n  Conditional BPC by direction (reference | instruction prompt)")
        print("  " + "-" * 90)
        header = [f"{'Model':30s}"]
        for src in direction_codes:
            header.append(f"{'cBPC[' + src + '->oc]':>14s}")
        print("  " + "  ".join(header))
        for name, payload in summary["results"].items():
            dirs = payload.get("directions") or {}
            row = [f"{name:30s}"]
            for src in direction_codes:
                d = dirs.get(src) or {}
                cond = d.get("conditional_ppl")
                cbpc = cond.get("bits_per_char") if cond else None
                cbpc_s = f"{cbpc:.4f}" if cbpc is not None else "   -   "
                row.append(f"{cbpc_s:>14s}")
            print("  " + "  ".join(row))

    boot = summary.get("bootstrap") or {}
    boot_cmp = boot.get("comparisons") or {}
    if boot_cmp:
        print("\n  Paired-bootstrap chrF++ (FullPipeline - Simple)")
        print("  " + "-" * 110)
        print(
            "  "
            + f"{'Direction':14s}  {'Observed':>9s}  {'CI95 Low':>9s}  {'CI95 High':>10s}  {'p-value':>8s}  {'sig@0.05':>9s}"
        )
        for direction, stats in boot_cmp.items():
            print(
                "  "
                + f"{direction:14s}  "
                + f"{stats['observed_diff_right_minus_left']:>9.2f}  "
                + f"{stats['ci95_low']:>9.2f}  "
                + f"{stats['ci95_high']:>10.2f}  "
                + f"{stats['p_value_two_sided']:>8.4f}  "
                + f"{'yes' if stats['significant_0p05'] else 'no':>9s}"
            )

    if ppl_names:
        print("\n  Unconditional BPC by Occitan corpus")
        print("  " + "-" * 90)
        header = [f"{'Model':30s}"]
        for pname in ppl_names:
            header.append(f"{'BPC[' + pname + ']':>14s}")
        print("  " + "  ".join(header))
        for name, payload in summary["results"].items():
            perps = payload.get("perplexities") or {}
            row = [f"{name:30s}"]
            for pname in ppl_names:
                ppl = perps.get(pname)
                bpc = ppl.get("bits_per_char") if ppl else None
                bpc_s = f"{bpc:.4f}" if bpc is not None else "   -   "
                row.append(f"{bpc_s:>14s}")
            print("  " + "  ".join(row))

    print(
        "\n  chrF++ = character-n-gram F-score of greedy output vs reference.\n"
        "    Higher is better. Permissive: rewards any output that shares\n"
        "    character n-grams with the reference.\n"
        "\n  cBPC = conditional bits-per-character: NLL of the reference given\n"
        "    the instruction-formatted prompt, divided by reference characters.\n"
        "    Lower is better. Strict: penalises any lexical or morphological\n"
        "    divergence from the exact reference sequence.\n"
        "\n  BPC = unconditional bits-per-character on raw held-out Occitan\n"
        "    corpora. Lower is better. May favor the broader pretraining\n"
        "    distribution of the simple baseline; this is the standard\n"
        "    instruction-tuning-vs-PPL tradeoff.\n"
        "\n  Raw perplexity and per-corpus metadata live in the per-model JSONs."
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
