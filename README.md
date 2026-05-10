# Romance Bridge Occitan

This repository provides a pipeline that builds and evaluates an Occitan-capable Llama 3.1 model using trans-tokenization, curriculum LoRA training, and a Romance-language Mixture of LoRA Experts (MoLE).

This repository excludes large raw corpora, trained model weights, local checkpoints, and result folders.

## What Is Included

Data collection, cleanup, and split preparation for HPLT and external evaluation data.
Synthetic Occitan generation scripts for FLORES-style text, morphosyntactic stress data, dialogue data, Alpaca-style instruction data, and minimal-pair challenge data.
Trans-tokenization utilities: Occitan tokenizer patching, Gemini token alignment, alignment weighting, and embedding initialization.
TIES adapter initialization from French and Catalan LoRA experts.
LoRA training scripts for French/Catalan Romance experts, the full Occitan curriculum expert, and the HPLT-only Occitan baseline.
Romance-MoLE architecture and final router-training script.
Benchmark scripts for the final thesis evaluation suite.
Small benchmark/evaluation files needed to reproduce the reported evaluations.

## Repository Layout

```text
src/
 data_processing/   Dataset cleanup, synthetic generation, and split builders
 tokenization/    Occitan tokenizer patching and embedding initialization
 merging/       TIES merge utilities for Occitan adapter initialization
 training/      LoRA, curriculum, baseline, and MoLE router training
 models/mole/     Romance-MoLE model implementation
 evaluation/     Benchmarks, metrics, smoke tests, and routing figures
data/
 eval/        Small held-out evaluation files
 synthetic/      Small synthetic datasets used by the final pipeline
flores_eval_data/   FLORES French/Catalan/Occitan text files
artifacts/       Small qualitative routing trace used for figures
```

## Installation

Use Python 3.10 or 3.11. A GPU-enabled PyTorch install is required for training and model evaluation.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install the CUDA build of PyTorch appropriate for your machine first.
# Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

For Conda users, `environment.yml` provides a minimal public environment that installs the same Python dependencies from `requirements.txt`.

## External Requirements

You need to provide or download:

A base Llama 3.1 / Carballo-compatible model, e.g. `proxectonos/Llama-3.1-Carballo`.
Raw HPLT Occitan data if you want to reproduce Stage 3a training from scratch.
Optional FastText language identification for HPLT filtering. To enable it, install `fasttext-wheel`, download `lid.176.bin` from [fastText language identification](https://fasttext.cc/docs/en/language-identification.html), and pass its path with `--lid-model`.
Hugging Face access for gated model or dataset downloads, when applicable.
A Gemini API key for synthetic data generation and token-alignment scripts:

```bash
export GEMINI_API_KEY=...
```

No API keys are stored in this repository.

## Hardware Guidance

Small preprocessing and plotting scripts can run on CPU.

Recommended hardware:

Tokenization, data generation, and metric aggregation: CPU is sufficient.
Single 8B LoRA training: at least one 24 GB GPU with bf16 and gradient checkpointing; A30/A40/A5000/A6000 class GPUs are suitable.
Full Occitan curriculum training: one 24 GB GPU is workable with batch size 1 and gradient accumulation; two 24 GB GPUs are more comfortable.
Romance-MoLE router training with three LoRA experts: two 24 GB GPUs are recommended, or a single 48 GB GPU.
Full benchmark generation over the 6-direction FLORES matrix can take many hours on one 24 GB GPU.

The final thesis runs were designed around limited university GPU access, primarily 24 GB A30-class devices.

## Pipeline Overview

The commands below are examples. Adjust model paths, checkpoint paths, and output directories for your machine.

### 1. HPLT Cleanup And Stage 3a Split

```bash
python -m src.data_processing.collect_hplt_raw \
 --input_dir data/hplt_v3/raw \
 --output_file data/hplt_v3/extracted_raw/all_data.jsonl

python -m src.data_processing.clean_hplt_occitan \
 --input data/hplt_v3/extracted_raw/all_data.jsonl \
 --output data/hplt_v3/cleaned/all_data.jsonl

# Optional stricter language-ID filtering used in the thesis runs:
# pip install fasttext-wheel
# python -m src.data_processing.clean_hplt_occitan \
#  --input data/hplt_v3/extracted_raw/all_data.jsonl \
#  --output data/hplt_v3/cleaned/all_data.jsonl \
#  --lid-model src/auxiliary/lid.176.bin

python -m src.data_processing.split_hplt_stage3a \
 --input data/hplt_v3/cleaned/all_data.jsonl \
 --output-dir data/hplt_v3/splits
```

### 2. Synthetic Occitan Data

```bash
python -m src.data_processing.generate_synthetic_flores_occitan
python -m src.data_processing.generate_synthetic_morpho_stress_occitan
python -m src.data_processing.generate_synthetic_dialogue_occitan
python -m src.data_processing.generate_synthetic_alpaca_occitan

python -m src.data_processing.split_stage3b_synthetic
python -m src.data_processing.split_alpaca_occitan
python -m src.data_processing.build_stage3b2_curriculum_data
```

### 3. Trans-Tokenization

```bash
python -m src.tokenization.build_occitan_tokenizer
python -m src.tokenization.align_occitan_tokens_with_gemini
python -m src.tokenization.weight_token_alignments

python -m src.tokenization.initialize_occitan_model_embeddings \
 --base-model proxectonos/Llama-3.1-Carballo \
 --patched-tokenizer models/occitan_llama_tokenizer_patched \
 --alignments data/alignments_gemini.json \
 --output models/llama-3.1-occitan-initialized
```

### 4. Romance Expert Adapters

```bash
python -m src.data_processing.prepare_romance_expert_data
python -m src.data_processing.augment_french_expert_data \
 --local-source data/cousin_data/cousin_fr_seed.jsonl \
 --output-path data/cousin_data/cousin_fr_augmented.jsonl
python -m src.data_processing.build_french_repair_set

python -m src.training.train_romance_expert_lora \
 --lang fr \
 --data data/cousin_data/cousin_fr_augmented.jsonl data/cousin_data/cousin_fr_repair.jsonl \
 --data_repeat 1 3 \
 --output_dir checkpoints/lora_fr

python -m src.training.train_romance_expert_lora \
 --lang ca \
 --data data/cousin_data/cousin_ca.jsonl \
 --output_dir checkpoints/lora_ca
```

### 5. Occitan Adapter Initialization And Curriculum Training

```bash
python -m src.merging.create_occitan_adapter_ties_merge \
 --base-model models/llama-3.1-occitan-initialized \
 --adapter-fr checkpoints/lora_fr/final \
 --adapter-ca checkpoints/lora_ca/final \
 --output-dir models/adapter_oc_init

python -m src.training.train_occitan_curriculum_lora \
 --stage hplt \
 --data data/hplt_v3/splits/stage3a_train_hplt.jsonl \
 --eval_data data/hplt_v3/splits/stage3a_dev_hplt_1000.jsonl \
 --adapter models/adapter_oc_init \
 --output_dir checkpoints/occitan_3a \
 --max_steps 2000 \
 --expandable-cuda-segments

python -m src.training.train_occitan_curriculum_lora \
 --\
 --data data/synthetic/stage3b2_merged_splits/stage3b2_train_merged.jsonl \
 --eval_data data/synthetic/stage3b2_merged_splits/stage3b2_dev_merged.jsonl \
 --adapter checkpoints/occitan_3a/final \
 --output_dir checkpoints/occitan_3b2 \
 --max_steps 2500 \
 --expandable-cuda-segments
```

The `--expandable-cuda-segments` flag is an opt-in CUDA memory-fragmentation workaround used in the thesis runs; omit it if your PyTorch/CUDA setup does not need it.

### 6. HPLT-Only Baseline

```bash
python -m src.training.train_occitan_hplt_baseline_lora \
 --base_model proxectonos/Llama-3.1-Carballo \
 --data data/hplt_v3/splits/stage3a_train_hplt.jsonl \
 --eval_data data/hplt_v3/splits/stage3a_dev_hplt_1000.jsonl \
 --output_dir checkpoints/oc_simple_carballo_hplt
```

### 7. Romance-MoLE Router

```bash
python -m src.data_processing.build_mole_router_dataset \
 --output_dir data/router_training
```

```bash
torchrun --nproc_per_node=2 -m src.training.train_romance_mole_router \
 --base_model models/llama-3.1-occitan-initialized \
 --adapters checkpoints/lora_fr/final checkpoints/lora_ca/final_compat checkpoints/occitan_3b2/final_compat \
 --adapter_names fr ca oc \
 --data data/router_training/mole_router_train.jsonl \
 --eval_data data/router_training/mole_router_dev.jsonl \
 --output_dir checkpoints/router_ablation_ladder/t8_aux001_sup002_cs000 \
 --sequence_route_threshold 8 \
 --router_aux_loss_coef 0.01 \
 --router_supervision_coef 0.02 \
 --expandable-cuda-segments \
 --disable-nccl-p2p \
 --isolate-visible-gpus
```

The three MoLE memory flags are opt-in cluster workarounds used in the thesis runs on memory-constrained multi-GPU jobs. Omit them unless your `torchrun` setup needs the same CUDA/NCCL behavior.

### 8. Benchmarks

Benchmark 1 compares the HPLT-only baseline to the full Occitan pipeline:

```bash
python -m src.evaluation.benchmark_occitan_pipeline \
 --simple-model checkpoints/oc_simple_carballo_hplt/final \
 --simple-base proxectonos/Llama-3.1-Carballo \
 --full-model checkpoints/occitan_3b2/final_compat \
 --full-base models/llama-3.1-occitan-initialized \
 --flores-pairs \
  fr=flores_eval_data/fra_Latn.txt:flores_eval_data/oci_Latn.txt \
  ca=flores_eval_data/cat_Latn.txt:flores_eval_data/oci_Latn.txt \
 --ppl-files \
  hplt=data/eval/oc_ppl_hplt_500.jsonl \
  ud=data/eval/stage3b_test_ud_occitan.jsonl \
  flores=flores_eval_data/oci_Latn.txt \
 --bootstrap-samples 500 \
 --output-dir results/benchmark1
```

Benchmark 2 evaluates MoLE across all French/Catalan/Occitan translation directions:

```bash
python -m src.evaluation.benchmark_romance_mole \
 --simple-model checkpoints/oc_simple_carballo_hplt/final \
 --simple-base proxectonos/Llama-3.1-Carballo \
 --full-model checkpoints/occitan_3b2/final_compat \
 --full-base models/llama-3.1-occitan-initialized \
 --mole-model checkpoints/router_ablation_ladder/t8_aux001_sup002_cs000/final \
 --mole-base models/llama-3.1-occitan-initialized \
 --mole-adapters checkpoints/lora_fr/final checkpoints/lora_ca/final_compat checkpoints/occitan_3b2/final_compat \
 --router-temperature 0.5 \
 --oc-flores-pairs fr=flores_eval_data/fra_Latn.txt:flores_eval_data/oci_Latn.txt ca=flores_eval_data/cat_Latn.txt:flores_eval_data/oci_Latn.txt \
 --fr-flores-pairs oc=flores_eval_data/oci_Latn.txt:flores_eval_data/fra_Latn.txt ca=flores_eval_data/cat_Latn.txt:flores_eval_data/fra_Latn.txt \
 --ca-flores-pairs oc=flores_eval_data/oci_Latn.txt:flores_eval_data/cat_Latn.txt fr=flores_eval_data/fra_Latn.txt:flores_eval_data/cat_Latn.txt \
 --oc-ppl-files hplt=data/eval/oc_ppl_hplt_500.jsonl ud=data/eval/stage3b_test_ud_occitan.jsonl flores=flores_eval_data/oci_Latn.txt \
 --fr-ppl-files flores=flores_eval_data/fra_Latn.txt \
 --ca-ppl-files flores=flores_eval_data/cat_Latn.txt \
 --bootstrap-samples 500 \
 --output-dir results/benchmark2
```

Benchmark 3 evaluates the Occitan minimal-pair challenge:

```bash
python -m src.evaluation.benchmark_occitan_minimal_pairs \
 --minimal-pairs data/eval/occitan_minimal_pairs_raw.jsonl \
 --simple-model checkpoints/oc_simple_carballo_hplt/final \
 --simple-base proxectonos/Llama-3.1-Carballo \
 --full-model checkpoints/occitan_3b2/final_compat \
 --full-base models/llama-3.1-occitan-initialized \
 --mole-model checkpoints/router_ablation_ladder/t8_aux001_sup002_cs000/final \
 --mole-base models/llama-3.1-occitan-initialized \
 --mole-adapters checkpoints/lora_fr/final checkpoints/lora_ca/final_compat checkpoints/occitan_3b2/final_compat \
 --router-temperature 0.5 \
 --output-dir results/benchmark3
```

### 9. Routing Figures

```bash
python -m src.evaluation.plot_routing_heatmaps_from_trace \
 --input artifacts/routing_smoke_trace_t8_aux001_sup002_cs000.txt \
 --output-dir results/routing_heatmaps \
 --response-only \
 --also-full \
 --max-tokens 90

python -m src.evaluation.plot_routing_summary_figures \
 --tokens-csv results/routing_heatmaps/routing_tokens.csv \
 --summary-csv results/routing_heatmaps/routing_summary.csv \
 --output-dir results/routing_paper_figures
```

## Evaluation & Benchmark Results

The pipeline's effectiveness is validated across three core benchmarks:

- **Curriculum Training Wins:** The full Occitan curriculum pipeline significantly outperforms the simple HPLT-only baseline on FLORES translation tasks (e.g., `fr→oc` chrF++ of 45.45 vs 30.88).
- **MoLE Preserves Expertise:** The Romance MoLE router preserves this specialized Occitan performance (chrF++ 44.70) while successfully adding strong French generation capabilities (`oc→fr` chrF++ 58.06) where the Occitan expert fails.
- **Grammatical Accuracy:** MoLE scores 81% accuracy on a curated minimal-pair diagnostic set, correctly handling Occitan-specific linguistic features like elision, articles, and contractions.

*(Note: Detailed methodology, significance tests, and raw perplexity findings will be available in the accompanying thesis document.)*

## Notes On Reproducibility

`MANIFEST.md` lists the curated files in this public snapshot.
Model weights are not included; all checkpoint paths in commands are placeholders matching the thesis directory layout.
Synthetic datasets included here are small enough for code-release reproducibility, but any claims should still report how they were generated and manually reviewed.
