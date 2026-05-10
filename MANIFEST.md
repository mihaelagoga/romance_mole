# Snapshot Manifest

Files: 89

## Included Files
`.gitignore`
`artifacts/routing_smoke_trace_t8_aux001_sup002_cs000.txt`
`data/alignments_gemini.json`
`data/eval/oc_ppl_hplt_500.jsonl`
`data/eval/occitan_minimal_pairs_raw.jsonl`
`data/eval/occitan_minimal_pairs_rejected.jsonl`
`data/eval/dev_flores200_dev_oc.jsonl`
`data/eval/dev_flores200_test_oc.jsonl`
`data/eval/eval_external_languedocien_filter_metadata.json`
`data/eval/eval_external_languedocien_filtered.jsonl`
`data/eval/external_eval_metadata.json`
`data/eval/test_external_occitan_combined.jsonl`
`data/eval/test_flores200_test_oc.jsonl`
`data/eval/test_ud_occitan.jsonl`
`data/eval/test_wikipedia_occitan.jsonl`
`data/synthetic/alpaca_occitan/splits/dev_alpaca_200.jsonl`
`data/synthetic/alpaca_occitan/splits/split_metadata.json`
`data/synthetic/alpaca_occitan/splits/test_alpaca_200.jsonl`
`data/synthetic/alpaca_occitan/splits/train_alpaca.jsonl`
`data/synthetic/alpaca_occitan_v2/alpaca_occitan_v2.jsonl`
`data/synthetic/dialogocc/synthetic_dialogue_250.jsonl`
`data/synthetic/flores200occ/synthetic_flores_500.jsonl`
`data/synthetic/morphostressocc/synthetic_stress_test_250.jsonl`
`data/synthetic/merged_splits/dev_merged.jsonl`
`data/synthetic/merged_splits/merge_metadata.json`
`data/synthetic/merged_splits/train_merged.jsonl`
`data/synthetic/synth_splits/dev_synth.jsonl`
`data/synthetic/synth_splits/synth_split_metadata.json`
`data/synthetic/synth_splits/test_synth.jsonl`
`data/synthetic/synth_splits/train_synth.jsonl`
`environment.yml`
`flores_eval_data/cat_Latn.txt`
`flores_eval_data/flores_aligned.jsonl`
`flores_eval_data/fra_Latn.txt`
`flores_eval_data/oci_Latn.txt`
`notebooks/occitan_llama_tokenizer_patched/added_tokens_list.txt`
`notebooks/occitan_llama_tokenizer_patched/patch_metadata.json`
`README.md`
`requirements.txt`
`src/__init__.py`
`src/data_processing/__init__.py`
`src/data_processing/augment_french_expert_data.py`
`src/data_processing/build_external_occitan_eval.py`
`src/data_processing/build_french_repair_set.py`
`src/data_processing/build_mole_router_dataset.py`
`src/data_processing/build_curriculum_data.py`
`src/data_processing/clean_hplt_occitan.py`
`src/data_processing/collect_hplt_raw.py`
`src/data_processing/fetch_flores_eval_data.py`
`src/data_processing/filter_flores_languedocien.py`
`src/data_processing/generate_occitan_minimal_pairs.py`
`src/data_processing/generate_synthetic_alpaca_occitan.py`
`src/data_processing/generate_synthetic_dialogue_occitan.py`
`src/data_processing/generate_synthetic_flores_occitan.py`
`src/data_processing/generate_synthetic_morpho_stress_occitan.py`
`src/data_processing/prepare_romance_expert_data.py`
`src/data_processing/split_alpaca_occitan.py`
`src/data_processing/split_hplt.py`
`src/data_processing/split_synthetic.py`
`src/evaluation/__init__.py`
`src/evaluation/benchmark_occitan_minimal_pairs.py`
`src/evaluation/benchmark_occitan_pipeline.py`
`src/evaluation/benchmark_romance_mole.py`
`src/evaluation/generate_predictions.py`
`src/evaluation/inspect_mole_routing.py`
`src/evaluation/plot_routing_heatmaps_from_trace.py`
`src/evaluation/plot_routing_summary_figures.py`
`src/evaluation/score_metrics.py`
`src/evaluation/smoke_test_catalan_expert.py`
`src/evaluation/smoke_test_french_expert.py`
`src/evaluation/tokenizer_comparative_test.py`
`src/merging/__init__.py`
`src/merging/create_occitan_adapter_ties_merge.py`
`src/merging/verify_merge.py`
`src/models/__init__.py`
`src/models/mole/__init__.py`
`src/models/mole/romance_mole.py`
`src/tokenization/__init__.py`
`src/tokenization/align_occitan_tokens_with_gemini.py`
`src/tokenization/build_occitan_tokenizer.py`
`src/tokenization/initialize_occitan_model_embeddings.py`
`src/tokenization/weight_token_alignments.py`
`src/training/__init__.py`
`src/training/make_peft_adapter_compatible.py`
`src/training/train_occitan_curriculum_lora.py`
`src/training/train_occitan_hplt_baseline_lora.py`
`src/training/train_romance_expert_lora.py`
`src/training/train_romance_mole_router.py`

## Exclusion Notes
Raw HPLT train/cleaned files are intentionally omitted.
Model checkpoints, adapter weights, generated results, and local virtual environments are intentionally omitted.
Vendored third-party source trees such as the local `mergekit/` checkout are intentionally omitted.
`src/data_processing/run_alpaca_occitan_with_keys.py` is intentionally omitted because it contained local API-key wiring.
