# Dataset construction

- `build_mmlu_truthfulqa_samples.py`: build nested target samples.
- `build_medmcqa_samples.py`: build MedMCQA target samples.
- `build_mmlu_pro_scalr_samples.py`: build MMLU-Pro and LegalBench SCALR target samples.
- `streams/build_stream_dataset.py`: assign ten pre-Target tasks per Target.
- `streams/precompute_similarity.py`: cache Qwen3-Embedding-8B similarities.
- `streams/validate_stream_dataset.py`: validate task and cache invariants.

Generated benchmark inputs belong under `data/<dataset>/sample`; scripts never
write experiment results.
