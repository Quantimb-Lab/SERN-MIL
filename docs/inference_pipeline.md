# Inference Pipeline

Entry points:

- `scripts/infer_pipeline.py`
- `python -m sern_mil.cli.main infer-pipeline --config configs/inference_pipeline.yaml`

Modes:

- Single slide: pass `--slide`
- Batch folder: pass `--wsis-dir` (or default `paths.wsis_dir`)

Output:

- `artifacts/inference_predictions.json`
