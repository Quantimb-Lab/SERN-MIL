# Training Pipeline

Entry points:

- `scripts/train_pipeline.py`
- `python -m sern_mil.cli.main train-pipeline --config configs/training_pipeline.yaml`

Stages:

1. Validate fused feature manifest
2. Prepare checkpoint directory
3. Train SER stage (placeholder hook)
4. Train MIL stage
5. Optional survival stage
6. Save summary at `artifacts/training_run_summary.json`
