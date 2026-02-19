# SERN-MIL

Official repository for the paper:
`SERN-MIL: Selective Embedding Retrieval and Nuclei-Feature Aggregation Based MIL for Prostate Gleason Grading`.

This repository contains the codebase for preprocessing, feature extraction, training, and inference with a
`CLAM + CellViT` dependent pipeline aligned with the paper methodology.

## Method Overview

1. WSI preprocessing at `0.50 um/px`
2. Tissue masking (Otsu + morphology)
3. Tiling (`4096x4096`) and patching (`256x256`)
4. CLAM-compatible tissue embedding extraction
5. CellViT-based nuclei feature extraction
6. Feature fusion with coordinate alignment
7. SER module for selective embedding retrieval
8. Region-aware MIL for GP classification and ISUP grading
9. Optional survival/BCR head

## Repository Layout

```text
configs/              # Pipeline, model, dataset, and experiment configs
docker/               # Dockerfile and compose
docs/                 # Architecture and pipeline docs
scripts/              # User-facing entry scripts
src/sern_mil/         # Core package
tests/                # Unit tests
```

## Installation

### Local

```bash
pip install -e .
```

### Docker

```bash
docker build -f docker/Dockerfile -t sern-mil:latest .
docker run --rm -it -v $PWD:/workspace sern-mil:latest bash
```

## Data Preparation

Place your slides in:

- `data/wsi/`

Optionally keep labels/splits in:

- `data/labels/`
- `data/splits/`

## Run Full Preprocessing + Feature Pipeline

### Make targets

```bash
make preprocess
make features
make fuse
```

### Equivalent explicit commands

```bash
python -m sern_mil.cli.main preprocess --config configs/default.yaml
python -m sern_mil.cli.main extract-clam --config configs/default.yaml
python -m sern_mil.cli.main extract-cellvit --config configs/default.yaml
python -m sern_mil.cli.main fuse --config configs/default.yaml
```

Generated manifests:

- `data_index/preprocess_manifest.jsonl`
- `data_index/clam_features_manifest.jsonl`
- `data_index/cellvit_features_manifest.jsonl`
- `data_index/fused_features_manifest.jsonl`

## Training

Training pipeline is separated from inference.

### Quick run

```bash
make train
```

### Explicit command

```bash
python scripts/train_pipeline.py --config configs/training_pipeline.yaml
```

### CLI alternative

```bash
python -m sern_mil.cli.main train-pipeline --config configs/training_pipeline.yaml
```

Training summary output:

- `artifacts/training_run_summary.json`

Checkpoints directory:

- `checkpoints/`

## Inference

### Single WSI

```bash
python scripts/infer_pipeline.py --config configs/inference_pipeline.yaml --slide data/wsi/sample.svs
```

### Batch folder

```bash
python scripts/infer_pipeline.py --config configs/inference_pipeline.yaml --wsis-dir data/wsi
```

### Make target

```bash
make infer
```

Inference output:

- `artifacts/inference_predictions.json`

## Main Config Files

- `configs/default.yaml`: preprocessing + feature pipeline settings
- `configs/training_pipeline.yaml`: training-specific settings
- `configs/inference_pipeline.yaml`: inference-specific settings
- `configs/models/sern_mil.yaml`: model-level defaults
- `configs/experiments/*.yaml`: ablation toggles

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

## Notes

- Pipeline code is structured for paper-faithful components (SER, NFI, MIL), with clean separation between:
  - preprocessing/feature generation
  - training
  - inference
- For full experimental reproduction, ensure your dataset splits and checkpoints match your study protocol.
