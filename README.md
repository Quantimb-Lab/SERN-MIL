# SERN-MIL

End-to-end repository for reproducing the SERN-MIL pipeline with
CLAM-compatible tissue embeddings and CellViT-based nuclei features.

## Pipeline

1. WSI preprocessing (`0.50 um/px`)
2. Tissue masking (Otsu + morphology)
3. Tiling (`4096x4096`) and patching (`256x256`)
4. CLAM feature extraction
5. CellViT nuclei feature extraction
6. Feature fusion
7. SER module
8. Region-aware MIL and ISUP grading
9. Optional BCR head

## Structure

```text
configs/
docker/
docs/
scripts/
src/sern_mil/
tests/
```

## Commands

```bash
make setup
make preprocess
make features
make fuse
make train
make infer
```

or:

```bash
python -m sern_mil.cli.main preprocess --config configs/default.yaml
python -m sern_mil.cli.main extract-clam --config configs/default.yaml
python -m sern_mil.cli.main extract-cellvit --config configs/default.yaml
python -m sern_mil.cli.main fuse --config configs/default.yaml
python scripts/train_pipeline.py --config configs/training_pipeline.yaml
python scripts/infer_pipeline.py --config configs/inference_pipeline.yaml --slide path/to/slide.svs
```
