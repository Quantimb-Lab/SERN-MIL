# Architecture Notes

## Preprocessing dependencies

- CLAM feature extraction depends on preprocessing manifest.
- CellViT feature extraction depends on preprocessing manifest.
- Fusion depends on both CLAM and CellViT manifests with exact coordinate keys:
  - `slide_id`
  - `tile_x`, `tile_y`
  - `patch_x`, `patch_y`

## SERN-MIL module boundaries

- `models/ser`: selective embedding retrieval
- `models/nfi`: nuclei feature integration
- `models/mil`: region-aware MIL classification
- `models/survival`: optional BCR prognostic head

## Pipeline split

- `training/pipeline.py`: stage-wise training orchestration
- `inference/pipeline.py`: single-slide and batch inference orchestration
