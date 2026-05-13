# Training

All paths are resolved relative to the repository root. YAML files can extend a
base configuration; nested mappings are merged. The checked-in defaults state
model dimensions, stage-specific optimizer settings, epoch limits, and loss rules.

## Tensor caches

Use torch.save dictionaries containing tensors and ordinary Python scalars.
Paths are listed in JSONL manifests. Every training row supplies `cache_path`,
`patient_id`, `dataset`, and `split`. Regional/tile rows also have `slide_id`.
Patient overlap across splits and PANDA grading training/validation are rejected.

### SER

Manifest: `paths.ser_manifest`.

A tile cache contains:

- `tokens`: [256, 1024], in row-major order.
- `shallow_features`: [256, 256, 16, 16].
- `coords`: [256, 2], global (x, y) target-resolution pixel coordinates.
- `tumor_target`, `boundary_target`: [1, 4096, 4096].
- `label_mask`: [4096, 4096], for annotated regional-cache preparation.
- `nuclei_features`: list of 256 tensors, each [nuclei_in_patch, 128].

Rows contain the tile origin `tile_x`, `tile_y` and, for training sampling,
`has_boundary`. Boundary/interior sampling mass defaults to 0.7/0.3 when both
types are present. Feature preparation creates paired image/mask augmentation
variants for training; these variants are excluded from MIL bag preparation.

Tumor loss is Dice + BCE. Boundary loss is BCE + soft IoU. Its weight increases
from 0.3 to 2.0 over the first 30 epochs. Defaults: AdamW 2e-4, weight decay 0.05,
five warm-up epochs, cosine decay to 1e-6, accumulation of four tiles,
gradient clipping at 1.0, EMA 0.999, 40 epochs, and validation-loss patience 8.

### NFI student

Manifest: `paths.student_manifest`. Each file contains:

- `embeddings`: [nuclei, 1024], extracted CellViT tokens.
- `neighborhoods`: [nuclei, 16, 16], binary local presence maps.
- `targets`: [nuclei, 128], paired teacher morphometric features.

The loss is half the mean squared feature error + cosine disagreement +
1e-4 times the squared output norm. Defaults: batch 1024, AdamW 1e-3,
weight decay 0.01, cosine decay to 1e-5, up to 100 epochs, and validation-loss
patience 10. The feature extractor commands use the saved student checkpoint.

### NFI aggregation and regional MIL

Manifest: `paths.mil_manifest`.

When this manifest is absent, the training pipeline prepares it from unaugmented
training/validation tile caches and the trained SER checkpoint. It uses reference
GP regions for bag formation and context-decoded SER features. It does not
construct test bags from annotations.

Each regional cache supplies `context_tokens`: [patches, 1024],
`nuclei_features`: one variable-size nuclei tensor per patch, and `label`: 3, 4,
or 5. Learned NFI aggregation and MIL are optimized together; the cached upstream
encoders/student are frozen. A final `features`: [patches, 2048] bank can be used
when `training.mil.pretrained_nfi` supplies the matching trained NFI checkpoint.

Defaults: one bag per step, AdamW 2e-4, weight decay 0.05, three warm-up epochs
from 1e-5, 22 cosine-decay epochs to zero, clipping 1.0, EMA 0.999, and
validation macro-F1 patience 5.

For the paper's three grading runs, supply the three corresponding AGGC22
patient-stratified train/validation manifests in separate run configurations.
PANDA is never used to select a checkpoint or hyperparameter.

### BCR

Manifest: `paths.survival_manifest`. One cache per patient contains
`pooled`: [2048], `morphology`: [14], `time`: observed years, and `event`: 0/1.
Supply the actual morphology feature order used in the study.

Full-batch AdamW runs up to 150 epochs with 1e-3 initial LR, weight decay 0.01,
five-epoch warm-up, cosine decay to 1e-6, clipping 1.0, and EMA 0.999.
Each seed changes initialization on the same supplied patient split.
Test records are excluded from training and baseline estimation.

## Outputs

Each grading stage saves its best EMA checkpoint, selected only on validation
data. BCR checkpoints save the EMA head and training-fitted baseline for each seed.
A checkpoint includes architecture configuration, seed, epoch, state dictionary,
and training history. A run summary is written only after the requested stages
have completed successfully.
