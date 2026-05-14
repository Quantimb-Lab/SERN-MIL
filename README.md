# SERN-MIL

Official repository for **SERN-MIL: Selective Embedding Retrieval and Nuclei-Feature Aggregation Based MIL for Prostate Gleason Grading**.

SERN-MIL combines tissue context and nuclei features to identify Gleason patterns,
estimate their proportions, and assign slide-level Gleason scores and ISUP grades.
An optional survival head predicts biochemical recurrence (BCR).

[Architecture](docs/architecture.md) | [Training guide](docs/training_pipeline.md) | [Inference guide](docs/inference_pipeline.md)

## Overview

```text
Whole-slide image
  -> Patch encoding and selective embedding retrieval (SER)
  -> Nuclei feature integration (NFI)
  -> Regional attention MIL
  -> GP masks, Gleason score, and ISUP grade
```

| Component | Default configuration |
| --- | --- |
| Image input | 4096 x 4096 tiles; 256 x 256 patches; 0.50 microns per pixel |
| SER | 1024-D tokens, 64 latent vectors, four transformer blocks, two mask decoders |
| Region selection | At least 30% tumor occupancy; boundary-separated regions of four or more patches |
| NFI | 128-D nuclei features, four learned clusters, 1024-D patch descriptor |
| Regional MIL | 2048-D combined descriptors; GP3, GP4, and GP5 classification |
| Slide grading | GP area proportions with specimen-specific minor-pattern rules |
| Optional BCR head | 2048-D pooled descriptor plus a supplied 14-D morphology vector |

NFI uses positive squared dot-product scores for soft assignment:

```python
weights = softmax(alpha * (features @ prototypes.T)**2, dim=-1)
```

Cluster-wise weighted sums are concatenated and projected to the nuclei descriptor.
A benign slide is assigned ISUP 0 and has no Gleason score.

Biopsy scores always include the highest observed GP, even below 5%. Minor
lower-grade patterns are included at 5% or more. Prostatectomy scores include an
additional higher-grade pattern only when it reaches 5%. For example, 60% GP4,
37% GP3, and 3% GP5 gives `4+5` for a biopsy and `4+3` for a prostatectomy.

## Installation

Use Python 3.10 or later and PyTorch 2.2 or later. For GPU training, install the
PyTorch build that matches your CUDA runtime first.

```bash
pip install -e ".[dev]"
```

To read whole-slide image formats, also install the pathology dependencies:

```bash
pip install -e ".[pathology]"
```

Configure pretrained feature extractors and trained checkpoints using the files
in `configs/`. The [training guide](docs/training_pipeline.md) describes the
required tensor caches and checkpoint formats.

## Data preparation

Create `data_index/slides.jsonl` with one JSON object per slide, on one line:

```json
{"slide_id":"case001","slide_path":"data/wsi/case001.svs","patient_id":"p001","dataset":"AGGC22","split":"train"}
```

| Field | Description |
| --- | --- |
| `slide_id` | Unique slide identifier matching the image filename stem |
| `slide_path` | Path to the whole-slide image |
| `patient_id` | Patient identifier used to keep splits separate |
| `dataset` | Dataset name, such as `AGGC22` or `PANDA` |
| `specimen_type` | `biopsy` or `prostatectomy`; needed for mixed cohorts such as AGGC22 |
| `split` | `train`, `val`, or `test` |
| `label_path` | GP label-mask path, required for annotated training bags |
| `isup` | Reference slide-level grade, required for evaluation |

Keep all slides from a patient in the same split. PANDA is reserved for external
testing. Paths are relative to the repository root unless absolute. GP mask values
are configurable; the defaults are 3, 4, and 5.

### Feature extractors

Both TorchScript wrappers accept RGB float tensors shaped `[1, 3, 256, 256]`,
with values from 0 to 1. Include the encoder's required normalization inside its
wrapper and export it in evaluation mode.

| Configuration key | Required output dictionary |
| --- | --- |
| `clam.checkpoint` | `tokens`: `[1, 1024]`; `shallow_features`: `[1, 256, 16, 16]` |
| `cellvit.checkpoint` | `embeddings`: `[nuclei, 1024]`; `neighborhoods`: `[nuclei, 16, 16]` |

The CellViT wrapper supplies one binary local presence map per nucleus. Train the
NFI student from paired embeddings and teacher features before extracting nuclei
descriptors, or configure an existing trained student checkpoint.

```bash
sern-mil train --config configs/student_training.yaml
```

### Build tile caches

```bash
sern-mil preprocess --config configs/default.yaml
sern-mil extract-clam --config configs/default.yaml
sern-mil extract-cellvit --config configs/default.yaml
sern-mil fuse --config configs/default.yaml
```

These commands save feature tensors and coordinate-aligned tile caches. The `fuse`
step joins the input feature banks; final descriptor concatenation takes place
after SER context decoding and learned NFI aggregation.

Training preprocessing applies orientation-relative rotations, extra rotation
coverage for high-grade slides, paired image-mask transformations, and photometric
augmentation. Validation and test slides receive no augmentation. Unaugmented
training copies supply the annotation-defined MIL bags.

## Training

Train SER, prepare annotated regional bags, and train NFI aggregation with MIL:

```bash
sern-mil train --config configs/training_pipeline.yaml
```

Grading checkpoints are selected using validation data. Stage-specific learning
rates, schedules, losses, and stopping criteria are defined in `configs/default.yaml`.

To train the optional BCR head with five initialization seeds on a supplied patient
split:

```bash
sern-mil train --config configs/survival_training.yaml
```

BCR training requires the pooled MIL descriptor, the matching 14-D morphology
vector, observed time, and event indicator for each patient.

## Inference

Prepare the slide's feature caches and configure the trained SER and MIL
checkpoints, then predict a single slide:

```bash
sern-mil predict --config configs/inference_pipeline.yaml --slide data/wsi/case001.svs
```

Or predict every supported slide in a folder:

```bash
sern-mil predict --config configs/inference_pipeline.yaml --wsis-dir data/wsi
```

Inference forms regional bags from the predicted tumor and boundary masks.
Reference GP masks are not used.

Inference uses the slide's `specimen_type` metadata. Without that field, PANDA
defaults to biopsy and LEOPARD to prostatectomy. For a mixed cohort, provide the
field in the slide manifest or set `inference.specimen_type` for a uniform batch.
Existing tile caches can use the field directly in `paths.ser_manifest`.
Each prediction records the resolved specimen type.

| Output | Location |
| --- | --- |
| Training summary | `artifacts/training_run_summary.json` |
| Trained checkpoints | `checkpoints/` |
| Slide predictions | `artifacts/inference_predictions.json` |
| Sparse GP masks | `artifacts/gp_masks/` |

## Evaluation

Add the reference `isup` labels to the slide manifest, then run:

```bash
sern-mil evaluate --config configs/inference_pipeline.yaml --predictions artifacts/inference_predictions.json
```

Evaluation reports both unweighted and quadratic-weighted Cohen's kappa. The
default uses quadratic weighting over ISUP classes 0 to 5. Macro precision, recall,
F1, AUROC, and AUPRC average grades 1 to 5. Class sets and weighting are explicit
in the configuration and saved evaluation results.

The Python metric API also accepts six-class probability arrays for AUROC, AUPRC,
15-bin expected calibration error, and Brier score. Bootstrap evaluation uses the
same slide resample for all trained models before averaging their metrics.

## Development checks

```bash
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

The suite checks model formulas and gradients, patch reconstruction, region
separation, grading rules, tied Cox risk sets, optimizer updates, checkpoint
loading, feature extraction, patient separation, and predicted-mask inference.
