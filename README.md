# SERN-MIL

Official repository for **SERN-MIL: Selective Embedding Retrieval and Nuclei-Feature
Aggregation Based MIL for Prostate Gleason Grading**.

The model combines a KimiaNet patch encoder, a Perceiver-style selective embedding
retrieval (SER) module, nuclei feature integration (NFI), and regional attention MIL.
GP region areas determine the slide-level Gleason score and ISUP grade. An optional
Cox head predicts biochemical recurrence from pooled MIL and morphological features.

## Method

- 4096 ? 4096 tiles at 0.50 ?m/px; 16 ? 16 patches of 256 ? 256 pixels.
- 1024-D patch tokens plus fixed 2D positional encodings.
- 64 learnable 1024-D latents, input cross-attention, four latent transformer blocks,
  and output cross-attention back to the patch lattice.
- Independent tumor and boundary decoders reconstruct 256 ? 256 patch logits and
  place them at their original coordinates in the 4096 ? 4096 tile.
- Tumor-mask occupancy of at least 0.30 retains a patch. Eight-neighbor components
  are split at boundary crossings, including across tile edges. Regions need at
  least four retained patches.
- The NFI student maps a 1024-D CellViT token and a projected 16 ? 16 neighborhood
  to 128-D features. Four learned prototypes use
  **softmax(+?(f?c)?)**. Assigned transformed features are summed, concatenated, and
  mapped to a 1024-D patch descriptor.
- Concatenating context and nuclei descriptors produces 2048-D MIL instances.
  Regional predictions are GP3, GP4, or GP5. Slide grading uses the stated 5%
  secondary-pattern cutoff; a benign slide has ISUP 0 and no Gleason score.
- The three-layer BCR head takes a 2048-D pooled vector and a supplied 14-D
  morphology vector. Its full-batch loss uses exact Cox risk sets and Efron ties.

See [architecture](docs/architecture.md), [training](docs/training_pipeline.md), and
[inference](docs/inference_pipeline.md) for input contracts and configuration.

## Install

Python 3.10+ and PyTorch 2.2+ are supported. Install the PyTorch build appropriate
for your CUDA runtime before installing this package if training on a GPU.

~~~bash
pip install -e ".[dev]"
# For reading whole-slide formats:
pip install -e ".[pathology]"
~~~

Pretrained KimiaNet/CellViT adapters, trained study checkpoints, and patient data are
separate inputs. The feature commands load real trained adapters; they do not
generate substitute features. The adapter interface is documented below.

## Prepare inputs

Create `data_index/slides.jsonl`, one JSON object per slide:

~~~json
{"slide_id":"case001","slide_path":"data/wsi/case001.svs","patient_id":"patient001","dataset":"AGGC22","split":"train","label_path":"data/labels/case001.tif"}
~~~

Use `train`, `val`, or `test`. Keep every patient's slides in one split.
PANDA is external testing only. `label_path` is needed for annotated training-bag
preparation; inference does not read label masks. Include `isup` for evaluation.
Paths in configuration and manifests are relative to the repository root unless
absolute. GP label values are configurable; the defaults are 3/4/5.

Provide these feature extractors, in eval mode with the required preprocessing
inside the exported wrapper:

| Config key | TorchScript input | Returned dictionary |
|---|---|---|
| `clam.checkpoint` | RGB float [1, 3, 256, 256], values 0?1 | `tokens`: [1, 1024], `shallow_features`: [1, 256, 16, 16] |
| `cellvit.checkpoint` | Same RGB input | `embeddings`: [nuclei, 1024], `neighborhoods`: [nuclei, 16, 16] |

The CellViT wrapper supplies one binary local presence map per nucleus. The NFI
student checkpoint is trained from paired nuclei embeddings and teacher features
as described in the training guide. An empty nuclei set has shape [0, 1024].

~~~bash
sern-mil preprocess --config configs/default.yaml
sern-mil extract-clam --config configs/default.yaml
sern-mil extract-cellvit --config configs/default.yaml
sern-mil fuse --config configs/default.yaml
~~~

These commands write actual tensor files and coordinate-aligned tile caches.
The `fuse` command joins the input feature banks; the final descriptor concatenation
takes place after SER context decoding and learned NFI aggregation.

Training preprocessing includes orientation-relative rotations, extra high-grade
rotation coverage, paired nearest-neighbor label transformations, and the
specified photometric/artifact jitter. Cached variants have explicit augmentation
IDs and deterministic seeds. Validation/test slides receive no augmentation.
Unaugmented training copies supply annotation-defined MIL bags.

## Train and infer

~~~bash
# Optional: train the NFI student from paired nuclei/teacher caches first.
sern-mil train --config configs/student_training.yaml

# Train SER, build annotated regional caches, and train NFI aggregation + MIL.
sern-mil train --config configs/training_pipeline.yaml

# Optional BCR head, five initialization seeds on one supplied patient split.
sern-mil train --config configs/survival_training.yaml

# Predict from the slide's encoded tile caches and trained checkpoints.
sern-mil predict --config configs/inference_pipeline.yaml --slide data/wsi/case001.svs
sern-mil predict --config configs/inference_pipeline.yaml --wsis-dir data/wsi

sern-mil evaluate --config configs/inference_pipeline.yaml --predictions artifacts/inference_predictions.json
~~~

Training uses validation-only checkpoint selection for grading. Evaluation records
the ? weighting, included classes, and sample count explicitly, and returns both
unweighted and quadratic-weighted ?. The default follows the manuscript's
quadratic-weighted description; class and weighting settings are configurable.
Macro precision/recall/F1, macro-AUROC, and macro-AUPRC average ISUP grades 1?5.

## Checks

~~~bash
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
~~~

The tests exercise formula normalization and gradients, full-resolution patch
placement, occupancy selection, boundary separation, grading cutoffs, exact tied
Cox risk sets, real optimizer updates, checkpoint reloads, feature extraction,
patient separation, and inference that ignores reference masks. They use small
controlled tensors and do not represent a rerun of the study cohorts.
