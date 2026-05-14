# Inference

Encode the test slides with the same trained KimiaNet and CellViT/student feature
extractors used for training. Do not enable training augmentation on these slides.
Run preprocessing, both feature commands, and `fuse` to create aligned tile caches.

~~~bash
sern-mil predict --config configs/inference_pipeline.yaml --slide data/wsi/sample.svs
~~~

The inference pipeline requires trained SER and regional MIL checkpoints.
Architecture dimensions are restored from each checkpoint. It uses the predicted
tumor/boundary maps for retrieval and bag formation. Reference GP masks and target
masks present in a cache are ignored.

## Specimen-specific grading

Use `specimen_type: biopsy` or `specimen_type: prostatectomy` in each slide
manifest record. Feature preparation carries this field into the tile manifest.
For existing tensor caches, add it to the corresponding `paths.ser_manifest`
records without regenerating the feature tensors.

The default `inference.specimen_type: auto` resolves the type from metadata;
without it, PANDA resolves to biopsy and LEOPARD to prostatectomy. AGGC22 contains
both specimen types, so each slide needs metadata or an explicit configuration
for a uniform cohort. Conflicting metadata/configuration raises an error.
Every inference result records the resolved type.

Biopsy scoring retains the highest GP even below 5%, while minor lower-grade
secondary patterns require at least 5%. Prostatectomy scoring requires an
additional pattern to reach 5%, and uses the highest eligible GP when higher than
the primary. Thus, 60% GP4 + 37% GP3 + 3% GP5 gives `4+5` on biopsy and `4+3`
on prostatectomy. A lower-grade secondary pattern at exactly 5% is included.

The direct `grade_from_regions` API also requires the `specimen_type` keyword.

For each slide the output includes:

- ISUP grade, Gleason score, and GP area proportions.
- Regional GP predictions, areas, confidence scores, and patch counts.
- A compressed sparse GP mask with global coordinates and GP labels.
- The pooled descriptor when at least one retained region is present.

A benign/no-retained-region output is ISUP 0, null Gleason score, and zero GP
proportions. Missing caches/checkpoints raise errors rather than producing a grade.

For optional BCR output, configure `inference.survival_checkpoint` and supply one
patient record with the matching 14-D morphology vector in
`paths.survival_manifest`. Observed time and event are not model inputs at inference.
The saved training baseline converts predicted log-risk to the median time and
the predefined <3, 3?10, >10-year groups when a median is reached.

## Evaluation

Put the reference slide-level `isup` labels in `paths.slide_manifest` and run:

~~~bash
sern-mil evaluate --config configs/inference_pipeline.yaml --predictions artifacts/inference_predictions.json
~~~

The evaluation JSON includes accuracy, macro precision/recall/F1 over grades 1?5,
MCC, benign NPV, confusion matrix, ? weighting/classes, and the ? sample count.
It returns both ordinary and quadratic-weighted ?. By default ? uses classes 0?5
and quadratic weighting, as specified in the manuscript's Methods.

Changing `evaluation.kappa_classes` restricts the ? matrix to those classes;
excluded true/predicted classes are counted explicitly. This setting does not
change the macro-metric class set. Macro averaging retains benign examples when
counting cancer-class false positives and false negatives.

The Python metric API also accepts normalized six-class probability arrays for
AUROC, AUPRC, 15-bin ECE, and multiclass Brier score. The deterministic area-based
grading rule does not invent class probabilities. `bootstrap_metrics` resamples
the same slides for all trained models, averages their metrics per replicate,
and forms percentile intervals from the averaged estimates.
