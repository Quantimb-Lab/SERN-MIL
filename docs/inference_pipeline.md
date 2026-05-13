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
