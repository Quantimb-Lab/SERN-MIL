# Architecture

## SER

`models/ser/perceiver.py` adds fixed 2D positional encodings, lets learned latents
query all patch tokens, refines them through four transformer blocks, and decodes
them with the original positioned patch queries. Patch outputs remain distinct.

`models/ser/module.py` fuses each decoded context token with its 16 ? 16 shallow
feature map. Separate decoders perform the 16 ? 32 ? 256 spatial reconstruction.
`assemble_patch_logits` places each patch prediction in row-major order; it does
not resize a 256 ? 256 tile mask to 4096 ? 4096. Chunked decoder execution and
activation checkpointing control training memory.

The sigmoid threshold is 0.5. Retrieval retains patches with at least 30% binary
tumor occupancy. Boundary probability is not an independent tumor-retention
criterion. The boundary mask prevents graph connections that cross a predicted
rim. An empty tumor selection remains empty.

## NFI and MIL

The student architecture is 1056 ? 512 ? 256 ? 128, with LayerNorm, GELU, and 0.1
dropout. The 16 ? 16 presence map has a 32-D learned projection.

Equation 7 uses positive squared dot-product scores in both numerator and
normalizer. It is implemented with stable softmax over the K prototypes.
Equation 8 is a weighted sum over nuclei. A 128 ? 256 feature transform precedes
aggregation; the K cluster vectors are concatenated and mapped to 1024 dimensions.
Padding and absent nuclei contribute no features.

NFI and SER descriptors are concatenated to 2048 dimensions. Gated attention pools
each regional bag into a GP3/4/5 prediction. Training bags are defined by GP
annotations. Inference bags use only SER-predicted masks. Bag labels use annotated
pixel area, including where a patch intersects more than one pattern.

The sparse GP mask output stores global patch coordinates, class labels, and
patch size. Region areas are measured on this patch grid. Primary and secondary
patterns follow area fractions; equal areas use mean MIL confidence and then
the higher GP. The secondary fraction must exceed 0.05. The rule applies to both
specimen types in this implementation.

## BCR

A 2048-D attention-pooled slide representation is concatenated with the study's
14-D morphology vector. The three-linear-layer MLP returns log-risk. Training
uses complete risk sets and Efron's correction for tied event times.

The baseline cumulative hazard is estimated on training patients with the
Breslow estimator. Individual survival follows S(t|?)=exp(-H0(t) exp(?)).
The median is the first available time with survival at most 0.5. If it is not
reached in the estimated curve, it is returned as null; a median is not extrapolated.

The exact 14-D morphology vector is an input contract. Its supplied feature order
must be identical during training and inference.
