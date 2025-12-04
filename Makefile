PY ?= python
CFG ?= configs/default.yaml
TRAIN_CFG ?= configs/training_pipeline.yaml
INFER_CFG ?= configs/inference_pipeline.yaml

.PHONY: setup preprocess features fuse train infer lint test

setup:
	$(PY) -m pip install -e .

preprocess:
	$(PY) -m sern_mil.cli.main preprocess --config $(CFG)

features:
	$(PY) -m sern_mil.cli.main extract-clam --config $(CFG)
	$(PY) -m sern_mil.cli.main extract-cellvit --config $(CFG)

fuse:
	$(PY) -m sern_mil.cli.main fuse --config $(CFG)

train:
	$(PY) scripts/train_pipeline.py --config $(TRAIN_CFG)

infer:
	$(PY) scripts/infer_pipeline.py --config $(INFER_CFG)

lint:
	$(PY) -m ruff check src tests scripts

test:
	$(PY) -m pytest -q
