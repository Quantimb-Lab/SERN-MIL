from pathlib import Path

from sern_mil.config import load_config
from sern_mil.inference.pipeline import InferencePipeline
from sern_mil.training.pipeline import TrainingPipeline


def test_training_pipeline_runs() -> None:
    cfg = load_config(Path("configs/training_pipeline.yaml"))
    out = TrainingPipeline(cfg).run()
    assert out.exists()
    assert out.name == "training_run_summary.json"


def test_inference_pipeline_runs_batch() -> None:
    cfg = load_config(Path("configs/inference_pipeline.yaml"))
    out = InferencePipeline(cfg).run()
    assert out.exists()
    assert out.name == "inference_predictions.json"
