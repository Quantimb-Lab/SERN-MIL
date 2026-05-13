import json
from pathlib import Path

from sern_mil.data.cache import load_manifest
from sern_mil.training.metrics import isup_metrics


def evaluate_predictions(cfg, predictions_path):
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    labels = {r["slide_id"]: r["isup"] for r in load_manifest(cfg, "paths.slide_manifest")}
    if not predictions or len({p["slide_id"] for p in predictions}) != len(predictions):
        raise ValueError("Predictions must contain unique nonempty slide records")
    if any(p["slide_id"] not in labels for p in predictions):
        raise ValueError("Reference ISUP labels are missing for some predicted slides")
    truth = [labels[p["slide_id"]] for p in predictions]
    pred = [p["predicted_isup"] for p in predictions]
    metrics = isup_metrics(truth, pred, **cfg.get("evaluation", {}))
    output = cfg.path("paths.artifacts_dir") / "evaluation_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return output
