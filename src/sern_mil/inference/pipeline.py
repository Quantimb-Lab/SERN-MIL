from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from sern_mil.inference.predict_wsi import load_grading_models, predict_slide
from sern_mil.utils.io import ensure_dir


@dataclass
class InferencePipeline:
    cfg: object
    models: object = field(default=None, init=False)

    def run_single(self, slide_path):
        slide_path = Path(slide_path)
        if not slide_path.is_file():
            raise FileNotFoundError(f"Slide not found: {slide_path}")
        if self.models is None:
            self.models = load_grading_models(self.cfg)
        return predict_slide(self.cfg, slide_path, self.models)

    def run_batch(self, wsis_dir):
        wsis_dir = Path(wsis_dir)
        if not wsis_dir.is_dir():
            raise FileNotFoundError(f"WSI directory not found: {wsis_dir}")
        supported = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".png", ".jpg", ".jpeg"}
        slides = sorted(
            p for p in wsis_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported
        )
        if not slides:
            raise ValueError("No slides found for inference")
        return [self.run_single(slide) for slide in slides]

    def save_predictions(self, predictions, output_path):
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        output_path.write_text(json.dumps(predictions, indent=2, allow_nan=False), encoding="utf-8")
        return output_path

    def run(self, slide=None, wsis_dir=None):
        if slide is not None and wsis_dir is not None:
            raise ValueError("Choose one slide or a WSI folder")
        predictions = (
            [self.run_single(slide)]
            if slide is not None
            else self.run_batch(wsis_dir or self.cfg.path("paths.wsis_dir"))
        )
        return self.save_predictions(
            predictions, self.cfg.path("paths.artifacts_dir") / "inference_predictions.json"
        )
