from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sern_mil.config import AppConfig
from sern_mil.inference.predict_wsi import predict_slide
from sern_mil.utils.io import ensure_dir
from sern_mil.utils.logging import get_logger


@dataclass
class InferencePipeline:
    cfg: AppConfig

    def __post_init__(self) -> None:
        self.logger = get_logger("inference_pipeline", self.cfg.path("paths.logs_dir"))

    def run_single(self, slide_path: Path) -> dict[str, Any]:
        if not slide_path.exists():
            raise FileNotFoundError(f"Slide not found: {slide_path}")
        pred = predict_slide(self.cfg, slide_path)
        self.logger.info("Generated prediction for %s", slide_path)
        return pred

    def run_batch(self, wsis_dir: Path) -> list[dict[str, Any]]:
        if not wsis_dir.exists():
            self.logger.warning("WSI directory does not exist: %s", wsis_dir)
            return []
        preds: list[dict[str, Any]] = []
        for ext in ("*.svs", "*.tif", "*.tiff", "*.ndpi", "*.mrxs"):
            for slide in sorted(wsis_dir.rglob(ext)):
                preds.append(self.run_single(slide))
        self.logger.info("Batch inference complete for %d slides", len(preds))
        return preds

    def save_predictions(self, predictions: list[dict[str, Any]], output_path: Path) -> Path:
        ensure_dir(output_path.parent)
        output_path.write_text(
            __import__("json").dumps(predictions, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        self.logger.info("Saved predictions: %s", output_path)
        return output_path

    def run(self, slide: Path | None = None, wsis_dir: Path | None = None) -> Path:
        if slide is None and wsis_dir is None:
            wsis_dir = self.cfg.path("paths.wsis_dir")
        output_path = self.cfg.path("paths.artifacts_dir") / "inference_predictions.json"

        if slide is not None:
            preds = [self.run_single(slide)]
        else:
            assert wsis_dir is not None
            preds = self.run_batch(wsis_dir)
        return self.save_predictions(preds, output_path)
