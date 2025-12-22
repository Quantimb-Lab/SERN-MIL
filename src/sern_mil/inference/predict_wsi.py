from __future__ import annotations

from pathlib import Path

from sern_mil.config import AppConfig
from sern_mil.utils.logging import get_logger


def predict_slide(cfg: AppConfig, slide_path: Path) -> dict:
    logger = get_logger("predict_wsi", cfg.path("paths.logs_dir"))
    logger.info("Running placeholder inference for %s", slide_path)
    return {
        "slide_id": slide_path.stem,
        "predicted_isup": 0,
        "gleason_score": "3+3",
        "gp_proportions": {"gp3": 0.0, "gp4": 0.0, "gp5": 0.0},
    }
