from __future__ import annotations

from dataclasses import dataclass

from sern_mil.config import AppConfig
from sern_mil.utils.logging import get_logger


@dataclass
class Trainer:
    cfg: AppConfig

    def fit(self) -> None:
        logger = get_logger("trainer", self.cfg.path("paths.logs_dir"))
        epochs = int(self.cfg.get("training.epochs", 1))
        logger.info("Starting training for %d epochs", epochs)
        for epoch in range(1, epochs + 1):
            logger.info("Epoch %d/%d | placeholder loop", epoch, epochs)
