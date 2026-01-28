from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sern_mil.config import AppConfig
from sern_mil.training.trainer import Trainer
from sern_mil.utils.io import ensure_dir, read_jsonl
from sern_mil.utils.logging import get_logger


@dataclass
class TrainingPipeline:
    cfg: AppConfig

    def __post_init__(self) -> None:
        self.logger = get_logger("training_pipeline", self.cfg.path("paths.logs_dir"))

    def _require_manifest(self, path_key: str) -> list[dict]:
        path = self.cfg.path(path_key)
        rows = read_jsonl(path)
        if not rows:
            self.logger.warning("Required manifest is empty or missing: %s", path)
        else:
            self.logger.info("Loaded %d rows from %s", len(rows), path)
        return rows

    def validate_inputs(self) -> dict[str, int]:
        fused_rows = self._require_manifest("paths.fused_manifest")
        counts = {"fused_rows": len(fused_rows)}
        self.logger.info("Input validation complete: %s", counts)
        return counts

    def prepare_training_dirs(self) -> Path:
        ckpt_dir = self.cfg.path("paths.checkpoints_dir")
        ensure_dir(ckpt_dir)
        self.logger.info("Checkpoint directory ready: %s", ckpt_dir)
        return ckpt_dir

    def run_stage_ser(self) -> dict[str, Any]:
        self.logger.info("Stage SER: training selective embedding retrieval module (placeholder)")
        return {"stage": "ser", "status": "ok"}

    def run_stage_mil(self) -> dict[str, Any]:
        self.logger.info("Stage MIL: training region-aware MIL head (placeholder)")
        Trainer(self.cfg).fit()
        return {"stage": "mil", "status": "ok"}

    def run_stage_survival(self) -> dict[str, Any]:
        enable_survival = bool(self.cfg.get("training.enable_survival_head", False))
        if not enable_survival:
            self.logger.info("Stage Survival: skipped (training.enable_survival_head=false)")
            return {"stage": "survival", "status": "skipped"}
        self.logger.info("Stage Survival: training BCR head (placeholder)")
        return {"stage": "survival", "status": "ok"}

    def save_run_summary(self, stage_results: list[dict[str, Any]]) -> Path:
        out_path = self.cfg.path("paths.artifacts_dir") / "training_run_summary.json"
        ensure_dir(out_path.parent)
        payload = {
            "experiment": self.cfg.get("project.experiment", "unknown"),
            "stages": stage_results,
        }
        out_path.write_text(__import__("json").dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        self.logger.info("Saved training summary: %s", out_path)
        return out_path

    def run(self) -> Path:
        self.validate_inputs()
        self.prepare_training_dirs()
        stages = [
            self.run_stage_ser(),
            self.run_stage_mil(),
            self.run_stage_survival(),
        ]
        return self.save_run_summary(stages)
