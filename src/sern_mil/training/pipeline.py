from __future__ import annotations

from dataclasses import dataclass
import json

from sern_mil.training.prepare import prepare_mil_caches
from sern_mil.training.trainer import Trainer
from sern_mil.utils.io import ensure_dir


@dataclass
class TrainingPipeline:
    cfg: object

    def run(self):
        stages = self.cfg.get("training.stages", ["ser", "mil"])
        allowed = {"ser", "student", "mil", "survival"}
        if not stages or any(s not in allowed for s in stages):
            raise ValueError(f"training.stages must select from {sorted(allowed)}")
        results = []
        for stage in stages:
            if stage == "mil" and not self.cfg.path("paths.mil_manifest").exists():
                if not self.cfg.get("training.prepare_mil_from_annotations", True):
                    raise FileNotFoundError("MIL bag manifest is missing")
                prepare_mil_caches(self.cfg)
            if stage == "survival":
                for seed in self.cfg.get("training.survival.seeds", [0, 1, 2, 3, 4]):
                    results.append(Trainer(self.cfg, stage).fit(seed=seed))
            else:
                results.append(Trainer(self.cfg, stage).fit())
        out = self.cfg.path("paths.artifacts_dir") / "training_run_summary.json"
        ensure_dir(out.parent)
        out.write_text(
            json.dumps(
                {"experiment": self.cfg.get("project.experiment"), "stages": results},
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        return out
