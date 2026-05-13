from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]
    config_path: Path
    root_dir: Path

    def get(self, key, default=None):
        node = self.raw
        for token in key.split("."):
            if not isinstance(node, dict) or token not in node:
                return default
            node = node[token]
        return node

    def path(self, key):
        value = self.get(key)
        if value is None:
            raise KeyError(f"Missing config path key: {key}")
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (self.root_dir / p).resolve()


def merge_dicts(base, override):
    out = deepcopy(base)
    for k, v in override.items():
        out[k] = (
            merge_dicts(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else deepcopy(v)
        )
    return out


def load_config(config_path):
    path = Path(config_path).resolve()

    def read(p, seen):
        if p in seen:
            raise ValueError(f"Cyclic configuration inheritance: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Configuration must be a mapping: {p}")
        parents = raw.pop("extends", [])
        if isinstance(parents, str):
            parents = [parents]
        result = {}
        for parent in parents:
            result = merge_dicts(result, read((p.parent / parent).resolve(), seen | {p}))
        return merge_dicts(result, raw)

    raw = read(path, set())
    config_dir = next((p for p in path.parents if p.name == "configs"), None)
    root = config_dir.parent if config_dir else path.parent
    return AppConfig(raw, path, root)


def build_model(cfg, stage):
    from sern_mil.models.grading import RegionGradingModel
    from sern_mil.models.nfi.student import NucleiStudent
    from sern_mil.models.ser.module import SERModule
    from sern_mil.models.survival.cox_head import CoxRiskHead

    factories = {
        "ser": SERModule,
        "student": NucleiStudent,
        "mil": RegionGradingModel,
        "survival": CoxRiskHead,
    }
    key = "grading" if stage == "mil" else stage
    return factories[stage](**cfg.get(f"model.{key}", {}))
