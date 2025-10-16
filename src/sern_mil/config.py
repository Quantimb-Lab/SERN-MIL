from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]
    config_path: Path
    root_dir: Path

    def get(self, key: str, default: Any = None) -> Any:
        node: Any = self.raw
        for token in key.split("."):
            if not isinstance(node, dict) or token not in node:
                return default
            node = node[token]
        return node

    def path(self, key: str) -> Path:
        value = self.get(key)
        if value is None:
            raise KeyError(f"Missing config path key: {key}")
        p = Path(value)
        if p.is_absolute():
            return p
        return (self.root_dir / p).resolve()


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).resolve()
    root_dir = path.parent.parent if path.parent.name == "configs" else path.parent
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(raw=raw, config_path=path, root_dir=root_dir)
