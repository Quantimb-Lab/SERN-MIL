from pathlib import Path

from sern_mil.config import load_config


def test_load_default_config() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg.get("project.name") == "sern_mil"
    assert cfg.path("paths.preproc_manifest").name == "preprocess_manifest.jsonl"
